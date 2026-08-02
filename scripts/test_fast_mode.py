#!/usr/bin/env python3
"""GS-Pose 式快速位姿模式离线对比（复用 dense80 落盘产物，不重跑匹配）。

变体（每帧同 GT 同口径）：
  A_init  掩码解析平移 + 检索旋转（无 PnP、无精化）           ≈ 0.05s
  A_ref   A_init → SSIM-only 精化（无 LPIPS + 早停）           ≈ 2s
  B_coarse PnP（逐模板 RANSAC + joint + alt 渲染验证）         ≈ 3s
  B_ref   B_coarse → SSIM-only 精化                            ≈ 5s
  C       基线 = outputs/cache_dense80/<obj>.jsonl（PnP + LPIPS 精化全流程）

用法：
    python scripts/test_fast_mode.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.pipeline import (TemplateBank, load_extracted_matches,
                          subsample_frames, template_bank_path)
from src.matching.correspondence import back_to_original_pixels
from src.solver.ransac_pnp import ransac_pnp
from src.solver.selection import rank_candidates


class MiniSolver:
    """PoseEstimator._solve_pnp + _to_model_frame 的轻量复制（免加载 MASt3R）。

    只用于离线实验；与在线路径的唯一差异是不创建匹配器（不用它）。
    """

    def __init__(self, cfg, bank):
        self.cfg, self.bank = cfg, bank
        self._lifting = cfg["matching"].get("lifting", "coord_map")
        assert self._lifting == "coord_map", "实验仅支持 coord_map 提升"

    def _to_model_frame(self, R, t):
        t_model = t / self.bank.scale
        if self.bank.has_align:
            from src.geometry.alignment import transform_pose_by_similarity
            return transform_pose_by_similarity(
                R, t_model, self.bank.align_s, self.bank.align_R,
                self.bank.align_t)
        return R, t_model

    def solve_pnp(self, ex, K_query):
        s_cfg = self.cfg["solver"]
        matches = sorted(ex["matches"], key=lambda m: m.score, reverse=True)
        cap = int(s_cfg.get("ransac_top_templates", 0))
        if cap > 0:
            matches = matches[:cap]
        crop_box_used = ex["crop_box_used"]
        s_leg_x, s_leg_y = ex["s_leg"]
        sx, sy = ex["sxy"]
        results = []
        corr_list = []
        for m in matches:
            pts2d = back_to_original_pixels(
                m.pix_q, (sx * s_leg_x, sy * s_leg_y), crop_box_used)
            cm = self.bank.coord_maps[m.template_idx]
            xt = m.pix_t[:, 0].astype(int)
            yt = m.pix_t[:, 1].astype(int)
            pts3d = cm[yt, xt]
            valid = np.abs(pts3d).sum(axis=1) > 0
            corr_list.append((pts2d[valid], pts3d[valid]))
            r = ransac_pnp(
                pts2d[valid], pts3d[valid], K_query,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=bool(s_cfg.get("refine_lm", True)),
                min_correspondences=int(s_cfg.get("min_correspondences", 6)),
                flag=str(s_cfg.get("pnp_flag", "epnp")))
            r.template_idx = m.template_idx
            r.template_score = m.score
            results.append(r)
        ranked = rank_candidates(results,
                                 strategy=s_cfg.get("selection", "inlier"))
        if not ranked:
            return None
        best = ranked[0]
        joint_k = int(s_cfg.get("joint_templates", 3))
        if joint_k >= 2 and len(corr_list) >= joint_k:
            j2 = np.concatenate([c[0] for c in corr_list[:joint_k]])
            j3 = np.concatenate([c[1] for c in corr_list[:joint_k]])
            r_j = ransac_pnp(
                j2, j3, K_query,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=bool(s_cfg.get("refine_lm", True)),
                min_correspondences=int(s_cfg.get("min_correspondences", 6)),
                flag=str(s_cfg.get("pnp_flag", "epnp")))
            if r_j.success and r_j.n_inliers >= best.n_inliers:
                r_j.template_idx = best.template_idx
                r_j.template_score = best.template_score
                best = r_j
        return best, results


def solve_with_alt(ex, K_query, solver, verifier):
    """PnP + 备选候选渲染验证消歧（等价 _solve 的粗位姿段）。"""
    solved = solver.solve_pnp(ex, K_query)
    if solved is None:
        return None
    best, _ = solved
    chosen = best
    if ex.get("alts") and verifier is not None:
        cands = [(ex, best)]
        for a in ex["alts"]:
            a_ex = {"crop": a["crop"], "mask_crop": a["mask_crop"],
                    "crop_box_used": a["crop_box_used"],
                    "s_leg": a["s_leg"], "sxy": a["sxy"],
                    "matches": a["matches"]}
            s = solver.solve_pnp(a_ex, K_query)
            if s is not None:
                cands.append((a_ex, s[0]))
        best_loss = float("inf")
        for cex, r in cands:
            x0, y0, _, _ = cex["crop_box_used"]
            K_crop = K_query.copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            loss = verifier.align_loss(cex["crop"], cex["mask_crop"],
                                       K_crop, r.R, r.t)
            if loss < best_loss:
                best_loss, chosen = loss, r
    return chosen


def eval_pose(ds, mcfg, model_pts, fr, R, t):
    from src.metrics.pose_metrics import evaluate_pose
    if R is None:
        return {"add_01d": 0.0, "proj_5px": 0.0, "cm_deg": 0.0}
    m = evaluate_pose(
        model_pts, ds.diameter, fr.K, fr.R_gt, fr.t_gt, R, t,
        symmetric=ds.symmetric,
        add_threshold_ratio=float(mcfg["add_threshold_ratio"]),
        proj_threshold_px=float(mcfg["proj_threshold_px"]),
        cm_threshold_mm=float(mcfg["cm_threshold_mm"]),
        deg_threshold=float(mcfg["deg_threshold"]),
        adds_definition=mcfg.get("adds_definition", "unidirectional"))
    return m


def refine_pose(refiner, ex, K_query, R, t):
    """裁剪坐标系 SSIM-only 精化，返回模型系 (R, t)。"""
    if R is None:
        return None, None
    x0, y0, _, _ = ex["crop_box_used"]
    K_crop = K_query.copy()
    K_crop[0, 2] -= x0
    K_crop[1, 2] -= y0
    R_r, t_r = refiner.refine(ex["crop"], ex["mask_crop"], K_crop, R, t)
    return R_r, t_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense80.yaml")
    ap.add_argument("--matches-dir", default="outputs/matches_dense80")
    ap.add_argument("--cache-dir", default="outputs/cache_dense80")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device
    mcfg = cfg["metrics"]
    objects = args.objects or cfg["dataset"]["objects"]

    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner

    # SSIM-only 精化器：无 LPIPS + 收敛早停（GS-Pose 式）
    refiner_kw = dict(lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.0,
                      lambda_dice=0.3, iterations=150, lr=0.02,
                      early_stop_patience=20, early_stop_tol=1e-4)
    # 对照：SSIM-only 跑满 150 迭代（无早停）——隔离"早停过激"与
    # "LPIPS 信号必要"两个因素
    refiner_full_kw = dict(lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.0,
                           lambda_dice=0.3, iterations=150, lr=0.02,
                           early_stop_patience=0)

    keys = ("add_01d", "proj_5px", "cm_deg")
    agg = {v: {k: [] for k in keys} for v in
           ("A_init", "A_ref", "B_coarse", "B_ref", "B_ref_full", "C")}
    time_agg = {v: 0.0 for v in
                ("A_init", "A_ref", "B_coarse", "B_ref", "B_ref_full")}

    print(f"{'物体':<12}{'A_init':>18}{'A_ref':>18}{'B_coarse':>18}"
          f"{'B_ref':>18}{'B_full':>18}{'C基线':>18}")
    print(f"{'':<12}{'(ADD/Proj/5cm)':>18}{'(ADD/Proj/5cm)':>18}"
          f"{'(ADD/Proj/5cm)':>18}{'(ADD/Proj/5cm)':>18}"
          f"{'(ADD/Proj/5cm)':>18}{'(ADD/Proj/5cm)':>18}")

    def fmt(v, a):
        return ("{:>5.1f}/{:<5.1f}/{:<5.1f}"
                .format(100 * np.mean(a[v]["add_01d"]),
                        100 * np.mean(a[v]["proj_5px"]),
                        100 * np.mean(a[v]["cm_deg"])))

    for obj in objects:
        bank_path = template_bank_path(cfg, obj)
        bank = TemplateBank(bank_path)
        solver = MiniSolver(cfg, bank)
        ckpt = str(bank_path.with_suffix(".pt"))
        refiner = PoseRefiner(ckpt, device=device, **refiner_kw)
        refiner_full = PoseRefiner(ckpt, device=device, **refiner_full_kw)
        verifier = PoseRefiner(ckpt, device=device, iterations=0)

        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = ds.eval_frames(
            exclude_refs=True,
            n_ref=int(cfg["onboard"].get("n_ref_views", 64)))
        frames = subsample_frames(frames, args.max_frames)
        model_pts = ds.model_points(max_points=2000)
        local = {v: {k: [] for k in keys} for v in
                 ("A_init", "A_ref", "B_coarse", "B_ref", "B_ref_full", "C")}

        # 基线 C：评估缓存（PnP + LPIPS 精化全流程）
        cache = {}
        cp = Path(args.cache_dir) / f"{obj}.jsonl"
        if cp.exists():
            for line in cp.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[int(rec["frame_id"])] = rec["m"]

        for fr in frames:
            npz = Path(args.matches_dir) / obj / f"{fr.frame_id:06d}.npz"
            if not npz.exists():
                for v in ("A_init", "A_ref", "B_coarse", "B_ref"):
                    for k in keys:
                        local[v][k].append(0.0)
                continue
            ex = load_extracted_matches(npz)
            Kq = fr.K

            # A：掩码解析平移 + 检索旋转
            t0 = time.time()
            Ra, ta = mask_ratio_init(solver, ex, Kq)
            time_agg["A_init"] += time.time() - t0
            Rm, tm = solver._to_model_frame(Ra, ta) if Ra is not None \
                else (None, None)
            m = eval_pose(ds, mcfg, model_pts, fr, Rm, tm)
            for k in keys:
                local["A_init"][k].append(m[k])

            # A_ref：A → SSIM-only 精化
            t0 = time.time()
            Ra_r, ta_r = refine_pose(refiner, ex, Kq, Ra, ta)
            time_agg["A_ref"] += time.time() - t0
            Rm, tm = (solver._to_model_frame(Ra_r, ta_r)
                      if Ra_r is not None else (None, None))
            m = eval_pose(ds, mcfg, model_pts, fr, Rm, tm)
            for k in keys:
                local["A_ref"][k].append(m[k])

            # B_coarse：PnP + alt 渲染验证
            t0 = time.time()
            chosen = solve_with_alt(ex, Kq, solver, verifier)
            time_agg["B_coarse"] += time.time() - t0
            Rm, tm = (solver._to_model_frame(chosen.R, chosen.t)
                      if chosen is not None else (None, None))
            m = eval_pose(ds, mcfg, model_pts, fr, Rm, tm)
            for k in keys:
                local["B_coarse"][k].append(m[k])

            # B_ref：B_coarse → SSIM-only 精化
            t0 = time.time()
            Rb_r, tb_r = refine_pose(
                refiner, ex, Kq,
                chosen.R if chosen is not None else None,
                chosen.t if chosen is not None else None)
            time_agg["B_ref"] += time.time() - t0
            Rm, tm = (solver._to_model_frame(Rb_r, tb_r)
                      if Rb_r is not None else (None, None))
            m = eval_pose(ds, mcfg, model_pts, fr, Rm, tm)
            for k in keys:
                local["B_ref"][k].append(m[k])

            # C 基线（来自评估缓存；缺帧按失败计）
            # B_ref_full：B_coarse → SSIM-only 跑满 150 迭代（无早停）
            t0 = time.time()
            Rbf, tbf = refine_pose(
                refiner_full, ex, Kq,
                chosen.R if chosen is not None else None,
                chosen.t if chosen is not None else None)
            time_agg["B_ref_full"] += time.time() - t0
            Rm, tm = (solver._to_model_frame(Rbf, tbf)
                      if Rbf is not None else (None, None))
            m = eval_pose(ds, mcfg, model_pts, fr, Rm, tm)
            for k in keys:
                local["B_ref_full"][k].append(m[k])

            c = cache.get(fr.frame_id)
            for k in keys:
                local["C"][k].append(c[k] if c is not None else 0.0)

        print(f"{obj:<12}{fmt('A_init', local):>18}{fmt('A_ref', local):>18}"
              f"{fmt('B_coarse', local):>18}{fmt('B_ref', local):>18}"
              f"{fmt('B_ref_full', local):>18}{fmt('C', local):>18}")
        for v in ("A_init", "A_ref", "B_coarse", "B_ref", "B_ref_full", "C"):
            for k in keys:
                agg[v][k].extend(local[v][k])

    print(f"\n{'MEAN':<12}{fmt('A_init', agg):>18}{fmt('A_ref', agg):>18}"
          f"{fmt('B_coarse', agg):>18}{fmt('B_ref', agg):>18}"
          f"{fmt('B_ref_full', agg):>18}{fmt('C', agg):>18}")
    print("\n平均每帧耗时: "
          + "  ".join(f"{v}={t/ (len(agg['A_init']['add_01d']) or 1):.2f}s"
                      for v, t in time_agg.items()))


def mask_ratio_init(solver, ex, K_query):
    """GS-Pose 式掩码解析平移（pipeline.PoseEstimator.mask_ratio_init 副本）。"""
    if not ex.get("matches"):
        return None, None
    idx = int(ex["matches"][0].template_idx)
    mask = ex.get("mask_crop")
    if mask is None or mask.sum() < 16:
        return None, None
    a_t = solver.bank.alphas[idx] > 0.5
    if a_t.sum() < 16:
        return None, None
    x0, y0, _, _ = ex["crop_box_used"]
    ys, xs = np.nonzero(mask)
    cx_q = x0 + float(xs.min() + xs.max()) / 2.0
    cy_q = y0 + float(ys.min() + ys.max()) / 2.0
    ratio = np.sqrt(float(mask.sum()) / float(a_t.sum()))
    T = solver.bank.poses[idx]
    tz_ref = float(T[2, 3])
    tz_q = tz_ref * (float(K_query[0, 0]) / float(solver.bank.K[0, 0])
                     ) / max(ratio, 1e-3)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = tz_q * (np.linalg.inv(K_query) @ np.array([cx_q, cy_q, 1.0],
                                                  dtype=np.float64))
    return R, t


if __name__ == "__main__":
    main()
