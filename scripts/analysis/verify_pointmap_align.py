#!/usr/bin/env python3
"""6d-pointmap-t1 离线验证：MASt3R pointmap vs coord_map 的 3D 一致性。

对 ape/duck 的 120 帧子集：
1. 逐帧读已落盘 matches（npz：pix_t 模板像素、pts3d_q 查询相机系 3D）
2. X_obj = coord_map[pix_t]（物体系 3D）；P_gt = R_gt @ X_obj + t_gt
3. 残差 = |P_qcam - P_gt|：pointmap 深度/尺度准不准
4. Umeyama 相似变换拟合 T_q（3D-3D 对齐，无内参）→ ADD(S)@0.1d 上界

GPU-only 依赖：无（bank/GT 均 CPU 可读）。零管线改动。
"""
import json
import sys
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))

from src.config import load_config
from src.pipeline import TemplateBank, template_bank_path, subsample_frames
from src.datasets.linemod import LinemodDataset


def umeyama(src, dst):
    """Umeyama 相似变换（带尺度）：dst ≈ s·R·src + t。返回 (s, R, t)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (src_c ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def eval_add(R, t, R_gt, t_gt, pts, sym_Rs, diameter):
    pts_t = (R @ pts.T + t[:, None]).T
    pts_gt = (R_gt @ pts.T + t_gt[:, None]).T
    if sym_Rs:
        errs = []
        for Rsym in sym_Rs:
            p_gt_s = (Rsym @ pts_gt.T + t_gt[:, None]).T
            d = np.linalg.norm(pts_t - p_gt_s, axis=1).mean()
            errs.append(d)
        err = min(errs)
    else:
        err = np.linalg.norm(pts_t - pts_gt, axis=1).mean()
    return err < 0.1 * diameter, err


def main():
    cfg = load_config("configs/current/dense80_depthc_guided.yaml")
    out = {}
    for obj, matches_dir in [("ape", "outputs/matches13_30k"),
                             ("duck", "outputs/matches13_30k")]:
        bank = TemplateBank(template_bank_path(cfg, obj))
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = ds.eval_frames(exclude_refs=True,
                                n_ref=int(cfg["onboard"].get("n_ref_views", 64)))
        frames = subsample_frames(frames, 120)
        cms = bank.coord_maps                 # (M,S,S,3) 物体系 3D（锚点系）
        # 对称物体（eggbox/glue 才有）— ape/duck 非对称
        sym_Rs = []
        med_res, n_good, n_frames = [], 0, 0
        add_hits = []
        for fr in frames:
            npz = Path(matches_dir) / obj / f"{fr.frame_id:06d}.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            pix_t = d["pix_t"].astype(int)
            tpl = d["template_idx"].astype(int)
            seg = d["seg"].astype(int)
            p3q = d["pts3d_q"]
            if len(pix_t) == 0 or len(p3q) == 0:
                continue
            # seg = 每模板点数前缀和 → 逐点展开模板索引
            tpl_exp = np.repeat(tpl, np.diff(seg))
            X_obj = np.stack([cms[ti][py, px] for ti, (px, py)
                              in zip(tpl_exp, pix_t)])
            valid = np.abs(X_obj).sum(axis=1) > 0
            if valid.sum() < 6:
                continue
            X, P = X_obj[valid], p3q[valid]
            # 残差：GT 位姿投影的物体系点 vs MASt3R pointmap
            P_gt = (fr.R_gt @ X.T + fr.t_gt[:, None]).T
            res = np.linalg.norm(P - P_gt, axis=1)
            med_res.append(np.median(res))
            n_good += int((res < 5.0).mean() > 0.5)
            n_frames += 1
            # 3D-3D 对齐（Umeyama 相似变换吸收 pointmap 尺度）→ ADD。
            # P_cam ≈ s·R·X + t：位姿 (R, t) 直接是查询相机系；尺度 s
            # 属于 pointmap 缩放，评估时把模型点乘 s 再比 GT。
            s, R, t = umeyama(X, P)
            hit, _ = eval_add(R, t, fr.R_gt, fr.t_gt,
                              s * ds.model_points(2000), sym_Rs, ds.diameter)
            add_hits.append(hit)
        med_res = np.array(med_res)
        add = 100.0 * np.mean(add_hits) if add_hits else float("nan")
        out[obj] = {
            "n_frames": n_frames,
            "resid_med_mm": float(np.median(med_res)) if len(med_res) else None,
            "resid_p50/p75": [float(np.percentile(med_res, p))
                              if len(med_res) else None for p in (50, 75)],
            "good_frac": n_good / max(n_frames, 1),
            "umeyama_add_01d": round(add, 2),
        }
        print(json.dumps({obj: out[obj]}, ensure_ascii=False, indent=1))
    json.dump(out, open("outputs/exp_t1/offline.json", "w"), indent=1)
    print("done → outputs/exp_t1/offline.json")


if __name__ == "__main__":
    main()
