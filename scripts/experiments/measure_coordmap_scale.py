#!/usr/bin/env python
"""逐帧量化 coord_map 3D 锚点系相对真实模型的等效尺度。

对每帧：从落盘 matches 重建 _solve_pnp 同款对应点 (pts2d, C=coord_map 3D)，
在 GT 位姿下拟合尺度 s：min Σ ||proj(R_gt, t_gt, C/s) - pts2d||²。
s > 1 → coord_map 比真实模型大（PnP 深度会偏浅）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import PoseEstimator, TemplateBank, template_bank_path
from src.config import load_config
from src.matching.correspondence import back_to_original_pixels
from src.datasets.linemod import LinemodDataset


def load_frame_matches(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pix_q = d["pix_q"]
    pix_t = d["pix_t"]
    t_idx = d["template_idx"]
    sims = d["sims"]
    seg = d["seg"]
    # 恢复与 _solve_pnp 相同的 match 结构（按模板分组）
    by_t = {}
    for i in range(len(t_idx)):
        ti = int(t_idx[i])
        by_t.setdefault(ti, []).append((pix_q[i], pix_t[i], float(sims[i])))
    return by_t, d


def main():
    cfg = load_config("configs/dense80.yaml")
    obj = "ape"
    bank = TemplateBank(template_bank_path(cfg, obj))
    ds = LinemodDataset(cfg["dataset"]["root"], obj,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = ds.eval_frames(exclude_refs=True,
                            n_ref=int(cfg["onboard"].get("n_ref_views", 64)))
    mdir = Path("outputs/matches_ape_full") / obj

    ss, dep = [], []
    n_frames = 0
    for fr in frames[::10]:          # 每 10 帧取 1，~117 帧
        npz = mdir / f"{fr.frame_id:06d}.npz"
        if not npz.exists():
            continue
        by_t, d = load_frame_matches(npz)
        if not by_t:
            continue
        crop_box = d["crop_box"]
        sxy = d["sxy"]
        s_leg = d["s_leg"]
        # 组装 top-12 联合对应（同 _solve_pnp）
        pix_q_all, C_all = [], []
        for ti, corr in by_t.items():
            if ti >= len(bank.coord_maps):
                continue
            pq = np.array([c[0] for c in corr])
            pt = np.array([c[1] for c in corr])
            pq = back_to_original_pixels(
                pq, (sxy[0] * s_leg[0], sxy[1] * s_leg[1]), crop_box)
            xt = pt[:, 0].astype(int); yt = pt[:, 1].astype(int)
            cm = bank.coord_maps[ti]
            h, w = cm.shape[:2]
            ok = (xt >= 0) & (xt < w) & (yt >= 0) & (yt < h)
            C = cm[yt[ok], xt[ok]]
            valid = np.abs(C).sum(axis=1) > 0
            pix_q_all.append(pq[ok][valid]); C_all.append(C[valid])
        if not C_all:
            continue
        q = np.concatenate(pix_q_all)
        C = np.concatenate(C_all)
        if len(q) < 20:
            continue
        # GT 投影拟合尺度 s：proj(C/s) ≈ q；用一维搜索（稳健）
        f = fr.K[0, 0]
        R = fr.R_gt; t = fr.t_gt
        Cc = (R @ C.T).T + t                      # 相机系
        def reproj_err(s):
            P = Cc / s
            px = P[:, :2] / P[:, 2:3] * f + fr.K[:2, 2]
            return np.median(np.linalg.norm(px - q, axis=1))
        best = min((reproj_err(s), s) for s in
                   np.linspace(0.8, 1.25, 46))
        ss.append(best[1])
        dep.append(float(np.median(Cc[:, 2])))
        n_frames += 1

    ss = np.array(ss)
    print(f"帧数: {n_frames}")
    print(f"coord_map 等效尺度 s: mean {ss.mean():.4f} med {np.median(ss):.4f} "
          f"p25-p75 {np.percentile(ss,25):.4f}-{np.percentile(ss,75):.4f}")
    print(f"s>1.02 的帧: {100*np.mean(ss>1.02):.1f}%  s<0.98: {100*np.mean(ss<0.98):.1f}%")
    print(f"预期深度比 1/s: med {1/np.median(ss):.4f} (实测 PnP 0.958)")


if __name__ == "__main__":
    main()
