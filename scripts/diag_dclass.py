#!/usr/bin/env python3
"""D 类深度病态诊断（CPU）：
对每个匹配锚点：
  P_obj = coord_map[template][pix_t]（物体系, 已对齐尺度）
  P     = R_tpl @ P_obj + t_tpl            # 模板相机系 = 物体系
  z_pred= (R_gt @ P + t_gt)[2]             # GT 相机系深度
  z_gt  = GT 表面在查询像素 pix_q 的深度（模型顶点最近邻）
偏差 = (z_pred - z_gt) / z_gt；>0 = 锚点比真实表面远。
按锚点在 GT 掩码内的"到轮廓距离"分层（边缘 0-3px / 内部 >3px），
验证边界高斯糊假设。逐帧 + 聚合输出。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.datasets.linemod import LinemodDataset
from src.datasets.ply_io import load_ply
from src.pipeline import TemplateBank, template_bank_path
from src.config import load_config


def main():
    obj = sys.argv[1] if len(sys.argv) > 1 else "duck"
    matches_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/matches13_ds"
    cfg = load_config("configs/dense80_depth_bg0.yaml")
    bank = TemplateBank(template_bank_path(cfg, obj))
    ds = LinemodDataset(cfg["dataset"]["root"], obj,
                        models_dir=cfg["dataset"].get("models_dir", "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = ds.eval_frames(exclude_refs=True, n_ref=0)[:120]
    verts, faces, _ = load_ply(ds.model_path)
    s = bank.scale
    # 模板位姿: w2c (object→template cam)；GT: w2c (object→query cam)
    R_t = bank.poses[:, :3, :3]
    t_t = bank.poses[:, :3, 3]

    agg = {"edge": [], "inner": []}
    per_frame = []
    for fr in frames:
        npz = Path(matches_dir) / obj / f"{fr.frame_id:06d}.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        if d["pix_q"].shape[0] == 0:
            continue
        seg = d["seg"].astype(int)
        R_gt, t_gt = fr.R_gt, fr.t_gt
        # pix_q 是裁剪坐标 → 全图坐标（与 _solve 的 back_to_original_pixels 同）
        x0, y0, x1, y1 = (int(v) for v in d["crop_box"])
        sx, sy = d["sxy"]
        slx, sly = d["s_leg"]
        pq_full = np.empty_like(d["pix_q"], dtype=float)
        pq_full[:, 0] = x0 + d["pix_q"][:, 0] / (sx * slx)
        pq_full[:, 1] = y0 + d["pix_q"][:, 1] / (sy * sly)
        # GT 掩码（查询图）→ 边缘距离图
        mask = cv2.imread(str(fr.mask_path), cv2.IMREAD_GRAYSCALE) > 0
        dist = cv2.distanceTransform(mask.astype(np.uint8),
                                     cv2.DIST_L2, 3)
        # GT 表面投影：模型顶点 → 查询像素 (u,v) 与深度 z
        verts_s = verts * s
        pc = (R_gt @ verts_s.T).T + t_gt
        proj = (fr.K @ pc.T).T
        px = proj[:, 0] / proj[:, 2]
        py = proj[:, 1] / proj[:, 2]
        keep = (px > 0) & (px < 639) & (py > 0) & (py < 479) & (proj[:, 2] > 0)
        tree = cKDTree(np.c_[px[keep], py[keep]])
        zs = pc[keep, 2]
        dd, idx = tree.query(pq_full, k=1)
        z_gt = zs[idx]
        okpx = dd < 2.0  # 查询像素 2px 内有 GT 表面
        # 锚点 3D
        z_pred_all, edge_all = [], []
        for i, ti in enumerate(d["template_idx"].astype(int)):
            s0, s1 = int(seg[i]), int(seg[i + 1])
            if s1 <= s0:
                continue
            P = bank.coord_maps[ti][d["pix_t"][s0:s1, 1].astype(int),
                                    d["pix_t"][s0:s1, 0].astype(int)]
            z_pred = (R_gt @ P.T).T + t_gt
            z_pred = z_pred[:, 2]
            pq = pq_full[s0:s1]
            if pq.shape[0] == 0:
                continue
            d_edges = dist[np.clip(pq[:, 1].astype(int), 0, 479),
                           np.clip(pq[:, 0].astype(int), 0, 639)]
            m = okpx[s0:s1] & (d_edges >= 0)
            if m.sum() == 0:
                continue
            bias = (z_pred[m] - z_gt[s0:s1][m]) / z_gt[s0:s1][m]
            edge = d_edges[m] <= 3.0
            agg["edge"] += list(bias[edge])
            agg["inner"] += list(bias[~edge])
            z_pred_all += list(bias)
        if z_pred_all:
            per_frame.append((fr.frame_id, float(np.mean(z_pred_all)),
                              float(np.std(z_pred_all))))
    for k, v in agg.items():
        v = np.array(v)
        print(f"[{obj}] {k:6s} n={len(v):6d}  mean_bias={v.mean():+.4f} "
              f"median={np.median(v):+.4f}  p10={np.percentile(v,10):+.4f} "
              f"p90={np.percentile(v,90):+.4f}")
    if per_frame:
        b = np.array([p[1] for p in per_frame])
        print(f"[{obj}] 逐帧 mean_bias: median={np.median(b):+.4f} "
              f"|bias|>2% 的帧数: {(np.abs(b)>0.02).sum()}/{len(b)}")


if __name__ == "__main__":
    main()
