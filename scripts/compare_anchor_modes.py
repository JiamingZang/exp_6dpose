#!/usr/bin/env python
"""对比三种 3D 锚点渲染方式的深度偏差：
1. 位置混合（现 coord_map，高斯中心 μ 的 α 混合）
2. 逆深度混合（官方式：colors=1/z_cam 的 α 混合，Hierarchical 3DGS 深度正则化同款）
3. CAD 表面（z-buffer 光栅化）

用同一批 matches 跑 PnP（no-refine），对比 t_z_pred/t_gt 中位。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.pipeline import TemplateBank, template_bank_path, PoseEstimator
from src.datasets.linemod import LinemodDataset
from src.solver.ransac_pnp import ransac_pnp
from src.matching.correspondence import back_to_original_pixels


def render_anchor_maps(ckpt_path, poses, K, size, mode):
    """用 gsplat 渲染锚点图。mode: 'pos'（μ 位置混合）| 'invdepth'（1/z 混合）。"""
    import torch
    from src.gaussian.gs_trainer import _import_gs
    torch, gsplat = _import_gs()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ck["splats"]
    means = splats["means"].cuda()
    quats = splats["quats"].cuda()
    scales = torch.exp(splats["scales"].cuda())
    opacities = torch.sigmoid(splats["opacities"].cuda())
    K_t = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    maps = []
    with torch.no_grad():
        for T in poses:
            Rt = torch.tensor(T[:3, :3], dtype=torch.float32, device="cuda")
            tt = torch.tensor(T[:3, 3], dtype=torch.float32, device="cuda")
            viewm = torch.eye(4, dtype=torch.float32, device="cuda")
            viewm[:3, :3] = Rt
            viewm[:3, 3] = tt
            z_cam = (means @ Rt.T + tt)[:, 2:3]          # (N,1)
            if mode == "invdepth":
                colors = (1.0 / z_cam).float()
            else:
                colors = means.float()                       # 位置混合
            renders, alphas, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewm[None], Ks=K_t, width=size, height=size,
                sh_degree=None, packed=False)
            m = renders[0].permute(1, 2, 0).cpu().numpy()       # (H,W,C)
            a = alphas[0].cpu().numpy()
            if mode == "invdepth":
                invd = m[..., 0]
                a = a[..., 0]
                z = np.where(a > 1e-4, 1.0 / np.maximum(invd, 1e-9), 0.0)
                z[a <= 1e-4] = 0.0
                maps.append(z.astype(np.float32))
            else:
                pos = m / np.maximum(a[..., None], 1e-6)
                maps.append(pos.astype(np.float32))
    torch.cuda.empty_cache()
    return maps


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pos"
    cfg = load_config("configs/dense80.yaml")
    obj = "ape"
    bank_path = template_bank_path(cfg, obj)
    d = np.load(bank_path, allow_pickle=True)
    print(f"[{mode}] 渲染 80 模板锚点图...")
    maps = render_anchor_maps(
        str(bank_path.with_suffix(".pt")), d["poses"], d["K"],
        int(cfg["templates"].get("image_size", 512)), mode)
    print(f"[{mode}] 锚点图渲染完成 ({len(maps)} 模板)")

    ds = LinemodDataset("data/lm", obj,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = ds.eval_frames(exclude_refs=True,
                            n_ref=int(cfg["onboard"].get("n_ref_views", 64)))
    tzs = []
    for k, fr in enumerate(frames[::8]):
        with np.load(f"outputs/matches_ape_full/ape/{fr.frame_id:06d}.npz",
                     allow_pickle=True) as npz:
            by_t = {}
            for i, ti in enumerate(npz["template_idx"]):
                by_t.setdefault(int(ti), []).append(i)
            q_all, C_all = [], []
            for ti, idxs in by_t.items():
                pq = back_to_original_pixels(
                    npz["pix_q"][idxs].astype(np.float64),
                    (npz["sxy"][0] * npz["s_leg"][0],
                     npz["sxy"][1] * npz["s_leg"][1]), npz["crop_box"])
                pt = npz["pix_t"][idxs].astype(int)
                if mode == "pos":
                    cm = maps[ti]
                    C = cm[pt[:, 1], pt[:, 0]]
                    valid = np.abs(C).sum(axis=1) > 0
                else:
                    z = maps[ti]
                    zz = z[pt[:, 1], pt[:, 0]]
                    valid = zz > 0
                    T = d["poses"][ti]
                    R, t = T[:3, :3], T[:3, 3]
                    Kinv = np.linalg.inv(d["K"])
                    pc = np.stack([
                        (pt[:, 0].astype(np.float64) - d["K"][0, 2]) / d["K"][0, 0] * zz,
                        (pt[:, 1].astype(np.float64) - d["K"][1, 2]) / d["K"][1, 1] * zz,
                        zz], axis=1)
                    C = (pc - t) @ R
                q_all.append(pq[valid]); C_all.append(C[valid])
        if not C_all:
            continue
        q = np.concatenate(q_all); C = np.concatenate(C_all)
        if len(q) < 20:
            continue
        r = ransac_pnp(q, C, fr.K, reproj_px=5.0, confidence=0.999,
                       iterations=1000, refine_lm=True,
                       min_correspondences=6, flag="epnp")
        if r.success:
            tzs.append(r.t[2] / fr.t_gt[2])
        if (k + 1) % 20 == 0:
            print(f"[{mode}] PnP {k+1} 帧完成, 当前 tz 中位 "
                  f"{np.median(tzs):.4f}", flush=True)
    tzs = np.array(tzs)
    print(f"[{mode}] {len(tzs)} 帧: t_z_pred/t_gt = mean {tzs.mean():.4f} "
          f"med {np.median(tzs):.4f} p25-p75 "
          f"{np.percentile(tzs,25):.4f}-{np.percentile(tzs,75):.4f}")


if __name__ == "__main__":
    main()
