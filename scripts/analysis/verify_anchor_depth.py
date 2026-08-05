"""锚点深度验证：参考帧 GT 位姿下，3DGS 渲染深度 vs CAD 真实深度。

3DGS 几何偏差（k1 面积 std 5.5%）是否反映为锚点深度偏差？直接对比
两个深度图（GT 掩码内），量化 PnP 锚点的深度精度天花板。
--mode 选择锚点渲染方式：
  coord（默认）：直接 μ 坐标 alpha 混合（template_renderer 现状）
  invdepth：逆深度混合（patch_depth_anchor_maps 同款，expected_invdepth）

用法: python scripts/analysis/verify_anchor_depth.py --obj holepuncher [--mode invdepth]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))


def render_coord_depth(ref, R, t, K_int, H, W, mode="coord"):
    """渲染 3DGS 物体系坐标图 → 相机系深度。

    coord：μ 坐标 alpha 混合（模板库现状，深层高斯拉远 → 深度偏大）
    invdepth：逆深度混合（patch_depth_anchor_maps 同款）——1/z 混合后
    取倒数，近处高斯主导，接近最近表面。
    """
    K_render = K_int.copy().astype(np.float64)
    K_render[0, 2] += 0.5
    K_render[1, 2] += 0.5
    Kt = torch.tensor(K_render, dtype=torch.float32, device="cuda")[None]
    Rt = torch.tensor(R, dtype=torch.float32, device="cuda")
    tt = torch.tensor(t, dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, device="cuda")
    viewmat[:3, :3] = Rt
    viewmat[:3, 3] = tt
    viewmat = viewmat[None]
    if mode == "invdepth":
        z_cam = (ref.splats["means"] @ Rt.T + tt)[:, 2:3]
        colors = (1.0 / z_cam.clamp(min=1e-3)).clamp(max=1e3)
        sh_deg = None
    else:
        colors = torch.cat([ref.splats["means"][:, None, :],
                            torch.zeros_like(ref.splats["shN"])], dim=1)
        sh_deg = ref.sh_degree
    renders, alphas, _ = ref.gsplat.rasterization(
        means=ref.splats["means"], quats=ref.splats["quats"],
        scales=torch.exp(ref.splats["scales"]),
        opacities=torch.sigmoid(ref.splats["opacities"]),
        colors=colors, viewmats=viewmat, Ks=Kt,
        width=W, height=H, sh_degree=sh_deg, packed=False)
    feat = renders[0].detach().cpu().numpy()
    alpha = alphas[0, ..., 0].detach().cpu().numpy()
    if mode == "invdepth":
        inv_d = feat[..., 0]
        depth = np.where(alpha > 1e-4, 1.0 / np.maximum(inv_d, 1e-9), 0.0)
    else:
        P = feat @ R.T + t.reshape(1, 3)          # 相机系
        depth = P[..., 2]
    return depth, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="holepuncher")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mode", default="coord", choices=["coord", "invdepth"])
    args = ap.parse_args()

    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner
    from src.geometry.cad_depth import rasterize_cad_depth

    ds = LinemodDataset("data/lm", args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    ref_ids = sorted(ds.reference_frame_ids(args.n))
    ref = PoseRefiner(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt",
                      device="cuda", iterations=0)
    from src.datasets.ply_io import load_ply
    verts, _, faces = load_ply(str(ds.model_path))

    import imageio.v2 as iio
    rels, d3s = [], []
    for fid in ref_ids[:args.n]:
        f = frames[fid]
        if f.mask_path is None:
            continue
        gm = iio.imread(f.mask_path) > 0
        H, W = gm.shape[:2]
        K = f.K
        d_gs, alpha = render_coord_depth(ref, f.R_gt, f.t_gt, K, H, W,
                                         mode=args.mode)
        d_cad = rasterize_cad_depth(verts, faces, f.R_gt, f.t_gt, K, (H, W))
        valid = gm & (d_gs > 0) & (d_cad > 0)
        if valid.sum() < 100:
            continue
        dgs, dcd = d_gs[valid], d_cad[valid]
        rel = (dgs - dcd) / dcd
        rels.append(rel)
        # 3D 误差：反投影 xy（K 整数约定）→ 相机系 3D 距离
        ys, xs = np.nonzero(valid)
        zz = dgs
        pc = np.stack([(xs.astype(np.float64) - K[0, 2]) / K[0, 0] * zz,
                       (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * zz,
                       zz], axis=1)
        pc_cad = np.stack([(xs.astype(np.float64) - K[0, 2]) / K[0, 0] * dcd,
                           (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * dcd,
                           dcd], axis=1)
        d3 = np.linalg.norm(pc - pc_cad, axis=1)
        d3s.append(d3)
    rel = np.concatenate(rels)
    print(f"{args.obj} [{args.mode}]: {len(rels)} ref frames, "
          f"{len(rel)} valid pixels")
    if d3s:
        d3 = np.concatenate(d3s)
        print(f"  3D err: med {np.median(d3):.2f}mm "
              f"mean {d3.mean():.2f}mm p90 {np.percentile(d3, 90):.2f}mm")
    print(f"3DGS-vs-CAD depth rel err: med {np.median(rel)*100:.2f}% "
          f"std {rel.std()*100:.2f}% p90 "
          f"{np.percentile(np.abs(rel),90)*100:.2f}%")
    frmed = np.array([np.median(r) for r in rels])
    print(f"per-frame median: med {np.median(frmed)*100:.2f}% "
          f"std {frmed.std()*100:.2f}% [min {frmed.min()*100:.2f}%, "
          f"max {frmed.max()*100:.2f}%]")


if __name__ == "__main__":
    main()
