#!/usr/bin/env python
"""把模板库 coord_maps 替换为"逆深度混合"渲染深度锚点（官方 depth-regularization
同款 expected_invdepth），替代 μ 位置混合。

- 训练监督目标（depth_l1_weight>0）就是渲染逆深度 ≈ CAD 表面深度；
- 锚点 = 同一渲染的逆深度反投影 → 3D 点 = 射线 × 表面深度，z/xy 同时正确，
  没有 μ 混合的切向收缩问题（旧代码即此路线）。

用法：
    python scripts/maintenance/patch_depth_anchor_maps.py --config configs/current/dense80.yaml --objects ape
"""
import argparse
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
from src.pipeline import template_bank_path


def render_invdepth_coord_maps(ckpt_path, poses, K, size, bg_thresh=0.5):
    import torch
    from src.gaussian.gs_trainer import _import_gs
    torch, gsplat = _import_gs()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ck["splats"]
    means = splats["means"].cuda()
    quats = splats["quats"].cuda()
    scales = torch.exp(splats["scales"].cuda())
    opacities = torch.sigmoid(splats["opacities"].cuda())
    K_render = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    K_render[0, 0, 2] += 0.5
    K_render[0, 1, 2] += 0.5
    maps = []
    with torch.no_grad():
        for T in poses:
            Rt = torch.tensor(T[:3, :3], dtype=torch.float32, device="cuda")
            tt = torch.tensor(T[:3, 3], dtype=torch.float32, device="cuda")
            viewm = torch.eye(4, dtype=torch.float32, device="cuda")
            viewm[:3, :3] = Rt
            viewm[:3, 3] = tt
            z_cam = (means @ Rt.T + tt)[:, 2:3]
            rend, alphas, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=(1.0 / z_cam).float(),
                viewmats=viewm[None], Ks=K_render, width=size, height=size,
                sh_degree=None, packed=False)
            inv_d = rend[0, :, :, 0].cpu().numpy()
            a = alphas[0, :, :, 0].cpu().numpy()
            z = np.where(a > 1e-4, 1.0 / np.maximum(inv_d, 1e-9), 0.0).astype(np.float32)
            fg = a > bg_thresh
            z[~fg] = 0.0
            ys, xs = np.nonzero(z > 0)
            zz = z[ys, xs]
            pc = np.stack([
                (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * zz,
                (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * zz,
                zz], axis=1)
            po = (pc - T[:3, 3]) @ T[:3, :3]
            cm = np.zeros((size, size, 3), dtype=np.float32)
            cm[ys, xs] = po.astype(np.float32)
            maps.append(cm)
    torch.cuda.empty_cache()
    return np.stack(maps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/current/dense80.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--write-field", default="coord_maps",
                    help="coord_maps=覆盖主字段（备份 .orig）；或写入 coord_maps_depth 字段")
    args = ap.parse_args()
    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]
    for obj in objects:
        bp = template_bank_path(cfg, obj)
        d = np.load(bp, allow_pickle=True)
        if args.write_field == "coord_maps" and bp.with_suffix(bp.suffix + ".orig").exists():
            print(f"[{obj}] 已存在 .orig，跳过（先恢复或删除备份）")
            continue
        print(f"[{obj}] 渲染 {len(d['poses'])} 模板逆深度锚点...")
        cm = render_invdepth_coord_maps(
            str(bp.with_suffix(".pt")), d["poses"], d["K"],
            int(cfg["templates"].get("image_size", 512)))
        out = {k: v for k, v in d.items()}
        if args.write_field == "coord_maps":
            backup = str(bp) + ".orig"
            Path(backup).write_bytes(bp.read_bytes())
            out["coord_maps"] = cm
            print(f"[{obj}] 已备份 {Path(backup).name}，覆盖 coord_maps")
        else:
            out["coord_maps_depth"] = cm
            print(f"[{obj}] 写入 coord_maps_depth 字段")
        np.savez_compressed(bp, **out)


if __name__ == "__main__":
    main()
