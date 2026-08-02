#!/usr/bin/env python
"""用"固定旧模板视图 + DS 训练高斯"重建模板库。

背景：重训 3DGS 后 onboard 重新采样模板视图（渲染距离随高斯包围盒微变），
与阶段 2 extract 时使用的模板像素（pix_t）错位 ~0.5%，系统性降低 ADD。
方案：模板视图（poses/K/radius）复用旧 bank（.orig，与 matches 一致），
images/alphas/coord_maps 用新（深度监督训练）高斯渲染，锚点为逆深度
混合反投影（官方 depth-regularization 同款），dino_feats 重新计算。

用法：
    python scripts/rebuild_bank_fixed_views.py --config configs/dense80.yaml \
        --objects ape --views-from outputs/templates/ape_3dgs_cad_80t_sa.npz.orig
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import template_bank_path


def render_all(ckpt_path, poses, K, size, bg_color=1.0, alpha_fg=0.5,
               sh_degree=3):
    """新高斯渲染：RGB / alpha / 逆深度锚点。返回 dict。"""
    import torch
    from src.gaussian.gs_trainer import _import_gs
    torch, gsplat = _import_gs()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ck["splats"]
    means = splats["means"].cuda()
    quats = splats["quats"].cuda()
    scales = torch.exp(splats["scales"].cuda())
    opacities = torch.sigmoid(splats["opacities"].cuda())
    sh0 = splats.get("sh0", splats.get("f_dc"))
    if sh0 is None:
        raise ValueError("splats 缺少 sh0/f_dc")
    sh0 = sh0.cuda()
    shN = splats.get("shN")
    shN = shN.cuda() if shN is not None else None
    colors_sh = torch.cat([sh0, shN], dim=1) if shN is not None else sh0
    K_render = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    K_render[0, 0, 2] += 0.5
    K_render[0, 1, 2] += 0.5
    images, alphas, cmaps = [], [], []
    with torch.no_grad():
        for T in poses:
            Rt = torch.tensor(T[:3, :3], dtype=torch.float32, device="cuda")
            tt = torch.tensor(T[:3, 3], dtype=torch.float32, device="cuda")
            viewm = torch.eye(4, dtype=torch.float32, device="cuda")
            viewm[:3, :3] = Rt
            viewm[:3, 3] = tt
            # RGB（完整 SH，与 trainer.render 一致）
            rgb, a_rgb, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors_sh, viewmats=viewm[None], Ks=K_render,
                width=size, height=size, sh_degree=sh_degree, packed=False)
            rgb = rgb[0].cpu().numpy()
            a = a_rgb[0, :, :, 0].cpu().numpy()
            rgb = np.clip(rgb, 0, 1)
            rgb = rgb + (1.0 - a[..., None]) * bg_color
            images.append((rgb * 255).astype(np.uint8))
            alphas.append(a.astype(np.float16))
            # 逆深度锚点
            z_cam = (means @ Rt.T + tt)[:, 2:3]
            rend, a_iz, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=(1.0 / z_cam).float(), viewmats=viewm[None], Ks=K_render,
                width=size, height=size, sh_degree=None, packed=False)
            inv_d = rend[0, :, :, 0].cpu().numpy()
            ai = a_iz[0, :, :, 0].cpu().numpy()
            z = np.where(ai > 1e-4, 1.0 / np.maximum(inv_d, 1e-9), 0.0).astype(np.float32)
            fg = ai > alpha_fg
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
            cmaps.append(cm)
    torch.cuda.empty_cache()
    return np.stack(images), np.stack(alphas), np.stack(cmaps)


def compute_dino_feats(images_u8, cfg_det):
    from src.detection.localize import Dinov2Embedder
    emb = Dinov2Embedder(cfg_det)
    return emb.template_features(images_u8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense80.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--views-from", default=None,
                    help="旧 bank npz（复用其 poses/K/radius）；缺省 = 目标库的 .orig")
    args = ap.parse_args()
    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]
    for obj in objects:
        bp = template_bank_path(cfg, obj)
        src = args.views_from or str(bp) + ".orig"
        old = np.load(src, allow_pickle=True)
        size = int(cfg["templates"].get("image_size", 512))
        print(f"[{obj}] 用 {Path(src).name} 的视图渲染 {len(old['poses'])} 模板...")
        bg = float(cfg["onboard"].get("bg_color", 1.0))
        images, alphas, cmaps = render_all(
            str(bp.with_suffix(".pt")), old["poses"], old["K"], size,
            bg_color=bg)
        cfg_det = cfg.get("detection", {})
        print(f"[{obj}] 计算 dino_feats...")
        feats = compute_dino_feats(images, cfg_det)
        out = {
            "images": images, "alphas": alphas, "coord_maps": cmaps,
            "poses": old["poses"].astype(np.float32),
            "K": old["K"].astype(np.float32),
            "radius": old["radius"], "scale": old["scale"],
            "dino_feats": feats,
            "bg_color": np.float32(bg),
        }
        backup = str(bp) + ".viewsbak"
        if not Path(backup).exists():
            Path(backup).write_bytes(bp.read_bytes())
        np.savez_compressed(bp, **out)
        print(f"[{obj}] 完成 → {bp.name}（备份 {Path(backup).name}）")


if __name__ == "__main__":
    main()
