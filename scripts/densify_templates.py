#!/usr/bin/env python3
"""模板加密：复用已训练的 3DGS 参数，只重新渲染更密模板库。

粗位姿精度瓶颈之一是模板视点间隔：cube8 顶点间角距离 70.5°（任意位姿
离最近模板方向最大偏 ~35°）。视点加密到 fibonacci 16 后最大间隔降到
48.6°，匹配对应的几何误差随之下降（与消融 02_n_templates 的 80 档一致）。

用法：
    python scripts/densify_templates.py --objects ape
    python scripts/densify_templates.py                    # 全部 13 物体

产物：<template_dir>/<obj>_3dgs_cad_80t_sa.npz（新文件名，不覆盖 40t），
并把旧 40t 的 .pt 软链到新文件名（refiner/渲染验证共用同一份 3DGS 参数）。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.pipeline import TemplateBank, template_bank_path


class SplatRenderer:
    """只渲染不训练的 3DGS 载体：从 onboard 落盘 ckpt 加载 splats。

    接口对齐 GaussianTrainer.render / gaussian_centers（template_renderer
    只依赖这两处 + torch 属性），无需重新初始化训练器。
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        import gsplat
        self.torch = torch
        self.gsplat = gsplat
        self.device = device
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.splats = {name: torch.nn.Parameter(v.to(device))
                       for name, v in ck["splats"].items()}
        self.sh_degree = int(ck.get("sh_degree", 3))

    def gaussian_centers(self):
        return self.splats["means"].detach()

    def render(self, viewmat: np.ndarray, K: np.ndarray,
               width: int, height: int, colors_override=None,
               sh_degree=None):
        viewmats = torch.tensor(viewmat, dtype=torch.float32,
                                device=self.device)[None]
        Ks = torch.tensor(K, dtype=torch.float32, device=self.device)[None]
        if colors_override is None:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
            sh_degree = self.sh_degree
        else:
            colors = colors_override
            sh_degree = None
        renders, alphas, _ = self.gsplat.rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height,
            sh_degree=sh_degree, packed=False,
        )
        return renders[0], alphas[0], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-viewpoints", type=int, default=16)
    ap.add_argument("--viewpoint-mode", default="fibonacci",
                    help="密档用 fibonacci（cube8 模式强制 8 视角）")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="目标模板库已存在时仍重渲染")
    args = ap.parse_args()

    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]
    bg = float(cfg["onboard"].get("bg_color", 1.0))
    orig_n = int(cfg["templates"]["n_viewpoints"])
    orig_mode = str(cfg["templates"]["viewpoint_mode"])

    from src.gaussian.template_renderer import render_template_bank

    for obj in objects:
        old_path = template_bank_path(cfg, obj)          # 40t（现有）
        if not old_path.exists():
            print(f"[densify:{obj}] 缺现有模板库 {old_path}，跳过")
            continue
        old_ckpt = old_path.with_suffix(".pt")
        if not old_ckpt.exists():
            print(f"[densify:{obj}] 缺 3DGS ckpt {old_ckpt}，跳过")
            continue

        cfg["templates"]["n_viewpoints"] = args.n_viewpoints
        cfg["templates"]["viewpoint_mode"] = args.viewpoint_mode
        new_path = template_bank_path(cfg, obj)          # 80t（新文件名）
        if new_path.exists() and not args.force:
            cfg["templates"]["n_viewpoints"] = orig_n    # 恢复，下次循环仍用旧路径
            cfg["templates"]["viewpoint_mode"] = orig_mode
            print(f"[densify:{obj}] 已存在，跳过: {new_path}")
            continue

        old = np.load(old_path, allow_pickle=True)
        renderer = SplatRenderer(str(old_ckpt), device=args.device)
        bank = render_template_bank(renderer, cfg["templates"], new_path,
                                    bg_color=bg)
        # 防呆：渲染数必须等于 视角数×平面内旋转数，否则文件名与内容不符
        expect = args.n_viewpoints * int(cfg["templates"]["n_inplane"])
        assert len(bank["images"]) == expect, \
            f"{obj} 渲染 {len(bank['images'])} 模板 != 期望 {expect}"
        # scale/align 与 3DGS 无关（几何/评测侧），直接沿用旧库
        for k in ("scale", "align_s", "align_R", "align_t"):
            if k in old:
                bank[k] = old[k]

        from src.detection.localize import Dinov2Embedder
        embedder = Dinov2Embedder(cfg["detection"], device=args.device)
        bank["dino_feats"] = embedder.template_features(
            bank["images"]).astype(np.float32)
        np.savez_compressed(new_path, **bank)
        TemplateBank(new_path)   # 校验完整性

        # refiner/渲染验证按 template_bank_path 推导 .pt 路径，
        # 新 80t 名没有自己的 .pt，软链旧参数（同一份模型）。
        # 必须用绝对路径目标 + lexists：相对链接从链接所在目录解析会断链，
        # .exists() 对断链返回 False（会重复 symlink 报 FileExistsError）
        new_ckpt = new_path.with_suffix(".pt")
        if not os.path.lexists(new_ckpt):
            os.symlink(os.path.abspath(old_ckpt), os.path.abspath(new_ckpt))
        print(f"[densify:{obj}] {len(bank['images'])} 模板 → {new_path}")

        cfg["templates"]["n_viewpoints"] = orig_n        # 恢复，后续物体用旧路径
        cfg["templates"]["viewpoint_mode"] = orig_mode


if __name__ == "__main__":
    main()
