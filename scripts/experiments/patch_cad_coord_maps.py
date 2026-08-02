#!/usr/bin/env python
"""根治修复：coord_map 改用 CAD 精确表面（替换高斯中心混合的位置图）。

根因：3DGS 的坐标图是『高斯中心 μ 的 alpha 混合』，训练后中心壳位于真实
表面内侧 ~4-5%（视觉表面 = μ+σ 与真实一致），PnP 的 3D 锚点系统性偏小
→ 深度系统性偏浅 4.2% → ADD@0.1d 崩塌。视觉/掩码/求解器/像素链均验证
无偏（见会话分析）。修复 = 用 pyrender 光栅化 CAD 网格得到逐像素精确表面
点（同一套 poses/K/size，与 3DGS 视觉几何一致，模板 RGB 匹配不受影响）。

用法: python scripts/patch_cad_coord_maps.py --objects ape [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.pipeline import template_bank_path
from src.geometry.cad_depth import rasterize_cad_depth, render_cad_coord_maps
from src.datasets.linemod import LinemodDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense80.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]
    size = int(cfg["templates"].get("image_size", 512))

    for obj in objects:
        bank_path = template_bank_path(cfg, obj)
        if not bank_path.exists():
            print(f"[{obj}] 无模板库 {bank_path}，跳过")
            continue
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        d = np.load(bank_path, allow_pickle=True)
        if "coord_maps_cad" in d.files:
            print(f"[{obj}] 已打过 CAD 补丁，跳过")
            continue
        print(f"[{obj}] 渲染 CAD 表面坐标图 "
              f"({len(d['poses'])} 模板, {size}px)...")
        from src.datasets.ply_io import load_ply
        verts, _, faces = load_ply(ds.model_path)
        cm_cad = render_cad_coord_maps(verts, faces, d["poses"], d["K"],
                                       size)
        print(f"[{obj}] coord_map 由高斯中心版 → CAD 表面版")
        print(f"    中心版半径均值: "
              f"{np.linalg.norm(d['coord_maps'].reshape(-1,3),axis=1)[np.abs(d['coord_maps'].reshape(-1,3)).sum(1)>0].mean():.2f} mm")
        print(f"    CAD 版半径均值: "
              f"{np.linalg.norm(cm_cad.reshape(-1,3),axis=1)[np.abs(cm_cad.reshape(-1,3)).sum(1)>0].mean():.2f} mm")
        if args.dry_run:
            continue
        bak = bank_path.with_suffix(".npz.orig")
        if not bak.exists():
            bank_path.replace(bak)
        out = {k: d[k] for k in d.files}
        out["coord_maps"] = cm_cad
        out["coord_maps_cad"] = np.uint8(1)
        np.savez_compressed(bank_path, **out)
        print(f"[{obj}] 已写入 {bank_path}（原库备份 {bak.name}）")


if __name__ == "__main__":
    main()
