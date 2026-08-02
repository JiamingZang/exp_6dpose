#!/usr/bin/env python3
"""分阶段计时（速度对比表用）。GPU 机器运行。

对指定物体的前 N 帧统计：定位（SAM+DINOv2）/ MASt3R 匹配 / RANSAC-PnP
三个阶段的均值耗时与端到端 FPS。首帧包含 CUDA 编译/缓存开销，默认丢弃
前 warmup 帧。

用法：
    python scripts/run_speed.py --object ape --n-frames 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.config import load_config
from src.datasets.linemod import LinemodDataset
from src.pipeline import PoseEstimator, TemplateBank, template_bank_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--object", default="ape")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["runtime"].get("device", "cuda")

    ds = LinemodDataset(cfg["dataset"]["root"], args.object,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"))
    bank = TemplateBank(template_bank_path(cfg, args.object))
    estimator = PoseEstimator(
        cfg, bank, device=device,
        refiner_ckpt=str(template_bank_path(cfg, args.object).with_suffix(".pt")))

    records = []
    for fr in ds.frames()[:args.n_frames + args.warmup]:
        img = cv2.cvtColor(cv2.imread(str(fr.rgb_path)), cv2.COLOR_BGR2RGB)
        gt_mask = None
        if fr.mask_path is not None:
            gt_mask = cv2.imread(str(fr.mask_path), cv2.IMREAD_GRAYSCALE) > 0
        res = estimator.estimate(img, fr.K, gt_bbox=fr.bbox_visib,
                                 gt_mask=gt_mask)
        records.append(res.timings)

    records = records[args.warmup:]
    stages = ["localize", "matching", "pnp"]
    print(f"\n物体 {args.object}，{len(records)} 帧"
          f"（K={cfg['matching']['top_k']}，"
          f"{cfg['templates']['n_viewpoints']}×{cfg['templates']['n_inplane']} 模板）")
    total = 0.0
    for s in stages:
        vals = [r[s] * 1000 for r in records if s in r]
        m = float(np.mean(vals)) if vals else 0.0
        total += m
        print(f"  {s:<10}: {m:8.1f} ms")
    print(f"  {'total':<10}: {total:8.1f} ms  →  {1000.0/total:.1f} FPS")


if __name__ == "__main__":
    main()
