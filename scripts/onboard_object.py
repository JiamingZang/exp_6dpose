#!/usr/bin/env python3
"""物体准备阶段（离线段）。GPU 机器运行。

用法：
    python scripts/onboard_object.py --config configs/default.yaml \
        --objects ape cat            # 缺省 = 配置里全部 13 物体

产物：outputs/templates/<obj>_<renderer>_<geo>_<N>t_<sa>.npz
    含 40 模板 RGB / alpha / 3D 坐标图 / 位姿 P_m / DINOv2 CLS 特征 / 尺度 s
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import onboard_object


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--objects", nargs="*", default=None,
                    help="缺省用配置中的全部物体")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["runtime"].get("device", "cuda")
    objects = args.objects or cfg["dataset"]["objects"]

    for obj in objects:
        t0 = time.time()
        path = onboard_object(cfg, obj, device=device)
        print(f"[done] {obj}: {time.time()-t0:.0f}s → {path}")


if __name__ == "__main__":
    main()
