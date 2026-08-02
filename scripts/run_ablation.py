#!/usr/bin/env python3
"""消融实验（8 组）。GPU 机器运行。

用法：
    python scripts/run_ablation.py --ablation configs/ablations/01_topk.yaml
    python scripts/run_ablation.py --all                 # 顺序跑全部 8 组
    python scripts/run_ablation.py --ablation ... --objects ape --max-frames 200

行为：
- 每个取值按点号路径覆盖 default.yaml 后全量评测；
- 标注 requires_reonboard 的组（模板数/几何/尺度/渲染器）会自动检查
  对应模板库是否存在，缺失则先跑 onboard（离线产物按配置组合命名，互不覆盖）。

输出：outputs/ablation_<name>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_ablation, load_config
from src.pipeline import onboard_object, template_bank_path
from run_linemod import print_table, run_eval  # noqa: E402（同目录导入）


def run_one_ablation(ablation_path, base_cfg, objects, device, max_frames):
    name, runs = load_ablation(base_cfg, ablation_path)
    all_results = {}
    for label, cfg, reonboard in runs:
        print(f"\n===== 消融 {label} =====")
        # 需要重建离线产物的组：缺模板库就地 onboard
        for obj in objects:
            if not template_bank_path(cfg, obj).exists():
                if not reonboard:
                    raise FileNotFoundError(
                        f"模板库缺失且该消融不改离线产物，请先跑默认 onboard: "
                        f"{template_bank_path(cfg, obj)}")
                print(f"[onboard] {label} / {obj} ...")
                onboard_object(cfg, obj, device=device)
        try:
            results = run_eval(cfg, objects, device, max_frames=max_frames)
        except NotImplementedError as e:
            # loftr 等 TODO 接口：记录跳过原因，不中断整组消融
            print(f"[skip] {label}: {e}")
            all_results[label] = {"skipped": str(e)}
            continue
        mean = print_table(results)
        all_results[label] = {"per_object": results, "mean": mean}
    return name, all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=None,
                    help="configs/ablations/ 下的单个 yaml")
    ap.add_argument("--all", action="store_true", help="顺序跑全部 8 组")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    device = args.device or base_cfg["runtime"].get("device", "cuda")
    objects = args.objects or base_cfg["dataset"]["objects"]

    if args.all:
        paths = sorted(Path("configs/ablations").glob("*.yaml"))
    elif args.ablation:
        paths = [Path(args.ablation)]
    else:
        ap.error("需要 --ablation <yaml> 或 --all")

    out_dir = Path(base_cfg["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        name, results = run_one_ablation(p, base_cfg, objects, device,
                                         args.max_frames)
        out = out_dir / f"ablation_{name}.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n[{name}] 结果已写入 {out}")


if __name__ == "__main__":
    main()
