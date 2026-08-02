#!/usr/bin/env python
"""从帧级缓存 jsonl 聚合全量指标（分片阶段 3 完成后一键出数）。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.pose_metrics import aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/cache_ape_full/ape.jsonl")
    ap.add_argument("--name", default="ape")
    args = ap.parse_args()

    recs = []
    seen = {}
    for line in Path(args.cache).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen[int(r["frame_id"])] = r   # 并行分片可能重复写帧，按 frame_id 去重
    recs = [seen[k] for k in sorted(seen)]
    ms = [r["m"] for r in recs]
    n_ok = sum(1 for r in recs if r["success"])
    print(f"{args.name}: {len(ms)} 帧, 成功 {n_ok} ({100*n_ok/max(1,len(ms)):.1f}%)")
    agg = aggregate(ms)
    print(f"  ADD(S)@0.1d={agg['add_01d']:.2f}%  Proj@5px={agg['proj_5px']:.2f}%  "
          f"5cm5°={agg['cm_deg']:.2f}%")
    if any("mssd_recall" in m for m in ms):
        from src.metrics.bop_metrics import aggregate_bop
        b = aggregate_bop(ms)
        print(f"  BOP: AR_MSSD={b['ar_mssd']:.2f}% AR_MSPD={b['ar_mspd']:.2f}% "
              f"AR2={b['ar_bop']:.2f}%")


if __name__ == "__main__":
    main()
