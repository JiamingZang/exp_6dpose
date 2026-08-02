#!/usr/bin/env python
"""按评测帧列表位置统计分片完成数（ID 有偏移，不能按帧号直接比较）。

用法: python scripts/count_seg_frames.py <obj> <first> <last> [--matches-dir DIR]
输出: <done>/<total>
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--matches-dir", default="outputs/matches_ape_full")
    args = ap.parse_args()

    cfg = load_config("configs/dense80.yaml")
    from src.datasets.linemod import LinemodDataset
    ds = LinemodDataset(cfg["dataset"]["root"], args.obj,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    ids = [f.frame_id for f in ds.eval_frames(
        exclude_refs=True,
        n_ref=int(cfg["onboard"].get("n_ref_views", 64)))]
    sl = set(ids[args.first:args.last])
    files = {int(os.path.basename(p)[:-4])
             for p in glob.glob(f"{args.matches_dir}/{args.obj}/*.npz")
             if "alt" not in os.path.basename(p)}
    print(f"{len(files & sl)}/{len(sl)}")


if __name__ == "__main__":
    main()
