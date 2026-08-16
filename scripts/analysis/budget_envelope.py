#!/usr/bin/env python3
"""预算-精度包络线（纯 CPU）：候选池信息量上限分析。

对每条帧按有效候选过滤序（与 simulate_adaptive_k 同语义）取前 k 个
候选，GT-ADD 择优（oracle），输出 (k, MEAN ADD)。k=40 与 k=12 相等
⇒ 池的全部信息在前 12 个有效候选内（通用结论，5 物体一致）；
官方 K 曲线的晚期增益来自原始序中无效候选占位。

用法：
  python3 scripts/analysis/budget_envelope.py [--objects duck,ape,...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/mast3r"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/mast3r/dust3r"))

from src.config import load_config  # noqa: E402
from src.datasets.linemod import LinemodDataset  # noqa: E402
from src.metrics.pose_metrics import add_error, adds_error  # noqa: E402

KS = [1, 2, 4, 8, 12, 20, 30, 40]


def load_seqs(cfg, obj):
    ds = LinemodDataset(cfg["dataset"]["root"], obj,
                        models_dir=cfg["dataset"].get("models_dir", "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = {fr.frame_id: fr for fr in ds.eval_frames(exclude_refs=True)}
    mp = ds.model_points(max_points=2000)
    diam = ds.diameter
    err_fn = adds_error if ds.symmetric else add_error
    seqs = []
    for line in (Path(f"outputs/exp_adaptive_k/cache/{obj}.jsonl")
                 .read_text().splitlines()):
        if not line or line.startswith('{"__meta__"'):
            continue
        r = json.loads(line)
        fr = frames[r["frame_id"]]
        cands = {}
        for tidx, inl, score, R, t in zip(
                r.get("cand_templates") or [], r.get("cand_inliers") or [],
                r.get("cand_scores") or [], r.get("cand_Rs") or [],
                r.get("cand_ts") or []):
            if R is None or inl is None:
                continue
            e = err_fn(mp, fr.R_gt, fr.t_gt, np.asarray(R), np.asarray(t))
            cands[tidx] = (inl, score, np.asarray(R), np.asarray(t), e)
        order = [t for t in (r.get("cand_order") or []) if t in cands]
        if order:
            seqs.append([cands[t] for t in order])
    return seqs, diam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="duck,ape,cat,holepuncher,phone")
    args = ap.parse_args()
    cfg = load_config("configs/current/dense80_depthc_guided.yaml")
    objs = args.objects.split(",")
    print(f"{'k':>3} | " + " ".join(f"{o[:4]:>7}" for o in objs) + " | MEAN")
    for k in KS:
        res = []
        for obj in objs:
            seqs, diam = load_seqs(cfg, obj)
            ok = 0
            for seq in seqs:
                best = min(seq[:k], key=lambda c: c[4])
                if best[4] <= 0.1 * diam:
                    ok += 1
            res.append(100.0 * ok / len(seqs))
        print(f"{k:>3} | " + " ".join(f"{r:>7.1f}" for r in res) +
              f" | {sum(res) / len(res):5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
