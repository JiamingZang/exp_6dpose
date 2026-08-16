#!/usr/bin/env python3
"""早停前缀内选择键离线扫描（纯 CPU）。

问题：v1 早停的前缀内择优固定用 inlier-best，但 6d-adaptive-k 已证实
正确候选内点"从不更高"（0% 帧更高，Δinlier 中位 -93）——选择键可能
不是最优。本脚本对同一缓存重放早停轨迹（与 simulate_adaptive_k.py
同款 plateau 语义），在停止前缀内按不同 key 择优，比较 ADD(S)@0.1d。

key 语义（与 rerank_selection.py / selection.py 对齐）：
  inlier         = n_inliers（v1 现状）
  sim            = MASt3R 匹配分数（cand_scores）
  weighted       = n_inliers * max(score, 0)
  inlier_ratio   = n_inliers / n_correspondences
  rank           = 解码序最前（DINOv2 相似度最高）
  oracle         = 前缀内 GT-ADD 最优（上界，不现实）

--floor 加"弱前缀不早停"门：前缀内当前最优内点 < floor 时禁止停
（plateau 信号在弱前缀里是噪声——hp 类正确候选排深，见 6d-adaptive-k）。
通用单参数，无逐物体调参。

用法：
  python3 scripts/analysis/simulate_select_key.py \
      --cache outputs/exp_adaptive_k/<obj>.jsonl --object <obj> [--floor 0]
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

KEYS = ["inlier", "sim", "weighted", "inlier_ratio", "rank", "oracle"]


def key_value(name, cand):
    inl, score, ncorr = cand["inl"], cand["score"], cand["ncorr"]
    if name == "inlier":
        return inl
    if name == "sim":
        return score if score is not None else -np.inf
    if name == "weighted":
        return inl * max(score or 0.0, 0.0)
    if name == "inlier_ratio":
        return inl / ncorr if ncorr and ncorr > 0 else 0.0
    if name == "rank":
        return -cand["rank"]
    raise ValueError(name)


def load_records(path: Path):
    recs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('{"__meta__"'):
            continue
        r = json.loads(line)
        recs[r["frame_id"]] = r
    return recs


def run(args):
    cfg = load_config("configs/current/dense80_depthc_guided.yaml")
    ds = LinemodDataset(cfg["dataset"]["root"], args.object,
                        models_dir=cfg["dataset"].get("models_dir", "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = {fr.frame_id: fr for fr in ds.eval_frames(exclude_refs=True)}
    model_pts = ds.model_points(max_points=2000)
    diam = ds.diameter
    err_fn = adds_error if ds.symmetric else add_error

    recs = load_records(Path(args.cache))
    ids = sorted(r for r in recs if r in frames)[:args.max_frames]

    rules = [(w, delta, ratio, min_k)
             for w, delta, ratio, min_k in (
                 (2, None, 0.05, 8),   # 在线 es 档
                 (3, None, 0.10, 5),   # 仿真最优档
                 (3, 50, None, 5),
                 (5, None, 0.05, 8))]
    print(f"[key] object={args.object} frames={len(ids)} floor={args.floor}")
    print(f"{'rule':>22} | " + " ".join(f"{k:>8}" for k in KEYS) + " |  meanK")
    for w, delta, ratio, min_k in rules:
        rname = (f"abs{delta}" if delta is not None
                 else f"rel{ratio}") + f" w{w} mk{min_k}"
        ok = {k: 0 for k in KEYS}
        ksum = 0
        for fid in ids:
            r = recs[fid]
            fr = frames[fid]
            cands = []
            for tidx, inl, ncorr, score, R, t in zip(
                    r.get("cand_templates") or [],
                    r.get("cand_inliers") or [],
                    r.get("cand_ncorr") or [],
                    r.get("cand_scores") or [],
                    r.get("cand_Rs") or [],
                    r.get("cand_ts") or []):
                if R is None or inl is None:
                    continue
                cands.append({"tidx": tidx, "inl": float(inl),
                              "ncorr": ncorr, "score": score,
                              "R": np.asarray(R), "t": np.asarray(t)})
            order = [t for t in (r.get("cand_order") or [])]
            by_tidx = {c["tidx"]: c for c in cands}
            seq = [by_tidx[t] for t in order if t in by_tidx]
            if not seq:
                ksum += 0
                continue
            best_c, best_inl = seq[0], seq[0]["inl"]
            stall = 0
            k_used = len(seq)
            for i, c in enumerate(seq[1:], start=2):
                improved = (c["inl"] > best_inl + delta if delta is not None
                            else c["inl"] > best_inl * (1.0 + ratio))
                if improved:
                    best_c, best_inl = c, c["inl"]
                    stall = 0
                else:
                    stall += 1
                    if i >= min_k and stall >= w and best_inl >= args.floor:
                        k_used = i - stall
                        break
            ksum += k_used
            prefix = seq[:k_used]
            for k in KEYS:
                if k == "oracle":
                    chosen = min(prefix, key=lambda c: err_fn(
                        model_pts, fr.R_gt, fr.t_gt, c["R"], c["t"]))
                else:
                    for rank, c in enumerate(prefix):
                        c["rank"] = rank
                    chosen = max(prefix, key=lambda c: key_value(k, c))
                if err_fn(model_pts, fr.R_gt, fr.t_gt,
                          chosen["R"], chosen["t"]) <= 0.1 * diam:
                    ok[k] += 1
        mean_k = ksum / len(ids)
        print(f"{rname:>22} | " + " ".join(
            f"{100.0 * ok[k] / len(ids):>7.2f}" for k in KEYS) +
            f" | {mean_k:>6.1f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--floor", type=float, default=0.0,
                    help="弱前缀不早停门：最优内点 < floor 时禁止停")
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
