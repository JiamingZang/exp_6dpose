#!/usr/bin/env python3
"""择优判据离线重排（纯 CPU）——07 组消融互补。

输入：带 cand_* 字段的帧缓存（metrics.topk_best 打开 + 08-13 后代码，
见 configs/experiments/dense80_topk_instr.yaml 与 scripts/analysis/
simulate_adaptive_k.py）。对每条择优策略按 selection.py 同款 key 重排
候选（只考虑成功候选，失败项占窗口不参与择优），取首位位姿重算
ADD(S)@0.1d，聚合输出。省去 similarity/weighted/inlier_ratio/reproj
各 3.6h 的 GPU 重跑。

用法：
  python3 scripts/analysis/rerank_selection.py \
      --cache outputs/exp_adaptive_k/duck_<hash8>.jsonl --object duck \
      [--max-frames 120]
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

STRATEGIES = ["inlier", "inlier_ratio", "reproj", "similarity", "weighted"]


def strategy_key(name, c):
    """与 src/solver/selection.py rank_candidates 同语义的排序 key。"""
    if name == "inlier":
        return c["n_inliers"]
    if name == "inlier_ratio":
        ncorr = c.get("n_correspondences") or 0
        return c["n_inliers"] / ncorr if ncorr > 0 else 0.0
    if name == "reproj":
        r = c.get("mean_inlier_reproj_px")
        return -r if r is not None and np.isfinite(r) else -np.inf
    if name == "similarity":
        return c["score"]
    if name == "weighted":
        return c["n_inliers"] * max(c.get("score") or 0.0, 0.0)
    raise ValueError(name)


def run(args):
    cfg = load_config("configs/current/dense80_depthc_guided.yaml")
    ds = LinemodDataset(cfg["dataset"]["root"], args.object,
                        models_dir=cfg["dataset"].get("models_dir", "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = {fr.frame_id: fr for fr in ds.eval_frames(exclude_refs=True)}
    model_pts = ds.model_points(max_points=2000)
    diam = ds.diameter
    err_fn = adds_error if ds.symmetric else add_error

    recs = {}
    for line in Path(args.cache).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('{"__meta__"'):
            continue
        r = json.loads(line)
        recs[r["frame_id"]] = r
    ids = sorted(f for f in recs if f in frames)[:args.max_frames]
    if len(ids) < 16:
        print(f"[rerank] 可用帧不足（{len(ids)}）——缓存需含 cand_* 字段",
              file=sys.stderr)
        return 1

    print(f"[rerank] object={args.object} frames={len(ids)}")
    for strat in args.strategies:
        ok = 0
        for fid in ids:
            r = recs[fid]
            fr = frames[fid]
            cands = []
            for tidx, inl, R, t, ncorr, rpj, sc in zip(
                    r.get("cand_templates") or [],
                    r.get("cand_inliers") or [],
                    r.get("cand_Rs") or [],
                    r.get("cand_ts") or [],
                    r.get("cand_ncorr") or [],
                    r.get("cand_reproj") or [],
                    r.get("cand_scores") or []):
                if R is None or inl is None:
                    continue  # 失败候选不参与择优（与 select_best_candidate 同）
                cands.append({"n_inliers": float(inl), "R": np.asarray(R),
                              "t": np.asarray(t), "n_correspondences": ncorr,
                              "mean_inlier_reproj_px": rpj, "score": sc})
            if not cands:
                continue
            best = max(cands, key=lambda c: strategy_key(strat, c))
            if err_fn(model_pts, fr.R_gt, fr.t_gt,
                      best["R"], best["t"]) <= 0.1 * diam:
                ok += 1
        print(f"  {strat:<12} ADD(S)@0.1d = {100.0 * ok / len(ids):6.2f}%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args()
    args.strategies = [s for s in args.strategies.split(",") if s]
    sys.exit(run(args))


if __name__ == "__main__":
    main()
