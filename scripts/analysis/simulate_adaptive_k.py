#!/usr/bin/env python3
"""自适应 K 早停离线仿真（纯 CPU）。

输入：带 cand_* 字段的帧缓存 jsonl（metrics.topk_best 打开 + 08-13 后代码
落盘的 cand_order/cand_inliers/cand_templates/cand_Rs/cand_ts）。
对每条早停规则（内点 plateau：连续 w 个解码模板内点增益 ≤ δ 即停，
最小解码数 min_k），重建逐帧解码轨迹 → 前缀内点最优位姿 → 重算
ADD(S)@0.1d 与平均有效 K，输出 (ADD, mean K) Pareto。

规则语义与在线实现一一对应：解码按定位 DINOv2 相似度降序（cand_order），
停表即"不再解码剩余模板"，位姿取已解码模板中内点最大者（同内点取先见者，
与择优稳定排序一致）。ADD 用与主表相同的模型点/阈值口径。

用法：
  python3 scripts/analysis/simulate_adaptive_k.py \
      --cache outputs/exp_adaptive_k/duck_<hash8>.jsonl --object duck \
      [--max-frames 120] [--w 2,3,5] [--delta 0,50,200] [--min-k 5,8,12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/mast3r"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/mast3r/dust3r"))

from src.datasets.linemod import LinemodDataset  # noqa: E402
from src.metrics.pose_metrics import add_error, adds_error  # noqa: E402
from src.config import load_config  # noqa: E402


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
    # 与 run_linemod 同序：frame_id 升序取前 max_frames 个
    ids = sorted(r for r in recs if r in frames)[:args.max_frames]
    if len(ids) < 16:
        print(f"[sim] 可用帧不足（{len(ids)}），检查缓存是否含 cand_* 字段 "
              f"（需 topk_best 打开 + 08-13 后代码重跑）", file=sys.stderr)
        return 1

    # 基线：全解码内点最优（= 主表粗位姿口径）
    base_ok = 0
    for fid in ids:
        r = recs[fid]
        fr = frames[fid]
        sol = {}
        for tidx, inl, R, t in zip(r.get("cand_templates") or [], r.get("cand_inliers") or [],
                                   r.get("cand_Rs") or [], r.get("cand_ts") or []):
            if R is None or inl is None:
                continue
            sol.setdefault(inl, (np.asarray(R), np.asarray(t)))
        if not sol:
            continue
        inl = max(sol)
        R, t = sol[inl]
        if err_fn(model_pts, fr.R_gt, fr.t_gt, R, t) <= 0.1 * diam:
            base_ok += 1
    base_add = 100.0 * base_ok / len(ids)

    print(f"[sim] object={args.object} frames={len(ids)} "
          f"baseline(K=40 inlier-best) ADD(S)@0.1d={base_add:.2f}%")
    print(f"{'w':>3} {'δ':>7} {'min_k':>5} | {'ADD%':>7} {'ΔADD':>7} "
          f"{'meanK':>6}")
    results = []
    rules = [(f"abs{delta}", delta, None) for delta in args.delta]
    rules += [(f"rel{ratio}", None, ratio) for ratio in args.ratio]
    for w in args.w:
        for rname, delta, ratio in rules:
            for min_k in args.min_k:
                ok, ksum = 0, 0
                for fid in ids:
                    r = recs[fid]
                    fr = frames[fid]
                    sol = {}
                    for tidx, inl, R, t in zip(
                            r.get("cand_templates") or [],
                            r.get("cand_inliers") or [],
                            r.get("cand_Rs") or [],
                            r.get("cand_ts") or []):
                        if R is None or inl is None:
                            continue
                        sol[tidx] = (float(inl), np.asarray(R), np.asarray(t))
                    order = [t for t in (r.get("cand_order") or []) if t in sol]
                    if not order:
                        ksum += 0
                        continue
                    # 早停扫描：增益须超阈值（绝对 δ 或相对 ratio）才不算停滞
                    best_t, best_inl = order[0], sol[order[0]][0]
                    stall = 0
                    k_used = len(order)
                    for i, tidx in enumerate(order[1:], start=2):
                        inl = sol[tidx][0]
                        improved = (inl > best_inl + delta if delta is not None
                                    else inl > best_inl * (1.0 + ratio))
                        if improved:
                            best_t, best_inl = tidx, inl
                            stall = 0
                        else:
                            stall += 1
                            if i >= min_k and stall >= w:
                                k_used = i - stall
                                break
                    ksum += k_used
                    R, t = sol[best_t][1], sol[best_t][2]
                    if err_fn(model_pts, fr.R_gt, fr.t_gt, R, t) <= 0.1 * diam:
                        ok += 1
                add = 100.0 * ok / len(ids)
                mean_k = ksum / len(ids)
                results.append((w, rname, min_k, add, add - base_add, mean_k))
    results.sort(key=lambda x: -x[3])
    for w, rname, min_k, add, dadd, mean_k in results:
        print(f"{w:>3} {rname:>7} {min_k:>5} | {add:>7.2f} {dadd:>+7.2f} "
              f"{mean_k:>6.1f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--w", default="2,3,5", help="停滞窗口（连续无增益模板数）")
    ap.add_argument("--delta", default="0,50,200", help="绝对增益阈值（内点数）")
    ap.add_argument("--ratio", default="0.02,0.05,0.10",
                    help="相对增益阈值（如 0.02 = 需超当前最优 2%）")
    ap.add_argument("--min-k", default="5,8,12", help="最小解码数")
    args = ap.parse_args()
    args.w = [int(x) for x in args.w.split(",")]
    args.delta = [int(x) for x in args.delta.split(",")]
    args.ratio = [float(x) for x in args.ratio.split(",")]
    args.min_k = [int(x) for x in args.min_k.split(",")]
    sys.exit(run(args))


if __name__ == "__main__":
    main()
