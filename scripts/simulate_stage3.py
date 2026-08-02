#!/usr/bin/env python3
"""阶段 3 离线仿真：不改 extract，模拟对应质量/择优策略对粗位姿的影响。

对落盘 matches 做两类消融（只算粗位姿+条件 IoU，不跑 LPIPS 精化）：
  A. sim_threshold：按每对匹配的相似度过滤（0.30 现状 / 0.35 / 0.40 / 0.45）
  B. selection strategy：inlier / weighted / reproj / similarity

用法：
    python scripts/simulate_stage3.py                # 全部组合
    python scripts/simulate_stage3.py --sims 0.30 0.40 --strat inlier weighted
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.pipeline import (TemplateBank, load_extracted_matches,
                          subsample_frames, template_bank_path)
from scripts.test_fast_mode import MiniSolver, solve_with_alt


def filter_matches(ex, sim_thresh):
    """按相似度阈值过滤每模板的匹配对（返回新 ex，不动原数据）。"""
    import copy
    ex2 = copy.deepcopy(ex)
    out = []
    for m in ex2["matches"]:
        keep = m.sims >= sim_thresh
        if keep.sum() < 6:
            continue
        m.pix_q = m.pix_q[keep]
        m.pix_t = m.pix_t[keep]
        m.sims = m.sims[keep]
        out.append(m)
    ex2["matches"] = out
    return ex2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense80.yaml")
    ap.add_argument("--matches-dir", default="outputs/matches_dense80")
    ap.add_argument("--cache-dir", default="outputs/cache_dense80_final")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--sims", nargs="*", type=float,
                    default=[0.30, 0.35, 0.40, 0.45])
    ap.add_argument("--strat", nargs="*",
                    default=["inlier", "weighted", "reproj", "similarity"])
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]

    from src.datasets.linemod import LinemodDataset

    # (sim, strat) -> per-frame metric lists
    results = {(s, st): [] for s in args.sims for st in args.strat}

    for obj in objects:
        bank_path = template_bank_path(cfg, obj)
        bank = TemplateBank(bank_path)
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = subsample_frames(ds.eval_frames(exclude_refs=True, n_ref=64),
                                  args.max_frames)
        model_pts = ds.model_points(max_points=2000)
        cache = {}
        cp = Path(args.cache_dir) / f"{obj}.jsonl"
        if cp.exists():
            for line in cp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[int(r["frame_id"])] = r

        from src.metrics.pose_metrics import evaluate_pose
        import copy as _copy
        solvers = {}
        for st in args.strat:
            cfg_s = _copy.deepcopy(cfg)
            cfg_s["solver"]["selection"] = st
            solvers[st] = MiniSolver(cfg_s, bank)
        for fr in frames:
            npz = Path(args.matches_dir) / obj / f"{fr.frame_id:06d}.npz"
            if not npz.exists():
                for key in results:
                    results[key].append(0.0)
                continue
            ex = load_extracted_matches(npz)
            for s in args.sims:
                ex_s = filter_matches(ex, s) if s > 0.30 else ex
                for st in args.strat:
                    chosen = solve_with_alt(ex_s, fr.K, solvers[st], None)
                    if chosen is None:
                        results[(s, st)].append(0.0)
                        continue
                    Rm, tm = solvers[st]._to_model_frame(chosen.R, chosen.t)
                    m = evaluate_pose(
                        model_pts, ds.diameter, fr.K, fr.R_gt, fr.t_gt,
                        Rm, tm, symmetric=ds.symmetric,
                        add_threshold_ratio=0.1, proj_threshold_px=5.0,
                        cm_threshold_mm=50.0, deg_threshold=5.0,
                        adds_definition="unidirectional")
                    results[(s, st)].append(m["proj_5px"])
        print(f"[{obj}] done", flush=True)

    print(f"\n{'sim\\strat':<12}" + "".join(f"{st:>12}" for st in args.strat))
    for s in args.sims:
        row = f"{s:<12}"
        for st in args.strat:
            row += f"{100*np.mean(results[(s, st)]):>11.1f}%"
        print(row)


if __name__ == "__main__":
    main()
