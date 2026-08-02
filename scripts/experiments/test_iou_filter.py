#!/usr/bin/env python3
"""爆炸位姿过滤验证：3DGS 渲染 mask 与 FastSAM mask 的 IoU 能否区分好坏候选。

对每帧 PnP 的 top-K 候选（按分数取前 12）渲染 alpha，与查询掩码算 IoU：
  - 爆炸帧（trans>500mm）的候选 IoU 应该普遍很低（渲染对不上）
  - 成功帧的候选 IoU 应该高
统计 IoU 阈值在 0.3/0.4/0.5 下的拦爆率与误伤率（拦掉的好候选比例）。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.pipeline import (TemplateBank, load_extracted_matches,
                          subsample_frames, template_bank_path)
from scripts.test_fast_mode import MiniSolver


def render_iou(verifier, ex, K_query, R, t):
    """渲染 3DGS 得到 alpha，与查询掩码算 IoU（裁剪坐标系）。"""
    x0, y0, _, _ = ex["crop_box_used"]
    K_crop = K_query.copy()
    K_crop[0, 2] -= x0
    K_crop[1, 2] -= y0
    import torch
    Rr = torch.tensor(R, dtype=torch.float32, device=verifier.device)
    tr = torch.tensor(t, dtype=torch.float32, device=verifier.device)
    Kt = torch.tensor(K_crop, dtype=torch.float32, device=verifier.device)
    _, alpha = verifier._render(Rr, tr, Kt,
                                ex["crop"].shape[1], ex["crop"].shape[0])
    a = (alpha[..., 0].detach().cpu().numpy() > 0.5)
    m = ex["mask_crop"]
    inter = np.logical_and(a, m).sum()
    union = np.logical_or(a, m).sum()
    return inter / max(union, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense80.yaml")
    ap.add_argument("--matches-dir", default="outputs/matches_dense80")
    ap.add_argument("--cache-dir", default="outputs/cache_dense80_final")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--objects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    objects = args.objects or cfg["dataset"]["objects"]

    out_fh = open("outputs/logs/iou_data.jsonl", "w")

    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner

    ious_ok, ious_close, ious_boom, ious_far = [], [], [], []
    for obj in objects:
        bank_path = template_bank_path(cfg, obj)
        bank = TemplateBank(bank_path)
        solver = MiniSolver(cfg, bank)
        verifier = PoseRefiner(str(bank_path.with_suffix(".pt")),
                               device=args.device, iterations=0)
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = subsample_frames(ds.eval_frames(exclude_refs=True, n_ref=64),
                                  args.max_frames)
        cache = {}
        cp = Path(args.cache_dir) / f"{obj}.jsonl"
        if cp.exists():
            for line in cp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[int(r["frame_id"])] = r

        for fr in frames:
            npz = Path(args.matches_dir) / obj / f"{fr.frame_id:06d}.npz"
            if not npz.exists():
                continue
            ex = load_extracted_matches(npz)
            solved = solver.solve_pnp(ex, fr.K)
            if solved is None:
                continue
            best, results = solved
            # 前 5 个候选的 IoU（含 best）
            cands = [best] + [r for r in results if r is not best][:4]
            ious = [render_iou(verifier, ex, fr.K, r.R, r.t) for r in cands]
            ref = cache.get(fr.frame_id, {}).get("m", {})
            proj = ref.get("proj", np.inf)
            trans = ref.get("trans_err", 0)
            rec = (obj, fr.frame_id, max(ious))
            if trans > 500:
                ious_boom.append(rec)
            elif proj <= 5:
                ious_ok.append(rec)
            elif proj <= 30:
                ious_close.append(rec)
            else:
                ious_far.append(rec)
            out_fh.write(json.dumps(rec) + "\n")
        print(f"[{obj}] done", flush=True)

    out_fh.close()

    def stats(name, arr):
        a = np.array([x[2] for x in arr])
        if len(a) == 0:
            print(f"{name:8s} n=0")
            return
        print(f"{name:8s} n={len(a):3d}  mean_iou={a.mean():.2f}  "
              f"<0.3:{np.mean(a < 0.3)*100:4.0f}%  <0.5:{np.mean(a < 0.5)*100:4.0f}%")

    print("\n=== 候选最佳 IoU 分布 ===")
    stats("成功", ious_ok)
    stats("接近", ious_close)
    stats("远错", ious_far)
    stats("爆炸", ious_boom)
    # 拦爆率：阈值下爆炸帧被拦（max_iou < thr）
    print("\n=== 阈值效果（用 max_iou 判定整帧） ===")
    for thr in (0.2, 0.3, 0.4, 0.5):
        keep_boom = sum(1 for _, _, i in ious_boom if i < thr)
        miss_ok = sum(1 for _, _, i in ious_ok if i < thr)
        print(f"thr={thr}: 拦爆 {keep_boom}/{len(ious_boom)}"
              f"  误伤成功帧 {miss_ok}/{len(ious_ok)}")


if __name__ == "__main__":
    main()
