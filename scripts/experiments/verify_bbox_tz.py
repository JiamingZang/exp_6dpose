#!/usr/bin/env python3
"""离线验证 bbox 尺度 tz 修正（固定 R，z 按投影宽度/bbox 宽度比缩放）：
对 cache13_dc2 全部帧做后处理，统计 ADD 变化 + OK 帧保护率。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.datasets.linemod import LinemodDataset
from src.datasets.ply_io import load_ply
from src.config import load_config
from src.pipeline import TemplateBank, subsample_frames, template_bank_path

OBJS = ["ape", "benchvise", "cam", "can", "cat", "driller", "duck",
        "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"]


def main():
    import json as j
    mi = j.load(open("data/lm/models_eval/models_info.json"))
    cfg = load_config("configs/dense80_batch8_bg0.yaml")
    for obj in OBJS:
        bank = TemplateBank(template_bank_path(cfg, obj))
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = subsample_frames(ds.eval_frames(exclude_refs=True, n_ref=64),
                                  120)
        fm = {f.frame_id: f for f in frames}
        cache_path = Path(f"outputs/cache13_dc2/{obj}.jsonl")
        if not cache_path.exists():
            continue
        cache = [j.loads(l) for l in open(cache_path)
                 if not l.startswith('{"__meta__"')]
        verts, _, _ = load_ply(ds.model_path)
        verts = verts * bank.scale
        diam = mi[str(ds.obj_id)]["diameter"]
        thr = 0.1 * diam
        ok0 = ok1 = kept = 0
        n_tz_fixed = 0
        for c in cache:
            fr = fm.get(c["frame_id"])
            if fr is None:
                continue
            K = fr.K
            R = np.array(c["R"]); t = np.array(c["t"])
            pc = (R @ verts.T).T + t
            u = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
            v = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
            okz = pc[:, 2] > 0
            if okz.sum() < 10:
                continue
            w_est = u[okz].max() - u[okz].min()
            h_est = v[okz].max() - v[okz].min()
            x, y, w, h = fr.bbox_visib
            z_new = t[2] * (w_est / w)
            # 保护：|Δz| 限制 ±10%，且修正后投影宽度更接近 bbox
            if abs(z_new - t[2]) / t[2] < 0.10:
                t2 = t.copy(); t2[2] = z_new
            else:
                t2 = t
            Rg, tg = fr.R_gt, fr.t_gt
            e0 = np.linalg.norm((R @ verts.T).T + t - ((Rg @ verts.T).T + tg),
                                axis=1).mean()
            e1 = np.linalg.norm((R @ verts.T).T + t2 - ((Rg @ verts.T).T + tg),
                                axis=1).mean()
            a0, a1 = e0 < thr, e1 < thr
            ok0 += a0; ok1 += a1
            if a0:
                kept += a1
            if not a0 and a1:
                n_tz_fixed += 1
        n = len(cache)
        print(f"{obj:12s} ADD {ok0/n*100:5.1f} → {ok1/n*100:5.1f}  "
              f"(修回 {n_tz_fixed:3d}，原OK保持 {kept}/{ok0})")


if __name__ == "__main__":
    main()
