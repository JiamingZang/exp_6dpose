"""验证 align_loss vs mask_iou 对 can 30k+invdepth 锚点坏帧的区分度。

exp_30k13 缓存下 can 坏帧（add_01d==0）：RANSAC 择优选到 tz 爆炸假设，
mask IoU 判据漏检（tz 缩放掩码 IoU 仍高）。验证 align_loss（L1+SSIM）
能否区分 best（爆炸）与 GT 位姿。

用法: python scripts/analysis/verify_align_select.py --obj can
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="can")
    ap.add_argument("--n", type=int, default=23)
    args = ap.parse_args()

    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner

    ds = LinemodDataset("data/lm", args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    ref = PoseRefiner(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt",
                      device="cuda", iterations=0)

    recs = [json.loads(l) for l in
            open(f"outputs/exp_30k13/cache/{args.obj}.jsonl")
            if not l.startswith('{"__meta__')]
    failed = [r for r in recs if r.get("m", {}).get("add_01d") == 0.0]
    print(f"{args.obj} failed frames (30k+invdepth): {len(failed)}")

    iou_ok = align_ok = 0
    n = min(len(failed), args.n)
    for r in failed[:args.n]:
        fid = r["frame_id"]
        f = frames[fid]
        npz = np.load(f"outputs/matches13_30k/{args.obj}/{fid:06d}.npz",
                      allow_pickle=True)
        crop, mask = npz["crop"], npz["mask_crop"]
        x0, y0 = int(npz["crop_box"][0]), int(npz["crop_box"][1])
        Kc = f.K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        R_b = np.asarray(r["R"], dtype=np.float64)
        t_b = np.asarray(r["t"], dtype=np.float64)
        la_b = ref.align_loss(crop, mask, Kc, R_b, t_b)
        iou_b = ref.mask_iou(R_b, t_b, Kc, mask)
        la_g = ref.align_loss(crop, mask, Kc, f.R_gt, f.t_gt)
        iou_g = ref.mask_iou(f.R_gt, f.t_gt, Kc, mask)
        iou_ok += iou_g > iou_b
        align_ok += la_g < la_b
    print(f"mask_iou 判对: {iou_ok}/{n}")
    print(f"align_loss 判对: {align_ok}/{n}")


if __name__ == "__main__":
    main()
