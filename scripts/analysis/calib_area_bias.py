"""面积校准系数统计（参考帧上）：
k1 = A_3DGS_render / A_GT_mask（渲染器几何偏差）
k2 = A_FastSAM / A_GT_mask（分割器偏差，--mode fastsam）

面积比 tz 修正的目标应为 A_render_target = A_mask × (k1/k2)。

用法:
  python scripts/analysis/calib_area_bias.py --obj holepuncher          # k1
  python scripts/analysis/calib_area_bias.py --obj holepuncher --mode fastsam  # k2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))


def stat_k1(args):
    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner

    ds = LinemodDataset("data/lm", args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    ref_ids = sorted(ds.reference_frame_ids(args.n))
    ref = PoseRefiner(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt",
                      device="cuda", iterations=0)

    import imageio.v2 as iio
    k1s, dice_s = [], []
    for fid in ref_ids:
        f = frames[fid]
        if f.mask_path is None:
            continue
        gm = iio.imread(f.mask_path) > 0
        A_gt = float(gm.sum())
        if A_gt < 100:
            continue
        Kt = torch.tensor(f.K, dtype=torch.float32, device="cuda")
        H, W = gm.shape[:2]
        R0t = torch.tensor(f.R_gt, dtype=torch.float32, device="cuda")
        t0t = torch.tensor(f.t_gt, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            _, alpha = ref._render(R0t, t0t, Kt, W, H)
        a = (alpha[..., 0].detach().cpu().numpy() > 0.5)
        A_r = float(a.sum())
        inter = (a & gm).sum()
        union = (a | gm).sum()
        k1s.append(A_r / A_gt)
        dice_s.append(2 * inter / max(union, 1))
    k1s = np.array(k1s)
    dice_s = np.array(dice_s)
    print(f"{args.obj} k1: n={len(k1s)} med {np.median(k1s):.4f} "
          f"mean {k1s.mean():.4f} std {k1s.std():.4f}")
    clean = k1s[dice_s >= 0.7]
    if len(clean):
        print(f"  clean(dice>=0.7) k1: med {np.median(clean):.4f} "
              f"std {clean.std():.4f} n={len(clean)}")


def stat_k2(args):
    from src.datasets.linemod import LinemodDataset
    from src.detection.localize import FastSamSegmenter

    ds = LinemodDataset("data/lm", args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    ref_ids = sorted(ds.reference_frame_ids(args.n))
    seg = FastSamSegmenter({}, device="cuda")

    import imageio.v2 as iio
    k2s, ious = [], []
    for fid in ref_ids:
        f = frames[fid]
        if f.mask_path is None:
            continue
        img = iio.imread(f.rgb_path)
        gm = iio.imread(f.mask_path) > 0
        A_gt = float(gm.sum())
        if A_gt < 100:
            continue
        masks = seg.generate(img)
        best_iou, best_area = 0.0, 0.0
        for m in masks:
            s = m["segmentation"]
            inter = (s & gm).sum()
            union = (s | gm).sum()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_area = float(s.sum())
        if best_iou > 0.3:
            k2s.append(best_area / A_gt)
            ious.append(best_iou)
    k2s = np.array(k2s)
    print(f"{args.obj} k2: n={len(k2s)} med {np.median(k2s):.4f} "
          f"std {k2s.std():.4f} best_iou med {np.median(ious):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="holepuncher")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--mode", default="k1", choices=["k1", "k2"])
    args = ap.parse_args()
    if args.mode == "k2":
        stat_k2(args)
    else:
        stat_k1(args)


if __name__ == "__main__":
    main()
