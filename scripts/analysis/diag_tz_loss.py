"""tz 损失面诊断：以 GT 位姿为初始，单变量扫 tz ±60mm，
比较 v1（全掩码 L1+SSIM+LPIPS+Dice，1x）与 v2（交集 L1+面积正则，2x）
的损失曲线 argmin 是否落在 GT 处——回答"tz 方向信号是否存在且正确"。

用法: python scripts/analysis/diag_tz_loss.py [--obj holepuncher] [--n 15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root / "src"
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))


def load_loss_fns(refiner, s: int):
    """返回 v1/v2 单步损失函数（复制 pose_refiner 逻辑，不动主代码）。"""
    from torch.nn import functional as F

    def _prep(crop, mask, K):
        gt_np = crop.astype(np.float32) / 255.0
        gt_np = np.where(mask[..., None].astype(bool), gt_np,
                         np.full_like(gt_np, refiner.bg_color))
        gt0 = torch.tensor(gt_np, dtype=torch.float32,
                           device=refiner.device).permute(2, 0, 1)
        msk0 = torch.tensor(mask, dtype=torch.float32,
                            device=refiner.device)
        if s > 1:
            gt_s = F.interpolate(gt0[None], scale_factor=s, mode="bilinear",
                                 align_corners=False)[0]
            msk_s = F.interpolate(msk0[None, None], scale_factor=s,
                                  mode="nearest")[0, 0]
        else:
            gt_s, msk_s = gt0, msk0
        return gt0, msk0, gt_s, msk_s

    def loss_v1(R, t, Kt, gt, msk, W, H):
        composed, alpha = refiner._render(R, t, Kt, W, H, scale=1)
        comp = composed.permute(2, 0, 1)
        a = alpha[..., 0]
        l1 = (torch.abs(comp - gt) * msk).sum() / msk.sum().clamp(min=1)
        ssim_val = refiner.ssim_fn(comp[None], gt[None])
        loss = l1 + refiner.lambda_ssim * (1.0 - ssim_val)
        if refiner.lpips is not None and refiner.lambda_lpips > 0:
            lp = refiner.lpips(comp[None] * 2 - 1, gt[None] * 2 - 1).mean()
            loss = loss + refiner.lambda_lpips * lp
        dice = (2 * (msk * a).sum() / (msk.sum() + a.sum() + 1e-6))
        return loss - refiner.lambda_dice * dice, dice

    def loss_v2(R, t, Kt, gt_s, msk_s, W, H, a_msk):
        composed, alpha = refiner._render(R, t, Kt, W, H, scale=s)
        comp = composed.permute(2, 0, 1)
        a = alpha[..., 0]
        ov = (a > 0.5) & (msk_s > 0.5)
        if ov.sum() >= 16:
            l1 = (torch.abs(comp - gt_s) * ov).sum() / ov.sum()
        else:
            l1 = (torch.abs(comp - gt_s) * msk_s).sum() / msk_s.sum().clamp(min=1)
        ssim_val = refiner.ssim_fn(comp[None], gt_s[None])
        loss = l1 + refiner.lambda_ssim * (1.0 - ssim_val)
        if refiner.lpips is not None and refiner.lambda_lpips > 0:
            lp = refiner.lpips(comp[None] * 2 - 1, gt_s[None] * 2 - 1).mean()
            loss = loss + refiner.lambda_lpips * lp
        dice = (2 * (msk_s * a).sum() / (msk_s.sum() + a.sum() + 1e-6))
        loss = loss - refiner.lambda_dice * dice
        if (refiner.lambda_area > 0 and refiner.area_gate_dice > 0
                and float(dice.detach()) >= refiner.area_gate_dice):
            ratio = (a.sum() + 1e-6) / (a_msk + 1e-6)
            loss = loss + refiner.lambda_area * torch.log(ratio) ** 2
        return loss, dice

    return _prep, loss_v1, loss_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="holepuncher")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    from datasets.linemod import LinemodDataset
    from gaussian.pose_refiner import PoseRefiner

    root = Path("data/lm")
    ds = LinemodDataset(root, args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    cache_path = Path(f"outputs/cache13_dc2/{args.obj}.jsonl")
    rows = [json.loads(l) for l in cache_path.read_text().splitlines()
            if not l.startswith('{"__meta__')]
    rows.sort(key=lambda r: -r["m"]["trans_err"])  # tz 错帧优先
    ids = [r["frame_id"] for r in rows[:args.n]]

    ckpt = args.ckpt or f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt"
    if not Path(ckpt).exists():
        import glob
        cand = glob.glob(f"outputs/templates/{args.obj}_3dgs*.pt")
        if not cand:
            print("no ckpt found"); return
        ckpt = cand[0]
    print(f"ckpt: {ckpt}")

    ref = PoseRefiner(ckpt, device="cuda", iterations=0,
                      supersample=2, stage1_iters=60,
                      lambda_area=2.0, area_gate_dice=0.6)
    _prep, loss_v1, loss_v2 = load_loss_fns(ref, s=2)

    print(f"{'fid':>5} {'trans':>7} {'v1_argmin':>10} {'v2_argmin':>10} "
          f"{'v1@0':>8} {'v2@0':>8} {'dice':>6}")
    for fid in ids:
        f = frames[fid]
        npz = np.load(f"outputs/matches13_dc2/{args.obj}/{fid:06d}.npz",
                      allow_pickle=True)
        crop, mask = npz["crop"], npz["mask_crop"]
        x0, y0 = int(npz["crop_box"][0]), int(npz["crop_box"][1])
        Kc = f.K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        gt0, msk0, gt_s, msk_s = _prep(crop, mask, Kc)
        a_msk = msk_s.sum()
        R0t = torch.tensor(f.R_gt, dtype=torch.float32, device="cuda")
        t0t = torch.tensor(f.t_gt, dtype=torch.float32, device="cuda")
        Kt = torch.tensor(Kc, dtype=torch.float32, device="cuda")
        H, W = crop.shape[:2]
        deltas = np.arange(-60, 61, 3.0)
        l1c, l2c = [], []
        d1, d2 = [], []
        with torch.no_grad():
            for d in deltas:
                tt = t0t.clone()
                tt[2] += d
                lv1, dv1 = loss_v1(R0t, tt, Kt, gt0, msk0, W, H)
                lv2, dv2 = loss_v2(R0t, tt, Kt, gt_s, msk_s, W, H, a_msk)
                l1c.append(float(lv1)); l2c.append(float(lv2))
                d1.append(float(dv1)); d2.append(float(dv2))
        l1c, l2c = np.array(l1c), np.array(l2c)
        i1, i2 = int(l1c.argmin()), int(l2c.argmin())
        trans = [r["m"]["trans_err"] for r in rows
                 if r["frame_id"] == fid][0]
        print(f"{fid:>5} {trans:>7.1f} {deltas[i1]:>10.1f} "
              f"{deltas[i2]:>10.1f} {l1c[20]:>8.4f} {l2c[20]:>8.4f} "
              f"{np.mean(d1):>6.2f}")


if __name__ == "__main__":
    main()
