"""GT 掩码面积比 tz 修正上限验证（holepuncher 等 D 类）。

对每帧：以 dc2 最终位姿为初始，按"渲染掩码面积 == GT 掩码面积"扫 tz
（±80mm 网格），评估面积信号在掩码完美时的上限收益。

用法: python scripts/verify_gtmask_tz.py --obj holepuncher
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="holepuncher")
    ap.add_argument("--tz-range", type=float, default=80.0)
    ap.add_argument("--tz-step", type=float, default=5.0)
    args = ap.parse_args()

    from src.datasets.linemod import LinemodDataset
    from src.gaussian.pose_refiner import PoseRefiner
    from src.metrics.pose_metrics import add_error, adds_error

    ds = LinemodDataset("data/lm", args.obj)
    frames = {f.frame_id: f for f in ds.frames()}
    model_pts = ds.model_points(max_points=2000)
    syms = [np.asarray(s, dtype=np.float64).reshape(4, 4)
            for s in ds.model_info.get("symmetries_discrete") or []]
    ref = PoseRefiner(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt",
                      device="cuda", iterations=0)

    rows = [json.loads(l) for l in
            open(f"outputs/cache13_dc2/{args.obj}.jsonl")
            if not l.startswith('{"__meta__')]
    # dedup：dc2 帧 refine≈5.7s；refine2 追加帧 ≈7.6s。每帧保留最接近 5.7 的
    best = {}
    for r in rows:
        key = r["frame_id"]
        d = abs(r["timings"].get("refine", 0) - 5.7)
        if key not in best or d < best[key][0]:
            best[key] = (d, r)
    clean = [v[1] for v in best.values()]
    print(f"clean frames: {len(clean)}")

    tzs = np.arange(-args.tz_range, args.tz_range + 1e-9, args.tz_step)
    thr = ds.diameter * 0.1
    ok_base = ok_gt = ok_area = 0
    improved = degraded = 0
    for r in clean:
        fid = r["frame_id"]
        f = frames[fid]
        npz = np.load(f"outputs/matches13_dc2/{args.obj}/{fid:06d}.npz",
                      allow_pickle=True)
        crop, mask = npz["crop"], npz["mask_crop"]
        x0, y0 = int(npz["crop_box"][0]), int(npz["crop_box"][1])
        Kc = f.K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        Kt = torch.tensor(Kc, dtype=torch.float32, device="cuda")
        H, W = crop.shape[:2]
        R0 = np.asarray(r["R"], dtype=np.float64)
        t0 = np.asarray(r["t"], dtype=np.float64)

        # GT 掩码面积（原图级 GT mask_visib 裁剪到 crop 坐标系）
        gt_mask = np.asarray(f.mask_path and __import__("imageio").v2.imread(
            f.mask_path) > 0) if f.mask_path else None
        if gt_mask is None:
            continue
        gm = gt_mask[y0:y0 + H, x0:x0 + W] if gt_mask.ndim == 2 else None
        if gm is None or gm.sum() < 16:
            continue
        A_gt = float(gm.sum())

        # 扫 tz：渲染 alpha 面积匹配 GT 掩码面积
        areas = []
        with torch.no_grad():
            for dz in tzs:
                tt = torch.tensor(t0, dtype=torch.float32, device="cuda")
                tt[2] += dz
                R0t = torch.tensor(R0, dtype=torch.float32, device="cuda")
                _, alpha = ref._render(R0t, tt, Kt, W, H)
                a = alpha[..., 0].detach().cpu().numpy()
                areas.append(float((a > 0.5).sum()))
        areas = np.array(areas)
        dz_best = tzs[int(np.abs(areas - A_gt).argmin())]
        t_new = t0.copy()
        t_new[2] += dz_best

        def _ok(R, t):
            if syms:
                err = adds_error(model_pts, f.R_gt, f.t_gt, R, t)
            else:
                err = add_error(model_pts, R, t, f.R_gt, f.t_gt)
            return err < thr, err

        b_ok, b_err = _ok(R0, t0)
        n_ok, n_err = _ok(R0, t_new)
        ok_base += b_ok
        ok_area += n_ok
        if n_ok and not b_ok:
            improved += 1
        elif b_ok and not n_ok:
            degraded += 1

    print(f"baseline ok: {ok_base}/{len(clean)}")
    print(f"gt-mask area-tz ok: {ok_area}/{len(clean)} "
          f"(improved {improved}, degraded {degraded})")
    print(f"thr = {thr:.1f}mm (diam {ds.diameter:.1f})")


if __name__ == "__main__":
    main()
