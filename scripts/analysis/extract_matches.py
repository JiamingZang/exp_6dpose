#!/usr/bin/env python3
"""阶段 2（粗匹配）：逐帧定位 + MASt3R Top-K 稠密对应，产物落盘。

用法：
    python scripts/analysis/extract_matches.py --objects ape
    python scripts/analysis/extract_matches.py                 # 全部 13 物体

产物：<matches_dir>/<obj>/<frame_id:06d>.npz（每帧自包含：对应点、
裁剪图/掩码、裁剪框与缩放，见 pipeline.save_extracted_matches）。
之后调 PnP/择优参数只需重跑阶段 3（run_linemod.py --matches-dir），
无需重跑最贵的 MASt3R 匹配。

GPU-only（FastSAM + DINOv2 + MASt3R）。
"""
import argparse
import sys
from pathlib import Path

def _repo_root() -> Path:
    for root in Path(__file__).resolve().parents:
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
    raise RuntimeError("Cannot locate repository root")

sys.path.insert(0, str(_repo_root()))

import cv2

from src.config import load_config
from src.pipeline import (PoseEstimator, TemplateBank, evaluate_object,
                          save_extracted_matches, subsample_frames,
                          template_bank_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/current/default.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--first", type=int, default=0,
                    help="帧列表起始下标（全量并行分片用）")
    ap.add_argument("--last", type=int, default=0,
                    help="帧列表结束下标（0 = 到末尾）")
    ap.add_argument("--matches-dir", default=None,
                    help="输出目录（默认 outputs/matches）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["runtime"].get("device", "cuda")
    objects = args.objects or cfg["dataset"]["objects"]
    mdir = Path(args.matches_dir or
                Path(cfg["runtime"].get("output_dir", "outputs")) / "matches")

    for obj in objects:
        bank = TemplateBank(template_bank_path(cfg, obj))
        # 裁剪填色必须与 bank 模板渲染背景一致（浅色物体黑背景、深色物体
        # 白背景）。bank 记录了自身 bg_color 时以 bank 为准并校验配置，
        # 不一致直接报错——静默域不匹配会让匹配质量系统性崩坏（ape 事故）。
        cfg_bg = float(cfg["onboard"].get("bg_color", 1.0))
        if bank.bg_color is not None and abs(bank.bg_color - cfg_bg) > 1e-6:
            raise ValueError(
                f"[{obj}] bank 背景色 {bank.bg_color} ≠ 配置 onboard.bg_color "
                f"{cfg_bg}：模板黑背景+裁剪白填充（或反之）属于域不匹配，"
                f"匹配结果不可信。请用与 bank 一致的配置（如 "
                f"configs/current/dense80_depth_bg0.yaml 或 dense80_depth_w1.yaml）"
                f"重新运行，或先重建 bank。")
        # 渲染验证（定位候选消歧）需要 3DGS 参数 ckpt；缺 .pt 时内部跳过
        refiner_ckpt = str(template_bank_path(cfg, obj).with_suffix(".pt"))
        est = PoseEstimator(cfg, bank, device=device,
                            refiner_ckpt=refiner_ckpt)
        out_dir = mdir / obj
        out_dir.mkdir(parents=True, exist_ok=True)

        from src.datasets.linemod import LinemodDataset
        ds = LinemodDataset(cfg["dataset"]["root"], obj,
                            models_dir=cfg["dataset"].get("models_dir",
                                                          "models_eval"),
                            splits_dir=cfg["dataset"].get("splits_dir"))
        frames = ds.eval_frames(
            exclude_refs=True,
            n_ref=int(cfg["onboard"].get("n_ref_views", 64)))
        frames = subsample_frames(frames, args.max_frames)
        if args.last > 0:
            frames = frames[args.first:args.last]

        done = n_fail = 0
        for fi, fr in enumerate(frames):
            npz = out_dir / f"{fr.frame_id:06d}.npz"
            if npz.exists():
                done += 1
                continue
            img = cv2.cvtColor(cv2.imread(str(fr.rgb_path)),
                               cv2.COLOR_BGR2RGB)
            gt_mask = None
            if fr.mask_path is not None:
                gt_mask = cv2.imread(str(fr.mask_path),
                                     cv2.IMREAD_GRAYSCALE) > 0
            ex = est.extract_matches(img, fr.K, gt_bbox=fr.bbox_visib,
                                     gt_mask=gt_mask, frame_id=fr.frame_id)
            if ex is None:
                n_fail += 1
                # 失败帧也落盘（空 matches），阶段 3 据此判定失败
                import numpy as np
                np.savez_compressed(
                    npz, pix_q=np.zeros((0, 2), np.uint16),
                    pix_t=np.zeros((0, 2), np.uint16),
                    sims=np.zeros((0,), np.float16),
                    seg=np.zeros((1,), np.int32),
                    template_idx=np.zeros((0,), np.int16),
                    score=np.zeros((0,), np.float32),
                    crop=np.zeros((8, 8, 3), np.uint8),
                    mask_crop=np.zeros((8, 8), np.uint8),
                    crop_box=np.zeros((4,), np.int32),
                    sxy=np.zeros((2,), np.float32),
                    s_leg=np.zeros((2,), np.float32),
                    loc_score=np.float32(0.0))
                continue
            save_extracted_matches(npz, ex)
            done += 1
            if verbose(fi, frames):
                print(f"  [{obj}] {fi+1}/{len(frames)} "
                      f"tpl={len(ex['matches'])}")
        print(f"[extract:{obj}] {done}/{len(frames)} 帧完成"
              f"（定位失败 {n_fail}）→ {out_dir}")


def verbose(fi, frames):
    return (fi + 1) % max(1, len(frames) // 20) == 0


if __name__ == "__main__":
    main()
