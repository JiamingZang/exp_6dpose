#!/usr/bin/env python3
"""主实验：LineMod 13 物体全量评测。GPU 机器运行。

用法：
    python scripts/run_linemod.py --config configs/default.yaml
    python scripts/run_linemod.py --objects ape cat --max-frames 100  # 冒烟

输出：outputs/linemod_main.json + 终端表格
    每物体 ADD(S)@0.1d / Proj@5pix / 5cm5° 与 13 物体均值
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.pipeline import evaluate_object, template_bank_path


def run_eval(cfg, objects, device, max_frames=0, verbose=True,
             collect_frames=False, bop_rows=None, cache_dir=None,
             matches_dir=None, frame_range=None):
    """跑一组物体的评测，返回 {obj: summary}。run_ablation.py 复用。

    collect_frames=True 时额外返回 {obj: 帧级明细}（aggregated 导出用）。
    bop_rows 传入 list 时收集 bop19 提交行（--bop-csv 导出用）。
    cache_dir 给定时按物体落帧级 jsonl 缓存，断点续跑（见 evaluate_object）。
    matches_dir 给定时跳过 MASt3R，从阶段 2 产物直接求解。
    """
    results, frames_by_obj = {}, {}
    for obj in objects:
        bank_path = template_bank_path(cfg, obj)
        if not bank_path.exists():
            raise FileNotFoundError(
                f"模板库不存在: {bank_path}\n先运行 scripts/onboard_object.py "
                f"--objects {obj}（当前配置的渲染器/几何/模板数组合）")
        cache_path = None
        if cache_dir is not None:
            cache_path = str(Path(cache_dir) / f"{obj}.jsonl")
        mdir = None
        if matches_dir is not None:
            mdir = str(Path(matches_dir) / obj)
        summary, per_frame, avg_t = evaluate_object(cfg, obj, device=device,
                                                    max_frames=max_frames,
                                                    verbose=verbose,
                                                    bop_rows=bop_rows,
                                                    cache_path=cache_path,
                                                    matches_dir=mdir,
                                                    frame_range=frame_range)
        summary["timings"] = avg_t
        results[obj] = summary
        frames_by_obj[obj] = per_frame
        if verbose:
            print(f"[{obj}] ADD(S)@0.1d={summary['add_01d']:.2f}% "
                  f"Proj@5pix={summary['proj_5px']:.2f}% "
                  f"5cm5°={summary['cm_deg']:.2f}%")
            if "bop" in summary:
                b = summary["bop"]
                print(f"    BOP(无VSD): AR_MSSD={b['ar_mssd']:.2f}% "
                      f"AR_MSPD={b['ar_mspd']:.2f}% AR2={b['ar_bop']:.2f}%")
            # metrics.topk_best 非空时打印各档位。K>1 都是 GT 择优上界，
            # 标出来免得终端数字被直接抄进结果表（只有 top1 是端到端）
            from src.metrics.legacy_format import topk_from_key
            for tier, tm in summary.get("topk_best", {}).items():
                if tier == "total_samples":
                    continue
                tag = "[端到端]" if topk_from_key(tier) == 1 else "[oracle 上界]"
                print(f"    {tier}{tag}: ADD={tm['add_success_rate']:.2f}% "
                      f"Proj={tm['proj_success_rate']:.2f}%")
    if collect_frames:
        return results, frames_by_obj
    return results


def print_table(results):
    print(f"\n{'物体':<14}{'ADD(S)@0.1d':>14}{'Proj@5pix':>12}{'5cm5°':>10}{'帧数':>8}")
    for obj, s in results.items():
        print(f"{obj:<14}{s['add_01d']:>13.2f}%{s['proj_5px']:>11.2f}%"
              f"{s['cm_deg']:>9.2f}%{s['n']:>8}")
    mean = {k: float(np.mean([s[k] for s in results.values()]))
            for k in ("add_01d", "proj_5px", "cm_deg")}
    print(f"{'MEAN':<14}{mean['add_01d']:>13.2f}%{mean['proj_5px']:>11.2f}%"
          f"{mean['cm_deg']:>9.2f}%")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0,
                    help=">0 时每物体只评前 N 帧（冒烟测试）")
    ap.add_argument("--first", type=int, default=0,
                    help="帧列表起始下标（全量并行分片用）")
    ap.add_argument("--last", type=int, default=0,
                    help="帧列表结束下标（0 = 到末尾）")
    ap.add_argument("--cache-dir", default=None,
                    help="帧级结果缓存目录（每物体一个 jsonl，中断后重跑"
                         "自动跳过已完成帧；默认关闭，传 outputs/cache 开启）")
    ap.add_argument("--matches-dir", default=None,
                    help="阶段 2 产物目录（scripts/extract_matches.py 生成，"
                         "每物体一个子目录）。给定时跳过 MASt3R 匹配，直接从"
                         "对应求解（调 PnP/择优参数无需重跑匹配）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aggregated-out", default=None,
                    help="额外导出历史 MyPose aggregated_metrics 兼容 JSON"
                         "（逐物体 + overall 的 add/proj 成功率，可与 "
                         "_prior_code 的真实结果直接对比）")
    ap.add_argument("--bop-csv", default=None,
                    help="额外导出 bop19 提交格式 CSV（可交官方 bop_toolkit "
                         "复算 MSSD/MSPD，与 FoundPose 等 BOP 口径方法对账）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["runtime"].get("device", "cuda")
    objects = args.objects or cfg["dataset"]["objects"]

    # P2-5 复审：CPU 环境跑 fastsam/sam 会立刻在权重下载/CUDA 分配上炸，
    # 用户看到的堆栈却是 torch/ultralytics 内部——这里提前拦一下，直接
    # 告诉用户该切 gt_bbox / gt_mask（本地/无 GPU 跑通全链条的正解）。
    seg = cfg.get("detection", {}).get("segmenter", "")
    if device == "cpu" and seg in ("fastsam", "sam"):
        print(f"[run_linemod] 检测到 runtime.device=cpu 且 "
              f"detection.segmenter={seg!r}。CPU 环境无法加载该分割器；"
              f"请把 segmenter 显式改成 gt_bbox 或 gt_mask（消融配置见 "
              f"configs/ablations/10_segmenter.yaml），或切到 GPU 机器。",
              file=sys.stderr)
        sys.exit(2)

    bop_rows = [] if args.bop_csv else None
    # 阶段 3 路径：给了 --matches-dir 就从落盘对应求解（跳过 MASt3R），
    # 否则完整跑阶段 2+3
    mdir = None
    if args.matches_dir:
        mdir = str(Path(args.matches_dir))
    frame_range = (args.first, args.last) if args.last > 0 else None
    results, frames_by_obj = run_eval(cfg, objects, device,
                                      max_frames=args.max_frames,
                                      collect_frames=True,
                                      bop_rows=bop_rows,
                                      cache_dir=args.cache_dir,
                                      matches_dir=mdir,
                                      frame_range=frame_range)
    mean = print_table(results)

    if args.bop_csv:
        from src.metrics.bop_metrics import save_bop_csv
        bop_path = Path(args.bop_csv)
        bop_path.parent.mkdir(parents=True, exist_ok=True)
        save_bop_csv(bop_path, bop_rows)
        print(f"bop19 提交 CSV 已写入 {bop_path}（{len(bop_rows)} 行，"
              f"仅成功帧；失败帧无位姿可提交，官方评分会按未命中计）")

    out = Path(args.out or Path(cfg["runtime"]["output_dir"]) / "linemod_main.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_object": results, "mean": mean},
                              indent=2, ensure_ascii=False))
    print(f"\n结果已写入 {out}")

    if args.aggregated_out:
        # 导出历史 aggregated_metrics_all_objects40.json 兼容格式（口径
        # 逐项对齐，见 src/metrics/legacy_format.py）；metrics.topk_best
        # 开启时同时导出 top1/3/5 消融格式
        from src.metrics.legacy_format import (aggregate_all_objects,
                                               aggregate_topk_all_objects,
                                               per_object_from_frames)
        agg_out = Path(args.aggregated_out)
        agg_out.parent.mkdir(parents=True, exist_ok=True)
        per_object = {obj: per_object_from_frames(frames)
                      for obj, frames in frames_by_obj.items()}
        agg_out.write_text(json.dumps(aggregate_all_objects(per_object),
                                      indent=2, ensure_ascii=False))
        print(f"aggregated 兼容结果已写入 {agg_out}")
        ks = [int(k) for k in (cfg["metrics"].get("topk_best") or ())]
        if ks and all("topk_best" in s for s in results.values()):
            topk_per_object = {obj: s["topk_best"]
                               for obj, s in results.items()}
            topk_out = agg_out.with_name(agg_out.stem + "_topk_best.json")
            topk_out.write_text(json.dumps(
                aggregate_topk_all_objects(topk_per_object, ks=ks),
                indent=2, ensure_ascii=False))
            print(f"topK best 兼容结果已写入 {topk_out}")


if __name__ == "__main__":
    main()
