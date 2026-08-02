#!/usr/bin/env python3
"""从历史管线的 top40 候选 JSON 做零 GPU 候选分析。

读 `top40_add_proj_results_<obj>.json`（历史管线输出，每样本含 40 个候选的
add/proj 与位姿 + gt_pose；出处与格式细节见 VERIFICATION.md §8.10），
纯 numpy 产出五张表：

1. top1 交叉验证 —— 候选按模板相似度降序存，故 `top5_details[0]` 就是
   端到端 top1。重算它并与 `aggregated_metrics_top1_top3_top5.json` 的
   49.49% 对账。**对不上就说明 top1 与 top40 两份数字来自两次设置不同的
   run，K 曲线不可同表。**
2. K 曲线 —— 前 K 个候选里的 best。K>1 全部是 **GT 择优 oracle 上界**。
3. 秩分布 —— 第一个 ADD 成功候选出现在第几名。相似度择优器的直接诊断：
   rank1 命中率就是 top1 精度，长尾说明正确位姿在池里但排名靠后。
4. 逐物体缺口表 —— top1 / top40-oracle / 缺口。用来把总缺口拆成
   「检索缺口」（oracle 高、缺口大）与「几何天花板」（oracle 本身就低）。
5. 误差三分解 —— 旋转测地角 / 平移模长 / |Δz|，对 top1 位姿与 oracle-best
   位姿各算一遍：若某物体 oracle-best 仍失败且 |Δz|/|Δt| 接近 1，
   说明它的天花板是深度/尺度问题，不是旋转问题。

⚠ **内点判据无法在这里重算**：历史 JSON 未存 RANSAC 内点数与模板相似度
分数（见 VERIFICATION.md §8.10）。要拿「内点数/内点比/残差择优」的数字
必须重跑一次 PnP（需要 2D-3D 对应，即需要重跑 MASt3R 前向）。本脚本给出
的是**不需要重跑就能拿到的那一半**。

用法：
    python scripts/analyze_prior_candidates.py --src-dir <存放 top40 json 的目录>
    python scripts/analyze_prior_candidates.py --src-dir ... --out results/prior/candidate_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.pose_utils import rotation_angle_deg

# K 曲线的采样点。1 是端到端，其余全是 oracle 上界
K_GRID = (1, 2, 3, 5, 10, 20, 40)

# 历史管线 PnP 失败时写入的占位位姿：单位旋转 + 零平移（VERIFICATION.md §8.10）。
# 这类候选必须排除在误差分解之外，否则 30° / 800mm 的假误差会污染统计。
_DUMMY_T = np.zeros(3)


def _is_dummy(pose: dict) -> bool:
    """占位位姿判定：平移严格为零向量（真实位姿的 t_z 恒 >0，不会命中）。"""
    return np.allclose(np.asarray(pose["t_mm"], dtype=float), _DUMMY_T)


def _pose_errors(pose: dict, gt: dict) -> Optional[Dict[str, float]]:
    """单个候选 vs GT 的三分解误差；占位位姿返回 None。"""
    if _is_dummy(pose):
        return None
    R = np.asarray(pose["R"], dtype=float)
    t = np.asarray(pose["t_mm"], dtype=float)
    R_gt = np.asarray(gt["gt_R"], dtype=float)
    t_gt = np.asarray(gt["gt_t_mm"], dtype=float)
    dt = t - t_gt
    return {
        "rot_deg": rotation_angle_deg(R_gt, R),
        "trans_mm": float(np.linalg.norm(dt)),
        "dz_mm": float(abs(dt[2])),
    }


def _summarize_errors(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """一组三分解误差的中位数/均值 + Δz 主导度。

    `dz_share` = median(|Δz|) / median(|Δt|)：接近 1 说明平移误差几乎全在
    光轴方向（深度/尺度问题），接近 0.58 是各向同性的期望值。
    """
    if not rows:
        return {"n": 0}
    rot = np.array([r["rot_deg"] for r in rows])
    trans = np.array([r["trans_mm"] for r in rows])
    dz = np.array([r["dz_mm"] for r in rows])
    med_trans = float(np.median(trans))
    return {
        "n": len(rows),
        "rot_deg_median": float(np.median(rot)),
        "rot_deg_mean": float(rot.mean()),
        "trans_mm_median": med_trans,
        "trans_mm_mean": float(trans.mean()),
        "dz_mm_median": float(np.median(dz)),
        "dz_share": float(np.median(dz) / med_trans) if med_trans > 0 else 0.0,
    }


def analyze_object(samples: List[dict]) -> dict:
    """单物体的全部五张表（`samples` = 该 JSON 的 add_results 列表）。"""
    n = len(samples)
    n_cand = [len(s.get("top5_details", [])) for s in samples]

    # --- 表 1/2: top1 与 K 曲线 ---
    # K 曲线取「前 K 个候选中 add 最小者」，与历史 argmin(GT ADD) 同口径。
    k_success = {k: 0 for k in K_GRID}
    k_add_sum = {k: 0.0 for k in K_GRID}
    k_add_n = {k: 0 for k in K_GRID}
    # --- 表 3: 第一个 ADD 成功候选的秩（1-based），0 = 全 40 个都失败 ---
    first_hit_rank: List[int] = []
    # --- 表 5: top1 位姿与 oracle-best 位姿的三分解 ---
    err_top1: List[Dict[str, float]] = []
    err_oracle: List[Dict[str, float]] = []
    err_oracle_failed: List[Dict[str, float]] = []

    for s in samples:
        det = s.get("top5_details", [])
        if not det:
            continue
        adds = np.array([d["add"] for d in det], dtype=float)
        succ = [bool(d["add_success"]) for d in det]

        for k in K_GRID:
            win = adds[:k]
            if win.size == 0:
                continue
            j = int(np.argmin(win))
            k_success[k] += int(succ[j])
            if np.isfinite(win[j]):
                k_add_sum[k] += float(win[j])
                k_add_n[k] += 1

        hits = [i for i, ok in enumerate(succ) if ok]
        first_hit_rank.append(hits[0] + 1 if hits else 0)

        gt = s.get("gt_pose")
        if gt is None:
            continue
        e1 = _pose_errors(det[0]["pose"], gt)
        if e1 is not None:
            err_top1.append(e1)
        best = int(np.argmin(adds))
        eo = _pose_errors(det[best]["pose"], gt)
        if eo is not None:
            err_oracle.append(eo)
            # oracle 都挑不中 → 几何天花板样本，单独统计
            if not succ[best]:
                err_oracle_failed.append(eo)

    def _rate(c: int) -> float:
        return round(100.0 * c / n, 4) if n else 0.0

    ranks = np.array(first_hit_rank) if first_hit_rank else np.zeros(0, dtype=int)
    n_never = int((ranks == 0).sum())
    hist = {}
    for lo, hi, name in ((1, 1, "rank1"), (2, 3, "rank2_3"), (4, 5, "rank4_5"),
                         (6, 10, "rank6_10"), (11, 40, "rank11_40")):
        hist[name] = int(((ranks >= lo) & (ranks <= hi)).sum())
    hist["never"] = n_never

    return {
        "total_samples": n,
        "n_candidates_min": int(min(n_cand)) if n_cand else 0,
        "n_candidates_max": int(max(n_cand)) if n_cand else 0,
        "k_curve": {
            f"top{k}": {
                "add_success_rate": _rate(k_success[k]),
                "avg_add_mm": round(k_add_sum[k] / k_add_n[k], 4) if k_add_n[k] else None,
                "is_oracle": k > 1,
            } for k in K_GRID
        },
        "first_hit_rank": hist,
        "error_decomposition": {
            "top1_pose": _summarize_errors(err_top1),
            "oracle_best_pose": _summarize_errors(err_oracle),
            "oracle_best_but_failed": _summarize_errors(err_oracle_failed),
        },
    }


def cross_validate(per_obj: Dict[str, dict], topk_ref: Optional[dict]) -> dict:
    """重算的 top1 与参考 JSON 的 49.49% 对账。

    容差 0.05 个百分点（参考 JSON 存的是 round(…, 2)）。超差就是硬警告：
    两份数字来自不同 run，K 曲线不能画在一张图上。
    """
    tot = sum(v["total_samples"] for v in per_obj.values())
    hit = sum(v["k_curve"]["top1"]["add_success_rate"] / 100.0 * v["total_samples"]
              for v in per_obj.values())
    recomputed = round(100.0 * hit / tot, 4) if tot else 0.0
    out = {"recomputed_top1_add_success_rate": recomputed, "total_samples": tot}
    if not topk_ref:
        out["status"] = "no_reference"
        return out
    ref = topk_ref.get("overall_metrics", {}).get("top1", {})
    ref_rate = ref.get("add_success_rate")
    ref_n = topk_ref.get("overall_metrics", {}).get("total_samples")
    out["reference_top1_add_success_rate"] = ref_rate
    out["reference_total_samples"] = ref_n
    if ref_rate is None:
        out["status"] = "reference_missing_top1"
    elif abs(recomputed - ref_rate) <= 0.05 and tot == ref_n:
        out["status"] = "match"
        out["note"] = "两份结果同源，K 曲线可同表"
    else:
        out["status"] = "MISMATCH"
        out["note"] = ("重算 top1 与参考不符（或样本数不同）→ top1 与 top40 "
                       "来自两次设置不同的 run，K 曲线不可同表，须重跑统一。")
    return out


def build_gap_table(per_obj: Dict[str, dict]) -> List[dict]:
    """逐物体缺口表 + 瓶颈归类。

    归类阈值：oracle 上界 < 60% ⇒ 几何天花板主导（再怎么择优也上不去）；
    否则若缺口 > 15 点 ⇒ 检索/择优主导；两者都不满足 ⇒ 已接近饱和。
    """
    rows = []
    for obj, v in sorted(per_obj.items()):
        t1 = v["k_curve"]["top1"]["add_success_rate"]
        t40 = v["k_curve"]["top40"]["add_success_rate"]
        gap = round(t40 - t1, 4)
        if t40 < 60.0:
            kind = "geometry_ceiling"
        elif gap > 15.0:
            kind = "retrieval_bound"
        else:
            kind = "near_saturated"
        rows.append({
            "object": obj, "total_samples": v["total_samples"],
            "top1": t1, "top40_oracle": t40, "gap": gap, "bottleneck": kind,
            "oracle_best_but_failed_n":
                v["error_decomposition"]["oracle_best_but_failed"].get("n", 0),
            "oracle_failed_dz_share":
                v["error_decomposition"]["oracle_best_but_failed"].get("dz_share"),
        })
    return sorted(rows, key=lambda r: -r["gap"])


def load_candidate_files(src_dir: Path) -> Dict[str, List[dict]]:
    """读 src_dir 下所有 top40_add_proj_results_<obj>.json。

    旧脚本的 JSON 顶层可能是裸列表（`json.dump(add_results, ...)`）或带
    包装的 dict，两种都吃。
    """
    out: Dict[str, List[dict]] = {}
    for p in sorted(src_dir.glob("top40_add_proj_results_*.json")):
        obj = p.stem[len("top40_add_proj_results_"):]
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            for key in ("add_results", "results", "details"):
                if key in raw:
                    raw = raw[key]
                    break
        if not isinstance(raw, list):
            raise ValueError(f"{p.name}: 无法定位候选列表（顶层既非 list 也无 "
                             f"add_results/results/details 键）")
        out[obj] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="存放 top40_add_proj_results_<obj>.json 的目录（从服务器拷下来）")
    ap.add_argument("--topk-ref", type=Path, default=None,
                    help="aggregated_metrics_top1_top3_top5.json，用于 top1 对账")
    ap.add_argument("--out", type=Path,
                    default=Path("results/prior/candidate_analysis.json"))
    args = ap.parse_args()

    if not args.src_dir.is_dir():
        print(f"[错误] 目录不存在: {args.src_dir}", file=sys.stderr)
        return 2
    per_samples = load_candidate_files(args.src_dir)
    if not per_samples:
        print(f"[错误] {args.src_dir} 下没有 top40_add_proj_results_*.json。\n"
              f"       这些文件在服务器上（历史管线的输出，见 VERIFICATION.md "
              f"§8.10），先拷到本地再跑。", file=sys.stderr)
        return 2

    per_obj = {obj: analyze_object(s) for obj, s in per_samples.items()}
    topk_ref = json.loads(args.topk_ref.read_text()) if args.topk_ref else None
    report = {
        "source_dir": str(args.src_dir),
        "cross_validation": cross_validate(per_obj, topk_ref),
        "gap_table": build_gap_table(per_obj),
        "per_object": per_obj,
        "caveats": [
            "top1 之外的所有 K 档位都是 GT 择优 oracle 上界（历史管线用 "
            "argmin(GT ADD) 挑候选，见 VERIFICATION.md §8），不可当端到端数字引用。",
            "内点数/内点比/残差择优无法在此重算：历史 JSON 未存内点数与相似度分数"
            "（见 VERIFICATION.md §8.10），需重跑 PnP。",
            "eggbox/glue 用的是双向 Chamfer ADD-S（非标准历史口径），比标准单向"
            "定义偏松，这两个物体的数字不可直接与文献比。",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    cv = report["cross_validation"]
    print(f"\n[top1 对账] 重算 {cv['recomputed_top1_add_success_rate']}% "
          f"({cv['total_samples']} 样本) / 参考 "
          f"{cv.get('reference_top1_add_success_rate')}% → {cv['status']}")
    if cv.get("note"):
        print(f"           {cv['note']}")
    print(f"\n{'物体':<14}{'top1':>8}{'top40[oracle]':>15}{'缺口':>9}  瓶颈")
    for r in report["gap_table"]:
        print(f"{r['object']:<14}{r['top1']:>8.2f}{r['top40_oracle']:>15.2f}"
              f"{r['gap']:>9.2f}  {r['bottleneck']}")
    print(f"\n写出: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
