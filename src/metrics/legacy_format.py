"""历史（MyPose）评估产物格式的对接层（见 VERIFICATION.md §8）。纯 numpy。

⚠ oracle 警告（审查结论，见 _prior_code/MyPose/AUDIT.md 与 VERIFICATION.md §8）：
历史管线的候选择优用 GT ADD 最小挑答案（用测试集真值挑候选），不存在按
内点数择优的逻辑。因此本模块产出/导入的 **K>1 的每一档**（含 Top-40 的
82.73%、top5_best 74.15%、top3_best 68.70%）都是 **GT 择优上界
（oracle upper bound）**，只能作"模板检索若完美时的性能潜力"分析，
**不得与端到端方法同表比较**；唯一的非 oracle 数字是
**top1 = ADD 49.49% / Proj 59.22%**（K=1 没有择优动作）。本库主路线的
择优判据是内点数（src/solver/selection.py，非 oracle），两者语义不同，
报告里必须分开标注。

这个性质**同时落成 JSON 数据字段**（只靠注释挡不住误引用）：`protocol`
（prior_MyPose_oracle_top40 / _oracle_topk）、`selection`（oracle_gt_add）、
`is_oracle_upper_bound`、`tiers[*].is_oracle`、`non_oracle_reference`。

两种历史 JSON 格式（均为真实实验产物，视为兼容基准，不要改 schema）：

1. aggregated_metrics_all_objects40.json（Top-40 **GT 择优上界**，13407 样本，
   ADD 82.73% / Proj 81.99%；同一批候选的非 oracle 数字是 top1 = 49.49%）
   - per_object_metrics：total_samples / add_success_count /
     add_success_rate / avg_add_mm / proj_success_count /
     proj_success_rate / avg_proj_error_px；平均误差只统计有限值
   - overall_metrics：成功率 = 总成功数/总样本；
     overall_avg_* = 各物体平均值的再平均，不是全样本平均——
     保留历史口径以便逐位对比

2. aggregated_metrics_top1_top3_top5.json（top-K best 消融）：
   同步选择语义——在前 K 个候选里取 ADD 最小者，投影误差用同一候选的值
   （不是独立取 proj 最小）。键名规则：K=1 → "top1"，K>1 → f"top{K}_best"。

本模块同时提供：
- 本库帧级结果 → 历史格式（评估导出，run_linemod.py --aggregated-out）
- 历史格式 → 本库报告格式（scripts/import_prior_metrics.py）
- topK best 同步选择与聚合（evaluate_object 的 metrics.topk_best）
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# topK best：同步选择（历史对照口径，见 VERIFICATION.md §8.4）
# ---------------------------------------------------------------------------
def topk_key(k: int) -> str:
    """历史 JSON 键名规则：top1 / top3_best / top5_best。"""
    return "top1" if int(k) == 1 else f"top{int(k)}_best"


def topk_best_pick(add_errors: Sequence[float], proj_errors: Sequence[float],
                   k: int) -> Tuple[int, float, float]:
    """前 K 个候选中按 ADD 最小同步选择（历史对照语义，勿改成独立取最小）。

    proj 用 ADD 最小候选**同一下标**的值，不是独立最小值（见
    VERIFICATION.md §8.4）。候选不足 K 个时在现有候选里选；无候选返回
    (-1, inf, inf)。

    Args:
        add_errors / proj_errors: 候选按预测排名排序的误差序列（同长）
    Returns:
        (选中下标, ADD 误差, 同一候选的投影误差)
    """
    adds = np.asarray(add_errors, dtype=np.float64)[:max(int(k), 0)]
    projs = np.asarray(proj_errors, dtype=np.float64)[:max(int(k), 0)]
    if len(adds) == 0:
        return -1, float("inf"), float("inf")
    i = int(np.argmin(adds))
    return i, float(adds[i]), float(projs[i])


def object_topk_metrics(cand_adds: List[Sequence[float]],
                        cand_projs: List[Sequence[float]],
                        add_thresh: float, proj_thresh: float = 5.0,
                        ks: Sequence[int] = (1, 3, 5)) -> Dict:
    """单物体的 topK best 聚合（历史 top1_top3_top5 JSON 的 per-object 段）。

    Args:
        cand_adds:  每帧一个序列——各候选（按预测排名）的 ADD 误差
        cand_projs: 同形状的投影误差；无候选帧传空序列（计入分母、判失败）
        add_thresh: ADD 成功阈值（0.1 × diameter，单位与误差一致）
        proj_thresh: 投影成功阈值（历史口径 5px）
    Returns:
        {"total_samples": n, topk_key(k): {add_success_count, add_success_rate,
         avg_add_mm, proj_success_count, proj_success_rate,
         avg_proj_error_px}, ...}（成功率 %，平均只计有限误差）

    **内部保留全精度，不在这里舍入**：overall 是对未舍入的逐物体平均值
    再平均，若这里先 round 到 4/2 位，`aggregate_topk_all_objects` 就是在
    对已舍入值求平均，overall 会有可见偏差。舍入只发生在落盘时
    （`aggregate_topk_all_objects` 的返回值）。
    """
    assert len(cand_adds) == len(cand_projs), "帧数不一致"
    n = len(cand_adds)
    out: Dict = {"total_samples": n}
    for k in ks:
        picks = [topk_best_pick(a, p, k)
                 for a, p in zip(cand_adds, cand_projs)]
        adds = np.array([p[1] for p in picks])
        projs = np.array([p[2] for p in picks])
        add_succ = int(np.sum(adds < add_thresh))
        proj_succ = int(np.sum(projs < proj_thresh))
        finite_a = adds[np.isfinite(adds)]
        finite_p = projs[np.isfinite(projs)]
        out[topk_key(k)] = {
            "add_success_count": add_succ,
            "add_success_rate": 100.0 * add_succ / n if n else 0.0,
            "avg_add_mm": float(finite_a.mean()) if len(finite_a) else 0.0,
            "proj_success_count": proj_succ,
            "proj_success_rate": 100.0 * proj_succ / n if n else 0.0,
            "avg_proj_error_px": float(finite_p.mean())
                                 if len(finite_p) else 0.0,
        }
    return out


def _round_topk_entry(entry: Dict) -> Dict:
    """按历史 JSON 的小数位舍入单个档位（rate 2 位 / avg_add 4 位 / avg_proj 2 位）。

    只在落盘时调用——聚合运算一律走全精度（见 object_topk_metrics）。
    """
    digits = {"add_success_rate": 2, "proj_success_rate": 2,
              "avg_add_mm": 4, "avg_proj_error_px": 2}
    return {k: (round(float(v), digits[k]) if k in digits else v)
            for k, v in entry.items()}


def aggregate_topk_all_objects(per_object: Dict[str, Dict],
                               ks: Sequence[int] = (1, 3, 5),
                               timestamp: Optional[str] = None) -> Dict:
    """多物体 topK best → 历史 aggregated_metrics_top1_top3_top5.json 格式。

    overall 口径同历史文件：成功率 = 总成功数/总样本；avg = 各物体平均的
    再平均（与 all_objects40 的 overall 口径一致）。**再平均用的是未舍入的
    逐物体值**，舍入只在最后落盘时做——传进来的 per_object 是
    `object_topk_metrics` 的全精度输出。
    """
    total = sum(o["total_samples"] for o in per_object.values())
    overall: Dict = {"total_samples": total}
    for k in ks:
        key = topk_key(k)
        add_succ = sum(o[key]["add_success_count"]
                       for o in per_object.values())
        proj_succ = sum(o[key]["proj_success_count"]
                        for o in per_object.values())
        avg_adds = [o[key]["avg_add_mm"] for o in per_object.values()]
        avg_projs = [o[key]["avg_proj_error_px"] for o in per_object.values()]
        overall[key] = _round_topk_entry({
            "add_success_rate": 100.0 * add_succ / total if total else 0.0,
            "avg_add_mm": float(np.mean(avg_adds)) if avg_adds else 0.0,
            "proj_success_rate": 100.0 * proj_succ / total if total else 0.0,
            "avg_proj_error_px": float(np.mean(avg_projs))
                                 if avg_projs else 0.0,
            "total_add_successes": add_succ,
            "total_proj_successes": proj_succ,
        })
    # per_object 也在落盘这一刻才舍入（不改调用方手上的全精度 dict）
    per_object_out = {
        obj: {kk: (_round_topk_entry(vv) if isinstance(vv, dict) else vv)
              for kk, vv in o.items()}
        for obj, o in per_object.items()}
    return {
        "overall_metrics": overall,
        "per_object_metrics": per_object_out,
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S",
                                                time.localtime()),
    }


# ---------------------------------------------------------------------------
# 本库帧级结果 → 历史 aggregated_metrics_all_objects40 格式
# ---------------------------------------------------------------------------
def per_object_from_frames(per_frame: List[Dict]) -> Dict:
    """evaluate_object 的帧级结果 → 历史 per_object_metrics 段。

    per_frame 每项含 add（误差 mm，失败帧为 inf）、proj（误差 px）、
    add_01d / proj_5px（0/1 命中）——与 src/metrics/pose_metrics.py
    evaluate_pose 的输出键一致。平均只统计有限误差（历史口径）。
    """
    n = len(per_frame)
    add_succ = int(sum(f["add_01d"] for f in per_frame))
    proj_succ = int(sum(f["proj_5px"] for f in per_frame))
    adds = np.array([f["add"] for f in per_frame], dtype=np.float64)
    projs = np.array([f["proj"] for f in per_frame], dtype=np.float64)
    finite_a = adds[np.isfinite(adds)]
    finite_p = projs[np.isfinite(projs)]
    return {
        "total_samples": n,
        "add_success_count": add_succ,
        "add_success_rate": 100.0 * add_succ / n if n else 0.0,
        "avg_add_mm": float(finite_a.mean()) if len(finite_a) else 0.0,
        "proj_success_count": proj_succ,
        "proj_success_rate": 100.0 * proj_succ / n if n else 0.0,
        "avg_proj_error_px": float(finite_p.mean()) if len(finite_p) else 0.0,
    }


def aggregate_all_objects(per_object: Dict[str, Dict],
                          timestamp: Optional[str] = None) -> Dict:
    """多物体 → 历史 aggregated_metrics_all_objects40.json 完整格式。

    overall 口径逐项对齐历史文件（见 VERIFICATION.md §8.4）：
    - 成功率 = 总成功数 / 总样本 × 100
    - overall_avg_add_mm / overall_avg_proj_error_px = 各物体平均的再平均
      （非全样本平均——有意保留历史口径以便直接对比）
    """
    total = sum(o["total_samples"] for o in per_object.values())
    add_succ = sum(o["add_success_count"] for o in per_object.values())
    proj_succ = sum(o["proj_success_count"] for o in per_object.values())
    avg_adds = [o["avg_add_mm"] for o in per_object.values()]
    avg_projs = [o["avg_proj_error_px"] for o in per_object.values()]
    return {
        "overall_metrics": {
            "total_samples": total,
            "add_success_rate": 100.0 * add_succ / total if total else 0.0,
            "overall_avg_add_mm": float(np.mean(avg_adds)) if avg_adds else 0.0,
            "proj_success_rate": 100.0 * proj_succ / total if total else 0.0,
            "overall_avg_proj_error_px": float(np.mean(avg_projs))
                                         if avg_projs else 0.0,
            "total_add_successes": add_succ,
            "total_proj_successes": proj_succ,
        },
        "per_object_metrics": per_object,
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S",
                                                time.localtime()),
    }


# ---------------------------------------------------------------------------
# 历史格式 → 本库评估报告格式（scripts/import_prior_metrics.py 用）
# ---------------------------------------------------------------------------
def non_oracle_reference(topk_agg: Optional[Dict]) -> Optional[Dict]:
    """从历史 top1/3/5 JSON 里取出**唯一的非 oracle 数字**（top1 档）。

    为什么单独抽出来：topK best 里只有 K=1 没有"在 K 个候选里按 GT 挑"的
    动作，所以 top1 是历史管线唯一可与端到端方法同表比较的数字；其余档位
    （含 Top-40 的 82.73%）都是 GT 择优上界。数值一律从历史 JSON 的
    overall_metrics.top1 读，**不许硬编码**——历史文件重算后本字段自动跟随。

    Args:
        topk_agg: 历史 aggregated_metrics_top1_top3_top5.json 解析结果；
                  None 或不含 top1 档时返回 None（调用方据此省略该字段）
    Returns:
        {"top1_add_01d": ..., "top1_proj_5px": ..., "note": ...} 或 None
    """
    if not topk_agg:
        return None
    top1 = (topk_agg.get("overall_metrics") or {}).get("top1")
    if not top1:
        return None
    return {
        "top1_add_01d": float(top1["add_success_rate"]),
        "top1_proj_5px": float(top1["proj_success_rate"]),
        "note": "top1 是历史管线唯一的非 oracle（端到端）数字，"
                "与端到端方法同表比较只能用它",
    }


def prior_to_report(agg: Dict, source: str = "",
                    topk_agg: Optional[Dict] = None) -> Dict:
    """历史 aggregated_metrics_all_objects*.json → 本库 linemod_main.json 格式。

    ⚠ 产出的数字是 **GT 择优 oracle 上界**（历史管线用 GT ADD 最小择优，
    见 VERIFICATION.md §8），不是端到端结果——所以 protocol 里写死
    `prior_MyPose_oracle_top40`，并在数据里落 `selection` /
    `is_oracle_upper_bound` 字段。**光靠注释挡不住误用，字段必须进 JSON**。

    映射：add_success_rate → add_01d，proj_success_rate → proj_5px，
    total_samples → n（aggregate() 的键，见 pose_metrics.py）。
    历史管线无 5cm5° 指标，置 null 而不是编数。mean = 各物体指标的算术
    平均（与 run_linemod.print_table 一致）。

    Args:
        topk_agg: 可选的历史 top1/3/5 JSON（同一批候选），给了就在顶层附
                  `non_oracle_reference`（top1 端到端数字），便于报告里
                  把 oracle 上界与可比数字并排放。
    """
    per_object = {}
    for obj, m in agg["per_object_metrics"].items():
        per_object[obj] = {
            "add_01d": float(m["add_success_rate"]),
            "proj_5px": float(m["proj_success_rate"]),
            "cm_deg": None,                      # 历史管线未测 5cm5°
            "n": int(m["total_samples"]),
            "avg_add_mm": float(m["avg_add_mm"]),
            "avg_proj_error_px": float(m["avg_proj_error_px"]),
        }
    mean = {
        "add_01d": float(np.mean([o["add_01d"] for o in per_object.values()])),
        "proj_5px": float(np.mean([o["proj_5px"]
                                   for o in per_object.values()])),
        "cm_deg": None,
    }
    report = {
        "source": source,
        # 标注来源 + oracle 性质，防与新库端到端结果混淆
        "protocol": "prior_MyPose_oracle_top40",
        "selection": "oracle_gt_add",         # 择优判据：GT ADD 最小
        "is_oracle_upper_bound": True,
        "overall": agg.get("overall_metrics", {}),
        "per_object": per_object,
        "mean": mean,
        "timestamp": agg.get("timestamp"),
    }
    ref = non_oracle_reference(topk_agg)
    if ref is not None:
        report["non_oracle_reference"] = ref
    return report


def topk_from_key(tier: str) -> int:
    """tier 键名 → K（topk_key 的逆）：'top1'→1，'top5_best'→5。

    解析失败返回 0（调用方视作"K 未知"，按 oracle 从严标注）。
    """
    s = str(tier)
    if not s.startswith("top"):
        return 0
    s = s[3:]
    if s.endswith("_best"):
        s = s[:-len("_best")]
    return int(s) if s.isdigit() else 0


def prior_topk_to_report(agg: Dict, source: str = "") -> Dict:
    """历史 aggregated_metrics_top1_top3_top5.json → 本库报告格式。

    每个 topK 档位展开成一份 per_object/mean 子表。

    ⚠ oracle 标注：K>1 的每一档都在 K 个候选里按 **GT ADD** 挑最优，
    是上界；只有 top1 没有这个动作，是端到端数字。故每档带 `is_oracle`，
    顶层带 `non_oracle_reference`（从本文件自身的 top1 档读，不硬编码）。
    """
    overall = agg["overall_metrics"]
    tiers = [k for k in overall if k != "total_samples"]
    report: Dict = {
        "source": source,
        "protocol": "prior_MyPose_oracle_topk",
        "selection": "oracle_gt_add",
        # K>1 的档位都是上界；top1 例外，逐档看 tiers[*].is_oracle
        "is_oracle_upper_bound": any(topk_from_key(t) != 1 for t in tiers),
        "overall": overall,
        "tiers": {},
        "timestamp": agg.get("timestamp"),
    }
    ref = non_oracle_reference(agg)
    if ref is not None:
        report["non_oracle_reference"] = ref
    for tier in tiers:
        per_object = {}
        for obj, m in agg["per_object_metrics"].items():
            t = m[tier]
            per_object[obj] = {
                "add_01d": float(t["add_success_rate"]),
                "proj_5px": float(t["proj_success_rate"]),
                "n": int(m["total_samples"]),
                "avg_add_mm": float(t["avg_add_mm"]),
                "avg_proj_error_px": float(t["avg_proj_error_px"]),
            }
        mean = {
            "add_01d": float(np.mean([o["add_01d"]
                                      for o in per_object.values()])),
            "proj_5px": float(np.mean([o["proj_5px"]
                                       for o in per_object.values()])),
        }
        report["tiers"][tier] = {
            "per_object": per_object, "mean": mean,
            # K=1 无 GT 择优动作 → 非 oracle；K>1 是上界
            "is_oracle": topk_from_key(tier) != 1,
        }
    return report
