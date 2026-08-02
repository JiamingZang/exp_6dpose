#!/usr/bin/env python3
"""我们 vs MyPose 逐档/逐物体对比表。

读我们 run_linemod.py 的 --aggregated-out 兼容产物（含 topk_best 档）与
_prior_code/MyPose 的历史 JSON，输出：
  1. 端到端 top1 逐物体表（我们 vs MyPose）
  2. oracle top3/top5/top40 潜力对比（我们 vs MyPose）
  3. 每档 overall 汇总

用法：python compare_prior.py <our_aggregated.json> [our_topk.json]
"""
import json
import sys
from pathlib import Path


def load(path):
    return json.load(open(path))


def fmt(v, suffix="%"):
    return f"{v:7.2f}{suffix}" if v is not None else "     n/a"


def obj_table(our, prior, tier_our=None, tier_prior=None):
    objs = sorted(set(our["per_object_metrics"]) | set(prior["per_object_metrics"]))
    print(f"\n{'物体':<12}{'我们ADD':>10}{'MyPose ADD':>12}{'我们Proj':>10}{'MyPose Proj':>12}")
    for o in objs:
        a = our["per_object_metrics"].get(o, {})
        b = prior["per_object_metrics"].get(o, {})
        if tier_our and isinstance(a, dict) and tier_our in a:
            a_add = a[tier_our].get("add_success_rate")
            a_proj = a[tier_our].get("proj_success_rate")
        elif isinstance(a, dict):
            a_add = a.get("add_success_rate")
            a_proj = a.get("proj_success_rate")
        else:
            a_add = a_proj = None
        if tier_prior and isinstance(b, dict) and tier_prior in b:
            b_add = b[tier_prior].get("add_success_rate")
            b_proj = b[tier_prior].get("proj_success_rate")
        elif isinstance(b, dict):
            b_add = b.get("add_success_rate")
            b_proj = b.get("proj_success_rate")
        else:
            b_add = b_proj = None
        print(f"{str(o)[:12]:<12}{fmt(a_add):>10}{fmt(b_add):>12}"
              f"{fmt(a_proj):>10}{fmt(b_proj):>12}")
    return objs


def overall_of(metrics, key):
    if key in metrics:
        return (metrics[key].get("add_success_rate"),
                metrics[key].get("proj_success_rate"))
    return None, None


def main():
    our_path = sys.argv[1]
    our = load(our_path)
    our_topk_path = sys.argv[2] if len(sys.argv) > 2 else None
    prior_all = load("/root/毕设/_prior_code/MyPose/aggregated_metrics_all_objects40.json")
    prior_topk = load("/root/毕设/_prior_code/MyPose/aggregated_metrics_top1_top3_top5.json")

    print("=" * 70)
    print("端到端 top1 逐物体对比（我们 = 内点择优 | MyPose = 检索 top1，均非 oracle）")
    print("=" * 70)
    obj_table(our, prior_topk, tier_prior="top1")
    # overall
    om = our["overall_metrics"]
    o_add = om.get("add_success_rate")
    o_proj = om.get("proj_success_rate")
    pt = prior_topk["overall_metrics"]["top1"]
    p_add, p_proj = pt["add_success_rate"], pt["proj_success_rate"]
    print(f"\n{'OVERALL':<12}{fmt(o_add):>10}{fmt(p_add):>12}"
          f"{fmt(o_proj):>10}{fmt(p_proj):>12}")

    if our_topk_path:
        our_topk = load(our_topk_path)
        print("\n" + "=" * 70)
        print("oracle topK 潜力对比（GT 择优上界，仅作检索潜力分析）")
        print("=" * 70)
        for tier in ("top1", "top3_best", "top5_best"):
            oa, op = overall_of(our_topk["overall_metrics"], tier)
            pa, pp = overall_of(prior_topk["overall_metrics"], tier)
            print(f"{tier:<12} 我们 ADD {fmt(oa)} / Proj {fmt(op)}   "
                  f"|  MyPose ADD {fmt(pa)} / Proj {fmt(pp)}")
        # top40 oracle（我们开 metrics.topk_best 含 40 时）
        oa, op = overall_of(our_topk["overall_metrics"], "top40_best")
        if oa is not None:
            print(f"{'top40_best':<12} 我们 ADD {fmt(oa)} / Proj {fmt(op)}   "
                  f"|  MyPose ADD {fmt(prior_all['overall_metrics']['add_success_rate'])}"
                  f" / Proj {fmt(prior_all['overall_metrics']['proj_success_rate'])}"
                  f"（aggregated_metrics_all_objects40）")


if __name__ == "__main__":
    main()
