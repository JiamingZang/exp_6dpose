#!/usr/bin/env python3
"""Top-K 精度-延迟曲线绘图（论文 §5.4 图5）。

读 outputs/ablation_topk.json（run_ablation 01_topk 组输出），画：
  1) 5 弱物体均值 ADD(S)@0.1d vs K（含数值标注）
  2) 逐物体曲线（duck/ape/cat/holepuncher/phone）
输出到 /root/毕设/thesis_figs/k_curve.png 与 k_curve_per_object.png。

用法：python3 scripts/analysis/plot_k_curve.py [--json outputs/ablation_topk.json]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJS = ["duck", "ape", "cat", "holepuncher", "phone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/ablation_topk.json")
    ap.add_argument("--out", default="/root/毕设/thesis_figs")
    args = ap.parse_args()

    d = json.load(open(args.json))
    ks = []
    mean = []
    per = {o: [] for o in OBJS}
    for key in sorted(d, key=lambda k: int(k.split("=")[1])):
        k = int(key.split("=")[1])
        ks.append(k)
        mean.append(d[key]["mean"]["add_01d"])
        for o in OBJS:
            per[o].append(d[key]["per_object"].get(o, {}).get("add_01d", 0.0))

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ks, mean, "o-", color="#1f77b4", lw=2, ms=6)
    for x, y in zip(ks, mean):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10)
    ax.set_xlabel("Top-K (number of decoded templates)")
    ax.set_ylabel("Mean ADD(S)@0.1d (%)")
    ax.set_title("Top-K precision curve (5 weak objects, 120 frames)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(args.out) / "k_curve.png", dpi=150)
    print("saved", Path(args.out) / "k_curve.png")

    fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for o, c in zip(OBJS, colors):
        ax2.plot(ks, per[o], "o-", color=c, lw=1.6, ms=4, label=o)
    ax2.set_xlabel("Top-K")
    ax2.set_ylabel("ADD(S)@0.1d (%)")
    ax2.set_title("Per-object Top-K curves")
    ax2.set_xticks(ks)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(Path(args.out) / "k_curve_per_object.png", dpi=150)
    print("saved", Path(args.out) / "k_curve_per_object.png")


if __name__ == "__main__":
    main()
