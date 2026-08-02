#!/usr/bin/env python3
"""导入 MyPose 历史实验结果并转成本库报告格式（见 VERIFICATION.md §8.5）。

读取 _prior_code/MyPose/aggregated_metrics_*.json（真实 Top-40 结果与
top1/3/5 消融，13407 样本），转换成本库评估报告格式写到 results/prior/，
表格可直接从本库产物目录生成，无需回头解析历史格式。

⚠ 历史结果的性质：除 top1 外全部档位（含 Top-40 的 82.73%）都是
**GT 择优 oracle 上界**（用测试集真值挑候选，见 VERIFICATION.md §8）。
转换后的 JSON 带 `protocol` / `is_oracle_upper_bound` / `tiers[*].is_oracle`
字段，终端输出也标 [oracle]，避免被当端到端数字引用。

用法：
    python scripts/import_prior_metrics.py                     # 默认路径
    python scripts/import_prior_metrics.py --src-dir <MyPose目录> --out-dir <输出>

输出（每个源文件一份）：
    results/prior/<原文件名去后缀>_report.json   # 本库格式（含 source 溯源）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.legacy_format import prior_to_report, prior_topk_to_report

# 默认历史结果目录：仓库同级的 _prior_code/MyPose（毕设目录布局）
DEFAULT_SRC = Path(__file__).resolve().parents[2] / "_prior_code" / "MyPose"


def convert_prior_file(src_path: Path, out_dir: Path,
                       topk_agg: dict = None) -> Path:
    """单个历史 aggregated_metrics JSON → 本库报告，返回输出路径。

    两种历史 schema 按 overall_metrics 里是否有 top1 键分流：
    - 有 top1 → top1/3/5 消融格式（prior_topk_to_report）
    - 无     → Top-40 oracle 上界格式（prior_to_report）

    Args:
        topk_agg: top1/3/5 那份 JSON 的解析结果。转 Top-40 时传入，报告
                  顶层就能带上 non_oracle_reference（top1 端到端数字），
                  让 oracle 上界与可比数字同在一个文件里。
    """
    agg = json.loads(src_path.read_text())
    overall = agg.get("overall_metrics", {})
    if "top1" in overall:
        report = prior_topk_to_report(agg, source=str(src_path))
    else:
        report = prior_to_report(agg, source=str(src_path),
                                 topk_agg=topk_agg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_path.stem}_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return out_path


def _find_topk_agg(files) -> dict:
    """在源文件里找出 top1/3/5 那份（overall 含 top1 键），找不到返回 None。"""
    for f in files:
        agg = json.loads(f.read_text())
        if "top1" in agg.get("overall_metrics", {}):
            return agg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC),
                    help="旧结果目录（含 aggregated_metrics_*.json）")
    ap.add_argument("--out-dir", default="results/prior",
                    help="输出目录（新库报告格式）")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    files = sorted(src_dir.glob("aggregated_metrics_*.json"))
    if not files:
        print(f"[import_prior_metrics] {src_dir} 下没有 "
              f"aggregated_metrics_*.json，检查 --src-dir", file=sys.stderr)
        sys.exit(1)

    topk_agg = _find_topk_agg(files)
    for f in files:
        out = convert_prior_file(f, out_dir, topk_agg=topk_agg)
        report = json.loads(out.read_text())
        if "tiers" in report:
            # 每档标 [oracle]/[端到端]，防止终端输出被直接抄进论文表格
            tiers = ", ".join(
                f"{t}{'[oracle]' if v['is_oracle'] else '[端到端]'}: "
                f"ADD {v['mean']['add_01d']:.2f}%/Proj "
                f"{v['mean']['proj_5px']:.2f}%"
                for t, v in report["tiers"].items())
            print(f"[import] {f.name} → {out}\n         {tiers}")
        else:
            o = report["overall"]
            tag = "[oracle 上界]" if report.get("is_oracle_upper_bound") else ""
            print(f"[import] {f.name} → {out}\n"
                  f"         overall{tag} ADD "
                  f"{o.get('add_success_rate', 0):.2f}% / "
                  f"Proj {o.get('proj_success_rate', 0):.2f}% "
                  f"({o.get('total_samples', 0)} 样本)")
            ref = report.get("non_oracle_reference")
            if ref:
                print(f"         非 oracle 参照（同批候选 top1，端到端）："
                      f"ADD {ref['top1_add_01d']:.2f}% / "
                      f"Proj {ref['top1_proj_5px']:.2f}%")


if __name__ == "__main__":
    main()
