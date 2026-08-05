# 结构问题审计

## 已确认问题

1. `scripts/` 原来混放数据准备、评测、诊断、维护脚本，`extract13_dc2.sh`/`extract_dc2_fix.sh` 这类一次性脚本没有归档，agent 很容易误当主链入口。
2. `configs/` 原来 29 个 root yaml 混放当前主线和历史实验；多份 `dense80_*` 名称相近但口径不同，容易拿错。
3. 多数脚本用 `Path(__file__).resolve().parents[1]` 推断仓库根，导致不能安全搬目录；已改为向上查找同时包含 `src/` 和 `configs/` 的仓库根。
4. `configs` 的 `base:` 继承按 yaml 所在目录解析；已让 loader 先 resolve symlink，并按 current/archive 重写相对 base。

## 整理结果

- 主链脚本：`scripts/data/`、`scripts/eval/`。
- 诊断和一次性维护：`scripts/analysis/`、`scripts/maintenance/`。
- 主线配置：`configs/current/`。
- 历史配置：`configs/archive/`。
- 旧路径：不再保留重复入口；历史日志只作追溯，运行命令以分类目录为准。
