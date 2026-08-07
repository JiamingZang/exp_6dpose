# task1-2-prior-insert —— 稳定先验接入点 B（判负）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task1-2-prior-insert |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md::task1-2-prior-insert` |

## Question

> 稳定先验软评分（score = inlier + λ·prior）能否提升弱物体？

## 结论

**判负（λ=0.5）**：duck -6.67 / ape +0.83 / cat +3.33 / holepuncher -5.83（弱 4 平均 -2.08），
can 0.00。机制：联合 PnP 吞掉择优 + 本底 18-20° 使 prior 在正误候选间重叠。
接入点 A 判死（模板物体姿态固定，加权恒等）。死因：先验与失败帧错配。
详细见 docs/EXPERIMENTS.md「任务 1.2」。

## Protocol

| 项 | 值 |
|---|---|
| Config | 见 QUEUE.md 对应行 |
| Code change | 见 git log（同 ID commit）|
| Data split | 120 帧子集 |
| Metrics | ADD/Proj/5cm5° |
| Baseline | docs/STATE.md 冠军表 |
| Success line | 见 QUEUE.md 对应行 |

## Commands

```bash
# 实际命令见 docs/EXPERIMENTS.md 对应节（脚本与命令已随 commit 落库）
```

## Live Log

- 2026-08-07：完成并判负/判死，详见 EXPERIMENTS.md

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| 见 docs/EXPERIMENTS.md 对应节 | — | — | — | 详细数字在 EXPERIMENTS.md |

## Decision

- 结论：`reject`（判负/判死）或见正文
- 原因：见 Question 下正文
- 下一步：见 QUEUE.md 最新行

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
