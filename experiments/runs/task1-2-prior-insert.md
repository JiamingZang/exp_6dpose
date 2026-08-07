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
