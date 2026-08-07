# task3-adaptive —— 验证器驱动自适应计算（判负）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task3-adaptive |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md`（任务清单）|

## Question

> 内点分级（高→跳过 refine / 低→升级档）能否在精度不降下提速？

## 结论

**判负（弱物体）**：duck 27.5 vs 基线 33.33（-5.83）；skip 档省 266s 丢 1 帧，
boost 档无效（困难帧本底 8%），std 档 Δ-9 为 rng 漂移副作用。
**can 可用**：Δ0 且 refine 3.3s vs ~7s（skip 档速度收益）。精度-延迟曲线数据点已入
EXPERIMENTS.md「任务 3」。
