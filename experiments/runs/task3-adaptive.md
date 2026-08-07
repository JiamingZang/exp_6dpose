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
