# task1-1-stable-prior —— 稳定摆放先验前提验证（通过）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task1-1-stable-prior |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md::task1-1-stable-prior` |

## Question

> 测试序列物体是否处于稳定摆放（GT 朝上轴 vs 稳定族共识方向）？

## 结论

**通过（判据修正后）**：pybullet 500 次自由落体得稳定族（duck 3/ape 6/cat 3/holepuncher 4/can 3）；
单方向 g* 迭代 + can 本底对照：duck 18.7°/cat 15.8°/holepuncher 18.0° vs can 20.1°（同水平），
ape 24.5°（40% 帧 >30°，软信号）。判据初版（多共识 46 簇稀释）假通过 4-6° 已更正。
脚本：scripts/analysis/stable_pose_prior.py；离线数据 outputs/stable_prior/<obj>.npz。
详细见 docs/EXPERIMENTS.md「任务 1.1」。

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
