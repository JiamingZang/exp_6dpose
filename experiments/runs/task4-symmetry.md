# task4-symmetry —— 近似对称量化（分析完成）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task4-symmetry |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md`（任务清单）|

## Question

> duck/ape 失败帧是否"只差一个近似对称轴旋转"（指标病态性）？

## 结论

**无显著近似对称**：自对齐扫描仅小角度平凡解（15° 残差 0.027-0.031×diam）；
失败帧绕对称轴旋转后进 ADD-S 阈值占比 duck 1%（1/88）、ape 3%（2/66）——
指标病态性叙事排除，失败根因在对应点/深度。脚本：scripts/analysis/approx_symmetry.py。

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
