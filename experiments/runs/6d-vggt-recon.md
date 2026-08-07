# 6d-vggt-recon —— VGGT-1B 成对位姿 sanity 判死

## Metadata

| 字段 | 值 |
|---|---|
| ID | 6d-vggt-recon |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md::6d-vggt-recon` |

## Question

> VGGT-1B 直接输出查询位姿（[模板渲染, 查询裁剪] 成对）能否过域差判据？

## 结论

**判死（1B）**：R_err 中位 94.1°、tz 反号、ADD 823mm（ape 120 帧）。
消融：同图对 0.02°（协议不崩）但跨视角 145°——跨图位姿回归不可靠，非纯域差。
VGGT-Omega（gated 待 HF 授权）可用同脚本复跑（/tmp/vggt_sanity_omega.py）。
详细数字见 docs/EXPERIMENTS.md「6d-vggt-recon sanity」。

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
