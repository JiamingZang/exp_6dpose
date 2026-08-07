# 6d-loc-upper —— GT 掩码定位上界（FastSAM 是否瓶颈）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-loc-upper` |
| Owner | `<agent/user>` |
| Status | `running` |
| Started | `<YYYY-MM-DD HH:MM>` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
| Queue row | `experiments/QUEUE.md::6d-loc-upper` |

## Question

这次只回答一个问题：

> <写清楚要验证的假设，不要把多个实验揉在一起。>

## Protocol

| 项 | 值 |
|---|---|
| Config | `<configs/current/...yaml>` |
| Code change | `<commit/diff/none>` |
| Data split | `<120-frame subset/full/object list>` |
| Metrics | `<ADD/Proj/5cm5°/...>` |
| Baseline | `<必须可追溯到 STATE/LEDGER>` |
| Success line | `<达到什么才算有效>` |

## Commands

```bash
# 逐条写实际命令；不要写“同上”
```

## Live Log

- `<time>`：<启动/中断/恢复/异常/观察>

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## Decision

- 结论：`keep/reject/retry/blocked`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
