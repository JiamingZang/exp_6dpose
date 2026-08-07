# <实验ID> —— <一句话目的>

## Metadata

| 字段 | 值 |
|---|---|
| ID | `<实验ID>` |
| Owner | `<agent/user>` |
| Status | `planned/running/done/dead/blocked` |
| Started | `<YYYY-MM-DD HH:MM>` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
| Queue row | `experiments/QUEUE.md::<实验ID>` |

## Question

这次只回答一个问题：

> <写清楚要验证的假设，不要把多个实验揉在一起。>

## Protocol

| 项 | 值 |
|---|---|
|  `<离线>`| ``离线验证（不改管线）` |
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

## Protocol

| 项 | 值 |
|---|---|
|  `<离线>`| 离线验证（不改管线）|
| Code change | git log 6d-pointmap-t1 |
| Data split | ape 120 帧 |
| Metrics | 3D 残差 / ADD |
| Baseline | 裸 PnP |
| Success line | 查询相机系 3D 存在且残差 <5mm |

## Commands

```bash
# 见 docs/EXPERIMENTS.md「6d-pointmap-t1」
```

## Live Log

- 2026-08-07：不可行结论落盘

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| 见 EXPERIMENTS.md | — | — | — | 成对输出统一系，查询系 3D 不存在 |

## Decision

- 结论：`reject`
- 原因：MASt3R 成对输出为 img1 统一系，Kabsch t_rel≈0
- 下一步：pointmap 价值=dc2 深度一致性（在用）

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 已更新
- [x] `docs/LEDGER.md` 已更新
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
