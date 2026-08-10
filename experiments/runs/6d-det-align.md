# 6d-det-align —— 检测框口径（GT bbox 定位上界）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-det-align` |
| Owner | `<agent/user>` |
| Status | `running` |
| Started | `<YYYY-MM-DD HH:MM>` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
| Queue row | `experiments/QUEUE.md::6d-det-align` |

## Question

这次只回答一个问题：

> GSPose 92.0 为检测框输入口径；本文无检测器端到端 78.07。检测框口径（GT bbox 定位、无 DINOv2 检索、无 GT 掩码依赖）下级联数字是多少？零样本定位是否是剩余差距的主要来源？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_gtbbox.yaml`（base: dense80_depthc_ia + segmenter: gt_bbox + gt_bbox_use_mask: false）|
| Code change | `<commit/diff/none>` |
| Data split | 120 帧子集先验证（ape/cat/duck/holepuncher/can），显著则全量 14968 帧 |
| Metrics | `<ADD/Proj/5cm5°/...>` |
| Baseline | 无检测器端到端级联 78.07（STATE 2026-08-10，exp_full_ia）|
| Success line | 检测框口径 13 物体数字出炉；与 78.07 的差值量化零样本定位的代价（GSPose 对比通道）|

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
