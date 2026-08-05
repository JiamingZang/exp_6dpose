# 6d-30k-invdepth-bank —— 30k 迭代 + invdepth 锚点重训对 ape/can/duck/holepuncher 的影响

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-30k-invdepth-bank` |
| Owner | `user`（补记，未走 QUEUE 流程直接跑，2026-08-04 事后补登） |
| Status | `done` |
| Started | `unknown（补记）` |
| Finished | `2026-08-04`（commit `2dfe5fe`） |
| Queue row | `experiments/QUEUE.md::6d-30k-invdepth-bank` |

## Question

这次只回答一个问题：

> 30k 迭代 + 128 参考帧训练（GSPose 同规格）+ bank 含 invdepth 锚点，相对 dc2
> 基线（7000 迭代 + coord），ape/can/duck/holepuncher 4 物体的 ADD 如何变化？

## Protocol

| 项 | 值 |
|---|---|
| Config | configs/archive/dense80_dc_b4.yaml （= dc2 同口径 + batch 4 防 OOM） |
| Code change | commit `2dfe5fe` |
| Data split | 120 帧子集，4 物体（ape/can/duck/holepuncher） |
| Metrics | ADD |
| Baseline | dc2（`docs/STATE.md` 冠军路线的 7000 迭代 + coord bank） |
| Success line | 弱物体（ape/duck/holepuncher）提升，无物体系统性崩溃 |

## Commands

```bash
# 补记：原始命令未留存，仅从 commit 2dfe5fe 与 docs/EXPERIMENTS.md 反推。
# 后续同类实验必须逐条记录，不能靠事后补登。
```

## Live Log

- （补记，无逐条日志——这正是绕开 QUEUE 流程的代价）

## Result

| 指标 | baseline (dc2) | this run (30k+invdepth) | delta | note |
|---|---:|---:|---:|---|
| ape ADD | 37.5 | 45.8 | +8.3 | |
| can ADD | 87.5 | 63.3 | -24.2 | 回归，见下 |
| duck ADD | 31.7 | 33.3 | +1.6 | |
| holepuncher ADD（先行） | 31.7 | 36.7 | +5.0 | |

## Decision

- 结论：`retry`（can 回归未定位到根因前不可推全量）
- 原因：can 44 坏帧中 32 帧在 dc2（7000+coord）下是好的 → 回归来自 bank 变化
  （30k 训练或 invdepth 锚点，二者未隔离）；align_select 重评估只修 10 帧又坏 13
  帧（60.8 < 63.3）→ 择优本身不是根因。
- 下一步：`6d-30k-can-coordbank`（隔离训练 vs 锚点贡献）→ 决定是否推全量 9 物体
  30k 流水线。

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新（补记为 `done`）
- [x] `docs/STATE.md` 已有「已知坑」相关记录（深度渲染部分），30k 数字本身见
      `docs/EXPERIMENTS.md`
- [x] `docs/LEDGER.md` 未新增行（补记时未改动 LEDGER，数字口径以 EXPERIMENTS.md 为准）
- [x] 结果文件路径写清楚：`docs/EXPERIMENTS.md`「30k 训练 + invdepth 锚点验证」小节
- [x] `python3 scripts/analysis/check_state.py` 通过
