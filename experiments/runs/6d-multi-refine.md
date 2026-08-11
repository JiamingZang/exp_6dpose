# 6d-multi-refine —— 多种子渲染对比优化（GS-Pose 差距靶向）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-refine` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 19:32` |
| Finished | `2026-08-11 20:41` |
| Queue row | `experiments/QUEUE.md::6d-multi-refine` |

## Question

> GS-Pose duck 77.2（其 GS-Refiner 从 init 38.8 拉到 77.2，+38.4）vs 我们
> 优化链只 +8.33——差距在渲染对比优化的修复能力。multi（+8.33）已证
> "正确盆入口"有用；把每个种子的优化从"匹配+PNP"（iter_align）升级为
> "渲染对比优化"（refiner 多种子启动），能否兑现 GS-Refiner 式增益？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_multirefine.yaml` （champion + iter_align_multi_hypo: 5 + iter_align_seed_refine_iters: 120）|
| Code change | `src/pipeline.py`：multi 种子循环内每种子 iter_align 后追加短渲染对比优化（`_refiner.refine`，120 步），精化变差回退种子位姿，align_loss 盆底择优 |
| Data split | duck 120 帧子集先验证（GS-Pose 直接对照口径）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 55.83（6d-ia-multi 已结案数字）|
| Success line | duck ≥ baseline +3（≥58.83）且无 -3 回退 → 扩 5 弱物体（含 GS-Pose 对照）|

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_multirefine.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_multirefine/cache \
    --out outputs/exp_multirefine/results/duck.json
```

## Live Log

- `08-11 17:0x`：实现（iter_align_seed_refine_iters 开关：种子级渲染对比
  优化 + 精化回退保护 + 盆底择优），202 测试过；GS-Pose duck 差距
  分解（77.2 = init 38.8 + refiner 38.4 vs 我们 47.50 + 优化链 8.33）
  作为动机。
- `08-11 17:20`：入队（GPU 队列 track→mmr→fb 后）。
- `08-11 19:32`：启动（fb 完成后接力）。
- `08-11 20:41`：**出炉 duck 49.17**（+1.67 vs 基线，-6.66 vs multi）。

## Result

| 指标 | baseline (ia) | multi | this run | delta vs multi |
|---|---:|---:|---:|---|
| ADD | 47.50 | 55.83 | 49.17 | **-6.66** |
| Proj@5px | 81.67 |  | 90.83 | +9.16 |
| 5cm5° |  |  | 66.67 |  |
| 单帧耗时 | 10.31s | 10.54s | iter_align 17.58s（种子 5×120 步精化）+ refine 6.38s | ~2.3× |

## Decision

- 结论：`drop`（种子级渲染对比优化判负——refiner 盆底择优失效）
- 原因：
  1. **ADD -6.66 但 Proj +9.16**：120 步 refiner 把位姿推到"渲染贴边但
     几何错"的盆底——refiner 损失面（掩码污染 + 光度局部极小）与
     ADD 相关性弱，盆底择优被带偏。这正是 6d-refiner-v2 单假设判负
     的同一机制（align_loss 对 duck 51% 判不准）在多种子框架下的复现；
  2. multi 的胜因恰恰是**不做**长精化：iter_align 种子择优在"对应
     质量"维度（MASt3R 内点），与几何真值相关性强；refiner 步数越多
     越往渲染自洽（非几何正确）方向漂；
  3. GS-Pose GS-Refiner +38.4 的前提我们不满足：其 init 由每物体
     训练的分割器 + 旋转感知检索给出（38.8 起点本身已好），且 refiner
     从掩码几何初始化出发步数 400——我们 FastSAM 掩码污染 + 无几何
     初始化，短精化探盆救不了选择倒挂。
- 下一步：渲染对比优化方向结案（两轮判负：6d-refiner-v2 单假设、
  6d-multi-refine 多种子）；差距回填到候选池生成（MASt3R 对应质量）
- 产物：`outputs/exp_multirefine/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
