# 6d-multi-refine —— 多种子渲染对比优化（GS-Pose 差距靶向）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-refine` |
| Owner | `qoder` |
| Status | `queued` |
| Started | 待 GPU（track→mmr→fb 队列后） |
| Finished | — |
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
- `08-11`：入队（GPU 队列 track→mmr→fb 后）。

## Result

| 指标 | baseline (multi) | this run | delta | note |
|---|---:|---:|---:|---|
| ADD | 55.83 |  |  |  |
| Proj@5px |  |  |  |  |
| 单帧耗时 | 10.54s |  |  | 种子 5×120 步精化是主要增量 |

## Decision

- 结论：待跑（queued）
- 原因：
  1. GS-Pose duck 77.2 差距分解：init 38.8 → refiner +38.4 → 77.2；
     我们优化链只 +8.33——渲染对比优化是最大未兑现差距；
  2. multi（+8.33）已证正确盆入口有用；把每种子优化从匹配+PNP 升级为
     渲染对比优化（refiner 120 步探盆 + 精化回退保护 + 盆底择优）；
  3. refiner 单假设判负教训已内建防护：精化变差回退种子位姿、交集掩码
     L1（防掩码损失面污染）、align_loss 盆底择优（非扰动随机种子）。
- 下一步：track→mmr→fb 队列后跑 duck 120 帧；≥58.83 扩 5 弱物体
- 产物：`outputs/exp_multirefine/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
