# 6d-iter-align —— 迭代渲染对齐（位姿优化章创新点候选）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-iter-align` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 17:20` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-iter-align` |

## Question

> 在当前位姿处重渲染 3DGS 新视角 → MASt3R 再匹配 → 渲染深度反投影到模型系
> → 重解 PnP（迭代 2 轮，渲染对齐损失接受/拒绝）能否把精化从净负转正、
> 救回"可恢复类"坏帧？

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_ia.yaml`（base: guided + `solver.iter_align_iters: 2`）|
| Code change | `src/gaussian/pose_refiner.py`：`render_rgbd()`（gsplat RGB+D）；`src/pipeline.py`：`_iter_align()` + `_solve` 接入点 |
| Data split | 120 帧子集（duck 先验，后推 ape/cat）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° / 分帧 GT 误差 |
| Baseline | duck 30.83/81.67/40.83（6d-rng-fix 新口径）|
| Success line | duck ADD ≥ 基线 +3 且无大类崩溃（接受/拒绝门防变差）|

## Commands

```bash
source env.sh
# 单帧/多帧 sanity（/tmp/sanity_iter_align.py，frame_id 列表）
ITER_ALIGN_DEBUG=1 python -u /tmp/sanity_iter_align.py duck 0 243,423,444
# 全量评估
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_ia/cache \
    --out outputs/exp_ia/results/duck.json
```

## Live Log

- `08-08 17:20`：实现（render_rgbd + _iter_align + 接入点 + 配置），192 测试通过
- `08-08 17:30`：单帧 sanity 发现全量互最近邻内存爆炸（Nq×Nr 点积，
  RAM 冲到 ~97GB）→ 加 n_sample_corr 采样上限（事故记录，与 matcher 同纪律）
- `08-08 17:40`：单帧机制验证：corr 1934→1930、PnP 内点 1637→1710 正常收敛；
  好帧被推偏（loss 0.2831→0.3271）→ 拒绝门正确拦截
- `08-08 17:50`：8 好帧扫描：3 帧被接受（1 帧误放行略差、2 帧改进）
- `08-08 18:10`：10 坏帧精选：**117 帧 111→1.8mm / 592 帧 53→3.8mm 大救援；
  222 帧 690→265mm 部分改善；灾难帧（243/507）无救**——机制对"可恢复类"
  坏帧真实有效
- `08-08 18:20`：全量 duck 评估启动（iter_align 全开）

## Result

待出（全量评估中）。

## Decision

待出。

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
