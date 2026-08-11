# 6d-track-seed —— 帧间跟踪种子（时间分布成簇靶向）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-track-seed` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 16:32` |
| Finished | `2026-08-11 17:17` |
| Queue row | `experiments/QUEUE.md::6d-track-seed` |

## Question

> 失败帧时间成簇（duck 孤立率仅 14%、最长连续段 19 帧；holepuncher 最长
> 25 帧）——视角片段级场景问题。段起始帧用前一成功帧位姿作 iter_align
> 种子（渲染视角连续 → MASt3R 匹配质量高），能否阻止整段失败？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_track.yaml` （champion + solver.track_seed: true，单假设）|
| Code change | 无（track_seed 随 6d-ia-multi 已实现：prev_pose 作为种子追加）|
| Data split | duck 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线）；对照 multi 55.83 |
| Success line | duck ≥ 基线 +3 且无 -3 回退 |
| 口径警告 | 视频模式（用序列时间信息）——BOP 官方评测逐帧独立，仅论文视频扩展章节实验，主表保持单帧口径 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_track.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_track/cache \
    --out outputs/exp_track/results/duck.json
```

## Live Log

- `08-11 16:32`：启动（等 multi 完成后）。
- `08-11 17:17`：**出炉 duck 50.83**（+3.33 vs 基线，-5.00 vs multi）。

## Result

| 指标 | baseline (ia) | multi | track | delta vs 基线 |
|---|---:|---:|---:|---|
| ADD | 47.50 | 55.83 | 50.83 | **+3.33** |
| Proj@5px | 81.67 |  | 78.33 | -3.34 |
| 5cm5° |  |  | 64.17 |  |
| 单帧耗时 | 10.31s | 10.54s | ~14.4s（iter_align 0.75s + alt_matching 3.99s + refine 5.29s）| +40% |

## Decision

- 结论：`keep（弱）`——track 有效但不敌 multi，且更贵
- 原因：
  1. 上帧位姿种子 +3.33（47.50→50.83）——时间连续性确实有用，但
     48.5% t 错失败帧的旋转本就对（R≤10°），种子收益主要落在
     36.2% R>60° 选错模板帧；
  2. **multi 55.83 > track 50.83**：池内多假设（单帧内）优于时间种子
     （跨帧）——选错模板帧的正确盆多由池内候选提供，不依赖视频模式；
  3. 代价 +40%（alt_matching 3.99s = 每帧多一次 MASt3R 重匹配），
     远贵于 multi 的 +2%——效率上 track 是劣质解；
  4. 口径警告兑现：BOP 主表逐帧独立，track 仅论文视频扩展章节素材。
- 下一步：不并入 champion；与 multi 组合（track+multi 种子叠加）不排
  （multi 已含池内 top-k，时间种子增量预期小且代价高）
- 产物：`outputs/exp_track/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
