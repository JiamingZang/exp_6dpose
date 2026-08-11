# 6d-ia-multi —— 多初始假设 iter_align（挑战 3：模板离散化/选错模板）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-ia-multi` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 14:55` |
| Finished | `2026-08-11 16:34` |
| Queue row | `experiments/QUEUE.md::6d-ia-multi` |

## Question

这次只回答一个问题：

> 失败帧 36.2%（1187 帧）为 R>60° 旋转大错，粗位姿 100% 远离 GT 最近模板
> （选错模板，模板球面 med 26° 覆盖稀）。单假设 iter_align 从错误盆出发
> 救不回；从池内 top-k 位姿各跑 1 轮 iter_align + 渲染对齐损失择优
> （iG-6DoF 多候选初始化启发），能否把部分选错模板帧拉回正确盆？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_multi.yaml` （champion + solver.iter_align_multi_hypo: 5）|
| Code change | `src/pipeline.py`：iter_align 段多假设分支（top-k 种子各跑 + align_loss 择优）；测试 202 过 |
| Data split | duck 120 帧子集先验证（R>60 失败 148/642）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线）|
| Success line | duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_multi.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_multi/cache \
    --out outputs/exp_multi/results/duck.json
```

## Live Log

- `08-11 14:55`：实现（iter_align_multi_hypo 开关 + top-k 种子 + align_loss 择优），
  202 测试过；文献支撑（iG-6DoF CVPR'25 多候选初始化、6DGS ECCV'24）。
- `08-11 14:59`：队列挂载（等 t-only 完成后自动启动）。
- `08-11 15:55`：启动（duck 120 帧）。
- `08-11 16:26`：96/120 帧滚动 58.33（+10.8）；效率实测 +2%（iter_align
  5 种子 +2.3s，被 refiner 早停省回 ~2.1s——好位姿收敛快）。
- `08-11 16:34`：**出炉 55.83（+8.33）**。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| ADD | 47.50 | 55.83 | **+8.33** | 端到端（align_loss 择优，非 GT oracle）|
| Proj@5px | 81.67 |  |  |  |
| 单帧耗时 | 10.31s | 10.54s | +2% | refiner 早停抵消 |

## Decision

- 结论：`keep`（**多初始假设 iter_align 有效，挑战 3 首正**）
- 原因：
  1. 池内 top-5 位姿各跑 iter_align + 渲染对齐损失择优 → duck +8.33，
     36.2% R>60° 选错模板帧部分救回（多个中间视角种子提供正确盆入口）；
  2. 端到端口径（align_loss 推理可用，非 GT 择优）；效率 +2% 几乎免费；
  3. 待验证：track（时间种子）/ mmr（池多样性）/ fb（失败全解码）能否
     叠加；扩 5 弱物体确认泛化。
- 下一步：multi 扩 5 弱物体全量验证（挂队列尾部）；与 track/mmr/fb 组合
- 产物：`outputs/exp_multi/results/duck.json`

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
