# 6d-ia-multi —— 多初始假设 iter_align（挑战 3：模板离散化/选错模板）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-ia-multi` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-11 14:55` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
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
