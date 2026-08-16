# 6d-gsrefiner —— 忠实复刻 GS-Refiner 损失（纯结构损失验证）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-gsrefiner` |
| Owner | agent |
| Status | `running` |
| Started | 2026-08-16 |
| Finished | empty |
| Queue row | `experiments/QUEUE.md::6d-gsrefiner` |

## Question

> GS-Pose 的 refiner 收益全在其损失设计（论文 Eq.6：L_gs = L_D-SSIM + L_D-MSSIM，无 L1/掩码项）。我们 v2 抄了优化器骨架却叠加 L1(1.0)+mask(0.5)+dice(0.3)+area（L1 权重最大），refiner 因此净负（回退保护止损）。换回论文的纯结构损失 + lr=5e-3 + 绝对阈值早停 η=1e-4 + 关回退保护后，refiner 能否从"组合效应配角"变成"纯增益"？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_gsrefiner.yaml`（base champion ia + solver.refine_loss_mode: gs_refine + refine_lr 0.005 + refine_early_stop_abs 1e-4 + refine_early_stop_grad_window 0 + refine_fallback_guard false）|
| Code change | `src/gaussian/pose_refiner.py`（loss_mode="gs_refine"：_step_loss 只返回 (1-SSIM)+(1-MS-SSIM)；early_stop_abs 绝对阈值早停）+ `src/pipeline.py`（refine_loss_mode/refine_early_stop_abs 透传 + refine_fallback_guard 开关）|
| Data split | 120 帧 × 5 弱物体（duck/ape/cat/holepuncher/phone）|
| Metrics | ADD(S)@0.1d（主判据）+ Proj@5px + timings |
| Baseline | champion ia 基线：duck 47.50 / ape 59.17 / cat 64.17 / holepuncher 56.67 / phone 77.50，MEAN **61.00**（同粗位姿/iter_align 口径，唯一差异=refiner 损失与保护）|
| Success line | **转正**：MEAN ≥ 62.00（ia 基线 +1.0 噪声带外）且至少 3/5 物体 ≥ 基线 → 忠实复刻成功，考虑全量升级。**判平**：60.00 ≤ MEAN < 62.00 → 损失更换中性，refiner 维持现状。**判负**：MEAN < 60.00 → GS-Refiner 机制在自采 3DGS 管线下不成立（可能因我们的 3DGS 由 LineMod 真实图 onboard、域差结构不同），方向结案 |
| 次级观测 | ① 回退保护关闭的代价：逐帧 guard 前后 ADD 对比（pipeline 已存 R_refined/粗位姿，可离线算 guard 保护帧数）；② 400 步占比与 timing（η 阈值是否实际触发）；③ 与 ia_norefine（refiner 关）对比 refiner 净增益 |

## Commands

```bash
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_gsrefiner.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_gsrefiner/cache \
    --out outputs/exp_gsrefiner/results/
```

## Live Log

- `08-16`：登记。动机（论文深读）：GS-Refiner 损失=纯 (1-SSIM)+(1-MS-SSIM)（Eq.6，消融证明两项即全部）；我们的 v2 叠加 L1(1.0)+mask(0.5)+dice(0.3)+area，L1 权重最大——对 3DGS 与查询的光度域差是噪声梯度（证据链：align_loss 与 ADD 相关性 ~51%、refiner 净负需回退保护、multi-refine 盆底择优 ADD -6.66 但 Proj +9.16）。早停语义也反了：论文等 loss 降到绝对阈值 η=1e-4（真对齐才停），我们等变平（坏盆地平台也停）。

## Result

| 物体 | ia 基线 | gs_refine | Δ |
|---|---:|---:|---:|
| duck | 47.50 |  |  |
| ape | 59.17 |  |  |
| cat | 64.17 |  |  |
| holepuncher | 56.67 |  |  |
| phone | 77.50 |  |  |
| MEAN | 61.00 |  |  |

## Decision

- 结论：`pending`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
