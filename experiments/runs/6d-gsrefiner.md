# 6d-gsrefiner —— 忠实复刻 GS-Refiner 损失（纯结构损失验证）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-gsrefiner` |
| Owner | agent |
| Status | `done` |
| Started | 2026-08-16 |
| Finished | 2026-08-17 01:15 |
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
- `08-17 00:15`：**先简单验证（用户要求）→ 假设证伪，全量 120 帧取消**。40 帧 duck 逐帧双损失对照（同一粗位姿）：旧损失精化改善 6 / 恶化 22 / 持平 12；**忠实 GS-Refiner 损失改善 4 / 恶化 28 / 持平 8（更差）**；粗位姿处两种损失与 ADD 误差均负相关（gs_loss r=-0.16 / align_loss r=-0.09），坏帧渲染-查询一致性反而更高——**"自洽地错"是损失面本身的主流，换损失救不了**。
- `08-17 01:10`：**尺度假设证伪（loss_downscale=4 变体）**：改善 2 / 恶化 26——降采样比较域不救。且核查 crop 实为**原分辨率**（duck 物体 ~55×48px、裁剪 ~66×58px），与 GS-Pose 全图物体尺度（~50-150px）本就同量级，尺度差异前提不成立。
- `08-17 01:15`：**结案（判负，先验证未跑全量）**。按粗位姿误差分组（好 <10.4mm n=25 / 中 10-30 n=13 / 大 >30 n=2）：**三个误差带内旧/gs/gs4x 全部净负，无一改善带**（好帧 gs 改善2/恶化18；中误差 2/8；大误差 0/2）。机制结论：渲染比较优化在本管线不可复现——损失面与位姿误差解耦（负相关），任何损失变体都救不了；GS-Pose 的 +36.5 refiner 增益来自其弱 init（55.5，旋转主导大误差，轮廓信号可见可修），我们的级联粗位姿已把旋转误差处理掉，剩余单目平移病态对任何 RGB 损失不可见（与 tz-depth +4.17 / mask-geo / tz_search 结案闭环）。

## Result

**先简单验证（40 帧 duck，同一粗位姿，唯一差异=refiner），未跑全量 120 帧：**

| 损失 | 改善 | 恶化 | 持平 | 命中(ADD@0.1d) |
|---|---:|---:|---:|---:|
| 粗位姿（入口） | — | — | — | 25/40 |
| 旧损失 + 回退保护（champion） | 6 | 22 | 12 | 24/40 |
| 忠实 GS-Refiner（gs_refine） | 4 | 28 | 8 | 25/40 |
| gs_refine + 4× 降采样 | 2 | 26 | 12 | 24/40 |

粗位姿处损失-误差相关性：gs_loss r=-0.16 / align_loss r=-0.09（均负）。

## Decision

- 结论：`dead`（先简单验证，未跑全量 120 帧——验证已充分证伪机制）
- 原因：忠实 GS-Refiner 损失 40 帧 duck 判负（改善 4 / 恶化 28，比旧损失还差）；两个损失变体（纯结构 / 4× 降采样）在三个误差带全部净负；损失-误差相关性为负（gs_loss r=-0.16）——"自洽地错"是渲染比较损失面的系统性属性，不是损失函数选错。GS-Pose 的 refiner 增益前提（损失面与位姿误差相关）在本管线不成立
- 下一步：refiner 方向整体结案（旧损失+回退保护保留为 champion 组件，其净贡献≈0 靠保护止损）；"与 GS-Pose 差距"回填到两级解释：① 他们的 +36.5 来自弱 init（55.5）的大旋转误差被轮廓信号修正，我们的 iter_align 级联已吃掉旋转误差；② 剩余单目平移病态对任何 RGB 渲染比较不可见（tz-depth 深度档 +4.17 是唯一有效信号）。论文 §5.3.1 补此机制结论

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
