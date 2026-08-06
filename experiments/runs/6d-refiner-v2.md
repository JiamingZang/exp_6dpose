# 6d-refiner-v2 —— 按 GS-Pose + 旧代码思路重做 refiner

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-refiner-v2` |
| Owner | agent |
| Status | `planned` |
| Started | empty |
| Finished | empty |
| Queue row | `experiments/QUEUE.md::6d-refiner-v2` |

## Question

> 当前 refiner 实测净负贡献（回退保护只是止损）。按 GS-Pose（SSIM+MS-SSIM、去 LPIPS、cosine lr 退火、edge_err 择优）+ 旧代码（mask_loss 形状主导、best-loss 回溯、多假设）重做后，精化能否从净负转正，弱项（duck/ape/cat/holepuncher）是否提升？

## Protocol

| 项 | 值 |
|---|---|
| Config | configs/current/dense80_depthc_guided.yaml （改 solver.refine 参数，或新档位）|
| Code change | `src/gaussian/pose_refiner.py`（损失/优化器/调度/多假设），pending |
| Data split | 120 帧子集先行（ape/duck/cat/holepuncher 弱项 + can 强项对照）|
| Metrics | ADD(S)@0.1d / Proj@5px |
| Baseline | 120 帧子集 MEAN 71.55（STATE 轮 10）；裸 PnP 70.97（6d-refine-two-tier）|
| Success line | 120 帧子集 MEAN ≥ 71.55 且弱项任一 +5 |

## 三方对比结论（2026-08-06 已做，见会话记录）

| 维度 | 我们 | 旧代码 MyPose refine.py | GS-Pose inference.py:454-580 |
|---|---|---|---|
| 损失 | L1+SSIM+**LPIPS**+dice | Stage1 **mask_loss×2.0** 主导；Stage2 L1+SSIM+mask | **SSIM+MS-SSIM**（纯光度）|
| 优化器 | Adam 固定 lr=0.02 | LBFGS+strong_wolfe | AdamW+**cosine warmup 退火→0** |
| 步数 | 150 固定无早停 | 20/轮×多轮+早停 | 400+5 步梯度均值早停 |
| 掩码 | 仅 L1 加权 | mask_loss 直接监督形状 | image*mask+非零截断 |
| 多假设 | 仅边缘情况 | 32 种子+凸包 IoU 剪枝+beam | ROT_TOPK+**Sobel edge_err 择优** |
| 防跳出 | 无 | **best-loss 回溯** | lr→0 天然收敛 |

怀疑点：LPIPS 参与优化（两者都不用）、无 lr 退火、无 mask 形状主导、无多假设择优。

## Commands

```bash
# 待写（领实验时填）
```

## Live Log

- `2026-08-06`：排队（用户要求），记录三方对比结论

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## Decision

- 结论：`planned`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
