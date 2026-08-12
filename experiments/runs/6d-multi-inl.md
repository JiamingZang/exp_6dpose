# 6d-multi-inl —— PnP inlier 几何择优（择优指标系列收官）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-inl` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-12 06:15` |
| Finished | `2026-08-12 10:00` |
| Queue row | `experiments/QUEUE.md::6d-multi-inl` |

## Question

> 择优指标系列（08-12）：align（ape -10）/ gate（挡收益）/ iou（ape
> -14，掩码偏差）全部判负——渲染比较量（光度量 align_loss、掩码量
> mask_iou）在弱纹理物体上系统性不可靠。**PnP inlier** 是 ia 基线选
> best 的同源指标（纯几何、不依赖渲染比较/掩码），能否同时保留池有货
> 收益（duck/cat +8~11）并消除池没货损失（ape/holepuncher/phone）？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_multi_inl.yaml` （multi + iter_align_multi_select: inlier + iter_align_multi_inl_gate: 500）|
| Code change | `src/pipeline.py`：`_iter_align` 返回 (R, t, n_inliers)；multi 择优加 inlier 档（候选须比 best 种子多 500 内点才替换）|
| Data split | 5 弱物体 120 帧（duck/ape/cat/holepuncher/phone）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | ia 基线均值 61.00；multi 56.67；iou 57.00 |
| Success line | 5 弱物体均值 ≥ ia 基线 61.00 且无单物体 -3 回退（相对 ia）→ 扩全量 13 物体 |

## Commands

```bash
source env.sh
for obj in duck ape cat holepuncher phone; do
  python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_multi_inl.yaml \
      --objects $obj --max-frames 120 --cache-dir outputs/exp_multi_inl/cache \
      --out outputs/exp_multi_inl/results/$obj.json
done
```

## Live Log

- `08-12 05:5x`：实现（_iter_align 返回 n_inliers + inlier 择优档），
  202 测试过；配置 multi_inl（inl_gate=500）。
- `08-12 06:15`：启动（iou phone 完成后接力，duck 起跑）。
- `08-12 10:00`：**出炉 5 物体均值 57.67（-3.33 vs ia）——判负**。

## Result

| 物体 | ia 基线 | multi | iou | inl (this run) | delta vs ia |
|---|---:|---:|---:|---:|---:|
| duck | 47.50 | 55.83 | 52.50 | 59.17 | **+11.67** |
| ape | 59.17 | 49.17 | 45.00 | 49.17 | -10.00 |
| cat | 64.17 | 75.00 | 74.17 | 69.17 | +5.00 |
| holepuncher | 56.67 | 40.83 | 50.00 | 44.17 | -12.50 |
| phone | 77.50 | 62.50 | 63.33 | 66.67 | -10.83 |
| MEAN | 61.00 | 56.67 | 57.00 | 57.67 | **-3.33** |

## Decision

- 结论：`drop`（择优指标系列收官，多候选择优方向整体结案）
- 原因：
  1. inlier 是四种指标里最优的（-3.33，duck +11.67 全场最佳）——PnP
     内点与 ADD 相关性确实最强（ia 选 best 同源指标验证）；
  2. 但池没货物体（ape -10 / holepuncher -12.5 / phone -10.83）依旧
     全负——**无论用什么择优指标，池没货时都在坏假设里挑相对不坏**，
     最优策略是保持 inlier 选出的 best（= ia 基线行为）；
  3. **通用结论（四指标证据）**：align -4.34 / iou -4.00 / inl -3.33
     全部无法超过 ia 基线——gap-oracle"候选池生成是总瓶颈"完全验证：
     择优环节（指标选择）不是瓶颈，多候选择优收益上限=池有货收益
     （duck/cat），被池没货损失（ape/holepuncher/phone）净抵消。
- 下一步：多候选择优结案，不再投入；champion 保持 78.07；论文叙事
  = 挑战 3 系统性探索（四指标全负 + 分型证据）
- 产物：`outputs/exp_multi_inl/results/*.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
