# 6d-multi-inl —— PnP inlier 几何择优（择优指标系列收官）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-inl` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 06:15` |
| Finished | — |
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
- `08-12`：5 弱物体逐个出炉。

## Result

| 物体 | ia 基线 | multi | iou | inl (this run) | delta vs ia |
|---|---:|---:|---:|---:|---:|
| duck | 47.50 | 55.83 | 52.50 |  |  |
| ape | 59.17 | 49.17 | 45.00 |  |  |
| cat | 64.17 | 75.00 | 74.17 |  |  |
| holepuncher | 56.67 | 40.83 | 50.00 |  |  |
| phone | 77.50 | 62.50 | 63.33 |  |  |
| MEAN | 61.00 | 56.67 | 57.00 |  |  |

## Decision

- 结论：待跑（running）
- 原因：inlier 是 PnP 重投影内点数——几何量、不依赖渲染/掩码，与
  ia 基线选 best 的指标同源（ia 基线 61.00 就是这么选出来的）。
  inl_gate=500 拒绝域：候选必须显著多内点才替换（防没货池乱换）。
- 下一步：出炉按 success line 判定；通过 → 扩全量 13 物体
- 产物：`outputs/exp_multi_inl/results/*.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
