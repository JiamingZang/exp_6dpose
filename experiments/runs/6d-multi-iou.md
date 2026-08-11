# 6d-multi-iou —— 择优指标几何化（渲染掩码 IoU 拒绝域）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-iou` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 01:06` |
| Finished | — |
| Queue row | `experiments/QUEUE.md::6d-multi-iou` |

## Question

> multi 泛化失败的根因 = 择优/门控基准选错（光度量 align_loss 与 ADD
> 相关性弱）。把基准换成**几何量**（渲染掩码 IoU）：候选 iou 须 >
> inlier 选出的 best 种子 iou + 0.02 才替换（几何拒绝域）。能否同时
> 保留池有货收益（duck/cat +8~11）并消除池没货损失（ape/holepuncher/
> phone -10~-15）？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_multi_iou.yaml` （multi + iter_align_multi_select: iou + iter_align_multi_iou_gate: 0.02）|
| Code change | `src/pipeline.py`：multi 分支择优指标可配置（align|iou）；iou 档用 `_verifier.mask_iou`（~30ms/次渲染）做择优与门控 |
| Data split | 5 弱物体 120 帧（duck/ape/cat/holepuncher/phone）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | ia 基线均值 61.00（duck 47.50 / ape 59.17 / cat 64.17 / holepuncher 56.67 / phone 77.50）；multi 均值 56.67 |
| Success line | 5 弱物体均值 ≥ ia 基线 61.00 且无单物体 -3 回退（相对 ia）→ 扩全量 13 物体 |

## Commands

```bash
source env.sh
for obj in duck ape cat holepuncher phone; do
  python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_multi_iou.yaml \
      --objects $obj --max-frames 120 --cache-dir outputs/exp_multi_iou/cache \
      --out outputs/exp_multi_iou/results/$obj.json
done
```

## Live Log

- `08-12 01:06`：启动（duck 起跑）。动机链：multi-ext 5 弱物体均值
  -4.34（分型实锤）→ gate 光度门控判负（duck 滚动 45.45≈基线，正确
  种子 align_loss 优势 <5% 被挡）→ 指标几何化。
- `08-12`：5 弱物体逐个出炉。

## Result

| 物体 | ia 基线 | multi | iou (this run) | delta vs ia |
|---|---:|---:|---:|---:|
| duck | 47.50 | 55.83 |  |  |
| ape | 59.17 | 49.17 |  |  |
| cat | 64.17 | 75.00 |  |  |
| holepuncher | 56.67 | 40.83 |  |  |
| phone | 77.50 | 62.50 |  |  |
| MEAN | 61.00 | 56.67 |  |  |

## Decision

- 结论：待跑（running）
- 原因：通用原则——择优/门控基准必须选与目标（ADD）相关性强的量；
  渲染掩码 IoU 是几何量（形状匹配），弱纹理物体上比光度 align_loss
  鲁棒。iou_gate=0.02 拒绝域防没货池乱换，几何优势保池有货收益。
- 下一步：出炉按 success line 判定；通过 → 扩全量 13 物体
- 产物：`outputs/exp_multi_iou/results/*.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
