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
- `08-12 02:01`：**bug 发现**——iou 档 best 种子被丢弃（best_la 初始
  inf 是 align_loss 越小越好语义，`la > inf` 永假，输出退回粗位姿，
  duck 30.83≈无 iter_align 基线）；修复（best 种子无条件接受，
  commit 0c0dd96）。
- `08-12 02:01`：重启后仍 30.83——**cfg_hash 只含配置不含代码版本**，
  代码修复后配置未变 → 读缓存命中 bug 版结果；清 cache 重跑。
- `08-12 03:01`：清 cache 后干净重启（duck 起跑）。
- `08-12`：5 弱物体逐个出炉。

## Result

| 物体 | ia 基线 | multi | iou (this run) | delta vs ia | delta vs multi |
|---|---:|---:|---:|---:|---:|
| duck | 47.50 | 55.83 | 52.50 | +5.00 | -3.33 |
| ape | 59.17 | 49.17 | 45.00 | -14.17 | -4.17 |
| cat | 64.17 | 75.00 | 74.17 | +10.00 | -0.83 |
| holepuncher | 56.67 | 40.83 | 50.00 | -6.67 | +9.17 |
| phone | 77.50 | 62.50 | 待 |  |  |
| 4 物体均值 | 56.88 | 55.21 | 55.42 | -1.46 | +0.21 |

## Decision

- 结论：待 phone 出炉（4 物体：iou 修复 multi 的没货灾难（holepuncher
  +9.17），但 ape 仍 -14.17——FastSAM 掩码偏差在弱纹理物体上污染
  mask_iou，与 6d-mask-geo 结案呼应）
- 原因：
  1. iou 在池有货物体（duck/cat）≈ multi 收益（+5/+10）——几何门控
     保留了正确替换；
  2. holepuncher 修复（-15.84 → -6.67）证明几何门控确实挡住乱换；
  3. ape 异常（-14.17）：54/120 帧替换且大多换坏——ape 掩码 IoU 噪声
     最大（弱纹理 + FastSAM 掩码偏差），iou 基准在 ape 上不可靠；
  4. 通用结论：渲染比较量（align_loss/mask_iou）作为择优基准在弱纹理
     物体上系统性不可靠——唯一未试的是 PnP inlier（纯几何、ia 选 best
     同源），见 6d-multi-inl。
- 下一步：6d-multi-inl（inlier 几何择优）收官对照
- 产物：`outputs/exp_multi_iou/results/*.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
