# 6d-multi-gate —— 多假设择优门控（multi 泛化修复）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-multi-gate` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-12 00:44` |
| Finished | `2026-08-12 01:06`（判负信号中止，duck 66/120 滚动 45.45）|
| Queue row | `experiments/QUEUE.md::6d-multi-gate` |

## Question

> multi（top-5 种子 + align_loss 无条件择优）在池有货物体大正
> （duck +8.33 / cat +10.83）、池没货物体大负（ape -10.00 /
> holepuncher ~-15）——逐帧诊断 ape 95/120 帧换了种子且大部分换错。
> 通用修复：**门控替换**——候选必须比 inlier 选出的 best 种子相对改善
> ≥ gate（5%）才接受，否则保持 best（等价 ia 单假设）。能否保留
> 有货收益同时消除没货损失？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_multigate.yaml` （multi + iter_align_multi_gate: 0.05）|
| Code change | `src/pipeline.py`：multi 分支择优门控（gate>0 时以 best 种子 la 为基准，候选需 ≤ la×(1-gate) 才替换）|
| Data split | 5 弱物体 120 帧（duck/ape/cat/holepuncher/phone）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | multi 各物体数字（duck 55.83 / cat 75.00 / ape 49.17 / holepuncher 待 / phone 待）；ia 基线 duck 47.50 / ape 59.17 / cat 64.17 / holepuncher 56.67 / phone 77.5 |
| Success line | 5 物体均值 ≥ multi 且无单物体 -3 回退（相对 multi）——即"保收益 + 去损失"|

## Commands

```bash
source env.sh
for obj in duck ape cat holepuncher phone; do
  python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_multigate.yaml \
      --objects $obj --max-frames 120 --cache-dir outputs/exp_multigate/cache \
      --out outputs/exp_multigate/results/$obj.json
done
```

## Live Log

- `08-12 00:1x`：multi-ext 泛化失败触发（ape -10.00 / holepuncher -13~15），
  逐帧诊断 ape 95/120 帧换种子；实现门控（iter_align_multi_gate），
  202 测试过；配置 multigate（gate=0.05）。
- `08-12 00:44`：启动（等 phone 完成后）。
- `08-12 01:06`：duck 66/120 滚动 45.45 ≈ 基线 47.50——**光度门控把
  收益也挡掉**（正确种子 align_loss 优势 <5% 但 ADD 优势大）；判负
  信号已足，中止队列切换几何化指标（6d-multi-iou）。

## Result

| 指标 | ia 基线 | multi | gate (this run) | delta vs multi |
|---|---:|---:|---:|---|
| duck（滚动 66/120）| 47.50 | 55.83 | ~45.5 | **-10** |

## Decision

- 结论：`drop`（门控方向对、指标错）
- 原因：拒绝域原则正确（multi-ext 证明无条件换种子净换坏），但拒绝域
  必须建在**几何量**（渲染掩码 IoU）而非**光度量**（align_loss）上——
  弱纹理物体 align_loss 与 ADD 相关性弱：正确种子 align_loss 优势
  <5%（噪声内），门控阈值 0.05 把真收益也挡掉，gate 档结果 ≈ 单假设
  基线。通用教训：**择优/门控基准的选择比门控本身更重要**。
- 下一步：6d-multi-iou（几何拒绝域）——已启动
- 产物：无（中途中止，滚动数据见 Live Log）

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
