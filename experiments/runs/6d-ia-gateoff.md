# 6d-ia-gateoff —— iter_align 接受/拒绝门消融

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-ia-gateoff` |
| Owner | qoder |
| Status | `planned` |
| Started | `empty` |
| Finished | `empty` |
| Queue row | `experiments/QUEUE.md::6d-ia-gateoff` |

## Question

_iter_align 的接受/拒绝门用渲染对齐损失（align_loss 变差回退粗位姿，
pipeline.py:1272-1280）。6d-multi-gate 已证明 align_loss 与 ADD 相关性弱
（正确种子优势 <5% 被挡）——同一个信号做迭代门，是否也在挡真收益？

> 假设：若 gate-off ≥ gate-on，门是净阻塞 → 改进机会（换几何门或去掉）；
> 若 gate-off < gate-on，门是必要保护 → §4.2 机制证据。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_ia_gateoff.yaml`（base=champion `dense80_depthc_ia.yaml` + `solver.iter_align_gate: false`）|
| Code change | `src/pipeline.py` _iter_align 加旗标（默认 true，champion 行为不变），本提交 |
| Data split | 120 帧 × 5 弱物体 |
| Metrics | ADD(S)@0.1d（+Proj/5cm5°）|
| Baseline | champion 子集口径（iter_align 2 轮 + 抛光）5 物体均值 61.00（6d-multi-inl 记录的 ia 基线）|
| Success line | gate-off 数字出炉；与基线差可归因（逐物体）|

## Commands

```bash
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_ia_gateoff.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_ia_gateoff/cache \
    --out outputs/exp_ia_gateoff/results/
```

## Live Log

- `08-14 18:30`：登记。旗标默认 true 保持 champion 行为；批处理进程已加载旧代码不受影响。

## Result

| 物体 | gate-on（ia 基线）| gate-off | Δ |
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
