# 6d-localt-off —— 定位候选消歧（§3.6.5 备选解码）消融

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-localt-off` |
| Owner | qoder |
| Status | `planned` |
| Started | `empty` |
| Finished | `empty` |
| Queue row | `experiments/QUEUE.md::6d-localt-off` |

## Question

§3.6.5 的定位候选消歧在 top1/top2 分数接近或低置信时对备选候选各跑一遍
MASt3R 匹配（alt 解码）供渲染消歧。实测弱物体 53% 帧触发、单次 3.8s、
平均 +2.0s/帧——该机制的 ADD 贡献从未被量化（6d-det-align 2a 混 3 个
变量未拆出它）。

> 假设：若 loc_alt off ≥ on，机制判负 → champion 可直接提速 2s/帧；
> 若 off < on，机制有效 → 代价如实披露（§5.4）。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_localt_off.yaml`（base=champion `dense80_depthc_ia.yaml` + `detection.loc_alt: false`）|
| Code change | `src/pipeline.py` extract_matches 备选块加 loc_alt 旗标（默认 true 行为不变），本提交 |
| Data split | 120 帧 × 5 弱物体 |
| Metrics | ADD(S)@0.1d（+Proj/5cm5°）|
| Baseline | champion 子集口径 ia 基线（duck 47.50 / ape 59.17 / cat 64.17 / holepuncher 56.67 / phone 77.50，MEAN 61.00）|
| Success line | off 档数字出炉；与 ia 基线差可归因；timings 验证 alt 成本消失 |

## Commands

```bash
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_localt_off.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_localt_off/cache \
    --out outputs/exp_localt_off/results/
```

## Live Log

- `08-14 18:15`：登记。动机数据（ε8 缓存 120 帧统计）：alt_matching 触发率
  duck 30% / ape 72% / cat 28% / holepuncher 79% / phone 57%（均值 53%），
  单次 3.5-4.0s → 平均 +2.0s/帧。链2（post_chain2.sh）gateoff 后自动跑。

## Result

| 物体 | ia 基线 | loc_alt off | Δ |
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
