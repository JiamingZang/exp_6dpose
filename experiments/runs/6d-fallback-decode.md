# 6d-fallback-decode —— 渲染验证失败帧自适应全解码

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-fallback-decode` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 17:38` |
| Finished | `2026-08-11 19:32` |
| Queue row | `experiments/QUEUE.md::6d-fallback-decode` |

## Question

> 36.2% 失败帧 R>60° 选错模板（正确模板可能在 DINOv2 top-40 池外）；
> duck 全解码历史收益 +6.67（6d-cand-pool 干净口径）。只在渲染验证
> 触发（IoU 低）的失败帧对全部模板重匹配重解，能否低成本兑现全解码收益？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_fb.yaml` （champion + solver.fallback_decode: true）|
| Code change | `src/pipeline.py`：rs_triggered 且 _depth==0 时全解码重匹配 → 递归 _solve（_depth=1）→ align_loss 择优 |
| Data split | duck 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（ia 基线）；对照 multi 55.83 |
| Success line | duck ≥ 基线 +3 且无 -3 回退 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_fb.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_fb/cache \
    --out outputs/exp_fb/results/duck.json
```

## Live Log

- `08-11 17:38`：启动，18 帧崩（`alphas or []` numpy 真值歧义）→ 修复
  （`len(self.bank.alphas)`，fb8cb7c）重启 → 二次崩溃（match 返回值
  解包漏 top_full）→ 修复（e7ee0cd）重启。
- `08-11 19:32`：**出炉 duck 46.67**（-0.83 vs 基线）。

## Result

| 指标 | baseline | this run | delta |
|---|---:|---:|---:|
| ADD | 47.50 | 46.67 | **-0.83** |
| Proj@5px | 81.67 | 81.67 | 0 |
| 5cm5° |  | 61.67 |  |
| 失败帧代价 |  | fallback_decode 20.82s/失败帧 |  |

## Decision

- 结论：`drop`（自适应全解码判负）
- 原因：
  1. 46.67 与基线持平微负——渲染验证触发的失败帧全解码没有兑现
     6d-cand-pool 的全解码收益（+6.67 是全帧统一全解码口径）；
  2. 失败帧 ~22% 付 20.8s/帧（+3.7s 全解码 + 重求解），总耗时显著
     上涨却换不来收益——自适应触发条件（rs_triggered）与全解码的
     收益帧不重合；
  3. 与 6d-prescreen2/6d-prescreen-mmr 一致：预筛/解码侧的候选池
     调整均无收益，瓶颈在 MASt3R 对应质量本身。
- 下一步：无（解码侧结案）
- 产物：`outputs/exp_fb/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
