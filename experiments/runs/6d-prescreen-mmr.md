# 6d-prescreen-mmr —— MMR 预筛多样性

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-prescreen-mmr` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 17:18` |
| Finished | `2026-08-11 17:38` |
| Queue row | `experiments/QUEUE.md::6d-prescreen-mmr` |

## Question

> DINOv2 预筛 top-K 模板高度相似（同视角近邻），池内多样性差——候选池
> oracle 显示 duck 池有货但选择倒挂。MMR（λ·sim - (1-λ)·max 模板互斥）
> 重排序能否用多样性换回被漏掉的正确模板？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_mmr.yaml` （champion + matching.prescreen_mmr: 0.7）|
| Code change | `src/detection/localize.py`：`mmr_reorder`（贪心 MMR，top-1 固定最相似）|
| Data split | duck 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（ia 基线）；对照 multi 55.83 |
| Success line | duck ≥ 基线 +3 且无 -3 回退 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_mmr.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_mmr/cache \
    --out outputs/exp_mmr/results/duck.json
```

## Live Log

- `08-11 17:18`：启动（等 track 完成后）。
- `08-11 17:38`：**出炉 duck 46.67**（-0.83 vs 基线）。

## Result

| 指标 | baseline | this run | delta |
|---|---:|---:|---:|
| ADD | 47.50 | 46.67 | **-0.83** |
| Proj@5px | 81.67 | 81.67 | 0 |
| 5cm5° |  | 62.50 |  |

## Decision

- 结论：`drop`（MMR 预筛多样性判负）
- 原因：
  1. 46.67 与基线 47.50 持平微负——用互斥换相似性没有兑现候选池收益
     （6d-gap-oracle 的"duck 池有货"结论不指向预筛多样性的可恢复性）；
  2. 结合 6d-prescreen2 判负：预筛阶段两度尝试（扩 K、多样性重排）均
     无收益——DINOv2 预筛本身不是瓶颈，瓶颈在 MASt3R 对应质量
     （top40 oracle 62.0 vs 端到端 61.2 说明择优几乎无损，池内无好货）。
- 下一步：无（预筛阶段结案）
- 产物：`outputs/exp_mmr/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
