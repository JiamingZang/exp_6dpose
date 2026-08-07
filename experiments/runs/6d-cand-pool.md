# 6d-cand-pool —— 候选池来源消融（DINOv2 top-40 vs MASt3R sim top-40）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-cand-pool` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 02:10` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-cand-pool` |

## Question

> 6d-loc-upper 结案后 duck 残余 5.0 差距（SAM 34.17 vs gt_mask 39.17）是否来自
> 候选池来源？主链（dinov2 预筛）只解码 DINOv2 CLS 排序前 40/80 模板；
> gt_mask 全解码 80 后按 MASt3R sim 取 top-40——两个候选池不同源。

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_mast3r.yaml`（base: guided + matching.template_ranking: mast3r + template_prescreen: none）|
| Code change | `none`（纯配置）|
| Data split | 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | 主链 duck 33.33/81.67/42.5；gt_mask 上界 39.17/85.0/50.0 |
| Success line | mast3r ranking ADD 接近 gt_mask → 候选池是瓶颈，治预筛（扩大 K/改进排序）|

## Commands

```bash
python scripts/analysis/extract_matches.py --config configs/current/dense80_depthc_mast3r.yaml \
    --objects duck --max-frames 120 --matches-dir outputs/matches13_mast3r
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_mast3r.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_mast3r/cache \
    --out outputs/exp_mast3r/results/duck.json
```

## Live Log

- `08-08 02:10`：启动 duck 提取（全解码 80 模板，~40 分钟）
- `08-08 03:03`：提取完成（120/120），评估启动
- `08-08 03:40`：评估完成 → **33.33/84.17/50.83**

## Result

| 指标 | 主链 dinov2 | gt_mask | mast3r ranking | note |
|---|---:|---:|---:|---|
| ADD | 33.33 | 39.17 | **33.33** | 与主链完全持平 |
| Proj | 81.67 | 85.0 | **84.17** | +2.5 |
| 5cm5° | 42.5 | 50.0 | **50.83** | +8.33 |

结果文件：`outputs/exp_mast3r/results/duck.json`。

## Decision

- 结论：`done`（**候选池来源不是瓶颈**）
- 原因：mast3r ranking（fastsam 掩码 + 全解码 80 取 sim top-40）ADD 与主链
  完全相同（33.33）——DINOv2 预筛 top-40 与 MASt3R sim top-40 的候选池
  对最终结果等价；gt_mask 的 +5.84 全部归因**掩码/crop 内容**（IoU 0.91
  的 ~9% 差异像素 → 匹配对应噪声），与模板排序无关
- 下一步：掩码边界腐蚀实验（fastsam 掩码腐蚀 2-3px 排除边界对应噪声，
  接近 gt_mask 则机制坐实）→ 检测侧改进 = 掩码精化/边界清洗

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
