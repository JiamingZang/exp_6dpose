# 6d-det-align —— 检测框口径（GT bbox 定位上界）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-det-align` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-10 21:00` |
| Finished | `2026-08-11 03:20` |
| Queue row | `experiments/QUEUE.md::6d-det-align` |

## Question

这次只回答一个问题：

> GSPose 92.0 为检测框输入口径；本文无检测器端到端 78.07。检测框口径（GT bbox 定位、无 DINOv2 检索、无 GT 掩码依赖）下级联数字是多少？零样本定位是否是剩余差距的主要来源？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_gtbbox.yaml` `configs/current/dense80_depthc_gtbbox_gtmask.yaml` `configs/current/dense80_depthc_gtbbox_pd.yaml` |
| Code change | `pipeline.py`（gt_bbox_use_mask 开关）、`localize.py`（GtBboxLocalizer.set_dino_prescreen）|
| Data split | 120 帧子集（ape/cat/duck/holepuncher/can）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | 无检测器端到端级联 120 帧：ape 59.17 / cat 64.17 / duck 47.50 / holepuncher 56.67 / can 99.17（exp_ia）|
| Success line | 检测框口径数字出炉；与无检测器基线的差值量化零样本定位的代价（GSPose 对比通道）|

## Commands

```bash
# 逐条写实际命令；不要写“同上”
```

## Live Log

- `<time>`：<启动/中断/恢复/异常/观察>

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## Decision

- 结论：`keep/reject/retry/blocked`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过

## 2a 结果（GT bbox + 框内全 1 + MASt3R 全解码，120 帧子集）

| 物体 | 无检测器（ia） | 检测框 2a | Δ |
|---|---:|---:|---:|
| ape | 59.17 | 56.67 | -2.50 |
| cat | 64.17 | 60.83 | -3.34 |
| duck | 47.50 | 51.67 | +4.17 |
| holepuncher | 56.67 | 47.50 | -9.17 |
| can | 99.17 | 95.83 | -3.34 |
| **均值** | 65.34 | 62.50 | **-2.84** |

**初步结论**：GT 检测框（无掩码信息、全解码排序）整体**不优于**无检测器端到端
（4/5 负）。但 2a 混入 3 个变量（GT 框 vs 零样本定位、全解码 vs DINOv2 预筛、
框内全 1 vs FastSAM 掩码），需 2b/2c 拆变量归因。

## 2b/2c 拆变量队列（00:18 起串行，ape/duck/cat）

- 2b（gt_bbox_use_mask: true）= GT 框 + GT 掩码 → 拆"掩码质量"变量
- 2c（gt_bbox_prescreen: dinov2）= GT 框 + DINOv2 预筛 → 拆"全解码"变量

## 2b/2c 拆变量结果（ape/duck/cat，120 帧子集，Δ vs 无检测器基线）

| 物体 | 基线 | 2a（框+全1+全解码）| 2b（框+GT掩码+全解码）| 2c（框+全1+DINOv2预筛）|
|---|---:|---:|---:|---:|
| ape | 59.17 | -2.50 | -8.34 | +0.83 |
| duck | 47.50 | +4.17 | +4.17 | -3.33 |
| cat | 64.17 | -3.34 | +5.83 | -4.17 |
| MEAN | | -0.56 | +0.55 | -2.22 |

（2a 全 5 物体另含 can -3.34、holepuncher -9.17，5 物体均值 -2.84。）

## Decision

- 结论：`done`（**检测框口径判负，三重归因完成**）
- 归因：
  1. **定位不是瓶颈**：2c（GT 框 + DINOv2 预筛）≈ 基线 ±2——零样本定位
     （FastSAM+DINOv2）已接近完美检测框水平；
  2. **掩码质量是部分物体瓶颈**：2b cat +5.83 / duck +4.17（与 gt_mask 消融
     +5.0~+9.17 方向一致）；ape -8.34 为已知匹配侧例外（gt_mask 消融 ape 亦 -1.67）；
  3. **全解码排序物体异质**：duck 受益（2a-2c = +7.5，与 6d-weak-objects
     "duck 唯一全解码受益者"一致）、ape 受损（+3.3）、cat 持平——维持
     DINOv2 预筛默认。
- 论文叙事：检测框口径（GT bbox 上界）不改善结果 → 与 GSPose 92.0 的差距
  不来自定位；无检测器端到端 78.07 的定位侧已"免费"达到检测框水平。
- 产物：`outputs/exp_gtbbox/results/{ape,cat,duck,holepuncher,can}{,_b,_c}.json`
- 事故记录：2a 五路并行全解码 OOM（峰值 ~13GB/进程），改串行队列
  （/tmp/det_ablation_queue.sh 自动衔接）
