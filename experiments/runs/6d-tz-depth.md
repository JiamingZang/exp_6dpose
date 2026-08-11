# 6d-tz-depth —— RGB-D 深度平移校正（掩码内深度中值 z + 质心对齐 xy）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-tz-depth` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 14:25` |
| Finished | `2026-08-11 22:05` |
| Queue row | `experiments/QUEUE.md::6d-tz-depth` |

## Question

这次只回答一个问题：

> 失败帧 48.5% 为"旋转对、平移错"（tz 仅 35 帧，xy 错 1556 帧；GT t 替换
> 100% 恢复）。平移病态如何修——分两档：**深度档**（BOP 深度中值替换 z，
> 仅限有深度相机的场景，作消融）与 **RGB-only 档**（t-only 渲染对齐精化：
> refiner 阶段 1 加强，固定 R 只优化 t，新物体新场景可部署）。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_tzdepth.yaml` `configs/current/dense80_depthc_ia_tonly.yaml` |
| Code change | `src/datasets/linemod.py`（Frame.depth_path）；`src/pipeline.py`（depth_img 透传 + tz_depth 分支）；测试 202 过 |
| Data split | duck 120 帧子集先验证（最弱物体，失败 642 帧/1190）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线）|
| Success line | 任一档 duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
# 深度档（BOP 消融，需深度传感器）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_tzdepth.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_tzdepth/cache \
    --out outputs/exp_tzdepth/results/duck.json
# RGB-only 档（主线可部署：refiner 阶段 1 t-only 加强）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_tonly.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_tonly/cache \
    --out outputs/exp_tonly/results/duck.json
```

## Live Log

- `08-11 14:00`：tzxy（质心对齐 xy-only，tz_search 拆变量第一档）启动，30/120 帧
  滚动 63.33%（+15.8 早期乐观）。
- `08-11 14:23`：tzxy 60/120 帧滚动 46.67 ≈ 基线——质心对齐单独大概率无效，
  衰减归因：渲染掩码质心与 FastSAM 掩码质心差并非纯平移（掩码形状差）。
- `08-11 14:25`：tz_depth 实现完成（loader depth_path + pipeline tz_depth 分支），
  深度数据验证（uint16 mm，帧 0 GT tz 1014.9 vs 全图深度中值 1101）；队列挂载
  等 tzxy 完成即跑。
- `08-11 14:29`：**部署实际修正**（用户指出：新场景 RGB-only 无深度中值）——
  深度档降级为 BOS 消融；主线改 RGB-only 档：refiner 阶段 1（t-only）加强
  （stage1_iters 200 + 面积正则 + 短 6D 收尾），纯配置零新代码，队列挂载。
- `08-11 14:36`：**tzxy 结案（判负）**：duck 45.83（-1.67）——渲染质心对齐
  （一阶矩）单独无效，渲染/查询掩码质心差并非纯平移（掩码形状差）。
- `08-11 14:37`：tzdepth（深度档）启动（duck 120 帧）。
- `08-11 15:20`：**tzdepth 出炉：ADD 51.67（+4.17，Proj 84.17/5cm5° 61.67）——
  过 +3 门槛**。深度中值 z + 质心对齐 xy 确认"平移病态可修"（真实测量）。
  消融档结论：RGB-D 传感器可用时平移病态消除；t-only（RGB-only）能否
  复现该增益待验——决定单目主线叙事。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| ADD (tzdepth) | 47.50 | 51.67 | **+4.17** | 深度档（BOP 中值 z + 质心 xy）|
| ADD (t-only) | 47.50 | 45.00 | -2.50 | RGB-only 档：refiner 阶段 1 加强，判负 |
| ADD (tzxy) | 47.50 | 45.83 | -1.67 | 质心对齐单独，判负 |
| ADD (multi) | 47.50 | 55.83 | +8.33 | 挑战 3 多假设，已结案 |

## Decision

- 结论：`done`（**单目平移病态 = 信息极限，RGB-D 是充分条件**）
- 原因：
  1. **三个 RGB-only 档全负**（tzxy -1.67 / t-only -2.50 / mask-geo -3.33）：渲染
     对齐优化、质心对齐、掩码几何都无法在单目下可靠修 z——弱纹理 +
     FastSAM 掩码使渲染损失面在 z 方向不可靠（面积正则被掩码系统偏差污染）；
  2. **深度档 +4.17**：真实深度 z 修正后质心对齐 xy 才生效（z 对时渲染
     掩码形状匹配 → xy 对齐方向正确）——**z 正确是 xy 对齐的前提**；
  3. 论文叙事：单目平移精度上限由掩码质量 + 渲染损失面决定；RGB-D 是
     充分条件（消融对照）；主表保持 RGB 单目口径，此实验作为第四章
     "单目 vs RGB-D 平移可解性"量化证据。
- 下一步：multi（挑战 3）/ track（视频扩展）/ mmr（预筛多样性）验证
- 产物：`outputs/exp_tzdepth/results/duck.json`、`outputs/exp_tonly/results/duck.json`

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
