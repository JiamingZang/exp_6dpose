# 6d-tz-depth —— RGB-D 深度平移校正（掩码内深度中值 z + 质心对齐 xy）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-tz-depth` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-11 14:25` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
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
