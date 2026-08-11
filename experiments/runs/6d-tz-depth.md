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
> 100% 恢复）。掩码几何（面积比/质心反投影）已判负。BOP 深度图是唯一
> 无假设的测量——掩码内深度中值替换 z + 渲染质心对齐 xy，能否补上
> PnP 平移病态？（文献启发：FoundationPose 平移初始化用深度图中值）

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_tzdepth.yaml` （champion + tz_search: true + tz_depth: true + tz_search_xy_only: true）|
| Code change | `src/datasets/linemod.py`（Frame.depth_path）；`src/pipeline.py`（depth_img 透传 + tz_depth 分支）；测试 202 过 |
| Data split | duck 120 帧子集先验证（最弱物体，失败 642 帧/1190）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线）|
| Success line | duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体（弱纹理主战场）|

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_tzdepth.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_tzdepth/cache \
    --out outputs/exp_tzdepth/results/duck.json
```

## Live Log

- `08-11 14:00`：tzxy（质心对齐 xy-only，tz_search 拆变量第一档）启动，30/120 帧
  滚动 63.33%（+15.8 早期乐观）。
- `08-11 14:23`：tzxy 60/120 帧滚动 46.67 ≈ 基线——质心对齐单独大概率无效，
  衰减归因：渲染掩码质心与 FastSAM 掩码质心差并非纯平移（掩码形状差）。
- `08-11 14:25`：tz_depth 实现完成（loader depth_path + pipeline tz_depth 分支），
  深度数据验证（uint16 mm，帧 0 GT tz 1014.9 vs 全图深度中值 1101）；队列挂载
  等 tzxy 完成即跑。

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
