# 6d-fillnorm —— 查询-模板填充比例归一化

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-fillnorm` |
| Owner | qoder |
| Status | `running` |
| Started | `2026-08-17 22:30` |
| Finished | `empty` |
| Queue row | `experiments/QUEUE.md::6d-fillnorm` |

## Question

这次只回答一个问题：

> 模板库物体填充约 41%（radius=2.5·diag, FOV 40°），查询 bbox+20% 裁剪
> 填充约 72%（实测恒定，与 1.75× 不匹配）——MASt3R patch 匹配在
> ~1.75× 查询/模板尺度差下工作。把查询裁剪缩放到目标填充
> （match_fill_norm=0.41，α≈0.57 常数修正）能否提升端到端 ADD？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_fillnorm.yaml`（base `configs/current/dense80_depthc_ia.yaml` + matching.match_fill_norm: 0.41）|
| Code change | `src/pipeline.py`：extract_matches 填充分块（sr 块后，α 并入 s_leg 链）；ex 新增 fill_scale 键；新增 `_apply_fill_scale`（K_crop 前两行乘 α，第三行不动）；11 处 K_crop 重建点接入。`tests/test_fill_norm.py` 5 例（坐标闭环/投影等价/identity）|
| Data split | 120 帧子集（弱 5 物体：duck/ape/hp/cat/phone，champion banks 不动）|
| Metrics | ADD（含 refine，帧级 mean）|
| Baseline | 同口径干净复跑：`dense80_depthc_ia.yaml` 同物体同 120 帧（历史子集数字受 cache/rng 污染事故影响，必须重跑对照）|
| Success line | 任一物体 ADD ≥ 基线 +3 → 全量验证；无 +3 判负（填充分均匀不成立）|

## Commands

```bash
# 基线（干净 cache，与 fill-norm 同口径可比）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia.yaml \
  --objects duck ape holepuncher cat phone --max-frames 120 \
  --cache-dir outputs/cache/fillnorm_base

# fill-norm（干净 cache）
python scripts/eval/run_linemod.py --config configs/experiments/dense80_fillnorm.yaml \
  --objects duck ape holepuncher cat phone --max-frames 120 \
  --cache-dir outputs/cache/fillnorm_fn

# 全量（若子集任一物体 +3）
python scripts/eval/run_linemod.py --config configs/experiments/dense80_fillnorm.yaml
```

## Live Log

- `2026-08-17 22:30`：登记。pipeline 修改完成（fill-norm 块 + fill_scale + 11 处 K_crop 接入），`tests/test_fill_norm.py` 5/5 过，全量 pytest 222 passed + 4 GPU skip 无回归。坐标链审计：主 PnP 走 back_to_original_pixels（s_leg 乘 α 自动吸收）；guided_local_matching 用 K_crop（裁剪系）+ sxy（匹配区），K_crop 乘 α 后投影自洽；align_loss/mask_iou/refine 全部裁剪系，K_crop 乘 α 保持投影等价。
- `2026-08-17 22:35`：启动基线评测（5 物体 × 120 帧）。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| duck ADD |  |  |  |  |
| ape ADD |  |  |  |  |
| hp ADD |  |  |  |  |
| cat ADD |  |  |  |  |
| phone ADD |  |  |  |  |

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
