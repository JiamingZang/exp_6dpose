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
- `2026-08-21 09:25`：基线完成——duck 52.50 / ape 60.00 / hp 55.83 / cat 79.17 / phone 75.83（MEAN 64.67；干净 cache `outputs/cache/fillnorm_base`，同口径可比）。启动 fill-norm 对照（`outputs/cache/fillnorm_fn`）。
- `2026-08-21 12:30`：对照完成——duck 37.50 / ape 49.17 / hp 53.33 / cat 79.17 / phone 76.67（MEAN 59.17，**-5.50**）。**判负**：无任一物体 +3。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| duck ADD | 52.50 | 37.50 | **-15.00** | 小物体像素预算最缺，缩小 α≈0.57 雪上加霜 |
| ape ADD | 60.00 | 49.17 | **-10.83** | Proj +1.66（匹配变好）但 ADD 崩 |
| hp ADD | 55.83 | 53.33 | -2.50 | 噪声带 |
| cat ADD | 79.17 | 79.17 | 0.00 | Proj +3.34——尺度对齐的匹配收益真实存在 |
| phone ADD | 75.83 | 76.67 | +0.84 | 噪声带 |
| MEAN | 64.67 | 59.17 | **-5.50** | 判负 |

## Decision

- 结论：`reject`
- 原因：fill-norm（查询缩小到模板尺度）判负——匹配几何确实变好（cat Proj +3.34 / ape Proj +1.66，尺度对齐让 patch 对应更准），但缩小把查询物体像素预算压掉 ~43%，duck/ape 这类本就像素不足的物体 ADD 崩（-15/-10.83）。查询侧"既不能放大（768/superres 已判负）也不能缩小"——512 输入下 bbox+20% 裁剪的天然尺度就是信息最优，闭合查询侧尺度搜索。
- 下一步：可选方向=模板侧填充对齐（模板渲染放大到 72% 填充，不伤查询预算），成本高（重渲染 + 全量评测），未承诺；fill-norm 代码保留作消融档（match_fill_norm 默认 0.0 关闭，零行为影响）。

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
