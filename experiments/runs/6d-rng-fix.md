# 6d-rng-fix —— 修复评估 rng 流污染（逐帧确定性种子）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-rng-fix` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 10:20` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-rng-fix` |

## Question

> evaluate 逐帧 solve 消耗 pipeline self.rng（seed 0）全局流，cache 命中的帧跳过
> solve 不消耗 rng → 不同 cache 状态 = 不同 rng 流 = 120 帧子集抖动 ±6 分。
> 改为每帧确定性种子（种子只依赖帧号）后，结果是否与 cache 状态无关、可复现？

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_guided.yaml`（champion，验证用 duck）|
| Code change | `src/pipeline.py`：`_frame_rng()` 辅助 + `extract_matches/estimate/_solve` 接 `frame_id`，评估循环传 `fr.frame_id` |
| Data split | 120 帧子集（duck）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | 事故记录：同配置同 cache 不同状态 33.33 vs 27.5（±6 分）|
| Success line | ① 新 cache 全量求解与② 半满 cache（一半帧命中）两次运行数字完全一致 |

## Commands

```bash
# 复现事故（改前验证非必须，历史已有记录）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_guided.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_rngfix/cache_a \
    --out outputs/exp_rngfix/results/duck_a.json
# 构造半满 cache：只留偶数帧记录（脚本 scripts/analysis/... 或内联 python）
# 第二次运行：另一半帧走完整管线
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_guided.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_rngfix/cache_a_half \
    --out outputs/exp_rngfix/results/duck_b.json
# 断言 duck_a.json == duck_b.json（三指标逐帧一致）
```

## Live Log

- `08-08 10:20`：开工。自检通过（192 passed / check_state OK）
- `08-08 10:25`：实现修复：`_frame_rng()` 辅助 + `extract_matches/estimate/_solve`
  接 `frame_id`，评估循环/`extract_matches.py`/`run_speed.py` 传 `fr.frame_id`
- `08-08 10:35`：Run A 首跑因缺 mast3r PYTHONPATH 失败，补 env 后重跑
- `08-08 11:00`：Run A 完成（全空 cache，120 帧全部现解）
- `08-08 11:05`：构造半满 cache（仅留偶数帧 56 条记录）
- `08-08 11:25`：Run B 完成（56 帧命中 + 64 帧现解）

## Result

| 指标 | Run A（全空 cache）| Run B（半满 cache）| 一致 |
|---|---:|---:|---|
| ADD | 30.83 | 30.83 | ✓ |
| Proj | 81.67 | 81.67 | ✓ |
| 5cm5° | 40.83 | 40.83 | ✓ |

结果文件：`outputs/exp_rngfix/results/duck_a.json`、`duck_b.json`。
duck 120 帧**新口径干净基线 = 30.83/81.67/40.83**（rng 流从全局改逐帧，
种子变了数字整体平移，不可与历史任何数字直接比；从此可复现）。

## Decision

- 结论：`done`（修复生效）
- 原因：rng 从"全局流逐帧消耗"改为"每帧 `default_rng(seed + frame_id)`"，
  种子只依赖帧号 → cache 命中跳帧不再影响任何其他帧的随机序列；
  半满/全空 cache 两种状态逐帧一致（旧代码抖动 ±6 分）
- 下一步：历史 120 帧子集数字全部作废需干净复跑；6d-weak-objects
  （全解码全物体验证）用新口径重提取重跑

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
