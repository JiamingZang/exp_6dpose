# 6d-loc-upper —— GT 掩码定位上界（FastSAM 是否瓶颈）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-loc-upper` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-07 22:40` |
| Finished | `<YYYY-MM-DD HH:MM 或 empty>` |
| Queue row | `experiments/QUEUE.md::6d-loc-upper` |

## Question

> FastSAM 分割质量是否是弱物体（duck/ape/cat/holepuncher）的定位瓶颈？
> gt_mask 上界 vs FastSAM 基线的差距，决定分割改进（SAM 对照）是否值得投入。

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_gtmask.yaml`（base: guided + segmenter: gt_mask）、`dense80_depthc_sam.yaml`（segmenter: sam） |
| Code change | `ed10be0`（新增 gtmask/sam 配置；检测器按 segmenter 名分派） |
| Data split | 120 帧子集（subsample linspace，与 guided 基线同口径） |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | guided 冠军基线（120 帧 30k）：duck 33.33/81.67/42.5、can 91.67/95.83/95.83 |
| Success line | SAM 接近 gt_mask 上界（duck 39.17）→ 分割改进可兑现收益 |

## Commands

```bash
python scripts/analysis/extract_matches.py --config configs/current/dense80_depthc_gtmask.yaml \
    --objects duck --max-frames 120 --matches-dir outputs/matches13_gtmask
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_gtmask.yaml \
    --objects duck --matches-dir outputs/matches13_gtmask --max-frames 120 \
    --cache-dir outputs/exp_gtmask/cache --out outputs/exp_gtmask/results/duck.json
# can 同构（matches13_gtmask / exp_gtmask）
python scripts/analysis/extract_matches.py --config configs/current/dense80_depthc_sam.yaml \
    --objects duck --max-frames 120 --matches-dir outputs/matches13_sam
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_sam.yaml \
    --objects duck --matches-dir outputs/matches13_sam --max-frames 120 \
    --cache-dir outputs/exp_sam/cache --out outputs/exp_sam/results/duck.json
```

## Live Log

- `08-07 22:40`：启动 gtmask duck 提取
- `08-07 23:20`：gtmask duck 评估完成
- `08-07 23:35`：gtmask can 提取+评估完成（96.67/95.83/95.83）
- `08-08 00:14`：SAM duck 提取启动（等 can gtmask 串行完成后自动接续）
- `08-08 00:38`：SAM duck 提取完成（120/120，定位失败 0）
- `08-08 00:51`：SAM duck 评估完成 → **34.17/78.33/46.67（判负）**
- `08-08 00:52`：gt_mask 扩展 ape/cat/holepuncher 启动（后台串行）

## Result

| 物体 | 指标 | fastsam 基线 | gt_mask | delta | SAM | SAM delta | note |
|---|---|---:|---:|---:|---:|---:|---|
| duck | ADD | 33.33 | **39.17** | +5.84 | 34.17 | +0.83 | SAM 兑现不了上界 |
| duck | Proj | 81.67 | **85.0** | +3.33 | 78.33 | **-3.33** | SAM 反而回退 |
| duck | 5cm5° | 42.5 | **50.0** | +7.5 | 46.67 | +4.17 | |
| can | ADD | 91.67 | **96.67** | +5.0 | — | — | 强物体也涨 |
| can | Proj | 95.83 | **95.83** | 0 | — | — | |
| can | 5cm5° | 95.83 | **95.83** | 0 | — | — | |

结果文件：`outputs/exp_gtmask/results/{duck,can}.json`；`outputs/exp_sam/results/duck.json`；
gt_mask 扩展：`outputs/exp_gtmask/results/{ape,cat,holepuncher}.json`（08-08 在跑）。

## Decision

- 结论：`keep`（gt_mask 上界确认定位是瓶颈；SAM 对照**判负**）
- 原因：duck +5.84 / can +5.0 全部来自掩码质量；但 SAM ViT-H 只兑现
  +0.83 ADD（Proj 还 -3.33）——缺口在**检测级候选生成**，换更强通用
  分割器不可行；gt_mask 39.17 留作检测改进的验收锚点
- 下一步：gt_mask 扩展 ape/cat/holepuncher 量化各弱物体上界 →
  定 6d-weak-objects 检测侧改进优先级（候选框/掩码生成）

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
