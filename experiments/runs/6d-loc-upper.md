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
| duck | ADD | 33.33 | **39.17** | +5.84 | 34.17 | +0.83 | 定位侧瓶颈 |
| duck | Proj | 81.67 | **85.0** | +3.33 | 78.33 | **-3.33** | |
| duck | 5cm5° | 42.5 | **50.0** | +7.5 | 46.67 | +4.17 | |
| can | ADD | 91.67 | **96.67** | +5.0 | — | — | 定位侧（强物体）|
| cat | ADD | 51.67 | **59.17** | **+7.5** | — | — | 定位侧瓶颈 |
| ape | ADD | 45.0 | 43.33 | **-1.67** | — | — | **匹配侧瓶颈（唯一例外）** |
| holepuncher | ADD | 52.5 | **61.67** | **+9.17** | — | — | 定位侧（最大收益）|

掩码 IoU（mask_crop 全图对齐）：duck fastsam-GT 0.910 / SAM 0.914（几乎相同）；
ape <0.5 IoU 36.67%、holepuncher 41.67%（候选掩码大面积选错）——但收益
与坏掩码占比无关（holepuncher +9.17 vs ape -1.67），ape 例外需匹配侧解释。

结果文件：`outputs/exp_gtmask/results/{duck,can,cat,ape,holepuncher}.json`；
`outputs/exp_sam/results/duck.json`。

## Decision

- 结论：`done`（上界量化完成：**4/5 弱物体定位侧瓶颈**，ape 唯一例外）
- 原因：duck +5.84 / cat +7.5 / can +5.0 / holepuncher **+9.17** 全部来自
  GT 掩码；ape -1.67（匹配侧）；SAM 掩码与 fastsam 几乎同质量（0.914 vs
  0.910），换分割器无收益——检测改进空间 = 掩码/裁剪的"准 GT 化"
- 下一步：6d-weak-objects 分流——duck/cat/holepuncher 检测侧改进
  （候选掩码生成/裁剪），ape 匹配/对应侧；duck 残余 5.0 拆解实验
  （gt_mask+真实检索 / crop 分解）

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
