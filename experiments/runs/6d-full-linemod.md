# 6d-full-linemod —— 全量 13 物体评估（回退保护管线）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-full-linemod` |
| Owner | agent |
| Status | `done` |
| Started | `2026-08-05` |
| Finished | `2026-08-07 05:05` |
| Queue row | `experiments/QUEUE.md::6d-full-linemod` |

## Question

> 120 帧子集 MEAN 71.55（回退保护）能否在全量 LineMod 13 物体上保持？全量口径的 ADD(S)/Proj/5cm5° 是多少，能否给出论文可直接引用的表？

## Protocol

| 项 | 值 |
|---|---|
| Config | configs/current/dense80_depthc_guided.yaml （回退保护）、configs/current/dense80_w1.yaml （cam/driller）|
| Code change | c553549（回滚 evaluate_object 路径拼接）、39d949f |
| Data split | 全量 eval 帧（exclude_refs + n_ref=64） |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | 120 帧子集 MEAN 71.55（docs/STATE.md） |
| Success line | 全量 13 物体完成，mean ADD/Proj/5cm5° 入 STATE |

## Commands

```bash
# 提取（已完成，除 iron/lamp/phone 补跑中）
python scripts/analysis/extract_matches.py --config configs/current/dense80_depthc_guided.yaml --objects <obj> --matches-dir outputs/<matches13_*>

# 评估（每物体一个进程，xargs -P 2）
bash /tmp/eval7.sh      # ape/benchvise/cam/can/cat/driller/duck（各物体用各自最佳 bank）
bash /tmp/eval_batchA.sh  # eggbox/glue/holepuncher

# 汇总
python scripts/eval/summarize13.py --results outputs/exp_full/results --out ...
```

## Live Log

- `08-05`：全量提取推进，10/13 物体完成（真实 eval 帧数：ape 1172/benchvise 1150/cam 1137/can 1132/cat 1115/driller 1124/duck 1190/eggbox 1189/glue 1156/holepuncher 1173）
- `08-05~06`：iron/lamp/phone 提取与评估并行时 GPU OOM 中断，各只完成 170/242/256 帧
- `08-06 09:07-09:09`：评估批跑崩——evaluate_object 双重 obj 路径 bug（`matches13_w1/cam/cam/...`）→ FileNotFoundError；已回滚（c553549）
- `08-06 09:19`：重启 eval7.sh（xargs -P2）+ 批 A；批 A 因坏 npz 再崩
- `08-06 09:30-10:00`：**发现 matches 数据损坏**（磁盘满/中断写入的截断 npz）：
  - eggbox/000062（缺 seg）、glue/000081、holepuncher/000002（zip 结构损坏）
  - **cat 399/1747 个文件损坏**（截断/CRC/解压错误，全帧范围分布）
  - 其余 9 物体完整性扫描全部干净（含 ape/benchvise/cam/can/driller/duck/iron/lamp/phone）
  - 根因：提取脚本 `if npz.exists(): continue` 跳过已有文件，截断文件永久带伤；损坏源为提取期磁盘 100% 满或进程被杀时的半写
  - 修复：删除 399+3 个坏文件，`dense80_depthc_b2fix.yaml`（batch_size=2，与评估并行省显存）补提取，全部完成（eggbox 1189/1189、glue 1156/1156、holepuncher 1173/1173、cat 1115/1115，重扫 0 坏）
- `08-06`：eval7.sh 评估中（ape ~775+/1172、benchvise ~600+/1150，约 6 帧/分钟/进程）；批 A 待 cat 补提取完成后重启

## Result

| 指标 | baseline（120帧子集） | this run（全量） | delta | note |
|---|---:|---:|---:|---|
| mean ADD(S)@0.1d | 71.55 | **69.74** | -1.81 | 14968 帧，正常衰减 |
| mean Proj@5px | — | **83.77** |  |  |
| mean 5cm5° | — | **68.69** |  |  |

逐物体（全量 ADD）：ape 42.49 / benchvise 79.65 / cam 63.68 / can 92.58 /
cat 51.30 / driller 90.57 / duck 32.27 / eggbox 95.63 / glue 75.52 /
holepuncher 44.50 / iron 87.59 / lamp 91.83 / phone 61.49。
外部对照：GSPose 92.0（YOLOv5 框口径）差 22.3；can/lamp 单项已接近。

## Decision

- 结论：`done`
- 原因：全量 13 物体完成，数字入 STATE（论文主表可引用）
- 下一步：`6d-refiner-v2`（重做 refiner，最大提升杠杆）

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
