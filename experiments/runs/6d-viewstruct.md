# 6d-viewstruct —— fibonacci 视角-roll 结构消融

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-viewstruct` |
| Status | `done`（08-21 05:30 结案） |
| Config | 三档：`configs/experiments/dense80_vs_16x1.yaml` / `dense80_vs_24x1.yaml` / `dense80_vs_24x2.yaml`（base ia）；densify 重渲染零训练 |
| 依赖 | trio 后 3DGS 世代：duck 7000/128v、ape/hp 30k/128v、cat/phone 7000/128v |

## Live Log

- `08-20 20:20`：V1 启动，densify 两个 bug 修复（sys.path parents 索引 + n_inplane 突变位置）。
- `08-21 05:01`：三细胞 × 5 物体全部完成。

## Commands

```bash
bash /tmp/viewstruct_probe.sh   # 备份 24t 旧库 → densify(16×1/24×1/24×2) → 5 弱物体 120 帧子集
```

## Question

> 模板结构（视角数 × roll 数）是否独立于训练配方是精度杠杆？fibonacci 模式下 roll 冗余是"自相似竞争者"还是"真实检索覆盖"？

## Protocol

| 项 | 值 |
|---|---|
| 命令 | `bash /tmp/viewstruct_probe.sh`（24t 旧库备份 .cube8bak → densify → 5 弱物体 120 帧子集）|
| 基线 | 当前 128v 子集：duck 52.50（7000 档）/ ape 60.00 / hp 46.67 / cat 80.00 / phone 75.83 |
| 成功线 | V2（24×1）任一物体 ≥ +3 → 全量；否则判死 |

## Result

| 细胞 | 结构 | duck | ape | hp | cat | phone | MEAN Δ |
|---|---|---|---|---|---|---|---|
| V1 | 16×1（16t）| 47.50 (-5.00) | 49.17 (-10.83) | 45.00 (-1.67) | 72.50 (-7.50) | 80.00 (+4.17) | **-4.17** |
| V2 | 24×1（24t）| 54.17 (+1.67) | 49.17 (-10.83) | 35.83 (-10.84) | 73.33 (-6.67) | 80.00 (+4.17) | **-4.50** |
| V3 | 24×2（48t）| **58.33 (+5.83)** | 56.67 (-3.33) | 41.67 (-5.00) | 81.67 (+1.67) | 79.17 (+3.34) | **+0.50** |

## Decision

- 结论：`done`（**判死**：V1/V2 全负，V3 边际 +0.50 不达采纳线）
- 原因：**roll 冗余是真实检索覆盖不是自相似竞争者**（fibonacci 模式下与 02_n_templates 的 cube8 结论不迁移）；ape/hp 对 roll 覆盖极度敏感（-10.8~-5.0）；视角加密补偿 duck/cat/phone 但不足
- 次级发现：**V3 duck +5.83**（24 视角 × 2 roll）——24 视角对 duck 明确有益（R 近失帧受益），如需可单物体全量确认（~6.5h，未纳入本次采纳）
- 下一步：champion 结构维持 80t（16×5）；待办 #52 fill-norm

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
