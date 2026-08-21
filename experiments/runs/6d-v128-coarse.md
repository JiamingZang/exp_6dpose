# 6d-v128-coarse —— 论文表2 取证：128v 配方干净粗位姿列

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-v128-coarse` |
| Owner | qoder |
| Status | `running` |
| Started | `2026-08-21 12:40` |
| Finished | `empty` |
| Queue row | `experiments/QUEUE.md::6d-v128-coarse` |

## Question

这次只回答一个问题：

> 表2 的级联-粗位姿 Δ（+8.34）混了 128v 配方升级（64v 粗位姿 vs 128v 级联）。
> 用当前 128v banks 重跑粗位姿全量，得到同配方的干净粗位姿列，Δ 才是
> iter_align 级联的纯增益（#50 论文更新取证）。

## Protocol

| 项 | 值 |
|---|---|
| Config | 粗位姿 `configs/current/dense80_depthc_guided.yaml`（cam/driller `dense80_w1.yaml`）|
| Code change | none（只评测，banks 即当前 champion 状态）|
| Data split | 全量 13 物体（14968 帧）|
| Metrics | ADD(S)@0.1d / Proj@5pix / 5cm5°（逐物体 + 帧加权 MEAN）|
| Baseline | 级联 128v `outputs/exp_full_v128/results/*.json`（duck 用 `exp_duck7000`）——Δ = 级联 − 本评测 |
| Success line | 13 物体全量出炉，Δ 全正（与 64v 时代方向一致）|

## Commands

```bash
# 已在后台跑（xargs -P 2，fresh cache）
printf '%s\n' "ape:dense80_depthc_guided" ... | xargs -P 2 -I {} bash -c '...'
# 输出: outputs/exp_full_v128_coarse/results/<obj>.json
```

## Live Log

- `2026-08-21 12:40`：启动 13 物体粗位姿全量（xargs -P 2），预计 12-14h。

## Result

| 物体 | 粗位姿 128v | 级联 128v | Δ | n |
|---|---:|---:|---:|---:|
| ape |  | 56.91 |  | 1172 |
| benchvise |  | 88.87 |  | 1150 |
| cam |  | 75.46 |  | 1137 |
| can |  | 98.59 |  | 1132 |
| cat |  | 76.50 |  | 1115 |
| driller |  | 97.51 |  | 1124 |
| duck |  | 49.58 |  | 1190 |
| eggbox |  | 98.74 |  | 1189 |
| glue |  | 87.02 |  | 1156 |
| holepuncher |  | 48.76 |  | 1173 |
| iron |  | 97.70 |  | 1088 |
| lamp |  | 95.61 |  | 1163 |
| phone |  | 71.50 |  | 1179 |
| MEAN（帧加权）|  | 79.98 |  | 14968 |

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
