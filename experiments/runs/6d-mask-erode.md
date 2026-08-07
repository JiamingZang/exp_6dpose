# 6d-mask-erode —— 掩码边界处理机制验证（腐蚀 1/3px）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-mask-erode` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 03:50` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-mask-erode` |

## Question

> 6d-cand-pool 结案：gt_mask +5.84 全部归因掩码/crop 内容。FastSAM 掩码
> 系统性比 GT 大 2-8%（溢出背景环）且有效对应少 11%（123k vs 137k）——
> 掩码边界腐蚀能否通过去掉溢出环恢复对应供给、接近 gt_mask？

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_erode.yaml`（base: guided + detection.mask_erode: 3→1）|
| Code change | `a224f04`（pipeline.py mask_erode：腐蚀 + bbox 重算，GT 掩码路线同构）|
| Data split | 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | 主链 duck 33.33/81.67/42.5；gt_mask 39.17/85.0/50.0 |
| Success line | 腐蚀后 ADD 回升接近 gt_mask → 边界溢出环是机制，掩码后处理可落地 |

## Commands

```bash
python scripts/analysis/extract_matches.py --config configs/current/dense80_depthc_erode.yaml \
    --objects duck --max-frames 120 --matches-dir outputs/matches13_erode{,1}
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_erode.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_erode{,1}/cache \
    --out outputs/exp_erode{,1}/results/duck.json
```

## Live Log

- `08-08 03:50`：erode3 启动
- `08-08 04:05`：erode3 完成 → **26.67/80.83/29.17（-6.66 崩盘）**
- `08-08 04:10`：对应质量离线分析（GT 掩码有效对应多 11%：137k vs 123k；
  sim 0.985 vs 0.984）→ 假说修正：溢出环 → 对应损失；3px 腐蚀矫枉过正
- `08-08 04:12`：erode1 启动（甜点档：去溢出环、保真实前景）
- `08-08 04:40`：erode1 完成 → **30.0/81.67/40.83（-3.33 仍降）**
- `08-08 04:45`：erode1-kb 启动（固定 bbox 隔离 bbox 变紧副作用）
- `08-08 05:05`：erode1-kb 完成 → **30.83/81.67/40.83（几乎相同）**
- `08-08 05:10`：gtfg 交叉实验启动（fastsam bbox + GT 掩码像素）

## Result

| 指标 | 主链 | erode3 | erode1 | erode1-kb | gtfg | gt_mask | note |
|---|---:|---:|---:|---:|---:|---:|---|
| ADD | 33.33 | 26.67 | 30.0 | 30.83 | 待出 | 39.17 | 腐蚀方向全负 |
| Proj | 81.67 | 80.83 | 81.67 | 81.67 | 待出 | 85.0 | |
| 5cm5° | 42.5 | 29.17 | 40.83 | 40.83 | 待出 | 50.0 | |

结果文件：`outputs/exp_erode{1,,_kb}/results/duck.json`、`outputs/exp_gtfg/results/duck.json`（待出）。

## Decision

- 结论：`running`（腐蚀三档全判负；交叉实验最后拆解）
- 原因：面积比反证欠分割（fastsam/GT 1.02-1.08 偏大）；对应数显示 GT 多
  11% 有效对应；但腐蚀 1/3px（bbox 重算或固定）均降——掩码收缩本身有害，
  bbox 变紧副作用可忽略（30.0 vs 30.83）→ "溢出环噪声"假说证伪；
  GT 收益 = 掩码形状保真（精确贴合），简单形态学后处理不可复现
- 下一步：gtfg（fastsam bbox + GT 掩码像素）→ 拆"掩码像素"与"crop 内容"；
  若 gtfg ≈ 39.17 → 掩码像素是全部贡献（需掩码精化路线）；若 ≈ 33 →
  crop 内容是关键（bbox 精化路线）

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
