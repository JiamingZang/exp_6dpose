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
- `08-08 05:10`：gtfg 启动（fastsam bbox + GT 掩码像素；06:01 才真正开始，
  bash 快照加载卡 50 分钟）
- `08-08 06:45`：gtfg 完成 → **19.17/80.0/43.33（-14.16 灾难崩盘）**
- `08-08 06:50`：gtfg 诊断：坏帧（4-5 帧）fastsam bbox 与 GT 完全错位
  （如 000613：fs bbox [149,196,256,286] vs GT [233,249,330,324]）→ GT
  掩码被裁掉（面积 143）→ 对应崩；好帧 bbox 偏 2-6px 也伤 crop
- `08-08 07:00`：gtbc 启动（GT bbox + fastsam 掩码像素，反向交叉）
- `08-08 07:30`：gtbc 完成 → **30.0/83.33/45.0（-3.33）**

## Result

| 配置 | bbox | 掩码像素 | ADD | Proj | 5cm5° | note |
|---|---|---|---:|---:|---:|---|
| 主链 | fastsam | fastsam | 33.33 | 81.67 | 42.5 | 基线 |
| gt_mask | GT | GT | **39.17** | 85.0 | 50.0 | 上界 +5.84 |
| erode3 | 重算 | fs 腐蚀3 | 26.67 | 80.83 | 29.17 | 收缩有害 |
| erode1 | 重算 | fs 腐蚀1 | 30.0 | 81.67 | 40.83 | 仍降 |
| erode1-kb | 固定 | fs 腐蚀1 | 30.83 | 81.67 | 40.83 | bbox 副作用≈0 |
| gtfg | fastsam | **GT** | 19.17 | 80.0 | 43.33 | 掩码被裁，灾难 |
| gtbc | **GT** | fastsam | 30.0 | 83.33 | 45.0 | 掩码与框不匹配 |

结果文件：`outputs/exp_erode{1,,_kb}/results/duck.json`、`outputs/exp_gtfg/results/duck.json`、
`outputs/exp_gtbc/results/duck.json`。

## Decision

- 结论：`done`（**腐蚀与交叉实验全部判负，上界拆解闭环**）
- 原因：①腐蚀方向（1/3px，bbox 重算或固定）全降——"溢出环噪声"假说
  证伪；②双向交叉（GT 掩码+fs bbox / GT bbox+fs 掩码）全崩——掩码与
  bbox 必须同源自洽；③GT 上界 +5.84 拆解：~4 分 = 坏帧候选选择失败
  （fastsam 选错候选，bbox 完全错位，GT 掩码 4-5 帧 IoU<0.5），~2 分 =
  掩码/bbox 边界精度；④形态学后处理与掩码替换均不可复现 GT 收益
- 下一步：坏帧候选选择失败（~4 分）是最大单来源——定位检索（DINOv2 CLS
  从候选池选）或分割候选生成；验证实验：离线统计坏帧里 GT 掩码的 CLS
  分数 vs 选中候选（区分"检索失败"与"候选池缺物体掩码"）

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
