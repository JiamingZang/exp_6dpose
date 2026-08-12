# 6d-pnp-multisol —— EPro-PnP 式 RANSAC 多 t 解（挑战 2）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-pnp-multisol` |
| Owner | `qoder` |
| Status | `done`（判死）|
| Started | `2026-08-12 10:30`（前提诊断）|
| Finished | `2026-08-12 11:00` |
| Queue row | `experiments/QUEUE.md::6d-pnp-multisol` |

## Question

> 挑战 2：弱纹理小物体 tz 病态（48.5% 失败帧 R≤10° 但 t 错）——PnP
> 是否存在多 t 解（EPro-PnP 动机：错误模型与正确模型在 ε 内点都多）？
> 若有，RANSAC 多假设 + 渲染择优能否提高选对概率？

## Protocol

| 项 | 值 |
|---|---|
| 方法 | 前提诊断（不改管线）：duck 60 帧，top1 模板对应，30 次独立 RANSAC 收集全部假设，逐假设算 ADD |
| Script | `/tmp/pnp_multisol_diag.py`（临时诊断，未入库）|
| Data split | duck 前 60 帧（120 帧子集前段）|
| 判死线 | 解集单解（无多解性）或解集不含好解 |

## Commands

```bash
source env.sh
python /tmp/pnp_multisol_diag.py --obj duck --max-frames 60 --n-ransac 30
# 输出 /tmp/pnp_multisol_diag.json（帧级解集），stdout 汇总
```

## Live Log

- `08-12 10:30`：写诊断脚本（复用 extract_matches + ransac_pnp 循环），
  修 3 处 bug（repo root、add_error 参数序、pix_q 原图系反变换）。
- `08-12 11:00`：**出炉——60/60 帧单解**。
- `08-12 11:3x`：补测 soft aggregation（用户追问挑战 2）——候选 t
  inlier 加权平均 vs 单选 best：坏帧 28 帧胜 17 负 11、改善中位 +5mm
  看似正向，但**候选旋转差中位 20.7°（37/40 帧 >5°）**——不同模板
  PnP 解旋转不一致，"保持 best R + 平均 t"物理不自洽；旋转一致的
  坏帧仅 3/28，信号不可用。soft 聚合同样判死。

## Result

| 指标 | 值 |
|---|---|
| 多解帧（解集唯一 ADD 数 >1）| **0/60**（全部单解）|
| 内点数有变化帧 | 0/60（30 次 RANSAC 内点数完全相同）|
| 解集含好解帧（best_add<10mm）| 11/60 |
| inlier 择优命中解集最优 | 60/60（单解时平凡成立）|
| soft 聚合坏帧胜/负（补测）| 17/11（改善中位 +5mm）|
| 候选旋转差中位（补测）| 20.7°（37/40 帧 >5°）——soft 聚合物理不自洽 |

## Decision

- 结论：`drop`（**前提不成立，判死**）
- 原因：
  1. 同一模板对应下，30 次独立 RANSAC 全部收敛到**同一解**（EPnP
     闭式解 + 内点占比高 → 每次采样选到一致内点 → 确定输出）；
  2. EPro-PnP 的多解性来自端到端可微 soft correspondence（训练侧
     分布），零样本测试侧硬对应管线拿不到——多 t 解前提在测试侧
     不成立；
  3. 多解性唯一来源是**跨模板**（不同模板对应 → 不同解），这正是
     multi 系列已测的种子池——择优指标四档全负（align -4.34 /
     iou -4.00 / inl -3.33）已结案；
  4. **论文素材**：单模板内 PnP 无多解 + inlier 择优 100% 命中解集
     最优 → 择优环节无提升空间，瓶颈确证在候选池生成（跨模板对应
     质量），与 gap-oracle（top40 池内 GT 择优 ≈ 端到端）闭环。
- 下一步：挑战 2 结案；剩余候选方向 6d-ablation-full（论文消融）、
  匹配分辨率提升（long_side 512→更高，候选池生成侧）
- 产物：无（诊断数据 /tmp/pnp_multisol_diag.json）

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
