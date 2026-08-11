# 6d-gap-oracle —— 候选池 vs 选择损失（topk oracle 上界）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-gap-oracle` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 08:02` |
| Finished | `2026-08-11 10:00` |
| Queue row | `experiments/QUEUE.md::6d-gap-oracle` |

## Question

这次只回答一个问题：

> 失败帧 90.5% 内点 >1000 却位姿自洽地错（6d-det-align 后差距分解，08-11）：
> 差距在"候选池里没有好假设"（生成损失）还是在"有好的但选择/验证没挑中"（选择损失）？
> topk oracle 上界（候选池内 GT 择优）量化：top40_best 高而 top1 低 → 选择损失；
> top40_best 也低 → 候选池生成损失。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_topk.yaml` （champion + metrics.topk_best: [1,3,5,40]，见配置注释）|
| Code change | none（复用 evaluate_object 既有 topk 路径）|
| Data split | 120 帧子集 × 5 弱物体（duck/holepuncher/ape/phone/cat，失败帧 83% 集中于此）|
| Metrics | ADD(S)@0.1d / Proj@5px，top1/3/5/40 档（top1 端到端；K>1 为 GT 择优 oracle 上界）|
| Baseline | 同子集 top1 档（端到端，同配置自洽）；全量加权 78.07 作外推参照 |
| Success line | 5 物体 top1/top3/top5/top40 oracle 数字出炉；差值归因到生成 vs 选择 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_topk.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_gap_oracle/cache \
    --out outputs/exp_gap_oracle/results/duck.json
# 5 物体串行队列：/tmp/gap_oracle_queue.sh（2 路并行节显存，峰值 ~12GB/进程）
```

## Live Log

- `08-11 08:02`：启动。背景：差距分解（gap_analysis.py 缓存重放，零 GPU）——
  优化已到顶（iter_align +8.23 / 抛光 +0.10 / tz ≈0；rescue 71 vs broke 56）；
  3282 失败帧 90.5% 内点 >1000 → 自洽错误位姿；6 弱物体占失败 83%。
- `08-11 08:42`：**duck 出炉**：top1 27.50 / top3_best 48.33 / top5_best 58.33 /
  top40_best **65.00**（端到端主数字 47.50，内点择优 + 渲染验证 + iter_align 口径）。
  初读：候选池内最好假设只能到 65.00（+17.5 选择损失 vs 端到端），
  剩余 ~35 分是候选池生成损失（无好假设）——与 6d-cand-pool"duck 候选池是瓶颈"一致。
- `08-11 09:05`：**holepuncher 出炉**：top1 34.17 / top3 49.17 / top5 53.33 /
  top40_best **57.50**（端到端 55.83）。选择损失仅 1.67 —— **候选池生成损失主导**
  （上限 57.5，42.5 分无好假设）。注意端到端 55.83 vs 6d-det-align 基线 56.67
  （差 0.84，待核对是否子集/缓存差异）。
- `08-11 09:30`：**phone 出炉**：top1 51.67 / top3 60.83 / top5 60.83 /
  top40_best **62.50**（端到端 **77.50**）。**池内最优 < 端到端（-15）**——优化
  （择优+iter_align+refiner）在 phone 上净贡献，候选窗口反而没货。
  注意：topk 候选窗口 = 相似度序前 40（含失败候选占名额），端到端 = 内点
  择优 + 完整优化，两者非同一条路径（README §8.4 口径警告）。
- `08-11 09:35`：跨物体瓶颈分型（池最优 top40 vs 端到端）：
  - duck 65.00 vs 47.50：**池有货，选择/优化没拿到（还倒挂 17.5）** → 治择优+优化不破坏
  - holepuncher 57.50 vs 55.83：**池没货**（≈持平）→ 治候选生成（匹配/模板）
  - phone 62.50 vs 77.50：池没货但**优化净赚 +15** → 生成若更好，优化可推更高
- `08-11 09:55`：**ape 出炉**：top1 24.17 / top3 41.67 / top5 49.17 /
  top40_best **53.33**（端到端 **60.00**）——池没货，优化净赚 +6.67。

## Result

| 物体 | 端到端 | top1 | top3_best | top5_best | top40_best | note |
|---|---:|---:|---:|---:|---:|---|
| duck | 47.50 | 27.50 | 48.33 | 58.33 | 65.00 | 池有货，选择/优化倒挂 -17.5 |
| holepuncher | 55.83 | 34.17 | 49.17 | 53.33 | 57.50 | 池没货（≈持平）|
| ape | 60.00 | 24.17 | 41.67 | 49.17 | 53.33 | 池没货，优化净赚 +6.7 |
| phone | 77.50 | 51.67 | 60.83 | 60.83 | 62.50 | 池没货，优化净赚 +15.0 |
| cat | 65.00 | 42.50 | 64.17 | 69.17 | 71.67 | 池有货，倒挂 -6.7 |
| **MEAN** | **61.17** | **36.00** | **52.83** | **58.17** | **62.00** | 池内 GT 择优仅 +0.8 |

（120 帧子集，dense80_depthc_ia_topk；top1 为相似度序窗口第 1 名 = 历史对照
口径非端到端；端到端 = 内点择优 + 渲染验证 + iter_align + 抛光完整管线。）

## Decision

- 结论：`done`（**候选池整体无货，瓶颈分型完成**）
- 原因：
  1. **top40 池内 GT 择优均值 62.0 ≈ 端到端 61.2**（+0.8）——即使择优完美，
     5 弱物体也拿不到更多；**候选池生成（匹配对应质量）是总瓶颈**；
  2. **物体分型**：duck/cat 池有货但选择/优化倒挂（-17.5/-6.7）→ 排序/择优
     环节损失；holepuncher/ape/phone 池没货但优化净赚（+1.7/+6.7/+15）→
     优化已把池榨干，池生成才是上限；
  3. 相似度序窗口 top1 均值仅 36.0 —— 与主路径内点序差异巨大，历史对照
     口径确认不可作端到端参照（README §8.4）；
  4. 与全量差距分解呼应：失败帧 48.5% 纯 t 错（R 对 t 错）→ 掩码几何平移
     候选（6d-mask-geo）从两个方向对症：duck/cat 补选择池、ape/phone 补
     生成池（无对应依赖的平移）。
- 下一步：6d-mask-geo 验证（duck 120 帧已排队，10:00 启动）；有效扩 5 弱物体
- 产物：`outputs/exp_gap_oracle/results/{duck,holepuncher,ape,phone,cat}.json`

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
