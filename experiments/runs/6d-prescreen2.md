# 6d-prescreen2 —— 两阶段候选筛选（DINOv2 粗召回 + MASt3R sim 精排）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-prescreen2` |
| Owner | `agent` |
| Status | `done` |
| Started | `2026-08-08 11:50` |
| Finished | `2026-08-08 15:05` |
| Queue row | `experiments/QUEUE.md::6d-prescreen2` |

## Question

> DINOv2 CLS top-40 预筛把正确模板挤出候选池（cand-pool 反转 +5.0）。
> 两阶段筛选——CLS 粗召回 60 个 → 全部解码 → 按 MASt3R sim(m) 精排回
> top-40——能否以 1.5× 解码代价（远小于全解码 2×）兑现候选池收益？

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_p2.yaml`（base: guided + `matching.top_k_prescreen: 60`）|
| Code change | `src/matching/mast3r_wrapper.py`：`decode_template_indices` 加 `decode_k`；`match()` 在 `decode_k > top_k` 时按 sim(m) 精排 |
| Data split | 120 帧子集（duck 先验，后推 ape/cat/holepuncher）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° / matching 耗时 |
| Baseline | duck 新口径：dinov2 top-40 = 30.83（6d-rng-fix）；全解码 = 6d-weak-objects 待出 |
| Success line | duck ADD ≥ 全解码 -1 且 matching 耗时 < 全解码（7.4s）与基线（3.7s）之间 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_p2.yaml \
    --objects duck --max-frames 120 \
    --cache-dir outputs/exp_p2/cache --out outputs/exp_p2/results/duck.json
```

## Live Log

- `08-08 11:50`：代码 + 配置就绪（192 测试通过），等 6d-weak-objects 提取完成
- `08-08 15:05`：duck 验证跑完（GPU 串行等待后）

## Result

duck 120 帧：两阶段 60→40 = **34.17/80.83/42.5**；同口径对照：
基线 dinov2 top-40 = 30.83；全解码 = 37.5。
结果文件：`outputs/exp_p2/results/duck.json`。

## Decision

- 结论：`done`（**判负，未达成功线**）
- 原因：成功线要求 duck ADD ≥ 全解码-1（≥36.5），实测 34.17——两阶段
  只兑现全解码收益的一半（+3.33 vs +6.67）；且 6d-weak-objects 已证
  全解码收益本身不泛化（仅 duck，ape/cat/holepuncher 平/负）——
  候选池类创新点整体被数据否决
- 下一步：粗位姿章创新点不押在预筛上；代码保留（decode_k 机制无害，
  可作消融档）；重心转向 6d-iter-align

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
