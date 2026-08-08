# 6d-weak-objects —— 候选池全解码收益全物体验证（新口径）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-weak-objects` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 11:40` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-weak-objects` |

## Question

> 旧口径（rng 污染）下全解码比 DINOv2 预筛 +5.0（duck 32.5 vs 27.5）。
> rng-fix 后新口径下，全解码（MASt3R sim 全 80 模板解码取 top-40）在
> 弱物体（duck/ape/cat/holepuncher）上是否仍系统性优于 DINOv2 预筛
> top-40？差距多大、是否值得为它做两阶段预筛（6d-prescreen2）？

## Protocol

| 项 | 值 |
|---|---|
| Config | 全解码：`dense80_depthc_mast3r.yaml`；基线：`dense80_depthc_guided.yaml`（duck 基线 30.83 已由 6d-rng-fix Run A 提供）|
| Code change | `none`（rng-fix 后的新代码，匹配 rng 逐帧确定性）|
| Data split | 120 帧子集 × 4 弱物体（duck/ape/cat/holepuncher）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 30.83/81.67/40.83（新口径，dinov2 预筛）；ape/cat/holepuncher 基线待同口径跑 |
| Success line | 任一弱物体全解码 ADD ≥ 基线 +3（新口径复现候选池收益）|

## Commands

```bash
source env.sh   # 每个 shell 都要（mast3r PYTHONPATH）

# 1) 全解码匹配提取（4 物体，~40 分钟/物体，共 ~3h）
python scripts/analysis/extract_matches.py \
    --config configs/current/dense80_depthc_mast3r.yaml \
    --objects duck ape cat holepuncher --max-frames 120 \
    --matches-dir outputs/matches13_mast3r2

# 2) 从落盘 matches 求解评估（新 cache，逐物体）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_mast3r.yaml \
    --objects <obj> --max-frames 120 \
    --matches-dir outputs/matches13_mast3r2 \
    --cache-dir outputs/exp_weakobj/cache_mast3r2 \
    --out outputs/exp_weakobj/results/<obj>_mast3r2.json

# 3) 基线（dinov2 预筛）同口径：ape/cat/holepuncher 端到端评估
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_guided.yaml \
    --objects <obj> --max-frames 120 \
    --cache-dir outputs/exp_weakobj/cache_base \
    --out outputs/exp_weakobj/results/<obj>_base.json
```

## Live Log

- `08-08 11:40`：开工（6d-rng-fix 结案后首项）
- `08-08 11:40`：启动 4 物体全解码提取（后台，~3h）

## Result

待出。

## Decision

待出。

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
