# 6d-match-768 —— 匹配分辨率 768（MASt3R 对应质量直接提升）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-match-768` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 12:02` |
| Finished |  |
| Queue row | `experiments/QUEUE.md::6d-match-768` |

## Question

这次只回答一个问题：

> gap-oracle 结案：**候选池生成（MASt3R 对应质量）是总瓶颈**（top40 池内 GT 择优
> 62.0 ≈ 端到端 61.2）。预筛/解码/择优侧已全部结案——唯一没动的环节是
> MASt3R 本身：输入长边 512 → 768 能否提升对应质量 → 端到端 ADD？
> （1024 已试 OOM 判死 24.8GB；768 冒烟峰值 6.6GB 可行，token 数 2.25×）

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_768.yaml`（champion + matching.image_size: 768）|
| Code change | 无（纯配置，`image_size` 已有配置项）|
| Data split | duck 120 帧子集先验证（最弱物体，ia 基线 47.50）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线，rng-fix 干净口径）|
| Success line | duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_768.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_match768/cache \
    --out outputs/exp_match768/results/duck.json
```

## Live Log

- `08-12 12:05`：内存冒烟通过——768 encode 峰值 3.2GB、成对 decode batch4
  峰值 6.6GB（1024 OOM 24.8GB 的历史判死不适用于 768）。
- `08-12 12:02`：登记入队（running），启动 duck 120 帧。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| ADD | 47.50 |  |  | ia 基线 120 帧 |
| Proj |  |  |  |  |
| 5cm5° |  |  |  |  |

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
