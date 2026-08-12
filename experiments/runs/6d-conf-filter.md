# 6d-conf-filter —— MASt3R desc 置信度过滤（唯一未用信号）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-conf-filter` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 14:40` |
| Finished |  |
| Queue row | `experiments/QUEUE.md::6d-conf-filter` |

## Question

这次只回答一个问题：

> 检索拆解闭环（正确模板在池内 96.7-100%）后，瓶颈 = MASt3R 稠密对应
> 质量。管线从未消费 MASt3R 自带 desc_conf（two_confs 头，exp 输出）——
> 探针（/tmp/conf_probe.py，10 帧 duck，GT 投影 5px 判定）显示**好/坏对应
> conf 均值差 +0.18~0.38，10/10 帧分离**。过滤低置信对应能否提升端到端 ADD？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_conf.yaml` （champion + matching.conf_tau: 1.5）|
| Code change | `src/matching/mast3r_wrapper.py`（_decode_batch 输出 conf_q/conf_t；两分支对应过滤 conf_tau）|
| Data split | duck 120 帧子集先验证（ia 基线 47.50）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线，rng-fix 干净口径）|
| Success line | duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_conf.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_conff/cache \
    --out outputs/exp_conff/results/duck.json
```

## Live Log

- `08-12 14:35`：探针 10/10 帧分离（good conf 1.70-2.57 vs bad 1.42-2.19，
  差 +0.18~+0.38）——conf 是可靠对应质量信号。
- `08-12 14:40`：登记入队（running），conf_tau 1.5 实现完成（融合/非融合
  两分支 + 7 元组 desc_cache），202 测试过。
- `08-12 15:05`：首跑崩 UnboundLocalError `it`（conf 过滤在 it 定义前引用）——
  改用 nn_q2t，修复重启。

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
