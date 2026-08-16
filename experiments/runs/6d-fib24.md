# 6d-fib24 —— fibonacci 24 视角模板密度消融

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-fib24` |
| Owner | agent |
| Status | `done` |
| Started | `2026-08-16 11:37`（onboard）/ `12:00` 评测 |
| Finished | `2026-08-16 14:38` |
| Queue row | `experiments/QUEUE.md::6d-fib24` |

## Question

视角采样密度消融缺口：16→24 fibonacci 视角（相邻夹角 47.5°→~36°）能否提升 MASt3R 对应质量 → 粗位姿/端到端 ADD？（80t 甜点此前只有 cube8-vs-fibonacci 跨口径证据）

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_fib24.yaml`（guided + fibonacci 24 视角 × 5 = 120t） |
| Code change | none |
| Data split | 120 帧 × 5 弱物体 |
| Metrics | 粗位姿 ADD(S)@0.1d（与 80t 基线同口径，含联合 PnP + guided） |
| Baseline | 80t 库粗位姿（采集缓存同配置）：duck 30.83 / ape 46.67 / cat 53.33 / hp 50.83 / phone 65.00，MEAN 49.33 |
| Success line | 粗位姿 ADD ≥ 基线 +2 → 全量升级；≤+1 则证实 80t 饱和 |

## Commands

```bash
# 120t 库 onboard（续链自动，11:37-11:53）
python3 -c "from src.config import load_config; from src.pipeline import onboard_object; cfg=load_config('configs/experiments/dense80_fib24.yaml'); onboard_object(cfg, '<obj>')"
# 评测
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_fib24.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_fib24/cache --out outputs/exp_fib24/result.json
```

## Live Log

- `08-16 11:37`：续链启动（首次链因 onboard 路径错中止后补跑）；5 物体 120t 库 onboard 完成（每物体 ~3 分钟）。
- `08-16 14:38`：评测完成 rc=0。

## Result

| 物体 | 80t 基线 | fib24 120t | Δ |
|---|---:|---:|---:|
| duck | 30.83 | 40.00 | +9.17 |
| ape | 46.67 | 35.83 | -10.84 |
| cat | 53.33 | 59.17 | +5.84 |
| holepuncher | 50.83 | 30.00 | **-20.83** |
| phone | 65.00 | 66.67 | +1.67 |
| MEAN | 49.33 | 46.33 | **-3.00** |

## Decision

- 结论：`reject`（120t 升级否定；80t 饱和证实）
- 原因：MEAN −3.00 判负；duck/cat/phone 受益（+1.67~+9.17）但 ape/hp 灾难（−10.84/−20.83）。机制：24 视角×5 的 120t 库经 DINOv2 预筛 top-40 后，自相似物体（hp 规则格栅、ape 低分化表面）的相似模板互相挤占名额，正确模板被挤出池；外观分化物体（duck）受益于密度。与 K 曲线/J 曲线/早停的物体异质性同构（见 6d-adaptive-k-sim 记录 08-16 14:00 条目）。
- 下一步：80t 保持默认；密度×预筛交互作为 §3.3.2 论断（"模板密度非单调，80t 饱和"）的完整证据。

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新（done）
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚（outputs/exp_fib24/result.json）
- [ ] `python3 scripts/analysis/check_state.py` 通过
