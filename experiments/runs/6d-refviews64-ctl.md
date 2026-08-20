# 6d-refviews64-ctl —— 归因对照：cat 64v 现代配方

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-refviews64-ctl` |
| Status | `done`（08-20 20:20 结案） |
| Config | onboard `configs/experiments/dense80_refviews64_ctl.yaml`（7000/64v 现代配方）；评测 `configs/current/dense80_depthc_ia.yaml` |

## Live Log

- `08-20 19:10`：cat 128v 库备份 .v128bak → onboard 64v 现代配方 → 子集评测。
- `08-20 19:58`：结果 67.50。

## Protocol

| 项 | 值 |
|---|---|
| 数据 | cat，120 帧子集 |
| 判读 | 64v 现代 ≈ 80.00 → 锚点/深度主因；≈ 64.17 → 视图主因 |

## Commands

```bash
python3 -c "from src.pipeline import onboard_object; from src.config import load_config; onboard_object(load_config(CTL), 'cat')"
python3 scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia.yaml --objects cat --max-frames 120 --cache-dir outputs/exp_ctl64_sub/cache --out outputs/exp_ctl64_sub/results/cat.json
```

## Question

> refviews 的 cat +15.83（64.17→80.00）来自视图数（64→128）还是现代配方（invdepth 锚点 + depth 0.6）？64v 现代配方对照分离两者。

## Result

| 档 | cat 120 帧 ADD | vs 旧世代 64v |
|---|---|---|
| 旧世代 64v（无指纹） | 64.17 | 基线 |
| **64v 现代配方** | **67.50** | **+3.33** |
| 128v 现代配方（refviews） | 80.00 | +15.83 |

**归因**：+15.83 总增益中，视图数贡献 ≈ **+12.50（79%）**、现代配方（锚点/深度）贡献 ≈ **+3.33（21%）**——**视图数是主因**。

## Decision

- 结论：`done`（视图数主导）
- 原因：64v 现代配方只兑现 21% 增益；128v 是关键变量
- 下一步：cat 128v bank 已从 .v128bak 恢复（指纹校验 7000/128v/invdepth ✓）；champion 不变

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
