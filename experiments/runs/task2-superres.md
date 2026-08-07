# task2-superres —— 查询裁剪超分（判负）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task2-superres |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md`（任务清单）|

## Question

> 定位后裁剪超分 ×2 喂 MASt3R 能否提升弱物体对应点供给（M 类病）？

## 结论

**判负**：bicubic/ESRGAN ×2（512 输入）duck ADD 均崩到 0.83%（基线 26.67）；
对应点全翻牌（同帧 pix_q 相同占比 0%）——超分图 resize 回 512 = 两次插值纯损失；
1024 输入 OOM（24.8GB 峰值）。M 类病瓶颈不在查询分辨率。
详细见 docs/EXPERIMENTS.md「任务 2」。

## Protocol

| 项 | 值 |
|---|---|
| Config | 见 QUEUE.md 对应行 |
| Code change | 见 git log（同 ID commit）|
| Data split | 120 帧子集 |
| Metrics | ADD/Proj/5cm5° |
| Baseline | docs/STATE.md 冠军表 |
| Success line | 见 QUEUE.md 对应行 |

## Commands

```bash
# 实际命令见 docs/EXPERIMENTS.md 对应节（脚本与命令已随 commit 落库）
```

## Live Log

- 2026-08-07：完成并判负/判死，详见 EXPERIMENTS.md

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| 见 docs/EXPERIMENTS.md 对应节 | — | — | — | 详细数字在 EXPERIMENTS.md |

## Decision

- 结论：`reject`（判负/判死）或见正文
- 原因：见 Question 下正文
- 下一步：见 QUEUE.md 最新行

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
