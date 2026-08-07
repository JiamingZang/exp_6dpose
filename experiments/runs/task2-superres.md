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
