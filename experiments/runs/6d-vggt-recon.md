# 6d-vggt-recon —— VGGT-1B 成对位姿 sanity 判死

## Metadata

| 字段 | 值 |
|---|---|
| ID | 6d-vggt-recon |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md::6d-vggt-recon` |

## Question

> VGGT-1B 直接输出查询位姿（[模板渲染, 查询裁剪] 成对）能否过域差判据？

## 结论

**判死（1B）**：R_err 中位 94.1°、tz 反号、ADD 823mm（ape 120 帧）。
消融：同图对 0.02°（协议不崩）但跨视角 145°——跨图位姿回归不可靠，非纯域差。
VGGT-Omega（gated 待 HF 授权）可用同脚本复跑（/tmp/vggt_sanity_omega.py）。
详细数字见 docs/EXPERIMENTS.md「6d-vggt-recon sanity」。
