# task4-symmetry —— 近似对称量化（分析完成）

## Metadata

| 字段 | 值 |
|---|---|
| ID | task4-symmetry |
| Status | done |
| Started | 2026-08-07 |
| Queue row | `experiments/QUEUE.md`（任务清单）|

## Question

> duck/ape 失败帧是否"只差一个近似对称轴旋转"（指标病态性）？

## 结论

**无显著近似对称**：自对齐扫描仅小角度平凡解（15° 残差 0.027-0.031×diam）；
失败帧绕对称轴旋转后进 ADD-S 阈值占比 duck 1%（1/88）、ape 3%（2/66）——
指标病态性叙事排除，失败根因在对应点/深度。脚本：scripts/analysis/approx_symmetry.py。
