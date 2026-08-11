# 6d-mask-geo —— 掩码几何平移候选（GS-Pose §3.2 式）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-mask-geo` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-11 09:10` |
| Finished | `2026-08-11 11:05` |
| Queue row | `experiments/QUEUE.md::6d-mask-geo` |

## Question

这次只回答一个问题：

> 全量失败帧 48.5% 为"旋转对、平移错"（RANSAC 深度病态：tz 错 30-40mm
> 重投影 <5px 不可分辨）。GS-Pose 式掩码几何解析平移（面积比 z + 质心 xy，
> 不依赖特征对应）作为独立候选与 PnP 结果竞争择优，能否补上这部分？
> FastSAM 掩码版判负后，用 GT 掩码判别：机制无效还是掩码拖累？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_geocand.yaml` `configs/current/dense80_depthc_ia_geocand_gtmask.yaml` |
| Code change | `src/pipeline.py`：`_mask_geo_translation()` + `_solve` 接入（`solver.mask_geo_candidate`）；5 条单测（tests/test_mask_geo.py）|
| Data split | duck 120 帧子集（FastSAM 掩码 + GT 掩码两档判别）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线）；GT 掩码档 51.67（2b 实测）|
| Success line | geocand ADD ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_geocand.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_geocand/cache \
    --out outputs/exp_geocand/results/duck.json
# 判别实验 B（GT 掩码）：
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_geocand_gtmask.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_geocand_gtmask/cache \
    --out outputs/exp_geocand_gtmask/results/duck.json
```

## Live Log

- `08-11 09:10`：实现（_mask_geo_translation + _solve 接入 + 配置 + 5 单测），202 测试过。
  机制：z = z_ref·sqrt(A_q/A_ref)（模板球面半径），xy = 掩码质心反投影，
  与 PnP 结果 align_loss 竞争择优（不进 IoU 接受门，绕开 tz_search 判据不可分）。
- `08-11 10:00`：duck 验证启动（geocand 队列自动衔接 gap-oracle）。
- `08-11 10:45`：**FastSAM 版 44.17（-3.33）判负**。混入掩码质量变量
  （2b 已知 FastSAM 掩码是部分瓶颈）→ 启动 GT 掩码判别实验 B。
- `08-11 11:05`：**GT 掩码版 45.00（-2.50）**——相对 GT 掩码档 51.67 为
  **-6.67**。判别完成：**机制无效，非掩码拖累**。

## Result

| 档位 | ADD | Δ vs 基线 47.50 | note |
|---|---:|---:|---|
| 基线（ia，FastSAM） | 47.50 | — | |
| GT 掩码档（2b 实测） | 51.67 | +4.17 | 掩码质量确为部分瓶颈 |
| geocand（FastSAM） | 44.17 | -3.33 | 判负 |
| geocand + GT 掩码 | 45.00 | -2.50 | 相对 GT 掩码档 -6.67 |

## Decision

- 结论：`reject`（**机制无效**，判别实验排除掩码拖累）
- 原因：
  1. **质心反投影 xy 近似不成立**：物体原点投影 ≠ 掩码质心（duck 形状
     不对称），xy 偏差直接进 t；
  2. **面积比 z 被模板渲染掩码偏差污染**：3DGS 渲染掩码 vs 真实掩码的
     面积系统偏差（壳/洞）→ z 系统性错；
  3. **align_loss 择优不足以支撑**：几何候选渲染损失未必更差，但 ADD
     更差——渲染对齐损失与 ADD 相关性弱于假设；
  4. 与 tz_search 结案互为印证：掩码几何信号（面积/质心）在 FastSAM/
      3DGS 掩码质量水平下不可用，不是接入方式问题（后处理 vs 候选竞争）。
- 下一步：纯 t 错帧的替代路线 = 渲染对比**连续优化**（GS-Refiner 式：
  粗位姿直接可微优化渲染损失，无离散候选池）——强化 iter_align/refiner
  的连续搜索能力；或掩码侧训练（超出零样本范围，论文已定位为取舍）。
- 产物：`outputs/exp_geocand/results/duck.json`、`outputs/exp_geocand_gtmask/results/duck.json`

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
