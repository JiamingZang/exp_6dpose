# 11_joint_templates —— 联合 PnP 合并模板数 J 消融（§3.6.3 缺口补全）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `11-joint-templates` |
| Owner | qoder |
| Status | `planned` |
| Started | `empty` |
| Finished | `empty` |
| Queue row | `experiments/QUEUE.md::11_joint_templates` |

## Question

联合 PnP 精化（§3.6.3）合并 sim 分数最高的 J 个模板的对应重解，是粗位姿
管线的正式组成部分，但其贡献从未被消融量化（10 组消融计划无此组）。
同时：正确模板常排 sim 4-12 位（候选池有货帧），J=12 是当前口径——
更深的池融合（J=20）能否增益？J 过小（1=纯择优）损失多少？

> 假设：J 曲线存在峰值；J=1（关联合）显著低于 J=12；J=20 是否回退
> 取决于无深度过滤的合并噪声 vs 立体覆盖增益。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/ablations/11_joint_templates.yaml`（base=`dense80_depthc_guided.yaml`，sweep solver.joint_templates ∈ {1,5,10,20}）|
| Code change | none（joint_templates 已是配置项；12=default.yaml 主表口径）|
| Data split | 120 帧 × 5 弱物体（duck/ape/cat/holepuncher/phone）|
| Metrics | ADD(S)@0.1d（+Proj/5cm5°）|
| Baseline | J=12 免费取自 01 K=40（49.17，同配置同口径）|
| Success line | 四档数字出炉；J=12 复现 ≈49.17±1 帧抖动；给出 J 曲线结论 |

## Commands

```bash
python3 scripts/eval/run_ablation.py --ablation configs/ablations/11_joint_templates.yaml \
    --config configs/current/dense80_depthc_guided.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/ablation_cache --device cuda
```

## Live Log

- `08-14 18:30`：登记。发现 champion 实际 joint_templates=12（default.yaml:168，
  非代码默认 3）；"实测 K=7 最佳"注释与现值不符（历史遗留，待本次曲线裁决）。

## Result

| J | ADD(S)@0.1d | note |
|---|---:|---|
| 1（关联合）|  |  |
| 5 |  |  |
| 10 |  |  |
| 12（默认）| 49.17 | 01 K=40 免费基线 |
| 20 |  |  |

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
