# 6d-ablation-full —— 论文 §3.3 十组消融（120 帧子集口径）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-ablation-full` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 16:10` |
| Finished |  |
| Queue row | `experiments/QUEUE.md::6d-ablation-full` |

## Question

这次只回答一个问题：

> 论文 §3.3 十组消融（topk/n_templates/matcher/localization/geometry/
> scale_align/selection/ransac_eps/segmenter/renderer）在 **120 帧子集口径**
> （用户 08-12 确认，磁盘 8.1G 不足以全量 14968 帧缓存）下跑齐，支撑
> 模板库构建 + dc2 方法贡献的消融证据。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/ablations/`（10 组 yaml），base = `configs/current/dense80_depthc_guided.yaml`（粗位姿，论文方法贡献口径）|
| Code change | 无（run_ablation.py 已有）|
| Data split | **120 帧子集 × 5 弱物体**（duck/ape/cat/holepuncher/phone，消融最有区分度）；全量口径待磁盘清理后补 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5°（run_ablation.py 自动出表）|
| Baseline | 粗位姿 120 帧口径（dense80_depthc_guided.yaml）|
| Success line | 10 组数字落 `outputs/ablation_<name>.json`，入论文表 |

## 组别状态

| 组 | sweep | reonboard | 状态 |
|---|---|---|---|
| 01 topk | K=1/5/10/20/40 | 否 | 第一批 |
| 03 matcher | mast3r/dinov2_patch/loftr | 否 | 第一批（loftr 抛 NotImplementedError 跳过）|
| 04 localization | fastsam/gt_bbox | 否 | 第一批 |
| 06 scale_align | true/false | 否 | 第一批（CAD+RGB-PnP 下数学恒等，诚实性检查）|
| 07 selection | inlier/similarity/weighted | 否 | 第一批 |
| 09 ransac_eps | ε=3/5/8/10 | 否 | 第一批 |
| 10 segmenter | fastsam/sam/gt_mask | 否 | 第一批 |
| 02 n_templates | 8/24/40/80 | 是（8/24 需 onboard，40/80 已有）| 第二批 |
| 08 renderer | 3dgs/pyrender_cad | 是 | 第二批 |
| 05 geometry | cad/vggt | 是 | **跳过**（VGGT 未装，onboard 会崩）|

## Commands

```bash
source env.sh
# 第一批（无 reonboard）
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/01_topk.yaml --objects duck ape cat holepuncher phone --max-frames 120
# ... 03/04/06/07/09/10 同式
# 第二批（需 onboard）
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/02_n_templates.yaml --objects duck ape cat holepuncher phone --max-frames 120
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/08_renderer.yaml --objects duck ape cat holepuncher phone --max-frames 120
```

## Live Log

- `08-12 16:10`：登记入队（running），子集口径（用户确认）。模板库覆盖检查：
  40t/80t 全 13 物体已有；8t/24t 缺（02 组需 onboard cube8 采样）；05 组
  VGGT 未装跳过（run_ablation 不 catch ImportError，会崩）。

## Result

| 组 | baseline | this run | delta | note |
|---|---|---:|---:|---|
| 01 topk |  |  |  |  |
| 03 matcher |  |  |  |  |
| 04 localization |  |  |  |  |
| 06 scale_align |  |  |  |  |
| 07 selection |  |  |  |  |
| 09 ransac_eps |  |  |  |  |
| 10 segmenter |  |  |  |  |
| 02 n_templates |  |  |  |  |
| 08 renderer |  |  |  |  |
| 05 geometry |  |  |  | 跳过（VGGT 未装）|

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
