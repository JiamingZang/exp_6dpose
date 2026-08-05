# STATE.md —— 唯一状态源（agent 进来先读这个）

> 更新规则：每轮实验出结果**同一次操作内**更新本文件。
> 上次更新：2026-08-04（整理自 docs/EXPERIMENTS.md 轮 9 记录）

## 冠军（论文主表数字）

| 项 | 值 |
|---|---|
| 数据 | LineMod 13 物体 × 120 帧均匀采样（子集，全量待跑）|
| **MEAN** | **ADD 69.36 / Proj 81.67 / 5cm5° 66.99**（轮 9：depth_consistency + guided_refine）|
| 基线对照 | 无 dc2：67.44 / 81.54 / 66.54；旧代码 MyPose top1：49.49 / 59.22（端到端可比）|
| 配置 | `configs/current/dense80_depthc_guided.yaml`（depth_consistency + tau_frac 0.05 + guided 2 轮）|
| 模板库 | dense80（fibonacci 16 视角 × 5 平面内旋转，512×512），固定视图 + 逆深度锚点 |
| 训练背景 | 按物体亮度：浅色黑背景（depth 0.6）、深色白背景（driller/cam）|
| 外部目标 | GSPose 92.0，差 22.6，缺口集中在 holepuncher/duck/ape（31-38）|
| 结果 | `matches13_dc2` 提取 + `scripts/eval/summarize13.py` 汇总 |

## 在跑 / 待办

| 项 | 说明 |
|---|---|
| 全量评估 | 1172 帧/物体（~15-18h）确认子集数字——**论文审稿必问** |
| refine 两档对比 | 纯几何档 vs 几何+精化档（论文 §3.7 占位，见 citecoon P4）|
| 帧间追踪 | 上帧位姿初始化跳过定位+匹配，7.1s → <1s（速度章）|
| D 类深挖 | duck/holepuncher/ape：锚点/训练级问题，guided 已证明覆盖不全 |

## 黑名单（已证伪/已定型，禁止回退重跑）

| 路线 | 死因 | 出处 |
|---|---|---|
| μ 混合锚点（无深度监督）| 体渲染表面在 μ+2σ，tz 系统性偏浅 4.2% | 轮 1-2 |
| CAD 深度监督单独用（不配逆深度锚点）| 切向收缩 ~3mm，全量 ADD 崩到 15.7% | 轮 3 |
| 重训后重采样模板视图 | 与像素对应错位 0.5%，**视图必须固定**（rebuild_bank_fixed_views）| 轮 4 |
| 全物体统一背景色 | 浅色需黑背景、深色需白背景，单一必崩一边（eggbox 9.2% / driller 61.7%）| 轮 6-8 |
| 旧表 67.63/80.06/65.13 | 误标背景的废数据，已作废（08-02 晚更正）| §3 更正 |
| holepuncher 靠 guided_refine | 完全无改善，根因在训练/锚点层 | §3 guided 测试 |

## 已知坑

- **子集 vs 全量口径**：主表是 120 帧子集（略偏乐观），ape 全量对照
  44.20 vs 子集 50.00——引用数字必须注明口径
- 旧代码 top3/5_best 是 GT 择优 oracle 上界，**不可与端到端比**；可比只有 top1 49.49
- `outputs/templates/*.npz.orig/.viewsbak` 是版本备份，恢复模板库前先认清版本
- 择优判据现为 `inlier`（原始内点数，src/solver/selection.py:49）；
  `inlier_ratio` 已实现未启用（selection.py:50-52），消融候选（P4 #1）
- **`template_source=depth_map` 深度渲染曾与训练监督不一致**（2026-08-04
  修）：`gs_trainer.py` 训练深度监督用逆深度 alpha 混合，`template_renderer.py`
  的 `coord_map` 路径也用，但 `depth_map` 路径独立手写了一套线性 z 混合，
  跟前两者不是同一套数学（同源于"深层高斯泄漏拉远"偏差却没跟着修）。已
  抽成 `GaussianTrainer.render_invdepth` 统一三处；`depth_map`/
  `depth_backproject` 是历史对照路线，当时不在冠军配置里，**数字未受
  影响**，但若之前跑过这条路线的对照数字，需重跑才可信
- **多处随机源未接 `runtime.seed`**（2026-08-04 grep 核实）：pipeline 主链
  已接线，但 `src/solver/ransac_pnp.py:185,271`、`src/geometry/alignment.py:77`、
  `src/matching/correspondence.py:254,297`、`src/matching/mast3r_wrapper.py:246`、
  `src/matching/alt_matchers.py:103`、`src/gaussian/gs_trainer.py:227`、
  `src/datasets/ply_io.py:161` 写死 `default_rng(0)`——改 seed 不影响这些环节，
  做 seed 敏感性实验前先接线
- `scripts/data/rebuild_bank_fixed_views.py:32` 前景掩码阈值 `alpha_fg=0.5`
  是函数默认值、调用处未暴露为配置——重建模板库时动它要改代码

## 复现主链

```bash
# 0 环境+数据（README「GPU 机器从零复现」节，约 6GB 数据）
# 1 onboard 13 物体（3DGS 7000 iter，黑/白背景按物体选，depth 0.6）
python scripts/data/onboard_object.py --config configs/current/dense80_depth_bg0.yaml <obj>
# 2 固定视图重建模板库（旧 poses + 新高斯 + 逆深度锚点）
python scripts/data/rebuild_bank_fixed_views.py ...
# 3 主评测（120 帧子集）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_guided.yaml
# 4 汇总
python scripts/eval/summarize13.py
```
