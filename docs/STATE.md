# STATE.md —— 唯一状态源（agent 进来先读这个）

> 更新规则：每轮实验出结果**同一次操作内**更新本文件。
> 上次更新：2026-08-10（6d-iter-align-ext 结案：13/13 全正，级联 champion 78.07；缓存重定向续跑缺陷修复）

## 冠军（论文主表数字）

| 项 | 值 |
|---|---|
| 数据 | LineMod 13 物体 × **全量 14968 帧**（排除参考帧）|
| **粗位姿 MEAN** | **ADD 69.74 / Proj 83.77 / 5cm5° 68.69**（回退保护：refiner 精化前后渲染对齐损失择优）|
| **级联 MEAN（iter_align 2 轮 + 抛光，champion）** | **ADD 78.07 / Proj 87.17 / 5cm5° 81.31**（13/13 全正，+1.55~+18.57，按帧加权 +8.34）|
| 子集对照 | 120 帧子集 71.55（08-05 口径，全量 -1.8 属正常衰减）|
| 基线对照 | 无 dc2：67.44 / 81.54 / 66.54；旧代码 MyPose top1：49.49 / 59.22（端到端可比）|
| 配置 | 粗位姿 `configs/current/dense80_depthc_guided.yaml`；级联 `dense80_depthc_ia.yaml`（深色物体 `dense80_w1_ia.yaml`）；均可微抛光（回退保护）|
| 模板库 | dense80（fibonacci 16 视角 × 5 平面内旋转，512×512），固定视图 + 逆深度锚点 |
| 训练背景 | 按物体亮度：浅色黑背景（depth 0.6）、深色白背景（driller/cam）|
| 外部目标 | GSPose 92.0（YOLOv5 检测框口径），级联后差 13.9；多数物体已超 90% 量级 |
| 结果 | 粗位姿 `outputs/exp_full/results/*.json`；级联 `outputs/exp_full_ia/results/*.json`（14968 帧全量）|

## 在跑 / 待办

| 项 | 说明 |
|---|---|
| 6d-ablation-full（优先级 1，待排期）| 论文 §5.3 十组消融全量（run_ablation.py --all）|
| 6d-ablation-full（优先级 1）| 论文 §3.3 八组消融全量跑齐（run_ablation.py --all），支撑模板库构建+dc2 方法贡献的消融证据 |
| 6d-gap-oracle（已结案 08-11）| 候选池 vs 选择损失：top40 池内 GT 择优 62.0 ≈ 端到端 61.2（+0.8）——**候选池生成是总瓶颈**；分型：duck/cat 池有货选择倒挂、ape/phone 池没货优化净赚 |
| 6d-mask-geo（已结案 08-11）| 掩码几何平移候选判负（duck FastSAM -3.33 / GT 掩码 -2.50）：机制无效非掩码拖累；失败帧再分解：48.5% 旋转对 t 错（tz 仅 35 帧、xy 错 1556 帧、GT t 替换 100% 恢复）→ 平移病态实为 xy；BOP 深度图可用（从未用于推理）|
| 6d-tz-depth（已结案 08-11）| 平移病态两档收官：深度档 +4.17（z 正确是 xy 对齐前提，RGB-D 充分条件）；三个 RGB-only 档全负（tzxy -1.67 / t-only -2.50 / mask-geo -3.33）——**单目平移病态=信息极限**；主表 RGB 单目口径，作第四章可解性证据 |
| 6d-multi-ext（已结案 08-12，**泛化失败 5 物体均值 -4.34**）| **分型实锤**：cat +10.83（池有货）但 ape -10.00 / holepuncher -15.84 / phone -15.00（池没货）——align_loss 无条件换种子把 best 也换掉（ape 95/120 帧换坏）；gap-oracle 分型完全验证 |
| 6d-multi-gate（已结案 08-12，判负）| 光度门控 duck 滚动 45.45 ≈ 基线：拒绝域方向对但指标错——align_loss 与 ADD 相关性弱，正确种子优势 <5% 被挡；通用教训：择优/门控基准选择比门控本身重要 |
| 6d-multi-iou（已结案 08-12，判负 -4.00 vs ia）| 5 物体均值 57.00：修复 multi 灾难（holepuncher -15.84→-6.67、保 duck/cat +5/+10）但三个没货物体仍负（ape/phone -14）——FastSAM 掩码偏差污染 mask_iou；**渲染比较量（align/iou）两候选全负**，inlier 档收官 |
| 6d-multi-inl（已结案 08-12，判负 -3.33 vs ia）| 5 物体均值 57.67，duck +11.67 全场最佳但池没货物体依旧全负（ape -10/holepuncher -12.5/phone -10.83）——**择优指标系列收官（align -4.34 / iou -4.00 / inl -3.33 全负）**，多候选择优结案：gap-oracle"候选池生成是总瓶颈"完全验证，champion 保持 78.07 |
| 6d-ia-multi（已结案 08-11，**+8.33**）| 挑战 3 多初始假设 iter_align 有效：duck 55.83（+8.33，端到端 align_loss 择优），36.2% 选错模板帧部分救回；效率 +2% 几乎免费；扩 5 弱物体验证中 |
| 6d-track-seed（已结案 08-11，弱正 +3.33）| 帧间跟踪种子 duck 50.83（基线 47.50 → +3.33）但远低于 multi 55.83——池内多假设优于时间种子；代价 +40% 不划算；BOP 主表逐帧独立，仅论文视频扩展素材 |
| 6d-multi-refine（已结案 08-11，判负）| 种子级渲染对比优化 duck 49.17（-6.66 vs multi）：refiner 盆底择优失效（ADD -6.66 但 Proj +9.16，渲染自洽非几何正确）——6d-refiner-v2 机制在多种子框架复现；渲染对比优化方向两轮判负结案，差距回填候选池生成 |
| 6d-prescreen-mmr（已结案 08-11，判负）| MMR 预筛多样性 duck 46.67（-0.83）：预筛阶段两度尝试（扩 K、多样性重排）均无收益——瓶颈在 MASt3R 对应质量本身 |
| 6d-fallback-decode（已结案 08-11，判负）| 失败帧自适应全解码 duck 46.67（-0.83）：触发帧与全解码收益帧不重合，失败帧付 20.8s 无收益；解码侧结案 |
| 6d-pnp-multisol（已结案 08-12，**判死**）| 挑战 2 前提诊断：duck 60 帧 × 30 次 RANSAC 全部单解（硬对应 + EPnP 闭式解无多解性）；inlier 择优 60/60 命中解集最优——瓶颈确证在候选池生成，与 gap-oracle 闭环；EPro-PnP 多解来自训练侧 soft correspondence |
| 6d-vggt-recon（Omega）| VGGT-1B sanity 判死（R_err 94°）；Omega 权重 gated 待 HF 授权，授权后同脚本复跑 |
| 帧间追踪 | 上帧位姿初始化跳过定位+匹配，7.1s → <1s（速度章，P5 排期后）|

## 已结案（08-10 迭代渲染对齐全量）

- **6d-iter-align 通过**：duck 30.83→47.50（+16.67）/ ape 47.5→59.17（+11.67），
  5cm5° 双双 +18~21；单帧 +0.5s；复现性 OK（gsplat 浮点噪声 ±1 帧）——
  **位姿优化章核心机制**（当前位姿重渲染 → MASt3R 再匹配 → 重解 PnP，
  接受/拒绝门保护）
- **6d-iter-align-ext 结案（13/13 全正）**：全量 14968 帧 MEAN ADD
  69.74→78.07（+8.34）；增益 +1.55（lamp）~+18.57（cat）；迭代轮数
  2 轮为甜点（3 轮仅 5cm5° +3.33）；组合效应=iter_align 单独 +1.67 vs
  级联 +16.67（120 帧 duck）——**级联 champion 入论文主表**
- 候选池系列结案：6d-weak-objects 全解码收益不泛化；6d-prescreen2 判负
- 代码修复：缓存重定向续跑缺陷（`_load_cache_records`，tests/test_cache_resume.py 5 条）
- **6d-det-align 结案（检测框口径判负）**：GT bbox 上界不改善结果（2a 均值 -2.84）；
  零样本定位已近检测框水平（2c ≈ 基线 ±2）；掩码质量是 cat/duck 瓶颈
  （2b +4~6）、ape 匹配侧例外；全解码排序物体异质（duck 受益）——
  与 GSPose 92.0 的差距不来自定位，双口径叙事入论文 §5.2.1

## 黑名单（已证伪/已定型，禁止回退重跑）

| 路线 | 死因 | 出处 |
|---|---|---|
| μ 混合锚点（无深度监督）| 体渲染表面在 μ+2σ，tz 系统性偏浅 4.2% | 轮 1-2 |
| CAD 深度监督单独用（不配逆深度锚点）| 切向收缩 ~3mm，全量 ADD 崩到 15.7% | 轮 3 |
| 重训后重采样模板视图 | 与像素对应错位 0.5%，**视图必须固定**（rebuild_bank_fixed_views）| 轮 4 |
| 全物体统一背景色 | 浅色需黑背景、深色需白背景，单一必崩一边（eggbox 9.2% / driller 61.7%）| 轮 6-8 |
| 查询裁剪超分（任务 2）| 512 输入下超分=两次插值纯损失：对应全翻牌（同帧 pix_q 相同占比 0%），bicubic/ESRGAN ADD 均崩到 0.83%；1024 输入 OOM | EXPERIMENTS.md 任务 2 |
| 稳定先验接入（任务 1.2 接入点 B）| 先验与失败帧错配：duck -6.67 / holepuncher -5.83（本底 18-20° 使 prior 在正误候选间重叠，择优被噪声主导）；联合 PnP 吞择优 | EXPERIMENTS.md 任务 1.2 |
| 旧表 67.63/80.06/65.13 | 误标背景的废数据，已作废（08-02 晚更正）| §3 更正 |
| holepuncher 靠 guided_refine | 完全无改善，根因在训练/锚点层 | §3 guided 测试 |
| VGGT-1B 成对位姿求解（在线）| R_err 中位 94°、tz 反号；同图对 0.02° 但跨视角 145°——跨图位姿回归不可靠（非纯域差）| EXPERIMENTS.md 6d-vggt-recon sanity |
| MASt3R pointmap 3D-3D 对齐替代 PnP | 成对输出统一系（img1 系），查询相机系 3D 不存在；域差 34% | EXPERIMENTS.md 6d-pointmap-t1 |
| refiner-v2（GS-Pose/旧代码思路重做）| 判据全失效（duck align_loss/mask_iou 均 ~51%），单起点局部光度优化净负 | EXPERIMENTS.md 6d-refiner-v2 |
| 换更强通用分割器（SAM ViT-H 自动掩码）| duck ΔADD 仅 +0.83 且 ΔProj -3.33，距 gt_mask 上界仍差 5.0——缺口在检测级（候选生成）不在分割质量 | EXPERIMENTS.md 6d-loc-upper |

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
- **评估 cache 状态污染 rng 流（已修复 08-08，6d-rng-fix）**：evaluate 逐帧
  solve 曾消耗 pipeline self.rng（seed 0）全局流，cache 命中帧跳过 solve 不
  消耗 rng → 不同 cache 状态 = 不同 rng 流 = 120 帧子集抖动 ±6 分（duck
  33.33 vs 27.5）。修复：每帧 `default_rng(seed + frame_id)`（_frame_rng），
  种子只依赖帧号；全空/半满 cache 逐帧一致（duck 30.83/81.67/40.83）。
  **rng-fix 前的历史 120 帧子集数字全部作废**；PnP 内部 default_rng(0)
  确定，无流问题
- **gsplat 光栅化浮点不确定（08-08 记录）**：光栅化原子累加顺序 GPU 级
  不确定 → 逐帧位姿 1e-4 级噪声、refiner 轨迹微扰；主指标（ADD/Proj）
  跨次稳定，5cm5° 偶见 ±1 帧（0.83 分）。与 rng 流污染（±6 分）不同，
  属可接受残留；对比实验取主指标判断
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
