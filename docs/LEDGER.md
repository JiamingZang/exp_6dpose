# LEDGER.md —— 结构化实验账本

状态标签：`champion`=冠军 | `done`=归档 | `running`=在跑 | `dead`=证伪（见 STATE 黑名单）
叙事与全部中间数字在 `EXPERIMENTS.md`，时间线在 `RESEARCH_LOG.md`。

## 主链轮次（LineMod 13 物体）

| 轮 | 改动 | 状态 | MEAN ADD | 一句话结论 |
|---|---|---|---|---|
| 1 | 管线打通（ape 基线）| done | ape 13.5% | tz 偏浅 4.2% 发现 |
| 2 | 深度偏差根因定位 | done | — | μ 混合锚点在表面内侧（μ+2σ 可见面）|
| 3 | CAD 深度监督 | dead（单独用）| 全量崩 15.7% | 切向收缩 ~3mm，须配逆深度锚点 |
| 4 | 固定视图 + 逆深度锚点 | done | ape 44.20% | tz 0.9983；视图必须固定 |
| 5 | 13 物体首轮（白背景）| done | 49.10% | eggbox 9.2% 异常暴露 |
| 6 | eggbox 根因（背景色）| done | eggbox 98.3% | 浅色≈白背景 → 边界梯度≈0 |
| 7 | 全物体黑背景 | done | 63.33% | 深色物体崩（driller 61.7）|
| 8 | 深色白背景（定型）| done | driller 95.0 | **背景色按物体亮度选** |
| 9 | **dc2 + guided_refine** | **champion** | **69.36%** | 弱物体普涨（holepuncher +7.5）|
| 10 | 回退保护 + 全量评估 | done | **69.74%**（全量 14968 帧）| 子集 71.55 全量衰减 -1.8；can 92.58/lamp 91.83 追近 GSPose 单项；refiner 净负贡献坐实 → 排队 6d-refiner-v2 |
| 11 | **iter_align 级联（6d-iter-align-ext）** | **champion** | **78.07%**（全量 14968 帧）| **13/13 全正**（+1.55~+18.57，加权 +8.34）；级联组合效应（iter_align 单独 +1.67 vs 级联 +16.67）；eggbox 98.40/driller 97.06 超 95% |

## 配置 × 结果对照

| 配置 | 状态 | 数字/用途 |
|---|---|---|
| dense80_depthc_guided.yaml | **champion** | 轮 9 子集 69.36 → 轮 10 全量 69.74（14968 帧）|
| dense80_depth_bg0.yaml | current | 浅色物体训练（黑背景 depth 0.6）|
| dense80_depth_w1.yaml | current | 深色物体训练（白背景）|
| dense80_depthc.yaml / dense80_dc2_t02.yaml / dense80_dc2_t02g20.yaml | done | dc2 调参过程 |
| dense80_guided.yaml | done | guided 单测（D 类 +4.2 均值）|
| dense80_norefine.yaml / dense80_depthc_norefine.yaml | current | refine 两档对比的"关"档（P4 #82）|
| dense80_depthc_gtmask.yaml | done | gt_mask 定位上界结案：4/5 弱物体定位侧瓶颈（holepuncher +9.17/cat +7.5/duck +5.84/can +5.0），ape 匹配侧例外（-1.67）|
| dense80_depthc_sam.yaml | dead | SAM ViT-H 对照判负（duck +0.83 ADD / -3.33 Proj；掩码 IoU 0.914 vs 0.910 几乎相同）|
| dense80_depthc_mast3r.yaml | done | 全解码对照：**收益不泛化**——duck +6.67（新口径 37.5 vs 30.83），ape/cat/holepuncher 平/负；候选池非一般性瓶颈 |
| dense80_depthc_p2.yaml | done | 两阶段预筛（top_k_prescreen: 60）判负：duck 34.17 只兑现一半收益；代码保留作消融档 |
| dense80_depthc_ia.yaml | **champion** | 迭代渲染对齐（iter_align_iters: 2）：**全 13 物体全量 MEAN 69.74→78.07（+8.34，14968 帧）**；13/13 全正（+1.55~+18.57）；位姿优化章核心机制 + 论文主表级联行 |
| dense80_w1_ia.yaml | done | 深色物体（白背景 bank）的 iter_align 档（base: dense80_w1 + iter_align_iters: 2）；cam 75.99（+12.31）/ driller 97.06（+6.49）|
| dense80_depthc_ia1.yaml | done | 消融 1 轮：duck 46.67/84.17/56.67（+15.84，接近 2 轮）|
| dense80_depthc_ia3.yaml | done | 消融 3 轮：duck 47.50/82.50/65.83（2 轮后增益递减）|
| dense80_depthc_ia_norefine.yaml | done | 消融 refiner 关：duck 32.50——**iter_align 单独仅 +1.67，增益=级联组合效应** |
| rng-fix（代码）| done | 每帧确定性 rng 种子（_frame_rng，frame_id 派生）：全空/半满 cache 逐帧一致（duck 30.83/81.67/40.83）；历史子集数字作废 |
| cache-redirect-resume（代码）| done | 缓存重定向后重启不加载重定向文件内容 → 已缓存帧全部重跑（lamp/phone 全量 ia 事故，重复处理 320-474 帧）；抽 `_load_cache_records` 修复 + 5 条回归测试（tests/test_cache_resume.py）|
| dense80_depthc_gtbbox{,_gtmask,_pd}.yaml | done | 6d-det-align 检测框口径拆变量：2a（框+全1+全解码）5 物体均值 -2.84 判负；2c（+DINOv2 预筛）≈基线 → 定位不是瓶颈；2b（GT 掩码）cat/duck +4~6；全解码排序物体异质（duck 受益）|
| dense80_depthc_ia_topk.yaml | done | 6d-gap-oracle 候选池 oracle 上界（champion + metrics.topk_best [1,3,5,40]）：top40 池内 GT 择优 62.0 ≈ 端到端 61.2 → 候选池生成是总瓶颈；duck/cat 池有货选择倒挂、ape/phone 优化净赚 |
| dense80_depthc_ia_geocand{,_gtmask}.yaml | done | 6d-mask-geo 掩码几何平移候选判负：duck FastSAM 44.17（-3.33）/ GT 掩码 45.00（-6.67 vs GT 掩码档 51.67）——机制无效非掩码拖累；代码保留（开关默认关）|
| dense80_depthc_ia_tzdepth.yaml / dense80_depthc_ia_tonly.yaml | running | 6d-tz-depth 平移校正两档：深度档（BOP 中值 z，消融）/ RGB-only 档（refine_stage1_iters 200 + 面积正则，主线可部署）|
| dense80_depthc_ia_multi.yaml | running | 6d-ia-multi 多初始假设 iter_align（iter_align_multi_hypo: 5，池内 top-k 各跑 + align_loss 择优）——挑战 3 选错模板帧 |
| dense80_depthc_mh.yaml / dense80_tzsearch.yaml / dense80_batch16.yaml / dense80_depth03_w1.yaml | archived | 过程变体 |
| legacy_mypose.yaml | archived | 旧管线复刻对照（README §8 口径警告）|
| experiments/dense80_gtmask.yaml | current | GT 掩码检索上界分析 |
| experiments/dense80_topk10.yaml / dense80k40_batch2/6.yaml / batch8.yaml / dense80_batch8.yaml | archived | K 值/批量消融过程 |
| experiments/dense80_scaletest.yaml | archived | 尺度测试 |

## 消融账（论文 §3.3，8 组）

跑法：`scripts/eval/run_ablation.py`；02/05/06/08 组需重建模板库（脚本自动 onboard）。
结果落 `outputs/ablation_<name>.json`。状态：**待全量跑**（子集数字优先）。

## 速度（论文 §3.4）

7.11 s/帧：localize 2.99（42%）+ matching 3.70（52%）+ pnp 0.42（6%）。
已实现未端到端验证：批量 DINOv2 CLS（2.5→~0.3s）、背景填色一致、top-40→top-10。
待做：帧间追踪（→<1s）。

## 择优策略（P4 候选）

- 现状：`inlier`（src/solver/selection.py:49，原始内点数 + 重投影二级排序
  见 ransac_pnp.py:160-163）
- 候选：`inlier_ratio`（selection.py:50-52 已实现，未启用）→ 消融 6 加一行
