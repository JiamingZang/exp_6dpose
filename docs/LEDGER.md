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
| dense80_depthc_mast3r.yaml | done | 全解码对照：结论反转后=候选池主要瓶颈 +5.0（旧口径 32.5 vs 27.5）；**rng-fix 后旧数字作废，待新口径重验** |
| rng-fix（代码）| done | 每帧确定性 rng 种子（_frame_rng，frame_id 派生）：全空/半满 cache 逐帧一致（duck 30.83/81.67/40.83）；历史子集数字作废 |
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
