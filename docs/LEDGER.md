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
| dense80_depthc_ia_tzdepth.yaml / dense80_depthc_ia_tonly.yaml | done | 6d-tz-depth 平移校正收官：深度档 +4.17（RGB-D 充分条件）；t-only -2.50 / tzxy -1.67（RGB-only 全负）——单目平移病态=信息极限 |
| dense80_depthc_ia_multi.yaml | done | 6d-ia-multi 多初始假设 iter_align 通过：duck 55.83（+8.33，端到端）；效率 +2%；挑战 3 首正，扩 5 弱物体 |
| dense80_depthc_ia_multigate.yaml | done | 6d-multi-gate 光度门控判负：duck 滚动 45.45≈基线——拒绝域方向对但指标错（align_loss 与 ADD 相关性弱）；代码保留（gate 档）|
| dense80_depthc_ia_multi_iou.yaml | done | 6d-multi-iou 几何择优判负：5 物体均值 57.00（-4.00 vs ia）——掩码偏差污染 mask_iou，三个没货物体仍负；渲染比较量两候选全负 |
| dense80_depthc_ia_multi_inl.yaml | done | 6d-multi-inl inlier 几何择优判负：5 物体均值 57.67（-3.33 vs ia）——duck +11.67 最佳但池没货物体全负；**择优指标系列收官（align/iou/inl 全负）**，多候选择优结案 |
| dense80_depthc_ia_768.yaml | done | 6d-match-768 分辨率侧收官判负：duck 29.17（-18.33，模板同步全修复版）——查询裁剪 512 是信息上限，768 插值放大污染 desc；附带代码修复（模板编码与查询长边同步 + 三处 pix_t 换算，512 档行为不变）|
| dense80_depthc_ia_conf.yaml | done | 6d-conf-filter desc_conf 过滤判负：v1 双侧 1.5 灾难 4.17%（模板合成图 conf 系统性低 p95≈0.6 被全滤）；v2 查询侧 1.3 仍负（-6.67）——好/坏 conf 重叠带大伤数量，RANSAC 本鲁棒；Proj 85.83 全场最高但 ADD 跌 = 平移病态印证 |
| dense80_depthc_lg.yaml | done | 6d-lightglue 稀疏匹配判负**全灭**：duck 120 帧 0.00/0.00/0.00——域差下稀疏对应 ~29 对 → PnP 数值爆炸（t~5e16）；对应质量两路线对比：稠密 desc+几何先验 >> 零训练稀疏匹配；与 conf-filter/768 三线闭合（代码 16eb089，LightGlueMatcher 保留作消融档）|
| dense80_depthc_dinov2patch.yaml | running | 6d-ablation-full 03_matcher 组：DINOv2 patch token 稠密匹配器消融（120 帧 × 5 弱物体，批处理 b4zzkzkup）|
| dense80_depthc_consensus.yaml | running | 6d-consensus 模板层解集共识择优：inlier 择优选中"自洽地错"解时，位姿聚类（10°/25mm）最大簇内 inlier 择优替换；无簇保守不换（安全门控）；纯几何不依赖渲染/掩码；08-13 对称等价位姿聚类 + joint 门控等价类判定（c48d8aa/51d70de）；5 弱物体 120 帧排队（duck verify 后自动跑）|
| experiments/dense80_topk_instr.yaml | done | 6d-adaptive-k-sim 数据采集档（guided + topk_best [40]）：缓存落盘逐候选 cand_*；duck/ape 采满（08-16），cat/hp/phone 链中；离线仿真 duck +3.33/ape +7.50 @ meanK~2.5 |
| experiments/dense80_es.yaml | done | 6d-adaptive-k-sim 在线验证粗位姿档（08-16）：-5.33（44.00 vs 49.33）——联合 PnP 池收窄是唯一损失（es_nostop 归因）|
| experiments/dense80_es_ia.yaml | done | 6d-adaptive-k-sim 在线验证级联档（08-16）：60.17 vs 61.00（-0.84 判平）——~90% 解码削减换噪声带内精度；速度杠杆成立 |
| experiments/dense80_es_nostop.yaml | done | 6d-adaptive-k-sim 对照档（08-16 结案）：独立 NN + K=40 = MEAN 49.00 vs 官方 49.33（-0.33）——**NN 匹配无损，早停损失全在排除本身**（hp -15.83 联合池收窄最致命）；v2 的 mk=12 才是 hp 修复 |
| experiments/dense80_es_fusion.yaml | done | 6d-adaptive-k-sim v2 档（08-16 判负 -6.50）：mk12+weighted 级联 54.50 vs 61.00——hp 修复线失败（43.33<53.67）；粗位姿口径 mk12≥mk8 级联层不兑现 |
| experiments/dense80_es_score.yaml | archived | 6d-adaptive-k-sim v2.1 档（08-16）：先验规则 v2 判负后仅佐证，粗位姿口径优化预期级联层不兑现，跳过未跑；early_stop_signal 代码保留（默认 inlier）|
| experiments/dense80_fib24.yaml | done | 6d-fib24 判负（08-16）：MEAN -3.00（46.33 vs 49.33），hp -20.83 最重——120t 预筛 top-40 被自相似模板挤占；80t 饱和证实 |
| ablations/11_joint_templates.yaml | done | 6d-ablation 第 11 组（08-14 补）：J=1 45.33 / J=5 49.33 / J=10 46.00 / J=20 45.83——J 曲线双峰（J∈{5,12} 并列），增益集中 J≤5（+4.0）；默认 J=12 与 J=5 并列最优（08-16 结案）|
| experiments/dense80_ia_gateoff.yaml | done | 6d-ia-gateoff（08-16 结案）：gate-off 57.33 vs gate-on 61.20（-3.87）——门是保护机制（hp -15.83 全靠门挡），不阻塞真收益（duck +16.67 级联增益未被挡）|
| experiments/dense80_localt_off.yaml | done | 6d-localt-off（08-16 结案，判负 -7.17）：off 档 53.83 vs ia 基线 61.00——ape/hp/phone -10~-12（触发率最高物体受损最重）、duck/cat 持平；loc_alt 备选解码是正贡献保留，代价 +2.0s/帧 53% 帧触发如实披露 |
| experiments/dense80_refviews128.yaml | done | 6d-refviews 结案（08-20）：**新 champion MEAN ADD 79.66 / Proj 88.54 / 5cm5° 82.58（14968 帧，较 78.07 +1.59）**——128v 配方全量兑现：cat +6.64 / benchvise +4.87 / hp +2.56（干净口径）/ iron +2.57 / lamp +2.24 / can +1.77 / glue +0.52 / driller +0.44 / eggbox +0.34 / phone +0.08 / cam -0.53（噪声带）；零回退；保留物体一致性 2/3 精确（hp 旧库污染已记录）；run record 见 6d-refviews.md |
| experiments/dense80_refviews128.yaml（trio 档）| done | 6d-trio 结案（08-20）：**迭代数影响物体相关**——duck 7000 全量 49.58（+4.03）升级、ape 持平（-0.83）、hp 7000 更差（-9.16）保留 30k；30k 时代保留三物体的决定对 hp 正确；decode-all 搭车三物体全负/平（预筛 top-40 截断验证通过，非杠杆）；ape/hp 库恢复 .30k128bak；duck 正式切 7000/128v；champion 帧加权 MEAN 79.66→79.98 |
| experiments/dense80_gsrefiner.yaml | done | 6d-gsrefiner 判负结案（08-17，先简单验证未跑全量）：40 帧 duck 忠实 GS-Refiner 损失改善 4/恶化 28（旧损失 6/22，更差）；4× 降采样 2/26；三个误差带全净负；损失-误差相关 r=-0.16——渲染比较损失面与位姿误差解耦（"自洽地错"是损失面系统性属性）；GS-Pose 的 +36.5 来自弱 init 的大旋转误差，iter_align 已吃掉；剩余单目平移病态 RGB 不可见（tz-depth 闭环）；refiner 方向整体结案（代码：loss_mode/early_stop_abs/loss_downscale/refine_fallback_guard 保留作消融档）|
| 6d-pnp-multisol（诊断）| done | 挑战 2 判死：duck 60 帧 × 30 次 RANSAC 全单解——硬对应 + EPnP 无多解性；inlier 择优 60/60 命中；瓶颈确证候选池生成 |
| dense80_depthc_ia_track.yaml | done | 6d-track-seed 帧间跟踪种子：duck 50.83（+3.33 vs 基线）但低于 multi 55.83；代价 +40% 不划算；仅论文视频扩展素材 |
| dense80_depthc_ia_multirefine.yaml | done | 6d-multi-refine 种子级渲染对比优化判负：duck 49.17（-6.66 vs multi）——refiner 盆底择优失效（ADD -6.66 但 Proj +9.16）；渲染对比优化两轮判负结案 |
| dense80_depthc_ia_mmr.yaml | done | 6d-prescreen-mmr MMR 预筛多样性判负：duck 46.67（-0.83）——预筛阶段结案 |
| dense80_depthc_ia_fb.yaml | done | 6d-fallback-decode 失败帧自适应全解码判负：duck 46.67（-0.83）——解码侧结案 |
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
| experiments/dense80_refviews64_ctl.yaml | done | 6d-refviews64-ctl 归因结案（08-20）：cat 64v 现代配方 67.50——视图数贡献 +12.50（79%）、现代配方（锚点/深度） +3.33（21%）；**视图数是 refviews 增益主因**；cat 128v 库已从 .v128bak 恢复 |
| experiments/dense80_vs_16x1.yaml / 24x1.yaml / 24x2.yaml | done | 6d-viewstruct 结案（08-21，判死）：fibonacci 模式 roll 冗余=真实检索覆盖（与 cube8 消融结论不迁移）；V1 -4.17 / V2 -4.50 / V3 +0.50（边际不采纳）；ape/hp 对 roll 极敏感（-10.8）；次级发现 duck 24×2 +5.83（24 视角有益，可单物体确认）；champion 维持 80t（16×5）|
| experiments/dense80_fillnorm.yaml | done | 6d-fillnorm 判负结案（08-21）：MEAN 59.17 vs 基线 64.67（-5.50）——duck -15.00 / ape -10.83 崩（小物体像素预算）、cat 0.00（Proj +3.34 匹配变好）、phone +0.84；**查询侧尺度两方向全判负**（放大 768/superres + 缩小 fillnorm），bbox+20% 天然尺度=信息最优；代码保留作消融档（match_fill_norm 默认 0.0）|
