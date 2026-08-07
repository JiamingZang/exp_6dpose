# 实验队列

只从这里领实验。新增实验先加一行，状态从 `todo` 改 `running` 后才能开跑。

## 冠军（当前基线，见 docs/STATE.md）

- **MEAN ADD 71.55**（120 帧子集，回退保护：refiner 精化前后渲染对齐损失择优，变差回退粗位姿）
- 外部目标：GSPose 92.0，差 20.5；弱项 duck 33.3 / ape 45.0 / cat 46.7 / holepuncher 52.5
- 已结案路线（禁止回退重跑）：30k 批量重训（13 物体 3 涨 9 跌）、tz_search（面积比信号死）、NCC 亚像素（噪声主导）、supersample2、择优歧义（duck 池无好假设）、μ 混合锚点、CAD 深度监督单独用、重采样视图、统一背景色

| ID | status | priority | config | run record | question | success line | notes |
|---|---|---:|---|---|---|---|---|
| 6d-full-linemod | done | 1 | `configs/current/dense80_depthc_guided.yaml`（回退保护）| `experiments/runs/6d-full-linemod.md` | 120 帧子集 71.55 能否在全量 LineMod 保持？| 全量 13 物体完成，mean ADD/Proj/5cm5° 入 STATE | **全量 MEAN 69.74 / 83.77 / 68.69（14968 帧）**；途中发现并修复 matches 损坏（cat 399 坏帧）；见 run record |
| 6d-refiner-v2 | done | 1 | 待改（`src/gaussian/pose_refiner.py`，配置新档位）| `experiments/runs/6d-refiner-v2.md` | 按 GS-Pose（SSIM+MS-SSIM、去 LPIPS、cosine lr 退火、edge_err 择优）+ 旧代码（mask_loss 形状主导、best-loss 回溯、多假设）重做 refiner 能否把精化从净负转正？| 120 帧子集 MEAN ≥ 71.55 且弱项（duck/ape/cat/holepuncher）任一 +5 | **判负**：cat 55.83（+4.2 唯一正）ape 45.0 持平 duck -6.6 holepuncher -5.8；duck 判据全失效（align_loss 51%）；多假设是最后机制差异待试（挂 task-2 之后） |
| 6d-pointmap-t1 | done | 1 | 离线验证（不改管线）+ `src/solver/rigid_align.py` 新求解器 | `experiments/runs/6d-pointmap-t1.md` | MASt3R pointmap（pts3d_q 已落盘）与 coord_map 的 3D-3D 刚体对齐能否替代 PnP？| 离线：好帧 3D 残差中位 <5mm；在线：ape/duck 120 帧 ADD ≥ 裸 PnP 且 M 类（duck）提升 | **不可行**：MASt3R 成对输出统一系（img1 系），查询相机系 3D 不存在；域差 34% 判死第三档；pointmap 价值=dc2 深度一致性（在用） |
| 6d-pointmap-t2 | todo | 2 | 第一档实现 + 置信度加权稠密对齐 | `experiments/runs/6d-pointmap-t2.md` | 稠密 pointmap 对齐能否把 duck/ape 对应点供给从"匹配数"解耦成"前景像素数"（治 M 类病）？| duck/ape 120 帧 ADD +5 且无大类崩溃 | 第二档：弱物体前景像素全部参与对齐 |
| 6d-pointmap-t3 | todo | 3 | 多视角 MASt3R-SfM/VGGT 全局对齐 | `experiments/runs/6d-pointmap-t3.md` | 8-16 模板视角+查询图全局对齐直接读位姿（无 Top-K/PnP/择优）？| ape/duck 120 帧 ADD ≥ 现管线 | 第三档：范式级；依赖第一档验证渲染-真实图 pointmap 域差 |
| 6d-vggt-recon | done | 2 | sanity test（[模板渲染, 查询裁剪] → VGGT-1B 相对位姿）| `experiments/runs/6d-vggt-recon.md` | VGGT-1B 直接输出查询位姿（c2w，世界系=模板系）能否过域差判据（tz 中位偏离 <3%）？| tz 中位偏离 1.0 <3% 且旋转误差中位 <5° | **判死（1B）**：R_err 中位 94°、tz 反号；同图对 0.02° 但跨视角 145°——跨图位姿回归不可靠非纯域差；Omega（gated 待授权）可用同脚本复跑 |
| task1-1-stable-prior | done | 1 | `scripts/analysis/stable_pose_prior.py` | `experiments/runs/task1-1-stable-prior.md`| 测试序列物体是否处于稳定摆放（GT 朝上轴 vs 稳定族共识方向）？| GT 中位角差 >30° 判死 | **通过（判据修正）**：单方向 g* 迭代 + can 本底对照；duck 18.7/cat 15.8/holepuncher 18.0 vs can 20.1（同水平），ape 24.5（40% >30° 软信号）→ 进入任务 1.2 接入点 B（A 判死：模板物体姿态固定，加权无区分度）|
| task1-2-prior-insert | done | 1 | 接入点 B：`src/solver/selection.py` prior 项（A 判死）| `experiments/runs/task1-2-prior-insert.md` | 稳定先验软评分（score = inlier + λ·prior）能否提升弱物体？| 4 弱物体 ADD ≥+2mm 且 can 不降（±1mm）| **判负**：duck -6.67 / holepuncher -5.83 / cat +3.33 / ape +0.83（平均 -2.08）；机制=联合 PnP 吞择优 + 本底 18-20° 使 prior 重叠；死因"先验与失败帧错配" |
| 6d-weak-objects | todo | 2 | 待定 | `experiments/runs/6d-weak-objects.md` | duck/ape/cat 失败帧（proj<5px 占 70%）有无训练/锚点级修复？| 任一弱项 +5，MEAN 不降 | 已确认是匹配精度极限（align 判对率 55%），需换信息源而非测试时微调 |
| 6d-tracking-speed | blocked | 3 | 待新增 | `experiments/runs/6d-tracking-speed.md` | 上帧位姿初始化能否把 7.1s/frame 降到 <1s？| 速度 <1s/frame 且 ADD 下降可解释 | 需要先定 tracking 协议 |

## 已完成（历史记录，勿重跑）

| ID | 结论 | 出处 |
|---|---|---|
| 6d-30k-invdepth-bank | 30k 批量重训整体失败（13 物体 3 涨 9 跌 1 平，glue -60.9）；ape/duck/holepuncher 保留 30k bank | EXPERIMENTS.md「30k 训练 + invdepth 锚点验证」+「refiner 负贡献发现」|
| 6d-30k-can-coordbank | can 回归源=30k 训练（30k+coord 40.0 < 30k+invdepth 63.3 < 7000+invdepth 70.8 < 7000+coord 87.5→裸 PnP 92.5）；can 恢复 .orig bank | `experiments/runs/6d-30k-can-coordbank.md` |
| 6d-refine-two-tier | 裸 PnP 70.97 vs 回退保护 71.55 vs 带 refine 36.7（holepuncher）；refiner 净负贡献，回退保护为通用修复 | EXPERIMENTS.md「refiner 回退保护：全 13 物体」|
| 6d-inlier-ratio | 择优歧义非根因（align_select 修 10 坏 13；duck 池无好假设 align 判对 55%）；择优类实验结案 | EXPERIMENTS.md + verify_align_select 诊断 |
