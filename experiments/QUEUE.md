# 实验队列

只从这里领实验。新增实验先加一行，状态从 `todo` 改 `running` 后才能开跑。

## 冠军（当前基线，见 docs/STATE.md）

- **全量 MEAN ADD 69.74 / Proj 83.77 / 5cm5° 68.69**（14968 帧，论文主表口径）
- 120 帧子集历史数字（71.55、duck 33.3 等）**受 cache/rng 污染事故影响，引用前须干净复跑**
- 外部目标：GSPose 92.0；弱项 duck/ape/cat/holepuncher（全量 32.3/42.5/51.3/44.5）
- 已结案路线（禁止回退重跑）：30k 批量重训（13 物体 3 涨 9 跌）、tz_search（面积比信号死）、NCC 亚像素（噪声主导）、supersample2、择优歧义（duck 池无好假设）、μ 混合锚点、CAD 深度监督单独用、重采样视图、统一背景色、掩码腐蚀/交叉、SAM ViT-H、稳定先验软评分、查询侧超分

| ID | status | priority | config | run record | question | success line | notes |
|---|---|---:|---|---|---|---|---|
| 6d-full-linemod | done | 1 | `configs/current/dense80_depthc_guided.yaml`（回退保护）| `experiments/runs/6d-full-linemod.md` | 120 帧子集 71.55 能否在全量 LineMod 保持？| 全量 13 物体完成，mean ADD/Proj/5cm5° 入 STATE | **全量 MEAN 69.74 / 83.77 / 68.69（14968 帧）**；途中发现并修复 matches 损坏（cat 399 坏帧）；见 run record |
| 6d-refiner-v2 | done | 1 | 待改（`src/gaussian/pose_refiner.py`，配置新档位）| `experiments/runs/6d-refiner-v2.md` | 按 GS-Pose（SSIM+MS-SSIM、去 LPIPS、cosine lr 退火、edge_err 择优）+ 旧代码（mask_loss 形状主导、best-loss 回溯、多假设）重做 refiner 能否把精化从净负转正？| 120 帧子集 MEAN ≥ 71.55 且弱项（duck/ape/cat/holepuncher）任一 +5 | **判负**：cat 55.83（+4.2 唯一正）ape 45.0 持平 duck -6.6 holepuncher -5.8；duck 判据全失效（align_loss 51%）；多假设是最后机制差异待试（挂 task-2 之后） |
| 6d-pointmap-t1 | done | 1 | 离线验证（不改管线）+ `src/solver/rigid_align.py` 新求解器 | `experiments/runs/6d-pointmap-t1.md` | MASt3R pointmap（pts3d_q 已落盘）与 coord_map 的 3D-3D 刚体对齐能否替代 PnP？| 离线：好帧 3D 残差中位 <5mm；在线：ape/duck 120 帧 ADD ≥ 裸 PnP 且 M 类（duck）提升 | **不可行**：MASt3R 成对输出统一系（img1 系），查询相机系 3D 不存在；域差 34% 判死第三档；pointmap 价值=dc2 深度一致性（在用）；**t2/t3 连带清出队列**（依赖的 t1 前提不成立）|
| 6d-vggt-recon | done | 2 | sanity test（[模板渲染, 查询裁剪] → VGGT-1B 相对位姿）| `experiments/runs/6d-vggt-recon.md` | VGGT-1B 直接输出查询位姿（c2w，世界系=模板系）能否过域差判据（tz 中位偏离 <3%）？| tz 中位偏离 1.0 <3% 且旋转误差中位 <5° | **判死（1B）**：R_err 中位 94°、tz 反号；同图对 0.02° 但跨视角 145°——跨图位姿回归不可靠非纯域差；Omega（gated 待授权）可用同脚本复跑 |
| task1-1-stable-prior | done | 1 | `scripts/analysis/stable_pose_prior.py` | `experiments/runs/task1-1-stable-prior.md`| 测试序列物体是否处于稳定摆放（GT 朝上轴 vs 稳定族共识方向）？| GT 中位角差 >30° 判死 | **通过（判据修正）**：单方向 g* 迭代 + can 本底对照；duck 18.7/cat 15.8/holepuncher 18.0 vs can 20.1（同水平），ape 24.5（40% >30° 软信号）→ 进入任务 1.2 接入点 B（A 判死：模板物体姿态固定，加权无区分度）|
| task1-2-prior-insert | done | 1 | 接入点 B：`src/solver/selection.py` prior 项（A 判死）| `experiments/runs/task1-2-prior-insert.md` | 稳定先验软评分（score = inlier + λ·prior）能否提升弱物体？| 4 弱物体 ADD ≥+2mm 且 can 不降（±1mm）| **判负**：duck -6.67 / holepuncher -5.83 / cat +3.33 / ape +0.83（平均 -2.08）；机制=联合 PnP 吞择优 + 本底 18-20° 使 prior 重叠；死因"先验与失败帧错配" |
| 6d-loc-upper | done | 1 | `configs/current/dense80_depthc_gtmask.yaml` + `dense80_depthc_sam.yaml` | `experiments/runs/6d-loc-upper.md` | FastSAM 分割是否是弱物体瓶颈（gt_mask 上界下 ADD 提升多少）？| gt_mask 下 duck ADD ≥ +5 → 定位是瓶颈 | **结案**：4/5 弱物体定位侧瓶颈（holepuncher +9.17/cat +7.5/duck +5.84/can +5.0），ape 唯一匹配侧例外（-1.67）；SAM 对照判负（掩码 IoU 0.914 vs 0.910 几乎相同）；6d-weak-objects 分流：duck/cat/holepuncher 检测侧，ape 匹配侧 |
| 6d-cand-pool | done | 1 | `configs/current/dense80_depthc_mast3r.yaml`（fastsam 掩码 + 全解码）| `experiments/runs/6d-cand-pool.md` | duck 残余 5.0 差距是否来自候选池来源（DINOv2 top-40 vs MASt3R sim top-40）？| mast3r ranking ADD 接近 gt_mask（39.17）→ 候选池是瓶颈，治预筛 | **结论反转（08-08 事故复跑）**：干净 cache 下全解码 32.5 vs DINOv2 预筛 27.5（原"持平 33.33"是 cache 污染幽灵数字）——**候选池是主要瓶颈 +5.0**；治预筛/全解码 |
| 6d-mask-erode | done | 1 | `configs/current/dense80_depthc_erode{,_kb,gtfg,gtbc}.yaml` | `experiments/runs/6d-mask-erode.md` | FastSAM 掩码溢出背景环能否用边界腐蚀修复？| 腐蚀后 ADD 回升接近 gt_mask（39.17）→ 掩码后处理可落地 | **结案（全判负）**：腐蚀 1/3px 全降（30.0/26.67）；双向交叉（GT 掩码+fs bbox=19.17 灾难 / GT bbox+fs 掩码=30.0）全崩——掩码与 bbox 必须同源自洽；GT 上界拆解：~4 分坏帧候选选择失败 + ~2 分边界精度 |
| 6d-rng-fix | done | 0 | 代码修复（`src/pipeline.py` `_frame_rng` + frame_id 贯通）| `experiments/runs/6d-rng-fix.md` | self.rng 全局流被逐帧消耗、cache 命中帧跳过 → 120 帧对比抖动 ±6 分；改为每帧确定性种子（frame_id 派生）后 120 帧子集可复现吗？| duck 同 cache 状态两次复跑数字一致；不同 cache 状态数字一致 | **通过**：全空 cache vs 半满 cache 逐帧一致（30.83/81.67/40.83）；新口径干净基线=duck 30.83；历史子集数字全部作废需复跑 |
| 6d-det-align | done | 1 | `configs/current/dense80_depthc_gtbbox{,_gtmask,_pd}.yaml` | `experiments/runs/6d-det-align.md` | 检测框口径（GT bbox 上界）vs 无检测器端到端 78.07，定位代价多大？| 检测框口径数字出炉且差值可归因 | **结案（判负，三重归因）**：2a（框+全1+全解码）5 物体均值 -2.84；2c（恢复 DINOv2 预筛）≈ 基线 ±2 → 定位不是瓶颈；2b（GT 掩码）cat/duck +4~6（掩码质量是部分物体瓶颈）、ape -8.34（匹配侧例外）；全解码排序物体异质（duck 受益，与 6d-weak-objects 一致）|
| 6d-ablation-full | todo | 1 | `configs/ablations/*.yaml`（8 组，`scripts/eval/run_ablation.py --all`）| `experiments/runs/6d-ablation-full.md` | 论文 §3.3 八组消融（topk/几何/尺度/渲染器等，02/05/06/08 组需重建模板库）全量跑齐，支撑模板库构建+dc2 两个方法贡献的消融证据 | 8 组 × 13 物体数字落 outputs/ablation_<name>.json，入论文表 | 前置 6d-rng-fix（干净 cache）；已有子集数字优先但需全量口径复跑 |
| 6d-gap-oracle | done | 1 | `configs/current/dense80_depthc_ia_topk.yaml`（champion + topk_best [1,3,5,40]）| `experiments/runs/6d-gap-oracle.md` | 失败帧 90.5% 内点 >1000 却自洽地错：差距在候选池生成（无好假设）还是选择/验证（没挑中）？topk oracle 上界拆开 | 5 弱物体（duck/holepuncher/ape/phone/cat）120 帧 top1/3/5/40 档出炉；top40_best 高而 top1 低 → 选择损失，反之生成损失 | **结案（候选池整体无货）**：top40 池内 GT 择优均值 62.0 ≈ 端到端 61.2（+0.8）——择优完美也拿不到更多，候选池生成（匹配对应质量）是总瓶颈；分型：duck/cat 池有货但选择倒挂（-17.5/-6.7）、ape/phone/holepuncher 池没货但优化净赚（+1.7~+15）|
| 6d-weak-objects | done | 1 | `dense80_depthc_mast3r.yaml`（全解码）vs `dense80_depthc_guided.yaml`（基线）| `experiments/runs/6d-weak-objects.md` | rng-fix 新口径下全解码（MASt3R sim top-40）在 4 弱物体上是否仍系统性优于 DINOv2 预筛 top-40？| 任一弱物体全解码 ADD ≥ 基线 +3 | **结案（不泛化）**：duck +6.67 复现，ape/cat/holepuncher 全平/负（-0.83/-2.5/-0.83）——候选池仅 duck 有效，非一般性瓶颈；新口径基线 ape 47.5/cat 53.33/holepuncher 50.83 |
| 6d-prescreen2 | done | 1 | `dense80_depthc_p2.yaml`（top_k_prescreen: 60）| `experiments/runs/6d-prescreen2.md` | 两阶段候选筛选（DINOv2 粗召回 60 → MASt3R 精排 top-40）能否以 1.5× 解码代价兑现候选池收益？| duck ADD ≥ 全解码-1 且 matching 耗时居中 | **判负**：duck 34.17（只兑现全解码 37.5 的一半收益），且收益本身不泛化；代码保留作消融档 |
| 6d-iter-align-ext | done | 1 | `dense80_depthc_ia{1,3,norefine}.yaml` | `experiments/runs/6d-iter-align.md` | iter_align 全物体验证 + 迭代轮数消融 + 全 13 物体全量升级 | 任一物体 ADD ≥ 基线 +3 且无崩溃 | **结案（13/13 全正）**：全量 14968 帧 MEAN ADD 69.74→**78.07**（+8.34，Proj 87.17/5cm5° 81.31）；增益 +1.55（lamp）~+18.57（cat）；迭代轮数 2 轮为甜点；组合效应=iter_align 单独 +1.67 vs 级联 +16.67（120 帧 duck）|
| 6d-mask-geo | done | 1 | `configs/current/dense80_depthc_ia_geocand{,_gtmask}.yaml`（champion + solver.mask_geo_candidate: true）| `experiments/runs/6d-mask-geo.md` | 失败帧 48.5% 纯 t 错（RANSAC 深度病态）：GS-Pose 式掩码几何解析平移候选能否补上？FastSAM 版判负后 GT 掩码判别 | duck 120 帧 geocand ≥ 基线+3 且无 -3 回退 | **判负（机制无效）**：FastSAM 版 44.17（-3.33）、GT 掩码版 45.00（相对 GT 掩码档 51.67 为 -6.67）——非掩码拖累；质心反投影 xy 近似不成立 + 面积比被渲染掩码偏差污染 + align_loss 择优不可分；与 tz_search 结案互为印证 |
| 6d-tracking-speed | blocked | 3 | 待新增 | `experiments/runs/6d-tracking-speed.md` | 上帧位姿初始化能否把 7.1s/frame 降到 <1s？| 速度 <1s/frame 且 ADD 下降可解释 | 需要先定 tracking 协议 |

## 已完成（历史记录，勿重跑）

| ID | 结论 | 出处 |
|---|---|---|
| 6d-30k-invdepth-bank | 30k 批量重训整体失败（13 物体 3 涨 9 跌 1 平，glue -60.9）；ape/duck/holepuncher 保留 30k bank | EXPERIMENTS.md「30k 训练 + invdepth 锚点验证」+「refiner 负贡献发现」|
| 6d-30k-can-coordbank | can 回归源=30k 训练（30k+coord 40.0 < 30k+invdepth 63.3 < 7000+invdepth 70.8 < 7000+coord 87.5→裸 PnP 92.5）；can 恢复 .orig bank | `experiments/runs/6d-30k-can-coordbank.md` |
| 6d-refine-two-tier | 裸 PnP 70.97 vs 回退保护 71.55 vs 带 refine 36.7（holepuncher）；refiner 净负贡献，回退保护为通用修复 | EXPERIMENTS.md「refiner 回退保护：全 13 物体」|
| 6d-inlier-ratio | 择优歧义非根因（align_select 修 10 坏 13；duck 池无好假设 align 判对 55%）；择优类实验结案 | EXPERIMENTS.md + verify_align_select 诊断 |
| 6d-pointmap-t2/t3 | 连带清出：t1 判死（MASt3R 成对输出统一 img1 系、域差 34%），t2/t3 前提不成立 | `experiments/runs/6d-pointmap-t1.md` |
