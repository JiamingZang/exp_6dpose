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
- `08-13 00:30`：01 组首跑（default.yaml base）暴露配置错误：**default.yaml 是
  40t 模板库（cube8×5）+ guided_refine:false，与主表配置（80t fibonacci +
  guided）口径不一致**，数字不可比主表；且 ape 40t 库缺 .pt（refine 需要）
  直接崩。修正：消融 base 一律用 `dense80_depthc_guided.yaml`。
- `08-13 01:00`：**缓存复用地图**（guided base）：K=1/5/10 档命中 8/12
  b2g0avtuk 遗留缓存（5 物体 121 帧全）；K=20 档 ape/duck/cat(119) 命中、
  holepuncher/phone 需补跑（b2g0avtuk 被 kill 时未跑）；K=40 档命中主表
  cache13_dc2（49138670）。09 组 5.0 档修正为 float（int 5 vs float 5.0
  hash 漂移 → 不命中主表，改为 5.0 后命中）。
- `08-13 01:05`：onboard 补齐脚本（/tmp/ablation_onboard_fill.py）入批处理：
  8t×4 + 24t×5 + 40t ape 补 pt 共 11 次（ape 40t npz 已备份 .40tprebak）；
  批处理链：01 → onboard fill → 02 → 09 → 07 → 08 → 03(dinov2patch) →
  tzdepth5（第四章 RGB-D 证据）。
- `08-13 02:13-02:32`：**GPU 竞争事故**——历史遗留的第二个批处理进程
  （00:55 起等 01_topk）与 00:50 的首个进程同台，02/09/07 三组全部 OOM
  （torch.OutOfMemoryError ×5），02 组 onboard 的 8t 模板已落盘但评测全废；
  02:32 起干净重跑（v3 脚本，当前 PID 469262）。
- `08-13 05:28-16:32`：02 组重跑成功（rc=0）：8t/24t/40t 全量 onboard +
  评测，结果见 Result 表。
- `08-13 16:32-`：09 组进行中（ε=3 全新匹配 3.6h 完成；ε=5.0 命中主表
  缓存 49138670 秒过逐物体重写；ε=8/10 全新匹配各 ~3.6h）。之后自动续
  07 → 08（osmesa）→ 03（dinov2patch）→ tzdepth5。
  监控信号：缓存文件 mtime（`ls -lt outputs/ablation_cache/`），日志因
  python 块缓冲滞后数小时，不可作实时信号。
- `08-13 22:10`：**07 耗时修正**——帧缓存的 cfg_hash 覆盖整个配置，selection
  非默认值（similarity/weighted）各自新 hash → **全新匹配各 ~3.6h**（不是
  缓存命中）。07 组预计 ~7.2h（inlier 档命中主缓存除外）。总链预计
  ~08-14 23:30 收尾（09 6h + 07 7.2h + 08 3.6h + 03 3.6h + tzdepth 2.7h）；
  inlier_ratio/reproj 策略由 6d-adaptive-k-sim 的 cand_* 缓存离线重排补齐
  （cand_ncorr/cand_reproj 已落盘，08-13），不再占用 GPU。
- `08-13 23:18`：**批处理重启（缓存 meta 修复 0299fbe）**——排查发现
  ε5.0 每物体 ~40min 是"假 cache hit"：缓存文件级 meta 比对含 matches_dir
  （主表运行是 matches13_dc2、消融是 None），_load_cache_records 整文件
  丢弃 → 主表 hash 档全部重匹配（ε5.0 已白烧 3.6h；07-inlier/08-3dgs 还
  会再各烧 3.6h）。修复：文件级 meta 只比对 cfg_hash（逐帧指纹兜底，
  206 测试通过），杀掉旧进程重启批处理——02 四档全部缓存命中（~15 min），
  07-inlier/08-3dgs 免烧 7.2h。新链：02 → 09（ε8/ε10 全新）→ 07 → 08 →
  03 → tzdepth，预计 ~08-14 20:00 收尾。
- `08-14 03:05`：**09 组 ε=3 预览（缓存聚合，非 run_ablation 正式输出）**：
  5 物体均值 ADD(S)@0.1d = **46.17**（duck 31.67 / ape 42.50 / cat 53.33 /
  holepuncher 43.33 / phone 60.00），vs 基线 ε=5.0（主表口径）49.17 =
  **-3.00**；Proj 持平（duck 82.50 vs 82.00）——更紧阈值减少对应数量、
  不伤投影精度但伤深度病态物体的 ADD。ε=8/10 待批处理出数（预计 06:10
  后）；正式数字以 ablation_ransac_eps.json 为准。
- `08-14 04:10`：**09 组 ε=8 预览（缓存聚合）**：MEAN **40.67**（duck 22.50 /
  ape 34.17 / cat 45.83 / holepuncher 42.50 / phone 58.33），vs 基线
  **-8.50**。ε 曲线呈锐峰：3→46.17（-3.00）、5→49.17、8→40.67（-8.50）——
  过松阈值放大假内点（"自洽地错"机制，§4.1）比过紧削减对应危害更大；
  duck 跌最狠（30.83→22.50）。ε=10 待出。
- `08-14 05:45`：**09 组正式结果（ablation_ransac_eps.json）**：
  ε=3 **46.17** / ε=5.0 **45.17** / ε=8 **40.67** / ε=10 **35.67**（5 物体均值）。
  **关键发现：ε=5.0（45.17）≠ 01 组 K=40（49.17）——同一配置两个数字**。
  时间线定位：01 K=40 写于 05:12（**joint dc2 修复 0f0d0bb 之前**），09 组
  由 16:32 启动的进程运行（**修复之后**）。逐物体：duck +5.00（30.00→35.00，
  修复动机物体）、ape -9.17（47.50→38.33）、cat -5.00、holepuncher -8.33、
  phone -2.50，**净 -4.00**。根因：MASt3R 成对重建逐对尺度漂移，合并集单一
  自校准深度比把第 2/3 模板对应整批误删，立体增益丢失（duck 的假内点被删
  是收益，其余物体的有效对应被删是损失）。**修复 1d6305f**：合并前按中位
  深度比对齐（joint_scale_align，默认关保持现状）；筛选档 A（align）/
  B（tau 0.12）已排入链（duck+ape 各 ~1.4h，ape 恢复即信号）。**主表 78.07
  为修复前代码；若筛选胜出 → 全量 champion 复跑；若失败 → 回退 0f0d0bb。**
- `08-14 07:15`：**决策：回退 0f0d0bb（fad8943）**——筛选（align/tau12）的
  期望值低于回退成本：即使 ape 恢复，post-fix+align 仍要重跑 01/02/09/主表
  （>50h）；而回退后 pre-fix 主线（78.07/K 曲线/01/02）全部成立，只需一次
  重启把 09/07/08/03/tzdepth 统一到 pre-fix（~38h 自动）。已删 post-fix
  缓存（1224de51/dabe9cfc/2fbd44c3/fe2de02e/49138670 × 5 物体，25 个文件）
  与 align 配置/链段；批处理重启（PID 537528，pre-fix 代码）。09 组将重跑
  pre-fix（~10h），ε5.0 应恢复 ≈49.17（与 01 K=40 一致性自检）。
  joint dc2 记为证伪实验：duck +5.00 / 净 -4.00，机制 = 逐对尺度漂移误删
  有效对应（§5.3 讨论素材）；per-template dc2（原功能）保留。

- `08-14 13:10`：**pre-fix ε=3 预览（缓存聚合）**：MEAN **48.73**（duck
  35.83 / ape 47.50 / cat 53.33 / holepuncher 47.50 / phone ~60.4），vs
  pre-fix 基线 ε5.0（49.17）= **-0.44 几乎持平**——post-fix 下 ε3 的 -3.00
  惩罚是 joint dc2 的伪影（紧阈值 × 深度检查叠加误删），回退后消失。
  ε8/ε10 待出（预计 22:50 前），正式数字以 ablation_ransac_eps.json 为准。
- `08-14 18:00`：**链保护**——post_consensus_chain.sh 今晨 11:17 超时死亡
  （720 次×60s 窗口不够），已改 5000 次重启（559082）；新增
  /tmp/chain_watchdog.sh（559083）：批处理结束后自动补拉起 verify/chain、
  清理残留批处理包装进程（其 cmdline 含 run_ablation_batch.sh 字样会卡死
  verify 的 pgrep 等待循环）。chain2（/tmp/post_chain2.sh，559911）排队：
  等 exp_fib24/DONE 后跑 11_joint_templates + ia-gateoff。
- `08-15 00:17`：**07-weighted 官方 49.50**（duck 30.83 / ape 47.50 / cat 53.33 /
  holepuncher 50.83 / phone 65.00）——与 similarity 逐物体完全相同、与 inlier
  仅差 ape 一帧（46.67→47.50）；**三档结案 inlier 49.33 / sim 49.50 /
  weighted 49.50**。逐帧分析（缓存聚合，00:00）：**88.2% 帧（529/600）三档
  数值位姿不同**（精化链把粗选差异放大进最终位姿），但 **ADD 命中结果仅
  1/600 帧（0.2%）不同**——择优判据对最终质量影响可忽略；"择优价值=为联合
  解/引导精化提供高质量起点"（论文 §3.6.2）获逐帧证据。预判（21:25
  "weighted 档也 ≈ 49-50"）命中 ✓
- `08-15 08:41`：**批处理结束**（tzdepth 后 08/03 重跑段全缓存命中）。
  链 1 启动（duck_kcurve_verify 已 checkout pre-fix fad8943）。
- `08-15 08:40`：**tzdepth 5 物体官方结案 MEAN 61.83**（duck 52.50 / ape 56.67 /
  cat 66.67 / hp 61.67 / phone 71.67）vs 冠军 ia 61.20（gap-oracle 端到端）——
  **+0.63 但高度异质**：tz 病态主导物体受益（duck +5.00 / hp +5.84 / cat +1.67），
  tz 已准物体回退（ape -3.33 / phone -5.83）——中值深度是近似非逐像素，叠加
  IoU 代理门无法保护 ADD。论文 §4.1.1(3) 已改写：RGB-D 可解性边界 = 逐像素
  深度（3D-3D 配准）才无副作用；单目平移信息极限论断保留。
- `08-15 06:00`：**03 matcher 官方结案 MEAN 10.16**（duck 0.00 / ape 0.83 /
  cat 10.83 / holepuncher 15.83 / phone 23.33）vs MASt3R 49.33（-39.2）——
  **几何对应是 MASt3R 成对解码的特有能力**：DINOv2 单图描述子检索召回够用
  但稠密对应弱纹理全灭（duck 0.00），纹理越弱差距越大；与 §4.1.1(2)
  对应质量=模型能力极限互证（05:00 预览命中 ✓）。论文 §5.3.3 匹配模型行已更新。
- `08-15 02:47`：**08 pyrender 臂真实数字（zfar 修复后）MEAN 1.67**（duck 0 /
  ape 0 / cat 0 / hp 3.33 / phone 5.00）——**成像域决定匹配质量**。排除渲染
  缺陷：模板库投影回检 100% 自洽（coord_maps 反投影全落 alpha 掩码内）；
  全 120 帧"成功"且内点 >1000 占 105-108/120 帧（与 3dgs 臂同级），但位姿
  全错——CAD 几何着色模板与真实照片域差使 MASt3R 对应**几何自洽但语义
  错位**（"自洽地错"从 ~50% 放大到 ~99% 帧）。3dgs 对照：49.33（含抛光）/
  49.17（粗位姿同口径，R_coarse 聚合）。论文 §5.3.3 渲染器行 + §3.3.2 证据
  标注已更新。08 组结案：**3DGS 照片级模板是稠密匹配成立的前提**。
- `08-15 01:20`：**08 pyrender 全空白模板根因定位与二次修复**——首跑 pyrender
  臂 0.00 分（全 120 帧失败）非 PnP/匹配问题：模板库 alphas/coord_maps 全零、
  images 全白。独立渲染探针定位：盒子网格正常、duck 网格正常、OSMesa
  GL 4.5 正常，唯独默认 `IntrinsicsCamera` 的 **zfar=10** 把 radius≈400 的
  渲染距离整体裁剪（近平面 0.05 也在物体内）。修复：
  `template_renderer.py` 加 `znear=radius/100, zfar=radius*4`；端到端冒烟
  通过（alphas 每视图 ~19.5k 像素）。空白库已删、9cac5520 失败缓存已清
  （zfar 是代码级修复，cfg_hash 不变，不清缓存会命中复用旧错误记录）；
  批处理加 08-refix 段，pyrender 臂重新 onboard+评测，预计 03:10 出真实数字。
- `08-15 01:00`：**精化链阶段贡献量化（主缓存 R_coarse vs m，论文 §4.1.1）**：
  5 弱物体 × 120 帧（guided+抛光口径），粗位姿独立命中率 49.2%（295/600），
  精化挽救 5 帧 vs 回归 4 帧（净 +1 帧，0.8% vs 0.7%）——**失败帧在精化前
  已确定，粗位姿就是决策点**；唯一的显著翻帧机制是 iter_align 重渲染
  （120 帧 duck 级联 +16.67）。与 gap-oracle"候选池生成=总瓶颈"互为印证；
  抛光单独在 5 弱物体上仅 +0.16（粗位姿 49.17 → 含抛光 49.33）。
- `08-15 00:30`：**08/03 组首跑崩溃定位与修复**——(1) 08 pyrender_cad 臂崩于
  PoseEstimator 无条件构造 PoseRefiner 缺 .pt（pyrender onboard 只建 .npz
  库不训练 3DGS）：load_ablation 新增 extra_overrides（按 sweep 值附加
  点号路径覆盖），08_renderer.yaml 给 pyrender_cad 臂关 refine_pose（guided
  不依赖 .pt 保留）; (2) 03 dinov2_patch 崩于 Dinov2PatchMatcher.match 返回
  3 值而 pipeline.py:786 解包 4 值：补第 4 返回值 top_full=None（无稠密解码，
  guided 自动跳过）。测试 206 过；批处理脚本追加两修复组重跑（tzdepth 后），
  全链重排：批结束约 06:30 → duck kcurve → consensus → adaptive-k → fib24。
  3dgs 臂粗位姿对照由主缓存 R_coarse（抛光前位姿）聚合，免重跑。
- `08-14 21:25`：**07-similarity 档（缓存聚合）MEAN 49.50**（duck 30.83 /
  ape 47.50 / cat 53.33 / holepuncher 50.83 / phone 65.00）≈ **inlier 49.33
  （几乎逐物体相同）**——**择优判据不敏感**：selection=similarity 的位姿
  输出同样经联合 PnP（J=12，top-12 sim 模板合并重解）替换，合并集与
  inlier 策略相同 → 输出几乎相同（Δ±0.17）。推论：**K=1（30.50）vs
  similarity（49.50）的 +19 分差距 = 联合 PnP + 多候选解码的共同贡献，
  而非择优判据**——K 曲线增益主因是"更多模板进联合解"，择优判据层的
  边际价值小（预判 weighted 档也 ≈ 49-50，待 0d4cd0d9 官方确认）。论文
  §5.3.3 择优判据行措辞按此写。
- `08-14 19:10`：**ε5 vs ε10 逐帧交叉分析（缓存，论文 §5.3.3 机制证据）**：
  变坏帧（ε5 对 ε10 错）101 vs 变好帧 33（duck 23:2 / ape 24:5 / cat 23:6 /
  holepuncher 23:15 / phone 8:5）——净 -68 帧；**变坏帧的内点数随 ε 膨胀
  +21~43%**（duck 28599→34762、cat 18302→26148、holepuncher 22715→28164）：
  过松阈值让自洽错误位姿获得更多假内点支撑（"自洽地错"放大机制的直接
  证据）；双错帧占大头（duck 81/120 本就在 ε5 失败），ε10 是加剧而非翻转。
- `08-14 18:45`：**07_selection 组**：inlier 档 = 49138670（主配置 hash，sweep 值
  与 base 相同 → 全缓存命中），结果 49.33 与基线精确一致（逐帧确定性复现 ✓，
  兼作缓存一致性检查）；**哈希勘误：fe2de02e 实为 similarity（非 08-13 笔记
  的 inlier）**——similarity 全新匹配中（~2.2h），weighted（0d4cd0d9）随后。
- `08-14 17:00`：**pre-fix ε8/ε10 预览（缓存聚合，方法与 ε5.0 校验一致——
  ε5.0 聚合 49.33 = 02-80t 官方值）**：ε8 MEAN **43.33**（duck 19.17 /
  ape 38.33 / cat 46.67 / holepuncher 51.67 / phone 60.83）= **-6.00**；
  ε10 前两物体 **22.08**（duck 13.33 / ape 30.83）= **-27 量级**。结论：
  **过松阈值惩罚是真实的（非 dc2 伪影）**——ε3 的惩罚是 dc2 伪影
  （pre-fix ε3 48.83 ≈ 基线 49.33），但 ε8/ε10 在 pre-fix 下依旧崩溃，
  与 post-fix 同向（40.67/35.67）；"假内点自洽地错"机制（§4.1）随 ε 放大，
  duck（弱纹理）最惨（30.83→19.17→13.33）。ε 曲线故事定型：3-5px 平台
  鲁棒，8-10px 崩溃，默认 5 合理。holepuncher ε8 +0.84 是唯一异质正例。
## Result

**01 topk 组已出（5 弱物体 × 120 帧均值，guided 粗位姿口径；K=1/5/10/20 来自
8/12 b2g0avtuk 遗留缓存，K=40 今天重跑）：**

| K | ADD(S)@0.1d | Proj@5px | 5cm5° |
|---|---:|---:|---:|
| 1 | 30.50 | 53.33 | 32.83 |
| 5 | 40.00 | 72.83 | 47.33 |
| 10 | 42.50 | 75.83 | 49.00 |
| 20 | 43.50 | 76.50 | 52.83 |
| 40 | 49.17 | 82.00 | 57.50 |

注意：duck 逐物体 K=20（20.0）< K=10（28.33）非单调——8/12 旧缓存存疑，
已排 duck 全档重跑验证（/tmp/duck_kcurve_verify.sh，v3 批处理后执行）。
其余物体 K=20≤K=40 单调性成立。K=1 档 8/12 缓存（default base 误跑 40t 库
faba5ca0）作废；765467ef（guided base）有效。

| 组 | baseline | this run | delta | note |
|---|---|---:|---:|---|
| 01 topk | K=40: 49.17 | K=1: 30.50 / K=5: 40.00 / K=10: 42.50 / K=20: 43.50 | 总体上升（duck K=20 单点回退，pre-fix 复现待 verify）| K=40=主表口径；K 曲线入论文 5.4；duck 全档验证排队 |
| 03 matcher | MASt3R: 49.33 | DINOv2-patch: **10.16** | -39.2 | **几何对应是 MASt3R 成对解码的特有能力**：单图描述子（DINOv2 patch）检索召回够用（§4.1.1(1) 96.7-100%）但稠密对应弱纹理全灭（duck 0.00 / ape 0.83），纹理越弱差距越大；官方 outputs/exp_dinov2patch/results/ + cache |
| 04 localization |  |  |  | 已有 6d-det-align 数字，跳过 |
| 06 scale_align |  |  |  | 默认档命中主表；false 档未排（价值低）|
| 07 selection | inlier: 49.33 | similarity: 49.50 / weighted: 49.50 | Δ≤0.17 | **择优判据不敏感结案**：三档命中结果仅 1/600 帧（0.2%）不同、逐物体几乎全同（唯一差异 ape 46.67→47.50 一帧）——联合 PnP（J=12）+精化级联吸收粗选差异；数值位姿虽在 88% 帧不同但全部落在同一命中桶内；K 曲线增益主因=更多模板进联合解（K=1 30.50→K=40 49.17）而非择优判据；官方 ablation_selection.json |
| 09 ransac_eps | ε5.0: 49.33 | ε3: 48.83 / ε8: 43.33 / ε10: 38.00 | ε 锐峰：3-5px 平台（Δ-0.5），≥8px 真崩溃（-6.0/-11.3）——过松阈值放大假内点自洽地错（§4.1），duck 最敏感（30.83→19.17→13.33）| 官方 ablation_ransac_eps.json（post-fix 备份 *_postfix_backup.json）；ε5.0=49.33 与 02-80t 一致 ✓ pre-fix 回退验证通过；holepuncher ε8 +0.84 唯一异质正例 |
| 10 segmenter |  |  |  | 已有 6d-loc-upper 数字，跳过 |
| 02 n_templates | 80t: 49.33 | 8t: 36.17 / 24t: 31.83 / 40t: 19.83 | **视角采样模式主导** | cube8 顶点采样系统性差（40t vs 80t 同为 5 旋转 -29.5）；cube8 下加旋转冗余有害（8t>24t>40t）；fibonacci 均匀覆盖是精度前提 |
| 08 renderer | 3dgs: 49.33（含抛光）/ 49.17（粗位姿同口径）| pyrender_cad: **1.67** | -47.7 | **成像域决定匹配质量**：CAD 几何着色模板对应海量但语义错位（内点 >1000 占 ~90% 帧，自洽地错近乎全覆盖），粗位姿全灭；3DGS 照片级模板是稠密匹配成立前提（§3.3.2 论断直接证据）；库几何自洽性投影回检 100% ✓ 排除渲染缺陷；官方 ablation_renderer.json |
| 05 geometry |  |  |  | 跳过（VGGT 未装）|
| tzdepth（§4.1.1(3) 证据） | 冠军 ia: 61.20 | 深度臂: 61.83 | +0.63 | **异质**：duck +5.00 / hp +5.84 / cat +1.67（tz 病态帧受益），ape -3.33 / phone -5.83（中值近似对已准帧回退）；逐像素深度才是无副作用充分条件 |

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
