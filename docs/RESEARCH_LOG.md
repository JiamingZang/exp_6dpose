# 研究日志（2026-07-31 → 2026-08-02）

本文件记录 exp_6dpose 从 GPU 端到端修复到 13 物体评测的完整过程。
每次关键决策的动机、验证数据与结论都在这里；代码级细节见
`VERIFICATION.md`，完整会话原始日志见 `docs/session/*.jsonl.gz`
（敏感信息已脱敏）。

---

## 1. 起点：GPU OOM 与管线打通（07-31）

- 3 个并发 MASt3R 进程峰值 >31.5GB → seg0 监督进程 batch4→batch2 重试兜底。
- 双链竞态（旧进程用绝对路径启动，kill 模式漏杀）→ 锁文件 + 完整 pgrep 模式。
- `torch.hub.load("facebookresearch/dinov2")` 不带 `:main` 会访问
  github.com 主站解析仓库（本机被断连）→ `RemoteDisconnected` 未被
  `except URLError` 捕获 → 崩溃。修复：固定 `:main` ref，走本地缓存。
- 效果：ape 全量 1172 帧端到端可跑。

## 2. 深度偏差 4% 的根因（08-01）

现象：tz 比率 0.958（预测深度偏浅 4.2%），ADD 只有 13.5%。

排除法验证链：
- 完美对应 PnP（tz 1.0012）→ 不是 PnP；
- 像素链精度验证 → 不是坐标换算；
- 掩码面积比深度（0.997）→ 不是尺度；
- 射线求交（0.93）→ 锚点确实偏内；
- 方向自检验（放大 X → 更深；缩小 X → 更浅）→ 小模型 = 浅深度。

结论：**coord_map 渲染的是高斯中心 μ 的 alpha 混合**；体渲染可见表面在
μ+2σ 处，中心壳系统性在表面内侧 ~4-7% → 3D 锚点偏小 → PnP 深度偏浅。

官方对照：diff-gaussian-rasterization 的深度 = **逆深度 alpha 混合**
（`expected_invdepth += (1/depth)·α·T`，Hierarchical 3DGS 深度正则化），
不是位置混合。旧代码（MyPose）就是官方深度正则化 + Depth Anything 管线。

## 3. CAD 深度监督训练（用户提议：用 CAD 渲染深度图训练 3DGS）

- 实现：CAD 网格 numpy z-buffer 光栅化（无 GL）→ 参考帧 GT 位姿渲染
  逆深度图 → `depth_l1_weight=0.3` 监督训练（134s/物体）。
- z-buffer 两个 bug：sub==0 时无法更新；背面剔除方向反转（屏幕 y 向下）→
  移除背面剔除，z-buffer 自然保留最近正面。
- 验证：120 帧 no-refine tz 中位 0.9997（vs 0.958），ADD 39.2%。

**但全量崩了（ADD 15.7%）**。深挖：
- μ 混合锚点仍有 ~3mm 切向收缩（深度监督只约束 1/z 标量，不约束切向）；
- 后段帧（对应点少/分布偏）对锚点尺度敏感 → tz 0.93-0.96。

## 4. 固定视图 + 逆深度锚点（架构自洽）

- 逆深度锚点 = 训练监督的同一渲染 → z/xy 同时正确（射线×表面深度），
  无切向收缩问题。
- 重训后 onboard 重采样模板视图（渲染距离 365.3 vs 363.4，差 0.5%）→
  matches 的 pix_t 与新模板错位 → **模板视图必须固定**（复用旧 poses）。
- `scripts/data/rebuild_bank_fixed_views.py`：旧视图 + 新高斯 + 逆深度锚点 +
  重算 dino_feats。
- ape 全量 1172 帧：**ADD 44.20% / Proj 88.91% / 5cm5° 57.00%**，
  tz 中位 0.9983（CAD 锚点基线 40.96% 被超越）。

## 5. 13 物体子集首轮（08-02 凌晨）

每物体 120 帧均匀采样（`subsample_frames` linspace），全链自动跑完：

| 物体 | ADD | Proj | 5cm5° |
|---|---|---|---|
| ape | 50.0 | 87.5 | 63.3 |
| benchvise | 83.3 | 87.5 | 74.2 |
| cam | 53.3 | 61.7 | 55.0 |
| can | 44.2 | 55.0 | 31.7 |
| cat | 34.2 | 65.8 | 40.0 |
| driller | 94.2 | 90.8 | 93.3 |
| duck | 17.5 | 60.8 | 24.2 |
| eggbox | **9.2** | **6.7** | **1.7** |
| glue | 56.7 | 70.0 | 44.2 |
| holepuncher | 28.3 | 72.5 | 47.5 |
| iron | 87.5 | 91.7 | 73.3 |
| lamp | 25.8 | 33.3 | 17.5 |
| phone | 54.2 | 63.3 | 54.2 |
| **MEAN** | **49.10** | **65.13** | **47.69** |

旧代码 MyPose 端到端 top1：ADD 49.49% / Proj 59.22%（13407 帧全量）。
top3/top5 为 GT 择优 oracle 上界，不可比。

## 6. eggbox 9.2% 的根因（08-02）

逐层排除：
1. 对称问题？对称感知 GT 正确率仍 0.9% → 排除；
2. 锚点问题？CAD 锚点同样 0.2% → 排除；
3. sims 分层正确率全 ~0.2% → 不是低置信随机错配；
4. 定位问题？GT 掩码定位 extract → ADD 97.5% → 定位也没坏
   （FastSAM 候选框与 GT 几乎重合，crop_box 是 (x0,y0,x1,y1) 格式
   曾被误读为 (x,y,w,h)）；
5. **真凶：训练背景色**。eggbox 浅黄 ≈ 白背景 → 边界像素 RGB 梯度≈0
   → 边界高斯糊 → 逆深度锚点系统性错 → 全链崩。

GSPose 官方对照：训练 `image * mask`（黑背景），`white_background=False`，
渲染背景 [0,0,0]，损失 `trunc_FG_mask` 前景截断。本库此前白背景偏离官方，
且 SSIM 全图统计会稀释物体区域约束。

修复：`configs/current/dense80_depth_bg0.yaml`（黑背景 + `depth_l1_weight: 0.6`）
→ eggbox **9.2% → 98.3%**（同 120 帧，Proj 94.2% / 5cm5° 80.0%）。

## 7. 对称感知 PnP（实现但非 eggbox 主因）

- BOP `models_info.json` 的 `symmetries_discrete`（eggbox/glue 各 1 个
  180° 类变换）→ `ransac_pnp(..., sym_transforms=...)`。
- 采样不展开（避免 EPnP 混合解），内点判定按对称展开投影取 min，
  LM 在内点最优分支上精化 — 与 ADD-S 口径一致。
- 结论：随机错配为主时（正确对应率 <1%）RANSAC 无一致样本可采，
  对称展开救不了；黑背景重训后无需对称展开也到 98.3%。

## 8. 第二轮：全物体黑背景重训（08-02 完成）

`scripts/maintenance/rerun13_bg0.sh`：11 物体（eggbox 已重训）黑背景 + depth 0.6
重训 → 重提取 120 帧 → 评估。**MEAN ADD 49.1% → 63.3%**，
eggbox 9.2 → 98.3、lamp 25.8 → 89.2、can 44.2 → 84.2。

但**深色物体崩了**：driller 94.2 → 61.7、cam 53.3 → 33.3
（黑背景 + 深色物体 = 白背景 + 浅色物体的同一问题）。
白背景重训（`configs/current/dense80_depth_w1.yaml`）恢复并超过首轮：
driller 95.0 / cam 55.8。

## 9. 最终 13 物体（按亮度选背景）

| 物体 | ADD | Proj | 5cm5° | 背景 |
|---|---|---|---|---|
| ape | 50.00 | 86.67 | 64.17 | 黑 |
| benchvise | 88.33 | 85.83 | 77.50 | 黑 |
| cam | 55.83 | 65.00 | 55.83 | 白 |
| can | 84.17 | 84.17 | 85.00 | 黑 |
| cat | 42.50 | 81.67 | 58.33 | 黑 |
| driller | 95.00 | 91.67 | 85.00 | 白 |
| duck | 28.33 | 80.83 | 42.50 | 黑 |
| eggbox | 98.33 | 94.17 | 80.00 | 黑 |
| glue | 71.67 | 74.17 | 55.00 | 黑 |
| holepuncher | 30.00 | 64.17 | 38.33 | 黑 |
| iron | 90.83 | 90.00 | 72.50 | 黑 |
| lamp | 89.17 | 80.83 | 80.00 | 黑 |
| phone | 55.00 | 61.67 | 52.50 | 黑 |
| **MEAN** | **67.63** | **80.06** | **65.13** | |

对比旧代码 MyPose 端到端 top1：ADD 49.49% / Proj 59.22%（13407 帧全量）。
注：本表为 120 帧均匀采样子集（1560 帧），旧代码为全量；子集数字略偏乐观，
全量（1172 帧/物体）待跑。

### 遗留短板
- duck 28.3 / holepuncher 30.0 / cat 42.5 / ape 50.0：多为 D 类
  （投影对但 ADD 错，逐帧深度误差）与 M 类（匹配对应不足）混合；
- 定位提速与 top_k=10 匹配提速已实现（批量 CLS + 背景填色一致，
  `src/detection/localize.py`），未做端到端验证。

## §10. ape 背景事故 + D 类深度诊断（2026-08-02）

### ape 事故（数据可信性）
1. rerun13_bg0 链只重训/重提取 12 物体，**漏了 ape**；汇总表里
   ape 50.0 实际是白背景旧 matches + 白背景 bank 评的，标"黑"是错的。
2. ape 全量黑背景 extract 用 `dense80_batch8.yaml`（继承默认
   `onboard.bg_color: 1.0`）→ 裁剪填白背景去匹配黑背景 bank 模板，
   域不匹配 → 全量 ADD 32.4 作废。
3. 修复（机制层）：bank npz 写入 `bg_color` 字段（onboard +
   rebuild_bank_fixed_views）；TemplateBank 读取；
   extract_matches 校验 bank.bg_color ≠ cfg 时直接报错。
   新配置 `configs/current/dense80_batch8_bg0.yaml`（bg_color 显式 0.0）。
4. 教训：任何"训练背景/模板背景/裁剪填充背景"三处一致性都应在
   产物中记录并在下游校验，不能靠配置默认值传递。

### D 类锚点诊断（scripts/analysis/diag_dclass.py，CPU）
- **coord_map 即物体系 3D 点，PnP 直接查表（pipeline.py:706），
  不需要再乘模板位姿**——诊断最初多乘了一次模板位姿变换，
  得到 -26% 假偏差；修正后：
- duck 120 帧匹配锚点深度偏差：边缘 median -0.2%、内部 -0.2%，
  逐帧 median -0.85%，无帧超 ±2% → **锚点本身是准的**。
- D 类真实失败模式（cache13_ds 逐帧 trans/rot）：
  - 双峰：约 20% 帧整体错（duck rot 8°/holepuncher rot 19.6°，
    模板选错或匹配崩）；其余帧 trans 10-35mm，恰好卡在
    ADD@0.1d 阈值（duck 直径 ~102mm → 阈值 10.2mm）边缘。
  - eggbox 对照：成功帧 trans med 6.9mm（阈值 15.4mm，富余）。
- 结论：D 类 = 部分"坏帧"（候选/模板选择） + 部分"tz 噪声带"
  （PnP 深度条件数，亚像素误差在 ~930mm 距离放大 ~1% tz）。
  光度精化（refine_pose，150 iter）已在跑但未救回 duck
  （消融验证中：dense80_norefine.yaml）。

### 缓存机制加固（同日）
- evaluate_object 帧级缓存新增内容指纹：首行 meta（matches_dir + 配置
  哈希），不匹配整体作废；追加模式保证 4 分片并行共享安全。
- 事故：ape 子集 stage3 复用 cache13_ds/ape.jsonl 旧缓存（12:53 白背景
  结果）→ 停掉重跑。cache13_ds 等无 meta 的旧缓存全部视为陈旧。

### D 类修复：查询侧深度一致性（进行中）
- 失败模式量化（duck no-refine）：120 帧中 29 帧整体错（rot med 45°，
  模板/匹配崩）+ 59 帧近失（Proj 对但 ADD 错：trans med 21mm、rot med
  6.1°，阈值 10.2mm）——即 tz/rot 噪声带。
- 光度精化（refine_pose，150 iter L1+SSIM+LPIPS+Dice）贡献仅 +5.0 ADD
  （26.7→31.7），救不回噪声带。
- 实现（代码完成，待 extract 重跑验证）：
  1. TemplateMatch/落盘新增 pts3d_q（MASt3R 成对重建查询侧 3D，度量尺度）
  2. ransac_pnp 新增 _ransac_pnp_depth：内点 = 重投影<ε 且深度比自校准
     后 |z_a - c·z_q| < 5%·c·z_q——把 5px 阈值内的错误对应按 3D 结构剔除
  3. 配置 solver.depth_consistency / depth_tau_frac（dense80_depthc*.yaml）
- 对照实验设计：duck 120 帧新 extract（batch8×2，~10min）→
  depthc_norefine stage3（快）对比 26.7 基线；有效再叠加 photometric refine。

## §11. ape 真值排查 + 全物体受控重跑（08-02 晚，结案）

### 排查结论（全部受控，120 帧子集，同代码）
| 配置 | ADD | 说明 |
|---|---|---|
| 黑0.6+高斯锚点 | 30.8 | 轮7/8 配方，ape 真值 |
| 黑0.3+高斯锚点 | 27.5 | depth 权重降低无益 |
| 黑0.3+CAD锚点 | 31.7 | CAD 锚点 ≈ 高斯锚点（0.6 深度监督已拉准 μ） |
| 白0.6+高斯锚点 | 11.7 | 白背景训练对 ape 明显差 |
| 白0.3+CAD锚点 | 36.7 | fv 配方复刻；训练曲线与 19:53 逐项一致仍到不了 50 |
| 白0.3+CAD(12:53 假) | 50.0 | 不可复现的历史状态，作废 |

- 背景色通过训练梯度影响 densify：白训练 30k 高斯 vs 黑 21k，
  但受控结果仍显示黑背景更稳。
- 关键教训：**0.6 深度监督已把 μ 拉准**（高斯锚点 ≈ CAD 锚点），
  在线匹配保持纯 3DGS 几何（无 CAD），符合"未知物体"设定。

### 事故与修复
- 批量 patch_cad_coord_maps 时未验证备份逻辑（`.orig` 存在即不备份），
  12 物体黑 0.6 高斯库被覆盖丢失 → 全量重训（黑0.6×10/白0.6×2）
  + 重提取 120 帧 + 重评估（4 并行 → 2 并行，3 进程 extract 即 OOM）。
- 教训：批量破坏性操作前必须验证备份完整与回滚路径。

### 结案数字（13 物体 × 120 帧，全真）
MEAN ADD 67.44 / Proj 81.54 / 5cm5° 66.54（ape 30.8 为真值；
cat 53.3 / phone 62.5 / duck 33.3 / can 90.0 较旧表提升；
holepuncher 24.2 最弱）。

## §12. D 类修复：查询侧深度一致性 PnP（08-03 凌晨）

- 实现：matches 落盘新增 pts3d_q（MASt3R 成对重建查询侧 3D，米制）；
  _ransac_pnp_depth 内点 = 重投影<ε 且 |z_anchor - c·z_q| < τ·c·z_q
  （c 为逐候选自校准尺度，吸收米/毫米单位差与成对尺度误差）。
- 第一轮 duck 验证（120 帧，tau=5%）：
  | 配置 | ADD | Proj | cm_deg |
  |---|---|---|---|
  | 基线 no-refine | 26.7 | 75.8 | 38.3 |
  | 深度一致性 no-refine | **31.7** | 71.7 | 37.5 |
  | 基线 +refine | 31.7 | 81.7 | 44.2 |
  | 深度一致性 +refine | 待出 | | |
- 解读：深度一致性单独 ≈ refine 效果（+5.0 ADD），Proj 略降（5px
  内错误对应被剔除的副作用，tau 敏感性待测：8%/10%）。
- Bug 修复：pts3d_q 落盘曾因 `p3q = p3q[0]` 取行导致 (120,) 错位
  （extract12_v4 的 matches 全受影响，需重提取才能用深度一致性）。

## §13. 迭代渲染对齐提出与全量结案（08-08 → 08-10）

- 08-08：6d-iter-align 提出（当前位姿重渲染 → 再匹配 → 重解 PnP，
  接受/拒绝门）；duck 30.83→47.50（+16.67）、ape +11.67；rng-fix 落地
  （每帧确定性种子，历史子集数字作废）
- 08-08 晚：迭代轮数消融（1 轮 46.67 / 2 轮 47.50 甜点 / 3 轮 5cm5°+3.33）；
  refiner 关断消融 32.50 → **组合效应坐实**（iter_align 单独 +1.67 vs
  级联 +16.67）；5/5 泛化全正
- 08-09~10：全 13 物体全量 champion 升级评估（14968 帧）；进程管理事故
  （4 路并发显存争抢误判停滞、误 kill lamp/phone）→ 暴露缓存重定向续跑
  缺陷并修复（_load_cache_records + 5 条回归）
- 08-10 结案：**13/13 全正，MEAN ADD 69.74→78.07（+8.34）**；
  级联配置入论文主表（dense80_depthc_ia champion）；eggbox 98.40 /
  driller 97.06 超 95%；四件套 + 论文/中期报告同步，commit 5ec4090
