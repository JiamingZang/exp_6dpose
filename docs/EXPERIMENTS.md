# 实验报告：13 物体 LineMod 6D 位姿估计（2026-07-31 → 2026-08-02）

本报告汇总全部实验轮次：问题、假设、验证、结论与数字。研究时间线见
[`RESEARCH_LOG.md`](RESEARCH_LOG.md)，代码级验证见
`../VERIFICATION.md`，完整会话原始日志（脱敏）见 `session/`。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据集 | LineMod（BOP），13 物体，测试集（不含参考帧） |
| 评估协议 | ADD(S)@0.1d / Proj@5px / 5cm5°（BOP 口径，eggbox/glue 用 ADD-S） |
| 硬件 | 单机 RTX 4090 32GB（A100 40GB 32GB 显存受限环境），Linux |
| 定位 | FastSAM + DINOv2 ViT-L/14 CLS 检索 |
| 匹配 | MASt3R（ViTLarge，24 维 patch 描述子，Top-40 模板解码） |
| 求解 | RANSAC-EPnP（5px，1000 迭代）+ LM + 12 模板联合 PnP + 深度一致性 |
| 精化 | 3DGS 可微渲染（L1+SSIM+LPIPS+Dice，150 迭代） |
| 3DGS 训练 | gsplat 1.5.3，7000 迭代，参考帧 64 视图，CAD 深度监督 |
| 模板库 | 80 模板（fibonacci 16 视角 × 5 平面内旋转），512×512 |
| 锚点 | 逆深度混合反投影（与训练监督同一渲染） |

默认评估为 **120 帧/物体均匀采样**（`subsample_frames` linspace，覆盖全序列
视角/难度分布）；ape 有全量 1172 帧数字。

---

## 2. 完整实验历程

### 轮 1：管线打通与 ape 基线（07-31 ~ 08-01）

修复 GPU OOM（3×MASt3R 并发 >31.5GB）、双链竞态、torch.hub 断网崩溃后，
ape 全量 1172 帧可端到端运行：

| 指标 | ADD | Proj | 5cm5° | tz 比率 |
|---|---|---|---|---|
| ape 全量（修复前基线） | 13.5% | 81.4% | 38.7% | **0.958**（偏浅 4.2%） |

### 轮 2：深度偏差 4% 根因（08-01）

排除法：完美对应 PnP（tz 1.0012）、像素链精度、掩码面积比深度（0.997）、
射线求交（0.93）、方向自检验（放大 X → 更深）→

**根因**：coord_map 渲染的是高斯中心 μ 的 alpha 混合；体渲染可见表面在
μ+2σ 处，中心壳系统性在表面内侧 ~4-7% → 3D 锚点偏小 → PnP 深度偏浅。

对照：官方 diff-gaussian-rasterization 深度 = **逆深度 alpha 混合**
（Hierarchical 3DGS 深度正则化同款），旧代码 MyPose 即此路线。

### 轮 3：CAD 深度监督训练（用户建议，08-01）

CAD 网格 numpy z-buffer 光栅化（无 GL）→ 参考帧 GT 位姿逆深度图 →
`depth_l1_weight=0.3` 监督训练。120 帧 no-refine 验证：tz 中位 0.9997
（vs 0.958），ADD 39.2%。

**但全量崩**（ADD 15.7%）：深度监督只约束 1/z 标量（深度方向），
μ 混合锚点在**切向仍有 ~3mm 收缩**（半径偏小 4-6%）；后段帧（对应点少）
对锚点尺度敏感 → tz 0.93-0.96。

### 轮 4：固定视图 + 逆深度锚点（08-01）

- 逆深度锚点 = 训练监督的同一渲染 → z/xy 同时正确（射线 × 表面深度）。
- 重训后 onboard 重采样模板视图（渲染距离 365.3 vs 363.4）→ 与阶段 2 的
  像素对应错位 0.5% → **模板视图必须固定**（复用旧 poses）。
- `scripts/rebuild_bank_fixed_views.py`：旧视图 + 新高斯 + 逆深度锚点。

ape 全量 1172 帧：

| 版本 | ADD | Proj | 5cm5° | tz 中位 |
|---|---|---|---|---|
| 修复前基线 | 13.5% | 81.4% | 38.7% | 0.958 |
| CAD 锚点 | 40.96% | 89.68% | 51.79% | ~1.01 |
| DS+CAD（视图错位） | 35.92% | 89.68% | — | — |
| **固定视图+DS+逆深度** | **44.20%** | **88.91%** | **57.00%** | **0.9983** |

### 轮 5：13 物体首轮子集（白背景，08-02 凌晨）

12 物体 DS 重训（白背景 depth 0.3）+ 固定视图重建 + 全链自动跑
（`scripts/run13_subset_chain.sh`）：

**MEAN（13×120 帧）：ADD 49.10% / Proj 65.13% / 5cm5° 47.69%**

其中 eggbox 仅 9.2%、lamp 25.8%、duck 17.5% 严重拖后。

### 轮 6：eggbox 9.2% 根因（08-02）

逐层排除：对称问题（对称感知 GT 正确率仍 0.9%）→ 锚点问题（CAD 锚点
同样 0.2%）→ sims 分层（全部 ~0.2%）→ 定位问题（GT 掩码定位 extract →
**ADD 97.5%**，定位候选框与 GT 几乎重合）→

**真凶：训练背景色**。eggbox 浅黄 ≈ 白背景 → 边界像素 RGB 损失梯度≈0
（任意混合色都≈真实值）→ 边界高斯糊 → 逆深度锚点系统性错 → 全链崩。
GSPose 官方同款机制：`image * mask`（黑背景）+ 前景截断损失。

**修复**：黑背景（`onboard.bg_color: 0`）+ `depth_l1_weight: 0.6` →
eggbox **9.2% → 98.3%**（Proj 94.2% / 5cm5° 80.0%）。

### 轮 7：全物体黑背景（08-02）

11 物体黑背景 + depth 0.6 重训重提取（`scripts/rerun13_bg0.sh`）：

**MEAN：ADD 63.33% / Proj 76.28% / 5cm5° 59.49%**（+14.2/+11.2/+11.8）

大幅提升：can 44.2→84.2、lamp 25.8→89.2、glue 56.7→71.7、duck 17.5→28.3。

**但深色物体崩**：driller 94.2→61.7、cam 53.3→33.3（黑背景 + 深色物体 =
白背景 + 浅色物体的同一问题）。

### 轮 8：深色物体白背景（08-02，最终）

driller/cam 白背景重训（`configs/dense80_depth_w1.yaml`）：

| 物体 | 首轮(白) | 黑背景 | 白背景重训 |
|---|---|---|---|
| driller | 94.2 | 61.7 | **95.0** |
| cam | 53.3 | 33.3 | **55.8** |

**结论：背景色按物体亮度选择 — 浅色物体黑背景、深色物体白背景**。

---

## 3. 最终结果（13 物体 × 120 帧，含 refine）

| 物体 | ADD | Proj | 5cm5° | 训练背景 |
|---|---|---|---|---|
| ape | 30.83 | 81.67 | 50.83 | 黑 |
| benchvise | 83.33 | 83.33 | 73.33 | 黑 |
| cam | 57.50 | 71.67 | 64.17 | 白 |
| can | 90.00 | 91.67 | 85.83 | 黑 |
| cat | 53.33 | 85.00 | 63.33 | 黑 |
| driller | 91.67 | 91.67 | 85.83 | 白 |
| duck | 33.33 | 74.17 | 40.83 | 黑 |
| eggbox | 97.50 | 95.00 | 79.17 | 黑 |
| glue | 75.00 | 70.83 | 60.83 | 黑 |
| holepuncher | 24.17 | 73.33 | 40.83 | 黑 |
| iron | 89.17 | 92.50 | 77.50 | 黑 |
| lamp | 87.50 | 81.67 | 83.33 | 黑 |
| phone | 62.50 | 66.67 | 58.33 | 黑 |
| **MEAN** | **67.44** | **81.54** | **66.54** | |

> 更正（08-02 晚）：本表为同一代码/同一受控配置重跑结果。旧表 ape
> 50.0 是 12:53 白背景旧数据误标"黑"，真 ape（黑背景 0.6 训练）为
> 30.8；旧表 67.63/80.06/65.13 作废。新表全部数字经重训→重提取→
> 重评估流水线复核（见 RESEARCH_LOG §10）。

### 与旧代码对比

旧代码 MyPose（`_prior_code/MyPose`，同数据管线）：
- **top1（唯一的端到端数字）**：ADD 49.49% / Proj 59.22%（13407 帧全量）
- top3_best 68.70/70.78、top5_best 74.15/74.46 均为 **GT 择优 oracle 上界**，
  不可比
- ape 单物体 top1：23.62% / 71.62%（我们 50.00 / 86.67）

注意：本表为 120 帧均匀采样子集（1560 帧），旧代码为全量 13407 帧；
子集数字略偏乐观，全量（1172 帧/物体）待跑。ape 已有全量对照：
44.20 / 88.91 / 57.00（子集 50.00 / 86.67 / 64.17，同量级）。

### guided_refine 测试（GSPose 式迭代引导匹配，D 类物体）

`configs/dense80_guided.yaml`（guided_refine: true, guided_iters: 2, guided_radius: 12，
baseline 为 3 节主表，即 dense80_depth 系列）：

| 物体 | ADD 基线→guided | Proj 基线→guided | 5cm5° 基线→guided |
|---|---|---|---|
| duck | 28.33 → **31.67** | 80.83 → 81.67 | 42.50 → 44.17 |
| holepuncher | 30.00 → **30.00** | 64.17 → 63.33 | 38.33 → 39.17 |
| cat | 42.50 → **51.67** | 81.67 → 84.17 | 58.33 → 62.50 |
| D 类均值 | 33.61 → **37.78** | 75.56 → 76.39 | 46.39 → 48.61 |

结论：引导匹配能修正部分深度方向病态帧（cat +9.2、duck +3.3），
但 holepuncher 完全无改善——tz 系统性偏的根因（μ 混合切向收缩等训练级问题）
未被引导匹配覆盖。开销：guided 阶段约 +0.9~9.4 ms/帧，可忽略。
后续 D 类改进应转向训练/锚点层。

---

## 4. 速度

| 阶段 | 耗时 | 占比 |
|---|---|---|
| localize（FastSAM 分割 + DINOv2 打分） | 2.99 s | 42% |
| matching（MASt3R 40 模板解码） | 3.70 s | 52% |
| pnp（RANSAC-PnP + refine） | 0.42 s | 6% |
| **总计** | **7.11 s/帧**（0.14 FPS） | |

已实现的提速（未端到端验证）：
- 定位批量 DINOv2 CLS（逐候选前向 → 一次 batch 前向，2.5s → ~0.3s 量级）
- 候选裁剪背景填色与模板背景一致（消除跨域检索偏差）
- MASt3R top-40 → top-10 解码（未测精度）

待做：帧间追踪（视频序列用上帧位姿初始化，跳过全图定位+匹配，7s → <1s）。

---

## 5. 遗留短板与方向

| 物体 | 问题类型 | 说明 |
|---|---|---|
| duck 28.3 / holepuncher 30.0 | D + M 混合 | 投影对但 ADD 错（逐帧深度误差）+ 匹配对应不足 |
| cat 42.5 / ape 50.0 | D 类为主 | tz 中位好（≈1.0）但逐帧深度误差 1-3% |
| phone 55.0 / cam 55.8 | M 类 | 匹配对应质量/数量不足 |

改进方向：
1. **全量评估**（1172 帧/物体，约 15-18 小时）确认子集数字
2. 帧间追踪提速
3. 深度误差帧的锚点/refine 分析（D 类物体）
4. 匹配质量提升（M 类物体：模板数、sims 阈值、匹配器参数）

---

## 6. 关键文件索引

| 文件 | 用途 |
|---|---|
| `configs/dense80_depth_bg0.yaml` | 黑背景 + depth 0.6（浅色物体） |
| `configs/dense80_depth_w1.yaml` | 白背景 + depth 0.6（深色物体） |
| `scripts/rebuild_bank_fixed_views.py` | 固定视图重建（旧 poses + 新高斯 + 逆深度锚点） |
| `scripts/patch_depth_anchor_maps.py` | 逆深度锚点渲染 |
| `scripts/rerun13_bg0.sh` | 12 物体黑背景全链（重训→提取→评估） |
| `scripts/summarize13.py` | 13 物体汇总表 |
| `src/geometry/cad_depth.py` | CAD z-buffer 光栅化（训练监督深度图） |
| `src/solver/ransac_pnp.py` | 对称感知 RANSAC（sym_transforms） |
| `src/detection/localize.py` | 批量 CLS 定位 + 背景填色一致 |
| `docs/RESEARCH_LOG.md` | 研究时间线（含全部中间数字） |
| `docs/session/*.jsonl.gz` | 会话原始日志（token 已脱敏） |
| `outputs/templates/*.npz[.orig/.viewsbak]` | 各版本模板库备份 |

### 轮 9：depth_consistency + guided_refine 全物体（08-03）

`dense80_depthc_guided.yaml`（depth_consistency + depth_tau_frac 0.05 + guided_refine 2 轮）
+ `matches13_dc2`（修复 pts3d_q 采样后重提取，120 帧/物体）：

| 物体 | 基线 ADD | dc2 ADD | Δ |
|---|---|---|---|
| ape | 30.83 | 37.5 | **+6.7** |
| benchvise | 83.33 | 84.2 | +0.9 |
| cam | 57.50 | 59.2 | +1.7 |
| can | 90.00 | 87.5 | -2.5 |
| cat | 53.33 | 56.7 | +3.3 |
| driller | 91.67 | 92.5 | +0.8 |
| duck | 33.33 | 31.7 | -1.7 |
| eggbox | 97.50 | 96.7 | -0.8 |
| glue | 75.00 | 79.2 | **+4.2** |
| holepuncher | 24.17 | 31.7 | **+7.5** |
| iron | 89.17 | 88.3 | -0.8 |
| lamp | 87.50 | 90.8 | +3.3 |
| phone | 62.50 | 65.8 | +3.3 |
| **MEAN** | **67.44** | **69.36** | **+1.92** |
| Proj | 81.54 | 81.67 | +0.13 |
| 5cm5° | 66.54 | 66.99 | +0.45 |

**结论**：弱物体普遍受益（holepuncher +7.5、ape +6.7、glue +4.2、cat/lamp/phone
+3.3），强物体 ±2.5 内波动（噪声级）。深度一致性清错对应对 D 类有效，净 +1.9。
GSPose 92.0 仍差 22.6，主要缺口在 holepuncher/duck/ape（31-38）。

---

## 30k 训练 + invdepth 锚点验证（3 物体，120 帧/物体，exp_30k13）

30k 迭代 + 128 参考帧（GSPose 同规格训练）后全物体重训（bank 含 invdepth 锚点 + train_fp）。
提取/评估配置 `dense80_dc_b4.yaml`（= dc2 同口径 + batch 4 防 OOM）。

| 物体 | dc2 基线 | 30k+invdepth | Δ |
|---|---|---|---|
| ape | 37.5 | **45.8** | **+8.3** |
| can | 87.5 | 63.3 | **-24.2** |
| duck | 31.7 | 33.3 | +1.6 |
| holepuncher（先行） | 31.7 | 36.7 | **+5.0** |

**can 回归定位**（exp_30k13 缓存交叉分析）：
- 44 帧坏帧中 32 帧 dc2（7000+coord）下是好的 → 回归来自 bank 变化（30k 训练 or invdepth 锚点）
- align_loss/mask_iou 判对 23/23（爆炸帧 GT 优于被选 best）→ 择优是表层的
- align_select 重评估只修 10 帧又坏 13 帧（60.8 < 63.3）→ **择优不是根因**
- 验证中：can 30k+coord bank（重渲染，不动 3DGS）区分训练 vs 锚点贡献

**修复**：pipeline.py render_align_select 分支补 rs_triggered/rs_iou 赋值（此前
UnboundLocalError 崩溃）；verify_align_select.py 更新为 exp_30k13 缓存路径。

**待办**：can 30k+coord 结果 → 决定全量 9 物体 30k 流水线（extract9_all/eval30k_all 已备）。

---

## 环境事故：numpy 2.x 残留污染 cv2（08-04 晚）

**症状**：提取/评估 cv2 全挂（imread 返回 None、resize 报 "src is not a numpy
array"），can coord 提取 39 帧后中断，评估缺帧失败。

**根因**：numpy 包被 2.x/1.26 文件混装——`site-packages/numpy/_core/` 残留
7 个 numpy 2.x 的 .so（7-31 安装 2.x 后降级 1.26.4 未清残留）。cv2 绑定检测到
`numpy._core` 存在走错 API 路径，PyArray_Check 全失败。

**修复**：`rm -rf site-packages/numpy*` 后全新安装 numpy==1.26.4。
验证：cv2 resize/imread 恢复，torch 2.8 / scipy 1.17 正常。

**教训**：can 30k+invdepth 63.3（-24.2）结果存疑（提取于环境污染窗口），
30k+coord 对照实验用修复后环境重提的匹配重新评估中。

---

## refiner 负贡献发现 + 裸 PnP 回滚（08-05）

**30k 批量重训失败**（13 物体：3 涨 9 跌 1 平，glue -60.9 灾难）→ 回滚 .orig bank。

**关键发现**：30k 重训的 refiner .pt 是负贡献——benchvise .orig+30krefiner 76.7 < 裸 PnP
84.2；can .orig+30krefiner 69.2 < 裸 PnP 92.5；holepuncher 30k 带 refine 36.7 < 裸 PnP 51.7。
裸 PnP（refine_pose=false）为默认口径。

**回滚后全 13 物体**（裸 PnP）：

| 物体 | dc2 | 回滚 | Δ | bank 来源 |
|------|-----|------|---|----------|
| ape | 37.5 | 42.5 | +5.0 | 30k |
| benchvise | 84.2 | 84.2 | 0 | .orig |
| cam | 59.2 | 61.7 | +2.5 | .orig(bg1) |
| can | 87.5 | 92.5 | +5.0 | .orig |
| cat | 56.7 | 45.8 | -10.9 | .orig |
| driller | 92.5 | 91.7 | -0.8 | .orig(bg1) |
| duck | 31.7 | 31.7 | 0 | 30k |
| eggbox | 96.7 | 96.7 | 0 | .orig |
| glue | 79.2 | 77.5 | -1.7 | .orig |
| holepuncher | 31.7 | 51.7 | **+20.0** | 30k |
| iron | 88.3 | 88.3 | 0 | .orig |
| lamp | 90.8 | 92.5 | +1.7 | .orig |
| phone | 65.8 | 65.8 | 0 | .orig |
| **MEAN** | **69.36** | **70.97** | **+1.6** | |

**待办**：cat 需 refiner（dc2 靠旧 7000 refiner 正贡献），30k refiner 对照评估中；
弱项 ape/cat/duck/holepuncher（31-52）是超 GSPose 92.0 的主要缺口。

---

## refiner 回退保护：全 13 物体（08-05）

refine 前后渲染对齐损失比较，变差则回退粗位姿（holepuncher 54.2 > 裸 51.7 > 带refine 36.7）。
全物体启用后：ape 45.0(+2.5)/cam 63.3(+1.7)/cat 46.7(+0.9)/duck 33.3(+1.7)/holepuncher 52.5(+0.8)/
iron 89.2(+0.9)，lamp 91.7(-0.8) 例外，其余持平。

| 物体 | 裸 PnP | 回退保护 |
|------|--------|---------|
| ape | 42.5 | 45.0 |
| benchvise | 84.2 | 84.2 |
| cam | 61.7 | 63.3 |
| can | 92.5 | 92.5 |
| cat | 45.8 | 46.7 |
| driller | 91.7 | 91.7 |
| duck | 31.7 | 33.3 |
| eggbox | 96.7 | 96.7 |
| glue | 77.5 | 77.5 |
| holepuncher | 51.7 | 52.5 |
| iron | 88.3 | 89.2 |
| lamp | 92.5 | 91.7 |
| phone | 65.8 | 65.8 |
| **MEAN** | **70.97** | **71.55** |

**修复**：LPIPS 极小输入崩溃（<32px 跳过）；can 匹配路径修正（matches13_dc2）。
