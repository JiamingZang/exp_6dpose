# exp_6dpose —— 基于3DGS模板匹配与几何后验证的未知物体6D位姿估计（实验代码）

硕士论文《基于3DGS模板匹配与几何后验证的未知物体6D位姿估计》的完整实验代码库。
论文正文见 `../thesis1_6d_pose.md`；各模块 docstring 均标注了对应的论文章节与公式。

> **当前运行配置（2026-08-02 起）**：与下文论文原描述的差异集中在
> （1）3DGS 训练增加 **CAD 深度图监督**（`gaussian.depth_l1_weight`，
> 参考帧 GT 位姿渲染的逆深度图）；（2）训练/渲染背景改**黑色**
> （`onboard.bg_color: 0`，浅色物体在白背景上边界高斯糊导致锚点系统性错，
> 见 `docs/RESEARCH_LOG.md` §6）；（3）模板库 80 模板
> （fibonacci 16×5 视角，非 40 模板 cube8×5）；（4）模板 3D 锚点用
> **逆深度混合反投影**（与训练监督同一渲染，非 μ 位置混合，§3-4）；
> （5）重训后**固定模板视图**（复用旧 poses，防止与阶段 2 像素对应错位，§4）；
> （6）对称物体 PnP 支持 **BOP 离散对称展开**（`ransac_pnp(..., sym_transforms)`，§7）。
> 完整实验过程与数据见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)（实验报告）
> 与 [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)（研究时间线），
> 会话原始日志（脱敏）见 `docs/session/`。

## 方法管线（论文 2.6.4 算法 1）

**离线（每物体一次）**：CAD 点云采样（主）/ VGGT 前馈重建（补充，2.2）
→ 物理尺度对齐 `s = f_query / f_ref`（2.2.3）
→ 3DGS 训练（gsplat，7000 迭代，`L = 0.8·L1 + 0.2·(1-SSIM)`，自适应密度控制，2.3.1）
→ 模板渲染：**8 立方体顶点视角 × 5 平面内旋转（72°间隔）= 40 模板**，256×256，
每模板记录位姿 `P_m` 与 3D 坐标图 `C_m`（alpha 混合渲染物体坐标，2.3.2/2.3.3）
→ DINOv2 模板 CLS 特征缓存。

**在线（每帧）**：FastSAM 自动掩码（主实验；SAM ViT-H 为消融对照）+ DINOv2 ViT-L/14 CLS 余弦相似度定位
（模板 max 聚合，argmax 选掩码，bbox 扩 20% 裁剪，2.4）
→ MASt3R（`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`）局部特征点积相似度，
模板级分数 `sim(m) = 查询前景像素对模板最大相似度均值`，Top-K 选择
（默认 K=40=全部模板，即不做候选裁剪；`template_ranking=dinov2` 复用定位相似度预筛，K<40 时只解码 K 个模板，2.5/3.4.3）
→ 每模板：互最近邻 + cycle consistency（τ=5px）、相似度阈值 0.3、采样 N_s=4096（2.5.2/2.5.3）
→ RANSAC-EPnP（重投影阈值 5px，置信度 0.999，迭代 1000），内点数择优输出（2.6）。

## 目录

```
exp_6dpose/
├── configs/
│   ├── current/               # 当前主线/复现入口：default、dense80_depthc_guided、legacy_mypose 等
│   ├── archive/               # 历史验证/失败路线/只为追溯保留的 dense80_* 配置
│   ├── ablations/             # 10 组消融（论文 4.3.1–4.3.9），每组一个 yaml
│   ├── experiments/           # 一次性实验配置归档（topk/背景/gtmask 变体）
│   └── *.yaml                 # 不保留根目录兼容入口；新实验必须显式选择 current/archive
├── src/
│   ├── geometry/              # 视角采样(2.3.2)、尺度对齐(2.2.3)、位姿/投影工具
│   ├── gaussian/              # 3DGS 训练(2.3.1)、模板+3D坐标图/深度图渲染(2.3.3) [GPU]
│   ├── detection/             # SAM+DINOv2 零样本定位(2.4)、YOLO+旧式裁剪(§8)  [GPU]
│   ├── matching/              # MASt3R 封装(2.5)、对应过滤、深度反投影提升(§8)
│   ├── solver/                # RANSAC-EPnP(2.6.1)、多候选择优/排序(2.6.2)
│   ├── datasets/              # BOP LineMod loader、PLY 读取、VGGT 接口
│   ├── metrics/               # ADD/ADD-S/Proj@5pix/5cm5°(3.1.2)、旧格式对接(§8)
│   └── pipeline.py            # onboard_object + PoseEstimator + evaluate_object
├── scripts/
│   ├── data/                  # 数据下载、3DGS onboard、固定视图模板重建
│   ├── eval/                  # LineMod 评测、速度、13 物体汇总、消融批跑
│   ├── analysis/              # 匹配提取、D 类诊断、深度/对齐/GT mask 验证
│   ├── maintenance/           # 一次性补丁和历史重跑链脚本
│   ├── experiments/           # 一次性实验脚本归档
│   └── *.py/*.sh              # 兼容 wrapper：旧命令 `python scripts/eval/run_linemod.py` 仍可用
├── docs/STRUCTURE_AUDIT.md    # 本次结构问题清单与整理依据
├── setup_gpu.sh               # GPU 机器一键部署（依赖+MASt3R克隆+权重下载）
├── requirements.txt           # GPU 完整依赖
├── requirements-local.txt     # 本地 CPU 测试依赖
└── tests/                     # 157 个 pytest 单测，本地（macOS，无CUDA）全绿
```

新实验优先写 `configs/current/...` 与 `scripts/<类别>/...`；历史实验日志可能保留旧路径；实际运行以本节和 AGENTS.md 的分类路径为准。

**GPU/CPU 边界**：`src/gaussian`、`src/detection`、`src/matching/mast3r_wrapper`、
`src/datasets/vggt_recon` 只能在 Linux GPU 机器运行；本地导入不报错，
实例化时抛出带安装指引的 `ImportError`。匹配过滤、PnP、指标、loader、
视角采样等纯逻辑全部本地可测。

## 本地开发（macOS，无 CUDA）

```bash
cd exp_6dpose
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m pytest tests/ -q     # 应全绿（见文末测试结果）
```

## GPU 机器从零复现论文主表

硬件：单张 NVIDIA RTX 4090（24GB）即可；≥16GB 显存的卡均可（3DGS 训练与
MASt3R 40 对解码峰值约 10–14GB）。系统：Ubuntu 20.04+，CUDA 12.x，
**torch>=2.7**（gsplat 1.5.3 的硬性要求，见 `requirements.txt`）。
外部 API 调用与 `third_party/` 各真实仓库的逐项核对记录见
[`VERIFICATION.md`](VERIFICATION.md)。

```bash
# 0. 环境（约 15 分钟）
conda create -n pose6d python=3.10 -y && conda activate pose6d
git clone <本仓库> exp_6dpose && cd exp_6dpose
bash setup_gpu.sh            # 装依赖 + 克隆 mast3r + 下载 MASt3R/SAM 权重(~4.9GB)
source env.sh                # 每个新 shell 都要（设置 mast3r PYTHONPATH）

# 1. 数据（约 6GB，10-30 分钟视网速）
bash scripts/data/download_data.sh data
# 确认 configs/current/default.yaml: dataset.root 指向 data/lm

# 2. 离线 onboard：13 物体（每物体 3DGS 7000 迭代约 4-6 分钟，共约 1-1.5 小时）
python scripts/data/onboard_object.py

# 3. 主实验：LineMod 13 物体全量评测（论文 3.2 表 1）
#    每帧约 0.6-1.2s（SAM 自动掩码为主要开销），13 物体 ×~1000 帧 ≈ 3-5 小时
python scripts/eval/run_linemod.py
#    → outputs/linemod_main.json：每物体与均值的 ADD(S)@0.1d / Proj@5pix / 5cm5°

# 4. 速度表（论文 3.4）
python scripts/eval/run_speed.py --object ape --n-frames 100

# 5. 消融（论文 3.3，8 组）。注意 02/05/06/08 需重建模板库（脚本自动 onboard）
python scripts/eval/run_ablation.py --ablation configs/ablations/01_topk.yaml
python scripts/eval/run_ablation.py --all          # 全部 8 组（约 1-2 天，可拆分并行）
#    → outputs/ablation_<name>.json
```

冒烟验证（跑通即环境正确，约 10 分钟）：

```bash
python scripts/data/onboard_object.py --objects ape
python scripts/eval/run_linemod.py --objects ape --max-frames 50
```

### 预计时长与显存汇总

| 阶段 | 时长 | 峰值显存 |
|---|---|---|
| onboard 单物体（3DGS 7000 迭代 + 40 模板 + DINOv2 特征） | ~5 min | ~6GB |
| 在线单帧（SAM ViT-H + DINOv2 + MASt3R 40对 + Top-5 PnP） | ~0.6–1.2 s | ~12GB |
| 13 物体主表 | 3–5 h | ~12GB |
| 8 组消融全量 | 1–2 天 | 同上 |

### 常用开关（configs/current/default.yaml）

- `detection.segmenter` —— `fastsam`（主实验，需 `pip install ultralytics`）| `sam`（ViT-H 消融）| `gt_mask`（分割上界）| `gt_bbox`（定位上界）。未知值直接 raise，不静默回退
- `matching.top_k` —— Top-K 候选模板数（默认 40=全部模板，即不裁候选；该项只决定候选数，与旧代码的 oracle 数字无关）
- `matching.template_ranking` —— `dinov2`（默认，复用定位相似度预筛，K<40 时只解码 K 个模板真省算）| `mast3r`（全解码后按 sim(m) 选 Top-K）
- `templates.n_viewpoints / n_inplane` —— 模板数（默认 8×5=40）
- `geometry.source: vggt` —— 切 VGGT 前馈重建路线（需 `pip install vggt`）；评测经 Umeyama+ICP 对齐回 CAD 系（见下）
- `renderer.backend: pyrender_cad` —— 直接光栅化 CAD 的模板库（消融 3.3.8）

§8 旧代码（MyPose）能力开关，默认值均等于新库原行为：

- `templates.template_source` —— `coord_map`（默认）| `depth_map`（额外渲染模板深度图，需重新 onboard；模板库文件名带 `_depth` 后缀不会覆盖坐标图库）
- `matching.lifting` —— `coord_map`（默认，alpha 混合坐标图查表）| `depth_backproject`（模板深度 `K_inv` 反投影 + 位姿逆变换，旧管线）
- `matching.template_prescreen` —— `dinov2`（默认）| `none`（跳过 DINOv2 预筛，全模板逐一 MASt3R 匹配 = 旧管线）。注意 `none` 与 `template_ranking: dinov2` 组合无意义（预筛被跳过，ranking 无从生效），配置期直接 raise
- `detection.segmenter: yolo` —— YOLO bbox + GT coseg mask 定位（需 `pip install ultralytics` + `detection.yolo_checkpoint`）。**这是独立消融项，不在 legacy 配置里**：旧代码虽然加载了 YOLO 检测，但主循环裁剪全程用 GT mask，YOLO 框从未参与计算（见 `configs/current/legacy_mypose.yaml` 注释）
- `detection.crop_mode` —— `context_pad`（默认，bbox 扩 20%）| `tight_square`（mask 涂黑 + 外接框 `crop_expand`=1.1 倍方形 + resize `crop_size`=512，旧管线）
- `solver.pnp_flag` —— `epnp`（默认）| `sqpnp`（旧管线）
- `solver.selection` —— `inlier`（默认，几何一致性择优）| `similarity`（旧代码的候选窗口顺序：按 MASt3R 模板相似度降序）| `weighted`
- `metrics.topk_best` —— `[]`（默认关）| 如 `[1,3,5,40]`：输出前 K 个候选按 **GT ADD** 择优的结果。**K>1 的每一档都是 oracle 上界**（用测试集真值挑答案），只有 `top1` 是端到端数字

### 旧代码能力融合（§8）

```bash
# 一份配置复现旧管线的设计选择（GT mask 定位 + 512 方形裁剪 + 深度反投影 +
# 无 DINOv2 预筛 + top_k 40 + SQPNP + 相似度候选序）。
# 注意需要深度图模板库，先重新 onboard：
python scripts/data/onboard_object.py --config configs/current/legacy_mypose.yaml
python scripts/eval/run_linemod.py --config configs/current/legacy_mypose.yaml \
    --aggregated-out outputs/legacy_aggregated.json
#   → outputs/legacy_aggregated.json：旧 aggregated_metrics 兼容格式，但内容是
#     **端到端**数字（内点择优的最终预测），只能与旧 `top1` = 49.49% 比。
#     ⚠ 不要拿它和旧 82.73% 比——后者是 GT 择优上界。
#   → outputs/legacy_aggregated_topk_best.json：top1/3/5/40 档位。与旧 82.73%
#     同口径可比的是这里的 `top40_best`（同为 GT 择优 oracle 上界）。

# 把旧代码的真实实验结果导入新库产物目录（论文表格直接从这里取数）
python scripts/experiments/import_prior_metrics.py
#   → results/prior/aggregated_metrics_all_objects40_report.json：
#     旧 Top-40 **GT 择优上界** ADD 82.73% / Proj 81.99%（13407 样本，oracle，
#     非端到端）；文件里带 protocol=prior_MyPose_oracle_top40、
#     is_oracle_upper_bound=true，以及 non_oracle_reference（同批候选的端到端
#     数字 top1 ADD 49.49% / Proj 59.22%）
#   → ..._top1_top3_top5_report.json：逐档带 is_oracle（top1=false，其余 true）
```

**旧数字的口径（务必先读）**：旧管线的最终候选是 `argmin(GT ADD)` 挑出来的
（`inference_on_LM.py:526`，`:459/:468` 传入 gt_pose），内点数只在 `:451` 做
`<6` 失败门槛、从未参与排序。所以旧 Top-40 的 **82.73% 是 GT 择优上界
（oracle，非端到端）**，`top5_best` 74.15% / `top3_best` 68.70% 同理；
同一批候选唯一的端到端数字是 `top1 = 49.49%`（Proj 59.22%）。
7 物体版 `aggregated_metrics_all_objects.json`（86.17%）经逐物体比对就是
`top5_best` 的聚合，同样是 oracle。**论文里 oracle 数字与端到端数字不得同表比较。**

`metrics.topk_best` 复现的是旧 top1/3/5 的**窗口语义**（相似度序 + 失败候选占
名额 + 同步选择），不是旧数值：模板打分函数不同（旧 `:362` 是互最近邻配对点积
**求和**，本库 `sim(m) = mean_y max_{y'} S`），另有若干口径差异逐条列在
`configs/current/legacy_mypose.yaml` 的"已知非复现项"。

逐项移植清单、旧代码 file:line 出处、对应测试名见
[`VERIFICATION.md`](VERIFICATION.md) §8。

### 数据划分协议（论文 4.1.1，防参考帧泄漏）

3DGS 参考视图与评测集必须零重叠。loader 支持可选的 PVNet 式官方划分：
放置 `data/splits/lm/<obj>_train.txt`（一行一个帧号，从 PVNet 仓库
`data/linemod` 目录获取）后，参考帧只取自 train 列表、评测在其补集（测试
划分）上进行。**无 split 文件时**参考帧从测试序列均匀抽样，`evaluate_object`
默认 `exclude_refs=True` 显式排除这些采样参考帧（`src/datasets/linemod.py`
的 `reference_frame_ids` / `eval_frames`）。

### VGGT 路线坐标系对齐（论文 3.2.2）

VGGT 重建点云在第一帧相机系、尺度相对，与 CAD 物体系差一个未知相似变换。
onboard 时：传入前景掩码只重建物体点 → 点云重心平移到原点 → 用 FPS 采样点做
Umeyama 闭式相似变换（可选 ICP 精化）求"重建系→CAD 系"变换并存入模板库。
评测时把估计位姿经该变换回 CAD 系再算指标（`src/geometry/alignment.py`）。
**CAD 仅用于评测侧对齐，不参与任何推理环节**（model-free 通行惯例）。

### 与论文的已知实现取舍

- 模板 3D 坐标图用 **alpha 混合渲染高斯中心 μ**（除以 alpha 归一），是论文
  2.3.3 argmax 主贡献高斯定义的光顺化实现，无需修改 gsplat 光栅器；
- 互最近邻 + cycle consistency（τ=5px）合并实现：严格互最近邻是 τ=0 特例，
  往返偏差 ≤ τ 的过滤天然包含互匹配对（`src/matching/correspondence.py`）；
- MASt3R 解码器是成对交叉注意力，跨帧可复用的只有 ViT 编码器 token——
  `Mast3rMatcher.prepare_templates` 预提取并缓存 40 个模板的编码器特征，
  在线每帧只编码查询一次 + 40 次解码（论文 2.6.4 复杂度分析中的主要瓶颈）；
- LoFTR 匹配器为 TODO 接口（消融 3.3.3 预留），调用时显式
  `NotImplementedError` 并附接入说明（`src/matching/alt_matchers.py`）。

## 本地测试结果

`cd exp_6dpose && .venv/bin/python -m pytest tests/ -q`（macOS arm64，
Python 3.9，torch CPU 版）：

```
........................................................................ [ 45%]
........................................................................ [ 91%]
.............                                                            [100%]
157 passed in 1.88s
```

覆盖：视角采样几何（正交性/球面半径/光轴指向/8 卦限覆盖/72° 平面内旋转）、
模板内参的整数像素索引约定与渲染器像素中心约定互换（半像素约定，见 VERIFICATION §8.8）、
尺度对齐（含投影缩放不变性）、互最近邻 + cycle consistency + 阈值 + 采样、
Top-K DINOv2 预筛（相似度降序传递 + 只解码 K 个模板 + 无效组合显式 raise）、
参考/评测数据划分（PVNet split 生效 + 无 split 时排除采样参考帧，防泄漏）、
VGGT→CAD 相似变换（Umeyama 恢复精度 / FPS / ICP 精化 / 位姿变换相机点一致性）、
分割器未知值显式 raise（禁止静默回退）、
配置闸门（`base` 链成环报链条 / 覆盖层空值盖配置段报错 / 深合并不改 base）、
模板库闸门（缺 `scale`、缺 `dino_feats`、`depth_maps` 形状不匹配一律 raise）、
RANSAC-EPnP（合成位姿 + 噪声 + 40% 外点恢复）、内点/相似度/加权择优、
ADD / ADD-S（对称物体 180° 等价位姿）/ Proj@5pix / 5cm5° 数值正确性、
BOP LineMod loader（mock 目录 + ascii/binary PLY）、10 组消融配置解析、
GPU-only 组件的 ImportError 提示；
§8 旧代码能力融合（`tests/test_legacy_mypose.py`，46 条）：深度反投影与旧代码
逐行转写逐点对拍（含无效深度分支）+ 越界显式 raise + 前向/反向闭环 +
渲染深度↔反投影半像素约定闭环、旧式方形裁剪逐像素对拍 + 回映射公式恒等、
全模板匹配开关、SQPNP 求解、topK 窗口语义（相似度序 + 失败候选占名额）与同步选择、
旧 aggregated JSON 口径回归（用真实旧结果文件重算 overall 一致 + 全精度再平均）、
真实结果导入转换与 oracle 标注字段、legacy_mypose.yaml 逐项对应与默认值不变性。

## 当前结果（2026-08-02，13 物体 × 120 帧均匀采样，含 refine）

配置：黑背景 + CAD 深度监督训练（depth 0.6）+ 固定视图 + 逆深度锚点。
完整表格与逐版本对比见 `docs/RESEARCH_LOG.md` §5/§8。

| 物体 | ADD | Proj | 5cm5° | 训练背景 |
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

背景选择规则：浅色/无纹理物体（eggbox/lamp/can/glue 等）用黑背景（对比度强），
深色物体（driller/cam）用白背景；白背景 + 深色物体与黑背景 + 浅色物体等价，
深度监督（depth 0.6）兜底几何。

旧代码 MyPose 端到端 top1 参照：ADD 49.49% / Proj 59.22%（13407 帧全量）。
