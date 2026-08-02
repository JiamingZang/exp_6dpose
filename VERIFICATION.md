# VERIFICATION.md —— 外部 API 调用逐项核对记录

对照 `third_party/` 下真实仓库源码逐项核对本代码库的外部 API 调用与协议假设。
格式：**我们的调用 → 真仓库出处 file:line → 结论（一致 / 已修正）**。
所有路径相对 `third_party/`；本库路径相对仓库根。核对基准：

| 仓库 | 版本 |
|---|---|
| gsplat | v1.5.3（`gsplat/gsplat/version.py:16`，commit 2b902ff） |
| mast3r / dust3r | naver 官方 main 浅克隆 |
| GSPose | 官方 main 浅克隆 |
| segment-anything | 官方 main 浅克隆 |
| dinov2 | 官方 main 浅克隆 |

---

## 1. gsplat（`src/gaussian/gs_trainer.py`、`src/gaussian/template_renderer.py`）

### 1.1 `rasterization()` 调用签名
- 我们的调用：`gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree=..., packed=False)`（`src/gaussian/gs_trainer.py` `render()`）
- 出处：`gsplat/gsplat/rendering.py:234-290`（形参名与顺序）、`:481-491`（返回 `render_colors [..,C,H,W,X]` / `render_alphas [..,C,H,W,1]` / `meta` dict）
- **一致**。`viewmats` 形如 `[C,4,4]`（我们 `[None]` 扩出 C=1），`Ks [C,3,3]`，`scales` 传 `exp(log_scales)`、`opacities` 传 `sigmoid(logit)`——均为激活后值，与官方 `simple_trainer.py` 的 `rasterize_splats` 用法相同。返回 `renders[0]/alphas[0]` 取 (H,W,C)/(H,W,1) 正确。

### 1.2 SH 系数 vs. N-D 特征两种 colors 语义（3D 坐标图渲染的关键）
- 我们的用法：RGB 渲染传 `colors=cat([sh0, shN], dim=1)`（`[N,K,3]`）+ `sh_degree=3`；坐标图渲染传 `colors_override=means`（`[N,3]` 逐高斯特征）+ `sh_degree=None`。
- 出处：`gsplat/gsplat/rendering.py:313-321`（sh_degree=None 时 colors 是 `[..., N, D]` 任意 D 维特征，D>32 时按 `channel_chunk` 分块 `rendering.py:439-441`；sh_degree 给定时 colors 是 `[N,K,D]` SH 系数，要求 `(sh_degree+1)^2 ≤ K`）
- **一致**：以高斯中心 μ 当 3 通道特征做 alpha 混合渲染 3D 坐标图，正是 N-D 特征路径，无需分批也无需改动光栅器。除以 alpha 的归一（`template_renderer.py`）与背景合成 `rgb + (1-alpha)*bg` 为标准 alpha 合成，不涉及 gsplat 内部假设。

### 1.3 `DefaultStrategy` 构造 / 状态协议
- 我们的调用：`DefaultStrategy(grow_grad2d=…, refine_start_iter=…, refine_stop_iter=…, refine_every=…, verbose=False)` + `check_sanity` + `initialize_state`。
- 出处：`gsplat/gsplat/strategy/default.py:99-114`（全部字段名）、`:132-156`（check_sanity 要求 params/optimizers 同键、每 optimizer 恰一个 param_group、必含 means/scales/quats/opacities）、`:116-130`（`initialize_state(scene_scale=1.0)`）
- **一致 + 已修正**：字段名全部存在；但 `initialize_state()` 原先未传 `scene_scale`，导致 grow/prune 的 3D 尺度阈值（`default.py:298-309,351-356`：`grow_scale3d * scene_scale`）用默认 1.0 与物体的 mm 级坐标错位（所有高斯恒判"小"，只 clone 不 split）。已修正为传入物体包围半径 ×1.1（官方以相机分布半径为场景尺度，`examples/simple_trainer.py:458`、`:511-512`）。

### 1.4 训练循环顺序（本次最重要的修正）
- 我们原先的顺序：`backward → step_post_backward → optimizer.step/zero_grad`。
- 官方顺序：`loss.backward()`（`examples/simple_trainer.py:998`）→ `optimizer.step(); optimizer.zero_grad(set_to_none=True)`（`:1131-1138`）→ `strategy.step_post_backward(...)`（`:1156-1165`）。
- **已修正**（`src/gaussian/gs_trainer.py` `train()`）：`step_post_backward` 内部会 clone/split/prune 并把优化器 param_groups 里的参数整体替换（`gsplat/gsplat/strategy/ops.py:96-137` `_update_param_with_optimizer`），若先于 `opt.step()` 执行，本次梯度会 step 到密度控制后的新参数上，优化器状态错位。`step_pre_backward` 在 backward 前调用（`strategy/default.py:158-170`，`retain_grad` 到 `meta["means2d"]`）保持不变，与官方 `simple_trainer.py:938-944` 相同。
- `packed=False` 同时传给 `rasterization` 与 `step_post_backward`，二者必须一致（`default.py:172-180,263-272` 按 packed 解析 meta），**一致**。meta 必含键 `means2d/radii/width/height/n_cameras/gaussian_ids`（`default.py:233-241` 断言）由 `rendering.py:660-676` 全部提供。

### 1.5 初始化与学习率
- KNN 尺度初始化：官方 `dist_avg = sqrt(mean(knn_d^2))`（`examples/simple_trainer.py:321-323`）；我们原先用普通平均距离，**已修正**为 RMS（`_knn_mean_dist`）。
- means lr 按场景尺度缩放：官方 `means_lr * scene_scale`（`simple_trainer.py:336`）；我们原先不缩放，**已修正**。
- lr 默认组（means 1.6e-4 / scales 5e-3 / opacities 5e-2 / quats 1e-3 / sh0 2.5e-3 / shN 2.5e-3/20）与 `simple_trainer.py:295-300` 逐项相同，Adam `eps=1e-15`（`simple_trainer.py:371`，BS=1）——**一致**。
- 不透明度初始化 0.1 过 logit（`simple_trainer.py:293,332`）、sh0 用 `(rgb-0.5)/C0`（`simple_trainer.py:344-345` `rgb_to_sh`）——**一致**。
- 四元数：官方随机（`simple_trainer.py:331`）、我们取单位四元数（确定性）；wxyz 约定且无需归一化（`rendering.py:400`）——**一致（有意取舍，已注释）**。
- 每优化器单参数的 `torch.optim.Adam([p], …)` 结构满足 ops 的替换协议（`ops.py:129-137` 逐 param_group 重挂 `[new_param]`）——**一致**。

### 1.6 版本
- gsplat 1.5.3 要求 `torch>=2.7`（`gsplat/setup.py:129-143`）。requirements.txt 已 pin `gsplat==1.5.3`、`torch>=2.7`（PyPI 已发布 1.5.3，核对过 releases）。

---

## 2. GSPose 协议对照（`src/datasets/linemod.py`、`src/metrics/`、`configs/default.yaml`、`src/detection/`）

### 2.1 LineMod 数据组织
- 物体→ID 映射（13 物体，剔除 bowl=3/cup=7）：`GSPose/dataset/inference_datasets.py:345-352` `name2classID`，与 `src/datasets/linemod.py` `LINEMOD_OBJECT_IDS` 逐项相同——**一致**。
- BOP 目录约定 `test/{obj_id:06d}/` + `scene_gt.json`（`cam_R_m2c` 9 元 + `cam_t_m2c` mm）+ `scene_gt_info.json` + `models_info.json`：`inference_datasets.py:360,373-380,407-419`——**一致**（我们全程 mm，GSPose 乘 `to_meter_scale=1e-3` 转米，`:364,382`；ADD@0.1d 是比值，两种单位约定等价）。
- diameter 来源：`models_info.json['<obj_id>']['diameter']`，单位 mm（`inference_datasets.py:382`）——**一致**。
- 对称物体：GSPose 由 `symmetries_*` 键推断（`inference_datasets.py:384-388`）；13 物体协议下命中 eggbox(10)/glue(11)，与我们硬编码的 `SYMMETRIC_OBJECTS` 等价——**一致**。

### 2.2 评测协议
- ADD：`mean ||R_gt p + t_gt - (R̂ p + t̂)||`，阈值 `0.1 × diameter`（`GSPose/misc_utils/metric_utils.py:24-47`）——**一致**（`src/metrics/pose_metrics.py` `add_error` + `evaluate_pose` 的 `add < 0.1d`）。
- ADD-S：GSPose `cKDTree(model_pred).query(model_target)`（`metric_utils.py:36-39`），我们 `cKDTree(pred).query(gt)`——查询方向相同，**一致**。
- Proj@5pix：`metric_utils.py:50-77` 与 `proj_error_px` 同式——**一致**。
- 失败帧计入分母：与 BOP/GSPose 汇总口径一致（`src/pipeline.py` `evaluate_object`）。

### 2.3 3DGS 物体表示构建超参
- GSPose 沿用 3DGS 官方 `OptimizationParams`（`GSPose/gaussian_object/arguments.py:81-96`）：position_lr_init 1.6e-4、scaling_lr 5e-3、opacity_lr 5e-2、rotation_lr 1e-3、feature_lr 2.5e-3、lambda_dssim 0.2、densify_from 500、densification_interval 100、densify_grad_threshold 2e-4、opacity_reset_interval 3000。
- 损失形式 `(1-λ)·L1 + λ·(1-SSIM)`、每迭代随机抽一个训练视角：`GSPose/gaussian_object/build_3DGaussianObject.py:93,107-114`——与我们 `gs_trainer.train()` 相同。
- `configs/default.yaml` gaussian 段已注明以上出处；差异项（iterations 7000 vs 30000、densify_end 5000 vs 15000）是论文有意取舍，已在 config 注释标注。gsplat `DefaultStrategy` 默认 `reset_every=3000` 与 GSPose `opacity_reset_interval=3000` 相同。

### 2.4 DINOv2 检测/模板检索流程
- GSPose 用 `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`（`GSPose/model/network.py:42`），检索用 `x_norm_clstoken`（`network.py:197-199`），其上另有自训练的 coseg/pose 头。
- 我们（论文 2.4）用 `dinov2_vitl14` 原生 CLS token 余弦检索、不引入训练头——加载方式（torch.hub）与 CLS 提取口径与 GSPose 相同，模型规格更大且零训练，属论文既定设计——**一致（不同但合理，已在 `src/detection/localize.py` 注明对照）**。

---

## 3. MASt3R（`src/matching/mast3r_wrapper.py`）

### 3.1 模型加载
- `AsymmetricMASt3R.from_pretrained(ckpt)`：本地 `.pth` 走 `load_model()`（`mast3r/mast3r/model.py:46-51 → 20-36`），并强制 `landscape_only=False`（`model.py:25-29`）→ 任意宽高比（竖版查询裁剪）可推理——**一致**。
- **已修正（协议错误）**：权重名原写作 `MASt3R_ViTLarge_BaseDecoder_512_dpt_metric`，官方唯一 ViT-L 权重是 **`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`**（`mast3r/README.md:134-142`，下载 URL 同）。已修正 4 处：`configs/default.yaml`（两个字段）、`setup_gpu.sh`（wget URL）、`README.md`、`mast3r_wrapper.py`（docstring 与部署提示）。

### 3.2 编码器 / 解码器 / 匹配头调用
- `_encode_image(img, true_shape)` 返回 `(x, pos, None)`：`dust3r/dust3r/model.py:128-140`——我们 `feat, pos, _ = model._encode_image(...)`，`true_shape` 为 `(B,2)` 的 (H,W) 整型张量（`dust3r/utils/image.py:122` 用 np.int32）——**一致**。
- `_decoder(f1, pos1, f2, pos2)` 返回逐层 token 元组对：`dust3r/dust3r/model.py:172-191`——我们成对堆叠批量调用，模板侧 256×256 token 形状一致可 cat，查询侧 repeat——**一致**（与官方 `forward` 的 `dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)`，`model.py:204` 等价批处理）。
- `_downstream_head(head_num, [tok.float() for tok in decN], true_shape)`：`dust3r/dust3r/model.py:193-197`，fp32/autocast-off 包裹同官方 `dust3r/model.py:206-209`、`mast3r/mast3r/model.py:208`——**一致**。
- 返回 dict 键：`pts3d` / `conf` / `desc` / `desc_conf`（`mast3r/mast3r/catmlp_dpt_head.py:27-40` postprocess）；`desc` 已由 `reg_desc` 单位范数归一（`catmlp_dpt_head.py:19-24,36`），点积即余弦——我们只取 `res["desc"]`，`sim_threshold=0.3` 作用在单位范数点积上——**一致**。
- 局部特征维度 d=24：`catmlp_dpt_head.py:152` `mlp_odim=24`——**一致**（docstring 已标注）。

### 3.3 输入预处理
- 归一化 `(x/255 - 0.5)/0.5`：官方 `ImgNorm = Normalize(mean=0.5, std=0.5)`（`dust3r/dust3r/utils/image.py:23`）——**一致**。
- 尺寸：官方 `load_images` 长边 512 + 中心裁剪到 16 的倍数；我们把两边各自 round 到 16 的倍数（轻微改变纵横比、不裁剪，坐标经逐轴比例映射回裁剪区，自洽）。模板 256×256 天然合规。**有意取舍（不丢像素），已在 docstring 说明**；模型侧无障碍（patch 16 整除 + landscape_only=False）。

### 3.4 互最近邻对照
- 官方 `fast_reciprocal_NNs` / `bruteforce_reciprocal_nns`：双向 argmax（dot 距离取 max，`mast3r/mast3r/fast_nn.py:16-71,111-140`），子采样迭代收敛到不动点。
- 我们：GPU 上双向 `argmax`（同 `bruteforce` 的 dot 分支）→ 纯 numpy `cycle_consistency_filter(τ=5px)`（`src/matching/correspondence.py`，τ=0 即严格互最近邻，官方不动点匹配是其特例）→ 阈值 → 加权采样。**一致（τ 放宽为论文 2.5.2 设计，经本地单测覆盖）**。

---

## 4. SAM + DINOv2（`src/detection/localize.py`、`src/matching/alt_matchers.py`）

### 4.1 segment-anything
- `sam_model_registry["vit_h"](checkpoint=path)`：`segment-anything/segment_anything/build_sam.py:14,47-52`——**一致**。
- `SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88)`：形参名与默认值 `automatic_mask_generator.py:36-41`（默认 32 / 0.88 即 ViT-H 推荐值）——**一致**。
- `generate(image)` 返回 dict 列表，键 `"segmentation"`（HxW bool，output_mode='binary_mask' 默认）、`"bbox"`（**XYWH**，`box_xyxy_to_xywh` 转换）、`"area"`、`"predicted_iou"`、`"stability_score"`：`automatic_mask_generator.py:137,184-192`——**一致**（我们按 XYWH 解包 `x, y, bw, bh = m["bbox"]`）。

### 4.2 dinov2
- hub 入口 `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")`：`dinov2/hubconf.py:7`——**一致**。
- CLS 提取：backbone 的 `forward` = `head(forward_features(...)["x_norm_clstoken"])`，`head` 为 `nn.Identity()`（`dinov2/dinov2/models/vision_transformer.py:169,348-353`），即 `self.dino(x)` 直接得 CLS token（ViT-L 1024 维）——**一致**。
- patch token（消融匹配器）：`forward_features(x)["x_norm_patchtokens"]`（`vision_transformer.py:261-271`），patch 14、输入边长须 14 的倍数——`alt_matchers.py` 的 `_dense_desc` 按 14 对齐——**一致**。
- ImageNet 归一化常数 (0.485,0.456,0.406)/(0.229,0.224,0.225)：`dinov2/dinov2/data/transforms.py:42-44`——**一致**。

---

## 5. 修正汇总与验证

| # | 文件 | 修正 |
|---|---|---|
| 1 | `src/gaussian/gs_trainer.py` | 训练循环顺序改为 backward → step/zero_grad → step_post_backward（simple_trainer.py:998,1131-1165） |
| 2 | `src/gaussian/gs_trainer.py` | `initialize_state(scene_scale=…)` + means lr × scene_scale（simple_trainer.py:336,458,511-512） |
| 3 | `src/gaussian/gs_trainer.py` | KNN 尺度初始化改 RMS（simple_trainer.py:321-323） |
| 4 | `configs/default.yaml` / `setup_gpu.sh` / `README.md` / `mast3r_wrapper.py` | MASt3R 权重名改为 `*_catmlpdpt_metric`（mast3r/README.md:134-142） |
| 5 | `requirements.txt` | pin `gsplat==1.5.3`（version.py:16）、`torch>=2.7`（gsplat/setup.py:141）；依赖注明出处 |
| 6 | `configs/default.yaml` / `src/datasets/linemod.py` / `src/metrics/pose_metrics.py` / `src/detection/localize.py` | 超参与协议注释补 GSPose/官方仓库 file:line 出处 |

验证：`cd exp_6dpose && .venv/bin/python -m pytest tests/ -q` → **79 passed**（本次修正全部位于 GPU 路径与注释/配置，未改变任何被单测覆盖的纯逻辑语义；配置断言仍满足）。

---

## 6. 修复记录（P0 五项）

| # | 问题 | 改动文件 | 验证方式 |
|---|---|---|---|
| 1 | 参考/评测数据划分泄漏：3DGS 参考帧从测试序列抽取又计入评测 | `src/datasets/linemod.py`（`train_split_ids`/`reference_frame_ids`/`eval_frames`，支持 `data/splits/lm/<obj>_train.txt`）、`src/pipeline.py`（`_build_reference_views` 用参考帧集合、`evaluate_object(exclude_refs=True)`）、`configs/default.yaml`（`dataset.splits_dir`）、`scripts/download_data.sh`（PVNet split 说明） | `tests/test_linemod_loader.py`：无 split 排除采样参考帧、有 split 只取 train 列表且评测测试划分、splits_dir 覆盖 |
| 2 | Top-K 语义错位：40 模板全解码后才选 K，DINOv2 相似度被丢弃 | `configs/default.yaml`（`matching.template_ranking=dinov2`、`top_k=40`）、`src/detection/localize.py`（`template_similarity_order` + `Localization.template_order`）、`src/matching/mast3r_wrapper.py`（`decode_template_indices` + `match(prefilter_order=…)` 只解码 K 个）、`src/matching/alt_matchers.py`、`src/pipeline.py`（传排序）、`configs/ablations/01_topk.yaml`（`[1,5,10,20,40]`） | `tests/test_detection_and_config.py`：排序降序、K 个解码、越界过滤、top_k==40 断言 |
| 3 | FastSAM 静默回退：`segmenter: fastsam` 悄悄落回 SAM | `src/detection/localize.py`（`FastSamSegmenter`、`SamDinoLocalizer(segmenter=…)`、未知值 raise）、`src/pipeline.py`（segmenter 显式分派，未知 raise）、`configs/default.yaml`（`segmenter: fastsam` + fastsam 参数）、`requirements.txt`（ultralytics 可选依赖） | `tests/test_alignment.py`：SamDinoLocalizer / PoseEstimator 对未知 segmenter raise ValueError |
| 4 | VGGT 路线坐标系未对齐：重建系 vs CAD 系差未知相似变换、无掩码、评测无意义 | `src/geometry/alignment.py`（`umeyama_alignment`/`farthest_point_sample`/`icp_refine`/`transform_pose_by_similarity`）、`src/pipeline.py`（VGGT 传前景掩码、点云重心归零、onboard 存 Umeyama+ICP 对齐、estimate 位姿变换回 CAD、`TemplateBank` 载入对齐）、`configs/default.yaml`（`geometry.vggt.n_align_points/icp_iterations`） | `tests/test_alignment.py`：合成相似变换（含噪声/反射）Umeyama 恢复、FPS 唯一性、ICP 精化、位姿变换相机点一致性 |
| 5 | 消融配置与论文第四章（9 组）不对齐 | `configs/ablations/09_ransac_eps.yaml`（ε∈[3,5,8,10]）、`configs/ablations/10_segmenter.yaml`（fastsam/sam/gt_mask，含 `GtMaskLocalizer` 路径）、`configs/ablations/06_scale_align.yaml`（改 VGGT-only + 去 requires_reonboard + 数学恒等注释）、`configs/ablations/04_localization.yaml`（fastsam/gt_bbox） | `tests/test_detection_and_config.py`：10 组 yaml 解析、09/10 sweep 值、06 不重训 |

验证（含新增用例）：`cd exp_6dpose && .venv/bin/python -m pytest tests/ -q` → **103 passed**（79 原有 + 24 新增，无删减）。

---

## 7. 第二轮复审修复（P1 三项 + P2 三项）

第二轮独立复查发现 5 项 P0 修复真到位（强断言测试全覆盖），但抓出 6 项设计/质量遗留问题，本轮全部修复。

| # | 级别 | 问题 | 改动文件 | 验证方式 |
|---|---|---|---|---|
| 1 | P1 | VGGT 路径的参考帧从全部帧均匀抽，绕开 `train_split_ids` 保护——split 模式下 VGGT 会吃到测试帧 | `src/datasets/linemod.py`（`vggt_reference_frame_ids` 复用 split 过滤）、`src/pipeline.py:103,453`（VGGT 用它，eval `extra_exclude` 一并扣除） | `tests/test_linemod_loader.py`：有 split 时 VGGT 帧全在 train 池、无 split 时被 eval 排除 |
| 2 | P1 | `PoseEstimator.estimate` 的 resize/crop/尺度反变换链条无单测 | 抽出纯函数 `back_to_original_pixels(pix_q, sxy, crop_box)` in `src/matching/correspondence.py:171`；`src/pipeline.py:380` 消费 | 新增单测：合成 GT 位姿→手工投影+resize+crop→反变换→PnP 恢复 GT（rtol<1%） |
| 3 | P1 | VGGT 前景掩码缺失时静默 fallback 到整图重建，背景噪点拉偏 Umeyama | `src/pipeline.py:181-206` `_load_reference_masks` 缺 mask 显式 raise ValueError，报明哪个物体哪一帧 | 新增单测：缺 mask 时 raise 且消息包含 obj/frame_id |
| 4 | P2 | `test_sam_localizer_import_hint` 依赖本地未装 segment_anything 才通过 | `tests/test_gpu_guards.py`：monkeypatch `sys.modules["segment_anything"]=None` 稳定触发；拆出 `test_sam_localizer_unknown_segmenter_raises` 单独测未知值 raise | 两条测试独立通过，装了 SAM 的机器也不会误判 |
| 5 | P2 | CPU 用户直接跑 run_linemod 会在 FastSAM/SAM 初始化时抛错 | `scripts/run_linemod.py:71-77`：runtime.device=cpu 且 segmenter∈{fastsam,sam} 时早期显式提示切 gt_bbox/gt_mask | 手工确认打印+提前退出 |
| 6 | P2 | ICP 无提前停止判据，硬跑 20 轮浪费算力 | `src/geometry/alignment.py:92,100,132`：加 `eps=1e-6` 提前停止 + verbose 模式打印每轮 residual | 新增单测：完美对应 1 轮内收敛提前退出 |

验证：`cd exp_6dpose && .venv/bin/python -m pytest tests/ -q` → **110 passed**（103 + 7 新增，无删减）。

---

## 8. 旧代码能力融合（MyPose → exp_6dpose）

> ### ⚠ 先读：旧代码全部 topK 数字都是 GT 择优 oracle 上界
>
> 旧管线的最终候选择优是 `min_add_idx = np.argmin(top5_adds)`
> （`inference_on_LM.py:526`），而 `top5_adds` 的每一项都是候选位姿与
> **GT 位姿**算出的 ADD 误差（`:459/:468` 传入 `gt_pose`）——即**用测试集真值
> 挑答案**。旧代码里**不存在按内点数择优**的逻辑（`inliers` 仅在 `:451` 做
> `<6` 的失败门槛，从未参与排序）。
>
> 因此：**Top-40 的 82.73%、`top5_best` 74.15%、`top3_best` 68.70% 全部是
> GT 择优 oracle 上界**，只能作"模板检索若完美时的性能潜力"分析，
> **不得与端到端方法同表比较**。同一批候选唯一的非 oracle（端到端）数字是
> **`top1` = ADD 49.49% / Proj 59.22%**。7 物体版
> `aggregated_metrics_all_objects.json`（86.17%）经逐物体数字比对就是
> `top5_best` 的聚合，同样是 oracle。
>
> 新库主路线的择优判据是内点数（`src/solver/selection.py`，非 oracle），
> 与旧数字语义不同；报告与论文里必须分开标注。转换产物已把这个性质落成
> JSON 数据字段（`protocol` / `is_oracle_upper_bound` / `tiers[*].is_oracle` /
> `non_oracle_reference`），不是只写在注释里。

把用户真实旧代码 `_prior_code/MyPose/` 里已验证的能力移植进本库，使新库成为
旧代码能力的**超集**。原则：**每项做成可配置开关，默认值等于新库原行为**，
`configs/legacy_mypose.yaml` 一份配置即可切回旧管线的**设计选择**
（不复现其数值，逐条见 §8.6 与该 yaml 的"已知非复现项"）。

核对基准（只读，未改动）：

| 旧文件 | 内容 |
|---|---|
| `_prior_code/MyPose/inference_on_LM.py`（776 行） | 真实推理管线（本节全部 file:line 出处） |
| `_prior_code/MyPose/config/inference_cfg.py` | `USE_YOLO_BBOX=True`（:32）等开关 |
| `_prior_code/MyPose/aggregated_metrics_all_objects40.json` | Top-40 **GT 择优上界（oracle）**：ADD 82.7254% / Proj 81.9945% / 13407 样本 |
| `_prior_code/MyPose/aggregated_metrics_top1_top3_top5.json` | top1 49.49/59.22（**唯一的端到端数字**）、top3_best 68.70/70.78、top5_best 74.15/74.46（后两档为 oracle 上界） |

### 8.1 深度图模板 + 深度反投影 2D-3D 提升

| 项 | 内容 |
|---|---|
| 旧代码出处 | 模板深度加载 `inference_on_LM.py:233-246`（512DepthCube 的 rgb+depth+poses.json）；`K_inv` `:256`；位姿逆变换 `:261-262`；逐点反投影 `:414-424` |
| 新库落点 | `src/matching/depth_lifting.py`（新增，`backproject_depth_to_model`）；模板深度渲染 `src/gaussian/template_renderer.py`（3DGS 分支 alpha 混合相机系 z / pyrender 分支直接取深度缓冲）；`TemplateBank.depth_maps` 与 `PoseEstimator.estimate` 的 lifting 分支 `src/pipeline.py` |
| 配置开关 | `templates.template_source: coord_map \| depth_map`、`matching.lifting: coord_map \| depth_backproject`、`matching.depth_max`（旧代码 `:419` 的 5.0 粗差上限） |
| 数学核对 | 模板位姿 `T_w2c=[R\|t]` 满足 `p_cam = R X + t` ⇒ `X = R^T(p_cam - t) = R^T p_cam - R^T t`，与旧代码 `R_cam2model = R.T`、`t_cam2model = -R_cam2model @ t` **逐项一致**；`K_inv` 方向为 `p_cam = d·K_inv·[u,v,1]^T`，同旧代码 `:423`。像素齐次向量用整数坐标（不加 0.5 中心偏移），同旧代码。**单位差异（有意取舍）**：旧代码深度为米、位姿 t 为 mm 故有 `/1000`（`:262`）；本库深度与位姿同在尺度对齐后的物体系单位，无需换算 |
| 测试 | `tests/test_legacy_mypose.py::test_backproject_matches_prior_reference_pointwise`（与旧代码逐行转写的参考实现逐点对拍 <1e-9）、`::test_backproject_forward_inverse_closure`（前向 z + 再投影闭环，核死 K_inv 方向与逆变换顺序）、`::test_backproject_invalid_depth_filtered`、`::test_backproject_lift_then_pnp_recovers_query_gt`、`::test_rendered_depth_backprojects_to_gaussian_centers`（假 CPU trainer 渲染深度 → 反投影还原高斯中心，并与坐标图路线互校）、`::test_template_bank_loads_depth_maps` |

模板库文件名带 `_depth` 后缀（`template_bank_path`），**避免深度库与坐标图库
互相静默覆盖**；`lifting=depth_backproject` 但模板库缺 `depth_maps` 时构造期
显式 raise 并给出"改 `template_source` 后重新 onboard"的修法
（`::test_estimator_raises_when_depth_lifting_without_depth_maps`）。

### 8.2 旧式方形裁剪 / YOLO bbox 定位（独立消融项）

| 项 | 内容 |
|---|---|
| 旧代码出处 | `inference_on_LM.py:227-229`（`use_gt_mask=True, load_yolo_det=CFG.USE_YOLO_BBOX`）+ `config/inference_cfg.py:32`；裁剪 `:286-311`（掩码外接框 → 中心 → `side = max(w,h)*1.1` → `max(0, c-half)` → 右/下 0 填充 → mask 涂黑 → resize 512 INTER_LINEAR）；坐标回映射 `:412,426-429` |
| ⚠ YOLO 的实际作用 | 旧代码只把 `load_yolo_det` 传给 dataset loader，**主循环 `:281-311` 的裁剪全程用 GT coseg mask 求外接框，YOLO bbox 从未参与任何计算**。因此 `configs/legacy_mypose.yaml` 用 `segmenter: gt_mask` 忠实复现这条数据流；`segmenter: yolo` 保留为**独立消融项**（想单独验证 YOLO 检测框对定位的影响时用），详见 §8.6 表格 |
| 新库落点 | `src/detection/localize.py`：`YoloBboxLocalizer`（ultralytics 可选依赖，缺库抛带安装指引的 ImportError）、纯函数 `legacy_square_crop` / `select_best_yolo_box`；`src/pipeline.py` 的 `crop_mode` 分支与复合缩放反变换 |
| 配置开关 | `detection.segmenter: yolo`、`detection.yolo_checkpoint/yolo_conf`、`detection.crop_mode: context_pad \| tight_square`、`detection.crop_size`（512 可配）、`detection.crop_expand`（1.1） |
| 反变换核对 | `legacy_square_crop` 返回 `sxy=(512/side, 512/side)` 与 `crop_box=(left,top,…)`，喂给既有纯函数 `back_to_original_pixels` 得 `pix/s + left`，与旧代码 `x_orig = x·(side/512) + left` **恒等**；`tight_square` 时总缩放取 `sx·s_leg`（匹配 resize × 方形 resize 的复合） |
| 测试 | `::test_legacy_crop_matches_prior_reference`（与旧代码逐行转写逐像素对拍）、`::test_legacy_crop_blackens_background_and_pads_at_border`（涂黑 + 越界 0 填充）、`::test_legacy_crop_backmap_matches_prior_formula`（回映射与旧公式恒等）、`::test_legacy_crop_empty_mask_returns_none`、`::test_select_best_yolo_box`、`::test_yolo_localizer_import_hint`、`::test_estimator_raises_on_unknown_crop_mode_and_lifting` |

### 8.3 全模板匹配模式（跳过 DINOv2 预筛）

| 项 | 内容 |
|---|---|
| 旧代码出处 | `inference_on_LM.py:321-385`：对全部 40 个模板逐一 MASt3R `inference` + `fast_reciprocal_NNs(subsample=8, dist='dot')`，无 DINOv2 预筛；`:366` 保留全部 40 个候选 |
| 新库落点 | `src/matching/mast3r_wrapper.py::resolve_prefilter_order`（纯函数，决定是否把定位排序传给 `match`）；`src/pipeline.py` 构造期校验取值 + estimate 内消费 |
| 配置开关 | `matching.template_prescreen: dinov2 \| none`（`none` = 旧行为）。未知取值显式 raise；**`none` + `template_ranking: dinov2` 这个组合也 raise**（独立验证发现：预筛被跳过后 Top-K 只能按 MASt3R sim(m) 选，`template_ranking` 完全失效，静默生效的假象会让消融结论归错因） |
| 测试 | `::test_resolve_prefilter_order_none_forces_full_matching`、`::test_resolve_prefilter_order_none_with_dinov2_ranking_raises`、`::test_resolve_prefilter_order_dinov2_passthrough_and_fallback`、`::test_resolve_prefilter_order_unknown_raises` |

顺带把旧代码的 PnP 参数做成开关：`solver.pnp_flag: epnp \| sqpnp`
（旧代码 `:448` `flags=cv2.SOLVEPNP_SQPNP`，`_pnp_flag` 在 OpenCV 缺该 flag
时报清晰错误），测试 `::test_ransac_pnp_sqpnp_recovers_synthetic_pose`、
`::test_ransac_pnp_unknown_flag_raises`。

### 8.4 top-K best 评估 + aggregated 兼容 JSON

| 项 | 内容 |
|---|---|
| 旧代码出处 | 同步选择 `inference_on_LM.py:524-533`（取 ADD 最小候选，投影误差用**同一候选**的值，注释明写"而不是独立最小值"）；per-object 聚合 `:687-721`（平均只统计有限值 `:706-710`）；overall `:738-768`（成功率=总成功/总样本；`overall_avg_*` 是**各物体平均的再平均** `:743-744`） |
| 新库落点 | `src/metrics/legacy_format.py`（新增：`topk_best_pick` / `topk_key` / `object_topk_metrics` / `aggregate_topk_all_objects` / `per_object_from_frames` / `aggregate_all_objects` / `prior_to_report` / `prior_topk_to_report`）；`src/solver/selection.py::rank_candidates`（抽出共享排序，`select_best_candidate` 变成取第一）；`EstimateResult.candidates` + `estimate(return_candidates=True)`；`evaluate_object` 的 `topk_best` 聚合；`scripts/run_linemod.py --aggregated-out` |
| 配置开关 | `metrics.topk_best: []`（默认关；如 `[1,3,5,40]` 开启）。**K>1 的每一档都是 GT 择优 oracle 上界**，只有 `top1` 是端到端数字 |
| 窗口语义（独立验证修复） | ① 候选窗口顺序须按**模板相似度降序**（旧代码 `:375`），故 `configs/legacy_mypose.yaml` 显式设 `solver.selection: similarity`——沿用新库默认 `inlier` 时 "top1" 会变成"内点最优候选"= 新库端到端预测本身，不是旧代码的"检索排名第一"；② 求解失败的候选须**占住 topK 名额**（旧代码 `:516-520` append `inf` + dummy 位姿），故 `rank_candidates(keep_failed=True)` 保留失败项、`evaluate_object` 对其记 `add=inf/proj=inf`——先剔除会让 top3/top5 系统性偏乐观。`select_best_candidate` 行为不变（主路线仍只从成功候选里选） |
| 口径回归 | `::test_aggregate_overall_matches_prior_real_file`：直接读真实旧文件的 `per_object_metrics`，用本库 `aggregate_all_objects` 重算 overall，与旧文件 overall **逐字段 rel=1e-9 一致**（证明同口径，不是近似复刻） |
| 测试 | `::test_topk_best_pick_sync_selection`（同步选择语义：proj 取 ADD 最小候选的值，不是独立最小）、`::test_topk_key_naming_matches_prior_json`、`::test_topk_from_key_inverts_topk_key`、`::test_object_topk_metrics_counts_and_schema`、`::test_per_object_and_aggregate_schema`（schema 与旧 JSON 完全一致）、`::test_aggregate_topk_all_objects_schema`、`::test_aggregate_topk_averages_unrounded_per_object`、`::test_rank_candidates_order_and_best_consistency`、`::test_rank_candidates_keep_failed_occupies_topk_slots`、`::test_rank_candidates_default_still_drops_failed` |

### 8.5 真实结果导入

`scripts/import_prior_metrics.py`（新增）读 `_prior_code/MyPose/aggregated_metrics_*.json`
→ 新库评估报告格式 → `results/prior/<原名>_report.json`，论文表格可直接从新库
产物生成。两种旧 schema 按 `overall_metrics` 是否含 `top1` 键自动分流。
旧管线未测 5cm5°，报告里该字段置 `null` 而非编数。

**oracle 性质进数据字段**（独立验证修复：原先只写在模块 docstring 里，JSON
产物本身看不出来，抄进论文就没人知道是上界）：

| 字段 | 值 |
|---|---|
| `protocol` | `prior_MyPose_oracle_top40` / `prior_MyPose_oracle_topk` |
| `selection` | `oracle_gt_add`（择优判据 = GT ADD 最小） |
| `is_oracle_upper_bound` | `true` |
| `tiers[*].is_oracle` | `top1` → `false`（端到端），K>1 → `true`（上界） |
| `non_oracle_reference` | `{top1_add_01d, top1_proj_5px}`，**从 top1/3/5 那份 JSON 的 `overall_metrics.top1` 读，不硬编码**；输入不含 top1 时省略该字段 |

实跑验证（本机真实文件）：

```
[import] aggregated_metrics_all_objects40.json → results/prior/aggregated_metrics_all_objects40_report.json
         overall[oracle 上界] ADD 82.73% / Proj 81.99% (13407 样本)
         非 oracle 参照（同批候选 top1，端到端）：ADD 49.49% / Proj 59.22%
[import] aggregated_metrics_top1_top3_top5.json → ..._report.json
         top1[端到端]: ADD 49.50%/Proj 59.12%, top3_best[oracle]: ADD 68.73%/Proj 70.67%, top5_best[oracle]: ADD 74.20%/Proj 74.37%
```

（tier 的 mean 是 13 物体指标的算术平均，与旧文件 overall 的"总成功/总样本"
口径不同——两者都在报告里保留：`overall` 段是旧口径原值，`tiers[*].mean`
是新库表格口径。）

测试 `::test_prior_import_conversion_real_files` 在真实文件上断言逐物体数值
（ape ADD 51.8095% / n=1050、overall 82.7254%（oracle 上界）、top5_best ape
42.57%）+ 全部 oracle 标注字段 + `non_oracle_reference` 的 49.49/59.22 确实来自
top1 档；`::test_non_oracle_reference_omitted_without_top1` 保证缺 top1 时省略
而不是编数。skip 条件同时覆盖两个源文件（原先只查 all_objects40，
函数体却也读 top1_top3_top5）。

### 8.6 `configs/legacy_mypose.yaml` 与旧管线逐项对应

配置用 `base: default.yaml` 覆盖式继承（`src/config.py::load_config` 新增的
深合并 overlay），只声明差异项，避免整份复制漂移。

| 覆盖项 | 值 | 旧代码出处 |
|---|---|---|
| `detection.segmenter` | `gt_mask` | `:227-229`（`use_gt_mask=True`）+ `:281-311`（裁剪全程用 GT coseg mask）。**不是 `yolo`**：旧代码只把 `load_yolo_det` 传给 dataset loader，主循环里 YOLO bbox 从未参与任何计算；本库这条路线下 `YoloBboxLocalizer` 的 bbox（被 `tight_square` 的方形框覆盖）、mask（就是传入的 GT mask 原样）、score（无人消费）全是死值，代价却是硬依赖 ultralytics + 需要仓库里不存在的 `weights/yolo_linemod.pt`（配置开箱跑不起来）+ 每帧多一次前向 + YOLO 漏检直接丢帧（旧代码没有这个失败模式）。`YoloBboxLocalizer` 保留为独立消融项 |
| `detection.crop_mode` / `crop_size` / `crop_expand` | `tight_square` / 512 / 1.1 | `:286-311` |
| `templates.template_source` | `depth_map` | `:233-246` |
| `matching.lifting` | `depth_backproject` | `:409-432` |
| `matching.template_prescreen` | `none` | `:321-385` |
| `matching.template_ranking` / `top_k` | `mast3r` / 40 | `:366`（保留全部 40 候选） |
| `matching.cycle_tau_px` | `0.0`（严格互最近邻） | `:329-335` `fast_reciprocal_NNs` |
| `matching.sim_threshold` | `-1.0`（不做阈值过滤） | 旧代码无相似度阈值筛选 |
| `solver.pnp_flag` / `ransac_reproj_px` / `ransac_iterations` / `refine_lm` | `sqpnp` / 2.0 / 400 / false | `:441-449`（旧代码无 LM 精化） |
| `solver.selection` | `similarity` | `:375`（top-K 窗口按 MASt3R 相似度降序，不是按内点数——见 §8.4 窗口语义） |
| `metrics.topk_best` | `[1,3,5,40]` | `:524-533` + 两份真实结果 JSON（**除 top1 外全是 oracle 上界**） |

**已知非复现项（逐条列清）**：本配置复现旧管线的**设计选择**，不复现其**数值**。

| # | 差异 | 旧代码 | 本库 | 后果 |
|---|---|---|---|---|
| 1 | ADD-S 定义 | `:79-88` **双向 Chamfer** `0.5*(d_1to2+d_2to1)` | `src/metrics/pose_metrics.py:33-46` 标准**单向** mean-min（GSPose/BOP 口径） | eggbox/glue 口径不同，旧数字偏乐观 |
| 2 | Proj 定义 | `:145/:156/:166-168` 对 pred/GT 各自 z>0 过滤后按 min_len 截断再配对（点对应被破坏） | 不过滤、逐点严格对应 | 旧 Proj 与本库不可直接比 |
| 3 | 内点门槛位置 | `:451` `>= 6` 卡在**内点数** | `src/solver/ransac_pnp.py:68` `min_correspondences=6` 卡在**输入对应点数**，成功判定是 `len(inliers) < 4` | 本库会接纳旧代码丢弃的 4-5 内点候选 |
| 4 | 边界匹配过滤 | `:338-343` 剔除距边界 <3px 的匹配 | 无此过滤 | — |
| 5 | 匹配采样密度 | `:331` `subsample_or_initxy1=8`（8 像素网格） | 全部前景像素后再采样 `n_sample_corr` | — |
| 6 | 模板打分函数 | `:362` 互最近邻配对点积**求和** | `sim(m) = mean_y max_{y'} S` | 相似度排序只是"同为相似度序"，排名并非逐位相同 |
| 7 | 模板来源与单位 | 独立管线渲染 512×512，深度单位为米 | 3DGS 渲染（`templates` 其余字段沿用 default），深度与位姿同在尺度对齐后的物体系单位 | 反投影数学等价 |
| 8 | 像素中心约定 | 整数 uv + `cx=S/2` | 渲染器 gsplat 像素中心为 `(j+0.5, i+0.5)`，内参按整数像素索引约定给出（`cx=S/2-0.5`，见 §8.8） | 与旧代码差半像素，本库两条 lifting 路线内部自洽 |
| 9 | 旧代码实现缺陷 | AUDIT #6 desc 索引 x/y 转置、#7 查询图未过 `ImgNorm` 归一化 | 天然没有 | 旧数值不可复现且偏移方向未知 |

测试 `::test_legacy_config_reproduces_prior_pipeline`（逐项断言上表）、
`::test_default_config_new_switches_keep_new_behavior`（新增开关默认值 =
新库原行为，主实验不受影响）、`::test_base_overlay_does_not_leak_into_default`、
`::test_template_bank_path_separates_depth_bank`。

### 8.7 独立验证发现与修复记录

独立验证（第三方复核本节全部移植项）提出的问题与本轮修法，逐条对应到测试。

**第 1 组：oracle 标注（关系论文数字能不能用）**

| # | 发现的问题 | 修法 | 对应测试 |
|---|---|---|---|
| 1.1 | `README.md` 三处出现 82.73% 且措辞是"与**主实验** 82.73% 一致"，全无 oracle 字样；`top_k=40` 的说明被错误绑定到该数字 | README 改为"旧管线 Top-40 **GT 择优上界** 82.73%（oracle，非端到端；同批候选的端到端数字是 top1=49.49%）"；`matching.top_k` 说明与该数字解耦（top_k 只决定候选数） | 文档改动（无代码断言） |
| 1.2 | `README` 说 `legacy_aggregated.json` 可与旧 `aggregated_metrics_all_objects40.json` **直接对比** —— 前者是端到端评估（`evaluate_object` 对内点择优结果评估），后者是 oracle 上界 | 改为：`legacy_aggregated.json` 只能与旧 `top1`(49.49%) 比；与 82.73% 同口径可比的是 `legacy_aggregated_topk_best.json` 的 `top40_best` | 文档改动 |
| 1.3 | `VERIFICATION.md §8` 全节 0 次提 oracle，且把 82.7254% 称"真实结果"/"主结果" | §8 开头加醒目 oracle 结论段（复用 `legacy_format.py` 模块 docstring 措辞）；订正基准表、§8.4、§8.5、§8.6 的措辞 | 文档改动 |
| 1.4 | oracle 性质只写在 Python docstring 里，**JSON 产物本身看不出来** | `prior_to_report` / `prior_topk_to_report` 落 `protocol=prior_MyPose_oracle_top40 / _oracle_topk`、`selection=oracle_gt_add`、`is_oracle_upper_bound`、`tiers[*].is_oracle`、`non_oracle_reference`（新函数 `non_oracle_reference` 从 top1 档读，不硬编码）；`scripts/import_prior_metrics.py` 终端输出标 `[oracle]`/`[端到端]`；`results/prior/` 已重跑刷新 | `::test_prior_import_conversion_real_files`、`::test_non_oracle_reference_omitted_without_top1`、`::test_topk_from_key_inverts_topk_key` |
| 1.5a | `topk_best` 的候选顺序默认走 `inlier`（legacy 配置未覆盖），于是"top1"= 内点最优候选 = 新库端到端预测本身，而旧代码的 top-K 窗口是**按 MASt3R 相似度降序**（`:375`） | `configs/legacy_mypose.yaml` 补 `solver.selection: similarity` 并注明出处 | `::test_legacy_config_reproduces_prior_pipeline` |
| 1.5b | `selection.py` 先剔除 `success=False`，而旧代码对失败候选 append `inf`+dummy（`:516-520`）**占用 top-3/top-5 名额** → 新库 top3/top5 系统性乐观 | `rank_candidates(keep_failed=True)`；`estimate` 的 `candidates` 保留失败项（带 `success` 键、R/t 为 None）；`evaluate_object` 对失败候选记 `add=inf/proj=inf`。`select_best_candidate` / 主路线择优行为不变 | `::test_rank_candidates_keep_failed_occupies_topk_slots`、`::test_rank_candidates_default_still_drops_failed`、`::test_rank_candidates_order_and_best_consistency` |
| 1.5c | "复现旧 top1/3/5 消融"的措辞不准确（模板打分函数不同等） | README / 本文件 / yaml 注释统一改成"复现窗口**语义**、不复现数值"，并列出打分函数差异（旧 `:362` 互最近邻配对点积**求和** vs 本库 `mean_y max_{y'} S`） | 文档改动 |

**第 2 组：legacy 配置的真实性**

| # | 发现的问题 | 修法 | 对应测试 |
|---|---|---|---|
| 2.1 | `segmenter: yolo` 在 legacy 配置下是纯负担：旧代码 `:281-311` 全程用 GT mask，YOLO bbox 从未参与；本库这条路线下 YOLO 的 bbox/mask/score **全是死值**，代价是硬依赖 ultralytics + 需要仓库里不存在的 `weights/yolo_linemod.pt`（**开箱跑不起来**）+ 每帧多一次前向 + 漏检丢帧（旧代码无此失败模式） | `legacy_mypose.yaml` 改 `segmenter: gt_mask` 并写清理由；订正 README 与 §8.2/§8.6 的"YOLO bbox 定位"表述；`YoloBboxLocalizer` 保留为独立消融项 | `::test_legacy_config_reproduces_prior_pipeline`（断言 `gt_mask`）；`::test_select_best_yolo_box` / `::test_yolo_localizer_import_hint` 继续守 YOLO 消融项 |
| 2.2 | "已知非复现项"只写了模板分辨率/单位一项，其余差异（ADD-S 定义、Proj 定义、内点门槛位置、边界过滤、采样密度、打分函数、半像素、旧代码两处缺陷）全部缺失 | `legacy_mypose.yaml` 文末 + 本文件 §8.6 补齐 9 条清单，并写明结论"复现**设计选择**，不复现**数值**" | 文档改动 |

**第 3 组：半像素约定** —— 见 §8.8。

**第 4 组：健壮性与测试缺口**

| # | 发现的问题 | 修法 | 对应测试 |
|---|---|---|---|
| 4.1 | `TemplateBank` 不校验形状；`backproject_depth_to_model` 静默剔除越界像素，掩盖分辨率不匹配 | `TemplateBank` 校验 `depth_maps.shape[:3] == images.shape[:3]`（含 M 维）；越界像素改为 raise（带越界计数与深度图尺寸） | `::test_template_bank_raises_on_depth_shape_mismatch`、`::test_backproject_out_of_bounds_raises` |
| 4.2 | `src/config.py` 的 `base` 链成环是无信息量的 `RecursionError` | `load_config` 带 `_chain` 守卫，成环抛带完整链条的 `ValueError` | `::test_load_config_detects_base_cycle` |
| 4.3 | `_deep_merge`：base 侧是 dict 而 overlay 侧是 `None`（YAML 空冒号）时静默替换整段，错误在远端以 `AttributeError` 现形 | 显式报错并提示"要清空子段请写 `{}`"，消息带键路径 | `::test_deep_merge_rejects_none_over_dict_section` |
| 4.4 | `template_prescreen: none` + `template_ranking: dinov2` 组合下 ranking 静默失效 | `resolve_prefilter_order` 对该组合 raise（与本库"未知取值一律 raise"同纪律） | `::test_resolve_prefilter_order_none_with_dinov2_ranking_raises` |
| 4.5 | `TemplateBank` 对 `scale` 静默回退 1.0、`dino_feats` 静默为 None。onboard 在 render 落盘后、DINOv2 前中断（OOM 常见）留下的残留文件能被"正常"加载，**平移量级系统性错误且无报错** | 两者缺失一律 raise，消息带"删除并重新 onboard"的修法 | `::test_template_bank_raises_on_missing_scale`、`::test_template_bank_raises_on_missing_dino_feats` |
| 4.6 | `aggregate_topk_all_objects` 对**已 round** 的逐物体值再平均，旧代码 `:743-744` 是对未舍入值平均 | `object_topk_metrics` 内部保留全精度；新增 `_round_topk_entry`，只在 `aggregate_topk_all_objects` 的落盘输出上舍入（overall 与 per_object），且不篡改调用方的全精度 dict | `::test_aggregate_topk_averages_unrounded_per_object` |
| 4.7a | 深度反投影的参考实现对拍只在"深度全有效"下有意义（参考实现对无效点 `continue`，被测函数返回全长+valid 掩码），无效点分支从未与旧代码比过 | 参考实现支持 `depth_max`；新增含 `d<=0` 与 `d>depth_max` 的对拍，断言 `pts3d[valid]` 与压紧输出逐点一致 | `::test_backproject_matches_prior_reference_with_invalid_depths` |
| 4.7b | `test_base_overlay_does_not_leak_into_default` 近乎空测（`load_config` 每次重读文件，任何实现都能过） | 改成直接对 `_deep_merge` 断言"返回值的任何修改（含嵌套层与列表）都不影响传入的 base"，反向也断言；原端到端回归保留 | `::test_deep_merge_does_not_mutate_base`、`::test_base_overlay_does_not_leak_into_default` |
| 4.7c | 真实文件导入测试的 skipif 只查 `all_objects40.json`，函数体还读 `top1_top3_top5.json` | skip 条件覆盖两个文件 | `::test_prior_import_conversion_real_files` |

### 8.8 半像素约定统一（gsplat 像素中心）

**发现的问题**：gsplat 的像素中心是 `(j+0.5, i+0.5)`，证据两处——

- `third_party/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu:108`
  `const float px = (float)out_x + 0.5f;`
- `third_party/gsplat/gsplat/cuda/_torch_impl.py:784`
  `pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5`

而 `depth_lifting.py` 用整数 uv（与旧代码一致）、`template_renderer.py` 的
pyrender 坐标图分支也按整数 `xs/ys` 反投影，两者都配 `cx=S/2` 的内参。
后果：深度反投影路线有 `0.5·d/f` 的系统性横向偏置（默认 256/fov40° 下约
0.36mm ≈ ADD 阈值的 3.5%），而坐标图路线直接查表**没有**这个偏置 ——
`lifting: coord_map` vs `depth_backproject` 的消融里混进了与"提升方式"
无关的系统项。

**选定方案：把约定挪进内参**（不在各消费方散加 ±0.5）：

| 侧 | 内参 | 谁用 |
|---|---|---|
| 整数像素索引约定 | `template_intrinsics()` → `cx = cy = S/2 - 0.5` | 落盘的 `bank["K"]`、深度反投影、pyrender 坐标图反投影 |
| 渲染器像素中心约定 | `to_pixel_center_intrinsics(K)` → 主点 +0.5 | `trainer.render(...)`（3DGS 三处调用）、`pyrender.IntrinsicsCamera` |

焦距不变，故 `f_ref`（尺度对齐，论文 2.2.3）与渲染距离都不受影响。
`depth_lifting.py` 的 docstring 说明"与旧代码差半像素、原因是本库渲染器
像素中心约定不同"，并给出上面两条 gsplat 证据行号。

**回归**：`tests/test_legacy_mypose.py::test_rendered_depth_backprojects_to_gaussian_centers`
的假 trainer 落点从 `int(round(uv))` 改成 `floor(uv)`（贴合 gsplat 真实
约定：像素 j 覆盖 `[j, j+1)`），容差从 `d_max/f` 收紧到
`√2·0.5·d_max/f`（两轴各 ≤0.5px）。实测：

```
对齐(cx=S/2-0.5): nn.max=0.6986  tol=0.7937  → 通过
退回(cx=S/2)    : nn.max=1.2493  tol=0.7937  → 失败
```

即这条断言现在真的能拦住半像素回退（退回后的 1.2493 与独立验证者实测数字
一致）。

### 8.9 验证

```
cd exp_6dpose && .venv/bin/python -m pytest tests/ -q
157 passed in 1.88s
```

**142 原有全部保留（无删减，其中 8 条按上表 1.5b/2.1/3/4.1/4.7a-c 的口径修正
断言）+ 15 新增**（`tests/test_legacy_mypose.py` 14 条、
`tests/test_view_sampling.py` 1 条）。

### 8.10 旧 top40 候选 JSON 的出处与限制（scripts/analyze_prior_candidates.py 的输入）

`top40_add_proj_results_<obj>.json` 由旧 `inference_on_LM.py:556-578` 写出，
每样本含 40 个候选的 add/proj 与位姿 + `gt_pose`。分析脚本依赖的旧实现事实：

- **候选顺序**：候选按模板相似度降序存（`:375` `sort(key=score,
  reverse=True)`），故 `top5_details[0]` 就是端到端 top1。
- **占位位姿格式**：PnP 失败时写入单位旋转 + 零平移的 dummy 位姿
  （`:511-515`）；真实位姿的 t_z 恒 >0，可用「平移严格为零向量」判定并
  从误差分解中剔除，否则 30°/800mm 级的假误差会污染统计。
- **内点数不可重算**：旧 JSON 未存 RANSAC 内点数与模板相似度分数
  （`:568-578` 只存 add/proj/pose），内点数只在 `:451` 当 `<6` 失败门槛用完
  即丢。要拿「内点数/内点比/残差择优」的数字必须重跑 PnP（需要 2D-3D
  对应，即需要重跑 MASt3R 前向）。
- **误差三分解的来历**：旧 `test.ipynb` cell 26 只跑了 1 个样本就 break；
  分析脚本的表 5 是其全量统计化版本。
- 这批 top40 文件在服务器上（旧脚本的输出），分析前需先拷到本地。

## 9. 外部支撑核验 + BOP 指标融合（FoundPose / bop_toolkit 实读）

依据：`毕设/_reference/foundpose`（ECCV 2024 官方实现）与
`毕设/_reference/bop_toolkit`（BOP 官方评测库）的浅克隆实读。本节所有
合理性论断只引外部来源，不引本项目论文稿。

### 9.1 已有设计选择的外部支撑

| 本库操作 | 外部支撑（文件:行号） | 结论 |
|---|---|---|
| ADD-S 单向定义（GT→pred KDTree 均值，`pose_metrics.py` 默认） | bop_toolkit `pose_error.py:227-247`（官方 `adi` 正是 cKDTree(pts_est).query(pts_gt) 单向） | 成立；`bidirectional_legacy` 维持"非标准、仅对照旧数字"定性 |
| 候选择优用内点数（`selection: inlier`） | FoundPose `scripts/infer.py:594-602`（取 quality 最大者）+ `utils/pnp_util.py:79`（quality = len(inliers)） | 成立，与 FoundPose 完全同判据 |
| RANSAC-PnP + 内点集 LM 精化 | FoundPose `utils/pnp_util.py:46-74`（solvePnPRansac + solvePnPRefineLM on inliers） | 结构一致；参数差异：FoundPose ε=10px/400 迭代（`configs/infer/lmo.json`），本库 ε=5px/1000 迭代——两者都在常见区间，ε 已列消融 |
| 失败帧计入分母 | bop_toolkit `pose_matching.py`（未匹配估计即 miss，无位姿=未命中） | 成立 |
| Top-K 模板 + 逐候选 PnP 择优 | FoundPose `configs/infer/lmo.json` match_top_n_templates=5 后逐候选解 PnP 再择优 | 结构一致 |

**已知差异（不定性为错，列为消融/披露项）**：模板数——本库 8 视点 × 5
面内 = 40（Pos3R 设计）；FoundPose 为 57 视点 × 14 面内 = 798
（`configs/gen_templates/lmo.json`）。视点密度消融已在消融设计 P6。

### 9.2 BOP MSSD/MSPD 移植（`src/metrics/bop_metrics.py`）

逐行对照官方移植，出处全部写在模块 docstring：
- mssd/mspd：`pose_error.py:159-207`（对 GT 施对称集，逐点 max、对称 min）
- 对称展开：`misc.py:42-89`，离散步长默认 0.01（`eval_calc_errors.py:65`）
- 阈值：MSSD 0.05d..0.5d、MSPD 5..50（`eval_bop19_pose.py:46-56`）
- 归一化：MSSD ÷ diameter、MSPD × 640/图宽（`eval_calc_scores.py:296-307`）
- 命中判据严格小于（`pose_matching.py:66`）
- 点集用 models_eval 全部顶点不抽稀（MSSD 是逐点 max，抽稀系统性偏低）

**局限（对外报数必须写明）**：不含 VSD（需深度渲染器），`ar_bop` 是
(AR_MSSD+AR_MSPD)/2，与官方 AR=(VSD+MSSD+MSPD)/3 不可直接混称。
与 FoundPose 对表时可比项是 AR_MSSD/AR_MSPD 分量，或把 `--bop-csv`
导出的 bop19 提交文件交官方 bop_toolkit 复算（对账通道）。

### 9.3 验证

- `tests/test_bop_metrics.py` 19 条：解析式断言（纯平移=||Δt||、max 而非
  mean、翻转对称豁免、阈值表、严格小于、宽度归一化、CSV 往返）+
  **与官方 bop_toolkit 的逐数值 parity 测试 2 条**（随机位姿对上
  mssd/mspd 与 `get_symmetry_transformations` 逐矩阵一致；官方克隆
  不在时自动跳过）。
- 污染验证：把 `<` 改 `<=` → strict_less_than 单测红；把逐点 max 改
  mean → parity + is_max 两处红；恢复后全绿。
- 全套 `195 passed`（176 原有 + 19 新增）。

## 11. 训练背景色与浅色物体（2026-08-02 实验记录）

### 11.1 现象
13 物体子集（120 帧均匀采样，固定视图 + DS 训练 + 逆深度锚点）首轮：
eggbox ADD 9.2% / Proj 6.7%（其余物体正常 17.5%-94.2%）。

### 11.2 根因链（逐层排除）
1. 对称问题？→ 对称感知 GT 验证（展开 180° 变换）正确率仍 0.9% — 排除。
2. 锚点问题？→ CAD 表面锚点 GT 正确率同为 0.2% — 排除（锚点没坏）。
3. sims 过滤？→ sims 分层正确率全部 ~0.1-0.3% — 排除（不是低置信随机错配）。
4. 定位问题？→ GT 掩码定位 extract → ADD 97.5% — 定位也没坏（FastSAM+DINOv2
   候选框与 GT 几乎重合）。
5. **真凶：3DGS 训练背景色**。eggbox 浅黄 ≈ 训练白背景 → 边界像素 RGB 损失
   梯度≈0（任意混合色都≈真实值）→ 边界高斯糊 → 逆深度锚点系统性错 →
   匹配/求解全崩。模板渲染边缘密度 0.3% 属正常（纯背景无纹理），不是证据。
6. 交叉验证：黑背景（bg_color=0）+ depth_l1_weight=0.6 重训 eggbox →
   **ADD 9.2% → 98.3%**（同帧 120，Proj 94.2% / 5cm5° 80.0%）。

### 11.3 与 GSPose 官方对照
GSPose 训练 = `image * mask`（背景乘 0 = 黑），`white_background` 默认
False，渲染背景 [0,0,0]（build_3DGaussianObject.py:55）；损失
`trunc_FG_mask` 只在前景区域。本库此前 bg_color=1.0（白）偏离官方。
SSIM 是全图统计，白背景稀释物体区域约束 — 黑背景 + 前景截断 + 深度监督
（官方没有）三者对齐后才稳。

### 11.4 修复配置
- `configs/dense80_depth_bg0.yaml`：base dense80 + `gaussian.depth_l1_weight: 0.6`
  + `onboard.bg_color: 0.0`（黑背景）。
- 对称感知 PnP（`ransac_pnp(..., sym_transforms=...)`）：对称物体内点判定
  展开物体系离散对称变换（BOP models_info symmetries_discrete），与 ADD-S
  口径一致；采样不展开（避免 EPnP 混合解），LM 在内点最优分支上精化。
  注：eggbox 最终 98.3% 来自黑背景重训，对称展开在随机错配为主的旧数据上
  无法单独救回（正确对应率 <1% 时 RANSAC 无一致样本可采）。

### 11.5 关键文件/备份
- bank 备份：`*.npz.orig`（重训前白背景）、`*.viewsbak`（视图重建前）、
  `bank_backup_cadpatch/`（CAD patch 版，旧视图源）。
- 链脚本：`scripts/rerun13_bg0.sh`（12 物体黑背景重训→重提取→评估）。
