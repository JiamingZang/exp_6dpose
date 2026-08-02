"""SAM 自动掩码 + DINOv2 模板匹配定位。GPU-only。

流程（在线步骤 1-3）：
1. SAM AutomaticMaskGenerator 生成候选掩码 {M_n}；
2. 每个候选掩码 bbox 裁剪 → 224×224 → DINOv2 ViT-L/14 CLS token f_n；
3. 与 40 个模板的 CLS 特征算余弦相似度，模板维 max 聚合：
       score(M_n) = max_i cos(f_n, f_i^T)
4. argmax 选目标掩码 n*，bbox 四周扩 20% 裁剪查询区域。

纯逻辑部分（余弦相似度 max 聚合 / argmax / bbox 扩展）拆在
`cosine_max_score` / `expand_bbox`，本地可测；模型推理封装在
SamDinoLocalizer，导入失败给出部署提示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_SAM_HINT = (
    "SAM/DINOv2 推理需要 GPU 机器：\n"
    "  pip install segment-anything  # + 下载 sam_vit_h_4b8939.pth（见 setup_gpu.sh）\n"
    "  DINOv2 经 torch.hub 自动下载（facebookresearch/dinov2, dinov2_vitl14）\n"
    "本地请使用 detection.segmenter=gt_bbox 或只跑 CPU 单测。"
)

_FASTSAM_HINT = (
    "FastSAM 推理需要 GPU 机器与 ultralytics 包（可选依赖）：\n"
    "  pip install ultralytics  # 权重 FastSAM-x.pt 首次运行自动下载\n"
    "本地请使用 detection.segmenter=gt_bbox / gt_mask 或只跑 CPU 单测。"
)

_YOLO_HINT = (
    "YOLO 检测需要 ultralytics 包（可选依赖，历史对照 YOLO bbox 定位路线）：\n"
    "  pip install ultralytics  # 权重按 detection.yolo_checkpoint 配置\n"
    "本地请使用 detection.segmenter=gt_bbox / gt_mask 或只跑 CPU 单测。"
)


# ---------------------------------------------------------------------------
# 纯逻辑（CPU 可测）
# ---------------------------------------------------------------------------
def cosine_max_score(feat: np.ndarray, template_feats: np.ndarray) -> float:
    """score(M_n) = max_i cos(f_n, f_i^T)（模板维 max 聚合）。"""
    f = feat / max(np.linalg.norm(feat), 1e-8)
    T = template_feats / np.maximum(
        np.linalg.norm(template_feats, axis=1, keepdims=True), 1e-8)
    return float((T @ f).max())


def centered_cosine_score(feat: np.ndarray, template_feats: np.ndarray,
                          f_white: np.ndarray) -> float:
    """白底中心化检索分数（cosine_max_score 的判别力改进版）。

    候选/模板图都是"白底 + 前景"，DINOv2 CLS token 被白底构图的主导分量
    拉向同一区域，全部候选分数挤在 0.4-0.6，argmax 被噪声碎片左右（实测
    背景小碎片反超目标掩码 0.019 分，lamp/eggbox 因此定位失败）。查询侧
    剥掉白底方向分量、模板侧剥掉模板均值后，分数分离度显著提高
    （lamp/eggbox/holepuncher 目标掩码从 #2/#2/#10 提到 #1）。
    """
    tc = template_feats - template_feats.mean(axis=0)
    tcn = tc / np.maximum(np.linalg.norm(tc, axis=1, keepdims=True), 1e-8)
    fc = feat - float(feat @ f_white) * f_white
    fcn = fc / max(np.linalg.norm(fc), 1e-8)
    return float((fcn @ tcn.T).max())


def template_similarity_order(feat: np.ndarray, template_feats: np.ndarray):
    """按余弦相似度对模板降序排序（Top-K 预筛用）。

    定位阶段已算出候选与全部模板的余弦相似度，此处直接复用得到排序，
    交给匹配器只解码 Top-K 个模板（K<M 时真正省算），无需额外前向。

    Returns:
        (order, sims)：order 为模板下标降序 (M,) int64；sims 为对应的
        降序余弦相似度 (M,)。
    """
    f = feat / max(np.linalg.norm(feat), 1e-8)
    T = template_feats / np.maximum(
        np.linalg.norm(template_feats, axis=1, keepdims=True), 1e-8)
    sims = T @ f
    order = np.argsort(-sims)
    return order.astype(np.int64), sims[order]


def expand_bbox(bbox_xywh, expand: float, img_w: int, img_h: int):
    """bbox 四周按边长比例扩 expand（默认 20%），并裁剪到图像范围内。"""
    x, y, w, h = [float(v) for v in bbox_xywh]
    dx, dy = w * expand, h * expand
    x0 = max(0, int(round(x - dx)))
    y0 = max(0, int(round(y - dy)))
    x1 = min(img_w, int(round(x + w + dx)))
    y1 = min(img_h, int(round(y + h + dy)))
    return x0, y0, x1, y1


def legacy_square_crop(img_rgb_u8: np.ndarray, mask: np.ndarray,
                       expand: float = 1.1, out_size: int = 512):
    """历史对照裁剪：mask 涂黑背景 + bbox 方形裁剪 + resize（见 VERIFICATION.md §8.2）。

    步骤：掩码外接框求中心（整除 2）→ 方形边长 = 长边 × expand →
    左上取 max(0, 中心-半边) → 右/下越界补 0 → mask 涂黑背景 →
    resize 到 out_size（INTER_LINEAR）。

    回映射：x_orig = x_out·(side/out) + left。本函数返回的
    sxy = out_size/side 与 crop_box=(left,top,...) 可直接喂给
    `back_to_original_pixels(pix, sxy, crop_box)`（pix/s + left 即上式）。

    Args:
        img_rgb_u8: (H,W,3) uint8 原图
        mask:       (H,W) bool 前景掩码
        expand:     方形边长相对掩码外接框长边的倍率（历史口径 1.1）
        out_size:   输出边长（历史口径 512）
    Returns:
        crop_u8:  (out,out,3) uint8，背景已涂黑并 resize
        mask_out: (out,out) bool，随裁剪同步 resize 的前景掩码
        crop_box: (left, top, left+side, top+side) 原图坐标
        sxy:      (s, s)，s = out_size/side，与 back_to_original_pixels 互逆
        失败（掩码为空）时返回 None
    """
    import cv2
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    side = int(max(x1 - x0, y1 - y0) * expand)
    if side < 2:
        return None

    half = side // 2
    top, left = max(0, int(cy) - half), max(0, int(cx) - half)
    bottom, right = top + side, left + side

    img_f = img_rgb_u8.astype(np.float32)
    cropped = img_f[top:bottom, left:right]
    mask_cropped = mask[top:bottom, left:right]

    h, w = cropped.shape[:2]
    if h < side or w < side:
        pad_h, pad_w = side - h, side - w
        cropped = np.pad(cropped, ((0, pad_h), (0, pad_w), (0, 0)),
                         constant_values=0.0)
        mask_cropped = np.pad(mask_cropped, ((0, pad_h), (0, pad_w)),
                              constant_values=0)

    masked = cropped * mask_cropped[..., None]
    resized = cv2.resize(masked, (out_size, out_size),
                         interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.resize(mask_cropped.astype(np.uint8), (out_size, out_size),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    s = out_size / float(side)
    crop_box = (left, top, left + side, top + side)
    return (np.clip(resized, 0, 255).astype(np.uint8), mask_out,
            crop_box, (s, s))


def select_best_yolo_box(boxes_xywh: np.ndarray, confs: np.ndarray):
    """从 YOLO 检测结果中取置信度最高的 bbox（单物体场景语义）。

    LineMod 每场景单物体，每帧只取一个框。返回 (bbox_xywh, conf)；
    无检测时返回 None。
    """
    boxes = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    confs = np.asarray(confs, dtype=np.float64).reshape(-1)
    if len(boxes) == 0 or len(confs) == 0:
        return None
    i = int(np.argmax(confs))
    return boxes[i], float(confs[i])


@dataclass
class Localization:
    """定位结果：目标掩码 + 扩展后的裁剪框（原图坐标）。"""
    mask: np.ndarray            # (H,W) bool，原图坐标系
    crop_box: Tuple[int, int, int, int]   # x0,y0,x1,y1（已扩 20%）
    score: float                # DINOv2 max 相似度
    best_template: int          # 最佳匹配模板下标（可视化/调试用）
    # DINOv2 相似度降序模板下标（Top-K 预筛用；gt_bbox/gt_mask
    # 无 DINOv2 检索时为 None，此时匹配器回退到 MASt3R 全解码排序）
    template_order: Optional[np.ndarray] = None
    # 按分数降序的 top-K 候选掩码（渲染验证消歧用，见 loc_n_candidates）。
    # 每项 {"mask","bbox_xywh","score","template_order"}；主候选 = 第一项。
    candidates: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU 推理封装
# ---------------------------------------------------------------------------
class Dinov2Embedder:
    """DINOv2 ViT-L/14 CLS 特征提取器。

    独立成类的原因：离线 onboard 阶段只需给 40 个模板提 CLS 特征，
    不应强制加载 2.4GB 的 SAM 权重。
    """

    def __init__(self, cfg_det: Dict, device: str = "cuda"):
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"{_SAM_HINT}\n原始错误: {e}") from e
        self.torch = torch
        self.device = device
        # hub 入口名见 dinov2/hubconf.py:7（dinov2_vitl14）
        self.dino = torch.hub.load("facebookresearch/dinov2:main",
                                   cfg_det.get("dinov2_model", "dinov2_vitl14")
                                   ).to(device).eval()
        self.input_size = int(cfg_det.get("dinov2_input_size", 224))
        # ImageNet 归一化（dinov2/data/transforms.py:42-47 官方常数）
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device
                                  ).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device
                                 ).view(3, 1, 1)

    def cls_feature(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        """任意尺寸 RGB uint8 → DINOv2 CLS token (1024,)。

        backbone 的 forward 返回 head(x_norm_clstoken)，head 是 Identity
        （dinov2/models/vision_transformer.py:169,348-353），即 CLS token 本体；
        GSPose 同样取 x_norm_clstoken 做检索（GSPose/model/network.py:197-199）。
        """
        return self.cls_features([img_rgb_u8])[0]

    def cls_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        """批量 CLS 特征：任意尺寸 RGB uint8 列表 → (N,1024)。

        定位阶段对全部 FastSAM 候选逐个前向（~100 次）是每帧 2.5s 的
        大头；拼 batch 一次前向（输入仅 ~15MB）后降到 ~0.3s。
        """
        import cv2
        torch = self.torch
        xs = []
        for img in imgs:
            im = cv2.resize(img, (self.input_size, self.input_size),
                            interpolation=cv2.INTER_LINEAR)
            x = torch.tensor(im, dtype=torch.float32, device=self.device
                             ).permute(2, 0, 1) / 255.0
            x = (x - self._mean) / self._std
            xs.append(x)
        x = torch.stack(xs)
        with torch.no_grad():
            feats = self.dino(x)                 # forward 默认返回 CLS
        return feats.float().cpu().numpy()

    def template_features(self, template_images: np.ndarray) -> np.ndarray:
        """预提取模板 CLS 特征 f_i^T（离线，onboard 时缓存到 npz）。"""
        return np.stack([self.cls_feature(im) for im in template_images])


class SamDinoLocalizer:
    """自动掩码（SAM ViT-H 或 FastSAM）+ DINOv2 CLS 检索的定位器。"""

    def __init__(self, cfg_det: Dict, device: str = "cuda",
                 embedder: Optional[Dinov2Embedder] = None,
                 segmenter: str = "sam",
                 bg_color: float = 1.0):
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"{_SAM_HINT}\n原始错误: {e}") from e
        self.torch = torch
        self.device = device
        self.cfg = cfg_det
        self.segmenter_name = segmenter
        # 候选裁剪的背景填充色须与模板渲染背景（onboard.bg_color）一致，
        # 否则 CLS 检索的"背景构图"分量与模板不同源（模板黑背景时涂白会
        # 系统性压低分数）。
        self._bg = int(round(bg_color * 255))

        if segmenter == "fastsam":
            # 主实验分割器：FastSAM（ultralytics）。掩码生成器统一暴露
            # generate() → SAM 风格 dict 列表
            self.mask_generator = FastSamSegmenter(cfg_det, device)
        elif segmenter == "sam":
            try:
                from segment_anything import (SamAutomaticMaskGenerator,
                                              sam_model_registry)
            except ImportError as e:
                raise ImportError(f"{_SAM_HINT}\n原始错误: {e}") from e
            # 构造协议见 segment_anything/build_sam.py:14,47-52（registry）与
            # automatic_mask_generator.py:35-40（points_per_side/pred_iou_thresh
            # 默认 32/0.88 即 SAM ViT-H 官方推荐）
            sam = sam_model_registry["vit_h"](
                checkpoint=cfg_det["sam_checkpoint"]).to(device)
            self.mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=int(cfg_det.get("sam_points_per_side", 32)),
                pred_iou_thresh=float(cfg_det.get("sam_pred_iou_thresh", 0.88)),
            )
        else:
            raise ValueError(
                f"未知 segmenter: {segmenter!r}（可选 sam / fastsam）")
        self.embedder = embedder or Dinov2Embedder(cfg_det, device)

        # 白底中心化检索基准（见 centered_cosine_score）：全白图的 CLS 特征
        # 代表"背景构图"分量，查询侧减去其方向。gt_bbox/gt_mask 模式不走
        # DINOv2 检索，不必算。
        self.f_white: Optional[np.ndarray] = None
        if cfg_det.get("loc_center_white", True):
            self.f_white = self._cls_feature(
                np.full((self.embedder.input_size,) * 2 + (3,), 255, np.uint8))
            self.f_white /= max(np.linalg.norm(self.f_white), 1e-8)

    def _cls_feature(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        return self.embedder.cls_feature(img_rgb_u8)

    # ------------------------------------------------------------------
    def localize(self, img_rgb_u8: np.ndarray,
                 template_feats: np.ndarray) -> Optional[Localization]:
        """在线定位。返回 None 表示无候选掩码（该帧失败）。

        generate() 返回 dict 列表，键 "segmentation"（HxW bool）与
        "bbox"（XYWH 格式）：SAM 见 automatic_mask_generator.py:184-192，
        FastSAM 由 FastSamSegmenter 转成同一格式。
        """
        h, w = img_rgb_u8.shape[:2]
        masks = self.mask_generator.generate(img_rgb_u8)
        if not masks:
            return None

        n_cand = int(self.cfg.get("loc_n_candidates", 3))
        # 批量 CLS 前向：全部候选一次前向（~100 候选逐前向是每帧 2.5s 的
        # 大头；拼 batch 输入仅 ~15MB，降到 ~0.3s）。
        crops, metas = [], []
        for m in masks:
            seg = m["segmentation"]
            x, y, bw, bh = m["bbox"]
            if bw < 8 or bh < 8:
                continue  # 过小碎片跳过：DINOv2 上采样后全是插值噪声
            crop = img_rgb_u8[int(y):int(y + bh), int(x):int(x + bw)]
            # 掩码外背景填模板同色背景（onboard.bg_color）：背景纹理会把
            # CLS token 拉向"构图相似"的假匹配（实测背景碎片反超本体）；
            # 填色必须与模板渲染背景一致，否则跨域检索分数系统性偏低
            seg_crop = seg[int(y):int(y + bh), int(x):int(x + bw)]
            crop = crop.copy()
            crop[~seg_crop] = self._bg
            crops.append(crop)
            metas.append((seg, (x, y, bw, bh)))
        if not crops:
            return None
        feats = self.embedder.cls_features(crops)
        scored = []
        for (seg, bbox), f in zip(metas, feats):
            x, y, bw, bh = bbox
            if self.f_white is not None:
                score = centered_cosine_score(f, template_feats, self.f_white)
            else:
                score = cosine_max_score(f, template_feats)
            scored.append((score, seg, bbox, f))
        if not scored:
            return None
        scored.sort(key=lambda s: -s[0])
        score, seg, bbox, feat = scored[0]
        # 定位阶段已算出的相似度直接复用为 Top-K 预筛排序，
        # best_template = 排序第一
        order, _ = template_similarity_order(feat, template_feats)
        crop_box = expand_bbox(bbox, float(self.cfg.get("bbox_expand", 0.2)),
                               w, h)
        candidates = []
        for s2, seg2, bbox2, feat2 in scored[:n_cand]:
            o2, _ = template_similarity_order(feat2, template_feats)
            candidates.append({
                "mask": seg2.astype(bool), "bbox_xywh": bbox2,
                "score": float(s2), "template_order": o2,
            })
        return Localization(mask=seg.astype(bool), crop_box=crop_box,
                            score=score, best_template=int(order[0]),
                            template_order=order, candidates=candidates)


class FastSamSegmenter:
    """FastSAM（ultralytics）自动实例分割（主实验分割器）。GPU-only。

    输出统一成 SAM 的掩码 dict 格式 [{"segmentation": HxW bool,
    "bbox": [x,y,w,h]}, ...]，供 SamDinoLocalizer 无感复用其 DINOv2 检索逻辑。
    """

    def __init__(self, cfg_det: Dict, device: str = "cuda"):
        try:
            from ultralytics import FastSAM
        except ImportError as e:
            raise ImportError(f"{_FASTSAM_HINT}\n原始错误: {e}") from e
        self.device = device
        self.model = FastSAM(cfg_det.get("fastsam_checkpoint", "FastSAM-x.pt"))
        self.conf = float(cfg_det.get("fastsam_conf", 0.4))
        self.iou = float(cfg_det.get("fastsam_iou", 0.9))

    def generate(self, img_rgb_u8: np.ndarray) -> List[Dict]:
        """单图 → SAM 风格掩码 dict 列表。"""
        results = self.model(img_rgb_u8, device=self.device, retina_masks=True,
                             conf=self.conf, iou=self.iou, verbose=False)
        out: List[Dict] = []
        for r in results:
            if r.masks is None:
                continue
            for seg_f in r.masks.data.cpu().numpy():
                seg = seg_f.astype(bool)
                ys, xs = np.nonzero(seg)
                if len(xs) == 0:
                    continue
                x0, y0 = int(xs.min()), int(ys.min())
                out.append({
                    "segmentation": seg,
                    "bbox": [x0, y0,
                             int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)],
                })
        return out


class GtBboxLocalizer:
    """定位消融上界：直接用 GT bbox（+20% 扩展）与 GT 可见掩码。"""

    def __init__(self, cfg_det: Dict):
        self.expand = float(cfg_det.get("bbox_expand", 0.2))

    def localize(self, img_rgb_u8: np.ndarray, bbox_xywh,
                 gt_mask: Optional[np.ndarray] = None) -> Localization:
        h, w = img_rgb_u8.shape[:2]
        crop_box = expand_bbox(bbox_xywh, self.expand, w, h)
        if gt_mask is None:
            # 无掩码时以 bbox 内全 1 近似（前景像素集合取 bbox 区域）
            gt_mask = np.zeros((h, w), dtype=bool)
            x, y, bw, bh = [int(v) for v in bbox_xywh]
            gt_mask[y:y + bh, x:x + bw] = True
        return Localization(mask=gt_mask.astype(bool), crop_box=crop_box,
                            score=1.0, best_template=-1)


class GtMaskLocalizer:
    """分割消融上界：直接用 GT 可见掩码定位（bbox 由掩码外接框推出）。

    loader 已提供 GT 掩码；此路径跳过 SAM/FastSAM，检验"分割完美"时的性能
    上界。
    """

    def __init__(self, cfg_det: Dict):
        self.expand = float(cfg_det.get("bbox_expand", 0.2))

    def localize(self, img_rgb_u8: np.ndarray,
                 gt_mask: np.ndarray) -> Optional[Localization]:
        h, w = img_rgb_u8.shape[:2]
        mask = np.asarray(gt_mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if len(xs) < 16:
            return None
        x0, y0 = int(xs.min()), int(ys.min())
        bbox = (x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1))
        crop_box = expand_bbox(bbox, self.expand, w, h)
        return Localization(mask=mask, crop_box=crop_box,
                            score=1.0, best_template=-1)


class YoloBboxLocalizer:
    """历史对照定位路线：YOLO bbox（+ 可选 GT coseg mask），见 VERIFICATION.md §8.2。

    YOLO 提供检测框，GT coseg mask 提供前景像素；gt_mask 给定时本定位器
    返回的 mask 就是 GT mask，YOLO 框仅在无 mask 时兜底（mask = 框内全 1）。

    ultralytics 为可选依赖，缺库时报带安装指引的 ImportError（不静默回退）。
    """

    def __init__(self, cfg_det: Dict, device: str = "cuda"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(f"{_YOLO_HINT}\n原始错误: {e}") from e
        self.device = device
        ckpt = cfg_det.get("yolo_checkpoint")
        if not ckpt:
            raise ValueError(
                "detection.segmenter=yolo 需要配置 detection.yolo_checkpoint"
                "（微调过的单物体检测权重路径）")
        self.model = YOLO(ckpt)
        self.conf = float(cfg_det.get("yolo_conf", 0.25))
        self.expand = float(cfg_det.get("bbox_expand", 0.2))

    def localize(self, img_rgb_u8: np.ndarray,
                 gt_mask: Optional[np.ndarray] = None
                 ) -> Optional[Localization]:
        h, w = img_rgb_u8.shape[:2]
        results = self.model(img_rgb_u8, device=self.device, conf=self.conf,
                             verbose=False)
        boxes, confs = [], []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for (bx0, by0, bx1, by1), c in zip(xyxy, cf):
                boxes.append([bx0, by0, bx1 - bx0, by1 - by0])
                confs.append(c)
        best = select_best_yolo_box(np.array(boxes), np.array(confs))
        if best is None:
            return None
        bbox, conf = best
        if gt_mask is not None:
            # 前景像素来自 GT coseg mask（历史对照口径）
            mask = np.asarray(gt_mask, dtype=bool)
            if mask.sum() < 16:
                return None
        else:
            mask = np.zeros((h, w), dtype=bool)
            bx, by, bw, bh = [int(round(v)) for v in bbox]
            mask[max(0, by):by + bh, max(0, bx):bx + bw] = True
        crop_box = expand_bbox(bbox, self.expand, w, h)
        return Localization(mask=mask, crop_box=crop_box,
                            score=conf, best_template=-1)
