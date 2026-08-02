"""VGGT 前馈重建接口（几何初始化的补充路线，列为消融项）。

给定 3-5 张参考图，VGGT 单次前向输出点图 P_i（第一帧相机系）与相机参数
g_i = [q_i, t_i, f_i]，据此得到：
- P_source：合并各帧点图的物体点云（用参考掩码裁剪前景）
- f_ref：   由 g_1 的视场角换算的参考焦距，供尺度对齐 s = f_query / f_ref

GPU-only：依赖 vggt 包与 CUDA，本地 macOS 无法运行；导入失败时给出明确
安装提示。本地单测不覆盖本模块。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_VGGT_HINT = (
    "VGGT 前馈重建需要 GPU 机器：\n"
    "  pip install vggt  # 或 git clone https://github.com/facebookresearch/vggt\n"
    "权重 facebook/VGGT-1B 首次运行自动从 HuggingFace 下载（约 5GB 显存开销）。\n"
    "本地 macOS（无 CUDA）请使用 geometry.source=cad 主路线。"
)


def reconstruct_with_vggt(image_paths: List[str],
                          masks: Optional[List[np.ndarray]] = None,
                          checkpoint: str = "facebook/VGGT-1B",
                          device: str = "cuda",
                          conf_percentile: float = 30.0,
                          ) -> Tuple[np.ndarray, float]:
    """VGGT 重建物体点云并返回 (points (N,3), f_ref)。

    Args:
        image_paths: 3-5 张参考图路径（M=3~5）
        masks: 可选前景掩码（与图像同尺寸）；给定时只保留前景点
        conf_percentile: 置信度分位过滤（低置信点多为背景/噪声）

    Raises:
        ImportError: 本地无 vggt/CUDA 时给出部署提示
    """
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as e:  # pragma: no cover - GPU 机器路径
        raise ImportError(f"{_VGGT_HINT}\n原始错误: {e}") from e

    if not (2 <= len(image_paths) <= 32):
        raise ValueError("VGGT 支持 2-32 帧输入，论文实验取 3 或 5 帧")

    model = VGGT.from_pretrained(checkpoint).to(device).eval()
    images = load_and_preprocess_images(image_paths).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        preds = model(images)

    # 点图：第一帧相机系（即世界系）
    pts = preds["world_points"][0].float().cpu().numpy()        # (M,H,W,3)
    conf = preds["world_points_conf"][0].float().cpu().numpy()  # (M,H,W)

    # f_ref：从第一帧位姿编码换算内参焦距（由视场角换算）
    extri, intri = pose_encoding_to_extri_intri(
        preds["pose_enc"], images.shape[-2:])
    f_ref = float(intri[0, 0, 0, 0].cpu())

    keep = conf > np.percentile(conf, conf_percentile)
    if masks is not None:
        import cv2
        for m_idx, m in enumerate(masks):
            mm = cv2.resize(m.astype(np.uint8), (pts.shape[2], pts.shape[1]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            keep[m_idx] &= mm
    points = pts[keep].reshape(-1, 3)
    return points, f_ref
