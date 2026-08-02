"""深度图反投影 2D-3D 提升（历史对照口径，见 VERIFICATION.md §8.1）。

数学核对（勿凭记忆改动）：模板位姿 T_w2c = [R|t] 把模型系点 X 映到相机系
p_cam = R X + t；反解 X = R^T (p_cam - t) = R^T p_cam - R^T t。
K_inv 方向：p_cam 与像素满足 d·K_inv·[u,v,1]^T = p_cam（z 归一化针孔）。

单位约定（勿"修正"）：本库模板深度图与位姿同在尺度对齐后的物体系单位
（LineMod 下为 mm），无需任何米/毫米换算——调用方保证 depth_map 与
T_w2c 单位一致即可。

半像素约定（原因在渲染器不在本函数）：本函数用**整数 uv** 构造齐次向量，
但传进来的 K_tmpl 是 `geometry/view_sampling.template_intrinsics` 的
**整数像素索引约定**内参（主点 = S/2 - 0.5）。这个 -0.5 来自本库渲染器：
gsplat 的像素中心是 `(j+0.5, i+0.5)`，证据
- `third_party/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu:108`
  `const float px = (float)out_x + 0.5f;`
- `third_party/gsplat/gsplat/cuda/_torch_impl.py:784`
  `pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5`
若用 cx = S/2 + 整数 uv，本路线会有 `0.5·d/f` 的系统性横向偏置
（默认 256/fov40° 下约 0.36mm ≈ ADD 阈值的 3.5%），而坐标图路线直接查表
**没有**这个偏置 —— `lifting: coord_map` vs `depth_backproject` 的消融就会
混进与"提升方式"无关的系统项。约定统一落在内参里（渲染侧过
`to_pixel_center_intrinsics` 加回 0.5），两条路线同时对齐渲染器约定。

纯 numpy，本地 CPU 可测。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def backproject_depth_to_model(pix_t: np.ndarray, depth_map: np.ndarray,
                               K_tmpl: np.ndarray, T_w2c: np.ndarray,
                               depth_max: Optional[float] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """模板像素经深度图反投影到模型系 3D 点（历史对照的 2D-3D 提升）。

    对每个模板像素 (x, y)：
        d = depth_map[y, x]                    （无效深度 d<=0 剔除）
        p_cam   = d * K_inv @ [x, y, 1]^T
        p_model = R^T @ (p_cam - t)

    Args:
        pix_t:     (N,2) 模板像素坐标 (x, y)，与坐标图路线的 pix_t 同约定
        depth_map: (S,S) 模板深度图，单位与 T_w2c 平移一致（本库为尺度
                   对齐后的物体系单位）
        K_tmpl:    (3,3) 模板虚拟相机内参
        T_w2c:     (4,4) 模板位姿（模型系 → 相机系，w2c）
        depth_max: 深度上限（粗差剔除，历史口径 5.0 米）；None 表示不设上限
    Returns:
        pts3d: (N,3) 模型系 3D 点（无效点为 0，配合 valid 使用）
        valid: (N,) bool，深度有效（0 < d，且 d <= depth_max 若给定）
    Raises:
        ValueError: 有像素落在 depth_map 之外。**不静默剔除**——匹配坐标
            越界只可能是模板分辨率与匹配侧分辨率不一致（或拿错模板库），
            静默丢点会把这个配置错误伪装成"匹配质量差"。
    """
    pix = np.asarray(pix_t, dtype=np.float64)
    depth_map = np.asarray(depth_map, dtype=np.float64)
    T = np.asarray(T_w2c, dtype=np.float64)
    K_inv = np.linalg.inv(np.asarray(K_tmpl, dtype=np.float64))

    n = pix.shape[0]
    pts3d = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return pts3d, np.zeros(0, dtype=bool)

    xs = pix[:, 0].astype(np.int64)
    ys = pix[:, 1].astype(np.int64)
    h, w = depth_map.shape[:2]
    oob = (xs < 0) | (xs >= w) | (ys < 0) | (ys >= h)
    if oob.any():
        i = int(np.argmax(oob))
        raise ValueError(
            f"模板像素越界：{int(oob.sum())}/{n} 个点落在深度图之外"
            f"（首个越界点 (x={xs[i]}, y={ys[i]})，深度图尺寸 {w}×{h}）。"
            f"这是分辨率/模板库不匹配，不是匹配质量问题——请确认 "
            f"templates.image_size 与模板库一致，且没有混用深度库与坐标图库。")
    d = depth_map[ys, xs]

    # 无效深度判据：d <= 0 或超上限剔除（历史对照口径）
    valid = d > 0
    if depth_max is not None:
        valid &= d <= float(depth_max)
    if not valid.any():
        return pts3d, valid

    # 齐次向量用整数像素坐标（uv1 = [x1, y1, 1.0]，不做 +0.5 中心偏移）——
    # 半像素约定已统一进 K_tmpl，见模块 docstring
    uv1 = np.stack([xs[valid].astype(np.float64),
                    ys[valid].astype(np.float64),
                    np.ones(valid.sum())], axis=1)          # (M,3)
    p_cam = d[valid][:, None] * (uv1 @ K_inv.T)             # d * K_inv @ uv1
    R, t = T[:3, :3], T[:3, 3]
    # p_model = R_cam2model @ p_cam + t_cam2model = R^T (p_cam - t)
    pts3d[valid] = (p_cam - t) @ R
    return pts3d, valid
