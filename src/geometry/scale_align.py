"""物理尺度对齐。

透视投影下 l_img = f · l_obj / Z。当参考图与查询图物距相近时（LineMod 协议
下参考与测试来自同一序列，该假设成立），投影尺度之比等于焦距之比，
因此尺度因子 s = f_query / f_ref，点云缩放 P_aligned = s · P_source。

纯 numpy 实现，本地 CPU 可测。
"""
from __future__ import annotations

import numpy as np


def scale_factor(f_query: float, f_ref: float, enabled: bool = True) -> float:
    """尺度因子 s = f_query / f_ref。

    Args:
        f_query: 查询图像焦距（像素）
        f_ref:   参考焦距——CAD 路线为渲染虚拟焦距；VGGT 路线由输出
                 相机参数 g_1 的视场角换算
        enabled: 尺度对齐消融关闭时返回 1.0
    """
    if not enabled:
        return 1.0
    if f_ref <= 0 or f_query <= 0:
        raise ValueError(f"焦距必须为正: f_query={f_query}, f_ref={f_ref}")
    return float(f_query) / float(f_ref)


def align_pointcloud(points: np.ndarray, s: float) -> np.ndarray:
    """点云尺度对齐 P_aligned = s · P_source。

    Args:
        points: (N,3) 原始点云（CAD 采样或 VGGT 输出）
        s: 尺度因子
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云形状应为 (N,3)，得到 {points.shape}")
    return points * float(s)
