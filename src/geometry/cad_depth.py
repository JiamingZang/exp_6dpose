"""CAD 网格 z-buffer 光栅化（numpy，无 GL 依赖）。

用途：
- 模板 3D 锚点（coord_map 表面版，patch_cad_coord_maps.py）
- 3DGS 训练的深度监督（参考视图 CAD 逆深度图，见 pipeline._build_reference_views）
"""
from __future__ import annotations

import numpy as np


def rasterize_cad_depth(verts, faces, R, t, K, size):
    """CAD 网格 z-buffer 光栅化，返回精确表面深度图 (size,size) float32。

    verts/faces: 模型顶点与三角面；R,t: w2c 位姿；K: 整数像素索引约定。
    0 = 无命中。
    """
    P = (R @ verts.T).T + t                      # (N,3) 相机系
    z = P[:, 2]
    vis = z > 0
    proj = np.full_like(P, -1e9)
    proj[vis, 0] = P[vis, 0] / z[vis] * K[0, 0] + K[0, 2]
    proj[vis, 1] = P[vis, 1] / z[vis] * K[1, 1] + K[1, 2]
    proj[vis, 2] = z[vis]

    depth = np.zeros((size, size), dtype=np.float32)
    xs = np.arange(size)
    ys = np.arange(size)
    for tri in faces:
        v = proj[tri]                            # (3,3): x,y,z
        if (v[:, 2] <= 0).any():
            continue
        x0, y0, z0 = v[0]; x1, y1, z1 = v[1]; x2, y2, z2 = v[2]
        # 不做背面剔除：屏幕 y 向下的绕序约定易错，z-buffer 本身
        # 保证只保留最近的正面（背面深度更大，会被正面覆盖）
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(size - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(size - 1, int(np.ceil(max(y0, y1, y2))))
        if xmin > xmax or ymin > ymax:
            continue
        gx, gy = np.meshgrid(xs[xmin:xmax + 1], ys[ymin:ymax + 1])
        gx = gx.astype(np.float32); gy = gy.astype(np.float32)
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        lam1 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        lam2 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        lam3 = 1.0 - lam1 - lam2
        inside = (lam1 >= 0) & (lam2 >= 0) & (lam3 >= 0)
        if not inside.any():
            continue
        gz = lam1 * z0 + lam2 * z1 + lam3 * z2
        sub = depth[ymin:ymax + 1, xmin:xmax + 1]
        upd = inside & ((sub == 0) | (gz < sub))
        sub[upd] = gz[upd]
    return depth


def render_cad_coord_maps(verts, faces, poses, K, size):
    """CAD 表面坐标图：z-buffer 反投影 → 物体系，逐像素精确表面点。"""
    coord_maps = []
    for T in poses:
        R, t = T[:3, :3], T[:3, 3]
        depth = rasterize_cad_depth(verts, faces, R, t, K, size)
        ys, xs = np.nonzero(depth > 0)
        z = depth[ys, xs]
        pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                       (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
        po = (pc - t) @ R                        # R^T (pc - t)，物体系
        cm = np.zeros((size, size, 3), dtype=np.float32)
        cm[ys, xs] = po
        coord_maps.append(cm)
    return np.stack(coord_maps)
