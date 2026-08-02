"""位姿与投影基础工具（重投影误差 e_j 中的透视投影 π(K,·) 等）。

约定：
- 相机坐标系为 OpenCV 约定（x 右、y 下、z 前）。
- 位姿统一用 world-to-camera 表示：x_cam = R @ x_world + t。
- 内参 K 为 3×3 上三角矩阵。
纯 numpy 实现，本地 CPU 可测。
"""
from __future__ import annotations

import numpy as np


def rotz(phi: float) -> np.ndarray:
    """绕 z 轴（相机光轴）旋转 phi 弧度的 3×3 旋转矩阵。

    用于平面内旋转增强：在 w2c 旋转左乘 Rz(φ) 等价于
    把渲染出的图像绕主点旋转 φ。
    """
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at_wc(eye: np.ndarray, target: np.ndarray,
               up_hint: np.ndarray | None = None) -> np.ndarray:
    """构造相机位于 eye、光轴指向 target 的 world-to-camera 4×4 位姿。

    相机朝向指向物体中心。OpenCV 约定下相机 y 轴朝下，
    因此取 y_cam ≈ -up_hint 方向，保证世界上方在图像中朝上。
    当视线与 up_hint 接近平行时（如立方体顶点视角不会发生，但保险起见）
    切换备用 up。
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z_cam = target - eye
    z_cam = z_cam / np.linalg.norm(z_cam)

    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    up_hint = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(z_cam, up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])

    # 右手系：x = y × z。先取 y_tmp = -up（OpenCV y 向下），再正交化
    y_tmp = -up_hint
    x_cam = np.cross(y_tmp, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    # camera-to-world 旋转的列向量是相机各轴在世界系下的方向
    R_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = R_c2w.T                      # w2c 旋转
    t = -R @ eye
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def camera_center(T_wc: np.ndarray) -> np.ndarray:
    """从 w2c 位姿恢复相机光心在世界系下的坐标 C = -R^T t。"""
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    return -R.T @ t


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """两个旋转矩阵间的测地距离（度）。用于 5cm5° 指标。"""
    cos_theta = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def project_points(pts3d: np.ndarray, K: np.ndarray,
                   R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """透视投影 π(K, R·P + t)（重投影误差 e_j 使用的投影函数）。

    Args:
        pts3d: (N,3) 世界/模型系 3D 点
        K:     (3,3) 内参
        R,t:   w2c 旋转 (3,3) 与平移 (3,)
    Returns:
        (N,2) 像素坐标
    """
    pc = pts3d @ R.T + t.reshape(1, 3)
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]
