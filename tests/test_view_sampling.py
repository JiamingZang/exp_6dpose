"""视角采样几何单测：位姿正交性、指向、覆盖、平面内旋转。"""
import numpy as np
import pytest

from src.geometry.pose_utils import camera_center, look_at_wc, rotz
from src.geometry.view_sampling import (cube_vertex_directions,
                                        fibonacci_directions,
                                        generate_template_poses,
                                        template_intrinsics,
                                        to_pixel_center_intrinsics)

RADIUS = 500.0


def _all_poses():
    return generate_template_poses(RADIUS, "cube8", 8, 5)


def test_template_count_8x5():
    poses = _all_poses()
    assert poses.shape == (40, 4, 4)


def test_rotation_orthogonality_and_det():
    """每个 P_m 的旋转块必须是行列式 +1 的正交矩阵（有效 SO(3)）。"""
    for T in _all_poses():
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    # 齐次末行不变
    assert np.allclose(_all_poses()[:, 3], [0, 0, 0, 1])


def test_camera_on_sphere_looking_at_origin():
    """相机在半径 r 球面上，光轴（相机 z 轴）指向原点。"""
    for T in _all_poses():
        C = camera_center(T)
        assert np.isclose(np.linalg.norm(C), RADIUS, atol=1e-9)
        # 相机 z 轴在世界系 = R 的第三行；应与 (原点-光心) 方向一致
        z_world = T[2, :3]
        expected = -C / np.linalg.norm(C)
        assert np.allclose(z_world, expected, atol=1e-9)


def test_viewpoint_coverage_all_octants():
    """8 立方体顶点方向应覆盖全部 8 个卦限（视角空间对称覆盖）。"""
    dirs = cube_vertex_directions()
    assert dirs.shape == (8, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    octants = set(map(tuple, np.sign(dirs).astype(int)))
    assert len(octants) == 8


def test_inplane_rotations_72deg_apart():
    """同一视角的 5 个平面内旋转应两两相差 72° 的整数倍（绕光轴）。"""
    poses = _all_poses()
    base = poses[0]
    for k in range(1, 5):
        T = poses[k]
        # 相对旋转 = R_k · R_0^T，应恰为 Rz(72°·k)
        R_rel = T[:3, :3] @ base[:3, :3].T
        expected = rotz(np.radians(72.0 * k))
        assert np.allclose(R_rel, expected, atol=1e-10)
        # 平面内旋转不改变光心位置
        assert np.allclose(camera_center(T), camera_center(base), atol=1e-9)


def test_all_poses_distinct():
    poses = _all_poses().reshape(40, -1)
    d = np.linalg.norm(poses[:, None] - poses[None, :], axis=-1)
    d += np.eye(40) * 1e9
    assert d.min() > 1e-6, "40 个模板位姿必须两两不同"


def test_fibonacci_uniformity():
    dirs = fibonacci_directions(16)
    assert dirs.shape == (16, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    # 均匀性弱检验：质心接近球心
    assert np.linalg.norm(dirs.mean(0)) < 0.15


def test_look_at_degenerate_up():
    """视线与默认 up 平行时应自动切换备用 up，不产生 NaN。"""
    T = look_at_wc(np.array([0, 0, 5.0]), np.zeros(3))
    assert np.all(np.isfinite(T))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_template_intrinsics_focal():
    K = template_intrinsics(256, 90.0)     # fov=90° → f = S/2
    assert np.isclose(K[0, 0], 128.0)
    # 整数像素索引约定：图像中心的连续坐标是 S/2，整数下标即 S/2-0.5
    # （gsplat 像素中心为 j+0.5，见 template_intrinsics docstring）
    assert np.isclose(K[0, 2], 127.5) and np.isclose(K[1, 2], 127.5)


def test_to_pixel_center_intrinsics_shifts_principal_point():
    """渲染器约定 = 整数像素索引约定的主点 +0.5，焦距不动，且不改原数组。"""
    K = template_intrinsics(256, 90.0)
    K_render = to_pixel_center_intrinsics(K)
    assert np.isclose(K_render[0, 2], 128.0) and np.isclose(K_render[1, 2], 128.0)
    assert np.isclose(K_render[0, 0], K[0, 0])
    assert np.isclose(K[0, 2], 127.5)      # 输入未被原地修改
