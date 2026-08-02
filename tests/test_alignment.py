"""Umeyama 相似变换 / FPS / ICP / 位姿变换单测（VGGT→CAD 对齐），
以及分割器显式报错单测（禁止静默回退）。全部本地 CPU 可测。"""
import numpy as np
import pytest

from src.geometry.alignment import (farthest_point_sample, icp_refine,
                                     transform_pose_by_similarity,
                                     umeyama_alignment)
from src.geometry.pose_utils import rotation_angle_deg


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ---------------------------------------------------------------------------
# Umeyama：已知相似变换 + 噪声下的恢复精度
# ---------------------------------------------------------------------------
def test_umeyama_recovers_similarity_clean():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3)) * 5.0
    R_true = _random_rotation(rng)
    s_true, t_true = 2.3, np.array([4.0, -7.0, 1.5])
    dst = s_true * (R_true @ src.T).T + t_true
    s, R, t = umeyama_alignment(src, dst, with_scale=True)
    assert np.isclose(s, s_true, rtol=1e-6)
    assert rotation_angle_deg(R, R_true) < 1e-4
    assert np.allclose(t, t_true, atol=1e-6)


def test_umeyama_recovers_with_noise():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(500, 3)) * 10.0
    R_true = _random_rotation(rng)
    s_true, t_true = 0.7, np.array([2.0, 3.0, -5.0])
    dst = s_true * (R_true @ src.T).T + t_true
    dst += rng.normal(scale=0.02, size=dst.shape)      # 轻噪声
    s, R, t = umeyama_alignment(src, dst)
    assert np.isclose(s, s_true, rtol=1e-2)
    assert rotation_angle_deg(R, R_true) < 1.0
    assert np.allclose(t, t_true, atol=0.2)


def test_umeyama_no_scale_locks_unit():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(100, 3))
    R_true = _random_rotation(rng)
    dst = (R_true @ src.T).T + np.array([1.0, 2.0, 3.0])
    s, R, t = umeyama_alignment(src, dst, with_scale=False)
    assert s == 1.0
    assert rotation_angle_deg(R, R_true) < 1e-5


def test_umeyama_rejects_reflection():
    """含反射的对应也应返回 det=+1 的合法旋转。"""
    rng = np.random.default_rng(3)
    src = rng.normal(size=(50, 3))
    dst = src.copy()
    dst[:, 0] *= -1                                    # 镜像
    _, R, _ = umeyama_alignment(src, dst)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_umeyama_too_few_points():
    with pytest.raises(ValueError):
        umeyama_alignment(np.zeros((2, 3)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# FPS / ICP
# ---------------------------------------------------------------------------
def test_fps_unique_and_count():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(1000, 3))
    idx = farthest_point_sample(pts, 128, rng)
    assert len(idx) == 128
    assert len(np.unique(idx)) == 128
    # n_sample >= N 时全量返回
    assert np.array_equal(farthest_point_sample(pts[:50], 100), np.arange(50))


def test_icp_refines_from_rough_init():
    """无点对点对应（打乱 dst），ICP 从粗初值（与 pipeline 同策略）精化。"""
    rng = np.random.default_rng(5)
    src = rng.normal(size=(300, 3)) * 3.0
    ang = np.radians(12.0)
    R_true = np.array([[np.cos(ang), -np.sin(ang), 0],
                       [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    s_true, t_true = 1.4, np.array([1.0, -2.0, 0.5])
    dst = s_true * (R_true @ src.T).T + t_true
    dst = dst[rng.permutation(len(dst))]               # 打乱顺序

    # 粗初值：尺度取包围盒对角线之比（同 pipeline._compute_vggt_cad_alignment），
    # 旋转取真值附近 ~5° 扰动，平移取质心差
    def _diag(p):
        return float(np.linalg.norm(p.max(0) - p.min(0)))
    s0 = _diag(dst) / _diag(src)
    off = np.radians(5.0)
    R_off = np.array([[np.cos(off), -np.sin(off), 0],
                      [np.sin(off), np.cos(off), 0], [0, 0, 1.0]])
    R0 = R_off @ R_true
    t0 = dst.mean(0) - s0 * (R0 @ src.mean(0))
    s, R, t = icp_refine(src, dst, s0, R0, t0, iterations=40)
    assert np.isclose(s, s_true, rtol=3e-2)
    assert rotation_angle_deg(R, R_true) < 2.0


# ---------------------------------------------------------------------------
# 位姿变换回 CAD 系
# ---------------------------------------------------------------------------
def test_transform_pose_by_similarity_identity():
    """恒等相似变换下位姿不变。"""
    rng = np.random.default_rng(6)
    R_pose = _random_rotation(rng)
    t_pose = np.array([1.0, 2.0, 3.0])
    R_cad, t_cad = transform_pose_by_similarity(
        R_pose, t_pose, 1.0, np.eye(3), np.zeros(3))
    assert np.allclose(R_cad, R_pose)
    assert np.allclose(t_cad, t_pose)


def test_transform_pose_matches_camera_points():
    """核心正确性：CAD 系位姿投影的相机点 = s·(重建系相机点)。

    推导见 transform_pose_by_similarity：对 X = s·R_a·Y + t_a，
    有 R_cad·X + t_cad = s·(R_pose·Y + t_pose)。"""
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(40, 3)) * 2.0                 # 重建系模型点
    R_a = _random_rotation(rng)
    s_a, t_a = 1.8, np.array([3.0, -1.0, 2.0])
    X = s_a * (R_a @ Y.T).T + t_a                      # CAD 系模型点
    R_pose = _random_rotation(rng)
    t_pose = np.array([0.5, -0.3, 4.0])
    R_cad, t_cad = transform_pose_by_similarity(R_pose, t_pose, s_a, R_a, t_a)
    lhs = (R_cad @ X.T).T + t_cad
    rhs = s_a * ((R_pose @ Y.T).T + t_pose)
    assert np.allclose(lhs, rhs, atol=1e-8)
    assert np.isclose(np.linalg.det(R_cad), 1.0, atol=1e-8)   # 合法旋转


# ---------------------------------------------------------------------------
# 分割器显式报错（不许静默回退到 SAM）
# ---------------------------------------------------------------------------
def test_sam_localizer_unknown_segmenter_raises():
    from src.detection.localize import SamDinoLocalizer
    with pytest.raises(ValueError, match="segmenter"):
        SamDinoLocalizer({"sam_checkpoint": "x"}, device="cpu",
                         segmenter="unknown_value")


def test_pose_estimator_unknown_segmenter_raises():
    """pipeline 对未知 segmenter 显式 raise，不静默换模型。"""
    from src.pipeline import PoseEstimator
    cfg = {"detection": {"segmenter": "unknown_value"},
           "matching": {}, "solver": {}, "runtime": {"seed": 0}}
    with pytest.raises(ValueError, match="segmenter"):
        PoseEstimator(cfg, bank=None, device="cpu")
