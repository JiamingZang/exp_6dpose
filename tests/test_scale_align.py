"""尺度对齐单测：s = f_query / f_ref。"""
import numpy as np
import pytest

from src.geometry.scale_align import align_pointcloud, scale_factor


def test_scale_factor_basic():
    assert np.isclose(scale_factor(1000.0, 500.0), 2.0)
    assert np.isclose(scale_factor(572.4114, 572.4114), 1.0)


def test_scale_factor_disabled_returns_one():
    """尺度对齐消融：关闭时 s=1。"""
    assert scale_factor(1000.0, 500.0, enabled=False) == 1.0


def test_scale_factor_invalid_focal():
    with pytest.raises(ValueError):
        scale_factor(-1.0, 500.0)
    with pytest.raises(ValueError):
        scale_factor(500.0, 0.0)


def test_align_pointcloud():
    pts = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    out = align_pointcloud(pts, 2.5)
    assert np.allclose(out, pts * 2.5)
    assert out is not pts          # 不改原数组


def test_align_pointcloud_shape_check():
    with pytest.raises(ValueError):
        align_pointcloud(np.zeros((3, 4)), 1.0)


def test_scale_projection_invariance():
    """核心性质：投影对 (s·X, s·t) 与 (X, t) 不变——pipeline 里
    t_model = t̂/s 换算正确性的依据。"""
    from src.geometry.pose_utils import project_points
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)) + [0, 0, 10.0]
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.1, -0.2, 2.0])
    s = 3.7
    uv1 = project_points(X, K, R, t)
    uv2 = project_points(X * s, K, R, t * s)
    assert np.allclose(uv1, uv2, atol=1e-9)
