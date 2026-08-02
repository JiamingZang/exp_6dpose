"""指标单测：ADD / ADD-S / Proj@5pix / 5cm5° 数值正确性。"""
import numpy as np
import pytest

from src.geometry.pose_utils import rotz
from src.metrics.pose_metrics import (add_error, adds_error, aggregate,
                                      cm_degree_errors, evaluate_pose,
                                      proj_error_px)

K = np.array([[572.4114, 0, 325.2611],
              [0, 573.5704, 242.0490],
              [0, 0, 1.0]])


def _cube_points(n=200, half=30.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=(n, 3))


def test_add_pure_translation():
    """纯平移偏移 Δt 下 ADD 应恰等于 ||Δt||。"""
    pts = _cube_points()
    R = np.eye(3)
    t_gt = np.array([0, 0, 500.0])
    dt = np.array([3.0, 4.0, 0.0])            # ||dt|| = 5
    assert np.isclose(add_error(pts, R, t_gt, R, t_gt + dt), 5.0)


def test_add_identity_is_zero():
    pts = _cube_points()
    R = rotz(0.3)
    t = np.array([1, 2, 300.0])
    assert add_error(pts, R, t, R, t) == pytest.approx(0.0, abs=1e-12)
    assert adds_error(pts, R, t, R, t) == pytest.approx(0.0, abs=1e-9)


def test_adds_symmetric_object_rotation():
    """对称物体绕对称轴旋转 180°：ADD 巨大，ADD-S 应≈0（
    eggbox/glue 用 ADD-S 的原因）。"""
    # 构造 z 轴 180° 旋转对称的点集：点对 (p, Rz(π)p) 成对出现
    rng = np.random.default_rng(1)
    half = rng.uniform(-30, 30, size=(100, 3))
    pts = np.concatenate([half, half @ rotz(np.pi).T])
    R_gt = np.eye(3)
    t = np.array([0, 0, 500.0])
    R_pred = rotz(np.pi)                       # 预测位姿差了个对称旋转
    add = add_error(pts, R_gt, t, R_pred, t)
    adds = adds_error(pts, R_gt, t, R_pred, t)
    assert add > 10.0                          # 普通 ADD 被对称性重罚
    assert adds < 1e-6                         # ADD-S 识别出等价位姿


def test_adds_leq_add():
    """任意情形下 ADD-S ≤ ADD（最近邻距离 ≤ 同点距离）。"""
    pts = _cube_points(seed=2)
    R_gt, t_gt = np.eye(3), np.array([0, 0, 400.0])
    R_pred = rotz(0.2)
    t_pred = t_gt + [5, -3, 8]
    assert adds_error(pts, R_gt, t_gt, R_pred, t_pred) <= \
        add_error(pts, R_gt, t_gt, R_pred, t_pred) + 1e-9


# ---------------------------------------------------------------------------
# ADD-S 定义开关：unidirectional（标准）| bidirectional_legacy（旧 :88）
# ---------------------------------------------------------------------------
def test_adds_bidirectional_is_mean_of_two_directions():
    """legacy 双向 = 0.5·(d_gt→pred + d_pred→gt)。反向距离用交换 GT/pred
    的单向调用独立复算（不复述实现）。"""
    pts = _cube_points(seed=3)
    R_gt, t_gt = np.eye(3), np.array([0, 0, 400.0])
    R_pred, t_pred = rotz(0.3), t_gt + [10, 0, -5]
    d_fwd = adds_error(pts, R_gt, t_gt, R_pred, t_pred)
    d_bwd = adds_error(pts, R_pred, t_pred, R_gt, t_gt)   # 交换两侧 = 反向
    d_bi = adds_error(pts, R_gt, t_gt, R_pred, t_pred,
                      definition="bidirectional_legacy")
    assert np.isclose(d_bi, 0.5 * (d_fwd + d_bwd), atol=1e-9)


def test_adds_definitions_agree_at_identity():
    pts = _cube_points(seed=4)
    R, t = np.eye(3), np.array([0, 0, 400.0])
    for definition in ("unidirectional", "bidirectional_legacy"):
        assert adds_error(pts, R, t, R, t, definition=definition) < 1e-9


def test_adds_unknown_definition_raises():
    pts = _cube_points(seed=5)
    R, t = np.eye(3), np.array([0, 0, 400.0])
    with pytest.raises(ValueError, match="definition"):
        adds_error(pts, R, t, R, t, definition="chamfer")


def test_evaluate_pose_threads_adds_definition():
    """evaluate_pose(symmetric=True) 的 adds_definition 要真的传到 adds_error：
    构造两个方向距离不对称的场景，两种定义必须给出不同 add 值。"""
    rng = np.random.default_rng(6)
    # 非均匀点云：主体 + 远处小簇，使 gt→pred 与 pred→gt 最近邻距离不对称
    pts = np.concatenate([rng.uniform(-30, 30, size=(80, 3)),
                          rng.uniform(-5, 5, size=(20, 3)) + [40, 40, 0]])
    R_gt, t_gt = np.eye(3), np.array([0, 0, 400.0])
    R_pred, t_pred = rotz(0.4), t_gt + [15, -8, 5]
    m_uni = evaluate_pose(pts, 100.0, K, R_gt, t_gt, R_pred, t_pred,
                          symmetric=True, adds_definition="unidirectional")
    m_leg = evaluate_pose(pts, 100.0, K, R_gt, t_gt, R_pred, t_pred,
                          symmetric=True,
                          adds_definition="bidirectional_legacy")
    assert not np.isclose(m_uni["add"], m_leg["add"]), \
        "两种定义给出相同值 → 参数没有透传到 adds_error"


def test_proj_error_known_shift():
    """深度不变的纯 x 平移：像素误差 = fx·Δx/Z（小孔模型解析值）。"""
    pts = np.zeros((10, 3))                   # 全部在物体原点
    R = np.eye(3)
    Z = 500.0
    t_gt = np.array([0, 0, Z])
    dx = 5.0
    err = proj_error_px(pts, K, R, t_gt, R, t_gt + [dx, 0, 0])
    assert np.isclose(err, K[0, 0] * dx / Z, rtol=1e-6)


def test_cm_degree_errors():
    R_gt = np.eye(3)
    R_pred = rotz(np.radians(3.0))
    t_gt = np.array([0, 0, 500.0])
    t_pred = t_gt + [30.0, 0, 0]
    te, re = cm_degree_errors(R_gt, t_gt, R_pred, t_pred)
    assert np.isclose(te, 30.0)
    assert np.isclose(re, 3.0, atol=1e-9)


def test_evaluate_pose_thresholds():
    """5cm5° / ADD@0.1d / Proj@5pix 的 0/1 判定边界。"""
    pts = _cube_points(seed=3)
    diameter = 100.0
    R = np.eye(3)
    t_gt = np.array([0, 0, 500.0])

    # 小误差：全指标命中
    m = evaluate_pose(pts, diameter, K, R, t_gt, rotz(np.radians(1.0)),
                      t_gt + [2.0, 0, 0])
    assert m["add_01d"] == 1.0 and m["cm_deg"] == 1.0 and m["proj_5px"] == 1.0

    # 平移差 60mm > 50mm：5cm5° 不中；ADD=60 > 0.1×100：ADD 不中
    m = evaluate_pose(pts, diameter, K, R, t_gt, R, t_gt + [60.0, 0, 0])
    assert m["cm_deg"] == 0.0 and m["add_01d"] == 0.0

    # 旋转 6° > 5°：5cm5° 不中
    m = evaluate_pose(pts, diameter, K, R, t_gt, rotz(np.radians(6.0)), t_gt)
    assert m["cm_deg"] == 0.0


def test_evaluate_pose_symmetric_flag():
    """symmetric=True 时 add 字段应为 ADD-S。"""
    half = _cube_points(50, seed=4)
    pts = np.concatenate([half, half @ rotz(np.pi).T])
    R = np.eye(3)
    t = np.array([0, 0, 500.0])
    m_sym = evaluate_pose(pts, 100.0, K, R, t, rotz(np.pi), t, symmetric=True)
    m_asym = evaluate_pose(pts, 100.0, K, R, t, rotz(np.pi), t, symmetric=False)
    assert m_sym["add_01d"] == 1.0
    assert m_asym["add_01d"] == 0.0


def test_aggregate_percentage():
    frames = [{"add_01d": 1.0, "proj_5px": 1.0, "cm_deg": 0.0},
              {"add_01d": 0.0, "proj_5px": 1.0, "cm_deg": 0.0},
              {"add_01d": 1.0, "proj_5px": 0.0, "cm_deg": 1.0},
              {"add_01d": 1.0, "proj_5px": 1.0, "cm_deg": 1.0}]
    agg = aggregate(frames)
    assert np.isclose(agg["add_01d"], 75.0)
    assert np.isclose(agg["proj_5px"], 75.0)
    assert np.isclose(agg["cm_deg"], 50.0)
    assert agg["n"] == 4
    assert aggregate([])["n"] == 0
