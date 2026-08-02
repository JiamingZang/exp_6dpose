"""BOP 指标单测：MSSD / MSPD / 对称展开 / recall 阈值 / bop19 CSV。

关键设计：除解析式断言外，还有一组与 bop_toolkit 官方实现的逐数值对照
（parity 测试）——官方库在 毕设/_reference/bop_toolkit，存在时逐帧比对
本移植与官方 pose_error.mssd/mspd、misc.get_symmetry_transformations 的
输出；不存在则跳过（解析式断言仍然覆盖定义本身）。
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from src.geometry.pose_utils import rotz
from src.metrics.bop_metrics import (MSPD_THRESHOLDS, MSSD_THRESHOLDS,
                                     aggregate_bop, mspd_error, mspd_recall,
                                     mssd_error, mssd_recall, save_bop_csv,
                                     symmetry_transformations)

K = np.array([[572.4114, 0, 325.2611],
              [0, 573.5704, 242.0490],
              [0, 0, 1.0]])

IDENTITY_SYMS = [{"R": np.eye(3), "t": np.zeros((3, 1))}]


def _pts(n=200, half=30.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-half, half, size=(n, 3))


# ---------------------------------------------------------------------------
# MSSD / MSPD 定义本身
# ---------------------------------------------------------------------------
def test_mssd_identity_is_zero():
    pts = _pts()
    R, t = rotz(0.3), np.array([1, 2, 500.0])
    assert mssd_error(pts, R, t, R, t, IDENTITY_SYMS) == pytest.approx(0.0)


def test_mssd_pure_translation_is_max_norm():
    """无对称 + 纯平移：MSSD = ||Δt||（每点误差同为 ||Δt||，max 亦然）。"""
    pts = _pts()
    R = np.eye(3)
    t = np.array([0, 0, 500.0])
    dt = np.array([3.0, 4.0, 0.0])
    assert mssd_error(pts, R, t, R, t + dt,
                      IDENTITY_SYMS) == pytest.approx(5.0)


def test_mssd_is_max_not_mean():
    """MSSD 取逐点 max：纯旋转扰动下必然 ≥ ADD（逐点 mean）。"""
    from src.metrics.pose_metrics import add_error
    pts = _pts()
    R_gt, t = np.eye(3), np.array([0, 0, 500.0])
    R_pred = rotz(0.1)
    e_mssd = mssd_error(pts, R_gt, t, R_pred, t, IDENTITY_SYMS)
    e_add = add_error(pts, R_gt, t, R_pred, t)
    assert e_mssd > e_add > 0


def test_mssd_symmetry_absolves_flip():
    """180° z 对称物体：预测差一个 180° 翻转时 MSSD ≈ 0（对称集 min 生效）。"""
    pts = _pts()
    R_gt, t = np.eye(3), np.array([0, 0, 500.0])
    R_pred = rotz(np.pi)
    flip = [{"R": np.eye(3), "t": np.zeros((3, 1))},
            {"R": rotz(np.pi), "t": np.zeros((3, 1))}]
    e_no_sym = mssd_error(pts, R_gt, t, R_pred, t, IDENTITY_SYMS)
    e_sym = mssd_error(pts, R_gt, t, R_pred, t, flip)
    assert e_no_sym > 10.0
    assert e_sym == pytest.approx(0.0, abs=1e-9)


def test_mspd_identity_and_flip_symmetry():
    pts = _pts()
    R_gt, t = np.eye(3), np.array([0, 0, 500.0])
    assert mspd_error(pts, K, R_gt, t, R_gt, t,
                      IDENTITY_SYMS) == pytest.approx(0.0)
    flip = [{"R": np.eye(3), "t": np.zeros((3, 1))},
            {"R": rotz(np.pi), "t": np.zeros((3, 1))}]
    assert mspd_error(pts, K, R_gt, t, rotz(np.pi), t,
                      flip) == pytest.approx(0.0, abs=1e-6)


def test_mspd_invariant_to_depth_translation_direction():
    """MSPD 是图像域误差：沿光轴平移的误差远小于同幅度横向平移。"""
    pts = _pts()
    R, t = np.eye(3), np.array([0, 0, 500.0])
    e_lateral = mspd_error(pts, K, R, t, R, t + np.array([10.0, 0, 0]),
                           IDENTITY_SYMS)
    e_depth = mspd_error(pts, K, R, t, R, t + np.array([0, 0, 10.0]),
                         IDENTITY_SYMS)
    assert e_lateral > 5 * e_depth


# ---------------------------------------------------------------------------
# 对称展开（models_info.json → 变换集）
# ---------------------------------------------------------------------------
def test_symmetry_transformations_no_sym_is_identity_only():
    syms = symmetry_transformations({"diameter": 100.0})
    assert len(syms) == 1
    np.testing.assert_allclose(syms[0]["R"], np.eye(3))


def test_symmetry_transformations_discrete_parses_4x4():
    """离散对称按 4x4 展平解析（BOP models_info.json 的 eggbox/glue 形式）。"""
    m = rotz(np.pi)
    sym_flat = np.eye(4)
    sym_flat[:3, :3] = m
    sym_flat[:3, 3] = [1.0, 2.0, 3.0]
    syms = symmetry_transformations(
        {"symmetries_discrete": [sym_flat.flatten().tolist()]})
    assert len(syms) == 2
    np.testing.assert_allclose(syms[1]["R"], m)
    np.testing.assert_allclose(syms[1]["t"].reshape(3), [1.0, 2.0, 3.0])


def test_symmetry_transformations_continuous_discretization_count():
    """连续对称步数 = ceil(pi/step)（misc.py:68），恒等在 i=0 处包含。"""
    info = {"symmetries_continuous": [{"axis": [0, 0, 1],
                                       "offset": [0, 0, 0]}]}
    syms = symmetry_transformations(info, max_sym_disc_step=0.5)
    assert len(syms) == int(np.ceil(np.pi / 0.5))     # 7
    np.testing.assert_allclose(syms[0]["R"], np.eye(3), atol=1e-12)


def test_symmetry_transformations_combines_disc_and_cont():
    m = np.eye(4)
    m[:3, :3] = rotz(np.pi)
    info = {"symmetries_discrete": [m.flatten().tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1],
                                       "offset": [0, 0, 0]}]}
    syms = symmetry_transformations(info, max_sym_disc_step=0.5)
    assert len(syms) == 2 * int(np.ceil(np.pi / 0.5))


# ---------------------------------------------------------------------------
# recall 阈值（官方 eval_bop19_pose.py:46-56 的 10 档 + 严格小于）
# ---------------------------------------------------------------------------
def test_threshold_tables_match_bop19():
    np.testing.assert_allclose(MSSD_THRESHOLDS, np.arange(0.05, 0.51, 0.05))
    np.testing.assert_allclose(MSPD_THRESHOLDS, np.arange(5, 51, 5))


def test_mssd_recall_extremes_and_middle():
    d = 100.0
    assert mssd_recall(0.0, d) == pytest.approx(1.0)
    assert mssd_recall(60.0, d) == pytest.approx(0.0)     # 0.6d > 0.5d
    # 0.26d：命中 {0.30,0.35,0.40,0.45,0.50} 共 5/10 档
    assert mssd_recall(26.0, d) == pytest.approx(0.5)


def test_mssd_recall_strict_less_than():
    """恰等于阈值不命中（pose_matching.py:66 用严格 <）。"""
    # 阈值表由 np.arange 生成含浮点尾差，用 0.2 档（二进制可精确表示的
    # 0.05*4 仍有尾差）——直接对表取值构造恰等误差
    d = 100.0
    th = float(MSSD_THRESHOLDS[3])                        # ≈0.20
    err = th * d
    hits = mssd_recall(err, d)
    assert hits == pytest.approx(np.mean(err / d < MSSD_THRESHOLDS))
    assert not (err / d < MSSD_THRESHOLDS[3])             # 本档必不命中


def test_mspd_recall_width_normalization():
    """640 宽系数为 1；1280 宽误差减半计（eval_calc_scores.py:303-307）。"""
    assert mspd_recall(4.9, 640) == pytest.approx(1.0)
    assert mspd_recall(9.8, 1280) == pytest.approx(1.0)
    assert mspd_recall(9.8, 640) == pytest.approx(0.9)    # 命中 10..50 共 9 档
    assert mspd_recall(51.0, 640) == pytest.approx(0.0)


def test_aggregate_bop_means_and_empty():
    frames = [{"mssd_recall": 1.0, "mspd_recall": 0.5},
              {"mssd_recall": 0.0, "mspd_recall": 0.5}]
    agg = aggregate_bop(frames)
    assert agg["ar_mssd"] == pytest.approx(50.0)
    assert agg["ar_mspd"] == pytest.approx(50.0)
    assert agg["ar_bop"] == pytest.approx(50.0)
    assert agg["n"] == 2
    assert aggregate_bop([]) == {"ar_mssd": 0.0, "ar_mspd": 0.0,
                                 "ar_bop": 0.0, "n": 0}


# ---------------------------------------------------------------------------
# bop19 CSV
# ---------------------------------------------------------------------------
def test_save_bop_csv_format(tmp_path):
    rows = [{"scene_id": 1, "im_id": 42, "obj_id": 1, "score": 87,
             "R": rotz(0.3), "t": np.array([1.0, 2.0, 500.0]),
             "time": 0.5}]
    path = tmp_path / "sub.csv"
    save_bop_csv(path, rows)
    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["scene_id", "im_id", "obj_id",
                                     "score", "R", "t", "time"]
        row = next(reader)
    R_back = np.array(list(map(float, row["R"].split()))).reshape(3, 3)
    t_back = np.array(list(map(float, row["t"].split())))
    np.testing.assert_allclose(R_back, rotz(0.3))
    np.testing.assert_allclose(t_back, [1.0, 2.0, 500.0])
    assert row["time"] == "0.5"


def test_save_bop_csv_default_time(tmp_path):
    rows = [{"scene_id": 1, "im_id": 0, "obj_id": 1, "score": 1,
             "R": np.eye(3), "t": np.zeros(3)}]
    path = tmp_path / "sub.csv"
    save_bop_csv(path, rows)
    assert path.read_text().splitlines()[1].endswith(",-1")


# ---------------------------------------------------------------------------
# 与官方 bop_toolkit 的逐数值对照（存在克隆才跑）
# ---------------------------------------------------------------------------
_BOP_TOOLKIT = Path(__file__).resolve().parents[2] / "_reference" / "bop_toolkit"


@pytest.fixture(scope="module")
def bop_official():
    if not (_BOP_TOOLKIT / "bop_toolkit_lib").exists():
        pytest.skip("官方 bop_toolkit 克隆不存在（毕设/_reference/bop_toolkit）")
    sys.path.insert(0, str(_BOP_TOOLKIT))
    try:
        from bop_toolkit_lib import misc as bop_misc
        from bop_toolkit_lib import pose_error as bop_pe
    except ImportError as e:                      # 官方库依赖缺失也跳过
        pytest.skip(f"bop_toolkit 导入失败: {e}")
    finally:
        sys.path.pop(0)
    return bop_misc, bop_pe


def test_parity_mssd_mspd_vs_official(bop_official):
    """随机位姿对 + 翻转对称：本移植与官方 mssd/mspd 逐数值一致。"""
    bop_misc, bop_pe = bop_official
    pts = _pts(seed=3)
    flip = np.eye(4)
    flip[:3, :3] = rotz(np.pi)
    info = {"symmetries_discrete": [flip.flatten().tolist()]}
    syms = symmetry_transformations(info)
    rng = np.random.default_rng(11)
    for _ in range(5):
        R_gt = rotz(rng.uniform(0, 2 * np.pi))
        t_gt = np.array([*rng.uniform(-20, 20, 2), rng.uniform(400, 600)])
        R_pr = rotz(rng.uniform(0, 2 * np.pi))
        t_pr = t_gt + rng.uniform(-10, 10, 3)
        ours_mssd = mssd_error(pts, R_gt, t_gt, R_pr, t_pr, syms)
        ours_mspd = mspd_error(pts, K, R_gt, t_gt, R_pr, t_pr, syms)
        ref_mssd = bop_pe.mssd(R_pr, t_pr.reshape(3, 1),
                               R_gt, t_gt.reshape(3, 1), pts, syms)
        ref_mspd = bop_pe.mspd(R_pr, t_pr.reshape(3, 1),
                               R_gt, t_gt.reshape(3, 1), K, pts, syms)
        assert ours_mssd == pytest.approx(ref_mssd, rel=1e-9)
        assert ours_mspd == pytest.approx(ref_mspd, rel=1e-6)


def test_parity_symmetry_transformations_vs_official(bop_official):
    """对称展开与官方 get_symmetry_transformations 逐矩阵一致。"""
    bop_misc, _ = bop_official
    flip = np.eye(4)
    flip[:3, :3] = rotz(np.pi)
    info = {"symmetries_discrete": [flip.flatten().tolist()],
            "symmetries_continuous": [{"axis": [0, 0, 1],
                                       "offset": [1.0, 0, 0]}]}
    ours = symmetry_transformations(info, max_sym_disc_step=0.3)
    ref = bop_misc.get_symmetry_transformations(info, 0.3)
    assert len(ours) == len(ref)
    for a, b in zip(ours, ref):
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-9)
        np.testing.assert_allclose(a["t"].reshape(3), b["t"].reshape(3),
                                   atol=1e-9)


def test_parity_general_rotations_and_tilted_axis(bop_official):
    """一般姿态对拍：随机任意轴旋转 + 斜对称轴（非 z、含偏移），补齐
    仅绕 z 轴场景测不到的旋转分量。"""
    from scipy.spatial.transform import Rotation
    bop_misc, bop_pe = bop_official
    pts = _pts(seed=5)
    tilt = np.eye(4)
    tilt[:3, :3] = Rotation.from_rotvec([0.4, -0.7, 1.1]).as_matrix()
    tilt[:3, 3] = [2.0, -1.0, 0.5]
    info = {"symmetries_discrete": [tilt.flatten().tolist()],
            "symmetries_continuous": [{"axis": [1, 1, 0],
                                       "offset": [0.5, 0, -0.3]}]}
    ours_syms = symmetry_transformations(info, max_sym_disc_step=0.4)
    ref_syms = bop_misc.get_symmetry_transformations(info, 0.4)
    assert len(ours_syms) == len(ref_syms)
    for a, b in zip(ours_syms, ref_syms):
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-9)
        np.testing.assert_allclose(a["t"].reshape(3), b["t"].reshape(3),
                                   atol=1e-9)
    rng = np.random.default_rng(23)
    for _ in range(5):
        R_gt = Rotation.from_rotvec(rng.uniform(-np.pi, np.pi, 3)).as_matrix()
        R_pr = Rotation.from_rotvec(rng.uniform(-np.pi, np.pi, 3)).as_matrix()
        t_gt = np.array([*rng.uniform(-30, 30, 2), rng.uniform(400, 700)])
        t_pr = t_gt + rng.uniform(-15, 15, 3)
        assert mssd_error(pts, R_gt, t_gt, R_pr, t_pr, ours_syms) == \
            pytest.approx(bop_pe.mssd(R_pr, t_pr.reshape(3, 1),
                                      R_gt, t_gt.reshape(3, 1),
                                      pts, ref_syms), rel=1e-9)
        assert mspd_error(pts, K, R_gt, t_gt, R_pr, t_pr, ours_syms) == \
            pytest.approx(bop_pe.mspd(R_pr, t_pr.reshape(3, 1),
                                      R_gt, t_gt.reshape(3, 1),
                                      K, pts, ref_syms), rel=1e-6)
