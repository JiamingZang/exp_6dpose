"""RANSAC-EPnP 与内点择优单测：合成已知位姿，投影加噪声/外点，
验证解回位姿误差在阈值内、择优逻辑正确。"""
import numpy as np
import pytest

from src.geometry.pose_utils import project_points, rotation_angle_deg
from src.solver.ransac_pnp import PnPResult, ransac_pnp
from src.solver.selection import select_best_candidate

K = np.array([[572.4114, 0, 325.2611],
              [0, 573.5704, 242.0490],
              [0, 0, 1.0]])


def _synth_scene(seed=0, n=500, noise_px=0.5, outlier_ratio=0.0):
    """合成场景：随机 3D 点（物体系，mm 量级）+ 已知位姿 + 投影观测。"""
    rng = np.random.default_rng(seed)
    pts3d = rng.uniform(-50, 50, size=(n, 3))              # ±5cm 物体
    # 随机旋转（QR 分解保证均匀正交）+ 前方 60cm 平移
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R_gt = Q
    t_gt = np.array([10.0, -20.0, 600.0])
    pts2d = project_points(pts3d, K, R_gt, t_gt)
    pts2d += rng.normal(scale=noise_px, size=pts2d.shape)
    n_out = int(n * outlier_ratio)
    if n_out:
        idx = rng.choice(n, n_out, replace=False)
        pts2d[idx] = rng.uniform([0, 0], [640, 480], size=(n_out, 2))
    return pts3d, pts2d, R_gt, t_gt, np.setdiff1d(np.arange(n),
                                                  idx if n_out else [])


def test_pnp_clean_recovers_pose():
    pts3d, pts2d, R_gt, t_gt, _ = _synth_scene(noise_px=0.3)
    res = ransac_pnp(pts2d, pts3d, K)
    assert res.success
    assert rotation_angle_deg(res.R, R_gt) < 1.0          # 旋转误差 < 1°
    assert np.linalg.norm(res.t - t_gt) < 5.0             # 平移误差 < 5mm
    assert res.n_inliers > 450


def test_pnp_robust_to_40pct_outliers():
    """RANSAC 核心价值：40% 外点下仍解回正确位姿。"""
    pts3d, pts2d, R_gt, t_gt, inlier_gt = _synth_scene(
        seed=1, noise_px=0.5, outlier_ratio=0.4)
    res = ransac_pnp(pts2d, pts3d, K)
    assert res.success
    assert rotation_angle_deg(res.R, R_gt) < 2.0
    assert np.linalg.norm(res.t - t_gt) < 10.0
    # 找到的内点应基本落在真实内点集合内
    precision = np.isin(res.inlier_idx, inlier_gt).mean()
    assert precision > 0.9


def test_pnp_too_few_points_fails_gracefully():
    pts3d = np.random.default_rng(0).uniform(-1, 1, size=(4, 3))
    pts2d = np.zeros((4, 2))
    res = ransac_pnp(pts2d, pts3d, K, min_correspondences=6)
    assert not res.success
    assert res.n_inliers == 0


def test_pnp_reproj_threshold_respected():
    """内点的重投影误差必须 < ε。"""
    pts3d, pts2d, *_ = _synth_scene(seed=2, noise_px=1.0, outlier_ratio=0.2)
    eps = 5.0
    res = ransac_pnp(pts2d, pts3d, K, reproj_px=eps, refine_lm=False)
    assert res.success
    uv = project_points(pts3d[res.inlier_idx], K, res.R, res.t)
    errs = np.linalg.norm(uv - pts2d[res.inlier_idx], axis=1)
    # OpenCV 判据即 e_j < ε；LM 关闭时内点集与判据严格对应
    assert np.all(errs < eps + 1e-6)


# ---------------------------------------------------------------------------
# 内点择优
# ---------------------------------------------------------------------------
def _mk(n_inl, score, tid, success=True):
    return PnPResult(success=success, n_inliers=n_inl,
                     template_idx=tid, template_score=score)


def test_selection_inlier_argmax():
    results = [_mk(120, 0.9, 0), _mk(300, 0.5, 1), _mk(50, 0.99, 2)]
    best = select_best_candidate(results, "inlier")
    assert best.template_idx == 1          # 内点数最多者胜


def test_selection_ignores_failed():
    results = [_mk(9999, 1.0, 0, success=False), _mk(10, 0.1, 1)]
    best = select_best_candidate(results, "inlier")
    assert best.template_idx == 1


def test_selection_all_failed_returns_none():
    results = [_mk(0, 0, 0, success=False), _mk(0, 0, 1, success=False)]
    assert select_best_candidate(results, "inlier") is None
    assert select_best_candidate([], "inlier") is None


def test_selection_similarity_strategy():
    results = [_mk(300, 0.5, 0), _mk(50, 0.99, 1)]
    best = select_best_candidate(results, "similarity")
    assert best.template_idx == 1          # 只看相似度


def test_selection_weighted_strategy():
    # 0: 100×0.8=80, 1: 200×0.3=60 → 选 0
    results = [_mk(100, 0.8, 0), _mk(200, 0.3, 1)]
    best = select_best_candidate(results, "weighted")
    assert best.template_idx == 0


def test_selection_unknown_strategy():
    with pytest.raises(ValueError):
        select_best_candidate([_mk(1, 1, 0)], "nope")


# ---------------------------------------------------------------------------
# 判据阶梯扩展：inlier_ratio / reproj
# ---------------------------------------------------------------------------
def _mk2(n_inl, tid, n_corr=0, reproj=float("inf"), success=True):
    return PnPResult(success=success, n_inliers=n_inl, template_idx=tid,
                     n_correspondences=n_corr, mean_inlier_reproj_px=reproj)


def test_selection_inlier_ratio_beats_raw_count():
    """匹配多的模板天然内点多——内点比要能反超绝对内点数。

    0 号 300/1000=0.30，1 号 90/100=0.90 → inlier 选 0、inlier_ratio 选 1。
    """
    results = [_mk2(300, 0, n_corr=1000), _mk2(90, 1, n_corr=100)]
    assert select_best_candidate(results, "inlier").template_idx == 0
    assert select_best_candidate(results, "inlier_ratio").template_idx == 1


def test_selection_inlier_ratio_zero_denominator():
    """n_correspondences=0 的候选不许除零，给 0 排末尾。"""
    results = [_mk2(0, 0, n_corr=0), _mk2(5, 1, n_corr=50)]
    assert select_best_candidate(results, "inlier_ratio").template_idx == 1


def test_selection_reproj_smaller_residual_wins():
    results = [_mk2(100, 0, n_corr=200, reproj=2.5),
               _mk2(100, 1, n_corr=200, reproj=0.8)]
    assert select_best_candidate(results, "reproj").template_idx == 1


def test_selection_reproj_failed_candidate_sinks():
    """失败候选残差为 inf → keep_failed=True 时必须排最后。"""
    from src.solver.selection import rank_candidates
    results = [_mk2(0, 0, success=False),
               _mk2(100, 1, n_corr=200, reproj=1.2)]
    ranked = rank_candidates(results, "reproj", keep_failed=True)
    assert [r.template_idx for r in ranked] == [1, 0]


def test_pnp_fills_criterion_fields():
    """真解路径：n_correspondences == 输入对应数；内点平均残差在 LM 精化后
    的最终位姿上重算，应与注入噪声同阶（0.3px → 远小于 1px）。"""
    pts3d, pts2d, R_gt, t_gt, _ = _synth_scene(noise_px=0.3)
    res = ransac_pnp(pts2d, pts3d, K, reproj_px=5.0)
    assert res.success
    assert res.n_correspondences == len(pts2d)
    assert 0.0 < res.mean_inlier_reproj_px < 1.0


def test_pnp_failure_path_carries_n_correspondences():
    """失败路径也要带 n_correspondences，否则 inlier_ratio 对失败候选无定义。"""
    res = ransac_pnp(np.zeros((3, 2)), np.zeros((3, 3)), K)
    assert not res.success and res.n_correspondences == 3
    assert res.mean_inlier_reproj_px == float("inf")


def test_end_to_end_topk_verification():
    """模拟 Top-K 几何验证：正确模板的对应能解出高内点位姿，错误模板的
    随机对应内点寥寥——择优应稳定选中正确模板。"""
    rng = np.random.default_rng(7)
    pts3d, pts2d, R_gt, t_gt, _ = _synth_scene(seed=7, noise_px=0.5)
    good = ransac_pnp(pts2d, pts3d, K)
    good.template_idx, good.template_score = 0, 0.7

    # 错误模板：2D-3D 完全乱配
    perm = rng.permutation(len(pts3d))
    bad = ransac_pnp(pts2d, pts3d[perm], K)
    bad.template_idx, bad.template_score = 1, 0.9

    best = select_best_candidate([good, bad], "inlier")
    assert best.template_idx == 0
    assert rotation_angle_deg(best.R, R_gt) < 1.5
