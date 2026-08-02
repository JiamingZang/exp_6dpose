"""对应过滤单测：互最近邻、cycle consistency、
阈值、Top-K、采样——全部用 numpy 合成特征验证。"""
import numpy as np
import pytest

from src.matching.correspondence import (back_to_original_pixels,
                                         build_correspondences,
                                         cycle_consistency_filter,
                                         mutual_nn_matches,
                                         sample_correspondences,
                                         template_score, topk_templates)


def _orthogonal_descs(n, d=32, seed=0):
    """构造两组一一对应的近正交描述子：第 i 个查询与第 i 个模板最相似。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    noise = rng.normal(scale=0.05, size=(n, d))
    return base, base + noise


def test_mutual_nn_identity_matching():
    dq, dt = _orthogonal_descs(64)
    iq, it, sims = mutual_nn_matches(dq, dt)
    # 近正交构造下应几乎全部匹配到同下标
    assert len(iq) >= 60
    assert np.mean(iq == it) > 0.95
    assert np.all(sims > 0.5)


def test_mutual_nn_rejects_one_directional():
    """模板侧塞一个『万能吸引子』描述子：它是很多查询的前向 NN，但它的
    反向 NN 只有一个查询——互最近邻应把其余单向匹配全部拒掉。"""
    rng = np.random.default_rng(1)
    dq = rng.normal(size=(20, 16))
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    attractor = dq.mean(axis=0, keepdims=True) * 10
    dt = np.concatenate([attractor, rng.normal(scale=1e-3, size=(5, 16))])
    iq, it, _ = mutual_nn_matches(dq, dt)
    # 吸引子最多贡献 1 对互最近邻
    assert np.sum(it == 0) <= 1


def test_mutual_nn_empty_input():
    iq, it, s = mutual_nn_matches(np.zeros((0, 8)), np.zeros((4, 8)))
    assert len(iq) == 0 and len(it) == 0 and len(s) == 0


def test_cycle_consistency_tau():
    """往返落点偏差 ≤ τ 保留、> τ 拒绝。"""
    # 4 个查询像素排成一列，间距 4px
    pix_q = np.array([[0.0, 0], [4.0, 0], [8.0, 0], [40.0, 0]])
    idx_q = np.array([0, 1, 2, 3])
    idx_t = np.array([0, 1, 2, 3])
    # 回程映射：0→自身(偏差0)，1→像素0(偏差4≤5)，2→像素3(偏差32>5)，3→自身
    nn_t2q = np.array([0, 0, 3, 3])
    keep = cycle_consistency_filter(pix_q, idx_q, idx_t, nn_t2q, tau_px=5.0)
    assert keep.tolist() == [True, True, False, True]


def test_template_score_ranks_correct_template_first():
    """正确模板（含匹配特征）的 sim(m) 应高于随机模板。"""
    # 注意：随机模板的种子必须与 _orthogonal_descs(seed=2) 不同，
    # 否则 default_rng(2) 会精确重放出同一批 base 描述子（cos=1）
    rng = np.random.default_rng(99)
    dq, dt_good = _orthogonal_descs(128, seed=2)
    dt_bad = rng.normal(size=(128, 32))
    s_good = template_score(dq, dt_good)
    s_bad = template_score(dq, dt_bad)
    assert s_good > s_bad
    assert s_good > 0.9        # 一一对应构造下接近 1


def test_topk_selection_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7, -0.2])
    assert topk_templates(scores, 3).tolist() == [1, 3, 2]
    # K 超过模板数时全量返回
    assert len(topk_templates(scores, 100)) == 5


def test_build_correspondences_end_to_end():
    """合成完整场景：正确匹配 + 干扰描述子，验证阈值与 3D 映射。"""
    n = 100
    dq, dt = _orthogonal_descs(n, seed=3)
    pix_q = np.stack([np.arange(n, dtype=float) * 3,
                      np.zeros(n)], axis=1)
    pix_t = pix_q.copy()
    coord_t = np.stack([np.arange(n, dtype=float),
                        np.arange(n, dtype=float) * 2,
                        np.ones(n)], axis=1)
    p2, p3, sims = build_correspondences(
        pix_q, dq, pix_t, dt, coord_t,
        sim_threshold=0.3, cycle_tau_px=5.0)
    assert len(p2) >= 90
    assert np.all(sims > 0.3)
    # 3D 锚点应与查询像素同下标（正确匹配 → Φ 映射一致）
    matched_idx = (p2[:, 0] / 3).astype(int)
    assert np.allclose(p3, coord_t[matched_idx])


def test_build_correspondences_threshold_kills_weak():
    """全随机特征 + 高阈值 → 对应集应大幅缩水甚至为空。"""
    rng = np.random.default_rng(4)
    dq = rng.normal(size=(50, 32))
    dt = rng.normal(size=(60, 32))
    pix_q = rng.uniform(0, 100, size=(50, 2))
    pix_t = rng.uniform(0, 100, size=(60, 2))
    coord_t = rng.normal(size=(60, 3))
    p2, p3, sims = build_correspondences(
        pix_q, dq, pix_t, dt, coord_t, sim_threshold=0.95, cycle_tau_px=5.0)
    assert len(p2) < 5


def test_sample_correspondences_cap():
    rng = np.random.default_rng(5)
    n = 10000
    p2 = rng.uniform(size=(n, 2))
    p3 = rng.uniform(size=(n, 3))
    s = rng.uniform(0.3, 1.0, size=n)
    o2, o3, os_ = sample_correspondences(p2, p3, s, n_sample=4096,
                                         rng=np.random.default_rng(0))
    assert o2.shape == (4096, 2) and o3.shape == (4096, 3)
    # 不足 N_s 时原样返回
    o2b, _, _ = sample_correspondences(p2[:100], p3[:100], s[:100], 4096)
    assert len(o2b) == 100


# ---------------------------------------------------------------------------
# 像素反变换：MASt3R 匹配区 → 裁剪区 → 原图（P1-2 复审）
# ---------------------------------------------------------------------------
def test_back_to_original_pixels_identity():
    """(sx,sy)=(1,1) 且 (x0,y0)=(0,0) 时应恒等。"""
    pix = np.array([[10.0, 20.0], [100.5, 42.25]])
    out = back_to_original_pixels(pix, (1.0, 1.0), (0, 0, 999, 999))
    assert np.allclose(out, pix)


def test_back_to_original_pixels_manual():
    """构造 crop=(100,50)、匹配 resize 因子 (2, 3)，人工核对反变换值。

    正向变换：orig=(150, 80) → crop=(50, 30) → matched=(100, 90)（乘 sx/sy）。
    反变换：matched=(100,90) --/(2,3)--> (50,30) --+(100,50)--> (150,80)。
    """
    matched = np.array([[100.0, 90.0]])
    out = back_to_original_pixels(matched, (2.0, 3.0), (100, 50, 200, 100))
    assert np.allclose(out, [[150.0, 80.0]])
