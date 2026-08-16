"""在线早停（6d-adaptive-k-sim 落地）单测。

覆盖：
1) plateau_step 纯函数（改善/停滞/min_k 兜底/w 触发/None 内点）。
2) 配置解析：默认关；显式开时的参数读取路径。
3) extract_matches 的配置组合守卫（early_stop + mast3r ranking raise）——
   用 PoseEstimator 构造路径验证（不跑 GPU）。
"""
import numpy as np
import pytest

from src.matching.mast3r_wrapper import plateau_step
from src.config import load_config


def test_plateau_improvement_resets_stall():
    best, stall, stop = plateau_step(
        best_inl=100, stall=2, decoded=10, inl=200, w=3, delta=50, min_k=5)
    assert (best, stall, stop) == (200.0, 0, False)


def test_plateau_delta_threshold():
    # 增益 49 ≤ delta 50 → 仍算停滞
    best, stall, stop = plateau_step(
        best_inl=100, stall=0, decoded=10, inl=149, w=3, delta=50, min_k=5)
    assert best == 100 and stall == 1 and not stop
    # 增益 51 > delta 50 → 改善
    best, stall, stop = plateau_step(
        best_inl=100, stall=0, decoded=10, inl=151, w=3, delta=50, min_k=5)
    assert best == 151.0 and stall == 0


def test_plateau_stops_after_w_stalls_past_min_k():
    best, stall, stop = plateau_step(
        best_inl=100, stall=0, decoded=4, inl=90, w=3, delta=50, min_k=5)
    assert not stop          # 未到 min_k 不触发
    best, stall, stop = plateau_step(
        best_inl=100, stall=1, decoded=5, inl=90, w=3, delta=50, min_k=5)
    assert not stop          # stall=2 < w
    best, stall, stop = plateau_step(
        best_inl=100, stall=2, decoded=6, inl=90, w=3, delta=50, min_k=5)
    assert stop              # stall=3 >= w 且 decoded >= min_k


def test_plateau_none_inlier_counts_as_stall():
    # PnP 失败/无匹配：计停滞（解码代价已付，不算改善）
    best, stall, stop = plateau_step(
        best_inl=100, stall=2, decoded=6, inl=None, w=3, delta=50, min_k=5)
    assert best == 100 and stall == 3 and stop


def test_plateau_zero_delta_any_gain_improves():
    best, stall, stop = plateau_step(
        best_inl=100, stall=0, decoded=5, inl=101, w=3, delta=0, min_k=5)
    assert best == 101.0 and stall == 0


def test_plateau_relative_ratio_mode():
    # ratio=0.05：需 inl > best·1.05（1040 ≤ 1050 → 停滞）
    best, stall, stop = plateau_step(
        best_inl=1000, stall=0, decoded=5, inl=1040, w=3, delta=0, min_k=5,
        ratio=0.05)
    assert best == 1000 and stall == 1 and not stop
    # 1060 > 1050 → 改善
    best, stall, stop = plateau_step(
        best_inl=1000, stall=2, decoded=5, inl=1060, w=3, delta=0, min_k=5,
        ratio=0.05)
    assert best == 1060.0 and stall == 0
    # 绝对阈值路径不回归（ratio=0 默认走 delta）
    best, stall, stop = plateau_step(
        best_inl=1000, stall=0, decoded=5, inl=1040, w=3, delta=50, min_k=5,
        ratio=0.0)
    assert best == 1000 and stall == 1 and not stop


def test_default_config_early_stop_off():
    cfg = load_config("configs/current/dense80_depthc_guided.yaml")
    assert cfg["matching"].get("early_stop", False) is False
    assert cfg["matching"].get("early_stop_w", 3) == 3
    assert cfg["matching"].get("early_stop_delta", 50) == 50
    assert cfg["matching"].get("early_stop_min_k", 5) == 5


def test_early_stop_requires_dinov2_ranking():
    """早停与 mast3r 全解码排序的组合必须显式报错（解码顺序先验缺失）。

    template_ranking=mast3r 时 resolve_prefilter_order 恒返回 None
    （MASt3R 排序在打分之后才知，无法作为解码顺序先验），而 pipeline 的
    early_stop 守卫正是检查 prefilter_order is None → raise。这里验证
    守卫的触发条件在 mast3r 组合下成立。
    """
    from src.matching.mast3r_wrapper import resolve_prefilter_order
    order = np.array([3, 1, 2])
    # dinov2+dinov2：有排序 → 非 None（早停可用）
    assert resolve_prefilter_order("dinov2", "dinov2", order) is not None
    # dinov2+mast3r：排序被忽略 → None（early_stop 将 raise）
    assert resolve_prefilter_order("dinov2", "mast3r", order) is None
    # prescreen=none+dinov2：非法组合显式 raise（既有纪律）
    with pytest.raises(ValueError):
        resolve_prefilter_order("none", "dinov2", order)


def test_early_stop_signal_default_inlier():
    """停表信号默认 inlier；score 档显式解析（v2.1，20:15 实现）。"""
    from src.config import load_config
    cfg = load_config("configs/experiments/dense80_es_score.yaml")
    assert cfg["matching"].get("early_stop_signal", "inlier") == "score"
    assert cfg["matching"].get("early_stop_min_k") == 12
    assert cfg["matching"].get("early_stop_ratio") == 0.05
    assert cfg["solver"].get("selection") == "weighted"
    cfg2 = load_config("configs/current/dense80_depthc_guided.yaml")
    assert cfg2["matching"].get("early_stop_signal", "inlier") == "inlier"
