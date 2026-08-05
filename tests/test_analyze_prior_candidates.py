"""scripts/experiments/analyze_prior_candidates.py 的合成数据回归。

不依赖服务器上的真实 top40 JSON：构造一份字段结构与旧
`inference_on_LM.py:556-578` 完全一致的小样本，验证五张表的算法正确性，
以及占位位姿（PnP 失败）不污染误差统计。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "analyze_prior_candidates.py"
_spec = importlib.util.spec_from_file_location("apc", _SCRIPT)
apc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apc)


def _rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return {"R": R.tolist(), "t_mm": list(map(float, t)), "T44": T.tolist()}


def _gt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return {"gt_R": R.tolist(), "gt_t_mm": list(map(float, t)),
            "gt_T44": T.tolist()}


_DUMMY = {"R": np.eye(3).tolist(), "t_mm": [0.0, 0.0, 0.0],
          "T44": np.eye(4).tolist()}


def _cand(add, ok, pose, proj=1.0):
    return {"add": add, "add_success": ok, "proj_error": proj,
            "proj_success": ok, "pose": pose}


def _sample(cands, gt):
    """按旧格式包一层：外层 add/pose 取 GT-argmin（与旧 :526 同口径）。"""
    adds = [c["add"] for c in cands]
    j = int(np.argmin(adds))
    return {"add": adds[j], "add_success": cands[j]["add_success"],
            "proj_error": cands[j]["proj_error"],
            "proj_success": cands[j]["proj_success"],
            "best_template_idx": j, "pose": cands[j]["pose"],
            "gt_pose": gt, "top5_details": cands}


def _pad_to_40(cands, pose):
    """补到 40 个候选（全部失败），让 top40 档位有定义。"""
    while len(cands) < 40:
        cands.append(_cand(999.0, False, pose))
    return cands


def test_k_curve_and_first_hit_rank():
    """样本 A 首个命中在 rank1，样本 B 在 rank3 → top1=50%、top3=100%。"""
    gt = _gt(np.eye(3), [0, 0, 800])
    good = _pose(np.eye(3), [0, 0, 801])
    bad = _pose(_rot_x(40), [0, 0, 900])
    a = _sample(_pad_to_40([_cand(3.0, True, good), _cand(50.0, False, bad)], bad), gt)
    b = _sample(_pad_to_40(
        [_cand(60.0, False, bad), _cand(55.0, False, bad), _cand(4.0, True, good)],
        bad), gt)
    r = apc.analyze_object([a, b])

    assert r["total_samples"] == 2
    assert r["k_curve"]["top1"]["add_success_rate"] == 50.0
    assert r["k_curve"]["top1"]["is_oracle"] is False
    assert r["k_curve"]["top2"]["add_success_rate"] == 50.0
    assert r["k_curve"]["top3"]["add_success_rate"] == 100.0
    assert r["k_curve"]["top3"]["is_oracle"] is True
    assert r["first_hit_rank"]["rank1"] == 1
    assert r["first_hit_rank"]["rank2_3"] == 1
    assert r["first_hit_rank"]["never"] == 0


def test_never_hit_counted():
    """40 个候选全失败 → never 计数，且 top40 成功率为 0。"""
    gt = _gt(np.eye(3), [0, 0, 800])
    bad = _pose(_rot_x(40), [0, 0, 900])
    s = _sample(_pad_to_40([_cand(70.0, False, bad)], bad), gt)
    r = apc.analyze_object([s])
    assert r["first_hit_rank"]["never"] == 1
    assert r["k_curve"]["top40"]["add_success_rate"] == 0.0


def test_dummy_pose_excluded_from_error_stats():
    """占位位姿（t=0，旧 :511-515 的 PnP 失败标记）必须不进误差统计。

    否则 t=[0,0,0] vs GT t=[0,0,800] 会贡献一个 800mm 的假平移误差。
    """
    gt = _gt(np.eye(3), [0, 0, 800])
    good = _pose(np.eye(3), [0, 0, 810])
    # rank0 是占位位姿 → top1 误差统计应为空
    s = _sample(_pad_to_40([_cand(float("inf"), False, _DUMMY),
                            _cand(5.0, True, good)], _DUMMY), gt)
    r = apc.analyze_object([s])
    assert r["error_decomposition"]["top1_pose"]["n"] == 0
    oracle = r["error_decomposition"]["oracle_best_pose"]
    assert oracle["n"] == 1
    assert oracle["trans_mm_median"] == pytest.approx(10.0)


def test_dz_share_detects_depth_dominated_error():
    """平移误差全在 z 上 → dz_share ≈ 1（几何天花板的深度病签名）。"""
    gt = _gt(np.eye(3), [0, 0, 800])
    zoff = _pose(np.eye(3), [0, 0, 830])
    s = _sample(_pad_to_40([_cand(30.0, False, zoff)], zoff), gt)
    r = apc.analyze_object([s])
    e = r["error_decomposition"]["oracle_best_pose"]
    assert e["dz_mm_median"] == pytest.approx(30.0)
    assert e["dz_share"] == pytest.approx(1.0)
    assert e["rot_deg_median"] == pytest.approx(0.0, abs=1e-6)
    # oracle 挑中也失败 → 单独统计里有它
    assert r["error_decomposition"]["oracle_best_but_failed"]["n"] == 1


def test_rotation_error_recovered():
    """旋转 25° → rot_deg_median ≈ 25。"""
    gt = _gt(np.eye(3), [0, 0, 800])
    rot = _pose(_rot_x(25), [0, 0, 800])
    s = _sample(_pad_to_40([_cand(40.0, False, rot)], rot), gt)
    r = apc.analyze_object([s])
    e = r["error_decomposition"]["oracle_best_pose"]
    assert e["rot_deg_median"] == pytest.approx(25.0, abs=1e-4)
    assert e["trans_mm_median"] == pytest.approx(0.0)


def test_gap_table_bottleneck_classification():
    """oracle 上界 <60% ⇒ geometry_ceiling；否则缺口 >15 ⇒ retrieval_bound。"""
    per_obj = {
        "iron": {"total_samples": 100, "k_curve": {
            "top1": {"add_success_rate": 39.73}, "top40": {"add_success_rate": 97.75}},
            "error_decomposition": {"oracle_best_but_failed": {"n": 2, "dz_share": 0.9}}},
        "duck": {"total_samples": 100, "k_curve": {
            "top1": {"add_success_rate": 15.77}, "top40": {"add_success_rate": 42.63}},
            "error_decomposition": {"oracle_best_but_failed": {"n": 57, "dz_share": 0.95}}},
        # can 的缺口 16.93 > 15：top1 已高不等于没有择优空间，17 个点仍可拿
        "can": {"total_samples": 100, "k_curve": {
            "top1": {"add_success_rate": 82.48}, "top40": {"add_success_rate": 99.41}},
            "error_decomposition": {"oracle_best_but_failed": {"n": 0, "dz_share": None}}},
        # 缺口 8.0 < 15 且 oracle 高 → 真饱和，择优改进空间已基本用尽
        "sat": {"total_samples": 100, "k_curve": {
            "top1": {"add_success_rate": 91.0}, "top40": {"add_success_rate": 99.0}},
            "error_decomposition": {"oracle_best_but_failed": {"n": 1, "dz_share": 0.6}}},
    }
    rows = apc.build_gap_table(per_obj)
    kinds = {r["object"]: r["bottleneck"] for r in rows}
    assert kinds["iron"] == "retrieval_bound"
    assert kinds["duck"] == "geometry_ceiling"
    assert kinds["can"] == "retrieval_bound"
    assert kinds["sat"] == "near_saturated"
    # 按缺口降序
    assert [r["object"] for r in rows] == ["iron", "duck", "can", "sat"]


def test_cross_validation_flags_mismatch():
    """重算 top1 与参考不符 → MISMATCH（两次 run 设置不同，K 曲线不可同表）。"""
    per_obj = {"ape": {"total_samples": 1000,
                       "k_curve": {"top1": {"add_success_rate": 60.0},
                                   "top40": {"add_success_rate": 90.0}},
                       "error_decomposition": {}}}
    ref = {"overall_metrics": {"total_samples": 1000,
                               "top1": {"add_success_rate": 49.49}}}
    cv = apc.cross_validate(per_obj, ref)
    assert cv["status"] == "MISMATCH"

    ref_ok = {"overall_metrics": {"total_samples": 1000,
                                  "top1": {"add_success_rate": 60.0}}}
    assert apc.cross_validate(per_obj, ref_ok)["status"] == "match"


def test_load_candidate_files_accepts_bare_list_and_wrapped(tmp_path):
    """旧脚本的 JSON 顶层可能是裸 list 或带 add_results 键，两种都要吃。"""
    (tmp_path / "top40_add_proj_results_ape.json").write_text(json.dumps([{"a": 1}]))
    (tmp_path / "top40_add_proj_results_cat.json").write_text(
        json.dumps({"add_results": [{"a": 2}]}))
    got = apc.load_candidate_files(tmp_path)
    assert got == {"ape": [{"a": 1}], "cat": [{"a": 2}]}


def test_load_candidate_files_rejects_unknown_schema(tmp_path):
    (tmp_path / "top40_add_proj_results_ape.json").write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError, match="无法定位候选列表"):
        apc.load_candidate_files(tmp_path)
