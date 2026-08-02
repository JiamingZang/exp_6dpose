"""定位纯逻辑与配置工具单测：余弦 max 聚合、bbox 扩 20%、消融覆盖。"""
from pathlib import Path

import numpy as np
import pytest

from src.config import apply_override, load_ablation, load_config
from src.detection.localize import (cosine_max_score, expand_bbox,
                                     template_similarity_order)
from src.matching.mast3r_wrapper import decode_template_indices

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = str(ROOT / "configs" / "default.yaml")


def _ab(name: str) -> str:
    return str(ROOT / "configs" / "ablations" / name)


def test_cosine_max_score():
    """score(M_n) = max_i cos(f_n, f_i^T)（模板维 max 聚合）。"""
    f = np.array([1.0, 0.0, 0.0])
    tmpl = np.array([[0.0, 1.0, 0.0],       # cos=0
                     [10.0, 0.0, 0.0],      # cos=1（模长不影响余弦）
                     [-1.0, 0.0, 0.0]])     # cos=-1
    assert np.isclose(cosine_max_score(f, tmpl), 1.0)
    tmpl2 = np.array([[1.0, 1.0, 0.0]])
    assert np.isclose(cosine_max_score(f, tmpl2), 1.0 / np.sqrt(2))


def test_expand_bbox_20pct():
    # bbox (100,100,50,40) 扩 20%：x±10, y±8
    box = expand_bbox((100, 100, 50, 40), 0.2, img_w=640, img_h=480)
    assert box == (90, 92, 160, 148)


def test_expand_bbox_clipped_to_image():
    box = expand_bbox((0, 0, 100, 100), 0.2, img_w=90, img_h=110)
    assert box == (0, 0, 90, 110)


# ---------------------------------------------------------------------------
# 配置与消融覆盖
# ---------------------------------------------------------------------------
def test_default_config_loads_and_matches_paper():
    """default.yaml 的关键超参必须与任务规格一致。"""
    cfg = load_config(DEFAULT_CFG)
    assert cfg["gaussian"]["iterations"] == 7000
    assert cfg["gaussian"]["lambda_ssim"] == 0.2
    n_tmpl = cfg["templates"]["n_viewpoints"] * cfg["templates"]["n_inplane"]
    assert n_tmpl == 40                      # 8 顶点 × 5 旋转
    assert cfg["templates"]["image_size"] == 512
    assert cfg["matching"]["top_k"] == 40
    assert cfg["matching"]["template_ranking"] == "dinov2"
    assert cfg["detection"]["segmenter"] == "fastsam"
    assert cfg["matching"]["sim_threshold"] == 0.3
    assert cfg["matching"]["cycle_tau_px"] == 5.0
    assert cfg["matching"]["n_sample_corr"] == 4096
    assert cfg["solver"]["ransac_reproj_px"] == 5.0
    assert cfg["solver"]["ransac_confidence"] == 0.999
    assert cfg["solver"]["ransac_iterations"] == 1000
    assert cfg["solver"]["joint_templates"] == 12
    assert cfg["onboard"]["train_crop"] == 512
    assert cfg["detection"]["bbox_expand"] == 0.2
    assert len(cfg["dataset"]["objects"]) == 13
    assert set(cfg["dataset"]["symmetric_objects"]) == {"eggbox", "glue"}


def test_apply_override_dotted():
    cfg = load_config(DEFAULT_CFG)
    out = apply_override(cfg, "matching.top_k", 10)
    assert out["matching"]["top_k"] == 10
    assert cfg["matching"]["top_k"] == 40     # 原 cfg 不被改动


def test_apply_override_preset():
    cfg = load_config(DEFAULT_CFG)
    out = apply_override(cfg, "templates.__preset__",
                         {"viewpoint_mode": "fibonacci",
                          "n_viewpoints": 16, "n_inplane": 5})
    assert out["templates"]["n_viewpoints"] == 16
    assert out["templates"]["image_size"] == 512   # 未覆盖字段保留


@pytest.mark.parametrize("path,n_runs", [
    (_ab("01_topk.yaml"), 5),
    (_ab("02_n_templates.yaml"), 4),
    (_ab("03_matcher.yaml"), 3),
    (_ab("04_localization.yaml"), 2),
    (_ab("05_geometry.yaml"), 2),
    (_ab("06_scale_align.yaml"), 2),
    (_ab("07_selection.yaml"), 3),
    (_ab("08_renderer.yaml"), 2),
    (_ab("09_ransac_eps.yaml"), 4),
    (_ab("10_segmenter.yaml"), 3),
])
def test_all_ablation_configs_parse(path, n_runs):
    """10 组消融 yaml 全部可解析且 sweep 数正确。"""
    cfg = load_config(DEFAULT_CFG)
    name, runs = load_ablation(cfg, path)
    assert len(runs) == n_runs
    for label, run_cfg, _ in runs:
        assert isinstance(run_cfg, dict)
        assert name in label


def test_ablation_n_templates_presets():
    cfg = load_config(DEFAULT_CFG)
    _, runs = load_ablation(cfg, _ab("02_n_templates.yaml"))
    counts = [r[1]["templates"]["n_viewpoints"] * r[1]["templates"]["n_inplane"]
              for r in runs]
    assert counts == [8, 24, 40, 80]
    assert all(r[2] for r in runs)           # 都需要重新 onboard


# ---------------------------------------------------------------------------
# Top-K 语义：DINOv2 排序传递 + 只解码 K 个
# ---------------------------------------------------------------------------
def test_template_similarity_order_ranks_desc():
    """fake CLS 特征：与候选最相似的模板排最前，返回降序 sims。"""
    feat = np.array([1.0, 0.0, 0.0])
    tmpl = np.array([[0.0, 1.0, 0.0],       # cos=0（下标 0）
                     [0.9, 0.1, 0.0],       # 高相似（下标 1）
                     [-1.0, 0.0, 0.0],      # cos=-1（下标 2）
                     [1.0, 0.0, 0.0]])      # cos=1（下标 3，最相似）
    order, sims = template_similarity_order(feat, tmpl)
    assert order[0] == 3                     # 最相似模板排第一
    assert order.tolist() == [3, 1, 0, 2]    # 全序降序
    assert np.all(np.diff(sims) <= 1e-9)     # sims 单调不增
    assert order.dtype == np.int64


def test_decode_indices_mast3r_ranking_decodes_all():
    """prefilter_order=None（mast3r 排序）：解码全部模板。"""
    idxs = decode_template_indices(40, top_k=5, prefilter_order=None)
    assert idxs == list(range(40))


def test_decode_indices_dinov2_ranking_only_k():
    """prefilter_order 给定（dinov2 排序）：只解码前 K 个，且保持排序。"""
    order = np.array([7, 3, 11, 0, 5, 9, 2])
    idxs = decode_template_indices(40, top_k=3, prefilter_order=order)
    assert idxs == [7, 3, 11]                # 恰 K=3 个，DINOv2 顺序
    # K 超过模板数时 clamp 到全序长度内的前 min(K,M) 个
    idxs_all = decode_template_indices(40, top_k=100, prefilter_order=order)
    assert idxs_all == [7, 3, 11, 0, 5, 9, 2]


def test_decode_indices_drops_out_of_range():
    """越界下标被过滤（模板库变小/排序陈旧时的鲁棒性）。"""
    order = np.array([2, 99, 1, -1, 0])
    idxs = decode_template_indices(3, top_k=5, prefilter_order=order)
    assert idxs == [2, 1, 0]


def test_ablation_scale_align_no_reonboard():
    """06 尺度对齐消融改为 VGGT-only，不再要求全量重训。"""
    cfg = load_config(DEFAULT_CFG)
    _, runs = load_ablation(cfg, _ab("06_scale_align.yaml"))
    assert [r[1]["scale_align"]["enabled"] for r in runs] == [True, False]
    assert not any(r[2] for r in runs)       # requires_reonboard=false


def test_ablation_ransac_eps_values():
    cfg = load_config(DEFAULT_CFG)
    _, runs = load_ablation(cfg, _ab("09_ransac_eps.yaml"))
    eps = [r[1]["solver"]["ransac_reproj_px"] for r in runs]
    assert eps == [3, 5, 8, 10]


def test_ablation_segmenter_values_include_gt_mask():
    cfg = load_config(DEFAULT_CFG)
    _, runs = load_ablation(cfg, _ab("10_segmenter.yaml"))
    segs = [r[1]["detection"]["segmenter"] for r in runs]
    assert segs == ["fastsam", "sam", "gt_mask"]
