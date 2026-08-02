"""历史对照口径单测（见 VERIFICATION.md §8，对照 _prior_code/MyPose/inference_on_LM.py）。

四条移植能力 + 结果导入的本地（无 GPU/权重）验证：
1. 深度反投影 2D-3D 提升：与旧代码逐行转写的参考实现逐点对拍 +
   前向/反向闭环 + 无效深度剔除 + PnP 端到端恢复 GT；
2. 旧式方形裁剪：与旧代码逐行转写的参考实现对拍 + 坐标回映射闭环；
3. 全模板匹配开关（template_prescreen）与候选排序；
4. topK best 同步选择与旧 aggregated JSON 格式（用真实旧结果文件做
   口径回归：从 per_object 重算 overall 必须与旧文件一致）；
5. legacy_mypose.yaml 配置继承与逐项覆盖。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.detection.localize import legacy_square_crop, select_best_yolo_box
from src.matching.correspondence import back_to_original_pixels
from src.matching.depth_lifting import backproject_depth_to_model
from src.matching.mast3r_wrapper import resolve_prefilter_order
from src.metrics.legacy_format import (aggregate_all_objects,
                                       aggregate_topk_all_objects,
                                       non_oracle_reference,
                                       object_topk_metrics,
                                       per_object_from_frames,
                                       prior_to_report, prior_topk_to_report,
                                       topk_best_pick, topk_from_key,
                                       topk_key)
from src.solver.ransac_pnp import PnPResult, ransac_pnp
from src.solver.selection import rank_candidates, select_best_candidate

ROOT = Path(__file__).resolve().parents[1]
PRIOR_DIR = ROOT.parent / "_prior_code" / "MyPose"

K_TMPL = np.array([[512.0, 0, 256.0],
                   [0, 512.0, 256.0],
                   [0, 0, 1.0]])
K_QUERY = np.array([[572.4114, 0, 325.2611],
                    [0, 573.5704, 242.0490],
                    [0, 0, 1.0]])


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ---------------------------------------------------------------------------
# 1. 深度反投影 2D-3D 提升
# ---------------------------------------------------------------------------
def _prior_backproject_reference(pix_t, depth_img, K, R_model2cam,
                                 t_model2cam, depth_max=None):
    """旧代码 inference_on_LM.py:256,261-262,414-424 的逐行转写（参考实现）。

    保留原始逐点循环与中间量命名，用于和向量化实现逐点对拍——两者若有
    数值分歧，以本参考实现（=旧代码）为准。无效深度按旧代码 `continue`
    跳过（输出被压紧），故对拍时要比 `pts3d[valid]`。
    """
    K_inv = np.linalg.inv(K)                     # :256
    R_cam2model = R_model2cam.T                  # :261
    t_cam2model = -R_cam2model @ t_model2cam     # :262
    pts = []
    for x1, y1 in pix_t:                         # :414-417
        d = depth_img[int(y1), int(x1)]          # :418
        if d <= 0 or (depth_max is not None and d > depth_max):   # :419
            continue
        uv1 = np.array([x1, y1, 1.0])            # :422
        p_cam = d * (K_inv @ uv1)                # :423
        p_model = R_cam2model @ p_cam + t_cam2model   # :424
        pts.append(p_model)
    return np.array(pts).reshape(-1, 3)


def test_backproject_matches_prior_reference_pointwise():
    """向量化实现与旧代码逐行转写在随机深度图上逐点一致（<1e-9）。"""
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 3.0, size=(64, 64))
    R = _random_rotation(rng)
    t = np.array([5.0, -3.0, 400.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    pix = np.stack([rng.integers(0, 64, 50), rng.integers(0, 64, 50)], axis=1)

    ref = _prior_backproject_reference(pix, depth, K_TMPL, R, t)
    pts3d, valid = backproject_depth_to_model(pix, depth, K_TMPL, T)
    assert valid.all()                        # 深度全部 > 0
    assert ref.shape == pts3d.shape
    np.testing.assert_allclose(pts3d, ref, atol=1e-9)


def test_backproject_forward_inverse_closure():
    """反投影点变回相机系应恰为 d·K_inv·uv1，且再投影回到出发像素。

    同时核死两处方向约定：K_inv 方向（像素→归一化相机射线）与位姿逆变换
    顺序（先减 t 再乘 R^T，绝不能写成 R^T p - t）。
    """
    rng = np.random.default_rng(7)
    R = _random_rotation(rng)
    t = np.array([-2.0, 8.0, 350.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    depth = np.zeros((32, 32))
    pix = np.array([[3, 5], [17, 29], [30, 2]])
    ds = np.array([200.0, 340.0, 415.5])
    depth[pix[:, 1], pix[:, 0]] = ds

    pts3d, valid = backproject_depth_to_model(pix, depth, K_TMPL, T)
    assert valid.all()
    # 前向：模型系 → 相机系，z 必须等于写入的深度
    p_cam = pts3d @ R.T + t
    np.testing.assert_allclose(p_cam[:, 2], ds, atol=1e-9)
    # 再投影回到出发像素
    uv = (p_cam @ K_TMPL.T)
    uv = uv[:, :2] / uv[:, 2:3]
    np.testing.assert_allclose(uv, pix.astype(float), atol=1e-9)


def test_backproject_invalid_depth_filtered():
    """d<=0 / 超上限走 valid 掩码剔除（旧代码 :419 判据）。"""
    T = np.eye(4)
    T[2, 3] = 100.0
    depth = np.zeros((8, 8))
    depth[1, 1] = 2.0        # 有效
    depth[2, 2] = -1.0       # 负深度
    depth[3, 3] = 99.0       # 超上限
    pix = np.array([[1, 1], [2, 2], [3, 3], [5, 5]])   # [5,5] 深度为 0
    pts3d, valid = backproject_depth_to_model(pix, depth, K_TMPL, T,
                                              depth_max=50.0)
    assert valid.tolist() == [True, False, False, False]
    assert np.all(pts3d[~valid] == 0)


def test_backproject_out_of_bounds_raises():
    """越界像素显式 raise（不静默剔除）：越界只可能是分辨率/模板库不匹配，
    静默丢点会把配置错误伪装成"匹配质量差"。"""
    T = np.eye(4)
    T[2, 3] = 100.0
    depth = np.full((8, 8), 2.0)
    with pytest.raises(ValueError, match="越界"):
        backproject_depth_to_model(np.array([[1, 1], [100, 3]]), depth,
                                   K_TMPL, T)
    with pytest.raises(ValueError, match="越界"):
        backproject_depth_to_model(np.array([[-1, 0]]), depth, K_TMPL, T)


def test_backproject_matches_prior_reference_with_invalid_depths():
    """含无效深度时也与旧代码参考实现逐点一致（补齐对拍覆盖）。

    参考实现对无效点 `continue`（输出被压紧），被测函数返回全长 + valid
    掩码——因此必须比较 `pts3d[valid]` 与压紧输出。原来的对拍只在"深度
    全有效"下有意义，无效点分支从未与旧代码比过。
    """
    rng = np.random.default_rng(21)
    depth = rng.uniform(0.5, 3.0, size=(32, 32))
    # 掺入两类无效深度：d<=0 与 d>depth_max
    depth[::5, ::3] = 0.0
    depth[1::7, 2::4] = -0.7
    depth[2::6, 1::5] = 12.0          # > depth_max=5.0
    R = _random_rotation(rng)
    t = np.array([3.0, -1.0, 250.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    pix = np.stack([rng.integers(0, 32, 120), rng.integers(0, 32, 120)],
                   axis=1)

    ref = _prior_backproject_reference(pix, depth, K_TMPL, R, t,
                                       depth_max=5.0)
    pts3d, valid = backproject_depth_to_model(pix, depth, K_TMPL, T,
                                              depth_max=5.0)
    assert not valid.all() and valid.any()        # 确实混了有效与无效点
    assert len(ref) == int(valid.sum())
    np.testing.assert_allclose(pts3d[valid], ref, atol=1e-9)
    assert np.all(pts3d[~valid] == 0)


def test_backproject_lift_then_pnp_recovers_query_gt():
    """端到端：模板深度提升的 3D 点 + 查询投影像素 → PnP 恢复查询 GT。

    即旧管线『模板深度反投影 → 模型系 3D → solvePnPRansac』的合成闭环
    （inference_on_LM.py:409-449）。3D 点由整数模板像素反投影生成，
    避免深度图量化误差污染断言。
    """
    rng = np.random.default_rng(42)
    R_t = _random_rotation(rng)
    t_t = np.array([0.0, 0.0, 300.0])
    T_t = np.eye(4)
    T_t[:3, :3], T_t[:3, 3] = R_t, t_t
    # 模板整数像素网格 + 随机深度 → 模型系 3D 点
    gx, gy = np.meshgrid(np.arange(50, 460, 30), np.arange(50, 460, 30))
    pix_t = np.stack([gx.ravel(), gy.ravel()], axis=1)
    depth = np.zeros((512, 512))
    depth[pix_t[:, 1], pix_t[:, 0]] = rng.uniform(250.0, 380.0, len(pix_t))
    pts3d, valid = backproject_depth_to_model(pix_t, depth, K_TMPL, T_t)
    assert valid.all()

    # 查询 GT 位姿投影 3D 点得查询像素
    R_q = _random_rotation(rng)
    t_q = np.array([10.0, -20.0, 600.0])
    p_cam = pts3d @ R_q.T + t_q
    uv = p_cam @ K_QUERY.T
    uv = uv[:, :2] / uv[:, 2:3]

    res = ransac_pnp(uv, pts3d, K_QUERY)
    assert res.success
    from src.geometry.pose_utils import rotation_angle_deg
    assert rotation_angle_deg(res.R, R_q) < 0.1
    assert np.linalg.norm(res.t - t_q) < 1.0


# ---------------------------------------------------------------------------
# 2. 旧式方形裁剪（mask 涂黑 + 1.1 倍方形 + resize 512）
# ---------------------------------------------------------------------------
def _prior_crop_reference(img_np, mask_bin, out=512):
    """旧代码 inference_on_LM.py:286-311 的逐行转写（参考实现）。"""
    import cv2
    coords = np.argwhere(mask_bin)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    side = int(max(x1 - x0, y1 - y0) * 1.1)
    half = side // 2
    top, left = max(0, cy - half), max(0, cx - half)
    bottom, right = top + side, left + side
    cropped = img_np[top:bottom, left:right]
    mask_cropped = mask_bin[top:bottom, left:right]
    h, w = cropped.shape[:2]
    if h < side or w < side:
        pad_h, pad_w = side - h, side - w
        cropped = np.pad(cropped, ((0, pad_h), (0, pad_w), (0, 0)),
                         constant_values=0.0)
        mask_cropped = np.pad(mask_cropped, ((0, pad_h), (0, pad_w)),
                              constant_values=0)
    masked = cropped * mask_cropped[..., None]
    resized = cv2.resize(masked, (out, out), interpolation=cv2.INTER_LINEAR)
    return resized, side, left, top


def test_legacy_crop_matches_prior_reference():
    """legacy_square_crop 与旧代码转写在随机图/掩码上逐像素一致。"""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(480, 640, 3)).astype(np.uint8)
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:260, 300:420] = True                # 160×120 前景块

    ref, side, left, top = _prior_crop_reference(img.astype(np.float32), mask)
    out = legacy_square_crop(img, mask, expand=1.1, out_size=512)
    assert out is not None
    crop, mask_out, crop_box, sxy = out
    assert crop_box == (left, top, left + side, top + side)
    assert sxy == (512.0 / side, 512.0 / side)
    np.testing.assert_allclose(
        crop.astype(np.float32), np.clip(ref, 0, 255).astype(np.uint8)
        .astype(np.float32), atol=1.0)           # uint8 取整容差


def test_legacy_crop_blackens_background_and_pads_at_border():
    """背景涂黑 + 贴边掩码触发 0 填充（旧代码 :304-310）。"""
    import cv2
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    # 宽扁掩码贴下边：方形边长按长边（宽 60）取 66，纵向必然越过下边界，
    # 触发旧代码 :304-308 的 0 填充分支
    mask[80:100, 40:100] = True
    out = legacy_square_crop(img, mask, expand=1.1, out_size=64)
    assert out is not None
    crop, mask_out, crop_box, _ = out
    assert crop.shape == (64, 64, 3)
    # RGB 走双线性 resize、掩码走最近邻，边界 1-2 像素带必然含插值值；
    # 因此只断言"离掩码 2 像素以外的背景"严格为 0
    far_bg = ~(cv2.dilate(mask_out.astype(np.uint8),
                          np.ones((5, 5), np.uint8)) > 0)
    assert far_bg.any()
    assert crop[far_bg].max() == 0
    assert crop[mask_out].mean() > 150           # 掩码内保留原亮度
    # 方形右下越界 → crop_box 超出原图（padding 语义），左上仍在图内
    assert crop_box[2] > 100 or crop_box[3] > 100


def test_legacy_crop_backmap_matches_prior_formula():
    """回映射闭环：crop 坐标经 back_to_original_pixels 还原 =
    旧代码 x_orig = x·(side/512) + left（inference_on_LM.py:412,426-429）。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=bool)
    mask[120:280, 200:380] = True
    crop, _, crop_box, sxy = legacy_square_crop(img, mask)
    left, top = crop_box[0], crop_box[1]
    side = crop_box[2] - left
    scale_factor = side / 512.0                  # 旧代码 :412
    pix_512 = np.array([[10.0, 20.0], [500.0, 30.0], [256.0, 256.0]])
    expected = pix_512 * scale_factor + np.array([left, top])   # :426-429
    got = back_to_original_pixels(pix_512, sxy, crop_box)
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_legacy_crop_empty_mask_returns_none():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert legacy_square_crop(img, np.zeros((50, 50), dtype=bool)) is None


# ---------------------------------------------------------------------------
# 3. YOLO 定位器 + 全模板匹配开关
# ---------------------------------------------------------------------------
def test_select_best_yolo_box():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 8, 8]])
    confs = np.array([0.3, 0.9, 0.5])
    bbox, conf = select_best_yolo_box(boxes, confs)
    np.testing.assert_array_equal(bbox, [5, 5, 20, 20])
    assert conf == 0.9
    assert select_best_yolo_box(np.zeros((0, 4)), np.zeros(0)) is None


def test_yolo_localizer_import_hint(monkeypatch):
    """缺 ultralytics 时给出带安装指引的 ImportError（不静默回退）。"""
    from src.detection.localize import YoloBboxLocalizer
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ImportError, match="ultralytics"):
        YoloBboxLocalizer({"yolo_checkpoint": "x.pt"}, device="cpu")


def test_resolve_prefilter_order_none_forces_full_matching():
    """prescreen=none：返回 None（全模板 MASt3R 匹配，即旧管线
    inference_on_LM.py:321-385 行为）。"""
    assert resolve_prefilter_order("none", "mast3r", None) is None
    assert resolve_prefilter_order("none", "mast3r",
                                   np.array([3, 1, 2, 0])) is None


def test_resolve_prefilter_order_none_with_dinov2_ranking_raises():
    """prescreen=none + ranking=dinov2 → raise。

    预筛被跳过后 Top-K 只能按 MASt3R sim(m) 选，`template_ranking: dinov2`
    完全失效；静默生效的假象会让消融结论归错因，故按本库纪律显式拒绝。
    """
    with pytest.raises(ValueError, match="template_prescreen=none"):
        resolve_prefilter_order("none", "dinov2", np.array([3, 1, 2, 0]))
    with pytest.raises(ValueError, match="template_ranking"):
        resolve_prefilter_order("none", "dinov2", None)


def test_resolve_prefilter_order_dinov2_passthrough_and_fallback():
    order = np.array([3, 1, 2, 0])
    got = resolve_prefilter_order("dinov2", "dinov2", order)
    np.testing.assert_array_equal(got, order)
    # gt_mask/gt_bbox 定位无 DINOv2 排序 → 回退全解码
    assert resolve_prefilter_order("dinov2", "dinov2", None) is None
    # ranking=mast3r 时也不预筛
    assert resolve_prefilter_order("dinov2", "mast3r", order) is None


def test_resolve_prefilter_order_unknown_raises():
    with pytest.raises(ValueError, match="template_prescreen"):
        resolve_prefilter_order("dino", "dinov2", None)


def test_rank_candidates_order_and_best_consistency():
    rs = [PnPResult(success=True, n_inliers=5, template_idx=0),
          PnPResult(success=False, n_inliers=99, template_idx=1),
          PnPResult(success=True, n_inliers=20, template_idx=2),
          PnPResult(success=True, n_inliers=11, template_idx=3)]
    ranked = rank_candidates(rs, strategy="inlier")
    assert [r.template_idx for r in ranked] == [2, 3, 0]   # 失败项被剔除
    assert select_best_candidate(rs).template_idx == 2      # 排序第一即最优


def test_rank_candidates_keep_failed_occupies_topk_slots():
    """keep_failed=True：失败候选按相似度序占住 topK 窗口名额。

    旧代码对求解失败的候选 append inf + dummy 位姿
    （inference_on_LM.py:516-520），失败项照样占掉 top-3/top-5 的名额；
    若先剔除，top3/top5 会系统性偏乐观。窗口顺序按模板相似度降序
    （旧代码 :375），失败候选的相似度同样有定义。
    """
    rs = [PnPResult(success=True, n_inliers=5, template_idx=0,
                    template_score=0.30),
          PnPResult(success=False, n_inliers=0, template_idx=1,
                    template_score=0.90),      # 相似度最高但 PnP 失败
          PnPResult(success=True, n_inliers=20, template_idx=2,
                    template_score=0.50)]
    window = rank_candidates(rs, strategy="similarity", keep_failed=True)
    assert [r.template_idx for r in window] == [1, 2, 0]
    assert [r.success for r in window] == [False, True, True]
    # 主路线择优不受影响：仍只从成功候选里选
    assert [r.template_idx for r in
            rank_candidates(rs, strategy="similarity")] == [2, 0]
    assert select_best_candidate(rs, strategy="similarity").template_idx == 2


def test_rank_candidates_default_still_drops_failed():
    """keep_failed 默认 False：主路线语义不变（失败候选不参与择优）。"""
    rs = [PnPResult(success=False, n_inliers=0, template_idx=0,
                    template_score=1.0),
          PnPResult(success=True, n_inliers=7, template_idx=1,
                    template_score=0.1)]
    assert [r.template_idx for r in rank_candidates(rs)] == [1]
    assert [r.template_idx for r in
            rank_candidates(rs, keep_failed=True)] == [1, 0]


# ---------------------------------------------------------------------------
# 4. PnP flag（旧代码 SOLVEPNP_SQPNP）
# ---------------------------------------------------------------------------
def test_ransac_pnp_sqpnp_recovers_synthetic_pose():
    """sqpnp（旧代码 :448）在合成数据上同样恢复 GT。"""
    rng = np.random.default_rng(11)
    pts3d = rng.uniform(-50, 50, size=(200, 3))
    R = _random_rotation(rng)
    t = np.array([5.0, -10.0, 700.0])
    p_cam = pts3d @ R.T + t
    uv = p_cam @ K_QUERY.T
    uv = uv[:, :2] / uv[:, 2:3]
    res = ransac_pnp(uv, pts3d, K_QUERY, flag="sqpnp",
                     reproj_px=2.0, iterations=400, refine_lm=False)
    assert res.success
    from src.geometry.pose_utils import rotation_angle_deg
    assert rotation_angle_deg(res.R, R) < 0.1
    assert np.linalg.norm(res.t - t) < 1.0


def test_ransac_pnp_unknown_flag_raises():
    with pytest.raises(ValueError, match="pnp_flag"):
        ransac_pnp(np.zeros((10, 2)), np.zeros((10, 3)), K_QUERY,
                   flag="p3p_magic")


# ---------------------------------------------------------------------------
# 5. topK best 同步选择 + 旧 aggregated JSON 格式
# ---------------------------------------------------------------------------
def test_topk_best_pick_sync_selection():
    """同步选择（inference_on_LM.py:526-532）：ADD 最小者的 proj，
    绝不是独立取 proj 最小。"""
    adds = [30.0, 5.0, 12.0]
    projs = [1.0, 9.0, 2.0]      # proj 最小在下标 0，但 ADD 最小在下标 1
    i, a, p = topk_best_pick(adds, projs, k=3)
    assert (i, a, p) == (1, 5.0, 9.0)
    # top1：只看第一个候选
    i, a, p = topk_best_pick(adds, projs, k=1)
    assert (i, a, p) == (0, 30.0, 1.0)
    # 无候选：inf（帧计入分母、判失败）
    assert topk_best_pick([], [], k=5) == (-1, float("inf"), float("inf"))


def test_topk_key_naming_matches_prior_json():
    assert topk_key(1) == "top1"
    assert topk_key(3) == "top3_best"
    assert topk_key(40) == "top40_best"


def test_object_topk_metrics_counts_and_schema():
    # 2 帧：帧1 top1 失败但 top3 内有好候选；帧2 无候选
    cand_adds = [[50.0, 4.0, 8.0], []]
    cand_projs = [[10.0, 3.0, 6.0], []]
    m = object_topk_metrics(cand_adds, cand_projs, add_thresh=10.0,
                            proj_thresh=5.0, ks=(1, 3))
    assert m["total_samples"] == 2
    assert m["top1"]["add_success_count"] == 0
    assert m["top3_best"]["add_success_count"] == 1     # 帧1 选中 add=4.0
    assert m["top3_best"]["proj_success_count"] == 1    # 同候选 proj=3.0 < 5
    assert m["top3_best"]["add_success_rate"] == 50.0
    # schema 与旧 JSON per-object 段一致
    assert set(m["top1"]) == {"add_success_count", "add_success_rate",
                              "avg_add_mm", "proj_success_count",
                              "proj_success_rate", "avg_proj_error_px"}


def test_per_object_and_aggregate_schema():
    frames = [
        {"add": 3.0, "proj": 1.0, "add_01d": 1.0, "proj_5px": 1.0},
        {"add": np.inf, "proj": np.inf, "add_01d": 0.0, "proj_5px": 0.0},
    ]
    po = per_object_from_frames(frames)
    assert po["total_samples"] == 2
    assert po["add_success_count"] == 1
    assert po["add_success_rate"] == 50.0
    assert po["avg_add_mm"] == 3.0            # inf 不入平均（旧代码 :706-710）
    agg = aggregate_all_objects({"ape": po, "cat": po})
    o = agg["overall_metrics"]
    assert o["total_samples"] == 4
    assert o["add_success_rate"] == 50.0
    assert o["total_add_successes"] == 2
    # schema 与旧 aggregated_metrics_all_objects40.json 完全一致
    assert set(o) == {"total_samples", "add_success_rate",
                      "overall_avg_add_mm", "proj_success_rate",
                      "overall_avg_proj_error_px", "total_add_successes",
                      "total_proj_successes"}


@pytest.mark.skipif(not (PRIOR_DIR / "aggregated_metrics_all_objects40.json"
                         ).exists(), reason="旧结果文件不在本机")
def test_aggregate_overall_matches_prior_real_file():
    """口径回归：从旧文件的 per_object 重算 overall，必须与旧文件的
    overall 一致（证明 aggregate_all_objects 与旧代码 :738-768 同口径）。"""
    agg = json.loads((PRIOR_DIR
                      / "aggregated_metrics_all_objects40.json").read_text())
    ours = aggregate_all_objects(agg["per_object_metrics"])["overall_metrics"]
    theirs = agg["overall_metrics"]
    for k in ("total_samples", "total_add_successes", "total_proj_successes"):
        assert ours[k] == theirs[k]
    for k in ("add_success_rate", "overall_avg_add_mm",
              "proj_success_rate", "overall_avg_proj_error_px"):
        assert ours[k] == pytest.approx(theirs[k], rel=1e-9)


@pytest.mark.skipif(
    not ((PRIOR_DIR / "aggregated_metrics_all_objects40.json").exists()
         and (PRIOR_DIR / "aggregated_metrics_top1_top3_top5.json").exists()),
    reason="旧结果文件不在本机（本测试同时读 all_objects40 与 top1_top3_top5）")
def test_prior_import_conversion_real_files():
    """scripts/import_prior_metrics 的转换函数在真实旧文件上产出新库格式。"""
    agg40 = json.loads((PRIOR_DIR
                        / "aggregated_metrics_all_objects40.json").read_text())
    topk = json.loads((PRIOR_DIR
                       / "aggregated_metrics_top1_top3_top5.json").read_text())
    rep = prior_to_report(agg40, source="test", topk_agg=topk)
    assert len(rep["per_object"]) == 13
    assert rep["per_object"]["ape"]["add_01d"] == pytest.approx(51.8095, abs=1e-3)
    assert rep["per_object"]["ape"]["n"] == 1050
    assert rep["overall"]["add_success_rate"] == pytest.approx(82.7254, abs=1e-3)
    assert rep["mean"]["cm_deg"] is None       # 旧管线无 5cm5°，不许编数
    # oracle 标注必须落进 JSON 数据字段（光靠注释挡不住误引用）
    assert rep["protocol"] == "prior_MyPose_oracle_top40"
    assert rep["selection"] == "oracle_gt_add"
    assert rep["is_oracle_upper_bound"] is True
    # 非 oracle 参照从 top1/3/5 文件读出，不是硬编码
    assert rep["non_oracle_reference"]["top1_add_01d"] == pytest.approx(49.49)
    assert rep["non_oracle_reference"]["top1_proj_5px"] == pytest.approx(59.22)
    # 不给 topk_agg 时省略该字段（不编数）
    assert "non_oracle_reference" not in prior_to_report(agg40)

    rep2 = prior_topk_to_report(topk, source="test")
    assert set(rep2["tiers"]) == {"top1", "top3_best", "top5_best"}
    assert rep2["overall"]["top1"]["add_success_rate"] == pytest.approx(49.49)
    assert rep2["tiers"]["top5_best"]["per_object"]["ape"]["add_01d"] \
        == pytest.approx(42.57)
    # 逐档 oracle 标记：只有 top1 是端到端数字
    assert rep2["protocol"] == "prior_MyPose_oracle_topk"
    assert rep2["tiers"]["top1"]["is_oracle"] is False
    assert rep2["tiers"]["top3_best"]["is_oracle"] is True
    assert rep2["tiers"]["top5_best"]["is_oracle"] is True
    assert rep2["non_oracle_reference"]["top1_add_01d"] == pytest.approx(49.49)


def test_topk_from_key_inverts_topk_key():
    """tier 键名 → K 的反解（oracle 标记依赖它区分 top1 与 K>1）。"""
    for k in (1, 3, 5, 40):
        assert topk_from_key(topk_key(k)) == k
    assert topk_from_key("total_samples") == 0   # 非 tier 键 → 0（从严）


def test_non_oracle_reference_omitted_without_top1():
    """输入不含 top1 档 → 返回 None（调用方省略字段，绝不编数）。"""
    assert non_oracle_reference(None) is None
    assert non_oracle_reference({"overall_metrics": {"top5_best": {}}}) is None
    ref = non_oracle_reference({"overall_metrics": {
        "top1": {"add_success_rate": 40.0, "proj_success_rate": 50.0}}})
    assert (ref["top1_add_01d"], ref["top1_proj_5px"]) == (40.0, 50.0)


def test_aggregate_topk_all_objects_schema():
    po = object_topk_metrics([[3.0], [20.0]], [[2.0], [9.0]],
                             add_thresh=10.0, ks=(1,))
    agg = aggregate_topk_all_objects({"ape": po}, ks=(1,))
    o = agg["overall_metrics"]
    assert o["total_samples"] == 2
    # schema 与旧 top1_top3_top5 JSON 的 overall 段一致
    assert set(o["top1"]) == {"add_success_rate", "avg_add_mm",
                              "proj_success_rate", "avg_proj_error_px",
                              "total_add_successes", "total_proj_successes"}


def test_aggregate_topk_averages_unrounded_per_object():
    """overall 的 avg 必须对**未舍入**的逐物体值求平均（旧代码 :743-744）。

    object_topk_metrics 保留全精度，舍入只发生在 aggregate 的落盘输出上。
    用 avg_proj_error_px 构造分歧（旧 JSON 保留 2 位，量化步长足够大）：
    三个物体的真实值 1.0 / 1.0051 / 1.0051 —— 先平均再舍入得 1.0，
    先各自舍入（1.0 / 1.01 / 1.01）再平均得 1.01。
    """
    objs = {}
    for name, proj in (("a", 1.0), ("b", 1.0051), ("c", 1.0051)):
        objs[name] = object_topk_metrics([[1.000051]], [[proj]],
                                         add_thresh=10.0, ks=(1,))
    # 逐物体值全精度保留（不得被提前 round）
    assert objs["b"]["top1"]["avg_proj_error_px"] == pytest.approx(1.0051,
                                                                  abs=1e-12)
    assert objs["b"]["top1"]["avg_add_mm"] == pytest.approx(1.000051,
                                                            abs=1e-12)

    agg = aggregate_topk_all_objects(objs, ks=(1,))
    assert agg["overall_metrics"]["top1"]["avg_proj_error_px"] == 1.0
    # 落盘的 per_object 才按旧 JSON 小数位舍入，且不篡改调用方手上的 dict
    assert agg["per_object_metrics"]["b"]["top1"]["avg_proj_error_px"] == 1.01
    assert objs["b"]["top1"]["avg_proj_error_px"] == pytest.approx(1.0051,
                                                                  abs=1e-12)


# ---------------------------------------------------------------------------
# 6. legacy_mypose.yaml 配置：base 继承 + 逐项复现旧管线
# ---------------------------------------------------------------------------
LEGACY_CFG = str(ROOT / "configs" / "legacy_mypose.yaml")
DEFAULT_CFG = str(ROOT / "configs" / "default.yaml")


def test_legacy_config_reproduces_prior_pipeline():
    cfg = load_config(LEGACY_CFG)
    # 定位：GT coseg mask（旧代码 :281-311 全程用 GT mask，YOLO 框未参与）
    assert cfg["detection"]["segmenter"] == "gt_mask"
    assert cfg["detection"]["crop_mode"] == "tight_square"
    assert cfg["detection"]["crop_size"] == 512
    assert cfg["detection"]["crop_expand"] == 1.1
    # 模板与 2D-3D 提升：深度图 + 反投影
    assert cfg["templates"]["template_source"] == "depth_map"
    assert cfg["matching"]["lifting"] == "depth_backproject"
    # 全模板匹配（无 DINOv2 预筛），保留 40 个候选
    assert cfg["matching"]["template_prescreen"] == "none"
    assert cfg["matching"]["template_ranking"] == "mast3r"
    assert cfg["matching"]["top_k"] == 40
    # 匹配过滤对齐旧行为：严格互最近邻、无相似度阈值
    assert cfg["matching"]["cycle_tau_px"] == 0.0
    assert cfg["matching"]["sim_threshold"] == -1.0
    # PnP 对齐旧参数：SQPNP / ε=2px / 400 迭代 / 无 LM
    assert cfg["solver"]["pnp_flag"] == "sqpnp"
    assert cfg["solver"]["ransac_reproj_px"] == 2.0
    assert cfg["solver"]["ransac_iterations"] == 400
    assert cfg["solver"]["refine_lm"] is False
    # 候选窗口顺序：按模板相似度降序（旧代码 :375），不是新库默认的 inlier
    assert cfg["solver"]["selection"] == "similarity"
    # topK best 消融档位（top1 端到端 + K>1 的 GT 择优上界）
    assert cfg["metrics"]["topk_best"] == [1, 3, 5, 40]
    # base 继承：未覆盖字段取自 default.yaml
    assert cfg["gaussian"]["iterations"] == 7000
    assert cfg["templates"]["n_viewpoints"] == 8   # templates 深合并不丢键
    assert len(cfg["dataset"]["objects"]) == 13


def test_default_config_new_switches_keep_new_behavior():
    """default.yaml 新增开关的默认值必须等于新库原行为（不动主实验）。"""
    cfg = load_config(DEFAULT_CFG)
    assert cfg["templates"]["template_source"] == "coord_map"
    assert cfg["matching"]["lifting"] == "coord_map"
    assert cfg["matching"]["template_prescreen"] == "dinov2"
    assert cfg["detection"]["crop_mode"] == "context_pad"
    assert cfg["solver"]["pnp_flag"] == "epnp"
    assert cfg["metrics"]["topk_best"] == []


def test_deep_merge_does_not_mutate_base():
    """_deep_merge 返回值的任何修改都不得影响传入的 base（含嵌套层）。

    原来的测试是"加载 legacy 后再加载 default，断言 default 干净"——
    `load_config` 每次都重新读文件，任何实现都能过，等于空测。这里直接
    对合并函数本身下手。
    """
    from src.config import _deep_merge
    base = {"detection": {"segmenter": "fastsam", "crop_size": 512},
            "objects": ["ape", "cat"]}
    overlay = {"detection": {"segmenter": "gt_mask"}}
    merged = _deep_merge(base, overlay)
    assert merged["detection"] == {"segmenter": "gt_mask", "crop_size": 512}
    # 改返回值的顶层、嵌套层与列表，base 都不能跟着变
    merged["detection"]["crop_size"] = 999
    merged["detection"]["new_key"] = 1
    merged["objects"] = ["mutated"]
    assert base["detection"] == {"segmenter": "fastsam", "crop_size": 512}
    assert base["objects"] == ["ape", "cat"]
    # 反过来改 base 也不影响已返回的结果
    base["detection"]["segmenter"] = "sam"
    assert merged["detection"]["segmenter"] == "gt_mask"


def test_deep_merge_rejects_none_over_dict_section():
    """overlay 侧空值盖掉 base 侧配置段 → 报错并提示显式写 {}。

    YAML 里写了 `matching:` 却没写内容会解析成 None，静默替换整段之后，
    错误要等到远端 `cfg["matching"].get(...)` 才以 AttributeError 现形。
    """
    from src.config import _deep_merge
    with pytest.raises(ValueError, match=r"matching"):
        _deep_merge({"matching": {"top_k": 40}}, {"matching": None})
    # 显式 {} 是允许的（真的要清空）
    assert _deep_merge({"matching": {"top_k": 40}},
                       {"matching": {}}) == {"matching": {"top_k": 40}}
    # 键路径要带上层级，便于定位
    with pytest.raises(ValueError, match=r"a\.b"):
        _deep_merge({"a": {"b": {"c": 1}}}, {"a": {"b": None}})


def test_load_config_detects_base_cycle(tmp_path):
    """base 链成环 → 带链条的 ValueError（原本是无信息量的 RecursionError）。"""
    (tmp_path / "x.yaml").write_text("base: y.yaml\nfoo: 1\n", encoding="utf-8")
    (tmp_path / "y.yaml").write_text("base: x.yaml\nbar: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="循环引用"):
        load_config(str(tmp_path / "x.yaml"))


def test_base_overlay_does_not_leak_into_default():
    """加载 legacy 后再加载 default，default 不被污染（端到端回归）。"""
    _ = load_config(LEGACY_CFG)
    cfg = load_config(DEFAULT_CFG)
    assert cfg["detection"]["segmenter"] == "fastsam"
    assert "base" not in cfg


# ---------------------------------------------------------------------------
# 7. TemplateBank 深度图加载 + EstimateResult 候选字段
# ---------------------------------------------------------------------------
def test_template_bank_loads_depth_maps(tmp_path):
    from src.pipeline import TemplateBank
    n, s = 2, 16
    bank = {
        "images": np.zeros((n, s, s, 3), dtype=np.uint8),
        "alphas": np.ones((n, s, s), dtype=np.float16),
        "coord_maps": np.zeros((n, s, s, 3), dtype=np.float32),
        "poses": np.stack([np.eye(4)] * n).astype(np.float32),
        "K": K_TMPL.astype(np.float32),
        "scale": np.float32(1.0),
        "dino_feats": np.zeros((n, 8), dtype=np.float32),
        "depth_maps": np.full((n, s, s), 2.5, dtype=np.float32),
    }
    p = tmp_path / "bank.npz"
    np.savez_compressed(p, **bank)
    tb = TemplateBank(p)
    assert tb.depth_maps is not None
    assert tb.depth_maps.shape == (n, s, s)
    # 旧格式 npz（无 depth_maps）加载后为 None，坐标图路线不受影响
    del bank["depth_maps"]
    p2 = tmp_path / "bank2.npz"
    np.savez_compressed(p2, **bank)
    assert TemplateBank(p2).depth_maps is None


def _minimal_bank_dict(n=2, s=16):
    """完整的最小模板库字典（含 scale / dino_feats），供闸门测试改造。"""
    return {
        "images": np.zeros((n, s, s, 3), dtype=np.uint8),
        "alphas": np.ones((n, s, s), dtype=np.float16),
        "coord_maps": np.zeros((n, s, s, 3), dtype=np.float32),
        "poses": np.stack([np.eye(4)] * n).astype(np.float32),
        "K": K_TMPL.astype(np.float32),
        "scale": np.float32(1.7),
        "dino_feats": np.zeros((n, 8), dtype=np.float32),
    }


def test_template_bank_raises_on_missing_scale(tmp_path):
    """缺 scale → raise 并提示重新 onboard。

    原实现静默回退 1.0：onboard 若在模板渲染落盘之后、DINOv2 之前中断
    （OOM 常见），残留文件能被"正常"加载，平移量级系统性错误且无任何报错。
    """
    from src.pipeline import TemplateBank
    d = _minimal_bank_dict()
    del d["scale"]
    p = tmp_path / "no_scale.npz"
    np.savez_compressed(p, **d)
    with pytest.raises(ValueError, match="scale"):
        TemplateBank(p)


def test_template_bank_raises_on_missing_dino_feats(tmp_path):
    """缺 dino_feats → raise（同属 onboard 未跑完的残留文件）。"""
    from src.pipeline import TemplateBank
    d = _minimal_bank_dict()
    del d["dino_feats"]
    p = tmp_path / "no_dino.npz"
    np.savez_compressed(p, **d)
    with pytest.raises(ValueError, match="dino_feats"):
        TemplateBank(p)


def test_template_bank_raises_on_depth_shape_mismatch(tmp_path):
    """depth_maps 前三维（含 M）必须与 images 一致，否则 raise。

    分辨率或模板数不匹配时反投影会取错模板/越界，静默产出错位 3D 锚点。
    """
    from src.pipeline import TemplateBank
    # 分辨率不匹配
    d = _minimal_bank_dict(n=2, s=16)
    d["depth_maps"] = np.full((2, 8, 8), 2.0, dtype=np.float32)
    p = tmp_path / "bad_res.npz"
    np.savez_compressed(p, **d)
    with pytest.raises(ValueError, match="depth_maps"):
        TemplateBank(p)
    # 模板数（M 维）不匹配
    d2 = _minimal_bank_dict(n=2, s=16)
    d2["depth_maps"] = np.full((3, 16, 16), 2.0, dtype=np.float32)
    p2 = tmp_path / "bad_m.npz"
    np.savez_compressed(p2, **d2)
    with pytest.raises(ValueError, match="depth_maps"):
        TemplateBank(p2)


def test_estimate_result_candidates_default_empty():
    from src.pipeline import EstimateResult
    r = EstimateResult(success=False)
    assert r.candidates == []


def _fake_bank(tmp_path, with_depth=False, name="bank.npz"):
    """写一个最小模板库 npz（不含 GPU 依赖），供构造校验测试用。"""
    n, s = 2, 16
    d = _minimal_bank_dict(n=n, s=s)
    d["scale"] = np.float32(1.0)
    if with_depth:
        d["depth_maps"] = np.full((n, s, s), 2.0, dtype=np.float32)
    p = tmp_path / name
    np.savez_compressed(p, **d)
    return p


def test_estimator_raises_when_depth_lifting_without_depth_maps(tmp_path):
    """depth_backproject + 无 depth_maps 的模板库 → 构造期显式报错，
    提示改 templates.template_source 并重新 onboard（不静默退回坐标图）。"""
    from src.config import apply_override
    from src.pipeline import PoseEstimator, TemplateBank
    # 用 gt_mask 定位避开 YOLO/ultralytics；lifting 校验在 matcher 之前
    cfg = apply_override(load_config(LEGACY_CFG), "detection.segmenter",
                         "gt_mask")
    bank = TemplateBank(_fake_bank(tmp_path, with_depth=False))
    with pytest.raises(ValueError, match="depth_maps"):
        PoseEstimator(cfg, bank, device="cpu")


def test_estimator_raises_on_unknown_crop_mode_and_lifting(tmp_path):
    from src.config import apply_override
    from src.pipeline import PoseEstimator, TemplateBank
    base = apply_override(load_config(DEFAULT_CFG), "detection.segmenter",
                          "gt_mask")
    bank = TemplateBank(_fake_bank(tmp_path, with_depth=True))
    cfg1 = apply_override(base, "detection.crop_mode", "magic_crop")
    with pytest.raises(ValueError, match="crop_mode"):
        PoseEstimator(cfg1, bank, device="cpu")
    cfg2 = apply_override(base, "matching.lifting", "magic_lift")
    with pytest.raises(ValueError, match="lifting"):
        PoseEstimator(cfg2, bank, device="cpu")


def test_template_bank_path_separates_depth_bank():
    """深度图模板库与坐标图模板库文件名必须不同（否则静默复用旧库）。"""
    from src.config import apply_override
    from src.pipeline import template_bank_path
    cfg = load_config(DEFAULT_CFG)
    p_coord = template_bank_path(cfg, "ape")
    p_depth = template_bank_path(
        apply_override(cfg, "templates.template_source", "depth_map"), "ape")
    assert p_coord != p_depth
    assert p_depth.name.endswith("_depth.npz")
    # legacy 配置走深度库
    assert template_bank_path(load_config(LEGACY_CFG), "ape") == p_depth


# ---------------------------------------------------------------------------
# 8. 深度模板渲染 ↔ 深度反投影：约定闭环（假 trainer，CPU）
# ---------------------------------------------------------------------------
class _DeltaTrainer:
    """最小假 trainer：把每个"高斯"投影成单像素、alpha=1 的 delta 光栅器。

    只为验证 render_template_bank 的深度/坐标图约定（相机系 z、alpha 归一、
    背景置 0）与 depth_lifting 的反投影互逆，不涉及 gsplat/GPU。

    落点用 `floor(uv)` 而不是 `round(uv)`：gsplat 的像素 j 覆盖连续坐标
    `[j, j+1)`、中心在 `j+0.5`（RasterizeToPixels3DGSSerialBatchFwd.cu:108
    `px = out_x + 0.5f`），所以连续坐标 u 的峰值像素是 `floor(u)`。
    """

    def __init__(self, centers):
        import torch
        self.torch = torch
        self._centers = torch.tensor(np.asarray(centers), dtype=torch.float32)

    def gaussian_centers(self):
        return self._centers

    def render(self, viewmat, K, width, height, colors_override=None,
               sh_degree=None):
        torch = self.torch
        T = torch.tensor(np.asarray(viewmat), dtype=torch.float32)
        Kt = torch.tensor(np.asarray(K), dtype=torch.float32)
        p_cam = self._centers @ T[:3, :3].T + T[:3, 3]
        uvw = p_cam @ Kt.T
        uv = uvw[:, :2] / uvw[:, 2:3]
        cols = (colors_override if colors_override is not None
                else torch.full((len(self._centers), 3), 0.5))
        img = torch.zeros(height, width, cols.shape[1])
        alpha = torch.zeros(height, width, 1)
        for i in range(len(uv)):
            # gsplat 约定：连续坐标 u 落在像素 floor(u)
            x = int(np.floor(float(uv[i, 0])))
            y = int(np.floor(float(uv[i, 1])))
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = cols[i]
                alpha[y, x, 0] = 1.0
        return img, alpha, {}


def test_rendered_depth_backprojects_to_gaussian_centers(tmp_path):
    """渲染深度图 → depth_lifting 反投影，必须还原高斯中心（约定闭环）。

    这是深度反投影路线的关键回归：若渲染侧的 z 定义或反投影侧的 K_inv /
    位姿逆变换顺序有一处写反，本测试立即失败。
    """
    from src.gaussian.template_renderer import render_template_bank
    rng = np.random.default_rng(5)
    centers = rng.uniform(-20, 20, size=(60, 3))
    trainer = _DeltaTrainer(centers)
    cfg_tpl = {"image_size": 128, "fov_deg": 40.0, "radius_scale": 2.5,
               "viewpoint_mode": "cube8", "n_viewpoints": 8, "n_inplane": 5,
               "template_source": "depth_map"}
    bank = render_template_bank(trainer, cfg_tpl, tmp_path / "b.npz",
                                bg_color=1.0)
    assert "depth_maps" in bank
    assert bank["depth_maps"].shape == (40, 128, 128)

    # 取第一个模板：所有前景像素反投影，必须落在原始高斯中心集合上
    depth = bank["depth_maps"][0]
    T = bank["poses"][0].astype(np.float64)
    ys, xs = np.nonzero(depth > 0)
    assert len(xs) > 10                      # delta 光栅器下大部分点可见
    pix = np.stack([xs, ys], axis=1)
    pts3d, valid = backproject_depth_to_model(pix, depth, bank["K"], T)
    assert valid.all()
    # 逐点找最近的原始中心；深度值本身精确（delta 光栅无混合），偏差只来自
    # 像素量化。半像素约定对齐后每轴误差 ≤0.5px（内参主点已含 -0.5，渲染侧
    # 过 to_pixel_center_intrinsics 加回 0.5），两轴合成 √2×0.5·d_max/f。
    # 这就是半像素约定的回归断言：若内参退回 cx=S/2（整数 uv 差半格），
    # 每轴误差退化到 ≤1px，本断言立即失败。
    d = np.linalg.norm(pts3d[:, None, :] - centers[None, :, :], axis=2)
    nn = d.min(axis=1)
    tol = (np.sqrt(2.0) * 0.5 * float(depth[ys, xs].max())
           / float(bank["K"][0, 0]))
    assert nn.max() < tol
    # 坐标图路线与深度路线在同一像素上应给出同一 3D 点（两条路线一致性）
    cm = bank["coord_maps"][0]
    np.testing.assert_allclose(cm[ys, xs], pts3d, atol=tol)
