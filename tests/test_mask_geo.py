"""掩码几何平移候选（6d-mask-geo）纯逻辑测试：GS-Pose §3.2 面积比+质心解析。"""
import numpy as np

from src.pipeline import PoseEstimator


def _estimator_with_bank(bank):
    est = object.__new__(PoseEstimator)
    est.bank = bank
    return est


def _bank(alphas=None, poses=None):
    class B:
        pass
    b = B()
    b.alphas = np.ones((1, 8, 8), dtype=np.float32) if alphas is None else alphas
    if poses is None:
        poses = np.eye(4)[None].astype(np.float32)
        poses[0, :3, 3] = [0, 0, 1]  # z_ref=1（真实 bank 为球面位姿，范数=radius）
    b.poses = poses
    b.dino_feats = np.zeros((1, 1), dtype=np.float32)
    b.images = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    b.coord_maps = np.zeros((1, 8, 8, 3), dtype=np.float32)
    b.K = np.eye(3)
    b.scale = 1.0
    return b


def _ex(mask, x0=0, y0=0):
    return {"mask_crop": mask.astype(bool), "crop_box_used": (x0, y0, 8, 8)}


def test_mask_geo_translation_recovers_center_depth():
    # 8x8 图，f=2 主点 (4,4)，掩码中心 4x4（A_q=16），模板全 1（A_ref=64），z_ref=1
    # z_q = 1*sqrt(16/64) = 0.5；质心由掩码实际坐标算 → t = K_inv @ [cx*z, cy*z, z]
    bank = _bank()
    K = np.array([[2.0, 0, 4.0], [0, 2.0, 4.0], [0, 0, 1.0]])
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    est = _estimator_with_bank(bank)
    t = est._mask_geo_translation(_ex(mask), K, 0)
    assert t is not None
    my, mx = np.nonzero(mask)
    cx, cy = mx.mean(), my.mean()
    K_inv = np.linalg.inv(K)
    np.testing.assert_allclose(t, K_inv @ [cx * 0.5, cy * 0.5, 0.5],
                               atol=1e-6)


def test_mask_geo_translation_crop_offset_uses_full_image_center():
    # 裁剪偏移 (x0, y0) 后质心应回到全图系
    bank = _bank()
    K = np.array([[2.0, 0, 4.0], [0, 2.0, 4.0], [0, 0, 1.0]])
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    est = _estimator_with_bank(bank)
    t_off = est._mask_geo_translation(_ex(mask, x0=10, y0=20), K, 0)
    assert t_off is not None
    my, mx = np.nonzero(mask)
    cx, cy = mx.mean() + 10, my.mean() + 20
    K_inv = np.linalg.inv(K)
    np.testing.assert_allclose(t_off,
                               K_inv @ [cx * 0.5, cy * 0.5, 0.5],
                               atol=1e-6)


def test_mask_geo_translation_scale_ratio():
    # A_q = A_ref → z_q = z_ref
    bank = _bank()
    K = np.array([[2.0, 0, 4.0], [0, 2.0, 4.0], [0, 0, 1.0]])
    mask = np.ones((8, 8), dtype=bool)
    est = _estimator_with_bank(bank)
    t = est._mask_geo_translation(_ex(mask), K, 0)
    np.testing.assert_allclose(t[2], 1.0, atol=1e-6)


def test_mask_geo_translation_invalid_inputs_return_none():
    bank = _bank()
    K = np.array([[2.0, 0, 4.0], [0, 2.0, 4.0], [0, 0, 1.0]])
    est = _estimator_with_bank(bank)
    assert est._mask_geo_translation(_ex(np.zeros((8, 8), dtype=bool)), K, 0) is None
    assert est._mask_geo_translation(_ex(np.ones((8, 8), dtype=bool)), K, -1) is None
    assert est._mask_geo_translation(_ex(np.ones((8, 8), dtype=bool)), K, 5) is None


def test_mask_geo_translation_uses_pose_norm_as_z_ref():
    # z_ref 取 poses 平移范数：t=(0,0,2) → z_ref=2 → z_q=2*sqrt(16/64)=1
    poses = np.eye(4)[None].astype(np.float32)
    poses[0, :3, 3] = [0, 0, 2]
    bank = _bank(poses=poses)
    K = np.array([[2.0, 0, 4.0], [0, 2.0, 4.0], [0, 0, 1.0]])
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    est = _estimator_with_bank(bank)
    t = est._mask_geo_translation(_ex(mask), K, 0)
    np.testing.assert_allclose(t[2], 1.0, atol=1e-6)
