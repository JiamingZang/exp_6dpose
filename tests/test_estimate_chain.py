"""PoseEstimator.estimate 坐标反变换链条闭环单测（P1-2 复审）。

覆盖 pipeline.py 里三条串起来的变换，先前无单测：
1) `back_to_original_pixels(pix_q, (sx,sy), crop_box)`：MASt3R 匹配像素
   → 裁剪区 → 原图像素（`src/matching/correspondence.py`）。
2) `t_model = best.t / bank.scale`：3D 锚点在 s 倍缩放的物体系中，投影
   对整体缩放不变，故 (R̂, t̂) 拟合 s·X 等价于 (R̂, t̂/s) 拟合 X
   （`src/pipeline.py` 与 `src/geometry/scale_align.py`）。
3) `transform_pose_by_similarity(R̂, t_model, s_a, R_a, t_a)`：VGGT 重建
   系位姿 → CAD 系位姿（`src/geometry/alignment.py`）。

思路：合成 GT 位姿 → 手工投影得 pix_q（含 resize + crop）→ 反变换 →
PnP，恢复 GT。链条 (1)+(2) 用 CAD 路线闭环、(1)+(2)+(3) 用 VGGT 路线
闭环。所有断言用相对/绝对阈值都留 1% 以上余量。
"""
import numpy as np
import pytest

from src.geometry.alignment import transform_pose_by_similarity
from src.geometry.pose_utils import project_points, rotation_angle_deg
from src.matching.correspondence import back_to_original_pixels
from src.solver.ransac_pnp import ransac_pnp

K = np.array([[572.4114, 0, 325.2611],
              [0, 573.5704, 242.0490],
              [0, 0, 1.0]])


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _synth_cad_scene(seed, s_bank=1.0, sxy=(1.0, 1.0), crop_xy=(0, 0)):
    """合成 CAD 路线场景：GT 位姿投影到原图 → 前向变换到匹配区。

    Returns pts3d_cad (物体原始 mm), pts3d_bank (bank 存的 s·X 坐标),
    pix_matched (MASt3R 输出坐标系), R_gt, t_gt。
    """
    rng = np.random.default_rng(seed)
    pts3d = rng.uniform(-50, 50, size=(400, 3))       # ±5cm 物体（mm）
    R_gt = _random_rotation(rng)
    t_gt = np.array([12.0, -30.0, 650.0])
    uv_orig = project_points(pts3d, K, R_gt, t_gt)    # 原图像素
    # 前向：原图 → 裁剪 → resize
    sx, sy = sxy
    x0, y0 = crop_xy
    uv_crop = uv_orig - np.array([x0, y0])
    uv_matched = uv_crop * np.array([sx, sy])
    # bank 存 s·X（3D 锚点在缩放物体系中）
    pts3d_bank = pts3d * s_bank
    return pts3d, pts3d_bank, uv_matched, R_gt, t_gt


def test_pipeline_pixel_backtransform_closes_pnp_loop_no_scale():
    """(1) 单独：resize/crop 反变换 + PnP → 回到 GT，bank.scale=1 无缩放。"""
    pts3d, pts3d_bank, uv_matched, R_gt, t_gt = _synth_cad_scene(
        seed=42, s_bank=1.0, sxy=(1.3, 1.6), crop_xy=(120, 80))
    # 走 pipeline 的反变换
    pix_orig = back_to_original_pixels(uv_matched, (1.3, 1.6), (120, 80, 0, 0))
    res = ransac_pnp(pix_orig, pts3d_bank, K)
    assert res.success
    # bank.scale=1 时 t_model = t / 1 = t
    t_model = res.t / 1.0
    assert rotation_angle_deg(res.R, R_gt) < 0.5
    assert np.linalg.norm(t_model - t_gt) < 2.0     # <2mm


def test_pipeline_scale_bank_recovers_original_translation():
    """(2) `t_model = t̂ / bank.scale` 换回原始模型单位。

    3D 锚点在 s 倍缩放的物体系中；投影对整体缩放不变，因此 (R̂, t̂) 拟合
    s·X 等价于 (R̂, t̂/s) 拟合 X。合成 s=1.5 的 bank，PnP
    直接拟合 s·X，反缩放后应恰好回到 GT 平移。
    """
    S_BANK = 1.5
    pts3d, pts3d_bank, uv_matched, R_gt, t_gt = _synth_cad_scene(
        seed=7, s_bank=S_BANK, sxy=(1.0, 1.0), crop_xy=(0, 0))
    # 关键：MASt3R 匹配区 == 原图（sxy=1, crop=0）
    pix_orig = back_to_original_pixels(uv_matched, (1.0, 1.0), (0, 0, 0, 0))
    # 但等等 —— GT 平移是 t_gt，然而拟合的是 s·X，pipeline 的做法是
    # PnP 解出 (R̂, t̂) 使 s·X 投影到 uv；应有 t̂ = s·t_gt。用 s·pts3d 与
    # 反缩放后的 t_model = t̂/s 来对照 t_gt。
    res = ransac_pnp(pix_orig, pts3d_bank, K)
    assert res.success
    assert rotation_angle_deg(res.R, R_gt) < 0.5
    # 未反缩放：t̂ 应约等 s·t_gt
    assert np.linalg.norm(res.t - S_BANK * t_gt) < 2.0
    # 反缩放后：与 GT 一致
    t_model = res.t / S_BANK
    assert np.linalg.norm(t_model - t_gt) < 2.0


def test_pipeline_full_chain_recovers_gt_cad_path():
    """(1)+(2)：resize/crop 反变换 + bank.scale 反缩放 → 恢复 GT，
    CAD 路线（无 VGGT→CAD 相似变换），合成含裁剪 + resize + s_bank>1。"""
    S_BANK = 1.4
    pts3d, pts3d_bank, uv_matched, R_gt, t_gt = _synth_cad_scene(
        seed=99, s_bank=S_BANK, sxy=(0.75, 0.75), crop_xy=(200, 150))
    pix_orig = back_to_original_pixels(uv_matched, (0.75, 0.75),
                                       (200, 150, 0, 0))
    res = ransac_pnp(pix_orig, pts3d_bank, K)
    assert res.success
    t_model = res.t / S_BANK
    assert rotation_angle_deg(res.R, R_gt) < 0.5
    assert np.linalg.norm(t_model - t_gt) < 2.0


def test_pipeline_full_chain_recovers_gt_vggt_path():
    """(1)+(2)+(3)：加上 VGGT 重建系 → CAD 系相似变换。

    合成路径：recon 系 3D 点 Y（bank 存 s_bank·Y）→ MASt3R 输出匹配区
    像素 → pipeline 反变换 → PnP 得 (R̂, t̂)，t̂ 位于 s_bank·Y 系 → t_model
    = t̂/s_bank 是 recon 系相机点 → transform_pose_by_similarity 映到 CAD
    系与 GT 比对。

    构造：先随机 (s_a, R_a, t_a) 相似变换，令 CAD 点 X = s_a·R_a·Y + t_a。
    我们知道 GT 是 CAD 系下 (R_cad, t_cad) 使得
    R_cad·X + t_cad = 相机点。据 transform_pose_by_similarity 的推导：
    R_cad·X + t_cad = s_a·(R_pose·Y + t_pose)，因此在 recon 系下的位姿
    是 (R_pose, t_pose) 满足此等式；我们从 (R_cad, t_cad) 反推
    (R_pose, t_pose) 作合成 GT。
    """
    rng = np.random.default_rng(2024)
    Y = rng.uniform(-30, 30, size=(400, 3))        # recon 系模型点
    R_a = _random_rotation(rng)
    s_a = 1.7
    t_a = np.array([1.5, -2.0, 3.5])               # CAD 系下相似变换

    R_cad_gt = _random_rotation(rng)
    t_cad_gt = np.array([8.0, -20.0, 700.0])

    # 推 (R_pose, t_pose)：设相机点 p = R_cad·X + t_cad = s_a·(R_pose·Y + t_pose)
    # 展开 X = s_a·R_a·Y + t_a，两侧代入：
    #   R_cad·s_a·R_a·Y + (R_cad·t_a + t_cad) = s_a·R_pose·Y + s_a·t_pose
    # 逐项对应：R_pose = R_cad·R_a，t_pose = (R_cad·t_a + t_cad)/s_a
    R_pose = R_cad_gt @ R_a
    t_pose = (R_cad_gt @ t_a + t_cad_gt) / s_a

    # bank 存 s_bank·Y
    S_BANK = 1.25
    pts3d_bank = S_BANK * Y
    # 相机点（CAD 系视角）→ 原图像素
    p_cam = (R_cad_gt @ (s_a * (R_a @ Y.T).T + t_a).T).T + t_cad_gt
    uv_orig = (p_cam @ K.T)[:, :2] / (p_cam @ K.T)[:, 2:3]
    # 前向：原图 → 裁剪 → resize
    sxy = (0.9, 1.1)
    crop = (180, 120)
    uv_matched = (uv_orig - np.array(crop)) * np.array(sxy)

    # 走完整 pipeline 反链
    pix_orig = back_to_original_pixels(uv_matched, sxy, crop + (0, 0))
    res = ransac_pnp(pix_orig, pts3d_bank, K)
    assert res.success
    t_model = res.t / S_BANK
    # PnP 拟合 (R_pose, t_pose)（recon 系相机点）
    assert rotation_angle_deg(res.R, R_pose) < 0.5
    assert np.linalg.norm(t_model - t_pose) < 3.0
    # 走 pipeline 的 transform_pose_by_similarity 映回 CAD 系
    R_out, t_out = transform_pose_by_similarity(res.R, t_model, s_a, R_a, t_a)
    assert rotation_angle_deg(R_out, R_cad_gt) < 0.5
    assert np.linalg.norm(t_out - t_cad_gt) < 5.0
