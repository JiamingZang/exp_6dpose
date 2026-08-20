"""6d-fill-norm 单测：查询裁剪填充归一化的坐标链一致性。

fill-norm 把裁剪内容缩放 α，s_leg 链乘 α 让主 PnP 反变换自动吸收；
下游裁剪系内参 K_crop 前两行乘 α（_apply_fill_scale）。这里用合成
像素验证两条链的数学闭环，防止改坏坐标变换。
"""
import numpy as np

from src.matching.correspondence import back_to_original_pixels
from src.pipeline import PoseEstimator


def test_back_to_original_pixels_with_fill_scale():
    """匹配区坐标 → 原图：除以 (sx·s_leg·α) 再平移，s_leg 吸收 α 后成立。"""
    crop_box = (30, 40, 230, 240)          # 原图裁剪区
    x0, y0 = crop_box[0], crop_box[1]
    sx = sy = 1.25                          # matcher resize 因子
    s_leg = 2.0                             # sr 链缩放
    alpha = 0.57                            # fill-norm 缩放
    orig = np.array([[100.0, 120.0], [150.0, 90.0]])
    # 原图 → 裁剪区 → 匹配区（含 fill-norm 的复合缩放）
    crop_px = orig - np.array([x0, y0])
    match_px = crop_px * (sx * s_leg * alpha)
    back = back_to_original_pixels(match_px, (sx * s_leg * alpha,
                                              sy * s_leg * alpha), crop_box)
    assert np.allclose(back, orig)


def test_fill_scale_identity_when_off():
    """fill_scale=1.0（未启用/α=1）时 K_crop 原样返回。"""
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    ex = {"fill_scale": 1.0}
    out = PoseEstimator._apply_fill_scale(None, ex, K)
    assert out is K


def test_fill_scale_scales_only_first_two_rows():
    """α 缩放：fx/fy/cx/cy 同乘 α，第三行 [0,0,1] 不动。"""
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    alpha = 0.57
    ex = {"fill_scale": alpha}
    out = PoseEstimator._apply_fill_scale(None, ex, K)
    assert np.allclose(out, [[525 * alpha, 0, 319.5 * alpha],
                             [0, 525 * alpha, 239.5 * alpha],
                             [0, 0, 1.0]])


def test_fill_scale_missing_key_defaults_to_one():
    """alt crop 等没有 fill_scale 键的 ex 默认不缩放。"""
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    out = PoseEstimator._apply_fill_scale(None, {}, K)
    assert out is K


def test_fill_scale_projection_equivariance():
    """投影等价：α·K 与 α 缩放的像素坐标给出同一归一化坐标。"""
    K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    alpha = 0.57
    P = np.array([0.1, -0.05, 2.0])          # 相机系 3D 点
    # 原裁剪系像素
    uv = K @ P
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    # α 缩放裁剪系：像素坐标乘 α，内参乘 α → 归一化坐标不变
    Ka = PoseEstimator._apply_fill_scale(None, {"fill_scale": alpha}, K)
    uv2 = Ka @ P
    u2, v2 = uv2[0] / uv2[2], uv2[1] / uv2[2]
    assert np.allclose([u * alpha, v * alpha], [u2, v2])
