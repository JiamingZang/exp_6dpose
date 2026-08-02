"""可微 SSIM（3DGS 训练损失的 λ·(1-SSIM) 项）。

标准 11×11 高斯窗 SSIM，实现与官方 3DGS 的 utils/loss_utils.py 等价，
但不依赖 fused-ssim。纯 torch，可在 CPU 上跑（供潜在的本地调试），
训练时在 GPU 上使用。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) \
        - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM(img1, img2)，输入 (B,C,H,W)，值域 [0,1]，返回标量均值。"""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channels,
                              img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad,
                         groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad,
                         groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad,
                       groups=channels) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
