"""查询裁剪超分（任务 2：M 类病信息预算修复）。

mode 1 = bicubic ×2（零成本基线；若此臂已有大部分收益，说明瓶颈纯是
分辨率，无需上 SR 模型）；mode 2 = Real-ESRGAN ×2（预留，权重从
GitHub release 获取，仓库网络受限时不可用，启用时自动提示）。

缩放因子统一并入 s_leg 链（pipeline 调用方处理），坐标映射不变。
"""
from __future__ import annotations

import cv2
import numpy as np

_SCALE = 2


def upscale_crop(crop: np.ndarray, mask: np.ndarray, mode: int = 1,
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """把 (crop, mask) 超分 _SCALE 倍，返回 (crop_sr, mask_sr, scale)。"""
    if mode == 1:                       # bicubic 基线
        crop_sr = cv2.resize(crop, None, fx=_SCALE, fy=_SCALE,
                             interpolation=cv2.INTER_CUBIC)
    elif mode == 2:                     # Real-ESRGAN（权重已从 HF 镜像获取）
        import sys
        import torchvision.transforms as T
        # realesrgan 0.3.0 引用 torchvision.transforms.functional_tensor
        # （torchvision>=0.17 移除），注入兼容 shim
        if not hasattr(T, "functional_tensor"):
            import types
            import torchvision.transforms.functional as F
            ft = types.ModuleType("torchvision.transforms.functional_tensor")
            ft.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = ft
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as e:
            raise ImportError(
                "crop_sr=2 需要 realesrgan + basicsr 包\n"
                "  pip install realesrgan basicsr\n"
                f"原始错误: {e}") from e
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path="weights/RealESRGAN_x4plus.pth",
                                 model=model, tile=256, tile_pad=10,
                                 pre_pad=0, half=False)
        crop_sr, _ = upsampler.enhance(crop, outscale=_SCALE)
        crop_sr = np.clip(crop_sr, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"未知 crop_sr 档位: {mode}（1=bicubic, 2=esrgan）")
    mask_sr = cv2.resize(mask.astype(np.uint8), (crop_sr.shape[1], crop_sr.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return crop_sr, mask_sr, float(_SCALE)
