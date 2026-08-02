"""GPU-only 组件的本地导入行为：模块可 import、构造时给出清晰的部署提示。

这些测试保证在 macOS（无 CUDA）上：
1. 任何 `import src.xxx` 都不会因缺 gsplat/mast3r/segment_anything 而炸；
2. 试图实例化 GPU 组件时抛 ImportError/NotImplementedError，
   且错误消息包含可执行的安装指引（而非裸 ModuleNotFoundError）。
"""
import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("mod", [
    "src.pipeline",
    "src.gaussian.gs_trainer",
    "src.gaussian.template_renderer",
    "src.detection.localize",
    "src.matching.mast3r_wrapper",
    "src.matching.alt_matchers",
    "src.datasets.vggt_recon",
])
def test_modules_importable_without_gpu(mod):
    """import 阶段不触碰 GPU 依赖（懒加载约定）。"""
    importlib.import_module(mod)


def test_gaussian_trainer_import_hint():
    import torch
    
    # 【新增】如果环境正常（有GPU且有gsplat），则跳过此测试
    if torch.cuda.is_available():
        try:
            import gsplat
            pytest.skip("Environment is valid (GPU+gsplat found), skipping guard check.")
        except ImportError:
            pass
            
    from src.gaussian.gs_trainer import GaussianTrainer
    pts = np.zeros((10, 3))
    with pytest.raises(ImportError, match="gsplat|GPU"):
        GaussianTrainer(pts, None, cfg={}, device="cuda")


def test_mast3r_matcher_import_hint():
    from src.matching.mast3r_wrapper import Mast3rMatcher
    with pytest.raises(Exception, match="tried to load.*from huggingface.*failed"):
        Mast3rMatcher({"mast3r_checkpoint": "x"}, device="cuda")


def test_sam_localizer_import_hint(monkeypatch):
    """缺 segment_anything 时抛 ImportError 且消息含部署提示。

    P2-4 复审修：原版测试直接 `segmenter='sam'` + 假 checkpoint 路径，
    本机装了 segment-anything 后会先命中 `FileNotFoundError` 而不是
    ImportError。这里用 `sys.modules[...] = None` 强制该 import 失败
    （Python 把 None 视为已知缺失，后续 `import segment_anything` 直接
    raise ImportError），使得无论本机装没装 segment-anything 都稳定
    命中 ImportError 分支，恰好检验错误消息的可执行部署提示。
    """
    import sys
    from src.detection.localize import SamDinoLocalizer
    monkeypatch.setitem(sys.modules, "segment_anything", None)
    with pytest.raises(ImportError, match="SAM|segment|GPU"):
        SamDinoLocalizer({"sam_checkpoint": "x"}, device="cpu",
                         segmenter="sam")


def test_sam_localizer_unknown_segmenter_raises():
    """未知 segmenter 值抛 ValueError（P2-4 复审：与 ImportError 分家，
    分别测『缺依赖』和『配置错』两条独立错误路径）。"""
    from src.detection.localize import SamDinoLocalizer
    with pytest.raises(ValueError, match="segmenter"):
        SamDinoLocalizer({"sam_checkpoint": "x"}, device="cpu",
                         segmenter="unknown_value")


def test_vggt_import_hint():
    from src.datasets.vggt_recon import reconstruct_with_vggt
    with pytest.raises(ImportError, match="vggt|VGGT|GPU"):
        reconstruct_with_vggt(["a.png", "b.png", "c.png"])


def test_loftr_todo_interface():
    from src.matching.alt_matchers import LoFTRMatcher
    with pytest.raises(NotImplementedError, match="LoFTR"):
        LoFTRMatcher({})


def test_ssim_cpu_sanity():
    """SSIM 实现可在 CPU 上验证：同图=1，噪声图<1（3DGS 损失项正确性）。"""
    import torch
    from src.gaussian.ssim import ssim
    g = torch.rand(1, 3, 32, 32)
    assert float(ssim(g, g)) == pytest.approx(1.0, abs=1e-5)
    noisy = torch.clamp(g + 0.3 * torch.randn_like(g), 0, 1)
    assert float(ssim(g, noisy)) < 0.95
