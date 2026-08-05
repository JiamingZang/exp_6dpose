"""用官方 render_template_bank 渲染 coord 锚点 bank，对照手写脚本。

用法: python scripts/experiments/render_coord_bank_official.py --obj can
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class FakeTrainer:
    """只实现 render_template_bank 需要的接口（render/gaussian_centers）。"""

    def __init__(self, ck):
        import gsplat
        self.torch = torch
        self.gsplat = gsplat
        self.device = "cuda"
        self.sh_degree = int(ck["sh_degree"])
        self.splats = {k: v.detach().to("cuda")
                       for k, v in ck["splats"].items()}

    def render(self, viewmat, K, width, height, colors_override=None,
               sh_degree=None):
        viewmats = torch.tensor(viewmat, dtype=torch.float32,
                                device=self.device)[None]
        Ks = torch.tensor(K, dtype=torch.float32, device=self.device)[None]
        if colors_override is None:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]],
                               dim=1)
            sh_degree = self.sh_degree
        else:
            colors = colors_override
            sh_degree = None
        renders, alphas, meta = self.gsplat.rasterization(
            means=self.splats["means"], quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors, viewmats=viewmats, Ks=Ks,
            width=width, height=height, sh_degree=sh_degree, packed=False)
        return renders[0], alphas[0], meta

    def gaussian_centers(self):
        return self.splats["means"].detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="can")
    args = ap.parse_args()

    from src.config import load_config
    from src.gaussian.template_renderer import render_template_bank

    cfg = load_config("configs/archive/dense80_dc_b4.yaml")
    ck = torch.load(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.pt",
                    map_location="cuda", weights_only=False)
    trainer = FakeTrainer(ck)
    out = f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.npz.official_coord"
    bank = render_template_bank(
        trainer, cfg["templates"], out,
        bg_color=float(cfg["onboard"].get("bg_color", 1.0)),
        anchor_mode="coord")
    old = np.load(f"outputs/templates/{args.obj}_3dgs_cad_80t_sa.npz",
                  allow_pickle=True)
    bank["scale"] = old["scale"]
    bank["bg_color"] = np.float32(cfg["onboard"].get("bg_color", 1.0))
    np.savez_compressed(out, **bank)
    print(f"{args.obj}: official coord bank -> {out}")


if __name__ == "__main__":
    main()
