"""3DGS 训练（gsplat 实现）。GPU-only。

以尺度对齐后的点云 P_aligned 初始化高斯集合 G = {μ, r, s, α, h}：
- 损失 L = (1-λ)·L1 + λ·(1-SSIM)，λ=0.2
- 7000 迭代（孤立物体场景简单，无需标准 3DGS 的 30000）
- 自适应密度控制：500-5000 迭代间每 100 次 clone/prune
  （gsplat DefaultStrategy 即 Kerbl et al. 2023 的官方策略实现）

本模块导入 gsplat 失败时抛出带部署提示的 ImportError——3DGS 训练只能在
Linux GPU 机器上运行，本地 macOS 仅做接口静态检查。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_GS_HINT = (
    "gsplat 未安装或无 CUDA。3DGS 训练/渲染只能在 Linux GPU 机器上运行：\n"
    "  pip install gsplat  # 需要 CUDA toolkit，见 setup_gpu.sh\n"
    "本地 macOS 请只运行 tests/ 下的 CPU 单测。"
)


def _import_gs():
    try:
        import torch
        import gsplat
        if not torch.cuda.is_available():
            raise ImportError("torch.cuda.is_available() == False")
        return torch, gsplat
    except ImportError as e:
        raise ImportError(f"{_GS_HINT}\n原始错误: {e}") from e


def _rgb_to_sh0(rgb):
    """RGB → 0 阶球谐系数（gsplat/3DGS 惯例：c = (rgb-0.5)/C0）。"""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _knn_mean_dist(pts: np.ndarray, k: int = 3) -> np.ndarray:
    """每点 3 近邻距离的 RMS，用于初始化高斯尺度。

    对齐 gsplat 官方初始化（examples/simple_trainer.py:321-323）：
    dist_avg = sqrt(mean(d_knn^2))，而非普通平均距离。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)     # 第 0 列是自身
    return np.sqrt((d[:, 1:] ** 2).mean(axis=1))


class GaussianTrainer:
    """物体级 3DGS 训练器。

    Args:
        points: (N,3) 尺度对齐后的初始化点云 P_aligned
        colors: (N,3) [0,1] 初始颜色（CAD 顶点色；无色时置 0.5 灰）
        cfg:    configs/current/default.yaml 的 gaussian 段
    """

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray],
                 cfg: Dict, device: str = "cuda"):
        torch, gsplat = _import_gs()
        self.torch, self.gsplat = torch, gsplat
        self.cfg = cfg
        self.device = device

        n = len(points)
        if colors is None:
            colors = np.full((n, 3), 0.5)
        # scene_scale：官方以相机分布半径为场景尺度（simple_trainer.py:458，
        # parser.scene_scale*1.1），参与 means lr 缩放与 DefaultStrategy 的
        # grow/prune 3D 尺度阈值归一。孤立物体没有相机先验，这里用点云
        # 包围半径 ×1.1 作为等价物（物体级训练的自然场景尺度）。
        center = points.mean(axis=0)
        self.scene_scale = 1.1 * float(
            np.linalg.norm(points - center, axis=1).max())
        means = torch.tensor(points, dtype=torch.float32, device=device)
        # 尺度用近邻距离的 log 初始化，避免初始高斯过大互相遮盖
        scales = torch.tensor(
            np.log(np.clip(_knn_mean_dist(points), 1e-7, None))[:, None]
            .repeat(3, axis=1), dtype=torch.float32, device=device)
        # 官方用随机四元数初始化（simple_trainer.py:331）；这里取单位四元数
        # （wxyz，rasterization 不要求归一化，见 gsplat/rendering.py:400），
        # 保证初始化确定性
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1.0
        opacities = torch.logit(torch.full((n,), 0.1, device=device))
        sh_degree = int(cfg.get("sh_degree", 3))
        n_sh = (sh_degree + 1) ** 2
        sh0 = torch.tensor(_rgb_to_sh0(colors), dtype=torch.float32,
                           device=device)[:, None, :]          # (N,1,3)
        shN = torch.zeros(n, n_sh - 1, 3, device=device)

        self.splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        })
        self.sh_degree = sh_degree

        # 学习率与 GSPose gaussian_object/arguments.py:81-89 一致；
        # means lr 按场景尺度缩放（gsplat examples/simple_trainer.py:336：
        # means_lr * scene_scale，等价 3DGS 官方 position_lr * spatial_lr_scale）
        lr = {
            "means": float(cfg.get("lr_means", 1.6e-4)) * self.scene_scale,
            "scales": float(cfg.get("lr_scales", 5e-3)),
            "opacities": float(cfg.get("lr_opacities", 5e-2)),
            "quats": 1e-3,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }
        self.optimizers = {
            name: torch.optim.Adam([p], lr=lr[name], eps=1e-15)
            for name, p in self.splats.items()
        }

        # 自适应密度控制：gsplat 官方 DefaultStrategy
        from gsplat.strategy import DefaultStrategy
        self.strategy = DefaultStrategy(
            grow_grad2d=float(cfg.get("densify_grad_thresh", 2e-4)),
            refine_start_iter=int(cfg.get("densify_start", 500)),
            refine_stop_iter=int(cfg.get("densify_end", 5000)),
            refine_every=int(cfg.get("densify_interval", 100)),
            verbose=False,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        # scene_scale 进入 strategy 状态（simple_trainer.py:511-512），
        # 决定 grow_scale3d/prune_scale3d 阈值的物理归一
        self.strategy_state = self.strategy.initialize_state(
            scene_scale=self.scene_scale)

    # ------------------------------------------------------------------
    def render(self, viewmat: np.ndarray, K: np.ndarray,
               width: int, height: int, colors_override=None,
               sh_degree=None):
        """单视角光栅化。colors_override 用于 3D 坐标图渲染（见 renderer）。

        gsplat.rasterization 契约（gsplat/rendering.py:234-290,313-321）：
        - sh_degree=None 时 colors 为逐高斯特征 [N,D]（D 任意，D≤32 走单
          chunk），坐标图渲染用 (N,3) 的 μ 即此路径；
        - sh_degree 给定时 colors 为 SH 系数 [N,K,3]，K≥(sh_degree+1)²；
        - 返回 renders [C,H,W,X]、alphas [C,H,W,1]、meta（含 means2d/radii/
          width/height/n_cameras/gaussian_ids，DefaultStrategy 需要）。

        Returns:
            render (H,W,C) tensor, alpha (H,W,1) tensor, meta dict
        """
        torch = self.torch
        viewmats = torch.tensor(viewmat, dtype=torch.float32,
                                device=self.device)[None]
        Ks = torch.tensor(K, dtype=torch.float32, device=self.device)[None]
        if colors_override is None:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
            sh_degree = self.sh_degree
        else:
            colors = colors_override
            sh_degree = None
        renders, alphas, meta = self.gsplat.rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height,
            sh_degree=sh_degree, packed=False,
        )
        return renders[0], alphas[0], meta

    # ------------------------------------------------------------------
    def render_invdepth(self, viewmat: np.ndarray, K: np.ndarray,
                        width: int, height: int):
        """官方逆深度渲染：高斯中心相机系 z 的倒数作为颜色 → α 混合。

        唯一实现，训练深度监督（train() 的 d_loss）与模板 3D 信息渲染
        （template_renderer.py 的 coord_map/depth_map）都调这里，不允许
        各自手写——直接 μ/z 位置混合会被深层高斯泄漏拉远（实测 ~4-7%），
        逆深度混合让近处高斯主导，才与真实表面一致。

        Returns:
            invdepth (H,W) tensor, alpha (H,W,1) tensor, meta dict
        """
        torch = self.torch
        Rt = torch.tensor(viewmat[:3, :3], dtype=torch.float32,
                          device=self.device)
        tt = torch.tensor(viewmat[:3, 3], dtype=torch.float32,
                          device=self.device)
        z_cam = (self.splats["means"] @ Rt.T + tt)[:, 2:3]
        render, alpha, meta = self.render(
            viewmat, K, width, height,
            colors_override=(1.0 / z_cam.clamp(min=1e-3)).float())
        return render[..., 0], alpha, meta

    # ------------------------------------------------------------------
    def train(self, views: List[Dict], log_every: int = 500,
              bg_color: float = 1.0):
        """训练循环。

        Args:
            views: 每项 {image (H,W,3) float [0,1], viewmat (4,4) w2c,
                    K (3,3), width, height}。物体前景外像素应已置背景色
                    （onboard 阶段用掩码处理）。深度监督时额外提供
                    invdepth (H,W) float（物体掩码外为 0）与 depth_mask
                    (H,W) bool。
            bg_color: 与训练视图一致的背景色。物体级训练必须把渲染与
                alpha 合成背景后再算损失——光栅化在无高斯覆盖的背景像素
                输出恒黑且没有可传梯度的参数，全图 L1 会被占画面 95%+
                的背景项（黑 vs 白，恒定 1.0）淹没，物体区域梯度微弱到
                训练不学习（曾导致 7000 迭代后高斯糊成单 blob）。

        深度正则化（官方 Hierarchical 3DGS 式）：渲染逆深度（高斯中心
        1/z 的 α 混合，官方 forward.cu 同款）与 CAD 渲染的 GT 逆深度做
        掩码内 L1。光度损失只约束视觉表面（μ+σ），中心壳 μ 会自由漂移
        （实测 ~4%），深度监督把 μ 拉向表面，3DGS 自身几何变准
        （coord_map/refine/render_select 全部受益）。
        """
        torch = self.torch
        from .ssim import ssim as ssim_fn
        iters = int(self.cfg.get("iterations", 7000))
        lam = float(self.cfg.get("lambda_ssim", 0.2))
        depth_w = float(self.cfg.get("depth_l1_weight", 0.0))
        rng = np.random.default_rng(0)

        images = [torch.tensor(v["image"], dtype=torch.float32,
                               device=self.device) for v in views]
        use_depth = (depth_w > 0
                     and all("invdepth" in v for v in views)
                     and all("depth_mask" in v for v in views))
        if use_depth:
            inv_gts = [torch.tensor(v["invdepth"], dtype=torch.float32,
                                    device=self.device) for v in views]
            d_masks = [torch.tensor(v["depth_mask"], dtype=torch.float32,
                                    device=self.device) for v in views]
        for it in range(iters):
            v_idx = int(rng.integers(len(views)))
            v = views[v_idx]
            gt = images[v_idx]
            render, alpha, meta = self.render(
                v["viewmat"], v["K"], v["width"], v["height"])
            render = torch.clamp(render, 0.0, 1.0)
            # 背景合成：无高斯覆盖的像素得背景色，其损失恒 0 且无梯度，
            # 损失只由物体区域驱动
            composed = render * alpha + (1.0 - alpha) * bg_color

            self.strategy.step_pre_backward(
                self.splats, self.optimizers, self.strategy_state, it, meta)

            l1 = torch.abs(composed - gt).mean()
            ssim_val = ssim_fn(composed.permute(2, 0, 1)[None],
                               gt.permute(2, 0, 1)[None])
            # L = (1-λ)·L1 + λ·(1-SSIM)
            loss = (1.0 - lam) * l1 + lam * (1.0 - ssim_val)
            if use_depth:
                inv_render, alpha_d, _ = self.render_invdepth(
                    v["viewmat"], v["K"], v["width"], v["height"])
                msk = d_masks[v_idx]
                d_loss = ((torch.abs(inv_render - inv_gts[v_idx]) * msk).sum()
                          / msk.sum().clamp(min=1.0))
                loss = loss + depth_w * d_loss
            loss.backward()

            # 官方顺序（gsplat examples/simple_trainer.py:998→1131-1138→1156）：
            # backward → optimizer.step/zero_grad → strategy.step_post_backward。
            # step_post_backward 会 clone/split/prune 并替换优化器内的参数
            # （strategy/ops.py:96-137），若放在 opt.step() 之前，本次梯度
            # 会作用到密度控制后的新参数上，状态错位。
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            self.strategy.step_post_backward(
                self.splats, self.optimizers, self.strategy_state, it, meta,
                packed=False)

            if log_every and (it + 1) % log_every == 0:
                print(f"  [3DGS] iter {it+1}/{iters} loss={loss.item():.4f} "
                      f"n_gaussians={len(self.splats['means'])}")

    # ------------------------------------------------------------------
    def gaussian_centers(self) -> "np.ndarray":
        """当前全部高斯中心 μ（3D 坐标图渲染的『颜色』来源）。"""
        return self.splats["means"].detach()
