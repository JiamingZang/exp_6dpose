"""测试时位姿精化（3DGS 可微渲染 + 感知损失）。GPU-only。

思路（GSPose 同款，见 _prior_code/MyPose/refine.py）：用粗位姿渲染 3DGS，
与真实查询裁剪图对齐——在掩码内最小化 L1 + (1-SSIM) + LPIPS，梯度只反向
到位姿的 se(3) 增量上（3DGS 参数冻结）。

依赖 onboard 保存的 3DGS 参数（.pt），无则构造时抛提示。
"""
from __future__ import annotations

import numpy as np
import torch

_HINT = (
    "位姿精化需要 onboard 保存的 3DGS 参数（<模板库>.pt）。当前库没有，"
    "请重新运行 scripts/onboard_object.py。"
)


def _se3_exp(omega: torch.Tensor, v: torch.Tensor):
    """se(3) 指数映射（Rodrigues）：返回 (ΔR, Δt)。"""
    theta = omega.norm() + 1e-8
    w = omega / theta
    wx = torch.zeros(3, 3, device=omega.device)
    wx[0, 1] = -w[2]
    wx[0, 2] = w[1]
    wx[1, 0] = w[2]
    wx[1, 2] = -w[0]
    wx[2, 0] = -w[1]
    wx[2, 1] = w[0]
    I = torch.eye(3, device=omega.device)
    dR = (I + torch.sin(theta) * wx
          + (1 - torch.cos(theta)) * (wx @ wx))
    a = (1 - torch.cos(theta)) / theta ** 2
    b = (theta - torch.sin(theta)) / theta ** 3
    V = I + a * wx + b * (wx @ wx)
    return dR, V @ v


class PoseRefiner:
    """基于 3DGS 可微渲染的测试时位姿精化器。"""

    def __init__(self, ckpt_path, device: str = "cuda",
                 lambda_l1: float = 1.0, lambda_ssim: float = 0.5,
                 lambda_lpips: float = 0.1, lambda_dice: float = 0.3,
                 lr: float = 0.02, iterations: int = 150,
                 bg_color: float = 1.0,
                 early_stop_patience: int = 0,
                 early_stop_tol: float = 1e-4):
        import gsplat
        from pathlib import Path
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"{_HINT}\n缺失: {p}")
        self.device = device
        self.gsplat = gsplat
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_dice = lambda_dice
        self.lr = lr
        self.iterations = iterations
        self.bg_color = bg_color
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol

        ck = torch.load(p, map_location=device, weights_only=False)
        self.splats = {}
        for name, val in ck["splats"].items():
            self.splats[name] = torch.nn.Parameter(val.to(device),
                                                   requires_grad=False)
        self.scene_scale = float(ck.get("scene_scale", 1.0))
        self.sh_degree = int(ck.get("sh_degree", 3))

        try:
            import lpips
            self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        except Exception:
            self.lpips = None
        from .ssim import ssim as ssim_fn
        self.ssim_fn = ssim_fn

    # ------------------------------------------------------------------
    def _render(self, R: torch.Tensor, t: torch.Tensor,
                K: torch.Tensor, width: int, height: int):
        """按 w2c (R,t) 渲染当前 3DGS，返回 (composed (H,W,3) [0,1], alpha)。"""
        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        viewmat = viewmat[None]
        Ks = K.clone()[None]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        renders, alphas, _ = self.gsplat.rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat, Ks=Ks, width=width, height=height,
            sh_degree=self.sh_degree, packed=False,
        )
        rgb = renders[0].clamp(0, 1)
        alpha = alphas[0]
        composed = rgb * alpha + (1.0 - alpha) * self.bg_color
        return composed, alpha

    # ------------------------------------------------------------------
    def refine(self, img_rgb_u8: np.ndarray, mask: np.ndarray,
               K: np.ndarray, R0: np.ndarray, t0: np.ndarray,
               verbose: bool = False):
        """精化位姿。

        Args:
            img_rgb_u8: (H,W,3) uint8 查询裁剪图（真实 RGB）
            mask: (H,W) bool 前景掩码（与裁剪图同坐标系）
            K: (3,3) 裁剪坐标系内参（主点已平移）
            R0/t0: 初始 w2c 位姿（模型系，mm 单位）
        Returns:
            (R, t) 精化后的 w2c 位姿；失败（无有效掩码）返回 (None, None)
        """
        if mask.sum() < 16:
            return None, None
        H, W = img_rgb_u8.shape[:2]
        # GT = 真实裁剪图，掩码外涂白（与模板/渲染背景一致）
        gt_np = img_rgb_u8.astype(np.float32) / 255.0
        gt_np = np.where(mask[..., None].astype(bool), gt_np,
                         np.full_like(gt_np, self.bg_color))
        gt = torch.tensor(gt_np, dtype=torch.float32,
                          device=self.device).permute(2, 0, 1)[None]
        gt = gt[0]                                     # (3,H,W)
        msk = torch.tensor(mask, dtype=torch.float32,
                           device=self.device)[None]   # (1,H,W)
        Kt = torch.tensor(K, dtype=torch.float32, device=self.device)

        R0t = torch.tensor(R0, dtype=torch.float32, device=self.device)
        t0t = torch.tensor(t0, dtype=torch.float32, device=self.device)
        delta = torch.zeros(6, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.lr)

        from .ssim import ssim as ssim_fn
        last_loss = float("inf")
        best_loss = float("inf")
        stale = 0
        for it in range(self.iterations):
            dR, dt = _se3_exp(delta[:3], delta[3:])
            R = dR @ R0t
            t = dR @ t0t + dt
            composed, alpha = self._render(R, t, Kt, W, H)
            comp = composed.permute(2, 0, 1)           # (3,H,W)
            # 掩码内 L1（背景已与渲染合成色对齐，全图算也不污染梯度主项）
            l1 = (torch.abs(comp - gt) * msk).sum() / msk.sum().clamp(min=1)
            ssim_val = ssim_fn(comp[None], gt[None])
            loss = (self.lambda_l1 * l1
                    + self.lambda_ssim * (1.0 - ssim_val))
            # LPIPS(VGG) 单次前向 ~10ms，是迭代里最贵的一项；GS-Pose 只用
            # SSIM 系损失即可收敛，lambda_lpips=0 时彻底跳过（省 2/3 耗时）
            if self.lpips is not None and self.lambda_lpips > 0:
                lp = self.lpips(comp[None] * 2 - 1, gt[None] * 2 - 1).mean()
                loss = loss + self.lambda_lpips * lp
            # 渲染轮廓与分割掩码的 Dice：轮廓对齐对平移是强几何约束，
            # 防止 mask 边界误差把位姿拉偏
            a = alpha[..., 0]
            dice = (2 * (msk[0] * a).sum()
                    / (msk.sum() + a.sum() + 1e-6))
            loss = loss - self.lambda_dice * dice
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            last_loss = float(loss.item())
            if self.early_stop_patience > 0:
                if last_loss < best_loss - self.early_stop_tol:
                    best_loss = last_loss
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.early_stop_patience:
                        break
            if verbose and (it + 1) % 50 == 0:
                print(f"    [refine] iter {it+1} loss={last_loss:.4f}")

        dR, dt = _se3_exp(delta[:3], delta[3:])
        R = (dR @ R0t).detach().cpu().numpy()
        t = (dR @ t0t + dt).detach().cpu().numpy()
        return R, t

    # ------------------------------------------------------------------
    def align_loss(self, img_rgb_u8: np.ndarray, mask: np.ndarray,
                   K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
        """前向渲染对齐损失（不优化）：掩码内 L1 + (1-SSIM)。

        定位候选消歧用（pipeline 渲染验证）：候选掩码 x 粗位姿组合中，
        正确的组合渲染与真实图对齐，损失显著低于错误候选（错误 mask 的
        crop 里没有目标物体，渲染内容对不上）。
        """
        H, W = img_rgb_u8.shape[:2]
        gt_np = img_rgb_u8.astype(np.float32) / 255.0
        gt_np = np.where(mask[..., None].astype(bool), gt_np,
                         np.full_like(gt_np, self.bg_color))
        gt = torch.tensor(gt_np, dtype=torch.float32,
                          device=self.device).permute(2, 0, 1)[None]
        gt = gt[0]                                     # (3,H,W)
        msk = torch.tensor(mask, dtype=torch.float32,
                           device=self.device)[None]   # (1,H,W)
        Kt = torch.tensor(K, dtype=torch.float32, device=self.device)
        R0t = torch.tensor(R, dtype=torch.float32, device=self.device)
        t0t = torch.tensor(t, dtype=torch.float32, device=self.device)
        composed, alpha = self._render(R0t, t0t, Kt, W, H)
        comp = composed.permute(2, 0, 1)
        l1 = (torch.abs(comp - gt) * msk).sum() / msk.sum().clamp(min=1)
        from .ssim import ssim as ssim_fn
        ssim_val = ssim_fn(comp[None], gt[None])
        return float(l1 + self.lambda_ssim * (1.0 - ssim_val))

    # ------------------------------------------------------------------
    def mask_iou(self, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                 mask: np.ndarray, alpha_thresh: float = 0.5) -> float:
        """渲染 alpha 与查询掩码的 IoU（PnP 候选渲染择优/爆炸过滤用）。

        掩码几何比整图光度更鲁棒：爆炸位姿（t 偏移数百 mm）渲染的 mask
        与真实掩码几乎不重叠，IoU≈0；正确候选 IoU 通常 >0.6。单次渲染
        ~30ms，可对 top-K 候选逐一计算后选优。
        """
        H, W = mask.shape[:2]
        Kt = torch.tensor(K, dtype=torch.float32, device=self.device)
        R0t = torch.tensor(R, dtype=torch.float32, device=self.device)
        t0t = torch.tensor(t, dtype=torch.float32, device=self.device)
        _, alpha = self._render(R0t, t0t, Kt, W, H)
        a = (alpha[..., 0].detach().cpu().numpy() > alpha_thresh)
        inter = np.logical_and(a, mask).sum()
        union = np.logical_or(a, mask).sum()
        return float(inter) / max(float(union), 1.0)
