"""测试时位姿精化（3DGS 可微渲染 + 感知损失）。GPU-only。

v2（6d-refiner-v2，2026-08-07）：按 GS-Pose（inference.py:454-580）与旧代码
MyPose refine.py 的思路重做——损失去掉 LPIPS（两者都不用，感知纹理损失对
位姿梯度弱且噪声大）、加 MS-SSIM 与掩码形状监督（旧代码粗搜阶段 mask_loss
主导）；优化改 AdamW + warmup + 余弦退火（GS-Pose 同款，lr→0 保证收敛）+
最近 N 步梯度均值早停 + best-delta 回溯（旧代码防跳出盆地）。
依赖 onboard 保存的 3DGS 参数（.pt），无则构造时抛提示。
"""
from __future__ import annotations

import math

import numpy as np
import torch

_HINT = (
    "位姿精化需要 onboard 保存的 3DGS 参数（<模板库>.pt）。当前库没有，"
    "请重新运行 scripts/data/onboard_object.py。"
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
                 lambda_ms_ssim: float = 1.0, lambda_mask: float = 0.5,
                 lambda_lpips: float = 0.0, lambda_dice: float = 0.3,
                 lr: float = 0.01, iterations: int = 400,
                 bg_color: float = 1.0,
                 warmup_steps: int = 10,
                 early_stop_grad_window: int = 5,
                 early_stop_grad_tol: float = 1e-4,
                 early_stop_patience: int = 0,
                 early_stop_tol: float = 1e-4,
                 supersample: int = 1,
                 stage1_iters: int = 0,
                 lambda_area: float = 0.0,
                 area_gate_dice: float = 0.0):
        import gsplat
        from pathlib import Path
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"{_HINT}\n缺失: {p}")
        self.device = device
        self.gsplat = gsplat
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_ms_ssim = lambda_ms_ssim
        self.lambda_mask = lambda_mask
        self.lambda_lpips = lambda_lpips
        self.lambda_dice = lambda_dice
        self.lr = lr
        self.iterations = iterations
        self.bg_color = bg_color
        self.warmup_steps = int(warmup_steps)
        self.early_stop_grad_window = int(early_stop_grad_window)
        self.early_stop_grad_tol = float(early_stop_grad_tol)
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.supersample = int(supersample)
        self.stage1_iters = int(stage1_iters)
        self.lambda_area = float(lambda_area)
        self.area_gate_dice = float(area_gate_dice)

        ck = torch.load(p, map_location=device, weights_only=False)
        self.splats = {}
        for name, val in ck["splats"].items():
            self.splats[name] = torch.nn.Parameter(val.to(device),
                                                   requires_grad=False)
        self.scene_scale = float(ck.get("scene_scale", 1.0))
        self.sh_degree = int(ck.get("sh_degree", 3))

        # LPIPS 默认关闭（v2 损失不含感知项）；lambda_lpips>0 时才加载
        self.lpips = None
        if self.lambda_lpips > 0:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
                for param in self.lpips.parameters():
                    param.requires_grad = False
            except Exception:
                self.lpips = None
        from .ssim import ssim as ssim_fn
        from .ssim import ms_ssim as ms_ssim_fn
        self.ssim_fn = ssim_fn
        self.ms_ssim_fn = ms_ssim_fn

    # ------------------------------------------------------------------
    def _render(self, R: torch.Tensor, t: torch.Tensor,
                K: torch.Tensor, width: int, height: int,
                scale: int = 1):
        """按 w2c (R,t) 渲染当前 3DGS，返回 (composed (H,W,3) [0,1], alpha)。

        scale>1 时按 scale× 分辨率渲染（K 的 fx/fy/cx/cy 同比例放大）：
        小物体轮廓/尺度信号被放大，tz 方向梯度更强（病态帧专用）。
        """
        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        viewmat = viewmat[None]
        Ks = K.clone()[None]
        if scale > 1:
            Ks[:, 0, 0] *= scale
            Ks[:, 1, 1] *= scale
            Ks[:, 0, 2] *= scale
            Ks[:, 1, 2] *= scale
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        renders, alphas, _ = self.gsplat.rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat, Ks=Ks,
            width=width * scale, height=height * scale,
            sh_degree=self.sh_degree, packed=False,
        )
        rgb = renders[0].clamp(0, 1)
        alpha = alphas[0]
        composed = rgb * alpha + (1.0 - alpha) * self.bg_color
        return composed, alpha

    # ------------------------------------------------------------------
    def refine(self, img_rgb_u8: np.ndarray, mask: np.ndarray,
               K: np.ndarray, R0: np.ndarray, t0: np.ndarray,
               verbose: bool = False, iterations: Optional[int] = None):
        """精化位姿。

        Args:
            img_rgb_u8: (H,W,3) uint8 查询裁剪图（真实 RGB）
            mask: (H,W) bool 前景掩码（与裁剪图同坐标系）
            K: (3,3) 裁剪坐标系内参（主点已平移）
            R0/t0: 初始 w2c 位姿（模型系，mm 单位）
            iterations: 覆盖构造时的迭代数（任务 3 自适应升级档用）
        Returns:
            (R, t) 精化后的 w2c 位姿；失败（无有效掩码）返回 (None, None)
        """
        if mask.sum() < 16:
            return None, None
        n_iters = int(iterations) if iterations else self.iterations
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

        s = self.supersample
        if s > 1:
            import torch.nn.functional as F
            gt = F.interpolate(gt[None], scale_factor=s, mode="bilinear",
                               align_corners=False)[0]
            msk = F.interpolate(msk[None], scale_factor=s, mode="nearest")[0]

        R0t = torch.tensor(R0, dtype=torch.float32, device=self.device)
        t0t = torch.tensor(t0, dtype=torch.float32, device=self.device)
        a_msk = msk.sum()                               # 查询前景面积（放大域）
        best_loss = float("inf")

        def _step_loss(R, t):
            """单步损失：交集掩码 L1 + SSIM + MS-SSIM + 掩码形状 + 面积正则。"""
            composed, alpha = self._render(R, t, Kt, W, H, scale=s)
            comp = composed.permute(2, 0, 1)           # (3,H,W)
            a = alpha[..., 0]                          # (H,W)
            ov = (a > 0.5) & (msk[0] > 0.5)
            # 交集掩码内 L1：遮挡/分割不一致区域（只有一边是前景）不参与
            # 光度对齐，避免查询掩码污染把位姿拉偏
            if ov.sum() >= 16:
                l1 = (torch.abs(comp - gt) * ov).sum() / ov.sum()
            else:
                l1 = (torch.abs(comp - gt) * msk[0]).sum() / msk.sum().clamp(min=1)
            ssim_val = self.ssim_fn(comp[None], gt[None])
            loss = (self.lambda_l1 * l1
                    + self.lambda_ssim * (1.0 - ssim_val))
            if self.lambda_ms_ssim > 0:
                ms_val = self.ms_ssim_fn(comp[None], gt[None])
                loss = loss + self.lambda_ms_ssim * (1.0 - ms_val)
            if self.lpips is not None and self.lambda_lpips > 0:
                # LPIPS 的 vgg 下采样在极小输入上崩（max_pool 输出为 0）：
                # 小于 32px 时跳过感知项，只留 L1+SSIM
                if min(comp.shape[1], comp.shape[2]) >= 32:
                    lp = self.lpips(comp[None] * 2 - 1,
                                    gt[None] * 2 - 1).mean()
                    loss = loss + self.lambda_lpips * lp
            # 掩码形状监督（旧代码 refine.py 粗搜核心）：渲染 alpha 与
            # 查询掩码的 L1，旋转错位时形状误差直接给梯度（光度项对
            # 旋转错位梯度弱）。GS-Pose 用 image*mask 截断实现同效。
            if self.lambda_mask > 0:
                loss = loss + self.lambda_mask * (a - msk[0]).abs().mean()
            dice = (2 * (msk[0] * a).sum()
                    / (msk.sum() + a.sum() + 1e-6))
            loss = loss - self.lambda_dice * dice
            # 面积对数正则：渲染面积与查询面积比直接约束 tz（尺度信号，
            # 对 tz 的梯度远强于光度项）。Dice 门控：掩码被遮挡污染时
            # （IoU 低）关掉，防面积偏差把 tz 拉偏。
            if (self.lambda_area > 0 and self.area_gate_dice > 0
                    and float(dice.detach()) >= self.area_gate_dice):
                ratio = (a.sum() + 1e-6) / (a_msk + 1e-6)
                loss = loss + self.lambda_area * torch.log(ratio) ** 2
            return loss

        # 阶段 1：只优化平移（tx,ty,tz）。旋转不变时，面积/轮廓对平移有
        # 直接梯度，先把 tz 拉回；全 6D 一起动时小物体容易陷在光度局部极小。
        if self.stage1_iters > 0:
            td = torch.zeros(3, device=self.device, requires_grad=True)
            opt1 = torch.optim.Adam([td], lr=self.lr)
            for _ in range(self.stage1_iters):
                loss = _step_loss(R0t, t0t + td)
                loss.backward()
                opt1.step()
                opt1.zero_grad(set_to_none=True)
            t0t = (t0t + td.detach()).clone()

        # 阶段 2：全 6D se(3) 增量。AdamW + warmup + 余弦退火（GS-Pose
        # 同款）：warmup 期 lr 线性升到峰值，之后余弦降到 0——后期步长
        # 收缩保证收敛不震荡（v1 固定 lr 的 Adam 实测常把位姿推坏）。
        delta = torch.zeros(6, device=self.device, requires_grad=True)
        opt = torch.optim.AdamW([delta], lr=self.lr, weight_decay=0.0)
        total = max(n_iters, 1)
        warmup = min(self.warmup_steps, total // 2)
        base_lrs = [pg["lr"] for pg in opt.param_groups]
        losses = []                                    # 梯度早停窗口
        best_delta = None                              # best-loss 回溯
        best_loss = float("inf")
        stale = 0
        for it in range(total):
            if it < warmup:
                frac = (it + 1) / warmup
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * frac
            else:
                # 余弦退火：warmup 结束后从峰值降到 0（GS-Pose 式）
                frac = (it - warmup) / max(total - warmup, 1)
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * 0.5 * (1 + math.cos(math.pi * frac))
            dR, dt = _se3_exp(delta[:3], delta[3:])
            R = dR @ R0t
            t = dR @ t0t + dt
            loss = _step_loss(R, t)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            last_loss = float(loss.item())
            losses.append(last_loss)
            if last_loss < best_loss:
                best_loss = last_loss
                best_delta = delta.detach().clone()
            # GS-Pose 式早停：最近 N 步损失梯度均值 < 阈值
            if (self.early_stop_grad_window > 0
                    and len(losses) > self.early_stop_grad_window):
                grads = [abs(losses[i] - losses[i - 1])
                         for i in range(len(losses) - self.early_stop_grad_window,
                                        len(losses))]
                if sum(grads) / len(grads) < self.early_stop_grad_tol:
                    break
            # 旧式 patience 早停（兼容）
            if (self.early_stop_patience > 0
                    and last_loss >= best_loss - self.early_stop_tol):
                stale += 1
                if stale >= self.early_stop_patience:
                    break
            else:
                stale = 0
            if verbose and (it + 1) % 50 == 0:
                print(f"    [refine] iter {it+1} loss={last_loss:.4f}")

        # best-loss 回溯（旧代码 refine.py 同款）：优化后期可能跳出好
        # 盆地，恢复历史最优 delta 而不是用最后一步
        if best_delta is not None:
            delta.data.copy_(best_delta)
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
