"""模板渲染与 3D 坐标图。GPU-only（依赖 gsplat）。

对 40 个模板位姿 P_m（8 立方体顶点 × 5 平面内旋转）分别渲染：
- RGB 模板图 T_m（256×256）
- alpha 图（前景掩码来源）
- 3D 坐标图 C_m：把每个高斯的中心 μ 当作 3 通道『颜色』做 alpha 混合，
  得到每像素的物体系 3D 坐标（Φ_i 的 alpha 混合实现——
  相比 argmax 主贡献高斯，alpha 混合是其光顺化版本，且无需改动光栅器）。
  混合结果除以 alpha 归一（消除背景透射的衰减），alpha 过低的像素视为背景。

产物统一保存为 templates.npz，供在线阶段与 MASt3R 特征缓存使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ..geometry.view_sampling import (generate_template_poses,
                                      template_intrinsics,
                                      to_pixel_center_intrinsics)


ALPHA_FG_THRESH = 0.5   # alpha 低于该值的像素视为背景（坐标图无效）


def render_template_bank(trainer, cfg_templates: Dict, out_path,
                         bg_color: float = 1.0,
                         anchor_mode: str = "invdepth") -> Dict[str, np.ndarray]:
    """渲染完整模板库并落盘（离线步骤 4-6）。

    Args:
        trainer: 训练完成的 GaussianTrainer
        cfg_templates: configs/current/default.yaml 的 templates 段
        out_path: 输出 npz 路径
    Returns:
        dict(images (M,S,S,3) uint8, alphas (M,S,S) float16,
             coord_maps (M,S,S,3) float32, poses (M,4,4), K (3,3))
    """
    torch = trainer.torch
    size = int(cfg_templates.get("image_size", 256))
    # K 为整数像素索引约定（落盘 + 下游深度反投影/坐标图查表用），
    # K_render 为 gsplat 的像素中心约定（主点 +0.5）。两者成对使用，
    # 深度反投影才与渲染落点闭环无偏——详见 view_sampling.template_intrinsics
    K = template_intrinsics(size, float(cfg_templates.get("fov_deg", 40.0)))
    K_render = to_pixel_center_intrinsics(K)

    # 渲染距离：包围盒对角线 × radius_scale（自适应设置）
    means = trainer.gaussian_centers().cpu().numpy()
    diag = float(np.linalg.norm(means.max(0) - means.min(0)))
    radius = diag * float(cfg_templates.get("radius_scale", 2.5))

    poses = generate_template_poses(
        radius=radius,
        viewpoint_mode=cfg_templates.get("viewpoint_mode", "cube8"),
        n_viewpoints=int(cfg_templates.get("n_viewpoints", 8)),
        n_inplane=int(cfg_templates.get("n_inplane", 5)),
    )

    # template_source=depth_map 时同时渲染深度图，供深度反投影 2D-3D 提升
    # （matching.lifting=depth_backproject，历史对照口径，见 VERIFICATION.md §8.1）。
    # 深度 = 高斯中心相机系 z 的 alpha 混合（与坐标图同一归一化）。
    store_depth = cfg_templates.get("template_source",
                                    "coord_map") == "depth_map"
    images, alphas, coord_maps, depth_maps = [], [], [], []
    centers = trainer.gaussian_centers()          # (N,3) torch
    with torch.no_grad():
        for T in poses:
            # RGB 模板
            rgb, alpha, _ = trainer.render(T, K_render, size, size)
            rgb = torch.clamp(rgb, 0, 1)
            a = alpha[..., 0]
            # 背景合成为纯色（DINOv2/MASt3R 输入更干净）
            rgb = rgb + (1.0 - alpha) * bg_color

            # 3D 坐标图：逆深度混合（expected_invdepth，官方
            # depth-regularization 同款，见 scripts/maintenance/patch_depth_anchor_maps.py
            # 与 docs/RESEARCH_LOG.md §2-4）。直接 μ 位置混合会被深层高斯
            # 泄漏拉远（中心壳偏内 4-7%，PnP 深度系统性偏浅/偏深），逆深度
            # 混合近处高斯主导，锚点深度与真实表面一致（实测偏差 +0.03%）。
            # anchor_mode=coord 恢复 μ 位置混合（历史口径，对照实验用）。
            Tt = torch.tensor(T, dtype=torch.float32, device=centers.device)
            if anchor_mode == "coord":
                cm_raw, alpha_c, _ = trainer.render(
                    T, K_render, size, size, colors_override=centers)
                cm = cm_raw / torch.clamp(alpha_c, min=1e-6)
                fg = a > ALPHA_FG_THRESH
                cm[~fg] = 0.0
            else:
                z_cam = (centers @ Tt[:3, :3].T + Tt[:3, 3])[:, 2:3]  # (N,1)
                inv_d, alpha_c, _ = trainer.render(
                    T, K_render, size, size,
                    colors_override=1.0 / z_cam.clamp(min=1e-3))
                z_mix = 1.0 / torch.clamp(inv_d[..., 0], min=1e-9)
                fg = a > ALPHA_FG_THRESH
                z_mix = torch.where(fg, z_mix, torch.zeros_like(z_mix))
                ys, xs = torch.nonzero(z_mix > 0, as_tuple=True)
                pc = torch.stack([
                    (xs.float() - K[0, 2]) / K[0, 0] * z_mix[ys, xs],
                    (ys.float() - K[1, 2]) / K[1, 1] * z_mix[ys, xs],
                    z_mix[ys, xs]], dim=1)
                po = (pc - Tt[:3, 3]) @ Tt[:3, :3]          # 物体系
                cm = torch.zeros((size, size, 3), device=z_mix.device)
                cm[ys, xs] = po
                cm[~fg] = 0.0

            images.append((rgb.cpu().numpy() * 255).astype(np.uint8))
            alphas.append(a.cpu().numpy().astype(np.float16))
            coord_maps.append(cm.cpu().numpy().astype(np.float32))

            if store_depth:
                # 相机系 z = (R μ + t)[2]，作为单通道特征 alpha 混合
                Tt = torch.tensor(T, dtype=torch.float32,
                                  device=centers.device)
                z_cam = (centers @ Tt[:3, :3].T + Tt[:3, 3])[:, 2:3]  # (N,1)
                dm, alpha_d, _ = trainer.render(T, K_render, size, size,
                                                colors_override=z_cam)
                dm = dm[..., 0] / torch.clamp(alpha_d[..., 0], min=1e-6)
                dm[~fg] = 0.0                    # 背景深度置 0（无效）
                depth_maps.append(dm.cpu().numpy().astype(np.float32))

    bank = {
        "images": np.stack(images),
        "alphas": np.stack(alphas),
        "coord_maps": np.stack(coord_maps),
        "poses": poses.astype(np.float32),
        "K": K.astype(np.float32),
        "radius": np.float32(radius),
        "anchor_mode": np.str_(anchor_mode),   # 锚点渲染方式：invdepth（当前，
        # 官方 expected_invdepth 同款）| coord（μ 位置混合，历史口径）
    }
    if store_depth:
        bank["depth_maps"] = np.stack(depth_maps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **bank)
    return bank


def render_template_bank_pyrender(mesh_path, scale: float,
                                  cfg_templates: Dict, out_path,
                                  device: str = "cuda") -> Dict[str, np.ndarray]:
    """渲染器消融：pyrender 直接光栅化 CAD 网格的模板库（替代 3DGS）。

    坐标图直接由深度反投影 + 相机→物体系变换得到（网格渲染无 alpha 混合，
    坐标图是精确的表面点）。依赖 pyrender + EGL 离屏渲染（GPU 机器）。
    """
    try:
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh
    except ImportError as e:
        raise ImportError(
            "pyrender_cad 渲染器需要: pip install pyrender trimesh\n"
            f"（仅 GPU 机器 / EGL 环境）原始错误: {e}") from e

    size = int(cfg_templates.get("image_size", 256))
    # 同 3DGS 分支：K 是整数像素索引约定（落盘 + 下面坐标图反投影用），
    # 相机对象要的是像素中心约定（主点 +0.5）
    K = template_intrinsics(size, float(cfg_templates.get("fov_deg", 40.0)))
    K_render = to_pixel_center_intrinsics(K)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.apply_scale(scale)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    radius = diag * float(cfg_templates.get("radius_scale", 2.5))
    poses = generate_template_poses(
        radius=radius,
        viewpoint_mode=cfg_templates.get("viewpoint_mode", "cube8"),
        n_viewpoints=int(cfg_templates.get("n_viewpoints", 8)),
        n_inplane=int(cfg_templates.get("n_inplane", 5)),
    )

    scene = pyrender.Scene(bg_color=[255, 255, 255, 0],
                           ambient_light=[0.4, 0.4, 0.4])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.IntrinsicsCamera(fx=K_render[0, 0], fy=K_render[1, 1],
                                    cx=K_render[0, 2], cy=K_render[1, 2])
    cam_node = scene.add(cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(intensity=3.0)
    light_node = scene.add(light, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(size, size)

    # OpenCV(w2c, y下z前) → OpenGL(c2w, y上z后) 的相机位姿变换
    cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])
    # pyrender 的 depth buffer 即精确 z-depth，template_source=depth_map 时
    # 直接保存供深度反投影路线使用（历史对照口径，见 VERIFICATION.md §8.1）
    store_depth = cfg_templates.get("template_source",
                                    "coord_map") == "depth_map"
    images, alphas, coord_maps, depth_maps = [], [], [], []
    for T in poses:
        c2w = np.linalg.inv(T) @ cv2gl
        scene.set_pose(cam_node, c2w)
        scene.set_pose(light_node, c2w)
        color, depth = r.render(scene)
        a = (depth > 0).astype(np.float32)
        # 深度反投影到相机系，再变换回物体系 → 精确 3D 坐标图。
        # xs/ys 是**整数像素下标**，必须配整数像素索引约定的 K（主点已含
        # -0.5），否则与渲染落点差半像素——原先用 cx=S/2 的 K 少了这半格
        ys, xs = np.nonzero(depth > 0)
        z = depth[ys, xs]
        pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                       (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
        R, t = T[:3, :3], T[:3, 3]
        po = (pc - t) @ R                       # R^T (pc - t)
        cm = np.zeros((size, size, 3), dtype=np.float32)
        cm[ys, xs] = po
        images.append(color[..., :3].astype(np.uint8))
        alphas.append(a.astype(np.float16))
        coord_maps.append(cm)
        if store_depth:
            depth_maps.append(depth.astype(np.float32))
    r.delete()

    bank = {
        "images": np.stack(images), "alphas": np.stack(alphas),
        "coord_maps": np.stack(coord_maps), "poses": poses.astype(np.float32),
        "K": K.astype(np.float32), "radius": np.float32(radius),
    }
    if store_depth:
        bank["depth_maps"] = np.stack(depth_maps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **bank)
    return bank
