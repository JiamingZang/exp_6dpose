"""端到端管线：离线 onboard 物体 + 在线单帧推理。

离线（onboard_object，每物体一次）：
  CAD 点云采样 / VGGT 重建 → 尺度对齐 s=f_q/f_r → 3DGS 训练（7000 迭代）
  → 40 模板渲染（RGB + alpha + 3D 坐标图）→ DINOv2 模板 CLS 特征缓存

在线（PoseEstimator.estimate，每帧）：
  SAM 自动掩码 + DINOv2 定位 → 裁剪（bbox 扩 20%）→ MASt3R 40 对解码打分
  → Top-K(=5) 稠密对应（互最近邻 + cycle τ=5px + 阈值 0.3 + 采样 4096）
  → 逐模板 RANSAC-EPnP（ε=5px, conf 0.999, 1000 迭代）→ 内点数择优

本模块 import 时不触碰任何 GPU 依赖；GPU 组件在构造时才加载，导入失败
抛带部署提示的 ImportError（详见各子模块）。
"""
from __future__ import annotations

import sys
import time
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .datasets.linemod import LinemodDataset
from .datasets.ply_io import load_ply, sample_mesh_points
from .geometry.scale_align import align_pointcloud, scale_factor
from .geometry.alignment import (umeyama_alignment, farthest_point_sample,
                                 icp_refine, transform_pose_by_similarity)
from .matching.correspondence import back_to_original_pixels
from .solver.ransac_pnp import PnPResult, ransac_pnp
from .solver.selection import rank_candidates


# ---------------------------------------------------------------------------
# 模板库产物
# ---------------------------------------------------------------------------
class TemplateBank:
    """onboard 产物：模板图/alpha/3D 坐标图/位姿 + DINOv2 CLS 特征。

    `scale` / `dino_feats` 缺失一律 raise（不静默回退）：
    `render_template_bank` 会先把渲染结果落到最终路径，之后 onboard 才补
    DINOv2 特征与尺度因子并二次落盘。若 onboard 在这两步之间中断（DINOv2
    OOM 很常见），残留文件在旧实现下能被"正常"加载——`scale` 静默回退 1.0
    会让平移量级系统性错误且无任何报错，`dino_feats=None` 会让 DINOv2 定位
    路线在推理中途才炸。宁可加载即失败并提示重新 onboard。
    """

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.images = d["images"]            # (M,S,S,3) uint8
        self.alphas = d["alphas"].astype(np.float32)
        self.coord_maps = d["coord_maps"]    # (M,S,S,3) 对齐尺度物体系
        # 锚点渲染方式：invdepth（逆深度混合，当前正确口径，深度偏差
        # ~0.03%）| coord（μ 位置混合，历史口径，深度系统性偏大 1.7%）。
        # 缺失字段 = 旧库（coord 或已 patch 无标记）——只警告不阻断。
        self.anchor_mode = d["anchor_mode"] if "anchor_mode" in d else None
        if self.anchor_mode is not None and self.anchor_mode != "invdepth":
            import warnings
            warnings.warn(
                f"模板库锚点渲染方式为 {self.anchor_mode!r}（期望 invdepth）："
                f"μ 位置混合锚点深度系统性偏大 ~1.7%，会污染 PnP 深度。"
                f"请运行 scripts/maintenance/patch_depth_anchor_maps.py 后重评估。"
                f"（{npz_path}）", stacklevel=2)
        self.poses = d["poses"]              # (M,4,4) w2c
        self.K = d["K"]
        if "scale" not in d:
            raise ValueError(
                f"模板库缺 scale（尺度对齐因子）: {npz_path}。这通常是 "
                f"onboard 在模板渲染落盘之后、DINOv2 特征之前中断留下的残留"
                f"文件——静默按 1.0 用会让平移量级系统性错误。请删除该文件并"
                f"重新运行 scripts/data/onboard_object.py。")
        self.scale = float(d["scale"])
        if "dino_feats" not in d:
            raise ValueError(
                f"模板库缺 dino_feats（DINOv2 模板 CLS 特征）: {npz_path}。"
                f"同上，属 onboard 未跑完的残留文件；请删除后重新运行 "
                f"scripts/data/onboard_object.py。")
        self.dino_feats = d["dino_feats"]
        # 模板渲染背景色（0=黑, 1=白）。缺失（旧库）时 None，extract 侧
        # 无法校验，只能信配置；新库强制写入，见 onboard_object。
        self.bg_color = float(d["bg_color"]) if "bg_color" in d else None
        # template_source=depth_map 时 onboard 额外渲染的模板深度图
        # （matching.lifting=depth_backproject 的 2D-3D 提升来源；历史对照
        # 口径，见 VERIFICATION.md §8.1）。缺省 None（坐标图路线）。
        self.depth_maps = d["depth_maps"] if "depth_maps" in d else None
        if self.depth_maps is not None:
            # 形状必须逐维对齐（含 M 维）：深度图分辨率或模板数与 images
            # 不一致时，反投影会拿错模板/越界取值，静默产出错位 3D 锚点
            if tuple(self.depth_maps.shape[:3]) != tuple(self.images.shape[:3]):
                raise ValueError(
                    f"模板库 depth_maps 形状与 images 不一致: "
                    f"depth_maps{tuple(self.depth_maps.shape)} vs "
                    f"images{tuple(self.images.shape)}（前三维 (M,S,S) 必须相同）。"
                    f"多半是深度库与坐标图库混用或渲染分辨率改过，"
                    f"请重新运行 scripts/data/onboard_object.py。")
        # VGGT 路线的重建系→CAD 系相似变换（仅评测侧对齐用）。
        # CAD 路线下无此变换，默认恒等。
        self.has_align = "align_R" in d
        self.align_s = float(d["align_s"]) if "align_s" in d else 1.0
        self.align_R = (d["align_R"] if "align_R" in d
                        else np.eye(3, dtype=np.float64))
        self.align_t = (d["align_t"] if "align_t" in d
                        else np.zeros(3, dtype=np.float64))

    def __len__(self):
        return len(self.images)


def template_bank_path(cfg: Dict, obj_name: str) -> Path:
    tdir = Path(cfg["runtime"].get("template_dir", "outputs/templates"))
    tag = cfg["renderer"].get("backend", "3dgs")
    n = int(cfg["templates"]["n_viewpoints"]) * int(cfg["templates"]["n_inplane"])
    geo = cfg["geometry"].get("source", "cad")
    sa = "sa" if cfg["scale_align"].get("enabled", True) else "nosa"
    # 深度图模板库与坐标图模板库内容不同（多一份 depth_maps），文件名
    # 必须区分，否则 legacy 配置会静默复用已有的 coord_map 库、depth_maps 缺失
    src = cfg["templates"].get("template_source", "coord_map")
    suffix = "_depth" if src == "depth_map" else ""
    return tdir / f"{obj_name}_{tag}_{geo}_{n}t_{sa}{suffix}.npz"


# ---------------------------------------------------------------------------
# 离线：物体准备
# ---------------------------------------------------------------------------
def onboard_object(cfg: Dict, obj_name: str, device: str = "cuda",
                   verbose: bool = True) -> Path:
    """物体准备：几何初始化 → 尺度对齐 → 3DGS → 模板库 + DINOv2 特征。

    GPU-only（3DGS 训练 / DINOv2）。返回模板库 npz 路径。
    """
    ds = LinemodDataset(cfg["dataset"]["root"], obj_name,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    frames = ds.frames()
    out_path = template_bank_path(cfg, obj_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 幂等：模板库 npz（含 scale/dino_feats）与 3DGS 参数 .pt 都在，且
    # 训练指纹与配置一致才跳过。指纹不一致（迭代数/参考帧数/深度监督/
    # 锚点渲染方式/背景色等变了）必须重训——文件名不含这些参数，静默
    # 复用旧库会让"改了配置但没生效"（30k 迭代事故，08-04）。
    def _train_fingerprint(cfg_):
        return {
            "iterations": int(cfg_["gaussian"].get("iterations", 7000)),
            "n_ref_views": int(cfg_["onboard"].get("n_ref_views", 64)),
            "depth_l1_weight": float(cfg_["gaussian"].get("depth_l1_weight", 0.0)),
            "bg_color": float(cfg_["onboard"].get("bg_color", 1.0)),
            "anchor_mode": "invdepth",
        }

    fp_cfg = _train_fingerprint(cfg)
    if out_path.exists() and out_path.with_suffix(".pt").exists():
        stale = None
        try:
            with np.load(out_path, allow_pickle=False) as d:
                if "train_fp" in d:
                    raw = d["train_fp"].item()
                    fp_bank = {k: float(v) if hasattr(v, "item")
                               else v for k, v in raw.items()}
                else:
                    fp_bank = None
            if fp_bank is None:
                # 旧库无指纹：无法确认训练配置，保守强制重训
                # （patch 后的库也不写指纹——指纹只有新 onboard 写）
                stale = "train_fp 缺失（旧库，无法确认配置）"
            else:
                for k, v in fp_cfg.items():
                    if k in fp_bank and abs(float(fp_bank[k]) - float(v)) > 1e-9:
                        stale = f"{k}: bank={fp_bank[k]} cfg={v}"
                        break
                if stale is None and "anchor_mode" in fp_bank:
                    if str(fp_bank["anchor_mode"]) != "invdepth":
                        stale = "anchor_mode 非 invdepth"
        except Exception:      # noqa: BLE001
            stale = "train_fp 读取失败"
        if stale is None:
            try:
                TemplateBank(out_path)
                if verbose:
                    print(f"[onboard:{obj_name}] 模板库已存在且指纹一致，"
                          f"跳过: {out_path}")
                return out_path
            except ValueError:
                pass    # 残留/不完整文件，重新 onboard
        else:
            print(f"[onboard:{obj_name}] 模板库指纹不一致（{stale}），"
                  f"强制重训 → {out_path}")

    # ---- 1. 几何初始化 ----
    geo_src = cfg["geometry"].get("source", "cad")
    recon_for_align = None       # VGGT 路线：重建系（重心归零）点云，评测对齐用
    if geo_src == "cad":
        verts, vcolors, faces = load_ply(ds.model_path)
        points, colors = sample_mesh_points(
            verts, faces, int(cfg["geometry"].get("n_sample_points", 8192)),
            colors=vcolors, rng=np.random.default_rng(cfg["runtime"]["seed"]))
    elif geo_src == "vggt":
        from .datasets.vggt_recon import reconstruct_with_vggt
        n_ref = int(cfg["geometry"]["vggt"].get("n_ref_images", 3))
        # VGGT 参考帧走与 3DGS 相同的 split 通道
        # （P1-1 复审修复）——train_split 存在时只从 train 抽，无 split 时
        # 抽出的帧也会通过 evaluate_object 的 extra_exclude_ids 从评测集扣除。
        ref_ids = ds.vggt_reference_frame_ids(n_ref)
        ref_frames = [fr for fr in frames if fr.frame_id in ref_ids]
        ref_paths = [str(fr.rgb_path) for fr in ref_frames]
        # 传入前景掩码，只重建物体点（避免背景污染点云）。
        # P1-3 复审：缺掩码直接 raise，绝不静默用整图重建——背景噪点会把
        # Umeyama 拉偏得没法评测。
        masks = _load_reference_masks(ref_frames)
        points, f_ref_vggt = reconstruct_with_vggt(
            ref_paths, masks=masks,
            checkpoint=cfg["geometry"]["vggt"]["checkpoint"],
            device=device)
        # 点云重心平移到原点作为重建模型系
        points = np.asarray(points, dtype=np.float64)
        points = points - points.mean(axis=0, keepdims=True)
        recon_for_align = points.copy()
        colors = None
    else:
        raise ValueError(f"未知几何来源: {geo_src}")

    # ---- 2. 物理尺度对齐：s = f_query / f_ref ----
    f_query = float(frames[0].K[0, 0])
    f_ref = (f_ref_vggt if geo_src == "vggt"
             else float(cfg["scale_align"].get("f_ref", f_query)))
    s = scale_factor(f_query, f_ref,
                     enabled=bool(cfg["scale_align"].get("enabled", True)))
    points = align_pointcloud(points, s)
    if verbose:
        print(f"[onboard:{obj_name}] 几何={geo_src} 点数={len(points)} "
              f"尺度因子 s={s:.4f}")

    # ---- 3-6. 渲染器分支 ----
    tpl_cfg = cfg["templates"]
    if cfg["renderer"].get("backend", "3dgs") == "pyrender_cad":
        # 渲染器消融：跳过 3DGS，直接光栅化 CAD
        from .gaussian.template_renderer import render_template_bank_pyrender
        bank = render_template_bank_pyrender(ds.model_path, s, tpl_cfg,
                                             out_path, device=device)
    else:
        # ---- 3. 3DGS 训练 ----
        from .gaussian.gs_trainer import GaussianTrainer
        from .gaussian.template_renderer import render_template_bank
        views = _build_reference_views(cfg, ds, s)
        trainer = GaussianTrainer(points, colors, cfg["gaussian"],
                                  device=device)
        t0 = time.time()
        trainer.train(views, bg_color=float(cfg["onboard"].get("bg_color", 1.0)))
        if verbose:
            print(f"[onboard:{obj_name}] 3DGS 训练 {time.time()-t0:.0f}s")
        # 保存 3DGS 参数（测试时位姿精化/任意视角渲染用；模板库只存渲染结果）
        ckpt_path = out_path.with_suffix(".pt")
        ckpt = {"splats": {k: v.detach().cpu()
                           for k, v in trainer.splats.items()},
                "scene_scale": np.float32(trainer.scene_scale),
                "sh_degree": np.int32(trainer.sh_degree)}
        torch.save(ckpt, ckpt_path)
        # ---- 4-6. 模板渲染 + 3D 坐标图 ----
        bank = render_template_bank(trainer, tpl_cfg, out_path,
                                    bg_color=float(cfg["onboard"]
                                                   .get("bg_color", 1.0)))

    # ---- 7. DINOv2 模板 CLS 特征 ----
    from .detection.localize import Dinov2Embedder
    embedder = Dinov2Embedder(cfg["detection"], device=device)
    dino_feats = embedder.template_features(bank["images"])

    bank["dino_feats"] = dino_feats.astype(np.float32)
    bank["scale"] = np.float32(s)
    # 训练/渲染背景色写入 bank：下游 extract 的裁剪填色必须与此一致，
    # 否则静默域不匹配（浅色物体黑背景、深色物体白背景，见 EXPERIMENTS）
    bank["bg_color"] = np.float32(float(cfg["onboard"].get("bg_color", 1.0)))
    # 训练指纹（幂等校验用，见 onboard_object 开头）：配置变了强制重训，
    # 不静默复用旧库
    bank["train_fp"] = np.array({
        "iterations": int(cfg["gaussian"].get("iterations", 7000)),
        "n_ref_views": int(cfg["onboard"].get("n_ref_views", 64)),
        "depth_l1_weight": float(cfg["gaussian"].get("depth_l1_weight", 0.0)),
        "bg_color": float(cfg["onboard"].get("bg_color", 1.0)),
        "anchor_mode": "invdepth",
    })
    # ---- 8. VGGT 路线：重建系→CAD 系相似变换（仅评测对齐）----
    if recon_for_align is not None and ds.model_path.exists():
        cad_verts, _, _ = load_ply(ds.model_path)
        s_a, R_a, t_a = _compute_vggt_cad_alignment(
            recon_for_align, cad_verts, cfg,
            rng=np.random.default_rng(cfg["runtime"]["seed"]))
        bank["align_s"] = np.float64(s_a)
        bank["align_R"] = R_a.astype(np.float64)
        bank["align_t"] = t_a.astype(np.float64)
        if verbose:
            print(f"[onboard:{obj_name}] VGGT→CAD 对齐 s={s_a:.4f}")
    np.savez_compressed(out_path, **bank)
    if verbose:
        print(f"[onboard:{obj_name}] 模板库 {len(bank['images'])} 个 → {out_path}")
    return out_path


def _load_reference_masks(frames):
    """加载参考帧 GT 前景掩码（VGGT 前景重建用）。

    P1-3 复审：任一帧缺掩码/读不出即 raise，不再静默 fallback 到整图重建。
    背景噪点会把 Umeyama 对齐拉偏，评测数字不可靠——宁可中止 onboard 让
    用户明确处理（补 mask 或改走 CAD 路线），也不产出无声无息错误的模板库。
    """
    import cv2
    masks = []
    for fr in frames:
        if fr.mask_path is None:
            raise ValueError(
                f"VGGT 参考帧 {fr.frame_id:06d} 缺前景掩码（mask_path=None）。"
                f"VGGT 路线必须要 GT 前景掩码，请检查 "
                f"{fr.rgb_path.parent.parent}/mask_visib/ 目录，"
                f"或改用 geometry.source=cad。")
        m = cv2.imread(str(fr.mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(
                f"VGGT 参考帧 {fr.frame_id:06d} 掩码文件读取失败: "
                f"{fr.mask_path}（cv2.imread 返回 None，文件可能为空或损坏）。")
        masks.append(m > 0)
    return masks


def _compute_vggt_cad_alignment(recon_pts: np.ndarray, cad_pts: np.ndarray,
                                cfg: Dict, rng) -> tuple:
    """重建系→CAD 系相似变换：FPS 下采样 + ICP 精化 Umeyama。

    重建点云与 CAD 无点对点对应，故用 ICP（迭代最近点建立对应 + 每步 Umeyama
    重解相似变换）。初值取尺度为两点云包围盒对角线之比、旋转恒等、平移为
    质心差，给 ICP 一个合理起点。
    """
    vcfg = cfg["geometry"].get("vggt", {})
    n_fps = int(vcfg.get("n_align_points", 2048))
    icp_iters = int(vcfg.get("icp_iterations", 20))
    recon = np.asarray(recon_pts, dtype=np.float64)
    cad = np.asarray(cad_pts, dtype=np.float64)
    si = farthest_point_sample(recon, n_fps, rng)
    di = farthest_point_sample(cad, n_fps, rng)
    src, dst = recon[si], cad[di]

    def _diag(p):
        return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    s0 = _diag(dst) / max(_diag(src), 1e-9)
    R0 = np.eye(3)
    t0 = dst.mean(axis=0) - s0 * src.mean(axis=0)
    if icp_iters > 0:
        return icp_refine(src, dst, s0, R0, t0, iterations=icp_iters,
                          with_scale=True)
    return s0, R0, t0


def _build_reference_views(cfg: Dict, ds: LinemodDataset,
                           s: float) -> List[Dict]:
    """3DGS 训练视图：参考帧 + GT 位姿 + 可见掩码抠背景。

    参考帧由 ds.reference_frame_ids 确定（有官方 split 时只取 train 划分，
    否则从测试序列均匀抽样并在评测时排除），与 evaluate_object 的排除逻辑
    共用同一来源，杜绝参考/评测泄漏。

    注意尺度一致性：点云被缩放 s 倍后，训练视图的 GT 平移也要同乘 s，
    投影几何才自洽（π 对 (R, s·X, s·t) 与 (R, X, t) 给出同一像素）。

    onboard.train_crop>0 时按 GT 掩码外接框扩 20% 裁剪并 resize 到
    train_crop，K 同步缩放（fx/fy 乘裁剪比例、主点平移）——物体级 3DGS
    监督像素从原图上的几百~几千骤增到 train_crop²，表面细节显著变细，
    模板 3D 锚点精度随之提升（实测 640×480 原图上 ape 仅 ~1600 前景像素，
    训练几乎学不到表面结构）。
    """
    frames = ds.frames()
    n_ref = int(cfg["onboard"].get("n_ref_views", 64))
    bg = float(cfg["onboard"].get("bg_color", 1.0))
    train_crop = int(cfg["onboard"].get("train_crop", 0) or 0)
    depth_on = (float(cfg.get("gaussian", {}).get("depth_l1_weight", 0.0))
                > 0)
    if depth_on:
        from .datasets.ply_io import load_ply
        from .geometry.cad_depth import rasterize_cad_depth
        _verts, _, _faces = load_ply(ds.model_path)
    ref_ids = ds.reference_frame_ids(n_ref)
    views = []
    for fr in frames:
        if fr.frame_id not in ref_ids:
            continue
        img = cv2.cvtColor(cv2.imread(str(fr.rgb_path)), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mask = None
        if fr.mask_path is not None:
            mask = cv2.imread(str(fr.mask_path), cv2.IMREAD_GRAYSCALE) > 0
            img[~mask] = bg
        T = np.eye(4)
        T[:3, :3] = fr.R_gt
        T[:3, 3] = fr.t_gt * s
        width, height = img.shape[1], img.shape[0]
        K = fr.K.copy()
        if train_crop > 0 and fr.mask_path is not None:
            ys, xs = np.nonzero(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            dx, dy = int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)
            x0, y0 = max(0, x0 - dx), max(0, y0 - dy)
            x1, y1 = min(img.shape[1], x1 + dx), min(img.shape[0], y1 + dy)
            crop = img[y0:y1, x0:x1]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = cv2.resize(crop, (train_crop, train_crop),
                                  interpolation=cv2.INTER_LINEAR)
                sx = train_crop / (x1 - x0)
                sy = train_crop / (y1 - y0)
                K = fr.K.copy()
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] = (fr.K[0, 2] - x0) * sx
                K[1, 2] = (fr.K[1, 2] - y0) * sy
                img = crop
                width = height = train_crop
        view = {"image": img, "viewmat": T, "K": K,
                "width": width, "height": height}
        if depth_on:
            # CAD 深度监督：GT 位姿渲染的精确表面深度（掩码外 0）
            depth = rasterize_cad_depth(
                _verts, _faces, fr.R_gt, fr.t_gt * s, K, width)
            d_mask = np.zeros((height, width), dtype=bool)
            if mask is not None:
                if train_crop > 0:
                    m_crop = mask[y0:y1, x0:x1]
                    d_mask = cv2.resize(
                        m_crop.astype(np.uint8), (width, height),
                        interpolation=cv2.INTER_NEAREST) > 0
                else:
                    d_mask = mask
            invdepth = np.zeros((height, width), dtype=np.float32)
            invdepth[d_mask] = 1.0 / np.maximum(depth[d_mask], 1e-3)
            view["invdepth"] = invdepth
            view["depth_mask"] = d_mask
        views.append(view)
    return views


# ---------------------------------------------------------------------------
# 在线：单帧推理
# ---------------------------------------------------------------------------
@dataclass
class EstimateResult:
    success: bool
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 原始模型单位(mm)
    n_inliers: int = 0
    best_template: int = -1
    timings: Dict[str, float] = field(default_factory=dict)     # 分阶段秒
    # 精化诊断用（6d-refiner-v2）：refiner 前的粗位姿与 refiner 后、
    # 回退保护前的精化位姿（同一模型坐标系）。无 refiner 或失败帧为 None。
    R_coarse: Optional[np.ndarray] = None
    t_coarse: Optional[np.ndarray] = None
    R_refined: Optional[np.ndarray] = None
    t_refined: Optional[np.ndarray] = None
    # return_candidates=True 时按择优判据降序的**全部**候选（含失败项）
    # [{"success","R","t","template_idx","n_inliers","score"}, ...]。
    # 成功项的 t 已换回原始模型单位并做过 VGGT→CAD 对齐，与最终输出同一
    # 坐标系；失败项 R/t 为 None，只占 topK 窗口名额（历史对照口径，
    # 见 VERIFICATION.md §8.4）。topK best 评估用。
    candidates: List[Dict] = field(default_factory=list)
    # 实际解码顺序（定位 DINOv2 相似度降序截断到 top_k；mast3r 全解码档为
    # None）。自适应 K 早停离线仿真用：与 cand_inliers 对齐可重建逐模板
    # 内点数随解码的累积轨迹。
    decode_order: Optional[List[int]] = None


class PoseEstimator:
    """在线推理器：加载模板库与各 GPU 组件，逐帧输出位姿。"""

    def __init__(self, cfg: Dict, bank: TemplateBank, device: str = "cuda",
                 refiner_ckpt: Optional[str] = None,
                 symmetric_transforms: Optional[List] = None,
                 stable_prior: Optional[Dict] = None):
        self.cfg = cfg
        self.bank = bank
        self.device = device
        # 稳定摆放先验（任务 1.2 接入点 B）：{"axes": (K,3), "g": (3,)}，
        # None = 不启用。软评分只影响择优排序，不硬过滤。
        self.stable_prior = stable_prior
        # 对称物体（eggbox/glue）的物体系离散对称变换：PnP 内点判定展开
        self._sym_T = symmetric_transforms or []

        seg = cfg["detection"].get("segmenter", "sam")
        if seg == "gt_bbox":
            from .detection.localize import GtBboxLocalizer
            self.localizer = GtBboxLocalizer(cfg["detection"])
            if cfg["detection"].get("gt_bbox_prescreen") == "dinov2":
                # 2c 口径：检测框定位 + DINOv2 模板预筛（模板选择不变）
                from .detection.localize import Dinov2Embedder
                self.localizer.set_dino_prescreen(
                    Dinov2Embedder(cfg["detection"], device=device),
                    self.bank.dino_feats,
                    float(cfg["onboard"].get("bg_color", 1.0)))
            self._loc_mode = "gt_bbox"
        elif seg == "gt_mask":
            from .detection.localize import GtMaskLocalizer
            self.localizer = GtMaskLocalizer(cfg["detection"])
            self._loc_mode = "gt_mask"
        elif seg == "yolo":
            # 历史对照定位路线：YOLO bbox + GT coseg mask（见 VERIFICATION.md §8.2）
            from .detection.localize import YoloBboxLocalizer
            self.localizer = YoloBboxLocalizer(cfg["detection"],
                                               device=device)
            self._loc_mode = "yolo"
        elif seg in ("sam", "fastsam"):
            from .detection.localize import SamDinoLocalizer
            self.localizer = SamDinoLocalizer(
                cfg["detection"], device=device, segmenter=seg,
                bg_color=float(cfg["onboard"].get("bg_color", 1.0)))
            self._loc_mode = "dino"
        else:
            # 显式报错，绝不静默换模型（历史上 fastsam 会悄悄落回 SAM）
            raise ValueError(
                f"未知 detection.segmenter: {seg!r}"
                f"（可选 fastsam / sam / yolo / gt_mask / gt_bbox）")

        # 裁剪方式。context_pad = 默认（bbox 扩 20% 直接裁）；
        # tight_square = 历史对照口径（mask 涂黑 + 方形裁剪 + resize，
        # 见 detection/localize.py legacy_square_crop）
        self._crop_mode = cfg["detection"].get("crop_mode", "context_pad")
        if self._crop_mode not in ("context_pad", "tight_square"):
            raise ValueError(
                f"未知 detection.crop_mode: {self._crop_mode!r}"
                f"（可选 context_pad / tight_square）")

        # 2D-3D 提升路线。coord_map = 默认（alpha 混合坐标图查表）；
        # depth_backproject = 历史对照口径（模板深度图 K_inv 反投影 +
        # 位姿逆变换，见 matching/depth_lifting.py）
        self._lifting = cfg["matching"].get("lifting", "coord_map")
        if self._lifting not in ("coord_map", "depth_backproject"):
            raise ValueError(
                f"未知 matching.lifting: {self._lifting!r}"
                f"（可选 coord_map / depth_backproject）")
        if self._lifting == "depth_backproject" and bank.depth_maps is None:
            raise ValueError(
                "matching.lifting=depth_backproject 需要模板深度图，但模板库"
                "缺 depth_maps。请在 configs 里设 templates.template_source: "
                "depth_map 并重新运行 scripts/data/onboard_object.py。")

        # Top-K 预筛排序来源。dinov2=复用定位相似度只解码 K 个；
        # mast3r=全解码后按 sim(m) 选 Top-K
        self._template_ranking = cfg["matching"].get("template_ranking",
                                                     "dinov2")
        # template_prescreen=none 时跳过 DINOv2 预筛、全模板逐一
        # MASt3R 匹配（历史对照口径）。取值在此即校验（fail fast）
        from .matching.mast3r_wrapper import resolve_prefilter_order
        self._template_prescreen = cfg["matching"].get("template_prescreen",
                                                       "dinov2")
        resolve_prefilter_order(self._template_prescreen,
                                self._template_ranking, None)

        matcher_name = cfg["matching"].get("matcher", "mast3r")
        if matcher_name == "mast3r":
            from .matching.mast3r_wrapper import Mast3rMatcher
            self.matcher = Mast3rMatcher(cfg["matching"], device=device)
        elif matcher_name == "dinov2_patch":
            from .matching.alt_matchers import Dinov2PatchMatcher
            self.matcher = Dinov2PatchMatcher(cfg["matching"], device=device)
        elif matcher_name == "loftr":
            from .matching.alt_matchers import LoFTRMatcher
            self.matcher = LoFTRMatcher(cfg["matching"], device=device)
        elif matcher_name == "lightglue":
            from .matching.alt_matchers import LightGlueMatcher
            self.matcher = LightGlueMatcher(cfg["matching"], device=device)
        else:
            raise ValueError(f"未知匹配器: {matcher_name}")
        # 模板特征预提取缓存（编码器 token 只算一次，供全部帧复用）
        self.matcher.prepare_templates(bank.images, bank.alphas)

        self._rng_seed = int(cfg["runtime"].get("seed", 0))
        self.rng = np.random.default_rng(self._rng_seed)
        # 无 frame_id 调用（脚本/单帧调试）时的兜底计数器
        self._rng_counter = 0

        # 测试时位姿精化（3DGS 可微渲染 + 感知损失，solver.refine_pose 开关）。
        # 需要 onboard 保存的 3DGS 参数 ckpt；缺 ckpt 时构造即抛错（fail fast）。
        self._refiner = None
        if cfg["solver"].get("refine_pose", False):
            from .gaussian.pose_refiner import PoseRefiner
            sc = cfg["solver"]
            self._refiner = PoseRefiner(
                refiner_ckpt, device=device,
                lr=float(sc.get("refine_lr", 0.01)),
                iterations=int(sc.get("refine_iters", 400)),
                lambda_ssim=float(sc.get("refine_lambda_ssim", 0.5)),
                lambda_ms_ssim=float(
                    sc.get("refine_lambda_ms_ssim", 1.0)),
                lambda_mask=float(sc.get("refine_lambda_mask", 0.5)),
                lambda_lpips=float(sc.get("refine_lambda_lpips", 0.0)),
                warmup_steps=int(sc.get("refine_warmup_steps", 10)),
                early_stop_grad_window=int(
                    sc.get("refine_early_stop_grad_window", 5)),
                early_stop_grad_tol=float(
                    sc.get("refine_early_stop_grad_tol", 1e-4)),
                early_stop_patience=int(
                    sc.get("refine_early_stop_patience", 0)),
                early_stop_tol=float(
                    sc.get("refine_early_stop_tol", 1e-4)),
                supersample=int(sc.get("refine_supersample", 1)),
                stage1_iters=int(sc.get("refine_stage1_iters", 0)),
                lambda_area=float(sc.get("refine_area_lambda", 0.0)),
                area_gate_dice=float(
                    sc.get("refine_area_gate_dice", 0.0)))
            # 多假设精化的轻量种子搜索器：SSIM-only 短迭代（快），只负责
            # 在扰动种子里粗筛出好盆地；最终精化仍由 LPIPS 主 refiner 完成
            if bool(sc.get("multi_hypo", False)):
                self._hypo_refiner = PoseRefiner(
                    refiner_ckpt, device=device,
                    lr=float(sc.get("refine_lr", 0.01)),
                    iterations=int(sc.get("multi_hypo_iters", 50)),
                    lambda_ssim=float(sc.get("refine_lambda_ssim", 0.5)),
                    lambda_lpips=0.0,
                    early_stop_patience=15,
                    early_stop_tol=float(
                        sc.get("refine_early_stop_tol", 1e-4)))
            else:
                self._hypo_refiner = None
        # 定位候选渲染验证（detection.loc_verify）：3DGS 前向渲染对齐损失
        # 在 top-K 候选掩码间消歧（不优化，迭代 0）。复用 refiner 实例；
        # refine_pose 关闭时单独构造一个只前向的渲染器。
        self._verifier = None
        if (cfg["detection"].get("loc_verify", True)
                and refiner_ckpt is not None
                and Path(refiner_ckpt).exists()):
            if self._refiner is not None:
                self._verifier = self._refiner
            else:
                from .gaussian.pose_refiner import PoseRefiner
                self._verifier = PoseRefiner(refiner_ckpt, device=device,
                                             iterations=0)

    # ------------------------------------------------------------------
    def _frame_rng(self, frame_id):
        """每帧确定性 rng：种子只依赖帧号（frame_id 派生），cache 命中
        跳帧不消耗流、也不影响其他帧——评估可复现性与 cache 状态无关。

        frame_id 为 None 时退回调用序计数器（脚本/单帧调试路径）。
        """
        if frame_id is None:
            self._rng_counter += 1
            return np.random.default_rng(self._rng_seed + self._rng_counter)
        return np.random.default_rng(self._rng_seed + int(frame_id))

    def extract_matches(self, img_rgb_u8: np.ndarray, K_query: np.ndarray,
                        gt_bbox=None, gt_mask=None, frame_id=None,
                        depth_img=None):
        """阶段 2（粗匹配）：定位 → 裁剪 → MASt3R Top-K 稠密对应。

        与 estimate() 的求解段解耦：对应产物可落盘（scripts/analysis/extract_matches.py
        的逐帧 npz），调 PnP/择优参数时无需重跑最贵的 MASt3R 阶段。

        Returns:
            dict(loc, crop, mask_crop, crop_box_used, s_leg, matches,
                 sxy, timings) 供 _solve() 消费；任一步失败返回 None
        """
        self.rng = self._frame_rng(frame_id)
        timings: Dict[str, float] = {}
        m_cfg = self.cfg["matching"]
        d_cfg = self.cfg["detection"]

        # ---- 步骤 1-3：定位 ----
        t0 = time.time()
        if self._loc_mode == "gt_bbox":
            if gt_bbox is None:
                return None
            # 严格检测框口径（gt_bbox_use_mask: false）：只给框不给掩码，
            # 前景=框内全 1；默认沿袭历史行为（GT coseg mask 提供前景）。
            if not bool(d_cfg.get("gt_bbox_use_mask", True)):
                gt_mask = None
            loc = self.localizer.localize(img_rgb_u8, gt_bbox, gt_mask)
        elif self._loc_mode == "gt_mask":
            if gt_mask is None:
                return None
            loc = self.localizer.localize(img_rgb_u8, gt_mask)
        elif self._loc_mode == "yolo":
            # 历史对照路线：YOLO bbox + GT coseg mask（有 mask 时前景取 mask）
            loc = self.localizer.localize(img_rgb_u8, gt_mask=gt_mask)
        else:
            loc = self.localizer.localize(img_rgb_u8, self.bank.dino_feats)
        timings["localize"] = time.time() - t0
        if loc is None:
            return None

        # ---- 掩码像素交叉实验（6d-mask-erode 拆解）：fastsam 定位（bbox
        # 保持）但前景掩码换 GT——隔离"掩码像素"与"crop 内容"的贡献 ----
        if (bool(d_cfg.get("use_gt_mask_for_fg", False))
                and gt_mask is not None and self._loc_mode != "gt_mask"):
            loc = dataclasses.replace(
                loc, mask=np.asarray(gt_mask, dtype=bool))

        # ---- 反向交叉：GT bbox（crop 内容） + fastsam 掩码像素 ----
        if (bool(d_cfg.get("use_gt_bbox_for_crop", False))
                and gt_mask is not None and self._loc_mode != "gt_mask"):
            gm = np.asarray(gt_mask, dtype=bool)
            ys, xs = np.nonzero(gm)
            if len(xs) < 16:
                return None
            bbox = (int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1))
            h, w = img_rgb_u8.shape[:2]
            from .detection.localize import expand_bbox
            loc = dataclasses.replace(
                loc, crop_box=expand_bbox(
                    bbox, float(d_cfg.get("bbox_expand", 0.2)), w, h))

        # ---- 裁剪（context_pad = 默认；tight_square = 历史对照口径）----
        if self._crop_mode == "tight_square":
            from .detection.localize import legacy_square_crop
            lc = legacy_square_crop(
                img_rgb_u8, loc.mask,
                expand=float(d_cfg.get("crop_expand", 1.1)),
                out_size=int(d_cfg.get("crop_size", 512)))
            if lc is None:
                return None
            crop, mask_crop, crop_box_used, (s_leg_x, s_leg_y) = lc
        else:
            # 掩码边界腐蚀（6d-weak-objects 机制验证）：FastSAM 掩码边界
            # 像素误当前景参与 MASt3R 匹配 → 对应噪声（gt_mask 上界 +5.84
            # 全部归因掩码/crop 内容，候选池消融已排除排序因素）。腐蚀后
            # 默认重算外接框 + expand 0.2（与 GT 掩码路线同构，localize.py:87）；
            # mask_erode_keep_bbox=true 时保持原 crop_box（隔离掩码像素变量）。
            erode = int(d_cfg.get("mask_erode", 0) or 0)
            if erode > 0 and self._loc_mode not in ("gt_mask", "gt_bbox"):
                import cv2
                m = cv2.erode(loc.mask.astype(np.uint8),
                              np.ones((erode, erode), np.uint8)).astype(bool)
                if m.sum() < 16:
                    return None
                if not bool(d_cfg.get("mask_erode_keep_bbox", False)):
                    ys, xs = np.nonzero(m)
                    bbox = (int(xs.min()), int(ys.min()),
                            int(xs.max() - xs.min() + 1),
                            int(ys.max() - ys.min() + 1))
                    h, w = img_rgb_u8.shape[:2]
                    from .detection.localize import expand_bbox
                    loc = dataclasses.replace(
                        loc, mask=m,
                        crop_box=expand_bbox(
                            bbox, float(d_cfg.get("bbox_expand", 0.2)), w, h))
                else:
                    loc = dataclasses.replace(loc, mask=m)
            x0, y0, x1, y1 = loc.crop_box
            crop = img_rgb_u8[y0:y1, x0:x1]
            mask_crop = loc.mask[y0:y1, x0:x1]
            crop_box_used = loc.crop_box
            s_leg_x = s_leg_y = 1.0     # context_pad 无额外缩放

        # ---- 查询裁剪超分（任务 2：M 类病信息预算修复）----
        # 定位后 bbox 裁剪 ×2 再喂 MASt3R：小物体（duck/ape）在 512 输入下
        # 像素太少被插值糊化，超分提供更多高频细节。缩放并入 s_leg 链，
        # back_to_original_pixels 自动映射回原图（对应点坐标无需改动）。
        sr = int(d_cfg.get("crop_sr", 0) or 0)
        if sr:
            from .detection.superres import upscale_crop
            crop, mask_crop, s_sr = upscale_crop(crop, mask_crop, mode=sr)
            s_leg_x *= s_sr
            s_leg_y *= s_sr
        if crop.size == 0 or mask_crop.sum() < 16:
            return None

        # ---- 步骤 4-7：MASt3R 打分 + Top-K 稠密对应 ----
        t0 = time.time()
        # prescreen=none 强制全模板匹配（历史对照口径）；
        # dinov2 时才把定位相似度顺序传给匹配器只解码 Top-K
        from .matching.mast3r_wrapper import resolve_prefilter_order
        prefilter_order = resolve_prefilter_order(
            self._template_prescreen, self._template_ranking,
            loc.template_order)
        # 预筛多样性（matching.prescreen_mmr > 0，08-11 挑战 3）：
        # DINOv2 CLS 相近视角模板高相关，冗余占用 top-k 名额；MMR 贪心
        # 重排（相似 + 互异），攻击"正确模板被挤出池"（36.2% 选错模板帧）。
        mmr_lam = float(m_cfg.get("prescreen_mmr", 0.0) or 0.0)
        if (mmr_lam > 0 and prefilter_order is not None
                and loc.template_sims is not None):
            from .detection.localize import mmr_reorder
            prefilter_order = mmr_reorder(
                prefilter_order, loc.template_sims,
                self.bank.dino_feats, int(m_cfg.get("top_k", 40)), mmr_lam)
        matches, (sx, sy), _scores, top_full = self.matcher.match(
            crop, mask_crop,
            top_k=int(m_cfg.get("top_k", 40)),
            sim_threshold=float(m_cfg.get("sim_threshold", 0.3)),
            cycle_tau_px=float(m_cfg.get("cycle_tau_px", 5.0)),
            n_sample=int(m_cfg.get("n_sample_corr", 4096)),
            rng=self.rng, prefilter_order=prefilter_order)
        timings["matching"] = time.time() - t0
        # top1 模板的稠密 desc（引导式对应精化用，solve 阶段复用）
        top_desc = {"template_idx": top_full[0], "desc_q": top_full[1],
                    "desc_t": top_full[2], "pix_t": top_full[3],
                    "sxy": (sx, sy)} if top_full is not None else None

        # ---- 备选候选匹配（渲染验证消歧的原料）----
        # 仅当 top1 与 top2 分数接近（或 top1 本身低置信）时才启用：
        # 平时不做无用功；近失时对 top-K 候选各跑一遍匹配，_solve 用
        # 3DGS 渲染对齐损失选优（错误 mask 的 crop 里没有目标物体）。
        alts = []
        if (self._loc_mode == "dino"
                and self._verifier is not None
                and len(loc.candidates) >= 2):
            gap = float(d_cfg.get("loc_verify_gap", 0.05))
            min_score = float(d_cfg.get("loc_verify_min_score", 0.35))
            if (loc.candidates[0]["score"] - loc.candidates[1]["score"]
                    < gap or loc.candidates[0]["score"] < min_score):
                t0 = time.time()
                from .detection.localize import expand_bbox
                for cand in loc.candidates[1:]:
                    x, y, bw, bh = cand["bbox_xywh"]
                    cb = expand_bbox((x, y, bw, bh),
                                     float(d_cfg.get("bbox_expand", 0.2)),
                                     img_rgb_u8.shape[1], img_rgb_u8.shape[0])
                    cx0, cy0, cx1, cy1 = cb
                    a_crop = img_rgb_u8[cy0:cy1, cx0:cx1]
                    a_mask = cand["mask"][cy0:cy1, cx0:cx1]
                    if a_crop.size == 0 or a_mask.sum() < 16:
                        continue
                    a_order = resolve_prefilter_order(
                        self._template_prescreen, self._template_ranking,
                        cand["template_order"])
                    a_matches, (asx, asy), _, _ = self.matcher.match(
                        a_crop, a_mask,
                        top_k=min(int(m_cfg.get("top_k", 40)),
                                  int(m_cfg.get("alt_topk", 10))),
                        sim_threshold=float(m_cfg.get("sim_threshold", 0.3)),
                        cycle_tau_px=float(m_cfg.get("cycle_tau_px", 5.0)),
                        n_sample=int(m_cfg.get("n_sample_corr", 4096)),
                        rng=self.rng, prefilter_order=a_order)
                    alts.append({
                        "crop": a_crop, "mask_crop": a_mask,
                        "crop_box_used": cb,
                        "s_leg": (1.0, 1.0), "sxy": (asx, asy),
                        "matches": a_matches,
                        "score": float(cand["score"]),
                    })
                timings["alt_matching"] = time.time() - t0

        return {"loc": loc, "crop": crop, "mask_crop": mask_crop,
                "crop_box_used": crop_box_used,
                "s_leg": (s_leg_x, s_leg_y),
                "matches": matches, "sxy": (sx, sy),
                "top_desc": top_desc,
                "alts": alts, "timings": timings,
                "depth_img": depth_img,
                "decode_order": (
                    list(prefilter_order[:int(m_cfg.get("top_k", 40))])
                    if prefilter_order is not None else None)}

    # ------------------------------------------------------------------
    def _solve_pnp(self, ex: Dict, K_query: np.ndarray):
        """逐模板 RANSAC-PnP → 择优 → 联合 PnP 精化。

        返回 (best: PnPResult, results: List[PnPResult])；全失败返回 None。
        best.R/best.t 为 3D 锚点系的 w2c（缩放系，见 _to_model_frame）。
        """
        s_cfg = self.cfg["solver"]
        m_cfg = self.cfg["matching"]
        # 按 sim(m) 降序：联合 PnP 的 corr_list[:joint_k] 语义就是"分数最高
        # 的 K 个模板"，DINOv2 预筛路径的 matches 顺序是 CLS 序，必须先排序
        # 才一致；ransac_top_templates 限制参与 RANSAC 的模板数（密模板档
        # 下 80 个模板全过阈值会让 CPU 端 RANSAC 拖到 ~6s/帧，而 top-K 之外
        # 的模板对择优几乎无贡献）
        matches = sorted(ex["matches"], key=lambda m: m.score, reverse=True)
        cap = int(s_cfg.get("ransac_top_templates", 0))
        if cap > 0:
            matches = matches[:cap]
        crop_box_used = ex["crop_box_used"]
        s_leg_x, s_leg_y = ex["s_leg"]
        sx, sy = ex["sxy"]

        # ---- 步骤 8：逐模板 RANSAC-PnP ----
        results: List[PnPResult] = []
        # 逐模板 (pts2d, pts3d, pix_t_valid, pix_q_match_valid, tpl_img)，
        # 联合 PnP 精化与 NCC 亚像素分组复用
        corr_list = []
        for m in matches:
            # 查询像素反变换：MASt3R 匹配区 → 裁剪区 → 原图坐标。
            # tight_square 时总缩放 = 匹配 resize × 方形裁剪 resize 的复合
            pts2d = back_to_original_pixels(
                m.pix_q, (sx * s_leg_x, sy * s_leg_y), crop_box_used)
            # 模板像素 → 3D 锚点（两条路线）
            if self._lifting == "depth_backproject":
                # 历史对照口径：深度图 K_inv 反投影 → 位姿逆变换到模型系
                # （见 depth_lifting.py）
                from .matching.depth_lifting import backproject_depth_to_model
                pts3d, valid = backproject_depth_to_model(
                    m.pix_t, self.bank.depth_maps[m.template_idx],
                    self.bank.K, self.bank.poses[m.template_idx],
                    depth_max=m_cfg.get("depth_max"))
            else:
                # 默认：坐标图查表 P = Φ_i(y'*)
                cm = self.bank.coord_maps[m.template_idx]
                xt = m.pix_t[:, 0].astype(int)
                yt = m.pix_t[:, 1].astype(int)
                pts3d = cm[yt, xt]
                # 坐标图无效像素（背景/alpha 过低置 0）剔除
                valid = np.abs(pts3d).sum(axis=1) > 0
            corr_list.append((pts2d[valid], pts3d[valid],
                              m.pix_t[valid], m.pix_q[valid],
                              self.bank.images[m.template_idx]
                              if self.bank.images is not None else None))
            # 查询侧 3D（MASt3R 成对重建，查询相机系）：有则做深度一致性
            # 内点判定（深度+重投影双条件），把 5px 阈值内的错误对应按
            # 3D 深度结构剔除，收紧 tz/rot 条件数（solver.depth_consistency）
            p3q = getattr(m, "pts3d_q", None)
            p3q_ok = (p3q is not None and len(p3q) == len(pts3d)
                      and bool(s_cfg.get("depth_consistency", False)))
            r = ransac_pnp(
                pts2d[valid], pts3d[valid], K_query,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=bool(s_cfg.get("refine_lm", True)),
                min_correspondences=int(s_cfg.get("min_correspondences", 6)),
                flag=str(s_cfg.get("pnp_flag", "epnp")),
                sym_transforms=self._sym_T or None,
                pts3d_q=p3q[valid] if p3q_ok else None,
                depth_tau_frac=float(s_cfg.get("depth_tau_frac", 0.05)),
                pix_t=m.pix_t[valid],
                pix_q_match=m.pix_q[valid],
                pix_scale=(float(sx * s_leg_x), float(sy * s_leg_y)),
                q_img=ex.get("crop"),
                t_img=self.bank.images[m.template_idx]
                if self.bank.images is not None else None,
                subpixel_px=float(s_cfg.get("subpixel_px", 0.0)))
            r.template_idx = m.template_idx
            r.template_score = m.score
            results.append(r)

        # ---- 深度一致性过滤（depth_filter）：掩码面积比预测深度 vs PnP
        # 深度的量级校验。爆炸位姿（t 偏移数百 mm）的深度与物体在掩码中
        # 的投影面积自洽性被破坏：tz_exp = tz_ref·(f_q/f_t)/sqrt(A_q/A_t)
        # （GS-Pose eq.3-4），候选 t_z 超出 [lo, hi]×tz_exp 判废——从源头
        # 阻止坏候选进入 inlier 择优（爆炸帧候选池常全部是坏候选）。
        if bool(s_cfg.get("depth_filter", False)) and ex.get("mask_crop") is not None:
            mask = ex["mask_crop"]
            A_q = float(mask.sum())
            if A_q >= 16:
                lo = float(s_cfg.get("depth_filter_lo", 0.2))
                hi = float(s_cfg.get("depth_filter_hi", 5.0))
                fq = float(K_query[0, 0])
                ft = float(self.bank.K[0, 0])
                for r in results:
                    if not r.success:
                        continue
                    a_t = self.bank.alphas[r.template_idx] > 0.5
                    if a_t.sum() < 16:
                        continue
                    tz_ref = float(self.bank.poses[r.template_idx][2, 3])
                    ratio = np.sqrt(A_q / float(a_t.sum()))
                    tz_exp = tz_ref * (fq / ft) / max(ratio, 1e-3)
                    tz = float(r.t[2])
                    if tz < lo * tz_exp or tz > hi * tz_exp:
                        r.success = False

        # ---- 步骤 9-10：几何一致性择优 ----
        strategy = s_cfg.get("selection", "inlier")
        prior_info = None
        if strategy == "prior_inlier" and self.stable_prior is not None:
            prior_info = (self.stable_prior["axes"], self.stable_prior["g"],
                          float(s_cfg.get("prior_lambda", 0.5)))
        ranked = rank_candidates(results, strategy=strategy,
                                 prior_info=prior_info)
        if not ranked:
            return None
        best = ranked[0]
        # 模板层解集共识（selection=consensus）：inlier 择优可能选中"自洽地错"
        # 的错误模板解（内点可超 1000）；正确模板的解在 3D 空间收敛成簇，
        # 错误解随机分散。最大簇（≥2）内 inlier 最大者替换 best；无簇不换
        # （安全门控：池内无好假设的物体不受损）。joint PnP 的替换门控随之
        # 收紧：共识生效时，joint 解须与共识解位姿接近（同盆地）才允许覆盖。
        consensus_active = False
        cb = None
        if strategy == "consensus":
            from .solver.selection import consensus_best
            cb = consensus_best(
                results,
                rot_tau_deg=float(s_cfg.get("consensus_rot_tau", 10.0)),
                trans_tau_mm=float(s_cfg.get("consensus_trans_tau", 25.0)),
                sym_transforms=self._sym_T or None)
            if cb is not None and (cb.template_idx != best.template_idx
                                   or cb.n_inliers > best.n_inliers):
                best = cb
                consensus_active = True

        # 联合 PnP 精化：单模板视角的对应集中在物体可见面（近平面点集），
        # EPnP 存在深度/旋转歧义；把 sim 分数最高的 joint_templates 个模板
        # 的对应合并重解，点集变立体后位姿显著更稳（实测 ADD 误差约减半）。
        # 只替换最终输出位姿，候选窗口与模板归属保持不变（消融口径不破）。
        joint_k = int(s_cfg.get("joint_templates", 3))
        if joint_k >= 2 and len(corr_list) >= joint_k:
            j2 = np.concatenate([c[0] for c in corr_list[:joint_k]])
            j3 = np.concatenate([c[1] for c in corr_list[:joint_k]])
            r_j = ransac_pnp(
                j2, j3, K_query,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=bool(s_cfg.get("refine_lm", True)),
                min_correspondences=int(s_cfg.get("min_correspondences", 6)),
                flag=str(s_cfg.get("pnp_flag", "epnp")),
                sym_transforms=self._sym_T or None)
            if r_j.success and r_j.n_inliers >= best.n_inliers:
                # 共识档：joint 解须与共识解同盆地（位姿接近）才允许覆盖，
                # 防合并对应把共识解换坏（multi 系"无条件换种子"教训）。
                # 对称物体按等价位姿类判定（与 consensus_best 聚类同口径）。
                if consensus_active:
                    from .solver.selection import _pose_neighbor
                    if not _pose_neighbor(
                            r_j.R, r_j.t, cb.R, cb.t,
                            float(s_cfg.get("consensus_rot_tau", 10.0)),
                            float(s_cfg.get("consensus_trans_tau", 25.0)),
                            self._sym_T or None):
                        r_j.success = False
                # joint 结果同样过深度一致性（复用 best 模板的深度预测）
                if (bool(s_cfg.get("depth_filter", False))
                        and ex.get("mask_crop") is not None and A_q >= 16):
                    a_t = self.bank.alphas[best.template_idx] > 0.5
                    if a_t.sum() >= 16:
                        tz_ref = float(
                            self.bank.poses[best.template_idx][2, 3])
                        ratio = np.sqrt(A_q / float(a_t.sum()))
                        tz_exp = tz_ref * (fq / ft) / max(ratio, 1e-3)
                        tz = float(r_j.t[2])
                        if tz < lo * tz_exp or tz > hi * tz_exp:
                            r_j.success = False
                if r_j.success:
                    # NCC 亚像素精化（联合路径，按模板分组）：内点按模板
                    # 切分，各自用对应模板图做 NCC，偏移转原图系后全量 LM
                    sp = float(s_cfg.get("subpixel_px", 0.0))
                    q_img = ex.get("crop")
                    if sp > 0 and q_img is not None:
                        r_j = self._joint_subpixel_refine(
                            r_j, corr_list[:joint_k], K_query, sp, q_img,
                            (float(sx * s_leg_x), float(sy * s_leg_y)))
                    r_j.template_idx = best.template_idx
                    r_j.template_score = best.template_score
                    best = r_j
        return best, results

    def _joint_subpixel_refine(self, r_j, corr_k, K_query, sp, q_img,
                               pix_scale):
        """联合 PnP 结果的 NCC 亚像素精化（按模板分组）。

        内点按模板切分，各自用对应模板图做 NCC（匹配系），偏移按
        pix_scale 转原图系后全量 LM 重解。
        """
        from .solver.ransac_pnp import PnPResult, _ncc_subpixel_refine
        import cv2
        idx = r_j.inlier_idx
        if len(idx) < 6:
            return r_j
        all_p2 = np.empty((len(idx), 2))
        all_p3 = np.empty((len(idx), 3))
        off = np.zeros((len(idx), 2))
        start = 0
        for j2k, j3k, ptk, pqk, tpl_img in corr_k:
            nk = len(j2k)
            in_tpl = (idx >= start) & (idx < start + nk)
            gpos = np.nonzero(in_tpl)[0]     # 内点序号（all_* 行索引）
            start += nk
            if len(gpos) == 0:
                continue
            lk = idx[in_tpl] - (start - nk)  # 模板内局部下标
            all_p2[gpos] = j2k[lk]
            all_p3[gpos] = j3k[lk]
            if tpl_img is not None and len(lk) >= 4:
                pqm_new = _ncc_subpixel_refine(pqk[lk], ptk[lk],
                                               q_img, tpl_img)
                off[gpos] = pqm_new - pqk[lk]
        moved = np.linalg.norm(off, axis=1)
        m = moved > sp
        if m.sum() < 4:
            return r_j
        p2n = all_p2.copy()
        p2n[m, 0] += off[m, 0] / pix_scale[0]
        p2n[m, 1] += off[m, 1] / pix_scale[1]
        rvec, _ = cv2.Rodrigues(r_j.R)
        tvec = r_j.t.reshape(3, 1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                all_p3, p2n.reshape(-1, 1, 2), K_query, None, rvec, tvec)
        except cv2.error:
            return r_j
        R, _ = cv2.Rodrigues(rvec)
        uv, _ = cv2.projectPoints(all_p3, rvec, tvec, K_query, None)
        residual = np.linalg.norm(uv.reshape(-1, 2) - p2n, axis=1)
        return PnPResult(success=True, R=R, t=tvec.reshape(3),
                         n_inliers=len(idx), inlier_idx=idx,
                         n_correspondences=r_j.n_correspondences,
                         mean_inlier_reproj_px=float(residual.mean()))

    # ------------------------------------------------------------------
    def mask_ratio_init(self, ex: Dict, K_query: np.ndarray):
        """GS-Pose 式掩码解析平移初始化（无 PnP）。

        旋转直接取检索到的 top1 模板（matches[0]），平移由掩码面积比定深度、
        掩码中心定射线（GS-Pose eq.3-4 的零样本版）：
            t_z_q = t_z_ref · (f_q/f_t) / sqrt(A_q/A_t)，t_q = t_z_q·K_q⁻¹·p̄
        物体系原点（物体中心）在模板相机系深度恰为 T[2,3]，面积比给出查询
        深度（单位同为锚点系 mm）；不需要任何 2D-3D 对应。

        Returns:
            (R, t) 锚点系 w2c；无掩码/无模板 alpha 时返回 None
        """
        if not ex.get("matches"):
            return None
        idx = int(ex["matches"][0].template_idx)
        mask = ex.get("mask_crop")
        if mask is None or mask.sum() < 16:
            return None
        a_t = self.bank.alphas[idx] > 0.5
        if a_t.sum() < 16:
            return None
        x0, y0, _, _ = ex["crop_box_used"]
        ys, xs = np.nonzero(mask)
        # 掩码外接框中心（原图坐标），与 GS-Pose 的 C_bbox 一致
        cx_q = x0 + float(xs.min() + xs.max()) / 2.0
        cy_q = y0 + float(ys.min() + ys.max()) / 2.0
        ratio = np.sqrt(float(mask.sum()) / float(a_t.sum()))
        T = self.bank.poses[idx]
        tz_ref = float(T[2, 3])
        tz_q = tz_ref * (float(K_query[0, 0]) / float(self.bank.K[0, 0])
                         ) / max(ratio, 1e-3)
        R = np.asarray(T[:3, :3], dtype=np.float64)
        t = tz_q * (np.linalg.inv(K_query) @ np.array(
            [cx_q, cy_q, 1.0], dtype=np.float64))
        return R, t

    # ------------------------------------------------------------------
    def _decode_top_desc(self, ex: Dict):
        """重解码 top1 模板的稠密 desc（引导式精化用，落盘路径按需调用）。"""
        if not ex.get("matches"):
            return None
        if not hasattr(self.matcher, "_decode_batch"):
            return None   # LightGlue 等稀疏匹配器无稠密解码
        m = ex["matches"][0]
        idx = int(m.template_idx)
        from .matching.mast3r_wrapper import _resize_to_multiple16
        q_img, (sx, sy) = _resize_to_multiple16(
            ex["crop"], int(self.cfg["matching"].get("image_size", 512)))
        fq, pq, sq = self.matcher._encode(q_img)
        for i, dq, dt, _, _, _, _ in self.matcher._decode_batch(fq, pq, sq, [idx]):
            fg = self.matcher._tmpl_fg[idx]
            tys, txs = np.nonzero(fg)
            if len(tys) == 0:
                return None
            flat = tys * fg.shape[1] + txs
            dt_fg = dt.reshape(-1, dt.shape[-1])[torch.tensor(
                flat, device=self.device)]
            return {"template_idx": idx,
                    "desc_q": dq.float().cpu().numpy().astype(np.float16),
                    "desc_t": dt_fg.half().cpu().numpy(),
                    "pix_t": (np.stack([txs, tys], axis=1)
                              * self.matcher._tmpl_scale),
                    "sxy": (sx, sy)}
        return None

    # ------------------------------------------------------------------
    def _iter_align(self, ex: Dict, K_query: np.ndarray,
                    R: np.ndarray, t: np.ndarray):
        """迭代渲染对齐（6d-iter-align）：在当前位姿处重渲染 3DGS 新视角，
        与查询裁剪重做 MASt3R 稠密对应 → 渲染深度反投影到模型系 → 重解
        PnP，迭代 iter_align_iters 轮。位姿越准渲染越像真图 → 域差随迭代
        收缩 → 对应质量递增（首轮匹配是在离线模板视角上做的，当前位姿
        渲染视角天然更接近查询视角）。接受/拒绝：渲染对齐损失变差回退。

        Returns:
            (R, t, n_inliers)：最后成功一轮 PnP 的内点数（multi 择优
            几何基准用；无 PnP 时 -1）。
        """
        s_cfg = self.cfg["solver"]
        iters = int(s_cfg.get("iter_align_iters", 0) or 0)
        if iters <= 0 or self._refiner is None:
            return R, t, -1
        if not hasattr(self.matcher, "model"):
            return R, t, -1   # LightGlue 等稀疏匹配器无成对解码器
        import cv2
        from .matching.mast3r_wrapper import _resize_to_multiple16
        from .matching.correspondence import mutual_nn_matches
        from .solver.ransac_pnp import ransac_pnp

        x0, y0, _, _ = ex["crop_box_used"]
        K_crop = K_query.copy()
        K_crop[0, 2] -= x0
        K_crop[1, 2] -= y0
        q_img, (sx, sy) = _resize_to_multiple16(
            ex["crop"], self.matcher.long_side)
        K512 = K_crop.copy()
        K512[0] *= sx
        K512[1] *= sy
        q_mask = cv2.resize(
            ex["mask_crop"].astype(np.uint8),
            (q_img.shape[1], q_img.shape[0]),
            interpolation=cv2.INTER_NEAREST).astype(bool)

        Rc = np.asarray(R, dtype=np.float64)
        tc = np.asarray(t, dtype=np.float64)
        device = self._refiner.device
        last_inl = -1
        for _ in range(iters):
            rgb, alpha, depth = self._refiner.render_rgbd(
                torch.tensor(Rc, dtype=torch.float32, device=device),
                torch.tensor(tc, dtype=torch.float32, device=device),
                torch.tensor(K512, dtype=torch.float32, device=device),
                q_img.shape[1], q_img.shape[0])
            rend_mask = alpha > 0.5
            if rend_mask.sum() < 16:
                break
            fq, pq, sq = self.matcher._encode(q_img)
            fr, pr, sr = self.matcher._encode(rgb)
            with torch.no_grad():
                dec1, dec2 = self.matcher.model._decoder(fq, pq, fr, pr)
                with torch.autocast("cuda", enabled=False):
                    res1 = self.matcher.model._downstream_head(
                        1, [tok.float() for tok in dec1], sq)
                    res2 = self.matcher.model._downstream_head(
                        2, [tok.float() for tok in dec2], sr)
            desc_q = res1["desc"][0].reshape(-1, res1["desc"].shape[-1])
            desc_r = res2["desc"][0].reshape(-1, res2["desc"].shape[-1])
            ys_q, xs_q = np.nonzero(q_mask)
            ys_r, xs_r = np.nonzero(rend_mask)
            if len(ys_q) < 16 or len(ys_r) < 16:
                break
            # 采样上限：全量互最近邻是 Nq×Nr 点积，小物体上万前景像素时
            # 单帧几十 GB 内存（事故 08-08）——与 matcher n_score_pixels 同纪律
            n_max = int(self.cfg["matching"].get("n_sample_corr", 4096))
            if len(ys_q) > n_max:
                sub = self.rng.choice(len(ys_q), n_max, replace=False)
                ys_q, xs_q = ys_q[sub], xs_q[sub]
            if len(ys_r) > n_max:
                sub = self.rng.choice(len(ys_r), n_max, replace=False)
                ys_r, xs_r = ys_r[sub], xs_r[sub]
            flat_q = ys_q * q_img.shape[1] + xs_q
            flat_r = ys_r * rgb.shape[1] + xs_r
            iq, ir, _ = mutual_nn_matches(
                desc_q[torch.tensor(flat_q, device=device)].float().cpu().numpy(),
                desc_r[torch.tensor(flat_r, device=device)].float().cpu().numpy())
            if len(iq) < 8:
                break
            # 渲染深度反投影 → 相机系 3D → 模型系（渲染位姿 w2c 逆变换）
            z = depth[ys_r[ir], xs_r[ir]]
            valid = (z > 1e-3) & np.isfinite(z)
            if valid.sum() < 8:
                break
            pix_q = np.stack([xs_q[iq][valid], ys_q[iq][valid]],
                             axis=1).astype(np.float64)
            u = xs_r[ir][valid].astype(np.float64)
            v = ys_r[ir][valid].astype(np.float64)
            z = z[valid]
            fx, fy = K512[0, 0], K512[1, 1]
            cx, cy = K512[0, 2], K512[1, 2]
            Xc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
            Xm = (Rc.T @ (Xc - tc).T).T
            res = ransac_pnp(
                pix_q, Xm, K512,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=True,
                min_correspondences=6,
                flag=str(s_cfg.get("pnp_flag", "epnp")))
            if res is None or not res.success:
                break
            Rc, tc = res.R, res.t
            last_inl = res.n_inliers
        # 接受/拒绝：渲染对齐损失变差回退粗位姿（与 guided_refine 同纪律）
        if self._verifier is not None:
            l_before = self._verifier.align_loss(
                ex["crop"], ex["mask_crop"], K_crop, R, t)
            l_after = self._verifier.align_loss(
                ex["crop"], ex["mask_crop"], K_crop, Rc, tc)
            if l_after > l_before:
                return R, t, last_inl
        return Rc, tc, last_inl

    # ------------------------------------------------------------------
    def _guided_refine(self, ex: Dict, K_query: np.ndarray,
                       R: np.ndarray, t: np.ndarray):
        """引导式对应精化：粗位姿投影 → 局部窗口 desc 重匹配 → PnP 迭代。

        用 top1 模板的稠密 desc（extract 落盘）在粗位姿投影位置的局部
        窗口内重新搜索对应，把重复纹理下错位一个格的对应拉回正确位置；
        新对应重解 PnP 后迭代（窗口逐轮收缩）。只返回成功精化后的位姿，
        失败时原样返回粗位姿。
        """
        s_cfg = self.cfg["solver"]
        iters = int(s_cfg.get("guided_iters", 2))
        radius = int(s_cfg.get("guided_radius", 12))
        if iters <= 0:
            return R, t
        td = ex.get("top_desc")
        if td is None:
            # 落盘路径（extract 不落盘稠密 desc，避免 9MB/帧）：重解码
            # top1 模板的稠密 desc（一次编码+解码 ~1s）
            td = self._decode_top_desc(ex)
            if td is None:
                return R, t
        bank = self.bank
        cm = bank.coord_maps[td["template_idx"]]
        pix_t = np.asarray(td["pix_t"], dtype=int)
        pts3d_t = cm[pix_t[:, 1], pix_t[:, 0]]        # (Nt,3) 缩放系
        valid = np.abs(pts3d_t).sum(axis=1) > 1e-6
        if valid.sum() < 8:
            return R, t
        pix_t = pix_t[valid]
        pts3d_t = pts3d_t[valid].astype(np.float64)
        desc_t = np.asarray(td["desc_t"], dtype=np.float32)[valid]
        x0, y0, _, _ = ex["crop_box_used"]
        K_crop = K_query.copy()
        K_crop[0, 2] -= x0
        K_crop[1, 2] -= y0
        from .matching.correspondence import guided_local_matching
        Rc, tc = np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)
        r_now = radius
        for _ in range(iters):
            p2, p3, sims = guided_local_matching(
                td["desc_q"], desc_t, pix_t, pts3d_t, Rc, tc, K_crop,
                td["sxy"], (x0, y0), r=r_now)
            if len(p2) < 8:
                break
            rn = ransac_pnp(
                p2, p3, K_query,
                reproj_px=float(s_cfg.get("ransac_reproj_px", 5.0)),
                confidence=float(s_cfg.get("ransac_confidence", 0.999)),
                iterations=int(s_cfg.get("ransac_iterations", 1000)),
                refine_lm=bool(s_cfg.get("refine_lm", True)),
                min_correspondences=int(s_cfg.get("min_correspondences", 6)),
                flag=str(s_cfg.get("pnp_flag", "epnp")),
                sym_transforms=self._sym_T or None)
            if not rn.success or rn.n_inliers < 8:
                break
            Rc, tc = rn.R, rn.t
            r_now = max(r_now - 3, 4)
        return Rc, tc

    # ------------------------------------------------------------------
    def _multi_hypothesis_refine(self, ex: Dict, K_crop: np.ndarray,
                                 R0: np.ndarray, t0: np.ndarray):
        """多假设精化：扰动出多个种子 → SSIM 短精化 → 渲染损失选优。

        单次精化从错误盆地出发时收敛到局部极小；对位姿加旋转/平移扰动
        生成多个种子（借鉴旧 MyPose refine.py generate_hypotheses），
        各自独立短精化后按 3DGS 渲染对齐损失选最优盆地。种子搜索用
        SSIM-only 轻量精化器（快），返回胜者由主 LPIPS refiner 收尾。
        """
        s_cfg = self.cfg["solver"]
        n_seeds = int(s_cfg.get("multi_hypo_seeds", 6))
        rot_deg = float(s_cfg.get("multi_hypo_rot_deg", 12.0))
        trans_mm = float(s_cfg.get("multi_hypo_trans_mm", 15.0))
        rng = self.rng
        import cv2
        seeds = [(np.asarray(R0, dtype=np.float64),
                  np.asarray(t0, dtype=np.float64))]
        for _ in range(n_seeds - 1):
            axis = rng.standard_normal(3)
            axis /= max(float(np.linalg.norm(axis)), 1e-8)
            ang = np.radians(rng.uniform(rot_deg * 0.5, rot_deg))
            dR, _ = cv2.Rodrigues(axis * ang)
            seeds.append((dR @ R0, t0 + rng.uniform(
                -trans_mm, trans_mm, 3)))
        best = (np.asarray(R0, dtype=np.float64),
                np.asarray(t0, dtype=np.float64))
        best_loss = float("inf")
        for R_s, t_s in seeds:
            R_r, t_r = self._hypo_refiner.refine(
                ex["crop"], ex["mask_crop"], K_crop, R_s, t_s)
            if R_r is None:
                continue
            loss = self._verifier.align_loss(
                ex["crop"], ex["mask_crop"], K_crop, R_r, t_r)
            if loss < best_loss:
                best_loss, best = loss, (R_r, t_r)
        return best

    def _mask_geo_translation(self, ex: Dict, K_query: np.ndarray,
                              template_idx) -> Optional[np.ndarray]:
        """掩码几何解析平移（GS-Pose §3.2 面积比 + 中心反投影）。

        z 由掩码面积比定尺度（模板球面半径已知：z_q = z_ref·sqrt(A_q/A_ref)），
        xy 由查询掩码质心反投影。旋转由调用方提供，只替换平移——对症
        RANSAC 深度病态（小物体 tz 错 30-40mm 重投影偏移 <5px 不可分辨；
        全量失败帧 48.5% 纯 t 错，gap 分解 08-11）。返回 None 表示无法构造。
        """
        bank = self.bank
        alpha = None
        if (template_idx is not None and template_idx >= 0
                and getattr(bank, "alphas", None) is not None
                and template_idx < len(bank.alphas)):
            alpha = np.asarray(bank.alphas[template_idx])
        if alpha is None:
            return None
        mask = np.asarray(ex["mask_crop"]) > 0
        A_q = float(mask.sum())
        A_ref = float((alpha > 0.5).sum())
        if A_q < 16 or A_ref < 16:
            return None
        z_ref = float(np.linalg.norm(bank.poses[template_idx][:3, 3]))
        if not np.isfinite(z_ref) or z_ref <= 0:
            return None
        z_q = z_ref * np.sqrt(A_q / A_ref)
        x0, y0, _, _ = ex["crop_box_used"]
        my, mx = np.nonzero(mask)
        cx_g = float(mx.mean()) + x0
        cy_g = float(my.mean()) + y0
        K_inv = np.linalg.inv(np.asarray(K_query, dtype=np.float64))
        return K_inv @ np.array([cx_g * z_q, cy_g * z_q, z_q])

    # ------------------------------------------------------------------
    def _solve(self, ex: Dict, K_query: np.ndarray,
               return_candidates: bool = False, frame_id=None,
               prev_pose=None, _depth: int = 0) -> EstimateResult:
        """阶段 3（求解优化）：逐模板 PnP → 择优 → 联合 PnP 精化 → refiner。

        消费 extract_matches() 的产物（也可从落盘 npz 重建后传入，见
        scripts/solve_poses.py），保证两阶段共用同一求解逻辑。

        定位候选渲染验证：主候选与备选候选（extract_matches 的 alts）各
        解出位姿后，用 3DGS 前向渲染对齐损失选优——错误候选的 crop 里
        没有目标物体，渲染内容对不上，损失显著更高。
        """
        if frame_id is not None:
            # 阶段 3 直解路径（matches_dir）不经 extract_matches，同样按帧
            # 重置 rng 保证可复现（estimate 路径这里再重置一次同种子流，无害）
            self.rng = self._frame_rng(frame_id)
        timings = dict(ex["timings"])
        s_cfg = self.cfg["solver"]
        m_cfg = self.cfg["matching"]

        t0 = time.time()
        solved = self._solve_pnp(ex, K_query)
        if solved is None:
            return EstimateResult(success=False, timings=timings)
        best, results = solved
        timings["pnp"] = time.time() - t0

        # ---- 渲染择优（render_select / render_align_select）----
        # render_select：mask IoU 条件触发式（IoU<0.4 才在 top-K 池重选
        # IoU 最高的候选）。注意 IoU 对 tz 方向爆炸不敏感（tz 偏 300mm
        # 掩码缩放后 IoU 仍 >0.4，can 事故 08-04）。
        # render_align_select：每帧直接对 [best + ranked top-K] 算渲染
        # 对齐损失（L1+SSIM，内容错位敏感）选最小——tz 爆炸位姿渲染与
        # 真实图完全错位，损失显著高于正确位姿（区分度实测见
        # scripts/analysis/verify_align_select.py）。每候选 ~30ms 前向渲染。
        if (bool(s_cfg.get("render_align_select", False))
                and self._verifier is not None):
            t0 = time.time()
            x0, y0, _, _ = ex["crop_box_used"]
            K_crop = K_query.copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            sel_n = int(s_cfg.get("render_select_n", 5))
            ranked = rank_candidates(
                results, strategy=s_cfg.get("selection", "inlier"))
            cands = [best] + [r for r in ranked[:sel_n] if r is not best]
            best_la, best_r = float("inf"), best
            for r in cands:
                la = self._verifier.align_loss(ex["crop"], ex["mask_crop"],
                                               K_crop, r.R, r.t)
                if la < best_la:
                    best_la, best_r = la, r
            if best_r is not best:
                best = best_r
            iou_min = float(s_cfg.get("render_select_min", 0.4))
            rs_iou = self._verifier.mask_iou(best.R, best.t, K_crop,
                                             ex["mask_crop"])
            rs_triggered = rs_iou < iou_min
            timings["render_select"] = time.time() - t0
        elif (bool(s_cfg.get("render_select", False))
                and self._verifier is not None):
            t0 = time.time()
            x0, y0, _, _ = ex["crop_box_used"]
            K_crop = K_query.copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            iou_min = float(s_cfg.get("render_select_min", 0.4))
            iou_best = self._verifier.mask_iou(best.R, best.t, K_crop,
                                               ex["mask_crop"])
            rs_triggered = iou_best < iou_min
            rs_iou = iou_best
            if rs_triggered:
                iou_n = int(s_cfg.get("render_select_n", 5))
                ranked = rank_candidates(
                    results, strategy=s_cfg.get("selection", "inlier"))
                best_iou, best_r = iou_best, best
                for r in [best] + ranked[:iou_n]:
                    iou = self._verifier.mask_iou(r.R, r.t, K_crop,
                                                  ex["mask_crop"])
                    if iou > best_iou:
                        best_iou, best_r = iou, r
                if best_r is not best:
                    best = best_r
                rs_iou = best_iou
            timings["render_select"] = time.time() - t0
        else:
            rs_triggered, rs_iou = False, 1.0

        # ---- 自适应全解码（6d-fallback-decode）：渲染验证失败帧第二意见 ----
        # 36.2% 失败帧 R>60° 选错模板（正确模板可能在 DINOv2 top-40 池外）；
        # duck 全解码历史收益 +6.67（6d-cand-pool 干净口径）。只在渲染验证
        # 触发（IoU 低）时对全部模板重匹配重解，渲染对齐损失择优；递归保护
        # （_depth）防自触发。代价：失败帧 ~22% 付全解码（+3.7s）与重求解。
        if (rs_triggered and _depth == 0
                and bool(s_cfg.get("fallback_decode", False))):
            t0 = time.time()
            m_cfg = self.cfg["matching"]
            fb_matches, (fb_sx, fb_sy), fb_scores, _ = self.matcher.match(
                ex["crop"], ex["mask_crop"],
                top_k=len(self.bank.alphas),
                sim_threshold=float(m_cfg.get("sim_threshold", 0.3)),
                cycle_tau_px=float(m_cfg.get("cycle_tau_px", 5.0)),
                n_sample=int(m_cfg.get("n_sample_corr", 4096)),
                rng=self.rng, prefilter_order=None)
            if fb_matches:
                ex_fb = dict(ex)
                ex_fb["matches"] = fb_matches
                ex_fb["sxy"] = (fb_sx, fb_sy)
                res_fb = self._solve(ex_fb, K_query,
                                     return_candidates=False,
                                     frame_id=frame_id,
                                     prev_pose=prev_pose, _depth=1)
                if res_fb.success:
                    x0, y0, _, _ = ex["crop_box_used"]
                    K_crop = K_query.copy()
                    K_crop[0, 2] -= x0
                    K_crop[1, 2] -= y0
                    l_cur = self._verifier.align_loss(
                        ex["crop"], ex["mask_crop"], K_crop,
                        best.R, best.t)
                    l_fb = self._verifier.align_loss(
                        ex["crop"], ex["mask_crop"], K_crop,
                        res_fb.R, res_fb.t)
                    if l_fb < l_cur:
                        best = SimpleNamespace(
                            R=res_fb.R, t=res_fb.t,
                            n_inliers=getattr(res_fb, "n_inliers", 0),
                            template_idx=-1)
                timings["fallback_decode"] = time.time() - t0

        # ---- 备选候选：逐候选 PnP + 渲染验证消歧 ----
        chosen, chosen_ex = best, ex
        # ---- 掩码几何候选（6d-mask-geo）：模板旋转 + 掩码面积/质心解析平移 ----
        # RANSAC 深度病态（小物体 tz 错 30-40mm 重投影 <5px 不可分辨）使 PnP
        # 平移不可靠（全量失败帧 48.5% 纯 t 错）；掩码面积比 z + 质心 xy 直接
        # 来自掩码几何（GS-Pose §3.2 同款）。作为候选与 PnP 结果用渲染对齐
        # 损失竞争，不进 IoU 接受门（tz_search 判据不可分结案）。
        if (bool(s_cfg.get("mask_geo_candidate", False))
                and self._verifier is not None):
            t0 = time.time()
            t_geo = self._mask_geo_translation(ex, K_query,
                                               best.template_idx)
            if t_geo is not None:
                x0, y0, _, _ = ex["crop_box_used"]
                K_crop = K_query.copy()
                K_crop[0, 2] -= x0
                K_crop[1, 2] -= y0
                l_pnp = self._verifier.align_loss(
                    ex["crop"], ex["mask_crop"], K_crop, best.R, best.t)
                l_geo = self._verifier.align_loss(
                    ex["crop"], ex["mask_crop"], K_crop, best.R, t_geo)
                if l_geo < l_pnp:
                    chosen = SimpleNamespace(
                        R=best.R, t=t_geo, n_inliers=best.n_inliers,
                        template_idx=best.template_idx)
            timings["mask_geo"] = time.time() - t0
        if ex.get("alts"):
            alt_solved = []
            for a in ex["alts"]:
                a_ex = {"crop": a["crop"], "mask_crop": a["mask_crop"],
                        "crop_box_used": a["crop_box_used"],
                        "s_leg": a["s_leg"], "sxy": a["sxy"],
                        "matches": a["matches"]}
                s = self._solve_pnp(a_ex, K_query)
                if s is not None:
                    alt_solved.append((a_ex, s[0]))
            if alt_solved and self._verifier is not None:
                t0 = time.time()
                cands = [(ex, best)] + alt_solved
                best_loss = float("inf")
                for cex, r in cands:
                    x0, y0, _, _ = cex["crop_box_used"]
                    K_crop = K_query.copy()
                    K_crop[0, 2] -= x0
                    K_crop[1, 2] -= y0
                    loss = self._verifier.align_loss(
                        cex["crop"], cex["mask_crop"], K_crop, r.R, r.t)
                    if loss < best_loss:
                        best_loss, chosen, chosen_ex = loss, r, cex
                timings["loc_verify"] = time.time() - t0

        candidates: List[Dict] = []
        if return_candidates:
            # topK best 评估的候选窗口：**保留失败候选占名额**
            # （keep_failed=True，历史对照口径，见 VERIFICATION.md §8.4）。
            # 这里若只放成功候选，top3/top5 会系统性偏乐观。
            # 主路线的 best 仍取自上面只含成功候选的 ranked，行为不变。
            for r in rank_candidates(results, strategy=s_cfg.get("selection", "inlier"),
                                     keep_failed=True):
                if r.success:
                    R_c, t_c = self._to_model_frame(r.R, r.t)
                else:
                    R_c, t_c = None, None
                candidates.append({
                    "success": bool(r.success),
                    "R": R_c, "t": t_c, "template_idx": r.template_idx,
                    "n_inliers": r.n_inliers, "score": r.template_score,
                    # 判据阶梯的原始量（离线重排/分析用，免得换判据重跑管线）
                    "n_correspondences": r.n_correspondences,
                    "mean_inlier_reproj_px": (
                        None if not np.isfinite(r.mean_inlier_reproj_px)
                        else float(r.mean_inlier_reproj_px)),
                })

        R_c, t_c = chosen.R, chosen.t
        # 引导式对应精化（机制改进）：粗位姿局部窗口重匹配 → PnP 迭代，
        # 把重复纹理下的错位对应拉回。用 3DGS 渲染对齐损失做接受/拒绝：
        # 精化后渲染 loss 更小才接受（窗口内无正确对应时精化位姿可能
        # 更差，不能无条件替换）
        if s_cfg.get("guided_refine", True):
            t0 = time.time()
            R_g, t_g = self._guided_refine(chosen_ex, K_query, R_c, t_c)
            if self._verifier is not None and not (
                    np.allclose(R_g, R_c) and np.allclose(t_g, t_c)):
                x0, y0, _, _ = chosen_ex["crop_box_used"]
                K_crop = K_query.copy()
                K_crop[0, 2] -= x0
                K_crop[1, 2] -= y0
                l_before = self._verifier.align_loss(
                    chosen_ex["crop"], chosen_ex["mask_crop"], K_crop,
                    R_c, t_c)
                l_after = self._verifier.align_loss(
                    chosen_ex["crop"], chosen_ex["mask_crop"], K_crop,
                    R_g, t_g)
                if l_after < l_before:
                    R_c, t_c = R_g, t_g
            else:
                R_c, t_c = R_g, t_g
            timings["guided"] = time.time() - t0

        # ---- 迭代渲染对齐（6d-iter-align）：当前位姿重渲染新视角 →
        # MASt3R 再匹配 → 重解 PnP（接受/拒绝逻辑在 _iter_align 内部）----
        # iter_align_multi_hypo（08-11 挑战 3：模板离散化/选错模板）：
        # 36.2% 失败帧 R>60°（粗位姿远离 GT 最近模板），单假设 iter_align
        # 从错误盆出发救不回；改为从池内 top-k 位姿各跑 1 轮，渲染对齐
        # 损失择优（iG-6DoF 多候选初始化同思路）。
        if int(s_cfg.get("iter_align_iters", 0) or 0) > 0:
            t0 = time.time()
            mh = int(s_cfg.get("iter_align_multi_hypo", 0) or 0)
            track_on = bool(s_cfg.get("track_seed", False))
            if mh > 0 or (track_on and prev_pose is not None):
                ranked = rank_candidates(
                    results, strategy=s_cfg.get("selection", "inlier"))
                seeds = [best] + [r for r in ranked[:mh] if r is not best]
                if track_on and prev_pose is not None:
                    R_p, t_p = np.asarray(prev_pose[0]), np.asarray(prev_pose[1])
                    if np.all(np.isfinite(R_p)) and np.all(np.isfinite(t_p)):
                        seeds.append(SimpleNamespace(R=R_p, t=t_p,
                                                     n_inliers=0,
                                                     template_idx=-1))
                best_la, best_r = float("inf"), (R_c, t_c)
                # 种子级渲染对比优化（08-11 multi-refine）：把每个种子的
                # 优化从"匹配+PNP"升级为可微渲染对齐（GS-Pose GS-Refiner
                # 同款损失），短步数探各盆底；择优仍在盆底做（refiner
                # 6DOF 单假设判负教训：从错误盆出发 + 掩码损失面污染）。
                # 精化变差回退种子位姿，防 refiner 把好种子推坏。
                seed_rf = int(s_cfg.get(
                    "iter_align_seed_refine_iters", 0) or 0)
                # 择优门控（08-11 multi-ext 泛化教训）：multi 在池有货
                # 物体（duck/cat）+8~11，在池没货物体（ape/holepuncher）
                # -10~-15——align_loss 无条件换种子把 inlier 选出的 best
                # 也换掉（95/120 帧换种子），弱纹理物体 align_loss 与
                # ADD 相关性弱，净换坏。gate>0 时只接受比 best 种子
                # 显著更优（相对改善 ≥ gate）的候选，否则保持 best。
                gate = float(s_cfg.get("iter_align_multi_gate", 0.0) or 0.0)
                # 择优指标（08-12 通用化系列）：
                #   align  = 光度 align_loss（现状，multi 原版）
                #   gate   = align_loss + 相对门控（6d-multi-gate，判负：
                #            正确种子光度优势 <5% 被挡）
                #   iou    = 渲染掩码 IoU 几何门控（6d-multi-iou，ape 仍负：
                #            FastSAM 掩码偏差污染 IoU）
                #   inlier = PnP 内点数（纯几何，ia 基线选 best 同源指标，
                #            与 ADD 相关性最直接；候选须比 best 种子多
                #            inl_gate 个内点才替换）
                sel = s_cfg.get("iter_align_multi_select", "align")
                sel_iou = sel == "iou"
                sel_inl = sel == "inlier"
                iou_gate = float(s_cfg.get(
                    "iter_align_multi_iou_gate", 0.02) or 0.02)
                inl_gate = int(s_cfg.get(
                    "iter_align_multi_inl_gate", 500) or 0)
                la_best_seed = None
                for s in seeds:
                    R_s, t_s = np.asarray(s.R), np.asarray(s.t)
                    R_i, t_i, n_i = self._iter_align(chosen_ex, K_query,
                                                     R_s, t_s)
                    if R_i is None:
                        continue
                    x0, y0, _, _ = chosen_ex["crop_box_used"]
                    K_crop = K_query.copy()
                    K_crop[0, 2] -= x0
                    K_crop[1, 2] -= y0
                    if (seed_rf > 0 and self._refiner is not None):
                        R_r, t_r = self._refiner.refine(
                            chosen_ex["crop"], chosen_ex["mask_crop"],
                            K_crop, R_i, t_i, iterations=seed_rf)
                        if R_r is not None:
                            la_pre = self._verifier.align_loss(
                                chosen_ex["crop"], chosen_ex["mask_crop"],
                                K_crop, R_i, t_i)
                            la_post = self._verifier.align_loss(
                                chosen_ex["crop"], chosen_ex["mask_crop"],
                                K_crop, R_r, t_r)
                            if la_post <= la_pre:
                                R_i, t_i = R_r, t_r
                    if sel_inl:
                        la = float(n_i)
                        if s is best:
                            la_best_seed = la
                            best_la, best_r = la, (R_i, t_i)
                            continue
                        if la_best_seed is None:
                            continue
                        if la >= la_best_seed + inl_gate:
                            best_la, best_r = la, (R_i, t_i)
                    elif sel_iou:
                        la = self._verifier.mask_iou(
                            R_i, t_i, K_crop, chosen_ex["mask_crop"])
                        if s is best:
                            la_best_seed = la
                            # best 种子无条件接受（best_la 初始 inf 是
                            # align_loss 语义，iou 越大越好——直接覆盖）
                            best_la, best_r = la, (R_i, t_i)
                            continue
                        if la_best_seed is None:
                            continue
                        if la >= la_best_seed + iou_gate:
                            best_la, best_r = la, (R_i, t_i)
                    elif gate > 0:
                        la = self._verifier.align_loss(
                            chosen_ex["crop"], chosen_ex["mask_crop"],
                            K_crop, R_i, t_i)
                        if s is best:
                            la_best_seed = la
                            if la < best_la:
                                best_la, best_r = la, (R_i, t_i)
                            continue
                        if la_best_seed is None:
                            continue
                        if la <= la_best_seed * (1.0 - gate):
                            best_la, best_r = la, (R_i, t_i)
                    else:
                        la = self._verifier.align_loss(
                            chosen_ex["crop"], chosen_ex["mask_crop"],
                            K_crop, R_i, t_i)
                        if la < best_la:
                            best_la, best_r = la, (R_i, t_i)
                R_c, t_c = best_r
            else:
                R_i, t_i, _n = self._iter_align(chosen_ex, K_query, R_c, t_c)
                if not (np.allclose(R_i, R_c) and np.allclose(t_i, t_c)):
                    R_c, t_c = R_i, t_i
            timings["iter_align"] = time.time() - t0

        R_out, t_out = self._to_model_frame(R_c, t_c)
        coarse_R, coarse_t = R_out, t_out
        # 测试时位姿精化：在裁剪坐标系渲染 3DGS，对齐真实图（mask 内
        # L1+SSIM+LPIPS），把粗位姿推入局部最优
        if self._refiner is not None:
            t0 = time.time()
            x0, y0, _, _ = chosen_ex["crop_box_used"]
            K_crop = K_query.copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            # 任务 3 自适应（solver.adaptive）：按内点数分级——高内点 =
            # 位姿已可靠，跳过 refiner 直接输出（省 ~7s/帧）；低内点 =
            # 困难帧，refiner 迭代 boost_iters 倍（升级档）。
            ad = s_cfg.get("adaptive", {}) or {}
            ad_on = bool(ad.get("enabled", False))
            n_inl = getattr(best, "n_inliers", 0)
            skip_refine = ad_on and n_inl >= int(ad.get("high", 10 ** 9))
            boost_iters = (int(ad.get("boost_iters", 0))
                           if ad_on and n_inl < int(ad.get("low", 0)) else 0)
            if not skip_refine:
                # 渲染校验触发过补救且补救后 IoU 仍低（候选池无好位姿）：
                # 单次精化大概率困在错误盆地，改用多假设精化——对当前位姿
                # 扰动出多个种子各自短精化，按渲染对齐损失选最优盆地，再
                # 由 LPIPS 主 refiner 收尾（借鉴旧 MyPose refine.py 的
                # generate_hypotheses + run_search_stage 思想）
                if (rs_triggered and rs_iou < float(s_cfg.get(
                        "multi_hypo_iou", 0.3))
                        and self._hypo_refiner is not None):
                    R_c, t_c = self._multi_hypothesis_refine(
                        chosen_ex, K_crop, R_c, t_c)
                    R_r, t_r = R_c, t_c
                    timings["multi_hypo"] = time.time() - t0
                else:
                    R_r, t_r = self._refiner.refine(
                        chosen_ex["crop"], chosen_ex["mask_crop"],
                        K_crop, R_c, t_c, iterations=boost_iters or None)
                    # 精化回退保护：refiner 损失面 tz 平坦 + 旋转有梯度，
                    # 常把粗位姿推坏（实测负贡献，holepuncher 51.7→36.7）。
                    # 用渲染对齐损失比较精化前后，变差则保留粗位姿。
                    if R_r is not None:
                        la_before = self._refiner.align_loss(
                            chosen_ex["crop"], chosen_ex["mask_crop"],
                            K_crop, R_c, t_c)
                        la_after = self._refiner.align_loss(
                            chosen_ex["crop"], chosen_ex["mask_crop"],
                            K_crop, R_r, t_r)
                        if la_after > la_before:
                            R_r, t_r = R_c, t_c
                if R_r is not None:
                    R_out, t_out = self._to_model_frame(R_r, t_r)
                timings["refine"] = time.time() - t0
        refined_R, refined_t = R_out, t_out

        # ---- tz 面积比校准（tz_search）：渲染掩码面积 ∝ 1/z²，用
        # 查询掩码面积比迭代校正深度；随后按掩码质心差校正 xy ----
        # 深度病态：小物体 tz 错 30-40mm 时重投影偏移 <5px（RANSAC 无法
        # 分辨），但渲染掩码面积对 z 敏感。面积比 r=sqrt(A_render/A_mask)，
        # z_new = z·r 使面积自洽（迭代 2 次收敛）；xy 按渲染掩码质心与
        # 查询掩码质心的像素差反投影校正（GSPose §3.2 同款 Δxy 机制）。
        # 3DGS 渲染掩码与 FastSAM 掩码面积存在物体相关系统偏差，直接
        # 面积比会把已准帧拉偏——用校准前后渲染 IoU 做接受判据：IoU
        # 提升才接受（GSPose 面积比给候选、IoU 验方向）。
        # tz_search_xy_only（08-11 gap 分解后加）：跳过面积比 z，只做
        # 渲染质心对齐 xy——失败帧 48.5% 为旋转对/xy 平移错（99.1%
        # 粗位姿 R≤10° 但 t 错，PnP 对应质心偏），xy 对齐直接对症；
        # 面积比 z 是 tz_search 结案时的破坏源，先拆掉单独验证 xy。
        if (bool(s_cfg.get("tz_search", False))
                and self._verifier is not None):
            t0 = time.time()
            x0, y0, _, _ = chosen_ex["crop_box_used"]
            K_crop = K_query.copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            mask = chosen_ex["mask_crop"]
            if mask is not None:
                mask = np.asarray(mask) > 0
                a_mask = float(mask.sum())
                if a_mask >= 16:
                    import torch
                    W, H = mask.shape[1], mask.shape[0]
                    Kt = torch.tensor(K_crop, dtype=torch.float32,
                                      device=self.device)
                    iou_before = self._verifier.mask_iou(
                        R_out, t_out, K_crop, mask)
                    t_cur = t_out.copy()
                    xy_only = bool(s_cfg.get("tz_search_xy_only", False))
                    # tz_depth（08-11 弱纹理/平移病态方向）：BOP 深度图真实
                    # 测量，掩码内深度中值直接替换 z——绕开 PnP 平移病态
                    # （失败帧 48.5% 旋转对 t 错，深度是唯一无假设的信号）。
                    if (bool(s_cfg.get("tz_depth", False))
                            and ex.get("depth_img") is not None):
                        depth = np.asarray(ex["depth_img"])
                        loc = ex.get("loc")
                        full_mask = None
                        if loc is not None and getattr(loc, "mask", None) is not None:
                            full_mask = np.asarray(loc.mask) > 0
                        if full_mask is None:
                            full_mask = np.zeros(depth.shape[:2], dtype=bool)
                            x0d, y0d, x1d, y1d = chosen_ex["crop_box_used"]
                            x0d, y0d = int(x0d), int(y0d)
                            x1d = min(int(x1d), depth.shape[1])
                            y1d = min(int(y1d), depth.shape[0])
                            full_mask[y0d:y1d, x0d:x1d] = True
                        if full_mask.shape == depth.shape[:2]:
                            vals = depth[full_mask]
                            vals = vals[np.isfinite(vals) & (vals > 0)]
                            if len(vals) >= 16:
                                z_med = float(np.median(vals))
                                if 0.5 * t_out[2] < z_med < 2.0 * t_out[2]:
                                    t_cur[2] = z_med
                    rend = None
                    for it in range(2):
                        R0t = torch.tensor(R_out, dtype=torch.float32,
                                           device=self.device)
                        t0t = torch.tensor(t_cur, dtype=torch.float32,
                                           device=self.device)
                        _, alpha = self._verifier._render(
                            R0t, t0t, Kt, W, H)
                        rend = (alpha[..., 0].detach().cpu().numpy() > 0.5)
                        a_render = float(rend.sum())
                        if a_render < 16:
                            rend = None
                            break
                        if xy_only:
                            break
                        r = np.sqrt(a_render / a_mask)
                        if not (0.8 <= r <= 1.25):
                            break
                        t_cur[2] *= r
                    # xy 质心对齐：渲染掩码质心 → 查询掩码质心
                    ry, rx = np.nonzero(rend) if rend is not None else ([], [])
                    my, mx = np.nonzero(mask)
                    if len(ry) >= 16 and len(my) >= 16:
                        dcx = (mx.mean() - rx.mean()) / K_crop[0, 0]
                        dcy = (my.mean() - ry.mean()) / K_crop[1, 1]
                        t_cur[0] += dcx * t_cur[2]
                        t_cur[1] += dcy * t_cur[2]
                    if (self._verifier.mask_iou(R_out, t_cur, K_crop, mask)
                            > iou_before):
                        t_out = t_cur
            timings["tz_search"] = time.time() - t0
        return EstimateResult(success=True, R=R_out, t=t_out,
                              n_inliers=chosen.n_inliers,
                              best_template=chosen.template_idx,
                              timings=timings, candidates=candidates,
                              R_coarse=coarse_R, t_coarse=coarse_t,
                              R_refined=refined_R, t_refined=refined_t,
                              decode_order=ex.get("decode_order"))

    # ------------------------------------------------------------------
    def estimate(self, img_rgb_u8: np.ndarray, K_query: np.ndarray,
                 gt_bbox=None, gt_mask=None,
                 return_candidates: bool = False, frame_id=None,
                 depth_img=None, prev_pose=None) -> EstimateResult:
        """单帧 6D 位姿估计（阶段 2 extract_matches + 阶段 3 _solve）。

        prev_pose: (R, t) 前一帧最终位姿（模型系，w2c），tracking 种子用。
        """
        ex = self.extract_matches(img_rgb_u8, K_query,
                                  gt_bbox=gt_bbox, gt_mask=gt_mask,
                                  frame_id=frame_id, depth_img=depth_img)
        if ex is None:
            return EstimateResult(success=False, timings={})
        return self._solve(ex, K_query, return_candidates=return_candidates,
                           frame_id=frame_id, prev_pose=prev_pose)

    def _to_model_frame(self, R: np.ndarray, t: np.ndarray):
        """PnP 输出 → 原始模型单位 + VGGT→CAD 对齐（最终输出坐标系）。

        3D 锚点在 s 倍缩放的物体系中；投影对整体缩放不变，因此
        (R̂, t̂) 拟合 s·X 等价于 (R̂, t̂/s) 拟合 X —— 换回原始模型单位
        与 GT (mm) 直接比对（见 geometry/scale_align.py）。
        VGGT 路线再经 Umeyama 相似变换回 CAD 系。
        """
        t_model = t / self.bank.scale
        if self.bank.has_align:
            return transform_pose_by_similarity(
                R, t_model, self.bank.align_s,
                self.bank.align_R, self.bank.align_t)
        return R, t_model


# ---------------------------------------------------------------------------
# 数据集级评测（run_linemod.py / run_ablation.py 复用）
# ---------------------------------------------------------------------------
def subsample_frames(frames: List, max_frames: int) -> List:
    """小规模测试的均衡抽样：在评测序列上均匀间隔取 max_frames 帧。

    只取序列前 N 帧会让评估集中在相近视角（LineMod 测试序列按时间
    排序，物体姿态随帧演化），观测偏差大；均匀抽样覆盖整个序列的
    视角/难度分布，小规模测试的数字更能代表全量水平。
    """
    if not max_frames or len(frames) <= max_frames:
        return frames
    idx = np.unique(np.round(
        np.linspace(0, len(frames) - 1, max_frames)).astype(int))
    return [frames[i] for i in idx]


def _pack_matches(matches):
    """TemplateMatch 列表 → (pix_q, pix_t, sims, seg, template_idx, score)。"""
    n_per = [len(m.pix_q) for m in matches]
    if n_per:
        pix_q = np.concatenate([m.pix_q for m in matches]).astype(np.uint16)
        pix_t = np.concatenate([m.pix_t for m in matches]).astype(np.uint16)
        sims = np.concatenate([m.sims for m in matches]).astype(np.float16)
        p3q = [m.pts3d_q for m in matches]
        pts3d_q = (np.concatenate(p3q, axis=0).astype(np.float32)
                   if all(x is not None for x in p3q) and p3q
                   else np.zeros((0, 3), np.float32))
    else:
        pix_q = np.zeros((0, 2), np.uint16)
        pix_t = np.zeros((0, 2), np.uint16)
        sims = np.zeros((0,), np.float16)
        pts3d_q = np.zeros((0, 3), np.float32)
    seg = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int32)
    return (pix_q, pix_t, sims, seg,
            np.asarray([m.template_idx for m in matches], dtype=np.int16),
            np.asarray([m.score for m in matches], dtype=np.float32),
            pts3d_q)


def save_extracted_matches(npz_path, ex: Dict):
    """阶段 2 产物落盘：定位+匹配结果逐帧存 npz（含 crop/mask_crop 图，
    求解阶段完全自包含，不依赖原始帧文件）。

    定位候选消歧用的备选匹配（ex["alts"]）另存 <stem>_alt<i>.npz，
    主 npz 记录 alt_n——solve 阶段从落盘重建时渲染验证不丢原料。
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    pix_q, pix_t, sims, seg, tpl_idx, score, pts3d_q = _pack_matches(
        ex["matches"])
    np.savez_compressed(
        npz_path,
        pix_q=pix_q, pix_t=pix_t, sims=sims, seg=seg,
        template_idx=tpl_idx, score=score, pts3d_q=pts3d_q,
        crop=ex["crop"].astype(np.uint8),
        mask_crop=ex["mask_crop"].astype(np.uint8),
        crop_box=np.asarray(ex["crop_box_used"], dtype=np.int32),
        sxy=np.asarray(ex["sxy"], dtype=np.float32),
        s_leg=np.asarray(ex["s_leg"], dtype=np.float32),
        loc_score=np.float32(ex["loc"].score),
        alt_n=np.int32(len(ex.get("alts") or ())),
    )
    for i, a in enumerate(ex.get("alts") or ()):
        ap = npz_path.with_name(f"{npz_path.stem}_alt{i}.npz")
        aq, at, asims, aseg, atpl, ascore, apts3d_q = _pack_matches(
            a["matches"])
        np.savez_compressed(
            ap,
            pix_q=aq, pix_t=at, sims=asims, seg=aseg,
            template_idx=atpl, score=ascore, pts3d_q=apts3d_q,
            crop=a["crop"].astype(np.uint8),
            mask_crop=a["mask_crop"].astype(np.uint8),
            crop_box=np.asarray(a["crop_box_used"], dtype=np.int32),
            sxy=np.asarray(a["sxy"], dtype=np.float32),
            s_leg=np.asarray(a["s_leg"], dtype=np.float32),
        )


def _unpack_matches(d) -> List:
    """npz/数组 → TemplateMatch 列表（与 _pack_matches 互逆）。"""
    from .matching.mast3r_wrapper import TemplateMatch
    seg = d["seg"]
    has_p3q = "pts3d_q" in d and d["pts3d_q"].shape[0] > 0
    matches = []
    for i, ti in enumerate(d["template_idx"]):
        s0, s1 = int(seg[i]), int(seg[i + 1])
        p3q = (d["pts3d_q"][s0:s1].astype(np.float64)
               if has_p3q else None)
        matches.append(TemplateMatch(
            template_idx=int(ti), score=float(d["score"][i]),
            pix_q=d["pix_q"][s0:s1].astype(np.float64),
            pix_t=d["pix_t"][s0:s1].astype(np.float64),
            sims=d["sims"][s0:s1].astype(np.float32),
            pts3d_q=p3q))
    return matches


def load_extracted_matches(npz_path) -> Dict:
    """读取阶段 2 产物 → 重建 extract_matches() 的 ex dict（可喂 _solve）。"""
    d = np.load(npz_path)
    ex = {"matches": _unpack_matches(d),
          "crop": d["crop"],
          "mask_crop": d["mask_crop"].astype(bool),
          "crop_box_used": tuple(int(v) for v in d["crop_box"]),
          "s_leg": tuple(float(v) for v in d["s_leg"]),
          "sxy": tuple(float(v) for v in d["sxy"]),
          "timings": {}}
    alts = []
    for i in range(int(d.get("alt_n", 0))):
        ap = Path(npz_path).with_name(f"{Path(npz_path).stem}_alt{i}.npz")
        if not ap.exists():
            continue
        a = np.load(ap)
        alts.append({
            "matches": _unpack_matches(a),
            "crop": a["crop"],
            "mask_crop": a["mask_crop"].astype(bool),
            "crop_box_used": tuple(int(v) for v in a["crop_box"]),
            "s_leg": tuple(float(v) for v in a["s_leg"]),
            "sxy": tuple(float(v) for v in a["sxy"]),
        })
    ex["alts"] = alts
    return ex


def _load_cache_records(cp: Path, meta: Dict) -> Dict[int, Dict]:
    """读取缓存 jsonl（meta 校验 + 逐行帧指纹校验），返回 {frame_id: rec}。

    文件不存在 / 空 / meta 不匹配时返回空 dict（不抛错，调用方决定
    是否重定向到指纹专属文件）。

    文件级 meta 只比对 cfg_hash：matches_dir 只是产物来源（live 提取 vs
    预提取落盘），同一 cfg 下二者内容逐帧等价（08-13 教训：比对它会让
    主表 hash 的消融档白白重匹配 ~3.6h/次，且提取代码代际本就无法用它
    区分——cfg 不变的代码改动两侧同样漏检）。真实内容变化（换 matches
    产物、换 bank、换 PnP 参数）全部反映在 cfg_hash 上，逐帧指纹兜底。
    """
    import json as _json
    out: Dict[int, Dict] = {}
    if not cp.exists() or cp.stat().st_size == 0:
        return out
    lines = cp.read_text().splitlines()
    head = _json.loads(lines[0]) if lines and lines[0].strip() else None
    head_meta = head.get("__meta__") if head else None
    if not head_meta or head_meta.get("cfg_hash") != meta["cfg_hash"]:
        return out
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
            if rec.get("cfg_hash") == meta["cfg_hash"]:
                out[int(rec["frame_id"])] = rec
        except (KeyError, ValueError, TypeError):
            pass
    return out


def evaluate_object(cfg: Dict, obj_name: str, device: str = "cuda",
                    max_frames: int = 0, verbose: bool = True,
                    exclude_refs: bool = True,
                    bop_rows: Optional[List[Dict]] = None,
                    cache_path: Optional[str] = None,
                    matches_dir: Optional[str] = None,
                    frame_range: Optional[Tuple[int, int]] = None):
    """单物体全量评测：逐帧 estimate → 指标聚合（ADD/ADD-S/Proj/5cm5°）。

    exclude_refs=True（默认）时排除参考帧，杜绝 3DGS 参考
    视图泄漏进评测集（有官方 split 则评测测试划分，无 split 则排除采样参考帧）。

    bop_rows 传入 list 时，成功帧的位姿按 bop19 提交行追加进去
    （save_bop_csv 落盘后可交官方 bop_toolkit 复算，见 bop_metrics.py）。

    cache_path 给定时按帧缓存结果（jsonl，每帧一行），中断后重跑自动
    跳过已完成帧——全量 1172 帧/物体约 6 小时，必须支持断点续跑。

    matches_dir 给定时跳过 MASt3R 阶段（阶段 2 产物已由
    scripts/analysis/extract_matches.py 落盘），直接求解+聚合——调 PnP/择优参数
    无需重跑最贵的匹配阶段。

    Returns:
        (物体级指标 dict, 帧级明细 list, 平均分阶段耗时 dict)
    """
    from .metrics.pose_metrics import aggregate, evaluate_pose
    from .metrics.pose_metrics import add_error, adds_error, proj_error_px
    from .metrics.legacy_format import object_topk_metrics
    from .metrics.bop_metrics import (aggregate_bop, mspd_error, mspd_recall,
                                      mssd_error, mssd_recall,
                                      symmetry_transformations)

    ds = LinemodDataset(cfg["dataset"]["root"], obj_name,
                        models_dir=cfg["dataset"].get("models_dir",
                                                      "models_eval"),
                        splits_dir=cfg["dataset"].get("splits_dir"))
    bank = TemplateBank(template_bank_path(cfg, obj_name))
    # 渲染验证/位姿精化都吃 3DGS 参数 ckpt（onboard 落盘 .pt）。
    # refine_pose=false 时也可能需要前向渲染（定位候选消歧），统一传入，
    # 缺 .pt 时 PoseEstimator 内部会跳过渲染验证。
    refiner_ckpt = str(template_bank_path(cfg, obj_name).with_suffix(".pt"))
    stable_prior = None
    if bool(cfg["solver"].get("stable_prior", False)):
        sp_path = Path("outputs/stable_prior") / f"{obj_name}.npz"
        if sp_path.exists():
            sp = np.load(sp_path)
            stable_prior = {"axes": sp["axes"], "g": sp["g"]}
        else:
            print(f"[evaluate:{obj_name}] stable_prior 开启但缺 {sp_path}，跳过",
                  file=sys.stderr)
    estimator = PoseEstimator(cfg, bank, device=device,
                              refiner_ckpt=refiner_ckpt,
                              symmetric_transforms=ds.discrete_symmetry_transforms(),
                              stable_prior=stable_prior)
    model_pts = ds.model_points(max_points=2000)   # 指标点数抽稀（BOP 惯例）
    mcfg = cfg["metrics"]
    # topK best 评估（历史对照口径，见 VERIFICATION.md §8.4）：K 个候选中按 GT 择优的上界
    topk_ks = tuple(int(k) for k in (mcfg.get("topk_best") or ()))
    # BOP MSSD/MSPD（与 FoundPose 等 BOP 口径方法对比用，见 bop_metrics.py）。
    # 对称变换集从 models_info.json 读（eggbox/glue 有 symmetries_discrete，
    # 普通物体退化为恒等），与 ADD-S 的 eggbox/glue 白名单相互独立。
    bop_on = bool(mcfg.get("bop", True))
    syms = symmetry_transformations(ds.model_info) if bop_on else None
    # MSSD 是逐点 max，抽稀会系统性偏低——官方 eval 用 models_eval 全部
    # 顶点（eval_calc_errors.py:191-200 加载完整模型），这里跟随
    bop_pts = ds.model_points() if bop_on else None

    per_frame, all_timings = [], []
    cand_adds_all, cand_projs_all = [], []   # topK best 用的逐候选误差
    prev_pose = None   # 帧间 tracking 种子（前一帧成功才更新）
    # 帧级缓存（断点续跑）：cache[frame_id] = 该帧完整结果。
    # 缓存带内容指纹（matches_dir + 配置哈希）：任何输入变化（换 matches、
    # 换 bank、换 PnP 参数）都会让旧缓存整体作废——只按 frame_id 复用会
    # 静默把旧配置的结果冒充新结果（ape 缓存事故，RESEARCH_LOG §10）。
    # 追加模式（分片并行共享同一缓存文件时截断会互相踩踏）；陈旧内容
    # 永远无法通过 meta 校验，load 时整体忽略。
    import json as _json
    import hashlib
    cache = {}
    cache_fh = None
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        meta = {"matches_dir": matches_dir,
                "cfg_hash": hashlib.sha1(_json.dumps(
                    cfg, sort_keys=True, default=str).encode()).hexdigest()}
        # 帧级指纹隔离：缓存文件可能被不同配置的追加写污染（refine2 追加
        # dc2 事故，08-04）。meta 不匹配时绝不追加进同一文件，而是写到
        # <stem>_<hash8>.jsonl 独立文件；读时逐行校验帧自身指纹。
        cp_matched = False
        if cp.exists() and cp.stat().st_size > 0:
            head_lines = cp.read_text().splitlines()
            head = (_json.loads(head_lines[0])
                    if head_lines and head_lines[0].strip() else None)
            cp_matched = bool(head and head.get("__meta__") == meta)
        if cp_matched:
            cache.update(_load_cache_records(cp, meta))
        else:
            if cp.exists() and cp.stat().st_size > 0:
                # 污染/异指纹文件：物理隔离到指纹专属文件，不碰原文件。
                # 重定向目标若已有内容（上一轮已重定向过）同样要加载，
                # 否则中断重启会把已缓存帧全部重跑一遍
                # （lamp/phone 全量 ia 评估重复处理事故，08-10）。
                h8 = meta["cfg_hash"][:8]
                cp = cp.with_name(f"{cp.stem}_{h8}.jsonl")
                cache.update(_load_cache_records(cp, meta))
        cache_fh = open(cp, "a")
        if cp.stat().st_size == 0:
            cache_fh.write(_json.dumps({"__meta__": meta}) + "\n")
            cache_fh.flush()
    # VGGT 路线的 3 张参考帧亦须扣除评测（P1-1 复审）：无 split 时 vggt
    # 参考帧从测试序列抽出，若不排除，几何初始化用过的帧就会进评测集。
    extra_exclude = None
    if cfg["geometry"].get("source") == "vggt":
        extra_exclude = ds.vggt_reference_frame_ids(
            int(cfg["geometry"].get("vggt", {}).get("n_ref_images", 3)))
    frames = ds.eval_frames(exclude_refs=exclude_refs,
                            n_ref=int(cfg["onboard"].get("n_ref_views", 64)),
                            extra_exclude_ids=extra_exclude)
    frames = subsample_frames(frames, max_frames)
    if frame_range is not None:
        a, b = int(frame_range[0]), int(frame_range[1])
        frames = frames[a:b] if b > 0 else frames[a:]
    # topk 候选评估与主指标必须同口径（对称物体的 ADD-S 定义一起切换）
    if ds.symmetric:
        from functools import partial
        err_fn = partial(adds_error,
                         definition=mcfg.get("adds_definition", "unidirectional"))
    else:
        err_fn = add_error
    for fi, fr in enumerate(frames):
        if fr.frame_id in cache:
            rec = cache[fr.frame_id]
            per_frame.append(rec["m"])
            all_timings.append(rec["timings"])
            cand_adds_all.append(rec.get("cand_adds", []))
            cand_projs_all.append(rec.get("cand_projs", []))
            if bop_rows is not None and rec.get("success"):
                bop_rows.append({
                    "scene_id": ds.obj_id, "im_id": fr.frame_id,
                    "obj_id": ds.obj_id, "score": rec["n_inliers"],
                    "R": np.asarray(rec["R"]), "t": np.asarray(rec["t"]),
                    "time": float(sum(rec["timings"].values())),
                })
            continue
        img = cv2.cvtColor(cv2.imread(str(fr.rgb_path)), cv2.COLOR_BGR2RGB)
        gt_mask = None
        if fr.mask_path is not None:
            gt_mask = cv2.imread(str(fr.mask_path), cv2.IMREAD_GRAYSCALE) > 0
        depth_img = None
        if fr.depth_path is not None:
            depth_img = cv2.imread(str(fr.depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                depth_img = None
        if matches_dir is not None:
            # 阶段 3 路径：从落盘的阶段 2 产物直接求解（跳过 MASt3R）。
            # run_linemod 传入的 matches_dir 已含 obj 子目录。
            npz = Path(matches_dir) / f"{fr.frame_id:06d}.npz"
            ex = load_extracted_matches(npz)
            res = estimator._solve(ex, fr.K,
                                   return_candidates=bool(topk_ks),
                                   frame_id=fr.frame_id)
        else:
            res = estimator.estimate(img, fr.K, gt_bbox=fr.bbox_visib,
                                     gt_mask=gt_mask,
                                     return_candidates=bool(topk_ks),
                                     frame_id=fr.frame_id,
                                     depth_img=depth_img,
                                     prev_pose=prev_pose)
        if res.success:
            m = evaluate_pose(
                model_pts, ds.diameter, fr.K, fr.R_gt, fr.t_gt, res.R, res.t,
                symmetric=ds.symmetric,
                add_threshold_ratio=float(mcfg["add_threshold_ratio"]),
                proj_threshold_px=float(mcfg["proj_threshold_px"]),
                cm_threshold_mm=float(mcfg["cm_threshold_mm"]),
                deg_threshold=float(mcfg["deg_threshold"]),
                adds_definition=mcfg.get("adds_definition", "unidirectional"))
        else:
            # 失败帧计入分母（全 0 命中），与 BOP 协议一致
            m = {"add": np.inf, "proj": np.inf, "trans_err": np.inf,
                 "rot_err": np.inf, "add_01d": 0.0, "proj_5px": 0.0,
                 "cm_deg": 0.0}
        if bop_on:
            if res.success:
                e_mssd = mssd_error(bop_pts, fr.R_gt, fr.t_gt,
                                    res.R, res.t, syms)
                e_mspd = mspd_error(bop_pts, fr.K, fr.R_gt, fr.t_gt,
                                    res.R, res.t, syms)
                m["mssd_recall"] = mssd_recall(e_mssd, ds.diameter)
                m["mspd_recall"] = mspd_recall(e_mspd, img.shape[1])
            else:
                m["mssd_recall"] = 0.0
                m["mspd_recall"] = 0.0
        if bop_rows is not None and res.success:
            # LineMod 场景号即 obj_id；score 用内点数（FoundPose 的位姿
            # quality 同为 len(inliers)，pnp_util.py:79）
            bop_rows.append({
                "scene_id": ds.obj_id, "im_id": fr.frame_id,
                "obj_id": ds.obj_id, "score": res.n_inliers,
                "R": res.R, "t": res.t,
                "time": float(sum(res.timings.values())),
            })
        per_frame.append(m)
        all_timings.append(res.timings)
        if res.success and bool(cfg["solver"].get("track_seed", False)):
            prev_pose = (res.R, res.t)
        c_adds, c_projs = [], []
        if topk_ks:
            # 逐候选评估：每个候选独立算 ADD/Proj，之后同步选择；
            # 无候选帧记空序列。失败候选（PnP 无解）记 inf 而不是跳过——
            # 照样占 top-3/top-5 名额（历史对照口径，见 VERIFICATION.md §8.4）。
            for c in res.candidates:
                if not c.get("success", True):
                    c_adds.append(np.inf)
                    c_projs.append(np.inf)
                    continue
                c_adds.append(err_fn(model_pts, fr.R_gt, fr.t_gt,
                                     c["R"], c["t"]))
                c_projs.append(proj_error_px(model_pts, fr.K, fr.R_gt,
                                             fr.t_gt, c["R"], c["t"]))
            cand_adds_all.append(c_adds)
            cand_projs_all.append(c_projs)
        if cache_fh is not None:
            rec = {
                "frame_id": int(fr.frame_id),
                "cfg_hash": meta["cfg_hash"],   # 帧级指纹（读时过滤）
                "success": bool(res.success),
                "R": (res.R.tolist() if res.success else None),
                "t": (res.t.tolist() if res.success else None),
                "R_coarse": (res.R_coarse.tolist()
                             if res.success and res.R_coarse is not None
                             else None),
                "t_coarse": (res.t_coarse.tolist()
                             if res.success and res.t_coarse is not None
                             else None),
                "R_refined": (res.R_refined.tolist()
                              if res.success and res.R_refined is not None
                              else None),
                "t_refined": (res.t_refined.tolist()
                              if res.success and res.t_refined is not None
                              else None),
                "n_inliers": int(res.n_inliers),
                "m": m, "timings": res.timings,
                "cand_adds": c_adds, "cand_projs": c_projs,
                # 候选模板索引（诊断用：GT 最近模板是否进池/被选，08-11）
                "cand_templates": ([c.get("template_idx") for c in res.candidates]
                                   if res.candidates else []),
                # 逐候选内点数/相似度（自适应 K 早停离线仿真用，08-13）
                "cand_inliers": ([c.get("n_inliers") for c in res.candidates]
                                 if res.candidates else []),
                "cand_scores": ([c.get("score") for c in res.candidates]
                                if res.candidates else []),
                "cand_order": res.decode_order,
                # 逐候选位姿（模型系，ADD 精确重算用；~4KB/帧可忽略）
                "cand_Rs": ([c.get("R") for c in res.candidates]
                            if res.candidates else []),
                "cand_ts": ([c.get("t") for c in res.candidates]
                            if res.candidates else []),
                # 逐候选对应数/内点重投影残差（inlier_ratio / reproj 策略
                # 离线重排用，与 07 组消融互补，08-13）
                "cand_ncorr": ([c.get("n_correspondences")
                                for c in res.candidates]
                               if res.candidates else []),
                "cand_reproj": ([c.get("mean_inlier_reproj_px")
                                 for c in res.candidates]
                                if res.candidates else []),
            }
            cache_fh.write(_json.dumps(rec, default=float) + "\n")
            cache_fh.flush()
        if verbose and (fi + 1) % max(1, len(frames) // 20) == 0:
            agg = aggregate(per_frame)
            print(f"  [{obj_name}] {fi+1}/{len(frames)} "
                  f"ADD(S)@0.1d={agg['add_01d']:.2f}%")
    if cache_fh is not None:
        cache_fh.close()

    summary = aggregate(per_frame)
    if bop_on:
        summary["bop"] = aggregate_bop(per_frame)
    if topk_ks:
        summary["topk_best"] = object_topk_metrics(
            cand_adds_all, cand_projs_all,
            add_thresh=float(mcfg["add_threshold_ratio"]) * ds.diameter,
            proj_thresh=float(mcfg["proj_threshold_px"]), ks=topk_ks)
    avg_t = {}
    keys = set(k for t in all_timings for k in t)
    for k in keys:
        vals = [t[k] for t in all_timings if k in t]
        avg_t[k] = float(np.mean(vals)) if vals else 0.0
    return summary, per_frame, avg_t
