"""2D-2D/2D-3D 对应构建与过滤。

本模块是匹配管线的纯逻辑核心：输入任意来源的稠密局部描述子（MASt3R 的
d=24 desc、DINOv2 patch 特征等），输出经过互最近邻 + cycle consistency
过滤、相似度阈值筛选、采样后的 2D-3D 对应集 C_i。

全部为 numpy 实现，不依赖 GPU，本地 pytest 直接覆盖。

符号约定：
- S(y,y') = f_y^q · f_{y'}^i      点积相似度（单位范数特征下等价余弦）
- sim(m)  = mean_y max_{y'} S     模板级分数（查询前景像素对模板最大
                                   相似度的均值），用于 Top-K 模板选择
- 互最近邻 + 循环条件 U_n^t = U_n^{t+1}（容差 τ=5px）对应快速互最近邻
  匹配的 cycle consistency 检验
"""
from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def mutual_nn_matches(desc_q: np.ndarray, desc_t: np.ndarray,
                      normalize: bool = True):
    """互最近邻匹配（快速互最近邻的朴素等价实现）。

    对查询侧每个描述子找模板侧点积最近邻，再反向验证：只保留
    NN_t(NN_q(y)) == y 的双向一致对。

    Args:
        desc_q: (Nq, d) 查询前景像素描述子
        desc_t: (Nt, d) 模板前景像素描述子
        normalize: 是否先归一化到单位范数（MASt3R desc 已归一化，
                   DINOv2 patch 特征需要）
    Returns:
        idx_q, idx_t: 匹配对在两侧的下标数组
        sims: 每对匹配的点积相似度 S(y, y')
    """
    if desc_q.shape[0] == 0 or desc_t.shape[0] == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), np.zeros(0)
    if normalize:
        desc_q = _l2_normalize(np.asarray(desc_q, dtype=np.float32))
        desc_t = _l2_normalize(np.asarray(desc_t, dtype=np.float32))
    sim = desc_q @ desc_t.T                     # (Nq, Nt)
    nn_q2t = np.argmax(sim, axis=1)             # 查询 → 模板
    nn_t2q = np.argmax(sim, axis=0)             # 模板 → 查询
    idx_q = np.arange(desc_q.shape[0])
    mutual = nn_t2q[nn_q2t] == idx_q
    idx_q = idx_q[mutual]
    idx_t = nn_q2t[mutual]
    return idx_q, idx_t, sim[idx_q, idx_t]


def geometric_consistency_filter(pts3d_q: np.ndarray, pts3d_t: np.ndarray,
                                 nn_q2t: np.ndarray,
                                 sims: np.ndarray,
                                 rng: np.random.Generator,
                                 tau_obj_frac: float = 0.08,
                                 ransac_iters: int = 120,
                                 max_ransac_pts: int = 3000) -> np.ndarray:
    """MASt3R 两视图 3D 一致性过滤（官方 fast_nn 的几何验证步）。

    MASt3R 对查询/模板各预测一组相机系 pts3d；同名像素的 3D 点应满足
    相似变换 s·R·P_q + t ≈ P_t（单目深度尺度任意，7 DOF）。desc 最近邻
    在重复纹理（eggbox 鸡蛋格、glue 圆柱、打孔板）下会找到视觉相似但
    几何错位的对应，cycle consistency 对自相似纹理同样失效——错配会
    原样进入 PnP。用 RANSAC-Umeyama 估计两视图相对相似变换后，按
    3D 残差过滤掉几何不一致的对应。

    Args:
        pts3d_q: (Nq,3) 查询前景像素的相机系 3D 点
        pts3d_t: (Nt,3) 模板前景像素的相机系 3D 点
        nn_q2t:  (Nq,) 查询 → 模板最近邻下标
        sims:    (Nq,) 对应相似度（RANSAC 采样按此降序取高置信子集）
        rng:     随机数生成器
        tau_obj_frac: 内点阈值 = tau_obj_frac × 模板侧物体半径
                   （MASt3R 深度无单位，按物体尺寸归一化）
        ransac_iters: RANSAC 迭代次数
        max_ransac_pts: 参与变换估计的最大对应数（高相似度子集）

    Returns:
        keep: (Nq,) bool，几何一致（3D 残差 < 阈值）的对应
    """
    pts3d_q = np.asarray(pts3d_q, dtype=np.float64)
    pts3d_t = np.asarray(pts3d_t, dtype=np.float64)
    nn_q2t = np.asarray(nn_q2t, dtype=np.int64)
    if len(nn_q2t) == 0:
        return np.zeros(0, dtype=bool)
    # 模板侧物体半径（尺度归一化基准）
    centre = pts3d_t.mean(axis=0)
    radius = float(np.median(np.linalg.norm(pts3d_t - centre, axis=1)))
    tau = tau_obj_frac * max(radius, 1e-9)

    # 高相似度子集（RANSAC 估计用；过滤时对全量）
    n = len(nn_q2t)
    m = min(n, max_ransac_pts)
    if sims is not None:
        sub = np.argsort(-np.asarray(sims))[:m]
    else:
        sub = np.arange(m)
    pq_s = pts3d_q[sub]
    pt_s = pts3d_t[nn_q2t[sub]]

    from ..geometry.alignment import umeyama_alignment
    best_s, best_R, best_t = 1.0, np.eye(3), np.zeros(3)
    best_cnt = -1
    for _ in range(ransac_iters):
        sel = rng.choice(m, 3, replace=False)
        a, b = pq_s[sel], pt_s[sel]
        # 退化防御：任一点与其余点重合时跳过（Umeyama SVD 奇异）。
        # 注意 a[0] 与自身距离恒为 0，须排除自身再比较
        if (np.linalg.norm(a[1:] - a[0], axis=1) < 1e-9).any():
            continue
        if (np.linalg.norm(b[1:] - b[0], axis=1) < 1e-9).any():
            continue
        try:
            s, R, t = umeyama_alignment(a, b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(s) or s < 1e-3 or s > 1e3:
            continue
        res = np.linalg.norm(s * (pq_s @ R.T) + t - pt_s, axis=1)
        cnt = int((res < tau).sum())
        if cnt > best_cnt:
            best_cnt, best_s, best_R, best_t = cnt, s, R, t
    if best_cnt < 3 or best_cnt < 0.15 * m:
        # 找不到明确几何主模式（高相似度子集中内点占比 < 15%）时放行：
        # 此时过滤会把仅有的少量正确对应一并删掉（实测 ape/can 某些帧
        # 支持率低的帧被滤到 n_inliers<900，PnP 直接退化）。回退到
        # 不过滤，让 PnP-RANSAC 自己处理——行为回到几何过滤之前。
        return np.ones(n, dtype=bool)
    # 全量过滤
    res = np.linalg.norm(best_s * (pts3d_q @ best_R.T) + best_t
                         - pts3d_t[nn_q2t], axis=1)
    return res < tau


def cycle_consistency_filter(pix_q: np.ndarray, idx_q: np.ndarray,
                             idx_t: np.ndarray, nn_t2q: np.ndarray,
                             tau_px: float = 5.0):
    """像素域 cycle consistency 过滤（τ=5px）。

    互最近邻要求往返回到同一像素（τ=0）；此处放宽为往返落点与出发像素的
    欧氏距离 ≤ τ 像素，以容忍相邻像素间的量化抖动：
        y --NN--> y' --NN--> y''，保留 ||y - y''||_2 ≤ τ 的匹配。

    Args:
        pix_q: (Nq, 2) 查询侧全部候选像素坐标
        idx_q, idx_t: 待过滤匹配对下标（查询侧 / 模板侧）
        nn_t2q: (Nt,) 模板侧每个描述子在查询侧的最近邻下标（回程映射）
        tau_px: 像素容差 τ
    Returns:
        过滤后保留的布尔掩码（与 idx_q 等长）
    """
    if len(idx_q) == 0:
        return np.zeros(0, dtype=bool)
    back = nn_t2q[idx_t]                        # y'' 的查询侧下标
    d = np.linalg.norm(pix_q[back] - pix_q[idx_q], axis=1)
    return d <= tau_px


def template_score(desc_q: np.ndarray, desc_t: np.ndarray,
                   normalize: bool = True) -> float:
    """模板级分数 sim(m) = 查询前景像素对模板最大相似度的均值。

    该分数衡量「模板 m 能多好地解释查询前景」，用于 Top-K 模板选择。
    """
    if desc_q.shape[0] == 0 or desc_t.shape[0] == 0:
        return -np.inf
    if normalize:
        desc_q = _l2_normalize(np.asarray(desc_q, dtype=np.float32))
        desc_t = _l2_normalize(np.asarray(desc_t, dtype=np.float32))
    sim = desc_q @ desc_t.T
    return float(sim.max(axis=1).mean())


def topk_templates(scores: np.ndarray, k: int) -> np.ndarray:
    """按模板级分数降序取 Top-K 下标 I_K（默认 K=5）。"""
    scores = np.asarray(scores, dtype=np.float64)
    k = min(int(k), len(scores))
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def build_correspondences(pix_q: np.ndarray, desc_q: np.ndarray,
                          pix_t: np.ndarray, desc_t: np.ndarray,
                          coord_t: np.ndarray,
                          sim_threshold: float = 0.3,
                          cycle_tau_px: float = 5.0,
                          normalize: bool = True):
    """单个模板的完整 2D-3D 对应集构建（对应集 C_i）。

    流程：互最近邻 → cycle consistency(τ px) → 相似度阈值
    S(y, y'^*) > τ_match → 经模板 3D 坐标图 Φ_i 映射到三维锚点。

    Args:
        pix_q:   (Nq,2) 查询前景像素坐标（原图坐标系）
        desc_q:  (Nq,d) 查询描述子
        pix_t:   (Nt,2) 模板前景像素坐标
        desc_t:  (Nt,d) 模板描述子
        coord_t: (Nt,3) 模板像素对应的三维锚点 Φ_i(y')（物体系）
        sim_threshold: 相似度阈值 τ_match（默认 0.3）
        cycle_tau_px:  cycle consistency 像素容差 τ（默认 5px）
    Returns:
        pts2d (M,2), pts3d (M,3), sims (M,)
    """
    if normalize:
        desc_q = _l2_normalize(np.asarray(desc_q, dtype=np.float32))
        desc_t = _l2_normalize(np.asarray(desc_t, dtype=np.float32))
    if desc_q.shape[0] == 0 or desc_t.shape[0] == 0:
        return (np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0))

    sim = desc_q @ desc_t.T
    nn_q2t = np.argmax(sim, axis=1)
    nn_t2q = np.argmax(sim, axis=0)
    idx_q = np.arange(desc_q.shape[0])
    # cycle consistency（τ px 往返容差）过滤。严格互最近邻是 τ=0 的特例
    # （往返回到自身，距离为 0），因此该过滤天然包含互最近邻匹配对。
    keep = cycle_consistency_filter(pix_q, idx_q, nn_q2t, nn_t2q,
                                    tau_px=cycle_tau_px)
    idx_q = idx_q[keep]
    idx_t = nn_q2t[keep]
    sims = sim[idx_q, idx_t]

    # 相似度阈值过滤：S > τ_match
    ok = sims > sim_threshold
    idx_q, idx_t, sims = idx_q[ok], idx_t[ok], sims[ok]

    return pix_q[idx_q].astype(np.float64), \
        coord_t[idx_t].astype(np.float64), sims.astype(np.float64)


def sample_correspondences(pts2d: np.ndarray, pts3d: np.ndarray,
                           sims: np.ndarray, n_sample: int = 4096,
                           rng: np.random.Generator | None = None):
    """对应集下采样至 N_s（默认 4096），控制 RANSAC 开销。

    超出 N_s 时按相似度加权随机采样——保留高置信匹配的同时维持空间多样性
    （MASt3R 论文观察到适度子采样具有离群点过滤效果）。
    """
    n = pts2d.shape[0]
    if n <= n_sample:
        return pts2d, pts3d, sims
    if rng is None:
        rng = np.random.default_rng(0)
    w = np.clip(sims, 1e-6, None)
    p = w / w.sum()
    idx = rng.choice(n, size=n_sample, replace=False, p=p)
    return pts2d[idx], pts3d[idx], sims[idx]


def guided_local_matching(desc_q, desc_t, pix_t, pts3d_t, R, t, K,
                          sxy, crop_xy, r=12, n_t=4000, sim_thresh=0.3,
                          device="cuda"):
    """粗位姿引导的局部对应重匹配（guided correspondence refinement）。

    全局 desc 最近邻在重复纹理下会把对应错到"视觉相似但几何错位"的同名点
    （eggbox 鸡蛋格等），cycle/3D 过滤都救不回来。这里用粗位姿把模板锚点
    投影到查询图，只在投影位置 ±r px 的局部窗口内做 desc 最近邻——窗口
    约束把搜索限制在正确区域的邻域，错位一个格的对应被拉回。

    Args:
        desc_q: (H,W,24) 查询匹配区稠密 desc（fp16/fp32，单位范数）
        desc_t: (Nt,24) 模板前景 desc（与 pix_t 对齐）
        pix_t:  (Nt,2) 模板前景像素（模板原生坐标，与 coord_map 同系）
        pts3d_t: (Nt,3) 模板锚点（物体系，与 R/t 同缩放系）
        R, t:   粗位姿 w2c（与 pts3d_t 同系）
        K:      (3,3) 裁剪系内参（主点已平移）
        sxy:    (sx, sy) 匹配区缩放（匹配区 = 裁剪区 × sxy）
        crop_xy: (x0, y0) 裁剪原点
        r:      搜索窗口半径（匹配区像素）
        n_t:    参与搜索的模板点数上限（采样）
        sim_thresh: 保留的相似度下限
        device: torch 设备

    Returns:
        (p2, p3, sims)：原图坐标 2D 点（与 K_query 配套）、模板锚点 3D
        （缩放系）、相似度
    """
    import torch
    Nt = len(pix_t)
    if Nt == 0:
        return (np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0))
    n_t = min(Nt, n_t)
    sel = np.random.default_rng(0).choice(Nt, n_t, replace=False)
    pix_t = np.asarray(pix_t)[sel]
    pts3d_t = np.asarray(pts3d_t, dtype=np.float64)[sel]
    desc_t = np.asarray(desc_t, dtype=np.float32)[sel]

    # 锚点投影：物体系 → 相机系 → 裁剪系 → 匹配区
    cam = R @ pts3d_t.T + t[:, None]              # (3,n)
    uv = K @ cam
    uv = uv[:2] / uv[2]                           # 裁剪系像素
    u_proj = uv.T * np.array([sxy[0], sxy[1]])    # (n,2) 匹配区

    dq = torch.tensor(np.asarray(desc_q, dtype=np.float32), device=device)
    dt = torch.tensor(desc_t, device=device)
    u = torch.round(torch.tensor(u_proj, device=device)).long()
    H, W = dq.shape[:2]
    best_sim = torch.full((n_t,), float("-inf"), device=device)
    best_xy = torch.zeros((n_t, 2), dtype=torch.long, device=device)
    for dy in range(-r, r + 1):
        y_in = (u[:, 1] + dy >= 0) & (u[:, 1] + dy < H)
        yy = (u[:, 1] + dy).clamp(0, H - 1)
        for dx in range(-r, r + 1):
            x_in = (u[:, 0] + dx >= 0) & (u[:, 0] + dx < W)
            inb = y_in & x_in
            xx = (u[:, 0] + dx).clamp(0, W - 1)
            w = dq[yy, xx]                        # (n,24)
            sim = (w * dt).sum(-1)
            sim = torch.where(inb, sim,
                              torch.full_like(sim, float("-inf")))
            upd = sim > best_sim
            if upd.any():
                best_sim[upd] = sim[upd]
                best_xy[upd, 0] = xx[upd]
                best_xy[upd, 1] = yy[upd]
    keep = best_sim > sim_thresh
    p2 = best_xy[keep].float().cpu().numpy()      # 匹配区
    p3 = pts3d_t[keep.cpu().numpy()]
    sims = best_sim[keep].cpu().numpy()
    if len(p2) == 0:
        return (np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0))
    # 匹配区 → 原图（back_to_original_pixels 的逆变换步骤）
    x0, y0 = float(crop_xy[0]), float(crop_xy[1])
    p2 = p2 / np.array([sxy[0], sxy[1]]) + np.array([x0, y0])
    return p2, p3, sims


def back_to_original_pixels(pix_q: np.ndarray, sxy, crop_box) -> np.ndarray:
    """把 MASt3R 匹配输出的查询像素坐标反变换回原图坐标。

    在线管线的定位 → 匹配链条依次做了两次坐标变换：
        原图 --[crop by (x0,y0)]--> 裁剪区 --[resize by (sx,sy)]--> 匹配区
    MASt3R 直接输出的 `pix_q` 位于最里层"匹配区"坐标系；PnP 需要的是原图
    像素（与 K_query 自洽）。反变换是这两步的逆：先除以 (sx, sy) 回到裁剪
    坐标，再加 (x0, y0) 平移回原图。

    抽成纯函数便于用合成 GT 位姿 → 投影 → 反变换 → PnP 回到 GT 做闭环单测
    （P1-2 复审）。resize 因子严格用除法而非乘法，与 MASt3R wrapper 里
    `sx = W_resize / W_crop`、`sy = H_resize / H_crop` 的定义严格互逆。

    Args:
        pix_q:    (N,2) 匹配区坐标下的查询像素
        sxy:      (sx, sy) 匹配 resize 缩放因子（W_resize/W_crop, H_resize/H_crop）
        crop_box: (x0, y0, x1, y1) 裁剪框（原图坐标，x1/y1 用不到，接口
                  形状与 Localization.crop_box 一致以便直接透传）
    Returns:
        (N,2) 原图像素坐标
    """
    sx, sy = float(sxy[0]), float(sxy[1])
    x0, y0 = float(crop_box[0]), float(crop_box[1])
    pix_q = np.asarray(pix_q, dtype=np.float64)
    return pix_q / np.array([sx, sy]) + np.array([x0, y0])
