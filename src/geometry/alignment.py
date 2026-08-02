"""相似变换对齐工具：VGGT 重建坐标系 → CAD 坐标系。

VGGT 输出的点云定义在第一帧相机系（重心平移到原点后作为重建模型系），
其尺度由训练数据归一化决定，与 CAD 的物体系相差一个未知相似变换
（尺度 s、旋转 R、平移 t）。评测时用 Umeyama 闭式解求该相似变换
（可选 ICP 精化），把估计位姿从重建系变换回 CAD 系再算指标。

CAD 模型仅用于评测侧对齐，不参与任何推理（model-free 通行惯例）。
纯 numpy 实现，本地 CPU 可测。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray,
                      with_scale: bool = True
                      ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama 闭式相似变换求解（Umeyama 1991，SVD 实现）。

    求 (s, R, t) 使 dst ≈ s·R·src + t 的均方误差最小（点已一一对应）。

    Args:
        src_pts: (N,3) 源点集（重建系）
        dst_pts: (N,3) 目标点集（CAD 系），与 src 逐行对应
        with_scale: False 时锁定 s=1（纯刚体对齐）

    Returns:
        (s, R, t)：s 标量、R (3,3) 正交旋转（det=+1）、t (3,)。
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst 形状须为一致的 (N,3)，得到 "
                         f"{src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"Umeyama 至少需要 3 对点，得到 {n}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 协方差 Σ = (1/n) dst_c^T src_c
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    # 处理反射：保证 det(R)=+1（Umeyama 1991 式 (39) 的 S 矩阵）
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def farthest_point_sample(points: np.ndarray, n_sample: int,
                          rng: np.random.Generator = None) -> np.ndarray:
    """最远点采样（FPS），返回选中点的行下标 (n_sample,)。

    用于从稠密点云均匀取有代表性的子集做 Umeyama 初对齐，避免全量点参与
    时被局部高密度区域主导。
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n_sample >= n:
        return np.arange(n)
    if rng is None:
        rng = np.random.default_rng(0)
    idx = np.empty(n_sample, dtype=np.int64)
    idx[0] = rng.integers(n)
    dist = np.full(n, np.inf)
    for i in range(1, n_sample):
        last = points[idx[i - 1]]
        d = ((points - last) ** 2).sum(axis=1)
        dist = np.minimum(dist, d)
        idx[i] = int(np.argmax(dist))
    return idx


def icp_refine(src_pts: np.ndarray, dst_pts: np.ndarray,
               s: float, R: np.ndarray, t: np.ndarray,
               iterations: int = 20, with_scale: bool = True,
               eps: float = 1e-6, verbose: bool = False
               ) -> Tuple[float, np.ndarray, np.ndarray]:
    """相似变换 ICP 精化：以 (s,R,t) 为初值，迭代最近点重估相似变换。

    每轮把 src 经当前 (s,R,t) 变换后，对每个变换点找 dst 中最近邻，再用
    Umeyama 重解相似变换。点集无需一一对应（初对齐后靠最近邻建立对应）。
    scipy 可用时走 cKDTree，否则退化为朴素 O(N·M) 最近邻。

    P2-6 复审：加"相邻两轮 (s,R,t) 变化 < eps 提前停止"判据——收敛后再迭
    代纯粹烧算力，尤其是 CAD 对齐这类点数上千的场景。收敛度量取三部分
    相对/绝对变化的合成：|Δs|/|s| + ||ΔR||_F + ||Δt||_2/max(||t||,1)。
    verbose=True 时每轮打印 residual = 均方最近邻距离，便于调参。
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        query = lambda pts: tree.query(pts)                    # (dist, idx)
    except ImportError:                       # pragma: no cover - scipy 缺失兜底
        def query(pts):
            d = ((pts[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
            nn = np.argmin(d, axis=1)
            return np.sqrt(d[np.arange(len(pts)), nn]), nn

    for it in range(iterations):
        moved = (s * (R @ src.T).T) + t
        dists, nn = query(moved)
        s_new, R_new, t_new = umeyama_alignment(src, dst[nn],
                                                with_scale=with_scale)
        # 相对变化度量：尺度用相对、旋转用 Frobenius 差、平移按当前平移
        # 尺度归一（t≈0 时退化到绝对差，用 max(||t||,1) 兜底）
        ds = abs(s_new - s) / max(abs(s), 1e-9)
        dR = float(np.linalg.norm(R_new - R))
        dt = float(np.linalg.norm(t_new - t)) / max(np.linalg.norm(t), 1.0)
        s, R, t = s_new, R_new, t_new
        if verbose:
            print(f"[icp] iter {it+1}/{iterations} "
                  f"residual={float(np.mean(dists**2)):.4g} "
                  f"Δs={ds:.2e} ΔR={dR:.2e} Δt={dt:.2e}")
        if ds + dR + dt < eps:
            break
    return s, R, t


def transform_pose_by_similarity(R_pose: np.ndarray, t_pose: np.ndarray,
                                 s: float, R_a: np.ndarray, t_a: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """把重建系下的估计位姿 (R_pose, t_pose) 变换回 CAD 系。

    设相似变换把重建模型点 Y 映到 CAD 点 X： X = s·R_a·Y + t_a。相机点
    p_cam = R_pose·Y + t_pose（t_pose 为重建模型单位）。以 CAD 系度量的等价
    刚体位姿为：

        R_cad = R_pose · R_a^T
        t_cad = s · t_pose − R_cad · t_a

    推导：Y = (1/s) R_a^T (X − t_a)，代入并整体乘 s 换算到 CAD 度量尺度，
    整理即得上式（R_cad 正交，为合法旋转）。
    """
    R_a = np.asarray(R_a, dtype=np.float64)
    t_a = np.asarray(t_a, dtype=np.float64)
    R_cad = np.asarray(R_pose, dtype=np.float64) @ R_a.T
    t_cad = float(s) * np.asarray(t_pose, dtype=np.float64) - R_cad @ t_a
    return R_cad, t_cad
