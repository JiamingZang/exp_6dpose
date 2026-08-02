"""RANSAC-EPnP 位姿求解。

对每个候选模板的 2D-3D 对应集 C_i 独立执行：
1. 随机采样最小样本集（4 对）→ EPnP 求候选位姿 [R^(s)|t^(s)]；
2. 全部对应投影，重投影误差 e_j = ||y_j - π(K, R P_j + t)||_2；
3. 统计内点数 N_inlier = |{j : e_j < ε}|，保留内点最多的假设；
4. 内点集上 Levenberg-Marquardt 精化。

实现使用 OpenCV solvePnPRansac（flags=SOLVEPNP_EPNP，ε=5px，置信度 0.999，
迭代 1000）+ solvePnPRefineLM。纯 CPU，本地可测。
外部支撑：FoundPose（ECCV 2024）官方实现同为 solvePnPRansac + 内点集上
solvePnPRefineLM（foundpose/utils/pnp_util.py:46-74）；其参数为
ε=10px / 400 迭代（configs/infer/lmo.json），与本库同量级——ε 已列入
消融（configs/ablations）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PnPResult:
    """单个候选模板的求解结果（多候选择优的输入）。"""
    success: bool
    R: np.ndarray = field(default_factory=lambda: np.eye(3))   # w2c 旋转
    t: np.ndarray = field(default_factory=lambda: np.zeros(3)) # 平移（模型单位，LineMod 为 mm）
    n_inliers: int = 0
    inlier_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    template_idx: int = -1        # 该结果来自哪个模板（择优后回溯用）
    template_score: float = 0.0   # 模板级相似度分数（weighted 择优用）
    # 择优判据阶梯的另两个量（择优判据消融用）：
    n_correspondences: int = 0             # 送进 PnP 的总对应数（内点比分母）
    mean_inlier_reproj_px: float = float("inf")  # 内点平均重投影残差（px）


def _pnp_flag(name: str) -> int:
    """PnP 求解器 flag：epnp=默认；sqpnp=历史对照口径（见 VERIFICATION.md §8.3）。
    未知值显式 raise。"""
    flags = {"epnp": cv2.SOLVEPNP_EPNP}
    # SQPNP 需要 OpenCV >= 4.5.3；老版本装了 legacy 配置也要报清晰错误
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags["sqpnp"] = cv2.SOLVEPNP_SQPNP
    if name not in flags:
        raise ValueError(
            f"未知 solver.pnp_flag: {name!r}（可选 {sorted(flags)}；"
            f"sqpnp 需要 opencv-python>=4.5.3）")
    return flags[name]


def ransac_pnp(pts2d: np.ndarray, pts3d: np.ndarray, K: np.ndarray,
               reproj_px: float = 5.0, confidence: float = 0.999,
               iterations: int = 1000, refine_lm: bool = True,
               min_correspondences: int = 6,
               flag: str = "epnp",
               sym_transforms: Optional[List[np.ndarray]] = None) -> PnPResult:
    """RANSAC-EPnP + LM 精化。

    Args:
        pts2d: (N,2) 查询像素坐标 y_j
        pts3d: (N,3) 三维锚点 P_j（物体/模型系）
        K:     (3,3) 查询相机内参
        reproj_px: 内点阈值 ε
        confidence / iterations: RANSAC 置信度与最大迭代次数
        refine_lm: 是否在内点集上 LM 精化
        min_correspondences: 少于该数直接判失败（EPnP 最少 4 对，留冗余）
        flag: PnP 最小解算法 epnp | sqpnp（sqpnp 为历史对照口径）
        sym_transforms: 对称物体（eggbox/glue）的物体系离散对称变换
            (K,4,4) 列表（不含恒等）。每个 3D 锚点展开为
            {T_sym @ p} ∪ {p} 共 K+1 个等价点（共享同一 2D 像素），
            使错配到对称等价位置的对应点可进入 RANSAC 内点集——
            ADD-S 评估允许对称等价位姿，内点判定必须同样允许。
    """
    solver_flag = _pnp_flag(flag)
    pts3d = np.ascontiguousarray(pts3d, dtype=np.float64).reshape(-1, 3)
    pts2d = np.ascontiguousarray(pts2d, dtype=np.float64).reshape(-1, 2)
    n = pts2d.shape[0]
    if n < max(min_correspondences, 4):
        return PnPResult(success=False, n_correspondences=n)
    if sym_transforms:
        return _ransac_pnp_sym(pts2d, pts3d, K, sym_transforms, reproj_px,
                               confidence, iterations, refine_lm,
                               min_correspondences, solver_flag)
    pts2d = pts2d.reshape(-1, 1, 2)
    pts3d = pts3d.reshape(-1, 1, 3)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, distCoeffs=None,
        reprojectionError=float(reproj_px),
        confidence=float(confidence),
        iterationsCount=int(iterations),
        flags=solver_flag,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return PnPResult(success=False, n_correspondences=n)
    inlier_idx = inliers.reshape(-1).astype(np.int64)

    if refine_lm:
        # LM 只在内点集上跑：外点会把最小二乘拉偏，这正是 RANSAC 先筛的意义
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3d[inlier_idx], pts2d[inlier_idx], K, None, rvec, tvec)
        except cv2.error:
            pass  # 精化失败就退回 RANSAC 解，不影响成功判定

    R, _ = cv2.Rodrigues(rvec)
    # 内点平均重投影残差：在最终位姿（含 LM 精化）上重算，供 reproj 择优
    # （selection.py）。RANSAC 内部的残差是精化前的，不能直接用。
    uv, _ = cv2.projectPoints(pts3d[inlier_idx], rvec, tvec, K, None)
    residual = np.linalg.norm(
        uv.reshape(-1, 2) - pts2d[inlier_idx].reshape(-1, 2), axis=1)
    return PnPResult(success=True, R=R, t=tvec.reshape(3),
                     n_inliers=len(inlier_idx), inlier_idx=inlier_idx,
                     n_correspondences=n,
                     mean_inlier_reproj_px=float(residual.mean()))


def _ransac_pnp_sym(pts2d, pts3d, K, sym_Ts, reproj_px, confidence,
                    iterations, refine_lm, min_correspondences,
                    solver_flag):
    """对称感知 RANSAC-EPnP + LM（对称物体专用）。

    采样阶段不展开（最小解必须来自同一物理位姿的一致点集）；内点判定
    用对称展开投影 e_j = min_k ||π(K, R·T_k·p_j + t) - y_j||（T_k 为物体系
    对称变换，含恒等），与 ADD-S 评估口径一致；LM 精化在内点的最优
    对称分支点上进行。采样 4 点若共面/退化则跳过该假设。
    """
    n = len(pts2d)
    rng = np.random.default_rng(0)
    branches = [pts3d]
    for T in sym_Ts:
        T = np.asarray(T, dtype=np.float64)
        branches.append(pts3d @ T[:3, :3].T + T[:3, 3])
    br = np.stack(branches, axis=0)            # (B,N,3)
    B = br.shape[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u_q = pts2d[:, 0]
    v_q = pts2d[:, 1]
    best_inl = min_correspondences
    best = None
    for it in range(int(iterations)):
        idx = rng.choice(n, 4, replace=False)
        ok, rvec, tvec = cv2.solvePnP(
            pts3d[idx], pts2d[idx].reshape(-1, 1, 2), K, None,
            flags=solver_flag)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        pc = br @ R.T + tvec.reshape(1, 3)     # (B,N,3)
        z = pc[:, :, 2]
        if (z <= 1e-6).all():
            continue
        zi = np.maximum(z, 1e-9)
        u = pc[:, :, 0] / zi * fx + cx
        v = pc[:, :, 1] / zi * fy + cy
        e_min = np.minimum.reduce(
            np.sqrt((u - u_q[None, :]) ** 2 + (v - v_q[None, :]) ** 2),
            axis=0)                            # (N,) 对称感知误差
        inl = np.nonzero(e_min < reproj_px)[0]
        if len(inl) > best_inl:
            best_inl = len(inl)
            best = (rvec.copy(), tvec.copy(), inl)
            p = best_inl / n
            if p > 0:
                need = int(np.ceil(np.log(1 - confidence)
                                   / np.log(max(1 - p ** 4, 1e-12))))
                if it >= need:
                    break
    if best is None:
        return PnPResult(success=False, n_correspondences=n)
    rvec, tvec, inl = best
    if refine_lm:
        R, _ = cv2.Rodrigues(rvec)
        pc = br[:, inl] @ R.T + tvec.reshape(1, 3)
        z = np.maximum(pc[:, :, 2], 1e-9)
        u = pc[:, :, 0] / z * fx + cx
        v = pc[:, :, 1] / z * fy + cy
        err_b = np.sqrt((u - u_q[inl][None, :]) ** 2
                        + (v - v_q[inl][None, :]) ** 2)
        best_k = err_b.argmin(axis=0)          # 每个内点的最优对称分支
        lm3d = np.stack([br[k, inl[i]] for i, k in enumerate(best_k)])
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                lm3d, pts2d[inl].reshape(-1, 1, 2), K, None, rvec, tvec)
        except cv2.error:
            pass
    R, _ = cv2.Rodrigues(rvec)
    pc = br @ R.T + tvec.reshape(1, 3)
    z = np.maximum(pc[:, :, 2], 1e-9)
    u = pc[:, :, 0] / z * fx + cx
    v = pc[:, :, 1] / z * fy + cy
    e_min = np.minimum.reduce(
        np.sqrt((u - u_q[None, :]) ** 2 + (v - v_q[None, :]) ** 2), axis=0)
    inl = np.nonzero(e_min < reproj_px)[0]
    if len(inl) < 4:
        return PnPResult(success=False, n_correspondences=n)
    return PnPResult(success=True, R=R, t=tvec.reshape(3),
                     n_inliers=len(inl), inlier_idx=inl,
                     n_correspondences=n,
                     mean_inlier_reproj_px=float(e_min[inl].mean()))
