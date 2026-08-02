"""评估指标：ADD、ADD-S、Proj@5pix、5cm5°。

单位约定：LineMod 模型点与平移均为毫米（mm），5cm 阈值即 50mm。
纯 numpy/scipy 实现，本地 CPU 可测。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.pose_utils import project_points, rotation_angle_deg


def _transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts @ R.T + t.reshape(1, 3)


def add_error(pts: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
              R_pred: np.ndarray, t_pred: np.ndarray) -> float:
    """ADD：同一模型点在 GT 与预测位姿下的平均欧氏距离。

    ADD = (1/|P|) Σ_p ||R p + t - R̂ p - t̂||_2
    与 GSPose calc_add_metric 的非对称分支一致
    （GSPose/misc_utils/metric_utils.py:24-47：mean_dist < 0.1·diameter）。
    """
    d = np.linalg.norm(_transform(pts, R_gt, t_gt)
                       - _transform(pts, R_pred, t_pred), axis=1)
    return float(d.mean())


def adds_error(pts: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
               R_pred: np.ndarray, t_pred: np.ndarray,
               definition: str = "unidirectional") -> float:
    """ADD-S：对称物体的平均最近邻距离。

    definition:
      unidirectional（默认，文献标准）：
        ADD-S = (1/|P|) Σ_p min_{p̂∈P} ||R p + t - R̂ p̂ - t̂||_2
        用 KDTree 在预测点云上查 GT 点云的最近邻，避免 O(N²)——查询方向
        与 GSPose 的 syn 分支一致（metric_utils.py:36-39：cKDTree(model_pred)
        再 query(model_target)）；BOP 官方 `adi` 亦是同一单向定义
        （bop_toolkit_lib/pose_error.py:227-247：cKDTree(pts_est) 上
        query(pts_gt) 取均值）。
      bidirectional_legacy（仅为对照历史数字保留，见 VERIFICATION.md §8.6）：
        0.5·(d_gt→pred + d_pred→gt) 双向 Chamfer。非标准定义，数字不可与
        文献比。旁证其口径与标准不一致：历史结果里 eggbox top40
        ADD-S=100.00% 而同位姿 Proj@5px 只有 90.05%、top1 时
        ADD-S 74.74% vs Proj 37.37%——同一批位姿两个指标互相打架。
    """
    gt = _transform(pts, R_gt, t_gt)
    pred = _transform(pts, R_pred, t_pred)
    d_gt2pred, _ = cKDTree(pred).query(gt, k=1)
    if definition == "unidirectional":
        return float(d_gt2pred.mean())
    if definition == "bidirectional_legacy":
        d_pred2gt, _ = cKDTree(gt).query(pred, k=1)
        return float(0.5 * (d_gt2pred.mean() + d_pred2gt.mean()))
    raise ValueError(f"未知 adds definition: {definition!r}"
                     "（可选 unidirectional | bidirectional_legacy）")


def proj_error_px(pts: np.ndarray, K: np.ndarray,
                  R_gt: np.ndarray, t_gt: np.ndarray,
                  R_pred: np.ndarray, t_pred: np.ndarray) -> float:
    """平均 2D 重投影误差，Proj@5pix 的原始量。"""
    uv_gt = project_points(pts, K, R_gt, t_gt)
    uv_pred = project_points(pts, K, R_pred, t_pred)
    return float(np.linalg.norm(uv_gt - uv_pred, axis=1).mean())


def cm_degree_errors(R_gt: np.ndarray, t_gt: np.ndarray,
                     R_pred: np.ndarray, t_pred: np.ndarray):
    """5cm5° 指标的原始量：平移误差（与 t 同单位）与旋转测地误差（度）。"""
    trans_err = float(np.linalg.norm(t_gt.reshape(3) - t_pred.reshape(3)))
    rot_err = rotation_angle_deg(R_gt, R_pred)
    return trans_err, rot_err


def evaluate_pose(pts: np.ndarray, diameter: float, K: np.ndarray,
                  R_gt: np.ndarray, t_gt: np.ndarray,
                  R_pred: np.ndarray, t_pred: np.ndarray,
                  symmetric: bool = False,
                  add_threshold_ratio: float = 0.1,
                  proj_threshold_px: float = 5.0,
                  cm_threshold_mm: float = 50.0,
                  deg_threshold: float = 5.0,
                  adds_definition: str = "unidirectional") -> Dict[str, float]:
    """单帧全指标评估。

    Returns:
        dict：原始误差（add / proj / trans / rot）与 0/1 命中
        （add_01d / proj_5px / cm_deg）。对称物体（eggbox/glue）的
        add 字段即 ADD-S（定义由 adds_definition 决定，见 adds_error）。
    """
    if symmetric:
        add = adds_error(pts, R_gt, t_gt, R_pred, t_pred,
                         definition=adds_definition)
    else:
        add = add_error(pts, R_gt, t_gt, R_pred, t_pred)
    proj = proj_error_px(pts, K, R_gt, t_gt, R_pred, t_pred)
    trans_err, rot_err = cm_degree_errors(R_gt, t_gt, R_pred, t_pred)
    return {
        "add": add,
        "proj": proj,
        "trans_err": trans_err,
        "rot_err": rot_err,
        "add_01d": float(add < add_threshold_ratio * diameter),
        "proj_5px": float(proj < proj_threshold_px),
        "cm_deg": float(trans_err < cm_threshold_mm and rot_err < deg_threshold),
    }


def aggregate(per_frame: List[Dict[str, float]]) -> Dict[str, float]:
    """帧级结果 → 物体级准确率（ADD(S)@0.1d / Proj@5pix / 5cm5° 百分比）。

    估计失败的帧应以全 0 命中记入 per_frame（分母包含失败帧，与 BOP 协议一致）。
    """
    if not per_frame:
        return {"add_01d": 0.0, "proj_5px": 0.0, "cm_deg": 0.0, "n": 0}
    return {
        "add_01d": 100.0 * float(np.mean([f["add_01d"] for f in per_frame])),
        "proj_5px": 100.0 * float(np.mean([f["proj_5px"] for f in per_frame])),
        "cm_deg": 100.0 * float(np.mean([f["cm_deg"] for f in per_frame])),
        "n": len(per_frame),
    }
