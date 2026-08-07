"""多候选几何一致性择优。

对 Top-K 个候选模板的 RANSAC-PnP 结果，以内点数为几何一致性判据择优：
    i* = argmax_i N_inlier^(i)
合理性：正确位姿使大量对应几何一致，内点数期望 Nω 远高于
错误位姿的偶然内点数（二项分布论证）。
外部支撑：FoundPose（ECCV 2024）官方实现用完全相同的判据——每个候选
模板独立解 PnP 后取 quality 最大者，quality 即 len(inliers)
（foundpose/scripts/infer.py:594-602 + utils/pnp_util.py:79）。

择优判据阶梯（成本相同，一次求解全部拿到，供消融）：
- inlier：      内点数 argmax（主路线）
- inlier_ratio：内点比 = 内点数/总对应数，去掉"匹配多的模板天然内点多"的偏差
- reproj：      内点平均重投影残差 argmin，内点数打平时的判别力更细
- similarity：  不看几何，直接取模板相似度最高者（验证几何验证的必要性；
                历史对照管线的唯一判据，见 VERIFICATION.md §8）
- weighted：    内点数 × 模板相似度，折中方案

纯 CPU，本地可测。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .ransac_pnp import PnPResult


def stable_prior_score(R: np.ndarray, axes: np.ndarray, g: np.ndarray) -> float:
    """稳定摆放先验分（任务 1.2 接入点 B）：候选位姿下物体"朝上轴"与台面法向的偏离。

    R 为候选 w2c 旋转（模型→查询相机）；axes 为稳定姿态的模型系朝上轴集合
    （(K,3) 单位向量），g 为查询系台面法向估计（单方向）。先验分取负最小夹角：
    物体处于任一稳定姿态（朝上轴对齐台面法向）时 ≈0，偏离越大越负。
    """
    d = (R @ axes.T)                       # (3,K)：稳定轴在查询系
    d = d / np.linalg.norm(d, axis=0, keepdims=True)
    cos = np.clip(d.T @ g, -1, 1)
    return -float(np.degrees(np.arccos(cos)).min())


def rank_candidates(results: List[PnPResult], strategy: str = "inlier",
                    keep_failed: bool = False,
                    prior_info: Optional[tuple] = None) -> List[PnPResult]:
    """按择优判据把候选降序排序（top-K best 评估复用）。

    Args:
        results: 各候选模板的 PnPResult
        strategy: inlier | similarity | weighted
        keep_failed: False（默认）= 只保留成功候选，排序第一即
            `select_best_candidate` 的输出（主路线择优，语义不变）；
            True = **失败候选一并参与排序占名额**，用于 topK 窗口——
            历史对照口径里失败项照样占掉 top-3/top-5 的名额
            （见 VERIFICATION.md §8.4）；若在这里先剔除，top3/top5 会
            系统性偏乐观。

    ⚠ 只有 `strategy="similarity"` 与历史对照的 topK 窗口顺序同语义
    （窗口按模板相似度降序，相似度对失败候选同样有定义）。
    `inlier` / `weighted` 下失败候选的 n_inliers 为 0，会被排到末尾，
    窗口是本库自身的端到端候选序，与历史 top1/3/5 不可比。
    """
    pool = list(results) if keep_failed else [r for r in results if r.success]
    if strategy == "inlier":
        key = lambda r: r.n_inliers
    elif strategy == "inlier_ratio":
        # 分母是送进 PnP 的总对应数；0 对应（早退失败）给 0 排末尾
        key = lambda r: (r.n_inliers / r.n_correspondences
                         if r.n_correspondences > 0 else 0.0)
    elif strategy == "reproj":
        # 残差越小越好 → 取负数做降序 key；失败候选残差为 inf 自然垫底
        key = lambda r: -r.mean_inlier_reproj_px
    elif strategy == "similarity":
        key = lambda r: r.template_score
    elif strategy == "weighted":
        # 内点数 × 相似度：相似度为非负权重（负相似度截断为 0，避免符号翻转）
        key = lambda r: r.n_inliers * max(r.template_score, 0.0)
    elif strategy == "prior_inlier":
        # 内点数 + λ·稳定先验分（任务 1.2 接入点 B）。prior_info=(axes, g, lam)，
        # 缺失时退化为纯 inlier（不报错，便于配置对照）。
        if prior_info is None:
            key = lambda r: r.n_inliers
        else:
            axes, g, lam = prior_info
            key = lambda r: (r.n_inliers
                             + lam * stable_prior_score(r.R, axes, g))
    else:
        raise ValueError(f"未知择优策略: {strategy}")
    return sorted(pool, key=key, reverse=True)


def select_best_candidate(results: List[PnPResult],
                          strategy: str = "inlier") -> Optional[PnPResult]:
    """从 K 个候选结果中择优输出最终位姿 [R*|t*]。

    Args:
        results: 各候选模板的 PnPResult（可含失败项，失败项不参与择优）
        strategy: inlier | similarity | weighted
    Returns:
        最优 PnPResult；全部失败时返回 None（上层记为该帧估计失败）
    """
    ranked = rank_candidates(results, strategy=strategy)
    return ranked[0] if ranked else None
