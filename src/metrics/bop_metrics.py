"""BOP 挑战赛指标（MSSD / MSPD / AR），供与 FoundPose 等 BOP 口径方法对比。

三个组件均逐行对照 bop_toolkit 官方实现移植（纯 numpy，本地 CPU 可测）：

- mssd / mspd：bop_toolkit_lib/pose_error.py:159-207（对 GT 施加对称变换集，
  逐点距离取 max、对称集上取 min）。
- symmetry_transformations：bop_toolkit_lib/misc.py:42-89（离散对称 4x4 直读，
  连续对称按 max_sym_disc_step=0.01 离散化——官方 eval_calc_errors.py:65 默认值）。
- recall 阈值与归一化：官方 eval_bop19_pose.py:46-56 —— MSSD 误差除以物体
  直径后阈值取 0.05:0.05:0.5；MSPD 误差乘 640/图宽（eval_calc_scores.py:303-307）
  后阈值取 5:5:50；命中判据是严格小于（pose_matching.py:66）。

本库不实现 VSD（需要深度渲染器），因此这里的 bop_ar 是 (AR_MSSD+AR_MSPD)/2，
与官方 AR=(AR_VSD+AR_MSSD+AR_MSPD)/3 不同——对外报数必须写明只含两项。
另两处官方评测行为未实现：visib_gt_min 可见性过滤与多实例贪心匹配
（pose_matching.match_poses）。LineMod 每帧单实例不受影响；迁 LM-O
（多实例、重遮挡）时这两处会产生口径差，届时应改用 save_bop_csv 导出
提交文件交官方 bop_toolkit 评分，而不是直接比本模块的 AR 分量。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.pose_utils import project_points

# 官方阈值表（eval_bop19_pose.py:46-56）
MSSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)   # × diameter
MSPD_THRESHOLDS = np.arange(5, 51, 5)           # × (640/im_width 归一化后的 px)
MAX_SYM_DISC_STEP = 0.01                        # eval_calc_errors.py:65


def symmetry_transformations(model_info: Dict,
                             max_sym_disc_step: float = MAX_SYM_DISC_STEP
                             ) -> List[Dict[str, np.ndarray]]:
    """models_info.json 的对称条目 → 变换集 [{"R","t"}]（含恒等）。

    对照 bop_toolkit misc.get_symmetry_transformations：离散对称是 4x4 矩阵
    展平；连续对称给轴+偏移，按步数 ceil(pi/step) 均分 2pi 离散化；两类做
    笛卡尔积组合。无对称条目时只返回恒等变换（普通物体也能走同一条路径）。
    """
    trans_disc = [{"R": np.eye(3), "t": np.zeros((3, 1))}]
    for sym in model_info.get("symmetries_discrete", []):
        m = np.reshape(np.asarray(sym, dtype=np.float64), (4, 4))
        trans_disc.append({"R": m[:3, :3], "t": m[:3, 3].reshape(3, 1)})

    trans_cont = []
    for sym in model_info.get("symmetries_continuous", []):
        axis = np.asarray(sym["axis"], dtype=np.float64)
        offset = np.asarray(sym["offset"], dtype=np.float64).reshape(3, 1)
        n_steps = int(np.ceil(np.pi / max_sym_disc_step))
        step = 2.0 * np.pi / n_steps
        for i in range(n_steps):
            R = Rotation.from_rotvec(i * step * axis / np.linalg.norm(axis)
                                     ).as_matrix()
            trans_cont.append({"R": R, "t": -R @ offset + offset})

    if not trans_cont:
        return trans_disc
    out = []
    for td in trans_disc:
        for tc in trans_cont:
            out.append({"R": tc["R"] @ td["R"],
                        "t": tc["R"] @ td["t"] + tc["t"]})
    return out


def _transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts @ R.T + t.reshape(1, 3)


def mssd_error(pts: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray,
               R_pred: np.ndarray, t_pred: np.ndarray,
               syms: List[Dict[str, np.ndarray]]) -> float:
    """MSSD（bop_toolkit pose_error.mssd）：逐点 max、对称集 min。单位同 pts。"""
    pts_est = _transform(pts, R_pred, t_pred)
    es = []
    for sym in syms:
        R_sym = R_gt @ sym["R"]
        t_sym = (R_gt @ sym["t"]).reshape(3) + t_gt.reshape(3)
        d = np.linalg.norm(pts_est - _transform(pts, R_sym, t_sym), axis=1)
        es.append(d.max())
    return float(min(es))


def mspd_error(pts: np.ndarray, K: np.ndarray,
               R_gt: np.ndarray, t_gt: np.ndarray,
               R_pred: np.ndarray, t_pred: np.ndarray,
               syms: List[Dict[str, np.ndarray]]) -> float:
    """MSPD（bop_toolkit pose_error.mspd）：投影误差逐点 max、对称集 min（px）。"""
    proj_est = project_points(pts, K, R_pred, t_pred)
    es = []
    for sym in syms:
        R_sym = R_gt @ sym["R"]
        t_sym = (R_gt @ sym["t"]).reshape(3) + t_gt.reshape(3)
        d = np.linalg.norm(proj_est - project_points(pts, K, R_sym, t_sym),
                           axis=1)
        es.append(d.max())
    return float(min(es))


def mssd_recall(err: float, diameter: float) -> float:
    """单帧 MSSD recall：10 档阈值（0.05d..0.5d）的命中率均值 ∈ [0,1]。"""
    return float(np.mean(err / diameter < MSSD_THRESHOLDS))


def mspd_recall(err: float, im_width: int) -> float:
    """单帧 MSPD recall：误差乘 640/im_width 归一后 10 档阈值命中率均值。"""
    return float(np.mean(err * 640.0 / im_width < MSPD_THRESHOLDS))


def aggregate_bop(per_frame: List[Dict[str, float]]) -> Dict[str, float]:
    """帧级 {"mssd_recall","mspd_recall"} → 物体级 AR（百分比）。

    失败帧应以 recall=0 记入（分母含失败帧，与主指标 aggregate 同规矩）。
    ar_bop = 两项均值，缺 VSD，报数时须注明口径。
    """
    if not per_frame:
        return {"ar_mssd": 0.0, "ar_mspd": 0.0, "ar_bop": 0.0, "n": 0}
    ar_mssd = 100.0 * float(np.mean([f["mssd_recall"] for f in per_frame]))
    ar_mspd = 100.0 * float(np.mean([f["mspd_recall"] for f in per_frame]))
    return {"ar_mssd": ar_mssd, "ar_mspd": ar_mspd,
            "ar_bop": 0.5 * (ar_mssd + ar_mspd), "n": len(per_frame)}


def save_bop_csv(path, rows: List[Dict]) -> None:
    """按 bop19 提交格式写 CSV（bop_toolkit inout.save_bop_results:380-401）。

    rows: [{"scene_id","im_id","obj_id","score","R"(3x3),"t"(3, mm),"time"}]。
    写出的文件可直接交给官方 bop_toolkit eval 脚本复算，作为本库指标的外部
    对账通道。
    """
    lines = ["scene_id,im_id,obj_id,score,R,t,time"]
    for r in rows:
        lines.append("{},{},{},{},{},{},{}".format(
            r["scene_id"], r["im_id"], r["obj_id"], r["score"],
            " ".join(map(str, np.asarray(r["R"]).flatten().tolist())),
            " ".join(map(str, np.asarray(r["t"]).flatten().tolist())),
            r.get("time", -1)))
    with open(path, "w") as f:
        f.write("\n".join(lines))
