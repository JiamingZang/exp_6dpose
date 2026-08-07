"""任务 4：近似对称量化（duck/ape）——纯分析。

两步：
1. 网格自对齐扫描：对物体网格绕候选轴旋转 θ 后与自身对齐（Umeyama 最优
   旋转后算 Chamfer/点距），找最优近似对称轴与角度（自对齐残差最小）。
2. 失败帧统计：LineMod 失败帧（ADD 超阈值）的预测位姿，绕近似对称轴
   旋转 θ* 后重算 ADD-S——若大量失败帧只差一个对称轴旋转 → 指标口径在
   惩罚近似对称物体（论文讨论章素材）。

用法: python3 scripts/analysis/approx_symmetry.py --objects duck,ape [--theta-step 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.linemod import LinemodDataset


def chamfer_nn(src, dst):
    """src 每点到 dst 的最近距离均值（KD 树近似：直接用全对距离，点数少）。"""
    d = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    return d.min(axis=1).mean()


def best_alignment_rot(src, dst):
    """Umeyama：src → dst 的最优旋转（去均值，纯旋转）。"""
    ms, md = src.mean(0), dst.mean(0)
    H = (dst - md).T @ (src - ms)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def scan_approx_symmetry(verts, n_theta=36):
    """扫描近似对称轴：对球面均匀候选轴，绕轴旋转 θ 后点集与自身 Chamfer。

    注意不做全局旋转补偿（补偿会把旋转抵消，残差恒 0）——近似对称的定义
    是 Rθ·V ≈ V 本身。返回 [(axis, theta_deg, residual)] 按残差升序。
    """
    # 球面候选轴（fibonacci 32 方向）
    idx = np.arange(32, dtype=np.float64)
    golden = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * idx / golden
    z = 1 - (2 * idx + 1) / 32
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    axes = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    diam = np.median(np.linalg.norm(verts - verts.mean(0), axis=1)) * 2
    results = []
    for a in axes:
        for k in range(1, n_theta):
            ang = 2 * np.pi * k / n_theta
            R = _rot_axis(a, ang)
            rot_verts = (R @ verts.T).T
            res = chamfer_nn(rot_verts, verts) / max(diam, 1e-9)
            results.append((a, np.degrees(ang), res))
    results.sort(key=lambda x: x[2])
    return results[:5]


def _rot_axis(a, ang):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="duck,ape")
    ap.add_argument("--n-theta", type=int, default=36)
    args = ap.parse_args()

    for obj in args.objects.split(","):
        ds = LinemodDataset("data/lm", obj, models_dir="models_eval")
        mesh = ds.load_model_mesh() if hasattr(ds, "load_model_mesh") else None
        if mesh is None:
            import trimesh
            mesh = trimesh.load(ds.model_path)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        print(f"\n== {obj}（{len(verts)} 顶点）近似对称扫描 ==", flush=True)
        top = scan_approx_symmetry(verts, n_theta=args.n_theta)
        for a, ang, res in top:
            print(f"  轴={np.round(a, 3)} 角度={ang:.0f}° 自对齐残差={res:.4f}×diam", flush=True)
        # 最优对称轴
        a_best, ang_best, res_best = top[0]
        if res_best > 0.05:
            print(f"[{obj}] 无显著近似对称（残差 {res_best:.4f} > 0.05×diam）", flush=True)
            continue
        # 失败帧统计：cache 位姿绕对称轴旋转后 ADD-S 进阈值占比
        import json
        from src.pipeline import subsample_frames
        frames = subsample_frames(ds.eval_frames(exclude_refs=True, n_ref=64), 120)
        pts = ds.model_points(2000)
        cache_path = Path("outputs/exp_rv2/cache") / f"{obj}.jsonl"
        if not cache_path.exists():
            print(f"[{obj}] 无 cache {cache_path}，跳过失败帧统计", flush=True)
            continue
        by_id = {fr.frame_id: fr for fr in frames}
        R_sym = _rot_axis(a_best, np.radians(ang_best))
        n_fail = n_saved = 0
        for line in open(cache_path):
            d = json.loads(line)
            if "frame_id" not in d or d["frame_id"] not in by_id or not d.get("success"):
                continue
            fr = by_id[d["frame_id"]]
            R, t = np.array(d["R"]), np.array(d["t"])
            err = np.linalg.norm((R @ pts.T + t[:, None]).T
                                 - (fr.R_gt @ pts.T + fr.t_gt[:, None]).T, axis=1).mean()
            if err < 0.1 * ds.diameter:
                continue
            n_fail += 1
            # 绕模型系对称轴旋转预测位姿（模型系旋转 → 查询系 = R @ Rsym^T）
            Rs = R @ R_sym.T
            err_s = np.linalg.norm((Rs @ pts.T + t[:, None]).T
                                   - (fr.R_gt @ pts.T + fr.t_gt[:, None]).T, axis=1).mean()
            if err_s < 0.1 * ds.diameter:
                n_saved += 1
        if n_fail:
            print(f"[{obj}] 失败帧 {n_fail} | 绕对称轴旋转后进 ADD-S 阈值: {n_saved} "
                  f"({n_saved / n_fail * 100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
