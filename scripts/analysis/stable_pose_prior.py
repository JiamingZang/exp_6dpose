"""任务 1.1：稳定摆放姿态先验——前提验证（离线，无 GPU）。

两步：
1. pybullet 自由落体：对每个物体从 ~500 随机初始朝向落到台面，收集静止姿态
   （模型系→台面系旋转），按旋转距离聚类 → 稳定姿态集合 S_obj + 稳定轴
   （模型系中"朝上"方向 a_S = R_s^T @ z_world）。
2. 击杀判据：LineMod 全部测试帧 GT 姿态 R_gt（模型→相机）。若物体稳定摆放，
   每帧至少一个候选稳定轴方向 R_gt @ a_S 应指向同一相机系方向（重力反方向）。
   - 共识方向 g* = 所有候选方向的 medoid（两两角距离最小者）
   - 每帧最近角 = min_S angle(R_gt @ a_S, g*)
   - 判死：GT 中位角度差 >30°（任务 1.1 判据）；>15° 占比 >30% 视为部分有效

用法: python3 scripts/analysis/stable_pose_prior.py [--objects duck,ape,cat,holepuncher,can] [--n-init 500] [--sim]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.datasets.linemod import LinemodDataset, LINEMOD_OBJECT_IDS

# 与 30k 冠军评估同口径的弱物体 + 强物体对照
DEFAULT_OBJECTS = ["duck", "ape", "cat", "holepuncher", "can"]


def rot_angle_deg(R1, R2):
    d = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(d))


def quat_to_R(q):
    """pybullet xyzw → 旋转矩阵。"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def uniform_quats(n, rng):
    """球面均匀四元数（xyzw）。"""
    u1, u2, u3 = rng.random((3, n))
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=1)
    return q


def simulate_rest_poses(ply_path, n_init, rng, drop_z=120.0, max_steps=4000,
                        seed=0):
    """pybullet 自由落体，返回静止姿态的模型系→台面系旋转列表 (n,3,3)。"""
    import tempfile
    import trimesh
    import pybullet as p
    # pybullet 不支持 .ply，转临时 .obj（trimesh 保真）
    mesh = trimesh.load(ply_path)
    with tempfile.TemporaryDirectory() as td:
        obj_path = Path(td) / "mesh.obj"
        mesh.export(str(obj_path))
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9810.0)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=50)
        # 台面
        plane = p.createCollisionShape(p.GEOM_PLANE, planeNormal=[0, 0, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane)
        # 物体（mm 单位网格）；GEOM_MESH 碰撞用原始三角网格
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(obj_path),
                                       meshScale=[1, 1, 1])
        body = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=shape)
        p.changeDynamics(body, -1, lateralFriction=0.9, spinningFriction=0.02,
                         rollingFriction=0.02, restitution=0.0)

        rests = []
        for i in range(n_init):
            quat = uniform_quats(1, rng)[0]
            p.resetBasePositionAndOrientation(body, [0, 0, drop_z], quat)
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            stable = 0
            for _ in range(max_steps):
                p.stepSimulation()
                lin, ang = p.getBaseVelocity(body)
                if np.linalg.norm(lin) < 1.0 and np.linalg.norm(ang) < 0.5:
                    stable += 1
                    if stable >= 100:
                        break
                else:
                    stable = 0
            pos, quat = p.getBasePositionAndOrientation(body)
            if 1.0 < pos[2] < drop_z + 200:   # 静止在台面上（未飞走）
                rests.append(quat_to_R(np.asarray(quat, dtype=float)))
        p.disconnect()
    return rests


def cluster_poses(rots, deg_thresh=12.0, min_count=10):
    """按"朝上轴方向"聚类（稳定姿态族）：yaw 是连续自由度，全旋转距离会把它切碎。

    每个静止姿态 R_s 的朝上轴 a = R_s^T z_world；同族姿态（站立/侧躺/趴）共享
    一个轴方向，只是 yaw 不同。返回 (代表旋转, 轴方向, 簇大小)，过滤小簇。
    """
    if not rots:
        return []
    z_w = np.array([0.0, 0.0, 1.0])
    axes = np.stack([(R.T @ z_w) / np.linalg.norm(R.T @ z_w) for R in rots])
    n = len(rots)
    kept = []
    used = set()
    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if j not in used:
                cos = np.clip(np.dot(axes[i], axes[j]), -1, 1)
                if np.degrees(np.arccos(cos)) < deg_thresh:
                    cluster.append(j)
        # 簇内轴方向 medoid
        sub = axes[cluster]
        med_axis = sub[int(np.argmin(
            np.degrees(np.arccos(np.clip(sub @ sub.T, -1, 1))).sum(axis=1)))]
        kept.append((rots[cluster[0]], med_axis, len(cluster)))
        used.update(cluster)
    kept = [k for k in kept if k[2] >= min_count]
    kept.sort(key=lambda k: -k[2])
    return kept


def consensus_directions(dirs, deg_thresh=20.0):
    """方向集的方向聚类（贪心）：角距离 < deg_thresh 聚簇，返回每簇 medoid 方向。

    物体可有多个稳定姿态（站立/侧躺），对应多个共识方向（重力反方向在每个
    稳定姿态下的不同表示）；每帧命中任一即可。
    """
    n = len(dirs)
    cos = np.clip(dirs @ dirs.T, -1, 1)
    ang = np.degrees(np.arccos(cos))
    clusters = []
    used = set()
    for i in range(n):
        if i in used:
            continue
        cl = [j for j in range(n) if j not in used and ang[i, j] < deg_thresh]
        med = int(np.argmin(ang[np.ix_(cl, cl)].sum(axis=1)))
        clusters.append(dirs[cl[med]])
        used.update(cl)
    return np.stack(clusters) if clusters else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    ap.add_argument("--n-init", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    objects = args.objects.split(",")
    rng = np.random.default_rng(args.seed)

    print(f"== 任务 1.1 稳定摆放先验（n_init={args.n_init}）==", flush=True)
    for obj in objects:
        ds = LinemodDataset("data/lm", obj, models_dir="models_eval")
        ply = ds.model_path
        # 1) pybullet 自由落体
        rests = simulate_rest_poses(ply, args.n_init, rng)
        print(f"[{obj}] 自由落体 {len(rests)}/{args.n_init} 个静止", flush=True)
        stable = cluster_poses(rests, min_count=max(8, int(args.n_init * 0.02)))
        print(f"[{obj}] 稳定姿态族（轴方向聚类后）: {len(stable)} 个",
              f"大小: {[c for _, _, c in stable]}", flush=True)
        if not stable:
            print(f"[{obj}] 无稳定姿态——跳过判据", flush=True)
            continue
        # 稳定轴（模型系朝上方向）
        axes = np.stack([a for _, a, _ in stable])          # (K,3) 已归一
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        # 2) 击杀判据：全部测试帧 GT
        frames = ds.eval_frames(exclude_refs=True, n_ref=64)
        print(f"[{obj}] 测试帧 {len(frames)}", flush=True)
        cands = []
        for fr in frames:
            for a in axes:
                d = fr.R_gt @ a
                d = d / np.linalg.norm(d)
                cands.append(d)
        cands = np.stack(cands)
        gs = consensus_directions(cands)
        # 每帧最近角 = min over 稳定轴, min over 共识方向
        per_frame = []
        for fr in frames:
            best = min(
                np.degrees(np.arccos(np.clip(np.dot(fr.R_gt @ a, g), -1, 1)))
                for a in axes for g in gs
            )
            per_frame.append(best)
        per_frame = np.array(per_frame)
        med = np.median(per_frame)
        gt15 = (per_frame > 15).mean() * 100
        gt30 = (per_frame > 30).mean() * 100
        verdict = "判死" if med > 30 else ("通过" if med < 15 else "部分有效")
        print(f"[{obj}] 稳定轴方向(模型系):")
        for a in axes:
            print(f"    {np.round(a, 3)}")
        print(f"[{obj}] GT 最近角: 中位 {med:.1f}° | >15°: {gt15:.1f}% | >30°: {gt30:.1f}% → {verdict}",
              flush=True)


if __name__ == "__main__":
    main()
