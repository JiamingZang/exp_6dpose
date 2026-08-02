"""视角采样：8 立方体顶点视角 × 5 平面内旋转 = 40 模板位姿。

采样策略源自 Pos3R 的 8 顶点 × 5 旋转模板采样思想：
1. 视角方向取单位立方体 8 个顶点方向 (±1,±1,±1)/√3，天然覆盖全部 8 个卦限；
   （模板数消融的 80 模板档改用 Fibonacci 球面采样取 16 视角）
2. 每个视角施加 n_inplane 个平面内旋转 φ = k·(360°/n_inplane)，默认 5 个、
   72° 间隔，覆盖图像平面内旋转变化。

每个模板记录 world-to-camera 位姿 P_m（4×4），供 3DGS 渲染与 3D 坐标图使用。
纯 numpy 实现，本地 CPU 可测。

外部参照：同类模板法的密度差异很大——FoundPose（ECCV 2024）官方配置为
57 视点 × 14 平面内旋转 = 798 模板（foundpose/configs/gen_templates/
lmo.json）。40 是 Pos3R 的设计选择而非行业定值，视点密度已列消融
（fibonacci 16 视角档即为此准备）。
"""
from __future__ import annotations

import numpy as np

from .pose_utils import look_at_wc, rotz


def cube_vertex_directions() -> np.ndarray:
    """单位立方体 8 顶点方向，(8,3)，均为单位向量。

    8 个方向分别落在 8 个卦限，保证视角空间的对称均匀覆盖。
    """
    signs = np.array([[sx, sy, sz]
                      for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0)
                      for sz in (-1.0, 1.0)])
    return signs / np.sqrt(3.0)


def fibonacci_directions(n: int) -> np.ndarray:
    """Fibonacci 球面均匀采样 n 个单位方向，(n,3)。

    用于模板数消融中视角数 ≠ 8 的档位（球面均匀采样）。
    """
    idx = np.arange(n, dtype=np.float64)
    # 黄金角递推：保证任意 n 下点分布近似均匀
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * idx / golden
    z = 1.0 - (2.0 * idx + 1.0) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def generate_template_poses(radius: float,
                            viewpoint_mode: str = "cube8",
                            n_viewpoints: int = 8,
                            n_inplane: int = 5) -> np.ndarray:
    """生成全部模板的 w2c 位姿 P_m。

    相机位置 c_v = radius · d_v，光轴指向物体中心（原点）；随后对 w2c 旋转
    左乘 Rz(φ) 实现平面内旋转增强（等价于绕光轴旋转相机）。

    Args:
        radius: 渲染距离 r（由物体包围盒对角线自适应）
        viewpoint_mode: cube8 | fibonacci
        n_viewpoints: 视角数（cube8 模式下强制为 8）
        n_inplane: 平面内旋转数，间隔 360°/n_inplane

    Returns:
        (V*n_inplane, 4, 4) 位姿数组，排列顺序为视角优先、旋转次之：
        index = v * n_inplane + k。
    """
    if viewpoint_mode == "cube8":
        dirs = cube_vertex_directions()
    elif viewpoint_mode == "fibonacci":
        dirs = fibonacci_directions(n_viewpoints)
    else:
        raise ValueError(f"未知视角采样模式: {viewpoint_mode}")

    phis = np.arange(n_inplane) * (2.0 * np.pi / n_inplane)
    poses = []
    for d in dirs:
        T_base = look_at_wc(eye=radius * d, target=np.zeros(3))
        for phi in phis:
            T = T_base.copy()
            Rz = rotz(phi)
            # 平面内旋转作用在相机系：R' = Rz·R, t' = Rz·t
            T[:3, :3] = Rz @ T_base[:3, :3]
            T[:3, 3] = Rz @ T_base[:3, 3]
            poses.append(T)
    return np.stack(poses, axis=0)


def template_intrinsics(image_size: int, fov_deg: float) -> np.ndarray:
    """由视场角构造模板渲染相机内参，**整数像素索引约定**。

    f = (S/2) / tan(fov/2)。该 f 即物理尺度对齐中的 f_ref
    ——CAD 渲染时的虚拟焦距。

    主点为什么是 `S/2 - 0.5` 而不是 `S/2`：本库渲染器 gsplat 的像素中心是
    `(j+0.5, i+0.5)`（证据 `third_party/gsplat/gsplat/cuda/csrc/
    RasterizeToPixels3DGSSerialBatchFwd.cu:108` `const float px =
    (float)out_x + 0.5f;` 与 `third_party/gsplat/gsplat/cuda/
    _torch_impl.py:784` `pixel_coords = ... + 0.5`）。图像中心的连续坐标
    是 `S/2`，换成**整数像素下标**就是 `S/2 - 0.5`。

    半像素约定统一落在内参里（而不是散在各消费方加 ±0.5）：所有拿整数像素
    下标查表/反投影的代码（`matching/depth_lifting.py` 的深度反投影、
    `gaussian/template_renderer.py` pyrender 分支的坐标图）直接用本函数的 K
    即无偏；只有真正调渲染器的地方要先过
    `to_pixel_center_intrinsics`。这样 `lifting: coord_map` 与
    `depth_backproject` 的消融里不再混进与"提升方式"无关的 `0.5·d/f`
    系统偏置（256/fov40° 下约 0.36mm ≈ ADD 阈值的 3.5%）。
    """
    f = (image_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c = image_size / 2.0 - 0.5
    return np.array([[f, 0.0, c],
                     [0.0, f, c],
                     [0.0, 0.0, 1.0]])


def to_pixel_center_intrinsics(K: np.ndarray) -> np.ndarray:
    """整数像素索引内参 → 渲染器的像素中心约定内参（主点 +0.5）。

    gsplat / pyrender 都按"像素 j 的中心在连续坐标 j+0.5"投影，因此渲染时
    必须把 `template_intrinsics` 的主点加回 0.5；反投影/查表侧则用原 K。
    两侧成对使用即闭环无偏（见 `template_intrinsics` docstring）。
    """
    K_out = np.array(K, dtype=np.float64, copy=True)
    K_out[0, 2] += 0.5
    K_out[1, 2] += 0.5
    return K_out
