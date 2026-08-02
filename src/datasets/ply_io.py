"""极简 PLY 读取与网格表面均匀采样（CAD 点云均匀采样）。

只支持 BOP 模型文件实际用到的两种编码：ascii 1.0 与 binary_little_endian 1.0，
读取 vertex 的 x/y/z（以及可选的 red/green/blue 颜色，供 3DGS 初始化）与
face 索引。避免引入 trimesh 等重依赖，保证本地 CPU 单测零负担。
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# PLY 标量类型 → (struct 格式符, 字节数)
_PLY_TYPES = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def load_ply(path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """读取 PLY 文件。

    Returns:
        vertices (N,3) float64，
        colors (N,3) float64 归一到 [0,1] 或 None，
        faces (M,3) int64 或 None
    """
    path = Path(path)
    with open(path, "rb") as f:
        # ---- 解析 header ----
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError(f"{path} 不是 PLY 文件")
        fmt = None
        elements = []           # [(name, count, [(prop_name, type, is_list, list_count_type)])]
        cur_props = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            tok = line.split()
            if not tok or tok[0] == "comment":
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                cur_props = []
                elements.append((tok[1], int(tok[2]), cur_props))
            elif tok[0] == "property":
                if tok[1] == "list":
                    cur_props.append((tok[4], tok[3], True, tok[2]))
                else:
                    cur_props.append((tok[2], tok[1], False, None))
        if fmt not in ("ascii", "binary_little_endian"):
            raise ValueError(f"不支持的 PLY 编码: {fmt}")

        verts, colors, faces = None, None, None
        for name, count, props in elements:
            if fmt == "ascii":
                rows = [f.readline().decode("ascii").split() for _ in range(count)]
                if name == "vertex":
                    verts, colors = _parse_vertex_ascii(rows, props)
                elif name == "face":
                    faces = np.array([[int(r[1]), int(r[2]), int(r[3])]
                                      for r in rows], dtype=np.int64)
            else:
                if name == "vertex":
                    verts, colors = _parse_vertex_binary(f, count, props)
                elif name == "face":
                    faces = _parse_face_binary(f, count, props)
                else:
                    _skip_element_binary(f, count, props)
    return verts, colors, faces


def _prop_names(props):
    return [p[0] for p in props]


def _parse_vertex_ascii(rows, props):
    names = _prop_names(props)
    arr = np.array(rows, dtype=np.float64)
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    verts = arr[:, [ix, iy, iz]]
    colors = None
    if all(c in names for c in ("red", "green", "blue")):
        idx = [names.index(c) for c in ("red", "green", "blue")]
        colors = arr[:, idx] / 255.0
    return verts, colors


def _parse_vertex_binary(f, count, props):
    # vertex 元素不含 list 属性，可一次性用结构化 dtype 读入
    fields = []
    for pname, ptype, is_list, _ in props:
        if is_list:
            raise ValueError("vertex 元素不应包含 list 属性")
        fmt_char, _ = _PLY_TYPES[ptype]
        fields.append((pname, "<" + fmt_char))
    dt = np.dtype(fields)
    data = np.frombuffer(f.read(dt.itemsize * count), dtype=dt, count=count)
    verts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    names = _prop_names(props)
    colors = None
    if all(c in names for c in ("red", "green", "blue")):
        colors = np.stack([data["red"], data["green"], data["blue"]],
                          axis=1).astype(np.float64) / 255.0
    return verts, colors


def _parse_face_binary(f, count, props):
    faces = np.empty((count, 3), dtype=np.int64)
    for i in range(count):
        row = _read_binary_row(f, props)
        idxs = row[0]                      # 第一个属性即 vertex_indices list
        if len(idxs) != 3:
            raise ValueError("仅支持三角面片")
        faces[i] = idxs
    return faces


def _skip_element_binary(f, count, props):
    for _ in range(count):
        _read_binary_row(f, props)


def _read_binary_row(f, props):
    row = []
    for pname, ptype, is_list, count_type in props:
        if is_list:
            cfmt, csize = _PLY_TYPES[count_type]
            n = struct.unpack("<" + cfmt, f.read(csize))[0]
            vfmt, vsize = _PLY_TYPES[ptype]
            row.append(struct.unpack("<" + vfmt * n, f.read(vsize * n)))
        else:
            vfmt, vsize = _PLY_TYPES[ptype]
            row.append(struct.unpack("<" + vfmt, f.read(vsize))[0])
    return row


def sample_mesh_points(verts: np.ndarray, faces: Optional[np.ndarray],
                       n: int, colors: Optional[np.ndarray] = None,
                       rng: Optional[np.random.Generator] = None):
    """CAD 网格表面均匀采样 n 个点（P_CAD）。

    有面片时按三角形面积加权采样（真正的表面均匀分布）；无面片时退化为
    顶点随机抽样。颜色按三角形顶点重心插值 / 顶点颜色带出，供 3DGS 初始化。

    Returns:
        points (n,3), point_colors (n,3) 或 None
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if faces is None or len(faces) == 0:
        idx = rng.choice(len(verts), size=n, replace=len(verts) < n)
        pc = None if colors is None else colors[idx]
        return verts[idx].copy(), pc

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if area.sum() <= 0:
        idx = rng.choice(len(verts), size=n, replace=len(verts) < n)
        pc = None if colors is None else colors[idx]
        return verts[idx].copy(), pc
    fidx = rng.choice(len(faces), size=n, p=area / area.sum())
    # 三角形内均匀采样的重心坐标（sqrt 技巧）
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    pts = (w0[:, None] * v0[fidx] + w1[:, None] * v1[fidx]
           + w2[:, None] * v2[fidx])
    pc = None
    if colors is not None:
        c0, c1, c2 = (colors[faces[fidx, 0]], colors[faces[fidx, 1]],
                      colors[faces[fidx, 2]])
        pc = w0[:, None] * c0 + w1[:, None] * c1 + w2[:, None] * c2
    return pts, pc
