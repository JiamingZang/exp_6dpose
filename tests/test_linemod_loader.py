"""LineMod BOP 格式 loader 单测：mock 目录结构 + 微型 PLY。"""
import json
import struct

import numpy as np
import pytest

from src.datasets.linemod import (LINEMOD_OBJECT_IDS, SYMMETRIC_OBJECTS,
                                  LinemodDataset)
from src.datasets.ply_io import load_ply, sample_mesh_points

# LineMod 标准内参
CAM_K = [572.4114, 0.0, 325.2611, 0.0, 573.5704, 242.0490, 0.0, 0.0, 1.0]


def _write_ascii_ply(path, with_faces=True):
    """带颜色的三角面片微型 PLY（ascii）。"""
    verts = [(0, 0, 0, 255, 0, 0), (10, 0, 0, 0, 255, 0),
             (0, 10, 0, 0, 0, 255), (0, 0, 10, 255, 255, 0)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    lines = ["ply", "format ascii 1.0",
             f"element vertex {len(verts)}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue"]
    if with_faces:
        lines += [f"element face {len(faces)}",
                  "property list uchar int vertex_indices"]
    lines.append("end_header")
    for v in verts:
        lines.append(" ".join(map(str, v)))
    if with_faces:
        for f in faces:
            lines.append("3 " + " ".join(map(str, f)))
    path.write_text("\n".join(lines) + "\n")


def _write_binary_ply(path):
    """binary_little_endian PLY（BOP 模型的实际编码）。"""
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
                     dtype=np.float32)
    faces = [(0, 1, 2), (1, 2, 3)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for v in verts:
            f.write(struct.pack("<fff", *v))
        for face in faces:
            f.write(struct.pack("<B", 3) + struct.pack("<iii", *face))


@pytest.fixture
def mock_bop_root(tmp_path):
    """构造最小 BOP lm 目录：ape (obj_000001) 场景 2 帧。"""
    root = tmp_path / "lm"
    models = root / "models_eval"
    models.mkdir(parents=True)
    _write_ascii_ply(models / "obj_000001.ply")
    (models / "models_info.json").write_text(json.dumps({
        "1": {"diameter": 102.099, "min_x": -37.9, "min_y": -38.7,
              "min_z": -45.8, "size_x": 75.8, "size_y": 77.5, "size_z": 91.6},
    }))

    scene = root / "test" / "000001"
    (scene / "rgb").mkdir(parents=True)
    (scene / "mask_visib").mkdir()
    # 两帧假图（loader 只存路径，不解码，可以是空文件）
    for i in range(2):
        (scene / "rgb" / f"{i:06d}.png").write_bytes(b"")
    (scene / "mask_visib" / "000000_000000.png").write_bytes(b"")
    # 注意：帧 1 故意不给 mask，测试 mask_path=None 分支

    R0 = np.eye(3).reshape(-1).tolist()
    R1 = [0, -1, 0, 1, 0, 0, 0, 0, 1.0]
    (scene / "scene_gt.json").write_text(json.dumps({
        "0": [{"obj_id": 1, "cam_R_m2c": R0, "cam_t_m2c": [10, 20, 500]}],
        "1": [{"obj_id": 1, "cam_R_m2c": R1, "cam_t_m2c": [0, 0, 700]}],
    }))
    (scene / "scene_camera.json").write_text(json.dumps({
        "0": {"cam_K": CAM_K, "depth_scale": 1.0},
        "1": {"cam_K": CAM_K, "depth_scale": 1.0},
    }))
    (scene / "scene_gt_info.json").write_text(json.dumps({
        "0": [{"bbox_obj": [100, 100, 60, 80], "bbox_visib": [102, 101, 58, 79]}],
        "1": [{"bbox_obj": [200, 150, 50, 50], "bbox_visib": [200, 150, 50, 50]}],
    }))
    return root


def test_object_id_table():
    assert LINEMOD_OBJECT_IDS["ape"] == 1
    assert LINEMOD_OBJECT_IDS["phone"] == 15
    assert len(LINEMOD_OBJECT_IDS) == 13          # 13 物体协议
    assert 3 not in LINEMOD_OBJECT_IDS.values()   # bowl 剔除
    assert 7 not in LINEMOD_OBJECT_IDS.values()   # cup 剔除
    assert set(SYMMETRIC_OBJECTS) == {"eggbox", "glue"}


def test_loader_frames(mock_bop_root):
    ds = LinemodDataset(mock_bop_root, "ape")
    assert len(ds) == 2
    assert not ds.symmetric
    assert np.isclose(ds.diameter, 102.099)

    f0, f1 = ds.frames()
    assert f0.frame_id == 0
    assert f0.rgb_path.name == "000000.png" and f0.rgb_path.exists()
    assert np.allclose(f0.K, np.array(CAM_K).reshape(3, 3))
    assert np.allclose(f0.R_gt, np.eye(3))
    assert np.allclose(f0.t_gt, [10, 20, 500])
    assert np.allclose(f0.bbox_visib, [102, 101, 58, 79])
    assert f0.mask_path is not None and f0.mask_path.exists()

    # 帧 1：旋转矩阵按行主序 reshape；无 mask 文件时为 None
    assert np.allclose(f1.R_gt, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert f1.mask_path is None


def test_loader_model_points(mock_bop_root):
    ds = LinemodDataset(mock_bop_root, "ape")
    pts = ds.model_points()
    assert pts.shape == (4, 3)
    assert np.allclose(pts[1], [10, 0, 0])


def test_loader_unknown_object(mock_bop_root):
    with pytest.raises(ValueError):
        LinemodDataset(mock_bop_root, "teapot")


def test_ply_ascii_roundtrip(tmp_path):
    p = tmp_path / "m.ply"
    _write_ascii_ply(p)
    verts, colors, faces = load_ply(p)
    assert verts.shape == (4, 3)
    assert faces.shape == (4, 3)
    assert colors.shape == (4, 3)
    assert np.isclose(colors[0, 0], 1.0)      # red=255 → 1.0


def test_ply_binary_roundtrip(tmp_path):
    p = tmp_path / "m.ply"
    _write_binary_ply(p)
    verts, colors, faces = load_ply(p)
    assert verts.shape == (4, 3)
    assert colors is None
    assert np.allclose(faces, [[0, 1, 2], [1, 2, 3]])


def test_sample_mesh_points_on_surface(tmp_path):
    """面积加权采样的点必须落在三角面上（重心坐标性质）。"""
    p = tmp_path / "m.ply"
    _write_ascii_ply(p)
    verts, colors, faces = load_ply(p)
    pts, pc = sample_mesh_points(verts, faces, 1000, colors=colors,
                                 rng=np.random.default_rng(0))
    assert pts.shape == (1000, 3)
    assert pc.shape == (1000, 3)
    # 四面体网格的所有采样点都应在包围盒内且非负侧
    assert pts.min() >= -1e-9
    assert pts.max() <= 10 + 1e-9
    # 无面片退化路径
    pts2, _ = sample_mesh_points(verts, None, 16)
    assert pts2.shape == (16, 3)


# ---------------------------------------------------------------------------
# 参考/评测划分（防参考帧泄漏）
# ---------------------------------------------------------------------------
def test_no_split_excludes_sampled_references(mock_bop_root):
    """无官方 split：参考帧从测试序列抽样，评测须排除这些帧。"""
    ds = LinemodDataset(mock_bop_root, "ape")
    assert ds.train_split_ids() is None
    ref_ids = ds.reference_frame_ids(n_ref=1)     # 2 帧抽 1 帧作参考
    assert len(ref_ids) == 1
    eval_frames = ds.eval_frames(exclude_refs=True, n_ref=1)
    eval_ids = {fr.frame_id for fr in eval_frames}
    # 参考帧与评测帧零重叠
    assert ref_ids.isdisjoint(eval_ids)
    assert len(eval_frames) == 1


def test_exclude_refs_false_keeps_all(mock_bop_root):
    ds = LinemodDataset(mock_bop_root, "ape")
    assert len(ds.eval_frames(exclude_refs=False)) == 2


def test_official_split_file_takes_effect(mock_bop_root, tmp_path):
    """存在 <splits_dir>/ape_train.txt 时：参考帧只取自 train 列表，
    评测在其补集（测试划分）上进行。"""
    splits_dir = mock_bop_root.parent / "splits" / "lm"
    splits_dir.mkdir(parents=True)
    # 把帧 1 指定为 train（参考），帧 0 应落在评测（测试划分）
    (splits_dir / "ape_train.txt").write_text("000001\n")
    ds = LinemodDataset(mock_bop_root, "ape")
    assert ds.train_split_ids() == {1}
    assert ds.reference_frame_ids(n_ref=64) == {1}   # 只从 train 列表取
    eval_ids = {fr.frame_id for fr in ds.eval_frames(exclude_refs=True)}
    assert eval_ids == {0}                            # 评测测试划分


def test_split_dir_override(mock_bop_root, tmp_path):
    """splits_dir 显式参数生效（帧号写法兼容 rgb/xxx.png）。"""
    custom = tmp_path / "mysplits"
    custom.mkdir()
    (custom / "ape_train.txt").write_text("rgb/000000.png\n")
    ds = LinemodDataset(mock_bop_root, "ape", splits_dir=custom)
    assert ds.train_split_ids() == {0}
