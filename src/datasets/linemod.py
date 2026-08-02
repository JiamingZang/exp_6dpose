"""LineMod 数据集 loader，标准 BOP 格式。

目录结构（lm_base + lm_models + lm_test_all 解压后）::

    lm/
      camera.json                    # 全局相机参数（fx fy cx cy）
      models_eval/                   # 评估用简化模型
        models_info.json             # diameter / 对称信息（单位 mm）
        obj_000001.ply ...
      test/
        000001/                      # 场景号 = 物体号（LineMod 每场景一物体）
          scene_camera.json          # 每帧 cam_K（3x3 展平）
          scene_gt.json              # 每帧 GT [R(9) | t(3, mm)]
          scene_gt_info.json         # 每帧 bbox_obj / bbox_visib
          rgb/000000.png ...
          mask_visib/000000_000000.png ...

13 物体协议：BOP lm 共 15 物体，惯例剔除 bowl(3)、cup(7)。
本模块只做解析，不做图像解码之外的任何 GPU 依赖，本地 CPU 可测
（单测用 mock 目录结构构造微型数据集）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .ply_io import load_ply

# BOP LineMod 物体名 → obj_id（scene 号与 obj_id 相同）。
# 与 GSPose 的 name2classID 完全一致（GSPose/dataset/inference_datasets.py:345-352）
LINEMOD_OBJECT_IDS: Dict[str, int] = {
    "ape": 1, "benchvise": 2, "cam": 4, "can": 5, "cat": 6,
    "driller": 8, "duck": 9, "eggbox": 10, "glue": 11,
    "holepuncher": 12, "iron": 13, "lamp": 14, "phone": 15,
}
# ADD-S 评估对象。GSPose 从 models_info.json 的 symmetries_*
# 键推断（inference_datasets.py:384-388），13 物体协议下命中的正是这两个
SYMMETRIC_OBJECTS = ("eggbox", "glue")


@dataclass
class Frame:
    """单帧样本：查询图像路径 + 内参 + GT 位姿 + bbox/mask。"""
    frame_id: int
    rgb_path: Path
    K: np.ndarray                 # (3,3)
    R_gt: np.ndarray              # (3,3) w2c（model-to-camera）
    t_gt: np.ndarray              # (3,) mm
    bbox_visib: Optional[np.ndarray] = None   # (4,) x,y,w,h
    mask_path: Optional[Path] = None


class LinemodDataset:
    """单物体的 BOP LineMod 测试集迭代器 + 模型点云/直径。"""

    def __init__(self, root, obj_name: str, split: str = "test",
                 models_dir: str = "models_eval", splits_dir=None):
        if obj_name not in LINEMOD_OBJECT_IDS:
            raise ValueError(f"未知 LineMod 物体: {obj_name}，"
                             f"可选 {sorted(LINEMOD_OBJECT_IDS)}")
        self.root = Path(root)
        self.obj_name = obj_name
        self.obj_id = LINEMOD_OBJECT_IDS[obj_name]
        self.symmetric = obj_name in SYMMETRIC_OBJECTS
        self.scene_dir = self.root / split / f"{self.obj_id:06d}"
        self.models_dir = self.root / models_dir
        # 官方 train split 目录（PVNet 式逐物体划分，防参考帧泄漏）。缺省惯例
        # 为 <root 同级>/splits/<数据集名>，如 data/lm → data/splits/lm。
        self.splits_dir = (Path(splits_dir) if splits_dir is not None
                           else self.root.parent / "splits" / self.root.name)

        info = json.loads((self.models_dir / "models_info.json").read_text())
        # BOP 的 models_info.json 键是字符串形式的 obj_id
        self.model_info = info[str(self.obj_id)]
        # diameter 字段单位 mm（GSPose 乘 1e-3 转米，见
        # inference_datasets.py:368,382；本库全程 mm，与 t_gt/模型点一致，
        # ADD@0.1d 比值无量纲，两种约定等价）
        self.diameter = float(self.model_info["diameter"])   # mm

        self._frames: Optional[List[Frame]] = None
        self._model_pts: Optional[np.ndarray] = None

    # ---- 模型 ----
    @property
    def model_path(self) -> Path:
        return self.models_dir / f"obj_{self.obj_id:06d}.ply"

    def model_points(self, max_points: int = 0) -> np.ndarray:
        """模型顶点（mm）。max_points>0 时均匀抽稀（指标计算加速用）。"""
        if self._model_pts is None:
            verts, _, _ = load_ply(self.model_path)
            self._model_pts = verts
        pts = self._model_pts
        if max_points and len(pts) > max_points:
            step = len(pts) // max_points
            pts = pts[::step][:max_points]
        return pts

    def discrete_symmetry_transforms(self) -> List[np.ndarray]:
        """BOP models_info symmetries_discrete（4x4 物体系变换，不含恒等）。

        对称物体（eggbox/glue）求解端做对称展开锚点，内点判定与
        ADD-S 评估口径一致；无离散对称时返回空列表。
        """
        syms = self.model_info.get("symmetries_discrete") or []
        return [np.asarray(s, dtype=np.float64).reshape(4, 4) for s in syms]

    # ---- 帧 ----
    def frames(self) -> List[Frame]:
        if self._frames is not None:
            return self._frames
        scene_gt = json.loads((self.scene_dir / "scene_gt.json").read_text())
        scene_cam = json.loads((self.scene_dir / "scene_camera.json").read_text())
        gt_info_path = self.scene_dir / "scene_gt_info.json"
        scene_gt_info = (json.loads(gt_info_path.read_text())
                         if gt_info_path.exists() else {})

        frames = []
        for fid_str in sorted(scene_gt, key=lambda s: int(s)):
            fid = int(fid_str)
            cam = scene_cam[fid_str]
            K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
            # LineMod 每场景单物体，但仍按 obj_id 过滤以兼容 LM-O 复用本 loader
            gt_idx = None
            for gi, g in enumerate(scene_gt[fid_str]):
                if int(g["obj_id"]) == self.obj_id:
                    gt_idx = gi
                    break
            if gt_idx is None:
                continue
            g = scene_gt[fid_str][gt_idx]
            R = np.array(g["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.array(g["cam_t_m2c"], dtype=np.float64).reshape(3)

            bbox = None
            if fid_str in scene_gt_info:
                bbox = np.array(
                    scene_gt_info[fid_str][gt_idx]["bbox_visib"], dtype=np.float64)
            mask_path = (self.scene_dir / "mask_visib"
                         / f"{fid:06d}_{gt_idx:06d}.png")
            frames.append(Frame(
                frame_id=fid,
                rgb_path=self.scene_dir / "rgb" / f"{fid:06d}.png",
                K=K, R_gt=R, t_gt=t, bbox_visib=bbox,
                mask_path=mask_path if mask_path.exists() else None,
            ))
        self._frames = frames
        return frames

    # ---- 参考/评测划分（PVNet 式逐物体划分，防参考帧泄漏）----
    def train_split_file(self) -> Path:
        """官方 train split 文件路径：<splits_dir>/<obj_name>_train.txt。"""
        return self.splits_dir / f"{self.obj_name}_train.txt"

    def train_split_ids(self) -> Optional[set]:
        """读取官方 train split 帧号集合；文件不存在返回 None。

        文件格式：一行一个帧号（PVNet 仓库 data/linemod 下的 train.txt，
        帧号即 rgb/{id:06d}.png 的整型 id）。存在时参考帧只从该列表取、
        评测在其补集（测试划分）上进行，参考与评测零重叠。
        """
        path = self.train_split_file()
        if not path.exists():
            return None
        ids = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # 兼容 "000042" / "42" / "rgb/000042.png" 三种写法
            token = Path(line).stem
            ids.add(int(token))
        return ids

    def _uniform_sample_ids(self, n: int) -> set:
        """在 (train split 或全体) 帧池上按 step 均匀抽 n 帧，返回帧号集合。

        3DGS 参考视图与 VGGT 前馈参考帧共享的采样通道：任意"参考帧集合"必
        经此方法产生，保证两条路径的 split 遵守语义完全一致。
        有官方 split 时只从 train 列表抽；无 split 时从全部测试帧抽——此时
        评测须显式排除这批采样参考帧（见 eval_frames）。
        """
        frames = self.frames()
        train_ids = self.train_split_ids()
        pool = ([fr for fr in frames if fr.frame_id in train_ids]
                if train_ids is not None else frames)
        if not pool or n <= 0:
            return set()
        step = max(1, len(pool) // n)
        return {fr.frame_id for fr in pool[::step][:n]}

    def reference_frame_ids(self, n_ref: int = 64) -> set:
        """3DGS 训练参考帧号集合（onboard 与 evaluate 必须给出同一结果）。

        有官方 split：从 train 列表均匀抽 n_ref 帧；无 split：从全部测试帧
        均匀抽 n_ref 帧（此时评测须显式排除这些帧，见 eval_frames）。
        """
        return self._uniform_sample_ids(n_ref)

    def vggt_reference_frame_ids(self, n_ref_images: int = 3) -> set:
        """VGGT 前馈重建参考帧号集合（M=3~5）。

        与 3DGS 参考帧共享 `_uniform_sample_ids` 通道，保证 VGGT 重建输入帧
        亦遵守 split：有 train split 时只从 train 列表抽；无 split 时该 3
        帧同样从测试序列抽出——因此评测必须把它们一并排除，否则 VGGT 几何
        初始化会吃到测试划分帧（P1-1 复审修复）。
        """
        return self._uniform_sample_ids(n_ref_images)

    def eval_frames(self, exclude_refs: bool = True,
                    n_ref: int = 64,
                    extra_exclude_ids: Optional[set] = None) -> List[Frame]:
        """评测帧列表。

        - 有官方 split：评测用测试划分 = 全部帧扣除 train 列表（参考帧取自
          train，天然不重叠，故排除整个 train 列表）；
        - 无官方 split：参考帧从测试序列抽样，须扣除这 n_ref 个采样参考帧，
          否则 3DGS 在评测帧上训练造成泄漏（exclude_refs=False 仅供调试）。

        VGGT 路线额外传入 vggt 参考帧号（`extra_exclude_ids`）以扣除前馈
        重建用的那 3 帧——train_split 存在时它们本就在 train 集里、天然
        被扣除，此参数只在无 split 时真正生效（幂等合并成集合）。
        """
        frames = self.frames()
        if not exclude_refs:
            return frames
        train_ids = self.train_split_ids()
        exclude = (set(train_ids) if train_ids is not None
                   else set(self.reference_frame_ids(n_ref)))
        if extra_exclude_ids:
            exclude |= set(extra_exclude_ids)
        return [fr for fr in frames if fr.frame_id not in exclude]

    def __len__(self) -> int:
        return len(self.frames())

    def __iter__(self):
        return iter(self.frames())
