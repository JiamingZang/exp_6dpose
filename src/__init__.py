"""exp_6dpose：《基于3DGS模板匹配与几何后验证的未知物体6D位姿估计》实验代码。

包结构：
- src.geometry   —— 几何初始化、物理尺度对齐、视角采样
- src.gaussian   —— 3DGS 训练与模板渲染、3D 坐标图
- src.detection  —— SAM/DINOv2 零样本定位
- src.matching   —— MASt3R 稠密 2D-3D 对应
- src.solver     —— Top-K RANSAC-PnP 几何后验证
- src.datasets   —— LineMod/BOP 数据集
- src.metrics    —— ADD/ADD-S/Proj/5cm5° 与 BOP MSSD/MSPD
- src.pipeline   —— 端到端管线（离线 onboard + 在线推理）
"""
