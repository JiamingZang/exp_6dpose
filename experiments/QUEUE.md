# 实验队列

只从这里领实验。新增实验先加一行，状态从 `todo` 改 `running` 后才能开跑。

## 冠军（当前基线，见 docs/STATE.md）

- **MEAN ADD 71.55**（120 帧子集，回退保护：refiner 精化前后渲染对齐损失择优，变差回退粗位姿）
- 外部目标：GSPose 92.0，差 20.5；弱项 duck 33.3 / ape 45.0 / cat 46.7 / holepuncher 52.5
- 已结案路线（禁止回退重跑）：30k 批量重训（13 物体 3 涨 9 跌）、tz_search（面积比信号死）、NCC 亚像素（噪声主导）、supersample2、择优歧义（duck 池无好假设）、μ 混合锚点、CAD 深度监督单独用、重采样视图、统一背景色

| ID | status | priority | config | run record | question | success line | notes |
|---|---|---:|---|---|---|---|---|
| 6d-full-linemod | running | 1 | `configs/current/dense80_depthc_guided.yaml`（回退保护）| `experiments/runs/6d-full-linemod.md` | 120 帧子集 71.55 能否在全量 LineMod 保持？| 全量 13 物体完成，mean ADD/Proj/5cm5° 入 STATE | 提取中：已完成 ape 1292/benchvise 1390/cam 1377/can 1252/cat 1355；剩余 8 物体补跑（extract_rest8.sh）|
| 6d-vggt-recon | todo | 2 | 待定（src/datasets/vggt_recon.py 新模块）| `experiments/runs/6d-vggt-recon.md` | VGGT 重建替代/辅助 MASt3R 能否提升弱项物体（duck/ape）的匹配精度？| 弱项任一 +5 且无大类崩溃 | 新架构新增模块，先小样本验证再全量 |
| 6d-weak-objects | todo | 2 | 待定 | `experiments/runs/6d-weak-objects.md` | duck/ape/cat 失败帧（proj<5px 占 70%）有无训练/锚点级修复？| 任一弱项 +5，MEAN 不降 | 已确认是匹配精度极限（align 判对率 55%），需换信息源而非测试时微调 |
| 6d-tracking-speed | blocked | 3 | 待新增 | `experiments/runs/6d-tracking-speed.md` | 上帧位姿初始化能否把 7.1s/frame 降到 <1s？| 速度 <1s/frame 且 ADD 下降可解释 | 需要先定 tracking 协议 |

## 已完成（历史记录，勿重跑）

| ID | 结论 | 出处 |
|---|---|---|
| 6d-30k-invdepth-bank | 30k 批量重训整体失败（13 物体 3 涨 9 跌 1 平，glue -60.9）；ape/duck/holepuncher 保留 30k bank | EXPERIMENTS.md「30k 训练 + invdepth 锚点验证」+「refiner 负贡献发现」|
| 6d-30k-can-coordbank | can 回归源=30k 训练（30k+coord 40.0 < 30k+invdepth 63.3 < 7000+invdepth 70.8 < 7000+coord 87.5→裸 PnP 92.5）；can 恢复 .orig bank | `experiments/runs/6d-30k-can-coordbank.md` |
| 6d-refine-two-tier | 裸 PnP 70.97 vs 回退保护 71.55 vs 带 refine 36.7（holepuncher）；refiner 净负贡献，回退保护为通用修复 | EXPERIMENTS.md「refiner 回退保护：全 13 物体」|
| 6d-inlier-ratio | 择优歧义非根因（align_select 修 10 坏 13；duck 池无好假设 align 判对 55%）；择优类实验结案 | EXPERIMENTS.md + verify_align_select 诊断 |
