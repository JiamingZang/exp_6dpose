# 实验队列

只从这里领实验。新增实验先加一行，状态从 `todo` 改 `running` 后才能开跑。

| ID | status | priority | config | run record | question | success line | notes |
|---|---|---:|---|---|---|---|---|
| 6d-inlier-ratio | todo | 1 | `configs/current/dense80_depthc_guided.yaml` + 新增选择策略配置 | `experiments/runs/6d-inlier-ratio.md` | PnP 候选择优从 raw inlier 改为 inlier ratio 是否提升 D 类和整体稳定性？ | 120 帧子集 mean ADD 不低于 69.36，D 类任一提升且无大类崩溃 | 先小样本 ape/duck/holepuncher，再 13 物体 |
| 6d-refine-two-tier | todo | 1 | `configs/current/dense80_depthc_guided.yaml` / `configs/current/dense80_depthc_norefine.yaml` | `experiments/runs/6d-refine-two-tier.md` | 纯几何档 vs 几何+精化档的真实贡献是多少？ | 输出两档同口径表，论文能直接引用 | 必须注明 120 帧/全量、是否 guided |
| 6d-30k-invdepth-bank | done | 1 | `configs/archive/dense80_dc_b4.yaml`（30k + invdepth 锚点 + train_fp） | `experiments/runs/6d-30k-invdepth-bank.md` | 30k 迭代 + invdepth 锚点重训后 3 物体（ape/can/duck，先行 holepuncher）相对 dc2 基线如何变化？ | ape/duck/holepuncher 提升，can 不崩溃 | 补记：2026-08-04 commit 2dfe5fe 时未走 QUEUE 流程直接跑，此行为事后补登 |
| 6d-30k-can-coordbank | todo | 1 | 不新建配置：复用 `configs/current/dense80_depthc_guided.yaml`，变量是 bank（`rebuild_bank_fixed_views.py --anchor coord` 已实现） | `experiments/runs/6d-30k-can-coordbank.md` | can -24.2 回归是来自 30k 训练本身还是 invdepth 锚点渲染？隔离两者贡献 | 定位到具体来源（训练 or 锚点），可解释后决定是否推全量 9 物体 30k 流水线 | 来源：docs/EXPERIMENTS.md「30k 训练 + invdepth 锚点验证」小节的"待办"；GPU 服务器命令已写入 run 记录 |
| 6d-full-linemod | todo | 2 | `configs/current/dense80_depthc_guided.yaml` | `experiments/runs/6d-full-linemod.md` | 120 帧子集数字能否在全量 LineMod 保持？ | 全量 13 物体完成，mean ADD/Proj/5cm5° 入 STATE | 预计 15-18h，先确认磁盘和 GPU |
| 6d-tracking-speed | blocked | 3 | 待新增 | `experiments/runs/6d-tracking-speed.md` | 上帧位姿初始化能否把 7.1s/frame 降到 <1s？ | 速度 <1s/frame 且 ADD 下降可解释 | 需要先定 tracking 协议 |
