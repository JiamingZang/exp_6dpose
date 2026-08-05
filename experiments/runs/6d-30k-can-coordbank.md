# 6d-30k-can-coordbank —— 隔离 can -24.2 回归的来源：30k 训练 vs invdepth 锚点

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-30k-can-coordbank` |
| Owner | `<agent/user>` |
| Status | `planned` |
| Started | `` |
| Finished | `` |
| Queue row | `experiments/QUEUE.md::6d-30k-can-coordbank` |

## Question

这次只回答一个问题：

> can 的 -24.2 ADD 回归，是来自 30k 迭代训练本身（3DGS 权重变化），还是来自
> bank 渲染改用 invdepth 锚点（渲染方式变化）？用 30k 训练权重 + 旧 coord bank
> 重渲染（不重新走 invdepth 锚点路线，只重渲染 coord），与 dc2 基线 / 30k+invdepth
> 结果三方对比来隔离。

## Protocol

| 项 | 值 |
|---|---|
| Config | **不新建配置**——评测复用冠军 configs/current/dense80_depthc_guided.yaml （本实验变量是 bank 锚点口径，不是评测配置） |
| Code change | `scripts/data/rebuild_bank_fixed_views.py` 新增 `--anchor {invdepth,coord}`（2026-08-04 已实现）：coord = μ 位置混合（历史口径复活，仅此消融用），输出 npz 带 `anchor_mode` 字段追溯；invdepth 默认行为不变 |
| Data split | can，120 帧子集 |
| Metrics | ADD |
| Baseline | dc2 can ADD 87.5；30k+invdepth can ADD 63.3（见 `6d-30k-invdepth-bank.md`） |
| Success line | 三方对比能定位回归来源（训练 or 锚点渲染），可解释后决定是否推全量 9 物体 30k 流水线 |

## Commands

**在 GPU 服务器上执行**（本机无权重/无 CUDA）。前置假设：can 的 30k
checkpoint 已在 `outputs/templates/can_3dgs_cad_80t_sa.pt`（30k 训练后
onboard 落盘位置，跑前 `ls -la` 确认；若仍是 7000 iter 旧权重，先按
`6d-30k-invdepth-bank` 同口径重训 can 30k）。

```bash
# 1 备份当前 bank（30k+invdepth 口径），实验后要能还原
cp outputs/templates/can_3dgs_cad_80t_sa.npz outputs/templates/can_3dgs_cad_80t_sa.npz.invdepth30k

# 2 用 30k 权重 + coord（μ 混合）锚点重建 bank（视图固定用 .orig，脚本默认）
python scripts/data/rebuild_bank_fixed_views.py \
    --config configs/current/dense80_depthc_guided.yaml \
    --objects can --anchor coord

# 3 评测 120 帧子集
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_guided.yaml --objects can

# 4 还原 invdepth bank（无论结果如何，冠军口径必须复原）
cp outputs/templates/can_3dgs_cad_80t_sa.npz.invdepth30k outputs/templates/can_3dgs_cad_80t_sa.npz
```

判读：coord-bank ADD ≈ 87.5 → 回归来自 invdepth 锚点渲染（30k 训练无害，
可推全量但需先修锚点）；≈ 63.3 → 回归来自 30k 训练本身；介于中间 → 两者
都有贡献，记录各自占比。

## Live Log

- `<time>`：<启动/中断/恢复/异常/观察>

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## Decision

- 结论：`keep/reject/retry/blocked`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
