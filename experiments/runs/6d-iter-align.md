# 6d-iter-align —— 迭代渲染对齐（位姿优化章创新点候选）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-iter-align` |
| Owner | `agent` |
| Status | `running` |
| Started | `2026-08-08 17:20` |
| Finished | `<empty>` |
| Queue row | `experiments/QUEUE.md::6d-iter-align` |

## Question

> 在当前位姿处重渲染 3DGS 新视角 → MASt3R 再匹配 → 渲染深度反投影到模型系
> → 重解 PnP（迭代 2 轮，渲染对齐损失接受/拒绝）能否把精化从净负转正、
> 救回"可恢复类"坏帧？

## Protocol

| 项 | 值 |
|---|---|
| Config | `dense80_depthc_ia.yaml`（base: guided + `solver.iter_align_iters: 2`）|
| Code change | `src/gaussian/pose_refiner.py`：`render_rgbd()`（gsplat RGB+D）；`src/pipeline.py`：`_iter_align()` + `_solve` 接入点 |
| Data split | 120 帧子集（duck 先验，后推 ape/cat）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° / 分帧 GT 误差 |
| Baseline | duck 30.83/81.67/40.83（6d-rng-fix 新口径）|
| Success line | duck ADD ≥ 基线 +3 且无大类崩溃（接受/拒绝门防变差）|

## Commands

```bash
source env.sh
# 单帧/多帧 sanity（/tmp/sanity_iter_align.py，frame_id 列表）
ITER_ALIGN_DEBUG=1 python -u /tmp/sanity_iter_align.py duck 0 243,423,444
# 全量评估
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_ia/cache \
    --out outputs/exp_ia/results/duck.json
```

## Live Log

- `08-08 17:20`：实现（render_rgbd + _iter_align + 接入点 + 配置），192 测试通过
- `08-08 17:30`：单帧 sanity 发现全量互最近邻内存爆炸（Nq×Nr 点积，
  RAM 冲到 ~97GB）→ 加 n_sample_corr 采样上限（事故记录，与 matcher 同纪律）
- `08-08 17:40`：单帧机制验证：corr 1934→1930、PnP 内点 1637→1710 正常收敛；
  好帧被推偏（loss 0.2831→0.3271）→ 拒绝门正确拦截
- `08-08 17:50`：8 好帧扫描：3 帧被接受（1 帧误放行略差、2 帧改进）
- `08-08 18:10`：10 坏帧精选：**117 帧 111→1.8mm / 592 帧 53→3.8mm 大救援；
  222 帧 690→265mm 部分改善；灾难帧（243/507）无救**——机制对"可恢复类"
  坏帧真实有效
- `08-08 18:20`：全量 duck 评估启动（iter_align 全开）
- `08-08 18:40`：duck 全量 **47.50（+16.67）**，帧级核验 63 帧位姿被改动
- `08-08 19:20`：复现验证（新 cache）ADD/Proj 一致，5cm5° 差 1 帧
  （gsplat 光栅化浮点噪声，~1e-4 级位姿差，非 rng 流问题）
- `08-08 20:00`：ape 泛化 **59.17（+11.67）**——非 duck 特例
- `08-08 20:40`：cat/holepuncher/can 泛化 **全正**（+10.83/+5.83/+5.83）——
  **5/5 物体全部提升，无回归**

## Result

| 物体 | 基线（新口径）| iter_align | ΔADD | Δ5cm5° |
|---|---:|---:|---:|---:|
| duck | 30.83/81.67/40.83 | **47.50/81.67/62.50** | **+16.67** | +21.67 |
| ape | 47.5/87.5/55.83 | **59.17/89.17/74.17** | **+11.67** | +18.33 |
| cat | 53.33/85.83/63.33 | **64.17/87.5/79.17** | **+10.83** | +15.83 |
| holepuncher | 50.83/83.33/65.0 | **56.67/85.0/75.83** | **+5.83** | +10.83 |
| can | 93.33/95.0/92.5 | **99.17/97.5/99.17** | **+5.83** | +6.67 |

结果文件：`outputs/exp_ia/results/{duck,duck_re,ape,cat,holepuncher,can}*.json`、
`outputs/exp_weakobj/results/{ape,cat,holepuncher,can}_base.json`。
120 帧子集、rng-fix 后新口径。单帧额外代价 ~0.5s（2 轮渲染+解码）。

## Decision

- 结论：`done`（**通过，位姿优化章核心机制，5/5 泛化**）
- 原因：当前位姿重渲染消除了"离线模板视角 ≠ 查询真实视角"的系统偏差，
  渲染↔查询匹配的对应质量随迭代提升（正反馈）；接受/拒绝门保证好帧
  不被推偏（Proj 全部持平或微升）。坏帧救援集中在"可恢复类"（粗位姿
  偏移 30-100mm 但渲染仍重叠），灾难帧（完全错位）无救
- 已知残留：gsplat 光栅化原子累加的 GPU 级浮点不确定（逐帧位姿 1e-4
  级噪声，主指标稳定，5cm5° 偶见 ±1 帧）
- 下一步：迭代轮数消融（iters 1/2/3）→ 全 13 物体 champion 升级评估

## 消融（08-08 晚，duck 120 帧）

| 配置 | iter_align | refiner | ADD | Proj | 5cm5° | 单帧 |
|---|---|---:|---:|---:|---:|---:|
| 基线 | 关 | 开 | 30.83 | 81.67 | 40.83 | ~7s |
| ia1 | 1 轮 | 开 | 待出 | | | |
| ia2 | 2 轮 | 开 | **47.50** | 81.67 | 62.50 | ~7.5s |
| ia3 | 3 轮 | 开 | **47.50** | 82.50 | 65.83 | ~8s |
| ianr | 2 轮 | **关** | **32.50** | 80.83 | 37.50 | ~6s |

**机制结论（组合效应）**：iter_align 单独只 +1.67（ianr）——+16.67 的
增益来自级联：iter_align 把粗位姿推进正确盆地，refiner 再从好初始点
精细抛光。两者互补（几何对应"导航" + 光度"抛光"），缺一不可。
ia2→ia3 增益递减（5cm5° +3.33），2 轮为性价比甜点。

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
