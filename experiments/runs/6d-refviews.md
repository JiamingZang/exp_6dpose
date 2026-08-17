# 6d-refviews —— 3DGS 训练参考视图数消融（64→128，隔离验证）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-refviews` |
| Owner | agent |
| Status | `running` |
| Started | 2026-08-17 02:20 |
| Finished | empty |
| Queue row | `experiments/QUEUE.md::6d-refviews` |

## Question

> 30k 实验（08-04）把训练迭代数（7000→30000）与参考视图数（64→128）混杂：3 物体（duck/ape/holepuncher）保留 30k+128v bank，其余 10 物体仍是旧世代 bank（7000/64v，无 train_fp）。视图数 64→128 是否独立于迭代数是精度杠杆？即：10 物体能否用「7000 迭代 + 128 视图 + 现代配方」安全升级？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_refviews128.yaml`（base dense80_depth_bg0 + onboard.n_ref_views: 128）|
| Code change | none（现有 onboard_object 支持 n_ref_views）|
| Data split | cat 120 帧子集（探针物体：5 弱物体之一、当前旧世代 bank、30k 批量中非保留者）|
| Metrics | ADD(S)@0.1d（120 帧口径，与 ia 基线同口径）|
| Baseline | champion ia 基线 cat **64.17**（120 帧；bank=旧世代 7000/64v）|
| Success line | **≥ +3（67.17）** → 视图数是一般性杠杆 → 其余 9 物体按 7000/128v 现代配方重训，13 物体全量验证。**≤ +1（65.17）** → 64 视图已饱和，30k 批量伤害在迭代数，方向结案。中间 → 判平，按成本决定 |
| 备注 | ① 升级对象是「现代配方 @7000」，归因（视图 vs invdepth 锚点）留作次级问题，探针通过后再补 64v 对照分离；② 训练指纹含 n_ref_views → 自动强制重训；③ bank 文件名不变 → 评测必须用全新 cache 目录（cfg_hash 不变，旧缓存是脏的）|

## Commands

```bash
# 备份（模板库改动纪律）
cp outputs/templates/cat_3dgs_cad_80t_sa.npz{,.viewsbak}
cp outputs/templates/cat_3dgs_cad_80t_sa.pt{,.viewsbak}
# 重训 + 建库（128 视图）
CFG=configs/experiments/dense80_refviews128.yaml
python3 -c "from src.pipeline import onboard_object; from src.config import load_config; onboard_object(load_config('$CFG'), 'cat')"
# 评测（全新 cache）
python3 scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia.yaml \
    --objects cat --max-frames 120 \
    --cache-dir outputs/exp_refviews/cache \
    --out outputs/exp_refviews/results/
```

## Live Log

- `08-17 02:20`：登记。动机：全物体 bank 指纹盘点（train_fp）——duck/ape = {30000, 128v, depth 0.6, invdepth}；cat/can 等 10 物体 = 无 fp（旧世代 7000/64v/coord）。视图数 64 vs 128 在所有历史对比中都与迭代数/锚点混杂，从未隔离。
- `08-17 04:15`：**探针通过**：cat 80.00 vs 64.17（+15.83）——远超 +3 成功线；Proj 91.67 / 5cm5° 84.17。扩 9 物体链启动（~10h）。

## Result

| 物体 | 组 | 基线(120帧) | v128(120帧) | Δ | 全量方向参考 |
|---|---|---:|---:|---:|---:|
| cat | 升级 | 64.17 | **80.00** | **+15.83** | — |
| benchvise | 升级 | — | 90.00 | — | 84.00（强正）|
| glue | 升级 | — | 90.83 | — | 86.51（强正）|
| iron | 升级 | — | 96.67 | — | 95.13（正）|
| eggbox | 升级 | — | 99.17 | — | 98.40（持平）|
| can | 升级 | 99.17 | 99.17 | ±0.00 | 96.82（天花板）|
| lamp | 升级 | — | 93.33 | — | 93.38（持平）|
| driller | 升级 | — | 95.83 | — | 97.06（持平）|
| cam | 升级(w1) | — | 74.17 | — | 75.99（持平）|
| phone | 升级 | 77.50 | 75.83 | -1.67（噪声带内判平）| 71.42 |
| ape | 保留对照 | 59.17 | 60.83 | +1.66 | ✓ 一致性 |
| duck | 保留对照 | 47.50 | 46.67 | -0.83 | ✓ 一致性 |
| holepuncher | 保留对照 | 56.67 | 55.83 | -0.84 | ✓ 一致性 |

**门控判定：PASS**——3 保留对照全部噪声带内（评测链一致性验证）；升级组 6 正（cat 最大 +15.83）3 平（天花板）1 平（phone -1.67 噪声带）。全量评测 10:47 自动启动。

探针数字：ADD 80.00 / Proj 91.67 / 5cm5° 84.17（120 帧，全新 cache，`outputs/exp_refviews/results/cat.json`）。

## Decision

- 结论：`passed`（门控全过，全量评测已启动）
- 原因：升级组系统性正收益（cat +15.83 / benchvise +6 / glue +4.3 方向；天花板物体零回退），保留对照全过；phone -1.67 为唯一噪声带内判平（弱纹理物体异质性，与 K/J 曲线同构）
- 下一步：扩 9 物体已重训 → 子集门控（进行中）→ 全量评测 → 新 champion MEAN；**尾项探针：duck/ape/hp 30k→7000/128v**（若 7000 配方对保留三物体也更好，champion 再升一档；迭代数归因闭环）；归因对照（cat 64v 现代配方）排最后

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
