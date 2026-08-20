# 6d-trio —— 迭代数隔离探针：duck/ape/hp 30k→7000

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-trio` |
| Status | `running`（08-20 08:30 启动） |
| Queue row | `experiments/QUEUE.md::6d-trio` |
| Config | onboard `configs/experiments/dense80_refviews128.yaml`（7000/128v）；评测 `configs/current/dense80_depthc_ia.yaml`（decode-all 搭车见 Commands） |
| 依赖 | stage 3 全量链完成（08-20 08:10 ✓，duck/hp 一致性检查已用旧 30k 库跑完） |

## Commands

```bash
bash /tmp/trio_probe.sh    # 备份 → 3 onboard(7000/128v) → trio 子集 → decode-all 搭车
# 对照基线（30k/128v 子集）：outputs/exp_v128_sub/results/{duck,ape,holepuncher}.json
```

## Question

> 30k 实验把迭代数(7000→30000)与视图数(64→128)混杂；refviews 已隔离视图数（128 是精度杠杆）。迭代数本身：7000 vs 30000 哪个好？保留三物体（duck/ape/hp）的 30k/128v 库换成 7000/128v 是否更好？

## Protocol

| 项 | 值 |
|---|---|
| 命令 | 见下 ## Commands |
| 基线 | 30k/128v 子集（exp_v128_sub）：duck 46.67 / ape 60.83 / hp 55.83 |
| 成功线 | 任一物体 ≥ 基线 +3 → 全量重评（fresh cache）；无 +3 判负（30k 对保留三物体无害） |
| 注意 | hp 子集高估 +9.6（120 帧代表性差），hp 的 +3 解读需打折 |

## Live Log

- `08-20 08:30`：stage 3 完成（新 champion 79.66），GPU 空闲，启动 trio。
- `08-20 08:20-10:23`：3 onboard（7000/128v，~7min/物体）→ trio 子集完成。
- `08-20 10:23`：decode-all 搭车首跑崩溃（prescreen=none 需 template_ranking: mast3r，配置耦合），修复后重跑中。
- `08-20 11:20`：**trio 子集结果**：duck 52.50 vs 46.67（**+5.83 通过**）/ ape 60.00 vs 60.83（-0.83 持平）/ hp 46.67 vs 55.83（**-9.16 明显更差**）——迭代数是物体相关的：duck 偏好 7000、hp 偏好 30k、ape 无感。
- `08-20 13:30`：**decode-all 搭车判负**：duck 50.83 vs 52.50（-1.67）/ ape 59.17 vs 60.00（-0.83）——新 7000 库下全解码不优于预筛 top-40（更多自相似坏候选稀释池）；**DINOv2 预筛截断在新库下验证通过，非杠杆**（旧验证结论在新库成立）。

## Result

| 物体 | 7000/128v trio 子集 | 30k/128v 基线 | Δ | 判定 |
|---|---|---|---|---|
| duck | 52.50 | 46.67 | **+5.83** | 通过 → 全量 49.58（30k 45.55，**+4.03**）|
| ape | 60.00 | 60.83 | -0.83 | 持平（噪声带）→ 保留 30k |
| holepuncher | 46.67 | 55.83 | **-9.16** | 更差（7000 伤害 hp）→ 保留 30k |

**结论**：迭代数影响是物体相关的——duck 偏好 7000、hp 偏好 30k（30k 时代"保留 hp"的决定被证明是对的）、ape 无感。30k 归因叙事 = "物体相关过训练敏感性"，不是一刀切毒药。hp 的 -9.16 与子集高估问题无关（同口径对比）。

## Decision

- 结论：`done`（duck 升级，ape/hp 保留 30k）
- 原因：duck 7000 全量 49.58（+4.03，子集预测 +5.83 兑现）；ape 持平（-0.83 噪声带）、hp 7000 更差（-9.16）→ 30k 时代"保留三物体"的决定对 hp 是对的；**迭代数影响物体相关**（duck 偏好 7000、hp 偏好 30k、ape 无感），非一刀切毒药
- 搭车：decode-all 三物体全负/平（duck -1.67 / ape -0.83 / hp -1.67）——预筛 top-40 截断在新库验证通过，非杠杆
- 下一步：duck bank 正式切 7000/128v（已生效）；ape/hp 恢复 30k/128v（已恢复）；30k 归因叙事 = 物体相关过训练敏感性

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
