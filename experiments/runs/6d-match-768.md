# 6d-match-768 —— 匹配分辨率 768（MASt3R 对应质量直接提升）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-match-768` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-12 12:02` |
| Finished | `2026-08-12 14:05` |
| Queue row | `experiments/QUEUE.md::6d-match-768` |

## Question

这次只回答一个问题：

> gap-oracle 结案：**候选池生成（MASt3R 对应质量）是总瓶颈**（top40 池内 GT 择优
> 62.0 ≈ 端到端 61.2）。预筛/解码/择优侧已全部结案——唯一没动的环节是
> MASt3R 本身：输入长边 512 → 768 能否提升对应质量 → 端到端 ADD？
> （1024 已试 OOM 判死 24.8GB；768 冒烟峰值 6.6GB 可行，token 数 2.25×）

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_ia_768.yaml` （champion + matching.image_size: 768 + batch_size 2）|
| Code change | 无（纯配置，`image_size` 已有配置项）|
| Data split | duck 120 帧子集先验证（最弱物体，ia 基线 47.50）|
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5° |
| Baseline | duck 47.50（120 帧 ia 基线，rng-fix 干净口径）|
| Success line | duck ≥ 基线 +3 且无 -3 回退 → 扩 5 弱物体 |

## Commands

```bash
source env.sh
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_ia_768.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_match768/cache \
    --out outputs/exp_match768/results/duck.json
```

## Live Log

- `08-12 12:05`：内存冒烟通过——768 encode 峰值 3.2GB、成对 decode batch4
  峰值 6.6GB（1024 OOM 24.8GB 的历史判死不适用于 768）。
- `08-12 12:02`：登记入队（running），启动 duck 120 帧。
- `08-12 12:10`：**首轮出炉 duck 35.00（-12.50 大判负）**——根因定位：
  查询 resize 768 而模板编码仍原生 512，MASt3R patch 固定 16px → 两侧
  patch 物理尺度不一致 → 互最近邻 desc 语义错位。修复（eeb277e）：
  prepare_templates 模板/alpha 同步 resize 到 long_side + pix_t 换算回
  原生系（coord_map 查表不变）；512 档 scale=1.0 行为不变，202 测试过。
- `08-12 12:12`：修复版复跑启动（新 cache-dir exp_match768b，cfg_hash 不含代码）。
- `08-12 12:50`：复跑崩 IndexError（pix_t 637 > 512）——换算方向 bug：
  `1/tscale` 放大 1.5 倍而非缩小 0.667 倍，改为 `tscale` 本身；测试过重启。
- `08-12 12:52`：二跑 OOM（27GB + 需 4.5GB）——模板同步 768 后成对
  cross-attn ~5×，batch 4 超限；改 batch_size: 2 重启。
- `08-12 13:10`：三跑崩 IndexError 512——_decode_top_desc（引导精化路径）
  自建 pix_t（_tmpl_fg 768 系）漏换算，补乘 _tmpl_scale；重启。
- `08-12 14:05`：**四跑出炉 duck 29.17（-18.33）**——模板同步 + 三处 pix_t
  换算全修后仍大负；Proj 68.33（比首轮 +5）证明 patch 尺度同步有效但
  ADD 更差——768 查询是 512 裁剪插值放大，无新信息且放大伪影污染对应。

## Result

| 指标 | baseline | this run | delta | note |
|---|---:|---:|---:|---|
| ADD（首轮，模板未同步）| 47.50 | 35.00 | -12.50 | 查询 768 vs 模板 512 patch 尺度不一致 |
| ADD（四跑，全修复）| 47.50 | 29.17 | -18.33 | 模板同步 + pix_t 换算全修，仍大负 |
| Proj（四跑）|  | 68.33 |  | patch 同步有效（首轮 63.33）但 ADD 更差 |
| 5cm5°（四跑）|  | 45.00 |  | 与基线持平 |

## Decision

- 结论：`done`（**分辨率侧结案：768 判负，非对应质量杠杆**）
- 原因：
  1. 两版独立大负（模板不同步 -12.50 / 全修复 -18.33）：查询裁剪 512 是
     信息上限，768 只是插值放大——MASt3R patch 更细但内容模糊，desc
     受插值伪影污染，对应精度不升反降；
  2. 模板同步确实有效（Proj 63.33→68.33，patch 尺度一致修复成立），
     但改变不了裁剪信息上限；
  3. 与 superres（判负）、1024（OOM）闭合：**分辨率不是候选池生成
     瓶颈的杠杆**——瓶颈在 MASt3R 模型对应能力本身（弱纹理 patch 匹配
     极限），单目 RGB 侧收口。
- 下一步：检索拆解诊断（池空物体 GT 最近模板是否在 DINOv2 池内）定
  最终叙事；6d-ablation-full 论文消融。
- 产物：`outputs/exp_match768d/results/duck.json`（修复版全量结果）

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
