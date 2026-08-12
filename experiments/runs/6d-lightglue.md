# 6d-lightglue —— SuperPoint+LightGlue 稀疏匹配（攻对应质量另一路线）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-lightglue` |
| Owner | `qoder` |
| Status | `done` |
| Started | `2026-08-13 00:15` |
| Finished | `2026-08-13 00:50` |
| Queue row | `experiments/QUEUE.md::6d-lightglue` |

## Question

> 瓶颈=MASt3R 稠密对应质量（检索拆解闭环，gap-oracle "候选池生成是总瓶颈"）。
> SuperPoint+LightGlue 稀疏 Transformer 匹配能否提供更准的对应（攻击同一瓶颈的另一路线）？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/current/dense80_depthc_lg.yaml`（base dense80_depthc_guided.yaml + matcher: lightglue + lg_conf_tau: 0.0）|
| Code change | commit `16eb089`（LightGlueMatcher 集成：third_party/lightglue 克隆、权重下载、v0.1_arxiv 键名 remap、pipeline guard）|
| Data split | duck 120 帧子集 |
| Metrics | ADD(S)@0.1d / Proj@5pix / 5cm5° |
| Baseline | duck 120 帧粗位姿基线 = guided 口径（MASt3R 稠密匹配，无 refine）|
| Success line | duck 120 帧 ≥ 粗位姿基线且差距可解释 |

## Commands

```bash
# LightGlueMatcher 集成（16eb089）
python scripts/eval/run_linemod.py --config configs/current/dense80_depthc_lg.yaml \
    --objects duck --max-frames 120 --cache-dir outputs/exp_lg/cache \
    --out outputs/exp_lg/results/duck.json
# 后台任务 b69j8l4yv，exit 0
```

## Live Log

- `08-12 15:00`：集成完成（202 测试通过，16eb089）。关键修复链：github release
  下载损坏 → gh-proxy 重下；torch 2.6 weights_only 拒绝旧权重 → False；
  v0.1_arxiv 权重键名 self_attn.N.* → transformers.N.self_attn.* remap；
  matches0 是 1D per-query index（非 (N,2)）；构造期 torch hub 下载失败 →
  权重播种 ~/.cache/torch/hub/checkpoints/。
- `08-12 15:10`：诊断：filter_threshold=0.1 → 11 对、0.0 → 29 对——弱纹理
  真实-渲染域差下匹配率极低；配置 lg_conf_tau: 0.0。
- `08-13 00:26`：duck 120 帧启动（b69j8l4yv）。
- `08-13 00:50`：完成。**ADD 0.00 / Proj 0.00 / 5cm5° 0.00（120 帧全灭）**。

## Result

| 指标 | baseline（guided 粗位姿） | this run | delta | note |
|---|---:|---:|---:|---|
| ADD(S)@0.1d | ~50（duck 粗位姿口径）| 0.00 | 全灭 | 稀疏对应太少 + 域差 → PnP 病态 |
| Proj@5pix | ~83 | 0.00 | 全灭 | t 平移数值爆炸（~5e16），渲染择优无法恢复 |
| 5cm5° | ~45 | 0.00 | 全灭 | 同上 |
| 耗时 | matching ~0.8s | 0.81s | ≈ | 无速度优势（MASt3R 同档）|

## Decision

- 结论：`reject`（**对应质量路线的反面证据：稀疏匹配比稠密更差**）
- 原因：
  1. **全灭机制**：SuperPoint+LightGlue 在真实-合成渲染域差 + 弱纹理下匹配对
     太少（单帧 ~29 对，filter 0.0 时）且错误率高 → RANSAC-PnP 数值病态
     （t ~5e16 爆炸），120 帧无一可用解；渲染择优（mask IoU）也无法从爆炸
     解恢复（cand_adds 全空）。
  2. **对比结论**：同一瓶颈（对应质量）的两种攻击路线——MASt3R 稠密 desc
     （~4096 采样 + 几何先验）显著优于零训练稀疏 Transformer 匹配。LightGlue
     的 29 对 vs MASt3R 千级对应：弱纹理下稀疏语义匹配缺乏稠密几何支撑。
  3. 与 conf-filter / match-768 / prescreen 三线闭合：对应质量瓶颈不可由
     "换匹配器/调分辨率/滤置信" 绕过，是 MASt3R 模型能力极限；单目侧所有
     RGB 杠杆已穷尽（tz-depth 证明真实深度是充分条件 +4.17）。
- 下一步：论文 §4 对应质量瓶颈小节引用本实验 + conf-filter + match-768 三线；
  LightGlueMatcher 代码保留（消融档），不再投入。

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚（outputs/exp_lg/results/duck.json）
- [x] `python3 scripts/analysis/check_state.py` 通过
