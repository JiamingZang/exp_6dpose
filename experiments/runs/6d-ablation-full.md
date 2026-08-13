# 6d-ablation-full —— 论文 §3.3 十组消融（120 帧子集口径）

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-ablation-full` |
| Owner | `qoder` |
| Status | `running` |
| Started | `2026-08-12 16:10` |
| Finished |  |
| Queue row | `experiments/QUEUE.md::6d-ablation-full` |

## Question

这次只回答一个问题：

> 论文 §3.3 十组消融（topk/n_templates/matcher/localization/geometry/
> scale_align/selection/ransac_eps/segmenter/renderer）在 **120 帧子集口径**
> （用户 08-12 确认，磁盘 8.1G 不足以全量 14968 帧缓存）下跑齐，支撑
> 模板库构建 + dc2 方法贡献的消融证据。

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/ablations/`（10 组 yaml），base = `configs/current/dense80_depthc_guided.yaml`（粗位姿，论文方法贡献口径）|
| Code change | 无（run_ablation.py 已有）|
| Data split | **120 帧子集 × 5 弱物体**（duck/ape/cat/holepuncher/phone，消融最有区分度）；全量口径待磁盘清理后补 |
| Metrics | ADD(S)@0.1d / Proj@5px / 5cm5°（run_ablation.py 自动出表）|
| Baseline | 粗位姿 120 帧口径（dense80_depthc_guided.yaml）|
| Success line | 10 组数字落 `outputs/ablation_<name>.json`，入论文表 |

## 组别状态

| 组 | sweep | reonboard | 状态 |
|---|---|---|---|
| 01 topk | K=1/5/10/20/40 | 否 | 第一批 |
| 03 matcher | mast3r/dinov2_patch/loftr | 否 | 第一批（loftr 抛 NotImplementedError 跳过）|
| 04 localization | fastsam/gt_bbox | 否 | 第一批 |
| 06 scale_align | true/false | 否 | 第一批（CAD+RGB-PnP 下数学恒等，诚实性检查）|
| 07 selection | inlier/similarity/weighted | 否 | 第一批 |
| 09 ransac_eps | ε=3/5/8/10 | 否 | 第一批 |
| 10 segmenter | fastsam/sam/gt_mask | 否 | 第一批 |
| 02 n_templates | 8/24/40/80 | 是（8/24 需 onboard，40/80 已有）| 第二批 |
| 08 renderer | 3dgs/pyrender_cad | 是 | 第二批 |
| 05 geometry | cad/vggt | 是 | **跳过**（VGGT 未装，onboard 会崩）|

## Commands

```bash
source env.sh
# 第一批（无 reonboard）
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/01_topk.yaml --objects duck ape cat holepuncher phone --max-frames 120
# ... 03/04/06/07/09/10 同式
# 第二批（需 onboard）
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/02_n_templates.yaml --objects duck ape cat holepuncher phone --max-frames 120
python scripts/eval/run_ablation.py --config configs/current/dense80_depthc_guided.yaml \
    --ablation configs/ablations/08_renderer.yaml --objects duck ape cat holepuncher phone --max-frames 120
```

## Live Log

- `08-12 16:10`：登记入队（running），子集口径（用户确认）。模板库覆盖检查：
  40t/80t 全 13 物体已有；8t/24t 缺（02 组需 onboard cube8 采样）；05 组
  VGGT 未装跳过（run_ablation 不 catch ImportError，会崩）。
- `08-13 00:30`：01 组首跑（default.yaml base）暴露配置错误：**default.yaml 是
  40t 模板库（cube8×5）+ guided_refine:false，与主表配置（80t fibonacci +
  guided）口径不一致**，数字不可比主表；且 ape 40t 库缺 .pt（refine 需要）
  直接崩。修正：消融 base 一律用 `dense80_depthc_guided.yaml`。
- `08-13 01:00`：**缓存复用地图**（guided base）：K=1/5/10 档命中 8/12
  b2g0avtuk 遗留缓存（5 物体 121 帧全）；K=20 档 ape/duck/cat(119) 命中、
  holepuncher/phone 需补跑（b2g0avtuk 被 kill 时未跑）；K=40 档命中主表
  cache13_dc2（49138670）。09 组 5.0 档修正为 float（int 5 vs float 5.0
  hash 漂移 → 不命中主表，改为 5.0 后命中）。
- `08-13 01:05`：onboard 补齐脚本（/tmp/ablation_onboard_fill.py）入批处理：
  8t×4 + 24t×5 + 40t ape 补 pt 共 11 次（ape 40t npz 已备份 .40tprebak）；
  批处理链：01 → onboard fill → 02 → 09 → 07 → 08 → 03(dinov2patch) →
  tzdepth5（第四章 RGB-D 证据）。
- `08-13 02:13-02:32`：**GPU 竞争事故**——历史遗留的第二个批处理进程
  （00:55 起等 01_topk）与 00:50 的首个进程同台，02/09/07 三组全部 OOM
  （torch.OutOfMemoryError ×5），02 组 onboard 的 8t 模板已落盘但评测全废；
  02:32 起干净重跑（v3 脚本，当前 PID 469262）。
- `08-13 05:28-16:32`：02 组重跑成功（rc=0）：8t/24t/40t 全量 onboard +
  评测，结果见 Result 表。
- `08-13 16:32-`：09 组进行中（ε=3 全新匹配 3.6h 完成；ε=5.0 命中主表
  缓存 49138670 秒过逐物体重写；ε=8/10 全新匹配各 ~3.6h）。之后自动续
  07 → 08（osmesa）→ 03（dinov2patch）→ tzdepth5。
  监控信号：缓存文件 mtime（`ls -lt outputs/ablation_cache/`），日志因
  python 块缓冲滞后数小时，不可作实时信号。
- `08-13 22:10`：**07 耗时修正**——帧缓存的 cfg_hash 覆盖整个配置，selection
  非默认值（similarity/weighted）各自新 hash → **全新匹配各 ~3.6h**（不是
  缓存命中）。07 组预计 ~7.2h（inlier 档命中主缓存除外）。总链预计
  ~08-14 23:30 收尾（09 6h + 07 7.2h + 08 3.6h + 03 3.6h + tzdepth 2.7h）；
  inlier_ratio/reproj 策略由 6d-adaptive-k-sim 的 cand_* 缓存离线重排补齐
  （cand_ncorr/cand_reproj 已落盘，08-13），不再占用 GPU。
- `08-13 23:18`：**批处理重启（缓存 meta 修复 0299fbe）**——排查发现
  ε5.0 每物体 ~40min 是"假 cache hit"：缓存文件级 meta 比对含 matches_dir
  （主表运行是 matches13_dc2、消融是 None），_load_cache_records 整文件
  丢弃 → 主表 hash 档全部重匹配（ε5.0 已白烧 3.6h；07-inlier/08-3dgs 还
  会再各烧 3.6h）。修复：文件级 meta 只比对 cfg_hash（逐帧指纹兜底，
  206 测试通过），杀掉旧进程重启批处理——02 四档全部缓存命中（~15 min），
  07-inlier/08-3dgs 免烧 7.2h。新链：02 → 09（ε8/ε10 全新）→ 07 → 08 →
  03 → tzdepth，预计 ~08-14 20:00 收尾。
- `08-14 03:05`：**09 组 ε=3 预览（缓存聚合，非 run_ablation 正式输出）**：
  5 物体均值 ADD(S)@0.1d = **46.17**（duck 31.67 / ape 42.50 / cat 53.33 /
  holepuncher 43.33 / phone 60.00），vs 基线 ε=5.0（主表口径）49.17 =
  **-3.00**；Proj 持平（duck 82.50 vs 82.00）——更紧阈值减少对应数量、
  不伤投影精度但伤深度病态物体的 ADD。ε=8/10 待批处理出数（预计 06:10
  后）；正式数字以 ablation_ransac_eps.json 为准。

## Result

**01 topk 组已出（5 弱物体 × 120 帧均值，guided 粗位姿口径；K=1/5/10/20 来自
8/12 b2g0avtuk 遗留缓存，K=40 今天重跑）：**

| K | ADD(S)@0.1d | Proj@5px | 5cm5° |
|---|---:|---:|---:|
| 1 | 30.50 | 53.33 | 32.83 |
| 5 | 40.00 | 72.83 | 47.33 |
| 10 | 42.50 | 75.83 | 49.00 |
| 20 | 43.50 | 76.50 | 52.83 |
| 40 | 49.17 | 82.00 | 57.50 |

注意：duck 逐物体 K=20（20.0）< K=10（28.33）非单调——8/12 旧缓存存疑，
已排 duck 全档重跑验证（/tmp/duck_kcurve_verify.sh，v3 批处理后执行）。
其余物体 K=20≤K=40 单调性成立。K=1 档 8/12 缓存（default base 误跑 40t 库
faba5ca0）作废；765467ef（guided base）有效。

| 组 | baseline | this run | delta | note |
|---|---|---:|---:|---|
| 01 topk | K=40: 49.17 | K=1: 30.50 / K=5: 40.00 / K=10: 42.50 / K=20: 43.50 | K↑ 单调增益（duck 待验证）| K=40=主表口径；K 曲线入论文 5.4 |
| 03 matcher |  |  |  | dinov2_patch 档排队（v3 批处理）|
| 04 localization |  |  |  | 已有 6d-det-align 数字，跳过 |
| 06 scale_align |  |  |  | 默认档命中主表；false 档未排（价值低）|
| 07 selection |  |  |  | inlier 命中主表；similarity/weighted 排队 |
| 09 ransac_eps |  |  |  | 5.0 档命中主表；3/8/10 排队 |
| 10 segmenter |  |  |  | 已有 6d-loc-upper 数字，跳过 |
| 02 n_templates | 80t: 49.33 | 8t: 36.17 / 24t: 31.83 / 40t: 19.83 | **视角采样模式主导** | cube8 顶点采样系统性差（40t vs 80t 同为 5 旋转 -29.5）；cube8 下加旋转冗余有害（8t>24t>40t）；fibonacci 均匀覆盖是精度前提 |
| 08 renderer |  |  |  | pyrender_cad 需 OSMesa（已配）；排队 |
| 05 geometry |  |  |  | 跳过（VGGT 未装）|

## Decision

- 结论：`pending`
- 原因：
- 下一步：

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
