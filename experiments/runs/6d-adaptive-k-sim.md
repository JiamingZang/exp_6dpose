# 6d-adaptive-k-sim —— 自适应 K 早停离线仿真 + 在线实施

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-adaptive-k-sim` |
| Owner | agent |
| Status | `running` |
| Started | `2026-08-16 03:55` |
| Finished | empty |
| Queue row | `experiments/QUEUE.md::6d-adaptive-k-sim` |

## Question

K 曲线饱和（K≥20 后 +1.0~+5.67）：逐帧"内点 plateau 早停"（连续 w 个解码模板增益 ≤ δ 即停，min_k 兜底）能把平均解码数压到多少、ADD 损失多少？是否存在"解码更少且精度更高"的 Pareto 点？

## Protocol

| 项 | 值 |
|---|---|
| Config | `configs/experiments/dense80_topk_instr.yaml`（采集）+ `scripts/analysis/simulate_adaptive_k.py`（仿真）+ `configs/experiments/dense80_es.yaml` / `dense80_es_ia.yaml`（在线验证） |
| Code change | `2c2d81e`（在线早停实现：plateau_step + matcher 早停模式 + pipeline _pnp_one 抽取）+ `53179c8`（验证配置） |
| Data split | 120 帧 × 5 弱物体（duck ape cat holepuncher phone） |
| Metrics | 粗位姿 ADD(S)@0.1d（仿真基线=K=40 inlier-best，与主表口径一致的逐帧重算）；在线验证含级联 |
| Baseline | K=40 粗位姿 49.33（5 弱物体均值）；champion 级联 61.20 |
| Success line | 存在规则：mean K ≤ 20 且 ADD ≥ K=40 基线 -1.0 → 实施在线早停 |

## Commands

```bash
# 采集（cand_* 落盘，topk_instr 档；recovery 链自动跑，03:55 起）
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_topk_instr.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_adaptive_k/cache --out outputs/exp_adaptive_k/result.json

# 逐物体早停仿真（纯 CPU，recovery 链自动跑）
python3 scripts/analysis/simulate_adaptive_k.py --cache outputs/exp_adaptive_k/cache/<obj>.jsonl \
    --object <obj> --w 2,3,5 --delta 0,50,200 --ratio 0.02,0.05,0.10 --min-k 5,8,12

# 在线验证（es_verify_watcher.sh，等主链退出后自动跑）
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_es.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_es/cache --out outputs/exp_es/result.json
python3 scripts/eval/run_linemod.py --config configs/experiments/dense80_es_ia.yaml \
    --objects duck ape cat holepuncher phone --max-frames 120 \
    --cache-dir outputs/exp_es_ia/cache --out outputs/exp_es_ia/result.json
```

## Live Log

- `08-16 03:55`：recovery 链启动采集（topk_instr，cand_* 字段确认落盘：cand_Rs/ts/inliers/scores/adds/order/templates/projs/reproj/ncorr）。
- `08-16 04:20`：duck 33 帧时仿真冒烟通过（缓存格式兼容，早停规则扫描可用）。
- `08-16 05:00`：duck 采满（120 帧），离线分析突破：
  - 池内有货 78/120（65%），inlier-best 只命中 33/120（27.5%），**选择失败 45 帧（37.5%）**；
  - 选择失败帧里正确候选内点**从不更高**（Δinlier 中位 -93，0% 帧更高）——自洽错候选内点虚高；
  - 但正确候选在解码顺序上**更靠前**（中位早 5 位，60% 帧）——早停可把后期坏候选排除。
  - **前缀重放复现 K 曲线 dip 机制**：K=1 17.5 → K=5 31.67（峰）→ K=10/15 26.67（谷）→ K=40 27.50——与官方 duck K 曲线 dip（K=20 19.17）同形。K 曲线非单调 = 后期自洽错候选被 inlier-best 选中的系统性偏差。
- `08-16 05:05`：duck 全量仿真：**baseline 27.50 → 早停最优 30.83（Δ+3.33）@ meanK 2.9**（规则 w=5/δ=50/min_k=5）；多个规则 Δ+2.5 且 meanK 2.2-2.6。
- `08-16 05:40`：ape 采满，全量仿真：**baseline 24.17 → 31.67（Δ+7.50）@ meanK 2.5**（w=2/δ=200 或 rel0.05/min_k=8）。
- `08-16 05:45`：排名分布：duck 有货帧正确候选 top-3 47%/top-5 65%；ape top-3 62%/top-5 70%——机制上限与早停增益一致。
- `08-16 06:00`：在线早停实现完成并提交（`2c2d81e`）：plateau_step（绝对/相对双阈值）+ matcher 逐模板解码早停模式（独立 NN，与融合互斥）+ pipeline `_pnp_one` 抽取共用 + 配置项（early_stop/w/delta/ratio/min_k）+ 8 个单测（214 全绿）。验证配置 `dense80_es.yaml`（粗位姿档）/`dense80_es_ia.yaml`（champion 级联档）已建（`53179c8` 已 push）。
- `08-16 06:05`：es_verify_watcher.sh（PID 667072）挂起，等主链退出后自动跑在线验证两档。
- `08-16 06:30`：**在线配置 Pareto 确认**（w=2/ratio=0.05/min_k=8）：duck +2.50 @ meanK 2.2、ape +7.50 @ meanK 2.5（min_k=8 同时保障联合 PnP 池 ≥8 模板）。
- `08-16 06:35`：**解码数分布**（在线规则重放）：duck meanK 2.2 p50=1 p90=6 max=9；ape meanK 2.5 p90=6 max=9——82%/72% 帧解码 ≤3，**0% 帧解码 >20（无长尾）**；粗位姿延迟估计 6.1s → ~0.7s（≈9×）。
- `08-16 07:00`：cat 采满，仿真：**baseline 42.50 → 52.50（Δ+10.00）@ meanK 1.6**（w=3/rel0.1/min_k=5）；在线配置档（rel0.05/min_k=8）+8.33——**三物体一致（duck +3.33 / ape +7.50 / cat +10.00）**。
- `08-16 07:20`：hp 采满，仿真：**baseline 34.17 → 35.83（Δ+1.67）@ meanK 3.6-5.1**——**四物体全正**（duck/ape/cat/hp），hp 增益最小（弱纹理最弱物体早停窗口窄）；phone 采集中。
- `08-16 07:40`：phone 采满（06:48 全采集完成），仿真：**baseline 51.67 → 54.17（Δ+2.50）@ meanK 3.5**——**5/5 全正，MEAN Δ+5.00 @ meanK ~2.7**（K=40 → ~2.7 = 解码降 94%）；成功线（meanK≤20 且 ADD≥基线-1.0）远超。链进入 fib24（onboard 120t + 评测 ~3h）。
- `08-16 07:10`：**语义澄清（hp 悖论）**：固定前缀重放（原始解码序，含 PnP 失败候选占位）hp 单调上升 14→34、duck 复现 dip——这与官方 K 曲线一致；仿真/在线早停的择优在**有效候选过滤序**上进行（失败候选不占位），plateau 停止是其增益来源，不是任何固定 K。两种语义各自自洽：官方 K 曲线=原始序固定 K；早停=过滤序自适应停止。rank-top1（27.0 均值）< inlier-best（36.0）< 早停（41.0）——排名做排除、内点做前缀内选择，组合才有效。
- `08-16 08:00`：**duck 在线粗位姿确认**：ADD(S)@0.1d **34.17 vs 基线 30.83（+3.34）**——仿真预测 +3.33 精确命中；Proj 77.50 vs 81.67（-4.17）。逐帧：ADD 救 25 丢 21（净 +4）、Proj 救 10 丢 16（净 -6）——46/120 帧选择结果改变，非微扰。解码 mean=8.1（min_k=8 地板主导）max=10；matching 2.24s vs 4.38s。**设计修正**：es 实验混杂了独立 NN vs 融合-12 匹配——补对照档 dense80_es_nostop.yaml（fusion off + K=40），分离早停效应与融合效应（control watcher 在 localt_off 后自动跑）。
- `08-16 08:20`：**ape 在线粗位姿 40.83 vs 官方 44.17（-3.34）**——与 duck 相反！ape 的粗位姿高度依赖联合 PnP（sim 无联合 24.17 → 官方含联合 44.17，+20），早停把联合池从 12 收窄到 8 且匹配改独立 NN，净亏。归因待对照档（es_nostop：独立 NN + K=40）判定：若 es_nostop≈40.83 则亏在独立 NN；若≈44.17 则亏在早停排除本身。
- `08-16 09:00`：**在线粗位姿全物体盘点（官方基线=采集缓存同配置）**：duck 30.83→34.17（+3.33）/ ape 46.67→40.83（-5.83）/ cat 53.33→49.17（-4.17）/ hp 50.83→35.00（-15.83）/ phone 65.00→60.83（-4.17）——**粗位姿口径判负（5 物体均值 44.00 vs 49.33，-5.33）**，仅 duck 净正。根因：粗位姿精度依赖融合-12 匹配 + 联合 PnP 池（hp 官方 K 曲线单调上升最依赖），早停同时收窄两者；仿真（单候选无联合）口径掩盖了这层依赖。**决策点移到 es_ia 级联档**（09:11 起跑，ETA ~12:00）：iter_align 重匹配能否吸收粗位姿差异。v2 候选方案：早停判定用独立 NN（省解码），最终匹配对解码前缀重跑融合（保质量）。
- `08-16 11:00`：**es_ia 中期（3/5）**：duck 45.00（-2.50）/ ape 60.00（+0.83）/ cat 74.17（**+10.00**）——级联层大幅吸收粗位姿差异并放大选择收益（cat 粗位姿 -4.17 → 级联 +10.00；逐帧救 21 丢 9）。**方法论结论：择优机制消融应以级联层为决策口径**（粗位姿口径双重误导：无联合 PnP 高估 + 无级联漏 basin 效应）。hp 89/120、phone 排队。
- `08-16 11:36`：**es_ia 终判：60.17 vs 61.00（-0.84，噪声带内）**——duck -2.50 / ape +0.83 / cat +10.00 / hp -10.00 / phone -2.50。级联吸收大部分粗位姿损失（粗位姿曾 -5.33），但 hp 重灾（-10.00，K 曲线单调上升型最依赖全池）。采纳判定：**±1.0 带内判平**，v2（前缀融合）若能救 hp 则整体转正；对照档（es_nostop）先判定 hp 损失归因（独立 NN vs 排除本身）。
- `08-16 13:00`：链内 fib24 中期（duck/ape 满）：**120t vs 80t：duck +9.17 / ape -10.84**——模板密度同样非单调有利（与 K/J 曲线非单调同构）；终判等 5 物体（见 runs/6d-fib24.md）。
- `08-16 14:00`：**fib24 3/5（duck +9.17 / ape -10.84 / cat +5.84）**——跨实验模式成型：**候选池侧每个旋钮（K/J/模板密度/早停）都呈同一物体异质性**。K 曲线 K=1→40 上升幅度：hp +22（最依赖池）> cat +16 > duck +13 > phone +8 > ape +2；早停损失排序（hp -10 > duck -2.5 > phone -2.5 > ape +0.8 > cat +10）与 K 曲线依赖度高度一致（hp 例外 ape：ape 池内无货但依赖联合 PnP，fib24 的 24 视角采样本身对 ape 不利）。**通用改进的收敛方向 = 排名感知的池预算分配**：按帧/物体的正确候选排名分布自适应分配解码预算（早停即其一，hp 类需更高 min_k）。
- `08-16 14:30`：**fib24 4/5（duck +9.17 / ape -10.84 / cat +5.84 / hp -20.83）**——120t 对 hp 灾难（phone 部分帧 37.5 不可信待满）。机制：24 视角×5 的 120t 库经 DINOv2 预筛 top-40 后，自相似物体（hp 规则格栅）的相似模板互相挤占名额，正确模板被挤出池；duck 类外观分化物体受益于密度。**80t 饱和 verdict 倾向成立**（成功线 ≤+1 证实 80t 饱和；4 物体均值 -4.17）。
- `08-16 16:20`：**前缀内选择键离线扫描（simulate_select_key.py，纯 CPU）**：v1/es_ia 前缀内择优固定 inlier-best，但已证实正确候选内点"从不更高"——对同缓存重放早停轨迹，在停止前缀内换择优键比较 ADD。**结论：weighted（inlier×score）与 sim 键在前缀内一致 ≥ inlier，5/5 物体无一为负**；abs50 w3 mk5 规则下 MEAN +3.33（duck +2.50 / ape +5.00 / cat +3.33 / hp +2.50 / phone +3.33），在线规则（rel0.05 w2 mk8）MEAN +1.33（cat/phone 各 +2.5，duck/hp +0.8，ape 0）。**与官方全路径消融对照关键**：K=40 全解码时选择键中性（ablation_selection.json：inlier 49.33 / sim 49.50 / weighted 49.50）——键的收益只在**短前缀**出现（前缀短时 inlier-best 更易被"自洽错"高内点候选劫持，score 加权起抗劫持作用）。⇒ **v2 配置 dense80_es_fusion.yaml 加 `solver.selection: weighted`**（一行，K=40 下键中性 → 不混淆 es_nostop 对照）；rank/键×规则交互留给 v2 后视结果再定。
- `08-16 16:45`：**两个补充探针（同缓存，纯 CPU）**：(1) **弱前缀 floor 门**（最优内点 < floor 禁止停，通用单参数）——hp 全 floor 档 ADD 平 33.33 无变化（plateau 停帧的最优内点已 >3500，floor 不绑定；oracle 39.17 说明 hp 收益在联合 PnP 不在继续解码）→ **floor 门判死**，hp 早停损失的归因继续指向联合 PnP 池（es_nostop 待判）。(2) **固定前缀长 × 键扫描**（k=4/8/12/40 × inl/wtd/sim，无联合口径）：**weighted 在全部 4 个前缀长 × 5 物体上一致 ≥ inlier**（k=8：duck +5/ape +7/cat +12/hp +1/phone +7 帧；k=40 无联合口径 MEAN +8.0）——与官方含联合 PnP 口径的键中性（49.33/49.50）合读：**联合 PnP 修复了选择信号，键差异是"无联合"口径伪差；前缀短时联合池受限，weighted 是保守安全选择**（两口径下均不劣于 inlier）。
- `08-16 17:15`：**停表信号对比（同缓存，纯 CPU）——score-plateau 停表胜出**。inlier 停表（在线 es 规则）+ weighted 择优 = MEAN 39.3 @ meanK 2-3；换 **score 停表**（MASt3R 分数跌出峰值 ×95% 才计停滞，w=2/min_k=8）+ weighted 择优 = **MEAN 42.3（+3.0）@ meanK 5-8**：duck +4.2 / ape -2.5 / cat +5.0 / hp +2.5 / phone +5.8——4/5 物体正，唯一负的是 ape（池没货，深排候选分数也不高）；AND 组合（双信号都停滞才停）MEAN 42.0（ape 只 -0.9，更稳）。**机制**：inlier 被"自洽地错"高内点候选虚高（停表信号本身被污染），MASt3R score 正确反映候选质量；score 停表还省掉决策阶段的逐模板 PnP。**v2.1 设计（v2 判决后实施）**：停表信号换 score（m.score 现成，决策阶段免 PnP）+ 前缀融合 + weighted 择优；配置 dense80_es_fusion_score.yaml 待登记。**纪律**：v2 链在跑，停表信号改动不动共享 es 代码，等 v2 出数后再实现（避免 06:50 式回归烧链）。
- `08-16 06:48`：**链事故 1（fib24 onboard 命令错）**：recovery 链 fib24 段调用 `scripts/data/onboard_object.py`——该路径不存在（AGENTS.md 主链命令过时；真实入口是 `src.pipeline.onboard_object` 函数），5 物体 onboard 全失败 + 评测缺库 → rc=1 链中止，localt_off 未跑。修复：AGENTS.md 主链命令更正；续链 /tmp/post_recovery2.sh（正确 python -c 调用 onboard_object）等 es 验证后补跑 fib24 + localt_off。
- `08-16 06:50`：**链事故 2（es_cb 闭包未绑定）**：es 粗位姿档首帧崩溃 `NameError: free variable 'sx'`——es_cb 闭包引用 `sx, sy`，但二者由 matcher.match 返回才绑定，回调在返回前被调用。修复 `661c5be`：matcher 回调签名改为 `cb(m, sx, sy)`（内部尺度直接传入）。pytest 214 全绿；重挂 watcher + 续链。帧 1 实测：decoded=8（min_k 地板）、matching 1.46s vs 40 解码 4.38s。

## Result

### 离线仿真（单候选 inlier-best 口径，无联合 PnP；官方 K=40 粗位姿含 J=12 联合，数字口径不同，Δ 是仿真内相对值）

| 物体 | K=40 基线 | 早停最优 ADD | ΔADD | meanK | 规则 |
|---|---|---:|---:|---:|---|
| duck | 27.50 | 30.83 | **+3.33** | 2.9 | w=5, δ=50, min_k=5 |
| ape | 24.17 | 31.67 | **+7.50** | 2.5 | w=2, rel0.05, min_k=8 |
| cat | 42.50 | 52.50 | **+10.00** | 1.6 | w=3, rel0.1, min_k=5 |
| holepuncher | 34.17 | 35.83 | **+1.67** | 3.6 | w=5, δ=50, min_k=5 |
| **MEAN** | 36.00 | 41.00 | **+5.00** | ~2.7 | 5/5 全正 |
| phone | 51.67 | 54.17 | **+2.50** | 3.5 | w=5, rel0.02, min_k=8 |

### 在线验证（待跑）

| 档位 | 基线 | this run | delta | note |
|---|---:|---:|---:|---|
| 粗位姿 K=40 | 49.33 | 44.00 | **-5.33** | 在线口径：duck +3.33 / ape -5.83 / cat -4.17 / hp -15.83 / phone -4.17；仿真 +5.00 被联合 PnP+融合依赖抵消 |
| champion 级联 | 61.00 | 60.17 | **-0.84** | 噪声带内：duck -2.50 / ape +0.83 / cat +10.00 / hp -10.00 / phone -2.50；hp 归因待对照档 |

## Decision

- 结论：`running`
- 原因：2/5 物体仿真确认机制（早期正确候选 vs 后期自洽错高内点候选），在线实施已提交；等全物体仿真 Pareto + 在线验证数字。
- 下一步：采集完 → 全物体仿真（链自动）→ 在线验证两档（watcher 自动）→ 数字入论文 §5.4 + 四件套收尾。

## Sync Checklist

- [ ] `experiments/QUEUE.md` 状态已更新（running）
- [ ] `docs/STATE.md` 冠军/在跑/下一步已更新
- [ ] `docs/LEDGER.md` 已新增或更新一行
- [ ] 结果文件路径写清楚
- [ ] `python3 scripts/analysis/check_state.py` 通过
