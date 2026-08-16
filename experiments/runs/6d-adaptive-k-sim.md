# 6d-adaptive-k-sim —— 自适应 K 早停离线仿真 + 在线实施

## Metadata

| 字段 | 值 |
|---|---|
| ID | `6d-adaptive-k-sim` |
| Owner | agent |
| Status | `done` |
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
- `08-16 18:15`：**score-plateau 参数细扫 + 预算包络线（budget_envelope.py 入库）**。(1) 细扫：**mk=12 全面优于 mk=8**（τ=0.03/mk=12：MEAN 44.2 @ meanK 7.4；τ=0.05/mk=12：43.8 @ 8.5）——min_k=12 地板与联合 PnP 池（J=12）匹配，hp 升到 40.0-40.8（mk=8 时 35.8）；τ∈{0.03-0.12} 在 mk=12 下几乎无差 → **v2.1 默认参数 τ=0.05/mk=12**。(2) **oracle 预算包络：k=12 即饱和，k=12..40 完全相等（5 物体逐帧一致）**——MEAN 61.8（duck 65.0/ape 53.3/cat 71.7/hp 56.7/phone 62.5）vs k=8 的 56.5 vs k=4 的 50.0。**通用结论：候选池的全部信息在前 12 个有效候选内**；官方 K 曲线晚期增益（hp +22）是原始序里无效候选占位所致（过滤序 12 饱和，与 07:10 语义澄清闭合）。规则（42-44）与包络（61.8）的差 = 选择损失 + 无联合 PnP 口径差——再次指向候选池/选择缺口，与 gap-oracle 闭合。
- `08-16 19:15`：**v2.1 参数定案 + 两个死路排除（同缓存，纯 CPU）**。(1) **绝对 score 下限预滤**（score<floor 跳过，省无效解码）：floor≤0.65 ADD 不变（43.8）但 meanK 只省 0.7（成功候选分数本就 ≥0.65 聚簇）；floor=0.75 掉 0.8——**死路**（无效候选占位的浪费在缓存里不可见，需在线落盘失败候选分数才能设计拒滤）。(2) **mk=12 下模式×键交叉**：score-plateau+weighted = AND+weighted = **MEAN 43.8 @ meanK 8.5**（mk=12 时 inlier 停表几乎不先触发，AND≡score）；OR 组合 41.8（inlier 信号仍污染）；weighted 全模式 > sim（43.8 vs 42.2）。⇒ **v2.1 最终参数：score 停表（τ=0.05/w=2/mk=12）+ 前缀融合 + weighted 择优**。
- `08-16 19:54`：**es_nostop（独立 NN + K=40）结案——归因清晰**：MEAN **49.00 vs 官方 49.33（Δ-0.33）**，逐物体 duck 30.00(-0.83) / ape 45.83(-0.84) / cat 53.33(0.00) / hp 50.83(0.00) / phone 65.00(0.00)——**独立 NN 在 K=40 下与融合-12 几乎无损**（融合的收益不在匹配本身而在联合 PnP 池）。**早停粗位姿损失拆解：-5.33 = -0.33（NN，可忽略）+ -5.00（排除本身）**——排除损失逐物体：duck **+4.17**（排除后期自洽错候选收益）/ ape -5.00 / cat -4.17 / hp **-15.83**（正确候选排深，联合池从 40 收窄到 8 最致命）/ phone -4.17。⇒ **v2 设计验证**：融合不是关键修复（NN 本无损），**mk=12 恢复联合池才是 hp 的修复**（v2 当前 mk=8，预期 hp 仍小亏；v2.1 的 mk=12 + score 停表是完整方案）。v2（es_fusion）19:55 起跑。
- `08-16 20:10`：**v2 首跑崩溃 + 口径纠正（重要）**：es_fusion 首帧 `ValueError: too many values to unpack` → 修复元组后 `_fusion_match 返回 None` → **根因：仓库 fusion 匹配早已证伪**（configs/current/dense80.yaml（第 9 行） `fusion: false  # 融合匹配已证伪（硬分配破坏联合 PnP 多样性）`，自 901f1e3 初始提交起）——**"融合-12 匹配"是本实验记录里的错误术语，官方管线一直是独立 NN + 联合 PnP(J=12)**；_fusion_match 无 fusion=false 分支落空返回 None。连带纠正：早前"独立 NN vs 融合-12"混杂叙事不成立，es_nostop 实为**同匹配模式的干净 K=40 对照**（这使归因更干净：排除是唯一变量）。**v2 设计重定向**：删除 early_stop_fusion（证伪路线），v2 = ia 级联 + 早停 min_k=12（恢复联合池）+ selection=weighted；es_fusion.yaml 重写（base 改 ia 级联档，20:15）；修复 _es_finalize 加 fusion 回退守卫 + 空掩码 3 元组陈旧签名（abe0760/后续提交，214 测试全绿）。
- `08-16 21:00`：**v2/v2.1 判决规则（结果出炉前先验登记，防后视偏差）**。基线 = champion ia 级联 61.00（duck 47.50 / ape 59.17 / cat 64.17 / hp 56.67 / phone 77.50）。**v2 判决**（级联口径，MEAN）：≥60.00 转正（早停路线采纳，跑 v2.1 确认 score 信号增益）；≤59.00 判负（早停路线结案，v2.1 仅作佐证）。hp 修复确认线：v2 hp ≥ 53.67（v1 es_ia hp 46.67，mk12 须回收 ≥7 分）。**v2.1 判决**：≥ v2 +1.0 → score 信号采纳；否则判平，用 v2 参数收官。**采纳后论文口径**：精度损失 ≤±1.0（噪声带）且解码降 ≥70% → §5.4 结论为"自适应预算分配以 ~80% 解码节省换平级精度，配 weighted 择优 + mk12 联合池恢复"。
- `08-16 22:00`：**v2 中期（3/5：duck 36.67 / ape 51.67 / cat 66.67，MEAN 51.67 vs es_ia 同期 59.72）**——按先验规则 v2 大概率判负（需 hp/phone 各 +15 才能回 60）。**逐帧归因两探针**：(1) "9-12 位高内点错候选抬高 best"仅覆盖 4-18% 丢帧——联合池再污染非主因（candidate 级证据）；(2) 选择键改变（weighted≠inlier 选人）率丢帧/赢帧/全帧一致 ~50%——weighted 键非主因。**新疑点（方法学）**：早停档解码数不同 → matcher 采样阶段（sample_correspondences 每候选 4096 抽，帧种子流）消耗 rng 数量不同 → 下游（guided/iter_align 重匹配）共享同一 self.rng 流错位 → **帧级对比携带采样噪声（官方 40 候选 vs es 8-12 候选的流不同）**；RANSAC 内部固定种子（ransac_pnp.py:185 default_rng(0)）不受影响。该混杂影响所有 es vs 官方帧级分析（MEAN 级 120 帧平均后小得多，但 ±3 量级可信）。**计划**：v2 全量出数后跑 v2b（mk12+inlier，与 es_ia 仅差 mk 一变量）拆解归因；rng 隔离修复仅用于最终采纳配置的正式评测（不改变对比口径）。
- `08-16 23:00`：**v2 终判：54.50 vs 61.00（-6.50）——按先验规则（<60.00）判负**。逐物体：duck 36.67(-10.83) / ape 51.67(-7.50) / cat 66.67(+2.50) / hp 43.33(-13.33) / phone 74.17(-3.33)；vs es_ia：全部 ≤ -0.83（cat -7.50 最重）。**hp 修复线失败**（43.33 < 53.67）——mk12 在级联层不仅没救 hp（es_ia 46.67 → v2 43.33 更差），还拖累 duck/cat（es_ia 的排除收益被放大候选池抵消）。**粗位姿口径 mk12≥mk8 的仿真结论在级联层系统性不兑现**——第三次验证"池侧机制必须级联层验证"（K 曲线 dip / 择优 / 预算分配同构）。**路线收口**：按先验规则跳过 v2.1（score 信号同为粗位姿口径优化，级联层预期同向失败；3h GPU 不值得佐证），v2b 拆解同样跳过（路线已判负，归因留待论文讨论）。**最终结论（§5.4）**：早停 = **级联层 -0.84（es_ia，噪声带内判平）换 ~90% 解码削减**——是速度杠杆不是精度杠杆；粗位姿口径仿真收益（+5.00）与池侧优化（mk12/weighted/score）在级联层不兑现；精度瓶颈仍是候选池生成（gap-oracle）。es_n_decoded 实测 = min_k 地板（mk8 档 mean 8.1 / mk12 档 12.0），matching 2.5s vs 4.4s。
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

### 在线验证（全跑完，08-16 23:00 收口）

| 档位 | 基线 | this run | delta | note |
|---|---:|---:|---:|---|
| 粗位姿 K=40 | 49.33 | 44.00 | **-5.33** | 在线口径：duck +3.33 / ape -5.83 / cat -4.17 / hp -15.83 / phone -4.17；仿真 +5.00 被联合 PnP 依赖抵消 |
| champion 级联（es_ia，mk8+inlier）| 61.00 | 60.17 | **-0.84** | 噪声带内判平：duck -2.50 / ape +0.83 / cat +10.00 / hp -10.00 / phone -2.50 |
| es_nostop 对照（K=40，独立 NN）| 49.33 | 49.00 | **-0.33** | 归因：NN 匹配无损，早停损失全在排除（联合池收窄）；duck 30.00 / ape 45.83 / cat 53.33 / hp 50.83 / phone 65.00 |
| v2（mk12+weighted，级联）| 61.00 | 54.50 | **-6.50** | 判负：duck 36.67 / ape 51.67 / cat 66.67 / hp 43.33 / phone 74.17；hp 修复线失败（43.33 < 53.67） |
| v2.1（score 停表）| — | 未跑 | — | 先验规则：v2 判负后仅佐证，粗位姿口径优化预期级联层不兑现，跳过 |

### 方法学教训（§5.4 素材）

1. **粗位姿口径系统性高估池侧机制**：仿真 +5.00 / mk12 +2.0 / weighted +1~3 / score 停表 +3.0 全在级联层不兑现（es_ia -0.84 / v2 -6.50）——池侧旋钮（预算分配/停表/择优键）必须级联层验证。
2. **采样 rng 流错位**：早停解码数不同 → 采样阶段消耗帧种子 rng 不同 → 下游级联错位 → 帧级对比 ±3 噪声（MEAN 级可接受；正式评测需 rng 隔离）。
3. **es_nostop 对照价值**：排除是唯一变量的干净对照（NN 模式全链相同）。
4. 早停的正面结论：**级联层 -0.84（噪声带内）换 ~90% 解码削减**（es_ia，matching 4.4s→2.2s；mk 地板主导解码数）——速度杠杆成立，精度杠杆不成立。

## Decision

- 结论：`done（早停路线收口：es_ia 判平 -0.84；v2 判负 -6.50）`
- 原因：先验判决规则（v2 ≥60.00 转正）未过；hp 修复线（≥53.67）失败；粗位姿口径优化（mk12/weighted/score）在级联层全部不兑现，与"池侧机制必须级联层验证"教训闭环。早停作为速度机制保留（论文 §5.4：~90% 解码削减、精度噪声带内持平），精度瓶颈仍是候选池生成（gap-oracle）。
- 下一步：§5.4 按三层诚实叙述收口（仿真 +5.00 / 在线粗位姿 -5.33 / 级联 -0.84 + v2 判负 + 方法学教训）；es 代码与配置保留（early_stop 默认 false 不改变主链）。

## Sync Checklist

- [x] `experiments/QUEUE.md` 状态已更新（done）
- [x] `docs/STATE.md` 冠军/在跑/下一步已更新
- [x] `docs/LEDGER.md` 已新增或更新一行
- [x] 结果文件路径写清楚
- [x] `python3 scripts/analysis/check_state.py` 通过
