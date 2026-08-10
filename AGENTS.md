# AGENTS.md —— 实验 agent 恢复协议

新会话进来按顺序做，**不要跳步**：

1. **读状态**：`docs/STATE.md`（冠军数字、在跑、黑名单、已知坑）
2. **读账本**：`docs/LEDGER.md`（轮次状态 + 配置归属）；细节在
   `docs/EXPERIMENTS.md`（全部中间数字）与 `docs/RESEARCH_LOG.md`（时间线）
3. **读实验队列**：`experiments/QUEUE.md`；没有把任务状态改成 `running` 前，禁止开跑
4. **动手前冒烟**：`python -m pytest -q`（197 过 + 4 GPU 跳过，CPU 可跑；GPU 项自动跳）
5. **流程自检**：`python3 scripts/analysis/check_state.py`；失败先修流程，不要继续实验

## 主链命令（顺序不可换）

```
scripts/data/download_data.sh                        # LineMod/BOP 数据 ~6GB
python scripts/data/onboard_object.py --config <cfg> <obj>   # 3DGS 训练（背景按物体选！）
python scripts/data/rebuild_bank_fixed_views.py ...          # 固定视图重建模板库
python scripts/eval/run_linemod.py --config <cfg>            # 评测
python scripts/eval/summarize13.py                           # 汇总表
```

**背景色规则**（轮 6-8 血泪教训）：浅色物体黑背景（dense80_depth_bg0），
深色物体白背景（dense80_depth_w1，driller/cam）。**模板视图必须固定**
（重训后走 rebuild_bank_fixed_views，不重采样视图）。

## 实验纪律

- 开跑前必须完成三件事：`experiments/QUEUE.md` 状态改 `running`、复制
  `experiments/RUN_TEMPLATE.md` 到 `experiments/runs/<ID>.md`、写清楚命令和成功线
- 新配置 = 新 yaml + LEDGER 登记；禁止复用旧 yaml 偷改参数跑
- 数字必须带口径：**120 帧子集 vs 全量**、**含 refine vs 不含**、
  ADD(S) 对称物体（eggbox/glue）——混口径 = 结论作废
- 与旧代码对比只认 top1（49.49）；top3/5_best 是 GT oracle 上界不可比
- 模板库改动前备份 `outputs/templates/*.npz`（版本后缀纪律已有：.orig/.viewsbak）
- 收尾必须同一次操作更新四件套：`experiments/runs/<ID>.md`、
  `experiments/QUEUE.md`、`docs/STATE.md`、`docs/LEDGER.md`
- 收尾最后一条命令必须是 `python3 scripts/analysis/check_state.py`，失败就不算完成

## 代码一致性（2026-08-04 血泪教训后加）

通用规则见文末「构建期写代码纪律」托管段（先搜后写、同一物理量一处实现）。
本仓库的具体事故背景，作为那两条规则为什么是硬约束的证据：

- 事故：`template_renderer.py` 的 `depth_map` 路径曾独立手写了一套线性
  z alpha 混合，跟 `coord_map`/训练监督用的逆深度混合不是同一套数学，
  没跟着后者的偏差修复走（`render_invdepth` 统一后修复，见 git log）。
- 本仓库高危重复点提示：深度渲染/坐标提升/位姿精化——这三类概念仓库里
  都已有权威实现（如 `gs_trainer.GaussianTrainer.render_invdepth`），
  动手前先搜它们。

## 禁区

- 不重跑 STATE 黑名单路线（μ 混合锚点 / 单独 CAD 监督 / 重采样视图 / 统一背景色）
- 不引用已作废数字（旧表 67.63/80.06/65.13；ape 50.0 白背景误标条目）
- 不动 legacy_mypose.yaml 的口径（README §8 有专门警告段）
<!-- explab:section 构建期写代码纪律 -->
## 构建期写代码纪律

- **先搜后写**：写任何"仓库里概念上已经存在"的代码前，先搜现有实现，
  复用或在原处扩展；禁止照字面需求另起一份手写实现
- **同一物理量/公式只允许一处实现**：发现两处各写了一遍同样的数学，
  视为 bug，抽成共享函数——不是"以后记得同步改"，记不住才是常态
- **新增模块自带冒烟**：新文件要有最小可跑入口或测试，写完立刻跑通；
  "之后再补测试"= 永远不补
- **不留半成品**：跑不通的实验分支不合并进主线；暂时不用的代码直接删，
  历史有 git 管。死代码比没代码更危险——它会误导下一个读代码的人
- **依赖要显式**：新装第三方包必须登记进依赖清单（pyproject/requirements），
  不允许"我机器上有"式隐性依赖
- **报错给人看**：异常信息写清"哪里错了 + 下一步怎么办"，不写裸
  `raise Exception("error")`
<!-- /explab:section -->

<!-- explab:section Git 纪律 -->
## Git 纪律（仓库是 git 仓库时）

- **开工前确认工作区干净**（`git status`）：在脏树上跑实验，数字无法
  归属到任何一份代码，等于白跑
- **开工前若当前状态还没提交，先提交**：尤其是准备动共享代码/做大改
  之前——没提交就没有"回得去的过去"
- **提交粒度 = 一个语义变更**：一次 bug 修复、一次实验收尾（代码 +
  四件套同一个 commit）。不按时间、不按文件凑提交
- **四件套必须与产生它的代码同一次提交**：数字、记录、代码锁在一个
  commit 里，将来 checkout 任何一个 commit 都能还原那个数字的现场
- **提交信息格式**：`<实验ID或模块>: 一句话结论`，例如
  `6d-30k: ape +8.3 / can -24.2，回归定位到 bank`
- **不提交大文件**：数据 / 权重 / checkpoint / outputs 一律进
  .gitignore；产物在 run 记录里写路径引用，不进库
- **分支**：主线实验在主分支顺序推进即可；只有大重构/高风险改动
  （如架构迁移）才开分支
- **push 必须用户明确要求才做**；提交本身也只在用户要求或协议
  明确允许时执行
<!-- /explab:section -->
