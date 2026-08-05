# 实验流程

本目录是 agent 做实验的唯一入口，避免口头待办、临时命令和结果记录散落。

## 强制顺序

1. 从 `experiments/QUEUE.md` 领取一项，把状态改为 `running`，同一时间只允许一项 `running`。
2. 复制 `experiments/RUN_TEMPLATE.md` 到 `experiments/runs/<实验ID>.md`，填完 Proposal/Plan 后再开跑。
3. 新实验必须新增配置文件，放在 `configs/current/`；禁止直接改旧配置复跑。
4. 跑完同一次操作内更新：
   - `experiments/runs/<实验ID>.md`
   - `experiments/QUEUE.md`
   - `docs/STATE.md`
   - `docs/LEDGER.md`
5. 收尾前运行：`python3 scripts/analysis/check_state.py`。

## 状态定义

- `todo`：可领取。
- `running`：正在跑；只能有一个。
- `blocked`：缺数据/权重/GPU/代码，不能盲跑。
- `done`：有结果、有结论、已同步 STATE/LEDGER。
- `dead`：已证伪，禁止重跑。
