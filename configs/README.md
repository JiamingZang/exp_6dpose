# configs 目录结构

- `current/`: 当前主线、复现入口和近期候选配置。
- `archive/`: 历史验证、失败路线或只为追溯保留的配置。
- `ablations/`: 论文消融配置。
- 根目录不放 yaml；新命令必须写 `configs/current/...` 或 `configs/archive/...`。
