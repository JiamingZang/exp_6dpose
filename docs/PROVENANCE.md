# PROVENANCE —— 参考代码来源登记

third_party/ 里每一个外部代码条目都必须在这里登记。
纪律：third_party 内代码只读、钉死版本；适配写在 src/；
必须打补丁时用 patch 文件并在"适配/补丁"列登记。

| 名字 | 来源（URL/仓库） | 版本/commit | 引入日期 | 用途 | 适配/补丁 |
|---|---|---|---|---|---|
| `mast3r` | https://github.com/naver/mast3r （含 dust3r 子模块，`--recursive` 克隆） | **未钉**：`scripts/setup/setup_gpu.sh:16` 克隆 HEAD，且本地树 `.git` 已缺失，原克隆 commit 不可恢复。上游参照：`f5209afc`（naver/mast3r HEAD，2026-08-04 经 `git ls-remote` 查得）；下次部署 GPU 环境时应重克隆并 `git checkout` 钉死该 commit，同时补回 `.git` 或把 commit 写入本表 | 不可考（树无 git 元数据） | 稠密局部特征匹配（模板检索 + 互最近邻对应），经 PYTHONPATH 引入 | 无就地修改；适配层在 `src/matching/mast3r_wrapper.py`（编码器特征预缓存、模板级打分、GPU argmax），对应点逻辑复用 `src/matching/correspondence.py`；官方权重 MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric 由 setup_gpu.sh 下载到 `weights/` |
