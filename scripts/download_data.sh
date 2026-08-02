#!/usr/bin/env bash
# LineMod（BOP 格式）下载脚本：lm_base + lm_models + lm_test_all
# 用法: bash scripts/download_data.sh [目标目录，默认 data/]
# 磁盘需求约 6GB。BOP 官方托管在 huggingface（bop-benchmark）。
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

SRC="https://huggingface.co/datasets/bop-benchmark/lm/resolve/main"

echo "==> 下载 lm_base.zip（场景元数据）"
wget -c "$SRC/lm_base.zip"
echo "==> 下载 lm_models.zip（CAD 模型 + models_info.json）"
wget -c "$SRC/lm_models.zip"
echo "==> 下载 lm_test_all.zip（全部测试图像，约 5GB）"
wget -c "$SRC/lm_test_all.zip"

echo "==> 解压"
unzip -qo lm_base.zip            # 解出 lm/
unzip -qo lm_models.zip -d lm    # 解出 lm/models*, 含 models_eval
unzip -qo lm_test_all.zip -d lm  # 解出 lm/test/

echo "==> 完成。目录结构："
ls lm
echo "配置 configs/default.yaml 的 dataset.root 应指向 $(pwd)/lm"

# 可选：PVNet 式官方 train/test 划分（防参考帧泄漏）。放置到
#   data/splits/lm/<obj>_train.txt（一行一个帧号），loader 存在时参考帧
#   只取自 train 列表、评测在测试划分上进行，杜绝参考视图泄漏。
# 这些 split 文件请从 PVNet 官方仓库获取（见 PVNet 仓库 data/linemod 目录，
# 各物体子目录下的 train.txt / test.txt），本脚本不编造下载 URL。
echo "（可选）如需 PVNet 官方划分，见 PVNet 仓库 data/linemod 目录下的 train.txt，"
echo "       放到 data/splits/lm/<obj>_train.txt 即可自动生效。"
