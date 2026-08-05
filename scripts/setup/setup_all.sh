#!/usr/bin/env bash
# 一键部署 + 冒烟自检总入口。租到 GPU 机器后在仓库根目录执行：
#
#   bash setup_all.sh          # 全流程：环境 → 数据 → 冒烟
#   bash setup_all.sh env      # 只装环境+权重（内部调 setup_gpu.sh）
#   bash setup_all.sh data     # 只下数据（内部调 scripts/data/download_data.sh）
#   bash setup_all.sh smoke    # 只跑冒烟（onboard ape → 20 帧评测）
#
# 幂等：已下载的权重/数据（wget -c）与已存在的 third_party 会跳过，重跑无害。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
STAGE="${1:-all}"

run_env() {
    echo "========== [1/3] 环境 + 权重 =========="
    bash setup_gpu.sh
}

run_data() {
    echo "========== [2/3] LineMod 数据（约 6GB）=========="
    bash scripts/data/download_data.sh data
}

run_smoke() {
    echo "========== [3/3] 冒烟自检 =========="
    # env.sh 由 setup_gpu.sh 生成（MASt3R 的 PYTHONPATH）
    [ -f env.sh ] && source env.sh
    echo "--> 单测（CPU 部分，应全绿）"
    python -m pytest tests/ -q
    echo "--> onboard ape（3DGS 训练 + 40 模板渲染，单卡约几分钟）"
    python scripts/data/onboard_object.py --objects ape
    echo "--> ape 前 20 帧端到端评测"
    python scripts/eval/run_linemod.py --objects ape --max-frames 20
    echo "冒烟通过。全量实验入口："
    echo "  python scripts/data/onboard_object.py                 # 全部 13 物体建模板库"
    echo "  python scripts/eval/run_linemod.py                    # 主实验（--bop-csv 可导出 BOP 提交文件）"
    echo "  python scripts/eval/run_ablation.py --help            # 消融"
}

case "$STAGE" in
    env)   run_env ;;
    data)  run_data ;;
    smoke) run_smoke ;;
    all)   run_env; run_data; run_smoke ;;
    *) echo "用法: bash setup_all.sh [all|env|data|smoke]"; exit 2 ;;
esac
