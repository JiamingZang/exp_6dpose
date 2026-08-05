#!/usr/bin/env bash
# GPU 机器一键部署（Ubuntu + CUDA 11.8/12.x，单卡 ≥16GB 显存，推荐 RTX 4090/24GB）
# 用法: bash setup_gpu.sh
# 完成后按 README「从零复现」小节顺序执行。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> 1/5 Python 环境（建议先 conda create -n pose6d python=3.10 && conda activate pose6d）"
pip install -r requirements.txt

echo "==> 2/5 克隆 MASt3R（含 dust3r 子模块），加入 PYTHONPATH"
if [ ! -d third_party/mast3r ]; then
    mkdir -p third_party
    git clone --recursive https://github.com/naver/mast3r third_party/mast3r
fi
# RoPE CUDA 核（可选加速；失败不影响正确性，会退回纯 PyTorch 实现）
(cd third_party/mast3r/dust3r/croco/models/curope \
    && python setup.py build_ext --inplace) || \
    echo "[warn] curope 编译失败，使用纯 PyTorch RoPE（速度略慢）"

echo "==> 3/5 下载权重到 weights/"
mkdir -p weights
# MASt3R 官方权重（论文 3.1.3；官方唯一 ViT-L 权重，
# 名称见 third_party/mast3r/README.md:134-142）
wget -c -P weights \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# SAM ViT-H（论文 2.4.1）
wget -c -P weights \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# DINOv2 ViT-L/14 经 torch.hub 首次调用时自动下载（~1.2GB，缓存在 ~/.cache/torch）

echo "==> 4/5 写 PYTHONPATH（每次新 shell 需要 source 本文件或手动 export）"
export PYTHONPATH="$ROOT/third_party/mast3r:$ROOT/third_party/mast3r/dust3r:${PYTHONPATH:-}"
echo "export PYTHONPATH=\"$ROOT/third_party/mast3r:$ROOT/third_party/mast3r/dust3r:\$PYTHONPATH\"" > env.sh
echo "已生成 env.sh，之后每个 shell 先: source env.sh"

echo "==> 5/5 自检：关键依赖可导入"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
import gsplat; print("gsplat", gsplat.__version__)
from mast3r.model import AsymmetricMASt3R; print("mast3r OK")
from segment_anything import sam_model_registry; print("segment_anything OK")
PY

echo "==> 完成。下一步: bash scripts/data/download_data.sh && source env.sh"
