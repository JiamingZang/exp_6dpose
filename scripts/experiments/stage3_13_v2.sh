#!/bin/bash
# 13 物体 stage3（并行 4 进程）：12 物体用 matches13_v2，ape 用黑 0.6 子集
set -e
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/stage3_13_v2
BLACK="benchvise can cat duck eggbox glue holepuncher iron lamp phone"
i=0
for obj in $BLACK; do
    python scripts/run_linemod.py --config configs/dense80_depth_bg0.yaml \
        --objects $obj --matches-dir outputs/matches13_v2 --max-frames 120 \
        --cache-dir outputs/cache13_v2 --out "outputs/linemod13_v2/$obj.json" \
        > "outputs/logs/stage3_13_v2/${obj}.log" 2>&1 &
    i=$((i+1)); [ $i -ge 4 ] && wait -n
done
python scripts/run_linemod.py --config configs/dense80_depth_w1.yaml \
    --objects driller --matches-dir outputs/matches13_v2 --max-frames 120 \
    --cache-dir outputs/cache13_v2 --out "outputs/linemod13_v2/driller.json" \
    > outputs/logs/stage3_13_v2/driller.log 2>&1 &
python scripts/run_linemod.py --config configs/dense80_depth_w1.yaml \
    --objects cam --matches-dir outputs/matches13_v2 --max-frames 120 \
    --cache-dir outputs/cache13_v2 --out "outputs/linemod13_v2/cam.json" \
    > outputs/logs/stage3_13_v2/cam.log 2>&1 &
wait
echo ALL_STAGE3_DONE
