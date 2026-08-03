#!/bin/bash
# 13 物体 stage3（并行 4 进程）：matches13_dc2 + depth_consistency + guided_refine
set -e
cd "$(dirname "$0")/../.."
source env.sh
mkdir -p outputs/logs/stage3_13_dc2
i=0
for obj in ape benchvise cam can cat driller duck eggbox glue holepuncher iron lamp phone; do
    python scripts/run_linemod.py --config configs/dense80_depthc_guided.yaml \
        --objects $obj --matches-dir outputs/matches13_dc2 --max-frames 120 \
        --cache-dir outputs/cache13_dc2 --out "outputs/linemod13_dc2/$obj.json" \
        > "outputs/logs/stage3_13_dc2/${obj}.log" 2>&1 &
    i=$((i+1)); [ $i -ge 4 ] && wait -n
done
wait
echo ALL_STAGE3_DONE
