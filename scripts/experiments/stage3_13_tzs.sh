#!/bin/bash
# 13 物体 stage3：depthc + guided + tz 面积比校准 + xy 质心对齐
cd "$(dirname "$0")/../.."
source env.sh
mkdir -p outputs/logs/stage3_13_tzs
i=0
for obj in ape benchvise cam can cat driller duck eggbox glue holepuncher iron lamp phone; do
    python scripts/run_linemod.py --config configs/dense80_tzsearch.yaml \
        --objects $obj --matches-dir outputs/matches13_dc2 --max-frames 120 \
        --cache-dir outputs/cache13_tzs --out "outputs/linemod13_tzs/$obj.json" \
        > "outputs/logs/stage3_13_tzs/${obj}.log" 2>&1 &
    i=$((i+1)); [ $i -ge 4 ] && wait -n
done
wait
echo ALL_STAGE3_TZS_DONE
