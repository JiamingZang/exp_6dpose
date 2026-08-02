#!/bin/bash
# 12 物体提取（4 并行，避免 OOM）；eggbox/holepuncher 已完成跳过
set -e
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/extract12_v3

run_one() {
    obj=$1; cfg=$2
    if [ -d "outputs/matches13_v2/$obj" ] && [ "$(ls outputs/matches13_v2/$obj/ 2>/dev/null | grep -cE '^[0-9]+\.npz$')" -ge 120 ]; then
        echo "[$obj] already done"; return
    fi
    python scripts/extract_matches.py --config "configs/$cfg" --objects "$obj" \
        --max-frames 120 --matches-dir outputs/matches13_v2 \
        > "outputs/logs/extract12_v3/${obj}.log" 2>&1
    echo "[$obj] done"
}

export -f run_one
BLACK="benchvise can cat duck glue iron lamp phone"
for obj in $BLACK; do
    run_one $obj dense80_batch8_bg0.yaml &
    [ $(jobs -r | wc -l) -ge 3 ] && wait -n
done
run_one driller dense80_batch8.yaml &
run_one cam dense80_batch8.yaml &
wait
echo ALL_EXTRACT_DONE
