#!/bin/bash
# 12 物体重新提取 120 帧（黑 batch8_bg0 / 白 batch8），并行 4 进程
set -e
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/extract12_v2

run_one() {
    obj=$1; cfg=$2; out=$3
    rm -rf "$out/$obj"
    python scripts/extract_matches.py --config "configs/$cfg" --objects "$obj" \
        --max-frames 120 --matches-dir "$out" \
        > "outputs/logs/extract12_v2/${obj}.log" 2>&1 &
    echo "[$obj] extract launched"
}

BLACK="benchvise can cat duck eggbox glue holepuncher iron lamp phone"
for obj in $BLACK; do run_one $obj dense80_batch8_bg0.yaml outputs/matches13_v2; done
run_one driller dense80_batch8.yaml outputs/matches13_v2
run_one cam dense80_batch8.yaml outputs/matches13_v2
wait
echo ALL_EXTRACT_DONE
