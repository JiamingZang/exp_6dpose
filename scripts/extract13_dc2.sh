#!/bin/bash
# 13 物体重提取（含 pts3d_q，2 并行防 OOM）
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/extract13_dc2

run_one() {
    obj=$1; cfg=$2; out=$3
    rm -rf "$out/$obj"
    python scripts/extract_matches.py --config "configs/$cfg" --objects "$obj" \
        --max-frames 120 --matches-dir "$out" \
        > "outputs/logs/extract13_dc2/${obj}.log" 2>&1
    echo "[$obj] done"
}
export -f run_one
# ape 单独（黑 batch8_bg0）
run_one ape dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one benchvise dense80_batch8_bg0.yaml outputs/matches13_dc2 &
run_one can dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one cat dense80_batch8_bg0.yaml outputs/matches13_dc2 &
run_one duck dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one eggbox dense80_batch8_bg0.yaml outputs/matches13_dc2 &
run_one glue dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one holepuncher dense80_batch8_bg0.yaml outputs/matches13_dc2 &
run_one iron dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one lamp dense80_batch8_bg0.yaml outputs/matches13_dc2 &
run_one phone dense80_batch8_bg0.yaml outputs/matches13_dc2 &
wait
run_one cam dense80_batch8.yaml outputs/matches13_dc2 &
run_one driller dense80_batch8.yaml outputs/matches13_dc2 &
wait
echo ALL_EXTRACT_DONE
