#!/bin/bash
cd "$(dirname "$0")/.."
source env.sh
run_one() {
    obj=$1; cfg=$2
    n=$(ls "outputs/matches13_dc2/$obj/" 2>/dev/null | grep -cE '^[0-9]+\.npz$')
    if [ "$n" -ge 120 ]; then echo "[$obj] ok($n)"; return; fi
    rm -rf "outputs/matches13_dc2/$obj"
    python scripts/extract_matches.py --config "configs/$cfg" --objects "$obj" \
        --max-frames 120 --matches-dir outputs/matches13_dc2 \
        > "outputs/logs/extract13_dc2/${obj}.log" 2>&1
    echo "[$obj] done"
}
export -f run_one
run_one eggbox dense80_batch8_bg0.yaml &
run_one glue dense80_batch8_bg0.yaml &
wait
run_one holepuncher dense80_batch8_bg0.yaml &
run_one iron dense80_batch8_bg0.yaml &
wait
run_one lamp dense80_batch8_bg0.yaml &
run_one phone dense80_batch8_bg0.yaml &
wait
run_one cam dense80_batch8.yaml &
run_one driller dense80_batch8.yaml &
wait
echo ALL_FIX_DONE
