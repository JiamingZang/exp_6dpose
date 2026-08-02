#!/bin/bash
# 12 物体提取（2 并行，安全显存）；已完成跳过
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/extract12_v4

run_one() {
    obj=$1; cfg=$2
    if [ -d "outputs/matches13_v2/$obj" ] && [ "$(ls outputs/matches13_v2/$obj/ 2>/dev/null | grep -cE '^[0-9]+\.npz$')" -ge 120 ]; then
        echo "[$obj] already done"; return
    fi
    python scripts/extract_matches.py --config "configs/$cfg" --objects "$obj" \
        --max-frames 120 --matches-dir outputs/matches13_v2 \
        > "outputs/logs/extract12_v4/${obj}.log" 2>&1
    echo "[$obj] done"
}

export -f run_one
run_one cam dense80_batch8.yaml &
run_one cat dense80_batch8_bg0.yaml &
wait
run_one duck dense80_batch8_bg0.yaml &
run_one driller dense80_batch8.yaml &
wait
run_one glue dense80_batch8_bg0.yaml &
run_one iron dense80_batch8_bg0.yaml &
wait
run_one lamp dense80_batch8_bg0.yaml &
run_one phone dense80_batch8_bg0.yaml &
wait
echo ALL_EXTRACT_DONE
