#!/bin/bash
# 12 物体重训（黑 0.6 ×10 / 白 0.6 ×2）+ 固定视图重建（高斯锚点）
set -e
cd "$(dirname "$0")/.."
source env.sh
mkdir -p outputs/logs/rebuild12_v2

run_one() {
    obj=$1; cfg=$2
    rm -f "outputs/templates/${obj}_3dgs_cad_80t_sa.npz" "outputs/templates/${obj}_3dgs_cad_80t_sa.pt"
    python scripts/onboard_object.py --config "configs/$cfg" --objects "$obj" \
        > "outputs/logs/rebuild12_v2/onboard_${obj}.log" 2>&1
    python scripts/rebuild_bank_fixed_views.py --config "configs/$cfg" --objects "$obj" \
        >> "outputs/logs/rebuild12_v2/onboard_${obj}.log" 2>&1
    echo "[$obj] done"
}

export -f run_one
BLACK="benchvise can cat duck eggbox glue holepuncher iron lamp phone"
for obj in $BLACK; do
    run_one $obj dense80_depth_bg0.yaml &
    [ $(jobs -r | wc -l) -ge 3 ] && wait -n
done
run_one driller dense80_depth_w1.yaml &
run_one cam dense80_depth_w1.yaml &
wait
echo ALL_ONBOARD_DONE
