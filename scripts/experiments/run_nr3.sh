#!/bin/bash
# ape/duck/holepuncher 30k bank 裸 PnP（3 物体串行）
cd /root/毕设/exp_6dpose
source env.sh
for o in ape duck holepuncher; do
  python scripts/eval/run_linemod.py --config configs/current/dense80_dc_norefine.yaml \
      --objects "$o" --matches-dir outputs/matches13_30k --max-frames 120 \
      --cache-dir outputs/exp_30k_norefine/cache \
      --out "outputs/exp_30k_norefine/results/$o.json" \
      > "outputs/logs/train30k/eval_30knr_$o.log" 2>&1
done
echo NR3_DONE
