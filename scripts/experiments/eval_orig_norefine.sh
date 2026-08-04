#!/bin/bash
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' benchvise cam cat driller eggbox glue iron lamp phone \
| xargs -P 4 -I {} bash -c '
  python scripts/run_linemod.py --config configs/dense80_dc_norefine.yaml \
      --objects "$1" --matches-dir outputs/matches13_orig --max-frames 120 \
      --cache-dir outputs/exp_orig_norefine/cache \
      --out "outputs/exp_orig_norefine/results/$1.json" \
      > "outputs/logs/train30k/eval_orig_$1.log" 2>&1' _ {}
echo EVAL_ORIG_NOREFINE_DONE
