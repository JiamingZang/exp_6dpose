#!/bin/bash
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' cam cat driller eggbox glue iron lamp phone \
| xargs -P 2 -I {} bash -c '
  python scripts/analysis/extract_matches.py --config configs/archive/dense80_dc_b4.yaml \
      --object "$1" --matches-dir outputs/matches13_orig --max-frames 120 \
      > "outputs/logs/train30k/extract_orig_$1.log" 2>&1' _ {}
echo EXTRACT_ORIG8_DONE
