#!/bin/bash
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' \
  "eggbox:matches13_orig" "glue:matches13_orig" \
  "holepuncher:matches13_30k" "iron:matches13_orig" \
  "lamp:matches13_orig" "phone:matches13_orig" \
| xargs -P 2 -I {} bash -c '
  spec="$1"; o="${spec%%:*}"; m="${spec#*:}"
  python scripts/analysis/extract_matches.py --config "configs/archive/dense80_dc_b2.yaml" \
      --object "$o" --matches-dir "outputs/$m" \
      > "outputs/logs/train30k/extract_last_$o.log" 2>&1
' _ {}
echo EXTRACT_LAST6_DONE
