#!/bin/bash
# 全量匹配提取（每物体全部帧，无 max-frames）
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' \
  "ape:matches13_30k" "benchvise:matches13_orig" "cam:matches13_w1" \
  "can:matches13_dc2" "cat:matches13_orig" "driller:matches13_w1" \
  "duck:matches13_30k" "eggbox:matches13_orig" "glue:matches13_orig" \
  "holepuncher:matches13_30k" "iron:matches13_orig" "lamp:matches13_orig" \
  "phone:matches13_orig" \
| xargs -P 2 -I {} bash -c '
  spec="$1"; o="${spec%%:*}"; m="${spec#*:}"
  cfg="dense80_dc_b4"
  [ "$o" = "cam" ] || [ "$o" = "driller" ] && cfg="dense80_w1_b4"
  python scripts/extract_matches.py --config "configs/$cfg.yaml" \
      --object "$o" --matches-dir "outputs/$m" \
      > "outputs/logs/train30k/extract_full_$o.log" 2>&1
' _ {}
echo EXTRACT_FULL_DONE
