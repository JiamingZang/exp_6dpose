#!/bin/bash
# 补跑剩余 8 物体全量提取（新架构路径 scripts/analysis/extract_matches.py）
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' \
  "driller:matches13_w1:dense80_w1" \
  "duck:matches13_30k:dense80_depthc_guided" \
  "eggbox:matches13_orig:dense80_depthc_guided" \
  "glue:matches13_orig:dense80_depthc_guided" \
  "holepuncher:matches13_30k:dense80_depthc_guided" \
  "iron:matches13_orig:dense80_depthc_guided" \
  "lamp:matches13_orig:dense80_depthc_guided" \
  "phone:matches13_orig:dense80_depthc_guided" \
| xargs -P 2 -I {} bash -c '
  spec="$1"; o="${spec%%:*}"; rest="${spec#*:}"
  m="${rest%%:*}"; c="${rest#*:}"
  python scripts/analysis/extract_matches.py --config "configs/current/$c.yaml" \
      --object "$o" --matches-dir "outputs/$m" \
      > "outputs/logs/train30k/extract_rest_$o.log" 2>&1
' _ {}
echo EXTRACT_REST8_DONE
