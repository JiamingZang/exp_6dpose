#!/bin/bash
# 回退保护全 13 物体评估（P4 并行；每物体用各自最优 bank 的匹配）
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' \
  "ape:matches13_30k:dense80_depthc_guided" \
  "benchvise:matches13_orig:dense80_depthc_guided" \
  "cam:matches13_w1:dense80_w1" \
  "can:matches13_orig:dense80_depthc_guided" \
  "cat:matches13_orig:dense80_depthc_guided" \
  "driller:matches13_w1:dense80_w1" \
  "duck:matches13_30k:dense80_depthc_guided" \
  "eggbox:matches13_orig:dense80_depthc_guided" \
  "glue:matches13_orig:dense80_depthc_guided" \
  "holepuncher:matches13_30k:dense80_depthc_guided" \
  "iron:matches13_orig:dense80_depthc_guided" \
  "lamp:matches13_orig:dense80_depthc_guided" \
  "phone:matches13_orig:dense80_depthc_guided" \
| xargs -P 4 -I {} bash -c '
  spec="$1"; o="${spec%%:*}"; rest="${spec#*:}"
  m="${rest%%:*}"; c="${rest#*:}"
  python scripts/run_linemod.py --config "configs/$c.yaml" \
      --objects "$o" --matches-dir "outputs/$m" --max-frames 120 \
      --cache-dir "outputs/exp_rollback/cache" \
      --out "outputs/exp_rollback/results/$o.json" \
      > "outputs/logs/train30k/eval_rb_$o.log" 2>&1
' _ {}
echo ROLLBACK_ALL_DONE
