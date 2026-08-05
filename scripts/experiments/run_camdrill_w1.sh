#!/bin/bash
# cam/driller w1(.orig bg1) 提取 + 裸 PnP 评估，串行防 OOM
cd /root/毕设/exp_6dpose
source env.sh
for o in cam driller; do
  python scripts/analysis/extract_matches.py --config configs/current/dense80_w1_norefine.yaml \
      --object "$o" --matches-dir outputs/matches13_w1 --max-frames 120 \
      > "outputs/logs/train30k/extract_w1_$o.log" 2>&1
  python scripts/eval/run_linemod.py --config configs/current/dense80_w1_norefine.yaml \
      --objects "$o" --matches-dir outputs/matches13_w1 --max-frames 120 \
      --cache-dir outputs/exp_w1_norefine/cache \
      --out "outputs/exp_w1_norefine/results/$o.json" \
      > "outputs/logs/train30k/eval_w1_$o.log" 2>&1
done
echo CAMDRILL_W1_DONE
