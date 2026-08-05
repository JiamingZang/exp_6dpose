#!/bin/bash
# 9 物体 30k bank 匹配提取（xargs -P 2 防 OOM，--max-frames 120 与评估同口径）
cd /root/毕设/exp_6dpose
source env.sh
printf '%s\n' benchvise cam cat driller eggbox glue iron lamp phone \
| xargs -P 2 -I {} bash -c '
  python scripts/analysis/extract_matches.py --config configs/archive/dense80_dc_b4.yaml \
      --object "$1" --matches-dir outputs/matches13_30k --max-frames 120 \
      > "outputs/logs/train30k/extract_$1.log" 2>&1' _ {}
echo EXTRACT9_120_DONE
