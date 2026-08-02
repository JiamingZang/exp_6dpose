#!/usr/bin/env bash
# seg0 监督器：batch4 跑，OOM/异常自动降 batch2 重试，直到 391 帧全部落盘。
# 文件按帧跳过，重试不浪费已算帧。
cd /root/毕设/exp_6dpose
source env.sh

frames_done() {
  find outputs/matches_ape_full/ape -name "*.npz" ! -name "*alt*" 2>/dev/null \
    | sed 's|.*/||;s|\.npz||' \
    | awk '{f=$1+0; if(f<391) n++} END {print n+0}'
}

for attempt in 1 2 3 4 5; do
  BATCH=$([ "$attempt" -le 2 ] && echo 4 || echo 2)
  echo "seg0 attempt $attempt (batch$BATCH) start: $(frames_done)/391"
  python scripts/extract_matches.py \
    --config configs/dense80k40_batch${BATCH}.yaml \
    --matches-dir outputs/matches_ape_full --first 0 --last 391 --objects ape \
    >> outputs/logs/extract_ape_full_0d.log 2>&1
  N=$(frames_done)
  if [ "$N" -ge 391 ]; then
    echo "seg0 DONE: $N/391 frames"
    exit 0
  fi
  echo "seg0 attempt $attempt ended with $N/391 frames, retrying"
done
echo "seg0 FAILED after 5 attempts: $(frames_done)/391"
exit 1
