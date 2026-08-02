#!/usr/bin/env bash
# dense80 并行 extract：3 进程 × batch8（单进程 ~6GB，32GB 卡余量充足）。
# 每进程约 4-5 个物体，产物落 outputs/matches_dense80（不覆盖 40t 的 matches）。
set -e
cd "$(dirname "$0")/.."

CFG=configs/dense80_batch8.yaml
MDIR=outputs/matches_dense80
LOG=outputs/logs/extract_dense80
mkdir -p "$LOG"

run() {  # run <logname> <objects...>
  local name=$1; shift
  nohup bash -c "source env.sh && python scripts/extract_matches.py \
    --config $CFG --matches-dir $MDIR --max-frames 10 --objects $*" \
    > "$LOG/$name.log" 2>&1 &
  echo "launched $name (pid $!) -> $LOG/$name.log"
}

run p1 ape benchvise cam can
run p2 cat driller duck eggbox
run p3 glue holepuncher iron lamp phone
wait
echo "all extract processes done"
