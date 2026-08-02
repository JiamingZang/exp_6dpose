#!/usr/bin/env bash
# 最终评估：dense80 + 排序修复 + RANSAC 上限，4 进程并行（~20GB 显存，GPU 拉满）
set -e
cd "$(dirname "$0")/.."
CFG=configs/dense80.yaml
MDIR=outputs/matches_dense80
CACHE=outputs/cache_dense80_rs6
LOG=outputs/logs/eval_rs6
mkdir -p "$LOG"

run() {  # run <name> <out-json> <objects...>
  local name=$1 out=$2; shift 2
  nohup bash -c "source env.sh && python scripts/run_linemod.py \
    --config $CFG --matches-dir $MDIR --cache-dir $CACHE --max-frames 10 \
    --out $out --aggregated-out ${out%.json}_agg.json --objects $*" \
    > "$LOG/$name.log" 2>&1 &
  echo "launched $name (pid $!) -> $LOG/$name.log"
}

run p1 outputs/linemod_rs6_p1.json ape benchvise cam can
run p2 outputs/linemod_rs6_p2.json cat driller duck eggbox
run p3 outputs/linemod_rs6_p3.json glue holepuncher iron lamp
run p4 outputs/linemod_rs6_p4.json phone
wait
echo "all 4 final eval processes done"
