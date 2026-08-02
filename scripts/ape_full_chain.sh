#!/usr/bin/env bash
# ape 全量测试集链路：seg0 重启（supervisor 防 OOM）→ extract 全部完成 → 4 分片阶段 3
cd /root/毕设/exp_6dpose
LOCK=outputs/logs/ape_full_chain.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another chain instance is running, exit"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
PROC() { pgrep -fc '^python scripts/extract_matches.py'; }
SEG0() { python scripts/count_seg_frames.py ape 0 391; }

while [ "$(PROC)" -ge 2 ]; do sleep 30; done

N0=$(SEG0)
if [ "$N0" != "391/391" ]; then
  echo "STEP1: seg0 未完成 ($N0)，启动 supervisor"
  nohup bash scripts/seg0_supervisor.sh >> outputs/logs/seg0_supervisor.log 2>&1 &
  SUP_PID=$!
  # 等 supervisor 真正结束（内部最多重试 5 次直到 391 帧落盘），
  # 而不是轮询进程数——python 启动有 30-60s 延迟，轮询会误判完成
  wait "$SUP_PID"
else
  echo "STEP1-skip: seg0 已完成 ($N0)"
fi

N0=$(SEG0)
if [ "$N0" != "391/391" ]; then
  echo "STEP2-FAIL: seg0 只有 $N0 帧，中止阶段 3"
  exit 1
fi
echo "STEP2: extract 全部完成，启动阶段 3 (4 分片)"

run3() {
  local name=$1 first=$2 last=$3
  nohup bash -c "source env.sh && python scripts/run_linemod.py \
    --config configs/dense80.yaml --matches-dir outputs/matches_ape_full \
    --cache-dir outputs/cache_ape_full --objects ape \
    --first $first --last $last \
    --out outputs/linemod_ape_full_$name.json \
    --aggregated-out outputs/linemod_ape_full_${name}_agg.json" \
    > outputs/logs/stage3_ape_full_$name.log 2>&1 &
  echo "launched stage3 $name ($first-$last, pid $!)"
}
run3 p0 0 293
run3 p1 293 586
run3 p2 586 879
run3 p3 879 1172
wait
echo "STEP3: ape 全量阶段 3 完成"
