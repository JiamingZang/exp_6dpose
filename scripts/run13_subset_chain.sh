#!/usr/bin/env bash
# 13 物体子集全链（自动，约 4 小时）：DS 重训+固定视图重建 → extract 120 帧
# 均匀采样（3 进程 batch4，OOM 自动 batch2 续跑）→ stage3 评估 → 汇总 → 关机
set -u
cd /root/毕设/exp_6dpose
LOCK=outputs/logs/run13_subset.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another chain instance is running, exit"; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
LOG=outputs/logs/run13_subset
mkdir -p "$LOG" outputs/matches13_ds outputs/linemod13_ds
OBJS12="benchvise cam can cat driller duck eggbox glue holepuncher iron lamp phone"

echo "==== [$(date '+%F %T')] PHASE1: 12 物体 DS 重训 + 固定视图重建 ===="
for obj in $OBJS12; do
  t0=$(date +%s)
  rm -f "outputs/templates/${obj}_3dgs_cad_80t_sa.npz" \
        "outputs/templates/${obj}_3dgs_cad_80t_sa.pt"
  bash -c "source env.sh && python scripts/onboard_object.py \
    --config configs/dense80_depth.yaml --objects $obj" \
    > "$LOG/onboard_$obj.log" 2>&1 || {
      echo "[FAIL] onboard $obj"; tail -8 "$LOG/onboard_$obj.log"; exit 1; }
  bash -c "source env.sh && python scripts/rebuild_bank_fixed_views.py \
    --config configs/dense80.yaml --objects $obj \
    --views-from outputs/templates/bank_backup_cadpatch/${obj}_3dgs_cad_80t_sa.npz" \
    > "$LOG/rebuild_$obj.log" 2>&1 || {
      echo "[FAIL] rebuild $obj"; tail -8 "$LOG/rebuild_$obj.log"; exit 1; }
  echo "[$(date '+%T')] phase1 $obj 完成 $(( $(date +%s) - t0 ))s"
done
echo "==== [$(date '+%F %T')] PHASE1 完成 ===="

echo "==== [$(date '+%F %T')] PHASE2: extract 120 帧 × 12 物体（3 进程 batch4）===="
run2() {
  local name=$1; shift
  bash -c "source env.sh && python scripts/extract_matches.py \
    --config configs/dense80_batch4.yaml --matches-dir outputs/matches13_ds \
    --max-frames 120 --objects $*" > "$LOG/extract_$name.log" 2>&1
}
run2 g1 benchvise cam can cat & P1=$!
run2 g2 driller duck eggbox glue & P2=$!
run2 g3 holepuncher iron lamp phone & P3=$!
wait $P1 $P2 $P3
# OOM 检测：按组用 batch2 续跑（extract 幂等，跳过已完成帧）
retry2() {
  local name=$1; shift
  bash -c "source env.sh && python scripts/extract_matches.py \
    --config configs/dense80_batch2.yaml --matches-dir outputs/matches13_ds \
    --max-frames 120 --objects $*" > "$LOG/extract_${name}_b2.log" 2>&1
}
for g in g1 g2 g3; do
  if grep -q "OutOfMemoryError" "$LOG/extract_$g.log" 2>/dev/null; then
    echo "[$(date '+%T')] WARN $g OOM → batch2 续跑"
    case $g in
      g1) retry2 g1 benchvise cam can cat ;;
      g2) retry2 g2 driller duck eggbox glue ;;
      g3) retry2 g3 holepuncher iron lamp phone ;;
    esac
  fi
done
# 完成度 gate：每物体至少 110/120 帧
for obj in $OBJS12; do
  n=$(find outputs/matches13_ds/$obj -name "*.npz" 2>/dev/null | wc -l)
  if [ "$n" -lt 110 ]; then
    echo "[FAIL] extract $obj 只有 $n 帧"; exit 1
  fi
  echo "[$(date '+%T')] extract $obj: $n 帧"
done

echo "==== [$(date '+%F %T')] PHASE3: stage3 评估 13 物体 × 120 帧（4 进程）===="
run3() {
  local obj=$1 mdir=$2
  bash -c "source env.sh && python scripts/run_linemod.py \
    --config configs/dense80.yaml --matches-dir $mdir --max-frames 120 \
    --objects $obj --cache-dir outputs/cache13_ds \
    --out outputs/linemod13_ds/$obj.json" > "$LOG/stage3_$obj.log" 2>&1
}
run3 ape outputs/matches_ape_full & J1=$!
run3 benchvise outputs/matches13_ds & J2=$!
run3 cam outputs/matches13_ds & J3=$!
run3 can outputs/matches13_ds & J4=$!
wait $J1 $J2 $J3 $J4
run3 cat outputs/matches13_ds & J1=$!
run3 driller outputs/matches13_ds & J2=$!
run3 duck outputs/matches13_ds & J3=$!
run3 eggbox outputs/matches13_ds & J4=$!
wait $J1 $J2 $J3 $J4
run3 glue outputs/matches13_ds & J1=$!
run3 holepuncher outputs/matches13_ds & J2=$!
run3 iron outputs/matches13_ds & J3=$!
run3 lamp outputs/matches13_ds & J4=$!
wait $J1 $J2 $J3 $J4
run3 phone outputs/matches13_ds & J1=$!
wait $J1
echo "==== [$(date '+%F %T')] PHASE3 完成 ===="

echo "==== [$(date '+%F %T')] PHASE4: 汇总 ===="
python scripts/summarize13.py | tee "$LOG/summary.txt"
echo "==== [$(date '+%F %T')] 全部完成，60 秒后关机 ===="
sleep 60
/usr/bin/shutdown
