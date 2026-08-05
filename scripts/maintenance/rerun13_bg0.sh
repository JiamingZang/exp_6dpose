#!/usr/bin/env bash
# 全部物体用黑背景+深度0.6 重训重建 → 重提取 → 评估（不关机）
set -u
cd /root/毕设/exp_6dpose
LOCK=outputs/logs/rerun13_bg0.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running"; exit 1; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
LOG=outputs/logs/rerun13_bg0
mkdir -p "$LOG"
OBJS12="benchvise cam can cat driller duck glue holepuncher iron lamp phone"

echo "==== [$(date '+%T')] PHASE1: 12 物体黑背景重训+重建 ===="
for obj in $OBJS12; do
  t0=$(date +%s)
  rm -f "outputs/templates/${obj}_3dgs_cad_80t_sa.npz" \
        "outputs/templates/${obj}_3dgs_cad_80t_sa.pt"
  bash -c "source env.sh && python scripts/data/onboard_object.py \
    --config configs/current/dense80_depth_bg0.yaml --objects $obj" \
    > "$LOG/onboard_$obj.log" 2>&1 || { echo "[FAIL] onboard $obj"; tail -8 "$LOG/onboard_$obj.log"; exit 1; }
  bash -c "source env.sh && python scripts/data/rebuild_bank_fixed_views.py \
    --config configs/current/dense80_depth_bg0.yaml --objects $obj \
    --views-from outputs/templates/bank_backup_cadpatch/${obj}_3dgs_cad_80t_sa.npz" \
    > "$LOG/rebuild_$obj.log" 2>&1 || { echo "[FAIL] rebuild $obj"; tail -8 "$LOG/rebuild_$obj.log"; exit 1; }
  echo "[$(date '+%T')] $obj 完成 $(( $(date +%s) - t0 ))s"
done

echo "==== [$(date '+%T')] PHASE2: 重提取 120 帧 × 12 物体 ===="
rm -rf outputs/matches13_ds/{benchvise,cam,can,cat,driller,duck,glue,holepuncher,iron,lamp,phone}
run2() {
  local name=$1; shift
  bash -c "source env.sh && python scripts/analysis/extract_matches.py \
    --config configs/archive/dense80_batch4.yaml --matches-dir outputs/matches13_ds \
    --max-frames 120 --objects $*" > "$LOG/extract_$name.log" 2>&1
}
run2 g1 benchvise cam can cat & P1=$!
run2 g2 driller duck glue holepuncher & P2=$!
run2 g3 iron lamp phone & P3=$!
wait $P1 $P2 $P3
for g in g1 g2 g3; do
  if grep -q "OutOfMemoryError" "$LOG/extract_$g.log" 2>/dev/null; then
    echo "[$(date '+%T')] WARN $g OOM"
    bash -c "source env.sh && python scripts/analysis/extract_matches.py \
      --config configs/experiments/dense80_batch2.yaml --matches-dir outputs/matches13_ds \
      --max-frames 120 --objects ${OBJS12}" > "$LOG/extract_${g}_b2.log" 2>&1
  fi
done
for obj in $OBJS12; do
  n=$(find outputs/matches13_ds/$obj -name "*.npz" 2>/dev/null | grep -vc _alt || true)
  if [ "$n" -lt 110 ]; then echo "[FAIL] extract $obj $n 帧"; exit 1; fi
  echo "[$(date '+%T')] extract $obj: $n 帧"
done

echo "==== [$(date '+%T')] PHASE3: stage3 13 物体 ===="
rm -rf outputs/cache13_ds outputs/linemod13_ds
mkdir -p outputs/linemod13_ds
run3() {
  local obj=$1 mdir=$2
  bash -c "source env.sh && python scripts/eval/run_linemod.py \
    --config configs/current/dense80.yaml --matches-dir $mdir --max-frames 120 \
    --objects $obj --cache-dir outputs/cache13_ds \
    --out outputs/linemod13_ds/$obj.json" > "$LOG/stage3_$obj.log" 2>&1
}
OBJS_ALL="ape benchvise cam can cat driller duck eggbox glue holepuncher iron lamp phone"
i=0
for obj in $OBJS_ALL; do
  if [ "$obj" = "ape" ]; then mdir=outputs/matches_ape_full; else mdir=outputs/matches13_ds; fi
  run3 "$obj" "$mdir" & P=$!
  i=$((i+1))
  if [ $((i % 4)) -eq 0 ]; then wait $P; fi
done
wait
echo "==== [$(date '+%T')] PHASE4: 汇总 ===="
python scripts/eval/summarize13.py | tee "$LOG/summary.txt"
