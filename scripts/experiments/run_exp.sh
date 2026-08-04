#!/bin/bash
# 实验编排：onboard → extract → evaluate 一条龙，产物按实验名隔离。
# 用法：./run_exp.sh <exp_name> <config> [objects...]
#   exp_name 决定 outputs/exp_<name>/{cache,logs,results} 目录（与其他实验
#   物理隔离，杜绝缓存串写）；结束后自动汇总到 docs/EXPERIMENTS.md 轮次表。
set -e
cd "$(dirname "$0")/.."
source env.sh

NAME="$1"; CFG="$2"; shift 2
[ -z "$NAME" ] && { echo "用法: $0 <exp_name> <config> [objects...]"; exit 1; }
OBJECTS="${@:-ape benchvise cam can cat driller duck eggbox glue holepuncher iron lamp phone}"

BASE="outputs/exp_${NAME}"
mkdir -p "$BASE"/{cache,logs,results}
echo "== 实验 $NAME ($CFG) 产物目录 $BASE"

echo "== 1/3 onboard（bank 指纹一致才跳过，不一致自动重训）"
python scripts/onboard_object.py --config "$CFG" --objects $OBJECTS \
    > "$BASE/logs/onboard.log" 2>&1 || { tail -20 "$BASE/logs/onboard.log"; exit 1; }
tail -3 "$BASE/logs/onboard.log"

echo "== 2/3 extract（匹配落盘，可复用）"
MATCHES="$BASE/matches"
if [ -d "$MATCHES" ] && [ -n "$(ls -A "$MATCHES" 2>/dev/null)" ]; then
  echo "  matches 已存在，跳过"
else
  python scripts/extract_matches.py --config "$CFG" --objects $OBJECTS \
      --matches-dir "$MATCHES" > "$BASE/logs/extract.log" 2>&1 \
      || { tail -20 "$BASE/logs/extract.log"; exit 1; }
fi
grep -c "帧完成" "$BASE/logs/extract.log" 2>/dev/null || true

echo "== 3/3 evaluate（cache 在实验目录内，自动隔离）"
i=0
for obj in $OBJECTS; do
  python scripts/run_linemod.py --config "$CFG" --objects $obj \
      --matches-dir "$MATCHES" --max-frames 120 \
      --cache-dir "$BASE/cache" \
      --out "$BASE/results/$obj.json" \
      > "$BASE/logs/eval_$obj.log" 2>&1 &
  i=$((i+1)); [ $i -ge 4 ] && wait -n
done
wait

echo "== 汇总"
python - "$BASE/results" "$NAME" "$CFG" <<'PYEOF'
import json, sys
from pathlib import Path
rdir = Path(sys.argv[1]); name = sys.argv[2]; cfg = sys.argv[3]
rows = []
for p in sorted(rdir.glob("*.json")):
    d = json.loads(p.read_text())
    po = d["per_object"]
    obj = next(iter(po))
    v = po[obj]
    rows.append((obj, v["add_01d"], v["proj_5px"], v["cm_deg"], v["n"]))
if not rows:
    print("无结果"); sys.exit(0)
n = len(rows)
mean = lambda i: sum(r[i] for r in rows) / n
print(f"\nexp_{name} ({cfg}): {n} objects, 120 帧/物体")
print(f"{'物体':<12} {'ADD':>7} {'Proj':>7} {'5cm5':>7}")
for obj, a, pj, c, _ in rows:
    print(f"{obj:<12} {a:>7.2f} {pj:>7.2f} {c:>7.2f}")
print(f"{'MEAN':<12} {mean(1):>7.2f} {mean(2):>7.2f} {mean(3):>7.2f}")

# 追加到 EXPERIMENTS.md 轮次表
md = Path("docs/EXPERIMENTS.md")
ts = __import__("datetime").datetime.now().strftime("%m-%d %H:%M")
lines = [f"\n### 轮次 {name}（{ts}，{cfg}）\n",
         f"| 物体 | ADD | Proj | 5cm5° |\n|---|---|---|---|\n"]
for obj, a, pj, c, _ in rows:
    lines.append(f"| {obj} | {a:.2f} | {pj:.2f} | {c:.2f} |\n")
lines.append(f"| **MEAN** | **{mean(1):.2f}** | **{mean(2):.2f}** | "
             f"**{mean(3):.2f}** |\n")
md.write_text(md.read_text() + "".join(lines))
print(f"已登记到 docs/EXPERIMENTS.md")
PYEOF
echo "== DONE exp_$NAME"
