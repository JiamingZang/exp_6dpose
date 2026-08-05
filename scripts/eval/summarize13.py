#!/usr/bin/env python
"""汇总 13 物体子集评估结果 → summary.json + 终端表格。"""
import json
from pathlib import Path

OUT = Path("outputs/linemod13_ds")
OBJS = ["ape", "benchvise", "cam", "can", "cat", "driller", "duck",
        "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"]

rows = []
tot = {"n": 0, "add": 0, "proj": 0, "cm": 0}
for obj in OBJS:
    p = OUT / f"{obj}.json"
    if not p.exists():
        rows.append({"obj": obj, "missing": True})
        continue
    po = json.load(open(p))["per_object"][obj]
    r = {"obj": obj, "n": po["n"],
         "add": po["add_01d"], "proj": po["proj_5px"], "cm": po["cm_deg"]}
    rows.append(r)
    tot["n"] += r["n"]
    tot["add"] += r["add"] * r["n"]
    tot["proj"] += r["proj"] * r["n"]
    tot["cm"] += r["cm"] * r["n"]

print(f"{'obj':12s} {'n':>5s} {'ADD':>7s} {'Proj':>7s} {'5cm5d':>7s}")
for r in rows:
    if r.get("missing"):
        print(f"{r['obj']:12s}  MISSING")
        continue
    print(f"{r['obj']:12s} {r['n']:5d} {r['add']:6.2f}% {r['proj']:6.2f}% "
          f"{r['cm']:6.2f}%")
print("-" * 40)
if tot["n"]:
    print(f"{'MEAN':12s} {tot['n']:5d} {tot['add']/tot['n']:6.2f}% "
          f"{tot['proj']/tot['n']:6.2f}% {tot['cm']/tot['n']:6.2f}%")
    json.dump({"rows": rows, "mean": {k: v / tot["n"] for k, v in
                                      tot.items() if k != "n"}, "n": tot["n"]},
              open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("已写", OUT / "summary.json")
