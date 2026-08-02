#!/usr/bin/env python
"""init(refine 关) vs refined(refine 开) 逐帧误差对比：定位深度误差来源。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics.pose_metrics import aggregate  # noqa: E402


def load_cache(path):
    seen = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen[int(r["frame_id"])] = r
    return seen


def main():
    ref = load_cache("outputs/cache_ape_full/ape.jsonl")
    init = load_cache("outputs/cache_ape_full_init/ape.jsonl")
    common = sorted(set(ref) & set(init))
    print(f"common frames: {len(common)}")
    if not common:
        return

    def col(cache, key, fid):
        return cache[fid]["m"].get(key, np.nan)

    rt = np.array([ref[f]["m"]["rot_err"] for f in common])
    it = np.array([init[f]["m"]["rot_err"] for f in common])
    rtr = np.array([ref[f]["m"]["trans_err"] for f in common])
    itr = np.array([init[f]["m"]["trans_err"] for f in common])

    print(f"{'':<18}{'init(no refine)':>18}{'refined':>18}")
    print(f"{'rot_err med':<18}{np.median(it):>17.1f}°{np.median(rt):>17.1f}°")
    print(f"{'rot<5deg':<18}{100*np.mean(it<5):>17.1f}%{100*np.mean(rt<5):>17.1f}%")
    print(f"{'trans med':<18}{np.median(itr):>17.1f}mm{np.median(rtr):>17.1f}mm")
    print(f"{'trans<10mm':<18}{100*np.mean(itr<10):>17.1f}%{100*np.mean(rtr<10):>17.1f}%")
    print(f"{'trans<25mm':<18}{100*np.mean(itr<25):>17.1f}%{100*np.mean(rtr<25):>17.1f}%")
    print(f"{'trans<50mm':<18}{100*np.mean(itr<50):>17.1f}%{100*np.mean(rtr<50):>17.1f}%")
    # refine 是否改善
    imp = rtr < itr - 5
    hurt = rtr > itr + 5
    print(f"\nrefine 改善 trans>5mm: {100*np.mean(imp):.1f}%  变差>5mm: {100*np.mean(hurt):.1f}%")
    # 相关
    print(f"corr(trans init, trans refined): {np.corrcoef(itr, rtr)[0,1]:.2f}")
    # 分桶：init 好的帧 refine 后如何
    good = itr < 10
    bad = itr >= 40
    print(f"\ninit trans<10mm 的帧({np.sum(good)}): refine 后 trans med "
          f"{np.median(rtr[good]):.1f}mm, <10mm 保持率 {100*np.mean(rtr[good]<10):.1f}%")
    print(f"init trans>=40mm 的帧({np.sum(bad)}): refine 后 trans med "
          f"{np.median(rtr[bad]):.1f}mm, <10mm 率 {100*np.mean(rtr[bad]<10):.1f}%")
    # 聚合指标对比
    for name, cache in (("init", init), ("refined", ref)):
        ms = [cache[f]["m"] for f in common]
        a = aggregate(ms)
        print(f"{name}: ADD@0.1d={a['add_01d']:.2f}% Proj@5px={a['proj_5px']:.2f}% "
              f"5cm5°={a['cm_deg']:.2f}%")


if __name__ == "__main__":
    main()
