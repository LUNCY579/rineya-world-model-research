"""LEAN: package strata of PlanBench Logistics t1 by id range (sizes only; no model)."""
import sys
from collections import Counter
import lgdom as L
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
T = L.planbench_t1()
for lo, hi in ((2, 200), (2, 201), (201, 286), (202, 286), (2, 286)):
    c = Counter(x["np"] for x in T if lo <= x["id"] <= hi)
    print(lo, hi, "n", sum(c.values()), sorted(c.items()), "| >=4:", sum(v for k, v in c.items() if k >= 4))
c = Counter((x["np"], x["nc"], x["nl"] // x["nc"], x["na"]) for x in T if x["np"] >= 4 and x["id"] <= 200)
print("OOR 2..200 (pk, cities, size, planes):", sorted(c.items()))
import statistics as st
for k in (4, 5, 6, 7):
    v = [len(x["gt"]) for x in T if x["np"] == k and x["id"] <= 200]
    w = [len(x["gt"]) for x in T if x["np"] == k and x["id"] > 200]
    print(k, "GT len 2..200", (min(v), round(st.mean(v), 1), max(v)) if v else None, "| 201..286", (min(w), round(st.mean(w), 1), max(w)) if w else None)
