"""Merge sliced eval parts episodes_{arm}_{split}__{lo}-{hi}.jsonl into episodes_{arm}_{split}.jsonl + summary.

Usage: python scripts/merge_parts.py --dir runs/EVAL_pilot_s1 --arm A10 --split id_val --parts 0-50,50-100,100-150,150-200
Refuses to merge unless every listed part exists (no silent partial results).
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import planner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--parts", required=True)
a = ap.parse_args()
d = ROOT / a.dir
eps = []
for p in a.parts.split(","):
    f = d / f"episodes_{a.arm}_{a.split}__{p}.jsonl"
    if not f.exists():
        raise SystemExit(f"missing part {f}")
    eps += [json.loads(l) for l in open(f, encoding="utf-8")]
with open(d / f"episodes_{a.arm}_{a.split}.jsonl", "w", encoding="utf-8") as fh:
    for e in eps:
        fh.write(json.dumps(e) + "\n")
sp = d / "summary.json"
S = json.loads(sp.read_text()) if sp.exists() else {"summary": {}}
sm = planner.summarize(eps)
sm["merged_from_parts"] = a.parts
S["summary"][f"{a.arm}/{a.split}"] = sm
sp.write_text(json.dumps(S, indent=1))
print(f"merged {len(eps)} episodes: {a.arm}/{a.split} success {sm['success']:.3f}")
