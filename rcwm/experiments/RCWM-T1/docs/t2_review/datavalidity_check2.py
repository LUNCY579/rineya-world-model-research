"""T2 review, data-validity lens, part 2 (CPU only): non-discriminating rows per stratum and
operative-level overlap with the historical transition records (Titan's training/scoring payloads).

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/datavalidity_check2.py
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from rcwmt import env, bank as bankmod  # noqa: E402
sys.path.insert(0, str(env.FDS))
import generate as fdsgen  # type: ignore  # noqa: E402

def J(o): return json.dumps(o, separators=(",", ":"), sort_keys=True)
rows = {s: [json.loads(l) for l in (ROOT / "data" / f"t2_{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "eval")}

def ignore_rule(fam, b, e):
    """Answer of a model that ignores the legality / branch rule."""
    if fam == "ledger_replay":
        return env.step(fam, b, dict(e, status="posted"))
    if fam == "prerequisite_plan":
        return (b & ~e["delete"]) | e["add"]
    if e["kind"] == "conditional":   # wrong branch
        g = b[e["other"]] > e["threshold"]
        out = dict(b); out[e["register"]] = (b[e["register"]] + (-e["value"] if g else e["value"])) % 97
        return out
    if e["kind"] in ("add", "mix"):  # no modulo
        out = dict(b); out[e["register"]] = b[e["register"]] + e["value"] + (2 * b[e["other"]] if e["kind"] == "mix" else 0)
        return out
    return None

print("== A. rows where copy / rule-ignoring answer equals gold (non-discriminating for the targeted rule)")
for s, rr in rows.items():
    c = Counter(); n = Counter()
    for r in rr:
        k = (r["family"][:1].upper(), r["stratum"]); n[k] += 1
        gold = json.loads(r["target"]); b, e = r["before"], r["event"]
        ig = ignore_rule(r["family"], b, e)
        if gold == b: c[(k, "copy")] += 1
        if ig is not None and ig == gold: c[(k, "rule-ignored")] += 1
        if gold == b and ig is not None and ig == gold: c[(k, "copy AND rule-ignored")] += 1
    print(f"  {s}:")
    for k in sorted(n):
        print(f"    {k[0]} {k[1]:<18} n={n[k]:4d}  copy={c[(k,'copy')]:3d}  rule-ignored={c[(k,'rule-ignored')]:3d}  both={c[(k,'copy AND rule-ignored')]:3d}")

print("== B. operative-level overlap with historical records (harvest_exclusions -> content_keys)")
hist = bankmod.load_exclusions()
op, io = fdsgen.content_keys(hist)
print("  historical payloads", len(hist), "operative keys", len(op))
for s, rr in rows.items():
    hit = Counter(); hit_io = Counter()
    for r in rr:
        k = (r["family"][:1].upper(), r["stratum"])
        if (r["family"], fdsgen.operative_key(r["family"], {"before": r["before"], "event": r["event"]})) in op:
            hit[k] += 1
        if (r["family"], J(r["before"]), r["target"]) in io:
            hit_io[k] += 1
    print(f"  {s}: payload {sum(J({'before': r['before'], 'event': r['event']}) in hist for r in rr)}  "
          f"operative {sum(hit.values())} {dict(hit)}  (before,answer) {sum(hit_io.values())} {dict(hit_io)}")
