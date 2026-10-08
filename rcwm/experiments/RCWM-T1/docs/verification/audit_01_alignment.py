"""Audit 1: index.json vs bank item order, split integrity, '-one' mapping, zero rows, layer identity."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, data, env

bank = data.load_bank("bank_pilot")
items = bankmod.items(bank)
for arm in ("sparse", "supa"):
    idx = json.loads((ROOT / "runs/bank_pilot" / arm / "index.json").read_text(encoding="utf-8"))
    assert idx["bank_sha256"] == hashlib.sha256((ROOT / "data/bank_pilot.json").read_bytes()).hexdigest(), "bank sha mismatch"
    assert len(idx["items"]) == len(items)
    bad = [i for i, (a, b) in enumerate(zip(idx["items"], items)) if (a["item_id"], a["split"], a["k"], a["format"], a["parent_id"]) != (b["item_id"], b["split"], b["k"], b["format"], b["parent_id"])]
    print(arm, "index vs bank.items mismatches:", len(bad), "layers", idx["layers"])
    ev = [i for i, it in enumerate(items) if it["format"] == "multi" and it["k"] == bank["horizon"]]
    print(arm, "event_rows match:", ev == idx["event_rows"], len(ev))
    # all ids unique
    ids = [it["item_id"] for it in idx["items"]]
    print(arm, "unique ids:", len(set(ids)) == len(ids))

# candidate_table vs bank.items consistency (targets): same gold state for every (cand,k)
rows = data.candidate_table(bank)
by_id = {it["item_id"]: it for it in items}
mm = 0
for r in rows:
    it = by_id[r["item_id"]]
    if it["gold"] != r["gold"] or it["split"] != r["split"] or it["features"] != r["features"]:
        mm += 1
    one = by_id[r["cand_id"] + "-one"]
    if r["k"] == 1 and (one["gold"] != r["gold"] or env.dumps(env.rollout(r["family"], r["s0"], r["events"])[-1]) != one["gold"]):
        mm += 1
    # the prompt actually contains these events
    u = json.loads(it["messages"][1]["content"])
    if u["events"] != r["events"] or u["before"] != r["s0"]:
        mm += 1
    if r["k"] == 1:
        u1 = json.loads(one["messages"][1]["content"])
        if u1["event"] != r["events"][0] or u1["before"] != r["s0"]:
            mm += 1
print("candidate_table vs prompt/gold mismatches:", mm, "rows", len(rows))

# split integrity
ps = {}
for p in bank["parents"]:
    ps.setdefault(p["parent_id"], set()).add(p["split"])
print("parents with >1 split:", sum(len(v) > 1 for v in ps.values()), "n parents", len(ps))
from collections import Counter
print(Counter((p["family"], p["split"]) for p in bank["parents"]))
# overlap of identical (s0, first event) across splits (near-duplicate leakage for one-step items)
key = {}
for p in bank["parents"]:
    for c in p["candidates"]:
        key.setdefault(env.dumps({"f": p["family"], "s0": p["s0"], "e": c["events"][0]}), set()).add(p["split"])
print("one-step (s0,e1) keys spanning >1 split:", sum(len(v) > 1 for v in key.values()))
key = {}
for p in bank["parents"]:
    key.setdefault(env.dumps({"f": p["family"], "s0": p["s0"]}), set()).add(p["split"])
print("s0 values spanning >1 split:", sum(len(v) > 1 for v in key.values()), "of", len(key))

# zero rows / layer identity / non-finite
A = np.load(ROOT / "runs/bank_pilot/sparse/last.npy", mmap_mode="r")
B = np.load(ROOT / "runs/bank_pilot/supa/last.npy", mmap_mode="r")
EA = np.load(ROOT / "runs/bank_pilot/sparse/events.npy", mmap_mode="r")
EB = np.load(ROOT / "runs/bank_pilot/supa/events.npy", mmap_mode="r")
print("shapes", A.shape, B.shape, EA.shape, EB.shape, A.dtype)
CH = 2000
zA = zB = 0; nfA = nfB = 0
ident = np.zeros(8, bool) + True
maxdiff = np.zeros(8)
for s in range(0, A.shape[0], CH):
    a = np.asarray(A[s:s+CH]).astype(np.float32); b = np.asarray(B[s:s+CH]).astype(np.float32)
    zA += int((np.abs(a).sum((1, 2)) == 0).sum()); zB += int((np.abs(b).sum((1, 2)) == 0).sum())
    nfA += int((~np.isfinite(a)).sum()); nfB += int((~np.isfinite(b)).sum())
    for li in range(8):
        d = np.abs(a[:, li] - b[:, li]).max()
        maxdiff[li] = max(maxdiff[li], d)
print("all-zero last rows sparse/supa:", zA, zB, "nonfinite", nfA, nfB)
print("max |sparse-supa| per layer (0,4,...,28):", maxdiff.round(4).tolist())
zE = 0
for s in range(0, EA.shape[0], 1000):
    a = np.asarray(EA[s:s+1000]).astype(np.float32); b = np.asarray(EB[s:s+1000]).astype(np.float32)
    zE += int((np.abs(a).sum((2, 3)) == 0).sum()) + int((np.abs(b).sum((2, 3)) == 0).sum())
print("all-zero event (row,j) slots across both arms:", zE)
# per-row near-duplicates between arms at layer 28: does row i in sparse best-match row i in supa?
rng = np.random.default_rng(0)
samp = rng.choice(A.shape[0], 300, replace=False)
for li, L in ((3, 12), (6, 24)):
    a = np.asarray(A[np.sort(samp), li]).astype(np.float32); b = np.asarray(B[np.sort(samp), li]).astype(np.float32)
    a = a - a.mean(0); b = b - b.mean(0)
    sim = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
    print(f"L{L}: sparse row i nearest supa row is i (centered cosine) for {np.mean(sim.argmax(1) == np.arange(len(samp))):.3f} of 300 sampled rows")
