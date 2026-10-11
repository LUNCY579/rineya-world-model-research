"""Audit the blind-authored wording pool and split it into P2 (DEV, visible) and P3 (SEALED, secret). RCX-E01R, CX D-33.

    python -B experiments/RCX-E01R/split_pool.py --expect-sha <sha256 from the writer> --expect-fnv <fnv1a32 from the
        workflow>

Reads `pools/POOL_ALL.json.secret` (git-ignored). Prints counts only, never a P3 or spare wording. Drops, in order:
  1. vocab: a template containing any status-like word (rcx.rcl.wording_audit.status_words);
  2. ref_near_dup: near-duplicate (word Jaccard >= 0.5 or a shared 5-word sequence) of any RCX train/held-out template,
     composition frame or clause;
  3. pool_near_dup: within the pool, near-duplicate clusters (Jaccard >= 0.6 or a shared 6-word sequence, across all
     cells) keep only their smallest id, so that no frame can sit on both sides of the split.
Then, per (pol, form) cell, survivors are ordered by sha256(salt + id) with a fresh secret salt and dealt alternately to
P2 and P3; an odd remainder goes to SPARE (secret). Outputs are create-only:
  pools/P2_DEV.json (committed), pools/P3_SEALED.json.secret and pools/SPARE.json.secret (git-ignored),
  pools/POOL_SPLIT.json (committed: counts, drop reasons by cell, sha256 of the secret files, sha256(salt)).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx.rcl import wording_audit as WA  # noqa: E402

POOLS = Path(__file__).resolve().parent / "pools"
CELLS = [(p, f) for p in ("POS", "NEG") for f in ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")]


def fnv1a32(text):
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOLS / "POOL_ALL.json.secret"))
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--expect-fnv", required=True)
    a = ap.parse_args()

    outs = {k: POOLS / n for k, n in (("p2", "P2_DEV.json"), ("p3", "P3_SEALED.json.secret"),
                                       ("spare", "SPARE.json.secret"), ("meta", "POOL_SPLIT.json"))}
    existing = [str(p) for p in outs.values() if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing} (create-only)")

    raw = Path(a.pool).read_bytes()
    text = raw.decode("ascii")
    got_sha, got_fnv = sha(raw), fnv1a32(text)
    if got_sha != a.expect_sha.lower() or got_fnv != a.expect_fnv.lower():
        raise SystemExit(f"pool file mismatch: sha {got_sha[:12]}.. (want {a.expect_sha[:12]}..), fnv {got_fnv} "
                         f"(want {a.expect_fnv}); the writer did not transcribe the payload exactly")
    pool = json.loads(text)["templates"]
    ids = [t["id"] for t in pool]
    if len(set(ids)) != len(ids) or any(t["text"].count(WA.SLOT) != 1 for t in pool):
        raise SystemExit("pool ids not unique or a template without exactly one {L} slot")

    vocab, refs = WA.status_words(), WA.reference_wordings()
    drops, keep = {}, []
    for t in pool:
        if WA.vocab_hits(t["text"], vocab):
            drops[t["id"]] = "vocab"
        elif WA.ref_hits(t["text"], refs):
            drops[t["id"]] = "ref_near_dup"
        else:
            keep.append(t)
    groups = WA.clusters([(t["id"], t["text"]) for t in keep])
    dup_ids = {i for g in groups for i in g[1:]}
    for i in dup_ids:
        drops[i] = "pool_near_dup"
    survivors = [t for t in keep if t["id"] not in dup_ids]

    salt = os.urandom(16).hex()
    p2, p3, spare = [], [], []
    for pol, form in CELLS:
        cell = sorted((t for t in survivors if t["pol"] == pol and t["form"] == form),
                      key=lambda t: sha((salt + t["id"]).encode()))
        half = len(cell) // 2
        p2 += cell[0:2 * half:2]
        p3 += cell[1:2 * half:2]
        spare += cell[2 * half:]

    p2_text = json.dumps({"schema": "rcx-e01r-pool-v1", "pool": "P2_DEV", "templates": p2}, indent=1) + "\n"
    p3_text = json.dumps({"schema": "rcx-e01r-pool-v1", "pool": "P3_SEALED", "salt": salt, "templates": p3}) + "\n"
    spare_text = json.dumps({"schema": "rcx-e01r-pool-v1", "pool": "SPARE", "templates": spare}) + "\n"
    by_cell = {}
    for pol, form in CELLS:
        k = f"{pol}/{form}"
        in_cell = [t for t in pool if t["pol"] == pol and t["form"] == form]
        by_cell[k] = {"verified": len(in_cell),
                      "drops": dict(Counter(drops[t["id"]] for t in in_cell if t["id"] in drops)),
                      "P2": sum(1 for t in p2 if t["pol"] == pol and t["form"] == form),
                      "P3": sum(1 for t in p3 if t["pol"] == pol and t["form"] == form),
                      "spare": sum(1 for t in spare if t["pol"] == pol and t["form"] == form)}
    meta = {"experiment": "RCX-E01R", "decision": "CX D-33",
            "pool_all_sha256": got_sha, "pool_all_fnv1a32": got_fnv, "verified_templates": len(pool),
            "drops": dict(Counter(drops.values())), "by_cell": by_cell,
            "totals": {"P2": len(p2), "P3": len(p3), "spare": len(spare)},
            "p2_sha256": sha(p2_text.encode()), "p3_sha256": sha(p3_text.encode()),
            "spare_sha256": sha(spare_text.encode()), "salt_sha256": sha(salt.encode()),
            "thresholds": {"ref": {"jaccard": 0.5, "ngram": 5}, "pool": {"jaccard": 0.6, "ngram": 6}},
            "code_sha256": {"split_pool.py": sha(Path(__file__).read_bytes()),
                            "wording_audit.py": sha((ROOT / "rcx/rcl/wording_audit.py").read_bytes()),
                            "templates.py": sha((ROOT / "rcx/rcl/templates.py").read_bytes()),
                            "vocab.py": sha((ROOT / "rcx/rcl/vocab.py").read_bytes())}}
    write_new(outs["p3"], p3_text)
    write_new(outs["spare"], spare_text)
    write_new(outs["p2"], p2_text)
    write_new(outs["meta"], json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"drops": meta["drops"], "totals": meta["totals"],
                      "by_cell": {k: (v["P2"], v["P3"], v["spare"]) for k, v in by_cell.items()},
                      "p3_sha256": meta["p3_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
