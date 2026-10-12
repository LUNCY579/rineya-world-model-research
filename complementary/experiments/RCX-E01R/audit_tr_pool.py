"""RCX-E01R R1 training-pool audit (PREREG_E01R section 3, R1 "Audit, by code" and "P3 custody"; review item 6).

    python -B experiments/RCX-E01R/audit_tr_pool.py            (defaults: the real files below; every path is an option)
Reads:
  pools/TR_v1_candidates.json      the blind-authored training candidates (visible)
  pools/P2_DEV_v2.json             P2_v2, the DEV2 battery pool (sha256 = POOL_REFINE.json outputs.P2.sha256)
  pools/P2_DEV.json                P2 v1 (sha256 = POOL_SPLIT.json p2_sha256): its 4 extra wordings were visible too,
                                   and recipes.audit_pool refuses near-duplicates of them
  pools/P3_SEALED_v2.json.secret   read at run time only: its sha256 must equal POOL_REFINE.json outputs.P3.sha256 (and
                                   its count and per-cell counts the committed ones) before it is parsed
A candidate is dropped for the first reason that applies, in this order:
  status   a status-like word (wording_audit.vocab_hits), an inflection of a real one (stem_hits, CX D-34) or a
           multi-word EXCLUDED phrase
  ref      near-duplicate of an RCX wording (t*, h*, frames, clauses): Jaccard >= 0.5 or a shared 5-word sequence
  p2       near-duplicate of a P2_v2 wording under the stricter rule (Jaccard >= 0.35 or a shared 4-word sequence), or
           of a P2 v1 wording under the ref rule (as recipes.audit_pool)
  p3       the same stricter rule against P3_v2, tested only on candidates that pass every visible check
  cluster  among the survivors, near-duplicate clusters (Jaccard >= 0.6 or a shared 6-word sequence, across cells)
           keep their smallest id
Refused before anything is written (SystemExit): malformed or repeated candidate ids, ids starting with h (review item
14: an h* id reads as a held-out template), a P2 path naming P3/SEALED/.secret, a P3 path that is not a *.secret file,
any sha or count mismatch, an existing output.

P3 custody. Nothing prints or writes a P3 id, text or pairing. Errors about the P3 file never quote it. Outputs (create-
only, in --out-dir, default pools/; <tag> from the candidates file name, e.g. TR_v1):
  <tag>_audited.json         surviving candidates (visible; no P3 text, safe to commit)
  <tag>_P3_drops.json.secret candidates dropped for p3 (training wordings only; git-ignored)
  <tag>_AUDIT.json           counts per cell and per reason, the minimum check (>= 6 per cell), sha256 of inputs/outputs
The survivors are visible, so which candidates were dropped for p3 can be inferred by re-running the visible checks.
That is accepted: it reveals only that some P3 wording shares >= 35% of word types or a 4-word sequence with a training
wording, never which one.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.rcl import vocab as V, wording_audit as WA  # noqa: E402

POOLS = Path(__file__).resolve().parent / "pools"
FORMS = ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")
CELLS = [f"{p}/{f}" for p in ("POS", "NEG") for f in FORMS]
REF = {"jac": 0.5, "n": 5}       # D-34 audit against RCX wordings
STRICT = {"jac": 0.35, "n": 4}   # against P2 and P3 (CX D-37 review)
POOL = {"jac": 0.6, "n": 6}      # within-pool clusters
REASONS = ("status", "ref", "p2", "p3", "cluster")
MIN_PER_CELL = 6
EXCLUDED_PHRASES = sorted(w.lower() for w in V.EXCLUDED if " " in w)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check_entries(pool, what, quiet=False):
    """Refuse (ValueError) a list that is not of {id, text, pol, form} single-{L} one-line templates with unique ids.
    With quiet, the message names no id or text (for the sealed file)."""
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"{what}: no template list")
    seen = set()
    for k, t in enumerate(pool):
        name = f"{what} entry {k}" if quiet else f"{what} {t.get('id') if isinstance(t, dict) else t!r}"
        if not isinstance(t, dict) or not {"id", "text", "pol", "form"} <= set(t):
            raise ValueError(f"{name}: not a dict with id, text, pol, form")
        if not isinstance(t["id"], str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", t["id"]) or t["id"] in seen:
            raise ValueError(f"{name}: id malformed or repeated")
        seen.add(t["id"])
        text = t["text"]
        if t["pol"] not in ("POS", "NEG") or t["form"] not in FORMS or not isinstance(text, str) \
                or text.count("{L}") != 1 or "\n" in text or re.search(r"[{}]", text.replace("{L}", "")):
            raise ValueError(f"{name}: bad pol/form or not a single-{{L}} one-line template")


def cell(t):
    return f"{t['pol']}/{t['form']}"


def status_reasons(text, vocab=None, real=None):
    """Status words, inflections of real ones, and multi-word EXCLUDED phrases in a wording."""
    hits = WA.vocab_hits(text, vocab) + WA.stem_hits(text, vocab=real)
    seq = f" {' '.join(WA.words(text))} "
    return hits + [p for p in EXCLUDED_PHRASES if f" {p} " in seq]


def original_refs():
    """wording_audit.reference_wordings() restricted to the original RCX wordings (t*/h* templates and frames, clauses).
    It reads templates.SPOS/SNEG at call time, so after recipes.register_pool it would also hold the registered pool
    wordings (tr:*, p2:*), and a pool wording would match itself."""
    return {k: v for k, v in WA.reference_wordings().items()
            if k.startswith("CLAUSE:") or re.fullmatch(r"[th][0-9]+", k.split(":", 1)[1])}


def visible_drops(cands, p2_texts, p2_v1_texts=(), refs=None):
    """{id: first reason} for the visible checks: status, ref (RCX wordings), p2 (P2_v2 under the stricter rule, P2 v1
    under the ref rule)."""
    vocab, real = WA.status_words(), WA.real_status_words()
    refs = original_refs() if refs is None else refs
    out = {}
    for t in cands:
        if status_reasons(t["text"], vocab, real):
            out[t["id"]] = "status"
        elif WA.ref_hits(t["text"], refs, **REF):
            out[t["id"]] = "ref"
        elif any(WA.near_dup(t["text"], x, **STRICT) for x in p2_texts) \
                or any(WA.near_dup(t["text"], x, **REF) for x in p2_v1_texts):
            out[t["id"]] = "p2"
    return out


def audit(cands, p2_texts, p3_texts, p2_v1_texts=()):
    """-> (survivors in candidate order, {id: reason}). p3 is tested only on candidates passing the visible checks."""
    drops = visible_drops(cands, p2_texts, p2_v1_texts)
    for t in cands:
        if t["id"] not in drops and any(WA.near_dup(t["text"], x, **STRICT) for x in p3_texts):
            drops[t["id"]] = "p3"
    keep = [t for t in cands if t["id"] not in drops]
    for g in WA.clusters([(t["id"], t["text"]) for t in keep], POOL["jac"], POOL["n"]):
        drops.update({i: "cluster" for i in g[1:]})
    return [t for t in cands if t["id"] not in drops], drops


def counts_by_cell(cands, drops):
    out = {}
    for c in CELLS:
        ids = [t["id"] for t in cands if cell(t) == c]
        why = Counter(drops[i] for i in ids if i in drops)
        out[c] = {"candidates": len(ids), **{r: why[r] for r in REASONS}, "kept": len(ids) - sum(why.values())}
    return out


def read_p3(path, refine):
    """P3_v2 texts, only after the sha256, count and per-cell counts equal POOL_REFINE.json's. Messages quote
    nothing."""
    want = refine["outputs"]["P3"]
    raw = Path(path).read_bytes()
    if sha(raw) != want["sha256"]:
        raise SystemExit("P3 file sha256 differs from POOL_REFINE.json outputs.P3.sha256; it was not parsed")
    try:
        tpls = json.loads(raw)["templates"]
        check_entries(tpls, "P3", quiet=True)
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"P3 file malformed ({type(e).__name__}); no content shown") from None
    if len(tpls) != want["n"] or dict(Counter(map(cell, tpls))) != refine["kept_by_cell"]["P3"]:
        raise SystemExit("P3 file counts differ from POOL_REFINE.json; no content shown")
    return [t["text"] for t in tpls]


def read_visible(path, want_sha, what):
    p = Path(path)
    if re.search(r"P3|SEALED|\.secret", p.name, re.I):
        raise SystemExit(f"{what}: refusing a sealed-looking path as a visible reference: {p.name}")
    raw = p.read_bytes()
    if sha(raw) != want_sha:
        raise SystemExit(f"{what}: {p.name} sha256 differs from the committed record")
    tpls = json.loads(raw)["templates"]
    check_entries(tpls, what)
    return raw, tpls


def write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(POOLS / "TR_v1_candidates.json"))
    ap.add_argument("--p2", default=str(POOLS / "P2_DEV_v2.json"))
    ap.add_argument("--p2-v1", default=str(POOLS / "P2_DEV.json"))
    ap.add_argument("--p3", default=str(POOLS / "P3_SEALED_v2.json.secret"))
    ap.add_argument("--refine", default=str(POOLS / "POOL_REFINE.json"))
    ap.add_argument("--split", default=str(POOLS / "POOL_SPLIT.json"))
    ap.add_argument("--out-dir", default=str(POOLS))
    a = ap.parse_args(argv)
    cand_p = Path(a.candidates)
    m = re.fullmatch(r"(TR_v[0-9]+)_candidates\.json", cand_p.name)
    if not m:
        raise SystemExit(f"candidates file must be named TR_v<k>_candidates.json, got {cand_p.name}")
    tag, out = m.group(1), Path(a.out_dir)
    outs = {"audited": out / f"{tag}_audited.json", "p3_drops": out / f"{tag}_P3_drops.json.secret",
            "audit": out / f"{tag}_AUDIT.json"}
    if any(p.exists() for p in outs.values()):
        raise SystemExit(f"refusing to overwrite {[p.name for p in outs.values() if p.exists()]} (create-only)")
    if not Path(a.p3).name.endswith(".secret"):
        raise SystemExit("--p3 must be the git-ignored *.secret file")
    cand_raw = cand_p.read_bytes()
    cands = json.loads(cand_raw)["templates"]
    try:
        check_entries(cands, "candidate")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if any(t["id"].lower().startswith("h") for t in cands):
        raise SystemExit("candidate ids starting with h are refused (they would read as held-out templates)")
    refine = json.loads(Path(a.refine).read_text(encoding="utf-8"))
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    p2_raw, p2 = read_visible(a.p2, refine["outputs"]["P2"]["sha256"], "P2_v2")
    p1_raw, p1 = read_visible(a.p2_v1, split["p2_sha256"], "P2_v1")
    p3_texts = read_p3(a.p3, refine)
    survivors, drops = audit(cands, [t["text"] for t in p2], p3_texts, [t["text"] for t in p1])
    p3_drops = [t for t in cands if drops.get(t["id"]) == "p3"]
    by_cell = counts_by_cell(cands, drops)
    short = [c for c in CELLS if by_cell[c]["kept"] < MIN_PER_CELL]
    texts = {"audited": json.dumps({"schema": "rcx-e01r-pool-v1", "pool": f"{tag}_audited",
                                    "audited_from": {"file": cand_p.name, "sha256": sha(cand_raw)},
                                    "audit": outs["audit"].name, "templates": survivors}, indent=1) + "\n",
             "p3_drops": json.dumps({"schema": "rcx-e01r-pool-v1", "pool": f"{tag}_P3_drops",
                                     "note": "training candidates dropped as near-duplicates of a P3_v2 wording; "
                                             "nothing about P3 itself", "templates": p3_drops}) + "\n"}
    meta = {"experiment": "RCX-E01R", "spec": "PREREG_E01R section 3, R1 audit (CX D-37)",
            "thresholds": {"ref_and_p2_v1": REF, "p2_v2_and_p3": STRICT, "pool_cluster": POOL},
            "reasons_in_order": list(REASONS),
            "inputs": {"candidates": {"file": cand_p.name, "sha256": sha(cand_raw), "n": len(cands)},
                       "P2_v2": {"file": Path(a.p2).name, "sha256": sha(p2_raw), "n": len(p2)},
                       "P2_v1": {"file": Path(a.p2_v1).name, "sha256": sha(p1_raw), "n": len(p1)},
                       "P3_v2": {"file": Path(a.p3).name, "sha256": refine["outputs"]["P3"]["sha256"],
                                 "n": len(p3_texts), "verified": "sha256, count and per-cell counts"}},
            "by_cell": by_cell, "totals": {r: sum(v[r] for v in by_cell.values()) for r in
                                           ("candidates", *REASONS, "kept")},
            "min_per_cell": MIN_PER_CELL, "meets_minimum": not short, "cells_below_minimum": short,
            "outputs": {k: {"file": outs[k].name, "sha256": sha(texts[k].encode()),
                            "n": len(survivors) if k == "audited" else len(p3_drops)} for k in texts},
            "code_sha256": {p.name: sha(p.read_bytes()) for p in (Path(__file__), ROOT / "rcx/rcl/wording_audit.py",
                                                                   ROOT / "rcx/rcl/templates.py",
                                                                   ROOT / "rcx/rcl/vocab.py")}}
    out.mkdir(parents=True, exist_ok=True)
    write_new(outs["p3_drops"], texts["p3_drops"])
    write_new(outs["audited"], texts["audited"])
    write_new(outs["audit"], json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"by_cell": {c: [v["kept"], {r: v[r] for r in REASONS if v[r]}] for c, v in by_cell.items()},
                      "totals": meta["totals"], "meets_minimum": meta["meets_minimum"],
                      "cells_below_minimum": short}, indent=1))
    return meta


if __name__ == "__main__":
    main()
