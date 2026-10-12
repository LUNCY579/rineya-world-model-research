"""RCX-E01R step B training-data recipes (DESIGN.md section 4): R2 complement-focused twins, R1 complement coverage.

Both return records in gen.make_train's schema (every record comes from gen.make_record; arith world, TRAIN split), so
native_e01.native_rows-style code (`native_rows` below) turns them into prompt/target rows. Matched to N-CFSD: N_ROWS
rows by default, the CF_SD regime's rule/status/word distributions and its surface factors; seeds are arguments (the
prereg fixes them; SEEDS holds the suggested values).
  R2  make_r2(n_rows, seed): CF_SD parents (base, rule_flip, status_swap, set_swap). A parent whose rule is a single S+
      atom also gets k in COMP_K pairs, one per training wording (k distinct tids of t0-t3): `comp_unlisted` (a fresh
      status outside the set, TRAIN words or nonce at the CF_SD mix: no-op) and `comp_listed` (a status in the set:
      apply). A pair shares state, event and surface; only the status differs. Whole parents only: the first parent
      that would overflow n_rows is dropped and the remainder is filled with base-only parents (`fill_base`, 1 row
      each).
  R1  make_r1(n_rows, seed, train_pool): gen.make_train("CF_SD") records; for single-atom S+/S- rules the wording is
      replaced with probability p_pool by a training-pool template of the same polarity (form uniform, then template),
      registered at runtime as templates.SPOS/SNEG['tr:<id>'] like run_e01r.load_p2 (nothing in rcx/rcl changes on
      disk). Refuses the whole pool if a template near-duplicates a P2 DEV wording (v1 or v2) or a spent h* template, or
      has a status-like word or an inflection of one (audit_pool). P3 is sealed: its audit is a separate blind step
      (audit_pool takes extra refs).
Pure Python (no torch/transformers). Summary: python -B experiments/RCX-E01R/recipes.py R2 [--seed s] [--n rows]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx.rcl import gen, templates as T, wording_audit as WA, world as W  # noqa: E402

HERE = Path(__file__).resolve().parent
P2_PATHS = (HERE / "pools" / "P2_DEV.json", HERE / "pools" / "P2_DEV_v2.json")  # every DEV wording ever visible
N_ROWS = 1501  # = native_e01.N_ROWS (that module imports torch; the tests check the literal)
SEEDS = {"R1": 20261263, "R2": 20261264}  # suggested defaults; the prereg fixes the seeds
BASE = "CF_SD"
COMP_K = (2, 3)  # training wordings (= complement pairs) per single-S+ parent
FORMS = ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")
POL_OP = {"POS": "S+", "NEG": "S-"}


# ----------------------------------------------------------------------------- R2: complement-focused twins

def _twin(rule, core, status, surf, rng, meta):
    core_v = dict(core)
    core_v["event"] = gen._set_status(core["event"], status)
    return gen.make_record(rule, core_v, surf, rng, meta)


def _r2_parent(rng, ws, seed, pid, base_only=False):
    """Records of one R2 parent: the CF_SD twins, plus complement pairs if the rule is a single S+ atom."""
    core, variants = gen.train_parent_variants(BASE, rng, ws)
    surf_rng = random.Random(rng.randrange(2 ** 31))
    x_rng = random.Random(rng.randrange(2 ** 31))

    def meta(v):
        return {"parent_id": f"tr-R2-{seed}-{pid}", "split": "train", "group": "R2", "variant": v}

    rule, status = variants[0][:2]
    if base_only:
        return [_twin(rule, core, status, gen.surface_spec(surf_rng, "sd"), surf_rng, meta("fill_base"))]
    recs = [_twin(r, core, s, gen.surface_spec(surf_rng, "sd"), surf_rng, meta(v)) for r, s, v in variants]
    if rule["op"] == "S+":
        k = x_rng.choice(COMP_K)
        used = set(rule["set"]) | {s for _, s, _ in variants}
        for tid, out in zip(x_rng.sample(T.TRAIN_IDS, k), ws.draw(x_rng, k, exclude=used)):
            surf = dict(gen.surface_spec(surf_rng, "sd"), tid=tid)
            pair = surf_rng.randrange(2 ** 31)  # same state order for both members
            for st, v in ((out, "comp_unlisted"), (x_rng.choice(rule["set"]), "comp_listed")):
                recs.append(_twin(rule, core, st, surf, random.Random(pair), meta(v)))
    return recs


def make_r2(n_rows=N_ROWS, seed=SEEDS["R2"]):
    """R2 records, exactly n_rows, whole parents only (see module doc)."""
    gen.use_world("arith")
    rng = random.Random(seed)
    ws = gen.regime_wordsource(BASE)
    recs, pid = [], 0
    while True:
        parent = _r2_parent(rng, ws, seed, pid)
        if len(recs) + len(parent) > n_rows:
            break
        recs += parent
        pid += 1
    while len(recs) < n_rows:
        recs += _r2_parent(rng, ws, seed, pid, base_only=True)
        pid += 1
    return recs


# ----------------------------------------------------------------------------- R1: training wording pool

def check_pool(pool, prefix="tr"):
    """Validate a whole pool for register_pool without registering anything; returns [(dict, tid, text)]. Refuses an
    entry that is not {id, text, pol, form}, a template that is not a single-{L} one-line format string, or an id that
    is malformed, repeated in the pool, or already registered with another text or polarity."""
    if not re.fullmatch(r"[a-z][a-z0-9]*", prefix) or prefix in ("p2", "p3"):
        raise ValueError(f"bad prefix {prefix!r} (p2/p3 are the test-pool namespaces)")
    todo, seen = [], set()
    for t in pool:
        if not isinstance(t, dict) or not {"id", "text", "pol", "form"} <= set(t):
            raise ValueError(f"pool entry is not a dict with id, text, pol, form: {t!r}")
        if not isinstance(t["id"], str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", t["id"]) or t["id"] in seen:
            raise ValueError(f"template id {t['id']!r} is malformed or repeated")
        seen.add(t["id"])
        if t["pol"] not in POL_OP or t["form"] not in FORMS:
            raise ValueError(f"{t['id']}: pol/form {t['pol']!r}/{t['form']!r} not in {tuple(POL_OP)} x {FORMS}")
        text, tid = t["text"], f"{prefix}:{t['id']}"
        if not isinstance(text, str) or text.count("{L}") != 1 or "\n" in text:
            raise ValueError(f"{tid}: not a single-{{L}} one-line format string")
        body = text.replace("{L}", "")
        if "{" in body or "}" in body or not body.strip():
            raise ValueError(f"{tid}: not a single-{{L}} one-line format string")
        own, other = (T.SPOS, T.SNEG) if t["pol"] == "POS" else (T.SNEG, T.SPOS)
        if tid in other or own.get(tid, text) != text:
            raise ValueError(f"{tid}: id collides with a registered template")
        todo.append((own, tid, text))
    return todo


def register_pool(pool, prefix="tr"):
    """Register pool templates ({id, text, pol, form}) as templates.SPOS/SNEG[f'{prefix}:{id}'] (cf. run_e01r.load_p2),
    all or nothing after check_pool. Re-registering an identical template is a no-op. Returns {id: tid}."""
    todo = check_pool(pool, prefix)
    for own, tid, text in todo:
        own[tid] = text
    return {tid.split(":", 1)[1]: tid for _, tid, _ in todo}


def unregister_pool(prefix="tr"):
    """Remove every '<prefix>:*' template from SPOS/SNEG; returns the number removed."""
    keys = [(d, k) for d in (T.SPOS, T.SNEG) for k in d if k.startswith(prefix + ":")]
    for d, k in keys:
        del d[k]
    return len(keys)


def p2_refs(paths=P2_PATHS):
    """{file:id: text} for the P2 DEV pools (read-only)."""
    refs = {}
    for p in map(Path, paths):
        for t in json.loads(p.read_text(encoding="utf-8"))["templates"]:
            refs[f"{p.stem}:{t['id']}"] = t["text"]
    return refs


def heldout_refs():
    """The spent held-out RCX wordings: h* rule templates and composition frames."""
    return {f"{n}:{k}": v for n, d in (("SPOS", T.SPOS), ("SNEG", T.SNEG), ("AND", T.AND_FRAMES), ("OR", T.OR_FRAMES))
            for k, v in d.items() if re.fullmatch(r"h[0-9]+", k)}


def audit_pool(pool, extra_refs=None, stems=True):
    """{id: [reasons]} for pool templates that must not be trained on (empty = clean): near-duplicate of a P2 DEV
    wording (v1 or v2), a spent h* template or an `extra_refs` wording (wording_audit.ref_hits: Jaccard >= 0.5 or a
    shared 5-word sequence); a status-like word; with `stems`, an inflection of a real one (stem_hits, CX D-34)."""
    refs = {**p2_refs(), **heldout_refs(), **(extra_refs or {})}
    if not any(k.startswith("P2_DEV_v2:") for k in refs):
        raise ValueError("P2_DEV_v2 wordings missing from the audit references")
    vocab, real = WA.status_words(), WA.real_status_words()
    bad = {}
    for t in pool:
        why = [f"near_dup:{k}" for k in WA.ref_hits(t["text"], refs)]
        why += [f"status_word:{w}" for w in WA.vocab_hits(t["text"], vocab)]
        why += [f"status_stem:{h}" for h in WA.stem_hits(t["text"], vocab=real)] if stems else []
        if why:
            bad[t["id"]] = why
    return bad


def make_r1(n_rows=N_ROWS, seed=SEEDS["R1"], train_pool=None, p_pool=0.5, prefix="tr", extra_refs=None):
    """R1 records: gen.make_train('CF_SD', n_rows, seed) with single-atom S+/S- wordings drawn from `train_pool` with
    probability p_pool (polarity-matched; form uniform among the pool's forms, then template uniform). Refuses the
    pool (ValueError, nothing registered) if check_pool or audit_pool flags any template."""
    if not train_pool:
        raise ValueError("R1 needs a non-empty training wording pool")
    if not 0.0 <= p_pool <= 1.0:
        raise ValueError(f"p_pool {p_pool}")
    check_pool(train_pool, prefix)
    bad = audit_pool(train_pool, extra_refs)
    if bad:
        raise ValueError(f"training pool refused ({len(bad)} templates): {bad}")
    register_pool(train_pool, prefix)
    by = {}
    for t in sorted(train_pool, key=lambda t: t["id"]):
        by.setdefault(POL_OP[t["pol"]], {}).setdefault(t["form"], []).append(f"{prefix}:{t['id']}")
    gen.use_world("arith")
    recs = gen.make_train(BASE, n_rows, seed)
    rng = random.Random(f"{seed}-R1-wording")
    for r in recs:
        r["group"], r["parent_id"] = "R1", r["parent_id"].replace(f"tr-{BASE}-", "tr-R1-", 1)
        forms = by.get(r["rule_ast"]["op"])
        if forms and rng.random() < p_pool:
            r["tid"] = rng.choice(forms[rng.choice(sorted(forms))])
            r["input"] = W.render_input(r["rule_ast"], r["state"], r["event"], r["tid"], r["state_order"],
                                        list(r["event"]))
    return recs


# ----------------------------------------------------------------------------- stats and native rows

def recipe_stats(records):
    """Descriptive composition of a record set (JSON-able)."""
    n = len(records)
    spos = [r for r in records if r["rule_ast"]["op"] == "S+"]
    comp = sum(r["gold"]["listed"] is False for r in spos)
    pos_noop = [r for r in records if (W.status_atom(r["rule_ast"]) or {}).get("op") == "S+"
                and r["gold"]["decision"] == "noop"]  # rules with an S+ atom (single or composed), no-op rows
    pos_noop_unl = sum(r["gold"]["listed"] is False for r in pos_noop)
    rule_dec = Counter((r["rule_ast"]["op"], r["gold"]["decision"]) for r in records)
    return {"rows": n, "parents": len({r["parent_id"] for r in records}),
            "noop_share": round(sum(r["gold"]["decision"] == "noop" for r in records) / max(1, n), 4),
            "spos_rows": len(spos), "spos_unlisted_rows": comp,  # single S+ atom, status outside the set: no-op
            "spos_unlisted_share": round(comp / max(1, n), 4),
            "spos_unlisted_frac_of_spos": round(comp / max(1, len(spos)), 4),
            "s_pos_noop_unlisted_frac": round(pos_noop_unl / max(1, len(pos_noop)), 4),
            "rule_decisions": {f"{op}|{d}": c for (op, d), c in sorted(rule_dec.items())},
            "tids": dict(sorted(Counter(r["tid"] for r in records).items())),
            "tid_family": dict(sorted(Counter(r["tid"].split(":")[0] if ":" in r["tid"] else "t*"
                                              for r in records).items())),
            "variants": dict(sorted(Counter(r["variant"] for r in records).items()))}


def target_json(after):
    return json.dumps({k: after[k] for k in sorted(after)}, separators=(",", ":"))  # = native_e01.target_json


def native_rows(recs, tok, regime, np_mod=None):
    """native_e01.native_rows for given records (same row keys). np_mod defaults to native_probe (imports torch, so
    build time only); tests pass its rcl_system/user_json. R1 records render only after register_pool in this
    process."""
    if np_mod is None:
        sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01N"))
        import native_probe as np_mod
    rows = []
    for r in recs:
        system = np_mod.rcl_system(T.render_rule(r["rule_ast"], r["tid"]))
        prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                          {"role": "user", "content": np_mod.user_json(r)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt, "target": target_json(r["gold"]["after"]), "regime": regime,
                     "parent_id": r["parent_id"], "variant": r["variant"], "decision": r["gold"]["decision"],
                     "semantic_hash": r["semantic_hash"], "skeleton_hash": r["skeleton_hash"],
                     "status": r["event"]["status"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", choices=["R1", "R2"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n", type=int, default=N_ROWS)
    ap.add_argument("--pool", help="R1: JSON file with a 'templates' list (training pool; never a sealed file)")
    a = ap.parse_args()
    seed = SEEDS[a.recipe] if a.seed is None else a.seed
    if a.recipe == "R2":
        recs = make_r2(a.n, seed)
    else:
        if not a.pool or a.pool.endswith(".secret"):
            raise SystemExit("R1 needs --pool <training pool json> (sealed files are refused)")
        recs = make_r1(a.n, seed, json.loads(Path(a.pool).read_text(encoding="utf-8"))["templates"])
    print(json.dumps(dict(recipe_stats(recs), seed=seed), indent=1))


if __name__ == "__main__":
    main()
