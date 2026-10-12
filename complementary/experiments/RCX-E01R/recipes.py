"""RCX-E01R step B training-data recipes (DESIGN.md section 4; FROZEN specs in PREREG_E01R_DRAFT.md section 3, CX D-37):
R2 complement-focused twins, R1 complement coverage.

Both return records in gen.make_train's schema (every record comes from gen.make_record; arith world, TRAIN split), so
`native_rows` below turns them into native_e01-style prompt/target rows. Both have N_ROWS rows (= N-CFSD's) by default;
the default seeds are the prereg's (SEEDS).
  R2  make_r2(n_rows, seed=20261264): parents drawn as gen.make_train("CF_SD")'s (base, rule_flip, status_swap,
      set_swap; TRAIN words, 50% nonce). A parent whose rule is a single status atom also gets k in COMP_K pairs, each
      under a distinct training wording (k distinct tids of t0-t3): `comp_unlisted` (a fresh status outside the set and
      the parent's statuses) and `comp_listed` (a status in the set): no-op/apply under S+, apply/no-op under S-
      (mirrored), so "unlisted" alone never predicts the target. A pair shares state, event and surface; only the
      status differs. For a size-1 set, a pair whose comp_listed twin duplicates the parent's base or status_swap record
      (same wording and semantic hash) is redrawn under an unused training wording (otherwise the pair is dropped).
      Whole parents only: the first parent that would overflow n_rows is dropped and the remainder is filled with
      base-only parents (`fill_base`, 1 row each). Only the row count is matched to N-CFSD: the pairs shift the rows
      toward single-atom rules and R2 has fewer parents (a declared confound; recipe_stats reports both).
  R1  make_r1(n_rows, seed=20261253, train_pool): gen.make_train("CF_SD", n_rows, seed) records (N-CFSD's data seed:
      parents, states, events, targets and order are N-CFSD's); each single-atom S+/S- row's wording is replaced with
      probability p_pool by a pool template of the same polarity (form uniform, then template), registered at runtime
      as templates.SPOS/SNEG['tr:<id>'] like run_e01r.load_p2 (nothing in rcx/rcl changes on disk). make_r1 takes the
      audited pool only: it refuses (nothing registered) a pool that check_pool or audit_pool flags, that holds a
      within-pool near-duplicate cluster, or that has fewer than MIN_PER_CELL templates in any of the 6 cells.
      select_pool applies the code audit (status words, multi-word status phrases and stems; RCX and P2 DEV
      near-duplicates; then one member per within-pool cluster). The stricter P2_v2/P3_v2 audit is a separate
      counts-only routine: P3 never enters audit_pool, which refuses P3/SEALED-named refs and reports `extra_refs` hits
      only as 'near_dup:blind'.
Pure Python (no torch/transformers). Summary: python -B experiments/RCX-E01R/recipes.py R2|R1 [--seed s] [--n rows]
[--pool <training pool json>]
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

from rcx.rcl import gen, templates as T, vocab as V, wording_audit as WA, world as W  # noqa: E402

HERE = Path(__file__).resolve().parent
P2_PATHS = (HERE / "pools" / "P2_DEV.json", HERE / "pools" / "P2_DEV_v2.json")  # every DEV wording ever visible
N_ROWS = 1501  # = native_e01.N_ROWS (that module imports torch; the tests check the literal)
SEEDS = {"R1": 20261253, "R2": 20261264}  # PREREG_E01R section 3/9 (frozen); R1 = N-CFSD's data seed
BASE = "CF_SD"
COMP_K = (2, 3)  # training wordings (= complement pairs) per single-atom parent
FORMS = ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")
POL_OP = {"POS": "S+", "NEG": "S-"}
CELLS = tuple((p, f) for p in POL_OP for f in FORMS)
MIN_PER_CELL = 6  # frozen R1 minimum per (polarity, form) cell
BLIND = re.compile(r"p3|sealed|secret", re.I)  # sealed material: never an audit_pool reference key or file
BLIND_HIT = "near_dup:blind"


# ----------------------------------------------------------------------------- R2: complement-focused twins

def _twin(rule, core, status, surf, rng, meta):
    core_v = dict(core)
    core_v["event"] = gen._set_status(core["event"], status)
    return gen.make_record(rule, core_v, surf, rng, meta)


def _dup_key(rec):
    """What makes two records of a parent duplicates: the same wording on the same rule, state and event."""
    return rec["tid"], rec["semantic_hash"]


def _r2_parent(rng, ws, seed, pid, base_only=False):
    """Records of one R2 parent: the CF_SD twins, plus complement pairs if the rule is a single status atom."""
    core, variants = gen.train_parent_variants(BASE, rng, ws)
    surf_rng = random.Random(rng.randrange(2 ** 31))
    x_rng = random.Random(rng.randrange(2 ** 31))

    def meta(v):
        return {"parent_id": f"tr-R2-{seed}-{pid}", "split": "train", "group": "R2", "variant": v}

    rule, status = variants[0][:2]
    if base_only:
        return [_twin(rule, core, status, gen.surface_spec(surf_rng, "sd"), surf_rng, meta("fill_base"))]
    recs = [_twin(r, core, s, gen.surface_spec(surf_rng, "sd"), surf_rng, meta(v)) for r, s, v in variants]
    if rule["op"] not in W.STATUS_OPS:
        return recs
    k = x_rng.choice(COMP_K)
    used = set(rule["set"]) | {s for _, s, _ in variants}
    seen = {_dup_key(r) for r in recs if r["variant"] in ("base", "status_swap")}
    tids = x_rng.sample(T.TRAIN_IDS, k)
    for i, out in enumerate(ws.draw(x_rng, k, exclude=used)):
        surf = dict(gen.surface_spec(surf_rng, "sd"), tid=tids[i])
        pair = surf_rng.randrange(2 ** 31)  # same state order for both members
        st = x_rng.choice(rule["set"])
        listed = _twin(rule, core, st, surf, random.Random(pair), meta("comp_listed"))
        if len(rule["set"]) == 1 and _dup_key(listed) in seen:  # frozen spec: redraw the wording, else drop
            free = [t for t in T.TRAIN_IDS if t not in tids]
            if not free:
                continue
            tids[i] = surf["tid"] = x_rng.choice(free)
            listed = _twin(rule, core, st, surf, random.Random(pair), meta("comp_listed"))
        recs += [_twin(rule, core, out, surf, random.Random(pair), meta("comp_unlisted")), listed]
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
    is malformed, starts with 'h' (the spent held-out namespace), is repeated in the pool, or is already registered
    with another text or polarity."""
    if not re.fullmatch(r"[a-z][a-z0-9]*", prefix) or prefix in ("p2", "p3") or prefix.startswith("h"):
        raise ValueError(f"bad prefix {prefix!r} (p2/p3 are the test-pool namespaces, h* the held-out one)")
    todo, seen = [], set()
    for t in pool:
        if not isinstance(t, dict) or not {"id", "text", "pol", "form"} <= set(t):
            raise ValueError(f"pool entry is not a dict with id, text, pol, form: {t!r}")
        if not isinstance(t["id"], str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", t["id"]) or t["id"] in seen:
            raise ValueError(f"template id {t['id']!r} is malformed or repeated")
        if t["id"][0] in "hH":
            raise ValueError(f"template id {t['id']!r} starts with 'h' (reserved for the spent held-out templates)")
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
    """{file:id: text} for the P2 DEV pools (read-only). Refuses, before reading, a file named like P3/SEALED/secret."""
    paths = [Path(p) for p in paths]
    if any(BLIND.search(p.name) for p in paths):
        raise ValueError("a reference file names P3/SEALED/secret material; P3 has its own counts-only audit")
    refs = {}
    for p in paths:
        for t in json.loads(p.read_text(encoding="utf-8"))["templates"]:
            refs[f"{p.stem}:{t['id']}"] = t["text"]
    return refs


def rcx_refs():
    """The built-in RCX wordings (wording_audit.reference_wordings: t*/h* templates, frames, clauses), without the
    templates registered at runtime (p2:*, tr:*)."""
    return {k: v for k, v in WA.reference_wordings().items() if ":" not in k.split(":", 1)[1]}


def status_phrases(vocab=None):
    """Excluded/probe status strings that are not one plain word, as word tuples ('not posted' -> ('not', 'posted'));
    one-word ones already in `vocab` are left to the word check."""
    vocab = WA.status_words() if vocab is None else vocab
    out = set()
    for s in set(V.EXCLUDED) | set(V.POSTEDLIKE) | set(V.NEARMISS_VOID):
        p = tuple(WA.words(s))
        if p and not re.fullmatch(r"[a-z]+", s.lower()) and (len(p) > 1 or p[0] not in vocab):
            out.add(p)
    return out


def phrase_hits(text, phrases=None):
    ws = WA.words(text)
    return sorted(" ".join(p) for p in (status_phrases() if phrases is None else phrases)
                  if any(tuple(ws[i:i + len(p)]) == p for i in range(len(ws) - len(p) + 1)))


def audit_pool(pool, extra_refs=None, stems=True):
    """{id: [reasons]} for pool templates that must not be trained on (empty = clean); the frozen R1 code audit
    (items 1-2) plus P2 DEV at the RCX level: status_word:<w> or status_phrase:<p> (a status-like word, or a multi-word
    excluded/probe string such as 'not posted'); with `stems`, status_stem:<w~s> (an inflection of a real one, CX D-34);
    near_dup:<key> for an RCX wording (t*, h*, frames, clauses) or a P2 DEV wording (v1 or v2): wording_audit.ref_hits,
    Jaccard >= 0.5 or a shared 5-word sequence. `extra_refs` ({key: text}) are compared the same way but reported
    without keys, one 'near_dup:blind' per hit; keys naming P3/SEALED/secret material are refused (P3 is audited only by
    its separate counts-only routine). Print results through public_audit, never raw."""
    extra = dict(extra_refs or {})
    if any(not isinstance(k, str) or BLIND.search(k) for k in extra):
        raise ValueError("extra_refs keys must not name P3/SEALED/secret material (P3 has its own counts-only audit)")
    refs = {**rcx_refs(), **p2_refs()}
    if not any(k.startswith("P2_DEV_v2:") for k in refs):
        raise ValueError("P2_DEV_v2 wordings missing from the audit references")
    vocab, real = WA.status_words(), WA.real_status_words()
    phrases = status_phrases(vocab)
    bad = {}
    for t in pool:
        why = [f"near_dup:{k}" for k in WA.ref_hits(t["text"], refs)]
        why += [BLIND_HIT] * len(WA.ref_hits(t["text"], extra))
        why += [f"status_word:{w}" for w in WA.vocab_hits(t["text"], vocab)]
        why += [f"status_phrase:{p}" for p in phrase_hits(t["text"], phrases)]
        why += [f"status_stem:{h}" for h in WA.stem_hits(t["text"], vocab=real)] if stems else []
        if why:
            bad[t["id"]] = why
    return bad


def public_audit(bad, pool):
    """A printable audit result: reasons by id without blind hits, and blind hits as template counts per cell only."""
    cell = {t["id"]: f"{t['pol']}/{t['form']}" for t in pool}
    shown = {i: [w for w in why if w != BLIND_HIT] for i, why in bad.items()}
    blind = Counter(cell.get(i, "?") for i, why in bad.items() if BLIND_HIT in why)
    return {"by_id": {i: w for i, w in sorted(shown.items()) if w}, "blind_by_cell": dict(sorted(blind.items()))}


def select_pool(pool, extra_refs=None, pre_drop=None):
    """The audited pool (cf. split_pool.py): drop every template audit_pool flags and every id in `pre_drop` ({id:
    [reasons]}, e.g. from the stricter P2_v2/P3_v2 routine; use 'near_dup:blind' for P3), then keep the smallest id of
    each within-pool near-duplicate cluster (wording_audit.clusters: Jaccard >= 0.6 or a shared 6-word sequence).
    Returns (kept, dropped {id: [reasons]}); a within-pool drop reads 'pool_near_dup:<kept id>'."""
    dropped = {**{i: list(w) for i, w in (pre_drop or {}).items()}, **audit_pool(pool, extra_refs)}
    keep = [t for t in pool if t["id"] not in dropped]
    for g in WA.clusters([(t["id"], t["text"]) for t in keep]):
        for i in g[1:]:
            dropped[i] = [f"pool_near_dup:{g[0]}"]
    return [t for t in keep if t["id"] not in dropped], dropped


def make_r1(n_rows=N_ROWS, seed=SEEDS["R1"], train_pool=None, p_pool=0.5, prefix="tr", extra_refs=None):
    """R1 records: gen.make_train('CF_SD', n_rows, seed) with single-atom S+/S- wordings drawn from `train_pool` with
    probability p_pool (polarity-matched; form uniform among the pool's forms, then template uniform). Refuses the
    pool (ValueError, nothing registered) if check_pool or audit_pool flags any template, if two templates are
    within-pool near-duplicates, or if a (polarity, form) cell has fewer than MIN_PER_CELL templates."""
    if not train_pool:
        raise ValueError("R1 needs a non-empty training wording pool")
    if not 0.0 <= p_pool <= 1.0:
        raise ValueError(f"p_pool {p_pool}")
    check_pool(train_pool, prefix)
    bad = audit_pool(train_pool, extra_refs)
    if bad:
        raise ValueError(f"training pool refused ({len(bad)} templates): {public_audit(bad, train_pool)}")
    groups = [g for g in WA.clusters([(t["id"], t["text"]) for t in train_pool]) if len(g) > 1]
    if groups:
        raise ValueError(f"training pool refused: within-pool near-duplicate clusters {groups} (select_pool keeps one)")
    n = Counter((t["pol"], t["form"]) for t in train_pool)
    short = {f"{p}/{f}": n[(p, f)] for p, f in CELLS if n[(p, f)] < MIN_PER_CELL}
    if short:
        raise ValueError(f"training pool refused: cells below {MIN_PER_CELL} templates: {short}")
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

def rule_type(rule):
    return rule["op"] if rule["op"] in W.STATUS_OPS else "composition"


def recipe_stats(records):
    """Descriptive composition of a record set (JSON-able). A parent's rule type is its first record's (base)."""
    n = len(records)
    spos = [r for r in records if r["rule_ast"]["op"] == "S+"]
    comp = sum(r["gold"]["listed"] is False for r in spos)
    pos_noop = [r for r in records if (W.status_atom(r["rule_ast"]) or {}).get("op") == "S+"
                and r["gold"]["decision"] == "noop"]  # rules with an S+ atom (single or composed), no-op rows
    pos_noop_unl = sum(r["gold"]["listed"] is False for r in pos_noop)
    rule_dec = Counter((r["rule_ast"]["op"], r["gold"]["decision"]) for r in records)
    first, keys = {}, Counter()
    for r in records:
        first.setdefault(r["parent_id"], rule_type(r["rule_ast"]))
        keys[(r["parent_id"],) + _dup_key(r)] += 1
    pairs = Counter((r["rule_ast"]["op"], r["variant"], r["gold"]["decision"]) for r in records
                    if r["variant"].startswith("comp_"))
    return {"rows": n, "parents": len(first),
            "noop_share": round(sum(r["gold"]["decision"] == "noop" for r in records) / max(1, n), 4),
            "spos_rows": len(spos), "spos_unlisted_rows": comp,  # single S+ atom, status outside the set: no-op
            "spos_unlisted_share": round(comp / max(1, n), 4),
            "spos_unlisted_frac_of_spos": round(comp / max(1, len(spos)), 4),
            "s_pos_noop_unlisted_frac": round(pos_noop_unl / max(1, len(pos_noop)), 4),
            "rule_type_rows": dict(sorted(Counter(rule_type(r["rule_ast"]) for r in records).items())),
            "rule_type_parents": dict(sorted(Counter(first.values()).items())),
            "rule_decisions": {f"{op}|{d}": c for (op, d), c in sorted(rule_dec.items())},
            "comp_rows": {f"{op}|{v}|{d}": c for (op, v, d), c in sorted(pairs.items())},
            "dup_rows_within_parent": sum(c - 1 for c in keys.values()),  # same wording, rule, state and event
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
    ap.add_argument("--pool", help="R1: JSON file with a 'templates' list (training candidates; never a sealed file); "
                                   "select_pool's survivors are used")
    a = ap.parse_args()
    seed = SEEDS[a.recipe] if a.seed is None else a.seed
    out = {}
    if a.recipe == "R2":
        recs = make_r2(a.n, seed)
    else:
        if not a.pool or BLIND.search(Path(a.pool).name):
            raise SystemExit("R1 needs --pool <training pool json> (P3/SEALED/secret files are refused)")
        pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))["templates"]
        kept, dropped = select_pool(pool)
        def kind(w):
            if w == BLIND_HIT or not w.startswith("near_dup:"):
                return w if w == BLIND_HIT else w.split(":")[0]
            return "near_dup:P2" if w.startswith("near_dup:P2_DEV") else "near_dup:RCX"

        n = Counter((t["pol"], t["form"]) for t in kept)
        why = Counter(k for ws in dropped.values() for k in {kind(w) for w in ws})  # templates per reason kind
        out["pool"] = {"candidates": len(pool), "kept": len(kept),
                       "kept_by_cell": {f"{p}/{f}": n[(p, f)] for p, f in CELLS},
                       "dropped_by_reason": dict(sorted(why.items())), "dropped": public_audit(dropped, pool)}
        recs = make_r1(a.n, seed, kept)
    print(json.dumps(dict(recipe_stats(recs), seed=seed, **out), indent=1))


if __name__ == "__main__":
    main()
