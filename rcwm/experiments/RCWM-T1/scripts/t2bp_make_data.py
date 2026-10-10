"""RCWM-T2b prerequisite screen: training rows (rendered four ways) and the fresh dev screen bank. No model call, no torch.

  python -B scripts/t2bp_make_data.py                          # writes data/t2bp_train_{dint,strace,bbits,ffill}.jsonl,
                                                               # data/t2bp_train.SOURCE.json, data/t2bp_dev.jsonl,
                                                               # data/t2bp_dev.SOURCE.json (create-only)
  python -B scripts/t2bp_make_data.py --rerender-canonical     # revision 3, once: re-render the canonical files from
                                                               # their own rows (never re-drawn); D-int/S-trace/B-bits
                                                               # must come out byte-identical; the F-filler file gets
                                                               # the amendment-4b constant fill. The superseded filler
                                                               # file and SOURCE are kept (renamed *.rev2*)
  python -B scripts/t2bp_make_data.py --surface-diversity on   # THE PRIMARY BUILD (revision 3, C1/C2): the SAME
                                                               # canonical rows, each with a seeded paraphrase, field-name
                                                               # variant and input key order (training subsets only):
                                                               # data/t2bp_train_sd_<arm>.jsonl + data/t2bp_train_sd.SOURCE.json
  python -B scripts/t2bp_make_data.py --verify                 # re-audit both builds, re-check the SOURCE hashes, golds
                                                               # and the sd -> canonical link, and re-check freshness
                                                               # with a fresh name-skipping harvest (exit 1 on failure)

Freshness. No row's (before, event) payload and no row's FDS operative content (before, add, delete, pre_on, pre_off)
equals one in: the historical records tree harvested by scripts/l1_make_ood.harvest_historical (rcwmt.bank.
load_exclusions() is NOT called: files named HSKILL_CONFIRM_V2* are skipped by name, never opened or hashed); every
transition of data/bank_pilot.json, bank_confirm.json, bank_confirm2.json, bank_s4000.json; every row of
data/t2_train.jsonl, t2_eval.jsonl, t2_eval_xframe.jsonl, t2_eval_supp.jsonl; SELECT-VIEWS-001; the L1 OOD bank;
the FDS/composition T banks (scripts/p3_make_bank.load_all_sources, pinned hashes); and the PREREG-3 bank
data/bank_p3_oor.jsonl. Training rows, dev one-step rows and dev chain transitions are also disjoint from each other
(payload and operative content) and unique within themselves. Gold = rcwmt.env.step, checked against the validated FDS
oracle (env.cross_check) on every row and chain transition. No model, model output or GPU is used.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # nothing here touches the driver; empty string, never "-1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bp_common as C  # noqa: E402
import l1_make_ood as L1  # noqa: E402  (name-skipping historical harvest; torch-free)
import p3_make_bank as MB  # noqa: E402  (pinned exclusion sources incl. L1 OOD and the FDS/cc T banks; torch-free)
from rcwmt import env  # noqa: E402

P = C.FAM
P3_BANK = ROOT / "data" / "bank_p3_oor.jsonl"
P3_BANK_SHA = "0fd18295d6eaa84d0f451c94b81ea1ecf37ab88e8ab298b58820d80ac740bf12"
T2_TRAIN = ROOT / "data" / "t2_train.jsonl"
CONFIRM2 = ROOT / "data" / "bank_confirm2.json"
MAX_DRAWS = 3_000_000
SUB_SEEDS = {"dev": 1, "chains": 2, "train": 3, "shuffle": 4, "surface": 5}


def rng_for(part: str) -> random.Random:
    return random.Random(C.SEED * 10 + SUB_SEEDS[part])


def log(lines: list, msg: str) -> None:
    L1.log(lines, msg)


def counter():
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(str(C.TOKENIZER_JSON))
    return lambda s: len(tk.encode(s, add_special_tokens=False).ids)


# ---------------------------------------------------------------------------------------------- exclusion sources
def load_sources(lines: list, historical: bool = True) -> dict:
    src = MB.load_all_sources(lines, historical=historical)
    sha = C.sha256(P3_BANK)
    if sha != P3_BANK_SHA:
        raise SystemExit(f"bank_p3_oor.jsonl sha256 {sha} != pinned")
    pay = MB.harvest_file(P3_BANK)
    src["bank_p3_oor.jsonl"] = {"payloads": pay, "ops": L1.op_keys_of_payloads(pay), "sha256": sha}
    log(lines, f"bank_p3_oor.jsonl: {len(pay)} payloads")
    return src


class Fresh:
    def __init__(self, pay: set, ops: set):
        self.pay, self.ops = set(pay), set(ops)

    def ok(self, items) -> bool:
        p = [C.payload(b, e) for b, e in items]
        o = [C.opkey(b, e) for b, e in items]
        if len(set(p)) != len(p) or len(set(o)) != len(o):
            return False
        return not any(x in self.pay for x in p) and not any(x in self.ops for x in o)

    def add(self, items) -> None:
        for b, e in items:
            self.pay.add(C.payload(b, e))
            self.ops.add(C.opkey(b, e))


# ---------------------------------------------------------------------------------------------- generators
def draw8(rng: random.Random, m) -> tuple[int, dict]:
    return env.gen_start(P, rng, m), env.gen_action_library(rng, m, 1)[0]


def gen_start_w(rng: random.Random, m, w: int) -> int:
    while True:
        n = min(w, max(1, round(rng.choice(m.p_before_bits) * w / 8)))
        x = env._pick_bits(rng, n, list(range(w)))
        if x >= 1 << 8:
            return x


def gen_action_w(rng: random.Random, m, w: int) -> dict:
    """env.gen_action_library's draw over w bits (same recorded bit-count marginals, name A1..A40)."""
    add = 1 << rng.randrange(w)
    pre_on = env._pick_bits(rng, rng.choice(m.p_pre_on_bits), [b for b in range(w) if not (add >> b) & 1])
    pre_off = env._pick_bits(rng, rng.choice(m.p_pre_off_bits), [b for b in range(w) if not (pre_on >> b) & 1])
    delete = env._pick_bits(rng, rng.choice(m.p_delete_bits), [b for b in range(w) if not (add >> b) & 1])
    return {"add": add, "cost": rng.choice(m.p_costs), "delete": delete, "kind": "action",
            "name": f"A{rng.randint(1, 40)}", "pre_off": pre_off, "pre_on": pre_on}


def kind_of(before: int, ev: dict) -> str:
    """Type, with the violation kind for inapplicable rows (e.g. 'Ib/pre_on')."""
    t = C.rtype(before, ev)
    v = C.violation(before, ev)
    return t if v is None else f"{t}/{v}"


def check_row(before: int, ev: dict) -> int:
    env.cross_check(P, before, ev)
    return C.gold(before, ev)


def mk_row(rid: str, split: str, stratum: str, before: int, ev: dict, w: int, **extra) -> dict:
    g = check_row(before, ev)
    return {"kind": "onestep", "id": rid, "split": split, "stratum": stratum, "type": C.rtype(before, ev),
            "violation": C.violation(before, ev), "width": w, "before": before, "event": ev, "gold": g,
            "applicable": C.applicable(before, ev), **extra}


# ---------------------------------------------------------------------------------------------- dev bank
def dev_accepts(stratum: str, sub: str | None, before: int, ev: dict, w: int) -> bool:
    k = kind_of(before, ev)
    if stratum in C.DEV_TYPE:
        return k == C.DEV_TYPE[stratum]
    if stratum == "D_inapp_addset":                 # inapplicable, add bit already set, delete clears a set bit
        return k == sub
    if stratum == "D_inapp_pre_on":
        return k in ("Ib/pre_on", "Ia/pre_on")
    if stratum == "D_inapp_pre_off":
        return k in ("Ib/pre_off", "Ia/pre_off")
    # OOD: high bits must be operative and the state out of the 8-bit range
    if before < 1 << 8 or (ev["add"] | ev["delete"] | ev["pre_on"] | ev["pre_off"]) < 1 << 8:
        return False
    if sub in ("A1", "A2", "A3", "A4"):
        return k == sub
    return k in (f"Ib/{sub[6:]}", f"Ia/{sub[6:]}")


def build_dev(m, fresh: Fresh, lines: list) -> list:
    rng = rng_for("dev")
    rows, draws = [], Counter()
    for stratum in C.DEV_STRATA:
        w = C.OOD_WIDTH.get(stratum, C.WIDTH_ID)
        if stratum in C.OOD_WIDTH:
            subs = [(s, C.DEV_N[stratum] // len(C.OOD_SUBTYPES)) for s in C.OOD_SUBTYPES]
        elif stratum == "D_inapp_addset":
            subs = [(s, C.DEV_N[stratum] // len(C.ADDSET_SUBTYPES)) for s in C.ADDSET_SUBTYPES]
        else:
            subs = [(None, C.DEV_N[stratum])]
        for sub, n in subs:
            got = 0
            while got < n:
                draws[stratum] += 1
                if draws[stratum] > MAX_DRAWS:
                    raise SystemExit(f"could not fill {stratum}/{sub}")
                if w == C.WIDTH_ID:
                    before, ev = draw8(rng, m)
                else:
                    before, ev = gen_start_w(rng, m, w), gen_action_w(rng, m, w)
                if not dev_accepts(stratum, sub, before, ev, w) or not fresh.ok([(before, ev)]):
                    continue
                fresh.add([(before, ev)])
                rows.append(mk_row(f"t2bp-dev-{stratum[2:]}-{len([r for r in rows if r['stratum'] == stratum]):03d}", "dev",
                                   stratum, before, ev, w, subtype=sub or C.rtype(before, ev)))
                got += 1
    log(lines, f"dev one-step: {len(rows)} rows from {sum(draws.values())} draws {dict(draws)}")
    return rows


def build_chains(m, fresh: Fresh, lines: list) -> list:
    """P chain parents exactly as rcwmt.bank.build draws them (gen_start, a 6-action library, 8 candidate 4-event plans
    from the library), all transitions fresh, >= 4 distinct final states per parent (confirm2's min_distinct_p)."""
    rng = rng_for("chains")
    parents, seen, rej = [], set(), Counter()
    while len(parents) < C.CHAIN_PARENTS:
        s0 = env.gen_start(P, rng, m)
        lib = env.gen_action_library(rng, m, C.CHAIN_ACTIONS)
        cands = [[dict(rng.choice(lib)) for _ in range(C.CHAIN_H)] for _ in range(C.CHAIN_K)]
        items = []
        for evs in cands:
            st = env.rollout(P, s0, evs)
            for t, ev in enumerate(evs):
                env.cross_check(P, st[t], ev)
                items.append((st[t], ev))
        uniq = list({C.payload(b, e): (b, e) for b, e in items}.values())
        if not fresh.ok(uniq):
            rej["collision"] += 1
            continue
        if len({env.rollout(P, s0, evs)[-1] for evs in cands}) < C.CHAIN_MIN_DISTINCT:
            rej["fewer than 4 distinct finals"] += 1
            continue
        key = env.dumps({"f": P, "s0": s0, "c": cands})
        if key in seen:
            continue
        seen.add(key)
        fresh.add(uniq)
        pid = "T2BP-P-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        parents.append({"kind": "chain_parent", "parent_id": pid, "split": "test", "family": P, "width": C.WIDTH_ID, "s0": s0,
                        "library": lib, "candidates": [{"cand_id": f"{pid}-c{ci}", "events": evs} for ci, evs in enumerate(cands)],
                        "golds": [[env.dumps(s) for s in env.rollout(P, s0, evs)] for evs in cands]})
    log(lines, f"dev chains: {len(parents)} parents x {C.CHAIN_K} candidates x {C.CHAIN_H} steps; rejected {dict(rej)}")
    return parents


# ---------------------------------------------------------------------------------------------- training rows
def draw_typed(rng: random.Random, m, want: str) -> tuple[int, dict]:
    """A standard 8-bit row whose kind_of() == want; Z types are standard draws with add set to 0 (delete-only action)."""
    for _ in range(MAX_DRAWS):
        before, ev = draw8(rng, m)
        if want.startswith("Z"):
            ev = {**ev, "add": 0}
        if kind_of(before, ev) == want:
            return before, ev
    raise SystemExit(f"could not draw {want}")


def flip_candidates(before: int, ev: dict, kind: str) -> list:
    out = []
    for j in range(C.WIDTH_ID):
        bit = 1 << j
        if ev["pre_on"] & bit or ev["pre_off"] & bit:
            continue
        if kind == "pre_on" and not before & bit and not ev["add"] & bit:
            out.append(j)
        if kind == "pre_off" and before & bit:
            out.append(j)
    return out


def build_train(m, fresh: Fresh, lines: list) -> list:
    rng = rng_for("train")
    rows, n_pair = [], 0
    draws = Counter()
    # minimal pairs first (they are the scarcest), flip kinds alternate pre_on / pre_off within each pair type
    for t, n in C.TRAIN_PAIRS.items():
        made = 0
        while made < n:
            flip = ("pre_on", "pre_off")[made % 2]
            draws[f"pair:{t}"] += 1
            before, ev = draw_typed(rng, m, t)
            cand = flip_candidates(before, ev, flip)
            if not cand:
                continue
            j = rng.choice(cand)
            twin = {**ev, flip: ev[flip] | (1 << j)}
            if C.rtype(before, twin) != C.PAIR_INAPP[t] or C.violation(before, twin) != flip:
                continue
            if not fresh.ok([(before, ev), (before, twin)]):
                continue
            fresh.add([(before, ev), (before, twin)])
            pid = f"mp-{n_pair:03d}"
            n_pair += 1
            rows.append(mk_row("", "train", t, before, ev, C.WIDTH_ID, pair_id=pid, pair_role="app", flip=None))
            rows.append(mk_row("", "train", C.PAIR_INAPP[t], before, twin, C.WIDTH_ID, pair_id=pid, pair_role="inapp", flip=flip))
            made += 1
    for t, n in C.TRAIN_UNPAIRED.items():
        inapp = t in ("Ib", "Ia", "Zi")
        for i in range(n):
            want = f"{t}/{('pre_on', 'pre_off')[i % 2]}" if inapp else t
            while True:
                draws[f"row:{t}"] += 1
                before, ev = draw_typed(rng, m, want)
                if fresh.ok([(before, ev)]):
                    break
            fresh.add([(before, ev)])
            rows.append(mk_row("", "train", t, before, ev, C.WIDTH_ID, pair_id=None, pair_role=None, flip=None))
    rng_for("shuffle").shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"t2bp-tr-{i:04d}"
    log(lines, f"training: {len(rows)} rows ({n_pair} minimal pairs); draws {dict(draws)}")
    return rows


# ---------------------------------------------------------------------------------------------- rendering
def assign_surfaces(rows: list, on: bool) -> dict:
    """{row id: surface or None}. Pair members share one surface: paraphrase, field names and the key order of every
    JSON object of the prompt, each drawn from the TRAINING subsets only (C.TRAIN_SURFACE_IDX, C.train_orders)."""
    if not on:
        return {r["id"]: None for r in rows}
    rng = rng_for("surface")
    by_pair, out = {}, {}
    for r in sorted(rows, key=lambda r: r["id"]):
        key = r.get("pair_id") or r["id"]
        if key not in by_pair:
            sf = {"para": rng.choice(C.TRAIN_SURFACE_IDX), "names": rng.choice(C.TRAIN_SURFACE_IDX)}
            sf["order"] = {obj: list(C.draw_order(rng, obj)) for obj in C.OBJ_SLOTS}
            by_pair[key] = sf
        out[r["id"]] = by_pair[key]
    return out


def check_fill_k(train: list, dev: list, parents: list, count) -> dict:
    """Amendment 4b: the constant fill length must equal BWM's formula over the training rows, the dev one-step rows and
    the dev chain steps (every surface variant has the same targets); refuse otherwise."""
    items = [(r["before"], r["event"]) for r in train + dev]
    for p in parents:
        for c in p["candidates"]:
            st = env.rollout(P, p["s0"], c["events"])
            items += [(st[t], e) for t, e in enumerate(c["events"])]
    req = C.fill_k_required(items, count)
    if req != C.FILL_K:
        raise SystemExit(f"amendment 4b: the constant fill length over train+dev is {req}, t2bp_common.FILL_K is {C.FILL_K}")
    heads = [count(C.trace_head(b, e)) for b, e in items]
    return {"fill_k": req, "items": len(items), "longest_trace_head_tokens": max(heads),
            "fill_head_tokens": count('{"fill":"' + C.FILL_UNIT * C.fill_units() + '"'),
            "rule": "units = max over train+dev of tokens(trace head '{\"cmp\":..,\"calc\":N') - tokens('{\"fill\":\"\"') "
                    "(BWM bwmn/traces.filler_length); one action kind ('action') in the prerequisite family"}


CANON_FIELDS = ("id", "split", "stratum", "type", "violation", "width", "before", "event", "gold", "applicable",
                "pair_id", "pair_role", "flip")


def render_arm(rows: list, arm: str, count, surfaces: dict) -> list:
    out = []
    for r in rows:
        sf = surfaces[r["id"]]
        msgs = C.messages(arm, r["before"], r["event"], r["width"], sf)
        tgt = C.target_text(arm, r["before"], r["event"], r["width"], count)
        out.append({"id": r["id"], "prompt": C.render_chat(msgs), "target": tgt, "family": P, "arm": arm,
                    "frame": f"t2bp-{arm}", **{k: r[k] for k in CANON_FIELDS if k != "id"}, "surface": sf})
    return out


def canonical_from_arm_rows(arm_rows: list) -> list:
    return [{k: r[k] for k in CANON_FIELDS} for r in arm_rows]


def _lookup_acc(xs: list, ys: list) -> float:
    """Majority lookup accuracy of predicting y from x (= the majority share of y when x carries no information)."""
    tab = {}
    for x, y in zip(xs, ys):
        tab.setdefault(x, Counter())[y] += 1
    return round(sum(max(c.values()) for c in tab.values()) / len(xs), 4)


def _mutual_info_bits(xs: list, ys: list) -> float:
    n = len(xs)
    pxy, px, py = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    mi = sum(c / n * math.log2((c / n) / ((px[x] / n) * (py[y] / n))) for (x, y), c in pxy.items())
    return round(abs(mi), 12)


def fill_units_of(target: str):
    d = C.loads_strict(target)
    if not isinstance(d, dict) or not isinstance(d.get("fill"), str):
        return None
    s = d["fill"]
    k = len(s) // len(C.FILL_UNIT)
    return k if s == C.FILL_UNIT * k else ("non-unit", s)


def rev2_fill_units(before: int, ev: dict, count) -> int:
    """The revision-2 (superseded) row-matched fill length in units: the fill that made the whole filler target exactly
    as long as the row's trace target, i.e. tokens(trace head '{"cmp":..,"calc":N') - tokens('{"fill":""') (= 3 under
    Titan's tokenizer). It reproduces every row of the kept file data/t2bp_train_ffill.rev2_rowmatched.jsonl (tested).
    This is the length the model decodes BEFORE 'after', so it is the quantity that could leak the answer."""
    return max(1, count(C.trace_head(before, ev)) - count(C.FILL_BASE_TEXT))


def rev2_length_leak(rows: list, count) -> dict:
    """Descriptive: how well the revision-2 row-matched fill length (rev2_fill_units) predicts applicability and change
    on the given rows (majority lookup per length value), next to each property's majority share. Revision 3 computed
    this from the token count of the FULL trace text, which includes the answer ('after'); that superseded figure is
    reported alongside, labelled, for the record."""
    u = [rev2_fill_units(r["before"], r["event"], count) for r in rows]
    full = [count(C.trace_text(r["before"], r["event"])) for r in rows]
    app = [r["applicable"] for r in rows]
    ch = [r["gold"] != r["before"] for r in rows]
    return {"rows": len(rows),
            "length": "revision-2 fill units = tokens(trace head) - tokens('{\"fill\":\"\"') (= 3): the fill decoded before 'after'",
            "applicable_lookup_accuracy": _lookup_acc(u, app), "applicable_majority_share": _lookup_acc([0] * len(rows), app),
            "changed_lookup_accuracy": _lookup_acc(u, ch), "changed_majority_share": _lookup_acc([0] * len(rows), ch),
            "distinct_fill_lengths": len(set(u)),
            "superseded_revision3_figure_from_the_full_trace_text_incl_after": {
                "applicable_lookup_accuracy": _lookup_acc(full, app), "changed_lookup_accuracy": _lookup_acc(full, ch)}}


def filler_information(ff_rows: list, count) -> dict:
    """Amendment 4b audit: the fill length must carry ZERO information about the answer (one value per action kind on
    every row, so predicting applicability / change / the after value / the row type from it is no better than the
    majority class and the mutual information is 0). For contrast (descriptive): what the revision-2 row-matched FILL
    length (rev2_fill_units: trace-head tokens - 3, the fill decoded before 'after') reveals on the same rows
    (rev2_length_leak). The residual (filler tokens - trace tokens) is reported as a descriptive covariate."""
    k = [fill_units_of(r["target"]) for r in ff_rows]
    kind = [r["event"].get("kind", "action") for r in ff_rows]
    ys = {"applicable": [r["applicable"] for r in ff_rows], "changed": [r["gold"] != r["before"] for r in ff_rows],
          "after": [r["gold"] for r in ff_rows], "type": [r["type"] for r in ff_rows]}
    trace_len = [count(C.trace_text(r["before"], r["event"])) for r in ff_rows]
    fill_len = [count(r["target"]) for r in ff_rows]
    resid = [f - t for f, t in zip(fill_len, trace_len)]
    by_kind = {}
    for kk, u in zip(kind, k):
        by_kind.setdefault(kk, set()).add(u)
    out = {"distinct_fill_lengths_per_kind": {kk: sorted(map(str, v)) for kk, v in by_kind.items()},
           "fill_units_constant": {kk: (len(v) == 1 and next(iter(v)) == C.FILL_K.get(kk)) for kk, v in by_kind.items()},
           "fill_length_lookup_accuracy": {y: _lookup_acc([str(x) for x in k], v) for y, v in ys.items()},
           "majority_share": {y: _lookup_acc([0] * len(v), v) for y, v in ys.items()},
           "mutual_information_bits": {y: _mutual_info_bits([str(x) for x in k], v) for y, v in ys.items()},
           "rev2_row_matched_length_would_reveal_descriptive": rev2_length_leak(ff_rows, count),
           "residual_tokens_descriptive": {
               "min": min(resid), "max": max(resid), "mean": round(sum(resid) / len(resid), 3),
               "share_positive": round(sum(x > 0 for x in resid) / len(resid), 4),
               "mean_applicable": round(sum(x for x, a in zip(resid, ys["applicable"]) if a) / max(1, sum(ys["applicable"])), 3),
               "mean_inapplicable": round(sum(x for x, a in zip(resid, ys["applicable"]) if not a)
                                          / max(1, len(resid) - sum(ys["applicable"])), 3)}}
    out["zero_information"] = (all(out["fill_units_constant"].values()) and
                               all(out["fill_length_lookup_accuracy"][y] == out["majority_share"][y] for y in ys) and
                               all(v == 0 for v in out["mutual_information_bits"].values()))
    out["never_shorter_than_trace"] = min(resid) >= 0
    return out


def surface_audit(arm_rows: dict) -> dict:
    """Proof that no training prompt uses a held-out surface (C2): every prompt's surface is READ BACK FROM ITS TEXT
    (t2bp_common.recover_surface, independent of the recorded surface) and must equal the recorded one and hold no
    held-out factor (paraphrase 4, field names 4, the reversed top-level order, any held-out event order); no prompt
    contains text that only a held-out surface produces; plus the diversity actually used."""
    frags = C.heldout_fragments()
    bad_rec, bad_held, bad_frag = [], [], []
    combos, ev_orders, top_orders = Counter(), Counter(), Counter()
    for a, rr in arm_rows.items():
        for r in rr:
            try:
                got = C.recover_surface(C.split_chat(r["prompt"]))
            except ValueError as e:
                bad_rec.append(f"{a}:{r['id']}: {e}")
                continue
            sf = r.get("surface")
            want = {"para": C.para_of(sf), "names": int((sf or {}).get("names", 0)),
                    "order": {o: list(C.order_of(sf, o)) for o in C.OBJ_SLOTS}, "width": r["width"]}
            if got != want:
                bad_rec.append(f"{a}:{r['id']}")
            if C.heldout_factors(got) or C.heldout_factors(sf):
                bad_held.append(f"{a}:{r['id']}: {C.heldout_factors(got) or C.heldout_factors(sf)}")
            hits = [k for k, f in frags.items() if f in r["prompt"]]
            if hits:
                bad_frag.append(f"{a}:{r['id']}: {hits}")
            if a == "dint":
                combos[(got["para"], got["names"])] += 1
                ev_orders[tuple(got["order"]["event"])] += 1
                top_orders[tuple(got["order"]["top"])] += 1
    n = sum(ev_orders.values())
    canon_rows = sum(1 for r in arm_rows.get("dint", []) if C.para_of(r.get("surface")) == 0 and
                     int((r.get("surface") or {}).get("names", 0)) == 0 and
                     all(C.order_of(r.get("surface"), o) == C.OBJ_SLOTS[o] for o in C.OBJ_SLOTS))
    stats = {"rows": n, "paraphrase_x_names_combinations": len(combos),
             "rows_in_the_exact_canonical_surface (the in-surface evaluation surface of E1/E2)": canon_rows,
             "rows_per_paraphrase": dict(sorted(Counter(p for p, _ in combos.elements()).items())),
             "rows_per_names": dict(sorted(Counter(q for _, q in combos.elements()).items())),
             "event_order_canonical_share": round(ev_orders[C.OBJ_SLOTS["event"]] / n, 4) if n else None,
             "distinct_event_orders": len(ev_orders), "top_orders": {",".join(k): v for k, v in top_orders.items()},
             "heldout_fragments_scanned": frags}
    return {"checks": {"every prompt's surface, read back from its text, equals the recorded surface": not bad_rec,
                       "no prompt uses a held-out paraphrase, field-name variant or key order": not bad_held,
                       "no prompt contains text only a held-out surface produces": not bad_frag},
            "failures": {"recover": bad_rec[:5], "heldout": bad_held[:5], "fragments": bad_frag[:5]}, "stats": stats}


def audit_files(files: dict, count, source: dict | None = None) -> dict:
    """The full training-data audit (also t2bp_audit.py): balance gates and quotas on the canonical rows, identical rows
    in every rendering, prompts == the arm's frame (D-int canonical prompts == T2's frame), every target parses back to
    the gold (strict parsers), the amendment-4b filler (one constant fill string per action kind on every row, never
    shorter than the row's trace, zero information about the answer), and the surface proof (no held-out surface in
    training; surfaces read back from the prompt text)."""
    arm_rows = {a: (C.load_jsonl(p) if isinstance(p, (str, Path)) else p) for a, p in files.items()}
    base = canonical_from_arm_rows(arm_rows["dint"])
    res = {"core": C.audit_rows(base)}
    checks = {}
    checks["every rendering has the same canonical rows in the same order"] = all(
        canonical_from_arm_rows(v) == base for v in arm_rows.values())
    bad_prompt, bad_parse, bad_fill, bad_t2frame = [], [], [], []
    for a, rr in arm_rows.items():
        for r in rr:
            msgs = C.messages(a, r["before"], r["event"], r["width"], r.get("surface"))
            if r["prompt"] != C.render_chat(msgs):
                bad_prompt.append(f"{a}:{r['id']}")
            if a == "dint" and not C.heldout_factors(r.get("surface")) and C.para_of(r.get("surface")) == 0 and \
                    int((r.get("surface") or {}).get("names", 0)) == 0 and \
                    all(C.order_of(r.get("surface"), o) == C.OBJ_SLOTS[o] for o in C.OBJ_SLOTS) and \
                    msgs != env.multi_step_messages(P, r["before"], [r["event"]]):
                bad_t2frame.append(r["id"])
            p = C.parse(a, r["target"], r["width"])
            sc = C.score_row(a, r["target"], r)
            if not (p["well_formed"] and p["after"] == r["gold"] and sc["correct"]
                    and sc.get("cmp_ok", True) and sc.get("calc_ok", True) and sc.get("flag_ok", True)):
                bad_parse.append(f"{a}:{r['id']}")
            if a == "ffill" and r["target"] != C.target_text("ffill", r["before"], r["event"], r["width"]):
                bad_fill.append(r["id"])
    checks["prompts == render_chat(messages(arm, row)) on every row"] = not bad_prompt
    checks["D-int prompts in the canonical surface == T2's frame env.multi_step_messages(P, before, [event])"] = not bad_t2frame
    checks["every target parses back to its gold (strict; cmp/calc/flag exact)"] = not bad_parse
    if "ffill" in arm_rows:
        info = filler_information(arm_rows["ffill"], count)
        res["filler_4b"] = info
        checks["filler target == the amendment-4b constant-fill target on every row"] = not bad_fill
        checks["fill length carries zero information about the answer (one value per kind; lookup == majority; MI 0)"] = \
            info["zero_information"]
        checks["no filler target is shorter than its row's trace target"] = info["never_shorter_than_trace"]
    sa = surface_audit(arm_rows)
    res["surface"] = sa
    checks.update(sa["checks"])
    if source is not None:
        checks["file sha256 == SOURCE record"] = all(
            C.sha256(p) == (source.get("files") or {}).get(Path(p).name) for p in files.values() if isinstance(p, (str, Path)))
    res["checks"] = checks
    res["failures"] = {"prompt": bad_prompt[:5], "parse": bad_parse[:5], "fill": bad_fill[:5], "t2frame": bad_t2frame[:5],
                       **sa["failures"]}
    res["ok"] = res["core"]["ok"] and all(checks.values())
    return res


# ---------------------------------------------------------------------------------------------- stats for the plan
def token_stats(arm_rows: dict, dev_rows: list, parents: list, count) -> dict:
    out = {"train": {}, "dev_onestep": {}, "dev_chain_steps": {}}
    for a, rr in arm_rows.items():
        out["train"][a] = {"rows": len(rr), "mean_prompt_tokens": round(sum(count(r["prompt"]) for r in rr) / len(rr), 2),
                           "mean_target_tokens": round(sum(count(r["target"]) for r in rr) / len(rr), 2)}
        out["train"][a]["mean_row_tokens_incl_eos"] = round(out["train"][a]["mean_prompt_tokens"] + out["train"][a]["mean_target_tokens"] + 1, 2)
    out["dev_onestep_heldout_surface"] = {}
    for a in C.ARMS:
        pt = [count(C.render_chat(C.messages(a, r["before"], r["event"], r["width"]))) for r in dev_rows]
        tt = [count(C.target_text(a, r["before"], r["event"], r["width"], count)) for r in dev_rows]
        out["dev_onestep"][a] = {"rows": len(dev_rows), "mean_prompt_tokens": round(sum(pt) / len(pt), 2),
                                 "mean_target_tokens": round(sum(tt) / len(tt), 2)}
        ph = [count(C.render_chat(C.messages(a, r["before"], r["event"], r["width"], C.heldout_surface_of(r)))) for r in dev_rows]
        out["dev_onestep_heldout_surface"][a] = {"rows": len(dev_rows), "mean_prompt_tokens": round(sum(ph) / len(ph), 2),
                                                 "mean_target_tokens": round(sum(tt) / len(tt), 2)}
        steps = [(json.loads(p["golds"][ci][t]), c["events"][t]) for p in parents for ci, c in enumerate(p["candidates"])
                 for t in range(C.CHAIN_H)]
        st = [count(C.target_text(a, b, e, C.WIDTH_ID, count)) for b, e in steps]
        sp = [count(C.render_chat(C.messages(a, b, e, C.WIDTH_ID))) for b, e in steps]
        out["dev_chain_steps"][a] = {"steps": len(steps), "mean_prompt_tokens": round(sum(sp) / len(sp), 2),
                                     "mean_target_tokens": round(sum(st) / len(st), 2)}
    t2 = C.load_jsonl(T2_TRAIN)
    out["t2_train_reference"] = {"rows": len(t2), "mean_row_tokens_incl_eos": round(
        sum(count(r["prompt"]) + count(r["target"]) + 1 for r in t2) / len(t2), 2)}
    # generation-rate calibration: mean gold answer tokens of PREREG-2's P chain_m1 (S4) and ledger chain_one (S1) steps
    b2 = json.loads(CONFIRM2.read_text(encoding="utf-8"))
    tok = {"prerequisite_plan": [], "ledger_replay": []}
    for p in b2["parents"]:
        if p["family"] in tok:
            for c in p["candidates"]:
                for s in env.rollout(p["family"], p["s0"], c["events"])[1:]:
                    tok[p["family"]].append(count(env.dumps(s)))
    out["rate_calibration"] = {"P_chain_m1_mean_answer_tokens": round(sum(tok["prerequisite_plan"]) / len(tok["prerequisite_plan"]), 3),
                               "L_chain_one_mean_answer_tokens": round(sum(tok["ledger_replay"]) / len(tok["ledger_replay"]), 3),
                               "source": "gold states of data/bank_confirm2.json (no model output read)"}
    return out


# ---------------------------------------------------------------------------------------------- writing
def write_create_only(path: Path, rows: list) -> str:
    if path.exists():
        raise SystemExit(f"{path} exists (create-only)")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return C.sha256(path)


def build_everything(lines: list, historical: bool = True):
    code = L1.check_code()
    log(lines, f"code identity: {code}")
    src = load_sources(lines, historical=historical)
    excl_pay = set().union(*(s["payloads"] for s in src.values()))
    excl_ops = set().union(*(s["ops"] for s in src.values()))
    log(lines, f"exclusion union: {len(excl_pay)} payloads, {len(excl_ops)} operative contents ({len(src)} sources)")
    m = env.load_marginals()
    fresh = Fresh(excl_pay, excl_ops)
    dev = build_dev(m, fresh, lines)
    parents = build_chains(m, fresh, lines)
    train = build_train(m, fresh, lines)
    # independent re-checks
    items = [(r["before"], r["event"]) for r in dev + train]
    for p in parents:
        for c in p["candidates"]:
            st = env.rollout(P, p["s0"], c["events"])
            items += [(st[t], e) for t, e in enumerate(c["events"])]
    for b, e in items[:200]:
        assert C.opkey(b, e) == L1.opkey(P, b, e), "opkey definition drift"
    overlaps = {}
    for name, s in src.items():
        overlaps[name] = {"payload": sum(C.payload(b, e) in s["payloads"] for b, e in items),
                          "operative": sum(C.opkey(b, e) in s["ops"] for b, e in items)}
    tr_ops = {C.opkey(r["before"], r["event"]) for r in train}
    dev_ops = {C.opkey(r["before"], r["event"]) for r in dev}
    ch_ops = set()
    for p in parents:
        for c in p["candidates"]:
            st = env.rollout(P, p["s0"], c["events"])
            ch_ops |= {C.opkey(st[t], e) for t, e in enumerate(c["events"])}
    internal = {"train&dev": len(tr_ops & dev_ops), "train&chains": len(tr_ops & ch_ops), "dev&chains": len(dev_ops & ch_ops)}
    if any(v["payload"] or v["operative"] for v in overlaps.values()) or any(internal.values()):
        raise SystemExit(f"freshness violated: {overlaps} {internal}")
    log(lines, f"overlaps with every source (payload / operative): all 0 ({len(src)} sources); internal {internal}")
    src_summary = {k: {"payloads": len(v["payloads"]), "operative_contents": len(v["ops"]), "sha256": v.get("sha256"),
                       **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"]} if "skipped_by_name_never_opened" in v else {})}
                   for k, v in src.items()}
    return code, src_summary, overlaps, internal, dev, parents, train


def describe_dev(dev: list, parents: list, train: list) -> dict:
    h = {}
    for s in C.DEV_STRATA:
        rr = [r for r in dev if r["stratum"] == s]
        floors = {k: round(sum(C.heuristics(r["before"], r["event"])[k] == r["gold"] for r in rr) / len(rr), 3)
                  for k in list(C.AUDITED) + list(C.DESCRIPTIVE_HEURISTICS)}
        sfl = {k: round(sum(C.SHORTCUT_POLICIES[k](r["before"], r["event"]) == r["gold"] for r in rr) / len(rr), 3)
               for k in C.SHORTCUT_POLICIES}
        h[s] = {"n": len(rr), "subtypes": dict(Counter(r["subtype"] for r in rr)), "types": dict(Counter(r["type"] for r in rr)),
                "applicable": sum(r["applicable"] for r in rr), "heuristic_floors": floors, "shortcut_floors": sfl}
    tr_struct = {C.structural_key(r["before"], r["event"], r["width"]) for r in train}
    struct = {s: round(sum(C.structural_key(r["before"], r["event"], r["width"]) in tr_struct
                           for r in dev if r["stratum"] == s) / C.DEV_N[s], 3) for s in C.DEV_STRATA}
    chains = C.dev_chains(parents)
    fin = [json.loads(p["golds"][ci][-1]) for p in parents for ci in range(C.CHAIN_K)]
    copy4 = sum(f == p["s0"] for p in parents for f in [json.loads(g[-1]) for g in p["golds"]]) / len(chains)
    step_types = Counter()
    for p in parents:
        for ci, c in enumerate(p["candidates"]):
            for t, e in enumerate(c["events"]):
                step_types[C.rtype(json.loads(p["golds"][ci][t]), e)] += 1
    k4 = {}
    pols = {"copy": lambda b, e: b, "apply_regardless": lambda b, e: C.heuristics(b, e)["apply_regardless"],
            **C.SHORTCUT_POLICIES}
    for name, f in pols.items():
        ok = 0
        for c in chains:
            st = c["s0"]
            for e in c["events"]:
                st = f(st, e)
                if not 0 <= st < 2 ** C.WIDTH_ID:
                    st = None
                    break
            ok += int(st is not None and st == C.chain_gold_states(c)[-1])
        k4[name] = round(ok / len(chains), 4)
    return {"strata": h, "structural_overlap_with_training_descriptive": struct, "chains": {
        "parents": len(parents), "chains": len(chains), "steps": len(chains) * C.CHAIN_H,
        "final_equals_s0_share (copy floor at k=4)": round(copy4, 4), "step_types": dict(step_types),
        "distinct_finals_per_parent_min": min(len(set(p_g[-1] for p_g in p["golds"])) for p in parents),
        "k4_exact_floors_of_policies (self-rollout; out-of-range = broken)": k4}}


def _carry(osrc: dict) -> tuple:
    return osrc["code_sha256"], osrc["exclusion_sources"], osrc["overlap_counts"], osrc["internal_overlap_operative"]


def canonical_rows_verified(lines: list) -> tuple[list, dict]:
    """The canonical rows, read from data/t2bp_train_dint.jsonl after its sha256 is checked against the canonical
    SOURCE record (never re-drawn: the records tree may have grown since the build)."""
    osrc = json.loads(C.TRAIN_SOURCE.read_text(encoding="utf-8"))
    if C.sha256(C.TRAIN_FILES["dint"]) != osrc["files"][C.TRAIN_FILES["dint"].name]:
        raise SystemExit("data/t2bp_train_dint.jsonl differs from its SOURCE record")
    train = canonical_from_arm_rows(C.load_jsonl(C.TRAIN_FILES["dint"]))
    log(lines, f"canonical rows taken from {C.TRAIN_FILES['dint'].name} (sha verified against {C.TRAIN_SOURCE.name})")
    return train, osrc


def _jsonl_text(rows: list) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def source_record(lines, sd: bool, train, dev, parents, arm_rows, au, shas, code, src_summary, overlaps, internal,
                  count, fillk, extra=None) -> dict:
    common = {"seed": C.SEED, "builder": "scripts/t2bp_make_data.py", "builder_sha256": C.sha256(Path(__file__)),
              "common_sha256": C.sha256(ROOT / "scripts" / "t2bp_common.py"), "code_sha256": code,
              "harvest": ("scripts/l1_make_ood.harvest_historical via scripts/p3_make_bank.load_all_sources: files named "
                          "HSKILL_CONFIRM_V2* are skipped by name before any open; rcwmt.bank.load_exclusions() is NOT called"),
              "exclusion_sources": src_summary, "overlap_counts": overlaps, "internal_overlap_operative": internal,
              "model_calls": 0, "torch_imported": "torch" in sys.modules, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec = {"record": "T2BP_TRAIN", "revision": 3, **common, "surface_diversity": "on" if sd else "off", "files": shas,
           "rows": len(train), "design": {"unpaired": C.TRAIN_UNPAIRED, "pairs": C.TRAIN_PAIRS, "pair_inapp": C.PAIR_INAPP},
           "surface_design": {"paraphrases": list(C.PARAPHRASES), "state_sentences": list(C.STATE_SENTENCES),
                              "field_variants": list(C.FIELD_VARIANTS), "train_index": list(C.TRAIN_SURFACE_IDX),
                              "heldout_index": C.HELDOUT_SURFACE_IDX, "object_slots": {k: list(v) for k, v in C.OBJ_SLOTS.items()},
                              "order_seed": C.ORDER_SEED, "p_identity_order": C.P_IDENTITY_ORDER,
                              "heldout_orders": {k: [list(p) for p in v] for k, v in C.HELDOUT_ORDERS.items()},
                              "heldout_orders_regenerated_from_seed": C.derive_heldout_orders() == C.HELDOUT_ORDERS,
                              "note": C.HELDOUT_ORDER_NOTE, "pair_members_share_one_surface": True,
                              "heldout_stage_E3": C.HELDOUT_SURFACE} if sd else None,
           "filler_4b": fillk, "audit": au, "token_stats": token_stats(arm_rows, dev, parents, count),
           "canonical_dint_sha256": C.sha256(C.TRAIN_FILES["dint"]),
           "targets": {a_: arm_rows[a_][0]["target"] for a_ in C.ARMS}, "example_row": train[0]["id"],
           "system_texts": {a_: C.system_text(a_, C.WIDTH_ID) for a_ in C.ARMS},
           "build_log": lines, **(extra or {})}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-diversity", choices=("off", "on"), default="off")
    ap.add_argument("--rerender-canonical", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    lines: list = []
    count = counter()
    if a.verify:
        return verify(count, lines)
    if a.rerender_canonical:
        return rerender_canonical(count, lines)
    sd = a.surface_diversity == "on"
    files = C.TRAIN_FILES_SD if sd else C.TRAIN_FILES
    source_path = C.TRAIN_SOURCE_SD if sd else C.TRAIN_SOURCE
    outs = list(files.values()) + [source_path] + ([] if sd else [C.DEV, C.DEV_SOURCE])
    if any(p.exists() for p in outs):
        raise SystemExit(f"refusing: outputs exist {[str(p) for p in outs if p.exists()]} (create-only)")
    log(lines, f"T2b prerequisite data: seed {C.SEED}, surface diversity {a.surface_diversity}; no model call, no torch")
    # T2's stored prerequisite prompts must be reproduced byte for byte by the D-int frame
    t2p = [r for r in C.load_jsonl(T2_TRAIN) if r["family"] == P]
    same = sum(C.render_chat(C.messages("dint", r["before"], r["event"], C.WIDTH_ID)) == r["prompt"] for r in t2p)
    if same != len(t2p):
        raise SystemExit(f"the D-int frame reproduces only {same}/{len(t2p)} T2 prerequisite prompts")
    log(lines, f"D-int frame == T2's stored prerequisite prompt on {same}/{len(t2p)} rows (render_chat == the chat template)")
    if C.derive_heldout_orders() != C.HELDOUT_ORDERS:
        raise SystemExit("t2bp_common.HELDOUT_ORDERS differs from derive_heldout_orders() (ORDER_SEED)")
    if sd:
        if not (C.TRAIN_FILES["dint"].exists() and C.DEV.exists()):
            raise SystemExit("the surface-diversity build reads the canonical rows: build the canonical files first")
        train, osrc = canonical_rows_verified(lines)
        if osrc.get("revision") != 3:
            raise SystemExit("the canonical SOURCE is not revision 3: run --rerender-canonical first (amendment 4b filler)")
        dev, parents = C.load_dev()
        code, src_summary, overlaps, internal = _carry(osrc)
    else:
        code, src_summary, overlaps, internal, dev, parents, train = build_everything(lines)
    fillk = check_fill_k(train, dev, parents, count)
    log(lines, f"amendment 4b: constant fill {fillk['fill_k']} units (longest trace head {fillk['longest_trace_head_tokens']} "
               f"tokens over {fillk['items']} train+dev items)")
    surfaces = assign_surfaces(train, sd)
    arm_rows = {arm: render_arm(train, arm, count, surfaces) for arm in C.ARMS}
    au = audit_files(arm_rows, count)
    log(lines, f"audit: ok={au['ok']} heuristics {au['core']['heuristic_accuracy']} quotas {au['core']['quotas']}")
    if not au["ok"]:
        raise SystemExit("AUDIT FAILED - nothing written: " + json.dumps({"checks": au["checks"], "failures": au["failures"]},
                                                                        default=str)[:3000])
    shas = {}
    for arm, p in files.items():
        shas[p.name] = write_create_only(p, arm_rows[arm])
    rec = source_record(lines, sd, train, dev, parents, arm_rows, au, shas, code, src_summary, overlaps, internal, count,
                        fillk)
    if not sd:
        dev_sha = write_create_only(C.DEV, dev + parents)
        drec = {"record": "T2BP_DEV", **{k: rec[k] for k in ("seed", "builder", "builder_sha256", "common_sha256", "code_sha256",
                                                             "harvest", "exclusion_sources", "overlap_counts",
                                                             "internal_overlap_operative", "model_calls", "torch_imported",
                                                             "built_utc")},
                "file": C.DEV.name, "file_sha256": dev_sha, "onestep_rows": len(dev),
                "chain_parents": len(parents), "strata_n": C.DEV_N, "describe": describe_dev(dev, parents, train),
                "use": "T2b prerequisite representation screen (seed 0) only; a dev bank, not a confirmation instrument",
                "build_log": lines}
        with C.DEV_SOURCE.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(drec, indent=1) + "\n")
        rec["dev_sha256"] = dev_sha
    else:
        rec["dev_sha256"] = C.sha256(C.DEV)
        rec["canonical_source_sha256"] = C.sha256(C.TRAIN_SOURCE)
    import psutil
    mi = psutil.Process().memory_info()
    log(lines, f"done in {time.time() - t0:.0f}s; peak RSS {getattr(mi, 'peak_wset', mi.rss) / 2 ** 20:.0f} MiB; files {shas}")
    with source_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, default=str) + "\n")
    return 0


REV2_FILLER = C.DATA / "t2bp_train_ffill.rev2_rowmatched.jsonl"
REV2_SOURCE = C.DATA / "t2bp_train.rev2.SOURCE.json"


def rerender_canonical(count, lines) -> int:
    """Revision 3, once: re-render the canonical files from their own rows. D-int, S-trace and B-bits must come out
    byte-identical (their prompts and targets did not change); the F-filler file gets the amendment-4b constant fill.
    The superseded row-matched filler file and the revision-2 SOURCE are kept, renamed (create-only)."""
    t0 = time.time()
    train, osrc = canonical_rows_verified(lines)
    for a_ in C.ARMS:
        if C.sha256(C.TRAIN_FILES[a_]) != osrc["files"][C.TRAIN_FILES[a_].name]:
            raise SystemExit(f"{C.TRAIN_FILES[a_].name} differs from its SOURCE record")
    if osrc.get("revision") == 3:
        print("the canonical files are already revision 3; nothing to do")
        return 0
    if REV2_FILLER.exists() or REV2_SOURCE.exists():
        raise SystemExit(f"{REV2_FILLER.name} or {REV2_SOURCE.name} exists: resolve by hand")
    dev, parents = C.load_dev()
    fillk = check_fill_k(train, dev, parents, count)
    surfaces = assign_surfaces(train, False)
    arm_rows = {arm: render_arm(train, arm, count, surfaces) for arm in C.ARMS}
    for a_ in ("dint", "strace", "bbits"):
        if _jsonl_text(arm_rows[a_]).encode("utf-8") != C.TRAIN_FILES[a_].read_bytes():
            raise SystemExit(f"re-rendering changed {C.TRAIN_FILES[a_].name}: refusing (only the filler may change)")
    au = audit_files(arm_rows, count)
    if not au["ok"]:
        raise SystemExit("AUDIT FAILED - nothing written: " + json.dumps({"checks": au["checks"], "failures": au["failures"]},
                                                                        default=str)[:3000])
    log(lines, f"re-render: D-int/S-trace/B-bits byte-identical; F-filler -> constant fill {fillk['fill_k']} (amendment 4b)")
    os.replace(C.TRAIN_FILES["ffill"], REV2_FILLER)
    os.replace(C.TRAIN_SOURCE, REV2_SOURCE)
    shas = {C.TRAIN_FILES[a_].name: C.sha256(C.TRAIN_FILES[a_]) for a_ in ("dint", "strace", "bbits")}
    shas[C.TRAIN_FILES["ffill"].name] = write_create_only(C.TRAIN_FILES["ffill"], arm_rows["ffill"])
    code, src_summary, overlaps, internal = _carry(osrc)
    log(lines, f"done in {time.time() - t0:.0f}s; files {shas}")
    rec = source_record(lines, False, train, dev, parents, arm_rows, au, shas, code, src_summary, overlaps, internal,
                        count, fillk, extra={
                            "dev_sha256": C.sha256(C.DEV),
                            "supersedes": {"record": REV2_SOURCE.name, "record_sha256": C.sha256(REV2_SOURCE),
                                           "filler_file": REV2_FILLER.name, "filler_sha256": C.sha256(REV2_FILLER),
                                           "reason": ("amendment 4b (DECISIONS D-36): the revision-2 filler matched its "
                                                      "length per row to the trace, which leaks the answer through the "
                                                      "length; the canonical rows, the D-int/S-trace/B-bits files and "
                                                      "the exclusion record are unchanged (carried over)")}})
    with C.TRAIN_SOURCE.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, default=str) + "\n")
    return 0


def verify(count, lines) -> int:
    res = {}
    for variant, files, sp in (("canonical", C.TRAIN_FILES, C.TRAIN_SOURCE), ("sd", C.TRAIN_FILES_SD, C.TRAIN_SOURCE_SD)):
        src = json.loads(sp.read_text(encoding="utf-8"))
        au = audit_files({a: files[a] for a in C.ARMS}, count, source=src)
        res[variant] = {"audit_ok": au["ok"], "checks": au["checks"]}
    sdsrc = json.loads(C.TRAIN_SOURCE_SD.read_text(encoding="utf-8"))
    canon = canonical_from_arm_rows(C.load_jsonl(C.TRAIN_FILES["dint"]))
    link_ok = (sdsrc.get("canonical_dint_sha256") == C.sha256(C.TRAIN_FILES["dint"]) and
               sdsrc.get("canonical_source_sha256") == C.sha256(C.TRAIN_SOURCE) and
               canonical_from_arm_rows(C.load_jsonl(C.TRAIN_FILES_SD["dint"])) == canon and
               all([r["target"] for r in C.load_jsonl(C.TRAIN_FILES_SD[a])] == [r["target"] for r in C.load_jsonl(C.TRAIN_FILES[a])]
                   for a in C.ARMS))
    dsrc = json.loads(C.DEV_SOURCE.read_text(encoding="utf-8"))
    dev_ok = C.sha256(C.DEV) == dsrc["file_sha256"]
    rows, parents = C.load_dev()
    gold_ok = all(check_row(r["before"], r["event"]) == r["gold"] for r in rows)
    for p in parents:
        for ci, c in enumerate(p["candidates"]):
            gold_ok &= [env.dumps(s) for s in env.rollout(P, p["s0"], c["events"])] == p["golds"][ci]
    fill_ok = check_fill_k(canon, rows, parents, count)["fill_k"] == C.FILL_K
    orders_ok = C.derive_heldout_orders() == C.HELDOUT_ORDERS
    # freshness, re-checked against a FRESH name-skipping harvest (the records tree may have grown since the build)
    items = [(r["before"], r["event"]) for r in rows + canon]
    for p in parents:
        for c in p["candidates"]:
            st = env.rollout(P, p["s0"], c["events"])
            items += [(st[t], e) for t, e in enumerate(c["events"])]
    srcs = load_sources(lines)
    overlaps = {name: {"payload": sum(C.payload(b, e) in s["payloads"] for b, e in items),
                       "operative": sum(C.opkey(b, e) in s["ops"] for b, e in items)} for name, s in srcs.items()}
    tr_ops = {C.opkey(r["before"], r["event"]) for r in canon}
    dev_ops = {C.opkey(r["before"], r["event"]) for r in rows}
    ch_ops = {C.opkey(b, e) for b, e in items[len(rows) + len(canon):]}
    internal = {"train&dev": len(tr_ops & dev_ops), "train&chains": len(tr_ops & ch_ops), "dev&chains": len(dev_ops & ch_ops)}
    fresh_ok = not any(v["payload"] or v["operative"] for v in overlaps.values()) and not any(internal.values())
    ok = (all(v["audit_ok"] for v in res.values()) and link_ok and dev_ok and gold_ok and fresh_ok and fill_ok and orders_ok)
    print(json.dumps({"ok": ok, "audits": res, "sd_rows_and_targets_equal_canonical": link_ok, "dev_sha_ok": dev_ok,
                      "dev_golds_ok": gold_ok, "fill_k_ok": fill_ok, "heldout_orders_regenerate": orders_ok,
                      "fresh_ok": fresh_ok, "sources": len(srcs), "items_checked": len(items),
                      "overlaps_nonzero": {k: v for k, v in overlaps.items() if v["payload"] or v["operative"]},
                      "internal_overlap_operative": internal}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
