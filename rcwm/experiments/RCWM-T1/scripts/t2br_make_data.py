"""T2b-register screen data, revision 5: fresh oracle register rows rendered three ways (plain / trace / filler), in two
prompt renderings (canonical 'base' = the rows' source of truth; surface-diversity 'sd' = the screen's ONLY training
data, decision C1), the dev screen bank (one-step strata + k=4 chains) and its held-out-surface rendering (C3). Built
WITHOUT any model call and without torch (Titan's tokenizer is read through the `tokenizers` library only, to count
tokens). Create-only.

  python -B scripts/t2br_make_data.py                    # writes data/t2br_train_{plain,trace,filler}.jsonl (canonical
                                                         # prompts) + data/t2br_train.SOURCE.json,
                                                         # data/t2br_train_sd_{plain,trace,filler}.jsonl (the SAME rows,
                                                         # surface-diversity prompts) + data/t2br_train_sd.SOURCE.json,
                                                         # data/t2br_dev.jsonl + data/t2br_dev_hs.jsonl +
                                                         # data/t2br_chains.jsonl + data/t2br_dev.SOURCE.json
  python -B scripts/t2br_make_data.py --verify [--no-historical]
                                                         # re-derives everything from the files (golds, targets, the 4b
                                                         # filler constants, the surface proof, freshness incl. a fresh
                                                         # name-skipping harvest, the audits), REBUILDS every file in
                                                         # memory and compares bytes; writes docs/t2b/register/AUDIT.json

TRAINING ROWS (750, register only; values / thresholds / registers from the recorded CANDIDATE_VIEWS marginals, so
-8 <= value <= 11 and 0 <= threshold <= 96). UNCHANGED from revision 2 (the build asserts the revision-2 row digests;
--verify also reconstructs the four revision-2 files byte for byte from the current rows):
  mix 300        150 wrap / 150 no-wrap (raw r + 2 o + v outside / inside [0, 96]); the state always changes
  wrap 150       add whose raw r + v leaves [0, 96]: 75 over (v > 0) / 75 under (v < 0)
  conditional 300 = 150 core rows at other = threshold - 1 / threshold / threshold + 1 (50 each; value != 0; the
                 roadmap's rows) + 150 MINIMAL-PAIR TWINS, each differing from its core row in ONE number and on the
                 OTHER side of the threshold at distance d uniform in 2..15 (t2br_common.TWIN_DIST):
                   core ok false (x = thr - 1 or thr) -> twin ok true,  x - thr' = +d  (or x' - thr = +d)
                   core ok true  (x = thr + 1)        -> twin ok false, x - thr' = -d  (or x' - thr = -d)
                 Per core offset 25 twins move the threshold and 25 move the other register (moving only the
                 threshold would make the threshold alone predictive of the branch). So add / subtract is exactly
                 150 / 150; no single surface field separates the branches; and the near-only rules that fit the core
                 rows perfectly (other == threshold + 1; unit-digit comparisons) are contradicted by the twins. Pair
                 wrap patterns are balanced (1/3 neither member wraps, 1/3 the add member wraps, 1/3 the subtract
                 member wraps), which keeps the unreduced-sum shortcut below the audit bar overall.
  Swap and add/no-wrap are deliberately absent (untouched skills = the screen's controls).
SURFACE DIVERSITY (revision 3, C1 + C2). Each training row gets ONE seeded surface (random.Random(SEED_SD), shared by
its three arms): a rule-text paraphrase from SD_PARAPHRASES[0..3], a field-name variant per field from SD_FIELDS[f][0..3]
and per JSON object type (top level / state / event) a key order from t2br_common.TRAIN_ORDERS. Paraphrase 4, field-name
variant 4 and the HELD-OUT key orders (reversed + 2 seeded permutations per type) never occur in training; --verify
decodes every training prompt (t2br_common.read_prompt) to prove it. Targets are unchanged.
FILLER (C4 = amendment 4b; the constant definition implemented in revision 4). fill = t2br_common.FILL_UNITS[kind]
units of " .": the smallest run such that the filler target is at least as long (Titan tokens) as the LONGEST trace
target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after'
(t2br_common.min_constant_fill: the kind's shortest filler target against its longest trace target); re-derived here from
the data and asserted equal to the pre-stated constants. Revision 3's code compared each row's filler with that row's
trace (FILL_UNITS_REV3 5 / 14 / 14 / 26); --verify reconstructs the revision-3 files from the current rows with those.

DEV SCREEN BANK (t2br_common.DEV_STRATA; 1,280 one-step rows + 48 chain parents x 8 candidates x 4 steps; the first
1,100 rows unchanged from revision 2, the first 1,200 byte for byte as in revision 4). In range: mix 100, wrap 80, cond_xm1 / cond_x0 / cond_xp1 60 each (60 TRIPLETS:
same row, threshold = x + 1 / x / x - 1), cond_mid 100 (50 pairs at |x - thr| 2..15), swap 80, add_nowrap 80. Out of
range (PRIMARY): mix_oor 100 and wrap_oor 80 (|value| 12..40), cond_oor_val 100 (50 boundary pairs, |value| 12..40),
cond_oor_thr 100 (50 pairs, threshold -40..-1 vs 97..136), cond_far 100 (50 pairs at |x - thr| 16..96, in-range values
and thresholds). IDENTIFICATION (revision 4, G1 alone): cond_sametens 100 (50 minimal pairs with tens(x) == tens(thr),
|x - thr| 2..9, one ok true / one ok false; |value| 12..40; neither branch wraps; 25 pairs move the threshold, 25 the
other register). It is drawn AFTER the chains with its own seed (SEED_SAMETENS) against the full registry (training,
dev, chains, exclusion union), so every earlier row and the chain bank are unchanged; its rows are appended (dv1100..).
IDENTIFICATION (revision 5, G1 alone): cond_oor_edge 80 (20 minimal QUADRUPLES at other - threshold = -1, 0 (ok false)
/ +1, +2 (ok true): the boundary of the strict predicate, so that with cond_oor_val's (0, +1) pairs G1 sees rows at
other = threshold - 1, threshold and threshold + 1 and every off-by-one variant is contradicted on >= 1/4 of the
stratum; |value| 12..40; neither branch wraps; in-range registers and thresholds; 10 quadruples move the threshold, 10
the other register). It is drawn AFTER cond_sametens with its own seed (SEED_EDGE) against the full registry, so every
earlier row (dv0000..dv1199), the chain bank and every training file are unchanged; its rows are appended
(dv1200..dv1279). --verify reconstructs the eight revision-4 data files byte for byte from the current rows.
Chains: CH_IN 32 parents (env.gen_event's recorded marginals) and CH_X 16 parents (the same, every add / mix /
conditional value replaced by |v| 12..40); each parent needs >= 4 distinct final states among its 8 candidates.
HELD-OUT-SURFACE BANK data/t2br_dev_hs.jsonl (C3): the 580 rows of the primary strata (mix_oor, cond_oor_val,
cond_oor_thr, cond_far) and of the identification strata (cond_sametens, cond_oor_edge) in dev order, the j-th rendered
in t2br_common.heldout_surface(j) (paraphrase 4 + names 4 + held-out key orders rotating by row); same ids and golds as
the in-surface rows (paired by row). Its first 400 rows are revision 3's bank and its first 500 revision 4's, byte for
byte.

--verify also REBUILDS every data file in memory with the current builder (the same fresh union) and requires byte
equality with the files on disk, so the provenance of the data never depends on a stale builder hash.

FRESHNESS. No (before, event) payload and no FDS operative content (generate.operative_key) of any training row, dev row
or chain transition equals one in the union used by PREREG-3's builder (scripts/p3_make_bank.load_all_sources: the
records tree harvested by scripts/l1_make_ood.harvest_historical, which skips files named HSKILL_CONFIRM_V2* BY NAME
before any open - rcwmt.bank.load_exclusions() is NOT called; every transition of bank_pilot / bank_confirm /
bank_confirm2 / bank_s4000; t2_train / t2_eval / t2_eval_xframe / t2_eval_supp; SELECT-VIEWS-001; the L1 OOD bank; the
FDS-era T_BANK* files), nor any transition of a multi-event register prompt in those files (rolled out with env.step).
The PREREG-3 bank is ledger-only (p3_common.LEDGER) and cannot share a register payload; it is not opened. Training,
dev and chain units are mutually disjoint at payload and operative-content level. (Freshness is a property of the
payloads; the surface rendering does not change them.)
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
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2br_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

FAM = C.FAM
SEED_TRAIN = 20261109            # not used by any earlier RCWM-T1 builder, analysis or bootstrap
SEED_DEV = 20261110
SEED_CHAINS = 20261111
SEED_SD = 20261112
SEED_SAMETENS = 20261114         # revision 4: cond_sametens (20261113 is t2br_common.HELDOUT_ORDER_SEED)
SEED_EDGE = 20261115             # revision 5: cond_oor_edge
SHUFFLE_SEED = 7                 # as make_t2_data.py
MAX_DRAWS = 5_000_000
TRAIN_QUOTA = OrderedDict([("mix/wrap", 150), ("mix/nowrap", 150), ("add/wrap_over", 75), ("add/wrap_under", 75)])
COND_OFFSETS = (-1, 0, 1)
COND_PATTERNS = ("none", "add_wraps", "sub_wraps")
COND_PATTERN_QUOTA = {-1: (17, 17, 16), 0: (17, 16, 17), 1: (16, 17, 17)}   # 50 pairs per core offset
TWIN_TYPES = ("threshold", "other")                                         # the one number the twin moves
TWIN_TYPE_QUOTA = 25                                                         # per core offset and twin type
DEV_PAIR_TYPE_QUOTA = 25                                                     # cond_mid: per pair type
EDGE_TYPE_QUOTA = 10                                                         # cond_oor_edge: quadruples per type
MIN_DISTINCT_FINALS = 4
T2_TRAIN = ROOT / "data" / "t2_train.jsonl"
T2_TRAIN_SHA = "bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744"


def log(lines: list, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    lines.append(line)
    print(line, flush=True)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------------------------- exclusion union
def _rollout_register(state, events):
    """(state_t, event_t) of a register multi-event prompt rolled out with env.step; stops at the first invalid step."""
    if not isinstance(state, dict) or not state or not all(_int(v) for v in state.values()):
        return
    for ev in events:
        if not isinstance(ev, dict) or ev.get("kind") not in ("swap", "add", "mix", "conditional") \
                or ev.get("register") not in state:
            return
        if ev["kind"] != "add" and ev.get("other") not in state:
            return
        if ev["kind"] in ("add", "mix", "conditional") and not _int(ev.get("value")):
            return
        if ev["kind"] == "conditional" and not _int(ev.get("threshold")):
            return
        yield state, ev
        state = env.step(FAM, state, ev)


def multi_event_register(L1, MB, lines: list) -> dict:
    """Every transition of every multi-event register prompt ({before|s0, events}) in the census files (the records tree
    walked with the same name-skipping rule, plus the pinned sources outside it)."""
    files, skipped = MB.census_files()
    pay, ops = set(), set()
    n_obj = n_files = 0
    for p in files:
        if p.name.startswith(L1.RESERVED_PREFIX):
            raise SystemExit("a reserved file reached the reader")          # unreachable by construction
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if '"events"' not in raw and '\\"events\\"' not in raw:
            continue
        docs = []
        if p.suffix == ".jsonl":
            for line in raw.splitlines():
                if '"events' in line or '\\"events' in line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                docs.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        del raw
        got = 0
        for doc in docs:
            for st, evs in MB._multi_event_objects(doc):
                try:
                    for b, e in _rollout_register(st, evs):
                        pay.add(L1.payload(b, e))
                        ops.add(L1.opkey(FAM, b, e))
                        got += 1
                except (KeyError, TypeError, ValueError):
                    continue
                n_obj += 1
        n_files += got > 0
        del docs
    log(lines, f"multi-event register prompts: {len(pay)} transitions from {n_files} files ({len(files)} scanned); "
               f"skipped by name, never opened: {skipped}")
    return {"payloads": pay, "ops": ops, "sha256": None, "files_scanned": len(files),
            "files_with_register_rollouts": n_files, "skipped_by_name_never_opened": skipped}


def load_union(lines: list, historical: bool = True) -> tuple[dict, set, set]:
    import l1_make_ood as L1        # name-skipping harvest (DECISIONS I-28); torch-free
    import p3_make_bank as MB       # PREREG-3's exclusion union; torch-free
    src = MB.load_all_sources(lines, historical=historical)
    src["multi_event_register_prompts"] = multi_event_register(L1, MB, lines)
    pay, ops = set(), set()
    for v in src.values():
        pay |= v["payloads"]
        ops |= v["ops"]
    log(lines, f"exclusion union: {len(pay)} payloads, {len(ops)} operative contents ({len(src)} sources)")
    return src, pay, ops


class Registry:
    """Freshness against the union and uniqueness across every unit of this build (payload + operative content)."""

    def __init__(self, excl_pay: set, excl_ops: set):
        import l1_make_ood as L1
        self.L1 = L1
        self.excl_pay, self.excl_ops = excl_pay, excl_ops
        self.pay, self.ops = {}, {}

    def keys(self, before, ev):
        return self.L1.payload(before, ev), self.L1.opkey(FAM, before, ev)

    def problem(self, items, owner: str):
        seen_p, seen_o = set(), set()
        for b, e in items:
            p, o = self.keys(b, e)
            if p in self.excl_pay:
                return "payload in an exclusion source"
            if o in self.excl_ops:
                return "operative content in an exclusion source"
            if self.pay.get(p, owner) != owner or self.ops.get(o, owner) != owner:
                return "shared with another unit of this build"
            seen_p.add(p)
            seen_o.add(o)
        return None

    def add(self, items, owner: str) -> None:
        for b, e in items:
            p, o = self.keys(b, e)
            self.pay[p] = owner
            self.ops[o] = owner


# ---------------------------------------------------------------------------------------------- draws
class Draw:
    def __init__(self, rng: random.Random, m):
        self.rng, self.m = rng, m
        self.values = list(m.r_values)
        self.values_nz = [v for v in m.r_values if v % C.MOD]
        self.thresholds = list(m.r_thresholds)
        self.befores = list(m.r_befores)

    def before(self) -> dict:
        return {r: self.rng.choice(self.befores) for r in C.REGS}

    def oor_value(self) -> int:
        return self.rng.choice((-1, 1)) * self.rng.randint(*C.OOR_V_ABS)

    def event(self, kind: str, value: int, threshold: int) -> dict:
        r = self.rng.choice(C.REGS)
        o = self.rng.choice([x for x in C.REGS if x != r])
        return {"kind": kind, "other": o, "register": r, "threshold": threshold, "tick": self.rng.randint(1, 99),
                "value": value}


def _pattern(b_add: dict, ev_add: dict, b_sub: dict, ev_sub: dict) -> str | None:
    wa, ws = C.wraps(b_add, ev_add), C.wraps(b_sub, ev_sub)
    if wa and ws:
        return None
    return "add_wraps" if wa else ("sub_wraps" if ws else "none")


def build_train(d: Draw, reg: Registry, lines: list) -> list:
    rows, rej, draws = [], Counter(), 0
    need = dict(TRAIN_QUOTA)
    while any(need.values()):
        draws += 1
        if draws > MAX_DRAWS:
            raise SystemExit(f"train quota not filled: {need}")
        kind = "mix" if need["mix/wrap"] or need["mix/nowrap"] else "add"
        b = d.before()
        ev = d.event(kind, d.rng.choice(d.values), d.rng.choice(d.thresholds))
        if kind == "mix":
            s = "mix/wrap" if C.wraps(b, ev) else "mix/nowrap"
        else:
            raw = C.raw_value(b, ev)
            s = "add/wrap_over" if raw >= C.MOD else ("add/wrap_under" if raw < 0 else None)
        if s is None or not need.get(s):
            rej["stratum full / not a wrap"] += 1
            continue
        if env.step(FAM, b, ev) == b:
            rej["state unchanged"] += 1
            continue
        uid = f"tr{len(rows):04d}"
        pr = reg.problem([(b, ev)], uid)
        if pr:
            rej[pr] += 1
            continue
        reg.add([(b, ev)], uid)
        rows.append({"uid": uid, "category": kind if kind == "mix" else "wrap", "stratum": s, "cluster": uid,
                     "role": "single", "before": b, "event": ev})
        need[s] -= 1
    log(lines, f"train mix/wrap: {len(rows)} rows from {draws} draws; rejected {dict(rej)}")
    # conditional core (roadmap: other = thr - 1 / thr / thr + 1) + minimal-pair twins at distance 2..15 on the other side
    pq = {off: dict(zip(COND_PATTERNS, q)) for off, q in COND_PATTERN_QUOTA.items()}
    tq = {off: {t: TWIN_TYPE_QUOTA for t in TWIN_TYPES} for off in COND_OFFSETS}
    rej, draws, npair = Counter(), 0, 0
    name = {-1: "xm1", 0: "x0", 1: "xp1"}
    while any(n for q in pq.values() for n in q.values()):
        draws += 1
        if draws > MAX_DRAWS:
            raise SystemExit(f"conditional quota not filled: {pq} {tq}")
        off = d.rng.choice([o for o in COND_OFFSETS if any(pq[o].values())])
        thr = d.rng.choice(d.thresholds)
        x = thr + off
        if not C.THR_IN[0] <= x <= C.THR_IN[1]:
            rej["x outside 0..96"] += 1
            continue
        ev = d.event("conditional", d.rng.choice(d.values_nz), thr)
        b = d.before()
        b[ev["other"]] = x
        ttype = d.rng.choice([t for t in TWIN_TYPES if tq[off][t]] or list(TWIN_TYPES))
        dist = d.rng.randint(*C.TWIN_DIST)
        core_ok = C.branch_ok(b, ev)
        if ttype == "threshold":
            b2, ev2 = dict(b), dict(ev, threshold=x + dist if core_ok else x - dist)
        else:
            b2, ev2 = dict(b, **{ev["other"]: thr - dist if core_ok else thr + dist}), dict(ev)
        if not (C.THR_IN[0] <= ev2["threshold"] <= C.THR_IN[1] and C.THR_IN[0] <= b2[ev["other"]] <= C.THR_IN[1]):
            rej["twin threshold / other outside 0..96"] += 1
            continue
        if C.branch_ok(b2, ev2) == core_ok or abs(C.cond_dist(b2, ev2)) != dist \
                or C.single_diff((b, ev), (b2, ev2)) != ttype:
            raise AssertionError("twin construction")
        (ba, ea), (bs, es) = ((b, ev), (b2, ev2)) if core_ok else ((b2, ev2), (b, ev))
        pat = _pattern(ba, ea, bs, es)
        if pat is None or not pq[off][pat] or not tq[off][ttype]:
            rej["wrap pattern / twin type full, or both wrap"] += 1
            continue
        cid = f"tp{npair:03d}"
        pr = reg.problem([(b, ev), (b2, ev2)], cid)
        if pr:
            rej[pr] += 1
            continue
        reg.add([(b, ev), (b2, ev2)], cid)
        rows.append({"uid": f"{cid}a", "category": "cond", "stratum": f"cond/{name[off]}", "cluster": cid, "role": "core",
                     "offset": off, "pattern": pat, "twin_type": ttype, "before": b, "event": ev})
        off2 = C.cond_dist(b2, ev2)
        rows.append({"uid": f"{cid}b", "category": "cond", "stratum": "cond/twin_gt" if off2 > 0 else "cond/twin_le",
                     "cluster": cid, "role": "twin", "offset": off2, "pattern": pat, "twin_type": ttype, "before": b2,
                     "event": ev2})
        pq[off][pat] -= 1
        tq[off][ttype] -= 1
        npair += 1
    log(lines, f"train conditional: {npair} pairs ({2 * npair} rows) from {draws} draws; rejected {dict(rej)}")
    return rows


def _single(d: Draw, reg: Registry, owner: str, make, accept, rej: Counter, max_draws: int = MAX_DRAWS):
    for _ in range(max_draws):
        b, ev = make()
        why = accept(b, ev)
        if why:
            rej[why] += 1
            continue
        pr = reg.problem([(b, ev)], owner)
        if pr:
            rej[pr] += 1
            continue
        reg.add([(b, ev)], owner)
        return b, ev
    raise SystemExit(f"could not draw a row for {owner}: {dict(rej)}")


def build_dev(d: Draw, reg: Registry, lines: list) -> list:
    rows = []
    rej = Counter()

    def add_row(stratum, b, ev, cluster, role, **extra):
        rows.append({"uid": f"dv{len(rows):04d}", "stratum": stratum, "group": C.DEV_STRATA[stratum][0],
                     "cluster": cluster, "role": role, "before": b, "event": ev, **extra})

    def mix_like(stratum, oor):
        for want_wrap, n in ((True, C.DEV_N[stratum] // 2), (False, C.DEV_N[stratum] - C.DEV_N[stratum] // 2)):
            for _ in range(n):
                uid = f"dv{len(rows):04d}"
                b, ev = _single(d, reg, uid,
                                lambda: (d.before(), d.event("mix", d.oor_value() if oor else d.rng.choice(d.values),
                                                             d.rng.choice(d.thresholds))),
                                lambda b, ev: None if (C.wraps(b, ev) == want_wrap and env.step(FAM, b, ev) != b)
                                else "wrap status / unchanged", rej)
                add_row(stratum, b, ev, uid, "single", wrap=want_wrap)

    def wrap_like(stratum, oor):
        half = C.DEV_N[stratum] // 2
        for sign, n in ((1, half), (-1, C.DEV_N[stratum] - half)):
            for _ in range(n):
                uid = f"dv{len(rows):04d}"

                def make(sign=sign):
                    v = abs(d.oor_value()) * sign if oor else d.rng.choice([x for x in d.values if x * sign > 0])
                    return d.before(), d.event("add", v, d.rng.choice(d.thresholds))
                b, ev = _single(d, reg, uid, make, lambda b, ev: None if C.wraps(b, ev) else "no wrap", rej)
                add_row(stratum, b, ev, uid, "single", wrap=True, direction="over" if sign > 0 else "under")

    mix_like("mix", False)
    wrap_like("wrap", False)
    # in-range conditional triplets: one row per threshold x + 1 / x / x - 1
    for t in range(C.DEV_N["cond_x0"]):
        cid = f"dt{t:03d}"
        for _ in range(MAX_DRAWS):
            ev = d.event("conditional", d.rng.choice(d.values_nz), 0)
            b = d.before()
            x = b[ev["other"]]
            if not 1 <= x <= 95:
                rej["x outside 1..95"] += 1
                continue
            trip = [(b, dict(ev, threshold=x + 1)), (b, dict(ev, threshold=x)), (b, dict(ev, threshold=x - 1))]
            pr = reg.problem(trip, cid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(trip, cid)
            for (bb, ee), name in zip(trip, ("cond_xm1", "cond_x0", "cond_xp1")):
                add_row(name, dict(bb), ee, cid, "triplet", offset=bb[ee["other"]] - ee["threshold"])
            break
        else:
            raise SystemExit("triplets not filled")
    # controls
    for _ in range(C.DEV_N["swap"]):
        uid = f"dv{len(rows):04d}"
        b, ev = _single(d, reg, uid, lambda: (d.before(), d.event("swap", d.rng.choice(d.values), d.rng.choice(d.thresholds))),
                        lambda b, ev: None if b[ev["register"]] != b[ev["other"]] else "equal registers", rej)
        add_row("swap", b, ev, uid, "single")
    for _ in range(C.DEV_N["add_nowrap"]):
        uid = f"dv{len(rows):04d}"
        b, ev = _single(d, reg, uid, lambda: (d.before(), d.event("add", d.rng.choice(d.values_nz), d.rng.choice(d.thresholds))),
                        lambda b, ev: None if not C.wraps(b, ev) else "wraps", rej)
        add_row("add_nowrap", b, ev, uid, "single", wrap=False)
    # out of range
    mix_like("mix_oor", True)
    wrap_like("wrap_oor", True)
    for t in range(C.DEV_N["cond_oor_val"] // 2):           # boundary pairs: x = thr (ok false) / thr - 1 (ok true)
        cid = f"dpv{t:03d}"
        for _ in range(MAX_DRAWS):
            thr = d.rng.choice([x for x in d.thresholds if x >= 1])
            ev = d.event("conditional", d.oor_value(), thr)
            b = d.before()
            b[ev["other"]] = thr
            pair = [(b, ev), (b, dict(ev, threshold=thr - 1))]
            pr = reg.problem(pair, cid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(pair, cid)
            for bb, ee in pair:
                add_row("cond_oor_val", dict(bb), ee, cid, "pair", offset=bb[ee["other"]] - ee["threshold"])
            break
        else:
            raise SystemExit("cond_oor_val not filled")
    for t in range(C.DEV_N["cond_oor_thr"] // 2):           # thresholds outside the register range, both sides
        cid = f"dpt{t:03d}"
        for _ in range(MAX_DRAWS):
            ev = d.event("conditional", d.rng.choice(d.values_nz), d.rng.randint(*C.OOR_THR[0]))
            b = d.before()
            pair = [(b, ev), (b, dict(ev, threshold=d.rng.randint(*C.OOR_THR[1])))]
            pr = reg.problem(pair, cid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(pair, cid)
            for bb, ee in pair:
                add_row("cond_oor_thr", dict(bb), ee, cid, "pair", thr_band="low" if ee["threshold"] < 0 else "high")
            break
        else:
            raise SystemExit("cond_oor_thr not filled")

    def dist_pairs(stratum, prefix, band, types):
        """Branch-balanced minimal pairs with BOTH members at |other - threshold| in band: one ok true, one ok false.
        'threshold' pairs share the row and move the threshold; 'other' pairs share the event and move the other
        register. Distances are uniform over what the 0..96 ranges allow."""
        lo, hi = band
        for t, ptype in enumerate(types):
            cid = f"{prefix}{t:03d}"
            for _ in range(MAX_DRAWS):
                ev = d.event("conditional", d.rng.choice(d.values_nz), 0)
                b = d.before()
                o = ev["other"]
                if ptype == "threshold":
                    x = b[o]
                    t_lo, t_hi = (max(0, x - hi), x - lo), (x + lo, min(96, x + hi))
                    if t_lo[0] > t_lo[1] or t_hi[0] > t_hi[1]:
                        rej[f"{stratum}: x leaves no room"] += 1
                        continue
                    pair = [(b, dict(ev, threshold=d.rng.randint(*t_lo))), (b, dict(ev, threshold=d.rng.randint(*t_hi)))]
                else:
                    thr = d.rng.choice(d.thresholds)
                    x_hi_, x_lo_ = (thr + lo, min(96, thr + hi)), (max(0, thr - hi), thr - lo)
                    if x_hi_[0] > x_hi_[1] or x_lo_[0] > x_lo_[1]:
                        rej[f"{stratum}: threshold leaves no room"] += 1
                        continue
                    ev = dict(ev, threshold=thr)
                    pair = [(dict(b, **{o: d.rng.randint(*x_hi_)}), ev), (dict(b, **{o: d.rng.randint(*x_lo_)}), ev)]
                if C.single_diff(*pair) != ptype or [C.branch_ok(*p) for p in pair] != [True, False]:
                    raise AssertionError(f"{stratum} pair construction")
                pr = reg.problem(pair, cid)
                if pr:
                    rej[pr] += 1
                    continue
                reg.add(pair, cid)
                for bb, ee in pair:
                    add_row(stratum, dict(bb), ee, cid, "pair", pair_type=ptype, offset=C.cond_dist(bb, ee))
                break
            else:
                raise SystemExit(f"{stratum} not filled")
    # cond_mid: the training twins' mix of pair types; cond_far: every pair moves the other register (a far pair that
    # moves the threshold is separated by the threshold alone: 'threshold < 48' would score ~0.96 on it)
    dist_pairs("cond_mid", "dmd", C.TWIN_DIST, [t for t in TWIN_TYPES for _ in range(DEV_PAIR_TYPE_QUOTA)])
    dist_pairs("cond_far", "dfr", C.FAR_DIST, ["other"] * (C.DEV_N["cond_far"] // 2))
    log(lines, f"dev: {len(rows)} rows {dict(Counter(r['stratum'] for r in rows))}; rejected {dict(rej)}")
    return rows


def sametens_problem(b: dict, ev: dict) -> str | None:
    """Why a conditional is NOT a cond_sametens row (None = it is one): tens(other) == tens(threshold), |other -
    threshold| in SAMETENS_DIST, |value| in OOR_V_ABS, in-range threshold and registers, neither branch wraps."""
    x, t, v, r = b[ev["other"]], ev["threshold"], ev["value"], b[ev["register"]]
    if ev["kind"] != "conditional" or x // 10 != t // 10:
        return "not same tens"
    if not C.SAMETENS_DIST[0] <= abs(x - t) <= C.SAMETENS_DIST[1]:
        return "distance"
    if not C.OOR_V_ABS[0] <= abs(v) <= C.OOR_V_ABS[1]:
        return "value"
    if not (C.THR_IN[0] <= t <= C.THR_IN[1] and all(C.THR_IN[0] <= y <= C.THR_IN[1] for y in b.values())):
        return "threshold / register range"
    if not (0 <= r - abs(v) and r + abs(v) <= C.MOD - 1):
        return "a branch wraps"
    return None


def build_sametens(d: Draw, reg: Registry, lines: list, start: int) -> list:
    """Revision 4: the identification stratum cond_sametens (G1, decided alone). 50 branch-balanced minimal pairs whose
    two members have tens(other) == tens(threshold), so the env's 'other > threshold' is decided by the units digit;
    |other - threshold| uniform over what the decade allows in SAMETENS_DIST (2..9), one member ok true, one ok false
    (both members lie in the ONE decade of the number they share, so d_true + |d_false| <= 9: the realized distances
    are 2..7, skewed to the small end);
    an out-of-range |value| 12..40 (both signs) with the updated register at |value|..96-|value| so that NEITHER branch
    wraps (a row's exact 'after' is right iff its branch is, given the in-branch arithmetic: the stratum isolates the
    comparison from the modular reduction that mix_oor / wrap_oor / cond_oor_* test). 25 pairs share the row and move
    the threshold, 25 share the event and move the other register (cond_mid's split). Drawn after the chains, with its
    own seed, against the full registry (exclusion union + training + dev + chains): fresh, and disjoint from every
    training payload and operative content. Rows are appended to the dev bank (uids from dv{start})."""
    rows, rej = [], Counter()
    lo, hi = C.SAMETENS_DIST
    types = [t for t in TWIN_TYPES for _ in range(DEV_PAIR_TYPE_QUOTA)]
    for t, ptype in enumerate(types):
        cid = f"dst{t:03d}"
        for _ in range(MAX_DRAWS):
            ev = d.event("conditional", d.oor_value(), 0)
            b = d.before()
            o, r, av = ev["other"], ev["register"], abs(ev["value"])
            if not av <= b[r] <= C.MOD - 1 - av:
                rej["a branch would wrap"] += 1
                continue
            if ptype == "threshold":
                x = b[o]
                dec = x - x % 10
                t_true, t_false = (max(dec, x - hi), x - lo), (x + lo, min(dec + 9, C.THR_IN[1], x + hi))
                if t_true[0] > t_true[1] or t_false[0] > t_false[1]:
                    rej["x leaves no same-tens room"] += 1
                    continue
                pair = [(b, dict(ev, threshold=d.rng.randint(*t_true))), (b, dict(ev, threshold=d.rng.randint(*t_false)))]
            else:
                thr = d.rng.choice(d.thresholds)
                dec = thr - thr % 10
                x_true, x_false = (thr + lo, min(dec + 9, C.THR_IN[1], thr + hi)), (max(dec, thr - hi), thr - lo)
                if x_true[0] > x_true[1] or x_false[0] > x_false[1]:
                    rej["threshold leaves no same-tens room"] += 1
                    continue
                ev = dict(ev, threshold=thr)
                pair = [(dict(b, **{o: d.rng.randint(*x_true)}), ev), (dict(b, **{o: d.rng.randint(*x_false)}), ev)]
            if C.single_diff(*pair) != ptype or [C.branch_ok(*p) for p in pair] != [True, False] \
                    or any(sametens_problem(*p) for p in pair):
                raise AssertionError("cond_sametens pair construction")
            pr = reg.problem(pair, cid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(pair, cid)
            for bb, ee in pair:
                rows.append({"uid": f"dv{start + len(rows):04d}", "stratum": "cond_sametens",
                             "group": C.DEV_STRATA["cond_sametens"][0], "cluster": cid, "role": "pair", "before": dict(bb),
                             "event": ee, "pair_type": ptype, "offset": C.cond_dist(bb, ee)})
            break
        else:
            raise SystemExit("cond_sametens not filled")
    log(lines, f"dev cond_sametens (revision 4, seed {SEED_SAMETENS}): {len(rows)} rows, {len(types)} pairs; "
               f"rejected {dict(rej)}")
    return rows


def edge_problem(b: dict, ev: dict) -> str | None:
    """Why a conditional is NOT a cond_oor_edge row (None = it is one): other - threshold in EDGE_OFFSETS (-1, 0, +1,
    +2), |value| in OOR_V_ABS, in-range threshold and registers, neither branch wraps."""
    x, t, v, r = b[ev["other"]], ev["threshold"], ev["value"], b[ev["register"]]
    if ev["kind"] != "conditional" or x - t not in C.EDGE_OFFSETS:
        return "offset"
    if not C.OOR_V_ABS[0] <= abs(v) <= C.OOR_V_ABS[1]:
        return "value"
    if not (C.THR_IN[0] <= t <= C.THR_IN[1] and all(C.THR_IN[0] <= y <= C.THR_IN[1] for y in b.values())):
        return "threshold / register range"
    if not (0 <= r - abs(v) and r + abs(v) <= C.MOD - 1):
        return "a branch wraps"
    return None


def build_edge(d: Draw, reg: Registry, lines: list, start: int) -> list:
    """Revision 5: the identification stratum cond_oor_edge (G1 alone: trace - plain and the trace predicate accuracy).
    20 branch-balanced minimal QUADRUPLES at the boundary of the env's strict 'other > threshold': the four members have
    other - threshold = -1, 0 (ok false) and +1, +2 (ok true), so every off-by-one variant of the comparison (any
    non-empty flip of the branch at other - threshold in {-1, 0, +1}) is contradicted on >= 1 of the 4 members of EVERY
    quadruple; 10 quadruples share the row and move the threshold (threshold = x + 1, x, x - 1, x - 2), 10 share the
    event and move the other register (other = threshold - 1 .. threshold + 2); an out-of-range |value| 12..40 (both
    signs) with the updated register at |value|..96-|value| so that NEITHER branch wraps (a row's 'after' is right iff
    its branch is, given the in-branch arithmetic); registers and thresholds in range. The decade of the boundary is
    drawn, not chosen. Drawn after cond_sametens, with its own seed, against the full registry (exclusion union +
    training + dev + chains + cond_sametens): fresh, and disjoint from every training payload and operative content.
    Rows are appended to the dev bank (uids from dv{start})."""
    rows, rej = [], Counter()
    lo, hi = min(C.EDGE_OFFSETS), max(C.EDGE_OFFSETS)
    types = [t for t in TWIN_TYPES for _ in range(EDGE_TYPE_QUOTA)]
    for q, ptype in enumerate(types):
        cid = f"deg{q:03d}"
        for _ in range(MAX_DRAWS):
            ev = d.event("conditional", d.oor_value(), 0)
            b = d.before()
            o, r, av = ev["other"], ev["register"], abs(ev["value"])
            if not av <= b[r] <= C.MOD - 1 - av:
                rej["a branch would wrap"] += 1
                continue
            if ptype == "threshold":
                x = b[o]
                if not C.THR_IN[0] + hi <= x <= C.THR_IN[1] + lo:
                    rej["x leaves no room"] += 1
                    continue
                quad = [(b, dict(ev, threshold=x - k)) for k in C.EDGE_OFFSETS]
            else:
                thr = d.rng.choice(d.thresholds)
                if not C.THR_IN[0] - lo <= thr <= C.THR_IN[1] - hi:
                    rej["threshold leaves no room"] += 1
                    continue
                ev = dict(ev, threshold=thr)
                quad = [(dict(b, **{o: thr + k}), ev) for k in C.EDGE_OFFSETS]
            if (any(C.single_diff(quad[0], p) != ptype for p in quad[1:])
                    or [C.cond_dist(*p) for p in quad] != list(C.EDGE_OFFSETS)
                    or [C.branch_ok(*p) for p in quad] != [False, False, True, True]
                    or any(edge_problem(*p) for p in quad)):
                raise AssertionError("cond_oor_edge quadruple construction")
            pr = reg.problem(quad, cid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(quad, cid)
            for bb, ee in quad:
                rows.append({"uid": f"dv{start + len(rows):04d}", "stratum": "cond_oor_edge",
                             "group": C.DEV_STRATA["cond_oor_edge"][0], "cluster": cid, "role": "quadruple",
                             "before": dict(bb), "event": ee, "pair_type": ptype, "offset": C.cond_dist(bb, ee)})
            break
        else:
            raise SystemExit("cond_oor_edge not filled")
    log(lines, f"dev cond_oor_edge (revision 5, seed {SEED_EDGE}): {len(rows)} rows, {len(types)} quadruples; "
               f"rejected {dict(rej)}")
    return rows


def build_chains(rng: random.Random, m, reg: Registry, lines: list) -> list:
    parents, rej = [], Counter()
    d = Draw(rng, m)
    for stratum, n in C.CHAIN_STRATA.items():
        made, draws = 0, 0
        while made < n:
            draws += 1
            if draws > 200_000:
                raise SystemExit(f"{stratum}: not filled")
            s0 = env.gen_start(FAM, rng, m)
            cands = []
            for c in range(C.K):
                evs = [env.gen_event(FAM, rng, m, tick=t + 1) for t in range(C.H)]
                if stratum == "CH_X":
                    for e in evs:
                        if e["kind"] != "swap":
                            e["value"] = d.oor_value()
                cands.append(evs)
            if stratum == "CH_X" and not all(any(e["kind"] != "swap" for e in evs) for evs in cands):
                rej["CH_X candidate without an out-of-range value"] += 1
                continue
            finals = {env.dumps(env.rollout(FAM, s0, evs)[-1]) for evs in cands}
            if len(finals) < MIN_DISTINCT_FINALS:
                rej[f"< {MIN_DISTINCT_FINALS} distinct final states"] += 1
                continue
            items = []
            for evs in cands:
                st = env.rollout(FAM, s0, evs)
                for t, e in enumerate(evs):
                    env.cross_check(FAM, st[t], e)
                    items.append((st[t], e))
            content = env.dumps({"s0": s0, "cands": cands})
            pid = "T2BR-" + hashlib.sha256(content.encode()).hexdigest()[:16]
            pr = reg.problem(items, pid)
            if pr:
                rej[pr] += 1
                continue
            reg.add(items, pid)
            parents.append({"parent_id": pid, "family": FAM, "split": "dev", "stratum": stratum, "s0": s0,
                            "candidates": [{"cand_id": f"{pid}-c{j}", "events": evs} for j, evs in enumerate(cands)]})
            made += 1
        log(lines, f"chains {stratum}: {n} parents x {C.K} candidates from {draws} draws")
    log(lines, f"chains rejected {dict(rej)}")
    return parents


# ---------------------------------------------------------------------------------------------- rendering
# Revision-2 digests of the rows (the post-review build of 2026-10-09 about 07:47), computed on the revision-2 files
# before the revision-3 rebuild: the revision-3 build must reproduce the SAME rows (ids, before, event, golds, plain and
# trace targets, canonical prompts; the chain bank byte for byte). The rows are never re-drawn; only the filler targets
# (4b), the residual / surface fields and the added renderings are new. Independently, --verify reconstructs the four
# revision-2 data files from the current rows (their amendment-4 per-row matched filler and field layout) and requires
# their recorded sha256 (REV2_FILE_SHA256, from docs/t2b/register/DESIGN_rev2.md / the revision-2 GEN_PLAN.json).
REV2_TRAIN_ROWS_DIGEST = "fd6710e20a5556708b296364718b6edb7275f7b3adac02a52b95700384302e42"
REV2_DEV_ROWS_DIGEST = "1113af0fedb5d9e955445472d703799eac21403e37fbec7838c45cb2bc6d5648"
REV2_CHAINS_SHA256 = "fe15c2b77c549942f79c389987404c96b691dd20efd006c3465319b800f44a62"
REV2_FILE_SHA256 = {"t2br_train_plain.jsonl": "49067515714f09a72838f7b3414523a937707dfadf565a107bef6709fd294534",
                    "t2br_train_trace.jsonl": "1bad81166f18494829b58a79e2c3cdaa34bb5b64f42f52068305d720b5658884",
                    "t2br_train_filler.jsonl": "3102361dae2f6dddf8b643635409caf790b3ac0dc454fdda51d0d37a66b6721d",
                    "t2br_dev.jsonl": "2c0250a2189e1b5a37ccdc6157f4b5f41d6a9c69eef4a4049472beff97070b43"}
# Revision 3's data files (built 2026-10-09 about 12:25; AUDIT.json of revision 3, files_sha256). Revision 4 adds the
# cond_sametens rows (appended) and implements the constant 4b definition (new FILL_UNITS), so the files change; the
# rows do not. --verify reconstructs these eight files byte for byte from the CURRENT rows with revision 3's fill
# constants (FILL_UNITS_REV3) and revision 3's strata (rev3_reconstruction). The superseded files themselves are kept,
# renamed, under data/t2br_superseded/rev3/. (The chain bank is unchanged: REV2_CHAINS_SHA256.)
REV3_STRATA = tuple(s for s in C.DEV_STRATA if s not in C.IDENT_STRATA)        # the 1,100 rows of revisions 2-3
REV3_FILE_SHA256 = {"t2br_train_plain.jsonl": "058acce806b2e40f087098387c4921a80440644fe1b4cf599008350f213e96b3",
                    "t2br_train_trace.jsonl": "81f4af025c254c97e86bce289ec8de3ded021cca6c9b85197d5e7c62db43a1c1",
                    "t2br_train_filler.jsonl": "726134cd2ed5a9cda6fc53ae4a3b65a96f8325afda12b80edc42e43736f77e5a",
                    "t2br_train_sd_plain.jsonl": "4c679f3596780237c57436b32be8a74723f1272007a5222957b5a1d09d516eef",
                    "t2br_train_sd_trace.jsonl": "941c14084708db53d01db7d9cf8589f12adfe5fc18e5bae25a8017b0e8ebc68e",
                    "t2br_train_sd_filler.jsonl": "e99f4f5f7015451c5e372ef4b821fbfc796d26fe0e5bbad98162144944d31bcd",
                    "t2br_dev.jsonl": "503d7686a38db51ae1a5ef59ee6b67c55defdc00f2a967562d1680d0877f02e8",
                    "t2br_dev_hs.jsonl": "1ced9b71716abb9883629b05e661b37b0c9fb518fa51cbaeca107511a7861939"}


# Revision 4's data files (built 2026-10-09 about 17:38; AUDIT.json of revision 4, files_sha256). Revision 5 only
# appends the cond_oor_edge rows (the 4b constants are unchanged: the new rows' traces are not longer than the kind's
# longest), so --verify reconstructs these eight files byte for byte from the CURRENT rows (rev4_reconstruction: the
# revision-4 strata and held-out-surface strata). The superseded files themselves are kept, renamed, under
# data/t2br_superseded/rev4/. (The chain bank is unchanged: REV2_CHAINS_SHA256.)
REV4_STRATA = tuple(s for s in C.DEV_STRATA if s != "cond_oor_edge")          # the 1,200 rows of revision 4
FILL_UNITS_REV4 = {"swap": 7, "add": 17, "mix": 17, "conditional": 29}        # revision 4's constants (== FILL_UNITS)
REV4_FILE_SHA256 = {"t2br_train_plain.jsonl": "3540a991710b8898ffc733bfd3da1d81023b5bf579abd7df9cefe884e5bab890",
                    "t2br_train_trace.jsonl": "c9cb9238e12a66ce9c93ad70639f5f8174b4ff66ed142dcbca3f590b13ae51dd",
                    "t2br_train_filler.jsonl": "4e207129c97b88898145fa43ee6f29573d2e16bc6a4bff9a98bde54b5cd5a433",
                    "t2br_train_sd_plain.jsonl": "d3de9554d31c3c7fc234554d19f83fe9d2f3ce38c5c9893d6ca9c12a4a8e43cc",
                    "t2br_train_sd_trace.jsonl": "cc9128c965c12293c4e32fbfb69bd05ba7bb640ffa199bff5320c5b76102e3a2",
                    "t2br_train_sd_filler.jsonl": "01165efe0d1f385905cb30f91e001c6852473bdb671d0f2868e6f1e656227011",
                    "t2br_dev.jsonl": "9aac40deffd678a4bdefffbb290081872015c88ea38296001c671e4067f46102",
                    "t2br_dev_hs.jsonl": "61845fbfc8635423257fac7c52b3ed1c70d19dd799720ca5f0e79866af1764f8"}


def rev2_matched_fill(trace: str, after: dict, count) -> tuple[str, int]:
    """Revision 2's (withdrawn) amendment-4 filler: the fill whose WHOLE target has exactly the trace's token count. Used
    ONLY to reconstruct the revision-2 files (provenance of the rows); never a training or scoring target."""
    n = count(trace)
    for k in range(1, n + 1):
        t = C.filler_text(k, after)
        c = count(t)
        if c == n:
            return t, k
        if c > n:
            break
    raise ValueError("no matched fill")


def rev2_reconstruction(train_base: dict, dev: list, count) -> dict:
    """The four revision-2 data files rebuilt from the CURRENT rows in revision 2's layout; sha256 of each."""
    out = {}
    for a in C.ARMS:
        rows = []
        for r, rt in zip(train_base[a], train_base["trace"]):
            fi, k = rev2_matched_fill(rt["target"], json.loads(r["gold_after"]), count)
            target = fi if a == "filler" else r["target"]
            rows.append({"id": r["id"], "prompt": r["prompt"], "target": target, "arm": a, "family": r["family"],
                         "category": r["category"], "stratum": r["stratum"], "op": r["op"], "cluster": r["cluster"],
                         "role": r["role"], "offset": r["offset"], "pattern": r["pattern"], "twin_type": r["twin_type"],
                         "before": r["before"], "event": r["event"], "gold_after": r["gold_after"],
                         "gold_cmp": r["gold_cmp"], "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"],
                         "fill_units": k, "n_prompt_tokens": r["n_prompt_tokens"], "n_target_tokens": count(target),
                         "sd": None})
        out[f"t2br_train_{a}.jsonl"] = hashlib.sha256(_jsonl_text(rows).encode("utf-8")).hexdigest()
    rows = []
    for r in [r for r in dev if r["stratum"] in REV3_STRATA]:
        fi, k = rev2_matched_fill(r["targets"]["trace"], json.loads(r["gold_after"]), count)
        tg = dict(r["targets"], filler=fi)
        rows.append({"id": r["id"], "stratum": r["stratum"], "group": r["group"], "cluster": r["cluster"],
                     "role": r["role"], "family": r["family"], "before": r["before"], "event": r["event"],
                     "messages": r["messages"], "op": r["op"], "gold_after": r["gold_after"], "gold_cmp": r["gold_cmp"],
                     "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"], "fill_units": k, "targets": tg,
                     "n_target_tokens": {a: count(tg[a]) for a in C.ARMS}, "n_prompt_tokens": r["n_prompt_tokens"],
                     **{x: r[x] for x in ("offset", "wrap", "direction", "thr_band", "pair_type") if x in r}})
    out["t2br_dev.jsonl"] = hashlib.sha256(_jsonl_text(rows).encode("utf-8")).hexdigest()
    return out


def train_rows_digest(rows_by_arm: dict) -> str:
    h = hashlib.sha256()
    for a in ("plain", "trace"):
        for r in rows_by_arm[a]:
            h.update(json.dumps([r["id"], r["before"], r["event"], r["target"], r["prompt"], r["gold_after"]],
                                sort_keys=True).encode())
    return h.hexdigest()


def dev_rows_digest(dev_rows: list) -> str:
    """Over the revision-2/3 strata only (the 1,100 rows that existed then; cond_sametens is new in revision 4)."""
    h = hashlib.sha256()
    for r in [r for r in dev_rows if r["stratum"] in REV3_STRATA]:
        h.update(json.dumps([r["id"], r["stratum"], r["cluster"], r["before"], r["event"], r["gold_after"],
                             r["targets"]["plain"], r["targets"]["trace"], r["messages"]], sort_keys=True).encode())
    return h.hexdigest()


def finish_row(r: dict, count, fill_table: dict | None = None) -> dict:
    """Gold, the three targets (amendment-4b filler) and their token counts for one (before, event). fill_table None =
    the current FILL_UNITS (C.FILL_UNITS_REV3 only for the revision-3 reconstruction)."""
    b, ev = r["before"], r["event"]
    env.cross_check(FAM, b, ev)
    if tuple(sorted(ev)) != C.EVENT_KEYS:
        raise AssertionError(f"unexpected event keys {sorted(ev)}")
    tg = C.targets(b, ev, fill_table)
    n = {a: count(tg[a]) for a in C.ARMS}
    return {**r, "op": C.op_class(b, ev), "gold_after": tg["after"], "gold_cmp": tg["cmp"], "gold_calc": tg["calc"],
            "gold_ok": None if tg["cmp"] is None else tg["cmp"]["ok"], "fill_units": tg["fill_units"],
            "fill_residual": n["filler"] - n["trace"], "targets": {a: tg[a] for a in C.ARMS}, "n_target_tokens": n}


def train_rows_for_arm(rows: list, arm: str, surfaces: list | None, count) -> list:
    """One arm's training rows; surfaces None = the canonical frame (base), else one surface per row (sd)."""
    out = []
    for i, r in enumerate(rows):
        sf = None if surfaces is None else surfaces[i]
        prompt = C.render_prompt(C.surface_messages(r["before"], r["event"], sf))
        out.append({"id": r["uid"], "prompt": prompt, "target": r["targets"][arm], "arm": arm, "family": FAM,
                    "category": r["category"], "stratum": r["stratum"], "op": r["op"], "cluster": r["cluster"],
                    "role": r["role"], "offset": r.get("offset"), "pattern": r.get("pattern"),
                    "twin_type": r.get("twin_type"),
                    "before": r["before"], "event": r["event"], "gold_after": r["gold_after"], "gold_cmp": r["gold_cmp"],
                    "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"], "fill_units": r["fill_units"],
                    "fill_residual": r["fill_residual"], "n_prompt_tokens": count(prompt),
                    "n_target_tokens": r["n_target_tokens"][arm], "surface": sf})
    return out


def dev_row_out(r: dict, count) -> dict:
    msgs = C.canonical_messages(r["before"], r["event"])
    return {"id": r["uid"], "stratum": r["stratum"], "group": r["group"], "cluster": r["cluster"], "role": r["role"],
            "family": FAM, "before": r["before"], "event": r["event"], "messages": msgs, "op": r["op"],
            "gold_after": r["gold_after"], "gold_cmp": r["gold_cmp"], "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"],
            "fill_units": r["fill_units"], "fill_residual": r["fill_residual"], "targets": r["targets"],
            "n_target_tokens": r["n_target_tokens"], "n_prompt_tokens": count(C.render_prompt(msgs)),
            **{k: r[k] for k in ("offset", "wrap", "direction", "thr_band", "pair_type") if k in r}}


def hs_rows(dev_out: list, count, strata: tuple = C.HS_STRATA) -> list:
    """C3: the held-out-surface strata's dev rows (the primaries + cond_sametens) in dev order, the j-th in the combined
    held-out surface heldout_surface(j). strata = C.HS_STRATA_REV3 only for the revision-3 reconstruction."""
    out = []
    for j, r in enumerate([r for r in dev_out if r["stratum"] in strata]):
        sf = C.heldout_surface(j)
        msgs = C.surface_messages(r["before"], r["event"], sf)
        out.append({"id": r["id"], "stratum": r["stratum"], "group": r["group"], "cluster": r["cluster"],
                    "role": r["role"], "family": FAM, "hs_index": j, "before": r["before"], "event": r["event"],
                    "gold_after": r["gold_after"], "messages": msgs, "surface": sf,
                    "n_prompt_tokens": count(C.render_prompt(msgs))})
    return out


TRAIN_ROW_KEYS = ("category", "stratum", "cluster", "role", "offset", "pattern", "twin_type", "before", "event")
DEV_EXTRA_KEYS = ("offset", "wrap", "direction", "thr_band", "pair_type")


def internal_train_rows(base_rows: list) -> list:
    """The builder's row records (before finish_row) recovered from a canonical training file."""
    return [{"uid": x["id"], **{k: x[k] for k in TRAIN_ROW_KEYS}} for x in base_rows]


def internal_dev_rows(dev_rows: list) -> list:
    """The builder's row records (before finish_row) recovered from the dev file."""
    return [{"uid": r["id"], **{k: r[k] for k in ("stratum", "group", "cluster", "role", "before", "event")},
             **{k: r[k] for k in DEV_EXTRA_KEYS if k in r}} for r in dev_rows]


def rev3_reconstruction(train_rows: list, surfaces: list, dev_rows: list, count) -> dict:
    """Revision 3's eight data files rebuilt from the CURRENT rows (train_rows / dev_rows: the builder's row records)
    with revision 3's fill constants (C.FILL_UNITS_REV3) and strata (no cond_sametens; the held-out-surface bank of the
    four primary strata): sha256 of each. Equal to REV3_FILE_SHA256 iff no row changed."""
    tr3 = [finish_row(r, count, C.FILL_UNITS_REV3) for r in train_rows]
    dv3 = [dev_row_out(finish_row(r, count, C.FILL_UNITS_REV3), count) for r in dev_rows if r["stratum"] in REV3_STRATA]
    texts = {**{C.train_path(a, "base").name: _jsonl_text(train_rows_for_arm(tr3, a, None, count)) for a in C.ARMS},
             **{C.train_path(a, "sd").name: _jsonl_text(train_rows_for_arm(tr3, a, surfaces, count)) for a in C.ARMS},
             C.DEV.name: _jsonl_text(dv3), C.DEV_HS.name: _jsonl_text(hs_rows(dv3, count, C.HS_STRATA_REV3))}
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts.items()}


def rev4_reconstruction(train_rows: list, surfaces: list, dev_rows: list, count) -> dict:
    """Revision 4's eight data files rebuilt from the CURRENT rows (the builder's row records) with revision 4's fill
    constants and strata (no cond_oor_edge; the held-out-surface bank of the primary strata + cond_sametens): sha256 of
    each. Equal to REV4_FILE_SHA256 iff no row and no constant changed."""
    tr4 = [finish_row(r, count, FILL_UNITS_REV4) for r in train_rows]
    dv4 = [dev_row_out(finish_row(r, count, FILL_UNITS_REV4), count) for r in dev_rows if r["stratum"] in REV4_STRATA]
    texts = {**{C.train_path(a, "base").name: _jsonl_text(train_rows_for_arm(tr4, a, None, count)) for a in C.ARMS},
             **{C.train_path(a, "sd").name: _jsonl_text(train_rows_for_arm(tr4, a, surfaces, count)) for a in C.ARMS},
             C.DEV.name: _jsonl_text(dv4), C.DEV_HS.name: _jsonl_text(hs_rows(dv4, count, C.HS_STRATA_REV4))}
    return {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts.items()}


def chain_steps(parents: list) -> list:
    """Every gold step (before, event) of every chain candidate, with its chain stratum."""
    out = []
    for p in parents:
        for c in p["candidates"]:
            st = env.rollout(FAM, p["s0"], c["events"])
            out.extend({"cell": f"chain {p['stratum']}", "before": st[t], "event": e} for t, e in enumerate(c["events"]))
    return out


def messages_of_prompt(prompt: str) -> list:
    """The (system, user) messages of a rendered one-step prompt (render_prompt's inverse; checked by re-rendering)."""
    head = "<|im_start|>system\n"
    if not prompt.startswith(head):
        raise ValueError("not a rendered prompt")
    system, rest = prompt[len(head):].split("<|im_end|>\n<|im_start|>user\n", 1)
    user = rest.split("<|im_end|>\n", 1)[0]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if C.render_prompt(msgs) != prompt:
        raise ValueError("prompt does not re-render")
    return msgs


def check_template() -> int:
    """render_prompt == the real chat template on every register one-step prompt of data/t2_train.jsonl."""
    if C.sha256(T2_TRAIN) != T2_TRAIN_SHA:
        raise SystemExit("t2_train.jsonl changed")
    n = 0
    for r in C.load_jsonl(T2_TRAIN):
        if r["family"] == FAM and r["frame"] == "one":
            if C.render_prompt(env.one_step_messages(FAM, r["before"], r["event"])) != r["prompt"]:
                raise SystemExit("render_prompt differs from the real chat template")
            n += 1
    return n


# ---------------------------------------------------------------------------------------------- audit
def _heur_counts(h: str, rows: list):
    app = [r for r in rows if C.heuristic_after(h, r["before"], r["event"]) is not None]
    ok = sum(env.dumps(C.heuristic_after(h, r["before"], r["event"])) == r["gold_after"] for r in app)
    return ok, len(app)


def audit(rows: list) -> dict:
    """Heuristic accuracies (exact after) on the training rows, decided on exact counts.
    GATED <= 0.55 (surface heuristics: no comparison of other with threshold): copy (all, mix, wrap, cond), unreduced
    (all, mix, wrap), always_add / always_sub / wrong_register / x_hi / thr_lo / v_pos (cond).
    RULE VARIANTS (comparisons that coincide with 'other > threshold' on a structural subset; a 0.55 bar is unreachable
    for them, see t2br_common.RULE_VARIANTS): each must be contradicted by >= 1/6 of the conditional rows.
    DIAGNOSTIC: unreduced (cond), mix_1x / mix_no_v (mix), always_wrap (wrap: 1.0 by design, every training add wraps)."""
    cats = {"all": rows, "mix": [r for r in rows if r["category"] == "mix"],
            "wrap": [r for r in rows if r["category"] == "wrap"], "cond": [r for r in rows if r["category"] == "cond"]}
    res = {"n": {k: len(v) for k, v in cats.items()}, "max": float(C.AUDIT_MAX), "acc": {}, "gated": {},
           "rule_variants": {}, "diagnostic": {}}
    counts = {}
    for h in C.GATED_HEURISTICS + C.RULE_VARIANTS + C.DIAG_HEURISTICS:
        res["acc"][h] = {}
        for cname, rs in cats.items():
            ok, n = _heur_counts(h, rs)
            if n:
                counts[(h, cname)] = (ok, n)
                res["acc"][h][cname] = {"n": n, "correct": ok, "acc": round(ok / n, 4)}
    gate_cells = {"copy": ("all", "mix", "wrap", "cond"), "unreduced": ("all", "mix", "wrap"),
                  **{h: ("cond",) for h in ("always_add", "always_sub", "wrong_register", "x_hi", "thr_lo", "v_pos")}}
    for h, cells in gate_cells.items():
        for c in cells:
            ok, n = counts[(h, c)]
            res["gated"][f"{h}|{c}"] = {"acc": round(ok / n, 4), "ok": Fraction(ok, n) <= C.AUDIT_MAX}
    bar = 1 - C.RULE_VARIANT_MIN_CONTRA
    for h in C.RULE_VARIANTS:
        ok, n = counts[(h, "cond")]
        res["rule_variants"][f"{h}|cond"] = {"acc": round(ok / n, 4), "contradicted_by": n - ok, "n": n,
                                             "ok": Fraction(ok, n) <= bar}
    res["rule_variant_bar"] = {"max_acc": round(float(bar), 4), "min_contradicting_share": str(C.RULE_VARIANT_MIN_CONTRA)}
    for h, c in (("unreduced", "cond"), ("mix_1x", "mix"), ("mix_no_v", "mix"), ("always_wrap", "wrap")):
        ok, n = counts[(h, c)]
        res["diagnostic"][f"{h}|{c}"] = {"acc": round(ok / n, 4), "above_0.55": Fraction(ok, n) > C.AUDIT_MAX}
    res["diagnostic_note"] = (
        "Not gated. unreduced on cond rows is reported (the conditional shortcut of interest is the branch). "
        "always_wrap ('add => wrap by 97') is 1.0 on the training adds BY DESIGN (every training add wraps; add without "
        "a wrap is the untouched add_nowrap control), so a G4 loss on add_nowrap shared by every trained arm is a "
        "property of the data, not of the trace format (the analysis reports trace - plain beside trace - OFF).")
    res["rule_variant_note"] = (
        "eq_plus1 = P(ok false) + P(other = threshold + 1) >= 0.5 + 50/300; non_strict = 1 - P(other = threshold) = "
        "1 - 50/300 by construction; a 0.55 bar would need >= 1,000 conditional rows. The twins (distance 2..15, on "
        "the other side of the threshold) contradict every near-only rule; the core rows contradict coarse comparisons.")
    cond_rows = cats["cond"]
    res["distance"] = {"core": dict(sorted(Counter(r["offset"] for r in cond_rows if r["role"] == "core").items())),
                       "twin_abs": [min(abs(r["offset"]) for r in cond_rows if r["role"] == "twin"),
                                    max(abs(r["offset"]) for r in cond_rows if r["role"] == "twin")],
                       "twin_type": dict(Counter(r["twin_type"] for r in cond_rows if r["role"] == "twin")),
                       "twin_side": dict(Counter("ok_true" if r["gold_ok"] else "ok_false" for r in cond_rows
                                                 if r["role"] == "twin"))}
    # best single-feature threshold stump for the branch (surface features of one row), vs a permutation null
    cond = cats["cond"]
    feats = {"X": lambda r: r["before"]["X"], "Y": lambda r: r["before"]["Y"], "Z": lambda r: r["before"]["Z"],
             "reg_val": lambda r: r["before"][r["event"]["register"]], "other_val": lambda r: r["before"][r["event"]["other"]],
             "threshold": lambda r: r["event"]["threshold"], "value": lambda r: r["event"]["value"],
             "tick": lambda r: r["event"]["tick"], "register_id": lambda r: C.REGS.index(r["event"]["register"]),
             "other_id": lambda r: C.REGS.index(r["event"]["other"])}
    y = [bool(r["gold_ok"]) for r in cond]

    n = len(cond)
    fvals = [[f(r) for r in cond] for f in feats.values()]
    forders = [sorted(range(n), key=lambda i, v=v: v[i]) for v in fvals]

    def best_stump(labels):
        """max over features and cuts of the accuracy of 'ok iff feature <= cut' (or its negation)."""
        best, tot_false = 0.0, n - sum(labels)
        for vals, order in zip(fvals, forders):
            ct = cf = i = 0
            while i < n:
                v = vals[order[i]]
                while i < n and vals[order[i]] == v:
                    if labels[order[i]]:
                        ct += 1
                    else:
                        cf += 1
                    i += 1
                acc = (ct + tot_false - cf) / n
                best = max(best, acc, 1 - acc)
        return best
    obs = best_stump(y)
    rng = random.Random(17)
    null = []
    for _ in range(100):
        yy = y[:]
        rng.shuffle(yy)
        null.append(best_stump(yy))
    null.sort()
    res["branch_stump"] = {"best_single_feature_accuracy": round(obs, 4), "permutation_null_p95": round(null[94], 4),
                           "within_null": obs <= null[94], "features": list(feats), "gated": False}
    res["branch_balance"] = {"ok_true": sum(y), "ok_false": len(y) - sum(y)}
    dig = {}
    for h in C.DIGIT_SHORTCUTS:
        ok_, n_ = _heur_counts(h, cats["cond"])
        dig[f"{h}|cond"] = {"acc": round(ok_ / n_, 4), "contradicted_by": n_ - ok_, "n": n_,
                            "contradicting_rows": dict(Counter(f"{r['role']} d={C.cond_dist(r['before'], r['event'])}"
                                                               for r in cats["cond"] if env.dumps(C.heuristic_after(
                                                                   h, r["before"], r["event"])) != r["gold_after"]))}
    res["digit_shortcuts"] = dig
    obo = {}
    for h in C.OFF_BY_ONE:
        ok_, n_ = _heur_counts(h, cats["cond"])
        obo[f"{h}|cond"] = {"form": C.OFF_BY_ONE_FORMS[h], "acc": round(ok_ / n_, 4), "contradicted_by": n_ - ok_,
                            "n": n_, "contradicted_share_ge_1/6": Fraction(n_ - ok_, n_) >= C.RULE_VARIANT_MIN_CONTRA}
    res["off_by_one"] = obo
    res["off_by_one_note"] = (
        "Revision 5, reported, NOT gated: the off-by-one family (the branch flipped where other - threshold is in a "
        "non-empty subset of {-1, 0, +1}). The 150 core rows at -1 / 0 / +1 (50 each) contradict every single flip on "
        "exactly 50 of the 300 conditional rows (1/6); the dev bank identifies the family through G1 (cond_oor_val and "
        "cond_oor_edge).")
    res["digit_shortcut_note"] = (
        "Revision 4, reported, NOT gated: the tens-digit shortcut 'add iff tens(other) > tens(threshold) or other == "
        "threshold + 1' and its mirror 'add iff tens(other) >= tens(threshold) and other != threshold' (derived for the "
        "env's strict '>'). On the training rows only the twins at |d| 2..9 inside one decade contradict them; the 750 "
        "rows are fixed (never re-drawn), so their identification is tested on the dev bank (G1's cond_sametens clause).")
    res["ok"] = all(v["ok"] for v in res["gated"].values()) and all(v["ok"] for v in res["rule_variants"].values()) \
        and sum(y) * 2 == len(y)
    return res


BRANCH_HEURISTICS = (("always_add", "always_sub", "wrong_register", "x_hi", "thr_lo", "v_pos") + C.RULE_VARIANTS
                     + C.DIGIT_SHORTCUTS + tuple(C.OFF_BY_ONE))


def dev_heuristics(dev: list) -> dict:
    """What each shortcut would score on each dev stratum and on the rule's outcome groups (data only, no model). Shows
    which strata a shortcut can pass: e.g. eq_plus1 is at 0.5 on cond_far, thr_lo is 1.0 on cond_oor_thr by design,
    tens_digit_gt (a coarse comparison) passes cond_oor_thr / cond_far but not the boundary pairs cond_oor_val (G1's
    cond_oor_val clause), and tens_or_eq_plus1 / tens_ge_not_eq pass every primary conditional stratum but score 0.5 on
    cond_sametens (G1's cond_sametens clause, revision 4); every off-by-one variant (revision 5) scores at most 3/4 on
    cond_oor_edge."""
    out = {}
    groups = {**{s: (s,) for s in C.DEV_STRATA}, "primary_conditional": C.PRIMARY["conditional"],
              "primary_mix": C.PRIMARY["mix"]}
    for g, ss in groups.items():
        rs = [r for r in dev if r["stratum"] in ss]
        hs = BRANCH_HEURISTICS if all(r["event"]["kind"] == "conditional" for r in rs) else \
            ("copy", "unreduced", "always_wrap", "mix_1x", "mix_no_v")
        cell = {}
        for h in hs:
            ok, n = _heur_counts(h, rs)
            if n:
                cell[h] = round(ok / n, 4)
        out[g] = cell
    return out


def fill_constant_audit(train: list, dev: list, parents: list, count) -> dict:
    """Amendment 4b (C4) on the data. (1) The constant per kind re-derived from train + dev (one-step rows and the
    chains' gold steps) with the revision-4 definition (t2br_common.min_constant_fill: the smallest fill such that the
    kind's SHORTEST filler target is at least as long as its LONGEST trace target) equals t2br_common.FILL_UNITS; the
    shortest filler target of every kind is >= its longest trace target. (2) Every filler target of a kind carries the
    SAME fill string, so it is identical up to 'after'. (3) Information test: within every cell the fill length's best
    single cut predicts the answer's wrap (add / mix) or branch (conditional) no better than the majority rate (it is
    constant). (4) The residual = filler tokens - trace tokens of the row (>= 0 everywhere), reported as a descriptive
    covariate with what it would tell about the label (NOT visible to the filler arm, whose fill is constant). Also
    recorded (descriptive): what revision 3's per-row derivation (min_per_row_fill_rev3) gives on the same rows."""
    recs = []
    for r in train:
        recs.append({"cell": f"train {r['category']}", "before": r["before"], "event": r["event"],
                     "fill": json.loads(r["targets"]["filler"])["fill"],
                     "n_trace": r["n_target_tokens"]["trace"], "n_filler": r["n_target_tokens"]["filler"]})
    for r in dev:
        recs.append({"cell": f"dev {r['stratum']}", "before": r["before"], "event": r["event"],
                     "fill": json.loads(r["targets"]["filler"])["fill"],
                     "n_trace": r["n_target_tokens"]["trace"], "n_filler": r["n_target_tokens"]["filler"]})
    for s in chain_steps(parents):
        tg = C.targets(s["before"], s["event"])
        recs.append({**s, "fill": json.loads(tg["filler"])["fill"], "n_trace": count(tg["trace"]),
                     "n_filler": count(tg["filler"])})
    for x in recs:
        k = x["kind"] = x["event"]["kind"]
        x["after"] = env.step(FAM, x["before"], x["event"])
        x["label"] = None if k == "swap" else (C.branch_ok(x["before"], x["event"]) if k == "conditional"
                                               else C.wraps(x["before"], x["event"]))
        x["residual"] = x["n_filler"] - x["n_trace"]
    out = {"constants": dict(C.FILL_UNITS), "per_kind": {}, "cells": {}}
    ok = True
    for k in C.KINDS:
        xs = [x for x in recs if x["kind"] == k]
        longest = max(xs, key=lambda x: x["n_trace"])
        uniq_afters = list({env.dumps(x["after"]): x["after"] for x in xs}.values())
        derived = C.min_constant_fill(uniq_afters, longest["n_trace"], count)
        per_row = C.min_per_row_fill_rev3([(x["after"], x["n_trace"]) for x in xs], count)
        fills = sorted({x["fill"] for x in xs})
        res = [x["residual"] for x in xs]
        shortest_filler = min(x["n_filler"] for x in xs)
        out["per_kind"][k] = {"n_rows": len(xs), "derived_constant": derived, "pre_stated": C.FILL_UNITS[k],
                              "revision3_per_row_derivation": per_row, "revision3_constant": C.FILL_UNITS_REV3[k],
                              "distinct_fill_strings": len(fills), "longest_trace_tokens": longest["n_trace"],
                              "longest_trace_cell": longest["cell"], "shortest_filler_tokens": shortest_filler,
                              "longest_filler_tokens": max(x["n_filler"] for x in xs),
                              "shortest_filler_ge_longest_trace": shortest_filler >= longest["n_trace"],
                              "residual_min": min(res),
                              "residual_mean": round(sum(res) / len(res), 4), "residual_max": max(res),
                              "residual_positive_share": round(sum(v > 0 for v in res) / len(res), 4)}
        ok &= derived == C.FILL_UNITS[k] and fills == [C.FILL_UNIT * C.FILL_UNITS[k]] and min(res) >= 0 \
            and shortest_filler >= longest["n_trace"]
    for cell in sorted({x["cell"] for x in recs}):
        for k in C.KINDS:
            xs = [x for x in recs if x["cell"] == cell and x["kind"] == k]
            if not xs:
                continue
            c = {"n": len(xs), "fill_units": sorted({len(x["fill"]) // len(C.FILL_UNIT) for x in xs}),
                 "residual_mean": round(sum(x["residual"] for x in xs) / len(xs), 4),
                 "residual_range": [min(x["residual"] for x in xs), max(x["residual"] for x in xs)]}
            if k != "swap":
                y = [bool(x["label"]) for x in xs]
                maj = max(sum(y), len(y) - sum(y)) / len(y)
                fl = C.best_cut_acc([len(x["fill"]) for x in xs], y)
                c.update(label="branch" if k == "conditional" else "wrap", majority=round(maj, 4),
                         fill_length_best_cut=round(fl, 4), fill_length_zero_information=abs(fl - maj) < 1e-12,
                         residual_best_cut_descriptive=round(C.best_cut_acc([x["residual"] for x in xs], y), 4))
                ok &= c["fill_length_zero_information"]
            out["cells"][f"{cell} / {k}"] = c
    out["ok"] = bool(ok)
    out["definition"] = ("per kind: the smallest run of ' .' units such that the filler target is at least as long "
                         "(Titan tokens) as the LONGEST trace target of that kind over train + dev + the chains' gold "
                         "steps, whatever its 'after' (the kind's shortest filler target vs its longest trace target); "
                         "revision 3's code compared each row's filler with that row's trace (revision3_per_row_"
                         "derivation)")
    out["note"] = ("fill length is a function of the action kind alone (prompt-derived): within a kind it is constant, "
                   "so it carries zero information about the answer (fill_length_best_cut == majority in every cell). "
                   "The residual (constant filler longer than the row's trace) is answer-dependent only through the "
                   "TRACE's length; it is a descriptive covariate, not something the filler arm emits.")
    return out


def surface_audit(base: dict, sd: dict, dev_out: list, hs: list) -> dict:
    """C2 / C3 proof on the data. TRAINING: every sd prompt decodes (t2br_common.read_prompt) to its row's (before,
    event) and to its recorded surface, which uses NO held-out factor (paraphrase 4, any field's name 4, any held-out key
    order), checked again on the raw strings (no held-out name as a JSON key, no held-out paraphrase fragment); the base
    prompts are Titan's verbatim frame. HELD-OUT SURFACE: the bank is the primary dev rows in dev order (same ids and
    golds), every prompt decodes to its row and uses ONLY held-out factors, key orders rotating by row."""
    held_names = {C.SD_FIELDS[f][C.HELDOUT_NAME_IDX] for f in C.SD_FIELDS}
    held_frags = [f for f in re.split(r"\{[a-z]+\}", C.SD_PARAPHRASES[C.HELDOUT_PARAPHRASE]) if len(f.strip()) >= 30]
    if not held_frags or any(f in C.SD_PARAPHRASES[i] for f in held_frags for i in C.TRAIN_PARAPHRASES):
        raise AssertionError("the held-out paraphrase needs distinctive fragments for the raw-string check")
    use = {"paraphrase": Counter(), "names": {f: Counter() for f in C.SD_FIELDS},
           "orders": {t: Counter() for t in C.OBJECT_KEYS}}
    bad_decode = held = raw_hits = canon_bad = 0
    for r in sd["plain"]:
        try:
            d = C.read_prompt(messages_of_prompt(r["prompt"]))
        except ValueError:
            bad_decode += 1
            continue
        bad_decode += env.dumps(d["before"]) != env.dumps(r["before"]) or env.dumps(d["event"]) != env.dumps(r["event"]) \
            or d["surface"] != r["surface"]
        hp = C.surface_heldout_parts(d["surface"])
        held += hp["paraphrase"] or bool(hp["names"]) or bool(hp["orders"])
        raw_hits += any(f'"{n}":' in r["prompt"] for n in held_names) or any(f in r["prompt"] for f in held_frags)
        use["paraphrase"][d["surface"]["paraphrase"]] += 1
        for f, ix in d["surface"]["names"].items():
            use["names"][f][ix] += 1
        for t, o in d["surface"]["orders"].items():
            use["orders"][t]["/".join(o)] += 1
    same_prompt_across_arms = all(x["prompt"] == y["prompt"] == z["prompt"]
                                  for x, y, z in zip(sd["plain"], sd["trace"], sd["filler"]))
    for r in base["plain"]:
        canon_bad += r["prompt"] != C.render_prompt(C.canonical_messages(r["before"], r["event"]))
    tr = {"n_rows": len(sd["plain"]), "decode_failures": bad_decode, "rows_using_a_heldout_factor": held,
          "rows_with_a_heldout_name_or_paraphrase_string": raw_hits, "identical_prompt_across_arms": same_prompt_across_arms,
          "base_prompts_not_canonical": canon_bad,
          "paraphrase_use": {str(k): v for k, v in sorted(use["paraphrase"].items())},
          "name_variant_use": {f: {str(k): v for k, v in sorted(c.items())} for f, c in use["names"].items()},
          "order_use": {"top": dict(sorted(use["orders"]["top"].items())),
                        "state": dict(sorted(use["orders"]["state"].items())),
                        "event_distinct_orders": len(use["orders"]["event"]),
                        "event_train_orders_available": len(C.TRAIN_ORDERS["event"])},
          "rows_fully_canonical": sum(r["surface"] == C.canonical_surface() for r in sd["plain"])}
    tr["ok"] = bad_decode == 0 and held == 0 and raw_hits == 0 and same_prompt_across_arms and canon_bad == 0 \
        and set(use["paraphrase"]) == set(C.TRAIN_PARAPHRASES) \
        and all(set(c) == set(C.TRAIN_NAME_IDX) for c in use["names"].values()) \
        and len(use["orders"]["state"]) == len(C.TRAIN_ORDERS["state"]) \
        and len(use["orders"]["top"]) == len(C.TRAIN_ORDERS["top"])
    want = [r for r in dev_out if r["stratum"] in C.HS_STRATA]
    hb = 0
    rot = {t: Counter() for t in C.OBJECT_KEYS}
    for j, (h, r) in enumerate(zip(hs, want)):
        try:
            d = C.read_prompt(h["messages"])
        except ValueError:
            hb += 1
            continue
        hp = C.surface_heldout_parts(d["surface"])
        hb += h["id"] != r["id"] or h["gold_after"] != r["gold_after"] or h["hs_index"] != j \
            or env.dumps(d["before"]) != env.dumps(r["before"]) or env.dumps(d["event"]) != env.dumps(r["event"]) \
            or d["surface"] != h["surface"] or h["surface"] != C.heldout_surface(j) or not hp["paraphrase"] \
            or hp["names"] != sorted(C.SD_FIELDS) or hp["orders"] != sorted(C.OBJECT_KEYS)
        for t, o in d["surface"]["orders"].items():
            rot[t]["/".join(o)] += 1
    hsr = {"n_rows": len(hs), "strata": dict(Counter(h["stratum"] for h in hs)), "bad_rows": hb,
           "same_ids_and_golds_as_in_surface": len(hs) == len(want) and all(
               h["id"] == r["id"] and h["gold_after"] == r["gold_after"] for h, r in zip(hs, want)),
           "order_rotation": {t: dict(sorted(c.items())) for t, c in rot.items()},
           "mean_prompt_tokens": {"held_out": round(sum(h["n_prompt_tokens"] for h in hs) / len(hs), 2),
                                  "in_surface_same_rows": round(sum(r["n_prompt_tokens"] for r in want) / len(want), 2)},
           "per_factor_renderings": "omitted by choice (a GPU-budget choice, not a capacity limit; see DESIGN.md)"}
    hsr["ok"] = hb == 0 and hsr["same_ids_and_golds_as_in_surface"] and len(hs) == len(want) and \
        all(len(c) == min(C.HS_ROTATION, len(C.HELDOUT_ORDERS[t])) for t, c in rot.items())
    return {"train_sd": tr, "heldout_surface_bank": hsr,
            "heldout": {"paraphrase": C.HELDOUT_PARAPHRASE, "field_names": {f: C.SD_FIELDS[f][C.HELDOUT_NAME_IDX]
                                                                            for f in C.SD_FIELDS},
                        "orders": {t: [list(o) for o in v] for t, v in C.HELDOUT_ORDERS.items()},
                        "order_seed": C.HELDOUT_ORDER_SEED,
                        "train_orders_per_type": {t: len(v) for t, v in C.TRAIN_ORDERS.items()},
                        "lists": "none in the register one-step prompt (no list-order factor)"},
            "ok": tr["ok"] and hsr["ok"]}


# ---------------------------------------------------------------------------------------------- build / verify
def _write_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1) + "\n")


def _rss_mb() -> float:
    """Peak working set (Windows) if available, else the current RSS, in MB."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)
    except Exception:
        return -1.0


def edge_describe(rows: list) -> dict:
    """cond_oor_edge (descriptive): offsets, branch, value signs, pair types, and how many members have the boundary
    across a decade (tens(other) != tens(threshold): the members where a digit-wise comparison can disagree)."""
    xs = [(r["before"][r["event"]["other"]], r["event"]["threshold"]) for r in rows]
    return {"n": len(rows), "offset_counts": {str(k): v for k, v in sorted(Counter(r["offset"] for r in rows).items())},
            "branch": dict(Counter(str(r["gold_ok"]) for r in rows)),
            "value_sign": dict(Counter("+" if r["event"]["value"] > 0 else "-" for r in rows)),
            "pair_types": dict(Counter(r["pair_type"] for r in rows)),
            "members_across_a_decade": sum(x // 10 != t // 10 for x, t in xs),
            "quadruples_with_a_member_across_a_decade": len({r["cluster"] for r, (x, t) in zip(rows, xs)
                                                             if x // 10 != t // 10}),
            "threshold_range": [min(t for _, t in xs), max(t for _, t in xs)], "seed": SEED_EDGE}


def describe_train(rows: list) -> dict:
    return {"n": len(rows), "by_stratum": dict(Counter(r["stratum"] for r in rows)),
            "by_category": dict(Counter(r["category"] for r in rows)),
            "branch": dict(Counter(str(r["gold_ok"]) for r in rows if r["category"] == "cond")),
            "cond_patterns_pairs": dict(Counter(r["pattern"] for r in rows if r["role"] == "core")),
            "cond_twin_types": dict(Counter(r["twin_type"] for r in rows if r["role"] == "twin")),
            "wrap_rows": sum(C.wraps(r["before"], r["event"]) for r in rows),
            "value_range": [min(r["event"]["value"] for r in rows), max(r["event"]["value"] for r in rows)],
            "threshold_range_cond": [min(r["event"]["threshold"] for r in rows if r["category"] == "cond"),
                                     max(r["event"]["threshold"] for r in rows if r["category"] == "cond")],
            "mean_target_tokens": {a: round(sum(r["n_target_tokens"][a] for r in rows) / len(rows), 2) for a in C.ARMS}}


def _jsonl_text(rows: list) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def build_content(lines: list, historical: bool = True) -> dict:
    """Everything the builder writes, in memory: the exact text of every data file (keyed by path) plus the rows and
    the union it was drawn against. Deterministic given the seeds and the union, so --verify can rebuild and compare."""
    n_tpl = check_template()
    log(lines, f"render_prompt == real chat template on {n_tpl} t2_train register prompts")
    m = env.load_marginals()
    if (min(m.r_values), max(m.r_values)) != C.IN_V or (min(m.r_thresholds), max(m.r_thresholds)) != C.THR_IN:
        raise SystemExit("recorded marginal ranges differ from t2br_common.IN_V / THR_IN")
    src, excl_pay, excl_ops = load_union(lines, historical)
    reg = Registry(excl_pay, excl_ops)
    count = C.token_counter()
    train = build_train(Draw(random.Random(SEED_TRAIN), m), reg, lines)
    train = [finish_row(r, count) for r in train]
    random.Random(SHUFFLE_SEED).shuffle(train)
    dev = build_dev(Draw(random.Random(SEED_DEV), m), reg, lines)
    dev = [finish_row(r, count) for r in dev]
    parents = build_chains(random.Random(SEED_CHAINS), m, reg, lines)
    st_rows = build_sametens(Draw(random.Random(SEED_SAMETENS), m), reg, lines, len(dev))   # revision 4, appended
    dev += [finish_row(r, count) for r in st_rows]
    eg_rows = build_edge(Draw(random.Random(SEED_EDGE), m), reg, lines, len(dev))           # revision 5, appended
    dev += [finish_row(r, count) for r in eg_rows]
    base = {a: train_rows_for_arm(train, a, None, count) for a in C.ARMS}
    rng_sd = random.Random(SEED_SD)
    surfaces = [C.draw_train_surface(rng_sd) for _ in train]
    sd = {a: train_rows_for_arm(train, a, surfaces, count) for a in C.ARMS}
    dev_out = [dev_row_out(r, count) for r in dev]
    hs = hs_rows(dev_out, count)
    texts = {**{C.train_path(a, "base"): _jsonl_text(base[a]) for a in C.ARMS},
             **{C.train_path(a, "sd"): _jsonl_text(sd[a]) for a in C.ARMS},
             C.DEV: _jsonl_text(dev_out), C.DEV_HS: _jsonl_text(hs), C.CHAINS: _jsonl_text(parents)}
    digests = {"train_rows": train_rows_digest(base), "dev_rows": dev_rows_digest(dev_out),
               "chains_sha256": hashlib.sha256(texts[C.CHAINS].encode("utf-8")).hexdigest()}
    same_rows = digests == {"train_rows": REV2_TRAIN_ROWS_DIGEST, "dev_rows": REV2_DEV_ROWS_DIGEST,
                            "chains_sha256": REV2_CHAINS_SHA256}
    log(lines, f"rows identical to revision 2 (never re-drawn): {same_rows}")
    rev3 = rev3_reconstruction(train, surfaces, dev, count)
    same_rev3 = rev3 == REV3_FILE_SHA256
    log(lines, f"revision-3 data files reconstructed byte for byte from these rows (FILL_UNITS_REV3, no cond_sametens): "
               f"{same_rev3}")
    rev4 = rev4_reconstruction(train, surfaces, dev, count)
    same_rev4 = rev4 == REV4_FILE_SHA256
    log(lines, f"revision-4 data files reconstructed byte for byte from these rows (FILL_UNITS_REV4, no cond_oor_edge): "
               f"{same_rev4}")
    return {"n_tpl": n_tpl, "src": src, "excl_pay": excl_pay, "excl_ops": excl_ops, "train": train, "dev": dev,
            "parents": parents, "base": base, "sd": sd, "surfaces": surfaces, "dev_out": dev_out, "hs": hs,
            "texts": texts, "count": count, "digests": digests, "same_rows_as_rev2": same_rows,
            "rev3_reconstruction": rev3, "same_rows_as_rev3": same_rev3, "rev4_reconstruction": rev4,
            "same_rows_as_rev4": same_rev4}


DESIGN = {"quota": TRAIN_QUOTA, "cond_offsets": COND_OFFSETS, "twin_dist": C.TWIN_DIST, "twin_types": TWIN_TYPES,
          "twin_type_quota_per_offset": TWIN_TYPE_QUOTA, "cond_pattern_quota": COND_PATTERN_QUOTA,
          "dev_pair_type_quota": DEV_PAIR_TYPE_QUOTA, "far_dist": C.FAR_DIST, "sametens_dist": C.SAMETENS_DIST,
          "edge_offsets": C.EDGE_OFFSETS, "edge_type_quota": EDGE_TYPE_QUOTA}
OUTPUTS = [C.train_path(a, "base") for a in C.ARMS] + [C.train_path(a, "sd") for a in C.ARMS] + \
    [C.train_source("base"), C.train_source("sd"), C.DEV, C.DEV_HS, C.CHAINS, C.DEV_SOURCE]


def surface_definition() -> dict:
    return {"paraphrases": list(C.SD_PARAPHRASES), "train_paraphrases": list(C.TRAIN_PARAPHRASES),
            "heldout_paraphrase": C.HELDOUT_PARAPHRASE, "fields": {f: list(v) for f, v in C.SD_FIELDS.items()},
            "train_name_idx": list(C.TRAIN_NAME_IDX), "heldout_name_idx": C.HELDOUT_NAME_IDX,
            "object_keys": {t: list(v) for t, v in C.OBJECT_KEYS.items()},
            "heldout_orders": {t: [list(o) for o in v] for t, v in C.HELDOUT_ORDERS.items()},
            "heldout_order_seed": C.HELDOUT_ORDER_SEED,
            "train_orders_per_type": {t: len(v) for t, v in C.TRAIN_ORDERS.items()}}


def build(historical: bool = True) -> int:
    ex = [str(p) for p in OUTPUTS if p.exists()]
    if ex:
        raise SystemExit(f"outputs exist (create-only): {ex}")
    t0 = time.time()
    lines: list = []
    log(lines, f"T2b-register data (revision {C.GO_RULE_REVISION}): seeds train {SEED_TRAIN} / dev {SEED_DEV} / chains "
               f"{SEED_CHAINS} / surfaces {SEED_SD} / cond_sametens {SEED_SAMETENS} / cond_oor_edge {SEED_EDGE}; no "
               "model call")
    bc = build_content(lines, historical)
    train, dev, parents, count = bc["train"], bc["dev"], bc["parents"], bc["count"]
    if not bc["same_rows_as_rev2"]:
        raise SystemExit(f"the rows differ from revision 2 (they must never be re-drawn): {bc['digests']}")
    if not bc["same_rows_as_rev3"]:
        raise SystemExit(f"the revision-3 files are not reconstructed from these rows: {bc['rev3_reconstruction']}")
    if not bc["same_rows_as_rev4"]:
        raise SystemExit(f"the revision-4 files are not reconstructed from these rows: {bc['rev4_reconstruction']}")
    aud = audit(train)
    log(lines, f"audit: {'PASS' if aud['ok'] else 'FAIL'} " + ", ".join(
        f"{k} {v['acc']:.3f}" for k, v in {**aud["gated"], **aud["rule_variants"]}.items()))
    if not aud["ok"]:
        raise SystemExit("surface-heuristic / rule-variant audit failed; nothing written")
    fca = fill_constant_audit(train, dev, parents, count)
    log(lines, "amendment 4b: " + ", ".join(f"{k} derived {v['derived_constant']} (pre-stated {v['pre_stated']}), "
                                             f"{v['distinct_fill_strings']} fill string(s)" for k, v in fca["per_kind"].items()))
    if not fca["ok"]:
        raise SystemExit("amendment-4b filler check failed (constants / zero information); nothing written")
    sfa = surface_audit(bc["base"], bc["sd"], bc["dev_out"], bc["hs"])
    log(lines, f"surface audit: training rows using a held-out factor {sfa['train_sd']['rows_using_a_heldout_factor']}; "
               f"held-out-surface bank ok {sfa['heldout_surface_bank']['ok']}")
    if not sfa["ok"]:
        raise SystemExit("surface audit failed; nothing written")
    dheur = dev_heuristics(dev)
    for p, text in bc["texts"].items():
        with p.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    src, excl_pay, excl_ops = bc["src"], bc["excl_pay"], bc["excl_ops"]
    excl = {k: {"payloads": len(v["payloads"]), "operative_contents": len(v["ops"]), "sha256": v.get("sha256"),
                **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"]}
                   if "skipped_by_name_never_opened" in v else {})} for k, v in src.items()}
    common = {"schema": 4, "revision": C.GO_RULE_REVISION, "experiment": "RCWM-T1 T2b-register screen (roadmap item 3; joint item 5b)",
              "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_calls": 0,
              "torch_imported": "torch" in sys.modules, "builder": "scripts/t2br_make_data.py",
              "builder_sha256_at_build": C.sha256(Path(__file__)),
              "common_sha256_at_build": C.sha256(ROOT / "scripts" / "t2br_common.py"),
              "builder_note": "the files' provenance is the --verify rebuild check (AUDIT.json): the CURRENT builder "
                              "must rebuild every data file byte for byte; the hashes above are the code at build time",
              "env_sha256": C.sha256(ROOT / "rcwmt" / "env.py"),
              "recorded_marginals_sha256": env.RECORDED_SHA, "tokenizer_json_sha256": C.sha256(C.TOKENIZER_JSON),
              "chat_template_check": f"render_prompt == real template on {bc['n_tpl']} t2_train register prompts",
              "rows_identical_to_revision_2": {"ok": bc["same_rows_as_rev2"], **bc["digests"]},
              "revision_3_files_reconstructed": {"ok": bc["same_rows_as_rev3"], **bc["rev3_reconstruction"]},
              "revision_4_files_reconstructed": {"ok": bc["same_rows_as_rev4"], **bc["rev4_reconstruction"]},
              "exclusion_sources": excl, "exclusion_union": {"payloads": len(excl_pay), "operative_contents": len(excl_ops)},
              "freshness": "zero payload and zero operative-content overlap with every source; train / dev / chain units "
                           "mutually disjoint (checked during the draw by Registry, re-checked by --verify)",
              "peak_rss_mb_at_write": _rss_mb()}
    _write_json(C.train_source("base"), {
        **common, "record": "training rows, canonical rendering (the rows' source of truth; NOT trained, C1)",
        "seed": SEED_TRAIN, "shuffle_seed": SHUFFLE_SEED,
        "files": {C.train_path(a, "base").name: C.sha256(C.train_path(a, "base")) for a in C.ARMS},
        "describe": describe_train(train), "audit": aud, "fill_constant": fca, "design": DESIGN, "build_log": lines})
    _write_json(C.train_source("sd"), {
        **common, "record": "training rows, surface-diversity rendering (the screen's ONLY training data, C1 + C2): the "
                            "SAME rows and targets, one seeded surface per row from the TRAIN subsets",
        "surface_seed": SEED_SD, "base_files": {C.train_path(a, "base").name: C.sha256(C.train_path(a, "base"))
                                                for a in C.ARMS},
        "files": {C.train_path(a, "sd").name: C.sha256(C.train_path(a, "sd")) for a in C.ARMS},
        "surface_definition": surface_definition(), "surface_audit_train": sfa["train_sd"],
        "mean_prompt_tokens": {"sd": round(sum(r["n_prompt_tokens"] for r in bc["sd"]["plain"]) / len(train), 2),
                               "base": round(sum(r["n_prompt_tokens"] for r in bc["base"]["plain"]) / len(train), 2)}})
    dv_desc = {"n": len(dev), "by_stratum": dict(Counter(r["stratum"] for r in dev)),
               "branch_by_stratum": {s: dict(Counter(str(r["gold_ok"]) for r in dev if r["stratum"] == s))
                                     for s in C.DEV_STRATA if s.startswith("cond")},
               "pair_types": {s: dict(Counter(r.get("pair_type") for r in dev if r["stratum"] == s))
                              for s in ("cond_mid", "cond_far", "cond_sametens", "cond_oor_edge")},
               "distance_range": {s: [min(abs(r["offset"]) for r in dev if r["stratum"] == s),
                                      max(abs(r["offset"]) for r in dev if r["stratum"] == s)]
                                  for s in ("cond_mid", "cond_far", "cond_sametens")},
               "cond_sametens": {"distance_counts": dict(sorted(Counter(abs(r["offset"]) for r in dev
                                                                        if r["stratum"] == "cond_sametens").items())),
                                 "value_sign": dict(Counter("+" if r["event"]["value"] > 0 else "-" for r in dev
                                                            if r["stratum"] == "cond_sametens")),
                                 "decades": dict(sorted(Counter(r["event"]["threshold"] // 10 for r in dev
                                                                if r["stratum"] == "cond_sametens").items())),
                                 "seed": SEED_SAMETENS},
               "cond_oor_edge": edge_describe([r for r in dev if r["stratum"] == "cond_oor_edge"]),
               "chains": {s: sum(p["stratum"] == s for p in parents) for s in C.CHAIN_STRATA},
               "chain_candidates": sum(len(p["candidates"]) for p in parents),
               "heldout_surface_rows": len(bc["hs"])}
    _write_json(C.DEV_SOURCE, {**common, "record": "dev screen bank (one-step strata + its held-out-surface rendering + "
                                                   "k=4 chains)", "seed": SEED_DEV, "chain_seed": SEED_CHAINS,
                               "strata": {k: list(v) for k, v in C.DEV_STRATA.items()},
                               "files": {C.DEV.name: C.sha256(C.DEV), C.DEV_HS.name: C.sha256(C.DEV_HS),
                                         C.CHAINS.name: C.sha256(C.CHAINS)},
                               "sametens_seed": SEED_SAMETENS, "edge_seed": SEED_EDGE, "describe": dv_desc,
                               "dev_heuristics": dheur,
                               "surface_audit_heldout": sfa["heldout_surface_bank"],
                               "surface_definition": surface_definition(), "go_rule": C.GO_RULE_TEXT,
                               "go_rule_revision": C.GO_RULE_REVISION, "build_log": lines})
    log(lines, f"written in {time.time() - t0:.0f}s; peak RSS ~{_rss_mb()} MB")
    return 0


# ---------------------------------------------------------------------------------------------- verify
def verify(historical: bool = True, write_record: bool = True, quiet: bool = False) -> tuple[bool, list]:
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:400]))
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)

    lines: list = []
    tsrc, ssrc, dsrc = (C.read_json(C.train_source("base")) or {}, C.read_json(C.train_source("sd")) or {},
                        C.read_json(C.DEV_SOURCE) or {})
    chk("train files (base + sd) match their SOURCE sha256",
        all(C.sha256(C.train_path(a, "base")) == tsrc["files"][C.train_path(a, "base").name] for a in C.ARMS)
        and all(C.sha256(C.train_path(a, "sd")) == ssrc["files"][C.train_path(a, "sd").name] for a in C.ARMS))
    chk("dev + held-out-surface + chain files match their SOURCE sha256", all(
        C.sha256(p) == dsrc["files"][p.name] for p in (C.DEV, C.DEV_HS, C.CHAINS)))
    chk("render_prompt == real chat template (t2_train register prompts)", check_template() > 0)
    count = C.token_counter()
    tr = {a: C.load_jsonl(C.train_path(a, "base")) for a in C.ARMS}
    sdr = {a: C.load_jsonl(C.train_path(a, "sd")) for a in C.ARMS}
    dev = C.load_dev()
    hs = C.load_dev_hs()
    parents = C.load_chain_bank()
    n = len(tr["plain"])
    key = lambda x: (x["id"], env.dumps(x["before"]), env.dumps(x["event"]), x["target"], x["gold_after"])  # noqa: E731
    chk("six renderings of the SAME rows: base and sd, each arm (ids, before, event, golds, order; sd targets == base "
        "targets)", all(len(tr[a]) == n and len(sdr[a]) == n and
                        all((x["id"], env.dumps(x["before"]), env.dumps(x["event"])) ==
                            (y["id"], env.dumps(y["before"]), env.dumps(y["event"])) for x, y in zip(tr["plain"], tr[a]))
                        and [key(x) for x in tr[a]] == [key(y) for y in sdr[a]] for a in C.ARMS), f"{n} rows")
    chk("identical prompts across arms (base; sd)", all(x["prompt"] == y["prompt"] == z["prompt"]
                                                        for d in (tr, sdr) for x, y, z in zip(d["plain"], d["trace"], d["filler"])))
    bad_gold = bad_t = bad_tok = bad_order = 0
    for a in C.ARMS:
        for r, s in zip(tr[a], sdr[a]):
            tg = C.targets(r["before"], r["event"])
            env.cross_check(FAM, r["before"], r["event"])
            bad_gold += tg["after"] != r["gold_after"] or env.dumps(env.step(FAM, r["before"], r["event"])) != r["gold_after"]
            bad_t += r["target"] != tg[a] or s["target"] != tg[a]
            resid = count(tg["filler"]) - count(tg["trace"])
            for x in (r, s):
                bad_tok += count(x["target"]) != x["n_target_tokens"] or count(x["prompt"]) != x["n_prompt_tokens"] \
                    or x["fill_units"] != C.FILL_UNITS[x["event"]["kind"]] or x["fill_residual"] != resid
            if a == "trace":
                p = C.parse_trace(r["target"])
                bad_order += p is None or not r["target"].startswith('{"cmp":')
            bad_t += r["prompt"] != C.render_prompt(C.canonical_messages(r["before"], r["event"])) or r["surface"] is not None
            bad_t += s["prompt"] != C.render_prompt(C.surface_messages(s["before"], s["event"], s["surface"]))
    chk("training golds = env.step = FDS oracle; targets re-derived; base prompts canonical; sd prompts = their recorded "
        "surface", bad_gold == 0 and bad_t == 0, f"gold {bad_gold}, target/prompt {bad_t}")
    chk("filler targets carry the kind's constant fill (amendment 4b); recorded token counts and residuals exact",
        bad_tok == 0, f"{bad_tok} mismatches")
    chk("trace key order literal (cmp, calc, after) and parseable", bad_order == 0)
    d_bad = 0
    for r in dev:
        tg = C.targets(r["before"], r["event"])
        env.cross_check(FAM, r["before"], r["event"])
        d_bad += tg["after"] != r["gold_after"] or any(tg[a] != r["targets"][a] for a in C.ARMS) \
            or not C.same_json(tg["cmp"], r["gold_cmp"]) or not C.same_json(tg["calc"], r["gold_calc"]) \
            or r["fill_units"] != C.FILL_UNITS[r["event"]["kind"]] \
            or r["fill_residual"] != count(r["targets"]["filler"]) - count(r["targets"]["trace"]) \
            or any(count(r["targets"][a]) != r["n_target_tokens"][a] for a in C.ARMS) \
            or r["messages"] != C.canonical_messages(r["before"], r["event"])
    chk("dev golds / targets / token counts re-derived; in-surface prompts are Titan's verbatim frame", d_bad == 0,
        f"{d_bad} bad")
    cnt = Counter(r["stratum"] for r in dev)
    chk("dev strata sizes as designed", dict(cnt) == C.DEV_N, dict(cnt))
    # stratum properties
    sp = []
    for r in dev:
        b, ev, s = r["before"], r["event"], r["stratum"]
        v = ev.get("value", 0)
        inr = C.IN_V[0] <= v <= C.IN_V[1]
        oorv = C.OOR_V_ABS[0] <= abs(v) <= C.OOR_V_ABS[1]
        cond = ev["kind"] == "conditional"
        x = b[ev["other"]] if cond else None
        dist = abs(x - ev["threshold"]) if cond else None
        thr_in = cond and C.THR_IN[0] <= ev["threshold"] <= C.THR_IN[1]
        ok = {"mix": ev["kind"] == "mix" and inr and C.wraps(b, ev) == r["wrap"],
              "wrap": ev["kind"] == "add" and inr and C.wraps(b, ev),
              "cond_xm1": cond and inr and x == ev["threshold"] - 1 and v % C.MOD != 0,
              "cond_x0": cond and inr and x == ev["threshold"] and v % C.MOD != 0,
              "cond_xp1": cond and inr and x == ev["threshold"] + 1 and v % C.MOD != 0,
              "cond_mid": cond and inr and v % C.MOD != 0 and thr_in and C.TWIN_DIST[0] <= dist <= C.TWIN_DIST[1],
              "swap": ev["kind"] == "swap" and b[ev["register"]] != b[ev["other"]],
              "add_nowrap": ev["kind"] == "add" and inr and v != 0 and not C.wraps(b, ev),
              "mix_oor": ev["kind"] == "mix" and oorv and C.wraps(b, ev) == r["wrap"],
              "wrap_oor": ev["kind"] == "add" and oorv and C.wraps(b, ev),
              "cond_oor_val": cond and oorv and x - ev["threshold"] in (0, 1),
              "cond_oor_thr": cond and inr and v != 0 and (
                  C.OOR_THR[0][0] <= ev["threshold"] <= C.OOR_THR[0][1] or C.OOR_THR[1][0] <= ev["threshold"] <= C.OOR_THR[1][1]),
              "cond_far": cond and inr and v % C.MOD != 0 and thr_in and C.FAR_DIST[0] <= dist <= C.FAR_DIST[1],
              "cond_sametens": cond and sametens_problem(b, ev) is None,
              "cond_oor_edge": cond and edge_problem(b, ev) is None}[s]
        ok = ok and (C.is_oor(b, ev) == (C.DEV_STRATA[s][0] == "oor"))
        if not ok:
            sp.append(r["id"])
    chk("every dev row satisfies its stratum definition (and C.is_oor agrees with its group)", not sp, sp[:5])
    clus = {}
    for r in dev:
        clus.setdefault(r["cluster"], []).append(r)
    multi = {c: rs for c, rs in clus.items() if len(rs) > 1}
    pair_bad = []
    for c, rs in multi.items():
        want = rs[0].get("pair_type", "threshold")
        if not all(C.single_diff((rs[0]["before"], rs[0]["event"]), (q["before"], q["event"])) == want for q in rs[1:]):
            pair_bad.append(c)
        if rs[0]["stratum"] in ("cond_mid", "cond_far", "cond_oor_val", "cond_oor_thr", "cond_sametens") and \
                sorted(bool(q["gold_ok"]) for q in rs) != [False, True]:
            pair_bad.append(c)
        if rs[0]["stratum"] == "cond_oor_edge" and ([C.cond_dist(q["before"], q["event"]) for q in rs] !=
                                                    list(C.EDGE_OFFSETS) or [bool(q["gold_ok"]) for q in rs] !=
                                                    [False, False, True, True]):
            pair_bad.append(c)
    bal = {s: Counter(r["gold_ok"] for r in dev if r["stratum"] == s) for s in ("cond_mid", "cond_oor_val", "cond_oor_thr",
                                                                                 "cond_far", "cond_sametens",
                                                                                 "cond_oor_edge")}
    ptypes = {s: Counter(r.get("pair_type") for r in dev if r["stratum"] == s) for s in ("cond_mid", "cond_far",
                                                                                          "cond_sametens",
                                                                                          "cond_oor_edge")}
    chk("pairs / triplets / quadruples differ in ONE number (the threshold; cond_mid and cond_sametens pairs: the "
        "threshold or the other register, 25 pairs each; cond_far pairs: the other register; cond_oor_edge quadruples: "
        "the threshold or the other register, 10 each, at other - threshold = -1, 0, +1, +2); pair strata "
        "branch-balanced, one ok-true + one ok-false member per pair (cond_oor_edge: two of each per quadruple)",
        not pair_bad and all(v[True] == v[False] for v in bal.values())
        and ptypes == {"cond_mid": Counter({"threshold": 50, "other": 50}), "cond_far": Counter({"other": 100}),
                       "cond_sametens": Counter({"threshold": 50, "other": 50}),
                       "cond_oor_edge": Counter({"threshold": 40, "other": 40})},
        f"bad {pair_bad[:3]}; {({k: dict(v) for k, v in bal.items()})}")
    # training design
    trc = tr["plain"]
    tcl = {}
    for r in trc:
        tcl.setdefault(r["cluster"], []).append(r)
    twin_bad = []
    for c, rs in tcl.items():
        if not c.startswith("tp"):
            continue
        core = [q for q in rs if q["role"] == "core"]
        twin = [q for q in rs if q["role"] == "twin"]
        if len(core) != 1 or len(twin) != 1:
            twin_bad.append(c)
            continue
        core, twin = core[0], twin[0]
        if C.single_diff((core["before"], core["event"]), (twin["before"], twin["event"])) != twin["twin_type"] \
                or core["gold_ok"] == twin["gold_ok"] \
                or C.cond_dist(core["before"], core["event"]) not in COND_OFFSETS \
                or not C.TWIN_DIST[0] <= abs(C.cond_dist(twin["before"], twin["event"])) <= C.TWIN_DIST[1]:
            twin_bad.append(c)
    core_off = Counter(C.cond_dist(r["before"], r["event"]) for r in trc if r["role"] == "core")
    ttype = Counter((C.cond_dist(r["before"], r["event"]), r["twin_type"]) for r in trc if r["role"] == "core")
    chk("training conditionals: 150 core rows at x - thr = -1 / 0 / +1 (50 each) + 150 twins differing in ONE number "
        "(25 threshold / 25 other per offset), branch flipped, |x - thr| in 2..15",
        not twin_bad and core_off == Counter({-1: 50, 0: 50, 1: 50}) and set(ttype.values()) == {25},
        f"{sum(1 for c in tcl if c.startswith('tp'))} pairs; bad {twin_bad[:3]}")
    chk("training value / threshold / register ranges in the recorded marginal", all(
        C.IN_V[0] <= r["event"]["value"] <= C.IN_V[1] and C.THR_IN[0] <= r["event"]["threshold"] <= C.THR_IN[1]
        and all(C.THR_IN[0] <= x <= C.THR_IN[1] for x in r["before"].values()) for r in trc))
    chk("no swap / add-no-wrap row in training (controls untouched)", not any(
        r["event"]["kind"] == "swap" or (r["event"]["kind"] == "add" and not C.wraps(r["before"], r["event"])) for r in trc))
    # chains
    ch_ok = True
    for p in parents:
        finals = {env.dumps(env.rollout(FAM, p["s0"], c["events"])[-1]) for c in p["candidates"]}
        ch_ok &= len(p["candidates"]) == C.K and all(len(c["events"]) == C.H for c in p["candidates"]) \
            and len(finals) >= MIN_DISTINCT_FINALS and p["split"] == "dev"
        if p["stratum"] == "CH_X":
            ch_ok &= all(C.OOR_V_ABS[0] <= abs(e["value"]) <= C.OOR_V_ABS[1] for c in p["candidates"]
                         for e in c["events"] if e["kind"] != "swap")
        for c in p["candidates"]:
            st = env.rollout(FAM, p["s0"], c["events"])
            for t, e in enumerate(c["events"]):
                env.cross_check(FAM, st[t], e)
    chk("chains: 8 x 4 per parent, >= 4 distinct finals, CH_X values out of range, oracle cross-check",
        ch_ok and Counter(p["stratum"] for p in parents) == Counter(C.CHAIN_STRATA), dict(Counter(p["stratum"] for p in parents)))
    dg = {"train_rows": train_rows_digest(tr), "dev_rows": dev_rows_digest(dev), "chains_sha256": C.sha256(C.CHAINS)}
    chk("rows identical to revision 2 (never re-drawn): training rows + plain / trace targets + canonical prompts, dev "
        "rows of the revision-2/3 strata, chain bank byte for byte",
        dg == {"train_rows": REV2_TRAIN_ROWS_DIGEST, "dev_rows": REV2_DEV_ROWS_DIGEST, "chains_sha256": REV2_CHAINS_SHA256},
        dg)
    r2 = rev2_reconstruction(tr, dev, count)
    chk("the four revision-2 data files are reconstructed byte for byte from the current rows (their recorded sha256)",
        r2 == REV2_FILE_SHA256, {k: v[:12] for k, v in r2.items()})
    r3 = rev3_reconstruction(internal_train_rows(tr["plain"]), [x["surface"] for x in sdr["plain"]],
                             internal_dev_rows(dev), count)
    chk("the eight revision-3 data files are reconstructed byte for byte from the current rows with revision 3's fill "
        "constants and strata (their recorded sha256): no row changed; revision 4 only appended cond_sametens and "
        "re-derived the 4b constants", r3 == REV3_FILE_SHA256, {k: v[:12] for k, v in r3.items()})
    r4 = rev4_reconstruction(internal_train_rows(tr["plain"]), [x["surface"] for x in sdr["plain"]],
                             internal_dev_rows(dev), count)
    chk("the eight revision-4 data files are reconstructed byte for byte from the current rows with revision 4's fill "
        "constants and strata (their recorded sha256): no row and no constant changed; revision 5 only appends "
        "cond_oor_edge", r4 == REV4_FILE_SHA256, {k: v[:12] for k, v in r4.items()})
    first = [r for r in dev if r["stratum"] in REV3_STRATA]
    first4 = [r for r in dev if r["stratum"] in REV4_STRATA]
    chk("the appends keep every earlier row in place: dev ids dv0000..dv1099 in revision-3 order, then cond_sametens "
        "(dv1100..dv1199, revision 4), then cond_oor_edge (dv1200..dv1279, revision 5); the held-out-surface bank's first "
        "400 rows are revision 3's bank and its first 500 revision 4's (same ids, hs_index, prompts)",
        dev[:len(first)] == first and dev[:len(first4)] == first4
        and [r["id"] for r in dev] == [f"dv{i:04d}" for i in range(len(dev))]
        and all(r["stratum"] == "cond_sametens" for r in dev[len(first):len(first4)])
        and all(r["stratum"] == "cond_oor_edge" for r in dev[len(first4):])
        and hashlib.sha256(_jsonl_text(hs[:400]).encode("utf-8")).hexdigest() == REV3_FILE_SHA256[C.DEV_HS.name]
        and hashlib.sha256(_jsonl_text(hs[:500]).encode("utf-8")).hexdigest() == REV4_FILE_SHA256[C.DEV_HS.name])
    # amendment 4b and the surface proof (data only)
    trc_full = [{**r, "targets": {a: tr[a][i]["target"] for a in C.ARMS},
                 "n_target_tokens": {a: tr[a][i]["n_target_tokens"] for a in C.ARMS}} for i, r in enumerate(trc)]
    fca = fill_constant_audit(trc_full, dev, parents, count)
    chk("amendment 4b (C4, revision-4 definition): the per-kind constant (smallest fill whose SHORTEST filler target is "
        ">= the kind's LONGEST trace target over train + dev + chain gold steps) re-derived == the pre-stated FILL_UNITS; "
        "ONE fill string per kind (every filler target identical up to 'after'); fill length carries zero information "
        "about wrap / branch in every cell (best cut == majority); residual >= 0", fca["ok"],
        {k: (v["derived_constant"], v["shortest_filler_tokens"], v["longest_trace_tokens"], v["distinct_fill_strings"])
         for k, v in fca["per_kind"].items()})
    sfa = surface_audit(tr, sdr, dev, hs)
    chk("C2: NO training prompt uses a held-out paraphrase, field name or key order (every sd prompt decoded by "
        "read_prompt to its row and recorded surface; raw-string check); every train paraphrase / name / state order "
        "used", sfa["train_sd"]["ok"],
        {k: sfa["train_sd"][k] for k in ("decode_failures", "rows_using_a_heldout_factor",
                                         "rows_with_a_heldout_name_or_paraphrase_string")})
    chk(f"C3: the held-out-surface bank = the {sum(C.DEV_N[s] for s in C.HS_STRATA)} dev rows of the primary strata + "
        "the identification strata (cond_sametens, cond_oor_edge) in dev order (same ids and golds); every prompt decodes "
        "to its row and uses ONLY held-out factors (paraphrase 4, names 4, held-out orders rotating by row)",
        sfa["heldout_surface_bank"]["ok"], {k: sfa["heldout_surface_bank"][k] for k in ("n_rows", "bad_rows")})
    # rebuild with the CURRENT builder (fresh union) and compare bytes; then freshness on the same union
    rebuilt = None
    if historical:
        bc = build_content(lines, True)
        src, excl_pay, excl_ops = bc["src"], bc["excl_pay"], bc["excl_ops"]
        diffs = [p.name for p, text in bc["texts"].items() if p.read_bytes() != text.encode("utf-8")]
        rebuilt = {"identical": not diffs, "differs": diffs,
                   "files": {p.name: hashlib.sha256(text.encode("utf-8")).hexdigest() for p, text in bc["texts"].items()}}
        chk("the CURRENT builder rebuilds every data file byte for byte (fresh name-skipping harvest)", not diffs, diffs)
        del bc
    else:
        src, excl_pay, excl_ops = load_union(lines, False)
    reg = Registry(excl_pay, excl_ops)
    units = []
    for r in trc:
        units.append((r["cluster"], [(r["before"], r["event"])]))
    for r in dev:
        units.append((r["cluster"], [(r["before"], r["event"])]))
    for p in parents:
        items = []
        for c in p["candidates"]:
            st = env.rollout(FAM, p["s0"], c["events"])
            items += [(st[t], e) for t, e in enumerate(c["events"])]
        units.append((p["parent_id"], items))
    probs = Counter()
    for owner, items in units:
        pr = reg.problem(items, owner)
        if pr:
            probs[pr] += 1
        reg.add(items, owner)
    chk("freshness: zero payload / operative-content overlap with every source; units mutually disjoint" +
        ("" if historical else " (historical harvest SKIPPED: --no-historical; rebuild check skipped)"), not probs, dict(probs))
    if historical:
        skipped = src["historical"]["skipped_by_name_never_opened"]
        chk("historical harvest skipped exactly the HSKILL_CONFIRM_V2* files, by name (never opened)",
            skipped and all(Path(p).name.startswith("HSKILL_CONFIRM_V2") for p in skipped), skipped)
    union_size = {"payloads": len(excl_pay), "operative_contents": len(excl_ops), "sources": len(src)}
    del excl_pay, excl_ops, reg
    aud = audit(trc)
    chk("audit: surface heuristics <= 0.55 (copy, unreduced, always-add/-subtract, wrong register, other > 48, "
        "threshold < 48, value > 0); rule variants contradicted by >= 1/6 of the conditional rows; branch 150 / 150",
        aud["ok"], ", ".join(f"{k} {v['acc']:.3f}" for k, v in {**aud["gated"], **aud["rule_variants"]}.items()))
    chk("audit recorded at build == recomputed", json.dumps([aud["gated"], aud["rule_variants"]], sort_keys=True) ==
        json.dumps([tsrc["audit"]["gated"], tsrc["audit"]["rule_variants"]], sort_keys=True))
    dheur = dev_heuristics(dev)
    chk("data-only tables recorded at build == recomputed (4b constants and information test, surface audit, dev "
        "heuristic floors)",
        json.dumps(fca, sort_keys=True) == json.dumps(tsrc.get("fill_constant"), sort_keys=True)
        and json.dumps(sfa["train_sd"], sort_keys=True) == json.dumps(ssrc.get("surface_audit_train"), sort_keys=True)
        and json.dumps(sfa["heldout_surface_bank"], sort_keys=True) == json.dumps(dsrc.get("surface_audit_heldout"),
                                                                                    sort_keys=True)
        and json.dumps(dheur, sort_keys=True) == json.dumps(dsrc.get("dev_heuristics"), sort_keys=True))
    chk(f"the dev SOURCE states the current go rule (revision {C.GO_RULE_REVISION})", dsrc.get("go_rule") == C.GO_RULE_TEXT
        and dsrc.get("go_rule_revision") == C.GO_RULE_REVISION)
    ok = all(c[1] for c in checks)
    if write_record:
        frozen = (ROOT / "docs" / "t2b" / "register" / "SCREEN_FREEZE.json").exists()
        if frozen:
            print("SCREEN_FREEZE.json exists: AUDIT.json is a frozen input and is not rewritten")
        else:
            C.DOCS.mkdir(parents=True, exist_ok=True)
            files = [C.train_path(a, v) for v in ("base", "sd") for a in C.ARMS] + \
                [C.DEV, C.DEV_HS, C.CHAINS, C.train_source("base"), C.train_source("sd"), C.DEV_SOURCE]
            rec = {"ok": ok, "revision": C.GO_RULE_REVISION, "historical_harvest": historical,
                   "checks": [{"check": a, "ok": b, "detail": c} for a, b, c in checks],
                   "rebuild": rebuilt, "exclusion_union": union_size, "audit": aud, "fill_constant": fca,
                   "surface": sfa, "dev_heuristics": dheur, "train_describe": describe_train(trc_full),
                   "rows_identical_to_revision_2": dg, "revision_2_files_reconstructed": r2,
                   "revision_3_files_reconstructed": r3, "revision_4_files_reconstructed": r4,
                   "dev_describe": {"by_stratum": dict(cnt), "chains": dict(Counter(p["stratum"] for p in parents)),
                                    "heldout_surface_rows": len(hs),
                                    "branch_by_stratum": {s: dict(Counter(str(r["gold_ok"]) for r in dev if r["stratum"] == s))
                                                          for s in C.DEV_STRATA if s.startswith("cond")}},
                   "files_sha256": {p.name: C.sha256(p) for p in files},
                   "code_sha256": {r: C.sha256(ROOT / r) for r in ("scripts/t2br_make_data.py", "scripts/t2br_common.py")},
                   "torch_imported": "torch" in sys.modules, "peak_rss_mb": _rss_mb(),
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with (C.DOCS / "AUDIT.json").open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec, indent=1) + "\n")
            print(f"wrote {C.DOCS / 'AUDIT.json'}")
    print(f"verify: {sum(c[1] for c in checks)}/{len(checks)} checks pass")
    return ok, checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-historical", action="store_true", help="--verify only: skip the records-tree harvest (quick)")
    a = ap.parse_args()
    if a.verify:
        ok, _ = verify(historical=not a.no_historical)
        return 0 if ok else 1
    return build()


if __name__ == "__main__":
    sys.exit(main())
