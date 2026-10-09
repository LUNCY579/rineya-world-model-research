"""T2b-register screen data: fresh oracle register rows rendered three ways (plain / trace / filler), the dev screen
bank (one-step strata + k=4 chains). Built WITHOUT any model call and without torch (Titan's tokenizer is read through
the `tokenizers` library only, to count tokens). Create-only.

  python -B scripts/t2br_make_data.py                          # writes data/t2br_train_{plain,trace,filler}.jsonl +
                                                               # data/t2br_train.SOURCE.json, data/t2br_dev.jsonl +
                                                               # data/t2br_chains.jsonl + data/t2br_dev.SOURCE.json
  python -B scripts/t2br_make_data.py --surface-diversity on   # ALSO renders the SAME training rows with rule-text
                                                               # paraphrases + field-name variants into
                                                               # data/t2br_train_sd_{arm}.jsonl (+ .SOURCE.json); needs
                                                               # the base files; golds / targets unchanged
  python -B scripts/t2br_make_data.py --verify [--no-historical]
                                                               # re-derives everything from the files (golds, targets,
                                                               # token equality, freshness incl. a fresh name-skipping
                                                               # harvest, the surface-heuristic audit); writes
                                                               # docs/t2b/register/AUDIT.json

TRAINING ROWS (750, register only; Titan's verbatim one-step frame; values / thresholds / registers from the recorded
CANDIDATE_VIEWS marginals, so -8 <= value <= 11 and 0 <= threshold <= 96):
  mix 300        150 wrap / 150 no-wrap (raw r + 2 o + v outside / inside [0, 96]); the state always changes
  wrap 150       add whose raw r + v leaves [0, 96]: 75 over (v > 0) / 75 under (v < 0)
  conditional 300 = 150 core rows at other = threshold - 1 / threshold / threshold + 1 (50 each; value != 0) + 150
                 MINIMAL-PAIR TWINS that differ from their core row ONLY in the threshold and flip the branch:
                   core x = thr - 1 (ok false) -> twin thr - 2 (x = thr' + 1, ok true)
                   core x = thr     (ok false) -> twin thr - 1 (x = thr' + 1, ok true)
                   core x = thr + 1 (ok true)  -> twin thr + 1 (x = thr',     ok false)
                 so add / subtract is exactly 150 / 150 and no input field other than (other vs threshold) separates
                 the branches. Pair wrap patterns are balanced (1/3 neither twin wraps, 1/3 the add twin wraps, 1/3 the
                 subtract twin wraps), which keeps the unreduced-sum shortcut below the audit bar overall.
  Swap and add/no-wrap are deliberately absent (untouched skills = the screen's controls).

DEV SCREEN BANK (t2br_common.DEV_STRATA; 900 one-step rows + 48 chain parents x 8 candidates x 4 steps). In range:
mix 100, wrap 80, cond_xm1 / cond_x0 / cond_xp1 60 each (60 TRIPLETS: same row, threshold = x + 1 / x / x - 1), swap
80, add_nowrap 80. Out of range (PRIMARY): mix_oor 100 and wrap_oor 80 (|value| 12..40), cond_oor_val 100 (50 boundary
pairs, |value| 12..40), cond_oor_thr 100 (50 pairs, threshold -40..-1 vs 97..136). Chains: CH_IN 32 parents (env.gen_
event's recorded marginals) and CH_X 16 parents (the same, every add / mix / conditional value replaced by |v| 12..40);
each parent needs >= 4 distinct final states among its 8 candidates.

FRESHNESS. No (before, event) payload and no FDS operative content (generate.operative_key) of any training row, dev row
or chain transition equals one in the union used by PREREG-3's builder (scripts/p3_make_bank.load_all_sources: the
records tree harvested by scripts/l1_make_ood.harvest_historical, which skips files named HSKILL_CONFIRM_V2* BY NAME
before any open - rcwmt.bank.load_exclusions() is NOT called; every transition of bank_pilot / bank_confirm /
bank_confirm2 / bank_s4000; t2_train / t2_eval / t2_eval_xframe / t2_eval_supp; SELECT-VIEWS-001; the L1 OOD bank; the
FDS-era T_BANK* files), nor any transition of a multi-event register prompt in those files (rolled out with env.step).
The PREREG-3 bank is ledger-only (p3_common.LEDGER) and cannot share a register payload; it is not opened. Training,
dev and chain units are mutually disjoint at payload and operative-content level.
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
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
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
SHUFFLE_SEED = 7                 # as make_t2_data.py
MAX_DRAWS = 5_000_000
TRAIN_QUOTA = OrderedDict([("mix/wrap", 150), ("mix/nowrap", 150), ("add/wrap_over", 75), ("add/wrap_under", 75)])
COND_OFFSETS = (-1, 0, 1)
COND_PATTERNS = ("none", "add_wraps", "sub_wraps")
COND_PATTERN_QUOTA = {-1: (17, 17, 16), 0: (17, 16, 17), 1: (16, 17, 17)}   # 50 pairs per core offset
TWIN_SHIFT = {-1: -2, 0: -1, 1: 1}                                           # twin threshold = thr + shift
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
    # conditional core + minimal-pair twins
    pq = {off: dict(zip(COND_PATTERNS, q)) for off, q in COND_PATTERN_QUOTA.items()}
    rej, draws, npair = Counter(), 0, 0
    while any(n for q in pq.values() for n in q.values()):
        draws += 1
        if draws > MAX_DRAWS:
            raise SystemExit(f"conditional quota not filled: {pq}")
        off = d.rng.choice([o for o in COND_OFFSETS if any(pq[o].values())])
        thr = d.rng.choice(d.thresholds)
        x = thr + off
        thr2 = thr + TWIN_SHIFT[off]
        if not (C.THR_IN[0] <= x <= C.THR_IN[1] and C.THR_IN[0] <= thr2 <= C.THR_IN[1]):
            rej["x or twin threshold outside 0..96"] += 1
            continue
        ev = d.event("conditional", d.rng.choice(d.values_nz), thr)
        b = d.before()
        b[ev["other"]] = x
        ev2 = dict(ev, threshold=thr2)
        if C.branch_ok(b, ev) == C.branch_ok(b, ev2):
            raise AssertionError("twin does not flip the branch")
        (ba, ea), (bs, es) = ((b, ev), (b, ev2)) if C.branch_ok(b, ev) else ((b, ev2), (b, ev))
        pat = _pattern(ba, ea, bs, es)
        if pat is None or not pq[off][pat]:
            rej["wrap pattern full / both wrap"] += 1
            continue
        cid = f"tp{npair:03d}"
        pr = reg.problem([(b, ev), (b, ev2)], cid)
        if pr:
            rej[pr] += 1
            continue
        reg.add([(b, ev), (b, ev2)], cid)
        name = {-1: "xm1", 0: "x0", 1: "xp1"}
        rows.append({"uid": f"{cid}a", "category": "cond", "stratum": f"cond/{name[off]}", "cluster": cid, "role": "core",
                     "offset": off, "pattern": pat, "before": b, "event": ev})
        off2 = b[ev["other"]] - thr2
        rows.append({"uid": f"{cid}b", "category": "cond", "stratum": f"cond/{name[off2]}", "cluster": cid,
                     "role": "twin", "offset": off2, "pattern": pat, "before": dict(b), "event": ev2})
        pq[off][pat] -= 1
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
    log(lines, f"dev: {len(rows)} rows {dict(Counter(r['stratum'] for r in rows))}; rejected {dict(rej)}")
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
def finish_row(r: dict, count) -> dict:
    """Gold, the three targets and their token counts for one (before, event)."""
    b, ev = r["before"], r["event"]
    env.cross_check(FAM, b, ev)
    tg = C.targets(b, ev, count)
    if count(tg["trace"]) != count(tg["filler"]):
        raise AssertionError("filler / trace token counts differ")
    return {**r, "op": C.op_class(b, ev), "gold_after": tg["after"], "gold_cmp": tg["cmp"], "gold_calc": tg["calc"],
            "gold_ok": None if tg["cmp"] is None else tg["cmp"]["ok"], "fill_units": tg["fill_units"],
            "targets": {a: tg[a] for a in C.ARMS}, "n_target_tokens": {a: count(tg[a]) for a in C.ARMS}}


def train_rows_for_arm(rows: list, arm: str, sd_draws: list | None, count) -> list:
    out = []
    for i, r in enumerate(rows):
        sd = None if sd_draws is None else sd_draws[i]
        msgs = C.sd_messages(r["before"], r["event"], sd)
        prompt = C.render_prompt(msgs)
        out.append({"id": r["uid"], "prompt": prompt, "target": r["targets"][arm], "arm": arm, "family": FAM,
                    "category": r["category"], "stratum": r["stratum"], "op": r["op"], "cluster": r["cluster"],
                    "role": r["role"], "offset": r.get("offset"), "pattern": r.get("pattern"),
                    "before": r["before"], "event": r["event"], "gold_after": r["gold_after"], "gold_cmp": r["gold_cmp"],
                    "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"], "fill_units": r["fill_units"],
                    "n_prompt_tokens": count(prompt), "n_target_tokens": r["n_target_tokens"][arm], "sd": sd})
    return out


def dev_row_out(r: dict, count) -> dict:
    msgs = C.canonical_messages(r["before"], r["event"])
    return {"id": r["uid"], "stratum": r["stratum"], "group": r["group"], "cluster": r["cluster"], "role": r["role"],
            "family": FAM, "before": r["before"], "event": r["event"], "messages": msgs, "op": r["op"],
            "gold_after": r["gold_after"], "gold_cmp": r["gold_cmp"], "gold_calc": r["gold_calc"], "gold_ok": r["gold_ok"],
            "fill_units": r["fill_units"], "targets": r["targets"], "n_target_tokens": r["n_target_tokens"],
            "n_prompt_tokens": count(C.render_prompt(msgs)),
            **{k: r[k] for k in ("offset", "wrap", "direction", "thr_band") if k in r}}


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
def audit(rows: list) -> dict:
    """Surface-heuristic accuracies (exact after) on the training rows; gated <= 0.55 globally and in the categories
    where each heuristic is the candidate shortcut (copy: all; unreduced: mix, wrap; always_add / always_sub: cond)."""
    cats = {"all": rows, "mix": [r for r in rows if r["category"] == "mix"],
            "wrap": [r for r in rows if r["category"] == "wrap"], "cond": [r for r in rows if r["category"] == "cond"]}
    res = {"n": {k: len(v) for k, v in cats.items()}, "max": C.AUDIT_MAX, "acc": {}, "gated": {}, "diagnostic": {}}
    for h in C.GATED_HEURISTICS + C.DIAG_HEURISTICS:
        res["acc"][h] = {}
        for cname, rs in cats.items():
            app = [r for r in rs if C.heuristic_after(h, r["before"], r["event"]) is not None]
            if not app:
                continue
            ok = sum(env.dumps(C.heuristic_after(h, r["before"], r["event"])) == r["gold_after"] for r in app)
            res["acc"][h][cname] = {"n": len(app), "acc": round(ok / len(app), 4)}
    gate_cells = {"copy": ("all", "mix", "wrap", "cond"), "unreduced": ("all", "mix", "wrap"),
                  "always_add": ("cond",), "always_sub": ("cond",)}
    for h, cells in gate_cells.items():
        for c in cells:
            a = res["acc"][h][c]["acc"]
            res["gated"][f"{h}|{c}"] = {"acc": a, "ok": a <= C.AUDIT_MAX}
    for h in C.DIAG_HEURISTICS:
        for c, v in res["acc"][h].items():
            res["diagnostic"][f"{h}|{c}"] = {"acc": v["acc"], "above_bar": v["acc"] > C.AUDIT_MAX}
    res["diagnostic_note"] = (
        "Not gated. non_strict (other >= threshold) is a RULE VARIANT, not a surface cue: it needs the comparison and "
        "differs from the env only at other == threshold (the x0 rows), so it scores 1 - share(x0) on conditional rows. "
        "unreduced on cond rows is reported (the conditional shortcut of interest is the branch).")
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
                           "within_null": obs <= null[94], "features": list(feats)}
    res["branch_balance"] = {"ok_true": sum(y), "ok_false": len(y) - sum(y)}
    res["ok"] = all(v["ok"] for v in res["gated"].values()) and sum(y) * 2 == len(y)
    return res


# ---------------------------------------------------------------------------------------------- build / verify
def _write_jsonl(path: Path, rows: list) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


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


def describe_train(rows: list) -> dict:
    return {"n": len(rows), "by_stratum": dict(Counter(r["stratum"] for r in rows)),
            "by_category": dict(Counter(r["category"] for r in rows)),
            "branch": dict(Counter(str(r["gold_ok"]) for r in rows if r["category"] == "cond")),
            "cond_patterns_pairs": dict(Counter(r["pattern"] for r in rows if r["role"] == "core")),
            "wrap_rows": sum(C.wraps(r["before"], r["event"]) for r in rows),
            "value_range": [min(r["event"]["value"] for r in rows), max(r["event"]["value"] for r in rows)],
            "threshold_range_cond": [min(r["event"]["threshold"] for r in rows if r["category"] == "cond"),
                                     max(r["event"]["threshold"] for r in rows if r["category"] == "cond")],
            "mean_target_tokens": {a: round(sum(r["n_target_tokens"][a] for r in rows) / len(rows), 2) for a in C.ARMS}}


def build(historical: bool = True) -> int:
    outs = [C.train_path(a) for a in C.ARMS] + [C.train_source(), C.DEV, C.CHAINS, C.DEV_SOURCE]
    ex = [str(p) for p in outs if p.exists()]
    if ex:
        raise SystemExit(f"outputs exist (create-only): {ex}")
    t0 = time.time()
    lines: list = []
    log(lines, f"T2b-register data: seeds train {SEED_TRAIN} / dev {SEED_DEV} / chains {SEED_CHAINS}; no model call")
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
    aud = audit(train)
    log(lines, f"audit: gated {'PASS' if aud['ok'] else 'FAIL'} " +
        ", ".join(f"{k} {v['acc']:.3f}" for k, v in aud["gated"].items()))
    if not aud["ok"]:
        raise SystemExit("surface-heuristic audit failed; nothing written")
    tr_files = {}
    for arm in C.ARMS:
        _write_jsonl(C.train_path(arm), train_rows_for_arm(train, arm, None, count))
        tr_files[arm] = C.sha256(C.train_path(arm))
    _write_jsonl(C.DEV, [dev_row_out(r, count) for r in dev])
    _write_jsonl(C.CHAINS, parents)
    excl = {k: {"payloads": len(v["payloads"]), "operative_contents": len(v["ops"]), "sha256": v.get("sha256"),
                **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"]}
                   if "skipped_by_name_never_opened" in v else {})} for k, v in src.items()}
    common = {"schema": 1, "experiment": "RCWM-T1 T2b-register screen (roadmap item 3; joint item 5b)",
              "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_calls": 0,
              "torch_imported": "torch" in sys.modules, "builder": "scripts/t2br_make_data.py",
              "builder_sha256": C.sha256(Path(__file__)), "common_sha256": C.sha256(ROOT / "scripts" / "t2br_common.py"),
              "env_sha256": C.sha256(ROOT / "rcwmt" / "env.py"),
              "recorded_marginals_sha256": env.RECORDED_SHA, "tokenizer_json_sha256": C.sha256(C.TOKENIZER_JSON),
              "chat_template_check": f"render_prompt == real template on {n_tpl} t2_train register prompts",
              "exclusion_sources": excl, "exclusion_union": {"payloads": len(excl_pay), "operative_contents": len(excl_ops)},
              "freshness": "zero payload and zero operative-content overlap with every source; train / dev / chain units "
                           "mutually disjoint (checked during the draw by Registry, re-checked by --verify)",
              "peak_rss_mb_at_write": _rss_mb()}
    _write_json(C.train_source(), {**common, "record": "training rows (three renderings of the same rows)",
                                   "seed": SEED_TRAIN, "shuffle_seed": SHUFFLE_SEED,
                                   "files": {C.train_path(a).name: tr_files[a] for a in C.ARMS},
                                   "describe": describe_train(train), "audit": aud,
                                   "design": {"quota": TRAIN_QUOTA, "cond_offsets": COND_OFFSETS, "twin_shift": TWIN_SHIFT,
                                              "cond_pattern_quota": COND_PATTERN_QUOTA},
                                   "surface_diversity": "off (default)", "build_log": lines})
    dv_desc = {"n": len(dev), "by_stratum": dict(Counter(r["stratum"] for r in dev)),
               "branch_by_stratum": {s: dict(Counter(str(r["gold_ok"]) for r in dev if r["stratum"] == s))
                                     for s in C.DEV_STRATA if s.startswith("cond")},
               "chains": {s: sum(p["stratum"] == s for p in parents) for s in C.CHAIN_STRATA},
               "chain_candidates": sum(len(p["candidates"]) for p in parents)}
    _write_json(C.DEV_SOURCE, {**common, "record": "dev screen bank (one-step strata + k=4 chains)", "seed": SEED_DEV,
                               "chain_seed": SEED_CHAINS, "strata": {k: list(v) for k, v in C.DEV_STRATA.items()},
                               "files": {C.DEV.name: C.sha256(C.DEV), C.CHAINS.name: C.sha256(C.CHAINS)},
                               "describe": dv_desc, "go_rule": C.GO_RULE_TEXT, "build_log": lines})
    log(lines, f"written in {time.time() - t0:.0f}s; peak RSS ~{_rss_mb()} MB")
    return 0


def build_sd() -> int:
    """Surface-diversity variant of the SAME training rows (switch: --surface-diversity on)."""
    outs = [C.train_path(a, "sd") for a in C.ARMS] + [C.train_source("sd")]
    ex = [str(p) for p in outs if p.exists()]
    if ex:
        raise SystemExit(f"outputs exist (create-only): {ex}")
    base = C.load_jsonl(C.train_path("plain"))
    src = C.read_json(C.train_source()) or {}
    for a in C.ARMS:
        if C.sha256(C.train_path(a)) != (src.get("files") or {}).get(C.train_path(a).name):
            raise SystemExit(f"{C.train_path(a).name} differs from its SOURCE record")
    count = C.token_counter()
    rng = random.Random(SEED_SD)
    draws = [C.draw_sd(rng) for _ in base]
    per_arm = {a: C.load_jsonl(C.train_path(a)) for a in C.ARMS}
    files = {}
    for a in C.ARMS:
        out = []
        for i, r in enumerate(per_arm[a]):
            msgs = C.sd_messages(r["before"], r["event"], draws[i])
            prompt = C.render_prompt(msgs)
            out.append({**r, "prompt": prompt, "n_prompt_tokens": count(prompt), "sd": draws[i]})
        _write_jsonl(C.train_path(a, "sd"), out)
        files[C.train_path(a, "sd").name] = C.sha256(C.train_path(a, "sd"))
    _write_json(C.train_source("sd"), {"record": "surface-diversity re-rendering of the base training rows",
                                       "base_files": src.get("files"), "seed": SEED_SD, "files": files,
                                       "paraphrases": list(C.SD_PARAPHRASES), "fields": C.SD_FIELDS,
                                       "draw_counts": {"paraphrase": dict(Counter(d["paraphrase"] for d in draws)),
                                                       **{f: dict(Counter(d[f] for d in draws)) for f in C.SD_FIELDS}},
                                       "builder_sha256": C.sha256(Path(__file__)), "n": len(base)})
    print(f"surface-diversity variant written: {files}")
    return 0


# ---------------------------------------------------------------------------------------------- verify
def verify(historical: bool = True, write_record: bool = True, quiet: bool = False) -> tuple[bool, list]:
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)[:400]))
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)

    lines: list = []
    tsrc, dsrc = C.read_json(C.train_source()) or {}, C.read_json(C.DEV_SOURCE) or {}
    chk("train files match their SOURCE sha256", all(C.sha256(C.train_path(a)) == tsrc["files"][C.train_path(a).name]
                                                     for a in C.ARMS))
    chk("dev + chain files match their SOURCE sha256", C.sha256(C.DEV) == dsrc["files"][C.DEV.name]
        and C.sha256(C.CHAINS) == dsrc["files"][C.CHAINS.name])
    chk("render_prompt == real chat template (t2_train register prompts)", check_template() > 0)
    count = C.token_counter()
    tr = {a: C.load_jsonl(C.train_path(a)) for a in C.ARMS}
    dev = C.load_dev()
    parents = C.load_chain_bank()
    n = len(tr["plain"])
    chk("three renderings of the SAME rows (ids, before, event, order)", all(
        len(tr[a]) == n and all((x["id"], env.dumps(x["before"]), env.dumps(x["event"])) ==
                                (y["id"], env.dumps(y["before"]), env.dumps(y["event"])) for x, y in zip(tr["plain"], tr[a]))
        for a in C.ARMS), f"{n} rows")
    chk("identical prompts across arms", all(x["prompt"] == y["prompt"] == z["prompt"]
                                             for x, y, z in zip(tr["plain"], tr["trace"], tr["filler"])))
    bad_gold = bad_t = bad_tok = bad_order = 0
    for a in C.ARMS:
        for r in tr[a]:
            tg = C.targets(r["before"], r["event"], count)
            env.cross_check(FAM, r["before"], r["event"])
            bad_gold += tg["after"] != r["gold_after"] or env.dumps(env.step(FAM, r["before"], r["event"])) != r["gold_after"]
            bad_t += r["target"] != tg[a]
            bad_tok += count(r["target"]) != r["n_target_tokens"] or count(r["prompt"]) != r["n_prompt_tokens"]
            if a == "trace":
                p = C.parse_trace(r["target"])
                bad_order += p is None or not r["target"].startswith('{"cmp":')
            if a == "filler":
                bad_tok += count(r["target"]) != count(tg["trace"])
            bad_t += r["prompt"] != C.render_prompt(C.canonical_messages(r["before"], r["event"]))
    chk("training golds = env.step = FDS oracle; targets re-derived; prompts canonical", bad_gold == 0 and bad_t == 0,
        f"gold {bad_gold}, target/prompt {bad_t}")
    chk("filler token count == trace token count on every row (Titan tokenizer); recorded counts exact", bad_tok == 0,
        f"{bad_tok} mismatches")
    chk("trace key order literal (cmp, calc, after) and parseable", bad_order == 0)
    d_bad = 0
    for r in dev:
        tg = C.targets(r["before"], r["event"], count)
        env.cross_check(FAM, r["before"], r["event"])
        d_bad += tg["after"] != r["gold_after"] or any(tg[a] != r["targets"][a] for a in C.ARMS) \
            or not C.same_json(tg["cmp"], r["gold_cmp"]) or not C.same_json(tg["calc"], r["gold_calc"]) \
            or count(r["targets"]["filler"]) != count(r["targets"]["trace"]) \
            or r["messages"] != C.canonical_messages(r["before"], r["event"])
    chk("dev golds / targets / token equality re-derived", d_bad == 0, f"{d_bad} bad")
    cnt = Counter(r["stratum"] for r in dev)
    chk("dev strata sizes as designed", dict(cnt) == C.DEV_N, dict(cnt))
    # stratum properties
    sp = []
    for r in dev:
        b, ev, s = r["before"], r["event"], r["stratum"]
        v = ev["value"]
        inr = C.IN_V[0] <= v <= C.IN_V[1]
        oorv = C.OOR_V_ABS[0] <= abs(v) <= C.OOR_V_ABS[1]
        x = b[ev["other"]]
        ok = {"mix": ev["kind"] == "mix" and inr and C.wraps(b, ev) == r["wrap"],
              "wrap": ev["kind"] == "add" and inr and C.wraps(b, ev),
              "cond_xm1": ev["kind"] == "conditional" and inr and x == ev["threshold"] - 1 and v % C.MOD != 0,
              "cond_x0": ev["kind"] == "conditional" and inr and x == ev["threshold"] and v % C.MOD != 0,
              "cond_xp1": ev["kind"] == "conditional" and inr and x == ev["threshold"] + 1 and v % C.MOD != 0,
              "swap": ev["kind"] == "swap" and b[ev["register"]] != b[ev["other"]],
              "add_nowrap": ev["kind"] == "add" and inr and v != 0 and not C.wraps(b, ev),
              "mix_oor": ev["kind"] == "mix" and oorv and C.wraps(b, ev) == r["wrap"],
              "wrap_oor": ev["kind"] == "add" and oorv and C.wraps(b, ev),
              "cond_oor_val": ev["kind"] == "conditional" and oorv and x - ev["threshold"] in (0, 1),
              "cond_oor_thr": ev["kind"] == "conditional" and inr and v != 0 and (
                  C.OOR_THR[0][0] <= ev["threshold"] <= C.OOR_THR[0][1] or C.OOR_THR[1][0] <= ev["threshold"] <= C.OOR_THR[1][1])}[s]
        if not ok:
            sp.append(r["id"])
    chk("every dev row satisfies its stratum definition", not sp, sp[:5])
    clus = {}
    for r in dev:
        clus.setdefault(r["cluster"], []).append(r)
    pair_ok = all(len({env.dumps(r["before"]) for r in rs}) == 1 and len({env.dumps({k: v for k, v in r["event"].items()
                                                                                     if k != "threshold"}) for r in rs}) == 1
                  for rs in clus.values() if len(rs) > 1)
    bal = {s: Counter(r["gold_ok"] for r in dev if r["stratum"] in grp) for s, grp in
           (("cond_oor_val", ("cond_oor_val",)), ("cond_oor_thr", ("cond_oor_thr",)))}
    chk("pairs / triplets differ ONLY in the threshold; OOR conditional strata branch-balanced", pair_ok and all(
        v[True] == v[False] for v in bal.values()), {k: dict(v) for k, v in bal.items()})
    # training design
    trc = tr["plain"]
    tcl = {}
    for r in trc:
        tcl.setdefault(r["cluster"], []).append(r)
    twin_ok = all(len(rs) == 2 and rs[0]["gold_ok"] != rs[1]["gold_ok"] and env.dumps(rs[0]["before"]) == env.dumps(rs[1]["before"])
                  and {k: v for k, v in rs[0]["event"].items() if k != "threshold"} ==
                  {k: v for k, v in rs[1]["event"].items() if k != "threshold"}
                  for c, rs in tcl.items() if c.startswith("tp"))
    chk("training conditional twins: same row except the threshold, branch flipped", twin_ok,
        f"{sum(1 for c in tcl if c.startswith('tp'))} pairs")
    chk("training value / threshold ranges in the recorded marginal", all(C.IN_V[0] <= r["event"]["value"] <= C.IN_V[1]
                                                                          and C.THR_IN[0] <= r["event"]["threshold"] <= C.THR_IN[1]
                                                                          for r in trc))
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
    # freshness
    src, excl_pay, excl_ops = load_union(lines, historical)
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
        ("" if historical else " (historical harvest SKIPPED: --no-historical)"), not probs, dict(probs))
    if historical:
        skipped = src["historical"]["skipped_by_name_never_opened"]
        chk("historical harvest skipped exactly the HSKILL_CONFIRM_V2* files, by name (never opened)",
            skipped and all(Path(p).name.startswith("HSKILL_CONFIRM_V2") for p in skipped), skipped)
    aud = audit(trc)
    chk("surface-heuristic audit: copy / unreduced / always-add / always-subtract <= 0.55; branch 150 / 150", aud["ok"],
        ", ".join(f"{k} {v['acc']:.3f}" for k, v in aud["gated"].items()))
    chk("audit recorded at build == recomputed", json.dumps(aud["gated"], sort_keys=True) ==
        json.dumps(tsrc["audit"]["gated"], sort_keys=True))
    ok = all(c[1] for c in checks)
    if write_record:
        frozen = (ROOT / "docs" / "t2b" / "register" / "SCREEN_FREEZE.json").exists()
        if frozen:
            print("SCREEN_FREEZE.json exists: AUDIT.json is a frozen input and is not rewritten")
        else:
            C.DOCS.mkdir(parents=True, exist_ok=True)
            rec = {"ok": ok, "historical_harvest": historical, "checks": [{"check": a, "ok": b, "detail": c} for a, b, c in checks],
                   "audit": aud, "train_describe": describe_train(
                       [{**r, "n_target_tokens": {a: tr[a][i]["n_target_tokens"] for a in C.ARMS}} for i, r in enumerate(trc)]),
                   "dev_describe": {"by_stratum": dict(cnt), "chains": dict(Counter(p["stratum"] for p in parents))},
                   "files_sha256": {**{C.train_path(a).name: C.sha256(C.train_path(a)) for a in C.ARMS},
                                    C.DEV.name: C.sha256(C.DEV), C.CHAINS.name: C.sha256(C.CHAINS),
                                    C.train_source().name: C.sha256(C.train_source()), C.DEV_SOURCE.name: C.sha256(C.DEV_SOURCE)},
                   "code_sha256": {r: C.sha256(ROOT / r) for r in ("scripts/t2br_make_data.py", "scripts/t2br_common.py")},
                   "torch_imported": "torch" in sys.modules, "peak_rss_mb": _rss_mb(),
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (C.DOCS / "AUDIT.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {C.DOCS / 'AUDIT.json'}")
    print(f"verify: {sum(c[1] for c in checks)}/{len(checks)} checks pass")
    return ok, checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-historical", action="store_true", help="--verify only: skip the records-tree harvest (quick)")
    ap.add_argument("--surface-diversity", choices=("off", "on"), default="off")
    a = ap.parse_args()
    if a.verify:
        ok, _ = verify(historical=not a.no_historical)
        return 0 if ok else 1
    if a.surface_diversity == "on":
        if not C.train_path("plain").exists():
            build()
        return build_sd()
    return build()


if __name__ == "__main__":
    sys.exit(main())
