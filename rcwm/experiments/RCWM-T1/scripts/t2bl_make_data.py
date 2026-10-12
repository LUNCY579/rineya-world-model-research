"""T2b-ledger data: the surface-diversified oracle-labelled training set and the fresh dev bank. Built WITHOUT any model
call, model output or torch (the `tokenizers` library reads tokenizer.json for token statistics and the token audit only).

  python -B scripts/t2bl_make_data.py                 # writes (create-only) data/t2bl_dev.jsonl + .SOURCE.json and
                                                      # data/t2bl_train.jsonl + .SOURCE.json, then docs/t2b/ledger/AUDIT.json
  python -B scripts/t2bl_make_data.py --out-dir DIR   # rehearsal: the same build into DIR (nothing under data/)
  python -B scripts/t2bl_make_data.py --verify        # re-derives both files from (seed, recorded rejections) byte for
                                                      # byte, rebuilds the freshness union, re-runs the audit; writes nothing
  python -B scripts/t2bl_make_data.py --no-harvest    # tests only: an empty union (the SOURCE says so; never registered)

TRAINING SET (data/t2bl_train.jsonl; kit revision 3, seed 20261165; 1,500 ledger one-step rows on the T2 recipe's frame
family; target env.dumps(after)):
  CANON block   300 rows in the exact canonical T2 surface (paraphrase 0, sorted keys, accounts A-D, |b| <= 98,
                trained amounts, statuses posted / void): 100 posted/void twin pairs + 100 posted singles. It anchors
                the in-distribution behaviour the dev gate's non-inferiority clause tests.
  DIV block     1,200 rows: 300 twin pairs (posted + a non-posted member, same base and surface) + 250 posted and 350
                non-posted singles. Each DIV slot has a SURFACE MODE, by exact counts within each slot type (kit
                revision 3; SURFACE_MODE_COUNTS): 'canonical' (the exact canonical T2 surface with the slot's own
                status, so the training words also occur on the canonical frame), 'stacked' (every surface factor at a
                non-canonical TRAINING level at once: paraphrase 1-3, training letters or training words as names,
                non-sorted training orders of the event and before objects, the reversed top-level order) and 'free'
                (every factor drawn independently from its TRAINING partition: paraphrases 0-3, event / before orders
                canonical w.p. 0.25, account names A-D-based 0.35, training letters 0.30, training words 0.35; the
                top-level order reversed on an exact half of the free slots of each slot type; conditioned on NOT
                being the full stack, which is the stacked mode's). Counts canonical / stacked / free: pairs 45 / 150
                / 105, posted singles 0 / 150 / 100, non-posted singles 100 / 150 / 100 (DIV rows 190 / 600 / 410).
                Stacked and free slots draw 4-7 accounts (4 w.p. 0.55), balances and amounts by digit class (balances
                t2 / 3-digit / 4-digit 0.40 / 0.30 / 0.30, scale rows t2 / 3-digit 0.50 / 0.50; add/transfer amounts
                t2 / 2-digit / 3-digit 0.45 / 0.30 / 0.25; scale factors always the recorded marginal), every input
                and posted-gold value at most 4 digits (redrawn otherwise). Non-posted statuses: void 50 and 600
                TRAINING words (not-applied 180,
                neutral 150, nonce 120, case / spelling variants of void 60, posted look-alikes 90; 15 rows per word, 10
                per void variant). Operator decision (kit revision 3, 2026-10-12): the counts are status-symmetric
                wherever the surface is revised, so posted and non-posted rows are EXACTLY equal on the stacked rows
                (300 / 300), on the reversed-top rows (403 / 403) and, with the CANON block, on the canonical surface
                (245 / 245): no surface-to-status cue on those surfaces (audited; binding band [0.49, 0.51]).
  Totals: posted 750 / non-posted 750; event kinds by PREREG-3's quota (transfer 0.45, scale 0.33, add 0.22); every
  posted event changes the state (so copy and apply-regardless score exactly 0.50).
  Kit revision 2 (spent; seed 20261161) differed only in the surfaces: the top-level order was never varied and the
  canonical / free choice was a 0.15 coin per slot, with no stacked mode. The first revision-3 build (generator
  t2bl-data-v3, same seed; superseded before any training or freeze, docs/t2b/ledger/superseded_rev3_preop/) assigned
  the modes by one quota 0.15 / 0.50 / 0.35 for every slot type and drew the free slots' top order by a 0.5 coin, so
  its stacked and reversed-top rows were 0.457 posted (a residual surface-to-status cue its records failed to call one).
DEV BANK (data/t2bl_dev.jsonl; kit revision 3, seed 20261166, the builder and strata of revision 2 unchanged; the DEV
  half of every split held-out partition): 100 bases (env.gen_start
  balances, |b| <= 98, accounts A-D; an env.gen_event-style event with a trained amount whose posted version changes the
  state; kind quota 45/33/22), every stratum the same base with ONE factor changed (or, for the combined surfaces, one
  surface), t2bl_common.DEV_STRATA: 19 strata, 1,801 rows (S_bal, S_amt, H_amt only on the 67 add/transfer bases).
FRESHNESS (binding): no (before, event) payload and no FDS operative content of any training or dev row equals one of
  the union (lbeam_problems.load_exclusions: the name-skipping historical harvest, every pinned source, PREREG-3's bank;
  every other data/**/*.json[l] payload and rolled-out multi-event transition, except this kit's own top-level
  data/t2bl_* files; since kit revision 3 the superseded and spent kit files in data/t2bl_superseded_rev*/ ARE in the
  union, so the fresh dev bank shares nothing with the spent revision-2 dev bank or training set; the binding
  transitions of the spent PREREG-LB dev sets AND of the spent PREREG-LB confirmation draw; every {before, event}
  payload under docs/ except docs/t2b/ledger/). HSKILL_CONFIRM_V2* files are skipped by NAME, never opened or hashed;
  rcwmt.bank.load_exclusions() is NOT called (I-28). Dev rows also exclude each other across bases; training rows
  exclude the dev bank.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # nothing here touches the driver; empty string, never "-1" (I-21)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bl_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

GEN_VERSION = "t2bl-data-v3.1"                   # t2bl-data-v3: the first revision-3 build (superseded, never used)
BUILD_CODE = ("scripts/t2bl_common.py", "scripts/t2bl_make_data.py", "scripts/t2bl_audit.py", "scripts/p3_common.py",
              "rcwmt/env.py")
MAX_ATTEMPTS = 200
OWN_PREFIX = "t2bl_"
SPENT_LB_SETS = ("docs/ledger_beam/dev_set.jsonl", "docs/ledger_beam/dev2_primary.jsonl",
                 "docs/ledger_beam/dev2_oor.jsonl", "docs/ledger_beam/LB_CONFIRM_DRAW.jsonl")
DOCS_SKIP = ("docs/t2b/ledger/",)
MAX_SCAN_BYTES = 60_000_000

# training composition (see the module docstring)
CANON_PAIRS, CANON_SINGLES = 100, 100
DIV_PAIRS, DIV_POSTED_SINGLES, DIV_NONPOSTED_SINGLES = 300, 250, 350
DIV_VOID = 50
DIV_WORD_QUOTA = OrderedDict([("not_applied", 180), ("neutral", 150), ("nonce", 120), ("void_variant", 60),
                              ("posted_like", 90)])
N_TRAIN = 2 * CANON_PAIRS + CANON_SINGLES + 2 * DIV_PAIRS + DIV_POSTED_SINGLES + DIV_NONPOSTED_SINGLES
P_COUNT = ((4, 0.55), (5, 0.15), (6, 0.15), (7, 0.15))
P_NAMES = (("canonical", 0.35), ("train_letters", 0.30), ("train_words", 0.35))
P_BAL_LEVEL = (("t2", 0.40), ("b3", 0.30), ("b4", 0.30))
P_BAL_LEVEL_SCALE = (("t2", 0.50), ("b3", 0.50))  # scale rows: at most 3-digit balances (no 4-digit x factor products)
BAL_MIX = {"t2": (("t2", 1.0),), "b3": (("t2", 0.4), ("b3", 0.6)), "b4": (("t2", 0.3), ("b3", 0.3), ("b4", 0.4))}
P_AMT_LEVEL = (("t2", 0.45), ("a2", 0.30), ("a3", 0.25))
# A share of DIV slots is rendered in the exact canonical T2 surface (paraphrase 0, sorted keys, accounts A-D, |b| <= 98,
# trained amounts) with the slot's own status mix, so that non-void training words also occur on the canonical surface:
# otherwise the canonical frame would only ever pair with posted / void and a surface-conditioned 'void' cue would fit
# the training set (canonical frame -> void trigger; any other surface -> read the status).
# Kit revision 3 (R29): surface modes by EXACT COUNTS within each DIV slot type. 'stacked' renders every surface factor
# at a non-canonical training level at once (the training analogue of the combined held-out surface, on which revision
# 2's posted twins degraded to copying). Operator decision (2026-10-12): the counts are status-symmetric wherever the
# surface is revised. Posted and non-posted singles get the same number of stacked slots (150 each) and of free slots
# (100 each; the free slots' top order is reversed on an exact half of each type, FREE_TOP_QUOTA), pair members share
# their surface, and the 100 extra non-posted singles (the DIV block is 550 posted / 650 non-posted, offsetting the
# CANON block's 200 / 100) sit on the canonical surface. So posted = non-posted on the stacked rows (300 / 300), on the
# reversed-top rows (403 / 403), on the free rows (205 / 205) and on the canonical surface (245 / 245 with the CANON
# block). A 'free' slot is never the full stack (redrawn otherwise), so the stacked surface is exactly the stacked mode.
# The first revision-3 build used one quota 0.15 / 0.50 / 0.35 for every slot type and a 0.5 coin for the free slots'
# top order (0.457 posted on the stacked and reversed-top rows); the pair counts are unchanged from it.
SURFACE_MODE_COUNTS = OrderedDict([
    ("pair", OrderedDict([("canonical", 45), ("stacked", 150), ("free", 105)])),
    ("posted_single", OrderedDict([("canonical", 0), ("stacked", 150), ("free", 100)])),
    ("nonposted_single", OrderedDict([("canonical", 100), ("stacked", 150), ("free", 100)])),
])
FREE_TOP_QUOTA = OrderedDict([("reversed", C.P_TOP_REVERSED), ("sorted", 1 - C.P_TOP_REVERSED)])
TOP_OF = {"reversed": (1, 0), "sorted": (0, 1)}
P_NAMES_STACKED = (("train_letters", 0.30 / 0.65), ("train_words", 0.35 / 0.65))


def _pick(rng: random.Random, table) -> object:
    u, acc = rng.random(), 0.0
    for v, p in table:
        acc += p
        if u < acc:
            return v
    return table[-1][0]


def _log(lines: list, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    lines.append(line)
    print(line, flush=True)


# ============================================================================================== the freshness union
def _builders():
    import lbeam_problems as P                    # torch-free; its _import_builders restores the environment
    MB = P._import_builders()
    return P, MB, MB.L1


def _harvest_add(excl: dict, rel: str, p: Path, MB, L1, P) -> None:
    try:
        sha = L1.sha256_file(p)
        e_pay = MB.harvest_file(p)
    except (OSError, ValueError) as e:
        excl["sources"][rel] = {"error": f"{type(e).__name__}: {e}"}
        return
    e_ops = L1.op_keys_of_payloads(e_pay)
    m_pay, m_ops, n_obj = P._multi_event_ledger_transitions(p, MB, L1)
    excl["sources"][rel] = {"payloads": len(e_pay), "operative": len(e_ops), "sha256": sha,
                            "multi_event_objects": n_obj, "multi_event_transitions": len(m_pay)}
    excl["pay"] |= e_pay | m_pay
    excl["ops"] |= e_ops | m_ops


def build_union(root: Path = ROOT, historical: bool = True, skip_prefix: str = OWN_PREFIX, log=print,
                extra_skip: tuple = ()) -> dict:
    """The freshness union (module docstring). `skip_prefix`: data files of this kit (never their own exclusion);
    PREREG-5's freeze passes skip_prefix=None so the T2b-ledger training and dev files ARE excluded from its bank."""
    P, MB, L1 = _builders()
    t0 = time.time()
    excl = P.load_exclusions(historical=historical, extra_data=False, log=log)
    root = Path(root)
    d = root / "data"
    for p in sorted(d.rglob("*")) if d.is_dir() else []:
        if not p.is_file() or p.suffix not in (".json", ".jsonl") or p.name.startswith(C.RESERVED_PREFIX):
            continue
        if p.name.endswith(".SOURCE.json") or p.name in P.NON_PAYLOAD_DATA or (p.parent == d and p.name in P.COVERED_DATA):
            continue
        if skip_prefix and p.parent == d and p.name.startswith(skip_prefix):
            continue                                  # this kit's own current files only (superseded kit files count)
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel in extra_skip:
            continue
        _harvest_add(excl, rel, p, MB, L1, P)
    for rel in SPENT_LB_SETS:
        p = root / rel
        if not p.is_file():
            excl["sources"][rel] = {"error": "missing"}
            continue
        pay, ops = set(), set()
        for inst in P.load_set(p):
            for b, e in P.binding_transitions(inst):
                pay.add(L1.payload(b, e))
                ops.add(L1.opkey(C.LEDGER, b, e))
        excl["sources"][rel] = {"payloads": len(pay), "operative": len(ops), "sha256": L1.sha256_file(p),
                                "kind": "binding transitions of a spent PREREG-LB instance set"}
        excl["pay"] |= pay
        excl["ops"] |= ops
    docs = root / "docs"
    for p in sorted(docs.rglob("*")) if docs.is_dir() else []:
        if not p.is_file() or p.suffix not in (".json", ".jsonl") or p.name.startswith(C.RESERVED_PREFIX):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel in SPENT_LB_SETS or any(rel.startswith(s) for s in DOCS_SKIP) or rel in extra_skip:
            continue
        try:
            if p.stat().st_size > MAX_SCAN_BYTES:
                excl["sources"][rel] = {"skipped": "larger than 60 MB"}
                continue
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            excl["sources"][rel] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if '"before' not in raw and '\\"before' not in raw and '"s0"' not in raw:
            continue
        del raw
        _harvest_add(excl, rel, p, MB, L1, P)
    excl["rule"] = ("lbeam_problems.load_exclusions(extra_data=False) (l1_make_ood.harvest_historical skips "
                    "HSKILL_CONFIRM_V2* by name, never opened; rcwmt.bank.load_exclusions() NOT called; every pinned "
                    "source; data/bank_p3_oor.jsonl) + every payload and rolled-out multi-event transition of every other "
                    "data/**/*.json[l] file" + (f" except the top-level data/{skip_prefix}* files (data/t2bl_superseded_rev*/ "
                                                "included)" if skip_prefix else "") +
                    " + the binding transitions of " + ", ".join(SPENT_LB_SETS) +
                    " + every {before, event} payload under docs/ except " + ", ".join(DOCS_SKIP))
    excl["skip_prefix"] = skip_prefix
    log(f"[t2bl] freshness union: {len(excl['pay'])} payloads, {len(excl['ops'])} operative contents from "
        f"{len(excl['sources'])} sources ({time.time() - t0:.0f}s)")
    return excl


def empty_union() -> dict:
    return {"pay": set(), "ops": set(), "sources": {}, "rule": "EMPTY (tests only; --no-harvest)", "historical": False}


def union_errors(excl: dict) -> list:
    return sorted(k for k, v in (excl.get("sources") or {}).items() if isinstance(v, dict) and "error" in v)


def opkey(before: dict, ev: dict) -> str:
    P, MB, L1 = _builders()
    return L1.opkey(C.LEDGER, before, ev)


class Fresh:
    """Union + within-set freshness. owner: a base (bank) or a slot (training)."""

    def __init__(self, excl: dict, extra_pay=(), extra_ops=()):
        self.excl_pay = excl["pay"]
        self.excl_ops = excl["ops"]
        self.extra_pay, self.extra_ops = set(extra_pay), set(extra_ops)
        self.pay: dict = {}
        self.ops: dict = {}
        _, _, self.L1 = _builders()

    def keys(self, before, ev) -> tuple:
        return self.L1.payload(before, ev), self.L1.opkey(C.LEDGER, before, ev)

    def problem(self, items: list, owner) -> str | None:
        for before, ev in items:
            p, o = self.keys(before, ev)
            if p in self.excl_pay or p in self.extra_pay:
                return "payload in an exclusion source"
            if o in self.excl_ops or o in self.extra_ops:
                return "operative content in an exclusion source"
            if self.pay.get(p, owner) != owner:
                return "payload shared with another unit"
            if self.ops.get(o, owner) != owner:
                return "operative content shared with another unit"
        return None

    def add(self, items: list, owner) -> None:
        for before, ev in items:
            p, o = self.keys(before, ev)
            self.pay.setdefault(p, owner)
            self.ops.setdefault(o, owner)


# ============================================================================================== bases and strata
def _schedule(rng: random.Random, n: int, quota: dict) -> list:
    """n items with the quota's proportions (largest remainder), shuffled."""
    raw = {k: n * v for k, v in quota.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - cnt[k]))[: n - sum(cnt.values())]:
        cnt[k] += 1
    out = [k for k, c in cnt.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def _schedule_counts(rng: random.Random, counts: dict) -> list:
    """Exactly counts[k] items k (in the dict's order), shuffled (the same shuffle as _schedule for the same length)."""
    out = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def draw_event(rng: random.Random, names: list, kind: str, amount: int, tick: int | None = None) -> dict:
    a = rng.choice(names)
    return {"account": a, "amount": amount, "kind": kind, "other": rng.choice([x for x in names if x != a]),
            "status": "posted", "tick": rng.randint(*C.TICKS) if tick is None else tick}


def draw_base(rng: random.Random, kind: str, m, pool: list) -> tuple:
    """(before A-D in the T2 range, posted event with a trained amount) whose posted version changes the state."""
    for _ in range(1000):
        before = {a: rng.choice(pool) for a in C.CANON_NAMES}
        ev = draw_event(rng, list(C.CANON_NAMES), kind, rng.choice(m.ledger_amounts))
        if C.effective(before, ev):
            return before, ev
    raise RuntimeError("could not draw an effective base")


def _with_status(ev: dict, status: str) -> dict:
    return {**ev, "status": status}


def derive(stratum: str, idx: int, before: dict, ev: dict, rng_of, pool: list, split: str = "dev") -> tuple | None:
    """(before, event, surface, meta) of a stratum on base idx (None: the stratum has no row on this base). rng_of(key)
    gives the base's seeded sub-RNG for a key; strata that share a surface use the same key. split: which half of the
    held-out partitions the bank renders (t2bl_common.heldout)."""
    H = C.heldout(split)
    canon = C.canonical_surface(before)
    ms, mk = C.mixed_status(idx)
    mmeta = {"status": ms, "status_kind": mk}
    if stratum == "ID_posted":
        return before, _with_status(ev, "posted"), canon, {}
    if stratum == "ID_void":
        return before, _with_status(ev, "void"), canon, {}
    if stratum == "S_words":
        kind, w = C.word_for(C.TRAIN_WORDS, idx)
        return before, _with_status(ev, w), canon, {"word_kind": kind, "word": w}
    if stratum in ("S_surface_posted", "S_surface_words"):
        rng = rng_of("S_surface")
        n = 5 + (idx // 2) % 3
        nk = "train_letters" if idx % 2 == 0 else "train_words"
        names = sorted(rng.sample(C.TRAIN_LETTERS if nk == "train_letters" else C.TRAIN_NAME_WORDS, n))
        b2, e2 = C.remap(before, ev, names[:4])               # the base's own in-range values and trained amount
        for k in names[4:]:
            b2[k] = rng.choice(pool)
        para = 1 + idx % 3
        eperm = C.draw_train_perm(rng, "event", allow_canon=False)
        bperm = C.draw_train_perm(rng, "before", len(names), allow_canon=False)
        surf = C.surface_with(b2, para, (0, 1), eperm, bperm)
        meta = {"names_kind": nk, "n_accounts": n, "para": para}
        if stratum == "S_surface_posted":
            return b2, _with_status(e2, "posted"), surf, meta
        kind, w = C.word_for(C.TRAIN_WORDS, idx)
        return b2, _with_status(e2, w), surf, {**meta, "word_kind": kind, "word": w}
    if stratum == "S_bal":
        if ev["kind"] == "scale":
            return None
        rng = rng_of("S_bal")
        e2 = _with_status(ev, "posted")
        for _ in range(500):
            b2 = {k: C.bal_value(rng, rng.choice(("b3", "b4")), pool) for k in before}
            if C.within_train_range(b2, e2) and max(abs(v) for v in b2.values()) >= 100:
                return b2, e2, C.canonical_surface(b2), {"band": "trained 3-4-digit balances"}
        raise RuntimeError("S_bal: no in-band draw")
    if stratum == "S_amt":
        if ev["kind"] == "scale":
            return None
        rng = rng_of("S_amt")
        return before, {**_with_status(ev, "posted"), "amount": C.amt_value(rng, rng.choice(("a2", "a3")), [])}, canon,             {"band": "trained 2-3-digit amounts"}
    if stratum == "H_words":
        kind, w = C.word_for(H["words"], idx)
        return before, _with_status(ev, w), canon, {"word_kind": kind, "word": w}
    if stratum in ("H_order", "H_order_inner"):
        j1, j2 = idx % 3, (idx // 3) % 3
        top = (1, 0) if stratum == "H_order" else (0, 1)
        surf = C.surface_with(before, 0, top, H["event"][j1], H["before4"][j2])
        return before, _with_status(ev, ms), surf, {"event_order": j1, "before_order": j2, **mmeta}
    if stratum == "H_order_top":
        surf = C.surface_with(before, 0, (1, 0), tuple(range(len(C.EVENT_KEYS))), tuple(range(len(before))))
        return before, _with_status(ev, ms), surf, dict(mmeta)
    if stratum in ("H_names", "HS_words", "HS_posted"):
        rng = rng_of("H_names" if stratum == "H_names" else "HS")
        letters = ((idx // 2) % 2 == 0) if stratum == "H_names" else (idx % 2 == 0)   # decoupled from the status parity
        if letters:
            names, nk = list(H["letters"]), "heldout_letters"
        else:
            names, nk = sorted(rng.sample(H["name_words"], 4)), "heldout_words"
        b2, e2 = C.remap(before, ev, names)
        if stratum == "H_names":
            return b2, _with_status(e2, ms), C.canonical_surface(b2), {"names_kind": nk, "names": names, **mmeta}
        j1, j2 = idx % 3, (idx // 3) % 3
        surf = C.surface_with(b2, H["para"], (1, 0), H["event"][j1], H["before4"][j2])
        meta = {"names_kind": nk, "names": names, "event_order": j1, "before_order": j2}
        if stratum == "HS_posted":
            return b2, _with_status(e2, "posted"), surf, meta
        kind, w = C.word_for(H["words"], idx)
        return b2, _with_status(e2, w), surf, {**meta, "word_kind": kind, "word": w}
    if stratum == "H_count":
        rng = rng_of("H_count")
        extra = sorted(rng.sample(C.TRAIN_LETTERS, 4))
        b2 = dict(before)
        for k in extra:
            b2[k] = rng.choice(pool)
        return b2, _with_status(ev, ms), C.canonical_surface(b2), {"n_accounts": 8, "extra": extra, **mmeta}
    if stratum == "H_para":
        return before, _with_status(ev, ms), {**canon, "para": H["para"]}, dict(mmeta)
    if stratum == "H_bal":
        rng = rng_of("H_bal")
        b2 = {k: C.bal_value(rng, "b5", pool) for k in before}
        return b2, _with_status(ev, "posted"), C.canonical_surface(b2), {}
    if stratum == "H_amt":
        if ev["kind"] == "scale":
            return None
        rng = rng_of("H_amt")
        return before, {**_with_status(ev, "posted"), "amount": C.amt_value(rng, "a4", [])}, canon, {}
    if stratum in ("LB_void", "LB_posted"):
        b2, e2 = C.remap(before, ev, list(C.LB_NAMES))
        surf = C.surface_with(b2, 0, C.LB_ORDER["top"], C.LB_ORDER["event"], tuple(range(3, -1, -1)))
        return b2, _with_status(e2, C.LB_STATUS if stratum == "LB_void" else "posted"), surf, {}
    if stratum == "P3_words":
        w = C.P3_WORDS[idx % len(C.P3_WORDS)]
        return before, _with_status(ev, w), canon, {"word": w}
    if stratum in C.P3.STRATA:
        w = C.p3_status(stratum, idx)
        return before, _with_status(ev, w), canon, {"word": w}
    raise ValueError(stratum)


def bank_row(prefix: str, stratum: str, idx: int, attempt: int, before, ev, surf, meta) -> dict:
    gold = C.gold_of(before, ev)
    rule = "apply" if ev["status"] == "posted" else "noop"
    want = C.STRATA[stratum][1]
    if want != "mixed" and want != rule:
        raise AssertionError(f"{stratum}: gold rule {rule} != {want}")
    return {"id": f"{prefix}-{stratum}-{idx:03d}", "kind": "onestep", "stratum": stratum, "factor": C.STRATA[stratum][0],
            "base": idx, "attempt": attempt, "before": before, "event": ev, "surface": surf, "gold": env.dumps(gold),
            "gold_rule": rule, "posted_gold": env.dumps(C.gold_of(before, _with_status(ev, "posted"))),
            "twin": f"{prefix}-{C.twin_stratum(stratum, idx)}-{idx:03d}", "meta": meta}


def build_bank(seed: int, strata: tuple, excl: dict, prefix: str, tag: str, extra_pay=(), extra_ops=(),
               replay=None, n_bases: int = C.N_BASES, split: str = "dev") -> tuple:
    """(rows in stratum-major order, record). Deterministic given (tag, seed, strata, split, union): base idx draws use
    random.Random(f'{tag}|{seed}|{idx}|{attempt}|<key>') (str seeds are SHA-512 hashed: PYTHONHASHSEED-independent).
    replay: the recorded freshness rejections [(idx, attempt)] reproduce the set without the union. split: the half of
    the held-out partitions the bank renders ('dev': the dev bank; 'confirm': PREREG-5's map bank)."""
    m = env.load_marginals()
    pool = C.small_pool(m)
    kinds = _schedule(random.Random(f"{tag}|{seed}|kinds"), n_bases, C.KIND_QUOTA)
    fresh = Fresh(excl, extra_pay, extra_ops)
    replay = {tuple(r) for r in (replay or [])}
    rows_by: dict = {}
    rejections, counts = [], Counter()
    for idx in range(n_bases):
        for attempt in range(MAX_ATTEMPTS):
            rng_of = lambda key, i=idx, a=attempt: random.Random(f"{tag}|{seed}|{i}|{a}|{key}")  # noqa: E731
            before, ev = draw_base(rng_of("base"), kinds[idx], m, pool)
            rows = []
            ok = True
            for s in strata:
                d = derive(s, idx, before, ev, rng_of, pool, split)
                if d is None:
                    continue
                b2, e2, surf, meta = d
                if C.STRATA[s][1] == "apply" and not C.effective(b2, e2):
                    ok = False
                    counts["not_effective"] += 1
                    break
                rows.append(bank_row(prefix, s, idx, attempt, b2, e2, surf, meta))
            if not ok:
                continue
            items = [(r["before"], r["event"]) for r in rows]
            prob = "replayed rejection" if (idx, attempt) in replay else fresh.problem(items, idx)
            if prob:
                rejections.append([idx, attempt])
                counts[prob] += 1
                continue
            fresh.add(items, idx)
            rows_by[idx] = rows
            break
        else:
            raise RuntimeError(f"base {idx}: no fresh draw in {MAX_ATTEMPTS} attempts")
    ordered = [r for s in strata for idx in range(n_bases) for r in rows_by[idx] if r["stratum"] == s]
    rec = {"tag": tag, "seed": seed, "n_bases": n_bases, "split": split, "strata": list(strata), "kind_schedule": kinds,
           "fresh_rejections": rejections, "rejection_counts": dict(counts), "n_rows": len(ordered)}
    return ordered, rec


# ============================================================================================== training rows
def train_plan(seed: int) -> list:
    """The slot list: block, slot, kind, members (statuses), surface mode (DIV: canonical / stacked / free by exact counts
    within each slot type, SURFACE_MODE_COUNTS; CANON: canon_block) and, on free DIV slots, the top-level order (an
    exact half reversed per slot type, FREE_TOP_QUOTA, drawn after everything else from the same plan stream).
    Deterministic from the seed."""
    rng = random.Random(f"t2bl-train-plan|{seed}")
    slots = []
    ck = _schedule(rng, CANON_PAIRS + CANON_SINGLES, C.KIND_QUOTA)
    for j in range(CANON_PAIRS + CANON_SINGLES):
        slots.append({"block": "canon", "slot": j, "kind": ck[j], "members": ["posted", "void"] if j < CANON_PAIRS else ["posted"],
                      "surface_mode": "canon_block"})
    n_div = DIV_PAIRS + DIV_POSTED_SINGLES + DIV_NONPOSTED_SINGLES
    dk = _schedule(rng, n_div, C.KIND_QUOTA)
    words = []
    for kind, q in DIV_WORD_QUOTA.items():
        ws = C.TRAIN_WORDS[kind]
        assert q % len(ws) == 0, (kind, q)
        words += [w for w in ws for _ in range(q // len(ws))]
    nonposted = ["void"] * DIV_VOID + words
    assert len(nonposted) == DIV_PAIRS + DIV_NONPOSTED_SINGLES
    rng.shuffle(nonposted)
    it = iter(nonposted)
    single_kinds = ["posted"] * DIV_POSTED_SINGLES + ["nonposted"] * DIV_NONPOSTED_SINGLES
    rng.shuffle(single_kinds)
    for j in range(n_div):
        if j < DIV_PAIRS:
            mem = ["posted", next(it)]
        else:
            mem = ["posted"] if single_kinds[j - DIV_PAIRS] == "posted" else [next(it)]
        slots.append({"block": "div", "slot": j, "kind": dk[j], "members": mem})
    div = [s for s in slots if s["block"] == "div"]
    types = OrderedDict([("pair", [s for s in div if len(s["members"]) == 2]),
                         ("posted_single", [s for s in div if s["members"] == ["posted"]]),
                         ("nonposted_single", [s for s in div if len(s["members"]) == 1 and s["members"] != ["posted"]])])
    for typ, ss in types.items():                     # exact counts per slot type (status-symmetric where revised)
        counts = SURFACE_MODE_COUNTS[typ]
        if sum(counts.values()) != len(ss):
            raise RuntimeError(f"surface mode counts of {typ}: {sum(counts.values())} != {len(ss)} slots")
        for s, mode in zip(ss, _schedule_counts(rng, counts)):
            s["surface_mode"] = mode
    for typ, ss in types.items():                     # free slots: the top order, an exact half reversed per type
        free = [s for s in ss if s["surface_mode"] == "free"]
        for s, top in zip(free, _schedule(rng, len(free), FREE_TOP_QUOTA)):
            s["free_top"] = top
    return slots


def draw_div(rng: random.Random, kind: str, m, pool: list, mode: str = "free", top: str | None = None) -> tuple:
    """(before, posted event, surface, meta) of a DIV slot in its surface mode: 'canonical' = the canonical T2 surface
    (the status still comes from the slot); 'stacked' = every surface factor at a non-canonical training level at once
    (paraphrase 1-3, training letters / words, non-sorted training event and before orders, the reversed top order);
    'free' = every factor drawn independently from its training partition, the top order given by the plan ('reversed'
    / 'sorted'), and the paraphrase and the event / before orders redrawn while the surface would be the full stack (the
    stacked mode's combination), so the stacked training surface is exactly the stacked mode."""
    if mode == "canonical":
        before, ev = draw_base(rng, kind, m, pool)
        return before, ev, C.canonical_surface(before), {"names_kind": "canonical", "n_accounts": 4, "bal_level": "t2",
                                                         "amt_level": "t2", "para": 0, "canonical_surface": True,
                                                         "surface_mode": "canonical"}
    if mode not in ("stacked", "free"):
        raise ValueError(mode)
    n = _pick(rng, P_COUNT)
    nk = _pick(rng, P_NAMES_STACKED if mode == "stacked" else P_NAMES)
    if nk == "canonical":
        names = sorted(list(C.CANON_NAMES) + rng.sample(C.TRAIN_LETTERS, n - 4))
    elif nk == "train_letters":
        names = sorted(rng.sample(C.TRAIN_LETTERS, n))
    else:
        names = sorted(rng.sample(C.TRAIN_NAME_WORDS, n))
    level = _pick(rng, P_BAL_LEVEL_SCALE if kind == "scale" else P_BAL_LEVEL)
    alevel = "t2" if kind == "scale" else _pick(rng, P_AMT_LEVEL)
    ev = None
    for _ in range(50):                               # balances at the level, then an effective in-range event
        for _ in range(200):
            before = {k: C.bal_value(rng, _pick(rng, BAL_MIX[level]), pool) for k in names}
            mb = max(abs(v) for v in before.values())
            got = "t2" if mb <= 98 else "b3" if mb <= 999 else "b4"
            if got == level:
                break
        else:
            raise RuntimeError("balance level not reached")
        for _ in range(200):
            amount = C.amt_value(rng, alevel, m.ledger_amounts) if kind != "scale" else rng.choice(m.ledger_amounts)
            cand = draw_event(rng, names, kind, amount)
            if C.effective(before, cand) and C.within_train_range(before, cand):
                ev = cand
                break
        if ev is not None:
            break
    else:
        raise RuntimeError("no effective in-range event")
    if mode == "stacked":
        para = rng.choice(C.TRAIN_PARA[1:])
        surf = C.surface_with(before, para, C.draw_train_perm(rng, "top", allow_canon=False),
                              C.draw_train_perm(rng, "event", allow_canon=False),
                              C.draw_train_perm(rng, "before", n, allow_canon=False))
    else:
        if top not in TOP_OF:
            raise ValueError(f"a free slot needs its planned top order, got {top!r}")
        for _ in range(1000):
            para = rng.choice(C.TRAIN_PARA)
            surf = C.surface_with(before, para, TOP_OF[top], C.draw_train_perm(rng, "event"),
                                  C.draw_train_perm(rng, "before", n))
            if not C.stacked_surface(C.surface_classes(surf, before, ev)):
                break
        else:
            raise RuntimeError("free slot: no draw off the full stack")
    return before, ev, surf, {"names_kind": nk, "n_accounts": n, "bal_level": level, "amt_level": alevel, "para": para,
                              "surface_mode": mode}


def train_row(slot: dict, member: str, attempt: int, before, ev, surf, meta) -> dict:
    msgs = C.messages(before, ev, surf)
    st = ev["status"]
    return {"id": f"t2bl-tr-{slot['block']}-{slot['slot']:04d}-{member}", "prompt": C.render_chat(msgs),
            "target": C.target_text(before, ev), "family": C.LEDGER, "frame": "one", "block": slot["block"],
            "slot": slot["slot"], "attempt": attempt, "pair": len(slot["members"]) == 2, "member": member,
            "status_kind": "posted" if st == "posted" else "void" if st == "void" else C.WORD_KIND_OF[st],
            "before": before, "event": ev, "surface": surf, "meta": meta}


def build_train(seed: int, excl: dict, extra_pay=(), extra_ops=(), replay=None) -> tuple:
    m = env.load_marginals()
    pool = C.small_pool(m)
    fresh = Fresh(excl, extra_pay, extra_ops)
    replay = {tuple(r) for r in (replay or [])}
    rows, rejections, counts = [], [], Counter()
    for slot in train_plan(seed):
        key = (slot["block"], slot["slot"])
        for attempt in range(MAX_ATTEMPTS):
            rng = random.Random(f"t2bl-train|{seed}|{slot['block']}|{slot['slot']}|{attempt}")
            if slot["block"] == "canon":
                before, ev = draw_base(rng, slot["kind"], m, pool)
                surf, meta = C.canonical_surface(before), {"names_kind": "canonical", "n_accounts": 4,
                                                           "bal_level": "t2", "amt_level": "t2", "para": 0,
                                                           "surface_mode": "canon_block"}
            else:
                before, ev, surf, meta = draw_div(rng, slot["kind"], m, pool, slot["surface_mode"],
                                                  slot.get("free_top"))
            mem_rows = [train_row(slot, "posted" if st == "posted" else "nonposted", attempt, before,
                                  _with_status(ev, st), surf, meta) for st in slot["members"]]
            items = [(r["before"], r["event"]) for r in mem_rows]
            prob = "replayed rejection" if (slot["block"], slot["slot"], attempt) in replay else fresh.problem(items, key)
            if prob:
                rejections.append([slot["block"], slot["slot"], attempt])
                counts[prob] += 1
                continue
            fresh.add(items, key)
            rows += mem_rows
            break
        else:
            raise RuntimeError(f"training slot {key}: no fresh draw")
    random.Random(f"t2bl-train-shuffle|{seed}").shuffle(rows)
    rec = {"seed": seed, "n_rows": len(rows), "fresh_rejections": rejections, "rejection_counts": dict(counts)}
    return rows, rec


# ============================================================================================== records
def composition(rows: list) -> dict:
    out: dict = {"n": len(rows)}
    cl = [C.surface_classes(r["surface"], r["before"], r["event"]) for r in rows]
    for k in ("para", "top", "event", "before", "names", "count", "status", "bal", "amt"):
        out[k] = dict(sorted(Counter(c[k] for c in cl).items()))
    out["status_kind"] = dict(sorted(Counter(r["status_kind"] for r in rows).items()))
    out["kinds"] = dict(sorted(Counter(r["event"]["kind"] for r in rows).items()))
    out["blocks"] = dict(sorted(Counter(r["block"] for r in rows).items()))
    out["pairs"] = sum(1 for r in rows if r["pair"]) // 2
    out["n_accounts"] = dict(sorted(Counter(len(r["before"]) for r in rows).items()))
    out["words_rows"] = dict(sorted(Counter(r["event"]["status"] for r in rows if r["event"]["status"] not in
                                            ("posted", "void")).items()))
    out["fully_canonical_surface_rows"] = sum(1 for c in cl if all(c[k] == v for k, v in (
        ("para", "canonical"), ("top", "canonical"), ("event", "canonical"), ("before", "canonical"),
        ("names", "canonical"), ("count", "canonical"), ("bal", "t2"), ("amt", "t2"))) and c["status"] in ("posted", "void"))
    canon_rows = [r for r, c in zip(rows, cl) if all(c[k] == v for k, v in (
        ("para", "canonical"), ("top", "canonical"), ("event", "canonical"), ("before", "canonical"),
        ("names", "canonical"), ("count", "canonical"), ("bal", "t2"), ("amt", "t2")))]
    out["canonical_surface_status_kinds"] = dict(sorted(Counter(r["status_kind"] for r in canon_rows).items()))
    out["canonical_surface_void_trigger_accuracy"] = round(sum(
        ("void" in r["event"]["status"].lower()) == (r["event"]["status"] != "posted") for r in canon_rows)
        / max(1, len(canon_rows)), 4)
    out["surface_modes"] = dict(sorted(Counter((r.get("meta") or {}).get("surface_mode", "?") for r in rows).items()))
    out["surface_modes_by_status"] = {m: dict(sorted(Counter(
        "posted" if r["event"]["status"] == "posted" else "nonposted" for r in rows
        if (r.get("meta") or {}).get("surface_mode", "?") == m).items())) for m in sorted(out["surface_modes"])}
    n_posted = sum(r["event"]["status"] == "posted" for r in rows)
    for label, sel in (("stacked_surface", lambda c: C.stacked_surface(c)), ("top_reversed", lambda c: c["top"] == "train"),
                       ("top_sorted", lambda c: c["top"] == "canonical"),
                       ("canonical_surface", lambda c: all(c[k] == "canonical" for k in ("para", "top", "event", "before",
                                                                                         "names", "count")))):
        sub = [r for r, c in zip(rows, cl) if sel(c)]
        k = sum(r["event"]["status"] == "posted" for r in sub)
        out[label] = {"rows": len(sub), "posted": k, "nonposted": len(sub) - k,
                      "share_of_all_rows": round(len(sub) / max(1, len(rows)), 4),
                      "share_of_posted_rows": round(k / max(1, n_posted), 4),
                      "posted_share_within": round(k / max(1, len(sub)), 4)}
    out["distinct_event_orders"] = len({tuple(r["surface"]["event"]) for r in rows})
    out["max_abs_balance"] = max(abs(v) for r in rows for v in r["before"].values())
    out["max_abs_value_inputs_and_targets"] = max(abs(v) for r in rows for v in list(r["before"].values()) +
                                                  list(json.loads(r["target"]).values()) + [r["event"]["amount"]])
    out["max_abs_scale_row_balance"] = max(abs(v) for r in rows if r["event"]["kind"] == "scale"
                                           for v in r["before"].values())
    out["max_abs_amount_add_transfer"] = max(abs(r["event"]["amount"]) for r in rows if r["event"]["kind"] != "scale")
    out["scale_amounts"] = sorted({r["event"]["amount"] for r in rows if r["event"]["kind"] == "scale"})
    return out


def heuristic_scores(rows: list) -> dict:
    """Training accuracy of every heuristic (C.heuristic_states) against the gold targets."""
    hits: Counter = Counter()
    by_block: dict = {}
    for r in rows:
        gold = json.loads(r["target"])
        bb = by_block.setdefault(r["block"], Counter())
        bb["_n"] += 1
        for k, st in C.heuristic_states(r["before"], r["event"]).items():
            hits[k] += int(st == gold)
            bb[k] += int(st == gold)
    n = len(rows)
    sc = {k: round(v / n, 4) for k, v in sorted(hits.items())}
    descr = {b: {k: round(v / c["_n"], 4) for k, v in sorted(c.items()) if k != "_n"} for b, c in sorted(by_block.items())}
    probs = []
    for k, (lo, hi) in C.BINDING_HEURISTICS.items():
        if not lo <= sc[k] <= hi:
            probs.append(f"heuristic {k} scores {sc[k]} outside [{lo}, {hi}]")
    for k, v in sc.items():
        if k in C.BINDING_HEURISTICS or k in C.RULE_EQUIVALENT:
            continue
        if v > C.STRING_HEURISTIC_MAX:
            probs.append(f"string cue {k} scores {v} > {C.STRING_HEURISTIC_MAX}")
    for k in C.RULE_EQUIVALENT:
        if sc[k] != 1.0:
            probs.append(f"{k} (the rule) scores {sc[k]} != 1.0")
    # kit revision 3 (binding; operator decision 2026-10-12): copy and apply-regardless within [0.49, 0.51] on the
    # revised surfaces (built exactly 0.50: no surface-to-status cue), and the share of posted rows on the stacked
    # training surface (>= 1/3)
    subsets: dict = {}
    lo, hi = C.SURFACE_SUBSET_BOUNDS
    cls = [C.surface_classes(r["surface"], r["before"], r["event"]) for r in rows]
    n_posted = sum(r["event"]["status"] == "posted" for r in rows)
    for label, sel, binding in (("stacked_surface", C.stacked_surface, True),
                                ("top_reversed", lambda c: c["top"] == "train", True),
                                ("top_reversed_not_stacked", lambda c: c["top"] == "train" and not C.stacked_surface(c), False),
                                ("canonical_surface", lambda c: all(c[k] == "canonical" for k in ("para", "top", "event",
                                                                                                   "before", "names",
                                                                                                   "count")), False)):
        sub = [r for r, c in zip(rows, cls) if sel(c)]
        if not sub:
            subsets[label] = {"n": 0}
            if binding:
                probs.append(f"no training row on the {label} surface")
            continue
        hc: Counter = Counter()
        for r in sub:
            gold = json.loads(r["target"])
            for k, st in C.heuristic_states(r["before"], r["event"]).items():
                hc[k] += int(st == gold)
        d = {"n": len(sub), "posted": sum(r["event"]["status"] == "posted" for r in sub), "binding": binding,
             **{k: round(hc[k] / len(sub), 4) for k in ("copy", "apply_regardless", "void_trigger")}}
        subsets[label] = d
        if binding:
            for k in ("copy", "apply_regardless"):
                if not lo <= d[k] <= hi:
                    probs.append(f"heuristic {k} scores {d[k]} on the {label} rows, outside [{lo}, {hi}]")
    stacked_posted = subsets["stacked_surface"].get("posted", 0)
    share = round(stacked_posted / max(1, n_posted), 4)
    if share < C.MIN_STACKED_POSTED_SHARE:
        probs.append(f"only {share} of the posted rows are on the stacked training surface (< {C.MIN_STACKED_POSTED_SHARE})")
    return {"scores": sc, "binding": {k: list(v) for k, v in C.BINDING_HEURISTICS.items()},
            "string_cue_max": C.STRING_HEURISTIC_MAX, "problems": probs, "ok": not probs,
            "by_block_descriptive": descr,
            "surface_subsets": subsets, "surface_subset_bounds": list(C.SURFACE_SUBSET_BOUNDS),
            "stacked_posted_share_of_posted": share, "min_stacked_posted_share": C.MIN_STACKED_POSTED_SHARE,
            "note": "apply_regardless == ignore_status (apply every event as if posted); by_block is descriptive (the "
                    "canonical block is 2/3 posted, the diversified block 11/24 posted; a 'copy on an unfamiliar surface' "
                    "cue is what the dev gate's clause G3 tests on the held-out posted twins: accuracy and copy rate). "
                    "Kit revision 3 (operator decision 2026-10-12) binds copy and apply-regardless within the subset "
                    "bounds [0.49, 0.51] on the stacked-surface and on the reversed-top rows (the builder makes posted and "
                    "non-posted exactly equal there, so both score 0.50), and at least 1/3 of the posted rows on the "
                    "stacked surface. The first revision-3 build (superseded, never used) was 0.457 posted on both "
                    "subsets under bounds [0.40, 0.60]."}


def bank_heuristics(rows: list) -> dict:
    """Descriptive: per stratum, the accuracy of every heuristic on a bank (string cues from t2bl_common.heuristic_states;
    token cues from t2bl_audit.token_cue_noop: void_token, anchor_trigger = no-op iff the token 'void' is present or the
    token 'posted' with another token, token_overlap = no-op iff any token is shared with a trained non-posted status)."""
    import t2bl_audit as AU
    acc: dict = {}
    for r in rows:
        g = json.loads(r["gold"])
        st = r["event"]["status"]
        hs = C.heuristic_states(r["before"], r["event"])
        applied, copy = hs["apply_regardless"], hs["copy"]
        for k, noop in AU.token_cue_noop(st).items():
            hs[k] = copy if noop else applied
        a = acc.setdefault(r["stratum"], Counter())
        a["_n"] += 1
        for k, v in hs.items():
            if k not in C.RULE_EQUIVALENT:
                a[k] += int(v == g)
    return {s: {k: round(v / c["_n"], 3) for k, v in sorted(c.items()) if k != "_n"} for s, c in acc.items()}


def token_stats(train_rows: list, dev_rows: list) -> dict:
    try:
        import tokenizers
    except ImportError:
        return {"available": False}
    tk = tokenizers.Tokenizer.from_file(str(C.TOKENIZER_JSON))
    count = lambda t: len(tk.encode(t, add_special_tokens=False).ids)  # noqa: E731
    tr = [count(r["prompt"]) + count(r["target"]) + 1 for r in train_rows]
    tg = [count(r["target"]) for r in train_rows]
    dv = [count(C.render_chat(C.messages(r["before"], r["event"], r["surface"]))) for r in dev_rows]
    dg = [count(r["gold"]) for r in dev_rows]
    t2 = [json.loads(l) for l in (C.DATA / "t2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t2 = [r for r in t2 if r["family"] == C.LEDGER]
    t2r = [count(r["prompt"]) + count(r["target"]) + 1 for r in t2]
    mean = lambda xs: round(sum(xs) / len(xs), 2)  # noqa: E731
    return {"available": True, "train": {"mean_row_tokens_incl_eos": mean(tr), "max_row_tokens": max(tr),
                                         "mean_target_tokens": mean(tg), "max_target_tokens": max(tg)},
            "dev": {"mean_prompt_tokens": mean(dv), "max_prompt_tokens": max(dv), "mean_target_tokens": mean(dg),
                    "max_target_tokens": max(dg)},
            "t2_ledger_reference": {"mean_row_tokens_incl_eos": mean(t2r), "rows": len(t2)},
            "tokenizer_json_sha256": C.sha256(C.TOKENIZER_JSON)}


def t2_prompt_check() -> dict:
    """The canonical renderer reproduces every stored T2 ledger training prompt byte for byte."""
    t2 = [json.loads(l) for l in (C.DATA / "t2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t2 = [r for r in t2 if r["family"] == C.LEDGER]
    bad = sum(C.render_chat(C.messages(r["before"], r["event"], C.canonical_surface(r["before"]))) != r["prompt"]
              or C.target_text(r["before"], r["event"]) != r["target"] for r in t2)
    return {"t2_ledger_rows": len(t2), "mismatches": bad, "ok": bad == 0}


def overlap(rows_items: list, excl: dict) -> dict:
    P, MB, L1 = _builders()
    p = o = 0
    for b, e in rows_items:
        p += L1.payload(b, e) in excl["pay"]
        o += L1.opkey(C.LEDGER, b, e) in excl["ops"]
    return {"items": len(rows_items), "payload_overlap": p, "operative_overlap": o}


def _source_summary(excl: dict) -> dict:
    return {"rule": excl.get("rule"), "n_payloads": len(excl["pay"]), "n_operative": len(excl["ops"]),
            "sources": excl.get("sources"), "errors": union_errors(excl),
            "skipped_by_name_never_opened": (excl.get("sources", {}).get("historical") or {}).get("skipped_by_name_never_opened")}


# ============================================================================================== main build
def build_all(excl: dict, replay_dev=None, replay_train=None) -> dict:
    dev_rows, dev_rec = build_bank(C.SEED_DEV, C.DEV_STRATA, excl, "t2bl-dev", "t2bl-dev", replay=replay_dev, split="dev")
    P, MB, L1 = _builders()
    dpay = {L1.payload(r["before"], r["event"]) for r in dev_rows}
    dops = {L1.opkey(C.LEDGER, r["before"], r["event"]) for r in dev_rows}
    train_rows, train_rec = build_train(C.SEED_TRAIN, excl, dpay, dops, replay=replay_train)
    return {"dev": dev_rows, "dev_rec": dev_rec, "train": train_rows, "train_rec": train_rec}


def write_all(out_dir: Path, excl: dict, harvest: bool, lines: list) -> dict:
    import t2bl_audit as AU
    out_dir = Path(out_dir)
    paths = {"train": out_dir / C.TRAIN.name, "train_src": out_dir / C.TRAIN_SOURCE.name,
             "dev": out_dir / C.DEV.name, "dev_src": out_dir / C.DEV_SOURCE.name}
    for p in paths.values():
        if p.exists():
            raise SystemExit(f"{p} exists (create-only)")
    t0 = time.time()
    B = build_all(excl)
    _log(lines, f"built {len(B['dev'])} dev rows and {len(B['train'])} training rows ({time.time() - t0:.0f}s)")
    dev_text, train_text = C.dumps_rows(B["dev"]), C.dumps_rows(B["train"])
    # regenerate from (seed, recorded rejections) without the union: byte-identical
    R = build_all(empty_union(), replay_dev=B["dev_rec"]["fresh_rejections"], replay_train=B["train_rec"]["fresh_rejections"])
    regen = C.dumps_rows(R["dev"]) == dev_text and C.dumps_rows(R["train"]) == train_text
    if not regen:
        raise SystemExit("the build does not regenerate from (seed, recorded rejections)")
    au = AU.audit(B["train"], B["dev"])
    if not au["ok"]:
        raise SystemExit(f"audit failed: {au['problems'][:8]}")
    heur = heuristic_scores(B["train"])
    if not heur["ok"]:
        raise SystemExit(f"heuristic audit failed: {heur['problems']}")
    t2c = t2_prompt_check()
    if not t2c["ok"]:
        raise SystemExit(f"the canonical renderer does not reproduce T2's ledger prompts: {t2c}")
    ov_dev = overlap([(r["before"], r["event"]) for r in B["dev"]], excl)
    ov_tr = overlap([(r["before"], r["event"]) for r in B["train"]], excl)
    if harvest and (ov_dev["payload_overlap"] or ov_dev["operative_overlap"] or ov_tr["payload_overlap"] or ov_tr["operative_overlap"]):
        raise SystemExit(f"freshness overlap: dev {ov_dev}, train {ov_tr}")
    ts = token_stats(B["train"], B["dev"])
    bh = bank_heuristics(B["dev"])
    code = C.code_hashes(BUILD_CODE)
    common = {"gen_version": GEN_VERSION, "kit_revision": C.KIT_REVISION,
              "spent": {"seeds": list(C.SPENT_KIT_SEEDS), "runs": list(C.SPENT_RUNS),
                        "files": "data/t2bl_superseded_rev2/ (revision 2's training set and dev bank; R29)"},
              "superseded_unused": "docs/t2b/ledger/superseded_rev3_preop/: the first revision-3 build (generator "
                                   "t2bl-data-v3, the same seeds; 0.457 posted on the stacked and reversed-top rows), "
                                   "replaced at the operator decision of 2026-10-12 before any freeze or training; never "
                                   "trained on or evaluated, so not spent and outside every freshness union (its dev "
                                   "bank equals this one byte for byte)",
              "code_sha256": code, "harvest": harvest, "union": _source_summary(excl),
              "regenerates_from_seed_and_rejections": regen, "built_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "no_model_call": True, "torch_imported": "torch" in sys.modules}
    dev_sha = C.sha256_text(dev_text)
    train_sha = C.sha256_text(train_text)
    dev_src = {"record": "T2BL_DEV_SOURCE", "file": C.DEV.name, "file_sha256": dev_sha, **B["dev_rec"],
               "strata": {s: {"n": sum(r["stratum"] == s for r in B["dev"]), "factor": C.STRATA[s][0],
                              "gold_rule": C.STRATA[s][1], "surface": C.STRATA[s][2], "description": C.STRATA[s][3]}
                          for s in C.DEV_STRATA},
               "overlap": ov_dev, "heuristics_by_stratum_descriptive": bh,
               "use": f"T2b-ledger dev screen of kit revision {C.KIT_REVISION} (t2bl_run.py E0/E stages into "
                      f"{C.DEV_RUN_REL}; t2bl_gate.py). Never the PREREG-5 confirmation bank (minted at its freeze from "
                      "the CONFIRM half of the held-out partitions, fresh against this file).",
               **common}
    train_src = {"record": "T2BL_TRAIN_SOURCE", "file": C.TRAIN.name, "file_sha256": train_sha, **B["train_rec"],
                 "composition": composition(B["train"]), "heuristics": heur, "token_stats": ts,
                 "t2_prompt_check": t2c, "overlap": ov_tr, "dev_file_sha256": dev_sha,
                 "partitions": partitions_record(), "audit_summary": {k: au[k] for k in au if k != "rows"}, **common}
    C.write_create(paths["dev"], dev_text)
    C.write_create(paths["dev_src"], json.dumps(dev_src, indent=1) + "\n")
    C.write_create(paths["train"], train_text)
    C.write_create(paths["train_src"], json.dumps(train_src, indent=1) + "\n")
    _log(lines, f"wrote {paths['dev']} (sha {dev_sha[:16]}) and {paths['train']} (sha {train_sha[:16]})")
    return {"dev_sha256": dev_sha, "train_sha256": train_sha, "audit": au, "heuristics": heur, "token_stats": ts,
            "dev_heuristics": bh,
            "overlap": {"dev": ov_dev, "train": ov_tr}, "composition": train_src["composition"], "union": _source_summary(excl),
            "paths": {k: str(v) for k, v in paths.items()}}


def partitions_record() -> dict:
    halves = lambda dev, conf: {"dev": list(dev), "confirm": list(conf)}  # noqa: E731
    return {"status": {"train": {k: list(v) for k, v in C.TRAIN_WORDS.items()},
                       "heldout": {k: halves(C.HELDOUT_WORDS_DEV[k], C.HELDOUT_WORDS_CONFIRM[k]) for k in C.WORD_KINDS},
                       "heldout_rule": "token-novel under Titan's tokenizer: no token 'void' or 'posted', and at least one "
                                       "token no trained status contains (t2bl_audit.token_audit, binding)",
                       "prereg3_reserved_never_trained": list(C.P3_WORDS), "trained_canonical": ["posted", "void"],
                       "excluded_no_gold": list(C.EXCLUDED_STATUS)},
            "names": {"train_letters": list(C.TRAIN_LETTERS),
                      "heldout_letters": halves(C.HELDOUT_LETTERS_DEV, C.HELDOUT_LETTERS_CONFIRM),
                      "train_words": list(C.TRAIN_NAME_WORDS),
                      "heldout_words": halves(C.HELDOUT_NAME_WORDS_DEV, C.HELDOUT_NAME_WORDS_CONFIRM),
                      "canonical": list(C.CANON_NAMES), "lb_names_reserved": list(C.LB_NAMES)},
            "counts": {"train": list(C.TRAIN_COUNTS), "heldout": list(C.HELDOUT_COUNTS),
                       "note": "not split (the dev gate does not read H_count)"},
            "magnitudes": {"train_balance_classes": {"t2": "the recorded pool |b| <= 98", "b3": [100, 999], "b4": [1000, 9999]},
                           "train_scale_row_balances": "t2 / b3 only (at most 3 digits)",
                           "train_amount_classes_add_transfer": {"t2": "the recorded marginal -9..12", "a2": [13, 99], "a3": [100, 999]},
                           "scale_factors": "the recorded marginal only (train and test)",
                           "max_abs_value_in_training": C.MAX_TRAIN_ABS,
                           "heldout_balance_band": list(C.HELDOUT_BAL), "heldout_amount_band_add_transfer": list(C.HELDOUT_AMT),
                           "note": "not split (the dev gate does not read H_bal / H_amt; the confirmation draws fresh values)"},
            "key_orders": {"order_seed": C.ORDER_SEED, "min_kendall_from_sorted": C.MIN_KENDALL,
                           "heldout_position_permutations": {k: [list(p) for p in v] for k, v in C.HELDOUT_PERMS.items()},
                           "dev_indices": list(C.DEV_PERM_IDX), "confirm_indices": list(C.CONFIRM_PERM_IDX),
                           "train": "event / before: every other permutation (the identity w.p. 0.25 on free slots, "
                                    "redrawn where a free slot would be the full stack; never on stacked slots); the 2-key "
                                    "top object: BOTH orders (kit revision 3; reversed on an exact half of the free slots "
                                    "of each DIV slot type and on every stacked slot; revision 2 held the reversal out)",
                           "top_train_perms": [list(p) for p in C.TOP_TRAIN_PERMS],
                           "top_heldout": "none since kit revision 3 (the bank strata still reverse the top object; it is "
                                          "a trained level there)",
                           "lb_heldout_combined": {"top": list(C.LB_ORDER["top"]), "event": list(C.LB_ORDER["event"]),
                                                   "before": "reversed (the confirm half's index 0)"}},
            "paraphrases": {"train": list(C.TRAIN_PARA), "heldout": {"dev": C.HELDOUT_PARA_DEV, "confirm": C.HELDOUT_PARA_CONFIRM},
                            "texts": list(C.PARAPHRASES)}}


def verify(data_dir: Path = C.DATA, harvest: bool = True, quiet: bool = False) -> list:
    """Re-derive both files from the SOURCE records (seed + recorded rejections), rebuild the union (unless
    harvest=False), re-run the audit. Returns the list of problems (empty = ok)."""
    import t2bl_audit as AU
    data_dir = Path(data_dir)
    probs = []
    srcs = {k: C.read_json(data_dir / n) for k, n in (("dev", C.DEV_SOURCE.name), ("train", C.TRAIN_SOURCE.name))}
    for k, n in (("dev", C.DEV.name), ("train", C.TRAIN.name)):
        if srcs[k] is None:
            return [f"{k}: SOURCE record missing"]
        if C.sha256(data_dir / n) != srcs[k]["file_sha256"]:
            probs.append(f"{n}: sha256 differs from its SOURCE record")
    R = build_all(empty_union(), replay_dev=srcs["dev"]["fresh_rejections"], replay_train=srcs["train"]["fresh_rejections"])
    if C.sha256_text(C.dumps_rows(R["dev"])) != srcs["dev"]["file_sha256"]:
        probs.append("the dev bank does not regenerate from its seed and recorded rejections")
    if C.sha256_text(C.dumps_rows(R["train"])) != srcs["train"]["file_sha256"]:
        probs.append("the training set does not regenerate from its seed and recorded rejections")
    dev, train = C.load_jsonl(data_dir / C.DEV.name), C.load_jsonl(data_dir / C.TRAIN.name)
    au = AU.audit(train, dev)
    probs += [f"audit: {p}" for p in au["problems"]]
    h = heuristic_scores(train)
    probs += [f"heuristics: {p}" for p in h["problems"]]
    if harvest:
        excl = build_union(log=(lambda *a: None) if quiet else print)
        probs += [f"union source error: {e}" for e in union_errors(excl)]
        for k, rows in (("dev", dev), ("train", train)):
            ov = overlap([(r["before"], r["event"]) for r in rows], excl)
            if ov["payload_overlap"] or ov["operative_overlap"]:
                probs.append(f"{k}: freshness overlap {ov}")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-harvest", action="store_true", help="tests only: an empty union")
    a = ap.parse_args()
    if a.verify:
        probs = verify(Path(a.out_dir) if a.out_dir else C.DATA, harvest=not a.no_harvest)
        print(json.dumps({"ok": not probs, "problems": probs}, indent=1))
        return 0 if not probs else 1
    lines: list = []
    out = Path(a.out_dir) if a.out_dir else C.DATA
    excl = empty_union() if a.no_harvest else build_union(log=lambda m: _log(lines, m))
    if not a.no_harvest and union_errors(excl):
        raise SystemExit(f"union sources failed: {union_errors(excl)}")
    res = write_all(out, excl, not a.no_harvest, lines)
    if out == C.DATA:
        rec = {"record": "T2BL_AUDIT", "kit_revision": C.KIT_REVISION,
               "seeds": {"train": C.SEED_TRAIN, "dev": C.SEED_DEV, "heldout_orders": C.ORDER_SEED},
               "spent": {"seeds": list(C.SPENT_KIT_SEEDS), "runs": list(C.SPENT_RUNS)},
               "audit": res["audit"], "heuristics": res["heuristics"],
               "dev_bank_heuristics_descriptive": res["dev_heuristics"],
               "token_stats": res["token_stats"], "overlap": res["overlap"], "composition": res["composition"],
               "files": {"dev": res["dev_sha256"], "train": res["train_sha256"]},
               "union": {k: v for k, v in res["union"].items() if k != "sources"},
               "union_sources": res["union"]["sources"], "code_sha256": C.code_hashes(BUILD_CODE), "build_log": lines}
        print("wrote", C.write_doc("AUDIT.json", json.dumps(rec, indent=1) + "\n"))
    print(json.dumps({"dev_sha256": res["dev_sha256"], "train_sha256": res["train_sha256"],
                      "heuristics": res["heuristics"]["scores"], "overlap": res["overlap"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
