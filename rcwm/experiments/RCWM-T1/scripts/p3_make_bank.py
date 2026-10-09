"""PREREG-3 sealed out-of-range (OOR) ledger bank (v2): rule, semantic cue, substring cue or trigger? Built WITHOUT any
model call, tokenizer or torch.

  python -B scripts/p3_make_bank.py            # writes data/bank_p3_oor.jsonl + data/bank_p3_oor.SOURCE.json (create-only)
  python -B scripts/p3_make_bank.py --verify   # re-checks the written bank from the file, incl. a fresh harvest and the
                                               # status census; writes nothing

Rows (scripts/p3_common.STRATA, 15 one-step strata x 100 = 1,500 rows, plus 80 four-step chains). 100 BASES
(env.gen_start balances, an env.gen_event event whose posted version changes the state, kind quota 45/33/22). Every
one-step stratum is the same 100 bases with exactly ONE factor changed:
  status (10 strata, only the status string changes): T_posted (the twin), C_void, U_pending, U_draft, U_cancelled,
    U_reversed, U_held, U_neutral (approved / active / ok / dax / wug), N_voidcase (Void / VOID / voided),
    N_postedlike (unposted / not_posted / 'not posted');
  M_bal (balances -> 4-5 digits), M_amt (amount -> |13..500|), A_renamed (A-D -> E-H or words), A_many (+1..4 extra
    accounts; the event is the base's), F_keyorder (key order: top-level swap only, or all three orders permuted).
  chains: CH_S (10 parents x 4 candidates, statuses mixed, in range) and CH_X (10 x 4, statuses mixed + OOR amounts +
    reordered keys), accounts A-D so the PREREG-2 decision pipeline applies.

Gold: rcwmt.env.step, asserted equal to p3_common.rule_gold (the frame's rule text, implemented independently) and to
the validated FDS oracle (env.cross_check) on every row and every chain transition.

Freshness. No (before, event) payload and no FDS operative content (generate.operative_key; for a non-posted event that
is (before, not-posted)) of any row or chain transition equals one of:
  * the historical records set, harvested by scripts/l1_make_ood.harvest_historical (FDS harvest_exclusions line for
    line, EXCEPT files named HSKILL_CONFIRM_V2* are skipped by name: never opened, never hashed; DECISIONS I-28) -
    rcwmt.bank.load_exclusions() is NOT called;
  * every transition of data/bank_pilot, bank_confirm, bank_confirm2, bank_s4000 (incl. order twins);
  * every row of data/t2_train, t2_eval, t2_eval_xframe, t2_eval_supp; SELECT-VIEWS-001;
  * the L1 OOD bank docs/l1/ood_bank.jsonl;
  * the FDS-era banks T_BANK.json, T_BANK_V2.json and composition_confirmation T_BANK_V3X / STAGE_A_T_V3X / T_BANK_V4
    (read-only, pinned; outside the records tree, found by the PREREG-2 review's machine scan).
Within the bank, payloads are unique except that an F_keyorder row has its twin's payload (only the key order of the
prompt differs, by design); an operative content repeats only inside one base or inside one chain parent.

Multi-event prompts (binding) and status census (descriptive), in one pass over the records tree (same name-skipping
walk, same 60 MB limit) and the pinned sources. (1) Every {before|s0, events} ledger prompt (also JSON-in-string), which
the payload harvest does not expand, is rolled out with env.step; no bank transition may equal one of its (state,
event) payloads or operative contents. (2) Every occurrence of a bank status string as a JSON "status" value, in ANY
context, split into ledger contexts (the status is a field of an object that also has 'kind' and 'amount': a ledger
event) and others (e.g. a generation record's "status": "ok").
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
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import l1_make_ood as L1  # noqa: E402  (name-skipping historical harvest, pinned exclusion sources; torch-free)
import p3_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

LEDGER = C.LEDGER
MAX_DRAWS = 2_000_000
BUSINESS = Path(r"C:/Users/clclc/BusinessProject")
L1_OOD = ROOT / "docs" / "l1" / "ood_bank.jsonl"
EXTRA_SHA = {
    "l1_ood_bank.jsonl": "ac665c115b8c098e6bf58f6182a4a5a73b6fa1d87401afb1b3a55eeb165e940c",
    "fds/T_BANK.json": "26c01effe9610230f7876bbe118792f85094605fe3dde5f873c78c67e1c493ba",
    "fds/T_BANK_V2.json": "fc094136a5b581cb9cc03393b1e5959ed8f69247638c81e37cf7f68fbf132e01",
    "cc1/T_BANK_V3X.json": "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c",
    "cc2/STAGE_A_T_V3X.json": "635c9c7ea6851d4a0147a299ecd158a880b91fa5b9046b0af787d2ebbd6eb0d4",
    "cc2/T_BANK_V4.json": "82f0a6423dd6b58b81319708e7fbda7c5880b9734742323ea11574ff354c93c0",
}
EXTRA_PATHS = {
    "fds/T_BANK.json": BUSINESS / "ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/T_BANK.json",
    "fds/T_BANK_V2.json": BUSINESS / "ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/T_BANK_V2.json",
    "cc1/T_BANK_V3X.json": BUSINESS / "ChatBubble-titan-clare-sparse-v1/experiments/composition_confirmation_v1/T_BANK_V3X.json",
    "cc2/STAGE_A_T_V3X.json": BUSINESS / "ChatBubble-titan-clare-sparse-v1/experiments/composition_confirmation_v2/STAGE_A_T_V3X.json",
    "cc2/T_BANK_V4.json": BUSINESS / "ChatBubble-titan-clare-sparse-v1/experiments/composition_confirmation_v2/T_BANK_V4.json",
}
T2_TRAIN = ROOT / "data" / "t2_train.jsonl"
TITAN_TRAIN = L1.RECORDS / "rrp015-20260917" / "control-prepared" / "microfit-train.jsonl"   # Titan's 384 training rows
EVENT_KEYS = ("account", "amount", "kind", "other", "status", "tick")
# The v1 bank (14 strata, independent draws for M/A/F) was superseded before any model saw it (PREREG-3 review,
# 2026-10-09: no neutral status stratum; M/A/F not paired with the twins). Kept, never used, in docs/prereg3/superseded/.
SUPERSEDES = {"file": "docs/prereg3/superseded/bank_p3_oor.v1.jsonl",
              "sha256": "9e7108df6aa681b8f08769b2e37b976933c8406ffd671034f542e4068881b545",
              "reason": "PREREG-3 review 2026-10-09: add a status stratum without a not-applied meaning (U_neutral); "
                        "derive M/A/F strata from the status bases (paired, kind-matched twins); split F_keyorder into a "
                        "top-level-only and an all-orders variant. No model, tokenizer or model output saw v1."}
CENSUS_WINDOW = 300


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------- exclusion sources
def harvest_file(path: Path) -> set:
    """The l1_make_ood/FDS visitor applied to one file: every {before, event} object, also inside strings."""
    G = L1._fds()
    seen: set[str] = set()
    raw = path.read_text(encoding="utf-8", errors="ignore")

    def visit(doc):
        stack = [doc]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "before" in cur and "event" in cur:
                    seen.add(G.dumps({"before": cur["before"], "event": cur["event"]}))
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str) and '"before"' in cur:
                for o in G._objects_in(cur):
                    seen.add(G.dumps({"before": o["before"], "event": o["event"]}))
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            if line.strip():
                try:
                    visit(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        visit(json.loads(raw))
    return seen


def load_all_sources(lines: list, historical: bool = True) -> dict:
    src = L1.load_sources(lines, historical=historical)      # historical (name-skipping) + 4 banks + 4 T2 files + SELECT-VIEWS
    sha = L1.sha256_file(L1_OOD)
    if sha != EXTRA_SHA["l1_ood_bank.jsonl"]:
        raise SystemExit(f"L1 OOD bank sha256 {sha} != pinned")
    pay, ops = set(), set()
    for line in L1_OOD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pay.add(L1.payload(r["before"], r["event"]))
            ops.add(L1.opkey(r["family"], r["before"], r["event"]))
    src["l1_ood_bank.jsonl"] = {"payloads": pay, "ops": ops, "sha256": sha}
    L1.log(lines, f"l1_ood_bank.jsonl: {len(pay)} payloads")
    for name, p in EXTRA_PATHS.items():
        sha = L1.sha256_file(p)
        if sha != EXTRA_SHA[name]:
            raise SystemExit(f"{name}: sha256 {sha} != pinned {EXTRA_SHA[name]}")
        pay = harvest_file(p)
        src[name] = {"payloads": pay, "ops": L1.op_keys_of_payloads(pay), "sha256": sha, "path": str(p)}
        L1.log(lines, f"{name}: {len(pay)} payloads")
    return src


def union_status_counts(src: dict) -> dict:
    """Status strings of every single-step ledger payload in the exclusion union (descriptive)."""
    cnt = Counter()
    for v in src.values():
        for s in v["payloads"]:
            if '"status"' not in s:
                continue
            try:
                u = json.loads(s)
                if isinstance(u.get("before"), dict) and "A" in u["before"]:
                    cnt[str(u["event"].get("status"))] += 1
            except (ValueError, AttributeError):
                continue
    return dict(sorted(cnt.items()))


# ---------------------------------------------------------------------------------------------- status census
def census_files() -> tuple[list, list]:
    """(files, skipped): the records tree walked as L1.harvest_historical walks it (files named HSKILL_CONFIRM_V2* are
    skipped by NAME before any open; > 60 MB skipped) plus the pinned sources outside it."""
    files, skipped = [], []
    for path in list(L1.RECORDS.rglob("*.jsonl")) + list(L1.RECORDS.rglob("*.json")):
        if path.name.startswith(L1.RESERVED_PREFIX):
            skipped.append(str(path.relative_to(L1.RECORDS)).replace("\\", "/"))
            continue
        try:
            if path.stat().st_size > 60_000_000:
                continue
        except OSError:
            continue
        files.append(path)
    files += [ROOT / "data" / n for n in L1.BANKS + L1.T2_FILES] + [L1_OOD] + list(EXTRA_PATHS.values())
    return files, sorted(skipped)


def _multi_event_objects(doc):
    """Every {before|s0, events: [...]} object in a parsed document, also inside JSON-in-string prompts."""
    stack = [doc]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            st = cur.get("before", cur.get("s0"))
            if isinstance(st, dict) and isinstance(cur.get("events"), list):
                yield st, cur["events"]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str) and '"events"' in cur:
            i, j = cur.find("{"), cur.rfind("}")
            if 0 <= i < j:
                try:
                    stack.append(json.loads(cur[i:j + 1]))
                except (json.JSONDecodeError, ValueError):
                    pass


def _rollout_items(state: dict, events: list):
    """(state_t, event_t) of a ledger multi-event prompt rolled out with env.step (any account names, any status)."""
    if not state or not all(isinstance(v, int) and not isinstance(v, bool) for v in state.values()):
        return
    for ev in events:
        if not isinstance(ev, dict) or ev.get("kind") not in ("add", "scale", "transfer") or ev.get("account") not in state \
                or not isinstance(ev.get("amount"), int) or (ev["kind"] == "transfer" and ev.get("other") not in state):
            return
        yield state, ev
        state = env.step(LEDGER, state, ev)


def status_census(statuses=C.NONPOSTED_POOL + ("void",), bank_pay: set | None = None, bank_ops: set | None = None) -> dict:
    """Occurrences of each status string as a JSON "status" value (plain or JSON-in-string escaped), any context; and,
    when the bank's payload / operative-content sets are given, the overlap of the bank with every transition of every
    MULTI-EVENT ledger prompt in the same files (rolled out with env.step; the payload harvest does not expand them)."""
    alt = "|".join(re.escape(s) for s in sorted(statuses, key=len, reverse=True))
    rx = re.compile(r'\\?"status\\?"\s*:\s*\\?"(' + alt + r')\\?"')
    files, skipped = census_files()
    total, ledger = Counter(), Counter()
    where = defaultdict(Counter)
    me = {"files_with_multi_event_prompts": 0, "multi_event_objects": 0, "transitions": 0, "payload_overlap": 0,
          "operative_overlap": 0}
    for p in files:
        if p.name.startswith(L1.RESERVED_PREFIX):
            raise SystemExit("census: a reserved file reached the reader")       # unreachable by construction
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if bank_pay is not None and ('"events"' in raw or '\\"events\\"' in raw):
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
            n_obj = 0
            for doc in docs:
                for st, evs in _multi_event_objects(doc):
                    n_obj += 1
                    try:
                        for b, e in _rollout_items(st, evs):
                            me["transitions"] += 1
                            me["payload_overlap"] += L1.payload(b, e) in bank_pay
                            me["operative_overlap"] += L1.opkey(LEDGER, b, e) in bank_ops
                    except (KeyError, TypeError, ValueError):
                        continue
            me["multi_event_objects"] += n_obj
            me["files_with_multi_event_prompts"] += n_obj > 0
            del docs
        if '"status' not in raw and '\\"status' not in raw:
            continue
        try:
            rel = str(p.relative_to(L1.RECORDS)).replace("\\", "/")
        except ValueError:
            rel = str(p.relative_to(BUSINESS)).replace("\\", "/")
        for mt in rx.finditer(raw):
            s = mt.group(1)
            total[s] += 1
            lb, rb = raw.rfind("{", 0, mt.start()), raw.find("}", mt.end())
            obj = raw[lb:rb] if 0 <= lb and rb > 0 and rb - lb <= CENSUS_WINDOW else ""
            if "kind" in obj and "amount" in obj:
                ledger[s] += 1
                where[s][rel] += 1
        del raw
    return {"statuses": list(statuses), "files_scanned": len(files), "skipped_by_name_never_opened": skipped,
            **({"multi_event_rollouts": me} if bank_pay is not None else {}),
            "ledger_context": {s: ledger.get(s, 0) for s in statuses},
            "any_context": {s: total.get(s, 0) for s in statuses},
            "ledger_context_files": {s: dict(sorted(where[s].items())) for s in statuses if s != "void" and where[s]},
            "definition": ("regex (optional backslash)\"status(optional backslash)\": (optional backslash)\"<s>(optional "
                           "backslash)\" over the raw text (plain and escaped JSON; single-step and multi-event prompts "
                           "alike); ledger context = the innermost {...} around the match (from the last '{' before it to "
                           f"the first '}}' after it, at most {CENSUS_WINDOW} characters) contains 'kind' and 'amount', "
                           "i.e. the status is a field of a ledger event object; other hits (e.g. a generation record's "
                           "\"status\": \"ok\") count as any-context only")}


def training_stats() -> dict:
    """What T2 (and Titan's own 384 rows) trained on, for the OOR property checks."""
    L = [json.loads(l) for l in T2_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    L = [r for r in L if r["family"] == LEDGER]
    am = [r["event"]["amount"] for r in L]
    bv = [v for r in L for v in r["before"].values()]
    tk = [r["event"]["tick"] for r in L]
    sorted_keys = all(r["prompt"].count(env.dumps({"before": r["before"], "event": r["event"]})) == 1 for r in L)
    titan = Counter(re.findall(r'\\"status\\":\\"([^\\"]*)\\"', TITAN_TRAIN.read_text(encoding="utf-8")))
    return {"t2_train_sha256": L1.sha256_file(T2_TRAIN), "t2_ledger_rows": len(L),
            "t2_statuses": dict(Counter(r["event"]["status"] for r in L)),
            "t2_kinds": dict(Counter(r["event"]["kind"] for r in L)), "t2_amount_range": [min(am), max(am)],
            "t2_amounts": sorted(set(am)), "t2_max_abs_before": max(abs(v) for v in bv),
            "t2_accounts": sorted({k for r in L for k in r["before"]}), "t2_tick_range": [min(tk), max(tk)],
            "t2_user_json_compact_sorted_keys": sorted_keys,
            "titan_384_train_file": str(TITAN_TRAIN), "titan_384_train_sha256": L1.sha256_file(TITAN_TRAIN),
            "titan_384_ledger_statuses": dict(titan)}


# ---------------------------------------------------------------------------------------------- freshness bookkeeping
class Fresh:
    """Exclusion-union and within-bank freshness. owner: a base or a chain parent."""

    def __init__(self, excl_pay: set, excl_ops: set):
        self.excl_pay, self.excl_ops = excl_pay, excl_ops
        self.pay: set = set()
        self.ops: dict = {}

    def problem(self, items: list, owner: str) -> str | None:
        loc = set()
        for before, ev in items:
            p, o = L1.payload(before, ev), L1.opkey(LEDGER, before, ev)
            if p in self.excl_pay:
                return "payload in an exclusion source"
            if o in self.excl_ops:
                return "operative content in an exclusion source"
            if p in self.pay or p in loc:
                return "duplicate payload within the bank"
            if o in self.ops and self.ops[o] != owner:
                return "operative content shared with another unit of the bank"
            loc.add(p)
        return None

    def add(self, items: list, owner: str) -> None:
        for before, ev in items:
            self.pay.add(L1.payload(before, ev))
            self.ops.setdefault(L1.opkey(LEDGER, before, ev), owner)


# ---------------------------------------------------------------------------------------------- draws
def _perm_not_sorted(rng, keys):
    keys = sorted(keys)
    while True:
        p = list(keys)
        rng.shuffle(p)
        if p != keys:
            return p


def reorder_ko(rng, before, ev) -> dict:
    return {"top": ["event", "before"], "event": _perm_not_sorted(rng, ev), "before": _perm_not_sorted(rng, before)}


def top_only_ko(before, ev) -> dict:
    return {"top": ["event", "before"], "event": sorted(ev), "before": sorted(before)}


def status_of(stratum: str, idx: int) -> str:
    s = C.STRATA[stratum][2]
    return s[idx % len(s)] if isinstance(s, tuple) else s


def derive(stratum: str, idx: int, before: dict, ev0: dict, rng, small_pool: list) -> tuple:
    """(before, event, key_order, meta) of a non-status stratum on base idx: the base with ONE factor changed (posted)."""
    ev = {**ev0, "status": "posted"}
    if stratum == "M_bal":
        digits = [rng.choice(C.BIG_DIGITS) for _ in env.ACCOUNTS]
        b = {a: rng.choice((-1, 1)) * rng.randint(10 ** (d - 1), 10 ** d - 1) for a, d in zip(env.ACCOUNTS, digits)}
        return b, ev, None, {"digits": digits}
    if stratum == "M_amt":
        lo, hi = C.AMT_BANDS[idx % 2]
        return dict(before), {**ev, "amount": rng.choice((-1, 1)) * rng.randint(lo, hi)}, None, {"amount_band": f"{lo}..{hi}"}
    if stratum == "A_renamed":
        naming, names = ("letters", list(C.LETTER_NAMES)) if idx % 2 == 0 else ("words", sorted(rng.sample(C.WORD_NAMES, 4)))
        mp = dict(zip(env.ACCOUNTS, names))
        return ({mp[a]: v for a, v in before.items()}, {**ev, "account": mp[ev["account"]], "other": mp[ev["other"]]}, None,
                {"naming": naming, "names": names})
    if stratum == "A_many":
        n = 5 + idx % 4
        b = dict(before)
        for a in C.MANY_NAMES[4:n]:
            b[a] = rng.choice(small_pool)
        return b, ev, None, {"n_accounts": n}
    if stratum == "F_keyorder":
        if idx % 2 == 0:
            return dict(before), ev, top_only_ko(before, ev), {"key_variant": "top_only"}
        return dict(before), ev, reorder_ko(rng, before, ev), {"key_variant": "all_orders"}
    raise ValueError(stratum)


def onestep_row(stratum, idx, before, ev, ko, base, meta) -> dict:
    gold = C.gold_of(before, ev)
    return {"id": f"p3-{stratum}-{idx:03d}", "kind": "onestep", "stratum": stratum, "group": C.STRATA[stratum][0],
            "base": base, "before": before, "event": ev, "key_order": ko, "messages": C.messages(before, ev, ko),
            "gold": env.dumps(gold), "gold_rule": C.STRATA[stratum][1],
            "posted_gold": env.dumps(C.gold_of(before, {**ev, "status": "posted"})), "meta": meta}


def chain_status(stratum, rng) -> str:
    u = rng.random()
    if stratum == "CH_S":
        if u < 0.50:
            return "posted"
        if u < 0.85:
            return rng.choice(C.UNSEEN_WORDS)
        return rng.choice(C.NEAR_VOID + C.POSTED_LIKE)
    return "posted" if u < 0.60 else rng.choice(C.NONPOSTED_POOL)


def step_class(ev: dict, ko) -> str:
    st = ev["status"]
    if st == "posted":
        c = "posted" + ("_oor_amount" if abs(ev["amount"]) >= 13 else "")
    elif st == "void":
        c = "void"
    elif st in C.NOT_APPLIED:
        c = "unseen"
    elif st in C.NEUTRAL:
        c = "neutral"
    elif st in C.NEAR_VOID:
        c = "near_void"
    elif st in C.POSTED_LIKE:
        c = "posted_like"
    else:
        raise ValueError(st)
    return c + ("+reordered" if ko is not None else "")


def draw_candidate(stratum, s0, rng, m):
    """4 events (and key orders) from s0; each event's posted version changes its gold input state."""
    events, kos, st = [], [], s0
    for _ in range(C.H):
        while True:
            ev = {**env.gen_event(LEDGER, rng, m, tick=rng.randint(1, 99)), "status": chain_status(stratum, rng)}
            if stratum == "CH_X" and ev["kind"] != "scale" and rng.random() < 0.40:
                lo, hi = C.AMT_BANDS[rng.randrange(2)]
                ev["amount"] = rng.choice((-1, 1)) * rng.randint(lo, hi)
            if C.gold_of(st, {**ev, "status": "posted"}) != st:
                break
        ko = reorder_ko(rng, st, ev) if (stratum == "CH_X" and rng.random() < 0.50) else None
        events.append(ev)
        kos.append(ko)
        st = C.gold_of(st, ev)
    return events, kos


def candidate_problem(stratum, events, kos) -> str | None:
    posted = [e["status"] == "posted" for e in events]
    if not any(posted) or all(posted):
        return "needs >= 1 posted and >= 1 non-posted event"
    if stratum == "CH_S" and any(k is not None for k in kos):
        return "CH_S has sorted keys only"
    if stratum == "CH_X":
        if not any(e["status"] == "posted" and abs(e["amount"]) >= 13 for e in events):
            return "needs >= 1 posted OOR-amount event"
        if not any(k is not None for k in kos):
            return "needs >= 1 reordered step"
    return None


# ---------------------------------------------------------------------------------------------- build
def build(lines: list) -> tuple[list, list, dict]:
    code = L1.check_code()
    L1.log(lines, f"code identity: {code}")
    src = load_all_sources(lines)
    excl_pay = set().union(*(v["payloads"] for v in src.values()))
    excl_ops = set().union(*(v["ops"] for v in src.values()))
    L1.log(lines, f"exclusion union: {len(excl_pay)} payloads, {len(excl_ops)} operative contents")
    m = env.load_marginals()
    small_pool = [b for b in m.ledger_balances if -99 <= b <= 99]          # env.gen_start's pool
    rng = random.Random(C.SEED)
    fresh = Fresh(excl_pay, excl_ops)
    rejected = {k: Counter() for k in ("bases",) + C.CHAIN_STRATA}
    draws = {}
    per = {s: [] for s in C.STRATA}
    # 1. bases: one (before, event) per base; every stratum derived from it (one factor changed)
    made, n = 0, 0
    quota = dict(C.KIND_QUOTA)
    while made < C.N_PER_STRATUM:
        n += 1
        if n > MAX_DRAWS:
            raise SystemExit("bases: could not fill")
        before = env.gen_start(LEDGER, rng, m)
        ev0 = env.gen_event(LEDGER, rng, m, tick=rng.randint(1, 99))
        if C.gold_of(before, {**ev0, "status": "posted"}) == before:
            rejected["bases"]["posted version is a no-op"] += 1
            continue
        if quota[ev0["kind"]] <= 0:
            rejected["bases"]["kind quota full"] += 1
            continue
        evs = {s: {**ev0, "status": status_of(s, made)} for s in C.STATUS_STRATA}
        der = {s: derive(s, made, before, ev0, rng, small_pool) for s in C.GEN_STRATA}
        for s, (b, e, _, _) in der.items():
            if C.gold_of(b, e) == b:
                raise AssertionError(f"derived {s} row is a no-op although its base is not")
        base = f"b{made:03d}"
        items = [(before, e) for e in evs.values()] + [(b, e) for s, (b, e, _, _) in der.items() if s != "F_keyorder"]
        why = fresh.problem(items, base)
        if why:
            rejected["bases"][why] += 1
            continue
        fresh.add(items, base)
        quota[ev0["kind"]] -= 1
        for s, e in evs.items():
            per[s].append(onestep_row(s, made, before, e, None, base, {"kind": e["kind"], "status": e["status"]}))
        for s, (b, e, ko, meta) in der.items():
            per[s].append(onestep_row(s, made, b, e, ko, base, {"kind": e["kind"], **meta}))
        made += 1
    draws["bases"] = n
    L1.log(lines, f"bases: {made} from {n} draws; rejected {dict(rejected['bases'])}")
    rows = [r for s in C.STRATA for r in per[s]]
    # 2. chains (k = 4), K candidates per parent, accounts A-D
    chains = []
    for cs in C.CHAIN_STRATA:
        n, n_par = 0, 0
        while n_par < C.N_CHAIN_PARENTS[cs]:
            s0 = env.gen_start(LEDGER, rng, m)
            cands = []
            while len(cands) < C.K_CANDIDATES:
                n += 1
                if n > MAX_DRAWS:
                    raise SystemExit(f"{cs}: could not fill")
                events, kos = draw_candidate(cs, s0, rng, m)
                why = candidate_problem(cs, events, kos)
                if why:
                    rejected[cs][why] += 1
                    continue
                cands.append((events, kos))
            pid = f"P3{cs.replace('_', '')}-" + sha256_bytes(env.dumps({"s0": s0, "c": [e for e, _ in cands]}).encode())[:16]
            items = []
            for events, _ in cands:
                st = env.rollout(LEDGER, s0, events)
                items += [(st[t], e) for t, e in enumerate(events)]
            why = fresh.problem(items, pid)
            if why:
                rejected[cs]["parent: " + why] += 1                 # the whole parent is redrawn
                continue
            fresh.add(items, pid)
            n_par += 1
            for c, (events, kos) in enumerate(cands):
                st = [s0]
                for e in events:
                    st.append(C.gold_of(st[-1], e))
                chains.append({"id": f"{pid}-c{c}", "kind": "chain", "stratum": cs, "parent_id": pid, "cand_id": f"{pid}-c{c}",
                               "s0": s0, "events": events, "key_orders": kos, "golds": [env.dumps(x) for x in st],
                               "meta": {"step_classes": [step_class(e, k) for e, k in zip(events, kos)]}})
        draws[cs] = n
        L1.log(lines, f"{cs}: {C.N_CHAIN_PARENTS[cs]} parents x {C.K_CANDIDATES} candidates from {n} candidate draws; "
                      f"rejected {dict(rejected[cs])}")
    overlaps = overlap_counts(rows, chains, src)
    if any(v["payload"] or v["operative"] for v in overlaps.values()):
        raise SystemExit(f"overlap with an exclusion source: {overlaps}")
    L1.log(lines, f"overlaps (payload / operative content) with every source: all 0 ({len(overlaps)} sources)")
    return rows, chains, {"src": src, "rejected": rejected, "draws": draws, "overlaps": overlaps, "code": code,
                          "excl_union": {"payloads": len(excl_pay), "operative": len(excl_ops)}}


def transitions(rows: list, chains: list):
    for r in rows:
        yield r["before"], r["event"]
    for c in chains:
        st = env.rollout(LEDGER, c["s0"], c["events"])
        for t, e in enumerate(c["events"]):
            yield st[t], e


def bank_payloads(rows: list, chains: list) -> set:
    return {L1.payload(b, e) for b, e in transitions(rows, chains)}


def bank_opkeys(rows: list, chains: list) -> set:
    return {L1.opkey(LEDGER, b, e) for b, e in transitions(rows, chains)}


def overlap_counts(rows: list, chains: list, src: dict) -> dict:
    pay = {L1.payload(b, e) for b, e in transitions(rows, chains)}
    ops = {L1.opkey(LEDGER, b, e) for b, e in transitions(rows, chains)}
    return {name: {"payload": len(pay & v["payloads"]), "operative": len(ops & v["ops"])} for name, v in src.items()}


def before_overlap(rows: list, chains: list, src: dict) -> dict:
    """Descriptive: bank before-states that occur as the before of SOME ledger payload in a source (any event)."""
    bs = {env.dumps(b) for b, _ in transitions(rows, chains)}
    out = {}
    for name, v in src.items():
        seen = set()
        for s in v["payloads"]:
            if '"A"' in s:
                try:
                    u = json.loads(s)
                    if isinstance(u.get("before"), dict):
                        seen.add(env.dumps(u["before"]))
                except ValueError:
                    continue
        out[name] = len(bs & seen)
    return out


# ---------------------------------------------------------------------------------------------- definitions / checks
def _same_except(a: dict, b: dict, keys) -> bool:
    return set(a) == set(b) and all(a[k] == b[k] for k in a if k not in keys)


def stratum_problem(r: dict, twin: dict, marg_amounts: set) -> str | None:
    """None if a one-step row satisfies its stratum's definition, re-derived from the row and its base's twin alone."""
    s, b, e, ko = r["stratum"], r["before"], r["event"], r["key_order"]
    tb, te = twin["before"], twin["event"]
    if tuple(sorted(e)) != EVENT_KEYS or not 1 <= e["tick"] <= 99 or e["kind"] not in ("add", "scale", "transfer"):
        return "event fields"
    if e["account"] == e["other"] or e["account"] not in b or e["other"] not in b:
        return "accounts"
    group, rule = C.STRATA[s][0], C.STRATA[s][1]
    if r["gold_rule"] != rule or r["group"] != group or r["base"] != twin["base"]:
        return "gold rule / group / base label"
    if C.gold_of(b, {**e, "status": "posted"}) == b:
        return "the posted version does not change the state"
    if (rule == "noop") != (e["status"] != "posted"):
        return "status does not match the gold rule"
    if (s == "F_keyorder") != (ko is not None):
        return "key order present iff F_keyorder"
    idx = int(r["base"][1:])
    if r["id"] != f"p3-{s}-{idx:03d}" or r["meta"].get("kind") != e["kind"]:
        return "id / meta"
    if sorted(tb) != list(env.ACCOUNTS) or any(abs(v) > 99 for v in tb.values()) or te["amount"] not in marg_amounts \
            or te["status"] != "posted":
        return "twin (base) definition"
    if group == "status":
        ok = e["status"] == status_of(s, idx) and b == tb and _same_except(e, te, ("status",))
        return None if ok else "status row: base with only the status changed"
    if s == "M_bal":
        ok = (sorted(b) == list(env.ACCOUNTS) and all(len(str(abs(v))) in C.BIG_DIGITS for v in b.values()) and e == te
              and r["meta"]["digits"] == [len(str(abs(b[a]))) for a in env.ACCOUNTS])
        return None if ok else "M_bal: the base's event on 4-5 digit balances"
    if s == "M_amt":
        lo, hi = C.AMT_BANDS[idx % 2]
        ok = b == tb and _same_except(e, te, ("amount",)) and lo <= abs(e["amount"]) <= hi and r["meta"]["amount_band"] == f"{lo}..{hi}"
        return None if ok else "M_amt: the base with only the amount changed"
    if s == "A_renamed":
        names = list(C.LETTER_NAMES) if idx % 2 == 0 else r["meta"]["names"]
        ok_names = (r["meta"]["naming"] == ("letters" if idx % 2 == 0 else "words") and len(set(names)) == 4
                    and (idx % 2 == 0 or (set(names) <= set(C.WORD_NAMES) and names == sorted(names))))
        mp = dict(zip(env.ACCOUNTS, names))
        ok = ok_names and b == {mp[a]: v for a, v in tb.items()} and e == {**te, "account": mp[te["account"]], "other": mp[te["other"]]}
        return None if ok else "A_renamed: the base relabelled"
    if s == "A_many":
        n = 5 + idx % 4
        ok = (sorted(b) == list(C.MANY_NAMES[:n]) and {a: b[a] for a in env.ACCOUNTS} == tb and e == te
              and all(abs(b[a]) <= 99 for a in C.MANY_NAMES[4:n]) and r["meta"]["n_accounts"] == n)
        return None if ok else "A_many: the base plus extra accounts"
    if s == "F_keyorder":
        if b != tb or e != te or ko["top"] != ["event", "before"]:
            return "F_keyorder: the base with only the key order changed"
        if idx % 2 == 0:
            ok = ko == top_only_ko(b, e) and r["meta"]["key_variant"] == "top_only"
        else:
            ok = ko["event"] != sorted(e) and ko["before"] != sorted(b) and r["meta"]["key_variant"] == "all_orders"
        return None if ok else "F_keyorder variant"
    return "unknown stratum"


def rule_consistency_table(rows: list) -> dict:
    """Per stratum: the clause of the frame's rule text that decides the gold, and the count of rows on which the bank
    is CONSISTENT with it (gold == before on no-op rows; gold == rule_gold != before on apply rows). This is a
    consistency check of the builder (it holds by construction, since gold_of asserts env.step == rule_gold); the
    argument that the rule TEXT implies each gold is the per-stratum prose in PREREG_3_DRAFT.md §2.3."""
    out = OrderedDict()
    for s, (group, rule, status, desc) in C.STRATA.items():
        rs = [r for r in rows if r["stratum"] == s]
        ok = sum((json.loads(r["gold"]) == r["before"]) if rule == "noop" else
                 (json.loads(r["gold"]) == C.rule_gold(r["before"], r["event"]) != r["before"]) for r in rs)
        out[s] = {"n": len(rs), "group": group, "gold_rule": rule, "statuses": dict(Counter(r["event"]["status"] for r in rs)),
                  "clause": C.RULE_CLAUSES[rule], "rows_consistent": ok, "description": desc}
    return out


def describe(rows: list, chains: list) -> dict:
    out = {}
    for s in C.STRATA:
        rs = [r for r in rows if r["stratum"] == s]
        bv = [v for r in rs for v in r["before"].values()]
        am = [r["event"]["amount"] for r in rs]
        d = {"kinds": dict(sorted(Counter(r["event"]["kind"] for r in rs).items())),
             "statuses": dict(sorted(Counter(r["event"]["status"] for r in rs).items())),
             "sub_variants": dict(sorted(Counter(C.sub_of(r) for r in rs).items())),
             "balance_range": [min(bv), max(bv)], "amount_range": [min(am), max(am)],
             "n_accounts": dict(sorted(Counter(len(r["before"]) for r in rs).items())),
             "gold_abs_max": max(abs(v) for r in rs for v in json.loads(r["gold"]).values())}
        for kc in ("at", "sc"):
            g = [max(abs(v) for v in json.loads(r["gold"]).values()) for r in rs if C.kind_class(r["event"]["kind"]) == kc]
            d[f"gold_abs_max_{kc}"] = {"median": statistics.median(g), "max": max(g), "n": len(g)}
        out[s] = d
    for cs in C.CHAIN_STRATA:
        cc = [c for c in chains if c["stratum"] == cs]
        fin = [max(abs(v) for v in json.loads(c["golds"][-1]).values()) for c in cc]
        out[cs] = {"parents": len({c["parent_id"] for c in cc}), "candidates": len(cc),
                   "step_classes": dict(sorted(Counter(x for c in cc for x in c["meta"]["step_classes"]).items())),
                   "copy_chain_exact_floor_k4": sum(c["golds"][4] == c["golds"][0] for c in cc) / len(cc),
                   "final_abs_max": max(fin),
                   "chains_with_a_final_balance_beyond_phi_clip_300": sum(f > 300 for f in fin)}
    return out


def oor_property_checks(rows: list, chains: list, tr: dict) -> dict:
    """Every OOR stratum lies outside what T2 trained on (statuses, magnitudes, names, key order)."""
    by = {s: [r for r in rows if r["stratum"] == s] for s in C.STRATA}
    trained_status = set(tr["t2_statuses"]) | set(tr["titan_384_ledger_statuses"])
    lo_a, hi_a = tr["t2_amount_range"]
    checks = {
        "unseen, neutral, near-miss and look-alike statuses never occur in T2's or Titan's ledger training rows":
            not (set(C.NONPOSTED_POOL) & trained_status) and trained_status == {"posted", "void"},
        "T_posted / C_void use only trained statuses (posted / void) on in-range bases":
            all(r["event"]["status"] == "posted" for r in by["T_posted"]) and all(r["event"]["status"] == "void" for r in by["C_void"]),
        "M_bal: every balance |b| >= 1000 > T2's max |before|": all(abs(v) >= 1000 > tr["t2_max_abs_before"] for r in by["M_bal"] for v in r["before"].values()),
        "M_amt: every |amount| >= 13, outside T2's amount range": all(not lo_a <= r["event"]["amount"] <= hi_a and abs(r["event"]["amount"]) >= 13 for r in by["M_amt"]),
        "A_renamed: no account named A-D": all(not set(r["before"]) & set(tr["t2_accounts"]) for r in by["A_renamed"]),
        "A_many: 5-8 accounts (T2: exactly A-D)": all(5 <= len(r["before"]) <= 8 for r in by["A_many"]),
        "F_keyorder: user JSON differs from the sorted frame T2 trained on, and starts with the event":
            tr["t2_user_json_compact_sorted_keys"] and all(r["messages"][1]["content"] != env.dumps({"before": r["before"], "event": r["event"]})
                                                       and r["messages"][1]["content"].startswith('{"event":') for r in by["F_keyorder"]),
        "every one-step row outside M_amt, and every CH_S event, keeps an amount T2 trained on (-9..12)":
            all(r["event"]["amount"] in tr["t2_amounts"] for r in rows if r["stratum"] != "M_amt")
            and all(e["amount"] in tr["t2_amounts"] for c in chains if c["stratum"] == "CH_S" for e in c["events"]),
        "every row outside M_bal keeps |before| <= 99 (T2's range)":
            all(abs(v) <= 99 for r in rows if r["stratum"] != "M_bal" for v in r["before"].values()),
        "chains: accounts A-D (the PREREG-2 decision pipeline applies)": all(sorted(c["s0"]) == list(env.ACCOUNTS) for c in chains),
    }
    return {k: bool(v) for k, v in checks.items()}


# ---------------------------------------------------------------------------------------------- verify
def verify(bank_path: Path = C.BANK, source_path: Path = C.SOURCE, historical: bool = True, quiet: bool = False) -> list:
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    rec = json.loads(Path(source_path).read_text(encoding="utf-8"))
    fsha = L1.sha256_file(bank_path)
    chk("bank file sha256 == its SOURCE record", fsha == rec["file_sha256"], fsha[:16])
    rows, chains = C.load_bank(bank_path)
    per = Counter(r["stratum"] for r in rows)
    chk(f"{len(C.STRATA)} one-step strata x {C.N_PER_STRATUM}, ids unique, in stratum order",
        per == Counter({s: C.N_PER_STRATUM for s in C.STRATA}) and len({r["id"] for r in rows}) == len(rows)
        and [r["stratum"] for r in rows] == sorted((r["stratum"] for r in rows), key=list(C.STRATA).index), dict(per))
    m = env.load_marginals()
    marg = set(m.ledger_amounts)
    twins = {r["base"]: r for r in rows if r["stratum"] == "T_posted"}
    bad = Counter()
    for r in rows:
        if r["messages"] != C.messages(r["before"], r["event"], r["key_order"]):
            bad["messages"] += 1
        if r["key_order"] is None and r["messages"] != env.one_step_messages(LEDGER, r["before"], r["event"]):
            bad["frame"] += 1
        if r["messages"][0]["content"] != C.RULE_TEXT:
            bad["rule text"] += 1
        try:
            if r["gold"] != env.dumps(C.gold_of(r["before"], r["event"])) or \
                    r["posted_gold"] != env.dumps(C.gold_of(r["before"], {**r["event"], "status": "posted"})):
                bad["gold"] += 1
        except AssertionError:
            bad["oracle"] += 1
        tw = twins.get(r.get("base"))
        why = "no twin" if tw is None else stratum_problem(r, tw, marg)
        if why is not None:
            bad["stratum: " + why] += 1
    chk("messages == the one-step frame (env.one_step_messages when keys are sorted), rule text verbatim",
        not (bad["messages"] or bad["frame"] or bad["rule text"]))
    chk("gold == env.step == the stated rule == the FDS oracle on every row (posted_gold likewise)", not (bad["gold"] or bad["oracle"]))
    chk("every row is its base with exactly its stratum's factor changed (re-derived from the row and its twin)",
        not [k for k in bad if k.startswith("stratum")], str({k: v for k, v in bad.items() if k.startswith("stratum")}))
    by_base = Counter(r["base"] for r in rows)
    chk(f"{C.N_PER_STRATUM} bases, each in all {len(C.STRATA)} strata", len(by_base) == C.N_PER_STRATUM
        and set(by_base.values()) == {len(C.STRATA)})
    chk(f"every one-step stratum holds the kind quota {C.KIND_QUOTA}",
        all(Counter(r["event"]["kind"] for r in rows if r["stratum"] == s) == Counter(C.KIND_QUOTA) for s in C.STRATA))
    cbad = Counter()
    pc = Counter(c["parent_id"] for c in chains)
    for c in chains:
        st = [c["s0"]]
        try:
            for e in c["events"]:
                st.append(C.gold_of(st[-1], e))
        except AssertionError:
            cbad["oracle"] += 1
            continue
        if c["golds"] != [env.dumps(x) for x in st]:
            cbad["golds"] += 1
        if candidate_problem(c["stratum"], c["events"], c["key_orders"]):
            cbad["definition"] += 1
        for t, e in enumerate(c["events"]):
            if C.gold_of(st[t], {**e, "status": "posted"}) == st[t]:
                cbad["posted version no-op"] += 1
            if c["key_orders"][t] is not None:
                C.user_json(st[t], e, c["key_orders"][t])
        if c["meta"]["step_classes"] != [step_class(e, k) for e, k in zip(c["events"], c["key_orders"])]:
            cbad["meta"] += 1
    chk(f"chains: {sum(C.N_CHAIN_PARENTS.values())} parents x {C.K_CANDIDATES} candidates x {C.H} events; golds roll out by "
        "env.step == rule == oracle; definitions hold", not cbad and len(chains) == sum(C.N_CHAIN_PARENTS.values()) * C.K_CANDIDATES
        and set(pc.values()) == {C.K_CANDIDATES}, str(dict(cbad)))
    pay = [L1.payload(b, e) for b, e in transitions([r for r in rows if r["stratum"] != "F_keyorder"], chains)]
    fk_twin = all(L1.payload(r["before"], r["event"]) == L1.payload(twins[r["base"]]["before"], twins[r["base"]]["event"])
                  for r in rows if r["stratum"] == "F_keyorder")
    chk("payloads unique within the bank, except that each F_keyorder row has its twin's payload (by design)",
        len(set(pay)) == len(pay) and fk_twin)
    lines: list = []
    src = load_all_sources(lines, historical=historical)
    ov = overlap_counts(rows, chains, src)
    chk("zero payload and zero operative-content overlap with every exclusion source"
        + ("" if historical else " (historical set not re-harvested here)"),
        all(v["payload"] == 0 and v["operative"] == 0 for v in ov.values()),
        "; ".join(f"{k} {v['payload']}/{v['operative']}" for k, v in ov.items()))
    chk("exclusion-source sizes equal the build record's",
        all(rec["exclusion_sources"][k]["payloads"] == len(v["payloads"]) for k, v in src.items()))
    if historical:
        sk = src["historical"]["skipped_by_name_never_opened"]
        chk("historical harvest skipped exactly the reserved HSKILL_CONFIRM_V2* files by name (never opened)",
            sk == rec["exclusion_sources"]["historical"]["skipped_by_name_never_opened"]
            and all(Path(p).name.startswith(L1.RESERVED_PREFIX) for p in sk), str(sk))
        cen = status_census(bank_pay=bank_payloads(rows, chains), bank_ops=bank_opkeys(rows, chains))
        mer = cen["multi_event_rollouts"]
        chk("zero payload and zero operative-content overlap with every multi-event ledger prompt rolled out in the "
            "records tree and the pinned sources",
            mer["payload_overlap"] == 0 and mer["operative_overlap"] == 0 and mer["transitions"] > 0,
            f"{mer['multi_event_objects']} prompts, {mer['transitions']} transitions in {mer['files_with_multi_event_prompts']} files")
        chk("status census (any context, incl. multi-event prompts) reproduces the build record; reserved files skipped by name",
            {k: v for k, v in cen.items() if k != "definition"} == {k: v for k, v in rec["status_census"].items() if k != "definition"}
            and cen["skipped_by_name_never_opened"] == sk)
    tr = training_stats()
    oor = oor_property_checks(rows, chains, tr)
    chk("OOR property checks (outside T2's training statuses, magnitudes, names, key order)", all(oor.values()),
        "; ".join(k for k, v in oor.items() if not v))
    rc = rule_consistency_table(rows)
    chk("rule-consistency check: gold == before on every no-op row, gold == rule_gold != before on every apply row",
        all(v["rows_consistent"] == v["n"] for v in rc.values()))
    chk("SOURCE record: no model call, no torch, seed and builder identity as fixed here",
        rec["model_calls"] == 0 and not rec["torch_imported"] and rec["seed"] == C.SEED
        and rec["builder_sha256"] == L1.sha256_file(Path(__file__)) and rec["strata"] == list(C.STRATA))
    same_common = rec["common_sha256"] == L1.sha256_file(ROOT / "scripts" / "p3_common.py")
    if not quiet:
        print(f"  [info] p3_common.py {'unchanged since the build' if same_common else 'CHANGED since the build'} "
              "(the frame, gold and stratum re-derivations above bind either way)", flush=True)
    return res


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="re-check the written bank; write nothing")
    ap.add_argument("--no-historical", action="store_true", help="with --verify: skip the records-tree re-harvest and census")
    ap.add_argument("--out-dir", default=None, help="rehearsal only: write/verify bank_p3_oor.jsonl + SOURCE there, not in data/")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    bank_p, src_p = C.BANK, C.SOURCE
    if a.out_dir:
        d = Path(a.out_dir).resolve()
        if d == (ROOT / "data").resolve():
            raise SystemExit("--out-dir is for rehearsals outside data/")
        d.mkdir(parents=True, exist_ok=True)
        bank_p, src_p = d / C.BANK.name, d / C.SOURCE.name
    if a.verify:
        res = verify(bank_p, src_p, historical=not a.no_historical)
        print(f"p3 bank verify: {sum(ok for _, ok, _ in res)}/{len(res)} checks pass", flush=True)
        if "torch" in sys.modules:
            raise SystemExit("torch was imported")
        return 0 if all(ok for _, ok, _ in res) else 1
    if bank_p.exists() or src_p.exists():
        raise SystemExit(f"refusing to overwrite {bank_p} / {src_p} (create-only)")
    lines: list = []
    t0 = time.time()
    L1.log(lines, f"PREREG-3 OOR bank v2: seed {C.SEED}, {C.N_PER_STRATUM} bases, strata {list(C.STRATA)}, chains "
                  f"{dict(C.N_CHAIN_PARENTS)} x {C.K_CANDIDATES}; no model call")
    rows, chains, info = build(lines)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows + chains)
    fsha = sha256_bytes(text.encode("utf-8"))
    L1.log(lines, f"bank: {len(rows)} one-step rows + {len(chains)} chains, sha256 {fsha}; {time.time() - t0:.0f}s")
    src = info["src"]
    tr = training_stats()
    oor = oor_property_checks(rows, chains, tr)
    if not all(oor.values()):
        raise SystemExit(f"OOR property check failed: {oor}")
    rc = rule_consistency_table(rows)
    if not all(v["rows_consistent"] == v["n"] for v in rc.values()):
        raise SystemExit("the bank is not consistent with the rule on every row")
    t1 = time.time()
    cen = status_census(bank_pay=bank_payloads(rows, chains), bank_ops=bank_opkeys(rows, chains))
    if cen["skipped_by_name_never_opened"] != src["historical"]["skipped_by_name_never_opened"]:
        raise SystemExit("census and harvest skipped different reserved files")
    mer = cen["multi_event_rollouts"]
    if mer["payload_overlap"] or mer["operative_overlap"]:
        raise SystemExit(f"overlap with a multi-event prompt rollout: {mer}")
    L1.log(lines, f"status census: {cen['files_scanned']} files ({time.time() - t1:.0f}s); ledger context {cen['ledger_context']}; "
                  f"multi-event rollouts {mer}")
    rec = {
        "record": "P3_OOR_BANK", "schema": 2, "prereg": "PREREG-3 (PREREG_3_DRAFT.md)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_calls": 0, "torch_imported": "torch" in sys.modules,
        "builder": "scripts/p3_make_bank.py", "builder_sha256": L1.sha256_file(Path(__file__)),
        "common_sha256": L1.sha256_file(ROOT / "scripts" / "p3_common.py"),
        "harvest_module": "scripts/l1_make_ood.py", "harvest_module_sha256": L1.sha256_file(ROOT / "scripts" / "l1_make_ood.py"),
        "code_sha256": info["code"], "recorded_marginals_sha256": env.RECORDED_SHA, "seed": C.SEED,
        "supersedes": SUPERSEDES,
        "n_per_stratum": C.N_PER_STRATUM, "strata": list(C.STRATA),
        "strata_definitions": {s: {"group": v[0], "gold_rule": v[1], "status": list(v[2]) if isinstance(v[2], tuple) else v[2],
                                   "description": v[3]} for s, v in C.STRATA.items()},
        "design": ("100 bases (env.gen_start balances; env.gen_event event whose posted version changes the state; kind quota "
                   f"{C.KIND_QUOTA}); every one-step stratum is the same 100 bases with exactly one factor changed "
                   "(status string, balances, amount, account names, number of accounts, key order); row id "
                   "p3-<stratum>-<base index>; the twin of every row is the T_posted row of its base"),
        "chains": {"strata": dict(C.CHAIN_DESC), "parents": dict(C.N_CHAIN_PARENTS), "candidates_per_parent": C.K_CANDIDATES,
                   "horizon": C.H},
        "frame": ("Titan's one-step frame: rcwmt.env.one_step_messages(ledger_replay, before, event) (SYSTEM_ONE, compact "
                  "JSON, sorted keys); F_keyorder rows and CH_X reordered steps: the same system text and compact JSON "
                  "with the recorded key order (p3_common.user_json)"),
        "rule_text": C.RULE_TEXT,
        "gold": ("rcwmt.env.step (any status other than the exact string 'posted' is a no-op), asserted equal to "
                 "p3_common.rule_gold (the rule text implemented independently) and to the validated FDS oracle "
                 "(env.cross_check) on every row and chain transition"),
        "rule_consistency": rc,
        "rule_consistency_note": ("a consistency check of the builder, true by construction; the argument that the rule "
                                  "TEXT implies each stratum's gold is the per-stratum prose in PREREG_3_DRAFT.md §2.3"),
        "scoring": ("correct iff p3_common.parse_ledger(text, accounts of the row) == gold (order-insensitive); A-D rows "
                    "use rcwmt.env.parse_state(lenient=True) itself"),
        "exclusion": ("no (before, event) payload and no FDS operative content of any row or chain transition equals "
                      "one of any source below; payloads unique within the bank except F_keyorder == its twin (only the "
                      "prompt's key order differs); an operative content repeats only within one base or one chain parent"),
        "exclusion_sources": {k: {"payloads": len(v["payloads"]), "operative_contents": len(v["ops"]),
                                  **({"sha256": v["sha256"]} if v.get("sha256") else {}),
                                  **({"path": v["path"]} if v.get("path") else {}),
                                  **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"],
                                      "harvest": ("scripts/l1_make_ood.harvest_historical: FDS generate.harvest_exclusions "
                                                  "line for line, except files named HSKILL_CONFIRM_V2* are skipped by "
                                                  "name before any open; rcwmt.bank.load_exclusions() is NOT called")}
                                     if k == "historical" else {})}
                              for k, v in src.items()},
        "exclusion_union": info["excl_union"],
        "overlap_counts": info["overlaps"],
        "before_state_overlap_descriptive": before_overlap(rows, chains, src),
        "union_status_counts": union_status_counts(src),
        "union_status_counts_note": "single-step {before, event} payloads only; multi-event prompts are not expanded",
        "status_census": cen,
        "training_stats": tr, "oor_property_checks": oor,
        "draws": info["draws"], "rejected": {s: dict(c) for s, c in info["rejected"].items()},
        "describe": describe(rows, chains),
        "provenance": ("Fresh. Built 2026-10-09 for PREREG-3 before any model saw it; no model, tokenizer or model "
                       "output was used. No runs/t2/*/gen_* file was opened."),
        "use": "PREREG-3 only (rule or trigger); spent by that use.",
        "file": str(bank_p), "file_sha256": fsha, "onestep_rows": len(rows), "chains": len(chains), "build_log": lines}
    for path, payload in ((bank_p, text.encode("utf-8")), (src_p, (json.dumps(rec, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "xb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    print(json.dumps({"file_sha256": fsha, "rows": len(rows), "chains": len(chains),
                      "per_stratum": dict(Counter(r["stratum"] for r in rows)), "overlap_counts": info["overlaps"],
                      "union_status_counts": rec["union_status_counts"], "census_ledger_context": cen["ledger_context"]},
                     indent=1), flush=True)
    if "torch" in sys.modules:
        raise SystemExit("torch was imported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
