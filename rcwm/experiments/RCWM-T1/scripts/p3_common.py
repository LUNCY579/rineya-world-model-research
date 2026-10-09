"""PREREG-3 shared definitions: the sealed out-of-range (OOR) bank design, Titan's one-step frame with optional key
orders, the ledger parser for any account naming, the explicit status rule, and the scorer. Torch-free.

Question (PREREG_3_DRAFT.md §1). Did the T2 ledger adapter learn the RULE its frame states ("If status is not posted,
leave all balances unchanged"; arithmetic at any magnitude, account naming and key order), a semantic cue (status words
that MEAN "not applied" -> no-op), a substring cue ("contains posted" -> apply) or a TRIGGER (the literal token 'void';
in-range arithmetic)? T2's training rows covered statuses posted/void only, amounts -9..12, |before| <= 98, accounts
A-D, sorted JSON keys (data/t2_train.jsonl; recomputed by p3_make_bank.py).

Gold. rcwmt.env.step treats every status other than the exact string 'posted' as a no-op, which is what the frame's
rule text says; it is used for every gold, checked against the validated FDS oracle (env.cross_check) and against
rule_gold() below (an explicit, independent implementation of the stated rule) on every row and chain transition.

Design (bank v2). 100 BASES (an env.gen_start balance set and an env.gen_event event whose posted version changes the
state; kind quota 45/33/22). EVERY one-step stratum is derived from the same 100 bases by changing exactly one factor:
the status string (10 status strata), the balances (M_bal), the amount (M_amt), the account names (A_renamed), the
number of accounts (A_many) or the JSON key order (F_keyorder). So each row has a posted in-distribution twin
(T_posted, same base) and every out-of-range effect is a paired, kind-matched contrast.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402

LEDGER = "ledger_replay"
BANK = ROOT / "data" / "bank_p3_oor.jsonl"
SOURCE = ROOT / "data" / "bank_p3_oor.SOURCE.json"
DOCS = ROOT / "docs" / "prereg3"
SEED = 20261103                    # not used by any earlier RCWM-T1 builder, analysis or bootstrap
N_PER_STRATUM = 100
N_CHAIN_PARENTS = {"CH_S": 10, "CH_X": 10}
K_CANDIDATES = 4
H = 4

# ---------------------------------------------------------------------------------------------- vocabularies
# Every non-posted status below is absent from T2's and Titan's ledger training rows (posted / void only).
NOT_APPLIED = ("pending", "draft", "cancelled", "reversed", "held")   # unseen words whose MEANING is "not applied"
NEUTRAL = ("approved", "active", "ok", "dax", "wug")                 # unseen words WITHOUT a not-applied meaning:
#   positive (approved, ok), neutral (active) and nonce (dax, wug). None is a case variant or a synonym of 'posted'
#   (booked, recorded, settled, cleared, completed, confirmed are left out for the same reason as Posted / POSTED).
UNSEEN_WORDS = NOT_APPLIED + NEUTRAL
NEAR_VOID = ("Void", "VOID", "voided")                            # near-miss spellings of the trained 'void'
POSTED_LIKE = ("unposted", "not_posted", "not posted")            # contain 'posted' but are not 'posted'
NONPOSTED_POOL = UNSEEN_WORDS + NEAR_VOID + POSTED_LIKE
LETTER_NAMES = ("E", "F", "G", "H")
WORD_NAMES = ("bank", "card", "cash", "fees", "fund", "loan", "rent", "wage")
MANY_NAMES = ("A", "B", "C", "D", "E", "F", "G", "H")
BIG_DIGITS = (4, 5)                                               # M_bal: every balance has 4 or 5 digits
AMT_BANDS = ((13, 99), (100, 500))                                # M_amt: |amount| 13..99 (even bases) / 100..500 (odd)

# ---------------------------------------------------------------------------------------------- strata
# name -> (group, gold rule, status or status cycle (by base index), description). gold rule 'noop' = gold equals before.
STRATA = OrderedDict([
    ("T_posted", ("status", "apply", "posted",
                  "the posted twin of every base (status 'posted'); in-distribution posted control")),
    ("C_void", ("status", "noop", "void", "the trained status 'void' on the same bases; in-distribution void control")),
    ("U_pending", ("status", "noop", "pending", "unseen status 'pending' (meaning: not applied) on the same bases")),
    ("U_draft", ("status", "noop", "draft", "unseen status 'draft' (meaning: not applied) on the same bases")),
    ("U_cancelled", ("status", "noop", "cancelled", "unseen status 'cancelled' (meaning: not applied) on the same bases")),
    ("U_reversed", ("status", "noop", "reversed", "unseen status 'reversed' (net effect: none) on the same bases")),
    ("U_held", ("status", "noop", "held", "unseen status 'held' (meaning: not executed) on the same bases")),
    ("U_neutral", ("status", "noop", NEUTRAL,
                   "unseen statuses WITHOUT a not-applied meaning: approved / active / ok / dax / wug (base index mod 5)")),
    ("N_voidcase", ("status", "noop", NEAR_VOID, "near-miss spellings of 'void': Void / VOID / voided (base index mod 3)")),
    ("N_postedlike", ("status", "noop", POSTED_LIKE,
                      "look-alikes containing 'posted': unposted / not_posted / 'not posted' (base index mod 3)")),
    ("M_bal", ("magnitude", "apply", "posted",
               "the base's event (posted) on new balances, every balance with 4 or 5 digits (sign x uniform in the band)")),
    ("M_amt", ("magnitude", "apply", "posted",
               "the base's balances and event (posted) with the amount replaced: |amount| 13..99 (even bases) or "
               "100..500 (odd bases), random sign")),
    ("A_renamed", ("naming", "apply", "posted",
                   "the base (posted) with A-D relabelled: E-H (even bases) or 4 of 8 lower-case words in sorted order "
                   "(odd bases)")),
    ("A_many", ("naming", "apply", "posted",
                "the base (posted) plus extra accounts E.. (5 / 6 / 7 / 8 accounts by base index mod 4; extra balances "
                "in range); the event is the base's (touches A-D)")),
    ("F_keyorder", ("format", "apply", "posted",
                    "the base (posted) with JSON keys reordered: 'event' first with inner keys sorted (even bases), or "
                    "'event' first with event and balance keys permuted, never sorted (odd bases)")),
])
STATUS_STRATA = tuple(s for s, v in STRATA.items() if v[0] == "status")
NOT_APPLIED_STRATA = ("U_pending", "U_draft", "U_cancelled", "U_reversed", "U_held")
U_STRATA = NOT_APPLIED_STRATA + ("U_neutral",)                    # unseen status WORDS
UNSEEN_STATUS_STRATA = U_STRATA + ("N_voidcase", "N_postedlike")  # every unseen-status stratum: the RULE conditions
TRIGGER_STRATA = U_STRATA + ("N_postedlike",)                     # unseen statuses with no form of 'void'
GEN_STRATA = ("M_bal", "M_amt", "A_renamed", "A_many", "F_keyorder")
# Verdict groups over cells = (strata, kind class) per seed. 'at' = add + transfer rows (67 per stratum), 'sc' = scale
# rows (33 per stratum): out-of-range scale rows test multi-digit multiplication, so magnitude is judged separately for
# the two kind classes. The scale cell pools the 33 M_bal and 33 M_amt scale rows (66 per seed): six 33-row cells would
# all meet the floor with probability only about 0.21 at a true accuracy of 0.90 (power table in the dry run).
GEN_GROUPS = OrderedDict([
    ("magnitude_at", ((("M_bal",), "at"), (("M_amt",), "at"))),
    ("magnitude_scale", ((("M_bal", "M_amt"), "sc"),)),
    ("naming", ((("A_renamed",), "all"), (("A_many",), "all"))),
    ("format", ((("F_keyorder",), "all"),)),
])
# Descriptive single-stratum cells (in-distribution vs out-of-range contrasts, breakdowns).
DESCRIPTIVE_CELLS = ((("M_bal",), "at"), (("M_bal",), "sc"), (("M_amt",), "at"), (("M_amt",), "sc"),
                     (("A_renamed",), "all"), (("A_many",), "all"), (("F_keyorder",), "all"))


def cell_key(strata, part: str) -> str:
    return "+".join(strata) + "|" + part
GROUP_LABEL = {"magnitude_at": "MAGNITUDE (add / transfer)", "magnitude_scale": "MAGNITUDE (scale: multi-digit multiplication)",
               "naming": "NAMING", "format": "FORMAT"}
CHAIN_STRATA = ("CH_S", "CH_X")
CHAIN_DESC = {
    "CH_S": "accounts A-D, balances and amounts in range, sorted keys; each event posted (p .50), an unseen status "
            "word (p .35, 10 equally: 5 not-applied, 5 neutral / nonce) or a near-miss / look-alike (p .15, 6 equally); "
            ">= 1 posted and >= 1 non-posted event",
    "CH_X": "accounts A-D, start in range; each event posted (p .60) or non-posted (p .40; the 16 non-posted strings "
            "equally); |amount| 13..500 on each add/transfer event w.p. .40 (scale keeps a trained amount, so products "
            "do not compound); each step's keys reordered w.p. .50; >= 1 posted, >= 1 non-posted, >= 1 posted "
            "OOR-amount event and >= 1 reordered step",
}
# The bases draw their event kinds to this quota: the recorded marginal (transfer 58, scale 42, add 28 of 128) rounded
# to 100. Fixed by design so that no stratum's difficulty depends on a chance kind mix; every stratum inherits it.
KIND_QUOTA = {"transfer": 45, "scale": 33, "add": 22}
# The frame's rule text, quoted for the rule-consistency table (rcwmt.env.SYSTEM_ONE['ledger_replay']).
RULE_TEXT = env.SYSTEM_ONE[LEDGER]
RULE_CLAUSES = {
    "noop": "\"If status is not posted, leave all balances unchanged.\"",
    "apply": "status is 'posted', so the arithmetic clauses apply (\"add: add amount to account. scale: multiply account "
             "by amount. transfer: subtract amount from account and add it to other, simultaneously.\")",
}


def kind_class(kind: str) -> str:
    return "sc" if kind == "scale" else "at"


def sub_of(row: dict) -> str:
    """The registered sub-variant of a one-step row: the status string for status strata with a status cycle, the
    F_keyorder variant, the naming, the number of accounts, the amount band; else 'all'."""
    s = row["stratum"]
    if isinstance(STRATA[s][2], tuple):
        return row["event"]["status"]
    m = row.get("meta") or {}
    if s == "F_keyorder":
        return m["key_variant"]
    if s == "A_renamed":
        return m["naming"]
    if s == "A_many":
        return str(m["n_accounts"])
    if s == "M_amt":
        return m["amount_band"]
    return "all"


# ---------------------------------------------------------------------------------------------- the stated rule
def rule_gold(before: dict, ev: dict) -> dict:
    """The frame's rule text implemented explicitly and independently of env.step: a status that is not exactly the
    string 'posted' leaves every balance unchanged; otherwise add / scale / transfer as stated. Any account names."""
    out = dict(before)
    if ev["status"] != "posted":
        return out
    a, amt = ev["account"], ev["amount"]
    if ev["kind"] == "add":
        out[a] = before[a] + amt
    elif ev["kind"] == "scale":
        out[a] = before[a] * amt
    elif ev["kind"] == "transfer":
        out[a] = before[a] - amt
        out[ev["other"]] = before[ev["other"]] + amt
    else:
        raise ValueError(ev["kind"])
    return out


def gold_of(before: dict, ev: dict) -> dict:
    """env.step (the registered semantics), asserted equal to rule_gold and to the FDS oracle."""
    g = env.step(LEDGER, before, ev)
    if g != rule_gold(before, ev):
        raise AssertionError(f"env.step != the stated rule for {before} {ev}")
    env.cross_check(LEDGER, before, ev)
    return g


# ---------------------------------------------------------------------------------------------- frame
def user_json(before: dict, ev: dict, ko: dict | None = None) -> str:
    """Titan's one-step user turn. ko None: exactly rcwmt.env.one_step_messages (compact, sorted keys). ko: the same
    compact JSON with key orders ko['top'] (a permutation of before/event), ko['event'] and ko['before']."""
    if ko is None:
        return env.dumps({"before": before, "event": ev})
    if sorted(ko["before"]) != sorted(before) or sorted(ko["event"]) != sorted(ev) or sorted(ko["top"]) != ["before", "event"]:
        raise ValueError("key order does not match the payload")
    parts = {"before": OrderedDict((k, before[k]) for k in ko["before"]),
             "event": OrderedDict((k, ev[k]) for k in ko["event"])}
    return json.dumps(OrderedDict((k, parts[k]) for k in ko["top"]), separators=(",", ":"))


def messages(before: dict, ev: dict, ko: dict | None = None) -> list[dict]:
    if ko is None:
        return env.one_step_messages(LEDGER, before, ev)
    return [{"role": "system", "content": env.SYSTEM_ONE[LEDGER]}, {"role": "user", "content": user_json(before, ev, ko)}]


def step_messages(chain: dict, state: dict, t: int) -> list[dict]:
    """Chain step t in the one-step frame, fed the model's own previous state (chain_one self-rollout)."""
    ko = (chain.get("key_orders") or [None] * len(chain["events"]))[t]
    return messages(state, chain["events"][t], ko)


# ---------------------------------------------------------------------------------------------- parsing and scoring
def parse_general(text: str, keys) -> dict | None:
    """rcwmt.env.parse_state(ledger, text, lenient=True) with the account set taken from the row instead of A-D.
    Same steps: strip, drop one trailing '.', json.loads; else rewrite `"K": a op b = c` to `"K": c`; else the last
    complete "after" value; unwrap {"after": ...}; exact key set; int (not bool) values."""
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        t2 = env._EQ_RHS.sub(r'"\1": \2', t)
        try:
            v = json.loads(t2)
        except (json.JSONDecodeError, ValueError):
            v = env._last_after_value(t2)
            if v is None:
                return None
    if isinstance(v, dict) and "after" in v:
        v = v["after"]
    keys = list(keys)
    if not isinstance(v, dict) or set(v) != set(keys):
        return None
    if not all(isinstance(v[k], int) and not isinstance(v[k], bool) for k in keys):
        return None
    return {k: v[k] for k in keys}


def parse_ledger(text: str, keys) -> dict | None:
    """The registered parser: accounts A-D -> rcwmt.env.parse_state itself (PREREG-1/2's parser, unchanged);
    any other account set -> parse_general (asserted equal to env.parse_state on A-D by the scorer self-test)."""
    if set(keys) == set(env.ACCOUNTS):
        return env.parse_state(LEDGER, text, lenient=True)
    return parse_general(text, keys)


def score_onestep(row: dict, text: str | None) -> dict:
    """correct iff the parsed answer equals the gold balances (order-insensitive). Error classes: for no-op rows
    'applied_as_posted' (the answer equals the posted twin's gold: the event was applied), 'other_change', 'unparsed';
    for apply rows 'copied' (answer == before), 'wrong_value', 'unparsed'."""
    st = None if text is None else parse_ledger(text, list(row["before"]))
    gold = json.loads(row["gold"])
    if st is None:
        cls = "unparsed"
    elif st == gold:
        cls = "correct"
    elif row["gold_rule"] == "noop":
        cls = "applied_as_posted" if st == json.loads(row["posted_gold"]) else "other_change"
    else:
        cls = "copied" if st == row["before"] else "wrong_value"
    return {"correct": cls == "correct", "cls": cls}


def chain_keys(chain: dict) -> list:
    return list(chain["s0"])


# ---------------------------------------------------------------------------------------------- bank I/O
def load_bank(path: Path | None = None) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in Path(path or BANK).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["kind"] == "onestep"], [r for r in rows if r["kind"] == "chain"]


def chain_bank_dict(chains: list[dict]) -> dict:
    """The chain rows as an rcwmt bank dict (data.candidate_table / decisions.g_only input; split 'test')."""
    par = OrderedDict()
    for c in chains:
        p = par.setdefault(c["parent_id"], {"family": LEDGER, "parent_id": c["parent_id"], "split": "test", "s0": c["s0"],
                                            "stratum": c["stratum"], "candidates": []})
        p["candidates"].append({"cand_id": c["cand_id"], "events": c["events"]})
    return {"parents": list(par.values())}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, c - h), min(1.0, c + h)
