"""T2b-ledger kit: shared definitions (torch-free). A surface-diversified ledger rule adapter on the T2 recipe, its dev
screen (scripts/t2bl_run.py, scripts/t2bl_gate.py) and the sealed confirmation PREREG-5 (scripts/t2bl_confirm_*.py,
PREREG_5_DRAFT.md). Design: docs/t2b/ledger/DESIGN.md.

Why (REPORT 3c, 3d; DECISIONS D-34/D-35/D-45). T2 s0 learned the literal token 'void' as the ledger no-op cue (PREREG-3
STATUS: TRIGGER; unseen statuses 0.01-0.18), fails out of range on magnitudes, reordered keys and word account names,
and plans at 0.706 on its training surface but 5/180 on PREREG-LB's held-out combined surface. T2b-ledger trains the
same recipe on a fresh oracle-labelled ledger one-step set whose SURFACE varies (status words, magnitudes, input key
orders, account names and counts, rule-text paraphrases), with explicit TRAIN vs HELD-OUT partitions fixed here, so the
dev screen and the confirmation can test on surfaces no training prompt used. Targets are unchanged: env.dumps(after),
the compact sorted-key balances object T2 trained on.

Frame. Titan's one-step ledger frame family: a system text (one of five paraphrases of the frame's rule; paraphrase 0 is
rcwmt.env.SYSTEM_ONE['ledger_replay'] verbatim) and a compact JSON user turn {"before": {...}, "event": {...}} whose
three objects are serialised in the row's recorded key orders. On the canonical surface (paraphrase 0, sorted keys,
accounts A-D, posted / void, |b| <= 98, trained amounts) messages() == rcwmt.env.one_step_messages byte for byte.

Gold. p3_common.gold_of: rcwmt.env.step (every status other than the exact string 'posted' is a no-op), asserted equal
to the explicit rule implementation p3_common.rule_gold and to the validated FDS oracle on every row.

Seeds (grep over the RCWM, BWM and complementary trees on 2026-10-10: unused; RCX owns 202612xx, D-39):
  20261161 training set, 20261162 dev bank, 20261163 held-out key orders, 20261164 tests only (never a registered set).
The PREREG-5 confirmation bank and planning draw are minted at its freeze from a commit-reveal seed (BWM's salt).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rcwmt import env  # noqa: E402  (torch-free)
import p3_common as P3  # noqa: E402  (PREREG-3's frame helpers, parser, scorer and gold; torch-free, read-only)

LEDGER = "ledger_replay"
DATA = ROOT / "data"
DOCS = ROOT / "docs" / "t2b" / "ledger"
OUT_ROOT = ROOT / "runs" / "t2bl"
TOKENIZER_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                     r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
SEED_TRAIN = 20261161
SEED_DEV = 20261162
ORDER_SEED = 20261163
TEST_SEED = 20261164
RESERVED_PREFIX = "HSKILL_CONFIRM_V2"            # never opened or hashed (DECISIONS I-28)

TRAIN = DATA / "t2bl_train.jsonl"
TRAIN_SOURCE = DATA / "t2bl_train.SOURCE.json"
DEV = DATA / "t2bl_dev.jsonl"
DEV_SOURCE = DATA / "t2bl_dev.SOURCE.json"

CANON_NAMES = env.ACCOUNTS                       # ("A", "B", "C", "D")
EVENT_KEYS = ("account", "amount", "kind", "other", "status", "tick")   # sorted (canonical)
TOP_KEYS = ("before", "event")
KINDS = ("add", "scale", "transfer")
KIND_QUOTA = {"transfer": 0.45, "scale": 0.33, "add": 0.22}           # PREREG-3's quota (recorded marginal 58/42/28)

# ================================================================================ status vocabularies (the partition)
# Every non-posted string below is a no-op under the frame's rule ("If status is not posted, leave all balances
# unchanged"). Kinds: words that MEAN "not applied"; neutral / positive words without that meaning (the rule and
# world-knowledge semantics come apart: an 'accepted' event would naturally be applied); nonce words; case and spelling
# variants of the trained 'void'; look-alikes that contain 'posted' but are not the string 'posted'.
# Case variants of 'void' are TRAINING items only ('VOId', 'vOid'; PREREG-3's 'Void' / 'VOID' stay reserved): a held-out
# case variant would lower-case to the trained 'void', so every HELD-OUT string differs from every trained string even
# after lower-casing (t2bl_audit checks it).
WORD_KINDS = ("not_applied", "neutral", "nonce", "void_variant", "posted_like")
TRAIN_WORDS = OrderedDict([
    ("not_applied", ("rejected", "declined", "failed", "blocked", "suspended", "withdrawn", "aborted", "revoked",
                     "annulled", "deferred", "queued", "paused")),
    ("neutral", ("open", "new", "flagged", "reviewed", "manual", "standard", "accepted", "checked", "pinned",
                 "tagged")),
    ("nonce", ("blick", "zorp", "kiki", "mipa", "tove", "frell", "glorp", "nim")),
    ("void_variant", ("voids", "voiding", "void_", "voidz", "VOId", "vOid")),
    ("posted_like", ("postedx", "xposted", "posted_no", "zposted", "posted2", "qposted")),
])
VOID_CASE_VARIANTS = tuple(w for w in TRAIN_WORDS["void_variant"] if w.lower() == "void")
HELDOUT_WORDS = OrderedDict([
    ("not_applied", ("rescinded", "refused", "stalled", "nullified")),
    ("neutral", ("verified", "valid", "signed", "routine")),
    ("nonce", ("toma", "zib", "fep", "wux")),
    ("void_variant", ("voidd", "re_void", "voidy", "xvoid")),
    ("posted_like", ("postedq", "yposted", "posted_zero", "posted9")),
])
P3_WORDS = tuple(P3.NONPOSTED_POOL)               # PREREG-3's 16 strings: reserved, never trained (secondary stratum)
# No defensible gold (the literal and the natural reading of the rule disagree; PREREG-3 Appendix B Q2): never used.
EXCLUDED_STATUS = ("Posted", "POSTED", "posted.", " posted", "posted ", "booked", "recorded", "settled", "cleared",
                   "completed", "confirmed", "logged", "entered", "applied", "executed", "processed", "done")
TRAIN_WORD_LIST = tuple(w for ws in TRAIN_WORDS.values() for w in ws)
HELDOUT_WORD_LIST = tuple(w for ws in HELDOUT_WORDS.values() for w in ws)
WORD_KIND_OF = {w: k for k, ws in list(TRAIN_WORDS.items()) + list(HELDOUT_WORDS.items()) for w in ws}
TRAIN_STATUSES = ("posted", "void") + TRAIN_WORD_LIST

# ================================================================================ account names and counts
TRAIN_LETTERS = ("I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W")   # not E-H (held out), O, X-Z
HELDOUT_LETTERS = tuple(P3.LETTER_NAMES)          # E F G H (PREREG-3's renamed letters)
TRAIN_NAME_WORDS = ("bills", "bonus", "cafe", "coins", "dues", "farm", "gifts", "house", "media", "petty", "pool",
                    "roof", "salary", "shop", "tips", "trust")
HELDOUT_NAME_WORDS = tuple(P3.WORD_NAMES) + ("barn", "dock", "lamp", "mill", "nest", "pier", "silo", "yard")
LB_NAMES = ("bank", "card", "cash", "fees")       # PREREG-LB's heldout_combined names (lbeam_f.WORD_NAMES4)
TRAIN_COUNTS = (4, 5, 6, 7)
HELDOUT_COUNTS = (8,)

# ================================================================================ magnitudes
# T2 trained balances |b| <= 98 (the recorded pool) and the recorded amount marginal -9..12. T2b-ledger widens both by
# two orders of magnitude, drawn by digit-count class so short numbers keep a large share (T2's range stays a subset,
# so its canonical rows are in the training distribution); the held-out band is the NEXT digit count. Scale factors
# stay the recorded marginal everywhere (multi-digit x multi-digit multiplication is a capacity question PREREG-3
# already separated; a larger factor would confound the magnitude test with multiplication).
BAL_CLASSES = {"t2": None, "b3": (100, 999), "b4": (1000, 9999)}       # training classes ('t2' = the recorded pool)
AMT_CLASSES = {"t2": None, "a2": (13, 99), "a3": (100, 999)}           # add / transfer only
HELDOUT_BAL = (10000, 99999)                      # 5-digit balances
HELDOUT_AMT = (1000, 9999)                        # 4-digit amounts (add / transfer)
TICKS = (1, 99)

# ================================================================================ rule-text paraphrases
PARAPHRASES = (
    env.SYSTEM_ONE[LEDGER],
    ("You are given account balances and one event. The event takes effect only if its status is posted; for any "
     "other status, every balance stays the same. add: increase account by amount. scale: multiply account by amount. "
     "transfer: take amount from account and give it to other at the same time. Reply with only the full set of "
     "resulting balances as a JSON object."),
    ("Update the balances using the event. An event whose status is not posted changes nothing. For add, add amount to "
     "account; for scale, multiply account by amount; for transfer, subtract amount from account and add it to other "
     "in one step. Output only the complete resulting balances as a JSON object."),
    ("Process one ledger event against the given balances. Only a posted event is applied; any other status leaves all "
     "balances as they were. add adds amount to account. scale multiplies account by amount. transfer moves amount "
     "from account to other simultaneously. Return just the complete balances after the event, as a JSON object."),
    ("Here are some account balances and a single event. Unless the event's status is posted, return the balances "
     "unchanged. With add, amount is added to account. With scale, account is multiplied by amount. With transfer, "
     "amount is subtracted from account and added to other at once. Answer with nothing but a JSON object holding "
     "every resulting balance."),
)
TRAIN_PARA = (0, 1, 2, 3)
HELDOUT_PARA = 4
HELDOUT_PARA_FRAGMENTS = ("Here are some account balances", "Unless the event's status is posted",
                          "Answer with nothing but a JSON object")

# ================================================================================ key orders
P_CANON_ORDER = 0.25                               # the canonical (sorted) order on about 1/4 of training rows


def _held_out_perms(n: int, rng: random.Random) -> list:
    """The held-out position permutations of an n-key object: the full reversal plus 2 fixed random permutations
    (neither the identity nor the reversal), drawn from rng in order."""
    rev = tuple(range(n - 1, -1, -1))
    out = [rev]
    while len(out) < 3:
        p = list(range(n))
        rng.shuffle(p)
        p = tuple(p)
        if p != tuple(range(n)) and p not in out:
            out.append(p)
    return out


def derive_heldout_orders() -> dict:
    """Regenerated from ORDER_SEED (the audit and the tests re-derive it independently): top {before, event} -> the
    reversed order only (2 keys); event (6 keys) and before (n accounts, n = 4..8, as position permutations of the
    sorted names) -> reversed + 2 fixed random permutations each."""
    rng = random.Random(ORDER_SEED)
    out = {"top": [(1, 0)], "event": _held_out_perms(len(EVENT_KEYS), rng)}
    for n in sorted(set(TRAIN_COUNTS + HELDOUT_COUNTS)):
        out[f"before{n}"] = _held_out_perms(n, rng)
    return out


HELDOUT_PERMS = derive_heldout_orders()
LB_ORDER = {"top": (1, 0), "event": tuple(range(len(EVENT_KEYS) - 1, -1, -1))}   # PREREG-LB keys_reversed


def perm_key(n: int) -> str:
    return f"before{n}"


_TRAIN_PERMS: dict = {}


def train_perms(obj: str, n: int | None = None) -> list:
    """Every permutation of the object's positions that is NOT held out (identity included), sorted."""
    k = obj if obj in ("top", "event") else perm_key(n)
    if k not in _TRAIN_PERMS:
        size = {"top": 2, "event": len(EVENT_KEYS)}.get(k) or int(k[6:])
        ho = set(HELDOUT_PERMS[k])
        _TRAIN_PERMS[k] = [p for p in itertools.permutations(range(size)) if p not in ho]
    return _TRAIN_PERMS[k]


def draw_train_perm(rng: random.Random, obj: str, n: int | None = None, allow_canon: bool = True) -> tuple:
    size = {"top": 2, "event": len(EVENT_KEYS)}.get(obj) or n
    ident = tuple(range(size))
    if obj == "top":
        return ident                               # the 2-key top object is never augmented (its only other order is held out)
    if allow_canon and rng.random() < P_CANON_ORDER:
        return ident
    pool = [p for p in train_perms(obj, n) if p != ident]
    return pool[rng.randrange(len(pool))]


def apply_perm(keys_sorted, perm) -> list:
    return [keys_sorted[i] for i in perm]


def perm_of(order, keys_sorted) -> tuple:
    return tuple(list(keys_sorted).index(k) for k in order)


def order_class(obj: str, perm: tuple, n: int | None = None) -> str:
    """'canonical' (identity), 'heldout' (a held-out permutation) or 'train'."""
    k = obj if obj in ("top", "event") else perm_key(n)
    if perm == tuple(range(len(perm))):
        return "canonical"
    return "heldout" if tuple(perm) in set(HELDOUT_PERMS[k]) else "train"


# ================================================================================ surfaces and rendering
def canonical_surface(before: dict) -> dict:
    return {"para": 0, "top": list(TOP_KEYS), "event": list(EVENT_KEYS), "before": sorted(before)}


def user_json(before: dict, ev: dict, surface: dict | None) -> str:
    """Compact JSON of {before, event} with the surface's key orders; surface None = env.dumps (sorted)."""
    if surface is None:
        return env.dumps({"before": before, "event": ev})
    if sorted(surface["before"]) != sorted(before) or sorted(surface["event"]) != sorted(ev) or \
            sorted(surface["top"]) != list(TOP_KEYS):
        raise ValueError("surface key orders do not match the payload")
    parts = {"before": OrderedDict((k, before[k]) for k in surface["before"]),
             "event": OrderedDict((k, ev[k]) for k in surface["event"])}
    return json.dumps(OrderedDict((k, parts[k]) for k in surface["top"]), separators=(",", ":"))


def messages(before: dict, ev: dict, surface: dict | None = None) -> list:
    para = 0 if surface is None else int(surface["para"])
    return [{"role": "system", "content": PARAPHRASES[para]}, {"role": "user", "content": user_json(before, ev, surface)}]


def render_chat(msgs: list) -> str:
    """Qwen3 chat template with add_generation_prompt=True, enable_thinking=False (== rcwmt.capture.render; the tiny test
    checks it against tok.apply_chat_template, the builder against T2's stored ledger prompts)."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


_CHAT_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
                      r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n", re.S)


def split_chat(prompt: str) -> list:
    m = _CHAT_RE.fullmatch(prompt)
    if not m:
        raise ValueError("not a rendered system+user prompt")
    return [{"role": "system", "content": m.group(1)}, {"role": "user", "content": m.group(2)}]


def target_text(before: dict, ev: dict) -> str:
    return env.dumps(P3.gold_of(before, ev))


# ================================================================================ gold, parsing, scoring
gold_of = P3.gold_of
rule_gold = P3.rule_gold
parse_ledger = P3.parse_ledger


def score(row: dict, text) -> dict:
    """PREREG-3's scorer (p3_common.score_onestep): correct iff the parsed balances (the row's account set, any order)
    equal the gold. Error classes: no-op rows applied_as_posted / other_change / unparsed; apply rows copied /
    wrong_value / unparsed."""
    return P3.score_onestep(row, text)


def payload(before: dict, ev: dict) -> str:
    return env.dumps({"before": before, "event": ev})


def effective(before: dict, ev: dict) -> bool:
    """The posted version of the event changes the state."""
    return P3.gold_of(before, {**ev, "status": "posted"}) != before


# ================================================================================ heuristics (training audit)
def heuristic_states(before: dict, ev: dict) -> dict:
    """What each heuristic predicts for a row. Binding audit members: copy, apply_regardless (= ignore_status: apply as
    if posted), void_trigger (T2's learned cue: no-op iff 'void' occurs in the lower-cased status). Descriptive string
    cues: semantic (no-op iff the word means 'not applied'), contains_posted (apply iff 'posted' is a substring),
    startswith_p, len6 (apply iff a 6-character status), endswith_ed (apply iff it ends with 'ed'); exact / ci_posted
    are the rule itself (case-insensitive equality agrees with the rule on every string this kit uses)."""
    st = ev["status"]
    applied = P3.gold_of(before, {**ev, "status": "posted"})
    copy = dict(before)
    sem_noop = st != "posted" and (st == "void" or WORD_KIND_OF.get(st) in ("not_applied", "void_variant")
                                   or st in P3.NOT_APPLIED or st in P3.NEAR_VOID)
    pick = lambda apply: applied if apply else copy  # noqa: E731
    return {"copy": copy, "apply_regardless": applied, "void_trigger": pick("void" not in st.lower()),
            "semantic": pick(not sem_noop), "contains_posted": pick("posted" in st),
            "startswith_p": pick(st.startswith("p")), "len6": pick(len(st) == 6), "endswith_ed": pick(st.endswith("ed")),
            "ci_posted": pick(st.lower() == "posted"), "exact": pick(st == "posted")}


BINDING_HEURISTICS = {"copy": (0.48, 0.52), "apply_regardless": (0.48, 0.52), "void_trigger": (0.0, 0.70)}
STRING_HEURISTIC_MAX = 0.95                        # every non-rule string cue is wrong on >= 5% of training rows
RULE_EQUIVALENT = ("exact", "ci_posted")

# ================================================================================ strata (dev bank and PREREG-5 map)
# name -> (factor, gold rule, surface class, description). gold rule 'noop' = gold equals before.
STRATA = OrderedDict([
    ("ID_posted", ("id", "apply", "canonical", "the posted base, canonical surface (in-distribution control; T_posted)")),
    ("ID_void", ("id", "noop", "canonical", "the base with status 'void', canonical surface (in-distribution control)")),
    ("S_words", ("words_train", "noop", "canonical", "a TRAINING non-posted word (kind by base index mod 5), canonical "
                                                       "surface otherwise (in-surface status control)")),
    ("S_surface_posted", ("surface_train", "apply", "train", "posted, a training surface: paraphrase 1-3, training "
                          "names (letters / words, 4-7 accounts), training key orders, training magnitude bands")),
    ("S_surface_words", ("surface_train", "noop", "train", "the same training surface with a training non-posted word")),
    ("H_words", ("words", "noop", "canonical", "a HELD-OUT non-posted word (kind by base index mod 5; 4 words per kind), "
                                                "canonical surface otherwise")),
    ("H_order", ("order", "apply", "heldout_order", "posted, every object in a held-out key order: top reversed, event "
                                                     "held-out order (index mod 3), before held-out order")),
    ("H_names", ("names", "apply", "heldout_names", "posted, accounts renamed to held-out names: E-H (even bases) or 4 "
                                                     "held-out words, sorted (odd bases)")),
    ("H_count", ("count", "apply", "heldout_count", "posted, 8 accounts (held-out count): A-D plus 4 training letters "
                                                     "with in-range balances; the event is the base's")),
    ("H_para", ("para", "mixed", "heldout_para", "the held-out rule paraphrase 4; posted (even bases) or void (odd)")),
    ("H_bal", ("magnitude", "apply", "heldout_bal", "posted, every balance redrawn in the held-out 5-digit band")),
    ("H_amt", ("magnitude", "apply", "heldout_amt", "posted, the amount redrawn in the held-out 4-digit band (add / "
                                                     "transfer bases only)")),
    ("HS_words", ("combined", "noop", "heldout_combined", "the COMBINED held-out surface (paraphrase 4, held-out names, "
                  "held-out key orders) with a held-out non-posted word")),
    ("HS_posted", ("combined", "apply", "heldout_combined", "the same combined held-out surface, posted (the twins)")),
    ("LB_void", ("lb_surface", "noop", "lb_heldout_combined", "PREREG-LB's heldout_combined rendering (paraphrase 0, keys "
                 "reversed, accounts bank/card/cash/fees) with 'void' rendered 'cancelled'")),
    ("LB_posted", ("lb_surface", "apply", "lb_heldout_combined", "the same rendering, posted")),
    ("P3_words", ("p3", "noop", "canonical", "PREREG-3's 16 reserved non-posted strings (base index mod 16), canonical "
                                             "surface (dev bank only)")),
])
for _s, (_g, _r, _st, _d) in list(P3.STRATA.items()):
    if _s in P3.UNSEEN_STATUS_STRATA:             # PREREG-3's eight unseen-status strata (PREREG-5 map only)
        STRATA[_s] = ("p3", "noop", "canonical", "PREREG-3 stratum, canonical surface: " + _d)
DEV_STRATA = ("ID_posted", "ID_void", "S_words", "S_surface_posted", "S_surface_words", "H_words", "H_order", "H_names",
              "H_count", "H_para", "H_bal", "H_amt", "HS_words", "HS_posted", "LB_void", "LB_posted", "P3_words")
CONFIRM_STRATA = tuple(s for s in DEV_STRATA if s != "P3_words") + tuple(P3.UNSEEN_STATUS_STRATA)
AT_ONLY = ("H_amt",)
N_BASES = 100
ID_STRATA = ("ID_posted", "ID_void")
CANON_TWIN = {"apply": "ID_posted", "noop": "ID_void"}


def twin_stratum(stratum: str, idx: int) -> str:
    """The canonical-surface version of the same base (the 'own canonical' reference of a row)."""
    rule = STRATA[stratum][1]
    if rule == "mixed":
        rule = "apply" if idx % 2 == 0 else "noop"
    return CANON_TWIN[rule]


def word_for(table: dict, idx: int) -> tuple:
    kind = WORD_KINDS[idx % len(WORD_KINDS)]
    ws = table[kind]
    return kind, ws[(idx // len(WORD_KINDS)) % len(ws)]


def p3_status(stratum: str, idx: int) -> str:
    s = P3.STRATA[stratum][2]
    return s[idx % len(s)] if isinstance(s, tuple) else s


# ================================================================================ draws
def bal_value(rng: random.Random, cls: str, pool: list) -> int:
    if cls == "t2":
        return rng.choice(pool)
    lo, hi = BAL_CLASSES[cls] if cls in BAL_CLASSES else {"b5": HELDOUT_BAL}[cls]
    return rng.choice((-1, 1)) * rng.randint(lo, hi) if rng.random() < 0.35 else rng.randint(lo, hi)


def amt_value(rng: random.Random, cls: str, marg_amounts: list) -> int:
    if cls == "t2":
        return rng.choice(marg_amounts)
    lo, hi = AMT_CLASSES[cls] if cls in AMT_CLASSES else {"a4": HELDOUT_AMT}[cls]
    v = rng.randint(lo, hi)
    return -v if rng.random() < 0.25 else v


def small_pool(m) -> list:
    """env.gen_start's recorded balance pool restricted to |b| <= 98 (T2's training range)."""
    return [b for b in m.ledger_balances if -98 <= b <= 98]


def remap(before: dict, ev: dict, names: list) -> tuple:
    """Rename A-D (in sorted order) to `names` (sorted), keeping positions."""
    names = sorted(names)
    mp = dict(zip(CANON_NAMES, names))
    return {mp[a]: v for a, v in before.items()}, {**ev, "account": mp[ev["account"]], "other": mp[ev["other"]]}


def surface_with(before: dict, para: int, top: tuple, eperm: tuple, bperm: tuple) -> dict:
    names = sorted(before)
    return {"para": para, "top": apply_perm(list(TOP_KEYS), top), "event": apply_perm(list(EVENT_KEYS), eperm),
            "before": apply_perm(names, bperm)}


def surface_classes(surface: dict, before: dict, ev: dict) -> dict:
    """Factor levels of a rendered row (from its payload and surface): paraphrase, orders, names, count, status,
    magnitudes. Used by the builders' composition records; the audit reads the same back from the prompt TEXT."""
    names = sorted(before)
    n = len(names)
    if set(names) <= set(CANON_NAMES) | set(TRAIN_LETTERS) and set(CANON_NAMES) <= set(names):
        nk = "canonical" if n == 4 else "canonical_plus_train_letters"
    elif set(names) <= set(TRAIN_LETTERS):
        nk = "train_letters"
    elif set(names) <= set(TRAIN_NAME_WORDS):
        nk = "train_words"
    elif set(names) <= set(HELDOUT_LETTERS):
        nk = "heldout_letters"
    elif set(names) <= set(HELDOUT_NAME_WORDS):
        nk = "heldout_words"
    else:
        nk = "other"
    st = ev["status"]
    sk = ("posted" if st == "posted" else "void" if st == "void" else "train_word" if st in TRAIN_WORD_LIST else
          "heldout_word" if st in HELDOUT_WORD_LIST else "p3_word" if st in P3_WORDS else "excluded" if st in EXCLUDED_STATUS
          else "other")
    mb = max(abs(v) for v in before.values())
    bal = "t2" if mb <= 98 else "b3" if mb <= 999 else "b4" if mb <= 9999 else "b5" if mb <= 99999 else "beyond"
    a = abs(ev["amount"])
    if ev["kind"] == "scale":
        amt = "t2" if ev["amount"] in TRAINED_AMOUNTS else "other"
    else:
        amt = ("t2" if ev["amount"] in TRAINED_AMOUNTS else "a2" if 13 <= a <= 99 else "a3" if 100 <= a <= 999 else
               "a4" if 1000 <= a <= 9999 else "other")
    return {"para": "canonical" if surface["para"] == 0 else "heldout" if surface["para"] == HELDOUT_PARA else "train",
            "top": order_class("top", perm_of(surface["top"], TOP_KEYS)),
            "event": order_class("event", perm_of(surface["event"], EVENT_KEYS)),
            "before": order_class("before", perm_of(surface["before"], names), n),
            "names": nk, "count": "canonical" if n == 4 else "heldout" if n in HELDOUT_COUNTS else "train" if n in TRAIN_COUNTS
            else "other", "status": sk, "bal": bal, "amt": amt}


TRAINED_AMOUNTS = frozenset(set(range(-9, 13)) - {-3})   # the recorded marginal's 21 values (-9..12 without -3)


def heldout_factors(sc: dict) -> list:
    """The held-out factors present in a row's classes (empty for every training row)."""
    out = []
    for k in ("para", "top", "event", "before"):
        if sc[k] == "heldout":
            out.append(k)
    if sc["names"] in ("heldout_letters", "heldout_words", "other"):
        out.append("names")
    if sc["count"] in ("heldout", "other"):
        out.append("count")
    if sc["status"] in ("heldout_word", "p3_word", "excluded", "other"):
        out.append("status")
    if sc["bal"] in ("b5", "beyond"):
        out.append("bal")
    if sc["amt"] in ("a4", "other"):
        out.append("amt")
    return out


# ================================================================================ statistics (numpy-free where possible)
def wilson(k: int, n: int) -> tuple:
    return P3.wilson(k, n)


def binom_cdf_le(k: int, n: int) -> float:
    """P(X <= k), X ~ Bin(n, 1/2), exact."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2)."""
    return min(1.0, 2.0 * binom_cdf_le(min(b, c), b + c))


def holm(ps: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (monotone step-down), keys preserved."""
    items = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


BOOT_B = 10000
BOOT_SEED = 17


def boot_lower(diffs_by_cluster, B: int = BOOT_B, seed: int = BOOT_SEED, alpha: float = 0.05) -> dict:
    """Paired percentile bootstrap over clusters (bases): diffs_by_cluster[i] = (sum of differences, count) of cluster i.
    Returns the mean difference and its one-sided (1 - alpha) lower bound (numpy default linear percentile)."""
    import numpy as np
    s = np.array([d[0] for d in diffs_by_cluster], dtype=float)
    c = np.array([d[1] for d in diffs_by_cluster], dtype=float)
    if c.sum() == 0:
        return {"diff": float("nan"), "lo": float("nan"), "n_clusters": 0}
    rng = np.random.default_rng(seed)
    n = len(s)
    means = np.empty(B)
    done = 0
    while done < B:
        k = min(2000, B - done)
        idx = rng.integers(0, n, size=(k, n))
        num, den = s[idx].sum(axis=1), c[idx].sum(axis=1)
        means[done:done + k] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        done += k
    means = means[~np.isnan(means)]
    return {"diff": float(s.sum() / c.sum()), "lo": float(np.percentile(means, 100 * alpha)), "n_clusters": int(n),
            "B": int(B), "seed": int(seed)}


# ================================================================================ files
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_create(path: Path, text: str) -> Path:
    """Create-only, LF only, via .partial + rename (DECISIONS I-30: never CRLF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists (create-only)")
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def run_outputs(out_root: Path | None = None) -> list:
    d = Path(out_root or OUT_ROOT)
    return sorted(str(p) for p in d.rglob("*") if p.is_file()) if d.is_dir() else []


def write_doc(name: str, text: str, docs: Path | None = None) -> Path:
    """Kit records under docs/t2b/ledger. Before any runs/t2bl output exists a rerun refreshes them (each carries the
    code hashes it was made with); once an output exists they are never overwritten: the new record goes next to the
    old one with a UTC timestamp suffix. LF only."""
    d = Path(docs or DOCS)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if p.exists() and run_outputs():
        p = d / f"{p.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{p.suffix}"
    tmp = p.with_name(p.name + ".partial")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, p)
    return p


def dumps_rows(rows: list) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def code_hashes(rels) -> dict:
    return {r: sha256(ROOT / r) for r in rels if (ROOT / r).is_file()}
