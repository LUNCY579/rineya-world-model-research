"""T2b-ledger kit: shared definitions (torch-free). A surface-diversified ledger rule adapter on the T2 recipe, its dev
screen (scripts/t2bl_run.py, scripts/t2bl_gate.py) and the sealed confirmation PREREG-5 (scripts/t2bl_confirm_*.py,
PREREG_5_DRAFT.md). Design: docs/t2b/ledger/DESIGN.md (kit revision 3).

Kit revision 3 (2026-10-12; DECISIONS R29, the gate rule's pre-stated revision path: only the training data change, a
fresh dev bank under a new seed, the same thresholds). The revision-2 dev screen (runs/t2bl/dev) stopped PARTIAL
(heldout_posted_twins): its training set (seed 20261161) and dev bank (seed 20261162) and every adapter trained on them
are SPENT (data/t2bl_superseded_rev2/). Revision 3 changes the training composition only: (a) the 2-key top-level object
of the user JSON is trained in BOTH orders (TOP_TRAIN_PERMS; revision 2 never varied it, so the reversed top order that
the dev bank's H_order and combined held-out surface carry was an untrained extrapolation factor there; it is now a
trained level, deliberately, NOT a new held-out factor); (b) half of the diversified slots are rendered on the STACKED
training surface (a training paraphrase 1-3, training names, non-sorted training orders of the event and before
objects and the reversed top order at once), so posted rows under stacked variation rise. Operator decision at kit
revision 3 (2026-10-12, before any freeze or training): posted and non-posted rows are EXACTLY balanced on the
stacked-surface rows and on the reversed-top rows (300 / 300 and 403 / 403; binding band [0.49, 0.51]), so there is no
surface-to-status cue on those surfaces. The first revision-3 build (kept in docs/t2b/ledger/superseded_rev3_preop/,
never trained on) was 0.457 posted on both and did NOT meet that; its 'no surface-to-status cue' wording was wrong.
Vocabularies, magnitude bands, paraphrases and every held-out partition (words, names,
event / before orders, paraphrases 4 / 5, the 8-account count, the magnitude bands) are unchanged and disjoint from
training. The dev bank is redrawn by the same builder under a new seed into a new run directory (DEV_RUN).

Why (REPORT 3c, 3d; DECISIONS D-34/D-35/D-45). T2 s0 learned the literal token 'void' as the ledger no-op cue (PREREG-3
STATUS: TRIGGER; unseen statuses 0.01-0.18), fails out of range on magnitudes, reordered keys and word account names,
and plans at 0.706 on its training surface but 5/180 on PREREG-LB's held-out combined surface. T2b-ledger trains the
same recipe on a fresh oracle-labelled ledger one-step set whose SURFACE varies (status words, magnitudes, input key
orders, account names and counts, rule-text paraphrases), with explicit TRAIN vs HELD-OUT partitions fixed here, so the
dev screen and the confirmation can test on surfaces no training prompt used. Targets are unchanged: env.dumps(after),
the compact sorted-key balances object T2 trained on.

Held-out halves (revision 2). Every held-out partition that the dev gate reads (status words, account names, key orders
of the event and before objects, the rule paraphrase) is split into a DEV half (the dev bank and the dev gate only) and
a CONFIRM half (PREREG-5's sealed map bank only); both halves are excluded from training. So the confirmation tests
held-out items the dev gate never conditioned on. Not split (stated limits): the top-level order (its only other order,
reversed, is in both banks; since kit revision 3 it is a TRAINED order, not held out), the 8-account count and the
magnitude bands (the gate reads neither; the confirmation's values are freshly drawn), and PREREG-LB's rendering (bank /
card / cash / fees, keys reversed, 'cancelled'), which both banks carry only in report-only strata. Every held-out
status word is token-novel under Titan's tokenizer: it contains neither the token 'void' nor the token 'posted', and not
all of its tokens occur in trained statuses (t2bl_audit, binding).

Frame. Titan's one-step ledger frame family: a system text (one of six paraphrases of the frame's rule; paraphrase 0 is
rcwmt.env.SYSTEM_ONE['ledger_replay'] verbatim) and a compact JSON user turn {"before": {...}, "event": {...}} whose
three objects are serialised in the row's recorded key orders. On the canonical surface (paraphrase 0, sorted keys,
accounts A-D, posted / void, |b| <= 98, trained amounts) messages() == rcwmt.env.one_step_messages byte for byte.

Gold. p3_common.gold_of: rcwmt.env.step (every status other than the exact string 'posted' is a no-op), asserted equal
to the explicit rule implementation p3_common.rule_gold and to the validated FDS oracle on every row.

Seeds (grep over the RCWM, BWM and complementary trees: unused when taken; RCX owns 202612xx, D-39):
  20261165 training set (kit revision 3; regenerated under the SAME seed at the operator decision of 2026-10-12: every
  sub-stream is derived from the seed exactly as before, only the plan's quotas changed; t2bl_make_data GEN_VERSION
  t2bl-data-v3.1), 20261166 dev bank (kit revision 3; byte-identical across that regeneration). A grep over every tree
  under WorldModel/ on 2026-10-12 (HSKILL_CONFIRM_V2* excluded by name) found 20261165-20261179 unused. 20261163
  held-out key orders (unchanged: the held-out partitions are unchanged), 20261164 tests only (never a registered set).
  SPENT (revision 2): 20261161 training set, 20261162 dev bank.
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

KIT_REVISION = 3
LEDGER = "ledger_replay"
DATA = ROOT / "data"
DOCS = ROOT / "docs" / "t2b" / "ledger"
OUT_ROOT = ROOT / "runs" / "t2bl"
RUN_NAME = "dev_rev3"                            # this kit revision's dev-screen directory (t2bl_run.py tag 'dev')
DEV_RUN = OUT_ROOT / RUN_NAME
DEV_RUN_REL = f"runs/t2bl/{RUN_NAME}"
SPENT_RUNS = ("runs/t2bl/dev",)                  # revision 2's dev screen (R29): spent, never an input
TOKENIZER_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                     r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
SEED_TRAIN = 20261165                            # kit revision 3
SEED_DEV = 20261166                              # kit revision 3
ORDER_SEED = 20261163
TEST_SEED = 20261164
SPENT_KIT_SEEDS = (20261161, 20261162)           # revision 2's training set and dev bank (spent, R29)
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
SPLITS = ("dev", "confirm")

# ================================================================================ status vocabularies (the partition)
# Every non-posted string below is a no-op under the frame's rule ("If status is not posted, leave all balances
# unchanged"). Kinds: words that MEAN "not applied"; neutral / positive words without that meaning (the rule and
# world-knowledge semantics come apart: an 'accepted' event would naturally be applied); nonce words; case and spelling
# variants of the trained 'void'; look-alikes built from 'posted' that are not the string 'posted' (affixed forms, never
# misspellings: a misspelt 'posted' has no defensible gold, PREREG-3 Appendix B Q2).
# Case variants of 'void' are TRAINING items only ('VOId', 'vOid'; PREREG-3's 'Void' / 'VOID' stay reserved): a held-out
# case variant would lower-case to the trained 'void', so every HELD-OUT string differs from every trained string even
# after lower-casing (t2bl_audit checks it).
# Held-out strings are TOKEN-NOVEL under Titan's tokenizer (in the JSON context '"status":"<w>"'): no token 'void', no
# token 'posted', and at least one token that no trained status contains (t2bl_audit.token_audit, binding). The trained
# variants keep the anchor token on purpose (voids = [void, s], xposted = [x, posted]): a trigger widened to the anchor
# tokens passes every trained variant and fails every held-out one.
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
HELDOUT_WORDS_DEV = OrderedDict([                  # the dev bank and the dev gate only
    ("not_applied", ("rescinded", "refused", "stalled", "nullified")),
    ("neutral", ("verified", "valid", "signed", "routine")),
    ("nonce", ("toma", "zib", "fep", "wux")),
    ("void_variant", ("voyd", "vooid", "v0id", "re_void")),
    ("posted_like", ("yposted", "mposted", "tposted", "posteda")),
])
HELDOUT_WORDS_CONFIRM = OrderedDict([              # PREREG-5's sealed map bank only
    ("not_applied", ("halted", "vetoed", "retracted", "denied")),
    ("neutral", ("marked", "linked", "normal", "seen")),
    ("nonce", ("blorf", "snik", "trav", "quib")),
    ("void_variant", ("viod", "vo_id", "vo1d", "vojd")),
    ("posted_like", ("aposted", "oposted", "sposted", "postedy")),
])
HELDOUT_WORDS = OrderedDict((k, HELDOUT_WORDS_DEV[k] + HELDOUT_WORDS_CONFIRM[k]) for k in WORD_KINDS)
P3_WORDS = tuple(P3.NONPOSTED_POOL)               # PREREG-3's 16 strings: reserved, never trained (secondary stratum)
LB_STATUS = "cancelled"                           # PREREG-LB's rendering of 'void' (one of PREREG-3's 16)
# No defensible gold (the literal and the natural reading of the rule disagree; PREREG-3 Appendix B Q2): never used.
EXCLUDED_STATUS = ("Posted", "POSTED", "posted.", " posted", "posted ", "booked", "recorded", "settled", "cleared",
                   "completed", "confirmed", "logged", "entered", "applied", "executed", "processed", "done")
TRAIN_WORD_LIST = tuple(w for ws in TRAIN_WORDS.values() for w in ws)
HELDOUT_WORD_LIST_DEV = tuple(w for ws in HELDOUT_WORDS_DEV.values() for w in ws)
HELDOUT_WORD_LIST_CONFIRM = tuple(w for ws in HELDOUT_WORDS_CONFIRM.values() for w in ws)
HELDOUT_WORD_LIST = HELDOUT_WORD_LIST_DEV + HELDOUT_WORD_LIST_CONFIRM
WORD_KIND_OF = {w: k for k, ws in list(TRAIN_WORDS.items()) + list(HELDOUT_WORDS.items()) for w in ws}
TRAIN_STATUSES = ("posted", "void") + TRAIN_WORD_LIST

# ================================================================================ account names and counts
TRAIN_LETTERS = ("I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V")   # not E-H, O, W-Z
HELDOUT_LETTERS_DEV = ("W", "X", "Y", "Z")
HELDOUT_LETTERS_CONFIRM = tuple(P3.LETTER_NAMES)  # E F G H (PREREG-3's renamed letters)
TRAIN_NAME_WORDS = ("bills", "bonus", "cafe", "coins", "dues", "farm", "gifts", "house", "media", "petty", "pool",
                    "roof", "salary", "shop", "tips", "trust")
HELDOUT_NAME_WORDS_DEV = ("barn", "dock", "lamp", "mill", "nest", "pier", "silo", "yard")
HELDOUT_NAME_WORDS_CONFIRM = ("fund", "loan", "rent", "wage", "fort", "pond", "vault", "gold")
LB_NAMES = ("bank", "card", "cash", "fees")       # PREREG-LB's heldout_combined names (lbeam_f.WORD_NAMES4): reserved
HELDOUT_LETTERS = HELDOUT_LETTERS_DEV + HELDOUT_LETTERS_CONFIRM
HELDOUT_NAME_WORDS = HELDOUT_NAME_WORDS_DEV + HELDOUT_NAME_WORDS_CONFIRM
TRAIN_COUNTS = (4, 5, 6, 7)
HELDOUT_COUNTS = (8,)

# ================================================================================ magnitudes
# T2 trained balances |b| <= 98 (the recorded pool) and the recorded amount marginal -9..12. T2b-ledger widens both by
# two orders of magnitude, drawn by digit-count class so short numbers keep a large share (T2's range stays a subset,
# so its canonical rows are in the training distribution); the held-out band is the NEXT digit count. No training
# prompt or target contains a number with more than 4 digits (MAX_TRAIN_ABS; the audit checks it), so 'unseen
# magnitude' means a digit length never seen anywhere in training, in inputs or in outputs. Scale factors stay the
# recorded marginal everywhere and scale rows' balances stay at most 3 digits in training (no multi-digit x factor
# product beyond 4 digits); magnitude claims cover add / transfer, scale at the held-out band is report-only.
BAL_CLASSES = {"t2": None, "b3": (100, 999), "b4": (1000, 9999)}       # training classes ('t2' = the recorded pool)
AMT_CLASSES = {"t2": None, "a2": (13, 99), "a3": (100, 999)}           # add / transfer only
HELDOUT_BAL = (10000, 99999)                      # 5-digit balances
HELDOUT_AMT = (1000, 9999)                        # 4-digit amounts (add / transfer)
MAX_TRAIN_ABS = 9999
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
    ("Apply one event to the account balances. If the event's status is anything other than posted, the balances do not "
     "change at all. add: account goes up by amount. scale: account becomes account times amount. transfer: amount "
     "leaves account and arrives in other in a single step. Give only a JSON object with all of the balances after "
     "the event."),
)
TRAIN_PARA = (0, 1, 2, 3)
HELDOUT_PARA_DEV = 4
HELDOUT_PARA_CONFIRM = 5
HELDOUT_PARAS = (HELDOUT_PARA_DEV, HELDOUT_PARA_CONFIRM)
HELDOUT_PARA_FRAGMENTS = ("Here are some account balances", "Unless the event's status is posted",
                          "Answer with nothing but a JSON object", "Apply one event to the account balances",
                          "anything other than posted", "Give only a JSON object with all of the balances")

# ================================================================================ key orders
P_CANON_ORDER = 0.25                               # the canonical (sorted) order on about 1/4 of training rows
N_HELDOUT_PERMS = 6                                # per object: the reversal + 5 fixed random permutations
MIN_KENDALL = 2                                    # a random held-out order is >= 2 adjacent swaps from the sorted one
DEV_PERM_IDX = (1, 2, 3)                           # three random orders: the dev half
CONFIRM_PERM_IDX = (0, 4, 5)                       # the reversal + two random orders: the confirm half
# The 2-key top-level object {before, event} has two orders. Kit revision 2 trained only the sorted one and held the
# reversal out (shared by both halves); kit revision 3 trains BOTH (R29: the reversal was an untrained factor of every
# H_order and combined-surface row). The top object therefore has no held-out order since revision 3.
TOP_TRAIN_PERMS = ((0, 1), (1, 0))
P_TOP_REVERSED = 0.5                               # the reversed top order: an exact half of the 'free' slots of each
                                                   # DIV slot type (t2bl_make_data.train_plan; kit revision 3)


def kendall(p) -> int:
    """Inversions of a position permutation (= adjacent swaps from the identity)."""
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])


def _held_out_perms(n: int, rng: random.Random) -> list:
    """The held-out position permutations of an n-key object: the full reversal, then 5 fixed random permutations
    (each >= MIN_KENDALL inversions from the identity, distinct), drawn from rng in order."""
    out = [tuple(range(n - 1, -1, -1))]
    while len(out) < N_HELDOUT_PERMS:
        p = list(range(n))
        rng.shuffle(p)
        p = tuple(p)
        if kendall(p) >= MIN_KENDALL and p not in out:
            out.append(p)
    return out


def derive_heldout_orders() -> dict:
    """Regenerated from ORDER_SEED (the audit and the tests re-derive it independently): event (6 keys) and before (n
    accounts, n = 4..8, as position permutations of the sorted names) -> 6 each (the reversal + 5 random), split DEV
    (indices 1-3) / CONFIRM (0, 4, 5). The top object has no held-out order since kit revision 3 (both trained; revision
    2's 'top' entry drew nothing from the RNG, so every event / before order is the same as in revision 2)."""
    rng = random.Random(ORDER_SEED)
    out = {"event": _held_out_perms(len(EVENT_KEYS), rng)}
    for n in sorted(set(TRAIN_COUNTS + HELDOUT_COUNTS)):
        out[f"before{n}"] = _held_out_perms(n, rng)
    return out


HELDOUT_PERMS = derive_heldout_orders()
LB_ORDER = {"top": (1, 0), "event": tuple(range(len(EVENT_KEYS) - 1, -1, -1))}   # PREREG-LB keys_reversed


def perm_key(n: int) -> str:
    return f"before{n}"


def split_perms(split: str, key: str) -> list:
    """The held-out permutations of one half for an object key (event, before<n>; the top object has none)."""
    if key == "top":
        raise ValueError("the top-level object has no held-out order since kit revision 3 (both orders are trained)")
    idx = DEV_PERM_IDX if split == "dev" else CONFIRM_PERM_IDX
    return [HELDOUT_PERMS[key][i] for i in idx]


def heldout(split: str) -> dict:
    """Everything a bank of this half renders as 'held out'."""
    if split not in SPLITS:
        raise ValueError(split)
    dev = split == "dev"
    return {"split": split, "words": HELDOUT_WORDS_DEV if dev else HELDOUT_WORDS_CONFIRM,
            "letters": HELDOUT_LETTERS_DEV if dev else HELDOUT_LETTERS_CONFIRM,
            "name_words": HELDOUT_NAME_WORDS_DEV if dev else HELDOUT_NAME_WORDS_CONFIRM,
            "para": HELDOUT_PARA_DEV if dev else HELDOUT_PARA_CONFIRM,
            "event": split_perms(split, "event"), "before4": split_perms(split, "before4")}


_TRAIN_PERMS: dict = {}


def train_perms(obj: str, n: int | None = None) -> list:
    """Every permutation of the object's positions that is NOT held out (identity included), sorted. The top object:
    both orders (kit revision 3)."""
    if obj == "top":
        return list(TOP_TRAIN_PERMS)
    k = obj if obj == "event" else perm_key(n)
    if k not in _TRAIN_PERMS:
        size = len(EVENT_KEYS) if k == "event" else int(k[6:])
        ho = set(HELDOUT_PERMS[k])
        _TRAIN_PERMS[k] = [p for p in itertools.permutations(range(size)) if p not in ho]
    return _TRAIN_PERMS[k]


def draw_train_perm(rng: random.Random, obj: str, n: int | None = None, allow_canon: bool = True) -> tuple:
    """A training order of one object: event / before: the identity w.p. P_CANON_ORDER (if allowed), otherwise a
    uniformly drawn non-identity training permutation; top (kit revision 3): reversed w.p. P_TOP_REVERSED (always
    reversed if allow_canon is False). The training builder no longer draws the top order of a 'free' slot here (its
    plan assigns it by exact quota); the stacked mode still calls it with allow_canon=False (no draw)."""
    if obj == "top":
        if not allow_canon or rng.random() < P_TOP_REVERSED:
            return (1, 0)
        return (0, 1)
    size = len(EVENT_KEYS) if obj == "event" else n
    ident = tuple(range(size))
    if allow_canon and rng.random() < P_CANON_ORDER:
        return ident
    pool = [p for p in train_perms(obj, n) if p != ident]
    return pool[rng.randrange(len(pool))]


def apply_perm(keys_sorted, perm) -> list:
    return [keys_sorted[i] for i in perm]


def perm_of(order, keys_sorted) -> tuple:
    return tuple(list(keys_sorted).index(k) for k in order)


def order_class(obj: str, perm: tuple, n: int | None = None) -> str:
    """'canonical' (identity), 'heldout_dev' / 'heldout_confirm' (a held-out permutation of that half) or 'train' (the
    reversed top order is 'train' since kit revision 3)."""
    if perm == tuple(range(len(perm))):
        return "canonical"
    if obj == "top":
        return "train" if tuple(perm) in TOP_TRAIN_PERMS else "other"
    k = obj if obj == "event" else perm_key(n)
    ho = HELDOUT_PERMS[k]
    if tuple(perm) in ho:
        return "heldout_dev" if ho.index(tuple(perm)) in DEV_PERM_IDX else "heldout_confirm"
    return "train"


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


def within_train_range(before: dict, ev: dict) -> bool:
    """Every input balance and every balance of the POSTED version's gold has at most 4 digits."""
    after = P3.gold_of(before, {**ev, "status": "posted"})
    return all(abs(v) <= MAX_TRAIN_ABS for v in list(before.values()) + list(after.values()))


# ================================================================================ heuristics (training audit)
def heuristic_states(before: dict, ev: dict) -> dict:
    """What each heuristic predicts for a row. Binding audit members: copy, apply_regardless (= ignore_status: apply as
    if posted), void_trigger (T2's learned cue: no-op iff 'void' occurs in the lower-cased status). Descriptive string
    cues: semantic (no-op iff the word means 'not applied'), contains_posted (apply iff 'posted' is a substring),
    startswith_p, len6 (apply iff a 6-character status), endswith_ed (apply iff it ends with 'ed'); exact / ci_posted
    are the rule itself (case-insensitive equality agrees with the rule on every string this kit uses). Token-level cues
    (anchor trigger, token overlap) are computed by t2bl_audit.token_heuristics."""
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
# Kit revision 3 (binding; operator decision 2026-10-12): no surface-to-status cue on the revised surfaces. On the
# stacked-surface rows and on the reversed-top rows the builder makes posted and non-posted EXACTLY equal (copy and
# apply-regardless 0.50 each); the audit binds the posted share, copy and apply-regardless there within these bounds, so
# 'copy on an unfamiliar surface' and 'apply on an unfamiliar surface' are each wrong on half of those rows. (The first
# revision-3 build had (0.40, 0.60) and was 0.457 posted on both subsets: a residual cue; superseded unused.)
SURFACE_SUBSET_BOUNDS = (0.49, 0.51)
MIN_STACKED_POSTED_SHARE = 1 / 3                   # binding: >= 1/3 of the posted training rows on the stacked surface

# ================================================================================ strata (dev bank and PREREG-5 map)
# name -> (factor, gold rule, surface class, description). gold rule 'noop' = gold equals before; 'mixed' = posted on
# even bases and the base's TRAINING non-posted word (the S_words word of that base) on odd bases, so apply-regardless
# and copy score 0.50 and the status field is read under the unseen factor without also needing an unseen word (that is
# the combined cell's job); the odd rows' canonical twin is the base's S_words row. (T2 s0 knows only 'void', so a
# trained word, unlike 'void', does not hand it the no-op half of a cell.)
# 'the bank's half' = the dev half in the dev bank, the confirm half in PREREG-5's map bank (t2bl_common.heldout).
STRATA = OrderedDict([
    ("ID_posted", ("id", "apply", "canonical", "the posted base, canonical surface (in-distribution control; T_posted)")),
    ("ID_void", ("id", "noop", "canonical", "the base with status 'void', canonical surface (in-distribution control)")),
    ("S_words", ("words_train", "noop", "canonical", "a TRAINING non-posted word (kind by base index mod 5), canonical "
                                                       "surface otherwise (trained-variant control of the words factor)")),
    ("S_surface_posted", ("surface_train", "apply", "train", "posted, a TRAINING surface on the base's own in-range "
                          "values: paraphrase 1-3, training names (letters / words), 5-7 accounts (extra balances in "
                          "range), training key orders of the event and before objects (trained-variant control)")),
    ("S_surface_words", ("surface_train", "noop", "train", "the same training surface with a training non-posted word")),
    ("S_bal", ("magnitude_train", "apply", "canonical", "posted, every balance redrawn in the TRAINED 3-4-digit band "
                                                         "(add / transfer bases only; trained-variant control)")),
    ("S_amt", ("magnitude_train", "apply", "canonical", "posted, the amount redrawn in the TRAINED 2-3-digit band (add "
                                                         "/ transfer bases only; trained-variant control)")),
    ("H_words", ("words", "noop", "canonical", "a HELD-OUT non-posted word of the bank's half (kind by base index mod 5; "
                                                "4 words per kind), canonical surface otherwise")),
    ("H_order", ("order", "mixed", "heldout_order", "event and before in held-out orders of the bank's half, top "
                                                     "reversed (a trained order since kit revision 3); posted (even "
                                                     "bases) or a training word (odd)")),
    ("H_names", ("names", "mixed", "heldout_names", "accounts renamed to held-out names of the bank's half (4 letters or "
                                                     "4 of 8 words, sorted); posted (even bases) or a training word (odd)")),
    ("H_count", ("count", "mixed", "heldout_count", "8 accounts (held-out count): A-D plus 4 training letters with "
                                                     "in-range balances; posted (even bases) or a training word (odd)")),
    ("H_para", ("para", "mixed", "heldout_para", "the held-out rule paraphrase of the bank's half; posted (even bases) or "
                                                  "a training word (odd)")),
    ("H_bal", ("magnitude", "apply", "heldout_bal", "posted, every balance redrawn in the held-out 5-digit band")),
    ("H_amt", ("magnitude", "apply", "heldout_amt", "posted, the amount redrawn in the held-out 4-digit band (add / "
                                                     "transfer bases only)")),
    ("HS_words", ("combined", "noop", "heldout_combined", "the COMBINED held-out surface of the bank's half (its "
                  "paraphrase, its names, held-out event and before orders; top reversed, a trained order since kit "
                  "revision 3) with a held-out non-posted word of that half")),
    ("HS_posted", ("combined", "apply", "heldout_combined", "the same combined held-out surface, posted (the twins)")),
    ("H_order_top", ("order_detail", "mixed", "heldout_order", "report-only: ONLY the top-level object reversed (event "
                                                               "and before sorted; a TRAINED order since kit revision 3, "
                                                               "so a trained-variant report); posted / training word by "
                                                               "base parity")),
    ("H_order_inner", ("order_detail", "mixed", "heldout_order", "report-only: top sorted, event and before in held-out "
                                                                 "orders of the bank's half; posted / training word by "
                                                                 "base parity")),
    ("LB_void", ("lb_surface", "noop", "lb_heldout_combined", "PREREG-LB's heldout_combined rendering (paraphrase 0, keys "
                 "reversed, accounts bank/card/cash/fees) with 'void' rendered 'cancelled' (report-only)")),
    ("LB_posted", ("lb_surface", "apply", "lb_heldout_combined", "the same rendering, posted (report-only)")),
    ("P3_words", ("p3", "noop", "canonical", "PREREG-3's 16 reserved non-posted strings (base index mod 16), canonical "
                                             "surface (dev bank only)")),
])
for _s, (_g, _r, _st, _d) in list(P3.STRATA.items()):
    if _s in P3.UNSEEN_STATUS_STRATA:             # PREREG-3's eight unseen-status strata (PREREG-5 map only)
        STRATA[_s] = ("p3", "noop", "canonical", "PREREG-3 stratum, canonical surface: " + _d)
DEV_STRATA = ("ID_posted", "ID_void", "S_words", "S_surface_posted", "S_surface_words", "S_bal", "S_amt", "H_words",
              "H_order", "H_names", "H_count", "H_para", "H_bal", "H_amt", "HS_words", "HS_posted", "LB_void",
              "LB_posted", "P3_words")
CONFIRM_STRATA = tuple(s for s in DEV_STRATA if s != "P3_words") + ("H_order_top", "H_order_inner") + \
    tuple(P3.UNSEEN_STATUS_STRATA)
AT_ONLY = ("S_bal", "S_amt", "H_amt")
N_BASES = 100
ID_STRATA = ("ID_posted", "ID_void")
CANON_TWIN = {"apply": "ID_posted", "noop": "ID_void"}


def twin_stratum(stratum: str, idx: int) -> str:
    """The canonical-surface version of the same base and status (the 'own canonical' reference of a row): ID_posted
    for apply rows, ID_void for no-op rows of 'noop' strata, S_words for the training-word rows of mixed strata."""
    rule = STRATA[stratum][1]
    if rule == "mixed":
        return "ID_posted" if idx % 2 == 0 else "S_words"
    return CANON_TWIN[rule]


def mixed_status(idx: int) -> tuple:
    """(status, status kind) of a mixed stratum's row on base idx: posted on even bases, the base's S_words word on odd."""
    if idx % 2 == 0:
        return "posted", "posted"
    kind, w = word_for(TRAIN_WORDS, idx)
    return w, kind


def word_for(table: dict, idx: int) -> tuple:
    kind = WORD_KINDS[idx % len(WORD_KINDS)]
    ws = table[kind]
    return kind, ws[(idx // len(WORD_KINDS)) % len(ws)]


def p3_status(stratum: str, idx: int) -> str:
    s = P3.STRATA[stratum][2]
    return s[idx % len(s)] if isinstance(s, tuple) else s


def bank_rows_expected(strata: tuple, n_bases: int = N_BASES, n_at: int | None = None) -> int:
    n_at = n_at if n_at is not None else round(n_bases * (KIND_QUOTA["transfer"] + KIND_QUOTA["add"]))
    return sum(n_at if s in AT_ONLY else n_bases for s in strata)


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


def name_class(names) -> str:
    s = set(names)
    if s <= set(CANON_NAMES) | set(TRAIN_LETTERS) and set(CANON_NAMES) <= s:
        return "canonical" if len(s) == 4 else "canonical_plus_train_letters"
    for label, pool in (("train_letters", TRAIN_LETTERS), ("train_words", TRAIN_NAME_WORDS),
                        ("heldout_dev_letters", HELDOUT_LETTERS_DEV), ("heldout_dev_words", HELDOUT_NAME_WORDS_DEV),
                        ("heldout_confirm_letters", HELDOUT_LETTERS_CONFIRM),
                        ("heldout_confirm_words", HELDOUT_NAME_WORDS_CONFIRM), ("lb", LB_NAMES)):
        if s <= set(pool):
            return label
    return "other"


def status_class(st: str) -> str:
    return ("posted" if st == "posted" else "void" if st == "void" else "train_word" if st in TRAIN_WORD_LIST else
            "heldout_dev_word" if st in HELDOUT_WORD_LIST_DEV else "heldout_confirm_word" if st in HELDOUT_WORD_LIST_CONFIRM
            else "p3_word" if st in P3_WORDS else "excluded" if st in EXCLUDED_STATUS else "other")


def surface_classes(surface: dict, before: dict, ev: dict) -> dict:
    """Factor levels of a rendered row (from its payload and surface): paraphrase, orders, names, count, status,
    magnitudes. Used by the builders' composition records; the audit reads the same back from the prompt TEXT."""
    names = sorted(before)
    n = len(names)
    mb = max(abs(v) for v in before.values())
    bal = "t2" if mb <= 98 else "b3" if mb <= 999 else "b4" if mb <= 9999 else "b5" if mb <= 99999 else "beyond"
    a = abs(ev["amount"])
    if ev["kind"] == "scale":
        amt = "t2" if ev["amount"] in TRAINED_AMOUNTS else "other"
    else:
        amt = ("t2" if ev["amount"] in TRAINED_AMOUNTS else "a2" if 13 <= a <= 99 else "a3" if 100 <= a <= 999 else
               "a4" if 1000 <= a <= 9999 else "other")
    para = int(surface["para"])
    return {"para": "canonical" if para == 0 else "heldout_dev" if para == HELDOUT_PARA_DEV else
            "heldout_confirm" if para == HELDOUT_PARA_CONFIRM else "train",
            "top": order_class("top", perm_of(surface["top"], TOP_KEYS)),
            "event": order_class("event", perm_of(surface["event"], EVENT_KEYS)),
            "before": order_class("before", perm_of(surface["before"], names), n),
            "names": name_class(names), "count": "canonical" if n == 4 else "heldout" if n in HELDOUT_COUNTS else
            "train" if n in TRAIN_COUNTS else "other", "status": status_class(ev["status"]), "bal": bal, "amt": amt}


TRAINED_AMOUNTS = frozenset(set(range(-9, 13)) - {-3})   # the recorded marginal's 21 values (-9..12 without -3)


def stacked_surface(sc: dict) -> bool:
    """Kit revision 3's STACKED training surface (from a row's classes): a training paraphrase (1-3), training names
    (training letters or training words, not the A-D-based sets), non-sorted training orders of the event and before
    objects and the reversed top order, all at once: the training-surface analogue of the combined held-out surface."""
    return (sc["para"] == "train" and sc["names"] in ("train_letters", "train_words") and sc["event"] == "train"
            and sc["before"] == "train" and sc["top"] == "train")


def heldout_factors(sc: dict) -> list:
    """The held-out factors present in a row's classes (empty for every training row)."""
    out = []
    for k in ("para", "top", "event", "before"):
        if sc[k].startswith("heldout") or sc[k] == "other":
            out.append(k)
    if sc["names"].startswith("heldout") or sc["names"] in ("lb", "other"):
        out.append("names")
    if sc["count"] in ("heldout", "other"):
        out.append("count")
    if sc["status"] in ("heldout_dev_word", "heldout_confirm_word", "p3_word", "excluded", "other"):
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


def cluster_ci(hits_by_cluster, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Report-only: an accuracy with its 95% percentile bootstrap interval, resampling clusters (e.g. status WORDS):
    hits_by_cluster[i] = (correct, n) of cluster i."""
    import numpy as np
    s = np.array([d[0] for d in hits_by_cluster], dtype=float)
    c = np.array([d[1] for d in hits_by_cluster], dtype=float)
    if c.sum() == 0:
        return {"acc": float("nan"), "ci95": [float("nan"), float("nan")], "n_clusters": 0, "B": B}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return {"acc": float(s.sum() / c.sum()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n_clusters": int(len(s)), "B": int(B), "seed": int(seed)}


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


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_create(path: Path, text: str, stale_partial: str = "refuse") -> Path:
    """Create-only, LF only, via .partial + rename (DECISIONS I-30: never CRLF). stale_partial='aside': a leftover
    '<name>.partial' of a single-writer record (a crash or reboot between the write and the rename; callers hold the run
    lock) is moved aside to '<name>.partial.stale.<UTC>_<pid>' (never deleted) before the write; 'refuse' (default)
    raises FileExistsError on it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists (create-only)")
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists() and stale_partial == "aside":
        os.replace(tmp, tmp.with_name(f"{tmp.name}.stale.{_stamp()}_{os.getpid()}"))
    with open(tmp, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def run_outputs(out_root: Path | None = None) -> list:
    """Every file of THIS kit revision's dev screen (DEV_RUN; a spent earlier revision's run directory, runs/t2bl/dev,
    is not this revision's output and does not count)."""
    d = Path(out_root or DEV_RUN)
    return sorted(str(p) for p in d.rglob("*") if p.is_file()) if d.is_dir() else []


# PREREG-5's pre-freeze records: the freeze binds them (each must carry the current code, draft and hook draft), so they
# are REWRITTEN in place by every rerun until PREREG_5_FREEZE.json exists, including after the dev screen's outputs exist
# (the documented path to the freeze reruns the tests after --render-draft). Every other kit record (the data audit,
# the dev gate's rule document and dry run, the dev plan, the dev-screen test records) is rewritten only while no output
# of this revision's dev screen (runs/t2bl/<RUN_NAME>) exists; afterwards a rerun writes a new record next to the old one
# with a UTC timestamp suffix.
PREFREEZE_DOCS = ("TEST_P5ANALYSIS.json", "TEST_P5TINY.json", "TEST_P5FREEZE.json", "TEST_PLAN.json", "TEST_HOOK.json",
                  "TEST_REHEARSAL.json", "P5_DRYRUN.json", "P5_DRYRUN.md", "P5_GEN_PLAN.json", "P5_POWER.json",
                  "P5_POWER_DEV.json")


def write_doc(name: str, text: str, docs: Path | None = None) -> Path:
    """Kit records under docs/t2b/ledger (or a test's docs directory; its root is three levels up). LF only."""
    d = Path(docs or DOCS)
    d.mkdir(parents=True, exist_ok=True)
    root = d.parents[2]
    p = d / name
    if p.exists():
        if name in PREFREEZE_DOCS:
            if (root / "PREREG_5_FREEZE.json").exists():
                p = d / f"{p.stem}.{_stamp()}{p.suffix}"
        elif run_outputs(root / "runs" / "t2bl" / RUN_NAME):
            p = d / f"{p.stem}.{_stamp()}{p.suffix}"
    tmp = p.with_name(p.name + ".partial")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, p)
    return p


def dumps_rows(rows: list) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def code_hashes(rels) -> dict:
    return {r: sha256(ROOT / r) for r in rels if (ROOT / r).is_file()}
