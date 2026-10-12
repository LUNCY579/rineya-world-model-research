"""T2b-ledger data audit: an INDEPENDENT read-back of every prompt TEXT (torch-free, model-free).

It does not use the rows' recorded surface, nor t2bl_common's renderer or class logic: each training prompt is split
with its own regular expression, the user turn is parsed with json.loads (object_pairs_hook keeps the literal key
order and rejects duplicate keys), the held-out key orders are re-derived from ORDER_SEED by this module's own copy of
the documented procedure, and every factor level is classified here from the partition tables (the definitions). The
gold is recomputed by an independent implementation of the frame's rule.

  audit(train_rows, dev_rows) -> {"ok", "problems", counts ...}

Token audit (binding; Titan's tokenizer.json via the `tokenizers` library, no model): every held-out status of both
halves, tokenised in its JSON context '"status":"<w>","tick"', contains neither the token 'void' nor the token 'posted'
and has at least one token that no trained status has (so neither a trigger widened to the anchor tokens nor a
trained-token lookup can tell a held-out word from posted). Also reported: PREREG-3's strings and the account names.
Training checks (binding): the composition (1,500 rows; posted 750 / non-posted 750; 300 canonical-block rows in the
exact canonical T2 frame; pair members share before, event fields other than status, and surface); the read-back of
every prompt equals the row's own payload and surface; NO held-out factor of either half in any training prompt
(paraphrases 4 and 5, a held-out event / before order, a held-out or PREREG-LB name, an 8-account object, a held-out,
PREREG-3 or excluded status, a 5-digit balance, a 4-digit amount, a scale factor outside the recorded marginal); no
number with more than 4 digits in any prompt or target; none of the held-out text fragments (the held-out paraphrases'
phrases, '"status":"<w>"' for every held-out / PREREG-3 / excluded string, '"<name>":' for every held-out and PREREG-LB
name) in any prompt or target; every target equals the independent gold. Kit revision 3: the reversed top-level order
is a TRAINING level (revision 2 held it out); both top orders occur on at least MIN_TOP_ORDER_ROWS training rows each,
and at least t2bl_common.MIN_STACKED_POSTED_SHARE of the posted rows are on the STACKED training surface, read back
here from the prompt text (a training paraphrase 1-3, training letters or words as names, non-sorted training event and
before orders, the reversed top order, at once).
Bank checks (binding; the dev bank here, PREREG-5's map bank at its freeze): every row is its stratum's declared factor
change of its base and nothing else, rendering the bank's own half of every held-out partition (classified from the
rendered text), mixed strata carry posted on even bases and the base's training word (its S_words word) on odd bases,
its gold is the independent gold, its twin exists, every stratum has its row count.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t2bl_common as C  # noqa: E402  (partition TABLES and paraphrase texts only: the definitions)

_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
                 r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n", re.S)


class Dup(ValueError):
    pass


def _pairs(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise Dup("duplicate key")
    return list(pairs)


def _inversions(p) -> int:
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])


def regen_heldout() -> dict:
    """Independent copy of the documented procedure (t2bl_common.derive_heldout_orders): random.Random(ORDER_SEED);
    event: the reversal, then shuffles of range(6) until 5 new permutations with >= 2 inversions; then before4..before8
    the same way, in that order. (No top-level entry: both top orders are trained since kit revision 3.)"""
    rng = random.Random(C.ORDER_SEED)
    out = {}
    for name, n in [("event", 6)] + [(f"before{k}", k) for k in range(4, 9)]:
        lst = [tuple(reversed(range(n)))]
        while len(lst) < 6:
            p = list(range(n))
            rng.shuffle(p)
            t = tuple(p)
            if _inversions(t) >= 2 and t not in lst:
                lst.append(t)
        out[name] = lst
    return out


HELD = regen_heldout()
DEV_IDX, CONFIRM_IDX = (1, 2, 3), (0, 4, 5)          # the documented split of each object's six held-out orders
MIN_TOP_ORDER_ROWS = 300                             # kit revision 3: each top-level order on >= 300 training rows


def read_back(prompt: str) -> dict:
    """The surface and payload encoded in a rendered prompt's TEXT."""
    m = _RE.fullmatch(prompt)
    if not m:
        raise ValueError("not a rendered system+user chat prompt")
    sys_text, user = m.group(1), m.group(2)
    top = json.loads(user, object_pairs_hook=_pairs)
    if not isinstance(top, list) or sorted(k for k, _ in top) != ["before", "event"]:
        raise ValueError("user turn is not {before, event}")
    d = dict(top)
    if not isinstance(d["before"], list) or not isinstance(d["event"], list):
        raise ValueError("before / event are not objects")
    before_order = [k for k, _ in d["before"]]
    event_order = [k for k, _ in d["event"]]
    if user != json.dumps({k: (dict(v)) for k, v in top}, separators=(",", ":")):
        raise ValueError("user turn is not compact JSON")
    para = C.PARAPHRASES.index(sys_text) if sys_text in C.PARAPHRASES else None
    return {"para": para, "system": sys_text, "top": [k for k, _ in top], "before_order": before_order,
            "event_order": event_order, "before": dict(d["before"]), "event": dict(d["event"])}


def _perm(order, ref) -> tuple:
    return tuple(sorted(ref).index(k) for k in order)


def _perm_level(key: str, perm: tuple) -> str:
    if perm == tuple(range(len(perm))):
        return "canonical"
    if key not in HELD:
        return "other"
    if perm in HELD[key]:
        return "heldout_dev" if HELD[key].index(perm) in DEV_IDX else "heldout_confirm"
    return "train"


def classify(rb: dict) -> dict:
    """Factor levels from a read-back (independent of t2bl_common.surface_classes)."""
    b, e = rb["before"], rb["event"]
    names = set(b)
    n = len(b)
    out = {}
    out["para"] = "unknown" if rb["para"] is None else "canonical" if rb["para"] == 0 else \
        "heldout_dev" if rb["para"] == 4 else "heldout_confirm" if rb["para"] == 5 else "train"
    out["top"] = "canonical" if rb["top"] == ["before", "event"] else "reversed"     # both trained (kit revision 3)
    out["event"] = _perm_level("event", _perm(rb["event_order"], C.EVENT_KEYS)) \
        if sorted(rb["event_order"]) == sorted(C.EVENT_KEYS) else "malformed"
    out["before"] = _perm_level(f"before{n}", _perm(rb["before_order"], b))
    lt, ld = set(C.TRAIN_LETTERS), set(C.CANON_NAMES)
    if names <= ld | lt and ld <= names:
        out["names"] = "canonical" if names == ld else "canonical_plus_train_letters"
    elif names <= lt:
        out["names"] = "train_letters"
    elif names <= set(C.TRAIN_NAME_WORDS):
        out["names"] = "train_words"
    elif names <= set(C.HELDOUT_LETTERS_DEV) or names <= set(C.HELDOUT_NAME_WORDS_DEV):
        out["names"] = "heldout_dev"
    elif names <= set(C.HELDOUT_LETTERS_CONFIRM) or names <= set(C.HELDOUT_NAME_WORDS_CONFIRM):
        out["names"] = "heldout_confirm"
    elif names <= set(C.LB_NAMES):
        out["names"] = "lb"
    else:
        out["names"] = "other"
    out["count"] = "canonical" if n == 4 else "train" if n in (5, 6, 7) else "heldout" if n == 8 else "other"
    st = e.get("status")
    if st in ("posted", "void"):
        out["status"] = st
    elif st in C.TRAIN_WORD_LIST:
        out["status"] = "train_word"
    elif st in C.HELDOUT_WORD_LIST_DEV:
        out["status"] = "heldout_dev_word"
    elif st in C.HELDOUT_WORD_LIST_CONFIRM:
        out["status"] = "heldout_confirm_word"
    elif st in C.P3_WORDS:
        out["status"] = "p3_word"
    elif st in C.EXCLUDED_STATUS:
        out["status"] = "excluded"
    else:
        out["status"] = "other"
    mb = max(abs(v) for v in b.values())
    out["bal"] = "t2" if mb <= 98 else "wide" if mb <= 9999 else "heldout"
    a = e.get("amount")
    if e.get("kind") == "scale":
        out["amt"] = "t2" if a in C.TRAINED_AMOUNTS else "heldout"
    else:
        out["amt"] = "t2" if a in C.TRAINED_AMOUNTS else "wide" if 13 <= abs(a) <= 999 else "heldout"
    return out


HELDOUT_LEVEL = {"para": ("heldout_dev", "heldout_confirm", "unknown"), "top": (),     # no held-out top order (rev 3)
                 "event": ("heldout_dev", "heldout_confirm", "malformed", "other"),
                 "before": ("heldout_dev", "heldout_confirm", "other"),
                 "names": ("heldout_dev", "heldout_confirm", "lb", "other"), "count": ("heldout", "other"),
                 "status": ("heldout_dev_word", "heldout_confirm_word", "p3_word", "excluded", "other"),
                 "bal": ("heldout",), "amt": ("heldout",)}


def heldout_hits(cl: dict) -> list:
    return [k for k, bad in HELDOUT_LEVEL.items() if cl[k] in bad]


def indep_gold(b: dict, e: dict) -> dict:
    """The frame's rule, implemented here: anything but the exact string 'posted' leaves every balance unchanged."""
    out = dict(b)
    if e["status"] != "posted":
        return out
    k, a, amt = e["kind"], e["account"], e["amount"]
    if k == "add":
        out[a] = b[a] + amt
    elif k == "scale":
        out[a] = b[a] * amt
    elif k == "transfer":
        out[a], out[e["other"]] = b[a] - amt, b[e["other"]] + amt
    else:
        raise ValueError(k)
    return out


def fragments() -> dict:
    frags = {f"held-out paraphrase: {f!r}": f for f in C.HELDOUT_PARA_FRAGMENTS}
    for w in C.HELDOUT_WORD_LIST + C.P3_WORDS + C.EXCLUDED_STATUS:
        frags[f"status {w!r}"] = '"status":' + json.dumps(w)
    for nm in C.HELDOUT_LETTERS + C.HELDOUT_NAME_WORDS + C.LB_NAMES:
        frags[f"name {nm!r} as a key"] = json.dumps(nm) + ":"
    return frags


def _canon_msgs_ok(r: dict) -> bool:
    from rcwmt import env
    msgs = env.one_step_messages(C.LEDGER, r["before"], r["event"])
    return C.render_chat(msgs) == r["prompt"]


# ============================================================================================== token audit (binding)
_TK: list = []


def _tokenizer():
    if not _TK:
        import tokenizers
        _TK.append(tokenizers.Tokenizer.from_file(str(C.TOKENIZER_JSON)))
    return _TK[0]


def status_tokens(w: str) -> tuple:
    """The tokens covering a status string in its JSON context '"status":"<w>","tick"' (Titan's tokenizer)."""
    full = '"status":%s,"tick"' % json.dumps(w)
    lo = len('"status":"')
    hi = lo + len(json.dumps(w)) - 2
    enc = _tokenizer().encode(full, add_special_tokens=False)
    return tuple(t for t, (a, b) in zip(enc.tokens, enc.offsets) if b > lo and a < hi)


def name_tokens(w: str) -> tuple:
    full = ',%s:12,' % json.dumps(w)
    lo, hi = 2, 2 + len(w)
    enc = _tokenizer().encode(full, add_special_tokens=False)
    return tuple(t for t, (a, b) in zip(enc.tokens, enc.offsets) if b > lo and a < hi)


def trained_status_tokens(noop_only: bool = False) -> set:
    out: set = set()
    for w in C.TRAIN_STATUSES:
        if noop_only and w == "posted":
            continue
        out.update(status_tokens(w))
    return out


def token_cue_noop(status: str) -> dict:
    """Token-level trigger heuristics (True = the cue makes the row a no-op): void_token (the token 'void' is present);
    anchor_trigger (the trigger widened to the trained anchors: the token 'void', or the token 'posted' with another
    token); token_overlap (the status is not the lone token 'posted' and shares a token with a trained non-posted
    status)."""
    t = status_tokens(status)
    noop_toks = _cached_noop_tokens()
    return {"void_token": "void" in t, "anchor_trigger": "void" in t or ("posted" in t and len(t) > 1),
            "token_overlap": t != ("posted",) and any(x in noop_toks for x in t)}


_NOOP: list = []


def _cached_noop_tokens() -> set:
    if not _NOOP:
        _NOOP.append(trained_status_tokens(noop_only=True))
    return _NOOP[0]


def token_audit() -> dict:
    """Binding: every held-out status (both halves) is token-novel. Descriptive: PREREG-3's strings, account names."""
    try:
        trained = trained_status_tokens()
    except Exception as e:                          # noqa: BLE001  (no tokenizer = the audit cannot pass)
        return {"ok": False, "problems": [f"token audit unavailable ({type(e).__name__}: {e})"]}
    probs, held = [], {}
    for half, ws in (("dev", C.HELDOUT_WORD_LIST_DEV), ("confirm", C.HELDOUT_WORD_LIST_CONFIRM)):
        for w in ws:
            t = status_tokens(w)
            anchor = [x for x in t if x in ("void", "posted")]
            novel = [x for x in t if x not in trained]
            held[w] = {"half": half, "kind": C.WORD_KIND_OF[w], "tokens": list(t), "anchor_tokens": anchor,
                       "untrained_tokens": novel}
            if anchor:
                probs.append(f"held-out status {w!r} ({half}) contains the anchor token(s) {anchor}")
            if not novel:
                probs.append(f"held-out status {w!r} ({half}) consists only of trained status tokens {list(t)}")
    p3 = {}
    for w in C.P3_WORDS:
        t = status_tokens(w)
        p3[w] = {"tokens": list(t), "anchor_tokens": [x for x in t if x in ("void", "posted")],
                 "all_tokens_trained": all(x in trained for x in t)}
    tn = set()
    for w in list(C.CANON_NAMES) + list(C.TRAIN_LETTERS) + list(C.TRAIN_NAME_WORDS):
        tn.update(name_tokens(w))
    names = {w: {"tokens": list(name_tokens(w)), "shared_with_trained_names": [x for x in name_tokens(w) if x in tn]}
             for w in C.HELDOUT_LETTERS + C.HELDOUT_NAME_WORDS + C.LB_NAMES}
    return {"ok": not probs, "problems": probs, "n_trained_status_tokens": len(trained), "heldout": held,
            "prereg3_descriptive": p3, "names_descriptive": names,
            "context": '"status":"<w>","tick" (names: ,"<name>":12,)', "tokenizer_json_sha256": C.sha256(C.TOKENIZER_JSON)}


# ============================================================================================== training audit
def audit_train(rows: list) -> dict:
    probs = []
    hits, frag_hits = Counter(), Counter()
    frags = fragments()
    cl_counts: dict = {}
    n_fully_canon = 0
    max_abs = 0
    stacked = Counter()                               # kit revision 3: the stacked training surface, from the TEXT
    top_orders = Counter()
    for r in rows:
        try:
            rb = read_back(r["prompt"])
        except (ValueError, Dup) as e:
            probs.append(f"{r.get('id')}: unreadable prompt ({e})")
            continue
        if rb["before"] != r["before"] or rb["event"] != r["event"] or rb["para"] != r["surface"]["para"] or \
                rb["top"] != r["surface"]["top"] or rb["event_order"] != r["surface"]["event"] or \
                rb["before_order"] != r["surface"]["before"]:
            probs.append(f"{r['id']}: the prompt text does not encode the row's own payload and surface")
        cl = classify(rb)
        for k, v in cl.items():
            cl_counts.setdefault(k, Counter())[v] += 1
        h = heldout_hits(cl)
        for k in h:
            hits[k] += 1
        if h:
            probs.append(f"{r['id']}: held-out factor(s) in a training prompt: {h}")
        for name, f in frags.items():
            if f in r["prompt"] or (name.startswith("name") and f in r["target"]):
                frag_hits[name] += 1
        tgt = json.loads(r["target"])
        if tgt != indep_gold(rb["before"], rb["event"]) or \
                r["target"] != json.dumps(indep_gold(rb["before"], rb["event"]), separators=(",", ":"), sort_keys=True):
            probs.append(f"{r['id']}: target differs from the independent gold")
        m = max(abs(v) for v in list(rb["before"].values()) + list(tgt.values()) + [rb["event"]["amount"]])
        max_abs = max(max_abs, m)
        if m > C.MAX_TRAIN_ABS:
            probs.append(f"{r['id']}: a number with more than 4 digits in a training prompt or target ({m})")
        canon = all(cl[k] == v for k, v in (("para", "canonical"), ("top", "canonical"), ("event", "canonical"),
                                            ("before", "canonical"), ("names", "canonical"), ("bal", "t2"), ("amt", "t2")))
        canon = canon and cl["status"] in ("posted", "void")
        n_fully_canon += canon
        is_posted = rb["event"]["status"] == "posted"
        top_orders[(cl["top"], "posted" if is_posted else "nonposted")] += 1
        if cl["para"] == "train" and cl["names"] in ("train_letters", "train_words") and cl["event"] == "train" and \
                cl["before"] == "train" and cl["top"] == "reversed":
            stacked["posted" if is_posted else "nonposted"] += 1
        if r["block"] == "canon" and not (canon and _canon_msgs_ok(r)):
            probs.append(f"{r['id']}: a canonical-block row is not Titan's exact canonical frame")
    if frag_hits:
        probs.append(f"held-out text fragments found: {dict(frag_hits)}")
    # composition and pairs
    by_slot: dict = {}
    for r in rows:
        by_slot.setdefault((r["block"], r["slot"]), []).append(r)
    for key, rs in by_slot.items():
        if len(rs) == 2:
            a, b = rs
            if a["before"] != b["before"] or a["surface"] != b["surface"] or \
                    {k: v for k, v in a["event"].items() if k != "status"} != {k: v for k, v in b["event"].items() if k != "status"} \
                    or sorted([a["event"]["status"] == "posted", b["event"]["status"] == "posted"]) != [False, True]:
                probs.append(f"pair {key}: members differ in more than the status, or are not posted + non-posted")
    posted = sum(r["event"]["status"] == "posted" for r in rows)
    comp = {"rows": len(rows), "posted": posted, "nonposted": len(rows) - posted,
            "canon_block": sum(r["block"] == "canon" for r in rows), "pairs": sum(len(v) == 2 for v in by_slot.values()),
            "fully_canonical_rows": n_fully_canon, "max_abs_value": max_abs}
    want = {"rows": 1500, "posted": 750, "nonposted": 750, "canon_block": 300, "pairs": 400}
    for k, v in want.items():
        if comp[k] != v:
            probs.append(f"composition {k} = {comp[k]} != {v}")
    if len({C.payload(r["before"], r["event"]) for r in rows}) != len(rows):
        probs.append("duplicate training payloads")
    # kit revision 3 (binding): both top-level orders trained; posted rows on the stacked training surface
    tops = {t: {"posted": top_orders[(t, "posted")], "nonposted": top_orders[(t, "nonposted")],
                "rows": top_orders[(t, "posted")] + top_orders[(t, "nonposted")]} for t in ("canonical", "reversed")}
    for t, d in tops.items():
        if d["rows"] < MIN_TOP_ORDER_ROWS:
            probs.append(f"top-level order {t} on {d['rows']} training rows (< {MIN_TOP_ORDER_ROWS}): both orders must be trained")
    st = {"rows": stacked["posted"] + stacked["nonposted"], "posted": stacked["posted"], "nonposted": stacked["nonposted"],
          "share_of_posted_rows": round(stacked["posted"] / max(1, posted), 4),
          "posted_share_within": round(stacked["posted"] / max(1, stacked["posted"] + stacked["nonposted"]), 4)}
    if st["share_of_posted_rows"] < C.MIN_STACKED_POSTED_SHARE:
        probs.append(f"only {st['share_of_posted_rows']} of the posted training rows are on the stacked training surface "
                     f"(< {C.MIN_STACKED_POSTED_SHARE})")
    lo, hi = C.SURFACE_SUBSET_BOUNDS
    for label, d in (("stacked surface", st), ("reversed top", tops["reversed"])):
        ps = d["posted"] / max(1, d["rows"])
        if not lo <= ps <= hi:
            probs.append(f"the {label} rows are {ps:.3f} posted, outside [{lo}, {hi}] (a surface-to-status cue)")
    comp["top_orders"] = tops
    comp["stacked_surface"] = st
    return {"problems": probs, "heldout_factor_hits": dict(hits), "fragment_hits": dict(frag_hits),
            "n_fragments_checked": len(frags), "classes": {k: dict(sorted(v.items())) for k, v in cl_counts.items()},
            "composition": comp, "n_prompts_read_back": len(rows)}


# ============================================================================================== bank audit
# The declared change of each stratum against the base's canonical row: {factor: allowed level(s)}; every factor not
# listed must stay canonical. 'HO' = the bank's own half ('heldout_dev' / 'heldout_confirm'); 'HOW' = its words.
MIXED = ("posted", "train_word")
DECLARED = {
    "ID_posted": {}, "ID_void": {"status": ("void",)},
    "S_words": {"status": ("train_word",)},
    "S_surface_posted": {"para": ("train",), "event": ("train",), "before": ("train",),
                         "names": ("train_letters", "train_words"), "count": ("train",)},
    "S_surface_words": {"para": ("train",), "event": ("train",), "before": ("train",),
                        "names": ("train_letters", "train_words"), "count": ("train",), "status": ("train_word",)},
    "S_bal": {"bal": ("wide",)},
    "S_amt": {"amt": ("wide",)},
    "H_words": {"status": ("HOW",)},
    "H_order": {"top": ("reversed",), "event": ("HO",), "before": ("HO",), "status": MIXED},
    "H_order_top": {"top": ("reversed",), "status": MIXED},
    "H_order_inner": {"event": ("HO",), "before": ("HO",), "status": MIXED},
    "H_names": {"names": ("HO",), "status": MIXED},
    "H_count": {"count": ("heldout",), "names": ("canonical_plus_train_letters",), "status": MIXED},
    "H_para": {"para": ("HO",), "status": MIXED},
    "H_bal": {"bal": ("heldout",)},
    "H_amt": {"amt": ("heldout",)},
    "HS_words": {"para": ("HO",), "top": ("reversed",), "event": ("HO",), "before": ("HO",), "names": ("HO",),
                 "status": ("HOW",)},
    "HS_posted": {"para": ("HO",), "top": ("reversed",), "event": ("HO",), "before": ("HO",), "names": ("HO",)},
    "LB_void": {"top": ("reversed",), "event": ("heldout_confirm",), "before": ("heldout_confirm",), "names": ("lb",),
                "status": ("p3_word",)},
    "LB_posted": {"top": ("reversed",), "event": ("heldout_confirm",), "before": ("heldout_confirm",), "names": ("lb",)},
    "P3_words": {"status": ("p3_word",)},
}
for _s in C.P3.UNSEEN_STATUS_STRATA:
    DECLARED[_s] = {"status": ("p3_word",)}
MIXED_STRATA = tuple(s for s, d in DECLARED.items() if d.get("status") == MIXED)
CANON_LEVEL = {"para": "canonical", "top": "canonical", "event": "canonical", "before": "canonical", "names": "canonical",
               "count": "canonical", "status": "posted", "bal": "t2", "amt": "t2"}


def _resolve(levels: tuple, split: str) -> tuple:
    return tuple(f"heldout_{split}" if lv == "HO" else f"heldout_{split}_word" if lv == "HOW" else lv for lv in levels)


def audit_bank(rows: list, strata: tuple, split: str = "dev") -> dict:
    probs = []
    ids = {r["id"] for r in rows}
    per = Counter(r["stratum"] for r in rows)
    for r in rows:
        prompt = C.render_chat(C.messages(r["before"], r["event"], r["surface"]))
        try:
            rb = read_back(prompt)
        except (ValueError, Dup) as e:
            probs.append(f"{r['id']}: unreadable ({e})")
            continue
        cl = classify(rb)
        dec = DECLARED[r["stratum"]]
        for k, lv in CANON_LEVEL.items():
            allowed = _resolve(dec.get(k, (lv,)), split)
            if cl[k] not in allowed:
                probs.append(f"{r['id']}: {k} is {cl[k]}, declared {allowed}")
        if r["stratum"] in MIXED_STRATA:
            kind = C.WORD_KINDS[r["base"] % 5]
            want = "posted" if r["base"] % 2 == 0 else C.TRAIN_WORDS[kind][(r["base"] // 5) % len(C.TRAIN_WORDS[kind])]
            if rb["event"]["status"] != want:
                probs.append(f"{r['id']}: a mixed stratum's status is not posted (even base) / the base's S_words word (odd)")
        g = indep_gold(rb["before"], rb["event"])
        if json.loads(r["gold"]) != g:
            probs.append(f"{r['id']}: gold differs from the independent gold")
        if (r["gold_rule"] == "noop") != (r["event"]["status"] != "posted"):
            probs.append(f"{r['id']}: gold rule inconsistent with the status")
        if r["gold_rule"] == "apply" and g == r["before"]:
            probs.append(f"{r['id']}: an apply row whose gold equals its input")
        if r["twin"] not in ids:
            probs.append(f"{r['id']}: twin {r['twin']} missing")
    n_bases = len({r["base"] for r in rows})        # 100 for the dev bank and the registered map bank
    for s in strata:
        want = n_bases
        if s in C.AT_ONLY:
            want = sum(1 for r in rows if r["stratum"] == "ID_posted" and r["event"]["kind"] != "scale")
        if per.get(s, 0) != want:
            probs.append(f"stratum {s}: {per.get(s, 0)} rows != {want}")
    extra = sorted(set(per) - set(strata))
    if extra:
        probs.append(f"undeclared strata {extra}")
    return {"problems": probs, "rows": len(rows), "per_stratum": dict(per), "split": split}


def partition_problems() -> list:
    """The partition tables themselves: every pair of partitions disjoint (exactly and after lower-casing, except the
    lower-case form 'void' itself), the two held-out halves disjoint, names, paraphrases and orders consistent."""
    probs = []
    tw = set(C.TRAIN_WORD_LIST) | {"void"}
    hd, hc, pw = set(C.HELDOUT_WORD_LIST_DEV), set(C.HELDOUT_WORD_LIST_CONFIRM), set(C.P3_WORDS)
    sets = {"train": tw, "heldout_dev": hd, "heldout_confirm": hc, "prereg3": pw}
    for (a, x), (b, y) in itertools.combinations(sets.items(), 2):
        if x & y:
            probs.append(f"status partitions {a} and {b} share {sorted(x & y)}")
        low = lambda s: {v.lower() for v in s if v.lower() != "void"}  # noqa: E731
        if low(x) & low(y):
            probs.append(f"status partitions {a} and {b} share a string after lower-casing")
    if any(x.lower() == "void" for x in hd | hc):
        probs.append("a held-out status lower-cases to the trained 'void'")
    if set(C.VOID_CASE_VARIANTS) & pw or not set(C.VOID_CASE_VARIANTS) <= tw:
        probs.append("the training case variants of 'void' collide with PREREG-3's reserved strings")
    if (tw | hd | hc | pw) & set(C.EXCLUDED_STATUS) or "posted" in (tw - {"void"}) | hd | hc | pw:
        probs.append("an excluded string or 'posted' is in a partition")
    for k in C.WORD_KINDS:
        if len(C.HELDOUT_WORDS_DEV[k]) != 4 or len(C.HELDOUT_WORDS_CONFIRM[k]) != 4:
            probs.append(f"held-out kind {k}: not 4 + 4 words")
    nm = {"train": set(C.TRAIN_LETTERS) | set(C.TRAIN_NAME_WORDS) | set(C.CANON_NAMES),
          "heldout_dev": set(C.HELDOUT_LETTERS_DEV) | set(C.HELDOUT_NAME_WORDS_DEV),
          "heldout_confirm": set(C.HELDOUT_LETTERS_CONFIRM) | set(C.HELDOUT_NAME_WORDS_CONFIRM), "lb": set(C.LB_NAMES)}
    for (a, x), (b, y) in itertools.combinations(nm.items(), 2):
        if x & y:
            probs.append(f"name partitions {a} and {b} share {sorted(x & y)}")
    allnames = set().union(*nm.values())
    fields = set(C.EVENT_KEYS) | {"before", "event", "add", "scale", "transfer", "posted"}
    if allnames & (fields | tw | hd | hc | pw):
        probs.append("a name collides with a field, kind or status string")
    if len(set(C.PARAPHRASES)) != len(C.PARAPHRASES) or set(C.TRAIN_PARA) & {4, 5} or \
            (C.HELDOUT_PARA_DEV, C.HELDOUT_PARA_CONFIRM) != (4, 5):
        probs.append("paraphrase partition inconsistent")
    if set(DEV_IDX) & set(CONFIRM_IDX) or sorted(DEV_IDX + CONFIRM_IDX) != list(range(6)) or \
            (C.DEV_PERM_IDX, C.CONFIRM_PERM_IDX) != (DEV_IDX, CONFIRM_IDX):
        probs.append("key-order halves inconsistent")
    return probs


def audit(train_rows: list, dev_rows: list | None) -> dict:
    probs = []
    if HELD != C.HELDOUT_PERMS:
        probs.append("t2bl_common's held-out key orders differ from their independent regeneration")
    tok = token_audit()
    probs += [f"token audit: {p}" for p in tok["problems"]]
    probs += partition_problems()
    tr = audit_train(train_rows)
    probs += tr["problems"][:200]
    dv = audit_bank(dev_rows, C.DEV_STRATA, "dev") if dev_rows is not None else {"problems": [], "rows": 0}
    probs += [f"dev: {p}" for p in dv["problems"][:200]]
    if dev_rows is not None:
        tp = {C.payload(r["before"], r["event"]) for r in train_rows}
        shared = sum(C.payload(r["before"], r["event"]) in tp for r in dev_rows)
        if shared:
            probs.append(f"{shared} dev rows share a payload with the training set")
    perms = {f"before{n}": len(list(itertools.permutations(range(n)))) for n in range(4, 9)}
    return {"ok": not probs, "problems": probs, "train": {k: v for k, v in tr.items() if k != "problems"},
            "dev": {k: v for k, v in dv.items() if k != "problems"}, "heldout_orders_regenerated": True,
            "token_audit": tok, "n_train_perms_before": {k: v - 6 for k, v in perms.items()}}
