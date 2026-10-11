"""T2b-ledger data audit: an INDEPENDENT read-back of every prompt TEXT (torch-free, model-free).

It does not use the rows' recorded surface, nor t2bl_common's renderer or class logic: each training prompt is split
with its own regular expression, the user turn is parsed with json.loads (object_pairs_hook keeps the literal key
order and rejects duplicate keys), the held-out key orders are re-derived from ORDER_SEED by this module's own copy of
the documented procedure, and every factor level is classified here from the partition tables (the definitions). The
gold is recomputed by an independent implementation of the frame's rule.

  audit(train_rows, dev_rows) -> {"ok", "problems", counts ...}

Training checks (binding): the composition (1,500 rows; posted 750 / non-posted 750; 300 canonical-block rows in the
exact canonical T2 frame; pair members share before, event fields other than status, and surface); the read-back of
every prompt equals the row's own payload and surface; NO held-out factor in any training prompt (paraphrase 4, the
reversed top order, a held-out event / before order, a held-out letter or word name, an 8-account object, a held-out,
PREREG-3 or excluded status, a 5-digit balance, a 4-digit amount, a scale factor outside the recorded marginal); none
of the held-out text fragments (paraphrase 4's phrases, '"status":"<w>"' for every held-out / PREREG-3 / excluded
string, '"<name>":' for every held-out name) in any prompt or target; every target equals the independent gold.
Dev checks (binding): every row is its stratum's declared factor change of its base and nothing else (classified from
the rendered text), its gold is the independent gold, its twin exists.
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


def regen_heldout() -> dict:
    """Independent copy of the documented procedure (t2bl_common.derive_heldout_orders): random.Random(ORDER_SEED);
    event: the reversal, then shuffles of range(6) until 2 new permutations that are neither identity nor reversal;
    then before4..before8 the same way, in that order."""
    rng = random.Random(C.ORDER_SEED)
    out = {"top": [(1, 0)]}
    for name, n in [("event", 6)] + [(f"before{k}", k) for k in range(4, 9)]:
        lst = [tuple(reversed(range(n)))]
        while len(lst) < 3:
            p = list(range(n))
            rng.shuffle(p)
            t = tuple(p)
            if t != tuple(range(n)) and t not in lst:
                lst.append(t)
        out[name] = lst
    return out


HELD = regen_heldout()


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


def classify(rb: dict) -> dict:
    """Factor levels from a read-back (independent of t2bl_common.surface_classes)."""
    b, e = rb["before"], rb["event"]
    names = set(b)
    n = len(b)
    out = {}
    out["para"] = "unknown" if rb["para"] is None else "canonical" if rb["para"] == 0 else \
        "heldout" if rb["para"] == C.HELDOUT_PARA else "train"
    out["top"] = "canonical" if rb["top"] == ["before", "event"] else "heldout"
    ep = _perm(rb["event_order"], C.EVENT_KEYS) if sorted(rb["event_order"]) == sorted(C.EVENT_KEYS) else None
    out["event"] = "malformed" if ep is None else "canonical" if ep == tuple(range(6)) else \
        "heldout" if ep in HELD["event"] else "train"
    bp = _perm(rb["before_order"], b)
    key = f"before{n}"
    out["before"] = "canonical" if bp == tuple(range(n)) else "heldout" if key in HELD and bp in HELD[key] else \
        "train" if key in HELD else "other"
    lt, ld = set(C.TRAIN_LETTERS), set(C.CANON_NAMES)
    if names <= ld | lt and ld <= names:
        out["names"] = "canonical" if names == ld else "canonical_plus_train_letters"
    elif names <= lt:
        out["names"] = "train_letters"
    elif names <= set(C.TRAIN_NAME_WORDS):
        out["names"] = "train_words"
    elif names & (set(C.HELDOUT_LETTERS) | set(C.HELDOUT_NAME_WORDS)):
        out["names"] = "heldout"
    else:
        out["names"] = "other"
    out["count"] = "canonical" if n == 4 else "train" if n in (5, 6, 7) else "heldout" if n == 8 else "other"
    st = e.get("status")
    if st in ("posted", "void"):
        out["status"] = st
    elif st in C.TRAIN_WORD_LIST:
        out["status"] = "train_word"
    elif st in C.HELDOUT_WORD_LIST:
        out["status"] = "heldout_word"
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


HELDOUT_LEVEL = {"para": ("heldout", "unknown"), "top": ("heldout",), "event": ("heldout", "malformed"),
                 "before": ("heldout", "other"), "names": ("heldout", "other"), "count": ("heldout", "other"),
                 "status": ("heldout_word", "p3_word", "excluded", "other"), "bal": ("heldout",), "amt": ("heldout",)}


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
    frags = {f"paraphrase 4: {f!r}": f for f in C.HELDOUT_PARA_FRAGMENTS}
    for w in C.HELDOUT_WORD_LIST + C.P3_WORDS + C.EXCLUDED_STATUS:
        frags[f"status {w!r}"] = '"status":' + json.dumps(w)
    for nm in C.HELDOUT_LETTERS + C.HELDOUT_NAME_WORDS:
        frags[f"name {nm!r} as a key"] = json.dumps(nm) + ":"
    return frags


def _canon_msgs_ok(r: dict) -> bool:
    from rcwmt import env
    msgs = env.one_step_messages(C.LEDGER, r["before"], r["event"])
    return C.render_chat(msgs) == r["prompt"]


def audit_train(rows: list) -> dict:
    probs = []
    hits, frag_hits = Counter(), Counter()
    frags = fragments()
    cl_counts: dict = {}
    n_fully_canon = 0
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
        if json.loads(r["target"]) != indep_gold(rb["before"], rb["event"]) or \
                r["target"] != json.dumps(indep_gold(rb["before"], rb["event"]), separators=(",", ":"), sort_keys=True):
            probs.append(f"{r['id']}: target differs from the independent gold")
        canon = all(cl[k] == v for k, v in (("para", "canonical"), ("top", "canonical"), ("event", "canonical"),
                                            ("before", "canonical"), ("names", "canonical"), ("bal", "t2"), ("amt", "t2")))
        canon = canon and cl["status"] in ("posted", "void")
        n_fully_canon += canon
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
            "fully_canonical_rows": n_fully_canon}
    want = {"rows": 1500, "posted": 750, "nonposted": 750, "canon_block": 300, "pairs": 400}
    for k, v in want.items():
        if comp[k] != v:
            probs.append(f"composition {k} = {comp[k]} != {v}")
    if len({C.payload(r["before"], r["event"]) for r in rows}) != len(rows):
        probs.append("duplicate training payloads")
    return {"problems": probs, "heldout_factor_hits": dict(hits), "fragment_hits": dict(frag_hits),
            "n_fragments_checked": len(frags), "classes": {k: dict(sorted(v.items())) for k, v in cl_counts.items()},
            "composition": comp, "n_prompts_read_back": len(rows)}


# the declared change of each stratum, against the base's canonical row: {factor: required level(s)}; every factor not
# listed must stay canonical (for S_surface_* the listed factors must be TRAINING levels).
DECLARED = {
    "ID_posted": {}, "ID_void": {"status": ("void",)},
    "S_words": {"status": ("train_word",)},
    "S_surface_posted": {"para": ("train",), "event": ("train",), "before": ("train",),
                         "names": ("train_letters", "train_words"), "count": ("canonical", "train"), "bal": ("wide",),
                         "amt": ("wide", "t2")},
    "S_surface_words": {"para": ("train",), "event": ("train",), "before": ("train",),
                        "names": ("train_letters", "train_words"), "count": ("canonical", "train"), "bal": ("wide",),
                        "amt": ("wide", "t2"), "status": ("train_word",)},
    "H_words": {"status": ("heldout_word",)},
    "H_order": {"top": ("heldout",), "event": ("heldout",), "before": ("heldout",)},
    "H_names": {"names": ("heldout",)},
    "H_count": {"count": ("heldout",), "names": ("canonical_plus_train_letters",)},
    "H_para": {"para": ("heldout",), "status": ("posted", "void")},
    "H_bal": {"bal": ("heldout",)},
    "H_amt": {"amt": ("heldout",)},
    "HS_words": {"para": ("heldout",), "top": ("heldout",), "event": ("heldout",), "before": ("heldout",),
                 "names": ("heldout",), "status": ("heldout_word",)},
    "HS_posted": {"para": ("heldout",), "top": ("heldout",), "event": ("heldout",), "before": ("heldout",),
                  "names": ("heldout",)},
    "LB_void": {"top": ("heldout",), "event": ("heldout",), "before": ("heldout",), "names": ("heldout",),
                "status": ("p3_word",)},
    "LB_posted": {"top": ("heldout",), "event": ("heldout",), "before": ("heldout",), "names": ("heldout",)},
    "P3_words": {"status": ("p3_word",)},
}
for _s in C.P3.UNSEEN_STATUS_STRATA:
    DECLARED[_s] = {"status": ("p3_word",)}
CANON_LEVEL = {"para": "canonical", "top": "canonical", "event": "canonical", "before": "canonical", "names": "canonical",
               "count": "canonical", "status": "posted", "bal": "t2", "amt": "t2"}


def audit_bank(rows: list, strata: tuple) -> dict:
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
            allowed = dec.get(k, (lv,))
            if cl[k] not in allowed:
                probs.append(f"{r['id']}: {k} is {cl[k]}, declared {allowed}")
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
    return {"problems": probs, "rows": len(rows), "per_stratum": dict(per)}


def audit(train_rows: list, dev_rows: list | None) -> dict:
    probs = []
    if HELD != C.HELDOUT_PERMS:
        probs.append("t2bl_common's held-out key orders differ from their independent regeneration")
    tr = audit_train(train_rows)
    probs += tr["problems"][:200]
    dv = audit_bank(dev_rows, C.DEV_STRATA) if dev_rows is not None else {"problems": [], "rows": 0}
    probs += [f"dev: {p}" for p in dv["problems"][:200]]
    if dev_rows is not None:
        tp = {C.payload(r["before"], r["event"]) for r in train_rows}
        shared = sum(C.payload(r["before"], r["event"]) in tp for r in dev_rows)
        if shared:
            probs.append(f"{shared} dev rows share a payload with the training set")
    # vocabulary partition sanity (exact and case-insensitive disjointness; no excluded strings anywhere)
    tw, hw, pw = set(C.TRAIN_WORD_LIST) | {"void"}, set(C.HELDOUT_WORD_LIST), set(C.P3_WORDS)
    if tw & hw or tw & pw or hw & pw:
        probs.append("status partitions are not disjoint")
    # case-insensitive disjointness, except the lower-case form 'void' itself: the training case variants of the trained
    # 'void' (VOId, vOid) and PREREG-3's reserved Void / VOID share it by design; no HELD-OUT string lower-cases to it
    low = lambda s: {x.lower() for x in s if x.lower() != "void"}  # noqa: E731
    if (low(tw) & low(hw)) or (low(tw) & low(pw)) or (low(hw) & low(pw)):
        probs.append("status partitions are not case-insensitively disjoint")
    if any(x.lower() == "void" for x in hw):
        probs.append("a held-out status lower-cases to the trained 'void'")
    if set(C.VOID_CASE_VARIANTS) & set(C.P3_WORDS) or not set(C.VOID_CASE_VARIANTS) <= tw:
        probs.append("the training case variants of 'void' collide with PREREG-3's reserved strings")
    if (tw | hw | pw) & set(C.EXCLUDED_STATUS) or "posted" in (tw - {"void"}) | hw | pw:
        probs.append("an excluded string or 'posted' is in a partition")
    nm_t = set(C.TRAIN_LETTERS) | set(C.TRAIN_NAME_WORDS) | set(C.CANON_NAMES)
    nm_h = set(C.HELDOUT_LETTERS) | set(C.HELDOUT_NAME_WORDS)
    if nm_t & nm_h:
        probs.append("name partitions are not disjoint")
    fields = set(C.EVENT_KEYS) | {"before", "event", "add", "scale", "transfer", "posted"}
    if (nm_t | nm_h) & (fields | tw | hw | pw):
        probs.append("a name collides with a field, kind or status string")
    perms = {f"before{n}": len(list(itertools.permutations(range(n)))) for n in range(4, 9)}
    return {"ok": not probs, "problems": probs, "train": {k: v for k, v in tr.items() if k != "problems"},
            "dev": {k: v for k, v in dv.items() if k != "problems"}, "heldout_orders_regenerated": True,
            "n_train_perms_before": {k: v - 3 for k, v in perms.items()}}
