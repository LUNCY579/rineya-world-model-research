"""Registered CPU replay of every logged world-model (F) query (research/PREREG_FREPLAY.md; F-09).

  python -B scripts/freplay.py selftest     # synthetic unit cases only (classifier, Clopper-Pearson); reads no run data
  python -B scripts/freplay.py run          # the registered replay -> runs/FREPLAY/FREPLAY.json + FREPLAY.md

Pure Python on bwmn/blocks.py (exact simulator B.step, planner parser B.parse_forward_delta). No model, no torch: the
pinned bwmn/blocks_planner.py is NOT imported (it imports bwmn.lm). Every input sha256 is pinned below (= the prereg's
table) and checked before anything is read; a mismatch aborts. Every invocation is appended to runs/FREPLAY/RUN_LOG.txt.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402

OUT = ROOT / "runs" / "FREPLAY"
PREREG = ROOT / "research" / "PREREG_FREPLAY.md"
PREREG_SHA = "994d74c0b721724a49cd24774b5ecf0fc12027d058718575f276b100b3e975f8"
BLOCKS_SHA = "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83"
PLANNER_SHA = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"  # semantics reference only

# set name -> (run dir, shards (0 = single cache file), instance source kind)
SETS = [
    ("b2_hard", "runs/BW_TEST_B2/hard_beam", 8, "hard"),
    ("b2_mirror", "runs/BW_TEST_B2/mirror_beam", 8, "mirror"),
    ("b2_pb602", "runs/BW_TEST_B2/planbench_beam", 4, "planbench"),
    ("h2_hard", "runs/BW_B3_HARD/h2_beam", 8, "hard"),
    ("distill_r1", "runs/BW_DISTILL_r1", 0, "distill"),
]
FCACHE_PINS = {  # path -> (lines, sha256), exactly the prereg §1 table
    "runs/BW_TEST_B2/hard_beam/shard_0of8/fcache.jsonl": (2417, "c84563c20fec05028b1a28a271848a6081b98c54f919bf379c242d44c07283f1"),
    "runs/BW_TEST_B2/hard_beam/shard_1of8/fcache.jsonl": (2551, "8931869e07489758ebc2df55855691367ac6690de6dc5562922ff7a06c41a73c"),
    "runs/BW_TEST_B2/hard_beam/shard_2of8/fcache.jsonl": (2534, "54db54c9ab01fd753b494376959b630d8bec7fb1464052c9b8dcc5a43d3ad170"),
    "runs/BW_TEST_B2/hard_beam/shard_3of8/fcache.jsonl": (2493, "ba4868f2b19fb708457e1bda95858f7ddc380b1accb39049f918e15d43ea7123"),
    "runs/BW_TEST_B2/hard_beam/shard_4of8/fcache.jsonl": (2685, "18b5135e49756cd46c92aef130797f2727c76d3a0aa6eb72cccca3647b5776c3"),
    "runs/BW_TEST_B2/hard_beam/shard_5of8/fcache.jsonl": (3149, "5d2f833fa7e535294df0c8f5f26457cfdbbee80c82fc350f9aeb2defefb15f00"),
    "runs/BW_TEST_B2/hard_beam/shard_6of8/fcache.jsonl": (2130, "7baf775dbe026ac1675a9eecbcbdcc718151fbaee860824c931572e411ec7098"),
    "runs/BW_TEST_B2/hard_beam/shard_7of8/fcache.jsonl": (2595, "8481815e1735f2f01b9565eb4aafa552f8c2b39f7fa2edaf597913d2606b6ce1"),
    "runs/BW_TEST_B2/mirror_beam/shard_0of8/fcache.jsonl": (3014, "54a71e26ecc30963ba38a26444c4adfaa95aed317845632adf85e77fa9342df3"),
    "runs/BW_TEST_B2/mirror_beam/shard_1of8/fcache.jsonl": (2647, "0d3380fb00276c27c842605821a72e57b70bc9c51334b5d8962e8223c5142e41"),
    "runs/BW_TEST_B2/mirror_beam/shard_2of8/fcache.jsonl": (2356, "1b5568e43421f0d636b9b402d937345df9e8d28ab9f8453c7302334d88da59c0"),
    "runs/BW_TEST_B2/mirror_beam/shard_3of8/fcache.jsonl": (2747, "e70a17d9d058f06d6b73d3378cffb956efdcd832a1be95474a3f70878d52c7d7"),
    "runs/BW_TEST_B2/mirror_beam/shard_4of8/fcache.jsonl": (3686, "99416006d1f298fc29fd9cbe72fecd48c97771437e7869acc3fed22b08261a08"),
    "runs/BW_TEST_B2/mirror_beam/shard_5of8/fcache.jsonl": (3082, "0fdfe929af2ec772909da5c66d26bdfddfca5d68f01a00b31e411950dfec7421"),
    "runs/BW_TEST_B2/mirror_beam/shard_6of8/fcache.jsonl": (3378, "6a2e8f086c94f7b5fa83007c89fe7b52687b4b0d644388f1e03d8ce2dd5812c7"),
    "runs/BW_TEST_B2/mirror_beam/shard_7of8/fcache.jsonl": (2842, "97563de12c847669849be906b93db169ad9da6961751c85f20f9a1b3654ad600"),
    "runs/BW_TEST_B2/planbench_beam/shard_0of4/fcache.jsonl": (466, "2c9cb1f6553423ecf613ec7fe225190ba687b92ad38a44912ccdd99f86f033c4"),
    "runs/BW_TEST_B2/planbench_beam/shard_1of4/fcache.jsonl": (424, "1fd852d9e16d0c23a2d54cea1cd8ae23d4f3ca121ad835ab034a3edf1baddfbf"),
    "runs/BW_TEST_B2/planbench_beam/shard_2of4/fcache.jsonl": (454, "7c115df45577c3222cb7235a66857f450cf6b8515f988770172d6b4c96cbcc75"),
    "runs/BW_TEST_B2/planbench_beam/shard_3of4/fcache.jsonl": (442, "f8ca5d7190c6da98aa08b0aa753090140ea2dfc69a88852003aec162b7c51287"),
    "runs/BW_B3_HARD/h2_beam/shard_0of8/fcache.jsonl": (2642, "c7b9dfb605868e53f1d0b6bac6068a80e5bbc87779bf53f44ec89e51b809e58a"),
    "runs/BW_B3_HARD/h2_beam/shard_1of8/fcache.jsonl": (3036, "28c5e6ee0d08ebd4a0895add7e2a88f5a6d98dce5ccc87edca17042242b8ca4b"),
    "runs/BW_B3_HARD/h2_beam/shard_2of8/fcache.jsonl": (2988, "13ccb217f8a54a56bb15c7035cb9e643a55ca0290d38885b162ae657cb2c2717"),
    "runs/BW_B3_HARD/h2_beam/shard_3of8/fcache.jsonl": (2890, "f1f2bf098263f283bde582aec94290a0f4bee0ab3f04852884b72f87a710054a"),
    "runs/BW_B3_HARD/h2_beam/shard_4of8/fcache.jsonl": (2919, "52b312b8722e2e1476cf4a82fdc8093c1895687229d4c50076859d8d0753bb4f"),
    "runs/BW_B3_HARD/h2_beam/shard_5of8/fcache.jsonl": (3057, "39c361eed0eff68fa78b3820f2f1cad957a6465f2e2db16401e6427b17275cc0"),
    "runs/BW_B3_HARD/h2_beam/shard_6of8/fcache.jsonl": (2395, "5b5418bd565da7f4036ace70bda362188cf28fad2cd918b6ec66f56809dd5bab"),
    "runs/BW_B3_HARD/h2_beam/shard_7of8/fcache.jsonl": (2983, "2c354964f8f58556313482fdf96d9708486d7121659db2150aeadbf7cf6bd998"),
    "runs/BW_DISTILL_r1/fcache.jsonl": (15030, "055d3b667f74e1675f92c0ea9855d300b08fbd99f67c6acb40ac62b0e7b78b46"),
}
OUTCOME_PINS = {  # merged outcome files and instance sources (prereg §1)
    "runs/BW_TEST_B2/hard_beam/plans.jsonl": "85812972a2e5e8bcedee245673dee98a0f0fdb93f02e2e852e44fa7c037fb3ee",
    "runs/BW_TEST_B2/mirror_beam/plans.jsonl": "22147f0750f7ceb1992504b20a3d44999d06ce48af357282ffb2b151ab913cfd",
    "runs/BW_TEST_B2/planbench_beam/plans.jsonl": "839142ab787be6d573c43c4f8876195b48f322dc3e3b4cbe2d2f2a45bf062db1",
    "runs/BW_B3_HARD/h2_beam/plans.jsonl": "3cf3cd01b8a73f3e036fda126a5d6b562b38f63e458197a188639f2df6e2b91b",
    "runs/BW_DISTILL_r1/solutions.jsonl": "fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c",
    "runs/BW_val_large/instances_mirror_hard.json": "2f61e38bf2f99e2c5f7524d8642618d51d1e2d83573123524b3d8e41155d53f2",
}
DIR_PINS = {  # directory digest: sha256 over "<name> <sha256>\n" lines sorted by file name (prereg §1)
    "bench/planbench/blocksworld_hard/generated": (110, "d7b7e446c1cea76da344830ab750138d89a44d311de0b8d0fb4a629018100b93"),
    "bench/planbench/blocksworld/generated_basic_3": (101, "46aceac767835c09360801823b4ac8ac0d2076e9b88120408c5bcc01bfccd3fe"),
    "bench/planbench/blocksworld/generated_basic": (501, "4807eb8bd6cb17d8eacdfc3fa0b05e49c79d4f4de0b5684ea0f566c4024d565e"),
}

CLASSES = ["ok", "false_illegal", "false_legal_accepted", "false_legal_rejected_by_support_check", "wrong_state",
           "malformed"]
SUBTYPES = ["ok_legal_exact", "ok_illegal_flag", "ok_illegal_flag_unsupported", "wrong_state", "fi_flag",
            "fi_support", "fi_malformed", "false_legal_accepted", "false_legal_rejected_by_support_check", "malformed"]
HARMFUL = ("false_illegal", "false_legal_accepted", "wrong_state")
SEVERITY = ["false_legal_accepted", "wrong_state", "false_illegal", "false_legal_rejected_by_support_check",
            "malformed", "ok"]  # conflict rule (prereg §2): most severe first
ACTIONS = ["pick-up", "put-down", "stack", "unstack"]
R2_THRESHOLD = 0.05
CLAIMS = {  # prereg §7: claim -> its own set
    "b2_hard": "P-08/P-10 hard: 'every plan S-beam verified in imagination was valid'",
    "b2_mirror": "P-10 mirror verification statement",
    "b2_pb602": "P-10 PB-602 verification statement",
    "h2_hard": "P-14: 'every plan the beam verified in imagination was valid (88/88)'",
    "distill_r1": "P-11 / B3: label precision 1.000 (211/211), 'no simulator labels'",
}
EXPLORATORY = {  # F-09 / panel scratch numbers (prereg §0), for the registered-vs-exploratory table only
    "b2_hard": {"unique_keys": 20530, "false_legal_accepted": 0, "wrong_state": 0, "false_illegal": 6,
                "flag_legal_on_illegal": 798, "illegal_keys": 4066},
    "h2_hard": {"unique_keys": 22895, "false_legal_accepted": 0, "wrong_state": 0, "false_illegal": 5,
                "flag_legal_on_illegal": 657, "illegal_keys": 2363},
    "b2_mirror": {"flag_legal_on_illegal": 1078, "illegal_keys": 4498},
    "b2_pb602": {"flag_legal_on_illegal": 10, "illegal_keys": 77},
    "distill_r1": {"flag_legal_share_approx": 0.22},
}


# ----------------------------------------------------------------------------- statistics

def _betacf(a: float, b: float, x: float) -> float:
    fpmin, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, 20000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return None
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1 - alpha / 2, k + 1, n - k)
    return [round(lo, 6), round(hi, 6)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "p": None if n == 0 else round(k / n, 6), "ci95": clopper_pearson(k, n)}


# ----------------------------------------------------------------------------- classification (prereg §3)

def schema_parse(text):
    """parse_forward_delta's steps WITHOUT the support condition: (add, dele, flag) or None."""
    try:
        d = json.loads(text.strip())
        if set(d) != {"add", "del", "legal"} or type(d["legal"]) is not bool:
            return None
        add = {tuple(x.split()) for x in d["add"]}
        dele = {tuple(x.split()) for x in d["del"]}
        return frozenset(add), frozenset(dele), d["legal"]
    except Exception:
        return None


def classify(s: frozenset, a: tuple, text) -> dict:
    ns, legal = B.step(s, a)
    p = B.parse_forward_delta(text, s)
    q = schema_parse(text)
    # consistency check (asserted for every key)
    if q is None:
        assert p is None, ("schema None but parser not None", s, a, text)
    else:
        add, dele, flag = q
        assert (p is None) == (not dele <= s), ("support mismatch", s, a, text)
        if p is not None:
            assert p == (frozenset((s - dele) | add), flag), ("parser value mismatch", s, a, text)
    if legal:
        if p is not None and p[1]:
            cls = sub = "ok" if p[0] == ns else "wrong_state"
            if cls == "ok":
                sub = "ok_legal_exact"
        else:
            cls = "false_illegal"
            sub = "fi_malformed" if q is None else ("fi_flag" if q[2] is False else "fi_support")
    else:
        if p is not None and p[1]:
            cls = sub = "false_legal_accepted"
        elif p is not None:
            cls, sub = "ok", "ok_illegal_flag"
        elif q is None:
            cls = sub = "malformed"
        elif q[2] is False:
            cls, sub = "ok", "ok_illegal_flag_unsupported"
        else:
            cls = sub = "false_legal_rejected_by_support_check"
    # second code path: planner-effective outcome vs ExactF outcome
    eff = "stop" if (p is None or not p[1]) else ("go", p[0])
    exact = ("go", ns) if legal else "stop"
    assert (eff != exact) == (cls in HARMFUL), ("second code path disagrees", s, a, text, cls)
    return {"cls": cls, "sub": sub, "legal": legal, "accepted": p is not None and p[1],
            "next": p[0] if (p is not None and p[1]) else None,
            "flag_known": q is not None, "flag_correct": q is not None and q[2] == legal,
            "delta_exact": q is not None and q[2] == legal and q[0] == ns - s and q[1] == s - ns}


# ----------------------------------------------------------------------------- inputs

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dir_digest(d: Path):
    fs = sorted(d.glob("*.pddl"), key=lambda p: p.name)
    lines = "".join(f"{p.name} {sha256_file(p)}\n" for p in fs)
    return len(fs), hashlib.sha256(lines.encode()).hexdigest()


def verify_pins() -> dict:
    bad, rec = [], {"fcache": {}, "outcomes": {}, "dirs": {}, "shard_plans": {}}
    if sha256_file(PREREG) != PREREG_SHA:
        bad.append(("prereg", sha256_file(PREREG)))
    if sha256_file(ROOT / "bwmn/blocks.py") != BLOCKS_SHA:
        bad.append(("bwmn/blocks.py", sha256_file(ROOT / "bwmn/blocks.py")))
    planner_sha = sha256_file(ROOT / "bwmn/blocks_planner.py")
    if planner_sha != PLANNER_SHA:
        bad.append(("bwmn/blocks_planner.py", planner_sha))
    for rel, (nl, sha) in FCACHE_PINS.items():
        p = ROOT / rel
        got = sha256_file(p)
        lines = len(p.read_text(encoding="utf-8").splitlines())
        rec["fcache"][rel] = {"sha256": got, "lines": lines, "bytes": p.stat().st_size}
        if got != sha or lines != nl:
            bad.append((rel, got, lines))
    for rel, sha in OUTCOME_PINS.items():
        got = sha256_file(ROOT / rel)
        rec["outcomes"][rel] = got
        if got != sha:
            bad.append((rel, got))
    for rel, (nf, sha) in DIR_PINS.items():
        n, got = dir_digest(ROOT / rel)
        rec["dirs"][rel] = {"files": n, "digest": got}
        if (n, got) != (nf, sha):
            bad.append((rel, n, got))
    for name, d, k, _ in SETS:
        for i in range(k):
            rel = f"{d}/shard_{i}of{k}/plans.jsonl"
            rec["shard_plans"][rel] = sha256_file(ROOT / rel)
    if bad:
        raise SystemExit(f"pinned input mismatch, abort: {bad}")
    rec.update(prereg_sha256=PREREG_SHA, blocks_sha256=BLOCKS_SHA, planner_sha256_reference=planner_sha)
    return rec


def set_files(d: str, k: int) -> list[str]:
    return [f"{d}/fcache.jsonl"] if k == 0 else [f"{d}/shard_{i}of{k}/fcache.jsonl" for i in range(k)]


def read_cache(rel: str):
    """As LMForwardBW loads its cache: read_text().splitlines(); torn lines skipped; last line per key wins."""
    memo, torn, lines, dup_same, dup_diff = {}, 0, 0, 0, 0
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        lines += 1
        try:
            r = json.loads(line)
        except Exception:
            torn += 1
            continue
        k = (frozenset(tuple(f) for f in r["s"]), tuple(r["a"]))
        if k in memo:
            if memo[k] == r["text"]:
                dup_same += 1
            else:
                dup_diff += 1
        memo[k] = r["text"]
    return memo, {"lines": lines, "torn_lines": torn, "keys": len(memo), "dup_lines_same_text": dup_same,
                  "dup_lines_diff_text": dup_diff}


def fact_s(f) -> str:
    return " ".join(f)


def key_json(k) -> dict:
    s, a = k
    return {"s": sorted(fact_s(f) for f in s), "a": " ".join(a)}


def key_id(k) -> str:
    return hashlib.sha256(json.dumps(key_json(k), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- aggregation

class Memo:
    def __init__(self):
        self.c = {}

    def get(self, k, text):
        t = (k, text)
        if t not in self.c:
            self.c[t] = classify(k[0], k[1], text)
        return self.c[t]


def build_units(files: list[str], caches: dict, memo: Memo):
    """Unique keys over files (prereg §2). Returns units {key: (result, text, n_distinct_texts)}, conflicts, occ."""
    occ = {}
    for fi, rel in enumerate(files):
        for k, text in caches[rel][0].items():
            occ.setdefault(k, []).append((fi, text))
    units, conflicts = {}, []
    occ_counts = Counter()
    for k, lst in occ.items():
        for fi, text in lst:
            occ_counts[memo.get(k, text)["cls"]] += 1
        texts = {}
        for fi, text in lst:
            texts.setdefault(text, fi)
        cands = sorted(((SEVERITY.index(memo.get(k, t)["cls"]), fi, t) for t, fi in texts.items()))
        _, _, text = cands[0]
        units[k] = (memo.get(k, text), text, len(texts))
        if len(texts) > 1:
            conflicts.append({"key_id": key_id(k), "key": key_json(k), "files": sorted({files[fi] for fi, _ in lst}),
                              "classes": sorted({memo.get(k, t)["cls"] for t in texts}),
                              "unit_class": units[k][0]["cls"]})
    conflicts.sort(key=lambda x: x["key_id"])
    return units, conflicts, {"occurrences": sum(occ_counts.values()), "by_class": {c: occ_counts[c] for c in CLASSES}}


def new_cell() -> Counter:
    return Counter()


def add_to_cell(c: Counter, r: dict):
    c["keys"] += 1
    c["legal" if r["legal"] else "illegal"] += 1
    c["cls:" + r["cls"]] += 1
    c["sub:" + r["sub"]] += 1
    c["harmful"] += r["cls"] in HARMFUL
    c["flag_known"] += r["flag_known"]
    c["flag_correct"] += r["flag_correct"]
    c["delta_exact"] += r["delta_exact"]


def render_cell(c: Counter) -> dict:
    il = c["illegal"]
    return {
        "keys": c["keys"], "legal": c["legal"], "illegal": il,
        "classes": {x: c["cls:" + x] for x in CLASSES},
        "subtypes": {x: c["sub:" + x] for x in SUBTYPES},
        "rates": {
            "harmful": rate(c["harmful"], c["keys"]),
            "false_illegal": rate(c["cls:false_illegal"], c["legal"]),
            "wrong_state": rate(c["cls:wrong_state"], c["legal"]),
            "false_legal_accepted": rate(c["cls:false_legal_accepted"], il),
            "support_share_R2": rate(c["cls:false_legal_rejected_by_support_check"], il),
            "flag_false_legal": rate(c["cls:false_legal_accepted"] + c["cls:false_legal_rejected_by_support_check"], il),
            "malformed": rate(c["cls:malformed"], c["keys"]),
        },
        "secondary": {"flag_accuracy": rate(c["flag_correct"], c["flag_known"]),
                      "delta_exact": rate(c["delta_exact"], c["keys"])},
    }


def strata(units: dict) -> dict:
    tot, by_n, by_a, by_na = new_cell(), {}, {}, {}
    for k, (r, _, _) in units.items():
        n, act = len(B.blocks_of(k[0])), k[1][0]
        add_to_cell(tot, r)
        add_to_cell(by_n.setdefault(n, new_cell()), r)
        add_to_cell(by_a.setdefault(act, new_cell()), r)
        add_to_cell(by_na.setdefault(f"{n}|{act}", new_cell()), r)
    return {"total": render_cell(tot),
            "by_blocks": {str(n): render_cell(by_n[n]) for n in sorted(by_n)},
            "by_action": {a: render_cell(by_a[a]) for a in sorted(by_a)},
            "by_blocks_action": {na: render_cell(by_na[na])
                                 for na in sorted(by_na, key=lambda x: (int(x.split("|")[0]), x.split("|")[1]))}}


# ----------------------------------------------------------------------------- instances and forests (prereg §5)

def load_instances(name: str, d: str, k: int, kind: str):
    if kind == "distill":
        rows = [json.loads(l) for l in (ROOT / d / "solutions.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        out = []
        for r in rows:
            out.append({"id": r["id"], "init": frozenset(map(tuple, r["init"])), "goal": frozenset(map(tuple, r["goal"])),
                        "n": len(r["objs"]), "success": bool(r["exact_valid_for_analysis_only"]),
                        "verified": bool(r["verified"]), "fail": None, "plan": [tuple(x) for x in r["plan"]],
                        "cache": f"{d}/fcache.jsonl"})
        return out
    shard_of = {}
    for i in range(k):
        for l in (ROOT / d / f"shard_{i}of{k}" / "plans.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                iid = json.loads(l)["id"]
                assert iid not in shard_of, ("instance in two shards", iid)
                shard_of[iid] = f"{d}/shard_{i}of{k}/fcache.jsonl"
    mirror = json.loads((ROOT / "runs/BW_val_large/instances_mirror_hard.json").read_text()) if kind == "mirror" else None
    out = []
    for l in (ROOT / d / "plans.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        iid = r["id"]
        if kind == "hard":
            objs, init, goal = B.parse_problem(B.BENCH / "blocksworld_hard" / "generated" / (iid.split("/")[1] + ".pddl"))
        elif kind == "planbench":
            sub, stem = iid.split("/")
            objs, init, goal = B.parse_problem(B.BENCH / "blocksworld" / sub / (stem + ".pddl"))
        else:
            m = mirror[int(iid.split("/")[1])]
            objs, init, goal = m["objs"], frozenset(map(tuple, m["init"])), frozenset(map(tuple, m["goal"]))
        assert len(objs) == r["n_blocks"], ("block count mismatch", iid)
        out.append({"id": iid, "init": init, "goal": goal, "n": len(objs), "success": bool(r["success"]),
                    "verified": bool(r["verified"]), "fail": r["fail"],
                    "plan": [tuple(x.split()) for x in r["plan"]], "cache": shard_of[iid]})
    assert len(shard_of) == len(out), ("shard ids vs merged ids", len(shard_of), len(out))
    return out


def forest(init: frozenset, by_state: dict, cache: dict, memo: Memo):
    seen, dq, keys = {init}, deque([init]), []
    while dq:
        s = dq.popleft()
        for a in by_state.get(s, ()):
            k = (s, a)
            r = memo.get(k, cache[k])
            keys.append((k, r))
            if r["accepted"] and r["next"] not in seen:
                seen.add(r["next"])
                dq.append(r["next"])
    return seen, keys


def coverage(it: dict, cache: dict, memo: Memo) -> dict:
    s, steps_ok = it["init"], 0
    for a in it["plan"]:
        k = (s, a)
        if k not in cache:
            return {"covered": False, "steps_ok": steps_ok, "reason": "step_not_cached", "goal_at_end": None}
        r = memo.get(k, cache[k])
        if not r["accepted"]:
            return {"covered": False, "steps_ok": steps_ok, "reason": "step_not_F_accepted", "goal_at_end": None}
        s, steps_ok = r["next"], steps_ok + 1
    return {"covered": True, "steps_ok": steps_ok, "reason": None, "goal_at_end": B.goal_met(s, it["goal"])}


def instances_block(name: str, d: str, k: int, kind: str, caches: dict, memo: Memo):
    insts = load_instances(name, d, k, kind)
    idx = {}
    for rel in {it["cache"] for it in insts}:
        bs = {}
        for (s, a) in caches[rel][0]:
            bs.setdefault(s, []).append(a)
        idx[rel] = {s: sorted(v) for s, v in bs.items()}
    rows = []
    for it in insts:
        cache = caches[it["cache"]][0]
        states, keys = forest(it["init"], idx[it["cache"]], cache, memo)
        cc = Counter(r["cls"] for _, r in keys)
        harm = sorted(({"key_id": key_id(kk), "class": r["cls"], "subtype": r["sub"], "key": key_json(kk),
                        "text": cache[kk]} for kk, r in keys if r["cls"] in HARMFUL), key=lambda x: x["key_id"])
        cov = coverage(it, cache, memo)
        rows.append({"id": it["id"], "n_blocks": it["n"], "success": it["success"], "verified": it["verified"],
                     "fail": it["fail"], "cache": it["cache"], "forest_states": len(states), "forest_keys": len(keys),
                     "forest_classes": {c: cc[c] for c in CLASSES}, "flagged": bool(harm),
                     "support_exposed": cc["false_legal_rejected_by_support_check"] > 0, "harmful_keys": harm,
                     "coverage": cov})
    return rows


# ----------------------------------------------------------------------------- readings

def reading_r1(rows: list[dict]) -> dict:
    flagged = [r for r in rows if r["flagged"]]
    obs = sum(r["success"] for r in rows)
    L = sum(r["success"] for r in rows if not r["flagged"])
    U = L + len(flagged)
    return {"n_instances": len(rows), "observed_success": obs, "flagged": len(flagged),
            "flagged_ids": [r["id"] for r in flagged], "flagged_solved": sum(r["success"] for r in flagged),
            "exactF_success_lower_L": L, "exactF_success_upper_U": U,
            "oracle_equivalent_all": not flagged,
            "ceiling_at_most_observed": U == obs,
            "support_exposed_instances": sum(r["support_exposed"] for r in rows)}


def reading_r2(name: str, cell: dict) -> dict:
    r = cell["rates"]["support_share_R2"]
    trig = r["p"] is not None and r["p"] > R2_THRESHOLD
    return {"R2": r, "threshold": R2_THRESHOLD, "wording_change_triggered": trig, "claim": CLAIMS.get(name)}


# ----------------------------------------------------------------------------- run

def run():
    t0 = time.time()
    pins = verify_pins()
    caches = {rel: read_cache(rel) for rel in FCACHE_PINS}
    memo = Memo()
    core = {"prereg_sha256": PREREG_SHA, "sets": {}}
    all_files = []
    for name, d, k, kind in SETS:
        files = set_files(d, k)
        all_files += files
        units, conflicts, occ = build_units(files, caches, memo)
        st = strata(units)
        rows = instances_block(name, d, k, kind, caches, memo)
        cov_fail = [r["id"] for r in rows if not r["coverage"]["covered"]]
        ver_goal_fail = [r["id"] for r in rows if r["verified"] and r["coverage"]["goal_at_end"] is not True]
        harmful_units = sorted(({"key_id": key_id(kk), "class": r["cls"], "subtype": r["sub"], "key": key_json(kk),
                                 "text": text, "n_blocks": len(B.blocks_of(kk[0]))}
                                for kk, (r, text, _) in units.items() if r["cls"] in HARMFUL),
                               key=lambda x: x["key_id"])
        core["sets"][name] = {
            "files": {rel: caches[rel][1] for rel in files},
            "unique_keys": len(units), "conflicting_keys": len(conflicts), "conflicts": conflicts,
            "occurrence_level": occ, "strata": st, "harmful_keys": harmful_units,
            "instances": rows,
            "checks": {"plans_not_covered": cov_fail, "verified_plans_not_goal_under_F": ver_goal_fail},
            "R1": reading_r1(rows), "R2": reading_r2(name, st["total"]),
        }
        print(f"[freplay] {name}: {len(units)} keys, R2 {st['total']['rates']['support_share_R2']['p']}, "
              f"flagged {core['sets'][name]['R1']['flagged']}", flush=True)
    units, conflicts, occ = build_units(all_files, caches, memo)
    core["pooled_all5"] = {"unique_keys": len(units), "conflicting_keys": len(conflicts), "conflicts": conflicts,
                           "occurrence_level": occ, "strata": strata(units)}
    core["R2_any_set_triggered"] = any(core["sets"][n]["R2"]["wording_change_triggered"] for n, *_ in SETS)
    canon = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    core_sha = hashlib.sha256(canon.encode()).hexdigest()
    meta = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "script": "scripts/freplay.py",
            "script_sha256": sha256_file(Path(__file__)), "inputs": pins, "python": sys.version.split()[0],
            "wall_s": round(time.time() - t0, 1), "torch_loaded": "torch" in sys.modules,
            "bwmn_lm_loaded": "bwmn.lm" in sys.modules, "classify_calls": len(memo.c)}
    out = {"core": core, "core_sha256": core_sha, "meta": meta, "comparison_to_exploratory": compare(core)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FREPLAY.json").write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
                                      encoding="utf-8")
    (OUT / "FREPLAY.md").write_text(render_md(core, core_sha, meta, out["comparison_to_exploratory"]), encoding="utf-8")
    return core_sha, meta


def compare(core: dict) -> dict:
    out = {}
    for name, ex in EXPLORATORY.items():
        t = core["sets"][name]["strata"]["total"]
        reg = {"unique_keys": core["sets"][name]["unique_keys"],
               "false_legal_accepted": t["classes"]["false_legal_accepted"],
               "wrong_state": t["classes"]["wrong_state"], "false_illegal": t["classes"]["false_illegal"],
               "flag_legal_on_illegal": t["rates"]["flag_false_legal"]["k"], "illegal_keys": t["illegal"],
               "support_rejected": t["classes"]["false_legal_rejected_by_support_check"]}
        out[name] = {"exploratory": ex, "registered": reg}
    return out


# ----------------------------------------------------------------------------- report

def pct(r: dict) -> str:
    if r["p"] is None:
        return f"{r['k']}/{r['n']} (n/a)"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {100 * r['p']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]"


def render_md(core: dict, core_sha: str, meta: dict, cmp: dict) -> str:
    L = []
    w = L.append
    w("# FREPLAY: registered CPU replay of every logged F query (PREREG_FREPLAY.md, F-09)\n")
    w(f"Written {meta['written']} by `scripts/freplay.py` (sha256 `{meta['script_sha256']}`). Prereg sha256 "
      f"`{core['prereg_sha256']}` (verified at run time; every input sha256 verified against the prereg table "
      f"before reading). No model and no torch were loaded (torch in sys.modules: {meta['torch_loaded']}; "
      f"bwmn.lm: {meta['bwmn_lm_loaded']}). Wall {meta['wall_s']} s.\n")
    w(f"`core` sha256 (byte-compare target, prereg §8): `{core_sha}`.\n")
    w("Not blind: the exploratory F-09 scratch numbers existed before this replay (prereg §0); see the last table.\n")
    w("## Per set: classes of unique (state, action) keys\n")
    w("| Set | Unique keys | Legal / illegal | ok | false_illegal | false_legal_accepted | "
      "false_legal_rejected_by_support_check | wrong_state | malformed | Conflicting keys |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for name, *_ in SETS:
        S = core["sets"][name]
        t = S["strata"]["total"]
        c = t["classes"]
        w(f"| {name} | {S['unique_keys']} | {t['legal']} / {t['illegal']} | {c['ok']} | {c['false_illegal']} | "
          f"{c['false_legal_accepted']} | {c['false_legal_rejected_by_support_check']} | {c['wrong_state']} | "
          f"{c['malformed']} | {S['conflicting_keys']} |")
    P = core["pooled_all5"]
    t = P["strata"]["total"]
    c = t["classes"]
    w(f"| all5 (pooled, secondary) | {P['unique_keys']} | {t['legal']} / {t['illegal']} | {c['ok']} | "
      f"{c['false_illegal']} | {c['false_legal_accepted']} | {c['false_legal_rejected_by_support_check']} | "
      f"{c['wrong_state']} | {c['malformed']} | {P['conflicting_keys']} |\n")
    w("Subtypes (prereg §3):\n")
    w("| Set | " + " | ".join(SUBTYPES) + " |")
    w("|---|" + "---|" * len(SUBTYPES))
    for name in [n for n, *_ in SETS] + ["all5"]:
        st = (core["pooled_all5"] if name == "all5" else core["sets"][name])["strata"]["total"]["subtypes"]
        w(f"| {name} | " + " | ".join(str(st[x]) for x in SUBTYPES) + " |")
    w("")
    w("## Rates (exact Clopper-Pearson 95% CIs)\n")
    w("| Set | harmful / keys | false_illegal / legal | wrong_state / legal | false_legal_accepted / illegal | "
      "**R2** support_share / illegal | F flag says legal / illegal | F-flag accuracy (schema-valid) | "
      "delta-exact |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name in [n for n, *_ in SETS] + ["all5"]:
        t = (core["pooled_all5"] if name == "all5" else core["sets"][name])["strata"]["total"]
        r, s2 = t["rates"], t["secondary"]
        w(f"| {name} | {pct(r['harmful'])} | {pct(r['false_illegal'])} | {pct(r['wrong_state'])} | "
          f"{pct(r['false_legal_accepted'])} | {pct(r['support_share_R2'])} | {pct(r['flag_false_legal'])} | "
          f"{pct(s2['flag_accuracy'])} | {pct(s2['delta_exact'])} |")
    w("")
    w("## R1 (oracle-equivalence; prereg §6)\n")
    w("| Set | Instances | Observed success | Flagged (harmful key in forest) | Flagged solved | "
      "Exact-F success bounds [L, U] | Ceiling <= observed | Support-exposed instances | Plans not covered | "
      "Verified plans not goal under F |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for name, *_ in SETS:
        S = core["sets"][name]
        R = S["R1"]
        w(f"| {name} | {R['n_instances']} | {R['observed_success']} | {R['flagged']} "
          f"{('(' + ', '.join(R['flagged_ids']) + ')') if R['flagged_ids'] else ''} | {R['flagged_solved']} | "
          f"[{R['exactF_success_lower_L']}, {R['exactF_success_upper_U']}] | {R['ceiling_at_most_observed']} | "
          f"{R['support_exposed_instances']} | {len(S['checks']['plans_not_covered'])} | "
          f"{len(S['checks']['verified_plans_not_goal_under_F'])} |")
    w("")
    w("Reading (registered): an unflagged instance's exact-F run is identical to the logged run by construction "
      "(D-50 CRN assumption). Where every flagged instance was solved, U equals the observed success, so the "
      "exact-F ceiling is AT MOST the observed success; exact equality needs an exact-F re-run of the flagged "
      "instances only (GPU; not part of this replay). Advisory consequence for the B4 draft (D-70 answer 5): an "
      "exact-F oracle arm can show no headroom on such a set; drop it or restrict it to flagged instances.\n")
    w("## Harmful keys (all sets; unit level)\n")
    w("| Set | key_id | Class / subtype | Blocks | Action | F text |")
    w("|---|---|---|---|---|---|")
    for name, *_ in SETS:
        for h in core["sets"][name]["harmful_keys"]:
            w(f"| {name} | `{h['key_id']}` | {h['class']} / {h['subtype']} | {h['n_blocks']} | {h['key']['a']} | "
              f"`{h['text']}` |")
    w("")
    w("## R2 (support-check dependence; prereg §7)\n")
    w("| Set | Claim | R2 | > 5%: wording change |")
    w("|---|---|---|---|")
    for name, *_ in SETS:
        R = core["sets"][name]["R2"]
        w(f"| {name} | {R['claim']} | {pct(R['R2'])} | {R['wording_change_triggered']} |")
    w("")
    w(f"R2 > 5% in any set: **{core['R2_any_set_triggered']}**.\n")
    if core["R2_any_set_triggered"]:
        w("Registered consequence: each claim whose set is triggered must say the plan was verified \"by the learned "
          "F plus a domain-agnostic delta-consistency check (a predicted deletion must be a fact of the state)\"; "
          "\"the domain rules are known only to the learned F\" is withdrawn from every document that states or "
          "quotes it (the pinned blocks_planner.py docstring is not edited).\n")
    w("## Strata by block count (unique keys)\n")
    for name, *_ in SETS:
        w(f"### {name}\n")
        w("| Blocks | Keys | Legal / illegal | harmful | false_illegal | R2 support_share / illegal | "
          "F flag says legal / illegal |")
        w("|---|---|---|---|---|---|---|")
        for n, cell in core["sets"][name]["strata"]["by_blocks"].items():
            r = cell["rates"]
            w(f"| {n} | {cell['keys']} | {cell['legal']} / {cell['illegal']} | {r['harmful']['k']} | "
              f"{cell['classes']['false_illegal']} | {pct(r['support_share_R2'])} | {pct(r['flag_false_legal'])} |")
        w("")
        w("| Action | Keys | Legal / illegal | harmful | R2 support_share / illegal | F flag says legal / illegal |")
        w("|---|---|---|---|---|---|")
        for a, cell in core["sets"][name]["strata"]["by_action"].items():
            r = cell["rates"]
            w(f"| {a} | {cell['keys']} | {cell['legal']} / {cell['illegal']} | {r['harmful']['k']} | "
              f"{pct(r['support_share_R2'])} | {pct(r['flag_false_legal'])} |")
        w("")
    w("Set x block count x action type strata with all rates and CIs: `FREPLAY.json` "
      "core.sets.<set>.strata.by_blocks_action.\n")
    w("## Input files\n")
    w("| File | Lines | Torn lines | Keys | Duplicate lines (same / different text) | sha256 |")
    w("|---|---|---|---|---|---|")
    for name, *_ in SETS:
        for rel, f in core["sets"][name]["files"].items():
            w(f"| `{rel}` | {f['lines']} | {f['torn_lines']} | {f['keys']} | {f['dup_lines_same_text']} / "
              f"{f['dup_lines_diff_text']} | `{meta['inputs']['fcache'][rel]['sha256']}` |")
    w("")
    for rel, sha in meta["inputs"]["outcomes"].items():
        w(f"- `{rel}` `{sha}`")
    for rel, v in meta["inputs"]["dirs"].items():
        w(f"- `{rel}/*.pddl` {v['files']} files, digest `{v['digest']}`")
    w(f"- `bwmn/blocks.py` `{meta['inputs']['blocks_sha256']}`; `bwmn/blocks_planner.py` (reference, not imported) "
      f"`{meta['inputs']['planner_sha256_reference']}`")
    w("- Per-shard plans.jsonl shas: `FREPLAY.json` meta.inputs.shard_plans.\n")
    w("## Registered vs exploratory (F-09 scratch; prereg §0)\n")
    w("| Set | Quantity | Exploratory | Registered |")
    w("|---|---|---|---|")
    for name, v in cmp.items():
        for q, x in v["exploratory"].items():
            reg = v["registered"].get(q)
            if q == "flag_legal_share_approx":
                reg = round(v["registered"]["flag_legal_on_illegal"] / max(1, v["registered"]["illegal_keys"]), 4)
            w(f"| {name} | {q} | {x} | {reg} |")
    w("")
    w("Definitions differ slightly from the scratch scripts (prereg §0 a-d); differences are reported, not "
      "reconciled. Independent second implementation from the prereg text alone (D-61/D-71 practice): not yet done; "
      "its target is the `core` sha256 above.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------- selftest (synthetic data only)

def selftest():
    s = B.towers_to_state([["a", "b"], ["c"]])  # a on table, b on a, c on table; hand empty
    T = lambda add, dele, legal: json.dumps({"add": add, "del": dele, "legal": legal})
    cases = [
        (("unstack", "b", "a"), T(["clear a", "holding b"], ["clear b", "handempty", "on b a"], True), "ok", "ok_legal_exact"),
        (("unstack", "b", "a"), T(["clear a"], ["clear b", "handempty", "on b a"], True), "wrong_state", "wrong_state"),
        (("unstack", "b", "a"), T([], [], False), "false_illegal", "fi_flag"),
        (("unstack", "b", "a"), T(["holding b"], ["on b c"], True), "false_illegal", "fi_support"),
        (("unstack", "b", "a"), "not json", "false_illegal", "fi_malformed"),
        (("pick-up", "a"), T(["holding a"], ["clear a", "handempty", "ontable a"], True),
         "false_legal_rejected_by_support_check", "false_legal_rejected_by_support_check"),  # a is not clear
        (("pick-up", "a"), T(["holding a"], ["handempty", "ontable a"], True), "false_legal_accepted", "false_legal_accepted"),
        (("pick-up", "a"), T([], [], False), "ok", "ok_illegal_flag"),
        (("pick-up", "a"), T([], ["clear a"], False), "ok", "ok_illegal_flag_unsupported"),
        (("put-down", "a"), T(["clear a"], ["holding a"], True), "false_legal_rejected_by_support_check",
         "false_legal_rejected_by_support_check"),
        (("put-down", "a"), '{"add": [], "del": [], "legal": "no"}', "malformed", "malformed"),
        (("put-down", "a"), '{"add": [], "del": [], "legal": false, "x": 1}', "malformed", "malformed"),
    ]
    bad = []
    for a, text, cls, sub in cases:
        r = classify(s, a, text)
        if (r["cls"], r["sub"]) != (cls, sub):
            bad.append((a, text, r["cls"], r["sub"], cls, sub))
    # forest (prereg §5) on a synthetic cache: s -unstack b a-> s1 (ok); s pick-up a (support-rejected);
    # s1 put-down b (legal, F says illegal: harmful); s2 = unreachable state's key must not enter the forest
    s1 = B.step(s, ("unstack", "b", "a"))[0]
    s2 = B.towers_to_state([["a"], ["b"], ["c"]])
    cache = {(s, ("unstack", "b", "a")): cases[0][1], (s, ("pick-up", "a")): cases[5][1],
             (s1, ("put-down", "b")): T([], [], False), (s2, ("pick-up", "a")): T([], [], False)}
    by_state = {}
    for (st, a) in cache:
        by_state.setdefault(st, []).append(a)
    m = Memo()
    states, keys = forest(s, {k: sorted(v) for k, v in by_state.items()}, cache, m)
    got = sorted((k[1], r["cls"]) for k, r in keys)
    want = sorted([(("unstack", "b", "a"), "ok"), (("pick-up", "a"), "false_legal_rejected_by_support_check"),
                   (("put-down", "b"), "false_illegal")])
    if got != want or states != {s, s1}:
        bad.append(("forest", got, len(states)))
    # conflict rule (prereg §2): same key, two files, texts ok vs false_illegal -> unit false_illegal; 2 occurrences
    k0 = (s, ("unstack", "b", "a"))
    fake = {"f0": ({k0: cases[0][1]}, {}), "f1": ({k0: T([], [], False)}, {})}
    units, conflicts, occ = build_units(["f0", "f1"], fake, Memo())
    if units[k0][0]["cls"] != "false_illegal" or len(conflicts) != 1 or occ["occurrences"] != 2:
        bad.append(("conflict rule", units[k0][0]["cls"], len(conflicts), occ))
    # Clopper-Pearson against values recorded in DECISIONS (P-08 73/110 0.567-0.751; P-14 88/110 0.713-0.870) and
    # textbook cases
    refs = [((73, 110), (0.567, 0.751)), ((88, 110), (0.713, 0.870)), ((0, 10), (0.0, 0.308)),
            ((10, 10), (0.692, 1.0)), ((5, 110), (0.015, 0.103))]
    for (k, n), (lo, hi) in refs:
        ci = clopper_pearson(k, n)
        if abs(ci[0] - lo) > 0.0006 or abs(ci[1] - hi) > 0.0006:
            bad.append(("cp", k, n, ci, lo, hi))
    big = clopper_pearson(798, 4066)  # large-n convergence
    if not (0.18 < big[0] < 0.196 < big[1] < 0.21):
        bad.append(("cp big", big))
    print("selftest", "FAIL" if bad else "ok", bad, "cp(798,4066) =", big)
    return 1 if bad else 0


def log_run(cmd: str, status: str):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "RUN_LOG.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} | {cmd} | script sha256 {sha256_file(Path(__file__))} | "
                 f"{status}\n")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        rc = selftest()
        log_run("selftest", f"rc {rc} (synthetic data only; no run data read)")
        return rc
    if cmd == "run":
        log_run("run", "started")
        try:
            core_sha, meta = run()
        except BaseException as e:  # noqa: BLE001 - every failed attempt is logged
            log_run("run", f"FAILED {type(e).__name__}: {str(e)[:300]}")
            raise
        log_run("run", f"ok core_sha256 {core_sha} wall {meta['wall_s']} s")
        print("[freplay] core sha256", core_sha)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
