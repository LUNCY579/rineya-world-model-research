"""Status vocabulary pools for RCL (SPEC_RCL.md section 5).

Real-word pools are frozen literals, disjoint across splits. Nonce words come from a
seeded syllable grammar and are bucketed by sha256 into splits. PREREG-3's exposed
vocabulary and the dataset factory's exposed nonce words are excluded everywhere.
"""
import hashlib
import random

ANCHOR_APPLY = "posted"   # RCWM ledger rule word
ANCHOR_NOOP = "void"      # T2's trained no-op word

# Exposed elsewhere; never used in any RCX pool (train or test).
EXCLUDED = frozenset({
    "pending", "draft", "cancelled", "reversed", "held", "approved", "active", "ok",
    "dax", "wug", "Void", "VOID", "voided", "unposted", "not_posted", "not posted",
    "xelth", "quom",
})

REAL_TRAIN = (
    "cleared", "settled", "booked", "queued", "flagged", "rejected", "final", "open",
    "closed", "locked", "waived", "refunded", "disputed", "archived", "verified",
    "expired", "staged", "signed", "filed", "tracked", "frozen", "paid", "billed", "logged",
)
REAL_DEV = (
    "parked", "routed", "tagged", "minted", "boxed", "netted", "shelved", "scored",
    "ranked", "hedged", "wired", "fenced",
)
REAL_SEALED = (
    "lapsed", "pruned", "masked", "graded", "dusted", "hashed", "linked", "rolled",
    "stamped", "tallied", "keyed", "pinned",
)

# Diagnostic near-miss probes of the trained no-op word (literal semantics: not "void").
NEARMISS_VOID = ("Void", "VOID", "voided", "void_")
# Posted-like strings that are NOT "posted" (literal semantics: no-op under S+{posted}).
POSTEDLIKE = ("nonposted", "posted_not", "reposted", "postedx", "un-posted")

_ONSETS = "bdfgklmnprstvz"
_VOWELS = "aeiou"
_CODAS = "lnrs"
NONCE_SEED = 20261201


def edit_distance(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _syllable(rng):
    s = rng.choice(_ONSETS) + rng.choice(_VOWELS)
    if rng.random() < 0.35:
        s += rng.choice(_CODAS)
    return s


def split_of_nonce(word):
    """sha256 bucket: 0-5 train, 6-7 dev, 8-9 sealed."""
    b = int(hashlib.sha256(word.encode()).hexdigest(), 16) % 10
    return "train" if b <= 5 else ("dev" if b <= 7 else "sealed")


def _build_nonce(n_target=900, seed=NONCE_SEED):
    rng = random.Random(seed)
    real = set(REAL_TRAIN) | set(REAL_DEV) | set(REAL_SEALED) | {ANCHOR_APPLY, ANCHOR_NOOP}
    real |= set(POSTEDLIKE) | {w.lower() for w in EXCLUDED}
    words, seen = [], set()
    tries = 0
    while len(words) < n_target and tries < 200000:
        tries += 1
        w = "".join(_syllable(rng) for _ in range(rng.choice((2, 2, 3))))
        if w in seen or not 4 <= len(w) <= 8:
            continue
        if any(edit_distance(w, r) <= 1 for r in real):
            continue
        if "post" in w or "void" in w:
            continue
        seen.add(w)
        words.append(w)
    pools = {"train": [], "dev": [], "sealed": []}
    for w in words:
        pools[split_of_nonce(w)].append(w)
    # Remove cross-split near-duplicates (edit distance <= 1) from the test pools.
    tr = pools["train"]
    for sp in ("dev", "sealed"):
        pools[sp] = [w for w in pools[sp] if all(edit_distance(w, t) > 1 for t in tr)]
    dv = pools["dev"]
    pools["sealed"] = [w for w in pools["sealed"] if all(edit_distance(w, d) > 1 for d in dv)]
    return {k: tuple(v) for k, v in pools.items()}


NONCE = _build_nonce()


def pools(split):
    """Return (real_words, nonce_words) for a split in {'train','dev','sealed'}."""
    real = {"train": REAL_TRAIN, "dev": REAL_DEV, "sealed": REAL_SEALED}[split]
    return real, NONCE[split]


def audit():
    """Assert the frozen disjointness invariants. Returns a summary dict."""
    real = {"train": set(REAL_TRAIN), "dev": set(REAL_DEV), "sealed": set(REAL_SEALED)}
    allw = {}
    for sp in ("train", "dev", "sealed"):
        allw[sp] = real[sp] | set(NONCE[sp])
    for a, b in (("train", "dev"), ("train", "sealed"), ("dev", "sealed")):
        assert not (allw[a] & allw[b]), (a, b, allw[a] & allw[b])
    for sp, ws in allw.items():
        assert not (ws & EXCLUDED), (sp, ws & EXCLUDED)
        assert ANCHOR_APPLY not in ws and ANCHOR_NOOP not in ws
        for w in ws:
            assert " " not in w and w, w
    for w in NONCE["dev"] + NONCE["sealed"]:
        assert all(edit_distance(w, t) > 1 for t in NONCE["train"]), w
    # Cross-split near-duplicates (edit distance <= 1) are forbidden; within a split
    # nonce near-neighbours are allowed. Anchors must be far from every pool word.
    for a_sp, b_sp in (("train", "dev"), ("train", "sealed"), ("dev", "sealed")):
        for a in allw[a_sp]:
            for b in allw[b_sp]:
                assert edit_distance(a, b) > 1, ("cross-split near-duplicate", a, b)
    for anchor in (ANCHOR_APPLY, ANCHOR_NOOP):
        for sp in allw:
            for w in allw[sp]:
                assert edit_distance(anchor, w) > 1, ("near anchor", anchor, w)
    for w in REAL_TRAIN + REAL_DEV + REAL_SEALED:
        for v in REAL_TRAIN + REAL_DEV + REAL_SEALED:
            assert w == v or edit_distance(w, v) > 1, ("real near-duplicate", w, v)
    return {sp: {"real": len(real[sp]), "nonce": len(NONCE[sp])} for sp in allw}
