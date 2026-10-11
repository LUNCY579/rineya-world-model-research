"""Audit of rule wordings for RCX-E01R (CX D-33): vocabulary collisions and near-duplicates.

A wording pool used for a held-out test must not reuse a frame of any wording that is or was visible to a model or to an
analysis: the training pools (t*), the spent held-out pool (h*), and the composition frames and clauses. Within a test
pool, near-duplicates are collapsed so that a DEV/SEALED split cannot place two versions of one frame on both sides.
"""
import re

from . import templates as T
from . import vocab as V

SLOT = "{L}"


def words(text):
    """Lowercase word tokens with the slot removed."""
    return re.findall(r"[a-z]+", text.replace(SLOT, " ").lower())


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def jaccard(a, b):
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / max(1, len(sa | sb))


def shared_ngram(a, b, n):
    return bool(ngrams(words(a), n) & ngrams(words(b), n))


def near_dup(a, b, jac=0.5, n=5):
    """True if two wordings share at least a fraction `jac` of their word types or any `n`-word sequence."""
    return jaccard(a, b) >= jac or shared_ngram(a, b, n)


def reference_wordings():
    """Every RCX wording that a model or an analysis has seen: train and spent held-out templates, frames, clauses."""
    refs = {}
    for name, pool in (("SPOS", T.SPOS), ("SNEG", T.SNEG), ("AND", T.AND_FRAMES), ("OR", T.OR_FRAMES),
                       ("CLAUSE", T.CLAUSE)):
        for tid, text in pool.items():
            refs[f"{name}:{tid}"] = text
    return refs


def status_words():
    """Every status-like word RCX or its peers use (all splits, nonce pools, anchors, probes, excluded words)."""
    out = set()
    for split in ("train", "dev", "sealed"):
        real, nonce = V.pools(split)
        out |= {w.lower() for w in real} | {w.lower() for w in nonce}
    out |= {w.lower() for w in V.EXCLUDED} | {V.ANCHOR_APPLY, V.ANCHOR_NOOP}
    out |= {w.lower() for w in V.NEARMISS_VOID} | {w.lower() for w in V.POSTEDLIKE}
    return {w for w in out if re.fullmatch(r"[a-z]+", w)}


def vocab_hits(text, vocab=None):
    vocab = status_words() if vocab is None else vocab
    return sorted(set(words(text)) & vocab)


def real_status_words():
    """The real-word statuses, anchors, probes and excluded words (no nonce words): the words whose inflections a rule
    wording must not contain either (e.g. 'posting' for the anchor 'posted', 'rejection' for 'rejected')."""
    out = set()
    for split in ("train", "dev", "sealed"):
        out |= {w.lower() for w in V.pools(split)[0]}
    out |= {w.lower() for w in V.EXCLUDED} | {V.ANCHOR_APPLY, V.ANCHOR_NOOP}
    out |= {w.lower() for w in V.NEARMISS_VOID} | {w.lower() for w in V.POSTEDLIKE}
    return {w for w in out if re.fullmatch(r"[a-z]+", w) and len(w) >= 4}


def stem_hits(text, min_prefix=4, vocab=None):
    """Words of `text` sharing a prefix of at least `min_prefix` letters with a real status-like word (CX D-34)."""
    vocab = real_status_words() if vocab is None else vocab
    hits = set()
    for w in set(words(text)):
        for s in vocab:
            k = 0
            while k < min(len(w), len(s)) and w[k] == s[k]:
                k += 1
            if k >= min_prefix:
                hits.add(f"{w}~{s}")
    return sorted(hits)


def ref_hits(text, refs=None, jac=0.5, n=5):
    refs = reference_wordings() if refs is None else refs
    return sorted(k for k, r in refs.items() if near_dup(text, r, jac, n))


def clusters(items, jac=0.6, n=6):
    """Union-find clusters of near-duplicate wordings among `items` (list of (id, text)); returns lists of ids, each
    sorted, clusters ordered by their smallest id."""
    parent = {i: i for i, _ in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if near_dup(items[a][1], items[b][1], jac, n):
                ra, rb = find(items[a][0]), find(items[b][0])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i, _ in items:
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
