"""Scoring, shortcut floors and aggregation for RCL batteries (SPEC_RCL.md 8-9)."""
from collections import Counter, defaultdict

from . import world as W

PRIMARY_STRATA = ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike",
                  "V1_pos", "V2_neg", "V3_inverted", "V5_posted_override")
FLOORS = ("COPY", "ALWAYS", "TRIGGER", "CONTAINS_POSTED", "STATUS_LOOKUP", "RULE_WORD_MATCH",
          "MAJORITY", "ORACLE")


def classify(rec, pred_after):
    """Return dict(exact, intent, decision_ok, err) for one prediction (None = malformed)."""
    gold = rec["gold"]
    before = rec["state"]
    if pred_after is None:
        return {"exact": 0, "intent": "format", "decision_ok": 0, "err": "format"}
    exact = int(pred_after == gold["after"] and list(pred_after) == list(gold["after"]))
    intent = "noop" if pred_after == before else "apply"
    dec_ok = int(intent == gold["decision"])
    if exact:
        err = None
    elif gold["decision"] == "noop" and intent == "apply":
        err = "applied_when_noop"
    elif gold["decision"] == "apply" and intent == "noop":
        err = "false_copy"
    else:
        err = "arith"
    return {"exact": exact, "intent": intent, "decision_ok": dec_ok, "err": err}


class FloorContext:
    """TRAIN-derived statistics for the data-dependent floors."""

    def __init__(self, train_sets):
        """train_sets: dict regime -> list of train records."""
        self.noop_words = set()
        self.lookup = {}
        self.majority = {}
        for regime, recs in train_sets.items():
            by = defaultdict(Counter)
            tot = Counter()
            for r in recs:
                d = r["gold"]["decision"]
                by[r["event"]["status"]][d] += 1
                tot[d] += 1
                if regime == "FX" and d == "noop":
                    self.noop_words.add(r["event"]["status"])
            self.lookup[regime] = {s: c.most_common(1)[0][0] for s, c in by.items()}
            self.majority[regime] = tot.most_common(1)[0][0]


def floor_after(name, rec, ctx, regime="FX"):
    st, ev = rec["state"], rec["event"]
    order = rec["state_order"]
    applied = W.applied(st, ev)
    noop = dict(st)

    def pick(apply):
        a = applied if apply else noop
        return {k: a[k] for k in order}

    if name == "COPY":
        return pick(False)
    if name == "ALWAYS":
        return pick(True)
    if name == "TRIGGER":
        return pick(ev["status"] not in ctx.noop_words)
    if name == "CONTAINS_POSTED":
        return pick("posted" in ev["status"])
    if name == "STATUS_LOOKUP":
        d = ctx.lookup[regime].get(ev["status"], ctx.majority[regime])
        return pick(d == "apply")
    if name == "RULE_WORD_MATCH":
        rule_line = rec["input"].split("\n", 1)[0]
        return pick(ev["status"] in rule_line)
    if name == "MAJORITY":
        return pick(ctx.majority[regime] == "apply")
    if name == "ORACLE":
        ast, _tid, state, event = W.parse_input(rec["input"])
        ok, after = W.transition(ast, state, event)
        return {k: after[k] for k in order}
    raise ValueError(name)


def aggregate(recs, results):
    """results: list of classify() dicts aligned with recs. Returns nested summary:
    by (group, surface): parent-macro exact, decision, parent-all-exact, error counts, n_parents."""
    cells = defaultdict(lambda: defaultdict(list))
    errs = defaultdict(Counter)
    for r, res in zip(recs, results):
        key = (r["group"], r["surface"])
        cells[key][r["parent_id"]].append(res)
        if res["err"]:
            errs[key][res["err"]] += 1
    out = {}
    for key, parents in cells.items():
        pm_exact = [sum(x["exact"] for x in v) / len(v) for v in parents.values()]
        pm_dec = [sum(x["decision_ok"] for x in v) / len(v) for v in parents.values()]
        all_ex = [int(all(x["exact"] for x in v)) for v in parents.values()]
        out[f"{key[0]}|{key[1]}"] = {
            "exact": sum(pm_exact) / len(pm_exact), "decision": sum(pm_dec) / len(pm_dec),
            "parent_all_exact": sum(all_ex) / len(all_ex), "n_parents": len(parents),
            "errors": dict(errs[key]),
        }
    return out


def parent_scores(recs, results, groups, surface, field="exact"):
    """dict parent_id -> list of per-variant scores, for the given groups and surface."""
    d = defaultdict(list)
    for r, res in zip(recs, results):
        if r["group"] in groups and r["surface"] == surface:
            d[r["parent_id"]].append(res[field])
    return dict(d)


def composite(summary, surface, strata=PRIMARY_STRATA, field="exact"):
    vals = [summary[f"{s}|{surface}"][field] for s in strata if f"{s}|{surface}" in summary]
    return sum(vals) / len(vals) if vals else None


def verdict_rule_trigger(summary, surface="in", bar=0.80, low=0.30):
    """PREREG-3-style per-arm verdict on fixed-rule strata, computed on probe (non-posted) variants
    via decision accuracy is not separable here; we use the parent-level exact of F strata and the
    error direction. Returns 'RULE', 'TRIGGER' or 'MIXED' (descriptive only)."""
    unseen = [summary.get(f"{s}|{surface}") for s in ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")]
    void = summary.get(f"F1_void|{surface}")
    if not all(unseen) or not void:
        return None
    if all(u["exact"] >= bar for u in unseen) and void["exact"] >= bar:
        return "RULE"
    if all(u["exact"] <= 0.5 + low / 2 for u in unseen) and void["exact"] >= bar:
        return "TRIGGER"
    return "MIXED"
