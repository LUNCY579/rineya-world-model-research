"""T2b-ledger dev gate: scorer self-test, the dev analysis and the PRE-STATED gate rule (CPU, numpy only, never torch).

  python -B scripts/t2bl_gate.py --selftest [--record]     # scorer self-test on the dev bank (golds, floors, negatives,
                                                           # parser identity) -> docs/t2b/ledger/SCORER_SELFTEST.json
  python -B scripts/t2bl_gate.py --dryrun [--record]       # every gate branch on synthetic outcomes through the real
                                                           # scorer, metrics, decide() and wording() -> GATE_DRYRUN.json
  python -B scripts/t2bl_gate.py --write-rule-md           # docs/t2b/ledger/GATE_RULE.md from RULE_TEXT (refused once
                                                           # this revision's run directory has an output)
  python -B scripts/t2bl_gate.py --freeze-ready            # docs/t2b/ledger/GATE_FREEZE_READY.json: what the freeze
                                                           # WOULD pin now (every input's sha256, PREREG-5's rules and
                                                           # code digests, the draft), the preconditions and the exact
                                                           # freeze command. Not a freeze: the runner ignores it.
  python -B scripts/t2bl_gate.py --freeze-rule --approval "<who, when, DECISIONS entry>"
                                                           # GATE_RULE.json: sha256 of every frozen input (create-only;
                                                           # refused if this revision's run directory has an output).
                                                           # NOT run by the kit build: freezing is the operator's step.
  python -B scripts/t2bl_gate.py [--run runs/t2bl/dev_rev3] # the registered analysis (once; create-only): GATE.md, then
                                                           # GATE.json (the completion marker); a stopped run gets a
                                                           # NO VERDICT record

Before any output is read: the frozen rule record verifies (registered mode) and the scorer self-test passes.
Kit revision 3 (R29): the rule text and every threshold are unchanged (byte-identical to the revision-2 rule frozen in
D-48); under its revision path only the training data changed and the dev bank was redrawn under a new seed
(t2bl_common.SEED_DEV) into a new run directory (t2bl_common.DEV_RUN). Revision 2's dev bank and adapters
(runs/t2bl/dev) are spent; this module never reads their outputs.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("t2bl_gate.py must run without torch")
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bl_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

RULE_VERSION = 2
SEEDS = (0, 1, 2)
BAR_HS = 0.80
KIND_FLOOR = 0.60
NI_MARGIN = 0.05
TWIN_BAR = 0.80
COPY_CAP = 0.10
ALIGN_MARGIN = 0.10
TRIGGER_TRAINED_MIN = 0.80
TRIGGER_HELDOUT_MAX = 0.30
RULE_TEXT = (
    "T2b-ledger dev gate, revision 2 (pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists). Inputs: "
    "the dev bank data/t2bl_dev.jsonl (seed 20261162; 100 bases; 19 strata; the DEV half of every split held-out "
    "partition: held-out words, names, key orders and paraphrase 4) generated in one run of scripts/t2bl_run.py (tag dev) "
    "with sides OFF (Titan-Sparse, adapter disabled), T2s0 (the PREREG-2 T2 seed-0 adapter) and ON (each T2b-ledger seed "
    "s = 0, 1, 2, unmerged), scored by p3_common.score_onestep (exact balances, any key order). For each seed s: G1 "
    "(held-out words on the combined held-out surface): the ON accuracy on HS_words (100 rows: a held-out non-posted "
    "status word in the combined held-out surface: rule paraphrase 4, held-out account names, every object in a held-out "
    "key order; gold = balances unchanged) is >= 0.80; G1b (held-out words on the canonical surface): the ON accuracy on "
    "H_words (100 rows) is >= 0.80; G1c (every word kind): on HS_words and on H_words, the ON accuracy of each of the "
    "five word kinds (not-applied, neutral, nonce, void variants, posted look-alikes; 20 rows each) is >= 0.60; G2 "
    "(in-distribution non-inferiority to T2 s0): the ON accuracy on ID (ID_posted and ID_void, 200 canonical-surface "
    "in-range rows) minus T2s0's accuracy on the same rows is >= -0.05 (point estimate; the paired base-bootstrap 95% "
    "lower bound is reported, not gated); G3 (the posted twins on the held-out surface): on HS_posted (the posted twins "
    "of HS_words' bases in the same surface) the ON accuracy is >= 0.80 and the share of ON answers equal to the input "
    "balances is <= 0.10; G4 (alignment with the seed's own canonical accuracy, as in PREREG-5's non-inferiority): the "
    "ON accuracies on HS_words and on H_words are each at least the seed's ON accuracy on ID_void minus 0.10. PROCEED to "
    "the sealed confirmation (PREREG-5) iff G1, G1b, G1c, G2, G3 and G4 hold in all three seeds, the run is complete "
    "(DONE.json, every stage recorded and unchanged) and the A0 anchor passed in every turn. Otherwise STOP, worded as: "
    "TRIGGER WIDENED iff in all three seeds the ON accuracy on S_words (training non-posted words, canonical surface) is "
    ">= 0.80, the ON accuracy on H_words (held-out words, canonical surface) is <= 0.30, and more than half of the "
    "H_words errors apply the event as if posted; PARTIAL otherwise, with the per-stratum, per-variant, per-seed table "
    "and the named patterns (heldout_below_bar, word_kind_below_floor, in_distribution_cost, heldout_posted_twins, "
    "not_aligned_with_own_canonical, trained_words_below_bar, seed_disagreement). NO VERDICT (instrument) iff the run is "
    "incomplete or stopped (STOPPED.json) or an anchor failed; the gate analysis then writes a NO VERDICT record. "
    "Revision path after a STOP (pre-stated): no confirmation is drafted from this kit; a revised kit may change only "
    "the training data (composition, vocabulary sizes, magnitude bands, paraphrases), must draw a fresh dev bank under a "
    "new seed (this dev bank and every adapter trained on the old data are then recorded as spent), and keeps these "
    "thresholds. T2b-ledger seed 0 is the registered artifact for PREREG-5's planning and retention parts; seeds 1 and 2 "
    "replicate; no seed is ever chosen by its outcome.")
PATTERNS = ("heldout_below_bar", "word_kind_below_floor", "in_distribution_cost", "heldout_posted_twins",
            "not_aligned_with_own_canonical", "trained_words_below_bar", "seed_disagreement")
CLAUSES = ("G1", "G1b", "G1c", "G2", "G3", "G4")
VARIANT_KEYS = {"S_words": "word_kind", "H_words": "word_kind", "HS_words": "word_kind", "S_surface_words": "word_kind",
                "H_names": "names_kind", "HS_posted": "names_kind", "H_order": "event_order", "H_count": "status_kind",
                "H_para": "status_kind", "P3_words": "word", "S_surface_posted": "names_kind"}
ANALYSIS_CODE = ("scripts/t2bl_gate.py", "scripts/t2bl_common.py", "scripts/p3_common.py", "rcwmt/env.py")
REV2_RULE_RECORD = C.DOCS / "superseded_rev2" / "GATE_RULE.json"     # the revision-2 freeze (D-48), kept for the record


def revision_note() -> str:
    """The inputs of this kit revision under the rule's pre-stated revision path (rendered into GATE_RULE.md and every
    GATE.md; the rule text itself is unchanged)."""
    return (f"Kit revision {C.KIT_REVISION} (2026-10-12; DECISIONS R29; the rule's pre-stated revision path). The rule text "
            "and every threshold are byte-identical to the revision-2 rule frozen on 2026-10-10 (D-48; "
            "docs/t2b/ledger/superseded_rev2/GATE_RULE.json). Only the training data changed (data/t2bl_train.jsonl, seed "
            f"{C.SEED_TRAIN}: both top-level key orders trained; half of the diversified slots on the stacked training "
            "surface), and the dev bank was drawn FRESH under a new seed by the same builder and strata "
            f"(data/t2bl_dev.jsonl, seed {C.SEED_DEV}; the DEV half of every held-out partition, unchanged). It replaces the "
            f"bank of seed {C.SPENT_KIT_SEEDS[1]} that the rule text names, which is spent together with every adapter "
            f"trained on the revision-2 data ({', '.join(C.SPENT_RUNS)}; seeds {C.SPENT_KIT_SEEDS[0]} and "
            f"{C.SPENT_KIT_SEEDS[1]}). The dev run is tag dev in {C.DEV_RUN_REL}. Since kit revision 3 the top-level "
            "{before, event} object is trained in both orders, so 'every object in a held-out key order' in the HS_words "
            "definition holds for the event and before objects, and the reversed top order of HS_words, HS_posted and "
            "H_order is a trained level there: a deliberate consequence of the revision path, not a new held-out factor.")


class GateError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise GateError(msg)


# ============================================================================================== scorer self-test
def renderings(gold: dict, row: dict) -> list:
    """Correct answers in several surface forms: compact sorted, the row's own key order, spaced, trailing period,
    pretty, {"before","after"}-wrapped, reversed key order."""
    keys = list(gold)
    rev = OrderedDict((k, gold[k]) for k in reversed(sorted(keys)))
    own = OrderedDict((k, gold[k]) for k in (row.get("surface") or {}).get("before", sorted(keys)))
    return [env.dumps(gold), json.dumps(own, separators=(",", ":")), json.dumps(gold), env.dumps(gold) + ".",
            json.dumps(gold, indent=2), json.dumps({"before": row["before"], "after": gold}),
            json.dumps(rev, separators=(",", ":"))]


def negatives(row: dict) -> list:
    g = json.loads(row["gold"])
    k0 = sorted(g)[0]
    out = [({**g, k0: g[k0] + 1}, "value +1"), ({**g, k0: g[k0] - 1}, "value -1"),
           ({k: v for k, v in g.items() if k != k0}, "missing key"), ({**g, "ZZ": 0}, "extra key"),
           ({**g, k0: str(g[k0])}, "string value"), ({**g, k0: True}, "bool value")]
    texts = [(json.dumps(x, separators=(",", ":")), why) for x, why in out]
    texts += [("", "empty"), ("The balances are unchanged.", "prose")]
    if row["gold_rule"] == "noop":
        texts.append((row["posted_gold"], "applied on a no-op row"))
    else:
        texts.append((env.dumps(row["before"]), "copied on an apply row"))
    return texts


def selftest(rows: list) -> dict:
    n_gold = n_gold_ok = n_neg = n_neg_wrong = n_cls_ok = 0
    copy_ok = True
    parser_same = parser_n = 0
    fails = []
    for r in rows:
        g = json.loads(r["gold"])
        for t in renderings(g, r):
            n_gold += 1
            sc = C.score(r, t)
            n_gold_ok += sc["correct"]
            if not sc["correct"] and len(fails) < 5:
                fails.append({"id": r["id"], "text": t[:120], "why": "gold rendering scored wrong"})
        sc = C.score(r, env.dumps(r["before"]))
        if sc["correct"] != (r["gold_rule"] == "noop"):
            copy_ok = False
        for t, why in negatives(r):
            n_neg += 1
            sc = C.score(r, t)
            n_neg_wrong += not sc["correct"]
            want = {"applied on a no-op row": "applied_as_posted", "copied on an apply row": "copied",
                    "empty": "unparsed", "prose": "unparsed"}.get(why)
            n_cls_ok += want is None or sc["cls"] == want
            if sc["correct"] and len(fails) < 10:
                fails.append({"id": r["id"], "text": t[:120], "why": f"negative ({why}) scored correct"})
        if set(r["before"]) == set(env.ACCOUNTS):
            for t in renderings(g, r) + [x for x, _ in negatives(r)]:
                parser_n += 1
                parser_same += C.P3.parse_general(t, list(env.ACCOUNTS)) == env.parse_state(C.LEDGER, t, lenient=True)
    ok = n_gold_ok == n_gold and n_neg_wrong == n_neg and n_cls_ok == n_neg and copy_ok and parser_same == parser_n
    return {"ok": ok, "rows": len(rows), "gold_renderings": n_gold, "gold_correct": n_gold_ok, "negatives": n_neg,
            "negatives_wrong": n_neg_wrong, "error_classes_right": n_cls_ok, "copy_scores_copy_floor": copy_ok,
            "parse_general_equals_env_parse_state_on_A_D": f"{parser_same}/{parser_n}", "failures": fails,
            "bank_sha256": C.sha256(C.DEV) if C.DEV.exists() else None}


# ============================================================================================== metrics
def score_side(rows: list, texts: list) -> list:
    return [C.score(r, t) for r, t in zip(rows, texts)]


def copied(r: dict, text) -> bool:
    st = None if text is None else C.parse_ledger(text, list(r["before"]))
    return st is not None and st == r["before"]


def stratum_metrics(rows: list, sc: list, texts: list) -> dict:
    out = {}
    for s in C.DEV_STRATA:
        ix = [i for i, r in enumerate(rows) if r["stratum"] == s]
        if not ix:
            continue
        k = sum(sc[i]["correct"] for i in ix)
        lo, hi = C.wilson(k, len(ix))
        d = {"n": len(ix), "correct": k, "acc": k / len(ix), "wilson": [lo, hi],
             "classes": dict(Counter(sc[i]["cls"] for i in ix)),
             "copied_rate": sum(copied(rows[i], texts[i]) for i in ix) / len(ix)}
        vk = VARIANT_KEYS.get(s)
        if vk:
            var: dict = {}
            for i in ix:
                v = str(rows[i]["meta"].get(vk))
                var.setdefault(v, [0, 0])
                var[v][0] += 1
                var[v][1] += sc[i]["correct"]
            d["variants"] = {v: {"n": n, "acc": c / n} for v, (n, c) in sorted(var.items())}
        out[s] = d
    return out


def id_diff(rows: list, sc_on: list, sc_ref: list) -> dict:
    ix = [i for i, r in enumerate(rows) if r["stratum"] in C.ID_STRATA]
    a = sum(sc_on[i]["correct"] for i in ix) / len(ix)
    b = sum(sc_ref[i]["correct"] for i in ix) / len(ix)
    by_base: dict = {}
    for i in ix:
        d = by_base.setdefault(rows[i]["base"], [0, 0])
        d[0] += int(sc_on[i]["correct"]) - int(sc_ref[i]["correct"])
        d[1] += 1
    bl = C.boot_lower([tuple(v) for _, v in sorted(by_base.items())])
    return {"n": len(ix), "acc_on": a, "acc_t2s0": b, "diff": a - b, "boot_lower_95_one_sided": bl["lo"]}


def metrics(rows: list, texts: dict) -> dict:
    """texts: {'off': [...], 't2s0': [...], 's0': [...], 's1': [...], 's2': [...]} in row order."""
    sc = {k: score_side(rows, v) for k, v in texts.items()}
    res = {"strata": {k: stratum_metrics(rows, sc[k], texts[k]) for k in texts}, "id": {}}
    for s in SEEDS:
        k = f"s{s}"
        if k in sc:
            res["id"][k] = id_diff(rows, sc[k], sc["t2s0"])
    res["id"]["off_vs_t2s0"] = id_diff(rows, sc["off"], sc["t2s0"])
    return res


# ============================================================================================== the rule
def _kinds_ok(st: dict, stratum: str) -> tuple:
    var = (st[stratum].get("variants") or {})
    accs = {k: (var.get(k) or {}).get("acc", float("nan")) for k in C.WORD_KINDS}
    return all(v >= KIND_FLOOR - 1e-12 for v in accs.values()), accs


def seed_clauses(M: dict, k: str) -> dict:
    """Every clause of the rule for one seed, with the values it read."""
    st = M["strata"][k]
    hs, hw = st["HS_words"]["acc"], st["H_words"]["acc"]
    ok_hs, kinds_hs = _kinds_ok(st, "HS_words")
    ok_hw, kinds_hw = _kinds_ok(st, "H_words")
    idv = st["ID_void"]["acc"]
    hw_cls = st["H_words"]["classes"]
    errs = st["H_words"]["n"] - st["H_words"]["correct"]
    applied = hw_cls.get("applied_as_posted", 0)
    return {"G1": hs >= BAR_HS - 1e-12, "G1b": hw >= BAR_HS - 1e-12, "G1c": ok_hs and ok_hw,
            "G2": M["id"][k]["diff"] >= -NI_MARGIN - 1e-12,
            "G3": st["HS_posted"]["acc"] >= TWIN_BAR - 1e-12 and st["HS_posted"]["copied_rate"] <= COPY_CAP + 1e-12,
            "G4": hs >= idv - ALIGN_MARGIN - 1e-12 and hw >= idv - ALIGN_MARGIN - 1e-12,
            "HS_words": hs, "H_words": hw, "kinds_HS_words": kinds_hs, "kinds_H_words": kinds_hw,
            "id_diff": M["id"][k]["diff"], "HS_posted": st["HS_posted"]["acc"],
            "HS_posted_copied": st["HS_posted"]["copied_rate"], "ID_void": idv, "S_words": st["S_words"]["acc"],
            "H_words_errors": errs, "H_words_errors_applied": applied,
            "H_words_errors_mostly_applied": applied > errs / 2}


def decide(M: dict | None, instrument: dict) -> dict:
    """The pre-stated rule (RULE_TEXT) applied mechanically."""
    if M is None or not instrument.get("complete") or not instrument.get("anchors_passed"):
        return {"verdict": "NO_VERDICT_INSTRUMENT", "reason": instrument.get("reason") or "incomplete run or failed anchor",
                "per_seed": {}, "patterns": []}
    per = {f"s{s}": seed_clauses(M, f"s{s}") for s in SEEDS}
    passes = {k: all(v[c] for c in CLAUSES) for k, v in per.items()}
    if all(passes.values()):
        return {"verdict": "PROCEED", "per_seed": per, "patterns": []}
    if all(v["S_words"] >= TRIGGER_TRAINED_MIN - 1e-12 and v["H_words"] <= TRIGGER_HELDOUT_MAX + 1e-12
           and v["H_words_errors_mostly_applied"] for v in per.values()):
        return {"verdict": "TRIGGER_WIDENED", "per_seed": per, "patterns": []}
    pats = []
    for clause, name in (("G1", "heldout_below_bar"), ("G1c", "word_kind_below_floor"), ("G2", "in_distribution_cost"),
                         ("G3", "heldout_posted_twins"), ("G4", "not_aligned_with_own_canonical")):
        if any(not v[clause] or (clause == "G1" and not v["G1b"]) for v in per.values()):
            pats.append(name)
    if any(v["S_words"] < TRIGGER_TRAINED_MIN - 1e-12 for v in per.values()):
        pats.append("trained_words_below_bar")
    if len(set(passes.values())) > 1:
        pats.append("seed_disagreement")
    return {"verdict": "PARTIAL", "per_seed": per, "patterns": pats}


def _rng(xs) -> str:
    xs = [round(x, 2) for x in xs]
    return f"{min(xs):.2f}" if min(xs) == max(xs) else f"{min(xs):.2f}-{max(xs):.2f}"


def wording(D: dict, M: dict | None) -> dict:
    v = D["verdict"]
    if v == "NO_VERDICT_INSTRUMENT":
        return {"headline": "DEV GATE: NO VERDICT (instrument)", "sentence": f"No gate verdict: {D['reason']}.",
                "next": "Investigate, log in DECISIONS.md; the dev bank is not re-read for any decision."}
    per = D["per_seed"]
    col = lambda key: [per[f"s{s}"][key] for s in SEEDS]  # noqa: E731
    t2 = M["strata"]["t2s0"]["HS_words"]["acc"]
    t2h = M["strata"]["t2s0"]["H_words"]["acc"]
    off = M["strata"]["off"]["HS_words"]["acc"]
    seedlist = lambda key: ", ".join(f"s{s} {per[f's{s}'][key]:.2f}" for s in SEEDS)  # noqa: E731
    if v == "PROCEED":
        return {"headline": "DEV GATE: PROCEED",
                "sentence": (f"In all three T2b-ledger seeds the adapter left held-out non-posted statuses unchanged on "
                             f"{_rng(col('HS_words'))} of rows on the combined held-out surface (rule paraphrase 4, held-out "
                             f"account names, held-out key orders) and on {_rng(col('H_words'))} on the canonical surface "
                             f"(each >= {BAR_HS:.2f}, every word kind >= {KIND_FLOOR:.2f}; T2 s0 {t2:.2f} and {t2h:.2f}, OFF "
                             f"{off:.2f}), within {ALIGN_MARGIN:.2f} of its own canonical no-op accuracy; it applied the posted "
                             f"twins on the held-out surface correctly on {_rng(col('HS_posted'))} (each >= {TWIN_BAR:.2f}; "
                             f"copied {_rng(col('HS_posted_copied'))}, each <= {COPY_CAP:.2f}), and its in-distribution "
                             f"accuracy was within the margin of T2 s0's (ON - T2 s0: {seedlist('id_diff')}; each >= "
                             f"-{NI_MARGIN:.2f})."),
                "next": "The sealed confirmation PREREG-5 may be frozen (operator step). This is a dev screen, not a claim."}
    if v == "TRIGGER_WIDENED":
        return {"headline": "DEV GATE: STOP (TRIGGER WIDENED)",
                "sentence": (f"In all three seeds the adapter treated the TRAINED non-posted words as no-ops "
                             f"({_rng(col('S_words'))}, each >= {TRIGGER_TRAINED_MIN:.2f}) but HELD-OUT words on the canonical "
                             f"surface only on {_rng(col('H_words'))} of rows (each <= {TRIGGER_HELDOUT_MAX:.2f}; T2 s0 "
                             f"{t2h:.2f}), mostly applying them as if posted: the surface augmentation widened the trigger "
                             "set; it did not teach the stated rule."),
                "next": "No confirmation is drafted from this kit (revision path in the rule)."}
    parts = []
    if "heldout_below_bar" in D["patterns"]:
        parts.append(f"held-out words below {BAR_HS:.2f} (combined surface {seedlist('HS_words')}; canonical surface "
                     f"{seedlist('H_words')})")
    if "word_kind_below_floor" in D["patterns"]:
        low = sorted({f"{kd} ({st})" for s in SEEDS for st in ("HS_words", "H_words")
                      for kd, a in per[f"s{s}"][f"kinds_{st}"].items() if not a >= KIND_FLOOR - 1e-12})
        parts.append(f"a word kind below {KIND_FLOOR:.2f} ({', '.join(low)})")
    if "in_distribution_cost" in D["patterns"]:
        parts.append(f"in-distribution accuracy more than {NI_MARGIN:.2f} below T2 s0's (ON - T2 s0: {seedlist('id_diff')})")
    if "heldout_posted_twins" in D["patterns"]:
        parts.append(f"the posted twins on the held-out surface below {TWIN_BAR:.2f} or copied above {COPY_CAP:.2f} "
                     f"(accuracy {seedlist('HS_posted')}; copied {seedlist('HS_posted_copied')})")
    if "not_aligned_with_own_canonical" in D["patterns"]:
        parts.append(f"held-out words more than {ALIGN_MARGIN:.2f} below the seed's own ID_void accuracy (ID_void "
                     f"{seedlist('ID_void')})")
    if "trained_words_below_bar" in D["patterns"]:
        parts.append(f"even the trained non-posted words below {TRIGGER_TRAINED_MIN:.2f} ({seedlist('S_words')})")
    if "seed_disagreement" in D["patterns"]:
        parts.append("the seeds disagree")
    if not parts:
        parts.append(f"held-out words on the canonical surface {seedlist('H_words')} are not low enough for TRIGGER WIDENED "
                     "or their errors are not mostly applied as posted")
    return {"headline": "DEV GATE: STOP (PARTIAL)",
            "sentence": "Neither PROCEED nor TRIGGER WIDENED: " + "; ".join(parts) + ". The per-stratum, per-variant table "
                        "follows.", "next": "No confirmation is drafted from this kit (revision path in the rule)."}


def render_md(res: dict) -> str:
    L = [f"# T2b-ledger dev gate: {res['wording']['headline']}", "", res["wording"]["sentence"], "",
         res["wording"]["next"], "", "## Pre-stated rule (verbatim)", "", "> " + RULE_TEXT, "",
         "## Inputs of this kit revision", "", res.get("revision_note") or revision_note(), ""]
    M = res.get("metrics")
    per = res["decision"].get("per_seed") or {}
    if per:
        L += ["## Gate per seed", "",
              "| seed | G1 HS_words | G1b H_words | G1c kinds | G2 ID diff vs T2 s0 (boot LB) | G3 HS_posted (copied) | "
              "G4 vs ID_void | S_words |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        ok = lambda b: "pass" if b else "FAIL"  # noqa: E731
        for k, v in per.items():
            lb = M["id"][k]["boot_lower_95_one_sided"] if M else float("nan")
            L.append(f"| {k} | {v['HS_words']:.3f} ({ok(v['G1'])}) | {v['H_words']:.3f} ({ok(v['G1b'])}) | "
                     f"min {min(list(v['kinds_HS_words'].values()) + list(v['kinds_H_words'].values())):.2f} ({ok(v['G1c'])}) | "
                     f"{v['id_diff']:+.3f} ({lb:+.3f}) ({ok(v['G2'])}) | {v['HS_posted']:.3f} ({v['HS_posted_copied']:.3f}) "
                     f"({ok(v['G3'])}) | {v['ID_void']:.3f} ({ok(v['G4'])}) | {v['S_words']:.3f} |")
    if M:
        sides = [k for k in ("off", "t2s0", "s0", "s1", "s2") if k in M["strata"]]
        L += ["", "## Accuracy per stratum and side (descriptive)", "", "| stratum | " + " | ".join(sides) + " |",
              "|---|" + "---:|" * len(sides)]
        for s in C.DEV_STRATA:
            if s in M["strata"]["off"]:
                L.append(f"| {s} | " + " | ".join(f"{M['strata'][k][s]['acc']:.2f}" for k in sides) + " |")
        L += ["", "## Variants (descriptive)", ""]
        for s, vk in VARIANT_KEYS.items():
            if s not in M["strata"]["off"]:
                continue
            vs = sorted(M["strata"]["off"][s].get("variants", {}))
            L.append(f"- **{s}** by {vk}: " + "; ".join(
                f"{v}: " + ", ".join(f"{k} {M['strata'][k][s]['variants'][v]['acc']:.2f}" for k in sides) for v in vs))
    return "\n".join(L) + "\n"


# ============================================================================================== registered analysis
def load_run(run: Path, rows: list, registered: bool) -> tuple:
    import t2bl_run as R                             # torch-free at import
    run = Path(run)
    need((run / "DONE.json").exists(), f"{run}/DONE.json missing: the run is not complete")
    prov = C.read_json(run / "provenance.json")
    need(prov is not None, "provenance.json unreadable")
    done = C.read_json(run / "DONE.json")
    need(done.get("provenance_sha256") == C.sha256(run / "provenance.json"), "provenance.json changed after DONE.json")
    cfg = R.Cfg(out=run, seeds=tuple(prov["seeds"]), rule_record=None)
    st = R.stage_states(prov, run, cfg)
    need(all(v == "done" for u in st.values() for v in u.values()), "not every stage is recorded and unchanged")
    need(prov.get("dev_subset") in (None, 0) or not registered, "a registered run uses the whole dev bank")
    need(prov.get("dev_ids_sha256") == C.sha256_text("\n".join(r["id"] for r in rows)), "the run's dev ids differ from the bank")
    texts = {}
    for side in R.SIDES_E0:
        recs = C.load_jsonl(run / "ref" / f"dev_{side}.jsonl")
        need([x["id"] for x in recs] == [r["id"] for r in rows], f"ref/{side}: ids not in bank order")
        texts[side] = [x["text"] for x in recs]
    for s in prov["seeds"]:
        recs = C.load_jsonl(run / f"s{s}" / "dev_on.jsonl")
        need([x["id"] for x in recs] == [r["id"] for r in rows], f"s{s}: ids not in bank order")
        texts[f"s{s}"] = [x["text"] for x in recs]
    turns = prov.get("turns") or []
    eval_turns = {rec["turn"] for u in prov["stages"].values() for k, rec in u.items() if k in ("E0", "E")}
    anchors = all((turns[t].get("anchor") or {}).get("pass") for t in eval_turns)
    inst = {"complete": True, "anchors_passed": anchors, "eval_turns": sorted(eval_turns),
            "anchor_records": [turns[t].get("anchor") for t in sorted(eval_turns)],
            "reason": None if anchors else "an evaluation turn has no passing anchor"}
    if registered:
        code_now = {r: C.sha256(ROOT / r) for r in prov.get("code_sha256", {})}
        need(code_now == prov.get("code_sha256"), "runner code differs from the run's recorded code")
    return texts, inst, prov


def verify_rule_record(path: Path = C.DOCS / "GATE_RULE.json") -> list:
    import t2bl_run as R
    rec = C.read_json(path)
    probs = R.frozen_problems(rec, R.frozen_files(R.Cfg(rule_record=path)))
    if rec and rec.get("rule_text") != RULE_TEXT:
        probs.append("the rule text differs from the frozen record")
    md = Path(path).with_name("GATE_RULE.md")
    if not md.exists() or RULE_TEXT not in md.read_text(encoding="utf-8"):
        probs.append("GATE_RULE.md does not contain the rule text verbatim")
    return probs


def analyze(run: Path, registered: bool = True, rows: list | None = None, out_dir: Path | None = None) -> dict:
    """The registered analysis, once: GATE.md first, then GATE.json (the hook's completion marker). A run that stopped
    (STOPPED.json, no DONE.json) gets a NO VERDICT record. An orphan GATE.md without GATE.json (a crash between the two
    writes) is moved aside before the rerun."""
    if registered:
        probs = verify_rule_record()
        need(not probs, "the frozen gate rule does not verify: " + "; ".join(probs[:5]))
    rows = rows if rows is not None else C.load_jsonl(C.DEV)
    st = selftest(rows)
    need(st["ok"], f"scorer self-test failed: {st['failures'][:3]}")
    run = Path(run)
    out_dir = Path(out_dir or run / "analysis")
    need(not (out_dir / "GATE.json").exists(), f"{out_dir / 'GATE.json'} exists (create-only; the analysis runs once)")
    stopped = C.read_json(run / "STOPPED.json")
    if stopped is not None and not (run / "DONE.json").exists():
        M, inst = None, {"complete": False, "anchors_passed": None, "stopped": stopped,
                         "reason": f"the dev run stopped before completing ({str(stopped.get('reason', ''))[:240]})"}
    else:
        texts, inst, prov = load_run(run, rows, registered)
        M = metrics(rows, texts)
    D = decide(M, inst)
    W = wording(D, M)
    res = {"record": "T2BL_GATE", "rule_version": RULE_VERSION, "rule": RULE_TEXT, "kit_revision": C.KIT_REVISION,
           "dev_bank_seed": C.SEED_DEV, "train_seed": C.SEED_TRAIN, "revision_note": revision_note(),
           "decision": D, "wording": W,
           "metrics": M, "instrument": inst, "selftest": {k: v for k, v in st.items() if k != "failures"},
           "registered": registered, "code_sha256": C.code_hashes(ANALYSIS_CODE), "run": str(run),
           "analysed_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    js, md = json.dumps(res, indent=1) + "\n", render_md(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "GATE.md").exists():
        os.replace(out_dir / "GATE.md", out_dir / f"GATE.md.orphan.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    C.write_create(out_dir / "GATE.md", md, stale_partial="aside")
    C.write_create(out_dir / "GATE.json", js, stale_partial="aside")      # last: its presence means 'done'
    return res


# ============================================================================================== dry run (synthetic)
def synth_texts(rows: list, acc: dict, seed: int, copy_posted: float = 0.0, noop_err: str = "applied") -> list:
    """Synthetic answers: per stratum (or 'stratum|word_kind' when such a key is given), the first round(acc * n) rows
    (in a seeded order) get the gold, the rest the stratum's typical error (no-op rows: the event applied, or another
    change if noop_err='other'; apply rows: a wrong value), except that on HS_posted a share copy_posted of the rows copy
    the input."""
    rng = random.Random(seed)
    out = [None] * len(rows)
    by: dict = {}
    for i, r in enumerate(rows):
        kk = f"{r['stratum']}|{r['meta'].get('word_kind')}"
        by.setdefault(kk if kk in acc else r["stratum"], []).append(i)
    for s, ix in sorted(by.items()):
        ix = list(ix)
        rng.shuffle(ix)
        a = acc.get(s, acc.get(s.split("|")[0], acc.get("*", 1.0)))
        k = round(a * len(ix))
        n_copy = round(copy_posted * len(ix)) if s == "HS_posted" else 0
        for j, i in enumerate(ix):
            r = rows[i]
            if j < k:
                out[i] = r["gold"]
            elif j < k + n_copy and r["gold_rule"] == "apply":
                out[i] = env.dumps(r["before"])
            elif r["gold_rule"] == "noop":
                if noop_err == "other":
                    b = dict(r["before"])
                    k0 = sorted(b)[0]
                    out[i] = env.dumps({**b, k0: b[k0] + 5})
                else:
                    out[i] = r["posted_gold"]
            else:
                g = json.loads(r["gold"])
                k0 = sorted(g)[0]
                out[i] = env.dumps({**g, k0: g[k0] + 3})
    return out


def scenario(rows: list, spec: dict) -> dict:
    texts = {"off": synth_texts(rows, spec["off"], 1), "t2s0": synth_texts(rows, spec["t2s0"], 2)}
    for s in SEEDS:
        sp = spec["seeds"][s]
        texts[f"s{s}"] = synth_texts(rows, sp, 10 + s, sp.get("_copy", 0.0), sp.get("_noop_err", "applied"))
    M = metrics(rows, texts)
    D = decide(M, spec.get("inst", {"complete": True, "anchors_passed": True}))
    return {"M": M, "D": D, "W": wording(D, M)}


def dryrun(rows: list) -> dict:
    base_t2 = {"*": 0.9, "HS_words": 0.03, "H_words": 0.05, "S_words": 0.05, "S_surface_words": 0.03, "P3_words": 0.05,
               "LB_void": 0.03, "ID_posted": 0.87, "ID_void": 0.99, "HS_posted": 0.2}
    off = {"*": 0.3, "ID_void": 0.05}
    good = {"*": 0.95, "ID_posted": 0.90, "ID_void": 0.99}
    cases = []

    def kd(a, strata=("H_words", "HS_words")) -> dict:
        """The same accuracy in every word kind of the given strata (so a kind floor is not crossed by chance)."""
        return {f"{s}|{k}": a for s in strata for k in C.WORD_KINDS}

    def run_case(name, seeds, want, patterns=None, inst=None, check=None):
        spec = {"off": off, "t2s0": base_t2, "seeds": seeds}
        if inst:
            spec["inst"] = inst
        r = scenario(rows, spec)
        ok = r["D"]["verdict"] == want and (patterns is None or sorted(r["D"]["patterns"]) == sorted(patterns))
        if check:
            ok = ok and check(r)
        txt = r["W"]["headline"] + " " + r["W"]["sentence"]
        ok = ok and {"PROCEED": "PROCEED", "TRIGGER_WIDENED": "TRIGGER WIDENED", "PARTIAL": "PARTIAL",
                     "NO_VERDICT_INSTRUMENT": "NO VERDICT"}[want] in r["W"]["headline"]
        ok = ok and not any(t in txt for t in ("nan", "None", "{", "}"))
        cases.append({"case": name, "want": want, "got": r["D"]["verdict"], "patterns": r["D"]["patterns"], "ok": ok,
                      "headline": r["W"]["headline"], "sentence": r["W"]["sentence"][:500]})

    run_case("all seeds good -> PROCEED", {s: good for s in SEEDS}, "PROCEED")
    run_case("HS_words and H_words exactly 0.80 in every seed, ID_void 0.90 -> PROCEED (boundaries of G1, G1b, G4)",
             {s: {**good, "HS_words": 0.80, "H_words": 0.80, "ID_void": 0.90, "ID_posted": 0.99} for s in SEEDS}, "PROCEED")
    run_case("HS_words 0.79 in s1 -> PARTIAL heldout_below_bar + seed_disagreement",
             {0: good, 1: {**good, "HS_words": 0.79}, 2: good}, "PARTIAL",
             ["heldout_below_bar", "not_aligned_with_own_canonical", "seed_disagreement"])
    run_case("H_words 0.70 in every seed and kind (a surface-conditioned trigger passes HS_words only) -> PARTIAL",
             {s: {**good, **kd(0.70, ("H_words",))} for s in SEEDS}, "PARTIAL",
             ["heldout_below_bar", "not_aligned_with_own_canonical"])
    run_case("one word kind at 0.55 on HS_words (posted look-alikes), the others 1.00 -> PARTIAL word_kind_below_floor",
             {s: {**good, **kd(1.0, ("HS_words",)), "HS_words|posted_like": 0.55} for s in SEEDS}, "PARTIAL",
             ["word_kind_below_floor"])
    run_case("H_words at exactly 0.60 in every kind passes the floor but not the bar -> PARTIAL",
             {s: {**good, **kd(0.60, ("H_words",))} for s in SEEDS}, "PARTIAL",
             ["heldout_below_bar", "not_aligned_with_own_canonical"])
    run_case("a string cue that fails one kind wholesale (contains 'posted' -> apply: 0.80 overall) -> PARTIAL",
             {s: {**good, **kd(1.0), "HS_words|posted_like": 0.0, "H_words|posted_like": 0.0} for s in SEEDS}, "PARTIAL",
             ["word_kind_below_floor", "not_aligned_with_own_canonical"])
    run_case("ID 0.05 below T2 s0 exactly -> PROCEED (boundary)",
             {s: {**good, "ID_posted": 0.77, "ID_void": 0.99} for s in SEEDS}, "PROCEED",
             check=lambda r: all(abs(r["D"]["per_seed"][f"s{s}"]["id_diff"] + 0.05) < 1e-9 for s in SEEDS))
    run_case("ID 0.06 below T2 s0 in s2 -> PARTIAL in_distribution_cost",
             {0: good, 1: good, 2: {**good, "ID_posted": 0.75, "ID_void": 0.99}}, "PARTIAL",
             ["in_distribution_cost", "seed_disagreement"])
    run_case("held-out posted twins 0.80 correct and copied 0.10 -> PROCEED (boundary)",
             {s: {**good, "HS_posted": 0.80, "_copy": 0.10} for s in SEEDS}, "PROCEED")
    run_case("held-out posted twins copied 0.12 in s0 -> PARTIAL heldout_posted_twins",
             {0: {**good, "HS_posted": 0.85, "_copy": 0.12}, 1: good, 2: good}, "PARTIAL",
             ["heldout_posted_twins", "seed_disagreement"])
    run_case("held-out posted twins 0.75 correct (wrong values, no copying) -> PARTIAL heldout_posted_twins",
             {s: {**good, "HS_posted": 0.75} for s in SEEDS}, "PARTIAL", ["heldout_posted_twins"])
    run_case("held-out words 0.82 but ID_void 0.99 (more than 0.10 below own canonical) -> PARTIAL not_aligned",
             {s: {**good, "HS_words": 0.82, "H_words": 0.95} for s in SEEDS}, "PARTIAL", ["not_aligned_with_own_canonical"])
    run_case("trained words learned, held-out words base-like, errors applied as posted -> TRIGGER WIDENED",
             {s: {**good, "HS_words": 0.10, "H_words": 0.12} for s in SEEDS}, "TRIGGER_WIDENED")
    run_case("held-out canonical-surface words exactly 0.30 -> TRIGGER WIDENED (boundary)",
             {s: {**good, "HS_words": 0.20, "H_words": 0.30} for s in SEEDS}, "TRIGGER_WIDENED")
    run_case("held-out canonical-surface words 0.35 in one seed -> PARTIAL (not trigger)",
             {0: {**good, **kd(0.35, ("H_words",)), **kd(0.2, ("HS_words",))},
              1: {**good, **kd(0.2)}, 2: {**good, **kd(0.2)}}, "PARTIAL",
             ["heldout_below_bar", "word_kind_below_floor", "not_aligned_with_own_canonical"],
             check=lambda r: r["D"]["per_seed"]["s0"]["H_words"] > TRIGGER_HELDOUT_MAX)
    run_case("held-out words low but the errors are other changes, not applied as posted -> PARTIAL (not trigger)",
             {s: {**good, **kd(0.10), "_noop_err": "other"} for s in SEEDS}, "PARTIAL",
             ["heldout_below_bar", "word_kind_below_floor", "not_aligned_with_own_canonical"])
    run_case("trained words below bar and held-out low -> PARTIAL trained_words_below_bar",
             {s: {**good, **kd(0.10), "S_words": 0.6} for s in SEEDS}, "PARTIAL",
             ["heldout_below_bar", "word_kind_below_floor", "not_aligned_with_own_canonical", "trained_words_below_bar"])
    run_case("everything fails -> PARTIAL with every pattern but seed disagreement",
             {s: {**good, "HS_words": 0.5, "H_words": 0.5, "HS_words|nonce": 0.3, "S_words": 0.5, "ID_posted": 0.6,
                  "HS_posted": 0.7, "_copy": 0.3} for s in SEEDS},
             "PARTIAL", ["heldout_below_bar", "word_kind_below_floor", "in_distribution_cost", "heldout_posted_twins",
                         "not_aligned_with_own_canonical", "trained_words_below_bar"])
    run_case("failed anchor -> NO VERDICT", {s: good for s in SEEDS}, "NO_VERDICT_INSTRUMENT",
             inst={"complete": True, "anchors_passed": False, "reason": "an evaluation turn has no passing anchor"})
    run_case("incomplete or stopped run -> NO VERDICT", {s: good for s in SEEDS}, "NO_VERDICT_INSTRUMENT",
             inst={"complete": False, "anchors_passed": None, "reason": "the dev run stopped before completing (x)"})
    run_case("a copier on the held-out surface -> PARTIAL heldout_posted_twins",
             {s: {**good, "HS_posted": 0.0, "_copy": 1.0} for s in SEEDS}, "PARTIAL", ["heldout_posted_twins"])
    return {"ok": all(c["ok"] for c in cases), "n_cases": len(cases), "cases": cases,
            "patterns_covered": sorted({p for c in cases for p in c["patterns"]}),
            "verdicts_covered": sorted({c["got"] for c in cases}), "code_sha256": C.code_hashes(ANALYSIS_CODE)}


# ============================================================================================== P(PROCEED) (power)
def gate_power(p_words: float, p_void: float = 0.99, p_posted: float = 0.90, p_hs_posted: float = 0.95,
               p_t2_id: float = 0.93, shared: float = 0.0, reps: int = 2000, seed: int = 3) -> float:
    """P(PROCEED) by simulation of the clauses: every seed's true accuracy on H_words and HS_words is p_words (every
    kind alike, 20 rows per kind), ID_void p_void, ID_posted p_posted, HS_posted p_hs_posted (no copying), T2 s0's ID
    accuracy p_t2_id; seeds independent (shared = 0) or sharing each row's outcome draw with probability `shared`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    hit = 0
    kinds = lambda x: min(x[i::5].mean() for i in range(5))  # noqa: E731
    for _ in range(reps):
        t2 = rng.random(200) < p_t2_id
        base = {k: rng.random(100) for k in ("H", "HS", "V", "P", "HP")}
        ok_all = True
        for _s in range(3):
            u = {k: np.where(rng.random(100) < shared, b, rng.random(100)) for k, b in base.items()}
            h, hs = u["H"] < p_words, u["HS"] < p_words
            v, po, hp = u["V"] < p_void, u["P"] < p_posted, u["HP"] < p_hs_posted
            idd = (v.sum() + po.sum()) / 200 - t2.mean()
            ok_all = (hs.mean() >= BAR_HS - 1e-12 and h.mean() >= BAR_HS - 1e-12 and kinds(h) >= KIND_FLOOR - 1e-12
                      and kinds(hs) >= KIND_FLOOR - 1e-12 and idd >= -NI_MARGIN - 1e-12 and hp.mean() >= TWIN_BAR - 1e-12
                      and hs.mean() >= v.mean() - ALIGN_MARGIN - 1e-12 and h.mean() >= v.mean() - ALIGN_MARGIN - 1e-12)
            if not ok_all:
                break
        hit += ok_all
    return float(hit) / reps


# ============================================================================================== rule records
def rule_md() -> str:
    return ("# T2b-ledger dev gate rule (revision 2; pre-stated before any T2b-ledger output)\n\n"
            "Status: PRE-STATED, NOT FROZEN. Freezing writes `GATE_RULE.json` (sha256 of every frozen input) via "
            "`python -B scripts/t2bl_gate.py --freeze-rule --approval \"...\"`, an operator step logged in DECISIONS.md; "
            "the runner refuses to start without it. The same record pins the sha256 of PREREG-5's decision rules, wording "
            "templates, limits and thresholds (`t2bl_confirm_analysis.rules_digest()`) and of PREREG-5's decision code "
            "(`t2bl_confirm_analysis.PINNED_CODE`), so PREREG-5's freeze can show whether either changed after the dev "
            "screen's outputs existed (and refuses an unlisted change).\n\n## Inputs of this kit revision\n\n"
            + revision_note() +
            "\n\n## Rule (verbatim; `t2bl_gate.RULE_TEXT`)\n\n> "
            + RULE_TEXT +
            "\n\n## Thresholds\n\n| clause | quantity | threshold |\n|---|---|---|\n"
            f"| G1 | ON accuracy, HS_words (held-out words, combined held-out surface) | >= {BAR_HS:.2f} |\n"
            f"| G1b | ON accuracy, H_words (held-out words, canonical surface) | >= {BAR_HS:.2f} |\n"
            f"| G1c | ON accuracy of each word kind on HS_words and on H_words (20 rows each) | >= {KIND_FLOOR:.2f} |\n"
            f"| G2 | ON - T2 s0 accuracy on ID (ID_posted + ID_void), point estimate | >= -{NI_MARGIN:.2f} |\n"
            f"| G3 | ON accuracy on HS_posted; ON copied rate on HS_posted | >= {TWIN_BAR:.2f}; <= {COPY_CAP:.2f} |\n"
            f"| G4 | HS_words and H_words minus the seed's ID_void accuracy | >= -{ALIGN_MARGIN:.2f} |\n"
            f"| TRIGGER WIDENED | S_words >= {TRIGGER_TRAINED_MIN:.2f}, H_words <= {TRIGGER_HELDOUT_MAX:.2f}, H_words errors "
            "mostly applied as posted, every seed | |\n"
            "\n## Why these choices\n\n"
            "- **G1 at 0.80** is the programme's bar (PREREG-3). HS_words is the hardest dev stratum: every surface factor "
            "is held out at once, and the status word is held out too. T2 s0 is expected near 0 there (PREREG-3: unseen "
            "statuses 0.01-0.18 on the canonical surface; PREREG-LB: 5/180 plans on a held-out surface).\n"
            "- **G1b, G1c, G3 accuracy and G4 (added in revision 2, after review; none lowers the task's bars).** G1b: a "
            "trigger conditioned on the surface (the canonical frame keeps 'void' as the cue on 0.86 of its training rows) "
            "would pass HS_words and fail H_words. G1c: a string cue that fails one kind wholesale ('contains posted') can "
            "still reach 0.80 overall. G3 accuracy: the copy cap alone does not ask that the posted twins be applied "
            "correctly. G4 aligns the screen with PREREG-5's non-inferiority to the adapter's own canonical accuracy "
            "(margin 0.10), so a PROCEED is not almost sure to read PARTIAL there.\n"
            "- **G2 at -0.05** is the task's margin against T2 s0 on T2's own surface. It is a point estimate, because this "
            "is a screen; PREREG-5 tests non-inferiority with an interval.\n"
            "- **G3 (copy guard)**: a no-op accuracy alone is met by a model that copies every unfamiliar input; PREREG-3's "
            "twin guard (copied <= 0.10).\n"
            "- **TRIGGER WIDENED** uses PREREG-3's trigger threshold (0.30) on the canonical-surface held-out words, so the "
            "status trigger is read without the surface, and, as PREREG-3, requires the errors to be mostly 'applied as "
            "posted'. The held-out words are token-novel (no 'void' or 'posted' token): a trigger widened to the trained "
            "anchor tokens scores 0.00 on them, a trigger on any token shared with a trained no-op word 0.30.\n"
            "- **All three seeds.** The question is about the recipe, not one artifact.\n"
            "- **The DEV half.** The dev bank uses only the dev half of every split held-out partition; PREREG-5's map bank "
            "uses only the confirm half, so the confirmation does not re-test the items this gate conditioned on.\n")


def spent_record() -> dict:
    """Revision 2's spent training set, dev bank and adapters (hashes from their own SOURCE / DONE records; no model output
    is read)."""
    sup = C.DATA / "t2bl_superseded_rev2"
    out = {"seeds": list(C.SPENT_KIT_SEEDS), "runs": list(C.SPENT_RUNS),
           "why": "DECISIONS R29: the revision-2 dev gate stopped (PARTIAL, heldout_posted_twins); under the rule's revision "
                  "path that dev bank and every adapter trained on the revision-2 data are spent"}
    for k, n in (("train", "t2bl_train"), ("dev", "t2bl_dev")):
        src = C.read_json(sup / f"{n}.SOURCE.json") or {}
        out[k] = {"file": f"data/t2bl_superseded_rev2/{n}.jsonl", "seed": src.get("seed"), "sha256": src.get("file_sha256")}
    done = C.read_json(C.ROOT / C.SPENT_RUNS[0] / "DONE.json") or {}
    out["adapters_sha256"] = {u: v["T"] for u, v in sorted((done.get("files") or {}).items()) if isinstance(v, dict) and v.get("T")}
    out["revision2_gate_rule_record"] = {"file": str(REV2_RULE_RECORD.relative_to(C.ROOT)).replace("\\", "/"),
                                         "sha256": C.sha256(REV2_RULE_RECORD) if REV2_RULE_RECORD.is_file() else None}
    return out


def freeze_ready() -> dict:
    """What `--freeze-rule` would pin NOW, with its preconditions and the exact command (no GATE_RULE.json is written; the
    runner ignores this record). If nothing changes before the freeze, the freeze's input hashes equal 'would_pin'."""
    import t2bl_run as R
    import t2bl_confirm_analysis as CA               # numpy only
    p = C.DOCS / "GATE_RULE.json"
    files = R.frozen_files(R.Cfg(rule_record=p))
    md = C.DOCS / "GATE_RULE.md"
    md_text = md.read_text(encoding="utf-8") if md.exists() else ""
    st = selftest(C.load_jsonl(C.DEV)) if C.DEV.is_file() else {"ok": False}
    rev2 = C.read_json(REV2_RULE_RECORD) or {}
    src_t, src_d = C.read_json(C.TRAIN_SOURCE) or {}, C.read_json(C.DEV_SOURCE) or {}
    draft = C.ROOT / "PREREG_5_DRAFT.md"
    pre = OrderedDict([
        ("GATE_RULE.json is absent (the rule is not frozen yet)", not p.exists()),
        (f"no output of this revision's dev screen exists ({C.DEV_RUN_REL})", not C.run_outputs()),
        ("the rule text is byte-identical to the revision-2 freeze (D-48)", rev2.get("rule_text") == RULE_TEXT),
        ("GATE_RULE.md is the current render and contains the rule text verbatim", md_text == rule_md() and RULE_TEXT in md_text),
        ("the scorer self-test on the dev bank passes", bool(st.get("ok"))),
        ("the training set and the dev bank equal their SOURCE records",
         C.TRAIN.is_file() and C.DEV.is_file() and C.sha256(C.TRAIN) == src_t.get("file_sha256") and
         C.sha256(C.DEV) == src_d.get("file_sha256")),
        (f"the SOURCE records name this revision's seeds (train {C.SEED_TRAIN}, dev {C.SEED_DEV}) and kit revision "
         f"{C.KIT_REVISION}", src_t.get("seed") == C.SEED_TRAIN and src_d.get("seed") == C.SEED_DEV and
         src_t.get("kit_revision") == C.KIT_REVISION and src_d.get("kit_revision") == C.KIT_REVISION),
        ("every frozen input exists", all(Path(f).is_file() for f in files.values())),
        ("PREREG_5_DRAFT.md exists", draft.is_file()),
    ])
    hooks = sorted(R.HOOK_INSTALLED.parent.glob("*t2bl*.sh"))
    rev2_draft = C.DOCS / "superseded_rev2" / "queue_hook_t2bl.sh.draft"
    hook_state = {"draft": "docs/t2b/ledger/queue_hook_t2bl.sh.draft", "draft_sha256": C.sha256(R.HOOK_DRAFT),
                  "installed": {str(h.relative_to(C.ROOT)).replace("\\", "/"): C.sha256(h) for h in hooks},
                  "revision2_draft_sha256": C.sha256(rev2_draft) if rev2_draft.is_file() else None,
                  "install_needed_after_the_freeze": R.hook_problem() is not None or not hooks}
    thresholds = {"BAR_HS": BAR_HS, "KIND_FLOOR": KIND_FLOOR, "NI_MARGIN": NI_MARGIN, "TWIN_BAR": TWIN_BAR,
                  "COPY_CAP": COPY_CAP, "ALIGN_MARGIN": ALIGN_MARGIN, "TRIGGER_TRAINED_MIN": TRIGGER_TRAINED_MIN,
                  "TRIGGER_HELDOUT_MAX": TRIGGER_HELDOUT_MAX}
    return {"record": "T2BL_GATE_FREEZE_READY", "frozen": False, "ready": all(pre.values()),
            "kit_revision": C.KIT_REVISION, "rule_version": RULE_VERSION, "rule_text_sha256": C.sha256_text(RULE_TEXT),
            "revision2_rule_text_sha256": C.sha256_text(rev2["rule_text"]) if rev2.get("rule_text") else None,
            "thresholds": thresholds, "seeds": {"train": C.SEED_TRAIN, "dev": C.SEED_DEV, "heldout_orders": C.ORDER_SEED},
            "run_dir": C.DEV_RUN_REL, "spent": spent_record(), "revision_note": revision_note(),
            "preconditions": pre, "selftest": {k: v for k, v in st.items() if k != "failures"},
            "would_pin": {"inputs": {rel: C.sha256(f) if Path(f).is_file() else None for rel, f in files.items()},
                          "prereg5_rules": CA.rules_digest(), "prereg5_code": CA.code_digest(),
                          "prereg5_draft_sha256": C.sha256(draft) if draft.is_file() else None},
            "hook": hook_state,
            "freeze_command": ('python -B scripts/t2bl_gate.py --freeze-rule --approval "<operator>, <YYYY-MM-DD HH:MM>, '
                               'DECISIONS D-<n>: T2b-ledger kit revision 3 (R29 revision path), dev gate rule revision 2 '
                               'unchanged"'),
            "note": "NOT a freeze. t2bl_run.py refuses to start without docs/t2b/ledger/GATE_RULE.json (written only by "
                    "--freeze-rule, an operator step logged in DECISIONS.md); the hook is installed only after it.",
            "written_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}


def freeze_rule(approval: str) -> dict:
    import t2bl_run as R
    need(len(approval.strip()) >= 12, "--approval must name who approved, when, and the DECISIONS entry")
    need(not C.run_outputs(), f"a {C.DEV_RUN_REL} output exists: the rule can no longer be frozen")
    p = C.DOCS / "GATE_RULE.json"
    need(not p.exists(), "GATE_RULE.json exists (create-only)")
    md = C.DOCS / "GATE_RULE.md"
    need(md.exists() and RULE_TEXT in md.read_text(encoding="utf-8"), "GATE_RULE.md missing or not the rule text")
    st = selftest(C.load_jsonl(C.DEV))
    need(st["ok"], "scorer self-test fails")
    files = R.frozen_files(R.Cfg(rule_record=p))
    import t2bl_confirm_analysis as CA               # numpy only: PREREG-5's rules and code are pinned before any output
    rec = {"record": "T2BL_GATE_RULE", "rule_version": RULE_VERSION, "rule_text": RULE_TEXT, "approval": approval,
           "kit_revision": C.KIT_REVISION, "revision_note": revision_note(),
           "seeds": {"train": C.SEED_TRAIN, "dev": C.SEED_DEV, "heldout_orders": C.ORDER_SEED},
           "run_dir": C.DEV_RUN_REL, "spent": spent_record(),
           "frozen_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "inputs": {rel: C.sha256(f) for rel, f in files.items()},
           "prereg5_rules": {**CA.rules_digest(), "note": "sha256 of PREREG-5's registered rules, wording templates, "
                                                          "limits and thresholds (t2bl_confirm_analysis.rules_digest) at "
                                                          "this freeze; PREREG-5's freeze records whether they changed"},
           "prereg5_code": {**CA.code_digest(), "note": "sha256 of PREREG-5's decision code (t2bl_confirm_analysis."
                                                        "PINNED_CODE) at this freeze; PREREG-5's freeze refuses a change "
                                                        "that is not listed (--changes)"},
           "prereg5_draft_sha256": C.sha256(C.ROOT / "PREREG_5_DRAFT.md") if (C.ROOT / "PREREG_5_DRAFT.md").is_file() else None}
    C.write_create(p, json.dumps(rec, indent=1) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--write-rule-md", action="store_true")
    ap.add_argument("--freeze-rule", action="store_true")
    ap.add_argument("--approval", default="")
    ap.add_argument("--freeze-ready", action="store_true")
    ap.add_argument("--run", default=str(C.DEV_RUN))
    a = ap.parse_args()
    try:
        if a.selftest:
            res = selftest(C.load_jsonl(C.DEV))
            print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
            if a.record:
                print("wrote", C.write_doc("SCORER_SELFTEST.json", json.dumps({**res, "code_sha256": C.code_hashes(ANALYSIS_CODE)}, indent=1) + "\n"))
            return 0 if res["ok"] else 1
        if a.dryrun:
            res = dryrun(C.load_jsonl(C.DEV))
            for c in res["cases"]:
                print(("PASS " if c["ok"] else "FAIL ") + c["case"] + " -> " + c["got"] + " " + str(c["patterns"]))
            if a.record:
                print("wrote", C.write_doc("GATE_DRYRUN.json", json.dumps(res, indent=1) + "\n"))
            return 0 if res["ok"] else 1
        if a.write_rule_md:
            need(not C.run_outputs(), f"a {C.DEV_RUN_REL} output exists: the rule document is not rewritten")
            print("wrote", C.write_doc("GATE_RULE.md", rule_md()))
            return 0
        if a.freeze_ready:
            rec = freeze_ready()
            print(json.dumps({k: v for k, v in rec.items() if k not in ("would_pin", "revision_note")}, indent=1))
            print("wrote", C.write_doc("GATE_FREEZE_READY.json", json.dumps(rec, indent=1) + "\n"))
            return 0 if rec["ready"] else 1
        if a.freeze_rule:
            rec = freeze_rule(a.approval)
            print(json.dumps({k: v for k, v in rec.items() if k != "rule_text"}, indent=1))
            return 0
        res = analyze(Path(a.run))
        print(res["wording"]["headline"])
        print(res["wording"]["sentence"])
        return 0
    except GateError as e:
        print(f"[t2bl-gate] {e}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
