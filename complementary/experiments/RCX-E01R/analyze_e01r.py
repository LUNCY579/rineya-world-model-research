"""RCX-E01R SEALED analysis (PREREG_E01R sections 5-7; branch B4, CX D-39): E-R2 and E-R3 against N-CFSD on the opened
SEALED battery (surfaces `in` + `p3`). Run once, after open-sealed and every SEALED eval:

  python -B experiments/RCX-E01R/analyze_e01r.py [--cells runs/RCX-E01R/cells] [--missing TAG[,TAG...]]

Runs once (review item 5). The outputs have a fixed path next to the sealed records: <cells>/ANALYSIS_E01R.md and
<cells>/ANALYSIS_E01R.json, preceded by the create-only marker <cells>/ANALYSIS_E01R.started. The run refuses if any of
the three exists. The marker is written after the custody checks below (refusals only: they compute no statistic and
print or write nothing, so a refused run may be repeated once its cause is fixed) and BEFORE any statistic is computed;
a marker without outputs is a crashed analysis (a logged deviation). --missing names exactly the cells declared
missing in DECISIONS (PREREG section 10); default none. Pure Python (no torch, no model, no tokenizer); the battery is
rebuilt in memory by native_e01r.sealed_battery, the only P3 read, and no P3 text, prompt or model output is written
or printed.

Order of work. Every refusal is a SystemExit raised before any statistic is computed:
  1. SEALED_COMMIT.json and SEALED_OPEN.json exist; the commitment binds every file of the section 4 bound-code list
     and each bound file's current sha256 equals the committed one, this file included (verify_bound_code);
  2. the committed arms are branch B4's: treatment_arms = E-R2, E-R3 less any arm dropped at the mint (with its reason),
     arms_required = N-CFSD_s0-s2 plus s0-s2 of each treatment arm (committed_arms);
  3. native_e01r.sealed_battery rebuilds the battery and checks the salt, seed, commitment sha, skeleton commitment, P3
     sha and the rendered battery's sha against SEALED_OPEN.json; open-sealed recorded an adapter sha for every
     confirmatory cell and the committed max_new;
  4. per cell (check_cell): the eval names the rendered battery (sha, seed, size, SEALED_OPEN.json sha) and its tag;
     the adapter file, the eval and open-sealed agree on the adapter sha; the eval used the committed max_new; no
     eval_sealed_<tag>.json, TRAIN_META.json or SMOKE_<tag>.json records a sha for a bound file that differs from the
     commitment; the raw generations file is the one the eval recorded and holds the preds' outputs; every stored score
     (and R3 trace diagnostic) equals a re-score of the raw output with the arm's frozen parser; the truncation is
     re-derived from the raw file (native_e01r.raw_finished, i.e. map_finished over its prompt token lengths and
     decode log) and equals the preds' `finished` flags and the eval's count. If that mapping fails, the eval must
     record truncated = null and truncation_unknown = true (native_e01r.score_outputs); the cell is accepted and its
     truncation is reported as unknown (review items 10 and 11);
  5. --missing names exactly the confirmatory cells without a complete (present and passing) eval: a named cell with a
     complete eval, or an unnamed confirmatory cell without one, is refused. OFF (descriptive) may be absent. Each
     named cell also needs native_e01r's attempts log (ATTEMPTS_E01R.jsonl) to show the section 10 attempt limit for
     its eval: at least 1 + 2 failed attempts (an attempt without an 'ok' end). Reading taken: a cell whose eval
     completed (an 'ok' attempt) and whose outputs later fail an integrity check (section 10 reasons 3 and 4) cannot
     be re-run, because the eval outputs are create-only, so its completed attempt suffices;
  6. parents in SEALED_OPEN.json train_overlap.parent_ids are dropped from the battery and every cell;
  7. each quantity's variant list and predicate (section 5.2) select the same records, with the stated records per
     parent (check_selection).
Then: C1/C2 and G1-G6 per treatment arm (one rcx.stats.seed_parent_bootstrap_test call each: 10,000 resamples, seed
20261265, alpha 0.05, p_floor None = 1/10,000, ties "both"), Holm over {C1, C2} x the committed treatment arms (p = 1
for a test with a missing cell), decide() per arm and decide_experiment(); then the descriptive extras of sections
5.4-5.6 (sub-quantities and their labels, the symmetric last-line re-score, per form / per template, the
template-cluster sensitivity with seed 20261268, head-to-head and OFF contrasts, seeds, behaviour, R3 trace
diagnostics, generation, covariates). Only decide() and decide_experiment() decide; everything else is descriptive.

Readings this code takes where PREREG_E01R leaves a detail open (reported with the code; none changes a threshold):
  - TRAIN_META.json bound-code check: applied to the E01R treatment adapters (trained under the commitment). The control
    N-CFSD's TRAIN_META.json files were written by E01N before the commitment existed (they record E01N-time shas of
    native_e01.py: s0 f023c201..., s1/s2 e4f630e0...), so they cannot match it; the control's custody is the CX D-30
    adapter sha chain (open-sealed record = adapter file = eval record), checked here.
  - Descriptive estimates for an arm with a missing cell use the seed indices both it and N-CFSD have; they never enter
    Holm (p = 1) or a decision.
  - The template-cluster bootstrap draws, per replicate, forms in the order LF, CF, IM; within a form the POS templates
    (id-sorted) and within a template its C1 parents (id-sorted), each with random.Random(20261268).randrange.
  - Arms against OFF: C1 and C2 only. E-R2 - E-R3 head to head: C1, C2 and G1-G6.
  - Per P3 template: id, polarity, form, parents, records and, per arm, decision counts on its no-op and apply items
    (pooled over seeds); never text.
  - R2 duplicate comp_listed twins (section 5.5 item 2) are counted from train_E-R2.jsonl (when its sha equals the
    commitment's): a comp_listed row whose semantic hash and system turn (the rendered rule wording) equal those of a
    base or status_swap row of the same parent.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import trace_target as TT  # noqa: E402  (pure Python)
from rcx import stats as ST  # noqa: E402
from rcx.rcl import score as S  # noqa: E402

EXPERIMENT = "RCX-E01R"
CONTROL = "N-CFSD"
BRANCH = "B4"  # CX D-39
BRANCH_TREATMENTS = ("E-R2", "E-R3")  # section 4: X1 = E-R2, X2 = E-R3
SEEDS = (0, 1, 2)
OFF = "OFF"
PREREG_RESAMPLES = 10000  # section 6
RESAMPLES = PREREG_RESAMPLES  # what main() uses (tests may lower it; the report then says it is not the prereg run)
BOOT_SEED = 20261265  # section 6 / 9: every contrast of sections 5-7
TEMPLATE_BOOT_SEED = 20261268  # section 5.6 / 9: template-cluster sensitivity (descriptive)
ALPHA = 0.05
P_FLOOR = None  # rcx.stats default: 1 / resamples = 1/10,000 (section 6)
P_TIES = "both"  # replicates within rcx.stats.ZERO_TOL of 0 count on both sides (section 6)
EPS = 1e-9  # section 6 tolerance
C1_PRACTICAL = 0.25  # ADVANCE: d(C1) >= +0.25 (CX D-38, every branch)
C1_PER_SEED = 0.10  # ADVANCE: d_k(C1) >= +0.10 for k = 0, 1, 2
STOP_C1_HI = 0.10  # STOP: hi(C1) < +0.10
STOP_C2_HI = 0.05  # STOP: hi(C2) < +0.05
DEGENERATE_POINT = -0.15  # STOP (degenerate trade-off): a G4-G6 point estimate <= -0.15 (CX D-38)
NONANCHOR_LABEL_POINT = -0.15  # section 5.4 label: G4' or G5' point <= -0.15 (descriptive)
GUARD_RULES = (("G1", "lo", ">", -0.03), ("G2", "lo", ">", -0.05), ("G3", "hi", "<=", 0.02),
               ("G4", "lo", ">", -0.05), ("G5", "lo", ">", -0.10), ("G6", "lo", ">", -0.08))  # CX D-38
DEGENERATE_GUARDS = ("G4", "G5", "G6")
T975 = {1: 12.706204736174698, 2: 4.302652729911275}  # Student t 0.975 quantile by df (descriptive seed intervals)

F_STRATA = ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike")
V_STRATA = ("V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V7_bigset")
STRATA = F_STRATA + V_STRATA  # = run_e01r.STRATA (checked against the commitment)
COMPOSITE = ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike", "V1_pos", "V2_neg", "V3_inverted",
             "V5_posted_override")  # E01N's composite
FORMS = ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")
FORM_SHORT = {"EXPLICIT_LISTED_FIRST": "LF", "EXPLICIT_COMPLEMENT_FIRST": "CF", "IMPLICIT": "IM"}
SURFACES = ("in", "p3")
SCORE_KEYS = ("exact", "intent", "decision_ok", "err")
ERR_KINDS = ("applied_when_noop", "false_copy", "arith", "format")
MANIFEST, AUDIT = "TRAIN_MANIFEST_E01R.json", "AUDIT_E01R.json"

INC_MISSING = "INCONCLUSIVE (missing)"
STOP_DEG = "STOP (degenerate trade-off)"
ORDER = ("ADVANCE", "STOP", STOP_DEG, "REVISE")  # section 7: the first that holds decides; else INCONCLUSIVE
STOPS = ("STOP", STOP_DEG)
PRIMARY = ("C1", "C2")
GUARDS = tuple(g for g, *_ in GUARD_RULES)
SUBS = ("C1-nF", "G4'", "G5'")

# Section 4 bound-code list: file name -> path relative to complementary/. The commitment must bind every one of them.
BOUND_CODE = {
    "native_e01r.py": "experiments/RCX-E01R/native_e01r.py",
    "analyze_e01r.py": "experiments/RCX-E01R/analyze_e01r.py",
    "audit_e01r.py": "experiments/RCX-E01R/audit_e01r.py",
    "decide_branch.py": "experiments/RCX-E01R/decide_branch.py",
    "recipes.py": "experiments/RCX-E01R/recipes.py",
    "trace_target.py": "experiments/RCX-E01R/trace_target.py",
    "run_e01r.py": "experiments/RCX-E01R/run_e01r.py",
    "stats.py": "rcx/stats.py",
    "wording_audit.py": "rcx/rcl/wording_audit.py",
    "gen.py": "rcx/rcl/gen.py",
    "vocab.py": "rcx/rcl/vocab.py",
    "templates.py": "rcx/rcl/templates.py",
    "world.py": "rcx/rcl/world.py",
    "score.py": "rcx/rcl/score.py",
    "native_e01.py": "experiments/RCX-E01N/native_e01.py",
    "native_probe.py": "experiments/RCX-E01N/native_probe.py",
    "lora_cpu.py": "rcx/native/lora_cpu.py",
}  # rcx/lane.py is not bound (shared lane infrastructure; no effect on any result)
OUTPUTS = ("ANALYSIS_E01R.md", "ANALYSIS_E01R.json", "ANALYSIS_E01R.started")  # fixed, next to the sealed records


def _ge(x, t):
    return x >= t - EPS


def _gt(x, t):
    return x > t + EPS


def _lt(x, t):
    return x < t - EPS


def _le(x, t):
    return x <= t + EPS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ne_module():
    """experiments/RCX-E01R/native_e01r.py (pure at import: it borrows native_e01/native_probe/lora_cpu helpers from
    their source and imports torch only inside its model commands, which this module never calls)."""
    import native_e01r
    return native_e01r


def split_tag(tag):
    """'E-R2_s1' -> ('E-R2', 1); 'OFF' -> ('OFF', None)."""
    if tag == OFF:
        return OFF, None
    arm, sep, s = tag.rpartition("_s")
    if not sep or not s.isdigit():
        raise ValueError(f"malformed cell tag {tag!r}")
    return arm, int(s)


def tags_of(cells, arm):
    """The present cells of an arm, by seed (OFF: ['OFF'] if present)."""
    if arm == OFF:
        return [OFF] if OFF in cells else []
    return sorted((t for t in cells if t != OFF and split_tag(t)[0] == arm), key=lambda t: split_tag(t)[1])


# ============================================================================ section 5.2: record selection

def _op(r):
    return r["rule_ast"]["op"]


def _dec(r):
    return r["gold"]["decision"]


def _is_f(r):
    return r["group"] in F_STRATA


def _c1_pred(r):
    return _op(r) == "S+" and _dec(r) == "noop"


def _g4_pred(r):
    return _op(r) == "S+" and _dec(r) == "apply"


def _g5_pred(r):
    return _op(r) == "S-" and _dec(r) == "apply"


_C1_ITEMS = [(s, "probe", None) for s in F_STRATA] + [("V1_pos", "unlisted", None),
                                                       ("V5_posted_override", "pos_x_status_posted", None),
                                                       ("V7_bigset", "unlisted", "S+")]
_G4_ITEMS = [(s, "twin_posted", None) for s in F_STRATA] + [("V1_pos", "listed", None), ("V3_inverted", "pos", None),
                                                             ("V5_posted_override", "pos_x_status_x", None),
                                                             ("V7_bigset", "listed", "S+")]
_G5_ITEMS = [("V2_neg", "unlisted", None), ("V5_posted_override", "neg_x_status_posted", None),
             ("V7_bigset", "unlisted", "S-")]
_G6_ITEMS = [("V2_neg", "listed", None), ("V3_inverted", "neg", None),
             ("V5_posted_override", "neg_posted_status_posted", None), ("V7_bigset", "listed", "S-")]
_ONE, _TWO_OR_FOUR, _UP_TO_EIGHT = (1,), (2, 4), tuple(range(1, 9))


def _quantity(surfaces, items, predicate, predicate_text, score, role, per_parent):
    return {"surfaces": surfaces, "items": items, "predicate": predicate, "predicate_text": predicate_text,
            "score": score, "role": role, "per_parent": per_parent}


# items: (stratum, variant or None = every variant, rule op or None); V7's polarity is drawn per parent (S+ / S-).
Q = {
    "C1": _quantity(("p3",), _C1_ITEMS, _c1_pred, "op = S+ and gold = noop", "decision_ok", "confirmatory", _ONE),
    "C2": _quantity(("p3",), [(s, None, None) for s in COMPOSITE], lambda r: r["group"] in COMPOSITE,
                    "stratum in composite", "exact", "confirmatory", _TWO_OR_FOUR),
    "G1": _quantity(("in",), [(s, None, None) for s in COMPOSITE], lambda r: r["group"] in COMPOSITE,
                    "stratum in composite", "exact", "guard", _TWO_OR_FOUR),
    "G2": _quantity(("in",), [(s, "twin_posted", None) for s in F_STRATA],
                    lambda r: _is_f(r) and r["variant"] == "twin_posted", "F stratum and variant = twin_posted",
                    "exact", "guard", _ONE),
    "G3": _quantity(("in", "p3"), [(s, None, None) for s in STRATA], lambda r: True, "all", "format", "guard",
                    _UP_TO_EIGHT),
    "G4": _quantity(("p3",), _G4_ITEMS, _g4_pred, "op = S+ and gold = apply", "decision_ok", "guard", _ONE),
    "G5": _quantity(("p3",), _G5_ITEMS, _g5_pred, "op = S- and gold = apply", "decision_ok", "guard", _ONE),
    "G6": _quantity(("p3",), _G6_ITEMS, lambda r: _op(r) == "S-" and _dec(r) == "noop", "op = S- and gold = noop",
                    "decision_ok", "guard", _ONE),
    "C1-nF": _quantity(("p3",), [x for x in _C1_ITEMS if x[0] not in F_STRATA],
                       lambda r: _c1_pred(r) and not _is_f(r), "C1 without the F strata", "decision_ok",
                       "descriptive", _ONE),
    "G4'": _quantity(("p3",), [x for x in _G4_ITEMS if x[0] not in F_STRATA], lambda r: _g4_pred(r) and not _is_f(r),
                     "G4 without the F strata", "decision_ok", "descriptive", _ONE),
    "G5'": _quantity(("p3",), [x for x in _G5_ITEMS if x[0] != "V5_posted_override"],
                     lambda r: _g5_pred(r) and r["group"] != "V5_posted_override", "G5 without V5", "decision_ok",
                     "descriptive", _ONE),
    "C1-in": _quantity(("in",), _C1_ITEMS, _c1_pred, "op = S+ and gold = noop (surface in)", "decision_ok",
                       "descriptive", _ONE),  # section 5.6: the `in` analogue of C1
}


def _in_items(r, items):
    return any(r["group"] == s and (v is None or r["variant"] == v) and (op is None or _op(r) == op)
               for s, v, op in items)


def check_battery(battery):
    """Structural facts the section 5 quantities rest on: the 10 strata, surfaces in/p3, single status atoms, p3 records
    carrying their form and template id. SystemExit otherwise."""
    groups = {r["group"] for r in battery}
    surfaces = {r["surface"] for r in battery}
    if groups != set(STRATA) or surfaces != set(SURFACES):
        raise SystemExit(f"battery strata {sorted(groups)} / surfaces {sorted(surfaces)} are not PREREG_E01R's")
    if any(_op(r) not in ("S+", "S-") for r in battery):
        raise SystemExit("battery holds a record whose rule is not a single status atom")
    if any(r["surface"] == "p3" and (r.get("p3_form") not in FORMS or not r.get("p3_template")) for r in battery):
        raise SystemExit("a p3 record lacks its P3 form or template id")


def check_selection(battery, qids=tuple(Q)):
    """{quantity: record indices}. Section 5.2: the variant list and the predicate must select the same records, with
    the stated records per parent (one for single-item quantities, 2 or 4 for C2/G1, up to 8 for G3), >= 2 parents."""
    sel = {}
    for qid in qids:
        q = Q[qid]
        by_list = [i for i, r in enumerate(battery) if r["surface"] in q["surfaces"] and _in_items(r, q["items"])]
        by_pred = [i for i, r in enumerate(battery) if r["surface"] in q["surfaces"] and q["predicate"](r)]
        if by_list != by_pred:
            a, b = set(by_list), set(by_pred)
            raise SystemExit(f"{qid}: the variant list and the predicate ({q['predicate_text']}) select different "
                             f"records: {len(a - b)} only by the list, {len(b - a)} only by the predicate "
                             "(PREREG_E01R section 5.2)")
        per = Counter(battery[i]["parent_id"] for i in by_list)
        bad = sorted(set(per.values()) - set(q["per_parent"]))
        if bad:
            raise SystemExit(f"{qid}: parents with {bad} records each (PREREG_E01R section 5.2 allows "
                             f"{list(q['per_parent'])})")
        if len(per) < 2:
            raise SystemExit(f"{qid}: {len(per)} parents (the estimator needs at least 2)")
        sel[qid] = by_list
    return sel


# ============================================================================ record scores (section 5.1)

def score_fn(name):
    if name == "format":
        return lambda c: int(c["intent"] == "format")
    return lambda c: c[name]


def _line1(text):
    """Line 1 as trace_target reads it (first line of the stripped output), parsed: (listed, truth, decision) of its
    single status atom, or None."""
    p = TT.parse_check(TT._split(text)[0])
    if p is None or len(p["atoms"]) != 1 or "listed" not in p["atoms"][0]:
        return None
    a = p["atoms"][0]
    return {"listed": a["listed"], "truth": a["truth"], "decision": p["decision"]}


def score_cell(battery, outs, arm, ne, finished=None):
    """Re-score one cell's raw outputs (pure): 'frozen' = score.classify under the arm's frozen parser
    (native_e01r.answer_parser: E-R3 the last non-empty line, every other arm native_probe.parse_answer on the whole
    output); 'last' = under the last-line parser for every arm (section 5.5, descriptive); disagreement counts of the
    two parsers; for E-R3 the trace_target.predicate_consistency diagnostics and line 1's stated membership."""
    if len(outs) != len(battery):
        raise ValueError(f"{len(outs)} outputs for {len(battery)} records")
    frozen_parse, parser = ne.answer_parser(arm)
    whole, _ = ne.answer_parser(CONTROL)
    last, _ = ne.answer_parser("E-R3")
    frozen, lastc, dis, dis_score = [], [], 0, 0
    for r, o in zip(battery, outs):
        keys = list(r["gold"]["after"])
        pw, pl = whole(o, keys), last(o, keys)
        cf = S.classify(r, frozen_parse(o, keys))
        cl = S.classify(r, pl)
        frozen.append(cf)
        lastc.append(cl)
        dis += pw != pl
        dis_score += S.classify(r, pw) != cl
    cell = {"arm": arm, "out": list(outs), "frozen": frozen, "last": lastc, "parser": parser,
            "finished": list(finished) if finished is not None else [None] * len(outs),
            "parser_disagree": dis, "parser_disagree_score": dis_score}
    if arm == "E-R3":
        cell["trace"] = [TT.predicate_consistency(r, o) for r, o in zip(battery, outs)]
        cell["line1"] = [_line1(o) for o in outs]
    return cell


def subset_cell(cell, keep):
    out = dict(cell)
    for k in ("out", "frozen", "last", "finished", "trace", "line1"):
        if k in cell:
            out[k] = [cell[k][i] for i in keep]
    return out


# ============================================================================ custody (sections 4, 5.1, 10)

class CellProblem(Exception):
    """A cell without a complete eval: absent files or a failed integrity check (section 10, missing reasons 2-4)."""


def bound_paths(ne):
    """{file name: path} for the section 4 list, plus any further file native_e01r binds (its bound_code())."""
    paths = {name: ROOT / rel for name, rel in BOUND_CODE.items()}
    paths["analyze_e01r.py"] = Path(__file__).resolve()  # this file's own sha
    for p in ne.bound_code():
        paths.setdefault(Path(p).name, Path(p))
    return paths


def current_code_sha256(ne=None, names=None):
    """{name: current sha256} of the bound files (names default: the section 4 list plus native_e01r's extras)."""
    ne = ne or ne_module()
    paths = bound_paths(ne)
    return {n: sha(paths[n]) for n in sorted(names or paths) if Path(paths[n]).exists()}


def verify_bound_code(commit, ne):
    """Section 4: the commitment binds every bound file and each one's current sha256 (this file's included) equals the
    committed one. Runs before anything else; SystemExit naming the files otherwise. Returns {name: sha}."""
    rec = commit.get("code_sha256") or {}
    unbound = sorted(set(BOUND_CODE) - set(rec))
    if unbound:
        raise SystemExit(f"SEALED_COMMIT.json does not bind the section 4 bound-code files {unbound}")
    paths = bound_paths(ne)
    unknown = sorted(n for n in rec if n not in paths)
    if unknown:
        raise SystemExit(f"SEALED_COMMIT.json binds files this analysis cannot locate: {unknown}")
    changed = sorted(n for n in rec if not Path(paths[n]).exists() or sha(paths[n]) != rec[n])
    if changed:
        raise SystemExit(f"bound code differs from SEALED_COMMIT.json: {changed} (PREREG_E01R section 4; the analysis "
                         "refuses)")
    return dict(sorted(rec.items()))


def committed_arms(commit):
    """(treatment arms in canonical order, dropped [{arm, reason}], confirmatory tags). Branch B4 only; the treatment
    arms plus any arm dropped at the mint (with a reason) are exactly E-R2, E-R3; arms_required is N-CFSD_s0-s2 plus
    s0-s2 of each treatment arm."""
    branch = (commit.get("branch") or {}).get("branch")
    if branch != BRANCH:
        raise SystemExit(f"SEALED_COMMIT.json records branch {branch!r}; PREREG_E01R's analysis is branch B4 (CX D-39)")
    family = list(commit.get("treatment_arms") or [])
    dropped = list(commit.get("dropped_arms") or [])
    d_arms = [d.get("arm") for d in dropped]
    if (not family or len(set(family)) != len(family) or sorted(family + d_arms) != sorted(BRANCH_TREATMENTS)
            or not all(isinstance(d.get("reason"), str) and d["reason"].strip() for d in dropped)):
        raise SystemExit(f"SEALED_COMMIT.json treatment arms {family} / dropped {d_arms} are not branch B4's "
                         f"{list(BRANCH_TREATMENTS)} (a drop needs its reason; at least one arm remains)")
    family = [x for x in BRANCH_TREATMENTS if x in family]
    want = [f"{a}_s{k}" for a in [CONTROL] + family for k in SEEDS]
    got = list(commit.get("arms_required") or [])
    if len(set(got)) != len(got) or sorted(got) != sorted(want):
        raise SystemExit(f"SEALED_COMMIT.json arms_required {got} != the branch-bound set {want}")
    return family, dropped, want


def code_record_problems(recorded, commit_code, label):
    """A record written under the commitment must carry every committed bound-code sha, each equal to the commitment."""
    if not recorded:
        return [f"{label} records no bound-code shas"]
    absent = sorted(n for n in commit_code if n not in recorded)
    bad = sorted(n for n, v in recorded.items() if n in commit_code and v != commit_code[n])
    return ([f"{label} lacks bound-code shas {absent}"] if absent else []) +         ([f"{label} records bound-code shas that differ from the commitment: {bad}"] if bad else [])


def load_preds(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def check_cell(ne, cells_dir, tag, battery, bsha, bseed, commit, opened, open_sha, warnings):
    """(score_cell result, info) of a complete cell, else CellProblem naming the failed checks (section 4 'Analyse')."""
    arm, seed = split_tag(tag)
    ep, pp = cells_dir / f"eval_sealed_{tag}.json", cells_dir / f"preds_sealed_{tag}.jsonl.gz"
    if not ep.exists() or not pp.exists():
        raise CellProblem("no eval (eval_sealed / preds_sealed file absent)")
    ev = read_json(ep)
    code = commit.get("code_sha256") or {}
    bad = []
    if ev.get("battery") != "sealed" or ev.get("battery_sha256") != bsha:
        bad.append(f"the eval names battery sha {str(ev.get('battery_sha256'))[:12]}..., not the rendered SEALED "
                   f"battery {bsha[:12]}...")
    if ev.get("battery_seed") != bseed or ev.get("n") != len(battery):
        bad.append("battery seed / size differ from the rebuilt SEALED battery")
    if ev.get("sealed_open_sha256") != open_sha:
        bad.append("the eval names another SEALED_OPEN.json")
    if ev.get("arm") != arm or ev.get("seed") != seed or ev.get("tag", tag) != tag:
        bad.append(f"the eval is for {ev.get('arm')} seed {ev.get('seed')}")
    if arm == OFF:
        adapter = None
        if ev.get("adapter_sha256") is not None:
            bad.append("OFF eval names an adapter")
    else:
        want = (opened.get("adapters_sha256") or {}).get(tag)
        ap = (Path(ne.E01N_CELLS) if arm == CONTROL else cells_dir) / tag / "adapter" / "adapter_model.safetensors"
        adapter = sha(ap) if ap.exists() else None
        if adapter is None:
            bad.append("adapter file missing")
        elif adapter != want:
            bad.append(f"adapter sha {adapter[:12]}... != {str(want)[:12]}... recorded by open-sealed")
        if ev.get("adapter_sha256") != want:
            bad.append("the eval names another adapter than open-sealed recorded")
    if ev.get("max_new") != commit.get("max_new"):
        bad.append(f"max_new {ev.get('max_new')} != the committed {commit.get('max_new')}")
    bad += code_record_problems(ev.get("code_sha256"), code, ep.name)
    if arm in BRANCH_TREATMENTS:
        for p in (cells_dir / tag / "TRAIN_META.json", cells_dir / f"SMOKE_{tag}.json"):
            if p.exists():
                bad += code_record_problems(read_json(p).get("code_sha256"), code, f"{tag}/{p.name}"
                                            if p.name == "TRAIN_META.json" else p.name)
            else:
                warnings.append(f"{tag}: {p.name} absent (its bound-code record cannot be checked)")
    if bad:
        raise CellProblem("; ".join(bad))
    rp = ne.raw_path(cells_dir, tag)
    if ev.get("raw_file") != rp.name or not rp.exists():
        raise CellProblem("the raw generations file is missing")
    if sha(rp) != ev.get("raw_sha256"):
        raise CellProblem("the raw generations file differs from the eval's record")
    with gzip.open(rp, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    want_raw = {"tag": tag, "battery_sha256": bsha, "adapter_sha256": ev.get("adapter_sha256"),
                "max_new": commit.get("max_new"), "n": len(battery)}
    off = sorted(k for k, v in want_raw.items() if raw.get(k) != v)
    if off:
        raise CellProblem(f"the raw generations were made for another {off}")
    preds = load_preds(pp)
    if len(preds) != len(battery) or any(p.get("i") != i for i, p in enumerate(preds)):
        raise CellProblem("preds are not aligned with the battery (i = 0..n-1)")
    outs = [p.get("out") for p in preds]
    if outs != raw.get("outs"):
        raise CellProblem("the preds' outputs differ from the raw generations")
    if any(not isinstance(o, str) for o in outs):
        raise CellProblem("an output is not text")
    fin = ne.raw_finished(raw)  # review item 10: the truncation re-derived from the raw file (map_finished)
    stored = [p.get("finished") for p in preds]
    if fin is None:  # review item 11: the decode log cannot be mapped to prompts; truncation unknown
        if any(f is not None for f in stored) or ev.get("truncated") is not None \
                or ev.get("truncation_unknown") is not True:
            raise CellProblem("the raw generations' truncation mapping fails, but the eval or preds record a "
                              "truncation (want truncated null, truncation_unknown true)")
        trunc = None
    else:
        n_fin = sum(a != b for a, b in zip(stored, fin))
        if n_fin:
            raise CellProblem(f"{n_fin} stored 'finished' flags differ from the truncation re-derived from the raw "
                              "generations")
        trunc = fin.count(False)
        if ev.get("truncated") != trunc or ev.get("truncation_unknown", False) is not False:
            raise CellProblem(f"the eval's truncation count {ev.get('truncated')} != {trunc} unfinished outputs "
                              "re-derived from the raw generations")
    sc = score_cell(battery, outs, arm, ne, finished=fin)
    if ev.get("parser") != sc["parser"]:
        raise CellProblem(f"the eval used parser {ev.get('parser')!r}, not the frozen {sc['parser']!r}")
    n_bad = sum({k: p.get(k) for k in SCORE_KEYS} != c for p, c in zip(preds, sc["frozen"]))
    if n_bad:
        raise CellProblem(f"{n_bad} stored scores differ from a re-score of the raw outputs")
    if arm == "E-R3":
        n_tr = sum(p.get("trace") != t for p, t in zip(preds, sc["trace"]))
        if n_tr:
            raise CellProblem(f"{n_tr} stored R3 trace diagnostics differ from a re-computation")
    info = {"eval_sha256": sha(ep), "preds_sha256": sha(pp), "raw_sha256": sha(rp), "adapter_sha256": adapter,
            "adapter_sha256_at_open": (opened.get("adapters_sha256") or {}).get(tag), "parser": sc["parser"],
            "truncated": trunc, "truncation_unknown": trunc is None, "n": len(preds), "wall_s": ev.get("wall_s")}
    if trunc is None:
        warnings.append(f"{tag}: truncation unknown (the raw decode log cannot be mapped to prompts); its "
                        "truncation counts are reported as unknown")
    return sc, info


def r2_duplicate_comp_listed(rows):
    """Section 5.5 item 2: comp_listed rows equal to a base or status_swap row of the same parent (same semantic hash,
    i.e. rule, state and event, and the same system turn, i.e. the same rendered wording)."""
    def key(r):
        return r.get("semantic_hash"), r["prompt"].split("<|im_start|>user", 1)[0]
    ref = defaultdict(set)
    for r in rows:
        if r.get("variant") in ("base", "status_swap"):
            ref[r["parent_id"]].add(key(r))
    return sum(1 for r in rows if r.get("variant") == "comp_listed" and key(r) in ref[r["parent_id"]])


def _tokens(entry):
    per = ((entry or {}).get("tokens") or {}).get("per_seed") or {}
    return {s: {k: v.get(k) for k in ("supervised_tokens", "tokens_processed")} for s, v in sorted(per.items())}


def _counts(balance):
    return {k: v.get("n") for k, v in (balance or {}).items()}


def covariates(cells_dir, commit, family, dropped, warnings):
    """Sections 3 / 5.5 covariates from TRAIN_MANIFEST_E01R.json and AUDIT_E01R.json (each checked against the
    commitment's sha; a differing file is reported, never used for the lead-candidate tie-break) and the R2 duplicate
    comp_listed count. Also {arm: seed-mean supervised tokens} for section 7's tie-break."""
    out, supervised = {}, {}
    arms = list(family) + [d.get("arm") for d in dropped]
    for name, key in ((MANIFEST, "train_manifest_sha256"), (AUDIT, "audit_e01r_sha256")):
        p = cells_dir / name
        if not p.exists():
            warnings.append(f"{name} absent: its covariates are not reported")
            out[name] = None
            continue
        ok = sha(p) == commit.get(key)
        if not ok:
            warnings.append(f"{name} differs from the sha bound by SEALED_COMMIT.json ({key})")
        out[name] = {"sha256": sha(p), "matches_commitment": ok, "data": read_json(p)}
    man = (out[MANIFEST] or {}).get("data") or {}
    aud = (out[AUDIT] or {}).get("data") or {}
    res = {"manifest_matches_commitment": (out[MANIFEST] or {}).get("matches_commitment"),
           "audit_matches_commitment": (out[AUDIT] or {}).get("matches_commitment"),
           "max_new": man.get("max_new"), "r3_gold_longest_tokens": man.get("r3_gold_longest_tokens"), "arms": {}}
    for arm in arms:
        m, a = (man.get("arms") or {}).get(arm) or {}, (aud.get("arms") or {}).get(arm) or {}
        rs = m.get("recipe_stats") or {}
        res["arms"][arm] = {
            "rows": a.get("rows", m.get("rows")), "parents": a.get("parents", m.get("parents")),
            "noop_share": a.get("noop_share"),
            "rows_by_rule_type": a.get("rows_by_rule_type", m.get("rows_by_rule_type")),
            "rows_by_wording_source": _counts(a.get("class_balance_by_wording_source")),
            "rows_by_n_accounts": _counts(a.get("class_balance_by_n_accounts")),
            "tokens_per_seed": _tokens(a) or _tokens(m), "dup_rows_within_parent": rs.get("dup_rows_within_parent")}
    ctl = aud.get("control") or {}
    res["arms"][CONTROL] = {"rows": ctl.get("rows"), "parents": ctl.get("parents"), "noop_share": ctl.get("noop_share"),
                            "rows_by_rule_type": ctl.get("rows_by_rule_type"),
                            "rows_by_wording_source": _counts(ctl.get("class_balance_by_wording_source")),
                            "rows_by_n_accounts": _counts(ctl.get("class_balance_by_n_accounts")),
                            "tokens_per_seed": _tokens(ctl)}
    if (out[AUDIT] or {}).get("matches_commitment"):
        for arm in arms + [CONTROL]:
            per = res["arms"][arm]["tokens_per_seed"]
            vals = [per.get(str(k), {}).get("supervised_tokens") for k in SEEDS]
            if all(isinstance(v, (int, float)) for v in vals):
                supervised[arm] = sum(vals) / len(vals)
    dup = None
    if "E-R2" in arms:
        p = cells_dir / "train_E-R2.jsonl"
        if p.exists() and sha(p) == (commit.get("rows_sha256") or {}).get("E-R2"):
            with open(p, encoding="utf-8") as f:
                dup = r2_duplicate_comp_listed([json.loads(x) for x in f])
        else:
            warnings.append("train_E-R2.jsonl is absent or differs from the committed rows sha: the R2 duplicate "
                            "comp_listed count is not reported")
    res["r2_duplicate_comp_listed"] = dup
    return res, supervised


def smoke_profiles(cells_dir, family, opened, commit, warnings):
    out = {}
    for x in family:
        for k in SEEDS:
            tag = f"{x}_s{k}"
            p = cells_dir / f"SMOKE_{tag}.json"
            if not p.exists():
                out[tag] = None
                continue
            s = read_json(p)
            out[tag] = {key: s.get(key) for key in ("n", "truncated", "truncated_share", "well_formed_share",
                                                    "scorable_share", "exact_share", "wall_s", "pass", "max_new")}
            out[tag]["adapter_matches_open"] = s.get("adapter_sha256") == (opened.get("adapters_sha256") or {}).get(tag)
            if s.get("pass") is not True or not out[tag]["adapter_matches_open"] or s.get("max_new") != commit.get(
                    "max_new"):
                warnings.append(f"SMOKE_{tag}.json does not record a passing smoke test of the opened adapter at the "
                                "committed max_new (section 11: such an opening voids the confirmation)")
    return out


def adapter_mismatch(ne, cells_dir, tag, opened):
    """True iff the cell's adapter file is missing or differs from the sha fixed by open-sealed (section 10 reason 3)."""
    want = ((opened or {}).get("adapters_sha256") or {}).get(tag)
    arm, _ = split_tag(tag)
    ad = (ne.E01N_CELLS if arm == "N-CFSD" else Path(cells_dir)) / tag / "adapter" / "adapter_model.safetensors"
    return want is None or not ad.exists() or sha(ad) != want


def missing_attempts(ne, cells_dir, named, problems, opened=None):
    """Section 10: which cells --missing may name. For each named cell, one of:
    (a) its eval failed the first attempt and the 2 further attempts (native_e01r's append-only ATTEMPTS_E01R.jsonl:
        failed >= 1 + MAX_FURTHER_ATTEMPTS);
    (b) an eval record exists (it is create-only and cannot be re-run) and now fails an integrity check (reason 4),
        whether or not the 'ok' event that should follow it was written;
    (c) its adapter file is missing or differs from the sha fixed at opening (reason 3; no attempt limit).
    Refused otherwise."""
    if not named:
        return {}
    need = 1 + ne.MAX_FURTHER_ATTEMPTS
    events = ne.read_attempts(cells_dir)
    out, short = {}, {}
    for t in named:
        s = dict(ne.attempt_summary(events, "eval", t))
        eval_exists = (Path(cells_dir) / f"eval_sealed_{t}.json").exists()
        s["eval_record_exists"] = eval_exists
        s["adapter_mismatch"] = adapter_mismatch(ne, cells_dir, t, opened) if t != OFF else False
        out[t] = s
        if s["failed"] >= need or (eval_exists and not problems[t].startswith("no eval")) or s["adapter_mismatch"]:
            continue
        short[t] = s
    if short:
        raise SystemExit(f"--missing names cells that do not reach the section 10 attempt limit in {ne.ATTEMPTS} "
                         f"and meet no other section 10 condition: {short}. A cell is "
                         f"missing only after {need} failed eval attempts (the first and {ne.MAX_FURTHER_ATTEMPTS} "
                         "further), after an existing eval record fails an integrity check, or when its adapter no "
                         "longer matches the sha fixed at opening (PREREG_E01R section 10)")
    return out


def load_sealed(cells_dir, missing=(), ne=None):
    """Custody checks 1-6 of the module doc, then the train-overlap drop. Returns (battery, cells, info): battery =
    the rebuilt SEALED records after the drop; cells = {tag: score_cell result} of every complete cell; info = custody,
    missing cells, covariates and smoke profiles (no record, prompt or output)."""
    ne = ne or ne_module()
    cells_dir = Path(cells_dir)
    commit_p, _, open_p = ne.sealed_paths(cells_dir)
    if not commit_p.exists():
        raise SystemExit(f"no SEALED_COMMIT.json in {cells_dir}")
    if not open_p.exists():
        raise SystemExit("the SEALED battery is not open (no SEALED_OPEN.json): there is nothing to analyse")
    commit = read_json(commit_p)
    code = verify_bound_code(commit, ne)  # 1: before anything else
    family, dropped, conf_tags = committed_arms(commit)  # 2
    if list(commit.get("strata") or []) != list(STRATA) or list(commit.get("surfaces") or []) != list(SURFACES):
        raise SystemExit("SEALED_COMMIT.json strata / surfaces are not PREREG_E01R's")
    named = list(missing)
    if len(set(named)) != len(named) or set(named) - set(conf_tags) - {OFF}:
        raise SystemExit(f"--missing {named}: each name must be a committed cell ({conf_tags}) or OFF, once")
    battery, bseed, rec, opened = ne.sealed_battery(cells_dir)  # 3
    if rec != commit:
        raise SystemExit("SEALED_COMMIT.json changed while the analysis read it")
    if opened.get("max_new") != commit.get("max_new"):
        raise SystemExit("SEALED_OPEN.json max_new differs from the commitment")
    unrecorded = [t for t in conf_tags if t not in (opened.get("adapters_sha256") or {})]
    if unrecorded:
        raise SystemExit(f"open-sealed recorded no adapter sha for {unrecorded}")
    bsha, open_sha = ne.NE.battery_sha256(battery), sha(open_p)
    warnings = []
    complete, problems, infos = {}, {}, {}
    for tag in conf_tags + [OFF]:  # 4
        try:
            complete[tag], infos[tag] = check_cell(ne, cells_dir, tag, battery, bsha, bseed, commit, opened, open_sha,
                                                   warnings)
        except CellProblem as e:
            problems[tag] = str(e)
    named_complete = [t for t in named if t in complete]  # 5
    if named_complete:
        raise SystemExit(f"--missing names {named_complete}, which have a complete eval (PREREG_E01R section 4)")
    unnamed = {t: problems[t] for t in conf_tags if t in problems and t not in named}
    if unnamed:
        raise SystemExit(f"confirmatory cells without a complete eval that --missing does not name: {unnamed} "
                         "(PREREG_E01R sections 4 and 10)")
    if OFF in problems and OFF not in named and not problems[OFF].startswith("no eval"):
        raise SystemExit(f"OFF: {problems[OFF]} (name it in --missing if it is declared missing)")
    attempts = missing_attempts(ne, cells_dir, named, problems, opened)  # 5: section 10
    extra = sorted(p.name for p in cells_dir.glob("eval_sealed_*.json")
                   if p.name[len("eval_sealed_"):-len(".json")] not in conf_tags + [OFF])
    if extra:
        warnings.append(f"ignored SEALED eval files of cells outside the committed arms and OFF: {extra}")
    if opened.get("code_changed_since_mint"):
        warnings.append(f"SEALED_OPEN.json records bound code changed between the mint and the opening: "
                        f"{opened['code_changed_since_mint']}")
    n_rebuilt, by_surface = len(battery), dict(Counter(r["surface"] for r in battery))
    dropped_parents = sorted((opened.get("train_overlap") or {}).get("parent_ids") or [])  # 6
    if dropped_parents:
        drop = set(dropped_parents)
        keep = [i for i, r in enumerate(battery) if r["parent_id"] not in drop]
        battery = [battery[i] for i in keep]
        complete = {t: subset_cell(c, keep) for t, c in complete.items()}
    cov, supervised = covariates(cells_dir, commit, family, dropped, warnings)
    missing_info = {t: problems.get(t, "declared missing") for t in named}
    if OFF in problems and OFF not in named:
        missing_info[OFF] = problems[OFF] + " (descriptive cell; affects nothing)"
    info = {"treatment_arms": family, "dropped_arms": dropped, "confirmatory_cells": conf_tags,
            "missing": missing_info, "missing_declared": named, "missing_attempts": attempts,
            "battery_sha256": bsha, "battery_seed": bseed,
            "n_records_rebuilt": n_rebuilt, "by_surface_rebuilt": by_surface,
            "dropped_train_overlap_parents": dropped_parents, "sealed_commit_sha256": sha(commit_p),
            "sealed_open_sha256": open_sha, "sha256_salt": commit.get("sha256_salt"),
            "battery_commit": commit.get("battery_commit"), "minted_at": commit.get("minted_at"),
            "opened_at": opened.get("opened_at"), "max_new": commit.get("max_new"),
            "max_new_covers_r3_gold_byte_bound": opened.get("max_new_covers_r3_gold_byte_bound"),
            "branch": commit.get("branch"),
            "p3": {k: (commit.get("p3") or {}).get(k) for k in ("sha256", "n", "counts")},
            "bound_code_sha256": code, "cells": infos, "covariates": cov, "supervised_tokens_seed_mean": supervised,
            "smoke": smoke_profiles(cells_dir, family, opened, commit, warnings), "warnings": warnings}
    return battery, complete, info


# ============================================================================ sections 5.3 / 6: estimator

def parent_map(battery, idx, cls, score):
    """parent_id -> [score of each of the quantity's records of that parent] in one cell."""
    by = defaultdict(list)
    for i in idx:
        by[battery[i]["parent_id"]].append(score(cls[i]))
    return dict(by)


def contrast(per_seed_a, per_seed_b, resamples=RESAMPLES, seed=BOOT_SEED):
    """Section 6: b - a by one rcx.stats.seed_parent_bootstrap_test call (seed x parent paired cluster bootstrap)."""
    return ST.seed_parent_bootstrap_test(per_seed_a=per_seed_a, per_seed_b=per_seed_b, resamples=resamples, seed=seed,
                                         alpha=ALPHA, p_floor=P_FLOOR, ties=P_TIES)


def estimate(battery, cells, idx, score_name, a_tags, b_tags, scoring="frozen", resamples=RESAMPLES, seed=BOOT_SEED,
             matched=True):
    """Contrast of cells b_tags - a_tags on records idx. matched: a_tags[j] and b_tags[j] share a seed, and the matched
    per-seed contrasts d_k = parent_macro(b_k) - parent_macro(a_k) are added (section 5.3)."""
    f = score_fn(score_name)
    pa = [parent_map(battery, idx, cells[t][scoring], f) for t in a_tags]
    pb = [parent_map(battery, idx, cells[t][scoring], f) for t in b_tags]
    res = dict(contrast(pa, pb, resamples, seed), a=list(a_tags), b=list(b_tags), scoring=scoring,
               records_per_cell=len(idx))
    if matched:
        res["per_seed"] = {f"s{split_tag(tb)[1]}": ST.parent_macro(mb) - ST.parent_macro(ma)
                           for tb, ma, mb in zip(b_tags, pa, pb)}
    return res


def matched_seeds(cells, x):
    have = {CONTROL: set(), x: set()}
    for t in cells:
        a, s = split_tag(t)
        if a in have:
            have[a].add(s)
    return sorted(have[CONTROL] & have[x])


def arm_block(battery, cells, sel, x, qids, scoring="frozen", resamples=RESAMPLES, seed=BOOT_SEED):
    """X - N-CFSD for each quantity on the matched seeds (all three for a complete arm), or None if none match."""
    seeds = matched_seeds(cells, x)
    if not seeds:
        return None
    a = [f"{CONTROL}_s{k}" for k in seeds]
    b = [f"{x}_s{k}" for k in seeds]
    return {"seeds": seeds, "complete": seeds == list(SEEDS),
            "q": {q: estimate(battery, cells, sel[q], Q[q]["score"], a, b, scoring, resamples, seed) for q in qids}}


def holm_family(blocks, family, complete):
    """Section 6: raw p of {C1, C2} x the committed treatment arms (p = 1 for an arm with a missing cell) and the
    rcx.stats.holm adjustment."""
    raw = {}
    for x in family:
        for q in PRIMARY:
            raw[f"{q}_{x}"] = blocks[x]["q"][q]["p"] if complete[x] and blocks.get(x) else 1.0
    return raw, ST.holm(raw)


# ============================================================================ section 7: decision

ADVANCE_KEYS = ("p_H(C1) < 0.05", "d(C1) >= +0.25", "lo(C1) > 0", "d_0(C1) >= +0.10", "d_1(C1) >= +0.10",
                "d_2(C1) >= +0.10", "p_H(C2) < 0.05", "d(C2) > 0", "guards_ok")


def _guard_key(g, bound, rel, m):
    return f"{bound}({g}) {rel} {m:+.2f}"


def decide(conf, guards, complete=True):
    """PREREG_E01R section 7, per treatment arm X, applied mechanically with the section 6 tolerance.
    conf = {"C1": {diff, lo, hi, p_holm, per_seed {"s0", "s1", "s2"}}, "C2": {diff, lo, hi, p_holm}};
    guards = {"G1".."G6": {diff, lo, hi}}; complete = all six cells X_s0-s2 and N-CFSD_s0-s2 present.
    Order: 0 INCONCLUSIVE (missing); 1 ADVANCE; 2 STOP; 3 STOP (degenerate trade-off); 4 REVISE; 5 INCONCLUSIVE.
    Returns the decision, every condition value, the outcomes that hold, the later ones that also hold and the failed
    ADVANCE conditions."""
    if not complete:
        return {"decision": INC_MISSING, "complete": False, "conditions": {}, "outcomes": {}, "also_satisfied": [],
                "failed_advance": []}
    c1, c2 = conf["C1"], conf["C2"]
    ps = c1["per_seed"]
    if sorted(ps) != [f"s{k}" for k in SEEDS]:
        raise ValueError(f"C1 per-seed contrasts must be s0, s1, s2; got {sorted(ps)}")
    cond = {"p_H(C1) < 0.05": _lt(c1["p_holm"], ALPHA), "d(C1) >= +0.25": _ge(c1["diff"], C1_PRACTICAL),
            "lo(C1) > 0": _gt(c1["lo"], 0.0)}
    for k in SEEDS:
        cond[f"d_{k}(C1) >= +0.10"] = _ge(ps[f"s{k}"], C1_PER_SEED)
    cond["p_H(C2) < 0.05"] = _lt(c2["p_holm"], ALPHA)
    cond["d(C2) > 0"] = _gt(c2["diff"], 0.0)
    cond["d(C1) > 0"] = _gt(c1["diff"], 0.0)
    gk = []
    for g, bound, rel, m in GUARD_RULES:
        x = guards[g][bound]
        key = _guard_key(g, bound, rel, m)
        cond[key] = _gt(x, m) if rel == ">" else _le(x, m)
        gk.append(key)
    cond["guards_ok"] = all(cond[k] for k in gk)
    cond["hi(C1) < +0.10"] = _lt(c1["hi"], STOP_C1_HI)
    cond["hi(C2) < +0.05"] = _lt(c2["hi"], STOP_C2_HI)
    for g in DEGENERATE_GUARDS:
        cond[f"d({g}) <= -0.15"] = _le(guards[g]["diff"], DEGENERATE_POINT)
    cond["degenerate"] = any(cond[f"d({g}) <= -0.15"] for g in DEGENERATE_GUARDS)
    cond["sig"] = ((cond["p_H(C1) < 0.05"] and cond["d(C1) > 0"]) or (cond["p_H(C2) < 0.05"] and cond["d(C2) > 0"]))
    outcomes = {"ADVANCE": all(cond[k] for k in ADVANCE_KEYS),
                "STOP": cond["hi(C1) < +0.10"] and cond["hi(C2) < +0.05"],
                STOP_DEG: cond["sig"] and cond["degenerate"],
                "REVISE": cond["sig"] and cond["d(C1) > 0"]}
    decision = next((o for o in ORDER if outcomes[o]), "INCONCLUSIVE")
    later = ORDER[ORDER.index(decision) + 1:] if decision in ORDER else ()
    return {"decision": decision, "complete": True, "conditions": cond, "outcomes": outcomes,
            "also_satisfied": [o for o in later if outcomes[o]],
            "failed_advance": [k for k in ADVANCE_KEYS if not cond[k]]
            + ([k for k in gk if not cond[k]] if not cond["guards_ok"] else [])}


def lead_candidate(advance, c1_diff, supervised=None):
    """Section 7: with more than one ADVANCE arm, the larger d(C1); a tie within the tolerance goes to the arm with
    fewer supervised tokens (seed mean, AUDIT_E01R.json). Returns (arm or None, reason)."""
    if not advance:
        return None, "no arm reaches ADVANCE"
    if len(advance) == 1:
        return advance[0], "the only ADVANCE arm"
    best = max(c1_diff[a] for a in advance)
    top = [a for a in advance if abs(c1_diff[a] - best) <= EPS]
    if len(top) == 1:
        return top[0], "larger d(C1)"
    sup = supervised or {}
    if not all(a in sup for a in top):
        return None, "d(C1) tied within the tolerance; supervised tokens unavailable (AUDIT_E01R.json): unresolved"
    low = min(sup[a] for a in top)
    fewest = [a for a in top if sup[a] == low]
    if len(fewest) == 1:
        return fewest[0], "d(C1) tied within the tolerance; fewer supervised tokens (seed mean)"
    return None, "d(C1) and supervised tokens tied: unresolved"


def decide_experiment(arm_decisions, family, c1_diff=None, supervised=None):
    """Section 7 experiment-level outcome (after opening): ADVANCE if any arm reaches ADVANCE (each named); STOP if
    every arm of the Holm family reaches STOP of either kind; REVISE if any arm reaches REVISE; else INCONCLUSIVE."""
    adv = [x for x in family if arm_decisions[x] == "ADVANCE"]
    if adv:
        lead, why = lead_candidate(adv, c1_diff or {}, supervised)
        return {"decision": "ADVANCE", "arms": adv, "lead_candidate": lead, "lead_reason": why}
    if family and all(arm_decisions[x] in STOPS for x in family):
        return {"decision": "STOP", "arms": list(family), "lead_candidate": None, "lead_reason": None}
    rev = [x for x in family if arm_decisions[x] == "REVISE"]
    if rev:
        return {"decision": "REVISE", "arms": rev, "lead_candidate": None, "lead_reason": None}
    return {"decision": "INCONCLUSIVE", "arms": [], "lead_candidate": None, "lead_reason": None}


def _decision_inputs(block, adj, x):
    conf = {q: dict(block["q"][q], p_holm=adj[f"{q}_{x}"]) for q in PRIMARY}
    return conf, {g: block["q"][g] for g in GUARDS}


def run_decisions(blocks, family, complete):
    raw, adj = holm_family(blocks, family, complete)
    dec = {}
    for x in family:
        if complete[x]:
            conf, guards = _decision_inputs(blocks[x], adj, x)
            dec[x] = decide(conf, guards, True)
        else:
            dec[x] = decide(None, None, False)
    return raw, adj, dec


# ============================================================================ descriptive (sections 5.4-5.6)

def nonanchor_labels(sub):
    """Section 5.4 readings (descriptive): 'degenerate on non-anchor items' if a G4' or G5' point is <= -0.15; 'C1 gain
    beyond the posted rule' yes iff C1-nF's CI lower bound > 0."""
    hits = [g for g in ("G4'", "G5'") if _le(sub["q"][g]["diff"], NONANCHOR_LABEL_POINT)]
    return {"degenerate_on_non_anchor_items": bool(hits), "non_anchor_guards_at_or_below": hits,
            "c1_gain_beyond_posted_rule": "yes" if _gt(sub["q"]["C1-nF"]["lo"], 0.0) else "not shown"}


def rescore_reading(frozen, rescored):
    """Section 5.5: every section 7 condition or outcome that holds under one scoring and fails under the other."""
    if not (frozen["complete"] and rescored["complete"]):
        return []
    diff = [k for k in frozen["conditions"] if frozen["conditions"][k] != rescored["conditions"][k]]
    diff += [f"outcome {o}" for o in frozen["outcomes"] if frozen["outcomes"][o] != rescored["outcomes"][o]]
    return diff


def template_cluster(battery, cells, sel, x, seeds, resamples=RESAMPLES, seed=TEMPLATE_BOOT_SEED):
    """Section 5.6 template-cluster sensitivity for C1 (descriptive): per replicate, for each form (LF, CF, IM) draw
    with replacement as many POS P3 templates of that form as C1 uses, then for each drawn template as many of its C1
    parents as it has; statistic = mean of v_X - v_N-CFSD over all drawn parents."""
    idx = sel["C1"]
    f = score_fn("decision_ok")
    vx, vn = defaultdict(float), defaultdict(float)
    for k in seeds:
        for t, v in ((f"{x}_s{k}", vx), (f"{CONTROL}_s{k}", vn)):
            for p, xs in parent_map(battery, idx, cells[t]["frozen"], f).items():
                v[p] += sum(xs) / len(xs) / len(seeds)
    diff = {p: vx[p] - vn[p] for p in vn}
    groups = defaultdict(lambda: defaultdict(list))
    for i in idx:
        r = battery[i]
        groups[r["p3_form"]][r["p3_template"]].append(r["parent_id"])
    plan = [[sorted(set(groups[fm][t])) for t in sorted(groups[fm])] for fm in FORMS if fm in groups]
    rng, boot = random.Random(seed), []
    for _ in range(resamples):
        tot = cnt = 0
        for tpl in plan:
            for _ in range(len(tpl)):
                ps = tpl[rng.randrange(len(tpl))]
                for _ in range(len(ps)):
                    tot += diff[ps[rng.randrange(len(ps))]]
                    cnt += 1
        boot.append(tot / cnt)
    lo, hi = ST._quantile(boot, ALPHA / 2), ST._quantile(boot, 1 - ALPHA / 2)
    return {"diff": sum(diff.values()) / len(diff), "lo": lo, "hi": hi, "parents": len(diff),
            "templates_per_form": {FORM_SHORT[fm]: len(groups[fm]) for fm in FORMS if fm in groups},
            "resamples": resamples, "seed": seed, "seeds": list(seeds),
            "c1_gain_shown_across_p3_wordings": "yes" if _gt(lo, 0.0) else "no"}


def per_form(battery, cells, sel, arms):
    """C1 and G4-G6 per form: per arm the record counts pooled over its cells (Wilson) and the level; X - N-CFSD."""
    out = {}
    for q in ("C1", "G4", "G5", "G6"):
        f = score_fn(Q[q]["score"])
        out[q] = {}
        for form in FORMS:
            idx = [i for i in sel[q] if battery[i].get("p3_form") == form]
            row = {"records_per_cell": len(idx), "parents": len({battery[i]["parent_id"] for i in idx}), "arms": {},
                   "contrast": {}}
            for arm in arms:
                tags = tags_of(cells, arm)
                if not tags or not idx:
                    continue
                k = sum(f(cells[t]["frozen"][i]) for t in tags for i in idx)
                n = len(idx) * len(tags)
                lo, hi = ST.wilson(k, n)
                lvl = sum(ST.parent_macro(parent_map(battery, idx, cells[t]["frozen"], f)) for t in tags) / len(tags)
                row["arms"][arm] = {"k": k, "n": n, "wilson_lo": lo, "wilson_hi": hi, "level": lvl, "cells": len(tags)}
            if CONTROL in row["arms"]:
                row["contrast"] = {a: row["arms"][a]["level"] - row["arms"][CONTROL]["level"] for a in row["arms"]
                                   if a not in (CONTROL, OFF)}
            out[q][FORM_SHORT[form]] = row
    return out


def per_template(battery, cells, arms):
    """Per P3 template id (never text): polarity, form, parents, records, and per arm the decision counts on its no-op
    and apply items, pooled over the arm's cells."""
    by = defaultdict(list)
    for i, r in enumerate(battery):
        if r["surface"] == "p3":
            by[r["p3_template"]].append(i)
    out = {}
    for t, idx in sorted(by.items()):
        r0 = battery[idx[0]]
        noop = [i for i in idx if _dec(battery[i]) == "noop"]
        app = [i for i in idx if _dec(battery[i]) == "apply"]
        row = {"pol": r0["p3_pol"], "form": FORM_SHORT[r0["p3_form"]],
               "parents": len({battery[i]["parent_id"] for i in idx}), "records": len(idx), "arms": {}}
        for arm in arms:
            tags = tags_of(cells, arm)
            if tags:
                row["arms"][arm] = {"noop_ok": sum(cells[t]["frozen"][i]["decision_ok"] for t in tags for i in noop),
                                    "noop_n": len(noop) * len(tags),
                                    "apply_ok": sum(cells[t]["frozen"][i]["decision_ok"] for t in tags for i in app),
                                    "apply_n": len(app) * len(tags)}
        out[t] = row
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_interval(values):
    xs = list(values)
    if len(xs) - 1 not in T975:
        return None
    m, sd = _mean(xs), _sd(xs)
    half = T975[len(xs) - 1] * sd / math.sqrt(len(xs))
    return {"k": len(xs), "df": len(xs) - 1, "mean": m, "sd": sd, "lo": m - half, "hi": m + half}


def seed_levels(battery, cells, sel, arms, qids):
    out = {}
    for arm in arms:
        tags = tags_of(cells, arm)
        if not tags:
            continue
        out[arm] = {}
        for q in qids:
            f = score_fn(Q[q]["score"])
            vals = {(t if arm == OFF else f"s{split_tag(t)[1]}"):
                    ST.parent_macro(parent_map(battery, sel[q], cells[t]["frozen"], f)) for t in tags}
            xs = list(vals.values())
            out[arm][q] = {"per_seed": vals, "mean": _mean(xs), "sd": _sd(xs), "range": max(xs) - min(xs)}
    return out


def _stratum_values(battery, cl, stratum, surface):
    ex, de = defaultdict(list), defaultdict(list)
    for r, c in zip(battery, cl):
        if r["group"] == stratum and r["surface"] == surface:
            ex[r["parent_id"]].append(c["exact"])
            de[r["parent_id"]].append(c["decision_ok"])
    if not ex:
        return None, None
    return ST.parent_macro(ex), ST.parent_macro(de)


def verdict(battery, cl, surface):
    """PREREG-3-style verdict on the F strata (as E01N's analyze_native.verdict; descriptive)."""
    t = defaultdict(lambda: [0, 0])
    for r, c in zip(battery, cl):
        if r["surface"] == surface and r["group"] in F_STRATA:
            t[(r["group"], r["variant"])][0] += (c["intent"] == "noop") if r["variant"] == "probe" else c["exact"]
            t[(r["group"], r["variant"])][1] += 1
    acc = {k: a / n for k, (a, n) in t.items()}
    unseen = [acc.get((g, "probe")) for g in ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")]
    void, near = acc.get(("F1_void", "probe")), acc.get(("F4_nearmiss", "probe"))
    if None in unseen or void is None:
        return "NA"
    if all(u <= 0.3 for u in unseen) and void <= 0.3:
        return "never no-op (ignores the rule)"
    if all(u >= 0.8 for u in unseen):
        return "generalises to unseen statuses (RULE-like)"
    if all(u <= 0.3 for u in unseen) and void >= 0.8:
        return "TRIGGER" + (" (trigger widened: near-misses)" if near is not None and near >= 0.8 else "")
    return "partial"


def describe_cell(battery, cell):
    cl = cell["frozen"]
    d = {}
    for s in SURFACES:
        strata = {}
        for st in STRATA:
            ex, de = _stratum_values(battery, cl, st, s)
            strata[st] = {"exact": ex, "decision": de}
        errs = Counter(c["err"] for r, c in zip(battery, cl) if r["surface"] == s and c["err"])
        v3 = defaultdict(dict)
        for r, c in zip(battery, cl):
            if r["group"] == "V3_inverted" and r["surface"] == s:
                v3[r["parent_id"]][r["variant"]] = c
        pairs = [v for v in v3.values() if "pos" in v and "neg" in v]
        d[s] = {"strata": strata, "errors": {k: errs.get(k, 0) for k in ERR_KINDS},
                "v3_sensitivity": _mean([int(v["pos"]["intent"] != v["neg"]["intent"]) for v in pairs]),
                "v3_both_exact": _mean([int(v["pos"]["exact"] and v["neg"]["exact"]) for v in pairs]),
                "verdict": verdict(battery, cl, s),
                "truncated": (None if any(fi is None for fi in cell["finished"]) else  # None: truncation unknown
                              sum(1 for r, fi in zip(battery, cell["finished"]) if r["surface"] == s and fi is False)),
                "records": sum(1 for r in battery if r["surface"] == s)}
    d["parser_disagree"] = cell["parser_disagree"]
    d["parser_disagree_score"] = cell["parser_disagree_score"]
    return d


def trace_diagnostics(battery, cell, sel):
    """Section 5.6 R3 line-1 diagnostics of one E-R3 cell (descriptive)."""
    tr, l1 = cell["trace"], cell["line1"]

    def rate(idx, key):
        vals = [tr[i][key] for i in idx if tr[i][key] is not None]
        return {"k": sum(vals), "n": len(vals), "rate": _mean(vals)}

    def listed_ok(idx):
        vals = [int(l1[i]["listed"] == battery[i]["gold"]["listed"]) for i in idx if l1[i] is not None]
        return {"k": sum(vals), "n": len(vals), "rate": _mean(vals)}

    out = {}
    for s in SURFACES:
        idx = [i for i, r in enumerate(battery) if r["surface"] == s]
        out[s] = {k: rate(idx, k) for k in ("well_formed", "stated_decision_ok", "balances_match_stated_decision",
                                            "line1_exact", "line1_coherent")}
        out[s]["listed_ok"] = listed_ok(idx)
    c1 = sel["C1"]
    out["C1_items"] = {"n": len(c1), "stated_decision_ok": rate(c1, "stated_decision_ok"), "listed_ok": listed_ok(c1),
                       "listed_no_only_listed_true": sum(1 for i in c1 if l1[i] and not l1[i]["listed"]
                                                         and l1[i]["truth"]),
                       "stated_apply": sum(1 for i in c1 if tr[i]["stated_decision"] == "apply")}
    return out


def stated_c1(battery, cells, sel, resamples, seed):
    """E-R3's C1 recomputed from line 1's stated decision, against N-CFSD's (balance-scored) C1 (section 6 estimator;
    descriptive)."""
    seeds = matched_seeds(cells, "E-R3")
    if not seeds or any("trace" not in cells[f"E-R3_s{k}"] for k in seeds):
        return None
    idx = sel["C1"]
    f = score_fn("decision_ok")
    pa = [parent_map(battery, idx, cells[f"{CONTROL}_s{k}"]["frozen"], f) for k in seeds]
    pb = []
    for k in seeds:
        tr = cells[f"E-R3_s{k}"]["trace"]
        by = defaultdict(list)
        for i in idx:
            by[battery[i]["parent_id"]].append(tr[i]["stated_decision_ok"])
        pb.append(dict(by))
    return dict(contrast(pa, pb, resamples, seed), seeds=seeds)


# ============================================================================ the whole analysis

def analyze(battery, cells, info, resamples=RESAMPLES, seed=BOOT_SEED, template_seed=TEMPLATE_BOOT_SEED):
    """Sections 5-7 on the custody-checked battery (after the drop) and cells. Returns the result dict (JSON-able; no
    record, prompt, output or P3 text)."""
    check_battery(battery)
    sel = check_selection(battery)
    family = list(info["treatment_arms"])
    complete = {x: matched_seeds(cells, x) == list(SEEDS) for x in family}
    main_q = PRIMARY + GUARDS
    blocks = {x: arm_block(battery, cells, sel, x, main_q, "frozen", resamples, seed) for x in family}
    raw, adj, dec = run_decisions(blocks, family, complete)
    c1_diff = {x: blocks[x]["q"]["C1"]["diff"] for x in family if complete[x]}
    exp = decide_experiment({x: dec[x]["decision"] for x in family}, family, c1_diff,
                            info.get("supervised_tokens_seed_mean"))
    subs = {x: arm_block(battery, cells, sel, x, SUBS, "frozen", resamples, seed) for x in family}
    labels = {x: nonanchor_labels(subs[x]) if subs[x] else None for x in family}
    rblocks = {x: arm_block(battery, cells, sel, x, main_q, "last", resamples, seed) for x in family}
    rraw, radj, rdec = run_decisions(rblocks, family, complete)
    reading = {x: rescore_reading(dec[x], rdec[x]) for x in family}
    arms_all = [CONTROL] + family + ([OFF] if OFF in cells else [])
    h2h = None
    if all(tags_of(cells, x) for x in BRANCH_TREATMENTS) and all(x in family for x in BRANCH_TREATMENTS):
        a, b = tags_of(cells, "E-R3"), tags_of(cells, "E-R2")
        h2h = {q: estimate(battery, cells, sel[q], Q[q]["score"], a, b, "frozen", resamples, seed, matched=False)
               for q in main_q}
    vs_off = {}
    if OFF in cells:
        for arm in [CONTROL] + family:
            if tags_of(cells, arm):
                vs_off[arm] = {q: estimate(battery, cells, sel[q], Q[q]["score"], [OFF], tags_of(cells, arm),
                                           "frozen", resamples, seed, matched=False) for q in PRIMARY}
    in_c1 = {x: arm_block(battery, cells, sel, x, ("C1-in",), "frozen", resamples, seed) for x in family}
    tclust = {x: template_cluster(battery, cells, sel, x, blocks[x]["seeds"], resamples, template_seed)
              for x in family if blocks[x]}
    secondary = {
        "per_form": per_form(battery, cells, sel, arms_all),
        "per_template": per_template(battery, cells, arms_all),
        "template_cluster_c1": tclust,
        "head_to_head_E-R2_minus_E-R3": h2h,
        "vs_OFF": vs_off,
        "seed_levels": seed_levels(battery, cells, sel, arms_all, main_q),
        "matched_seed_contrasts": {x: {q: {"per_seed": blocks[x]["q"][q]["per_seed"],
                                           "t_interval": t_interval(blocks[x]["q"][q]["per_seed"].values())}
                                       for q in main_q} for x in family if blocks[x]},
        "cells": {t: describe_cell(battery, cells[t]) for t in sorted(cells)},
        "c1_in_analogue": in_c1,
        "r3_trace": {t: trace_diagnostics(battery, cells[t], sel) for t in tags_of(cells, "E-R3")
                     if "trace" in cells[t]},
        "r3_stated_decision_c1": stated_c1(battery, cells, sel, resamples, seed) if "E-R3" in family else None,
    }
    selection = {q: {"surfaces": list(Q[q]["surfaces"]), "role": Q[q]["role"], "score": Q[q]["score"],
                     "predicate": Q[q]["predicate_text"], "items": [list(x) for x in Q[q]["items"]],
                     "records_per_cell": len(sel[q]),
                     "parents": len({battery[i]["parent_id"] for i in sel[q]})} for q in Q}
    params = {"resamples": resamples, "prereg_resamples": resamples == PREREG_RESAMPLES, "bootstrap_seed": seed,
              "template_bootstrap_seed": template_seed, "alpha": ALPHA, "p_floor": P_FLOOR,
              "p_floor_effective": 1.0 / resamples, "p_ties": P_TIES, "zero_tol": ST.ZERO_TOL, "eps": EPS,
              "c1_practical": C1_PRACTICAL, "c1_per_seed": C1_PER_SEED, "stop_c1_hi": STOP_C1_HI,
              "stop_c2_hi": STOP_C2_HI, "degenerate_point": DEGENERATE_POINT,
              "guard_rules": [list(x) for x in GUARD_RULES], "decision_order": list(ORDER), "control": CONTROL,
              "treatment_arms": family, "composite": list(COMPOSITE), "strata": list(STRATA)}
    return {"experiment": EXPERIMENT, "analysis": "SEALED confirmation (PREREG_E01R sections 5-7, branch B4)",
            "decision": exp["decision"], "experiment_decision": exp,
            "arm_decisions": {x: dec[x]["decision"] for x in family}, "decision_trace": dec,
            "complete": complete, "holm": {"family": list(raw), "raw_p": raw, "adjusted_p": adj},
            "estimates": blocks, "sub_quantities": subs, "labels": labels,
            "rescore": {"estimates": rblocks, "holm": {"raw_p": rraw, "adjusted_p": radj}, "decision_trace": rdec,
                        "arm_decisions": {x: rdec[x]["decision"] for x in family}, "reading": reading,
                        "parser_disagree": {t: {"records": cells[t]["parser_disagree"],
                                                "scores_changed": cells[t]["parser_disagree_score"]}
                                            for t in sorted(cells)}},
            "selection": selection, "parameters": params, "inputs": dict(info, n_records_analysed=len(battery)),
            "secondary": secondary}


# ============================================================================ report

def _f(x, signed=True):
    if x is None:
        return "—"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _ci(r):
    return f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _yn(b):
    return "yes" if b else "no"


def _trunc(k, n):
    return f"unknown (mapping failed)/{n}" if k is None else f"{k}/{n}"


def _p(x):
    return "—" if x is None else (f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}")


def _arm_label(res, x):
    lab = res["labels"].get(x)
    tag = res["arm_decisions"][x]
    if lab and lab["degenerate_on_non_anchor_items"]:
        tag += ' — labelled "degenerate on non-anchor items" (section 5.4)'
    return tag


def render(res):
    inp, par, sec = res["inputs"], res["parameters"], res["secondary"]
    family = par["treatment_arms"]
    exp = res["experiment_decision"]
    L = [f"# {EXPERIMENT} SEALED confirmation (PREREG_E01R sections 5-7)", "",
         f"**EXPERIMENT DECISION: {exp['decision']}**"
         + (f" (arms: {', '.join(exp['arms'])})" if exp["arms"] else "")
         + " — applied mechanically by `analyze_e01r.decide` / `decide_experiment` (section 7).", ""]
    for x in family:
        L.append(f"- **{x}: {_arm_label(res, x)}**")
    if exp["decision"] == "ADVANCE":
        L.append(f"- Lead candidate: {exp['lead_candidate'] or 'unresolved'} ({exp['lead_reason']}).")
    L.append("")
    if not par["prereg_resamples"]:
        L += [f"**NOT THE PREREGISTERED ANALYSIS: {par['resamples']} resamples (PREREG_E01R fixes "
              f"{PREREG_RESAMPLES}).**", ""]
    if inp["missing"]:
        att = inp.get("missing_attempts") or {}
        L += ["**Missing cells** (declared via --missing, never replaced; section 10): "
              + "; ".join(f"{t}: {why}" + (f" (eval attempts: {att[t]['failed']} failed, {att[t]['ok']} completed)"
                                           if t in att else "") for t, why in inp["missing"].items())
              + ". An arm with a missing confirmatory cell is INCONCLUSIVE (missing); its C1/C2 enter Holm with p = 1.",
              ""]
    L += [f"**Warning:** {w}" for w in inp["warnings"]] + ([""] if inp["warnings"] else [])
    L += ["## Inputs and custody", "",
          f"- SEALED battery: seed {inp['battery_seed']}, sha256 `{inp['battery_sha256']}`, {inp['n_records_rebuilt']} "
          f"records rebuilt in memory; train-overlap parents dropped: {len(inp['dropped_train_overlap_parents'])}; "
          f"records analysed: {inp['n_records_analysed']}.",
          f"- Commitment: sha256(salt) `{inp['sha256_salt']}`; SEALED_COMMIT.json `{inp['sealed_commit_sha256']}` "
          f"(minted {inp['minted_at']}); SEALED_OPEN.json `{inp['sealed_open_sha256']}` (opened {inp['opened_at']}); "
          f"P3 sha256 `{inp['p3']['sha256']}`; max_new {inp['max_new']}.",
          f"- Branch {(inp['branch'] or {}).get('branch')}; treatment arms {', '.join(family)}"
          + ("; dropped at the mint: " + "; ".join(f"{d['arm']} ({d['reason']})" for d in inp["dropped_arms"])
             if inp["dropped_arms"] else "") + ".",
          f"- Bound code verified against the commitment ({len(inp['bound_code_sha256'])} files, this analysis "
          "included): " + ", ".join(f"{k} `{v[:12]}`" for k, v in inp["bound_code_sha256"].items()) + ".",
          "- Cells (each eval names the rendered battery; adapter file = open-sealed record = eval record; committed "
          "max_new; bound-code records match; every score re-derived from the raw output):", "",
          "| cell | eval sha256 | raw sha256 | adapter sha256 | parser | truncated |", "|---|---|---|---|---|---|"]
    for t, c in inp["cells"].items():
        L.append(f"| {t} | `{c['eval_sha256'][:12]}` | `{c['raw_sha256'][:12]}` | "
                 f"{'—' if not c['adapter_sha256'] else '`' + c['adapter_sha256'][:12] + '`'} | {c['parser']} | "
                 f"{_trunc(c['truncated'], c['n'])} |")
    L += ["", f"- Method: seed × parent paired cluster bootstrap (`rcx.stats.seed_parent_bootstrap_test`), "
          f"{par['resamples']} resamples, seed {par['bootstrap_seed']}; two-sided p = 2·min(share ≤ 0, share ≥ 0) "
          f"(replicates within {par['zero_tol']:g} of 0 count on both sides), floored at {par['p_floor_effective']:g}; "
          f"Holm over {', '.join(res['holm']['family'])} at α = {par['alpha']}. CIs are 95% percentile intervals, not "
          f"multiplicity-adjusted. Tolerance ε = {par['eps']:g}.", ""]
    L += ["## Confirmatory (Holm family)", "",
          "| arm | endpoint | N-CFSD | X | d = X − N-CFSD | 95% CI | p | p (Holm) | d_k (s0, s1, s2) | parents |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in family:
        b = res["estimates"][x]
        for q in PRIMARY:
            key = f"{q}_{x}"
            if not b:
                L.append(f"| {x} | {q} | — | — | — | — | — | {_p(res['holm']['adjusted_p'][key])} | — | — |")
                continue
            e = b["q"][q]
            note = "" if res["complete"][x] else " (descriptive: seeds " + ", ".join(map(str, b["seeds"])) + ")"
            L.append(f"| {x}{note} | {q} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} | "
                     f"{_p(e['p'])} | {_p(res['holm']['adjusted_p'][key])} | "
                     f"{', '.join(f'{v:+.3f}' for v in e['per_seed'].values())} | {e['n_parents']} |")
    L += ["", "## Protected strata (ADVANCE guards; CX D-38 margins)", "",
          "| arm | guard | N-CFSD | X | d | 95% CI | rule | within margin | point ≤ −0.15 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in family:
        b, d = res["estimates"][x], res["decision_trace"][x]
        if not b:
            continue
        for g, bound, rel, m in GUARD_RULES:
            e = b["q"][g]
            ok = d["conditions"].get(_guard_key(g, bound, rel, m)) if d["complete"] else None
            deg = d["conditions"].get(f"d({g}) <= -0.15") if d["complete"] and g in DEGENERATE_GUARDS else None
            L.append(f"| {x} | {g} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} | "
                     f"{bound} {rel} {m:+.2f} | {'—' if ok is None else _yn(ok)} | "
                     f"{'—' if deg is None else _yn(deg)} |")
    L += ["", "## Decision trace (section 7; the first outcome that holds decides)", ""]
    for x in family:
        d = res["decision_trace"][x]
        L.append(f"### {x}: {_arm_label(res, x)}")
        L.append("")
        if not d["complete"]:
            L += ["Not computed: a SEALED cell of this arm or of N-CFSD is missing (outcome 0, INCONCLUSIVE "
                  "(missing)).", ""]
            continue
        L += ["| condition | holds |", "|---|---|"] + [f"| {k} | {_yn(v)} |" for k, v in d["conditions"].items()]
        L += ["", "| outcome | holds |", "|---|---|"] + [f"| {o} | {_yn(d['outcomes'][o])} |" for o in ORDER]
        L += ["", f"- Decision: **{d['decision']}**."
              + (f" Later outcomes that also hold: {', '.join(d['also_satisfied'])}." if d["also_satisfied"] else
                 " No later outcome holds."),
              f"- Failed ADVANCE conditions: {', '.join(d['failed_advance']) or 'none'}.", ""]
    L += ["## Experiment-level outcome (section 7)", "",
          "- Per arm: " + "; ".join(f"{x} {_arm_label(res, x)}" for x in family) + ".",
          f"- **{exp['decision']}**" + (f": {', '.join(exp['arms'])}" if exp["arms"] else "") + "."
          + (f" Lead candidate: {exp['lead_candidate'] or 'unresolved'} ({exp['lead_reason']})."
             if exp["decision"] == "ADVANCE" else ""), ""]
    L += ["## Non-anchor sub-quantities (section 5.4; descriptive)", "",
          "| arm | quantity | N-CFSD | X | d | 95% CI | parents |", "|---|---|---|---|---|---|---|"]
    for x in family:
        b = res["sub_quantities"][x]
        if not b:
            continue
        for q in SUBS:
            e = b["q"][q]
            L.append(f"| {x} | {q} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} | "
                     f"{e['n_parents']} |")
    L.append("")
    for x in family:
        lab = res["labels"][x]
        if lab:
            L.append(f"- {x}: degenerate on non-anchor items: **{_yn(lab['degenerate_on_non_anchor_items'])}**"
                     + (f" ({', '.join(lab['non_anchor_guards_at_or_below'])} point ≤ −0.15)"
                        if lab["non_anchor_guards_at_or_below"] else "")
                     + f"; C1 gain beyond the posted rule: **{lab['c1_gain_beyond_posted_rule']}**.")
    rs = res["rescore"]
    L += ["", "## Symmetric last-line re-score (section 5.5; descriptive — the decision uses the frozen scoring)", "",
          "| cell | records where the two parsers disagree | of which the score changes |", "|---|---|---|"]
    L += [f"| {t} | {v['records']} | {v['scores_changed']} |" for t, v in rs["parser_disagree"].items()]
    L += ["", "| arm | quantity | N-CFSD | X | d | 95% CI | p (Holm, re-score family) |",
          "|---|---|---|---|---|---|---|"]
    for x in family:
        b = rs["estimates"][x]
        if not b:
            continue
        for q in PRIMARY + GUARDS:
            e = b["q"][q]
            ph = rs["holm"]["adjusted_p"].get(f"{q}_{x}") if q in PRIMARY else None
            L.append(f"| {x} | {q} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} | "
                     f"{_p(ph)} |")
    L.append("")
    for x in family:
        diff = rs["reading"][x]
        L.append(f"- {x}: re-scored decision {rs['arm_decisions'][x]}; "
                 + (f"**conditions that differ between the two scorings: {', '.join(diff)}**" if diff else
                    "no section 7 condition differs between the two scorings") + ".")
    L += ["", "## Secondary (descriptive; no test)", ""]
    L += ["### C1 and G4–G6 per form (records pooled over seeds, Wilson; level = seed-averaged parent macro)", "",
          "| quantity | form | records/cell | arm | k/n [Wilson] | level | X − N-CFSD |",
          "|---|---|---|---|---|---|---|"]
    for q, forms in sec["per_form"].items():
        for fm, row in forms.items():
            for arm, v in row["arms"].items():
                L.append(f"| {q} | {fm} | {row['records_per_cell']} | {arm} | {v['k']}/{v['n']} "
                         f"[{v['wilson_lo']:.3f}, {v['wilson_hi']:.3f}] | {v['level']:.3f} | "
                         f"{_f(row['contrast'].get(arm))} |")
    arms_all = [CONTROL] + family + ([OFF] if OFF in sec["seed_levels"] else [])
    L += ["", "### Per P3 template (ids and counts only): no-op items decided no-op · apply items decided apply", "",
          "| template | pol | form | parents | " + " | ".join(arms_all) + " |",
          "|---|---|---|---|" + "---|" * len(arms_all)]
    for t, row in sec["per_template"].items():
        cells_ = []
        for arm in arms_all:
            v = row["arms"].get(arm)
            cells_.append("—" if not v else f"{v['noop_ok']}/{v['noop_n']} · {v['apply_ok']}/{v['apply_n']}")
        L.append(f"| {t} | {row['pol']} | {row['form']} | {row['parents']} | " + " | ".join(cells_) + " |")
    L += ["", f"### Template-cluster sensitivity for C1 ({par['resamples']} resamples, seed "
              f"{par['template_bootstrap_seed']})",
          "", "| arm | d | 95% interval | POS templates per form | C1 gain shown across P3 wordings |",
          "|---|---|---|---|---|"]
    for x, v in sec["template_cluster_c1"].items():
        L.append(f"| {x} | {v['diff']:+.3f} | {_ci(v)} | "
                 f"{', '.join(f'{k} {n}' for k, n in v['templates_per_form'].items())} | "
                 f"{v['c1_gain_shown_across_p3_wordings']} |")
    if sec["head_to_head_E-R2_minus_E-R3"]:
        L += ["", "### E-R2 − E-R3 head to head (paired, section 6 estimator; p unadjusted)", "",
              "| quantity | E-R3 | E-R2 | E-R2 − E-R3 | 95% CI | p |", "|---|---|---|---|---|---|"]
        for q, e in sec["head_to_head_E-R2_minus_E-R3"].items():
            L.append(f"| {q} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} | {_p(e['p'])} |")
    if sec["vs_OFF"]:
        L += ["", "### Against OFF (frozen core; section 6 estimator; p unadjusted)", "",
              "| arm | quantity | OFF | arm | arm − OFF | 95% CI |", "|---|---|---|---|---|---|"]
        for arm, qs in sec["vs_OFF"].items():
            for q, e in qs.items():
                L.append(f"| {arm} | {q} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} |")
    L += ["", "### Seeds (parent macro per cell; mean, SD, range)", "",
          "| arm | quantity | per seed | mean | SD | range |", "|---|---|---|---|---|---|"]
    for arm, qs in sec["seed_levels"].items():
        for q, v in qs.items():
            L.append(f"| {arm} | {q} | {', '.join(f'{k} {x:.3f}' for k, x in v['per_seed'].items())} | "
                     f"{v['mean']:.3f} | {_f(v['sd'], False)} | {v['range']:.3f} |")
    L += ["", "### Matched per-seed contrasts X_sk − N-CFSD_sk with a t interval (df = k − 1)", "",
          "| arm | quantity | per seed | mean [t interval] |", "|---|---|---|---|"]
    for x, qs in sec["matched_seed_contrasts"].items():
        for q, v in qs.items():
            ti = v["t_interval"]
            L.append(f"| {x} | {q} | {', '.join(f'{k} {d:+.3f}' for k, d in v['per_seed'].items())} | "
                     + ("—" if not ti else f"{ti['mean']:+.3f} [{ti['lo']:+.3f}, {ti['hi']:+.3f}]") + " |")
    L += ["", "### C1 `in` analogue (op = S+ and gold = noop on the training wording; section 6 estimator)", "",
          "| arm | N-CFSD | X | d | 95% CI |", "|---|---|---|---|---|"]
    for x, b in sec["c1_in_analogue"].items():
        if b:
            e = b["q"]["C1-in"]
            L.append(f"| {x} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | {e['diff']:+.3f} | {_ci(e)} |")
    for s in SURFACES:
        L += ["", f"### Per stratum, exact / decision (per cell), surface `{s}`", "",
              "| cell | " + " | ".join(st.split("_")[0] for st in STRATA)
              + " | verdict | V3 sensitivity | V3 both exact |",
              "|---|" + "---|" * (len(STRATA) + 3)]
        for t, d in sec["cells"].items():
            x = d[s]
            L.append(f"| {t} | " + " | ".join(f"{_f(x['strata'][st]['exact'], False)}/"
                                              f"{_f(x['strata'][st]['decision'], False)}" for st in STRATA)
                     + f" | {x['verdict']} | {_f(x['v3_sensitivity'], False)} | {_f(x['v3_both_exact'], False)} |")
    L += ["", "### Error directions and truncation (counts per cell)", "",
          "| cell | surface | " + " | ".join(ERR_KINDS) + " | truncated |", "|---|---|" + "---|" * (len(ERR_KINDS) + 1)]
    for t, d in sec["cells"].items():
        for s in SURFACES:
            L.append(f"| {t} | {s} | " + " | ".join(str(d[s]["errors"][k]) for k in ERR_KINDS)
                     + f" | {_trunc(d[s]['truncated'], d[s]['records'])} |")
    if sec["r3_trace"]:
        L += ["", "### R3 trace diagnostics (line 1; never scored)", "",
              "| cell | surface | well-formed | stated decision ok | balances = stated decision | line 1 exact | "
              "line 1 coherent | listed= ok |", "|---|---|---|---|---|---|---|---|"]
        for t, d in sec["r3_trace"].items():
            for s in SURFACES:
                x = d[s]
                L.append(f"| {t} | {s} | " + " | ".join(
                    f"{x[k]['k']}/{x[k]['n']}" for k in ("well_formed", "stated_decision_ok",
                                                          "balances_match_stated_decision", "line1_exact",
                                                          "line1_coherent", "listed_ok")) + " |")
        L += ["", "| cell | C1 items | stated decision ok | listed= ok | 'listed=no only-listed:true' | "
              "stated '-> apply' |", "|---|---|---|---|---|---|"]
        for t, d in sec["r3_trace"].items():
            c = d["C1_items"]
            L.append(f"| {t} | {c['n']} | {c['stated_decision_ok']['k']}/{c['stated_decision_ok']['n']} | "
                     f"{c['listed_ok']['k']}/{c['listed_ok']['n']} | {c['listed_no_only_listed_true']} | "
                     f"{c['stated_apply']} |")
        sc = sec["r3_stated_decision_c1"]
        if sc:
            L.append(f"\n- C1 from E-R3's stated decision vs N-CFSD's C1: {sc['mean_b']:.3f} vs {sc['mean_a']:.3f}, "
                     f"d {sc['diff']:+.3f} {_ci(sc)} (descriptive).")
    sm = inp["smoke"]
    if sm:
        L += ["", "### Smoke profiles (TRAIN prompts; section 10)", "",
              "| adapter | truncated | well-formed | scorable | exact | wall s | pass |",
              "|---|---|---|---|---|---|---|"]
        for t, v in sm.items():
            if v is None:
                L.append(f"| {t} | — | — | — | — | — | — |")
            else:
                L.append(f"| {t} | {v['truncated']}/{v['n']} | {_f(v['well_formed_share'], False)} | "
                         f"{_f(v['scorable_share'], False)} | {_f(v['exact_share'], False)} | {v['wall_s']} | "
                         f"{v['pass']} |")
    cov = inp["covariates"]
    L += ["", "### Covariates (sections 3 and 5.5; nothing is matched on them)", "",
          f"- TRAIN_MANIFEST_E01R.json matches the commitment: {cov['manifest_matches_commitment']}; AUDIT_E01R.json: "
          f"{cov['audit_matches_commitment']}; max_new {cov['max_new']}; R3 gold longest tokens "
          f"{cov['r3_gold_longest_tokens']}.",
          f"- R2 duplicate comp_listed twins (size-2 sets, not redrawn by the frozen spec): "
          f"{'—' if cov['r2_duplicate_comp_listed'] is None else cov['r2_duplicate_comp_listed']}.", "",
          "| arm | rows | parents | no-op share | rows by rule type | supervised tokens per seed | processed tokens "
          "per seed | dup rows within parent |", "|---|---|---|---|---|---|---|---|"]
    for arm, v in cov["arms"].items():
        tok = v["tokens_per_seed"] or {}
        L.append(f"| {arm} | {v['rows']} | {v['parents']} | {v['noop_share']} | {v['rows_by_rule_type']} | "
                 f"{', '.join(str(t.get('supervised_tokens')) for t in tok.values()) or '—'} | "
                 f"{', '.join(str(t.get('tokens_processed')) for t in tok.values()) or '—'} | "
                 f"{'—' if v.get('dup_rows_within_parent') is None else v['dup_rows_within_parent']} |")
    L += ["", "| arm | rows by wording source | rows by account count |", "|---|---|---|"]
    L += [f"| {arm} | {v.get('rows_by_wording_source') or '—'} | {v.get('rows_by_n_accounts') or '—'} |"
          for arm, v in cov["arms"].items()]
    L += ["", "### Record selection (section 5.2; variant list = predicate, checked)", "",
          "| quantity | role | surface | predicate | score | records per cell | parents |",
          "|---|---|---|---|---|---|---|"]
    for q, v in res["selection"].items():
        L.append(f"| {q} | {v['role']} | {', '.join(v['surfaces'])} | {v['predicate']} | {v['score']} | "
                 f"{v['records_per_cell']} | {v['parents']} |")
    return "\n".join(L) + "\n"


# ============================================================================ CLI

def _write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:  # create-only: the analysis runs once
        f.write(text)


def output_paths(cells):
    """(report .md, result .json, started marker): fixed paths next to the sealed records (review item 5)."""
    return tuple(Path(cells) / name for name in OUTPUTS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RCX-E01R SEALED analysis (PREREG_E01R sections 5-7; run once)")
    ap.add_argument("--cells", default=str(ROOT / "runs" / "RCX-E01R" / "cells"))
    ap.add_argument("--missing", default="", help="comma-separated cells declared missing in DECISIONS (section 10)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cells = Path(a.cells)
    out, jout, marker = output_paths(cells)
    existing = [p.name for p in (marker, out, jout) if p.exists()]
    if existing:
        raise SystemExit(f"refusing: {existing} exist in {cells}: the SEALED analysis runs once (PREREG_E01R section "
                         "10; a started marker without outputs is a crashed analysis, a logged deviation)")
    missing = [t.strip() for t in a.missing.split(",") if t.strip()]
    battery, cells_d, info = load_sealed(cells, missing)  # custody: refusals only, no statistic, nothing written
    _write_new(marker, json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "pid": os.getpid(),
                                   "argv": list(sys.argv[1:] if argv is None else argv), "missing": missing,
                                   "sealed_open_sha256": info["sealed_open_sha256"],
                                   "analyze_e01r_sha256": sha(Path(__file__).resolve())}, indent=1) + "\n")
    res = analyze(battery, cells_d, info, resamples=RESAMPLES, seed=BOOT_SEED, template_seed=TEMPLATE_BOOT_SEED)
    txt = render(res)
    _write_new(jout, json.dumps(res, indent=1) + "\n")
    _write_new(out, txt)
    print(txt)
    print(f"DECISION: {res['decision']}  ({'; '.join(f'{x} {_arm_label(res, x)}' for x in res['arm_decisions'])}; "
          f"written {out} and {jout})")
    return res


if __name__ == "__main__":
    main()
