"""Tests for experiments/RCX-E01R/analyze_e01r.py (PREREG_E01R sections 5-7: the SEALED analysis). Pure Python: no
torch, no tokenizer, no model. The SEALED battery is the real P3-free skeleton (native_e01r.build_skeleton, a few
parents per stratum) rendered with fake P3 templates; every prediction is synthetic (scripted per record class and
seed); the commitment, opening record, adapters, evals, raw generations and covariate files are fakes in a temp dir,
written in native_e01r's formats (preds rows by native_e01r.score_outputs). A read of a *.secret file outside the temp
dir fails the test before a byte is read. Bootstraps use few resamples (the prereg's 10,000 are checked as defaults).

Run: python -B -m unittest tests.test_analyze_e01r -v   (CPU, single thread)
"""
import ast
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import unittest
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01R"))
HEAVY = ("torch", "transformers", "peft", "native_probe", "native_e01", "rcx.native.lora_cpu")
_HEAVY_BEFORE = {m for m in HEAVY if m in sys.modules}

import analyze_e01r as A  # noqa: E402
import native_e01r as M  # noqa: E402
import recipes as RC  # noqa: E402
import trace_target as TT  # noqa: E402
from rcx import stats as ST  # noqa: E402
from rcx.rcl import world as W  # noqa: E402

_HEAVY_AFTER = {m for m in HEAVY if m in sys.modules}
_REAL_OPEN = io.open
COUNTS = {"POS/EXPLICIT_LISTED_FIRST": 7, "POS/EXPLICIT_COMPLEMENT_FIRST": 10, "POS/IMPLICIT": 6,
          "NEG/EXPLICIT_LISTED_FIRST": 7, "NEG/EXPLICIT_COMPLEMENT_FIRST": 9, "NEG/IMPLICIT": 9}  # POOL_REFINE P3
ARMS = ("N-CFSD", "E-R2", "E-R3")
R = 400  # bootstrap resamples in the tests (floor 1/400; Holm x 4 still < 0.05)
MAX_NEW = 170
SALT = "e01r-analysis-test-salt"
DEFAULT_ACC = 0.97


def write(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return A.sha(p)


def fake_p3_templates():
    out, k = [], 0
    for cell, n in COUNTS.items():
        pol, form = cell.split("/")
        for i in range(n):
            out.append({"id": f"q{k:03d}", "text": f"Fake {pol} {form.lower()} rule {i}: {{L}} decides it.",
                        "pol": pol, "form": form, "author": "test"})
            k += 1
    random.Random(7).shuffle(out)
    return out


def ids_by_cell():
    by = defaultdict(list)
    for t in fake_p3_templates():
        by[f"{t['pol']}/{t['form']}"].append(t["id"])
    return {c: sorted(v) for c, v in by.items()}


_SKELETONS = {}


def synth_battery(n_parents=6, seed=4242):
    """The real skeleton (in + p3 copies) with each p3 record's drawn template id set; no template is registered."""
    if (n_parents, seed) not in _SKELETONS:
        _SKELETONS[(n_parents, seed)] = M.build_skeleton(seed, COUNTS, n_parents)
    ids, out = ids_by_cell(), []
    for q in copy.deepcopy(_SKELETONS[(n_parents, seed)]):
        if q["surface"] == "p3":
            tid = ids[f"{q['p3_pol']}/{q['p3_form']}"][q["p3_j"]]
            q["tid"], q["p3_template"] = f"p3:{tid}", tid
        out.append(q)
    return out


def rclass(r):
    if r["surface"] == "in":
        return "in"
    return {("S+", "noop"): "C1", ("S+", "apply"): "G4", ("S-", "apply"): "G5",
            ("S-", "noop"): "G6"}[(r["rule_ast"]["op"], r["gold"]["decision"])]


def native_out(r, decision):
    if decision == "format":
        return "no answer"
    if decision == r["gold"]["decision"]:
        after = r["gold"]["after"]
    else:
        after = dict(r["state"]) if decision == "noop" else W.applied(r["state"], r["event"])
    return json.dumps({k: after[k] for k in sorted(after)}, separators=(",", ":"))


def r3_out(r, decision):
    if decision == "format":
        return "check nothing"
    line1 = TT.check_line(r["rule_ast"], r["state"], r["event"])
    if decision != r["gold"]["decision"]:
        line1 = line1.rsplit(" -> ", 1)[0] + " -> " + decision
    return line1 + "\n" + native_out(r, decision)


def outputs(battery, arm, seed, accs, preamble=False):
    """Per record: the gold decision if u(seed, record) < the arm's accuracy for the record's class, else the other
    decision. u is shared by all arms, so arms with equal accuracy on a class give identical outcomes there."""
    acc = accs.get(arm, {})
    outs = []
    for i, r in enumerate(battery):
        a = acc.get(rclass(r), DEFAULT_ACC)
        if a == "format":
            dec = "format"
        else:
            u = random.Random(f"u-{seed}-{i}").random()
            dec = r["gold"]["decision"] if u < a else ("noop" if r["gold"]["decision"] == "apply" else "apply")
        o = (r3_out if arm == "E-R3" else native_out)(r, dec)
        outs.append("Answer:\n" + o if preamble else o)
    return outs


def make_cells(battery, accs, arms=ARMS, seeds=(0, 1, 2), off=False, preamble=()):
    cells = {}
    for arm in arms:
        for k in seeds:
            outs = outputs(battery, arm, k, accs, preamble=arm in preamble)
            cells[f"{arm}_s{k}"] = A.score_cell(battery, outs, arm, M, finished=[True] * len(outs))
    if off:
        outs = outputs(battery, "OFF", 0, {"OFF": {c: 0.5 for c in ("in", "C1", "G4", "G5", "G6")}})
        cells["OFF"] = A.score_cell(battery, outs, "OFF", M, finished=[True] * len(outs))
    return cells


def base_info(family=("E-R2", "E-R3"), dropped=(), supervised=None):
    """The custody/info keys load_sealed returns (values for an in-memory analysis)."""
    return {"treatment_arms": list(family), "dropped_arms": list(dropped), "missing": {}, "missing_declared": [],
            "warnings": [], "battery_seed": 4242, "battery_sha256": "0" * 64, "n_records_rebuilt": 0,
            "by_surface_rebuilt": {}, "dropped_train_overlap_parents": [], "sha256_salt": "1" * 64,
            "sealed_commit_sha256": "2" * 64, "sealed_open_sha256": "3" * 64, "minted_at": "t0", "opened_at": "t1",
            "p3": {"sha256": "4" * 64}, "max_new": MAX_NEW, "branch": {"branch": "B4"}, "bound_code_sha256": {},
            "cells": {}, "smoke": {}, "supervised_tokens_seed_mean": supervised or {},
            "covariates": {"manifest_matches_commitment": None, "audit_matches_commitment": None, "max_new": None,
                           "r3_gold_longest_tokens": None, "r2_duplicate_comp_listed": None, "arms": {}}}


def run(accs, family=("E-R2", "E-R3"), drop_cells=(), off=False, preamble=(), resamples=R, n_parents=6):
    bat = synth_battery(n_parents)
    cells = make_cells(bat, accs, arms=("N-CFSD",) + tuple(family), off=off, preamble=preamble)
    for t in drop_cells:
        del cells[t]
    res = A.analyze(bat, cells, base_info(family), resamples=resamples, seed=A.BOOT_SEED,
                    template_seed=A.TEMPLATE_BOOT_SEED)
    json.dumps(res)  # JSON-able
    A.render(res)  # renders
    return res


# ----------------------------------------------------------------------------- decide(): the section 7 truth table

def mk(c1=(0.30, 0.20, 0.40), c1_p=0.001, per_seed=(0.30, 0.30, 0.30), c2=(0.10, 0.05, 0.15), c2_p=0.001,
       guards=None):
    conf = {"C1": {"diff": c1[0], "lo": c1[1], "hi": c1[2], "p_holm": c1_p,
                   "per_seed": {f"s{k}": v for k, v in enumerate(per_seed)}},
            "C2": {"diff": c2[0], "lo": c2[1], "hi": c2[2], "p_holm": c2_p}}
    g = {x: {"diff": 0.0, "lo": -0.01, "hi": 0.01} for x in A.GUARDS}
    for k, v in (guards or {}).items():
        g[k] = dict(g[k], **v)
    return conf, g


def decision(**kw):
    conf, g = mk(**kw)
    return A.decide(conf, g)


class TestModule(unittest.TestCase):
    def test_no_heavy_imports(self):
        self.assertEqual(_HEAVY_AFTER - _HEAVY_BEFORE, set())
        src = (ROOT / "experiments" / "RCX-E01R" / "analyze_e01r.py").read_text(encoding="utf-8")
        mods = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
        mods |= {(n.module or "").split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(mods & {"torch", "transformers", "peft", "numpy"})

    def test_prereg_constants(self):
        self.assertEqual((A.PREREG_RESAMPLES, A.RESAMPLES, A.BOOT_SEED, A.TEMPLATE_BOOT_SEED, A.ALPHA, A.P_FLOOR,
                          A.P_TIES, A.EPS), (10000, 10000, 20261265, 20261268, 0.05, None, "both", 1e-9))
        self.assertEqual(A.GUARD_RULES, (("G1", "lo", ">", -0.03), ("G2", "lo", ">", -0.05), ("G3", "hi", "<=", 0.02),
                                         ("G4", "lo", ">", -0.05), ("G5", "lo", ">", -0.10),
                                         ("G6", "lo", ">", -0.08)))  # CX D-38
        self.assertEqual((A.C1_PRACTICAL, A.C1_PER_SEED, A.STOP_C1_HI, A.STOP_C2_HI, A.DEGENERATE_POINT),
                         (0.25, 0.10, 0.10, 0.05, -0.15))
        self.assertEqual(A.STRATA, M.STRATA)
        self.assertEqual(A.FORMS, M.FORMS)
        self.assertEqual(set(A.BOUND_CODE), {
            "native_e01r.py", "analyze_e01r.py", "audit_e01r.py", "decide_branch.py", "recipes.py", "trace_target.py",
            "run_e01r.py", "stats.py", "wording_audit.py", "gen.py", "vocab.py", "templates.py", "world.py",
            "score.py", "native_e01.py", "native_probe.py", "lora_cpu.py", "lane.py"})  # section 4
        for name, rel in A.BOUND_CODE.items():
            self.assertTrue((ROOT / rel).exists() or name == "analyze_e01r.py", name)


class TestDecide(unittest.TestCase):
    def test_advance(self):
        d = decision()
        self.assertEqual(d["decision"], "ADVANCE")
        self.assertTrue(all(d["conditions"][k] for k in A.ADVANCE_KEYS))
        self.assertEqual(d["failed_advance"], [])
        self.assertEqual(d["also_satisfied"], ["REVISE"])  # sig and d(C1) > 0 hold too

    def test_ordered_rule(self):
        # (overrides, expected decision, expected later outcomes that also hold)
        table = [
            ({}, "ADVANCE", ["REVISE"]),
            ({"c1": (0.24, 0.15, 0.33)}, "REVISE", []),                                   # below +0.25
            ({"per_seed": (0.30, 0.09, 0.30)}, "REVISE", []),                             # one seed below +0.10
            ({"c2_p": 0.06}, "REVISE", []),                                               # C2 not Holm-significant
            ({"c2": (-0.01, -0.03, 0.01), "c2_p": 0.5}, "REVISE", []),                    # C2 not positive
            ({"guards": {"G4": {"lo": -0.06}}}, "REVISE", []),                             # guard outside margin
            ({"guards": {"G3": {"hi": 0.03}}}, "REVISE", []),
            ({"guards": {"G5": {"diff": -0.16, "lo": -0.25}}}, A.STOP_DEG, ["REVISE"]),   # degenerate point
            ({"guards": {"G6": {"diff": -0.15, "lo": -0.20}}}, A.STOP_DEG, ["REVISE"]),   # exactly -0.15
            ({"guards": {"G6": {"diff": -0.149, "lo": -0.20}}}, "REVISE", []),            # point above -0.15
            ({"c1": (0.05, 0.01, 0.09), "c2": (0.01, 0.005, 0.04)}, "STOP", ["REVISE"]),  # STOP precedes REVISE
            ({"c1": (0.05, 0.01, 0.09), "c2": (0.01, 0.005, 0.04), "guards": {"G4": {"diff": -0.2, "lo": -0.3}}},
             "STOP", [A.STOP_DEG, "REVISE"]),
            ({"c1": (0.0, -0.05, 0.05), "c1_p": 1.0, "c2": (0.0, -0.02, 0.02), "c2_p": 1.0}, "STOP", []),
            ({"c1": (0.0, -0.02, 0.12), "c1_p": 1.0, "c2": (0.08, 0.04, 0.12)}, "INCONCLUSIVE", []),  # C2 only
            ({"c1": (-0.02, -0.06, 0.12), "c1_p": 0.5, "c2": (0.08, 0.04, 0.12),
              "guards": {"G5": {"diff": -0.3, "lo": -0.4}}}, A.STOP_DEG, []),  # sig via C2 + degenerate
            ({"c1": (0.08, -0.01, 0.17), "c1_p": 0.2, "c2": (0.02, -0.01, 0.06), "c2_p": 0.3}, "INCONCLUSIVE", []),
        ]
        for over, want, later in table:
            d = decision(**over)
            self.assertEqual((d["decision"], d["also_satisfied"]), (want, later), over)

    def test_failed_advance_named(self):
        d = decision(c1=(0.20, 0.12, 0.28), per_seed=(0.3, 0.05, 0.3), guards={"G1": {"lo": -0.04}})
        self.assertEqual(d["decision"], "REVISE")
        self.assertEqual(d["failed_advance"], ["d(C1) >= +0.25", "d_1(C1) >= +0.10", "guards_ok", "lo(G1) > -0.03"])

    def test_tolerance(self):
        e = A.EPS
        cases = [({"c1": (0.25 - e / 2, 0.2, 0.3)}, "d(C1) >= +0.25", True),
                 ({"c1": (0.25 - 2 * e, 0.2, 0.3)}, "d(C1) >= +0.25", False),
                 ({"c1_p": 0.05 - e / 2}, "p_H(C1) < 0.05", False),
                 ({"c1_p": 0.05 - 2 * e}, "p_H(C1) < 0.05", True),
                 ({"c1": (0.3, e / 2, 0.4)}, "lo(C1) > 0", False),
                 ({"per_seed": (0.10 - e / 2, 0.3, 0.3)}, "d_0(C1) >= +0.10", True),
                 ({"guards": {"G1": {"lo": -0.03 + e / 2}}}, "lo(G1) > -0.03", False),
                 ({"guards": {"G1": {"lo": -0.03 + 2 * e}}}, "lo(G1) > -0.03", True),
                 ({"guards": {"G3": {"hi": 0.02 + e / 2}}}, "hi(G3) <= +0.02", True),
                 ({"guards": {"G3": {"hi": 0.02 + 2 * e}}}, "hi(G3) <= +0.02", False),
                 ({"guards": {"G4": {"diff": -0.15 + e / 2}}}, "d(G4) <= -0.15", True),
                 ({"guards": {"G4": {"diff": -0.15 + 2 * e}}}, "d(G4) <= -0.15", False),
                 ({"c1": (0.3, 0.2, 0.10 - e / 2)}, "hi(C1) < +0.10", False),
                 ({"c2": (0.0, -0.01, 0.05 - 2 * e)}, "hi(C2) < +0.05", True)]
        for over, key, want in cases:
            self.assertEqual(decision(**over)["conditions"][key], want, (over, key))

    def test_missing_and_inputs(self):
        d = A.decide(None, None, complete=False)
        self.assertEqual((d["decision"], d["conditions"], d["also_satisfied"]), (A.INC_MISSING, {}, []))
        conf, g = mk()
        del conf["C1"]["per_seed"]["s2"]
        with self.assertRaises(ValueError):
            A.decide(conf, g)

    def test_experiment_level(self):
        fam = ["E-R2", "E-R3"]
        cases = [({"E-R2": "ADVANCE", "E-R3": "STOP"}, "ADVANCE", ["E-R2"]),
                 ({"E-R2": "REVISE", "E-R3": "ADVANCE"}, "ADVANCE", ["E-R3"]),
                 ({"E-R2": "STOP", "E-R3": A.STOP_DEG}, "STOP", fam),
                 ({"E-R2": A.STOP_DEG, "E-R3": A.STOP_DEG}, "STOP", fam),
                 ({"E-R2": "REVISE", "E-R3": "STOP"}, "REVISE", ["E-R2"]),
                 ({"E-R2": "REVISE", "E-R3": A.INC_MISSING}, "REVISE", ["E-R2"]),
                 ({"E-R2": "INCONCLUSIVE", "E-R3": "STOP"}, "INCONCLUSIVE", []),
                 ({"E-R2": A.INC_MISSING, "E-R3": "STOP"}, "INCONCLUSIVE", []),
                 ({"E-R2": A.INC_MISSING, "E-R3": A.INC_MISSING}, "INCONCLUSIVE", [])]
        for arms, want, named in cases:
            e = A.decide_experiment(arms, fam, {"E-R2": 0.3, "E-R3": 0.3})
            self.assertEqual((e["decision"], e["arms"]), (want, named), arms)
        self.assertEqual(A.decide_experiment({"E-R3": "STOP"}, ["E-R3"])["decision"], "STOP")  # one arm left

    def test_lead_candidate(self):
        both = {"E-R2": "ADVANCE", "E-R3": "ADVANCE"}
        e = A.decide_experiment(both, ["E-R2", "E-R3"], {"E-R2": 0.31, "E-R3": 0.36})
        self.assertEqual((e["arms"], e["lead_candidate"], e["lead_reason"]), (["E-R2", "E-R3"], "E-R3", "larger d(C1)"))
        tie = {"E-R2": 0.30, "E-R3": 0.30 + A.EPS / 2}
        e = A.decide_experiment(both, ["E-R2", "E-R3"], tie, {"E-R2": 61000.0, "E-R3": 58000.0})
        self.assertEqual(e["lead_candidate"], "E-R3")
        self.assertIn("fewer supervised tokens", e["lead_reason"])
        e = A.decide_experiment(both, ["E-R2", "E-R3"], tie, {})
        self.assertIsNone(e["lead_candidate"])
        self.assertIn("unresolved", e["lead_reason"])
        e = A.decide_experiment(both, ["E-R2", "E-R3"], {"E-R2": 0.30, "E-R3": 0.30 + 2 * A.EPS},
                                {"E-R2": 1.0, "E-R3": 9.0})
        self.assertEqual(e["lead_candidate"], "E-R3")  # outside the tolerance: d(C1) decides


# ----------------------------------------------------------------------------- statistics: section 6

class TestStatistics(unittest.TestCase):
    def test_contrast_is_the_prereg_call(self):
        pa, pb = [{"a": [0], "b": [1]}], [{"a": [1], "b": [1]}]
        with mock.patch.object(A.ST, "seed_parent_bootstrap_test", return_value={"ok": 1}) as m:
            A.contrast(pa, pb)
        m.assert_called_once_with(per_seed_a=pa, per_seed_b=pb, resamples=10000, seed=20261265, alpha=0.05,
                                  p_floor=None, ties="both")

    def test_ties_counted_on_both_sides(self):
        pa, pb = [{"a": [0], "b": [1]}], [{"a": [1], "b": [0]}]  # parent contrasts +1 and -1
        r = A.contrast(pa, pb, resamples=1000)
        self.assertEqual(r["ties"], "both")
        self.assertGreater(r["n_eq0"], 300)  # about half the replicates are exactly 0
        self.assertEqual(r["p"], min(1.0, 2 * min(r["n_le0"], r["n_ge0"]) / 1000))
        self.assertEqual(r["p"], 1.0)
        low = ST.seed_parent_bootstrap_test(pa, pb, resamples=1000, seed=A.BOOT_SEED, ties="low")
        self.assertLess(low["p"], 0.75)  # ties on one side only would call it far more significant
        same = A.contrast([{"a": [1], "b": [0]}], [{"a": [1], "b": [0]}], resamples=500)
        self.assertEqual((same["p"], same["n_eq0"]), (1.0, 500))  # identical arms: p = 1, never the floor
        up = A.contrast([{"a": [0], "b": [0], "c": [1]}], [{"a": [1], "b": [1], "c": [2]}], resamples=500)
        self.assertEqual(up["p"], 1 / 500)  # every replicate > 0: the floor 1/resamples

    def test_interval_equals_seed_parent_bootstrap(self):
        rng = random.Random(3)
        pa = [{f"p{i}": [rng.random() < 0.4, rng.random() < 0.5] for i in range(30)} for _ in range(3)]
        pb = [{f"p{i}": [rng.random() < 0.6] for i in range(30)} for _ in range(3)]
        pa = [{k: [float(x) for x in v] for k, v in a.items()} for a in pa]
        pb = [{k: [float(x) for x in v] for k, v in b.items()} for b in pb]
        r = A.contrast(pa, pb, resamples=500)
        ref = ST.seed_parent_bootstrap(pa, pb, resamples=500, seed=A.BOOT_SEED, resample_seeds=False)
        self.assertEqual((r["diff"], r["lo"], r["hi"]), (ref["diff"], ref["lo"], ref["hi"]))
        v = {p: (sum(sum(b[p]) / len(b[p]) for b in pb) - sum(sum(a[p]) / len(a[p]) for a in pa)) / 3 for p in pa[0]}
        self.assertAlmostEqual(r["diff"], sum(v.values()) / len(v), places=12)  # section 5.3: seeds averaged

    def test_holm_family(self):
        fam = ["E-R2", "E-R3"]

        def blocks(p):
            return {x: {"q": {"C1": {"p": p[f"C1_{x}"]}, "C2": {"p": p[f"C2_{x}"]}}} for x in fam}
        p = {"C1_E-R2": 0.01, "C2_E-R2": 0.04, "C1_E-R3": 0.03, "C2_E-R3": 0.005}
        raw, adj = A.holm_family(blocks(p), fam, {"E-R2": True, "E-R3": True})
        self.assertEqual(list(raw), ["C1_E-R2", "C2_E-R2", "C1_E-R3", "C2_E-R3"])
        want = {"C1_E-R2": 0.03, "C2_E-R2": 0.06, "C1_E-R3": 0.06, "C2_E-R3": 0.02}  # step-down x4, x3, x2, x1
        for k in want:
            self.assertAlmostEqual(adj[k], want[k], places=12)
        raw, adj = A.holm_family(blocks(p), fam, {"E-R2": False, "E-R3": True})  # a missing E-R2 cell
        self.assertEqual((raw["C1_E-R2"], raw["C2_E-R2"]), (1.0, 1.0))
        self.assertAlmostEqual(adj["C2_E-R3"], 0.02, places=12)
        self.assertAlmostEqual(adj["C1_E-R3"], 0.09, places=12)
        raw, adj = A.holm_family({"E-R3": blocks(p)["E-R3"]}, ["E-R3"], {"E-R3": True})  # an arm dropped at the mint
        self.assertEqual(list(raw), ["C1_E-R3", "C2_E-R3"])
        self.assertAlmostEqual(adj["C2_E-R3"], 0.01, places=12)
        self.assertAlmostEqual(adj["C1_E-R3"], 0.03, places=12)


# ----------------------------------------------------------------------------- record selection: section 5.2

class TestSelection(unittest.TestCase):
    def test_predicates_equal_variant_lists(self):
        bat = synth_battery(6)
        A.check_battery(bat)
        sel = A.check_selection(bat)
        parents = {q: len({bat[i]["parent_id"] for i in idx}) for q, idx in sel.items()}
        v7 = Counter(r["rule_ast"]["op"] for r in bat if r["group"] == "V7_bigset" and r["surface"] == "p3"
                     and r["variant"] == "listed")
        self.assertEqual(parents["C1"], 5 * 6 + 6 + 6 + v7["S+"])
        self.assertEqual((parents["C2"], parents["G1"], parents["G2"], parents["G3"]), (42, 42, 30, 60))
        self.assertEqual(parents["G4"], 30 + 6 + 6 + 6 + v7["S+"])
        self.assertEqual((parents["G5"], parents["G6"]), (6 + 6 + v7["S-"], 6 + 6 + 6 + v7["S-"]))
        self.assertEqual((parents["C1-nF"], parents["G4'"], parents["G5'"]),
                         (6 + 6 + v7["S+"], 18 + v7["S+"], 6 + v7["S-"]))
        p3 = {i for i, r in enumerate(bat) if r["surface"] == "p3"}
        parts = [set(sel[q]) for q in ("C1", "G4", "G5", "G6")]
        self.assertEqual(set().union(*parts), p3)  # the p3 records partition into C1, G4, G5, G6
        self.assertEqual(sum(map(len, parts)), len(p3))
        self.assertEqual(len(sel["G3"]), len(bat))
        self.assertEqual({bat[i]["surface"] for i in sel["C1-in"]}, {"in"})

    def test_mismatch_refused(self):
        bat = synth_battery(6)
        i = next(i for i, r in enumerate(bat) if r["surface"] == "p3" and r["group"] == "V1_pos"
                 and r["variant"] == "unlisted")
        bat[i]["variant"] = "listed"  # the C1 list now misses a record its predicate selects
        with self.assertRaises(SystemExit) as cm:
            A.check_selection(bat)
        self.assertIn("C1", str(cm.exception))
        self.assertIn("variant list and the predicate", str(cm.exception))
        bat = synth_battery(6)
        j = next(i for i, r in enumerate(bat) if r["surface"] == "p3" and r["group"] == "F2_unseen_real"
                 and r["variant"] == "probe")
        bat.append(copy.deepcopy(bat[j]))  # two C1 records for one parent
        with self.assertRaises(SystemExit) as cm:
            A.check_selection(bat)
        self.assertIn("records each", str(cm.exception))


# ----------------------------------------------------------------------------- decision branches on synthetic preds

class TestBranches(unittest.TestCase):
    def test_advance(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95}, "E-R3": {"C1": 0.2}}, off=True)
        self.assertEqual(res["arm_decisions"], {"E-R2": "ADVANCE", "E-R3": "STOP"})
        self.assertEqual((res["decision"], res["experiment_decision"]["arms"],
                          res["experiment_decision"]["lead_candidate"]), ("ADVANCE", ["E-R2"], "E-R2"))
        e = res["estimates"]["E-R2"]["q"]
        self.assertGreaterEqual(e["C1"]["diff"], 0.6)
        self.assertEqual(sorted(e["C1"]["per_seed"]), ["s0", "s1", "s2"])
        for g in A.GUARDS:  # equal accuracy on guard items: identical outcomes, a zero contrast
            self.assertEqual((e[g]["diff"], e[g]["lo"], e[g]["hi"]), (0.0, 0.0, 0.0))
        self.assertEqual(res["holm"]["family"], ["C1_E-R2", "C2_E-R2", "C1_E-R3", "C2_E-R3"])
        self.assertEqual(res["holm"]["raw_p"]["C1_E-R3"], 1.0)  # identical arms: ties on both sides
        self.assertTrue(res["secondary"]["vs_OFF"])
        self.assertEqual(res["secondary"]["template_cluster_c1"]["E-R2"]["c1_gain_shown_across_p3_wordings"], "yes")
        self.assertEqual(res["labels"]["E-R2"]["c1_gain_beyond_posted_rule"], "yes")
        self.assertFalse(res["labels"]["E-R2"]["degenerate_on_non_anchor_items"])
        self.assertFalse(res["parameters"]["prereg_resamples"])

    def test_both_advance_lead(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95}, "E-R3": {"C1": 0.8}})
        self.assertEqual(res["arm_decisions"], {"E-R2": "ADVANCE", "E-R3": "ADVANCE"})
        ex = res["experiment_decision"]
        self.assertEqual((ex["decision"], ex["arms"], ex["lead_candidate"]), ("ADVANCE", ["E-R2", "E-R3"], "E-R2"))

    def test_stop_null(self):
        res = run({})
        self.assertEqual(res["arm_decisions"], {"E-R2": "STOP", "E-R3": "STOP"})
        self.assertEqual(res["decision"], "STOP")
        self.assertEqual(set(res["holm"]["raw_p"].values()), {1.0})

    def test_stop_degenerate(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95, "G5": 0.1}, "E-R3": {"C1": 0.2}})
        d = res["decision_trace"]["E-R2"]
        self.assertEqual((d["decision"], d["also_satisfied"]), (A.STOP_DEG, ["REVISE"]))
        self.assertTrue(d["conditions"]["d(G5) <= -0.15"])
        self.assertFalse(d["conditions"]["guards_ok"])
        self.assertEqual(res["arm_decisions"]["E-R3"], "STOP")
        self.assertEqual(res["decision"], "STOP")  # every arm STOP of either kind
        self.assertTrue(res["labels"]["E-R2"]["degenerate_on_non_anchor_items"])  # G5' carries the whole loss
        self.assertIn("degenerate on non-anchor items", A.render(res))

    def test_revise(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.33}, "E-R3": {"C1": 0.2}})
        d = res["decision_trace"]["E-R2"]
        self.assertEqual(d["decision"], "REVISE")
        self.assertIn("d(C1) >= +0.25", d["failed_advance"])
        self.assertTrue(d["conditions"]["p_H(C1) < 0.05"])
        self.assertEqual((res["arm_decisions"]["E-R3"], res["decision"]), ("STOP", "REVISE"))

    def test_inconclusive_c2_only(self):
        res = run({"N-CFSD": {"G6": 0.3}, "E-R2": {"G6": 0.97}, "E-R3": {"G6": 0.3}})
        d = res["decision_trace"]["E-R2"]
        self.assertEqual(res["estimates"]["E-R2"]["q"]["C1"]["diff"], 0.0)
        self.assertTrue(d["conditions"]["sig"])
        self.assertFalse(d["conditions"]["d(C1) > 0"])
        self.assertFalse(d["outcomes"]["STOP"])
        self.assertEqual((d["decision"], res["decision"]), ("INCONCLUSIVE", "INCONCLUSIVE"))

    def test_missing_treatment_cell(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95}, "E-R3": {"C1": 0.35}}, drop_cells=("E-R2_s1",))
        self.assertEqual(res["arm_decisions"]["E-R2"], A.INC_MISSING)
        self.assertEqual((res["holm"]["raw_p"]["C1_E-R2"], res["holm"]["raw_p"]["C2_E-R2"]), (1.0, 1.0))
        self.assertEqual(res["estimates"]["E-R2"]["seeds"], [0, 2])  # descriptive, matched seeds only
        self.assertGreater(res["estimates"]["E-R2"]["q"]["C1"]["diff"], 0.5)  # a large gain decides nothing
        self.assertEqual(res["arm_decisions"]["E-R3"], "REVISE")  # d(C1) ~ +0.15, Holm still over the 4 tests
        self.assertEqual(res["decision"], "REVISE")

    def test_missing_control_cell(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95}}, drop_cells=("N-CFSD_s0",))
        self.assertEqual(res["arm_decisions"], {"E-R2": A.INC_MISSING, "E-R3": A.INC_MISSING})
        self.assertEqual(res["decision"], "INCONCLUSIVE")

    def test_dropped_arm_family(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R3": {"C1": 0.95}}, family=("E-R3",))
        self.assertEqual(res["holm"]["family"], ["C1_E-R3", "C2_E-R3"])
        self.assertEqual((res["arm_decisions"], res["decision"]), ({"E-R3": "ADVANCE"}, "ADVANCE"))
        self.assertIsNone(res["secondary"]["head_to_head_E-R2_minus_E-R3"])

    def test_symmetric_rescore_reading(self):
        res = run({}, preamble=("E-R2",))  # E-R2 writes "Answer:" before an otherwise control-identical answer
        dis = res["rescore"]["parser_disagree"]
        n = len(synth_battery(6))
        self.assertEqual(dis["E-R2_s0"], {"records": n, "scores_changed": n})
        self.assertEqual(dis["N-CFSD_s0"], {"records": 0, "scores_changed": 0})
        self.assertEqual(dis["E-R3_s0"]["records"], n)  # two-line outputs: only the last-line parser reads them
        self.assertEqual(dis["E-R3_s0"]["scores_changed"], n)
        self.assertEqual(res["estimates"]["E-R2"]["q"]["G3"]["diff"], 1.0)  # frozen: every output a format error
        self.assertEqual(res["rescore"]["estimates"]["E-R2"]["q"]["G3"]["diff"], 0.0)
        self.assertIn("hi(G3) <= +0.02", res["rescore"]["reading"]["E-R2"])
        self.assertEqual(res["rescore"]["reading"]["E-R3"], [])
        self.assertIn("conditions that differ between the two scorings", A.render(res))

    def test_r3_trace_diagnostics(self):
        res = run({"N-CFSD": {"C1": 0.2}, "E-R3": {"C1": 0.5}}, family=("E-R3",))
        bat = synth_battery(6)
        sel = A.check_selection(bat)
        tr = res["secondary"]["r3_trace"]["E-R3_s0"]
        n_p3 = sum(r["surface"] == "p3" for r in bat)
        self.assertEqual((tr["p3"]["well_formed"]["k"], tr["p3"]["well_formed"]["n"]), (n_p3, n_p3))
        self.assertEqual(tr["p3"]["line1_coherent"]["n"], n_p3)
        outs = outputs(bat, "E-R3", 0, {"E-R3": {"C1": 0.5}})
        wrong = sum(1 for i in sel["C1"] if outs[i].split("\n")[0].endswith("-> apply"))
        self.assertEqual(tr["C1_items"]["stated_apply"], wrong)
        self.assertEqual(tr["C1_items"]["listed_ok"]["k"], len(sel["C1"]))  # the scripted line 1 keeps listed=no
        sc = res["secondary"]["r3_stated_decision_c1"]
        self.assertAlmostEqual(sc["diff"], res["estimates"]["E-R3"]["q"]["C1"]["diff"], places=12)

    def test_template_cluster(self):
        bat = synth_battery(6)
        cells = make_cells(bat, {"N-CFSD": {"C1": 0.0}, "E-R2": {"C1": 1.0}}, arms=("N-CFSD", "E-R2"))
        sel = A.check_selection(bat)
        t = A.template_cluster(bat, cells, sel, "E-R2", [0, 1, 2], resamples=200)
        self.assertEqual((t["diff"], t["lo"], t["hi"], t["c1_gain_shown_across_p3_wordings"]), (1.0, 1.0, 1.0, "yes"))
        tpls = {bat[i]["p3_template"] for i in sel["C1"]}
        self.assertEqual(sum(t["templates_per_form"].values()), len(tpls))
        self.assertEqual(t, A.template_cluster(bat, cells, sel, "E-R2", [0, 1, 2], resamples=200))  # deterministic
        self.assertEqual(A.TEMPLATE_BOOT_SEED, t["seed"])


# ----------------------------------------------------------------------------- custody: files in a temp dir

class SecretGuard:
    """Patches builtins.open and io.open: reading a *.secret / P3_SEALED file outside `allowed` raises first."""

    def __init__(self, allowed):
        self.allowed = Path(allowed).resolve()

    def _open(self, file, mode="r", *args, **kw):
        if isinstance(file, (str, os.PathLike)):
            p = Path(os.fspath(file)).resolve()
            if (p.name.endswith(".secret") or "P3_SEALED" in p.name) and ("r" in mode or "+" in mode) \
                    and self.allowed not in p.parents:
                raise AssertionError(f"test code tried to read a sealed file: {p.name}")
        return _REAL_OPEN(file, mode, *args, **kw)

    def start(self, tc):
        for target in ("builtins.open", "io.open"):
            pt = mock.patch(target, self._open)
            pt.start()
            tc.addCleanup(pt.stop)


def gz_json(p, obj):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(json.dumps(obj))
    return A.sha(p)


def gz_lines(p, rows):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return A.sha(p)


class Sealed(unittest.TestCase):
    """A complete fake SEALED directory: commitment (bound code = the current files), opening record, adapters, every
    cell's eval / preds / raw generations (preds by native_e01r.score_outputs), TRAIN_META, SMOKE and covariates."""

    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        SecretGuard(self.tmp).start(self)
        self.cells, self.e01n = self.tmp / "cells", self.tmp / "e01n"
        for name, val in (("P3_PATH", self.tmp / "pools" / "P3_SEALED_v2.json.secret"), ("E01N_CELLS", self.e01n)):
            pt = mock.patch.object(M, name, val)
            pt.start()
            self.addCleanup(pt.stop)
        self.addCleanup(RC.unregister_pool, "p3")

    def build(self, accs=None, family=("E-R2", "E-R3"), dropped=(), overlap=0, off=True, n_parents=3):
        accs = {"N-CFSD": {"C1": 0.2}, "E-R2": {"C1": 0.95}, "E-R3": {"C1": 0.2}} if accs is None else accs
        p3sha = write(M.P3_PATH, json.dumps({"schema": "rcx-e01r-pool-v1", "templates": fake_p3_templates()}))
        seed = M.NE.sealed_seed(SALT)
        skel = M.build_skeleton(seed, COUNTS, n_parents)
        rows = [{"parent_id": "a", "variant": "base", "semantic_hash": "h1", "prompt": "SYS1<|im_start|>user U1"},
                {"parent_id": "a", "variant": "comp_listed", "semantic_hash": "h1",
                 "prompt": "SYS1<|im_start|>user U1b"},  # duplicate: same wording, rule, state and event
                {"parent_id": "a", "variant": "comp_unlisted", "semantic_hash": "h2",
                 "prompt": "SYS1<|im_start|>user U2"},
                {"parent_id": "b", "variant": "status_swap", "semantic_hash": "h3", "prompt": "SYS2<|im_start|>user"},
                {"parent_id": "b", "variant": "comp_listed", "semantic_hash": "h3", "prompt": "SYS3<|im_start|>user"}]
        rows_sha = {"E-R2": write(self.cells / "train_E-R2.jsonl", "".join(json.dumps(r) + "\n" for r in rows)),
                    "E-R3": write(self.cells / "train_E-R3.jsonl", "{}\n")}
        tok = {str(k): {"supervised_tokens": 1000 + k, "tokens_processed": 9000 + k} for k in range(3)}
        man = write(self.cells / A.MANIFEST, json.dumps({"max_new": MAX_NEW, "r3_gold_longest_tokens": {"x": 50},
                                                         "arms": {a: {"sha256": rows_sha[a], "rows": 5,
                                                                      "recipe_stats": {"dup_rows_within_parent": 1}}
                                                                  for a in rows_sha}}))
        aud = write(self.cells / A.AUDIT, json.dumps({
            "ok": True, "arms": {a: {"rows": 5, "parents": 2, "noop_share": 0.4, "rows_by_rule_type": {"S+": 5},
                                     "tokens": {"per_seed": tok}} for a in rows_sha},
            "control": {"rows": 5, "parents": 2, "noop_share": 0.4, "tokens": {"per_seed": tok}}}))
        tags = [f"{a}_s{k}" for a in ("N-CFSD",) + tuple(family) for k in range(3)]
        code = A.current_code_sha256(M)
        commit = {"experiment": "RCX-E01R", "sha256_salt": hashlib.sha256(SALT.encode()).hexdigest(),
                  "battery_commit": M.NE.battery_commit(SALT, M.NE.battery_sha256(skel)), "n_parents": n_parents,
                  "strata": list(M.STRATA), "surfaces": ["in", "p3"], "forms": list(M.FORMS),
                  "p3": {"file": M.P3_PATH.name, "sha256": p3sha, "n": 48, "counts": COUNTS}, "arms_required": tags,
                  "treatment_arms": list(family), "branch": {"file": "BRANCH_STEP_A.json", "branch": "B4"},
                  "dropped_arms": list(dropped), "max_new": MAX_NEW, "train_manifest_sha256": man,
                  "rows_sha256": rows_sha, "audit_e01r_sha256": aud, "code_sha256": code, "minted_at": "mint"}
        self.commit_sha = write(self.cells / "SEALED_COMMIT.json", json.dumps(commit, indent=1))
        battery = M.render_p3(skel, M.load_p3(commit["p3"]))
        bsha = M.NE.battery_sha256(battery)
        adapters = {}
        for t in tags:
            arm = t.rsplit("_s", 1)[0]
            d = (self.e01n if arm == "N-CFSD" else self.cells) / t
            adapters[t] = write(d / "adapter" / "adapter_model.safetensors", f"adapter {t}")
            if arm != "N-CFSD":
                write(d / "TRAIN_META.json",
                      json.dumps({"arm": arm, "adapter_sha256": adapters[t], "code_sha256": code}))
                write(self.cells / f"SMOKE_{t}.json", json.dumps({
                    "adapter_sha256": adapters[t], "max_new": MAX_NEW, "n": 50, "truncated": 0, "truncated_share": 0.0,
                    "well_formed_share": 1.0, "scorable_share": 1.0, "exact_share": 0.9, "wall_s": 60.0, "pass": True}))
        pids = sorted({r["parent_id"] for r in battery})
        opened = {"opened_at": "open", "salt": SALT, "seed": seed, "battery_sha256": bsha, "n_records": len(battery),
                  "sealed_commit_sha256": self.commit_sha, "adapters_sha256": adapters, "max_new": MAX_NEW,
                  "train_overlap": {"parent_ids": pids[:overlap]}, "code_changed_since_mint": [],
                  "max_new_covers_r3_gold_byte_bound": True}
        self.open_sha = write(self.cells / "SEALED_OPEN.json", json.dumps(opened, indent=1))
        self.battery, self.code, self.adapters = battery, code, adapters
        for t in tags + (["OFF"] if off else []):
            self.cell(t, accs)
        return battery

    def cell(self, tag, accs):
        arm, k = ("OFF", 0) if tag == "OFF" else (tag.rsplit("_s", 1)[0], int(tag.rsplit("_s", 1)[1]))
        outs = outputs(self.battery, arm, k, accs)
        n = len(outs)
        bsha = M.NE.battery_sha256(self.battery)
        ad = None if arm == "OFF" else self.adapters[tag]
        raw = {"tag": tag, "battery_sha256": bsha, "adapter_sha256": ad, "max_new": MAX_NEW, "n": n, "arm": arm,
               "outs": outs, "prompt_tokens": [100] * n, "decode_calls": [[o, True] for o in outs], "wall_s": 1.0}
        raw_sha = gz_json(M.raw_path(self.cells, tag), raw)
        parse, parser = M.answer_parser(arm)
        _, _, rows = M.score_outputs(self.battery, raw, arm, parse)
        gz_lines(self.cells / f"preds_sealed_{tag}.jsonl.gz", rows)
        write(self.cells / f"eval_sealed_{tag}.json", json.dumps({
            "arm": arm, "seed": None if arm == "OFF" else k, "tag": tag, "battery": "sealed",
            "battery_seed": M.NE.sealed_seed(SALT), "battery_sha256": bsha, "adapter_sha256": ad, "n": n,
            "surfaces": ["in", "p3"], "max_new": MAX_NEW, "parser": parser, "truncated": 0, "wall_s": 1.0,
            "raw_file": M.raw_path(self.cells, tag).name, "raw_sha256": raw_sha, "sealed_open_sha256": self.open_sha,
            "code_sha256": self.code}))

    def load(self, missing=()):
        with contextlib.redirect_stdout(io.StringIO()):
            return A.load_sealed(self.cells, list(missing), ne=M)

    def refused(self, *needles, missing=()):
        with self.assertRaises(SystemExit) as cm:
            self.load(missing)
        for x in needles:
            self.assertIn(x, str(cm.exception))
        return str(cm.exception)

    def edit_json(self, p, **kw):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        d.update(kw)
        Path(p).write_text(json.dumps(d, indent=1), encoding="utf-8")


class TestCustody(Sealed):
    def test_end_to_end_main(self):
        self.build()
        out = self.tmp / "report" / "ANALYSIS_E01R.md"
        with mock.patch.object(A, "RESAMPLES", R), contextlib.redirect_stdout(io.StringIO()) as buf:
            res = A.main(["--cells", str(self.cells), "--out", str(out)])
        self.assertEqual((res["decision"], res["arm_decisions"]), ("ADVANCE", {"E-R2": "ADVANCE", "E-R3": "STOP"}))
        md, js = out.read_text(encoding="utf-8"), out.with_suffix(".json").read_text(encoding="utf-8")
        self.assertIn("EXPERIMENT DECISION: ADVANCE", md)
        self.assertIn("NOT THE PREREGISTERED ANALYSIS", md)
        self.assertIn("DECISION: ADVANCE", buf.getvalue())
        for text in (md, js, buf.getvalue()):  # no P3 text, prompt or model output leaves the analysis
            self.assertNotIn("Fake ", text)
            self.assertNotIn("decides it", text)
            self.assertNotIn("check status=", text)
        j = json.loads(js)
        self.assertEqual(j["inputs"]["covariates"]["r2_duplicate_comp_listed"], 1)
        self.assertEqual(j["inputs"]["supervised_tokens_seed_mean"]["E-R2"], 1001.0)
        self.assertEqual(j["inputs"]["sealed_commit_sha256"], self.commit_sha)
        self.assertEqual(set(j["inputs"]["bound_code_sha256"]), set(self.code))
        self.assertEqual(j["inputs"]["warnings"], [])
        self.assertEqual(set(j["inputs"]["cells"]), {f"{a}_s{k}" for a in ARMS for k in range(3)} | {"OFF"})
        with self.assertRaises(SystemExit) as cm:  # create-only: the analysis runs once
            A.main(["--cells", str(self.cells), "--out", str(out)])
        self.assertIn("runs once", str(cm.exception))

    def test_bound_code_refused_before_anything(self):
        self.build()
        cp = self.cells / "SEALED_COMMIT.json"
        original = cp.read_text(encoding="utf-8")
        for name in ("stats.py", "analyze_e01r.py", "native_e01r.py", "world.py"):
            self.edit_json(cp, code_sha256=dict(self.code, **{name: "0" * 64}))
            with mock.patch.object(M, "sealed_battery", side_effect=AssertionError("battery rebuilt")):
                self.refused("bound code differs", name)
            cp.write_text(original, encoding="utf-8")
        self.edit_json(cp, code_sha256={k: v for k, v in self.code.items() if k != "lane.py"})
        self.refused("does not bind", "lane.py")
        cp.write_text(original, encoding="utf-8")
        self.edit_json(cp, code_sha256=dict(self.code, **{"elsewhere.py": "0" * 64}))
        self.refused("cannot locate", "elsewhere.py")

    def test_recorded_code_shas(self):
        self.build()
        self.edit_json(self.cells / "eval_sealed_E-R2_s0.json", code_sha256=dict(self.code, **{"gen.py": "1" * 64}))
        self.refused("E-R2_s0", "eval_sealed_E-R2_s0.json records bound-code shas", "gen.py")
        self.build()
        self.edit_json(self.cells / "E-R3_s1" / "TRAIN_META.json",
                       code_sha256=dict(self.code, **{"stats.py": "2" * 64}))
        self.refused("E-R3_s1", "TRAIN_META.json records bound-code shas", "stats.py")
        self.build()
        self.edit_json(self.cells / "SMOKE_E-R2_s2.json", code_sha256=dict(self.code, **{"recipes.py": "3" * 64}))
        self.refused("E-R2_s2", "SMOKE_E-R2_s2.json records bound-code shas", "recipes.py")

    def test_cell_integrity_refusals(self):
        cases = [
            (lambda: self.edit_json(self.cells / "eval_sealed_N-CFSD_s1.json", battery_sha256="f" * 64),
             ("N-CFSD_s1", "not the rendered SEALED battery")),
            (lambda: write(self.e01n / "N-CFSD_s2" / "adapter" / "adapter_model.safetensors", "retrained"),
             ("N-CFSD_s2", "recorded by open-sealed")),
            (lambda: write(self.cells / "E-R3_s0" / "adapter" / "adapter_model.safetensors", "other"),
             ("E-R3_s0", "recorded by open-sealed")),
            (lambda: self.edit_json(self.cells / "eval_sealed_E-R2_s1.json", max_new=MAX_NEW + 1),
             ("E-R2_s1", "max_new")),
            (lambda: self.edit_json(self.cells / "eval_sealed_E-R2_s1.json", parser="something else"),
             ("E-R2_s1", "parser")),
            (lambda: self.edit_json(self.cells / "eval_sealed_E-R2_s1.json", truncated=3),
             ("E-R2_s1", "truncation count")),
            (lambda: self.tamper_preds("E-R3_s2", "exact"), ("E-R3_s2", "stored scores differ from a re-score")),
            (lambda: self.tamper_preds("E-R3_s2", "trace"), ("E-R3_s2", "trace diagnostics differ")),
            (lambda: self.tamper_preds("E-R2_s0", "out"), ("E-R2_s0", "differ from the raw generations")),
            (lambda: M.raw_path(self.cells, "N-CFSD_s0").unlink(), ("N-CFSD_s0", "raw generations file is missing")),
            (lambda: self.tamper_raw("N-CFSD_s0"), ("N-CFSD_s0", "differs from the eval's record")),
        ]
        for tamper, needles in cases:
            self.build()
            tamper()
            msg = self.refused(*needles)
            self.assertIn("--missing does not name", msg)

    def tamper_preds(self, tag, key):
        p = self.cells / f"preds_sealed_{tag}.jsonl.gz"
        rows = A.load_preds(p)
        j = next(i for i, r in enumerate(rows) if r["exact"] == 1)
        if key == "exact":
            rows[j]["exact"] = 0
        elif key == "trace":
            rows[j]["trace"]["line1_exact"] = 1 - rows[j]["trace"]["line1_exact"]
        else:
            rows[j]["out"] = rows[j]["out"] + " "
        gz_lines(p, rows)

    def tamper_raw(self, tag):
        p = M.raw_path(self.cells, tag)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        raw["wall_s"] = 2.0
        gz_json(p, raw)

    def test_missing_reconciliation(self):
        self.build()
        (self.cells / "eval_sealed_E-R2_s1.json").unlink()
        self.refused("E-R2_s1", "no eval", "--missing does not name")
        battery, cells, info = self.load(["E-R2_s1"])
        self.assertNotIn("E-R2_s1", cells)
        self.assertIn("no eval", info["missing"]["E-R2_s1"])
        res = A.analyze(battery, cells, info, resamples=R)
        self.assertEqual(res["arm_decisions"]["E-R2"], A.INC_MISSING)
        self.assertEqual(res["holm"]["raw_p"]["C1_E-R2"], 1.0)
        self.assertIn("Missing cells", A.render(res))
        self.refused("which have a complete eval", "E-R3_s0", missing=["E-R2_s1", "E-R3_s0"])
        self.refused("--missing", missing=["E-R2_s1", "N-VR_s0"])
        self.refused("--missing", missing=["E-R2_s1", "E-R2_s1"])
        # a cell failing an integrity check may be declared missing (section 10, reason 3 or 4)
        write(self.cells / "E-R3_s2" / "adapter" / "adapter_model.safetensors", "lost and not restorable")
        _, cells, info = self.load(["E-R2_s1", "E-R3_s2"])
        self.assertNotIn("E-R3_s2", cells)
        self.assertIn("recorded by open-sealed", info["missing"]["E-R3_s2"])

    def test_off_is_descriptive(self):
        self.build(off=False)
        write(self.cells / "eval_sealed_N-VR_s0.json", "{}")  # section 4: N-VR is not evaluated; never read here
        battery, cells, info = self.load()
        self.assertNotIn("OFF", cells)
        self.assertIn("no eval", info["missing"]["OFF"])
        self.assertTrue(any("eval_sealed_N-VR_s0.json" in w for w in info["warnings"]))
        res = A.analyze(battery, cells, info, resamples=R)
        self.assertEqual(res["secondary"]["vs_OFF"], {})
        self.assertEqual(res["arm_decisions"]["E-R2"], "ADVANCE")
        self.build()
        self.edit_json(self.cells / "eval_sealed_OFF.json", max_new=MAX_NEW + 1)  # present but failing: not ignored
        self.refused("OFF", "max_new")
        _, cells, info = self.load(["OFF"])
        self.assertNotIn("OFF", cells)

    def test_arm_structure_refusals(self):
        self.build()
        cp = self.cells / "SEALED_COMMIT.json"
        original = cp.read_text(encoding="utf-8")
        for kw, needle in (({"branch": {"branch": "B3"}}, "branch B4"),
                           ({"arms_required": [f"N-CFSD_s{k}" for k in range(3)] + ["E-R2_s0", "E-R2_s1"]},
                            "arms_required"),
                           ({"treatment_arms": ["E-R2"]}, "treatment arms"),
                           ({"treatment_arms": ["E-R2"], "dropped_arms": [{"arm": "E-R3", "reason": " "}]},
                            "treatment arms"),
                           ({"treatment_arms": ["E-R2", "E-R1"]}, "treatment arms")):
            self.edit_json(cp, **kw)
            self.refused(needle)
            cp.write_text(original, encoding="utf-8")

    def test_dropped_arm(self):
        self.build(accs={"N-CFSD": {"C1": 0.2}, "E-R3": {"C1": 0.95}}, family=("E-R3",),
                   dropped=[{"arm": "E-R2", "reason": "rows fail the training-row audit"}])
        battery, cells, info = self.load()
        self.assertEqual(info["treatment_arms"], ["E-R3"])
        res = A.analyze(battery, cells, info, resamples=R)
        self.assertEqual(res["holm"]["family"], ["C1_E-R3", "C2_E-R3"])
        self.assertEqual(res["arm_decisions"], {"E-R3": "ADVANCE"})
        self.assertIn("dropped at the mint: E-R2 (rows fail the training-row audit)", A.render(res))

    def test_train_overlap_drop(self):
        self.build(overlap=2)
        battery, cells, info = self.load()
        dropped = info["dropped_train_overlap_parents"]
        self.assertEqual(len(dropped), 2)
        self.assertFalse({r["parent_id"] for r in battery} & set(dropped))
        n_drop = sum(r["parent_id"] in dropped for r in self.battery)
        self.assertEqual(len(battery), len(self.battery) - n_drop)
        self.assertEqual(info["n_records_rebuilt"], len(self.battery))
        self.assertTrue(all(len(c["frozen"]) == len(battery) == len(c["out"]) for c in cells.values()))
        A.check_selection(battery)

    def test_battery_and_opening_checks(self):
        self.build()
        self.edit_json(self.cells / "SEALED_OPEN.json", battery_sha256="e" * 64)
        self.refused("differs from the one recorded at opening")
        self.build()
        self.edit_json(self.cells / "SEALED_OPEN.json", max_new=MAX_NEW + 8)
        self.refused("max_new")
        self.build()
        (self.cells / "SEALED_OPEN.json").unlink()
        self.refused("not open")

    def test_covariate_files_checked(self):
        self.build()
        self.edit_json(self.cells / A.AUDIT, note="edited after the mint")
        _, _, info = self.load()
        self.assertEqual(info["supervised_tokens_seed_mean"], {})  # never used for the tie-break when it differs
        self.assertTrue(any("AUDIT_E01R.json differs" in w for w in info["warnings"]))
        self.assertFalse(info["covariates"]["audit_matches_commitment"])

    def test_r2_duplicate_count(self):
        rows = [{"parent_id": "p", "variant": "base", "semantic_hash": "s", "prompt": "A<|im_start|>user x"},
                {"parent_id": "p", "variant": "comp_listed", "semantic_hash": "s", "prompt": "A<|im_start|>user y"},
                {"parent_id": "p", "variant": "comp_listed", "semantic_hash": "s", "prompt": "B<|im_start|>user x"},
                {"parent_id": "q", "variant": "comp_listed", "semantic_hash": "s", "prompt": "A<|im_start|>user x"},
                {"parent_id": "p", "variant": "rule_flip", "semantic_hash": "t", "prompt": "C<|im_start|>user"},
                {"parent_id": "p", "variant": "comp_listed", "semantic_hash": "t", "prompt": "C<|im_start|>user"}]
        self.assertEqual(A.r2_duplicate_comp_listed(rows), 1)  # only the first comp_listed matches base/status_swap


if __name__ == "__main__":
    unittest.main()
