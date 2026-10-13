"""Tests for experiments/RCX-E05/analyze_e05.py (PREREG_E05_DRAFT.md §6-7, §2.2; DESIGN rev 3 §6.4-6.5).

Synthetic per-parent nreg arrays only (no oracle, no model; literal seeds and string-seeded random.Random). The attempt
directory, pilot report and a throwaway git repository holding DECISIONS.md live in temp dirs (nothing under runs/).
Bootstraps use R = 400 (R = 200 through main) except the exact rcx.stats equivalence, which uses the prereg's 10,000.
Never reported as results.
Run: python -B -m unittest tests.test_e05_analyze -v   (CPU, one thread; a few seconds)
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx import stats as ST  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("analyze_e05_ut", "experiments/RCX-E05/analyze_e05.py")
R = 400
REPS = A.REPLICATES
N_STARTS = 4
SIZES = {"H-ID-S": 4, "H-ID-U": 6, "H-K": 8, "H-NEG": 8, "H-COST": 4}
FACTS4 = A._facts(4)


def wvec(items, n=4):
    w = [0] * (n * (n + 1))
    for f, x in items.items():
        w[FACTS4.index(f) if n == 4 else A._facts(n).index(f)] = x
    return w


def make_manifest(sizes=SIZES, n_starts=N_STARTS):
    parents, subset = [], []
    for h, n in sizes.items():
        fam, st = ("H-ID", h[-1]) if h.startswith("H-ID-") else (h, "")
        for i in range(n):
            conflict = i % 4 == 0  # on(0, 1) with ontable(0): unsatisfiable
            w = wvec({("on", 0, 1): 2, ("ontable", 0 if conflict else 2): 1})
            parents.append(dict(family=fam, stratum=st, key=f"lin|free|{h}|{i:02d}", starts=list(range(n_starts)),
                                K=2 + i % 3, n_neg=1 if fam == "H-NEG" else 0, rtype="lin", w=w))
            if fam in A.PRIMARY and i < (2 if fam == "H-ID" else 4):
                subset.append(parents[-1]["key"])
    return dict(parents=parents, subset=subset)


MANIFEST = make_manifest()
DESIGN = A.make_design(MANIFEST)
DET = dict(p1_comparator_invalid=False, p2_dropped=False, pg400=A.PGV, q9_failed_no_bug=False,
           kstar_descriptive=False, r_ext=False, cut_items=[])
IDEAL = {A.SF: 0.20, A.UVFA: 0.35, A.AUX: 0.30, A.NOGPI: 0.30, A.SFL: 0.40, A.SF5: 0.22, A.SHUF: 0.90, A.PGV: 0.22,
         A.FT: 0.24, A.SFFT: 0.21, "PGV_1600": 0.15, "PGV_6400": 0.10, A.ENUM: 0.25, A.ENUM_H: 0.10, A.GNN: 0.40,
         A.XGPI: 0.30, A.BEST: 0.50, A.GREEDY: {"H-NEG": 0.8, "*": 1.0}, A.STOP: 1.0, A.RAND: 0.95, A.ORACLE: 0.0}
EXACT = {A.GREEDY, A.STOP, A.ORACLE}  # floors without noise


def _value(m, info, i, rep):
    if callable(m):
        return m(info, i, rep)
    if isinstance(m, dict):
        return m.get(info["stratum"], m.get(info["family"], m.get("*")))
    return m


def make_cells(design=DESIGN, spec=None, noise=None, amp=0.03):
    """{arm: {replicate: {key: {"nreg": [x] * starts}}}}: per-parent x = base + amp * u, u ~ U(-1, 1) seeded by
    (arm, key, replicate), clipped to [0, 1]; replicated arms get 3 replicates (SF-shuffled / SF-FT_400: 20261246)."""
    spec = dict(IDEAL, **(spec or {}))
    noise = noise or {}
    cells = {}
    for arm, m in spec.items():
        reps = (REPS[0],) if arm in (A.SHUF, A.SFFT) else (REPS if arm in A.REPLICATED else (None,))
        a = noise.get(arm, 0.0 if arm in EXACT else amp)
        for rep in reps:
            slot = cells.setdefault(arm, {}).setdefault(rep, {})
            for h, keys in design["strata"].items():
                for i, k in enumerate(keys):
                    base = _value(m, design["parents"][k], i, rep)
                    if base is None:
                        continue
                    x = min(1.0, max(0.0, base + a * random.Random(f"{arm}|{k}|{rep}").uniform(-1, 1)))
                    slot[k] = {"nreg": [x] * design["parents"][k]["n_starts"]}
    return cells


def run(spec=None, det=None, defects=(), drop=(), noise=None, resamples=R):
    cells = make_cells(spec=spec, noise=noise)
    for g in drop:
        arm, _, rep = g.partition("@")
        del cells[arm][int(rep) if rep else None]
    return A.analyze(DESIGN, cells, dict(DET, **(det or {})), defects=defects, resamples=resamples)


class Basics(unittest.TestCase):
    def test_derive_and_facts_match_e05_modules(self):
        fam = _load("families_e05_ana_ut", "experiments/RCX-E05/families_e05.py")
        from rcx.bw import soft as SO
        for tag in ("boot:H-ID-S", "boot:H-K", "test:1", ""):
            self.assertEqual(A.derive(20261250, tag), fam.derive(20261250, tag))
        for n in (3, 4, 5):
            self.assertEqual(tuple(A._facts(n)), SO.facts(n))

    def test_index_stream_is_rcx_stats_draw_order(self):
        idx = A.index_matrix("H-K", 7, 50)
        rng = random.Random(A.derive(20261250, "boot:H-K"))
        self.assertEqual(idx, [[rng.randrange(7) for _ in range(7)] for _ in range(50)])

    def test_tolerance(self):
        self.assertTrue(A._ge(0.03 - 5e-10, 0.03))
        self.assertFalse(A._ge(0.03 - 2e-9, 0.03))
        self.assertFalse(A._lt(0.05 - 5e-10, 0.05))
        self.assertTrue(A._lt(0.05 - 2e-9, 0.05))
        self.assertFalse(A._gt(5e-10, 0))
        self.assertTrue(A._le(0.5 + 5e-10, 0.5))
        self.assertFalse(A._le(0.5 + 2e-9, 0.5))

    def test_conflict_equals_unsatisfiable_at_n4(self):
        from rcx.bw import space as SP
        sp, rng = SP.get_space(4), random.Random(3)
        seen = set()
        for _ in range(300):
            K = rng.randint(1, 4)
            fs = rng.sample(FACTS4, K)
            cg = SP.CompiledGoal(list(range(4)), frozenset(fs))
            sat = cg.consistent and bool(sp.goal_mask(cg).any())
            self.assertEqual(A.has_conflict(wvec({f: 1 for f in fs})), not sat, fs)
            seen.add(sat)
        self.assertEqual(seen, {True, False})
        self.assertTrue(A.has_conflict(wvec({("on", 0, 1): 1, ("on", 1, 2): 1, ("on", 2, 0): 1})))  # 3-cycle
        self.assertFalse(A.has_conflict(wvec({("on", 0, 1): 1, ("on", 1, 0): -2})))  # negatives ignored

    def test_design_strata_sorted_and_subset(self):
        self.assertEqual(set(DESIGN["strata"]), set(SIZES))
        for h, ks in DESIGN["strata"].items():
            self.assertEqual(ks, sorted(ks))
        self.assertEqual(len(DESIGN["subset"]), 12)
        bad = dict(MANIFEST, parents=MANIFEST["parents"] + [dict(MANIFEST["parents"][0])])
        with self.assertRaises(ValueError):
            A.make_design(bad)
        noid = dict(MANIFEST, parents=[dict(MANIFEST["parents"][0], stratum="")])
        with self.assertRaises(ValueError):
            A.make_design(noid)
        dev = dict(MANIFEST, parents=MANIFEST["parents"] + [dict(family="DEV-ID", stratum="S", key="dev")])
        self.assertEqual(A.make_design(dev)["ignored"], {"DEV-ID": 1})


class BootstrapEquivalence(unittest.TestCase):
    """M3: on one stratum, E05's stratified bootstrap = rcx.stats.seed_parent_bootstrap_test exactly (R = 10,000)."""

    def setUp(self):
        sizes = {"H-ID-S": 5, "H-ID-U": 7, "H-K": 9, "H-NEG": 6}
        self.design = A.make_design(make_manifest(sizes))
        rng = random.Random(1)
        self.cells = {}
        for arm, reps in ((A.SF, REPS), (A.UVFA, REPS), (A.PGV, (None,))):
            for rep in reps:
                self.cells.setdefault(arm, {})[rep] = {
                    k: {"nreg": [rng.random() for _ in range(N_STARTS)]} for k in self.design["parents"]}

    def per_seed(self, arm, keys):
        return [{k: by_key[k]["nreg"] for k in keys} for _, by_key in sorted(self.cells[arm].items(),
                                                                                key=lambda t: A._rep_key(t[0]))]

    def check(self, x, y, h, est, family=None):
        keys = self.design["strata"][h]
        ref = ST.seed_parent_bootstrap_test(self.per_seed(y, keys), self.per_seed(x, keys), resamples=10000,
                                            seed=A.derive(20261250, f"boot:{h}"), p_floor=1e-4, ties="both")
        for s in (A.summ(est["strata"][h], 10000),) + ((A.summ(est["families"][family], 10000),) if family else ()):
            self.assertEqual(s["point"], ref["diff"])
            self.assertEqual((s["lo"], s["hi"], s["p"]), (ref["lo"], ref["hi"], ref["p"]))
        return ref

    def test_d1_every_primary_stratum_exact(self):
        est = A.contrast(self.design, self.cells, A.UVFA, A.SF, 10000)
        for h in ("H-ID-S", "H-ID-U", "H-K", "H-NEG"):
            self.check(A.UVFA, A.SF, h, est, family=h if h in ("H-K", "H-NEG") else None)
        self.assertEqual(est["families"]["H-ID"]["n"], 12)

    def test_d2_with_deterministic_arm_and_identical_arms(self):
        est = A.contrast(self.design, self.cells, A.SF, A.PGV, 10000)
        self.check(A.SF, A.PGV, "H-NEG", est, family="H-NEG")
        self.cells[A.UVFA] = {r: {k: {"nreg": list(c["nreg"])} for k, c in v.items()} for r, v in
                              self.cells[A.SF].items()}
        est = A.contrast(self.design, self.cells, A.UVFA, A.SF, 10000)
        ref = self.check(A.UVFA, A.SF, "H-K", est)
        self.assertEqual(ref["p"], 1.0)

    def test_family_and_macro_formulas(self):
        est = A.contrast(self.design, self.cells, A.UVFA, A.SF, 300)
        s, u = est["strata"]["H-ID-S"], est["strata"]["H-ID-U"]
        hid = est["families"]["H-ID"]
        self.assertEqual(hid["point"], (5 * s["point"] + 7 * u["point"]) / 12)
        self.assertEqual(hid["boot"][17], (5 * s["boot"][17] + 7 * u["boot"][17]) / 12)
        f = est["families"]
        self.assertEqual(est["macro"]["boot"][3], (f["H-ID"]["boot"][3] + f["H-K"]["boot"][3] + f["H-NEG"]["boot"][3])
                         / 3)


class Statistics(unittest.TestCase):
    def test_parent_value_order(self):
        cells = {A.SF: {REPS[1]: {"k": {"nreg": [0.1, 0.2, 0.4]}}, REPS[0]: {"k": {"nreg": [0.3, 0.3, 0.6]}}}}
        v = A.arm_values(cells, A.SF)["k"]
        self.assertEqual(v, (sum([0.3, 0.3, 0.6]) / 3 + sum([0.1, 0.2, 0.4]) / 3) / 2)
        self.assertEqual(A.arm_values(cells, A.SF, REPS[1])["k"], sum([0.1, 0.2, 0.4]) / 3)

    def test_ni_p(self):
        boot = [0.05] * 3 + [0.05 - 5e-10] * 2 + [0.0] * 95
        self.assertEqual(A.p_ni(boot, 100), 0.05)  # 0.05 - 5e-10 counts as >= 0.05 (tolerance)
        self.assertEqual(A.p_ni([0.049] * 100, 100), 0.01)  # floor 1/R
        self.assertEqual(A.p_ni([0.2] * 100, 100), 1.0)

    def test_iut_p_is_max_and_holm(self):
        res = run(spec={A.SF: 0.3, A.UVFA: lambda info, i, rep: 0.34 + (0.3 if i % 2 == 0 else -0.3),
                        A.PGV: 0.1, A.ENUM: 0.3}, noise={A.SF: 0.0, A.UVFA: 0.0})
        P1 = res["P1"]
        self.assertNotEqual(P1["p_macro"], P1["p_hid"])
        self.assertEqual(P1["p"], max(P1["p_macro"], P1["p_hid"]))
        self.assertEqual(res["raw_p"]["P1"], P1["p"])
        self.assertEqual(res["holm_family"], ["P1", "P2"])
        self.assertEqual(res["holm_adjusted"], ST.holm(res["raw_p"]))
        self.assertEqual(res["P2"]["p"], res["raw_p"]["P2"])

    def test_holm_family_dropped_and_invalid(self):
        res = run(det=dict(p2_dropped=True, pg400=None))
        self.assertEqual(res["holm_family"], ["P1"])
        self.assertEqual(res["holm_adjusted"], {"P1": res["raw_p"]["P1"]})
        self.assertIsNone(res["P2"]["passes"])
        res = run(det=dict(p1_comparator_invalid=True))
        self.assertEqual(res["holm_family"], ["P2"])
        self.assertEqual(res["holm_adjusted"], {"P2": res["raw_p"]["P2"]})
        self.assertIsNone(res["P1"]["passes"])
        self.assertIsNone(res["P1"]["p_holm"])
        res = run(det=dict(p1_comparator_invalid=True, p2_dropped=True, pg400=None))
        self.assertEqual((res["holm_family"], res["holm_adjusted"]), ([], {}))

    def test_validity_never_changes_holm_family(self):
        ideal, bad = run(), run(spec={A.UVFA: 0.9, A.PGV: 0.8}, noise={A.UVFA: 0.0, A.PGV: 0.0})
        self.assertFalse(bad["validity"]["p1_valid"] or bad["validity"]["p2_valid"])
        self.assertEqual(ideal["holm_family"], bad["holm_family"])
        self.assertEqual(bad["holm_adjusted"], ST.holm(bad["raw_p"]))
        self.assertTrue(bad["P1"]["passes"] and bad["P2"]["passes"])

    def test_condition4_on_hid(self):
        hid = {REPS[0]: 0.40, REPS[1]: 0.40, REPS[2]: 0.18}
        res = run(spec={A.UVFA: lambda info, i, rep: hid[rep] if info["family"] == "H-ID" else 0.35},
                  noise={A.SF: 0.0, A.UVFA: 0.0})
        c = res["P1"]["conditions"]
        self.assertTrue(c["holm_p"] and c["points"] and c["lower_bounds"])
        self.assertFalse(c["replicate_signs"])
        self.assertLess(res["P1"]["replicates"][str(REPS[2])]["hid"], 0)
        self.assertGreater(res["P1"]["replicates"][str(REPS[2])]["macro"], 0)
        self.assertFalse(res["P1"]["passes"])
        self.assertNotEqual(res["decision"], A.ADVANCE)

    def test_condition4_on_macro(self):
        res = run(spec={A.UVFA: lambda info, i, rep: 0.05 if rep == REPS[2] and info["family"] in ("H-K", "H-NEG")
                        else 0.35}, noise={A.SF: 0.0, A.UVFA: 0.0})
        r3 = res["P1"]["replicates"][str(REPS[2])]
        self.assertLess(r3["macro"], 0)
        self.assertGreater(r3["hid"], 0)
        c = res["P1"]["conditions"]
        self.assertTrue(c["holm_p"] and c["points"] and c["lower_bounds"])
        self.assertFalse(c["replicate_signs"] or res["P1"]["passes"])

    def test_kstar(self):
        self.assertEqual(A.kstar_point(0.3, [0.2, 0.1, 0.0], A.KS), -math.inf)
        self.assertEqual(A.kstar_point(0.0, [0.2, 0.1, 0.05], A.KS), math.inf)
        self.assertAlmostEqual(A.kstar_point(0.15, [0.2, 0.1, 0.0], A.KS), math.log10(800), places=12)
        res = run(noise={A.SF: 0.0, A.PGV: 0.0, "PGV_1600": 0.0, "PGV_6400": 0.0})
        k = res["kstar"]["PGV"]
        self.assertEqual(k["n_parents"], 12)
        self.assertAlmostEqual(10 ** k["log10"], 400 * 4 ** ((0.22 - 0.20) / (0.22 - 0.15)), places=6)
        self.assertEqual(res["kstar"]["FT"]["kstar"], "> 400")
        res = run(spec={A.PGV: 0.1}, det=dict(kstar_descriptive=True))
        self.assertEqual(res["kstar"]["PGV"]["kstar"], "<= 400")
        self.assertTrue(res["kstar"]["descriptive_only"] and res["qualifiers"]["kstar"]["descriptive_only"])

    def test_core_groups_cuts_pg400_and_r_ext(self):
        sizes = {"H-ID-S": 2, "H-ID-U": 2, "H-K": 2, "H-NEG": 2, "H-COST": 2, "H-KN": 2, "H-DYN": 2}
        d = A.make_design(make_manifest(sizes))
        prim, sub = A.family_keys(d, A.PRIMARY), set(d["subset"])
        g = A.core_groups(d, dict(DET, cut_items=[3, 5]))
        self.assertEqual(g[A.GREEDY], A.family_keys(d, {"H-ID", "H-K", "H-NEG", "H-COST", "H-DYN"}))  # H-KN cut
        self.assertEqual(g[A.GNN], A.family_keys(d, ("H-ID", "H-K")))
        self.assertEqual(g["SF-GPI@20261248"], prim)
        self.assertNotIn(f"{A.FT}@20261246", g)  # FT is not PG400: not core
        self.assertEqual((g[A.PGV], g["PGV_1600"]), (prim, sub))
        g = A.core_groups(d, dict(DET, pg400=A.FT, r_ext=True, cut_items=[7],
                                  kcurve_arms={"400": "PGV-RX_400", "1600": "PGV-RX_1600", "6400": "PGV-RX_6400"}))
        self.assertEqual({g[f"{A.FT}@{k}"] == prim for k in REPS}, {True})
        self.assertEqual((g["PGV-RX_400"], g[A.PGV]), (sub, prim))
        self.assertEqual(g[A.ORACLE], A.family_keys(d, {"H-ID", "H-K", "H-NEG", "H-COST", "H-KN"}))  # H-DYN cut
        self.assertNotIn("PGV_1600", g)

    def test_binary_view(self):
        cells = {A.SF: {r: {"a": {"nreg": [0.0, 0.0]}, "b": {"nreg": [0.0, 1.0]}, "c": {"nreg": [1.0, 1.0]}}
                        for r in REPS},
                 A.UVFA: {r: {"a": {"nreg": [1.0, 1.0]}, "b": {"nreg": [1.0, 1.0]}, "c": {"nreg": [0.0, 5e-10]}}
                          for r in REPS}}
        b = A.binary_view(cells, A.SF, A.UVFA, {"a", "b", "c"})
        self.assertEqual((b["n"], b["only_y"], b["only_x"]), (3, 1, 2))
        self.assertEqual(b["p"], ST.mcnemar_exact(1, 2))


class DecisionBranches(unittest.TestCase):
    """§7.2: every branch through analyze() on synthetic per-parent arrays."""

    def test_advance(self):
        res = run()
        self.assertEqual(res["decision"], A.ADVANCE, res["notes"])
        self.assertTrue(res["validity"]["p1_counts"] and res["validity"]["p2_counts"] and res["guard"]["G1"])
        self.assertEqual(res["failed_advance"], [])
        self.assertEqual(res["qualifiers"]["new_directions"]["wording"], "new preference directions")
        self.assertEqual(res["qualifiers"]["gpi"]["wording"], "GPI")
        self.assertFalse(res["qualifiers"]["attribution"]["holds"])
        self.assertEqual(res["qualifiers"]["naming"], A.NAMING)

    def test_inconclusive_defect_and_missing(self):
        res = run(defects=["eval used a stale ψ cache"])
        self.assertEqual(res["decision"], A.INC_DEFECT)
        res = run(drop=["UVFA@20261247"])
        self.assertEqual(res["decision"], A.INC_MISSING)
        self.assertEqual(res["missing"], {"UVFA@20261247": 26})
        self.assertFalse(res["P1"]["conditions"]["replicate_signs"])  # descriptive, on what exists
        res = run(drop=["GREEDY"])
        self.assertEqual(res["missing"], {"GREEDY": 30})
        self.assertEqual(res["decision"], A.INC_MISSING)
        res = run(drop=["SF-FT_400@20261246", "GNN-ENUM±"])  # SF-FT_400 not core; GNN core on H-ID and H-K
        self.assertEqual(res["missing"], {"GNN-ENUM±": 18})

    def test_stop_d(self):
        res = run(spec={A.ENUM: 0.05})
        self.assertEqual(res["decision"], A.STOP_D)
        self.assertTrue(res["validity"]["p1_counts"] and res["validity"]["p2_counts"])  # dominance comes first

    def test_stop_m_only_when_neither_counts(self):
        base = {A.SF: 0.45, A.UVFA: 0.45, A.AUX: 0.45, A.ENUM: 0.45, A.XGPI: 0.6}
        res = run(spec=dict(base, **{A.PGV: 0.2}))
        self.assertEqual(res["decision"], A.STOP_M)
        self.assertLess(res["P1"]["macro"]["hi"], 0.05)  # STOP-F would hold too: the order decides
        res = run(spec=dict(base, **{A.PGV: 0.45}))
        self.assertTrue(res["validity"]["p2_counts"])
        self.assertEqual(res["decision"], A.REVISE)
        self.assertIn("label-cost claim only (P2 counts, P1 does not)", res["failed_advance"])
        self.assertTrue(any(n.startswith("library ceiling >= 0.5") for n in res["failed_advance"]))

    def test_stop_f_variants(self):
        res = run(spec={A.SF: 0.3, A.UVFA: 0.3, A.PGV: 0.1, A.ENUM: 0.3})
        self.assertEqual(res["decision"], A.STOP_F)
        # significant but small P1: passes and counts, interval entirely below 0.05
        res = run(spec={A.SF: 0.3, A.UVFA: 0.335, A.PGV: 0.1, A.ENUM: 0.3}, noise={A.SF: 0.002, A.UVFA: 0.002})
        self.assertTrue(res["P1"]["passes"] and res["validity"]["p1_counts"])
        self.assertLess(res["P1"]["macro"]["hi"], 0.05)
        self.assertEqual(res["decision"], A.STOP_F)
        # P2 dropped at the pilot
        res = run(spec={A.SF: 0.3, A.UVFA: 0.3, A.ENUM: 0.3}, det=dict(p2_dropped=True, pg400=None))
        self.assertEqual(res["decision"], A.STOP_F)
        # P2 passes but is comparator-invalid on TEST
        res = run(spec={A.SF: 0.68, A.UVFA: 0.68, A.PGV: 0.70, A.ENUM: 0.68})
        self.assertTrue(res["P2"]["passes"])
        self.assertFalse(res["validity"]["p2_valid"])
        self.assertEqual(res["decision"], A.STOP_F)

    def test_revise_p1_only(self):
        res = run(spec={A.PGV: 0.05})
        self.assertEqual(res["decision"], A.REVISE)
        self.assertIn("P1 counts, P2 does not", res["failed_advance"])

    def test_revise_p1_m_only(self):
        res = run(spec={A.UVFA: {"H-ID": 0.20, "*": 0.45}})
        self.assertFalse(res["P1"]["passes"])
        self.assertTrue(res["P1"]["conjuncts"]["M"]["passes"])
        self.assertFalse(res["P1"]["conjuncts"]["ID"]["passes"])
        self.assertEqual(res["decision"], A.REVISE)
        self.assertTrue(any(n.startswith("P1-M passes, P1-ID does not, P2 counts") for n in res["failed_advance"]))

    def test_revise_sf_floor(self):
        res = run(spec={A.SF: {"H-NEG": 0.60, "*": 0.20}, A.UVFA: {"H-NEG": 0.80, "*": 0.35},
                        A.PGV: {"H-NEG": 0.60, "*": 0.22}, A.ENUM: {"H-NEG": 0.65, "*": 0.25}})
        self.assertTrue(res["validity"]["p1_counts"] and res["validity"]["p2_counts"] and res["guard"]["G1"])
        self.assertEqual(res["sf_floor"]["failed"], ["H-NEG"])
        self.assertEqual(res["decision"], A.REVISE)
        self.assertIn("SF floor fails: SF-GPI nreg > 0.50 on H-NEG", res["failed_advance"])

    def test_revise_g1_middle_zone(self):
        res = run(spec={A.SF: 0.35, A.UVFA: 0.48, A.PGV: 0.36, A.ENUM: lambda info, i, rep: 0.0 if i % 2 == 0 else 0.58},
                  noise={A.SF: 0.0, A.UVFA: 0.01, A.ENUM: 0.0})
        g = res["guard"]["D_G"]
        self.assertGreaterEqual(g["point"], 0.05)
        self.assertLessEqual(g["lo"], 0)
        self.assertFalse(res["guard"]["G1"] or res["guard"]["stop_d_condition"])
        self.assertEqual(res["decision"], A.REVISE)
        self.assertTrue(any(n.startswith("G1 fails in the middle zone") for n in res["failed_advance"]))

    def test_inconclusive_otherwise(self):
        res = run(spec={A.SF: 0.3, A.UVFA: lambda info, i, rep: 0.34 + (0.3 if i % 2 == 0 else -0.3), A.PGV: 0.1,
                        A.ENUM: 0.3}, noise={A.SF: 0.0, A.UVFA: 0.0})
        self.assertFalse(res["P1"]["passes"] or res["P2"]["passes"])
        self.assertGreaterEqual(res["P1"]["macro"]["hi"], 0.05)
        self.assertGreaterEqual(res["P1"]["hid"]["hi"], 0.05)
        self.assertEqual(res["decision"], A.INCONCLUSIVE)

    def test_decide_order_directly(self):
        f = dict(defects=[], missing=[], stop_d=False, p1_counts=True, p2_counts=True, g1=True, sf_floor=True,
                 xgpi=0.7, d1_macro_hi=0.01, d1_hid_hi=0.01)
        self.assertEqual(A.decide(f)[0], A.ADVANCE)  # ADVANCE precedes STOP-M and STOP-F
        self.assertEqual(A.decide(dict(f, defects=["x"], missing=["y"]))[0], A.INC_DEFECT)
        self.assertEqual(A.decide(dict(f, missing=["y"], stop_d=True))[0], A.INC_MISSING)
        self.assertEqual(A.decide(dict(f, sf_floor=False, p2_counts=False))[0], A.STOP_F)
        self.assertEqual(A.decide(dict(f, sf_floor=False, p2_counts=False, d1_macro_hi=0.06, d1_hid_hi=0.06))[0],
                         A.REVISE)
        self.assertEqual(A.decide(dict(f, p1_counts=False, p2_counts=False))[0], A.STOP_M)
        self.assertEqual(A.decide(dict(f, p1_counts=False, p2_counts=False, xgpi=0.5 - 5e-10))[0], A.STOP_M)
        self.assertEqual(A.decide(dict(f, p1_counts=False, p2_counts=False, xgpi=0.49, d1_macro_hi=0.05 - 5e-10,
                                       d1_hid_hi=0.2))[0], A.INCONCLUSIVE)


class CannotAdvance(unittest.TestCase):
    """B3: degenerate comparators or a degenerate SF never yield ADVANCE."""

    def test_copy_state_sf(self):
        res = run(spec={A.SF: IDEAL[A.GREEDY]}, noise={A.SF: 0.0})
        cells = make_cells(spec={A.SF: IDEAL[A.GREEDY]}, noise={A.SF: 0.0})
        sf, gr = A.arm_values(cells, A.SF), A.arm_values(cells, A.GREEDY)
        pos = A.family_keys(DESIGN, ("H-ID", "H-K"))
        self.assertTrue(all(sf[k] == gr[k] == 1.0 for k in pos))  # exactly 1 on H-ID and H-K
        for k in set(sf) - pos:
            self.assertAlmostEqual(sf[k], gr[k], places=12)
        self.assertFalse(res["sf_floor"]["holds"])
        self.assertNotEqual(res["decision"], A.ADVANCE)
        f = dict(defects=[], missing=[], stop_d=False, p1_counts=True, p2_counts=True, g1=True, sf_floor=False,
                 xgpi=0.3, d1_macro_hi=0.2, d1_hid_hi=0.2)
        self.assertEqual(A.decide(f)[0], A.REVISE)  # even if both tests counted

    def test_constant_uvfa(self):
        res = run(spec={A.UVFA: 0.9}, noise={A.UVFA: 0.0})
        self.assertTrue(res["P1"]["passes"])
        self.assertFalse(res["validity"]["p1_valid"] or res["validity"]["p1_counts"])
        self.assertAlmostEqual(res["validity"]["uvfa_test_hid"], 0.9, places=12)
        self.assertEqual(res["decision"], A.REVISE)
        self.assertTrue(any(n.startswith("P1 passed but does not count") for n in res["failed_advance"]))

    def test_pg400_collapsed_on_test(self):
        res = run(spec={A.PGV: 0.8}, noise={A.PGV: 0.0})
        self.assertTrue(res["P2"]["passes"])
        self.assertFalse(res["validity"]["p2_valid"] or res["validity"]["p2_counts"])
        self.assertEqual(res["decision"], A.REVISE)
        self.assertTrue(any(n.startswith("P2 passed but does not count") for n in res["failed_advance"]))

    def test_pg400_is_ft_and_collapses(self):
        res = run(spec={A.FT: 0.8}, det=dict(pg400=A.FT), noise={A.FT: 0.0})
        self.assertAlmostEqual(res["validity"]["pg400_test_primary_macro"], 0.8, places=12)
        self.assertNotEqual(res["decision"], A.ADVANCE)
        self.assertIn("D2 (non-selected candidate) = SF-GPI - PGV_400", res["contrasts"])

    def test_p1_comparator_invalid_at_pilot(self):
        res = run(det=dict(p1_comparator_invalid=True))
        self.assertEqual(res["decision"], A.REVISE)
        self.assertIn("P1 comparator-invalid at the pilot: P1 descriptive, ADVANCE unavailable", res["notes"])
        res = run(det=dict(p1_comparator_invalid=True, p2_dropped=True, pg400=None))
        self.assertNotIn(res["decision"], (A.ADVANCE, A.REVISE))
        self.assertEqual(res["decision"], A.INCONCLUSIVE)


class QualifiersAndValidity(unittest.TestCase):
    def test_attribution_with_tolerance(self):
        exact = {A.SF: 0.0, A.AUX: 0.0}
        self.assertTrue(run(spec={A.AUX: 0.21})["qualifiers"]["attribution"]["holds"])
        self.assertFalse(run(spec={A.AUX: 0.25})["qualifiers"]["attribution"]["holds"])
        q = run(spec={A.SF: 0.25, A.AUX: 0.27 + 5e-10}, noise=exact)["qualifiers"]["attribution"]
        self.assertGreater(q["uvfa_aux"] - q["sf_gpi"], 0.02)  # above the margin, inside the 1e-9 tolerance
        self.assertTrue(q["holds"])
        self.assertFalse(run(spec={A.SF: 0.25, A.AUX: 0.27 + 2e-9}, noise=exact)["qualifiers"]["attribution"]["holds"])

    def test_gpi_new_directions_guard(self):
        q = run(spec={A.NOGPI: 0.205}, det=dict(q9_failed_no_bug=True))
        self.assertEqual(q["qualifiers"]["gpi"]["wording"], "goal-conditioned endpoint predictor read out linearly")
        self.assertEqual(q["qualifiers"]["guard_quality"]["wording"], A.GUARD_NOTE)
        self.assertEqual(q["guard"]["statement_suffix"], A.GUARD_NOTE)
        q = run(spec={A.UVFA: {"H-ID-U": 0.21, "*": 0.35}})["qualifiers"]["new_directions"]
        self.assertEqual(q["wording"], "re-weighting of seen fact patterns")
        q = run(det=dict(p2_dropped=True, pg400=None))["qualifiers"]["new_directions"]
        self.assertFalse(q["holds"])

    def test_validity_and_floor_tolerance(self):
        res = run(spec={A.UVFA: {"H-ID": 0.5 + 5e-10, "*": 0.35}}, noise={A.UVFA: 0.0})
        self.assertTrue(res["validity"]["p1_valid"])
        res = run(spec={A.UVFA: {"H-ID": 0.5 + 3e-9, "*": 0.35}}, noise={A.UVFA: 0.0})
        self.assertFalse(res["validity"]["p1_valid"])
        res = run(spec={A.SF: {"H-K": 0.5 + 5e-10, "*": 0.2}, A.UVFA: {"H-K": 0.7, "*": 0.35}},
                  noise={A.SF: 0.0, A.UVFA: 0.0})
        self.assertTrue(res["sf_floor"]["holds"])

    def test_strata_k_conflict_and_extras(self):
        cells = make_cells()
        for rep in cells[A.SF].values():
            for k, c in rep.items():
                c["loop"] = [0.0, 1.0, 0.0, 0.0]
        res = A.analyze(DESIGN, cells, DET, resamples=R)
        self.assertEqual(res["extras"][A.SF]["loop"]["primary_macro"], 0.25)
        conf = res["strata_k_conflict"]["D1"]["conflict"]["H-K"]
        self.assertEqual((conf["True"]["n"], conf["False"]["n"]), (2, 6))
        self.assertEqual(sum(g["n"] for g in res["strata_k_conflict"]["D1"]["K"]["H-NEG"].values()), 8)
        json.dumps(A._jsonable(res))
        self.assertIn("RCX-E05 confirmatory analysis: ADVANCE", A.render(res))


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=rcx-test", "-c", "user.email=rcx-test@example.invalid",
                    *args], check=True, capture_output=True)


@unittest.skipIf(shutil.which("git") is None, "git not available")
class Custody(unittest.TestCase):
    """§2.2 / §6.4: refusals write nothing; outputs are create-only; the sha must be in the COMMITTED DECISIONS.md."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.base = Path(cls.tmp.name)
        cls.repo = cls.base / "repo"
        cls.repo.mkdir()
        cls.mtext = json.dumps(MANIFEST, indent=1) + "\n"
        cls.msha = hashlib.sha256(cls.mtext.encode()).hexdigest()
        cls.digest = "d" * 64
        cls.m99 = json.dumps(dict(MANIFEST, attempt=99), indent=1) + "\n"
        cls.sha99 = hashlib.sha256(cls.m99.encode()).hexdigest()
        _git(cls.repo, "init", "-q")
        (cls.repo / "DECISIONS.md").write_text(f"# DECISIONS\n- TEST_MANIFEST sha256 `{cls.msha}`\n- attempt 99 "
                                               f"`{cls.sha99}`\n", encoding="utf-8")
        _git(cls.repo, "add", "DECISIONS.md")
        _git(cls.repo, "commit", "-q", "-m", "test manifest")
        cls.pilot = cls.base / "PILOT_REPORT.json"
        cls.pilot.write_text(json.dumps({"determinations": DET}), encoding="utf-8")
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def attempt(self, manifest_text=None, oracle=None, drop=(), mutate=None):
        Custody.n += 1
        adir = self.base / "confirm" / f"attempt-{Custody.n}"
        (adir / "evals").mkdir(parents=True)
        (adir / "TEST_MANIFEST.json").write_text(manifest_text or self.mtext, encoding="utf-8", newline="\n")
        rep = dict(status="PASS", manifest_sha256=self.msha, code_digest=self.digest, checks={}) if oracle is None \
            else oracle
        if rep != "absent":
            (adir / "ORACLE_TEST_REPORT.json").write_text(json.dumps(rep), encoding="utf-8")
        cells, i = make_cells(), 0
        for arm, by_rep in cells.items():
            for r, body in by_rep.items():
                if arm in drop or f"{arm}@{r}" in drop:
                    continue
                d = dict(manifest_sha256=self.msha, code_digest=self.digest, arm=arm, replicate=r,
                         cells={k: c["nreg"] for k, c in body.items()})
                if mutate:
                    mutate(d)
                (adir / "evals" / f"eval_{i:03d}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                i += 1
        return Custody.n, adir

    def main(self, n, sha=None, extra=()):
        argv = ["--attempt", str(n), "--manifest-sha", sha or self.msha, "--code-digest", self.digest,
                "--confirm-dir", str(self.base / "confirm"), "--pilot-report", str(self.pilot),
                "--repo-root", str(self.repo), *extra]
        with mock.patch.object(A, "RESAMPLES", 200), contextlib.redirect_stdout(io.StringIO()):
            return A.main(argv)

    def refused(self, n, adir, **kw):
        with self.assertRaises(SystemExit) as cm:
            self.main(n, **kw)
        self.assertIn("refusing", str(cm.exception))
        self.assertEqual([p for p in adir.iterdir() if p.name.startswith("ANALYSIS_E05")], [])
        return str(cm.exception)

    def test_committed_text_is_head_not_working_copy(self):
        dec = self.repo / "DECISIONS.md"
        orig = dec.read_text(encoding="utf-8")
        try:
            dec.write_text(orig + "- working copy only `" + "e" * 64 + "`\n", encoding="utf-8")
            txt = A.committed_text(self.repo)
            self.assertIn(self.msha, txt)
            self.assertNotIn("e" * 64, txt)
        finally:
            dec.write_text(orig, encoding="utf-8")
        self.assertIsNone(A.committed_text(self.base))  # not a git repository

    def test_run_once_create_only(self):
        n, adir = self.attempt()
        res = self.main(n)
        self.assertEqual(res["decision"], A.ADVANCE)
        self.assertFalse(res["prereg_resamples"])
        outs = {p.name: p.read_bytes() for p in adir.iterdir() if p.name.startswith("ANALYSIS_E05")}
        self.assertEqual(set(outs), set(A.OUTPUTS))
        j = json.loads(outs["ANALYSIS_E05.json"])
        self.assertEqual((j["decision"], j["custody"]["manifest_sha256"]), (A.ADVANCE, self.msha))
        self.assertEqual(len(j["custody"]["eval_files"]), len(list((adir / "evals").glob("*.json"))))
        with self.assertRaises(SystemExit):
            self.main(n)
        self.assertEqual({p.name: p.read_bytes() for p in adir.iterdir() if p.name.startswith("ANALYSIS_E05")}, outs)

    def test_refuses_without_oracle_pass(self):
        n, adir = self.attempt(oracle=dict(status="VOID", manifest_sha256=self.msha, code_digest=self.digest))
        self.assertIn("not PASS", self.refused(n, adir))
        n, adir = self.attempt(oracle="absent")
        self.refused(n, adir)
        n, adir = self.attempt(oracle=dict(status="PASS", manifest_sha256=self.msha, code_digest="x" * 64))
        self.refused(n, adir)

    def test_refuses_manifest_or_digest_mismatch(self):
        z = "0" * 64  # everything names z, but the manifest file's sha is not z
        n, adir = self.attempt(oracle=dict(status="PASS", manifest_sha256=z, code_digest=self.digest),
                               mutate=lambda d: d.update(manifest_sha256=z))
        self.assertIn("differs from --manifest-sha", self.refused(n, adir, sha=z))
        n, adir = self.attempt()
        self.refused(n, adir, sha="not-a-sha")
        n, adir = self.attempt(mutate=lambda d: d.update(manifest_sha256="1" * 64) if d["arm"] == A.ENUM else None)
        self.assertIn("names manifest", self.refused(n, adir))
        n, adir = self.attempt(mutate=lambda d: d.update(code_digest="2" * 64) if d["arm"] == A.UVFA else None)
        self.assertIn("names code digest", self.refused(n, adir))

    def test_refuses_sha_absent_from_committed_decisions(self):
        other = json.dumps(dict(MANIFEST, note="redraw"), indent=1) + "\n"
        osha = hashlib.sha256(other.encode()).hexdigest()
        n, adir = self.attempt(manifest_text=other, oracle=dict(status="PASS", manifest_sha256=osha,
                                                                code_digest=self.digest),
                               mutate=lambda d: d.update(manifest_sha256=osha))
        dec = self.repo / "DECISIONS.md"
        orig = dec.read_text(encoding="utf-8")
        try:
            dec.write_text(orig + f"- uncommitted `{osha}`\n", encoding="utf-8")
            self.assertIn("COMMITTED DECISIONS.md", self.refused(n, adir, sha=osha))
        finally:
            dec.write_text(orig, encoding="utf-8")

    def test_refuses_other_attempts_manifest(self):
        s = self.sha99
        n, adir = self.attempt(manifest_text=self.m99, oracle=dict(status="PASS", manifest_sha256=s,
                                                                   code_digest=self.digest),
                               mutate=lambda d: d.update(manifest_sha256=s))
        self.assertIn("is attempt 99", self.refused(n, adir, sha=s))

    def test_refuses_malformed_cells(self):
        def short(d):
            if d["arm"] == A.SF:
                k = next(iter(d["cells"]))
                d["cells"][k] = d["cells"][k][:-1]
        n, adir = self.attempt(mutate=short)
        self.assertIn("one per start", self.refused(n, adir))
        n, adir = self.attempt(mutate=lambda d: d.update(replicate=None) if d["arm"] == A.UVFA else None)
        self.refused(n, adir)

    def test_missing_must_be_declared_exactly(self):
        n, adir = self.attempt(drop=("UVFA@20261247",))
        self.assertIn("absent but undeclared ['UVFA@20261247']", self.refused(n, adir))
        res = self.main(n, extra=("--missing", "UVFA@20261247"))
        self.assertEqual(res["decision"], A.INC_MISSING)
        n, adir = self.attempt()
        self.assertIn("declared but complete ['GREEDY']", self.refused(n, adir, extra=("--missing", "GREEDY")))
        res = self.main(n, extra=("--defect", "ψ cache built from a stale checkpoint"))
        self.assertEqual(res["decision"], A.INC_DEFECT)


if __name__ == "__main__":
    unittest.main()
