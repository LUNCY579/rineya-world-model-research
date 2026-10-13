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


BANK = {"regret1": [0.0, 2.0], "top1": [1.0, 0.0]}  # a two-state one-step bank result


def make_cells(design=DESIGN, spec=None, noise=None, amp=0.03):
    """{arm: {replicate: {key: {"nreg": [x] * starts, + BANK unless the arm is bank-exempt}}}}: per-parent
    x = base + amp * u, u ~ U(-1, 1) seeded by (arm, key, replicate), clipped to [0, 1]; replicated arms get 3
    replicates (SF-shuffled / SF-FT_400: 20261246)."""
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
                    if arm not in A.BANK_EXEMPT:
                        slot[k].update({f: list(v) for f, v in BANK.items()})
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
        kept = A.family_keys(d, {"H-ID", "H-K", "H-NEG", "H-COST", "H-DYN"})  # H-KN cut
        self.assertEqual(g[A.GREEDY], kept)
        self.assertEqual(g[A.GNN], A.family_keys(d, ("H-ID", "H-K")))
        self.assertEqual((g["SF-GPI@20261248"], g["SF-shuffled@20261246"]), (kept, kept))
        self.assertNotIn("SF-shuffled@20261247", g)
        self.assertNotIn(f"{A.FT}@20261246", g)  # FT is not PG400: not core
        self.assertEqual((g[A.PGV], g["PGV_1600"]), (prim, sub))
        self.assertEqual((g["bank:UVFA@20261247"], g["bank:PGV_400"], g["bank:STOP"]), (kept, prim, kept))
        self.assertFalse({"bank:RAND", "bank:BEST-SINGLE(L)", "bank:ENUM-H*±", "bank:GNN-ENUM±"} & set(g))
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
        self.assertEqual(res["missing"], {"UVFA@20261247": 30, "bank:UVFA@20261247": 30})  # every retained family
        self.assertFalse(res["P1"]["conditions"]["replicate_signs"])  # descriptive, on what exists
        res = run(drop=["GREEDY"])
        self.assertEqual(res["missing"], {"GREEDY": 30, "bank:GREEDY": 30})
        self.assertEqual(res["decision"], A.INC_MISSING)
        res = run(drop=["SF-FT_400@20261246", "GNN-ENUM±"])  # SF-FT_400 not core; GNN core on H-ID and H-K, no bank
        self.assertEqual(res["missing"], {"GNN-ENUM±": 18})

    def test_missing_bank_only(self):
        cells = make_cells()
        for c in cells[A.SF][REPS[0]].values():
            del c["regret1"]
        for c in cells[A.RAND][None].values():
            self.assertNotIn("regret1", c)  # RAND is bank-exempt
        key = DESIGN["strata"]["H-COST"][0]
        del cells[A.PGV][None][key]["regret1"]  # PGV_400's bank is protected on the primary parents only
        res = A.analyze(DESIGN, cells, DET, resamples=R)
        self.assertEqual((res["decision"], res["missing"]), (A.INC_MISSING, {"bank:SF-GPI@20261246": 30}))

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


NATIVE_CFG = {"PGV": "w64s500", "FT": "s100"}
NATIVE_PILOT = dict(n=7, p1_comparator_invalid=False, p2_dropped=False, pg400="PGV", kstar="decision-driving",
                    kcurve_recipe="original", G_P1={"Q9": {"ok": True}}, G_P5={"cuts": []},
                    G_P3={"grid": {k: {"cfg": v} for k, v in NATIVE_CFG.items()}},
                    dev_nreg={"PG400=PGV": {"DEV-OOD:H-K": 0.3, "DEV-OOD:H-NEG": 0.35}})


def native_manifest(attempt, digest, manifest=MANIFEST):
    """run_e05.draw_test's layout: parents by family, pgv_subset, attempt, code_digest_t3."""
    fams = {}
    for p in manifest["parents"]:
        fams.setdefault(p["family"], []).append(p)
    return dict(experiment="RCX-E05", kind="test", attempt=attempt, code_digest_t3=digest, parents=fams,
                pgv_subset=manifest["subset"])


def native_tag(arm, rep):
    """run_e05's (arm, tag) of an analysis arm (None: run_e05 does not produce the arm)."""
    if arm in A.MODEL_OF:
        return arm, f"{A.MODEL_OF[arm]}|s{rep}"
    if arm == A.PGV:
        return "PGV", f"{NATIVE_CFG['PGV']}|k400"
    if arm in ("PGV_1600", "PGV_6400"):
        k = arm.split("_")[1]
        return "PGV", f"k{k}|k{k}"
    if arm == A.FT:
        return "FT", f"{NATIVE_CFG['FT']}|s{rep}|k400"
    if arm == A.SFFT:
        return None
    return arm, "L" if arm in (A.XGPI, A.BEST) else "-"


def write_native(adir, cells, attempt, digest, mutate=None, skip=()):
    """cells -> run_e05's keyed logs: cells.jsonl (one record per start; ENUM-MIS± with a diag) and banks.jsonl (the
    cells' BANK rows; RAND / BEST-SINGLE(L) records skipped, none for ENUM-H*± / GNN-ENUM±, as run_e05 writes them).
    mutate(rec) may edit a cell record; skip = {(arm, rep, key, s0)}."""
    split = f"test:{attempt}"
    with open(adir / "cells.jsonl", "w", encoding="utf-8", newline="\n") as fc, \
            open(adir / "banks.jsonl", "w", encoding="utf-8", newline="\n") as fb:
        for arm, by_rep in cells.items():
            for rep, body in by_rep.items():
                at = native_tag(arm, rep)
                if at is None:
                    continue
                for key, c in body.items():
                    for s0, x in zip(DESIGN["parents"][key]["starts"], c["nreg"]):
                        if (arm, rep, key, s0) in skip:
                            continue
                        rec = dict(key=f"cell|{split}|{at[0]}|{at[1]}|{key}|{s0}", split=split,
                                   family=DESIGN["parents"][key]["family"], arm=at[0], tag=at[1], parent=key, s0=s0,
                                   nreg=x, J=0.0, T=3, loop=s0 == 1, capped=False, final=0, dt=0.01, digest=digest)
                        if arm == A.ENUM:
                            rec["diag"] = dict(decisions=2, all_tied=1, targets=5)
                        if mutate:
                            mutate(rec)
                        fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if arm in (A.ENUM_H, A.GNN):
                        continue
                    b = dict(key=f"bank|{split}|{at[0]}|{at[1]}|{key}", split=split, arm=at[0], tag=at[1],
                             parent=key, digest=digest)
                    if arm in (A.RAND, A.BEST):
                        b["skipped"] = "start-dependent arm"
                    else:
                        b.update(c=0.25, rows=[dict(s=i, vstar=1.0, regret1=r, top1=bool(t))
                                               for i, (r, t) in enumerate(zip(c["regret1"], c["top1"]))])
                    fb.write(json.dumps(b, ensure_ascii=False) + "\n")
        fc.write('{"key": "cell|torn')  # a torn last line (crash mid-append) is ignored, as run_e05 does


class NativeLayout(unittest.TestCase):
    """run_e05.py's own TEST_MANIFEST / PILOT_REPORT / cells.jsonl layout."""

    def test_manifest_by_family_equals_list_form(self):
        d = A.make_design(native_manifest(1, "d" * 64))
        self.assertEqual((d["strata"], d["subset"], d["parents"]), (DESIGN["strata"], DESIGN["subset"],
                                                                   DESIGN["parents"]))
        bad = native_manifest(1, "d" * 64)
        bad["parents"]["H-K"] = bad["parents"]["H-K"] + [dict(bad["parents"]["H-NEG"][0])]
        with self.assertRaises(ValueError):
            A.make_design(bad)
        with self.assertRaises(ValueError):
            A.make_design(dict(MANIFEST, pgv_subset=MANIFEST["subset"][:3]))

    def test_native_determinations(self):
        det = A.pilot_determinations(NATIVE_PILOT)
        self.assertEqual({k: det[k] for k in A.DET_KEYS}, {k: DET[k] for k in A.DET_KEYS})
        self.assertEqual((det["cut_items"], det["native_cfg"], A.kcurve_arms(det)[1600]), ([], NATIVE_CFG, "PGV_1600"))
        self.assertEqual(det["pg400_dev_ood"], NATIVE_PILOT["dev_nreg"]["PG400=PGV"])
        det = A.pilot_determinations(dict(NATIVE_PILOT, pg400="FT", kstar="descriptive only", kcurve_recipe="R_ext",
                                          G_P1={"Q9": {"ok": False}}, G_P5={"cuts": ["H-DYN", "SF-FT_400", "H-KN"]}))
        self.assertEqual((det["pg400"], det["kstar_descriptive"], det["r_ext"], det["q9_failed_no_bug"]),
                         (A.FT, True, True, True))
        self.assertEqual(det["cut_items"], [2, 3, 7])
        self.assertEqual(A.kcurve_arms(det), {400: "PGV-RX_400", 1600: "PGV-RX_1600", 6400: "PGV-RX_6400"})
        det = A.pilot_determinations(dict(NATIVE_PILOT, pg400=None, p2_dropped=True))
        self.assertIsNone(det["pg400"])
        for bad in (dict(NATIVE_PILOT, pg400=None), dict(NATIVE_PILOT, G_P5={"cuts": ["H-SPAN"]}),
                    dict(NATIVE_PILOT, pg400="PGV_400"), dict(NATIVE_PILOT, kcurve_recipe="rext"),
                    {k: v for k, v in NATIVE_PILOT.items() if k != "G_P1"}):
            with self.assertRaises(ValueError):
                A.pilot_determinations(bad)

    def test_native_arm(self):
        cfg = NATIVE_CFG
        cases = {("SF-GPI", "SF|s20261247"): (A.SF, 20261247), ("SF-L", "SF|s20261246"): (A.SFL, 20261246),
                 ("UVFA+aux", "UVFA+aux|s20261248"): (A.AUX, 20261248),
                 ("SF-shuffled", "SF-shuffled|s20261246"): (A.SHUF, 20261246),
                 ("PGV", "w64s500|k400"): (A.PGV, None), ("PGV", "k1600|k1600"): ("PGV_1600", None),
                 ("PGV", "rext400|k400"): ("PGV-RX_400", None), ("PGV", "rext6400|k6400"): ("PGV-RX_6400", None),
                 ("PGV", "w256s2000|k400"): ("PGV[w256s2000|k400]", None),
                 ("FT", "s100|s20261248|k400"): (A.FT, 20261248), ("FT", "s400|s20261246|k400"): ("FT[s400]", 20261246),
                 ("GREEDY", "-"): (A.GREEDY, None), ("EXACT-GPI(L)", "L+base"): (A.XGPI, None),
                 ("GNN-ENUM±", "-"): (A.GNN, None)}
        for (arm, tag), want in cases.items():
            self.assertEqual(A.native_arm(arm, tag, cfg), want, (arm, tag))
        for arm, tag in (("SF-GPI", "UVFA|s20261246"), ("UVFA", "UVFA"), ("PGV", "w64s500"), ("FT", "s100|k400"),
                         ("SF@x", "-"), ("STOP", "")):
            with self.assertRaises(ValueError):
                A.native_arm(arm, tag, cfg)


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
                         cells={k: c["nreg"] if arm in A.BANK_EXEMPT else c for k, c in body.items()})  # bare = nreg
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
            if d["arm"] in (A.SF, A.RAND):
                k = next(iter(d["cells"]))
                c = d["cells"][k]
                d["cells"][k] = c[:-1] if isinstance(c, list) else dict(c, nreg=c["nreg"][:-1])
        n, adir = self.attempt(mutate=short)
        self.assertIn("one per start", self.refused(n, adir))
        n, adir = self.attempt(mutate=lambda d: d["cells"][next(iter(d["cells"]))].update(nreg=[1.5] * N_STARTS)
                               if d["arm"] == A.UVFA else None)
        self.assertIn("in [0, 1]", self.refused(n, adir))
        n, adir = self.attempt(mutate=lambda d: d.update(replicate=None) if d["arm"] == A.UVFA else None)
        self.refused(n, adir)

    def test_missing_must_be_declared_exactly(self):
        n, adir = self.attempt(drop=("UVFA@20261247",))
        self.assertIn("absent but undeclared ['UVFA@20261247', 'bank:UVFA@20261247']", self.refused(n, adir))
        self.assertIn("absent but undeclared ['bank:UVFA@20261247']",
                      self.refused(n, adir, extra=("--missing", "UVFA@20261247")))
        res = self.main(n, extra=("--missing", "UVFA@20261247,bank:UVFA@20261247"))
        self.assertEqual(res["decision"], A.INC_MISSING)
        n, adir = self.attempt()
        self.assertIn("declared but complete ['GREEDY']", self.refused(n, adir, extra=("--missing", "GREEDY")))
        res = self.main(n, extra=("--defect", "ψ cache built from a stale checkpoint"))
        self.assertEqual(res["decision"], A.INC_DEFECT)


@unittest.skipIf(shutil.which("git") is None, "git not available")
class NativeCustody(unittest.TestCase):
    """The analysis on run_e05.py's own layout (manifest by family, pilot-report fields, cells.jsonl / banks.jsonl)
    equals the analysis of the same cells passed directly; its custody refusals write nothing."""

    ATTEMPTS = range(1, 9)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.base = Path(cls.tmp.name)
        cls.repo = cls.base / "repo"
        cls.repo.mkdir()
        cls.digest = "c" * 64
        cls.mtext = {k: json.dumps(native_manifest(k, "f" * 64 if k == 8 else cls.digest), indent=1, sort_keys=True)
                     + "\n" for k in cls.ATTEMPTS}  # attempt 8: drawn under another code digest
        cls.msha = {k: hashlib.sha256(t.encode()).hexdigest() for k, t in cls.mtext.items()}
        _git(cls.repo, "init", "-q")
        (cls.repo / "DECISIONS.md").write_text("# DECISIONS\n" + "".join(f"- attempt {k} TEST_MANIFEST `{s}`\n" for k, s
                                                                         in cls.msha.items()), encoding="utf-8")
        _git(cls.repo, "add", "DECISIONS.md")
        _git(cls.repo, "commit", "-q", "-m", "test manifests")
        cls.pilot = cls.base / "PILOT_REPORT.json"
        cls.pilot.write_text(json.dumps(NATIVE_PILOT, ensure_ascii=False), encoding="utf-8")
        cls.cells = make_cells()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def attempt(self, k, **kw):
        adir = self.base / "confirm" / f"attempt-{k}"
        adir.mkdir(parents=True)
        (adir / "TEST_MANIFEST.json").write_text(self.mtext[k], encoding="utf-8", newline="\n")
        (adir / "ORACLE_TEST_REPORT.json").write_text(json.dumps(dict(
            attempt=k, status="PASS", manifest_sha256=self.msha[k], code_digest=self.digest, checks={})),
            encoding="utf-8")
        write_native(adir, self.cells, k, self.digest, **kw)
        return adir

    def main(self, k, extra=()):
        argv = ["--attempt", str(k), "--manifest-sha", self.msha[k], "--code-digest", self.digest,
                "--confirm-dir", str(self.base / "confirm"), "--pilot-report", str(self.pilot),
                "--repo-root", str(self.repo), *extra]
        with mock.patch.object(A, "RESAMPLES", 200), contextlib.redirect_stdout(io.StringIO()):
            return A.main(argv)

    def refused(self, k, adir, extra=()):
        with self.assertRaises(SystemExit) as cm:
            self.main(k, extra)
        self.assertIn("refusing", str(cm.exception))
        self.assertEqual([p for p in adir.iterdir() if p.name.startswith("ANALYSIS_E05")], [])
        return str(cm.exception)

    def test_native_equals_direct(self):
        self.attempt(1)
        res = self.main(1)
        cells = {a: v for a, v in self.cells.items() if a != A.SFFT}
        ref = A.analyze(DESIGN, cells, A.pilot_determinations(NATIVE_PILOT), resamples=200)
        self.assertEqual(res["decision"], A.ADVANCE)
        for fld in ("decision", "P1", "P2", "guard", "validity", "sf_floor", "raw_p", "holm_adjusted", "kstar",
                    "per_replicate", "arm_means"):
            self.assertEqual(res[fld], ref[fld], fld)
        self.assertEqual(res["contrasts"]["D1 = UVFA - SF-GPI"], ref["contrasts"]["D1 = UVFA - SF-GPI"])
        cu = res["custody"]
        self.assertEqual((cu["eval_source"], cu["pilot_determinations_source"]),
                         ("cells.jsonl (run_e05 native)", "run_e05 pilot-report fields"))
        banked = sum(len(body) for a, by in cells.items() if a not in A.BANK_EXEMPT for body in by.values())
        self.assertEqual((cu["eval_files"][1]["attached"], cu["eval_files"][1]["skipped"]), (banked, 2 * 30))
        ex = res["extras"][A.SF]
        self.assertEqual((ex["regret1"]["primary_macro"], ex["top1"]["primary_macro"], ex["loop"]["primary_macro"]),
                         (1.0, 0.5, 0.25))
        self.assertEqual({f: t["share"] for f, t in res["enum_all_tied"][A.ENUM].items()},
                         {f: 0.5 for f in ("H-ID", "H-K", "H-NEG", "H-COST")})
        self.assertNotIn("decisions", res["extras"].get(A.ENUM, {}))
        self.assertEqual(res["validity"]["pg400_dev_ood"], NATIVE_PILOT["dev_nreg"]["PG400=PGV"])

    def test_incomplete_parent_is_missing(self):
        key = DESIGN["strata"]["H-K"][0]
        adir = self.attempt(2, skip={(A.UVFA, REPS[1], key, 2)})
        self.assertIn("absent but undeclared ['UVFA@20261247', 'bank:UVFA@20261247']", self.refused(2, adir))
        res = self.main(2, ("--missing", "UVFA@20261247,bank:UVFA@20261247"))
        self.assertEqual((res["decision"], res["missing"]), (A.INC_MISSING, {"UVFA@20261247": 1,
                                                                             "bank:UVFA@20261247": 1}))
        self.assertEqual(res["custody"]["eval_files"][0]["incomplete"], {"UVFA@20261247": 1})

    def test_refusals(self):
        def digest(rec):
            if rec["arm"] == A.UVFA and rec["s0"] == 3:
                rec["digest"] = "e" * 64
        self.assertIn("names code digest", self.refused(3, self.attempt(3, mutate=digest)))
        self.assertIn("has split", self.refused(4, self.attempt(4, mutate=lambda r: r.update(split="pilot")
                                                                 if r["arm"] == A.GREEDY else None)))
        self.assertIn("is not a manifest start", self.refused(5, self.attempt(5, mutate=lambda r: r.update(s0=99)
                                                                             if r["arm"] == A.STOP else None)))
        self.assertIn("is not 'SF|s<seed>'", self.refused(6, self.attempt(6, mutate=lambda r: r.update(
            tag="UVFA|s20261246") if r["arm"] == A.SFL else None)))
        adir = self.attempt(7)
        (adir / "evals").mkdir()
        (adir / "evals" / "x.json").write_text("{}", encoding="utf-8")
        self.assertIn("ambiguous", self.refused(7, adir))
        self.assertIn("drawn under code digest", self.refused(8, self.attempt(8)))


if __name__ == "__main__":
    unittest.main()
