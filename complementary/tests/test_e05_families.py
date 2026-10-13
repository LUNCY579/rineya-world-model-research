"""Tests for RCX-E05 families, keys, splits, renaming, exclusions and seeds (experiments/RCX-E05/families_e05.py;
PREREG_E05_DRAFT.md §4 and App. A).

n = 3-5 spaces only, literal seeds 0-9, synthetic exclusion sets (the real EXCLUSIONS.json is not loaded); BWM's pinned
blocks code is loaded (read-only) for the exclusion adaptor. Never reported as results.
Run: python -B -m unittest tests.test_e05_families -v   (CPU, one thread; a few seconds)
"""
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import unittest
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import excl as EX  # noqa: E402
from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("families_e05_ut", os.path.join(ROOT, "experiments", "RCX-E05",
                                                                              "families_e05.py"))
FAM = importlib.util.module_from_spec(_spec)
sys.modules["families_e05_ut"] = FAM
_spec.loader.exec_module(FAM)

SMALL_PILOT = dict(train=12, dev_id_s=3, dev_id_u=3, dev_ood=2, bank=32)
SMALL_TEST = dict(hid_s_max=4, bank=32, **{"H-COST": 2, "H-SPAN": 2, "H-KN": 2, "H-CONJ": 2, "H-DYN": 2})


def wvec(n, items):
    """Weight vector over facts(n) from {fact: weight}."""
    w, fi = np.zeros(n * (n + 1), dtype=np.int64), SO.fact_index(n)
    for f, x in items.items():
        w[fi[f]] = x
    return w


def synth_ex(n, invariants=(), signatures=(), sig_ns=None):
    """A synthetic `ex` with the fields excl.row_exclusion_reason / exclusion_reason read."""
    return {"invariants": frozenset(invariants), "signatures": frozenset(signatures),
            "raw": {"test_signature_block_counts": [n] if sig_ns is None else list(sig_ns)},
            "selection_signatures": frozenset(), "selection_invariants": frozenset()}


def draw_small(n=5, excl=None):
    sp = SP.get_space(n)
    reg = FAM.draw_pilot(sp, excl=excl, sizes=SMALL_PILOT, rng=random.Random(0))
    pilot = [p.to_json() for p in reg.all_parents()]
    FAM.draw_test(sp, reg, attempt=1, n_id=8, n_k=3, n_neg=3, excl=excl, sizes=SMALL_TEST, rng=random.Random(1))
    return sp, reg, pilot


class TestSeeds(unittest.TestCase):
    def test_derive(self):
        for seed, tag in ((0, "x"), (20261242, "test:1"), (20261250, "boot:H-ID-S")):
            want = int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")
            self.assertEqual(FAM.derive(seed, tag), want)
        self.assertNotEqual(FAM.derive(20261242, "test:1"), FAM.derive(20261242, "test:2"))
        a, b = FAM.stream(1, "t"), FAM.stream(1, "t")
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])
        self.assertEqual(FAM.SEEDS["replicates"], (20261246, 20261247, 20261248))

    def test_draw_checked(self):
        pool = list(range(100, 160))
        big, _ = FAM.draw_checked(pool, 40, random.Random(2))
        small, _ = FAM.draw_checked(pool, 10, random.Random(2))
        self.assertEqual(big[:10], small)  # nested prefixes (400 < 1600 < 6400)
        self.assertEqual(len(set(big)), 40)
        self.assertTrue(set(big) <= set(pool))
        out, cnt = FAM.draw_checked(pool, 60, random.Random(3), lambda i: "odd" if i % 2 else None)
        self.assertEqual((sorted(out), cnt), (pool[::2], Counter(odd=30)))


class TestKeys(unittest.TestCase):
    def test_keys_invariant_under_renaming(self):
        rng, n = random.Random(4), 5
        for _ in range(60):
            w = np.zeros(30, dtype=np.int64)
            for k in rng.sample(range(30), rng.randint(1, 6)):
                w[k] = rng.choice((-2, -1, 1, 2, 3))
            c = rng.choice((0.125, 0.25, 0.75))
            sig = rng.sample(range(n), n)
            ws = FAM.rename_weight(w, sig)
            self.assertEqual(FAM.canon_key(w, c, n), FAM.canon_key(ws, c, n))
            self.assertEqual(FAM.canon_key(w, c, n, dyn="height-3"), FAM.canon_key(ws, c, n, dyn="height-3"))
            for signed in (False, True):
                self.assertEqual(FAM.support_key(w, n, signed), FAM.support_key(ws, n, signed))
            self.assertEqual(FAM.support_fingerprint(w, n), FAM.support_fingerprint(ws, n))
            G = rng.sample(range(20), 2)
            fp = SO.fact_perm(n, sig)
            self.assertEqual(FAM.canon_key((), 0.25, n, "conj", G=G, R=3.0),
                             FAM.canon_key((), 0.25, n, "conj", G=[int(fp[k]) for k in G], R=3.0))
        self.assertNotEqual(FAM.canon_key(w, c, n), FAM.canon_key(w, c, n, dyn="height-3"))

    def test_nine_two_fact_support_classes(self):
        for n in (4, 5):
            keys = {FAM.support_key(FAM.indicator(n, pair), n) for pair in itertools.combinations(range(n * n), 2)}
            self.assertEqual(len(keys), 9)
            self.assertEqual(FAM.n_support_classes(n, 2), 9)
        brute = {FAM.support_key(FAM.indicator(4, t), 4) for t in itertools.combinations(range(16), 3)}
        self.assertEqual(FAM.n_support_classes(4, 3), len(brute))

    def test_scale_equivalence_m6(self):
        n = 5
        sup = {("on", 0, 1): 1, ("ontable", 2): 1, ("on", 3, 4): 1}
        one = wvec(n, sup)
        key = lambda w, c: FAM.canon_key(w, c, n)  # noqa: E731
        self.assertEqual(key(one, 0.125), key(2 * one, 0.25))  # all-1 at c = 0.125 = all-2 at c = 0.25
        self.assertEqual(key(3 * one, 0.75), key(one, 0.25))  # all-3 at c = 0.75 = all-1 at c = 0.25
        self.assertNotEqual(key(one, 0.25), key(2 * one, 0.25))  # w vs 2w at the same c
        self.assertNotEqual(key(one, 0.25), key(one, 0.125))
        mixed = wvec(n, {("on", 0, 1): 1, ("ontable", 2): 3})
        sig = [4, 2, 0, 1, 3]
        self.assertEqual(key(mixed, 0.125), key(FAM.rename_weight(2 * mixed, sig), 0.25))  # renamed and scaled
        self.assertEqual(FAM.primitive_vector(3 * one, 0.75)[-1], -1)
        self.assertEqual(sorted(set(FAM.primitive_vector(3 * one, 0.75)[:-1])), [0, 4])
        with self.assertRaises(ValueError):
            FAM.primitive_vector(one, 0.1)
        self.assertEqual(FAM.canon_key((), 0.25, n, "conj", G=[0, 5], R=3.0),
                         FAM.canon_key((), 0.5, n, "conj", G=[0, 5], R=6.0))
        self.assertNotEqual(FAM.canon_key((), 0.25, n, "conj", G=[0, 5], R=3.0),
                            FAM.canon_key(1.5 * FAM.indicator(n, [0, 5]), 0.25, n))

    def test_signed_keys(self):
        n = 5
        a = wvec(n, {("on", 0, 1): 2, ("ontable", 2): -1})
        b = wvec(n, {("on", 0, 1): -2, ("ontable", 2): 1})
        self.assertEqual(FAM.support_key(a, n), FAM.support_key(b, n))
        self.assertNotEqual(FAM.support_key(a, n, True), FAM.support_key(b, n, True))
        self.assertTrue(FAM.support_key(a, n, True).startswith("ssupp|"))

    def test_stabilizer(self):
        rng, n = random.Random(5), 5
        for _ in range(20):
            g = FAM.draw_goal(rng.choice(("TRAIN", "H-NEG", "H-SPAN")), rng, n)
            w = g["w"]
            st = FAM.stabilizer(w, n)
            brute = [s for s in itertools.permutations(range(n)) if np.array_equal(FAM.rename_weight(w, s), w)]
            self.assertEqual(sorted(map(tuple, st.tolist())), sorted(brute))
            self.assertIn(tuple(range(n)), brute)
            S = set(brute)
            for s, t in itertools.product(brute, brute):
                self.assertIn(tuple(s[t[i]] for i in range(n)), S)  # closed under composition
            m = len(FAM.mentioned_blocks(w, n))
            self.assertGreaterEqual(len(st), math.factorial(n - m))
        self.assertEqual(len(FAM.stabilizer(wvec(n, {("on", 0, 1): 1}), n)), math.factorial(3))
        self.assertEqual(len(FAM.stabilizer(wvec(n, {("ontable", 0): 1, ("ontable", 1): 1}), n)), 2 * 6)


class TestGenerators(unittest.TestCase):
    def test_draw_goal_specs(self):
        rng, n = random.Random(6), 5
        for base in ("TRAIN", "ID", "H-K", "H-NEG", "H-COST", "H-SPAN", "H-KN", "H-CONJ", "H-DYN"):
            Ks = Counter()
            for _ in range(200):
                g = FAM.draw_goal(base, rng, n)
                Ks[g["K"]] += 1
                if base == "H-CONJ":
                    self.assertTrue(len(g["G"]) == g["K"] in (2, 3) and all(k < 20 for k in g["G"]))
                    self.assertEqual((g["R"], g["c"]), (3.0, 0.25))
                    continue
                w = g["w"]
                ontab, clr = w[:n * n], w[n * n:]
                self.assertEqual(int((ontab != 0).sum()), g["K"])
                neg = int((w < 0).sum())
                self.assertEqual(neg, g["n_neg"])
                if base in ("H-NEG", "H-KN"):
                    self.assertTrue(set(w[w < 0].tolist()) <= {-1, -2} and (w > 0).any())
                    self.assertTrue(1 <= neg <= (1 if g["K"] == 2 else 2))
                else:
                    self.assertEqual(neg, 0)
                self.assertTrue(set(ontab[ontab > 0].tolist()) <= {1, 2, 3})
                if base == "H-SPAN":
                    self.assertTrue(1 <= int((clr != 0).sum()) <= 2 and set(clr[clr != 0].tolist()) <= {1, 2})
                else:
                    self.assertFalse(clr.any())
                self.assertEqual(g["c"] in (0.125, 0.75), base == "H-COST")
                self.assertEqual(g["dyn"], "height-3" if base == "H-DYN" else "free")
            want = {"TRAIN": {1, 2, 3, 4}, "ID": {2, 3, 4}, "H-K": {5, 6}, "H-NEG": {2, 3, 4}, "H-COST": {2, 3, 4},
                    "H-SPAN": {1, 2, 3}, "H-KN": {5, 6}, "H-CONJ": {2, 3}, "H-DYN": {1, 2, 3, 4}}[base]
            self.assertEqual(set(Ks), want, base)

    def test_pilot_and_test_draw(self):
        sp, reg, _ = draw_small()
        res = FAM.assert_splits(reg)
        self.assertTrue(all(ok for ok, _ in res.values()))
        fams = reg.parents
        self.assertEqual([len(fams[f]) for f in ("TRAIN", "DEV-ID", "H-ID", "H-K", "H-NEG")], [12, 6, 8, 3, 3])
        for f in FAM.DEV_OOD:
            self.assertEqual(len(fams[f"DEV-OOD:{f}"]), 2)
        for f in FAM.SECONDARY:
            self.assertEqual(len(fams[f]), 2)
        dev_s = [p for p in fams["DEV-ID"] if p.stratum == "S"]
        self.assertEqual(len(dev_s), 3)
        supp_l, supp_dev = reg.supports("TRAIN"), reg.supports("DEV-ID")
        hid_s = [p for p in fams["H-ID"] if p.stratum == "S"]
        hid_u = [p for p in fams["H-ID"] if p.stratum == "U"]
        nt = reg.notes["H-ID"]
        self.assertTrue(len(hid_s) == 4 or len(hid_s) == nt["available_s"] - nt["failed_s"])  # min(quota, available)
        self.assertTrue(all(p.supp_u in supp_l for p in hid_s))
        self.assertTrue(all(p.supp_u not in supp_l | supp_dev for p in hid_u))
        self.assertTrue(all(p.K in (2, 3, 4) for p in fams["H-ID"]))
        for f, ps in fams.items():
            if f != "TRAIN":
                self.assertEqual(len({p.supp for p in ps}), len(ps), f)  # one parent per support class
        self.assertTrue(all(p.supp.startswith("ssupp|") for p in fams["H-NEG"] + fams["H-KN"]))
        self.assertEqual(len(reg.keys), sum(len(v) for v in fams.values()))
        phi = SO.get_phi(sp)
        hok = SO.height_ok(sp, 3)
        for p in reg.all_parents():
            if p.family == "TRAIN":
                self.assertEqual((p.starts, p.bank), ([], []))
                continue
            self.assertEqual(len(p.starts), 16)
            self.assertEqual(len(p.bank), 32 if not p.family.startswith("DEV-OOD") else 0)
            v = FAM.verify_parent(sp, phi, p)
            self.assertTrue(v["starts_ok"] and v["bank_ok"], (p.family, v))
            if p.family == "H-DYN":
                self.assertTrue(hok[p.starts].all())
                tab, w = FAM.parent_table(sp, phi, p), p.weights()
                self.assertTrue(np.array_equal(tab.V, SO.soft_values(sp, w, p.c, SO.height_limit_mask(sp), phi)))
                self.assertTrue((tab.V <= SO.soft_values(sp, w, p.c, None, phi)).all())
            if p.family == "H-CONJ":
                self.assertEqual(p.rtype, "conj")
                tab = FAM.parent_table(sp, phi, p)
                fs = SO.facts(5)
                self.assertTrue(np.array_equal(tab.V, SO.conj_values(sp, [fs[k] for k in p.G], 3.0, 0.25)))
            self.assertEqual(FAM.Parent.from_json(json.loads(json.dumps(p.to_json()))), p)
        self.assertGreater(sum(sum(c.values()) for c in reg.rejections.values()), 0)

    def test_determinism(self):
        a = draw_small()[1]
        b = draw_small()[1]
        self.assertEqual([p.to_json() for p in a.all_parents()], [p.to_json() for p in b.all_parents()])
        self.assertEqual({k: dict(v) for k, v in a.rejections.items()}, {k: dict(v) for k, v in b.rejections.items()})
        self.assertEqual(FAM.pgv_subset(a, 2, 1), FAM.pgv_subset(b, 2, 1))
        sub = FAM.pgv_subset(a, 2, 1)
        self.assertEqual(len(sub), 6)
        self.assertTrue(set(sub) <= {p.key for f in FAM.PRIMARY for p in a.parents[f]})
        dk = FAM.dev_kcurve_subset(a, 2)
        self.assertEqual(Counter(a.keys[k] for k in dk), Counter({"DEV-ID": 4}))

    def test_disjointness_assertions(self):
        n = 5
        sp = SP.get_space(n)
        reg = FAM.Registry(n)
        one = wvec(n, {("on", 0, 1): 1, ("ontable", 2): 1})
        g = dict(w=2 * one, c=0.25, rtype="lin", dyn="free", G=(), R=0.0, K=2, n_neg=0)
        reg.add(FAM.make_parent("TRAIN", n, g))
        dup = FAM.make_parent("H-COST", n, dict(g, w=FAM.rename_weight(one, [3, 4, 0, 1, 2]), c=0.125))
        with self.assertRaises(AssertionError):  # scale- and renaming-equivalent: one class
            reg.add(dup)
        reg.parents["H-COST"].append(dup)  # bypass the registry: check_splits must catch it
        res = FAM.check_splits(reg)
        self.assertFalse(res["class_disjoint"][0])
        with self.assertRaises(AssertionError):
            FAM.assert_splits(reg)
        reg2 = FAM.Registry(n)
        reg2.add(FAM.make_parent("TRAIN", n, g))
        for x in (1, 3):  # two H-K parents with one support class
            reg2.add(FAM.make_parent("H-K", n, dict(g, w=x * one)))
        mix = wvec(n, {("on", 0, 1): 1, ("ontable", 2): 2})
        reg2.add(FAM.make_parent("H-ID", n, dict(g, w=mix), stratum="U"))  # seen support in the U stratum
        res = FAM.check_splits(reg2)
        self.assertFalse(res["support_unique"][0])
        self.assertFalse(res["H-ID_strata"][0])
        p = reg2.parents["H-K"][0]
        p.key = FAM.canon_key(5 * one, 0.25, n)
        self.assertFalse(FAM.check_splits(reg2)["keys_recomputed"][0])
        # the generator rejects duplicate classes: replaying the same stream re-draws only used classes first
        reg3 = FAM.Registry(n)
        kw = dict(split="t", n_starts=0, n_bank=0, unique_support=False)
        FAM.draw_family(reg3, "TRAIN", 5, random.Random(7), sp, None, **kw)
        first = reg3.notes["TRAIN"]["draws"]
        FAM.draw_family(reg3, "TRAIN", 5, random.Random(7), sp, None, **kw)
        self.assertGreaterEqual(reg3.rejections["TRAIN"]["duplicate class"], first)
        self.assertEqual(len(reg3.keys), 10)
        with self.assertRaises(RuntimeError):
            FAM.draw_family(reg3, "DEV-OOD:H-K", 1, random.Random(7), sp, None, max_draws=0, **kw)

    def test_hid_s_takes_min_quota_available(self):
        n = 5
        sp = SP.get_space(n)
        reg = FAM.Registry(n)
        FAM.draw_family(reg, "TRAIN", 6, random.Random(8), sp, None, split="t", n_starts=0, n_bank=0,
                        unique_support=False)
        supp_l = reg.supports("TRAIN")
        avail = {p.supp_u for p in reg.parents["TRAIN"] if p.K in (2, 3, 4)}
        out = FAM.draw_family(reg, "H-ID", 10, random.Random(9), sp, SO.get_phi(sp), split="t", n_bank=0,
                              quota_s=40, seen=supp_l, blocked=supp_l)
        n_s = sum(p.stratum == "S" for p in out)
        self.assertEqual(len(out), 10)
        self.assertEqual(n_s, len(avail) - reg.notes["H-ID"]["failed_s"])  # S = min(40, available)
        self.assertTrue({p.supp_u for p in out if p.stratum == "S"} <= avail)
        self.assertTrue(all(p.supp_u not in supp_l for p in out if p.stratum == "U"))


class TestExclusions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B, cls.VL = BI.load()
        cls.sp = SP.get_space(4)
        cls.objs = list(cls.B.NAMES[:4])

    def _setup(self, sig_ns):
        goal = frozenset({("on", 0, 1), ("ontable", 2)})
        X0 = FAM.Exclusions(self.sp, synth_ex(4))
        named = X0.name(goal)
        s_sig = self.sp.index[(1, 4, 4, 4)]  # a on b, hand empty
        s_inv = self.sp.index[(5, 4, 4, 4)]  # holding a; towers (1, 1, 1, 1)
        sig = self.B.signature(self.objs, X0.facts_of(s_sig), named)
        inv = self.VL.invariant(self.objs, X0.facts_of(s_inv), named)
        ex = synth_ex(4, {inv}, {sig}, sig_ns)
        return goal, named, ex, FAM.Exclusions(self.sp, ex)

    def _check_adaptor(self, X, goal, expected, kind):
        kept, check, cnt = X.prefilter(np.arange(len(self.sp)), goal, kind)
        ks = set(kept.tolist())
        removed = [i for i in range(len(self.sp)) if i not in ks]
        self.assertTrue(all(expected[i] for i in removed))
        self.assertEqual(cnt, Counter(expected[i] for i in removed))
        for i in ks:
            self.assertEqual(check(i) if check else None, expected[i])
        out, c2 = FAM.sample_states(np.arange(len(self.sp)), len(self.sp), random.Random(0), goal, X, kind)
        self.assertEqual(set(out), {i for i, r in expected.items() if not r})
        self.assertEqual(c2, Counter(r for r in expected.values() if r))
        return check

    def test_row_adaptor_equals_row_exclusion_reason(self):
        for sig_ns in ([4], []):  # signature branch on (deferred per state) / off (class decisions only)
            goal, named, ex, X = self._setup(sig_ns)
            expected = {i: EX.row_exclusion_reason(self.objs, X.facts_of(i), named, ex) for i in range(len(self.sp))}
            got = Counter(expected.values())
            self.assertEqual(got["invariant"], 5)  # all-on-table + the 4 single-holding states
            self.assertEqual(got["signature"], 1 if sig_ns else 0)
            check = self._check_adaptor(X, goal, expected, "row")
            self.assertEqual(check is None, not sig_ns)

    def test_instance_adaptor_equals_exclusion_reason(self):
        goal, named, ex, X = self._setup([4])
        expected = {i: EX.exclusion_reason(self.objs, X.facts_of(i), named, ex) for i in range(len(self.sp))}
        self.assertEqual(Counter(expected.values())["signature"], 1)
        self._check_adaptor(X, goal, expected, "instance")

    def test_starts_bank_labels_never_excluded(self):
        """Synthetic exclusions of every all-on-table row (any goal tower pattern, including those of conflicting goal
        facts) at n = 4: drawn starts, bank states and label rows avoid them, and the matches are counted."""
        sp, phi = self.sp, SO.get_phi(self.sp)
        pats = {t for L in range(1, 5) for t in itertools.combinations_with_replacement(range(1, 9), L)}
        ex = synth_ex(4, {(4, (1, 1, 1, 1), t) for t in pats})
        X = FAM.Exclusions(sp, ex)
        banned = {i for i in range(len(sp)) if X.VL.towers_of(X.facts_of(i), X.objs) == [1, 1, 1, 1]}
        self.assertEqual(len(banned), 5)
        reg = FAM.Registry(4)
        out = FAM.draw_family(reg, "H-NEG", 3, random.Random(1), sp, phi, split="t", excl=X, n_starts=8, n_bank=8)
        n_counted = 0
        for p in out:
            lab, cl = FAM.draw_labels(sp, p, 60, X)
            self.assertFalse(banned & set(p.starts + p.bank + lab))
            self.assertEqual(cl, Counter(invariant=5))
            v = FAM.verify_parent(sp, phi, p, X, lab)
            self.assertEqual((v["excluded_starts"], v["excluded_bank"], v["excluded_labels"]), (0, 0, 0))
            n_counted += sum(p.counts["starts_excluded"].values()) + sum(p.counts["bank_excluded"].values())
            lab2, _ = FAM.draw_labels(sp, p, 20, X)
            self.assertEqual(lab[:20], lab2)  # nested label sets
        self.assertGreater(n_counted, 0)
        # H-CONJ starts go through excl.exclusion_reason (instance view)
        out = FAM.draw_family(reg, "H-CONJ", 2, random.Random(2), sp, phi, split="t", excl=X, n_starts=8, n_bank=8)
        for p in out:
            self.assertFalse(banned & set(p.starts + p.bank))
            self.assertEqual(FAM.verify_parent(sp, phi, p, X)["excluded_starts"], 0)
        self.assertGreater(X.calls["instance"], 0)


if __name__ == "__main__":
    unittest.main()
