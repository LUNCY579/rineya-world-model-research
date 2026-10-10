"""Unit tests for rcx.stats, including parity with the bundle reference tool paired_metrics.py."""
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
from rcx import stats as S  # noqa: E402

BUNDLE_TOOL = Path('C:/Users/clclc/BusinessProject/WorldModel/rineya_complementary_research_2026-10-09'
                   '/08_reference_tools/paired_metrics.py')
# Computed on 2026-10-09 by calling the bundle's paired_metrics.mcnemar_exact on each (b, c).
MCNEMAR_EXPECTED = {
    (0, 0): 1.0, (1, 0): 1.0, (0, 1): 1.0, (3, 3): 1.0, (4, 5): 1.0, (0, 5): 0.0625,
    (2, 8): 0.10937500000000001, (10, 3): 0.09228515625000001, (68, 0): 6.776263578034422e-21,
    (15, 30): 0.03569780355519454, (40, 60): 0.056887933640979826, (7, 1): 0.07031249999999994,
}
# Runs once in a subprocess: load the bundle tool read-only (no bytecode) and report reference outputs.
_BUNDLE_SCRIPT = r'''
import json, runpy, sys
sys.dont_write_bytecode = True
tool = runpy.run_path(sys.argv[1])
req = json.load(sys.stdin)
print(json.dumps(dict(
    mcnemar=[tool['mcnemar_exact'](b, c) for b, c in req['pairs']],
    floats=tool['analyze'](req['float_rows'], 2000, 17),
    binary=tool['analyze'](req['binary_rows'], 1000, 5))))
'''


def make_arms(n_parents=30, shift=0.0, noise=0.1, seed=3, variants=(1, 4)):
    """Two paired arms with parent effects shared across arms; b = a-like + shift."""
    rng = random.Random(seed)
    a, b = {}, {}
    for i in range(n_parents):
        base, k = rng.random(), rng.randint(*variants)
        a[f'p{i:02d}'] = [base + rng.gauss(0, noise) for _ in range(k)]
        b[f'p{i:02d}'] = [base + shift + rng.gauss(0, noise) for _ in range(k)]
    return a, b


def to_rows(a, b):
    return [dict(parent_id=p, variant_id=f'v{j}', baseline=x, candidate=y)
            for p in a for j, (x, y) in enumerate(zip(a[p], b[p]))]


class BundleParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.float_a, cls.float_b = make_arms(shift=0.05, seed=11)
        rng = random.Random(23)
        cls.bin_a = {f'q{i:02d}': [float(rng.random() < 0.6)] for i in range(40)}
        cls.bin_b = {f'q{i:02d}': [float(rng.random() < 0.75)] for i in range(40)}
        cls.ref = None
        if BUNDLE_TOOL.exists():
            req = dict(pairs=list(MCNEMAR_EXPECTED), float_rows=to_rows(cls.float_a, cls.float_b),
                       binary_rows=to_rows(cls.bin_a, cls.bin_b))
            cmd = [sys.executable, '-I', '-B', '-c', _BUNDLE_SCRIPT, str(BUNDLE_TOOL)]
            out = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True, timeout=120,
                                 cwd=Path(__file__).parent)
            if out.returncode != 0:
                raise RuntimeError(out.stderr)
            cls.ref = json.loads(out.stdout)

    def test_mcnemar_matches_hardcoded_bundle_values(self):
        for (b, c), expected in MCNEMAR_EXPECTED.items():
            self.assertEqual(S.mcnemar_exact(b, c), expected, (b, c))

    def test_mcnemar_matches_live_bundle(self):
        if self.ref is None:
            self.skipTest('bundle tool not present')
        self.assertEqual([S.mcnemar_exact(b, c) for b, c in MCNEMAR_EXPECTED], self.ref['mcnemar'])

    def test_cluster_bootstrap_matches_live_bundle(self):
        if self.ref is None:
            self.skipTest('bundle tool not present')
        ref, got = self.ref['floats'], S.cluster_bootstrap_diff(self.float_a, self.float_b, resamples=2000, seed=17)
        self.assertEqual(got['n_parents'], ref['parents'])
        self.assertAlmostEqual(got['mean_a'], ref['baseline_parent_macro'], places=12)
        self.assertAlmostEqual(got['mean_b'], ref['candidate_parent_macro'], places=12)
        self.assertAlmostEqual(got['diff'], ref['candidate_minus_baseline'], places=12)
        self.assertAlmostEqual(got['lo'], ref['percentile_interval_95'][0], places=12)
        self.assertAlmostEqual(got['hi'], ref['percentile_interval_95'][1], places=12)
        self.assertAlmostEqual(S.parent_macro(self.float_a), ref['baseline_parent_macro'], places=12)

    def test_binary_table_matches_live_bundle(self):
        if self.ref is None:
            self.skipTest('bundle tool not present')
        ref = self.ref['binary']
        b, c = S.paired_binary_table(self.bin_a, self.bin_b)
        self.assertEqual((b, c), (ref['baseline_only'], ref['candidate_only']))
        self.assertEqual(S.mcnemar_exact(b, c), ref['mcnemar_two_sided_exact'])

    def test_mcnemar_validation(self):
        for bad in [(-1, 2), (1.0, 2), (True, 2)]:
            with self.assertRaises(ValueError):
                S.mcnemar_exact(*bad)


class WilsonHolmTest(unittest.TestCase):
    def test_wilson_known_values(self):
        cases = {(0, 10): (0.0, 0.2775), (5, 10): (0.2366, 0.7634), (10, 10): (0.7225, 1.0)}
        for (k, n), (lo, hi) in cases.items():
            got = S.wilson(k, n)
            self.assertAlmostEqual(got[0], lo, places=4)
            self.assertAlmostEqual(got[1], hi, places=4)
        # scipy 1.17 binomtest(...).proportion_ci(method='wilson') uses z = norm.ppf(0.975).
        z = 1.959963984540054
        scipy_ref = {(1, 20): (0.008881448800795402, 0.23613119344674205),
                     (81, 263): (0.2552885198782742, 0.36620957698280004),
                     (37, 50): (0.6044684273515115, 0.8412847250644762)}
        for (k, n), (lo, hi) in scipy_ref.items():
            got = S.wilson(k, n, z=z)
            self.assertAlmostEqual(got[0], lo, places=10)
            self.assertAlmostEqual(got[1], hi, places=10)
        self.assertEqual(S.wilson(0, 0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            S.wilson(5, 4)

    def test_holm_known_example(self):
        got = S.holm({'a': 0.01, 'b': 0.04, 'c': 0.03, 'd': 0.005})
        expected = {'a': 0.03, 'b': 0.06, 'c': 0.06, 'd': 0.02}
        self.assertEqual(list(got), ['a', 'b', 'c', 'd'])
        for key in expected:
            self.assertAlmostEqual(got[key], expected[key], places=12)
        self.assertEqual(S.holm({'x': 0.5, 'y': 0.6}), {'x': 1.0, 'y': 1.0})
        self.assertEqual(S.holm({'only': 0.02}), {'only': 0.02})
        self.assertEqual(S.holm({}), {})
        with self.assertRaises(ValueError):
            S.holm({'bad': float('nan')})


class BootstrapTest(unittest.TestCase):
    def test_parent_macro_weights_parents_equally(self):
        self.assertAlmostEqual(S.parent_macro({'x': [1, 0, 0, 0], 'y': [1]}), 0.625)
        self.assertAlmostEqual(S.parent_macro({'x': 0.2, 'y': [0.4, 0.6]}), 0.35)

    def test_identical_arms_ci_is_zero(self):
        a, _ = make_arms()
        r = S.cluster_bootstrap_diff(a, a)
        self.assertEqual((r['diff'], r['lo'], r['hi']), (0.0, 0.0, 0.0))

    def test_null_arms_ci_contains_zero(self):
        a, b = make_arms(shift=0.0, noise=0.1, seed=7)
        r = S.cluster_bootstrap_diff(a, b, seed=1)
        self.assertTrue(r['lo'] < 0 < r['hi'], r)
        self.assertEqual(set(r), {'mean_a', 'mean_b', 'diff', 'lo', 'hi', 'n_parents'})

    def test_strong_difference_excludes_zero(self):
        a, b = make_arms(shift=0.3, noise=0.1, seed=7)
        r = S.cluster_bootstrap_diff(a, b)
        self.assertGreater(r['lo'], 0.2)
        r = S.cluster_bootstrap_diff(b, a)
        self.assertLess(r['hi'], -0.2)

    def test_null_coverage_is_near_nominal(self):
        rng, hits, sims = random.Random(99), 0, 100
        for s in range(sims):
            a = {i: [rng.gauss(0, 1)] for i in range(30)}
            b = {i: [a[i][0] + rng.gauss(0, 1)] for i in range(30)}
            r = S.cluster_bootstrap_diff(a, b, resamples=400, seed=s)
            hits += r['lo'] <= 0 <= r['hi']
        self.assertTrue(0.85 <= hits / sims <= 1.0, hits)

    def test_cluster_bootstrap_mean(self):
        a, _ = make_arms(seed=5)
        r = S.cluster_bootstrap_mean(a)
        self.assertTrue(r['lo'] < r['mean'] < r['hi'])
        self.assertAlmostEqual(r['mean'], S.parent_macro(a))
        self.assertEqual(S.cluster_bootstrap_mean({'x': [0.5], 'y': [0.5]})['lo'], 0.5)

    def test_validation(self):
        a, b = make_arms()
        with self.assertRaises(ValueError):
            S.cluster_bootstrap_diff(a, dict(list(b.items())[1:]))
        with self.assertRaises(ValueError):
            S.cluster_bootstrap_diff({'x': [1.0]}, {'x': [0.0]})
        with self.assertRaises(ValueError):
            S.cluster_bootstrap_diff(a, b, resamples=50)
        with self.assertRaises(ValueError):
            S.cluster_bootstrap_mean({'x': [float('nan')], 'y': [1.0]})
        with self.assertRaises(ValueError):
            S.parent_macro({'x': []})
        for raw in ('75', '10', b'1'):
            with self.assertRaises(ValueError):
                S.parent_macro({'x': raw, 'y': [1.0]})
        self.assertEqual(S.parent_macro({'x': ['0.25', '0.75'], 'y': 1}), 0.75)


class SeedBootstrapTest(unittest.TestCase):
    def test_single_seed_equals_cluster_bootstrap(self):
        a, b = make_arms(shift=0.1, seed=8)
        self.assertEqual(S.seed_parent_bootstrap([a], [b], seed=4),
                         dict(S.cluster_bootstrap_diff(a, b, seed=4), n_seeds_a=1, n_seeds_b=1))

    def test_seeds_averaged_within_parent(self):
        a1, b1 = make_arms(shift=0.2, seed=1)
        a2, b2 = make_arms(shift=0.2, seed=2)
        r = S.seed_parent_bootstrap([a1, a2], [b1, b2, b1])
        self.assertAlmostEqual(r['mean_a'], (S.parent_macro(a1) + S.parent_macro(a2)) / 2, places=12)
        self.assertAlmostEqual(r['mean_b'], (2 * S.parent_macro(b1) + S.parent_macro(b2)) / 3, places=12)
        self.assertGreater(r['lo'], 0.0)
        self.assertEqual((r['n_seeds_a'], r['n_seeds_b'], r['n_parents']), (2, 3, 30))
        one = S.seed_parent_bootstrap([a1, a2])
        self.assertAlmostEqual(one['mean'], r['mean_a'], places=12)
        self.assertTrue(one['lo'] < one['mean'] < one['hi'])

    def test_resample_seeds_widens_ci_under_seed_variance(self):
        a = {f'p{i}': [0.5] for i in range(20)}
        per_seed_b = [{p: [0.5 + off + 0.001 * i] for i, p in enumerate(a)} for off in (-0.2, 0.0, 0.2)]
        flat = S.seed_parent_bootstrap([a], per_seed_b)
        deep = S.seed_parent_bootstrap([a], per_seed_b, resample_seeds=True)
        self.assertAlmostEqual(flat['diff'], deep['diff'], places=12)
        self.assertLess(flat['hi'] - flat['lo'], 0.01)
        self.assertGreater(deep['hi'] - deep['lo'], 0.2)

    def test_parent_mismatch_across_seeds_raises(self):
        a1, b1 = make_arms()
        with self.assertRaises(ValueError):
            S.seed_parent_bootstrap([a1, dict(list(a1.items())[:-1])], [b1])
        with self.assertRaises(ValueError):
            S.seed_parent_bootstrap([], [b1])


class BinaryTableTest(unittest.TestCase):
    def test_threshold_defines_parent_correctness(self):
        a = {'p1': [1, 1], 'p2': [1, 0], 'p3': [0, 0], 'p4': [1], 'p5': [0]}
        b = {'p1': [0, 0], 'p2': [1, 1], 'p3': [1, 1], 'p4': [1], 'p5': [0]}
        self.assertEqual(S.paired_binary_table(a, b), (1, 1))
        self.assertEqual(S.paired_binary_table(a, b, threshold=1.0), (1, 2))
        self.assertEqual(S.paired_binary_table({'x': 1, 'y': 1, 'z': 0}, {'x': 0, 'y': 1, 'z': 1}), (1, 1))
        self.assertEqual(S.paired_binary_table({'x': [1], 'y': [1]}, {'x': [0], 'y': [0]}), (2, 0))
        with self.assertRaises(ValueError):
            S.paired_binary_table({'x': [1]}, {'y': [1]})


if __name__ == '__main__':
    unittest.main()
