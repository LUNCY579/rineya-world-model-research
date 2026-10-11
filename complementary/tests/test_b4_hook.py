"""Fast tests for rcx.bw.b4_hook (the B4 `unmet=` hook). Tiny randomly initialised RelGNNs only, no training.

Run: python -B -m unittest tests.test_b4_hook -v   (CPU, 1 thread; well under a second after the torch import)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)

from rcx.bw import b4_hook as HK  # noqa: E402
from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import encode as EN  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

M = HK._torch_models()[1]
OBJS = ["a", "b", "c", "d"]
GOAL = frozenset({("on", "a", "b"), ("on", "b", "c")})
# valid states over OBJS (compact codes: TABLE = 4, HAND = 5)
VALID = [SP.to_facts(p, OBJS) for p in [(4, 4, 4, 4), (1, 2, 4, 4), (5, 4, 1, 4), (3, 0, 4, 4), (1, 2, 3, 4)]]
S0 = VALID[0]
H = ("handempty",)


def mod(s, add=(), rm=()):
    return (frozenset(s) - frozenset(rm)) | frozenset(add)


# one hand-made state per inconsistency type: (state, expected reason)
INCONSISTENT = [
    (mod(S0, add=[("on", "a", "c"), ("on", "b", "c")], rm=[("ontable", "a"), ("ontable", "b"), ("clear", "c")]),
     "shared_support"),
    (mod(S0, add=[("on", "a", "b")], rm=[("clear", "b")]), "multiple_positions"),
    (mod(S0, add=[("holding", "a")], rm=[("ontable", "a"), ("clear", "a")]), "holding_and_handempty"),
    (mod(S0, rm=[("ontable", "d")]), "no_position"),
    (frozenset({("on", "a", "b"), ("on", "b", "a"), ("ontable", "c"), ("ontable", "d"), ("clear", "c"),
                ("clear", "d"), H}), "cycle"),
    (mod(S0, add=[("ontable", "zz"), ("clear", "zz")]), "unknown_block"),     # with the instance registry
    (mod(S0, rm=[("clear", "c")]), "missing_clear"),
    (mod(VALID[1], add=[("clear", "b")]), "wrong_clear"),
    (mod(S0, add=[("holding", "a"), ("holding", "b")],
         rm=[("ontable", "a"), ("ontable", "b"), H, ("clear", "a"), ("clear", "b")]), "multiple_held"),
    (mod(S0, rm=[H]), "hand_unspecified"),
    (frozenset({("holding", "b"), ("on", "a", "b"), ("clear", "a"), ("ontable", "c"), ("ontable", "d"),
                ("clear", "c"), ("clear", "d")}), "on_held_block"),
    (mod(S0, add=[("on", "a")]), "malformed_fact"),
    (mod(S0, add=[()]), "malformed_fact"),
    (mod(S0, add=[("foo", "a")]), "malformed_fact"),
    (mod(S0, add=["clear a"]), "malformed_fact"),
    (mod(S0, add=[("on", 1, "b")]), "malformed_fact"),
    (mod(S0, add=[("clear", None)]), "malformed_fact"),
    (mod(S0, add=[("handempty", "x")]), "malformed_fact"),
    (mod(S0, add=[("on", "a", "a")]), "self_on"),
    (frozenset(), "hand_unspecified"),
]
WEIRD_INPUTS = [None, 42, "abc", [("on", "a", "b")], {("ontable", "a")}, [["on", "a"]], frozenset({(("x",),)})]


def tiny_gnn(seed=0, **kw):
    torch.manual_seed(seed)
    return M.RelGNN(M.RelGNNConfig(hidden=8, rounds=2, head_hidden=8, **kw)).eval()


def direct(model, s, goal, objs=None):
    objs = sorted(BI_blocks(s) | BI_blocks(goal)) if objs is None else objs
    return float(M.predict(model, [EN.graph(s, goal, objs, derived=model.cfg.n_rel > len(EN.EDGE_TYPES))])[0])


def BI_blocks(facts):
    return {a for f in facts for a in f[1:]}


class TestHookValues(unittest.TestCase):
    def setUp(self):
        self.model = tiny_gnn()
        self.hook = HK.LearnedUnmet(self.model, instances=[{"goal": GOAL, "objs": OBJS}])

    def test_valid_states_equal_model_prediction(self):
        for s in VALID:
            self.assertEqual(HK.is_consistent(s, GOAL, OBJS), (True, []))
            v = self.hook(s, GOAL)
            self.assertIsInstance(v, float)
            self.assertTrue(math.isfinite(v))
            self.assertEqual(v, direct(self.model, s, GOAL))
        self.assertEqual(self.hook.counters["model_evals"], len(VALID))
        self.assertEqual(self.hook.counters["invalid_states"], 0)
        # derived-edge variant (n_rel = 8) uses the derived encoding
        m8 = tiny_gnn(1, n_rel=8)
        h8 = HK.LearnedUnmet(m8)
        self.assertTrue(h8.derived)
        self.assertEqual(h8(VALID[4], GOAL), direct(m8, VALID[4], GOAL))

    def test_each_inconsistency_type_is_finite_and_flagged(self):
        seen = set()
        for s, why in INCONSISTENT:
            ok, reasons = HK.is_consistent(s, GOAL, OBJS)
            self.assertFalse(ok, (s, why))
            self.assertIn(why, reasons, (s, why))
            seen.add(why)
            v = self.hook(s, GOAL)
            self.assertTrue(math.isfinite(v) and HK.PENALTY <= v < HK.FAILSAFE, (s, why, v))
            sane, objs = HK.sanitise(s, GOAL, OBJS)
            SP.from_facts(sane, objs)  # the sanitised copy is a valid state (raises otherwise)
            self.assertEqual(v, HK.PENALTY + HK.misplaced_sanitised(s, GOAL, OBJS))
        self.assertEqual(seen, set(HK.REASONS))
        for x in WEIRD_INPUTS:  # never raises, always finite, for any state or goal argument
            for args in ((x, GOAL), (S0, x), (x, x)):
                v = self.hook(*args)
                self.assertTrue(math.isfinite(v), (args, v))
        self.assertTrue(math.isfinite(self.hook(S0, frozenset({("ontable", "a"), ("on", "b", "c")}))))

    def test_raw_policy_scores_wellformed_facts(self):
        hook = HK.LearnedUnmet(self.model, invalid_policy="raw")
        for s, _ in INCONSISTENT:
            v = hook(s, GOAL)
            self.assertTrue(math.isfinite(v))
            a = HK.analyse(s, GOAL)
            self.assertEqual(v, direct(self.model, a["state"], a["goal"], a["objs"]))
        self.assertEqual(hook.counters["fallbacks"], 0)

    def test_batch_equals_individual_and_cache_free(self):
        states = VALID + [s for s, _ in INCONSISTENT] + WEIRD_INPUTS
        b = self.hook.batch(states, GOAL)
        nocache = HK.LearnedUnmet(self.model, cache_size=0, instances=[{"goal": GOAL, "objs": OBJS}])
        self.assertEqual(b, [nocache(s, GOAL) for s in states])
        self.assertEqual(b, self.hook.batch(list(reversed(states)), GOAL)[::-1])

    def test_counters(self):
        h = self.hook
        h(VALID[1], GOAL)
        h(VALID[1], GOAL)
        h(INCONSISTENT[0][0], GOAL)
        h(None, GOAL)
        c = h.counters
        self.assertEqual((c["calls"], c["cache_hits"], c["evaluations"], c["model_evals"]), (4, 1, 3, 1))
        self.assertEqual((c["invalid_states"], c["fallbacks"], c["failsafe"], c["analysis_errors"]), (1, 1, 1, 1))
        self.assertEqual(h.reasons["shared_support"], 1)
        h(S0, frozenset({("ontable", "a")}) | GOAL)
        self.assertEqual(c["goal_non_on"], 1)
        h(S0, GOAL | {("on", "x")})
        self.assertEqual(c["bad_goal_facts"], 1)
        nan_model = tiny_gnn()
        with torch.no_grad():
            nan_model.head[-1].bias.fill_(float("nan"))
        hn = HK.LearnedUnmet(nan_model)
        self.assertEqual(hn(S0, GOAL), HK.PENALTY + HK.misplaced_sanitised(S0, GOAL))
        self.assertEqual((hn.counters["nonfinite"], hn.counters["fallbacks"]), (1, 1))
        big = tiny_gnn()
        with torch.no_grad():
            big.head[-1].bias.fill_(1e6)
        hb = HK.LearnedUnmet(big)
        self.assertEqual(hb(S0, GOAL), HK.MODEL_CLIP)
        self.assertEqual(hb.counters["clipped"], 1)
        enc = HK.LearnedUnmet(self.model)
        orig = EN.graph
        try:
            EN.graph = lambda *a, **k: 1 / 0
            self.assertEqual(enc(S0, GOAL), HK.PENALTY + HK.misplaced_sanitised(S0, GOAL))
        finally:
            EN.graph = orig
        self.assertEqual(enc.counters["encode_errors"], 1)

    def test_determinism(self):
        h2 = HK.LearnedUnmet(tiny_gnn(), cache_size=2, instances=[{"goal": GOAL, "objs": OBJS}])
        states = VALID + [s for s, _ in INCONSISTENT]
        first = [self.hook(s, GOAL) for s in states]
        self.assertEqual(first, [h2(s, GOAL) for s in reversed(states)][::-1])
        self.assertEqual(first, [self.hook(s, GOAL) for s in states])

    def test_hybrid_and_rounding(self):
        h = HK.LearnedUnmet(self.model, hybrid_lambda=0.5)
        for s in VALID:
            mis = SP.misplaced_compact(SP.from_facts(s, OBJS), SP.CompiledGoal(OBJS, GOAL))[0]
            self.assertEqual(HK.misplaced_sanitised(s, GOAL), mis)
            self.assertAlmostEqual(h(s, GOAL), direct(self.model, s, GOAL) + 0.5 * mis, places=6)
        r = HK.LearnedUnmet(self.model, round_to=0.25)
        v = r(VALID[2], GOAL)
        self.assertEqual(v, round(direct(self.model, VALID[2], GOAL) / 0.25) * 0.25)


class TestConsistencyAndHost(unittest.TestCase):
    def test_is_consistent_matches_from_facts(self):
        objs = ["a", "b", "c"]
        sp = SP.StateSpace(3)
        universe = ([("on", x, y) for x in objs for y in objs] + [(p, x) for p in ("ontable", "clear", "holding")
                                                                  for x in objs] + [H])
        n = 0
        for i in range(len(sp)):
            s = sp_state(sp, i, objs)
            self.assertEqual(HK.is_consistent(s, None, objs), (True, []))
            for f in universe:
                t = s ^ {f}
                try:
                    SP.from_facts(t, objs)
                    ref = True
                except ValueError:
                    ref = False
                ok, reasons = HK.is_consistent(t, None, objs)
                self.assertEqual(ok, ref and "on_held_block" not in reasons, (t, reasons))
                if ok:
                    self.assertEqual(HK._sanitised_pos(t, objs), SP.from_facts(t, objs))
                n += 1
        self.assertGreater(n, 400)

    def test_host_neutral(self):
        loaded_bwm = "B" in BI._STATE
        torch.set_num_threads(2)
        try:
            h = HK.LearnedUnmet(tiny_gnn())
            h(S0, GOAL)
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(1)
        self.assertEqual("B" in BI._STATE, loaded_bwm)  # the hook never loads BWM code
        self.assertTrue(bool(h))
        self.assertIs(h or (lambda st, g: len(g - st)), h)  # plan_beam's `unmet or default` binding

    def test_from_file_and_describe(self):
        model = tiny_gnn(3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gnn_s1.pt"
            torch.save(model.state_dict(), p)
            p.with_suffix(".json").write_text(json.dumps({"config": M.config_dict(model)}), encoding="utf-8")
            rng = torch.get_rng_state()
            h = HK.LearnedUnmet.from_file(p)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))  # host RNG untouched by construction
            for s in VALID:
                self.assertEqual(h(s, GOAL), HK.LearnedUnmet(model)(s, GOAL))
            desc = h.describe()
            self.assertEqual(desc["model_sha256"], HK._file_sha(p))
            self.assertEqual(desc["weights_sha256"], HK.weights_sha256(model))
            self.assertEqual(desc["config"], M.config_dict(model))
            self.assertEqual(desc["config_source"], "sidecar json 'config'")
            self.assertEqual(desc["counters"]["calls"], len(VALID))
            json.dumps(desc)
            bad = dict(M.config_dict(model), n_params=1)
            with self.assertRaises(ValueError):
                HK.LearnedUnmet.from_file(p, config=bad)

    @unittest.skipUnless(HK.BWM_PLANNER.exists(), "BWM bundle not present")
    def test_planner_hook_usage(self):
        r = HK.planner_hook_usage()
        self.assertTrue(r["matches_pin"])
        self.assertEqual([c["line"] for c in r["call_sites"]], [286, 319])
        self.assertTrue(all(c["n_positional"] == 2 and not c["keywords"] for c in r["call_sites"]))
        self.assertTrue(r["hook_keyword_only"] and r["facts"]["hook_selected_by_truthiness"])
        self.assertTrue(all(u["found"] for u in r["value_uses"].values()))
        self.assertEqual(r["arithmetic_or_conversion_on_value"], [])


def sp_state(sp, i, objs):
    return SP.to_facts(tuple(int(v) for v in sp.states[i]), objs)


if __name__ == "__main__":
    unittest.main()
