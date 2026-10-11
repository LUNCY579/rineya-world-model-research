"""Fast tests for RCX-E11 (rcx.bw.macros; rcx.bw.subgoal labels, achiever and baseline selectors). Pure Python: no
torch, no training, no experiment data; searches only on one hand-made 6-block fixture.

Run: python -B -m unittest tests.test_e11 -v   (CPU, single thread; a few seconds)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import itertools  # noqa: E402
import random  # noqa: E402
import unittest  # noqa: E402

from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import heuristics as H  # noqa: E402
from rcx.bw import macros as MA  # noqa: E402
from rcx.bw import search as S  # noqa: E402
from rcx.bw import space as SP  # noqa: E402
from rcx.bw import subgoal as SG  # noqa: E402

B, _ = BI.load()
TEST_RNG = "20261256:tests"  # derived from the E11 block; unit tests only, never used for reported numbers
HE = ("handempty",)

UNSTACK_PUT = (("unstack", 0, 1), ("put-down", 0))
BUILD2 = (("pick-up", 0), ("stack", 0, 1), ("pick-up", 2), ("stack", 2, 0))
HOLD_BUILD = (("stack", 0, 1), ("pick-up", 2), ("stack", 2, 0))
DISMANTLE2 = (("unstack", 0, 1), ("put-down", 0), ("unstack", 1, 2), ("put-down", 1))


def all_states(n):
    objs = B.NAMES[:n]
    return objs, [SP.to_facts(tuple(int(v) for v in r), objs) for r in SP.get_space(n).states]


def fs(*facts):
    return frozenset(facts)


class TestSchemaAndRegression(unittest.TestCase):
    def test_schema_matches_bwm_step(self):
        for n in (3, 4):
            objs, states = all_states(n)
            for s in states:
                for a in B.ground_actions(objs):
                    P, A, D = MA.schema(a[0], a[1:])
                    app = P <= s and not (a[0] == "stack" and a[1] == a[2])
                    self.assertEqual(B.applicable(s, a), app, (sorted(s), a))
                    if app:
                        self.assertEqual(B.step(s, a)[0], (s - D) | A)

    def test_regression_hand_cases(self):
        m = MA.Macro(UNSTACK_PUT)
        self.assertTrue(m.consistent)
        self.assertEqual(m.pre, fs(("on", 0, 1), ("clear", 0), HE))
        self.assertEqual(m.add, fs(("clear", 1), ("clear", 0), HE, ("ontable", 0)))
        self.assertEqual(m.dele, fs(("on", 0, 1), ("holding", 0)))
        b = MA.Macro(BUILD2, goal_on=((0, 1), (2, 0)))
        self.assertEqual(b.pre, fs(("clear", 0), ("ontable", 0), HE, ("clear", 1), ("clear", 2), ("ontable", 2)))
        self.assertEqual(b.add, fs(("on", 0, 1), HE, ("clear", 2), ("on", 2, 0)))
        self.assertEqual(b.dele, fs(("ontable", 0), ("clear", 1), ("holding", 0), ("ontable", 2), ("clear", 0),
                                    ("holding", 2)))
        self.assertEqual(b.key, "pick-up ?0; stack ?0 ?1; pick-up ?2; stack ?2 ?0 | on(?0,?1),on(?2,?0)")

    def test_sound_macros_accepted(self):
        for steps, n in ((UNSTACK_PUT, 3), (BUILD2, 4), (HOLD_BUILD, 4), (DISMANTLE2, 4)):
            v = MA.validate(MA.Macro(steps), n_check=n)
            self.assertTrue(v["accepted"], v)
            self.assertGreater(v["counts"]["executable_pairs"], 0)

    def test_unsound_or_useless_macros_rejected(self):
        m = MA.Macro(UNSTACK_PUT)
        cases = {
            "inconsistent": MA.Macro((("pick-up", 0), ("pick-up", 1))),
            "self_pair": MA.Macro((("pick-up", 0), ("stack", 0, 0))),
            "wrong_effect": MA.Macro(UNSTACK_PUT, claimed=(m.pre, m.add - {("clear", 1)}, m.dele)),
            "too_weak_pre": MA.Macro(UNSTACK_PUT, claimed=(m.pre - {("clear", 0)}, m.add, m.dele)),
            "too_strong_pre": MA.Macro(UNSTACK_PUT, claimed=(m.pre | {("ontable", 1)}, m.add, m.dele)),
            "vacuous": MA.Macro((("unstack", 0, 1), ("put-down", 0), ("unstack", 2, 1))),
            "identity_pick_put": MA.Macro((("pick-up", 0), ("put-down", 0))),
            "identity_stack_unstack": MA.Macro((("stack", 0, 1), ("unstack", 0, 1))),
        }
        expect = {"inconsistent": "inconsistent", "self_pair": "inconsistent", "wrong_effect": "unsound",
                  "too_weak_pre": "unsound", "too_strong_pre": "incomplete_pre", "vacuous": "vacuous",
                  "identity_pick_put": "identity", "identity_stack_unstack": "identity"}
        for name, mac in cases.items():
            v = MA.validate(mac, n_check=4)
            self.assertFalse(v["accepted"], name)
            self.assertEqual(v["reason"], expect[name], (name, v.get("counts")))


class TestApplication(unittest.TestCase):
    def test_bindings_bruteforce_and_application_equals_replay(self):
        objs, states = all_states(5)
        goal = fs(("on", "b", "a"), ("on", "c", "b"), ("on", "e", "d"))
        gidx = MA.goal_index(goal)
        macros = [MA.Macro(BUILD2, ((0, 1), (2, 0))), MA.Macro(DISMANTLE2), MA.Macro(HOLD_BUILD, ((0, 1), (2, 0))),
                  MA.Macro(UNSTACK_PUT)]
        n_apps = 0
        for k, s in enumerate(states):
            sidx = MA.StateIndex(s)
            for m in macros:
                got = m.bindings(sidx, gidx)
                sub = lambda facts, beta: {(f[0],) + tuple(beta[v] for v in f[1:]) for f in facts}  # noqa: E731
                if k % 3 == 0:
                    gon = lambda beta: {("on", beta[i], beta[j]) for i, j in m.goal_on}  # noqa: E731
                    brute = sorted(beta for beta in itertools.permutations(objs, m.nvars)
                                   if sub(m.pre, beta) <= s and gon(beta) <= goal)
                    self.assertEqual(got, brute, (m.key, sorted(s)))
                for beta in got:
                    traj = m.replay(B, s, beta)
                    self.assertIsNotNone(traj, (m.key, beta))
                    self.assertEqual(traj[-1], m.apply_symbolic(s, beta))
                    n_apps += 1
        self.assertGreater(n_apps, 500)


class TestMining(unittest.TestCase):
    def test_abstraction_counts_and_selection(self):
        plan = [("unstack", "x", "y"), ("put-down", "x"), ("pick-up", "y"), ("stack", "y", "z")]
        goal = fs(("on", "y", "z"))
        steps, gon, names = MA.abstract(plan, goal)
        self.assertEqual(steps, (("unstack", 0, 1), ("put-down", 0), ("pick-up", 1), ("stack", 1, 2)))
        self.assertEqual((gon, names), (((1, 2),), ["x", "y", "z"]))
        plans = [{"id": f"p{i}", "plan": plan, "goal": goal} for i in range(3)]
        table, st = MA.mine(plans, lengths=(2,), max_vars=4)
        self.assertEqual(st["segments"], 9)
        e = table["pick-up ?0; stack ?0 ?1 | on(?0,?1)"]
        self.assertEqual((e["occurrences"], e["support"], e["score"]), (3, 3, 3))
        self.assertIn("unstack ?0 ?1; put-down ?0 | -", table)
        self.assertIn("put-down ?0; pick-up ?1 | -", table)
        vc = MA.ValidationCache(n_check_max=4)
        lib, log = MA.select_library(MA.rank(table), K=3, min_support=2, vcache=vc)
        self.assertEqual(len(lib), 3)
        self.assertTrue(all(x["decision"] == "accepted" for x in log))
        mined_steps = {m.step_key for m in lib}
        rnd, rlog = MA.random_library(lib, random.Random(TEST_RNG), vc, exclude_keys={m.key for m in lib},
                                      exclude_step_keys=mined_steps)
        self.assertEqual(len({r.step_key for r in rnd}), len(rnd))
        for t, r, lg in zip(lib, rnd, rlog):
            self.assertEqual((r.L, len(r.goal_on)), (t.L, len(t.goal_on)))
            self.assertNotIn(r.key, {m.key for m in lib})
            self.assertTrue(vc(r)["accepted"])
            if lg["tier"] == MA.RANDOM_TIERS[0]:
                self.assertEqual(r.nvars, t.nvars)
                self.assertNotIn(r.step_key, mined_steps)
        # L = 2, V = 2 has exactly three sound non-identity patterns; all three are mined here, so tier 0 is empty
        self.assertEqual(len([p for p in MA.pattern_universe(2, 2) if vc(MA.Macro(p))["accepted"]]), 3)
        self.assertIsNone(MA.pre_mutex(MA.Macro(BUILD2).pre))
        self.assertEqual(MA.pre_mutex(MA.Macro((("stack", 0, 1), ("stack", 2, 3))).pre), "two_held")
        back = MA.library_from_json(MA.library_json(lib + rnd))
        self.assertEqual([m.key for m in back], [m.key for m in lib + rnd])

    def test_shuffled_library_disjoint_from_mined_and_length_matched(self):
        mined = [MA.Macro((("pick-up", 0), ("stack", 0, 1)), ((0, 1),)), MA.Macro(UNSTACK_PUT),
                 MA.Macro(HOLD_BUILD, ((0, 1), (2, 0))), MA.Macro(BUILD2, ((0, 1), (2, 0)))]

        def cand(steps, gon, occ, sup):
            return {"key": MA.macro_key(steps, gon), "step_key": MA.step_key(steps), "steps": steps, "goal_on": gon,
                    "L": len(steps), "nvars": MA.Macro(steps).nvars, "occurrences": occ, "support": sup,
                    "score": occ * (len(steps) - 1)}

        table = [cand(mined[0].steps, ((0, 1),), 100, 50),                 # an M macro itself
                 cand(mined[0].steps, (), 90, 40),                         # M's step signature, other goal_on
                 cand(UNSTACK_PUT, (), 85, 40),                            # an M macro itself
                 cand((("put-down", 0), ("pick-up", 1)), (), 80, 30),
                 cand((("put-down", 0), ("unstack", 1, 2)), (), 70, 30),
                 cand(HOLD_BUILD, ((0, 1), (2, 0)), 65, 30),               # an M macro itself (L = 3)
                 cand((("put-down", 0), ("unstack", 1, 2), ("put-down", 1)), (), 60, 3),  # L = 3, support < 5
                 cand((("stack", 0, 1), ("stack", 2, 3)), ((0, 1),), 55, 30),           # vacuous (two held)
                 cand((("unstack", 0, 1), ("stack", 0, 2)), ((0, 2),), 50, 20)]           # L = 2 (any-L fill)
        ranked = MA.rank({c["key"]: c for c in table})
        vc = MA.ValidationCache(n_check_max=4)
        lib, log = MA.select_matched_library(ranked, mined, 5, vc, exclude_step_keys={m.step_key for m in mined},
                                             fill_min_support=1)
        self.assertEqual(len(lib), len(mined))
        self.assertFalse({m.key for m in lib} & {m.key for m in mined})
        self.assertFalse({m.step_key for m in lib} & {m.step_key for m in mined})
        tiers = [m.meta["tier"] for m in lib]
        # L = 2, 2 from support >= 5; L = 3 only below support 5 (tier 1); no L = 4 candidate (tier 2: any L)
        self.assertEqual([m.L for m in lib], [2, 2, 3, 2])
        self.assertEqual(tiers, ["same_L_min_support", "same_L_min_support", "same_L_fill_support",
                                 "any_L_min_support"])
        self.assertTrue(all(vc(m)["accepted"] for m in lib))
        summary = log[-1]
        self.assertIn(MA.macro_key(mined[0].steps, ()), summary["skipped_excluded_keys"])


class TestMacroSearch(unittest.TestCase):
    OBJS = ["a", "b", "c", "d", "e", "f"]
    INIT = B.towers_to_state([["a", "b", "c"], ["d"], ["e", "f"]])
    GOAL = fs(("on", "a", "c"), ("on", "e", "f"), ("on", "d", "e"), ("on", "b", "d"))

    def test_empty_library_reproduces_search(self):
        keys = ("success", "found", "plan", "plan_len", "expansions", "generated", "h_calls", "reason")
        for budget in (3, 40, 400):
            a = S.gbfs(self.OBJS, self.INIT, self.GOAL, H.misplaced_blocks, max_expansions=budget)
            b = MA.gbfs_macro(self.OBJS, self.INIT, self.GOAL, H.misplaced_blocks, (), max_expansions=budget)
            self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})
            a = S.beam_search(self.OBJS, self.INIT, self.GOAL, H.misplaced_blocks, width=4, max_expansions=budget)
            b = MA.beam_macro(self.OBJS, self.INIT, self.GOAL, H.misplaced_blocks, (), width=4, max_expansions=budget)
            self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})

    def test_macro_search_plans_are_valid(self):
        lib = [MA.Macro(BUILD2, ((0, 1), (2, 0))), MA.Macro(UNSTACK_PUT), MA.Macro(HOLD_BUILD, ((0, 1), (2, 0)))]
        for fn in (MA.gbfs_macro, MA.beam_macro):
            r = fn(self.OBJS, self.INIT, self.GOAL, H.misplaced_blocks, lib, max_expansions=400)
            self.assertTrue(r["success"], r["reason"])
            self.assertTrue(B.execute(self.INIT, self.GOAL, r["plan"])["success"])
            self.assertEqual(r["sym_mismatch"], 0)
            self.assertGreater(r["macro_children"], 0)
            self.assertGreaterEqual(r["prim_steps"], r["generated"] - 1)


class TestSubgoalLabels(unittest.TestCase):
    OBJS = ["a", "b", "c", "d"]

    def labels(self, goal, init):
        L = SP.GoalLabels(self.OBJS, goal)
        return SG.label_set(L, L.sid(init))

    def test_four_block_hand_cases(self):
        tower = fs(("on", "a", "b"), ("on", "b", "c"))
        flat = B.towers_to_state([["a"], ["b"], ["c"], ["d"]])
        self.assertEqual(self.labels(tower, flat), fs(("on", "b", "c")))
        c_on_a = B.towers_to_state([["a", "c"], ["b"], ["d"]])
        self.assertEqual(self.labels(tower, c_on_a), fs(("on", "b", "c")))
        two = fs(("on", "a", "b"), ("on", "c", "d"))
        self.assertEqual(self.labels(two, flat), two)
        # b sits on a: the goal's lower fact on(b, c) still comes first (a must leave b's way later anyway)
        b_on_a = B.towers_to_state([["a", "b"], ["c"], ["d"]])
        self.assertEqual(self.labels(tower, b_on_a), fs(("on", "b", "c")))

    def test_labels_equal_bruteforce_over_optimal_plans(self):
        goal = fs(("on", "a", "b"), ("on", "b", "c"))
        L = SP.GoalLabels(self.OBJS, goal)
        checked = 0
        for i in SG.handempty_ids(L):
            s = L.state(i)
            unmet = {f for f in goal if f not in s}
            found = set()

            def walk(cur):
                h = L.h(cur)
                for a in B.ground_actions(self.OBJS):
                    if not B.applicable(cur, a):
                        continue
                    nxt = B.step(cur, a)[0]
                    if L.h(nxt) != h - 1:
                        continue
                    if a[0] == "stack" and ("on", a[1], a[2]) in unmet:
                        found.add(("on", a[1], a[2]))
                    else:
                        walk(nxt)

            walk(s)
            self.assertEqual(SG.label_set(L, i), frozenset(found), sorted(s))
            checked += 1
        self.assertGreater(checked, 50)


class TestAchieverAndSelectors(unittest.TestCase):
    def test_achiever_and_baseline_execution(self):
        objs, states = all_states(5)
        goal = fs(("on", "b", "a"), ("on", "c", "b"), ("on", "e", "d"))
        bu, rf = SG.BottomUp(), SG.ReadyFirst()
        rnd = SG.RandomOrder(goal, TEST_RNG)
        n_runs = 0
        for s in states:
            if HE not in s or B.goal_met(s, goal):
                continue
            for f in SG.unmet_facts(s, goal):
                plan, s2 = SG.achieve(B, s, f)
                self.assertIn(f, s2)
                self.assertIn(HE, s2)
                self.assertTrue(B.execute(s, fs(f), plan)["success"])
            for sel in (bu, rf, rnd):
                r = SG.execute(objs, s, goal, sel, max_calls=6 * len(goal), max_steps=16 * len(objs))
                self.assertTrue(r["success"], (sel.name, sorted(s), r["reason"]))
                n_runs += 1
        self.assertGreater(n_runs, 1000)

    def test_structure_and_strata(self):
        objs = ["a", "b", "c", "d", "e", "f", "g"]
        goal = fs(("on", "b", "a"), ("on", "c", "b"), ("on", "e", "d"))
        st = SG.structure(goal, objs)
        self.assertEqual((st["heights"], st["k"], st["hmax"]), ([3, 2, 1, 1], 2, 3))
        self.assertEqual(SG.stratum_of(st["heights"]), "S")
        self.assertEqual(SG.stratum_of((2, 2, 2)), "T")
        self.assertEqual(SG.stratum_of((5, 1)), "H")
        self.assertEqual(SG.stratum_of((5, 2, 2)), "TH")
        self.assertEqual(len(SG.partitions(7)), 15)
        self.assertEqual(sorted(SG.partitions_in(6, "T")), [(2, 2, 2)])
        rng = random.Random(TEST_RNG)
        g = SG.make_goal(rng, objs, (4, 2, 1))
        self.assertEqual(SG.structure(g, objs)["heights"], [4, 2, 1])


if __name__ == "__main__":
    unittest.main()
