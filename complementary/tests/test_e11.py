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


class TestCustodyAndLeakage(unittest.TestCase):
    """Data-custody fixes: renaming-canonical row keys (E11b split hygiene), the teacher-row filter, the runner's
    output-directory rule, the BWM bytecode record and the seed block (runner checks parsed with ast: no torch)."""
    RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "RCX-E11",
                          "run_e11.py")

    def _runner_ast(self):
        import ast
        with open(self.RUNNER, encoding="utf-8") as fh:
            return ast.parse(fh.read())

    def test_canonical_row_key_is_exact_renaming_class(self):
        objs1, objs2 = ["a", "b", "c", "d"], ["p", "q", "r", "s"]
        pi = {"a": "r", "b": "p", "c": "s", "d": "q"}  # not order-preserving
        rows = SP.get_space(4).states
        for heights in ((2, 2), (2, 1, 1), (3, 1)):
            g1 = SG.make_goal(random.Random(TEST_RNG), objs1, heights)
            g2 = frozenset((f[0], pi[f[1]], pi[f[2]]) for f in g1)
            k1, s1 = {}, {}
            k2, s2 = {}, {}
            for r in rows:
                st = SP.to_facts(tuple(int(v) for v in r), objs1)
                k1[st] = SG.canonical_row_key(st, g1, objs1)
                s1[st] = B.signature(objs1, st, g1)
                st2 = frozenset((f[0],) + tuple(pi[x] for x in f[1:]) for f in st)
                self.assertEqual(SG.canonical_row_key(st2, g2, objs2), k1[st], heights)  # renaming invariant
                st3 = SP.to_facts(tuple(int(v) for v in r), objs2)
                k2[st3] = SG.canonical_row_key(st3, g2, objs2)
                s2[st3] = B.signature(objs2, st3, g2)
            # key equality <=> BWM renaming-signature equality, across the two instances of one goal class
            for a in k1:
                for b in k2:
                    self.assertEqual(k1[a] == k2[b], s1[a] == s2[b], (heights, sorted(a), sorted(b)))
            # the leakage the runner flags: two instances of one class have the SAME row classes (n = 4: all of them)
            he1 = {k1[s] for s in k1 if HE in s and not B.goal_met(s, g1)}
            he2 = {k2[s] for s in k2 if HE in s and not B.goal_met(s, g2)}
            self.assertEqual(he1, he2)
            self.assertGreater(len(he1), 10)
        g22 = SG.make_goal(random.Random(TEST_RNG), objs1, (2, 2))
        g211 = SG.make_goal(random.Random(TEST_RNG), objs1, (2, 1, 1))
        a = {SG.canonical_row_key(SP.to_facts(tuple(int(v) for v in r), objs1), g22, objs1) for r in rows}
        b = {SG.canonical_row_key(SP.to_facts(tuple(int(v) for v in r), objs1), g211, objs1) for r in rows}
        self.assertFalse(a & b)  # different goal structures never share a row key

    def test_teacher_rows_strict_filter_and_counts(self):
        objs = ["a", "b", "c"]
        init = B.towers_to_state([["a"], ["b"], ["c"]])
        goal = fs(("on", "a", "b"))
        good = [("pick-up", "a"), ("stack", "a", "b")]
        traj, s = [], init
        for a in good:
            s = B.step(s, a)[0]
            traj.append(sorted(map(list, s)))
        J = lambda x: sorted(map(list, x))  # noqa: E731

        def row(i, verified, plan=good, **kw):
            return dict({"id": f"distill/{i}", "n": 3, "objs": objs, "init": J(init), "goal": J(goal),
                         "plan": [list(a) for a in plan], "f_states": traj, "verified": verified}, **kw)

        rows = [row(0, True), row(1, "true"), row(2, 1), row(3, False, exact_valid_for_analysis_only=True),
                row(4, None), row(5, True, plan=[("pick-up", "a"), ("put-down", "a")])]
        plans, cust = MA.teacher_plans(rows, B, expected={"rows": 6, "verified": 2})
        self.assertEqual([p["id"] for p in plans], ["distill/0"])
        self.assertEqual((cust["not_verified_skipped"], cust["verified_but_exact_invalid_excluded"],
                          cust["f_states_match_exact"]), (4, 1, 1))
        with self.assertRaises(MA.CustodyError):
            MA.teacher_plans(rows, B, expected={"rows": 6, "verified": 3})
        with self.assertRaises(MA.CustodyError):
            MA.teacher_plans([row(0, True), row(0, True)], B)

    def test_runner_output_dir_rule(self):
        import ast
        from pathlib import Path
        tree = self._runner_ast()
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "out_dir_problem")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), self.RUNNER, "exec"), ns)
        f = ns["out_dir_problem"]
        root = Path("C:/W/complementary")
        allowed, full = root / "runs" / "RCX-E11", root / "runs" / "RCX-E11" / "main"
        bwm, rcwm = Path("C:/W/Rineya_BWM_001_Research_Bundle"), Path("C:/W/RCWM_bundle")
        forbidden = [bwm, rcwm, root / "experiments", root / "runs" / "RCX-E07"]
        self.assertIsNone(f(full, False, allowed, forbidden, full))
        self.assertIsNone(f(allowed / "smoke", True, allowed, forbidden, full))
        for p, smoke in ((bwm / "native_v1" / "runs" / "x", True), (rcwm / "x", True), (allowed, True),
                         (root / "runs" / "RCX-E07" / "x", True), (root / "experiments" / "RCX-E11" / "x", True),
                         (Path("C:/W/elsewhere"), True), (allowed / "smoke", False)):
            self.assertIsNotNone(f(p, smoke, allowed, forbidden, full), (p, smoke))

    def test_runner_records_bwm_bytecode_in_finally_and_seeds_in_block(self):
        import ast
        tree = self._runner_ast()
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        tries = [n for n in ast.walk(main) if isinstance(n, ast.Try) and n.finalbody]
        called = {c.func.id for t in tries for s in t.finalbody for c in ast.walk(s)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        self.assertIn("record_bwm_bytecode", called)
        consts = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", None) in (
                    "SEED_BLOCK", "SEEDS", "DERIVED"):
                consts[n.targets[0].id] = ast.literal_eval(n.value)
        lo, hi = consts["SEED_BLOCK"]
        self.assertEqual((lo, hi), (20261256, 20261259))
        self.assertTrue(all(lo <= v <= hi for v in consts["SEEDS"].values()))
        self.assertTrue(all(int(v.split(":")[0]) in range(lo, hi + 1) for v in consts["DERIVED"].values()))


class TestComputeAccounting(unittest.TestCase):
    """h_calls accounting (each state scored at most once per run, so a per-run A3 hook's calls = evaluations) and the
    compute-matched view derived from expansion-budget runs (macros.compute_matched: exact or censored)."""
    F = TestMacroSearch
    LIB = [MA.Macro(BUILD2, ((0, 1), (2, 0))), MA.Macro(UNSTACK_PUT), MA.Macro(HOLD_BUILD, ((0, 1), (2, 0)))]
    ALGOS = (("gbfs", MA.gbfs_macro), ("beam4", MA.beam_macro))

    def run_(self, fn, lib, budget, h=H.misplaced_blocks):
        return fn(self.F.OBJS, self.F.INIT, self.F.GOAL, h, lib, max_expansions=budget)

    def test_each_state_scored_once_so_h_calls_counts_evaluations(self):
        for lib in ((), self.LIB):
            for name, fn in self.ALGOS:
                scored = []

                def h(s, goal):
                    scored.append(s)
                    return H.misplaced_blocks(s, goal)

                r = self.run_(fn, lib, 400, h)
                self.assertTrue(r["success"], name)
                self.assertEqual(len(scored), r["h_calls"], (name, len(lib)))
                self.assertEqual(len(set(scored)), len(scored), (name, len(lib)))  # a per-run cache never hits

    def test_compute_matched_agrees_with_truncated_runs_and_flags_censoring(self):
        for lib in ((), self.LIB):
            for name, fn in self.ALGOS:
                full = self.run_(fn, lib, 400)
                self.assertTrue(full["success"], name)
                parts = [self.run_(fn, lib, e) for e in range(full["expansions"])]  # each a prefix of `full`
                self.assertTrue(all(p["reason"] == "budget" and not p["success"] for p in parts))
                for field in MA.COMPUTE_FIELDS:
                    v = full[field]
                    self.assertIs(MA.compute_matched(full, field, v), True)
                    self.assertIs(MA.compute_matched(full, field, v - 1), False)
                    censored = exact = 0
                    for p in parts:
                        for t in range(v + 3):
                            d = MA.compute_matched(p, field, t)
                            if d is None:  # the expansion budget bound first: unknown, never a failure
                                censored += 1
                                self.assertLessEqual(p[field], t)
                            else:  # exact: equals the outcome derived from the longer run
                                exact += 1
                                self.assertEqual(d, MA.compute_matched(full, field, t), (name, field, t))
                    self.assertGreater(censored, 0)
                    self.assertGreater(exact, 0)

    def test_compute_matched_cases_and_counts(self):
        def rec(success, reason, v):
            return {"success": success, "reason": reason, "h_calls": v, "prim_steps": v}

        cm = MA.compute_matched
        self.assertIs(cm(rec(True, "goal", 1000), "h_calls", 1000), True)
        self.assertIs(cm(rec(True, "goal", 1001), "h_calls", 1000), False)
        self.assertIs(cm(rec(False, "exhausted", 50), "h_calls", 1000), False)
        self.assertIsNone(cm(rec(False, "budget", 800), "h_calls", 1000))
        self.assertIsNone(cm(rec(False, "budget", 1000), "prim_steps", 1000))
        self.assertIs(cm(rec(False, "budget", 1001), "h_calls", 1000), False)
        self.assertIs(cm(rec(False, "goal", 10), "h_calls", 1000), False)  # found but invalid (gate G2)
        recs = [rec(True, "goal", 900), rec(True, "goal", 1100), rec(False, "budget", 700), rec(False, "exhausted", 5)]
        self.assertEqual(MA.compute_matched_counts(recs, "h_calls", 1000),
                         {"n": 4, "success": 1, "censored": 1, "exact": False})
        with self.assertRaises(ValueError):
            cm(rec(True, "goal", None), "h_calls", 10)  # counters of a derived (smaller-budget) record
        with self.assertRaises(ValueError):
            cm(rec(True, "goal", 1), "seconds", 10)

    def test_runner_builds_a_fresh_heuristic_per_search_run(self):
        """run_e11.phase_a_search (read as source, no torch): make_heuristic is called inside the per-run loop, and the
        compute-matched view goes through macros.compute_matched (censoring), not a bare h_calls <= t count."""
        import ast
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "experiments" / "RCX-E11" / "run_e11.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        loops = [n for n in ast.walk(fns["phase_a_search"]) if isinstance(n, ast.For)]
        run_loop = next(lp for lp in loops if any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "a_record"
                                                  for c in ast.walk(lp)))
        self.assertTrue(any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "make_heuristic"
                            for c in ast.walk(run_loop)))
        view = ast.unparse(fns["compute_matched_view"])
        self.assertIn("MA.compute_matched(", view)
        self.assertIn("MA.compute_matched_counts(", view)


class TestAchieverLevelValues(unittest.TestCase):
    """subgoal.achiever_search / achiever_q / achiever_label_set / AchieverOracle against an independent Dijkstra over
    the full ACHIEVE-call graph of every hand-empty 5-block state (no heuristic), and the label-misalignment hand case."""

    def exact_values(self, objs, states, goal):
        """V_A for every hand-empty state: backward Dijkstra over the ACHIEVE-call graph (independent of achiever_search)."""
        import heapq
        rev, V, heap = {}, {}, []
        for s in states:
            if HE not in s:
                continue
            if B.goal_met(s, goal):
                V[s] = 0
                heap.append((0, sorted(s), s))
                continue
            for f in SG.unmet_facts(s, goal):
                acts, v = SG.achieve(B, s, f)
                rev.setdefault(v, []).append((len(acts), s))
        heapq.heapify(heap)
        done = set()
        while heap:
            d, _, v = heapq.heappop(heap)
            if v in done:
                continue
            done.add(v)
            for c, u in rev.get(v, ()):
                if d + c < V.get(u, float("inf")):
                    V[u] = d + c
                    heapq.heappush(heap, (d + c, sorted(u), u))
        return V

    def test_values_labels_and_oracle_equal_exhaustive_dijkstra(self):
        objs, states = all_states(5)
        for goal in (fs(("on", "b", "a"), ("on", "c", "b"), ("on", "e", "d")),
                     fs(("on", "b", "a"), ("on", "c", "b"), ("on", "d", "c"), ("on", "e", "d"))):
            V = self.exact_values(objs, states, goal)
            L = SP.GoalLabels(objs, goal)
            checked, strictly_better_than_bu = 0, 0
            for k, s in enumerate(sorted(V, key=sorted)):
                if V[s] == 0 or k % 5:
                    continue
                r = SG.achiever_search(objs, s, goal)
                self.assertEqual(r["value"], V[s], sorted(s))
                self.assertEqual(len(r["plan"]), V[s])
                self.assertTrue(B.execute(s, goal, r["plan"])["success"])
                self.assertGreaterEqual(V[s], L.h(s))
                q = SG.achiever_q(objs, s, goal)
                want = {f: len(SG.achieve(B, s, f)[0]) + V[SG.achieve(B, s, f)[1]] for f in SG.unmet_facts(s, goal)}
                self.assertEqual(q, want)
                self.assertEqual(SG.achiever_label_set(objs, s, goal),
                                 frozenset(f for f, v in want.items() if v == min(want.values())))
                ex = SG.execute(objs, s, goal, SG.AchieverOracle(), max_calls=6 * len(goal), max_steps=16 * 5)
                self.assertTrue(ex["success"])
                self.assertEqual(ex["plan_len"], V[s])
                bu = SG.execute(objs, s, goal, SG.BottomUp(), max_calls=6 * len(goal), max_steps=16 * 5)
                self.assertGreaterEqual(bu["plan_len"], V[s])
                strictly_better_than_bu += bu["plan_len"] > V[s]
                checked += 1
            self.assertGreater(checked, 50)
            self.assertGreater(strictly_better_than_bu, 0)  # bottom-up is not the achiever-level optimum
        flat = B.towers_to_state([[o] for o in objs])
        self.assertIsNone(SG.achiever_search(objs, flat, goal, max_expansions=0)["value"])  # over budget -> None

    def test_optimal_plan_labels_can_be_disjoint_from_achiever_labels(self):
        """Optimal primitive plans first unstack d and c, re-make on(d, c) and stack b on d (A(s) = {on(b, d)}, h* = 10);
        the achiever cannot re-make on(d, c) on the way, so on(b, d) first costs 14 and on(a, e) first costs 12."""
        objs = ["a", "b", "c", "d", "e"]
        s = B.towers_to_state([["a", "b", "c", "d"], ["e"]])
        goal = fs(("on", "d", "c"), ("on", "b", "d"), ("on", "a", "e"))
        L = SP.GoalLabels(objs, goal)
        self.assertEqual(L.h(s), 10)
        self.assertEqual(SG.label_set(L, L.sid(s)), fs(("on", "b", "d")))
        self.assertEqual(SG.achiever_q(objs, s, goal), {("on", "a", "e"): 12, ("on", "b", "d"): 14})
        self.assertEqual(SG.achiever_label_set(objs, s, goal), fs(("on", "a", "e")))
        bu = SG.execute(objs, s, goal, SG.BottomUp(), max_calls=18, max_steps=80)
        self.assertEqual((bu["plan_len"], bu["choices"][0]), (14, "on b d"))  # BU's first choice is the plan label
        orc = SG.execute(objs, s, goal, SG.AchieverOracle(), max_calls=18, max_steps=80)
        self.assertEqual((orc["plan_len"], orc["choices"][0]), (12, "on a e"))


class TestMacroControlsAndSearchInvariants(unittest.TestCase):
    """Internal no-op pairs (dominated macros that can carry an excluded pattern into a control), goal-annotation
    alignment, the goal-gated single-move control, and macro-search invariants (duplicate library entries, goal reached
    inside a macro, budget derivation of the full record)."""
    F = TestMacroSearch
    LIB = [MA.Macro(BUILD2, ((0, 1), (2, 0))), MA.Macro(UNSTACK_PUT), MA.Macro(HOLD_BUILD, ((0, 1), (2, 0)))]
    PADDED_GOAL_MOVE = (("pick-up", 0), ("put-down", 0), ("pick-up", 1), ("stack", 1, 2))  # dry-run S macro

    @staticmethod
    def reduce(steps):
        steps = list(steps)
        while MA.noop_pairs(steps):
            k = MA.noop_pairs(steps)[0]
            del steps[k:k + 2]
        return tuple(steps)

    def test_noop_pairs_dominated_and_pair_free_patterns_always_change_state(self):
        pats = [p for L in (2, 3, 4) for V in (1, 2, 3, 4) for p in MA.pattern_universe(L, V)]
        padded = [p for p in pats if MA.noop_pairs(p)]
        self.assertEqual((len(pats), len(padded)), (162, 98))
        for p in pats:
            if not MA.noop_pairs(p):
                m = MA.Macro(p)
                self.assertTrue(m.pre & m.dele, MA.step_key(p))  # every executable binding changes the state
        objs, states = all_states(4)
        idx = [MA.StateIndex(s) for s in states]
        for p in padded:
            full, r = MA.Macro(p), self.reduce(p)
            red = MA.Macro(MA.canonical_steps(r)) if r else None
            for s, si in zip(states, idx):
                ends = {full.apply_symbolic(s, b) for b in full.bindings(si)}
                base = {s} if red is None else {red.apply_symbolic(s, b) for b in red.bindings(si)}
                if ends:
                    self.assertLessEqual(ends, base, (MA.step_key(p), sorted(s)))
        # the dry-run S macro only ever generates end states of M's `pick-up ?0; stack ?0 ?1 | on(?0,?1)`
        objs6, states6 = all_states(6)
        goal = fs(("on", objs6[1], objs6[0]), ("on", objs6[2], objs6[1]), ("on", objs6[4], objs6[3]))
        gidx = MA.goal_index(goal)
        s8, m4 = MA.Macro(self.PADDED_GOAL_MOVE, ((1, 2),)), MA.Macro((("pick-up", 0), ("stack", 0, 1)), ((0, 1),))
        equal = 0
        for s in states6[::4]:
            si = MA.StateIndex(s)
            e8 = {s8.apply_symbolic(s, b) for b in s8.bindings(si, gidx)}
            e4 = {m4.apply_symbolic(s, b) for b in m4.bindings(si, gidx)}
            self.assertLessEqual(e8, e4)
            equal += bool(e8) and e8 == e4
        self.assertGreater(equal, 20)

    def test_selectors_exclude_noop_candidates(self):
        vc = MA.ValidationCache(n_check_max=4)
        mined = [MA.Macro((("pick-up", 0), ("stack", 0, 1)), ((0, 1),))]

        def cand(steps, gon, occ, sup):
            return {"key": MA.macro_key(steps, gon), "step_key": MA.step_key(steps), "steps": steps, "goal_on": gon,
                    "L": len(steps), "nvars": MA.Macro(steps).nvars, "occurrences": occ, "support": sup,
                    "score": occ * (len(steps) - 1)}

        padded = cand(self.PADDED_GOAL_MOVE, ((1, 2),), 90, 30)
        clean = cand((("put-down", 0), ("unstack", 1, 2), ("put-down", 1), ("pick-up", 2)), (), 10, 6)
        ranked = MA.rank({c["key"]: c for c in (padded, clean)})
        templ = [MA.Macro(BUILD2, ((0, 1), (2, 0)))]
        lib, _ = MA.select_matched_library(ranked, templ, 5, vc, exclude_step_keys={m.step_key for m in mined})
        self.assertEqual(lib[0].key, padded["key"])  # the syntactic signature rule lets the padded copy in
        lib, log = MA.select_matched_library(ranked, templ, 5, vc, exclude_step_keys={m.step_key for m in mined},
                                             exclude_noop=True)
        self.assertEqual(lib[0].key, clean["key"])
        self.assertEqual(log[-1]["skipped_internal_noop_keys"], [padded["key"]])
        lib, log = MA.select_library(ranked, 1, 5, vc, exclude_noop=True)
        self.assertEqual([m.key for m in lib], [clean["key"]])
        self.assertEqual(log[0]["decision"], "excluded_internal_noop")
        templates = [MA.Macro(UNSTACK_PUT), MA.Macro(HOLD_BUILD, ((0, 1), (2, 0)))]  # L <= 3: cheap universes
        kw = dict(exclude_keys={m.key for m in templates}, exclude_step_keys={m.step_key for m in templates})
        r0, _ = MA.random_library(templates, random.Random(TEST_RNG), vc, **kw)
        r0b, _ = MA.random_library(templates, random.Random(TEST_RNG), vc, exclude_noop=False, **kw)
        self.assertEqual([m.key for m in r0], [m.key for m in r0b])  # default = the original rule and stream
        r1, rlog = MA.random_library(templates, random.Random(TEST_RNG), vc, exclude_noop=True, **kw)
        self.assertFalse(any(MA.noop_pairs(m.steps) for m in r1))
        self.assertEqual([(m.L, len(m.goal_on)) for m in r1], [(t.L, len(t.goal_on)) for t in templates])
        self.assertTrue(all(vc(m)["accepted"] for m in r1))
        self.assertGreater(sum(lg["skipped"].get("internal_noop", 0) for lg in rlog), 0)

    def test_annotation_alignment_classes(self):
        A = lambda steps, gon: MA.annotation_alignment(MA.Macro(steps, gon))["literals"]  # noqa: E731
        self.assertEqual(A(BUILD2, ((0, 1), (2, 0))), ["established", "established"])
        self.assertEqual(A((("pick-up", 0), ("stack", 0, 1), ("pick-up", 2)), ((0, 1), (2, 0))),
                         ["established", "pending"])
        self.assertEqual(A(UNSTACK_PUT, ((0, 1),)), ["destroyed"])
        self.assertEqual(A(UNSTACK_PUT, ((1, 0),)), ["unrelated"])
        # dry-run R macro: fires only where the goal fact on(?0, ?1) already holds, and breaks it
        self.assertEqual(A((("unstack", 0, 1), ("put-down", 0), ("unstack", 1, 2)), ((0, 1), (2, 0))),
                         ["destroyed", "unrelated"])
        # unstacking a satisfied goal fact is 'destroyed' even though the macro ends holding the block over its support
        self.assertEqual(A((("put-down", 0), ("unstack", 1, 2)), ((1, 2),)), ["destroyed"])
        a = MA.annotation_alignment(MA.Macro(BUILD2, ((0, 1), (2, 0))))
        self.assertEqual((a["established"], a["aligned"], a["destroyed"]), (2, 2, 0))

    def test_goal_move_control_is_sound_aligned_and_usable(self):
        lib = MA.goal_move_library()
        self.assertEqual([m.key for m in lib], ["pick-up ?0; stack ?0 ?1 | on(?0,?1)",
                                                "unstack ?0 ?1; stack ?0 ?2 | on(?0,?2)"])
        for m in lib:
            self.assertTrue(MA.validate(m, n_check=4)["accepted"], m.key)
            a = MA.annotation_alignment(m)
            self.assertEqual((a["established"], a["aligned"]), (len(m.goal_on), len(m.goal_on)))
            self.assertFalse(MA.noop_pairs(m.steps))
        for fn in (MA.gbfs_macro, MA.beam_macro):
            r = fn(self.F.OBJS, self.F.INIT, self.F.GOAL, H.misplaced_blocks, lib, max_expansions=400)
            self.assertTrue(r["success"])
            self.assertEqual(r["sym_mismatch"], 0)

    def test_duplicate_library_entries_only_add_bindings(self):
        keys = ("success", "found", "plan", "plan_len", "expansions", "generated", "h_calls", "macro_children",
                "macro_edges", "macro_uses", "goal_mid_macro", "reason")
        for fn in (MA.gbfs_macro, MA.beam_macro):
            a = fn(self.F.OBJS, self.F.INIT, self.F.GOAL, H.misplaced_blocks, self.LIB, max_expansions=400)
            b = fn(self.F.OBJS, self.F.INIT, self.F.GOAL, H.misplaced_blocks, self.LIB + self.LIB, max_expansions=400)
            self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})  # copies only generate duplicates
            self.assertGreater(b["macro_bindings"], a["macro_bindings"])
            self.assertGreater(b["prim_steps"], a["prim_steps"])

    def test_goal_reached_inside_a_macro(self):
        objs = ["a", "b", "c"]
        init = B.towers_to_state([["a"], ["b"], ["c"]])
        goal = fs(("on", "a", "b"))
        lib = [MA.Macro((("pick-up", 0), ("stack", 0, 1), ("pick-up", 2)), ((0, 1),))]
        for fn in (MA.gbfs_macro, MA.beam_macro):
            r = fn(objs, init, goal, H.h0, lib, max_expansions=10)
            self.assertTrue(r["success"])
            self.assertEqual(r["plan"], [("pick-up", "a"), ("stack", "a", "b")])  # truncated at the goal
            self.assertEqual((r["expansions"], r["goal_mid_macro"], r["macro_edges"], r["plan_len"]), (1, 1, 1, 2))

    def test_budget_derivation_reproduces_direct_reruns(self):
        """run_e11.at_budget's rule on macro search: at b >= the solving expansion the run (plan and counters) is
        identical; below it, a budget failure with expansions = b."""
        keys = ("success", "found", "plan", "expansions", "generated", "h_calls", "prim_steps", "macro_edges", "reason")
        for fn in (MA.gbfs_macro, MA.beam_macro):
            big = fn(self.F.OBJS, self.F.INIT, self.F.GOAL, H.misplaced_blocks, self.LIB, max_expansions=400)
            self.assertEqual(big["reason"], "goal")
            for b in range(big["expansions"] + 3):
                r = fn(self.F.OBJS, self.F.INIT, self.F.GOAL, H.misplaced_blocks, self.LIB, max_expansions=b)
                if b >= big["expansions"]:
                    self.assertEqual({k: r[k] for k in keys}, {k: big[k] for k in keys}, b)
                else:
                    self.assertEqual((r["success"], r["reason"], r["expansions"]), (False, "budget", b))

    def test_runner_wires_noop_flag_alignment_and_goal_move_arm(self):
        """run_e11 read as source (no torch): the three selectors get exclude_noop from cfg['exclude_noop_macros'],
        FULL declares the flag, the G library is added only when 'G' is an arm, and the library descriptors carry
        no-op pairs and annotation alignment (functions executed from the source with macros as MA)."""
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / "experiments" / "RCX-E11" / "run_e11.py").read_text(
            encoding="utf-8"))
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = {c.func.attr: c for c in ast.walk(fns["phase_a_mine"]) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute) and getattr(c.func.value, "id", None) == "MA"}
        for sel in ("select_library", "random_library", "select_matched_library"):
            kw = {k.arg: ast.unparse(k.value) for k in calls[sel].keywords}
            self.assertEqual(kw.get("exclude_noop"), "xn", sel)
        src = ast.unparse(fns["phase_a_mine"])
        self.assertIn("xn = cfg['exclude_noop_macros']", src)
        self.assertIn("if 'G' in cfg['arms']:", src)
        full = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "FULL")
        self.assertIn("exclude_noop_macros", {k.arg for k in full.value.keywords})
        ns = {"MA": MA}
        exec(compile(ast.Module(body=[fns["describe_lib"], fns["alignment_summary"]], type_ignores=[]), "run_e11",
                     "exec"), ns)
        lib = [MA.Macro(BUILD2, ((0, 1), (2, 0))), MA.Macro(self.PADDED_GOAL_MOVE, ((1, 2),)),
               MA.Macro(UNSTACK_PUT, ((0, 1),))]
        d = ns["describe_lib"](lib)
        self.assertEqual([x["noop_pairs"] for x in d], [[], [0], []])
        self.assertEqual(d[2]["alignment"]["literals"], ["destroyed"])
        s = ns["alignment_summary"](lib)
        self.assertEqual((s["literals"], s["established"], s["destroyed"], s["aligned"], s["noop_macros"]),
                         (4, 3, 1, 3, 1))


class TestAchieverLabelWiring(unittest.TestCase):
    """E11b label wiring (reconciliation): subgoal.row_label_sets returns both exact sets; run_e11.b_rows (executed from
    the source with stub R/EX, no torch) stores both, picks the training target from cfg['b_label_target'] and flags
    rows seen in an earlier split; phase_b_data computes OPT_A; headroom_closed is the documented fraction."""
    RUNNER = TestCustodyAndLeakage.RUNNER

    def _fns(self):
        import ast
        with open(self.RUNNER, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def test_row_label_sets_equal_both_definitions(self):
        objs, states = all_states(4)
        goal = fs(("on", "b", "a"), ("on", "d", "c"))
        L, cache = SP.GoalLabels(objs, goal), {}
        k = 0
        for i in SG.handempty_ids(L):
            s = L.state(i)
            got = SG.row_label_sets(L, i, objs, goal, cache=cache)
            self.assertEqual(got, {"plan": SG.label_set(L, i), "achiever": SG.achiever_label_set(objs, s, goal)})
            k += 1
        self.assertGreater(k, 20)
        objs5 = ["a", "b", "c", "d", "e"]  # the hand case where the two sets are disjoint
        s = B.towers_to_state([["a", "b", "c", "d"], ["e"]])
        goal5 = fs(("on", "d", "c"), ("on", "b", "d"), ("on", "a", "e"))
        L5 = SP.GoalLabels(objs5, goal5)
        self.assertEqual(SG.row_label_sets(L5, L5.sid(s), objs5, goal5),
                         {"plan": fs(("on", "b", "d")), "achiever": fs(("on", "a", "e"))})
        self.assertEqual(SG.LABEL_TARGETS, ("plan", "achiever"))

    def test_runner_b_rows_stores_both_sets_and_flags_seen_rows(self):
        import ast
        from collections import Counter
        _, fns = self._fns()

        class Ctx:
            def __init__(self, target):
                self.cfg = {"b_label_target": target, "b_ach_budget": 200_000}

            def courtesy(self):
                pass

        class StubR:
            @staticmethod
            def fs_json(s):
                return sorted(map(list, s))

        class StubEX:
            @staticmethod
            def row_exclusion_reason(objs, s, goal, ex):
                return None

        ns = {"random": random, "Counter": Counter, "SP": SP, "SG": SG, "R": StubR, "EX": StubEX,
              "DERIVED": {"b_states": "20261256:tests:states:{split}"}}
        exec(compile(ast.Module(body=[fns["b_rows"]], type_ignores=[]), self.RUNNER, "exec"), ns)
        b_rows = ns["b_rows"]
        rng = random.Random(TEST_RNG)
        insts = []
        for k in range(2):  # two instances of one goal class (n = 4): the second's rows are renamings of the first's
            objs, init, goal = None, None, None
            while objs is None:
                g = SG.make_instance(rng, 4, (2, 2))
                if g is not None:
                    objs, init, goal = g
            insts.append({"id": f"t{k}", "n": 4, "objs": objs, "init": init, "goal": goal})
        for target, key in (("plan", "labels_plan"), ("achiever", "labels_ach")):
            tr, st_tr, keys_tr = b_rows(Ctx(target), "train", insts[:1], None, 1000)
            va, st_va, _ = b_rows(Ctx(target), "val", insts[1:], None, 1000, frozenset(keys_tr))
            self.assertTrue(tr and va)
            self.assertFalse(any(r["seen"] for r in tr))
            self.assertTrue(all(r["seen"] for r in va))  # n = 4: every row class already occurs in train
            self.assertEqual((st_va["rows_seen"], st_va["rows"]), (len(va), len(va)))
            for r in tr + va:
                self.assertEqual(r["labels"], r[key])
                self.assertTrue(r["labels_plan"] and r["labels_ach"])
                self.assertTrue(all(f in r["cands"] for f in r["labels_plan"] + r["labels_ach"]))
            self.assertEqual(st_tr.get("plan_eq_ach_rows", 0), sum(r["labels_plan"] == r["labels_ach"] for r in tr))
        with self.assertRaises(ValueError):
            b_rows(Ctx("both"), "train", insts[:1], None, 10)

    def test_headroom_closed_and_runner_wiring(self):
        import ast
        import math
        tree, fns = self._fns()
        ns = {"math": math}
        exec(compile(ast.Module(body=[fns["headroom_closed"]], type_ignores=[]), self.RUNNER, "exec"), ns)
        f = ns["headroom_closed"]
        self.assertAlmostEqual(f([20, 40], [16, 30], [16, 30]), 1.0)
        self.assertAlmostEqual(f([20, 40], [20, 40], [16, 30]), 0.0)
        self.assertAlmostEqual(f([20], [math.sqrt(20 * 10)], [10]), 0.5)
        self.assertIsNone(f([], [], []))
        self.assertIsNone(f([20], [18], [20]))  # no excess over the ceiling
        data_src = ast.unparse(fns["phase_b_data"])
        self.assertIn("SG.achiever_search(", data_src)
        self.assertIn("opt_ach=", data_src)
        self.assertIn("SG.row_label_sets(", ast.unparse(fns["b_rows"]))
        self.assertIn("cfg['b_label_target'] not in SG.LABEL_TARGETS", ast.unparse(fns["main"]))
        full = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "FULL")
        kw = {k.arg: ast.unparse(k.value) for k in full.value.keywords}
        self.assertEqual(kw.get("b_label_target"), "'achiever'")  # prereg H-b1 (achiever-level training target)
        self.assertEqual(kw.get("b_ach_budget"), "200000")
        ev = ast.unparse(fns["phase_b_eval"])
        self.assertIn("|vs_plan", ev)
        self.assertIn("|vs_achiever", ev)


class TestPreregSettingsAndDecisionRules(unittest.TestCase):
    """The runner's default config equals the prereg's settings (FULL, and SMOKE inherits them), and a_decision /
    b_decision (executed from the runner source, no torch) follow the prereg tables: rows top to bottom, first match
    wins; E11a attribution by discordants; E11b ADVANCE needs L-r0 vs RND < 0 and L-r0 vs RF <= 0."""
    RUNNER = TestCustodyAndLeakage.RUNNER
    THRESHOLDS = [500, 1000, 2500, 5000]

    def _parse(self):
        import ast
        with open(self.RUNNER, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        return tree, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _fn(self, name):
        import ast
        _, fns = self._parse()
        ns = {}
        exec(compile(ast.Module(body=[fns[name]], type_ignores=[]), self.RUNNER, "exec"), ns)
        return ns[name]

    def test_config_defaults_equal_prereg_settings(self):
        import ast
        from pathlib import Path
        tree, _ = self._parse()
        cfgs = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) in ("FULL", "SMOKE")}
        full = {k.arg: k.value for k in cfgs["FULL"].keywords}
        lit = {k: ast.literal_eval(v) for k, v in full.items() if k in (
            "exclude_noop_macros", "b_label_target", "arms", "hcall_thresholds", "prim_thresholds",
            "compute_secondary_threshold", "a_g_clause_margin", "primary_budget", "b_select_unseen_val")}
        self.assertIs(lit["exclude_noop_macros"], True)
        self.assertEqual(lit["b_label_target"], "achiever")
        self.assertEqual(lit["arms"], ["P", "M", "R", "S", "G"])
        self.assertEqual(lit["hcall_thresholds"], self.THRESHOLDS)
        self.assertEqual(lit["prim_thresholds"], self.THRESHOLDS)
        self.assertEqual(lit["compute_secondary_threshold"], 1000)
        self.assertIn(lit["compute_secondary_threshold"], lit["hcall_thresholds"])
        self.assertEqual((lit["a_g_clause_margin"], lit["primary_budget"]), (5, 200))
        self.assertIs(lit["b_select_unseen_val"], True)
        self.assertNotIn("a_attr_margin", full)  # attribution is by discordants (no count margin)
        self.assertEqual(ast.unparse(cfgs["SMOKE"].args[0]), "FULL")
        smoke = {k.arg for k in cfgs["SMOKE"].keywords}
        for k in lit:  # the smoke runs the settings that will be frozen (primary_budget is scaled down)
            if k != "primary_budget":
                self.assertNotIn(k, smoke)
        here = Path(self.RUNNER).parent
        prereg = here / "PREREG_E11.md" if (here / "PREREG_E11.md").exists() else here / "PREREG_E11_DRAFT.md"
        text = prereg.read_text(encoding="utf-8")
        for s in ("`exclude_noop_macros=True`", '`b_label_target="achiever"`', "{500, 1,000, 2,500, 5,000}",
                  "| **G** (descriptive) |", "within 1,000 heuristic calls", "M-only > control-only discordants",
                  "the L-r0 vs RND point estimate < 0", "the L-r0 vs RF point estimate ≤ 0",
                  "Rows are evaluated top to bottom; the first match is the outcome."):
            self.assertIn(s, text, s)

    def test_b_decision_rows_in_order(self):
        import math
        f = self._fn("b_decision")
        cfg = {"model_seeds": [1, 2, 3], "b_advance_log_ratio": math.log(0.95), "b_success_margin": 2,
               "b_stop_success_frac": 0.8}

        def m(diff=-0.08, hi=-0.02, rnd=(-0.3, -0.2), rf=-0.01, sL=90, sBU=90, reps=(-0.05, -0.06)):
            return {"complete": True, "primary": {"diff": diff, "hi": hi},
                    "vs_random_order": {"diff": rnd[0], "hi": rnd[1]}, "vs_ready_first": {"diff": rf},
                    "ood_success": {"learned_s1": sL, "bottom_up": sBU},
                    "replicates_log_ratio_vs_BU": {"learned_s1": diff, "learned_s2": reps[0], "learned_s3": reps[1]}}

        self.assertEqual(f(cfg, {"complete": False})["outcome"], "NOT_RUN")
        self.assertEqual(f(cfg, m())["outcome"], "ADVANCE")
        self.assertEqual(f(cfg, m(rf=0.0))["outcome"], "ADVANCE")  # RF: point estimate <= 0
        d = f(cfg, m(rf=0.02))  # beats BU by the margin but not RF: REVISE (RND CI < 0), with the RF clause
        self.assertEqual((d["outcome"], d["rf_clause"], d["vs_rf_point_le_0"]), ("REVISE", True, False))
        self.assertEqual(f(cfg, m(rf=0.02, rnd=(-0.1, 0.05)))["outcome"], "INCONCLUSIVE")
        # no better than random order although BU is beaten by > 5% (random beats BU): STOP, never ADVANCE
        d = f(cfg, m(rnd=(0.01, 0.1)))
        self.assertEqual((d["outcome"], d["vs_rnd_point_negative"], d["rf_clause"]), ("STOP", False, False))
        self.assertEqual(f(cfg, m(rnd=(0.0, 0.1)))["outcome"], "STOP")
        # first match wins: ADVANCE and STOP's success clause (3 < 0.8 * 5) both hold -> ADVANCE
        self.assertEqual(f(cfg, m(sL=3, sBU=5))["outcome"], "ADVANCE")
        self.assertEqual(f(cfg, m(sL=87))["outcome"], "REVISE")  # success margin missed (87 < 90 - 2)
        self.assertEqual(f(cfg, m(sL=70))["outcome"], "STOP")  # 70 < 0.8 * 90
        self.assertEqual(f(cfg, m(diff=-0.03))["outcome"], "REVISE")  # not >= 5% shorter
        self.assertEqual(f(cfg, m(hi=0.01))["outcome"], "REVISE")
        self.assertEqual(f(cfg, m(reps=(-0.05, 0.01)))["outcome"], "REVISE")
        self.assertEqual(f(cfg, m(diff=-0.03, rnd=(-0.1, 0.05)))["outcome"], "INCONCLUSIVE")

    def test_a_decision_rows_attribution_and_wording_flags(self):
        f = self._fn("a_decision")
        cfg = {"alpha": 0.05, "primary_heur": "A3", "primary_algo": "gbfs", "primary_budget": 200,
               "compute_secondary_threshold": 1000, "a_g_clause_margin": 5}

        def mc(p, co, bo, cs=60, bs=50, exact=True):
            return {"p_exact": p, "p_holm": p, "cand_only": co, "base_only": bo, "cand_success": cs,
                    "base_success": bs, "exact": exact}

        def m(prim=(0.01, 12, 2), attr=((0.01, 9, 1), (0.02, 8, 1)), stop=(False, False), ch=(0.01, 10, 1, True),
              g=(60, 57)):
            out = {"complete": True, "gates": {"ok": True}, "primary": mc(*prim),
                   "attribution": {"M_vs_R": mc(*attr[0]), "M_vs_S": mc(*attr[1])},
                   "stop_check_M_not_better": {"A3|200": stop[0], "A3|2000": stop[1]},
                   "compute_matched_mcnemar": {"M_vs_P|h_calls|A3|gbfs|1000": mc(ch[0], ch[1], ch[2], exact=ch[3]),
                                               "M_vs_P|prim_steps|A3|gbfs|1000": mc(0.5, 3, 2, exact=False)},
                   "secondary_mcnemar": {}}
            if g is not None:
                out["secondary_mcnemar"]["M_vs_G|A3|gbfs|200"] = mc(0.3, 5, 2, cs=g[0], bs=g[1])
            return out

        self.assertEqual(f(cfg, {"complete": False}, True)["outcome"], "NOT_RUN")
        self.assertEqual(f(cfg, m(), False)["outcome"], "INVALID")
        bad = m()
        bad["gates"]["ok"] = False
        self.assertEqual(f(cfg, bad, True)["outcome"], "INVALID")
        d = f(cfg, m(), True)
        self.assertEqual((d["outcome"], d["M_minus_G_primary_cell"], d["g_clause"]), ("ADVANCE", 3, True))
        self.assertTrue(d["compute_matched_secondary"]["saving_claim_allowed"])
        self.assertFalse(d["compute_matched_secondary"]["prim_steps"]["evaluable"])
        d = f(cfg, m(g=(60, 50)), True)
        self.assertEqual((d["outcome"], d["g_clause"]), ("ADVANCE", False))
        self.assertIsNone(f(cfg, m(g=None), True)["g_clause"])  # arm G not configured
        # attribution needs M-only > control-only discordants (no +5 count margin) and Holm p < alpha
        self.assertEqual(f(cfg, m(attr=((0.01, 9, 1), (0.04, 4, 3))), True)["outcome"], "ADVANCE")
        self.assertEqual(f(cfg, m(attr=((0.01, 9, 1), (0.04, 3, 3))), True)["outcome"], "REVISE")
        self.assertEqual(f(cfg, m(attr=((0.01, 9, 1), (0.06, 8, 1))), True)["outcome"], "REVISE")
        self.assertEqual(f(cfg, m(prim=(0.01, 2, 12)), True)["outcome"], "STOP")
        self.assertEqual(f(cfg, m(prim=(0.2, 8, 4), stop=(True, True)), True)["outcome"], "STOP")
        self.assertEqual(f(cfg, m(prim=(0.2, 8, 4), stop=(True, False)), True)["outcome"], "INCONCLUSIVE")
        # first match wins: a significant primary in M's favour is REVISE even if the stop check holds
        self.assertEqual(f(cfg, m(attr=((0.2, 9, 1), (0.2, 8, 1)), stop=(True, True)), True)["outcome"], "REVISE")
        # the compute-matched secondary grades wording only and needs an exact cell
        for ch in ((0.01, 10, 1, False), (0.2, 10, 6, True), (0.01, 1, 10, True)):
            d = f(cfg, m(ch=ch), True)
            self.assertEqual(d["outcome"], "ADVANCE")
            self.assertFalse(d["compute_matched_secondary"]["saving_claim_allowed"], ch)


if __name__ == "__main__":
    unittest.main()
