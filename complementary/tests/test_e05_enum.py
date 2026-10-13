"""Tests for RCX-E05 ENUM± planners and the non-learned references (experiments/RCX-E05/enum_e05.py;
PREREG_E05_DRAFT.md §2.1 test table, §5.2 incl. B1 and L1).

n = 3-5 spaces only, literal seeds 0-9, never reported as results. torch only in TestGnnEnum (a tiny random RelGNN; the
frozen R-04 model is never loaded here).
Run: python -B -m unittest tests.test_e05_enum -v   (CPU, one thread; a few seconds)
"""
import importlib.util
import itertools
import json
import os
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", "RCX-E05", fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FAM = _load("families_e05_ut", "families_e05.py")
EN5 = _load("enum_e05_ut", "enum_e05.py")
COSTS = (0.125, 0.25, 0.75)


def state_of(space, towers, held=None):
    n, pos = space.n, [None] * space.n
    for t in towers:
        pos[t[0]] = n
        for lo, hi in zip(t, t[1:]):
            pos[hi] = lo
    if held is not None:
        pos[held] = n + 1
    return space.index[tuple(pos)]


def wvec(n, items):
    w, fi = np.zeros(n * (n + 1), dtype=np.int64), SO.fact_index(n)
    for f, x in items.items():
        w[fi[f]] = x
    return w


def lin(w, c):
    return SimpleNamespace(rtype="lin", w=tuple(int(x) for x in w), c=float(c), G=(), R=0.0, dyn="free")


def setup_goal(sp, phi, w, c):
    p = lin(w, c)
    return p, SO.sf_table(sp, SO.stop_values(phi, p.w), c)


def pos_of(sp, s):
    return tuple(int(v) for v in sp.states[s])


class TestMenus(unittest.TestCase):
    def test_menu_sizes_and_counts(self):
        self.assertEqual(len(EN5.menu(("on", 0, 1), 7)), 14)
        self.assertEqual(len(EN5.menu(("ontable", 0), 7)), 7)
        self.assertEqual(len(EN5.menu(("clear", 0), 7)), 7)
        self.assertEqual(EN5.menu(("on", 0, 1), 7, "rev2"), (("ontable", 0), ("holding", 0), ("clear", 1)))
        on, tab = 1 + len(EN5.menu(("on", 0, 1), 7)), 1 + len(EN5.menu(("ontable", 0), 7))
        self.assertEqual((2 ** 2 * on ** 2, 2 ** 4 * on ** 2, 2 ** 6, tab), (900, 3600, 64, 8))  # §5.2 bounds

    def test_menus_imply_not_and_cover(self):
        """L1: every menu fact implies not-f, and f is false iff some menu fact holds (all n = 5 states)."""
        n = 5
        sp, fs = SP.get_space(n), SO.facts(n)
        phi = SO.phi_table(sp)
        for i in range(len(sp)):
            pos = pos_of(sp, i)
            for k, f in enumerate(fs):
                self.assertEqual(EN5.holds(f, pos), bool(phi[i, k]))
                M = EN5.menu(f, n)
                for m in M:
                    if EN5.holds(m, pos):
                        self.assertFalse(EN5.holds(f, pos), (f, m, pos))
                self.assertEqual(not EN5.holds(f, pos), any(EN5.holds(m, pos) for m in M))
        # rev 2's 3-fact on-menu did not cover: a on c, b covered by d
        s = state_of(sp, [[2, 0], [1, 3], [4]])
        f = ("on", 0, 1)
        self.assertFalse(EN5.holds(f, pos_of(sp, s)))
        self.assertFalse(any(EN5.holds(m, pos_of(sp, s)) for m in EN5.menu(f, n, "rev2")))


class TestTargets(unittest.TestCase):
    def test_satisfiability_pruning_and_dedup(self):
        n = 4
        sp = SP.get_space(n)
        w = wvec(n, {("on", 0, 1): 2, ("ontable", 0): -1, ("on", 2, 3): -1})
        T = EN5.linear_targets(sp, w)
        self.assertEqual(T.n_raw, 2 * (1 + len(EN5.menu(("ontable", 0), n))) * (1 + len(EN5.menu(("on", 2, 3), n))))
        self.assertEqual(len(set(T.goals)), len(T.goals))
        fs, P = SO.facts(n), {("on", 0, 1): 2.0}
        for G, u in zip(T.goals, T.U0):
            cg = EN5.compile_goal(n, G)
            self.assertTrue(cg.consistent and sp.goal_mask(cg).any())
            self.assertEqual(u, sum(P.get(f, 0.0) for f in G))
        # every literal combination is either kept (as a set) or unsatisfiable
        kept = set(T.goals)
        for gp in ((), (("on", 0, 1),)):
            for r1 in (None,) + EN5.menu(("ontable", 0), n):
                for r2 in (None,) + EN5.menu(("on", 2, 3), n):
                    G = frozenset(gp + tuple(f for f in (r1, r2) if f is not None))
                    cg = EN5.compile_goal(n, G)
                    self.assertEqual(G in kept, bool(cg.consistent and sp.goal_mask(cg).any()))
        two_held = frozenset({("holding", 0), ("holding", 2)})
        self.assertTrue(EN5.compile_goal(n, two_held).consistent)  # pruned by goal_mask, not by consistency
        self.assertNotIn(two_held, kept)
        self.assertNotIn(frozenset({("on", 0, 1), ("holding", 0)}), kept)
        # persistence rule: G u {f} satisfiable unless G falsifies f
        j = T.neg.index(SO.fact_index(n)[("ontable", 0)])
        for G, row in zip(T.goals, T.sat):
            self.assertEqual(row[j], float(not any(f in G for f in EN5.menu(("ontable", 0), n))
                                           and ("on", 0, 1) not in G))

    def test_dedup_equals_literal_enumeration(self):
        """Score = max over the literal (G+, R) enumeration with U = sum_{G+} w + penalties (n = 4, H-NEG)."""
        n, rng = 4, random.Random(0)
        sp = SP.get_space(n)
        phi, fs = SO.phi_table(sp), SO.facts(n)
        for _ in range(4):
            g = FAM.draw_goal("H-NEG", rng, n)
            p, tab = setup_goal(sp, phi, g["w"], g["c"])
            arm = EN5.enum_arm(sp, phi, p, tab, "mis")
            w = np.asarray(p.w, dtype=np.float64)
            P = [fs[k] for k in np.flatnonzero(w > 0)]
            N = [k for k in np.flatnonzero(w < 0)]
            sat = EN5.Satisfiable(sp)
            for s in rng.sample(range(len(sp)), 25):
                pos, best = pos_of(sp, s), -np.inf
                for bits in itertools.product((0, 1), repeat=len(P)):
                    gp = [f for f, b in zip(P, bits) if b]
                    for R in itertools.product(*[(None,) + EN5.menu(fs[k], n) for k in N]):
                        G = frozenset(gp + [f for f in R if f is not None])
                        if not sat(G):
                            continue
                        U = sum(w[SO.fact_index(n)[f]] for f in gp)
                        U += sum(w[k] for k in N if phi[s, k] and sat(G | {fs[k]}))
                        H = [f for f in G if f[0] == "holding"]
                        lb = SP.misplaced_compact(pos, EN5.compile_goal(n, G - set(H)))[1]
                        h = max(lb, EN5.h_hold(pos, H[0][1] if H else -1))
                        best = max(best, U - p.c * (1 + h))
                self.assertEqual(arm.score(s), best)

    def test_holding_targets(self):
        n = 4
        sp = SP.get_space(n)
        w = wvec(n, {("on", 0, 1): -2, ("ontable", 3): 1})
        T = EN5.linear_targets(sp, w)
        G = frozenset({("ontable", 3), ("holding", 0)})
        t = T.goals.index(G)
        self.assertEqual((T.hold[t], T.U0[t]), (0, 1.0))
        self.assertTrue(any(("holding", 1) in g for g in T.goals))
        held0, empty, held2 = (pos_of(sp, state_of(sp, [[1], [2], [3]], held=0)), pos_of(sp, state_of(sp, [[1, 0], [2], [3]])),
                               pos_of(sp, state_of(sp, [[1, 0], [3]], held=2)))
        self.assertEqual([EN5.h_hold(p, 0) for p in (held0, empty, held2)], [0, 1, 2])
        self.assertEqual(EN5.h_hold(empty, -1), 0)
        h = EN5.HMis(sp, T, "mis")
        self.assertEqual(h.value(state_of(sp, [[1, 0], [3]], held=2), t), 2.0)
        self.assertEqual(EN5.HMis(sp, T, "rev2").value(state_of(sp, [[1, 0], [3]], held=2), t), 0.0)  # B1 defect


def has_clear(G) -> bool:
    return any(f[0] == "clear" for f in G)


class TestAdmissible(unittest.TestCase):
    def test_hmis_le_hstar(self):
        """ĥ_MIS(s, G) <= h*(s, G) (backward_bfs) on 500 random (s, G) at n = 5 without `clear` facts, incl. holding
        targets (the prereg's claim); with `clear` facts (on-menu falsifiers clear(b), H-SPAN) ĥ_MIS can exceed h* by
        1 (it is NOT admissible there, a finding for the coordinator), while the corrected ĥ (mis_adm) is <= h* on
        every target."""
        n, rng = 5, random.Random(1)
        sp = SP.get_space(n)
        pool = []
        while len(pool) < 40:
            g = FAM.draw_goal(rng.choice(("H-NEG", "H-NEG", "H-SPAN")), rng, n)
            pool.append(EN5.linear_targets(sp, g["w"]))
        n_hold = n_free = n_clear = over = 0
        while n_free < 500:
            T = rng.choice(pool)
            t = rng.randrange(len(T))
            if rng.random() < 0.5:  # oversample holding targets
                hs = np.flatnonzero(T.hold >= 0)
                t = int(rng.choice(hs)) if len(hs) else t
            s = rng.randrange(len(sp))
            hstar = sp.backward_bfs(sp.goal_mask(T.full[t]))[s]
            self.assertGreaterEqual(hstar, 0)
            vals = {v: EN5.HMis(sp, T, v).value(s, t) for v in ("mis", "mis_adm", "rev2", "mis_noh")}
            self.assertLessEqual(vals["mis_adm"], hstar, (sorted(T.goals[t]), pos_of(sp, s)))
            self.assertLessEqual(vals["mis"] - 1, hstar)
            self.assertLessEqual(vals["mis_adm"], vals["mis"])
            self.assertGreaterEqual(vals["mis"], vals["mis_noh"])
            self.assertEqual(EN5.HStar(sp, T).value(s, t), float(hstar))
            if has_clear(T.goals[t]):
                n_clear += 1
                over += vals["mis"] > hstar
                continue
            self.assertLessEqual(vals["mis"], hstar, (sorted(T.goals[t]), pos_of(sp, s)))
            self.assertLessEqual(vals["rev2"], hstar)
            self.assertEqual(vals["mis_adm"], vals["mis"])  # the correction only touches clear targets
            n_free += 1
            n_hold += T.hold[t] >= 0
        self.assertGreater(n_hold, 150)
        self.assertGreater(n_clear, 50)
        self.assertGreater(over, 0)

    def test_clear_target_counterexample(self):
        """n = 5, tower 4-0-2-1 (bottom to top), 3 on the table: G = {clear(4), on(3,2)}: ĥ_MIS = 8 > h* = 7 (the plan
        ends with unstack(0, 4), keeping 0 in the hand)."""
        sp = SP.get_space(5)
        s = state_of(sp, [[4, 0, 2, 1], [3]])
        G = frozenset({("clear", 4), ("on", 3, 2)})
        T = EN5.Targets(5, [G], [0.0], [], [], np.zeros((1, 0)), 1)
        self.assertEqual(sp.backward_bfs(sp.goal_mask(T.full[0]))[s], 7)
        self.assertEqual(EN5.HMis(sp, T, "mis").value(s, 0), 8.0)
        self.assertEqual(EN5.HMis(sp, T, "mis_adm").value(s, 0), 7.0)

    def test_mis_adm_exhaustive_n4(self):
        """mis_adm <= h* and mis <= h* + 1 at every n = 4 state for every target of 12 random H-NEG/H-SPAN goals."""
        n, rng = 4, random.Random(7)
        sp = SP.get_space(n)
        for _ in range(12):
            T = EN5.linear_targets(sp, FAM.draw_goal(rng.choice(("H-NEG", "H-SPAN")), rng, n)["w"])
            for t in range(len(T)):
                hs = sp.backward_bfs(sp.goal_mask(T.full[t]))
                a, m = EN5.HMis(sp, T, "mis_adm"), EN5.HMis(sp, T, "mis")
                for s in range(len(sp)):
                    self.assertLessEqual(a.value(s, t), hs[s], (sorted(T.goals[t]), pos_of(sp, s)))
                    self.assertLessEqual(m.value(s, t), hs[s] + 1)
                    if not has_clear(T.goals[t]):
                        self.assertLessEqual(m.value(s, t), hs[s])


class TestB1HandCase(unittest.TestCase):
    """§5.2 hand case: n = 4, a on b, c and d on the table, hand empty; on(a,b) = -2, ontable(d) = +1, c' = 0.25."""

    def setUp(self):
        self.sp = sp = SP.get_space(4)
        self.phi = SO.phi_table(sp)
        self.p, self.tab = setup_goal(sp, self.phi, wvec(4, {("on", 0, 1): -2, ("ontable", 3): 1}), 0.25)
        self.s0 = state_of(sp, [[1, 0], [2], [3]])
        self.acts, self.dsts = SO.moves(sp, self.s0)

    def arm(self, kind, version="full"):
        return EN5.enum_arm(self.sp, self.phi, self.p, self.tab, kind, version)

    def test_moves_and_oracle(self):
        self.assertEqual([self.sp.decode_action(int(a)) for a in self.acts],
                         [(SP.PICK, 2, 2), (SP.PICK, 3, 3), (SP.UNSTACK, 0, 1)])
        self.assertEqual((self.tab.stop[self.s0], self.tab.V[self.s0]), (-1.0, 0.75))

    def test_enum_mis_plays_the_optimum(self):
        arm = self.arm("mis")
        sc = arm.move_scores(self.s0, self.acts, self.dsts)
        self.assertEqual(sc.tolist(), [0.25, 0.25, 0.75])
        self.assertFalse((sc == sc[0]).all())
        res = SO.evaluate(self.sp, self.tab, self.s0, self.arm("mis").policy())
        held_a = state_of(self.sp, [[1], [2], [3]], held=0)
        self.assertEqual((res["nreg"], res["T"], res["final"], res["loop"]), (0.0, 1, held_a, False))

    def test_negative_controls_tie(self):
        for version in ("full", "rev2"):  # rev 2's bound ties all three successors (either menu)
            sc = self.arm("rev2", version).move_scores(self.s0, self.acts, self.dsts)
            self.assertEqual(sc.tolist(), [0.75, 0.75, 0.75])
        sc = self.arm("mis_noh").move_scores(self.s0, self.acts, self.dsts)  # G \ H without h_hold
        self.assertEqual(sc.tolist(), [0.75, 0.5, 0.75])
        self.assertEqual(SO.decide(self.tab.stop[self.s0], sc), 1)  # pick-up(c): the wrong move
        self.assertGreater(SO.evaluate(self.sp, self.tab, self.s0, self.arm("mis_noh").policy())["nreg"], 0.0)

    def test_penalty_avoided_vs_positive_only(self):
        """ENUM± removes the penalty; rev 1's positive-only ENUM does not (it loops back and stops: nreg 1)."""
        r1 = SO.evaluate(self.sp, self.tab, self.s0, self.arm("mis", "pos").policy())
        self.assertEqual((r1["nreg"], r1["loop"]), (1.0, True))
        sp = SP.get_space(3)
        phi = SO.phi_table(sp)
        p, tab = setup_goal(sp, phi, wvec(3, {("ontable", 2): -2, ("ontable", 0): 1}), 0.25)
        s0 = state_of(sp, [[0], [1], [2]])
        self.assertEqual((tab.stop[s0], tab.V[s0]), (-1.0, 0.75))
        res = SO.evaluate(sp, tab, s0, EN5.enum_arm(sp, phi, p, tab, "mis").policy())
        self.assertEqual((res["nreg"], res["final"]), (0.0, state_of(sp, [[0], [1]], held=2)))
        r1 = SO.evaluate(sp, tab, s0, EN5.enum_arm(sp, phi, p, tab, "mis", "pos").policy())
        self.assertEqual((r1["nreg"], r1["loop"]), (1.0, True))


class TestNonTie(unittest.TestCase):
    def test_non_tie_on_random_hneg(self):
        """20 random H-NEG weights x 20 random hand-empty starts with >= 2 moves (n = 5): all-tied share <= 0.5 and
        below the rev-2 control's; every weight has a start with >= 2 distinct move scores."""
        n, rng = 5, random.Random(2)
        sp = SP.get_space(n)
        phi = SO.phi_table(sp)
        he = [int(s) for s in np.flatnonzero(SO.handempty_mask(sp)) if len(SO.moves(sp, int(s))[1]) >= 2]
        tied = {"mis": 0, "rev2": 0}
        total = 0
        for _ in range(20):
            g = FAM.draw_goal("H-NEG", rng, n)
            p, tab = setup_goal(sp, phi, g["w"], g["c"])
            arms = {k: EN5.enum_arm(sp, phi, p, tab, k) for k in tied}
            distinct = False
            for s in rng.sample(he, 20):
                acts, dsts = SO.moves(sp, s)
                for k, arm in arms.items():
                    sc = arm.move_scores(s, acts, dsts)
                    tied[k] += int((sc == sc[0]).all())
                    if k == "mis":
                        distinct |= len(set(sc.tolist())) >= 2
                total += 1
            self.assertTrue(distinct)
            self.assertEqual(arms["mis"].stats["multi"], 20)
            self.assertEqual(EN5.tie_share(arms["mis"].stats), arms["mis"].stats["all_tied"] / 20)
        self.assertLessEqual(tied["mis"] / total, 0.5)
        self.assertLess(tied["mis"], tied["rev2"])


class TestReferences(unittest.TestCase):
    def test_hstar_equals_oracle_on_single_target_goals(self):
        """ENUM-H*± move scores = ORACLE's (-c + V*(s')) exactly at every state: single positive facts and H-CONJ."""
        n, rng = 5, random.Random(3)
        sp = SP.get_space(n)
        phi, fs = SO.phi_table(sp), SO.facts(n)
        goals = []
        for _ in range(4):
            k = rng.randrange(n * n)
            goals.append((lin(np.eye(n * (n + 1), dtype=np.int64)[k] * rng.choice((1, 2, 3)), rng.choice(COSTS)),
                          None))
        while len(goals) < 8:
            G = tuple(sorted(rng.sample(range(n * (n - 1)), rng.choice((2, 3)))))
            if FAM.conj_satisfiable(sp, G):
                c = rng.choice(COSTS)
                stop = 3.0 * sp.goal_mask(SO.compile_goal(n, [fs[k] for k in G])).astype(np.float64)
                goals.append((SimpleNamespace(rtype="conj", w=(), c=c, G=G, R=3.0), SO.sf_table(sp, stop, c)))
        for p, tab in goals:
            tab = tab or SO.sf_table(sp, SO.stop_values(phi, p.w), p.c)
            arm = EN5.named_arm("ENUM-HSTAR", sp, phi, p, tab)
            for s in range(len(sp)):
                acts, dsts = SO.moves(sp, s)
                self.assertTrue(np.array_equal(arm.move_scores(s, acts, dsts), tab.V[dsts] - p.c))
            for s0 in FAM.start_pool(sp, tab)[:40].tolist():
                self.assertEqual(SO.evaluate(sp, tab, s0, arm.policy())["nreg"], 0.0)

    def test_positive_only_reduces_to_rev1(self):
        n, rng = 5, random.Random(4)
        sp = SP.get_space(n)
        phi = SO.phi_table(sp)
        for fam in ("TRAIN", "ID", "H-K", "H-COST"):
            g = FAM.draw_goal(fam, rng, n)
            p, tab = setup_goal(sp, phi, g["w"], g["c"])
            full, r1 = EN5.enum_arm(sp, phi, p, tab, "mis"), EN5.enum_arm(sp, phi, p, tab, "mis", "pos")
            self.assertEqual(full.T.goals, r1.T.goals)
            self.assertEqual(EN5.linear_targets(sp, p.w, "rev2").goals, r1.T.goals)
            self.assertEqual(full.T.n_raw, 2 ** sum(1 for x in p.w if x > 0))
            for s in rng.sample(range(len(sp)), 100):
                self.assertEqual(full.score(s), r1.score(s))

    def test_exact_gpi_best_single_and_floors(self):
        n, rng = 5, random.Random(5)
        sp = SP.get_space(n)
        phi = SO.phi_table(sp)
        lib = []
        for _ in range(5):
            g = FAM.draw_goal("TRAIN", rng, n)
            lib.append(SO.sf_table(sp, SO.stop_values(phi, g["w"]), g["c"]))
        for _ in range(4):
            g = FAM.draw_goal(rng.choice(("ID", "H-NEG")), rng, n)
            p, tab = setup_goal(sp, phi, g["w"], g["c"])
            starts = FAM.start_pool(sp, tab)[:25].tolist()
            for s0 in starts:
                gp = SO.evaluate(sp, tab, s0, EN5.exact_gpi(tab.stop, p.c, lib))
                pol, z = EN5.best_single(tab.stop, p.c, lib, s0)
                bs = SO.evaluate(sp, tab, s0, pol)
                vz = tab.stop[lib[z].end[s0]] - p.c * lib[z].T[s0]
                self.assertEqual((bs["J"], bs["final"], bs["T"]), (vz, int(lib[z].end[s0]), int(lib[z].T[s0])))
                self.assertGreaterEqual(gp["J"], max(vz, tab.stop[s0]))  # Q4
                # with the goal's own table in L both are exact
                self.assertEqual(SO.evaluate(sp, tab, s0, EN5.exact_gpi(tab.stop, p.c, lib + [tab]))["nreg"], 0.0)
                pol, z = EN5.best_single(tab.stop, p.c, lib + [tab], s0)
                self.assertEqual(SO.evaluate(sp, tab, s0, pol)["nreg"], 0.0)
                self.assertEqual(SO.evaluate(sp, tab, s0, EN5.stop_arm())["nreg"], 1.0)
                self.assertEqual(SO.evaluate(sp, tab, s0, EN5.oracle(tab))["nreg"], 0.0)
                gr = SO.evaluate(sp, tab, s0, EN5.greedy(tab.stop, p.c))
                self.assertTrue(0.0 <= gr["nreg"] <= 1.0)

    def test_rand_stream(self):
        self.assertEqual(EN5.derive(20261244, "rand:k:3"), FAM.derive(20261244, "rand:k:3"))
        a, b = EN5.rand_rng("test", "k", 3), FAM.stream(20261244, "rand:test:k:3")  # run_e05's tag
        self.assertEqual([a.random() for _ in range(4)], [b.random() for _ in range(4)])
        self.assertNotEqual(EN5.rand_rng("test", "k", 3).random(), EN5.rand_rng("test", "k", 4).random())
        sp = SP.get_space(4)
        phi = SO.phi_table(sp)
        p, tab = setup_goal(sp, phi, wvec(4, {("on", 0, 1): 2, ("on", 2, 3): 1}), 0.25)
        s0 = FAM.start_pool(sp, tab)[0]
        self.assertEqual(SO.evaluate(sp, tab, s0, EN5.rand("pilot", "k", s0)),
                         SO.evaluate(sp, tab, s0, EN5.rand("pilot", "k", s0)))

    def test_masked_hstar_and_hdyn_enum(self):
        """ENUM-H*± under the height-limit mask uses exact masked distances (vs an explicit BFS)."""
        n, rng = 5, random.Random(6)
        sp = SP.get_space(n)
        phi, m = SO.phi_table(sp), SO.height_limit_mask(sp, 3)
        src = np.repeat(np.arange(len(sp)), np.diff(sp.fwd_ptr))
        g = FAM.draw_goal("H-DYN", rng, n)
        p, _ = setup_goal(sp, phi, g["w"], g["c"])
        T = EN5.linear_targets(sp, p.w)
        h = EN5.HStar(sp, T, m)
        for t in range(len(T)):
            goal = sp.goal_mask(T.full[t])
            d = np.full(len(sp), np.inf)
            d[goal] = 0
            front, k = np.flatnonzero(goal), 0
            while len(front):
                k += 1
                e = m & np.isin(sp.fwd_dst, front) & np.isinf(d[src])
                front = np.unique(src[e])
                d[front] = k
            self.assertTrue(np.array_equal(h.table(t), d))
        tab = SO.sf_table(sp, SO.stop_values(phi, p.w), p.c, m)
        arm = EN5.enum_arm(sp, phi, p, tab, "hstar", edge_mask=m)
        for s0 in FAM.start_pool(sp, tab, "height-3")[:10].tolist():
            res = SO.evaluate(sp, tab, s0, arm.policy(), edge_mask=m)
            self.assertTrue(0.0 <= res["nreg"] <= 1.0)
        # BEST-SINGLE: a library move that is illegal under the mask is replaced by STOP
        free = SO.sf_table(sp, SO.stop_values(phi, wvec(n, {("on", 0, 1): 1, ("on", 1, 2): 1, ("on", 2, 3): 1})),
                           0.25)
        s = state_of(sp, [[3, 2, 1], [4]], held=0)  # stack(0, 1) would make a 4-tower
        self.assertEqual(sp.states[int(free.nxt[s])][0], 1)
        pol, _ = EN5.best_single(free.stop, 0.25, [free], s)
        acts, dsts = SO.moves(sp, s, SO.sorted_mask(sp, m))
        self.assertEqual(pol(s, acts, dsts), 0)


class TestEnumPolicy(unittest.TestCase):
    """run_e05's entry point enum_policy(space, w, c, kind, stop, edge_mask) -> (policy, diag)."""

    def test_adapter_equals_named_arms_and_caches(self):
        n, rng = 5, random.Random(8)
        sp = SP.get_space(n)
        phi = SO.get_phi(sp)
        EN5._SCORERS.clear()
        kinds = (("MIS", "ENUM-MIS"), ("H*", "ENUM-HSTAR"), ("MIS-rev1", "ENUM-MIS-rev1"), ("MIS-rev2", "ENUM-MIS-rev2"))
        for fam in ("H-NEG", "ID", "H-NEG"):
            g = FAM.draw_goal(fam, rng, n)
            p, tab = setup_goal(sp, phi, g["w"], g["c"])
            w = np.asarray(p.w, dtype=np.float64)
            for kind, name in kinds:
                for s0 in FAM.start_pool(sp, tab)[:5].tolist():
                    pol, diag = EN5.enum_policy(sp, w, p.c, kind=kind, stop=tab.stop)
                    ref = EN5.named_arm(name, sp, phi, p, tab)
                    res = SO.evaluate(sp, tab, s0, pol)
                    self.assertEqual(res, SO.evaluate(sp, tab, s0, ref.policy()))
                    st = ref.stats
                    self.assertEqual((diag["decisions"], diag["all_tied"], diag["steps"]),
                                     (st["multi"], st["all_tied"], st["decisions"]))
                    self.assertEqual((diag["targets"], diag["targets_raw"]), (ref.T.n_kept, ref.T.n_raw))
                    again, d2 = EN5.enum_policy(sp, w, p.c, kind=kind, stop=tab.stop)  # cached scorer: no h calls
                    self.assertEqual(SO.evaluate(sp, tab, s0, again), res)
                    self.assertEqual((d2["h_calls"], d2["steps"]), (0, diag["steps"]))
                    self.assertTrue(all(isinstance(v, int) for v in d2.values()))
                self.assertLessEqual(len(EN5._SCORERS), EN5.CACHE_PARENTS)
        pol, diag = EN5.enum_policy(sp, w, p.c, kind="MIS-adm")  # default stop = w.phi
        self.assertTrue(0.0 <= SO.evaluate(sp, tab, s0, pol)["nreg"] <= 1.0)
        with self.assertRaises(KeyError):
            EN5.enum_policy(sp, w, p.c, kind="ENUM-MIS")

    def test_adapter_masked_dynamics(self):
        n, rng = 5, random.Random(9)
        sp = SP.get_space(n)
        phi, m = SO.get_phi(sp), SO.height_limit_mask(sp, 3)
        g = FAM.draw_goal("H-DYN", rng, n)
        p, _ = setup_goal(sp, phi, g["w"], g["c"])
        tab = SO.sf_table(sp, SO.stop_values(phi, p.w), p.c, m)
        for s0 in FAM.start_pool(sp, tab, "height-3")[:5].tolist():
            for kind, h in (("H*", "hstar"), ("MIS", "mis")):
                pol, _ = EN5.enum_policy(sp, np.asarray(p.w, dtype=np.float64), p.c, kind=kind, stop=tab.stop,
                                         edge_mask=m)
                ref = EN5.enum_arm(sp, phi, p, tab, h, edge_mask=m)
                self.assertEqual(SO.evaluate(sp, tab, s0, pol, edge_mask=m),
                                 SO.evaluate(sp, tab, s0, ref.policy(), edge_mask=m))


class TestGnnEnum(unittest.TestCase):
    def test_gnn_heuristic_plumbing(self):
        import torch

        from rcx.bw import encode as EN
        from rcx.bw import models as M
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = M.RelGNN(M.RelGNNConfig(hidden=8, rounds=2, head_hidden=8)).eval()
        n = 4
        sp = SP.get_space(n)
        phi = SO.phi_table(sp)
        p, tab = setup_goal(sp, phi, wvec(n, {("on", 0, 1): -2, ("ontable", 3): 1, ("on", 2, 3): 2}), 0.25)
        arm = EN5.enum_arm(sp, phi, p, tab, "gnn", model=model)
        h, T = arm.h, arm.T
        objs = list("abcd")
        for s in (0, 7, 40):
            v = h.vector(s)
            st = SP.to_facts(pos_of(sp, s), objs)
            for t in range(len(T)):
                G = frozenset((f[0],) + tuple(objs[x] for x in f[1:]) for f in T.goals[t])
                if T.full[t].met(pos_of(sp, s)):
                    self.assertEqual(v[t], 0.0)
                else:
                    one = float(M.predict(model, [EN.graph(st, G, objs)])[0])
                    self.assertAlmostEqual(v[t], one, places=5)
            self.assertTrue(np.array_equal(h.vector(s), v))  # cached
            self.assertEqual(arm.score(s), float((arm.U(s) - 0.25 * (1 + v)).max()))
        s0 = FAM.start_pool(sp, tab)[0]
        res = SO.evaluate(sp, tab, s0, arm.policy())
        self.assertTrue(0.0 <= res["nreg"] <= 1.0)
        pol, diag = EN5.enum_policy(sp, np.asarray(p.w, dtype=np.float64), 0.25, kind="GNN", stop=tab.stop,
                                    model=model)
        self.assertEqual(SO.evaluate(sp, tab, s0, pol), res)
        self.assertEqual(diag["targets"], len(T))
        rep = arm.report()
        self.assertEqual((rep["n_kept"], rep["decisions"] > 0), (len(T), True))
        # load_gnn: sidecar config, strict load, sha prefix check
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tiny.pt")
            torch.save(model.state_dict(), path)
            with open(os.path.join(d, "tiny.json"), "w", encoding="utf-8") as f:
                json.dump({"config": M.config_dict(model)}, f)
            m2, prov = EN5.load_gnn(path, expect_sha8=None)
            g = [EN.graph(SP.to_facts(pos_of(sp, 5), objs), frozenset({("on", "a", "b")}), objs)]
            self.assertEqual(M.predict(m2, g).tolist(), M.predict(model, g).tolist())
            self.assertEqual(prov["config"]["hidden"], 8)
            with self.assertRaises(ValueError):
                EN5.load_gnn(path, expect_sha8="00000000")


if __name__ == "__main__":
    unittest.main()
