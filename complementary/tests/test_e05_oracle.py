"""Tests for the RCX-E05 exact soft-goal oracle (rcx/bw/soft.py; PREREG_E05_DRAFT.md §3, Q1-Q7 of DESIGN App. B).

n = 3-5 spaces only, literal seeds 0-9, never reported as results. Pure numpy: no torch, no BWM code.
Run: python -B -m unittest tests.test_e05_oracle -v   (CPU, one thread; a few seconds)
"""
import importlib.util
import os
import random
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("families_e05_ut", os.path.join(ROOT, "experiments", "RCX-E05",
                                                                              "families_e05.py"))
FAM = importlib.util.module_from_spec(_spec)
sys.modules["families_e05_ut"] = FAM
_spec.loader.exec_module(FAM)

COSTS = (0.125, 0.25, 0.75)
NAMES = "abcdefg"


def random_w(rng, n, signed=True, clear=True, kmax=5):
    """Random small-integer weights over facts(n) (negatives and clear facts allowed)."""
    F = n * (n + 1)
    ids = list(range(F)) if clear else list(range(n * n))
    w = np.zeros(F, dtype=np.int64)
    for k in rng.sample(ids, rng.randint(1, kmax)):
        w[k] = rng.choice((-2, -1, 1, 2, 3) if signed else (1, 2, 3))
    return w


def all_pairs(space, edge_mask=None):
    """d(s, t) for all pairs by simultaneous BFS (float, inf = unreachable)."""
    N = len(space)
    src, dst = np.repeat(np.arange(N), np.diff(space.fwd_ptr)), space.fwd_dst
    if edge_mask is not None:
        src, dst = src[edge_mask], dst[edge_mask]
    A = np.zeros((N, N), dtype=np.float32)
    A[src, dst] = 1
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0.0)
    front, d = np.eye(N, dtype=np.float32), 0
    while True:
        d += 1
        new = ((front @ A) > 0) & np.isinf(D)
        if not new.any():
            return D
        D[new] = d
        front = new.astype(np.float32)


def state_of(space, towers, held=None):
    """Index of the state with the given towers (bottom first, block ids) and optional held block."""
    n, pos = space.n, [None] * space.n
    for t in towers:
        pos[t[0]] = n
        for lo, hi in zip(t, t[1:]):
            pos[hi] = lo
    if held is not None:
        pos[held] = n + 1
    return space.index[tuple(pos)]


class TestFeatures(unittest.TestCase):
    def test_phi_equals_to_facts(self):
        for n in (3, 4):
            sp, fs = SP.get_space(n), SO.facts(n)
            self.assertEqual(len(fs), n * (n + 1))
            phi = SO.phi_table(sp)
            for i, row in enumerate(sp.states):
                f = SP.to_facts(tuple(int(v) for v in row), list(range(n)))
                self.assertEqual({fs[k] for k in np.flatnonzero(phi[i])}, {x for x in f if x[0] != "holding"
                                                                            and x[0] != "handempty"})
        self.assertEqual(len(SO.facts(7)), 56)


class TestValueIteration(unittest.TestCase):
    def test_q1_vi_equals_brute_force(self):
        """Q1: VI = max over s' of [w.phi(s') - c d(s, s')] exactly, n = 5, 200 random w (negatives, clear)."""
        sp = SP.get_space(5)
        phi, D = SO.phi_table(sp), all_pairs(sp)
        rng = random.Random(0)
        for _ in range(200):
            w, c = random_w(rng, 5), rng.choice(COSTS)
            stop = SO.stop_values(phi, w)
            V = SO.soft_values(sp, w, c, phi=phi)
            self.assertTrue(np.array_equal(V, (stop[None, :] - c * D).max(axis=1)))
            self.assertTrue((V >= stop).all())

    def test_q1_masked_dynamics(self):
        sp = SP.get_space(5)
        phi, m = SO.phi_table(sp), SO.height_limit_mask(sp, 3)
        D = all_pairs(sp, m)
        rng = random.Random(1)
        for _ in range(30):
            w, c = random_w(rng, 5), rng.choice(COSTS)
            stop = SO.stop_values(phi, w)
            self.assertTrue(np.array_equal(SO.soft_values(sp, w, c, m, phi), (stop[None, :] - c * D).max(axis=1)))

    def test_stop_semantics(self):
        sp = SP.get_space(4)
        phi = SO.phi_table(sp)
        rng = random.Random(2)
        for _ in range(40):
            w, c = random_w(rng, 4), rng.choice(COSTS)
            tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
            stops = tab.nxt == np.arange(len(sp))
            self.assertTrue(np.array_equal(stops, tab.stop == tab.V))  # STOP iff it is optimal (first in ties)
            self.assertTrue((tab.V >= tab.stop).all())
            s0 = rng.randrange(len(sp))
            ep = SO.run_episode(sp, s0, SO.stop_policy())  # STOP is always legal
            self.assertEqual((ep.final, ep.T, ep.loop), (s0, 0, False))
            self.assertEqual(SO.episode_return(tab.stop, c, ep), tab.stop[s0])
        # a held block has no facts: STOP while holding 0 collects nothing for ontable(0)/clear(0)
        k_tab, k_clr = SO.fact_index(4)[("ontable", 0)], SO.fact_index(4)[("clear", 0)]
        w = np.zeros(20, dtype=np.int64)
        w[[k_tab, k_clr]] = 1
        s = state_of(sp, [[1], [2], [3]], held=0)
        self.assertEqual(SO.stop_values(phi, w)[s], 0.0)

    def test_tie_rule(self):
        sp = SP.get_space(3)
        phi, fi = SO.phi_table(sp), SO.fact_index(3)
        # equal-value moves: lowest action code (stack(0, 1) before stack(0, 2))
        w = np.zeros(12, dtype=np.int64)
        w[[fi[("on", 0, 1)], fi[("on", 0, 2)]]] = 1
        tab = SO.sf_table(sp, SO.stop_values(phi, w), 0.25)
        h0 = state_of(sp, [[1], [2]], held=0)
        self.assertEqual(int(tab.nxt[h0]), state_of(sp, [[1, 0], [2]]))
        self.assertEqual(SO.oracle_policy(tab)(h0, *SO.moves(sp, h0)), 2)  # [STOP, put-down, stack(0,1), ...]
        # STOP vs move tie: stack(0, 1) worth 1 - c = 0 = stop -> STOP
        w = np.zeros(12, dtype=np.int64)
        w[fi[("on", 0, 1)]] = 1
        tab = SO.sf_table(sp, SO.stop_values(phi, w), 1.0)
        self.assertEqual((tab.V[h0], int(tab.nxt[h0])), (0.0, h0))
        self.assertEqual(SO.decide(0.0, [0.0, -1.0]), 0)
        self.assertEqual(SO.decide(0.0, [-1.0, 0.5, 0.5]), 2)

    def test_q2_conj_values(self):
        """Q2: H-CONJ V* = max(R 1[G <= s], R - c h*(s, G)) = max-plus VI from R 1[G <= s], vs GoalLabels at n = 5."""
        sp, rng, done = SP.get_space(5), random.Random(3), 0
        fs = SO.facts(5)
        while done < 25:
            G = [fs[k] for k in rng.sample(range(20), rng.choice((2, 3)))]
            cg = SP.CompiledGoal(list(range(5)), frozenset(G))
            if not cg.consistent or not sp.goal_mask(cg).any():
                continue
            c = rng.choice(COSTS)
            h = SP.GoalLabels(list(NAMES[:5]), {(f[0], NAMES[f[1]], NAMES[f[2]]) for f in G}).hstar_array()
            mask = sp.goal_mask(cg)
            ref = np.maximum(3.0 * mask, 3.0 - c * h)
            V = SO.conj_values(sp, G, 3.0, c)
            self.assertTrue(np.array_equal(V, ref))
            self.assertTrue(np.array_equal(SO.max_plus_vi(sp, 3.0 * mask, c), ref))
            done += 1


class TestPoliciesAndSF(unittest.TestCase):
    def test_q3_rollouts_equal_tables(self):
        """Q3: step-by-step rollouts reproduce the tabled endpoints and lengths; V* = w-bar.psi exactly."""
        sp = SP.get_space(5)
        phi, rng = SO.phi_table(sp), random.Random(4)
        for _ in range(20):
            w, c = random_w(rng, 5), rng.choice(COSTS)
            tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
            ps = SO.psi(phi, tab, np.arange(len(sp)))
            self.assertTrue(np.array_equal(SO.wbar_dot(ps, w, c), tab.V))
            self.assertTrue(np.array_equal(tab.stop[tab.end] - c * tab.T, tab.V))
            for s0 in rng.sample(range(len(sp)), 50):
                s, T = s0, 0
                while int(tab.nxt[s]) != s:
                    s, T = int(tab.nxt[s]), T + 1
                self.assertEqual((s, T), (int(tab.end[s0]), int(tab.T[s0])))
                ep = SO.run_episode(sp, s0, SO.oracle_policy(tab), hmax=10 ** 6)
                self.assertEqual((ep.final, ep.T, ep.loop), (s, T, False))
                self.assertEqual(SO.episode_return(tab.stop, c, ep), tab.V[s0])

    def test_q4_gpi_dominates_library(self):
        """Q4: exact GPI closed-loop value >= max_i V^{pi_i}_{w'}(s0) at every start (n = 5)."""
        sp = SP.get_space(5)
        phi, rng = SO.phi_table(sp), random.Random(5)
        lib = []
        for _ in range(6):
            g = FAM.draw_goal("TRAIN", rng, 5)
            lib.append(SO.sf_table(sp, SO.stop_values(phi, g["w"]), g["c"]))
        he = np.flatnonzero(SO.handempty_mask(sp))
        for _ in range(8):
            w, c = random_w(rng, 5), rng.choice(COSTS)
            stop = SO.stop_values(phi, w)
            G = SO.gpi_table(stop, c, lib)
            best = np.max([stop[t.end] - c * t.T for t in lib], axis=0)
            self.assertTrue(np.array_equal(G, best))
            pol = SO.gpi_policy(stop, c, G)
            for s0 in he:
                ep = SO.run_episode(sp, int(s0), pol, hmax=10 ** 6)
                self.assertFalse(ep.loop)
                self.assertGreaterEqual(SO.episode_return(stop, c, ep), max(best[s0], stop[s0]))
            # gpi_scores on psi arrays = the table form
            s = int(he[0])
            acts, dsts = SO.moves(sp, s)
            psis = np.stack([SO.psi(phi, t, dsts) for t in lib], axis=1)  # [moves, |C|, F + 1]
            self.assertTrue(np.array_equal(SO.gpi_scores(psis, w, c), G[dsts] - c))

    def test_q5_renaming(self):
        """Q5: V*_{sigma w}(sigma s) = V*_w(s); phi and SF renaming; the conjugate-policy psi identity."""
        sp = SP.get_space(5)
        phi, rng = SO.phi_table(sp), random.Random(6)
        for _ in range(30):
            w, c = random_w(rng, 5), rng.choice(COSTS)
            sig = rng.sample(range(5), 5)
            fp, spm = SO.fact_perm(5, sig), SO.state_perm(sp, sig)
            self.assertEqual(sorted(spm.tolist()), list(range(len(sp))))
            self.assertTrue(np.array_equal(phi[spm][:, fp], phi))
            ws = SO.rename_weight(w, sig)
            tab = SO.sf_table(sp, SO.stop_values(phi, w), c)
            tab_s = SO.sf_table(sp, SO.stop_values(phi, ws), c)
            self.assertTrue(np.array_equal(tab_s.V[spm], tab.V))
            conj = phi[spm[tab.end]] @ ws.astype(np.float64) - c * tab.T  # w-bar_sigma . sigma psi^{pi_w}(s)
            self.assertTrue(np.array_equal(conj, tab_s.V[spm]))
            # the tie rule is not renaming-equivariant, but both endpoint choices are optimal for sigma w
            self.assertTrue(np.array_equal(tab_s.stop[tab_s.end[spm]] - c * tab_s.T[spm], tab.V))

    def test_q6_height_limit_mask(self):
        """Q6: the height-limit-3 edge mask equals a direct rule check on every edge (n = 5)."""
        sp = SP.get_space(5)
        m = SO.height_limit_mask(sp, 3)
        e = 0
        for i, row in enumerate(sp.states):
            for code, nxt in SP.successors(tuple(int(v) for v in row)):
                self.assertEqual(sp.index[nxt], sp.fwd_dst[e])
                ok = True
                if code[0] == SP.STACK:
                    h, j = 1, code[1]
                    while nxt[j] < 5:
                        j, h = nxt[j], h + 1
                    ok = h <= 3
                self.assertEqual(bool(m[e]), ok)
                e += 1
        self.assertEqual(e, sp.n_edges)
        lev = SO.tower_levels(sp)
        tall = state_of(sp, [[0, 1, 2, 3], [4]])
        self.assertEqual(lev[tall].tolist(), [1, 2, 3, 4, 1])
        self.assertFalse(SO.height_ok(sp, 3)[tall])


class TestReferencesAndNreg(unittest.TestCase):
    def test_nreg(self):
        self.assertEqual(SO.nreg(5.0, 1.0, 5.0), 0.0)
        self.assertEqual(SO.nreg(5.0, 1.0, 1.0), 1.0)
        self.assertEqual(SO.nreg(5.0, 1.0, 3.0), 0.5)
        self.assertEqual(SO.nreg(5.0, 1.0, 0.0), 1.0)
        self.assertEqual(SO.nreg(5.0, 1.0, 6.0), 0.0)
        with self.assertRaises(ValueError):
            SO.nreg(1.0, 1.0, 1.0)

    def test_q7_identities_positive_goals(self):
        """Q7: ORACLE nreg = 0 and STOP = 1 exactly at every start; GREEDY = 1 exactly at every hand-empty start of a
        positive on/ontable goal; RAND's mean < 1 (n = 5)."""
        sp = SP.get_space(5)
        phi, rng = SO.phi_table(sp), random.Random(7)
        rand_rng, rand_all, n_starts = random.Random(8), [], 0
        for _ in range(12):
            g = FAM.draw_goal(rng.choice(("ID", "H-K", "H-COST", "TRAIN")), rng, 5)
            tab = SO.sf_table(sp, SO.stop_values(phi, g["w"]), g["c"])
            for s0 in FAM.start_pool(sp, tab).tolist():  # hand-empty, gap >= 1
                self.assertEqual(SO.evaluate(sp, tab, s0, SO.oracle_policy(tab))["nreg"], 0.0)
                self.assertEqual(SO.evaluate(sp, tab, s0, SO.stop_policy())["nreg"], 1.0)
                gr = SO.evaluate(sp, tab, s0, SO.greedy_policy(tab.stop, tab.c))
                self.assertEqual((gr["nreg"], gr["T"]), (1.0, 0))
                r = SO.evaluate(sp, tab, s0, SO.rand_policy(rand_rng))["nreg"]
                self.assertTrue(0.0 <= r <= 1.0)
                rand_all.append(r)
                n_starts += 1
        self.assertGreater(n_starts, 100)
        self.assertLess(sum(rand_all) / len(rand_all), 1.0)

    def test_greedy_moves_on_negative_hand_case(self):
        """B1 hand case (n = 4): a on b, c and d on the table; on(a,b) = -2, ontable(d) = +1, c = 0.25: V* = 0.75 by
        unstack(a, b) then STOP; ORACLE and GREEDY both reach it (GREEDY is informative on negative weights)."""
        sp = SP.get_space(4)
        phi, fi = SO.phi_table(sp), SO.fact_index(4)
        w = np.zeros(20, dtype=np.int64)
        w[fi[("on", 0, 1)]], w[fi[("ontable", 3)]] = -2, 1
        tab = SO.sf_table(sp, SO.stop_values(phi, w), 0.25)
        s0 = state_of(sp, [[1, 0], [2], [3]])
        self.assertEqual((tab.stop[s0], tab.V[s0]), (-1.0, 0.75))
        held_a = state_of(sp, [[1], [2], [3]], held=0)
        self.assertEqual((int(tab.end[s0]), int(tab.T[s0])), (held_a, 1))
        for pol in (SO.oracle_policy(tab), SO.greedy_policy(tab.stop, tab.c)):
            res = SO.evaluate(sp, tab, s0, pol)
            self.assertEqual((res["nreg"], res["final"], res["T"]), (0.0, held_a, 1))

    def test_loop_and_cap(self):
        sp = SP.get_space(3)
        phi = SO.phi_table(sp)
        tab = SO.sf_table(sp, SO.stop_values(phi, np.zeros(12)), 0.25)
        s0 = state_of(sp, [[0], [1], [2]])
        # pick-up then put-down returns to s0: forced STOP there, flagged as a loop, both moves counted
        ep = SO.run_episode(sp, s0, lambda s, a, d: 1)
        self.assertEqual((ep.final, ep.T, ep.loop, ep.capped), (s0, 2, True, False))
        ep = SO.run_episode(sp, s0, lambda s, a, d: 1, hmax=1)
        self.assertEqual((ep.T, ep.loop, ep.capped), (1, False, True))
        self.assertEqual(SO.episode_return(tab.stop, 0.25, ep), -0.25)
        with self.assertRaises(ValueError):
            SO.run_episode(sp, s0, lambda s, a, d: len(d) + 1)

    def test_bank_metrics(self):
        sp = SP.get_space(4)
        phi, rng = SO.phi_table(sp), random.Random(9)
        w = random_w(rng, 4, signed=False, clear=False)
        tab = SO.sf_table(sp, SO.stop_values(phi, w), 0.25)
        for s in FAM.bank_pool(tab)[:30].tolist():
            acts, dsts = SO.moves(sp, s)
            q = SO.q_star(tab, s, dsts)
            self.assertEqual(q.max(), tab.V[s])
            k = SO.oracle_policy(tab)(s, acts, dsts)
            self.assertEqual(SO.regret1(q, k, 0.25), (0.0, True))
            self.assertEqual(SO.regret1(q, 0, 0.25)[1], False)  # STOP is not optimal on bank states
            self.assertTrue(list(acts) == sorted(acts))


if __name__ == "__main__":
    unittest.main()
