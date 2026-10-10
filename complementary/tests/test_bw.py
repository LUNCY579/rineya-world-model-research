"""Tests for rcx.bw (RCX-E07/E08 Blocksworld infrastructure).

Run: python -B -m unittest tests.test_bw -v   (CPU only, 2 threads, about 1 minute; the GNN smoke test trains <= 30 s)
Every test instance is either fresh (excluded-filtered) or used only as a (state, goal) row that passed the row-level
exclusion check; seeds come from data.SEEDS['tests'] (+ offsets inside the RCX-E07 range).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import random  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import data as D  # noqa: E402
from rcx.bw import encode as EN  # noqa: E402
from rcx.bw import excl as EX  # noqa: E402
from rcx.bw import heuristics as H  # noqa: E402
from rcx.bw import models as M  # noqa: E402
from rcx.bw import search as S  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

SEED = D.SEEDS["tests"]
EXPECTED_PINS = {
    "bwmn/blocks.py": "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83",
    "scripts/blocks_val_large.py": "6a66d489944150b50294a75f8c07671ad722b51ede09409ddee1d2cfc91f2356",
    "scripts/bw_distill.py": "2b2d518206b2e36c252ac9b87f991a73ff701b54404758e0363670d573d49fa1",
}
B, VL = BI.load()
EXC = EX.load_exclusions()


def random_goal(rng, n):
    """(objs, goal) from B.generate (both goal styles); the init is discarded."""
    while True:
        g = B.generate(rng, n, full_goal_frac=rng.choice([0.0, 1.0]))
        if g is not None:
            return g[0], g[2]


def clean_state(rng, L):
    """A random state (holding states included) whose (state, goal) row passes the row-level exclusion check."""
    while True:
        i = rng.randrange(len(L.space))
        s = L.state(i)
        if EX.row_exclusion_reason(L.objs, s, L.goal, EXC) is None:
            return i, s


def rename_instance(rng, objs, facts_list):
    names = rng.sample(B.NAMES, len(objs))
    m = dict(zip(objs, names))
    objs2 = names[:]
    rng.shuffle(objs2)
    return objs2, [EN.rename(f, m) for f in facts_list]


class TestPinsAndHygiene(unittest.TestCase):
    def test_pins(self):
        for rel, sha in EXPECTED_PINS.items():
            self.assertEqual(BI.PINS[rel], sha)
            self.assertEqual(BI.file_sha(BI.BWM_ROOT / rel), sha)
        self.assertTrue(sys.dont_write_bytecode)
        self.assertIs(sys.modules["bwmn.blocks"], B)
        self.assertIs(VL.B, B)
        self.assertNotIn(str(BI.BWM_ROOT), sys.path)
        self.assertNotIn(str(BI.BWM_ROOT / "scripts"), sys.path)
        problems, seed = BI.distill_problems()
        self.assertEqual(seed, 20261010)
        self.assertTrue(EXC["raw"]["bw_distill_problems_check"]["regenerated_sha256_matches_file"])

    def test_pin_mismatch_fails_closed(self):
        saved = dict(BI.PINS)
        try:
            BI.PINS["bwmn/blocks.py"] = "0" * 64
            with self.assertRaises(BI.PinError):
                BI.read_pinned("bwmn/blocks.py")
        finally:
            BI.PINS.clear()
            BI.PINS.update(saved)

    def test_no_bytecode_in_bwm_bundle(self):
        diff = BI.check_no_bytecode()
        self.assertEqual(diff["new_pycache_dirs"], [])
        self.assertEqual(diff["ours"], [])


class TestExclusions(unittest.TestCase):
    def test_nonempty_and_sources_unchanged(self):
        c = EXC["raw"]["counts"]
        self.assertGreater(c["test_signatures"], 0)
        self.assertGreater(c["hard_signatures"], 0)
        self.assertGreater(c["invariants_union"], 0)
        for k in ("hard110", "instances_val_large.json", "instances_mirror_hard.json", "instances_b3_fresh.json",
                  "bw_distill.problems(240)"):
            self.assertGreater(c[f"instances:{k}"], 0, k)
        self.assertEqual(EX.verify_sources(EXC["raw"]), [])
        self.assertEqual(len(EXC["signatures"]), c["signatures_union"])
        self.assertEqual(len(EXC["invariants"]), c["invariants_union"])

    def test_recipe_bites(self):
        rng = random.Random(SEED + 0)
        reasons = set()
        for _ in range(40):  # every 3-block generated instance is a PlanBench test instance up to renaming
            g = None
            while g is None:
                g = B.generate(rng, 3, full_goal_frac=rng.choice([0.0, 1.0]))
            self.assertEqual(EX.exclusion_reason(*g, EXC), "signature")
        for _ in range(60):
            g = B.generate(rng, 6, full_goal_frac=1.0)
            if g is not None:
                reasons.add(EX.exclusion_reason(*g, EXC))
        self.assertIn("invariant", reasons)

    def test_fresh_instances_never_excluded(self):
        seen = set()
        inst, st = D.fresh_instances(SEED, [4, 5, 6, 7, 8, 9, 10, 12, 15], seen=seen, prefix="test")
        self.assertEqual(len(inst), 9)
        for it in inst:
            self.assertFalse(EX.is_excluded(it["objs"], it["init"], it["goal"], EXC))
            self.assertIsNone(EX.exclusion_reason(it["objs"], it["init"], it["goal"], EXC))
        inst2, _ = D.fresh_instances(SEED, [4, 5, 6, 7], seen=seen, prefix="test2")  # same seed: all repeats skipped
        keys = {D.instance_key(i["objs"], i["init"], i["goal"]) for i in inst}
        self.assertTrue(all(D.instance_key(i["objs"], i["init"], i["goal"]) not in keys for i in inst2))
        with self.assertRaises(ValueError):
            D.fresh_instances(20261010, [5])
        want = {"chains:5:1,4", "chains:5:5"}  # class filter + goal-level dedupe
        acc, st = D.fresh_instances(SEED + 9, [5, 5, 5], accept=want.__contains__, unique="goal", strict=False,
                                    max_draws_per_item=300, prefix="test")
        self.assertEqual({it["goal_key"] for it in acc}, want)
        self.assertEqual((len(acc), st["exhausted"]), (2, 1))


class TestSpace(unittest.TestCase):
    def test_state_counts(self):
        for n, want in ((3, 22), (4, 125), (5, 866), (6, 7057), (7, 65990)):
            self.assertEqual(len(SP.get_space(n)), want)
            self.assertEqual(SP.n_states_formula(n), want)

    def test_successors_match_bwm_step(self):
        rng = random.Random(SEED + 1)
        for n in (3, 4, 5, 6):
            objs = sorted(rng.sample(B.NAMES, n))
            sp = SP.get_space(n)
            acts = B.ground_actions(objs)
            ids = range(len(sp)) if n <= 5 else rng.sample(range(len(sp)), 300)
            for i in ids:
                pos = tuple(int(v) for v in sp.states[i])
                s = SP.to_facts(pos, objs)
                self.assertEqual(SP.from_facts(s, objs), pos)
                ref = {(a, B.step(s, a)[0]) for a in acts if B.applicable(s, a)}
                got = {(SP.action_tuple(c, objs), SP.to_facts(nx, objs)) for c, nx in SP.successors(pos)}
                self.assertEqual(got, ref)
        with self.assertRaises(ValueError):
            SP.from_facts(frozenset({("on", "a", "b"), ("on", "b", "a"), ("handempty",)}), ["a", "b"])

    def test_hstar_matches_bfs_plan(self):
        rng = random.Random(SEED + 2)
        for k in range(30):
            n = 3 + k % 4
            objs, goal = random_goal(rng, n)
            L = SP.GoalLabels(objs, goal)
            self.assertEqual(int((L.dist < 0).sum()), 0)  # no dead ends
            for _ in range(2):
                i, s = clean_state(rng, L)
                plan = B.bfs_plan(s, goal, objs)
                self.assertIsNotNone(plan)
                self.assertEqual(L.h(s), float(len(plan)), (n, k))
                self.assertTrue(B.execute(s, goal, plan)["success"])
                if plan:
                    self.assertIn(plan[0], L.opt_actions(s))

    def test_optimal_action_sets(self):
        rng = random.Random(SEED + 3)
        for k in range(12):
            n = 3 + k % 4
            objs, goal = random_goal(rng, n)
            L = SP.GoalLabels(objs, goal)
            for _ in range(10):
                i, s = clean_state(rng, L)
                h = L.h(s)
                opt = L.opt_actions(s)
                want = sorted(a for a in B.ground_actions(objs)
                              if B.applicable(s, a) and L.h(B.step(s, a)[0]) == h - 1)
                self.assertEqual(opt, want)
                self.assertEqual(len(opt) == 0, h == 0)
                cur, plan = s, []
                while not B.goal_met(cur, goal):  # following any optimal action reaches the goal in exactly h* steps
                    a = rng.choice(L.opt_actions(cur))
                    plan.append(a)
                    cur = B.step(cur, a)[0]
                self.assertEqual(len(plan), h)
                self.assertTrue(B.execute(s, goal, plan)["success"])

    def test_misplaced_bound_admissible_and_astar_exact(self):
        rng = random.Random(SEED + 4)
        for k in range(10):
            n = 3 + k % 4
            objs, goal = random_goal(rng, n)
            L = SP.GoalLabels(objs, goal)
            cg = SP.CompiledGoal(objs, goal)
            for i in range(len(L.space)):
                pos = tuple(int(v) for v in L.space.states[i])
                self.assertLessEqual(SP.misplaced_compact(pos, cg)[1], L.dist[i])
                self.assertEqual(cg.met(pos), L.dist[i] == 0)
            for _ in range(3):
                i, s = clean_state(rng, L)
                self.assertEqual(SP.astar_hstar(objs, s, goal)[0], L.h_id(i))
                self.assertLessEqual(H.misplaced_lb(s, goal), L.h(s))

    def test_astar_optimal_matches_space_labels(self):
        rng = random.Random(SEED + 9)
        for k in range(16):
            n = 4 + k % 4
            objs, goal = random_goal(rng, n)
            L = SP.GoalLabels(objs, goal)
            for _ in range(4):
                i, s = clean_state(rng, L)
                h, acts, _ = SP.astar_optimal(objs, s, goal)
                self.assertEqual(h, L.h_id(i))
                self.assertEqual(acts, L.opt_actions_id(i))
        self.assertEqual(SP.astar_optimal(objs, s, goal, max_expansions=0)[0] in (None, 0.0), True)


class TestEncodings(unittest.TestCase):
    def _pairs(self, rng, k, ns=range(3, 13)):
        out = []
        for _ in range(k):
            n = rng.choice(list(ns))
            objs, goal = random_goal(rng, n)
            s = B.towers_to_state(B.random_towers(objs, rng))
            for _ in range(rng.randint(0, 12)):  # random walk: holding states included
                s = SP.to_facts(rng.choice(SP.successors(SP.from_facts(s, objs)))[1], objs)
            out.append((objs, s, goal))
        return out

    def test_deterministic_and_flat_renaming_consistent(self):
        rng = random.Random(SEED + 5)
        for objs, s, goal in self._pairs(rng, 200):
            g1, g2 = EN.graph(s, goal, objs), EN.graph(s, goal, objs)
            self.assertTrue(np.array_equal(g1["x"], g2["x"]) and np.array_equal(g1["edges"], g2["edges"]))
            v = EN.flat(s, goal, objs)
            self.assertEqual(v.shape, (EN.flat_dim(),))
            self.assertTrue(np.array_equal(v, EN.flat(s, goal, objs)))
            self.assertEqual(EN.text(s, goal), EN.text(s, goal))
            objs2, (s2, goal2) = rename_instance(rng, objs, [s, goal])
            self.assertTrue(np.array_equal(v, EN.flat(s2, goal2, objs2)))
            self.assertEqual(EN.text_canonical(s, goal, objs), EN.text_canonical(s2, goal2, objs2))
        self.assertEqual(EN.text(s, goal), B.canonical({"goal": B.goal_json(goal), "state": B.state_json(s)}))
        self.assertIn(EN.text(s, goal), B.proposer_prompt(s, goal).replace('"outcome":"success",', ""))

    def test_gnn_invariant_to_renaming_and_node_order(self):
        rng = random.Random(SEED + 6)
        torch.manual_seed(SEED)
        for cfg in (M.RelGNNConfig(), M.RelGNNConfig(agg="mean", readout="mean", rounds=3),
                    M.RelGNNConfig(readout="sum", share_rounds=True, n_rel=len(EN.EDGE_TYPES) + 2)):
            model = M.RelGNN(cfg)
            with torch.no_grad():
                for p in model.parameters():
                    p.normal_(0.0, 0.3)
            derived = cfg.n_rel > len(EN.EDGE_TYPES)
            pairs = self._pairs(rng, 40)
            a, b = [], []
            for objs, s, goal in pairs:
                objs2, (s2, goal2) = rename_instance(rng, objs, [s, goal])
                a.append(EN.graph(s, goal, objs, derived))
                b.append(EN.graph(s2, goal2, objs2, derived))
            pa, pb = M.predict(model, a), M.predict(model, b)
            np.testing.assert_allclose(pa, pb, rtol=1e-4, atol=1e-4)
            self.assertGreater(float(np.std(pa)), 1e-3)  # not a constant function


class TestSearch(unittest.TestCase):
    def test_plans_valid(self):
        inst, _ = D.fresh_instances(SEED + 1, [8, 9, 10, 12], prefix="test")
        for it in inst:
            for r in (S.gbfs(it["objs"], it["init"], it["goal"], H.misplaced_blocks, max_expansions=3000),
                      S.beam_search(it["objs"], it["init"], it["goal"], H.misplaced_lb, width=8, max_expansions=3000)):
                self.assertEqual(r["success"], r["found"])
                if r["found"]:
                    self.assertTrue(B.execute(it["init"], it["goal"], r["plan"])["success"])
                self.assertLessEqual(r["expansions"], 3000)
        r = S.gbfs(inst[0]["objs"], inst[0]["init"], inst[0]["goal"], H.h0, max_expansions=5)
        self.assertEqual((r["success"], r["reason"]), (False, "budget"))

    def test_oracle_gbfs_is_optimal_and_batching(self):
        inst, _ = D.fresh_instances(SEED + 2, [4, 5, 6, 6], prefix="test")
        oracle = H.OracleH()

        class Batched:
            calls = 0

            def batch(self, states, goal):
                Batched.calls += 1
                return [oracle(s, goal) for s in states]

        for it in inst:
            hstar = SP.hstar(it["objs"], it["init"], it["goal"])
            r = S.gbfs(it["objs"], it["init"], it["goal"], oracle)
            self.assertTrue(r["success"])
            self.assertEqual(r["plan_len"], hstar)
            self.assertEqual(r["expansions"], hstar)  # goal test at generation: one expansion per plan step
            r2 = S.gbfs(it["objs"], it["init"], it["goal"], Batched())
            self.assertEqual(r2["plan"], r["plan"])
            self.assertEqual(S.beam_search(it["objs"], it["init"], it["goal"], oracle, width=1)["plan_len"], hstar)
        self.assertGreater(Batched.calls, 0)

    def test_baseline_heuristics(self):
        inst, _ = D.fresh_instances(SEED + 3, [6], prefix="test")
        it = inst[0]
        L = SP.GoalLabels(it["objs"], it["goal"])
        goal_states = [L.state(i) for i in np.flatnonzero(L.dist == 0)[:5]]
        for s in goal_states:
            for name, h in H.BASELINES.items():
                self.assertEqual(h(s, it["goal"]), 0.0, name)
        s = it["init"]
        self.assertEqual(H.unmet_goal_facts(s, it["goal"]), len(it["goal"] - s))
        self.assertGreater(H.h_add(s, it["goal"]), 0)
        self.assertLessEqual(H.misplaced_lb(s, it["goal"]), L.h(s))


class TestTraining(unittest.TestCase):
    def _rows(self, ns, k_per_n, seed):
        rng = random.Random(seed)
        inst = []
        for n in ns:
            for j in range(k_per_n):
                objs, goal = random_goal(rng, n)
                inst.append({"id": f"test/{n}/{j}", "n": n, "objs": objs, "goal": goal,
                             "init": B.towers_to_state(B.random_towers(objs, rng))})
        rows, st = D.build_dataset(inst, include_init=False, with_opt=False, exclude_rows=True, ex=EXC)
        self.assertGreater(len(rows), 100)
        self.assertEqual(st.get("dead", 0), 0)
        return rows

    def test_split_by_instance(self):
        inst, _ = D.fresh_instances(SEED + 4, [5] * 12 + [6] * 8, prefix="test")
        sp = D.split_by_instance(inst, {"train": 0.75, "val": 0.25}, group_key=lambda it: it["goal_key"])
        self.assertEqual(sum(map(len, sp.values())), len(inst))
        gk = [{it["goal_key"] for it in v} for v in sp.values()]
        self.assertEqual(gk[0] & gk[1], set())
        self.assertEqual(sp, D.split_by_instance(inst, {"train": 0.75, "val": 0.25},
                                                 group_key=lambda it: it["goal_key"]))

    def test_trainer_deterministic(self):
        rows = self._rows([3, 4], 2, SEED + 7)
        ex = [EN.flat(r["state"], r["goal"], r["objs"]) for r in rows]
        y = [r["h"] for r in rows]
        hists = []
        for _ in range(2):
            torch.manual_seed(SEED)
            m = M.FlatMLP()
            self.assertTrue(50_000 <= M.n_params(m) <= 200_000)
            hists.append(M.fit(m, ex, y, steps=15, batch_size=32, seed=SEED))
        self.assertEqual(hists[0], hists[1])

    def test_gnn_smoke_training_reduces_loss(self):
        rows = self._rows([3, 4], 6, SEED + 8)
        ex = [EN.graph(r["state"], r["goal"], r["objs"]) for r in rows]
        y = np.array([r["h"] for r in rows], dtype=np.float32)
        torch.manual_seed(SEED)
        model = M.RelGNN(M.RelGNNConfig())
        self.assertTrue(50_000 <= M.n_params(model) <= 200_000, M.n_params(model))
        mae0 = float(np.abs(M.predict(model, ex) - y).mean())
        hist = M.fit(model, ex, y, steps=400, time_budget_s=30, batch_size=64, lr=3e-3, seed=SEED)
        mae1 = float(np.abs(M.predict(model, ex) - y).mean())
        first = np.mean([l for _, l in hist[:10]])
        last = np.mean([l for _, l in hist[-10:]])
        self.assertLess(last, 0.5 * first, (first, last, len(hist)))
        self.assertLess(mae1, 0.5 * mae0, (mae0, mae1))
        lh = M.LearnedH(model)  # batched learned heuristic plugs into search
        objs, goal = rows[0]["objs"], rows[0]["goal"]
        s = rows[0]["state"]
        self.assertEqual(len(lh.batch([s, s], goal)), 2)
        r = S.gbfs(objs, s, goal, lh, max_expansions=200)
        if r["found"]:
            self.assertTrue(B.execute(s, goal, r["plan"])["success"])


if __name__ == "__main__":
    unittest.main()
