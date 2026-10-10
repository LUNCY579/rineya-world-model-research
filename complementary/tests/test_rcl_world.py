import random
import unittest
from collections import defaultdict

from rcx.rcl import gen, score as S, templates as T, vocab, world as W


class TestWorld(unittest.TestCase):
    def test_semantics(self):
        st = {"A": 10, "B": 5}
        ev = {"account": "A", "amount": 3, "kind": "transfer", "other": "B", "status": "tavo"}
        self.assertEqual(W.transition({"op": "S+", "set": ["tavo"]}, st, ev), (True, {"A": 7, "B": 8}))
        self.assertEqual(W.transition({"op": "S-", "set": ["tavo"]}, st, ev), (False, {"A": 10, "B": 5}))
        self.assertFalse(W.eval_rule({"op": "S+", "set": ["void"]}, st, dict(ev, status="Void")))
        r = {"op": "AND", "args": [{"op": "S+", "set": ["tavo"]}, {"op": "AGE", "t": 4}]}
        self.assertFalse(W.eval_rule(r, st, ev))
        r = {"op": "OR", "args": [{"op": "S-", "set": ["tavo"]}, {"op": "BGE", "t": 10}]}
        self.assertTrue(W.eval_rule(r, st, ev))
        self.assertEqual(W.listed(r, ev), True)

    def test_targets_and_strict_parse(self):
        st = {"B": 1, "A": 2}
        ev = {"account": "A", "amount": 4, "kind": "add", "other": "B", "status": "x"}
        rule = {"op": "S+", "set": ["x"]}
        t = W.target_trace(rule, st, ev, ["B", "A"])
        self.assertEqual(t, '{"cmp":{"status":"x","listed":true,"ok":true},"calc":{"A":6},"after":{"B":1,"A":6}}')
        self.assertIsNotNone(W.parse_output(t, "trace", ["B", "A"]))
        self.assertIsNone(W.parse_output(t, "trace", ["A", "B"]))
        self.assertIsNone(W.parse_output('{"after":{"B":1,"A":6}} x', "ans", ["B", "A"]))
        self.assertIsNone(W.parse_output('{"after":{"B":1,"A":6.0}}', "ans", ["B", "A"]))
        self.assertIsNone(W.parse_output('{"after":{"B":1,"A":6,"C":0}}', "ans", ["B", "A"]))
        self.assertIsNone(W.parse_output('{"after":{"B":01,"A":6}}', "ans", ["B", "A"]))
        self.assertIsNotNone(W.parse_output(' {"after":{"B":1,"A":6}}\n', "ans", ["B", "A"]))
        f = W.target_fill({"B": 1, "A": 6}, ["B", "A"], 3)
        self.assertEqual(f, '{"fill":" . . .","after":{"B":1,"A":6}}')
        self.assertIsNotNone(W.parse_output(f, "fill", ["B", "A"]))

    def test_rule_parser_all_templates(self):
        rng = random.Random(0)
        for _ in range(3000):
            op = rng.choice(["S+", "S-", "AND", "OR"])
            words = rng.sample(list(vocab.NONCE["train"]) + list(vocab.REAL_TRAIN), 3)
            sa = {"op": rng.choice(["S+", "S-"]), "set": words[: rng.choice([1, 2, 3])]}
            if op in ("S+", "S-"):
                ast = {"op": op, "set": sa["set"]}
            else:
                oth = rng.choice([{"op": "K+", "kinds": [rng.choice(["add", "transfer"])]},
                                  {"op": "AGE", "t": rng.randint(-5, 20)}, {"op": "ALT", "t": rng.randint(0, 20)},
                                  {"op": "BGE", "t": rng.randint(0, 99)}])
                ast = {"op": op, "args": [sa, oth]}
            tid = rng.choice(T.TRAIN_IDS + T.HELDOUT_IDS)
            got, _ = W.parse_rule(T.render_rule(ast, tid))
            self.assertEqual(got, ast)

    def test_canonicalize_roundtrip(self):
        bat = gen.make_battery("dev", 5, 7)
        for r in bat:
            ctext, corder, back = W.canonicalize(r["input"])
            ast, tid, cst, cev = W.parse_input(ctext)
            self.assertTrue(tid.endswith("t0") or tid == "SOLO")
            ok, cafter = W.transition(ast, cst, cev)
            self.assertEqual(ok, r["gold"]["ok"])
            self.assertEqual(list(back(cafter).items()), list(r["gold"]["after"].items()))


class TestGen(unittest.TestCase):
    def test_split_hygiene(self):
        tr = []
        for reg in gen.REGIMES:
            tr += gen.make_train(reg, 1500, 11)
        dev = gen.make_battery("dev", 30, 12)
        sea = gen.make_battery("sealed", 30, 13)
        h = lambda rs: {r["semantic_hash"] for r in rs}
        sk = lambda rs: {r["skeleton_hash"] for r in rs}
        self.assertFalse(h(tr) & h(dev))
        self.assertFalse(h(tr) & h(sea))
        self.assertFalse(h(dev) & h(sea))
        self.assertFalse(sk(tr) & sk(dev))
        self.assertFalse(sk(tr) & sk(sea))
        trw = {r["event"]["status"] for r in tr}
        for rs, sp in ((dev, "dev"), (sea, "sealed")):
            real, nonce = vocab.pools(sp)
            for r in rs:
                if r["group"] in ("F2_unseen_real", "F3_unseen_nonce") and r["variant"] == "probe":
                    self.assertNotIn(r["event"]["status"], trw)
                    self.assertIn(r["event"]["status"], set(real) | set(nonce))
        for r in tr:
            self.assertIn(r["tid"], T.TRAIN_IDS)
            self.assertNotIn(tuple(r["event"].keys()), gen.HELDOUT_EVENT_ORDERS)
            self.assertTrue(set(r["state"]) <= set("ABCD"))
            if r["rule_ast"]["op"] in ("AND", "OR"):
                self.assertIn((r["rule_ast"]["op"], r["rule_ast"]["args"][1]["op"]), gen.TRAIN_COMBOS)

    def test_battery_twins_balanced(self):
        bat = gen.make_battery("dev", 20, 3)
        by = defaultdict(set)
        for r in bat:
            by[(r["parent_id"], r["surface"])].add(r["gold"]["decision"])
        self.assertTrue(all(v == {"apply", "noop"} for v in by.values()))

    def test_cf_variants(self):
        recs = gen.make_train("CF", 400, 5)
        by = defaultdict(dict)
        for r in recs:
            by[r["parent_id"]][r["variant"]] = r
        for v in by.values():
            if len(v) < 4:
                continue
            self.assertNotEqual(v["base"]["gold"]["listed"], v["status_swap"]["gold"]["listed"])
            self.assertNotEqual(v["base"]["gold"]["listed"], v["set_swap"]["gold"]["listed"])
            if v["base"]["rule_ast"]["op"] in ("S+", "S-"):
                self.assertNotEqual(v["base"]["gold"]["ok"], v["rule_flip"]["gold"]["ok"])

    def test_floors_separate(self):
        tr = {reg: gen.make_train(reg, 1000, 21) for reg in gen.REGIMES}
        bat = gen.make_battery("dev", 30, 22)
        ctx = S.FloorContext(tr)
        comps = {}
        for fl in S.FLOORS:
            res = [S.classify(x, S.floor_after(fl, x, ctx, "VR_SD")) for x in bat]
            comps[fl] = S.composite(S.aggregate(bat, res), "ho")
        self.assertEqual(comps["ORACLE"], 1.0)
        self.assertTrue(all(v <= 0.75 for k, v in comps.items() if k != "ORACLE"), comps)


if __name__ == "__main__":
    unittest.main()
