"""Tests for the RCX-E01R R3 predicate-first trace target (experiments/RCX-E01R/trace_target.py). Pure Python and
fast: no torch, no model, no tokenizer (tiny fake tokenizers stand in)."""
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01R"))
_TORCH_BEFORE = "torch" in sys.modules

import trace_target as TT  # noqa: E402
from rcx.rcl import gen, score as S, world as W  # noqa: E402

BATTERY_SEED = 20261240  # native_probe.BATTERY_SEED (the E01N DEV battery)
_CACHE = {}


def battery(n_parents=12, surfaces=("in", "ho_tpl", "ho_many", "ho_name")):
    key = (n_parents, surfaces)
    if key not in _CACHE:
        gen.use_world("arith")
        _CACHE[key] = gen.make_battery("dev", n_parents, BATTERY_SEED, surfaces=surfaces)
    return _CACHE[key]


def rec_of(rule, state, event):
    ok, after = W.transition(rule, state, event)
    return {"rule_ast": rule, "state": dict(state), "event": dict(event),
            "gold": {"ok": ok, "listed": W.listed(rule, event), "after": after, "decision": "apply" if ok else "noop"}}


def sorted_json(d):
    return json.dumps({k: d[k] for k in sorted(d)}, separators=(",", ":"))


def first(recs, pred):
    return next(r for r in recs if pred(r))


class CharTok:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}


class WordTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class TestTraceTarget(unittest.TestCase):
    def check_round_trip(self, recs):
        for r in recs:
            t = TT.trace_target(r)
            self.assertEqual(t, TT.trace_target(r))
            line1, line2 = t.split("\n")
            self.assertEqual(line2, sorted_json(r["gold"]["after"]))
            keys = list(r["gold"]["after"])
            after, dec, wf = TT.parse_trace_answer(t, keys)
            self.assertTrue(wf, t)
            self.assertEqual((after, list(after), dec), (r["gold"]["after"], keys, r["gold"]["decision"]))
            self.assertEqual(S.classify(r, after)["exact"], 1)
            p = TT.parse_check(line1)
            rule, st, ev = r["rule_ast"], r["state"], r["event"]
            args = rule["args"] if rule["op"] in ("AND", "OR") else [rule]
            self.assertEqual([a["truth"] for a in p["atoms"]], [W.eval_atom(a, st, ev) for a in args])
            self.assertEqual([a["op"] for a in p["atoms"]], [a["op"] for a in args])
            self.assertEqual(p["conn"], rule["op"] if rule["op"] in ("AND", "OR") else None)
            self.assertEqual(p["decision"], "apply" if W.eval_rule(rule, st, ev) else "noop")
            sa = W.status_atom(rule)
            if sa is not None:
                self.assertEqual(p["atoms"][args.index(sa)]["listed"], W.listed(rule, ev))
                self.assertEqual(p["atoms"][args.index(sa)]["status"], ev["status"])
            self.assertEqual(TT.predicate_consistency(r, t),
                             {"well_formed": 1, "stated_decision": r["gold"]["decision"], "stated_decision_ok": 1,
                              "balances_match_stated_decision": 1, "line1_exact": 1, "line1_coherent": 1})

    def test_round_trip_dev_battery(self):
        recs = battery()
        self.assertGreaterEqual(len(recs), 300)
        self.assertEqual({r["group"] for r in recs}, set(gen.STRATA))
        ops = {r["rule_ast"]["op"] for r in recs}
        self.assertTrue({"S+", "S-", "AND", "OR"} <= ops, ops)
        self.assertEqual({r["gold"]["decision"] for r in recs if r["rule_ast"]["op"] == "OR"}, {"apply", "noop"})
        self.assertEqual({r["gold"]["decision"] for r in recs if r["rule_ast"]["op"] == "AND"}, {"apply", "noop"})
        self.check_round_trip(recs)

    def test_line1_ignores_wording_and_account_names(self):
        by = defaultdict(set)
        for r in battery():
            by[(r["parent_id"], r["variant"])].add(TT.trace_target(r).split("\n")[0])
        self.assertTrue(all(len(v) == 1 for v in by.values()))  # same line 1 under t0, held-out wordings, renamings

    def test_round_trip_train_regimes(self):
        gen.use_world("arith")
        for regime, seed in (("CF_SD", 20261253), ("VR", 20261252)):
            recs = gen.make_train(regime, 300, seed)
            self.assertTrue({"AND", "OR"} <= {r["rule_ast"]["op"] for r in recs})
            self.check_round_trip(recs)

    def test_copy_world_and_hand_built_atoms(self):
        try:
            gen.use_world("copy")
            recs = gen.make_battery("dev", 6, BATTERY_SEED, strata=("V6_compose", "V1_pos", "V2_neg"),
                                    surfaces=("in",))
        finally:
            gen.use_world("arith")
        self.assertTrue({r["event"]["kind"] for r in recs} <= {"set", "swap"})
        st = {"B": 7, "A": 12}
        ev = {"account": "A", "amount": -3, "kind": "transfer", "other": "B", "status": "tavo"}
        rules = [{"op": "OR", "args": [{"op": "K+", "kinds": ["add", "transfer"]}, {"op": "BGE", "t": 20}]},
                 {"op": "AND", "args": [{"op": "BGE", "t": 12}, {"op": "S-", "set": ["x", "tavo", "y"]}]},
                 {"op": "AND", "args": [{"op": "AGE", "t": -3}, {"op": "ALT", "t": 0}, {"op": "S+", "set": ["tavo"]}]},
                 {"op": "OR", "args": [{"op": "S+", "set": ["q"]}, {"op": "S-", "set": ["tavo"]}]},
                 {"op": "K+", "kinds": ["add"]}, {"op": "BGE", "t": 13}]
        hand = [rec_of(r, st, ev) for r in rules]
        self.check_round_trip(recs + hand)
        self.assertEqual(TT.trace_target(hand[0]).split("\n")[0],
                         "check kind=transfer in=add|transfer:true balance=12 at-least=20:false OR -> apply")
        self.assertEqual(TT.trace_target(hand[1]).split("\n")[0],
                         'check balance=12 at-least=12:true status="tavo" listed=yes all-but-listed:false AND -> noop')
        self.assertEqual(TT.trace_target(hand[2]).split("\n")[0],
                         'check amount=-3 at-least=-3:true amount=-3 below=0:true status="tavo" listed=yes '
                         'only-listed:true AND -> apply')

    def test_status_strings_cannot_break_the_parse(self):
        st = {"A": 1, "B": 2}
        for s in ('x -> noop', 'a AND', '" listed=yes', 'k=v:true OR', 'tab\there', 'é', ''):
            for rule in ({"op": "S+", "set": [s]}, {"op": "S-", "set": ["zz"]},
                         {"op": "OR", "args": [{"op": "S-", "set": [s]}, {"op": "AGE", "t": 3}]}):
                r = rec_of(rule, st, {"account": "A", "amount": 2, "kind": "add", "other": "B", "status": s})
                self.check_round_trip([r])

    def test_trace_target_fails_closed(self):
        r = dict(first(battery(), lambda x: x["gold"]["decision"] == "noop"))
        r["gold"] = dict(r["gold"], decision="apply")
        with self.assertRaises(ValueError):
            TT.trace_target(r)
        with self.assertRaises(ValueError):
            TT.check_line({"op": "AND", "args": [{"op": "S+", "set": ["x"]}]}, {"A": 1},
                          {"account": "A", "amount": 1, "kind": "add", "other": "A", "status": "x"})


class TestParsing(unittest.TestCase):
    def setUp(self):
        self.comp = first(battery(), lambda x: x["rule_ast"]["op"] == "AND" and x["gold"]["decision"] == "apply")
        self.single = first(battery(), lambda x: x["group"] == "V1_pos" and x["gold"]["decision"] == "noop")

    def parts(self, r):
        line1, line2 = TT.trace_target(r).split("\n")
        return line1, line2, list(r["gold"]["after"])

    def test_malformed_line1_keeps_line2_balances(self):
        line1, line2, keys = self.parts(self.comp)
        s1, _, _ = self.parts(self.single)
        st = self.comp["event"]["status"]
        bad = [line1.replace(f'status="{st}"', f"status={st}"), line1.replace("-> apply", "-> maybe"),
               line1.replace(" listed=", "  listed="), line1.replace(":true", ":True", 1),
               line1.replace(" AND ->", " ->"), line1.replace(" AND ->", " OR AND ->"), line1 + " ok",
               line1.replace("check ", "Check "), line1.replace("check ", ""), line1.replace(" -> ", " => "),
               s1.replace(" -> ", " AND -> "), line1.replace(f'"{st}"', '"\\u0061"'),
               line1.replace("listed=yes", "listed=true").replace("listed=no", "listed=false"),
               "check -> apply", "apply", "-"]
        amt = self.comp["event"]["amount"]
        bad.append(line1.replace(f"amount={amt} ", f"amount=0{abs(amt)} "))
        bad.append(line1.replace(f"amount={amt} ", "amount=-0 "))
        for b in bad:
            self.assertNotEqual(b, line1)
            after, dec, wf = TT.parse_trace_answer(b + "\n" + line2, keys)
            self.assertEqual((after, dec, wf), (self.comp["gold"]["after"], None, False), b)

    def test_malformed_line2(self):
        line1, line2, keys = self.parts(self.comp)
        gold = self.comp["gold"]["after"]
        k0 = sorted(gold)[0]
        bad = ["", line2[:-1], line2 + " x", line2 + "\nmore", line2 + "\n" + line2,
               line2.replace(f'"{k0}":{gold[k0]}', f'"{k0}":{gold[k0]}.0'),
               line2.replace(f'"{k0}":{gold[k0]}', f'"{k0}":true'),
               line2.replace(f'"{k0}":{gold[k0]}', f'"{k0}":"{gold[k0]}"'),
               line2.replace(f'"{k0}":{gold[k0]},', ""), line2[:-1] + ',"Z":1}', "[" + line2 + "]",
               "{" + f'"{k0}":{gold[k0] + 1},' + line2[1:],     # duplicate key; json.loads would keep the gold value
               "{" + f'"{k0}":{gold[k0]},' + line2[1:]]         # duplicate key with the same value
        for b in bad:
            after, dec, wf = TT.parse_trace_answer(line1 + "\n" + b, keys)
            self.assertEqual((after, dec, wf), (None, "apply", False), b)
        self.assertEqual(TT.parse_trace_answer(line1, keys), (None, "apply", False))        # no line 2
        self.assertEqual(TT.parse_trace_answer(line2, keys), (None, None, False))           # bare native answer
        self.assertEqual(TT.parse_trace_answer("\n" + line2, keys), (None, None, False))
        self.assertEqual(TT.parse_trace_answer(line2 + "\n" + line1, keys), (None, None, False))  # swapped lines

    def test_duplicate_keys_rejected(self):
        st = {"A": 5, "B": 7}
        ev = {"account": "A", "amount": 2, "kind": "add", "other": "B", "status": "x"}
        r = rec_of({"op": "S+", "set": ["x"]}, st, ev)
        line1 = TT.trace_target(r).split("\n")[0]
        for b in ('{"A":1,"A":7,"B":7}', '{"A":7,"A":7,"B":7}', '{"after":{"A":7,"B":7,"B":7}}',
                  '{"after":{"A":7,"B":7},"after":{"A":7,"B":7}}'):
            self.assertEqual(TT.parse_trace_answer(line1 + "\n" + b, ["A", "B"]), (None, "apply", False), b)
        self.assertEqual(TT.parse_trace_answer(line1 + '\n{"A":7,"B":7}', ["A", "B"]),
                         ({"A": 7, "B": 7}, "apply", True))

    def test_inherited_parse_answer_leniency_and_whitespace(self):
        line1, line2, keys = self.parts(self.comp)
        gold = self.comp["gold"]["after"]
        rev = json.dumps({k: gold[k] for k in reversed(sorted(gold))})
        for ok in ("  " + line1 + "\n" + line2 + "\n", line1 + "\r\n" + line2, line1 + " \n" + line2,
                   line1 + "\n" + line2 + ".", line1 + '\n{"after":' + line2 + "}", line1 + "\n " + rev):
            self.assertEqual(TT.parse_trace_answer(ok, keys), (gold, "apply", True), ok)

    def test_parse_last_line_frozen_rule(self):
        line1, line2, keys = self.parts(self.comp)
        gold = self.comp["gold"]["after"]
        rev = json.dumps({k: gold[k] for k in reversed(sorted(gold))})
        ok = [line1 + "\n" + line2, line2, "  " + line2 + "  ", line1 + "\n" + line2 + "\n\n \t\n",
              line1 + "\r\n" + line2 + "\r\n", "\n\n" + line1 + "\n\n" + line2, line1 + "\nJUNK\n" + line2,
              "check nonsense -> noop\n" + line2, line2 + ".", line1 + '\n{"after":' + line2 + "}", line1 + "\n" + rev,
              line2 + "\n" + line2]
        for t in ok:
            self.assertEqual(TT.parse_last_line(t, keys), gold, t)
        k0 = sorted(gold)[0]
        bad = ["", " \n\t\n", line1, line2 + "\n" + line1, line1 + "\n" + line2 + "\nmore", line1 + "\n" + line2 + " x",
               line1 + "\n" + json.dumps(gold, indent=1), line1 + "\n{}", "[" + line2 + "]",
               line1 + "\n{" + f'"{k0}":{gold[k0] + 1},' + line2[1:],  # duplicate key (native parse_answer refuses)
               line1 + "\n" + line2.replace(f'"{k0}":{gold[k0]}', f'"{k0}":true')]
        for t in bad:
            self.assertIsNone(TT.parse_last_line(t, keys), t)

    def test_parse_last_line_vs_native_and_strict_readings(self):
        for r in battery():
            t = TT.trace_target(r)
            keys = list(r["gold"]["after"])
            after = TT.parse_last_line(t, keys)
            self.assertEqual(after, TT.parse_trace_answer(t, keys)[0])     # well-formed traces: the same balances
            self.assertEqual(S.classify(r, after)["exact"], 1)
            line2 = t.split("\n")[1]
            self.assertEqual(TT.parse_last_line(line2, keys), TT.parse_answer(line2, keys))  # one line: = native
        line1, line2, keys = self.parts(self.comp)
        self.assertEqual(TT.parse_trace_answer(line1 + "\nJUNK\n" + line2, keys)[2], False)  # strict format fails...
        self.assertEqual(TT.parse_last_line(line1 + "\nJUNK\n" + line2, keys), self.comp["gold"]["after"])  # ...scored

    def test_borrowed_native_code(self):
        self.assertTrue(TT.parse_answer.__code__.co_filename.endswith("native_probe.py"))
        self.assertTrue(TT.target_json.__code__.co_filename.endswith("native_e01.py"))
        self.assertEqual(TT.target_json({"b": 1, "a": -2}), '{"a":-2,"b":1}')
        self.assertIsNone(TT.parse_answer('{"A":1,"A":1}', ["A"]))
        self.assertIsNone(TT.parse_answer('{"A":true}', ["A"]))
        self.assertEqual(TT.parse_answer('{"after":{"A":1}}.', ["A"]), {"A": 1})
        if not _TORCH_BEFORE:
            self.assertNotIn("torch", sys.modules)

    def test_predicate_consistency(self):
        r = self.single  # S+ with an unlisted status: gold noop
        line1, line2, keys = self.parts(r)
        st = r["event"]["status"]
        applied = W.applied(r["state"], r["event"])
        app2 = TT.target_json({k: applied[k] for k in keys})
        wrong_coherent = f'check status="{st}" listed=yes only-listed:true -> apply'
        wrong_incoherent = f'check status="{st}" listed=no only-listed:true -> apply'
        cases = [
            (wrong_coherent + "\n" + app2, {"stated_decision": "apply", "stated_decision_ok": 0,
                                            "balances_match_stated_decision": 1, "line1_exact": 0,
                                            "line1_coherent": 1, "well_formed": 1}),
            (wrong_incoherent + "\n" + line2, {"stated_decision": "apply", "stated_decision_ok": 0,
                                               "balances_match_stated_decision": 0, "line1_exact": 0,
                                               "line1_coherent": 0, "well_formed": 1}),
            (line1 + "\n" + app2, {"stated_decision": "noop", "stated_decision_ok": 1,
                                   "balances_match_stated_decision": 0, "line1_exact": 1, "line1_coherent": 1,
                                   "well_formed": 1}),
            ("check nonsense -> noop\n" + line2, {"stated_decision": None, "stated_decision_ok": 0,
                                                   "balances_match_stated_decision": None, "line1_exact": 0,
                                                   "line1_coherent": None, "well_formed": 0}),
            (line1 + "\n{}", {"stated_decision": "noop", "stated_decision_ok": 1,
                              "balances_match_stated_decision": None, "line1_exact": 1, "line1_coherent": 1,
                              "well_formed": 0}),
        ]
        for text, want in cases:
            self.assertEqual(TT.predicate_consistency(r, text), want, text)


class TestTokenBudget(unittest.TestCase):
    def test_lora_cpu_generate_facts(self):
        n = TT.token_budget_note()
        self.assertTrue(n["accepts_max_new"])
        self.assertEqual((n["max_new_default"], n["max_batch_tokens_default"]), (64, 3000))
        self.assertEqual(n["eos_ids"], [151645, 151643])
        self.assertTrue(n["stops_only_at_eos"])
        self.assertEqual(n["model_generate_kwargs"]["max_new_tokens"], "max_new")

    def test_generate_kwargs(self):
        self.assertEqual((TT.TRACE_MAX_NEW, TT.TRACE_MARGIN), (160, 16))  # frozen rule: max(160, longest gold + 16)
        self.assertEqual(TT.generate_kwargs_for_trace(), {"max_new": TT.TRACE_MAX_NEW})
        recs = battery(40, tuple(gen.SURFACES))
        targets = [TT.trace_target(r) for r in recs]
        longest = max(len(t.encode("utf-8")) for t in targets)
        self.assertLessEqual(longest + 1, TT.TRACE_MAX_NEW)  # the floor covers the whole DEV generator draw (+ EOS)
        self.assertGreater(longest + 1, 64)                   # ... and the native default would not
        self.assertEqual(TT.trace_max_new(targets), longest + 16)  # bytes, no tokenizer: 154 + 16 on this draw
        self.assertEqual(TT.generate_kwargs_for_trace(targets), {"max_new": longest + 16})
        self.assertEqual(TT.generate_kwargs_for_trace(targets, CharTok()), {"max_new": longest + 16})
        words = max(len(t.split()) for t in targets)
        self.assertLess(words + 16, 160)
        self.assertEqual(TT.generate_kwargs_for_trace(targets, WordTok()), {"max_new": 160})  # never below the floor

    def test_trace_max_new_rule(self):
        self.assertEqual(TT.trace_max_new(["x" * 10]), 160)
        self.assertEqual(TT.trace_max_new(["x" * 144]), 160)
        self.assertEqual(TT.trace_max_new(["x" * 145]), 161)
        self.assertEqual(TT.trace_max_new(["é" * 100]), 216)                  # UTF-8 bytes, not characters
        self.assertEqual(TT.trace_max_new(["é" * 100], CharTok()), 216)
        self.assertEqual(TT.trace_max_new(["a b"] * 3 + [" ".join("w" * 200)], WordTok()), 216)
        self.assertEqual(TT.trace_max_new(iter(["y" * 200, "z"])), 216)       # any iterable, longest wins
        with self.assertRaises(ValueError):
            TT.trace_max_new([])

    def test_doc_states_the_committed_max_new(self):
        """Review finding 10: every E01R eval uses the committed build-time max_new (>= 160), never generate's 64."""
        doc = " ".join(TT.__doc__.split())
        self.assertNotIn("E01N/E01R evals use the default max_new=64", doc)
        self.assertIn("every arm (N-CFSD, E-R*, OFF) is decoded with the committed build-time max_new (>= 160", doc)
        self.assertIn("N-VR is not evaluated", doc)  # PREREG_E01R section 4: eval refuses N-VR
        self.assertNotIn("OFF, N-VR", doc)


if __name__ == "__main__":
    unittest.main()
