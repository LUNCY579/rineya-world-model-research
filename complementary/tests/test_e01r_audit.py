"""Tests for the RCX-E01R training-row audit (experiments/RCX-E01R/audit_e01r.py) and the R1 training-pool audit
(experiments/RCX-E01R/audit_tr_pool.py). Pure Python: no torch, no model, no tokenizer (fakes only).

The pool audit runs only on fake TR/P2/P3 files in a temporary directory: no sealed file is opened. Otherwise read-only:
pools/P2_DEV.json and pools/P2_DEV_v2.json, the RCWM reference tree, the DEV2 battery file (CLI test) and, if present,
runs/RCX-E01N/cells/train_N-CFSD.jsonl.
Run: python -B -m unittest tests.test_e01r_audit -v   (CPU, single thread; about 10 s)
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments" / "RCX-E01R"
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
HEAVY = ("torch", "transformers")
_HEAVY_BEFORE = {m for m in HEAVY if m in sys.modules}

import audit_e01r as AE  # noqa: E402
import audit_tr_pool as ATP  # noqa: E402
import recipes as R  # noqa: E402
import trace_target as TT  # noqa: E402
from rcx.rcl import gen, templates as T, wording_audit as WA, world as W  # noqa: E402

_HEAVY_AFTER = {m for m in HEAVY if m in sys.modules}
E01N = ROOT / "experiments" / "RCX-E01N"
CF_SEED = 20261253  # N-CFSD's data seed
CELLS = [(p, f) for p in ("POS", "NEG") for f in ATP.FORMS]
LETTERS = "cwxyqhj"  # no nonce word uses these letters; 3-letter words cannot be status stems (>= 4 letters)


def pseudo(k):
    return LETTERS[k // 49 % 7] + LETTERS[k // 7 % 7] + LETTERS[k % 7]


def wording(k, extra=()):
    """Four fresh pseudo-words around the slot (no word shared with wording(k'), k' != k), then extra words."""
    w = [pseudo(4 * k + i) for i in range(4)]
    tail = "".join(f" {pseudo(e)}" for e in extra)
    return f"{w[0].capitalize()} {w[1]} {{L}} {w[2]} {w[3]}{tail}."


FAKE = [{"id": f"q{i}", "text": wording(i), "pol": p, "form": f} for i, (p, f) in enumerate(CELLS)]
POS_IDS, NEG_IDS = ["tr:q0", "tr:q1", "tr:q2"], ["tr:q3", "tr:q4", "tr:q5"]


def records(n=40, seed=CF_SEED):
    gen.use_world("arith")
    return gen.make_train("CF_SD", n, seed)


def row_of(rec, target=None):
    """native_e01.native_rows' row for a record, with the Qwen3 template written out (render_literal)."""
    system = AE.rcl_system(T.render_rule(rec["rule_ast"], rec["tid"]))
    return {"prompt": AE.render_literal(system, AE.user_json(rec)),
            "target": TT.target_json(rec["gold"]["after"]) if target is None else target, "regime": "X",
            "parent_id": rec["parent_id"], "variant": rec["variant"], "decision": rec["gold"]["decision"],
            "semantic_hash": rec["semantic_hash"], "skeleton_hash": rec["skeleton_hash"],
            "status": rec["event"]["status"]}


def pooled(recs):
    """R1-style copies: every other single-atom record gets a fake pool wording of its polarity."""
    out, k = [], 0
    for r in recs:
        q = json.loads(json.dumps(r))
        if r["rule_ast"]["op"] in ("S+", "S-"):
            if k % 2 == 0:
                q["tid"] = (POS_IDS if r["rule_ast"]["op"] == "S+" else NEG_IDS)[k // 2 % 3]
            k += 1
        out.append(q)
    return out


def mk(rule, status="cleared", tid="t0", amount=3, state=None):
    state = dict(state or {"A": 10, "B": 20, "C": 30, "D": 40})
    event = {"account": "A", "amount": amount, "kind": "add", "other": "B", "status": status}
    ok, after = W.transition(rule, state, event)
    return {"rule_ast": rule, "tid": tid, "state": state, "event": event, "parent_id": "p", "variant": "base",
            "gold": {"ok": ok, "after": after, "decision": "apply" if ok else "noop"},
            "semantic_hash": W.semantic_hash(rule, state, event), "skeleton_hash": W.skeleton_hash(state, event)}


def run(arms, control=None, **kw):
    kw.setdefault("batteries", {})
    kw.setdefault("rcwm_texts", {})
    kw.setdefault("n_rows", len(next(iter(arms.values()))["rows"]))
    return AE.audit_e01r(arms, control=control, **kw)


def flagged(res, arm="X"):
    return {f for f, k in res["arms"][arm]["hard_check_rows"].items() if k}


class Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class WordTok:
    """Fake tokenizer: one token per whitespace-separated word; also a Qwen3-style chat template (text only)."""
    eos_token_id = -1

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return Enc(input_ids=text.split())

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return ("".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _cleanup_templates():
    R.unregister_pool("tr")
    R.unregister_pool("tq")
    R.unregister_pool("p2")


# ----------------------------------------------------------------------------- reader (review item 5)

class TestReader(unittest.TestCase):
    def setUp(self):
        _cleanup_templates()
        R.register_pool(FAKE, "tr")

    def tearDown(self):
        _cleanup_templates()

    def test_original_sentences_read_as_world_parse_rule(self):
        read = AE.make_reader()
        gen.use_world("arith")
        recs = gen.make_battery("dev", 3, 20261240) + records(300)
        for r in recs:
            s = T.render_rule(r["rule_ast"], r["tid"])
            ast_, tid, rendered = read(s)
            self.assertEqual((ast_, tid), W.parse_rule(s), s)
            self.assertEqual((ast_, rendered), (r["rule_ast"], s))
        self.assertEqual({AE.wording_source(read(T.render_rule(r["rule_ast"], r["tid"]))[1]) for r in recs},
                         {"single:t*", "single:h*", "frame:t*", "frame:h*"})

    def test_pooled_r1_row_reads_back_to_ast_and_gold(self):
        recs = pooled(records(80))
        pool_recs = [r for r in recs if r["tid"].startswith("tr:")]
        self.assertGreater(len(pool_recs), 10)
        read = AE.make_reader()
        for r in pool_recs:
            sentence = T.render_rule(r["rule_ast"], r["tid"])
            with self.assertRaises(ValueError):
                W.parse_rule(sentence)  # world._SINGLE was compiled at import: the bug the E01R reader fixes
            rb = AE.read_back(row_of(r), read)
            self.assertEqual((rb["rule"], rb["tid"]), (r["rule_ast"], f"{r['rule_ast']['op']}:{r['tid']}"))
            self.assertEqual((rb["ok"], rb["after"]), (r["gold"]["ok"], r["gold"]["after"]))
            self.assertEqual((rb["state"], rb["event"]), (r["state"], r["event"]))
            self.assertTrue(rb["frame_exact"])
            self.assertEqual(rb["semantic_hash"], r["semantic_hash"])
            self.assertEqual(AE.wording_source(rb["tid"]), "single:tr")
        np_mod = types.SimpleNamespace(rcl_system=AE.rcl_system, user_json=AE.user_json)
        built = R.native_rows(recs, WordTok(), "R1", np_mod=np_mod)  # the builder's rows, through a fake template
        self.assertEqual([(b["prompt"], b["target"]) for b in built], [(x["prompt"], x["target"])
                                                                        for x in map(row_of, recs)])

    def test_reader_is_a_snapshot_after_register_pool(self):
        _cleanup_templates()
        before = AE.make_reader()
        R.register_pool(FAKE, "tr")
        after = AE.make_reader()
        s = T.render_rule({"op": "S+", "set": ["cleared", "booked"]}, "tr:q2")
        with self.assertRaises(ValueError):
            before(s)
        self.assertEqual(after(s)[:2], ({"op": "S+", "set": ["cleared", "booked"]}, "S+:tr:q2"))
        self.assertEqual(sorted(after.templates), sorted(f"{op}:{k}" for op, d in (("S+", T.SPOS), ("S-", T.SNEG))
                                                         for k in d))

    def test_unique_reading_required(self):
        T.SPOS["tq:dup"] = T.SPOS["t0"]  # same text, other id: two readings of every S+ t0 sentence
        read = AE.make_reader()
        s = T.render_rule({"op": "S+", "set": ["cleared"]}, "t0")
        with self.assertRaises(ValueError):
            read(s)
        res = run({"X": {"rows": [row_of(mk({"op": "S+", "set": ["cleared"]}))]}})
        self.assertIn("frame", flagged(res))
        self.assertIn("frame:X", res["failures"])

    def test_heldout_fragments_from_original_templates_only(self):
        (native_frags,) = AE._borrow(E01N / "native_e01.py", ("heldout_template_fragments",), T=T)
        base = AE.heldout_fragments()
        self.assertEqual(base, native_frags())
        frag = ", carry out the event; for any other status, do nothing."  # SPOS h0's fixed text
        self.assertIn(frag, base)
        T.SPOS["tq:frag"] = "Cwx {L}" + frag
        self.assertEqual(AE.heldout_fragments(), base)       # a registered wording cannot hide a held-out fragment
        self.assertNotIn(frag, native_frags())              # ... as it would in native_e01's version
        res = run({"X": {"rows": [row_of(mk({"op": "S+", "set": ["cleared"]}, tid="tq:frag"))]}})
        self.assertIn("heldout_template", flagged(res))

    def test_pool_id_starting_with_h_reads_as_heldout(self):
        T.SPOS["tq:h7"] = wording(30)  # registered directly: recipes may refuse such ids
        res = run({"X": {"rows": [row_of(mk({"op": "S+", "set": ["cleared"]}, tid="tq:h7"))]}})
        self.assertIn("heldout_template", flagged(res))
        self.assertEqual(AE.wording_source("S+:tq:h7"), "single:tq")

    def test_wording_source(self):
        self.assertEqual([AE.wording_source(t) for t in ("S+:t0", "S-:h1", "S+:tr:q1", "AND:t1", "OR:h0", "SOLO",
                                                          "S+:p2:w007")],
                         ["single:t*", "single:h*", "single:tr", "frame:t*", "frame:h*", "solo", "single:p2"])


# ----------------------------------------------------------------------------- row audit

class TestAudit(unittest.TestCase):
    def setUp(self):
        _cleanup_templates()
        R.register_pool(FAKE, "tr")
        self.recs = records(40)
        self.control = [row_of(r) for r in self.recs]

    def tearDown(self):
        _cleanup_templates()

    def arms(self):
        r1 = pooled(self.recs)
        return {"R1": {"rows": [row_of(r) for r in r1], "target": "native", "match": "wording"},
                "R3": {"rows": [row_of(r, TT.trace_target(r)) for r in self.recs], "target": "trace",
                       "match": "prompts"},
                "R2": {"rows": [row_of(r) for r in records(40, 7)], "target": "native"}}

    def test_clean_arms_pass_with_covariates(self):
        gen.use_world("arith")
        bat = {"tiny": gen.make_battery("dev", 2, 20261240, surfaces=("in",))}
        cells = {f"tr:{t['id']}": f"{t['pol']}/{t['form']}" for t in FAKE}
        res = run(self.arms(), control=self.control, batteries=bat, pool_cells=cells, tok=WordTok())
        self.assertTrue(res["ok"], res["failures"])
        self.assertEqual(res["control_failures"], [])
        json.dumps(res)
        r1, r3, ctl = res["arms"]["R1"], res["arms"]["R3"], res["control"]
        n_pool = sum(r["tid"].startswith("tr:") for r in pooled(self.recs))
        self.assertEqual(r1["class_balance_by_wording_source"]["single:tr"]["n"], n_pool)
        self.assertEqual(sum(r1["pooled_rows_by_cell"].values()), n_pool)
        self.assertEqual(sum(r1["rows_by_rule_type"].values()), 40)
        self.assertEqual(r1["rows_by_rule_type"], ctl["rows_by_rule_type"])
        self.assertEqual((r1["noop_share"], r1["parents"]), (ctl["noop_share"], ctl["parents"]))
        self.assertEqual(r3["control_mismatch_rows"], 0)
        self.assertEqual(r1["control_mismatch_rows"], 0)
        self.assertIsNone(res["arms"]["R2"]["control_mismatch_rows"])
        self.assertGreater(r3["target_bytes"]["mean"], ctl["target_bytes"]["mean"] + 30)
        self.assertEqual(res["reader"]["registered"], sorted(f"{'S+' if i < 3 else 'S-'}:tr:q{i}" for i in range(6)))
        self.assertEqual(res["arms"]["R1"]["overlap"], {"tiny": {"semantic": 0, "skeleton": 0}})
        self.assertIn("token_covariates_vs_control", res["matched"])

    def test_row_count_and_control_relations(self):
        arms = self.arms()
        r3 = arms["R3"]["rows"]
        r3[0], r3[1] = r3[1], r3[0]
        r1, p = arms["R1"]["rows"], pooled(self.recs)
        i = next(k for k, r in enumerate(self.recs) if r["rule_ast"]["op"] == "S+" and r["tid"] != "t1"
                 and p[k]["tid"] == r["tid"])  # a row R1 left in its training wording
        r1[i] = row_of(dict(self.recs[i], tid="t1"))  # a valid row, but not the control's prompt
        res = run(arms, control=self.control)
        self.assertEqual(res["arms"]["R3"]["control_mismatch_rows"], 2)
        self.assertEqual(res["arms"]["R1"]["control_mismatch_rows"], 1)
        self.assertIn("control_match:R3", res["failures"])
        self.assertIn("control_match:R1", res["failures"])
        self.assertNotIn("gold:R1", res["failures"])
        res = run({"R3": {"rows": self.arms()["R3"]["rows"][:39], "target": "trace", "match": "prompts"}},
                  control=self.control, n_rows=40)
        self.assertEqual(set(res["failures"]), {"rows_count:R3", "control_match:R3"})

    def test_corruptions_are_flagged(self):
        sp = {"op": "S+", "set": ["cleared"]}
        base = row_of(mk(sp))
        self.assertEqual(flagged(run({"X": {"rows": [base]}})), set())
        T.SPOS["tq:cp"] = json.loads((HERE / "pools" / "P2_DEV_v2.json").read_text(encoding="utf-8"))[
            "templates"][0]["text"]
        T.SPOS["p2:zz"] = wording(31)
        T.SPOS["tq:cap"] = "Parked {L} cwq."
        gold_plus = dict(base, target=TT.target_json(dict(mk(sp)["gold"]["after"], A=14)))
        cases = [
            ("frame", dict(base, prompt=base["prompt"].replace("Apply one event", "Apply an event")), {"frame"}),
            ("user json", dict(base, prompt=base["prompt"].replace('{"before":{', '{"before": {')), {"frame"}),
            ("target", gold_plus, {"gold"}),
            ("decision", dict(base, decision="noop"), {"gold"}),
            ("status field", dict(base, status="booked"), {"gold"}),
            ("hash", dict(base, semantic_hash="0" * 64), {"readback_hash"}),
            ("dev word", row_of(mk({"op": "S+", "set": ["parked"]}, "parked")), {"split_words", "non_train_words"}),
            ("probe", row_of(mk(sp, "reposted")), {"probe_words", "non_train_words"}),
            ("excluded", row_of(mk(sp, "pending")), {"excluded_words", "non_train_words"}),
            ("non-train", row_of(mk(sp, "zzq")), {"non_train_words"}),
            ("h1", row_of(mk(sp, tid="h1")), {"heldout_template"}),
            ("size 3", row_of(mk({"op": "S-", "set": ["cleared", "booked", "queued"]})), {"heldout_rule_factor"}),
            ("combo", row_of(mk({"op": "AND", "args": [sp, {"op": "ALT", "t": 5}]})), {"heldout_rule_factor"}),
            ("magnitude", row_of(mk(sp, amount=50)), {"heldout_magnitude"}),
            ("balance", row_of(mk(sp, state={"A": 120, "B": 1, "C": 2, "D": 3})), {"heldout_magnitude"}),
            ("P2 copy", row_of(mk(sp, tid="tq:cp")), {"pool_wording"}),
            ("p2 id", row_of(mk(sp, tid="p2:zz")), {"pool_wording"}),
            ("capitalised dev word", row_of(mk(sp, tid="tq:cap")), {"split_words", "pool_wording"}),
        ]
        for name, row, want in cases:
            with self.subTest(case=name):
                res = run({"X": {"rows": [row]}})
                self.assertEqual(flagged(res), want)
                self.assertFalse(res["ok"])
        res = run({"X": {"rows": [row_of(mk(sp, tid="tq:cp"))]}})
        self.assertTrue(res["arms"]["X"]["pool_templates_flagged"]["S+:tq:cp"] in ("p2", "ref"))
        self.assertEqual(run({"X": {"rows": [row_of(mk(sp, tid="p2:zz"))]}})["arms"]["X"]["pool_templates_flagged"],
                         {"S+:p2:zz": "test_pool"})
        self.assertEqual(flagged(run({"X": {"rows": [base]}}, rcwm_texts={"x": '"status":"cleared"'})),
                         {"rcwm_heldout_text"})

    def test_trace_targets(self):
        rec = mk({"op": "S+", "set": ["booked"]})  # unlisted: noop
        good = TT.trace_target(rec)
        line1, line2 = good.split("\n")
        self.assertIn("listed=no", line1)
        cases = [(good, set()), (line1.replace("listed=no", "listed=yes"), {"trace_line1"}),
                 (line1.replace("-> noop", "-> apply"), {"trace_line1"}),
                 (line1 + "\n" + line2.replace(":10", ":13"), {"gold"}), (line2, {"gold", "trace_line1"}),
                 (good + "\n" + line2, {"gold"}), (line1 + "\n" + line2 + " ", {"gold"})]
        for target, want in cases:
            if "\n" not in target and target != line2:
                target = target + "\n" + line2
            with self.subTest(target=target):
                res = run({"X": {"rows": [row_of(rec, target)], "target": "trace"}})
                self.assertEqual(flagged(res), want)
        self.assertEqual(flagged(run({"X": {"rows": [row_of(rec, good)], "target": "native"}})), {"gold"})
        with self.assertRaises(ValueError):
            run({"X": {"rows": [row_of(rec)], "target": "json"}})

    def test_overlap_with_batteries(self):
        rec = mk({"op": "S+", "set": ["cleared"]})
        same_core = dict(mk({"op": "S-", "set": ["booked"]}), semantic_hash="x")  # same state/event, other rule
        res = run({"X": {"rows": [row_of(rec)]}}, batteries={"b1": [dict(rec)], "b2": [same_core]})
        self.assertEqual(res["arms"]["X"]["overlap"], {"b1": {"semantic": 1, "skeleton": 1},
                                                       "b2": {"semantic": 0, "skeleton": 1}})
        self.assertEqual(set(res["failures"]), {"semantic_overlap:b1:X", "skeleton_overlap:b1:X",
                                                "skeleton_overlap:b2:X"})

    def test_token_covariates_and_max_len(self):
        rows = [row_of(r) for r in self.recs]
        res = run({"X": {"rows": rows}}, tok=WordTok(), seeds=(0, 2))
        tk = res["arms"]["X"]["tokens"]
        sup = [len(r["target"].split()) + 1 for r in rows]
        proc = [len(r["prompt"].split()) + s for r, s in zip(rows, sup)]
        for s in (0, 2):
            order = AE.row_order_for(40, AE.STEPS, AE.BATCH, s)[:AE.STEPS * AE.BATCH]
            self.assertEqual(tk["per_seed"][str(s)], {"supervised_tokens": sum(sup[i] for i in order),
                                                      "tokens_processed": sum(proc[i] for i in order)})
        self.assertEqual((tk["max_processed_per_row"], tk["rows_over_max_len"]), (max(proc), 0))
        small = run({"X": {"rows": rows}}, tok=WordTok(), max_len=max(proc) - 1)
        self.assertIn("over_max_len:X", small["failures"])
        self.assertNotIn("tokens", run({"X": {"rows": rows}})["arms"]["X"])


# ----------------------------------------------------------------------------- references and custody

class TestReferences(unittest.TestCase):
    def test_constants_match_native_sources(self):
        tree = ast.parse((E01N / "native_e01.py").read_text(encoding="utf-8"))
        facts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                names = [getattr(e, "id", None) for e in node.targets[0].elts]
                for nm in ("STD_BAL", "STD_AMT_ABS", "N_ROWS", "STEPS", "BATCH"):
                    if nm in names:
                        facts[nm] = ast.literal_eval(node.value.elts[names.index(nm)])
            elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_PROMPT_RE":
                facts["prompt_re"] = node.value.args[0].value
            elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("VERBATIM_CLAUSE",
                                                                                         "HARD_FLAGS"):
                facts[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(facts["prompt_re"], AE._PROMPT_RE.pattern)
        self.assertEqual((facts["STD_BAL"], facts["STD_AMT_ABS"]), (AE.STD_BAL, AE.STD_AMT_ABS))
        self.assertEqual((facts["N_ROWS"], facts["STEPS"], facts["BATCH"]), (AE.N_ROWS, AE.STEPS, AE.BATCH))
        self.assertEqual(facts["VERBATIM_CLAUSE"], AE.VERBATIM_CLAUSE)
        self.assertTrue(set(facts["HARD_FLAGS"]) <= set(AE.HARD_FLAGS))
        for f in (AE.rcl_system, AE.render_literal, AE.encode_pair, AE.row_order_for, AE.rcwm_heldout_texts):
            self.assertTrue(Path(f.__code__.co_filename).name in ("native_probe.py", "native_e01.py", "lora_cpu.py"))
        k = {"op": "K+", "kinds": ["add"]}
        self.assertEqual(AE.SOLO_FRAME.format(C=T.clause(k)), T.render_rule(k, "t0"))
        self.assertEqual(W._parse_clause(W._SOLO.match(AE.SOLO_FRAME.format(C=T.clause(k))).group("C")), k)
        self.assertEqual(_HEAVY_AFTER, _HEAVY_BEFORE)

    def test_default_batteries(self):
        specs = AE.battery_specs()
        self.assertEqual(specs["DEV2"][1:4], (30, 20261261, ("F1_void", "F2_unseen_real", "F3_unseen_nonce",
                                                               "F4_nearmiss", "F5_postedlike", "V1_pos", "V2_neg",
                                                               "V3_inverted", "V5_posted_override", "V7_bigset")))
        bats = AE.build_batteries()
        self.assertEqual({k: len(v) for k, v in bats.items()}, {"E01N_DEV": 6720, "DEV2": 660, "E01N_SEALED": 2240})
        sealed = "".join(json.dumps(r) + "\n" for r in bats["E01N_SEALED"]).encode()
        self.assertEqual(hashlib.sha256(sealed).hexdigest(),  # runs/RCX-E01N/ANALYSIS_SEALED.md (opened battery)
                         "c87765eaf275554946bc92513359cecfcc04c5eb16bf0b30d7ec80f33256df48")

    def test_require_audit(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                AE.require_audit_e01r(d, "R3", "abc")
            p = Path(d) / "AUDIT_E01R.json"
            p.write_text(json.dumps({"ok": True, "rows_sha256": {"R3": "abc"}}), encoding="utf-8")
            self.assertEqual(AE.require_audit_e01r(d, "R3", "abc"), hashlib.sha256(p.read_bytes()).hexdigest())
            for arm, s in (("R3", "abd"), ("R1", "abc")):
                with self.assertRaises(SystemExit):
                    AE.require_audit_e01r(d, arm, s)
            p.write_text(json.dumps({"ok": False, "rows_sha256": {"R3": "abc"}}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                AE.require_audit_e01r(d, "R3", "abc")

    @unittest.skipUnless((ROOT / "runs/RCX-E01N/cells/train_N-CFSD.jsonl").exists(), "N-CFSD rows not present")
    def test_real_control_rows_and_r3(self):
        """The N-CFSD rows on disk equal row_of(gen.make_train('CF_SD', 1501, 20261253)) byte for byte, and an R3 arm
        built on them passes every hard check against the default batteries and the RCWM references."""
        _cleanup_templates()
        p = ROOT / "runs/RCX-E01N/cells/train_N-CFSD.jsonl"
        self.assertTrue(hashlib.sha256(p.read_bytes()).hexdigest().startswith("de23a923"))
        ctl = AE.load_rows(p)
        recs = records(AE.N_ROWS)
        self.assertEqual([(r["prompt"], r["target"]) for r in ctl], [(x["prompt"], x["target"])
                                                                      for x in map(row_of, recs)])
        r3 = [dict(c, target=TT.trace_target(r)) for c, r in zip(ctl, recs)]
        rcwm = AE.RCWM_T1 if (AE.RCWM_T1 / "rcwmt" / "env.py").exists() else None
        res = AE.audit_e01r({"R3": {"rows": r3, "target": "trace", "match": "prompts"}}, control=ctl,
                            rcwm_texts=None if rcwm else {})
        self.assertTrue(res["ok"], res["failures"])
        self.assertEqual(res["control_failures"], [])
        self.assertEqual(res["arms"]["R3"]["rows_by_rule_type"], res["control"]["rows_by_rule_type"])
        if rcwm:
            self.assertTrue(res["rcwm_heldout_texts"])


class TestCli(unittest.TestCase):
    """audit_e01r.main on temporary row files (full 1,501-row arms built from the generator; fake pool files)."""

    @classmethod
    def setUpClass(cls):
        _cleanup_templates()
        R.register_pool(FAKE, "tr")
        cls.recs = records(AE.N_ROWS)
        cls.control = [row_of(r) for r in cls.recs]
        cls.r1 = [row_of(r) for r in pooled(cls.recs)]
        cls.r3 = [row_of(r, TT.trace_target(r)) for r in cls.recs]
        _cleanup_templates()

    def setUp(self):
        _cleanup_templates()
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.out, self.cdir, self.pdir = d / "cells", d / "control", d / "pools"
        for p in (self.out, self.cdir, self.pdir):
            p.mkdir()
        self.pool_sha = self.write_pool(True)
        man = {}
        for name, rows in (("E-R1", self.r1), ("E-R3", self.r3)):
            data = "".join(json.dumps(r) + "\n" for r in rows).encode()
            (self.out / f"train_{name}.jsonl").write_bytes(data)
            man[name] = {"path": f"train_{name}.jsonl", "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
        self.man = man
        self.write_manifest(man)
        data = "".join(json.dumps(r) + "\n" for r in self.control).encode()
        (self.cdir / "train_N-CFSD.jsonl").write_bytes(data)
        (self.cdir / "TRAIN_MANIFEST.json").write_text(json.dumps({"N-CFSD": {
            "sha256": hashlib.sha256(data).hexdigest(), "rows": len(self.control)}}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()
        _cleanup_templates()

    def write_manifest(self, arms, pool_sha=None):
        """native_e01r.build_cells' layout: per-arm entries under "arms", the R1 pool record under "r1_pool"."""
        man = {"experiment": "RCX-E01R", "r1_pool": {"sha256": pool_sha or self.pool_sha}, "arms": arms}
        (self.out / "TRAIN_MANIFEST_E01R.json").write_text(json.dumps(man), encoding="utf-8")

    def write_pool(self, meets):
        text = json.dumps({"schema": "rcx-e01r-pool-v1", "templates": FAKE}, indent=1) + "\n"
        (self.pdir / "TR_v9_audited.json").write_text(text, encoding="utf-8", newline="\n")
        sha = hashlib.sha256((self.pdir / "TR_v9_audited.json").read_bytes()).hexdigest()
        (self.pdir / "TR_v9_AUDIT.json").write_text(json.dumps({"outputs": {"audited": {"sha256": sha}},
                                                                "meets_minimum": meets}), encoding="utf-8")
        return sha

    def main(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return AE.main(["--out", str(self.out), "--control-dir", str(self.cdir), *extra]), buf.getvalue()

    def test_cli_pass_and_require(self):
        res, out = self.main("--arms", "E-R1,R3", "--pool", str(self.pdir / "TR_v9_audited.json"))  # R3 = alias
        self.assertTrue(res["ok"], res["failures"])
        self.assertEqual(res["control_failures"], [])
        saved = json.loads((self.out / "AUDIT_E01R.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["rows_sha256"], {k: v["sha256"] for k, v in self.man.items()})
        self.assertEqual(saved["pool"]["n"], 6)
        self.assertEqual(sum(saved["arms"]["E-R1"]["pooled_rows_by_cell"].values()),
                         saved["arms"]["E-R1"]["class_balance_by_wording_source"]["single:tr"]["n"])
        self.assertEqual(saved["arms"]["E-R3"]["control_mismatch_rows"], 0)
        self.assertEqual(set(saved["batteries"]), {"E01N_DEV", "DEV2", "E01N_SEALED"})
        self.assertIn(saved["dev2_battery_file_used"], (True, None))
        for arm in ("E-R1", "E-R3"):
            AE.require_audit_e01r(self.out, arm, self.man[arm]["sha256"])
        self.assertIn('"ok": true', out)

    def test_cli_failures(self):
        self.write_pool(False)
        self.write_manifest(dict(self.man, **{"E-R3": dict(self.man["E-R3"], rows=1500)}), pool_sha="0" * 64)
        with self.assertRaises(SystemExit):
            self.main("--arms", "E-R1,E-R3", "--pool", str(self.pdir / "TR_v9_audited.json"))
        saved = json.loads((self.out / "AUDIT_E01R.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["ok"])
        self.assertEqual(saved["failures"], ["pool_audit", "pool_manifest", "manifest_sha:E-R3"])
        with self.assertRaises(SystemExit):
            AE.require_audit_e01r(self.out, "E-R3", self.man["E-R3"]["sha256"])
        (self.pdir / "pool.json").write_bytes((self.pdir / "TR_v9_audited.json").read_bytes())
        for argv in (("--arms", "E-R1", "--pool", str(self.pdir / "pool.json")),
                     ("--arms", "E-R1", "--pool", str(self.pdir / "TR_v9_audited.json.secret")),
                     ("--arms", "E-R4"), ("--arms", ""), ("--arms", "E-R3,R3")):
            with self.assertRaises(SystemExit):
                self.main(*argv)


# ----------------------------------------------------------------------------- R1 training-pool audit (fake files)

class TestTrPool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.p2 = [{"id": "w1", "text": wording(42), "pol": "POS", "form": "IMPLICIT"},
                   {"id": "w2", "text": wording(43), "pol": "NEG", "form": "IMPLICIT"}]
        self.p1_extra = {"id": "w3", "text": wording(44), "pol": "NEG", "form": "IMPLICIT"}
        self.p3 = [{"id": "zsealedid1", "text": wording(45), "pol": "POS", "form": "IMPLICIT", "author": "Z"},
                   {"id": "zsealedid2", "text": wording(46), "pol": "NEG", "form": "EXPLICIT_LISTED_FIRST"}]
        cands = [{"id": f"r{c}{j}", "text": wording(7 * c + j), "pol": p, "form": f, "author": "T"}
                 for c, (p, f) in enumerate(CELLS) for j in range(7)]
        plant = {"x1": (wording(42, range(200, 208)), "p2"),       # 4-gram of a P2_v2 wording; Jaccard 0.33
                 "x2": (wording(44, [216]), "p2"),                 # P2 v1-only wording under the 0.5/5 rule
                 "x3": (wording(45, range(208, 216)), "p3"),       # 4-gram of a P3 wording
                 "x4": (f"Cwq {{L}} settled {pseudo(218)}.", "status"),
                 "x5": (f"Cwq {{L}} posting {pseudo(220)}.", "status"),
                 "x6": (T.SPOS["t0"], "ref"),
                 "x7": (wording(47), None), "x8": (wording(47, [217]), "cluster")}
        self.expect = {k: v for k, (_, v) in plant.items()}
        cands += [{"id": k, "text": t, "pol": "POS", "form": "IMPLICIT", "author": "T"} for k, (t, _) in plant.items()]
        self.cands = cands
        self.write_inputs()

    def tearDown(self):
        self.tmp.cleanup()

    def write_inputs(self, p3_bytes=None):
        d = self.d
        doc = lambda tpls, **kw: json.dumps({"schema": "rcx-e01r-pool-v1", **kw, "templates": tpls})  # noqa: E731
        (d / "TR_v1_candidates.json").write_text(doc(self.cands), encoding="utf-8")
        (d / "P2_DEV_v2.json").write_text(doc(self.p2), encoding="utf-8")
        (d / "P2_DEV.json").write_text(doc(self.p2 + [self.p1_extra]), encoding="utf-8")
        p3 = (doc(self.p3, salt="s") + "\n").encode() if p3_bytes is None else p3_bytes
        (d / "P3_SEALED_v2.json.secret").write_bytes(p3)
        sha = lambda p: hashlib.sha256((d / p).read_bytes()).hexdigest()  # noqa: E731
        refine = {"outputs": {"P2": {"sha256": sha("P2_DEV_v2.json"), "n": 2},
                              "P3": {"sha256": hashlib.sha256(doc(self.p3, salt="s").encode() + b"\n").hexdigest(),
                                     "n": 2}},
                  "kept_by_cell": {"P3": {"POS/IMPLICIT": 1, "NEG/EXPLICIT_LISTED_FIRST": 1}}}
        (d / "POOL_REFINE.json").write_text(json.dumps(refine), encoding="utf-8")
        (d / "POOL_SPLIT.json").write_text(json.dumps({"p2_sha256": sha("P2_DEV.json")}), encoding="utf-8")

    def argv(self, **over):
        d = self.d
        a = {"--candidates": d / "TR_v1_candidates.json", "--p2": d / "P2_DEV_v2.json", "--p2-v1": d / "P2_DEV.json",
             "--p3": d / "P3_SEALED_v2.json.secret", "--refine": d / "POOL_REFINE.json",
             "--split": d / "POOL_SPLIT.json", "--out-dir": d / "out"}
        a.update(over)
        for v in a.values():  # every path is inside the temp dir: no real (sealed) file is ever opened
            self.assertTrue(Path(v).resolve().is_relative_to(d.resolve()), v)
        return [x for k, v in a.items() for x in (k, str(v))]

    def main(self, **over):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            meta = ATP.main(self.argv(**over))
        return meta, buf.getvalue()

    def p3_strings(self):
        return [t["id"] for t in self.p3] + [w for t in self.p3 for w in WA.words(t["text"])]

    def test_end_to_end_counts_and_custody(self):
        meta, out = self.main()
        o = self.d / "out"
        audited = json.loads((o / "TR_v1_audited.json").read_text(encoding="utf-8"))
        drops = json.loads((o / "TR_v1_P3_drops.json.secret").read_text(encoding="utf-8"))
        aud_text = (o / "TR_v1_AUDIT.json").read_text(encoding="utf-8")
        kept = {t["id"] for t in audited["templates"]}
        self.assertEqual(kept, {t["id"] for t in self.cands} - {k for k, v in self.expect.items() if v})
        self.assertEqual([t["id"] for t in drops["templates"]], ["x3"])
        self.assertEqual(meta["totals"], {"candidates": 50, "status": 2, "ref": 1, "p2": 2, "p3": 1, "cluster": 1,
                                          "kept": 43})
        self.assertEqual(meta["by_cell"]["POS/IMPLICIT"]["kept"], 8)
        self.assertTrue(meta["meets_minimum"])
        for name, k in (("TR_v1_audited.json", "audited"), ("TR_v1_P3_drops.json.secret", "p3_drops")):
            self.assertEqual(meta["outputs"][k]["sha256"], hashlib.sha256((o / name).read_bytes()).hexdigest())
        self.assertEqual(meta["inputs"]["P3_v2"]["n"], 2)
        tokens = lambda s: set(re.findall(r"[a-z0-9_]+", s.lower()))  # noqa: E731
        visible = out + aud_text + (o / "TR_v1_audited.json").read_text(encoding="utf-8")
        self.assertFalse(tokens(visible) & set(self.p3_strings()))  # no P3 id or word in anything visible
        secret = (o / "TR_v1_P3_drops.json.secret").read_text(encoding="utf-8")
        self.assertFalse(tokens(secret) & ({t["id"] for t in self.p3} | set(WA.words(self.p3[1]["text"]))))
        self.assertEqual(json.loads(secret)["templates"], [t for t in self.cands if t["id"] == "x3"])
        self.assertNotIn("x3", out)  # stdout: counts only, no ids
        before = {p.name: p.read_bytes() for p in o.iterdir()}
        with self.assertRaises(SystemExit):
            self.main()  # create-only
        self.assertEqual(before, {p.name: p.read_bytes() for p in o.iterdir()})

    def test_minimum_per_cell(self):
        self.cands = [t for t in self.cands if not (t["pol"] == "NEG" and t["form"] == "IMPLICIT" and t["id"] > "r52")]
        self.write_inputs()
        meta, out = self.main()
        self.assertFalse(meta["meets_minimum"])
        self.assertEqual(meta["cells_below_minimum"], ["NEG/IMPLICIT"])
        self.assertEqual(meta["by_cell"]["NEG/IMPLICIT"]["kept"], 3)
        self.assertIn('"meets_minimum": false', out)

    def test_p3_custody_refusals(self):
        tokens = lambda s: set(re.findall(r"[a-z0-9_]+", s.lower()))  # noqa: E731
        bad = json.dumps({"templates": self.p3[::-1]}).encode()
        self.write_inputs(p3_bytes=bad)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            ATP.main(self.argv())
        self.assertIn("sha256", str(cm.exception))
        self.assertFalse(tokens(str(cm.exception) + buf.getvalue()) & set(self.p3_strings()))
        self.assertFalse((self.d / "out").exists())
        self.write_inputs()
        (self.d / "P3_copy.json").write_bytes((self.d / "P3_SEALED_v2.json.secret").read_bytes())
        with self.assertRaises(SystemExit):
            self.main(**{"--p3": self.d / "P3_copy.json"})  # the sealed file must be a *.secret file
        for name in ("P3_SEALED_v2.json", "SEALED_like.json", "x.json.secret"):
            (self.d / name).write_bytes((self.d / "P2_DEV_v2.json").read_bytes())
            with self.assertRaises(SystemExit):
                self.main(**{"--p2": self.d / name})  # never a sealed-looking file as a visible reference
        refine = json.loads((self.d / "POOL_REFINE.json").read_text(encoding="utf-8"))
        refine["kept_by_cell"]["P3"] = {"POS/IMPLICIT": 2}
        (self.d / "POOL_REFINE.json").write_text(json.dumps(refine), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            self.main()
        self.assertIn("counts differ", str(cm.exception))
        self.assertFalse(tokens(str(cm.exception)) & set(self.p3_strings()))
        self.assertFalse((self.d / "out").exists())

    def test_candidate_refusals(self):
        for bad in ({"id": "h01", "text": wording(60), "pol": "POS", "form": "IMPLICIT"},
                    {"id": "Hx", "text": wording(60), "pol": "POS", "form": "IMPLICIT"},
                    {"id": "r00", "text": wording(60), "pol": "POS", "form": "IMPLICIT"},
                    {"id": "b1", "text": "No slot here.", "pol": "POS", "form": "IMPLICIT"},
                    {"id": "b2", "text": wording(60), "pol": "pos", "form": "IMPLICIT"}):
            with self.subTest(bad=bad["id"]):
                cands = self.cands
                self.cands = cands + [bad]
                self.write_inputs()
                with self.assertRaises(SystemExit):
                    self.main()
                self.cands = cands
        self.write_inputs()
        (self.d / "TR_x.json").write_bytes((self.d / "TR_v1_candidates.json").read_bytes())
        with self.assertRaises(SystemExit):
            self.main(**{"--candidates": self.d / "TR_x.json"})
        self.assertFalse((self.d / "out").exists())

    def test_audit_order_and_thresholds(self):
        c = [{"id": "a", "text": wording(42, range(200, 208)), "pol": "POS", "form": "IMPLICIT"},  # P2 and P3 at once
             {"id": "b", "text": wording(56, range(300, 308)), "pol": "POS", "form": "IMPLICIT"}]
        p2, p3 = [wording(42)], [wording(42), wording(56)]
        surv, drops = ATP.audit(c, p2, p3)
        self.assertEqual((surv, drops), ([], {"a": "p2", "b": "p3"}))  # p3 tested only after the visible checks
        self.assertLess(WA.jaccard(c[1]["text"], wording(56)), 0.35)  # the 4-gram alone triggers the stricter rule
        self.assertFalse(WA.near_dup(c[1]["text"], wording(56)))       # ... which the 0.5/5 rule would miss
        why = ATP.status_reasons("Never when {L} is not posted.")
        self.assertEqual((why[0], why[-1]), ("posted", "not posted"))  # word, stems, then the multi-word phrase
        self.assertTrue(any(w.startswith("posted~") for w in why))
        self.assertEqual(ATP.status_reasons("Solely {L} events get to move money."), [])
        self.assertEqual(ATP.EXCLUDED_PHRASES, ["not posted"])

    def test_original_refs_ignore_registered_wordings(self):
        base = ATP.original_refs()
        self.assertEqual(base, WA.reference_wordings())  # nothing registered
        try:
            R.register_pool(FAKE, "tr")
            self.assertEqual(ATP.original_refs(), base)
            self.assertIn("SPOS:tr:q0", WA.reference_wordings())
            self.assertEqual(ATP.visible_drops(FAKE, [], []), {})  # a pool wording is not its own near-duplicate
        finally:
            _cleanup_templates()
        self.assertTrue({"SPOS:t0", "SNEG:h1", "AND:h0", "OR:t1", "CLAUSE:S+"} <= set(base))
        self.assertEqual((ATP.STRICT, ATP.REF, ATP.POOL, ATP.MIN_PER_CELL),
                         ({"jac": 0.35, "n": 4}, {"jac": 0.5, "n": 5}, {"jac": 0.6, "n": 6}, 6))


if __name__ == "__main__":
    unittest.main()
