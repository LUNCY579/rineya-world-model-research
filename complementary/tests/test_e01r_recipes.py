"""Fast tests for the RCX-E01R training recipes (experiments/RCX-E01R/recipes.py). Pure Python: no torch, no tokenizer
(a fake chat template), no model; a small fake training pool (the real one is not authored yet). Reads only
pools/P2_DEV.json and pools/P2_DEV_v2.json, and the source text of two E01N scripts (parsed, never imported).

Run: python -B -m unittest tests.test_e01r_recipes -v   (CPU, single thread; a few seconds)
"""
import ast
import importlib.util
import json
import os
import random
import re
import sys
import time
import types
import unittest
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcx.rcl import gen, templates as T, vocab as V, world as W  # noqa: E402

HEAVY = ("torch", "transformers")
_HEAVY_BEFORE = {m for m in HEAVY if m in sys.modules}
_spec = importlib.util.spec_from_file_location("rcx_e01r_recipes", ROOT / "experiments" / "RCX-E01R" / "recipes.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
_HEAVY_AFTER = {m for m in HEAVY if m in sys.modules}

E01N = ROOT / "experiments" / "RCX-E01N"
P2_V2 = ROOT / "experiments" / "RCX-E01R" / "pools" / "P2_DEV_v2.json"
CF_VARIANTS = ("base", "rule_flip", "status_swap", "set_swap")
TRAIN_WORDS = set(V.REAL_TRAIN) | set(V.NONCE["train"]) | {gen.POSTED, gen.VOID}
FAKE_POOL = [
    {"id": "f01", "text": "A {L} event moves money, and an event bearing any other status moves none.", "pol": "POS",
     "form": "EXPLICIT_LISTED_FIRST"},
    {"id": "f02", "text": "By default no event moves money, yet an event marked {L} does.", "pol": "POS",
     "form": "EXPLICIT_COMPLEMENT_FIRST"},
    {"id": "f03", "text": "Solely {L} events get to move money.", "pol": "POS", "form": "IMPLICIT"},
    {"id": "f04", "text": "A {L} event moves no money, and an event bearing any other status moves money.",
     "pol": "NEG", "form": "EXPLICIT_LISTED_FIRST"},
    {"id": "f05", "text": "By default an event moves money, unless it is marked {L}.", "pol": "NEG",
     "form": "EXPLICIT_COMPLEMENT_FIRST"},
    {"id": "f06", "text": "Money moves for everything but {L} events.", "pol": "NEG", "form": "IMPLICIT"},
]


def _source_tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _native_probe_funcs():
    """rcl_system and user_json from native_probe.py, compiled alone (the module imports torch at top level)."""
    fns = [n for n in _source_tree(E01N / "native_probe.py").body
           if isinstance(n, ast.FunctionDef) and n.name in ("rcl_system", "user_json")]
    assert len(fns) == 2
    ns = {"json": json}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "native_probe.py", "exec"), ns)
    return types.SimpleNamespace(rcl_system=ns["rcl_system"], user_json=ns["user_json"])


def _native_e01_facts():
    """N_ROWS, the read-back prompt regex and the row keys of native_rows, from native_e01.py's source."""
    out = {}
    for node in _source_tree(E01N / "native_e01.py").body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [getattr(e, "id", None) for e in node.targets[0].elts]
            if "N_ROWS" in names:
                out["N_ROWS"] = node.value.elts[names.index("N_ROWS")].value
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_PROMPT_RE":
            out["prompt_re"] = re.compile(node.value.args[0].value, re.S)
        elif isinstance(node, ast.FunctionDef) and node.name == "native_rows":
            for d in ast.walk(node):
                if isinstance(d, ast.Dict) and any(getattr(k, "value", None) == "prompt" for k in d.keys):
                    out["row_keys"] = {k.value for k in d.keys}
    return out


class FakeTok:
    """Qwen3-style chat-template text only (no vocabulary, nothing loaded)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return ("".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def by_parent(recs):
    out = defaultdict(list)
    for r in recs:
        out[r["parent_id"]].append(r)
    return out


def dumps(recs):
    return json.dumps(recs, sort_keys=True)


def check_gold(tc, recs, sample=300):
    """Gold, targets and input are those of rcx.rcl.world for every record; eval_rule cross-checked on a sample."""
    for r in recs:
        ok, after = W.transition(r["rule_ast"], r["state"], r["event"])
        tc.assertEqual((r["gold"]["ok"], r["gold"]["after"]), (ok, after))
        tc.assertEqual(r["gold"]["decision"], "apply" if ok else "noop")
        tc.assertEqual(r["gold"]["listed"], W.listed(r["rule_ast"], r["event"]))
        tc.assertEqual(r["targets"]["ans"], W.target_ans(after, r["state_order"]))
        tc.assertEqual(r["input"], W.render_input(r["rule_ast"], r["state"], r["event"], r["tid"], r["state_order"],
                                                  list(r["event"])))
        tc.assertEqual(list(r["state"]), r["state_order"])
    for r in random.Random(20261264).sample(recs, min(sample, len(recs))):
        tc.assertEqual(W.eval_rule(r["rule_ast"], r["state"], r["event"]), r["gold"]["decision"] == "apply")


def check_schema(tc, recs):
    ref = gen.make_train("CF_SD", 4, 1)[0]
    for r in recs:
        tc.assertEqual(set(r), set(ref))
        tc.assertEqual(set(r["gold"]), set(ref["gold"]))
        tc.assertEqual(set(r["targets"]), set(ref["targets"]))
        tc.assertEqual((r["split"], r["surface"]), ("train", "sd"))


class TestPool(unittest.TestCase):
    def tearDown(self):
        R.unregister_pool("tr")
        R.unregister_pool("tq")

    def test_fake_pool_is_clean(self):
        self.assertEqual(R.audit_pool(FAKE_POOL), {})

    def test_register_and_render(self):
        ids = R.register_pool(FAKE_POOL)
        self.assertEqual(ids["f01"], "tr:f01")
        self.assertEqual(T.SPOS["tr:f03"], FAKE_POOL[2]["text"])
        self.assertEqual(T.SNEG["tr:f06"], FAKE_POOL[5]["text"])
        self.assertNotIn("tr:f01", T.SNEG)
        self.assertEqual(T.render_rule({"op": "S+", "set": ["kavo", "booked"]}, "tr:f03"),
                         "Solely kavo or booked events get to move money.")
        R.register_pool(FAKE_POOL)  # identical re-registration is a no-op
        self.assertEqual(R.unregister_pool("tr"), 6)
        self.assertFalse([k for k in list(T.SPOS) + list(T.SNEG) if k.startswith("tr:")])

    def test_register_refuses_bad_templates(self):
        good = FAKE_POOL[0]
        bads = [dict(good, id="b1", text="Apply {L} events and skip {L} ones."),
                dict(good, id="b2", text="Apply every event."),
                dict(good, id="b3", text="Apply {L} events {now}."),
                dict(good, id="b4", text="Apply {L} events; skip the rest }."),
                dict(good, id="b5", text="Apply {L} events.\nSkip the rest."),
                dict(good, id="b6", text="{L}"),
                dict(good, id="b7", text=None),
                dict(good, id="b8", text="Apply {L1} events."),
                dict(good, id="b9", pol="pos"),
                dict(good, id="b10", form="EXPLICIT"),
                {"id": "b11", "text": "Apply {L} events.", "pol": "POS"},
                dict(good, id="b:12"),
                dict(good, id=""),
                dict(good, id=7),
                "Apply {L} events."]
        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    R.register_pool([FAKE_POOL[1], bad])
                self.assertNotIn("tr:f02", T.SPOS)  # all or nothing

    def test_register_refuses_id_collisions(self):
        with self.assertRaises(ValueError):
            R.register_pool([FAKE_POOL[0], dict(FAKE_POOL[1], id="f01")])  # repeated in the pool
        R.register_pool(FAKE_POOL[:1])
        with self.assertRaises(ValueError):
            R.register_pool([dict(FAKE_POOL[0], text="Only {L} events move money at all.")])  # other text, same id
        with self.assertRaises(ValueError):
            R.register_pool([dict(FAKE_POOL[0], pol="NEG")])  # same id, other polarity
        for prefix in ("p2", "p3", "TR", "t r", ""):
            with self.assertRaises(ValueError):
                R.register_pool(FAKE_POOL[:1], prefix=prefix)
        R.register_pool([dict(FAKE_POOL[0], text="Only {L} events move money at all.")], prefix="tq")
        self.assertEqual(T.SPOS["tq:f01"], "Only {L} events move money at all.")

    def test_audit_flags_every_p2_wording_and_edits(self):
        p2 = json.loads(P2_V2.read_text(encoding="utf-8"))["templates"]
        self.assertGreaterEqual(len(p2), 45)
        for t in p2:
            bad = R.audit_pool([dict(t, id="cp")])
            self.assertIn(f"near_dup:P2_DEV_v2:{t['id']}", bad.get("cp", []), t["id"])
        edited = dict(p2[0], id="ed", text=p2[0]["text"].replace("balances", "accounts", 1) + " Always.")
        self.assertTrue(any(x.startswith("near_dup:P2_DEV_v2:") for x in R.audit_pool([edited])["ed"]))

    def test_audit_flags_heldout_status_words_and_stems(self):
        def tpl(i, text):
            return {"id": i, "text": text, "pol": "POS", "form": "IMPLICIT"}

        bad = R.audit_pool([tpl("a", "Balances change only for {L} events."), tpl("b", "Solely {L} events get paid."),
                            tpl("c", "Solely {L} events get posting rights.")])
        self.assertIn("near_dup:SPOS:h1", bad["a"])
        self.assertIn("status_word:paid", bad["b"])
        self.assertTrue(any(x.startswith("status_stem:posting~") for x in bad["c"]))
        self.assertNotIn("c", R.audit_pool([tpl("c", "Solely {L} events get posting rights.")], stems=False))

    def test_make_r1_refuses_p2_copies(self):
        p2 = json.loads(P2_V2.read_text(encoding="utf-8"))["templates"]
        for t in (p2[0], p2[-1]):
            with self.subTest(p2=t["id"]):
                with self.assertRaises(ValueError) as cm:
                    R.make_r1(50, 1, FAKE_POOL + [dict(t, id="cp1")])
                self.assertIn(f"near_dup:P2_DEV_v2:{t['id']}", str(cm.exception))
                self.assertFalse([k for k in list(T.SPOS) + list(T.SNEG) if k.startswith("tr:")])
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, [])
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, FAKE_POOL, p_pool=1.5)


class TestR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = R.make_r1(R.N_ROWS, R.SEEDS["R1"], FAKE_POOL)

    @classmethod
    def tearDownClass(cls):
        R.unregister_pool("tr")

    def test_rows_and_determinism(self):
        self.assertEqual(len(self.recs), R.N_ROWS)
        for n in (1, 5, 77):
            self.assertEqual(len(R.make_r1(n, 3, FAKE_POOL)), n)
        self.assertEqual(dumps(R.make_r1(R.N_ROWS, R.SEEDS["R1"], FAKE_POOL)), dumps(self.recs))
        self.assertNotEqual(dumps(R.make_r1(200, 4, FAKE_POOL)), dumps(R.make_r1(200, 5, FAKE_POOL)))
        self.assertEqual(dumps(R.make_r1(200, 4, FAKE_POOL)), dumps(R.make_r1(200, 4, FAKE_POOL[::-1])))

    def test_schema_and_gold(self):
        check_schema(self, self.recs)
        check_gold(self, self.recs)

    def test_same_as_cf_sd_except_wording(self):
        gen.use_world("arith")
        cf = gen.make_train("CF_SD", R.N_ROWS, R.SEEDS["R1"])
        changed = 0
        for a, b in zip(self.recs, cf):
            for k in set(b) - {"tid", "input", "group", "parent_id"}:
                self.assertEqual(a[k], b[k], k)
            self.assertEqual((a["group"], a["parent_id"]), ("R1", b["parent_id"].replace("tr-CF_SD-", "tr-R1-")))
            if a["tid"] == b["tid"]:
                self.assertEqual(a["input"], b["input"])
            else:
                changed += 1
                self.assertTrue(a["tid"].startswith("tr:") and b["tid"] in T.TRAIN_IDS)
        self.assertGreater(changed, 0)
        p0 = R.make_r1(200, 9, FAKE_POOL, p_pool=0.0)
        self.assertEqual([r["input"] for r in p0], [r["input"] for r in gen.make_train("CF_SD", 200, 9)])

    def test_pool_wordings_only_on_single_atoms(self):
        single = [r for r in self.recs if r["rule_ast"]["op"] in ("S+", "S-")]
        pooled = [r for r in single if r["tid"].startswith("tr:")]
        self.assertFalse([r for r in self.recs if r["rule_ast"]["op"] not in ("S+", "S-") and ":" in r["tid"]])
        self.assertTrue(0.42 <= len(pooled) / len(single) <= 0.58, len(pooled) / len(single))
        texts = {f"tr:{t['id']}": t for t in FAKE_POOL}
        for r in pooled:
            t = texts[r["tid"]]
            self.assertEqual(R.POL_OP[t["pol"]], r["rule_ast"]["op"])
            rule = t["text"].format(L=T.render_list(r["rule_ast"]["set"]))
            self.assertEqual(r["input"].split("\n")[0], "RULE: " + rule)
        self.assertEqual({r["tid"] for r in pooled}, set(texts))
        self.assertTrue({r["tid"] for r in single if not r["tid"].startswith("tr:")} <= set(T.TRAIN_IDS))


class TestR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        cls.parents = by_parent(cls.recs)

    def test_row_count_exact(self):
        self.assertEqual(R.N_ROWS, _native_e01_facts()["N_ROWS"])
        self.assertEqual(len(self.recs), R.N_ROWS)
        for n in (0, 1, 2, 9, 10, 11, 57, 400):
            self.assertEqual(len(R.make_r2(n, 7)), n)

    def test_determinism(self):
        self.assertEqual(dumps(R.make_r2(R.N_ROWS, R.SEEDS["R2"])), dumps(self.recs))
        self.assertNotEqual(dumps(R.make_r2(300, 11)), dumps(R.make_r2(300, 12)))

    def test_schema_and_gold(self):
        check_schema(self, self.recs)
        check_gold(self, self.recs)

    def test_complement_twins_gold(self):
        unl = [r for r in self.recs if r["variant"] == "comp_unlisted"]
        lst = [r for r in self.recs if r["variant"] == "comp_listed"]
        self.assertGreater(len(unl), 100)
        self.assertEqual(len(unl), len(lst))
        for r in unl:
            self.assertEqual(r["rule_ast"]["op"], "S+")
            self.assertNotIn(r["event"]["status"], r["rule_ast"]["set"])
            self.assertEqual((r["gold"]["decision"], r["gold"]["listed"]), ("noop", False))
        for r in lst:
            self.assertEqual(r["rule_ast"]["op"], "S+")
            self.assertIn(r["event"]["status"], r["rule_ast"]["set"])
            self.assertEqual((r["gold"]["decision"], r["gold"]["listed"]), ("apply", True))

    def test_balance_wordings_and_pairs(self):
        n_comp = 0
        for pid, recs in self.parents.items():
            base = [r for r in recs if r["variant"] == "base"]
            comp = [r for r in recs if r["variant"].startswith("comp_")]
            if not base:
                continue
            if base[0]["rule_ast"]["op"] != "S+":
                self.assertEqual(comp, [], pid)
                continue
            n_comp += 1
            rule = base[0]["rule_ast"]
            tids = defaultdict(dict)
            for r in comp:
                self.assertEqual(r["rule_ast"], rule)
                self.assertNotIn(r["variant"], tids[r["tid"]])
                tids[r["tid"]][r["variant"]] = r
            self.assertIn(len(tids), R.COMP_K, pid)
            self.assertTrue(set(tids) <= set(T.TRAIN_IDS))
            others = {r["event"]["status"] for r in recs if not r["variant"].startswith("comp_")}
            fresh = [p["comp_unlisted"]["event"]["status"] for p in tids.values()]
            self.assertEqual(len(set(fresh)), len(fresh))
            self.assertFalse(set(fresh) & (others | set(rule["set"])), pid)
            for tid, p in tids.items():
                self.assertEqual(set(p), {"comp_unlisted", "comp_listed"})
                u, li = p["comp_unlisted"], p["comp_listed"]
                self.assertEqual((u["state"], u["state_order"], u["tid"]), (li["state"], li["state_order"], tid))
                self.assertEqual({k: v for k, v in u["event"].items() if k != "status"},
                                 {k: v for k, v in li["event"].items() if k != "status"})
                self.assertEqual(u["skeleton_hash"], base[0]["skeleton_hash"])
        self.assertGreater(n_comp, 50)

    def test_whole_parents_and_fill(self):
        order = [r["parent_id"] for r in self.recs]
        seen = set()
        for i, pid in enumerate(order):  # each parent's records are contiguous
            if i and pid != order[i - 1]:
                self.assertNotIn(pid, seen)
            seen.add(pid)
        for n in (R.N_ROWS, 9, 57, 400):
            recs = R.make_r2(n, 7) if n != R.N_ROWS else self.recs
            fill_started = False
            for pid, prs in by_parent(recs).items():
                vs = [r["variant"] for r in prs]
                if vs == ["fill_base"]:
                    fill_started = True
                    continue
                self.assertFalse(fill_started, "a full parent after a fill parent")
                self.assertEqual(vs[:4], list(CF_VARIANTS))
                self.assertEqual(len(vs), 4 + (2 * len({r["tid"] for r in prs if r["variant"] == "comp_listed"})))
            self.assertLess(sum(r["variant"] == "fill_base" for r in recs), 4 + 2 * max(R.COMP_K))

    def test_train_words_only(self):
        for r in self.recs:
            self.assertIn(r["event"]["status"], TRAIN_WORDS)
            sa = W.status_atom(r["rule_ast"])
            self.assertTrue(set(sa["set"] if sa else ()) <= TRAIN_WORDS)
            self.assertNotIn(":", r["tid"])

    def test_fast_and_no_heavy_imports(self):
        t0 = time.time()
        R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        self.assertLess(time.time() - t0, 5.0)
        t0 = time.time()
        R.make_r1(R.N_ROWS, R.SEEDS["R1"], FAKE_POOL)
        self.assertLess(time.time() - t0, 5.0)
        R.unregister_pool("tr")
        self.assertEqual(_HEAVY_AFTER, _HEAVY_BEFORE)
        mods = set()
        for node in ast.walk(_source_tree(ROOT / "experiments" / "RCX-E01R" / "recipes.py")):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertFalse(mods & set(HEAVY))


class TestStatsAndRows(unittest.TestCase):
    def tearDown(self):
        R.unregister_pool("tr")

    def test_recipe_stats(self):
        r2 = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        s = R.recipe_stats(r2)
        json.dumps(s)
        self.assertEqual((s["rows"], s["parents"]), (R.N_ROWS, len(by_parent(r2))))
        self.assertEqual(s["variants"]["comp_listed"], s["variants"]["comp_unlisted"])
        self.assertEqual(s["tid_family"], {"t*": R.N_ROWS})
        gen.use_world("arith")
        cf = R.recipe_stats(gen.make_train("CF_SD", R.N_ROWS, 20261253))  # N-CFSD's DATA_SEED
        self.assertGreater(s["spos_unlisted_share"], cf["spos_unlisted_share"] + 0.05)
        self.assertTrue(0.4 < s["noop_share"] < 0.6)
        self.assertEqual(s["spos_unlisted_rows"], sum(r["rule_ast"]["op"] == "S+" and r["gold"]["decision"] == "noop"
                                                      for r in r2))
        s1 = R.recipe_stats(R.make_r1(300, 2, FAKE_POOL))
        self.assertGreater(s1["tid_family"]["tr"], 0)
        self.assertEqual(sum(s1["tids"].values()), 300)

    def test_native_rows_fake_tokenizer(self):
        facts, np_mod = _native_e01_facts(), _native_probe_funcs()
        pre, post = np_mod.rcl_system("\x00").split("\x00")
        recs = R.make_r2(120, 5) + R.make_r1(120, 6, FAKE_POOL)
        rows = R.native_rows(recs, FakeTok(), "RX", np_mod=np_mod)
        self.assertEqual(len(rows), len(recs))
        for row, r in zip(rows, recs):
            self.assertEqual(set(row), facts["row_keys"])
            m = facts["prompt_re"].fullmatch(row["prompt"])
            self.assertIsNotNone(m)
            system, user = m.group(1), m.group(2)
            self.assertTrue(system.startswith(pre) and system.endswith(post))
            sentence = system[len(pre):len(system) - len(post)]
            self.assertEqual(sentence, T.render_rule(r["rule_ast"], r["tid"]))
            if ":" not in r["tid"]:
                self.assertEqual(W.parse_rule(sentence)[0], r["rule_ast"])
            u = json.loads(user)
            self.assertEqual(u["before"], r["state"])
            self.assertEqual({k: v for k, v in u["event"].items() if k != "tick"}, r["event"])
            self.assertEqual(json.loads(row["target"]), r["gold"]["after"])
            self.assertEqual((row["decision"], row["status"], row["regime"]),
                             (r["gold"]["decision"], r["event"]["status"], "RX"))
        self.assertTrue(any(":" in r["tid"] for r in recs))


if __name__ == "__main__":
    unittest.main()
