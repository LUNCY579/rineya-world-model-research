"""Fast tests for the RCX-E01R training recipes (experiments/RCX-E01R/recipes.py). Pure Python: no torch, no tokenizer
(a fake chat template), no model; a fake training pool (36 templates, 6 per cell). Reads only pools/P2_DEV.json and
pools/P2_DEV_v2.json, the source text of two E01N scripts (parsed, never imported) and, if present, the N-CFSD rows
runs/RCX-E01N/cells/train_N-CFSD.jsonl (read-only). Sealed material is only ever faked, in a temp directory.

Run: python -B -m unittest tests.test_e01r_recipes -v   (CPU, single thread; a few seconds)
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import random
import re
import sys
import tempfile
import time
import types
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

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
N_CFSD_ROWS = ROOT / "runs" / "RCX-E01N" / "cells" / "train_N-CFSD.jsonl"
N_CFSD_ROWS_SHA = "de23a9233785273b429d5debd311b6309fccedfc1964ed61f1d3faedcd72e401"  # PREREG_E01R section 3
N_CFSD_DATA_SEED = 20261253
CF_VARIANTS = ("base", "rule_flip", "status_swap", "set_swap")
TRAIN_WORDS = set(V.REAL_TRAIN) | set(V.NONCE["train"]) | {gen.POSTED, gen.VOID}
_P, _N = "POS", "NEG"
_LF, _CF, _IM = R.FORMS
FAKE_POOL = [{"id": i, "text": t, "pol": p, "form": f} for i, t, p, f in [
    ("f01", "A {L} event moves money, and an event bearing any other status moves none.", _P, _LF),
    ("f02", "By default no event moves money, yet an event marked {L} does.", _P, _CF),
    ("f03", "Solely {L} events get to move money.", _P, _IM),
    ("f04", "Entries marked {L} move no money, whereas the remaining entries move it as usual.", _N, _LF),
    ("f05", "By default an event moves money, unless it is marked {L}.", _N, _CF),
    ("f06", "Money moves for everything but {L} events.", _N, _IM),
    ("f07", "Money moves for {L} events; for every remaining event it stays put.", _P, _LF),
    ("f08", "Most entries are left alone; those labelled {L} get carried through.", _P, _CF),
    ("f09", "Just the {L} entries count.", _P, _IM),
    ("f10", "A {L} item is skipped and anything else is run.", _N, _LF),
    ("f11", "Ordinarily every entry is run, but {L} entries are set aside.", _N, _CF),
    ("f12", "Skip any {L} entry.", _N, _IM),
    ("f13", "Let {L} events shift funds and leave all others idle.", _P, _LF),
    ("f14", "Nothing happens for ordinary entries, but entries with the label {L} are run.", _P, _CF),
    ("f15", "Funds shift exclusively for {L} items.", _P, _IM),
    ("f16", "Items labelled {L} stay idle; all others shift funds.", _N, _LF),
    ("f17", "All items shift funds except where they carry the label {L}.", _N, _CF),
    ("f18", "Only {L} items are kept out of the run.", _N, _IM),
    ("f19", "An entry labelled {L} gets run, while one with a different label sits untouched.", _P, _LF),
    ("f20", "Leave every item idle unless it is marked {L}, in which case process it.", _P, _CF),
    ("f21", "Run nothing beyond the {L} entries.", _P, _IM),
    ("f22", "Hold back the {L} entries and carry out the rest.", _N, _LF),
    ("f23", "Everything goes through, apart from what is marked {L}.", _N, _CF),
    ("f24", "The {L} label blocks an entry; nothing else does.", _N, _IM),
    ("f25", "Process {L} items as normal and discard every other item unprocessed.", _P, _LF),
    ("f26", "The funds stay where they are for all events except {L} ones, which shift them.", _P, _CF),
    ("f27", "The {L} label alone unlocks any movement.", _P, _IM),
    ("f28", "Funds never move on {L} items, though they move on every other item.", _N, _LF),
    ("f29", "Money normally moves, but not for an event with label {L}.", _N, _CF),
    ("f30", "Funds shift for any item not labelled {L}.", _N, _IM),
    ("f31", "Entries whose label reads {L} count toward the totals; the rest count for nothing.", _P, _LF),
    ("f32", "Other labels change no figures; the label {L} changes them.", _P, _CF),
    ("f33", "Movement of money is reserved for {L} items.", _P, _IM),
    ("f34", "Under the label {L} nothing is done; under any other label the entry runs.", _N, _LF),
    ("f35", "Run each entry as a rule; an entry marked {L} is the exception and stays idle.", _N, _CF),
    ("f36", "Every entry runs save the {L} ones.", _N, _IM)]]
NEAR_F07 = {"id": "g07", "text": "Money moves for {L} events; for every remaining event it stays where it is.",
            "pol": _P, "form": _LF}  # a within-pool near-duplicate of f07 (shared 6-word sequence), otherwise clean


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


def registered(prefix="tr"):
    return [k for k in list(T.SPOS) + list(T.SNEG) if k.startswith(prefix + ":")]


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


def listed_dups(recs):
    """comp_listed records that repeat their parent's base/status_swap record (same tid and semantic hash), by set
    size."""
    out = Counter()
    for prs in by_parent(recs).values():
        keys = {(r["tid"], r["semantic_hash"]) for r in prs if r["variant"] in ("base", "status_swap")}
        for r in prs:
            if r["variant"] == "comp_listed" and (r["tid"], r["semantic_hash"]) in keys:
                out[len(r["rule_ast"]["set"])] += 1
    return out


class TestPool(unittest.TestCase):
    def tearDown(self):
        R.unregister_pool("tr")
        R.unregister_pool("tq")

    def test_fake_pool_is_clean_and_complete(self):
        self.assertEqual(R.audit_pool(FAKE_POOL), {})
        self.assertEqual(R.select_pool(FAKE_POOL), (FAKE_POOL, {}))
        self.assertEqual(Counter((t["pol"], t["form"]) for t in FAKE_POOL), {c: 6 for c in R.CELLS})
        self.assertEqual((R.MIN_PER_CELL, len(R.CELLS)), (6, 6))

    def test_register_and_render(self):
        ids = R.register_pool(FAKE_POOL)
        self.assertEqual(ids["f01"], "tr:f01")
        self.assertEqual(T.SPOS["tr:f03"], FAKE_POOL[2]["text"])
        self.assertEqual(T.SNEG["tr:f06"], FAKE_POOL[5]["text"])
        self.assertNotIn("tr:f01", T.SNEG)
        self.assertEqual(T.render_rule({"op": "S+", "set": ["kavo", "booked"]}, "tr:f03"),
                         "Solely kavo or booked events get to move money.")
        R.register_pool(FAKE_POOL)  # identical re-registration is a no-op
        self.assertEqual(R.unregister_pool("tr"), len(FAKE_POOL))
        self.assertFalse(registered("tr"))

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
                dict(good, id="h1"), dict(good, id="h01"), dict(good, id="hx"), dict(good, id="H2"),  # held-out ns
                "Apply {L} events."]
        for bad in bads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    R.register_pool([FAKE_POOL[1], bad])
                self.assertNotIn("tr:f02", T.SPOS)  # all or nothing
        with self.assertRaises(ValueError) as cm:
            R.check_pool([dict(good, id="h1")])
        self.assertIn("starts with 'h'", str(cm.exception))
        R.check_pool([dict(good, id="xh1")])  # 'h' elsewhere is fine

    def test_register_refuses_id_collisions(self):
        with self.assertRaises(ValueError):
            R.register_pool([FAKE_POOL[0], dict(FAKE_POOL[1], id="f01")])  # repeated in the pool
        R.register_pool(FAKE_POOL[:1])
        with self.assertRaises(ValueError):
            R.register_pool([dict(FAKE_POOL[0], text="Only {L} events move money at all.")])  # other text, same id
        with self.assertRaises(ValueError):
            R.register_pool([dict(FAKE_POOL[0], pol="NEG")])  # same id, other polarity
        for prefix in ("p2", "p3", "TR", "t r", "", "h", "hx"):
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

    def test_audit_flags_rcx_refs_status_words_phrases_and_stems(self):
        def tpl(i, text):
            return {"id": i, "text": text, "pol": "POS", "form": "IMPLICIT"}

        bad = R.audit_pool([tpl("a", "Balances change only for {L} events."), tpl("b", "Solely {L} events get paid."),
                            tpl("c", "Solely {L} events get posting rights."),
                            tpl("d", "Events whose status is {L} update the balances; all other events are ignored."),
                            tpl("e", "Its status is in {L}, so apply."),
                            tpl("f", "Solely {L} events move money, never the not posted ones.")])
        self.assertIn("near_dup:SPOS:h1", bad["a"])
        self.assertIn("status_word:paid", bad["b"])
        self.assertTrue(any(x.startswith("status_stem:posting~") for x in bad["c"]))
        self.assertNotIn("c", R.audit_pool([tpl("c", "Solely {L} events get posting rights.")], stems=False))
        self.assertIn("near_dup:SPOS:t3", bad["d"])  # training templates are references too (frozen R1 item 2)
        self.assertIn("near_dup:CLAUSE:S+", bad["e"])  # and the composition clauses
        self.assertIn("status_phrase:not posted", bad["f"])
        self.assertIn("status_word:posted", bad["f"])
        self.assertIn(("not", "posted"), R.status_phrases())
        self.assertIn(("un", "posted"), R.status_phrases())
        with mock.patch.object(V, "EXCLUDED", V.EXCLUDED | {"on hold", "write_off"}):  # a multi-word phrase alone
            bad = R.audit_pool([tpl("g", "Solely {L} events move money; anything on hold waits."),
                                tpl("h", "Solely {L} events move money, never a write off."), FAKE_POOL[2]])
        self.assertEqual(bad, {"g": ["status_phrase:on hold"], "h": ["status_phrase:write off"]})

    def test_runtime_templates_are_not_references(self):
        R.register_pool(FAKE_POOL)
        R.register_pool(FAKE_POOL, prefix="tq")
        self.assertEqual(R.audit_pool(FAKE_POOL), {})
        self.assertFalse([k for k in R.rcx_refs() if k.count(":") > 1])
        self.assertTrue({"SPOS:t0", "SNEG:h1", "AND:t1", "OR:h0", "CLAUSE:S+"} <= set(R.rcx_refs()))

    def test_blind_refs_are_counts_only(self):
        blind = {"b0": FAKE_POOL[2]["text"], "b1": FAKE_POOL[2]["text"].replace("money", "cash"),
                 "b2": "Nothing in this sentence resembles any pool template {L}."}
        bad = R.audit_pool(FAKE_POOL, extra_refs=blind)
        self.assertEqual(bad, {"f03": ["near_dup:blind", "near_dup:blind"]})
        self.assertEqual(R.public_audit(bad, FAKE_POOL), {"by_id": {}, "blind_by_cell": {"POS/IMPLICIT": 1}})
        mixed = R.audit_pool(FAKE_POOL + [dict(FAKE_POOL[2], id="z9", text=FAKE_POOL[2]["text"] + " Paid.")],
                             extra_refs=blind)
        pub = R.public_audit(mixed, FAKE_POOL + [dict(FAKE_POOL[2], id="z9")])
        self.assertEqual(list(pub["by_id"]), ["z9"])
        self.assertIn("status_word:paid", pub["by_id"]["z9"])
        self.assertNotIn("near_dup:blind", pub["by_id"]["z9"])
        self.assertEqual(mixed["z9"].count("near_dup:blind"), 2)
        self.assertEqual(pub["blind_by_cell"], {"POS/IMPLICIT": 2})
        with self.assertRaises(ValueError) as cm:
            R.make_r1(50, 1, FAKE_POOL, extra_refs=blind)
        msg = str(cm.exception)
        self.assertIn("'POS/IMPLICIT': 1", msg)
        for leak in ("f03", "b0", "b1", "near_dup:blind'", FAKE_POOL[2]["text"]):
            self.assertNotIn(leak, msg)
        self.assertFalse(registered("tr"))
        for key in ("P3_SEALED_v2:x1", "p3:x1", "SEALED-3", "pool.secret:1", "my_p3_ref", "Sealed"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as cm:
                    R.audit_pool(FAKE_POOL, extra_refs={key: "Fake {L} text."})
                self.assertNotIn(key, str(cm.exception))
                with self.assertRaises(ValueError):
                    R.make_r1(50, 1, FAKE_POOL, extra_refs={key: "Fake {L} text."})
                with self.assertRaises(ValueError):
                    R.select_pool(FAKE_POOL, extra_refs={key: "Fake {L} text."})

    def test_p2_refs_refuse_sealed_files(self):
        with tempfile.TemporaryDirectory() as d:
            body = json.dumps({"templates": [{"id": "a", "text": "Fake {L} wording."}]})
            for name in ("P3_SEALED_v2.json.secret", "P3_x.json", "SEALED.json", "pool.json.secret", "p3.json",
                         "P2_DEV_v2.secret"):
                p = Path(d) / name
                p.write_text(body, encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        R.p2_refs(R.P2_PATHS + (p,))
            ok = Path(d) / "P2_fake.json"
            ok.write_text(body, encoding="utf-8")
            refs = R.p2_refs(R.P2_PATHS + (ok,))
        self.assertEqual(refs["P2_fake:a"], "Fake {L} wording.")
        self.assertTrue(any(k.startswith("P2_DEV_v2:") for k in refs))

    def test_make_r1_refuses_p2_copies(self):
        p2 = json.loads(P2_V2.read_text(encoding="utf-8"))["templates"]
        for t in (p2[0], p2[-1]):
            with self.subTest(p2=t["id"]):
                with self.assertRaises(ValueError) as cm:
                    R.make_r1(50, 1, FAKE_POOL + [dict(t, id="cp1")])
                self.assertIn(f"near_dup:P2_DEV_v2:{t['id']}", str(cm.exception))
                self.assertFalse(registered("tr"))
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, [])
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, FAKE_POOL, p_pool=1.5)

    def test_make_r1_requires_six_per_cell(self):
        for pol, form in R.CELLS:
            with self.subTest(cell=(pol, form)):
                drop = next(t for t in FAKE_POOL if (t["pol"], t["form"]) == (pol, form))
                with self.assertRaises(ValueError) as cm:
                    R.make_r1(50, 1, [t for t in FAKE_POOL if t is not drop])
                self.assertIn(f"'{pol}/{form}': 5", str(cm.exception))
                self.assertFalse(registered("tr"))
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, FAKE_POOL[:6])  # one template per cell
        with self.assertRaises(ValueError):
            R.make_r1(50, 1, [t for t in FAKE_POOL if t["pol"] == "POS"])  # one polarity only
        self.assertEqual(len(R.make_r1(50, 1, FAKE_POOL)), 50)

    def test_make_r1_refuses_clusters_and_select_pool_keeps_one(self):
        self.assertEqual(R.audit_pool([NEAR_F07]), {})
        with self.assertRaises(ValueError) as cm:
            R.make_r1(50, 1, FAKE_POOL + [NEAR_F07])
        self.assertIn("['f07', 'g07']", str(cm.exception))
        self.assertFalse(registered("tr"))
        kept, dropped = R.select_pool(FAKE_POOL + [NEAR_F07])
        self.assertEqual((kept, dropped), (FAKE_POOL, {"g07": ["pool_near_dup:f07"]}))
        kept, dropped = R.select_pool(FAKE_POOL + [NEAR_F07], pre_drop={"f07": ["near_dup:blind"]})
        self.assertEqual([t["id"] for t in kept], [t["id"] for t in FAKE_POOL if t["id"] != "f07"] + ["g07"])
        self.assertEqual(dropped, {"f07": ["near_dup:blind"]})  # drops come before clustering
        p2 = json.loads(P2_V2.read_text(encoding="utf-8"))["templates"][0]
        kept, dropped = R.select_pool(FAKE_POOL + [dict(p2, id="cp1"), dict(p2, id="cp2")])
        self.assertEqual(kept, FAKE_POOL)
        self.assertTrue(all(any(w.startswith("near_dup:P2_DEV_v2:") for w in dropped[i]) for i in ("cp1", "cp2")))
        self.assertEqual(len(R.make_r1(50, 1, kept)), 50)


class TestR1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = R.make_r1(train_pool=FAKE_POOL)  # the defaults: 1,501 rows, seed 20261253

    @classmethod
    def tearDownClass(cls):
        R.unregister_pool("tr")

    def test_default_seed_is_n_cfsd_data_seed(self):
        self.assertEqual(R.SEEDS["R1"], N_CFSD_DATA_SEED)
        sig = inspect.signature(R.make_r1).parameters
        self.assertEqual((sig["n_rows"].default, sig["seed"].default, sig["p_pool"].default),
                         (1501, N_CFSD_DATA_SEED, 0.5))
        self.assertEqual(R.SEEDS["R2"], 20261264)

    def test_rows_and_determinism(self):
        self.assertEqual(len(self.recs), R.N_ROWS)
        for n in (1, 5, 77):
            self.assertEqual(len(R.make_r1(n, 3, FAKE_POOL)), n)
        self.assertEqual(dumps(R.make_r1(R.N_ROWS, N_CFSD_DATA_SEED, FAKE_POOL)), dumps(self.recs))
        self.assertNotEqual(dumps(R.make_r1(200, 4, FAKE_POOL)), dumps(R.make_r1(200, 5, FAKE_POOL)))
        self.assertEqual(dumps(R.make_r1(200, 4, FAKE_POOL)), dumps(R.make_r1(200, 4, FAKE_POOL[::-1])))

    def test_schema_and_gold(self):
        check_schema(self, self.recs)
        check_gold(self, self.recs)

    def test_same_as_n_cfsd_records_except_wording(self):
        gen.use_world("arith")
        cf = gen.make_train("CF_SD", 1501, N_CFSD_DATA_SEED)
        self.assertEqual(len(cf), len(self.recs))
        changed = 0
        for a, b in zip(self.recs, cf):
            self.assertEqual({k: v for k, v in a.items() if k not in ("tid", "input", "group", "parent_id")},
                             {k: v for k, v in b.items() if k not in ("tid", "input", "group", "parent_id")})
            self.assertEqual((a["group"], a["parent_id"]), ("R1", b["parent_id"].replace("tr-CF_SD-", "tr-R1-")))
            if a["tid"] == b["tid"]:
                self.assertEqual(dict(a, group=b["group"], parent_id=b["parent_id"]), b)  # non-pooled: identical
            else:
                changed += 1
                self.assertTrue(a["tid"].startswith("tr:") and b["tid"] in T.TRAIN_IDS)
        self.assertGreater(changed, 0)
        p0 = R.make_r1(200, 9, FAKE_POOL, p_pool=0.0)
        self.assertEqual([r["input"] for r in p0], [r["input"] for r in gen.make_train("CF_SD", 200, 9)])

    @unittest.skipUnless(N_CFSD_ROWS.exists(), "N-CFSD training rows not present (git-ignored artifact)")
    def test_native_rows_equal_n_cfsd_file(self):
        """Non-pooled R1 prompts and all targets equal train_N-CFSD.jsonl byte for byte (frozen R1 required test)."""
        raw = N_CFSD_ROWS.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), N_CFSD_ROWS_SHA)
        ref = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        gen.use_world("arith")
        cf = gen.make_train("CF_SD", 1501, N_CFSD_DATA_SEED)
        rows = R.native_rows(self.recs, FakeTok(), "CF_SD", np_mod=_native_probe_funcs())
        self.assertEqual(len(rows), len(ref))
        same = 0
        for row, b, r, c in zip(rows, ref, self.recs, cf):
            self.assertEqual(row["target"], b["target"])
            for k in ("regime", "variant", "decision", "semantic_hash", "skeleton_hash", "status"):
                self.assertEqual(row[k], b[k], k)
            self.assertEqual(row["parent_id"], b["parent_id"].replace("tr-CF_SD-", "tr-R1-"))
            if r["tid"].startswith("tr:"):  # only the rule sentence differs
                self.assertNotEqual(row["prompt"], b["prompt"])
                self.assertEqual(row["prompt"].replace(T.render_rule(r["rule_ast"], r["tid"]),
                                                       T.render_rule(c["rule_ast"], c["tid"]), 1), b["prompt"])
            else:
                same += 1
                self.assertEqual(row["prompt"], b["prompt"])
        self.assertGreater(same, 900)  # 1,501 rows, about a third pooled (p_pool 0.5 on single-atom rows)

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
        forms = Counter((texts[r["tid"]]["pol"], texts[r["tid"]]["form"]) for r in pooled)
        for pol in R.POL_OP:  # form uniform within a polarity
            n = sum(forms[(pol, f)] for f in R.FORMS)
            self.assertTrue(all(abs(forms[(pol, f)] / n - 1 / 3) < 0.08 for f in R.FORMS), forms)
        self.assertTrue({r["tid"] for r in single if not r["tid"].startswith("tr:")} <= set(T.TRAIN_IDS))


class TestR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        cls.parents = by_parent(cls.recs)

    def test_row_count_exact(self):
        self.assertEqual(R.N_ROWS, _native_e01_facts()["N_ROWS"])
        self.assertEqual(len(self.recs), R.N_ROWS)
        self.assertEqual(inspect.signature(R.make_r2).parameters["seed"].default, 20261264)
        for n in (0, 1, 2, 9, 10, 11, 57, 400):
            self.assertEqual(len(R.make_r2(n, 7)), n)

    def test_determinism(self):
        self.assertEqual(dumps(R.make_r2()), dumps(self.recs))
        self.assertNotEqual(dumps(R.make_r2(300, 11)), dumps(R.make_r2(300, 12)))

    def test_schema_and_gold(self):
        check_schema(self, self.recs)
        check_gold(self, self.recs)

    def test_complement_twins_gold_mirrored(self):
        want = {("S+", "comp_unlisted"): ("noop", False), ("S+", "comp_listed"): ("apply", True),
                ("S-", "comp_unlisted"): ("apply", False), ("S-", "comp_listed"): ("noop", True)}
        n = Counter()
        for r in self.recs:
            if r["variant"].startswith("comp_"):
                op = r["rule_ast"]["op"]
                ok, after = W.transition(r["rule_ast"], r["state"], r["event"])  # gold from rcx.rcl.world
                self.assertEqual((("apply" if ok else "noop"), W.listed(r["rule_ast"], r["event"])),
                                 want[(op, r["variant"])])
                self.assertEqual((r["gold"]["decision"], r["gold"]["listed"]), want[(op, r["variant"])])
                self.assertEqual(r["gold"]["after"], after)
                self.assertEqual(r["event"]["status"] in r["rule_ast"]["set"], r["variant"] == "comp_listed")
                n[(op, r["variant"])] += 1
        self.assertEqual(set(n), set(want))
        for op in ("S+", "S-"):
            self.assertEqual(n[(op, "comp_unlisted")], n[(op, "comp_listed")])
            self.assertGreater(n[(op, "comp_unlisted")], 100)
        unl = [r for r in self.recs if r["variant"] == "comp_unlisted"]  # "unlisted" alone does not predict
        self.assertTrue(0.35 < sum(r["gold"]["decision"] == "noop" for r in unl) / len(unl) < 0.65)

    def test_balance_wordings_and_pairs(self):
        n_comp = Counter()
        for pid, recs in self.parents.items():
            base = [r for r in recs if r["variant"] == "base"]
            comp = [r for r in recs if r["variant"].startswith("comp_")]
            if not base:
                continue
            if base[0]["rule_ast"]["op"] not in ("S+", "S-"):
                self.assertEqual(comp, [], pid)
                continue
            n_comp[base[0]["rule_ast"]["op"]] += 1
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
                self.assertEqual(list(u["event"]), list(li["event"]))
                self.assertNotEqual(u["gold"]["decision"], li["gold"]["decision"])
                self.assertEqual(u["skeleton_hash"], base[0]["skeleton_hash"])
        self.assertGreater(n_comp["S+"], 40)
        self.assertGreater(n_comp["S-"], 40)

    def test_no_duplicated_size1_listed_twin(self):
        for recs in (self.recs, R.make_r2(1501, 7), R.make_r2(1501, 8)):
            self.assertEqual(listed_dups(recs)[1], 0)
        size1 = [prs for prs in self.parents.values() if prs[0]["variant"] == "base"
                 and prs[0]["rule_ast"]["op"] in ("S+", "S-") and len(prs[0]["rule_ast"]["set"]) == 1]
        self.assertGreater(len(size1), 30)
        # the check is load-bearing: without it the same generator repeats size-1 listed records
        with mock.patch.object(R, "_dup_key", lambda rec: object()):
            raw = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        self.assertGreater(listed_dups(raw)[1], 5)
        # with every training wording taken (k = 4), a duplicate pair cannot be redrawn and is dropped
        with mock.patch.object(R, "COMP_K", (4,)):
            full = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        self.assertEqual((len(full), listed_dups(full)[1]), (R.N_ROWS, 0))
        npairs = Counter()
        for prs in by_parent(full).values():
            if prs[0]["variant"] == "base" and prs[0]["rule_ast"]["op"] in ("S+", "S-"):
                k = sum(r["variant"] == "comp_listed" for r in prs)
                self.assertEqual(k, sum(r["variant"] == "comp_unlisted" for r in prs))
                npairs[(len(prs[0]["rule_ast"]["set"]), k)] += 1
        self.assertTrue(npairs[(1, 3)] > 0 and npairs[(1, 4)] > 0, npairs)
        self.assertEqual({k for s, k in npairs if s == 2}, {4})

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
        cf = R.recipe_stats(gen.make_train("CF_SD", R.N_ROWS, N_CFSD_DATA_SEED))
        self.assertGreater(s["spos_unlisted_share"], cf["spos_unlisted_share"] + 0.05)
        self.assertTrue(0.4 < s["noop_share"] < 0.6)
        self.assertEqual(s["spos_unlisted_rows"], sum(r["rule_ast"]["op"] == "S+" and r["gold"]["decision"] == "noop"
                                                      for r in r2))
        s1 = R.recipe_stats(R.make_r1(300, 2, FAKE_POOL))
        self.assertGreater(s1["tid_family"]["tr"], 0)
        self.assertEqual(sum(s1["tids"].values()), 300)

    def test_rule_type_stats(self):
        r2 = R.make_r2(R.N_ROWS, R.SEEDS["R2"])
        gen.use_world("arith")
        cf_recs = gen.make_train("CF_SD", R.N_ROWS, N_CFSD_DATA_SEED)
        for recs in (r2, cf_recs):
            s = R.recipe_stats(recs)
            self.assertTrue(set(s["rule_type_rows"]) <= {"S+", "S-", "composition"})
            self.assertEqual(sum(s["rule_type_rows"].values()), s["rows"])
            self.assertEqual(sum(s["rule_type_parents"].values()), s["parents"])
            self.assertEqual(s["rule_type_rows"], dict(Counter(
                r["rule_ast"]["op"] if r["rule_ast"]["op"] in ("S+", "S-") else "composition" for r in recs)))
        s, cf = R.recipe_stats(r2), R.recipe_stats(cf_recs)
        self.assertEqual(cf["parents"], 376)  # N-CFSD (PREREG section 3)
        self.assertLess(s["parents"], cf["parents"])
        self.assertEqual(set(s["comp_rows"]), {"S+|comp_unlisted|noop", "S+|comp_listed|apply",
                                               "S-|comp_unlisted|apply", "S-|comp_listed|noop"})
        self.assertEqual(s["comp_rows"]["S+|comp_unlisted|noop"], s["comp_rows"]["S+|comp_listed|apply"])
        self.assertEqual(s["comp_rows"]["S-|comp_unlisted|apply"], s["comp_rows"]["S-|comp_listed|noop"])
        self.assertEqual((cf["comp_rows"], cf["dup_rows_within_parent"]), ({}, 0))
        self.assertEqual(s["dup_rows_within_parent"], sum(listed_dups(r2).values()))

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
