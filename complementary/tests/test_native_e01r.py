"""Tests for experiments/RCX-E01R/native_e01r.py (E01R rows, train call, smoke, SEALED mint/open/eval custody). Pure
Python: no torch, no tokenizer (FakeTok), no model. P3, the salt, POOL_REFINE, the E01N control and every adapter are
fakes in a temp dir; every model command's heavy import (_heavy) is replaced by a sentinel, or (smoke and eval end to
end) by FakeLC, whose scripted generate follows lora_cpu.generate's decode order, with a stand-in `peft` module. The
lane lock lives in the temp dir (RINEYA_LANE_DIR). A read of a *.secret file outside the temp dir fails the test before
a byte is read. Reads (read-only): pools/P2_DEV*.json and POOL_REFINE.json, runs/RCX-E01N/cells/{train_N-CFSD.jsonl,
SEALED_OPEN.json, N-*/TRAIN_META.json} when present, and source files.

Run: python -B -m unittest tests.test_native_e01r -v   (CPU, single thread; about 15 s)
"""
import ast
import contextlib
import gzip
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import types
import unittest
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01R"))
HEAVY = ("torch", "transformers", "peft", "native_probe", "native_e01", "rcx.native.lora_cpu")
_HEAVY_BEFORE = {m for m in HEAVY if m in sys.modules}

import native_e01r as M  # noqa: E402
import recipes as RC  # noqa: E402
import trace_target as TT  # noqa: E402
from rcx import lane  # noqa: E402
from rcx.rcl import gen, score as SC, templates as T, world as W  # noqa: E402

_HEAVY_AFTER = {m for m in HEAVY if m in sys.modules}
REAL_SKELETON = M.build_skeleton
_REAL_OPEN = io.open
HERE = ROOT / "experiments" / "RCX-E01R"
E01N_SRC = ROOT / "experiments" / "RCX-E01N"
REAL_E01N = ROOT / "runs" / "RCX-E01N" / "cells"
REAL_NCFSD = REAL_E01N / "train_N-CFSD.jsonl"
COUNTS = {"POS/EXPLICIT_LISTED_FIRST": 7, "POS/EXPLICIT_COMPLEMENT_FIRST": 10, "POS/IMPLICIT": 6,
          "NEG/EXPLICIT_LISTED_FIRST": 7, "NEG/EXPLICIT_COMPLEMENT_FIRST": 9, "NEG/IMPLICIT": 9}  # POOL_REFINE P3
SALT = "e01r-skeleton-test-salt"
TAGS_R3 = "N-CFSD_s0,N-CFSD_s1,N-CFSD_s2,E-R3_s0,E-R3_s1,E-R3_s2"
TAGS_B4 = "N-CFSD_s0,N-CFSD_s1,N-CFSD_s2,E-R2_s0,E-R2_s1,E-R2_s2,E-R3_s0,E-R3_s1,E-R3_s2"  # CX D-39: branch B4
TREAT_B4 = [t for t in TAGS_B4.split(",") if t.startswith("E-")]
EOS = [151645, 151643]


class FakeLC:
    """Stand-in for rcx.native.lora_cpu in the model commands: load_core returns a FakeTok and a marker model, and
    generate follows lora_cpu.generate's decode order (prompts stably sorted by token length; one tok.decode per row,
    called on the tokenizer proxy it is given) with outputs scripted by script(j, prompt) -> (text, has_eos)."""
    EOS_IDS = EOS

    def __init__(self, script, paths=None):
        self.script, self.max_new_seen, self.loads, self.PATHS = script, [], 0, paths or {}

    def load_core(self, name):
        assert name == "sparse"
        self.loads += 1
        return "fake-model", FakeTok()

    def generate(self, model, tok, prompts, max_new=64, max_batch_tokens=3000, guard=None, log=None):
        self.max_new_seen.append(max_new)
        if guard is not None:
            guard.check()
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        outs = [None] * len(enc)
        for j in order:
            text, eos = self.script(j, prompts[j])
            outs[j] = tok.decode(FakeTok()(text)["input_ids"] + (EOS if eos else []), skip_special_tokens=True)
        return outs


class FakeGuard:
    def __init__(self, log=None):
        pass

    def check(self):
        pass


def fake_peft(loaded):
    """A stand-in `peft` module (sys.modules) whose PeftModel.from_pretrained records the adapter directory."""
    mod = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, is_trainable=True):
            assert is_trainable is False
            loaded.append(Path(path))
            return SimpleNamespace(eval=lambda: ("adapted", model))

    mod.PeftModel = PeftModel
    return mod


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def write(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return M.sha(p)


class Enc(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class FakeTok:
    """Qwen3 chat-template text (native_e01.render_literal, which E01N's audit matched byte for byte to the real
    template); one token id per 3 UTF-8 bytes (Qwen averages 3-4), above every special id. Nothing is loaded."""
    eos_token_id, pad_token_id, BASE = 151645, 151643, 1 << 26

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert (tokenize, add_generation_prompt, enable_thinking) == (False, True, False)
        assert [m["role"] for m in msgs] == ["system", "user"]
        return M.NE.render_literal(msgs[0]["content"], msgs[1]["content"])

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        b = text.encode("utf-8")
        return Enc(input_ids=[self.BASE + (int.from_bytes(b[i:i + 3], "big") << 2) + len(b[i:i + 3])
                              for i in range(0, len(b), 3)])

    def decode(self, ids, skip_special_tokens=True):
        v = [i - self.BASE for i in ids if i >= self.BASE]
        return b"".join((x >> 2).to_bytes(x & 3, "big") for x in v).decode("utf-8")


class SecretGuard:
    """Patches builtins.open and io.open: reading a *.secret / P3_SEALED file outside `allowed` raises before anything
    is read; every opened path is recorded."""

    def __init__(self, allowed):
        self.allowed, self.opened = Path(allowed).resolve(), []

    def _open(self, file, mode="r", *args, **kw):
        if isinstance(file, (str, os.PathLike)):
            p = Path(os.fspath(file)).resolve()
            sealed = p.name.endswith(".secret") or "P3_SEALED" in p.name
            if sealed and ("r" in mode or "+" in mode) and self.allowed not in p.parents:
                raise AssertionError(f"test code tried to read a sealed file: {p.name}")
            self.opened.append((p, mode))
        return _REAL_OPEN(file, mode, *args, **kw)

    def start(self, tc):
        for target in ("builtins.open", "io.open"):
            pt = mock.patch(target, self._open)
            pt.start()
            tc.addCleanup(pt.stop)


def fake_r1_pool(per_cell=6):
    """per_cell nonsense wordings per cell: x-initial words, so no P2/h* near-duplicate, status word or stem."""
    pool, k = [], 0
    for pol in M.POLS:
        for form in M.FORMS:
            for _ in range(per_cell):
                w = [f"x{chr(97 + (k * 5 + j) // 26)}{chr(97 + (k * 5 + j) % 26)}j" for j in range(5)]
                pool.append({"id": f"x{k:02d}", "text": f"{w[0]} {w[1]} {{L}} {w[2]} {w[3]} {w[4]}.", "pol": pol,
                             "form": form})
                k += 1
    return pool


def fake_p3_templates():
    out, k = [], 0
    for cell, n in COUNTS.items():
        pol, form = cell.split("/")
        for i in range(n):
            out.append({"id": f"q{k:03d}", "text": f"Fake {pol} {form.lower()} rule {i}: {{L}} decides it.",
                        "pol": pol, "form": form, "author": "test"})
            k += 1
    random.Random(7).shuffle(out)  # file order must not matter (cells are id-sorted)
    return out


def by_cell_of(templates):
    by = defaultdict(list)
    for t in templates:
        by[f"{t['pol']}/{t['form']}"].append((t["id"], t["text"]))
    return {c: sorted(by[c]) for c in M.CELLS}


def write_branch(path, branch, code_sha=None):
    arms = {"B0": [], "B1": [], "B2": [], "B3": ["R1", "R3"], "B4": ["R2", "R3"]}[branch]
    return write(path, json.dumps({"branch": branch, "arms": arms, "code_sha256": code_sha or M.DECIDE_SHA256}))


def fn_node(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(path, func, attr):
    return [([ast.unparse(x) for x in c.args], {k.arg: ast.unparse(k.value) for k in c.keywords})
            for c in ast.walk(fn_node(path, func)) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == attr]


def assigns(path, func):
    return {ast.unparse(n.targets[0]).strip("()"): ast.unparse(n.value) for n in fn_node(path, func).body
            if isinstance(n, ast.Assign) and len(n.targets) == 1}


class Base(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)  # native_e01.load_rows (borrowed verbatim) leaves a file open
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.guard = SecretGuard(self.tmp)
        self.guard.start(self)
        for name, val in (("P3_PATH", self.tmp / "pools" / "P3_SEALED_v2.json.secret"),
                          ("POOL_REFINE_PATH", self.tmp / "pools" / "POOL_REFINE.json"), ("OUT", self.tmp / "cells"),
                          ("BRANCH_PATH", self.tmp / "BRANCH_STEP_A.json"), ("R1_POOL_PATH", self.tmp / "TR.json"),
                          ("ANALYZE_PATH", self.tmp / "analyze_e01r.py"), ("E01N_CELLS", self.tmp / "e01n")):
            pt = mock.patch.object(M, name, val)
            pt.start()
            self.addCleanup(pt.stop)
        (self.tmp / "lane").mkdir()
        pt = mock.patch.dict(os.environ, {"RINEYA_LANE_DIR": str(self.tmp / "lane")})
        pt.start()
        self.addCleanup(pt.stop)
        os.environ.pop(M.LANE_TOKEN_ENV, None)  # restored by patch.dict
        self.addCleanup(RC.unregister_pool, "p3")
        self.addCleanup(RC.unregister_pool, "tr")

    def hold_lane(self, party="rcx", pid=None, export=True):
        """Take the (temp-dir) lane lock as a chain would, and export its token unless export is False."""
        rec = lane.try_acquire("E01R test", os.getpid() if pid is None else pid, 1, party=party)
        self.assertIsNotNone(rec)
        if export:
            os.environ[M.LANE_TOKEN_ENV] = rec["token"]
        return rec


# ----------------------------------------------------------------------------- module, borrowed code, train call

class TestModule(Base):
    def test_no_heavy_imports(self):
        self.assertEqual(_HEAVY_AFTER, _HEAVY_BEFORE)
        tree = ast.parse((HERE / "native_e01r.py").read_text(encoding="utf-8"))
        top = set()
        for n in tree.body:
            if isinstance(n, ast.Import):
                top |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                top |= {f"{n.module}.{a.name}" for a in n.names} | {n.module}
        self.assertFalse({t for t in top if t.split(".")[0] in ("torch", "transformers", "peft", "native_probe",
                                                                 "native_e01") or t.endswith("lora_cpu")})
        inner = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)
                 for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom))}
        self.assertEqual(inner, {"_heavy", "cmd_build", "cmd_smoke", "cmd_eval", "cmd_mint_commit"})

    def test_borrowed_values(self):
        self.assertEqual((M.N_ROWS, M.STEPS, M.BATCH, M.CFSD_SEED, M.R2_SEED), (1501, 190, 16, 20261253, 20261264))
        self.assertEqual(M.R2_SEED, RC.SEEDS["R2"])
        self.assertEqual((M.LCP.EOS_IDS, M.NPP.BATTERY_SEED), ([151645, 151643], 20261240))
        self.assertEqual(M.LCP.row_order_for(1501, 190, 16, 1)[:5], M.LCP.row_order_for(1501, 190, 16, 1)[:5])
        sft = fn_node(ROOT / "rcx" / "native" / "lora_cpu.py", "train_sft").args
        kw = {a.arg: d.value for a, d in zip(sft.kwonlyargs, sft.kw_defaults) if isinstance(d, ast.Constant)}
        self.assertEqual(kw["max_len"], M.MAX_LEN)
        self.assertEqual(M.NE.target_json({"B": 2, "A": 1}), '{"A":1,"B":2}')
        self.assertEqual(M.NPP.parse_answer('{"A":1,"B":2}', ["A", "B"]), {"A": 1, "B": 2})
        self.assertIsNone(M.NPP.parse_answer('{"A":1,"A":2,"B":2}', ["A", "B"]))
        with self.assertRaises(ImportError):
            M._borrow(E01N_SRC / "native_e01.py", ("no_such_name",))
        self.assertEqual(M.STRATA, ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike",
                                    "V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V7_bigset"))

    def test_frozen_references(self):
        self.assertEqual(M.sha(HERE / "decide_branch.py"), M.DECIDE_SHA256)
        real = json.loads((HERE / "pools" / "POOL_REFINE.json").read_text(encoding="utf-8"))
        self.assertEqual((real["outputs"]["P3"]["sha256"], real["outputs"]["P3"]["file"]),
                         (M.P3_SHA256, "P3_SEALED_v2.json.secret"))
        ref = M.p3_reference(HERE / "pools" / "POOL_REFINE.json")  # POOL_REFINE only; P3 itself is never opened
        self.assertEqual((ref["counts"], ref["n"]), (COUNTS, 48))
        d30 = {"N-CFSD_s0": "97cc48e8", "N-CFSD_s1": "8e710e46", "N-CFSD_s2": "652b497d", "N-VR_s0": "2f376882",
               "N-VR_s1": "52adbde9", "N-VR_s2": "876ac52d"}  # DECISIONS CX D-30
        self.assertEqual({k: v[:8] for k, v in M.E01N_ADAPTERS.items()}, d30)
        if (REAL_E01N / "SEALED_OPEN.json").exists():
            self.assertEqual(M.sha(REAL_E01N / "SEALED_OPEN.json"), M.E01N_OPEN_SHA256)
            opened = json.loads((REAL_E01N / "SEALED_OPEN.json").read_text(encoding="utf-8"))["adapters_sha256"]
            for tag, want in M.E01N_ADAPTERS.items():
                meta = json.loads((REAL_E01N / tag / "TRAIN_META.json").read_text(encoding="utf-8"))
                self.assertEqual((opened[tag], meta["adapter_sha256"]), (want, want))
                if tag.startswith("N-CFSD"):
                    self.assertEqual(meta["rows_sha256"], M.NCFSD_ROWS_SHA256)
        self.assertFalse([o for o in self.guard.opened if o[0].name.endswith(".secret")])


class TestTrainCall(Base):
    NE01 = E01N_SRC / "native_e01.py"
    ME = HERE / "native_e01r.py"

    def test_train_sft_call_is_native_e01s(self):
        for attr in ("train_sft", "attach_lora", "row_order_for", "load_core", "save_adapter"):
            with self.subTest(attr=attr):
                self.assertEqual(calls(self.ME, "cmd_train", attr), calls(self.NE01, "cmd_train", attr))
                self.assertEqual(len(calls(self.ME, "cmd_train", attr)), 1)
        kw = calls(self.ME, "cmd_train", "train_sft")[0][1]
        self.assertEqual(set(kw), {"steps", "batch_size", "seed", "row_order", "logger", "guard", "ckpt_dir",
                                   "ckpt_every"})  # lr, max_len, micro, grad_clip: train_sft's defaults, as in E01N
        a, b = assigns(self.ME, "cmd_train"), assigns(self.NE01, "cmd_train")
        for name in ("out", "rows_sha", "rows", "model, tok", "base_sha", "pm", "order", "guard"):
            self.assertEqual(a[name], b[name], name)
        self.assertEqual(a["NP, LC, ThermalGuard"], "_heavy()")
        heavy = fn_node(self.ME, "_heavy")
        imps = {(n.module, x.name) for n in ast.walk(heavy) if isinstance(n, ast.ImportFrom) for x in n.names}
        self.assertIn(("rcx.native", "lora_cpu"), imps)
        self.assertEqual(ast.unparse(heavy.body[-1]), "return (native_probe, lora_cpu, ThermalGuard)")
        lit = {ast.unparse(n.targets[0]).strip("()"): ast.unparse(n.value) for n in ast.parse(self.NE01.read_text(
            encoding="utf-8")).body if isinstance(n, ast.Assign) and len(n.targets) == 1}
        self.assertEqual(lit["N_ROWS, STEPS, BATCH"], "(1501, 190, 16)")
        self.assertNotIn("STEPS", a)
        self.assertNotIn("BATCH", a)

    def test_checks_precede_the_model(self):
        fn = fn_node(self.ME, "cmd_train")
        line = {}
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                line.setdefault(c.func.id, c.lineno)
        self.assertLess(line["require_audit"], line["_heavy"])
        self.assertLess(line["require_lane"], line["_heavy"])

    def test_cmd_train_refusals(self):
        cells = self.tmp / "cells"
        rows = write(cells / "train_E-R3.jsonl", '{"prompt": "p", "target": "t"}\n')
        man = write(cells / M.MANIFEST, json.dumps({"arms": {"E-R3": {"sha256": rows}}, "max_new": 176}))
        a = SimpleNamespace(out=str(cells), arm="E-R3", seed=0)
        sentinel = RuntimeError("model would load here")

        def refuse(text):
            with mock.patch.object(M, "_heavy", side_effect=AssertionError("_heavy reached")):
                with self.assertRaises(SystemExit) as cm:
                    M.cmd_train(a)
            self.assertIn(text, str(cm.exception))

        refuse("mint-commit precedes")
        commit = {"arms_required": TAGS_R3.split(","), "train_manifest_sha256": man, "rows_sha256": {"E-R3": rows},
                  "code_sha256": M.code_sha256()}
        write(cells / "SEALED_COMMIT.json", json.dumps(commit))
        for bad in (SimpleNamespace(out=str(cells), arm="E-R1", seed=0),
                    SimpleNamespace(out=str(cells), arm="E-R3", seed=3), SimpleNamespace(out=str(cells), arm="N-CFSD",
                                                                                        seed=0)):
            with self.assertRaises(SystemExit):
                M.cmd_train(bad)
        refuse("AUDIT_E01R.json")
        write(cells / M.AUDIT, json.dumps({"ok": False, "rows_sha256": {"E-R3": rows}}))
        refuse("AUDIT_E01R.json")
        write(cells / M.AUDIT, json.dumps({"ok": False, "arm_ok": {"E-R3": False, "E-R2": True},
                                           "rows_sha256": {"E-R3": rows}}))
        refuse("AUDIT_E01R.json")
        aud = write(cells / M.AUDIT, json.dumps({"ok": True, "rows_sha256": {"E-R3": rows}}))
        refuse("not the row audit bound by SEALED_COMMIT.json")  # the commitment binds no / another audit
        write(cells / "SEALED_COMMIT.json", json.dumps(dict(commit, audit_e01r_sha256="0" * 64)))
        refuse("not the row audit bound by SEALED_COMMIT.json")
        drifted = dict(commit["code_sha256"], **{"stats.py": "0" * 64})
        write(cells / "SEALED_COMMIT.json", json.dumps(dict(commit, audit_e01r_sha256=aud, code_sha256=drifted)))
        refuse("bound code differs from SEALED_COMMIT.json: ['stats.py']")  # bound code is frozen from the mint on
        write(cells / "SEALED_COMMIT.json", json.dumps(dict(commit, audit_e01r_sha256=aud)))
        refuse("lane lock")
        self.hold_lane(party="bwm")
        refuse("lane lock")
        lane.lock_path().unlink()
        self.hold_lane(export=False)
        refuse(M.LANE_TOKEN_ENV)
        lane.lock_path().unlink()
        self.hold_lane()
        with mock.patch.object(M, "_heavy", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as cm:
                M.cmd_train(a)
        self.assertIs(cm.exception, sentinel)  # every check passed; the model load comes next
        base = self.tmp / "base"
        write(base / "model.safetensors", "not the Titan base")
        lc = SimpleNamespace(PATHS={"sparse": base}, load_core=mock.Mock(side_effect=AssertionError("loaded")))
        with mock.patch.object(M, "_heavy", return_value=(SimpleNamespace(need_ram=lambda: None), lc, FakeGuard)):
            with self.assertRaises(SystemExit) as cm:
                M.cmd_train(a)
        self.assertIn("not Titan-Sparse FP32", str(cm.exception))  # checked before the model is loaded
        lc.load_core.assert_not_called()
        write(cells / "SEALED_COMMIT.json", json.dumps(dict(commit, audit_e01r_sha256=aud, rows_sha256={"E-R3": "0"})))
        refuse("differs from")
        write(cells / "SEALED_COMMIT.json", json.dumps(dict(commit, audit_e01r_sha256=aud)))
        write(cells / "train_E-R3.jsonl", '{"prompt": "p", "target": "u"}\n')
        refuse("differs from")
        write(cells / "SEALED_OPEN.json", "{}")
        refuse("no adapter is trained after opening")


# ----------------------------------------------------------------------------- build

class TestBuild(Base):
    def build(self, branch, arms, pool=None, fallback=False):
        write_branch(self.tmp / "BRANCH_STEP_A.json", branch)
        if pool is not None:
            write(self.tmp / "TR.json", json.dumps({"templates": pool}))
        ref = REAL_NCFSD if REAL_NCFSD.exists() else None
        with contextlib.redirect_stdout(io.StringIO()):
            return M.build_cells(self.tmp / "cells", arms, FakeTok(), ncfsd_rows_path=ref, strict=False,
                                 r1_fallback=fallback)

    def rows(self, arm):
        return [json.loads(x) for x in (self.tmp / "cells" / f"train_{arm}.jsonl").read_text(
            encoding="utf-8").splitlines()]

    def check_common(self, man, arms):
        self.assertEqual(sorted(man["arms"]), sorted(arms))
        for arm in arms:
            m = man["arms"][arm]
            self.assertEqual((m["rows"], m["sha256"]), (1501, M.sha(self.tmp / "cells" / f"train_{arm}.jsonl")))
            self.assertLessEqual(m["tokens"]["max_processed_per_row"], M.MAX_LEN)
        longest = max(len(FakeTok()(t)["input_ids"]) for v in M.r3_gold_targets().values() for t in v)
        self.assertEqual(man["max_new"], max(160, longest + 16))
        self.assertEqual(max(man["r3_gold_longest_tokens"].values()), longest)
        self.assertEqual(json.loads((self.tmp / "cells" / M.MANIFEST).read_text(encoding="utf-8")), man)
        with self.assertRaises(SystemExit):  # built once
            self.build(man["branch"]["branch"], arms)

    def test_b3_r1_r3(self):
        pool = fake_r1_pool()
        man = self.build("B3", ["E-R1", "E-R3"], pool)
        self.check_common(man, ["E-R1", "E-R3"])
        r3, r1 = self.rows("E-R3"), self.rows("E-R1")
        for a in r3:  # structure only; byte identity with N-CFSD's file: test_r1_r3_byte_identity_with_real_ncfsd
            line1, line2 = a["target"].split("\n")
            self.assertEqual(TT.parse_check(line1)["decision"], a["decision"])
            self.assertEqual(line2, M.NE.target_json(json.loads(line2)))
        self.assertEqual(man["arms"]["E-R3"]["vs_N-CFSD"]["prompts_identical"], 1501)
        words = {w for t in pool for w in t["text"].split()}
        pooled = [a for a in r1 if set(M.NE._WORD_RE.findall(a["prompt"])) & words]
        self.assertEqual(len(pooled), man["arms"]["E-R1"]["vs_N-CFSD"]["prompts_changed"])
        self.assertGreater(len(pooled), 100)
        self.assertEqual(man["r1_pool"]["counts"], {c: 6 for c in M.CELLS})
        recs = M.treatment_records("E-R1", pool)  # review item 12: N-CFSD's records apart from the wording
        gen.use_world("arith")
        for a, b in zip(recs, gen.make_train("CF_SD", 1501, 20261253)):
            self.assertEqual({k: v for k, v in a.items() if k not in ("tid", "input", "parent_id", "group")},
                             {k: v for k, v in b.items() if k not in ("tid", "input", "parent_id", "group")})

    @unittest.skipUnless(REAL_NCFSD.exists(), "runs/RCX-E01N/cells/train_N-CFSD.jsonl absent: byte identity with "
                                              "N-CFSD's CX D-30 rows cannot be tested (no rebuilt stand-in)")
    def test_r1_r3_byte_identity_with_real_ncfsd(self):
        """Against N-CFSD's real row file (sha256 de23a923..., CX D-30) only; never against rows rebuilt by the same
        code, which would make the comparison tautological."""
        self.assertEqual(M.sha(REAL_NCFSD), M.NCFSD_ROWS_SHA256)
        pool = fake_r1_pool()
        man = self.build("B3", ["E-R1", "E-R3"], pool)
        self.assertEqual(man["reference_rows"], {"source": "train_N-CFSD.jsonl", "sha256": M.NCFSD_ROWS_SHA256})
        ref = [json.loads(x) for x in REAL_NCFSD.read_text(encoding="utf-8").splitlines()]
        r3, r1 = self.rows("E-R3"), self.rows("E-R1")
        self.assertEqual([r["prompt"] for r in r3], [r["prompt"] for r in ref])  # byte-identical inputs
        for a, b in zip(r3, ref):
            line1, line2 = a["target"].split("\n")
            self.assertEqual(line2, b["target"])
            self.assertEqual(TT.parse_check(line1)["decision"], b["decision"])
        self.assertEqual([r["target"] for r in r1], [r["target"] for r in ref])
        changed = [a for a, b in zip(r1, ref) if a["prompt"] != b["prompt"]]
        self.assertEqual(len(changed), man["arms"]["E-R1"]["vs_N-CFSD"]["prompts_changed"])
        self.assertGreater(len(changed), 100)
        words = {w for t in pool for w in t["text"].split()}
        self.assertTrue(all(set(M.NE._WORD_RE.findall(a["prompt"])) & words for a in changed))

    def test_b4_r2_r3_and_b1(self):
        man = self.build("B4", ["E-R2", "E-R3"])
        self.check_common(man, ["E-R2", "E-R3"])
        r2 = man["arms"]["E-R2"]
        self.assertIsNone(r2["vs_N-CFSD"])
        self.assertEqual((r2["data_seed"], sum(r2["rows_by_rule_type"].values())), (20261264, 1501))
        self.assertEqual(set(r2["rows_by_rule_type"]) - {"S+", "S-", "composition"}, set())
        (self.tmp / "cells" / M.MANIFEST).unlink()
        for p in (self.tmp / "cells").glob("train_*.jsonl"):
            p.unlink()
        man = self.build("B1", [])
        self.assertEqual((man["arms"], man["r1_pool"]), ({}, None))
        self.assertGreaterEqual(man["max_new"], 160)

    def test_branch_gates_and_fallback(self):
        for branch in ("B0", "B2"):
            with self.subTest(branch=branch), self.assertRaises(SystemExit):
                self.build(branch, [])
        for branch, arms in (("B3", ["E-R2", "E-R3"]), ("B3", ["E-R1"]), ("B4", ["E-R3", "E-R2"]), ("B1", ["E-R3"])):
            with self.subTest(branch=branch, arms=arms), self.assertRaises(SystemExit):
                self.build(branch, arms, fake_r1_pool())
        for branch, arms, fb in (("B4", ["E-R1", "E-R3"], False), ("B4", ["E-R2", "E-R3"], True),
                                 ("B1", ["E-R1"], False), ("B1", [], True)):  # the R1 path exists only under B3
            with self.subTest(branch=branch, arms=arms, fallback=fb), self.assertRaises(SystemExit) as cm, \
                    mock.patch.object(M, "load_r1_pool", side_effect=AssertionError("R1 pool read")):
                self.build(branch, arms, fake_r1_pool(5), fallback=fb)
            self.assertIn("only under branch B3", str(cm.exception))
            self.assertFalse((self.tmp / "cells" / M.MANIFEST).exists())
        with mock.patch.object(M, "load_r1_pool", side_effect=AssertionError("R1 pool read")):
            self.assertEqual(self.build("B4", ["E-R2", "E-R3"])["r1_pool"], None)  # B4 never reads the R1 pool
        for p in [self.tmp / "cells" / M.MANIFEST] + list((self.tmp / "cells").glob("train_*.jsonl")):
            p.unlink()
        write_branch(self.tmp / "BRANCH_STEP_A.json", "B3", code_sha="0" * 64)
        with self.assertRaises(SystemExit):
            M.branch_arms(self.tmp / "BRANCH_STEP_A.json")
        with self.assertRaises(SystemExit):  # the pool meets its minimum: no fallback
            self.build("B3", ["E-R2", "E-R3"], fake_r1_pool(), fallback=True)
        with self.assertRaises(SystemExit) as cm:  # 5 per cell: R1 refused
            self.build("B3", ["E-R1", "E-R3"], fake_r1_pool(5))
        self.assertIn("below 6 per cell", str(cm.exception))
        self.assertFalse((self.tmp / "cells" / M.MANIFEST).exists())
        man = self.build("B3", ["E-R2", "E-R3"], fake_r1_pool(5), fallback=True)
        self.assertEqual((man["fallback"]["from"], man["fallback"]["to"]), ("E-R1", "E-R2"))
        self.assertEqual(man["fallback"]["shortfall"], {c: 5 for c in M.CELLS})
        with self.assertRaises(SystemExit):
            M.load_r1_pool(self.tmp / "x.json.secret")


# ----------------------------------------------------------------------------- SEALED skeleton and draw

class TestSkeleton(Base):
    def test_pinned_draw_sequence(self):
        seed = M.NE.sealed_seed(SALT)
        self.assertEqual(seed, 38007660)
        pins = {"sealed-F1_void-0": ("EXPLICIT_LISTED_FIRST", 0, 5),
                "sealed-F1_void-1": ("EXPLICIT_COMPLEMENT_FIRST", 3, 4), "sealed-F1_void-2": ("IMPLICIT", 4, 3),
                "sealed-V3_inverted-5": ("IMPLICIT", 5, 6),
                "sealed-V5_posted_override-39": ("EXPLICIT_LISTED_FIRST", 0, 2),
                "sealed-V2_neg-17": ("IMPLICIT", 0, 1), "sealed-V7_bigset-3": ("EXPLICIT_LISTED_FIRST", 0, 5)}
        for pid, (form, jp, jn) in pins.items():
            self.assertEqual(M.p3_draw(seed, pid, form, COUNTS), {"POS": jp, "NEG": jn}, pid)
            rng = random.Random(f"{seed}-{pid}")  # the documented sequence, spelled out
            self.assertEqual((rng.randrange(COUNTS[f"POS/{form}"]), rng.randrange(COUNTS[f"NEG/{form}"])), (jp, jn))
        sk = M.build_skeleton(seed, COUNTS)
        forms = {r["parent_id"]: r["p3_form"] for r in sk if r["surface"] == "p3"}
        for pid, (form, _, _) in pins.items():
            self.assertEqual(forms[pid], form)
        table = json.dumps([[pid, M.p3_draw(seed, pid, f, COUNTS)] for pid, f in forms.items()], separators=(",", ":"))
        self.assertEqual(sha_b(table.encode()), "05302c0f91b1339f096f2a6b0ff69d9c1c4cc9ac0398be7176605980164e49c6")
        self.assertEqual(M.NE.battery_sha256(sk), "8a2b95f2f0d50d347fb33400cc6519add784a826215c7666aa88df2e6316eb4b")

    def test_structure_and_rederived_draw(self):
        seed = M.NE.sealed_seed(SALT)
        self.guard.opened.clear()
        sk = M.build_skeleton(seed, COUNTS)
        self.assertEqual(self.guard.opened, [])  # the skeleton reads no file at all
        ins, p3 = [r for r in sk if r["surface"] == "in"], [r for r in sk if r["surface"] == "p3"]
        self.assertEqual((len(ins), len(p3), len({r["parent_id"] for r in ins})), (880, 880, 400))
        self.assertEqual(sk[:880], ins)
        gen.use_world("arith")
        self.assertEqual(ins, gen.make_battery("sealed", 40, seed, strata=M.STRATA, surfaces=("in",)))
        self.assertTrue(all(r["tid"] == "t0" for r in ins))
        rank, seen = Counter(), {}
        for a, b in zip(ins, p3):
            self.assertEqual({k: v for k, v in a.items() if k not in ("surface", "tid", "input")},
                             {k: v for k, v in b.items() if k not in ("surface", "tid", "input", "p3_pol", "p3_form",
                                                                      "p3_j", "p3_template")})
            self.assertEqual((b["surface"], b["tid"], b["input"], b["p3_template"]), ("p3", None, None, None))
            pid = b["parent_id"]
            if pid not in seen:
                seen[pid] = M.FORMS[rank[b["group"]] % 3]
                rank[b["group"]] += 1
            form = seen[pid]
            rng = random.Random(f"{seed}-{pid}")
            want = {"POS": rng.randrange(COUNTS[f"POS/{form}"]), "NEG": rng.randrange(COUNTS[f"NEG/{form}"])}
            pol = {"S+": "POS", "S-": "NEG"}[b["rule_ast"]["op"]]
            self.assertEqual((b["p3_form"], b["p3_pol"], b["p3_j"]), (form, pol, want[pol]))
        self.assertEqual(set(rank), set(M.STRATA))
        self.assertTrue(all(v == 40 for v in rank.values()))
        self.assertEqual(Counter(seen.values()), Counter({f: 140 - 10 * (f != M.FORMS[0]) for f in M.FORMS}))
        self.assertNotEqual(M.NE.battery_sha256(M.build_skeleton(seed + 1, COUNTS)), M.NE.battery_sha256(sk))

    def test_render(self):
        seed = M.NE.sealed_seed(SALT)
        sk = M.build_skeleton(seed, COUNTS)
        tpl = fake_p3_templates()
        by = by_cell_of(tpl)
        bat = M.render_p3(sk, by)
        self.assertEqual(M.NE.battery_sha256(bat), M.NE.battery_sha256(M.render_p3(M.build_skeleton(seed, COUNTS),
                                                                                    by)))
        self.assertEqual(bat[:880], sk[:880])
        text = {t["id"]: t["text"] for t in tpl}
        for q, r in zip(sk[880:], bat[880:]):
            tid = by[f"{q['p3_pol']}/{q['p3_form']}"][q["p3_j"]][0]
            self.assertEqual((r["tid"], r["p3_template"]), (f"p3:{tid}", tid))
            rule = text[tid].format(L=T.render_list(r["rule_ast"]["set"]))
            self.assertEqual(T.render_rule(r["rule_ast"], r["tid"]), rule)
            self.assertEqual(r["input"].split("\n")[0], "RULE: " + rule)
            self.assertEqual(W.parse_input(r["input"].replace("RULE: " + rule, "RULE: " + T.render_rule(
                r["rule_ast"], "t0")))[0], r["rule_ast"])
        used = {r["p3_template"] for r in bat[880:]}
        self.assertGreater(len(used), 40)
        with self.assertRaises(SystemExit):  # a p3 id registered with another text
            M.register_p3({"POS/IMPLICIT": [(by["POS/IMPLICIT"][0][0], "Other {L} text.")]})


# ----------------------------------------------------------------------------- SEALED protocol (fakes only)

class TestSealedProtocol(Base):
    def setUp(self):
        super().setUp()
        self.cells = self.tmp / "cells"
        self.tpl = fake_p3_templates()
        p3sha = write(M.P3_PATH, json.dumps({"schema": "rcx-e01r-pool-v1", "templates": self.tpl}))
        write(M.POOL_REFINE_PATH, json.dumps({"outputs": {"P3": {"file": M.P3_PATH.name, "sha256": p3sha, "n": 48}},
                                              "kept_by_cell": {"P3": COUNTS}}))
        write(M.ANALYZE_PATH, "# fake analysis\n")
        gen.use_world("arith")
        recs = gen.make_train("CF_SD", 5, 1)
        native = RC.native_rows(recs, FakeTok(), "CF_SD", np_mod=M.NPP)
        fake_rows = "".join(json.dumps(r) + "\n" for r in native)
        r3_rows = "".join(json.dumps(dict(r, target=TT.trace_target(x))) + "\n" for r, x in zip(native, recs))
        r2_rows = "".join(json.dumps(r) + "\n" for r in RC.native_rows(gen.make_train("CF_SD", 5, 2), FakeTok(), "R2",
                                                                      np_mod=M.NPP))
        ncfsd = write(M.E01N_CELLS / "train_N-CFSD.jsonl", fake_rows)
        self.rows = {"E-R2": write(self.cells / "train_E-R2.jsonl", r2_rows),
                     "E-R3": write(self.cells / "train_E-R3.jsonl", r3_rows)}
        self.branch_sha = write_branch(M.BRANCH_PATH, "B4")  # CX D-39
        self.write_manifest(self.rows)
        self.audit_sha = write(self.cells / M.AUDIT, json.dumps({"ok": True, "arm_ok": {a: True for a in self.rows},
                                                                 "rows_sha256": self.rows}))
        known = {}
        for k in range(3):
            tag = f"N-CFSD_s{k}"
            known[tag] = write(M.E01N_CELLS / tag / "adapter" / "adapter_model.safetensors", f"fake control {k}")
            write(M.E01N_CELLS / tag / "TRAIN_META.json",
                  json.dumps({"adapter_sha256": known[tag], "rows_sha256": ncfsd, "train": {"steps": 190},
                              "base_sha256": M.BASE_SHA256}))
        e01n_open = write(M.E01N_CELLS / "SEALED_OPEN.json", json.dumps({"adapters_sha256": known}))
        for name, val in (("P3_SHA256", p3sha), ("E01N_ADAPTERS", known), ("E01N_OPEN_SHA256", e01n_open),
                          ("NCFSD_ROWS_SHA256", ncfsd)):
            pt = mock.patch.object(M, name, val)
            pt.start()
            self.addCleanup(pt.stop)

    def write_manifest(self, rows, branch_sha=None, fallback=None):
        """native_e01r.build_cells' layout: the branch record, the per-arm entries, max_new."""
        return write(self.cells / M.MANIFEST, json.dumps({
            "branch": {"file": "BRANCH_STEP_A.json", "sha256": branch_sha or self.branch_sha}, "fallback": fallback,
            "arms": {a: {"sha256": s, "rows": 5} for a, s in rows.items()}, "max_new": 176}))

    def mint(self, tags=TAGS_B4, drop="", reason=""):
        with contextlib.redirect_stdout(io.StringIO()):
            M.cmd_mint_commit(SimpleNamespace(out=str(self.cells), arms_required=tags, drop_arm=drop, reason=reason))
        return M.sha(self.cells / "SEALED_COMMIT.json")

    def open_(self, commit_sha):
        with contextlib.redirect_stdout(io.StringIO()):
            M.cmd_open_sealed(SimpleNamespace(out=str(self.cells), expect_commit_sha=commit_sha))

    def adapter(self, tag, smoke_pass=True, **meta):
        """A completed fake treatment run (TRAIN_META as cmd_train writes it, after the mint) and, unless smoke_pass
        is None, its smoke record."""
        a = write(self.cells / tag / "adapter" / "adapter_model.safetensors", f"adapter {tag}")
        arm, seed = tag.rsplit("_s", 1)
        write(self.cells / tag / "TRAIN_META.json", json.dumps({
            "arm": arm, "seed": int(seed), "adapter_sha256": a, "rows_sha256": self.rows[arm], "train": {"steps": 190},
            "sealed_commit_sha256": M.sha(self.cells / "SEALED_COMMIT.json"), "audit_sha256": self.audit_sha,
            "base_sha256": M.BASE_SHA256, **meta}))
        if smoke_pass is not None:
            write(self.cells / f"SMOKE_{tag}.json", json.dumps({"adapter_sha256": a, "max_new": 176, "n": 50,
                                                                "truncated": 0 if smoke_pass else 2,
                                                                "pass": smoke_pass}))
        return a

    def all_adapters(self):
        return {t: self.adapter(t) for t in TREAT_B4}

    def test_mint_reads_no_p3(self):
        hidden = self.tmp / "elsewhere.bin"
        M.P3_PATH.rename(hidden)  # the mint must work without the P3 file
        self.guard.opened.clear()
        commit_sha = self.mint()
        hidden.rename(M.P3_PATH)
        self.assertFalse([p for p, _ in self.guard.opened if "P3_SEALED" in p.name or p == hidden])
        rec = json.loads((self.cells / "SEALED_COMMIT.json").read_text(encoding="utf-8"))
        salt = (self.cells / "SEALED_SALT.secret").read_text(encoding="utf-8")
        self.assertEqual(len(salt), 64)
        self.assertEqual(rec["sha256_salt"], sha_b(salt.encode()))
        skel = M.build_skeleton(M.NE.sealed_seed(salt), COUNTS)
        self.assertEqual(rec["battery_commit"], M.NE.battery_commit(salt, M.NE.battery_sha256(skel)))
        self.assertEqual((rec["p3"]["counts"], rec["p3"]["sha256"], rec["max_new"], rec["treatment_arms"]),
                         (COUNTS, M.P3_SHA256, 176, ["E-R2", "E-R3"]))
        self.assertEqual((rec["skeleton_records"], rec["surfaces"], rec["n_parents"]), (1760, ["in", "p3"], 40))
        self.assertEqual((rec["arms_required"], rec["dropped_arms"], rec["branch"]["branch"], rec["branch"]["sha256"]),
                         (TAGS_B4.split(","), [], "B4", self.branch_sha))
        self.assertEqual((rec["rows_sha256"], rec["audit_e01r_sha256"]), (self.rows, self.audit_sha))
        bound = {"native_e01r.py": HERE / "native_e01r.py", "recipes.py": HERE / "recipes.py",
                 "trace_target.py": HERE / "trace_target.py", "audit_e01r.py": HERE / "audit_e01r.py",
                 "decide_branch.py": HERE / "decide_branch.py", "analyze_e01r.py": M.ANALYZE_PATH,
                 "stats.py": ROOT / "rcx" / "stats.py", "wording_audit.py": ROOT / "rcx" / "rcl" / "wording_audit.py"}
        for name, path in bound.items():  # the commitment binds the analysis, statistics and audit code
            self.assertEqual(rec["code_sha256"][name], M.sha(path), name)
        self.assertEqual(rec["code_sha256"]["decide_branch.py"], M.DECIDE_SHA256)
        raw = (self.cells / "SEALED_COMMIT.json").read_text(encoding="utf-8")
        self.assertFalse([t for t in self.tpl if t["text"] in raw or t["id"] in raw])
        with self.assertRaises(SystemExit):
            self.mint()
        self.assertEqual(M.sha(self.cells / "SEALED_COMMIT.json"), commit_sha)

    def test_mint_refusals(self):
        b4 = TAGS_B4.split(",")
        for tags in ("N-CFSD_s0,N-CFSD_s1,N-CFSD_s2",  # T2: control only, against a manifest with treatments
                     TAGS_R3,  # E-R3 alone: E-R2 may be left out only by --drop-arm
                     "N-CFSD_s0,N-CFSD_s1,N-CFSD_s2,E-R1_s0,E-R1_s1,E-R1_s2,E-R3_s0,E-R3_s1,E-R3_s2",  # B3's arms
                     TAGS_B4 + ",E-R1_s0,E-R1_s1,E-R1_s2", ",".join(b4[1:]), ",".join(b4[3:]),
                     TAGS_B4 + ",N-CFSD_s0", TAGS_B4 + ",X_s0", TAGS_B4.replace("E-R2_s2", "E-R2_s3"),
                     ",".join(b4[:-1])):
            with self.subTest(tags=tags), self.assertRaises(SystemExit):
                self.mint(tags)
        (self.cells / "E-R3_s0").mkdir()
        with self.assertRaises(SystemExit):  # an adapter dir before the mint
            self.mint()
        (self.cells / "E-R3_s0").rmdir()
        aud = (self.cells / M.AUDIT).read_bytes()
        (self.cells / M.AUDIT).unlink()
        with self.assertRaises(SystemExit):
            self.mint()
        (self.cells / M.AUDIT).write_bytes(aud)
        man = (self.cells / M.MANIFEST).read_bytes()
        for rows, branch_sha, fb in (({"E-R3": self.rows["E-R3"]}, None, None),  # manifest arms != branch arms
                                     (dict(self.rows, **{"E-R1": "0" * 64}), None, None),
                                     (self.rows, "0" * 64, None),  # built from another BRANCH_STEP_A.json
                                     (self.rows, None, {"from": "E-R1", "to": "E-R2"})):  # a B3-only fallback
            self.write_manifest(rows, branch_sha, fb)
            with self.subTest(rows=sorted(rows), branch_sha=branch_sha, fallback=fb), \
                    self.assertRaises(SystemExit):
                self.mint()
        (self.cells / M.MANIFEST).write_bytes(man)
        rows_r2 = (self.cells / "train_E-R2.jsonl").read_bytes()
        write(self.cells / "train_E-R2.jsonl", "{}\n")
        with self.assertRaises(SystemExit):  # rows differ from the manifest
            self.mint()
        (self.cells / "train_E-R2.jsonl").write_bytes(rows_r2)
        write_branch(M.BRANCH_PATH, "B4", code_sha="0" * 64)
        with self.assertRaises(SystemExit):  # BRANCH_STEP_A.json not from the frozen decide_branch.py
            self.mint()
        M.BRANCH_PATH.unlink()
        with self.assertRaises(SystemExit):
            self.mint()
        M.ANALYZE_PATH.unlink()
        with self.assertRaises(SystemExit):
            self.mint()
        self.assertFalse((self.cells / "SEALED_COMMIT.json").exists() or (self.cells / "SEALED_SALT.secret").exists())

    def test_mint_refuses_without_analysis_code(self):
        M.ANALYZE_PATH.unlink()
        with self.assertRaises(SystemExit) as cm:
            self.mint()
        self.assertIn("analyze_e01r.py", str(cm.exception))
        self.assertFalse((self.cells / "SEALED_COMMIT.json").exists() or (self.cells / "SEALED_SALT.secret").exists())

    def test_mint_b1_control_only(self):
        self.branch_sha = write_branch(M.BRANCH_PATH, "B1")
        self.write_manifest({})
        with self.assertRaises(SystemExit):  # B1 commits no treatment arm
            self.mint()
        self.mint("N-CFSD_s0,N-CFSD_s1,N-CFSD_s2")
        rec = json.loads((self.cells / "SEALED_COMMIT.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["treatment_arms"], rec["arms_required"], rec["branch"]["branch"]),
                         ([], ["N-CFSD_s0", "N-CFSD_s1", "N-CFSD_s2"], "B1"))

    def test_mint_drop_arm(self):
        # E-R2's rows fail the row audit (PREREG section 2): E-R2 may be dropped, only explicitly and with a reason
        self.audit_sha = write(self.cells / M.AUDIT, json.dumps({
            "ok": False, "failures": ["frame:E-R2"], "arm_ok": {"E-R2": False, "E-R3": True},
            "rows_sha256": self.rows}))
        with self.assertRaises(SystemExit):  # no drop: E-R2's audit failed
            self.mint()
        for tags, drop, reason in ((TAGS_R3, "", ""), (TAGS_R3, "E-R2", ""), (TAGS_R3, "", "rows fail"),
                                   (TAGS_R3, "E-R1", "rows fail"), (TAGS_B4, "E-R2", "rows fail"),
                                   ("N-CFSD_s0,N-CFSD_s1,N-CFSD_s2", "E-R2", "rows fail")):
            with self.subTest(tags=tags, drop=drop, reason=reason), self.assertRaises(SystemExit):
                self.mint(tags, drop, reason)
        self.assertFalse((self.cells / "SEALED_COMMIT.json").exists())
        self.mint(TAGS_R3, "E-R2", "frame check fails on 3 rows; not fixable without changing R2")
        rec = json.loads((self.cells / "SEALED_COMMIT.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["treatment_arms"], rec["arms_required"]), (["E-R3"], TAGS_R3.split(",")))
        self.assertEqual(rec["dropped_arms"], [{"arm": "E-R2", "reason": "frame check fails on 3 rows; not fixable "
                                                                          "without changing R2"}])

    def test_open_refusals_precede_the_p3_read(self):
        commit_sha = self.mint()
        real = M.load_p3
        spy = mock.Mock(side_effect=real)
        pt = mock.patch.object(M, "load_p3", spy)
        pt.start()
        self.addCleanup(pt.stop)

        def refused(text=None):
            self.guard.opened.clear()
            with self.assertRaises(SystemExit) as cm:
                self.open_(commit_sha)
            spy.assert_not_called()
            self.assertFalse([p for p, _ in self.guard.opened if "P3_SEALED" in p.name])
            self.assertFalse((self.cells / "SEALED_OPEN.json").exists())
            if text:
                self.assertIn(text, str(cm.exception))

        refused("adapter missing")
        shas = self.all_adapters()
        self.adapter("E-R2_s1", smoke_pass=False)
        refused("smoke test failed")
        self.adapter("E-R2_s1", smoke_pass=None)
        (self.cells / "SMOKE_E-R2_s1.json").unlink()
        refused("smoke test missing")
        self.adapter("E-R2_s1", sealed_commit_sha256="0" * 64)
        refused("another SEALED_COMMIT.json")
        self.adapter("E-R2_s1", base_sha256="0" * 64)
        refused("base model sha")
        self.adapter("E-R2_s1", audit_sha256="0" * 64)
        refused("AUDIT_E01R.json")
        shas["E-R2_s1"] = self.adapter("E-R2_s1")
        for path, what in ((self.cells / "train_E-R3.jsonl", "training-row file"),
                           (M.E01N_CELLS / "train_N-CFSD.jsonl", "training-row file"),
                           (self.cells / M.AUDIT, "AUDIT_E01R.json"), (self.cells / M.MANIFEST, "TRAIN_MANIFEST"),
                           (self.cells / "SEALED_SALT.secret", "salt")):
            keep = path.read_bytes()
            path.write_bytes(keep + b"0")  # any extra byte (a newline would be stripped from the salt)
            with self.subTest(changed=path.name):
                refused(what)
            path.unlink()
            with self.subTest(missing=path.name):
                refused()
            path.write_bytes(keep)
        with mock.patch.object(M, "build_skeleton", lambda seed, counts, n=40: REAL_SKELETON(seed + 1, counts, n)):
            refused("skeleton")
        self.open_(commit_sha)
        spy.assert_called_once()  # P3 is read once, after every check passed
        o = json.loads((self.cells / "SEALED_OPEN.json").read_text(encoding="utf-8"))
        self.assertEqual(o["adapters_sha256"], {**shas, **M.E01N_ADAPTERS})
        self.assertEqual(o["code_changed_since_mint"], [])

    def test_open_eval_custody(self):
        commit_sha = self.mint()
        with self.assertRaises(SystemExit) as cm:
            self.open_(commit_sha)
        self.assertIn("adapter missing", str(cm.exception))
        shas = self.all_adapters()
        shas["E-R3_s2"] = self.adapter("E-R3_s2", smoke_pass=False)
        with self.assertRaises(SystemExit) as cm:
            self.open_(commit_sha)
        self.assertIn("smoke test failed", str(cm.exception))
        shas["E-R3_s2"] = self.adapter("E-R3_s2")
        with self.assertRaises(SystemExit):
            self.open_("0" * 64)
        with mock.patch.object(M, "E01N_ADAPTERS", dict(M.E01N_ADAPTERS, **{"N-CFSD_s1": "0" * 64})):
            with self.assertRaises(SystemExit) as cm:
                self.open_(commit_sha)
            self.assertIn("N-CFSD_s1", str(cm.exception))
        self.assertFalse((self.cells / "SEALED_OPEN.json").exists())
        self.open_(commit_sha)
        o = json.loads((self.cells / "SEALED_OPEN.json").read_text(encoding="utf-8"))
        salt = (self.cells / "SEALED_SALT.secret").read_text(encoding="utf-8")
        self.assertEqual((o["salt"], o["seed"], o["n_records"]), (salt, M.NE.sealed_seed(salt), 1760))
        self.assertEqual(o["adapters_sha256"], {**shas, **M.E01N_ADAPTERS})
        self.assertEqual(set(o["train_overlap"]["train_files"]), {"train_N-CFSD.jsonl", "train_E-R2.jsonl",
                                                                  "train_E-R3.jsonl"})
        self.assertFalse([t for t in self.tpl if t["text"] in json.dumps(o)])
        with self.assertRaises(SystemExit):
            self.open_(commit_sha)
        bat, seed, rec, opened = M.sealed_battery(self.cells)
        self.assertEqual((M.NE.battery_sha256(bat), seed), (o["battery_sha256"], o["seed"]))
        self.assertEqual(o["train_overlap"], M.train_overlap(  # computed on the skeleton = on the battery
            [M.E01N_CELLS / "train_N-CFSD.jsonl", self.cells / "train_E-R2.jsonl", self.cells / "train_E-R3.jsonl"],
            bat))
        self.assertEqual(o["r3_gold_utf8_bytes_plus_eos"], max(len(TT.trace_target(r).encode()) for r in bat) + 1)
        by = by_cell_of(self.tpl)
        for r in bat[880:]:
            self.assertEqual(r["p3_template"], by[f"{r['p3_pol']}/{r['p3_form']}"][r["p3_j"]][0])
        # eval-side custody (no model): fixed adapters only
        self.assertEqual(M.eval_adapter(self.cells, "E-R3", "E-R3_s1", opened)[1], shas["E-R3_s1"])
        self.assertEqual(M.eval_adapter(self.cells, "N-CFSD", "N-CFSD_s2", opened),
                         (M.E01N_CELLS / "N-CFSD_s2" / "adapter", M.E01N_ADAPTERS["N-CFSD_s2"]))
        self.assertEqual(M.eval_adapter(self.cells, "OFF", "OFF", opened), (None, None))
        for arm, tag in (("E-R1", "E-R1_s0"), ("E-R3", "E-R3_s3"), ("X", "X_s0")):
            with self.subTest(tag=tag), self.assertRaises(SystemExit):
                M.eval_adapter(self.cells, arm, tag, opened)
        write(self.cells / "E-R3_s0" / "adapter" / "adapter_model.safetensors", "tampered")
        with self.assertRaises(SystemExit):
            M.eval_adapter(self.cells, "E-R3", "E-R3_s0", opened)
        with mock.patch.object(M, "_heavy", side_effect=AssertionError("_heavy reached")):
            with self.assertRaises(SystemExit) as cm:  # lane not held: refused before any model load
                M.cmd_eval(SimpleNamespace(out=str(self.cells), arm="E-R3", seed=1, battery="sealed"))
        self.assertIn("lane lock", str(cm.exception))
        tpl = [dict(t, text=t["text"].replace("decides", "settles")) if i == 0 else t for i, t in enumerate(self.tpl)]
        write(M.P3_PATH, json.dumps({"schema": "rcx-e01r-pool-v1", "templates": tpl}))
        with self.assertRaises(SystemExit) as cm:
            M.sealed_battery(self.cells)
        self.assertIn("sha256 differs", str(cm.exception))
        self.assertNotIn("settles", str(cm.exception))
        (self.cells / "E-R3_s0" / "adapter" / "adapter_model.safetensors").unlink()
        with self.assertRaises(SystemExit) as cm:  # no training after opening
            M.cmd_train(SimpleNamespace(out=str(self.cells), arm="E-R3", seed=0))
        self.assertIn("after opening", str(cm.exception))

    # ------------------------------------------------------------------- T6: smoke and eval end to end (fake model)

    def run_cmd(self, fn, lc, **args):
        loaded = []
        with mock.patch.object(M, "_heavy", return_value=(SimpleNamespace(need_ram=lambda: None), lc, FakeGuard)), \
                mock.patch.dict(sys.modules, {"peft": fake_peft(loaded)}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            fn(SimpleNamespace(out=str(self.cells), battery="sealed", **args))
        return loaded, out.getvalue()

    def test_cmd_smoke_end_to_end(self):
        self.mint()
        for t in TREAT_B4:
            self.adapter(t, smoke_pass=None)
        rows = [json.loads(x) for x in (self.cells / "train_E-R3.jsonl").read_text(encoding="utf-8").splitlines()]
        target = {r["prompt"]: r["target"] for arm in ("E-R2", "E-R3") for r in map(json.loads, (
            self.cells / f"train_{arm}.jsonl").read_text(encoding="utf-8").splitlines())}

        def script(n_trunc):
            def f(j, prompt):
                if j < n_trunc:  # no EOS within max_new: truncated
                    return target[prompt][:25], False
                return (target[prompt] if j % 3 else target[prompt].split("\n")[-1]), True  # 2 lines, or bare JSON
            return f

        with self.assertRaises(SystemExit) as cm:  # lane not held: refused before the model load
            self.run_cmd(M.cmd_smoke, FakeLC(script(0)), arm="E-R3", seed=0)
        self.assertIn("lane lock", str(cm.exception))
        self.hold_lane()
        lc = FakeLC(script(1))
        loaded, _ = self.run_cmd(M.cmd_smoke, lc, arm="E-R3", seed=0)
        s = json.loads((self.cells / "SMOKE_E-R3_s0.json").read_text(encoding="utf-8"))
        self.assertEqual((lc.max_new_seen, lc.loads, loaded), ([176], 1, [self.cells / "E-R3_s0" / "adapter"]))
        self.assertEqual((s["max_new"], s["n"], s["truncated"], s["pass"], s["row_indices"]),
                         (176, 50, 1, True, M.smoke_select(5)))
        self.assertEqual(s["outs"], [script(1)(j, rows[i]["prompt"])[0] for j, i in enumerate(s["row_indices"])])
        self.assertEqual([x["finished"] for x in s["items"]], [False] + [True] * 49)
        n_bare = sum(j % 3 == 0 for j in range(1, 50))
        self.assertEqual((s["scorable_share"], s["exact_share"], s["well_formed_share"]),
                         (49 / 50, 49 / 50, (49 - n_bare) / 50))  # last-line rule scores the bare JSON too
        self.assertEqual(M.smoke_problem(self.cells, "E-R3_s0", s["adapter_sha256"], 176), None)
        lc2 = FakeLC(script(2))
        self.run_cmd(M.cmd_smoke, lc2, arm="E-R2", seed=1)
        s = json.loads((self.cells / "SMOKE_E-R2_s1.json").read_text(encoding="utf-8"))
        self.assertEqual((s["truncated"], s["pass"], lc2.max_new_seen), (2, False, [176]))
        self.assertIn("smoke test failed", M.smoke_problem(self.cells, "E-R2_s1", s["adapter_sha256"], 176))
        lc3 = FakeLC(script(0))
        _, out = self.run_cmd(M.cmd_smoke, lc3, arm="E-R3", seed=0)  # create-only: never rerun
        self.assertEqual((out.strip(), lc3.max_new_seen), ("smoke exists", []))
        self.adapter("E-R3_s1", smoke_pass=None, base_sha256="0" * 64)
        with self.assertRaises(SystemExit) as cm:
            self.run_cmd(M.cmd_smoke, FakeLC(script(0)), arm="E-R3", seed=1)
        self.assertIn("base model sha", str(cm.exception))

    def eval_script(self, battery):
        """Per battery record j: 0 the R3 gold, 1 the bare native JSON, 2 line 1 + the unchanged balances, 3 no
        balances at all, 4 cut off before EOS (truncated)."""
        def f(j, prompt):
            r = battery[j]
            gold = TT.trace_target(r)
            line1 = gold.split("\n")[0]
            return [(gold, True), (TT.target_json(r["gold"]["after"]), True),
                    (line1 + "\n" + TT.target_json(r["state"]), True), (line1, True), (gold[:30], False)][j % 5]
        return f

    def opened_cells(self):
        commit_sha = self.mint()
        shas = self.all_adapters()
        self.open_(commit_sha)
        self.hold_lane()
        return shas, M.sealed_battery(self.cells)[0]

    def test_cmd_eval_end_to_end(self):
        shas, bat = self.opened_cells()
        script = self.eval_script(bat)
        lc = FakeLC(script)
        loaded, _ = self.run_cmd(M.cmd_eval, lc, arm="E-R3", seed=1)
        self.assertEqual((lc.max_new_seen, lc.loads, loaded), ([176], 1, [self.cells / "E-R3_s1" / "adapter"]))
        ev = json.loads((self.cells / "eval_sealed_E-R3_s1.json").read_text(encoding="utf-8"))
        n_trunc = sum(j % 5 == 4 for j in range(len(bat)))
        self.assertEqual((ev["n"], ev["max_new"], ev["truncated"], ev["adapter_sha256"], ev["parser"]),
                         (1760, 176, n_trunc, shas["E-R3_s1"], "trace_target.parse_last_line"))
        with gzip.open(self.cells / "raw_sealed_E-R3_s1.json.gz", "rt", encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(ev["raw_sha256"], M.sha(self.cells / "raw_sealed_E-R3_s1.json.gz"))
        self.assertEqual((raw["max_new"], raw["adapter_sha256"], raw["battery_sha256"], len(raw["decode_calls"])),
                         (176, shas["E-R3_s1"], ev["battery_sha256"], 1760))
        self.assertEqual(raw["outs"], [script(j, None)[0] for j in range(len(bat))])
        with gzip.open(self.cells / "preds_sealed_E-R3_s1.jsonl.gz", "rt", encoding="utf-8") as f:
            preds = [json.loads(x) for x in f]
        self.assertEqual([p["i"] for p in preds], list(range(1760)))
        for j, (p, r) in enumerate(zip(preds, bat)):
            want = SC.classify(r, M.last_line_answer(script(j, None)[0], list(r["gold"]["after"])))
            self.assertEqual({k: p[k] for k in want}, want)
            self.assertEqual((p["out"], p["finished"]), (script(j, None)[0], j % 5 != 4))
            self.assertIn("trace", p)
            if j % 5 in (0, 1):  # R3 gold and bare JSON: the last non-empty line is the exact answer
                self.assertEqual(p["exact"], 1)
            if j % 5 == 3:
                self.assertEqual(p["err"], "format")
            if j % 5 == 0:
                self.assertEqual((p["trace"]["well_formed"], p["trace"]["stated_decision_ok"]), (1, 1))
        self.assertEqual(ev["cells"], SC.aggregate(bat, [{k: p[k] for k in ("exact", "intent", "decision_ok",
                                                                                     "err")} for p in preds]))
        lc2 = FakeLC(script)
        _, out = self.run_cmd(M.cmd_eval, lc2, arm="E-R3", seed=1)  # create-only: never decoded twice
        self.assertEqual((out.strip(), lc2.max_new_seen), ("eval exists", []))
        # a non-trace arm parses the whole output: the two-line outputs are format errors there
        lc3 = FakeLC(script)
        self.run_cmd(M.cmd_eval, lc3, arm="E-R2", seed=0)
        with gzip.open(self.cells / "preds_sealed_E-R2_s0.jsonl.gz", "rt", encoding="utf-8") as f:
            preds = [json.loads(x) for x in f]
        self.assertEqual({p["err"] for j, p in enumerate(preds) if j % 5 in (0, 2, 3, 4)}, {"format"})
        self.assertEqual({p["exact"] for j, p in enumerate(preds) if j % 5 == 1}, {1})
        self.assertNotIn("trace", preds[0])
        self.assertEqual(lc3.max_new_seen, [176])  # the committed max_new for every arm

    def test_cmd_eval_raw_first_and_resume(self):
        _, bat = self.opened_cells()
        script = self.eval_script(bat)
        lc = FakeLC(script)
        with mock.patch.object(M, "map_finished", side_effect=RuntimeError("mapping failed")):
            with self.assertRaises(RuntimeError):
                self.run_cmd(M.cmd_eval, lc, arm="E-R3", seed=2)
        raw_p = self.cells / "raw_sealed_E-R3_s2.json.gz"
        self.assertTrue(raw_p.exists())  # the generations survive a failed post-processing
        self.assertFalse((self.cells / "eval_sealed_E-R3_s2.json").exists()
                         or (self.cells / "preds_sealed_E-R3_s2.jsonl.gz").exists())
        raw_sha = M.sha(raw_p)
        with mock.patch.object(M, "_heavy", side_effect=AssertionError("no model load when resuming")), \
                contextlib.redirect_stdout(io.StringIO()):
            M.cmd_eval(SimpleNamespace(out=str(self.cells), arm="E-R3", seed=2, battery="sealed"))
        ev = json.loads((self.cells / "eval_sealed_E-R3_s2.json").read_text(encoding="utf-8"))
        self.assertEqual((ev["truncated"], ev["raw_sha256"], M.sha(raw_p), lc.max_new_seen),
                         (sum(j % 5 == 4 for j in range(len(bat))), raw_sha, raw_sha, [176]))
        # a raw file generated for another adapter is never post-processed under this tag
        lc2 = FakeLC(script)
        with mock.patch.object(M, "map_finished", side_effect=RuntimeError("mapping failed")):
            with self.assertRaises(RuntimeError):
                self.run_cmd(M.cmd_eval, lc2, arm="E-R2", seed=2)
        os.replace(self.cells / "raw_sealed_E-R2_s2.json.gz", self.cells / "raw_sealed_E-R2_s1.json.gz")
        with self.assertRaises(SystemExit) as cm:
            self.run_cmd(M.cmd_eval, FakeLC(script), arm="E-R2", seed=1)
        self.assertIn("generated for another", str(cm.exception))

    def test_eval_refused_before_opening(self):
        self.mint()
        for arm, seed in (("N-CFSD", 0), ("E-R3", 0), ("OFF", 0)):
            with self.subTest(arm=arm), self.assertRaises(SystemExit) as cm:
                M.cmd_eval(SimpleNamespace(out=str(self.cells), arm=arm, seed=seed, battery="sealed"))
            self.assertIn("not open", str(cm.exception))

    def test_load_p3_checks(self):
        ref = M.p3_reference()
        self.assertEqual(M.load_p3(ref), by_cell_of(self.tpl))
        for bad in ({"text": "No slot here."}, {"text": "Two {L} and {L}."}, {"text": "Brace {x} {L}."},
                    {"pol": "BOTH"}, {"id": "bad id"}, {"id": self.tpl[1]["id"]}):
            tpl = [dict(self.tpl[0], **bad)] + self.tpl[1:]
            data = json.dumps({"templates": tpl}).encode()
            M.P3_PATH.write_bytes(data)
            with self.subTest(bad=bad), self.assertRaises(SystemExit) as cm:
                M.load_p3(dict(ref, sha256=sha_b(data)))
            self.assertNotIn("Two", str(cm.exception))
        tpl = self.tpl[1:]
        data = json.dumps({"templates": tpl}).encode()
        M.P3_PATH.write_bytes(data)
        with self.assertRaises(SystemExit):  # per-cell counts
            M.load_p3(dict(ref, sha256=sha_b(data)))
        with self.assertRaises(SystemExit):  # sha checked before parsing
            M.load_p3(ref)

    def test_train_overlap(self):
        gen.use_world("arith")
        bat = M.render_p3(M.build_skeleton(M.NE.sealed_seed(SALT), COUNTS), by_cell_of(self.tpl))
        clean = M.train_overlap([self.cells / "train_E-R3.jsonl"], bat)
        self.assertEqual((clean["semantic_parents"], clean["parent_ids"]), (0, []))
        r = bat[3]
        row = {"prompt": f"... {r['event']['status']} ...", "semantic_hash": r["semantic_hash"], "skeleton_hash": "x"}
        write(self.tmp / "leak.jsonl", json.dumps(row) + "\n")
        hit = M.train_overlap([self.tmp / "leak.jsonl"], bat)
        self.assertEqual(hit["parent_ids"], [r["parent_id"]])
        if r["event"]["status"] not in ("posted", "void"):
            self.assertIn(r["event"]["status"], hit["status_words_in_train_prompts"])


# ----------------------------------------------------------------------------- scoring, truncation, smoke, lane

class TestScoringSmokeLane(Base):
    def test_last_line_rule(self):
        k = ["A", "B"]
        self.assertEqual(M.last_line_answer('check status="x" listed=no only-listed:false -> noop\n{"A":1,"B":2}', k),
                         {"A": 1, "B": 2})
        self.assertEqual(M.last_line_answer('{"A":1,"B":2}', k), {"A": 1, "B": 2})
        self.assertEqual(M.last_line_answer('{"A":1,"B":2}\n \n', k), {"A": 1, "B": 2})
        for bad in ('{"A":1,"B":2}\nnot json', "", "\n", 'x\n{"A":1}', 'check -> noop'):
            self.assertIsNone(M.last_line_answer(bad, k), bad)
        parse, name = M.answer_parser("E-R1")
        self.assertIs(parse, M.NPP.parse_answer)
        self.assertIsNone(parse('check x -> apply\n{"A":1,"B":2}', k))  # whole output for non-trace arms
        parse, name = M.answer_parser("E-R3")
        self.assertEqual(parse('check x -> apply\n{"A":1,"B":2}', k), {"A": 1, "B": 2})
        if hasattr(TT, "parse_last_line"):
            self.assertEqual(name, "trace_target.parse_last_line")
        with mock.patch.object(TT, "parse_last_line", lambda text, keys: None, create=True):
            with self.assertRaises(SystemExit):
                M.answer_parser("E-R3")

    def test_eos_probe(self):
        src = ast.unparse(fn_node(ROOT / "rcx" / "native" / "lora_cpu.py", "generate"))
        self.assertIn("order = sorted(range(len(enc)), key=lambda i: len(enc[i]))", src)
        self.assertIn("outs[j] = tok.decode(out[r, maxlen:], skip_special_tokens=True)", src)
        tok = FakeTok()
        prompts = ["ccccccc", "a", "bbbb", "dddd", "e"]  # 3, 1, 2, 2, 1 tokens: ties keep prompt order (stable sort)
        outs = ["X3", "X1", "X2", "X4", "X5"]
        probe = M.EosProbe(tok, [151645, 151643])
        order = sorted(range(5), key=lambda i: len(tok(prompts[i])["input_ids"]))
        self.assertEqual(order, [1, 4, 2, 3, 0])
        ends = {0: [151645], 1: [], 2: [151643, 151643], 3: [], 4: [151645, 151643]}
        for j in order:
            self.assertEqual(probe.decode(tok(outs[j])["input_ids"] + ends[j], skip_special_tokens=True), outs[j])
        self.assertEqual(probe.finished(prompts, outs), [True, False, True, False, True])
        lens = probe.prompt_lengths(prompts)
        self.assertEqual(lens, [3, 1, 2, 2, 1])
        calls = [[t, e] for t, e in probe.calls]  # as stored in raw_sealed_<tag>.json.gz
        self.assertEqual(M.map_finished(lens, calls, outs), [True, False, True, False, True])  # no tokenizer needed
        for bad in ((lens[:4], calls, outs), (lens, calls[:4], outs), (lens, calls, outs[::-1])):
            with self.assertRaises(RuntimeError):
                M.map_finished(*bad)
        self.assertEqual(probe.pad_token_id, 151643)
        bad = M.EosProbe(tok, [151645])
        for j in reversed(order):
            bad.decode(tok(outs[j])["input_ids"])
        with self.assertRaises(RuntimeError):
            bad.finished(prompts, outs)

    def test_smoke(self):
        idx = M.smoke_select(1501)
        self.assertEqual((len(idx), len(set(idx)), idx[0], idx == M.smoke_select(1501)), (50, 50, 0, True))
        self.assertTrue(all(0 <= i < 1501 for i in idx))
        gold = '{"A":1,"B":2}'
        rows = [{"target": 'check status="x" listed=no only-listed:false -> noop\n' + gold}] * 50
        outs = [rows[0]["target"]] * 48 + ['{"A":1,"B":2}', 'check status="x" listed=no']
        for n_trunc, ok in ((0, True), (1, True), (2, False)):
            rep = M.smoke_report("E-R3", rows, outs, [True] * (50 - n_trunc) + [False] * n_trunc)
            self.assertEqual((rep["truncated"], rep["pass"]), (n_trunc, ok))
        self.assertEqual((rep["well_formed_share"], rep["scorable_share"], rep["exact_share"]), (0.96, 0.98, 0.98))
        rep = M.smoke_report("E-R2", [{"target": gold}] * 50, [gold] * 49 + ["{"], [True] * 50)
        self.assertEqual((rep["well_formed_share"], rep["pass"]), (0.98, True))
        cells = self.tmp / "cells"
        self.assertEqual(M.smoke_problem(cells, "E-R3_s0", "a", 176), "smoke test missing")
        write(cells / "SMOKE_E-R3_s0.json", json.dumps({"adapter_sha256": "a", "max_new": 176, "n": 50, "pass": True}))
        self.assertIsNone(M.smoke_problem(cells, "E-R3_s0", "a", 176))
        self.assertIsNotNone(M.smoke_problem(cells, "E-R3_s0", "b", 176))
        self.assertIsNotNone(M.smoke_problem(cells, "E-R3_s0", "a", 160))

    def test_require_lane(self):
        def refused(text):
            with self.assertRaises(SystemExit) as cm:
                M.require_lane()
            self.assertIn(text, str(cm.exception))

        refused("not held by 'rcx'")  # no lock
        self.hold_lane(party="bwm")  # a foreign lock, even with its token exported
        refused("not held by 'rcx'")
        lane.lock_path().unlink()
        self.hold_lane(pid=2 ** 31 - 4)  # T3: a stale lock (no such process), token exported
        self.assertFalse(lane.pid_alive(2 ** 31 - 4))
        refused("stale")
        lane.lock_path().unlink()
        rec = self.hold_lane()
        with mock.patch.object(lane, "pid_alive", return_value=False):  # the chain died after acquiring
            refused("stale")
        for bad in ("garbage", None):
            lane.lock_path().write_text(json.dumps(dict(rec, pid=bad)), encoding="utf-8")
            refused("stale")
        lane.lock_path().write_text(json.dumps(rec), encoding="utf-8")
        os.environ.pop(M.LANE_TOKEN_ENV)  # a live rcx lock, but this process is not in the chain that holds it
        refused(M.LANE_TOKEN_ENV)
        os.environ[M.LANE_TOKEN_ENV] = "0" * 32  # another chain's (or an old) token
        refused(M.LANE_TOKEN_ENV)
        lane.lock_path().write_text(json.dumps(dict(rec, token="")), encoding="utf-8")
        os.environ[M.LANE_TOKEN_ENV] = ""
        refused(M.LANE_TOKEN_ENV)
        lane.lock_path().write_text(json.dumps(rec), encoding="utf-8")
        os.environ[M.LANE_TOKEN_ENV] = rec["token"]
        got = M.require_lane()
        self.assertEqual((got["holder"], got["pid"], got["token"]), ("rcx", os.getpid(), rec["token"]))
        self.assertTrue(str(lane.lock_path()).startswith(str(self.tmp)))
        self.assertTrue(lane.lock_path().exists())  # never reclaimed or released by the check


if __name__ == "__main__":
    unittest.main()
