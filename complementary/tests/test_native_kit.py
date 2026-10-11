"""Tests for the RCX-E01N native kit (data hygiene, matching, provenance). Never loads 0.6B weights: models are tiny
randomly initialised Qwen3s; only the real tokenizer is loaded (local files only)."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RCX_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01N"))

import torch  # noqa: E402

torch.set_num_threads(2)

import native_e01 as NE  # noqa: E402
import native_probe as NP  # noqa: E402
from rcx.native import lora_cpu as LC  # noqa: E402
from rcx.rcl import gen, templates as T, world as W  # noqa: E402

CELLS = ROOT / "runs" / "RCX-E01N" / "cells"
TOK_DIR = LC.PATHS["sparse"]
_TOK = []


def get_tok():
    if not _TOK:
        if not (TOK_DIR / "tokenizer_config.json").exists():
            raise unittest.SkipTest("Titan tokenizer not available")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOK_DIR), local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = "<|endoftext|>"
        _TOK.append(tok)
    return _TOK[0]


def tiny_model(vocab):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).float().eval()


class LiteralTok:
    """Chat-template stand-in for native_rows (the audit proves render_literal == the real template)."""

    def apply_chat_template(self, msgs, **kw):
        assert kw.get("add_generation_prompt") and kw.get("enable_thinking") is False and not kw.get("tokenize")
        return NE.render_literal(msgs[0]["content"], msgs[1]["content"])


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_cells(d, rows_by_arm):
    man = {}
    for arm, rows in rows_by_arm.items():
        p = Path(d) / f"train_{arm}.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        man[arm] = {"path": p.name, "sha256": NE.sha(p), "rows": len(rows)}
    (Path(d) / "TRAIN_MANIFEST.json").write_text(json.dumps(man), encoding="utf-8", newline="\n")


def native_row(rec, tid=None):
    """A row exactly as native_rows builds it, from any RCL record (optionally re-rendered with template tid)."""
    system = NP.rcl_system(T.render_rule(rec["rule_ast"], tid or rec["tid"]))
    return {"prompt": NE.render_literal(system, NP.user_json(rec)), "target": NE.target_json(rec["gold"]["after"]),
            "regime": "VR", "parent_id": rec["parent_id"], "variant": rec["variant"],
            "decision": rec["gold"]["decision"], "semantic_hash": rec["semantic_hash"],
            "skeleton_hash": rec["skeleton_hash"], "status": rec["event"]["status"]}


class TestAuditCleanBuild(unittest.TestCase):
    """Self-contained: rebuild the three arms with the literal template, audit, then corrupt and re-audit."""

    @classmethod
    def setUpClass(cls):
        cls.rows = {arm: NE.native_rows(reg, LiteralTok()) for arm, reg in NE.ARMS.items()}

    def test_literal_rebuild_equals_built_files(self):
        if not CELLS.exists():
            self.skipTest("cells not built")
        for arm in NE.ARMS:
            self.assertEqual(self.rows[arm], NE.load_rows(CELLS, arm), arm)

    def test_clean_build_passes_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            write_cells(d, self.rows)
            res = NE.audit_native(d, tok=None)
            self.assertTrue(res["ok"], res["failures"])
            self.assertTrue((Path(d) / "AUDIT.json").exists())
            for arm, v in res["arms"].items():
                self.assertEqual(v["rows"], NE.N_ROWS)
                self.assertEqual(set(v["overlap"].values()), {0}, arm)
                self.assertEqual(set(v["hard_check_rows"].values()), {0}, arm)
            # (4) every stratum x surface present, each parent has an apply and a no-op variant
            self.assertEqual(len(res["battery"]["strata"]), len(gen.STRATA) * 2)
            for k, c in res["battery"]["strata"].items():
                self.assertEqual(c["parents"], 40, k)
                self.assertEqual(c["parents_with_apply_and_noop"], 40, k)
            # PREREG-3 words appear in the battery only as F4 near-miss probes (SPEC 7)
            self.assertEqual(set(res["battery"]["prereg3_words_by_stratum"]), {"F4_nearmiss"})
            # (3) N-FX vs T2 is reported; (5) N-CFSD's 3-account rows are reported as a design note
            if res["n_fx_vs_t2"] is not None:
                self.assertFalse(res["n_fx_vs_t2"]["material"])
            self.assertIn(3, res["arms"]["N-CFSD"]["class_balance_by_n_accounts"])
            self.assertTrue(any(n.startswith("N-CFSD:") for n in res["design_notes"]))
            self.assertEqual(res["battery"]["n_accounts"], {"ho_tpl:4": 1120, "in:4": 1120})

    def test_negative_controls_are_caught(self):
        gen.use_world("arith")
        bat = gen.make_battery("dev", 40, NP.BATTERY_SEED)
        v1 = next(r for r in bat if r["surface"] == "in" and r["group"] == "V1_pos")
        vr = [dict(r) for r in self.rows["N-VR"]]
        tr = [r for r in gen.make_train("VR", 8, NE.DATA_SEED["VR"])]
        vr[0] = native_row(v1)                                      # eval item leaked into training
        vr[1] = native_row(tr[1], tid="h0")                         # held-out template wording
        rec = json.loads(json.dumps(tr[2]))
        rec["event"]["status"] = "pending"                          # PREREG-3 word
        vr[2] = native_row(rec)
        u = json.loads(NP.user_json(tr[3]))
        u["dest"] = 1                                               # RCWM register field-name-4 text
        vr[3] = dict(native_row(tr[3]), prompt=vr[3]["prompt"].replace(
            NP.user_json(tr[3]), json.dumps(u, separators=(",", ":"), sort_keys=True)))
        vr[4] = dict(vr[4], target=vr[4]["target"].replace(":", ":1", 1))   # wrong gold
        rec = json.loads(json.dumps(tr[5]))
        rec["state"][next(iter(rec["state"]))] = 500                # M1_bal magnitude
        vr[5] = native_row(rec)
        with tempfile.TemporaryDirectory() as d:
            write_cells(d, self.rows)
            with open(Path(d) / "train_N-VR.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in vr:
                    f.write(json.dumps(r) + "\n")
            res = NE.audit_native(d, tok=None)
        self.assertFalse(res["ok"])
        want = {"semantic_overlap:N-VR", "skeleton_overlap:N-VR", "split_words:N-VR", "non_train_words:N-VR",
                "heldout_template:N-VR", "excluded_words:N-VR", "rcwm_heldout_text:N-VR", "frame:N-VR",
                "gold:N-VR", "heldout_magnitude:N-VR", "manifest_sha:N-VR"}
        if res["rcwm_heldout_texts"] is None:
            want.discard("rcwm_heldout_text:N-VR")
        self.assertTrue(want <= set(res["failures"]), want - set(res["failures"]))
        self.assertFalse([f for f in res["failures"] if f.endswith(("N-FX", "N-CFSD"))], res["failures"])

    def test_rcwm_heldout_texts_match_design_docs(self):
        if not (NE.RCWM_T1 / "scripts" / "t2bp_common.py").exists():
            self.skipTest("RCWM tree not available")
        t = NE.rcwm_heldout_texts()
        self.assertTrue(t["prereq paraphrase 4"].startswith("Apply the actions in order. Each action fires only when"))
        self.assertEqual(t["prereq state sentence 4"] + "{w}" + t["prereq state sentence 4 tail"],
                         "The state is a flag mask {w} bits wide")
        names = {v for k, v in t.items() if "field name 4" in k}
        for n in ("mask", "moves", "plus", "minus", "must_on", "must_off",
                  "dest", "peer", "offset", "pivot", "opcode", "memory", "operation"):
            self.assertIn(json.dumps(n) + ":", names)


class TestAuditBuiltRowsWithTokenizer(unittest.TestCase):
    def test_built_rows_pass_with_token_covariates(self):
        if not CELLS.exists():
            self.skipTest("cells not built")
        tok = get_tok()
        with tempfile.TemporaryDirectory() as d:
            for p in list(CELLS.glob("train_N-*.jsonl")) + [CELLS / "TRAIN_MANIFEST.json"]:
                shutil.copy(p, d)
            res = NE.audit_native(d, tok=tok)
        self.assertTrue(res["ok"], res["failures"])
        for arm, v in res["arms"].items():
            self.assertEqual(set(v["tokens"]["per_seed"]), {"0", "1", "2"})
            self.assertLessEqual(v["tokens"]["max_processed_per_row"], 512)  # train_sft max_len: no silent drops
        self.assertIn("token_covariates_vs_N-FX", res["matched"])
        self.assertEqual(set(res["matched"]["rows"].values()), {NE.N_ROWS})

    def test_literal_template_equals_tokenizer_template(self):
        tok = get_tok()
        gen.use_world("arith")
        for r in gen.make_train("CF_SD", 12, 5):
            msgs = [{"role": "system", "content": NP.rcl_system(T.render_rule(r["rule_ast"], r["tid"]))},
                    {"role": "user", "content": NP.user_json(r)}]
            self.assertEqual(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False),
                             NE.render_literal(msgs[0]["content"], msgs[1]["content"]))

    def test_token_covariates_match_train_sft_counts(self):
        tok = get_tok()
        allr = NE.native_rows("CF_SD", LiteralTok())
        rows = [r for r in allr if len(NE.read_back(r)["state"]) == 3][:6] + \
            [r for r in allr if len(NE.read_back(r)["state"]) == 4][:6]   # 3- and 4-account rows
        pm = LC.attach_lora(tiny_model(len(tok)), seed=0)
        log = LC.train_sft(pm, tok, rows, steps=2, batch_size=4, micro=2, seed=0, log_every=100,
                           logger=lambda m: None)
        cov = NE.token_covariates(rows, tok, steps=2, batch=4, seeds=(0,))
        self.assertEqual(cov["per_seed"]["0"]["supervised_tokens"], log["supervised_tokens"])
        self.assertEqual(cov["per_seed"]["0"]["tokens_processed"], log["tokens_processed"])


class TestTrainProvenanceAndGate(unittest.TestCase):
    def setUp(self):
        self.saved = {"need_ram": NP.need_ram, "load_core": LC.load_core, "PATHS": LC.PATHS,
                      "train_sft": LC.train_sft, "file": LC.__file__}
        self.calls = []
        NP.need_ram = lambda: 99.0
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.d = Path(self.tmp.name)
        core = self.d / "core"
        core.mkdir()
        (core / "model.safetensors").write_bytes(b"not a model")
        LC.PATHS = {"sparse": core}
        self.rows_path = self.d / "train_N-VR.jsonl"
        with open(self.rows_path, "w", encoding="utf-8", newline="\n") as f:
            for r in NE.native_rows("VR", LiteralTok())[:40]:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        NP.need_ram = self.saved["need_ram"]
        LC.load_core, LC.PATHS = self.saved["load_core"], self.saved["PATHS"]
        LC.train_sft, LC.__file__ = self.saved["train_sft"], self.saved["file"]
        self.tmp.cleanup()

    def _audit(self, ok=True, rows_sha=None):
        (self.d / "AUDIT.json").write_text(json.dumps({"ok": ok, "rows_sha256": {
            "N-VR": rows_sha or NE.sha(self.rows_path)}}), encoding="utf-8", newline="\n")

    def _load_core_spy(self, name="sparse"):
        self.calls.append(name)
        raise AssertionError("model must not be loaded")

    def test_train_refuses_unaudited_rows(self):
        LC.load_core = self._load_core_spy
        a = Namespace(out=str(self.d), arm="N-VR", seed=0)
        with self.assertRaises(SystemExit):
            NE.cmd_train(a)                                   # no AUDIT.json
        self._audit(ok=False)
        with self.assertRaises(SystemExit):
            NE.cmd_train(a)                                   # failed audit
        self._audit(rows_sha="0" * 64)
        with self.assertRaises(SystemExit):
            NE.cmd_train(a)                                   # audit of other rows
        self.assertEqual(self.calls, [])

    def test_shas_are_fixed_before_training(self):
        tok = get_tok()
        self._audit()
        rows_sha0, code_sha0 = NE.sha(self.rows_path), NE.sha(Path(self.saved["file"]))
        changed = self.d / "lora_cpu_changed.py"
        changed.write_text("# edited while the run was training\n", encoding="utf-8")
        LC.load_core = lambda name="sparse": (tiny_model(len(tok)), tok)

        def fake_train_sft(pm, tok_, rows, **kw):
            # simulate edits on disk during an hours-long run: rows rebuilt, trainer code changed
            self.rows_path.write_text(self.rows_path.read_text(encoding="utf-8").split("\n", 1)[1],
                                      encoding="utf-8", newline="\n")
            LC.__file__ = str(changed)
            return {"wall_s": 0.0, "steps": 0, "rows_trained": len(rows)}

        LC.train_sft = fake_train_sft
        NE.cmd_train(Namespace(out=str(self.d), arm="N-VR", seed=0))
        meta = json.loads((self.d / "N-VR_s0" / "TRAIN_META.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["rows_sha256"], rows_sha0)
        self.assertEqual(meta["code_sha256"]["lora_cpu.py"], code_sha0)
        self.assertEqual(meta["audit_sha256"], NE.sha(self.d / "AUDIT.json"))
        self.assertEqual(meta["train"]["rows_trained"], 40)


class TestEvalBatterySha(unittest.TestCase):
    def test_eval_sha_equals_probe_battery_file(self):
        tok = get_tok()
        saved = (NP.need_ram, LC.load_core, LC.generate)
        NP.need_ram = lambda: 99.0
        LC.load_core = lambda name="sparse": (object(), tok)
        LC.generate = lambda model, tok_, prompts, **kw: ["{}"] * len(prompts)
        try:
            with tempfile.TemporaryDirectory() as d:
                NE.cmd_eval(Namespace(out=d, arm="OFF", seed=0, n_parents=1, surfaces="in,ho_tpl", battery="dev"))
                rec = json.loads((Path(d) / "eval_OFF.json").read_text(encoding="utf-8"))
                bpath = Path(d) / "battery.jsonl"                 # written exactly as native_probe.main does
                with open(bpath, "w", encoding="utf-8", newline="\n") as f:
                    for r in NE.eval_battery(1, ["in", "ho_tpl"]):
                        f.write(json.dumps(r) + "\n")
                self.assertEqual(rec["battery_sha256"], NP.sha_file(bpath))
        finally:
            NP.need_ram, LC.load_core, LC.generate = saved


class TestSealedCustody(unittest.TestCase):
    """SEALED protocol (PREREG_E01N section 2) in a temp dir with fake adapters and tiny training-row files: never the
    real cells dir, never a model load (need_ram/load_core raise AssertionError, so a refusal must come first)."""
    TAGS = "N-VR_s0,N-CFSD_s0"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.d = Path(self.tmp.name)
        gen.use_world("arith")
        write_cells(self.d, {"N-VR": [native_row(r) for r in gen.make_train("VR", 6, NE.DATA_SEED["VR"])],
                             "N-CFSD": [native_row(r) for r in gen.make_train("CF_SD", 6, NE.DATA_SEED["CF_SD"])]})
        self.man = json.loads((self.d / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
        self.saved = (NP.need_ram, LC.load_core, LC.generate, LC.PATHS, gen.make_battery)
        NP.need_ram = LC.load_core = self._no_model
        core = self.d / "core"
        core.mkdir()
        (core / "model.safetensors").write_bytes(b"not a model")
        LC.PATHS = {"sparse": core, "supa": core}

    def tearDown(self):
        NP.need_ram, LC.load_core, LC.generate, LC.PATHS, gen.make_battery = self.saved
        self.tmp.cleanup()

    @staticmethod
    def _no_model(*a, **k):
        raise AssertionError("a model must not be loaded here")

    def ns(self, **kw):
        base = dict(out=str(self.d), arm="N-VR", seed=0, n_parents=2, surfaces="in,ho_tpl", battery="sealed",
                    arms_required=self.TAGS, expect_commit_sha="")
        base.update(kw)
        return Namespace(**base)

    def fake_adapter(self, tag, payload=b"w", meta_over=None, steps=NE.STEPS):
        arm, seed = NE.SEALED_TAG_RE.fullmatch(tag).groups()
        ad = self.d / tag / "adapter"
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "adapter_model.safetensors").write_bytes(payload + tag.encode())
        meta = {"arm": arm, "seed": int(seed), "rows_sha256": self.man[arm]["sha256"],
                "adapter_sha256": NE.sha(ad / "adapter_model.safetensors"), "train": {"steps": steps}}
        meta.update(meta_over or {})
        (self.d / tag / "TRAIN_META.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")

    def mint(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            NE.cmd_mint_commit(self.ns())
        return NE.sha(self.d / "SEALED_COMMIT.json"), buf.getvalue()

    def open_ok(self):
        csha, _ = self.mint()
        for t in self.TAGS.split(","):
            self.fake_adapter(t)
        self.quiet(NE.cmd_open_sealed, self.ns(expect_commit_sha=csha))
        return csha, json.loads((self.d / "SEALED_OPEN.json").read_text(encoding="utf-8"))

    @staticmethod
    def quiet(fn, *args):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)

    def assertRefused(self, fn, *args):
        with self.assertRaises(SystemExit):
            self.quiet(fn, *args)

    def test_mint_validates_tags_and_never_reveals_the_salt(self):
        for bad in ("", "N-VR_s0, N-VR-s1", "N-VR_s0,N-VR_s0", "T2,N-VR_s0"):
            self.assertRefused(NE.cmd_mint_commit, self.ns(arms_required=bad))
        self.assertEqual(sorted(p.name for p in self.d.glob("SEALED_*")), [])
        csha, printed = self.mint()
        salt = (self.d / "SEALED_SALT.secret").read_text(encoding="utf-8").strip()
        text = (self.d / "SEALED_COMMIT.json").read_text(encoding="utf-8")
        rec = json.loads(text)
        self.assertNotIn(salt, text)
        self.assertNotIn(salt, printed)
        if NE.sealed_seed(salt) >= 10 ** 6:                                    # short seeds may occur in shas by chance
            self.assertNotIn(str(NE.sealed_seed(salt)), text)
        self.assertEqual(rec["sha256_salt"], hashlib.sha256(salt.encode()).hexdigest())
        self.assertEqual(rec["arms_required"], ["N-VR_s0", "N-CFSD_s0"])
        bsha = NE.battery_sha256(NE._sealed_records(NE.sealed_seed(salt), 2, ["in", "ho_tpl"]))
        self.assertEqual(rec["battery_commit"], NE.battery_commit(salt, bsha))
        self.assertIn(csha, printed)
        self.assertRefused(NE.cmd_mint_commit, self.ns())                   # no re-mint
        self.assertEqual(NE.sha(self.d / "SEALED_COMMIT.json"), csha)

    def test_salt_file_is_git_ignored_and_commit_is_not(self):
        import subprocess
        try:
            def ignored(rel):
                return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT,
                                      capture_output=True).returncode == 0
            salt_ign, commit_ign = (ignored("runs/RCX-E01N/cells/SEALED_SALT.secret"),
                                    ignored("runs/RCX-E01N/cells/SEALED_COMMIT.json"))
        except OSError:
            self.skipTest("git not available")
        self.assertTrue(salt_ign)
        self.assertFalse(commit_ign)

    def test_eval_sealed_refuses_before_open(self):
        self.mint()
        for t in self.TAGS.split(","):
            self.fake_adapter(t)
        self.assertRefused(NE.cmd_eval, self.ns(arm="OFF"))
        self.assertRefused(NE.cmd_eval, self.ns(arm="N-VR"))
        self.assertRefused(NE.sealed_battery, self.d, ["in", "ho_tpl"])
        self.assertEqual(list(self.d.glob("eval_*")) + list(self.d.glob("preds_*")), [])

    def test_open_refuses_incomplete_or_tampered_runs(self):
        csha, _ = self.mint()
        op = self.ns(expect_commit_sha=csha)
        self.assertRefused(NE.cmd_open_sealed, op)                           # no adapters
        self.fake_adapter("N-VR_s0")
        self.assertRefused(NE.cmd_open_sealed, op)                           # one adapter missing
        bad_runs = [dict(meta_over={"adapter_sha256": "0" * 64}), dict(steps=3), dict(meta_over={"seed": 1}),
                    dict(meta_over={"rows_sha256": self.man["N-VR"]["sha256"]})]
        for kw in bad_runs:
            self.fake_adapter("N-CFSD_s0", **kw)
            self.assertRefused(NE.cmd_open_sealed, op)
        self.fake_adapter("N-CFSD_s0")
        (self.d / "N-CFSD_s0" / "ckpt").mkdir()
        (self.d / "N-CFSD_s0" / "ckpt" / "state.pt").write_bytes(b"x")       # interrupted run
        self.assertRefused(NE.cmd_open_sealed, op)
        (self.d / "N-CFSD_s0" / "ckpt" / "state.pt").unlink()
        (self.d / "N-CFSD_s0" / "TRAIN_META.json").unlink()                  # adapter saved, meta not yet written
        self.assertRefused(NE.cmd_open_sealed, op)
        self.fake_adapter("N-CFSD_s0")
        for wrong in ("", "0" * 64):
            self.assertRefused(NE.cmd_open_sealed, self.ns(expect_commit_sha=wrong))
        salt_p = self.d / "SEALED_SALT.secret"
        salt = salt_p.read_text(encoding="utf-8")
        salt_p.write_text("f" * 64, encoding="utf-8", newline="\n")
        self.assertRefused(NE.cmd_open_sealed, op)                           # salt swapped
        salt_p.write_text(salt, encoding="utf-8", newline="\n")
        self.assertFalse((self.d / "SEALED_OPEN.json").exists())
        self.quiet(NE.cmd_open_sealed, op)
        self.assertTrue((self.d / "SEALED_OPEN.json").exists())

    def test_generator_change_after_mint_is_caught(self):
        csha, _ = self.mint()
        for t in self.TAGS.split(","):
            self.fake_adapter(t)
        real = self.saved[4]
        gen.make_battery = lambda *a, **k: real(*a, **k)[1:]                  # e.g. a stratum edited after DEV
        self.assertRefused(NE.cmd_open_sealed, self.ns(expect_commit_sha=csha))
        gen.make_battery = real
        self.quiet(NE.cmd_open_sealed, self.ns(expect_commit_sha=csha))
        gen.make_battery = lambda *a, **k: real(*a, **k)[1:]
        self.assertRefused(NE.sealed_battery, self.d, ["in", "ho_tpl"])

    def test_open_once_and_eval_bound_to_the_opened_battery_and_adapters(self):
        csha, o = self.open_ok()
        salt = (self.d / "SEALED_SALT.secret").read_text(encoding="utf-8").strip()
        bat, seed = NE.sealed_battery(self.d, ["ho_tpl", "in"])
        self.assertEqual((o["salt"], o["seed"], seed), (salt, NE.sealed_seed(salt), NE.sealed_seed(salt)))
        self.assertEqual(o["battery_sha256"], NE.battery_sha256(bat))
        self.assertEqual(set(o["adapters_sha256"]), {"N-VR_s0", "N-CFSD_s0"})
        self.assertEqual((o["train_overlap"]["parent_ids"], o["train_overlap"]["status_words_in_train_prompts"]),
                         ([], []))
        self.assertTrue(all(r["split"] == "sealed" and r["parent_id"].startswith("sealed-") for r in bat))
        before = NE.sha(self.d / "SEALED_OPEN.json")
        self.assertRefused(NE.cmd_open_sealed, self.ns(expect_commit_sha=csha))    # write-once
        self.assertEqual(NE.sha(self.d / "SEALED_OPEN.json"), before)
        self.assertRefused(NE.cmd_eval, self.ns(arm="OFF", surfaces="in"))          # not the committed surfaces
        (self.d / "N-VR_s0" / "adapter" / "adapter_model.safetensors").write_bytes(b"swapped after opening")
        self.assertRefused(NE.cmd_eval, self.ns(arm="N-VR", seed=0))
        self.fake_adapter("N-CFSD_s1")                                              # trained after opening
        self.assertRefused(NE.cmd_eval, self.ns(arm="N-CFSD", seed=1))
        NP.need_ram = lambda: 99.0
        LC.load_core = lambda name="sparse": (object(), LiteralTok())
        LC.generate = lambda model, tok, prompts, **kw: ["{}"] * len(prompts)
        self.quiet(NE.cmd_eval, self.ns(arm="OFF"))
        rec = json.loads((self.d / "eval_sealed_OFF.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["battery"], rec["battery_seed"], rec["battery_sha256"], rec["n"]),
                         ("sealed", seed, o["battery_sha256"], len(bat)))
        self.assertFalse((self.d / "eval_OFF.json").exists())

    def test_overlap_of_a_leaked_sealed_item_is_recorded(self):
        _, o = self.open_ok()
        bat, _ = NE.sealed_battery(self.d, ["in", "ho_tpl"])
        leak = next(r for r in bat if r["group"] == "V1_pos" and r["surface"] == "in")
        with open(self.d / "train_N-FX.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(native_row(leak)) + "\n")
        ov = NE.sealed_train_overlap(self.d, bat)
        self.assertEqual(ov["parent_ids"], [leak["parent_id"]])
        self.assertGreaterEqual(ov["semantic_parents"], 1)
        self.assertIn(leak["event"]["status"], ov["status_words_in_train_prompts"])


# ------------------------------------------------------------------------------------------------ recipe parity
# Self-contained block: lora_cpu vs the vendored BWM D-03 / RCWM T2 trainer (rcwm/.../vendor_bwm/bwmn/lm.py and
# scripts/train_generic.py), imported read-only by file path with bytecode writing off (nothing is written under
# ../rcwm). Tiny random Qwen3 (fixed seed) with the real Titan tokenizer's vocabulary; never loads 0.6B weights.

import ast  # noqa: E402
import contextlib as _rp_contextlib  # noqa: E402
import gc as _rp_gc  # noqa: E402
import importlib.util as _rp_ilu  # noqa: E402
import inspect as _rp_inspect  # noqa: E402
import io as _rp_io  # noqa: E402
import random as _rp_random  # noqa: E402
import warnings as _rp_warnings  # noqa: E402
from unittest import mock as _rp_mock  # noqa: E402

_RP_ROOT = Path(__file__).resolve().parents[1]
for _p in (_RP_ROOT, _RP_ROOT / "experiments" / "RCX-E01N"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import native_e01 as _RP_NE  # noqa: E402
from rcx.native import lora_cpu as _RP_LC  # noqa: E402

_RP_RCWM_T1 = _RP_ROOT.parent / "rcwm" / "experiments" / "RCWM-T1"
_RP_VENDOR = _RP_RCWM_T1 / "vendor_bwm"
_RP_T2_CFG = _RP_RCWM_T1 / "runs" / "t2" / "t2_s0" / "adapter" / "adapter_config.json"
_RP_HAVE = ((_RP_VENDOR / "bwmn" / "lm.py").exists() and (_RP_VENDOR / "scripts" / "train_generic.py").exists()
            and (_RP_LC.PATHS["sparse"] / "tokenizer_config.json").exists())
RP_STEPS, RP_BATCH, RP_MICRO, RP_SEED, RP_LR = 6, 8, 4, 0, 1e-3
_RP = {}


def _rp_load_by_path(name, path):
    """Import a file under a private module name without writing bytecode next to it."""
    saved_flag, saved_path = sys.dont_write_bytecode, list(sys.path)
    sys.dont_write_bytecode = True
    os.environ["BWM_DEVICE"] = "cpu"  # vendored CPU lane: no CUDA calls, no GPU thermal pauses
    try:
        spec = _rp_ilu.spec_from_file_location(name, str(path))
        mod = _rp_ilu.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses with postponed annotations look their module up here
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path  # train_generic.py inserts vendor_bwm into sys.path
        sys.dont_write_bytecode = saved_flag
    return mod


def _rp_vendored(which):
    if which not in _RP:
        rel = {"lm": ("bwmn", "lm.py"), "tg": ("scripts", "train_generic.py")}[which]
        _RP[which] = _rp_load_by_path(f"rcx_test_vendored_{which}", _RP_VENDOR.joinpath(*rel))
    return _RP[which]


def _rp_tok():
    if "tok" not in _RP:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(_RP_LC.PATHS["sparse"]), local_files_only=True)
        assert tok.eos_token_id == 151645 and tok.pad_token_id == 151643
        _RP["tok"] = tok
    return _RP["tok"]


def _rp_model(seed=5):
    """Tiny random Qwen3ForCausalLM; vocab = len(tok) (tok.vocab_size excludes the EOS/pad special ids)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(_rp_tok()), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).eval()


_RP_SYS = ("Apply one event to the supplied account balances. {rule} add: add amount to account. scale: multiply "
           "account by amount. transfer: subtract amount from account and add it to other, simultaneously. Return "
           "only the complete resulting balances as a JSON object.")


def _rp_rows(n=40, seed=7):
    """Ledger-like prompt/target rows in Titan's chat frame; targets of varying length (exercises the token-weighted
    micro-batch accumulation and padding)."""
    tok, rng, rows = _rp_tok(), _rp_random.Random(seed), []
    for i in range(n):
        keys = sorted(rng.sample(["A", "B", "C", "D", "E", "F"], rng.randint(2, 5)))
        before = {k: rng.randint(-99, 999) for k in keys}
        acc, oth = rng.sample(keys, 2)
        kind, amt, status = rng.choice(["add", "transfer"]), rng.randint(1, 50), rng.choice(["posted", "void", "tavo"])
        after = dict(before)
        if status != "void":
            after[acc] += amt if kind == "add" else -amt
            if kind == "transfer":
                after[oth] += amt
        rule = "Only events with status posted{} take effect; other events change nothing.".format(
            " or tavo" if i % 2 else "")
        user = json.dumps({"before": before, "event": {"account": acc, "amount": amt, "kind": kind, "other": oth,
                                                       "status": status, "tick": 1}},
                          separators=(",", ":"), sort_keys=True)
        prompt = tok.apply_chat_template([{"role": "system", "content": _RP_SYS.format(rule=rule)},
                                          {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt, "target": json.dumps(after, separators=(",", ":"), sort_keys=True)})
    return rows


def _rp_lora(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _rp_gc_state(model):
    """Gradient-checkpointing / train-mode state of the PEFT-wrapped Qwen3ForCausalLM."""
    inner = model.base_model.model
    gc_mods = [m for m in inner.modules() if getattr(m, "gradient_checkpointing", False)]
    kws = sorted({tuple(sorted(m._gradient_checkpointing_func.keywords.items())) for m in gc_mods})
    return {"training": inner.training, "n_gc_modules": len(gc_mods), "gc_kwargs": kws,
            "embed_hooks": len(inner.get_input_embeddings()._forward_hooks) > 0}


class _RPRecorder:
    """Patches torch.optim.AdamW, torch.nn.utils.clip_grad_norm_ and <module>.chunked_ce (identically for both
    trainers) to record optimizer hyperparameters, clipping calls and per-micro-batch (token-mean loss, n tokens)."""

    def __init__(self, module):
        self.module, self.opts, self.clips, self.micro = module, [], [], []

    def __enter__(self):
        rec = self
        real_adamw, real_clip, real_ce = torch.optim.AdamW, torch.nn.utils.clip_grad_norm_, self.module.chunked_ce

        class RecAdamW(real_adamw):
            def __init__(self, params, *args, **kw):
                params = list(params)
                super().__init__(params, *args, **kw)
                rec.opts.append({"args": args, "kw": dict(kw), "defaults": dict(self.defaults),
                                 "shapes": [tuple(p.shape) for p in params]})

        def clip(params, max_norm, *args, **kw):
            params = list(params)
            gn = real_clip(params, max_norm, *args, **kw)
            rec.clips.append({"max_norm": max_norm, "args": args, "kw": kw, "n_params": len(params),
                              "gn": float(gn)})
            return gn

        def ce(model, ids, att, lab, *args, **kw):
            loss, n = real_ce(model, ids, att, lab, *args, **kw)
            rec.micro.append((float(loss), int(n)))
            return loss, n

        self._patches = [_rp_mock.patch.object(torch.optim, "AdamW", RecAdamW),
                         _rp_mock.patch.object(torch.nn.utils, "clip_grad_norm_", clip),
                         _rp_mock.patch.object(self.module, "chunked_ce", ce)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    def step_losses(self, micro_per_step):
        """Per-step batch loss = token-weighted mean of the step's micro-batch losses (what both trainers log)."""
        out = []
        for i in range(0, len(self.micro), micro_per_step):
            chunk = self.micro[i:i + micro_per_step]
            n = sum(c[1] for c in chunk)
            out.append(sum(loss * k / n for loss, k in chunk))
        return out


class _RPGuard:
    """rcx guard stand-in: records the training state before every micro-batch; optionally raises on call k
    (a crash/kill mid-run)."""

    class Stop(Exception):
        pass

    def __init__(self, model, raise_on_call=None):
        self.model, self.raise_on_call, self.calls, self.states = model, raise_on_call, 0, []

    def check(self):
        self.calls += 1
        if self.raise_on_call is not None and self.calls == self.raise_on_call:
            raise _RPGuard.Stop(f"simulated interruption at guard call {self.calls}")
        self.states.append(_rp_gc_state(self.model))


def _rp_close(tc, a, b, rel=1e-6, what=""):
    if isinstance(a, dict):
        tc.assertEqual(sorted(a), sorted(b), what)
        for k in a:
            _rp_close(tc, a[k], b[k], rel, f"{what}[{k}]")
    elif torch.is_tensor(a):
        scale = max(float(a.abs().max()), float(b.abs().max()), 1e-30)
        tc.assertLessEqual(float((a - b).abs().max()), rel * scale, what)
    else:
        tc.assertLessEqual(abs(a - b), rel * max(abs(a), abs(b), 1e-30), what)


@unittest.skipUnless(_RP_HAVE, "vendored RCWM trainer or Titan tokenizer not present")
class TestRecipeParity(unittest.TestCase):
    """Same tiny random Qwen3 (fixed seed), LoRA attached by each kit's attach_lora (same seed), 6 steps x batch 8
    (micro 4) on 40 rows tokenised with the real Titan tokenizer, same row_order and seed: vendored lm.train_sft
    (gpu_yield no-op'd: no sleeps, no nvidia-smi subprocess) vs lora_cpu.train_sft."""

    @classmethod
    def setUpClass(cls):
        cls.lm, cls.tok, cls.rows = _rp_vendored("lm"), _rp_tok(), _rp_rows(40)
        cls.order = _RP_LC.row_order_for(len(cls.rows), RP_STEPS, RP_BATCH, RP_SEED)
        base_v, base_r = _rp_model(5), _rp_model(5)
        for (na, pa), (nb, pb) in zip(base_v.state_dict().items(), base_r.state_dict().items()):
            assert na == nb and torch.equal(pa, pb)
        cls.pm_v = cls.lm.attach_lora(base_v, seed=RP_SEED)
        cls.pm_r = _RP_LC.attach_lora(base_r, seed=RP_SEED)
        cls.init_v, cls.init_r = _rp_lora(cls.pm_v), _rp_lora(cls.pm_r)
        cls.yield_states = []
        with _RPRecorder(cls.lm) as rv, _rp_mock.patch.object(
                cls.lm, "gpu_yield", lambda *a, **k: cls.yield_states.append(_rp_gc_state(cls.pm_v))):
            cls.out_v = cls.lm.train_sft(cls.pm_v, cls.tok, cls.rows, steps=RP_STEPS, batch_size=RP_BATCH, lr=RP_LR,
                                         seed=RP_SEED, row_order=cls.order, max_len=512, micro=RP_MICRO,
                                         logger=lambda m: None)
        cls.rec_v, cls.guard_r = rv, _RPGuard(cls.pm_r)
        with _RPRecorder(_RP_LC) as rr:
            cls.out_r = _RP_LC.train_sft(cls.pm_r, cls.tok, cls.rows, steps=RP_STEPS, batch_size=RP_BATCH, lr=RP_LR,
                                         seed=RP_SEED, row_order=cls.order, micro=RP_MICRO, guard=cls.guard_r,
                                         logger=lambda m: None)
        cls.rec_r = rr
        cls.final_v, cls.final_r = _rp_lora(cls.pm_v), _rp_lora(cls.pm_r)

    # ---------------------------------------------------------------- end-to-end
    def test_loss_trajectory_matches(self):
        lv, lr_ = self.rec_v.step_losses(RP_BATCH // RP_MICRO), self.rec_r.step_losses(RP_BATCH // RP_MICRO)
        self.assertEqual((len(lv), len(lr_)), (RP_STEPS, RP_STEPS))
        for s, (a, b) in enumerate(zip(lv, lr_)):
            _rp_close(self, a, b, 1e-6, f"step {s} loss")
        self.assertEqual([n for _, n in self.rec_v.micro], [n for _, n in self.rec_r.micro])
        # returned summaries (the vendored hist is float32-rounded via torch.tensor(): compare at 1e-6 relative)
        for k in ("final_loss_avg50", "max_grad_norm"):
            _rp_close(self, self.out_v[k], self.out_r[k], 1e-6, k)
        for k in ("steps", "batch_size", "rows_available", "lr_peak", "tokens_processed", "supervised_tokens"):
            self.assertEqual(self.out_v[k], self.out_r[k], k)
        self.assertGreater(lv[0], lv[-1])  # the run actually trains

    def test_grad_norms_match(self):
        self.assertEqual((len(self.rec_v.clips), len(self.rec_r.clips)), (RP_STEPS, RP_STEPS))
        for a, b in zip(self.rec_v.clips, self.rec_r.clips):
            _rp_close(self, a["gn"], b["gn"], 1e-6, "grad norm")

    def test_final_lora_weights_match(self):
        _rp_close(self, self.init_v, self.init_r, 0.0, "init")
        _rp_close(self, self.final_v, self.final_r, 1e-6, "final")
        self.assertTrue(all(not torch.equal(self.final_r[n], self.init_r[n]) for n in self.final_r))

    # ---------------------------------------------------------------- components
    def test_row_order_matches_train_generic(self):
        """Run the real vendored train_generic.main() (core load, LoRA, training and save stubbed) and compare the
        row_order / arguments it hands to lm.train_sft with lora_cpu.row_order_for and lora_cpu.train_sft defaults."""
        tg = _rp_vendored("tg")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td, _rp_warnings.catch_warnings():
            _rp_warnings.simplefilter("ignore", ResourceWarning)  # train_generic never closes its files
            tmp = Path(td)
            for n_rows, steps, batch, seed in [(40, 6, 8, 0), (1501, 190, 16, 0), (1501, 190, 16, 1), (7, 5, 16, 3)]:
                rows_p = tmp / f"rows_{n_rows}.jsonl"
                rows_p.write_text("".join(json.dumps({"prompt": f"p{i}", "target": f"t{i}"}) + "\n"
                                          for i in range(n_rows)), encoding="utf-8", newline="\n")
                got = {}

                def fake_train(model, tok, rows, **kw):
                    got.update({k: v for k, v in kw.items() if k != "logger"}, n=len(rows))
                    return {"stub": True}

                argv = ["train_generic.py", "--rows", str(rows_p), "--steps", str(steps), "--batch", str(batch),
                        "--seed", str(seed), "--out", str(tmp / f"out_{n_rows}_{seed}")]
                with _rp_mock.patch.object(tg.lm, "load_core", lambda name: ("MODEL", "TOK")), \
                        _rp_mock.patch.object(tg.lm, "attach_lora",
                                              lambda model, seed=0, **kw: got.update(attach_seed=seed) or model), \
                        _rp_mock.patch.object(tg.lm, "train_sft", fake_train), \
                        _rp_mock.patch.object(tg.lm, "save_adapter", lambda *a, **k: None), \
                        _rp_mock.patch.object(sys, "argv", argv), \
                        _rp_contextlib.redirect_stdout(_rp_io.StringIO()):
                    tg.main()
                _rp_gc.collect()
                self.assertEqual(got["row_order"], _RP_LC.row_order_for(n_rows, steps, batch, seed))
                self.assertEqual((got["attach_seed"], got["seed"], got["n"]), (seed, seed, n_rows))
            # T2's CLI defaults (train_generic) == lora_cpu.train_sft defaults, which native_e01 relies on
            sig = _rp_inspect.signature(_RP_LC.train_sft).parameters
            vsig = _rp_inspect.signature(self.lm.train_sft).parameters
            self.assertEqual(got["lr"], sig["lr"].default)
            self.assertEqual(got["max_len"], sig["max_len"].default)
            self.assertEqual(sig["batch_size"].default, 16)
            for k in ("grad_clip", "micro"):
                self.assertEqual(vsig[k].default, sig[k].default, k)
            self.assertIs(vsig["grad_ckpt"].default, True)

    def test_wsd_lr_equal(self):
        for s in range(190):
            self.assertEqual(_RP_LC.wsd_lr(s, 190, 1e-4), self.lm.wsd_lr(s, 190, 1e-4), s)
        for total in range(1, 25):
            for s in range(total):
                self.assertEqual(_RP_LC.wsd_lr(s, total, 3e-4), self.lm.wsd_lr(s, total, 3e-4), (s, total))
        self.assertAlmostEqual(_RP_LC.wsd_lr(0, 190, 1e-4), 1e-4 / 19)
        self.assertEqual(_RP_LC.wsd_lr(18, 190, 1e-4), 1e-4)
        self.assertEqual(_RP_LC.wsd_lr(151, 190, 1e-4), 1e-4)
        self.assertAlmostEqual(_RP_LC.wsd_lr(189, 190, 1e-4), 1e-4 * (1 - 0.9 * 37 / 38))

    def test_encode_pair_and_masking_equal(self):
        eos = self.tok.eos_token_id
        for r in self.rows:
            for use_eos in (True, False):
                a = _RP_LC.encode_pair(self.tok, r["prompt"], r["target"], 512, eos=use_eos)
                b = self.lm.encode_pair(self.tok, r["prompt"], r["target"], 512, eos=use_eos)
                self.assertEqual((a.input_ids, a.labels, a.n_target), (b.input_ids, b.labels, b.n_target))
            p = self.tok(r["prompt"], add_special_tokens=False).input_ids
            t = self.tok(r["target"], add_special_tokens=False).input_ids
            a = _RP_LC.encode_pair(self.tok, r["prompt"], r["target"], 512)
            self.assertEqual(a.input_ids, p + t + [eos])
            self.assertEqual(a.labels, [-100] * len(p) + t + [eos])
            self.assertEqual(a.n_target, len(t) + 1)
            n = len(p) + len(t) + 1
            self.assertIsNone(_RP_LC.encode_pair(self.tok, r["prompt"], r["target"], n - 1))
            self.assertIsNone(self.lm.encode_pair(self.tok, r["prompt"], r["target"], n - 1))
            self.assertIsNotNone(_RP_LC.encode_pair(self.tok, r["prompt"], r["target"], n))
        enc = [_RP_LC.encode_pair(self.tok, r["prompt"], r["target"], 512) for r in self.rows[:6]]
        for x, y in zip(_RP_LC.collate(enc, self.tok.pad_token_id), self.lm.collate(enc, self.tok.pad_token_id)):
            self.assertTrue(torch.equal(x, y))
        for trainer, pm in ((_RP_LC.train_sft, self.pm_r), (self.lm.train_sft, self.pm_v)):  # refuse, never truncate
            with self.assertRaises(ValueError):
                trainer(pm, self.tok, self.rows[:1], steps=1, batch_size=1, max_len=8, micro=1)

    def test_chunked_ce_equal_and_exact(self):
        m = _rp_model(9)
        enc = [_RP_LC.encode_pair(self.tok, r["prompt"], r["target"], 512) for r in self.rows[:4]]
        ids, lab, att = _RP_LC.collate(enc, self.tok.pad_token_id)
        with torch.no_grad():
            a, na = _RP_LC.chunked_ce(m, ids, att, lab, chunk=7)
            b, nb = self.lm.chunked_ce(m, ids, att, lab, chunk=7)
            logits = m(input_ids=ids, attention_mask=att).logits.float()
            ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                    lab[:, 1:].reshape(-1), ignore_index=-100)
        self.assertEqual((na, nb), (sum(e.n_target for e in enc),) * 2)
        self.assertEqual(float(a), float(b))
        _rp_close(self, float(a), float(ref), 1e-5, "chunked vs full-logits CE")

    def test_lora_config_equal(self):
        cv, cr = self.pm_v.peft_config["default"].to_dict(), self.pm_r.peft_config["default"].to_dict()
        self.assertEqual(cv, cr)
        self.assertEqual((cr["r"], cr["lora_alpha"], cr["lora_dropout"], cr["bias"]), (16, 32, 0.05, "none"))
        self.assertEqual(str(getattr(cr["task_type"], "value", cr["task_type"])), "CAUSAL_LM")
        self.assertEqual(set(cr["target_modules"]), {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                     "down_proj"})
        self.assertEqual(sorted(self.init_v), sorted(self.init_r))
        self.assertEqual(len(self.init_r), 2 * 7 * 2)  # 2 layers x 7 targets x (A, B)
        if _RP_T2_CFG.exists():  # the registered T2 s0 adapter's config (read-only)
            t2 = json.loads(_RP_T2_CFG.read_text(encoding="utf-8"))
            mine = json.loads(json.dumps(cr, default=lambda o: sorted(o) if isinstance(o, set)
                                         else getattr(o, "value", str(o))))
            for k in ("r", "lora_alpha", "lora_dropout", "bias", "task_type", "use_rslora", "use_dora",
                      "init_lora_weights", "fan_in_fan_out", "modules_to_save", "layers_to_transform"):
                self.assertEqual(t2[k], mine[k], k)
            self.assertEqual(set(t2["target_modules"]), set(cr["target_modules"]))

    def test_optimizer_and_clip_equal(self):
        self.assertEqual((len(self.rec_v.opts), len(self.rec_r.opts)), (1, 1))
        ov, orr = self.rec_v.opts[0], self.rec_r.opts[0]
        self.assertEqual((ov["args"], ov["kw"], ov["defaults"], ov["shapes"]),
                         (orr["args"], orr["kw"], orr["defaults"], orr["shapes"]))
        d = orr["defaults"]
        self.assertEqual((d["betas"], d["eps"], d["weight_decay"], d["amsgrad"]), ((0.9, 0.95), 1e-8, 0.0, False))
        self.assertEqual(len(orr["shapes"]), len(self.init_r))  # exactly the LoRA tensors are optimised
        for a, b in zip(self.rec_v.clips, self.rec_r.clips):
            self.assertEqual((a["max_norm"], a["args"], a["kw"], a["n_params"]),
                             (b["max_norm"], b["args"], b["kw"], b["n_params"]))
            self.assertEqual(b["max_norm"], 1.0)

    def test_gradient_checkpointing_equal(self):
        self.assertEqual(len(self.yield_states), 2 * RP_STEPS)
        self.assertEqual(len(self.guard_r.states), (RP_BATCH // RP_MICRO) * RP_STEPS)

        def dump(states):
            return {json.dumps(s, sort_keys=True) for s in states}

        self.assertEqual(dump(self.yield_states), dump(self.guard_r.states))
        s = self.guard_r.states[0]
        self.assertTrue(s["training"] and s["embed_hooks"])
        self.assertGreater(s["n_gc_modules"], 0)
        self.assertEqual(s["gc_kwargs"], [(("use_reentrant", False),)])
        for pm in (self.pm_v, self.pm_r):  # both restore eval mode and switch checkpointing off afterwards
            st = _rp_gc_state(pm)
            self.assertEqual((st["training"], st["n_gc_modules"]), (False, 0))

    # ---------------------------------------------------------------- resume
    def test_resume_reproduces_uninterrupted_run(self):
        """A 6-step run (ckpt_every=2) killed after 4 completed steps, resumed by a fresh call on a freshly built
        model, equals the uninterrupted 6-step run bit for bit."""
        mps = RP_BATCH // RP_MICRO
        with tempfile.TemporaryDirectory() as td:
            pm1 = _RP_LC.attach_lora(_rp_model(5), seed=RP_SEED)
            with _RPRecorder(_RP_LC) as r1, self.assertRaises(_RPGuard.Stop):
                _RP_LC.train_sft(pm1, self.tok, self.rows, steps=RP_STEPS, batch_size=RP_BATCH, lr=RP_LR,
                                 seed=RP_SEED, row_order=self.order, micro=RP_MICRO,
                                 guard=_RPGuard(pm1, raise_on_call=4 * mps + 1), logger=lambda m: None, ckpt_dir=td,
                                 ckpt_every=2)
            self.assertEqual(torch.load(Path(td) / "state.pt", map_location="cpu", weights_only=False)["step"], 4)
            pm2, logs = _RP_LC.attach_lora(_rp_model(5), seed=RP_SEED), []
            with _RPRecorder(_RP_LC) as r2:
                out2 = _RP_LC.train_sft(pm2, self.tok, self.rows, steps=RP_STEPS, batch_size=RP_BATCH, lr=RP_LR,
                                        seed=RP_SEED, row_order=self.order, micro=RP_MICRO, guard=_RPGuard(pm2),
                                        logger=logs.append, ckpt_dir=td, ckpt_every=2)
        self.assertIn("resumed from step 4", logs)
        self.assertEqual(r1.micro, self.rec_r.micro[:4 * mps])
        self.assertEqual(r2.micro, self.rec_r.micro[4 * mps:])
        for k in ("final_loss_avg50", "loss_curve_every50", "max_grad_norm", "tokens_processed", "supervised_tokens",
                  "steps", "rows_available"):
            self.assertEqual(out2[k], self.out_r[k], k)
        fin = _rp_lora(pm2)
        self.assertEqual(sorted(fin), sorted(self.final_r))
        for n in fin:
            self.assertTrue(torch.equal(fin[n], self.final_r[n]), n)

    def test_resume_refuses_a_different_runs_checkpoint(self):
        """A checkpoint left by another run (rebuilt rows file, other steps/lr/seed/micro/row order) must not be
        spliced into this one (before the fix train_sft resumed from it silently)."""
        mps = RP_BATCH // RP_MICRO
        kw = dict(steps=RP_STEPS, batch_size=RP_BATCH, lr=RP_LR, seed=RP_SEED, row_order=self.order, micro=RP_MICRO,
                  logger=lambda m: None, ckpt_every=2)
        with tempfile.TemporaryDirectory() as td:
            pm = _RP_LC.attach_lora(_rp_model(5), seed=RP_SEED)
            with self.assertRaises(_RPGuard.Stop):
                _RP_LC.train_sft(pm, self.tok, self.rows, ckpt_dir=td, guard=_RPGuard(pm, raise_on_call=2 * mps + 1),
                                 **kw)
            rows2 = [dict(r) for r in self.rows]
            rows2[0]["target"] = '{"A":1}'
            variants = {"rows": (rows2, {}), "steps": (self.rows, {"steps": RP_STEPS + 2}),
                        "lr": (self.rows, {"lr": RP_LR / 2}), "seed": (self.rows, {"seed": RP_SEED + 1}),
                        "micro": (self.rows, {"micro": RP_MICRO // 2}),
                        "row_order": (self.rows, {"row_order": self.order[1:] + self.order[:1]})}
            for name, (rows, over) in variants.items():
                pm = _RP_LC.attach_lora(_rp_model(5), seed=RP_SEED)
                with self.assertRaisesRegex(ValueError, "different run", msg=name):
                    _RP_LC.train_sft(pm, self.tok, rows, ckpt_dir=td, **dict(kw, **over))


class TestNativeE01Recipe(unittest.TestCase):
    """native_e01 trains every arm with T2's recipe: 1,501 rows, 190 x 16, train_generic's row order, attach_lora and
    train_sft seeded with the run seed, lr/max_len/micro/grad_clip left at the recipe defaults (checked statically)."""

    def test_constants_and_train_call(self):
        self.assertEqual((_RP_NE.N_ROWS, _RP_NE.STEPS, _RP_NE.BATCH), (1501, 190, 16))
        tree = ast.parse(_rp_inspect.getsource(_RP_NE.cmd_train))
        calls = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id == "LC":
                calls.setdefault(n.func.attr, []).append(n)
        self.assertEqual({k: len(v) for k, v in calls.items() if k in ("train_sft", "row_order_for", "attach_lora")},
                         {"train_sft": 1, "row_order_for": 1, "attach_lora": 1})
        kw = {k.arg: ast.unparse(k.value) for k in calls["train_sft"][0].keywords}
        self.assertEqual((kw["steps"], kw["batch_size"], kw["seed"], kw["row_order"]),
                         ("STEPS", "BATCH", "a.seed", "order"))
        for k in ("lr", "max_len", "micro", "grad_clip"):
            self.assertNotIn(k, kw)
        self.assertEqual([ast.unparse(x) for x in calls["row_order_for"][0].args],
                         ["len(rows)", "STEPS", "BATCH", "a.seed"])
        self.assertEqual({k.arg: ast.unparse(k.value) for k in calls["attach_lora"][0].keywords}, {"seed": "a.seed"})


# ------------------------------------------------------------------------------------- SEALED analysis (PREREG_E01N)
import analyze_native as AN  # noqa: E402
from rcx import stats as ST  # noqa: E402

_SEALED_NTAGS = ("N-VR_s0", "N-VR_s1", "N-VR_s2", "N-CFSD_s0", "N-CFSD_s1", "N-CFSD_s2")


def _sealed_out_text(r, mode):
    """A raw model output for battery record r: 'exact' (the gold JSON), 'wrong' (an off-by-1000 balance, never a
    copy of the state) or 'format' (not JSON)."""
    if mode == "format":
        return "not json"
    after = dict(r["gold"]["after"])
    if mode == "wrong":
        k = next(iter(after))
        after[k] += 1000
    return json.dumps(after, separators=(",", ":"))


def _sealed_preds(battery, mode):
    from rcx.rcl import score as S
    out = []
    for i, r in enumerate(battery):
        o = _sealed_out_text(r, mode(r))
        out.append({"i": i, "out": o, **S.classify(r, NP.parse_answer(o, list(r["gold"]["after"].keys())))})
    return out


def _write_sealed_cell(d, tag, battery, bseed, preds, bsha=None):
    import gzip
    arm, seed = AN.split_tag(tag)
    with gzip.open(Path(d) / f"preds_sealed_{tag}.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    ad = Path(d) / tag / "adapter" / "adapter_model.safetensors"
    rec = {"arm": arm, "seed": 0 if seed is None else seed, "battery": "sealed", "battery_seed": bseed,
           "battery_sha256": bsha or AN.battery_sha256(battery),
           "adapter_sha256": NE.sha(ad) if seed is not None else None, "n": len(battery), "surfaces": "in,ho_tpl"}
    (Path(d) / f"eval_sealed_{tag}.json").write_text(json.dumps(rec), encoding="utf-8", newline="\n")


def _pid(r):
    return int(r["parent_id"].rsplit("-", 1)[1])


def _exact_all(r):
    return "exact"


def _noop_wrong_on_ho(cond=lambda r: True):
    """Wording-bound behaviour: on held-out paraphrases (ho_tpl) the no-op items (where cond holds) are applied."""
    return lambda r: "wrong" if r["surface"] == "ho_tpl" and r["gold"]["decision"] == "noop" and cond(r) else "exact"


class TestSealedAnalysis(unittest.TestCase):
    """analyze_native --battery sealed on a synthetic SEALED battery (temp dir; the real mint-commit / open-sealed with
    fake adapters; 4 parents per stratum) and synthetic predictions. No model load, never the real cells dir."""
    N_PARENTS = 4

    @classmethod
    def setUpClass(cls):
        import contextlib
        import io
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.d = Path(cls.tmp.name) / "cells"
        cls.d.mkdir()
        gen.use_world("arith")
        write_cells(cls.d, {"N-VR": [native_row(r) for r in gen.make_train("VR", 4, NE.DATA_SEED["VR"])],
                            "N-CFSD": [native_row(r) for r in gen.make_train("CF_SD", 4, NE.DATA_SEED["CF_SD"])]})
        man = json.loads((cls.d / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
        for tag in _SEALED_NTAGS:                                   # fake completed training runs
            arm, seed = AN.split_tag(tag)
            ad = cls.d / tag / "adapter"
            ad.mkdir(parents=True)
            (ad / "adapter_model.safetensors").write_bytes(b"fake adapter " + tag.encode())
            meta = {"arm": arm, "seed": seed, "rows_sha256": man[arm]["sha256"], "train": {"steps": NE.STEPS},
                    "adapter_sha256": NE.sha(ad / "adapter_model.safetensors")}
            (cls.d / tag / "TRAIN_META.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")
        ns = Namespace(out=str(cls.d), arms_required=",".join(_SEALED_NTAGS), n_parents=cls.N_PARENTS,
                       surfaces="in,ho_tpl", battery="sealed", arm="OFF", seed=0, expect_commit_sha="")
        saved = (NP.need_ram, LC.load_core, LC.generate)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                NE.cmd_mint_commit(ns)
                ns.expect_commit_sha = NE.sha(cls.d / "SEALED_COMMIT.json")
                NE.cmd_open_sealed(ns)
                NP.need_ram = lambda: 99.0                          # OFF through the real eval path, fake decoder
                LC.load_core = lambda name="sparse": (object(), LiteralTok())
                LC.generate = lambda model, tok, prompts, **kw: ["{}"] * len(prompts)
                NE.cmd_eval(ns)
        finally:
            NP.need_ram, LC.load_core, LC.generate = saved
        cls.battery, cls.bseed = NE.sealed_battery(cls.d, ["in", "ho_tpl"])
        for t in _SEALED_NTAGS + ("T2",):
            P = _sealed_preds(cls.battery, _noop_wrong_on_ho() if t.startswith("N-VR") else _exact_all)
            _write_sealed_cell(cls.d, t, cls.battery, cls.bseed, P)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy_cells(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        dst = Path(tmp.name) / "cells"
        shutil.copytree(self.d, dst)
        return dst

    def arms(self, vr, cf, off=_exact_all, t2=_exact_all):
        """In-memory SEALED cells: vr / cf are 3 per-seed behaviours (record -> 'exact' | 'wrong' | 'format')."""
        a = {f"N-VR_s{s}": _sealed_preds(self.battery, vr[s]) for s in range(3)}
        a.update({f"N-CFSD_s{s}": _sealed_preds(self.battery, cf[s]) for s in range(3)})
        a.update(OFF=_sealed_preds(self.battery, off), T2=_sealed_preds(self.battery, t2))
        return a

    def run_decision(self, arms, revise_rule=None):
        conf = AN.confirmatory(self.battery, arms, resamples=1000)
        prot = AN.protected(self.battery, arms, resamples=1000)
        return conf, prot, AN.decide(conf, prot["ok"], revise_rule)

    # ---- inputs: battery sha, adapters, re-scoring, missing cells, create-only outputs
    def test_load_checks_battery_sha_against_eval_records(self):
        battery, arms, meta = AN.load_sealed(self.d, ne=NE)
        self.assertEqual(sorted(arms), sorted(_SEALED_NTAGS + ("OFF", "T2")))
        self.assertEqual((meta["missing"], meta["warnings"], meta["dropped_train_overlap_parents"]), ([], [], []))
        off = json.loads((self.d / "eval_sealed_OFF.json").read_text(encoding="utf-8"))   # written by cmd_eval
        self.assertEqual(meta["battery_sha256"], off["battery_sha256"])
        self.assertEqual(meta["battery_sha256"], NE.battery_sha256(battery))
        self.assertEqual(len(battery), len(self.battery))
        d = self.copy_cells()
        p = d / "eval_sealed_T2.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(dict(rec, battery_sha256="0" * 64)), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(SystemExit, "T2: battery sha"):    # T2 scored on another battery
            AN.load_sealed(d, ne=NE)
        p.write_text(json.dumps(dict(rec, battery="dev")), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(SystemExit, "not 'sealed'"):
            AN.load_sealed(d, ne=NE)

    def test_load_refuses_tampered_adapters_scores_and_missing_cells(self):
        d = self.copy_cells()
        (d / "N-CFSD_s1" / "adapter" / "adapter_model.safetensors").write_bytes(b"swapped after opening")
        with self.assertRaisesRegex(SystemExit, "N-CFSD_s1: adapter sha"):
            AN.load_sealed(d, ne=NE)
        d = self.copy_cells()
        P = _sealed_preds(self.battery, _exact_all)
        P[3] = dict(P[3], exact=1 - P[3]["exact"])                  # a stored score the raw output does not give
        _write_sealed_cell(d, "N-VR_s2", self.battery, self.bseed, P)
        with self.assertRaisesRegex(SystemExit, "re-score"):
            AN.load_sealed(d, ne=NE)
        d = self.copy_cells()
        (d / "preds_sealed_N-VR_s2.jsonl.gz").unlink()
        with self.assertRaisesRegex(SystemExit, "missing"):
            AN.load_sealed(d, ne=NE)
        battery, arms, meta = AN.load_sealed(d, allow_missing=True, ne=NE)
        self.assertEqual(meta["missing"], ["N-VR_s2"])
        conf = AN.confirmatory(battery, arms, resamples=200)
        self.assertEqual((conf["C1"]["n_seeds_a"], conf["C1"]["n_seeds_b"]), (2, 3))

    def test_main_sealed_is_create_only_and_pins_the_resamples(self):
        d = self.copy_cells()
        out = d.parent / "ANALYSIS_SEALED.md"
        a = Namespace(cells=str(d), out=None, json_out=None, resamples=1000, allow_missing=False)
        with self.assertRaisesRegex(SystemExit, "10000"):
            AN.main_sealed(a)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            AN.main_sealed(Namespace(**dict(vars(a), resamples=10000)))         # the preregistered analysis
        res = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual((res["decision"], res["parameters"]["resamples"], res["parameters"]["prereg_resamples"]),
                         ("ADVANCE", 10000, True))
        self.assertNotIn(b"\r", out.read_bytes())
        self.assertNotIn("NOT THE PREREGISTERED", out.read_text(encoding="utf-8"))
        before = out.read_bytes()
        with self.assertRaisesRegex(SystemExit, "create-only"):                  # analysed once: never overwritten
            AN.main_sealed(Namespace(**dict(vars(a), resamples=10000)))
        self.assertEqual(out.read_bytes(), before)

    def test_end_to_end_advance_report(self):
        battery, arms, meta = AN.load_sealed(self.d, ne=NE)
        res, md = AN.analyze_sealed(battery, arms, meta, resamples=1000)
        self.assertEqual(res["decision"], "ADVANCE")
        c2 = res["confirmatory"]["C2"]
        self.assertAlmostEqual(c2["diff"], 0.5)
        self.assertEqual((c2["p"], c2["p_holm"]), (AN.P_FLOOR, 2 * AN.P_FLOOR))
        self.assertTrue(res["protected"]["ok"])
        self.assertIn("**DECISION: ADVANCE**", md)
        self.assertIn("NOT THE PREREGISTERED ANALYSIS", md)        # 1000 resamples here
        self.assertNotIn("\r", md)
        json.loads(json.dumps(res))
        self.assertEqual(set(res["secondary"]["arms"]), {"OFF", "T2", "N-VR", "N-CFSD"})
        self.assertEqual(sorted(c2["per_seed"]), ["s0", "s1", "s2"])
        for v in c2["per_seed"].values():
            self.assertAlmostEqual(v, 0.5)
        self.assertEqual(res["secondary"]["matched_seed_differences"]["s1"]["ho_tpl"], c2["per_seed"]["s1"])
        ti = res["secondary"]["seed_t_interval"]["ho_tpl"]
        self.assertEqual((ti["k"], ti["df"], ti["sd"]), (3, 2, 0.0))
        sens = res["secondary"]["seed_sensitivity"]
        self.assertEqual((sens["seeds"], sens["endpoints"]["C2"]["control"], sens["endpoints"]["C2"]["treatment"]),
                         ([1, 2], ["N-VR_s1", "N-VR_s2"], ["N-CFSD_s1", "N-CFSD_s2"]))
        self.assertAlmostEqual(sens["endpoints"]["C2"]["diff"], 0.5)
        self.assertEqual([sens["endpoints"][e]["point_ge_practical_and_ci_lo_gt_0"] for e in ("C1", "C2")],
                         [False, True])
        self.assertIn("Seed-0 sensitivity", md)
        self.assertEqual(res["parameters"]["decision_order"], ["ADVANCE", "STOP", "REVISE"])
        self.assertEqual(res["secondary"]["cells"]["OFF"]["in"]["errors"]["format"],
                         sum(r["surface"] == "in" for r in battery))   # cmd_eval's fake '{}' outputs are malformed

    # ---- the decision rule
    def test_decision_rule_constructed_cases(self):
        vr = [_noop_wrong_on_ho()] * 3
        odd, even = (lambda r: _pid(r) % 2 == 1), (lambda r: _pid(r) % 2 == 0)
        cases = {
            "ADVANCE": (vr, [_exact_all] * 3),                                   # C2 +0.50 on every parent
            "REVISE": (vr, [_noop_wrong_on_ho(odd), vr[0], vr[0]]),               # C2 +1/12: significant, < +0.10
            "STOP": (vr, vr),                                                     # identical arms: both CIs [0, 0]
            "INCONCLUSIVE": ([_noop_wrong_on_ho(odd)] * 3, [_noop_wrong_on_ho(even)] * 3),   # C2 0 +- 0.5
        }
        for want, (a, b) in cases.items():
            conf, prot, dec = self.run_decision(self.arms(a, b))
            self.assertEqual(dec["decision"], want, {k: (v["diff"], v["lo"], v["hi"], v["p_holm"])
                                                     for k, v in conf.items()})
            if want == "REVISE":
                self.assertAlmostEqual(conf["C2"]["diff"], 1 / 12)
                self.assertTrue(dec["per_endpoint"]["C2"]["holm_p_lt_alpha"])

        def f0_lost(r):    # posted twins of F strata outside the composite (F1, F4) become wrong: C1/C2 unchanged
            return ("wrong" if r["group"] in ("F1_void", "F4_nearmiss") and r["variant"] == "twin_posted"
                    and r["surface"] == "in" else "exact")

        def fmt_up(r):     # malformed outputs on M1/M2 (outside the composite)
            return "format" if r["group"] in ("M1_bal", "M2_amt") else "exact"

        for bad, which in ((f0_lost, "F0_twin_posted"), (fmt_up, "format_error_rate")):
            conf, prot, dec = self.run_decision(self.arms(vr, [bad] * 3))
            self.assertFalse(prot[which]["ok"], which)
            self.assertTrue(dec["per_endpoint"]["C2"]["meets_advance_endpoint"])
            self.assertEqual(dec["decision"], "REVISE")                         # +0.10 reached, a guard failed
            self.assertEqual(AN.decide(conf, prot["ok"], "literal")["decision"], "INCONCLUSIVE")   # pre-review text
        # one seed pair without an effect: C2 is still >= +0.10 overall, but ADVANCE needs every seed pair > 0
        conf, prot, dec = self.run_decision(self.arms(vr, [_exact_all, _exact_all, vr[0]]))
        self.assertEqual(conf["C2"]["per_seed"]["s2"], 0.0)
        self.assertTrue(dec["per_endpoint"]["C2"]["meets_advance_triple"])
        self.assertFalse(dec["per_endpoint"]["C2"]["all_seeds_gt_0"])
        self.assertEqual(dec["decision"], "REVISE")
        self.assertEqual(AN.decide(conf, prot["ok"], per_seed_rule=False)["decision"], "ADVANCE")

    def test_missing_cells_confirmatory_forces_inconclusive_reference_does_not(self):
        d = self.copy_cells()
        (d / "preds_sealed_N-VR_s2.jsonl.gz").unlink()
        battery, arms, meta = AN.load_sealed(d, allow_missing=True, ne=NE)
        self.assertEqual((meta["missing"], meta["missing_confirmatory"]), (["N-VR_s2"], ["N-VR_s2"]))
        res, md = AN.analyze_sealed(battery, arms, meta, resamples=1000)
        self.assertEqual(res["decision"], "INCONCLUSIVE")          # C2 is +0.50 on the remaining seeds
        self.assertFalse(res["decision_trace"]["complete"])
        self.assertTrue(res["decision_trace"]["per_endpoint"]["C2"]["meets_advance_endpoint"])
        c2 = res["confirmatory"]["C2"]
        self.assertEqual((c2["n_seeds_a"], c2["n_seeds_b"], sorted(c2["per_seed"])), (2, 2, ["s0", "s1"]))
        self.assertIn("A confirmatory cell is missing", md)
        d = self.copy_cells()
        (d / "eval_sealed_T2.json").unlink()
        with self.assertRaisesRegex(SystemExit, "missing"):
            AN.load_sealed(d, ne=NE)
        battery, arms, meta = AN.load_sealed(d, allow_missing=True, ne=NE)
        self.assertEqual((meta["missing"], meta["missing_confirmatory"]), (["T2"], []))
        res, md = AN.analyze_sealed(battery, arms, meta, resamples=1000)
        self.assertEqual(res["decision"], "ADVANCE")
        self.assertNotIn(f"{AN.TREATMENT} - T2", res["secondary"]["reference_contrasts"])
        self.assertIn("Only reference cells are missing", md)

    def test_load_refuses_unbound_adapters_and_other_commitments(self):
        d = self.copy_cells()
        p = d / "SEALED_OPEN.json"
        o = json.loads(p.read_text(encoding="utf-8"))
        del o["adapters_sha256"]["N-CFSD_s2"]
        p.write_text(json.dumps(o, indent=1), encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(SystemExit, "no adapter sha for confirmatory cells"):
            AN.load_sealed(d, ne=NE)
        d = self.copy_cells()
        p = d / "SEALED_COMMIT.json"
        c = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(dict(c, arms_required=c["arms_required"][:4]), indent=1), encoding="utf-8",
                     newline="\n")
        with self.assertRaisesRegex(SystemExit, "arms_required"):
            AN.load_sealed(d, ne=NE)

    def test_decide_edges_precedence_and_holm(self):
        def ep(diff, lo, hi, p, per_seed=None):
            return {"diff": diff, "lo": lo, "hi": hi, "p_holm": p,
                    "per_seed": {f"s{k}": diff for k in range(3)} if per_seed is None else per_seed}

        def D(c1, c2, prot=True, **kw):
            return AN.decide({"C1": c1, "C2": c2}, prot, **kw)

        self.assertEqual((AN.DECISION_ORDER, AN.REVISE_RULE, AN.P_TIES, AN.ADVANCE_PER_SEED),
                         (("ADVANCE", "STOP", "REVISE"), "not_advance", "both", True))
        null = ep(0.0, -0.01, 0.01, 1.0)
        self.assertEqual(D(ep(0.12, 0.05, 0.19, 0.001), ep(0.0, -0.04, 0.04, 0.9))["decision"], "ADVANCE")
        # a significant gain of >= +0.10 whose ADVANCE guards fail is REVISE (the pre-review text left it INCONCLUSIVE)
        c1_up_c2_down = (ep(0.12, 0.05, 0.19, 0.001), ep(-0.02, -0.06, 0.02, 0.3))   # other lower bound <= -0.05
        self.assertEqual(D(*c1_up_c2_down)["decision"], "REVISE")
        self.assertEqual(D(*c1_up_c2_down, revise_rule="literal")["decision"], "INCONCLUSIVE")
        self.assertEqual(D(null, ep(0.15, 0.03, 0.27, 0.001), prot=False)["decision"], "REVISE")
        # a significant drop on the other endpoint within the -0.05 margin does not block ADVANCE
        self.assertEqual(D(ep(-0.03, -0.045, -0.015, 0.001), ep(0.15, 0.03, 0.27, 0.001))["decision"], "ADVANCE")
        # float noise at the thresholds: 0.1 - 1e-12 reaches +0.10; p = 0.05 is not < 0.05; lo = 0 is not > 0
        self.assertEqual(D(null, ep(0.1 - 1e-12, 0.02, 0.18, 0.0002))["decision"], "ADVANCE")
        self.assertEqual(D(null, ep(0.15, 0.03, 0.27, 0.05))["decision"], "INCONCLUSIVE")
        self.assertEqual(D(null, ep(0.15, 0.0, 0.27, 0.01))["decision"], "REVISE")
        # the per-seed condition: one seed pair <= 0 blocks ADVANCE (REVISE); ignored when per_seed_rule=False
        weak = ep(0.15, 0.03, 0.27, 0.001, {"s0": 0.3, "s1": 0.16, "s2": 0.0})
        self.assertEqual(D(null, weak)["decision"], "REVISE")
        self.assertEqual(D(null, weak, per_seed_rule=False)["decision"], "ADVANCE")
        with self.assertRaises(ValueError):
            AN.decide({"C1": null, "C2": {"diff": 0.15, "lo": 0.03, "hi": 0.27, "p_holm": 0.001}}, True)
        # a missing confirmatory cell: INCONCLUSIVE whatever the estimates
        for c in ((null, ep(0.15, 0.03, 0.27, 0.001)), (null, null), (null, ep(0.04, 0.01, 0.07, 0.001))):
            self.assertEqual(D(*c, complete=False)["decision"], "INCONCLUSIVE")
        # REVISE and STOP can both hold (small significant effect): STOP is tested first
        dec = D(ep(0.02, 0.005, 0.035, 0.01), ep(0.01, -0.02, 0.04, 0.4))
        self.assertEqual((dec["decision"], dec["also_satisfied"]), ("STOP", ["REVISE"]))
        dec = D(ep(0.02, 0.005, 0.035, 0.01), ep(0.01, -0.02, 0.04, 0.4), order=("ADVANCE", "REVISE", "STOP"))
        self.assertEqual((dec["decision"], dec["also_satisfied"]), ("REVISE", ["STOP"]))
        with self.assertRaises(ValueError):
            D(null, null, order=("ADVANCE", "STOP"))
        self.assertEqual(D(null, ep(-0.08, -0.12, -0.04, 0.001))["decision"], "STOP")   # significant but negative
        self.assertEqual(D(null, ep(0.04, -0.01, 0.09, 0.2))["decision"], "INCONCLUSIVE")
        self.assertEqual(D(ep(-0.04, -0.07, -0.01, 0.001), ep(0.04, 0.01, 0.07, 0.001))["decision"], "REVISE")
        # Holm: raw p 0.03 / 0.04 would ADVANCE; Holm-adjusted 0.06 / 0.06 does not
        raw = {"C1": 0.03, "C2": 0.04}
        adj = ST.holm(raw)
        self.assertEqual(adj, {"C1": 0.06, "C2": 0.06})
        self.assertEqual(ST.holm({"C1": 0.2, "C2": 0.01}), {"C1": 0.2, "C2": 0.02})
        self.assertEqual(D(*(ep(0.12, 0.01, 0.23, raw[k]) for k in ("C1", "C2")))["decision"], "ADVANCE")
        self.assertEqual(D(*(ep(0.12, 0.01, 0.23, adj[k]) for k in ("C1", "C2")))["decision"], "INCONCLUSIVE")

    def test_confirmatory_applies_holm_and_matches_seed_parent_bootstrap(self):
        arms = self.arms([_noop_wrong_on_ho()] * 3, [_noop_wrong_on_ho(lambda r: _pid(r) % 2 == 1)] * 3)
        conf = AN.confirmatory(self.battery, arms, resamples=1000)
        self.assertEqual({k: v["p_holm"] for k, v in conf.items()}, ST.holm({k: v["p"] for k, v in conf.items()}))
        for k, s in AN.ENDPOINTS:
            pa = [AN.parents(self.battery, arms[f"N-VR_s{i}"], AN.COMPOSITE, s) for i in range(3)]
            pb = [AN.parents(self.battery, arms[f"N-CFSD_s{i}"], AN.COMPOSITE, s) for i in range(3)]
            ref = ST.seed_parent_bootstrap(pa, pb, resamples=1000, seed=AN.BOOT_SEED_SEALED)
            for f in ("mean_a", "mean_b", "diff", "lo", "hi", "n_parents"):
                self.assertEqual(conf[k][f], ref[f], (k, f))
            self.assertEqual(conf[k]["n_parents"], 7 * self.N_PARENTS)

    def test_p_value_floor_and_ties(self):
        p2 = ST.bootstrap_p_two_sided
        self.assertEqual(p2([0.2] * 10000, floor=AN.P_FLOOR), 1e-4)       # all > 0: 2 * min(0, 1) = 0 -> floor
        self.assertEqual(p2([-0.2] * 10000, floor=AN.P_FLOOR), 1e-4)      # two-sided
        self.assertEqual(p2([0.2] * 1000), 1e-3)                           # default floor 1 / resamples
        self.assertEqual(p2([-1.0] * 30 + [1.0] * 70), 0.6)
        self.assertAlmostEqual(p2([-1.0] * 3 + [0.0] * 2 + [1.0] * 95), 0.10)           # zeros on the <= 0 side
        self.assertAlmostEqual(p2([-1.0] * 3 + [0.0] * 2 + [1.0] * 95, ties="split"), 0.08)
        self.assertEqual(p2([0.0] * 100, floor=1e-4), 1e-4)                # literal rule: identically 0 -> floor
        self.assertEqual(p2([0.0] * 100, ties="split"), 1.0)
        self.assertEqual(p2([1e-17] * 50 + [1.0] * 50), 1.0)               # float noise counts as 0
        with self.assertRaises(ValueError):
            p2([])
        with self.assertRaises(ValueError):
            p2([0.1] * 100, ties="mid")
        # ties='both' (the analysis default): zeros count on both sides
        self.assertEqual(AN.P_TIES, "both")
        self.assertAlmostEqual(p2([-1.0] * 3 + [0.0] * 2 + [1.0] * 95, ties="both"), 0.10)
        self.assertEqual(p2([0.0] * 100, floor=1e-4, ties="both"), 1.0)    # identical arms: p = 1, not the floor
        self.assertEqual(p2([0.2] * 10000, floor=1e-4, ties="both"), 1e-4)
        self.assertEqual(p2([-1.0] * 30 + [1.0] * 70, ties="both"), 0.6)   # no ties: same as 'low'
        # the review case: every replicate <= 0, 6% exactly 0 (CI upper bound 0): 'low' calls it p = 1e-4
        tied = [-0.002] * 9400 + [0.0] * 600
        self.assertEqual(p2(tied, floor=1e-4), 1e-4)
        self.assertAlmostEqual(p2(tied, floor=1e-4, ties="both"), 0.12)
        # with 10,000 replicates, p('both') < 0.05 implies that the 95% percentile interval excludes 0
        for k in (249, 250, 251):
            for sign in (1.0, -1.0):
                boot = [-sign * 0.01] * (k - 1) + [0.0] + [sign * 0.01 * (1 + i / 10000) for i in range(10000 - k)]
                lo, hi = ST._interval(boot, 0.05)
                p = p2(boot, floor=1e-4, ties="both")
                self.assertEqual(p < 0.05, k <= 249, (k, sign, p))
                if p < 0.05:
                    self.assertTrue(lo > 0 if sign > 0 else hi < 0, (k, sign, lo, hi))
        a = [{"p": [0, 1], "q": 1, "r": [0], "s": [1, 1]}, {"p": [1, 1], "q": 0, "r": [1], "s": [0, 1]}]
        b = [{"p": [1, 1], "q": 1, "r": [1], "s": [1, 1]}] * 3
        t = ST.seed_parent_bootstrap_test(a, b, resamples=500, seed=7, p_floor=1e-4)
        ref = ST.seed_parent_bootstrap(a, b, resamples=500, seed=7)
        self.assertEqual({k: t[k] for k in ref}, ref)
        boot = ST._boot_means([0.25, 0.5, 0.5, 0.25], 500, 7)            # per-parent b - a for p, q, r, s
        self.assertEqual(t["p"], p2(boot, 1e-4))
        self.assertEqual((t["n_le0"], t["n_gt0"], t["n_ge0"], t["ties"]),
                         (sum(x <= 0 for x in boot), sum(x > 0 for x in boot), sum(x >= 0 for x in boot), "low"))
        t = ST.seed_parent_bootstrap_test(a, b, resamples=500, seed=7, p_floor=1e-4, ties="both")
        self.assertEqual(t["p"], p2(boot, 1e-4, ties="both"))


if __name__ == "__main__":
    unittest.main()
