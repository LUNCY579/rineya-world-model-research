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
                NE.cmd_eval(Namespace(out=d, arm="OFF", seed=0, n_parents=1, surfaces="in,ho_tpl"))
                rec = json.loads((Path(d) / "eval_OFF.json").read_text(encoding="utf-8"))
                bpath = Path(d) / "battery.jsonl"                 # written exactly as native_probe.main does
                with open(bpath, "w", encoding="utf-8", newline="\n") as f:
                    for r in NE.eval_battery(1, ["in", "ho_tpl"]):
                        f.write(json.dumps(r) + "\n")
                self.assertEqual(rec["battery_sha256"], NP.sha_file(bpath))
        finally:
            NP.need_ram, LC.load_core, LC.generate = saved



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


if __name__ == "__main__":
    unittest.main()
