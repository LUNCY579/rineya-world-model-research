"""CPU test of scripts/ret_run.py (retention runner) and its hand-off to scripts/ret_analysis.py, with tiny random
Qwen3 stand-ins. No real model is loaded (AutoModelForCausalLM.from_pretrained is disabled), no GPU is touched, no lease
file is read or written (a fake rcwmt.runtime supplies the lease, loads and pacing); outputs go to a temporary directory.
Threads 2. One process.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary (151,936) and the real
tokenizer (read-only); "sparse" and "supa" from different seeds; three LoRA adapters (r 4, the T2 target modules, A and
B drawn non-zero, different seeds) saved with PEFT: a T2-s0 stand-in and an extra adapter on "sparse", one extra on
"supa". The runner's own RealBackend (lease call, PEFT loading, fingerprint and runtime checks, pacing hooks) runs on
the fake runtime. Deviations from the GPU path, all pacing-only: breathe / throttle are no-ops, device 'cpu', decode
cap 8 tokens. Draws are minted by ret_gen.mint in a test root (3 items per cell, scan skipped, no freeze).

Checks
  1  dev mode: Supa only, no log-probs, one lease 'rcwm-t1:retention', resource wait before the load; every draw
     prompt answered once; a rerun is a no-op; ret_analysis.dev_select reads the run
  2  confirm mode: SPARSE (off, t2s0, x_bwm) in ONE process then SUPA (supa, xs_sx) after unloading; one lease; lp
     arrays (n, 151936) float16, clamped at -60, rows normalised; prefill first token == generated first token;
     adapters active and distinct; each side equals an independent load (bare Titan-Sparse for 'off', the t2s0
     adapter alone, bare Supa, the Supa adapter alone); provenance records the pinned tokenizer files and the
     chat-template hash of each load; the analysis accepts the run (all comparisons, both graders, KL available);
     a rerun is a no-op
  3  crash in SPARSE after one side: lease released, nothing but .partial files; the resume orphans them, completes,
     and every stage file equals the uninterrupted run byte for byte
  4  turn cap: nothing fits -> exit 5, no load; SPARSE fits but SUPA does not -> exit 5 with SPARSE recorded; the next
     turn completes and equals the uninterrupted run
  5  deterministic stops: a recorded output changed (persisted; the rerun refuses on STOPPED.json); fingerprint
     mismatch and runtime drift (persisted, nothing generated); one-waiter rule (live RCWM marker / live RCWM holder
     refuse without a lease, not persisted; a dead marker, a reused pid and a live BWM holder proceed); a live run lock
     refuses, a stale one is reclaimed
  6  pre-lease refusals: tampered draw, purpose/mode mismatch, unscanned draw, T2 s0 pin mismatch, an adapter trained
     on another base (unless --allow-base-mismatch), a registered run in an unfrozen root, a tokenizer directory whose
     tokenizer_config.json (chat template) differs from the pin; after the load: a chat template that differs from the
     pin stops the run (persisted, nothing generated)

  python -B scripts/ret_test_tiny.py [--keep] [--no-record]
Writes docs/retention/TINY_TEST.json (unless --no-record; refused once the kit is frozen).
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # NOT "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.synchronize = lambda *a, **k: None
torch.set_num_threads(2)

import gc  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import transformers  # noqa: E402
from transformers import AutoTokenizer, LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

import ret_common as RC  # noqa: E402
import ret_gen as G  # noqa: E402
import ret_run as R  # noqa: E402

_TORCH = sys.modules.pop("torch")                   # ret_analysis refuses to import next to torch (it never uses it)
try:
    import ret_analysis as RA  # noqa: E402
finally:
    sys.modules["torch"] = _TORCH


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models

KEEP = "--keep" in sys.argv
RECORD = "--no-record" not in sys.argv
MAXNEW = 8
CFG = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
           tie_word_embeddings=True)
SEEDS = {"sparse": 2002, "supa": 1001}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOKENIZER_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(f"{name}: {detail}")


@torch.no_grad()
def build(arm: str):
    cfg = Qwen3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(SEEDS[arm])
    m = Qwen3ForCausalLM(cfg)
    g = torch.Generator().manual_seed(SEEDS[arm])
    for _, p in m.named_parameters():
        if p.dim() == 2:
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


@torch.no_grad()
def make_adapter(path: Path, base: str, seed: int, base_name: str) -> None:
    pm = get_peft_model(build(base), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS,
                                                task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(seed)
    for n, p in pm.named_parameters():
        if "lora_B" in n or "lora_A" in n:
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    pm.save_pretrained(str(path))
    cfgp = path / "adapter_config.json"
    c = json.loads(cfgp.read_text(encoding="utf-8"))
    c["base_model_name_or_path"] = f"C:/fake/{base_name}"
    cfgp.write_text(json.dumps(c, indent=2), encoding="utf-8")
    del pm
    gc.collect()


_TOK = []


def tok():
    if not _TOK:
        _TOK.append(AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True))
    return _TOK[0]


class FakeRuntime:
    """Stands in for rcwmt.runtime under RealBackend: tiny models, recorded calls, a lock file in the temp dir only."""

    def __init__(self, tmp: Path):
        self.LOCK_PATH = tmp / "fake_gpu.lock"
        self.WAITING = {"rcwm": tmp / "fake_gpu.waiting.rcwm.json", "bwm": tmp / "fake_gpu.waiting.bwm.json"}
        self.TOKENIZER_DIR = TOKENIZER_DIR              # the real, pinned tokenizer (read-only)
        self.calls: list = []
        self.fp_override: dict = {}
        self.runtime_override: list = []
        self.on_unload = None
        self.template_suffix = None

    @contextmanager
    def gpu_lease(self, label, job):
        self.calls.append(("lease", label))
        self.LOCK_PATH.write_text(json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}), encoding="utf-8")
        try:
            yield
        finally:
            self.LOCK_PATH.unlink()
            self.calls.append(("release", label))

    def wait_for_resources(self):
        self.calls.append(("wait",))

    def load_arm(self, arm):
        self.calls.append(("load", arm))
        return build(arm), {"param_fingerprint": self.fp_override.get(arm, R.FP[arm] + "-tiny-standin")}

    def tokenizer(self):
        if self.template_suffix:                        # a fresh tokenizer whose chat template differs from the pin
            t = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
            t.chat_template = t.chat_template + self.template_suffix
            return t
        return tok()

    def unload(self, m):
        self.calls.append(("unload",))
        gc.collect()
        if self.on_unload:
            self.on_unload()

    def runtime_info(self):
        if self.runtime_override:
            return self.runtime_override.pop(0)
        return {"torch": torch.__version__, "transformers": transformers.__version__}

    def breathe(self):
        pass

    def throttle_list(self):
        return LogitsProcessorList([])


def backend(fake):
    b = R.RealBackend(rt=fake)
    b.device = "cpu"
    return b


class Stub:
    device = "cpu"

    def pace(self):
        pass

    def processors(self):
        return LogitsProcessorList([])


def stage_files(out: Path) -> dict:
    return {str(p.relative_to(out)).replace("\\", "/"): RC.sha256_file(p)
            for d in ("sparse", "supa") if (out / d).exists() for p in sorted((out / d).iterdir())}


def main() -> int:
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="ret_tiny_"))
    root = tmp / "root"
    (root / "docs" / "retention" / "draws").mkdir(parents=True)
    fake = FakeRuntime(tmp)
    try:
        ad = {"t2": tmp / "ad_t2s0", "x": tmp / "ad_x", "sx": tmp / "ad_sx", "wrongbase": tmp / "ad_wrongbase"}
        make_adapter(ad["t2"], "sparse", 11, R.CORE_DIRNAME["sparse"])
        make_adapter(ad["x"], "sparse", 12, R.CORE_DIRNAME["sparse"])
        make_adapter(ad["sx"], "supa", 13, R.CORE_DIRNAME["supa"])
        make_adapter(ad["wrongbase"], "sparse", 14, "Some-Other-Core")
        pins = {f: RC.sha256_file(ad["t2"] / f) for f in ("adapter_model.safetensors", "adapter_config.json")}
        dseal = G.mint("dev", seed=31, scan=False, root=root, require_freeze=False, n=3)
        ddraw = root / dseal["file"]

        def cfg(draw, mode, out, **kw):
            base = dict(draw=draw, mode=mode, root=root, out=out, t2s0_dir=ad["t2"], s0_pins=pins, require_freeze=False,
                        allow_unscanned=True, max_new=MAXNEW)
            if mode == "confirm":
                base.update(extra={"bwm": ad["x"]}, extra_supa={"sx": ad["sx"]})
            base.update(kw)
            return R.Cfg(**base)

        # ------------------------------------------------------------------ 1 dev mode
        print("[1 dev mode]", flush=True)
        out_dev = tmp / "runs" / "dev"
        fake.calls.clear()
        rc = R.run(cfg(ddraw, "dev", out_dev), backend(fake))
        drows, _ = G.load_draw(ddraw)
        gens = RC.read_jsonl(out_dev / "supa" / "gen_supa.jsonl")
        check("dev run: exit 0, DONE.json, every prompt answered once, no log-probs",
              rc == 0 and (out_dev / "DONE.json").exists() and [g["pid"] for g in gens] == [r["pid"] for r in drows]
              and not list((out_dev / "supa").glob("lp_*")), (rc, len(gens), len(drows)))
        check("dev run: one lease 'rcwm-t1:retention', wait before the only load (supa)",
              fake.calls == [("lease", "rcwm-t1:retention"), ("wait",), ("load", "supa"), ("unload",),
                             ("release", "rcwm-t1:retention")], fake.calls)
        fake.calls.clear()
        check("dev rerun is a no-op", R.run(cfg(ddraw, "dev", out_dev), backend(fake)) == 0 and fake.calls == [])
        # the analysis reads the dev run from the test root's draws dir (the run dir layout is what matters)
        run_link = root / "runs" / "retention" / dseal["draw_id"]
        shutil.copytree(out_dev, run_link)
        sel = RA.dev_select(run_link, root, registered=False, write=False)
        check("dev_select reads the tiny dev run (24 cells graded)", sum(len(v) for v in sel["accuracy"].values()) == 24,
              {"selected": sel["selected"], "excluded": sel["excluded"]})
        (root / "docs" / "retention" / "DEV_SELECTION.json").write_text(json.dumps(
            {"selected": {c: 1 for c in RC.CATEGORIES}, "excluded": [], "surface_diversity": False}), encoding="utf-8")
        cseal = G.mint("confirm", seed=32, scan=False, root=root, require_freeze=False, n=3)
        cdraw = root / cseal["file"]
        crows, _ = G.load_draw(cdraw)

        # ------------------------------------------------------------------ 2 confirm mode
        print("[2 confirm mode]", flush=True)
        out = tmp / "runs" / "confirm_full"
        fake.calls.clear()
        rc = R.run(cfg(cdraw, "confirm", out), backend(fake))
        check("confirm run: exit 0 and DONE.json", rc == 0 and (out / "DONE.json").exists(), rc)
        check("one lease; SPARSE then SUPA; resource wait before each load; Sparse unloaded before Supa",
              fake.calls == [("lease", "rcwm-t1:retention"), ("wait",), ("load", "sparse"), ("unload",), ("wait",),
                             ("load", "supa"), ("unload",), ("release", "rcwm-t1:retention")], fake.calls)
        prov = RC.read_json(out / "provenance.json")
        check("sides recorded: SPARSE off/t2s0/x_bwm, SUPA supa/xs_sx",
              prov["sides"] == {"SPARSE": ["off", "t2s0", "x_bwm"], "SUPA": ["supa", "xs_sx"]}, prov["sides"])
        t0 = prov["turns"][0]
        check("provenance: pinned tokenizer files, and the chat-template hash of each load equals the pin",
              prov["tokenizer"]["files"] == RC.TOKENIZER_PINS
              and t0.get("sparse_chat_template_sha256") == RC.CHAT_TEMPLATE_SHA
              and t0.get("supa_chat_template_sha256") == RC.CHAT_TEMPLATE_SHA,
              (prov["tokenizer"], t0.get("sparse_chat_template_sha256")))
        lp = np.load(out / "sparse" / "lp_t2s0.npy")
        rowsum = np.exp(lp.astype(np.float64)).sum(1)
        check("lp arrays: (n, 151936) float16, within [-60, 0], rows normalised",
              lp.shape == (len(crows), 151936) and lp.dtype == np.float16 and float(lp.max()) <= 0.0
              and float(lp.min()) >= -60.0 and np.allclose(rowsum, 1.0, atol=2e-2), (lp.shape, float(rowsum.min())))
        agree = {s: v for st in prov["stages"].values() for s, v in (st["first_token_agreement"] or {}).items()}
        check("prefill first token == generated first token (every side)", all(v == 1.0 for v in agree.values()), agree)
        txt = {s: [g["text"] for g in RC.read_jsonl(out / R.STAGE_DIR[st] / f"gen_{s}.jsonl")]
               for st, ss in prov["sides"].items() for s in ss}
        check("adapters active and distinct: off != t2s0 != x_bwm; supa != xs_sx",
              txt["off"] != txt["t2s0"] and txt["t2s0"] != txt["x_bwm"] and txt["supa"] != txt["xs_sx"])
        stub = Stub()

        def indep(model):
            return [r["text"] for r in R.generate_side(model.eval(), tok(), crows, MAXNEW, stub, None)]

        bare_sparse = build("sparse")
        same_off = indep(bare_sparse) == txt["off"]
        only_t2 = PeftModel.from_pretrained(bare_sparse, str(ad["t2"]), adapter_name="t2s0", is_trainable=False)
        same_t2 = indep(only_t2) == txt["t2s0"]
        del only_t2, bare_sparse
        bare_supa = build("supa")
        same_supa = indep(bare_supa) == txt["supa"]
        only_sx = PeftModel.from_pretrained(bare_supa, str(ad["sx"]), adapter_name="xs_sx", is_trainable=False)
        same_sx = indep(only_sx) == txt["xs_sx"]
        del only_sx, bare_supa
        gc.collect()
        check("each side equals an independent load (disable_adapter == bare core; switching == loading alone)",
              same_off and same_t2 and same_supa and same_sx, (same_off, same_t2, same_supa, same_sx))
        run_c = root / "runs" / "retention" / cseal["draw_id"]
        shutil.copytree(out, run_c)
        res = RA.analyze_run(run_c, root, registered=False)
        check("analysis accepts the run: every comparison, both graders, KL available, verdicts from the rule; gating = "
              "t2s0 / x_bwm vs off and xs_sx vs supa only",
              set(res["comparisons"]) == {"sparse_vs_supa", "t2s0_vs_off", "x_bwm_vs_off", "xs_sx_vs_supa", "t2s0_vs_supa"}
              and all(set(c["graders"]) == {"primary", "lenient"} for c in res["comparisons"].values())
              and all(c["kl"]["available"] for c in res["comparisons"].values())
              and all(c["verdict"] in ("RETAINED", "LOSS DETECTED", "MARGIN EXCEEDED") for c in res["comparisons"].values())
              and set(res["go"]) == {"t2s0_vs_off", "x_bwm_vs_off", "xs_sx_vs_supa"},
              {n: c["verdict"] for n, c in res["comparisons"].items()})
        fake.calls.clear()
        check("confirm rerun is a no-op", R.run(cfg(cdraw, "confirm", out), backend(fake)) == 0 and fake.calls == [])
        ref_files = stage_files(out)

        # ------------------------------------------------------------------ 3 crash + resume
        print("[3 crash + resume]", flush=True)
        out3 = tmp / "runs" / "confirm_crash"
        real_gen = R.generate_side
        n_calls = [0]

        def crashing(*a, **k):
            n_calls[0] += 1
            if n_calls[0] == 2:
                raise RuntimeError("injected crash in the second SPARSE side")
            return real_gen(*a, **k)

        R.generate_side = crashing
        fake.calls.clear()
        try:
            R.run(cfg(cdraw, "confirm", out3), backend(fake))
            crashed = False
        except RuntimeError:
            crashed = True
        finally:
            R.generate_side = real_gen
        gc.collect()                                     # release the crashed side's memmap before the resume moves it
        partials = sorted(p.name for p in (out3 / "sparse").iterdir())
        check("crash: lease released; only .partial files; nothing recorded",
              crashed and fake.calls[-1] == ("release", "rcwm-t1:retention") and partials
              and all(p.endswith(".partial") for p in partials) and not RC.read_json(out3 / "provenance.json")["stages"],
              partials)
        rc = R.run(cfg(cdraw, "confirm", out3), backend(fake))
        check("resume: partial stage orphaned, run completes, stage files == uninterrupted run byte for byte",
              rc == 0 and (out3 / "_orphans").exists() and stage_files(out3) == ref_files, rc)

        # ------------------------------------------------------------------ 4 turn cap
        print("[4 turn cap]", flush=True)
        out4 = tmp / "runs" / "confirm_cap"
        fake.calls.clear()
        rc = R.run(cfg(cdraw, "confirm", out4, cap_h=0.0), backend(fake))
        check("cap 0: exit 5, nothing loaded, everything deferred",
              rc == 5 and not any(c[0] == "load" for c in fake.calls)
              and RC.read_json(out4 / "provenance.json")["turns"][-1]["deferred"] == ["SPARSE", "SUPA"], fake.calls)
        real_time = R.time.time
        offset = [0.0]
        R.time.time = lambda: real_time() + offset[0]

        def jump():
            offset[0] += 2000.0                         # SPARSE "took" 2,000 s more: SUPA no longer fits a 1 h cap
        fake.on_unload = jump
        try:
            rc = R.run(cfg(cdraw, "confirm", out4, cap_h=1.0), backend(fake))
        finally:
            R.time.time = real_time
            fake.on_unload = None
        st = RC.read_json(out4 / "provenance.json")
        check("cap 1 h with a slow SPARSE: exit 5, SPARSE recorded, SUPA deferred (D-22 margin)",
              rc == 5 and set(st["stages"]) == {"SPARSE"} and st["turns"][-1]["deferred"] == ["SUPA"], st["turns"][-1])
        rc = R.run(cfg(cdraw, "confirm", out4), backend(fake))
        check("next turn completes; stage files == uninterrupted run", rc == 0 and stage_files(out4) == ref_files, rc)

        # ------------------------------------------------------------------ 5 deterministic stops
        print("[5 deterministic stops]", flush=True)
        out5 = tmp / "runs" / "confirm_tamper"
        R.time.time = lambda: real_time() + offset[0]
        offset[0] = 0.0
        fake.on_unload = jump
        try:
            R.run(cfg(cdraw, "confirm", out5, cap_h=1.0), backend(fake))
        finally:
            R.time.time = real_time
            fake.on_unload = None
        (out5 / "sparse" / "gen_off.jsonl").write_text("tampered\n", encoding="utf-8")

        def main_like(c):
            try:
                return R.run(c, backend(fake)), None
            except RC.RetStop as e:
                R.persist_stop(c.out, e)
                return 3, e

        rc, e = main_like(cfg(cdraw, "confirm", out5))
        check("a recorded output changed: stop, persisted", rc == 3 and (out5 / "STOPPED.json").exists(), e)
        rc, e = main_like(cfg(cdraw, "confirm", out5))
        check("the rerun refuses on STOPPED.json", rc == 3 and "STOPPED.json" in str(e), e)
        out6 = tmp / "runs" / "confirm_fp"
        fake.fp_override = {"sparse": "deadbeef0000"}
        rc, e = main_like(cfg(cdraw, "confirm", out6))
        fake.fp_override = {}
        check("fingerprint mismatch: stop persisted, nothing generated",
              rc == 3 and (out6 / "STOPPED.json").exists() and not list(out6.glob("sparse/*")), e)
        out7 = tmp / "runs" / "confirm_rt"
        fake.runtime_override = [{"torch": "9.9.9", "transformers": transformers.__version__}]
        rc, e = main_like(cfg(cdraw, "confirm", out7))
        fake.runtime_override = []
        check("runtime drift: stop persisted, nothing generated",
              rc == 3 and (out7 / "STOPPED.json").exists() and not list(out7.glob("sparse/*")), e)
        ppid = os.getppid()
        pct = psutil.Process(ppid).create_time()
        cases = [("live RCWM waiting marker", fake.WAITING["rcwm"], {"pid": ppid, "label": "rcwm-t1:x", "since": time.time()}, False),
                 ("dead RCWM waiting marker", fake.WAITING["rcwm"], {"pid": 999999, "label": "rcwm-t1:x", "since": time.time()}, True),
                 ("reused pid in the marker", fake.WAITING["rcwm"], {"pid": ppid, "label": "rcwm-t1:x", "since": pct - 100}, True),
                 ("live RCWM lock holder", fake.LOCK_PATH, {"pid": ppid, "label": "rcwm-t1:other", "created": time.time()}, False),
                 ("live BWM lock holder", fake.LOCK_PATH, {"pid": ppid, "label": "bwm:bundle", "created": time.time()}, True)]
        for i, (name, path, rec, proceeds) in enumerate(cases):
            o = tmp / "runs" / f"waiter_{i}"
            path.write_text(json.dumps(rec), encoding="utf-8")
            fake.calls.clear()
            try:
                rc, e = main_like(cfg(cdraw, "confirm", o, cap_h=0.0))
            finally:
                if path.exists():
                    path.unlink()
            if proceeds:
                check(f"one-waiter: {name} -> proceeds (lease taken, deferred by cap 0)",
                      rc == 5 and fake.calls and fake.calls[0][0] == "lease", (rc, e))
            else:
                check(f"one-waiter: {name} -> refused, no lease, not persisted",
                      rc == 3 and not fake.calls and not (o / "STOPPED.json").exists(), (rc, str(e)[:80]))
        o = tmp / "runs" / "runlock"
        o.mkdir(parents=True)
        (o / ".run.lock").write_text(json.dumps({"pid": ppid, "create_time": pct}), encoding="utf-8")
        rc, e = main_like(cfg(cdraw, "confirm", o, cap_h=0.0))
        check("a live run lock refuses (not persisted)", rc == 3 and "run lock" in str(e) and not (o / "STOPPED.json").exists(), e)
        (o / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
        rc, e = main_like(cfg(cdraw, "confirm", o, cap_h=0.0))
        check("a stale run lock is reclaimed", rc == 5, (rc, e))

        # ------------------------------------------------------------------ 6 pre-lease refusals
        print("[6 pre-lease refusals]", flush=True)
        bad = tmp / "tampered"
        bad.mkdir()
        shutil.copy(cdraw, bad / cdraw.name)
        shutil.copy(cdraw.with_name(cseal["draw_id"] + ".SEAL.json"), bad / (cseal["draw_id"] + ".SEAL.json"))
        t = (bad / cdraw.name).read_text(encoding="utf-8")
        (bad / cdraw.name).write_text(t.replace('"value":', '"value" :', 1), encoding="utf-8")
        refusals = [("tampered draw", cfg(bad / cdraw.name, "confirm", tmp / "runs" / "r1")),
                    ("purpose/mode mismatch", cfg(ddraw, "confirm", tmp / "runs" / "r2")),
                    ("unscanned draw", cfg(cdraw, "confirm", tmp / "runs" / "r3", allow_unscanned=False)),
                    ("T2 s0 pin mismatch", cfg(cdraw, "confirm", tmp / "runs" / "r4", s0_pins={"adapter_model.safetensors": "0" * 64,
                                                                                               "adapter_config.json": "0" * 64})),
                    ("adapter trained on another base", cfg(cdraw, "confirm", tmp / "runs" / "r5", extra={"bad": ad["wrongbase"]})),
                    ("registered run in an unfrozen root", cfg(cdraw, "confirm", tmp / "runs" / "r6", require_freeze=True))]
        for name, c in refusals:
            fake.calls.clear()
            rc, e = main_like(c)
            check(f"refused before the lease: {name}", rc == 3 and not fake.calls and not (c.out / "STOPPED.json").exists(),
                  str(e)[:90])
        bad_tok = tmp / "tokenizer_other_template"
        bad_tok.mkdir()
        for f in RC.TOKENIZER_PINS:
            shutil.copy(TOKENIZER_DIR / f, bad_tok / f)
        cfgj = json.loads((bad_tok / "tokenizer_config.json").read_text(encoding="utf-8"))
        cfgj["chat_template"] = str(cfgj.get("chat_template", "")) + " "
        (bad_tok / "tokenizer_config.json").write_text(json.dumps(cfgj), encoding="utf-8")
        fake.TOKENIZER_DIR = bad_tok
        fake.calls.clear()
        c8 = cfg(cdraw, "confirm", tmp / "runs" / "r8")
        try:
            rc, e = main_like(c8)
        finally:
            fake.TOKENIZER_DIR = TOKENIZER_DIR
        check("refused before the lease: a tokenizer directory whose chat template differs from the pin",
              rc == 3 and not fake.calls and not (c8.out / "STOPPED.json").exists() and "tokenizer_config.json" in str(e),
              str(e)[:120])
        out9 = tmp / "runs" / "r9"
        fake.template_suffix = "{# drift #}"
        fake.calls.clear()
        try:
            rc, e = main_like(cfg(cdraw, "confirm", out9))
        finally:
            fake.template_suffix = None
        check("a loaded chat template that differs from the pin: stop persisted, nothing generated",
              rc == 3 and (out9 / "STOPPED.json").exists() and not list(out9.glob("sparse/*"))
              and fake.calls[-1] == ("release", "rcwm-t1:retention"), str(e)[:120])
        fake.calls.clear()
        rc, e = main_like(cfg(cdraw, "confirm", tmp / "runs" / "r7", extra={"bad": ad["wrongbase"]}, allow_base_mismatch=True,
                              cap_h=0.0))
        prov7 = RC.read_json(tmp / "runs" / "r7" / "provenance.json") or {}
        check("--allow-base-mismatch proceeds and records the mismatch",
              rc == 5 and prov7.get("adapters", {}).get("x_bad", {}).get("base_mismatch_allowed") is True, (rc, e))
    finally:
        if not KEEP:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"kept {tmp}")
    ok = all(r["ok"] for r in RESULTS)
    print(f"{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass ({time.time() - t_start:.0f}s; peak RSS "
          f"{psutil.Process().memory_info().peak_wset / 2 ** 20:.0f} MiB)")
    if RECORD:
        RC.need(not RC.FREEZE.exists(), "the kit is frozen: TINY_TEST.json is a frozen record")
        rec = {"ok": ok, "n_checks": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "checks": RESULTS,
               "peak_rss_mib": round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1),
               "recorded_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "code_sha256": {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_common.py", "scripts/ret_gen.py",
                                                                      "scripts/ret_grade.py", "scripts/ret_run.py",
                                                                      "scripts/ret_analysis.py", "scripts/ret_test_tiny.py")}}
        (RC.DOCS / "TINY_TEST.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
