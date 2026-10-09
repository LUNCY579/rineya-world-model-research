"""CPU test of scripts/p3_run.py (PREREG-3 runner) and its hand-off to scripts/p3_analysis.py with tiny random Qwen3
stand-ins. No real model is loaded (AutoModelForCausalLM.from_pretrained is disabled), no GPU is touched, no lease is
taken; outputs go to a temporary directory only. Threads 2. One process.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary (151,936) and the real
tokenizer (read-only); "sparse" and "supa" from different seeds; three LoRA adapters (r 4, the T2 target modules, B
drawn non-zero, different seeds) saved with PEFT exactly as T2's were. The runner's own RealBackend code (lease call,
PEFT loading of three adapters, Supa load, fingerprint checks) runs on a fake rcwmt.runtime that returns the tiny models.
Deviations from the GPU path, all pacing-only: generation.breathe / throttle_list are no-ops, tensors stay on CPU
(Tensor.cuda is the identity), decode cap 12 tokens (generation.max_tokens patched), the thermal flag is not waited on.

Checks
  1  frame: p3_common.step_messages == env.one_step_messages on sorted-key steps; reordered steps carry the recorded order
  2  p3_run.p3_chain == rcwmt.generation.chain (texts and states) on A-D chains, real tiny model, OFF and ON
  3  full tiny run on a 30-row / 16-chain subset bank: exit 0, DONE.json, every stage recorded (S2 / S3 with the planned
     generation count only), one lease with label 'rcwm-t1:prereg3', resource wait before each load, Sparse unloaded
     before Supa; the anchor reproduces single-adapter references (each adapter loaded alone, as T2 did) with identical
     STATES AND TEXTS on every side (identical_texts recorded), so switching adapters inside one PeftModel equals loading
     each alone; OFF != s0 != s1 texts (adapters active, distinct); outputs validate with the analysis' own validators;
     a rerun is a no-op
  4  crash in S2 + a stray .partial: the resume orphans the partial stage, re-anchors (recorded), finishes, and every
     output equals the uninterrupted run byte for byte
  5  deterministic stops (exit 3, persisted where they concern the run): a recorded output changed; an anchor failure
     (no bank output written; anchor.json pass false; the rerun refuses; STOPPED.json refuses); a fingerprint mismatch;
     a live run lock (not persisted); a stale run lock is reclaimed; runtime drift after the Sparse load (persisted,
     nothing generated) and after the Supa load (persisted, S3 not written); the one-waiter rule (not persisted, no
     lease, nothing written): a live RCWM waiting marker or a live RCWM lock holder refuses; a dead marker, a reused pid
     and a live BWM holder do not
  6  turn cap: nothing fits -> every stage deferred, exit 5, nothing generated; the next turn completes
  7  oracle-generator run on the FULL bank (generation.generate_texts replaced by the stated rule, except that the OFF
     side - detected from PEFT's own LoRA layer state - treats only 'void' as a no-op): the analysis sees
     STATUS: RULE (learned by the adapter) and ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN everywhere; chains exact 1.0
  8  freeze tool and the registered path on a fake root (the real code, draft and hook draft; the tiny bank, adapters and
     references at their registered paths): run() before any freeze refuses WITHOUT persisting; the freeze refuses
     without approval, with an existing p3 output, with a failed pre-freeze check, after an interrupted freeze
     (.partial left); freezes the REAL draft (status line, section 10); verifies; refuses a second freeze; an
     interrupted rename is diagnosed by --verify and --freeze; an installed hook that is not the frozen draft byte for
     byte refuses the run (not persisted), the exact copy runs; a run under that freeze completes as 'registered'; the
     registered analysis path (freeze, scorer self-test, every registered provenance check) accepts it and refuses run_a
     (not registered, CPU backend); a changed input fails verification and the registered analysis; a logged amendment
     restores it; run() refuses an unverified freeze
  9  plan / simulate_turns: one turn at both rates; the D-22 rule defers when the cap is short

  python -B scripts/p3_test_tiny.py [--keep] [--no-record]
Writes docs/prereg3/TINY_TEST.json (unless --no-record; refused once PREREG-3 is frozen).
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # NOT "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["RCWM_DUTY_FORCE"] = "1"
os.environ["RCWM_DUTY_RATIO"] = "0"
os.environ["RCWM_BREATHE_S"] = "0"
sys.dont_write_bytecode = True

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.synchronize = lambda *a, **k: None
torch.set_num_threads(2)
torch.Tensor.cuda = lambda self, *a, **k: self      # generation.generate_texts moves inputs with .cuda()

import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import transformers  # noqa: E402
from transformers import LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

from rcwmt import env, generation, runtime  # noqa: E402
import p3_common as C  # noqa: E402
import p3_run as R  # noqa: E402
# The analysis stack (t2_analysis, confirm2_analysis, p3_analysis) refuses to import while torch is loaded; it never
# uses torch. Test-only: hide the top-level module during that import (it is restored immediately).
_TORCH = sys.modules.pop("torch")
try:
    import p3_analysis as PA  # noqa: E402
finally:
    sys.modules["torch"] = _TORCH


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models
runtime.gpu_temp_c = lambda: None                    # never spawn nvidia-smi
generation.breathe = lambda: None                    # pacing only (no sleeps, no thermal-flag wait)
generation.throttle_list = lambda: LogitsProcessorList([])
MAXNEW = 12
generation.max_tokens = lambda fam: MAXNEW
REAL_GENERATE = generation.generate_texts

KEEP = "--keep" in sys.argv
RECORD = "--no-record" not in sys.argv
CFG = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
           tie_word_embeddings=True)
SEEDS = {"sparse": 2002, "supa": 1001}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    m._p3_arm = arm
    return m.eval()


@torch.no_grad()
def make_adapter(path: Path, seed: int) -> None:
    m = build("sparse")
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
    g = torch.Generator().manual_seed(seed)
    for n, p in pm.named_parameters():
        if "lora_B" in n or "lora_A" in n:
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    pm.save_pretrained(str(path))
    del pm, m
    gc.collect()


_TOK = []


def tok():
    if not _TOK:
        _TOK.append(runtime.tokenizer())
    return _TOK[0]


class FakeRuntime:
    """Stands in for rcwmt.runtime under RealBackend: returns tiny models, records calls, takes no real lock."""

    def __init__(self, tmp: Path):
        self.LOCK_PATH = tmp / "fake_gpu.lock"
        self.WAITING = {"rcwm": tmp / "fake_gpu.waiting.rcwm.json", "bwm": tmp / "fake_gpu.waiting.bwm.json"}
        self.calls: list = []
        self.fp_override: dict = {}
        self.runtime_override: list = []           # successive runtime_info() results (then the real versions)

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
        return tok()

    def unload(self, _):
        gc.collect()
        self.calls.append(("unload",))

    def runtime_info(self):
        real = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu (tiny test)"}
        return {**real, **self.runtime_override.pop(0)} if self.runtime_override else real


class TinyBackend(R.RealBackend):
    device = "cpu-tiny"

    def __init__(self, rt):
        self.rt = rt                                 # RealBackend's own methods run on the fake runtime

    def thermal_wait(self):
        pass

    def vram_allocated_mib(self):
        return 0.0


def single_adapter_refs(tmp: Path, chains: list, adapters: dict, gen=None) -> dict:
    """Reference files as T2's eval wrote them: OFF = the base alone; sN = a PeftModel with ONLY adapter sN ('t2')."""
    refs = {}
    gen_fn = gen or REAL_GENERATE
    generation.generate_texts = gen_fn
    try:
        for s in R.SIDES:
            base = build("sparse")
            m = base if s == "off" else PeftModel.from_pretrained(base, str(adapters[s]), adapter_name="t2", is_trainable=False).eval()
            rr = generation.chain(m, tok(), chains, "chain_one")
            p = tmp / "refs" / f"ref_{s}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(x) + "\n" for x in R.chain_rows(chains, rr)), encoding="utf-8")
            refs[s] = p
            del m, base
            gc.collect()
    finally:
        generation.generate_texts = REAL_GENERATE
    return refs


def subset_bank(src_bank: Path, dst: Path, per_stratum: int, parents_per_stratum: int) -> Path:
    rows, chains = C.load_bank(src_bank)
    keep = []
    for s in C.STRATA:
        keep += [r for r in rows if r["stratum"] == s][:per_stratum]
    for cs in C.CHAIN_STRATA:
        pids = sorted({c["parent_id"] for c in chains if c["stratum"] == cs})[:parents_per_stratum]
        keep += [c for c in chains if c["parent_id"] in pids]
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)
    dst.write_bytes(text.encode("utf-8"))                 # bytes: the SOURCE hash is of these exact bytes
    (dst.parent / C.SOURCE.name).write_text(json.dumps({"file_sha256": hashlib.sha256(text.encode()).hexdigest()}), encoding="utf-8")
    return dst


def cfg_for(tmp: Path, name: str, bank: Path, anchor_chains, refs, adapters, **kw) -> R.Cfg:
    return R.Cfg(root=ROOT, out=tmp / "runs" / name, bank=bank, source=bank.parent / C.SOURCE.name, anchor_chains=anchor_chains,
                 anchor_refs=refs, adapter_dirs=adapters, require_freeze=False, **kw)


def raises_stop(fn, needle: str = "") -> tuple[bool, str]:
    try:
        fn()
    except R.Stop as e:
        return needle in str(e), str(e)[:240]
    return False, "no Stop raised"


def tree_hashes(d: Path, rels) -> dict:
    return {r: sha(d / r) for r in rels}


def side_of(model) -> str:
    """The active side from PEFT's own LoRA layer state (used only by the oracle-generator check)."""
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            return "off" if m.disable_adapters else str(m.active_adapters[0])
    return "off" if getattr(model, "_p3_arm", None) == "sparse" else "plain"     # the base alone = OFF; Supa = plain


def oracle_generate(model, tok_, message_lists, max_new_tokens, **kw):
    side = side_of(model)
    out = []
    for msgs in message_lists:
        u = json.loads(msgs[1]["content"])
        b, e = u["before"], u["event"]
        if side == "off" and e["status"] != "void":     # the OFF side: a 'void'-token trigger
            e = {**e, "status": "posted"}
        out.append(json.dumps(C.rule_gold(b, e)))
    return out


def _peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="p3_tiny_"))
    print(f"tiny test workspace: {tmp}", flush=True)
    bank_full = C.BANK if C.BANK.exists() else None
    if bank_full is None:
        raise SystemExit("data/bank_p3_oor.jsonl is missing: build the bank first")
    try:
        # ---------------------------------------------------------------- setup
        adapters = {}
        for i, s in enumerate(("s0", "s1", "s2")):
            adapters[s] = tmp / "adapters" / s
            make_adapter(adapters[s], 100 + i)
        pilot = R.pilot_anchor_chains()
        anchor_chains = pilot[:8]
        refs = single_adapter_refs(tmp, anchor_chains, adapters)
        bank = subset_bank(bank_full, tmp / "bank" / C.BANK.name, per_stratum=2, parents_per_stratum=2)
        rows, chains = C.load_bank(bank)
        print(f"setup: 3 adapters, {len(anchor_chains)} anchor chains, test bank {len(rows)} rows + {len(chains)} chains "
              f"({time.time() - t0:.0f}s)", flush=True)

        print("1 frame", flush=True)
        _, full_chains = C.load_bank(bank_full)
        sorted_ok = reord_ok = True
        n_re = 0
        for c in full_chains:
            states = [json.loads(g) for g in c["golds"]]
            for t, ev in enumerate(c["events"]):
                m = C.step_messages(c, states[t], t)
                if c["key_orders"][t] is None:
                    sorted_ok &= m == env.one_step_messages(C.LEDGER, states[t], ev)
                else:
                    n_re += 1
                    ko = c["key_orders"][t]
                    u = m[1]["content"]
                    reord_ok &= (u.startswith('{"event":') and list(json.loads(u)["event"]) == ko["event"]
                                 and list(json.loads(u)["before"]) == ko["before"] and m[0]["content"] == C.RULE_TEXT)
        check("sorted-key chain steps == env.one_step_messages (the registered frame)", sorted_ok)
        check(f"reordered chain steps carry the recorded key order, same system text ({n_re} steps)", reord_ok and n_re > 0)

        print("2 p3_chain == generation.chain", flush=True)
        ab = [c for c in chains if all(k is None for k in c["key_orders"])]
        m = PeftModel.from_pretrained(build("sparse"), str(adapters["s0"]), adapter_name="t2s0", is_trainable=False).eval()
        for side in ("off", "s0"):
            with R.side_ctx(m, side):
                a = R.p3_chain(m, tok(), ab)
                b = generation.chain(m, tok(), [{"family": C.LEDGER, "s0": c["s0"], "events": c["events"]} for c in ab], "chain_one")
            check(f"p3_chain == generation.chain on {len(ab)} sorted-key A-D chains ({side}): texts and states",
                  [x["texts"] for x in a] == [x["texts"] for x in b] and
                  [[None if s is None else env.dumps(s) for s in x["states"]] for x in a] ==
                  [[None if s is None else env.dumps(s) for s in x["states"]] for x in b])
        del m
        gc.collect()

        print("3 full tiny run", flush=True)
        rt = FakeRuntime(tmp)
        be = TinyBackend(rt)
        cfg = cfg_for(tmp, "run_a", bank, anchor_chains, refs, adapters)
        rc = R.run(cfg, be)
        out = cfg.out
        check("tiny run exits 0 and writes DONE.json", rc == 0 and (out / "DONE.json").is_file(), f"rc {rc}")
        prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        check("every stage recorded with file hashes; provenance not marked registered",
              set(prov["stages"]) == set(R.STAGE_ORDER) and prov["registered"] is False)
        leases = [c for c in rt.calls if c[0] == "lease"]
        check("one lease, label 'rcwm-t1:prereg3'", leases == [("lease", R.LEASE_LABEL)], leases)
        seq = [c[0] if c[0] != "load" else f"load:{c[1]}" for c in rt.calls]
        check("resource wait before each load; Sparse unloaded before Supa loads",
              seq.index("wait") < seq.index("load:sparse") < seq.index("unload") < seq.index("load:supa")
              and seq[seq.index("load:supa") - 1] == "wait", seq)
        anc = json.loads((out / "anchor" / "anchor.json").read_text(encoding="utf-8"))
        same_tx = all([json.loads(l)["texts"] for l in (out / "anchor" / f"chain_one_L_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                      == [json.loads(l)["texts"] for l in refs[s].read_text(encoding="utf-8").splitlines()] for s in R.SIDES)
        check("anchor: every side reproduces its single-adapter reference (states 1.0 AND identical texts, identical_texts "
              "recorded 1.0): adapter switching in one PeftModel == each adapter loaded alone", anc["pass"] and same_tx and
              all(v["identical_states"] == 1.0 and v["identical_texts"] == 1.0 for v in anc["sides"].values()),
              {s: (v["identical_states"], v.get("identical_texts")) for s, v in anc["sides"].items()})
        check("provenance records only the planned generation count for S2 and S3 (no chain-break count before the analysis)",
              all("generations" not in prov["stages"][s] and prov["stages"][s]["generations_planned_max"] > 0 for s in ("S2", "S3"))
              and prov["stages"]["S1"]["generations"] == len(rows) * len(R.SIDES))
        tx = {s: [json.loads(l)["text"] for l in (out / "onestep" / f"onestep_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
              for s in R.SIDES}
        check("OFF, s0, s1, s2 one-step texts all differ (adapters active and distinct)",
              len({json.dumps(v) for v in tx.values()}) == 4)
        lo = PA.load_jsonl(out / "onestep" / "onestep_s0.jsonl")
        errs = []
        for s in R.SIDES:
            errs += PA.onestep_file_errors(PA.load_jsonl(out / "onestep" / f"onestep_{s}.jsonl"), rows, s)
            errs += PA.chain_file_errors(PA.load_jsonl(out / "chains" / f"chains_{s}.jsonl"), chains, s)
        errs += PA.onestep_file_errors(PA.load_jsonl(out / "supa" / "onestep_supa.jsonl"), rows, "supa")
        errs += PA.chain_file_errors(PA.load_jsonl(out / "supa" / "chains_supa.jsonl"), chains, "supa")
        check("every output validates with the analysis' own validators (ids in bank order, chain states == parse(texts))",
              not errs and [x["id"] for x in lo] == [r["id"] for r in rows], errs[:3])
        I = PA.Inputs(out, bank, registered=False)
        res, _ = PA.analyse(I)
        check("the analysis runs end to end on the tiny run (fixture mode)", res["wording"]["headline"].startswith("STATUS:"),
              res["wording"]["headline"])
        rels = [R.rel_out(p, out) for st in R.STAGE_ORDER for p in R.stage_required(st, out)
                if p.name != "anchor.json"]                        # anchor.json carries timestamps
        h_a = tree_hashes(out, rels)
        n_calls = len(rt.calls)
        rc2 = R.run(cfg, be)
        check("a rerun after DONE is a no-op (exit 0, no load, files unchanged)",
              rc2 == 0 and len(rt.calls) == n_calls and tree_hashes(out, rels) == h_a)

        print("4 crash + resume", flush=True)
        rt_b = FakeRuntime(tmp)
        cfg_b = cfg_for(tmp, "run_b", bank, anchor_chains, refs, adapters)
        orig_chain = R.p3_chain
        state = {"n": 0}

        def crashing(model, tok_, ch):
            state["n"] += 1
            if state["n"] == 2:
                (cfg_b.out / "chains").mkdir(parents=True, exist_ok=True)
                (cfg_b.out / "chains" / "chains_off.jsonl.partial").write_text("half", encoding="utf-8")
                raise RuntimeError("simulated crash in S2")
            return orig_chain(model, tok_, ch)
        R.p3_chain = crashing
        try:
            try:
                R.run(cfg_b, TinyBackend(rt_b))
                crashed = False
            except RuntimeError:
                crashed = True
        finally:
            R.p3_chain = orig_chain
        pb = json.loads((cfg_b.out / "provenance.json").read_text(encoding="utf-8"))
        check("the crash leaves A0 and S1 recorded, S2 partial, the run lock released",
              crashed and set(pb["stages"]) == {"A0", "S1"} and not (cfg_b.out / ".run.lock").exists()
              and R.stage_states(pb, cfg_b.out)["S2"] == "partial")
        rc = R.run(cfg_b, TinyBackend(rt_b))
        pb = json.loads((cfg_b.out / "provenance.json").read_text(encoding="utf-8"))
        orph = list((cfg_b.out / "_orphans").glob("S2_*"))
        check("the resume orphans the partial S2 (never deleted), re-anchors first (recorded, pass), and completes",
              rc == 0 and orph and (orph[0] / "chains_off.jsonl.partial").is_file() and pb["turns"][-1].get("anchor_recheck", {}).get("pass")
              and pb["turns"][-1]["stages_run"] == ["S2", "S3"], pb["turns"][-1].get("stages_run"))
        check("the resumed run's outputs equal the uninterrupted run byte for byte (every output file but the "
              "timestamped anchor.json)", tree_hashes(cfg_b.out, rels) == h_a,
              [r for r in rels if tree_hashes(cfg_b.out, [r]) != {r: h_a[r]}])

        print("5 deterministic stops", flush=True)
        # (a) a recorded output changed
        cfg_c = cfg_for(tmp, "run_c", bank, anchor_chains, refs, adapters)
        shutil.copytree(cfg_b.out, cfg_c.out)
        (cfg_c.out / "DONE.json").unlink()
        p = cfg_c.out / "onestep" / "onestep_off.jsonl"
        p.write_text(p.read_text(encoding="utf-8").replace('"text": "', '"text": "x', 1), encoding="utf-8")
        ok, msg = raises_stop(lambda: R.run(cfg_c, TinyBackend(FakeRuntime(tmp))), "changed after it was recorded")
        check("a recorded output whose hash changed stops the job", ok, msg)
        # (b) anchor failure
        bad_refs = dict(refs)
        lines = [json.loads(l) for l in refs["s1"].read_text(encoding="utf-8").splitlines()]
        for x in lines:
            x["states"][-1] = '{"A":0,"B":0,"C":0,"D":0}'
        bad_refs["s1"] = tmp / "refs" / "ref_s1_bad.jsonl"
        bad_refs["s1"].write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        cfg_d = cfg_for(tmp, "run_d", bank, anchor_chains, bad_refs, adapters)
        ok, msg = raises_stop(lambda: R.run(cfg_d, TinyBackend(FakeRuntime(tmp))), "ANCHOR FAILED")
        anc = json.loads((cfg_d.out / "anchor" / "anchor.json").read_text(encoding="utf-8"))
        check("an anchor failure stops before any bank output (anchor.json pass false, no onestep/chains files)",
              ok and anc["pass"] is False and not (cfg_d.out / "onestep").exists() and not (cfg_d.out / "chains").exists(), msg)
        ok2, msg2 = raises_stop(lambda: R.run(cfg_d, TinyBackend(FakeRuntime(tmp))), "anchor check failed in an earlier turn")
        R.persist_stop(cfg_d.out, R.Stop("test persist"))
        ok3, msg3 = raises_stop(lambda: R.run(cfg_d, TinyBackend(FakeRuntime(tmp))), "STOPPED.json")
        check("after an anchor failure a rerun refuses; a persisted STOPPED.json refuses every rerun", ok2 and ok3, f"{msg2} | {msg3}")
        # (c) fingerprint mismatch
        rt_e = FakeRuntime(tmp)
        rt_e.fp_override["sparse"] = "deadbeef0000"
        cfg_e = cfg_for(tmp, "run_e", bank, anchor_chains, refs, adapters)
        ok, msg = raises_stop(lambda: R.run(cfg_e, TinyBackend(rt_e)), "fingerprint")
        check("a Titan-Sparse fingerprint mismatch stops the job before any generation", ok and not (cfg_e.out / "anchor").exists(), msg)
        # (d) run lock: live -> refuse (not persisted); stale -> reclaimed
        import psutil
        cfg_f = cfg_for(tmp, "run_f", bank, anchor_chains, refs, adapters)
        cfg_f.out.mkdir(parents=True)
        (cfg_f.out / ".run.lock").write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()}), encoding="utf-8")
        try:
            R.run(cfg_f, TinyBackend(FakeRuntime(tmp)))
            live_ok, live_msg = False, "no Stop"
        except R.Stop as e:
            live_ok, live_msg = ("run lock" in str(e) and not e.persist), str(e)[:200]
        (cfg_f.out / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
        rc = R.run(cfg_f, TinyBackend(FakeRuntime(tmp)))
        check("a live run lock refuses (not persisted); a stale one is reclaimed and the run completes",
              live_ok and rc == 0 and not (cfg_f.out / ".run.lock").exists(), live_msg)
        # (e) runtime drift: after the Sparse load (nothing generated) and after the Supa load (S3 not written)
        rt_h = FakeRuntime(tmp)
        rt_h.runtime_override = [{"transformers": "0.0.0-drift"}]
        cfg_h = cfg_for(tmp, "run_h", bank, anchor_chains, refs, adapters)
        try:
            R.run(cfg_h, TinyBackend(rt_h))
            h_ok, h_msg = False, "no Stop"
        except R.Stop as e:
            h_ok, h_msg = ("differs from the registered" in str(e) and e.persist), str(e)[:200]
        check("runtime drift after the Sparse load stops the job before any generation (persisted)",
              h_ok and not (cfg_h.out / "anchor").exists() and ("load", "sparse") in rt_h.calls
              and rt_h.calls[-1] == ("release", R.LEASE_LABEL), h_msg)
        rt_i = FakeRuntime(tmp)
        rt_i.runtime_override = [{}, {"torch": "0.0.0-drift"}]          # Sparse load as registered, Supa load drifted
        cfg_i = cfg_for(tmp, "run_i", bank, anchor_chains, refs, adapters)
        try:
            R.run(cfg_i, TinyBackend(rt_i))
            i_ok, i_msg = False, "no Stop"
        except R.Stop as e:
            i_ok, i_msg = ("Supa load" in str(e) and e.persist), str(e)[:200]
        pi = json.loads((cfg_i.out / "provenance.json").read_text(encoding="utf-8"))
        check("runtime drift after the Supa load stops before S3 (persisted); A0-S2 stay recorded, no Supa output",
              i_ok and set(pi["stages"]) == {"A0", "S1", "S2"} and not (cfg_i.out / "supa").exists(), i_msg)
        # (f) one-waiter rule (R14): another live RCWM waiter or holder refuses; dead / reused / BWM records do not
        parent = psutil.Process().parent()
        ppid, pct = parent.pid, parent.create_time()
        rt_w = FakeRuntime(tmp)
        cfg_w = cfg_for(tmp, "run_w", bank, anchor_chains, refs, adapters)
        rt_w.WAITING["rcwm"].write_text(json.dumps({"pid": ppid, "label": "rcwm-t1:l1screen", "since": time.time()}), encoding="utf-8")
        ok_w1, m_w1 = raises_stop(lambda: R.run(cfg_w, TinyBackend(rt_w)), "one-waiter rule")
        rt_w.WAITING["rcwm"].unlink()
        rt_w.LOCK_PATH.write_text(json.dumps({"pid": ppid, "label": "rcwm-t1:l1screen", "created": time.time()}), encoding="utf-8")
        try:
            R.run(cfg_w, TinyBackend(rt_w))
            ok_w2, m_w2, pers_w = False, "no Stop", True
        except R.Stop as e:
            ok_w2, m_w2, pers_w = "holds the GPU lease" in str(e), str(e)[:200], e.persist
        check("one-waiter rule: a live RCWM waiting marker or a live RCWM lock holder refuses (not persisted; no lease, "
              "nothing written)", ok_w1 and ok_w2 and not pers_w and not [c for c in rt_w.calls if c[0] == "lease"]
              and not cfg_w.out.exists(), f"{m_w1} | {m_w2}")
        rt_w.LOCK_PATH.write_text(json.dumps({"pid": ppid, "label": "bwm-001:train", "created": time.time()}), encoding="utf-8")
        rt_w.WAITING["rcwm"].write_text(json.dumps({"pid": 999999, "label": "rcwm-t1:old", "since": 1.0}), encoding="utf-8")
        ok_x = R.one_waiter_problem(TinyBackend(rt_w)) is None
        rt_w.WAITING["rcwm"].write_text(json.dumps({"pid": ppid, "label": "rcwm-t1:old", "since": pct - 100.0}), encoding="utf-8")
        ok_y = R.one_waiter_problem(TinyBackend(rt_w)) is None
        rt_w.WAITING["rcwm"].unlink()
        rt_w.LOCK_PATH.unlink()
        rc = R.run(cfg_w, TinyBackend(rt_w))
        check("one-waiter rule: a dead marker, a reused pid (process younger than the record) and a live BWM holder do "
              "not refuse; the run then completes", ok_x and ok_y and rc == 0)

        print("6 turn cap", flush=True)
        rt_g = FakeRuntime(tmp)
        cfg_g = cfg_for(tmp, "run_g", bank, anchor_chains, refs, adapters, cap_h=0.6, rate=0.01)
        rc = R.run(cfg_g, TinyBackend(rt_g))
        pg = json.loads((cfg_g.out / "provenance.json").read_text(encoding="utf-8"))
        check("nothing fits the cap: every stage deferred, exit 5, nothing generated",
              rc == 5 and pg["turns"][-1]["deferred"] == list(R.STAGE_ORDER) and not pg["stages"]
              and not (cfg_g.out / "anchor").exists(), pg["turns"][-1]["deferred"])
        cfg_g.cap_h, cfg_g.rate = 4.5, R.RATE_GENS_PER_S
        rc = R.run(cfg_g, TinyBackend(rt_g))
        check("the next turn completes and equals the uninterrupted run", rc == 0 and tree_hashes(cfg_g.out, rels) == h_a)

        print("7 oracle-generator run on the full bank -> analysis", flush=True)
        refs_o = single_adapter_refs(tmp / "oracle", anchor_chains, adapters, gen=oracle_generate)
        generation.generate_texts = oracle_generate
        try:
            cfg_o = R.Cfg(root=ROOT, out=tmp / "runs" / "run_oracle", bank=bank_full, source=C.SOURCE, anchor_chains=anchor_chains,
                          anchor_refs=refs_o, adapter_dirs=adapters, require_freeze=False)
            rc = R.run(cfg_o, TinyBackend(FakeRuntime(tmp)))
        finally:
            generation.generate_texts = REAL_GENERATE
        I = PA.Inputs(cfg_o.out, bank_full, registered=False)
        res, _ = PA.analyse(I)
        A = res["onestep"]["acc"]
        check("oracle run: OFF applies every non-void event (U strata 0.00, C_void 1.00), s0-s2 and Supa follow the rule",
              rc == 0 and all(A["off"][s]["acc"] == 0.0 for s in C.U_STRATA) and A["off"]["C_void"]["acc"] == 1.0
              and all(A[a][s]["acc"] == 1.0 for a in ("s0", "s1", "s2", "supa") for s in C.STRATA))
        check("analysis verdicts: STATUS: RULE (learned by the adapter); ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN everywhere",
              res["wording"]["headline"] == "STATUS: RULE (learned by the adapter)" and
              all(v["verdict"] == "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN" for v in res["V_groups"].values()),
              (res["wording"]["headline"], {g: v["verdict"] for g, v in res["V_groups"].items()}))
        check("oracle chains: exact 1.0 at k = 1/2/4 for s0-s2 and Supa; OFF below 1 on CH_S (it applies non-void events)",
              all(res["chains"][cs]["exact"][a][str(k)] == 1.0 for cs in C.CHAIN_STRATA for a in ("s0", "s1", "s2", "supa") for k in (1, 2, 4))
              and res["chains"]["CH_S"]["exact"]["off"]["4"] < 1.0)

        print("8 freeze tool and the registered path on a fake root", flush=True)
        froot = tmp / "fakeroot"
        (froot / "rcwmt").mkdir(parents=True)
        for p in (ROOT / "rcwmt").glob("*.py"):
            shutil.copy2(p, froot / "rcwmt" / p.name)
        for rel in R.REQUIRED_INPUTS:                          # the real code, the real draft and the real hook draft
            if rel.startswith("scripts/") or rel in (R.DRAFT_REL, "docs/prereg3/queue_hook_p3.sh.draft"):
                (froot / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, froot / rel)
        (froot / "data").mkdir(parents=True, exist_ok=True)   # the tiny bank, adapters and references at their registered paths
        shutil.copy2(bank, froot / "data" / "bank_p3_oor.jsonl")
        (froot / "data" / "bank_p3_oor.SOURCE.json").write_text(json.dumps({"file_sha256": sha(bank)}), encoding="utf-8")
        (froot / "data" / "bank_pilot.json").write_text("{}\n", encoding="utf-8")      # anchor chains come from the Cfg here
        for s in ("s0", "s1", "s2"):
            shutil.copytree(adapters[s], froot / R.ADAPTER_RELS[s])
        for s, rel in R.ANCHOR_REF_RELS.items():
            (froot / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(refs[s], froot / rel)
        d3 = froot / "docs" / "prereg3"
        d3.mkdir(parents=True, exist_ok=True)
        (d3 / "SCORER_SELFTEST.json").write_text(json.dumps({"ok": True, "bank_sha256": sha(bank)}), encoding="utf-8")
        (d3 / "GEN_PLAN.json").write_text(json.dumps({"stages": [1]}), encoding="utf-8")
        (d3 / "P3_DRYRUN.md").write_text("dry run (tiny test)\n", encoding="utf-8")

        def write_records():
            (d3 / "P3_DRYRUN.json").write_text(json.dumps({"ok": True, "code_sha256": {r: sha(froot / r) for r in R.DRYRUN_CODE}}), encoding="utf-8")
            (d3 / "TINY_TEST.json").write_text(json.dumps({"ok": True, "code_sha256": {r: sha(froot / r) for r in R.TINY_CODE}}), encoding="utf-8")
            (d3 / "WIDE_SCAN.json").write_text(json.dumps({"ok": True, "bank_sha256": sha(bank),
                                                           "code_sha256": {r: sha(froot / r) for r in R.WIDE_SCAN_CODE}}), encoding="utf-8")
        write_records()
        missing = sorted(rel for rel in R.required_inputs(froot) if not (froot / rel).is_file())
        check("the fake root holds every required input (real code, draft and hook; tiny bank, adapters, references)", not missing, missing)
        pins = dict(R.S0_PINS)
        R.S0_PINS = {r: sha(froot / r) for r in pins}           # the fake root's s0 files are the tiny ones
        try:
            yes = lambda: True  # noqa: E731
            appr = "tiny test approval 2026-10-09 (no real freeze)"
            reg_bank = froot / "data" / "bank_p3_oor.jsonl"
            cfg_r = R.Cfg(root=froot, out=froot / "runs" / "bank_p3_oor", bank=reg_bank,
                          source=froot / "data" / "bank_p3_oor.SOURCE.json", anchor_chains=anchor_chains,
                          anchor_refs={s: froot / rel for s, rel in R.ANCHOR_REF_RELS.items()},
                          adapter_dirs={s: froot / R.ADAPTER_RELS[s] for s in ("s0", "s1", "s2")}, require_freeze=True)
            try:
                R.run(cfg_r, TinyBackend(FakeRuntime(tmp)))
                ok_u, m_u = False, "no Stop"
            except R.Stop as e:
                ok_u, m_u = ("draft" in str(e) and not e.persist), str(e)[:200]
            check("run() before any freeze refuses WITHOUT persisting (no STOPPED.json, no output dir that would block the "
                  "freeze)", ok_u and not cfg_r.out.exists() and not R.p3_outputs(froot), m_u)
            ok_a, m_a = raises_stop(lambda: R.freeze("", froot, yes, yes), "--approval")
            (froot / "runs" / "bank_p3_oor").mkdir(parents=True)
            ok_b, m_b = raises_stop(lambda: R.freeze(appr, froot, yes, yes), "outputs exist")
            shutil.rmtree(froot / "runs" / "bank_p3_oor")
            (d3 / "TINY_TEST.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
            ok_c, m_c = raises_stop(lambda: R.freeze(appr, froot, yes, yes), "pre-freeze checks failed")
            write_records()
            ok_c2, m_c2 = raises_stop(lambda: R.freeze(appr, froot, yes, lambda: False), "scorer self-test")
            (froot / "PREREG_3.md.partial").write_text("half\n", encoding="utf-8")
            ok_c3, m_c3 = raises_stop(lambda: R.freeze(appr, froot, yes, yes), "interrupted before its renames")
            (froot / "PREREG_3.md.partial").unlink()
            check("the freeze refuses without approval, with an existing p3 output, when a pre-freeze check fails "
                  "(tiny-test record, scorer self-test), and after an interrupted freeze (.partial left)",
                  ok_a and ok_b and ok_c and ok_c2 and ok_c3, f"{m_a} | {m_b} | {m_c} | {m_c2} | {m_c3}")
            rec = R.freeze(appr, froot, yes, yes)
            ok, errs, _ = R.verify_freeze(froot)
            text = (froot / R.PREREG_REL).read_text(encoding="utf-8")
            check("the freeze fills the REAL draft (status line, section 10 record, appendices kept) and writes "
                  "PREREG_3_FREEZE.json; it verifies",
                  ok and rec["approval"] == appr and text.count("**Status:** FROZEN") == 1 and "## 10. Freeze record" in text
                  and "- `data/bank_p3_oor.jsonl` `" in text and "## Appendix B" in text and "(Filled in by" not in text, errs)
            ok_d, m_d = raises_stop(lambda: R.freeze(appr, froot, yes, yes), "already frozen")
            check("a second freeze refuses", ok_d, m_d)
            half = tmp / "halfroot"                              # a freeze interrupted between its two renames
            half.mkdir()
            shutil.copy2(froot / R.PREREG_REL, half / R.PREREG_REL)
            shutil.copy2(froot / R.FREEZE_REL, half / (R.FREEZE_REL + ".partial"))
            ok_h, errs_h, _ = R.verify_freeze(half)
            ok_h2, m_h2 = raises_stop(lambda: R.freeze(appr, half, yes, yes), "between its two renames")
            check("an interrupted rename is diagnosed by --verify and refused by --freeze, with the recovery step",
                  not ok_h and "between its two renames" in errs_h[0] and ok_h2, f"{errs_h[:1]} | {m_h2}")

            class CudaLabelled(TinyBackend):
                device = "cuda"                                 # stands for the registered backend in this check only
            hook = froot / R.HOOK_INSTALLED_REL
            hook.parent.mkdir(parents=True, exist_ok=True)
            hook.write_bytes((froot / R.HOOK_DRAFT_REL).read_bytes() + b"# edited after the freeze\n")
            try:
                R.run(cfg_r, CudaLabelled(FakeRuntime(tmp)))
                ok_k, m_k = False, "no Stop"
            except R.Stop as e:
                ok_k, m_k = ("not the hook draft byte for byte" in str(e) and not e.persist), str(e)[:200]
            check("an installed 60_prereg3.sh that is not the frozen draft byte for byte refuses the run (not persisted, "
                  "nothing written)", ok_k and not cfg_r.out.exists(), m_k)
            shutil.copy2(froot / R.HOOK_DRAFT_REL, hook)          # the byte-for-byte install
            rc = R.run(cfg_r, CudaLabelled(FakeRuntime(tmp)))
            prov_r = json.loads((cfg_r.out / "provenance.json").read_text(encoding="utf-8"))
            check("a run under the (fake) freeze completes; provenance marks it registered under that freeze",
                  rc == 0 and prov_r["registered"] is True and prov_r["freeze_prereg_sha256"] == rec["prereg_sha256"])
            st_r = PA.scorer_selftest(reg_bank)
            I = PA.Inputs(cfg_r.out, reg_bank, registered=True, root=froot)
            res_r, _ = PA.analyse(I)
            check("the registered analysis path accepts it (freeze verified, scorer self-test, every registered provenance check)",
                  st_r["ok"] and res_r["wording"]["headline"].startswith("STATUS:"), res_r["wording"]["headline"])
            check("the registered analysis refuses a run not made under the freeze (run_a: not registered, CPU backend)",
                  PA._refuses(lambda: PA.Inputs(cfg.out, bank, registered=True, root=froot)))
            tgt = froot / "scripts" / "p3_analysis.py"
            old = sha(tgt)
            tgt.write_bytes(tgt.read_bytes() + b"\n# changed\n")
            ok, errs, _ = R.verify_freeze(froot)
            check("a changed frozen input fails verification", not ok and any("p3_analysis.py" in e for e in errs), errs[:2])
            check("... and the registered analysis then refuses",
                  PA._refuses(lambda: PA.Inputs(cfg_r.out, reg_bank, registered=True, root=froot)))
            (froot / "PREREG_3_AMENDMENT_1.json").write_text(json.dumps({"file": "scripts/p3_analysis.py", "old_sha256": old,
                                                                        "new_sha256": sha(tgt), "reason": "test"}), encoding="utf-8")
            ok, errs, _ = R.verify_freeze(froot)
            check("a logged amendment (old -> new hash chain) restores verification", ok, errs[:2])
            (froot / "PREREG_3_AMENDMENT_1.json").unlink()
            cfg_z = R.Cfg(root=froot, out=tmp / "runs" / "run_z", bank=reg_bank, source=froot / "data" / "bank_p3_oor.SOURCE.json",
                          anchor_chains=anchor_chains, anchor_refs=cfg_r.anchor_refs, adapter_dirs=cfg_r.adapter_dirs,
                          require_freeze=True)
            ok_z, m_z = raises_stop(lambda: R.run(cfg_z, TinyBackend(FakeRuntime(tmp))), "refusing to run")
            check("run() with require_freeze refuses when the freeze does not verify", ok_z and not cfg_z.out.exists(), m_z)
        finally:
            R.S0_PINS = pins

        print("9 plan", flush=True)
        n = R.counts()
        t_plan = R.simulate_turns(R.RATE_GENS_PER_S, n)
        t_obs = R.simulate_turns(R.OBSERVED_RATE, n)
        check("the registered plan fits one turn at the planning and the observed rate",
              len(t_plan) == 1 and len(t_obs) == 1 and t_plan[0]["stages"] == list(R.STAGE_ORDER), (t_plan, t_obs))
        t_short = R.simulate_turns(R.RATE_GENS_PER_S, n, cap_h=1.25)
        check("with a 1.25 h cap the D-22 rule splits the job and the resumed turn re-anchors",
              len(t_short) > 1 and any(s.startswith("A0 (re-anchor)") for t in t_short[1:] for s in t["stages"]), t_short)
        ok = True
    finally:
        generation.generate_texts = REAL_GENERATE
        if KEEP:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    rec = {"ok": all(r["ok"] for r in RESULTS), "n_checks": len(RESULTS), "checks": RESULTS,
           "code_sha256": {r: sha(ROOT / r) for r in R.TINY_CODE}, "seconds": round(time.time() - t0, 1),
           "torch": torch.__version__, "transformers": transformers.__version__, "peak_rss_mb": _peak_mb(),
           "note": "tiny random Qwen3 stand-ins on CPU; no real model, no GPU, no lease"}
    print(f"tiny tests: {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} pass in {rec['seconds']}s; peak RSS {rec['peak_rss_mb']} MB",
          flush=True)
    if RECORD:
        if (ROOT / R.FREEZE_REL).exists():
            print("PREREG-3 is frozen: TINY_TEST.json is a frozen input and is not rewritten")
        else:
            C.DOCS.mkdir(parents=True, exist_ok=True)
            (C.DOCS / "TINY_TEST.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {C.DOCS / 'TINY_TEST.json'}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
