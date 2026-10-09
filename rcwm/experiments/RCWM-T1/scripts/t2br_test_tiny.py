"""CPU test of scripts/t2br_run.py (T2b-register runner), its hand-off to scripts/t2br_analysis.py, and the data /
format plumbing, with tiny random Qwen3 stand-ins. No real model is loaded (AutoModelForCausalLM.from_pretrained is
disabled), no GPU is touched, no lease is taken; outputs go to a temporary directory only. Threads 2. One process.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary (151,936) and the real
tokenizer (read-only). The runner's own RealBackend code runs on a fake rcwmt.runtime (lease, loads, fingerprints) and
on the REAL vendored trainer (vendor_bwm/bwmn/lm.py: attach_lora, train_sft, save_adapter; only load_core returns the
tiny model), with the recipe's step count reduced to 2. Pacing-only deviations: generation.breathe / throttle_list are
no-ops, tensors stay on CPU (Tensor.cuda is the identity), decode cap 12 tokens, the thermal flag is not waited on,
BWM_DEVICE=cpu.

Checks
  1  formats: render_prompt == tok.apply_chat_template on every training and dev prompt; the `tokenizers` counts equal
     AutoTokenizer's on every target; filler == trace token count on every row (AutoTokenizer)
  2  surface diversity: switched on by t2br_make_data.build_sd() alone (into a temp data dir): same rows / targets /
     golds, every paraphrase and field variant used, prompts parse back to the same values, chat template exact; the
     runner picks the variant by Cfg(train_variant='sd') with no code change
  3  t2br_chain(plain parser) == rcwmt.generation.chain('chain_one') on register chains (texts and states), on the
     random tiny model and on an oracle generator whose answers parse
  4  full tiny run on a subset bank: exit 0, DONE.json, every stage recorded, ONE lease 'rcwm-t1:t2b-register', a
     resource wait before each of the 4 model loads, three distinct adapters, OFF != ON texts; the outputs validate with
     the analysis' own loaders (states == texts parsed by each side's parser); a rerun is a no-op
  5  adapter switching inside one PeftModel == loading each adapter alone (identical texts, every arm)
  6  crash in E2 (+ a stray .partial): the resume orphans the stage, re-anchors (recorded), finishes; every output equals
     the uninterrupted run byte for byte (adapters too: CPU training is deterministic)
  7  crash inside T_trace: the resume orphans train_trace.partial, retrains, finishes; adapters byte-identical to 4
  8  turn cap (D-22): nothing fits -> exit 5, nothing written; training deferred mid-way -> exit 5 then complete; eval
     stage E2 deferred -> exit 5, the next turn re-anchors and completes; outputs byte-identical to 4
  9  deterministic stops (exit 3): anchor failure (no dev output; anchor.json pass false; STOPPED.json; the rerun
     refuses); fingerprint mismatch; runtime drift (nothing trained); a recorded output changed; the one-waiter rule
     (not persisted, no lease): a live RCWM waiting marker or a live RCWM lock holder refuses, a dead marker, a reused
     pid and a live BWM holder do not; a live run lock refuses (not persisted), a stale one is reclaimed
 10  oracle-generator run on the FULL dev bank and chain bank (generate_texts replaced by per-side policies read from
     PEFT's own LoRA state: OFF unreduced + coin branch, plain mod-in-range + coin branch, trace gold, filler = plain's
     answers): the analysis on the run dir says GO; with filler = gold it says STOP (decision point 3)
 11  freeze tool on a fake root (the real code, docs and data): run() refuses before a freeze (not persisted); --freeze
     refuses without approval and with an output present; freezes; verifies; refuses a second freeze; an installed
     queue hook that is not the frozen draft byte for byte is flagged, the exact copy is not; a changed input fails
     verification and run() refuses
 12  plan / simulate_turns: the real bank fits one turn at both rates; a 3 h cap needs two turns; an impossible cap
     raises

  python -B scripts/t2br_test_tiny.py [--keep] [--no-record]
Writes docs/t2b/register/TINY_TEST.json (unless --no-record; refused once the screen is frozen).
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
os.environ["BWM_DEVICE"] = "cpu"                   # the vendored trainer's CPU lane: never calls into CUDA
os.environ["BWM_DUTY_RATIO"] = "0"
os.environ["BWM_YIELD_S"] = "0"
sys.dont_write_bytecode = True

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.synchronize = lambda *a, **k: None
torch.set_num_threads(2)
torch.Tensor.cuda = lambda self, *a, **k: self      # generation.generate_texts moves inputs with .cuda()

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor_bwm"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import transformers  # noqa: E402
from transformers import LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from peft import PeftModel  # noqa: E402

from rcwmt import env, generation, runtime  # noqa: E402
from bwmn import lm as LM  # noqa: E402  (the vendored trainer, imported after BWM_* are set)
import confirm2_gen as G2  # noqa: E402
import t2br_common as C  # noqa: E402
import t2br_make_data as MD  # noqa: E402
import t2br_run as R  # noqa: E402
_TORCH = sys.modules.pop("torch")                  # the analysis refuses to import next to torch (it never uses it)
try:
    import t2br_analysis as A  # noqa: E402
finally:
    sys.modules["torch"] = _TORCH


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models
runtime.gpu_temp_c = lambda: None                   # never spawn nvidia-smi
generation.breathe = lambda: None
generation.throttle_list = lambda: LogitsProcessorList([])
MAXNEW = 12
generation.max_tokens = lambda fam: MAXNEW
REAL_GENERATE = generation.generate_texts
REAL_CHAIN = R.t2br_chain
REAL_TRAIN_SFT = LM.train_sft
REAL_EST = R.stage_est_s

KEEP = "--keep" in sys.argv
RECORD = "--no-record" not in sys.argv
CFG = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
           tie_word_embeddings=True)
SEED_SPARSE = 2002
TINY_STEPS = 2
RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def build():
    cfg = Qwen3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(SEED_SPARSE)
    m = Qwen3ForCausalLM(cfg)
    g = torch.Generator().manual_seed(SEED_SPARSE)
    for _, p in m.named_parameters():
        if p.dim() == 2:
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


_TOK = []


def tok():
    if not _TOK:
        t = runtime.tokenizer()
        if t.pad_token_id is None:
            t.pad_token = "<|endoftext|>"
        _TOK.append(t)
    return _TOK[0]


class FakeRuntime:
    """Stands in for rcwmt.runtime under RealBackend: tiny models, recorded calls, no real lock."""

    def __init__(self, tmp: Path):
        self.LOCK_PATH = tmp / "fake_gpu.lock"
        self.WAITING = {"rcwm": tmp / "fake_gpu.waiting.rcwm.json", "bwm": tmp / "fake_gpu.waiting.bwm.json"}
        self.calls: list = []
        self.fp_override = None
        self.runtime_override: list = []

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
        return build(), {"param_fingerprint": self.param_fingerprint(None)}

    def param_fingerprint(self, model):
        return self.fp_override or (R.FP["sparse"] + "-tiny-standin")

    def tokenizer(self):
        return tok()

    def unload(self, _):
        gc.collect()
        self.calls.append(("unload",))

    def runtime_info(self):
        real = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu (tiny test)"}
        return {**real, **self.runtime_override.pop(0)} if self.runtime_override else real


class FakeLM:
    """The vendored trainer with only load_core replaced (tiny model + the real tokenizer)."""

    def __init__(self, rt):
        self.rt = rt
        self.attach_lora = LM.attach_lora
        self.save_adapter = LM.save_adapter

    def load_core(self, name):
        self.rt.calls.append(("train_load", name))
        return build(), tok()

    def train_sft(self, *a, **k):
        return LM.train_sft(*a, **k)


class TinyBackend(R.RealBackend):
    device = "cpu-tiny"

    def thermal_wait(self):
        pass

    def vram_allocated_mib(self):
        return 0.0


def run_cli(cfg, backend) -> int:
    """t2br_run.main()'s handling around run(): Stop -> exit 3 (persisted when it concerns the run)."""
    try:
        return R.run(cfg, backend)
    except R.Stop as e:
        print(str(e), flush=True)
        R.persist_stop(cfg.out, e)
        return 3


# ---------------------------------------------------------------- oracle generator
SIDE_OF = {v: k for k, v in R.ADAPTER_NAMES.items()}
POLICY = {}


def active_side(model) -> str:
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "disable_adapters"):
            if m.disable_adapters:
                return "off"
            return SIDE_OF[m.active_adapters[0]]
    return "off"


def oracle_generate(model, tok_, message_lists, max_new_tokens, **k):
    side = active_side(model)
    how, fmt = POLICY[side]
    count = C.token_counter()
    out = []
    for msgs in message_lists:
        u = json.loads(msgs[1]["content"])
        st = A._answer(u["before"], u["event"], how, "oracle")
        out.append(A._fmt(u["before"], u["event"], st, fmt, count))
    return out


# ---------------------------------------------------------------- fixtures
def subset_files(tmp: Path) -> tuple:
    dev = C.load_dev()
    keep, seen = [], {}
    for r in dev:
        if seen.get(r["stratum"], 0) < 2:
            keep.append(r)
            seen[r["stratum"]] = seen.get(r["stratum"], 0) + 1
    parents = C.load_chain_bank()
    ps = [next(p for p in parents if p["stratum"] == s) for s in C.CHAIN_STRATA]
    d, c = tmp / "dev_subset.jsonl", tmp / "chains_subset.jsonl"
    d.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    c.write_text("".join(json.dumps(p) + "\n" for p in ps), encoding="utf-8")
    return d, c


def anchor_fixture(tmp: Path, name: str, gen=None) -> tuple:
    """A few pilot TEST register chains and an OFF reference made by generation.chain on the tiny base alone."""
    chains = R.pilot_anchor_chains()[:4]
    generation.generate_texts = gen or REAL_GENERATE
    try:
        rr = generation.chain(build(), tok(), chains, "chain_one")
    finally:
        generation.generate_texts = REAL_GENERATE
    p = tmp / f"anchor_ref_{name}.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in R.chain_rows(chains, rr)), encoding="utf-8")
    return chains, p


def make_cfg(tmp: Path, name: str, dev: Path, chains: Path, anchor: tuple, **kw) -> R.Cfg:
    return R.Cfg(root=ROOT, out=tmp / "runs" / name, dev=dev, chains=chains, anchor_chains=anchor[0], anchor_ref=anchor[1],
                 require_freeze=False, steps=TINY_STEPS, **kw)


def outputs(out: Path) -> dict:
    fs = {}
    for st in R.STAGE_ORDER:
        for p in R.stage_required(st, out):
            if p.name not in ("metrics.json", "anchor.json"):     # records with timestamps / pids
                fs[R.rel_out(p, out)] = sha(p)
    return fs


# ---------------------------------------------------------------- checks
def t_formats():
    t = tok()
    tr = {a: C.load_jsonl(C.train_path(a)) for a in C.ARMS}
    dev = C.load_dev()
    bad_tpl = sum(r["prompt"] != t.apply_chat_template(C.canonical_messages(r["before"], r["event"]), tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
                  for r in tr["plain"])
    bad_tpl += sum(C.render_prompt(r["messages"]) != t.apply_chat_template(r["messages"], tokenize=False,
                                                                            add_generation_prompt=True, enable_thinking=False)
                   for r in dev)
    check("1 render_prompt == apply_chat_template on every training and dev prompt", bad_tpl == 0,
          f"{len(tr['plain']) + len(dev)} prompts")
    cnt = C.token_counter()
    n_at = lambda s: len(t(s, add_special_tokens=False).input_ids)   # noqa: E731
    bad = 0
    for a in C.ARMS:
        for r in tr[a]:
            bad += cnt(r["target"]) != n_at(r["target"]) or n_at(r["target"]) != r["n_target_tokens"]
    for r in dev:
        for a in C.ARMS:
            bad += cnt(r["targets"][a]) != n_at(r["targets"][a])
        bad += n_at(r["targets"]["filler"]) != n_at(r["targets"]["trace"])
    bad += sum(n_at(f["target"]) != n_at(x["target"]) for f, x in zip(tr["filler"], tr["trace"]))
    check("1 tokenizers counts == AutoTokenizer on every target; filler == trace tokens on every row (AutoTokenizer)",
          bad == 0, f"{3 * len(tr['plain']) + 3 * len(dev)} targets")


def t_surface_diversity(tmp: Path):
    sd_dir = tmp / "sd_data"
    sd_dir.mkdir()
    for a in C.ARMS:
        shutil.copy(C.train_path(a), sd_dir / C.train_path(a).name)
    shutil.copy(C.train_source(), sd_dir / C.train_source().name)
    old = C.DATA
    C.DATA = sd_dir
    try:
        MD.build_sd()
        base = {a: C.load_jsonl(C.train_path(a)) for a in C.ARMS}
        sd = {a: C.load_jsonl(C.train_path(a, "sd")) for a in C.ARMS}
        cfg = R.Cfg(train_variant="sd", steps=TINY_STEPS)
        picked = all(cfg.train_files[a] == sd_dir / f"t2br_train_sd_{a}.jsonl" for a in C.ARMS)
    finally:
        C.DATA = old
    same = all(len(base[a]) == len(sd[a]) and all(x["target"] == y["target"] and x["gold_after"] == y["gold_after"] and
                                                   x["id"] == y["id"] for x, y in zip(base[a], sd[a])) for a in C.ARMS)
    draws = [r["sd"] for r in sd["plain"]]
    used = {"paraphrase": set(d["paraphrase"] for d in draws), **{f: set(d[f] for d in draws) for f in C.SD_FIELDS}}
    all_used = used["paraphrase"] == set(range(len(C.SD_PARAPHRASES))) and all(used[f] == set(v) for f, v in C.SD_FIELDS.items())
    t = tok()
    back_ok = tpl_ok = 0
    for r in sd["plain"]:
        d = r["sd"]
        u = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
        inv = {d[f]: f for f in C.SD_FIELDS}
        ev = {inv.get(k, k): v for k, v in u[d["event"]].items()}
        back_ok += u[d["before"]] == r["before"] and ev == r["event"]
        msgs = C.sd_messages(r["before"], r["event"], d)
        tpl_ok += r["prompt"] == t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    differ = sum(x["prompt"] != y["prompt"] for x, y in zip(base["plain"], sd["plain"]))
    check("2 surface diversity: same rows/targets/golds; every paraphrase + field variant used; prompts parse back; "
          "chat template exact; Cfg(train_variant='sd') picks the files", same and all_used and picked and
          back_ok == len(draws) and tpl_ok == len(draws) and differ > 0.8 * len(draws),
          f"{differ}/{len(draws)} prompts differ from canonical")


def t_chain_equality():
    chains = [c for c in C.flat_chains(C.load_chain_bank())][:6]
    m = build()
    a = generation.chain(m, tok(), chains, "chain_one")
    b = REAL_CHAIN(m, tok(), chains, "plain")
    same_rand = all(x["texts"] == y["texts"] and [None if s is None else env.dumps(s) for s in x["states"]] ==
                    [None if s is None else env.dumps(s) for s in y["states"]] for x, y in zip(a, b))
    POLICY.clear()
    POLICY["off"] = ("offlike", "plain")
    generation.generate_texts = oracle_generate
    try:
        a = generation.chain(m, tok(), chains, "chain_one")
        b = REAL_CHAIN(m, tok(), chains, "plain")
    finally:
        generation.generate_texts = REAL_GENERATE
    same_or = all(x["texts"] == y["texts"] and x["states"] == y["states"] for x, y in zip(a, b))
    lived = sum(s is not None for x in b for s in x["states"])
    check("3 t2br_chain(plain) == generation.chain chain_one (random tiny model and an oracle that parses)",
          same_rand and same_or and lived == len(chains) * (C.H + 1), f"{len(chains)} chains")


def t_full(tmp, dev, chains):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    anchor = anchor_fixture(tmp, "full")
    cfg = make_cfg(tmp, "full", dev, chains, anchor)
    rc = run_cli(cfg, be)
    out = cfg.out
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    leases = [c for c in rt.calls if c[0] == "lease"]
    loads = [i for i, c in enumerate(rt.calls) if c[0] in ("load", "train_load")]
    waits_before = all(rt.calls[i - 1] == ("wait",) for i in loads)
    ads = {a: sha(out / f"train_{a}" / "adapter" / "adapter_model.safetensors") for a in C.ARMS}
    t_off = [json.loads(x)["text"] for x in (out / "onestep" / "onestep_off.jsonl").read_text(encoding="utf-8").splitlines()]
    t_on = {s: [json.loads(x)["text"] for x in (out / "onestep" / f"onestep_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in C.ARMS}
    check("4 full tiny run: exit 0, DONE.json, all stages recorded, one lease 'rcwm-t1:t2b-register', a wait before "
          "each of 4 loads", rc == 0 and (out / "DONE.json").exists() and set(prov["stages"]) == set(R.STAGE_ORDER)
          and leases == [("lease", R.LEASE_LABEL)] and len(loads) == 4 and waits_before,
          f"calls {[c[0] for c in rt.calls]}")
    check("4 three distinct adapters; every ON side differs from OFF", len(set(ads.values())) == 3 and
          all(t_on[s] != t_off for s in C.ARMS))
    devr, par = C.load_dev(dev), C.load_chain_bank(chains)
    texts, gens = A.load_run_outputs(out, devr, par)
    info = A.provenance_check(out, registered=False)
    check("4 outputs validate with the analysis loaders (states == texts parsed by each side's parser); provenance ok",
          len(texts) == 4 and info["turns"] == 1 and info["anchor_identical_states"] == 1.0)
    before = len(rt.calls)
    rc2 = run_cli(cfg, be)
    check("4 rerun is a no-op (exit 0, no lease)", rc2 == 0 and len(rt.calls) == before)
    return cfg, outputs(out), anchor


def t_switching(full_cfg, dev):
    rows = C.load_dev(dev)
    ok = True
    for a in C.ARMS:
        m = PeftModel.from_pretrained(build(), str(full_cfg.out / f"train_{a}" / "adapter"), adapter_name="only",
                                      is_trainable=False).eval()
        got = R.onestep_texts(m, tok(), rows)
        ref = [json.loads(x)["text"] for x in (full_cfg.out / "onestep" / f"onestep_{a}.jsonl").read_text(encoding="utf-8").splitlines()]
        ok &= got == ref
        del m
        gc.collect()
    check("5 adapter switching in one PeftModel == each adapter loaded alone (identical texts)", ok)


def t_crash_eval(tmp, dev, chains, ref_out, anchor):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = make_cfg(tmp, "crash_eval", dev, chains, anchor)
    calls = {"n": 0}

    def boom(model, tok_, ch, arm):
        if arm == "trace":
            calls["n"] += 1
            raise RuntimeError("injected crash in E2 (trace side)")
        return REAL_CHAIN(model, tok_, ch, arm)
    R.t2br_chain = boom
    crashed = False
    try:
        run_cli(cfg, be)
    except RuntimeError:
        crashed = True
    finally:
        R.t2br_chain = REAL_CHAIN
    released = rt.calls[-1][0] == "release" or ("release", R.LEASE_LABEL) in rt.calls
    (cfg.out / "chains").mkdir(parents=True, exist_ok=True)
    (cfg.out / "chains" / "chains_off.jsonl.partial").write_text("stray", encoding="utf-8")
    rc = run_cli(cfg, be)
    prov = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
    orph = list((cfg.out / "_orphans").glob("E2_*"))
    check("6 crash in E2: the lease is released; the resume orphans the stray partial, re-anchors (recorded), finishes; "
          "outputs byte-identical to the uninterrupted run (adapters included)",
          crashed and released and rc == 0 and len(orph) == 1 and "anchor_recheck" in prov["turns"][-1]
          and prov["turns"][-1]["anchor_recheck"]["pass"] and outputs(cfg.out) == ref_out,
          f"orphans {len(orph)}; differing {[k for k in ref_out if outputs(cfg.out).get(k) != ref_out[k]][:4]}")


def t_crash_train(tmp, dev, chains, ref_out, anchor):
    rt = FakeRuntime(tmp)
    lmf = FakeLM(rt)
    be = TinyBackend(rt, lmf)
    cfg = make_cfg(tmp, "crash_train", dev, chains, anchor)
    n = {"k": 0}

    def flaky(*a, **k):
        n["k"] += 1
        if n["k"] == 2:
            raise RuntimeError("injected crash inside T_trace")
        return LM.train_sft(*a, **k)
    lmf.train_sft = flaky
    crashed = False
    try:
        run_cli(cfg, be)
    except RuntimeError:
        crashed = True
    partial = (cfg.out / "train_trace.partial").exists()
    lmf.train_sft = LM.train_sft
    rc = run_cli(cfg, be)
    orph = list((cfg.out / "_orphans").glob("T_trace_*"))
    check("7 crash inside T_trace: .partial left; the resume orphans it, retrains, finishes; byte-identical to 4",
          crashed and partial and rc == 0 and len(orph) == 1 and outputs(cfg.out) == ref_out)


def t_deferral(tmp, dev, chains, ref_out, anchor):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = make_cfg(tmp, "defer_none", dev, chains, anchor, cap_h=0.4)
    rc = run_cli(cfg, be)
    prov = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
    check("8 cap: nothing fits -> exit 5, nothing trained or generated, every stage deferred",
          rc == 5 and not prov["stages"] and prov["turns"][0]["deferred"] == list(R.STAGE_ORDER)
          and not any(c[0] in ("load", "train_load") for c in rt.calls))
    est = {"T_plain": 60.0, "T_trace": 1e9, "T_filler": 60.0, "A0": 60.0, "E1": 60.0, "E2": 60.0}
    R.stage_est_s = lambda st, rate, n: est[st]
    try:
        cfg = make_cfg(tmp, "defer_mid", dev, chains, anchor)
        rc1 = run_cli(cfg, be)
        p1 = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
        est["T_trace"] = 60.0
        est["E2"] = 1e9
        rc2 = run_cli(cfg, be)
        p2 = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
        est["E2"] = 60.0
        rc3 = run_cli(cfg, be)
        p3 = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
    finally:
        R.stage_est_s = REAL_EST
    check("8 training deferred mid-way (exit 5), then E2 deferred (exit 5), then the next turn re-anchors and completes; "
          "byte-identical to 4",
          rc1 == 5 and set(p1["stages"]) == {"T_plain"} and rc2 == 5 and set(p2["stages"]) == {"T_plain", "T_trace",
                                                                                              "T_filler", "A0", "E1"}
          and p2["turns"][-1]["deferred"] == ["E2"] and rc3 == 0 and p3["turns"][-1]["anchor_recheck"]["pass"]
          and outputs(cfg.out) == ref_out, f"rc {rc1} {rc2} {rc3}")


def t_stops(tmp, dev, chains, anchor):
    import psutil
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    # anchor failure
    bad_ref = tmp / "anchor_ref_bad.jsonl"
    rows = [json.loads(x) for x in Path(anchor[1]).read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["states"] = [r["states"][0]] + ['{"X":0,"Y":0,"Z":0}'] * (len(r["states"]) - 1)
    bad_ref.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = make_cfg(tmp, "stop_anchor", dev, chains, (anchor[0], bad_ref))
    rc = run_cli(cfg, be)
    a0 = json.loads((cfg.out / "anchor" / "anchor.json").read_text(encoding="utf-8"))
    rc2 = run_cli(cfg, be)
    check("9 anchor failure: exit 3, anchor.json pass false, no dev output, STOPPED.json; the rerun refuses",
          rc == 3 and not a0["pass"] and not (cfg.out / "onestep").exists() and (cfg.out / "STOPPED.json").exists()
          and rc2 == 3)
    # fingerprint
    rt.fp_override = "deadbeef0000"
    cfg = make_cfg(tmp, "stop_fp", dev, chains, anchor)
    rc = run_cli(cfg, be)
    rt.fp_override = None
    check("9 fingerprint mismatch: exit 3, persisted, nothing trained", rc == 3 and (cfg.out / "STOPPED.json").exists()
          and not (cfg.out / "train_plain").exists())
    # runtime drift
    rt.runtime_override = [{"transformers": "9.9.9"}]
    n_tl = sum(c[0] == "train_load" for c in rt.calls)
    cfg = make_cfg(tmp, "stop_rt", dev, chains, anchor)
    rc = run_cli(cfg, be)
    check("9 runtime drift: exit 3, persisted, nothing loaded or trained", rc == 3 and (cfg.out / "STOPPED.json").exists()
          and sum(c[0] == "train_load" for c in rt.calls) == n_tl)
    # recorded output changed
    est = {"T_plain": 60.0, "T_trace": 60.0, "T_filler": 60.0, "A0": 60.0, "E1": 60.0, "E2": 1e9}
    R.stage_est_s = lambda st, rate, n: est[st]
    try:
        cfg = make_cfg(tmp, "stop_hash", dev, chains, anchor)
        rc1 = run_cli(cfg, be)
        f = cfg.out / "onestep" / "onestep_off.jsonl"
        f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        est["E2"] = 60.0
        rc2 = run_cli(cfg, be)
    finally:
        R.stage_est_s = REAL_EST
    check("9 a recorded output changed: exit 3, persisted", rc1 == 5 and rc2 == 3 and (cfg.out / "STOPPED.json").exists())
    # one-waiter rule
    parent = psutil.Process(os.getpid()).parent()
    ppid, pct = parent.pid, parent.create_time()
    n_lease = sum(c[0] == "lease" for c in rt.calls)
    rt.WAITING["rcwm"].write_text(json.dumps({"pid": ppid, "label": "rcwm-t1:other", "since": time.time()}), encoding="utf-8")
    cfg = make_cfg(tmp, "stop_waiter", dev, chains, anchor)
    rc_w = run_cli(cfg, be)
    rt.WAITING["rcwm"].write_text(json.dumps({"pid": 999999, "label": "rcwm-t1:dead", "since": time.time()}), encoding="utf-8")
    dead_ok = R.one_waiter_problem(be) is None
    rt.WAITING["rcwm"].write_text(json.dumps({"pid": ppid, "label": "rcwm-t1:reused", "since": pct - 100}), encoding="utf-8")
    reused_ok = R.one_waiter_problem(be) is None
    rt.WAITING["rcwm"].unlink()
    rt.LOCK_PATH.write_text(json.dumps({"label": "rcwm-t1:holder", "pid": ppid, "created": time.time()}), encoding="utf-8")
    holder = R.one_waiter_problem(be)
    rt.LOCK_PATH.write_text(json.dumps({"label": "bwm:peer", "pid": ppid, "created": time.time()}), encoding="utf-8")
    bwm_ok = R.one_waiter_problem(be) is None
    rt.LOCK_PATH.unlink()
    check("9 one-waiter rule: a live RCWM waiter refuses (exit 3, not persisted, no lease); a live RCWM holder refuses; a "
          "dead marker, a reused pid and a live BWM holder do not",
          rc_w == 3 and not (cfg.out / "STOPPED.json").exists() and sum(c[0] == "lease" for c in rt.calls) == n_lease
          and dead_ok and reused_ok and holder is not None and bwm_ok)
    # run lock
    cfg = make_cfg(tmp, "stop_lock", dev, chains, anchor)
    cfg.out.mkdir(parents=True)
    (cfg.out / ".run.lock").write_text(json.dumps({"pid": ppid, "create_time": pct}), encoding="utf-8")
    rc_l = run_cli(cfg, be)
    live_ok = rc_l == 3 and not (cfg.out / "STOPPED.json").exists()
    (cfg.out / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
    est = {s: 60.0 for s in R.STAGE_ORDER}
    est["T_plain"] = 1e9
    R.stage_est_s = lambda st, rate, n: est[st]
    try:
        rc_s = run_cli(cfg, be)
    finally:
        R.stage_est_s = REAL_EST
    check("9 run lock: a live holder refuses (exit 3, not persisted); a stale one is reclaimed",
          live_ok and rc_s == 5 and not (cfg.out / ".run.lock").exists())


def t_oracle(tmp):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    POLICY.clear()
    POLICY.update({"off": ("offlike", "plain"), "plain": ("plainlike", "plain"), "trace": ("gold", "trace"),
                   "filler": ("plainlike", "filler")})
    anchor = anchor_fixture(tmp, "oracle", gen=oracle_generate)     # restores the real generator when done
    generation.generate_texts = oracle_generate
    try:
        cfg = make_cfg(tmp, "oracle_go", C.DEV, C.CHAINS, anchor)
        rc = run_cli(cfg, be)
        POLICY["filler"] = ("gold", "filler")
        cfg2 = make_cfg(tmp, "oracle_stop", C.DEV, C.CHAINS, anchor)
        rc2 = run_cli(cfg2, be)
    finally:
        generation.generate_texts = REAL_GENERATE
    res = A.analyse_runs([cfg.out], registered=False, out_dir=tmp / "analysis_go", n_boot=1000)
    res2 = A.analyse_runs([cfg2.out], registered=False, out_dir=tmp / "analysis_stop", n_boot=1000)
    v, v2 = res["verdicts"]["s0"], res2["verdicts"]["s0"]
    check("10 oracle run on the FULL dev + chain bank: the analysis says GO; with filler = gold, STOP (decision point 3)",
          rc == 0 and rc2 == 0 and v == "GO" and v2.startswith("STOP (decision point 3)")
          and (tmp / "analysis_go" / "T2BR_RESULTS_s0.md").exists(), f"{v} | {v2}")
    pl = res["per_seed"]["s0"]["chains"]["ALL"]["product_law"]["trace"]
    check("10 oracle run: gold trace side regret k4 0, product law predicted == observed == 1",
          res["per_seed"]["s0"]["chains"]["ALL"]["k"]["4"]["sides"]["trace"]["regret"]["mean"] == 0.0
          and pl["predicted_k4_exact_product"] == 1.0 and pl["observed_k4_exact"] == 1.0)


def t_freeze(tmp):
    fake = tmp / "fake_root"
    rels = R.freeze_inputs(ROOT)
    for rel in rels:
        if not (ROOT / rel).is_file():               # TINY_TEST.json is written by this test (synthetic copy below)
            continue
        dst = fake / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    tt = {"ok": True, "code_sha256": {r: sha(fake / r) for r in R.TINY_CODE}}
    (fake / "docs/t2b/register/TINY_TEST.json").write_text(json.dumps(tt), encoding="utf-8")
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = R.Cfg(root=fake, out=tmp / "runs" / "frz", steps=TINY_STEPS)
    rc0 = run_cli(cfg, be)
    no_persist = not (cfg.out / "STOPPED.json").exists()
    errs = []
    try:
        R.freeze("", fake)
    except R.Stop as e:
        errs.append(("no approval", str(e)))
    (fake / "runs" / "t2b_register" / "s0").mkdir(parents=True)
    try:
        R.freeze("tiny test approval 2026-10-09", fake)
    except R.Stop as e:
        errs.append(("output present", str(e)))
    shutil.rmtree(fake / "runs" / "t2b_register")
    rec = R.freeze("tiny test approval 2026-10-09 (fake root)", fake)
    ok1, e1, _ = R.verify_freeze(fake)
    try:
        R.freeze("tiny test approval again", fake)
    except R.Stop as e:
        errs.append(("second freeze", str(e)))
    fz = json.loads((fake / R.FREEZE_REL).read_text(encoding="utf-8"))
    hook = fake / "scripts" / "queue_hooks" / "80_t2br.sh"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text((fake / R.HOOK_DRAFT_REL).read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    hook_bad = R.hook_problem(fake, fz) is not None
    shutil.copy(fake / R.HOOK_DRAFT_REL, hook)
    hook_ok = R.hook_problem(fake, fz) is None
    p = fake / "data" / C.DEV.name
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    ok2, e2, _ = R.verify_freeze(fake)
    cfg2 = R.Cfg(root=fake, out=tmp / "runs" / "frz2", steps=TINY_STEPS)
    rc2 = run_cli(cfg2, be)
    check("11 freeze tool (fake root): run() refuses before a freeze (not persisted); refuses without approval and with "
          "an output present; freezes; verifies; refuses a second freeze; an installed hook that is not the draft byte "
          "for byte is flagged, the exact copy is not; a changed input fails verification; run() refuses", rc0 == 3 and no_persist and [e[0] for e in errs] == ["no approval", "output present", "second freeze"]
          and rec["go_rule_text"] == C.GO_RULE_TEXT and ok1 and not e1 and not ok2 and rc2 == 3 and hook_bad and hook_ok
          and not any(c[0] == "lease" for c in rt.calls), f"{[e[0] for e in errs]} {e2[:1]}")


def t_plan():
    n = R.counts(R.Cfg())
    t_p = R.simulate_turns(R.RATE_GENS_PER_S, n)
    t_o = R.simulate_turns(R.OBSERVED_RATE, n)
    t_3 = R.simulate_turns(R.RATE_GENS_PER_S, n, cap_h=3.0)
    raised = False
    try:
        R.simulate_turns(R.RATE_GENS_PER_S, n, cap_h=0.4)
    except RuntimeError:
        raised = True
    check("12 plan: one turn at the planning and observed rates; a 3 h cap needs 2 turns (eval resumes with a re-anchor); "
          "an impossible cap raises", len(t_p) == 1 and len(t_o) == 1 and len(t_3) == 2
          and t_3[1]["stages"][0] in ("A0", "A0 (re-anchor)") and raised,
          f"{t_p[0]['projected_h']} h / {t_o[0]['projected_h']} h; 3 h cap: {t_3}")


def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="t2br_tiny_"))
    print(f"[t2br-tiny] temp dir {tmp}", flush=True)
    ok = False
    try:
        t_formats()
        t_surface_diversity(tmp)
        t_chain_equality()
        dev, chains = subset_files(tmp)
        full_cfg, ref_out, anchor = t_full(tmp, dev, chains)
        t_switching(full_cfg, dev)
        t_crash_eval(tmp, dev, chains, ref_out, anchor)
        t_crash_train(tmp, dev, chains, ref_out, anchor)
        t_deferral(tmp, dev, chains, ref_out, anchor)
        t_stops(tmp, dev, chains, anchor)
        t_oracle(tmp)
        t_freeze(tmp)
        t_plan()
        ok = all(r["ok"] for r in RESULTS)
    finally:
        try:
            import psutil
            peak = getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2 ** 20
        except Exception:
            peak = -1
        print(f"[t2br-tiny] {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass in {time.time() - t0:.0f}s; "
              f"peak working set {peak:.0f} MB", flush=True)
        if RECORD:
            if (ROOT / R.FREEZE_REL).exists():
                print("SCREEN_FREEZE.json exists: TINY_TEST.json is a frozen input and is not rewritten")
            else:
                C.DOCS.mkdir(parents=True, exist_ok=True)
                (C.DOCS / "TINY_TEST.json").write_text(json.dumps({
                    "ok": ok, "n_checks": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "checks": RESULTS,
                    "seconds": round(time.time() - t0, 1), "peak_working_set_mb": round(peak, 1), "threads": 2,
                    "code_sha256": {r: sha(ROOT / r) for r in R.TINY_CODE},
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n", encoding="utf-8")
        if KEEP:
            print(f"[t2br-tiny] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
