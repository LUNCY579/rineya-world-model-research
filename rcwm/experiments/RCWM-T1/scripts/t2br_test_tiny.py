"""CPU test of scripts/t2br_run.py (T2b-register runner, revision 5), its hand-off to scripts/t2br_analysis.py, and the
data / format / surface plumbing, with tiny random Qwen3 stand-ins. No real model is loaded
(AutoModelForCausalLM.from_pretrained is disabled), no GPU is touched, no lease is taken; outputs go to a temporary
directory only. Threads 2. One process.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary (151,936) and the real
tokenizer (read-only). The runner's own RealBackend code runs on a fake rcwmt.runtime (lease, loads, fingerprints) and
on the REAL vendored trainer (vendor_bwm/bwmn/lm.py: attach_lora, train_sft, save_adapter; only load_core returns the
tiny model), with the recipe's step count reduced to 2. Pacing-only deviations: generation.breathe / throttle_list are
no-ops, tensors stay on CPU (Tensor.cuda is the identity), decode cap 12 tokens, the thermal flag is not waited on,
BWM_DEVICE=cpu.

Checks
  1  formats: render_prompt == tok.apply_chat_template on every training prompt (canonical and surface-diversity), every
     dev prompt and every held-out-surface prompt; the `tokenizers` counts equal AutoTokenizer's on every target;
     amendment 4b with AutoTokenizer: ONE fill string per action kind over train + dev, every filler target at least
     as long as that row's trace target (the residual >= 0), and (revision-4 definition) the SHORTEST filler target of
     every kind at least as long as the LONGEST trace target of that kind, with one unit less failing that
  2  surfaces (C2 / C3 proof on the files the runner trains and evaluates): every 'sd' training prompt decodes
     (t2br_common.read_prompt) to its row and recorded surface and uses NO held-out paraphrase, field name or key order
     (and the detector does flag a held-out rendering of the same rows); same rows / targets / golds as the canonical
     rendering; every held-out-surface prompt (primary + identification strata, 580) decodes to its dev row and uses ONLY
     held-out factors; Cfg()
     trains on data/t2br_train_sd_<arm>.jsonl by default and run() refuses the canonical variant (exit 3, not
     persisted, no lease)
  3  t2br_chain(plain parser) == rcwmt.generation.chain('chain_one') on register chains (texts and states), on the
     random tiny model and on an oracle generator whose answers parse
  4  full tiny run on a subset bank (+ its held-out-surface rows): exit 0, DONE.json, every stage recorded (E1H
     included), ONE lease 'rcwm-t1:t2b-register', a resource wait before each of the 4 model loads, three distinct
     adapters, OFF != ON texts; the outputs validate with the analysis' own loaders (states == texts parsed by each
     side's parser; held-out-surface ids / hs_index == the bank); a rerun is a no-op
  5  adapter switching inside one PeftModel == loading each adapter alone (identical texts, every arm, both renderings)
  6  crash in E2 (+ a stray .partial): the resume orphans the stage, re-anchors (recorded), finishes; every output equals
     the uninterrupted run byte for byte (adapters too: CPU training is deterministic)
  7  crash inside T_trace: the resume orphans train_trace.partial, retrains, finishes; adapters byte-identical to 4
  8  turn cap (D-22): nothing fits -> exit 5, nothing written; training deferred mid-way -> exit 5; the held-out-surface
     stage E1H deferred -> exit 5, the next turn re-anchors and runs it; E2 deferred -> exit 5, the next turn re-anchors
     and completes; outputs byte-identical to 4. The gate is re-checked AFTER the resource / thermal wait: a wait that
     uses up the turn before a training load or before the eval load defers (exit 5, nothing loaded), and the resumed
     run completes byte-identically; --cap-h above 4.5 is refused (exit 3, not persisted, no lease)
  9  deterministic stops (exit 3): anchor failure (no dev output; anchor.json pass false; STOPPED.json; the rerun
     refuses); fingerprint mismatch; runtime drift (nothing trained); a recorded output changed; the one-waiter rule
     (not persisted, no lease): a live RCWM waiting marker or a live RCWM lock holder refuses, a dead marker, a reused
     pid and a live BWM holder do not; a live run lock refuses (not persisted), a stale one is reclaimed
 10  oracle-generator runs on the FULL dev bank, held-out-surface bank and chain bank (generate_texts replaced by
     per-side policies read from PEFT's own LoRA state and from the prompt's decoded surface: OFF unreduced + coin
     branch, plain mod-in-range + coin branch, trace gold, filler = plain's answers in the 4b format): the analysis on
     the run dir says GO; with filler = gold it says STOP (decision point 3); with a trace arm that is gold in Titan's
     frame but plain-like on the held-out surface it says NO-GO on exactly G1 / G2's held-out parts; with a plain arm
     weak on every out-of-range row and a trace arm following the tens-digit shortcut it says NO-GO on exactly the
     predicate clauses (revision 5)
 11  freeze tool on a fake root (the real code, docs and data): run() refuses before a freeze (not persisted); --freeze
     refuses without approval and with an output present; freezes; verifies; refuses a second freeze; an installed
     queue hook that is not the frozen draft byte for byte is flagged, the exact copy is not; a changed input fails
     verification and run() refuses
 12  plan / simulate_turns: the real bank fits one turn at the observed rate and two at the planning rate (the second
     re-anchors, then E2); a 3 h cap also needs two turns; an impossible cap raises

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


def write_lf(p: Path, text: str) -> None:
    Path(p).write_text(text, encoding="utf-8", newline="\n")


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
        write_lf(self.LOCK_PATH, json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}))
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
SEEN_SURFACES = {"canonical": 0, "heldout": 0}


def active_side(model) -> str:
    for m in model.modules():
        if hasattr(m, "lora_A") and hasattr(m, "disable_adapters"):
            if m.disable_adapters:
                return "off"
            return SIDE_OF[m.active_adapters[0]]
    return "off"


def oracle_generate(model, tok_, message_lists, max_new_tokens, **k):
    """Per-side policy; the prompt is decoded in ANY surface (read_prompt), and a policy's third entry (if any) is used
    on held-out-surface prompts."""
    side = active_side(model)
    pol = POLICY[side]
    out = []
    for msgs in message_lists:
        d = C.read_prompt(msgs)
        hp = C.surface_heldout_parts(d["surface"])
        held = hp["paraphrase"] or bool(hp["names"]) or bool(hp["orders"])
        SEEN_SURFACES["heldout" if held else "canonical"] += 1
        how = pol[2] if held and len(pol) > 2 else pol[0]
        st = A._answer(d["before"], d["event"], how, "oracle")
        out.append(A._fmt(d["before"], d["event"], st, pol[1]))
    return out


# ---------------------------------------------------------------- fixtures
def subset_files(tmp: Path) -> tuple:
    dev = C.load_dev()
    keep, seen = [], {}
    for r in dev:
        if seen.get(r["stratum"], 0) < 2:
            keep.append(r)
            seen[r["stratum"]] = seen.get(r["stratum"], 0) + 1
    ids = {r["id"] for r in keep}
    hs = [h for h in C.load_dev_hs() if h["id"] in ids]
    parents = C.load_chain_bank()
    ps = [next(p for p in parents if p["stratum"] == s) for s in C.CHAIN_STRATA]
    d, c, h = tmp / "dev_subset.jsonl", tmp / "chains_subset.jsonl", tmp / "dev_hs_subset.jsonl"
    write_lf(d, "".join(json.dumps(r) + "\n" for r in keep))
    write_lf(c, "".join(json.dumps(p) + "\n" for p in ps))
    write_lf(h, "".join(json.dumps(x) + "\n" for x in hs))
    return d, c, h


def anchor_fixture(tmp: Path, name: str, gen=None) -> tuple:
    """A few pilot TEST register chains and an OFF reference made by generation.chain on the tiny base alone."""
    chains = R.pilot_anchor_chains()[:4]
    generation.generate_texts = gen or REAL_GENERATE
    try:
        rr = generation.chain(build(), tok(), chains, "chain_one")
    finally:
        generation.generate_texts = REAL_GENERATE
    p = tmp / f"anchor_ref_{name}.jsonl"
    write_lf(p, "".join(json.dumps(x) + "\n" for x in R.chain_rows(chains, rr)))
    return chains, p


def make_cfg(tmp: Path, name: str, files: tuple, anchor: tuple, **kw) -> R.Cfg:
    dev, chains, dev_hs = files
    return R.Cfg(root=ROOT, out=tmp / "runs" / name, dev=dev, chains=chains, dev_hs=dev_hs, anchor_chains=anchor[0],
                 anchor_ref=anchor[1], require_freeze=False, steps=TINY_STEPS, **kw)


def outputs(out: Path) -> dict:
    fs = {}
    for st in R.STAGE_ORDER:
        for p in R.stage_required(st, out):
            if p.name not in ("metrics.json", "anchor.json"):     # records with timestamps / pids
                fs[R.rel_out(p, out)] = sha(p)
    return fs


def read_texts(p: Path) -> list:
    return [json.loads(x)["text"] for x in p.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- checks
def t_formats():
    t = tok()
    tpl = lambda msgs: t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,  # noqa: E731
                                             enable_thinking=False)
    base = {a: C.load_jsonl(C.train_path(a, "base")) for a in C.ARMS}
    sd = {a: C.load_jsonl(C.train_path(a, "sd")) for a in C.ARMS}
    dev = C.load_dev()
    hs = C.load_dev_hs()
    bad_tpl = sum(r["prompt"] != tpl(C.canonical_messages(r["before"], r["event"])) for r in base["plain"])
    bad_tpl += sum(r["prompt"] != tpl(C.surface_messages(r["before"], r["event"], r["surface"])) for r in sd["plain"])
    bad_tpl += sum(C.render_prompt(r["messages"]) != tpl(r["messages"]) for r in dev)
    bad_tpl += sum(C.render_prompt(h["messages"]) != tpl(h["messages"]) for h in hs)
    n_p = 2 * len(base["plain"]) + len(dev) + len(hs)
    check("1 render_prompt == apply_chat_template on every training prompt (canonical + surface-diversity), dev prompt "
          "and held-out-surface prompt", bad_tpl == 0, f"{n_p} prompts")
    cnt = C.token_counter()
    n_at = lambda s: len(t(s, add_special_tokens=False).input_ids)   # noqa: E731
    bad = 0
    for a in C.ARMS:
        for r, s in zip(base[a], sd[a]):
            bad += cnt(r["target"]) != n_at(r["target"]) or n_at(r["target"]) != r["n_target_tokens"] \
                or s["target"] != r["target"] or s["n_target_tokens"] != r["n_target_tokens"]
            bad += n_at(r["prompt"]) != r["n_prompt_tokens"] or n_at(s["prompt"]) != s["n_prompt_tokens"]
    for r in dev:
        for a in C.ARMS:
            bad += cnt(r["targets"][a]) != n_at(r["targets"][a]) or n_at(r["targets"][a]) != r["n_target_tokens"][a]
    check("1 tokenizers counts == AutoTokenizer on every target (and the recorded prompt / target token counts)",
          bad == 0, f"{3 * len(base['plain']) + 3 * len(dev)} targets")
    fills, neg_res = {}, 0
    for r, x in zip(base["filler"], base["trace"]):
        fills.setdefault(r["event"]["kind"], set()).add(json.loads(r["target"])["fill"])
        neg_res += n_at(r["target"]) < n_at(x["target"]) or n_at(r["target"]) - n_at(x["target"]) != r["fill_residual"]
    for r in dev:
        fills.setdefault(r["event"]["kind"], set()).add(json.loads(r["targets"]["filler"])["fill"])
        neg_res += n_at(r["targets"]["filler"]) < n_at(r["targets"]["trace"])
    one = all(v == {C.FILL_UNIT * C.FILL_UNITS[k]} for k, v in fills.items()) and set(fills) == set(C.KINDS)
    check("1 amendment 4b (AutoTokenizer): ONE fill string per action kind over train + dev (fill length identical on "
          "every row of a kind); every filler target >= its row's trace target; recorded residuals exact",
          one and neg_res == 0, {k: len(next(iter(v))) // 2 for k, v in fills.items()})
    # revision-4 definition, with AutoTokenizer: per kind, the SHORTEST filler target >= the LONGEST trace target (train +
    # dev + the chains' gold steps), and FILL_UNITS is the smallest such constant (one unit less fails)
    pairs = [(r["before"], r["event"]) for r in base["plain"]] + [(r["before"], r["event"]) for r in dev] + \
        [(s_["before"], s_["event"]) for s_ in MD.chain_steps(C.load_chain_bank())]
    longest, afters = {}, {}
    for b, e in pairs:
        tg = C.targets(b, e)
        longest[e["kind"]] = max(longest.get(e["kind"], 0), n_at(tg["trace"]))
        afters.setdefault(e["kind"], {})[tg["after"]] = json.loads(tg["after"])
    per = {}
    for k in C.KINDS:
        sh = min(n_at(C.filler_text(C.FILL_UNITS[k], a)) for a in afters[k].values())
        sh1 = min(n_at(C.filler_text(C.FILL_UNITS[k] - 1, a)) for a in afters[k].values())
        per[k] = (C.FILL_UNITS[k], sh, longest[k], sh1 < longest[k])
    check("1 amendment 4b, revision-4 definition (AutoTokenizer): per kind the SHORTEST filler target >= the LONGEST trace "
          "target over train + dev + chain gold steps, and FILL_UNITS is the smallest such constant",
          all(v[1] >= v[2] and v[3] for v in per.values()), per)


def t_surfaces(tmp: Path):
    base = {a: C.load_jsonl(C.train_path(a, "base")) for a in C.ARMS}
    sd = {a: C.load_jsonl(C.train_path(a, "sd")) for a in C.ARMS}
    dev = C.load_dev()
    hs = C.load_dev_hs()
    same = all(len(base[a]) == len(sd[a]) and all(
        x["target"] == y["target"] and x["gold_after"] == y["gold_after"] and x["id"] == y["id"]
        and x["before"] == y["before"] and x["event"] == y["event"] for x, y in zip(base[a], sd[a])) for a in C.ARMS)
    held = dec_bad = 0
    used = {"paraphrase": set(), "names": set(), "state": set(), "event": set()}
    for r in sd["plain"]:
        d = C.read_prompt(MD.messages_of_prompt(r["prompt"]))
        dec_bad += env.dumps(d["before"]) != env.dumps(r["before"]) or env.dumps(d["event"]) != env.dumps(r["event"]) \
            or d["surface"] != r["surface"]
        hp = C.surface_heldout_parts(d["surface"])
        held += hp["paraphrase"] or bool(hp["names"]) or bool(hp["orders"])
        used["paraphrase"].add(d["surface"]["paraphrase"])
        used["names"] |= set(d["surface"]["names"].values())
        used["state"].add(tuple(d["surface"]["orders"]["state"]))
        used["event"].add(tuple(d["surface"]["orders"]["event"]))
    same_prompts = all(x["prompt"] == y["prompt"] == z["prompt"] for x, y, z in zip(sd["plain"], sd["trace"], sd["filler"]))
    # negative control: the detector flags a held-out rendering of the same training rows
    flagged = 0
    for j, r in enumerate(sd["plain"][:30]):
        d = C.read_prompt(C.surface_messages(r["before"], r["event"], C.heldout_surface(j)))
        hp = C.surface_heldout_parts(d["surface"])
        flagged += hp["paraphrase"] and hp["names"] == sorted(C.SD_FIELDS) and hp["orders"] == sorted(C.OBJECT_KEYS)
    hb = A.heldout_bank_check(dev, hs)
    check("2 C2 proof: every 'sd' training prompt decodes to its row and recorded surface and uses NO held-out "
          "paraphrase / field name / key order (the detector flags a held-out rendering of the same rows); same rows, "
          "targets and golds as the canonical rendering; identical prompts across arms",
          same and dec_bad == 0 and held == 0 and flagged == 30 and same_prompts
          and used["paraphrase"] == set(C.TRAIN_PARAPHRASES) and used["names"] == set(C.TRAIN_NAME_IDX)
          and used["state"] == set(C.TRAIN_ORDERS["state"]) and not (used["event"] & set(C.HELDOUT_ORDERS["event"])),
          f"{len(sd['plain'])} prompts; event orders used {len(used['event'])}")
    check("2 C3 proof: every held-out-surface prompt decodes to its dev row (primary strata + cond_sametens + "
          "cond_oor_edge; same id, gold; dev order) and uses ONLY held-out factors (paraphrase 4, names 4, held-out key "
          "orders)", hb["ok"] and hb["n"] == sum(C.DEV_N[s] for s in C.HS_STRATA) == 580, hb)
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    picked = all(R.Cfg().train_files[a] == C.DATA / f"t2br_train_sd_{a}.jsonl" for a in C.ARMS)
    cfg = R.Cfg(out=tmp / "runs" / "variant_base", train_variant="base", require_freeze=False, steps=TINY_STEPS)
    rc = run_cli(cfg, be)
    check("2 C1: Cfg() trains on data/t2br_train_sd_<arm>.jsonl; run() refuses the canonical variant (exit 3, not "
          "persisted, no lease)", picked and rc == 3 and not (cfg.out / "STOPPED.json").exists()
          and not any(c[0] == "lease" for c in rt.calls))


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


def t_full(tmp, files):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    anchor = anchor_fixture(tmp, "full")
    cfg = make_cfg(tmp, "full", files, anchor)
    rc = run_cli(cfg, be)
    out = cfg.out
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    leases = [c for c in rt.calls if c[0] == "lease"]
    loads = [i for i, c in enumerate(rt.calls) if c[0] in ("load", "train_load")]
    waits_before = all(rt.calls[i - 1] == ("wait",) for i in loads)
    ads = {a: sha(out / f"train_{a}" / "adapter" / "adapter_model.safetensors") for a in C.ARMS}
    t_off = read_texts(out / "onestep" / "onestep_off.jsonl")
    t_on = {s: read_texts(out / "onestep" / f"onestep_{s}.jsonl") for s in C.ARMS}
    meta = json.loads((out / "train_trace" / "metrics.json").read_text(encoding="utf-8"))
    check("4 full tiny run: exit 0, DONE.json, all stages recorded (E1H included), one lease 'rcwm-t1:t2b-register', a "
          "wait before each of 4 loads; the arms trained on the 'sd' files",
          rc == 0 and (out / "DONE.json").exists() and set(prov["stages"]) == set(R.STAGE_ORDER) and "E1H" in prov["stages"]
          and leases == [("lease", R.LEASE_LABEL)] and len(loads) == 4 and waits_before
          and meta["args"]["rows"].endswith("t2br_train_sd_trace.jsonl") and prov["train_variant"] == "sd",
          f"calls {[c[0] for c in rt.calls]}")
    check("4 three distinct adapters; every ON side differs from OFF", len(set(ads.values())) == 3 and
          all(t_on[s] != t_off for s in C.ARMS))
    devr, par, hsr = C.load_dev(files[0]), C.load_chain_bank(files[1]), C.load_dev_hs(files[2])
    texts, gens, hs_texts = A.load_run_outputs(out, devr, par, hsr)
    info = A.provenance_check(out, registered=False)
    crlf = sum(b"\r\n" in p.read_bytes() for p in out.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".txt")
               and p.parent.name != "adapter")       # adapter/ is written by the vendored trainer + PEFT, not the runner
    check("4 outputs validate with the analysis loaders (states == texts parsed by each side's parser; held-out-surface "
          "ids / hs_index == the bank); provenance ok; every text file the runner writes is LF-only",
          len(texts) == 4 and len(hs_texts) == 4 and all(len(v) == len(hsr) for v in hs_texts.values())
          and info["turns"] == 1 and info["anchor_identical_states"] == 1.0 and crlf == 0, f"{len(hsr)} HS rows")
    before = len(rt.calls)
    rc2 = run_cli(cfg, be)
    check("4 rerun is a no-op (exit 0, no lease)", rc2 == 0 and len(rt.calls) == before)
    return cfg, outputs(out), anchor


def t_switching(full_cfg, files):
    rows = C.load_dev(files[0])
    hsr = C.load_dev_hs(files[2])
    ok = True
    for a in C.ARMS:
        m = PeftModel.from_pretrained(build(), str(full_cfg.out / f"train_{a}" / "adapter"), adapter_name="only",
                                      is_trainable=False).eval()
        ok &= R.onestep_texts(m, tok(), rows) == read_texts(full_cfg.out / "onestep" / f"onestep_{a}.jsonl")
        ok &= R.onestep_texts(m, tok(), hsr) == read_texts(full_cfg.out / "onestep" / f"onestep_hs_{a}.jsonl")
        del m
        gc.collect()
    check("5 adapter switching in one PeftModel == each adapter loaded alone (identical texts, both renderings)", ok)


def t_crash_eval(tmp, files, ref_out, anchor):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = make_cfg(tmp, "crash_eval", files, anchor)
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
    write_lf(cfg.out / "chains" / "chains_off.jsonl.partial", "stray")
    rc = run_cli(cfg, be)
    prov = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
    orph = list((cfg.out / "_orphans").glob("E2_*"))
    check("6 crash in E2: the lease is released; the resume orphans the stray partial, re-anchors (recorded), finishes; "
          "outputs byte-identical to the uninterrupted run (adapters included)",
          crashed and released and rc == 0 and len(orph) == 1 and "anchor_recheck" in prov["turns"][-1]
          and prov["turns"][-1]["anchor_recheck"]["pass"] and outputs(cfg.out) == ref_out,
          f"orphans {len(orph)}; differing {[k for k in ref_out if outputs(cfg.out).get(k) != ref_out[k]][:4]}")


def t_crash_train(tmp, files, ref_out, anchor):
    rt = FakeRuntime(tmp)
    lmf = FakeLM(rt)
    be = TinyBackend(rt, lmf)
    cfg = make_cfg(tmp, "crash_train", files, anchor)
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


def t_deferral(tmp, files, ref_out, anchor):
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = make_cfg(tmp, "defer_none", files, anchor, cap_h=0.4)
    rc = run_cli(cfg, be)
    prov = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
    check("8 cap: nothing fits -> exit 5, nothing trained or generated, every stage deferred",
          rc == 5 and not prov["stages"] and prov["turns"][0]["deferred"] == list(R.STAGE_ORDER)
          and not any(c[0] in ("load", "train_load") for c in rt.calls))
    est = {"T_plain": 60.0, "T_trace": 1e9, "T_filler": 60.0, "A0": 60.0, "E1": 60.0, "E1H": 60.0, "E2": 60.0}
    R.stage_est_s = lambda st, rate, n: est[st]
    rcs, provs = [], []
    try:
        cfg = make_cfg(tmp, "defer_mid", files, anchor)
        for upd in ({}, {"T_trace": 60.0, "E1H": 1e9}, {"E1H": 60.0, "E2": 1e9}, {"E2": 60.0}):
            est.update(upd)
            rcs.append(run_cli(cfg, be))
            provs.append(json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8")))
    finally:
        R.stage_est_s = REAL_EST
    p1, p2, p3, p4 = provs
    check("8 training deferred mid-way (exit 5); E1H deferred (exit 5), the next turn re-anchors and runs it; E2 deferred "
          "(exit 5), the next turn re-anchors and completes; byte-identical to 4",
          rcs == [5, 5, 5, 0] and set(p1["stages"]) == {"T_plain"}
          and set(p2["stages"]) == {"T_plain", "T_trace", "T_filler", "A0", "E1"} and p2["turns"][-1]["deferred"] == ["E1H", "E2"]
          and p3["turns"][-1]["stages_run"] == ["E1H"] and p3["turns"][-1]["anchor_recheck"]["pass"]
          and p3["turns"][-1]["deferred"] == ["E2"] and p4["turns"][-1]["stages_run"] == ["E2"]
          and p4["turns"][-1]["anchor_recheck"]["pass"] and outputs(cfg.out) == ref_out, f"rc {rcs}")


class WaitEatsTurn(TinyBackend):
    """A resource / thermal wait that uses up the turn at the n-th wait (R.can_start then refuses every stage)."""

    def __init__(self, rt, lm, at: int):
        super().__init__(rt, lm)
        self.at, self.n_wait, self.eaten = at, 0, False

    def wait(self):
        super().wait()
        self.n_wait += 1
        self.eaten = self.eaten or self.n_wait == self.at


def t_wait_recheck(tmp, files, ref_out, anchor):
    real_can_start = R.can_start
    res = {}
    for name, at in (("wait_train", 1), ("wait_eval", 4)):
        rt = FakeRuntime(tmp)
        be = WaitEatsTurn(rt, FakeLM(rt), at)
        R.can_start = lambda elapsed, est, cap, be=be: (not be.eaten) and real_can_start(elapsed, est, cap)
        try:
            cfg = make_cfg(tmp, name, files, anchor)
            rc1 = run_cli(cfg, be)
            p1 = json.loads((cfg.out / "provenance.json").read_text(encoding="utf-8"))
        finally:
            R.can_start = real_can_start
        loads = [c[0] for c in rt.calls if c[0] in ("load", "train_load")]
        rc2 = run_cli(cfg, TinyBackend(rt, FakeLM(rt)))
        res[name] = (rc1, p1["turns"][0].get("deferred_after_wait"), sorted(p1["stages"]), loads, rc2, outputs(cfg.out) == ref_out)
    wt, we = res["wait_train"], res["wait_eval"]
    ok = wt[0] == 5 and wt[1] == "T_plain" and wt[2] == [] and wt[3] == [] and wt[4] == 0 and wt[5] \
        and we[0] == 5 and we[1] == "eval load" and we[2] == ["T_filler", "T_plain", "T_trace"] \
        and we[3] == ["train_load"] * 3 and we[4] == 0 and we[5]
    check("8 the D-22 gate is re-checked after the resource wait: a wait that uses up the turn defers before a training "
          "load / before the eval load (exit 5, nothing loaded); the resumed run completes byte-identically to 4", ok,
          f"{ {k: v[:4] for k, v in res.items()} }")
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    cfg = make_cfg(tmp, "cap_too_big", files, anchor, cap_h=5.0)
    rc = run_cli(cfg, be)
    check("8 --cap-h above 4.5 is refused (exit 3, not persisted, no lease)", rc == 3 and not (cfg.out / "STOPPED.json").exists()
          and not any(c[0] == "lease" for c in rt.calls))


def t_stops(tmp, files, anchor):
    import psutil
    rt = FakeRuntime(tmp)
    be = TinyBackend(rt, FakeLM(rt))
    # anchor failure
    bad_ref = tmp / "anchor_ref_bad.jsonl"
    rows = [json.loads(x) for x in Path(anchor[1]).read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["states"] = [r["states"][0]] + ['{"X":0,"Y":0,"Z":0}'] * (len(r["states"]) - 1)
    write_lf(bad_ref, "".join(json.dumps(r) + "\n" for r in rows))
    cfg = make_cfg(tmp, "stop_anchor", files, (anchor[0], bad_ref))
    rc = run_cli(cfg, be)
    a0 = json.loads((cfg.out / "anchor" / "anchor.json").read_text(encoding="utf-8"))
    rc2 = run_cli(cfg, be)
    check("9 anchor failure: exit 3, anchor.json pass false, no dev output, STOPPED.json; the rerun refuses",
          rc == 3 and not a0["pass"] and not (cfg.out / "onestep").exists() and (cfg.out / "STOPPED.json").exists()
          and rc2 == 3)
    # fingerprint
    rt.fp_override = "deadbeef0000"
    cfg = make_cfg(tmp, "stop_fp", files, anchor)
    rc = run_cli(cfg, be)
    rt.fp_override = None
    check("9 fingerprint mismatch: exit 3, persisted, nothing trained", rc == 3 and (cfg.out / "STOPPED.json").exists()
          and not (cfg.out / "train_plain").exists())
    # runtime drift
    rt.runtime_override = [{"transformers": "9.9.9"}]
    n_tl = sum(c[0] == "train_load" for c in rt.calls)
    cfg = make_cfg(tmp, "stop_rt", files, anchor)
    rc = run_cli(cfg, be)
    check("9 runtime drift: exit 3, persisted, nothing loaded or trained", rc == 3 and (cfg.out / "STOPPED.json").exists()
          and sum(c[0] == "train_load" for c in rt.calls) == n_tl)
    # recorded output changed
    est = {s: 60.0 for s in R.STAGE_ORDER}
    est["E2"] = 1e9
    R.stage_est_s = lambda st, rate, n: est[st]
    try:
        cfg = make_cfg(tmp, "stop_hash", files, anchor)
        rc1 = run_cli(cfg, be)
        f = cfg.out / "onestep" / "onestep_hs_off.jsonl"
        write_lf(f, f.read_text(encoding="utf-8") + "\n")
        est["E2"] = 60.0
        rc2 = run_cli(cfg, be)
    finally:
        R.stage_est_s = REAL_EST
    check("9 a recorded output changed (a held-out-surface output): exit 3, persisted",
          rc1 == 5 and rc2 == 3 and (cfg.out / "STOPPED.json").exists())
    # one-waiter rule
    parent = psutil.Process(os.getpid()).parent()
    ppid, pct = parent.pid, parent.create_time()
    n_lease = sum(c[0] == "lease" for c in rt.calls)
    write_lf(rt.WAITING["rcwm"], json.dumps({"pid": ppid, "label": "rcwm-t1:other", "since": time.time()}))
    cfg = make_cfg(tmp, "stop_waiter", files, anchor)
    rc_w = run_cli(cfg, be)
    write_lf(rt.WAITING["rcwm"], json.dumps({"pid": 999999, "label": "rcwm-t1:dead", "since": time.time()}))
    dead_ok = R.one_waiter_problem(be) is None
    write_lf(rt.WAITING["rcwm"], json.dumps({"pid": ppid, "label": "rcwm-t1:reused", "since": pct - 100}))
    reused_ok = R.one_waiter_problem(be) is None
    rt.WAITING["rcwm"].unlink()
    write_lf(rt.LOCK_PATH, json.dumps({"label": "rcwm-t1:holder", "pid": ppid, "created": time.time()}))
    holder = R.one_waiter_problem(be)
    write_lf(rt.LOCK_PATH, json.dumps({"label": "bwm:peer", "pid": ppid, "created": time.time()}))
    bwm_ok = R.one_waiter_problem(be) is None
    rt.LOCK_PATH.unlink()
    check("9 one-waiter rule: a live RCWM waiter refuses (exit 3, not persisted, no lease); a live RCWM holder refuses; a "
          "dead marker, a reused pid and a live BWM holder do not",
          rc_w == 3 and not (cfg.out / "STOPPED.json").exists() and sum(c[0] == "lease" for c in rt.calls) == n_lease
          and dead_ok and reused_ok and holder is not None and bwm_ok)
    # run lock
    cfg = make_cfg(tmp, "stop_lock", files, anchor)
    cfg.out.mkdir(parents=True)
    write_lf(cfg.out / ".run.lock", json.dumps({"pid": ppid, "create_time": pct}))
    rc_l = run_cli(cfg, be)
    live_ok = rc_l == 3 and not (cfg.out / "STOPPED.json").exists()
    write_lf(cfg.out / ".run.lock", json.dumps({"pid": 999999, "create_time": 1.0}))
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
    full = (C.DEV, C.CHAINS, C.DEV_HS)
    SEEN_SURFACES.update(canonical=0, heldout=0)
    generation.generate_texts = oracle_generate
    try:
        cfg = make_cfg(tmp, "oracle_go", full, anchor)
        rc = run_cli(cfg, be)
        seen_go = dict(SEEN_SURFACES)
        POLICY["filler"] = ("gold", "filler")
        cfg2 = make_cfg(tmp, "oracle_stop", full, anchor)
        rc2 = run_cli(cfg2, be)
        POLICY["filler"] = ("plainlike", "filler")
        POLICY["trace"] = ("gold", "trace", "plainlike")
        cfg3 = make_cfg(tmp, "oracle_locked", full, anchor)
        rc3 = run_cli(cfg3, be)
        # revision 5, decision (1): a plain arm weak on every out-of-range row lets the tens-digit shortcut through every
        # relative G1 part; the predicate clause, read from the run's own trace texts, stops it
        POLICY.update({"plain": ("weak_oor", "plain"), "trace": ("rule:tens_or_eq_plus1", "trace"),
                       "filler": ("weak_oor", "filler")})
        cfg4 = make_cfg(tmp, "oracle_weak_shortcut", full, anchor)
        rc4 = run_cli(cfg4, be)
    finally:
        generation.generate_texts = REAL_GENERATE
    res = A.analyse_runs([cfg.out], registered=False, out_dir=tmp / "analysis_go", n_boot=1000)
    res2 = A.analyse_runs([cfg2.out], registered=False, out_dir=tmp / "analysis_stop", n_boot=1000)
    res3 = A.analyse_runs([cfg3.out], registered=False, out_dir=tmp / "analysis_locked", n_boot=1000)
    res4 = A.analyse_runs([cfg4.out], registered=False, out_dir=tmp / "analysis_weak", n_boot=1000)
    v, v2, v3, v4 = (r_["verdicts"]["s0"] for r_ in (res, res2, res3, res4))
    n_hs_gens = 4 * len(C.load_dev_hs())
    check("10 oracle runs on the FULL dev + held-out-surface + chain banks: the analysis says GO; filler = gold -> STOP "
          "(decision point 3); a trace arm locked to Titan's surface -> NO-GO on exactly G1 / G2's held-out parts",
          rc == rc2 == rc3 == 0 and v == "GO" and v2.startswith("STOP (decision point 3)")
          and v3 == "NO-GO (failed: G1 [conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge predicate HS, "
                    "cond_sametens HS, cond_sametens predicate HS]; G2 [mix HS])"
          and seen_go["heldout"] == n_hs_gens and (tmp / "analysis_go" / "T2BR_RESULTS_s0_sd.md").exists(),
          f"{v} | {v2[:40]} | {v3}; held-out prompts seen {seen_go['heldout']}")
    pl = res["per_seed"]["s0"]["chains"]["ALL"]["product_law"]["trace"]
    sa = res3["per_seed"]["s0"]["surface_audit"]["trace"]
    check("10 oracle run: gold trace side regret k4 0, product law predicted == observed == 1; the locked trace arm's "
          "surface audit shows the held-out loss on every primary stratum",
          res["per_seed"]["s0"]["chains"]["ALL"]["k"]["4"]["sides"]["trace"]["regret"]["mean"] == 0.0
          and pl["predicted_k4_exact_product"] == 1.0 and pl["observed_k4_exact"] == 1.0
          and all(sa[m]["held_out_minus_in"] < 0 for m in A.HS_METRICS))
    comps4 = {c["part"]: c["holds"] for c in res4["per_seed"]["s0"]["rule"]["components"] if c["condition"] == "G1"}
    check("10 oracle run, revision 5 decision (1): plain weak on every out-of-range row + a trace arm following the "
          "tens-digit shortcut 'tens(x) > tens(t) or x == t + 1' (its cmp.ok states that shortcut): every relative G1 part "
          "holds, the predicate clauses read from the run's trace texts fail on both surfaces -> NO-GO on exactly those",
          rc4 == 0 and v4 == "NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens predicate, cond_oor_edge "
                             "predicate HS, cond_sametens predicate HS])"
          and all(h for k, h in comps4.items() if "predicate" not in k), v4)


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
    write_lf(fake / "docs/t2b/register/TINY_TEST.json", json.dumps(tt))
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
    (fake / "runs" / "t2b_register" / "s0_sd").mkdir(parents=True)
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
    write_lf(hook, (fake / R.HOOK_DRAFT_REL).read_text(encoding="utf-8") + "# edited\n")
    hook_bad = R.hook_problem(fake, fz) is not None
    shutil.copy(fake / R.HOOK_DRAFT_REL, hook)
    hook_ok = R.hook_problem(fake, fz) is None
    p = fake / "data" / C.DEV_HS.name
    write_lf(p, p.read_text(encoding="utf-8") + "\n")
    ok2, e2, _ = R.verify_freeze(fake)
    cfg2 = R.Cfg(root=fake, out=tmp / "runs" / "frz2", steps=TINY_STEPS)
    rc2 = run_cli(cfg2, be)
    check("11 freeze tool (fake root): run() refuses before a freeze (not persisted); refuses without approval and with "
          "an output present; freezes (the go rule revision 5, the 'sd' variant, the held-out-surface bank and the "
          "superseded rule revisions 1-4 are frozen inputs); verifies; refuses a second freeze; an installed hook that is "
          "not the draft byte for byte is flagged, the exact copy is not; a changed held-out-surface bank fails "
          "verification; run() refuses",
          rc0 == 3 and no_persist and [e[0] for e in errs] == ["no approval", "output present", "second freeze"]
          and rec["go_rule_text"] == C.GO_RULE_TEXT and rec["go_rule_revision"] == 5 and rec["train_variant"] == "sd"
          and all(r in rec["inputs"] for r in ("data/t2br_dev_hs.jsonl", "docs/t2b/register/GO_RULE_rev1.md",
                                                "docs/t2b/register/GO_RULE_rev2.md", "docs/t2b/register/GO_RULE_rev3.md",
                                                "docs/t2b/register/GO_RULE_rev4.md", "docs/t2b/register/DESIGN_rev2.md",
                                                "docs/t2b/register/DESIGN_rev3.md", "docs/t2b/register/DESIGN_rev4.md"))
          and all(rec["prefreeze_checks"].get(f"GO_RULE_rev{k}.md and DESIGN_rev{k}.md keep the superseded revision {k} "
                                               "verbatim") for k in (1, 2, 3, 4))
          and ok1 and not e1 and not ok2 and rc2 == 3 and hook_bad and hook_ok
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
    check("12 plan: one turn at the observed rate; two at the planning rate (turn 2 re-anchors, then E2; every gating "
          "stage incl. E1H in turn 1); a 3 h cap needs 2 turns; an impossible cap raises",
          len(t_o) == 1 and len(t_p) == 2 and t_p[1]["stages"] == ["A0 (re-anchor)", "E2"] and "E1H" in t_p[0]["stages"]
          and len(t_3) == 2 and t_3[1]["stages"][0] in ("A0", "A0 (re-anchor)") and raised,
          f"planning {t_p}; observed {t_o[0]['projected_h']} h; 3 h cap: {t_3}")


def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="t2br_tiny_"))
    print(f"[t2br-tiny] temp dir {tmp}", flush=True)
    ok = False
    try:
        t_formats()
        t_surfaces(tmp)
        t_chain_equality()
        files = subset_files(tmp)
        full_cfg, ref_out, anchor = t_full(tmp, files)
        t_switching(full_cfg, files)
        t_crash_eval(tmp, files, ref_out, anchor)
        t_crash_train(tmp, files, ref_out, anchor)
        t_deferral(tmp, files, ref_out, anchor)
        t_wait_recheck(tmp, files, ref_out, anchor)
        t_stops(tmp, files, anchor)
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
                write_lf(C.DOCS / "TINY_TEST.json", json.dumps({
                    "ok": ok, "n_checks": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "checks": RESULTS,
                    "seconds": round(time.time() - t0, 1), "peak_working_set_mb": round(peak, 1), "threads": 2,
                    "go_rule_revision": C.GO_RULE_REVISION, "train_variant": C.TRAIN_VARIANT,
                    "code_sha256": {r: sha(ROOT / r) for r in R.TINY_CODE},
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
        if KEEP:
            print(f"[t2br-tiny] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
