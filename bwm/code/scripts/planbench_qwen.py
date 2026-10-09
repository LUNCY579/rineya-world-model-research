"""Local comparators on PlanBench Blocksworld Task 1 (plan generation): same instances, same local VAL as our system.

Resumable CPU-only harness. Each instance's prompt is PlanBench's own natural-language `query` (the exact text the
published models saw, read from the published result JSON), wrapped in the model's chat template as one user turn after
PlanBench's system message for chat engines, "You are the planner assistant who comes up with correct plans."
(LLMs-Planning send_query, used for every '_chat' result we compare against; --system-prompt none drops it).
The free-text answer is converted to PDDL by three scorers and each plan is checked by the local VAL
(bwmn.blocks.val_check) on bench/planbench/blocksworld/<generated_basic|generated_basic_3>/instance-<id>.pddl, i.e. the
same problem files and validator used for our system (scripts/blocks_eval.py) and for the published ledger
(scripts/planbench_ledger.py): our rule-based extractor (primary, val_valid), PlanBench's own text_to_plan_blocksworld
(pb_val_valid) and the same parser after markdown-only normalisation (pb_md_val_valid); see bwmn/planbench_extract.py.
Instance ids match blocks_eval: "generated_basic_3/instance-7", "generated_basic/instance-12".

  python scripts/planbench_qwen.py --model qwen3-0.6b --thinking off --limit 20
  python scripts/planbench_qwen.py --model qwen3-4b --thinking on --think-budget 2048 --ids @subset.txt
  python scripts/planbench_qwen.py --model titan-sparse --ids generated_basic_3/instance-1,generated_basic/instance-2
  python scripts/planbench_qwen.py --make-subset 60 --subset-seed 0 --subset-out subset.txt   # stratified ids (no LM)
  python scripts/planbench_qwen.py --selftest            # scorers vs published GPT-4 / o1-preview extractions (no LM)
  python scripts/planbench_qwen.py --model qwen3-4b --dry-run   # resolve paths/instances/gates/resume, load nothing

Models: qwen3-4b (BF16; HF snapshot dir), qwen3-0.6b (FP32; lm.MODELS), titan-sparse (FP32-only core; lm.MODELS).
Decoding: --thinking off -> greedy, at most --max-new new tokens (template's enable_thinking=False inserts the empty
<think></think> block). --thinking on -> Qwen-recommended sampling (T=0.6, top_p=0.95, top_k=20) with a fixed
per-instance seed; phase 1 generates up to --think-budget tokens and stops at </think>; if </think> was not produced
within the budget, "\\n</think>\\n\\n" is appended (budget forcing) and phase 2 generates the answer with at most
--max-new tokens (same RNG stream, KV cache reused). Only text after the last </think> is parsed for a plan.
--max-new defaults to 1024 and smaller values need --allow-short (smoke tests only): PlanBench's chat engines were
queried without max_tokens (its 500-token cap applies to completion engines only), so a short cap would score
truncated answers; rows record hit_eos and the summary counts answers cut at the cap.
FLOPs per instance (bwmn/flops.py, the same formula the BWM planner uses): prompt, forced and generated tokens form one
causal sequence of T tokens: 2*N_nonemb*T + 4*L*H*d_head*T(T+1)/2 (attention scores, prefill + decode) +
2*d_model*vocab*gen; a KV cache that could not be reused adds its re-prefilled prefix in full (N, L, H, d_head, d,
vocab from config.json; forced = budget-forcing delimiter tokens).
Gates: one run at a time (exclusive lock runs/planbench_qwen/run.lock; a lock left by a dead process is reclaimed);
qwen3-4b refuses to start while any other native_v1 python job runs (command line or working directory under
native_v1; --force overrides) or when available RAM < 12000 MB; before every instance (all models) available RAM must
be >= --ram-floor MB and, for qwen3-4b, no other native_v1 job may have appeared, else the run stops (resumable).
KV-cache continuation that the installed transformers cannot do (API/shape errors, never memory errors) falls back to
re-prefilling; the error is recorded per row, the re-prefill counted in FLOPs, and the run aborts if the very first
instance needed it. Resume: rows carry the full config (code sha256s, threads, model snapshot + weight-header sha,
library versions, system prompt, chat template and query sha); a file with another config is refused, a partial last
line (interrupted write) is truncated, any other unreadable line is refused.
"""
from __future__ import annotations

import os

# CPU-only by construction: hide CUDA before torch can be imported (bwmn.lm imports torch) and put bwmn.lm in CPU lane.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["BWM_DEVICE"] = "cpu"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import flops as FL  # noqa: E402
from bwmn.planbench_extract import (CANON, COLORS, RES, SET_ALIAS, SETS, TASK_FILE, _plan_lines,  # noqa: E402,F401
                                    answer_text, extract_plan, infer_mapping, load_instances,
                                    mapping_from_translation, pb_markdown_text, planbench_plans, stratified_subset,
                                    text_to_plan_blocksworld)
from bwmn.flops import flop_params, safetensors_nonemb  # noqa: E402,F401

QWEN3_4B_SNAPSHOTS = Path(r"C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots")
MODEL_CHOICES = ("qwen3-4b", "qwen3-0.6b", "titan-sparse")
RAM_GATE_MB = 12000
RAM_FLOOR_MB = 3000  # per instance, model resident: 4B BF16 KV cache ~0.15 MB/token, so ~0.6 GB at 4k tokens + slack
RAM_CMD = "(Get-Counter '\\Memory\\Available MBytes').CounterSamples.CookedValue"
FORCE_THINK_END = "\n</think>\n\n"
QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
# LLMs-Planning plan-bench/utils/llm_utils.py send_query (commit fc638a1a): the system message of every '_chat' engine.
PLANBENCH_SYSTEM = "You are the planner assistant who comes up with correct plans."
MAX_NEW_MIN = 1024
LOCK_PATH = ROOT / "runs" / "planbench_qwen" / "run.lock"
IDLE_JOBS = ("scripts/health_log.py",)  # native_v1 monitors that are always on and nearly idle; never a competing lane
CODE_FILES = ("scripts/planbench_qwen.py", "bwmn/planbench_extract.py", "bwmn/flops.py")
# generate() errors that mean "cannot continue from the passed KV cache" (API / shape mismatch). Memory errors are
# re-raised even when they arrive as RuntimeError; anything else propagates.
CACHE_ERRORS = (TypeError, ValueError, IndexError, AttributeError, NotImplementedError, RuntimeError)


# ----------------------------------------------------------------------------- model side (lazy torch import)

def model_path(name: str) -> Path:
    if name == "qwen3-4b":
        snaps = sorted(p for p in QWEN3_4B_SNAPSHOTS.glob("*") if (p / "config.json").exists())
        if len(snaps) != 1:
            raise SystemExit(f"expected exactly one Qwen3-4B snapshot with config.json under {QWEN3_4B_SNAPSHOTS}, "
                             f"found {[s.name for s in snaps]}")
        return snaps[0]
    from bwmn import lm
    return Path(lm.MODELS[name])


def model_dtype_name(name: str) -> str:
    return "bfloat16" if name == "qwen3-4b" else "float32"  # titan-sparse is FP32-only by contract


def flops(fp: dict, prompt: int, forced: int, gen: int, reprefill: int = 0) -> int:
    return FL.generation_flops(fp, prompt, gen, forced, reprefill)


def available_mb() -> float | None:
    """Available physical memory (MB): GlobalMemoryStatusEx (instant, as scripts/health_log.py), else the
    'Available MBytes' performance counter via PowerShell."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.ullAvailPhys / 2 ** 20
    except (AttributeError, OSError):
        pass
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", RAM_CMD], capture_output=True,
                             text=True, timeout=120).stdout.strip()
        return float(out.splitlines()[-1].replace(",", "."))
    except Exception:
        return None


def ram_gate(name: str) -> None:
    if name != "qwen3-4b":
        return
    mb = available_mb()
    if mb is None or mb < RAM_GATE_MB:
        raise SystemExit(f"RAM gate: Available MBytes = {mb} < {RAM_GATE_MB}; refusing to load qwen3-4b")
    print(f"RAM gate ok: {mb:.0f} MB available", flush=True)


# ----------------------------------------------------------------------------- run gates: lock, competing jobs

def _norm_path(s: str) -> str:
    return (s or "").replace("\\", "/").lower().rstrip("/")


def python_jobs() -> list[dict] | None:
    """Running python processes other than this one, its ancestors (launcher shims, an orchestrator that started us)
    and its descendants: [{pid, cmd, cwd}]. None when the process table cannot be read (psutil missing)."""
    try:
        import psutil
    except ImportError:
        return None
    me = psutil.Process()
    skip = {me.pid} | {p.pid for p in me.parents()} | {c.pid for c in me.children(recursive=True)}
    jobs = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        info = p.info
        if info["pid"] in skip or not (info.get("name") or "").lower().startswith("python"):
            continue
        jobs.append({"pid": info["pid"], "cmd": " ".join(info.get("cmdline") or []), "cwd": info.get("cwd") or ""})
    return jobs


def native_jobs(jobs: list[dict], root: Path = ROOT, idle=IDLE_JOBS) -> list[dict]:
    """The jobs that belong to native_v1: working directory under root, or root named on the command line (relative
    script paths such as 'scripts/orchestrate.py' are caught by the working directory). Always-on monitors excluded."""
    r = _norm_path(str(root))
    out = []
    for j in jobs:
        cmd, cwd = _norm_path(j.get("cmd", "")), _norm_path(j.get("cwd", ""))
        if not (cwd == r or cwd.startswith(r + "/") or r in cmd):
            continue
        if any(x in cmd for x in idle):
            continue
        out.append(j)
    return out


def peer_gate(name: str, force: bool, when: str = "start") -> None:
    """qwen3-4b runs only while no other native_v1 python job runs (PREREG_PLANBENCH: CPU lanes idle)."""
    if name != "qwen3-4b":
        return
    jobs = python_jobs()
    if jobs is None:
        if force:
            print("  peer gate: process table unreadable (psutil missing); --force given, continuing", flush=True)
            return
        raise SystemExit("peer gate: cannot read the process table (psutil missing); refusing qwen3-4b without --force")
    busy = native_jobs(jobs)
    if not busy:
        return
    desc = "; ".join(f"pid {j['pid']}: {j['cmd'][:100]}" for j in busy[:6])
    if force:
        print(f"  peer gate ({when}): {len(busy)} other native_v1 python job(s) running, --force given: {desc}",
              flush=True)
        return
    raise SystemExit(f"peer gate ({when}): {len(busy)} other native_v1 python job(s) running; refusing qwen3-4b "
                     f"(--force overrides): {desc}")


def _pid_alive(pid: int, created: float | None = None) -> bool:
    """True if pid runs and (when known) started before the lock was created; unknown -> True (never reclaim on
    ignorance)."""
    try:
        import psutil
    except ImportError:
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                                 timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return True
        return str(pid) in out
    try:
        p = psutil.Process(pid)
        return not (created is not None and p.create_time() > created + 1)  # a later process reusing the pid
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


def acquire_lock(path: Path, info: dict) -> str:
    """Exclusive run lock (O_EXCL create). A lock whose holder process is gone is moved aside (<lock>.stale-<pid>)
    and taken; a live holder -> SystemExit. Returns the token for release_lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    rec = {**info, "pid": os.getpid(), "created": time.time(), "token": token}
    for attempt in range(2):
        try:
            with open(path, "x", encoding="utf-8") as fh:
                json.dump(rec, fh)
            return token
        except FileExistsError:
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old = {}
            pid = old.get("pid")
            if attempt == 0 and isinstance(pid, int) and not _pid_alive(pid, old.get("created")):
                try:
                    os.replace(path, path.with_name(f"{path.name}.stale-{pid}"))
                    print(f"  run lock: reclaimed a stale lock of dead pid {pid}", flush=True)
                except FileNotFoundError:
                    pass
                continue
            raise SystemExit(f"run lock {path} is held by pid {pid} ({old.get('cmd')}, since "
                             f"{time.strftime('%H:%M:%S', time.localtime(old.get('created', 0)))}): one planbench_qwen "
                             f"run at a time. Delete the lock only if that process is gone.")
    raise SystemExit(f"run lock {path}: could not acquire")


def release_lock(path: Path, token: str) -> None:
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
            path.unlink()
    except (OSError, json.JSONDecodeError):
        pass


# ----------------------------------------------------------------------------- provenance (resume config)

def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_shas() -> dict:
    return {f: file_sha(ROOT / f) for f in CODE_FILES}


def lib_versions() -> dict:
    v = {"python": platform.python_version()}
    for p in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "accelerate"):
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v[p] = None
    return v


def model_identity(path: Path) -> dict:
    """Snapshot id (HF commit dir / core dir name), config sha and a sha over every safetensors file's name, size and
    header (tensor names, shapes, dtypes, offsets): identifies the weights without reading them."""
    h = hashlib.sha256()
    for f in sorted(Path(path).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = fh.read(8)
            hdr = fh.read(int.from_bytes(n, "little"))
        h.update(f.name.encode() + b"\0" + str(f.stat().st_size).encode() + b"\0" + hdr)
    return {"snapshot": Path(path).name, "config_sha": file_sha(Path(path) / "config.json"),
            "weights_header_sha": h.hexdigest()}


def chat_template_sha(path: Path) -> str | None:
    p = Path(path)
    if (p / "chat_template.jinja").exists():
        return file_sha(p / "chat_template.jinja")
    tc = p / "tokenizer_config.json"
    t = json.loads(tc.read_text(encoding="utf-8")).get("chat_template") if tc.exists() else None
    return hashlib.sha256(json.dumps(t).encode()).hexdigest() if t else None


def queries_sha(insts: list[dict]) -> str:
    h = hashlib.sha256()
    for i in insts:
        h.update(i["id"].encode() + b"\0" + i["query"].encode() + b"\0")
    return h.hexdigest()


# ----------------------------------------------------------------------------- generation

def load_model(name: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    path = model_path(name)
    dtype = getattr(torch, model_dtype_name(name))
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    assert all(p.device.type == "cpu" for p in model.parameters()), "CPU-only harness"
    assert next(model.parameters()).dtype == dtype, (next(model.parameters()).dtype, dtype)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    return model, tok, path


def render_prompt(tok, query: str, thinking: bool, system: str | None = PLANBENCH_SYSTEM) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def instance_seed(base: int, iid: str) -> int:
    return (base * 1_000_003 + int(hashlib.sha256(iid.encode()).hexdigest()[:8], 16)) % (2 ** 31)


def _is_memory_error(e: BaseException) -> bool:
    import torch
    oom = getattr(torch, "OutOfMemoryError", None)
    msg = str(e).lower()
    return (oom is not None and isinstance(e, oom)) or "out of memory" in msg or "not enough memory" in msg \
        or "defaultcpuallocator" in msg


def _gen(model, ids, cache, max_new: int, stop_ids: list[int], sample: bool, pad_id: int):
    """One generate() call continuing `cache` if given. Returns (new tokens up to the first stop, reusable cache or
    None, cache error or None). A cache continuation that fails with a CACHE_ERRORS (not memory) error is retried
    without the cache, i.e. the whole prefix is re-prefilled; the caller counts those FLOPs."""
    import torch
    kw = dict(max_new_tokens=max_new, eos_token_id=stop_ids, pad_token_id=pad_id, return_dict_in_generate=True)
    kw.update(dict(do_sample=True, **QWEN_SAMPLING) if sample else dict(do_sample=False, temperature=None, top_p=None,
                                                                         top_k=None))
    att = torch.ones_like(ids)
    err = None
    try:
        out = model.generate(input_ids=ids, attention_mask=att, past_key_values=cache, **kw)
    except CACHE_ERRORS as e:
        if cache is None or _is_memory_error(e):
            raise
        err = f"{type(e).__name__}: {str(e)[:300]}"
        out = model.generate(input_ids=ids, attention_mask=att, **kw)  # re-prefill if cache continuation unsupported
    new = out.sequences[0, ids.shape[1]:].tolist()
    cut = next((k + 1 for k, t in enumerate(new) if t in stop_ids), len(new))
    return new[:cut], (out.past_key_values if cut == len(new) else None), err


def generate_one(model, tok, prompt: str, *, thinking: bool, think_budget: int, max_new: int, seed: int) -> dict:
    """One completion with optional budget forcing. Returns token accounting + raw text (special tokens removed).
    reprefill_tokens: prefix tokens (prompt + thinking) processed a second time because phase 2 could not continue
    the phase-1 KV cache (cache_fallback_errors says why; reprefill_reason 'no_cache' if none was returned)."""
    import torch
    from bwmn import lm
    eos = list(lm.EOS_IDS)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[-1]
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    n_prompt = ids.shape[1]
    torch.manual_seed(seed)
    t0 = time.time()
    gen_ids: list[int] = []
    think_tokens = forced_tokens = reprefill = 0
    forced = closed = ended = False
    errors: list[str] = []
    reason = None
    cache = None
    with torch.no_grad():
        if thinking:
            think_end = tok.convert_tokens_to_ids("</think>")
            new, cache, _ = _gen(model, ids, None, think_budget, eos + [think_end], True, pad)
            gen_ids += new
            think_tokens = len(new)
            ended = bool(new) and new[-1] in eos
            closed = bool(new) and new[-1] == think_end
            ids = torch.cat([ids, torch.tensor([new], dtype=ids.dtype)], dim=1)
            if not ended and not closed:
                fids = tok(FORCE_THINK_END, add_special_tokens=False).input_ids
                gen_ids += fids
                forced, forced_tokens = True, len(fids)
                ids = torch.cat([ids, torch.tensor([fids], dtype=ids.dtype)], dim=1)
        answer_tokens = 0
        if not ended:
            if thinking and cache is None:
                reason, reprefill = "no_cache", n_prompt + think_tokens
            new, _, err = _gen(model, ids, cache, max_new, eos, thinking, pad)
            if err:
                errors.append(err)
                reason, reprefill = "cache_error", n_prompt + think_tokens
            gen_ids += new
            answer_tokens = len(new)
            ended = bool(new) and new[-1] in eos
    wall = time.time() - t0
    return {"prompt_tokens": n_prompt, "gen_tokens": think_tokens + answer_tokens, "think_tokens": think_tokens,
            "answer_tokens": answer_tokens, "forced_think_end": forced, "forced_tokens": forced_tokens,
            "think_closed_naturally": closed, "hit_eos": ended, "cache_fallback": bool(errors),
            "cache_fallback_errors": errors, "reprefill_tokens": reprefill, "reprefill_reason": reason,
            "wall_s": round(wall, 2), "raw_text": tok.decode(gen_ids, skip_special_tokens=True)}


# ----------------------------------------------------------------------------- scoring

def score_plan(inst: dict, plan: list[tuple]) -> dict:
    val = B.val_check(inst["path"], plan) if plan else False
    sim = B.execute(inst["init"], inst["goal"], plan)["success"] if plan else False
    return {"val_valid": bool(val), "sim_valid": bool(sim)}


def score_variants(inst: dict, text: str, cmap: dict, thinking: bool) -> dict:
    """All three scorers on one completion; VAL once per distinct plan."""
    plan, xinfo = extract_plan(text, cmap, thinking=thinking)
    pb = planbench_plans(text, thinking=thinking)
    sc = score_plan(inst, plan)
    verdict = {tuple(plan): sc["val_valid"]}
    for p in pb.values():
        if tuple(p) not in verdict:
            verdict[tuple(p)] = bool(B.val_check(inst["path"], p)) if p else False
    pddl = lambda p: [f"({' '.join(x)})" for x in p]
    return {"extracted_plan": pddl(plan), "val_valid": sc["val_valid"], "sim_valid": sc["sim_valid"],
            "pb_plan": pddl(pb["pb_raw"]), "pb_val_valid": verdict[tuple(pb["pb_raw"])],
            "pb_md_plan": pddl(pb["pb_md"]), "pb_md_val_valid": verdict[tuple(pb["pb_md"])], "extract_info": xinfo,
            "n_plan": len(plan)}


def variant_summary(rows: list[dict]) -> str:
    n = len(rows)
    k = lambda f: sum(bool(r.get(f)) for r in rows)
    dis = sum(r.get("val_valid") != r.get("pb_val_valid") for r in rows)
    cut = sum(not r.get("hit_eos") for r in rows)
    fl = sum(r.get("flops", 0) for r in rows)
    return (f"{n} rows | VAL ours {k('val_valid')}/{n}, PlanBench parser {k('pb_val_valid')}/{n}, PlanBench parser on "
            f"markdown-normalised text {k('pb_md_val_valid')}/{n} | ours!=PlanBench {dis} | simulator {k('sim_valid')}"
            f" | answers cut at --max-new {cut} | forced think end {k('forced_think_end')} | cache fallbacks "
            f"{k('cache_fallback')} | FLOPs {fl:.3e}")


# ----------------------------------------------------------------------------- run

def default_out(a) -> Path:
    tag = f"think-on_B{a.think_budget}" if a.thinking == "on" else "think-off"
    sys_tag = "" if a.system_prompt == "planbench" else "_nosys"
    return ROOT / "runs" / "planbench_qwen" / f"{a.model}_{tag}_max{a.max_new}_{a.prompt_type}{sys_tag}_s{a.seed}.jsonl"


def run_config(a, path: Path, all_insts: list[dict]) -> dict:
    """Everything a row's numbers depend on; a resumed file must carry exactly this config."""
    system = PLANBENCH_SYSTEM if a.system_prompt == "planbench" else None
    prompt = {"system": system, "chat_template_sha": chat_template_sha(path), "queries_sha": queries_sha(all_insts)}
    prompt["sha"] = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
    return {"model": a.model, "dtype": model_dtype_name(a.model), "thinking": a.thinking,
            "think_budget": a.think_budget if a.thinking == "on" else None, "max_new": a.max_new, "seed": a.seed,
            "prompt_type": a.prompt_type, "query_source": a.query_source,
            "decoding": ("sample " + json.dumps(QWEN_SAMPLING)) if a.thinking == "on" else "greedy",
            "threads": a.threads, "model_id": model_identity(path), "prompt": prompt, "code_sha": code_shas(),
            "versions": lib_versions(), "flops_formula": "dense + causal attention scores + lm_head; re-prefill added"}


def read_rows(out: Path, cfg: dict, repair: bool) -> list[dict]:
    """Rows of an existing output file. A partial last line (interrupted write) is truncated when repair is set
    (reported otherwise); any other non-JSON line, a row with another config, or a duplicate id -> SystemExit."""
    if not out.exists():
        return []
    data = out.read_bytes()
    cut = data.rfind(b"\n") + 1 if data and not data.endswith(b"\n") else len(data)
    if cut < len(data):
        tail = data[cut:]
        what = "truncated" if repair else "would truncate (not in --dry-run)"
        print(f"  {out.name}: partial last line ({len(tail)} bytes, {tail[:60]!r}...) {what}", flush=True)
        if repair:
            with open(out, "r+b") as fh:
                fh.truncate(cut)
                fh.flush()
                os.fsync(fh.fileno())
    rows = []
    for k, ln in enumerate(data[:cut].decode("utf-8").splitlines()):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            raise SystemExit(f"{out}:{k + 1} is not valid JSON and is not the last line; refusing to resume")
        if r.get("config") != cfg:
            old = r.get("config") or {}
            diff = sorted(x for x in set(cfg) | set(old) if cfg.get(x) != old.get(x))
            raise SystemExit(f"{out} holds rows from a different config (differs in {diff}); use another --out")
        rows.append(r)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{out} has duplicate ids (concurrent writers?); refusing to resume")
    return rows


def select(insts: list[dict], a) -> list[dict]:
    if a.ids:
        spec = a.ids
        if spec.startswith("@"):
            spec = Path(spec[1:]).read_text(encoding="utf-8").replace("\n", ",")
        want = [s.strip() for s in spec.split(",") if s.strip()]
        known = {i["id"]: i for i in insts}
        missing = [w for w in want if w not in known]
        if missing:
            raise SystemExit(f"unknown ids: {missing[:5]} (format: generated_basic/instance-12)")
        insts = [known[w] for w in want]
    return insts[: a.limit] if a.limit else insts


def instance_gates(a, n_done: int) -> float | None:
    """Before every instance: available RAM >= --ram-floor (all models) and, for qwen3-4b, still no other native_v1
    job. A failed gate stops the run; finished rows are already on disk, so it resumes later."""
    mb = available_mb()
    if mb is None or mb < a.ram_floor:
        raise SystemExit(f"RAM floor: Available MBytes = {mb} < {a.ram_floor} before instance {n_done + 1}; stopping "
                         f"(resumable)")
    peer_gate(a.model, a.force, when=f"before instance {n_done + 1}")
    return mb


def check_fallback(g: dict, n_done: int, iid: str) -> None:
    """The KV cache must work in this environment: if the first instance of the run needed a re-prefill, abort before
    writing it (every later row would silently carry the re-prefill cost)."""
    if g["cache_fallback"] and n_done == 0:
        raise SystemExit(f"KV-cache continuation failed on the first instance ({iid}): {g['cache_fallback_errors']}; "
                         f"aborting (fix the cache path before running)")
    if g["cache_fallback"]:
        print(f"  WARNING cache fallback on {iid}: {g['cache_fallback_errors']}", flush=True)


def run(a) -> None:
    all_insts = load_instances(SET_ALIAS["all"], a.prompt_type, a.query_source)
    insts = select([i for i in all_insts if i["set"] in SET_ALIAS[a.set]], a)
    path = model_path(a.model)
    cfg = run_config(a, path, all_insts)
    out = Path(a.out) if a.out else default_out(a)
    fp = flop_params(path / "config.json")
    if a.dry_run:
        rows = read_rows(out, cfg, repair=False)
        done = {r["id"] for r in rows}
        todo = [i for i in insts if i["id"] not in done]
        print(f"[{time.strftime('%H:%M:%S')}] {a.model} ({cfg['dtype']}, {path}) thinking={a.thinking} "
              f"budget={cfg['think_budget']} max_new={a.max_new} system={cfg['prompt']['system']!r}: {len(insts)} "
              f"selected, {len(done)} done, {len(todo)} to run -> {out}  N_nonemb={fp['n_nonemb']:,}", flush=True)
        if LOCK_PATH.exists():
            print(f"  run lock present: {LOCK_PATH.read_text(encoding='utf-8')[:200]}", flush=True)
        if rows:
            print("  " + variant_summary(rows), flush=True)
        ram_gate(a.model)
        peer_gate(a.model, a.force)
        for i in todo[:3]:
            print("  ", i["id"], i["objs"], i["map_info"], flush=True)
        return
    token = acquire_lock(LOCK_PATH, {"cmd": " ".join(sys.argv), "out": str(out), "model": a.model})
    try:
        _run_locked(a, insts, cfg, out, fp)
    finally:
        release_lock(LOCK_PATH, token)


def _run_locked(a, insts: list[dict], cfg: dict, out: Path, fp: dict) -> None:
    rows = read_rows(out, cfg, repair=True)
    done = {r["id"] for r in rows}
    todo = [i for i in insts if i["id"] not in done]
    print(f"[{time.strftime('%H:%M:%S')}] {a.model} ({cfg['dtype']}) thinking={a.thinking} budget={cfg['think_budget']}"
          f" max_new={a.max_new}: {len(insts)} selected, {len(done)} done, {len(todo)} to run -> {out}  "
          f"N_nonemb={fp['n_nonemb']:,}", flush=True)
    if not todo:
        return
    peer_gate(a.model, a.force)
    ram_gate(a.model)
    t0 = time.time()
    model, tok, _ = load_model(a.model, a.threads)
    print(f"  loaded in {time.time() - t0:.0f}s, threads={a.threads}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    system = cfg["prompt"]["system"]
    thinking = a.thinking == "on"
    n_ok = n = 0
    new_rows = []
    with open(out, "a", encoding="utf-8") as fh:
        for inst in todo:
            ram = instance_gates(a, n)
            seed = instance_seed(a.seed, inst["id"])
            prompt = render_prompt(tok, inst["query"], thinking, system)
            g = generate_one(model, tok, prompt, thinking=thinking, think_budget=a.think_budget or 0,
                             max_new=a.max_new, seed=seed)
            check_fallback(g, n, inst["id"])
            sv = score_variants(inst, g["raw_text"], inst["cmap"], thinking)
            terms = FL.flops_breakdown(fp, g["prompt_tokens"], g["gen_tokens"], g["forced_tokens"],
                                       g["reprefill_tokens"])
            row = {"id": inst["id"], **{k: g[k] for k in ("prompt_tokens", "gen_tokens", "wall_s")},
                   "raw_text": g["raw_text"],
                   **{k: sv[k] for k in ("extracted_plan", "val_valid", "sim_valid", "pb_plan", "pb_val_valid",
                                         "pb_md_plan", "pb_md_val_valid")},
                   **{k: g[k] for k in ("think_tokens", "answer_tokens", "forced_think_end", "forced_tokens",
                                        "think_closed_naturally", "hit_eos", "cache_fallback", "cache_fallback_errors",
                                        "reprefill_tokens", "reprefill_reason")},
                   "flops": terms["total"], "flops_terms": terms, "n_nonemb": fp["n_nonemb"], "seed": seed,
                   "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(), "ram_mb_before": round(ram or 0),
                   "color_map": inst["cmap"], "map_info": inst["map_info"], "extract_info": sv["extract_info"],
                   "config": cfg, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            new_rows.append(row)
            n += 1
            n_ok += sv["val_valid"]
            print(f"  {inst['id']:<32} p={g['prompt_tokens']} g={g['gen_tokens']}{' F' if g['forced_think_end'] else ''}"
                  f" {g['wall_s']:.1f}s plan={sv['n_plan']} VAL={sv['val_valid']} pb={sv['pb_val_valid']}"
                  f" pb_md={sv['pb_md_val_valid']}{'' if sv['val_valid'] == sv['sim_valid'] else ' SIM-DISAGREE'}"
                  f"{'' if g['hit_eos'] else ' CUT'}  [{n_ok}/{n}]", flush=True)
    print("  this run: " + variant_summary(new_rows), flush=True)
    print("  file:     " + variant_summary(rows + new_rows), flush=True)


# ----------------------------------------------------------------------------- stratified subset (no LM)

def make_subset(a) -> int:
    insts = load_instances(SET_ALIAS[a.set], a.prompt_type, a.query_source)
    sub, table = stratified_subset(insts, a.make_subset, seed=a.subset_seed)
    for k, (size, drawn) in table.items():
        print(f"  {k:<16} {drawn:>3} / {size}", file=sys.stderr)
    text = "".join(i["id"] + "\n" for i in sub)
    if a.subset_out:
        Path(a.subset_out).write_text(text, encoding="utf-8")
        print(f"{len(sub)} ids ({len(table)} strata, seed {a.subset_seed}) -> {a.subset_out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


# ----------------------------------------------------------------------------- extractor self-test (no LM)

UNIT_CASES = [
    ("1. Unstack the blue block from on top of the orange block\n2. Put down the blue block\n3. Pick up the orange "
     "block\n4. Stack the orange block on top of the blue block",
     [("unstack", "b", "c"), ("put-down", "b"), ("pick-up", "c"), ("stack", "c", "b")]),
    ("<think>\nmaybe pick up the red block\n</think>\n\nPick up the red block and stack it on the yellow block.",
     [("pick-up", "a"), ("stack", "a", "d")]),
    ("<think>\nstill thinking, pick up the red block", []),
    ("### Step 1: Unstack the blue block from the orange block\n- The blue block is clear.\n### Step 2: Put it "
     "down on the table\n### Step 3: Pick up the orange block\n### Step 4: Stack it on the blue block\n\n"
     "### Verification:\n1. Unstack blue from orange -> holding blue",
     [("unstack", "b", "c"), ("put-down", "b"), ("pick-up", "c"), ("stack", "c", "b")]),
    ("Let's analyze.\n\n**Plan:**\n1. **Unstack** the blue block from the orange block.\n   - Since the hand is "
     "empty, you can unstack it.\n2. **Put down** the blue block.\n\n**Final Plan:**\n```\n(unstack blue orange)\n"
     "(put-down blue)\n```\nThis plan ensures that the orange block is clear.",
     [("unstack", "b", "c"), ("put-down", "b")]),
    ("Plan: unstack blue from orange, put it on the table, then pick up orange and place it onto the blue block.",
     [("unstack", "b", "c"), ("put-down", "b"), ("pick-up", "c"), ("stack", "c", "b")]),
    ("1. Pick up the green block\n2. Stack the green block on the red block", [("pick-up", "green"),
                                                                                ("stack", "green", "a")]),
    ("1. Unstack the blue block\n2. Move the blue block onto the red block", []),
    ("1. **Stack blue on yellow**: Stack the blue block (which you're holding) onto the clear yellow block.",
     [("stack", "b", "d")]),
    ("1. **Unstack(Blue, Orange)**\n2. **PutDown(Blue)**", [("unstack", "b", "c"), ("put-down", "b")]),
    ("Call the orange block (O) and the blue block (B).\n1. Unstack B from O\n2. Put down B",
     [("unstack", "b", "c"), ("put-down", "b")]),
    ("1. Unstack the red block from the yellow block\n   - Now that the red block is clear, unstack it from the "
     "yellow block.\n2. Put down the red block\n\nThis sequence works, placing the red block on the table.",
     [("unstack", "a", "d"), ("put-down", "a")]),
    ("1. Pick up the red block (with the blue block on top)\n2. Unstack the blue block from the table",
     [("pick-up", "a")]),
    # must-fix cases (2026-10-07): commentary under a step, same-line plan mentions, '1)' lists, markdown tables
    ("1. Unstack the blue block from the orange block\n   - Alternatively, pick up the red block first.\n"
     "2. Put down the blue block", [("unstack", "b", "c"), ("put-down", "b")]),
    ("1. Unstack the blue block from the orange block\n2. Put down the blue block\n3. Pick up the orange block\n"
     "Finally, to complete the plan: stack the orange block on the blue block.",
     [("unstack", "b", "c"), ("put-down", "b"), ("pick-up", "c"), ("stack", "c", "b")]),
    ("Plan: 1) unstack the blue block from the orange block 2) put down the blue block",
     [("unstack", "b", "c"), ("put-down", "b")]),
    ("| Step | Action | Block | Target |\n|---|---|---|---|\n| 1 | Unstack | blue | orange |\n| 2 | Put down | blue | "
     "table |", [("unstack", "b", "c"), ("put-down", "b")]),
]


def selftest(a) -> int:
    ledger_p = B.BENCH / "LEDGER.json"
    ledger = json.loads(ledger_p.read_text()) if ledger_p.exists() else {}
    models = [m.strip() for m in a.selftest_models.split(",") if m.strip()]
    rc = 0
    # mapping inference over all instances (query source = o1-preview, which also carries raw_translation)
    insts = load_instances(SET_ALIAS[a.set], "zero_shot", "o1-preview_chat")
    src = {}
    for i in insts:
        k = (i["map_info"]["source"], i["map_info"]["n_consistent"] > 1, i["map_info"]["canonical"])
        src[k] = src.get(k, 0) + 1
    rt_ok = rt_n = 0
    for i in insts:
        tm = mapping_from_translation(i["record"])
        if tm:
            rt_n += 1
            rt_ok += all(i["cmap"].get(c) == l for c, l in tm.items())
    print(f"mapping: {len(insts)} instances; (source, ambiguous, canonical) -> count: {src}")
    print(f"mapping vs o1-preview raw_translation/extracted_llm_plan pairs: {rt_ok}/{rt_n} consistent")
    by_id = {i["id"]: i for i in insts}
    for model in models:
        key = f"{model}|task_1_plan_generation_zero_shot"
        tot = {"n": 0, "seq": 0, "val": 0, "mine_ok": 0, "theirs_ok": 0, "val_calls": 0, "sim_mismatch": 0,
               "pb_raw": 0, "pb_md": 0, "pb_raw_same": 0, "pb_md_same": 0}
        tab = {(x, y): 0 for x in (0, 1) for y in (0, 1)}
        bad = []
        for inst_dir in SET_ALIAS[a.set]:
            f = RES / SETS[inst_dir] / model / TASK_FILE["zero_shot"]
            for r in json.loads(f.read_text(encoding="utf-8"))["instances"]:
                iid = f"{inst_dir}/instance-{r['instance_id']}"
                inst = by_id.get(iid)
                if inst is None:
                    continue
                mine, info = extract_plan(r["llm_raw_response"] or "", inst["cmap"])
                theirs = [tuple(ln.split()) for ln in _plan_lines(r.get("extracted_llm_plan"))]
                v_theirs = ledger.get(key, {}).get("per_instance", {}).get(iid)
                if v_theirs is None:
                    v_theirs = B.val_check(inst["path"], theirs) if theirs else False
                    tot["val_calls"] += 1
                verdict = {tuple(theirs): bool(v_theirs)}  # identical PDDL plan on the identical problem: same verdict
                if tuple(mine) not in verdict:
                    verdict[tuple(mine)] = bool(B.val_check(inst["path"], mine)) if mine else False
                    tot["val_calls"] += bool(mine)
                v_mine = verdict[tuple(mine)]
                same = mine == theirs
                sim = B.execute(inst["init"], inst["goal"], mine)["success"] if mine else False
                tot["sim_mismatch"] += sim != v_mine
                for name, p in planbench_plans(r["llm_raw_response"] or "").items():
                    if tuple(p) not in verdict:
                        verdict[tuple(p)] = bool(B.val_check(inst["path"], p)) if p else False
                        tot["val_calls"] += bool(p)
                    tot[name] += verdict[tuple(p)]
                    tot[name + "_same"] += p == theirs
                tot["n"] += 1
                tot["seq"] += same
                tot["val"] += v_mine == v_theirs
                tot["mine_ok"] += v_mine
                tot["theirs_ok"] += v_theirs
                tab[(int(v_theirs), int(v_mine))] += 1
                if not same:
                    bad.append((iid, v_theirs, v_mine, theirs, mine, info, r["llm_raw_response"]))
        n = max(tot["n"], 1)
        print(f"\n{model}: n={tot['n']}  exact action-sequence agreement {tot['seq']}/{n} ({100 * tot['seq'] / n:.1f}%)"
              f"  VAL-verdict agreement {tot['val']}/{n} ({100 * tot['val'] / n:.1f}%)")
        print(f"  VAL success: published extraction {tot['theirs_ok']}, ours {tot['mine_ok']}; "
              f"(theirs,ours) table {tab}; VAL calls {tot['val_calls']}; VAL-vs-simulator mismatches "
              f"{tot['sim_mismatch']}")
        print(f"  PlanBench parser (2nd scorer): VAL success pb_raw {tot['pb_raw']}, pb_md {tot['pb_md']}; plan "
              f"identical to the published extraction pb_raw {tot['pb_raw_same']}/{n}, pb_md {tot['pb_md_same']}/{n}")
        for iid, vt, vm, th, mi, info, raw in bad[: a.show]:
            print(f"  -- {iid} VAL theirs={vt} ours={vm} region={info['region']} issues={info['issues'][:3]}")
            print(f"     theirs={[' '.join(x) for x in th]}")
            print(f"     ours  ={[' '.join(x) for x in mi]}")
            if a.show_raw:
                print("     raw:", (raw or "")[: a.show_raw].replace("\n", "\n          "))
        if tot["sim_mismatch"]:
            rc = 1
    # PlanBench parser port vs every published extraction that PlanBench made with that parser (no raw_translation)
    print("\nPlanBench parser port vs published rule-based extractions (exact plan; '*'-stripped for Gemini-1.5):")
    pb_fail = 0
    for d in sorted(p for p in (RES / "blocksworld").iterdir() if p.is_dir()):
        same = n = 0
        for inst_dir in SET_ALIAS[a.set]:
            f = RES / SETS[inst_dir] / d.name / TASK_FILE["zero_shot"]
            if not f.exists():
                continue
            for r in json.loads(f.read_text(encoding="utf-8"))["instances"]:
                if r.get("raw_translation") or "extracted_llm_plan" not in r:
                    continue
                txt = r["llm_raw_response"] or ""
                txt = txt.replace("*", "") if d.name.startswith("gemini-1.5") else txt
                n += 1
                same += text_to_plan_blocksworld(txt)[0] == [tuple(ln.split()) for ln in
                                                             _plan_lines(r["extracted_llm_plan"])]
        if n:
            print(f"  {d.name:<26} {same}/{n}")
            pb_fail += same != n and d.name in ("gpt-4_chat", "gpt-4o_chat", "claude-3-opus", "gemini-pro")
    # unit cases (synthetic answers in styles the comparators are expected to produce)
    cm = {"red": "a", "blue": "b", "orange": "c", "yellow": "d"}
    fails = 0
    for txt, want in UNIT_CASES:
        got, info = extract_plan(txt, cm)
        if got != want:
            fails += 1
            print(f"  UNIT FAIL: {txt[:70]!r}\n     want {want}\n     got  {got} {info}")
    print(f"\nunit cases: {len(UNIT_CASES) - fails}/{len(UNIT_CASES)} pass")
    # FLOP parameter count: config formula vs safetensors headers; closed form vs per-token sum
    for name in MODEL_CHOICES:
        try:
            p = model_path(name)
            fp = flop_params(p / "config.json")
            n_st = safetensors_nonemb(p)
            print(f"N_nonemb {name}: config {fp['n_nonemb']:,} vs safetensors {n_st:,} "
                  f"({'match' if n_st == fp['n_nonemb'] else 'DIFF'}); d={fp['d_model']} vocab={fp['vocab']} "
                  f"L={fp['n_layers']} H={fp['n_heads']} d_head={fp['head_dim']}")
        except SystemExit as e:
            print(f"N_nonemb {name}: {e}")
    fp = {"n_nonemb": 7, "d_model": 3, "vocab": 5, "n_layers": 2, "n_heads": 2, "head_dim": 4}
    brute = sum(2 * 7 + 4 * 2 * 2 * 4 * (t + 1) for t in range(40 + 3 + 25)) + 2 * 3 * 5 * 25 \
        + sum(2 * 7 + 4 * 2 * 2 * 4 * (t + 1) for t in range(40 + 10))
    flop_ok = flops(fp, 40, 3, 25, reprefill=50) == brute
    print(f"FLOPs closed form vs per-token sum: {'match' if flop_ok else 'DIFF'}")
    return rc or (1 if fails or pb_fail or not flop_ok else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=MODEL_CHOICES)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--think-budget", type=int, help="thinking-token budget B (required with --thinking on)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW_MIN,
                    help=f"answer tokens (after </think> when thinking is on); >= {MAX_NEW_MIN} unless --allow-short")
    ap.add_argument("--allow-short", action="store_true", help=f"permit --max-new < {MAX_NEW_MIN} (smoke tests only)")
    ap.add_argument("--system-prompt", choices=["planbench", "none"], default="planbench",
                    help=f"planbench = {PLANBENCH_SYSTEM!r} (PlanBench's chat engines); none = user turn only")
    ap.add_argument("--ids", help="comma-separated ids (generated_basic/instance-12,...) or @file with one id per line")
    ap.add_argument("--limit", type=int, help="first N selected instances (order: generated_basic_3, generated_basic)")
    ap.add_argument("--set", choices=sorted(SET_ALIAS), default="all")
    ap.add_argument("--prompt-type", choices=sorted(TASK_FILE), default="zero_shot")
    ap.add_argument("--query-source", default="o1-preview_chat",
                    help="published result dir to read 'query' from (13/14 models share these exact bytes; gpt-4_chat "
                         "differs only by trailing whitespace)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true", help="run qwen3-4b although other native_v1 python jobs run")
    ap.add_argument("--ram-floor", type=int, default=RAM_FLOOR_MB, help="available MB required before each instance")
    ap.add_argument("--dry-run", action="store_true", help="resolve paths, instances, resume state, gates; no model")
    ap.add_argument("--make-subset", type=int, metavar="N", help="write a stratified subset of N ids (no model)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--subset-out", help="file for --make-subset ids (default: stdout); use with --ids @file")
    ap.add_argument("--selftest", action="store_true", help="extractor tests on published responses; no model")
    ap.add_argument("--selftest-models", default="gpt-4_chat,o1-preview_chat")
    ap.add_argument("--show", type=int, default=0, help="selftest: print N disagreements per model")
    ap.add_argument("--show-raw", type=int, default=0, help="selftest: print first N chars of each shown raw response")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a)
    if a.make_subset is not None:
        return make_subset(a)
    if not a.model:
        ap.error("--model is required (or use --selftest / --make-subset)")
    if a.thinking == "on" and not (a.think_budget and a.think_budget > 0):
        ap.error("--thinking on requires --think-budget B > 0")
    if a.thinking == "off" and a.think_budget is not None:
        ap.error("--think-budget only applies with --thinking on")
    if a.max_new <= 0 or (a.limit is not None and a.limit <= 0) or a.threads <= 0:
        ap.error("--max-new, --limit and --threads must be positive")
    if a.max_new < MAX_NEW_MIN and not a.allow_short:
        ap.error(f"--max-new {a.max_new} < {MAX_NEW_MIN}: answers would be truncated (PlanBench's chat engines had no "
                 f"cap); pass --allow-short for smoke tests")
    if a.prompt_type not in TASK_FILE or not (RES / "blocksworld" / a.query_source / TASK_FILE[a.prompt_type]).exists():
        ap.error(f"no {TASK_FILE[a.prompt_type]} under results/*/{a.query_source}")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
