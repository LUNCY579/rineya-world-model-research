"""Local comparators on PlanBench Blocksworld Task 1 (plan generation): same instances, same local VAL as our system.

Resumable CPU-only harness for Qwen3-4B (and qwen3-0.6b / titan-sparse) on two instance sets:
- PB-600 (--set all|bw3|bw; ids "generated_basic_3/instance-7", "generated_basic/instance-12"): each prompt is
  PlanBench's own natural-language `query` (the exact text the published models saw, read from the published result
  JSON), as one user turn after PlanBench's system message for chat engines, "You are the planner assistant who comes
  up with correct plans." (LLMs-Planning send_query). Scorers: our rule-based extractor (primary, val_valid),
  PlanBench's own text_to_plan_blocksworld (pb_val_valid) and the same parser after markdown-only normalisation
  (pb_md_val_valid); see bwmn/planbench_extract.py.
- Blocksworld-hard (--set hard; 110 instances, 6-15 blocks; ids "hard/instance-N" as scripts/blocks_eval.py): each
  prompt is the zero-shot PDDL query DeepSeek-R1, o1-preview and o1-mini were sent (read from the sha-pinned upstream
  per-instance files and checked byte for byte against upstream's query builder on the local problem file), as the
  only (user) turn: upstream sent no system message to o1-*/deepseek-r1 (--system-prompt auto resolves to none on
  hard, planbench on PB-600). Scorers: our PDDL extractor (primary, val_valid) and upstream's own save_gpt3_response
  text checked by VAL as is (pb_val_valid); pb_md_* is None.
Every plan is checked by the local VAL (bench/planbench/VAL) on the instance's problem and domain files, i.e. the
same files and validator used for our system (scripts/blocks_eval.py) and for the published ledger; sim_valid is the
exact simulator's verdict on the primary plan.

  python scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking off                       # Q4N, all 110
  python scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking on --iso H2_hard --subset 40   # Q4T-iso
  python scripts/planbench_qwen.py --model qwen3-4b --set bw3 --limit 3 --thinking off             # throughput probe
  python scripts/planbench_qwen.py --model qwen3-4b --thinking on --think-budget 2048 --ids @subset.txt
  python scripts/planbench_qwen.py --make-subset 40 --set hard        # the systematic hard subset (no LM)
  python scripts/planbench_qwen.py --iso-ledger                       # H2/B2 compute ledger + iso budgets (no LM)
  python scripts/planbench_qwen.py --selftest                         # scorers, ports, FLOPs, lane lock (no LM)
  python scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking on --iso H2_hard --subset 40 --dry-run

Models: qwen3-4b (BF16; the one HF snapshot dir), qwen3-0.6b (FP32), titan-sparse (FP32-only core); paths equal
bwmn.lm.MODELS (asserted at load; not imported before, so --dry-run/--selftest never import torch).
Decoding: --thinking off -> greedy, at most --max-new new tokens (template's enable_thinking=False inserts the empty
<think></think> block). --thinking on -> Qwen-recommended sampling (T=0.6, top_p=0.95, top_k=20) with a fixed
per-instance seed; phase 1 generates up to the thinking budget and stops at </think>; if </think> was not produced
within the budget, "\\n</think>\\n\\n" is appended (budget forcing) and phase 2 generates the answer (same RNG stream,
KV cache reused). Only text after the last </think> is parsed for a plan. The thinking budget is either fixed
(--think-budget B; phase 2 gets --max-new) or iso-FLOP (--iso KEY [--iso-fraction f]): the per-instance FLOP cap
f x ISO_REGISTRY[KEY] (H2's measured mean inference FLOPs per instance, from --iso-ledger) is converted with
bwmn/flops.py into the largest total sequence T (prompt + thinking + forced + answer) within the cap; the thinking
budget is T - prompt - forced - A (A = --answer-reserve, default --max-new) and phase 2 may use the rest of T, so every
completion's FLOPs are <= the cap (a cache re-prefill, if any, is recorded and flagged over the cap).
--max-new defaults to 1024 and smaller values need --allow-short (smoke tests only): PlanBench's chat engines were
queried without max_tokens (its 500-token cap applies to completion engines only), so a short cap would score
truncated answers; rows record hit_eos and the summary counts answers cut at the cap.
FLOPs per instance (bwmn/flops.py, the same formula and code as the BWM planner's compute ledger): prompt, forced and
generated tokens form one causal sequence of T tokens: 2*N_nonemb*T + 4*L*H*d_head*T(T+1)/2 (attention scores,
prefill + decode) + 2*d_model*vocab*gen; a KV cache that could not be reused adds its re-prefilled prefix in full.
Gates (model runs only; --dry-run reports them without failing):
- CPU lane (D-76, shared with RCX, rcx/lane.py protocol): %TEMP%/rineya-cpu-lane.json {holder 'bwm', job, pid,
  started, expected_end, token} created with O_CREAT|O_EXCL before the model loads and deleted at exit; a lock whose
  holder pid is dead (psutil.pid_exists or OpenProcess, never os.kill on Windows) is renamed to
  rineya-cpu-lane.json.stale-<pid>.bak and reclaimed; the lane is never taken while RCX's lock or its
  rineya-cpu-lane.waiting.rcx marker exists; while waiting, rineya-cpu-lane.waiting.bwm is written; when waiting.rcx
  appears, the run yields at the next instance boundary (lane released, model freed; it waits and resumes, or exits 75
  with --lane-no-wait). RINEYA_LANE_DIR overrides the directory (tests only).
- one planbench_qwen run at a time (runs/planbench_qwen/run.lock); <= 4 threads (D-64/D-76);
- RAM: >= 12000 MB available before any model load; >= --ram-floor (4000) MB before every instance; during
  generation, every --guard-every tokens, >= 2000 MB, else the instance is abandoned unwritten and the run stops
  (resumable);
- qwen3-4b: no other native_v1 python job (command line or working directory under native_v1; the always-on
  health_log/orchestrate/gpu_run wrappers excepted, so a BWM model job = gpu_run's child) at start, before every
  instance and every --guard-every tokens; --force overrides;
- D-35: bwmn.lm is imported before transformers (CUDA patched off; refused otherwise); CPU thermal guard between
  instances.
KV-cache continuation that the installed transformers cannot do (API/shape errors, never memory errors) falls back to
re-prefilling; the error is recorded per row, the re-prefill counted in FLOPs, and the run aborts if the very first
instance needed it. Resume: rows carry the full config (code sha256s, threads, model snapshot + weight-header sha,
library versions, system prompt, chat template and query sha, set/subset, budget rule); a file with another config is
refused, a partial last line (interrupted write) is truncated, any other unreadable line is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import flops as FL  # noqa: E402
from bwmn import planbench_extract as X  # noqa: E402
from bwmn.planbench_extract import (CANON, COLORS, RES, SET_ALIAS, SETS, TASK_FILE, _plan_lines,  # noqa: E402,F401
                                    answer_text, extract_plan, infer_mapping, load_instances,
                                    mapping_from_translation, pb_markdown_text, planbench_plans, stratified_subset,
                                    text_to_plan_blocksworld)
from bwmn.flops import flop_params, safetensors_nonemb  # noqa: E402,F401

QWEN3_4B_SNAPSHOTS = Path(r"C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots")
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
MODEL_DIRS = {  # == bwmn.lm.MODELS[name] (asserted in load_model; bwmn.lm imports torch, so it is not imported here)
    "qwen3-0.6b": Path(r"C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
                       r"c1899de289a04d12100db370d81485cdf75e47ca"),
    "titan-sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
}
MODEL_CHOICES = ("qwen3-4b", "qwen3-0.6b", "titan-sparse")
SET_CHOICES = ("all", "bw3", "bw", "hard")
RAM_GATE_MB = 12000  # before any model load (PREREG_PLANBENCH: >= 12 GB; D-64: 12 GB for model loads)
RAM_FLOOR_MB = 4000  # before every instance, model resident (D-64: 4 GB free floor for jobs over 2 GB)
RAM_HARD_FLOOR_MB = 2000  # during generation (every --guard-every tokens): below it the instance is abandoned
MAX_THREADS = 4  # D-64 / D-76: one CPU lane at <= 4 threads
RAM_CMD = "(Get-Counter '\\Memory\\Available MBytes').CounterSamples.CookedValue"
FORCE_THINK_END = "\n</think>\n\n"
QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
# LLMs-Planning plan-bench/utils/llm_utils.py send_query (commit fc638a1a): the system message of every '_chat' engine.
PLANBENCH_SYSTEM = "You are the planner assistant who comes up with correct plans."
MAX_NEW_MIN = 1024
LOCK_PATH = ROOT / "runs" / "planbench_qwen" / "run.lock"
# native_v1 python processes that are always on and never hold a model: the health logger, the GPU queue
# orchestrator and the GPU-lease wrapper (scripts/gpu_run.py; the model job is its child, which is detected).
IDLE_JOBS = ("scripts/health_log.py", "scripts/orchestrate.py", "scripts/gpu_run.py")
CODE_FILES = ("scripts/planbench_qwen.py", "bwmn/planbench_extract.py", "bwmn/flops.py")
# generate() errors that mean "cannot continue from the passed KV cache" (API / shape mismatch). Memory errors are
# re-raised even when they arrive as RuntimeError, and a RuntimeError counts only if it names a shape/cache problem.
CACHE_ERRORS = (TypeError, ValueError, IndexError, AttributeError, NotImplementedError, RuntimeError)
CACHE_RUNTIME_HINTS = ("size", "shape", "dimension", "cache", "past_key", "position")
EXIT_YIELD = 75  # "yield, requeue" (the convention of lm.wait_host_ram and scripts/bundle.py)

# ----------------------------------------------------------------------------- registered iso-FLOP budgets
# Mean inference FLOPs per Blocksworld-hard instance of the frozen S-beam runs, recomputed by --iso-ledger from the
# run artifacts only (beam_ledger: unique F keys per shard from fcache.jsonl, proposer tokens and rounds from
# plans.jsonl, prompt lengths by the Titan tokenizer, bwmn/flops.py with the active LoRA parameters). "upper" is the
# registered budget: every F query counted as a full sequence (no credit for the shared system-prompt KV cache),
# expansions at their maximum given rounds, proposer prompts at the longest state on the submitted plan's trajectory,
# the convexity bound on within-sample attention. It credits the comparator with at least H2's compute.
ISO_REGISTRY = {  # registered 2026-10-11 (research/QWEN_COMPARATOR_PLAN.md), before any Qwen output exists
    "H2_hard": {"run": "runs/BW_B3_HARD/h2_beam", "freeze": "runs/FREEZE_B3_H2.json", "estimate": "upper",
                "flops_per_instance": 52988071221360, "lower_flops_per_instance": 33998377648358,
                "inputs_sha256": {
                    "plans.jsonl": "3cf3cd01b8a73f3e036fda126a5d6b562b38f63e458197a188639f2df6e2b91b",
                    "FREEZE_B3_H2.json": "a433e499e23cf6fb1edd353c50b8cc9466e224b8adca56ef5731306ce46bda51",
                    "titan tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                    "shard_0of8/fcache.jsonl": "c7b9dfb605868e53f1d0b6bac6068a80e5bbc87779bf53f44ec89e51b809e58a",
                    "shard_1of8/fcache.jsonl": "28c5e6ee0d08ebd4a0895add7e2a88f5a6d98dce5ccc87edca17042242b8ca4b",
                    "shard_2of8/fcache.jsonl": "13ccb217f8a54a56bb15c7035cb9e643a55ca0290d38885b162ae657cb2c2717",
                    "shard_3of8/fcache.jsonl": "f1f2bf098263f283bde582aec94290a0f4bee0ab3f04852884b72f87a710054a",
                    "shard_4of8/fcache.jsonl": "52b312b8722e2e1476cf4a82fdc8093c1895687229d4c50076859d8d0753bb4f",
                    "shard_5of8/fcache.jsonl": "39c361eed0eff68fa78b3820f2f1cad957a6465f2e2db16401e6427b17275cc0",
                    "shard_6of8/fcache.jsonl": "5b5418bd565da7f4036ace70bda362188cf28fad2cd918b6ec66f56809dd5bab",
                    "shard_7of8/fcache.jsonl": "2c354964f8f58556313482fdf96d9708486d7121659db2150aeadbf7cf6bd998"}},
}
REPORT_LEDGERS = {"B2_hard": {"run": "runs/BW_TEST_B2/hard_beam", "freeze": "runs/FREEZE_B2_beam.json"},
                  "B2_pb600": {"run": "runs/BW_TEST_B2/planbench_beam", "freeze": "runs/FREEZE_B2_beam.json"}}
# Q4T's hard subset: systematic_subset(load_hard_instances(), 40) (sorted by blocks, optimal length, id); sha256 of
# its ids, one per line with trailing newline (ids_sha).
HARD_SUBSET_40_SHA = "f952f532da5cba077729abea3894ca04c8b9ed9ee4999e2040638881b46ff237"


# ----------------------------------------------------------------------------- process environment

def cpu_only_env(threads: int | None = None) -> None:
    """CPU-only by construction: hide CUDA and put bwmn.lm in CPU-lane mode (its D-35 patch) before torch is imported;
    pin the BLAS/OpenMP pools to `threads`. Called by main() and load_model(); importing this module changes nothing."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["BWM_DEVICE"] = "cpu"
    if threads:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = str(threads)


# ----------------------------------------------------------------------------- model side (lazy torch import)

def model_path(name: str) -> Path:
    if name == "qwen3-4b":
        snaps = sorted(p for p in QWEN3_4B_SNAPSHOTS.glob("*") if (p / "config.json").exists())
        if len(snaps) != 1:
            raise SystemExit(f"expected exactly one Qwen3-4B snapshot with config.json under {QWEN3_4B_SNAPSHOTS}, "
                             f"found {[s.name for s in snaps]}")
        return snaps[0]
    return MODEL_DIRS[name]


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
    """>= RAM_GATE_MB available before loading any model (all comparators load >= 2 GB of weights)."""
    mb = available_mb()
    if mb is None or mb < RAM_GATE_MB:
        raise SystemExit(f"RAM gate: Available MBytes = {mb} < {RAM_GATE_MB}; refusing to load {name}")
    print(f"RAM gate ok: {mb:.0f} MB available", flush=True)


# ----------------------------------------------------------------------------- run gates: competing jobs

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
    script paths such as 'scripts/orchestrate.py' are caught by the working directory). Always-on wrappers excluded."""
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


def peer_problem(name: str, force: bool) -> str | None:
    """None if the peer rule allows qwen3-4b now, else the reason (PREREG_PLANBENCH: CPU lanes idle; D-75.4: only
    while no BWM model is loaded)."""
    if name != "qwen3-4b":
        return None
    jobs = python_jobs()
    if jobs is None:
        return None if force else "cannot read the process table (psutil missing)"
    busy = native_jobs(jobs)
    if not busy or force:
        if busy:
            print(f"  peer gate: {len(busy)} other native_v1 python job(s) running, --force given", flush=True)
        return None
    return f"{len(busy)} other native_v1 python job(s) running: " + "; ".join(
        f"pid {j['pid']}: {j['cmd'][:100]}" for j in busy[:6])


def peer_gate(name: str, force: bool, when: str = "start") -> None:
    msg = peer_problem(name, force)
    if msg:
        raise SystemExit(f"peer gate ({when}): {msg}; refusing qwen3-4b (--force overrides)")


# ----------------------------------------------------------------------------- liveness, run lock, CPU lane (D-76)

def pid_alive(pid, started=None) -> bool:
    """Liveness without signals (never os.kill(pid, 0) on Windows, where it can terminate the target):
    psutil.pid_exists, else OpenProcess + GetExitCodeProcess (access denied = exists). With psutil and a numeric
    `started` (the lock's creation time), a live pid whose process was created after started + 2 s is a reused pid,
    i.e. the holder is dead. Unknown -> alive (never reclaim on ignorance)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            if not psutil.pid_exists(pid):
                return False
        except (OverflowError, ValueError):
            return False
        if isinstance(started, (int, float)):
            try:
                return psutil.Process(pid).create_time() <= float(started) + 2
            except psutil.NoSuchProcess:
                return False
            except psutil.Error:
                return True
        return True
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        if pid > 0xFFFFFFFF:
            return False
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k.GetLastError() == 5  # access denied: the process exists
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)  # POSIX only
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path, info: dict) -> str:
    """Exclusive run lock (O_EXCL create). A lock whose holder process is gone is moved aside (<lock>.stale-<pid>.bak)
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
            if attempt == 0 and isinstance(pid, int) and not pid_alive(pid, old.get("created")):
                try:
                    os.replace(path, path.with_name(f"{path.name}.stale-{pid}.bak"))
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


LANE_FILE = "rineya-cpu-lane.json"
LANE_PARTY = "bwm"
LANE_PRIORITY = ("rcx",)  # parties whose waiting marker makes BWM yield and not (re-)acquire (D-76)
LANE_UNREADABLE_S = 120  # an unparseable lock older than this is a torn write by a dead writer: moved aside


class LaneBusy(Exception):
    pass


def lane_dir() -> Path:
    return Path(os.environ.get("RINEYA_LANE_DIR") or os.environ.get("TEMP") or os.environ.get("TMP")
                or tempfile.gettempdir())


def lane_lock_path(d: Path | None = None) -> Path:
    return (d or lane_dir()) / LANE_FILE


def lane_waiting_path(party: str, d: Path | None = None) -> Path:
    return (d or lane_dir()) / f"rineya-cpu-lane.waiting.{party}"


def lane_read(d: Path | None = None) -> dict | None:
    """The lock record, None if there is no lock, {'_unreadable': age_s} if it cannot be parsed."""
    p = lane_lock_path(d)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        try:
            return {"_unreadable": time.time() - p.stat().st_mtime}
        except OSError:
            return None


def lane_status(d: Path | None = None) -> dict:
    d = d or lane_dir()
    return {"dir": str(d), "lock": lane_read(d),
            "waiting": sorted(p.name for p in Path(d).glob("rineya-cpu-lane.waiting.*"))}


def lane_try_acquire(job: str, expected_s: float, d: Path | None = None, pid: int | None = None,
                     log=print) -> dict | None:
    """One attempt at the shared CPU lane: None while a live lock exists or a priority party's waiting marker exists.
    A lock whose holder pid is dead is renamed to rineya-cpu-lane.json.stale-<pid>.bak first. The lock is created with
    O_CREAT|O_EXCL, so of two racing parties exactly one wins."""
    d = d or lane_dir()
    lp = lane_lock_path(d)
    lock = lane_read(d)
    if lock is not None:
        if "_unreadable" in lock:
            if lock["_unreadable"] < LANE_UNREADABLE_S:
                return None
            stale = lp.with_name(f"{LANE_FILE}.stale-unreadable-{int(time.time())}.bak")
        elif pid_alive(lock.get("pid"), lock.get("started")):
            return None
        else:
            stale = lp.with_name(f"{LANE_FILE}.stale-{lock.get('pid')}.bak")
        try:
            os.replace(lp, stale)
            log(f"  lane: moved a stale lock aside ({lock.get('holder')}, pid {lock.get('pid')}) -> {stale.name}")
        except FileNotFoundError:
            pass
        except OSError:
            return None
    if any(lane_waiting_path(p, d).exists() for p in LANE_PRIORITY):
        return None
    now = time.time()
    rec = {"holder": LANE_PARTY, "job": job, "pid": int(os.getpid() if pid is None else pid), "started": now,
           "expected_end": now + float(expected_s), "token": uuid.uuid4().hex}
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
        fh.flush()
        os.fsync(fh.fileno())
    return rec


def lane_acquire(job: str, expected_s: float, *, wait: bool = True, poll_s: float = 60, max_wait_s: float | None = None,
                 d: Path | None = None, pid: int | None = None, log=print, sleep=time.sleep) -> dict:
    """Take the lane, waiting (wait=True) with rineya-cpu-lane.waiting.bwm written until it is free and no priority
    party waits; wait=False or max_wait_s exceeded -> LaneBusy. The waiting marker is removed on every exit path."""
    d = d or lane_dir()
    marker = lane_waiting_path(LANE_PARTY, d)
    t0, said = time.time(), None
    try:
        while True:
            rec = lane_try_acquire(job, expected_s, d, pid, log)
            if rec:
                log(f"  lane: acquired {lane_lock_path(d)} for {job!r}")
                return rec
            st = lane_status(d)
            if not wait:
                raise LaneBusy(f"CPU lane busy: {st}")
            marker.write_text(json.dumps({"party": LANE_PARTY, "job": job, "since": t0,
                                          "pid": int(os.getpid() if pid is None else pid)}), encoding="utf-8")
            lock = st["lock"] or {}
            msg = (f"  lane: waiting (holder {lock.get('holder')} pid {lock.get('pid')} job {lock.get('job')!r}; "
                   f"markers {st['waiting']}); wrote {marker.name}")
            if msg != said:
                log(msg)
                said = msg
            if max_wait_s is not None and time.time() - t0 > max_wait_s:
                raise LaneBusy(f"CPU lane not free after {max_wait_s:.0f} s: {st}")
            sleep(poll_s)
    finally:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass


def lane_holds(rec: dict, d: Path | None = None) -> bool:
    lock = lane_read(d)
    return bool(lock) and lock.get("token") == rec.get("token") and lock.get("holder") == LANE_PARTY


def lane_should_yield(d: Path | None = None) -> bool:
    return any(lane_waiting_path(p, d).exists() for p in LANE_PRIORITY)


def lane_release(rec: dict | None, d: Path | None = None) -> bool:
    """Delete the lock only if it is still ours (token); never another holder's."""
    if not rec or not lane_holds(rec, d):
        return False
    try:
        lane_lock_path(d).unlink()
    except FileNotFoundError:
        return False
    return True


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


def ids_sha(insts: list[dict]) -> str:
    return hashlib.sha256("".join(i["id"] + "\n" for i in insts).encode()).hexdigest()


# ----------------------------------------------------------------------------- prompts and token counts

def render_qwen3_prompt(query: str, thinking: bool, system: str | None = None) -> str:
    """The Qwen3 chat template's output for [system?, user] + generation prompt (enable_thinking=False appends the
    empty think block). Used for torch-free token counts (--dry-run, --iso-ledger); a run records whether the
    tokenizer's own apply_chat_template produced the same string (prompt_render_match)."""
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    s += f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    return s if thinking else s + "<think>\n\n</think>\n\n"


def rust_tokenizer(model_dir: Path):
    """The model's tokenizer.json through the `tokenizers` library only (no transformers/torch import)."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def count_tokens(tok, texts: list[str]) -> list[int]:
    return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]


# ----------------------------------------------------------------------------- generation

def load_model(name: str, threads: int):
    cpu_only_env(threads)
    from bwmn import lm  # D-35: with BWM_DEVICE=cpu, importing lm patches torch.cuda.is_available/device_count
    import torch
    if getattr(torch.cuda.is_available, "__name__", "") != "<lambda>":
        raise SystemExit("D-35: torch was imported before bwmn.lm in CPU mode (CUDA not patched off); refusing")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    path = model_path(name)
    if name in MODEL_DIRS and Path(lm.MODELS[name]) != MODEL_DIRS[name]:
        raise SystemExit(f"{name}: MODEL_DIRS {MODEL_DIRS[name]} != bwmn.lm.MODELS {lm.MODELS[name]}")
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
    torch = sys.modules.get("torch")  # never imports torch: without it no torch OOM type can have been raised
    oom = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    if isinstance(e, MemoryError):
        return True
    msg = str(e).lower()
    return (oom is not None and isinstance(e, oom)) or "out of memory" in msg or "not enough memory" in msg \
        or "defaultcpuallocator" in msg


def _is_cache_error(e: BaseException) -> bool:
    if _is_memory_error(e):
        return False
    if isinstance(e, RuntimeError) and not isinstance(e, NotImplementedError):
        return any(h in str(e).lower() for h in CACHE_RUNTIME_HINTS)
    return isinstance(e, CACHE_ERRORS)


class GuardStop(Exception):
    """Raised after generate() when the in-generation guard stopped the instance (its row is not written)."""


class GenGuard:
    """In-generation checks every `every` tokens (a transformers StoppingCriteria wraps step()): check() returns a
    reason to stop (RAM below the hard floor, a native_v1 job appeared for qwen3-4b, the lane lock is no longer ours)
    or None. The first reason is kept; generation then stops and the caller raises GuardStop."""

    def __init__(self, every: int, check):
        self.every, self.check, self.last, self.reason = int(every), check, None, None

    def step(self, length: int) -> bool:
        if self.reason is not None:
            return True
        if self.last is None:
            self.last = length
        elif length - self.last >= self.every:
            self.last = length
            self.reason = self.check()
        return self.reason is not None

    def criteria(self):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        guard = self

        class _Crit(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                stop = guard.step(int(input_ids.shape[1]))
                return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

        return StoppingCriteriaList([_Crit()])


def _gen(model, ids, cache, max_new: int, stop_ids: list[int], sample: bool, pad_id: int, guard: GenGuard | None = None):
    """One generate() call continuing `cache` if given. Returns (new tokens up to the first stop, reusable cache or
    None, cache error or None). A cache continuation that fails with a cache/shape (not memory) error is retried
    without the cache, i.e. the whole prefix is re-prefilled; the caller counts those FLOPs."""
    import torch
    kw = dict(max_new_tokens=max_new, eos_token_id=stop_ids, pad_token_id=pad_id, return_dict_in_generate=True)
    kw.update(dict(do_sample=True, **QWEN_SAMPLING) if sample else dict(do_sample=False, temperature=None, top_p=None,
                                                                         top_k=None))
    if guard is not None:
        kw["stopping_criteria"] = guard.criteria()
    att = torch.ones_like(ids)
    err = None
    try:
        out = model.generate(input_ids=ids, attention_mask=att, past_key_values=cache, **kw)
    except CACHE_ERRORS as e:
        if cache is None or not _is_cache_error(e):
            raise
        err = f"{type(e).__name__}: {str(e)[:300]}"
        out = model.generate(input_ids=ids, attention_mask=att, **kw)  # re-prefill if cache continuation unsupported
    if guard is not None and guard.reason:
        raise GuardStop(guard.reason)
    new = out.sequences[0, ids.shape[1]:].tolist()
    cut = next((k + 1 for k, t in enumerate(new) if t in stop_ids), len(new))
    return new[:cut], (out.past_key_values if cut == len(new) else None), err


def generate_one(model, tok, prompt: str, *, thinking: bool, think_budget: int, max_new: int, seed: int,
                 total_cap: int | None = None, guard: GenGuard | None = None) -> dict:
    """One completion with optional budget forcing. Returns token accounting + raw text (special tokens removed).
    total_cap (iso-FLOP mode): phase 2 may generate up to total_cap - (prompt + thinking + forced) tokens instead of
    max_new. reprefill_tokens: prefix tokens (prompt + thinking) processed a second time because phase 2 could not
    continue the phase-1 KV cache (cache_fallback_errors says why; reprefill_reason 'no_cache' if none returned)."""
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
            new, cache, _ = _gen(model, ids, None, think_budget, eos + [think_end], True, pad, guard)
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
        answer_cap = max_new if total_cap is None else total_cap - ids.shape[1]
        if not ended and answer_cap > 0:
            if thinking and cache is None:
                reason, reprefill = "no_cache", n_prompt + think_tokens
            new, _, err = _gen(model, ids, cache, answer_cap, eos, thinking, pad, guard)
            if err:
                errors.append(err)
                reason, reprefill = "cache_error", n_prompt + think_tokens
            gen_ids += new
            answer_tokens = len(new)
            ended = bool(new) and new[-1] in eos
    wall = time.time() - t0
    return {"prompt_tokens": n_prompt, "gen_tokens": think_tokens + answer_tokens, "think_tokens": think_tokens,
            "answer_tokens": answer_tokens, "answer_cap": answer_cap, "forced_think_end": forced,
            "forced_tokens": forced_tokens, "think_closed_naturally": closed, "hit_eos": ended,
            "cache_fallback": bool(errors), "cache_fallback_errors": errors, "reprefill_tokens": reprefill,
            "reprefill_reason": reason, "wall_s": round(wall, 2),
            "raw_text": tok.decode(gen_ids, skip_special_tokens=True)}


# ----------------------------------------------------------------------------- iso-FLOP budgets

def iso_caps(fp: dict, prompt_tokens: int, flop_cap: float, forced: int, reserve: int) -> dict:
    """Per-instance budget under a FLOP cap: total_cap = the largest causal length T (prompt + thinking + forced +
    answer) whose cost is <= flop_cap with every non-prompt token generated (an LM-head row each; forced tokens need
    none, so a budget-forced completion of the same length costs less): both the natural-close and the forced path stay
    within the cap. think_budget = T - prompt - forced - reserve (>= 1 or ValueError)."""
    T = FL.max_total_tokens(fp, prompt_tokens, flop_cap, 0)
    if T is None:
        raise ValueError(f"FLOP cap {flop_cap:.3e} is below the prompt's own cost ({prompt_tokens} tokens)")
    think = T - prompt_tokens - forced - reserve
    if think < 1:
        raise ValueError(f"FLOP cap {flop_cap:.3e}: total {T} tokens leave no thinking budget after prompt "
                         f"{prompt_tokens}, forced {forced} and answer reserve {reserve}")
    return {"flop_cap": int(flop_cap), "total_cap": int(T), "think_budget": int(think), "answer_reserve": int(reserve)}


def iso_flop_cap(key: str, fraction: float) -> float:
    v = ISO_REGISTRY[key]["flops_per_instance"]
    if v is None:
        raise SystemExit(f"iso budget {key} is not registered yet (run --iso-ledger and register it)")
    return float(v) * float(fraction)


# ----------------------------------------------------------------------------- BWM compute ledger (no model)

def _problem_path(iid: str) -> Path:
    sub, name = iid.split("/")
    if sub == "hard":
        return X.HARD_DIR / "generated" / f"{name}.pddl"
    return B.BENCH / "blocksworld" / sub / f"{name}.pddl"


def beam_ledger(run_dir: Path, freeze_path: Path, core_dir: Path = MODEL_DIRS["titan-sparse"]) -> dict:
    """Inference FLOPs per instance of a frozen S-beam run (bwmn.blocks_planner.plan_beam with LMForwardBW in the delta
    format and LMProposerBW's shared-prefix sampler), from its artifacts only; no model, no torch.
    F: every unique (state, action) key of each shard's fcache.jsonl was computed once by the model in that shard
    (the planner memoises F; restarts replayed the cache, so this also fixes the FR-02 undercount of summary.json);
    prompt = B.forward_prompt_delta tokens, output = re-tokenized text + EOS, capped at LMForwardBW's 40. upper: each
    query a full causal sequence (flops_breakdown); lower: its suffix after the shard's shared token prefix (the
    system prompt KV cache that greedy_shared_prefix reuses), the prefix itself not counted.
    Proposer: plans.jsonl gen_tokens (all K samples of all expansions, EOS included) and rounds; expansions are not
    logged, so upper = min(1 + W(rounds-1), E - 1 + W) expansions at the longest proposer prompt over the states of the
    submitted plan's trajectory with the convexity bound on within-sample pairs (beam_proposer_pairs_bound, cap M);
    lower = max(rounds, ceil(gen / (K M))) expansions at the shortest such prompt with equal-length samples.
    Active LoRA parameters (unmerged) are added to N per role. Mean per instance = (F total + proposer total) / n."""
    run_dir, freeze_path = Path(run_dir), Path(freeze_path)
    args = json.loads(freeze_path.read_text())["args"]
    if args.get("planner") != "beam" or args.get("fwd_fmt") != "delta" or args.get("proposer") != "lm":
        raise ValueError(f"{freeze_path}: not an LM S-beam delta-F configuration")
    K, W, E, M = int(args["K"]), int(args["W"]), int(args["E"]), int(args["M"])
    fa = ROOT / args["fwd_adapter"] / "adapter_model.safetensors"
    pa = ROOT / args["pa"] / "adapter_model.safetensors"
    fpF = FL.flop_params(Path(core_dir) / "config.json", FL.safetensors_params(fa))
    fpP = FL.flop_params(Path(core_dir) / "config.json", FL.safetensors_params(pa))
    tok = rust_tokenizer(core_dir)
    f_up = f_lo = 0
    n_keys = p_tok = o_tok = 0
    shards, all_keys, pfx_seen = {}, set(), set()
    for sh in sorted(p for p in run_dir.glob("shard_*") if p.is_dir()):
        fc = sh / "fcache.jsonl"
        keys: dict = {}
        for ln in fc.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue  # torn last line after a crash (LMForwardBW skips it too)
            keys.setdefault((frozenset(map(tuple, r["s"])), tuple(r["a"])), r["text"])
        ks = list(keys)
        prompts = [B.forward_prompt_delta(s, a) for s, a in ks]
        enc = tok.encode_batch(prompts, add_special_tokens=False)
        P = [len(e.ids) for e in enc]
        O = [min(o + 1, 40) for o in count_tokens(tok, [keys[k] for k in ks])]
        pfx = len(os.path.commonprefix([e.ids for e in enc])) if enc else 0
        pfx = min([pfx] + [p - 1 for p in P]) if P else 0
        pfx_seen.add(pfx)
        for p, o in zip(P, O):
            f_up += FL.flops_breakdown(fpF, p, o)["total"]
            f_lo += FL.forward_flops(fpF, p - pfx + o, n_ctx=pfx, n_logits=o)
        n_keys += len(ks)
        p_tok += sum(P)
        o_tok += sum(O)
        all_keys |= set(ks)
        shards[sh.name] = {"unique_keys": len(ks), "fcache_sha256": file_sha(fc), "shared_prefix_tokens": pfx}
    rows = [json.loads(ln) for ln in (run_dir / "plans.jsonl").read_text().splitlines() if ln.strip()]
    a = 4 * fpP["n_layers"] * fpP["n_heads"] * fpP["head_dim"]
    p_up = p_lo = 0
    ex_lb_sum = ex_ub_sum = gen_sum = 0
    for r in rows:
        objs, init, goal = B.parse_problem(_problem_path(r["id"]))
        s, states = init, [init]
        for act in r.get("plan") or []:
            t = tuple(act.split())
            if not B.applicable(s, t):
                break
            s, _ = B.step(s, t)
            states.append(s)
        L = count_tokens(tok, [B.proposer_prompt(x, goal) for x in states])
        pmin, pmax = min(L), max(L)
        G, R = int(r["gen_tokens"]), int(r["rounds"])
        ub = min(1 + W * (R - 1), E - 1 + W)
        lb = max(R, math.ceil(G / (K * M)))
        ns = K * ub
        g = G / ns
        p_up += (2 * fpP["n_nonemb"] * (ub * pmax + G)
                 + a * (ub * pmax * (pmax + 1) // 2 + pmax * G + FL.beam_proposer_pairs_bound(G, M))
                 + 2 * fpP["d_model"] * fpP["vocab"] * G)
        p_lo += (2 * fpP["n_nonemb"] * (lb * pmin + G) + a * (lb * pmin * (pmin + 1) / 2 + pmin * G + ns * g * (g + 1) / 2)
                 + 2 * fpP["d_model"] * fpP["vocab"] * G)
        ex_lb_sum, ex_ub_sum, gen_sum = ex_lb_sum + lb, ex_ub_sum + ub, gen_sum + G
    n = len(rows)
    return {"run": str(run_dir.relative_to(ROOT)).replace("\\", "/"), "freeze": str(freeze_path.relative_to(ROOT)).replace("\\", "/"),
            "freeze_sha256": file_sha(freeze_path), "plans_sha256": file_sha(run_dir / "plans.jsonl"),
            "tokenizer_sha256": file_sha(Path(core_dir) / "tokenizer.json"), "n": n,
            "success": sum(bool(r.get("val", r.get("success"))) for r in rows),
            "args": {k: args[k] for k in ("K", "W", "E", "M", "R", "P")},
            "n_nonemb_core": FL.flop_params(Path(core_dir) / "config.json")["n_nonemb"],
            "lora_params": {"fwd": fpF["n_nonemb"] - FL.flop_params(Path(core_dir) / "config.json")["n_nonemb"],
                            "pa": fpP["n_nonemb"] - FL.flop_params(Path(core_dir) / "config.json")["n_nonemb"]},
            "f_unique_keys_sum_over_shards": n_keys, "f_unique_keys_across_shards": len(all_keys),
            "f_keys_per_instance": round(n_keys / n, 2), "f_prompt_tokens_mean": round(p_tok / n_keys, 2),
            "f_output_tokens_mean": round(o_tok / n_keys, 2), "f_shared_prefix_tokens": sorted(pfx_seen),
            "proposer_gen_tokens_per_instance": round(gen_sum / n, 1),
            "expansions_per_instance": {"lower": round(ex_lb_sum / n, 2), "upper": round(ex_ub_sum / n, 2)},
            "flops_per_instance": {"f": {"upper": f_up / n, "lower": f_lo / n},
                                   "proposer": {"upper": p_up / n, "lower": p_lo / n},
                                   "total": {"upper": (f_up + p_up) / n, "lower": (f_lo + p_lo) / n}},
            "shards": shards}


def hard_prompt_tokens(model_dir: Path, insts: list[dict], thinking: bool, system: str | None) -> dict:
    tok = rust_tokenizer(model_dir)
    n = count_tokens(tok, [render_qwen3_prompt(i["query"], thinking, system) for i in insts])
    return dict(zip((i["id"] for i in insts), n))


def iso_table(fp: dict, prompts: dict, flop_cap: float, forced: int, reserve: int) -> dict:
    caps = {iid: iso_caps(fp, p, flop_cap, forced, reserve) for iid, p in prompts.items()}
    tc = sorted(c["total_cap"] for c in caps.values())
    tb = sorted(c["think_budget"] for c in caps.values())
    pr = sorted(prompts.values())
    med = lambda v: v[len(v) // 2]
    return {"flop_cap": int(flop_cap), "n": len(caps), "prompt_tokens": [pr[0], med(pr), pr[-1]],
            "total_cap": [tc[0], med(tc), tc[-1]], "think_budget": [tb[0], med(tb), tb[-1]], "per_instance": caps}


def cmd_iso_ledger(a) -> int:
    out = {"registered": {}, "report_only": {}}
    rc = 0
    for key, spec in ISO_REGISTRY.items():
        led = beam_ledger(ROOT / spec["run"], ROOT / spec["freeze"])
        up, lo = led["flops_per_instance"]["total"]["upper"], led["flops_per_instance"]["total"]["lower"]
        reg = spec["flops_per_instance"]
        got_inputs = {"plans.jsonl": led["plans_sha256"], Path(spec["freeze"]).name: led["freeze_sha256"],
                      "titan tokenizer.json": led["tokenizer_sha256"],
                      **{f"{s}/fcache.jsonl": v["fcache_sha256"] for s, v in led["shards"].items()}}
        in_ok = got_inputs == spec.get("inputs_sha256")
        ok = reg is not None and int(round(up)) == int(reg) and in_ok
        rc |= 0 if ok else 1
        if not in_ok:
            print(f"  {key}: input sha256s differ from the registered ones: "
                  f"{sorted(k for k in got_inputs if got_inputs[k] != (spec.get('inputs_sha256') or {}).get(k))}")
        out["registered"][key] = {"ledger": led, "registered_flops_per_instance": reg, "reproduced": ok}
        print(f"{key}: mean FLOPs/instance upper {up:.6e} (int {int(round(up))}), lower {lo:.6e}; registered {reg} -> "
              f"{'REPRODUCED' if ok else 'NOT REGISTERED / DIFFERS'}")
        print(f"  F keys {led['f_unique_keys_sum_over_shards']} ({led['f_keys_per_instance']}/instance; across shards "
              f"{led['f_unique_keys_across_shards']}), F prompt/output tokens {led['f_prompt_tokens_mean']}/"
              f"{led['f_output_tokens_mean']}, proposer gen {led['proposer_gen_tokens_per_instance']}/instance, "
              f"expansions {led['expansions_per_instance']}")
        fl = led["flops_per_instance"]
        print(f"  F {fl['f']['upper']:.3e} / {fl['f']['lower']:.3e}; proposer {fl['proposer']['upper']:.3e} / "
              f"{fl['proposer']['lower']:.3e} (upper / lower)")
    if not a.iso_ledger_quick:
        for key, spec in REPORT_LEDGERS.items():
            if not (ROOT / spec["run"]).exists():
                continue
            led = beam_ledger(ROOT / spec["run"], ROOT / spec["freeze"])
            out["report_only"][key] = led
            fl = led["flops_per_instance"]["total"]
            print(f"{key} (report-only): upper {fl['upper']:.4e}, lower {fl['lower']:.4e}; F keys "
                  f"{led['f_keys_per_instance']}/instance; success {led['success']}/{led['n']}")
    # Qwen3-4B budgets on hard at the registered cap (and the descriptive fractions)
    path = model_path("qwen3-4b")
    fp = flop_params(path / "config.json")
    insts = X.load_hard_instances()
    prompts = hard_prompt_tokens(path, insts, True, None)
    forced = count_tokens(rust_tokenizer(path), [FORCE_THINK_END])[0]
    base = ISO_REGISTRY["H2_hard"]["flops_per_instance"] or out["registered"]["H2_hard"]["ledger"][
        "flops_per_instance"]["total"]["upper"]
    out["qwen3_4b"] = {"n_nonemb": fp["n_nonemb"], "forced_tokens": forced, "budgets": {}}
    for frac, reserve in ((0.5, a.max_new), (1.0, a.max_new), (2.0, a.max_new)):
        try:
            t = iso_table(fp, prompts, base * frac, forced, reserve)
        except ValueError as e:
            print(f"  qwen3-4b x{frac}: {e}")
            continue
        out["qwen3_4b"]["budgets"][f"x{frac:g}_A{reserve}"] = t
        print(f"  qwen3-4b iso x{frac:g} (cap {t['flop_cap']:.4e}, answer reserve {reserve}): prompt tokens "
              f"{t['prompt_tokens']}, total cap {t['total_cap']}, thinking budget {t['think_budget']} [min, median, max]")
    sub, _ = X.systematic_subset(insts, 40)
    out["hard_subset_40"] = {"ids": [i["id"] for i in sub], "ids_sha256": ids_sha(sub), "registered": HARD_SUBSET_40_SHA}
    print(f"hard systematic subset 40: ids sha256 {ids_sha(sub)} (registered {HARD_SUBSET_40_SHA})")
    if a.iso_ledger_out:
        Path(a.iso_ledger_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.iso_ledger_out}")
    return rc


# ----------------------------------------------------------------------------- scoring

def score_plan(inst: dict, plan: list[tuple]) -> dict:
    dom = inst.get("domain") or B.DOMAIN_PDDL
    val = B.val_check(inst["path"], plan, dom) if plan else False
    sim = B.execute(inst["init"], inst["goal"], plan)["success"] if plan else False
    return {"val_valid": bool(val), "sim_valid": bool(sim)}


def _pddl(p):
    return [f"({' '.join(x)})" for x in p]


def score_pddl(inst: dict, text: str, thinking: bool) -> dict:
    """Hard (PDDL answers): primary = extract_pddl_plan + VAL; second = upstream's save_gpt3_response text + VAL on that
    exact text (pb_*); no markdown variant (pb_md_* None)."""
    plan, xinfo = X.extract_pddl_plan(text, inst["objs"], thinking=thinking)
    sc = score_plan(inst, plan)
    up = X.upstream_pddl_text(text, thinking)
    up_ok = X.val_check_text(inst["path"], up, inst["domain"])
    return {"extracted_plan": _pddl(plan), "val_valid": sc["val_valid"], "sim_valid": sc["sim_valid"],
            "pb_plan": up.split("\n") if up else [], "pb_val_valid": bool(up_ok), "pb_md_plan": None,
            "pb_md_val_valid": None, "extract_info": xinfo, "n_plan": len(plan)}


def score_variants(inst: dict, text: str, cmap: dict, thinking: bool) -> dict:
    """All scorers on one completion; VAL once per distinct plan."""
    if inst.get("fmt") == "pddl":
        return score_pddl(inst, text, thinking)
    plan, xinfo = extract_plan(text, cmap, thinking=thinking)
    pb = planbench_plans(text, thinking=thinking)
    sc = score_plan(inst, plan)
    verdict = {tuple(plan): sc["val_valid"]}
    for p in pb.values():
        if tuple(p) not in verdict:
            verdict[tuple(p)] = bool(B.val_check(inst["path"], p)) if p else False
    return {"extracted_plan": _pddl(plan), "val_valid": sc["val_valid"], "sim_valid": sc["sim_valid"],
            "pb_plan": _pddl(pb["pb_raw"]), "pb_val_valid": verdict[tuple(pb["pb_raw"])],
            "pb_md_plan": _pddl(pb["pb_md"]), "pb_md_val_valid": verdict[tuple(pb["pb_md"])], "extract_info": xinfo,
            "n_plan": len(plan)}


def variant_summary(rows: list[dict]) -> str:
    n = len(rows)
    k = lambda f: sum(bool(r.get(f)) for r in rows)
    dis = sum(r.get("val_valid") != r.get("pb_val_valid") for r in rows)
    cut = sum(not r.get("hit_eos") for r in rows)
    fl = sum(r.get("flops", 0) for r in rows)
    md = [r for r in rows if r.get("pb_md_val_valid") is not None]
    md_s = f"{sum(bool(r['pb_md_val_valid']) for r in md)}/{len(md)}" if md else "n/a"
    over = sum(r.get("within_flop_cap") is False for r in rows)
    return (f"{n} rows | VAL ours {k('val_valid')}/{n}, upstream scorer {k('pb_val_valid')}/{n}, PlanBench parser on "
            f"markdown-normalised text {md_s} | ours!=upstream {dis} | simulator {k('sim_valid')} | answers cut "
            f"{cut} | forced think end {k('forced_think_end')} | cache fallbacks {k('cache_fallback')} | over FLOP cap "
            f"{over} | FLOPs {fl:.3e}")


# ----------------------------------------------------------------------------- run

def resolve_system(a) -> str | None:
    choice = a.system_prompt
    if choice == "auto":
        choice = "none" if a.set == "hard" else "planbench"
    return PLANBENCH_SYSTEM if choice == "planbench" else None


def budget_spec(a) -> dict:
    if a.thinking == "off":
        return {"mode": "greedy_no_thinking", "max_new": a.max_new}
    if a.iso:
        return {"mode": "iso_flop", "iso_key": a.iso, "iso_flops_per_instance": ISO_REGISTRY[a.iso]["flops_per_instance"],
                "iso_fraction": a.iso_fraction, "answer_reserve": a.answer_reserve}
    return {"mode": "fixed_think_budget", "think_budget": a.think_budget, "max_new": a.max_new}


def default_out(a) -> Path:
    if a.thinking == "off":
        tag = "think-off"
    elif a.iso:
        tag = f"think-on_iso-{a.iso}x{a.iso_fraction:g}_A{a.answer_reserve}"
    else:
        tag = f"think-on_B{a.think_budget}"
    sys_tag = "" if resolve_system(a) == (None if a.set == "hard" else PLANBENCH_SYSTEM) else (
        "_nosys" if resolve_system(a) is None else "_pbsys")
    sub = f"_sub{a.subset}" if a.subset else ""
    pt = "" if a.set == "hard" else f"_{a.prompt_type}"
    return ROOT / "runs" / "planbench_qwen" / f"{a.model}_{a.set}{sub}_{tag}_max{a.max_new}{pt}{sys_tag}_s{a.seed}.jsonl"


def load_all(a) -> list[dict]:
    """Every instance of the set family the run draws from (hard: the 110; PB-600: both sets), in id order."""
    if a.set == "hard":
        return X.load_hard_instances()
    return load_instances(SET_ALIAS["all"], a.prompt_type, a.query_source)


def run_config(a, path: Path, all_insts: list[dict], selected: list[dict] | None = None) -> dict:
    """Everything a row's numbers depend on; a resumed file must carry exactly this config."""
    system = resolve_system(a)
    prompt = {"system": system, "chat_template_sha": chat_template_sha(path), "queries_sha": queries_sha(all_insts)}
    if a.set == "hard":
        prompt["upstream_files_sha256"] = dict(X.HARD_RESULT_SHA)
    prompt["sha"] = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
    return {"model": a.model, "dtype": model_dtype_name(a.model), "thinking": a.thinking,
            "think_budget": a.think_budget if (a.thinking == "on" and not a.iso) else None, "max_new": a.max_new,
            "seed": a.seed, "set": a.set,
            "subset": ({"n": a.subset, "rule": "systematic" if a.set == "hard" else f"stratified_seed{a.subset_seed}",
                        "ids_sha256": ids_sha(selected or [])} if a.subset else None),
            "prompt_type": None if a.set == "hard" else a.prompt_type,
            "query_source": "upstream_hard_fc638a1a" if a.set == "hard" else a.query_source,
            "budget": budget_spec(a),
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


def apply_subset(insts: list[dict], a) -> list[dict]:
    """--subset N: hard -> systematic_subset (no seed); PB-600 -> stratified_subset(seed=--subset-seed)."""
    if not a.subset:
        return insts
    return (X.systematic_subset(insts, a.subset)[0] if a.set == "hard"
            else stratified_subset(insts, a.subset, seed=a.subset_seed)[0])


def select(insts: list[dict], a) -> list[dict]:
    """Set -> --subset N (hard: systematic; PB: stratified, --subset-seed) -> --ids -> --limit."""
    insts = apply_subset(insts, a)
    if a.ids:
        spec = a.ids
        if spec.startswith("@"):
            spec = Path(spec[1:]).read_text(encoding="utf-8").replace("\n", ",")
        want = [s.strip() for s in spec.split(",") if s.strip()]
        known = {i["id"]: i for i in insts}
        missing = [w for w in want if w not in known]
        if missing:
            raise SystemExit(f"unknown ids (or not in the subset): {missing[:5]} (format: generated_basic/instance-12, "
                             f"hard/instance-7)")
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


def expected_seconds(a, n_todo: int, rows: list[dict]) -> float:
    """Lane lock's expected_end (informative): median wall time of finished rows, else tokens / --assumed-tok-s."""
    walls = sorted(r["wall_s"] for r in rows if r.get("wall_s"))
    if len(walls) >= 3:
        per = walls[len(walls) // 2]
    else:
        if a.thinking == "off":
            toks = a.max_new
        elif a.iso:
            toks = 6000 * a.iso_fraction
        else:
            toks = a.think_budget + a.max_new
        per = toks / a.assumed_tok_s
    return 300 + n_todo * per


def run(a) -> int:
    all_insts = load_all(a)
    pool = all_insts if a.set in ("hard", "all") else [i for i in all_insts if i["set"] in SET_ALIAS[a.set]]
    insts = select(pool, a)
    path = model_path(a.model)
    cfg = run_config(a, path, all_insts, apply_subset(pool, a))
    out = Path(a.out) if a.out else default_out(a)
    fp = flop_params(path / "config.json")
    if a.dry_run:
        return dry_run(a, insts, cfg, out, fp, path)
    token = acquire_lock(LOCK_PATH, {"cmd": " ".join(sys.argv), "out": str(out), "model": a.model})
    status = "done"
    try:
        while True:
            status = _run_locked(a, insts, cfg, out, fp)
            if status != "yield" or a.lane_no_wait:
                break
            import gc
            gc.collect()  # the model went out of scope in _run_locked; return its RAM before waiting for the lane
            print(f"[{time.strftime('%H:%M:%S')}] yielded the CPU lane (waiting.rcx); model freed; waiting to "
                  f"re-acquire", flush=True)
    finally:
        release_lock(LOCK_PATH, token)
    return EXIT_YIELD if status in ("yield", "busy") else 0


def dry_run(a, insts, cfg, out, fp, path) -> int:
    rows = read_rows(out, cfg, repair=False)
    done = {r["id"] for r in rows}
    todo = [i for i in insts if i["id"] not in done]
    print(f"[{time.strftime('%H:%M:%S')}] DRY RUN {a.model} ({cfg['dtype']}, {path}) set={a.set} thinking={a.thinking} "
          f"budget={cfg['budget']} system={cfg['prompt']['system']!r}: {len(insts)} selected, {len(done)} done, "
          f"{len(todo)} to run -> {out}  N_nonemb={fp['n_nonemb']:,} threads={a.threads}", flush=True)
    if cfg["subset"]:
        print(f"  subset: {cfg['subset']}", flush=True)
    if LOCK_PATH.exists():
        print(f"  run lock present: {LOCK_PATH.read_text(encoding='utf-8')[:200]}", flush=True)
    if rows:
        print("  " + variant_summary(rows), flush=True)
    st = lane_status()
    free = st["lock"] is None and not any(f"waiting.{p}" in w for w in st["waiting"] for p in LANE_PRIORITY)
    print(f"  CPU lane: {st} -> {'free' if free else 'busy: a run would wait (waiting.bwm marker)'}", flush=True)
    mb = available_mb()
    print(f"  RAM: {mb if mb is None else round(mb)} MB available; gate {RAM_GATE_MB} -> "
          f"{'ok' if mb is not None and mb >= RAM_GATE_MB else 'a run would refuse now'}", flush=True)
    pm = peer_problem(a.model, a.force)
    print(f"  peer gate: {'ok' if pm is None else 'a run would refuse now: ' + pm}", flush=True)
    est = expected_seconds(a, len(todo), rows)
    print(f"  expected wall time at {a.assumed_tok_s} tok/s: {est / 3600:.1f} h", flush=True)
    if a.set == "hard" and path.joinpath("tokenizer.json").exists():
        system = cfg["prompt"]["system"]
        prompts = hard_prompt_tokens(path, todo or insts, a.thinking == "on", system)
        pr = sorted(prompts.values())
        print(f"  prompt tokens (template render + tokenizer.json): min {pr[0]} median {pr[len(pr) // 2]} max {pr[-1]}",
              flush=True)
        if a.iso:
            forced = count_tokens(rust_tokenizer(path), [FORCE_THINK_END])[0]
            t = iso_table(fp, prompts, iso_flop_cap(a.iso, a.iso_fraction), forced, a.answer_reserve)
            print(f"  iso budgets (cap {t['flop_cap']:.4e} FLOPs, forced {forced}, answer reserve {a.answer_reserve}): "
                  f"total cap {t['total_cap']}, thinking budget {t['think_budget']} [min, median, max]", flush=True)
    for i in todo[:3]:
        print("  ", i["id"], i["objs"], i["map_info"], flush=True)
    return 0


def _guard_check(a, lane: dict | None):
    def check():
        mb = available_mb()
        if mb is not None and mb < RAM_HARD_FLOOR_MB:
            return f"RAM {mb:.0f} MB < {RAM_HARD_FLOOR_MB} MB during generation"
        pm = peer_problem(a.model, a.force)
        if pm:
            return f"peer job appeared during generation: {pm}"
        if lane is not None and not lane_holds(lane):
            return "the CPU-lane lock is no longer ours"
        return None
    return check


def _run_locked(a, insts: list[dict], cfg: dict, out: Path, fp: dict) -> str:
    """Returns 'done', 'yield' (waiting.rcx seen at an instance boundary) or 'busy' (--lane-no-wait, lane taken)."""
    rows = read_rows(out, cfg, repair=True)
    done = {r["id"] for r in rows}
    todo = [i for i in insts if i["id"] not in done]
    print(f"[{time.strftime('%H:%M:%S')}] {a.model} ({cfg['dtype']}) set={a.set} thinking={a.thinking} "
          f"budget={cfg['budget']}: {len(insts)} selected, {len(done)} done, {len(todo)} to run -> {out}  "
          f"N_nonemb={fp['n_nonemb']:,}", flush=True)
    if not todo:
        return "done"
    try:
        lane = lane_acquire(f"BWM planbench_qwen {a.model} {a.set} {cfg['budget']['mode']} ({len(todo)} instances)",
                            expected_seconds(a, len(todo), rows), wait=not a.lane_no_wait, poll_s=a.lane_poll_s,
                            max_wait_s=a.lane_max_wait_h * 3600 if a.lane_max_wait_h else None)
    except LaneBusy as e:
        print(f"  {e}; exiting {EXIT_YIELD} (resumable)", flush=True)
        return "busy"
    try:
        peer_gate(a.model, a.force)
        ram_gate(a.model)
        t0 = time.time()
        model, tok, _ = load_model(a.model, a.threads)
        from bwmn import lm
        print(f"  loaded in {time.time() - t0:.0f}s, threads={a.threads}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        system = cfg["prompt"]["system"]
        thinking = a.thinking == "on"
        n_forced = len(tok(FORCE_THINK_END, add_special_tokens=False).input_ids)
        cap = iso_flop_cap(a.iso, a.iso_fraction) if a.iso else None
        n_ok = n = 0
        new_rows = []
        with open(out, "a", encoding="utf-8") as fh:
            for inst in todo:
                if lane_should_yield():
                    print(f"  waiting.rcx present: yielding the CPU lane before {inst['id']} "
                          f"({n} done in this pass)", flush=True)
                    return "yield"
                if not lane_holds(lane):
                    raise SystemExit("the CPU-lane lock is no longer ours (reclaimed or deleted); stopping")
                lm.cpu_thermal_guard()
                ram = instance_gates(a, n)
                seed = instance_seed(a.seed, inst["id"])
                prompt = render_prompt(tok, inst["query"], thinking, system)
                n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
                budget = iso_caps(fp, n_prompt, cap, n_forced, a.answer_reserve) if cap is not None else None
                guard = GenGuard(a.guard_every, _guard_check(a, lane))
                try:
                    g = generate_one(model, tok, prompt, thinking=thinking,
                                     think_budget=(budget["think_budget"] if budget else (a.think_budget or 0)),
                                     max_new=a.max_new, seed=seed,
                                     total_cap=budget["total_cap"] if budget else None, guard=guard)
                except GuardStop as e:
                    raise SystemExit(f"stopped during {inst['id']}: {e}; row not written (resumable)")
                check_fallback(g, n, inst["id"])
                sv = score_variants(inst, g["raw_text"], inst["cmap"], thinking)
                terms = FL.flops_breakdown(fp, g["prompt_tokens"], g["gen_tokens"], g["forced_tokens"],
                                           g["reprefill_tokens"])
                row = {"id": inst["id"], **{k: g[k] for k in ("prompt_tokens", "gen_tokens", "wall_s")},
                       "raw_text": g["raw_text"],
                       **{k: sv[k] for k in ("extracted_plan", "val_valid", "sim_valid", "pb_plan", "pb_val_valid",
                                             "pb_md_plan", "pb_md_val_valid")},
                       **{k: g[k] for k in ("think_tokens", "answer_tokens", "answer_cap", "forced_think_end",
                                            "forced_tokens", "think_closed_naturally", "hit_eos", "cache_fallback",
                                            "cache_fallback_errors", "reprefill_tokens", "reprefill_reason")},
                       "flops": terms["total"], "flops_terms": terms, "n_nonemb": fp["n_nonemb"], "seed": seed,
                       "budget": budget, "within_flop_cap": (terms["total"] <= cap) if cap is not None else None,
                       "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                       "prompt_render_match": prompt == render_qwen3_prompt(inst["query"], thinking, system),
                       "ram_mb_before": round(ram or 0), "color_map": inst["cmap"], "map_info": inst["map_info"],
                       "extract_info": sv["extract_info"], "config": cfg, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                new_rows.append(row)
                n += 1
                n_ok += sv["val_valid"]
                print(f"  {inst['id']:<32} p={g['prompt_tokens']} g={g['gen_tokens']}"
                      f"{' F' if g['forced_think_end'] else ''} {g['wall_s']:.1f}s plan={sv['n_plan']} "
                      f"VAL={sv['val_valid']} up={sv['pb_val_valid']} pb_md={sv['pb_md_val_valid']}"
                      f"{'' if sv['val_valid'] == sv['sim_valid'] else ' SIM-DISAGREE'}"
                      f"{'' if g['hit_eos'] else ' CUT'}  [{n_ok}/{n}]", flush=True)
        print("  this run: " + variant_summary(new_rows), flush=True)
        print("  file:     " + variant_summary(rows + new_rows), flush=True)
        return "done"
    finally:
        lane_release(lane)


# ----------------------------------------------------------------------------- subsets (no LM)

def make_subset(a) -> int:
    if a.set == "hard":
        insts = X.load_hard_instances()
        sub, table = X.systematic_subset(insts, a.make_subset)
        rule = "systematic (blocks, optimal length, id)"
    else:
        insts = load_instances(SET_ALIAS[a.set], a.prompt_type, a.query_source)
        sub, table = stratified_subset(insts, a.make_subset, seed=a.subset_seed)
        rule = f"stratified, seed {a.subset_seed}"
    for k, (size, drawn) in table.items():
        print(f"  {k!s:<16} {drawn:>3} / {size}", file=sys.stderr)
    text = "".join(i["id"] + "\n" for i in sub)
    print(f"{len(sub)} ids ({len(table)} strata, {rule}), sha256 {ids_sha(sub)}", file=sys.stderr)
    if a.subset_out:
        Path(a.subset_out).write_text(text, encoding="utf-8")
        print(f"-> {a.subset_out}", file=sys.stderr)
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

# PDDL-format answers (Blocksworld-hard): (text, objects, expected plan)
PDDL_CASES = [
    ("(unstack d b)\n(put-down d)\n(pick-up a)\n(stack a d)", "abcd",
     [("unstack", "d", "b"), ("put-down", "d"), ("pick-up", "a"), ("stack", "a", "d")]),
    ("<think>\n(unstack a b) maybe\n</think>\n\n```lisp\n(unstack d b)\n(put-down d)\n```", "abcd",
     [("unstack", "d", "b"), ("put-down", "d")]),
    ("<think>\nnever closed (unstack d b)", "abcd", []),
    ("**Plan:**\n1. (unstack d b)\n2. (put-down d)\n\n**Verification:**\n1. (unstack d b): d is clear, hand empty.",
     "abcd", [("unstack", "d", "b"), ("put-down", "d")]),
    ("First attempt:\n(unstack d b)\n(stack d a)\n\nCorrected plan:\n(unstack d b)\n(put-down d)", "abcd",
     [("unstack", "d", "b"), ("put-down", "d")]),
    ("(unstack d b) ; d was clear\n(put-down d table)\n(pickup a)\n(stack a d)", "abcd",
     [("unstack", "d", "b"), ("pick-up", "a"), ("stack", "a", "d")]),
    ("Plan: (unstack d b) (put-down d)", "abcd", [("unstack", "d", "b"), ("put-down", "d")]),
    ("1. unstack d b\n2. put-down d\nThen stack them all.", "abcd", [("unstack", "d", "b"), ("put-down", "d")]),
    ("```\n(unstack d b)\n(put-down d)\n```\n\n### Check\n```\n(unstack d b) -> holding d\n```", "abcd",
     [("unstack", "d", "b"), ("put-down", "d")]),
    ("Initial state:\n(on d b)\n(clear d)\n\nPlan:\n(unstack d b)\n(put-down d)", "abcd",
     [("unstack", "d", "b"), ("put-down", "d")]),
]


def _selftest_hard() -> int:
    """Hard-set checks: loader (110 queries byte-identical to upstream's builder in all three files), the
    save_gpt3_response port vs every published step-0 llm_plan, local VAL on that exact text vs the published VAL
    verdict, our primary extractor's verdict, and simulator vs VAL."""
    t0 = time.time()
    insts = X.load_hard_instances()
    by = {i["instance_id"]: i for i in insts}
    print(f"\nhard: {len(insts)} instances; every recorded step-0 query (3 files) == hard_query(local problem); files: "
          + ", ".join(f"{m} {X.hard_result_path(m).parent.name}/{X.HARD_RESULT_SHA[m][:12]}" for m in X.HARD_MODELS))
    bad = 0
    for m in X.HARD_MODELS:
        c = {"n": 0, "port": 0, "val_pub": 0, "pub": 0, "prim_val": 0, "prim_agree": 0, "prim_same": 0, "sim": 0}
        for r in X.load_hard_results(m)["instances"]:
            inst = by[int(r["instance_id"])]
            content, llm_plan, pub = X.hard_step0(r)
            up = X.upstream_pddl_text(content)
            v_up = X.val_check_text(inst["path"], up, X.HARD_DOMAIN)
            prim, _ = X.extract_pddl_plan(content, inst["objs"])
            v_pr = bool(B.val_check(inst["path"], prim, X.HARD_DOMAIN)) if prim else False
            sim = B.execute(inst["init"], inst["goal"], prim)["success"] if prim else False
            c["n"] += 1
            c["port"] += up == llm_plan
            c["val_pub"] += v_up == pub
            c["pub"] += bool(pub)
            c["prim_val"] += v_pr
            c["prim_agree"] += v_pr == pub
            c["prim_same"] += prim == X.pddl_text_actions(up)
            c["sim"] += sim == v_pr
        print(f"  {m:<16} n={c['n']} port==published llm_plan {c['port']}; local VAL(text)==published verdict "
              f"{c['val_pub']} (published valid {c['pub']}); primary valid {c['prim_val']}, verdict==published "
              f"{c['prim_agree']}, same actions {c['prim_same']}; simulator==VAL {c['sim']}")
        bad += (c["port"] != c["n"]) + (c["val_pub"] != c["n"]) + (c["sim"] != c["n"])
    print(f"  ({time.time() - t0:.0f}s)")
    return 1 if bad else 0


def _selftest_lane() -> int:
    """Lane protocol in a temporary directory with fake pids (never the real %TEMP% lane)."""
    fails = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rec = lane_try_acquire("selftest", 60, d, log=lambda m: None)
        lock = json.loads(lane_lock_path(d).read_text())
        if not rec or set(lock) != {"holder", "job", "pid", "started", "expected_end", "token"} or lock["holder"] != "bwm":
            fails.append("acquire/fields")
        if lane_try_acquire("second", 60, d, log=lambda m: None) is not None:
            fails.append("second acquire while held")
        if not lane_release(rec, d) or lane_lock_path(d).exists():
            fails.append("release")
        lane_lock_path(d).write_text(json.dumps({"holder": "rcx", "pid": 4_000_000_000, "started": time.time(),
                                                 "job": "x", "expected_end": 0, "token": "t"}))
        r2 = lane_try_acquire("after-stale", 60, d, log=lambda m: None)
        if not r2 or not (d / f"{LANE_FILE}.stale-4000000000.bak").exists():
            fails.append("stale reclaim")
        lane_release(r2, d)
        lane_waiting_path("rcx", d).write_text("{}")
        if lane_try_acquire("blocked", 60, d, log=lambda m: None) is not None or not lane_should_yield(d):
            fails.append("waiting.rcx respected")
    print(f"lane lock (temp dir, fake pids): {'ok' if not fails else 'FAIL ' + str(fails)}")
    return 1 if fails else 0


def selftest(a) -> int:
    ledger_p = B.BENCH / "LEDGER.json"
    ledger = json.loads(ledger_p.read_text()) if ledger_p.exists() else {}
    models = [m.strip() for m in a.selftest_models.split(",") if m.strip()]
    rc = 0
    pb_fail = 0
    if a.set != "hard":
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
                    verdict = {tuple(theirs): bool(v_theirs)}  # identical PDDL plan on the identical problem
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
            print(f"\n{model}: n={tot['n']}  exact action-sequence agreement {tot['seq']}/{n} "
                  f"({100 * tot['seq'] / n:.1f}%)  VAL-verdict agreement {tot['val']}/{n} ({100 * tot['val'] / n:.1f}%)")
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
    if a.set in ("all", "hard"):
        rc |= _selftest_hard()
    # unit cases (synthetic answers in styles the comparators are expected to produce)
    cm = {"red": "a", "blue": "b", "orange": "c", "yellow": "d"}
    fails = 0
    for txt, want in UNIT_CASES:
        got, info = extract_plan(txt, cm)
        if got != want:
            fails += 1
            print(f"  UNIT FAIL: {txt[:70]!r}\n     want {want}\n     got  {got} {info}")
    for txt, objs, want in PDDL_CASES:
        got, info = X.extract_pddl_plan(txt, list(objs))
        if got != want:
            fails += 1
            print(f"  PDDL UNIT FAIL: {txt[:70]!r}\n     want {want}\n     got  {got} {info}")
    print(f"\nunit cases: NL {len(UNIT_CASES)} + PDDL {len(PDDL_CASES)}; {fails} fail")
    # FLOP parameter count: config formula vs safetensors headers; closed form vs per-token sum
    for name in MODEL_CHOICES:
        try:
            p = model_path(name)
            fp = flop_params(p / "config.json")
            n_st = safetensors_nonemb(p)
            print(f"N_nonemb {name}: config {fp['n_nonemb']:,} vs safetensors {n_st:,} "
                  f"({'match' if n_st == fp['n_nonemb'] else 'DIFF'}); d={fp['d_model']} vocab={fp['vocab']} "
                  f"L={fp['n_layers']} H={fp['n_heads']} d_head={fp['head_dim']}")
        except (SystemExit, FileNotFoundError) as e:
            print(f"N_nonemb {name}: {e}")
    fp = {"n_nonemb": 7, "d_model": 3, "vocab": 5, "n_layers": 2, "n_heads": 2, "head_dim": 4}
    brute = sum(2 * 7 + 4 * 2 * 2 * 4 * (t + 1) for t in range(40 + 3 + 25)) + 2 * 3 * 5 * 25 \
        + sum(2 * 7 + 4 * 2 * 2 * 4 * (t + 1) for t in range(40 + 10))
    flop_ok = flops(fp, 40, 3, 25, reprefill=50) == brute
    T = FL.max_total_tokens(fp, 40, brute, 3)
    inv_ok = T is not None and FL.flops_breakdown(fp, 40, T - 43, 3)["total"] <= brute < \
        FL.flops_breakdown(fp, 40, T - 42, 3)["total"]
    sp_ok = FL.shared_prefix_breakdown(fp, 40, [25]) == FL.flops_breakdown(fp, 40, 25)
    print(f"FLOPs closed form vs per-token sum: {'match' if flop_ok else 'DIFF'}; max_total_tokens inverse: "
          f"{'ok' if inv_ok else 'FAIL'}; shared prefix (1 sample) == single sequence: {'ok' if sp_ok else 'FAIL'}")
    rc |= _selftest_lane()
    return rc or (1 if fails or pb_fail or not (flop_ok and inv_ok and sp_ok) else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=MODEL_CHOICES)
    ap.add_argument("--set", choices=SET_CHOICES, default="all", help="PB-600 (all|bw3|bw) or Blocksworld-hard (hard)")
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--think-budget", type=int, help="fixed thinking-token budget B (--thinking on, without --iso)")
    ap.add_argument("--iso", choices=sorted(ISO_REGISTRY), help="iso-FLOP thinking budget from a registered ledger")
    ap.add_argument("--iso-fraction", type=float, default=1.0, help="FLOP cap = fraction x the registered budget")
    ap.add_argument("--answer-reserve", type=int, help="iso mode: answer tokens kept after forced think end "
                                                       "(default --max-new)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW_MIN,
                    help=f"answer tokens (after </think> when thinking is on); >= {MAX_NEW_MIN} unless --allow-short")
    ap.add_argument("--allow-short", action="store_true", help=f"permit --max-new < {MAX_NEW_MIN} (smoke tests only)")
    ap.add_argument("--system-prompt", choices=["auto", "planbench", "none"], default="auto",
                    help=f"auto = none on hard (what o1/R1 were sent), planbench on PB-600; planbench = "
                         f"{PLANBENCH_SYSTEM!r}; none = user turn only")
    ap.add_argument("--subset", type=int, help="hard: systematic subset of N; PB: stratified subset (--subset-seed)")
    ap.add_argument("--ids", help="comma-separated ids (generated_basic/instance-12, hard/instance-7) or @file")
    ap.add_argument("--limit", type=int, help="first N selected instances")
    ap.add_argument("--prompt-type", choices=sorted(TASK_FILE), default="zero_shot", help="PB-600 only")
    ap.add_argument("--query-source", default="o1-preview_chat",
                    help="PB-600: published result dir to read 'query' from (13/14 models share these exact bytes; "
                         "gpt-4_chat differs only by trailing whitespace). Hard: fixed (all three files identical)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=MAX_THREADS)
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true", help="run qwen3-4b although other native_v1 python jobs run")
    ap.add_argument("--ram-floor", type=int, default=RAM_FLOOR_MB, help="available MB required before each instance")
    ap.add_argument("--guard-every", type=int, default=256, help="in-generation RAM/peer/lane checks every N tokens")
    ap.add_argument("--lane-no-wait", action="store_true", help=f"exit {EXIT_YIELD} instead of waiting for the lane")
    ap.add_argument("--lane-poll-s", type=float, default=60.0)
    ap.add_argument("--lane-max-wait-h", type=float, default=0.0, help="give up waiting after H hours (0 = never)")
    ap.add_argument("--assumed-tok-s", type=float, default=4.0, help="for expected_end / dry-run estimates only")
    ap.add_argument("--dry-run", action="store_true", help="resolve paths, instances, resume state, gates; no model")
    ap.add_argument("--make-subset", type=int, metavar="N", help="write a subset of N ids (no model)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--subset-out", help="file for --make-subset ids (default: stdout); use with --ids @file")
    ap.add_argument("--iso-ledger", action="store_true", help="BWM compute ledger + Qwen iso budgets; no model")
    ap.add_argument("--iso-ledger-quick", action="store_true", help="--iso-ledger without the report-only B2 ledgers")
    ap.add_argument("--iso-ledger-out", help="write the ledger JSON here")
    ap.add_argument("--selftest", action="store_true", help="extractor tests on published responses; no model")
    ap.add_argument("--selftest-models", default="gpt-4_chat,o1-preview_chat")
    ap.add_argument("--show", type=int, default=0, help="selftest: print N disagreements per model")
    ap.add_argument("--show-raw", type=int, default=0, help="selftest: print first N chars of each shown raw response")
    a = ap.parse_args(argv)
    if a.answer_reserve is None:
        a.answer_reserve = a.max_new
    if a.threads <= 0 or a.threads > MAX_THREADS:
        ap.error(f"--threads must be in 1..{MAX_THREADS} (one CPU lane at <= {MAX_THREADS} threads, D-64/D-76)")
    cpu_only_env(a.threads)
    if a.selftest:
        return selftest(a)
    if a.iso_ledger:
        return cmd_iso_ledger(a)
    if a.make_subset is not None:
        return make_subset(a)
    if not a.model:
        ap.error("--model is required (or use --selftest / --make-subset / --iso-ledger)")
    if a.thinking == "on" and not a.iso and not (a.think_budget and a.think_budget > 0):
        ap.error("--thinking on requires --think-budget B > 0 or --iso KEY")
    if a.iso and a.thinking != "on":
        ap.error("--iso applies to --thinking on")
    if a.iso and a.think_budget is not None:
        ap.error("--iso and --think-budget are exclusive")
    if a.thinking == "off" and a.think_budget is not None:
        ap.error("--think-budget only applies with --thinking on")
    if not 0 < a.iso_fraction <= 16:
        ap.error("--iso-fraction must be in (0, 16]")
    if a.max_new <= 0 or (a.limit is not None and a.limit <= 0) or (a.subset is not None and a.subset <= 0) \
            or a.answer_reserve <= 0 or a.guard_every <= 0:
        ap.error("--max-new, --limit, --subset, --answer-reserve and --guard-every must be positive")
    if (a.max_new < MAX_NEW_MIN or a.answer_reserve < MAX_NEW_MIN) and not a.allow_short:
        ap.error(f"--max-new/--answer-reserve below {MAX_NEW_MIN}: answers would be truncated (PlanBench's chat "
                 f"engines had no cap); pass --allow-short for smoke tests or descriptive budget points")
    if a.set != "hard" and (a.prompt_type not in TASK_FILE
                            or not (RES / "blocksworld" / a.query_source / TASK_FILE[a.prompt_type]).exists()):
        ap.error(f"no {TASK_FILE[a.prompt_type]} under results/*/{a.query_source}")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
