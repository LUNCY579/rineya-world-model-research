"""T2b-ledger dev screen: GPU runner (train seeds 0, 1, 2 on the T2 recipe; evaluate OFF / T2 s0 / T2b-ledger ON).

  python -B scripts/t2bl_run.py --plan      # CPU only, no torch, no model, no lease file read: generations, GPU time
                                            # per stage and per 4.5 h turn (planning and observed rates); writes
                                            # docs/t2b/ledger/GEN_PLAN.json (refused once a runs/t2bl output exists)
  python -B scripts/t2bl_run.py             # the GPU job (tag 'dev', seeds 0,1,2). Refuses before the lease unless the
                                            # dev gate rule is frozen (docs/t2b/ledger/GATE_RULE.json, written by
                                            # `t2bl_gate.py --freeze-rule`) and every frozen input matches it.

Recipe (T2, DECISIONS D-15/D-16, unchanged): the vendored BWM-001 trainer (vendor_bwm/scripts/train_generic.py, pinned,
invoked exactly as scripts/run_t2.py invokes it): LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down of
frozen FP32 Titan-Sparse, WSD schedule, lr 1e-4, batch 16 (4 x 4 token-weighted microbatches), 190 steps, duty-capped,
honours the shared thermal flag. Rows: data/t2bl_train.jsonl (1,500 ledger one-step rows, prompt + target), the SAME
file for every seed; the trainer's seeded row order differs by seed (random.Random(10007 * seed + 17)). One adapter
per seed, never merged (D-28).

Evaluation (D-19 instrument style; one process per load): Titan-Sparse FP32 (runtime.load_arm('sparse'), fingerprint
818be9c6b6a3) + T2 s0 (runs/t2/t2_s0/adapter, PREREG-2's pins) as PEFT adapter 't2' + the seed's adapter as 't2bl'
(load_adapter), unmerged; sides OFF = disable_adapter(), T2s0 = set_adapter('t2'), ON = set_adapter('t2bl'). Greedy,
rcwmt.generation.generate_texts (160-token cap, its batching), prompts t2bl_common.messages(before, event, surface).
  A0  once per turn, before the turn's first evaluation stage: PREREG-2's anchor (the 208 pilot TEST ledger chain_one
      chains, rcwmt.generation.chain) OFF and T2s0 against runs/t2/t2_s0/gen_chain_one_base.jsonl / gen_chain_one.jsonl
      (PREREG-2 pins): identical states on >= 99% of chains per side, else the job stops (persisted) before any
      evaluation output of that turn. It also shows that loading a second adapter leaves OFF and T2s0 unchanged.
  E0  once: the 1,801 dev rows (19 strata, every surface of the bank) OFF and T2s0 (the two reference sides).
  T   per seed: training (stage T; the adapter directory).
  E   per seed: the 1,801 dev rows ON (the seed's adapter).
The rows, sides and strata the gate reads are fixed in docs/t2b/ledger/GATE_RULE.md (pre-stated before any output).

Pairing (D-22 style, ONE lease 'rcwm-t1:t2b-ledger', 4.5 h cap from acquisition): seeds run in order 0, 1, 2. A seed's
training starts only if 1.25 x (its remaining training + evaluation [+ E0 + A0 if still due] + model load) fits the
remaining turn and never within 0.5 h of the cap; each evaluation stage starts only if 1.25 x its estimate fits.
Otherwise that seed (and every later one) is deferred: exit 5, and the next call resumes from the recorded stages
(re-anchoring first). Estimates are corrected by the observed / planned ratio of the stages already run.

Before the lease (exit 3, not persisted): the frozen gate rule and every frozen input (incl. the hook draft; every
installed scripts/queue_hooks/*t2bl*.sh, documented name 88_t2bl.sh, must be that draft byte for byte); the training file and dev bank
equal their SOURCE records; T2 s0's adapter equals PREREG-2's pins; the vendored trainer is the pinned one; every
training prompt == tok.apply_chat_template(messages(row)); the one-waiter rule (R14: another RCWM job waiting for or
holding the GPU lease refuses this one; BWM jobs are waited for in the lease); a one-runner lock on runs/t2bl/<tag>/.
A resumed run refuses if a frozen code file differs from the hashes recorded at its first turn. Outputs are create-only
under runs/t2bl/<tag>/; a stage is DONE only if its files exist AND provenance.json records them with matching sha256;
anything else a stage wrote is PARTIAL: moved to _orphans/ (never deleted) and redone in full. A recorded output whose
hash changes stops the job.

Two consecutive turns that record no stage stop the job (persisted; a cap or rate amendment is needed). --cap-h is
clamped to 4.5 h (the agreed turn cap).

Exit codes: 0 complete (DONE.json) or already complete; 3 deterministic stop (never retried; persisted as STOPPED.json
when it concerns the run); 5 deferred by the turn cap (resumable); 1 crash (an exception; the lease is released).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if "--plan" in sys.argv:                           # never touch the driver: empty string, NEVER "-1" (I-21)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_gen as G2  # noqa: E402  (torch-free at import: D-22 stage rule, pins, registered runtime, pilot loader)
import t2bl_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

# ---------------------------------------------------------------- constants
LEASE_LABEL = "rcwm-t1:t2b-ledger"
RCWM_LABEL_PREFIX = "rcwm-t1:"
TAG = "dev"
SEEDS = (0, 1, 2)
CAP_H = 4.5
STEPS, LR = 190, 1e-4
ADAPTER_T2 = "t2"                                  # T2 s0 (PREREG-2's adapter name)
ADAPTER_ON = "t2bl"                                # the seed's adapter
SIDES_E0 = ("off", "t2s0")
FP = dict(G2.FP)
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
VRAM_EMPTY_MIB = 512.0
T2S0_DIR = ROOT / "runs" / "t2" / "t2_s0" / "adapter"
T2S0_PINS = {"adapter_model.safetensors": G2.ADAPTER_SHA, "adapter_config.json": G2.ADAPTER_CONFIG_SHA}
ANCHOR_REFS = {"off": ROOT / "runs" / "t2" / "t2_s0" / "gen_chain_one_base.jsonl",
               "t2s0": ROOT / "runs" / "t2" / "t2_s0" / "gen_chain_one.jsonl"}
ANCHOR_REF_SHA = {"off": G2.ANCHOR_REF_SHA["gen_chain_one_base.jsonl"], "t2s0": G2.ANCHOR_REF_SHA["gen_chain_one.jsonl"]}
ANCHOR_MIN_IDENTICAL = 0.99
N_ANCHOR_CHAINS, ANCHOR_STEPS = 208, 4
VENDOR = {"vendor_bwm/scripts/train_generic.py": "816d2f562946177397a5f186e9fcadd7febd2b49c894afc9abafe9474665ad61",
          "vendor_bwm/bwmn/lm.py": "3d3f8d00b84af39dabd46b415ae559754066bdcf3c64de1127b138a27da72b57"}
FROZEN_CODE = ("scripts/t2bl_common.py", "scripts/t2bl_make_data.py", "scripts/t2bl_audit.py", "scripts/t2bl_run.py",
               "scripts/t2bl_gate.py", "scripts/p3_common.py", "scripts/confirm2_gen.py", "rcwmt/env.py",
               "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/capture.py", *VENDOR)
FROZEN_DATA = ("data/t2bl_train.jsonl", "data/t2bl_train.SOURCE.json", "data/t2bl_dev.jsonl", "data/t2bl_dev.SOURCE.json",
               "docs/t2b/ledger/GATE_RULE.md", "docs/t2b/ledger/queue_hook_t2bl.sh.draft")
GATE_RULE_JSON = C.DOCS / "GATE_RULE.json"
HOOK_DRAFT = C.DOCS / "queue_hook_t2bl.sh.draft"
HOOK_INSTALLED = ROOT / "scripts" / "queue_hooks" / "88_t2bl.sh"
# timing model (no model is loaded to make it). Training: T2's three 190-step trainings took 1859 / 2191 / 1656 s
# (runs/t2/*/log.txt) on rows of 152.76 tokens (incl. EOS; t2bp's token statistics of data/t2_train.jsonl); scaled by
# this file's tokens per row. Generation: PREREG-3's S1 (the same frame, 6,000 one-step generations on in- and
# out-of-range ledger rows over four sides incl. OFF) ran at 6,000 / 1,229 s = 4.88 gens/s with ~20-token golds
# (observed); planning uses PREREG-2/3's registered 2.7 gens/s. Both scale with (answer tokens + 1) / 21.
T2_TRAIN_S_MAX = 2191.3
T2_TRAIN_S_MEAN = (1859.2 + 2191.3 + 1656.1) / 3
T2_ROW_TOKENS = 152.76
OBS_RATE, PLAN_RATE, REF_ANSWER_TOKENS = 6000 / 1229.0, 2.7, 20.0
ANCHOR_ANSWER_TOKENS = 20.0
LOAD_S = 120.0
NO_PROGRESS_STOP = 2


class Stop(Exception):
    """A deterministic stop (exit 3, never retried). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


class Deferred(Exception):
    pass


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[t2bl] {msg}", persist)


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- configuration
@dataclass
class Cfg:
    root: Path = ROOT
    tag: str = TAG
    seeds: tuple = SEEDS
    out: Path | None = None
    train: Path = C.TRAIN
    train_source: Path = C.TRAIN_SOURCE
    dev: Path = C.DEV
    dev_source: Path = C.DEV_SOURCE
    steps: int = STEPS
    lr: float = LR
    cap_h: float = CAP_H
    planning: bool = True
    anchor_chains: list | None = None              # None: the pilot TEST ledger chains (208)
    anchor_refs: dict = field(default_factory=lambda: dict(ANCHOR_REFS))
    anchor_ref_sha: dict | None = field(default_factory=lambda: dict(ANCHOR_REF_SHA))
    t2s0_dir: Path = T2S0_DIR
    t2s0_pins: dict | None = field(default_factory=lambda: dict(T2S0_PINS))
    dev_subset: int | None = None                 # n rows per stratum: tiny tests only
    check_vendor: bool = True
    rule_record: Path | None = GATE_RULE_JSON      # tests pass a temporary record (or None for mechanics only)

    def __post_init__(self):
        if self.out is None:
            self.out = C.OUT_ROOT / self.tag


def pilot_anchor_chains() -> list:
    return G2.load_bank_chains(G2.PILOT_PATH, G2.PILOT_SHA, "L", "test")


def load_dev(cfg: Cfg) -> list:
    rows = C.load_jsonl(cfg.dev)
    if cfg.dev_subset:
        keep = []
        for s in C.DEV_STRATA:
            keep += [r for r in rows if r["stratum"] == s][:cfg.dev_subset]
        rows = keep
    return rows


def dev_messages(rows: list) -> list:
    return [C.messages(r["before"], r["event"], r["surface"]) for r in rows]


# ---------------------------------------------------------------- estimates (no torch)
def s_per_gen(answer_tokens: float, planning: bool) -> float:
    rate = PLAN_RATE if planning else OBS_RATE
    return (answer_tokens + 1.0) / (REF_ANSWER_TOKENS + 1.0) / rate


def estimates(cfg: Cfg, n_rows: int | None = None, n_anchor: int | None = None) -> dict:
    src = C.read_json(cfg.train_source) or C.read_json(C.TRAIN_SOURCE) or {}
    ts = src.get("token_stats") or {}
    row_tok = (ts.get("train") or {}).get("mean_row_tokens_incl_eos", 165.0)
    ans = (ts.get("dev") or {}).get("mean_target_tokens", 25.0)
    n_rows = n_rows if n_rows is not None else sum(v["n"] for v in (C.read_json(C.DEV_SOURCE) or {}).get("strata", {}).values()) or 1801
    n_anchor = n_anchor if n_anchor is not None else N_ANCHOR_CHAINS * ANCHOR_STEPS
    t2 = T2_TRAIN_S_MAX if cfg.planning else T2_TRAIN_S_MEAN
    return {"T": LOAD_S + t2 * (row_tok / T2_ROW_TOKENS) * (cfg.steps / STEPS),
            "A0": 2 * n_anchor * s_per_gen(ANCHOR_ANSWER_TOKENS, cfg.planning),
            "E0": 2 * n_rows * s_per_gen(ans, cfg.planning), "E": n_rows * s_per_gen(ans, cfg.planning),
            "load": LOAD_S, "gens": {"A0": 2 * n_anchor, "E0": 2 * n_rows, "E": n_rows},
            "inputs": {"train_row_tokens": row_tok, "dev_answer_tokens": ans, "n_rows": n_rows, "n_anchor_steps": n_anchor}}


def seed_todo(states: dict, seed: int) -> list:
    return [s for s in ("T", "E") if states[f"s{seed}"][s] != "done"]


def pair_estimate(est: dict, todo: list, need_e0: bool, anchored: bool, corr: dict) -> float:
    t = corr["T"] * est["T"] if "T" in todo else 0.0
    if "E" in todo or need_e0:
        t += est["load"] + (0.0 if anchored else corr["E"] * est["A0"])
        t += corr["E"] * (est["E0"] if need_e0 else 0.0) + (corr["E"] * est["E"] if "E" in todo else 0.0)
    return t


def simulate_turns(cfg: Cfg, est: dict | None = None) -> list:
    est = est or estimates(cfg)
    cap = cfg.cap_h * 3600
    todo = {s: ["T", "E"] for s in cfg.seeds}
    e0 = True
    turns = []
    while any(todo.values()) or e0:
        t, ran, anchored = 0.0, [], False
        for s in cfg.seeds:
            if not todo[s] and not e0:
                continue
            pair = pair_estimate(est, todo[s], e0, anchored, {"T": 1.0, "E": 1.0})
            if not G2.can_start(t, pair, 0.0, cap):
                break
            if "T" in todo[s]:
                t += est["T"]
                todo[s].remove("T")
                ran.append(f"s{s}:T")
            t += est["load"]
            if not anchored:
                t += est["A0"]
                anchored = True
                ran.append("A0")
            if e0:
                t += est["E0"]
                e0 = False
                ran.append("E0")
            if "E" in todo[s]:
                t += est["E"]
                todo[s].remove("E")
                ran.append(f"s{s}:E")
        if not [r for r in ran if r != "A0"]:
            raise RuntimeError("no stage fits an empty turn")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(verbose: bool = True) -> dict:
    res = {"lease": LEASE_LABEL, "cap_h": CAP_H, "seeds": list(SEEDS), "steps": STEPS,
           "stage_rule": (f"a seed's training starts only if {G2.STAGE_SAFETY} x its whole remaining pair (training + "
                          "model load + [A0 + E0 if due] + its evaluation) fits the turn and never within "
                          f"{G2.STAGE_MARGIN_H} h of the cap; each evaluation stage starts only if {G2.STAGE_SAFETY} x "
                          "its estimate fits; otherwise exit 5 and resume next turn (re-anchored)"),
           "timing_model": {"train": "T2 190-step training (planning max 2191 s; observed mean 1902 s) x tokens/row / "
                                     "152.76", "generation": "planning 2.7 gens/s, observed 4.88 gens/s (PREREG-3 S1), "
                                     "both x (answer tokens + 1) / 21", "load_s": LOAD_S}}
    for mode in ("planning", "observed"):
        cfg = Cfg(planning=(mode == "planning"))
        est = estimates(cfg)
        turns = simulate_turns(cfg, est)
        res[mode] = {"stage_s": {k: round(v, 1) for k, v in est.items() if isinstance(v, float)},
                     "generations": est["gens"], "inputs": est["inputs"], "turns": turns,
                     "gpu_h": round(sum(t["projected_h"] for t in turns), 2),
                     "per_seed_pair_h": round((est["T"] + est["load"] + est["E"]) / 3600, 2)}
    res["generations_total"] = (res["planning"]["generations"]["E0"] + len(SEEDS) * res["planning"]["generations"]["E"]
                                + res["planning"]["generations"]["A0"] * len(res["planning"]["turns"]))
    if verbose:
        for mode in ("planning", "observed"):
            r = res[mode]
            print(f"[{mode}] " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in r["stage_s"].items()))
            print("  " + "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(r["turns"]))
                  + f"  (total {r['gpu_h']} h)")
    return res


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(unit: str, st: str, out: Path) -> list:
    d = Path(out) / unit
    if unit == "ref" and st == "E0":
        return [d / f"dev_{s}.jsonl" for s in SIDES_E0]
    if st == "T":
        return [d / "train" / "adapter" / "adapter_model.safetensors", d / "train" / "adapter" / "adapter_config.json"]
    if st == "E":
        return [d / "dev_on.jsonl"]
    raise KeyError((unit, st))


def stage_owned(unit: str, st: str, out: Path) -> list:
    if st == "T":
        return [Path(out) / unit / "train"]
    fs = stage_required(unit, st, out)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path) -> str:
    return str(Path(p).relative_to(Path(out))).replace("\\", "/")


def units(cfg: Cfg) -> list:
    return [("ref", "E0")] + [(f"s{s}", st) for s in cfg.seeds for st in ("T", "E")]


def stage_states(prov: dict, out: Path, cfg: Cfg) -> dict:
    res: dict = {}
    for u, st in units(cfg):
        req = stage_required(u, st, out)
        rec = ((prov.get("stages") or {}).get(u) or {}).get(st)
        if all(p.exists() for p in req) and rec:
            files = rec.get("files") or {}
            for p in req:
                r = rel_out(p, out)
                need(r in files, f"{u}/{st}: {r} not recorded in provenance")
                need(C.sha256(p) == files[r], f"{u}/{st}: {r} changed after it was recorded (sha256)")
            res.setdefault(u, {})[st] = "done"
        elif any(p.exists() for p in stage_owned(u, st, out)):
            res.setdefault(u, {})[st] = "partial"
        else:
            res.setdefault(u, {})[st] = "todo"
    for s in cfg.seeds:
        if res[f"s{s}"]["T"] != "done" and res[f"s{s}"]["E"] == "done":
            raise Stop(f"[t2bl] s{s}: an evaluation is recorded without a recorded adapter")
    return res


def orphan(unit: str, st: str, out: Path) -> Path:
    dst = Path(out) / "_orphans" / f"{unit}_{st}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    dst.mkdir(parents=True, exist_ok=False)
    for p in stage_owned(unit, st, out):
        if p.exists():
            shutil.move(str(p), str(dst / p.name))
    print(f"[t2bl] {unit}/{st} was partly written; moved to {dst}", flush=True)
    return dst


def write_jsonl(path: Path, rows) -> None:
    if Path(path).exists():
        raise Stop(f"[t2bl] {path} exists (create-only)")
    C.write_create(path, C.dumps_rows(rows))


# ---------------------------------------------------------------- run lock, one-waiter rule
def acquire_run_lock(out: Path) -> Path:
    import psutil
    out.mkdir(parents=True, exist_ok=True)
    lk = out / ".run.lock"
    if lk.exists():
        rec = C.read_json(lk) or {}
        alive = False
        try:
            alive = psutil.pid_exists(int(rec.get("pid", -1))) and \
                abs(psutil.Process(int(rec["pid"])).create_time() - float(rec.get("create_time", 0))) < 1.0
        except (psutil.Error, ValueError, TypeError, KeyError):
            alive = False
        if alive:
            raise Stop(f"[t2bl] another t2bl_run.py holds the run lock {rec}", persist=False)
        print(f"[t2bl] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink(missing_ok=True)
    with lk.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return lk


def _live_other(pid, since) -> bool:
    import psutil
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid == os.getpid() or not psutil.pid_exists(pid):
        return False
    try:
        ct = psutil.Process(pid).create_time()
    except psutil.Error:
        return False
    try:
        return since is None or ct <= float(since) + 1.0
    except (TypeError, ValueError):
        return True


def one_waiter_problem(backend) -> str | None:
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for the GPU lease or holds it. BWM jobs are fine."""
    w = C.read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = C.read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and _live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


def runtime_problem(rt: dict, versions: dict | None = None) -> str | None:
    bad = {k: rt.get(k) for k in ("torch", "transformers") if rt.get(k) != REGISTERED_RUNTIME[k]}
    if versions is not None and versions.get("peft") != REGISTERED_RUNTIME["peft"]:
        bad["peft"] = versions.get("peft")
    return f"runtime {bad} differs from the registered {REGISTERED_RUNTIME}" if bad else None


# ---------------------------------------------------------------- model-facing pieces (torch imported lazily)
class RealBackend:
    """The GPU path: rcwmt.runtime (lease, numerics, loads), the vendored trainer, PEFT as run_t2.py / p3_run.py use it."""
    device = "cuda"

    def __init__(self):
        import torch  # noqa: F401
        from rcwmt import runtime
        self.rt = runtime

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 T2b-ledger dev screen (train seeds 0-2, eval OFF/T2s0/ON)")

    def waiting_marker_path(self) -> Path:
        return Path(self.rt.WAITING["rcwm"])

    def lock_path(self) -> Path:
        return Path(self.rt.LOCK_PATH)

    def lease_acquired_unix(self) -> tuple:
        try:
            rec = json.loads(Path(self.rt.LOCK_PATH).read_text(encoding="utf-8"))
            if rec.get("label") == LEASE_LABEL and rec.get("pid") == os.getpid():
                return float(rec["created"]), "lock record"
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        return time.time(), "lease entry (lock record unavailable)"

    def thermal_wait(self) -> None:
        if THERMAL_FLAG.exists():
            print("[t2bl] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)

    def wait(self) -> None:
        self.thermal_wait()
        self.rt.wait_for_resources()

    def tokenizer(self):
        return self.rt.tokenizer()

    def train(self, rows_path: Path, out_dir: Path, steps: int, seed: int, lr: float) -> None:
        """The vendored trainer, invoked exactly as scripts/run_t2.py._train_body invokes it."""
        for p in (ROOT / "vendor_bwm", ROOT / "vendor_bwm" / "scripts"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import train_generic  # noqa: E402
        old = sys.argv
        sys.argv = ["train_generic.py", "--core", "titan-sparse", "--rows", str(rows_path), "--steps", str(steps),
                    "--seed", str(seed), "--lr", str(lr), "--out", str(out_dir)]
        try:
            train_generic.main()
        finally:
            sys.argv = old
            self.free()

    def free(self) -> None:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_eval(self, t2s0_dir: Path, on_dir: Path | None):
        """Titan-Sparse + T2 s0 ('t2') [+ the seed's adapter ('t2bl')], unmerged."""
        from peft import PeftModel
        import peft
        base, p = self.rt.load_arm("sparse")
        model = PeftModel.from_pretrained(base, str(t2s0_dir), adapter_name=ADAPTER_T2, is_trainable=False)
        if on_dir is not None:
            model.load_adapter(str(on_dir), adapter_name=ADAPTER_ON, is_trainable=False)
        model.eval()
        return model, p["param_fingerprint"], {"peft": peft.__version__}

    def unload(self) -> None:
        self.rt.unload(None)

    def vram_allocated_mib(self) -> float:
        import torch
        return torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0

    def runtime_info(self) -> dict:
        return self.rt.runtime_info()

    def check_fingerprint(self, which: str, fp: str) -> None:
        need(fp.startswith(FP[which]), f"{which} fingerprint {fp[:12]} != {FP[which]}")


def peft_side(model) -> str:
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            return "off" if m.disable_adapters else str(m.active_adapters[0])
    return "plain"


def any_merged(model) -> bool:
    from peft.tuners.lora import LoraLayer
    return any(getattr(m, "merged", False) for m in model.modules() if isinstance(m, LoraLayer))


@contextmanager
def side_ctx(model, side: str):
    """off = disable_adapter(); t2s0 = adapter 't2'; on = adapter 't2bl'; the active side is checked from PEFT's own
    LoRA layer state before anything is generated, and no LoRA layer may be merged (D-28)."""
    if any_merged(model):
        raise RuntimeError("a LoRA layer is merged: adapters must stay unmerged (D-28)")
    if side == "off":
        with model.disable_adapter():
            if peft_side(model) != "off":
                raise RuntimeError("OFF side but PEFT reports an active adapter")
            yield
        return
    name = {"t2s0": ADAPTER_T2, "on": ADAPTER_ON}[side]
    model.set_adapter(name)
    if peft_side(model) != name:
        raise RuntimeError(f"side {side}: PEFT reports {peft_side(model)!r}")
    yield


def gen_texts(model, tok, message_lists: list) -> list:
    from rcwmt import generation
    return generation.generate_texts(model, tok, message_lists, generation.max_tokens(C.LEDGER))


def _ref_rows(path: Path) -> dict:
    ref = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ref[row["cand_id"]] = row
    return ref


def run_anchor(model, tok, chains: list, refs: dict) -> dict:
    """PREREG-2's A0 (as lbeam_run.run_anchor): pilot TEST ledger chain_one OFF and T2s0, each against its reference;
    identical STATES on >= 99% of chains per side."""
    from rcwmt import generation
    t0 = time.time()
    res = {"pid": os.getpid(), "unix": time.time(), "n": len(chains)}
    for side in ("off", "t2s0"):
        ts = time.time()
        with side_ctx(model, side):
            rr = generation.chain(model, tok, chains, "chain_one")
        ref = _ref_rows(refs[side])
        st = sum([None if x is None else env.dumps(x) for x in r["states"]] == (ref.get(c["cand_id"]) or {}).get("states")
                 for c, r in zip(chains, rr))
        tx = sum(r["texts"] == (ref.get(c["cand_id"]) or {}).get("texts") for c, r in zip(chains, rr))
        res[side] = {"identical_states": st / len(chains), "identical_texts": tx / len(chains),
                     "start_unix": ts, "end_unix": time.time(), "generations": sum(len(r["texts"]) for r in rr)}
    res["pass"] = all(res[s]["identical_states"] >= ANCHOR_MIN_IDENTICAL for s in ("off", "t2s0"))
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- input checks
def check_inputs(cfg: Cfg) -> dict:
    shas = {}
    for p, sp in ((cfg.train, cfg.train_source), (cfg.dev, cfg.dev_source)):
        src = C.read_json(sp)
        need(src is not None, f"{sp} missing or unreadable", persist=False)
        need(Path(p).is_file(), f"{p} missing", persist=False)
        shas[Path(p).name] = C.sha256(p)
        need(shas[Path(p).name] == src.get("file_sha256"), f"{Path(p).name} differs from its SOURCE record", persist=False)
    if cfg.t2s0_pins:
        for n, h in cfg.t2s0_pins.items():
            need(C.sha256(Path(cfg.t2s0_dir) / n) == h, f"T2 s0 {n} differs from PREREG-2's pin", persist=False)
    if cfg.anchor_ref_sha:
        for s, h in cfg.anchor_ref_sha.items():
            need(C.sha256(cfg.anchor_refs[s]) == h, f"anchor reference {s} changed (sha256)", persist=False)
    if cfg.check_vendor:
        for rel, h in VENDOR.items():
            need(C.sha256(ROOT / rel) == h, f"{rel} differs from the T2 trainer (sha256)", persist=False)
    return shas


def frozen_files(cfg: Cfg) -> dict:
    files = {rel: ROOT / rel for rel in FROZEN_CODE}
    files["data/" + C.TRAIN.name] = Path(cfg.train)
    files["data/" + C.TRAIN_SOURCE.name] = Path(cfg.train_source)
    files["data/" + C.DEV.name] = Path(cfg.dev)
    files["data/" + C.DEV_SOURCE.name] = Path(cfg.dev_source)
    files["docs/t2b/ledger/GATE_RULE.md"] = C.DOCS / "GATE_RULE.md"
    files["docs/t2b/ledger/queue_hook_t2bl.sh.draft"] = HOOK_DRAFT
    return files


def hook_problem(installed: Path | None = None, draft: Path = HOOK_DRAFT) -> str | None:
    """Every installed queue hook whose name contains 't2bl' (the documented name is scripts/queue_hooks/88_t2bl.sh)
    must be the hook draft byte for byte (the draft is bound by the frozen gate rule). installed: one file (tests) or
    None = scan scripts/queue_hooks/*t2bl*.sh."""
    hooks = [Path(installed)] if installed is not None else sorted(HOOK_INSTALLED.parent.glob("*t2bl*.sh"))
    hooks = [h for h in hooks if h.exists()]
    if not hooks:
        return None
    want = C.sha256(draft) if Path(draft).is_file() else None
    bad = [str(h) for h in hooks if C.sha256(h) != want]
    return f"{', '.join(bad)} is not the hook draft {draft} byte for byte" if bad else None


def frozen_problems(rec, files: dict) -> list:
    if not isinstance(rec, dict) or rec.get("record") != "T2BL_GATE_RULE" or not isinstance(rec.get("inputs"), dict):
        return ["docs/t2b/ledger/GATE_RULE.json is missing or not a gate-rule record (t2bl_gate.py --freeze-rule)"]
    bad = []
    for rel, p in files.items():
        want = rec["inputs"].get(rel)
        if want is None:
            bad.append(f"{rel}: not recorded in the frozen gate rule")
        elif not Path(p).is_file() or C.sha256(Path(p)) != want:
            bad.append(f"{rel}: differs from the frozen gate rule (sha256)")
    return bad


def check_frozen(cfg: Cfg) -> None:
    if cfg.rule_record is None:                    # tiny tests of other mechanics only
        return
    probs = frozen_problems(C.read_json(cfg.rule_record), frozen_files(cfg))
    need(not probs, "the dev gate rule is not frozen or a frozen input changed (run nothing until it verifies): "
         + "; ".join(probs[:6]), persist=False)


def code_hashes() -> dict:
    return {r: C.sha256(ROOT / r) for r in FROZEN_CODE if (ROOT / r).is_file()}


def check_prompts(cfg: Cfg, tok) -> int:
    """Every training prompt == tok.apply_chat_template(messages(row)); every prompt the eval renders is a chat prompt."""
    from rcwmt.capture import render
    rows = C.load_jsonl(cfg.train)
    bad = sum(render(tok, C.messages(r["before"], r["event"], r["surface"])) != r["prompt"] for r in rows)
    need(bad == 0, f"{bad} training prompts differ from the chat template of their messages", persist=False)
    return len(rows)


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = Path(cfg.out)
    if (out / "DONE.json").exists():
        done = C.read_json(out / "DONE.json") or {}
        need(done.get("seeds") == list(cfg.seeds), f"DONE.json records seeds {done.get('seeds')}, this call asks for "
             f"{list(cfg.seeds)}: a completed run is never extended", persist=False)
        print(f"[t2bl] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; investigate, log in DECISIONS.md "
                   "and move it aside before any rerun", persist=False)
    shas = check_inputs(cfg)
    check_frozen(cfg)
    hp = hook_problem() if cfg.rule_record is not None else None
    need(hp is None, f"{hp}; reinstall it from the draft (or remove it) before any run", persist=False)
    dev_rows = load_dev(cfg)
    anchor_chains = cfg.anchor_chains if cfg.anchor_chains is not None else pilot_anchor_chains()
    tok = backend.tokenizer()
    n_checked = check_prompts(cfg, tok)
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[t2bl] {ow}: one-waiter rule (R14). Start only when no other RCWM GPU job runs or waits", persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, shas, dev_rows, anchor_chains, tok, n_checked)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg, backend, out, shas, dev_rows, anchor_chains, tok, n_checked) -> int:
    prov_path = out / "provenance.json"
    prov = C.read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
        need(prov.get("seeds") == list(cfg.seeds) and prov.get("steps") == cfg.steps and prov.get("tag") == cfg.tag,
             "seeds / steps / tag differ from the recorded run", persist=False)
        need(prov.get("data_sha256") == shas, "input data differ from the recorded run")
        changed = sorted(r for r, h in code_hashes().items() if (prov.get("code_sha256") or {}).get(r) != h)
        need(not changed, f"code changed since the run's first turn ({changed}): a resumed run uses the code it started "
             "with; restore it, or move the run aside and log it in DECISIONS.md", persist=False)
    else:
        prov = {"tag": cfg.tag, "seeds": list(cfg.seeds), "steps": cfg.steps, "lr": cfg.lr, "data_sha256": shas,
                "dev_subset": cfg.dev_subset, "n_dev_rows": len(dev_rows), "dev_ids_sha256": C.sha256_text(
                    "\n".join(r["id"] for r in dev_rows)), "lease_label": LEASE_LABEL, "code_sha256": code_hashes(),
                "gate_rule_sha256": C.sha256(cfg.rule_record) if cfg.rule_record and Path(cfg.rule_record).is_file() else None,
                "t2s0": {n: C.sha256(Path(cfg.t2s0_dir) / n) for n in ("adapter_model.safetensors", "adapter_config.json")},
                "stages": {}, "turns": []}

    def save():
        tmp = _partial(prov_path)
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(prov, indent=1) + "\n")
        os.replace(tmp, prov_path)

    states = stage_states(prov, out, cfg)
    for u, st in units(cfg):
        if states[u][st] == "partial":
            orphan(u, st, out)
            (prov["stages"].get(u) or {}).pop(st, None)
            states[u][st] = "todo"
            if st == "T":
                need(states[u]["E"] != "done", f"{u}: evaluation recorded for an adapter being retrained")

    def finalize() -> None:
        st2 = stage_states(prov, out, cfg)
        need(all(v == "done" for u in st2.values() for v in u.values()), "finalize: not every stage recorded and unchanged")
        C.write_create(out / "DONE.json", json.dumps({
            "finished_unix": time.time(), "seeds": list(cfg.seeds),
            "files": {u: {st: prov["stages"][u][st]["files"] for st in prov["stages"][u]} for u in prov["stages"]},
            "provenance_sha256": C.sha256(prov_path)}, indent=1) + "\n", stale_partial="aside")
        print("[t2bl] all seeds complete; DONE.json written", flush=True)

    todo = {u: [st for st in ("E0", "T", "E") if st in states[u] and states[u][st] != "done"] for u in states}
    print(f"[t2bl] stages to run: { {u: v for u, v in todo.items() if v} }", flush=True)
    if not any(todo.values()):
        save()
        finalize()
        return 0
    est = estimates(cfg, n_rows=len(dev_rows), n_anchor=len(anchor_chains) * ANCHOR_STEPS)
    corr = {"T": 1.0, "E": 1.0}
    msgs = dev_messages(dev_rows)
    with backend.lease():
        t_acq, t_src = backend.lease_acquired_unix()
        cap_s = cfg.cap_h * 3600
        turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                "stages_run": [], "deferred": [], "device": backend.device, "prompts_checked": n_checked}
        prov["turns"].append(turn)
        save()
        anchored = False

        def fits(sec: float) -> bool:
            return G2.can_start(time.time() - t_acq, sec, 0.0, cap_s)

        def record(u: str, st: str, extra: dict) -> None:
            prov["stages"].setdefault(u, {})[st] = {"turn": len(prov["turns"]) - 1, **extra, "finished_unix": time.time(),
                                                    "files": {rel_out(p, out): C.sha256(p) for p in stage_required(u, st, out)}}
            turn["stages_run"].append(f"{u}:{st}")
            save()

        def generate_sides(model, sides: tuple) -> tuple:
            tx, meta = {}, {}
            for side in sides:
                ts = time.time()
                with side_ctx(model, side):
                    tx[side] = gen_texts(model, tok, msgs)
                meta[side] = {"start_unix": ts, "end_unix": time.time(), "pid": os.getpid()}
            return tx, meta

        try:
            for s in cfg.seeds:
                u = f"s{s}"
                need_e0 = "E0" in todo["ref"]
                if not todo[u] and not need_e0:
                    continue
                pair = pair_estimate(est, todo[u], need_e0, anchored, corr)
                if not fits(pair):
                    raise Deferred(f"{u} (pair estimate {pair / 3600:.2f} h)")
                if "T" in todo[u]:
                    alloc = backend.vram_allocated_mib()
                    need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated before training {u}", persist=False)
                    backend.wait()
                    t0 = time.time()
                    tdir = out / u / "train"
                    tdir.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[t2bl] {u}: training {cfg.steps} steps, seed {s} on {Path(cfg.train).name}", flush=True)
                    backend.train(Path(cfg.train), tdir, cfg.steps, s, cfg.lr)
                    sec = time.time() - t0
                    need(all(p.exists() for p in stage_required(u, "T", out)), f"{u}: no adapter after training")
                    corr["T"] = max(0.5, sec / est["T"])
                    record(u, "T", {"seconds": round(sec, 1), "planned_s": round(est["T"], 1), "seed": s,
                                    "rows": Path(cfg.train).name})
                    todo[u].remove("T")
                    print(f"[t2bl] {u}: trained in {sec:.0f}s (planned {est['T']:.0f}s)", flush=True)
                if "E" not in todo[u] and not need_e0:
                    continue
                alloc = backend.vram_allocated_mib()
                need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated before the evaluation load", persist=False)
                backend.wait()
                ad_dir = out / u / "train" / "adapter"
                if "E" in todo[u]:
                    ad_sha = {p.name: C.sha256(p) for p in stage_required(u, "T", out)}
                    need(ad_sha == {Path(k).name: v for k, v in prov["stages"][u]["T"]["files"].items()}, f"{u}: adapter changed")
                model, fp, versions = backend.load_eval(Path(cfg.t2s0_dir), ad_dir if "E" in todo[u] else None)
                try:
                    backend.check_fingerprint("sparse", fp)
                    if cfg.t2s0_pins:
                        for n, h in cfg.t2s0_pins.items():
                            need(C.sha256(Path(cfg.t2s0_dir) / n) == h, f"T2 s0 {n} changed at load")
                    turn.setdefault("loads", []).append({"unit": u, "fingerprint": fp, "versions": versions,
                                                         "runtime": backend.runtime_info(), "with_on": "E" in todo[u]})
                    rp = runtime_problem(turn["loads"][-1]["runtime"], versions)
                    if rp:
                        save()
                        raise Stop(f"[t2bl] {rp}: restore the registered environment (nothing evaluated)")
                    save()
                    if not anchored:
                        res = run_anchor(model, tok, anchor_chains, cfg.anchor_refs)
                        turn["anchor"] = res
                        save()
                        if not res["pass"]:
                            raise Stop(f"[t2bl] ANCHOR FAILED (identical states OFF {res['off']['identical_states']:.3f}, "
                                       f"T2s0 {res['t2s0']['identical_states']:.3f} < {ANCHOR_MIN_IDENTICAL}): the instrument "
                                       "does not reproduce PREREG-2's chain_one files")
                        anchored = True
                        corr["E"] = max(0.5, res["seconds"] / max(1.0, est["A0"]))
                    if need_e0:
                        if not fits(corr["E"] * est["E0"]):
                            raise Deferred("ref:E0")
                        t0 = time.time()
                        tx, meta = generate_sides(model, SIDES_E0)
                        for side, p in zip(SIDES_E0, stage_required("ref", "E0", out)):
                            write_jsonl(p, [{"id": r["id"], "text": t} for r, t in zip(dev_rows, tx[side])])
                        sec = time.time() - t0
                        corr["E"] = max(0.5, sec / est["E0"])
                        record("ref", "E0", {"seconds": round(sec, 1), "planned_s": round(est["E0"], 1), "sides": meta,
                                             "generations": est["gens"]["E0"]})
                        todo["ref"].remove("E0")
                    if "E" in todo[u]:
                        if not fits(corr["E"] * est["E"]):
                            raise Deferred(f"{u}:E")
                        t0 = time.time()
                        tx, meta = generate_sides(model, ("on",))
                        write_jsonl(stage_required(u, "E", out)[0], [{"id": r["id"], "text": t} for r, t in zip(dev_rows, tx["on"])])
                        sec = time.time() - t0
                        corr["E"] = max(0.5, sec / est["E"])
                        record(u, "E", {"seconds": round(sec, 1), "planned_s": round(est["E"], 1), "sides": meta,
                                        "generations": est["gens"]["E"], "seed": s})
                        todo[u].remove("E")
                finally:
                    del model
                    backend.unload()
        except Deferred as d:
            rest = [f"{u}:{st}" for u, v in todo.items() for st in v]
            turn["deferred"] = rest
            print(f"[t2bl] cap: deferring {rest} to the next turn ({d})", flush=True)
        turn["released_unix"] = time.time()
        turn["no_progress"] = not turn["stages_run"]
        prov["no_progress_turns"] = (prov.get("no_progress_turns") or 0) + 1 if turn["no_progress"] else 0
        save()
    st_end = stage_states(prov, out, cfg)
    if all(v == "done" for u in st_end.values() for v in u.values()):
        finalize()
        return 0
    if prov.get("no_progress_turns", 0) >= NO_PROGRESS_STOP:
        raise Stop(f"[t2bl] no stage was recorded in {prov['no_progress_turns']} consecutive turns (deferred "
                   f"{turn['deferred']}): stop (a cap or rate amendment is needed; investigate, log)")
    print("[t2bl] turn ended; the remaining stages resume next turn", flush=True)
    return 5


def cap_from_arg(x) -> float:
    """--cap-h may lower the turn cap (tests, a short turn), never raise it above the agreed 4.5 h."""
    return min(float(x), CAP_H)


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        with (Path(out) / "STOPPED.json").open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)},
                                indent=1) + "\n")
    except FileExistsError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="CPU only: no torch import, no model; projected GPU time per turn")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan:
        res = plan()
        need("torch" not in sys.modules, "--plan must not import torch", persist=False)
        res["torch_imported"] = "torch" in sys.modules
        res["train_source_sha256"] = C.sha256(C.TRAIN_SOURCE) if C.TRAIN_SOURCE.exists() else None
        res["code_sha256"] = {"scripts/t2bl_run.py": C.sha256(Path(__file__))}
        if C.run_outputs():
            print("[t2bl] a runs/t2bl output exists: GEN_PLAN.json is not rewritten (printed only)")
            print(json.dumps(res, indent=1))
            return 0
        print("wrote", C.write_doc("GEN_PLAN.json", json.dumps(res, indent=1) + "\n"))
        return 0
    cfg = Cfg(cap_h=cap_from_arg(a.cap_h))
    try:
        return run(cfg, RealBackend())
    except Stop as e:
        print(str(e), flush=True)
        persist_stop(Path(cfg.out), e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
