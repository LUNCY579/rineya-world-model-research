"""Retention suite GPU runner (roadmap item 4 / joint 5c). docs/retention/DESIGN.md section 6.

  python -B scripts/ret_run.py --plan [--mode dev|confirm] [--draw D] [--extra-adapter NAME=DIR]... [--record]
                                  # CPU only: no torch import, no model, no lease file read. Generations, GPU-h per
                                  # turn, disk; --record writes docs/retention/RET_PLAN.json (refused once frozen)
  python -B scripts/ret_run.py --mode dev --draw docs/retention/draws/dev_20261121.jsonl
                                  # Supa 1.7 only, on the dev draw (no log-probs). Feeds ret_analysis.py dev-select.
  python -B scripts/ret_run.py --mode confirm --draw docs/retention/draws/confirm_<seed>.jsonl
                                  [--extra-adapter NAME=DIR]...        # extra PEFT adapters on Titan-Sparse
                                  [--extra-adapter-supa NAME=DIR]...   # extra PEFT adapters on Supa 1.7

Arms (confirm). One model at a time, under ONE lease 'rcwm-t1:retention' (rcwmt.runtime.gpu_lease: alternation,
thermal flag, duty cycle, RAM floor, the dedicated-memory holder check; 4.5 h turn cap from acquisition, D-22 stage
rule):
  SPARSE  Titan-Sparse (FP32, fingerprint 818be9c6b6a3) + the T2 s0 LoRA UNMERGED (PEFT adapter 't2s0', the PREREG-2
          artifact, pinned sha256) + any extra adapters (PEFT 'x_<name>'), all in the SAME process: side 'off' =
          disable_adapter() = Titan-Sparse, side 't2s0' = set_adapter('t2s0'), side 'x_<name>'. Never merged (D-28).
  SUPA    unload Sparse (VRAM checked empty), load Supa 1.7 (fingerprint a9952c55f5a1): side 'supa' (+ 'xs_<name>').
Every side: greedy (do_sample False, temperature/top_p/top_k None), <= 48 new tokens, EOS [151645, 151643], thinking
disabled (chat template enable_thinking=False), left-padded batches (<= 32 rows, <= 8,000 tokens per batch), FP32,
TF32 off (runtime.load_arm). Per prompt: the text, EOS termination, token count, the first generated token and, in
confirm mode, the full first-token log-probability vector from a prefill forward of the same batch (fp16, clamped at
-60, as rcwmt.offtask / R14 store them) -> lp_<side>.npy, for the report-only KL-vs-correctness calibration.

Integrity before the lease: the kit freeze (docs/retention/RET_FREEZE.json) verifies; the draw hashes to its seal, was
minted under this freeze, passed its freshness scan, has the purpose of the mode, and (confirm) was minted under the
current dev selection of its surface-diversity setting; the tokenizer files the runtime loads (tokenizer.json,
tokenizer_config.json with the chat template, vocab.json, merges.txt) equal the pins in ret_common.TOKENIZER_PINS (bound
by the freeze); T2 s0's adapter files equal the PREREG-2 pins; extra adapters are hashed (again at load, must be
unchanged); no other RCWM job holds or waits for the GPU lease (one-waiter rule, R14); one runner per output directory
(run lock). After each load: the parameter fingerprint, torch / transformers / peft == the registered runtime, and the
sha256 of the loaded tokenizer's chat template == ret_common.CHAT_TEMPLATE_SHA, else stop before anything is
generated. The tokenizer hashes and the chat-template hash are recorded in provenance.json.

Outputs (create-only) under runs/retention/<draw_id>/: provenance.json, sparse/gen_<side>.jsonl (+ lp_<side>.npy),
supa/gen_<side>.jsonl (+ lp_<side>.npy), DONE.json. A stage (SPARSE, SUPA) is written only when ALL its sides are
generated; it is DONE only if its files exist and provenance.json records their sha256. Anything else a stage wrote
is PARTIAL: moved to _orphans/ (never deleted) and regenerated in full on the next run (resumable). A recorded output
whose hash changed stops the job.

Exit codes: 0 complete (or already complete); 3 deterministic stop, never retried (persisted as STOPPED.json when it
concerns the run; "another RCWM job holds/waits for the lease", "run lock held" and pre-run integrity refusals are not
persisted); 5 resumable (the turn cap left a stage for the next turn); 1 crash (the lease is released).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True


def _cpu_only_mode(argv) -> bool:
    return "--plan" in argv


if _cpu_only_mode(sys.argv):                         # never touch the driver: empty string, NEVER "-1" (I-21)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_gen as G2  # noqa: E402  (torch-free at import: pins, registered runtime, D-22 can_start)
import ret_common as RC  # noqa: E402
import ret_gen as RGEN  # noqa: E402

# ---------------------------------------------------------------- registered constants
LEASE_LABEL = "rcwm-t1:retention"
RCWM_LABEL_PREFIX = "rcwm-t1:"
T2S0_DIR_REL = "runs/t2/t2_s0/adapter"
S0_PINS = {"adapter_model.safetensors": G2.ADAPTER_SHA, "adapter_config.json": G2.ADAPTER_CONFIG_SHA}
FP = dict(G2.FP)                                     # sparse 818be9c6b6a3, supa a9952c55f5a1
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)     # torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0
CORE_DIRNAME = {"sparse": "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze", "supa": "Rineya-Supa-1.7.0-devfreeze"}
EOS = (151645, 151643)
PAD = 151643
MAX_BATCH, TOKEN_BUDGET = 32, 8000                   # rcwmt.generation.generate_texts' batching
LP_CLAMP = -60.0
VOCAB = 151936
RATE_PLAN = 1.0           # gens/s, conservative: 48-token free-text answers from a chat model, duty cycle on
RATE_OPT = 2.7            # gens/s, PREREG-2's registered planning rate (short answers; observed 3.8)
PREFILL_FRAC = 0.10       # the first-token prefill forward, as a fraction of a side's generation time (confirm only)
LOAD_S = 120.0
ADAPTER_LOAD_S = 10.0
CAP_H = 4.5
VRAM_EMPTY_MIB = 512.0
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
STAGES = {"dev": ("SUPA",), "confirm": ("SPARSE", "SUPA")}
STAGE_DIR = {"SPARSE": "sparse", "SUPA": "supa"}
NAME_RE = re.compile(r"^[a-z0-9_]{1,24}$")
PROV_CODE = ("scripts/ret_run.py", "scripts/ret_common.py", "scripts/ret_gen.py", "scripts/ret_grade.py",
             "rcwmt/runtime.py", "rcwmt/capture.py", "rcwmt/lease_vendored.py")

Stop = RC.RetStop
need = RC.need


def _read_json(p: Path):
    return RC.read_json(p)


# ---------------------------------------------------------------- configuration
@dataclass
class Cfg:
    draw: Path
    mode: str = "confirm"
    root: Path = ROOT
    out: Path | None = None
    extra: dict = field(default_factory=dict)          # name -> adapter dir (Titan-Sparse base)
    extra_supa: dict = field(default_factory=dict)     # name -> adapter dir (Supa base)
    t2s0_dir: Path = ROOT / T2S0_DIR_REL
    s0_pins: dict = field(default_factory=lambda: dict(S0_PINS))   # the tiny tests pin their stand-in adapter
    require_freeze: bool = True                        # False only from the tiny tests (never from the CLI)
    allow_unscanned: bool = False                      # tests only
    allow_base_mismatch: bool = False
    cap_h: float = CAP_H
    rate: float = RATE_PLAN
    max_new: int = RC.MAX_NEW_TOKENS

    def out_dir(self, did: str) -> Path:
        return Path(self.out) if self.out else RC.RUNS / did


def sparse_adapters(cfg: Cfg) -> dict:
    """PEFT adapter name -> dir, in load order (t2s0 first)."""
    if cfg.mode != "confirm":
        return {}
    return {"t2s0": Path(cfg.t2s0_dir), **{f"x_{n}": Path(d) for n, d in cfg.extra.items()}}


def supa_adapters(cfg: Cfg) -> dict:
    return {f"xs_{n}": Path(d) for n, d in cfg.extra_supa.items()} if cfg.mode == "confirm" else {}


def stage_sides(cfg: Cfg) -> dict:
    if cfg.mode == "dev":
        return {"SUPA": ["supa"]}
    return {"SPARSE": ["off", *sparse_adapters(cfg)], "SUPA": ["supa", *supa_adapters(cfg)]}


def side_model(side: str) -> str:
    return "supa" if side == "supa" or side.startswith("xs_") else "sparse"


def wants_lp(cfg: Cfg) -> bool:
    return cfg.mode == "confirm"


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(cfg: Cfg, stage: str, out: Path) -> list:
    d = Path(out) / STAGE_DIR[stage]
    sides = stage_sides(cfg)[stage]
    fs = [d / f"gen_{s}.jsonl" for s in sides]
    if wants_lp(cfg):
        fs += [d / f"lp_{s}.npy" for s in sides]
    return fs


def stage_owned(cfg: Cfg, stage: str, out: Path) -> list:
    fs = stage_required(cfg, stage, out)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path) -> str:
    return str(Path(p).relative_to(Path(out))).replace("\\", "/")


def stage_states(cfg: Cfg, prov: dict, out: Path) -> dict:
    res = {}
    for st in STAGES[cfg.mode]:
        req = stage_required(cfg, st, out)
        rec = (prov.get("stages") or {}).get(st)
        if all(p.exists() for p in req) and rec:
            files = rec.get("files") or {}
            for p in req:
                r = rel_out(p, out)
                need(r in files, f"stage {st}: {r} not recorded in provenance", persist=True)
                need(RC.sha256_file(p) == files[r], f"stage {st}: {r} changed after it was recorded (sha256)", persist=True)
            res[st] = "done"
        elif any(p.exists() for p in stage_owned(cfg, st, out)):
            res[st] = "partial"
        else:
            res[st] = "todo"
    return res


def orphan(cfg: Cfg, stage: str, out: Path) -> Path:
    dst = Path(out) / "_orphans" / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    dst.mkdir(parents=True, exist_ok=False)
    for p in stage_owned(cfg, stage, out):
        if p.exists():
            shutil.move(str(p), str(dst / p.name))
    print(f"[retention] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), f"{path} exists (create-only)", persist=True)
    tmp = _partial(path)
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- counts, estimates, plan
def counts(n_prompts: int, cfg: Cfg) -> dict:
    sides = stage_sides(cfg)
    return {st: n_prompts * len(sides[st]) for st in STAGES[cfg.mode]}


def stage_est_s(cfg: Cfg, st: str, n: dict, rate: float) -> float:
    gen = n[st] / rate * (1 + (PREFILL_FRAC if wants_lp(cfg) else 0.0))
    ads = len(sparse_adapters(cfg)) if st == "SPARSE" else len(supa_adapters(cfg))
    return LOAD_S + ADAPTER_LOAD_S * ads + gen


def supa_reserve_s(cfg: Cfg, n: dict, rate: float) -> float:
    return G2.STAGE_SAFETY * stage_est_s(cfg, "SUPA", n, rate)


def simulate_turns(cfg: Cfg, n: dict, rate: float, cap_h: float = CAP_H) -> list:
    cap, left, turns = cap_h * 3600, list(STAGES[cfg.mode]), []
    while left:
        t, ran = 0.0, []
        for st in list(left):
            res = supa_reserve_s(cfg, n, rate) if (st != "SUPA" and "SUPA" in left) else 0.0
            if not G2.can_start(t, stage_est_s(cfg, st, n, rate), res, cap):
                break
            t += stage_est_s(cfg, st, n, rate)
            ran.append(st)
            left.remove(st)
        if not ran:
            raise RuntimeError(f"no stage fits an empty turn at {rate} gens/s")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 3)})
    return turns


def plan_for(cfg: Cfg, n_prompts: int) -> dict:
    n = counts(n_prompts, cfg)
    sides = stage_sides(cfg)
    res = {"mode": cfg.mode, "n_prompts": n_prompts, "sides": sides, "max_new_tokens": cfg.max_new,
           "rates_gens_per_s": {"planning (conservative)": RATE_PLAN, "optimistic (PREREG-2 registered)": RATE_OPT},
           "load_s_per_model": LOAD_S, "prefill_frac": PREFILL_FRAC if wants_lp(cfg) else 0.0,
           "stages": [{"stage": st, "sides": sides[st], "generations": n[st],
                       "projected_h_planning": round(stage_est_s(cfg, st, n, RATE_PLAN) / 3600, 3),
                       "projected_h_optimistic": round(stage_est_s(cfg, st, n, RATE_OPT) / 3600, 3)} for st in STAGES[cfg.mode]],
           "total_generations": sum(n.values()),
           "turns_planning": simulate_turns(cfg, n, RATE_PLAN), "turns_optimistic": simulate_turns(cfg, n, RATE_OPT),
           "per_arm_h_planning": round(n_prompts / RATE_PLAN * (1 + (PREFILL_FRAC if wants_lp(cfg) else 0)) / 3600, 3),
           "per_arm_h_optimistic": round(n_prompts / RATE_OPT * (1 + (PREFILL_FRAC if wants_lp(cfg) else 0)) / 3600, 3),
           "lp_disk_gb": round(n_prompts * VOCAB * 2 * sum(len(v) for v in sides.values()) / 1e9, 3) if wants_lp(cfg) else 0.0,
           "stage_rule": f"D-22: no stage starts within {G2.STAGE_MARGIN_H} h of the {CAP_H} h cap; a stage starts only if "
                         f"{G2.STAGE_SAFETY} x its estimate (+ the Supa reserve) fits"}
    res["gpu_h_planning"] = round(sum(t["projected_h"] for t in res["turns_planning"]), 3)
    res["gpu_h_optimistic"] = round(sum(t["projected_h"] for t in res["turns_optimistic"]), 3)
    return res


def design_prompt_counts() -> dict:
    """Prompt counts implied by the design (no draw needed): dev = 8 categories x 3 levels x 75 (+75 twins per
    unknown_twin level); confirm with every category selected = 8 x 75 (+75 twins)."""
    per_level = len(RC.CATEGORIES) * RC.N_PER_CELL + RC.N_PER_CELL
    return {"dev": per_level * len(RC.LEVELS), "confirm_all_categories": per_level}


# ---------------------------------------------------------------- integrity before the lease (no torch)
def check_adapter_dir(d: Path) -> dict:
    need(Path(d).is_dir(), f"adapter directory {d} does not exist")
    shas = {}
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        need((Path(d) / f).is_file(), f"{d}/{f} missing")
        shas[f] = RC.sha256_file(Path(d) / f)
    return shas


def adapter_base(d: Path) -> str:
    cfg = _read_json(Path(d) / "adapter_config.json") or {}
    return str(cfg.get("base_model_name_or_path") or "")


def preflight(cfg: Cfg, tokenizer_dir: Path | None = None) -> dict:
    """Everything checked before the lease. Returns {rows, seal, adapters, freeze_sha256, tokenizer}."""
    root = Path(cfg.root)
    need(cfg.mode in STAGES, f"unknown mode {cfg.mode}")
    fz_sha = None
    if cfg.require_freeze:
        ok, errs, fz = RC.verify_freeze(root)
        need(ok, "the retention kit freeze does not verify: " + "; ".join(errs[:4]))
        need(fz.get("tokenizer_pins") == RC.TOKENIZER_PINS and fz.get("chat_template_sha256") == RC.CHAT_TEMPLATE_SHA,
             "the freeze record does not carry the tokenizer / chat-template pins of the code")
        fz_sha = RC.sha256_file(root / "docs" / "retention" / "RET_FREEZE.json")
    tdir = Path(tokenizer_dir or RC.TOKENIZER_DIR)
    tp = RC.tokenizer_problem(tdir)
    need(tp is None, f"{tp} in {tdir}: the runtime would load another tokenizer or chat template")
    rows, seal = RGEN.load_draw(cfg.draw)
    need(seal.get("purpose") == cfg.mode, f"draw purpose {seal.get('purpose')} != mode {cfg.mode}")
    if cfg.require_freeze:
        need(seal.get("freeze_sha256") == fz_sha, "the draw was not minted under the current kit freeze")
    fr = seal.get("freshness") or {}
    need(cfg.allow_unscanned or fr.get("ok") is True, "the draw's freshness scan was skipped or failed")
    if cfg.mode == "confirm" and cfg.require_freeze:
        sel = root / RC.selection_rel(bool(seal.get("surface_diversity")))
        need(sel.is_file() and RC.sha256_file(sel) == seal.get("selection_sha256"),
             f"the draw was not minted under the current {sel.name}")
    ads = {}
    for name in list(cfg.extra) + list(cfg.extra_supa):
        need(bool(NAME_RE.match(name)), f"adapter name {name!r} must match {NAME_RE.pattern}")
    need(not (set(cfg.extra) & set(cfg.extra_supa)), "an extra adapter name is used twice")
    for name, d in sparse_adapters(cfg).items():
        ads[name] = {"dir": str(d), "base": "sparse", "sha256": check_adapter_dir(d), "base_model_name_or_path": adapter_base(d)}
    for name, d in supa_adapters(cfg).items():
        ads[name] = {"dir": str(d), "base": "supa", "sha256": check_adapter_dir(d), "base_model_name_or_path": adapter_base(d)}
    if "t2s0" in ads:
        need(ads["t2s0"]["sha256"] == cfg.s0_pins, "T2 s0 adapter files differ from the PREREG-2 pins")
        need(cfg.s0_pins == S0_PINS or not cfg.require_freeze, "a registered run must use the PREREG-2 pins")
    for name, a in ads.items():
        if Path(a["base_model_name_or_path"]).name != CORE_DIRNAME[a["base"]]:
            need(cfg.allow_base_mismatch, f"adapter {name} was trained on {a['base_model_name_or_path']!r}, not "
                                          f"{CORE_DIRNAME[a['base']]} (pass --allow-base-mismatch to record and proceed)")
            a["base_mismatch_allowed"] = True
    return {"rows": rows, "seal": seal, "adapters": ads, "freeze_sha256": fz_sha,
            "tokenizer": {"dir": str(tdir), "files": RC.tokenizer_hashes(tdir)}}


# ---------------------------------------------------------------- run lock and one-waiter rule
def acquire_run_lock(out: Path) -> Path:
    import psutil
    out.mkdir(parents=True, exist_ok=True)
    lk = out / ".run.lock"
    if lk.exists():
        rec = _read_json(lk) or {}
        alive = False
        try:
            alive = psutil.pid_exists(int(rec.get("pid", -1))) and \
                abs(psutil.Process(int(rec["pid"])).create_time() - float(rec.get("create_time", 0))) < 1.0
        except (psutil.Error, ValueError, TypeError, KeyError):
            alive = False
        need(not alive, f"another ret_run.py holds the run lock {rec}")
        print(f"[retention] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink()
    with lk.open("x", encoding="utf-8") as fh:
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
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for or holds the GPU lease (live, not a reused pid).
    The peer's (BWM) jobs are allowed: this job then waits in the lease as usual."""
    w = _read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = _read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and _live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


def runtime_problem(rt: dict, versions: dict | None = None) -> str | None:
    bad = {k: rt.get(k) for k in ("torch", "transformers") if rt.get(k) != REGISTERED_RUNTIME[k]}
    if versions is not None and versions.get("peft") is not None and versions.get("peft") != REGISTERED_RUNTIME["peft"]:
        bad["peft"] = versions.get("peft")
    return f"runtime {bad} differs from the registered {REGISTERED_RUNTIME}" if bad else None


# ---------------------------------------------------------------- model-facing pieces (torch imported lazily)
class RealBackend:
    """The registered GPU path: rcwmt.runtime (lease, numerics, loads, pacing) and PEFT as PREREG-2 / PREREG-3 load it."""
    device = "cuda"

    def __init__(self, rt=None):
        if rt is None:
            import torch  # noqa: F401
            from rcwmt import runtime as rt
        self.rt = rt

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 retention suite (Supa / Titan-Sparse / T2 s0 OFF-ON)")

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

    def wait(self) -> None:
        if THERMAL_FLAG.exists():
            print("[retention] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)
        self.rt.wait_for_resources()

    def _wrap(self, base, adapters: dict):
        from peft import PeftModel
        names = list(adapters)
        model = PeftModel.from_pretrained(base, str(adapters[names[0]]), adapter_name=names[0], is_trainable=False)
        for n in names[1:]:
            model.load_adapter(str(adapters[n]), adapter_name=n, is_trainable=False)
        return model.eval()

    def load(self, which: str, adapters: dict):
        import peft
        base, p = self.rt.load_arm(which)
        model = self._wrap(base, adapters) if adapters else base
        return model, p["param_fingerprint"], {"peft": peft.__version__ if adapters else None}

    def tokenizer(self):
        return self.rt.tokenizer()

    def tokenizer_dir(self) -> Path:
        return Path(self.rt.TOKENIZER_DIR)

    def unload(self) -> None:
        self.rt.unload(None)

    def vram_allocated_mib(self) -> float:
        import torch
        return torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0

    def runtime_info(self) -> dict:
        return self.rt.runtime_info()

    def pace(self) -> None:
        self.rt.breathe()

    def processors(self):
        return self.rt.throttle_list()

    def check_fingerprint(self, which: str, fp: str) -> None:
        need(fp.startswith(FP[which]), f"{which} fingerprint {fp[:12]} != {FP[which]}", persist=True)


@contextmanager
def side_ctx(model, side: str):
    """'off' / 'supa' = every adapter disabled (the bare core); any other side = only that adapter active."""
    is_peft = hasattr(model, "disable_adapter") and hasattr(model, "set_adapter")
    if side in ("off", "supa"):
        with (model.disable_adapter() if is_peft else nullcontext()):
            yield
        return
    need(is_peft, f"side {side} needs a PEFT model", persist=True)
    model.set_adapter(side)
    act = model.active_adapter
    need(act == side or act == [side], f"active adapter {act} != {side}", persist=True)
    yield


def generate_side(model, tok, rows: list, max_new: int, backend, lp_out=None) -> list:
    """Greedy answers for every prompt (draw order), rcwmt.generation.generate_texts' batching and decoding, plus
    (lp_out given) the first-token log-probabilities of a prefill forward of the same left-padded batch."""
    import numpy as np
    import torch
    from rcwmt.capture import render
    dev = backend.device
    prompts = [render(tok, r["messages"]) for r in rows]
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc[i]))
    batches, cur = [], []
    for i in order:
        width = len(enc[i]) + max_new
        if cur and (len(cur) + 1 > MAX_BATCH or (len(cur) + 1) * width > TOKEN_BUDGET):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    out = [None] * len(prompts)
    with torch.no_grad():
        for idx in batches:
            L = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), L), PAD, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x = enc[i]
                ids[r, L - len(x):] = torch.tensor(x)
                att[r, L - len(x):] = 1
            ids, att = ids.to(dev), att.to(dev)
            first_pf = [None] * len(idx)
            if lp_out is not None:
                pos = (att.cumsum(-1) - 1).clamp(min=0)
                o = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=1)
                lp = torch.log_softmax(o.logits[:, -1].float(), -1)
                first_pf = lp.argmax(-1).tolist()
                arr = lp.clamp(min=LP_CLAMP).cpu().numpy().astype(np.float16)
                for r, i in enumerate(idx):
                    lp_out[i] = arr[r]
                del o, lp, arr
            g = model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=max_new,
                               eos_token_id=list(EOS), pad_token_id=PAD, temperature=None, top_p=None, top_k=None,
                               logits_processor=backend.processors())
            for r, i in enumerate(idx):
                new = g[r, L:].tolist()
                cut = next((j for j, t in enumerate(new) if t in EOS), len(new))
                toks = new[:cut]
                out[i] = {"pid": rows[i]["pid"], "text": tok.decode(toks, skip_special_tokens=True).strip(),
                          "terminated": cut < len(new), "n_new": len(toks),
                          "first_token": new[0] if new else None, "first_token_prefill": first_pf[r], "lp_row": i}
            del g
            backend.pace()
    return out


# ---------------------------------------------------------------- the job
def chat_template_sha(tok) -> str | None:
    ct = getattr(tok, "chat_template", None)
    return RC.sha256_bytes(ct.encode("utf-8")) if isinstance(ct, str) else None


def run(cfg: Cfg, backend) -> int:
    pf = preflight(cfg, backend.tokenizer_dir())
    rows, seal = pf["rows"], pf["seal"]
    out = cfg.out_dir(seal["draw_id"])
    if (out / "DONE.json").exists():
        print(f"[retention] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    need(not (out / "STOPPED.json").exists(), f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; an "
         "operator must investigate, log it in DECISIONS.md and move the file aside before any rerun")
    ow = one_waiter_problem(backend)
    need(ow is None, f"{ow}: one-waiter rule (R14). Start the retention job only when no other RCWM GPU job is running "
         "or waiting")
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, pf)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg: Cfg, backend, out: Path, pf: dict) -> int:
    import numpy as np
    rows, seal = pf["rows"], pf["seal"]
    prov_path = out / "provenance.json"
    prov = _read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable", persist=True)
        need(prov.get("draw_sha256") == seal["file_sha256"] and prov.get("mode") == cfg.mode,
             "provenance.json belongs to another draw or mode", persist=True)
        need(prov.get("sides") == stage_sides(cfg), f"the sides differ from the first invocation's {prov.get('sides')} "
             "(same adapters required to resume)", persist=True)
        need({k: v["sha256"] for k, v in prov.get("adapters", {}).items()} ==
             {k: v["sha256"] for k, v in pf["adapters"].items()}, "adapter files differ from the first invocation's",
             persist=True)
        need((prov.get("tokenizer") or {}).get("files") == pf["tokenizer"]["files"],
             "tokenizer files differ from the first invocation's", persist=True)
    else:
        prov = {"draw_id": seal["draw_id"], "draw_sha256": seal["file_sha256"], "seal_sha256": RC.sha256_file(
            Path(cfg.draw).with_name(seal["draw_id"] + ".SEAL.json")), "mode": cfg.mode, "freeze_sha256": pf["freeze_sha256"],
            "registered": bool(cfg.require_freeze), "sides": stage_sides(cfg), "adapters": pf["adapters"],
            "tokenizer": {**pf["tokenizer"], "chat_template_sha256_pin": RC.CHAT_TEMPLATE_SHA},
            "max_new_tokens": cfg.max_new, "lp": wants_lp(cfg), "stages": {}, "turns": []}
    prov["code_sha256"] = {r: RC.sha256_file(ROOT / r) for r in PROV_CODE if (ROOT / r).is_file()}

    def save():
        tmp = _partial(prov_path)
        tmp.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, prov_path)

    states = stage_states(cfg, prov, out)
    for st, s in states.items():
        if s == "partial":
            orphan(cfg, st, out)
            prov["stages"].pop(st, None)
            states[st] = "todo"
    done = {st for st, s in states.items() if s == "done"}
    order = STAGES[cfg.mode]

    def record(st: str, extra: dict) -> None:
        prov["stages"][st] = {"name": st, "turn": len(prov["turns"]) - 1, **extra, "finished_unix": time.time(),
                              "files": {rel_out(p, out): RC.sha256_file(p) for p in stage_required(cfg, st, out)}}
        done.add(st)
        save()

    def finalize() -> None:
        need(set(prov["stages"]) == set(order) and stage_states(cfg, prov, out) == {s: "done" for s in order},
             "finalize: not every stage is recorded and unchanged", persist=True)
        with (out / "DONE.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps({"finished_unix": time.time(), "draw_id": seal["draw_id"], "mode": cfg.mode,
                                "files": {st: prov["stages"][st]["files"] for st in order},
                                "provenance_sha256": RC.sha256_file(prov_path)}, indent=1) + "\n")
        print("[retention] all stages complete; DONE.json written", flush=True)

    todo = [st for st in order if st not in done]
    print(f"[retention] {seal['draw_id']} ({cfg.mode}): stages to run {todo}", flush=True)
    if not todo:
        save()
        finalize()
        return 0
    n = counts(len(rows), cfg)
    sides = stage_sides(cfg)
    with backend.lease():
        t_acq, t_src = backend.lease_acquired_unix()
        cap_s = cfg.cap_h * 3600
        turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                "stages_run": [], "deferred": [], "device": backend.device}
        prov["turns"].append(turn)
        save()
        rate = [cfg.rate]
        for st in todo:
            reserve = supa_reserve_s(cfg, n, rate[0]) if (st != "SUPA" and "SUPA" not in done) else 0.0
            if not G2.can_start(time.time() - t_acq, stage_est_s(cfg, st, n, rate[0]), reserve, cap_s):
                turn["deferred"] = [s for s in todo if s not in done]
                print(f"[retention] cap: deferring {turn['deferred']} to the next turn", flush=True)
                save()
                break
            if st == "SUPA" and "SPARSE" in order:
                alloc = backend.vram_allocated_mib()
                need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated after unloading Titan-Sparse; not "
                     "loading Supa")
            which = "sparse" if st == "SPARSE" else "supa"
            ads = sparse_adapters(cfg) if st == "SPARSE" else supa_adapters(cfg)
            backend.wait()
            t0 = time.time()
            for name, d in ads.items():                 # hashed again at load: unchanged since the pre-lease check
                need(check_adapter_dir(d) == pf["adapters"][name]["sha256"], f"adapter {name} changed before load",
                     persist=True)
            model, fp, versions = backend.load(which, ads)
            try:
                backend.check_fingerprint(which, fp)
                prov[f"{which}_fingerprint"] = fp
                rt = backend.runtime_info()
                turn[f"{which}_runtime"], turn[f"{which}_versions"] = rt, versions
                rp = runtime_problem(rt, versions)
                if rp:
                    save()
                    raise Stop(f"[retention] {rp}: restore the registered environment (nothing generated in {st})",
                               persist=True)
                turn[f"{which}_load_s"] = round(time.time() - t0, 1)
                tok = backend.tokenizer()
                cts = chat_template_sha(tok)
                turn[f"{which}_chat_template_sha256"] = cts
                save()
                need(cts == RC.CHAT_TEMPLATE_SHA, f"the loaded chat template hashes to {cts}, not the pinned "
                     f"{RC.CHAT_TEMPLATE_SHA} (nothing generated in {st})", persist=True)
                d = out / STAGE_DIR[st]
                d.mkdir(parents=True, exist_ok=True)
                res, times = {}, {}
                for side in sides[st]:
                    ts = time.time()
                    lp = None
                    if wants_lp(cfg):
                        lp_p = _partial(d / f"lp_{side}.npy")
                        need(not lp_p.exists(), f"{lp_p} exists", persist=True)
                        lp = np.lib.format.open_memmap(str(lp_p), mode="w+", dtype=np.float16,
                                                       shape=(len(rows), VOCAB))
                    with side_ctx(model, side):
                        res[side] = generate_side(model, tok, rows, cfg.max_new, backend, lp)
                    if lp is not None:
                        lp.flush()
                        del lp
                    times[side] = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                    print(f"[retention] {st}/{side}: {len(rows)} prompts in {time.time() - ts:.0f}s", flush=True)
                for side in sides[st]:
                    write_jsonl(d / f"gen_{side}.jsonl", res[side])
                    if wants_lp(cfg):
                        os.replace(_partial(d / f"lp_{side}.npy"), d / f"lp_{side}.npy")
            finally:
                del model
                backend.unload()
            sec = time.time() - t0
            rate[0] = max(0.2, n[st] / max(sec, 1e-9))
            record(st, {"generations": n[st], "seconds": round(sec, 1), "pid": os.getpid(), "sides": times,
                        "first_token_agreement": {s: _first_agree(res[s]) for s in sides[st]} if wants_lp(cfg) else None})
            turn["stages_run"].append(st)
            save()
        turn["released_unix"] = time.time()
        save()
    if done >= set(order):
        finalize()
        return 0
    print(f"[retention] turn ended; remaining stages {[s for s in order if s not in done]} resume next turn", flush=True)
    return 5


def _first_agree(recs: list) -> float | None:
    pairs = [(r["first_token"], r["first_token_prefill"]) for r in recs if r["first_token_prefill"] is not None]
    return round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None


def persist_stop(out: Path, e: Stop) -> None:
    if not getattr(e, "persist", False):
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "STOPPED.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"unix": time.time(), "pid": os.getpid(), "reason": str(e)}, indent=1) + "\n")
    except FileExistsError:
        pass


def _kv_args(vals) -> dict:
    out = {}
    for v in vals or []:
        need("=" in v, f"--extra-adapter expects NAME=DIR, got {v!r}")
        k, d = v.split("=", 1)
        need(k not in out, f"adapter name {k} given twice")
        out[k] = Path(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="CPU only: counts, GPU-h per turn, disk; no torch, no lease")
    ap.add_argument("--record", action="store_true", help="with --plan: write docs/retention/RET_PLAN.json")
    ap.add_argument("--mode", choices=("dev", "confirm"), default=None)
    ap.add_argument("--draw", default=None)
    ap.add_argument("--extra-adapter", action="append", default=[])
    ap.add_argument("--extra-adapter-supa", action="append", default=[])
    ap.add_argument("--allow-base-mismatch", action="store_true")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    out_for_stop, seal = None, {}
    try:
        extra, extra_supa = _kv_args(a.extra_adapter), _kv_args(a.extra_adapter_supa)
        if a.plan:
            return cmd_plan(a, extra, extra_supa)
        need(a.mode is not None and a.draw is not None, "--mode and --draw are required for a run")
        cfg = Cfg(draw=Path(a.draw), mode=a.mode, extra=extra, extra_supa=extra_supa,
                  allow_base_mismatch=a.allow_base_mismatch, cap_h=a.cap_h)
        seal = _read_json(Path(a.draw).with_name(Path(a.draw).name[: -len(".jsonl")] + ".SEAL.json")) or {}
        out_for_stop = cfg.out_dir(seal.get("draw_id", "unknown_draw"))
        return run(cfg, RealBackend())
    except Stop as e:
        print(str(e), flush=True)
        if out_for_stop is not None and seal.get("draw_id"):
            persist_stop(out_for_stop, e)
        return 3


def cmd_plan(a, extra, extra_supa) -> int:
    res = {"design_prompt_counts": design_prompt_counts(), "plans": {}}
    modes = [a.mode] if a.mode else ["dev", "confirm"]
    for mode in modes:
        if a.draw and a.mode:
            rows = RC.read_jsonl(Path(a.draw))
            n_prompts = len(rows)
        else:
            n_prompts = res["design_prompt_counts"]["dev" if mode == "dev" else "confirm_all_categories"]
        cfg = Cfg(draw=Path(a.draw or "none"), mode=mode, extra=extra, extra_supa=extra_supa)
        res["plans"][mode] = plan_for(cfg, n_prompts)
    for mode, p in res["plans"].items():
        print(f"[{mode}] {p['n_prompts']} prompts; sides {p['sides']}")
        for s in p["stages"]:
            print(f"   {s['stage']:<6} gens {s['generations']:>5}  {s['projected_h_planning']:.2f} h @ {RATE_PLAN}/s   "
                  f"{s['projected_h_optimistic']:.2f} h @ {RATE_OPT}/s")
        print(f"   GPU-h total {p['gpu_h_planning']} (planning) / {p['gpu_h_optimistic']} (optimistic); per arm "
              f"{p['per_arm_h_planning']} / {p['per_arm_h_optimistic']} h; turns {p['turns_planning']}; lp disk "
              f"{p['lp_disk_gb']} GB")
    res["torch_imported"] = "torch" in sys.modules
    need(not res["torch_imported"], "--plan must not import torch")
    if a.record:
        out = RC.DOCS / "RET_PLAN.json"
        need(not RC.FREEZE.exists(), "the kit is frozen: RET_PLAN.json is a frozen record and is not rewritten")
        res["ok"] = True
        res["code_sha256"] = {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_run.py", "scripts/ret_common.py")}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
