"""T2b-register screen runner (RCWM roadmap item 3; joint item 5b): train the three R-only LoRA arms with the T2 recipe,
then evaluate adapter OFF and every arm ON in ONE process, all under ONE GPU lease. Modelled on scripts/p3_run.py.

  python -B scripts/t2br_run.py --plan        # CPU only (no torch, no model, no lease): stages, generations, training
                                              # tokens, GPU time per turn; writes docs/t2b/register/GEN_PLAN.json
  python -B scripts/t2br_run.py --verify      # CPU only: docs/t2b/register/SCREEN_FREEZE.json against the files
  python -B scripts/t2br_run.py --freeze --approval "<who approved, when, DECISIONS entry>"
                                              # CPU only: pre-freeze checks (audit, scorer self-test, analysis dry run,
                                              # tiny tests, plan; no output exists), then hashes every input
                                              # (create-only; refused if frozen)
  python -B scripts/t2br_run.py [--seed 0] [--train-variant base|sd]    # the GPU job (refuses unless the freeze verifies)

GPU job. Before the lease: the freeze must verify; no other RCWM job may hold or wait for the lease (one-waiter rule,
R14); one runner per output dir (run lock). One lease, label 'rcwm-t1:t2b-register' (rcwmt.runtime.gpu_lease:
alternation, thermal flag, duty cycle, RAM floor), turn cap 4.5 h from lock acquisition with D-22's stage rule (no stage
starts within 0.5 h of the cap; a stage starts only if 1.25 x its estimate fits). One model at a time. Stages, in order:
  T_plain, T_trace, T_filler   the T2 recipe per arm (vendored BWM trainer, D-15: bwmn.lm.load_core('titan-sparse') FP32,
                               attach_lora r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down, train_sft 190 steps x
                               batch 16, lr 1e-4 WSD, max_len 512, seed s; train_generic's row order 10007 s + 17), on that
                               arm's rendering of the SAME 750 register rows (data/t2br_train_<arm>.jsonl). The core's
                               fingerprint must be Titan-Sparse's (818be9c6b6a3). Written to train_<arm>.partial/, then
                               renamed (a crash leaves only a .partial, orphaned on resume).
  then load Titan-Sparse (rcwmt.runtime.load_arm, FP32) + the three adapters UNMERGED (PEFT; never a dense merge, D-28):
  A0 anchor   adapter OFF (disable_adapter() = Titan-Sparse) chain_one on the 208 pilot TEST register chains must
              reproduce runs/t2/t2_s0/gen_chain_one_base.jsonl (identical states on >= 99% of chains), else stop before
              any dev-bank output. A resumed turn re-anchors before E1/E2 (recorded; a failure is persisted).
  E1          the 900 one-step dev rows (generation.generate_texts, 160-token cap) per side OFF / plain / trace / filler
  E2          the 384 k=4 dev chains (chain_one self-rollout in the one-step frame; each side's state = its own parser's
              'after': plain/OFF env.parse_state(lenient), trace/filler the strict joint parsers) per side
  Every eval stage generates ALL four sides in the same process and writes them only when all are done.
Outputs (create-only) under runs/t2b_register/s<seed>[_<variant>]/. A stage is DONE only if all its files exist AND
provenance.json records them with matching sha256; anything else written by a stage makes it PARTIAL: moved to
_orphans/ (never deleted) and redone in full. A recorded output whose hash changes stops the job. DONE.json at the end.

Exit codes: 0 complete / already complete; 3 deterministic stop, never retried (freeze, one-waiter, run lock, runtime,
fingerprint, hash, anchor; those that concern the run are persisted in STOPPED.json); 5 resumable (cap reached: the
remaining stages run in the next turn); 1 crash (an exception; the lease is released) - may be retried once.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_gen as G2  # noqa: E402  (torch-free at import: D-22 stage rule, pilot loader, pins, registered runtime)
import t2br_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

# ---------------------------------------------------------------- constants
LEASE_LABEL = "rcwm-t1:t2b-register"
OUT_ROOT = ROOT / "runs" / "t2b_register"
FREEZE_REL = "docs/t2b/register/SCREEN_FREEZE.json"
STAGE_ORDER = ("T_plain", "T_trace", "T_filler", "A0", "E1", "E2")
TRAIN_STAGES = {"T_plain": "plain", "T_trace": "trace", "T_filler": "filler"}
EVAL_STAGES = ("A0", "E1", "E2")
ADAPTER_NAMES = {"plain": "t2br_plain", "trace": "t2br_trace", "filler": "t2br_filler"}
RECIPE = {"core": "titan-sparse", "steps": 190, "batch": 16, "lr": 1e-4, "max_len": 512, "rank": 16, "alpha": 32,
          "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
          "schedule": "WSD (bwmn.lm.wsd_lr: warm 10%, decay last 20% to 0.1x)", "micro": 4,
          "row_order": "train_generic: Random(10007 * seed + 17), concatenated shuffled epochs",
          "source": "T2 (runs/t2/t2_s0/metrics.json args; scripts/run_t2.py _train_body)"}
FP = dict(G2.FP)                                 # sparse 818be9c6b6a3
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)  # torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0
ANCHOR_REF_REL = "runs/t2/t2_s0/gen_chain_one_base.jsonl"
ANCHOR_REF_SHA = G2.ANCHOR_REF_SHA["gen_chain_one_base.jsonl"]
ANCHOR_MIN_IDENTICAL = 0.99
# planning model (D-22 rule inputs)
T2_TRAIN_S = 1858.8              # runs/t2/t2_s0/metrics.json train.wall_s (190 x 16, duty-capped trainer)
T2_TRAIN_TOKENS = 464409         # its train.tokens_processed (non-pad prompt + target + EOS tokens)
RATE_GENS_PER_S = 2.7            # registered planning rate for ~16-token answers (PREREG-2 / PREREG-3)
OBSERVED_RATE = 3.80             # PREREG-2's observed chain_one rate
LOAD_S = 120.0
CAP_H = 4.5
VRAM_EMPTY_MIB = 512.0
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
RCWM_LABEL_PREFIX = "rcwm-t1:"
FREEZE_SCRIPTS = ("scripts/t2br_common.py", "scripts/t2br_make_data.py", "scripts/t2br_run.py", "scripts/t2br_analysis.py",
                  "scripts/t2br_test_tiny.py", "scripts/confirm2_gen.py", "scripts/l1_make_ood.py", "scripts/p3_make_bank.py",
                  "vendor_bwm/scripts/train_generic.py", "vendor_bwm/bwmn/lm.py")
HOOK_DRAFT_REL = "docs/t2b/register/queue_hook_t2br.sh.draft"
FREEZE_DOCS = ("docs/t2b/register/DESIGN.md", "docs/t2b/register/AUDIT.json", "docs/t2b/register/SCORER_SELFTEST.json",
               "docs/t2b/register/DRYRUN.json", "docs/t2b/register/TINY_TEST.json", "docs/t2b/register/GEN_PLAN.json",
               HOOK_DRAFT_REL)
PROV_CODE = ("scripts/t2br_run.py", "scripts/t2br_common.py", "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/env.py",
             "rcwmt/capture.py", "vendor_bwm/bwmn/lm.py")
TINY_CODE = ("scripts/t2br_run.py", "scripts/t2br_common.py", "scripts/t2br_analysis.py", "scripts/t2br_test_tiny.py")
DRYRUN_CODE = ("scripts/t2br_analysis.py", "scripts/t2br_common.py")


class Stop(Exception):
    """A deterministic stop (exit 3). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[t2br] {msg}", persist)


def sha256(p: Path) -> str:
    return C.sha256(p)


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- configuration
@dataclass
class Cfg:
    root: Path = ROOT
    seed: int = 0
    steps: int = RECIPE["steps"]
    train_variant: str = "base"
    out: Path | None = None                      # default runs/t2b_register/s<seed>[_<variant>]
    dev: Path = C.DEV
    chains: Path = C.CHAINS
    train_files: dict | None = None              # default data/t2br_train[_<variant>]_<arm>.jsonl
    anchor_chains: list | None = None            # None: the 208 pilot TEST register chains
    anchor_ref: Path = ROOT / ANCHOR_REF_REL
    require_freeze: bool = True                  # False only from the tiny tests (never reachable from the CLI)
    cap_h: float = CAP_H
    rate: float = RATE_GENS_PER_S

    def __post_init__(self):
        if self.out is None:
            self.out = OUT_ROOT / (f"s{self.seed}" + ("" if self.train_variant == "base" else f"_{self.train_variant}"))
        if self.train_files is None:
            self.train_files = {a: C.train_path(a, self.train_variant) for a in C.ARMS}
        self.out = Path(self.out)


def pilot_anchor_chains() -> list:
    return G2.load_bank_chains(G2.PILOT_PATH, G2.PILOT_SHA, "R", "test")


def row_order(n_rows: int, steps: int, batch: int, seed: int) -> list:
    """scripts/train_generic.py's order (vendor_bwm), so the three arms see identical rows in identical order."""
    rng = random.Random(10007 * seed + 17)
    order = []
    while len(order) < steps * batch:
        idx = list(range(n_rows))
        rng.shuffle(idx)
        order.extend(idx)
    return order


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(stage: str, out: Path) -> list:
    o = Path(out)
    if stage in TRAIN_STAGES:
        d = o / f"train_{TRAIN_STAGES[stage]}"
        return [d / "adapter" / "adapter_model.safetensors", d / "adapter" / "adapter_config.json", d / "metrics.json"]
    if stage == "A0":
        return [o / "anchor" / "chain_one_R_off.jsonl", o / "anchor" / "anchor.json"]
    if stage == "E1":
        return [o / "onestep" / f"onestep_{s}.jsonl" for s in C.SIDES]
    if stage == "E2":
        return [o / "chains" / f"chains_{s}.jsonl" for s in C.SIDES]
    raise KeyError(stage)


def stage_owned(stage: str, out: Path) -> list:
    o = Path(out)
    if stage in TRAIN_STAGES:
        d = o / f"train_{TRAIN_STAGES[stage]}"
        return [d, _partial(d)]
    if stage == "A0":
        return [o / "anchor"]
    fs = stage_required(stage, o)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path) -> str:
    return str(Path(p).relative_to(Path(out))).replace("\\", "/")


def stage_states(prov: dict, out: Path) -> dict:
    res = {}
    for st in STAGE_ORDER:
        req = stage_required(st, out)
        rec = (prov.get("stages") or {}).get(st)
        if all(p.exists() for p in req) and rec:
            files = rec.get("files") or {}
            for p in req:
                r = rel_out(p, out)
                need(r in files, f"stage {st}: {r} not recorded in provenance")
                need(sha256(p) == files[r], f"stage {st}: {r} changed after it was recorded (sha256)")
            res[st] = "done"
        elif any(p.exists() for p in stage_owned(st, out)):
            res[st] = "partial"
        else:
            res[st] = "todo"
    return res


def orphan(stage: str, out: Path) -> Path:
    dst = Path(out) / "_orphans" / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    dst.mkdir(parents=True, exist_ok=False)
    for p in stage_owned(stage, out):
        if p.exists():
            shutil.move(str(p), str(dst / p.name))
    print(f"[t2br] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Stop(f"[t2br] {path} exists (create-only)")
    tmp = _partial(path)
    with tmp.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- counts, estimates, the D-22 rule, turns
def counts(cfg: Cfg | None = None) -> dict:
    cfg = cfg or Cfg()
    dev = C.load_dev(cfg.dev)
    parents = C.load_chain_bank(cfg.chains)
    n_cand = sum(len(p["candidates"]) for p in parents)
    n_anchor = len(cfg.anchor_chains) if cfg.anchor_chains is not None else len(pilot_anchor_chains())
    tok = {}
    for a in C.ARMS:
        rows = C.load_jsonl(cfg.train_files[a])
        order = row_order(len(rows), cfg.steps, RECIPE["batch"], cfg.seed)
        per = [r["n_prompt_tokens"] + r["n_target_tokens"] + 1 for r in rows]
        tok[a] = {"rows": len(rows), "row_visits": len(order[: cfg.steps * RECIPE["batch"]]),
                  "epochs": round(cfg.steps * RECIPE["batch"] / len(rows), 3),
                  "tokens": sum(per[i] for i in order[: cfg.steps * RECIPE["batch"]])}
    mean = {a: sum(r["n_target_tokens"][a] for r in dev) / len(dev) for a in C.ARMS}
    long_factor = (mean["trace"] + 1) / (mean["plain"] + 1)
    return {"rows": len(dev), "chains": n_cand, "anchor_chains": n_anchor, "A0": n_anchor * C.H,
            "E1": {"short": 2 * len(dev), "long": 2 * len(dev)}, "E2": {"short": 2 * n_cand * C.H, "long": 2 * n_cand * C.H},
            "train": tok, "mean_target_tokens_dev": {a: round(v, 2) for a, v in mean.items()},
            "long_factor": round(long_factor, 4)}


def stage_est_s(st: str, rate: float, n: dict) -> float:
    """Seconds. Training: T2's measured time scaled by the arm's processed tokens, plus one load. Generation: short
    answers (OFF, plain) at `rate`, trace / filler answers at rate / long_factor (decode time ~ answer length). A0
    includes the eval model load (it is the first eval step of every turn that evaluates)."""
    if st in TRAIN_STAGES:
        return LOAD_S + T2_TRAIN_S * n["train"][TRAIN_STAGES[st]]["tokens"] / T2_TRAIN_TOKENS
    if st == "A0":
        return LOAD_S + n["A0"] / rate
    return n[st]["short"] / rate + n[st]["long"] * n["long_factor"] / rate


def can_start(elapsed_s: float, est_s: float, cap_s: float) -> bool:
    return G2.can_start(elapsed_s, est_s, 0.0, cap_s)


def simulate_turns(rate: float, n: dict, cap_h: float = CAP_H) -> list:
    """The runner's own scheduling rules on projected times (see _run_locked)."""
    cap = cap_h * 3600
    left = list(STAGE_ORDER)
    turns = []
    while left:
        t, ran = 0.0, []
        for st in [s for s in left if s in TRAIN_STAGES]:
            if not can_start(t, stage_est_s(st, rate, n), cap):
                break
            t += stage_est_s(st, rate, n)
            ran.append(st)
            left.remove(st)
        if not any(s in TRAIN_STAGES for s in left):
            ev = [s for s in left if s in EVAL_STAGES and s != "A0"]
            a0_todo = "A0" in left
            if a0_todo or ev:
                first = stage_est_s("A0", rate, n) + (stage_est_s(ev[0], rate, n) if ev else 0.0)
                if can_start(t, first, cap):
                    t += stage_est_s("A0", rate, n)
                    ran.append("A0" if a0_todo else "A0 (re-anchor)")
                    if a0_todo:
                        left.remove("A0")
                    for st in ev:
                        if not can_start(t, stage_est_s(st, rate, n), cap):
                            break
                        t += stage_est_s(st, rate, n)
                        ran.append(st)
                        left.remove(st)
        if not [r for r in ran if not r.startswith("A0 (")]:
            raise RuntimeError(f"no stage fits an empty turn at {rate} gens/s")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(cfg: Cfg | None = None, verbose: bool = True) -> dict:
    cfg = cfg or Cfg()
    n = counts(cfg)
    what = {"T_plain": "train plain arm", "T_trace": "train trace arm", "T_filler": "train filler arm",
            "A0": f"load Sparse + 3 adapters; anchor OFF chain_one, {n['anchor_chains']} pilot TEST register chains",
            "E1": f"{n['rows']} one-step dev rows x OFF/plain (short) + trace/filler (long)",
            "E2": f"{n['chains']} k={C.H} dev chains x OFF/plain (short) + trace/filler (long)"}
    gens = {"A0": n["A0"], "E1": n["E1"]["short"] + n["E1"]["long"], "E2": n["E2"]["short"] + n["E2"]["long"]}
    res = {"rate_gens_per_s": RATE_GENS_PER_S, "observed_rate_gens_per_s": OBSERVED_RATE, "load_s_each": LOAD_S,
           "cap_h": CAP_H, "recipe": RECIPE, "seed": cfg.seed, "train_variant": cfg.train_variant,
           "long_answer_factor": n["long_factor"], "mean_target_tokens_dev": n["mean_target_tokens_dev"],
           "train": n["train"], "t2_reference": {"train_wall_s": T2_TRAIN_S, "tokens_processed": T2_TRAIN_TOKENS},
           "stages": [{"stage": s, "what": what[s], "generations_max": gens.get(s, 0),
                       "projected_s_planning": round(stage_est_s(s, RATE_GENS_PER_S, n), 1),
                       "projected_s_observed": round(stage_est_s(s, OBSERVED_RATE, n), 1)} for s in STAGE_ORDER],
           "total_generations_max": sum(gens.values()), "model_loads": 4,
           "stage_rule": f"no stage starts within {G2.STAGE_MARGIN_H} h of the cap; a stage starts only if "
                         f"{G2.STAGE_SAFETY} x its estimate fits; eval starts only if A0 + the next eval stage fits; a "
                         "turn that evaluates after an earlier A0 re-anchors first",
           "turns_at_planning_rate": simulate_turns(RATE_GENS_PER_S, n),
           "turns_at_observed_rate": simulate_turns(OBSERVED_RATE, n)}
    res["projected_gpu_h_planning"] = round(sum(t["projected_h"] for t in res["turns_at_planning_rate"]), 2)
    res["projected_gpu_h_observed"] = round(sum(t["projected_h"] for t in res["turns_at_observed_rate"]), 2)
    if verbose:
        print(f"T2b-register GPU plan (no model loaded; {LOAD_S:.0f} s per model load; long-answer factor "
              f"{n['long_factor']:.2f})")
        for r in res["stages"]:
            print(f"  {r['stage']:<9} {r['what']:<78} gens<= {r['generations_max']:>5}  "
                  f"{r['projected_s_planning'] / 60:6.1f} min @{RATE_GENS_PER_S}/s  {r['projected_s_observed'] / 60:6.1f} "
                  f"min @{OBSERVED_RATE}/s")
        for a, v in n["train"].items():
            print(f"  train {a:<7} {v['rows']} rows, {v['row_visits']} row visits ({v['epochs']} epochs), {v['tokens']:,} tokens")
        for k, rate in (("turns_at_planning_rate", RATE_GENS_PER_S), ("turns_at_observed_rate", OBSERVED_RATE)):
            print(f"  turns at {rate} gens/s (cap {CAP_H} h): " +
                  "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(res[k])))
        print(f"  projected GPU-h: {res['projected_gpu_h_planning']} (planning) / {res['projected_gpu_h_observed']} (observed)")
    return res


# ---------------------------------------------------------------- freeze (a screen lock: the go rule and code fixed before any output)
def freeze_inputs(root: Path) -> list:
    root = Path(root)
    rels = list(FREEZE_SCRIPTS) + list(FREEZE_DOCS) + [ANCHOR_REF_REL, "data/bank_pilot.json"]
    rels += [f"data/{C.train_path(a).name}" for a in C.ARMS] + [f"data/{C.train_source().name}",
                                                                f"data/{C.DEV.name}", f"data/{C.CHAINS.name}",
                                                                f"data/{C.DEV_SOURCE.name}"]
    for a in C.ARMS:                                  # the surface-diversity variant, if it was built before the freeze
        if (root / "data" / C.train_path(a, "sd").name).is_file():
            rels.append(f"data/{C.train_path(a, 'sd').name}")
    if (root / "data" / C.train_source("sd").name).is_file():
        rels.append(f"data/{C.train_source('sd').name}")
    rels += [f"rcwmt/{p.name}" for p in sorted((root / "rcwmt").glob("*.py"))]
    return sorted(set(rels))


def outputs_present(root: Path) -> list:
    d = Path(root) / "runs" / "t2b_register"
    return sorted(str(p.relative_to(root)) for p in d.iterdir()) if d.exists() else []


def verify_freeze(root: Path | None = None) -> tuple[bool, list, dict | None]:
    root = Path(root or ROOT)
    fz = C.read_json(root / FREEZE_REL)
    if fz is None:
        return False, ["SCREEN_FREEZE.json does not exist or is unreadable (the screen is not frozen)"], None
    errs = []
    inputs = dict(fz.get("inputs") or {})
    base_need = [r for r in freeze_inputs(root) if "_sd" not in r]
    missing = sorted(set(base_need) - set(inputs))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing}")
    for rel, h in sorted(inputs.items()):
        p = root / rel
        if not p.is_file():
            errs.append(f"{rel} missing")
        elif sha256(p) != h:
            errs.append(f"{rel} differs from the freeze")
    if inputs.get(ANCHOR_REF_REL) != ANCHOR_REF_SHA:
        errs.append("anchor reference is not the pinned T2 s0 file")
    if fz.get("registered_runtime") != REGISTERED_RUNTIME:
        errs.append("registered runtime differs")
    if fz.get("go_rule_text") != C.GO_RULE_TEXT:
        errs.append("go rule text differs from scripts/t2br_common.GO_RULE_TEXT")
    if not fz.get("approval"):
        errs.append("no approval recorded")
    return not errs, errs, fz


def prefreeze_checks(root: Path) -> dict:
    root = Path(root)
    out = {}
    au = C.read_json(root / "docs/t2b/register/AUDIT.json") or {}
    files = au.get("files_sha256") or {}
    out["AUDIT.json ok (full verify incl. the historical harvest) on the current data and builder"] = bool(au.get("ok")) \
        and bool(au.get("historical_harvest")) and all(files.get(n) == sha256(root / "data" / n) for n in files) and bool(files) \
        and all((au.get("code_sha256") or {}).get(r) == sha256(root / r) for r in ("scripts/t2br_make_data.py", "scripts/t2br_common.py"))
    sf = C.read_json(root / "docs/t2b/register/SCORER_SELFTEST.json") or {}
    out["scorer self-test passed on this dev bank"] = bool(sf.get("ok")) and sf.get("dev_sha256") == sha256(root / "data" / C.DEV.name)
    dr = C.read_json(root / "docs/t2b/register/DRYRUN.json") or {}
    out["analysis dry run passed with the current analysis code"] = bool(dr.get("ok")) and all(
        (dr.get("code_sha256") or {}).get(r) == sha256(root / r) for r in DRYRUN_CODE)
    tt = C.read_json(root / "docs/t2b/register/TINY_TEST.json") or {}
    out["tiny-model runner tests passed with the current code"] = bool(tt.get("ok")) and all(
        (tt.get("code_sha256") or {}).get(r) == sha256(root / r) for r in TINY_CODE)
    gp = C.read_json(root / "docs/t2b/register/GEN_PLAN.json") or {}
    out["GEN_PLAN.json present"] = bool(gp.get("stages"))
    out["DESIGN.md states the go rule verbatim"] = (root / "docs/t2b/register/DESIGN.md").is_file() and \
        C.GO_RULE_TEXT in (root / "docs/t2b/register/DESIGN.md").read_text(encoding="utf-8")
    out["no T2b-register output exists"] = not outputs_present(root)
    return out


def freeze(approval: str, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    if not approval or len(approval.strip()) < 8:
        raise Stop("--approval must state who approved the freeze, when, and the DECISIONS entry", persist=False)
    fzp = root / FREEZE_REL
    if fzp.exists() or _partial(fzp).exists():
        raise Stop("the screen is already frozen (or a freeze was interrupted: SCREEN_FREEZE.json.partial)", persist=False)
    chk = prefreeze_checks(root)
    bad = [k for k, v in chk.items() if not v]
    if bad:
        raise Stop(f"pre-freeze checks failed: {bad}", persist=False)
    inputs = {}
    for rel in freeze_inputs(root):
        if not (root / rel).is_file():
            raise Stop(f"frozen input missing: {rel}", persist=False)
        inputs[rel] = sha256(root / rel)
    if inputs[ANCHOR_REF_REL] != ANCHOR_REF_SHA:
        raise Stop("the anchor reference is not the pinned T2 s0 file", persist=False)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    rec = {"frozen_at": stamp, "approval": approval.strip(), "inputs": inputs, "prefreeze_checks": chk,
           "go_rule_text": C.GO_RULE_TEXT, "registered_runtime": REGISTERED_RUNTIME, "lease_label": LEASE_LABEL,
           "recipe": RECIPE, "out_dir": "runs/t2b_register", "no_outputs_before_freeze": True}
    with _partial(fzp).open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    os.rename(_partial(fzp), fzp)
    ok, errs, _ = verify_freeze(root)
    if not ok:
        raise Stop(f"the freeze record was written but does not verify: {errs}", persist=False)
    print(f"[t2br] FROZEN {stamp}; SCREEN_FREEZE.json sha {sha256(fzp)}", flush=True)
    return rec


# ---------------------------------------------------------------- run lock, one-waiter rule, runtime
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
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for or holds the GPU lease. The peer's jobs are allowed
    (this job then waits in the lease as usual)."""
    w = C.read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = C.read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and _live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


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
            raise Stop(f"[t2br] another t2br_run.py holds the run lock {rec}", persist=False)
        print(f"[t2br] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink()
    with lk.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return lk


def hook_problem(root: Path, fz: dict | None) -> str | None:
    """An installed scripts/queue_hooks/*t2br*.sh must be the hook draft byte for byte (the frozen sha256 once frozen)."""
    root = Path(root)
    hooks = sorted((root / "scripts" / "queue_hooks").glob("*t2br*.sh"))
    if not hooks:
        return None
    if fz is not None:
        allowed = {(fz.get("inputs") or {}).get(HOOK_DRAFT_REL)}
    else:
        allowed = {sha256(root / HOOK_DRAFT_REL)} if (root / HOOK_DRAFT_REL).is_file() else set()
    bad = [h.name for h in hooks if sha256(h) not in allowed]
    return f"installed hook(s) {bad} are not {HOOK_DRAFT_REL} byte for byte" if bad else None


def runtime_problem(rt: dict, versions: dict | None = None) -> str | None:
    bad = {k: rt.get(k) for k in ("torch", "transformers") if rt.get(k) != REGISTERED_RUNTIME[k]}
    if versions is not None and versions.get("peft") != REGISTERED_RUNTIME["peft"]:
        bad["peft"] = versions.get("peft")
    return f"runtime {bad} differs from the registered {REGISTERED_RUNTIME}" if bad else None


# ---------------------------------------------------------------- model-facing pieces (torch imported lazily)
class RealBackend:
    """The GPU path: rcwmt.runtime (lease, numerics, loads) and the vendored BWM trainer (vendor_bwm/bwmn/lm.py)."""
    device = "cuda"

    def __init__(self, rt=None, lm=None):
        if rt is None:
            import torch  # noqa: F401
            from rcwmt import runtime as rt
        if lm is None:
            sys.path.insert(0, str(ROOT / "vendor_bwm"))
            from bwmn import lm
        self.rt, self.lm = rt, lm

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 T2b-register screen: 3 R-only LoRA arms (plain/trace/"
                                                        "filler) + OFF/ON eval, one lease (D-22 style)")

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
            print("[t2br] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)

    def wait(self) -> None:
        self.thermal_wait()
        self.rt.wait_for_resources()

    def train_core(self):
        """The trainer's own loader (bwmn.lm.load_core: FP32, TF32 off, memory fraction 0.70), as T2 trained."""
        model, tok = self.lm.load_core(RECIPE["core"])
        return model, tok, self.rt.param_fingerprint(model)

    def load_eval(self, adapter_dirs: dict):
        from peft import PeftModel
        import peft
        base, p = self.rt.load_arm("sparse")
        model = PeftModel.from_pretrained(base, str(adapter_dirs["plain"]), adapter_name=ADAPTER_NAMES["plain"],
                                          is_trainable=False)
        for a in ("trace", "filler"):
            model.load_adapter(str(adapter_dirs[a]), adapter_name=ADAPTER_NAMES[a], is_trainable=False)
        model.eval()
        return model, p["param_fingerprint"], {"peft": peft.__version__}

    def tokenizer(self):
        return self.rt.tokenizer()

    def unload(self) -> None:
        self.rt.unload(None)

    def vram_allocated_mib(self) -> float:
        import torch
        return torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0

    def runtime_info(self) -> dict:
        return self.rt.runtime_info()

    def versions(self) -> dict:
        import peft
        return {"peft": peft.__version__}

    def check_fingerprint(self, which: str, fp: str) -> None:
        need(fp.startswith(FP[which]), f"{which} fingerprint {fp[:12]} != {FP[which]}")


@contextmanager
def side_ctx(model, side: str):
    """off = every adapter disabled (Titan-Sparse); otherwise only that arm's adapter is active."""
    if side == "off":
        with model.disable_adapter():
            yield
        return
    name = ADAPTER_NAMES[side]
    model.set_adapter(name)
    act = model.active_adapter
    need(act == name or act == [name], f"active adapter {act} != {name}")
    yield


def t2br_chain(model, tok, chains: list, arm: str) -> list:
    """chain_one self-rollout over register chains: rcwmt.generation.chain's loop for one family, with the decision
    parser of `arm` (plain: env.parse_state(lenient) exactly as generation.chain; trace / filler: the strict joint
    parsers, then env.parse_state on 'after'). A chain breaks (None) once an answer has no parseable 'after'."""
    from rcwmt import generation
    states = [[c["s0"]] for c in chains]
    texts = [[] for _ in chains]
    H = max(len(c["events"]) for c in chains)
    for t in range(H):
        live = [i for i, c in enumerate(chains) if t < len(c["events"]) and states[i][-1] is not None]
        if live:
            msgs = [env.one_step_messages(C.FAM, states[i][-1], chains[i]["events"][t]) for i in live]
            txt = generation.generate_texts(model, tok, msgs, generation.max_tokens(C.FAM))
            for i, s in zip(live, txt):
                texts[i].append(s)
                states[i].append(C.parse_after(s, arm))
        for i, c in enumerate(chains):
            if t < len(c["events"]) and states[i][-1] is None and len(states[i]) == t + 1:
                states[i].append(None)
    return [{"states": s, "texts": x} for s, x in zip(states, texts)]


def chain_rows(chains, results) -> list:
    return [{"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]], "texts": r["texts"]}
            for c, r in zip(chains, results)]


def onestep_texts(model, tok, rows) -> list:
    from rcwmt import generation
    return generation.generate_texts(model, tok, [r["messages"] for r in rows], generation.max_tokens(C.FAM))


def run_anchor(model, tok, chains: list, ref_path: Path) -> tuple:
    from rcwmt import generation
    t0 = time.time()
    with side_ctx(model, "off"):
        rr = generation.chain(model, tok, chains, "chain_one")
    ref = {}
    for line in Path(ref_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ref[row["cand_id"]] = row
    st = sum([None if x is None else env.dumps(x) for x in r["states"]] == (ref.get(c["cand_id"]) or {}).get("states")
             for c, r in zip(chains, rr))
    tx = sum(r["texts"] == (ref.get(c["cand_id"]) or {}).get("texts") for c, r in zip(chains, rr))
    res = {"pid": os.getpid(), "side": "off", "n": len(chains), "identical_states": st / len(chains),
           "identical_texts": tx / len(chains), "start_unix": t0, "end_unix": time.time(),
           "generations": sum(len(r["texts"]) for r in rr), "min_identical": ANCHOR_MIN_IDENTICAL}
    res["pass"] = res["identical_states"] >= ANCHOR_MIN_IDENTICAL
    return rr, res


def train_arm(cfg: Cfg, backend, arm: str, dst: Path) -> dict:
    """One arm with the T2 recipe, written into dst (a .partial dir; the caller renames it)."""
    import gc
    rows = C.load_jsonl(cfg.train_files[arm])
    order = row_order(len(rows), cfg.steps, RECIPE["batch"], cfg.seed)
    dst.mkdir(parents=True, exist_ok=False)
    logf = (dst / "log.txt").open("a", encoding="utf-8")

    def P(*s):
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    model, tok, fp = backend.train_core()
    pm = None
    try:
        backend.check_fingerprint("sparse", fp)
        P(f"[t2br] {arm}: rows {len(rows)} from {cfg.train_files[arm].name}; steps {cfg.steps} x {RECIPE['batch']} = "
          f"{cfg.steps * RECIPE['batch'] / len(rows):.2f} epochs; core fingerprint {fp[:12]}")
        pm = backend.lm.attach_lora(model, rank=RECIPE["rank"], alpha=RECIPE["alpha"], dropout=RECIPE["dropout"],
                                    targets=tuple(RECIPE["targets"]), seed=cfg.seed)
        t0 = time.time()
        log = backend.lm.train_sft(pm, tok, rows, steps=cfg.steps, batch_size=RECIPE["batch"], lr=RECIPE["lr"],
                                   seed=cfg.seed, logger=P, row_order=order, max_len=RECIPE["max_len"])
        meta = {"core": RECIPE["core"], "arm": arm, "rows": len(rows), "train": log, "fingerprint": fp,
                "args": {"rows": str(cfg.train_files[arm].relative_to(cfg.root)) if cfg.train_files[arm].is_relative_to(cfg.root)
                         else str(cfg.train_files[arm]), "steps": cfg.steps, "batch": RECIPE["batch"], "lr": RECIPE["lr"],
                         "seed": cfg.seed, "max_len": RECIPE["max_len"], "rank": RECIPE["rank"], "alpha": RECIPE["alpha"],
                         "dropout": RECIPE["dropout"], "targets": RECIPE["targets"]},
                "data_sha256": sha256(cfg.train_files[arm])}
        backend.lm.save_adapter(pm, dst / "adapter", meta)
        (dst / "metrics.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        P(f"[t2br] {arm}: done in {time.time() - t0:.0f}s")
        return meta
    finally:
        logf.close()
        del pm, model
        gc.collect()
        backend.unload()


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = cfg.out
    fz = None
    if cfg.require_freeze:
        ok, errs, fz = verify_freeze(cfg.root)
        if not ok:
            raise Stop("refusing to run the T2b-register GPU job: " + "; ".join(errs[:6]),
                       persist=(Path(cfg.root) / FREEZE_REL).exists())
        ins = fz["inputs"]
        for a in C.ARMS:
            rel = f"data/{cfg.train_files[a].name}"
            need(rel in ins and sha256(cfg.train_files[a]) == ins[rel], f"{rel} is not a frozen input (or differs)")
        need(sha256(cfg.dev) == ins[f"data/{C.DEV.name}"] and sha256(cfg.chains) == ins[f"data/{C.CHAINS.name}"],
             "dev bank differs from the freeze")
        need(sha256(cfg.anchor_ref) == ANCHOR_REF_SHA, "anchor reference differs from the pinned T2 s0 file")
    hp = hook_problem(cfg.root, fz)
    if hp:
        raise Stop(f"[t2br] {hp}; reinstall it byte for byte from {HOOK_DRAFT_REL} (or remove it) before any run",
                   persist=False)
    if (out / "DONE.json").exists():
        print(f"[t2br] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; an operator must investigate, "
                   "log it in DECISIONS.md and move the file aside before any rerun", persist=False)
    a0j = out / "anchor" / "anchor.json"
    if a0j.exists() and not (C.read_json(a0j) or {}).get("pass"):
        raise Stop("the anchor check failed in an earlier turn (investigate, log)")
    if list((out / "anchor").glob("RECHECK_FAILED_*.json")):
        raise Stop("an anchor re-check failed in an earlier turn (investigate, log)")
    dev = C.load_dev(cfg.dev)
    parents = C.load_chain_bank(cfg.chains)
    chains = C.flat_chains(parents)
    anchor_chains = cfg.anchor_chains if cfg.anchor_chains is not None else pilot_anchor_chains()
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[t2br] {ow}: one-waiter rule (R14). Start only when no other RCWM GPU job is running or waiting",
                   persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, fz, dev, chains, anchor_chains)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg, backend, out, fz, dev, chains, anchor_chains) -> int:
    prov_path = out / "provenance.json"
    prov = C.read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
    else:
        prov = {"registered": bool(cfg.require_freeze), "freeze_sha256": sha256(cfg.root / FREEZE_REL) if fz else None,
                "seed": cfg.seed, "steps": cfg.steps, "train_variant": cfg.train_variant, "recipe": RECIPE,
                "lease_label": LEASE_LABEL, "stages": {}, "turns": []}
    prov["data_sha256"] = {**{a: sha256(cfg.train_files[a]) for a in C.ARMS}, "dev": sha256(cfg.dev),
                           "chains": sha256(cfg.chains), "anchor_ref": sha256(cfg.anchor_ref)}
    prov["code_sha256"] = {r: sha256(ROOT / r) for r in PROV_CODE}

    def save():
        tmp = _partial(prov_path)
        tmp.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, prov_path)

    states = stage_states(prov, out)
    for st, s in states.items():
        if s == "partial":
            orphan(st, out)
            prov["stages"].pop(st, None)
            states[st] = "todo"
    done = {st for st, s in states.items() if s == "done"}

    def record(st: str, extra: dict) -> None:
        prov["stages"][st] = {"name": st, "turn": len(prov["turns"]) - 1, **extra,
                              "files": {rel_out(p, out): sha256(p) for p in stage_required(st, out)}, "finished_unix": time.time()}
        done.add(st)
        save()

    def finalize() -> None:
        need(set(prov["stages"]) == set(STAGE_ORDER) and stage_states(prov, out) == {s: "done" for s in STAGE_ORDER},
             "finalize: not every stage is recorded and unchanged")
        with (out / "DONE.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps({"finished_unix": time.time(), "stages": list(STAGE_ORDER),
                                "files": {st: prov["stages"][st]["files"] for st in STAGE_ORDER},
                                "provenance_sha256": sha256(prov_path)}, indent=1) + "\n")
        print("[t2br] all stages complete; DONE.json written", flush=True)

    todo = [st for st in STAGE_ORDER if st not in done]
    print(f"[t2br] stages to run: {todo}", flush=True)
    if not todo:
        save()
        finalize()
        return 0
    n = counts(cfg)
    with backend.lease():
        t_acq, t_src = backend.lease_acquired_unix()
        cap_s = cfg.cap_h * 3600
        turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                "code_sha256": dict(prov["code_sha256"]), "stages_run": [], "deferred": [], "device": backend.device}
        prov["turns"].append(turn)
        save()
        turn["runtime"], turn["versions"] = backend.runtime_info(), backend.versions()
        rp = runtime_problem(turn["runtime"], turn["versions"])
        if rp:
            save()
            raise Stop(f"[t2br] {rp}: restore the registered environment (nothing was trained or generated)")
        rate = [cfg.rate]

        def fits(est: float) -> bool:
            return can_start(time.time() - t_acq, est, cap_s)

        def vram_empty(what: str) -> None:
            alloc = backend.vram_allocated_mib()
            need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated before {what}; not loading", persist=False)

        # ---- training stages
        for st in [s for s in STAGE_ORDER if s in TRAIN_STAGES and s not in done]:
            if not fits(stage_est_s(st, rate[0], n)):
                turn["deferred"] = [s for s in STAGE_ORDER if s not in done]
                print(f"[t2br] cap: deferring {turn['deferred']} to the next turn", flush=True)
                save()
                break
            arm = TRAIN_STAGES[st]
            vram_empty(f"training {arm}")
            backend.wait()
            t0 = time.time()
            final = out / f"train_{arm}"
            meta = train_arm(cfg, backend, arm, _partial(final))
            os.rename(_partial(final), final)
            record(st, {"seconds": round(time.time() - t0, 1), "pid": os.getpid(), "fingerprint": meta["fingerprint"],
                        "final_loss_avg50": meta["train"].get("final_loss_avg50"),
                        "supervised_tokens": meta["train"].get("supervised_tokens"),
                        "tokens_processed": meta["train"].get("tokens_processed")})
            turn["stages_run"].append(st)
            save()
            print(f"[t2br] {st} done in {time.time() - t0:.0f}s", flush=True)
        # ---- evaluation stages (all arms trained; same process, OFF + three adapters)
        ev_todo = [s for s in EVAL_STAGES if s not in done]
        if ev_todo and all(s in done for s in TRAIN_STAGES) and not turn["deferred"]:
            nxt = [s for s in ev_todo if s != "A0"]
            first = stage_est_s("A0", rate[0], n) + (stage_est_s(nxt[0], rate[0], n) if nxt else 0.0)
            if not fits(first):
                turn["deferred"] = list(ev_todo)
                print(f"[t2br] cap: deferring {ev_todo} to the next turn (A0 + the next stage do not fit)", flush=True)
                save()
            else:
                vram_empty("the eval load")
                backend.wait()
                t1 = time.time()
                ads = {a: out / f"train_{a}" / "adapter" for a in C.ARMS}
                ad_sha = {a: sha256(ads[a] / "adapter_model.safetensors") for a in C.ARMS}
                for a in C.ARMS:
                    rec = prov["stages"][f"T_{a}"]["files"][rel_out(ads[a] / "adapter_model.safetensors", out)]
                    need(ad_sha[a] == rec, f"adapter {a} changed after training (sha256)")
                model, fp, versions = backend.load_eval(ads)
                try:
                    backend.check_fingerprint("sparse", fp)
                    turn["eval_fingerprint"] = fp
                    turn["eval_versions"] = versions
                    turn["eval_runtime"] = backend.runtime_info()
                    rp = runtime_problem(turn["eval_runtime"], versions)
                    if rp:
                        save()
                        raise Stop(f"[t2br] {rp}: restore the registered environment (nothing generated)")
                    turn["eval_load_s"] = round(time.time() - t1, 1)
                    tok = backend.tokenizer()
                    save()
                    if "A0" in done:                           # a resumed eval turn re-anchors first
                        _, res = run_anchor(model, tok, anchor_chains, cfg.anchor_ref)
                        turn["anchor_recheck"] = res
                        save()
                        if not res["pass"]:
                            with (out / "anchor" / f"RECHECK_FAILED_{int(time.time())}.json").open("x", encoding="utf-8") as fh:
                                fh.write(json.dumps(res, indent=1) + "\n")
                            raise Stop(f"[t2br] ANCHOR RECHECK FAILED ({res['identical_states']:.3f} identical states)")
                    for st in ev_todo:
                        if st != "A0" and not fits(stage_est_s(st, rate[0], n)):
                            turn["deferred"] = ev_todo[ev_todo.index(st):]
                            print(f"[t2br] cap: deferring {turn['deferred']} to the next turn", flush=True)
                            save()
                            break
                        t0 = time.time()
                        sides_t = {}
                        if st == "A0":
                            rr, res = run_anchor(model, tok, anchor_chains, cfg.anchor_ref)
                            write_jsonl(out / "anchor" / "chain_one_R_off.jsonl", chain_rows(anchor_chains, rr))
                            with (out / "anchor" / "anchor.json").open("x", encoding="utf-8") as fh:
                                fh.write(json.dumps(res, indent=1) + "\n")
                            n_g = res["generations"]
                            if not res["pass"]:
                                save()
                                raise Stop(f"[t2br] ANCHOR FAILED ({res['identical_states']:.3f} identical states): the "
                                           "eval instrument does not reproduce T2's OFF chains; stop before any dev output")
                            sides_t = {"off": {"start_unix": res["start_unix"], "end_unix": res["end_unix"]}}
                        elif st == "E1":
                            res_t = {}
                            for s in C.SIDES:
                                ts = time.time()
                                with side_ctx(model, s):
                                    res_t[s] = onestep_texts(model, tok, dev)
                                sides_t[s] = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                            for s in C.SIDES:
                                write_jsonl(out / "onestep" / f"onestep_{s}.jsonl",
                                            [{"id": r["id"], "text": t} for r, t in zip(dev, res_t[s])])
                            n_g = len(dev) * len(C.SIDES)
                        else:
                            res_c = {}
                            for s in C.SIDES:
                                ts = time.time()
                                with side_ctx(model, s):
                                    res_c[s] = t2br_chain(model, tok, chains, C.SIDE_ARM[s])
                                sides_t[s] = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                            for s in C.SIDES:
                                write_jsonl(out / "chains" / f"chains_{s}.jsonl", chain_rows(chains, res_c[s]))
                            n_g = sum(len(r["texts"]) for s in C.SIDES for r in res_c[s])
                        sec = time.time() - t0
                        if st != "A0":
                            rate[0] = max(0.3, (n[st]["short"] + n[st]["long"] * n["long_factor"]) / max(sec, 1e-9))
                        record(st, {"generations_planned_max": (n["A0"] if st == "A0" else n[st]["short"] + n[st]["long"]),
                                    "generations": n_g, "seconds": round(sec, 1), "pid": os.getpid(), "sides": sides_t})
                        turn["stages_run"].append(st)
                        save()
                        print(f"[t2br] {st} done in {sec:.0f}s ({n_g} generations)", flush=True)
                finally:
                    del model
                    backend.unload()
        turn["released_unix"] = time.time()
        save()
    if done >= set(STAGE_ORDER):
        finalize()
        return 0
    print(f"[t2br] turn ended; remaining stages {[s for s in STAGE_ORDER if s not in done]} resume next turn", flush=True)
    return 5


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "STOPPED.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)},
                                indent=1) + "\n")
    except FileExistsError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="CPU only: no torch, no model; projected generations and GPU time")
    g.add_argument("--verify", action="store_true", help="CPU only: verify SCREEN_FREEZE.json against the files")
    g.add_argument("--freeze", action="store_true", help="CPU only: freeze the screen (refuses if any output exists)")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-variant", choices=("base", "sd"), default="base",
                    help="sd = the surface-diversity rendering (built by t2br_make_data.py --surface-diversity on)")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan or a.verify or a.freeze:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
            os.environ[v] = "2"
    cfg = None
    try:
        if a.plan:
            cfg = Cfg(seed=a.seed, train_variant=a.train_variant)
            res = plan(cfg)
            ok, errs, _ = verify_freeze()
            res["checks"] = {"dev_sha256": sha256(C.DEV), "chains_sha256": sha256(C.CHAINS),
                             "train_sha256": {x: sha256(cfg.train_files[x]) for x in C.ARMS},
                             "anchor_ref_matches_pin": sha256(ROOT / ANCHOR_REF_REL) == ANCHOR_REF_SHA,
                             "outputs_present": outputs_present(ROOT), "freeze_ok": ok, "freeze_blockers": errs[:3],
                             "torch_imported": "torch" in sys.modules}
            print(json.dumps(res["checks"], indent=1))
            need("torch" not in sys.modules, "--plan must not import torch", persist=False)
            if (ROOT / FREEZE_REL).exists():
                print("the screen is frozen: GEN_PLAN.json is a frozen input and is not rewritten")
                return 0
            C.DOCS.mkdir(parents=True, exist_ok=True)
            (C.DOCS / "GEN_PLAN.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {C.DOCS / 'GEN_PLAN.json'}")
            return 0
        if a.verify:
            ok, errs, _ = verify_freeze()
            print("T2b-register freeze verifies" if ok else "T2b-register freeze does NOT verify:\n  " + "\n  ".join(errs))
            need("torch" not in sys.modules, "--verify must not import torch", persist=False)
            return 0 if ok else 1
        if a.freeze:
            freeze(a.approval or "")
            return 0
        cfg = Cfg(seed=a.seed, train_variant=a.train_variant, cap_h=a.cap_h)
        return run(cfg, RealBackend())
    except Stop as e:
        print(str(e), flush=True)
        if cfg is not None and not (a.plan or a.verify or a.freeze):
            persist_stop(cfg.out, e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
