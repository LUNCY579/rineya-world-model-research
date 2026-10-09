"""RCWM-T2b prerequisite representation screen: GPU runner (train each arm with the T2 recipe, evaluate OFF/ON).

  python -B scripts/t2bp_run.py --plan          # CPU only, no torch, no model, no lease: generations, GPU time per arm
                                                # and per 4.5 h turn; writes docs/t2b/prereq/GEN_PLAN.json
  python -B scripts/t2bp_run.py                 # the GPU job: seed 0, arms D-int, S-trace, B-bits, trained on the
                                                # surface-diversity build (revision 3, C1), tag sd_s0
  python -B scripts/t2bp_run.py --tag sd_s0_ffill --arms ffill      # the R5 decode-compute control: ONLY after the
                                                                    # screen names S-trace (then t2bp_analysis.py
                                                                    # --filler-run runs/t2bp/sd_s0_ffill)
  python -B scripts/t2bp_run.py --with-filler   # alternative: the filler inside the sd_s0 run (decided before its first
                                                # turn; after DONE an arms mismatch is a stop, never a silent no-op)
  python -B scripts/t2bp_run.py --tag sd_s1 --seed 1 --arms <winner>   # seeds 1-2 of the arm the screen names

Revision 3 (operator decision C1): the screen trains ONLY the surface-diversity build (data/t2bp_train_sd_<arm>.jsonl:
the canonical rows with seeded paraphrases, field-name variants and input key orders from the training subsets); the
canonical-only build is not run (a non-sd configuration is refused before the lease). Tags get the prefix "sd_".

Recipe (T2, DECISIONS D-15/D-16, unchanged): vendored BWM-001 trainer (vendor_bwm/scripts/train_generic.py, called
exactly as scripts/run_t2.py calls it): LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down of frozen FP32
Titan-Sparse, WSD schedule, lr 1e-4, batch 16 (4 x 4 token-weighted microbatches), 190 steps, seed 0, duty-capped.
Every arm trains on the SAME 1,500 canonical rows in the same file order (so the trainer's seeded row order is
identical across arms) with the same per-row surface; only the arm's rendering differs (t2bp_common.py).

Evaluation (D-19 instrument style): one process, the adapter loaded unmerged with PEFT exactly as run_t2.py loads it
(runtime.load_arm('sparse') + PeftModel.from_pretrained); OFF = model.disable_adapter() (Titan-Sparse) and ON in the
same process; greedy FP32, rcwmt.generation.generate_texts (160-token cap, its batching):
  E1  the dev bank's 1,050 one-step rows (9 strata incl. 10/12-bit masks), in the arm's own frame (the canonical
      surface: in-surface), OFF and ON
  E2  the dev bank's 240 k=4 chains (30 parents x 8 plans): self-rollout in the arm's frame (canonical surface), the
      arm's after-only parser feeds each predicted state back (a broken chain stays broken), OFF and ON
  E3  held-out surface (C3): the same 1,050 one-step rows and golds in the COMBINED held-out surface (paraphrase 4 with
      its state sentence, field names 4, the reversed top-level key order, and the held-out event key orders rotating
      by the row's index within its stratum: t2bp_common.heldout_surface_of), adapter ON only, paired by row with E1.
      HS (the seven 8-bit strata) enters the screen's ranking. The adapter-OFF side and per-factor held-out renderings
      are not run: they do not fit the 2-turn planning budget (GEN_PLAN.json) and the rule does not use them.
A0 anchor (once per turn, before the first evaluation stage of that turn): adapter disabled, pilot TEST prerequisite
chain_m1 (224 chains, rcwmt.generation.chain) must reproduce runs/t2/t2_s0/gen_chain_m1_base.jsonl (sha pinned) with
identical states on >= 99% of chains, else the job stops (persisted) before any evaluation output of that turn.

Pairing (D-22 style, one lease 'rcwm-t1:t2b-prereq', 4.5 h cap from acquisition): arms run in order; an arm's
training starts only if 1.25 x (its remaining training + evaluation + model load [+ the anchor]) fits the remaining
turn (and never within 0.5 h of the cap); each evaluation stage starts only if 1.25 x its estimate fits. Otherwise the
arm (and every later arm) is deferred to the next turn: exit 5, and the next call resumes from the recorded stages.
Estimates are corrected by the observed/planned ratio of the stages already run. Planning rate: base Titan has never
seen the S-trace, B-bits or filler formats, so their adapter-OFF generations are planned at the 160-token cap (the
worst case; D-int OFF is T2's native frame); ON generations at the gold answer length.

Before the lease: input hashes (training files and dev bank == their SOURCE records, anchor reference, vendored trainer),
the frozen screen rule (docs/t2b/prereq/SCREEN_RULE.json: every frozen code and data file unchanged; refuse, not
persisted, otherwise), every training prompt == tok.apply_chat_template(messages(row)) (so training and evaluation frames
are identical), the one-waiter rule (R14: refuse if another RCWM job waits for or holds the GPU lease; BWM jobs are
fine), and a one-runner lock on the output directory. A resumed run refuses (not persisted) if any frozen code file
differs from the hashes recorded at its first turn. A completed run (DONE.json) refuses a call with other arms. Outputs are create-only under runs/t2bp/<tag>/. A stage is DONE only if its files exist
AND provenance.json records them with matching sha256; anything else a stage wrote is PARTIAL: moved to
runs/t2bp/<tag>/_orphans/ (never deleted) and redone in full (OFF and ON together; a partial training directory is
retrained from scratch). A recorded output whose hash changes stops the job.

Exit codes: 0 complete (DONE.json) or already complete; 3 deterministic stop (never retried; persisted as STOPPED.json
when it concerns the run); 5 deferred by the turn cap (resumable); 1 crash (an exception; the lease is released).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_gen as G2  # noqa: E402  (torch-free at import: D-22 stage rule, registered runtime, fingerprints)
import t2bp_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

# ---------------------------------------------------------------- constants
LEASE_LABEL = "rcwm-t1:t2b-prereq"
OUT_ROOT = ROOT / "runs" / "t2bp"
RCWM_LABEL_PREFIX = "rcwm-t1:"
CAP_H = 4.5
STEPS, LR = 190, 1e-4
STAGES = ("T", "E1", "E2", "E3")      # revision 3: every run is the surface-diversity build with the held-out stage E3
STAGES_SD = STAGES
E3_SIDES = ("on",)                   # the held-out-surface stage generates adapter ON only (the rule reads ON)
FACTOR_ROWS = 750                    # per-factor held-out renderings (descriptive, NOT run): 3 factors x the 8-bit rows
NATIVE_OFF_ARMS = ("dint",)          # base Titan's own frame (T2's prerequisite prompt): OFF planned at the gold length
GEN_CAP_TOKENS = 160                 # rcwmt.generation.max_tokens
FP = dict(G2.FP)
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
VRAM_EMPTY_MIB = 512.0
ANCHOR_REF = ROOT / "runs" / "t2" / "t2_s0" / "gen_chain_m1_base.jsonl"
ANCHOR_REF_SHA = "7ebc82889e271ced7599af7ff0c2568aa795a9e76888101da8cf454824252f9a"
ANCHOR_MIN_IDENTICAL = 0.99
VENDOR = {"vendor_bwm/scripts/train_generic.py": "816d2f562946177397a5f186e9fcadd7febd2b49c894afc9abafe9474665ad61",
          "vendor_bwm/bwmn/lm.py": "3d3f8d00b84af39dabd46b415ae559754066bdcf3c64de1127b138a27da72b57"}
ADAPTER_NAME = "t2bp"
# timing model (no model is loaded to make it): T2's three 190-step trainings took 1859 / 2191 / 1656 s (runs/t2/*/log.txt)
# on rows of 152.8 tokens; training time is scaled by the arm's tokens per row. Generation: PREREG-2's provenance gives
# 6.52 gens/s for prerequisite chain_m1 (S4: 9,225 gens / 1,414.6 s; gold answers 2.57 tokens) and 3.80 gens/s for
# ledger chain_one (S1: 9,588 / 2,525.8 s; 20.14 tokens) -> s/gen = a + b * (answer tokens + 1). Planning = observed x
# 2.7/3.8 (the registered planning rate of PREREG-2/3 relative to its observed rate).
T2_TRAIN_S_MAX = 2191.3
T2_TRAIN_S_MEAN = (1859.2 + 2191.3 + 1656.1) / 3
P_M1_S_PER_GEN = 1414.6 / 9225
L_ONE_S_PER_GEN = 2525.8 / 9588
PLANNING_FACTOR = 3.80 / 2.7
LOAD_S = 120.0


class Stop(Exception):
    """A deterministic stop (exit 3, never retried). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


class Deferred(Exception):
    pass


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[t2bp] {msg}", persist)


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- configuration
@dataclass
class Cfg:
    root: Path = ROOT
    tag: str = C.SCREEN_TAG
    seed: int = 0
    arms: tuple = C.SCREEN_ARMS
    data_variant: str = C.DATA_VARIANT            # "sd" (revision 3, C1); anything else is refused by run()
    out: Path | None = None
    train_files: dict | None = None
    train_source: Path | None = None
    dev: Path = C.DEV
    dev_source: Path = C.DEV_SOURCE
    steps: int = STEPS
    lr: float = LR
    cap_h: float = CAP_H
    planning: bool = True                         # estimates at the planning rate (conservative)
    anchor_chains: list | None = None             # None: the pilot TEST prerequisite chains (224)
    anchor_ref: Path = ANCHOR_REF
    anchor_ref_sha: str | None = ANCHOR_REF_SHA
    dev_subset: tuple | None = None               # (n one-step rows, n chain parents): tiny tests only
    check_vendor: bool = True
    rule_record: Path | None = C.DOCS / C.SCREEN_RULE_JSON   # the frozen screen rule (tests pass a temporary record)

    def __post_init__(self):
        if self.out is None:
            self.out = OUT_ROOT / self.tag
        if self.train_files is None:
            self.train_files = dict(C.TRAIN_FILES_SD if self.data_variant == "sd" else C.TRAIN_FILES)
        if self.train_source is None:
            self.train_source = C.TRAIN_SOURCE_SD if self.data_variant == "sd" else C.TRAIN_SOURCE

    @property
    def stages(self) -> tuple:
        return STAGES


def pilot_anchor_chains() -> list:
    return G2.load_bank_chains(G2.PILOT_PATH, G2.PILOT_SHA, "P", "test")


def load_dev(cfg: Cfg) -> tuple[list, list]:
    rows, parents = C.load_dev(cfg.dev)
    if cfg.dev_subset:
        n1, n2 = cfg.dev_subset
        keep = []
        for s in C.DEV_STRATA:
            keep += [r for r in rows if r["stratum"] == s][:n1]
        rows, parents = keep, parents[:n2]
    return rows, C.dev_chains(parents)


# ---------------------------------------------------------------- estimates (no torch)
def s_per_gen(target_tokens: float, planning: bool) -> float:
    b = (L_ONE_S_PER_GEN - P_M1_S_PER_GEN) / (20.144 - 2.57)
    a = P_M1_S_PER_GEN - b * (2.57 + 1)
    return (a + b * (target_tokens + 1)) * (PLANNING_FACTOR if planning else 1.0)


def off_tokens(arm: str, gold_tokens: float, planning: bool) -> float:
    """Adapter-OFF answer length: the 160-token cap for the non-native formats at the planning rate (worst case)."""
    return float(GEN_CAP_TOKENS) if (planning and arm not in NATIVE_OFF_ARMS) else gold_tokens


def estimates(cfg: Cfg, n_rows: int | None = None, n_steps: int | None = None, n_anchor: int | None = None) -> dict:
    src = _read_json(cfg.train_source) or _read_json(C.TRAIN_SOURCE) or {}
    ts = src.get("token_stats") or {}
    t2_tok = (ts.get("t2_train_reference") or {}).get("mean_row_tokens_incl_eos", 152.76)
    n_rows = n_rows if n_rows is not None else sum(C.DEV_N.values())
    n_steps = n_steps if n_steps is not None else C.CHAIN_PARENTS * C.CHAIN_K * C.CHAIN_H
    n_anchor = n_anchor if n_anchor is not None else 224 * C.CHAIN_H
    out = {"anchor": n_anchor * s_per_gen(2.57, cfg.planning), "load": LOAD_S, "arms": {}}
    t2_s = T2_TRAIN_S_MAX if cfg.planning else T2_TRAIN_S_MEAN
    for a in C.ARMS:
        tr = (ts.get("train") or {}).get(a, {}).get("mean_row_tokens_incl_eos", 260.0)
        d1 = (ts.get("dev_onestep") or {}).get(a, {}).get("mean_target_tokens", 31.0)
        d2 = (ts.get("dev_chain_steps") or {}).get(a, {}).get("mean_target_tokens", 31.0)
        e1 = n_rows * (s_per_gen(off_tokens(a, d1, cfg.planning), cfg.planning) + s_per_gen(d1, cfg.planning))
        out["arms"][a] = {"T": LOAD_S + t2_s * (tr / t2_tok) * (cfg.steps / STEPS),
                          "E1": e1,
                          "E2": n_steps * (s_per_gen(off_tokens(a, d2, cfg.planning), cfg.planning) + s_per_gen(d2, cfg.planning)),
                          "E3": n_rows * s_per_gen(d1, cfg.planning),                     # ON only
                          "gens": {"E1": 2 * n_rows, "E2": 2 * n_steps, "E3": n_rows},
                          "per_factor_not_run": 3 * FACTOR_ROWS * s_per_gen(d1, cfg.planning)}
    return out


def simulate_turns(cfg: Cfg, arms, est: dict | None = None) -> list:
    est = est or estimates(cfg)
    cap = cfg.cap_h * 3600
    todo = {a: list(cfg.stages) for a in arms}
    turns = []
    while any(todo.values()):
        t, ran, anchored = 0.0, [], False
        for a in arms:
            if not todo[a]:
                continue
            pair = sum(est["arms"][a][s] for s in todo[a]) + (LOAD_S if {"E1", "E2", "E3"} & set(todo[a]) else 0.0) + \
                (0.0 if anchored else est["anchor"])
            if "T" in todo[a]:
                if not G2.can_start(t, pair, 0.0, cap):
                    break
                t += est["arms"][a]["T"]
                todo[a].remove("T")
                ran.append(f"{a}:T")
            t += LOAD_S
            if not anchored:
                t += est["anchor"]
                anchored = True
                ran.append("A0")
            stop = False
            for s in ("E1", "E2", "E3"):
                if s in todo[a]:
                    if not G2.can_start(t, est["arms"][a][s], 0.0, cap):
                        stop = True
                        break
                    t += est["arms"][a][s]
                    todo[a].remove(s)
                    ran.append(f"{a}:{s}")
            if stop:
                break
        if not [r for r in ran if r != "A0"]:
            raise RuntimeError("no stage fits an empty turn")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(verbose: bool = True) -> dict:
    """Revision 3: every projection is the surface-diversity build (C1) with the held-out stage E3 (ON only)."""
    res = {"cap_h": CAP_H, "load_s_each": LOAD_S, "data_variant": C.DATA_VARIANT,
           "train_source": C.TRAIN_SOURCE_SD.name, "stages_per_arm": list(STAGES),
           "stage_rule": f"a stage (an arm's training: its whole remaining train+eval pair) starts only if "
                         f"{G2.STAGE_SAFETY} x its estimate fits the turn and never within {G2.STAGE_MARGIN_H} h of the "
                         "cap; otherwise the arm (and every later arm) is deferred to the next turn (exit 5) and resumed",
           "timing_model": {
               "train_s": "T2 190-step training time (planning: max 2191 s; observed: mean 1902 s) x arm tokens/row / 152.8",
               "s_per_gen": "a + b*(answer tokens+1) fitted to PREREG-2 S4 (P chain_m1 6.52/s at 2.57 tokens) and S1 "
                            "(ledger chain_one 3.80/s at 20.14 tokens); planning = observed x 3.8/2.7",
               "off_answer_tokens": (f"planning: the {GEN_CAP_TOKENS}-token cap for the adapter-OFF side of every arm "
                                     f"except {list(NATIVE_OFF_ARMS)} (base Titan has never seen those formats; worst "
                                     "case); observed: the gold answer length (best case). The true screen time lies "
                                     "between the two."),
               "E3": "the held-out-surface stage: the 1,050 one-step rows, adapter ON only, at the gold answer length",
               "a_s": round(P_M1_S_PER_GEN - (L_ONE_S_PER_GEN - P_M1_S_PER_GEN) / (20.144 - 2.57) * 3.57, 4),
               "b_s_per_token": round((L_ONE_S_PER_GEN - P_M1_S_PER_GEN) / (20.144 - 2.57), 5)}}
    for mode in ("planning", "observed"):
        cfg = Cfg(planning=(mode == "planning"))
        est = estimates(cfg)
        res[mode] = {"anchor_s": round(est["anchor"], 1), "arms": {
            a: {"train_s": round(est["arms"][a]["T"], 1), "E1_s": round(est["arms"][a]["E1"], 1),
                "E2_s": round(est["arms"][a]["E2"], 1), "E3_s": round(est["arms"][a]["E3"], 1),
                "generations": est["arms"][a]["gens"],
                "pair_h": round((est["arms"][a]["T"] + LOAD_S + sum(est["arms"][a][s] for s in STAGES[1:])) / 3600, 2)}
            for a in C.ARMS},
            "turns_screen": simulate_turns(cfg, C.SCREEN_ARMS, est),
            "turns_filler_control_run_if_strace_wins": simulate_turns(cfg, C.CONTROL_ARMS, est),
            "turns_screen_plus_filler_in_s0": simulate_turns(cfg, C.SCREEN_ARMS + C.CONTROL_ARMS, est)}
        # per-factor held-out renderings (C3: descriptive only if they fit): what the screen would need with them
        est_pf = json.loads(json.dumps(est))
        for a in C.ARMS:
            est_pf["arms"][a]["E3"] += est_pf["arms"][a]["per_factor_not_run"]
        res[mode]["turns_screen_if_per_factor_heldout_added"] = simulate_turns(cfg, C.SCREEN_ARMS, est_pf)
        res[mode]["per_factor_heldout_extra_h"] = {a: round(est["arms"][a]["per_factor_not_run"] / 3600, 2) for a in C.ARMS}
        for k in ("turns_screen", "turns_filler_control_run_if_strace_wins", "turns_screen_plus_filler_in_s0",
                  "turns_screen_if_per_factor_heldout_added"):
            res[mode][k.replace("turns", "gpu_h")] = round(sum(t["projected_h"] for t in res[mode][k]), 2)
    n_turns_pf = len(res["planning"]["turns_screen_if_per_factor_heldout_added"])
    res["per_factor_heldout_renderings"] = {
        "run": False,
        "reason": (f"adding the three per-factor held-out renderings (paraphrase only, field names only, key order only; "
                   f"{FACTOR_ROWS} 8-bit rows each, ON) would take the screen from "
                   f"{len(res['planning']['turns_screen'])} to {n_turns_pf} turns at the planning rate; C3 makes them "
                   "descriptive only if they fit the budget, so they are omitted")}
    res["generations_total_screen"] = sum(res["planning"]["arms"][a]["generations"][s] for a in C.SCREEN_ARMS
                                          for s in STAGES[1:]) + 224 * 4 * len(res["planning"]["turns_screen"])
    res["generations_note"] = ("E1 + E2 per screen arm (OFF + ON) + E3 (ON, held-out surface) + one 224-chain x 4-step "
                               "anchor per turn (planning turn count)")
    res["seeds_1_2_per_seed_h_planning"] = {a: res["planning"]["arms"][a]["pair_h"] for a in C.ARMS}
    if verbose:
        for mode in ("planning", "observed"):
            r = res[mode]
            print(f"[{mode}] anchor {r['anchor_s'] / 60:.1f} min")
            for a in C.ARMS:
                x = r["arms"][a]
                print(f"  {C.ARM_LABEL[a]:<9} train {x['train_s'] / 60:5.1f} min  E1 {x['E1_s'] / 60:5.1f} min  "
                      f"E2 {x['E2_s'] / 60:5.1f} min  E3 {x['E3_s'] / 60:5.1f} min  pair {x['pair_h']:.2f} h  "
                      f"gens {x['generations']}")
            for k in ("turns_screen", "turns_filler_control_run_if_strace_wins", "turns_screen_plus_filler_in_s0",
                      "turns_screen_if_per_factor_heldout_added"):
                print(f"  {k}: " + "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(r[k]))
                      + f"  (total {r[k.replace('turns', 'gpu_h')]} h)")
        print(res["per_factor_heldout_renderings"]["reason"])
    return res


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(arm: str, st: str, out: Path) -> list:
    d = Path(out) / arm
    if st == "T":
        return [d / "train" / "adapter" / "adapter_model.safetensors", d / "train" / "adapter" / "adapter_config.json"]
    if st == "E1":
        return [d / "onestep_off.jsonl", d / "onestep_on.jsonl"]
    if st == "E2":
        return [d / "chains_off.jsonl", d / "chains_on.jsonl"]
    if st == "E3":
        return [d / f"onestep_heldout_{side}.jsonl" for side in E3_SIDES]
    raise KeyError(st)


def stage_owned(arm: str, st: str, out: Path) -> list:
    if st == "T":
        return [Path(out) / arm / "train"]
    fs = stage_required(arm, st, out)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path) -> str:
    return str(Path(p).relative_to(Path(out))).replace("\\", "/")


def stage_states(prov: dict, out: Path, arms, stages=STAGES) -> dict:
    res = {}
    for a in arms:
        res[a] = {}
        for st in stages:
            req = stage_required(a, st, out)
            rec = ((prov.get("stages") or {}).get(a) or {}).get(st)
            if all(p.exists() for p in req) and rec:
                files = rec.get("files") or {}
                for p in req:
                    r = rel_out(p, out)
                    need(r in files, f"{a}/{st}: {r} not recorded in provenance")
                    need(C.sha256(p) == files[r], f"{a}/{st}: {r} changed after it was recorded (sha256)")
                res[a][st] = "done"
            elif any(p.exists() for p in stage_owned(a, st, out)):
                res[a][st] = "partial"
            else:
                res[a][st] = "todo"
        if res[a]["T"] != "done" and any(res[a][s] == "done" for s in stages if s != "T"):
            raise Stop(f"[t2bp] {a}: evaluation recorded without a recorded adapter")
    return res


def orphan(arm: str, st: str, out: Path) -> Path:
    dst = Path(out) / "_orphans" / f"{arm}_{st}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    dst.mkdir(parents=True, exist_ok=False)
    for p in stage_owned(arm, st, out):
        if p.exists():
            shutil.move(str(p), str(dst / p.name))
    print(f"[t2bp] {arm}/{st} was partly written; moved to {dst}", flush=True)
    return dst


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Stop(f"[t2bp] {path} exists (create-only)")
    tmp = _partial(path)
    with tmp.open("x", encoding="utf-8", newline="\n") as f:       # LF only (DECISIONS I-30)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- run lock, one-waiter rule
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
        if alive:
            raise Stop(f"[t2bp] another t2bp_run.py holds the run lock {rec}", persist=False)
        print(f"[t2bp] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink()
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
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for the GPU lease (live waiting marker of another
    process) or holds it (live lock holder with an rcwm-t1:* label). BWM jobs are fine: this job then waits as usual."""
    w = _read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = _read_json(backend.lock_path())
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
    """The GPU path: rcwmt.runtime (lease, numerics, loads), the vendored trainer, PEFT as run_t2.py uses it."""
    device = "cuda"

    def __init__(self):
        import torch  # noqa: F401
        from rcwmt import runtime
        self.rt = runtime

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 T2b prerequisite representation screen (train + eval OFF/ON per arm)")

    def waiting_marker_path(self) -> Path:
        return Path(self.rt.WAITING["rcwm"])

    def lock_path(self) -> Path:
        return Path(self.rt.LOCK_PATH)

    def lease_acquired_unix(self) -> tuple[float, str]:
        try:
            rec = json.loads(Path(self.rt.LOCK_PATH).read_text(encoding="utf-8"))
            if rec.get("label") == LEASE_LABEL and rec.get("pid") == os.getpid():
                return float(rec["created"]), "lock record"
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        return time.time(), "lease entry (lock record unavailable)"

    def thermal_wait(self) -> None:
        if THERMAL_FLAG.exists():
            print("[t2bp] thermal pause flag set before a model load; waiting", flush=True)
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

    def load_eval(self, adapter_dir: Path):
        from peft import PeftModel
        import peft
        base, p = self.rt.load_arm("sparse")
        model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=ADAPTER_NAME, is_trainable=False).eval()
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


@contextmanager
def side_ctx(model, side: str):
    if side == "off":
        with model.disable_adapter():
            yield
        return
    model.set_adapter(ADAPTER_NAME)
    yield


def gen_texts(model, tok, message_lists: list) -> list:
    from rcwmt import generation
    return generation.generate_texts(model, tok, message_lists, generation.max_tokens(C.FAM))


def t2bp_chain(model, tok, arm: str, chains: list) -> list:
    """Self-rollout in the arm's frame: rcwmt.generation.chain's loop for one family; each step's prompt carries the
    previous PREDICTED state (rendered in the arm's representation), parsed by the arm's after-only parser (None once a
    chain breaks). For D-int this is generation.chain(..., 'chain_m1') on prerequisite chains (tested)."""
    states = [[c["s0"]] for c in chains]
    texts = [[] for _ in chains]
    H = max(len(c["events"]) for c in chains)
    for t in range(H):
        live = [i for i, c in enumerate(chains) if t < len(c["events"]) and states[i][-1] is not None]
        if live:
            msgs = [C.messages(arm, states[i][-1], chains[i]["events"][t], chains[i]["width"]) for i in live]
            txt = gen_texts(model, tok, msgs)
            for i, s in zip(live, txt):
                texts[i].append(s)
                states[i].append(C.parse_after(arm, s, chains[i]["width"]))
        for i, c in enumerate(chains):
            if t < len(c["events"]) and states[i][-1] is None and len(states[i]) == t + 1:
                states[i].append(None)
    return [{"states": s, "texts": x} for s, x in zip(states, texts)]


def chain_rows(chains, results) -> list:
    return [{"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]], "texts": r["texts"]}
            for c, r in zip(chains, results)]


def run_anchor(model, tok, chains: list, ref_path: Path) -> dict:
    from rcwmt import generation
    t0 = time.time()
    with side_ctx(model, "off"):
        rr = generation.chain(model, tok, chains, "chain_m1")
    ref = {}
    for line in Path(ref_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ref[row["cand_id"]] = row
    st = sum([None if x is None else env.dumps(x) for x in r["states"]] == (ref.get(c["cand_id"]) or {}).get("states")
             for c, r in zip(chains, rr))
    tx = sum(r["texts"] == (ref.get(c["cand_id"]) or {}).get("texts") for c, r in zip(chains, rr))
    return {"n": len(chains), "identical_states": st / len(chains), "identical_texts": tx / len(chains),
            "pass": st / len(chains) >= ANCHOR_MIN_IDENTICAL, "seconds": round(time.time() - t0, 1),
            "generations": sum(len(r["texts"]) for r in rr), "pid": os.getpid(), "unix": time.time()}


# ---------------------------------------------------------------- input checks
def check_inputs(cfg: Cfg) -> dict:
    src = _read_json(cfg.train_source)
    need(src is not None, f"{cfg.train_source} missing or unreadable", persist=False)
    shas = {}
    for a in cfg.arms:
        p = cfg.train_files[a]
        need(p.is_file(), f"{p} missing", persist=False)
        shas[p.name] = C.sha256(p)
        need(shas[p.name] == (src.get("files") or {}).get(p.name), f"{p.name} differs from its SOURCE record")
    if cfg.data_variant == "sd" and src.get("canonical_dint_sha256") is not None:
        # the sd rows are the canonical rows: the canonical row source must be the file the sd build read
        need(C.sha256(C.TRAIN_FILES["dint"]) == src["canonical_dint_sha256"],
             "data/t2bp_train_dint.jsonl (the canonical row source) differs from the sd SOURCE's record")
    dsrc = _read_json(cfg.dev_source) or {}
    shas[Path(cfg.dev).name] = C.sha256(cfg.dev)
    need(shas[Path(cfg.dev).name] == dsrc.get("file_sha256"), "dev bank differs from its SOURCE record")
    if cfg.anchor_ref_sha:
        need(C.sha256(cfg.anchor_ref) == cfg.anchor_ref_sha, "anchor reference changed (sha256)")
    if cfg.check_vendor:
        for rel, h in VENDOR.items():
            need(C.sha256(ROOT / rel) == h, f"{rel} differs from the T2 trainer (sha256)")
    return shas


def frozen_files(cfg: Cfg) -> dict:
    """{repo-relative name: path} of the frozen inputs this run uses: the code, the dev bank, the arms' surface-diversity
    training files and their SOURCE, and the canonical row source (the canonical D-int file and SOURCE)."""
    files = {rel: ROOT / rel for rel in C.FROZEN_CODE}
    files["data/" + C.DEV.name] = Path(cfg.dev)
    files["data/" + C.DEV_SOURCE.name] = Path(cfg.dev_source)
    files["data/" + C.TRAIN_FILES["dint"].name] = C.TRAIN_FILES["dint"]
    files["data/" + C.TRAIN_SOURCE.name] = C.TRAIN_SOURCE
    for a in cfg.arms:
        files["data/" + C.TRAIN_FILES_SD[a].name] = Path(cfg.train_files[a])
    files["data/" + C.TRAIN_SOURCE_SD.name] = Path(cfg.train_source)
    return files


def check_frozen(cfg: Cfg) -> None:
    if cfg.rule_record is None:                    # tiny tests of other mechanics only
        return
    probs = C.frozen_problems(C.read_rule_record(cfg.rule_record), frozen_files(cfg))
    need(not probs, "the frozen screen rule does not verify (run nothing until it does): " + "; ".join(probs[:6]),
         persist=False)


def code_hashes() -> dict:
    return {r: C.sha256(ROOT / r) for r in C.FROZEN_CODE}


def check_prompts(cfg: Cfg, tok) -> dict:
    """Training prompts == tok.apply_chat_template(messages(row)): training and evaluation frames are identical."""
    from rcwmt.capture import render
    res = {}
    for a in cfg.arms:
        bad = 0
        rows = C.load_jsonl(cfg.train_files[a])
        for r in rows:
            msgs = C.messages(a, r["before"], r["event"], r["width"], r.get("surface"))
            bad += int(render(tok, msgs) != r["prompt"])
        need(bad == 0, f"{a}: {bad} training prompts differ from the chat template of their messages", persist=False)
        res[a] = len(rows)
    return res


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = Path(cfg.out)
    need(cfg.data_variant == C.DATA_VARIANT, f"data variant {cfg.data_variant!r}: revision 3 (operator decision C1) trains "
         f"only the surface-diversity build ({C.DATA_VARIANT!r}); the canonical-only variant is not run", persist=False)
    for a in cfg.arms:
        need(a in C.ARMS, f"unknown arm {a}", persist=False)
    if (out / "DONE.json").exists():
        done = _read_json(out / "DONE.json") or {}
        need(done.get("arms") == list(cfg.arms), f"{out / 'DONE.json'} records arms {done.get('arms')}, this call asks for "
             f"{list(cfg.arms)}: a completed run is never extended (the R5 filler control runs as its own run: --tag "
             f"{C.FILLER_TAG} --arms ffill)", persist=False)
        print(f"[t2bp] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; investigate, log in DECISIONS.md "
                   "and move it aside before any rerun", persist=False)
    shas = check_inputs(cfg)
    check_frozen(cfg)
    dev_rows, chains = load_dev(cfg)
    anchor_chains = cfg.anchor_chains if cfg.anchor_chains is not None else pilot_anchor_chains()
    tok = backend.tokenizer()
    n_checked = check_prompts(cfg, tok)
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[t2bp] {ow}: one-waiter rule (R14). Start only when no other RCWM GPU job runs or waits", persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, shas, dev_rows, chains, anchor_chains, tok, n_checked)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg, backend, out, shas, dev_rows, chains, anchor_chains, tok, n_checked) -> int:
    prov_path = out / "provenance.json"
    prov = _read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
        need(prov.get("arms") == list(cfg.arms) and prov.get("seed") == cfg.seed and prov.get("steps") == cfg.steps
             and prov.get("data_variant") == cfg.data_variant, "arms/seed/steps/data variant differ from the recorded run",
             persist=False)
        need(prov.get("data_sha256") == shas, "input data differ from the recorded run")
        changed = sorted(r for r, h in code_hashes().items() if (prov.get("code_sha256") or {}).get(r) != h)
        need(not changed, f"code changed since the run's first turn ({changed}): a resumed run uses the code it started "
             "with; restore it, or move the run aside and log it in DECISIONS.md", persist=False)
    else:
        prov = {"tag": cfg.tag, "seed": cfg.seed, "steps": cfg.steps, "lr": cfg.lr, "arms": list(cfg.arms),
                "data_variant": cfg.data_variant, "stages_per_arm": list(cfg.stages), "data_sha256": shas,
                "dev_subset": cfg.dev_subset, "n_dev_rows": len(dev_rows), "n_chains": len(chains),
                "lease_label": LEASE_LABEL, "code_sha256": code_hashes(),
                "screen_rule_sha256": C.sha256(cfg.rule_record) if cfg.rule_record and Path(cfg.rule_record).is_file() else None,
                "stages": {}, "turns": []}

    def save():
        tmp = _partial(prov_path)
        tmp.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, prov_path)

    STG = cfg.stages
    states = stage_states(prov, out, cfg.arms, STG)
    for a in cfg.arms:
        for st in STG:
            if states[a][st] == "partial":
                orphan(a, st, out)
                (prov["stages"].get(a) or {}).pop(st, None)
                states[a][st] = "todo"
                if st == "T":                      # a retrained adapter invalidates nothing recorded (E* need T)
                    for s2 in STG[1:]:
                        need(states[a][s2] != "done", f"{a}: evaluation recorded for an adapter being retrained")

    def finalize() -> None:
        st2 = stage_states(prov, out, cfg.arms, STG)
        need(all(v == "done" for a in cfg.arms for v in st2[a].values()), "finalize: not every stage recorded and unchanged")
        with (out / "DONE.json").open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"finished_unix": time.time(), "arms": list(cfg.arms),
                                "files": {a: {s: prov["stages"][a][s]["files"] for s in STG} for a in cfg.arms},
                                "provenance_sha256": C.sha256(prov_path)}, indent=1) + "\n")
        print("[t2bp] all arms complete; DONE.json written", flush=True)

    todo = {a: [s for s in STG if states[a][s] != "done"] for a in cfg.arms}
    print(f"[t2bp] stages to run: { {a: v for a, v in todo.items() if v} }", flush=True)
    if not any(todo.values()):
        save()
        finalize()
        return 0
    est = estimates(cfg, n_rows=len(dev_rows), n_steps=len(chains) * C.CHAIN_H, n_anchor=len(anchor_chains) * C.CHAIN_H)
    corr = {"T": 1.0, "E": 1.0}
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

        def record(a: str, st: str, extra: dict) -> None:
            prov["stages"].setdefault(a, {})[st] = {"turn": len(prov["turns"]) - 1, **extra, "finished_unix": time.time(),
                                                    "files": {rel_out(p, out): C.sha256(p) for p in stage_required(a, st, out)}}
            turn["stages_run"].append(f"{a}:{st}")
            save()

        try:
            for ai, a in enumerate(cfg.arms):
                if not todo[a]:
                    continue
                e = est["arms"][a]
                ev_todo = [s for s in STG[1:] if s in todo[a]]
                pair = (corr["T"] * e["T"] if "T" in todo[a] else 0.0) + sum(corr["E"] * e[s] for s in ev_todo) + \
                    (LOAD_S if ev_todo else 0.0) + (0.0 if anchored else corr["E"] * est["anchor"])
                if not fits(pair):
                    raise Deferred(f"{a} (pair estimate {pair / 3600:.2f} h)")
                if "T" in todo[a]:
                    alloc = backend.vram_allocated_mib()
                    need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated before training {a}", persist=False)
                    backend.wait()
                    t0 = time.time()
                    tdir = out / a / "train"
                    tdir.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[t2bp] {a}: training {cfg.steps} steps, seed {cfg.seed} on {cfg.train_files[a].name}", flush=True)
                    backend.train(cfg.train_files[a], tdir, cfg.steps, cfg.seed, cfg.lr)
                    sec = time.time() - t0
                    need(all(p.exists() for p in stage_required(a, "T", out)), f"{a}: no adapter after training")
                    corr["T"] = max(0.5, sec / e["T"])
                    record(a, "T", {"seconds": round(sec, 1), "planned_s": round(e["T"], 1), "rows": str(cfg.train_files[a].name)})
                    print(f"[t2bp] {a}: trained in {sec:.0f}s (planned {e['T']:.0f}s)", flush=True)
                if not ev_todo:
                    continue
                alloc = backend.vram_allocated_mib()
                need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated before the evaluation load", persist=False)
                backend.wait()
                adapter_dir = out / a / "train" / "adapter"
                ad_sha = {p.name: C.sha256(p) for p in stage_required(a, "T", out)}
                need(ad_sha == {Path(k).name: v for k, v in prov["stages"][a]["T"]["files"].items()}, f"{a}: adapter changed")
                model, fp, versions = backend.load_eval(adapter_dir)
                try:
                    backend.check_fingerprint("sparse", fp)
                    turn.setdefault("loads", []).append({"arm": a, "fingerprint": fp, "versions": versions,
                                                         "runtime": backend.runtime_info()})
                    rp = runtime_problem(turn["loads"][-1]["runtime"], versions)
                    if rp:
                        save()
                        raise Stop(f"[t2bp] {rp}: restore the registered environment (nothing evaluated)")
                    save()
                    if not anchored:
                        res = run_anchor(model, tok, anchor_chains, cfg.anchor_ref)
                        turn["anchor"] = res
                        save()
                        if not res["pass"]:
                            raise Stop(f"[t2bp] ANCHOR FAILED (identical states {res['identical_states']:.3f} < "
                                       f"{ANCHOR_MIN_IDENTICAL}): the instrument does not reproduce T2's chain_m1 OFF file")
                        anchored = True
                    for st in ev_todo:
                        if not fits(corr["E"] * e[st]):
                            raise Deferred(f"{a}:{st}")
                        t0 = time.time()
                        sides = {}
                        if st in ("E1", "E3"):
                            if st == "E3":       # the combined held-out surface, row by row (C3); ON only
                                msgs = [C.messages(a, r["before"], r["event"], r["width"], C.heldout_surface_of(r))
                                        for r in dev_rows]
                                run_sides = E3_SIDES
                            else:
                                msgs = [C.messages(a, r["before"], r["event"], r["width"]) for r in dev_rows]
                                run_sides = ("off", "on")
                            tx = {}
                            for side in run_sides:
                                ts = time.time()
                                with side_ctx(model, side):
                                    tx[side] = gen_texts(model, tok, msgs)
                                sides[side] = {"start_unix": ts, "end_unix": time.time()}
                            for side, p in zip(run_sides, stage_required(a, st, out)):
                                write_jsonl(p, [{"id": r["id"], "text": t} for r, t in zip(dev_rows, tx[side])])
                        else:
                            rc = {}
                            for side in ("off", "on"):
                                ts = time.time()
                                with side_ctx(model, side):
                                    rc[side] = t2bp_chain(model, tok, a, chains)
                                sides[side] = {"start_unix": ts, "end_unix": time.time()}
                            for side, p in zip(("off", "on"), stage_required(a, st, out)):
                                write_jsonl(p, chain_rows(chains, rc[side]))
                        sec = time.time() - t0
                        corr["E"] = max(0.5, sec / e[st])
                        record(a, st, {"seconds": round(sec, 1), "planned_s": round(e[st], 1), "generations_planned_max": e["gens"][st],
                                       "sides": sides, "pid": os.getpid(),
                                       **({"surface": C.HELDOUT_SURFACE, "sides_run": list(E3_SIDES)}
                                          if st == "E3" else {})})
                        print(f"[t2bp] {a}:{st} done in {sec:.0f}s (planned {e[st]:.0f}s)", flush=True)
                finally:
                    del model
                    backend.unload()
        except Deferred as d:
            rest = [f"{a}:{s}" for a in cfg.arms for s in STG if s in todo[a] and f"{a}:{s}" not in turn["stages_run"]]
            turn["deferred"] = rest
            print(f"[t2bp] cap: deferring {rest} to the next turn ({d})", flush=True)
        turn["released_unix"] = time.time()
        save()
    st_end = stage_states(prov, out, cfg.arms, STG)
    if all(v == "done" for a in cfg.arms for v in st_end[a].values()):
        finalize()
        return 0
    print("[t2bp] turn ended; the remaining stages resume next turn", flush=True)
    return 5


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "STOPPED.json").open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)}, indent=1) + "\n")
    except FileExistsError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="CPU only: no torch import, no model; projected GPU time per turn")
    ap.add_argument("--tag", default=C.SCREEN_TAG)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(C.SCREEN_ARMS))
    ap.add_argument("--with-filler", action="store_true", help="add the F-filler control arm")
    ap.add_argument("--data-variant", choices=(C.DATA_VARIANT,), default=C.DATA_VARIANT,
                    help="revision 3 (C1): only the surface-diversity build is trained")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        res = plan()
        need("torch" not in sys.modules, "--plan must not import torch", persist=False)
        res["torch_imported"] = "torch" in sys.modules
        res["train_source_sha256"] = C.sha256(C.TRAIN_SOURCE_SD) if C.TRAIN_SOURCE_SD.exists() else None
        res["code_sha256"] = {"scripts/t2bp_run.py": C.sha256(Path(__file__))}
        p = C.write_doc("GEN_PLAN.json", json.dumps(res, indent=1) + "\n")
        print(f"wrote {p}")
        return 0
    arms = tuple(x for x in a.arms.split(",") if x)
    if a.with_filler and "ffill" not in arms:
        arms = arms + ("ffill",)
    tag = a.tag if a.tag.startswith("sd_") else "sd_" + a.tag
    cfg = Cfg(tag=tag, seed=a.seed, arms=arms, data_variant=a.data_variant, cap_h=a.cap_h)
    try:
        return run(cfg, RealBackend())
    except Stop as e:
        print(str(e), flush=True)
        persist_stop(Path(cfg.out), e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
