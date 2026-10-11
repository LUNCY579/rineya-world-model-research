"""PREREG-5 runner, freeze tool and freeze verifier (the sealed confirmation of T2b-ledger; PREREG_5_DRAFT.md sections 3,
7 and 10). DRAFT KIT: nothing here runs on the GPU before PREREG-5 is frozen, and the freeze needs the dev gate's
PROCEED (runs/t2bl/dev/analysis/GATE.json) and BWM's co-sign with a salt commitment. Modelled on scripts/lbeam_run.py
(PREREG-LB), whose torch-free helpers it imports read-only (anchor, chain rows, episode rows, accounting, the pinned BWM
planner import, the BWM module hashes), and on scripts/t2bl_run.py.

  python -B scripts/t2bl_confirm_run.py --plan     # CPU only, no torch: generations, GPU time per stage and per 4.5 h turn
                                                   # -> docs/t2b/ledger/P5_GEN_PLAN.json (refused once frozen)
  python -B scripts/t2bl_confirm_run.py --verify   # CPU only: PREREG_5_FREEZE.json (+ PREREG_5_AMENDMENT_*) against files
  python -B scripts/t2bl_confirm_run.py --union-digest
                                                   # CPU only: rebuild the freshness union and print its digest (the
                                                   # operator publishes it before BWM's co-sign, which quotes it)
  python -B scripts/t2bl_confirm_run.py --freeze --approval "<who, when, DECISIONS D-<n>>" \
         --bwm-cosign "<BWM's co-sign: reference, this draft's full sha256, union:<digest>, commit:<sha256 of salt>>" \
         --bwm-salt "<the salt BWM reveals at the freeze>" [--changes <JSON change list>]
                                                   # CPU only: the pre-freeze checks, the freshness union rebuilt (data/
                                                   # unchanged while it is built; its digest must be the one BWM's co-sign
                                                   # quotes), the map bank and the planning draw MINTED from the seed rule
                                                   # (t2bl_confirm_bank.mint; sealed, never printed), every input hashed,
                                                   # PREREG_5.md + PREREG_5_FREEZE.json (.partial first, renamed in order;
                                                   # create-only). If PREREG-5's rules or decision code differ from those
                                                   # pinned at the dev gate's freeze, --changes must list every change
                                                   # (rendered into PREREG_5.md Appendix A), else the freeze refuses.
  python -B scripts/t2bl_confirm_run.py            # the registered GPU job (parts a and b)
  python -B scripts/t2bl_confirm_run.py --ret-mint # part (c), CPU, once: mint the 5c confirmation draw at run time
                                                   # (ret_gen.mint('confirm'), OS-entropy seed) -> runs/prereg5/RET_DRAW.json
  python -B scripts/t2bl_confirm_run.py --ret-record
                                                   # part (c), CPU: bind the retention run's registered result (or its stop)
                                                   # to PREREG-5 -> runs/prereg5/RETENTION.json (create-only)

Job (parts a and b). Before anything: the freeze verifies; an installed scripts/queue_hooks/90_prereg5.sh is the frozen
hook draft byte for byte; no other RCWM job holds or waits for the GPU lease (one-waiter rule, R14); a one-runner lock
(runs/prereg5/.run.lock). GPU stages run under ONE lease 'rcwm-t1:prereg5' (rcwmt.runtime.gpu_lease: alternation,
thermal flag, duty cycle, RAM floor), the 4.5 h turn cap from lock acquisition and the D-22 stage rule. One model per
turn: Titan-Sparse FP32 + T2 s0 (PEFT adapter 't2', PREREG-2's pins) + T2b-ledger seeds 0, 1, 2 (adapters 't2bl0',
't2bl1', 't2bl2'), all unmerged (D-28); OFF = model.disable_adapter() in the same process.
  A0  instrument identity, BEFORE any output and again at the start of every later turn that loads the model:
      (a) PREREG-2's anchor (lbeam_run.run_anchor: 208 pilot TEST ledger chain_one chains, OFF and T2 s0 against
          runs/t2/t2_s0 gen_chain_one_base.jsonl / gen_chain_one.jsonl): identical states on >= 99% per side;
      (b) adapter identity: per T2b-ledger seed, the first 50 ID_posted and the first 50 HS_words rows of the dev bank
          regenerated ON and compared with the dev screen's recorded answers (runs/t2bl/dev/s<k>/dev_on.jsonl, hashed in
          the freeze): identical parsed states on >= 97% per seed (it shows that adapter 't2bl<k>' is seed k's adapter
          and that loading four adapters changes nothing);
      (c) batch invariance of the planning F: 64 keys (init x vocabulary of the first 8 instances of the SPENT dev set
          dev2_primary, never a confirmation key) for T2b-ledger s0 in the canonical frame and in the held-out combined
          rendering, as ONE predict list vs one key per call: identical parsed states on >= 99% per arm.
      Any failure stops the job (persisted) before any output of that turn.
  M1  part (a): the sealed map bank (2,701 rows), sides OFF, T2s0, T2BL s0, s1, s2, in one process, written together
  P1  part (b) PRIMARY arm: plan_beam enum W2 E12 on the sealed planning draw, T2BL@HS (SurfaceAuditF('T2',
      'heldout_combined') with adapter 't2bl0')
  P2  part (b) PRIMARY comparator: the same, T2s0@HS (adapter 't2')
  P3  part (b): plan_beam enum W2 E12, T2BL@canon (LedgerF('T2') with adapter 't2bl0')
  P4  part (b): plan_beam enum W2 E12, T2s0@canon (LedgerF('T2') with adapter 't2')
  C1  CPU, never under the lease, only after A0 passed in an EARLIER invocation, in a process that has not initialised
      CUDA (checked and recorded): ORACLE and IGNORE_STATUS at enum W2 E12 (exact / rule F, no model)
Every F is lbeam_f.guarded(...) (the registered input-range guard); every plan_beam call uses lbeam_proposer.beam_kwargs
and the blind lbeam_problems.planner_instances. Each planning arm has its own cache (one stage per arm).

Outputs (create-only) under runs/prereg5/. A stage is DONE only if its files exist AND provenance.json records them with
matching sha256; anything else a stage wrote is PARTIAL: moved to runs/prereg5/_orphans/ (never deleted) and redone in
full. A recorded output whose hash changes stops the job. Attempts: every stage start is recorded (turn, pid, boot
time); a started stage that was never recorded FAILED if it ended in a recorded Python exception, or if its process
died without one while the machine kept running (same boot); an attempt cut off by a reboot (Kernel-Power 41, TDR) is
logged as interrupted and NOT counted; a stage that failed three times stops the job. The stage-start rule bounds a
planning stage by its ceiling generation count at the slowest measured planning rate (the registered rate before the
first planning stage; start_estimate_s).
Deferral: exit 5 at a stage boundary; two consecutive turns without a recorded stage stop the job. Parts: (a) needs A0 and M1; (b) needs A0, P1-P4 and C1. A
deterministic stop records which parts are verdict-eligible (their stages recorded, every recorded output unchanged,
the first A0 passed); the analysis renders the other parts 'not evaluable'.

Exit codes: 0 complete (DONE.json) or already complete; 3 deterministic stop, never retried (persisted as STOPPED.json
when it concerns the run; NOT persisted: "not frozen", "hook not the frozen draft", "another RCWM job holds or waits for
the lease", "run lock held", "STOPPED.json exists"); 5 deferred by the turn cap, or C1 still to run (resumes next call;
the hook sleeps >= 60 s first so BWM's yield loop sees the release); 1 crash (the hook retries once).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_CPU_FLAGS = ("--plan", "--verify", "--freeze", "--ret-mint", "--ret-record", "--union-digest")
if __name__ == "__main__" and any(f in sys.argv for f in _CPU_FLAGS):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""        # never touch the driver: empty string, NEVER "-1" (I-21)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
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
import lbeam_problems as P  # noqa: E402  (torch-free)
import lbeam_proposer as PR  # noqa: E402  (torch-free)
import lbeam_f as LF  # noqa: E402  (torch-free at import)
import lbeam_run as LR  # noqa: E402  (torch-free at import; imported read-only, FROZEN by PREREG-LB)
import t2bl_common as C  # noqa: E402
import t2bl_confirm_bank as CB  # noqa: E402

# ---------------------------------------------------------------- registered constants
LEASE_LABEL = "rcwm-t1:prereg5"
RCWM_LABEL_PREFIX = "rcwm-t1:"
OUT = ROOT / "runs" / "prereg5"
OUT_REL = "runs/prereg5"
DRAFT_REL = "PREREG_5_DRAFT.md"
PREREG_REL, FREEZE_REL = "PREREG_5.md", "PREREG_5_FREEZE.json"
AMEND_GLOB = "PREREG_5_AMENDMENT_*.json"
MAP_BANK_REL, MAP_SOURCE_REL = CB.MAP_REL, CB.MAP_SOURCE_REL
PLAN_DRAW_REL, PLAN_SOURCE_REL = CB.PLAN_REL, CB.PLAN_SOURCE_REL
SEALED = (MAP_BANK_REL, MAP_SOURCE_REL, PLAN_DRAW_REL, PLAN_SOURCE_REL)
HOOK_DRAFT_REL = "docs/t2b/ledger/queue_hook_prereg5.sh.draft"
HOOK_INSTALLED_REL = "scripts/queue_hooks/90_prereg5.sh"
DEV_RUN_REL = "runs/t2bl/dev"
GATE_REL = "runs/t2bl/dev/analysis/GATE.json"
GATE_RULE_REL = "docs/t2b/ledger/GATE_RULE.json"
T2S0_REL = "runs/t2/t2_s0/adapter"
SEEDS = (0, 1, 2)
T2BL_RELS = {s: f"runs/t2bl/dev/s{s}/train/adapter" for s in SEEDS}
DEV_ON_RELS = {s: f"runs/t2bl/dev/s{s}/dev_on.jsonl" for s in SEEDS}
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
ADAPTER_NAMES = {"t2s0": "t2", "t2bl0": "t2bl0", "t2bl1": "t2bl1", "t2bl2": "t2bl2"}
MAP_SIDES = ("off", "t2s0", "t2bl0", "t2bl1", "t2bl2")
SURFACE = "heldout_combined"
CANON = LF.CANONICAL
PLAN_ARMS = {"T2BL@HS": ("t2bl0", SURFACE), "T2s0@HS": ("t2s0", SURFACE), "T2BL@canon": ("t2bl0", CANON),
             "T2s0@canon": ("t2s0", CANON)}
STAGE_ARMS = {"P1": ("T2BL@HS",), "P2": ("T2s0@HS",), "P3": ("T2BL@canon",), "P4": ("T2s0@canon",)}
EVAL_SEED = 0
W, E = 2, 12
BWM_ROOT, BWM_PLANNER_REL, BWM_PLANNER_SHA = LR.BWM_ROOT, LR.BWM_PLANNER_REL, LR.BWM_PLANNER_SHA
BWM_INFO_RELS = LR.BWM_INFO_RELS
ANCHOR_REF_RELS = dict(LR.ANCHOR_REF_RELS)          # {"off": ..., "on": ...} (T2 s0 = the anchor's ON side)
DEV_SET_REL, DEV_SET_SHA = LR.DEV_REL, LR.DEV_SHA   # dev2_primary (spent): batch-invariance keys only
PINS = {f"{T2S0_REL}/adapter_model.safetensors": G2.ADAPTER_SHA, f"{T2S0_REL}/adapter_config.json": G2.ADAPTER_CONFIG_SHA,
        ANCHOR_REF_RELS["off"]: G2.ANCHOR_REF_SHA["gen_chain_one_base.jsonl"],
        ANCHOR_REF_RELS["on"]: G2.ANCHOR_REF_SHA["gen_chain_one.jsonl"], "data/bank_pilot.json": G2.PILOT_SHA,
        DEV_SET_REL: DEV_SET_SHA}
FP = dict(G2.FP)
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)
SEED_TAGS = (CB.TAG_MAP, CB.TAG_PLAN)
SALT_MIN_LEN = 16
STAGE_ORDER = ("A0", "M1", "P1", "P2", "P3", "P4", "C1")
GPU_STAGES = ("A0", "M1", "P1", "P2", "P3", "P4")
PART_STAGES = {"a": ("A0", "M1"), "b": ("A0", "P1", "P2", "P3", "P4", "C1")}
MAX_FAILED = 3
NO_PROGRESS_STOP = 2
ANCHOR_MIN_IDENTICAL = 0.99
IDENT_PER_STRATUM, IDENT_STRATA, IDENT_MIN = 50, ("ID_posted", "HS_words"), 0.97
BI_N_INST, BI_MIN = 8, 0.99
N_ANCHOR_CHAINS, ANCHOR_STEPS = 208, 4
RATE_PLAN, RATE_OBS = 2.7, 3.80                    # planning F (PREREG-LB's registered and observed ledger rates)
MAP_PLAN, MAP_OBS, REF_ANSWER_TOKENS = 2.7, 6000 / 1229.0, 20.0   # map: PREREG-3 S1 observed 4.88 gens/s, scaled
BATCH1_RATE = 0.30
LOAD_S = 120.0
CAP_H = 4.5
# expected NEW F keys per instance per planning arm (PREREG-LB's dev sizing, lbeam_run.SIZING: a T2-like canonical F
# 69.91, an OFF-like F 80.76, which PREREG-LB also used for the held-out audit arm); ceiling (E + W - 1) x V = 104
SIZING = {"T2BL@HS": LR.SIZING["G1"]["T2"], "T2s0@HS": LR.SIZING["G4"]["T2@heldout_combined"],
          "T2BL@canon": LR.SIZING["G1"]["T2"], "T2s0@canon": LR.SIZING["G1"]["T2"]}
CEILING = (E + W - 1) * P.V_DEFAULT
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
FILES = {
    "A0": {"anchor_off": "anchor/chain_one_L_off.jsonl", "anchor_on": "anchor/chain_one_L_t2s0.jsonl",
           "anchor": "anchor/anchor.json", "ident_rows": "anchor/adapter_identity.jsonl",
           "ident": "anchor/adapter_identity.json", "bi_rows": "anchor/batch_invariance.jsonl",
           "bi": "anchor/batch_invariance.json"},
    "M1": {**{f"map:{s}": f"map/map_{s}.jsonl" for s in MAP_SIDES}, "acct": "map/accounting.json"},
    "P1": {"eps:T2BL@HS": "plan/episodes_T2BL_HS.jsonl", "cache:T2BL@HS": "plan/fcache_T2BL_HS.jsonl",
           "acct": "plan/accounting_P1.json"},
    "P2": {"eps:T2s0@HS": "plan/episodes_T2s0_HS.jsonl", "cache:T2s0@HS": "plan/fcache_T2s0_HS.jsonl",
           "acct": "plan/accounting_P2.json"},
    "P3": {"eps:T2BL@canon": "plan/episodes_T2BL_canon.jsonl", "cache:T2BL@canon": "plan/fcache_T2BL_canon.jsonl",
           "acct": "plan/accounting_P3.json"},
    "P4": {"eps:T2s0@canon": "plan/episodes_T2s0_canon.jsonl", "cache:T2s0@canon": "plan/fcache_T2s0_canon.jsonl",
           "acct": "plan/accounting_P4.json"},
    "C1": {"eps:ORACLE": "cpu/episodes_ORACLE.jsonl", "eps:IGNORE_STATUS": "cpu/episodes_IGNORE_STATUS.jsonl",
           "acct": "cpu/accounting.json"},
}
EPISODE_FILES = {"T2BL@HS": FILES["P1"]["eps:T2BL@HS"], "T2s0@HS": FILES["P2"]["eps:T2s0@HS"],
                 "T2BL@canon": FILES["P3"]["eps:T2BL@canon"], "T2s0@canon": FILES["P4"]["eps:T2s0@canon"],
                 "ORACLE": FILES["C1"]["eps:ORACLE"]}
REPORT_EPISODE_FILES = {"IGNORE_STATUS": FILES["C1"]["eps:IGNORE_STATUS"]}
RET_DRAW_REL = f"{OUT_REL}/RET_DRAW.json"
RET_INTENT_REL = f"{OUT_REL}/RET_DRAW.intent.json"
RETENTION_RESULT_POINTER = f"{OUT_REL}/RETENTION.json"
RET_EXTRA_NAME = "t2bl_s0"                         # ret_run.py --extra-adapter t2bl_s0=<T2BL_RELS[0]> -> side x_t2bl_s0
KIT_CODE = ("scripts/t2bl_common.py", "scripts/t2bl_make_data.py", "scripts/t2bl_audit.py", "scripts/t2bl_run.py",
            "scripts/t2bl_gate.py", "scripts/t2bl_confirm_bank.py", "scripts/t2bl_confirm_run.py",
            "scripts/t2bl_confirm_analysis.py", "scripts/t2bl_test.py")
LB_CODE = ("scripts/lbeam_f.py", "scripts/lbeam_problems.py", "scripts/lbeam_proposer.py", "scripts/lbeam_run.py",
           "scripts/lbeam_analysis.py")
RET_CODE = ("scripts/ret_common.py", "scripts/ret_gen.py", "scripts/ret_run.py", "scripts/ret_analysis.py",
            "scripts/ret_grade.py", "docs/retention/RET_FREEZE.json", "docs/retention/DEV_SELECTION.json")
TEST_PARTS = ("p5analysis", "p5tiny", "p5freeze", "plan", "hook", "rehearsal")
REBOOT_TOL_S = 5.0                                  # psutil.boot_time() jitter allowed when comparing boots
PREFREEZE_RECORDS = ("docs/t2b/ledger/P5_DRYRUN.json", "docs/t2b/ledger/P5_DRYRUN.md",
                     "docs/t2b/ledger/P5_POWER_DEV.json", "docs/t2b/ledger/P5_GEN_PLAN.json",
                     *[f"docs/t2b/ledger/TEST_{p.upper()}.json" for p in TEST_PARTS])
DEV_SCREEN_INPUTS = (GATE_RULE_REL, "docs/t2b/ledger/GATE_RULE.md", f"{DEV_RUN_REL}/DONE.json",
                     f"{DEV_RUN_REL}/provenance.json", GATE_REL, *DEV_ON_RELS.values(),
                     *[f"{T2BL_RELS[s]}/{f}" for s in SEEDS for f in ADAPTER_FILES])
DATA_INPUTS = ("data/t2bl_train.jsonl", "data/t2bl_train.SOURCE.json", "data/t2bl_dev.jsonl", "data/t2bl_dev.SOURCE.json",
               "docs/ledger_beam/dev2_primary.SOURCE.json")
REQUIRED_INPUTS = (DRAFT_REL, *KIT_CODE, *LB_CODE, *RET_CODE, "scripts/p3_common.py", "scripts/p3_make_bank.py",
                   "scripts/l1_make_ood.py", "scripts/confirm2_gen.py", *PINS.keys(), *DATA_INPUTS, *DEV_SCREEN_INPUTS,
                   *PREFREEZE_RECORDS, HOOK_DRAFT_REL, *SEALED)
PROV_CODE = ("scripts/t2bl_confirm_run.py", "scripts/t2bl_common.py", "scripts/t2bl_confirm_bank.py", "scripts/lbeam_f.py",
             "scripts/lbeam_problems.py", "scripts/lbeam_proposer.py", "scripts/lbeam_run.py", "scripts/p3_common.py",
             "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/env.py", "rcwmt/capture.py")


class Stop(Exception):
    """A deterministic stop (exit 3, never retried). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


class Deferred(Exception):
    pass


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[prereg5] {msg}", persist)


def sha256(p: Path) -> str:
    return C.sha256(Path(p))


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _partial(p: Path) -> Path:
    return Path(p).with_name(Path(p).name + ".partial")


def write_text_x(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise Stop(f"[prereg5] {path} exists (create-only)")
    C.write_create(path, text)


def write_json_x(path: Path, obj, stale_partial: str = "refuse") -> None:
    path = Path(path)
    if path.exists():
        raise Stop(f"[prereg5] {path} exists (create-only)")
    C.write_create(path, json.dumps(obj, indent=1) + "\n", stale_partial=stale_partial)


def union_commit(cosign: str) -> str | None:
    m = re.search(r"union:([0-9a-f]{64})\b", str(cosign or ""))
    return m.group(1) if m else None


# ---------------------------------------------------------------- configuration (the registered job uses the defaults)
@dataclass
class Cfg:
    root: Path = ROOT
    out: Path = OUT
    map_bank: Path = ROOT / MAP_BANK_REL
    map_source: Path = ROOT / MAP_SOURCE_REL
    plan_draw: Path = ROOT / PLAN_DRAW_REL
    plan_source: Path = ROOT / PLAN_SOURCE_REL
    dev_set: Path = ROOT / DEV_SET_REL              # dev2_primary: batch-invariance keys
    dev_bank: Path = C.DEV                          # T2b-ledger dev bank: adapter-identity rows
    dev_on: dict = field(default_factory=lambda: {s: ROOT / r for s, r in DEV_ON_RELS.items()})
    t2s0_dir: Path = ROOT / T2S0_REL
    t2bl_dirs: dict = field(default_factory=lambda: {s: ROOT / r for s, r in T2BL_RELS.items()})
    anchor_chains: list | None = None               # None: the pilot TEST ledger chains (208)
    anchor_refs: dict = field(default_factory=lambda: {s: ROOT / r for s, r in ANCHOR_REF_RELS.items()})
    require_freeze: bool = True                     # False only from the tests (never reachable from the CLI)
    cap_h: float = CAP_H
    rate: float = RATE_PLAN
    map_rate: float = MAP_PLAN
    bi_n_inst: int = BI_N_INST
    ident_per_stratum: int = IDENT_PER_STRATUM
    ident_min: float = IDENT_MIN

    def identity(self, side: str) -> dict:
        d = self.t2s0_dir if side == "t2s0" else self.t2bl_dirs[int(side[-1])]
        return {"model": "Titan-Sparse FP32 " + FP["sparse"], "adapter_name": ADAPTER_NAMES[side],
                "adapter_sha256": sha256(Path(d) / "adapter_model.safetensors"),
                "decode": "greedy, 160 new tokens, EOS [151645, 151643]"}


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def stage_required(stage: str, out: Path) -> list:
    return [Path(out) / rel for rel in FILES[stage].values()]


def stage_owned(stage: str, out: Path) -> list:
    if stage == "A0":
        return [Path(out) / "anchor"]
    fs = stage_required(stage, out)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path) -> str:
    return str(Path(p).relative_to(Path(out))).replace("\\", "/")


def stage_states(prov: dict, out: Path) -> dict:
    """{stage: 'done' | 'partial' | 'todo'}; a recorded file whose sha256 changed stops the job."""
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
    print(f"[prereg5] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


def eligible_parts(prov: dict | None, out: Path) -> list:
    """Parts whose stages are all recorded with unchanged outputs, given that A0's first run passed."""
    if not prov:
        return []
    st = prov.get("stages") or {}
    a0 = (C.read_json(Path(out) / FILES["A0"]["anchor"]) or {}).get("pass") is True and \
        (C.read_json(Path(out) / FILES["A0"]["ident"]) or {}).get("pass") is True and \
        (C.read_json(Path(out) / FILES["A0"]["bi"]) or {}).get("pass") is True
    unchanged = True
    for s, r in st.items():
        for p in stage_required(s, out) if s in FILES else []:
            if not p.is_file() or sha256(p) != (r.get("files") or {}).get(rel_out(p, out)):
                unchanged = False
    if not (a0 and unchanged):
        return []
    return [pt for pt, need_st in PART_STAGES.items() if all(s in st for s in need_st)]


# ---------------------------------------------------------------- estimates and the turn plan (no torch)
def map_s_per_gen(answer_tokens: float, rate: float) -> float:
    return (answer_tokens + 1.0) / (REF_ANSWER_TOKENS + 1.0) / rate


def counts(n_map: int, n_inst: int, n_anchor: int = N_ANCHOR_CHAINS, bi_n_inst: int = BI_N_INST,
           ident: int = IDENT_PER_STRATUM * len(IDENT_STRATA) * len(SEEDS), answer_tokens: float = 25.0) -> dict:
    bi_keys = bi_n_inst * P.V_DEFAULT
    return {"map_rows": n_map, "inst": n_inst, "anchor_gens": n_anchor * ANCHOR_STEPS * 2, "ident_gens": ident,
            "bi_batched": 2 * bi_keys, "bi_batch1": 2 * bi_keys, "map_gens": n_map * len(MAP_SIDES),
            "answer_tokens": answer_tokens}


def stage_gens(st: str, n: dict, mode: str = "expected") -> float:
    if st == "A0":
        return n["anchor_gens"] + n["ident_gens"] + n["bi_batched"] + n["bi_batch1"]
    if st == "M1":
        return n["map_gens"]
    if st in STAGE_ARMS:
        return n["inst"] * sum((SIZING[a] if mode == "expected" else CEILING) for a in STAGE_ARMS[st])
    return 0.0


def stage_est_s(st: str, n: dict, rate: float, map_rate: float, mode: str = "expected") -> float:
    if st == "A0":
        return (n["anchor_gens"] / rate + n["ident_gens"] * map_s_per_gen(n["answer_tokens"], map_rate)
                + n["bi_batched"] / rate + n["bi_batch1"] / BATCH1_RATE)
    if st == "M1":
        return n["map_gens"] * map_s_per_gen(n["answer_tokens"], map_rate)
    return stage_gens(st, n, mode) / rate


def start_estimate_s(st: str, n: dict, rate: float, map_rate: float, measured_spg: float | None) -> float:
    """The estimate the runner's stage-start rule uses (D-22: a stage starts only if 1.25 x this fits the turn and not
    within 0.5 h of the cap). A0 and M1 have deterministic generation counts: their expected time at the registered
    rates. A planning stage (P1-P4) is bounded by its CEILING generation count ((E + W - 1) x V = 104 new keys per
    instance; the expected count is 70-81) times the seconds per generation: the slowest rate measured on a planning
    stage already recorded in this run, or the registered planning rate before the first one."""
    if st in STAGE_ARMS:
        spg = measured_spg if measured_spg else 1.0 / rate
        return stage_gens(st, n, "ceiling") * spg
    return stage_est_s(st, n, rate, map_rate)


def measured_planning_spg(prov: dict) -> float | None:
    """The slowest seconds per generation of the planning stages recorded in this run (None before the first)."""
    xs = [float(r["seconds"]) / float(r["generations"]) for st, r in (prov.get("stages") or {}).items()
          if st in STAGE_ARMS and r.get("generations") and r.get("seconds")]
    return max(xs) if xs else None


def simulate_turns(n: dict, true_rate: float, true_map_rate: float, sizes: str = "expected", cap_h: float = CAP_H,
                   plan_rate: float = RATE_PLAN, plan_map_rate: float = MAP_PLAN) -> list:
    """The turns the runner's own start rule produces when every stage takes its time at the TRUE rates (planning
    stages at their expected or ceiling sizes), with the start decisions made exactly as in _run_locked: the model load
    and the longest A0 so far added before the first stage of a turn, A0 re-run at the start of every later turn, a
    planning stage gated on start_estimate_s with the rate measured on the planning stages already done."""
    cap = cap_h * 3600
    left = list(GPU_STAGES)
    turns, a0_s, spg = [], None, None
    true_s = lambda st: (stage_gens(st, n, sizes) / true_rate if st in STAGE_ARMS  # noqa: E731
                         else stage_est_s(st, n, true_rate, true_map_rate))
    est = lambda st: start_estimate_s(st, n, plan_rate, plan_map_rate, spg)  # noqa: E731
    while left:
        t, ran = 0.0, []
        extra = LOAD_S + (a0_s if a0_s is not None else 0.0)
        if not G2.can_start(extra, est(left[0]), 0.0, cap):
            raise RuntimeError(f"no stage fits an empty turn ({sizes}, {true_rate}/s)")
        t = LOAD_S
        if a0_s is not None:
            t += a0_s
            ran.append("A0 re-check")
        for st in list(left):
            if not G2.can_start(t, est(st), 0.0, cap):
                break
            d = true_s(st)
            t += d
            ran.append(st)
            left.remove(st)
            if st == "A0":
                a0_s = d
            if st in STAGE_ARMS:
                spg = max(spg or 0.0, 1.0 / true_rate)
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


SCHEDULES = (("expected sizes, observed rates", "expected", RATE_OBS, MAP_OBS),
             ("expected sizes, planning rates", "expected", RATE_PLAN, MAP_PLAN),
             ("ceiling sizes, observed rates", "ceiling", RATE_OBS, MAP_OBS),
             ("ceiling sizes, planning rates", "ceiling", RATE_PLAN, MAP_PLAN),
             ("ceiling sizes, 2.5 generations/s (below the planning rate)", "ceiling", 2.5, 2.5))


def plan(verbose: bool = True) -> dict:
    src = C.read_json(C.DEV_SOURCE) or {}
    n_map = C.bank_rows_expected(C.CONFIRM_STRATA)
    ans = ((C.read_json(C.TRAIN_SOURCE) or {}).get("token_stats") or {}).get("dev", {}).get("mean_target_tokens", 25.0)
    n = counts(n_map, CB.N_PER_DEPTH * len(CB.DEPTHS), answer_tokens=ans)
    what = {"A0": "PREREG-2 anchor (208 chains x 4 x OFF/T2s0) + adapter identity (3 x 100 dev rows) + batch invariance "
                  "(64 keys x T2BL canonical / held-out x batched and batch 1)",
            "M1": f"part (a): the map bank ({n_map} rows) x OFF, T2s0, T2BL s0-s2",
            "P1": "part (b) PRIMARY arm: plan_beam enum W2 E12, T2BL@HS (held-out combined rendering)",
            "P2": "part (b) PRIMARY comparator: plan_beam enum W2 E12, T2s0@HS (held-out combined rendering)",
            "P3": "part (b): plan_beam enum W2 E12, T2BL@canon (canonical frame)",
            "P4": "part (b): plan_beam enum W2 E12, T2s0@canon (canonical frame)"}
    res = {"lease": LEASE_LABEL, "cap_h": CAP_H, "load_s": LOAD_S, "counts": n, "sizing_new_keys_per_instance": SIZING,
           "ceiling_per_instance": CEILING, "rates": {"planning_F_plan": RATE_PLAN, "planning_F_observed": RATE_OBS,
                                                      "map_plan": MAP_PLAN, "map_observed": round(MAP_OBS, 3),
                                                      "batch1": BATCH1_RATE},
           "stages": [], "cpu_stage": "C1 (ORACLE and IGNORE_STATUS at enum W2 E12): CPU only, never under the lease, at "
                                      "the start of the invocation after the one that records A0; about 1 CPU-min",
           "retention_part_c": "a separate RCWM turn under lease 'rcwm-t1:retention' (ret_run.py --mode confirm with "
                               "--extra-adapter t2bl_s0=...): 0.46-1.11 GPU-h by the 5c plan (docs/retention/RET_PLAN.json "
                               "scale; one extra Titan-Sparse side)",
           "stage_rule": (f"no stage starts within {G2.STAGE_MARGIN_H} h of the {CAP_H} h cap; a stage starts only if "
                          f"{G2.STAGE_SAFETY} x its start estimate fits (A0, M1: expected time at the registered rates; "
                          "P1-P4: the CEILING generation count x the slowest seconds per generation measured on a "
                          "recorded planning stage, or the registered planning rate before the first); every turn "
                          "after the first re-runs A0 "
                          "before any stage; the start-of-turn check adds the model load and the longest A0 measured so "
                          "far, and the model is not loaded if the first remaining stage does not fit"),
           "dev_answer_tokens": ans, "dev_source_present": bool(src), "schedules": []}
    for st in GPU_STAGES:
        res["stages"].append({"stage": st, "what": what[st], "generations_expected": round(stage_gens(st, n)),
                              "generations_ceiling": round(stage_gens(st, n, "ceiling")),
                              "h_planning": round(stage_est_s(st, n, RATE_PLAN, MAP_PLAN) / 3600, 2),
                              "h_observed": round(stage_est_s(st, n, RATE_OBS, MAP_OBS) / 3600, 2),
                              "h_ceiling_planning": round(stage_est_s(st, n, RATE_PLAN, MAP_PLAN, "ceiling") / 3600, 2),
                              "h_start_estimate_first": round(start_estimate_s(st, n, RATE_PLAN, MAP_PLAN, None) / 3600, 2)})
    for name, sizes, rate, mrate in SCHEDULES:
        try:
            turns = simulate_turns(n, rate, mrate, sizes)
            res["schedules"].append({"name": name, "sizes": sizes, "true_rate_F": rate, "true_rate_map": round(mrate, 3),
                                     "turns": turns, "gpu_h": round(sum(t["projected_h"] for t in turns), 2),
                                     "max_turn_h": max(t["projected_h"] for t in turns)})
        except RuntimeError as e:
            res["schedules"].append({"name": name, "sizes": sizes, "turns": str(e), "gpu_h": None, "max_turn_h": None})
    if verbose:
        print(f"PREREG-5 GPU plan ({n_map} map rows, {n['inst']} planning instances)")
        for r in res["stages"]:
            print(f"  {r['stage']}  {r['what'][:96]:<96} gens ~{r['generations_expected']:>6} (<= {r['generations_ceiling']:>6})"
                  f"  {r['h_planning']:5.2f} h plan  {r['h_observed']:5.2f} h obs  {r['h_ceiling_planning']:5.2f} h ceil")
        for sc in res["schedules"]:
            v = sc["turns"]
            print(f"  {sc['name']}: " + ("; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(v))
                                         if isinstance(v, list) else v) + f"  (max {sc['max_turn_h']} h)")
    return res


# ---------------------------------------------------------------- freeze verification
def _num(rx: str, name: str) -> int:
    m = re.search(rx, name)
    return int(m.group(1)) if m else -1


def required_inputs(root: Path) -> set:
    return set(REQUIRED_INPUTS) | {f"rcwmt/{p.name}" for p in (Path(root) / "rcwmt").glob("*.py")}


def _amendments(root: Path) -> list:
    return sorted(Path(root).glob(AMEND_GLOB), key=lambda p: _num(r"_(\d+)\.json$", p.name))


def salt_commit(salt: str) -> str:
    return hashlib.sha256(str(salt).encode("utf-8")).hexdigest()


def cosign_commit(cosign: str) -> str | None:
    m = re.search(r"commit:([0-9a-f]{64})\b", str(cosign or ""))
    return m.group(1) if m else None


def verify_freeze(root: Path | None = None) -> tuple:
    root = Path(root or ROOT)
    fz = C.read_json(root / FREEZE_REL)
    if fz is None:
        half = interrupted_freeze(root)
        return False, [half or "PREREG_5_FREEZE.json does not exist or is unreadable (PREREG-5 is a draft)"], None
    errs = []
    pr = root / PREREG_REL
    if not pr.is_file():
        errs.append("PREREG_5.md does not exist")
    elif sha256(pr) != fz.get("prereg_sha256"):
        errs.append("PREREG_5.md differs from the freeze")
    inputs = dict(fz.get("inputs") or {})
    missing = sorted(required_inputs(root) - set(inputs))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing[:8]}")
    for rel, h in PINS.items():
        if inputs.get(rel) != h:
            errs.append(f"{rel}: freeze record differs from the registered pin")
    expected = dict(inputs)
    for a in _amendments(root):
        am = C.read_json(a) or {}
        if am.get("file") not in expected or expected[am["file"]] != am.get("old_sha256") or not am.get("new_sha256"):
            errs.append(f"{a.name}: old hash does not match the chain")
            continue
        if not str(am.get("reason") or "").strip() or not str(am.get("approved_by") or "").strip():
            errs.append(f"{a.name}: an amendment must state its reason and who approved it (approved_by)")
        expected[am["file"]] = am["new_sha256"]
    for rel, h in sorted(expected.items()):
        p = root / rel
        if not p.is_file():
            errs.append(f"{rel} missing")
        elif sha256(p) != h:
            errs.append(f"{rel} differs from the freeze (+ amendments)")
    bwm = fz.get("bwm") or {}
    if bwm.get("planner_sha256") != BWM_PLANNER_SHA:
        errs.append("the freeze record's BWM planner sha is not the pinned one")
    bp = BWM_ROOT / BWM_PLANNER_REL
    if not bp.is_file() or sha256(bp) != BWM_PLANNER_SHA:
        errs.append(f"BWM's {BWM_PLANNER_REL} differs from the pinned sha256")
    cos = str(fz.get("bwm_cosign") or "")
    sealed = fz.get("sealed") or {}
    try:
        errs += CB.verify_sealed(root, {"map_bank": sealed.get("map_bank") or {}, "plan_draw": sealed.get("plan_draw") or {}})
    except Exception as e:                            # noqa: BLE001  (a malformed sealed file is a verification failure)
        errs.append(f"sealed sets do not verify ({type(e).__name__})")
    for rel, key in ((MAP_SOURCE_REL, "map_bank"), (PLAN_SOURCE_REL, "plan_draw")):
        src = C.read_json(root / rel) or {}
        sr = src.get("seed_rule") or {}
        if sr.get("draft_sha256") != fz.get("draft_sha256") or sr.get("draft_sha256") != inputs.get(DRAFT_REL):
            errs.append(f"{key}: the seed rule's draft sha256 is not the frozen draft's")
        if sr.get("bwm_commit") != cosign_commit(cos) or salt_commit(str(sr.get("bwm_salt"))) != sr.get("bwm_commit") or \
                len(str(sr.get("bwm_salt") or "")) < SALT_MIN_LEN:
            errs.append(f"{key}: BWM's revealed salt does not match the commitment in its co-sign")
        if (sealed.get(key) or {}).get("seed") != sr.get("seed"):
            errs.append(f"{key}: the seed differs between the freeze record and its SOURCE record")
    if not fz.get("draft_sha256") or str(fz.get("draft_sha256")) not in cos:
        errs.append("BWM's co-sign does not quote the frozen draft's sha256")
    if union_commit(cos) is None or union_commit(cos) != sealed.get("union_digest"):
        errs.append("BWM's co-sign does not quote the union digest the sealed sets were minted from")
    if fz.get("registered_runtime") != REGISTERED_RUNTIME:
        errs.append("registered runtime differs")
    if not fz.get("approval") or not fz.get("bwm_cosign"):
        errs.append("no approval / BWM co-sign recorded")
    if (fz.get("dev_gate") or {}).get("verdict") != "PROCEED":
        errs.append("the freeze record does not show the dev gate's PROCEED")
    return not errs, errs, fz


def allowed_hashes(fz: dict, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    allowed = {k: {v} for k, v in (fz.get("inputs") or {}).items()}
    for a in _amendments(root):
        am = C.read_json(a) or {}
        if am.get("file") in allowed and am.get("new_sha256"):
            allowed[am["file"]].add(am["new_sha256"])
    return allowed


def prereg5_outputs(root: Path) -> list:
    """Anything a PREREG-5 run or analysis writes. The freeze refuses if one exists."""
    root = Path(root)
    found = []
    runs = root / "runs"
    if runs.exists():
        found += [str(p.relative_to(root)) for p in runs.iterdir() if p.name.startswith("prereg5")]
        found += [str(p.relative_to(root)) for p in runs.rglob("PREREG_5_RESULTS*")]
    return sorted(set(found))


def interrupted_freeze(root: Path) -> str | None:
    root = Path(root)
    rels = (*SEALED, PREREG_REL, FREEZE_REL)
    partials = [r for r in rels if _partial(root / r).exists()]
    pr, fz = root / PREREG_REL, root / FREEZE_REL
    if pr.exists() and not fz.exists():
        rec = C.read_json(_partial(fz)) or {}
        if _partial(fz).exists() and rec.get("prereg_sha256") == sha256(pr):
            return ("the freeze was interrupted before its last rename: PREREG_5.md is in place and "
                    "PREREG_5_FREEZE.json.partial records its sha256. Rename PREREG_5_FREEZE.json.partial to "
                    "PREREG_5_FREEZE.json, run --verify, and log it in DECISIONS.md")
        return ("PREREG_5.md exists without PREREG_5_FREEZE.json (and without a matching .partial record): not a freeze. "
                "Move PREREG_5.md, the sealed sets and any .partial aside, log it in DECISIONS.md, rerun --freeze")
    if not fz.exists() and partials:
        return (f"a freeze was interrupted before its renames ({partials} exist as .partial; nothing is frozen): move the "
                ".partial files (and any renamed sealed file) aside UNREAD, log it in DECISIONS.md, rerun --freeze")
    if not fz.exists() and any((root / r).exists() for r in SEALED):
        return ("a sealed set exists without a freeze: move it aside UNREAD, log it in DECISIONS.md, and rerun --freeze "
                "(the seed rule re-mints the same sets if the union and BWM's salt are unchanged)")
    return None


def _record_ok(rec: dict | None, root: Path, code: tuple) -> bool:
    return bool(rec and rec.get("ok")) and all((rec.get("code_sha256") or {}).get(r) == sha256(Path(root) / r)
                                               for r in code if (Path(root) / r).is_file())


def _binds_draft(rec: dict | None, root: Path) -> bool:
    root = Path(root)
    return bool(rec) and (root / DRAFT_REL).is_file() and (root / HOOK_DRAFT_REL).is_file() and \
        rec.get("draft_sha256") == sha256(root / DRAFT_REL) and rec.get("hook_draft_sha256") == sha256(root / HOOK_DRAFT_REL)


def dev_gate_status(root: Path) -> dict:
    """The dev screen's registered outcome, read from its create-only records (never re-computed)."""
    root = Path(root)
    g = C.read_json(root / GATE_REL) or {}
    gr = C.read_json(root / GATE_RULE_REL) or {}
    done = C.read_json(root / DEV_RUN_REL / "DONE.json") or {}
    return {"verdict": (g.get("decision") or {}).get("verdict"), "registered": g.get("registered") is True,
            "gate_sha256": sha256(root / GATE_REL) if (root / GATE_REL).is_file() else None,
            "gate_rule_frozen": gr.get("record") == "T2BL_GATE_RULE", "dev_run_done": bool(done),
            "rules_digest_at_gate_freeze": (gr.get("prereg5_rules") or {}).get("sha256"),
            "code_at_gate_freeze": (gr.get("prereg5_code") or {}).get("files"),
            "gate_frozen_local": gr.get("frozen_local")}


def prefreeze_checks(root: Path) -> dict:
    """Everything that must hold before PREREG-5 can freeze (also run by the tests on a fake root)."""
    import t2bl_confirm_analysis as CA               # numpy only; torch never imported
    root = Path(root)
    d = root / "docs" / "t2b" / "ledger"
    out = {}
    for part in TEST_PARTS:
        rec = C.read_json(d / f"TEST_{part.upper()}.json")
        out[f"test record TEST_{part.upper()}.json passed with the current kit code"] = _record_ok(rec, root, KIT_CODE)
        out[f"test record TEST_{part.upper()}.json was produced with the current draft and hook draft"] = _binds_draft(rec, root)
    dr = C.read_json(d / "P5_DRYRUN.json")
    out["analysis dry run (every wording branch) passed with the current analysis code"] = _record_ok(
        dr, root, ("scripts/t2bl_confirm_analysis.py", "scripts/t2bl_common.py"))
    draft = root / DRAFT_REL
    miss = CA.registered_strings_missing(draft.read_text(encoding="utf-8")) if draft.is_file() else ["draft missing"]
    out["every rule, limit and sentence template of the analysis appears verbatim in the draft"] = not miss
    pw = C.read_json(d / "P5_POWER_DEV.json") or {}
    out["power recomputed from the dev screen's outcomes (P5_POWER_DEV.json)"] = str(pw.get("assumption_source", "")).endswith("GATE.json")
    gp = C.read_json(d / "P5_GEN_PLAN.json") or {}
    out["P5_GEN_PLAN.json present"] = bool(gp.get("stages"))
    out["queue hook draft present"] = (root / HOOK_DRAFT_REL).is_file()
    gs = dev_gate_status(root)
    out["the dev gate rule was frozen before the dev screen (GATE_RULE.json)"] = gs["gate_rule_frozen"]
    out["the dev screen completed (runs/t2bl/dev/DONE.json)"] = gs["dev_run_done"]
    out["the registered dev gate says PROCEED"] = gs["verdict"] == "PROCEED" and gs["registered"]
    out["T2b-ledger adapters of seeds 0, 1, 2 present"] = all((root / T2BL_RELS[s] / f).is_file() for s in SEEDS
                                                             for f in ADAPTER_FILES)
    out["registered pins (T2 s0 adapter, anchor references, pilot bank, dev2_primary) match"] = all(
        (root / r).is_file() and sha256(root / r) == h for r, h in PINS.items())
    out["BWM planner is the pinned file"] = (BWM_ROOT / BWM_PLANNER_REL).is_file() and \
        sha256(BWM_ROOT / BWM_PLANNER_REL) == BWM_PLANNER_SHA
    rf = C.read_json(root / "docs" / "retention" / "RET_FREEZE.json")
    out["the 5c retention suite is frozen and has its dev selection"] = bool(rf) and \
        (root / "docs" / "retention" / "DEV_SELECTION.json").is_file()
    return out


def _fill_prereg(text: str, stamp: str, inputs: dict, approval: str, cosign: str, sealed: dict, gate: dict,
                 rules: dict, changes: list | None = None) -> str:
    lines = text.splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("**Status:**")]
    if len(st) != 1:
        raise Stop("the draft has no unique **Status:** line", persist=False)
    lines[st[0]] = (f"**Status:** FROZEN at {stamp} by `scripts/t2bl_confirm_run.py --freeze` (approval: {approval}; BWM "
                    f"co-sign: {cosign}). The SHA-256 of this file is in `PREREG_5_FREEZE.json`.")
    i10 = [i for i, l in enumerate(lines) if l.startswith("## 10.")]
    if len(i10) != 1:
        raise Stop("the draft has no unique section 10", persist=False)
    nxt = next((i for i in range(i10[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    m, p = sealed["map_bank"], sealed["plan_draw"]
    sec = ["## 10. Freeze record", "",
           f"Frozen at {stamp} by `scripts/t2bl_confirm_run.py --freeze`. Dev gate: {gate['verdict']} (GATE.json sha256 "
           f"`{gate['gate_sha256']}`). Decision rules unchanged since the dev gate rule was frozen: {rules['unchanged']} "
           f"(rules digest `{rules['sha256']}`).", "",
           f"Map bank minted at the freeze: seed {m['seed']} (rule: {CB.SEED_FORMULA}, tag '{CB.TAG_MAP}'), {m['n']} rows, "
           f"sealed as `{MAP_BANK_REL}` sha256 `{m['file_sha256']}`. Planning draw minted at the freeze: seed {p['seed']} "
           f"(tag '{CB.TAG_PLAN}'), {p['n']} instances, sealed as `{PLAN_DRAW_REL}` sha256 `{p['file_sha256']}`. Draft "
           f"sha256 {sealed['draft_sha256']}; union digest {sealed['union_digest']}; BWM salt revealed at the freeze, "
           f"commitment {sealed['bwm_commit']}. BWM planner `{BWM_PLANNER_REL}` sha256 `{BWM_PLANNER_SHA}`. data/ was "
           f"unchanged while the union was built ({sealed['data_files']} files, snapshot digest "
           f"{sealed['data_snapshot_digest']}).", "", "SHA-256 of every frozen input:", ""]
    sec += [f"- `{k}` `{v}`" for k, v in sorted(inputs.items())]
    sec.append("")
    out = lines[: i10[0]] + sec + lines[nxt:]
    if changes:
        ia = [i for i, l in enumerate(out) if l.startswith("## Appendix A.")]
        if len(ia) != 1:
            raise Stop("the draft has no unique Appendix A", persist=False)
        na = next((i for i in range(ia[0] + 1, len(out)) if out[i].startswith("## ")), len(out))
        add = ["", "### Changes after the dev gate rule was frozen (listed at the freeze, --changes)", ""]
        add += [f"- {c['what']} (files: {', '.join(c.get('files') or ['the registered rules'])}). Reason: {c['why']}"
                for c in changes]
        out = out[:na] + add + [""] + out[na:]
    return "\n".join(out) + "\n"


def check_changes(changes, rules: dict) -> list:
    """The change list a freeze needs when PREREG-5's rules or decision code changed after the dev gate's freeze: every
    changed code file and, if the rules digest changed, an entry with 'rules': true; each entry states what and why."""
    probs = []
    if not isinstance(changes, list) or not changes:
        return ["no change list (--changes)"]
    for c in changes:
        if not isinstance(c, dict) or not str(c.get("what") or "").strip() or not str(c.get("why") or "").strip():
            probs.append("every change states 'what' and 'why'")
    listed = {f for c in changes if isinstance(c, dict) for f in (c.get("files") or [])}
    for f in rules.get("code_changed") or []:
        if f not in listed:
            probs.append(f"{f} changed since the dev gate's freeze and is not listed")
    if rules.get("rules_changed") and not any(isinstance(c, dict) and c.get("rules") is True for c in changes):
        probs.append("the registered rules changed since the dev gate's freeze and no change says 'rules': true")
    return probs


def freeze(approval: str, bwm_cosign: str, bwm_salt: str, root: Path | None = None, checks=None, union_builder=None,
           changes: list | None = None) -> dict:
    """The freeze (section 7.2 of the draft). union_builder: tests pass a small stand-in (never the registered path)."""
    import t2bl_confirm_analysis as CA
    import t2bl_make_data as MD
    root = Path(root or ROOT)
    if not approval or len(approval.strip()) < 8 or not re.search(r"\bD-\w+", approval):
        raise Stop("--approval must state who approved the freeze, when, and cite the DECISIONS entry ('D-<n>')",
                   persist=False)
    cosign = (bwm_cosign or "").strip()
    if len(cosign) < 8:
        raise Stop("--bwm-cosign must cite BWM's co-sign (reference, this draft's full sha256, commit:<sha256 of the "
                   "salt>)", persist=False)
    half = interrupted_freeze(root)
    if half:
        raise Stop(f"refusing to freeze: {half}", persist=False)
    if (root / FREEZE_REL).exists() or (root / PREREG_REL).exists():
        raise Stop("PREREG-5 is already frozen (PREREG_5_FREEZE.json or PREREG_5.md exists)", persist=False)
    outs = prereg5_outputs(root)
    if outs:
        raise Stop(f"PREREG-5 outputs exist before the freeze: {outs}; investigate before freezing", persist=False)
    if not (root / DRAFT_REL).is_file():
        raise Stop(f"{DRAFT_REL} missing", persist=False)
    draft_bytes = (root / DRAFT_REL).read_bytes()            # read ONCE: the seeds, the inputs and PREREG_5.md
    draft_sha = hashlib.sha256(draft_bytes).hexdigest()
    if draft_sha not in cosign:
        raise Stop(f"--bwm-cosign does not quote this draft's sha256 ({draft_sha}); BWM co-signs a stated text",
                   persist=False)
    commit = cosign_commit(cosign)
    if commit is None:
        raise Stop("--bwm-cosign carries no salt commitment 'commit:<sha256 of BWM's salt>'", persist=False)
    quoted_union = union_commit(cosign)
    if quoted_union is None:
        raise Stop("--bwm-cosign carries no union digest 'union:<digest>' (publish it with --union-digest before the "
                   "co-sign)", persist=False)
    salt = str(bwm_salt or "")
    if len(salt) < SALT_MIN_LEN or salt_commit(salt) != commit:
        raise Stop(f"--bwm-salt is missing, shorter than {SALT_MIN_LEN} characters, or does not match the commitment",
                   persist=False)
    miss = sorted(r for r in required_inputs(root) if r not in SEALED and not (root / r).is_file())
    if miss:
        raise Stop(f"frozen inputs missing (nothing was minted): {miss[:10]}", persist=False)
    chk = (checks or prefreeze_checks)(root)
    bad = [k for k, v in chk.items() if not v]
    if bad:
        raise Stop(f"pre-freeze checks failed: {bad}", persist=False)
    gate = dev_gate_status(root)
    rd = CA.rules_digest()
    cd = CA.code_digest(root)
    code0 = gate.get("code_at_gate_freeze") or {}
    code_changed = sorted(r for r in set(code0) | set(cd["files"]) if code0.get(r) != cd["files"].get(r))
    rules_changed = not (bool(gate["rules_digest_at_gate_freeze"]) and rd["sha256"] == gate["rules_digest_at_gate_freeze"])
    rules = {"sha256": rd["sha256"], "at_gate_freeze": gate["rules_digest_at_gate_freeze"], "rules_changed": rules_changed,
             "code_sha256": cd["files"], "code_changed": code_changed,
             "unchanged": not rules_changed and not code_changed and bool(code0),
             "gate_frozen_local": gate["gate_frozen_local"], "changes": changes or []}
    if not rules["unchanged"]:
        if not code0:
            rules["code_changed"] = code_changed = sorted(cd["files"])
        bad_ch = check_changes(changes, rules)
        if bad_ch:
            raise Stop("PREREG-5's rules or decision code differ from those pinned when the dev gate rule was frozen "
                       f"(rules changed: {rules_changed}; code changed: {code_changed}); the freeze needs a change list "
                       f"naming each change and why (--changes): {bad_ch}", persist=False)
    snap0 = LR.data_snapshot(root)
    excl = (union_builder or (lambda r: MD.build_union(root=r, skip_prefix=None)))(root)
    bad_src = MD.union_errors(excl)
    if bad_src:
        raise Stop(f"a source of the freshness union could not be read: {bad_src}; nothing was minted", persist=False)
    if CB.union_digest(excl) != quoted_union:
        raise Stop("the rebuilt freshness union's digest differs from the one BWM's co-sign quotes (data/ or docs/ "
                   "changed since it was published); nothing was minted. Publish the new digest (--union-digest) and get "
                   "a NEW co-sign with a NEW salt (this salt was revealed)", persist=False)
    snap1 = LR.data_snapshot(root)
    if snap1 != snap0:
        changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
        raise Stop(f"data/ changed while the freshness union was built ({changed[:6]}); freeze only while no agent "
                   "writes to data/. Nothing was minted", persist=False)
    if sha256(root / DRAFT_REL) != draft_sha:
        raise Stop("the draft changed during the freeze; nothing was minted", persist=False)
    try:
        M = CB.mint(draft_sha, excl, salt)
    except CB.BankError as e:
        raise Stop(f"minting failed: {e}; nothing was written", persist=False)
    del excl
    staged = {MAP_BANK_REL: M["map_text"], MAP_SOURCE_REL: json.dumps(M["map_source"], indent=1) + "\n",
              PLAN_DRAW_REL: M["plan_text"], PLAN_SOURCE_REL: json.dumps(M["plan_source"], indent=1) + "\n"}
    for rel, t in staged.items():
        with open(_partial(root / rel), "x", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
    inputs = {}
    for rel in sorted(required_inputs(root)):
        p = _partial(root / rel) if rel in staged else root / rel
        if not p.is_file():
            raise Stop(f"frozen input missing: {rel}", persist=False)
        inputs[rel] = sha256(p)
    for rel, h in PINS.items():
        if inputs.get(rel) != h:
            raise Stop(f"{rel} is not the registered pinned file", persist=False)
    if inputs.get(DRAFT_REL) != draft_sha:
        raise Stop("the draft changed during the freeze (the .partial files are not renamed; see --verify)", persist=False)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    ms, ps = M["map_source"], M["plan_source"]
    sealed = {"map_bank": {"seed": ms["seed_rule"]["seed"], "n": len(M["map_rows"]), "file_sha256": ms["file_sha256"],
                           "strata": ms["strata"], "overlap": ms["overlap"], "fresh_rejections": len(ms["fresh_rejections"])},
              "plan_draw": {"seed": ps["seed_rule"]["seed"], "n": len(M["plan_insts"]), "file_sha256": ps["file_sha256"],
                            "per_depth": ps["per_depth"], "overlap": ps["overlap"],
                            "fresh_rejections": len(ps["fresh_rejections"])},
              "draft_sha256": draft_sha, "union_digest": ms["seed_rule"]["union_digest"], "bwm_commit": commit,
              "data_files": len(snap0), "data_snapshot_digest": LR.snapshot_digest(snap0)}
    text_pr = _fill_prereg(draft_bytes.decode("utf-8"), stamp, inputs, approval.strip(), cosign, sealed, gate, rules,
                           changes if not rules["unchanged"] else None)
    with open(_partial(root / PREREG_REL), "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text_pr)
    frec = {"frozen_at": stamp, "approval": approval.strip(), "bwm_cosign": cosign,
            "prereg_sha256": sha256(_partial(root / PREREG_REL)), "draft_sha256": draft_sha, "inputs": inputs,
            "prefreeze_checks": chk, "registered_runtime": REGISTERED_RUNTIME, "lease_label": LEASE_LABEL,
            "out_dir": OUT_REL, "no_outputs_before_freeze": True, "sealed": sealed, "dev_gate": gate, "rules": rules,
            "bwm": {"root": str(BWM_ROOT), "planner_rel": BWM_PLANNER_REL, "planner_sha256": BWM_PLANNER_SHA,
                    "salt_commit": commit, "imported_files_recorded_not_binding": {
                        r: sha256(BWM_ROOT / r) for r in BWM_INFO_RELS if (BWM_ROOT / r).is_file()}}}
    with open(_partial(root / FREEZE_REL), "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(frec, indent=1) + "\n")
    for rel in (*SEALED, PREREG_REL, FREEZE_REL):
        os.rename(_partial(root / rel), root / rel)          # os.rename refuses an existing target on Windows
    ok, errs, _ = verify_freeze(root)
    if not ok:
        raise Stop(f"the freeze record was written but does not verify: {errs}", persist=False)
    print(f"[prereg5] FROZEN {stamp}; PREREG_5.md sha {frec['prereg_sha256']}; map bank sealed sha "
          f"{sealed['map_bank']['file_sha256']} ({sealed['map_bank']['n']} rows), planning draw sealed sha "
          f"{sealed['plan_draw']['file_sha256']} ({sealed['plan_draw']['n']} instances); contents not shown", flush=True)
    return frec


# ---------------------------------------------------------------- run lock, one-waiter rule, installed hook, runtime
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
            raise Stop(f"[prereg5] another t2bl_confirm_run.py holds the run lock {rec}", persist=False)
        print(f"[prereg5] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink(missing_ok=True)
    with open(lk, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return lk


def one_waiter_problem(backend) -> str | None:
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for or holds the GPU lease. BWM's jobs are allowed."""
    w = C.read_json(backend.waiting_marker_path())
    if w and LR._live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = C.read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and LR._live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


def hook_problem(root: Path, fz: dict | None) -> str | None:
    """Every installed queue hook whose name contains 'prereg5' (the documented name is 90_prereg5.sh) must be the
    frozen hook draft byte for byte."""
    root = Path(root)
    hooks = sorted((root / "scripts" / "queue_hooks").glob("*prereg5*.sh"))
    if not hooks:
        return None
    if fz is not None:
        allowed = allowed_hashes(fz, root).get(HOOK_DRAFT_REL, set())
    else:
        allowed = {sha256(root / HOOK_DRAFT_REL)} if (root / HOOK_DRAFT_REL).is_file() else set()
    bad = [f"{p.relative_to(root).as_posix()} (sha256 {sha256(p)[:12]})" for p in hooks if sha256(p) not in allowed]
    return f"{', '.join(bad)} is not the hook draft byte for byte" if bad else None


def runtime_problem(rt: dict, versions: dict | None = None) -> str | None:
    bad = {k: rt.get(k) for k in ("torch", "transformers") if rt.get(k) != REGISTERED_RUNTIME[k]}
    if versions is not None and versions.get("peft") != REGISTERED_RUNTIME["peft"]:
        bad["peft"] = versions.get("peft")
    return f"runtime {bad} differs from the registered {REGISTERED_RUNTIME}" if bad else None


# ---------------------------------------------------------------- model-facing pieces (torch imported lazily)
class RealBackend:
    """The registered GPU path: rcwmt.runtime (lease, resources, loads) and PEFT as p3_run.py / t2bl_run.py use it."""
    device = "cuda"

    def __init__(self):
        import torch  # noqa: F401
        from rcwmt import runtime
        self.rt = runtime

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 PREREG-5 (T2b-ledger map + held-out-surface planning)")

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
            print("[prereg5] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)

    def wait(self) -> None:
        self.thermal_wait()
        self.rt.wait_for_resources()

    def load(self, cfg: Cfg):
        """(model, tok, info): Titan-Sparse + T2 s0 ('t2') + T2b-ledger s0-s2 ('t2bl0'..'t2bl2'), unmerged."""
        import peft
        from peft import PeftModel
        base, p = self.rt.load_arm("sparse")
        model = PeftModel.from_pretrained(base, str(cfg.t2s0_dir), adapter_name=ADAPTER_NAMES["t2s0"], is_trainable=False)
        for s in SEEDS:
            model.load_adapter(str(cfg.t2bl_dirs[s]), adapter_name=ADAPTER_NAMES[f"t2bl{s}"], is_trainable=False)
        model.eval()
        return model, self.rt.tokenizer(), {"fingerprint": p["param_fingerprint"], "versions": {"peft": peft.__version__}}

    def tokenizer(self):
        return self.rt.tokenizer()

    def unload(self) -> None:
        self.rt.unload(None)

    def runtime_info(self) -> dict:
        return self.rt.runtime_info()


@contextmanager
def side_ctx(model, side: str):
    """off = disable_adapter(); any other side = set_adapter(its PEFT name); the active side is read back from PEFT's own
    LoRA layer state before anything is generated, and no LoRA layer may be merged (D-28)."""
    if LF.any_merged(model):
        raise RuntimeError("a LoRA layer is merged: adapters must stay unmerged (D-28)")
    if side == "off":
        with model.disable_adapter():
            if LF.peft_side(model) != "off":
                raise RuntimeError("OFF side but PEFT reports an active adapter")
            yield
        return
    name = ADAPTER_NAMES[side]
    model.set_adapter(name)
    if LF.peft_side(model) != name:
        raise RuntimeError(f"side {side}: PEFT reports {LF.peft_side(model)!r}")
    yield


def gen_texts(model, tok, message_lists: list) -> list:
    from rcwmt import generation
    return generation.generate_texts(model, tok, message_lists, generation.max_tokens(C.LEDGER))


def map_messages(rows: list) -> list:
    return [C.messages(r["before"], r["event"], r["surface"]) for r in rows]


def ident_rows(cfg: Cfg) -> list:
    rows = C.load_jsonl(cfg.dev_bank)
    out = []
    for s in IDENT_STRATA:
        out += [r for r in rows if r["stratum"] == s][:cfg.ident_per_stratum]
    return out


def run_identity(model, tok, cfg: Cfg) -> tuple:
    """A0 (b): per T2b-ledger seed, the identity rows ON vs the dev screen's recorded answers (parsed states)."""
    rows = ident_rows(cfg)
    msgs = map_messages(rows)
    res, out_rows = {"pid": os.getpid(), "unix": time.time(), "n_rows": len(rows), "seeds": {}}, []
    for s in SEEDS:
        ref = {x["id"]: x["text"] for x in C.load_jsonl(cfg.dev_on[s])}
        with side_ctx(model, f"t2bl{s}"):
            tx = gen_texts(model, tok, msgs)
        same_s = same_t = 0
        for r, t in zip(rows, tx):
            a = C.parse_ledger(t, list(r["before"]))
            b = C.parse_ledger(ref.get(r["id"]), list(r["before"])) if ref.get(r["id"]) is not None else "missing"
            same_s += a == b
            same_t += t == ref.get(r["id"])
            out_rows.append({"seed": s, "id": r["id"], "text": t})
        res["seeds"][f"s{s}"] = {"identical_states": same_s / len(rows), "identical_texts": same_t / len(rows)}
    res["pass"] = all(v["identical_states"] >= cfg.ident_min for v in res["seeds"].values())
    return out_rows, res


def batchinv_keys(cfg: Cfg) -> list:
    dev = P.load_set(cfg.dev_set)
    return [(i["init"], a) for i in dev[:cfg.bi_n_inst] for a in i["vocab"]]


def make_f(arm: str, model, tok, cfg: Cfg):
    side, frame = PLAN_ARMS[arm]
    ident = cfg.identity(side)
    if frame == CANON:
        return LF.LedgerF("T2", model=model, tok=tok, adapter_name=ADAPTER_NAMES[side], identity=ident)
    return LF.SurfaceAuditF("T2", frame, model=model, tok=tok, adapter_name=ADAPTER_NAMES[side], identity=ident)


def run_batch_invariance(model, tok, keys: list, cfg: Cfg) -> tuple:
    """A0 (c): T2BL@canon and T2BL@HS, the keys as ONE predict list vs one key per predict call."""
    rows, res = [], {"pid": os.getpid(), "unix": time.time(), "n_keys": len(keys), "arms": {}}
    for arm in ("T2BL@canon", "T2BL@HS"):
        fb = make_f(arm, model, tok, cfg)
        rb = fb.predict(keys)
        f1 = make_f(arm, model, tok, cfg)
        r1 = [f1.predict([k])[0] for k in keys]
        res["arms"][arm] = {"identical_states": sum(a == b for a, b in zip(rb, r1)) / len(keys),
                            "identical_texts": sum(fb.texts[k] == f1.texts[k] for k in keys) / len(keys),
                            "generation_calls_batched": fb.n_generate_fn_calls,
                            "generation_calls_batch1": f1.n_generate_fn_calls,
                            "max_prompts_per_model_generate_batch1": max(f1.prompts_per_model_generate or [0])}
        rows += [{"arm": arm, "s": k[0], "a": k[1], "text_batched": fb.texts[k], "text_batch1": f1.texts[k]} for k in keys]
    res["pass"] = all(v["identical_states"] >= BI_MIN and v["generation_calls_batched"] == 1
                      and v["generation_calls_batch1"] == len(keys) and v["max_prompts_per_model_generate_batch1"] <= 1
                      for v in res["arms"].values())
    return rows, res


def run_a0(model, tok, cfg: Cfg, chains: list, keys: list) -> dict:
    """A0 (a), (b), (c) in that order; stops at the first failure. Returns the records and rows (nothing written)."""
    t0 = time.time()
    r_off, r_on, an = LR.run_anchor(model, tok, chains, cfg.anchor_refs)
    out = {"anchor": an, "anchor_off": LR.chain_rows(chains, r_off), "anchor_on": LR.chain_rows(chains, r_on)}
    if not an["pass"]:
        out["seconds"] = round(time.time() - t0, 1)
        return out
    out["ident_rows"], out["ident"] = run_identity(model, tok, cfg)
    if not out["ident"]["pass"]:
        out["seconds"] = round(time.time() - t0, 1)
        return out
    out["bi_rows"], out["bi"] = run_batch_invariance(model, tok, keys, cfg)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def a0_pass(r: dict) -> bool:
    return bool(r.get("anchor", {}).get("pass") and (r.get("ident") or {}).get("pass") and (r.get("bi") or {}).get("pass"))


def run_map_stage(model, tok, rows: list, out: Path) -> dict:
    msgs = map_messages(rows)
    texts, sides = {}, {}
    for side in MAP_SIDES:
        t0 = time.time()
        with side_ctx(model, side):
            texts[side] = gen_texts(model, tok, msgs)
        sides[side] = {"pid": os.getpid(), "start_unix": t0, "end_unix": time.time(), "n": len(rows)}
        if len(texts[side]) != len(rows):
            raise RuntimeError(f"map {side}: {len(texts[side])} texts for {len(rows)} rows")
    files = {FILES["M1"][f"map:{s}"]: C.dumps_rows([{"id": r["id"], "text": t} for r, t in zip(rows, texts[s])])
             for s in MAP_SIDES}
    files[FILES["M1"]["acct"]] = json.dumps({"stage": "M1", "sides": sides, "n_rows": len(rows)}, indent=1) + "\n"
    for rel, text in files.items():                  # written together, only after every side completed
        write_text_x(out / rel, text)
    return {"sides": sides, "n_new": len(rows) * len(MAP_SIDES)}


def run_planning_stage(st: str, BP, model, tok, cfg: Cfg, out: Path, insts: list, PI: list, own: dict) -> dict:
    ids = [i["id"] for i in insts]
    files, acct, sides, n_new = {}, {"stage": st, "configs": {}}, {}, 0
    for arm in STAGE_ARMS[st]:
        t0 = time.time()
        inner = make_f(arm, model, tok, cfg)
        F = LF.guarded(inner, insts)
        prop = PR.EnumerateProposer()
        eps = BP.plan_beam(PI, prop, F, eval_seed=EVAL_SEED, **PR.beam_kwargs(prop, W, E))
        need([e["id"] for e in eps] == ids, f"stage {st} {arm}: episode ids are not the draw's ids in order")
        files[FILES[st][f"eps:{arm}"]] = "".join(json.dumps(LR.ep_row(e)) + "\n" for e in eps)
        files[FILES[st][f"cache:{arm}"]] = LR.cache_text(inner._header(), None, inner)
        acct["configs"][arm] = LR.accounting(inner, F, own, ids, tok, time.time() - t0)
        sides[arm] = {"pid": os.getpid(), "start_unix": t0, "end_unix": time.time()}
        n_new += inner.model_queries
    files[FILES[st]["acct"]] = json.dumps(acct, indent=1) + "\n"
    for rel, text in files.items():
        write_text_x(out / rel, text)
    return {"sides": sides, "n_new": n_new}


def run_cpu_stage(BP, out: Path, insts: list, PI: list, own: dict) -> dict:
    ids = [i["id"] for i in insts]
    files, acct = {}, {"stage": "C1", "configs": {}}
    for arm in ("ORACLE", "IGNORE_STATUS"):
        t0 = time.time()
        inner = LF.LedgerF(arm)
        F = LF.guarded(inner, insts)
        prop = PR.EnumerateProposer()
        eps = BP.plan_beam(PI, prop, F, eval_seed=EVAL_SEED, **PR.beam_kwargs(prop, W, E))
        need([e["id"] for e in eps] == ids, f"stage C1 {arm}: episode ids are not the draw's ids in order")
        files[FILES["C1"][f"eps:{arm}"]] = "".join(json.dumps(LR.ep_row(e)) + "\n" for e in eps)
        acct["configs"][arm] = LR.accounting(inner, F, own, ids, None, time.time() - t0)
    files[FILES["C1"]["acct"]] = json.dumps(acct, indent=1) + "\n"
    for rel, text in files.items():
        write_text_x(out / rel, text)
    return {"pid": os.getpid()}


# ---------------------------------------------------------------- inputs (sealed sets)
def load_map(cfg: Cfg) -> list:
    src = C.read_json(cfg.map_source) or {}
    need(Path(cfg.map_bank).is_file(), f"the map bank {cfg.map_bank} is missing")
    need(sha256(cfg.map_bank) == src.get("file_sha256"), "the map bank differs from its SOURCE record")
    rows = C.load_jsonl(cfg.map_bank)
    need({r["stratum"] for r in rows} == set(C.CONFIRM_STRATA), "the map bank's strata are not the registered ones")
    return rows


def load_plan(cfg: Cfg) -> list:
    src = C.read_json(cfg.plan_source) or {}
    need(Path(cfg.plan_draw).is_file(), f"the planning draw {cfg.plan_draw} is missing")
    need(sha256(cfg.plan_draw) == src.get("file_sha256"), "the planning draw differs from its SOURCE record")
    insts = P.load_set(cfg.plan_draw)
    need(P.set_sha(insts) == src.get("file_sha256"), "the planning draw does not re-serialise to its sha256")
    need(all(i["range"] == "in" for i in insts), "the planning draw contains a non-primary instance")
    return insts


def check_inputs(cfg: Cfg, fz: dict | None) -> dict:
    shas = {"t2s0": {f: sha256(Path(cfg.t2s0_dir) / f) for f in ADAPTER_FILES}}
    for s in SEEDS:
        shas[f"t2bl{s}"] = {f: sha256(Path(cfg.t2bl_dirs[s]) / f) for f in ADAPTER_FILES}
    if fz is not None:
        ins = fz["inputs"]
        need(sha256(cfg.map_bank) == ins[MAP_BANK_REL], "the map bank differs from the freeze")
        need(sha256(cfg.plan_draw) == ins[PLAN_DRAW_REL], "the planning draw differs from the freeze")
        for f in ADAPTER_FILES:
            need(shas["t2s0"][f] == ins[f"{T2S0_REL}/{f}"], f"T2 s0 {f} differs from the freeze")
            for s in SEEDS:
                need(shas[f"t2bl{s}"][f] == ins[f"{T2BL_RELS[s]}/{f}"], f"T2b-ledger s{s} {f} differs from the freeze")
        for s, p in cfg.anchor_refs.items():
            need(sha256(p) == ins[ANCHOR_REF_RELS[s]], f"anchor reference {s} differs from the freeze")
        need(sha256(cfg.dev_set) == ins[DEV_SET_REL], "dev2_primary (batch-invariance keys) differs from the freeze")
        need(sha256(cfg.dev_bank) == ins["data/t2bl_dev.jsonl"], "the T2b-ledger dev bank differs from the freeze")
        for s in SEEDS:
            need(sha256(cfg.dev_on[s]) == ins[DEV_ON_RELS[s]], f"the dev screen's s{s} answers differ from the freeze")
    else:
        need(shas["t2s0"]["adapter_model.safetensors"] is not None, "T2 s0 adapter missing", persist=False)
    return shas


def measured_a0_s(prov: dict, n: dict, cfg: Cfg) -> float:
    xs = [float((prov.get("stages") or {}).get("A0", {}).get("seconds") or 0.0)]
    xs += [float(t.get("a0_recheck_seconds") or 0.0) for t in prov.get("turns") or []]
    m = max(xs)
    return m if m > 0 else stage_est_s("A0", n, cfg.rate, cfg.map_rate)


# ---------------------------------------------------------------- attempts (reboot-aware)
def current_boot_time() -> float:
    import psutil
    return float(psutil.boot_time())


def new_attempt(turn: int | None, where: str) -> dict:
    return {"turn": turn, "where": where, "pid": os.getpid(), "boot_time": current_boot_time(), "start_unix": time.time()}


@contextmanager
def attempt_guard(att: dict, save):
    """Records an exception that ends a stage attempt (then re-raises): such an attempt counts as failed."""
    try:
        yield
    except BaseException as e:                       # noqa: BLE001  (recorded, then re-raised unchanged)
        att["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
        att["end_unix"] = time.time()
        try:
            save()
        except Exception:                            # noqa: BLE001
            pass
        raise


def _pid_alive(pid) -> bool:
    import psutil
    try:
        return int(pid) != os.getpid() and psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return False


def classify_attempts(attempts: list, boot: float) -> tuple:
    """(failed, interrupted_by_reboot) among the attempts of a stage that is not recorded. Failed: a recorded exception,
    or a process that died without one in the current boot (a native crash, a kill); interrupted: an attempt from an
    earlier boot without a recorded exception (cut off by a reboot). Legacy entries (not dicts) count as failed."""
    failed, interrupted = [], []
    for a in attempts:
        if not isinstance(a, dict):
            failed.append(a)
        elif a.get("exception"):
            failed.append(a)
        elif a.get("boot_time") is not None and abs(float(a["boot_time"]) - boot) > REBOOT_TOL_S:
            interrupted.append(a)
        elif not _pid_alive(a.get("pid")):
            failed.append(a)
    return failed, interrupted


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = Path(cfg.out)
    fz = None
    if cfg.require_freeze:
        ok, errs, fz = verify_freeze(cfg.root)
        if not ok:
            raise Stop("refusing to run the PREREG-5 job: " + "; ".join(errs), persist=(Path(cfg.root) / FREEZE_REL).exists())
    hp = hook_problem(cfg.root, fz)
    if hp:
        raise Stop(f"[prereg5] {hp}; reinstall it byte for byte from {HOOK_DRAFT_REL} (or remove it) before any run",
                   persist=False)
    if (out / "DONE.json").exists():
        print(f"[prereg5] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; an operator must investigate, "
                   "log it in DECISIONS.md and move the file aside before any rerun", persist=False)
    for name in ("anchor", "ident", "bi"):
        rec = out / FILES["A0"][name]
        if rec.exists() and not (C.read_json(rec) or {}).get("pass"):
            raise Stop(f"the A0 {name} check failed in an earlier turn (deviation policy: investigate, log)")
    if list((out / "anchor").glob("RECHECK_FAILED_*.json")):
        raise Stop("an A0 re-check failed in an earlier turn (deviation policy: investigate, log)")
    shas = check_inputs(cfg, fz)
    rows, insts = load_map(cfg), load_plan(cfg)
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[prereg5] {ow}: one-waiter rule (R14). Start PREREG-5 only when no other RCWM GPU job is running or "
                   "waiting", persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, fz, rows, insts, shas)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg: Cfg, backend, out: Path, fz, rows: list, insts: list, shas: dict) -> int:
    prov_path = out / "provenance.json"
    prov = C.read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
        need(prov.get("map_sha256") == sha256(cfg.map_bank) and prov.get("plan_sha256") == sha256(cfg.plan_draw),
             "provenance belongs to another map bank or planning draw")
        need(prov.get("adapters") == shas, "adapter files differ from the run's first invocation")
    else:
        prov = {"map_sha256": sha256(cfg.map_bank), "plan_sha256": sha256(cfg.plan_draw),
                "freeze_prereg_sha256": (fz or {}).get("prereg_sha256"), "registered": bool(cfg.require_freeze),
                "lease_label": LEASE_LABEL, "adapters": shas, "stages": {}, "turns": [], "cpu_runs": []}
    for k, v in (("attempts", {}), ("orphaned", []), ("no_progress_turns", 0)):
        prov.setdefault(k, v)
    prov["anchor_refs"] = {s: sha256(p) for s, p in cfg.anchor_refs.items()}
    prov["code_sha256"] = {r: sha256(ROOT / r) for r in PROV_CODE if (ROOT / r).is_file()}
    prov["bwm_planner_sha256"] = sha256(BWM_ROOT / BWM_PLANNER_REL)
    need(prov["bwm_planner_sha256"] == BWM_PLANNER_SHA, "BWM's planner differs from the pinned sha256")

    def save():
        tmp = _partial(prov_path)
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(prov, indent=1) + "\n")
        os.replace(tmp, prov_path)

    states = stage_states(prov, out)
    for st, s in states.items():
        if s == "partial":
            d = orphan(st, out)
            prov["orphaned"].append({"stage": st, "dir": rel_out(d, out), "unix": time.time()})
            prov["stages"].pop(st, None)
            states[st] = "todo"
    done = {st for st, s in states.items() if s == "done"}
    boot = current_boot_time()
    for st in STAGE_ORDER:                                  # a started attempt that was never recorded: failed or interrupted
        if st in done:
            continue
        failed, interrupted = classify_attempts(prov["attempts"].get(st) or [], boot)
        prov.setdefault("attempts_classified", {})[st] = {"failed": len(failed), "interrupted_by_reboot": len(interrupted)}
        if len(failed) >= MAX_FAILED:
            save()
            raise Stop(f"[prereg5] stage {st} failed {len(failed)} attempts (limit {MAX_FAILED}; {len(interrupted)} more "
                       "were cut off by a reboot and not counted): stop (deviation policy: investigate, log)")

    def record(st: str, extra: dict) -> None:
        prov["stages"][st] = {"name": st, "turn": len(prov["turns"]) - 1, **extra,
                              "files": {rel_out(p, out): sha256(p) for p in stage_required(st, out)},
                              "finished_unix": time.time()}
        done.add(st)
        save()

    def finalize() -> int:
        need(set(prov["stages"]) == set(STAGE_ORDER), "finalize: not every stage is recorded")
        need(all(stage_states(prov, out)[s] == "done" for s in prov["stages"]), "finalize: a recorded stage changed")
        save()
        write_json_x(out / "DONE.json", {"finished_unix": time.time(), "stages": list(STAGE_ORDER),
                                         "files": {st: prov["stages"][st]["files"] for st in STAGE_ORDER},
                                         "provenance_sha256": sha256(prov_path)}, stale_partial="aside")
        print("[prereg5] all stages complete; DONE.json written", flush=True)
        return 0

    todo = [st for st in STAGE_ORDER if st not in done]
    print(f"[prereg5] stages to run: {todo}", flush=True)
    save()
    if not todo:
        return finalize()
    PI, own = P.planner_instances(insts), P.owner_of_action(insts)
    BP = LR.import_plan_beam()
    bwm_mods = LR.bwm_module_hashes()
    n = counts(len(rows), len(insts), len(cfg.anchor_chains) if cfg.anchor_chains is not None else N_ANCHOR_CHAINS,
               cfg.bi_n_inst, cfg.ident_per_stratum * len(IDENT_STRATA) * len(SEEDS))

    if "A0" in done and "C1" not in done:     # A0 passed in an EARLIER invocation: C1 first, before any lease or CUDA use
        att = new_attempt(None, "cpu")
        prov["attempts"].setdefault("C1", []).append(att)
        save()
        cuda0 = LR.cuda_initialized()
        need(not cuda0, "C1 would run in a process that has initialised CUDA")
        t0 = time.time()
        with attempt_guard(att, save):
            run_cpu_stage(BP, out, insts, PI, own)
        prov["cpu_runs"].append({"pid": os.getpid(), "start_unix": t0, "end_unix": time.time(),
                                 "code_sha256": dict(prov["code_sha256"]), "bwm_planner_sha256": prov["bwm_planner_sha256"],
                                 "bwm_modules_sha256": bwm_mods, "lease_held": False, "cuda_initialized_before": cuda0,
                                 "cuda_initialized_after": LR.cuda_initialized()})
        record("C1", {"seconds": round(time.time() - t0, 1), "pid": os.getpid(), "after_a0": True, "lease_held": False})
        print(f"[prereg5] C1 (CPU: ORACLE / IGNORE_STATUS) done in {time.time() - t0:.0f}s", flush=True)
    gpu_todo = [st for st in GPU_STAGES if st not in done]
    turn = None
    if gpu_todo:
        chains = cfg.anchor_chains if cfg.anchor_chains is not None else LR.pilot_anchor_chains()
        keys = batchinv_keys(cfg)
        msgs_ok = sum(1 for r in rows if r["surface"]) == len(rows)
        need(msgs_ok, "a map row has no surface record")
        with backend.lease():
            t_acq, t_src = backend.lease_acquired_unix()
            cap_s = cfg.cap_h * 3600
            turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                    "code_sha256": dict(prov["code_sha256"]), "bwm_planner_sha256": prov["bwm_planner_sha256"],
                    "bwm_modules_sha256": bwm_mods, "cap_h": cfg.cap_h, "rate": cfg.rate, "map_rate": cfg.map_rate,
                    "stages_run": [], "deferred": [], "device": backend.device}
            prov["turns"].append(turn)
            t_idx = len(prov["turns"]) - 1
            save()

            def start_est(st: str) -> float:
                return start_estimate_s(st, n, cfg.rate, cfg.map_rate, measured_planning_spg(prov))

            def ok_to_start(st: str) -> bool:
                return G2.can_start(time.time() - t_acq, start_est(st), 0.0, cap_s)

            extra = LOAD_S + (measured_a0_s(prov, n, cfg) if "A0" in done else 0.0)
            turn["start_check"] = {"load_s": LOAD_S, "a0_s": round(extra - LOAD_S, 1), "first_stage": gpu_todo[0],
                                   "first_stage_estimate_s": round(start_est(gpu_todo[0]), 1),
                                   "measured_planning_spg": measured_planning_spg(prov)}
            if not G2.can_start(time.time() - t_acq + extra, start_est(gpu_todo[0]), 0.0, cap_s):
                turn["deferred"] = list(gpu_todo)
                print(f"[prereg5] cap: deferring {turn['deferred']} to the next turn (model not loaded)", flush=True)
                save()
                gpu_todo = []
            if gpu_todo:
                backend.wait()
                t1 = time.time()
                ad = {"t2s0": {f: sha256(Path(cfg.t2s0_dir) / f) for f in ADAPTER_FILES},
                      **{f"t2bl{s}": {f: sha256(Path(cfg.t2bl_dirs[s]) / f) for f in ADAPTER_FILES} for s in SEEDS}}
                need(ad == prov["adapters"], "adapter hashes at load differ from the run's record")
                model, tok, info = backend.load(cfg)
                try:
                    need(str(info["fingerprint"]).startswith(FP["sparse"]),
                         f"Titan-Sparse fingerprint {str(info['fingerprint'])[:12]} != {FP['sparse']}")
                    need(not LF.any_merged(model), "a LoRA layer is merged after load (D-28)")
                    turn["adapter_sha256_at_load"] = ad
                    turn["fingerprint"] = info["fingerprint"]
                    turn["versions"] = info["versions"]
                    turn["runtime"] = backend.runtime_info()
                    rp = runtime_problem(turn["runtime"], info["versions"])
                    if rp:
                        save()
                        raise Stop(f"[prereg5] {rp}: restore the registered environment (nothing was generated)")
                    turn["load_s"] = round(time.time() - t1, 1)
                    save()
                    if "A0" in done:                              # every resumed turn re-runs A0 before any stage
                        t_r = time.time()
                        r = run_a0(model, tok, cfg, chains, keys)
                        turn["a0_recheck"] = {k: r.get(k) for k in ("anchor", "ident", "bi")}
                        turn["a0_recheck_seconds"] = round(time.time() - t_r, 1)
                        save()
                        if not a0_pass(r):
                            write_json_x(out / "anchor" / f"RECHECK_FAILED_{int(time.time())}.json",
                                         {"turn": t_idx, **turn["a0_recheck"]})
                            raise Stop("[prereg5] A0 RE-CHECK FAILED; stop before any stage of this turn (deviation policy)")
                    for i, st in enumerate(gpu_todo):
                        if not ok_to_start(st):
                            turn["deferred"] = gpu_todo[i:]
                            print(f"[prereg5] cap: deferring {turn['deferred']} to the next turn", flush=True)
                            save()
                            break
                        att = new_attempt(t_idx, "gpu")
                        prov["attempts"].setdefault(st, []).append(att)
                        save()
                        t0 = time.time()
                        with attempt_guard(att, save):
                            if st == "A0":
                                r = run_a0(model, tok, cfg, chains, keys)
                                write_text_x(out / FILES["A0"]["anchor_off"], C.dumps_rows(r["anchor_off"]))
                                write_text_x(out / FILES["A0"]["anchor_on"], C.dumps_rows(r["anchor_on"]))
                                write_json_x(out / FILES["A0"]["anchor"], r["anchor"])
                                prov["anchor"] = r["anchor"]
                                if not r["anchor"]["pass"]:
                                    save()
                                    raise Stop(f"[prereg5] ANCHOR FAILED {r['anchor'].get('off')} {r['anchor'].get('on')}: the "
                                               "instrument does not reproduce T2 s0; stop before any output (deviation policy)")
                                write_text_x(out / FILES["A0"]["ident_rows"], C.dumps_rows(r["ident_rows"]))
                                write_json_x(out / FILES["A0"]["ident"], r["ident"])
                                prov["adapter_identity"] = r["ident"]
                                if not r["ident"]["pass"]:
                                    save()
                                    raise Stop(f"[prereg5] ADAPTER IDENTITY FAILED {r['ident']['seeds']}: a loaded T2b-ledger "
                                               "adapter does not reproduce the dev screen's answers; stop before any output")
                                write_text_x(out / FILES["A0"]["bi_rows"], C.dumps_rows(r["bi_rows"]))
                                write_json_x(out / FILES["A0"]["bi"], r["bi"])
                                prov["batch_invariance"] = r["bi"]
                                if not r["bi"]["pass"]:
                                    save()
                                    raise Stop(f"[prereg5] BATCH INVARIANCE FAILED {r['bi']['arms']}: stop before any output")
                                record("A0", {"seconds": round(time.time() - t0, 1), "pid": os.getpid(),
                                              "generations": stage_gens("A0", n)})
                            elif st == "M1":
                                res = run_map_stage(model, tok, rows, out)
                                record("M1", {"seconds": round(time.time() - t0, 1), "pid": os.getpid(), "sides": res["sides"],
                                              "generations": res["n_new"]})
                            else:
                                res = run_planning_stage(st, BP, model, tok, cfg, out, insts, PI, own)
                                record(st, {"seconds": round(time.time() - t0, 1), "pid": os.getpid(), "sides": res["sides"],
                                            "generations": res["n_new"],
                                            "generations_planned_expected": round(stage_gens(st, n)),
                                            "generations_planned_ceiling": round(stage_gens(st, n, "ceiling")),
                                            "start_estimate_s": round(start_est(st), 1)})
                        turn["stages_run"].append(st)
                        save()
                        print(f"[prereg5] {st} done in {time.time() - t0:.0f}s", flush=True)
                finally:
                    del model
                    backend.unload()
            turn["released_unix"] = time.time()
            turn["no_progress"] = not turn["stages_run"]
            prov["no_progress_turns"] = (prov.get("no_progress_turns") or 0) + 1 if turn["no_progress"] else 0
            save()
    if "A0" in done and "C1" not in done:
        print("[prereg5] C1 (CPU) runs at the start of the next invocation, before the lease and before any CUDA use",
              flush=True)
    if turn is not None and turn["no_progress"] and prov["no_progress_turns"] >= NO_PROGRESS_STOP:
        raise Stop(f"[prereg5] no stage was recorded in {prov['no_progress_turns']} consecutive turns (deferred "
                   f"{turn['deferred']}): stop (deviation policy; a cap or rate amendment is needed)")
    if done >= set(STAGE_ORDER):
        return finalize()
    print(f"[prereg5] remaining stages {[s for s in STAGE_ORDER if s not in done]} resume next turn", flush=True)
    return 5


def cap_from_arg(x) -> float:
    """--cap-h may lower the turn cap (tests, a short turn), never raise it above the agreed 4.5 h."""
    return min(float(x), CAP_H)


def stop_record(out: Path, e: Stop) -> dict:
    rec = {"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)}
    pv = Path(out) / "provenance.json"
    prov = C.read_json(pv) if pv.exists() else None
    parts = eligible_parts(prov, out)
    if prov:
        rec.update({"provenance_sha256": sha256(pv), "stages_recorded": [s for s in STAGE_ORDER if s in prov.get("stages", {})]})
    rec["eligible_parts"] = parts
    rec["verdict_eligible"] = bool(parts)
    return rec


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        rec = stop_record(out, e)
        with open(Path(out) / "STOPPED.json", "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
    except FileExistsError:
        pass


def run_main(cfg: Cfg, backend) -> int:
    """The job's exit-code mapping (main() and the tests): 0 / 5 from run(); a Stop (this module's or lbeam_run's) ->
    printed, persisted when it concerns the run, exit 3; any other exception propagates (exit 1, the hook retries once)."""
    try:
        return run(cfg, backend)
    except (Stop, LR.Stop) as e:
        e2 = e if isinstance(e, Stop) else Stop(str(e), getattr(e, "persist", True))
        print(str(e2), flush=True)
        persist_stop(Path(cfg.out), e2)
        return 3


# ---------------------------------------------------------------- part (c): the retention guard binding (CPU)
RET_CLAIM_RECORDS = ("docs/retention/BWM_RELEASE_DRAW.txt",)   # other consumers' records of the confirmation draw they use


def claimed_draws(root: Path, exclude: Path | None = None) -> set:
    """Draw ids that another consumer has recorded as its own (hook 85's BWM release draw, any other run's RET_DRAW)."""
    root = Path(root)
    out: set = set()
    for rel in RET_CLAIM_RECORDS:
        p = root / rel
        if p.is_file():
            out |= set(re.findall(r"confirm_[0-9]+", p.read_text(encoding="utf-8", errors="ignore")))
    runs = root / "runs"
    for p in (runs.glob("*/RET_DRAW.json") if runs.is_dir() else []):
        if exclude is not None and p.resolve() == Path(exclude).resolve():
            continue
        d = (C.read_json(p) or {}).get("draw_id")
        if d:
            out.add(str(d))
    return out


def ret_mint(root: Path | None = None, require_freeze: bool = True, mint_fn=None) -> dict:
    """Mint the 5c confirmation draw ONCE, at run time, after the freeze; record it (create-only). An interrupted mint
    is recovered from the intent record: a confirmation seal written after the intent (seal file time) and claimed by
    no other consumer is adopted if it is the only one; if none exists and no draw file without a seal appeared after
    the intent (a crash before the mint wrote anything), a fresh mint is allowed (recorded in the intent's retries);
    anything else stops for the operator."""
    root = Path(root or ROOT)
    if require_freeze:
        ok, errs, _ = verify_freeze(root)
        need(ok, "PREREG-5 is not frozen or does not verify: " + "; ".join(errs[:3]), persist=False)
    rec_p, intent_p = root / RET_DRAW_REL, root / RET_INTENT_REL
    need(not rec_p.exists(), f"{RET_DRAW_REL} exists: the PREREG-5 retention draw is minted once", persist=False)
    draws = root / "docs" / "retention" / "draws"
    now = sorted(p.name for p in draws.glob("confirm_*.SEAL.json")) if draws.is_dir() else []
    if mint_fn is None:
        import ret_gen as RGEN
        mint_fn = lambda: RGEN.mint("confirm", root=root)  # noqa: E731
    seal, how = None, None
    if intent_p.exists():
        intent = C.read_json(intent_p) or {}
        t_int = float(intent.get("unix") or 0.0)
        before = set(intent.get("confirm_seals_before") or [])
        claimed = claimed_draws(root, exclude=rec_p)
        new = [n for n in sorted(set(now) - before) if (draws / n).stat().st_mtime >= t_int - 2.0
               and n[: -len(".SEAL.json")] not in claimed]
        need(len(new) <= 1, f"an interrupted retention mint left {len(new)} unclaimed new confirmation seals: "
                            "investigate, log", persist=False)
        if new:
            seal = C.read_json(draws / new[0]) or {}
            how = "adopted after an interrupted mint (intent record; seal written after the intent, claimed by no one)"
        else:
            orphans = [p.name for p in draws.glob("confirm_*.jsonl") if p.stat().st_mtime >= t_int - 2.0 and
                       not p.with_name(p.name[: -len(".jsonl")] + ".SEAL.json").exists()] if draws.is_dir() else []
            need(not orphans, f"an interrupted retention mint left draw files without a seal {orphans}: move them aside "
                              "unread, log it in DECISIONS.md", persist=False)
            intent.setdefault("retries", []).append({"unix": time.time(), "confirm_seals_before": now})
            tmp = intent_p.with_name(intent_p.name + ".partial")
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(intent, indent=1) + "\n")
            os.replace(tmp, intent_p)
            seal = mint_fn()
            how = "minted by --ret-mint after an interrupted mint that wrote nothing (intent retry)"
    else:
        C.write_create(intent_p, json.dumps({"confirm_seals_before": now, "unix": time.time()}, indent=1) + "\n",
                       stale_partial="aside")
        seal = mint_fn()
        how = "minted by --ret-mint"
    need(seal.get("purpose") == "confirm" and seal.get("draw_id"), "the minted seal is not a confirmation draw seal")
    rec = {"record": "PREREG5_RET_DRAW", "draw_id": seal["draw_id"], "file": seal.get("file"),
           "file_sha256": seal.get("file_sha256"), "seal_sha256": sha256(draws / f"{seal['draw_id']}.SEAL.json")
           if (draws / f"{seal['draw_id']}.SEAL.json").is_file() else None, "minted_local": seal.get("minted_local"),
           "how": how, "freeze_prereg_sha256": (C.read_json(root / FREEZE_REL) or {}).get("prereg_sha256"),
           "run_cmd": (f"ret_run.py --mode confirm --draw {seal.get('file')} --extra-adapter "
                       f"{RET_EXTRA_NAME}={T2BL_RELS[0]}")}
    C.write_create(rec_p, json.dumps(rec, indent=1) + "\n", stale_partial="aside")
    print(f"[prereg5] retention draw {seal['draw_id']} recorded in {RET_DRAW_REL}", flush=True)
    return rec


def ret_record(root: Path | None = None, require_freeze: bool = True) -> dict:
    """Bind the retention run of the recorded draw to PREREG-5 (create-only pointer): its registered result, or its
    persisted stop. Checks: the run's draw is the recorded draw; side x_t2bl_s0 is T2b-ledger s0's frozen adapter."""
    root = Path(root or ROOT)
    fz = None
    if require_freeze:
        ok, errs, fz = verify_freeze(root)
        need(ok, "PREREG-5 is not frozen or does not verify: " + "; ".join(errs[:3]), persist=False)
    ptr = root / RETENTION_RESULT_POINTER
    need(not ptr.exists(), f"{RETENTION_RESULT_POINTER} exists (create-only)", persist=False)
    rd = C.read_json(root / RET_DRAW_REL)
    need(rd is not None, f"{RET_DRAW_REL} missing: run --ret-mint first", persist=False)
    run_dir = root / "runs" / "retention" / rd["draw_id"]
    prov = C.read_json(run_dir / "provenance.json") or {}
    stopped = C.read_json(run_dir / "STOPPED.json")
    res_p = run_dir / "analysis" / "RET_RESULTS.json"
    rec = {"record": "PREREG5_RETENTION", "draw_id": rd["draw_id"], "run": str(run_dir.relative_to(root)).replace("\\", "/")}
    if res_p.is_file():
        need(prov.get("draw_sha256") == rd["file_sha256"], "the retention run used another draw", persist=False)
        ad = (prov.get("adapters") or {}).get(f"x_{RET_EXTRA_NAME}") or {}
        want = {f: (fz["inputs"][f"{T2BL_RELS[0]}/{f}"] if fz else sha256(root / T2BL_RELS[0] / f)) for f in ADAPTER_FILES}
        need(ad.get("sha256") == want, "the retention run's x_t2bl_s0 adapter is not T2b-ledger s0's frozen adapter",
             persist=False)
        res = C.read_json(res_p) or {}
        need(res.get("draw_id") == rd["draw_id"] and f"x_{RET_EXTRA_NAME}_vs_off" in (res.get("comparisons") or {}),
             "the retention result is not this draw's or lacks x_t2bl_s0_vs_off", persist=False)
        rec.update({"status": "analysed", "results": str(res_p.relative_to(root)).replace("\\", "/"),
                    "results_sha256": sha256(res_p)})
    elif stopped:
        rec.update({"status": "stopped", "reason": str(stopped.get("reason", ""))[:300],
                    "stopped_sha256": sha256(run_dir / "STOPPED.json")})
    else:
        raise Stop("the retention run has neither a registered result nor a persisted stop yet", persist=False)
    C.write_create(ptr, json.dumps(rec, indent=1) + "\n", stale_partial="aside")
    print(f"[prereg5] retention {rec['status']}: {RETENTION_RESULT_POINTER} written", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="CPU only, no torch import: projected generations, GPU time, turns")
    g.add_argument("--verify", action="store_true", help="CPU only: verify PREREG_5_FREEZE.json (+ amendments)")
    g.add_argument("--freeze", action="store_true", help="CPU only: freeze PREREG-5 and mint the sealed sets")
    g.add_argument("--ret-mint", action="store_true", help="CPU only: mint and record the 5c confirmation draw (once)")
    g.add_argument("--ret-record", action="store_true", help="CPU only: bind the retention result (or stop) to PREREG-5")
    g.add_argument("--union-digest", action="store_true", help="CPU only: rebuild the freshness union, print its digest")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--bwm-cosign", default=None)
    ap.add_argument("--bwm-salt", default=None)
    ap.add_argument("--changes", default=None, help="JSON file: the list of changes since the dev gate's freeze")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    cap_h = cap_from_arg(a.cap_h)
    if not (a.plan or a.verify or a.freeze or a.ret_mint or a.ret_record or a.union_digest):
        return run_main(Cfg(cap_h=cap_h), RealBackend())
    try:
        if a.plan:
            res = plan()
            ok, errs, _ = verify_freeze()
            res["checks"] = {"pins_match": all((ROOT / r).is_file() and sha256(ROOT / r) == h for r, h in PINS.items()),
                             "bwm_planner_pinned": sha256(BWM_ROOT / BWM_PLANNER_REL) == BWM_PLANNER_SHA,
                             "prereg5_outputs_present": prereg5_outputs(ROOT), "freeze_ok": ok, "freeze_blockers": errs[:3],
                             "torch_imported": "torch" in sys.modules}
            res["code_sha256"] = {"scripts/t2bl_confirm_run.py": sha256(Path(__file__))}
            need("torch" not in sys.modules, "--plan must not import torch", persist=False)
            if (ROOT / FREEZE_REL).exists():
                print("PREREG-5 is frozen: P5_GEN_PLAN.json is a frozen input and is not rewritten")
                return 0
            print("wrote", C.write_doc("P5_GEN_PLAN.json", json.dumps(res, indent=1) + "\n"))
            return 0
        if a.verify:
            ok, errs, _ = verify_freeze()
            print("PREREG-5 freeze verifies" if ok else "PREREG-5 freeze does NOT verify:\n  " + "\n  ".join(errs))
            return 0 if ok else 1
        if a.ret_mint:
            ret_mint()
            return 0
        if a.ret_record:
            ret_record()
            return 0
        if a.union_digest:
            import t2bl_make_data as MD
            excl = MD.build_union(root=ROOT, skip_prefix=None)
            need(not MD.union_errors(excl), f"union sources could not be read: {MD.union_errors(excl)}", persist=False)
            print(json.dumps({"union_digest": CB.union_digest(excl), "n_payloads": len(excl["pay"]),
                              "n_operative": len(excl["ops"]), "n_sources": len(excl["sources"]),
                              "computed_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                              "quote_in_cosign_as": "union:" + CB.union_digest(excl)}, indent=1))
            return 0
        changes = None
        if a.changes:
            changes = C.read_json(Path(a.changes))
            need(isinstance(changes, list), "--changes must name a JSON file holding a list of changes", persist=False)
        freeze(a.approval or "", a.bwm_cosign or "", a.bwm_salt or "", changes=changes)
        return 0
    except (Stop, LR.Stop) as e:
        print(str(e), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
