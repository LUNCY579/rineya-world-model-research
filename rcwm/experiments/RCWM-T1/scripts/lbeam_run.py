"""PREREG-LB runner, freeze tool and freeze verifier (joint roadmap item 3; docs/ledger_beam/PREREG_LB_DRAFT.md sections
7, 9, 10). The GPU job refuses to run before PREREG-LB is frozen. Modelled on scripts/p3_run.py (PREREG-3) and
scripts/confirm2_gen.py (PREREG-2), whose torch-free helpers (pilot-chain loader, D-22 stage rule, registered runtime,
T2 s0 pins) it imports.

  python -B scripts/lbeam_run.py --plan     # CPU only, no torch import: generations, GPU time and turns per stage;
                                            # writes docs/ledger_beam/GEN_PLAN_LB.json (refused once frozen)
  python -B scripts/lbeam_run.py --verify   # CPU only: PREREG_LB_FREEZE.json (+ PREREG_LB_AMENDMENT_*) against the files
  python -B scripts/lbeam_run.py --freeze --approval "<who, when, DECISIONS entry>" \
         --bwm-cosign "<BWM's co-sign: reference, this draft's full sha256, commit:<sha256 of BWM's salt>>" \
         --bwm-salt "<the salt BWM reveals at the freeze>"
                                            # CPU only. Refuses if frozen (or a freeze was interrupted), if ANY lbeam
                                            # output or a confirmation draw exists, if a pre-freeze check fails, if the
                                            # co-sign does not quote this draft's sha256 and a salt commitment the salt
                                            # matches, if BWM's draft is not the cited one, if a source of the freshness
                                            # union errs or data/ changes while the union is built. Rebuilds the union,
                                            # MINTS the confirmation draw from the recorded seed rule (draft sha256, union
                                            # digest, BWM's salt; sealed by sha256; its instances are never printed),
                                            # hashes every input, writes PREREG_LB.md + PREREG_LB_FREEZE.json (all
                                            # .partial first, then renamed; create-only)
  python -B scripts/lbeam_run.py            # the registered job

Job. Before anything: the freeze must verify; an installed scripts/queue_hooks/70_lbeam.sh must be the frozen hook draft
byte for byte; no other RCWM job may hold or wait for the GPU lease (one-waiter rule, R14); a one-runner lock
(runs/lbeam/.run.lock). GPU stages run under ONE lease 'rcwm-t1:lbeam' (rcwmt.runtime.gpu_lease: alternation, thermal
flag, duty cycle, RAM floor), the 4.5 h turn cap from lock acquisition and the D-22 stage rule. One model: Titan-Sparse
FP32 + the T2 s0 LoRA unmerged (lbeam_f.load_registered_t2, exactly as confirm2_gen loads it); T2 = adapter 't2',
OFF = model.disable_adapter() in the same process.
  A0  instrument identity, BEFORE any planning output:
      (a) PREREG-2's anchor exactly as confirm2_gen defines it: pilot TEST ledger chain_one (208 chains,
          generation.chain) OFF and ON; identical states on >= 99% of chains per side against runs/t2/t2_s0
          gen_chain_one_base.jsonl (OFF) and gen_chain_one.jsonl (ON);
      (b) batch invariance: 256 dev keys (init x vocabulary of the first 32 dev2_primary instances; never a
          confirmation key) per arm, T2 and OFF, as ONE batched predict list vs batch 1 (one key per generation call);
          identical parsed states on >= 99% of keys per arm.
      Either failure stops the job (STOPPED.json, exit 3) before any planning output. Every later turn that loads the
      model re-runs (a) and (b) before any planning stage; a failure is persisted and final.
  G1  PRIMARY: plan_beam enum W2 E12 on the sealed draw, T2 then OFF, same process, written together
  G2  select_fixed T2 and OFF (PREREG-2 replication; report-only)
  G3  plan_beam enum W4 E24, T2 (report-only curve point)
  G4  SurfaceAuditF('T2', 'heldout_combined') plan_beam enum W2 E12 (D-35 audit; report-only)
  C1  CPU, never under the lease, only after A0 passed: ORACLE enum W2 E12 and W4 E24, IGNORE_STATUS enum W2 E12,
      select_fixed ORACLE and IGNORE_STATUS (exact / rule F: no model, no GPU). It runs only at the START of an
      invocation whose A0 passed in an earlier invocation: before the lease, in a process that has not initialised CUDA
      (checked and recorded). The invocation that records A0 therefore ends with exit 5 if C1 is still to run, even when
      every GPU stage is done; the hook's one resume call runs C1 (and the rest).
Every F is LF.guarded(...) (the registered input-range guard), every plan_beam call uses PR.beam_kwargs(...) and the
blind P.planner_instances(...). F is the FIRST computed answer per key, per arm: a stage loads the arm's cumulative
cache of the latest earlier RECORDED stage read-only and writes its own cumulative cache (create-only) when it completes.

Outputs (create-only) under runs/lbeam/. A stage is DONE only if all its files exist AND provenance.json records them
with matching sha256; anything else written by a stage makes it PARTIAL: moved to runs/lbeam/_orphans/ (never deleted)
and regenerated in full. A recorded output whose hash changes stops the job. Provenance and the job log record only
PLANNED generation counts for planning stages (realised calls are outputs, read by the analysis only).
Attempts: every stage start is recorded; a started stage that is not recorded failed. A report-only GPU stage (G2, G3,
G4) that failed twice is ABANDONED (not evaluable; the job continues); a core stage (A0, G1, C1) that failed three times
stops the job (no verdict). Deferral: the start-of-turn check uses the measured A0 time; two consecutive turns without a
recorded stage stop the job (exit 3).
Verdict: the analysis needs only A0, G1 and C1. DONE.json is written when every stage is recorded or abandoned; a
deterministic stop after A0, G1 and C1 are recorded (and no recorded output changed) writes a STOPPED.json marked
"verdict_eligible": true, and the analysis renders the verdict with the missing report-only stages 'not evaluable'.

Exit codes: 0 complete (DONE.json) or already complete; 3 deterministic stop, never retried (persisted as STOPPED.json
when it concerns the run, which includes a failed verification of an existing freeze; NOT persisted: "not frozen", "hook
not the frozen draft", "another RCWM job holds or waits for the lease", "run lock held", "STOPPED.json exists"); 5
deferred by the turn cap, or C1 still to run (resumes next call); 1 crash (the hook retries once).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_gen as G2  # noqa: E402  (torch-free at import: pilot loader, D-22 rule, registered pins)
import lbeam_problems as P  # noqa: E402  (torch-free)
import lbeam_proposer as PR  # noqa: E402  (torch-free)
import lbeam_f as LF  # noqa: E402  (torch-free at import)

# ---------------------------------------------------------------- registered constants
LEASE_LABEL = "rcwm-t1:lbeam"
OUT = ROOT / "runs" / "lbeam"
DOCS = ROOT / "docs" / "ledger_beam"
DRAFT_REL = "docs/ledger_beam/PREREG_LB_DRAFT.md"
PREREG_REL, FREEZE_REL = "PREREG_LB.md", "PREREG_LB_FREEZE.json"
AMEND_GLOB = "PREREG_LB_AMENDMENT_*.json"
DRAW_REL = "docs/ledger_beam/LB_CONFIRM_DRAW.jsonl"
DRAW_SOURCE_REL = "docs/ledger_beam/LB_CONFIRM_DRAW.SOURCE.json"
DEV_REL = "docs/ledger_beam/dev2_primary.jsonl"
DEV_SHA = "bb63aef48262f4643e8bc5bfc97990ee71f43ff8bee4a047e0603b04c1f10d05"
SPENT_DEV_SETS = ("docs/ledger_beam/dev_set.jsonl", "docs/ledger_beam/dev2_primary.jsonl", "docs/ledger_beam/dev2_oor.jsonl")
SPENT_SEEDS = (P.DEV_SEED_V1, P.DEV_SEED, P.DEV_OOR_SEED)
SEED_TAG = "PREREG-LB confirmation draw v2"           # v2: the rule includes BWM's revealed salt (commit-reveal)
SALT_MIN_LEN = 16
RESERVED_PREFIX = "HSKILL_CONFIRM_V2"              # never opened or hashed (I-28)
SET_NAME = "lbc"
N_PER_DEPTH = 36                                   # N = 180 (BWM draft); no enrichment of the needs-lookahead stratum
DEPTHS = (2, 3, 4, 5, 6)
HOOK_DRAFT_REL = "docs/ledger_beam/queue_hook_lbeam.sh.draft"
HOOK_INSTALLED_REL = "scripts/queue_hooks/70_lbeam.sh"
BWM_ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
BWM_PLANNER_REL = "bwmn/blocks_planner.py"
BWM_PLANNER_SHA = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"
BWM_DRAFT_REL = "research/PREREG_LEDGER_BEAM_DRAFT.md"
BWM_DRAFT_SHA = "b2896ba39301083a71db490c37c68d5826240faa4c10017830537d0b3e44c34f"
BWM_INFO_RELS = ("bwmn/__init__.py", "bwmn/blocks.py", "bwmn/lm.py")    # imported by the planner; recorded, not binding
ADAPTER_REL = "runs/t2/t2_s0/adapter"
ADAPTER_NAME = LF.T2_ADAPTER_NAME                  # 't2', as confirm2_gen
ANCHOR_REF_RELS = {"off": "runs/t2/t2_s0/gen_chain_one_base.jsonl", "on": "runs/t2/t2_s0/gen_chain_one.jsonl"}
PINS = {f"{ADAPTER_REL}/adapter_model.safetensors": G2.ADAPTER_SHA,
        f"{ADAPTER_REL}/adapter_config.json": G2.ADAPTER_CONFIG_SHA,
        ANCHOR_REF_RELS["off"]: G2.ANCHOR_REF_SHA["gen_chain_one_base.jsonl"],
        ANCHOR_REF_RELS["on"]: G2.ANCHOR_REF_SHA["gen_chain_one.jsonl"],
        "data/bank_pilot.json": G2.PILOT_SHA, DEV_REL: DEV_SHA}
FP = dict(G2.FP)
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)
IDENTITY = {"model": "Titan-Sparse FP32 " + G2.FP["sparse"], "adapter": ADAPTER_REL, "adapter_sha256": G2.ADAPTER_SHA,
            "adapter_name": ADAPTER_NAME, "decode": "greedy, 160 new tokens, EOS [151645, 151643]"}
EVAL_SEED = 0
CONFIGS = {"W2E12": (2, 12), "W4E24": (4, 24)}
AUDIT_SURFACE = "heldout_combined"
ANCHOR_MIN_IDENTICAL = 0.99
BATCHINV_N_INST = 32                               # 32 dev instances x 8 vocabulary actions = 256 keys per arm
BATCHINV_MIN_IDENTICAL = 0.99
STAGE_ORDER = ("A0", "G1", "G2", "G3", "G4", "C1")
GPU_STAGES = ("A0", "G1", "G2", "G3", "G4")
CORE_STAGES = ("A0", "G1", "C1")                  # the verdict needs exactly these
REPORT_ONLY_GPU = ("G2", "G3", "G4")
MAX_FAILED_REPORT_ONLY = 2                        # a report-only stage that failed twice is abandoned (not evaluable)
MAX_FAILED_CORE = 3                               # a core stage that failed three times stops the job (no verdict)
NO_PROGRESS_STOP = 2                              # consecutive turns without a recorded stage -> deterministic stop
RATE_GENS_PER_S = 2.7            # registered planning rate (PREREG-2's)
OBSERVED_RATE = 3.80             # PREREG-2's observed ledger chain_one rate on its sealed bank
BATCH1_RATE = 0.30               # planning assumption for one-prompt generate calls (duty cycle + throttle per call)
LOAD_S = 120.0
CAP_H = 4.5
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
RCWM_LABEL_PREFIX = "rcwm-t1:"
N_ANCHOR_CHAINS, ANCHOR_STEPS = 208, 4
# Expected NEW model generations per instance per stage (distinct keys not yet in the arm's cache), measured on the
# 120-instance dev set with model-free SimulatedF stand-ins (docs/ledger_beam/LB_POWER.json 'sizing'; SIM_T2_P3 sizes
# T2, SIM_OFF sizes OFF and the heldout_combined audit arm); the per-instance ceilings are hard bounds of the configs.
SIZING = {"G1": {"T2": 69.91, "OFF": 80.76}, "G2": {"T2": 14.23, "OFF": 17.33}, "G3": {"T2": 55.56},
          "G4": {"T2@heldout_combined": 80.76}}
CEILING = {"G1": {"T2": 104, "OFF": 104}, "G2": {"T2": 48, "OFF": 48}, "G3": {"T2": 216},
           "G4": {"T2@heldout_combined": 104}}
FILES = {
    "A0": {"anchor_off": "anchor/chain_one_L_off.jsonl", "anchor_on": "anchor/chain_one_L_on.jsonl",
           "anchor": "anchor/anchor.json", "bi_rows": "anchor/batch_invariance.jsonl", "bi": "anchor/batch_invariance.json"},
    "G1": {"eps:T2:W2E12": "primary/episodes_T2_W2E12.jsonl", "eps:OFF:W2E12": "primary/episodes_OFF_W2E12.jsonl",
           "cache:T2": "primary/fcache_T2.jsonl", "cache:OFF": "primary/fcache_OFF.jsonl", "acct": "primary/accounting.json"},
    "G2": {"fixed:T2": "fixed/fixed_T2.jsonl", "fixed:OFF": "fixed/fixed_OFF.jsonl", "cache:T2": "fixed/fcache_T2.jsonl",
           "cache:OFF": "fixed/fcache_OFF.jsonl", "acct": "fixed/accounting.json"},
    "G3": {"eps:T2:W4E24": "w4e24/episodes_T2_W4E24.jsonl", "cache:T2": "w4e24/fcache_T2.jsonl",
           "acct": "w4e24/accounting.json"},
    "G4": {"eps:T2@heldout_combined:W2E12": "audit/episodes_T2_heldout_combined_W2E12.jsonl",
           "cache:T2@heldout_combined": "audit/fcache_T2_heldout_combined.jsonl", "acct": "audit/accounting.json"},
    "C1": {"eps:ORACLE:W2E12": "cpu/episodes_ORACLE_W2E12.jsonl", "eps:ORACLE:W4E24": "cpu/episodes_ORACLE_W4E24.jsonl",
           "eps:IGNORE_STATUS:W2E12": "cpu/episodes_IGNORE_STATUS_W2E12.jsonl", "fixed:ORACLE": "cpu/fixed_ORACLE.jsonl",
           "fixed:IGNORE_STATUS": "cpu/fixed_IGNORE_STATUS.jsonl", "acct": "cpu/accounting.json"},
}
KIT_CODE = ("scripts/lbeam_f.py", "scripts/lbeam_problems.py", "scripts/lbeam_proposer.py", "scripts/lbeam_test.py")
RUN_CODE = ("scripts/lbeam_run.py", "scripts/lbeam_analysis.py", "scripts/lbeam_test_run.py", "scripts/lbeam_f.py",
            "scripts/lbeam_problems.py", "scripts/lbeam_proposer.py")
DRYRUN_CODE = ("scripts/lbeam_analysis.py", "scripts/lbeam_run.py")
RUN_TEST_PARTS = ("plan", "analysis", "tiny", "freeze", "power", "rehearsal", "hook")
REQUIRED_INPUTS = (
    DRAFT_REL, "scripts/lbeam_run.py", "scripts/lbeam_analysis.py", "scripts/lbeam_test_run.py", *KIT_CODE,
    "scripts/p3_common.py", "scripts/p3_make_bank.py", "scripts/l1_make_ood.py", "scripts/confirm2_gen.py",
    "data/bank_pilot.json", *PINS.keys(), "docs/ledger_beam/dev2_primary.SOURCE.json", "docs/ledger_beam/INTERFACE.md",
    "docs/ledger_beam/TEST_ORACLE.json", "docs/ledger_beam/TEST_TINY.json", "docs/ledger_beam/TEST_FRESH.json",
    *[f"docs/ledger_beam/RUN_TEST_{p.upper()}.json" for p in RUN_TEST_PARTS],
    "docs/ledger_beam/LB_DRYRUN.json", "docs/ledger_beam/LB_DRYRUN.md", "docs/ledger_beam/LB_POWER.json",
    "docs/ledger_beam/GEN_PLAN_LB.json", HOOK_DRAFT_REL, DRAW_REL, DRAW_SOURCE_REL)
PROV_CODE = ("scripts/lbeam_run.py", "scripts/lbeam_f.py", "scripts/lbeam_problems.py", "scripts/lbeam_proposer.py",
             "scripts/p3_common.py", "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/env.py", "rcwmt/capture.py")


class Stop(Exception):
    """A deterministic stop (exit 3, never retried). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[lbeam] {msg}", persist)


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_text_x(path: Path, text: str) -> None:
    """Create-only text file (LF), written to .partial and renamed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Stop(f"[lbeam] {path} exists (create-only)")
    tmp = _partial(path)
    with open(tmp, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows) -> None:
    write_text_x(path, "".join(json.dumps(r) + "\n" for r in rows))


def write_json_x(path: Path, obj) -> None:
    write_text_x(path, json.dumps(obj, indent=1) + "\n")


# ---------------------------------------------------------------- configuration (the registered job uses the defaults)
@dataclass
class Cfg:
    root: Path = ROOT
    out: Path = OUT
    draw: Path = ROOT / DRAW_REL
    draw_source: Path = ROOT / DRAW_SOURCE_REL
    dev: Path = ROOT / DEV_REL
    anchor_chains: list | None = None            # None: pilot TEST ledger chains (208)
    anchor_refs: dict = field(default_factory=lambda: {s: ROOT / r for s, r in ANCHOR_REF_RELS.items()})
    adapter_dir: Path = ROOT / ADAPTER_REL
    require_freeze: bool = True                  # False only from the tests (never reachable from the CLI)
    cap_h: float = CAP_H
    rate: float = RATE_GENS_PER_S
    bi_n_inst: int = BATCHINV_N_INST
    identity: dict = field(default_factory=lambda: dict(IDENTITY))


def pilot_anchor_chains() -> list:
    return G2.load_bank_chains(G2.PILOT_PATH, G2.PILOT_SHA, "L", "test")


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return Path(p).with_name(Path(p).name + ".partial")


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
    print(f"[lbeam] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


# ---------------------------------------------------------------- the confirmation draw (minted at the freeze)
def confirm_spec(seed: int) -> P.SetSpec:
    return P.SetSpec(name=SET_NAME, seed=seed, n_in_per_depth=N_PER_DEPTH, n_oor_per_depth=0, depths=DEPTHS,
                     V=P.V_DEFAULT, K=P.K_FIXED, d_close=P.D_CLOSE)


def union_digest(excl: dict) -> str:
    h = hashlib.sha256()
    for s in sorted(excl["pay"]):
        h.update(s.encode("utf-8") + b"\n")
    h.update(b"\x00")
    for s in sorted(excl["ops"]):
        h.update(s.encode("utf-8") + b"\n")
    return h.hexdigest()


def seed_from_rule(draft_sha: str, digest: str, salt: str) -> int:
    """The recorded rule: the first 12 hex digits of sha256('<tag>|<draft sha256>|<freshness-union digest>|<BWM salt>').
    No input can be chosen by looking at instances (the draft is final and co-signed before the freeze; the union is
    rebuilt at the freeze from the data on disk), and RCWM cannot compute the draw before the freeze: BWM commits to the
    salt (sha256) in its co-sign and reveals it only at the freeze (commit-reveal)."""
    return int(hashlib.sha256(f"{SEED_TAG}|{draft_sha}|{digest}|{salt}".encode("utf-8")).hexdigest()[:12], 16)


SEED_FORMULA = "int(sha256('<tag>|<draft sha256>|<union digest>|<BWM salt>')[:12], 16)"


def salt_commit(salt: str) -> str:
    return hashlib.sha256(str(salt).encode("utf-8")).hexdigest()


def cosign_commit(cosign: str) -> str | None:
    m = re.search(r"commit:([0-9a-f]{64})\b", str(cosign or ""))
    return m.group(1) if m else None


def _data_files(root: Path) -> list:
    """Every data/**/*.json[l] file the freshness union may read (HSKILL_CONFIRM_V2* skipped by name, never opened)."""
    d = Path(root) / "data"
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")
                  and not p.name.startswith(RESERVED_PREFIX))


def data_snapshot(root: Path) -> dict:
    """{rel: sha256} of every data/**/*.json[l] file: taken before and after the union build (concurrent writers)."""
    return {str(p.relative_to(Path(root))).replace("\\", "/"): sha256(p) for p in _data_files(root)}


def snapshot_digest(snap: dict) -> str:
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(snap.items())).encode("utf-8")).hexdigest()


def build_exclusions(root: Path, log=print) -> dict:
    """The kit's freshness union (lbeam_problems.load_exclusions: name-skipping historical harvest, every pinned source,
    PREREG-3's bank, every other top-level data/*.json[l] payload and multi-event rollout), rebuilt NOW, plus every
    payload of the data/ SUBDIRECTORIES (e.g. superseded training files) and every binding transition of the three spent
    dev sets."""
    excl = P.load_exclusions(historical=True, log=log)
    MB = P._import_builders()
    L1 = MB.L1
    d = Path(root) / "data"
    for p in _data_files(root):
        if p.parent == d or p.name.endswith(".SOURCE.json"):
            continue
        rel = str(p.relative_to(Path(root))).replace("\\", "/")
        try:
            sha = L1.sha256_file(p)
            e_pay = MB.harvest_file(p)
        except (OSError, ValueError) as e:
            excl["sources"][rel] = {"error": f"{type(e).__name__}: {e}"}
            continue
        e_ops = L1.op_keys_of_payloads(e_pay)
        m_pay, m_ops, n_obj = P._multi_event_ledger_transitions(p, MB, L1)
        excl["sources"][rel] = {"payloads": len(e_pay), "operative": len(e_ops), "sha256": sha,
                                "multi_event_objects": n_obj, "multi_event_transitions": len(m_pay)}
        excl["pay"] |= e_pay | m_pay
        excl["ops"] |= e_ops | m_ops
    excl["rule"] += " + every payload of the data/ subdirectories"
    L1g = P._L1()
    for rel in SPENT_DEV_SETS:
        pay, ops = set(), set()
        for inst in P.load_set(Path(root) / rel):
            for b, e in P.binding_transitions(inst):
                pay.add(L1g.payload(b, e))
                ops.add(L1g.opkey(P.LEDGER, b, e))
        excl["sources"][rel] = {"payloads": len(pay), "operative": len(ops), "sha256": sha256(Path(root) / rel)}
        excl["pay"] |= pay
        excl["ops"] |= ops
    excl["rule"] += " + every binding transition of the spent dev sets " + ", ".join(SPENT_DEV_SETS)
    return excl


def union_errors(excl: dict) -> list:
    return sorted(k for k, v in (excl.get("sources") or {}).items() if isinstance(v, dict) and "error" in v)


def mint_draw(draft_sha: str, excl: dict, salt: str) -> tuple:
    """(instances, SOURCE record, file text). Nothing about an instance is printed or returned to the operator."""
    dig = union_digest(excl)
    seed = seed_from_rule(draft_sha, dig, salt)
    if seed in SPENT_SEEDS:
        raise Stop(f"the seed rule gave a spent seed {seed}; amend the rule before freezing", persist=False)
    spec = confirm_spec(seed)
    insts, rec = P.make_set(spec, excl, log=None)
    ov = P.overlap_counts(insts, excl)
    if ov["payload_overlap"] or ov["operative_overlap"]:
        raise Stop(f"the minted draw overlaps the freshness union {ov}", persist=False)
    ver = P.verify_instances(insts)
    if not ver["ok"]:
        raise Stop(f"the minted draw does not verify ({len(ver.get('problems') or [])} problems; not shown)", persist=False)
    regen, _ = P.make_set(spec, replay_rejections=rec["fresh_rejections"])
    text = P.dumps_set(insts)
    regen_ok = P.set_sha(regen) == rec["file_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not regen_ok:
        raise Stop("the minted draw does not regenerate from (seed, spec, recorded rejections)", persist=False)
    per_depth = {d: sum(1 for i in insts if i["opt_depth"] == d) for d in DEPTHS}
    if per_depth != {d: N_PER_DEPTH for d in DEPTHS}:
        raise Stop(f"the minted draw is not {N_PER_DEPTH} per d*", persist=False)
    rec.update({"record": "LBEAM_CONFIRMATION_DRAW", "seed_rule": {"tag": SEED_TAG, "draft_sha256": draft_sha,
                                                                   "union_digest": dig, "bwm_salt": salt,
                                                                   "bwm_commit": salt_commit(salt), "seed": seed,
                                                                   "formula": SEED_FORMULA},
                "overlap": ov, "verify_ok": True, "regenerated_equal": True,
                "exclusion_union_size": {"payloads": len(excl["pay"]), "operative": len(excl["ops"])},
                "use": "PREREG-LB confirmation draw: sealed at the freeze; read only by lbeam_run.py and lbeam_analysis.py",
                "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    rec["freshness"] = {"rule": excl["rule"], "sources": excl["sources"]}
    return insts, rec, text


def load_draw(cfg: Cfg) -> list:
    src = _read_json(cfg.draw_source) or {}
    need(Path(cfg.draw).is_file(), f"the confirmation draw {cfg.draw} is missing")
    need(sha256(cfg.draw) == src.get("file_sha256"), "the confirmation draw differs from its SOURCE record")
    insts = P.load_set(cfg.draw)
    need(P.set_sha(insts) == src.get("file_sha256"), "the confirmation draw does not re-serialise to its sha256")
    need(all(i["range"] == "in" for i in insts), "the confirmation draw contains a non-primary instance")
    return insts


# ---------------------------------------------------------------- generation counts, estimates, the D-22 stage rule
def counts(n_inst: int, n_anchor: int = N_ANCHOR_CHAINS, bi_keys: int = BATCHINV_N_INST * P.V_DEFAULT) -> dict:
    return {"inst": n_inst, "A0_batched": n_anchor * ANCHOR_STEPS * 2 + bi_keys * 2, "A0_batch1": bi_keys * 2}


def stage_gens(st: str, n: dict, mode: str = "expected") -> float:
    if st == "A0":
        return n["A0_batched"] + n["A0_batch1"]
    tab = SIZING if mode == "expected" else CEILING
    return n["inst"] * sum(tab[st].values())


def stage_est_s(st: str, rate: float, n: dict, mode: str = "expected") -> float:
    if st == "A0":
        return n["A0_batched"] / rate + n["A0_batch1"] / BATCH1_RATE
    return stage_gens(st, n, mode) / rate


def simulate_turns(rate: float, n: dict, cap_h: float = CAP_H, mode: str = "expected") -> list:
    cap = cap_h * 3600
    left = list(GPU_STAGES)
    turns = []
    while left:
        t, ran = LOAD_S, []
        if "A0" not in left:
            t += stage_est_s("A0", rate, n)
            ran.append("A0 re-check (anchor + batch invariance)")
        for st in list(left):
            if not G2.can_start(t, stage_est_s(st, rate, n, mode), 0.0, cap):
                break
            t += stage_est_s(st, rate, n, mode)
            ran.append(st)
            left.remove(st)
        if not [r for r in ran if not r.startswith("A0 re-check")]:
            raise RuntimeError(f"no stage fits an empty turn at {rate} gens/s ({mode})")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(verbose: bool = True) -> dict:
    n = counts(N_PER_DEPTH * len(DEPTHS))
    what = {"A0": f"anchor {N_ANCHOR_CHAINS} chains x {ANCHOR_STEPS} steps x OFF/ON + batch invariance "
                  f"{BATCHINV_N_INST * P.V_DEFAULT} keys x T2/OFF x (batched, batch 1)",
            "G1": "PRIMARY plan_beam enum W2 E12, T2 + OFF", "G2": "select_fixed T2 + OFF (new keys only)",
            "G3": "plan_beam enum W4 E24, T2 (new keys only)", "G4": "SurfaceAuditF T2 heldout_combined, enum W2 E12"}
    res = {"n_instances": n["inst"], "rate_gens_per_s": RATE_GENS_PER_S, "observed_rate_gens_per_s": OBSERVED_RATE,
           "batch1_rate_gens_per_s": BATCH1_RATE, "load_s": LOAD_S, "cap_h": CAP_H, "sizing_per_instance": SIZING,
           "ceiling_per_instance": CEILING, "stages": [], "cpu_stage": "C1 (ORACLE / IGNORE_STATUS beams and select_fixed): "
                                                                     "CPU only, never under the lease, at the start of the "
                                                                     "invocation after the one that records A0 (before its "
                                                                     "lease, no CUDA use); about 1-2 CPU-min"}
    for st in GPU_STAGES:
        res["stages"].append({"stage": st, "what": what[st],
                              "generations_expected": round(stage_gens(st, n, "expected")),
                              "generations_ceiling": round(stage_gens(st, n, "ceiling")),
                              "h_expected_planning": round(stage_est_s(st, RATE_GENS_PER_S, n) / 3600, 2),
                              "h_expected_observed": round(stage_est_s(st, OBSERVED_RATE, n) / 3600, 2),
                              "h_ceiling_planning": round(stage_est_s(st, RATE_GENS_PER_S, n, "ceiling") / 3600, 2)})
    res["stage_rule"] = (f"no stage starts within {G2.STAGE_MARGIN_H} h of the {CAP_H} h cap; a stage starts only if "
                         f"{G2.STAGE_SAFETY} x its estimate fits; every turn after the first re-runs A0 (anchor + batch "
                         "invariance) before any planning stage; the start-of-turn check adds the model load and the longest A0 "
                         "measured so far; two consecutive turns without a recorded stage stop the job; the running rate "
                         "is re-measured after each planning stage")
    for key, rate, mode in (("turns_expected_planning_rate", RATE_GENS_PER_S, "expected"),
                            ("turns_expected_observed_rate", OBSERVED_RATE, "expected"),
                            ("turns_ceiling_planning_rate", RATE_GENS_PER_S, "ceiling"),
                            ("turns_ceiling_observed_rate", OBSERVED_RATE, "ceiling")):
        try:
            res[key] = simulate_turns(rate, n, CAP_H, mode)
        except RuntimeError as e:
            res[key] = str(e)
        res[key.replace("turns", "gpu_h")] = (round(sum(t["projected_h"] for t in res[key]), 2)
                                              if isinstance(res[key], list) else None)
    if verbose:
        print(f"PREREG-LB GPU plan ({n['inst']} instances; {LOAD_S:.0f} s per load; batch-1 rate {BATCH1_RATE}/s)")
        for r in res["stages"]:
            print(f"  {r['stage']}  {r['what']:<92} gens ~{r['generations_expected']:>6} (<= {r['generations_ceiling']:>6})"
                  f"  {r['h_expected_planning']:5.2f} h @{RATE_GENS_PER_S}/s  {r['h_expected_observed']:5.2f} h @{OBSERVED_RATE}/s")
        for key in ("turns_expected_planning_rate", "turns_expected_observed_rate", "turns_ceiling_planning_rate",
                    "turns_ceiling_observed_rate"):
            v = res[key]
            print(f"  {key}: " + ("; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(v))
                                  if isinstance(v, list) else v))
    return res


# ---------------------------------------------------------------- freeze verification (shared with lbeam_analysis.py)
def _num(rx: str, name: str) -> int:
    m = re.search(rx, name)
    return int(m.group(1)) if m else -1


def required_inputs(root: Path) -> set:
    return set(REQUIRED_INPUTS) | {f"rcwmt/{p.name}" for p in (Path(root) / "rcwmt").glob("*.py")}


def _amendments(root: Path) -> list:
    return sorted(Path(root).glob(AMEND_GLOB), key=lambda p: _num(r"_(\d+)\.json$", p.name))


def verify_freeze(root: Path | None = None) -> tuple:
    root = Path(root or ROOT)
    fz = _read_json(root / FREEZE_REL)
    if fz is None:
        half = interrupted_freeze(root)
        return False, [half or "PREREG_LB_FREEZE.json does not exist or is unreadable (PREREG-LB is a draft)"], None
    errs = []
    pr = root / PREREG_REL
    if not pr.is_file():
        errs.append("PREREG_LB.md does not exist")
    elif sha256(pr) != fz.get("prereg_sha256"):
        errs.append("PREREG_LB.md differs from the freeze")
    inputs = dict(fz.get("inputs") or {})
    missing = sorted(required_inputs(root) - set(inputs))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing}")
    for rel, h in PINS.items():
        if inputs.get(rel) != h:
            errs.append(f"{rel}: freeze record differs from the registered pin")
    expected = dict(inputs)
    for a in _amendments(root):
        am = _read_json(a) or {}
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
    if bwm.get("draft_sha256_at_freeze") != BWM_DRAFT_SHA or bwm.get("draft_sha256_cited") != BWM_DRAFT_SHA:
        errs.append("BWM's draft at the freeze was not the cited one")
    bp = BWM_ROOT / BWM_PLANNER_REL
    if not bp.is_file() or sha256(bp) != BWM_PLANNER_SHA:
        errs.append(f"BWM's {BWM_PLANNER_REL} differs from the pinned sha256")
    dr = fz.get("draw") or {}
    src = _read_json(root / DRAW_SOURCE_REL) or {}
    sr = src.get("seed_rule") or {}
    if not (dr.get("file_sha256") == src.get("file_sha256") == inputs.get(DRAW_REL)):
        errs.append("the draw's sha256 differs between the freeze record, its SOURCE record and the inputs")
    if sr.get("draft_sha256") != inputs.get(DRAFT_REL) or sr.get("draft_sha256") != fz.get("draft_sha256") or \
            sr.get("tag") != SEED_TAG or dr.get("seed") != sr.get("seed") or \
            sr.get("seed") != seed_from_rule(str(sr.get("draft_sha256")), str(sr.get("union_digest")), str(sr.get("bwm_salt"))) or \
            (src.get("spec") or {}).get("seed") != sr.get("seed") or sr.get("seed") in SPENT_SEEDS:
        errs.append("the draw's seed does not follow the recorded seed rule")
    cos = str(fz.get("bwm_cosign") or "")
    if salt_commit(str(sr.get("bwm_salt"))) != sr.get("bwm_commit") or sr.get("bwm_commit") != cosign_commit(cos) or \
            len(str(sr.get("bwm_salt") or "")) < SALT_MIN_LEN:
        errs.append("BWM's revealed salt does not match the commitment in its co-sign")
    if not fz.get("draft_sha256") or str(fz.get("draft_sha256")) not in cos:
        errs.append("BWM's co-sign does not quote the frozen draft's sha256")
    spec = src.get("spec") or {}
    if spec.get("n_in_per_depth") != N_PER_DEPTH or tuple(spec.get("depths") or ()) != DEPTHS or spec.get("n_oor_per_depth"):
        errs.append("the draw's spec is not the registered one")
    if fz.get("registered_runtime") != REGISTERED_RUNTIME:
        errs.append("registered runtime differs")
    if not fz.get("approval") or not fz.get("bwm_cosign"):
        errs.append("no approval / BWM co-sign recorded")
    return not errs, errs, fz


def allowed_hashes(fz: dict, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    allowed = {k: {v} for k, v in (fz.get("inputs") or {}).items()}
    for a in _amendments(root):
        am = _read_json(a) or {}
        if am.get("file") in allowed and am.get("new_sha256"):
            allowed[am["file"]].add(am["new_sha256"])
    return allowed


def lbeam_outputs(root: Path) -> list:
    """Anything a PREREG-LB run or analysis writes. The freeze refuses if one exists."""
    root = Path(root)
    found = []
    runs = root / "runs"
    if runs.exists():
        for p in runs.iterdir():
            if p.name.startswith("lbeam"):
                found.append(str(p.relative_to(root)))
        found += [str(p.relative_to(root)) for p in runs.rglob("PREREG_LB_RESULTS*")]
    return sorted(set(found))


def _fill_prereg(text: str, stamp: str, inputs: dict, approval: str, cosign: str, draw: dict) -> str:
    lines = text.splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("**Status:**")]
    if len(st) != 1:
        raise Stop("the draft has no unique **Status:** line", persist=False)
    lines[st[0]] = (f"**Status:** FROZEN at {stamp} by `scripts/lbeam_run.py --freeze` (approval: {approval}; BWM co-sign: "
                    f"{cosign}). The SHA-256 of this file is in `PREREG_LB_FREEZE.json`.")
    i10 = [i for i, l in enumerate(lines) if l.startswith("## 10.")]
    if len(i10) != 1:
        raise Stop("the draft has no unique section 10", persist=False)
    nxt = next((i for i in range(i10[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    sec = ["## 10. Freeze record", "",
           f"Frozen at {stamp} by `scripts/lbeam_run.py --freeze`. Confirmation draw minted at the freeze: seed {draw['seed']} "
           f"(rule: {draw['formula']}; draft sha256 {draw['draft_sha256']}; union digest {draw['union_digest']}; BWM salt "
           f"revealed at the freeze, commitment {draw['bwm_commit']}), {draw['n']} instances, sealed as "
           f"`{DRAW_REL}` sha256 `{draw['file_sha256']}`. BWM planner `{BWM_PLANNER_REL}` sha256 `{BWM_PLANNER_SHA}`. "
           f"data/ was unchanged while the union was built ({draw['data_files']} files, snapshot digest "
           f"{draw['data_snapshot_digest']}).", "",
           "SHA-256 of every frozen input:", ""]
    sec += [f"- `{k}` `{v}`" for k, v in sorted(inputs.items())]
    sec.append("")
    return "\n".join(lines[: i10[0]] + sec + lines[nxt:]) + "\n"


def _record_ok(rec: dict | None, root: Path, code: tuple) -> bool:
    return bool(rec and rec.get("ok")) and all((rec.get("code_sha256") or {}).get(r) == sha256(Path(root) / r) for r in code)


def _binds_draft(rec: dict | None, root: Path) -> bool:
    """The record was produced with the current draft and hook draft (the verbatim and wording checks bind the text)."""
    root = Path(root)
    return bool(rec) and (root / DRAFT_REL).is_file() and (root / HOOK_DRAFT_REL).is_file() and \
        rec.get("draft_sha256") == sha256(root / DRAFT_REL) and rec.get("hook_draft_sha256") == sha256(root / HOOK_DRAFT_REL)


def prefreeze_checks(root: Path) -> dict:
    """Everything that must hold before PREREG-LB can freeze (also run by the tests on a fake root)."""
    root = Path(root)
    d = root / "docs" / "ledger_beam"
    out = {}
    for t in ("TEST_ORACLE", "TEST_TINY", "TEST_FRESH"):
        rec = _read_json(d / f"{t}.json") or {}
        out[f"kit record {t}.json passed with the current kit code and the pinned BWM planner"] = (
            bool(rec.get("ok")) and rec.get("bwm_blocks_planner_sha256") == BWM_PLANNER_SHA and
            all((rec.get("code_sha256") or {}).get(r) == sha256(root / r) for r in KIT_CODE))
    for part in RUN_TEST_PARTS:
        out[f"runner test record RUN_TEST_{part.upper()}.json passed with the current code"] = _record_ok(
            _read_json(d / f"RUN_TEST_{part.upper()}.json"), root, RUN_CODE)
    for part in ("ANALYSIS", "HOOK"):
        out[f"RUN_TEST_{part}.json was produced with the current draft and hook draft"] = _binds_draft(
            _read_json(d / f"RUN_TEST_{part}.json"), root)
    dr = _read_json(d / "LB_DRYRUN.json")
    out["analysis dry run (every wording branch) passed with the current analysis code"] = _record_ok(dr, root, DRYRUN_CODE)
    out["analysis dry run was produced with the current draft and hook draft"] = _binds_draft(dr, root)
    import lbeam_analysis as LA                      # torch-free; imported lazily (it imports this module)
    draft = root / DRAFT_REL
    miss = LA.registered_strings_missing(draft.read_text(encoding="utf-8")) if draft.is_file() else ["draft missing"]
    out["every sentence the analysis can render appears verbatim in the draft"] = not miss
    pw = _read_json(d / "LB_POWER.json") or {}
    out["power and sizing record present (computed, with the current code)"] = _record_ok(pw, root, RUN_CODE)
    gp = _read_json(d / "GEN_PLAN_LB.json") or {}
    out["GEN_PLAN_LB.json present"] = bool(gp.get("stages"))
    out["queue hook draft present"] = (root / HOOK_DRAFT_REL).is_file()
    out["registered pins (adapter, anchor references, pilot bank, dev set) match"] = all(
        (root / r).is_file() and sha256(root / r) == h for r, h in PINS.items())
    out["BWM planner is the pinned file"] = (BWM_ROOT / BWM_PLANNER_REL).is_file() and \
        sha256(BWM_ROOT / BWM_PLANNER_REL) == BWM_PLANNER_SHA
    bd = BWM_ROOT / BWM_DRAFT_REL
    out["BWM's draft is the cited one (sha256 b2896ba3...)"] = bd.is_file() and sha256(bd) == BWM_DRAFT_SHA
    return out


def interrupted_freeze(root: Path) -> str | None:
    """A freeze writes the draw, its SOURCE, PREREG_LB.md and PREREG_LB_FREEZE.json as .partial files, then renames them
    in that order. Anything else on disk is an interrupted freeze; returns the recovery instruction."""
    root = Path(root)
    rels = (DRAW_REL, DRAW_SOURCE_REL, PREREG_REL, FREEZE_REL)
    partials = [r for r in rels if _partial(root / r).exists()]
    pr, fz = root / PREREG_REL, root / FREEZE_REL
    if pr.exists() and not fz.exists():
        rec = _read_json(_partial(fz)) or {}
        if _partial(fz).exists() and rec.get("prereg_sha256") == sha256(pr):
            return ("the freeze was interrupted before its last rename: PREREG_LB.md is in place and "
                    "PREREG_LB_FREEZE.json.partial records its sha256. Rename PREREG_LB_FREEZE.json.partial to "
                    "PREREG_LB_FREEZE.json, run --verify, and log it in DECISIONS.md")
        return ("PREREG_LB.md exists without PREREG_LB_FREEZE.json (and without a matching .partial record): not a "
                "freeze. Move PREREG_LB.md, the draw and any .partial aside, log it in DECISIONS.md, rerun --freeze")
    if not fz.exists() and partials:
        return (f"a freeze was interrupted before its renames ({partials} exist as .partial; nothing is frozen): move "
                "the .partial files (and any renamed draw file) aside UNREAD, log it in DECISIONS.md, rerun --freeze")
    if not fz.exists() and ((root / DRAW_REL).exists() or (root / DRAW_SOURCE_REL).exists()):
        return ("a confirmation draw exists without a freeze: move it aside UNREAD, log it in DECISIONS.md, and rerun "
                "--freeze (the seed rule re-mints the same draw if the union and BWM's salt are unchanged)")
    return None


def freeze(approval: str, bwm_cosign: str, bwm_salt: str, root: Path | None = None, checks=None) -> dict:
    root = Path(root or ROOT)
    if not approval or len(approval.strip()) < 8:
        raise Stop("--approval must state who approved the freeze, when, and the DECISIONS entry", persist=False)
    if not re.search(r"\bD-\w+", approval):
        raise Stop("--approval must cite the DECISIONS entry that approves the freeze ('D-<n>')", persist=False)
    if not bwm_cosign or len(bwm_cosign.strip()) < 8:
        raise Stop("--bwm-cosign must cite BWM's co-sign of this draft (BWM DECISIONS entry or message, with the "
                   "draft's full sha256 and BWM's salt commitment 'commit:<sha256>')", persist=False)
    half = interrupted_freeze(root)
    if half:
        raise Stop(f"refusing to freeze: {half}", persist=False)
    if (root / FREEZE_REL).exists() or (root / PREREG_REL).exists():
        raise Stop("PREREG-LB is already frozen (PREREG_LB_FREEZE.json or PREREG_LB.md exists)", persist=False)
    outs = lbeam_outputs(root)
    if outs:
        raise Stop(f"PREREG-LB outputs exist before the freeze: {outs}; investigate before freezing", persist=False)
    if not (root / DRAFT_REL).is_file():
        raise Stop(f"{DRAFT_REL} missing", persist=False)
    draft_bytes = (root / DRAFT_REL).read_bytes()             # read ONCE: the seed, the inputs and PREREG_LB.md
    draft_sha = hashlib.sha256(draft_bytes).hexdigest()
    cosign = bwm_cosign.strip()
    if draft_sha not in cosign:
        raise Stop(f"--bwm-cosign does not quote this draft's sha256 ({draft_sha}); BWM co-signs a stated text",
                   persist=False)
    commit = cosign_commit(cosign)
    if commit is None:
        raise Stop("--bwm-cosign carries no salt commitment 'commit:<sha256 of BWM's salt>'", persist=False)
    salt = str(bwm_salt or "")
    if len(salt) < SALT_MIN_LEN or salt_commit(salt) != commit:
        raise Stop(f"--bwm-salt is missing, shorter than {SALT_MIN_LEN} characters, or does not match the commitment in "
                   "BWM's co-sign", persist=False)
    bwm_draft = BWM_ROOT / BWM_DRAFT_REL
    bwm_draft_sha = sha256(bwm_draft) if bwm_draft.is_file() else None
    if bwm_draft_sha != BWM_DRAFT_SHA:
        raise Stop(f"BWM's draft {BWM_DRAFT_REL} is not the cited version (sha256 {bwm_draft_sha} != {BWM_DRAFT_SHA}); "
                   "revise this draft to cite BWM's current draft and get the co-sign again", persist=False)
    miss = sorted(r for r in required_inputs(root) if r not in (DRAW_REL, DRAW_SOURCE_REL) and not (root / r).is_file())
    if miss:
        raise Stop(f"frozen inputs missing (nothing was minted): {miss}", persist=False)
    chk = (checks or prefreeze_checks)(root)
    bad = [k for k, v in chk.items() if not v]
    if bad:
        raise Stop(f"pre-freeze checks failed: {bad}", persist=False)
    snap0 = data_snapshot(root)
    excl = build_exclusions(root)
    bad_src = union_errors(excl)
    if bad_src:
        raise Stop(f"a source of the freshness union could not be read: {bad_src}; nothing was minted", persist=False)
    snap1 = data_snapshot(root)
    if snap1 != snap0:
        changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
        raise Stop(f"data/ changed while the freshness union was built ({changed[:6]}); freeze only while no agent "
                   "writes to data/. Nothing was minted", persist=False)
    if sha256(root / DRAFT_REL) != draft_sha:
        raise Stop("the draft changed during the freeze; nothing was minted", persist=False)
    insts, rec, text = mint_draw(draft_sha, excl, salt)
    del excl
    rec["data_snapshot"] = {"files": len(snap0), "digest": snapshot_digest(snap0),
                            "rule": "sha256 of every data/**/*.json[l] (HSKILL_CONFIRM_V2* skipped by name) before and "
                                    "after the union build; equal"}
    src_text = json.dumps(rec, indent=1) + "\n"
    staged = {DRAW_REL: text, DRAW_SOURCE_REL: src_text}
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
    sr = rec["seed_rule"]
    draw = {"seed": sr["seed"], "formula": sr["formula"], "draft_sha256": draft_sha, "union_digest": sr["union_digest"],
            "bwm_commit": sr["bwm_commit"], "n": rec["n"], "file_sha256": rec["file_sha256"],
            "per_stratum": rec["per_stratum"], "overlap": rec["overlap"],
            "fresh_rejections": len(rec["fresh_rejections"]), "verify_ok": True, "regenerated_equal": True,
            "data_files": len(snap0), "data_snapshot_digest": rec["data_snapshot"]["digest"]}
    text_pr = _fill_prereg(draft_bytes.decode("utf-8"), stamp, inputs, approval.strip(), cosign, draw)
    with open(_partial(root / PREREG_REL), "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text_pr)
    frec = {"frozen_at": stamp, "approval": approval.strip(), "bwm_cosign": cosign,
            "prereg_sha256": sha256(_partial(root / PREREG_REL)), "draft_sha256": draft_sha, "inputs": inputs,
            "prefreeze_checks": chk, "registered_runtime": REGISTERED_RUNTIME, "lease_label": LEASE_LABEL,
            "out_dir": "runs/lbeam", "no_outputs_before_freeze": True, "draw": draw,
            "bwm": {"root": str(BWM_ROOT), "planner_rel": BWM_PLANNER_REL, "planner_sha256": BWM_PLANNER_SHA,
                    "draft_rel": BWM_DRAFT_REL, "draft_sha256_cited": BWM_DRAFT_SHA,
                    "draft_sha256_at_freeze": bwm_draft_sha, "salt_commit": commit,
                    "imported_files_recorded_not_binding": {r: sha256(BWM_ROOT / r) for r in BWM_INFO_RELS
                                                            if (BWM_ROOT / r).is_file()}}}
    with open(_partial(root / FREEZE_REL), "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(frec, indent=1) + "\n")
    for rel in (DRAW_REL, DRAW_SOURCE_REL, PREREG_REL, FREEZE_REL):
        os.rename(_partial(root / rel), root / rel)         # os.rename refuses an existing target on Windows
    ok, errs, _ = verify_freeze(root)
    if not ok:
        raise Stop(f"the freeze record was written but does not verify: {errs}", persist=False)
    print(f"[lbeam] FROZEN {stamp}; PREREG_LB.md sha {frec['prereg_sha256']}; draw sealed sha {rec['file_sha256']} "
          f"({rec['n']} instances; contents not shown)", flush=True)
    return frec


# ---------------------------------------------------------------- run lock, one-waiter rule, installed hook, runtime
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
            raise Stop(f"[lbeam] another lbeam_run.py holds the run lock {rec}", persist=False)
        print(f"[lbeam] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink()
    with open(lk, "x", encoding="utf-8", newline="\n") as fh:
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
    """R14: refuse if another RCWM job (label rcwm-t1:*) waits for or holds the GPU lease. BWM's jobs are allowed."""
    w = _read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = _read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and _live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


def hook_problem(root: Path, fz: dict | None) -> str | None:
    root = Path(root)
    inst = root / HOOK_INSTALLED_REL
    if not inst.exists():
        return None
    h = sha256(inst)
    if fz is not None:
        allowed = allowed_hashes(fz, root).get(HOOK_DRAFT_REL, set())
    else:
        allowed = {sha256(root / HOOK_DRAFT_REL)} if (root / HOOK_DRAFT_REL).is_file() else set()
    return None if h in allowed else f"{HOOK_INSTALLED_REL} (sha256 {h[:12]}) is not the hook draft byte for byte"


def runtime_problem(rt: dict, versions: dict | None = None) -> str | None:
    bad = {k: rt.get(k) for k in ("torch", "transformers") if rt.get(k) != REGISTERED_RUNTIME[k]}
    if versions is not None and versions.get("peft") != REGISTERED_RUNTIME["peft"]:
        bad["peft"] = versions.get("peft")
    return f"runtime {bad} differs from the registered {REGISTERED_RUNTIME}" if bad else None


# ---------------------------------------------------------------- model-facing pieces (torch imported lazily)
class RealBackend:
    """The registered GPU path: rcwmt.runtime (lease, resources) and lbeam_f.load_registered_t2 (PREREG-2's loading)."""
    device = "cuda"

    def __init__(self):
        import torch  # noqa: F401
        from rcwmt import runtime
        self.rt = runtime

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 PREREG-LB ledger beam (Sparse + T2 s0 ON/OFF)")

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
            print("[lbeam] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)

    def wait(self) -> None:
        self.thermal_wait()
        self.rt.wait_for_resources()

    def load(self, cfg: Cfg):
        """(model, tok, info). The adapter files were hash-checked against the freeze before this call."""
        try:
            model, tok, prov = LF.load_registered_t2(ADAPTER_NAME)
        except RuntimeError as e:
            if any(w in str(e) for w in ("fingerprint", "hash", "merged")):
                raise Stop(f"[lbeam] registered load refused: {e}")
            raise
        return model, tok, {"fingerprint": prov["sparse_fingerprint"], "versions": {"peft": prov["peft"]}}

    def tokenizer(self):
        return self.rt.tokenizer()

    def unload(self) -> None:
        self.rt.unload(None)

    def runtime_info(self) -> dict:
        return self.rt.runtime_info()


def import_plan_beam():
    """BWM's planner, read-only, at its pinned sha (no bytecode is written into the BWM bundle). bwmn.lm imports torch
    at module level."""
    sys.dont_write_bytecode = True
    import torch  # noqa: F401
    if str(BWM_ROOT) not in sys.path:
        sys.path.append(str(BWM_ROOT))
    need(sha256(BWM_ROOT / BWM_PLANNER_REL) == BWM_PLANNER_SHA, f"BWM's {BWM_PLANNER_REL} differs from the pinned sha256")
    from bwmn import blocks_planner as BP
    need(Path(BP.__file__).resolve() == (BWM_ROOT / BWM_PLANNER_REL).resolve(),
         f"blocks_planner imported from {BP.__file__}, not the pinned file")
    return BP


def _ref_rows(path: Path) -> dict:
    ref = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ref[row["cand_id"]] = row
    return ref


def chain_rows(chains, results) -> list:
    from rcwmt import env
    return [{"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]], "texts": r["texts"]}
            for c, r in zip(chains, results)]


def run_anchor(model, tok, chains: list, refs: dict) -> tuple:
    """PREREG-2's A0 anchor exactly as confirm2_gen.run_gpu.anchor: pilot TEST ledger chain_one, OFF (inside
    model.disable_adapter()) then ON (adapter 't2'), each against its T2 s0 reference file (identical batch
    composition). Gate: identical STATES on >= 99% of chains per side; identical texts recorded as a diagnostic."""
    from rcwmt import env, generation
    t0 = time.time()
    with model.disable_adapter():
        r_off = generation.chain(model, tok, chains, "chain_one")
    t1 = time.time()
    model.set_adapter(ADAPTER_NAME)
    need(LF.peft_side(model) == ADAPTER_NAME, f"anchor ON side: PEFT side {LF.peft_side(model)!r}")
    r_on = generation.chain(model, tok, chains, "chain_one")
    t2 = time.time()
    res = {"pid": os.getpid(), "unix": time.time(), "sides": {"off": {"pid": os.getpid(), "start_unix": t0, "end_unix": t1},
                                                              "on": {"pid": os.getpid(), "start_unix": t1, "end_unix": t2}}}
    for s, rr in (("off", r_off), ("on", r_on)):
        ref = _ref_rows(refs[s])
        st = sum([None if x is None else env.dumps(x) for x in r["states"]] == (ref.get(c["cand_id"]) or {}).get("states")
                 for c, r in zip(chains, rr))
        tx = sum(r["texts"] == (ref.get(c["cand_id"]) or {}).get("texts") for c, r in zip(chains, rr))
        res[s] = {"n": len(chains), "identical_states": st / len(chains), "identical_texts": tx / len(chains)}
    res["pass"] = all(res[s]["identical_states"] >= ANCHOR_MIN_IDENTICAL for s in ("off", "on"))
    return r_off, r_on, res


def batchinv_keys(cfg: Cfg) -> list:
    dev = P.load_set(cfg.dev)
    return [(i["init"], a) for i in dev[:cfg.bi_n_inst] for a in i["vocab"]]


def run_batch_invariance(model, tok, keys: list, cfg: Cfg) -> tuple:
    """Per arm (T2, OFF): the keys as ONE LedgerF predict list (one generate_texts call, registered sub-batches) vs one
    key per predict call (batch 1). Gate: identical parsed states on >= 99% of keys per arm."""
    rows, res = [], {"pid": os.getpid(), "unix": time.time(), "n_keys": len(keys), "arms": {}}
    for arm in ("T2", "OFF"):
        kw = {"adapter_name": ADAPTER_NAME} if arm == "T2" else {}
        fb = LF.LedgerF(arm, model=model, tok=tok, identity=cfg.identity, **kw)
        rb = fb.predict(keys)
        f1 = LF.LedgerF(arm, model=model, tok=tok, identity=cfg.identity, **kw)
        r1 = [f1.predict([k])[0] for k in keys]
        same_s = sum(a == b for a, b in zip(rb, r1))
        same_t = sum(fb.texts[k] == f1.texts[k] for k in keys)
        res["arms"][arm] = {"identical_states": same_s / len(keys), "identical_texts": same_t / len(keys),
                            "generation_calls_batched": fb.n_generate_fn_calls,
                            "generation_calls_batch1": f1.n_generate_fn_calls,
                            "model_generate_calls_batched": fb.n_model_generate_calls,
                            "model_generate_calls_batch1": f1.n_model_generate_calls,
                            "max_prompts_per_model_generate_batch1": max(f1.prompts_per_model_generate or [0])}
        rows += [{"arm": arm, "s": k[0], "a": k[1], "text_batched": fb.texts[k], "text_batch1": f1.texts[k]} for k in keys]
    res["pass"] = all(v["identical_states"] >= BATCHINV_MIN_IDENTICAL and v["generation_calls_batched"] == 1
                      and v["generation_calls_batch1"] == len(keys) and v["max_prompts_per_model_generate_batch1"] <= 1
                      for v in res["arms"].values())
    return rows, res


def _tokens_by_owner(inner, own: dict, tok) -> dict:
    out: dict = {}
    if tok is None:
        return out
    for k in inner.requested:
        t = inner.texts.get(k)
        if t is None:
            continue
        i = own.get(k[1])
        out[i] = out.get(i, 0) + len(tok(t, add_special_tokens=False)["input_ids"])
    return out


def accounting(inner, F, own: dict, ids: list, tok, seconds: float) -> dict:
    calls = inner.calls_by_owner(own)
    ref = F.refused_by_owner(own)
    toks = _tokens_by_owner(inner, own, tok)
    return {"f_keys_by_instance": {i: calls.get(i, 0) for i in ids}, "refused_by_instance": {i: ref.get(i, 0) for i in ids},
            "tokens_by_instance": {i: toks.get(i, 0) for i in ids}, "guard": F.stats(),
            "generated_tokens_new_exact": getattr(inner, "generated_tokens", 0), "model_queries_new": inner.model_queries,
            "cache_loaded": getattr(inner, "cache_loaded", 0), "requested_unique": len(inner.requested),
            "seconds": round(seconds, 1)}


def ep_row(e: dict) -> dict:
    keep = ("id", "verified", "fail", "rounds", "expansions", "gen_tokens", "selected", "selected_kind", "n_unsupported")
    r = {k: e[k] for k in keep if k in e}
    r["plan"] = [list(a) for a in (e.get("plan") or [])]
    if "pred_unmet" in e:
        r["pred_unmet"] = [None if (isinstance(x, float) and x == float("inf")) else x for x in e["pred_unmet"]]
    return r


def cache_header(arm: str, frame: str, identity: dict) -> dict:
    return {"schema": LF.CACHE_SCHEMA, "class": "LedgerF" if frame == LF.CANONICAL else "SurfaceAuditF", "arm": arm,
            "frame": frame, "identity": identity}


def make_inner(arm: str, frame: str, model, tok, cfg: Cfg, prior: Path | None):
    """The arm's F. With a prior cumulative cache (a recorded output of an earlier stage), the kit's own loader reads it
    (header checked: class, arm, frame, identity) and the F is then detached from that file (nothing is appended to a
    recorded output); the stage writes its own cumulative cache when it completes."""
    kw = {"adapter_name": ADAPTER_NAME} if arm == "T2" else {}
    if frame == LF.CANONICAL:
        inner = LF.LedgerF(arm, model=model, tok=tok, cache=prior, identity=cfg.identity, **kw)
    else:
        inner = LF.SurfaceAuditF(arm, frame, model=model, tok=tok, cache=prior, identity=cfg.identity, **kw)
    need(inner._header() == cache_header(arm, frame, cfg.identity), "cache header layout differs from the kit's")
    inner.cache = None
    return inner


def cache_text(header: dict, prior: Path | None, inner) -> str:
    lines = [json.dumps(header)]
    if prior is not None:
        pl = Path(prior).read_text(encoding="utf-8").splitlines()
        need(json.loads(pl[0]) == header, f"prior cache {prior} header differs")
        lines += [l for l in pl[1:] if l.strip()]
    lines += [json.dumps({"s": k[0], "a": k[1], "text": inner.texts[k]}) for k in inner.new_keys]
    return "\n".join(lines) + "\n"


def prior_cache(stage: str, key: str, out: Path, done=None) -> Path | None:
    """The arm's cumulative cache of the latest earlier RECORDED stage (an abandoned stage's cache never counts)."""
    for st in reversed(STAGE_ORDER[:STAGE_ORDER.index(stage)]):
        rel = FILES[st].get(f"cache:{key}")
        if rel and (done is None or st in done):
            p = Path(out) / rel
            need(p.is_file(), f"stage {stage}: the prior cache {rel} is missing")
            return p
    return None


def run_planning_stage(st: str, BP, model, tok, cfg: Cfg, out: Path, insts: list, PI: list, own: dict,
                       done=None) -> dict:
    """G1-G4: every arm of the stage in this process; all files written only after every arm is done."""
    ids = [i["id"] for i in insts]
    jobs = {"G1": [("T2", LF.CANONICAL, "beam", "W2E12"), ("OFF", LF.CANONICAL, "beam", "W2E12")],
            "G2": [("T2", LF.CANONICAL, "fixed", None), ("OFF", LF.CANONICAL, "fixed", None)],
            "G3": [("T2", LF.CANONICAL, "beam", "W4E24")],
            "G4": [("T2", AUDIT_SURFACE, "beam", "W2E12")]}[st]
    files, acct, sides, n_new = {}, {"stage": st, "configs": {}}, {}, 0
    for arm, frame, kind, cname in jobs:
        key = arm if frame == LF.CANONICAL else f"{arm}@{frame}"
        prior = prior_cache(st, key, out, done)
        t0 = time.time()
        inner = make_inner(arm, frame, model, tok, cfg, prior)
        F = LF.guarded(inner, insts)
        if kind == "beam":
            prop = PR.EnumerateProposer()
            W, E = CONFIGS[cname]
            eps = BP.plan_beam(PI, prop, F, eval_seed=EVAL_SEED, **PR.beam_kwargs(prop, W, E))
            files[FILES[st][f"eps:{key}:{cname}"]] = "".join(json.dumps(ep_row(e)) + "\n" for e in eps)
            ck = f"{key}:{cname}"
        else:
            eps = PR.select_fixed(insts, F, eval_seed=EVAL_SEED)
            files[FILES[st][f"fixed:{key}"]] = "".join(json.dumps(ep_row(e)) + "\n" for e in eps)
            ck = f"{key}:fixed"
        need([e["id"] for e in eps] == ids, f"stage {st} {ck}: episode ids are not the draw's ids in order")
        files[FILES[st][f"cache:{key}"]] = cache_text(cache_header(arm, frame, cfg.identity), prior, inner)
        acct["configs"][ck] = accounting(inner, F, own, ids, tok, time.time() - t0)
        sides[ck] = {"pid": os.getpid(), "start_unix": t0, "end_unix": time.time()}
        n_new += inner.model_queries
    files[FILES[st]["acct"]] = json.dumps(acct, indent=1) + "\n"
    for rel, text in files.items():                  # written together, only after every arm completed
        write_text_x(out / rel, text)
    return {"sides": sides, "n_new": n_new}


def run_cpu_stage(BP, out: Path, insts: list, PI: list, own: dict) -> dict:
    """C1: exact / rule F only (no model, no GPU, never under the lease)."""
    ids = [i["id"] for i in insts]
    files, acct = {}, {"stage": "C1", "configs": {}}
    for arm, kind, cname in (("ORACLE", "beam", "W2E12"), ("ORACLE", "beam", "W4E24"), ("IGNORE_STATUS", "beam", "W2E12"),
                             ("ORACLE", "fixed", None), ("IGNORE_STATUS", "fixed", None)):
        t0 = time.time()
        inner = LF.LedgerF(arm)
        F = LF.guarded(inner, insts)
        if kind == "beam":
            prop = PR.EnumerateProposer()
            W, E = CONFIGS[cname]
            eps = BP.plan_beam(PI, prop, F, eval_seed=EVAL_SEED, **PR.beam_kwargs(prop, W, E))
            files[FILES["C1"][f"eps:{arm}:{cname}"]] = "".join(json.dumps(ep_row(e)) + "\n" for e in eps)
            ck = f"{arm}:{cname}"
        else:
            eps = PR.select_fixed(insts, F, eval_seed=EVAL_SEED)
            files[FILES["C1"][f"fixed:{arm}"]] = "".join(json.dumps(ep_row(e)) + "\n" for e in eps)
            ck = f"{arm}:fixed"
        need([e["id"] for e in eps] == ids, f"stage C1 {ck}: episode ids are not the draw's ids in order")
        acct["configs"][ck] = accounting(inner, F, own, ids, None, time.time() - t0)
    files[FILES["C1"]["acct"]] = json.dumps(acct, indent=1) + "\n"
    for rel, text in files.items():
        write_text_x(out / rel, text)
    return {"pid": os.getpid()}


def cuda_initialized() -> bool:
    """torch.cuda.is_initialized() if torch is imported (a Python-level flag: it never calls into the driver)."""
    t = sys.modules.get("torch")
    try:
        return bool(t is not None and t.cuda.is_initialized())
    except Exception:                                 # noqa: BLE001  (an unusual torch build: count it as initialised)
        return True


def bwm_module_hashes() -> dict:
    """sha256 of every loaded module file under the BWM bundle (recorded per turn and CPU run; only the planner binds)."""
    out = {}
    base = BWM_ROOT.resolve()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            p = Path(f).resolve()
            rel = p.relative_to(base)
        except (ValueError, OSError):
            continue
        if p.is_file():
            out[str(rel).replace("\\", "/")] = sha256(p)
    return dict(sorted(out.items()))


def measured_a0_s(prov: dict, n: dict, rate: float) -> float:
    """The longest measured A0 (first run or a re-check) of this run; the estimate if none was measured."""
    xs = [float((prov.get("stages") or {}).get("A0", {}).get("seconds") or 0.0)]
    xs += [float(t.get("a0_recheck_seconds") or 0.0) for t in prov.get("turns") or []]
    m = max(xs)
    return m if m > 0 else stage_est_s("A0", rate, n)


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = Path(cfg.out)
    fz = None
    if cfg.require_freeze:
        ok, errs, fz = verify_freeze(cfg.root)
        if not ok:
            raise Stop("refusing to run the PREREG-LB job: " + "; ".join(errs), persist=(Path(cfg.root) / FREEZE_REL).exists())
        ins = fz["inputs"]
        need(sha256(cfg.draw) == ins[DRAW_REL], "the confirmation draw differs from the freeze")
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            need(sha256(Path(cfg.adapter_dir) / f) == ins[f"{ADAPTER_REL}/{f}"], f"adapter {f} differs from the freeze")
        for s, p in cfg.anchor_refs.items():
            need(sha256(p) == ins[ANCHOR_REF_RELS[s]], f"anchor reference {s} differs from the freeze")
        need(sha256(cfg.dev) == ins[DEV_REL], "the dev set (batch-invariance keys) differs from the freeze")
    hp = hook_problem(cfg.root, fz)
    if hp:
        raise Stop(f"[lbeam] {hp}; reinstall it byte for byte from {HOOK_DRAFT_REL} (or remove it) before any run",
                   persist=False)
    if (out / "DONE.json").exists():
        print(f"[lbeam] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; an operator must investigate, "
                   "log it in DECISIONS.md and move the file aside before any rerun", persist=False)
    for name in ("anchor.json", "batch_invariance.json"):
        rec = out / "anchor" / name
        if rec.exists() and not (_read_json(rec) or {}).get("pass"):
            raise Stop(f"the A0 {name} check failed in an earlier turn (deviation policy: investigate, log)")
    if list((out / "anchor").glob("RECHECK_FAILED_*.json")):
        raise Stop("an A0 re-check failed in an earlier turn (deviation policy: investigate, log)")
    insts = load_draw(cfg)
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[lbeam] {ow}: one-waiter rule (R14). Start PREREG-LB only when no other RCWM GPU job is running or "
                   "waiting (never next to a live master chain)", persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, fz, insts)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg: Cfg, backend, out: Path, fz, insts: list) -> int:
    prov_path = out / "provenance.json"
    prov = _read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
    else:
        prov = {"draw_sha256": sha256(cfg.draw), "freeze_prereg_sha256": (fz or {}).get("prereg_sha256"),
                "registered": bool(cfg.require_freeze), "lease_label": LEASE_LABEL, "stages": {}, "turns": [],
                "cpu_runs": []}
    for k, v in (("attempts", {}), ("abandoned", {}), ("orphaned", []), ("no_progress_turns", 0)):
        prov.setdefault(k, v)
    need(prov.get("draw_sha256") == sha256(cfg.draw), "provenance belongs to another draw")
    prov["adapter"] = {f: sha256(Path(cfg.adapter_dir) / f) for f in ("adapter_model.safetensors", "adapter_config.json")}
    prov["anchor_refs"] = {s: sha256(p) for s, p in cfg.anchor_refs.items()}
    prov["code_sha256"] = {r: sha256(ROOT / r) for r in PROV_CODE}
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
    for st in STAGE_ORDER:                                  # a started attempt that was never recorded failed
        if st in done or st in prov["abandoned"]:
            continue
        fails = len(prov["attempts"].get(st) or [])
        if st in REPORT_ONLY_GPU and fails >= MAX_FAILED_REPORT_ONLY:
            prov["abandoned"][st] = {"failed_attempts": fails, "attempts": prov["attempts"].get(st),
                                     "unix": time.time(), "rule": f"a report-only stage that failed {MAX_FAILED_REPORT_ONLY} "
                                                                  "times is abandoned: not evaluable"}
            print(f"[lbeam] report-only stage {st} failed {fails} attempts: abandoned (not evaluable)", flush=True)
        elif st in CORE_STAGES and fails >= MAX_FAILED_CORE:
            save()
            raise Stop(f"[lbeam] core stage {st} failed {fails} attempts (limit {MAX_FAILED_CORE}): stop, no verdict "
                       "(deviation policy: investigate, log)")
    abandoned = set(prov["abandoned"])

    def record(st: str, extra: dict) -> None:
        prov["stages"][st] = {"name": st, "turn": len(prov["turns"]) - 1, **extra,
                              "files": {rel_out(p, out): sha256(p) for p in stage_required(st, out)},
                              "finished_unix": time.time()}
        done.add(st)
        save()

    def finalize() -> int:
        need(set(prov["stages"]) | abandoned == set(STAGE_ORDER) and not (set(prov["stages"]) & abandoned),
             "finalize: not every stage is recorded or abandoned")
        need(all(stage_states(prov, out)[s] == "done" for s in prov["stages"]), "finalize: a recorded stage changed")
        save()
        rec = [s for s in STAGE_ORDER if s in prov["stages"]]
        write_json_x(out / "DONE.json", {"finished_unix": time.time(), "stages": rec,
                                         "not_evaluable": {s: prov["abandoned"][s] for s in STAGE_ORDER if s in abandoned},
                                         "files": {st: prov["stages"][st]["files"] for st in rec},
                                         "provenance_sha256": sha256(prov_path)})
        print("[lbeam] all stages complete" + (f" ({sorted(abandoned)} abandoned: not evaluable)" if abandoned else "")
              + "; DONE.json written", flush=True)
        return 0

    todo = [st for st in STAGE_ORDER if st not in done and st not in abandoned]
    print(f"[lbeam] stages to run: {todo}", flush=True)
    save()
    if not todo:
        return finalize()
    PI, own = P.planner_instances(insts), P.owner_of_action(insts)
    BP = import_plan_beam()
    bwm_mods = bwm_module_hashes()
    n = counts(len(insts), len(cfg.anchor_chains) if cfg.anchor_chains is not None else N_ANCHOR_CHAINS,
               cfg.bi_n_inst * P.V_DEFAULT)

    def run_c1() -> None:                 # CPU stage: never under the lease, only after A0 passed, never after CUDA use
        prov["attempts"].setdefault("C1", []).append("cpu")
        save()
        cuda0 = cuda_initialized()
        need(not cuda0, "C1 would run in a process that has initialised CUDA")
        t0 = time.time()
        run_cpu_stage(BP, out, insts, PI, own)
        cpu = {"pid": os.getpid(), "start_unix": t0, "end_unix": time.time(), "code_sha256": dict(prov["code_sha256"]),
               "bwm_planner_sha256": prov["bwm_planner_sha256"], "bwm_modules_sha256": bwm_mods, "lease_held": False,
               "cuda_initialized_before": cuda0, "cuda_initialized_after": cuda_initialized()}
        prov["cpu_runs"].append(cpu)
        record("C1", {"seconds": round(time.time() - t0, 1), "pid": os.getpid(), "after_a0": True, "lease_held": False})
        print(f"[lbeam] C1 (CPU: ORACLE / IGNORE_STATUS) done in {time.time() - t0:.0f}s", flush=True)

    if "A0" in done and "C1" not in done:     # A0 passed in an EARLIER invocation: C1 first, before any lease or CUDA use
        run_c1()
    gpu_todo = [st for st in GPU_STAGES if st not in done and st not in abandoned]
    turn = None
    if gpu_todo:
        anchor_chains = cfg.anchor_chains if cfg.anchor_chains is not None else pilot_anchor_chains()
        bi_keys = batchinv_keys(cfg)
        with backend.lease():
            t_acq, t_src = backend.lease_acquired_unix()
            cap_s = cfg.cap_h * 3600
            turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                    "code_sha256": dict(prov["code_sha256"]), "bwm_planner_sha256": prov["bwm_planner_sha256"],
                    "bwm_modules_sha256": bwm_mods, "cap_h": cfg.cap_h, "rate_gens_per_s": cfg.rate,
                    "stages_run": [], "deferred": [], "device": backend.device}
            prov["turns"].append(turn)
            t_idx = len(prov["turns"]) - 1
            save()
            rate = [cfg.rate]

            def ok_to_start(st: str) -> bool:
                return G2.can_start(time.time() - t_acq, stage_est_s(st, rate[0], n), 0.0, cap_s)

            extra = LOAD_S + (measured_a0_s(prov, n, rate[0]) if "A0" in done else 0.0)
            turn["start_check"] = {"load_s": LOAD_S, "a0_s": round(extra - LOAD_S, 1), "first_stage": gpu_todo[0]}
            if not G2.can_start(time.time() - t_acq + extra, stage_est_s(gpu_todo[0], rate[0], n), 0.0, cap_s):
                turn["deferred"] = list(gpu_todo)               # nothing fits: the model is not loaded at all
                print(f"[lbeam] cap: deferring {turn['deferred']} to the next turn (model not loaded)", flush=True)
                save()
                gpu_todo = []
            if gpu_todo:
                backend.wait()
                t1 = time.time()
                ad_sha = {f: sha256(Path(cfg.adapter_dir) / f) for f in ("adapter_model.safetensors", "adapter_config.json")}
                if fz is not None:
                    for f, h in ad_sha.items():
                        need(h == fz["inputs"][f"{ADAPTER_REL}/{f}"], f"adapter {f} hash at load differs from the freeze")
                model, tok, info = backend.load(cfg)
                try:
                    need(str(info["fingerprint"]).startswith(FP["sparse"]),
                         f"Titan-Sparse fingerprint {str(info['fingerprint'])[:12]} != {FP['sparse']}")
                    need(not LF.any_merged(model), "a LoRA layer is merged after load (D-28)")
                    prov["sparse_fingerprint"] = info["fingerprint"]
                    turn["adapter_sha256_at_load"] = ad_sha
                    turn["versions"] = info["versions"]
                    turn["runtime"] = backend.runtime_info()
                    rp = runtime_problem(turn["runtime"], info["versions"])
                    if rp:
                        save()
                        raise Stop(f"[lbeam] {rp}: restore the registered environment (nothing was generated)")
                    turn["load_s"] = round(time.time() - t1, 1)
                    save()
                    print(f"[lbeam] Titan-Sparse {str(info['fingerprint'])[:12]} + T2 s0 adapter loaded in {turn['load_s']}s",
                          flush=True)
                    if "A0" in done:                           # every resumed turn re-runs A0 before any planning
                        t_r = time.time()
                        _, _, res = run_anchor(model, tok, anchor_chains, cfg.anchor_refs)
                        turn["anchor_recheck"] = res
                        save()
                        bi_res = None
                        if res["pass"]:
                            _, bi_res = run_batch_invariance(model, tok, bi_keys, cfg)
                            turn["batchinv_recheck"] = bi_res
                        turn["a0_recheck_seconds"] = round(time.time() - t_r, 1)
                        save()
                        if not res["pass"] or not bi_res["pass"]:
                            write_json_x(out / "anchor" / f"RECHECK_FAILED_{int(time.time())}.json",
                                         {"turn": t_idx, "anchor": res, "batch_invariance": bi_res})
                            raise Stop(f"[lbeam] A0 RE-CHECK FAILED (anchor pass {res['pass']}, batch invariance "
                                       f"{None if bi_res is None else bi_res['pass']}); stop before any planning (deviation policy)")
                    for i, st in enumerate(gpu_todo):
                        if not ok_to_start(st):
                            turn["deferred"] = gpu_todo[i:]
                            print(f"[lbeam] cap: deferring {turn['deferred']} to the next turn", flush=True)
                            save()
                            break
                        prov["attempts"].setdefault(st, []).append(t_idx)
                        save()
                        t0 = time.time()
                        if st == "A0":
                            r_off, r_on, res = run_anchor(model, tok, anchor_chains, cfg.anchor_refs)
                            write_jsonl(out / FILES["A0"]["anchor_off"], chain_rows(anchor_chains, r_off))
                            write_jsonl(out / FILES["A0"]["anchor_on"], chain_rows(anchor_chains, r_on))
                            write_json_x(out / FILES["A0"]["anchor"], res)
                            prov["anchor"] = res
                            if not res["pass"]:
                                save()
                                raise Stop(f"[lbeam] ANCHOR FAILED {res['off']} {res['on']}: the instrument does not "
                                           "reproduce T2 s0; stop before any planning output (deviation policy)")
                            rows, bi_res = run_batch_invariance(model, tok, bi_keys, cfg)
                            write_jsonl(out / FILES["A0"]["bi_rows"], rows)
                            write_json_x(out / FILES["A0"]["bi"], bi_res)
                            prov["batch_invariance"] = bi_res
                            if not bi_res["pass"]:
                                save()
                                raise Stop(f"[lbeam] BATCH INVARIANCE FAILED {bi_res['arms']}: F depends on batch "
                                           "composition beyond the registered tolerance; stop before any planning output")
                            record("A0", {"seconds": round(time.time() - t0, 1), "pid": os.getpid(),
                                          "generations": stage_gens("A0", n), "sides": res["sides"]})
                        else:
                            r = run_planning_stage(st, BP, model, tok, cfg, out, insts, PI, own, done)
                            sec = time.time() - t0
                            rate[0] = max(0.5, r["n_new"] / max(sec, 1e-9)) if r["n_new"] else rate[0]   # memory only
                            record(st, {"seconds": round(sec, 1), "pid": os.getpid(), "sides": r["sides"],
                                        "generations_planned_expected": round(stage_gens(st, n, "expected")),
                                        "generations_planned_ceiling": round(stage_gens(st, n, "ceiling"))})
                        turn["stages_run"].append(st)
                        save()
                        print(f"[lbeam] {st} done in {time.time() - t0:.0f}s (planned ~{round(stage_gens(st, n))} "
                              "generations)", flush=True)
                finally:
                    del model
                    backend.unload()
            turn["released_unix"] = time.time()
            turn["no_progress"] = not turn["stages_run"]
            prov["no_progress_turns"] = (prov.get("no_progress_turns") or 0) + 1 if turn["no_progress"] else 0
            save()
    if "A0" in done and "C1" not in done:     # A0 passed in THIS invocation: C1 waits for the next one (exit 5), which
        print("[lbeam] C1 (CPU) runs at the start of the next invocation, before the lease and before any CUDA use",
              flush=True)                     # runs it before the lease in a process that never touched CUDA
    if turn is not None and turn["no_progress"] and prov["no_progress_turns"] >= NO_PROGRESS_STOP:
        raise Stop(f"[lbeam] no stage was recorded in {prov['no_progress_turns']} consecutive turns (deferred "
                   f"{turn['deferred']}): stop (deviation policy; a cap or rate amendment is needed)")
    if done | abandoned >= set(STAGE_ORDER):
        return finalize()
    print(f"[lbeam] remaining stages {[s for s in STAGE_ORDER if s not in done and s not in abandoned]} resume next turn",
          flush=True)
    return 5


def stop_record(out: Path, e: Stop) -> dict:
    """STOPPED.json content: the reason and whether the stop leaves a verdict-eligible run (A0, G1 and C1 recorded,
    every recorded output unchanged, the first A0 passed)."""
    rec = {"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)}
    pv = Path(out) / "provenance.json"
    prov = _read_json(pv) if pv.exists() else None
    eligible = False
    if prov:
        st = prov.get("stages") or {}
        unchanged = True
        for s, r in st.items():
            for p in stage_required(s, out) if s in FILES else []:
                if not p.is_file() or sha256(p) != (r.get("files") or {}).get(rel_out(p, out)):
                    unchanged = False
        a0 = (_read_json(Path(out) / "anchor" / "anchor.json") or {}).get("pass") is True and \
            (_read_json(Path(out) / "anchor" / "batch_invariance.json") or {}).get("pass") is True
        eligible = all(s in st for s in CORE_STAGES) and unchanged and a0
        rec.update({"provenance_sha256": sha256(pv), "stages_recorded": [s for s in STAGE_ORDER if s in st],
                    "abandoned": sorted(prov.get("abandoned") or {}), "recorded_outputs_unchanged": unchanged})
    rec["verdict_eligible"] = bool(eligible)
    return rec


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        rec = stop_record(out, e)
        with open(out / "STOPPED.json", "x", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
    except FileExistsError:
        pass


def run_main(cfg: Cfg, backend) -> int:
    """The job's exit-code mapping (main() and the tests): 0 / 5 from run(); a Stop -> printed, persisted when it
    concerns the run, exit 3; any other exception propagates (the interpreter exits 1, the hook retries once)."""
    try:
        return run(cfg, backend)
    except Stop as e:
        print(str(e), flush=True)
        persist_stop(Path(cfg.out), e)
        return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="CPU only, no torch import: projected generations, GPU time, turns")
    g.add_argument("--verify", action="store_true", help="CPU only: verify PREREG_LB_FREEZE.json (+ amendments)")
    g.add_argument("--freeze", action="store_true", help="CPU only: freeze PREREG-LB and mint the sealed draw")
    ap.add_argument("--approval", default=None, help="with --freeze: who approved, when, DECISIONS entry")
    ap.add_argument("--bwm-cosign", default=None,
                    help="with --freeze: BWM's co-sign (reference, this draft's full sha256, commit:<sha256 of the salt>)")
    ap.add_argument("--bwm-salt", default=None, help="with --freeze: the salt BWM reveals at the freeze")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan or a.verify or a.freeze:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
            os.environ[v] = "2"
    if not (a.plan or a.verify or a.freeze):
        return run_main(Cfg(cap_h=a.cap_h), RealBackend())
    try:
        if a.plan:
            res = plan()
            ok, errs, _ = verify_freeze()
            res["checks"] = {"pins_match": all((ROOT / r).is_file() and sha256(ROOT / r) == h for r, h in PINS.items()),
                             "bwm_planner_pinned": sha256(BWM_ROOT / BWM_PLANNER_REL) == BWM_PLANNER_SHA,
                             "lbeam_outputs_present": lbeam_outputs(ROOT), "freeze_ok": ok, "freeze_blockers": errs[:3],
                             "torch_imported": "torch" in sys.modules}
            print(json.dumps(res["checks"], indent=1))
            need("torch" not in sys.modules, "--plan must not import torch", persist=False)
            if (ROOT / FREEZE_REL).exists():
                print("PREREG-LB is frozen: GEN_PLAN_LB.json is a frozen input and is not rewritten")
                return 0
            DOCS.mkdir(parents=True, exist_ok=True)
            with open(DOCS / "GEN_PLAN_LB.json", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(res, indent=1) + "\n")
            print(f"wrote {DOCS / 'GEN_PLAN_LB.json'}")
            return 0
        if a.verify:
            ok, errs, _ = verify_freeze()
            print("PREREG-LB freeze verifies" if ok else "PREREG-LB freeze does NOT verify:\n  " + "\n  ".join(errs))
            need("torch" not in sys.modules, "--verify must not import torch", persist=False)
            return 0 if ok else 1
        freeze(a.approval or "", a.bwm_cosign or "", a.bwm_salt or "")
        return 0
    except Stop as e:
        print(str(e), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
