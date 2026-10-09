"""PREREG-3 GPU generation runner, freeze tool and freeze verifier (PREREG_3_DRAFT.md §7, §9, §10).
The GPU job refuses to run before PREREG-3 is frozen. Modelled on scripts/confirm2_gen.py (PREREG-2), whose
torch-free helpers (pilot-chain loader, D-22 stage rule, registered runtime, s0 pins) it imports.

  python -B scripts/p3_run.py --plan      # CPU only: no torch, no model, no lease. Generations, GPU time, turns; writes
                                          # docs/prereg3/GEN_PLAN.json (refused once frozen)
  python -B scripts/p3_run.py --verify    # CPU only: PREREG_3_FREEZE.json (+ PREREG_3_AMENDMENT_*) against the files
  python -B scripts/p3_run.py --freeze --approval "<who approved, when, DECISIONS entry>"
                                          # CPU only. Refuses if PREREG-3 is frozen (or a freeze was interrupted) or ANY
                                          # p3 output exists; re-verifies the bank, the scorer self-test, the dry run and
                                          # the tiny tests; hashes every input; writes PREREG_3.md + PREREG_3_FREEZE.json
                                          # (each to .partial first, then renamed; create-only)
  python -B scripts/p3_run.py             # the registered GPU job

GPU job. Before the lease: the freeze must verify; an installed scripts/queue_hooks/60_prereg3.sh must be the frozen
hook draft byte for byte; and no other RCWM job may hold or wait for the GPU lease (one-waiter rule, R14). One shared
lease (rcwmt.runtime.gpu_lease, label 'rcwm-t1:prereg3': alternation, thermal flag, duty cycle, RAM floor, the 4.5 h turn
cap from lock acquisition, D-22 stage rules). One model at a time:

  load Titan-Sparse (FP32, fingerprint 818be9c6b6a3) and the three T2 LoRAs unmerged (PEFT adapters t2s0/t2s1/t2s2,
  each hashed again at load; s0 = PREREG-2's registered artifact); torch / transformers / peft must equal the registered
  runtime before anything is generated. Sides: OFF (disable_adapter() = Titan-Sparse), s0, s1, s2 (set_adapter). Every
  stage generates all four sides in the SAME process and writes them only when all are done:
    A0 anchor   pilot TEST ledger chain_one (208 chains, generation.chain) per side; each side must reproduce its T2
                eval file (OFF: t2_s0 gen_chain_one_base; sN: t2_sN gen_chain_one) with identical states on >= 99% of
                chains (identical texts recorded as a diagnostic). This also shows that adapter switching in one model
                equals each adapter's own load. Else stop before any bank output. A resumed turn that runs S1 or S2
                re-anchors first; a failure is persisted.
    S1          the one-step rows of data/bank_p3_oor.jsonl (1,500) per side (generation.generate_texts, 160-token cap)
    S2          the 80 four-step chains per side (chain_one self-rollout in the one-step frame; recorded key orders)
  unload (VRAM checked empty; pause flag and resource floor re-checked); load Supa 1.7 (fingerprint a9952c55f5a1;
  runtime re-checked):
    S3          Supa on the one-step rows and the chains
  DONE.json (create-only, sha256 of every output) when every stage is complete. Provenance and the job log record only
  the PLANNED generation count of S2 and S3 (the actual count would reveal how many chains broke before the analysis).

Outputs (create-only) under runs/bank_p3_oor/. A stage is DONE only if all its files exist AND provenance.json records
them with matching sha256; anything else written by a stage makes it PARTIAL: moved to runs/bank_p3_oor/_orphans/ (never
deleted) and regenerated in full, all sides together. A recorded output whose hash changes stops the job.

Exit codes (for the queue hook): 0 complete (DONE.json) or already complete; 3 deterministic stop - never retried
(freeze, hook identity, one-waiter rule, runtime, hash, fingerprint, anchor, sanity; a stop that concerns the run is
persisted as STOPPED.json and the job refuses to run again until an operator moves it aside and logs why; "not frozen",
"hook not the frozen draft", "another RCWM job holds or waits for the lease" and "run lock held" are NOT persisted);
5 resumable (cap reached; the remaining stages run next turn); 1 crash (an exception; the lease is released) - the hook
retries once.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
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
import confirm2_gen as G2  # noqa: E402  (torch-free at import: loaders, D-22 rule, pins)
import p3_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

# ---------------------------------------------------------------- registered constants
LEASE_LABEL = "rcwm-t1:prereg3"
OUT = ROOT / "runs" / "bank_p3_oor"
DRAFT_REL, PREREG_REL, FREEZE_REL = "PREREG_3_DRAFT.md", "PREREG_3.md", "PREREG_3_FREEZE.json"
SIDES = ("off", "s0", "s1", "s2")
ADAPTER_NAMES = {"s0": "t2s0", "s1": "t2s1", "s2": "t2s2"}
ADAPTER_RELS = {s: f"runs/t2/t2_{s}/adapter" for s in ("s0", "s1", "s2")}
ANCHOR_REF_RELS = {"off": "runs/t2/t2_s0/gen_chain_one_base.jsonl", "s0": "runs/t2/t2_s0/gen_chain_one.jsonl",
                   "s1": "runs/t2/t2_s1/gen_chain_one.jsonl", "s2": "runs/t2/t2_s2/gen_chain_one.jsonl"}
S0_PINS = {"runs/t2/t2_s0/adapter/adapter_model.safetensors": G2.ADAPTER_SHA,
           "runs/t2/t2_s0/adapter/adapter_config.json": G2.ADAPTER_CONFIG_SHA,
           "runs/t2/t2_s0/gen_chain_one_base.jsonl": G2.ANCHOR_REF_SHA["gen_chain_one_base.jsonl"],
           "runs/t2/t2_s0/gen_chain_one.jsonl": G2.ANCHOR_REF_SHA["gen_chain_one.jsonl"]}
FP = dict(G2.FP)
REGISTERED_RUNTIME = dict(G2.REGISTERED_RUNTIME)
STAGE_ORDER = ("A0", "S1", "S2", "S3")
SPARSE_STAGES = ("A0", "S1", "S2")
ANCHOR_MIN_IDENTICAL = 0.99
RATE_GENS_PER_S = 2.7            # registered planning rate (PREREG-2's)
OBSERVED_RATE = 3.80             # PREREG-2's observed ledger chain_one rate on the sealed bank (S1: 9,588 gens / 2,526 s)
LOAD_S = 120.0
CAP_H = 4.5
VRAM_EMPTY_MIB = 512.0
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
HOOK_DRAFT_REL = "docs/prereg3/queue_hook_p3.sh.draft"
HOOK_INSTALLED_REL = "scripts/queue_hooks/60_prereg3.sh"
RCWM_LABEL_PREFIX = "rcwm-t1:"
REQUIRED_INPUTS = (
    DRAFT_REL, "scripts/p3_common.py", "scripts/p3_make_bank.py", "scripts/p3_run.py", "scripts/p3_analysis.py",
    "scripts/p3_test_tiny.py", "scripts/p3_wide_scan.py", "scripts/l1_make_ood.py", "scripts/confirm2_gen.py",
    "scripts/confirm2_analysis.py", "scripts/t2_analysis.py", "data/bank_p3_oor.jsonl", "data/bank_p3_oor.SOURCE.json",
    "data/bank_pilot.json",
    *[f"{d}/{f}" for d in ADAPTER_RELS.values() for f in ("adapter_model.safetensors", "adapter_config.json")],
    *ANCHOR_REF_RELS.values(),
    "docs/prereg3/GEN_PLAN.json", "docs/prereg3/SCORER_SELFTEST.json", "docs/prereg3/P3_DRYRUN.json",
    "docs/prereg3/P3_DRYRUN.md", "docs/prereg3/TINY_TEST.json", "docs/prereg3/WIDE_SCAN.json", HOOK_DRAFT_REL)
WIDE_SCAN_CODE = ("scripts/p3_wide_scan.py", "scripts/p3_make_bank.py", "scripts/p3_common.py")
PROV_CODE = ("scripts/p3_run.py", "scripts/p3_common.py", "rcwmt/generation.py", "rcwmt/runtime.py", "rcwmt/env.py",
             "rcwmt/capture.py")
# files that must carry the current code hash in the pre-freeze records (dry run, tiny test)
DRYRUN_CODE = ("scripts/p3_analysis.py", "scripts/p3_common.py")
TINY_CODE = ("scripts/p3_run.py", "scripts/p3_common.py", "scripts/p3_analysis.py", "scripts/p3_test_tiny.py")


class Stop(Exception):
    """A deterministic stop (exit 3, never retried). persist=True: written to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = True):
        super().__init__(msg)
        self.persist = persist


def sha256(p: Path) -> str:
    return G2.sha256(p)


def need(cond: bool, msg: str, persist: bool = True) -> None:
    if not cond:
        raise Stop(f"[prereg3] {msg}", persist)


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- configuration (the registered job uses the defaults)
@dataclass
class Cfg:
    root: Path = ROOT
    out: Path = OUT
    bank: Path = C.BANK
    source: Path = C.SOURCE
    anchor_chains: list | None = None            # None: pilot TEST ledger chains (208)
    anchor_refs: dict = field(default_factory=lambda: {s: ROOT / r for s, r in ANCHOR_REF_RELS.items()})
    adapter_dirs: dict = field(default_factory=lambda: {s: ROOT / r for s, r in ADAPTER_RELS.items()})
    require_freeze: bool = True                  # False only from the tiny tests (never reachable from the CLI)
    cap_h: float = CAP_H
    rate: float = RATE_GENS_PER_S


def pilot_anchor_chains() -> list:
    return G2.load_bank_chains(G2.PILOT_PATH, G2.PILOT_SHA, "L", "test")


# ---------------------------------------------------------------- stage bookkeeping (no torch)
def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(stage: str, out: Path) -> list[Path]:
    o = Path(out)
    if stage == "A0":
        return [o / "anchor" / f"chain_one_L_{s}.jsonl" for s in SIDES] + [o / "anchor" / "anchor.json"]
    if stage == "S1":
        return [o / "onestep" / f"onestep_{s}.jsonl" for s in SIDES]
    if stage == "S2":
        return [o / "chains" / f"chains_{s}.jsonl" for s in SIDES]
    if stage == "S3":
        return [o / "supa" / "onestep_supa.jsonl", o / "supa" / "chains_supa.jsonl"]
    raise KeyError(stage)


def stage_owned(stage: str, out: Path) -> list[Path]:
    o = Path(out)
    if stage == "A0":
        return [o / "anchor"]
    fs = stage_required(stage, o)
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
    print(f"[prereg3] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Stop(f"[prereg3] {path} exists (create-only)")
    tmp = _partial(path)
    with tmp.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- generation counts, estimates, the D-22 stage rule
def counts(cfg: Cfg | None = None) -> dict:
    cfg = cfg or Cfg()
    rows, chains = C.load_bank(cfg.bank)
    n_anchor = len(cfg.anchor_chains) if cfg.anchor_chains is not None else len(pilot_anchor_chains())
    return {"rows": len(rows), "chains": len(chains), "anchor_chains": n_anchor,
            "A0": n_anchor * C.H * len(SIDES), "S1": len(rows) * len(SIDES), "S2": len(chains) * C.H * len(SIDES),
            "S3": len(rows) + len(chains) * C.H}


def stage_est_s(st: str, rate: float, n: dict) -> float:
    return (LOAD_S if st == "S3" else 0.0) + n[st] / rate


def supa_reserve_s(rate: float, n: dict) -> float:
    return G2.STAGE_SAFETY * stage_est_s("S3", rate, n)


def simulate_turns(rate: float, n: dict, cap_h: float = CAP_H) -> list[dict]:
    cap = cap_h * 3600
    left = list(STAGE_ORDER)
    turns = []
    while left:
        t, ran = 0.0, []
        sparse_left = [s for s in left if s != "S3"]
        if sparse_left:
            t += LOAD_S
            if "A0" not in left:
                t += stage_est_s("A0", rate, n)
                ran.append("A0 (re-anchor)")
            for st in sparse_left:
                res = supa_reserve_s(rate, n) if "S3" in left else 0.0
                if not G2.can_start(t, stage_est_s(st, rate, n), res, cap):
                    break
                t += stage_est_s(st, rate, n)
                ran.append(st)
                left.remove(st)
        if "S3" in left and G2.can_start(t, stage_est_s("S3", rate, n), 0.0, cap):
            t += stage_est_s("S3", rate, n)
            ran.append("S3")
            left.remove("S3")
        if not [r for r in ran if not r.startswith("A0 (")]:
            raise RuntimeError(f"no stage fits an empty turn at {rate} gens/s")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(verbose: bool = True) -> dict:
    n = counts()
    what = {"A0": f"anchor: pilot TEST ledger chain_one ({n['anchor_chains']} chains x {C.H} steps) x OFF/s0/s1/s2",
            "S1": f"one-step bank rows ({n['rows']}) x OFF/s0/s1/s2",
            "S2": f"bank chains ({n['chains']} x {C.H} steps) x OFF/s0/s1/s2",
            "S3": "Supa 1.7 (after unloading Sparse): one-step rows + chains"}
    res = {"rate_gens_per_s": RATE_GENS_PER_S, "observed_rate_gens_per_s": OBSERVED_RATE, "load_s_each": LOAD_S, "cap_h": CAP_H,
           "stages": [{"stage": s, "what": what[s], "generations_max": n[s],
                       "projected_s_planning": round(stage_est_s(s, RATE_GENS_PER_S, n), 1),
                       "projected_s_observed": round(stage_est_s(s, OBSERVED_RATE, n), 1)} for s in STAGE_ORDER],
           "total_generations_max": sum(n[s] for s in STAGE_ORDER), "model_loads": 2,
           "stage_rule": f"no stage starts within {G2.STAGE_MARGIN_H} h of the cap; a stage starts only if "
                         f"{G2.STAGE_SAFETY} x its estimate + the Supa reserve ({G2.STAGE_SAFETY} x S3) fits; a resumed "
                         "turn with S1/S2 re-anchors",
           "turns_at_planning_rate": simulate_turns(RATE_GENS_PER_S, n),
           "turns_at_observed_rate": simulate_turns(OBSERVED_RATE, n)}
    res["projected_gpu_h_planning"] = round(sum(t["projected_h"] for t in res["turns_at_planning_rate"]), 2)
    res["projected_gpu_h_observed"] = round(sum(t["projected_h"] for t in res["turns_at_observed_rate"]), 2)
    if verbose:
        print(f"PREREG-3 GPU plan (no model loaded; {LOAD_S:.0f} s per model load)")
        for r in res["stages"]:
            print(f"  {r['stage']}  {r['what']:<76} gens<= {r['generations_max']:>6}  {r['projected_s_planning'] / 60:6.1f} min "
                  f"@{RATE_GENS_PER_S}/s  {r['projected_s_observed'] / 60:6.1f} min @{OBSERVED_RATE}/s")
        print(f"  total generations <= {res['total_generations_max']:,}")
        for k, rate in (("turns_at_planning_rate", RATE_GENS_PER_S), ("turns_at_observed_rate", OBSERVED_RATE)):
            print(f"  turns at {rate} gens/s (cap {CAP_H} h): " +
                  "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(res[k])))
    return res


# ---------------------------------------------------------------- freeze verification (shared with p3_analysis.py)
def _num(rx: str, name: str) -> int:
    m = re.search(rx, name)
    return int(m.group(1)) if m else -1


def required_inputs(root: Path) -> set:
    return set(REQUIRED_INPUTS) | {f"rcwmt/{p.name}" for p in (Path(root) / "rcwmt").glob("*.py")}


def verify_freeze(root: Path | None = None) -> tuple[bool, list[str], dict | None]:
    root = Path(root or ROOT)
    fz = _read_json(root / FREEZE_REL)
    if fz is None:
        half = interrupted_freeze(root)
        return False, [half or "PREREG_3_FREEZE.json does not exist or is unreadable (PREREG-3 is a draft)"], None
    errs = []
    pr = root / PREREG_REL
    if not pr.is_file():
        errs.append("PREREG_3.md does not exist")
    elif sha256(pr) != fz.get("prereg_sha256"):
        errs.append("PREREG_3.md differs from the freeze")
    inputs = dict(fz.get("inputs") or {})
    missing = sorted(required_inputs(root) - set(inputs))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing}")
    for rel, h in S0_PINS.items():
        if inputs.get(rel) != h:
            errs.append(f"{rel}: freeze record differs from the PREREG-2 pin")
    expected = dict(inputs)
    for a in sorted(root.glob("PREREG_3_AMENDMENT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name)):
        am = _read_json(a) or {}
        if am.get("file") not in expected or expected[am["file"]] != am.get("old_sha256") or not am.get("new_sha256"):
            errs.append(f"{a.name}: old hash does not match the chain")
            continue
        expected[am["file"]] = am["new_sha256"]
    for rel, h in sorted(expected.items()):
        p = root / rel
        if not p.is_file():
            errs.append(f"{rel} missing")
        elif sha256(p) != h:
            errs.append(f"{rel} differs from the freeze (+ amendments)")
    if fz.get("registered_runtime") != REGISTERED_RUNTIME:
        errs.append("registered runtime differs")
    if not fz.get("approval"):
        errs.append("no approval recorded")
    return not errs, errs, fz


def allowed_hashes(fz: dict, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    allowed = {k: {v} for k, v in (fz.get("inputs") or {}).items()}
    for a in sorted(root.glob("PREREG_3_AMENDMENT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name)):
        am = _read_json(a) or {}
        if am.get("file") in allowed and am.get("new_sha256"):
            allowed[am["file"]].add(am["new_sha256"])
    return allowed


def p3_outputs(root: Path) -> list[str]:
    """Anything a PREREG-3 run or analysis writes. The freeze refuses if one exists."""
    root = Path(root)
    found = []
    runs = root / "runs"
    if runs.exists():
        for p in runs.iterdir():
            if p.name.startswith("bank_p3_oor") or "prereg3" in p.name.lower():
                found.append(str(p.relative_to(root)))
        found += [str(p.relative_to(root)) for p in runs.rglob("PREREG_3_RESULTS*")]
    return sorted(set(found))


def _fill_prereg(text: str, stamp: str, inputs: dict, approval: str) -> str:
    lines = text.splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("**Status:**")]
    if len(st) != 1:
        raise Stop("the draft has no unique **Status:** line", persist=False)
    lines[st[0]] = (f"**Status:** FROZEN at {stamp} by `scripts/p3_run.py --freeze` (approval: {approval}). The SHA-256 "
                    "of this file is in `PREREG_3_FREEZE.json`.")
    i10 = [i for i, l in enumerate(lines) if l.startswith("## 10.")]
    if len(i10) != 1:
        raise Stop("the draft has no unique §10", persist=False)
    nxt = next((i for i in range(i10[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    sec = ["## 10. Freeze record", "", f"Frozen at {stamp} by `scripts/p3_run.py --freeze`. SHA-256 of every frozen input:", ""]
    sec += [f"- `{k}` `{v}`" for k, v in sorted(inputs.items())]
    sec.append("")
    return "\n".join(lines[: i10[0]] + sec + lines[nxt:]) + "\n"


def prefreeze_checks(root: Path, bank_verify=None, selftest=None) -> dict:
    """Everything that must hold before PREREG-3 can freeze (also run by the tiny test on a fake root)."""
    root = Path(root)
    out = {}
    src = _read_json(root / "data" / "bank_p3_oor.SOURCE.json") or {}
    out["bank sha256 == SOURCE record"] = (root / "data" / "bank_p3_oor.jsonl").is_file() and \
        sha256(root / "data" / "bank_p3_oor.jsonl") == src.get("file_sha256")
    if bank_verify is None:
        import p3_make_bank as MB
        res = MB.verify(root / "data" / "bank_p3_oor.jsonl", root / "data" / "bank_p3_oor.SOURCE.json", historical=True, quiet=True)
        out["bank verify (incl. a fresh name-skipping harvest)"] = all(ok for _, ok, _ in res)
    else:
        out["bank verify (incl. a fresh name-skipping harvest)"] = bool(bank_verify())
    if selftest is None:
        import p3_analysis as PA
        st = PA.scorer_selftest(root / "data" / "bank_p3_oor.jsonl")
        out["scorer self-test (100% on golds, copy floor exact, negatives 0%)"] = bool(st["ok"])
    else:
        out["scorer self-test (100% on golds, copy floor exact, negatives 0%)"] = bool(selftest())
    dr = _read_json(root / "docs" / "prereg3" / "P3_DRYRUN.json") or {}
    out["dry run on synthetic fixtures passed with the current analysis code"] = bool(dr.get("ok")) and all(
        (dr.get("code_sha256") or {}).get(r) == sha256(root / r) for r in DRYRUN_CODE)
    tt = _read_json(root / "docs" / "prereg3" / "TINY_TEST.json") or {}
    out["tiny-model runner tests passed with the current code"] = bool(tt.get("ok")) and all(
        (tt.get("code_sha256") or {}).get(r) == sha256(root / r) for r in TINY_CODE)
    sf = _read_json(root / "docs" / "prereg3" / "SCORER_SELFTEST.json") or {}
    out["recorded scorer self-test passed on this bank"] = bool(sf.get("ok")) and sf.get("bank_sha256") == src.get("file_sha256")
    ws = _read_json(root / "docs" / "prereg3" / "WIDE_SCAN.json") or {}
    out["wide freshness scan: zero overlaps, on this bank, with the current code"] = bool(ws.get("ok")) and \
        ws.get("bank_sha256") == src.get("file_sha256") and all(
            (ws.get("code_sha256") or {}).get(r) == sha256(root / r) for r in WIDE_SCAN_CODE)
    gp = _read_json(root / "docs" / "prereg3" / "GEN_PLAN.json") or {}
    out["GEN_PLAN.json present"] = bool(gp.get("stages"))
    out["queue hook draft present"] = (root / "docs" / "prereg3" / "queue_hook_p3.sh.draft").is_file()
    return out


def _partial_rel(rel: str) -> str:
    return rel + ".partial"


def interrupted_freeze(root: Path) -> str | None:
    """A freeze writes PREREG_3.md.partial and PREREG_3_FREEZE.json.partial, then renames PREREG_3.md and then
    PREREG_3_FREEZE.json into place. Anything else on disk is an interrupted freeze; returns the recovery instruction."""
    root = Path(root)
    pr, fz = root / PREREG_REL, root / FREEZE_REL
    prp, fzp = root / _partial_rel(PREREG_REL), root / _partial_rel(FREEZE_REL)
    if pr.exists() and not fz.exists():
        rec = _read_json(fzp) or {}
        if fzp.exists() and rec.get("prereg_sha256") == sha256(pr):
            return ("the freeze was interrupted between its two renames: PREREG_3.md is in place and "
                    "PREREG_3_FREEZE.json.partial records its sha256. Rename PREREG_3_FREEZE.json.partial to "
                    "PREREG_3_FREEZE.json, run --verify, and log it in DECISIONS.md")
        return ("PREREG_3.md exists without PREREG_3_FREEZE.json (and without a matching .partial record): not a "
                "freeze. Move PREREG_3.md (and any .partial) aside, log it in DECISIONS.md, and rerun --freeze")
    if not pr.exists() and (prp.exists() or fzp.exists()):
        return ("a freeze was interrupted before its renames (.partial files exist, nothing is frozen): move the "
                ".partial files aside, log it in DECISIONS.md, and rerun --freeze")
    return None


def freeze(approval: str, root: Path | None = None, bank_verify=None, selftest=None) -> dict:
    root = Path(root or ROOT)
    if not approval or len(approval.strip()) < 8:
        raise Stop("--approval must state who approved the freeze, when, and the DECISIONS entry", persist=False)
    half = interrupted_freeze(root)
    if half:
        raise Stop(f"refusing to freeze: {half}", persist=False)
    if (root / FREEZE_REL).exists() or (root / PREREG_REL).exists():
        raise Stop("PREREG-3 is already frozen (PREREG_3_FREEZE.json or PREREG_3.md exists)", persist=False)
    outs = p3_outputs(root)
    if outs:
        raise Stop(f"PREREG-3 outputs exist before the freeze: {outs}; investigate before freezing", persist=False)
    if not (root / DRAFT_REL).is_file():
        raise Stop("PREREG_3_DRAFT.md missing", persist=False)
    chk = prefreeze_checks(root, bank_verify, selftest)
    bad = [k for k, v in chk.items() if not v]
    if bad:
        raise Stop(f"pre-freeze checks failed: {bad}", persist=False)
    inputs = {}
    for rel in sorted(required_inputs(root)):
        if not (root / rel).is_file():
            raise Stop(f"frozen input missing: {rel}", persist=False)
        inputs[rel] = sha256(root / rel)
    for rel, h in S0_PINS.items():
        if inputs.get(rel) != h:
            raise Stop(f"{rel} is not PREREG-2's pinned file", persist=False)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    text = _fill_prereg((root / DRAFT_REL).read_text(encoding="utf-8"), stamp, inputs, approval.strip())
    prp, fzp = root / _partial_rel(PREREG_REL), root / _partial_rel(FREEZE_REL)
    with prp.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    rec = {"frozen_at": stamp, "approval": approval.strip(), "prereg_sha256": sha256(prp),
           "draft_sha256": sha256(root / DRAFT_REL), "inputs": inputs, "prefreeze_checks": chk,
           "registered_runtime": REGISTERED_RUNTIME, "lease_label": LEASE_LABEL, "out_dir": "runs/bank_p3_oor",
           "no_outputs_before_freeze": True}
    with fzp.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    os.rename(prp, root / PREREG_REL)                  # os.rename refuses an existing target on Windows (create-only)
    os.rename(fzp, root / FREEZE_REL)
    ok, errs, _ = verify_freeze(root)
    if not ok:
        raise Stop(f"the freeze record was written but does not verify: {errs}", persist=False)
    print(f"[prereg3] FROZEN {stamp}; PREREG_3.md sha {rec['prereg_sha256']}", flush=True)
    return rec


# ---------------------------------------------------------------- run lock (one runner process per output dir)
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
            raise Stop(f"[prereg3] another p3_run.py holds the run lock {rec}", persist=False)
        print(f"[prereg3] reclaiming a stale run lock {rec}", flush=True)
        lk.unlink()
    with lk.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return lk


# ---------------------------------------------------------------- one-waiter rule, installed hook, runtime
def _live_other(pid, since) -> bool:
    """pid is alive, is not this process, and existed when the record was written (not a reused pid)."""
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
    """R14 one-waiter rule: refuse if another RCWM job (label rcwm-t1:*) waits for the GPU lease (the rcwm waiting
    marker names a live process other than this one) or holds it (the lock record's label is rcwm-t1:* and its process
    is alive). The peer's (BWM) jobs are allowed: this job then waits in the lease as usual."""
    w = _read_json(backend.waiting_marker_path())
    if w and _live_other(w.get("pid"), w.get("since")):
        return f"another RCWM job waits for the GPU lease ({w.get('label')}, pid {w.get('pid')})"
    lk = _read_json(backend.lock_path())
    if lk and str(lk.get("label", "")).startswith(RCWM_LABEL_PREFIX) and _live_other(lk.get("pid"), lk.get("created")):
        return f"another RCWM job holds the GPU lease ({lk.get('label')}, pid {lk.get('pid')})"
    return None


def hook_problem(root: Path, fz: dict | None) -> str | None:
    """An installed scripts/queue_hooks/60_prereg3.sh must be the hook draft byte for byte: the frozen sha256 (with the
    amendment chain) once frozen, else the current draft."""
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
    """The registered GPU path: rcwmt.runtime (lease, numerics, loads) and PEFT exactly as PREREG-2 loads the adapter."""
    device = "cuda"

    def __init__(self):
        import torch  # noqa: F401
        from rcwmt import runtime
        self.rt = runtime

    def lease(self):
        return self.rt.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 PREREG-3 rule-or-trigger generation (Sparse + T2 s0/s1/s2 OFF/ON, Supa)")

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
            print("[prereg3] thermal pause flag set before a model load; waiting", flush=True)
            while THERMAL_FLAG.exists():
                time.sleep(5)

    def wait(self) -> None:
        self.thermal_wait()
        self.rt.wait_for_resources()

    def load_sparse(self, adapter_dirs: dict):
        from peft import PeftModel
        import peft
        base, p = self.rt.load_arm("sparse")
        model = PeftModel.from_pretrained(base, str(adapter_dirs["s0"]), adapter_name=ADAPTER_NAMES["s0"], is_trainable=False)
        for s in ("s1", "s2"):
            model.load_adapter(str(adapter_dirs[s]), adapter_name=ADAPTER_NAMES[s], is_trainable=False)
        model.eval()
        return model, p["param_fingerprint"], {"peft": peft.__version__}

    def load_supa(self):
        m, p = self.rt.load_arm("supa")
        return m, p["param_fingerprint"]

    def tokenizer(self):
        return self.rt.tokenizer()

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
    """OFF = every adapter disabled (Titan-Sparse); sN = only adapter t2sN active."""
    if side == "off":
        with model.disable_adapter():
            yield
        return
    name = ADAPTER_NAMES[side]
    model.set_adapter(name)
    act = model.active_adapter
    need(act == name or act == [name], f"active adapter {act} != {name}")
    yield


def p3_chain(model, tok, chains: list) -> list:
    """chain_one self-rollout over bank chains: rcwmt.generation.chain's loop for one family, with the one-step frame
    built by p3_common.step_messages (== env.one_step_messages when a step has no recorded key order) and the parser
    p3_common.parse_ledger (== env.parse_state for A-D). None once a chain breaks."""
    from rcwmt import generation
    states = [[c["s0"]] for c in chains]
    texts = [[] for _ in chains]
    H = max(len(c["events"]) for c in chains)
    for t in range(H):
        live = [i for i, c in enumerate(chains) if t < len(c["events"]) and states[i][-1] is not None]
        if live:
            msgs = [C.step_messages(chains[i], states[i][-1], t) for i in live]
            txt = generation.generate_texts(model, tok, msgs, generation.max_tokens(C.LEDGER))
            for i, s in zip(live, txt):
                texts[i].append(s)
                states[i].append(C.parse_ledger(s, C.chain_keys(chains[i])))
        for i, c in enumerate(chains):
            if t < len(c["events"]) and states[i][-1] is None and len(states[i]) == t + 1:
                states[i].append(None)
    return [{"states": s, "texts": x} for s, x in zip(states, texts)]


def chain_rows(chains, results) -> list:
    return [{"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]], "texts": r["texts"]}
            for c, r in zip(chains, results)]


def onestep_texts(model, tok, rows) -> list:
    from rcwmt import generation
    return generation.generate_texts(model, tok, [r["messages"] for r in rows], generation.max_tokens(C.LEDGER))


def _ref_rows(path: Path) -> dict:
    ref = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ref[row["cand_id"]] = row
    return ref


def run_anchor(model, tok, chains, refs: dict) -> tuple[dict, dict]:
    """Pilot TEST ledger chain_one per side against that side's T2 eval file (identical batch composition). The gate is
    identical STATES on >= 99% of chains; identical texts are recorded as a diagnostic (as PREREG-2's anchor did)."""
    from rcwmt import generation
    out, res = {}, {"pid": os.getpid(), "unix": time.time(), "sides": {}}
    for s in SIDES:
        t0 = time.time()
        with side_ctx(model, s):
            rr = generation.chain(model, tok, chains, "chain_one")
        ref = _ref_rows(refs[s])
        st = sum([None if x is None else env.dumps(x) for x in r["states"]] == (ref.get(c["cand_id"]) or {}).get("states")
                 for c, r in zip(chains, rr))
        tx = sum(r["texts"] == (ref.get(c["cand_id"]) or {}).get("texts") for c, r in zip(chains, rr))
        out[s] = rr
        res["sides"][s] = {"n": len(chains), "identical_states": st / len(chains), "identical_texts": tx / len(chains),
                           "start_unix": t0, "end_unix": time.time(), "generations": sum(len(r["texts"]) for r in rr)}
    res["pass"] = all(res["sides"][s]["identical_states"] >= ANCHOR_MIN_IDENTICAL for s in SIDES)
    return out, res


# ---------------------------------------------------------------- the job
def run(cfg: Cfg, backend) -> int:
    out = Path(cfg.out)
    fz = None
    if cfg.require_freeze:
        ok, errs, fz = verify_freeze(cfg.root)
        if not ok:
            # "not frozen" says nothing about the run: persisted only when a freeze record exists but fails
            raise Stop("refusing to run the PREREG-3 GPU job: " + "; ".join(errs),
                       persist=(Path(cfg.root) / FREEZE_REL).exists())
        ins = fz["inputs"]
        need(sha256(cfg.bank) == ins["data/bank_p3_oor.jsonl"], "bank hash differs from the freeze")
        for s, d in cfg.adapter_dirs.items():
            for f in ("adapter_model.safetensors", "adapter_config.json"):
                need(sha256(Path(d) / f) == ins[f"{ADAPTER_RELS[s]}/{f}"], f"adapter {s} {f} differs from the freeze")
        for s, p in cfg.anchor_refs.items():
            need(sha256(p) == ins[ANCHOR_REF_RELS[s]], f"anchor reference {s} differs from the freeze")
    hp = hook_problem(cfg.root, fz)
    if hp:
        raise Stop(f"[prereg3] {hp}; reinstall it byte for byte from {HOOK_DRAFT_REL} (or remove it) before any run",
                   persist=False)
    if (out / "DONE.json").exists():
        print(f"[prereg3] {out / 'DONE.json'} exists: the job is complete", flush=True)
        return 0
    if (out / "STOPPED.json").exists():
        raise Stop(f"{out / 'STOPPED.json'} exists: a deterministic stop was persisted; an operator must investigate, "
                   "log it in DECISIONS.md and move the file aside before any rerun", persist=False)
    anchor_json = out / "anchor" / "anchor.json"
    if anchor_json.exists() and not (_read_json(anchor_json) or {}).get("pass"):
        raise Stop("the anchor check failed in an earlier turn (deviation policy: investigate, log)")
    if list((out / "anchor").glob("RECHECK_FAILED_*.json")):
        raise Stop("an anchor re-check failed in an earlier turn (deviation policy: investigate, log)")
    rows, chains = C.load_bank(cfg.bank)
    src = _read_json(cfg.source) or {}
    need(sha256(cfg.bank) == src.get("file_sha256"), "bank sha256 differs from its SOURCE record")
    anchor_chains = cfg.anchor_chains if cfg.anchor_chains is not None else pilot_anchor_chains()
    ow = one_waiter_problem(backend)
    if ow:
        raise Stop(f"[prereg3] {ow}: one-waiter rule (R14). Start PREREG-3 only when no other RCWM GPU job is running "
                   "or waiting (never next to a live master chain)", persist=False)
    lock = acquire_run_lock(out)
    try:
        return _run_locked(cfg, backend, out, fz, rows, chains, anchor_chains)
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _run_locked(cfg, backend, out, fz, rows, chains, anchor_chains) -> int:
    prov_path = out / "provenance.json"
    prov = _read_json(prov_path) if prov_path.exists() else None
    if prov_path.exists():
        need(prov is not None, "provenance.json unreadable")
    else:
        prov = {"bank_sha256": sha256(cfg.bank), "freeze_prereg_sha256": (fz or {}).get("prereg_sha256"),
                "registered": bool(cfg.require_freeze), "adapters": {}, "anchor_refs": {}, "stages": {}, "turns": []}
    for s, d in cfg.adapter_dirs.items():
        prov["adapters"][s] = {f: sha256(Path(d) / f) for f in ("adapter_model.safetensors", "adapter_config.json")}
    for s, p in cfg.anchor_refs.items():
        prov["anchor_refs"][s] = sha256(p)
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
        print("[prereg3] all stages complete; DONE.json written", flush=True)

    todo = [st for st in STAGE_ORDER if st not in done]
    print(f"[prereg3] stages to run: {todo}", flush=True)
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
        rate = [cfg.rate]

        def ok_to_start(st: str) -> bool:
            reserve = supa_reserve_s(rate[0], n) if (st != "S3" and "S3" not in done) else 0.0
            return G2.can_start(time.time() - t_acq, stage_est_s(st, rate[0], n), reserve, cap_s)

        def first_fits(st: str) -> bool:
            """Before loading: the first sparse stage must fit after the re-anchor a resumed turn runs first."""
            extra = stage_est_s("A0", rate[0], n) if "A0" in done else 0.0
            reserve = supa_reserve_s(rate[0], n) if "S3" not in done else 0.0
            return G2.can_start(time.time() - t_acq + extra, stage_est_s(st, rate[0], n), reserve, cap_s)

        sparse_todo = [st for st in todo if st != "S3"]
        if sparse_todo and not first_fits(sparse_todo[0]):
            turn["deferred"] = list(sparse_todo)          # nothing fits: do not load the model at all
            print(f"[prereg3] cap: deferring {turn['deferred']} to the next turn", flush=True)
            save()
            sparse_todo = []
        if sparse_todo:
            backend.wait()
            t1 = time.time()
            ad_sha = {s: {f: sha256(Path(d) / f) for f in ("adapter_model.safetensors", "adapter_config.json")}
                      for s, d in cfg.adapter_dirs.items()}
            if fz is not None:
                for s in ad_sha:
                    for f, h in ad_sha[s].items():
                        need(h == fz["inputs"][f"{ADAPTER_RELS[s]}/{f}"], f"adapter {s} {f} hash at load")
            model, fp_sparse, extra_versions = backend.load_sparse(cfg.adapter_dirs)
            try:
                backend.check_fingerprint("sparse", fp_sparse)
                prov["sparse_fingerprint"] = fp_sparse
                turn["adapter_sha256_at_load"] = ad_sha
                turn["versions"] = extra_versions
                turn["runtime"] = backend.runtime_info()
                rp = runtime_problem(turn["runtime"], extra_versions)
                if rp:
                    save()
                    raise Stop(f"[prereg3] {rp}: restore the registered environment (nothing was generated)")
                turn["sparse_load_s"] = round(time.time() - t1, 1)
                tok = backend.tokenizer()
                print(f"[prereg3] Titan-Sparse {fp_sparse[:12]} + T2 s0/s1/s2 adapters loaded in {turn['sparse_load_s']}s", flush=True)
                save()
                if "A0" in done:                           # a resumed turn re-anchors before any bank stage
                    _, res = run_anchor(model, tok, anchor_chains, cfg.anchor_refs)
                    turn["anchor_recheck"] = res
                    save()
                    if not res["pass"]:
                        with (out / "anchor" / f"RECHECK_FAILED_{int(time.time())}.json").open("x", encoding="utf-8") as fh:
                            fh.write(json.dumps(res, indent=1) + "\n")
                        raise Stop(f"[prereg3] ANCHOR RECHECK FAILED {res['sides']}; stop (deviation policy)")
                for i, st in enumerate(sparse_todo):
                    if not ok_to_start(st):
                        turn["deferred"] = sparse_todo[i:]
                        print(f"[prereg3] cap: deferring {turn['deferred']} to the next turn", flush=True)
                        save()
                        break
                    t0 = time.time()
                    sides_t = {}
                    if st == "A0":
                        rr, res = run_anchor(model, tok, anchor_chains, cfg.anchor_refs)
                        for s in SIDES:
                            write_jsonl(out / "anchor" / f"chain_one_L_{s}.jsonl", chain_rows(anchor_chains, rr[s]))
                        with (out / "anchor" / "anchor.json").open("x", encoding="utf-8") as fh:
                            fh.write(json.dumps(res, indent=1) + "\n")
                        prov["anchor"] = res
                        n_g = sum(res["sides"][s]["generations"] for s in SIDES)
                        if not res["pass"]:
                            save()
                            raise Stop(f"[prereg3] ANCHOR FAILED {res['sides']}: the instrument does not reproduce the T2 "
                                       "eval files; stop before any bank generation (deviation policy)")
                        sides_t = res["sides"]
                    elif st == "S1":
                        res_t = {}
                        for s in SIDES:
                            ts = time.time()
                            with side_ctx(model, s):
                                res_t[s] = onestep_texts(model, tok, rows)
                            sides_t[s] = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                        for s in SIDES:
                            write_jsonl(out / "onestep" / f"onestep_{s}.jsonl",
                                        [{"id": r["id"], "text": t} for r, t in zip(rows, res_t[s])])
                        n_g = len(rows) * len(SIDES)
                    else:
                        res_c = {}
                        for s in SIDES:
                            ts = time.time()
                            with side_ctx(model, s):
                                res_c[s] = p3_chain(model, tok, chains)
                            sides_t[s] = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                        for s in SIDES:
                            write_jsonl(out / "chains" / f"chains_{s}.jsonl", chain_rows(chains, res_c[s]))
                        n_g = sum(len(r["texts"]) for s in SIDES for r in res_c[s])   # in memory only (rate)
                    sec = time.time() - t0
                    rate[0] = max(0.5, n_g / max(sec, 1e-9))
                    rec_g = {"generations": n_g} if st in ("A0", "S1") else {}
                    record(st, {**rec_g, "generations_planned_max": n[st], "seconds": round(sec, 1), "pid": os.getpid(),
                                "sides": sides_t})
                    turn["stages_run"].append(st)
                    save()
                    print(f"[prereg3] {st} done in {sec:.0f}s (planned <= {n[st]} generations)", flush=True)
            finally:
                del model
                backend.unload()
        if "S3" not in done:
            if ok_to_start("S3"):
                alloc = backend.vram_allocated_mib()
                need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated after unloading Titan-Sparse; not loading Supa",
                     persist=False)
                backend.wait()
                t0 = time.time()
                supa, fp_supa = backend.load_supa()
                try:
                    backend.check_fingerprint("supa", fp_supa)
                    prov["supa_fingerprint"] = fp_supa
                    turn["supa_runtime"] = backend.runtime_info()
                    save()
                    rp = runtime_problem(turn["supa_runtime"])
                    if rp:
                        raise Stop(f"[prereg3] Supa load: {rp}: restore the registered environment (S3 not generated)")
                    tok = backend.tokenizer()
                    ts = time.time()
                    tx = onestep_texts(supa, tok, rows)
                    t_one = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                    ts = time.time()
                    rc = p3_chain(supa, tok, chains)
                    t_ch = {"pid": os.getpid(), "start_unix": ts, "end_unix": time.time()}
                    write_jsonl(out / "supa" / "onestep_supa.jsonl", [{"id": r["id"], "text": t} for r, t in zip(rows, tx)])
                    write_jsonl(out / "supa" / "chains_supa.jsonl", chain_rows(chains, rc))
                finally:
                    del supa
                    backend.unload()
                record("S3", {"generations_planned_max": n["S3"], "seconds": round(time.time() - t0, 1), "pid": os.getpid(),
                              "sides": {"onestep_supa": t_one, "chains_supa": t_ch}})
                turn["stages_run"].append("S3")
            else:
                turn["deferred"].append("S3")
        turn["released_unix"] = time.time()
        save()
    if done >= set(STAGE_ORDER):
        finalize()
        return 0
    print(f"[prereg3] turn ended; remaining stages {[s for s in STAGE_ORDER if s not in done]} resume next turn", flush=True)
    return 5


def persist_stop(out: Path, e: Stop) -> None:
    if not e.persist:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "STOPPED.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"unix": time.time(), "local": iso(time.time()), "pid": os.getpid(), "reason": str(e)}, indent=1) + "\n")
    except FileExistsError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="CPU only: no torch import, no model; projected generations and GPU time")
    g.add_argument("--verify", action="store_true", help="CPU only: verify PREREG_3_FREEZE.json (+ amendments) against the files")
    g.add_argument("--freeze", action="store_true", help="CPU only: freeze PREREG-3 (refuses if any p3 output exists)")
    ap.add_argument("--approval", default=None, help="with --freeze: who approved, when, DECISIONS entry")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan or a.verify or a.freeze:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
            os.environ[v] = "2"
    try:
        if a.plan:
            res = plan()
            ok, errs, _ = verify_freeze()
            src = _read_json(C.SOURCE) or {}
            res["checks"] = {"bank_sha256": sha256(C.BANK) if C.BANK.exists() else None,
                             "bank_matches_source": C.BANK.exists() and sha256(C.BANK) == src.get("file_sha256"),
                             "s0_pins_match": all(sha256(ROOT / r) == h for r, h in S0_PINS.items()),
                             "s1_s2_adapters_present": all((ROOT / ADAPTER_RELS[s] / "adapter_model.safetensors").is_file() for s in ("s1", "s2")),
                             "anchor_reference_files_present": all((ROOT / r).is_file() for r in ANCHOR_REF_RELS.values()),
                             "p3_outputs_present": p3_outputs(ROOT), "freeze_ok": ok, "freeze_blockers": errs[:3],
                             "torch_imported": "torch" in sys.modules}
            print(json.dumps(res["checks"], indent=1))
            need("torch" not in sys.modules, "--plan must not import torch", persist=False)
            if (ROOT / FREEZE_REL).exists():
                print("PREREG-3 is frozen: GEN_PLAN.json is a frozen input and is not rewritten")
                return 0
            outp = C.DOCS / "GEN_PLAN.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {outp}")
            return 0
        if a.verify:
            ok, errs, _ = verify_freeze()
            print("PREREG-3 freeze verifies" if ok else "PREREG-3 freeze does NOT verify:\n  " + "\n  ".join(errs))
            need("torch" not in sys.modules, "--verify must not import torch", persist=False)
            return 0 if ok else 1
        if a.freeze:
            freeze(a.approval or "")
            return 0
        cfg = Cfg(cap_h=a.cap_h)
        return run(cfg, RealBackend())
    except Stop as e:
        print(str(e), flush=True)
        if not (a.plan or a.verify or a.freeze):
            persist_stop(OUT, e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
