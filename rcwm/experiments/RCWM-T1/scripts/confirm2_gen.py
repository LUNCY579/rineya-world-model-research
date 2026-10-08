"""PREREG-2 GPU generation runner, freeze tool and freeze verifier (PREREG_2_DRAFT.md §7, §9, §10).
The GPU job must NOT run before PREREG-2 is frozen; it refuses unless the freeze verifies.

  python scripts/confirm2_gen.py --plan        # CPU only: no torch, no model, no lease. Projects generations, GPU time, turns.
  python scripts/confirm2_gen.py --verify      # CPU only: checks PREREG_2_FREEZE.json (+ PREREG_2_AMENDMENT_*) against the files.
  python scripts/confirm2_gen.py --freeze --d23-offtask-seen {yes,no}
                                               # CPU only. Applies the §9 freeze condition MECHANICALLY: runs the frozen
                                               # scripts/t2_analysis.py on runs/t2 (tags t2_s0,t2_s1,t2_s2), reads
                                               # E['L|chain_one'].drop / drop_ci95 for seeds 1 and 2, and freezes only if
                                               # both have drop >= 0.15 and ci95[0] > 0. Otherwise writes
                                               # docs/prereg2/FREEZE_GATE_FAILED.json and freezes nothing.
  python scripts/confirm2_gen.py               # the registered GPU job

The freeze tool lives here (not in a separate freeze script) because the PREREG-2 drafting workflow may create only the
files listed in DECISIONS D-24; the GPU runner and confirm2_analysis.py share verify_freeze().

GPU job. One shared lease (rcwmt.runtime.gpu_lease, label 'rcwm-t1:prereg2'; alternation, thermal flag, duty cycle, RAM
floor and the 4.5 h turn cap measured from lock acquisition, with D-22's stage rules: no stage starts within 0.5 h of the
cap, and a stage starts only if 1.25 x its estimate, plus the Supa reserve while S6 is outstanding, fits). One model at a time:

  load Titan-Sparse (FP32, fingerprint 818be9c6b6a3) + the fixed T2 seed-0 LoRA (PEFT, exactly as scripts/run_t2.py does;
  adapter_model.safetensors sha 92bc7c74..., hashed again at load). Every stage generates adapter OFF
  (model.disable_adapter(), = Titan-Sparse) and adapter ON in the SAME process and writes both only when both are complete:
    A0 anchor   pilot TEST ledger chain_one OFF+ON; must reproduce runs/t2/t2_s0 gen_chain_one_base / gen_chain_one
                (ledger rows) on >= 99% of chains per side, else the job stops before touching the sealed bank.
                A resumed turn that runs any sealed-bank stage re-anchors first; a failed re-anchor is persisted and final.
    S1 primary  confirm2 ledger chain_one OFF+ON
    S2 guard    off-task first-token log-probs OFF and ON (rcwmt.offtask.capture_offtask) + 64-token greedy replies OFF/ON
    S3          confirm2 register chain_one OFF+ON
    S4          confirm2 prerequisite chain_m1 OFF+ON
    S5          confirm2 ledger chain_m1 OFF+ON (transfer)
  unload (VRAM checked empty; pause flag and resource floor re-checked); load Supa 1.7 (fingerprint a9952c55f5a1):
    S6 guard    Supa off-task first-token log-probs (the KL(Supa||Sparse) reference) + Supa 64-token greedy replies
  DONE.json (create-only, with the sha256 of every output) when every stage is complete.

Outputs (create-only) under runs/bank_confirm2/. A stage is DONE only if all its files exist AND provenance.json records
it with matching sha256. Files present without that record (a crash between writing and recording), stray *.partial files,
or a half-written stage make the stage PARTIAL: it is moved to runs/bank_confirm2/_orphans/ (never deleted) and regenerated
in full, OFF and ON together (PREREG-2 deviation policy). A recorded output whose hash later changes stops the job.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- registered constants
BANK_PATH = ROOT / "data" / "bank_confirm2.json"
BANK_SHA = "fafb13e573a20eb5045ef204a7cdbf8da4393d21604245e9df26995f5be58aea"
PILOT_PATH = ROOT / "data" / "bank_pilot.json"
PILOT_SHA = "7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a"
T2_RUN = ROOT / "runs" / "t2" / "t2_s0"
ADAPTER_DIR = T2_RUN / "adapter"
ADAPTER_SHA = "92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a"
ADAPTER_CONFIG_SHA = "50d9a776e093a952ac440a77c9cd7cd8e51eb22a85eb93b383aee1eacc185d9d"
ANCHOR_REF_SHA = {"gen_chain_one_base.jsonl": "c082a5324ae63f76b6550381a5368faf9ae8669f17c7da510ee5234bb44255ae",
                  "gen_chain_one.jsonl": "5f3a69f3a67b56df4657a31f7a675a07ea15d6b33d9ab4b2dfcb3c9db066f47b"}
OFFTASK_PROMPTS = ROOT / "data" / "offtask_prompts.jsonl"
OFFTASK_SHA = "ae92870bef9ab18e99229af71d764d12b0544d40590f2faf2c7d1e9356f3d6bc"
T2_ANALYSIS_SHA = "d699dd5afa432dfef8b1c39cd14af26b92750237c7108328f7a61403decbcaf0"
RUN_T2_SHA = "ccbb2c6c19f76130d9b2e4885a04ca25d7215975e0be207d307c57bf1c4877f4"
REGISTERED_RUNTIME = {"torch": "2.6.0+cu124", "transformers": "5.14.1", "peft": "0.20.0"}
FP = {"sparse": "818be9c6b6a3", "supa": "a9952c55f5a1"}
OUT = ROOT / "runs" / "bank_confirm2"
FREEZE = ROOT / "PREREG_2_FREEZE.json"
THERMAL_FLAG = ROOT / "runs" / ".thermal_pause"
LEASE_LABEL = "rcwm-t1:prereg2"
FULL = {"L": "ledger_replay", "R": "register_machine", "P": "prerequisite_plan"}
# (stage, family code, mode) for the sealed-bank chain stages, in run order
CHAIN_STAGES = (("S1", "L", "chain_one"), ("S3", "R", "chain_one"), ("S4", "P", "chain_m1"), ("S5", "L", "chain_m1"))
STAGE_ORDER = ("A0", "S1", "S2", "S3", "S4", "S5", "S6")
SPARSE_STAGES = ("A0", "S1", "S2", "S3", "S4", "S5")
ANCHOR_MIN_IDENTICAL = 0.99
RATE_GENS_PER_S = 2.7            # registered planning rate (conservative; T2 s0 chain stages ran at 5.6-7.3 gens/s)
T2_OBSERVED = {"chain_one OFF": (2560, 349), "chain_one ON": (2560, 394), "chain_m1 OFF": (2560, 362), "chain_m1 ON": (2560, 454)}
LOAD_S = 120.0                   # per model load (FP32 0.6B from disk) incl. fingerprint
OFFTASK_CAPTURE_S = 60.0         # 248 prompts, batch 8, one forward each, with duty-cycle pauses (pilot: 5 s)
CAP_H = 4.5                      # turn cap shared with the peer, from lock acquisition (DECISIONS I-24, D-22)
STAGE_MARGIN_H = 0.5             # D-22: no stage starts within 0.5 h of the cap
STAGE_SAFETY = 1.25              # D-22-style: a stage starts only if 1.25 x its estimate (+ the Supa reserve) fits
VRAM_EMPTY_MIB = 512.0           # after unloading Sparse, at most this much may remain allocated before Supa loads
N_OFFTASK = 248

# ---------------------------------------------------------------- freeze record (PREREG-2 §7.1, §9, §10)
DRAFT_REL = "PREREG_2_DRAFT.md"
PREREG_REL = "PREREG_2.md"
FREEZE_REL = "PREREG_2_FREEZE.json"
GATE_TAGS = ("t2_s0", "t2_s1", "t2_s2")
GATE_SEEDS = ("t2_s1", "t2_s2")
GATE_CELL = "L|chain_one"
GATE_MIN_DROP = 0.15
GATE_DIR_REL = "docs/prereg2/freeze_gate"
GATE_JSON_REL = GATE_DIR_REL + "/T2_GATE.json"
SEED_OUTPUT_FILES = ("eval.json", "gen_chain_one.jsonl", "gen_chain_one_base.jsonl", "gen_chain_m1.jsonl", "gen_chain_m1_base.jsonl")
# files whose content a pre-seed snapshot locks; after the latest snapshot only logged typo/crash edits may change them
SNAPSHOT_TRACKED = ("PREREG_2_DRAFT.md", "scripts/confirm2_gen.py", "scripts/confirm2_analysis.py", "scripts/build_confirm2_bank.py")
POSTSEED_EDIT_TYPES = ("typo", "crash")
REQUIRED_INPUTS = (
    "scripts/confirm2_gen.py", "scripts/confirm2_analysis.py", "scripts/build_confirm2_bank.py",
    "scripts/t2_analysis.py", "scripts/run_t2.py", "scripts/make_t2_data.py",
    "data/bank_confirm2.json", "data/bank_pilot.json", "data/offtask_prompts.jsonl", "data/t2_train.jsonl",
    "data/t2_eval.jsonl", "data/t2_eval_xframe.jsonl", "data/t2_eval_supp.jsonl", "data/t2_eval_substrata.jsonl",
    "runs/t2/t2_s0/adapter/adapter_model.safetensors", "runs/t2/t2_s0/adapter/adapter_config.json",
    "runs/t2/t2_s0/gen_chain_one_base.jsonl", "runs/t2/t2_s0/gen_chain_one.jsonl",
    "docs/prereg2/bank_build_log.json", "docs/prereg2/GEN_PLAN.json", "docs/prereg2/PREREG_2_DRYRUN.json",
    "docs/prereg2/PREREG_2_DRYRUN.md", "docs/prereg2/PREREG_2_DRYRUN_PER_PARENT.npz",
    GATE_JSON_REL, GATE_DIR_REL + "/T2_GATE.md")
# code whose provenance hashes the analysis compares with the freeze record
PROV_CODE = ("scripts/confirm2_gen.py", "scripts/run_t2.py", "rcwmt/generation.py", "rcwmt/offtask.py", "rcwmt/runtime.py",
             "rcwmt/env.py", "rcwmt/capture.py")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(cond: bool, msg: str) -> None:
    """Integrity check that survives `python -O` (asserts do not)."""
    if not cond:
        raise SystemExit(f"[prereg2] integrity check failed: {msg}")


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _num(rx: str, name: str) -> int:
    m = re.search(rx, name)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------- stage bookkeeping (no torch)


def _partial(p: Path) -> Path:
    return p.with_name(p.name + ".partial")


def stage_required(stage: str, out: Path | None = None) -> list[Path]:
    """Files whose joint presence marks a stage complete (with the provenance record, see stage_states)."""
    o = Path(out or OUT)
    if stage == "A0":
        return [o / "anchor" / "chain_one_L_off.jsonl", o / "anchor" / "chain_one_L_on.jsonl", o / "anchor" / "anchor.json"]
    for st, f, m in CHAIN_STAGES:
        if st == stage:
            return [o / "chains" / f"{m}_{f}_{s}.jsonl" for s in ("off", "on")]
    if stage == "S2":
        return [o / "offtask" / d / n for d in ("off", "on") for n in ("offtask_meta.json", "offtask_logprobs.npy")] + \
               [o / "offtask" / "offtask_gen.jsonl"]
    if stage == "S6":
        return [o / "offtask" / "supa" / "offtask_meta.json", o / "offtask" / "supa" / "offtask_logprobs.npy",
                o / "offtask" / "supa_gen.jsonl"]
    raise KeyError(stage)


def stage_owned(stage: str, out: Path | None = None) -> list[Path]:
    """Every path a stage may write (moved aside together if the stage is partial)."""
    o = Path(out or OUT)
    if stage == "A0":
        return [o / "anchor"]
    if stage == "S2":
        return [o / "offtask" / "off", o / "offtask" / "on", o / "offtask" / "offtask_gen.jsonl",
                _partial(o / "offtask" / "offtask_gen.jsonl")]
    if stage == "S6":
        return [o / "offtask" / "supa", o / "offtask" / "supa_gen.jsonl", _partial(o / "offtask" / "supa_gen.jsonl")]
    fs = stage_required(stage, o)
    return fs + [_partial(p) for p in fs]


def rel_out(p: Path, out: Path | None = None) -> str:
    return str(Path(p).relative_to(Path(out or OUT))).replace("\\", "/")


def stage_states(prov: dict, out: Path | None = None) -> dict:
    """{stage: 'done' | 'partial' | 'todo'}. Done = all required files present AND recorded in provenance with equal sha256.
    A recorded file whose sha256 changed is not a crash artefact: the job stops."""
    o = Path(out or OUT)
    res = {}
    for st in STAGE_ORDER:
        req = stage_required(st, o)
        rec = (prov.get("stages") or {}).get(st)
        present = [p.exists() for p in req]
        if all(present) and rec:
            files = rec.get("files") or {}
            for p in req:
                r = rel_out(p, o)
                need(r in files, f"stage {st}: {r} not recorded in provenance")
                need(sha256(p) == files[r], f"stage {st}: {r} changed after it was recorded (sha256)")
            res[st] = "done"
        elif any(p.exists() for p in stage_owned(st, o)):
            res[st] = "partial"
        else:
            res[st] = "todo"
    return res


def orphan(stage: str, out: Path | None = None) -> Path:
    """Move every path of a partly written stage aside (never delete), so the stage regenerates OFF and ON together."""
    o = Path(out or OUT)
    dst = o / "_orphans" / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    dst.mkdir(parents=True, exist_ok=False)
    for p in stage_owned(stage, o):
        if p.exists():
            shutil.move(str(p), str(dst / p.name))
    print(f"[prereg2] stage {stage} was partly written; moved to {dst}", flush=True)
    return dst


def load_bank_chains(path: Path, sha: str, fam_code: str, split: str | None = None):
    need(sha256(path) == sha, f"{Path(path).name}: sha256 mismatch")
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    fam = FULL[fam_code]
    return [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"]}
            for p in b["parents"] if p["family"] == fam and (split is None or p["split"] == split) for c in p["candidates"]]


# ---------------------------------------------------------------- time estimates and the D-22 stage rule (shared by plan and run)


def stage_est_s(st: str, rate: float, n_chains: dict) -> float:
    if st == "A0":
        return n_chains["A0"] * 8 / rate
    if st == "S2":
        return 2 * OFFTASK_CAPTURE_S + 2 * N_OFFTASK / rate
    if st == "S6":
        return LOAD_S + OFFTASK_CAPTURE_S + N_OFFTASK / rate
    return n_chains[st] * 8 / rate


def supa_reserve_s(rate: float, n_chains: dict) -> float:
    return STAGE_SAFETY * stage_est_s("S6", rate, n_chains)


def can_start(elapsed_s: float, est_s: float, reserve_s: float, cap_s: float) -> bool:
    """D-22: never start within STAGE_MARGIN_H of the cap; start only if 1.25 x estimate + reserve fits."""
    left = cap_s - elapsed_s
    return left >= STAGE_MARGIN_H * 3600 and STAGE_SAFETY * est_s + reserve_s <= left


def chain_counts() -> dict:
    n = {"A0": len(load_bank_chains(PILOT_PATH, PILOT_SHA, "L", "test"))}
    for st, f, _ in CHAIN_STAGES:
        n[st] = len(load_bank_chains(BANK_PATH, BANK_SHA, f))
    return n


def simulate_turns(rate: float, n_chains: dict, cap_h: float = CAP_H) -> list[dict]:
    """Turn schedule under the runner's rules (stage order fixed; a resumed turn with sealed-bank stages re-anchors)."""
    cap = cap_h * 3600
    left = list(STAGE_ORDER)
    turns = []
    while left:
        t, ran = 0.0, []
        sparse_left = [s for s in left if s != "S6"]
        if sparse_left:
            t += LOAD_S
            if "A0" not in left and any(s != "S2" for s in sparse_left):
                t += stage_est_s("A0", rate, n_chains)                 # re-anchor
                ran.append("A0 (re-anchor)")
            for st in sparse_left:
                res = 0.0 if "S6" not in left else supa_reserve_s(rate, n_chains)
                if not can_start(t, stage_est_s(st, rate, n_chains), res, cap):
                    break
                t += stage_est_s(st, rate, n_chains)
                ran.append(st)
                left.remove(st)
        if "S6" in left and can_start(t, stage_est_s("S6", rate, n_chains), 0.0, cap):
            t += stage_est_s("S6", rate, n_chains)
            ran.append("S6")
            left.remove("S6")
        if not [r for r in ran if not r.startswith("A0 (")]:
            raise RuntimeError(f"no stage fits an empty turn at {rate} gens/s")
        turns.append({"stages": ran, "projected_h": round(t / 3600, 2)})
    return turns


def plan(verbose: bool = True) -> dict:
    """Generation counts per stage (upper bounds: a chain stops early only if an answer is unparseable)."""
    n = chain_counts()
    rows = []
    for st in STAGE_ORDER:
        if st == "A0":
            rows.append((st, "anchor: pilot TEST ledger chain_one OFF+ON (reproduce T2 s0)", n["A0"] * 8, 0))
        elif st == "S2":
            rows.append((st, "off-task first-token log-probs OFF+ON; 64-token greedy OFF+ON", 2 * N_OFFTASK, 2 * N_OFFTASK))
        elif st == "S6":
            rows.append((st, "Supa off-task first-token log-probs + 64-token greedy (after unloading Sparse)", N_OFFTASK, N_OFFTASK))
        else:
            f, m = next((f, m) for s, f, m in CHAIN_STAGES if s == st)
            rows.append((st, f"confirm2 {FULL[f]} {m} OFF+ON", n[st] * 8, 0))
    total_g = sum(r[2] for r in rows)
    obs_rate = sum(g for g, _ in T2_OBSERVED.values()) / sum(s for _, s in T2_OBSERVED.values())
    sec = {r[0]: stage_est_s(r[0], RATE_GENS_PER_S, n) for r in rows}
    total_s = sum(sec.values()) + LOAD_S                      # Sparse load (Supa's load is inside S6's estimate)
    turns = simulate_turns(RATE_GENS_PER_S, n)
    turns_obs = simulate_turns(obs_rate, n)
    res = {"rate_gens_per_s": RATE_GENS_PER_S,
           "stages": [{"stage": r[0], "what": r[1], "generations_max": r[2], "forward_prompts": r[3],
                       "chains": n.get(r[0], 0), "projected_s": round(sec[r[0]], 1)} for r in rows],
           "model_loads": 2, "load_s_each": LOAD_S, "total_generations_max": total_g,
           "projected_total_h_single_pass": round(total_s / 3600, 2), "cap_h": CAP_H,
           "stage_rule": f"no stage starts within {STAGE_MARGIN_H} h of the cap; a stage starts only if {STAGE_SAFETY} x its "
                         f"estimate + the Supa reserve ({STAGE_SAFETY} x S6) fits; a resumed turn re-anchors",
           "turns_at_planning_rate": turns,
           "projected_gpu_h_at_planning_rate_incl_reanchor_and_reloads": round(sum(t["projected_h"] for t in turns), 2),
           "t2_observed_chain_rate_gens_per_s": round(obs_rate, 2), "turns_at_t2_observed_rate": turns_obs,
           "projected_gpu_h_at_t2_observed_rate": round(sum(t["projected_h"] for t in turns_obs), 2)}
    if verbose:
        print(f"PREREG-2 GPU plan (no model loaded; rate {RATE_GENS_PER_S} gens/s, {LOAD_S:.0f} s per model load)")
        cum = LOAD_S
        for r in res["stages"]:
            cum += r["projected_s"]
            print(f"  {r['stage']}  {r['what']:<78} gens<= {r['generations_max']:>6}  fwd {r['forward_prompts']:>4}  "
                  f"{r['projected_s'] / 60:6.1f} min  (cum {cum / 3600:4.2f} h)")
        print(f"  total generations <= {total_g:,}; single pass {res['projected_total_h_single_pass']} h at {RATE_GENS_PER_S} gens/s")
        print(f"  turns at {RATE_GENS_PER_S} gens/s (cap {CAP_H} h, D-22 stage rule): " +
              "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(turns)))
        print(f"  turns at the T2 s0 observed chain rate ({res['t2_observed_chain_rate_gens_per_s']} gens/s): " +
              "; ".join(f"turn {i + 1}: {t['stages']} {t['projected_h']} h" for i, t in enumerate(turns_obs)))
    return res


# ---------------------------------------------------------------- freeze verification (shared with confirm2_analysis.py)


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _close(a, b, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def gate_from_results(G: dict) -> dict:
    """The §9 rule applied to a t2_analysis results JSON: per gated seed, E['L|chain_one'] drop >= 0.15 and ci95[0] > 0."""
    out, ok = {}, True
    for s in GATE_SEEDS:
        try:
            E = G["results"][s]["E"][GATE_CELL]
            drop, ci = float(E["drop"]), [float(x) for x in E["drop_ci95"]]
        except (KeyError, TypeError, ValueError):
            out[s] = {"error": f"no E['{GATE_CELL}'] drop/drop_ci95 for {s}"}
            ok = False
            continue
        passed = drop >= GATE_MIN_DROP - 1e-12 and ci[0] > 0
        out[s] = {"drop": drop, "ci95": ci, "off": E.get("off"), "on": E.get("on"), "n_paired": E.get("n_paired"),
                  "passed": bool(passed)}
        ok = ok and passed
    out["passed"] = bool(ok)
    return out


def latest_snapshot(root: Path):
    snaps = sorted((root / "docs" / "prereg2").glob("PRE_SEED_SNAPSHOT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name))
    return snaps[-1] if snaps else None


def seed_output_times(root: Path, runs_root: Path) -> dict:
    """Earliest mtime of any seed output (stat only) per gated seed."""
    out = {}
    for s in GATE_SEEDS:
        ts = [(runs_root / s / f).stat().st_mtime for f in SEED_OUTPUT_FILES if (runs_root / s / f).exists()]
        out[s] = {"earliest_output_mtime": iso(min(ts)) if ts else None, "eval_json_mtime":
                  iso((runs_root / s / "eval.json").stat().st_mtime) if (runs_root / s / "eval.json").exists() else None}
    return out


def verify_freeze(root: Path | None = None, runs_root: Path | None = None) -> tuple[bool, list[str], dict | None]:
    """Everything the GPU job and the registered analysis require before they run. Returns (ok, errors, freeze record)."""
    root = Path(root or ROOT)
    runs_root = Path(runs_root or root / "runs" / "t2")
    fz = _read_json(root / FREEZE_REL)
    if fz is None:
        return False, ["PREREG_2_FREEZE.json does not exist or is unreadable (PREREG-2 is a draft)"], None
    errs = []
    pr = root / PREREG_REL
    if not pr.is_file():
        errs.append("PREREG_2.md does not exist")
    elif sha256(pr) != fz.get("prereg_sha256"):
        errs.append("PREREG_2.md differs from the freeze")
    inputs = dict(fz.get("inputs") or {})
    snap_rel = (fz.get("pre_seed_snapshot") or {}).get("file")
    required = set(REQUIRED_INPUTS) | {f"rcwmt/{p.name}" for p in (root / "rcwmt").glob("*.py")}
    if snap_rel:
        required.add(snap_rel)
    else:
        errs.append("no pre-seed snapshot recorded")
    missing = sorted(required - set(inputs))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing}")
    expected = dict(inputs)                          # amendment chain, as scripts/verify_freeze.py does for PREREG-1
    for a in sorted(root.glob("PREREG_2_AMENDMENT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name)):
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
    fc = fz.get("freeze_condition") or {}
    G = _read_json(root / GATE_JSON_REL)
    if fc.get("gate_json") != GATE_JSON_REL or G is None:
        errs.append("freeze condition: the t2_analysis gate output is not recorded or unreadable")
    else:
        meta = G.get("meta") or {}
        if meta.get("script_sha256") != T2_ANALYSIS_SHA:
            errs.append("freeze condition: the gate output was not produced by the frozen t2_analysis.py")
        if list(meta.get("tags") or []) != list(GATE_TAGS):
            errs.append(f"freeze condition: gate tags {meta.get('tags')} != {list(GATE_TAGS)}")
        g = gate_from_results(G)
        for s in GATE_SEEDS:
            rec = fc.get(s) or {}
            if "error" in g[s] or not (_close(rec.get("drop"), g[s]["drop"]) and len(rec.get("ci95") or []) == 2
                                       and all(_close(a, b) for a, b in zip(rec["ci95"], g[s]["ci95"]))):
                errs.append(f"freeze condition: recorded numbers for {s} differ from the gate output")
        if not g["passed"]:
            errs.append("freeze condition NOT met (recomputed from the gate output): " +
                        "; ".join(f"{s} drop {g[s].get('drop')} ci95 {g[s].get('ci95')}" for s in GATE_SEEDS))
    if snap_rel:
        snap = _read_json(root / snap_rel) or {}
        ex = snap.get("exists") or {}
        for s in GATE_SEEDS:
            for f in SEED_OUTPUT_FILES:
                if ex.get(f"runs/t2/{s}/{f}", True):
                    errs.append(f"pre-seed snapshot does not show runs/t2/{s}/{f} absent")
        now = seed_output_times(root, runs_root)
        try:
            t_snap = datetime.datetime.fromisoformat(snap["written_local"])
            for s in GATE_SEEDS:
                rec = (fz.get("seed_outputs") or {}).get(s) or {}
                if rec.get("eval_json_mtime") != now[s]["eval_json_mtime"]:
                    errs.append(f"runs/t2/{s}/eval.json changed after the freeze")
                if not rec.get("earliest_output_mtime") or datetime.datetime.fromisoformat(rec["earliest_output_mtime"]) <= t_snap:
                    errs.append(f"{s}: a seed output predates the pre-seed snapshot")
        except (KeyError, TypeError, ValueError) as e:
            errs.append(f"pre-seed ordering not verifiable: {e}")
    if fz.get("registered_runtime") != REGISTERED_RUNTIME:
        errs.append("registered runtime differs")
    return not errs, errs, fz


def allowed_hashes(fz: dict, root: Path | None = None) -> dict:
    """{file: {sha256 at freeze, + every logged amendment's new sha256}} (for checking per-turn provenance hashes)."""
    root = Path(root or ROOT)
    allowed = {k: {v} for k, v in (fz.get("inputs") or {}).items()}
    for a in sorted(root.glob("PREREG_2_AMENDMENT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name)):
        am = _read_json(a) or {}
        if am.get("file") in allowed and am.get("new_sha256"):
            allowed[am["file"]].add(am["new_sha256"])
    return allowed


# ---------------------------------------------------------------- freeze (CPU only; reads the seed results ONCE, by the frozen code)


def compute_gate(root: Path, runs_root: Path) -> dict:
    """Run the frozen t2_analysis.py in-process on GATE_TAGS; outputs go to docs/prereg2/freeze_gate/ (not docs/t2_analysis)."""
    import t2_analysis as T2                  # torch-free (stubs rcwmt.heads)
    need(sha256(Path(T2.__file__)) == T2_ANALYSIS_SHA, "scripts/t2_analysis.py is not the frozen D-19 analysis")
    out_md = root / GATE_DIR_REL / "T2_GATE.md"
    res = T2.run(list(GATE_TAGS), runs_root, out_md)
    G = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    need(G["meta"]["script_sha256"] == T2_ANALYSIS_SHA, "gate output script sha")
    need("torch" not in sys.modules, "the freeze imported torch")
    return {"G": G, "gate": gate_from_results(G)}


def _fill_prereg(text: str, stamp: str, inputs: dict, fc: dict) -> str:
    lines = text.splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("**Status:**")]
    need(len(st) == 1, "the draft has no unique **Status:** line")
    lines[st[0]] = (f"**Status:** FROZEN at {stamp} (DECISIONS D-24; freeze condition of §9 met: "
                    + "; ".join(f"{s} drop {fc[s]['drop']:+.3f} [{fc[s]['ci95'][0]:+.3f}, {fc[s]['ci95'][1]:+.3f}]" for s in GATE_SEEDS)
                    + "). The SHA-256 of this file is in `PREREG_2_FREEZE.json`.")
    i10 = [i for i, l in enumerate(lines) if l.startswith("## 10.")]
    need(len(i10) == 1, "the draft has no unique §10")
    nxt = next((i for i in range(i10[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    sec = ["## 10. Freeze record", "",
           f"Frozen at {stamp} by `scripts/confirm2_gen.py --freeze`. SHA-256 of every frozen input:", ""]
    sec += [f"- `{k}` `{v}`" for k, v in sorted(inputs.items())]
    sec += ["", "Freeze condition (computed by the frozen `scripts/t2_analysis.py` into `" + GATE_JSON_REL + "`):", ""]
    sec += [f"- {s}: ledger chain_one k=4 regret drop {fc[s]['drop']:+.4f}, 95% CI [{fc[s]['ci95'][0]:+.4f}, {fc[s]['ci95'][1]:+.4f}]"
            for s in GATE_SEEDS]
    sec.append("")
    return "\n".join(lines[: i10[0]] + sec + lines[nxt:]) + "\n"


def freeze(d23_seen: str, root: Path | None = None, runs_root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    runs_root = Path(runs_root or root / "runs" / "t2")
    need(d23_seen in ("yes", "no"), "--d23-offtask-seen must be yes or no")
    if (root / FREEZE_REL).exists() or (root / PREREG_REL).exists():
        raise SystemExit("PREREG-2 is already frozen (PREREG_2_FREEZE.json or PREREG_2.md exists)")
    if (root / "runs" / "bank_confirm2").exists():
        raise SystemExit("runs/bank_confirm2 exists before the freeze: investigate before freezing")
    if (root / "docs" / "prereg2" / "FREEZE_GATE_FAILED.json").exists():
        raise SystemExit("the freeze condition already failed (docs/prereg2/FREEZE_GATE_FAILED.json); PREREG-2 is not run as drafted")
    draft = root / DRAFT_REL
    need(draft.is_file(), "PREREG_2_DRAFT.md missing")
    # 1. the latest pre-seed snapshot must predate every seed output, and nothing it locks may have changed since,
    #    except logged typo/crash edits (docs/prereg2/POSTSEED_EDIT_*.json: file, old_sha256, new_sha256, type, diff, statement)
    snap_p = latest_snapshot(root)
    need(snap_p is not None, "no docs/prereg2/PRE_SEED_SNAPSHOT_*.json")
    snap = json.loads(snap_p.read_text(encoding="utf-8"))
    for s in GATE_SEEDS:
        for f in SEED_OUTPUT_FILES:
            need(not (snap.get("exists") or {}).get(f"runs/t2/{s}/{f}", True),
                 f"the latest snapshot ({snap_p.name}) does not show runs/t2/{s}/{f} absent")
    chain = {rel: (snap.get("sha256") or {}).get(rel) for rel in SNAPSHOT_TRACKED}
    edits = sorted((root / "docs" / "prereg2").glob("POSTSEED_EDIT_*.json"), key=lambda p: _num(r"_(\d+)\.json$", p.name))
    for e_p in edits:
        e = json.loads(e_p.read_text(encoding="utf-8"))
        need(e.get("file") in chain and chain[e["file"]] == e.get("old_sha256"), f"{e_p.name}: old hash does not match the chain")
        need(e.get("type") in POSTSEED_EDIT_TYPES and e.get("diff") and e.get("statement"),
             f"{e_p.name}: only logged typo/crash edits with a diff and a statement are allowed after the seeds")
        chain[e["file"]] = e.get("new_sha256")
    changed = [rel for rel in SNAPSHOT_TRACKED if chain.get(rel) != sha256(root / rel)]
    need(not changed, f"changed after the last pre-seed snapshot without a logged typo/crash edit: {changed}")
    times = seed_output_times(root, runs_root)
    t_snap = datetime.datetime.fromisoformat(snap["written_local"])
    for s in GATE_SEEDS:
        need((runs_root / s / "eval.json").is_file(), f"runs/t2/{s}/eval.json missing: seed {s} has not finished")
        need(datetime.datetime.fromisoformat(times[s]["earliest_output_mtime"]) > t_snap,
             f"{s}: a seed output predates the latest pre-seed snapshot")
    # 2. the gate, mechanically
    g = compute_gate(root, runs_root)
    gate = g["gate"]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    if not gate["passed"]:
        with (root / "docs" / "prereg2" / "FREEZE_GATE_FAILED.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"checked_at": stamp, "rule": f"each of {list(GATE_SEEDS)}: E['{GATE_CELL}'].drop >= {GATE_MIN_DROP} "
                                 "and drop_ci95[0] > 0", "gate": gate, "gate_json": GATE_JSON_REL,
                                 "gate_json_sha256": sha256(root / GATE_JSON_REL)}, indent=1) + "\n")
        raise SystemExit("[prereg2] FREEZE CONDITION NOT MET: " + "; ".join(f"{s} {gate[s]}" for s in GATE_SEEDS)
                         + ". PREREG-2 is not frozen and not run as drafted; report the seed instability (§9).")
    # 3. hash everything, write PREREG_2.md (the draft is kept as the pre-freeze record) and PREREG_2_FREEZE.json
    rels = set(REQUIRED_INPUTS) | {f"rcwmt/{p.name}" for p in (root / "rcwmt").glob("*.py")} | \
        {str(snap_p.relative_to(root)).replace("\\", "/")} | {str(e.relative_to(root)).replace("\\", "/") for e in edits}
    inputs = {}
    for rel in sorted(rels):
        need((root / rel).is_file(), f"frozen input missing: {rel}")
        inputs[rel] = sha256(root / rel)
    fc = {s: {k: gate[s][k] for k in ("drop", "ci95", "off", "on", "n_paired")} for s in GATE_SEEDS}
    text = _fill_prereg(draft.read_text(encoding="utf-8"), stamp, inputs, fc)
    with (root / PREREG_REL).open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    loc = root / "runs" / "bank_pilot" / "localized"
    rec = {"frozen_at": stamp, "prereg_sha256": sha256(root / PREREG_REL), "draft_sha256": sha256(draft), "inputs": inputs,
           "freeze_condition": {"rule": f"each of {list(GATE_SEEDS)}: t2_analysis E['{GATE_CELL}'].drop >= {GATE_MIN_DROP} "
                                        "and drop_ci95[0] > 0 (pilot TEST, D-19 instrument)",
                                "tags": list(GATE_TAGS), "gate_json": GATE_JSON_REL, "t2_analysis_sha256": T2_ANALYSIS_SHA,
                                **fc, "passed": True},
           "pre_seed_snapshot": {"file": str(snap_p.relative_to(root)).replace("\\", "/"), "written_local": snap["written_local"],
                                 "file_mtime": iso(snap_p.stat().st_mtime)},
           "postseed_edits": [str(e.relative_to(root)).replace("\\", "/") for e in edits],
           "seed_outputs": times,
           "d23_offtask_outputs_seen_before_freeze": d23_seen,
           "d23_localized_listing": sorted(str(p.relative_to(loc)).replace("\\", "/") for p in loc.rglob("*")) if loc.exists() else None,
           "registered_runtime": REGISTERED_RUNTIME, "bank_sha256": BANK_SHA, "adapter_sha256": ADAPTER_SHA}
    with (root / FREEZE_REL).open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    ok, errs, _ = verify_freeze(root, runs_root)
    need(ok, f"the freeze record was written but does not verify: {errs}")
    print(f"[prereg2] FROZEN {stamp}; PREREG_2.md sha {rec['prereg_sha256']}; " +
          "; ".join(f"{s} drop {fc[s]['drop']:+.3f} [{fc[s]['ci95'][0]:+.3f}, {fc[s]['ci95'][1]:+.3f}]" for s in GATE_SEEDS))
    return rec


# ---------------------------------------------------------------- GPU path (imports torch; never used by --plan/--verify/--freeze)


def _wait_pause_and_resources(runtime) -> None:
    if THERMAL_FLAG.exists():
        print("[prereg2] thermal pause flag set before a model load; waiting", flush=True)
        while THERMAL_FLAG.exists():
            time.sleep(5)
    runtime.wait_for_resources()


def _lease_acquired_unix(runtime) -> tuple[float, str]:
    try:
        rec = json.loads(Path(runtime.LOCK_PATH).read_text(encoding="utf-8"))
        if rec.get("label") == LEASE_LABEL and rec.get("pid") == os.getpid():
            return float(rec["created"]), "lock record"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return time.time(), "lease entry (lock record unavailable)"


def run_gpu(cap_h: float) -> None:
    ok, errs, fz = verify_freeze()
    if not ok:
        raise SystemExit("refusing to run the PREREG-2 GPU job: " + "; ".join(errs))
    need(sha256(BANK_PATH) == BANK_SHA, "bank hash")
    need(sha256(ADAPTER_DIR / "adapter_model.safetensors") == ADAPTER_SHA, "adapter hash")
    need(sha256(ADAPTER_DIR / "adapter_config.json") == ADAPTER_CONFIG_SHA, "adapter config hash")
    need(sha256(OFFTASK_PROMPTS) == OFFTASK_SHA, "off-task prompts hash")
    for n, h in ANCHOR_REF_SHA.items():
        need(sha256(T2_RUN / n) == h, f"anchor reference {n} hash")
    if (OUT / "DONE.json").exists():
        raise SystemExit(f"{OUT / 'DONE.json'} exists: the job is complete")
    anchor_json = OUT / "anchor" / "anchor.json"
    if anchor_json.exists() and not (_read_json(anchor_json) or {}).get("pass"):
        raise SystemExit("the anchor check failed in an earlier turn; PREREG-2 deviation policy applies (investigate, log)")
    if list((OUT / "anchor").glob("RECHECK_FAILED_*.json")):
        raise SystemExit("an anchor re-check failed in an earlier turn; PREREG-2 deviation policy applies (investigate, log)")
    import torch
    from rcwmt import env, generation, offtask, runtime
    from run_t2 import gen_prompts           # the T2 instrument for the 64-token greedy replies

    OUT.mkdir(parents=True, exist_ok=True)
    prov_path = OUT / "provenance.json"
    if prov_path.exists():
        prov = _read_json(prov_path)
        need(prov is not None, "provenance.json unreadable")
    else:
        prov = {"bank_sha256": BANK_SHA, "adapter_sha256_registered": ADAPTER_SHA, "adapter_config_sha256": ADAPTER_CONFIG_SHA,
                "offtask_prompts_sha256": OFFTASK_SHA, "freeze_prereg_sha256": fz["prereg_sha256"], "stages": {}, "turns": []}
    prov["code_sha256"] = {r: sha256(ROOT / r) for r in PROV_CODE}

    def save():
        tmp = _partial(prov_path)
        tmp.write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, prov_path)

    states = stage_states(prov)
    for st, s in states.items():
        if s == "partial":
            orphan(st)
            prov["stages"].pop(st, None)
            states[st] = "todo"
    done = {st for st, s in states.items() if s == "done"}

    def record(st: str, extra: dict) -> None:
        prov["stages"][st] = {"name": st, "turn": len(prov["turns"]), **extra,
                              "files": {rel_out(p): sha256(p) for p in stage_required(st)}, "finished_unix": time.time()}
        done.add(st)
        save()

    def finalize() -> None:
        need(set(prov["stages"]) == set(STAGE_ORDER) and stage_states(prov) == {s: "done" for s in STAGE_ORDER},
             "finalize: not every stage is recorded and unchanged")
        with (OUT / "DONE.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps({"finished_unix": time.time(), "stages": list(STAGE_ORDER),
                                "files": {st: prov["stages"][st]["files"] for st in STAGE_ORDER},
                                "provenance_sha256": sha256(prov_path)}, indent=1) + "\n")
        print("[prereg2] all stages complete; DONE.json written", flush=True)

    def write_jsonl(path: Path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit(f"{path} exists (create-only)")
        tmp = _partial(path)
        with tmp.open("x", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def chain_rows(chains, results):
        return [{"cand_id": c["cand_id"], "states": [None if s is None else env.dumps(s) for s in r["states"]], "texts": r["texts"]}
                for c, r in zip(chains, results)]

    def side(fn):
        t = {"pid": os.getpid(), "start_unix": time.time()}
        out = fn()
        t["end_unix"] = time.time()
        return out, t

    def off_on_chains(model, tok, chains, mode):
        """OFF then ON in this process; nothing is written until both are complete."""
        def off():
            with model.disable_adapter():
                return generation.chain(model, tok, chains, mode)
        r_off, t_off = side(off)
        r_on, t_on = side(lambda: generation.chain(model, tok, chains, mode))
        return r_off, r_on, {"off": t_off, "on": t_on}

    def anchor(model, tok):
        """Pilot TEST ledger chain_one, OFF and ON, against the T2 s0 files (identical batch composition)."""
        chains = load_bank_chains(PILOT_PATH, PILOT_SHA, "L", "test")
        r_off, r_on, sides = off_on_chains(model, tok, chains, "chain_one")
        res = {"pid": os.getpid(), "unix": time.time(), "sides": sides}
        for s, rr, ref_name in (("off", r_off, "gen_chain_one_base.jsonl"), ("on", r_on, "gen_chain_one.jsonl")):
            ref = {}
            for line in (T2_RUN / ref_name).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    ref[row["cand_id"]] = row
            st = sum([None if x is None else env.dumps(x) for x in r["states"]] == ref[c["cand_id"]]["states"] for c, r in zip(chains, rr))
            tx = sum(r["texts"] == ref[c["cand_id"]]["texts"] for c, r in zip(chains, rr))
            res[s] = {"n": len(chains), "identical_states": st / len(chains), "identical_texts": tx / len(chains)}
        res["pass"] = all(res[s]["identical_states"] >= ANCHOR_MIN_IDENTICAL for s in ("off", "on"))
        return chains, r_off, r_on, res

    todo = [st for st in STAGE_ORDER if st not in done]
    print(f"[prereg2] stages to run: {todo}", flush=True)
    if not todo:
        finalize()
        return
    n_chains = chain_counts()
    with runtime.gpu_lease(label=LEASE_LABEL, job="RCWM-T1 PREREG-2 confirm2 generation (Sparse+T2 adapter OFF/ON, Supa off-task)"):
        t_acq, t_src = _lease_acquired_unix(runtime)
        cap_s = cap_h * 3600
        turn = {"pid": os.getpid(), "acquired_unix": t_acq, "acquired_source": t_src, "entered_unix": time.time(),
                "code_sha256": dict(prov["code_sha256"]), "stages_run": [], "deferred": []}
        prov["turns"].append(turn)
        save()
        rate = [RATE_GENS_PER_S]

        def ok_to_start(st: str) -> bool:
            reserve = supa_reserve_s(rate[0], n_chains) if (st != "S6" and "S6" not in done) else 0.0
            return can_start(time.time() - t_acq, stage_est_s(st, rate[0], n_chains), reserve, cap_s)

        sparse_todo = [st for st in todo if st != "S6"]
        if sparse_todo:
            from peft import PeftModel
            import peft
            _wait_pause_and_resources(runtime)
            t1 = time.time()
            base, p_sparse = runtime.load_arm("sparse")
            need(p_sparse["param_fingerprint"].startswith(FP["sparse"]), "Titan-Sparse fingerprint")
            prov["sparse_fingerprint"] = p_sparse["param_fingerprint"]
            ad_sha = sha256(ADAPTER_DIR / "adapter_model.safetensors")
            cfg_sha = sha256(ADAPTER_DIR / "adapter_config.json")
            need(ad_sha == ADAPTER_SHA and cfg_sha == ADAPTER_CONFIG_SHA, "adapter hash at load")
            model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), adapter_name="t2", is_trainable=False).eval()
            turn["adapter_sha256_at_load"], turn["adapter_config_sha256_at_load"] = ad_sha, cfg_sha
            turn["peft_version"] = peft.__version__
            turn["runtime"] = runtime.runtime_info()
            turn["sparse_load_s"] = round(time.time() - t1, 1)
            tok = runtime.tokenizer()
            print(f"[prereg2] Titan-Sparse {p_sparse['param_fingerprint'][:12]} + T2 s0 adapter loaded in {turn['sparse_load_s']}s", flush=True)
            save()
            try:
                if "A0" in done and any(st != "S2" for st in sparse_todo):    # a resumed turn re-anchors before sealed-bank chains
                    _, _, _, res = anchor(model, tok)
                    turn["anchor_recheck"] = res
                    save()
                    if not res["pass"]:
                        with (OUT / "anchor" / f"RECHECK_FAILED_{int(time.time())}.json").open("x", encoding="utf-8") as fh:
                            fh.write(json.dumps(res, indent=1) + "\n")
                        raise SystemExit(f"[prereg2] ANCHOR RECHECK FAILED {res}; stop (deviation policy)")
                for i, st in enumerate(sparse_todo):
                    if not ok_to_start(st):
                        turn["deferred"] = sparse_todo[i:]
                        print(f"[prereg2] cap: deferring {turn['deferred']} to the next turn", flush=True)
                        save()
                        break
                    t0 = time.time()
                    if st == "A0":
                        chains, r_off, r_on, res = anchor(model, tok)
                        write_jsonl(OUT / "anchor" / "chain_one_L_off.jsonl", chain_rows(chains, r_off))
                        write_jsonl(OUT / "anchor" / "chain_one_L_on.jsonl", chain_rows(chains, r_on))
                        with anchor_json.open("x", encoding="utf-8") as fh:
                            fh.write(json.dumps(res, indent=1) + "\n")
                        prov["anchor"] = res
                        n_g = sum(len(r["texts"]) for r in r_off + r_on)
                        if not res["pass"]:
                            save()
                            raise SystemExit(f"[prereg2] ANCHOR FAILED {res}: the instrument does not reproduce T2 s0; "
                                             "stop before any sealed-bank generation (deviation policy)")
                        extra = {"sides": res["sides"]}
                    elif st == "S2":
                        for d in ("off", "on"):
                            (OUT / "offtask" / d).mkdir(parents=True, exist_ok=False)

                        def cap_off():
                            with model.disable_adapter():
                                offtask.capture_offtask(model, tok, OUT / "offtask" / "off")
                        _, t_off = side(cap_off)
                        _, t_on = side(lambda: offtask.capture_offtask(model, tok, OUT / "offtask" / "on"))
                        ot = offtask.load_prompts()
                        op = [generation.render(tok, r["messages"]) for r in ot]

                        def gen_off():
                            with model.disable_adapter():
                                return gen_prompts(model, tok, op, max_new=64)
                        ob, g_off = side(gen_off)
                        oo, g_on = side(lambda: gen_prompts(model, tok, op, max_new=64))
                        write_jsonl(OUT / "offtask" / "offtask_gen.jsonl",
                                    [{"id": r["id"], "reply_class": r["reply_class"], "off_text": b, "on_text": t} for r, b, t in zip(ot, ob, oo)])
                        n_g = 2 * len(ot)
                        extra = {"sides": {"capture_off": t_off, "capture_on": t_on, "greedy_off": g_off, "greedy_on": g_on}}
                    else:
                        f, m = next((f, m) for s, f, m in CHAIN_STAGES if s == st)
                        chains = load_bank_chains(BANK_PATH, BANK_SHA, f)
                        r_off, r_on, sides = off_on_chains(model, tok, chains, m)
                        write_jsonl(OUT / "chains" / f"{m}_{f}_off.jsonl", chain_rows(chains, r_off))
                        write_jsonl(OUT / "chains" / f"{m}_{f}_on.jsonl", chain_rows(chains, r_on))
                        n_g = sum(len(r["texts"]) for r in r_off + r_on)
                        extra = {"sides": sides}
                    sec = time.time() - t0
                    if st != "S2":
                        rate[0] = max(0.5, n_g / max(sec, 1e-9))
                    record(st, {"generations": n_g, "seconds": round(sec, 1), **extra})
                    turn["stages_run"].append(st)
                    save()
                    print(f"[prereg2] {st} done: {n_g} generations in {sec:.0f}s ({n_g / max(sec, 1e-9):.2f}/s)", flush=True)
            finally:
                del model, base
                runtime.unload(None)
        if "S6" not in done:
            if ok_to_start("S6"):
                alloc = torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
                need(alloc <= VRAM_EMPTY_MIB, f"{alloc:.0f} MiB still allocated after unloading Titan-Sparse; not loading Supa")
                _wait_pause_and_resources(runtime)
                t0 = time.time()
                supa, p_supa = runtime.load_arm("supa")
                try:
                    need(p_supa["param_fingerprint"].startswith(FP["supa"]), "Supa fingerprint")
                    prov["supa_fingerprint"] = p_supa["param_fingerprint"]
                    turn["supa_runtime"] = runtime.runtime_info()
                    save()
                    tok = runtime.tokenizer()
                    (OUT / "offtask" / "supa").mkdir(parents=True, exist_ok=False)
                    _, t_cap = side(lambda: offtask.capture_offtask(supa, tok, OUT / "offtask" / "supa"))
                    ot = offtask.load_prompts()
                    op = [generation.render(tok, r["messages"]) for r in ot]
                    sg, t_gen = side(lambda: gen_prompts(supa, tok, op, max_new=64))
                    write_jsonl(OUT / "offtask" / "supa_gen.jsonl",
                                [{"id": r["id"], "reply_class": r["reply_class"], "supa_text": t} for r, t in zip(ot, sg)])
                finally:
                    del supa
                    runtime.unload(None)
                record("S6", {"seconds": round(time.time() - t0, 1), "generations": len(ot),
                              "sides": {"capture_supa": t_cap, "greedy_supa": t_gen}})
                turn["stages_run"].append("S6")
            else:
                turn["deferred"].append("S6")
        turn["released_unix"] = time.time()
        save()
    if done >= set(STAGE_ORDER):
        finalize()
    else:
        print(f"[prereg2] turn ended; remaining stages {[s for s in STAGE_ORDER if s not in done]} resume next turn", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="CPU only: no torch import, no model; projected generations and GPU time")
    g.add_argument("--verify", action="store_true", help="CPU only: verify PREREG_2_FREEZE.json (+ amendments) against the files")
    g.add_argument("--freeze", action="store_true", help="CPU only: apply the §9 freeze condition and freeze if it is met")
    ap.add_argument("--d23-offtask-seen", choices=("yes", "no"), help="with --freeze: had anyone seen D-23's off-task KL numbers?")
    ap.add_argument("--cap-h", type=float, default=CAP_H)
    a = ap.parse_args()
    if a.plan or a.verify or a.freeze:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # belt and braces: these modes never import torch
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
            os.environ[v] = "2"
    if a.plan:
        need(sha256(BANK_PATH) == BANK_SHA, "bank hash")
        res = plan()
        ok, errs, _ = verify_freeze()
        res["checks"] = {"bank_sha256": BANK_SHA,
                         "adapter_sha256_matches": sha256(ADAPTER_DIR / "adapter_model.safetensors") == ADAPTER_SHA,
                         "adapter_config_sha256_matches": sha256(ADAPTER_DIR / "adapter_config.json") == ADAPTER_CONFIG_SHA,
                         "offtask_prompts_sha256_matches": sha256(OFFTASK_PROMPTS) == OFFTASK_SHA,
                         "anchor_reference_files_match": all(sha256(T2_RUN / n) == h for n, h in ANCHOR_REF_SHA.items()),
                         "output_dir_exists": OUT.exists(), "freeze_ok": ok, "freeze_blockers": errs[:3],
                         "torch_imported": "torch" in sys.modules}
        print(json.dumps(res["checks"], indent=1))
        need("torch" not in sys.modules, "--plan must not import torch")
        if FREEZE.exists():
            print("PREREG-2 is frozen: GEN_PLAN.json is a frozen input and is not rewritten")
            return
        out = ROOT / "docs" / "prereg2" / "GEN_PLAN.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return
    if a.verify:
        ok, errs, _ = verify_freeze()
        print("PREREG-2 freeze verifies" if ok else "PREREG-2 freeze does NOT verify:\n  " + "\n  ".join(errs))
        need("torch" not in sys.modules, "--verify must not import torch")
        sys.exit(0 if ok else 1)
    if a.freeze:
        if a.d23_offtask_seen is None:
            raise SystemExit("--freeze needs --d23-offtask-seen yes|no (§5b)")
        freeze(a.d23_offtask_seen)
        return
    run_gpu(a.cap_h)


if __name__ == "__main__":
    main()
