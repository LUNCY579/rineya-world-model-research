"""RCX-E01R native cells (PREREG_E01R sections 3, 4 and 10): treatment rows, training, the pre-opening smoke test and
the salt-committed SEALED battery on the P3 wordings. Mirrors experiments/RCX-E01N/native_e01.py and reuses its code:
pure helpers are executed from native_e01 / native_probe / lora_cpu's own source (_borrow; those scripts import torch
at module top), and the torch-backed functions are imported only inside the model commands (train, smoke, eval).
No statistics here (analyze_e01r.py, a later step).

Arms (matched to N-CFSD: 1,501 rows, 190 steps x 16, the D-03 recipe; training seed k sets attach_lora(k) and
row_order_for(1501, 190, 16, k), exactly as for N-CFSD_sk):
  E-R1  recipes.make_r1(1501, seed=20261253, train_pool=pools/TR_v1_audited.json, p_pool=0.5)   branch B3
  E-R2  recipes.make_r2(1501, seed=20261264)                                                   branch B4
  E-R3  gen.make_train("CF_SD", 1501, 20261253); target = trace_target.trace_target(rec)       B3 and B4
  N-CFSD  control: the CX D-30 adapters, read-only from runs/RCX-E01N/cells (sha256 = D-30 value = E01N TRAIN_META.json
          = E01N SEALED_OPEN.json record); never retrained. N-VR (descriptive) and OFF may also be evaluated.
Rows are written as runs/RCX-E01R/cells/train_<arm>.jsonl (native_e01.native_rows keys) with TRAIN_MANIFEST_E01R.json,
which also records max_new = max(160, longest tokenizer-measured R3 gold target + 16) for EVERY arm's SEALED eval.

SEALED battery (surfaces `in` + `p3`, 40 parents per stratum, run_e01r.STRATA = F1-F5, V1, V2, V3, V5, V7):
  seed = int(sha256(salt)[:8], 16) % 10**8 (native_e01.sealed_seed). `in` = gen.make_battery("sealed", 40, seed,
  strata=STRATA, surfaces=("in",)) (880 records, canonical t0). `p3` = a copy of every `in` record (880) whose rule is
  rendered by a P3 wording. Exact P3 draw, per parent in battery order (review item 17):
    1. form = FORMS[rank % 3], rank = the parent's index within its stratum (a rotation as in run_e01r; no draw);
    2. rng = random.Random(f"{seed}-{parent_id}") (one per parent);
    3. j_POS = rng.randrange(n[POS/form]), then 4. j_NEG = rng.randrange(n[NEG/form]) (both always drawn, this order;
       n = the committed per-cell P3 counts of POOL_REFINE.json);
    5. a p3 record of polarity P (S+ -> POS, S- -> NEG) gets the j_P-th template of cell P/form in id-sorted order.
  mint-commit reads NO P3 file: it commits sha256(salt + ":" + sha256(skeleton jsonl)), where the skeleton holds every
  `in` record and every p3 record with tid/input/p3_template = None and its p3_pol, p3_form, p3_j. open-sealed (once,
  write-once SEALED_OPEN.json) is the first P3 read: P3_SEALED_v2.json.secret, sha256 checked against POOL_REFINE.json
  and the commitment BEFORE parsing, per-cell counts checked; templates are registered as templates.SPOS/SNEG['p3:<id>']
  in memory only. Every other check that can refuse the opening (commitment, salt, skeleton, manifest, row audit, row
  files and their shas, adapters, smoke records) runs BEFORE that read. No command prints P3 text, and the rendered
  battery is never written to disk.

Commands, in operator order (from complementary/). Branch B4 (CX D-39) is shown; B3 would use E-R1 in place of E-R2,
and the R1 path (E-R1, --r1-fallback) refuses unless BRANCH_STEP_A.json says B3:
  python -B experiments/RCX-E01R/decide_branch.py                        (BRANCH_STEP_A.json; once, after step A)
  python -B experiments/RCX-E01R/native_e01r.py build --arms E-R2,E-R3    (B4; B3: E-R1,E-R3; B1: none; tokenizer only)
  python -B experiments/RCX-E01R/audit_e01r.py --out runs/RCX-E01R/cells --arms E-R2,E-R3 --tok
      -> AUDIT_E01R.json (create-only) {"ok", "arm_ok": {arm: bool}, "rows_sha256": {arm: sha256}} (train/mint need it)
  <full freeze in DECISIONS>
  python -B experiments/RCX-E01R/native_e01r.py mint-commit --arms-required N-CFSD_s0,N-CFSD_s1,N-CFSD_s2,E-R2_s0,\
      E-R2_s1,E-R2_s2,E-R3_s0,E-R3_s1,E-R3_s2                         (B1: the three N-CFSD tags only)
      The treatment arms must equal the manifest's, which must equal BRANCH_STEP_A.json's. One arm may be dropped
      (PREREG section 2: its rows fail the row audit) only with --drop-arm <arm> --reason "<why>", recorded in
      SEALED_COMMIT.json. The commitment binds the sha256 of bound_code(), analyze_e01r.py and rcx/stats.py included;
      train refuses if any bound file's sha256 differs from it (open-sealed records any difference).
  lane chain: native_e01r.py train --arm E-R2 --seed 0 (... 6 adapters), then smoke --arm E-R2 --seed 0 (... 6)
  python -B experiments/RCX-E01R/native_e01r.py open-sealed --expect-commit-sha <SEALED_COMMIT.json sha256, DECISIONS>
  lane chain: eval --battery sealed --arm N-CFSD --seed 0|1|2 ; --arm E-R2 --seed 0|1|2 ; --arm E-R3 --seed 0|1|2 ;
      --arm OFF (optional: --arm N-VR --seed 0); then analyze_e01r.py.
Lane (CX D-36): train, smoke and eval refuse unless the lock %TEMP%/rineya-cpu-lane.json is held by 'rcx', its holder
pid is alive (a stale lock is refused) and the environment variable RCX_LANE_TOKEN equals the lock's token. The chain
exports it after acquiring:
  export RCX_LANE_TOKEN=$(python -B -c "from rcx import lane; print(lane.read_lock()['token'])")
eval writes the raw generations (raw_sealed_<tag>.json.gz, create-only) right after decoding, before the truncation
mapping, scoring or any other post-processing; if that later step fails, a rerun post-processes the raw file without
loading the model or decoding again.
CPU only; model loads refuse below 12 GB available RAM (native_probe.need_ram); rcx.guard between micro-batches.
"""
import argparse
import ast as pyast
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E01N = ROOT / "experiments" / "RCX-E01N"
for _p in (E01N, HERE, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import recipes as RC  # noqa: E402  (pure Python)
import run_e01r as RA  # noqa: E402  (pure at import: STRATA, FORMS)
import trace_target as TT  # noqa: E402  (pure Python)
from rcx import lane  # noqa: E402
from rcx.rcl import gen, score as S, templates as T, vocab as V, world as W  # noqa: E402


def _borrow(path, names, **ns):
    """Execute the named top-level definitions (def, class or assignment) of `path` from its own source, in source
    order, with globals `ns`; returns them as a namespace. Reuses native_e01 / native_probe / lora_cpu code verbatim
    without importing those scripts, which import torch at module top (cf. trace_target._borrow)."""
    tree = pyast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    want, found, body = set(names), set(), []
    for n in tree.body:
        if isinstance(n, (pyast.FunctionDef, pyast.ClassDef)):
            got = {n.name}
        elif isinstance(n, pyast.Assign):
            got = {e.id for t in n.targets for e in (t.elts if isinstance(t, pyast.Tuple) else [t])
                   if isinstance(e, pyast.Name)}
        else:
            continue
        if got & want:
            body.append(n)
            found |= got & want
    if found != want:
        raise ImportError(f"{Path(path).name}: no top-level definition of {sorted(want - found)}")
    ns.setdefault("__name__", Path(path).stem)
    exec(compile(pyast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in sorted(want)})


NPP = _borrow(E01N / "native_probe.py", ("BATTERY_SEED", "rcl_system", "user_json", "parse_answer", "_no_dup_keys"),
              json=json)
LCP = _borrow(ROOT / "rcx" / "native" / "lora_cpu.py", ("CORES", "PATHS", "EOS_IDS", "Encoded", "encode_pair",
                                                        "row_order_for"), Path=Path, random=random, dataclass=dataclass)
NE = _borrow(E01N / "native_e01.py", ("N_ROWS", "STEPS", "BATCH", "DATA_SEED", "target_json", "load_rows", "LF",
                                      "_write_once", "sealed_seed", "battery_sha256", "battery_commit", "_PROMPT_RE",
                                      "_WORD_RE", "FRAME_PRE", "FRAME_POST", "render_literal", "read_back",
                                      "token_covariates"),
             json=json, re=re, hashlib=hashlib, Path=Path, W=W, gen=gen, NP=NPP, LC=LCP)
N_ROWS, STEPS, BATCH = NE.N_ROWS, NE.STEPS, NE.BATCH
load_rows = NE.load_rows
CFSD_SEED = NE.DATA_SEED["CF_SD"]  # 20261253: N-CFSD's data seed (R1 and R3 rows)
R2_SEED = 20261264  # PREREG_E01R section 9
MAX_LEN = 512  # lora_cpu.train_sft's max_len default (tests check it); rows above it would be refused by train_sft

OUT = ROOT / "runs" / "RCX-E01R" / "cells"
BRANCH_PATH = ROOT / "runs" / "RCX-E01R" / "dev2" / "BRANCH_STEP_A.json"
DECIDE_SHA256 = "6835b0402120a2847423d8af08814cc89f311292b110365487b39d61636080d6"  # decide_branch.py, CX D-37
POOLS = HERE / "pools"
R1_POOL_PATH = POOLS / "TR_v1_audited.json"
POOL_REFINE_PATH = POOLS / "POOL_REFINE.json"
P3_PATH = POOLS / "P3_SEALED_v2.json.secret"  # read only by open-sealed and sealed_battery, never at the mint
P3_SHA256 = "88dd432bf59ba82617035d3a96280be74ef9fce233cfafa03e539cb489bcf052"  # CX D-34; PREREG_E01R section 4
ANALYZE_PATH = HERE / "analyze_e01r.py"
MANIFEST, AUDIT = "TRAIN_MANIFEST_E01R.json", "AUDIT_E01R.json"
BASE_SHA256 = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"  # Titan-Sparse FP32 model.safetensors
LANE_TOKEN_ENV = "RCX_LANE_TOKEN"  # exported by the lane chain after rcx.lane acquire (the lock's token)
E01N_CELLS = ROOT / "runs" / "RCX-E01N" / "cells"
NCFSD_ROWS_SHA256 = "de23a9233785273b429d5debd311b6309fccedfc1964ed61f1d3faedcd72e401"  # CX D-30, train_N-CFSD.jsonl
E01N_OPEN_SHA256 = "22ccc4344fa547944d3a999ecd2f17cae901cb81774059805bf77b327a9796b6"  # E01N SEALED_OPEN.json, CX D-30
E01N_ADAPTERS = {  # CX D-30 (prefixes in DECISIONS; full values as in E01N's SEALED_OPEN.json and TRAIN_META.json)
    "N-CFSD_s0": "97cc48e8bd1695fa3e21c4a63bc028cbcaa9ec8d5f20eddc931838eefa17099f",
    "N-CFSD_s1": "8e710e46b3ea61550fbc4bb76fdc81dc2bfa79ab0eae0247a1f8f0dac924c3e8",
    "N-CFSD_s2": "652b497d044f8d4d4f176a2b8c985ac464662552ad774ad4b41db28359bf941e",
    "N-VR_s0": "2f37688256132a8c5bbc9aec4be748d7008f2441e526239185c36287856c0ef1",
    "N-VR_s1": "52adbde929984bbaaf452d0385e6e8c2db9856f09adc283276476ec2df9f8200",
    "N-VR_s2": "876ac52d0054f940a9699043fcc58dd47f742b713e408543e6f51fe9f32405e6"}

N_PARENTS = 40
STRATA, FORMS = RA.STRATA, RA.FORMS
POLS = ("POS", "NEG")
CELLS = tuple(f"{p}/{f}" for p in POLS for f in FORMS)
SURFACES = ("in", "p3")
TREATMENTS = {"E-R1": "R1", "E-R2": "R2", "E-R3": "R3"}
BRANCH_ARMS = {"B1": [], "B3": ["E-R1", "E-R3"], "B4": ["E-R2", "E-R3"]}  # PREREG_E01R section 2 (FROZEN)
TAG_RE = re.compile(r"(N-CFSD|E-R1|E-R2|E-R3)_s(\d+)")
R1_MIN_PER_CELL = 6
MAX_NEW_FLOOR, MAX_NEW_SLACK = TT.TRACE_MAX_NEW, 16  # review item 15: max(160, longest R3 gold + 16)
SMOKE_N, SMOKE_MAX_TRUNC = 50, 0.02  # review item 16
TOL = 1e-9
DRAW = ("per parent in battery order: form = FORMS[rank % 3] (rank within its stratum; no draw); "
        "rng = random.Random(f'{seed}-{parent_id}'); j_POS = rng.randrange(n[POS/form]); "
        "j_NEG = rng.randrange(n[NEG/form]); a p3 record of polarity P uses the j_P-th template of cell P/form, "
        "id-sorted")
DERIVATION = ("seed = int(sha256(salt)[:8], 16) % 10**8; in = gen.make_battery('sealed', n_parents, seed, strata, "
              "surfaces=('in',)); skeleton = in + p3 copies (tid/input/p3_template None; p3_pol, p3_form, p3_j); "
              "battery_commit = sha256(salt + ':' + sha256(skeleton jsonl bytes))")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def bound_code():
    """The code whose sha256 the SEALED commitment binds (PREREG_E01R section 4): this module, the recipes, the R3
    target, the row audits, the frozen branch rule, the step A runner, the E01N code it borrows, lora_cpu, the lane,
    the rcx.rcl generator/score/wording-audit modules, rcx.stats and the analysis (analyze_e01r.py; must exist at the
    mint)."""
    return [Path(__file__), HERE / "recipes.py", HERE / "trace_target.py", HERE / "audit_e01r.py",
            HERE / "audit_tr_pool.py", HERE / "decide_branch.py", HERE / "run_e01r.py", E01N / "native_e01.py",
            E01N / "native_probe.py", ROOT / "rcx" / "native" / "lora_cpu.py", ROOT / "rcx" / "lane.py",
            ROOT / "rcx" / "stats.py", ROOT / "rcx" / "rcl" / "wording_audit.py"] \
        + [Path(m.__file__) for m in (gen, V, T, W, S)] + [ANALYZE_PATH]


def code_sha256(strict=False):
    """{file name: sha256} of bound_code(). strict (the mint): SystemExit if any bound file is missing; otherwise a
    missing file (analyze_e01r.py before it is written) is left out."""
    files = bound_code()
    missing = [p.name for p in files if not p.exists()]
    if strict and missing:
        raise SystemExit(f"the SEALED commitment binds files that do not exist: {missing}")
    if len({p.name for p in files}) != len(files):
        raise RuntimeError("bound code file names collide")
    return {p.name: sha(p) for p in files if p.exists()}


def _heavy():
    """The torch-backed modules, imported only by the model commands (never at import time, never in tests)."""
    import native_probe
    from rcx.guard import ThermalGuard
    from rcx.native import lora_cpu
    return native_probe, lora_cpu, ThermalGuard


def require_lane():
    """The lane chain acquires rcx.lane (CX D-36) and exports the lock's token as RCX_LANE_TOKEN. A model command
    refuses unless 'rcx' holds the lock, the holder pid is alive (a stale lock of a dead chain is refused, never
    reclaimed here) and RCX_LANE_TOKEN equals the lock's token (this process runs inside the chain that acquired it)."""
    lk = lane.read_lock()
    if not lk or lk.get("holder") != lane.PARTY:
        raise SystemExit(f"refusing: the CPU lane lock is not held by {lane.PARTY!r} (lock: "
                         f"{lk and {k: lk.get(k) for k in ('holder', 'job')}}); run inside a chain that holds it")
    try:
        alive = lane.pid_alive(int(lk.get("pid")))
    except (TypeError, ValueError, OverflowError):
        alive = False
    if not alive:
        raise SystemExit(f"refusing: the CPU lane lock is stale (holder pid {lk.get('pid')!r} is not alive); "
                         "run inside a chain that holds the lane")
    if not lk.get("token") or os.environ.get(LANE_TOKEN_ENV) != lk.get("token"):
        raise SystemExit(f"refusing: {LANE_TOKEN_ENV} does not equal the CPU lane lock's token; run inside the chain "
                         "that acquired the lane (it exports the token after acquiring)")
    return {k: lk.get(k) for k in ("holder", "job", "pid", "token")}


def require_audit(cells, arm, rows_sha):
    """Refuse rows that have not passed the E01R row audit: AUDIT_E01R.json audited this arm's rows (sha) and passed
    them, i.e. ok, or arm_ok[arm] (no failure naming this arm and no arm-independent failure; audit_e01r.py). Returns
    the AUDIT_E01R.json sha256."""
    p = Path(cells) / AUDIT
    aud = read_json(p) if p.exists() else {}
    passed = aud.get("ok") is True or (aud.get("arm_ok") or {}).get(arm) is True
    if not passed or aud.get("rows_sha256", {}).get(arm) != rows_sha:
        raise SystemExit(f"refusing {arm}: {AUDIT} is missing, failed for this arm, or audited other rows")
    return sha(p)


def treatment_tag(arm, seed):
    if arm not in TREATMENTS or seed not in (0, 1, 2):
        raise SystemExit(f"--arm must be one of {sorted(TREATMENTS)} and --seed 0, 1 or 2 (got {arm!r}, {seed})")
    return f"{arm}_s{seed}"


# ----------------------------------------------------------------------------- build (rows, manifest, max_new)

def branch_arms(path=None):
    """(branch, treatment arms, record) from BRANCH_STEP_A.json, written by the frozen decide_branch.py."""
    p = Path(path or BRANCH_PATH)
    if not p.exists():
        raise SystemExit(f"{p} missing: run decide_branch.py after step A first")
    q = read_json(p)
    b = q.get("branch")
    if q.get("code_sha256") != DECIDE_SHA256:
        raise SystemExit("BRANCH_STEP_A.json was not written by the frozen decide_branch.py (CX D-37)")
    if b not in BRANCH_ARMS:
        raise SystemExit(f"branch {b}: no E01R rows are built (PREREG_E01R section 2)")
    if [f"E-{x}" for x in q.get("arms", [])] != BRANCH_ARMS[b]:
        raise SystemExit(f"BRANCH_STEP_A.json arms {q.get('arms')} disagree with branch {b}")
    return b, list(BRANCH_ARMS[b]), {"file": p.name, "sha256": sha(p), "branch": b, "arms": q.get("arms")}


def require_r1_branch(branch, what):
    """The R1 path (E-R1 rows, the R1 pool, --r1-fallback) exists only under branch B3 (PREREG_E01R section 2)."""
    if branch != "B3":
        raise SystemExit(f"{what}: the R1 path exists only under branch B3; BRANCH_STEP_A.json says {branch}")


def committed_arms(branch, manifest):
    """The treatment arms the mint must commit: BRANCH_STEP_A.json's (B4: E-R2, E-R3; B3: E-R1, E-R3; B1: none), or,
    after build's section 2 fallback (B3 only, recorded in the manifest), E-R2 and E-R3."""
    fb = manifest.get("fallback")
    if not fb:
        return list(BRANCH_ARMS[branch])
    require_r1_branch(branch, "the manifest's R1 -> R2 fallback")
    if (fb.get("from"), fb.get("to")) != ("E-R1", "E-R2"):
        raise SystemExit(f"{MANIFEST} records a fallback other than section 2's R1 -> R2: {fb}")
    return list(BRANCH_ARMS["B4"])


def load_r1_pool(path=None):
    """(templates, info) of the audited R1 training pool ({"templates": [...]}; never a *.secret file)."""
    p = Path(path or R1_POOL_PATH)
    if p.name.endswith(".secret"):
        raise SystemExit("the R1 pool is never a sealed file")
    if not p.exists():
        return None, {"file": p.name, "missing": True, "counts": {c: 0 for c in CELLS}}
    pool = read_json(p)["templates"]
    n = Counter(f"{t.get('pol')}/{t.get('form')}" for t in pool)
    return pool, {"file": p.name, "sha256": sha(p), "n": len(pool), "counts": {c: n.get(c, 0) for c in CELLS}}


def r1_shortfall(info):
    return {c: k for c, k in info["counts"].items() if k < R1_MIN_PER_CELL}


def treatment_records(arm, r1_pool=None):
    """Generator records of a treatment arm (PREREG_E01R section 3, FROZEN specifications); deterministic."""
    gen.use_world("arith")
    if arm == "E-R1":
        return RC.make_r1(N_ROWS, seed=CFSD_SEED, train_pool=r1_pool, p_pool=0.5)
    if arm == "E-R2":
        return RC.make_r2(N_ROWS, seed=R2_SEED)
    if arm == "E-R3":
        return gen.make_train("CF_SD", N_ROWS, CFSD_SEED)
    raise ValueError(arm)


def arm_rows(arm, tok, r1_pool=None):
    """(records, rows): native_e01.native_rows rows (recipes.native_rows); R3 replaces only the target."""
    recs = treatment_records(arm, r1_pool)
    rows = RC.native_rows(recs, tok, TREATMENTS[arm], np_mod=NPP)
    if arm == "E-R3":
        for row, r in zip(rows, recs):
            row["target"] = TT.trace_target(r)
    return recs, rows


def ncfsd_reference(tok, path=None, strict=True):
    """N-CFSD's training rows: runs/RCX-E01N/cells/train_N-CFSD.jsonl (sha256 = CX D-30), or, with strict=False and
    no such file, the same rows rebuilt with `tok`."""
    p = Path(path) if path else E01N_CELLS / "train_N-CFSD.jsonl"
    if p.exists():
        if sha(p) != NCFSD_ROWS_SHA256:
            raise SystemExit(f"{p} is not N-CFSD's CX D-30 row file")
        with open(p, encoding="utf-8") as f:
            return [json.loads(x) for x in f], {"source": p.name, "sha256": NCFSD_ROWS_SHA256}
    if strict:
        raise SystemExit(f"{p} missing: the build compares R1/R3 rows with N-CFSD's")
    gen.use_world("arith")
    return RC.native_rows(gen.make_train("CF_SD", N_ROWS, CFSD_SEED), tok, "CF_SD", np_mod=NPP), {"source": "rebuilt"}


def compare_ncfsd(arm, rows, ref):
    """Build-time identity checks against N-CFSD's rows: E-R3 prompts are byte-identical and its last target line is
    N-CFSD's target; E-R1 targets and row metadata are identical and only pooled prompts differ. SystemExit on any
    difference; returns counts (None for E-R2, whose parents differ by design)."""
    if arm == "E-R2":
        return None
    if len(rows) != len(ref):
        raise SystemExit(f"{arm}: {len(rows)} rows vs N-CFSD {len(ref)}")
    same = 0
    for i, (a, b) in enumerate(zip(rows, ref)):
        keys = ("variant", "decision", "semantic_hash", "skeleton_hash", "status")
        bad = [k for k in keys if a[k] != b[k]]
        if arm == "E-R3":
            bad += [k for k, x, y in (("prompt", a["prompt"], b["prompt"]),
                                      ("parent_id", a["parent_id"], b["parent_id"]),
                                      ("target", a["target"].split("\n")[-1], b["target"])) if x != y]
        else:
            bad += ["target"] if a["target"] != b["target"] else []
        if bad:
            raise SystemExit(f"{arm} row {i}: {bad} differ from N-CFSD's")
        same += a["prompt"] == b["prompt"]
    if arm == "E-R1" and not 0 < same < len(rows):
        raise SystemExit(f"E-R1: {len(rows) - same} pooled prompts of {len(rows)}")
    return {"rows": len(rows), "prompts_identical": same, "prompts_changed": len(rows) - same}


def r3_gold_targets():
    """R3 gold targets whose tokenized length fixes max_new: the R3 training targets and the E01N DEV battery's `in`
    records of the E01R strata (seed 20261240; the SEALED battery is unknown at build time)."""
    gen.use_world("arith")
    dev = gen.make_battery("dev", N_PARENTS, NPP.BATTERY_SEED, strata=STRATA, surfaces=("in",))
    return {"R3 training targets": [TT.trace_target(r) for r in gen.make_train("CF_SD", N_ROWS, CFSD_SEED)],
            f"DEV battery {NPP.BATTERY_SEED} in, E01R strata": [TT.trace_target(r) for r in dev]}


def measure_max_new(tok, groups=None):
    """(max_new, longest tokens per group): max(160, longest tokenized R3 gold target + 16) (review item 15)."""
    groups = r3_gold_targets() if groups is None else groups
    longest = {k: max(len(tok(t, add_special_tokens=False)["input_ids"]) for t in v) for k, v in groups.items()}
    return max(MAX_NEW_FLOOR, max(longest.values()) + MAX_NEW_SLACK), longest


def write_rows(p, rows):
    """Rows as native_e01.cmd_build writes them; an existing file must already hold the same bytes (built once)."""
    data = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    if p.exists() and p.read_bytes() != data:
        raise SystemExit(f"{p.name} exists with other content; rows are built once")
    p.write_bytes(data)


def rule_kind(rule):
    return "composition" if rule["op"] in ("AND", "OR") else rule["op"]


def build_cells(out, arms, tok, branch_path=None, r1_pool_path=None, ncfsd_rows_path=None, strict=True,
                r1_fallback=False):
    """Rows + TRAIN_MANIFEST_E01R.json (create-only) for the treatment arms of BRANCH_STEP_A.json. --r1-fallback
    (PREREG_E01R section 2: B3 with an R1 pool below 6 per cell) builds E-R2 + E-R3 instead and records why."""
    out = Path(out)
    branch, want, brec = branch_arms(branch_path)
    if r1_fallback:
        require_r1_branch(branch, "--r1-fallback")
    if "E-R1" in arms:
        require_r1_branch(branch, "building E-R1")
    pool, pool_info = load_r1_pool(r1_pool_path) if branch == "B3" else (None, None)  # never read the pool otherwise
    fallback = None
    if r1_fallback:
        short = r1_shortfall(pool_info)
        if branch != "B3" or arms != BRANCH_ARMS["B4"] or not short:
            raise SystemExit("--r1-fallback applies only under B3, with arms E-R2,E-R3, when the R1 pool is below its "
                             f"minimum (shortfall {short})")
        fallback = {"from": "E-R1", "to": "E-R2", "r1_pool": pool_info, "shortfall": short}
    elif arms != want:
        raise SystemExit(f"--arms {arms} != the arms of branch {branch}: {want}")
    if "E-R1" in arms and r1_shortfall(pool_info):
        raise SystemExit(f"R1 pool below {R1_MIN_PER_CELL} per cell: {r1_shortfall(pool_info)} (one blind top-up "
                         "round, then --r1-fallback; PREREG_E01R section 2)")
    if (out / MANIFEST).exists():
        raise SystemExit(f"{MANIFEST} exists; E01R rows are built once")
    out.mkdir(parents=True, exist_ok=True)
    ref, ref_info = ncfsd_reference(tok, ncfsd_rows_path, strict) if arms else (None, None)
    man = {"experiment": "RCX-E01R", "branch": brec, "fallback": fallback, "reference_rows": ref_info,
           "r1_pool": pool_info if "E-R1" in arms else None, "arms": {}}
    for arm in arms:
        recs, rows = arm_rows(arm, tok, pool)
        cmp = compare_ncfsd(arm, rows, ref)
        p = out / f"train_{arm}.jsonl"
        write_rows(p, rows)
        ntok = [len(tok(r["prompt"] + r["target"], add_special_tokens=False).input_ids) for r in rows]
        cov = NE.token_covariates(rows, tok)
        if cov["max_processed_per_row"] > MAX_LEN:
            raise SystemExit(f"{arm}: a row has {cov['max_processed_per_row']} tokens > max_len {MAX_LEN}")
        man["arms"][arm] = {"path": p.name, "sha256": sha(p), "rows": len(rows), "recipe": TREATMENTS[arm],
                            "data_seed": R2_SEED if arm == "E-R2" else CFSD_SEED,
                            "noop": sum(r["decision"] == "noop" for r in rows),
                            "parents": len({r["parent_id"] for r in rows}), "max_tokens": max(ntok),
                            "mean_tokens": sum(ntok) / len(ntok), "tokens": cov,
                            "rows_by_rule_type": dict(sorted(Counter(rule_kind(r["rule_ast"]) for r in recs).items())),
                            "recipe_stats": RC.recipe_stats(recs), "vs_N-CFSD": cmp}
        print(arm, {k: man["arms"][arm][k] for k in ("sha256", "rows", "noop", "parents", "max_tokens")}, flush=True)
    man["max_new"], man["r3_gold_longest_tokens"] = measure_max_new(tok)
    man["max_new_rule"] = f"max({MAX_NEW_FLOOR}, longest tokenized R3 gold target + {MAX_NEW_SLACK}); every arm"
    man["tokenizer"] = type(tok).__name__
    man["code_sha256"] = code_sha256()
    man["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    NE._write_once(out / MANIFEST, json.dumps(man, indent=1))
    return man


def cmd_build(a):
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if arms == ["none"]:
        arms = []
    elif not arms:
        raise SystemExit("--arms is mandatory: E-R1,E-R3 (B3) | E-R2,E-R3 (B4) | none (B1)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LCP.PATHS["sparse"]), local_files_only=True)
    man = build_cells(a.out, arms, tok, r1_fallback=a.r1_fallback)
    print(json.dumps({k: man[k] for k in ("branch", "fallback", "max_new", "r3_gold_longest_tokens")}, indent=1))


# ----------------------------------------------------------------------------- train (D-03, exactly native_e01's)

def cmd_train(a):
    tag = treatment_tag(a.arm, a.seed)
    out = Path(a.out) / f"{a.arm}_s{a.seed}"
    if (out / "adapter" / "adapter_model.safetensors").exists():
        print("adapter exists; refusing to retrain under the same id", flush=True)
        return
    commit_p, _, open_p = sealed_paths(a.out)
    if open_p.exists():
        raise SystemExit("the SEALED battery is open; no adapter is trained after opening (PREREG_E01R section 11)")
    if not commit_p.exists():
        raise SystemExit("no SEALED_COMMIT.json: mint-commit precedes treatment training (PREREG_E01R section 10)")
    commit = read_json(commit_p)
    if tag not in commit["arms_required"]:
        raise SystemExit(f"{tag} is not in the committed arms_required {commit['arms_required']}")
    man_p = Path(a.out) / MANIFEST
    if sha(man_p) != commit["train_manifest_sha256"]:
        raise SystemExit(f"{MANIFEST} changed after the mint")
    # Provenance is fixed BEFORE the hours-long run (as in native_e01.cmd_train).
    rows_sha = sha(Path(a.out) / f"train_{a.arm}.jsonl")
    if read_json(man_p)["arms"][a.arm]["sha256"] != rows_sha \
            or commit.get("rows_sha256", {}).get(a.arm, rows_sha) != rows_sha:
        raise SystemExit(f"train_{a.arm}.jsonl differs from {MANIFEST} / SEALED_COMMIT.json")
    rows = load_rows(a.out, a.arm)
    audit_sha = require_audit(a.out, a.arm, rows_sha)
    if audit_sha != commit.get("audit_e01r_sha256"):
        raise SystemExit(f"{AUDIT} is not the row audit bound by SEALED_COMMIT.json (changed after the mint)")
    drift = _code_diff(commit)
    if drift:
        raise SystemExit(f"bound code differs from SEALED_COMMIT.json: {drift} (PREREG_E01R section 4)")
    lock = require_lane()
    code_sha = code_sha256()
    NP, LC, ThermalGuard = _heavy()
    NP.need_ram()
    base_sha = sha(LC.PATHS["sparse"] / "model.safetensors")
    if base_sha != BASE_SHA256:
        raise SystemExit(f"the base model.safetensors sha256 {base_sha[:12]} is not Titan-Sparse FP32 "
                         f"{BASE_SHA256[:12]} (PREREG_E01R section 1)")
    model, tok = LC.load_core("sparse")
    pm = LC.attach_lora(model, seed=a.seed)
    order = LC.row_order_for(len(rows), STEPS, BATCH, a.seed)
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    logf = open(Path(a.out) / f"train_{a.arm}_s{a.seed}.log", "a", encoding="utf-8", newline="\n")

    def P(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log = LC.train_sft(pm, tok, rows, steps=STEPS, batch_size=BATCH, seed=a.seed, row_order=order, logger=P,
                       guard=guard, ckpt_dir=str(out / "ckpt"), ckpt_every=10)
    meta = {"arm": a.arm, "recipe_id": TREATMENTS[a.arm], "seed": a.seed, "core": "titan-sparse",
            "base_sha256": base_sha, "rows_sha256": rows_sha, "audit_sha256": audit_sha,
            "sealed_commit_sha256": sha(commit_p),
            "recipe": "D-03 (r16/a32/do0.05; 190x16; lr 1e-4 WSD; AdamW 0.9/0.95) on CPU FP32, unmerged; "
                      "native_e01.cmd_train's train_sft call", "train": log, "code_sha256": code_sha, "lane": lock}
    LC.save_adapter(pm, out / "adapter", meta)
    meta["adapter_sha256"] = sha(out / "adapter" / "adapter_model.safetensors")
    (out / "TRAIN_META.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    ck = out / "ckpt" / "state.pt"
    if ck.exists():
        ck.unlink()
    P(f"done {log['wall_s']}s adapter sha {meta['adapter_sha256'][:12]}")


def completed_adapter(cells, tag, manifest):
    """(record, None) if treatment `tag` is a completed, untampered 190-step run on the manifest's rows (no resume
    checkpoint left), trained on the Titan-Sparse FP32 base (BASE_SHA256) under the current SEALED_COMMIT.json and its
    row audit, else (None, reason). The E01R analogue of native_e01.completed_adapter."""
    m = TAG_RE.fullmatch(tag)
    if not m or m.group(1) not in TREATMENTS:
        return None, "not a treatment adapter tag"
    d = Path(cells) / tag
    ad, meta_p = d / "adapter" / "adapter_model.safetensors", d / "TRAIN_META.json"
    commit_p = sealed_paths(cells)[0]
    if not ad.exists():
        return None, "adapter missing"
    if not meta_p.exists():
        return None, "TRAIN_META.json missing (training incomplete)"
    if (d / "ckpt" / "state.pt").exists():
        return None, "resume checkpoint still present (training incomplete)"
    if not commit_p.exists():
        return None, "no SEALED_COMMIT.json (treatments are trained after the mint)"
    meta, got = read_json(meta_p), sha(ad)
    if meta.get("adapter_sha256") != got:
        return None, "adapter sha differs from TRAIN_META.json"
    if (meta.get("arm"), meta.get("seed")) != (m.group(1), int(m.group(2))):
        return None, "TRAIN_META.json arm/seed differ from the tag"
    if meta.get("rows_sha256") != manifest.get("arms", {}).get(m.group(1), {}).get("sha256"):
        return None, f"rows sha differs from {MANIFEST}"
    if meta.get("train", {}).get("steps") != STEPS:
        return None, f"not a {STEPS}-step run"
    if meta.get("sealed_commit_sha256") != sha(commit_p):
        return None, "TRAIN_META.json names another SEALED_COMMIT.json (not trained under this commitment)"
    if meta.get("audit_sha256") != read_json(commit_p).get("audit_e01r_sha256"):
        return None, f"TRAIN_META.json names another {AUDIT} than the commitment"
    if meta.get("base_sha256") != BASE_SHA256:
        return None, "base model sha is not Titan-Sparse FP32 (BASE_SHA256)"
    return {"adapter_sha256": got, "adapter_dir": str(d / "adapter"), "train_meta_sha256": sha(meta_p),
            "rows_sha256": meta["rows_sha256"], "base_sha256": meta.get("base_sha256"),
            "code_sha256": meta.get("code_sha256")}, None


def e01n_adapter(tag):
    """(record, None) if `tag` is a CX D-30 E01N adapter whose file sha equals the D-30 value, its TRAIN_META.json and
    E01N's SEALED_OPEN.json record (190 steps, no checkpoint; N-CFSD rows = de23a923...), else (None, reason)."""
    d, op = E01N_CELLS / tag, E01N_CELLS / "SEALED_OPEN.json"
    ad, meta_p = d / "adapter" / "adapter_model.safetensors", d / "TRAIN_META.json"
    if tag not in E01N_ADAPTERS:
        return None, "not a CX D-30 adapter"
    if not (ad.exists() and meta_p.exists()) or (d / "ckpt" / "state.pt").exists():
        return None, "adapter or TRAIN_META.json missing, or a checkpoint present"
    if not op.exists() or sha(op) != E01N_OPEN_SHA256:
        return None, "E01N SEALED_OPEN.json missing or not the CX D-30 record"
    meta, got = read_json(meta_p), sha(ad)
    if not got == E01N_ADAPTERS[tag] == meta.get("adapter_sha256") == read_json(op)["adapters_sha256"].get(tag):
        return None, "adapter sha differs from CX D-30 / TRAIN_META.json / E01N SEALED_OPEN.json"
    if meta.get("train", {}).get("steps") != STEPS:
        return None, f"not a {STEPS}-step run"
    if meta.get("base_sha256") != BASE_SHA256:
        return None, "base model sha is not Titan-Sparse FP32 (BASE_SHA256)"
    if tag.startswith("N-CFSD") and meta.get("rows_sha256") != NCFSD_ROWS_SHA256:
        return None, "rows sha is not N-CFSD's"
    return {"adapter_sha256": got, "adapter_dir": str(d / "adapter"), "train_meta_sha256": sha(meta_p),
            "rows_sha256": meta.get("rows_sha256"), "e01n_sealed_open_sha256": E01N_OPEN_SHA256}, None


# ----------------------------------------------------------------------------- scoring and decoding helpers

def last_line_answer(text, keys):
    """PREREG_E01R section 3 (FROZEN) R3 scoring: the balances on the last non-empty line, parsed by
    native_probe.parse_answer; a one-line output (e.g. the bare native JSON) is its own last line."""
    lines = [x for x in text.split("\n") if x.strip()]
    return NPP.parse_answer(lines[-1], keys) if lines else None


_K = ["A", "B"]
LAST_LINE_PROBES = (('check status="vervil" listed=no only-listed:false -> noop\n{"A":1,"B":2}', _K),
                    ('{"A":1,"B":2}', _K), ('{"A":1,"B":2}\n\n  \n', _K), ('{"A":1,"B":2}\nnot json', _K), ("", _K),
                    ('{"A":1,"B":2}.', _K), ('x\n{"A":1}', _K), ('{"after":{"A":1,"B":2}}', _K), ("\n", _K),
                    ('{"A":1,"B":2}\n{"A":3,"B":4}', _K), ('check x -> apply\n{"A":1,"A":2,"B":3}', _K))


def answer_parser(arm):
    """(parse(text, keys), name). E-R3: the last non-empty line (trace_target.parse_last_line when it exists, checked
    against the frozen rule); every other arm: native_probe.parse_answer on the whole output (E01N, unchanged)."""
    if arm != "E-R3":
        return NPP.parse_answer, "native_probe.parse_answer(whole output)"
    f = getattr(TT, "parse_last_line", None)
    if f is None:
        return last_line_answer, "last non-empty line (native_e01r.last_line_answer)"
    if any(f(t, k) != last_line_answer(t, k) for t, k in LAST_LINE_PROBES):
        raise SystemExit("trace_target.parse_last_line disagrees with the frozen R3 scoring rule")
    return f, "trace_target.parse_last_line"


class EosProbe:
    """Tokenizer proxy for lora_cpu.generate (frozen; it returns text only). It records, for every decoded row, whether
    the generated ids contain an EOS id; a row without one ran to max_new, i.e. it is truncated."""

    def __init__(self, tok, eos_ids):
        self._tok, self._eos, self.calls = tok, {int(i) for i in eos_ids}, []

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, *args, **kw):
        return self._tok(*args, **kw)

    def decode(self, ids, **kw):
        text = self._tok.decode(ids, **kw)
        vals = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        self.calls.append((text, any(int(i) in self._eos for i in vals)))
        return text

    def prompt_lengths(self, prompts):
        """Prompt token lengths exactly as lora_cpu.generate computes them (it sorts its batches by them)."""
        return [len(self._tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    def finished(self, prompts, outs, lens=None):
        """Per prompt, in prompt order (map_finished over this probe's decode log)."""
        return map_finished(self.prompt_lengths(prompts) if lens is None else lens, self.calls, outs)


def map_finished(lens, calls, outs):
    """Per prompt, in prompt order: whether its decoded ids held an EOS id. lora_cpu.generate decodes rows in order of
    prompt token length `lens` (stable sort); `calls` = [(text, has_eos)] in decode order. Every decoded text is checked
    against the returned output, so any other order fails closed (RuntimeError). Pure: no tokenizer, no model."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    if not len(calls) == len(lens) == len(outs) or any(outs[j] != calls[k][0] for k, j in enumerate(order)):
        raise RuntimeError("cannot map decoded rows to prompts; truncation unknown")
    fin = [None] * len(outs)
    for k, j in enumerate(order):
        fin[j] = bool(calls[k][1])
    return fin


# ----------------------------------------------------------------------------- smoke (review item 16)

def smoke_select(n_rows, k=SMOKE_N):
    """k TRAIN rows, evenly spaced (index i * n_rows // k): no randomness, never a DEV or SEALED record."""
    return [i * n_rows // k for i in range(k)]


def smoke_report(arm, rows, outs, finished):
    """Shares well-formed / scorable / exact / truncated on TRAIN prompts; pass iff truncated share <= 2%."""
    parse, _ = answer_parser(arm)
    items = []
    for r, o, fin in zip(rows, outs, finished):
        gold = json.loads(r["target"].split("\n")[-1])
        got = parse(o, list(gold))
        wf = TT.parse_trace_answer(o, list(gold))[2] if arm == "E-R3" else got is not None
        items.append({"finished": bool(fin), "well_formed": int(wf), "scorable": int(got is not None),
                      "exact": int(got == gold)})
    n = len(items)
    share = {f"{k}_share": sum(x[k] for x in items) / n for k in ("well_formed", "scorable", "exact")}
    trunc = sum(not x["finished"] for x in items)
    return {"n": n, "truncated": trunc, "truncated_share": trunc / n, **share, "max_truncated_share": SMOKE_MAX_TRUNC,
            "pass": trunc / n <= SMOKE_MAX_TRUNC + TOL, "items": items}


def smoke_problem(cells, tag, adapter_sha, max_new):
    p = Path(cells) / f"SMOKE_{tag}.json"
    if not p.exists():
        return "smoke test missing"
    s = read_json(p)
    if (s.get("adapter_sha256"), s.get("max_new"), s.get("n")) != (adapter_sha, max_new, SMOKE_N):
        return "smoke test was run on another adapter, max_new or row count"
    return None if s.get("pass") is True else f"smoke test failed ({s.get('truncated')}/{s.get('n')} truncated > 2%)"


def cmd_smoke(a):
    tag = treatment_tag(a.arm, a.seed)
    cells = Path(a.out)
    res_p = cells / f"SMOKE_{tag}.json"
    if res_p.exists():
        print("smoke exists", flush=True)
        return
    commit_p, _, open_p = sealed_paths(cells)
    if open_p.exists():
        raise SystemExit("the smoke test runs before open-sealed")
    if not commit_p.exists():
        raise SystemExit("no SEALED_COMMIT.json: the smoke test follows the mint and the training")
    commit = read_json(commit_p)
    if tag not in commit["arms_required"]:
        raise SystemExit(f"{tag} is not in the committed arms_required {commit['arms_required']}")
    man_p = cells / MANIFEST
    if sha(man_p) != commit["train_manifest_sha256"]:
        raise SystemExit(f"{MANIFEST} changed after the mint")
    man = read_json(man_p)
    max_new = commit["max_new"]  # the committed value (= the manifest's), as in the SEALED eval
    if man.get("max_new") != max_new:
        raise SystemExit("max_new: the manifest differs from the commitment")
    info, why = completed_adapter(cells, tag, man)
    if why:
        raise SystemExit(f"{tag}: {why}")
    if sha(cells / f"train_{a.arm}.jsonl") != info["rows_sha256"]:
        raise SystemExit(f"train_{a.arm}.jsonl differs from the adapter's training rows")
    rows = load_rows(a.out, a.arm)
    idx = smoke_select(len(rows))
    sel = [rows[i] for i in idx]
    answer_parser(a.arm)  # fails closed before any model load
    lock = require_lane()
    NP, LC, ThermalGuard = _heavy()
    NP.need_ram()
    model, tok = LC.load_core("sparse")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, info["adapter_dir"], is_trainable=False).eval()
    probe = EosProbe(tok, LC.EOS_IDS)
    prompts = [r["prompt"] for r in sel]
    lens = probe.prompt_lengths(prompts)
    t0 = time.time()
    outs = LC.generate(model, probe, prompts, max_new=max_new,
                       guard=ThermalGuard(log=lambda m: print("guard:", m, flush=True)))
    wall = round(time.time() - t0, 1)
    rep = smoke_report(a.arm, sel, outs, map_finished(lens, probe.calls, outs))
    rec = {"tag": tag, "adapter_sha256": info["adapter_sha256"], "rows_sha256": info["rows_sha256"],
           "row_indices": idx, "max_new": max_new, **rep, "outs": outs, "wall_s": wall, "lane": lock}
    NE._write_once(res_p, json.dumps(rec, indent=1))
    print(f"[{tag}] smoke: truncated {rec['truncated']}/{rec['n']}, well-formed {rec['well_formed_share']:.2f}, "
          f"pass {rec['pass']}", flush=True)


# ----------------------------------------------------------------------------- SEALED battery (in + p3)

def sealed_paths(out):
    out = Path(out)
    return out / "SEALED_COMMIT.json", out / "SEALED_SALT.secret", out / "SEALED_OPEN.json"


def p3_reference(path=None):
    """The committed P3_v2 facts from POOL_REFINE.json (never the P3 file): file name, sha256, n, per-cell counts."""
    ref = read_json(path or POOL_REFINE_PATH)
    o = ref["outputs"]["P3"]
    counts = {c: int(ref["kept_by_cell"]["P3"].get(c, 0)) for c in CELLS}
    if (o["file"], o["sha256"]) != (P3_PATH.name, P3_SHA256) or sum(counts.values()) != o["n"] \
            or min(counts.values()) < 1:
        raise SystemExit("POOL_REFINE.json does not record the CX D-34 P3_v2 file and its per-cell counts")
    return {"file": o["file"], "sha256": o["sha256"], "n": o["n"], "counts": counts}


def p3_draw(seed, parent_id, form, counts):
    """The parent's P3 template indices (see the module doc): one Random per parent; POS index, then NEG index."""
    rng = random.Random(f"{seed}-{parent_id}")
    j_pos = rng.randrange(counts[f"POS/{form}"])
    j_neg = rng.randrange(counts[f"NEG/{form}"])
    return {"POS": j_pos, "NEG": j_neg}


def build_skeleton(seed, counts, n_parents=N_PARENTS):
    """The P3-free SEALED skeleton (reads no file): `in` records, then a p3 copy of each with tid/input/p3_template
    None and its p3_pol, p3_form and p3_j."""
    gen.use_world("arith")
    base = gen.make_battery("sealed", n_parents, seed, strata=STRATA, surfaces=("in",))
    parents = defaultdict(list)
    for r in base:
        parents[r["parent_id"]].append(r)
    out, rank = list(base), Counter()
    for pid, recs in parents.items():
        stratum = recs[0]["group"]
        form = FORMS[rank[stratum] % len(FORMS)]
        rank[stratum] += 1
        j = p3_draw(seed, pid, form, counts)
        for r in recs:
            pol = {"S+": "POS", "S-": "NEG"}.get(r["rule_ast"]["op"])
            if pol is None:
                raise SystemExit(f"{pid}: non-atomic rule {r['rule_ast']['op']} in a single-atom stratum")
            q = json.loads(json.dumps(r))
            q["surface"], q["tid"], q["input"] = "p3", None, None
            q.update({"p3_pol": pol, "p3_form": form, "p3_j": j[pol], "p3_template": None})
            out.append(q)
    return out


def load_p3(ref, path=None):
    """{cell: [(id, text)] id-sorted} from the sealed P3 file (open-sealed and later only). The sha256 is checked
    against the commitment BEFORE parsing; then ids, single-slot form and per-cell counts. Errors never show text."""
    data = Path(path or P3_PATH).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise SystemExit("the P3 file's sha256 differs from the commitment")
    by, ids = defaultdict(list), set()
    for k, t in enumerate(json.loads(data.decode("utf-8"))["templates"]):
        cell, text, tid = f"{t.get('pol')}/{t.get('form')}", t.get("text"), t.get("id")
        body = text.replace("{L}", "", 1) if isinstance(text, str) else "{"
        if cell not in CELLS or not isinstance(tid, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", tid) or tid in ids \
                or "{" in body or "}" in body or "\n" in body or body == text:
            raise SystemExit(f"P3 entry #{k} is malformed (text not shown)")
        ids.add(tid)
        by[cell].append((tid, text))
    if {c: len(by[c]) for c in CELLS} != ref["counts"]:
        raise SystemExit("P3 per-cell counts differ from the commitment")
    return {c: sorted(by[c]) for c in CELLS}


def register_p3(by_cell):
    """templates.SPOS/SNEG['p3:<id>'] in this process only (as run_e01r.load_p2 does for P2)."""
    for cell, items in by_cell.items():
        own, other = (T.SPOS, T.SNEG) if cell.startswith("POS/") else (T.SNEG, T.SPOS)
        for tid, text in items:
            if own.get(f"p3:{tid}", text) != text or f"p3:{tid}" in other:
                raise SystemExit("a p3 template id is already registered with another text or polarity")
            own[f"p3:{tid}"] = text


def render_p3(skeleton, by_cell):
    """The SEALED battery: the skeleton's p3 records rendered with their drawn P3 templates (in memory only)."""
    register_p3(by_cell)
    out = []
    for q in skeleton:
        if q["surface"] == "p3":
            q = json.loads(json.dumps(q))
            tid = by_cell[f"{q['p3_pol']}/{q['p3_form']}"][q["p3_j"]][0]
            q["tid"], q["p3_template"] = f"p3:{tid}", tid
            q["input"] = W.render_input(q["rule_ast"], q["state"], q["event"], q["tid"], q["state_order"],
                                        list(q["event"].keys()))
        out.append(q)
    return out


def check_arms_required(tags, manifest):
    """Treatment arms of --arms-required. Refused: malformed or duplicate tags, an arm without exactly seeds 0, 1, 2,
    no N-CFSD control, or a treatment arm without built rows."""
    seeds = defaultdict(set)
    for t in tags:
        m = TAG_RE.fullmatch(t)
        if not m:
            raise SystemExit(f"--arms-required: malformed tag {t!r} (want e.g. N-CFSD_s0, E-R3_s2)")
        seeds[m.group(1)].add(int(m.group(2)))
    if len(set(tags)) != len(tags) or "N-CFSD" not in seeds or any(s != {0, 1, 2} for s in seeds.values()):
        raise SystemExit(f"--arms-required {tags}: every arm needs exactly seeds 0, 1, 2, and N-CFSD is required")
    arms = [x for x in TREATMENTS if x in seeds]
    if [x for x in arms if x not in manifest.get("arms", {})]:
        raise SystemExit(f"--arms-required names arms without built rows in {MANIFEST}")
    return arms


def cmd_mint_commit(a):
    import secrets
    cells = Path(a.out)
    commit_p, salt_p, open_p = sealed_paths(cells)
    if commit_p.exists() or salt_p.exists() or open_p.exists():
        raise SystemExit("sealed commitment already exists; refusing to re-mint")
    tags = [t.strip() for t in a.arms_required.split(",") if t.strip()]
    man_p = cells / MANIFEST
    if not man_p.exists():
        raise SystemExit(f"no {MANIFEST} (build first; it records max_new)")
    man = read_json(man_p)
    branch, _, brec = branch_arms()
    if (man.get("branch") or {}).get("sha256") != brec["sha256"]:
        raise SystemExit(f"{MANIFEST} was not built from this BRANCH_STEP_A.json")
    want = committed_arms(branch, man)
    built = list(man.get("arms", {}))
    if sorted(built) != sorted(want):
        raise SystemExit(f"{MANIFEST} arms {built} != the arms of branch {branch}: {want}")
    drop, reason = (getattr(a, "drop_arm", "") or "").strip(), (getattr(a, "reason", "") or "").strip()
    dropped = []
    if drop or reason:
        if drop not in want or not reason:
            raise SystemExit(f"--drop-arm must name one of the branch's arms {want} and --reason must say why "
                             "(PREREG_E01R section 2: an arm whose rows fail the training-row audit)")
        dropped = [{"arm": drop, "reason": reason}]
    keep = [x for x in want if x != drop]
    arms = check_arms_required(tags, man)
    if arms != keep:
        raise SystemExit(f"--arms-required treatment arms {arms} != {keep} (branch {branch}"
                         f"{'; dropped ' + drop if drop else ''}): each with seeds 0, 1, 2, plus N-CFSD_s0-s2")
    early = [t for t in tags if t.split("_s")[0] in TREATMENTS and (cells / t).exists()]
    if early:
        raise SystemExit(f"the mint precedes treatment training; found {early}")
    rows_sha = {}
    for arm in arms:
        rows_sha[arm] = sha(cells / f"train_{arm}.jsonl")
        if rows_sha[arm] != man["arms"][arm]["sha256"]:
            raise SystemExit(f"train_{arm}.jsonl differs from {MANIFEST}")
        require_audit(cells, arm, rows_sha[arm])
    if sha(HERE / "decide_branch.py") != DECIDE_SHA256:
        raise SystemExit("decide_branch.py differs from its CX D-37 freeze")
    if not ANALYZE_PATH.exists():
        raise SystemExit(f"{ANALYZE_PATH.name} must exist (frozen) before the mint")
    code = code_sha256(strict=True)
    p3 = p3_reference()
    salt = secrets.token_hex(32)
    skel = build_skeleton(NE.sealed_seed(salt), p3["counts"])  # in memory only, never shown
    rec = {"experiment": "RCX-E01R", "sha256_salt": hashlib.sha256(salt.encode()).hexdigest(),
           "battery_commit": NE.battery_commit(salt, NE.battery_sha256(skel)), "skeleton_records": len(skel),
           "n_parents": N_PARENTS, "strata": list(STRATA), "surfaces": list(SURFACES), "forms": list(FORMS),
           "split": "sealed", "world": "arith", "derivation": DERIVATION, "p3_draw": DRAW, "p3": p3,
           "arms_required": tags, "treatment_arms": arms, "branch": brec, "dropped_arms": dropped,
           "max_new": man["max_new"], "train_manifest_sha256": sha(man_p), "rows_sha256": rows_sha,
           "audit_e01r_sha256": sha(cells / AUDIT) if (cells / AUDIT).exists() else None,
           "code_sha256": code, "python": sys.version.split()[0],
           "minted_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    NE._write_once(salt_p, salt)
    NE._write_once(commit_p, json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)
    print(f"record in DECISIONS: sha256_salt {rec['sha256_salt']}; SEALED_COMMIT.json sha256 {sha(commit_p)} "
          f"(open-sealed --expect-commit-sha)", flush=True)


def train_overlap(files, battery):
    """native_e01.sealed_train_overlap over an explicit list of training-row files: recorded and read-back semantic and
    skeleton hashes, and battery status words (other than posted/void) in any training prompt. parent_ids = battery
    parents with any hash overlap (dropped from every arm by the analysis)."""
    tr_sem, tr_skel, tr_tok, info = set(), set(), set(), {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f]
        info[Path(p).name] = {"sha256": sha(p), "rows": len(rows)}
        for r in rows:
            tr_sem.add(r.get("semantic_hash"))
            tr_skel.add(r.get("skeleton_hash"))
            tr_tok |= set(NE._WORD_RE.findall(r["prompt"]))
            try:
                rb = NE.read_back(r)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue  # pooled R1 wordings: W.parse_rule does not know them; their recorded hashes are used
            tr_sem.add(rb["semantic_hash"])
            tr_skel.add(rb["skeleton_hash"])
    tr_sem.discard(None)
    tr_skel.discard(None)
    sem_p, skel_p, words = set(), set(), set()
    for r in battery:
        if {r["semantic_hash"], W.semantic_hash(r["rule_ast"], r["state"], r["event"])} & tr_sem:
            sem_p.add(r["parent_id"])
        if {r["skeleton_hash"], W.skeleton_hash(r["state"], r["event"])} & tr_skel:
            skel_p.add(r["parent_id"])
        sa = W.status_atom(r["rule_ast"])
        words |= ({r["event"]["status"]} | set(sa["set"] if sa else ())) & tr_tok
    return {"train_files": info, "semantic_parents": len(sem_p), "skeleton_parents": len(skel_p),
            "status_words_in_train_prompts": sorted(words - {gen.POSTED, gen.VOID}),
            "parent_ids": sorted(sem_p | skel_p)}


def _code_diff(rec):
    """Bound code files whose sha256 differs from the commitment's (descriptive; named on a skeleton mismatch)."""
    now, then = code_sha256(), rec.get("code_sha256", {})
    return sorted(k for k in set(now) | set(then) if now.get(k) != then.get(k))


def cmd_open_sealed(a):
    """Open the SEALED battery once. Every check that can refuse runs BEFORE the first read of P3 (load_p3): the
    commitment (sha, salt, regenerated skeleton), the manifest, the row audit, the training-row files and their shas,
    the adapters and their smoke records. The descriptive records (R3 gold byte bound, training overlap) are computed
    on the skeleton, which carries every P3-independent field of the battery."""
    cells = Path(a.out)
    commit_p, salt_p, open_p = sealed_paths(cells)
    if not commit_p.exists():
        raise SystemExit("no SEALED_COMMIT.json (mint-commit first)")
    if open_p.exists():
        raise SystemExit("the SEALED battery is already open; SEALED_OPEN.json is write-once")
    if not a.expect_commit_sha or a.expect_commit_sha != sha(commit_p):
        raise SystemExit("--expect-commit-sha must equal the SEALED_COMMIT.json sha256 recorded in DECISIONS")
    rec = read_json(commit_p)
    man_p = cells / MANIFEST
    if not man_p.exists() or sha(man_p) != rec["train_manifest_sha256"]:
        raise SystemExit(f"{MANIFEST} is missing or changed after the mint")
    man = read_json(man_p)
    if man.get("max_new") != rec["max_new"]:
        raise SystemExit("max_new: the manifest differs from the commitment")
    if rec.get("audit_e01r_sha256") is not None and (not (cells / AUDIT).exists()
                                                     or sha(cells / AUDIT) != rec["audit_e01r_sha256"]):
        raise SystemExit(f"{AUDIT} is missing or changed after the mint")
    files = [E01N_CELLS / "train_N-CFSD.jsonl"] + [cells / f"train_{x}.jsonl" for x in rec["treatment_arms"]]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"training-row files missing: {missing}")
    if sha(files[0]) != NCFSD_ROWS_SHA256 or any(
            sha(p) != man["arms"][x]["sha256"] or sha(p) != rec.get("rows_sha256", {}).get(x, sha(p))
            for p, x in zip(files[1:], rec["treatment_arms"])):
        raise SystemExit("a training-row file differs from its CX D-30 / manifest / commitment sha")
    adapters, bad = {}, {}
    for t in rec["arms_required"]:
        info, why = e01n_adapter(t) if t.startswith("N-CFSD") else completed_adapter(cells, t, man)
        if not why and not t.startswith("N-CFSD"):
            why = smoke_problem(cells, t, info["adapter_sha256"], rec["max_new"])
        if why:
            bad[t] = why
        else:
            adapters[t] = info
    if bad:
        raise SystemExit(f"refusing to open the SEALED battery: {bad}")
    if not salt_p.exists():
        raise SystemExit("SEALED_SALT.secret is missing")
    salt = salt_p.read_text(encoding="utf-8").strip()
    if hashlib.sha256(salt.encode()).hexdigest() != rec["sha256_salt"]:
        raise SystemExit("salt does not match the commitment")
    seed = NE.sealed_seed(salt)
    skel = build_skeleton(seed, rec["p3"]["counts"], rec["n_parents"])
    if NE.battery_commit(salt, NE.battery_sha256(skel)) != rec["battery_commit"]:
        raise SystemExit(f"the regenerated skeleton is not the committed one; changed since the mint: "
                         f"{_code_diff(rec)}")
    gold = max(len(TT.trace_target(r).encode("utf-8")) for r in skel) + 1  # P3-independent fields only
    overlap = train_overlap(files, skel)  # hashes, rule, state and event: identical in the rendered battery
    battery = render_p3(skel, load_p3(rec["p3"]))  # the first read of P3 outside the R1 audit (B3 only); last step
    o = {"opened_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "salt": salt, "seed": seed,
         "battery_sha256": NE.battery_sha256(battery), "n_records": len(battery),
         "by_surface": dict(Counter(r["surface"] for r in battery)), "sealed_commit_sha256": sha(commit_p),
         "battery_commit_checked": True, "p3_sha256": rec["p3"]["sha256"],
         "adapters_sha256": {t: v["adapter_sha256"] for t, v in adapters.items()}, "adapters": adapters,
         "train_overlap": overlap, "max_new": rec["max_new"],
         "r3_gold_utf8_bytes_plus_eos": gold, "max_new_covers_r3_gold_byte_bound": gold <= rec["max_new"],
         "code_sha256": code_sha256(), "code_changed_since_mint": _code_diff(rec)}
    NE._write_once(open_p, json.dumps(o, indent=1))
    print(json.dumps({k: v for k, v in o.items() if k not in ("adapters", "code_sha256")}, indent=1), flush=True)
    if o["train_overlap"]["parent_ids"]:
        print(f"WARNING: {len(o['train_overlap']['parent_ids'])} SEALED parents overlap training rows; they are "
              "dropped from every arm", flush=True)
    if o["code_changed_since_mint"]:
        print(f"NOTE: bound code changed since the mint: {o['code_changed_since_mint']}", flush=True)


def sealed_battery(out):
    """(battery, seed, commit record, open record) of the opened SEALED battery, rebuilt in memory and checked
    against SEALED_COMMIT.json and SEALED_OPEN.json; SystemExit before opening or on any mismatch."""
    commit_p, _, open_p = sealed_paths(out)
    if not open_p.exists():
        raise SystemExit("SEALED battery is not open (run open-sealed after all required adapters exist)")
    rec, o = read_json(commit_p), read_json(open_p)
    if hashlib.sha256(o["salt"].encode()).hexdigest() != rec["sha256_salt"]:
        raise SystemExit("SEALED_OPEN.json salt does not match the commitment")
    if o["seed"] != NE.sealed_seed(o["salt"]):
        raise SystemExit("SEALED_OPEN.json seed is not the committed derivation of the salt")
    if o.get("sealed_commit_sha256") != sha(commit_p):
        raise SystemExit("SEALED_COMMIT.json changed after the battery was opened")
    skel = build_skeleton(o["seed"], rec["p3"]["counts"], rec["n_parents"])
    if NE.battery_commit(o["salt"], NE.battery_sha256(skel)) != rec["battery_commit"]:
        raise SystemExit(f"the regenerated skeleton differs from the committed one: {_code_diff(rec)}")
    bat = render_p3(skel, load_p3(rec["p3"]))
    if NE.battery_sha256(bat) != o["battery_sha256"]:
        raise SystemExit("the rendered SEALED battery differs from the one recorded at opening")
    return bat, o["seed"], rec, o


# ----------------------------------------------------------------------------- eval (SEALED only)

def eval_adapter(cells, arm, tag, opened):
    """(adapter dir or None, sha or None) for a SEALED eval. A committed arm (N-CFSD, E-R*) is evaluated only with the
    adapter fixed at opening; N-VR (descriptive) only as a CX D-30 adapter; OFF has none."""
    if arm == "OFF":
        return None, None
    if arm == "N-VR":
        info, why = e01n_adapter(tag)
        if why:
            raise SystemExit(f"{tag}: {why}")
        return Path(info["adapter_dir"]), info["adapter_sha256"]
    if arm != "N-CFSD" and arm not in TREATMENTS:
        raise SystemExit(f"unknown arm {arm}")
    want = opened["adapters_sha256"].get(tag)
    if want is None:
        raise SystemExit(f"{tag} is not among the adapters fixed by open-sealed {sorted(opened['adapters_sha256'])}; "
                         "no treatment is evaluated, and no committed cell is added or replaced, outside them")
    d = (E01N_CELLS if arm == "N-CFSD" else Path(cells)) / tag / "adapter"
    got = sha(d / "adapter_model.safetensors")
    if got != want:
        raise SystemExit(f"{tag}: adapter sha {got[:12]} != {want[:12]} recorded by open-sealed")
    return d, got


def raw_path(cells, tag):
    return Path(cells) / f"raw_sealed_{tag}.json.gz"


def write_raw(p, raw):
    """The raw generations of a SEALED eval, create-only (gzip 'xt'), written right after decoding and before any
    post-processing that can raise (truncation mapping, scoring, trace diagnostics)."""
    with gzip.open(p, "xt", encoding="utf-8") as f:
        f.write(json.dumps(raw))


def read_raw(p, want):
    """A raw-generations file, refused unless generated for exactly `want` (tag, battery, adapter, max_new, n)."""
    with gzip.open(p, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    bad = sorted(k for k, v in want.items() if raw.get(k) != v)
    if bad:
        raise SystemExit(f"{Path(p).name} was generated for another {bad}; refusing to post-process it")
    return raw


def score_outputs(battery, raw, arm, parse):
    """(finished, classifications, prediction rows) from raw generations. Pure (no model, no tokenizer): the truncation
    mapping (map_finished over the recorded prompt token lengths and decode log), then score.classify of each output as
    parsed by the arm's frozen parser (E-R3: the last non-empty line), plus the R3 line-1 diagnostics."""
    outs = raw["outs"]
    if len(outs) != len(battery):
        raise RuntimeError(f"{len(outs)} raw outputs for {len(battery)} battery records")
    fin = map_finished(raw["prompt_tokens"], raw["decode_calls"], outs)
    res = [S.classify(r, parse(o, list(r["gold"]["after"].keys()))) for r, o in zip(battery, outs)]
    rows = []
    for i, (r, o, x, fi) in enumerate(zip(battery, outs, res, fin)):
        row = {"i": i, "out": o, "finished": fi, **x}
        if arm == "E-R3":
            row["trace"] = TT.predicate_consistency(r, o)
        rows.append(row)
    return fin, res, rows


def cmd_eval(a):
    cells = Path(a.out)
    battery, bseed, rec, opened = sealed_battery(cells)
    tag = "OFF" if a.arm == "OFF" else f"{a.arm}_s{a.seed}"
    res_path = cells / f"eval_sealed_{tag}.json"
    if res_path.exists():
        print("eval exists", flush=True)
        return
    adapter_dir, adapter_sha = eval_adapter(cells, a.arm, tag, opened)
    if sha(cells / MANIFEST) != rec["train_manifest_sha256"] or opened.get("max_new") != rec["max_new"]:
        raise SystemExit("max_new: the manifest or SEALED_OPEN.json differs from the commitment")
    max_new = rec["max_new"]
    parse, parser = answer_parser(a.arm)
    lock = require_lane()
    raw_p = raw_path(cells, tag)
    want = {"tag": tag, "battery_sha256": NE.battery_sha256(battery), "adapter_sha256": adapter_sha,
            "max_new": max_new, "n": len(battery)}
    if raw_p.exists():  # decoding finished earlier but post-processing did not: never decode again
        raw = read_raw(raw_p, want)
        print(f"[{tag}] {raw_p.name} exists; post-processing it without loading the model", flush=True)
    else:
        NP, LC, ThermalGuard = _heavy()
        NP.need_ram()
        model, tok = LC.load_core("sparse")
        if adapter_dir is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False).eval()

        def render(system, user):
            return tok.apply_chat_template([{"role": "system", "content": system},
                                            {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)

        prompts = [render(NPP.rcl_system(T.render_rule(r["rule_ast"], r["tid"])), NPP.user_json(r)) for r in battery]
        probe = EosProbe(tok, LC.EOS_IDS)
        lens = probe.prompt_lengths(prompts)
        guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
        t0 = time.time()
        outs = LC.generate(model, probe, prompts, max_new=max_new, guard=guard, log=lambda m: print(m, flush=True))
        raw = dict(want, arm=a.arm, outs=list(outs), prompt_tokens=lens,
                   decode_calls=[[t, bool(e)] for t, e in probe.calls], wall_s=round(time.time() - t0, 1),
                   generated_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), lane=lock)
        write_raw(raw_p, raw)  # create-only, BEFORE the truncation mapping and scoring
    fin, res, rows = score_outputs(battery, raw, a.arm, parse)
    summ = S.aggregate(battery, res)
    out_rec = {"arm": a.arm, "seed": None if a.arm == "OFF" else a.seed, "tag": tag, "battery": "sealed",
               "battery_seed": bseed, "battery_sha256": want["battery_sha256"], "adapter_sha256": adapter_sha,
               "adapter_dir": None if adapter_dir is None else str(adapter_dir), "n": len(battery),
               "surfaces": list(SURFACES), "max_new": max_new, "parser": parser, "truncated": fin.count(False),
               "wall_s": raw["wall_s"], "composite": {s: S.composite(summ, s) for s in SURFACES},
               "composite_decision": {s: S.composite(summ, s, field="decision") for s in SURFACES}, "cells": summ,
               "raw_file": raw_p.name, "raw_sha256": sha(raw_p), "sealed_open_sha256": sha(sealed_paths(cells)[2]),
               "code_sha256": code_sha256(), "lane": lock}
    preds_p = cells / f"preds_sealed_{tag}.jsonl.gz"
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if preds_p.exists():  # an earlier post-processing wrote it, then stopped: it must be this exact content
        with gzip.open(preds_p, "rt", encoding="utf-8") as f:
            if f.read() != text:
                raise SystemExit(f"{preds_p.name} exists with other content")
    else:
        with gzip.open(preds_p, "xt", encoding="utf-8") as f:  # create-only
            f.write(text)
    NE._write_once(res_path, json.dumps(out_rec, indent=1))
    print(f"[{tag}] composite {out_rec['composite']} decision {out_rec['composite_decision']} truncated "
          f"{out_rec['truncated']} {out_rec['wall_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="RCX-E01R native cells (see the module docstring)")
    ap.add_argument("cmd", choices=["build", "train", "smoke", "mint-commit", "open-sealed", "eval"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--arms", default="", help="build: the treatment arms of BRANCH_STEP_A.json, or none (B1)")
    ap.add_argument("--r1-fallback", dest="r1_fallback", action="store_true",
                    help="build under B3 with E-R2,E-R3 when the R1 pool is below its minimum (PREREG section 2)")
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery", default="sealed", choices=["sealed"])
    ap.add_argument("--arms-required", dest="arms_required", default="",
                    help="mint-commit: N-CFSD_s0,N-CFSD_s1,N-CFSD_s2 plus each treatment arm's s0-s2")
    ap.add_argument("--drop-arm", dest="drop_arm", default="",
                    help="mint-commit: drop this branch arm (PREREG section 2: its rows fail the row audit); "
                         "needs --reason, recorded in SEALED_COMMIT.json")
    ap.add_argument("--reason", default="", help="mint-commit: why --drop-arm drops the arm")
    ap.add_argument("--expect-commit-sha", dest="expect_commit_sha", default="",
                    help="open-sealed: SEALED_COMMIT.json sha256 as recorded in DECISIONS (mandatory)")
    a = ap.parse_args()
    {"build": cmd_build, "train": cmd_train, "smoke": cmd_smoke, "mint-commit": cmd_mint_commit,
     "open-sealed": cmd_open_sealed, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
