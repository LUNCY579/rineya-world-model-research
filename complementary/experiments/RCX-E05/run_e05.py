"""RCX-E05 runner (PREREG_E05_DRAFT.md §2 custody, §9 pilot gates and fill, §10 order, lane and VOID, binding; DESIGN.md
rev 3 §9-§10). Commands, from complementary/, in the §10.1 order:

  code-digest [--stage t2|t3]                 §2.1 digest (reads only; prints JSON with test-file and reference shas)
  pilot-a  --expect-code-digest <T2>          Phase 1a: draw TRAIN/DEV -> pilot/PILOT_MANIFEST.json (create-only), the
                                              library tables, Q1-Q3, Q5, Q6, Q8, the reference arms on DEV (Q7, Q9,
                                              G-P2, then Q4), selection training (seed 20261245), DEV evaluation and
                                              banks, G-P4 (+ the one UVFA extension, the 800-label fallback)
  pilot-b  --expect-code-digest <T2>          Phase 1b: PGV_400/FT grids (G-P3 + the one extension), the k-curve on the
                                              8 DEV-ID parents (G-P3b + R_ext), the selected PG400 on the 24 DEV-OOD
                                              H-K/H-NEG parents (M5), the G-P5 rates
  pilot-report --expect-code-digest <T2>      pilot/PILOT_REPORT.json (create-only; no lane, reads outputs only): G-P1 to
                                              G-P5 (Q7, Q9), sigma-hats, §8 sizing and powers, §10.3 cuts, the analysis
                                              determinations and the §9.2 FILL rows
  draw-test --attempt k --expect-prereg-sha <T3> --expect-code-digest <T3>   confirm/attempt-k/TEST_MANIFEST.json
  oracle-test --attempt k --manifest-sha <T4> --code-digest <T3>             §10.4 VOID checks -> ORACLE_TEST_REPORT.json
  train --expect-prereg-sha <T3> --expect-code-digest <T3>   replicates 20261246-48 of SF, UVFA, UVFA+aux; SF-shuffled
  pgv | ft | eval | extras --attempt k --manifest-sha <T4> --code-digest <T3>
      pgv: PGV_400 on every primary parent (G-P3 config), then the k-curve on the 48-parent subset; ft: FT-UVFA_400 per
      replicate; eval: every non-extra arm on every retained family plus the one-step banks, then evals/*.json;
      extras (descriptive, cut items 1-2): the budget curve and SF-FT_400, then their evals/*.json.

Exit codes of lane phases: 0 = complete; 75 = yielded at a safe point or slice used up, work left (re-run the same
command: it resumes from its checkpoints and keyed outputs); 3 = the lane start condition does not hold (nothing
written). Custody refusals exit with a message and write nothing.

Custody (§2):
  * code digest = sha256 over the sorted lines "<relpath>\\t<sha256>\\n" (relpaths from complementary/) of the new
    outcome sources (T2: soft.py, families_e05, enum_e05, learners_e05, run_e05; T3: + analyze_e05) and the pinned
    files (PINNED; their 8-hex prefixes are checked, a mismatch refuses every phase). lane_e05.py, rcx/lane.py and
    rcx/guard.py are operational (M1): outside the digest, their shas recorded in every provenance record.
  * pilot phases refuse unless --expect-code-digest equals the current T2 digest and occurs in the COMMITTED
    DECISIONS.md; every keyed output carries the digest, and a phase refuses if its directory holds outputs of another
    digest (§11: re-run the affected pilot phase in a fresh --pilot-dir, logged).
  * committed-sha refusal (L5): oracle-test, pgv, ft, eval, extras refuse unless --manifest-sha is a full 64-hex sha256
    occurring in `git show HEAD:./DECISIONS.md` (never the working copy); draw-test and train need the T3 prereg sha and
    T3 digest committed likewise. pgv/ft/eval/extras also need the attempt's ORACLE_TEST_REPORT.json = PASS (B4) and log
    "first-learned-test" in ATTEMPTS.jsonl before their first unit; oracle-test refuses once that event exists (VOID is
    declared only inside oracle-test, before any learned-arm TEST computation). In oracle-test a committed manifest sha
    that the file no longer matches, or a committed T3 digest the code no longer matches, is recorded as VOID (custody
    fault between T4 and PASS); an uncommitted or malformed argument is a refusal (nothing written).
  * outputs: manifests and reports create-only (atomic hard link); per-cell results in append-only keyed JSONL (a key is
    written at most once; completed keys are never recomputed); runs/RCX-E05/ATTEMPTS.jsonl (slices, manifests,
    oracle-test results, first learned-arm TEST computation); runs/RCX-E05/PROVENANCE.jsonl (digest, file shas,
    reference shas, test shas, seeds, versions, threads, arguments) for every slice.

Lane (§10.2): every lane phase runs in slices through lane_e05.Session (party rcx-e05, try_acquire only). A slice's
length is L = lane_e05.slice_budget_min (<= 15 min and, in day mode, ending <= 10 min before the next window; in
night-tail mode <= W.end - 5 min); it ends at the first safe point after L, on any yield condition, or when the work is
done (the projected work of the visible units is logged). Plans may be staged: the driver runs the pending units in
order and re-plans when they are used up. Safe points: before every unit and after every evaluation cell or optimizer
step (a lane check once >= 60 s have passed since the previous one), and at epoch and BatchedMLP-batch ends (a forced
check). One-off pilot units without internal safe points (draw, library tables, Q checks, training-data tables, the
G-P5 bench) can exceed 30 s at n = 7 (a disclosed reading of M2, whose unit list names optimizer steps and evaluation
cells).

Adapters (class Adapters): the ONLY call sites of the two modules written in parallel, through the contract they
document: enum_e05.enum_policy(space, w, c, kind, stop, edge_mask, phi) -> (policy, fresh per-call diag with
"decisions" (>= 2 legal moves) and "all_tied"); learners_e05.train_zero_shot / fit_pergoal / load_model / predict_psi /
predict_value / time_step (see learners_e05's docstring). Tests substitute a fake (Ctx.adapters).

Readings taken where the prereg leaves a detail open (listed for the coordinator):
  * H-CONJ: every arm except ORACLE receives the linearization w = (3/|G|) 1_G and scores STOP as w.phi(s) (§3 "every
    arm scores STOP exactly as w'.phi(s) and receives w'"; DESIGN §3 lists "H-CONJ's linearization" among GREEDY's
    positive goals); ORACLE, the starts, the banks and every nreg use the true reward 3 1[G <= s].
  * RAND: one stream per cell, 20261244:rand:<split>:<parent key>:<s0> (= enum_e05.rand_rng).
  * Per-goal seeds (one integer per job; learners derive order and stabilizer streams from it): pilot (selection)
    20261245:<PGV|FT>:<grid>:<k>:<key>; TEST PGV 20261249:pgv-init:<key>:<k>; FT <replicate>:ft:<key>; SF-FT
    20261249:sf-ft:<key>. Each grid point is its own fit; a fit stops at its horizon (the same result as that checkpoint
    of the longer run: constant lr, early-stopping checks <= the horizon).
  * Zero-shot epochs: the 20-epoch model is the 20-epoch snapshot of the 40-epoch run; a confirmation replicate trains
    exactly the selected epochs.
  * SF-FT_400 is scored with its own code only (C = {w'}); the budget-curve SF-GPI uses its own 16 / 64 training goals
    as the library (TRAIN goals only). Q2 runs at n = 6 (DESIGN App. B). Q4 runs on free-dynamics parents (library
    policies are not height-3 policies). Q8's n = 7 assertion reads the files listed in rcx_test_sets_n7. R_ext k-curve
    arms are named PGV-RX_<k> (analyze_e05's name).
  * Seeds and stream tags beyond families_e05: Q checks 20261242:q-checks:<Q>; base-policy labels
    20261243:train-base:<fact>.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist, stdev
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

RUNS = ROOT / "runs" / "RCX-E05"
EXIT_REQUEUE, EXIT_REFUSED = 75, 3


def _load(name: str):
    """Import an experiments/RCX-E05 module by path, registered in sys.modules first (dataclasses need it)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


FA = _load("families_e05")
LA = _load("lane_e05")
SEEDS = FA.SEEDS
BAR, TOL, WD = 0.50, 1e-9, 1e-4
REF_PILOT = ("EXACT-GPI(L)", "ORACLE", "STOP", "RAND", "GREEDY", "ENUM-MIS±")
ZS_PILOT = ("SF-GPI", "SF-noGPI", "SF-L", "UVFA", "UVFA+aux")
ZS_TEST = ("SF-GPI", "SF-noGPI", "SF-L", "SF-GPI-5", "UVFA", "UVFA+aux")
ARM_MODEL = {"SF-GPI": "SF", "SF-noGPI": "SF", "SF-L": "SF", "SF-GPI-5": "SF", "SF-shuffled": "SF-shuffled",
             "UVFA": "UVFA", "UVFA+aux": "UVFA+aux"}
SELECT_BY = {"SF": "SF-GPI", "UVFA": "UVFA", "UVFA+aux": "UVFA+aux"}
POSITIVE_DEV = ("DEV-ID", "DEV-OOD:H-K", "DEV-OOD:H-COST")
POSITIVE_TEST = ("H-ID", "H-K", "H-COST", "H-DYN", "H-CONJ")
ENUM_KIND = {"ENUM-MIS±": "MIS", "ENUM-H*±": "H*", "GNN-ENUM±": "GNN", "ENUM-MIS (rev 1)": "MIS-rev1",
             "ENUM-MIS± (rev 2)": "MIS-rev2"}

# Per-goal grid points (§5.2-5.3, fixed): name -> (kind, learners_e05.fit_pergoal cfg). Each is its own fit; `keep` is
# the early-stopping horizon (steps <= keep are run).
_PGV, _FT = dict(lr=1e-3, batch=64, eval_every=100, val_n=40, wd=1e-4), dict(lr=3e-4, batch=64, eval_every=25, val_n=40,
                                                                            step0=True, wd=1e-4)
_KC = dict(lr=1e-3, eval_every=100, val_frac=0.1, wd=1e-4)
GRIDS = {
    **{f"w{w}s{s}": ("PGV", dict(_PGV, width=w, steps=s, keep=s)) for w in (64, 256) for s in (500, 2000)},
    "w128s4000": ("PGV", dict(_PGV, width=128, steps=4000, keep=4000, lr=3e-4)),
    "s100": ("FT", dict(_FT, steps=100, keep=100)), "s400": ("FT", dict(_FT, steps=400, keep=400)),
    "s1000": ("FT", dict(_FT, steps=1000, keep=1000, lr=1e-4)),
    "k1600": ("PGV", dict(_KC, width=256, batch=128, steps=2000, keep=2000)),
    "k6400": ("PGV", dict(_KC, width=256, batch=256, steps=2500, keep=2500)),
    **{f"rext{k}": ("PGV", dict(_KC, width=128, batch=b, steps=4000, keep=4000, lr=3e-4, eval_every=200))
       for k, b in ((400, 64), (1600, 128), (6400, 256))},
}
PGV_GRID, PGV_EXT, FT_GRID, FT_EXT = ("w64s500", "w64s2000", "w256s500", "w256s2000"), ("w128s4000",), ("s100", "s400"), \
    ("s1000",)
KCURVE = {1600: "k1600", 6400: "k6400"}
R_EXT = {400: "rext400", 1600: "rext1600", 6400: "rext6400"}
ZS_BASE = dict(epochs=40, lr=1e-3, keep=(20, 40))
ZS_EXT = dict(epochs=80, lr=3e-4, keep=(80,))
TRAIN_LABELS_MAX, BASE_LABELS_MAX, DEV_PERGOAL = 1600, 800, 400


def ge(x, t):
    return x >= t - TOL


def le(x, t):
    return x <= t + TOL


def lt(x, t):
    return x < t - TOL


def gt(x, t):
    return x > t + TOL


def kj(*parts) -> str:
    """Keyed-output key: canonical JSON of its parts."""
    return json.dumps([p if isinstance(p, (str, int, float, bool)) or p is None else str(p) for p in parts])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fname(key: str) -> str:
    """File-system-safe short name for a parent key or arm name."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ============================================================================= custody (§2)

NEW_T2 = ("rcx/bw/soft.py", "experiments/RCX-E05/families_e05.py", "experiments/RCX-E05/enum_e05.py",
          "experiments/RCX-E05/learners_e05.py", "experiments/RCX-E05/run_e05.py")
NEW_T3 = NEW_T2 + ("experiments/RCX-E05/analyze_e05.py",)
PINNED = {"rcx/bw/space.py": "29d0b420", "rcx/bw/bwm_import.py": "84140c7e", "rcx/bw/excl.py": "eb31d9eb",
          "experiments/RCX-E07/EXCLUSIONS.json": "2f211de3", "rcx/bw/models.py": "4c2e3242",
          "rcx/bw/encode.py": "32980f14", "runs/RCX-E07/main/models/gnn_s20261276.pt": "9cc417a3",
          "rcx/stats.py": "06a30c10"}
TEST_FILES = ("tests/test_e05_oracle.py", "tests/test_e05_families.py", "tests/test_e05_enum.py",
              "tests/test_e05_learners.py", "tests/test_e05_analyze.py", "tests/test_e05_lane.py",
              "tests/test_e05_run.py")


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_digest(stage: str = "t2", root: Path = ROOT) -> dict:
    """{"stage", "digest" (None if a file is missing), "files", "missing", "pins_bad"} (§2.1)."""
    if stage not in ("t2", "t3"):
        raise ValueError(stage)
    rels = sorted((NEW_T3 if stage == "t3" else NEW_T2) + tuple(PINNED))
    files, missing = {}, []
    for rel in rels:
        p = Path(root) / rel
        if p.exists():
            files[rel] = sha_file(p)
        else:
            missing.append(rel)
    text = "".join(f"{rel}\t{files[rel]}\n" for rel in sorted(files))
    pins_bad = {rel: (files.get(rel, "")[:8], pre) for rel, pre in PINNED.items()
                if not files.get(rel, "").startswith(pre)}
    return {"stage": stage, "digest": None if missing else sha_bytes(text.encode("utf-8")), "files": files,
            "missing": missing, "pins_bad": pins_bad}


def test_shas(root: Path = ROOT) -> dict:
    return {rel: (sha_file(Path(root) / rel) if (Path(root) / rel).exists() else None) for rel in TEST_FILES}


def committed_decisions(root: Path = ROOT) -> str:
    """The committed DECISIONS.md (`git -C <complementary> show HEAD:./DECISIONS.md`), never the working copy."""
    r = subprocess.run(["git", "-C", str(root), "show", "HEAD:./DECISIONS.md"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"refusing: cannot read the committed DECISIONS.md ({r.stderr.decode(errors='replace')[:200]})")
    return r.stdout.decode("utf-8", errors="replace")


HEX64 = re.compile(r"[0-9a-f]{64}")


def require_committed(value: str, what: str, text: str) -> None:
    if not HEX64.fullmatch(value or ""):
        raise SystemExit(f"refusing: {what} must be a full lower-case 64-hex sha256 (got {value!r})")
    if value not in text:
        raise SystemExit(f"refusing: {what} {value[:12]}... does not occur in the committed DECISIONS.md "
                         "(git show HEAD:./DECISIONS.md; PREREG §2.2 L5)")


def require_digest(ctx, stage: str, expect: str) -> dict:
    """expect is committed in DECISIONS, every file is present, the pins hold and the current digest equals expect."""
    require_committed(expect, f"{stage.upper()} code digest", ctx.decisions())
    cd = code_digest(stage, ctx.root)
    if cd["missing"]:
        raise SystemExit(f"refusing: {stage.upper()} digest files missing: {cd['missing']}")
    if cd["pins_bad"]:
        raise SystemExit(f"refusing: pinned files changed (sha prefix now, pinned): {cd['pins_bad']}")
    if cd["digest"] != expect:
        raise SystemExit(f"refusing: current {stage.upper()} code digest {cd['digest'][:12]}... != expected "
                         f"{expect[:12]}...")
    return cd


def require_prereg(ctx, expect: str) -> str:
    require_committed(expect, "PREREG_E05.md sha256", ctx.decisions())
    p = Path(ctx.prereg)
    if not p.exists():
        raise SystemExit("refusing: PREREG_E05.md (T3 freeze) does not exist")
    sha = sha_file(p)
    if sha != expect:
        raise SystemExit(f"refusing: PREREG_E05.md sha256 {sha[:12]}... != --expect-prereg-sha {expect[:12]}...")
    return sha


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, default=_jdefault, ensure_ascii=True) + "\n").encode("utf-8")


def _atomic_link(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)  # fails if path exists: never overwrites
    finally:
        tmp.unlink(missing_ok=True)


def write_create_only(path, obj) -> str:
    """Write JSON atomically and create-only (FileExistsError if present); returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(obj)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _atomic_link(tmp, path)
    return sha_bytes(data)


def save_npz_create_only(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}.npz")
    np.savez(tmp, **arrays)
    _atomic_link(tmp, path)


def _drop_torn_tail(path: Path) -> None:
    """A last line without its newline is a write that a crash interrupted (never read as a record): cut it off."""
    size = path.stat().st_size
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        pos = size - 1
        while pos > 0:
            f.seek(max(0, pos - 4096))
            chunk = f.read(pos - max(0, pos - 4096))
            i = chunk.rfind(b"\n")
            if i >= 0:
                pos = max(0, pos - 4096) + i + 1
                break
            pos = max(0, pos - 4096)
        f.truncate(pos)


def _append_lines(path, recs) -> None:
    """Append JSON lines and fsync once (a torn last line from a crash is cut off first)."""
    if not recs:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        _drop_torn_tail(path)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(r, sort_keys=True, default=_jdefault) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())


def _append_line(path, rec) -> None:
    _append_lines(path, [rec])


def _read_lines(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out, lines = [], path.read_text(encoding="utf-8").splitlines()
    for k, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            if k == len(lines):  # torn last line (crash): ignored
                continue
            raise SystemExit(f"{path.name} line {k} is not JSON")
    return out


class KeyedLog:
    """Append-only JSONL of records {"key": ..., ...}: a key is written at most once (keyed outputs, §2.2)."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        for r in _read_lines(self.path):
            if r.get("key") in self.rows:
                raise SystemExit(f"{self.path.name}: duplicate key {r.get('key')}")
            self.rows[r.get("key")] = r

    def __contains__(self, key):
        return key in self.rows

    def get(self, key, default=None):
        return self.rows.get(key, default)

    def add_many(self, items, digest: str | None = None) -> None:
        """items = [(key, record)]; one write and one fsync."""
        at, recs, keys = time.strftime("%Y-%m-%d %H:%M:%S"), [], set()
        for key, rec in items:
            if key in self.rows or key in keys:
                raise RuntimeError(f"keyed output {key} already written (append-only)")
            keys.add(key)
            recs.append(dict(rec, key=key, digest=digest, at=at))
        _append_lines(self.path, recs)
        for r in recs:
            self.rows[r["key"]] = r

    def add(self, key, rec: dict, digest: str | None = None) -> dict:
        self.add_many([(key, rec)], digest)
        return self.rows[key]

    def digests(self) -> set:
        return {r.get("digest") for r in self.rows.values()}


def attempts_path(ctx) -> Path:
    return ctx.runs / "ATTEMPTS.jsonl"


def log_attempt(ctx, event: str, **info) -> None:
    _append_line(attempts_path(ctx), dict(event=event, at=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(), **info))


def attempt_events(ctx, attempt=None) -> list:
    evs = _read_lines(attempts_path(ctx))
    return [e for e in evs if attempt is None or e.get("attempt") == attempt]


def provenance(ctx, phase: str, cd: dict | None, **extra) -> dict:
    vers = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    if "torch" in sys.modules:
        vers["torch"] = sys.modules["torch"].__version__
    threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                              "CUDA_VISIBLE_DEVICES")}
    return dict(phase=phase, digest_stage=(cd or {}).get("stage"), code_digest=(cd or {}).get("digest"),
                code_files=(cd or {}).get("files"), reference_shas=LA.reference_shas(), test_shas=test_shas(ctx.root),
                seeds=SEEDS, versions=vers, threads=threads, n=ctx.n, **extra)


# ============================================================================= context and oracle data

def default_exclusions(space):
    from rcx.bw import excl as EX
    return FA.Exclusions(space, EX.with_selection_sets(EX.load_exclusions()))


@dataclass
class Ctx:
    runs: Path = RUNS
    root: Path = ROOT
    n: int = 7
    pilot_dir: str = "pilot"
    env: object = None
    decisions: Callable[[], str] = committed_decisions
    exclusions: Callable = default_exclusions
    adapters: object = None
    prereg: Path = HERE / "PREREG_E05.md"
    sizes: dict | None = None  # families_e05.SIZES overrides (tests only)
    bench_override: dict | None = None  # G-P5 rates replacing the measured ones in pilot-report (tests only)
    q_sizes: dict = field(default_factory=lambda: dict(q1=200, q2_goals=20, q2_states=10, q2_n=6, q3=1000, q5=50))
    cost_scale: float = 1.0

    def __post_init__(self):
        self.runs, self.root = Path(self.runs), Path(self.root)
        if self.env is None:
            self.env = LA.Env(paths=LA.Paths(events=self.runs / "LANE_EVENTS.jsonl"))
        if self.adapters is None:
            self.adapters = Adapters()

    @property
    def pdir(self) -> Path:
        return self.runs / self.pilot_dir

    def adir(self, k: int) -> Path:
        return self.runs / "confirm" / f"attempt-{int(k)}"

    def sz(self) -> dict:
        return dict(FA.SIZES, **(self.sizes or {}))


def flat_parents(manifest) -> list:
    ps = manifest["parents"]
    return [x for v in ps.values() for x in v] if isinstance(ps, dict) else list(ps)


class Data:
    """Oracle data for one manifest (pilot or TEST): space, phi, parents by family (manifest order), library and a small
    LRU of per-parent tables (evaluation units run parent-major)."""

    LRU = 8

    def __init__(self, ctx, manifest: dict, lib_dir: Path):
        self.ctx, self.m, self.lib_dir = ctx, manifest, Path(lib_dir)
        self.n = int(manifest["n"])
        self.space = SP.get_space(self.n)
        self.phi = SO.get_phi(self.space)
        self.fam = {}
        for d in flat_parents(manifest):
            p = FA.Parent.from_json(d)
            self.fam.setdefault(p.family, []).append(p)
        self.by_key = {p.key: p for ps in self.fam.values() for p in ps}
        self._lru = OrderedDict()
        self._lib = {}

    def _slot(self, p) -> dict:
        s = self._lru.get(p.key)
        if s is None:
            s = self._lru[p.key] = {}
            while len(self._lru) > self.LRU:
                self._lru.popitem(last=False)
        self._lru.move_to_end(p.key)
        return s

    def table(self, p) -> SO.SFTable:
        s = self._slot(p)
        if "tab" not in s:
            s["tab"] = FA.parent_table(self.space, self.phi, p)
        return s["tab"]

    def arm_stop(self, p) -> np.ndarray:
        """STOP values of the arms that receive w' (H-CONJ: the linearization; module docstring)."""
        if p.rtype != "conj":
            return self.table(p).stop
        s = self._slot(p)
        if "lin" not in s:
            s["lin"] = SO.stop_values(self.phi, p.weights())
        return s["lin"]

    def cache(self, p, name, make):
        s = self._slot(p)
        if name not in s:
            s[name] = make()
        return s[name]

    def mask(self, p):
        return FA.edge_mask(self.space, p.dyn)

    def library(self, base: bool = False) -> SimpleNamespace:
        """L = the TRAIN goals (+ the 49 single-fact base policies if base): W [m, F], c [m], end/T/nxt [m, N]."""
        if base not in self._lib:
            parts = [self._lib_part("train")] + ([self._lib_part("base")] if base else [])
            self._lib[base] = SimpleNamespace(**{k: np.concatenate([getattr(x, k) for x in parts]) for k in
                                                 ("W", "c", "end", "T", "nxt")})
        return self._lib[base]

    def _lib_part(self, part: str) -> SimpleNamespace:
        z = np.load(self.lib_dir / f"oracle_library_{part}.npz")
        return SimpleNamespace(W=z["W"], c=z["c"], end=z["end"], T=z["T"], nxt=z["nxt"])

    def lib_goals(self, part: str) -> list:
        """(w, c) of the library goals of `part` in library order (base: on/ontable single facts, c = 0.25)."""
        if part == "train":
            return [(np.asarray(p.w, dtype=np.float64), p.c) for p in self.fam["TRAIN"]]
        F = self.n * (self.n + 1)
        return [(np.eye(F)[k], FA.C_TRAIN) for k in range(self.n * self.n)]


def lib_tag(base: bool) -> str:
    return "L+base" if base else "L"


def build_library(d: Data, part: str, path: Path) -> None:
    goals = d.lib_goals(part)
    tabs = [SO.sf_table(d.space, SO.stop_values(d.phi, w), c) for w, c in goals]
    save_npz_create_only(path, W=np.stack([w for w, _ in goals]), c=np.array([c for _, c in goals]),
                         end=np.stack([t.end for t in tabs]), T=np.stack([t.T for t in tabs]),
                         nxt=np.stack([t.nxt for t in tabs]))


def tabs_of(lib) -> list:
    return [SimpleNamespace(end=lib.end[i], T=lib.T[i]) for i in range(len(lib.c))]


def train_rows(d: Data, goals, idxs) -> dict:
    """Exact zero-shot labels (learners_e05.train_zero_shot's data): rows of states, goal id, the goal's W and c,
    psi = (phi(end), T) and V* = w-bar.psi (asserted, Q3), for goals [(w, c)] (free dynamics) and label index lists."""
    S, G, W, C, Ps, Vs = [], [], [], [], [], []
    for g, ((w, c), idx) in enumerate(zip(goals, idxs)):
        tab = SO.sf_table(d.space, SO.stop_values(d.phi, w), c)
        idx = np.asarray(idx, dtype=np.int64)
        ps = SO.psi(d.phi, tab, idx)
        if not np.array_equal(SO.wbar_dot(ps, w, c), tab.V[idx]):
            raise AssertionError("V* != w-bar.psi (Q3)")
        S.append(idx)
        G.append(np.full(len(idx), g, dtype=np.int64))
        W.append(np.repeat(np.asarray(w, dtype=np.float64)[None], len(idx), axis=0))
        C.append(np.full(len(idx), float(c)))
        Ps.append(ps)
        Vs.append(tab.V[idx])
    return dict(states=np.concatenate(S), goal=np.concatenate(G), W=np.concatenate(W), c=np.concatenate(C),
                psi=np.concatenate(Ps), v=np.concatenate(Vs))


# ============================================================================= adapters (enum_e05, learners_e05)

class Adapters:
    """The ONLY call sites of enum_e05 and learners_e05 (module docstring; their documented run_e05 contract). Tests
    substitute an object with these methods (Ctx.adapters)."""

    @staticmethod
    def EN():
        return _load("enum_e05")

    @staticmethod
    def LE():
        return _load("learners_e05")

    def enum_policy(self, d: Data, p, arm: str):
        """(policy, fresh diag) of an ENUM± arm for the goal the arm receives (H-CONJ: the linearization)."""
        return self.EN().enum_policy(d.space, p.weights(), p.c, kind=ENUM_KIND[arm], stop=d.arm_stop(p),
                                     edge_mask=d.mask(p), phi=d.phi)

    def train_zero_shot(self, model: str, d: Data, data: dict, *, seed: int, cfg: dict, out_dir, safe_point):
        """{epoch: model path} or None (yielded; the same call resumes)."""
        return self.LE().train_zero_shot(model, d.space, data, seed=seed, epochs=cfg["epochs"], lr=cfg["lr"],
                                         keep=tuple(cfg["keep"]), out_dir=str(out_dir), safe_point=safe_point)

    def fit_pergoal(self, kind: str, d: Data, jobs: list, cfg: dict, *, out_dir, safe_point):
        """{job key: model path} or None (yielded; the same call resumes)."""
        return self.LE().fit_pergoal(kind, d.space, jobs, cfg, out_dir=str(out_dir), safe_point=safe_point)

    def kind(self, path) -> str:
        return self._loaded(path).kind

    def _loaded(self, path):
        cache = self.__dict__.setdefault("_models", OrderedDict())
        path = str(path)
        if path not in cache:
            cache[path] = self.LE().load_model(path)
            while len(cache) > 16:
                cache.popitem(last=False)
        return cache[path]

    def psi(self, path, d: Data, states, W, c) -> np.ndarray:
        """psi-hat [len(states), m, F + 1] for the m codes (W, c)."""
        return np.asarray(self.LE().predict_psi(self._loaded(path), d.space, np.asarray(states), np.atleast_2d(W),
                                                np.atleast_1d(c)), dtype=np.float64)

    def value(self, path, d: Data, states, W=None, c=None) -> np.ndarray:
        """V-hat [len(states), m] (a PGV network ignores the code)."""
        return np.asarray(self.LE().predict_value(self._loaded(path), d.space, np.asarray(states),
                                                  None if W is None else np.atleast_2d(W),
                                                  None if c is None else np.atleast_1d(c)), dtype=np.float64)

    def time_step(self, model: str, n: int, width: int, batch: int, B: int) -> float:
        return float(self.LE().time_step(model, width=width, batch=batch, B=B, n=n))


# ============================================================================= arms and cells

def best_single_policy(d: Data, p, s0: int, base: bool):
    """BEST-SINGLE(L): the library policy with the best exact value w'.phi(e_i(s0)) - c' T_i(s0) (first maximum),
    followed in closed loop (STOP where it stops, or where its move is not a legal candidate)."""
    lib, stop = d.library(base), d.arm_stop(p)
    i = int(np.argmax(stop[lib.end[:, s0]] - p.c * lib.T[:, s0].astype(np.float64)))
    nxt = lib.nxt[i]

    def pol(s, acts, dsts):
        j = np.flatnonzero(np.asarray(dsts) == nxt[s])
        return 0 if nxt[s] == s or not len(j) else int(j[0]) + 1
    return pol


def ref_policy(arm: str, d: Data, p, s0: int, split: str, base: bool, ad):
    """(policy, diag dict or None) of a reference / ENUM± arm at start s0."""
    if arm == "ORACLE":
        return SO.oracle_policy(d.table(p)), None
    if arm == "STOP":
        return SO.stop_policy(), None
    if arm == "GREEDY":
        return SO.greedy_policy(d.arm_stop(p), p.c), None
    if arm == "RAND":
        return SO.rand_policy(FA.stream(SEEDS["starts"], f"rand:{split}:{p.key}:{int(s0)}")), None
    if arm == "EXACT-GPI(L)":
        stop = d.arm_stop(p)
        G = d.cache(p, ("gpi", base), lambda: SO.gpi_table(stop, p.c, tabs_of(d.library(base))))
        return SO.gpi_policy(stop, p.c, G), None
    if arm == "BEST-SINGLE(L)":
        return best_single_policy(d, p, int(s0), base), None
    if arm in ENUM_KIND:
        return ad.enum_policy(d, p, arm)
    raise KeyError(arm)


def codes_for(arm: str, p, lib) -> tuple:
    """(W [m, F], c [m]) of an SF arm's GPI code set C: SF-GPI and SF-shuffled L + {w'}; SF-noGPI and SF-FT_400 {w'};
    SF-L L; SF-GPI-5 {w'} + the 4 library codes nearest to z' = (w'/3, c') by cosine (ties: lower library index)."""
    wq, cq = p.weights()[None], np.array([p.c])
    if arm in ("SF-noGPI", "SF-FT_400"):
        return wq, cq
    if arm == "SF-L":
        return lib.W, lib.c
    if arm == "SF-GPI-5":
        Z = np.concatenate([lib.W / 3.0, lib.c[:, None]], axis=1)
        z = np.concatenate([wq[0] / 3.0, cq])
        near = np.argsort(-(Z @ z) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(z)), kind="stable")[:4]
        return np.concatenate([wq, lib.W[near]]), np.concatenate([cq, lib.c[near]])
    return np.concatenate([lib.W, wq]), np.concatenate([lib.c, cq])


def learned_policy(arm: str, path, d: Data, p, base: bool, ad, lib=None):
    """Closed-loop policy of a learned arm: SF arms (an sf network) score moves by GPI over their code set, value arms
    (uvfa, aux, pgv, FT) by -c' + V-hat(s'); STOP is scored exactly as w'.phi(s)."""
    stop, w, c = d.arm_stop(p), p.weights(), p.c
    if ad.kind(path) == "sf":
        W, C = codes_for(arm, p, d.library(base) if lib is None else lib)
        return SO.scored_policy(stop, lambda s, a, dsts: SO.gpi_scores(ad.psi(path, d, dsts, W, C), w, c))
    return SO.scored_policy(stop, lambda s, a, dsts: ad.value(path, d, dsts, w[None], np.array([c]))[:, 0] - c)


def model_values(ad, path, d: Data, p, S) -> np.ndarray:
    """V-hat on states S (G-P4, banks): w-bar'.psi-hat(s, w') for an sf network, else V-hat(s, w')."""
    w, c = p.weights(), p.c
    if ad.kind(path) == "sf":
        return SO.wbar_dot(ad.psi(path, d, S, w[None], np.array([c]))[:, 0], w, c)
    return ad.value(path, d, S, w[None], np.array([c]))[:, 0]


def eval_cells(log: KeyedLog, d: Data, sl, *, split, family, p, arm, tag, policy_fn, digest, xarm=None,
               rep=None) -> bool:
    """One (arm, tag, parent) unit: a keyed cell per start (a safe point after each; the unit's cells are written with
    one fsync, also when it stops early). True when all starts are written."""
    tab, mask, buf, stopped = d.table(p), d.mask(p), [], False
    for s0 in p.starts:
        key = kj("cell", split, arm, tag, p.key, int(s0))
        if key in log:
            continue
        t0 = time.perf_counter()
        pol, diag = policy_fn(int(s0))
        r = SO.evaluate(d.space, tab, int(s0), pol, mask)
        rec = dict(split=split, family=family, arm=arm, tag=tag, parent=p.key, s0=int(s0), nreg=r["nreg"], J=r["J"],
                   T=r["T"], loop=r["loop"], capped=r["capped"], final=r["final"], dt=time.perf_counter() - t0)
        if diag is not None:
            rec["diag"] = {k: (int(v) if isinstance(v, (bool, int, np.integer)) else v) for k, v in dict(diag).items()}
        if xarm is not None:
            rec.update(xarm=xarm, rep=rep)
        buf.append((key, rec))
        if sl is not None and sl.safe_point("cell"):
            stopped = True
            break
    log.add_many(buf, digest)
    return not stopped or cells_done(log, split, arm, tag, p)


def cells_done(log, split, arm, tag, p) -> bool:
    return all(kj("cell", split, arm, tag, p.key, int(s)) in log for s in p.starts)


def parent_nreg(log, split, arm, tag, p) -> float:
    """Mean nreg over the parent's starts (sum / len, as rcx.stats)."""
    v = [log.get(kj("cell", split, arm, tag, p.key, int(s)))["nreg"] for s in p.starts]
    return sum(v) / len(v)


def family_mean(log, split, arm, tag, ps) -> float:
    return mean(parent_nreg(log, split, arm, tag, p) for p in ps)


def bank_unit(log: KeyedLog, d: Data, *, split, p, arm, tag, choose, vhat, digest, xarm=None, rep=None) -> bool:
    """One-step bank of a parent for one arm: per state top-1 and regret1 (if `choose`) and |V-hat - V*| (if `vhat`)."""
    key = kj("bank", split, arm, tag, p.key)
    if key in log:
        return True
    tab, keep = d.table(p), SO.sorted_mask(d.space, d.mask(p))
    vh = vhat(np.asarray(p.bank, dtype=np.int64)) if vhat is not None else None
    rows = []
    for i, s in enumerate(p.bank):
        acts, dsts = SO.moves(d.space, int(s), keep)
        row = dict(s=int(s), vstar=float(tab.V[s]))
        if choose is not None:
            r1, top = SO.regret1(SO.q_star(tab, int(s), dsts), choose(int(s), acts, dsts), tab.c)
            row.update(regret1=r1, top1=top)
        if vh is not None:
            row["vhat"] = float(vh[i])
        rows.append(row)
    rec = dict(split=split, arm=arm, tag=tag, parent=p.key, c=tab.c, rows=rows)
    if xarm is not None:
        rec.update(xarm=xarm, rep=rep)
    log.add(key, rec, digest)
    return True


def _bank_rows(log, split, arm, tag, ps):
    return [(p, r) for p in ps for r in log.get(kj("bank", split, arm, tag, p.key))["rows"]]


def acc_half(log, split, arm, tag, ps) -> float:
    """Share of bank pairs with |V-hat - V*| < c/2 (G-P4)."""
    rows = _bank_rows(log, split, arm, tag, ps)
    return sum(abs(r["vhat"] - r["vstar"]) < p.c / 2 for p, r in rows) / len(rows)


def top1_rate(log, split, arm, tag, ps) -> float:
    rows = _bank_rows(log, split, arm, tag, ps)
    return sum(bool(r["top1"]) for _, r in rows) / len(rows)


# ============================================================================= phase driver (lane slices)

class Unit(NamedTuple):
    key: str
    cost_s: float
    run: Callable


class Slice:
    """Safe points of one lane slice: stop on a lane_e05 yield condition or once the planned length L is used."""

    def __init__(self, sess, L_min: float, now):
        self.sess, self.L_s, self.now, self.t0 = sess, 60.0 * L_min, now, now()
        self.reason = None

    def safe_point(self, kind: str = "step") -> bool:
        """True = stop now (the caller checkpoints first). kind 'batch' / 'epoch' forces a lane check; otherwise it
        runs once >= 60 s have passed since the previous one (lane_e05.Session.check)."""
        if self.reason:
            return True
        r = self.sess.check(force=kind in ("batch", "epoch"))
        if r:
            self.reason = r
        elif (self.now() - self.t0).total_seconds() >= self.L_s:
            self.reason = "slice length reached"
        return bool(self.reason)


def drive(ctx, phase: str, plan: Callable[[], list], *, attempt, cd: dict | None, args: dict | None = None) -> int:
    """Run pending units of `phase` in one lane slice (module docstring); returns the exit code."""
    pending = plan()
    if not pending:
        print(f"{phase}: complete (nothing pending)", flush=True)
        return 0
    need = LA.projected_min(seconds=ctx.cost_scale * sum(u.cost_s for u in pending))
    L = LA.slice_budget_min(ctx.env)  # the slice ends earlier once the work is done (module docstring)
    if not L > 0:
        print(f"{phase}: not starting (no room before the window-rule deadline)", flush=True)
        return EXIT_REFUSED
    try:
        sess = LA.Session.open(ctx.env, f"RCX E05 {phase}", L)
    except LA.Refused as e:
        print(f"{phase}: lane start condition does not hold; nothing written:\n  - " +
              "\n  - ".join(e.decision.reasons), flush=True)
        return EXIT_REFUSED
    sl = Slice(sess, L, ctx.env.now)
    sid = secrets.token_hex(6)
    log_attempt(ctx, "start", phase=phase, attempt=attempt, slice=sid, mode=sess.mode, L_min=round(L, 3),
                projected_min=round(need, 3), digest=(cd or {}).get("digest"))
    _append_line(ctx.runs / "PROVENANCE.jsonl", dict(provenance(ctx, phase, cd, args=args or {}), slice=sid,
                                                    attempt=attempt, lane_mode=sess.mode, lane_token=sess.rec["token"]))
    done, status, last, queue = False, "error", None, deque(pending)
    try:
        while True:
            if not queue:  # re-plan only when the current list is used up (plans can be staged)
                queue = deque(plan())
                if not queue:
                    done = True
                    break
                if queue[0].key == last:
                    raise RuntimeError(f"{phase}: unit {last} still pending after it reported completion")
            if sl.safe_point("unit"):
                break
            u = queue.popleft()
            print(f"{time.strftime('%H:%M:%S')} {phase}: {u.key}", flush=True)
            if not u.run(sl):
                break
            last = u.key
        status = "ok"
    finally:
        reason = "done" if done else (sl.reason or ("error" if status != "ok" else "stopped"))
        sess.release(reason)
        log_attempt(ctx, "end", phase=phase, attempt=attempt, slice=sid, status=status, reason=reason, complete=done)
    print(f"{phase}: {'complete' if done else 'yielded: ' + reason}", flush=True)
    return 0 if done else EXIT_REQUEUE


def _sp(sl):
    return sl.safe_point if sl is not None else (lambda kind="step": False)


# ============================================================================= instrument checks (pilot)

def q1_check(rng, n_w: int) -> dict:
    """Q1 on n = 5: value iteration == brute force max over s' of [w.phi(s') - c d(s, s')] for random w (incl.
    negatives and clear facts) and c in {0.125, 0.25, 0.75}."""
    sp = SP.get_space(5)
    phi = SO.get_phi(sp)
    N = len(sp)
    D = np.empty((N, N), dtype=np.float64)
    for j in range(N):
        m = np.zeros(N, dtype=bool)
        m[j] = True
        D[:, j] = sp.backward_bfs(m)
    bad = 0
    for _ in range(n_w):
        w = np.array([rng.choice((-2, -1, 0, 0, 0, 1, 2, 3)) for _ in range(phi.shape[1])], dtype=np.float64)
        c = rng.choice((0.125, 0.25, 0.75))
        stop = SO.stop_values(phi, w)
        bad += int(not np.array_equal(SO.soft_values(sp, w, c, phi=phi), (stop[None, :] - c * D).max(axis=1)))
    return dict(value=bad, n=n_w, ok=bad == 0)


def q2_check(rng, n_goals: int, n_states: int, n: int = 6) -> dict:
    """Q2: H-CONJ V* == max(3 1[G <= s], 3 - c h*(s, G)) with h* from GoalLabels, and == max-plus VI (n = 6)."""
    sp, fs = SP.get_space(n), SO.facts(n)
    bad = pairs = goals = 0
    while goals < n_goals:
        G = tuple(sorted(rng.sample(range(n * (n - 1)), rng.choice((2, 3)))))
        if not FA.conj_satisfiable(sp, G):
            continue
        goals += 1
        goal = [fs[k] for k in G]
        V = SO.conj_values(sp, goal, 3.0, 0.25)
        m = sp.goal_mask(SO.compile_goal(n, goal))
        vi = SO.max_plus_vi(sp, 3.0 * m, 0.25)
        gl = SP.GoalLabels(list(range(n)), frozenset(goal))
        for s in rng.sample(range(len(sp)), n_states):
            ref = max(3.0 * bool(m[s]), 3.0 - 0.25 * gl.h_id(s))
            bad += int(not (V[s] == ref == vi[s]))
            pairs += 1
    return dict(value=bad, n=pairs, ok=bad == 0)


def q3_check(d: Data, rng, n_pairs: int) -> dict:
    """Q3: rollouts of pi_w reproduce the tabled endpoints and lengths, and V* = w-bar.psi exactly (pilot goals; the
    pairs are drawn parent by parent, ceil(n_pairs / parents) random states each, so every table is solved once)."""
    ps = [p for ps in d.fam.values() for p in ps]
    bad = n = 0
    for p in ps:
        tab = d.table(p)
        for _ in range(-(-n_pairs // len(ps))):
            s = rng.randrange(len(d.space))
            ep = SO.run_episode(d.space, s, SO.oracle_policy(tab), d.mask(p))
            v = float(tab.stop[tab.end[s]]) - tab.c * float(tab.T[s])
            bad += int(not (ep.final == tab.end[s] and ep.T == tab.T[s] and not ep.loop and v == tab.V[s]))
            n += 1
    return dict(value=bad, n=n, ok=bad == 0)


def q4_parent(d: Data, p, base: bool) -> int:
    """Starts of p where EXACT-GPI(L)'s closed-loop return < max over library i of V^{pi_i}_{w'}(s0)."""
    lib, stop = d.library(base), d.arm_stop(p)
    G = d.cache(p, ("gpi", base), lambda: SO.gpi_table(stop, p.c, tabs_of(lib)))
    bad = 0
    for s0 in p.starts:
        J = SO.episode_return(stop, p.c, SO.run_episode(d.space, int(s0), SO.gpi_policy(stop, p.c, G)))
        bad += int(J < float((stop[lib.end[:, s0]] - p.c * lib.T[:, s0].astype(np.float64)).max()))
    return bad


def q4_check(d: Data, families, base: bool) -> dict:
    """Q4: GPI >= every library policy at every start (free-dynamics parents; scored with the arms' STOP values)."""
    bad = n = 0
    for f in families:
        for p in d.fam[f]:
            if p.dyn == "free":
                bad += q4_parent(d, p, base)
                n += len(p.starts)
    return dict(value=bad, n=n, ok=bad == 0)


def q5_check(d: Data, rng, n_sigma: int) -> dict:
    """Q5 (pilot goals): V*_{sigma w}(sigma s) == V*_w(s) on all states, canonical keys invariant, and the conjugate
    psi identity w-bar_sigma . sigma psi^{pi_w}(s) == V*_{sigma w}(sigma s), for n_sigma random renamings."""
    ps = [p for f, ps in d.fam.items() if f != "TRAIN" for p in ps if p.rtype == "lin" and p.dyn == "free"]
    bad = 0
    for _ in range(n_sigma):
        p = ps[rng.randrange(len(ps))]
        sigma = FA.random_sigma(rng, d.n)
        tab = d.table(p)
        ws = SO.rename_weight(p.weights(), sigma)
        Vs = SO.soft_values(d.space, ws, p.c, phi=d.phi)
        sp = SO.state_perm(d.space, sigma)
        conj = SO.stop_values(d.phi, ws)[sp[tab.end]] - p.c * tab.T.astype(np.float64)
        bad += int(not (np.array_equal(Vs[sp], tab.V) and np.array_equal(conj, tab.V)
                        and FA.canon_key(ws, p.c, d.n) == p.key))
    return dict(value=bad, n=n_sigma, ok=bad == 0)


def q6_check() -> dict:
    """Q6 on n = 5: height_limit_mask(h = 3) == a direct rule check on every edge (the stacked block's level in the
    successor, walked down to the table, is <= 3)."""
    sp, n = SP.get_space(5), 5
    mask = SO.height_limit_mask(sp, 3)
    bad = 0
    for e in range(sp.n_edges):
        op, x = int(sp.fwd_act[e]) // (n * n), (int(sp.fwd_act[e]) // n) % n
        legal = True
        if op == SP.STACK:
            pos, lev, b = sp.states[sp.fwd_dst[e]], 1, x
            while pos[b] != n:
                b, lev = int(pos[b]), lev + 1
            legal = lev <= 3
        bad += int(legal != bool(mask[e]))
    return dict(value=bad, n=int(sp.n_edges), ok=bad == 0)


def q8_check(d: Data, excl, label_sets: dict) -> dict:
    """Q8: split hygiene on the M6 key (families_e05.check_splits) and every start, bank state and label row
    re-verified (distinct, valid, not excluded now)."""
    reg = FA.Registry.from_parents(d.n, [p for ps in d.fam.values() for p in ps])
    splits = {k: [bool(v[0]), v[1]] for k, v in FA.check_splits(reg).items()}
    bad = Counter()
    for ps in d.fam.values():
        for p in ps:
            v = FA.verify_parent(d.space, d.phi, p, excl, labels=label_sets.get(p.key, ()))
            bad["starts"] += int(bool(p.starts) and not v["starts_ok"])
            bad["bank"] += int(bool(p.bank) and not v["bank_ok"])
            for k in ("excluded_starts", "excluded_bank", "excluded_labels"):
                bad[k] += int(v.get(k, 0))
    return dict(value=dict(splits=splits, defects=dict(bad)), ok=all(v[0] for v in splits.values())
                and not sum(bad.values()))


# ============================================================================= pilot (Phase 1a / 1b)

class Pilot:
    """State and units of the pilot phases in ctx.pdir (keyed logs: cells, banks, checks, fits)."""

    def __init__(self, ctx, cd: dict | None):
        self.ctx, self.cd, self.ad = ctx, cd, ctx.adapters
        self.digest = (cd or {}).get("digest")
        self.dir = ctx.pdir
        self.manifest_path = self.dir / "PILOT_MANIFEST.json"
        self.cells = KeyedLog(self.dir / "cells.jsonl")
        self.banks = KeyedLog(self.dir / "banks.jsonl")
        self.checks = KeyedLog(self.dir / "checks.jsonl")
        self.fits = KeyedLog(self.dir / "fits.jsonl")
        self._d = None
        for lg in (self.cells, self.banks, self.checks, self.fits):
            other = lg.digests() - {self.digest}
            if self.digest and other:
                raise SystemExit(f"refusing: {lg.path} holds outputs of another code digest {sorted(map(str, other))}"
                                 " (§11: re-run the affected pilot phase in a fresh --pilot-dir and log it)")

    # ---- data
    def data(self) -> Data:
        if self._d is None:
            self._d = Data(self.ctx, read_json(self.manifest_path), self.dir)
        return self._d

    @property
    def m(self) -> dict:
        return self.data().m

    def dev(self) -> dict:
        return {f: ps for f, ps in self.data().fam.items() if f.startswith("DEV")}

    def dev_parents(self) -> list:
        return [(f, p) for f, ps in self.dev().items() for p in ps]

    # ---- unit: draw (TRAIN, DEV, labels, base-policy labels)
    def unit_draw(self, sl) -> bool:
        ctx, sz = self.ctx, self.ctx.sz()
        space = SP.get_space(ctx.n)
        phi = SO.get_phi(space)
        excl = ctx.exclusions(space)
        reg = FA.draw_pilot(space, phi, excl, ctx.sizes)
        FA.assert_splits(reg)
        dev_k = FA.dev_kcurve_subset(reg, sz["dev_kcurve"] // 2)
        kmax = max(FA.SIZES["kcurve"])
        train_labels, pergoal, base_labels, lab_excl = {}, {}, {}, Counter()
        for p in reg.parents["TRAIN"]:
            idx, cnt = FA.draw_labels(space, p, TRAIN_LABELS_MAX, excl, seed=SEEDS["train_labels"], tag="train")
            train_labels[p.key] = idx
            lab_excl.update(cnt)
        for p in reg.parents["DEV-ID"] + reg.parents.get("DEV-OOD:H-K", []) + reg.parents.get("DEV-OOD:H-NEG", []):
            idx, cnt = FA.draw_labels(space, p, kmax if p.key in dev_k else DEV_PERGOAL, excl)
            pergoal[p.key] = idx
            lab_excl.update(cnt)
        fs = SO.facts(ctx.n)
        for k in range(ctx.n * ctx.n):
            idx, cnt = FA.sample_states(np.arange(len(space)), BASE_LABELS_MAX,
                                        FA.stream(SEEDS["train_labels"], f"train-base:{k}"), frozenset([fs[k]]), excl)
            base_labels[str(k)] = idx
            lab_excl.update(cnt)
        man = dict(experiment="RCX-E05", kind="pilot", n=ctx.n, code_digest_t2=self.digest,
                   parents=[p.to_json() for p in reg.all_parents()],
                   rejections={f: dict(c) for f, c in reg.rejections.items()}, notes=dict(reg.notes),
                   train_labels=train_labels, pergoal_labels=pergoal, base_labels=base_labels,
                   dev_kcurve=list(dev_k), label_exclusions=dict(lab_excl),
                   exclusions=None if excl is None else dict(calls=dict(excl.calls), sha=sha_file(
                       ctx.root / "experiments/RCX-E07/EXCLUSIONS.json")),
                   streams=dict(families="20261242:pilot", starts="20261244:starts:pilot:<key>",
                                bank="20261244:bank:pilot:<key>", train="20261243:train:<key>",
                                pergoal="20261249:labels:<key>", base="20261243:train-base:<fact>",
                                dev_kcurve="20261242:dev-kcurve"),
                   provenance=provenance(ctx, "pilot-a", self.cd))
        sha = write_create_only(self.manifest_path, man)
        log_attempt(ctx, "manifest", attempt="pilot", file="PILOT_MANIFEST.json", sha256=sha)
        return True

    def unit_library(self, part: str):
        def run(sl):
            build_library(self.data(), part, self.dir / f"oracle_library_{part}.npz")
            return True
        return run

    def lib_ready(self, base: bool) -> bool:
        return all((self.dir / f"oracle_library_{x}.npz").exists() for x in (("train", "base") if base else ("train",)))

    # ---- unit: Q checks
    def unit_check(self, q: str):
        def run(sl):
            d, qs, rng = self.data(), self.ctx.q_sizes, FA.stream(SEEDS["families"], f"q-checks:{q}")
            if q == "Q1":
                r = q1_check(rng, qs["q1"])
            elif q == "Q2":
                r = q2_check(rng, qs["q2_goals"], qs["q2_states"], qs.get("q2_n", 6))
            elif q == "Q3":
                r = q3_check(d, rng, qs["q3"])
            elif q == "Q4":
                r = q4_check(d, [f for f in d.fam if f.startswith("DEV")], self.base())
            elif q == "Q5":
                r = q5_check(d, rng, qs["q5"])
            elif q == "Q6":
                r = q6_check()
            else:
                labs = dict(self.m["pergoal_labels"], **self.m["train_labels"])
                r = q8_check(d, self.ctx.exclusions(d.space), labs)
            self.checks.add(self.check_key(q), r, self.digest)
            return True
        return run

    def check_key(self, q):
        return kj("check", q, self.base() if q == "Q4" else None)

    def check(self, q: str):
        return self.checks.get(self.check_key(q))

    # ---- reference arms on DEV
    @staticmethod
    def ref_tag(arm, base=False):
        return lib_tag(base) if arm in ("EXACT-GPI(L)", "BEST-SINGLE(L)") else "-"

    def ref_unit(self, arm, f, p, base):
        def run(sl):
            d = self.data()
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=arm, tag=self.ref_tag(arm, base),
                              policy_fn=lambda s0: ref_policy(arm, d, p, s0, "pilot", base, self.ad),
                              digest=self.digest)
        return run

    # ---- gates of pilot A
    def gp2(self):
        """G-P2: DEV-ID EXACT-GPI(L) mean nreg >= 0.5 -> add the 49 single-fact base policies (None if not yet known)."""
        ps = self.dev().get("DEV-ID", [])
        if not ps or not all(cells_done(self.cells, "pilot", "EXACT-GPI(L)", "L", p) for p in ps):
            return None
        v = family_mean(self.cells, "pilot", "EXACT-GPI(L)", "L", ps)
        return dict(value=v, base=ge(v, 0.5))

    def base(self) -> bool:
        g = self.gp2()
        return bool(g and g["base"])

    def stage_labels(self) -> int:
        """400, or 800 once G-P4(a) failed at 400 (§9.1)."""
        g = self.gp4(400)
        return 800 if g is not None and not g["a"]["ok"] else 400

    # ---- training data and selection models
    def data_path(self, labels, base) -> Path:
        return self.dir / f"train_data_L{labels}{'_base' if base else ''}.npz"

    def train_goals(self, labels: int, base: bool, n_goals: int | None = None) -> tuple:
        d = self.data()
        ps = d.fam["TRAIN"][:n_goals] if n_goals else d.fam["TRAIN"]
        goals = [(p.weights(), p.c) for p in ps]
        idxs = [self.m["train_labels"][p.key][:labels] for p in ps]
        if base:
            goals += d.lib_goals("base")
            idxs += [self.m["base_labels"][str(k)][:labels] for k in range(d.n * d.n)]
        return goals, idxs

    def unit_train_data(self, labels, base, path=None, n_goals=None):
        def run(sl):
            goals, idxs = self.train_goals(labels, base, n_goals)
            save_npz_create_only(path or self.data_path(labels, base), **train_rows(self.data(), goals, idxs))
            return True
        return run

    def zs_key(self, model, labels, base, cfg):
        return kj("train", model, SEEDS["selection"], labels, base, cfg)

    def unit_train(self, model, labels, base, cfg):
        def run(sl):
            p = self.data_path(labels, base)
            out = self.dir / "models" / f"{model}_s{SEEDS['selection']}_L{labels}{'_base' if base else ''}_{cfg}"
            spec = ZS_BASE if cfg == "base" else ZS_EXT
            res = self.ad.train_zero_shot(model, self.data(), dict(np.load(p)), seed=SEEDS["selection"], cfg=spec,
                                          out_dir=out, safe_point=_sp(sl))
            if res is None:
                return False
            models = {str(k): str(v) for k, v in res.items()}
            self.fits.add(self.zs_key(model, labels, base, cfg),
                          dict(model=model, labels=labels, base=base, cfg=cfg, models=models,
                               shas={k: _sha_path(v) for k, v in models.items()}), self.digest)
            return True
        return run

    def model_path(self, model, labels, base, cfg, epoch):
        r = self.fits.get(self.zs_key(model, labels, base, cfg))
        return None if r is None else r["models"][str(epoch)]

    @staticmethod
    def zs_tag(model, labels, base, cfg, epoch):
        return f"{model}|L{labels}{'+base' if base else ''}|{cfg}|e{epoch}"

    def zs_unit(self, arm, f, p, labels, base, cfg, epoch):
        def run(sl):
            d = self.data()
            pol = learned_policy(arm, self.model_path(ARM_MODEL[arm], labels, base, cfg, epoch), d, p, base, self.ad)
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=arm,
                              tag=self.zs_tag(ARM_MODEL[arm], labels, base, cfg, epoch),
                              policy_fn=lambda s0: (pol, None), digest=self.digest)
        return run

    def zs_units(self, arms, fam_parents, labels, base, cfg, epochs) -> list:
        """Parent-major evaluation units of zero-shot arms at the given configs."""
        U = []
        for f, p in fam_parents:
            for arm in arms:
                for e in epochs:
                    tag = self.zs_tag(ARM_MODEL[arm], labels, base, cfg, e)
                    if not cells_done(self.cells, "pilot", arm, tag, p):
                        U.append(Unit(f"zs:{arm}:{tag}:{f}:{p.key}", 16.0, self.zs_unit(arm, f, p, labels, base, cfg,
                                                                                        e)))
        return U

    def zs_mean(self, arm, f, labels, base, cfg, epoch):
        ps = self.dev()[f]
        tag = self.zs_tag(ARM_MODEL[arm], labels, base, cfg, epoch)
        if not all(cells_done(self.cells, "pilot", arm, tag, p) for p in ps):
            return None
        return family_mean(self.cells, "pilot", arm, tag, ps)

    def selection(self, labels, base):
        """Selected epoch per model by the DEV-ID nreg of its selecting arm (ties: fewer epochs)."""
        out = {}
        for model, arm in SELECT_BY.items():
            vals = {e: self.zs_mean(arm, "DEV-ID", labels, base, "base", e) for e in ZS_BASE["keep"]}
            if any(v is None for v in vals.values()):
                return None
            e = min(vals, key=lambda k: (vals[k], k))
            out[model] = dict(cfg="base", epoch=e, dev_id=vals[e], candidates=vals)
        return out

    def gp4(self, labels, *, final_uvfa=True):
        """G-P4 at `labels` (None until computable): (a) SF value-exact and ranking vs the copy floor; (b) UVFA validity
        (base config, then the one extension)."""
        if self.gp2() is None:
            return None
        base = self.base()
        sel = self.selection(labels, base)
        if sel is None:
            return None
        dev = self.dev()["DEV-ID"]
        sf_tag = self.zs_tag("SF", labels, base, "base", sel["SF"]["epoch"])
        uv_tag = self.zs_tag("UVFA", labels, base, "base", sel["UVFA"]["epoch"])
        need = [("SF-own", sf_tag), ("SF-noGPI", sf_tag), ("UVFA", uv_tag), ("COPY", "-"), ("GREEDY", "-")]
        if not all(kj("bank", "pilot", a, t, p.key) in self.banks for a, t in need for p in dev):
            return None
        copy_acc = acc_half(self.banks, "pilot", "COPY", "-", dev)
        a_acc = acc_half(self.banks, "pilot", "SF-own", sf_tag, dev) - copy_acc
        a_top = top1_rate(self.banks, "pilot", "SF-noGPI", sf_tag, dev) - top1_rate(self.banks, "pilot", "GREEDY", "-",
                                                                                    dev)
        a = dict(acc_delta=a_acc, top1_delta=a_top, copy_acc=copy_acc, ok=ge(a_acc, 0.40) and ge(a_top, 0.20))
        b_acc = acc_half(self.banks, "pilot", "UVFA", uv_tag, dev) - copy_acc
        b_nreg = sel["UVFA"]["dev_id"]
        b = dict(acc_delta=b_acc, dev_nreg=b_nreg, ok=ge(b_acc, 0.20) and le(b_nreg, BAR), cfg="base",
                 epoch=sel["UVFA"]["epoch"])
        if not b["ok"] and final_uvfa:
            ext = self.uvfa_ext(labels, base, copy_acc)
            b["extension"] = ext
            if ext is not None and ext["ok"]:
                b.update(ok=True, cfg="ext", epoch=ZS_EXT["keep"][-1])
        return dict(labels=labels, base=base, selection=sel, a=a, b=b)

    def uvfa_ext(self, labels, base, copy_acc):
        e = ZS_EXT["keep"][-1]
        v = self.zs_mean("UVFA", "DEV-ID", labels, base, "ext", e)
        tag = self.zs_tag("UVFA", labels, base, "ext", e)
        dev = self.dev()["DEV-ID"]
        if v is None or not all(kj("bank", "pilot", "UVFA", tag, p.key) in self.banks for p in dev):
            return None
        acc = acc_half(self.banks, "pilot", "UVFA", tag, dev) - copy_acc
        return dict(acc_delta=acc, dev_nreg=v, ok=ge(acc, 0.20) and le(v, BAR))

    def final_configs(self):
        """{model: [labels, base, cfg, epoch]} at the final stage, None while G-P4 is pending, or 'n6' if (a) failed
        at 400 and at 800 labels."""
        g = self.gp4(400)
        if g is None:
            return None
        if not g["a"]["ok"]:
            g = self.gp4(800)
            if g is None:
                return None
            if not g["a"]["ok"]:
                return "n6"
        if not g["b"]["ok"] and g["b"].get("extension") is None:
            return None
        L, base, sel = g["labels"], g["base"], g["selection"]
        out = {m: [L, base, "base", s["epoch"]] for m, s in sel.items()}
        if g["b"]["cfg"] == "ext":
            out["UVFA"] = [L, base, "ext", ZS_EXT["keep"][-1]]
        return out

    # ---- bank units (G-P4)
    def bank_unit(self, arm, p, labels, base, cfg, epoch):
        def run(sl):
            d = self.data()
            stop, c = d.arm_stop(p), p.c
            if arm == "COPY":
                return bank_unit(self.banks, d, split="pilot", p=p, arm=arm, tag="-", choose=None,
                                 vhat=lambda S: stop[S], digest=self.digest)
            if arm == "GREEDY":
                return bank_unit(self.banks, d, split="pilot", p=p, arm=arm, tag="-", choose=SO.greedy_policy(stop, c),
                                 vhat=None, digest=self.digest)
            model = "SF" if arm.startswith("SF") else "UVFA"
            tag = self.zs_tag(model, labels, base, cfg, epoch)
            path = self.model_path(model, labels, base, cfg, epoch)
            if arm == "SF-own":
                return bank_unit(self.banks, d, split="pilot", p=p, arm=arm, tag=tag, choose=None,
                                 vhat=lambda S: model_values(self.ad, path, d, p, S), digest=self.digest)
            return bank_unit(self.banks, d, split="pilot", p=p, arm=arm, tag=tag,
                             choose=learned_policy(arm, path, d, p, base, self.ad),
                             vhat=(lambda S: model_values(self.ad, path, d, p, S)) if arm == "UVFA" else None,
                             digest=self.digest)
        return run

    # ---- plan of pilot A
    def plan_a(self) -> list:
        if not self.manifest_path.exists():
            return [Unit("draw", 180.0, self.unit_draw)]
        d, U = self.data(), []
        if not self.lib_ready(False):
            return [Unit("library:train", 0.3 * len(d.fam["TRAIN"]), self.unit_library("train"))]
        for q, cost in (("Q1", 30.0), ("Q2", 30.0), ("Q3", 120.0), ("Q5", 60.0), ("Q6", 5.0), ("Q8", 60.0)):
            if self.check(q) is None:
                U.append(Unit(f"check:{q}", cost, self.unit_check(q)))
        for f, p in self.dev_parents():
            for arm in REF_PILOT:
                if not cells_done(self.cells, "pilot", arm, self.ref_tag(arm, False), p):
                    U.append(Unit(f"ref:{arm}:{f}:{p.key}", 8.0, self.ref_unit(arm, f, p, False)))
        g2 = self.gp2()
        if g2 is None:
            return U
        base = g2["base"]
        if base and not self.lib_ready(True):
            U.append(Unit("library:base", 0.3 * d.n * d.n, self.unit_library("base")))
            return U
        if self.check("Q4") is None:
            U.append(Unit("check:Q4", 30.0, self.unit_check("Q4")))
        if base:
            for f, p in self.dev_parents():
                if not cells_done(self.cells, "pilot", "EXACT-GPI(L)", "L+base", p):
                    U.append(Unit(f"ref:EXACT-GPI(L)+base:{f}:{p.key}", 8.0, self.ref_unit("EXACT-GPI(L)", f, p, True)))
        if U:
            return U
        for labels in (400, 800):
            if labels == 800 and self.stage_labels() != 800:
                break
            U = self.stage_units(labels, base)
            if U:
                return U
        fc = self.final_configs()
        if fc in (None, "n6"):
            return U
        for arm in ZS_PILOT:  # DEV-OOD at the selected configs
            L, b, cfg, e = fc[ARM_MODEL[arm]]
            U += self.zs_units((arm,), [(f, p) for f, p in self.dev_parents() if f != "DEV-ID"], L, b, cfg, (e,))
        return U

    def stage_units(self, labels, base) -> list:
        if not self.data_path(labels, base).exists():
            return [Unit(f"traindata:L{labels}", 60.0, self.unit_train_data(labels, base))]
        n_rows = len(self.train_goals(labels, base)[0]) * labels
        steps = ZS_BASE["epochs"] * n_rows / 256
        U = [Unit(f"train:{m}:L{labels}", steps * 0.05, self.unit_train(m, labels, base, "base"))
             for m in ("SF", "UVFA", "UVFA+aux") if self.zs_key(m, labels, base, "base") not in self.fits]
        if U:
            return U
        dev = [("DEV-ID", p) for p in self.dev()["DEV-ID"]]
        U = self.zs_units(ZS_PILOT, dev, labels, base, "base", ZS_BASE["keep"])
        if U:
            return U
        sel = self.selection(labels, base)
        for arm, model in (("SF-own", "SF"), ("SF-noGPI", "SF"), ("UVFA", "UVFA"), ("COPY", None), ("GREEDY", None)):
            e = sel[model]["epoch"] if model else None
            tag = self.zs_tag(model, labels, base, "base", e) if model else "-"
            for _, p in dev:
                if kj("bank", "pilot", arm, tag, p.key) not in self.banks:
                    U.append(Unit(f"bank:{arm}:{tag}:{p.key}", 10.0, self.bank_unit(arm, p, labels, base, "base", e)))
        if U:
            return U
        g = self.gp4(labels, final_uvfa=False)
        if not g["b"]["ok"] and (g["a"]["ok"] or labels == 800):  # the one UVFA extension (80 epochs, lr 3e-4)
            e = ZS_EXT["keep"][-1]
            if self.zs_key("UVFA", labels, base, "ext") not in self.fits:
                return [Unit(f"train:UVFA-ext:L{labels}", steps * 2 * 0.05, self.unit_train("UVFA", labels, base, "ext"))]
            U = self.zs_units(("UVFA",), dev, labels, base, "ext", (e,))
            tag = self.zs_tag("UVFA", labels, base, "ext", e)
            U += [Unit(f"bank:UVFA-ext:{p.key}", 10.0, self.bank_unit("UVFA", p, labels, base, "ext", e)) for _, p in dev
                  if kj("bank", "pilot", "UVFA", tag, p.key) not in self.banks]
        return U

    # ---- pilot B: per-goal grids, k-curve, M5, G-P5
    def pg_key(self, grid, k, scope):
        return kj("fit", grid, k, scope)

    def pg_jobs(self, grid, ps, k) -> list:
        """fit_pergoal jobs (selection seed per job, module docstring); FT starts from the selected UVFA."""
        d, kind = self.data(), GRIDS[grid][0]
        jobs = []
        for p in ps:
            idx = np.asarray(self.m["pergoal_labels"][p.key][:k], dtype=np.int64)
            job = dict(key=p.key, states=idx, v=d.table(p).V[idx], stab=FA.stabilizer(np.asarray(p.w), d.n),
                       seed=FA.derive(SEEDS["selection"], f"{kind}:{grid}:{k}:{p.key}"), w=p.weights(), c=p.c)
            if kind == "FT":
                job["init"] = self.model_path("UVFA", *self.final_configs()["UVFA"])
            jobs.append(job)
        return jobs

    def unit_pg(self, grid, ps, k, scope):
        def run(sl):
            kind, cfg = GRIDS[grid]
            res = self.ad.fit_pergoal(kind, self.data(), self.pg_jobs(grid, ps, k), cfg,
                                      out_dir=self.dir / "models" / f"pg_{grid}_k{k}_{scope}", safe_point=_sp(sl))
            if res is None:
                return False
            self.fits.add(self.pg_key(grid, k, scope), dict(kind=kind, grid=grid, k=k, scope=scope,
                                                            models={str(a): str(b) for a, b in res.items()}),
                          self.digest)
            return True
        return run

    def pg_unit_eval(self, grid, k, scope, f, p):
        kind = GRIDS[grid][0]

        def run(sl):
            d = self.data()
            pol = learned_policy(kind, self.fits.get(self.pg_key(grid, k, scope))["models"][p.key], d, p, False,
                                 self.ad)
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=kind, tag=f"{grid}|k{k}",
                              policy_fn=lambda s0: (pol, None), digest=self.digest)
        return run

    def pg_mean(self, kind, grid, k, ps):
        if not ps or not all(cells_done(self.cells, "pilot", kind, f"{grid}|k{k}", p) for p in ps):
            return None
        return family_mean(self.cells, "pilot", kind, f"{grid}|k{k}", ps)

    def _pg_stage(self, grids, fam_parents, k, scope) -> list:
        """Fit units of the grid points (one fit each), then their evaluation units (parent-major)."""
        ps = [p for _, p in fam_parents]
        U = [Unit(f"fit:{g}:k{k}:{scope}", 0.02 * GRIDS[g][1]["steps"] * max(1, len(ps) // 16),
                  self.unit_pg(g, ps, k, scope)) for g in grids if self.pg_key(g, k, scope) not in self.fits]
        if U:
            return U
        for f, p in fam_parents:
            for g in grids:
                if not cells_done(self.cells, "pilot", GRIDS[g][0], f"{g}|k{k}", p):
                    U.append(Unit(f"pg:{g}:k{k}:{p.key}", 16.0, self.pg_unit_eval(g, k, scope, f, p)))
        return U

    def gp3(self):
        """G-P3 (None until computable): best PGV and FT configs on DEV-ID, competence <= 0.50, the one extension each,
        PG400 (ties to FT-UVFA_400) or P2 dropped."""
        dev = self.dev().get("DEV-ID", [])
        vals = {(kind, g): self.pg_mean(kind, g, 400, dev) for kind, grid in (("PGV", PGV_GRID), ("FT", FT_GRID))
                for g in grid}
        if any(v is None for v in vals.values()):
            return None
        res = {}
        for kind in ("PGV", "FT"):
            v, _, g = min((v, i, g) for i, ((kd, g), v) in enumerate(vals.items()) if kd == kind)
            res[kind] = dict(cfg=g, dev_id=v, competent=le(v, BAR))
        extended = False
        if not any(r["competent"] for r in res.values()):
            ext = {"PGV": self.pg_mean("PGV", "w128s4000", 400, dev), "FT": self.pg_mean("FT", "s1000", 400, dev)}
            if any(v is None for v in ext.values()):
                return dict(grid=res, extension_pending=True)
            extended = True
            for kind, g in (("PGV", "w128s4000"), ("FT", "s1000")):
                if lt(ext[kind], res[kind]["dev_id"]):
                    res[kind] = dict(cfg=g, dev_id=ext[kind], competent=le(ext[kind], BAR))
        comp = [k for k in ("FT", "PGV") if res[k]["competent"]]
        pg = None if not comp else min(comp, key=lambda k: (res[k]["dev_id"], 0 if k == "FT" else 1))
        return dict(grid=res, extended=extended, pg400=pg, p2_dropped=pg is None,
                    candidates={f"{k}:{g}": v for (k, g), v in vals.items()})

    def gp3b(self):
        g3 = self.gp3()
        if g3 is None or g3.get("extension_pending"):
            return None
        kdev = [self.data().by_key[k] for k in self.m["dev_kcurve"]]
        m = {400: self.pg_mean("PGV", g3["grid"]["PGV"]["cfg"], 400, kdev), 1600: self.pg_mean("PGV", "k1600", 1600, kdev),
             6400: self.pg_mean("PGV", "k6400", 6400, kdev)}
        if any(v is None for v in m.values()):
            return None

        def passes(m):
            return le(m[6400], 0.25) and le(m[1600], m[400]) and le(m[6400], m[1600])
        out = dict(original=m, original_ok=passes(m), recipe="original", kstar="decision-driving")
        if not out["original_ok"]:
            mx = {k: self.pg_mean("PGV", f"rext{k}", k, kdev) for k in (400, 1600, 6400)}
            if any(v is None for v in mx.values()):
                return dict(out, rext_pending=True)
            out.update(rext=mx, rext_ok=passes(mx))
            out.update(recipe="R_ext" if out["rext_ok"] else "original",
                       kstar="decision-driving" if out["rext_ok"] else "descriptive only")
        return out

    def plan_b(self) -> list:
        if self.plan_a():
            return []
        fc = self.final_configs()
        if fc in (None, "n6"):
            return []
        dev = [("DEV-ID", p) for p in self.dev()["DEV-ID"]]
        U = self._pg_stage(tuple(PGV_GRID) + tuple(FT_GRID), dev, 400, "dev")
        if U:
            return U
        g3 = self.gp3()
        if g3.get("extension_pending"):
            return self._pg_stage(tuple(PGV_EXT) + tuple(FT_EXT), dev, 400, "dev")
        kdev = [("DEV-ID", self.data().by_key[k]) for k in self.m["dev_kcurve"]]
        for k, g in KCURVE.items():
            U += self._pg_stage((g,), kdev, k, "kdev")
        if U:
            return U
        g3b = self.gp3b()
        if g3b.get("rext_pending"):
            for k, g in R_EXT.items():
                U += self._pg_stage((g,), kdev, k, "kdev")
            return U
        if not g3["p2_dropped"]:  # M5: the selected PG400 on the 24 DEV-OOD H-K/H-NEG parents
            ood = [(f, p) for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG") for p in self.dev().get(f, [])]
            U = self._pg_stage((g3["grid"][g3["pg400"]]["cfg"],), ood, 400, "ood")
            if U:
                return U
        if kj("bench") not in self.checks:
            U.append(Unit("bench:G-P5", 120.0, self.unit_bench))
        return U

    def unit_bench(self, sl) -> bool:
        """G-P5: optimizer-step seconds per (model, B, batch), VI seconds per weight, and evaluation seconds per cell per
        arm (the recorded pilot cells; ENUM± and GNN per family)."""
        d = self.data()
        step_s = {}
        for m, B, bs in LA.STEP_UNITS:
            width = {"PGV64": 64, "PGV128": 128, "PGV256": 256}.get(m, 256)
            step_s[LA.step_name(m, B, bs)] = self.ad.time_step(m, d.n, width, bs, B)
        t0 = time.perf_counter()
        for p in d.fam["TRAIN"][:5]:
            SO.sf_table(d.space, SO.stop_values(d.phi, p.weights()), p.c)
        vi_s = (time.perf_counter() - t0) / min(5, len(d.fam["TRAIN"]))
        cell = defaultdict(list)
        for r in self.cells.rows.values():
            cell[r["arm"]].append(r["dt"])
            if r["arm"].startswith(("ENUM", "GNN")):
                cell[f"{r['arm']}@{FA.base_family(r['family'])}"].append(r["dt"])
        cell_s = {k: dict(mean=mean(v), max=max(v), n=len(v)) for k, v in cell.items()}
        self.checks.add(kj("bench"), dict(step_s=step_s, vi_s=vi_s, cell_s=cell_s), self.digest)
        return True


def _sha_path(p) -> str | None:
    p = Path(p)
    if p.is_file():
        return sha_file(p)
    if p.is_dir():
        h = hashlib.sha256()
        for q in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f"{q.relative_to(p).as_posix()}\t{sha_file(q)}\n".encode())
        return h.hexdigest()
    return None


# ============================================================================= pilot report (gates, sizing, powers, cuts)

PHI = NormalDist().cdf
Z_P1, Z_P2 = 2.2414, 1.96
N_ID_RANGE, N_K_RANGE = (96, 152), (56, 72)
SECONDARY_SIZES = {"H-COST": 40, "H-SPAN": 40, "H-KN": 24, "H-CONJ": 40, "H-DYN": 40}
CUT_ORDER = ("budget-curve", "SF-FT_400", "H-KN", "GNN-ENUM±@H-NEG", "FT-UVFA_400-1rep", "ENUM-H*±@H-NEG-24", "H-DYN")
CAP_MIN = 150.0
PG_ARM = {"PGV": "PGV_400", "FT": "FT-UVFA_400"}


def macro_se(sigma, n_id, n_k, n_neg):
    return (sigma / 3.0) * math.sqrt(1.0 / n_id + 1.0 / n_k + 1.0 / n_neg)


def power_p1(sigma, n_id, n_k=None, n_neg=None, effect=0.05, macro=True) -> float:
    """§8: Phi((effect - max(z SE, 0.03)) / SE), z = 2.2414; macro SE = (sigma/3) sqrt(sum 1/n), H-ID SE = sigma/sqrt n."""
    se = macro_se(sigma, n_id, n_k, n_neg) if macro else sigma / math.sqrt(n_id)
    return _phi_ratio(effect - max(Z_P1 * se, 0.03), se)


def power_p2(sigma2, n_id, n_k, n_neg, d2=0.0, delta=0.05) -> float:
    """§8 NI power: Phi((delta - D2) / SE - 1.96) with the macro SE."""
    se = macro_se(sigma2, n_id, n_k, n_neg)
    return _phi_ratio(delta - d2 - Z_P2 * se, se)


def _phi_ratio(num: float, se: float) -> float:
    """Phi(num / se); the se = 0 limit (a zero DEV variance) is 1, 0.5 or 0 by the sign of num."""
    if se > 0:
        return PHI(num / se)
    return 1.0 if num > 0 else (0.5 if num == 0 else 0.0)


def n_id_for(sigma_hat_id: float) -> int:
    """§8 step 3: min(152, max(96, ceil(3,802 (1.15 sigma-hat)^2)))."""
    sp = 1.15 * sigma_hat_id
    return int(min(N_ID_RANGE[1], max(N_ID_RANGE[0], math.ceil(3802.0 * sp * sp - TOL))))


def size_plan(s_id, s_ood, s2_id, s2_ood, *, p2_dropped, fits=lambda n_id, n_k: True, max_n_id=None) -> dict:
    """§8 steps 1-5. fits(n_id, n_k) = the protected core fits 150 CPU-min; max_n_id = n_S + available unseen classes."""
    sp1 = 1.15 * max(s_id, s_ood)
    sp2 = None if p2_dropped else 1.15 * max(s2_id, s2_ood)
    n_id = n_id_for(s_id)
    low = power_p1(sp1, n_id, 56, 56) < 0.80 - TOL or (sp2 is not None and power_p2(sp2, n_id, 56, 56) < 0.80 - TOL)
    n_k = 72 if low else 56
    caps = []
    if max_n_id is not None and n_id > max_n_id:
        n_id = max(N_ID_RANGE[0], int(max_n_id))
        caps.append("unseen support classes")
    while not fits(n_id, n_k) and n_id > N_ID_RANGE[0]:
        n_id -= 1
        if "cpu" not in caps:
            caps.append("cpu")
    while not fits(n_id, n_k) and n_k > N_K_RANGE[0]:
        n_k = N_K_RANGE[0]
        caps.append("cpu (n_K)")
    pw = dict(p1_macro=power_p1(sp1, n_id, n_k, n_k), p1_id=power_p1(1.15 * s_id, n_id, macro=False),
              p1_id_at_003=power_p1(1.15 * s_id, n_id, macro=False, effect=0.03),
              p1_id_at_004=power_p1(1.15 * s_id, n_id, macro=False, effect=0.04))
    if sp2 is not None:
        pw.update(p2_d0=power_p2(sp2, n_id, n_k, n_k), p2_d002=power_p2(sp2, n_id, n_k, n_k, d2=0.02))
    return dict(sigma_plan_id=1.15 * s_id, sigma_plan_p1m=sp1, sigma_plan_2=sp2, n_id=n_id, n_k=n_k, n_neg=n_k,
                caps=caps, fits=fits(n_id, n_k), powers=pw)


def project_confirmation(rates: dict, n_id: int, n_k: int, n_neg: int, cfg: dict, cuts=()) -> dict:
    """Projected confirmation CPU-min per item at the G-P5 rates (PREREG §10.3; DESIGN §9.1). rates: step_s
    {step_name: s}, vi_s, cell_s {arm or arm@family: {"mean"}}. cfg: zs_steps {model: steps per run}, pgv {width,
    steps}, ft_steps, pg400 ("PGV" | "FT" | None), rext (bool). Returns {"items": {name: min}, "total", "protected"}."""
    cuts = set(cuts)
    sec = {f: n for f, n in SECONDARY_SIZES.items() if not (f == "H-KN" and "H-KN" in cuts)
           and not (f == "H-DYN" and "H-DYN" in cuts)}
    fams = {"H-ID": n_id, "H-K": n_k, "H-NEG": n_neg, **sec}
    n_p, n_all = n_id + n_k + n_neg, n_id + n_k + n_neg + sum(sec.values())
    st = rates["step_s"]

    def step(m, B, b):
        return st[LA.step_name(m, B, b)]

    def cell(arm, fam=None):
        cs = rates["cell_s"]
        v = cs.get(f"{arm}@{FA.base_family(fam)}" if fam else "") or cs.get(arm) or {"mean": 1.0}
        x = v["mean"]
        if fam == "H-KN" and arm.endswith("±") and f"{arm}@H-KN" not in cs:
            x *= 4.0  # H-KN menus: up to 3,600 targets vs H-NEG's 900 (not in DEV)
        return x
    it = {"test-oracle": 2 * 1.5 * n_all * rates["vi_s"] / 60.0}  # draw-test + oracle-test (incl. rejected draws)
    zs = cfg["zs_steps"]
    it["train"] = (3 * sum(zs[m] * step(m, 1, 256) for m in ("SF", "UVFA", "UVFA+aux"))
                   + zs["SF"] * step("SF", 1, 256)) / 60.0
    pw = {64: "PGV64", 128: "PGV128", 256: "PGV256"}[cfg["pgv"]["width"]]
    it["PGV_400"] = n_p * cfg["pgv"]["steps"] * step(pw, 64, 64) / 64 / 60.0
    ft_reps = 1 if ("FT-UVFA_400-1rep" in cuts and cfg["pg400"] != "FT") else 3
    it["FT-UVFA_400"] = ft_reps * n_p * cfg["ft_steps"] * step("UVFA", 64, 64) / 64 / 60.0
    if cfg.get("rext"):
        it["k-curve"] = 48 * 4000 * sum(step("PGV128", 64, b) for b in (64, 128, 256)) / 64 / 60.0
    else:
        it["k-curve"] = 48 * (2000 * step("PGV256", 64, 128) + 2500 * step("PGV256", 64, 256)) / 64 / 60.0
    ev = 0.0
    for f, n in fams.items():
        ev += 16 * n * (3 * sum(cell(a) for a in ZS_TEST) + cell("SF-GPI")) / 60.0  # + SF-shuffled
        ev += 16 * n * sum(cell(a, f) for a in ("ENUM-MIS±", "GREEDY", "STOP", "RAND", "ORACLE", "EXACT-GPI(L)",
                                               "BEST-SINGLE(L)")) / 60.0
        if f in FA.PRIMARY:
            ev += 16 * n * (cell("PGV") + 3 * cell("FT")) / 60.0
    it["eval"] = ev
    for arm in ("ENUM-H*±", "GNN-ENUM±"):
        for f in FA.PRIMARY:
            n = fams[f]
            if f == "H-NEG" and arm == "GNN-ENUM±" and "GNN-ENUM±@H-NEG" in cuts:
                continue
            if f == "H-NEG" and arm == "ENUM-H*±" and "ENUM-H*±@H-NEG-24" in cuts:
                n = min(n, 24)
            it[f"{arm}@{f}"] = 16 * n * cell(arm, f) / 60.0
    it["banks"] = 64 * n_all * 0.01 * len(ZS_TEST) / 60.0
    if "budget-curve" not in cuts:
        it["budget-curve"] = 2 * zs["SF"] * step("SF", 1, 256) / 60.0
    if "SF-FT_400" not in cuts:
        it["SF-FT_400"] = n_p * cfg["ft_steps"] * step("SF", 64, 64) / 64 / 60.0
    protected = sum(v for k, v in it.items() if k not in ("budget-curve", "SF-FT_400"))
    return dict(items=it, total=sum(it.values()), protected=protected)


def choose_cuts(project: Callable, cap: float = CAP_MIN) -> tuple:
    """Apply CUT_ORDER until project(cuts)['total'] <= cap; returns (cuts, projection)."""
    cuts = []
    pr = project(cuts)
    for c in CUT_ORDER:
        if le(pr["total"], cap):
            break
        cuts.append(c)
        pr = project(cuts)
    return cuts, pr


def pilot_report(P: Pilot, *, rates_override=None) -> dict:
    """Every pilot gate and the PILOT FILL values (§9). Raises SystemExit if the pilot is incomplete."""
    if P.plan_a() or P.plan_b():
        raise SystemExit("refusing: the pilot phases are not complete (pilot-a / pilot-b still have pending units)")
    fc = P.final_configs()
    if fc == "n6":
        raise SystemExit("G-P4(a) failed at 400 and 800 labels: the n = 6 fallback applies (pilot-a --n 6)")
    d, dev, cells = P.data(), P.dev(), P.cells
    base = P.base()
    rep = dict(n=d.n, digest_t2=P.digest, manifest_sha256=sha_file(P.manifest_path))

    def cell(arm, p, s, tag="-"):
        return cells.get(kj("cell", "pilot", arm, tag, p.key, int(s)))
    # Q7 (B2) and Q9 (B1) on DEV
    q7 = dict(oracle_all_zero=all(cell("ORACLE", p, s)["nreg"] == 0.0 for ps in dev.values() for p in ps
                                  for s in p.starts),
              stop_all_one=all(cell("STOP", p, s)["nreg"] == 1.0 for ps in dev.values() for p in ps for s in p.starts),
              greedy_identity=all(cell("GREEDY", p, s)["nreg"] == 1.0 for f in POSITIVE_DEV for p in dev.get(f, [])
                                  for s in p.starts))
    prim = [f for f in ("DEV-ID", "DEV-OOD:H-K", "DEV-OOD:H-NEG") if f in dev]
    q7["rand_macro"] = mean(family_mean(cells, "pilot", "RAND", "-", dev[f]) for f in prim)
    q7["ok"] = q7["oracle_all_zero"] and q7["stop_all_one"] and q7["greedy_identity"] and lt(q7["rand_macro"], 1.0)
    neg = dev.get("DEV-OOD:H-NEG", [])
    tied = Counter()
    for p in neg:
        for s in p.starts:
            dg = cell("ENUM-MIS±", p, s).get("diag") or {}
            tied["decisions"] += int(dg.get("decisions", 0))  # decisions with >= 2 legal moves (enum_policy diag)
            tied["all_tied"] += int(dg.get("all_tied", 0))
    q9_enum = family_mean(cells, "pilot", "ENUM-MIS±", "-", neg) if neg else float("nan")
    q9_greedy = family_mean(cells, "pilot", "GREEDY", "-", neg) if neg else float("nan")
    share = tied["all_tied"] / tied["decisions"] if tied["decisions"] else 0.0
    q9 = dict(enum_mis=q9_enum, greedy=q9_greedy, all_tied_share=share, decisions_multi=tied["decisions"],
              ok=bool(lt(q9_enum, q9_greedy) and share < 0.5),
              note="a failure triggers the enum_e05 audit (G-P1); with no bug found the guard-quality qualifier applies")
    qs = {q: P.check(q) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q8")}
    gp1 = dict(checks={q: dict(ok=bool(v["ok"]), value=v["value"]) for q, v in qs.items()}, Q7=q7, Q9=q9,
               ok_q1_q8=all(v["ok"] for v in qs.values()) and q7["ok"],
               ok=all(v["ok"] for v in qs.values()) and q7["ok"] and q9["ok"])
    g4 = P.gp4(400)
    gp4 = dict(at400=g4)
    if not g4["a"]["ok"]:
        gp4["at800"] = P.gp4(800)
    gfin = gp4.get("at800") or g4
    p1_invalid = not gfin["b"]["ok"]
    gp3, gp3b = P.gp3(), P.gp3b()
    L = fc["SF"][0]
    tags = {arm: P.zs_tag(ARM_MODEL[arm], *fc[ARM_MODEL[arm]]) for arm in ZS_PILOT}
    tags.update({a: P.ref_tag(a, False) for a in REF_PILOT})
    table = {}
    for arm, tag in list(tags.items()) + ([("EXACT-GPI(L)+base", "L+base")] if base else []):
        a = arm.replace("+base", "")
        table[arm] = {f: family_mean(cells, "pilot", a, tag, ps) for f, ps in dev.items()
                      if all(cells_done(cells, "pilot", a, tag, p) for p in ps)}
    for kind, grid in (("PGV", PGV_GRID + PGV_EXT), ("FT", FT_GRID + FT_EXT)):
        for g in grid:
            v = P.pg_mean(kind, g, 400, dev["DEV-ID"])
            if v is not None:
                table[f"{PG_ARM[kind]}:{g}"] = {"DEV-ID": v}
    pg = gp3["pg400"]
    pg_tag = None if pg is None else f"{gp3['grid'][pg]['cfg']}|k400"
    if pg is not None:
        for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG"):
            if f in dev and all(cells_done(cells, "pilot", pg, pg_tag, p) for p in dev[f]):
                table.setdefault(f"PG400={PG_ARM[pg]}", {})[f] = family_mean(cells, "pilot", pg, pg_tag, dev[f])

    def y(arm, tag, p):
        return parent_nreg(cells, "pilot", arm, tag, p)
    ood24 = [p for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG") for p in dev.get(f, [])]
    did = dev["DEV-ID"]

    def sd(xs):
        return stdev(xs) if len(xs) > 1 else 0.0
    s_id = sd([y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in did])
    s_ood = sd([y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in ood24])
    s2_id = s2_ood = None
    if pg is not None:
        s2_id = sd([y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in did])
        s2_ood = sd([y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in ood24])
    contrasts = {}
    for f, ps in dev.items():
        if all(cells_done(cells, "pilot", a, tags[a], p) for a in ("UVFA", "SF-GPI", "ENUM-MIS±") for p in ps):
            contrasts[f] = dict(D1=mean(y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in ps),
                                D_G=mean(y("SF-GPI", tags["SF-GPI"], p) - y("ENUM-MIS±", "-", p) for p in ps))
            if pg is not None and all(cells_done(cells, "pilot", pg, pg_tag, p) for p in ps):
                contrasts[f]["D2"] = mean(y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in ps)
    # G-P5 rates, cuts and sizing
    bench = rates_override or P.ctx.bench_override or P.checks.get(kj("bench"))
    pgv_cfg, ft_cfg = GRIDS[gp3["grid"]["PGV"]["cfg"]][1], GRIDS[gp3["grid"]["FT"]["cfg"]][1]
    n_lib = len(d.fam["TRAIN"]) + (d.n * d.n if base else 0)
    zs_steps = {m: fc[m][3] * n_lib * L / 256 for m in ("SF", "UVFA", "UVFA+aux")}  # selected epochs
    pcfg = dict(zs_steps=zs_steps, pgv=dict(width=pgv_cfg["width"], steps=pgv_cfg["keep"]), ft_steps=ft_cfg["keep"],
                pg400=pg, rext=gp3b["recipe"] == "R_ext")
    seen = {p.supp_u for p in d.fam["TRAIN"] if p.K in (2, 3, 4)}
    blocked = seen | {p.supp_u for p in did}
    unseen_avail = sum(FA.n_support_classes(d.n, K) for K in (2, 3, 4)) - len(blocked)
    n_s = min(P.ctx.sz()["hid_s_max"], len(seen))

    def fits(n_id, n_k):
        return le(choose_cuts(lambda c: project_confirmation(bench, n_id, n_k, n_k, pcfg, c))[1]["protected"], CAP_MIN)
    plan = size_plan(s_id, s_ood, s2_id or 0.0, s2_ood or 0.0, p2_dropped=pg is None, fits=fits,
                     max_n_id=n_s + unseen_avail)
    cuts, proj = choose_cuts(lambda c: project_confirmation(bench, plan["n_id"], plan["n_k"], plan["n_neg"], pcfg, c))
    units = LA.plan_costs(bench["step_s"], {f"cell:{a}": v["max"] for a, v in bench["cell_s"].items() if "@" not in a})
    unit_check = LA.validate_units(units)
    gp5 = dict(rates=bench, projection=proj, cuts=cuts, units=unit_check,
               ok=unit_check["ok"] and le(proj["total"], CAP_MIN))
    kc_arms = {str(k): (f"PGV-RX_{k}" if gp3b["recipe"] == "R_ext" else f"PGV_{k}") for k in (400, 1600, 6400)}
    det = dict(p1_comparator_invalid=p1_invalid, p2_dropped=pg is None, pg400=PG_ARM.get(pg),
               q9_failed_no_bug=not q9["ok"], kstar_descriptive=gp3b["kstar"] == "descriptive only",
               r_ext=gp3b["recipe"] == "R_ext", cut_items=[CUT_ORDER.index(c) + 1 for c in cuts], kcurve_arms=kc_arms,
               n=d.n)
    rep.update(G_P1=gp1, G_P2=P.gp2(), G_P3=gp3, G_P3b=gp3b, G_P4=gp4, G_P5=gp5,
               library=dict(size=n_lib, train=len(d.fam["TRAIN"]), base_policies=base), train_labels_per_goal=L,
               selected=fc,
               p1_comparator_invalid=p1_invalid, p2_dropped=pg is None, pg400=pg, kstar=gp3b["kstar"],
               kcurve_recipe=gp3b["recipe"], dev_nreg=table, dev_contrasts=contrasts,
               sigma_hat=dict(id=s_id, ood=s_ood, two_id=s2_id, two_ood=s2_ood), sizing=plan,
               h_id=dict(n_s=n_s, n_u=plan["n_id"] - n_s, available_seen=len(seen), available_unseen=unseen_avail),
               exclusions=P.m.get("exclusions"), reference_shas=LA.reference_shas(), determinations=det)
    rep["fill"] = fill_fields(rep)
    return rep


def fill_fields(r: dict) -> dict:
    """The §9.2 PILOT FILL rows, as text copied verbatim into PREREG_E05.md at T3."""
    g1, g4, g3, g3b, g5, sz = r["G_P1"], r["G_P4"], r["G_P3"], r["G_P3b"], r["G_P5"], r["sizing"]
    gf = g4.get("at800") or g4["at400"]
    f = dict(
        digests=f"T2 digest {r['digest_t2']}; PILOT_MANIFEST.json sha256 {r['manifest_sha256']}",
        n_labels=f"n = {r['n']}; TRAIN labels per goal {r['train_labels_per_goal']}",
        library=(f"{r['library']['size']} ({r['library']['train']}"
                 f"{' + ' + str(r['library']['size'] - r['library']['train']) if r['library']['base_policies'] else ''}"
                 f"); G-P2 DEV-ID EXACT-GPI(L) nreg {r['G_P2']['value']:.4f}"),
        G_P1=(f"Q7 ORACLE all 0: {g1['Q7']['oracle_all_zero']}, STOP all 1: {g1['Q7']['stop_all_one']}, RAND DEV "
              f"primary macro {g1['Q7']['rand_macro']:.4f}, GREEDY identity: {g1['Q7']['greedy_identity']}; Q9 ENUM-MIS± "
              f"{g1['Q9']['enum_mis']:.4f} vs GREEDY {g1['Q9']['greedy']:.4f}, all-tied share "
              f"{g1['Q9']['all_tied_share']:.4f} ({'pass' if g1['Q9']['ok'] else 'FAIL: audit'}); "
              + ", ".join(f"{q} {'pass' if v['ok'] else 'FAIL'}" for q, v in g1["checks"].items())),
        selection_G_P4=(f"selected {json.dumps(r['selected'])}; UVFA extension used: {gf['b']['cfg'] == 'ext'}; G-P4 "
                        f"acc½ delta SF {gf['a']['acc_delta']:.4f}, top-1 delta {gf['a']['top1_delta']:.4f}, UVFA acc½ "
                        f"delta {gf['b']['acc_delta']:.4f}, UVFA DEV-ID nreg {gf['b']['dev_nreg']:.4f}; "
                        f"{'P1 comparator-invalid' if r['p1_comparator_invalid'] else 'P1 valid'}"),
        G_P3=(f"PGV_400 {g3['grid']['PGV']['cfg']} DEV-ID {g3['grid']['PGV']['dev_id']:.4f}, FT-UVFA_400 "
              f"{g3['grid']['FT']['cfg']} {g3['grid']['FT']['dev_id']:.4f} (bar 0.50); extensions used: "
              f"{g3.get('extended', False)}; PG400 = {PG_ARM.get(r['pg400'], 'P2 dropped')}; DEV-OOD "
              + json.dumps({k: v for k, v in r["dev_nreg"].items() if k.startswith("PG400")})),
        G_P3b=(f"original k-curve {json.dumps(g3b['original'])} pass {g3b['original_ok']}; R_ext used: "
               f"{'rext' in g3b}; k* {g3b['kstar']}"),
        dev_nreg=json.dumps(r["dev_nreg"], sort_keys=True) + "; DEV contrasts " + json.dumps(r["dev_contrasts"]),
        sigma=json.dumps(r["sigma_hat"]) + "; plan " + json.dumps({k: sz[k] for k in ("sigma_plan_id", "sigma_plan_p1m",
                                                                                       "sigma_plan_2")}),
        sizes=(f"n_ID {sz['n_id']}, n_S {r['h_id']['n_s']}, n_U {r['h_id']['n_u']}, n_K = n_NEG {sz['n_k']}; available "
               f"unseen support classes {r['h_id']['available_unseen']}; caps {sz['caps']}"),
        powers=json.dumps(sz["powers"]),
        G_P5=(f"projected confirmation {g5['projection']['total']:.1f} CPU-min; cuts {g5['cuts']}; units "
              f"{json.dumps(g5['units'])}"),
        exclusions=json.dumps(r["exclusions"]), reference_shas=json.dumps(r["reference_shas"]),
        T3="<filled at T3: full digest incl. analyze_e05.py; test pass log>")
    return f


# ============================================================================= TEST side (§10.1 steps 5-8)

def pilot_report_or_refuse(ctx) -> dict:
    p = ctx.pdir / "PILOT_REPORT.json"
    if not p.exists():
        raise SystemExit("refusing: pilot/PILOT_REPORT.json does not exist")
    return read_json(p)


def manifest_check(ctx, k: int, manifest_sha: str) -> tuple:
    """(manifest or None, problem or None): refuses (SystemExit) unless --manifest-sha is a committed 64-hex sha and
    the manifest exists; a file whose sha differs is reported as a problem (VOID in oracle-test, refusal elsewhere)."""
    require_committed(manifest_sha, "TEST_MANIFEST.json sha256", ctx.decisions())
    p = ctx.adir(k) / "TEST_MANIFEST.json"
    if not p.exists():
        raise SystemExit(f"refusing: attempt {k} has no TEST_MANIFEST.json")
    sha = sha_file(p)
    if sha != manifest_sha:
        return None, f"TEST_MANIFEST.json sha256 {sha} != committed --manifest-sha {manifest_sha}"
    return read_json(p), None


def test_manifest(ctx, k: int, manifest_sha: str) -> dict:
    m, problem = manifest_check(ctx, k, manifest_sha)
    if problem:
        raise SystemExit(f"refusing: {problem}")
    return m


def oracle_report(ctx, k: int) -> dict | None:
    p = ctx.adir(k) / "ORACLE_TEST_REPORT.json"
    return read_json(p) if p.exists() else None


def require_oracle_pass(ctx, k: int, manifest_sha: str, digest: str) -> dict:
    r = oracle_report(ctx, k)
    if r is None or r.get("status") != "PASS":
        raise SystemExit(f"refusing: attempt {k} ORACLE_TEST_REPORT.json is "
                         f"{'missing' if r is None else repr(r.get('status'))}, not PASS (PREREG §2.2, B4)")
    if r.get("manifest_sha256") != manifest_sha or r.get("code_digest") != digest:
        raise SystemExit("refusing: ORACLE_TEST_REPORT.json names another manifest or code digest")
    return r


def first_learned(ctx, k: int) -> bool:
    return any(e.get("event") == "first-learned-test" for e in attempt_events(ctx, k))


def check_attempt_allowed(ctx, k: int) -> None:
    """Attempt k in 1..3; k > 1 only after attempt k - 1 ended VOID inside oracle-test (at most two redraws, §7.2)."""
    if not 1 <= k <= 3:
        raise SystemExit("refusing: attempts are 1, 2 or 3 (at most two redraws, §7.2)")
    if k > 1:
        r = oracle_report(ctx, k - 1)
        if r is None or r.get("status") != "VOID":
            raise SystemExit(f"refusing: attempt {k} needs attempt {k - 1} to have ended VOID in oracle-test")


def draw_test(ctx, args) -> int:
    k = int(args.attempt)
    check_attempt_allowed(ctx, k)
    prereg = require_prereg(ctx, args.expect_prereg_sha)
    cd = require_digest(ctx, "t3", args.expect_code_digest)
    rep = pilot_report_or_refuse(ctx)
    out = ctx.adir(k) / "TEST_MANIFEST.json"
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (create-only)")
    pm = read_json(ctx.pdir / "PILOT_MANIFEST.json")
    n = int(pm["n"])
    space = SP.get_space(n)
    try:
        excl = ctx.exclusions(space)
    except Exception as e:  # noqa: BLE001 - load_exclusions fails closed (§4.4 L6): refuse before draw-test
        raise SystemExit(f"refusing: exclusions do not load ({e!r}); restore or re-pin by a logged deviation") from None

    def run(sl):
        phi = SO.get_phi(space)
        reg = FA.Registry.from_parents(n, [FA.Parent.from_json(x) for x in flat_parents(pm)])
        sz, s, ov = rep["sizing"], ctx.sz(), ctx.sizes or {}
        reg = FA.draw_test(space, reg, k, ov.get("test_n_id", sz["n_id"]), ov.get("test_n_k", sz["n_k"]),
                           ov.get("test_n_neg", sz["n_neg"]), phi, excl, ctx.sizes)
        FA.assert_splits(reg)
        subset = FA.pgv_subset(reg, s["subset_per_family"], s["subset_per_stratum"])
        kmax = max(FA.SIZES["kcurve"])
        labels = {p.key: FA.draw_labels(space, p, kmax if p.key in subset else FA.SIZES["pergoal"], excl)[0]
                  for f in FA.PRIMARY for p in reg.parents[f]}
        man = dict(experiment="RCX-E05", kind="test", attempt=k, n=n, prereg_sha256=prereg, code_digest_t3=cd["digest"],
                   pilot_manifest_sha256=sha_file(ctx.pdir / "PILOT_MANIFEST.json"),
                   pilot_report_sha256=sha_file(ctx.pdir / "PILOT_REPORT.json"), sizes=sz,
                   cuts=list(rep["G_P5"]["cuts"]),
                   parents=[p.to_json() for f in FA.TEST_ORDER for p in reg.parents[f]],
                   rejections={f: dict(reg.rejections[f]) for f in FA.TEST_ORDER},
                   notes={f: reg.notes.get(f, {}) for f in FA.TEST_ORDER}, subset=subset, pergoal_labels=labels,
                   exclusions=None if excl is None else dict(calls=dict(excl.calls)),
                   streams=dict(test=f"20261242:test:{k}", subset="20261242:pgv-subset",
                                starts=f"20261244:starts:test:{k}:<key>", labels="20261249:labels:<key>"),
                   provenance=provenance(ctx, "draw-test", cd))
        sha = write_create_only(out, man)
        log_attempt(ctx, "manifest", attempt=k, file="TEST_MANIFEST.json", sha256=sha,
                    note="commit this sha to DECISIONS before any arm is evaluated on TEST")
        print(f"TEST_MANIFEST.json sha256 {sha}", flush=True)
        return True
    return drive(ctx, "draw-test", lambda: [] if out.exists() else [Unit("draw-test", 300.0, run)], attempt=k, cd=cd,
                 args=vars(args))


def rcx_test_sets_n7(root: Path) -> dict:
    """§4.4 item 4: RCX evaluation sets as DESIGN §4.5 lists them (E07 test, E07b TEST-B, E11a, E11b exec) and their
    n = 7 instance counts (a missing file counts as a failure); E11b's test_id/ood_small splits are reported beside."""
    def load(rel):
        p = Path(root) / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    out, extra = {}, {}
    e07 = load("runs/RCX-E07/main/data/instances.json")
    out["E07 test"] = None if e07 is None else sum(r.get("split") == "test" and len(r["objs"]) == 7 for r in e07)
    e07b = load("runs/RCX-E07/e07b/data/testb_instances.json")
    out["E07b TEST-B"] = None if e07b is None else sum(len(r["objs"]) == 7 for r in e07b)
    e11a = load("runs/RCX-E11/main/a/instances.json")
    out["E11a"] = None if e11a is None else sum(len(r["objs"]) == 7 for r in e11a)
    e11b = load("runs/RCX-E11/main/b/data/instances.json")
    out["E11b exec"] = None if e11b is None else sum(r.get("split") == "exec" and len(r["objs"]) == 7 for r in e11b)
    if e11b is not None:
        extra = {s: sum(r.get("split") == s and len(r["objs"]) == 7 for r in e11b) for s in ("test_id", "ood_small")}
    return dict(counts=out, e11b_other_splits_n7=extra, ok=all(v == 0 for v in out.values()))


class TestState:
    """Attempt-k state: Data over TEST_MANIFEST (+ the pilot library), keyed logs, the pilot report."""

    def __init__(self, ctx, k, manifest, digest):
        self.ctx, self.k, self.m, self.digest, self.ad = ctx, k, manifest, digest, ctx.adapters
        self.dir = ctx.adir(k)
        self.d = Data(ctx, manifest, ctx.pdir)
        self.oc = KeyedLog(self.dir / "oracle_cells.jsonl")
        self.cells = KeyedLog(self.dir / "cells.jsonl")
        self.banks = KeyedLog(self.dir / "banks.jsonl")
        self.fits = KeyedLog(self.dir / "fits.jsonl")
        self.rep = read_json(ctx.pdir / "PILOT_REPORT.json")
        self.base = bool(self.rep["library"]["base_policies"])
        self.cuts = set(self.rep["G_P5"]["cuts"])
        self.split = f"test:{k}"

    def families(self) -> list:
        return [f for f in FA.TEST_ORDER if f in self.d.fam and not (f == "H-KN" and "H-KN" in self.cuts)
                and not (f == "H-DYN" and "H-DYN" in self.cuts)]

    def parents(self, families=None) -> list:
        return [(f, p) for f in (families or [f for f in FA.TEST_ORDER if f in self.d.fam]) for p in self.d.fam[f]]


def oracle_test(ctx, args) -> int:
    """§10.4: every VOID check before any learned-arm TEST computation -> ORACLE_TEST_REPORT.json (PASS or VOID)."""
    k = int(args.attempt)
    if oracle_report(ctx, k) is not None:
        raise SystemExit("refusing: ORACLE_TEST_REPORT.json exists (create-only)")
    if first_learned(ctx, k):
        raise SystemExit("refusing: a learned-arm TEST computation already ran in this attempt; no VOID exists now")
    m, problem = manifest_check(ctx, k, args.manifest_sha)
    require_committed(args.code_digest, "T3 code digest", ctx.decisions())
    cd = code_digest("t3", ctx.root)
    custody = {}
    if problem:
        custody["manifest"] = dict(value=problem, status="VOID")
    if cd["digest"] != args.code_digest or cd["pins_bad"]:
        custody["code_digest"] = dict(value=dict(now=cd["digest"], expected=args.code_digest, pins_bad=cd["pins_bad"],
                                                 missing=cd["missing"]), status="VOID")
    elif m is not None and m.get("code_digest_t3") != cd["digest"]:
        custody["code_digest"] = dict(value="the manifest names another T3 digest", status="VOID")
    if custody:  # custody fault between T4 and PASS: VOID without computing anything
        return write_oracle_report(ctx, k, args, cd, custody)
    T = TestState(ctx, k, m, cd["digest"])

    def unit_parent(f, p):
        def run(sl):
            d, out = T.d, {}
            tab, mask = d.table(p), d.mask(p)
            q3 = 0
            for s0 in p.starts:
                ep = SO.run_episode(d.space, int(s0), SO.oracle_policy(tab), mask)
                v = float(tab.stop[tab.end[s0]]) - tab.c * float(tab.T[s0])
                q3 += int(not (ep.final == tab.end[s0] and ep.T == tab.T[s0] and not ep.loop and v == tab.V[s0]))
            out["q3_bad"] = q3
            if p.dyn == "free":
                out["q4_bad"] = q4_parent(d, p, T.base)
            out["nreg"] = {arm: [SO.evaluate(d.space, tab, int(s0), ref_policy(arm, d, p, int(s0), T.split, T.base,
                                                                              T.ad)[0],
                                             mask)["nreg"] for s0 in p.starts] for arm in ("ORACLE", "STOP", "GREEDY",
                                                                                           "RAND")}
            T.oc.add(kj("oracle", p.key), dict(family=f, parent=p.key, **out), T.digest)
            return True
        return run

    def unit_report(sl):
        checks, rows = {}, {r["parent"]: r for r in T.oc.rows.values()}
        q3 = sum(r["q3_bad"] for r in rows.values())
        q4 = sum(r.get("q4_bad", 0) for r in rows.values())
        checks["Q3"] = dict(value=q3, status="PASS" if q3 == 0 else "VOID")
        checks["Q4"] = dict(value=q4, status="PASS" if q4 == 0 else "VOID",
                            note="free-dynamics parents (H-DYN excluded: library policies are not height-3 policies)")
        fam = defaultdict(list)
        for r in rows.values():
            fam[r["family"]].append(r)
        o0 = all(v == 0.0 for r in rows.values() for v in r["nreg"]["ORACLE"])
        s1 = all(v == 1.0 for r in rows.values() for v in r["nreg"]["STOP"])
        g1 = all(v == 1.0 for f in POSITIVE_TEST for r in fam.get(f, []) for v in r["nreg"]["GREEDY"])
        rand = mean(mean(mean(r["nreg"]["RAND"]) for r in fam[f]) for f in FA.PRIMARY)
        checks["Q7"] = dict(value=dict(oracle_all_zero=o0, stop_all_one=s1, greedy_identity=g1, rand_primary_macro=rand),
                            status="PASS" if (o0 and s1 and g1 and lt(rand, 1.0)) else "VOID")
        pilot = [FA.Parent.from_json(x) for x in flat_parents(read_json(ctx.pdir / "PILOT_MANIFEST.json"))]
        reg = FA.Registry.from_parents(T.d.n, pilot + [p for _, p in T.parents()])
        sp = {kk: [bool(v[0]), v[1]] for kk, v in FA.check_splits(reg).items()}
        checks["Q8:splits"] = dict(value=sp, status="PASS" if all(v[0] for v in sp.values()) else "VOID")
        try:
            excl = ctx.exclusions(T.d.space)
            exc = Counter()
            for _, p in T.parents():
                v = FA.verify_parent(T.d.space, T.d.phi, p, excl, labels=T.m["pergoal_labels"].get(p.key, ()))
                for kk in ("excluded_starts", "excluded_bank", "excluded_labels"):
                    exc[kk] += int(v.get(kk, 0))
                exc["invalid"] += int(not v["starts_ok"] or not v["bank_ok"])
            checks["Q8:exclusions"] = dict(value=dict(exc), status="PASS" if not sum(exc.values()) else "VOID")
        except Exception as e:  # noqa: BLE001 - exclusion basis changed (load_exclusions fails closed, §4.4)
            checks["Q8:exclusions"] = dict(value=f"exclusion basis changed: {e!r}", status="VOID")
        n7 = rcx_test_sets_n7(ctx.root)
        checks["Q8:n7"] = dict(value=n7, status="PASS" if n7["ok"] else "VOID")
        s = ctx.sz()
        sub = FA.pgv_subset(FA.Registry.from_parents(T.d.n, [p for _, p in T.parents(FA.PRIMARY)]),
                            s["subset_per_family"], s["subset_per_stratum"])
        checks["Q8:subset"] = dict(value=dict(as_drawn=sub == T.m["subset"]), status="PASS" if sub == T.m["subset"]
                                   else "VOID")
        return write_oracle_report(ctx, k, args, cd, dict(manifest=dict(value=args.manifest_sha, status="PASS"),
                                                          code_digest=dict(value=cd["digest"], status="PASS"),
                                                          **checks)) == 0

    def plan():
        if oracle_report(ctx, k) is not None:
            return []
        U = [Unit(f"oracle:{p.key}", 16 * 4 * 0.2 + 1.0, unit_parent(f, p)) for f, p in T.parents()
             if kj("oracle", p.key) not in T.oc]
        return U or [Unit("oracle-test:report", 120.0, unit_report)]
    return drive(ctx, "oracle-test", plan, attempt=k, cd=cd, args=vars(args))


def write_oracle_report(ctx, k, args, cd, checks) -> int:
    status = "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "VOID"
    rep = dict(attempt=k, status=status, checks=checks, manifest_sha256=args.manifest_sha, code_digest=args.code_digest,
               provenance=provenance(ctx, "oracle-test", cd))
    sha = write_create_only(ctx.adir(k) / "ORACLE_TEST_REPORT.json", rep)
    log_attempt(ctx, "oracle-test", attempt=k, status=status, sha256=sha,
                void=[c for c, v in checks.items() if v["status"] != "PASS"])
    if status == "VOID":
        print(f"attempt {k}: VOID " + ("(third VOID: INCONCLUSIVE + design review, §7.2)" if k >= 3 else
                                      f"-> fix, then draw-test --attempt {k + 1}"), flush=True)
    return 0


def confirm_models_dir(ctx) -> Path:
    return ctx.runs / "confirm" / "models"


def train_replicates(ctx, args) -> int:
    """Confirmation replicates 20261246-48 of SF, UVFA, UVFA+aux at the PILOT-FILL configs, once each; SF-shuffled
    (20261246). They never touch TEST here."""
    require_prereg(ctx, args.expect_prereg_sha)
    cd = require_digest(ctx, "t3", args.expect_code_digest)
    rep = pilot_report_or_refuse(ctx)
    log = KeyedLog(confirm_models_dir(ctx) / "fits.jsonl")
    P = Pilot(ctx, None)
    L, base, sel = rep["train_labels_per_goal"], rep["library"]["base_policies"], rep["selected"]
    jobs = [(m, s) for m in ("SF", "UVFA", "UVFA+aux") for s in SEEDS["replicates"]]
    jobs.append(("SF-shuffled", SEEDS["replicates"][0]))
    path = P.data_path(L, base)
    if not path.exists():
        raise SystemExit(f"refusing: the pilot's training data {path.name} is missing")

    def unit(model, seed):
        def run(sl):
            _, _, cfg, epoch = sel["SF" if model == "SF-shuffled" else model]
            spec = dict(ZS_BASE if cfg == "base" else ZS_EXT, epochs=int(epoch), keep=(int(epoch),))
            res = ctx.adapters.train_zero_shot(model, P.data(), dict(np.load(path)), seed=seed, cfg=spec,
                                               out_dir=confirm_models_dir(ctx) / f"{model}_s{seed}", safe_point=_sp(sl))
            if res is None:
                return False
            p = str(res[int(epoch)])
            log.add(kj("replicate", model, seed), dict(model=model, seed=seed, cfg=cfg, epoch=int(epoch), path=p,
                                                       sha=_sha_path(p), labels=L, base=base), cd["digest"])
            return True
        return run
    plan = lambda: [Unit(f"train:{m}:s{s}", 600.0, unit(m, s)) for m, s in jobs  # noqa: E731
                    if kj("replicate", m, s) not in log]
    return drive(ctx, "train", plan, attempt="replicates", cd=cd, args=vars(args))


def replicate_models(ctx) -> dict:
    """{(model, seed): path} of the confirmation replicates (never selection models)."""
    out = {}
    for r in KeyedLog(confirm_models_dir(ctx) / "fits.jsonl").rows.values():
        if int(r["seed"]) not in SEEDS["replicates"]:
            raise SystemExit("refusing: a non-replicate model is registered for TEST (selection models never touch "
                             "TEST, §2.3)")
        out[(r["model"], int(r["seed"]))] = r["path"]
    return out


def learned_test_gate(ctx, args) -> tuple:
    """pgv/ft/eval/extras entry: committed manifest sha, T3 digest, ORACLE_TEST_REPORT PASS for this manifest."""
    k = int(args.attempt)
    m = test_manifest(ctx, k, args.manifest_sha)
    cd = require_digest(ctx, "t3", args.code_digest)
    if m.get("code_digest_t3") != cd["digest"]:
        raise SystemExit("refusing: the manifest's code digest differs from the current T3 digest")
    require_oracle_pass(ctx, k, args.manifest_sha, cd["digest"])
    return k, cd, TestState(ctx, k, m, cd["digest"])


def mark_first_learned(ctx, k, phase):
    if not first_learned(ctx, k):
        log_attempt(ctx, "first-learned-test", attempt=k, phase=phase)


def pg_fit_specs(T: TestState, kind: str) -> list:
    """[(fit name, fit kind, cfg, parents, k, analysis arm, replicate, init model)] of the pgv / ft phases: PGV_400 at
    the G-P3 PGV config on every primary parent (in every case), the k-curve (original or R_ext, G-P3b) on the
    48-parent subset; FT-UVFA_400 at the G-P3 FT config from each UVFA replicate (1 replicate under cut item 5)."""
    rep, prim = T.rep, [p for _, p in T.parents(FA.PRIMARY)]
    sub = set(T.m["subset"])
    out = []
    if kind == "PGV":
        g = rep["G_P3"]["grid"]["PGV"]["cfg"]
        out.append((f"PGV_400:{g}", "PGV", GRIDS[g][1], prim, 400, "PGV_400", None, None))
        for kk, g in (R_EXT if rep["kcurve_recipe"] == "R_ext" else KCURVE).items():
            arm = rep["determinations"]["kcurve_arms"][str(kk)]
            out.append((f"kcurve:{g}", "PGV", GRIDS[g][1], [p for p in prim if p.key in sub], kk, arm, None, None))
    else:
        g = rep["G_P3"]["grid"]["FT"]["cfg"]
        n_rep = 1 if ("FT-UVFA_400-1rep" in T.cuts and rep["pg400"] != "FT") else 3
        reps = replicate_models(T.ctx)
        for seed in SEEDS["replicates"][:n_rep]:
            out.append((f"FT:{g}:s{seed}", "FT", GRIDS[g][1], prim, 400, "FT-UVFA_400", seed, reps[("UVFA", seed)]))
    return out


def pg_jobs_test(T: TestState, kind: str, ps, kk: int, rep_seed, init) -> list:
    """fit_pergoal jobs on TEST (seeds: module docstring); SF-FT targets psi, PGV and FT V*."""
    jobs = []
    for p in ps:
        idx = np.asarray(T.m["pergoal_labels"][p.key][:kk], dtype=np.int64)
        tab = T.d.table(p)
        seed = {"PGV": lambda: FA.derive(SEEDS["pergoal"], f"pgv-init:{p.key}:{kk}"),
                "FT": lambda: FA.derive(int(rep_seed), f"ft:{p.key}"),
                "SF-FT": lambda: FA.derive(SEEDS["pergoal"], f"sf-ft:{p.key}")}[kind]()
        job = dict(key=p.key, states=idx, v=tab.V[idx], stab=FA.stabilizer(np.asarray(p.w), T.d.n), seed=seed,
                   w=p.weights(), c=p.c)
        if kind == "SF-FT":
            job["psi"] = SO.psi(T.d.phi, tab, idx)
        if init is not None:
            job["init"] = init
        jobs.append(job)
    return jobs


def pgv_ft(ctx, args, kind) -> int:
    k, cd, T = learned_test_gate(ctx, args)
    if kind == "FT":
        replicate_models_or_refuse(ctx, ("UVFA",))
    specs = pg_fit_specs(T, kind)

    def unit(name, fkind, cfg, ps, kk, arm, seed, init):
        def run(sl):
            mark_first_learned(ctx, k, kind.lower())
            res = T.ad.fit_pergoal(fkind, T.d, pg_jobs_test(T, fkind, ps, kk, seed, init), cfg,
                                   out_dir=T.dir / "models" / f"{fname(name)}_k{kk}", safe_point=_sp(sl))
            if res is None:
                return False
            T.fits.add(kj("fit", name, kk), dict(kind=fkind, name=name, k=kk, arm=arm, rep=seed,
                                                 models={str(a): str(b) for a, b in res.items()}), T.digest)
            return True
        return run
    plan = lambda: [Unit(f"fit:{s[0]}:k{s[4]}", 0.02 * s[2]["steps"] * max(1, len(s[3]) // 16), unit(*s))  # noqa
                    for s in specs if kj("fit", s[0], s[4]) not in T.fits]
    return drive(ctx, kind.lower(), plan, attempt=k, cd=cd, args=vars(args))


def replicate_models_or_refuse(ctx, models=("SF", "UVFA", "UVFA+aux")) -> dict:
    reps = replicate_models(ctx)
    lost = [f"{m}@{s}" for m in models for s in SEEDS["replicates"] if (m, s) not in reps]
    if "SF" in models and ("SF-shuffled", SEEDS["replicates"][0]) not in reps:
        lost.append("SF-shuffled")
    if lost:
        raise SystemExit(f"refusing: confirmation replicates missing {lost} (run train first)")
    return reps


TEST_REF_ARMS = ("ORACLE", "STOP", "RAND", "GREEDY", "EXACT-GPI(L)", "BEST-SINGLE(L)", "ENUM-MIS±")
NO_BANK = ("RAND", "BEST-SINGLE(L)", "ENUM-H*±", "GNN-ENUM±")  # start-dependent, or too costly per bank state


class Spec(NamedTuple):
    family: str
    p: object
    arm: str      # policy arm
    tag: str      # cell tag
    how: tuple    # ("ref",) | ("zs", model path, library or None) | ("pg", model path)
    xarm: str     # analysis arm name (evals/*.json)
    rep: int | None


def eval_specs(T: TestState) -> list:
    """Every (family, parent, arm) of the eval phase, parent-major: the references and ENUM± arms (ENUM-H*± and
    GNN-ENUM± on the primary families, cut items 4 and 6; rev 1 / rev 2 ENUM-MIS on H-NEG), the zero-shot arms x 3
    replicates and SF-shuffled on every retained family, the per-goal fits on the primary families."""
    out, reps, cuts = [], replicate_models(T.ctx), T.cuts
    fits = list(T.fits.rows.values())
    neg = [p.key for p in T.d.fam.get("H-NEG", [])]
    s0 = SEEDS["replicates"][0]
    for f in T.families():
        for p in T.d.fam[f]:
            out += [Spec(f, p, arm, Pilot.ref_tag(arm, T.base), ("ref",), arm, None) for arm in TEST_REF_ARMS]
            if f in FA.PRIMARY:
                for arm in ("ENUM-H*±", "GNN-ENUM±"):
                    if f == "H-NEG" and arm == "GNN-ENUM±" and "GNN-ENUM±@H-NEG" in cuts:
                        continue
                    if f == "H-NEG" and arm == "ENUM-H*±" and "ENUM-H*±@H-NEG-24" in cuts and neg.index(p.key) >= 24:
                        continue
                    out.append(Spec(f, p, arm, "-", ("ref",), arm, None))
            if f == "H-NEG":
                out += [Spec(f, p, arm, "-", ("ref",), arm, None) for arm in ("ENUM-MIS (rev 1)", "ENUM-MIS± (rev 2)")]
            out += [Spec(f, p, arm, f"{ARM_MODEL[arm]}|s{seed}", ("zs", reps[(ARM_MODEL[arm], seed)], None), arm, seed)
                    for arm in ZS_TEST for seed in SEEDS["replicates"]]
            out.append(Spec(f, p, "SF-shuffled", f"SF-shuffled|s{s0}", ("zs", reps[("SF-shuffled", s0)], None),
                            "SF-shuffled", s0))
            if f in FA.PRIMARY:
                out += [Spec(f, p, r["kind"], f"{r['name']}|k{r['k']}", ("pg", r["models"][p.key]), r["arm"], r["rep"])
                        for r in fits if p.key in r["models"]]
    return out


def extras_specs(T: TestState) -> list:
    """The extras' evaluation (primary families): budget-curve SF-GPI (library = its training goals) and UVFA,
    SF-FT_400 (C = {w'})."""
    out, s0 = [], SEEDS["replicates"][0]
    rows = list(KeyedLog(T.dir / "extras_models.jsonl").rows.values())
    for f, p in T.parents(FA.PRIMARY):
        for r in rows:
            if r["what"] == "budget":
                lib = SimpleNamespace(W=np.asarray(r["lib_W"], dtype=np.float64), c=np.asarray(r["lib_c"]))
                out.append(Spec(f, p, r["arm"], f"{r['name']}|s{s0}", ("zs", r["path"], lib), f"{r['arm']} [{r['name']}]",
                                s0))
            elif p.key in r.get("models", {}):
                out.append(Spec(f, p, "SF-FT_400", f"SF-FT_400|s{s0}", ("zs", r["models"][p.key], None), "SF-FT_400", s0))
    return out


def spec_policy(T: TestState, s: Spec, s0: int):
    d = T.d
    if s.how[0] == "ref":
        return ref_policy(s.arm, d, s.p, s0, T.split, T.base, T.ad)
    lib = s.how[2] if s.how[0] == "zs" else None
    return d.cache(s.p, ("pol", s.arm, s.tag), lambda: learned_policy(s.arm, s.how[1], d, s.p, T.base, T.ad, lib)), None


def eval_phase(ctx, args, extras=False) -> int:
    k, cd, T = learned_test_gate(ctx, args)
    phase = "extras" if extras else "eval"
    replicate_models_or_refuse(ctx)
    if not extras:
        lost = [s[0] for kind in ("PGV", "FT") for s in pg_fit_specs(T, kind) if kj("fit", s[0], s[4]) not in T.fits]
        if lost:
            raise SystemExit(f"refusing: per-goal fits missing {lost} (run pgv and ft first, §10.1 step 8)")
    elif {"budget-curve", "SF-FT_400"} <= T.cuts:
        print("extras: both items are cut (§10.3 items 1-2); nothing to do", flush=True)
        return 0

    def unit(s: Spec):
        def run(sl):
            mark_first_learned(ctx, k, phase)
            return eval_cells(T.cells, T.d, sl, split=T.split, family=s.family, p=s.p, arm=s.arm, tag=s.tag,
                              policy_fn=lambda s0: spec_policy(T, s, s0), digest=T.digest, xarm=s.xarm, rep=s.rep)
        return run

    def bank(s: Spec):
        def run(sl):
            mark_first_learned(ctx, k, phase)
            d = T.d
            vhat = None
            if s.how[0] == "pg" or (s.how[0] == "zs" and s.arm in ("SF-noGPI", "UVFA", "UVFA+aux", "SF-FT_400")):
                vhat = lambda S: model_values(T.ad, s.how[1], d, s.p, S)  # noqa: E731
            return bank_unit(T.banks, d, split=T.split, p=s.p, arm=s.arm, tag=s.tag,
                             choose=spec_policy(T, s, int(s.p.bank[0]))[0], vhat=vhat, digest=T.digest, xarm=s.xarm,
                             rep=s.rep)
        return run

    def plan():
        U = extras_train_units(ctx, T, k) if extras else []
        if U:
            return U
        specs = extras_specs(T) if extras else eval_specs(T)
        for s in specs:
            if not cells_done(T.cells, T.split, s.arm, s.tag, s.p):
                U.append(Unit(f"{phase}:{s.family}:{s.arm}:{s.tag}:{s.p.key}", 16.0, unit(s)))
            if s.arm not in NO_BANK and s.p.bank and kj("bank", T.split, s.arm, s.tag, s.p.key) not in T.banks:
                U.append(Unit(f"bank:{s.family}:{s.arm}:{s.tag}:{s.p.key}", 5.0, bank(s)))
        if U or not export_pending(T, specs):
            return U
        return [Unit(f"export:{phase}", 30.0, lambda sl: export_evals(T, specs, args) >= 0)]
    return drive(ctx, phase, plan, attempt=k, cd=cd, args=vars(args))


def extras_train_units(ctx, T: TestState, k) -> list:
    """Descriptive extras (cut items 1-2): the budget curve (SF and UVFA, replicate 20261246, at 16 x 400 and 64 x 1,600
    TRAIN labels, the selected epochs; the SF-GPI library = its training goals) and SF-FT_400 (SF replicate 20261246
    fine-tuned on the psi targets of the 400 per-goal labels at FT's selected grid point)."""
    rep, s0 = T.rep, SEEDS["replicates"][0]
    ex = KeyedLog(T.dir / "extras_models.jsonl")
    P = Pilot(ctx, None)
    U = []
    if "budget-curve" not in T.cuts:
        for name, labels, n_goals in (("16x400", 400, 16), ("64x1600", 1600, None)):
            dpath = T.dir / "extras" / f"train_data_{name}.npz"
            if not dpath.exists():
                U.append(Unit(f"extras:data:{name}", 60.0, P.unit_train_data(labels, False, dpath, n_goals)))
                continue
            for model, arm in (("SF", "SF-GPI"), ("UVFA", "UVFA")):
                if kj("budget", name, model) not in ex:
                    U.append(Unit(f"extras:train:{model}:{name}", 600.0, _budget_unit(ctx, T, k, ex, P, model, arm,
                                                                                      name, dpath)))
    if "SF-FT_400" not in T.cuts and kj("sf-ft") not in ex:
        g = rep["G_P3"]["grid"]["FT"]["cfg"]
        init = replicate_models(ctx)[("SF", s0)]
        ps = [p for _, p in T.parents(FA.PRIMARY)]

        def run_ft(sl):
            mark_first_learned(ctx, k, "extras")
            res = T.ad.fit_pergoal("SF-FT", T.d, pg_jobs_test(T, "SF-FT", ps, 400, s0, init), GRIDS[g][1],
                                   out_dir=T.dir / "extras" / "SF-FT_400", safe_point=_sp(sl))
            if res is None:
                return False
            ex.add(kj("sf-ft"), dict(what="sf-ft", grid=g, models={str(a): str(b) for a, b in res.items()}), T.digest)
            return True
        U.append(Unit("extras:SF-FT_400", 0.02 * GRIDS[g][1]["steps"] * max(1, len(ps) // 16), run_ft))
    return U


def _budget_unit(ctx, T, k, ex, P, model, arm, name, dpath):
    def run(sl):
        mark_first_learned(ctx, k, "extras")
        _, _, cfg, epoch = T.rep["selected"][model]
        spec = dict(ZS_BASE if cfg == "base" else ZS_EXT, epochs=int(epoch), keep=(int(epoch),))
        data = dict(np.load(dpath))
        res = T.ad.train_zero_shot(model, P.data(), data, seed=SEEDS["replicates"][0], cfg=spec,
                                   out_dir=T.dir / "extras" / f"{model}_{name}", safe_point=_sp(sl))
        if res is None:
            return False
        first = [int(np.flatnonzero(data["goal"] == i)[0]) for i in range(int(data["goal"].max()) + 1)]
        ex.add(kj("budget", name, model), dict(what="budget", name=name, model=model, arm=arm, path=str(res[int(epoch)]),
                                               lib_W=data["W"][first].tolist(), lib_c=data["c"][first].tolist()),
               T.digest)
        return True
    return run


def eval_file(T: TestState, xarm: str, rep) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", xarm)
    return T.dir / "evals" / f"{safe}_{fname(xarm)[:6]}__{rep if rep is not None else 'det'}.json"


def export_pending(T: TestState, specs) -> bool:
    return any(not eval_file(T, s.xarm, s.rep).exists() for s in specs)


def export_evals(T: TestState, specs, args) -> int:
    """evals/<arm>__<replicate>.json (create-only; analyze_e05 input contract) for every (analysis arm, replicate)
    group of `specs`: {manifest_sha256, code_digest, arm, replicate, cells: {parent key: {"nreg": [per start, manifest
    order], "loop": [...], and from the bank "regret1", "top1", "value_abs_err" where measured}}}."""
    groups = defaultdict(list)
    for s in specs:
        groups[(s.xarm, s.rep)].append(s)
    written = 0
    for (xarm, rep), ss in groups.items():
        path = eval_file(T, xarm, rep)
        if path.exists():
            continue
        body = {}
        for s in ss:
            cs = [T.cells.get(kj("cell", T.split, s.arm, s.tag, s.p.key, int(x))) for x in s.p.starts]
            cell = dict(nreg=[c["nreg"] for c in cs], loop=[float(c["loop"]) for c in cs])
            if all("diag" in c for c in cs):  # ENUM± tie diagnostics per start
                cell.update({k: [float(c["diag"].get(k, 0)) for c in cs] for k in ("decisions", "all_tied")})
            b = T.banks.get(kj("bank", T.split, s.arm, s.tag, s.p.key))
            if b is not None and b.get("rows"):
                rows = b["rows"]
                if "regret1" in rows[0]:
                    cell.update(regret1=[r["regret1"] for r in rows], top1=[float(r["top1"]) for r in rows])
                if "vhat" in rows[0]:
                    cell["value_abs_err"] = [abs(r["vhat"] - r["vstar"]) for r in rows]
            body[s.p.key] = cell
        write_create_only(path, dict(manifest_sha256=args.manifest_sha, code_digest=T.digest, arm=xarm, replicate=rep,
                                     cells=body, attempt=T.k))
        written += 1
    log_attempt(T.ctx, "evals-exported", attempt=T.k, files=written)
    return written


# ============================================================================= CLI

def pilot_entry(ctx, args, phase) -> int:
    cd = require_digest(ctx, "t2", args.expect_code_digest)
    if ctx.n == 6:  # the n = 6 fallback only after G-P4(a) failed at 400 and 800 labels at n = 7
        P7 = Pilot(Ctx(runs=ctx.runs, root=ctx.root, n=7, env=ctx.env, decisions=ctx.decisions,
                       exclusions=ctx.exclusions, adapters=ctx.adapters, sizes=ctx.sizes), cd)
        if not (P7.manifest_path.exists() and P7.final_configs() == "n6"):
            raise SystemExit("refusing: --n 6 is the G-P4 fallback; the n = 7 pilot has not failed G-P4(a) twice")
    P = Pilot(ctx, cd)
    if phase == "pilot-a":
        return drive(ctx, "pilot-a", P.plan_a, attempt="pilot", cd=cd, args=vars(args))
    if phase == "pilot-b":
        if P.plan_a():
            raise SystemExit("refusing: pilot-a is not complete")
        if P.final_configs() == "n6":
            raise SystemExit("G-P4(a) failed at 400 and 800 labels: run pilot-a --n 6 (fallback)")
        return drive(ctx, "pilot-b", P.plan_b, attempt="pilot", cd=cd, args=vars(args))
    if (ctx.pdir / "PILOT_REPORT.json").exists():
        raise SystemExit("refusing: pilot/PILOT_REPORT.json exists (create-only)")
    rep = pilot_report(P)
    rep["provenance"] = provenance(ctx, "pilot-report", cd)
    sha = write_create_only(ctx.pdir / "PILOT_REPORT.json", rep)
    log_attempt(ctx, "pilot-report", attempt="pilot", sha256=sha)
    print(json.dumps(rep["fill"], indent=1, ensure_ascii=False))
    print(f"PILOT_REPORT.json sha256 {sha}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(RUNS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("code-digest")
    c.add_argument("--stage", choices=("t2", "t3"), default="t2")
    for name in ("pilot-a", "pilot-b", "pilot-report"):
        a = sub.add_parser(name)
        a.add_argument("--expect-code-digest", required=True)
        a.add_argument("--n", type=int, choices=(6, 7), default=7)
    a = sub.add_parser("draw-test")
    a.add_argument("--attempt", type=int, required=True)
    a.add_argument("--expect-prereg-sha", required=True)
    a.add_argument("--expect-code-digest", required=True)
    a = sub.add_parser("train")
    a.add_argument("--expect-prereg-sha", required=True)
    a.add_argument("--expect-code-digest", required=True)
    for name in ("oracle-test", "pgv", "ft", "eval", "extras"):
        a = sub.add_parser(name)
        a.add_argument("--attempt", type=int, required=True)
        a.add_argument("--manifest-sha", required=True)
        a.add_argument("--code-digest", required=True)
    return ap


def run_command(ctx, args) -> int:
    if args.cmd in ("pilot-a", "pilot-b", "pilot-report"):
        return pilot_entry(ctx, args, args.cmd)
    if args.cmd == "draw-test":
        return draw_test(ctx, args)
    if args.cmd == "oracle-test":
        return oracle_test(ctx, args)
    if args.cmd == "train":
        return train_replicates(ctx, args)
    if args.cmd in ("pgv", "ft"):
        return pgv_ft(ctx, args, args.cmd.upper())
    return eval_phase(ctx, args, extras=args.cmd == "extras")


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):  # unit keys and the FILL rows contain ± and ½ (any console code page)
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    if args.cmd == "code-digest":
        out = {s: code_digest(s) for s in ("t2", "t3")}
        print(json.dumps(dict(out[args.stage], tests=test_shas(), reference_shas=LA.reference_shas(),
                              other_stage={k: v["digest"] for k, v in out.items()}), indent=1))
        return 0
    runs = Path(args.runs)
    n = getattr(args, "n", None)
    if n is None:  # TEST-side commands follow the pilot that was reported (the n = 6 fallback, if it was taken)
        n = 6 if (runs / "pilot-n6" / "PILOT_REPORT.json").exists() else 7
    return run_command(Ctx(runs=runs, n=n, pilot_dir="pilot" if n == 7 else f"pilot-n{n}"), args)


if __name__ == "__main__":
    sys.exit(main())
