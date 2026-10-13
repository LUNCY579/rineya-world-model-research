"""RCX-E05 runner (PREREG_E05_DRAFT.md §2 custody, §9 pilot gates and fill, §10 order/lane/VOID, binding; DESIGN.md
rev 3 §9-§10). Commands, from complementary/, in the §10.1 order:

  code-digest [--stage t2|t3]                         §2.1 digest (pure; prints JSON incl. test-file and reference shas)
  pilot-a  --expect-code-digest <T2>                  Phase 1a: draw TRAIN/DEV -> pilot/PILOT_MANIFEST.json (create-only),
                                                      library tables, Q1-Q6/Q8 checks, reference arms on DEV (Q7, Q9,
                                                      G-P2), selection training (seed 20261245), DEV evaluation, G-P4 (+
                                                      the UVFA extension, the 800-label fallback)
  pilot-b  --expect-code-digest <T2>                  Phase 1b: PGV_400/FT grids (G-P3 + extensions), k-curve on 8 DEV-ID
                                                      parents (G-P3b + R_ext), selected PG400 on the 24 DEV-OOD H-K/H-NEG
                                                      parents (M5), G-P5 rates
  pilot-report --expect-code-digest <T2>              pilot/PILOT_REPORT.json (create-only; no lane: reads outputs only):
                                                      gates G-P1..G-P5 (Q7, Q9), sigma-hats, §8 sizing, powers, cuts, FILL
  draw-test --attempt k --expect-prereg-sha <T3> --expect-code-digest <T3>
                                                      confirm/attempt-k/TEST_MANIFEST.json (create-only)
  oracle-test --attempt k --manifest-sha <T4> --code-digest <T3>
                                                      every §10.4 VOID check -> ORACLE_TEST_REPORT.json (create-only)
  train --expect-prereg-sha <T3> --expect-code-digest <T3>   replicates 20261246-48 of SF, UVFA, UVFA+aux; SF-shuffled
  pgv | ft | eval | extras --attempt k --manifest-sha <T4> --code-digest <T3>

Exit codes of lane phases: 0 = phase complete; 75 = yielded at a safe point or slice used up, work left (re-run the same
command: it resumes from its checkpoints and keyed outputs); 3 = the lane start condition does not hold (nothing
written). Custody refusals exit with a message and write nothing.

Custody (§2):
  * code digest = sha256 over the sorted lines "<relpath>\\t<sha256>\\n" (relpaths from complementary/) of the new
    outcome sources (T2: soft.py, families_e05, enum_e05, learners_e05, run_e05; T3: + analyze_e05) and the pinned
    files (PINNED; their 8-hex prefixes are checked, a mismatch refuses every phase). lane_e05.py, rcx/lane.py and
    rcx/guard.py are operational (M1): outside the digest, their shas recorded in every provenance record.
  * pilot phases refuse unless --expect-code-digest equals the current T2 digest and occurs in the COMMITTED
    DECISIONS.md; every keyed output carries the digest, and a slice refuses if the directory holds outputs of another
    digest (re-run the affected phase in a fresh --pilot-dir, logged).
  * committed-sha refusal (L5): oracle-test, pgv, ft, eval, extras refuse unless the full 64-hex TEST_MANIFEST sha256
    equals the file's sha and occurs in `git show HEAD:./DECISIONS.md` (not the working copy); draw-test and train
    likewise need the T3 prereg sha and T3 digest committed. pgv/ft/eval/extras also need ORACLE_TEST_REPORT.json = PASS
    (B4); the first of them logs "first-learned-test" in ATTEMPTS.jsonl, after which oracle-test refuses (VOID is
    declared only inside oracle-test, before any learned-arm TEST computation).
  * outputs: manifests/reports create-only (atomic link); per-cell results in append-only keyed JSONL (a key is never
    written twice; completed keys are never recomputed); runs/RCX-E05/ATTEMPTS.jsonl (phase slices, manifests,
    oracle-test results, redraws); <dir>/PROVENANCE.jsonl (digest, file shas, reference shas, seeds, versions, args).

Lane (§10.2): every lane phase runs in slices through lane_e05.Session (party rcx-e05, try_acquire only). A slice is
planned at L = min(projected remaining work x 1.2, lane_e05.slice_budget_min) <= 15 min and stops at the first safe
point after L or on any yield condition; units are evaluation cells (one (arm, parent, start)) and optimizer steps
inside the learners' trainers (safe_point callback).

The two modules written in parallel (enum_e05, learners_e05) are called ONLY from the "adapters" section below (names
from PREREG §2.1/§5; reconcile there). Seeds and stream tags: families_e05 docstring plus, here, RAND per cell
`20261244:rand:<split>:<key>:<s0>`, pilot Q checks `20261242:q-checks:<Q>`, single-fact base-policy labels
`20261243:train-base:<fact>`, DEV per-goal fit inits `20261245:<kind>:<cfg>:<key>`, TEST PGV inits
`20261249:pgv-init:<key>:<k>`, FT data order `<replicate>:ft:<key>`.
"""
from __future__ import annotations

import argparse
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
from collections import Counter, defaultdict
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
BAR, SF_FLOOR_BAR, TOL = 0.50, 0.50, 1e-9
REF_PILOT = ("ORACLE", "STOP", "RAND", "GREEDY", "EXACT-GPI(L)", "ENUM-MIS±")
ZS_PILOT = ("SF-GPI", "SF-noGPI", "SF-L", "UVFA", "UVFA+aux")
ARM_MODEL = {"SF-GPI": "SF", "SF-noGPI": "SF", "SF-L": "SF", "SF-GPI-5": "SF", "SF-shuffled": "SF-shuffled",
             "UVFA": "UVFA", "UVFA+aux": "UVFA+aux"}
SELECT_BY = {"SF": "SF-GPI", "UVFA": "UVFA", "UVFA+aux": "UVFA+aux"}
POSITIVE_DEV = ("DEV-ID", "DEV-OOD:H-K", "DEV-OOD:H-COST")
POSITIVE_TEST = ("H-ID", "H-K", "H-COST", "H-DYN", "H-CONJ")
PGV_GRID = {"w64s500": dict(width=64, steps=2000, keep=500), "w64s2000": dict(width=64, steps=2000, keep=2000),
            "w256s500": dict(width=256, steps=2000, keep=500), "w256s2000": dict(width=256, steps=2000, keep=2000)}
PGV_COMMON = dict(batch=64, lr=1e-3, eval_every=100, val_n=40)
PGV_EXT = {"w128s4000": dict(width=128, steps=4000, keep=4000, lr=3e-4, batch=64, eval_every=100, val_n=40)}
FT_GRID = {"s100": dict(steps=400, keep=100), "s400": dict(steps=400, keep=400)}
FT_COMMON = dict(batch=64, lr=3e-4, eval_every=25, val_n=40, step0=True)
FT_EXT = {"s1000": dict(steps=1000, keep=1000, lr=1e-4, batch=64, eval_every=25, val_n=40, step0=True)}
KCURVE = {1600: dict(width=256, batch=128, steps=2000, keep=2000, lr=1e-3, eval_every=100, val_frac=0.1),
          6400: dict(width=256, batch=256, steps=2500, keep=2500, lr=1e-3, eval_every=100, val_frac=0.1)}
R_EXT = {k: dict(width=128, batch=b, steps=4000, keep=4000, lr=3e-4, eval_every=200, val_frac=0.1)
         for k, b in ((400, 64), (1600, 128), (6400, 256))}
ZS_BASE = dict(epochs=40, lr=1e-3, keep=(20, 40))
ZS_EXT = dict(epochs=80, lr=3e-4, keep=(80,))


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
PREREG = HERE / "PREREG_E05.md"


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
    pins_bad = {rel: (files.get(rel, "")[:8], pre) for rel, pre in PINNED.items() if not files.get(rel, "").startswith(pre)}
    return {"stage": stage, "digest": None if missing else sha_bytes(text.encode("utf-8")), "files": files,
            "missing": missing, "pins_bad": pins_bad}


def test_shas(root: Path = ROOT) -> dict:
    return {rel: (sha_file(Path(root) / rel) if (Path(root) / rel).exists() else None) for rel in TEST_FILES}


def committed_decisions(root: Path = ROOT) -> str:
    """The committed DECISIONS.md (`git show HEAD:./DECISIONS.md` in complementary/), never the working copy."""
    r = subprocess.run(["git", "-C", str(root), "show", "HEAD:./DECISIONS.md"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"refusing: cannot read the committed DECISIONS.md ({r.stderr.decode(errors='replace')[:200]})")
    return r.stdout.decode("utf-8", errors="replace")


HEX64 = re.compile(r"[0-9a-f]{64}")


def require_committed(value: str, what: str, text: str) -> None:
    if not HEX64.fullmatch(value or ""):
        raise SystemExit(f"refusing: {what} must be a full 64-hex sha256 (got {value!r})")
    if value not in text:
        raise SystemExit(f"refusing: {what} {value[:12]}... does not occur in the committed DECISIONS.md "
                         "(git show HEAD:./DECISIONS.md; PREREG §2.2 L5)")


def require_digest(ctx, stage: str, expect: str) -> dict:
    """Current digest of `stage` == expect, every file present, pins intact, and expect committed in DECISIONS."""
    cd = code_digest(stage, ctx.root)
    if cd["missing"]:
        raise SystemExit(f"refusing: {stage.upper()} digest files missing: {cd['missing']}")
    if cd["pins_bad"]:
        raise SystemExit(f"refusing: pinned files changed (sha prefix now, pinned): {cd['pins_bad']}")
    if cd["digest"] != expect:
        raise SystemExit(f"refusing: current {stage.upper()} code digest {cd['digest'][:12]}... != expected "
                         f"{(expect or '')[:12]}...")
    require_committed(expect, f"{stage.upper()} code digest", ctx.decisions())
    return cd


def require_prereg(ctx, expect: str) -> str:
    p = ctx.root / "experiments" / "RCX-E05" / "PREREG_E05.md"
    if not p.exists():
        raise SystemExit("refusing: PREREG_E05.md (T3 freeze) does not exist")
    sha = sha_file(p)
    if sha != expect:
        raise SystemExit(f"refusing: PREREG_E05.md sha256 {sha[:12]}... != --expect-prereg-sha {(expect or '')[:12]}...")
    require_committed(expect, "PREREG_E05.md sha256", ctx.decisions())
    return sha


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, default=_jdefault, ensure_ascii=True) + "\n").encode("utf-8")


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
    try:
        os.link(tmp, path)  # fails if path exists: never overwrites
    finally:
        tmp.unlink(missing_ok=True)
    return sha_bytes(data)


def _append_line(path, rec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = path.exists() and path.stat().st_size > 0 and not path.read_bytes()[-1:] == b"\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(("\n" if torn else "") + json.dumps(rec, sort_keys=True, default=_jdefault) + "\n")
        f.flush()
        os.fsync(f.fileno())


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

    def add(self, key, rec: dict, digest: str | None = None) -> dict:
        if key in self.rows:
            raise RuntimeError(f"keyed output {key} already written (append-only)")
        rec = dict(rec, key=key, digest=digest, at=time.strftime("%Y-%m-%d %H:%M:%S"))
        _append_line(self.path, rec)
        self.rows[key] = rec
        return rec

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
                                              "CUDA_VISIBLE_DEVICES", "RCX_THREADS")}
    return dict(phase=phase, digest_stage=(cd or {}).get("stage"), code_digest=(cd or {}).get("digest"),
                code_files=(cd or {}).get("files"), reference_shas=LA.reference_shas(), test_shas=test_shas(ctx.root),
                seeds=SEEDS, versions=vers, threads=threads, n=ctx.n, **extra)


# ============================================================================= context, data, oracle

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
    sizes: dict | None = None
    q_sizes: dict = field(default_factory=lambda: dict(q1=200, q2_goals=20, q2_states=10, q3=1000, q5=500))
    cost_scale: float = 1.0

    def __post_init__(self):
        self.runs, self.root = Path(self.runs), Path(self.root)
        if self.env is None:
            self.env = LA.Env(paths=LA.Paths(events=self.runs / "LANE_EVENTS.jsonl"))

    @property
    def pdir(self) -> Path:
        return self.runs / self.pilot_dir

    def adir(self, k: int) -> Path:
        return self.runs / "confirm" / f"attempt-{int(k)}"


class Data:
    """Lazy oracle data for one manifest (pilot or TEST): space, phi, parents, cached tables, library."""

    def __init__(self, ctx, manifest: dict, lib_dir: Path):
        self.ctx, self.m, self.lib_dir = ctx, manifest, Path(lib_dir)
        self.n = int(manifest["n"])
        self.space = SP.get_space(self.n)
        self.phi = SO.get_phi(self.space)
        self.fam = {f: [FA.Parent.from_json(d) for d in ps] for f, ps in manifest["parents"].items()}
        self.by_key = {p.key: p for ps in self.fam.values() for p in ps}
        self._tabs = {}
        self._lib = {}

    def table(self, p) -> SO.SFTable:
        t = self._tabs.get(p.key)
        if t is None:
            t = self._tabs[p.key] = FA.parent_table(self.space, self.phi, p)
        return t

    def arm_stop(self, p) -> np.ndarray:
        """STOP scores every non-ORACLE arm uses: w'.phi with the w' the arm receives (H-CONJ: the linearization)."""
        return SO.stop_values(self.phi, p.weights()) if p.rtype == "conj" else self.table(p).stop

    def mask(self, p):
        return FA.edge_mask(self.space, p.dyn)

    def library(self, base: bool = False) -> SimpleNamespace:
        """Library L = the TRAIN goals (+ the 49 single-fact base policies if base): W [m, F], c [m], end/T [m, N]."""
        if base not in self._lib:
            parts = [self._lib_part("train")] + ([self._lib_part("base")] if base else [])
            self._lib[base] = SimpleNamespace(**{k: np.concatenate([getattr(x, k) for x in parts]) for k in
                                                 ("W", "c", "end", "T")})
        return self._lib[base]

    def _lib_part(self, part: str) -> SimpleNamespace:
        p = self.lib_dir / f"oracle_library_{part}.npz"
        z = np.load(p)
        return SimpleNamespace(W=z["W"], c=z["c"], end=z["end"], T=z["T"])

    def lib_goals(self, part: str) -> list:
        """(w, c) of the library goals of `part` in library order."""
        if part == "train":
            return [(np.asarray(p.w, dtype=np.float64), p.c) for p in self.fam["TRAIN"]]
        F = self.n * (self.n + 1)
        return [(np.eye(F)[k], FA.C_TRAIN) for k in range(self.n * self.n)]


def build_library(d: Data, part: str, path: Path) -> None:
    goals = d.lib_goals(part)
    ends, Ts = [], []
    for w, c in goals:
        tab = SO.sf_table(d.space, SO.stop_values(d.phi, w), c)
        ends.append(tab.end)
        Ts.append(tab.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez(tmp, W=np.stack([w for w, _ in goals]), c=np.array([c for _, c in goals]), end=np.stack(ends),
             T=np.stack(Ts))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def tabs_of(lib) -> list:
    return [SimpleNamespace(end=lib.end[i], T=lib.T[i]) for i in range(len(lib.c))]


# ============================================================================= adapters (enum_e05, learners_e05)
# The ONLY call sites of the two modules written in parallel. Expected API (PREREG §2.1, §5):
#   enum_e05.enum_policy(space, w, c, kind, stop, edge_mask=None) -> (policy, diag)
#       kind in {"MIS", "H*", "GNN"}; policy = rcx.bw.soft Policy (s, acts, dsts) -> candidate index; diag = mutable
#       dict updated per decision with "decisions" (>= 2 legal moves), "all_tied" and "targets".
#   learners_e05.train_zero_shot(arm, space, data, *, seed, epochs, lr, keep, out_dir, safe_point) -> {epoch: path}|None
#       arm in {"SF", "UVFA", "UVFA+aux", "SF-shuffled"}; data = dict(states, W, c, psi, v, goal); None = stopped at a
#       safe point (checkpoint in out_dir; the same call resumes).
#   learners_e05.fit_pergoal(kind, space, jobs, cfg, *, out_dir, safe_point) -> {job key: path}|None
#       kind "PGV" | "FT"; jobs = [dict(key, states, v, stab, seed, w, c, init)]; cfg = one grid point (width, steps,
#       keep, lr, batch, eval_every, val_n or val_frac, step0); returns the early-stopped model per job.
#   learners_e05.load_model(path); learners_e05.predict_psi(model, space, states, W, c) -> [len(states), m, F + 1];
#   learners_e05.predict_value(model, space, states, W, c) -> [len(states), m] (PGV ignores W, c);
#   learners_e05.time_step(model, width, batch, B, steps) -> seconds per optimizer step (G-P5).

def EN():
    return _load("enum_e05")


def LE():
    return _load("learners_e05")


def enum_policy(d: Data, p, kind: str):
    return EN().enum_policy(d.space, p.weights(), p.c, kind=kind, stop=d.arm_stop(p), edge_mask=d.mask(p))


def train_zero_shot(arm, d: Data, data, *, seed, cfg, out_dir, safe_point):
    return LE().train_zero_shot(arm, d.space, data, seed=seed, epochs=cfg["epochs"], lr=cfg["lr"], keep=cfg["keep"],
                                out_dir=str(out_dir), safe_point=safe_point)


def fit_pergoal(kind, d: Data, jobs, cfg, *, out_dir, safe_point):
    return LE().fit_pergoal(kind, d.space, jobs, cfg, out_dir=str(out_dir), safe_point=safe_point)


_MODELS = {}


def load_model(path):
    m = _MODELS.get(path)
    if m is None:
        m = _MODELS[path] = LE().load_model(path)
    return m


def psi_hat(model, d: Data, states, W, c) -> np.ndarray:
    return np.asarray(LE().predict_psi(model, d.space, np.asarray(states), np.asarray(W), np.asarray(c)), np.float64)


def value_hat(model, d: Data, states, W, c) -> np.ndarray:
    return np.asarray(LE().predict_value(model, d.space, np.asarray(states), np.asarray(W), np.asarray(c)), np.float64)


def time_step(model, width, batch, B, steps=5) -> float:
    return float(LE().time_step(model, width=width, batch=batch, B=B, steps=steps))


# ============================================================================= arms and cells

def codes_for(arm: str, d: Data, p, base: bool):
    """(W [m, F], c [m]) of the GPI code set C of an SF arm: SF-GPI L + {w'}, SF-noGPI {w'}, SF-L L, SF-GPI-5 {w'} +
    the 4 library codes nearest to z' = (w'/3, c') by cosine (ties: lower library index)."""
    lib = d.library(base)
    wq, cq = p.weights()[None], np.array([p.c])
    if arm in ("SF-noGPI",):
        return wq, cq
    if arm == "SF-L":
        return lib.W, lib.c
    if arm == "SF-GPI-5":
        Z = np.concatenate([lib.W / 3.0, lib.c[:, None]], axis=1)
        z = np.concatenate([wq[0] / 3.0, cq])
        cos = Z @ z / (np.linalg.norm(Z, axis=1) * np.linalg.norm(z))
        near = np.argsort(-cos, kind="stable")[:4]
        return np.concatenate([wq, lib.W[near]]), np.concatenate([cq, lib.c[near]])
    return np.concatenate([lib.W, wq]), np.concatenate([lib.c, cq])  # SF-GPI, SF-shuffled


def learned_policy(arm: str, model, d: Data, p, base: bool):
    stop, w, c = d.arm_stop(p), p.weights(), p.c
    if ARM_MODEL.get(arm, "").startswith("SF"):
        W, C = codes_for(arm, d, p, base)
        return SO.scored_policy(stop, lambda s, a, dsts: SO.gpi_scores(psi_hat(model, d, dsts, W, C), w, c))
    return SO.scored_policy(stop, lambda s, a, dsts: value_hat(model, d, dsts, w[None], np.array([c]))[:, 0] - c)


def best_single_policy(d: Data, p, s0: int, base: bool):
    """BEST-SINGLE(L): the library policy with the best exact value w'.phi(e_i(s0)) - c' T_i(s0) (first maximum),
    followed in closed loop (STOP where it stops, or where its move is not a legal candidate)."""
    lib, stop = d.library(base), d.arm_stop(p)
    vals = stop[lib.end[:, s0]] - p.c * lib.T[:, s0].astype(np.float64)
    i = int(np.argmax(vals))
    nxt = SO.sf_table(d.space, SO.stop_values(d.phi, lib.W[i]), float(lib.c[i])).nxt

    def pol(s, acts, dsts):
        j = np.flatnonzero(np.asarray(dsts) == nxt[s])
        return 0 if nxt[s] == s or not len(j) else int(j[0]) + 1
    return pol


def ref_policy(arm: str, d: Data, p, s0: int, split: str, base: bool):
    tab, stop = d.table(p), d.arm_stop(p)
    if arm == "ORACLE":
        return SO.oracle_policy(tab), None
    if arm == "STOP":
        return SO.stop_policy(), None
    if arm == "GREEDY":
        return SO.greedy_policy(stop, p.c), None
    if arm == "RAND":
        return SO.rand_policy(FA.stream(SEEDS["starts"], f"rand:{split}:{p.key}:{s0}")), None
    if arm == "EXACT-GPI(L)":
        G = d._lib.setdefault(("gpi", p.key, base), SO.gpi_table(stop, p.c, tabs_of(d.library(base))))
        return SO.gpi_policy(stop, p.c, G), None
    if arm == "BEST-SINGLE(L)":
        return best_single_policy(d, p, s0, base), None
    if arm.startswith("ENUM-"):
        kind = {"ENUM-MIS±": "MIS", "ENUM-H*±": "H*", "GNN-ENUM±": "GNN"}[arm]
        return enum_policy(d, p, kind)
    raise KeyError(arm)


def eval_cells(log: KeyedLog, d: Data, sl, *, split, family, p, arm, tag, policy_fn, digest) -> bool:
    """One (arm, tag, parent) unit: every start a keyed cell (safe point after each). True when all are written."""
    tab, mask = d.table(p), d.mask(p)
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
            rec["diag"] = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in dict(diag).items()}
        log.add(key, rec, digest)
        if sl is not None and sl.safe_point("cell"):
            return all(kj("cell", split, arm, tag, p.key, int(s)) in log for s in p.starts)
    return True


def cells_done(log, split, arm, tag, p) -> bool:
    return all(kj("cell", split, arm, tag, p.key, int(s)) in log for s in p.starts)


def parent_nreg(log, split, arm, tag, p) -> float:
    """Mean nreg over the parent's starts (sum / len, as rcx.stats)."""
    v = [log.get(kj("cell", split, arm, tag, p.key, int(s)))["nreg"] for s in p.starts]
    return sum(v) / len(v)


def family_mean(log, split, arm, tag, ps) -> float:
    return mean(parent_nreg(log, split, arm, tag, p) for p in ps)


def bank_unit(log: KeyedLog, d: Data, sl, *, split, p, arm, tag, choose, vhat, digest) -> bool:
    """One-step bank of a parent for one arm: per state top-1, regret1 and (value arms) |V-hat - V*|."""
    key = kj("bank", split, arm, tag, p.key)
    if key in log:
        return True
    tab, keep = d.table(p), SO.sorted_mask(d.space, d.mask(p))
    rows = []
    vh = vhat(np.asarray(p.bank)) if vhat is not None else None
    for i, s in enumerate(p.bank):
        acts, dsts = SO.moves(d.space, int(s), keep)
        q = SO.q_star(tab, int(s), dsts)
        row = dict(s=int(s), vstar=float(tab.V[s]))
        if choose is not None:
            k = choose(int(s), acts, dsts)
            r1, top = SO.regret1(q, k, tab.c)
            row.update(regret1=r1, top1=top)
        if vh is not None:
            row["vhat"] = float(vh[i])
        rows.append(row)
    log.add(key, dict(split=split, arm=arm, tag=tag, parent=p.key, c=tab.c, rows=rows), digest)
    return sl is None or not sl.safe_point("cell") or True


def acc_half(log, split, arm, tag, ps) -> float:
    rows = [r for p in ps for r in log.get(kj("bank", split, arm, tag, p.key))["rows"]]
    c = {p.key: p.c for p in ps}
    out = []
    for p in ps:
        for r in log.get(kj("bank", split, arm, tag, p.key))["rows"]:
            out.append(abs(r["vhat"] - r["vstar"]) < c[p.key] / 2)
    del rows
    return sum(out) / len(out)


def top1_rate(log, split, arm, tag, ps) -> float:
    rows = [r for p in ps for r in log.get(kj("bank", split, arm, tag, p.key))["rows"]]
    return sum(bool(r["top1"]) for r in rows) / len(rows)


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
        """True = stop now (the caller checkpoints first). kind 'batch'/'epoch'/'unit' forces a lane check."""
        if self.reason:
            return True
        r = self.sess.check(force=kind in ("batch", "epoch", "unit"))
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
    L = min(need, LA.slice_budget_min(ctx.env))
    if not L > 0:
        print(f"{phase}: not starting (no time before the next window rule deadline)", flush=True)
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
                digest=(cd or {}).get("digest"))
    _append_line(ctx.runs / "PROVENANCE.jsonl", dict(provenance(ctx, phase, cd, args=args or {}), slice=sid,
                                                    attempt=attempt, lane_mode=sess.mode, lane_token=sess.rec["token"]))
    done, status, last = False, "error", None
    try:
        while True:
            pending = plan()
            if not pending:
                done = True
                break
            if sl.safe_point("unit"):
                break
            u = pending[0]
            if u.key == last:
                raise RuntimeError(f"{phase}: unit {u.key} still pending after it reported completion")
            print(f"{time.strftime('%H:%M:%S')} {phase}: {u.key}", flush=True)
            if not u.run(sl):
                break
            last = u.key
        status = "ok"
    finally:
        reason = "done" if done else (sl.reason or ("error" if status != "ok" else "stopped"))
        sess.release(reason)
        log_attempt(ctx, "end", phase=phase, attempt=attempt, slice=sid, status=status, reason=reason,
                    complete=done)
    print(f"{phase}: {'complete' if done else 'yielded: ' + reason}", flush=True)
    return 0 if done else EXIT_REQUEUE


# ============================================================================= instrument checks (Q1-Q6, Q8 at the pilot)

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


def q2_check(rng, n_goals: int, n_states: int) -> dict:
    """Q2 on n = 6: H-CONJ V* == max(3 1[G <= s], 3 - c h*(s, G)) with h* from GoalLabels, and == max-plus VI."""
    sp = SP.get_space(6)
    fs = SO.facts(6)
    bad, pairs = 0, 0
    while pairs < n_goals * n_states:
        G = tuple(sorted(rng.sample(range(30), rng.choice((2, 3)))))
        if not FA.conj_satisfiable(sp, G):
            continue
        goal = [fs[k] for k in G]
        V = SO.conj_values(sp, goal, 3.0, 0.25)
        m = sp.goal_mask(SO.compile_goal(6, goal))
        vi = SO.max_plus_vi(sp, 3.0 * m, 0.25)
        gl = SP.GoalLabels(list(range(6)), frozenset(goal))
        for s in rng.sample(range(len(sp)), n_states):
            ref = max(3.0 * bool(m[s]), 3.0 - 0.25 * gl.h_id(s))
            bad += int(not (V[s] == ref == vi[s]))
            pairs += 1
    return dict(value=bad, n=pairs, ok=bad == 0)


def q3_check(d: Data, rng, n_pairs: int) -> dict:
    """Q3: rollouts of pi_w reproduce the tabled endpoints and lengths, and V* = w-bar.psi exactly (pilot goals)."""
    ps = [p for ps in d.fam.values() for p in ps]
    bad = 0
    for _ in range(n_pairs):
        p = ps[rng.randrange(len(ps))]
        tab, s = d.table(p), rng.randrange(len(d.space))
        ep = SO.run_episode(d.space, s, SO.oracle_policy(tab), d.mask(p))
        v = float(tab.stop[tab.end[s]]) - tab.c * float(tab.T[s])
        bad += int(not (ep.final == tab.end[s] and ep.T == tab.T[s] and not ep.loop and v == tab.V[s]))
    return dict(value=bad, n=n_pairs, ok=bad == 0)


def q4_check(d: Data, families, base: bool) -> dict:
    """Q4: EXACT-GPI(L) closed-loop return >= max over library i of V^{pi_i}_{w'}(s0) at every start (free dynamics)."""
    lib = d.library(base)
    bad = n = 0
    for f in families:
        for p in d.fam[f]:
            if p.dyn != "free":
                continue
            tab, stop = d.table(p), d.arm_stop(p)
            G = SO.gpi_table(stop, p.c, tabs_of(lib))
            for s0 in p.starts:
                ep = SO.run_episode(d.space, int(s0), SO.gpi_policy(stop, p.c, G))
                J = SO.episode_return(stop, p.c, ep)
                best = float((stop[lib.end[:, s0]] - p.c * lib.T[:, s0].astype(np.float64)).max())
                bad += int(J < best)
                n += 1
            del tab
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
        key_ok = FA.canon_key(ws, p.c, d.n) == p.key
        bad += int(not (np.array_equal(Vs[sp], tab.V) and np.array_equal(conj, tab.V) and key_ok))
    return dict(value=bad, n=n_sigma, ok=bad == 0)


def q6_check() -> dict:
    """Q6 on n = 5: height_limit_mask(h = 3) == a direct rule check on every edge (the stacked block's level in the
    successor, walked down to the table, is <= 3)."""
    sp = SP.get_space(5)
    n, mask = 5, SO.height_limit_mask(sp, 3)
    src = np.repeat(np.arange(len(sp)), np.diff(sp.fwd_ptr))
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
    del src
    return dict(value=bad, n=int(sp.n_edges), ok=bad == 0)


def q8_check(d: Data, excl, label_sets: dict) -> dict:
    """Q8: split hygiene on the M6 key (families_e05.check_splits) and every start, bank state and label row
    re-verified (distinct, valid, not excluded now)."""
    reg = FA.Registry.from_parents(d.n, [p for ps in d.fam.values() for p in ps])
    splits = {k: [bool(v[0]), v[1]] for k, v in FA.check_splits(reg).items()}
    bad = Counter()
    for f, ps in d.fam.items():
        for p in ps:
            v = FA.verify_parent(d.space, d.phi, p, excl, labels=label_sets.get(p.key, ()))
            if p.starts and not v["starts_ok"]:
                bad["starts"] += 1
            if p.bank and not v["bank_ok"]:
                bad["bank"] += 1
            for k in ("excluded_starts", "excluded_bank", "excluded_labels"):
                bad[k] += int(v.get(k, 0))
    ok = all(v[0] for v in splits.values()) and not sum(bad.values())
    return dict(value=dict(splits=splits, defects=dict(bad)), ok=ok)


# ============================================================================= pilot (Phase 1a / 1b)

TRAIN_LABELS_MAX = 800
DEV_PERGOAL = 400


class Pilot:
    """State and units of the pilot phases in ctx.pdir (keyed logs: cells, banks, checks, fits)."""

    def __init__(self, ctx, cd: dict | None):
        self.ctx, self.cd = ctx, cd
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
        d = self.data()
        return {f: ps for f, ps in d.fam.items() if f.startswith("DEV")}

    # ---- unit: draw
    def unit_draw(self, sl) -> bool:
        ctx = self.ctx
        space = SP.get_space(ctx.n)
        phi = SO.get_phi(space)
        excl = ctx.exclusions(space)
        reg = FA.draw_pilot(space, phi, excl, ctx.sizes)
        FA.assert_splits(reg)
        dev_k = set(FA.dev_kcurve_subset(reg))
        kmax = max(FA.SIZES["kcurve"])
        train_labels, pergoal, base_labels, lab_excl = {}, {}, {}, Counter()
        for p in reg.parents["TRAIN"]:
            idx, cnt = FA.draw_labels(space, p, TRAIN_LABELS_MAX, excl, seed=SEEDS["train_labels"], tag="train")
            train_labels[p.key] = idx
            lab_excl.update(cnt)
        for p in reg.parents["DEV-ID"] + reg.parents.get("DEV-OOD:H-K", []) + reg.parents.get("DEV-OOD:H-NEG", []):
            k = kmax if p.key in dev_k else DEV_PERGOAL
            idx, cnt = FA.draw_labels(space, p, k, excl)
            pergoal[p.key] = idx
            lab_excl.update(cnt)
        fs = SO.facts(ctx.n)
        for k in range(ctx.n * ctx.n):
            idx, cnt = FA.sample_states(np.arange(len(space)), TRAIN_LABELS_MAX,
                                        FA.stream(SEEDS["train_labels"], f"train-base:{k}"), frozenset([fs[k]]), excl)
            base_labels[str(k)] = idx
            lab_excl.update(cnt)
        man = dict(experiment="RCX-E05", kind="pilot", n=ctx.n, code_digest_t2=self.digest,
                   parents={f: [p.to_json() for p in ps] for f, ps in reg.parents.items()},
                   rejections={f: dict(c) for f, c in reg.rejections.items()}, notes=dict(reg.notes),
                   train_labels=train_labels, pergoal_labels=pergoal, base_labels=base_labels,
                   dev_kcurve=sorted(dev_k), label_exclusions=dict(lab_excl),
                   exclusions=None if excl is None else dict(calls=dict(excl.calls),
                                                              sha=sha_file(ctx.root / "experiments/RCX-E07/EXCLUSIONS.json")),
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
                r = q2_check(rng, qs["q2_goals"], qs["q2_states"])
            elif q == "Q3":
                r = q3_check(d, rng, qs["q3"])
            elif q == "Q4":
                r = q4_check(d, [f for f in d.fam if f.startswith("DEV")], self.base())
            elif q == "Q5":
                r = q5_check(d, rng, qs["q5"])
            elif q == "Q6":
                r = q6_check()
            else:
                excl = self.ctx.exclusions(d.space)
                labs = dict(self.m["pergoal_labels"])
                labs.update({k: v[:self.train_labels()] for k, v in self.m["train_labels"].items()})
                r = q8_check(d, excl, labs)
            self.checks.add(kj("check", q, self.base() if q == "Q4" else None), r, self.digest)
            return True
        return run

    def check(self, q: str):
        return self.checks.get(kj("check", q, self.base() if q == "Q4" else None))

    # ---- reference arms on DEV
    def ref_unit(self, arm, f, p, base):
        def run(sl):
            d = self.data()
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=arm, tag=self.ref_tag(arm, base),
                              policy_fn=lambda s0: ref_policy(arm, d, p, s0, "pilot", base), digest=self.digest)
        return run

    @staticmethod
    def ref_tag(arm, base=False):
        return ("L+base" if base else "L") if arm in ("EXACT-GPI(L)", "BEST-SINGLE(L)") else "-"

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

    def train_labels(self) -> int:
        return 800 if self.stage_labels() == 800 else 400

    def stage_labels(self) -> int:
        """400, or 800 once G-P4(a) failed at 400 (§9.1)."""
        g = self.gp4(400)
        return 800 if g is not None and not g["a"]["ok"] else 400

    # ---- training data and selection models
    def data_path(self, labels, base) -> Path:
        return self.dir / f"train_data_L{labels}{'_base' if base else ''}.npz"

    def unit_train_data(self, labels, base):
        def run(sl):
            d = self.data()
            goals = [(p.key, p.weights(), p.c, self.m["train_labels"][p.key][:labels]) for p in d.fam["TRAIN"]]
            if base:
                goals += [(f"base:{k}", w, c, self.m["base_labels"][str(k)][:labels])
                          for k, (w, c) in enumerate(d.lib_goals("base"))]
            S, Wl, cl, Ps, Vs, gid = [], [], [], [], [], []
            for g, (_, w, c, idx) in enumerate(goals):
                tab = SO.sf_table(d.space, SO.stop_values(d.phi, w), c)
                idx = np.asarray(idx, dtype=np.int64)
                S.append(idx)
                Wl.append(np.repeat(w[None], len(idx), axis=0))
                cl.append(np.full(len(idx), c))
                Ps.append(SO.psi(d.phi, tab, idx))
                Vs.append(tab.V[idx])
                gid.append(np.full(len(idx), g))
            p = self.data_path(labels, base)
            tmp = p.with_name(f".{p.stem}.tmp-{os.getpid()}.npz")
            np.savez(tmp, states=np.concatenate(S), W=np.concatenate(Wl), c=np.concatenate(cl),
                     psi=np.concatenate(Ps), v=np.concatenate(Vs), goal=np.concatenate(gid))
            try:
                os.link(tmp, p)
            finally:
                tmp.unlink(missing_ok=True)
            return True
        return run

    def zs_key(self, arm, labels, base, cfg):
        return kj("train", arm, SEEDS["selection"], labels, base, cfg)

    def unit_train(self, arm, labels, base, cfg):
        def run(sl):
            data = dict(np.load(self.data_path(labels, base)))
            out = self.dir / "models" / f"{arm}_s{SEEDS['selection']}_L{labels}{'_base' if base else ''}_{cfg}"
            spec = ZS_BASE if cfg == "base" else ZS_EXT
            res = train_zero_shot(arm, self.data(), data, seed=SEEDS["selection"], cfg=spec, out_dir=out,
                                  safe_point=sl.safe_point if sl is not None else (lambda kind="step": False))
            if res is None:
                return False
            models = {str(k): str(v) for k, v in res.items()}
            self.fits.add(self.zs_key(arm, labels, base, cfg),
                          dict(arm=arm, labels=labels, base=base, cfg=cfg, models=models,
                               shas={k: _sha_path(v) for k, v in models.items()}), self.digest)
            return True
        return run

    def model_path(self, arm_model, labels, base, cfg, epoch):
        r = self.fits.get(self.zs_key(arm_model, labels, base, cfg))
        return None if r is None else r["models"][str(epoch)]

    def zs_tag(self, model, labels, base, cfg, epoch):
        return f"{model}|L{labels}{'+base' if base else ''}|{cfg}|e{epoch}"

    def zs_unit(self, arm, f, p, labels, base, cfg, epoch):
        def run(sl):
            d = self.data()
            model = load_model(self.model_path(ARM_MODEL[arm], labels, base, cfg, epoch))
            pol = learned_policy(arm, model, d, p, base)
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=arm,
                              tag=self.zs_tag(ARM_MODEL[arm], labels, base, cfg, epoch),
                              policy_fn=lambda s0: (pol, None), digest=self.digest)
        return run

    def zs_mean(self, arm, f, labels, base, cfg, epoch):
        ps = self.dev()[f]
        tag = self.zs_tag(ARM_MODEL[arm], labels, base, cfg, epoch)
        if not all(cells_done(self.cells, "pilot", arm, tag, p) for p in ps):
            return None
        return family_mean(self.cells, "pilot", arm, tag, ps)

    def selection(self, labels, base):
        """Selected (cfg, epoch) per model by DEV-ID nreg of its selecting arm (ties: fewer epochs); UVFA uses the
        extension config if G-P4(b) failed on the base config and the extension exists."""
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
        base = self.base() if self.gp2() is not None else None
        if base is None:
            return None
        sel = self.selection(labels, base)
        if sel is None:
            return None
        dev = self.dev()["DEV-ID"]
        sf_tag = self.zs_tag("SF", labels, base, "base", sel["SF"]["epoch"])
        uv_tag = self.zs_tag("UVFA", labels, base, "base", sel["UVFA"]["epoch"])
        need = [("SF-own", sf_tag), ("SF-noGPI", sf_tag), ("UVFA", uv_tag), ("COPY", "-"), ("GREEDY", "-")]
        if not all(kj("bank", "pilot", a, t, p.key) in self.banks for a, t in need for p in dev):
            return None
        copy = acc_half(self.banks, "pilot", "COPY", "-", dev)
        a_acc = acc_half(self.banks, "pilot", "SF-own", sf_tag, dev) - copy
        a_top = top1_rate(self.banks, "pilot", "SF-noGPI", sf_tag, dev) - top1_rate(self.banks, "pilot", "GREEDY", "-", dev)
        a = dict(acc_delta=a_acc, top1_delta=a_top, copy_acc=copy, ok=ge(a_acc, 0.40) and ge(a_top, 0.20))
        b_acc = acc_half(self.banks, "pilot", "UVFA", uv_tag, dev) - copy
        b_nreg = sel["UVFA"]["dev_id"]
        b = dict(acc_delta=b_acc, dev_nreg=b_nreg, ok=ge(b_acc, 0.20) and le(b_nreg, BAR), cfg="base",
                 epoch=sel["UVFA"]["epoch"])
        if not b["ok"] and final_uvfa:
            ext = self.uvfa_ext(labels, base, copy)
            b["extension"] = ext
            if ext is not None and ext["ok"]:
                b.update(ok=True, cfg="ext", epoch=ZS_EXT["keep"][-1])
        return dict(labels=labels, base=base, selection=sel, a=a, b=b)

    def uvfa_ext(self, labels, base, copy):
        e = ZS_EXT["keep"][-1]
        v = self.zs_mean("UVFA", "DEV-ID", labels, base, "ext", e)
        tag = self.zs_tag("UVFA", labels, base, "ext", e)
        dev = self.dev()["DEV-ID"]
        if v is None or not all(kj("bank", "pilot", "UVFA", tag, p.key) in self.banks for p in dev):
            return None
        acc = acc_half(self.banks, "pilot", "UVFA", tag, dev) - copy
        return dict(acc_delta=acc, dev_nreg=v, ok=ge(acc, 0.20) and le(v, BAR))

    def final_configs(self):
        """{model: (labels, base, cfg, epoch)} at the final stage, or None while G-P4 is pending; 'n6' if (a) failed
        at 800 labels."""
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
        out = {m: (L, base, "base", s["epoch"]) for m, s in sel.items()}
        if g["b"]["cfg"] == "ext":
            out["UVFA"] = (L, base, "ext", ZS_EXT["keep"][-1])
        return out

    # ---- bank units (G-P4)
    def bank_unit(self, arm, p, labels, base, cfg, epoch):
        def run(sl):
            d = self.data()
            tab, stop, w, c = d.table(p), d.arm_stop(p), p.weights(), p.c
            if arm == "COPY":
                return bank_unit(self.banks, d, sl, split="pilot", p=p, arm=arm, tag="-", choose=None,
                                 vhat=lambda S: stop[S], digest=self.digest)
            if arm == "GREEDY":
                return bank_unit(self.banks, d, sl, split="pilot", p=p, arm=arm, tag="-",
                                 choose=SO.greedy_policy(stop, c), vhat=None, digest=self.digest)
            model_name = "SF" if arm.startswith("SF") else "UVFA"
            tag = self.zs_tag(model_name, labels, base, cfg, epoch)
            model = load_model(self.model_path(model_name, labels, base, cfg, epoch))
            if arm == "SF-own":
                vh = lambda S: SO.wbar_dot(psi_hat(model, d, S, w[None], np.array([c]))[:, 0], w, c)  # noqa: E731
                return bank_unit(self.banks, d, sl, split="pilot", p=p, arm=arm, tag=tag, choose=None, vhat=vh,
                                 digest=self.digest)
            if arm == "SF-noGPI":
                return bank_unit(self.banks, d, sl, split="pilot", p=p, arm=arm, tag=tag,
                                 choose=learned_policy("SF-noGPI", model, d, p, base), vhat=None, digest=self.digest)
            vh = lambda S: value_hat(model, d, S, w[None], np.array([c]))[:, 0]  # noqa: E731
            return bank_unit(self.banks, d, sl, split="pilot", p=p, arm="UVFA", tag=tag,
                             choose=learned_policy("UVFA", model, d, p, base), vhat=vh, digest=self.digest)
        return run

    # ---- plan of pilot A
    def plan_a(self) -> list:
        if not self.manifest_path.exists():
            return [Unit("draw", 180.0, self.unit_draw)]
        d, U = self.data(), []
        vi = 0.3
        if not self.lib_ready(False):
            return [Unit("library:train", vi * len(d.fam["TRAIN"]), self.unit_library("train"))]
        for q, cost in (("Q1", 30.0), ("Q2", 30.0), ("Q3", 120.0), ("Q5", 60.0), ("Q6", 5.0), ("Q8", 60.0)):
            if self.check(q) is None:
                U.append(Unit(f"check:{q}", cost, self.unit_check(q)))
        dev = self.dev()
        for arm in ("EXACT-GPI(L)", "ORACLE", "STOP", "RAND", "GREEDY", "ENUM-MIS±"):
            for f, ps in dev.items():
                for p in ps:
                    if not cells_done(self.cells, "pilot", arm, self.ref_tag(arm, False), p):
                        U.append(Unit(f"ref:{arm}:{f}:{p.key}", 16 * 0.5, self.ref_unit(arm, f, p, False)))
        g2 = self.gp2()
        if g2 is None:
            return U
        base = g2["base"]
        if base and not self.lib_ready(True):
            U.append(Unit("library:base", vi * d.n * d.n, self.unit_library("base")))
            return U
        if self.check("Q4") is None:
            U.append(Unit("check:Q4", 30.0, self.unit_check("Q4")))
        if base:
            for f, ps in dev.items():
                for p in ps:
                    if not cells_done(self.cells, "pilot", "EXACT-GPI(L)", "L+base", p):
                        U.append(Unit(f"ref:EXACT-GPI(L)+base:{f}:{p.key}", 8.0, self.ref_unit("EXACT-GPI(L)", f, p,
                                                                                               True)))
        for labels in (400, 800):
            if labels == 800 and self.stage_labels() != 800:
                break
            U += self.stage_units(labels, base, U)
            if U:
                return U
        fc = self.final_configs()
        if fc in (None, "n6"):
            return U
        for arm in ZS_PILOT:  # DEV-OOD at the selected configs
            L, b, cfg, e = fc[ARM_MODEL[arm]]
            for f, ps in dev.items():
                if f == "DEV-ID":
                    continue
                for p in ps:
                    if not cells_done(self.cells, "pilot", arm, self.zs_tag(ARM_MODEL[arm], L, b, cfg, e), p):
                        U.append(Unit(f"zs:{arm}:{f}:{p.key}", 16 * 1.0, self.zs_unit(arm, f, p, L, b, cfg, e)))
        return U

    def stage_units(self, labels, base, U0) -> list:
        U = []
        if not self.data_path(labels, base).exists():
            return [Unit(f"traindata:L{labels}", 60.0, self.unit_train_data(labels, base))]
        steps = ZS_BASE["epochs"] * len(self.m["train_labels"]) * labels / 256
        for model in ("SF", "UVFA", "UVFA+aux"):
            if self.zs_key(model, labels, base, "base") not in self.fits:
                U.append(Unit(f"train:{model}:L{labels}", steps * 0.05, self.unit_train(model, labels, base, "base")))
        if U:
            return U
        dev = self.dev()["DEV-ID"]
        for arm in ZS_PILOT:
            for e in ZS_BASE["keep"]:
                for p in dev:
                    if not cells_done(self.cells, "pilot", arm, self.zs_tag(ARM_MODEL[arm], labels, base, "base", e), p):
                        U.append(Unit(f"zs:{arm}:e{e}:DEV-ID:{p.key}", 16.0, self.zs_unit(arm, "DEV-ID", p, labels,
                                                                                          base, "base", e)))
        if U:
            return U
        sel = self.selection(labels, base)
        for arm, model in (("SF-own", "SF"), ("SF-noGPI", "SF"), ("UVFA", "UVFA"), ("COPY", None),
                           ("GREEDY", None)):
            e = sel[model]["epoch"] if model else None
            tag = self.zs_tag(model, labels, base, "base", e) if model else "-"
            for p in dev:
                if kj("bank", "pilot", arm, tag, p.key) not in self.banks:
                    U.append(Unit(f"bank:{arm}:{p.key}", 10.0, self.bank_unit(arm, p, labels, base, "base", e)))
        if U:
            return U
        g = self.gp4(labels, final_uvfa=False)
        if not g["b"]["ok"]:  # the one UVFA extension (80 epochs, lr 3e-4)
            e = ZS_EXT["keep"][-1]
            if self.zs_key("UVFA", labels, base, "ext") not in self.fits:
                return [Unit(f"train:UVFA-ext:L{labels}", steps * 2 * 0.05, self.unit_train("UVFA", labels, base, "ext"))]
            tag = self.zs_tag("UVFA", labels, base, "ext", e)
            for p in dev:
                if not cells_done(self.cells, "pilot", "UVFA", tag, p):
                    U.append(Unit(f"zs:UVFA-ext:{p.key}", 16.0, self.zs_unit("UVFA", "DEV-ID", p, labels, base, "ext", e)))
            for p in dev:
                if kj("bank", "pilot", "UVFA", tag, p.key) not in self.banks:
                    U.append(Unit(f"bank:UVFA-ext:{p.key}", 10.0, self.bank_unit("UVFA", p, labels, base, "ext", e)))
        return U

    # ---- pilot B: per-goal grids, k-curve, M5, G-P5
    def pg_key(self, kind, cfg, k, scope):
        return kj("fit", kind, cfg, k, scope)

    def pg_jobs(self, kind, ps, k, cfg_name):
        d, fc = self.data(), self.final_configs()
        jobs = []
        for p in ps:
            labels = self.m["pergoal_labels"][p.key][:k]
            tab = d.table(p)
            job = dict(key=p.key, states=np.asarray(labels, dtype=np.int64), v=tab.V[np.asarray(labels, dtype=np.int64)],
                       stab=FA.stabilizer(np.asarray(p.w), d.n), w=p.weights(), c=p.c,
                       seed=FA.derive(SEEDS["selection"], f"{kind}:{cfg_name}:{k}:{p.key}"))
            if kind == "FT":
                L, b, cfg, e = fc["UVFA"]
                job["init"] = self.model_path("UVFA", L, b, cfg, e)
            jobs.append(job)
        return jobs

    def unit_pg(self, kind, cfg_name, cfg, ps, k, scope):
        def run(sl):
            out = self.dir / "models" / f"{kind}_{cfg_name}_k{k}_{scope}"
            res = fit_pergoal(kind, self.data(), self.pg_jobs(kind, ps, k, cfg_name), cfg, out_dir=out,
                              safe_point=sl.safe_point if sl is not None else (lambda kind="step": False))
            if res is None:
                return False
            models = {str(key): str(v) for key, v in res.items()}
            self.fits.add(self.pg_key(kind, cfg_name, k, scope), dict(kind=kind, cfg=cfg_name, k=k, scope=scope,
                                                                      models=models), self.digest)
            return True
        return run

    def pg_unit_eval(self, kind, cfg_name, k, scope, f, p):
        def run(sl):
            d = self.data()
            path = self.fits.get(self.pg_key(kind, cfg_name, k, scope))["models"][p.key]
            model = load_model(path)
            stop, w, c = d.arm_stop(p), p.weights(), p.c
            pol = SO.scored_policy(stop, lambda s, a, dsts: value_hat(model, d, dsts, w[None], np.array([c]))[:, 0] - c)
            return eval_cells(self.cells, d, sl, split="pilot", family=f, p=p, arm=kind, tag=f"{cfg_name}|k{k}",
                              policy_fn=lambda s0: (pol, None), digest=self.digest)
        return run

    def pg_mean(self, kind, cfg_name, k, ps):
        if not ps or not all(cells_done(self.cells, "pilot", kind, f"{cfg_name}|k{k}", p) for p in ps):
            return None
        return family_mean(self.cells, "pilot", kind, f"{cfg_name}|k{k}", ps)

    def _pg_stage(self, U, kind, grid, base_cfg, ps, k, scope, f):
        """Append fit and eval units of a grid (fit configs share a run when they differ only in `keep`)."""
        for name, spec in grid.items():
            cfg = dict(base_cfg, **spec)
            if self.pg_key(kind, name, k, scope) not in self.fits:
                U.append(Unit(f"fit:{kind}:{name}:k{k}:{scope}", 0.02 * cfg["steps"] * len(ps),
                              self.unit_pg(kind, name, cfg, ps, k, scope)))
                continue
            for p in ps:
                if not cells_done(self.cells, "pilot", kind, f"{name}|k{k}", p):
                    U.append(Unit(f"pg:{kind}:{name}:k{k}:{p.key}", 16.0, self.pg_unit_eval(kind, name, k, scope, f, p)))

    def gp3(self):
        """G-P3 (None until computable): best PGV and FT configs on DEV-ID, competence <= 0.50, PG400, extensions."""
        dev = self.dev().get("DEV-ID", [])
        vals = {}
        for kind, grid in (("PGV", PGV_GRID), ("FT", FT_GRID)):
            for name in grid:
                vals[(kind, name)] = self.pg_mean(kind, name, 400, dev)
        if any(v is None for v in vals.values()):
            return None
        best = {kind: min(((v, i, name) for i, ((kd, name), v) in enumerate(vals.items()) if kd == kind))
                for kind in ("PGV", "FT")}
        res = {kind: dict(cfg=b[2], dev_id=b[0], competent=le(b[0], BAR)) for kind, b in best.items()}
        extended = False
        if not any(r["competent"] for r in res.values()):
            ext = {"PGV": self.pg_mean("PGV", "w128s4000", 400, dev), "FT": self.pg_mean("FT", "s1000", 400, dev)}
            if any(v is None for v in ext.values()):
                return dict(grid=res, extension_pending=True)
            extended = True
            for kind, name in (("PGV", "w128s4000"), ("FT", "s1000")):
                if lt(ext[kind], res[kind]["dev_id"]):
                    res[kind] = dict(cfg=name, dev_id=ext[kind], competent=le(ext[kind], BAR))
        comp = [k for k in ("FT", "PGV") if res[k]["competent"]]
        pg = None if not comp else min(comp, key=lambda k: (res[k]["dev_id"], 0 if k == "FT" else 1))
        return dict(grid=res, extended=extended, pg400=pg, p2_dropped=pg is None, candidates={f"{k}:{n}": v for (k, n), v
                                                                                              in vals.items()})

    def gp3b(self):
        g3 = self.gp3()
        if g3 is None or g3.get("extension_pending"):
            return None
        kdev = [self.data().by_key[k] for k in self.m["dev_kcurve"]]
        pcfg = g3["grid"]["PGV"]["cfg"]
        m = {400: self.pg_mean("PGV", pcfg, 400, kdev), 1600: self.pg_mean("PGV", "k1600", 1600, kdev),
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
        U, dev = [], self.dev()["DEV-ID"]
        self._pg_stage(U, "PGV", PGV_GRID, PGV_COMMON, dev, 400, "dev", "DEV-ID")
        self._pg_stage(U, "FT", FT_GRID, FT_COMMON, dev, 400, "dev", "DEV-ID")
        if U:
            return U
        g3 = self.gp3()
        if g3.get("extension_pending"):
            self._pg_stage(U, "PGV", PGV_EXT, {}, dev, 400, "dev", "DEV-ID")
            self._pg_stage(U, "FT", FT_EXT, {}, dev, 400, "dev", "DEV-ID")
            return U
        kdev = [self.data().by_key[k] for k in self.m["dev_kcurve"]]
        for k, spec in KCURVE.items():
            self._pg_stage(U, "PGV", {f"k{k}": spec}, {}, kdev, k, "kdev", "DEV-ID")
        if U:
            return U
        g3b = self.gp3b()
        if g3b.get("rext_pending"):
            for k, spec in R_EXT.items():
                self._pg_stage(U, "PGV", {f"rext{k}": spec}, {}, kdev, k, "kdev", "DEV-ID")
            return U
        if not g3["p2_dropped"]:  # M5: the selected PG400 on the 24 DEV-OOD H-K/H-NEG parents
            ood = [(f, p) for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG") for p in self.dev().get(f, [])]
            kind = g3["pg400"]
            name = g3["grid"][kind]["cfg"]
            spec = {**(PGV_GRID if kind == "PGV" else FT_GRID), **PGV_EXT, **FT_EXT}[name]
            common = PGV_COMMON if name in PGV_GRID else (FT_COMMON if name in FT_GRID else {})
            ps = [p for _, p in ood]
            if self.pg_key(kind, name, 400, "ood") not in self.fits:
                U.append(Unit(f"fit:{kind}:{name}:ood", 0.02 * spec["steps"] * len(ps),
                              self.unit_pg(kind, name, dict(common, **spec), ps, 400, "ood")))
                return U
            for f, p in ood:
                if not cells_done(self.cells, "pilot", kind, f"{name}|k400", p):
                    U.append(Unit(f"pg:{kind}:{name}:ood:{p.key}", 16.0, self.pg_unit_eval(kind, name, 400, "ood", f, p)))
            if U:
                return U
        if kj("bench") not in self.checks:
            U.append(Unit("bench:G-P5", 120.0, self.unit_bench))
        return U

    def unit_bench(self, sl) -> bool:
        """G-P5: optimizer-step seconds per (model, B, batch) (learners_e05.time_step), VI seconds per weight, and
        evaluation seconds per cell per arm (from the recorded pilot cells; ENUM± per family)."""
        d = self.data()
        step_s = {}
        for m, B, bs in LA.STEP_UNITS:
            width = {"PGV64": 64, "PGV128": 128, "PGV256": 256}.get(m, 256)
            step_s[LA.step_name(m, B, bs)] = time_step(m, width, bs, B)
        t0 = time.perf_counter()
        for p in d.fam["TRAIN"][:5]:
            SO.sf_table(d.space, SO.stop_values(d.phi, p.weights()), p.c)
        vi_s = (time.perf_counter() - t0) / 5
        cell = defaultdict(list)
        for r in self.cells.rows.values():
            cell[r["arm"]].append(r["dt"])
            if r["arm"].startswith("ENUM") or r["arm"].startswith("GNN"):
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
Z_P1, Z_P2, Z_BETA = 2.2414, 1.96, 0.8416
N_ID_RANGE, N_K_RANGE = (96, 152), (56, 72)
SECONDARY_SIZES = {"H-COST": 40, "H-SPAN": 40, "H-KN": 24, "H-CONJ": 40, "H-DYN": 40}
CUT_ORDER = ("budget-curve", "SF-FT_400", "H-KN", "GNN-ENUM±@H-NEG", "FT-UVFA_400-1rep", "ENUM-H*±@H-NEG-24",
             "H-DYN")
CAP_MIN = 150.0


def macro_se(sigma, n_id, n_k, n_neg):
    return (sigma / 3.0) * math.sqrt(1.0 / n_id + 1.0 / n_k + 1.0 / n_neg)


def power_p1(sigma, n_id, n_k=None, n_neg=None, effect=0.05, macro=True) -> float:
    """§8: Phi((effect - max(z SE, 0.03)) / SE), z = 2.2414; macro SE = (sigma/3) sqrt(sum 1/n), H-ID SE = sigma/sqrt n."""
    se = macro_se(sigma, n_id, n_k, n_neg) if macro else sigma / math.sqrt(n_id)
    return PHI((effect - max(Z_P1 * se, 0.03)) / se)


def power_p2(sigma2, n_id, n_k, n_neg, d2=0.0, delta=0.05) -> float:
    """§8 NI power: Phi((delta - D2) / SE - 1.96) with the macro SE."""
    se = macro_se(sigma2, n_id, n_k, n_neg)
    return PHI((delta - d2) / se - Z_P2)


def n_id_for(sigma_hat_id: float) -> int:
    """§8 step 3: min(152, max(96, ceil(3,802 (1.15 sigma-hat)^2)))."""
    sp = 1.15 * sigma_hat_id
    return int(min(N_ID_RANGE[1], max(N_ID_RANGE[0], math.ceil(3802.0 * sp * sp - TOL))))


def size_plan(s_id, s_ood, s2_id, s2_ood, *, p2_dropped, fits=lambda n_id, n_k: True, max_n_id=None) -> dict:
    """§8 steps 1-5. fits(n_id, n_k) = the protected core fits 150 CPU-min; max_n_id = n_S + available unseen classes."""
    sp1 = 1.15 * max(s_id, s_ood)
    sp2 = None if p2_dropped else 1.15 * max(s2_id, s2_ood)
    n_id = n_id_for(s_id)

    def n_k_for(n_id):
        low = power_p1(sp1, n_id, 56, 56) < 0.80 - TOL or (sp2 is not None and power_p2(sp2, n_id, 56, 56) < 0.80 - TOL)
        return 72 if low else 56
    n_k = n_k_for(n_id)
    capped = []
    if max_n_id is not None and n_id > max_n_id:
        n_id = max(N_ID_RANGE[0], max_n_id)
        capped.append("unseen support classes")
    while not fits(n_id, n_k) and n_id > N_ID_RANGE[0]:
        n_id -= 1
        capped.append("cpu") if "cpu" not in capped else None
    if not fits(n_id, n_k) and n_k > N_K_RANGE[0]:
        n_k = N_K_RANGE[0]
        capped.append("cpu (n_K)")
    pw = dict(p1_macro=power_p1(sp1, n_id, n_k, n_k), p1_id=power_p1(1.15 * s_id, n_id, macro=False),
              p1_id_at_003=power_p1(1.15 * s_id, n_id, macro=False, effect=0.03),
              p1_id_at_004=power_p1(1.15 * s_id, n_id, macro=False, effect=0.04))
    if sp2 is not None:
        pw.update(p2_d0=power_p2(sp2, n_id, n_k, n_k), p2_d002=power_p2(sp2, n_id, n_k, n_k, d2=0.02))
    return dict(sigma_plan_id=1.15 * s_id, sigma_plan_p1m=sp1, sigma_plan_2=sp2, n_id=n_id, n_k=n_k, n_neg=n_k,
                caps=capped, fits=fits(n_id, n_k), powers=pw)


def project_confirmation(rates: dict, n_id: int, n_k: int, n_neg: int, cfg: dict, cuts=()) -> dict:
    """Projected confirmation CPU-min per item at the G-P5 rates (PREREG §10.3; DESIGN §9.1 items 2-8). rates:
    step_s {step_name: s}, vi_s, cell_s {arm or arm@family: {"mean"}}. cfg: zs_steps (per run), pgv (cfg dict),
    ft_steps, pg400 ("PGV"|"FT"|None), rext (bool), labels_per_goal. Returns {"items": {name: min}, "total",
    "protected"}."""
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
        base = FA.base_family(fam) if fam else None
        v = cs.get(f"{arm}@{base}") or cs.get(arm) or {"mean": 1.0}
        x = v["mean"]
        if fam == "H-KN" and arm.endswith("±") and f"{arm}@H-KN" not in cs:
            x *= 4.0  # H-KN menus: up to 3,600 targets vs H-NEG's 900 (not in DEV; DESIGN §9.1 note)
        return x
    it = {}
    it["test-oracle"] = 2 * 1.5 * n_all * rates["vi_s"] / 60.0  # draw-test + oracle-test (incl. rejected draws)
    zs = cfg["zs_steps"]
    it["train"] = 3 * sum(zs[m] * step(m, 1, 256) for m in ("SF", "UVFA", "UVFA+aux")) / 60.0 + zs["SF"] * step("SF", 1, 256) / 60.0
    pw = {64: "PGV64", 128: "PGV128", 256: "PGV256"}[cfg["pgv"]["width"]]
    it["PGV_400"] = n_p * cfg["pgv"]["steps"] * step(pw, 64, 64) / 64 / 60.0
    ft_reps = 1 if ("FT-UVFA_400-1rep" in cuts and cfg["pg400"] != "FT") else 3
    it["FT-UVFA_400"] = ft_reps * n_p * cfg["ft_steps"] * step("UVFA", 64, 64) / 64 / 60.0
    if cfg.get("rext"):
        it["k-curve"] = 48 * 4000 * sum(step("PGV128", 64, b) for b in (64, 128, 256)) / 64 / 60.0
    else:
        it["k-curve"] = 48 * (2000 * step("PGV256", 64, 128) + 2500 * step("PGV256", 64, 256)) / 64 / 60.0
    zs_arms = ("SF-GPI", "SF-noGPI", "SF-L", "SF-GPI-5", "UVFA", "UVFA+aux")
    ev = 0.0
    for f, n in fams.items():
        ev += 16 * n * (3 * sum(cell(a) for a in zs_arms) + cell("SF-shuffled") if True else 0) / 60.0
        ev += 16 * n * sum(cell(a, f) for a in ("ENUM-MIS±", "GREEDY", "STOP", "RAND", "ORACLE", "EXACT-GPI(L)",
                                               "BEST-SINGLE(L)")) / 60.0
        ev += 16 * n * (cell("PGV") + 3 * cell("FT-UVFA")) / 60.0 if f in ("H-ID", "H-K", "H-NEG") else 0.0
    it["eval"] = ev
    for arm, fams_on in (("ENUM-H*±", ("H-ID", "H-K", "H-NEG")), ("GNN-ENUM±", ("H-ID", "H-K", "H-NEG"))):
        for f in fams_on:
            n = fams[f]
            if f == "H-NEG" and arm == "GNN-ENUM±" and "GNN-ENUM±@H-NEG" in cuts:
                continue
            if f == "H-NEG" and arm == "ENUM-H*±" and "ENUM-H*±@H-NEG-24" in cuts:
                n = min(n, 24)
            it[f"{arm}@{f}"] = 16 * n * cell(arm, f) / 60.0
    it["banks"] = 64 * n_all * 0.01 * len(zs_arms) / 60.0
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
    """All pilot gates and the PILOT FILL values (§9). Raises SystemExit if the pilot is incomplete."""
    if P.plan_a() or P.plan_b():
        raise SystemExit("refusing: the pilot phases are not complete (pilot-a / pilot-b still have pending units)")
    fc = P.final_configs()
    if fc == "n6":
        raise SystemExit("G-P4(a) failed at 400 and 800 labels: the n = 6 fallback applies (pilot-a --n 6)")
    d, dev = P.data(), P.dev()
    base = P.base()
    cells = P.cells
    rep = dict(n=d.n, digest_t2=P.digest, manifest_sha256=sha_file(P.manifest_path))
    # Q7 (B2) and Q9 (B1) on DEV
    q7 = dict(oracle_all_zero=all(cells.get(kj("cell", "pilot", "ORACLE", "-", p.key, int(s)))["nreg"] == 0.0
                                  for ps in dev.values() for p in ps for s in p.starts),
              stop_all_one=all(cells.get(kj("cell", "pilot", "STOP", "-", p.key, int(s)))["nreg"] == 1.0
                               for ps in dev.values() for p in ps for s in p.starts),
              greedy_identity=all(cells.get(kj("cell", "pilot", "GREEDY", "-", p.key, int(s)))["nreg"] == 1.0
                                  for f in POSITIVE_DEV for p in dev.get(f, []) for s in p.starts))
    prim = [f for f in ("DEV-ID", "DEV-OOD:H-K", "DEV-OOD:H-NEG") if f in dev]
    q7["rand_macro"] = mean(family_mean(cells, "pilot", "RAND", "-", dev[f]) for f in prim)
    q7["ok"] = q7["oracle_all_zero"] and q7["stop_all_one"] and q7["greedy_identity"] and lt(q7["rand_macro"], 1.0)
    neg = dev.get("DEV-OOD:H-NEG", [])
    tied = Counter()
    for p in neg:
        for s in p.starts:
            dg = cells.get(kj("cell", "pilot", "ENUM-MIS±", "-", p.key, int(s))).get("diag") or {}
            tied["decisions"] += int(dg.get("decisions", 0))
            tied["all_tied"] += int(dg.get("all_tied", 0))
    q9_enum, q9_greedy = family_mean(cells, "pilot", "ENUM-MIS±", "-", neg), family_mean(cells, "pilot", "GREEDY", "-", neg)
    share = tied["all_tied"] / tied["decisions"] if tied["decisions"] else 0.0
    q9 = dict(enum_mis=q9_enum, greedy=q9_greedy, all_tied_share=share, ok=lt(q9_enum, q9_greedy) and share < 0.5,
              note="a failure triggers the enum_e05 audit (G-P1); with no bug found the guard-quality qualifier applies")
    qs = {q: P.check(q) for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q8")}
    gp1 = dict(checks={q: dict(ok=bool(v["ok"]), value=v["value"]) for q, v in qs.items()}, Q7=q7, Q9=q9,
               ok_q1_q8=all(v["ok"] for v in qs.values()) and q7["ok"], ok=all(v["ok"] for v in qs.values())
               and q7["ok"] and q9["ok"])
    gp2 = P.gp2()
    g4 = P.gp4(400)
    gp4 = dict(at400=g4)
    if not g4["a"]["ok"]:
        gp4["at800"] = P.gp4(800)
    gfin = gp4.get("at800") or g4
    p1_invalid = not gfin["b"]["ok"]
    gp3, gp3b = P.gp3(), P.gp3b()
    # DEV values of every pilot arm (selected configs; reference arms)
    L, b = fc["SF"][0], fc["SF"][1]
    tags = {arm: P.zs_tag(ARM_MODEL[arm], *fc[ARM_MODEL[arm]]) for arm in ZS_PILOT}
    tags.update({a: P.ref_tag(a, False) for a in REF_PILOT})
    if base:
        tags["EXACT-GPI(L)+base"] = "L+base"
    table = {}
    for arm, tag in tags.items():
        a = arm.replace("+base", "")
        table[arm] = {f: family_mean(cells, "pilot", a, tag, ps) for f, ps in dev.items()
                      if all(cells_done(cells, "pilot", a, tag, p) for p in ps)}
    for kind, grid in (("PGV", {**PGV_GRID, **PGV_EXT}), ("FT", {**FT_GRID, **FT_EXT})):
        for name in grid:
            v = P.pg_mean(kind, name, 400, dev["DEV-ID"])
            if v is not None:
                table[f"{kind}_400:{name}"] = {"DEV-ID": v}
    pg = gp3["pg400"]
    pg_tag = None if pg is None else f"{gp3['grid'][pg]['cfg']}|k400"
    if pg is not None:
        for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG"):
            if f in dev and all(cells_done(cells, "pilot", pg, pg_tag, p) for p in dev[f]):
                table.setdefault(f"PG400={pg}", {})[f] = family_mean(cells, "pilot", pg, pg_tag, dev[f])

    def y(arm, tag, p):
        return parent_nreg(cells, "pilot", arm, tag, p)
    ood24 = [p for f in ("DEV-OOD:H-K", "DEV-OOD:H-NEG") for p in dev.get(f, [])]
    did = dev["DEV-ID"]
    sd = lambda xs: stdev(xs) if len(xs) > 1 else 0.0  # noqa: E731
    s_id = sd([y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in did])
    s_ood = sd([y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in ood24])
    s2_id = s2_ood = None
    if pg is not None:
        s2_id = sd([y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in did])
        s2_ood = sd([y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in ood24])
    dev_contrasts = {}
    for f, ps in dev.items():
        if all(cells_done(cells, "pilot", "UVFA", tags["UVFA"], p) and cells_done(cells, "pilot", "SF-GPI", tags["SF-GPI"], p)
               for p in ps):
            dev_contrasts[f] = dict(D1=mean(y("UVFA", tags["UVFA"], p) - y("SF-GPI", tags["SF-GPI"], p) for p in ps),
                                    D_G=mean(y("SF-GPI", tags["SF-GPI"], p) - y("ENUM-MIS±", "-", p) for p in ps))
            if pg is not None and all(cells_done(cells, "pilot", pg, pg_tag, p) for p in ps):
                dev_contrasts[f]["D2"] = mean(y("SF-GPI", tags["SF-GPI"], p) - y(pg, pg_tag, p) for p in ps)
    # G-P5 rates, cuts and sizing
    bench = rates_override or P.checks.get(kj("bench"))
    pgv_cfg = dict(PGV_COMMON, **{**PGV_GRID, **PGV_EXT}[gp3["grid"]["PGV"]["cfg"]])
    ft_spec = {**FT_GRID, **FT_EXT}[gp3["grid"]["FT"]["cfg"]]
    zs_steps = {m: (ZS_EXT if (m == "UVFA" and fc["UVFA"][2] == "ext") else ZS_BASE)["epochs"]
                * (len(d.fam["TRAIN"]) + (d.n * d.n if base else 0)) * L / 256 for m in ("SF", "UVFA", "UVFA+aux")}
    pcfg = dict(zs_steps=zs_steps, pgv=dict(pgv_cfg, steps=pgv_cfg["keep"]), ft_steps=ft_spec["keep"], pg400=pg,
                rext=gp3b["recipe"] == "R_ext")
    reg = FA.Registry.from_parents(d.n, [p for ps in d.fam.values() for p in ps])
    lib_k = [p for p in d.fam["TRAIN"] if p.K in (2, 3, 4)]
    seen = {p.supp_u for p in lib_k}
    blocked = {p.supp_u for p in lib_k} | {p.supp_u for p in dev["DEV-ID"] if p.K in (2, 3, 4)}
    total_cls = sum(FA.n_support_classes(d.n, K) for K in (2, 3, 4))
    unseen_avail = total_cls - len(blocked)
    n_s = min(FA.SIZES["hid_s_max"], len(seen))
    del reg

    def fits(n_id, n_k):
        return le(choose_cuts(lambda c: project_confirmation(bench, n_id, n_k, n_k, pcfg, c))[1]["protected"], CAP_MIN)
    plan = size_plan(s_id, s_ood, s2_id or 0.0, s2_ood or 0.0, p2_dropped=pg is None, fits=fits,
                     max_n_id=n_s + unseen_avail)
    cuts, proj = choose_cuts(lambda c: project_confirmation(bench, plan["n_id"], plan["n_k"], plan["n_neg"], pcfg, c))
    units = LA.plan_costs(bench["step_s"], {f"cell:{a.split('@')[0]}": v["max"] for a, v in bench["cell_s"].items()})
    unit_check = LA.validate_units(units)
    gp5 = dict(rates=bench, projection=proj, cuts=cuts, units=unit_check, ok=unit_check["ok"] and le(proj["total"],
                                                                                                     CAP_MIN))
    rep.update(
        G_P1=gp1, G_P2=gp2, G_P3=gp3, G_P3b=gp3b, G_P4=gp4, G_P5=gp5,
        library=dict(size=len(d.fam["TRAIN"]) + (d.n * d.n if base else 0), base_policies=base),
        train_labels_per_goal=L, selected=fc, p1_comparator_invalid=p1_invalid, p2_dropped=gp3["p2_dropped"],
        pg400=pg, kstar=gp3b["kstar"], kcurve_recipe=gp3b["recipe"], dev_nreg=table, dev_contrasts=dev_contrasts,
        sigma_hat=dict(id=s_id, ood=s_ood, two_id=s2_id, two_ood=s2_ood), sizing=plan,
        h_id=dict(n_s=n_s, n_u=plan["n_id"] - n_s, available_seen=len(seen), available_unseen=unseen_avail),
        exclusions=P.m.get("exclusions"),
        reference_shas=LA.reference_shas())
    rep["fill"] = fill_fields(rep)
    return rep


def fill_fields(r: dict) -> dict:
    """The §9.2 PILOT FILL rows, as text copied verbatim into PREREG_E05.md at T3."""
    g1, g4, g3, g3b, g5, sz = r["G_P1"], r["G_P4"], r["G_P3"], r["G_P3b"], r["G_P5"], r["sizing"]
    f = {}
    f["digests"] = f"T2 digest {r['digest_t2']}; PILOT_MANIFEST.json sha256 {r['manifest_sha256']}"
    f["n_labels"] = f"n = {r['n']}; TRAIN labels per goal {r['train_labels_per_goal']}"
    f["library"] = (f"{r['library']['size']} ({'64 + 49' if r['library']['base_policies'] else '64'}); G-P2 DEV-ID "
                    f"EXACT-GPI(L) nreg {r['G_P2']['value']:.4f}")
    f["G_P1"] = (f"Q7 ORACLE all 0: {g1['Q7']['oracle_all_zero']}, STOP all 1: {g1['Q7']['stop_all_one']}, RAND DEV "
                 f"primary macro {g1['Q7']['rand_macro']:.4f}, GREEDY identity: {g1['Q7']['greedy_identity']}; Q9 "
                 f"ENUM-MIS± {g1['Q9']['enum_mis']:.4f} vs GREEDY {g1['Q9']['greedy']:.4f}, all-tied share "
                 f"{g1['Q9']['all_tied_share']:.4f} ({'pass' if g1['Q9']['ok'] else 'FAIL: audit'}); Q1-Q6, Q8: "
                 + ", ".join(f"{q} {'pass' if v['ok'] else 'FAIL'}" for q, v in g1["checks"].items()))
    gf = g4.get("at800") or g4["at400"]
    f["selection_G_P4"] = (f"selected {json.dumps(r['selected'])}; UVFA extension used: {gf['b']['cfg'] == 'ext'}; "
                           f"G-P4 acc½ delta SF {gf['a']['acc_delta']:.4f}, top-1 delta {gf['a']['top1_delta']:.4f}, "
                           f"UVFA acc½ delta {gf['b']['acc_delta']:.4f}, UVFA DEV-ID nreg {gf['b']['dev_nreg']:.4f}; "
                           f"{'P1 comparator-invalid' if r['p1_comparator_invalid'] else 'P1 valid'}")
    f["G_P3"] = (f"PGV_400 {g3['grid']['PGV']['cfg']} DEV-ID {g3['grid']['PGV']['dev_id']:.4f}, FT-UVFA_400 "
                 f"{g3['grid']['FT']['cfg']} {g3['grid']['FT']['dev_id']:.4f} (bar 0.50); extensions used: "
                 f"{g3.get('extended', False)}; PG400 = {r['pg400'] or 'P2 dropped'}")
    f["G_P3b"] = (f"original k-curve {json.dumps(g3b['original'])} pass {g3b['original_ok']}; R_ext used: "
                  f"{'rext' in g3b}; k* {g3b['kstar']}")
    f["dev_nreg"] = json.dumps(r["dev_nreg"], sort_keys=True) + "; DEV contrasts " + json.dumps(r["dev_contrasts"])
    f["sigma"] = json.dumps(r["sigma_hat"]) + "; plan " + json.dumps({k: sz[k] for k in ("sigma_plan_id",
                                                                                         "sigma_plan_p1m",
                                                                                         "sigma_plan_2")})
    f["sizes"] = (f"n_ID {sz['n_id']}, n_S {r['h_id']['n_s']}, n_U {r['h_id']['n_u']}, n_K = n_NEG {sz['n_k']}; "
                  f"available unseen support classes {r['h_id']['available_unseen']}")
    f["powers"] = json.dumps(sz["powers"])
    f["G_P5"] = (f"projected confirmation {g5['projection']['total']:.1f} CPU-min; cuts {g5['cuts']}; largest units "
                 f"{json.dumps(g5['units'])}")
    f["exclusions"] = json.dumps(r["exclusions"])
    f["reference_shas"] = json.dumps(r["reference_shas"])
    f["T3"] = "<filled at T3: full digest incl. analyze_e05.py; test pass log>"
    return f


# ============================================================================= TEST side (§10.1 steps 5-8)

def pilot_report_or_refuse(ctx) -> dict:
    p = ctx.pdir / "PILOT_REPORT.json"
    if not p.exists():
        raise SystemExit("refusing: pilot/PILOT_REPORT.json does not exist")
    return read_json(p)


def test_manifest(ctx, k: int, manifest_sha: str) -> dict:
    p = ctx.adir(k) / "TEST_MANIFEST.json"
    if not p.exists():
        raise SystemExit(f"refusing: attempt {k} has no TEST_MANIFEST.json")
    sha = sha_file(p)
    if sha != manifest_sha:
        raise SystemExit(f"refusing: TEST_MANIFEST.json sha256 {sha[:12]}... != --manifest-sha {manifest_sha[:12]}...")
    require_committed(manifest_sha, "TEST_MANIFEST.json sha256", ctx.decisions())
    return read_json(p)


def oracle_report(ctx, k: int) -> dict | None:
    p = ctx.adir(k) / "ORACLE_TEST_REPORT.json"
    return read_json(p) if p.exists() else None


def require_oracle_pass(ctx, k: int) -> dict:
    r = oracle_report(ctx, k)
    if r is None or r.get("status") != "PASS":
        raise SystemExit(f"refusing: attempt {k} ORACLE_TEST_REPORT.json is "
                         f"{'missing' if r is None else r.get('status')!r}, not PASS (PREREG §2.2, B4)")
    return r


def first_learned(ctx, k: int) -> bool:
    return any(e.get("event") == "first-learned-test" for e in attempt_events(ctx, k))


def check_attempt_allowed(ctx, k: int) -> None:
    """Attempt k >= 1; k > 1 only after attempt k - 1 ended VOID inside oracle-test; at most two redraws."""
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

    def run(sl):
        pm = read_json(ctx.pdir / "PILOT_MANIFEST.json")
        n = int(pm["n"])
        space = SP.get_space(n)
        phi = SO.get_phi(space)
        try:
            excl = ctx.exclusions(space)
        except Exception as e:  # load_exclusions fails closed (§4.4 L6): refuse before draw-test
            raise SystemExit(f"refusing: exclusions do not load ({e!r}); restore or re-pin by a logged deviation")
        reg = FA.Registry.from_parents(n, [FA.Parent.from_json(x) for ps in pm["parents"].values() for x in ps])
        sz = rep["sizing"]
        cuts = set(rep["G_P5"]["cuts"])
        sizes = dict(ctx.sizes or {})
        reg = FA.draw_test(space, reg, k, sz["n_id"], sz["n_k"], sz["n_neg"], phi, excl, sizes or None)
        FA.assert_splits(reg)
        subset = FA.pgv_subset(reg)
        labels = {}
        kmax = max(FA.SIZES["kcurve"])
        for f in FA.PRIMARY:
            for p in reg.parents[f]:
                labels[p.key] = FA.draw_labels(space, p, kmax if p.key in subset else FA.SIZES["pergoal"], excl)[0]
        man = dict(experiment="RCX-E05", kind="test", attempt=k, n=n, prereg_sha256=prereg, code_digest_t3=cd["digest"],
                   pilot_manifest_sha256=sha_file(ctx.pdir / "PILOT_MANIFEST.json"),
                   pilot_report_sha256=sha_file(ctx.pdir / "PILOT_REPORT.json"), sizes=sz, cuts=sorted(cuts),
                   parents={f: [p.to_json() for p in reg.parents[f]] for f in FA.TEST_ORDER},
                   rejections={f: dict(reg.rejections[f]) for f in FA.TEST_ORDER},
                   notes={f: reg.notes.get(f, {}) for f in FA.TEST_ORDER}, pgv_subset=subset, pergoal_labels=labels,
                   exclusions=None if excl is None else dict(calls=dict(excl.calls)),
                   streams=dict(test=f"20261242:test:{k}", subset="20261242:pgv-subset",
                                starts=f"20261244:starts:test:{k}:<key>", labels="20261249:labels:<key>"),
                   provenance=provenance(ctx, "draw-test", cd))
        sha = write_create_only(out, man)
        log_attempt(ctx, "manifest", attempt=k, file="TEST_MANIFEST.json", sha256=sha,
                    note="commit this sha to DECISIONS before any arm is evaluated on TEST")
        print(f"TEST_MANIFEST.json sha256 {sha}", flush=True)
        return True
    plan = lambda: [] if out.exists() else [Unit("draw-test", 300.0, run)]  # noqa: E731
    return drive(ctx, "draw-test", plan, attempt=k, cd=cd, args=vars(args))


def rcx_test_sets_n7(root: Path) -> dict:
    """§4.4 item 4: RCX evaluation sets as DESIGN §4.5 lists them (E07 test, E07b TEST-B, E11a, E11b exec) and their
    n = 7 counts; E11b's test_id/ood_small splits are reported beside (descriptive)."""
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
    """Attempt-k state: Data over TEST_MANIFEST (+ the pilot library), keyed logs."""

    def __init__(self, ctx, k, manifest, digest):
        self.ctx, self.k, self.m, self.digest = ctx, k, manifest, digest
        self.dir = ctx.adir(k)
        self.d = Data(ctx, manifest, ctx.pdir)
        self.oc = KeyedLog(self.dir / "oracle_cells.jsonl")
        self.cells = KeyedLog(self.dir / "cells.jsonl")
        self.banks = KeyedLog(self.dir / "banks.jsonl")
        self.fits = KeyedLog(self.dir / "fits.jsonl")
        self.rep = read_json(ctx.pdir / "PILOT_REPORT.json")
        self.base = bool(self.rep["library"]["base_policies"])
        self.split = f"test:{k}"

    def parents(self):
        return [(f, p) for f in FA.TEST_ORDER for p in self.d.fam.get(f, [])]


def oracle_test(ctx, args) -> int:
    k = int(args.attempt)
    ts_m = test_manifest(ctx, k, args.manifest_sha)
    cd = require_digest(ctx, "t3", args.code_digest)
    if ts_m.get("code_digest_t3") != cd["digest"]:
        raise SystemExit("refusing: the manifest's code digest differs from the current T3 digest")
    if oracle_report(ctx, k) is not None:
        raise SystemExit("refusing: ORACLE_TEST_REPORT.json exists (create-only)")
    if first_learned(ctx, k):
        raise SystemExit("refusing: a learned-arm TEST computation already ran in this attempt; no VOID exists now")
    T = TestState(ctx, k, ts_m, cd["digest"])

    def unit_parent(f, p):
        def run(sl):
            d, out = T.d, {}
            tab, stop, mask = d.table(p), d.arm_stop(p), d.mask(p)
            q3 = 0
            for s0 in p.starts:
                ep = SO.run_episode(d.space, int(s0), SO.oracle_policy(tab), mask)
                v = float(tab.stop[tab.end[s0]]) - tab.c * float(tab.T[s0])
                q3 += int(not (ep.final == tab.end[s0] and ep.T == tab.T[s0] and v == tab.V[s0]))
            out["q3_bad"] = q3
            if p.dyn == "free":
                lib = d.library(T.base)
                G = SO.gpi_table(stop, p.c, tabs_of(lib))
                bad = 0
                for s0 in p.starts:
                    J = SO.episode_return(stop, p.c, SO.run_episode(d.space, int(s0), SO.gpi_policy(stop, p.c, G)))
                    bad += int(J < float((stop[lib.end[:, s0]] - p.c * lib.T[:, s0].astype(np.float64)).max()))
                out["q4_bad"] = bad
            nr = {}
            for arm in ("ORACLE", "STOP", "GREEDY", "RAND"):
                pols = [ref_policy(arm, d, p, int(s0), T.split, T.base)[0] for s0 in p.starts]
                nr[arm] = [SO.evaluate(d.space, tab, int(s0), pol, mask)["nreg"] for s0, pol in zip(p.starts, pols)]
            out["nreg"] = nr
            T.oc.add(kj("oracle", p.key), dict(family=f, parent=p.key, **out), T.digest)
            return True
        return run

    def unit_report(sl):
        checks = {}
        rows = {r["parent"]: r for r in T.oc.rows.values()}
        q3 = sum(r["q3_bad"] for r in rows.values())
        q4 = sum(r.get("q4_bad", 0) for r in rows.values())
        checks["Q3"] = dict(value=q3, status="PASS" if q3 == 0 else "VOID")
        checks["Q4"] = dict(value=q4, status="PASS" if q4 == 0 else "VOID",
                            note="free-dynamics families (H-DYN excluded: library policies are not policies of the "
                                 "height-limited MDP)")
        fam = defaultdict(list)
        for r in rows.values():
            fam[r["family"]].append(r)
        o0 = all(v == 0.0 for r in rows.values() for v in r["nreg"]["ORACLE"])
        s1 = all(v == 1.0 for r in rows.values() for v in r["nreg"]["STOP"])
        g1 = all(v == 1.0 for f in POSITIVE_TEST for r in fam.get(f, []) for v in r["nreg"]["GREEDY"])
        rand = mean(mean(mean(r["nreg"]["RAND"]) for r in fam[f]) for f in FA.PRIMARY)
        checks["Q7"] = dict(value=dict(oracle_all_zero=o0, stop_all_one=s1, greedy_identity=g1, rand_primary_macro=rand),
                            status="PASS" if (o0 and s1 and g1 and lt(rand, 1.0)) else "VOID")
        reg = FA.Registry.from_parents(T.d.n, [p for ps in T.d.fam.values() for p in ps] +
                                       [FA.Parent.from_json(x) for ps in read_json(ctx.pdir / "PILOT_MANIFEST.json")
                                        ["parents"].values() for x in ps])
        sp = {kk: [bool(v[0]), v[1]] for kk, v in FA.check_splits(reg).items()}
        checks["Q8:splits"] = dict(value=sp, status="PASS" if all(v[0] for v in sp.values()) else "VOID")
        try:
            excl = ctx.exclusions(T.d.space)
            exc = Counter()
            for f, p in T.parents():
                v = FA.verify_parent(T.d.space, T.d.phi, p, excl, labels=T.m["pergoal_labels"].get(p.key, ()))
                for kk in ("excluded_starts", "excluded_bank", "excluded_labels"):
                    exc[kk] += int(v.get(kk, 0))
                exc["invalid"] += int(not v["starts_ok"] or not v["bank_ok"])
            checks["Q8:exclusions"] = dict(value=dict(exc), status="PASS" if not sum(exc.values()) else "VOID")
        except Exception as e:  # exclusion basis changed (load_exclusions fails closed, §4.4)
            checks["Q8:exclusions"] = dict(value=f"exclusion basis changed: {e!r}", status="VOID")
        n7 = rcx_test_sets_n7(ctx.root)
        checks["Q8:n7"] = dict(value=n7, status="PASS" if n7["ok"] else "VOID")
        sub = FA.pgv_subset(FA.Registry.from_parents(T.d.n, [p for f in FA.PRIMARY for p in T.d.fam[f]]))
        checks["Q8:subset"] = dict(value=dict(as_drawn=sub == T.m["pgv_subset"]),
                                   status="PASS" if sub == T.m["pgv_subset"] else "VOID")
        checks["manifest"] = dict(value=args.manifest_sha, status="PASS")  # file = committed sha (checked at entry)
        checks["code_digest"] = dict(value=cd["digest"], status="PASS" if cd["digest"] == args.code_digest else "VOID")
        status = "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "VOID"
        rep = dict(attempt=k, status=status, checks=checks, manifest_sha256=args.manifest_sha, code_digest=cd["digest"],
                   provenance=provenance(ctx, "oracle-test", cd))
        sha = write_create_only(T.dir / "ORACLE_TEST_REPORT.json", rep)
        log_attempt(ctx, "oracle-test", attempt=k, status=status, sha256=sha,
                    void=[c for c, v in checks.items() if v["status"] != "PASS"])
        if status == "VOID":
            print(f"attempt {k}: VOID " + ("(third VOID: INCONCLUSIVE + design review, §7.2)" if k >= 3 else
                                          f"-> fix, then draw-test --attempt {k + 1}"), flush=True)
        return True

    def plan():
        if oracle_report(ctx, k) is not None:
            return []
        U = [Unit(f"oracle:{p.key}", 16 * 4 * 0.2 + 1.0, unit_parent(f, p)) for f, p in T.parents()
             if kj("oracle", p.key) not in T.oc]
        return U or [Unit("oracle-test:report", 120.0, unit_report)]
    return drive(ctx, "oracle-test", plan, attempt=k, cd=cd, args=vars(args))


def confirm_models_dir(ctx) -> Path:
    return ctx.runs / "confirm" / "models"


def train_replicates(ctx, args) -> int:
    require_prereg(ctx, args.expect_prereg_sha)
    cd = require_digest(ctx, "t3", args.expect_code_digest)
    rep = pilot_report_or_refuse(ctx)
    pm = read_json(ctx.pdir / "PILOT_MANIFEST.json")
    log = KeyedLog(confirm_models_dir(ctx) / "fits.jsonl")
    P = Pilot(ctx, None)
    P.digest = None
    L, base = rep["train_labels_per_goal"], rep["library"]["base_policies"]
    sel = rep["selected"]
    jobs = [(m, s, sel[m][2], sel[m][3]) for m in ("SF", "UVFA", "UVFA+aux") for s in SEEDS["replicates"]]
    jobs.append(("SF-shuffled", SEEDS["replicates"][0], sel["SF"][2], sel["SF"][3]))

    def unit(model, seed, cfg, epoch):
        def run(sl):
            if not P.data_path(L, base).exists():
                P.unit_train_data(L, base)(sl)
            data = dict(np.load(P.data_path(L, base)))
            spec = dict(ZS_BASE if cfg == "base" else ZS_EXT)
            spec.update(epochs=int(epoch), keep=(int(epoch),))
            out = confirm_models_dir(ctx) / f"{model}_s{seed}"
            res = train_zero_shot(model.replace("SF-shuffled", "SF-shuffled"), P.data(), data, seed=seed, cfg=spec,
                                  out_dir=out, safe_point=sl.safe_point)
            if res is None:
                return False
            path = str(res[int(epoch)] if int(epoch) in res else res[str(epoch)])
            log.add(kj("replicate", model, seed), dict(model=model, seed=seed, cfg=cfg, epoch=int(epoch), path=path,
                                                       sha=_sha_path(path), labels=L, base=base), cd["digest"])
            return True
        return run
    del pm
    plan = lambda: [Unit(f"train:{m}:s{s}", 600.0, unit(m, s, c, e)) for m, s, c, e in jobs  # noqa: E731
                    if kj("replicate", m, s) not in log]
    return drive(ctx, "train", plan, attempt="replicates", cd=cd, args=vars(args))


def learned_test_gate(ctx, args, phase) -> tuple:
    """pgv/ft/eval/extras entry: committed manifest sha, T3 digest, ORACLE_TEST_REPORT PASS; logs the first
    learned-arm TEST computation."""
    k = int(args.attempt)
    m = test_manifest(ctx, k, args.manifest_sha)
    cd = require_digest(ctx, "t3", args.code_digest)
    if m.get("code_digest_t3") != cd["digest"]:
        raise SystemExit("refusing: the manifest's code digest differs from the current T3 digest (defect after the "
                         "first learned-arm TEST computation: INCONCLUSIVE, §10.4)")
    require_oracle_pass(ctx, k)
    T = TestState(ctx, k, m, cd["digest"])
    return k, cd, T


def mark_first_learned(ctx, k, phase):
    if not first_learned(ctx, k):
        log_attempt(ctx, "first-learned-test", attempt=k, phase=phase)


def pgv_ft(ctx, args, kind) -> int:
    k, cd, T = learned_test_gate(ctx, args, kind.lower())
    rep, prim = T.rep, [(f, p) for f in FA.PRIMARY for p in T.d.fam[f]]
    sub = set(T.m["pgv_subset"])
    if kind == "FT" and rep["p2_dropped"] and False:
        pass
    reps = read_json_lines_models(ctx)
    units_spec = []
    if kind == "PGV":
        name = rep["G_P3"]["grid"]["PGV"]["cfg"]
        cfg400 = dict(PGV_COMMON if name in PGV_GRID else {}, **{**PGV_GRID, **PGV_EXT}[name])
        units_spec.append(("PGV", name, cfg400, [p for _, p in prim], 400))
        curve = R_EXT if rep["kcurve_recipe"] == "R_ext" else KCURVE
        for kk, spec in curve.items():
            if kk == 400 and rep["kcurve_recipe"] != "R_ext":
                continue
            units_spec.append(("PGV", f"{'rext' if rep['kcurve_recipe'] == 'R_ext' else 'k'}{kk}", spec,
                               [p for _, p in prim if p.key in sub], kk))
    else:
        name = rep["G_P3"]["grid"]["FT"]["cfg"]
        cfg = dict(FT_COMMON if name in FT_GRID else {}, **{**FT_GRID, **FT_EXT}[name])
        n_rep = 1 if ("FT-UVFA_400-1rep" in rep["G_P5"]["cuts"] and rep["pg400"] != "FT") else 3
        for seed in SEEDS["replicates"][:n_rep]:
            units_spec.append(("FT", f"{name}|s{seed}", dict(cfg, init_seed=seed), [p for _, p in prim], 400))

    def unit(kind_, name, cfg, ps, kk):
        def run(sl):
            mark_first_learned(ctx, k, kind.lower())
            jobs = []
            for p in ps:
                idx = np.asarray(T.m["pergoal_labels"][p.key][:kk], dtype=np.int64)
                job = dict(key=p.key, states=idx, v=T.d.table(p).V[idx], stab=FA.stabilizer(np.asarray(p.w), T.d.n),
                           w=p.weights(), c=p.c, seed=FA.derive(SEEDS["pergoal"], f"pgv-init:{p.key}:{kk}"))
                if kind_ == "FT":
                    seed = cfg["init_seed"]
                    job.update(init=reps[("UVFA", seed)], seed=FA.derive(seed, f"ft:{p.key}"))
                jobs.append(job)
            res = fit_pergoal(kind_, T.d, jobs, {x: v for x, v in cfg.items() if x != "init_seed"},
                              out_dir=T.dir / "models" / f"{kind_}_{name.replace('|', '_')}_k{kk}",
                              safe_point=sl.safe_point)
            if res is None:
                return False
            T.fits.add(kj("fit", kind_, name, kk), dict(kind=kind_, cfg=name, k=kk,
                                                        models={str(a): str(b) for a, b in res.items()}), T.digest)
            return True
        return run
    plan = lambda: [Unit(f"fit:{a}:{b}:k{e}", 0.02 * c["steps"] * len(d), unit(a, b, c, d, e))  # noqa: E731
                    for a, b, c, d, e in units_spec if kj("fit", a, b, e) not in T.fits]
    return drive(ctx, kind.lower(), plan, attempt=k, cd=cd, args=vars(args))


def read_json_lines_models(ctx) -> dict:
    """{(model, seed): path} of the confirmation replicates (never selection models)."""
    log = KeyedLog(confirm_models_dir(ctx) / "fits.jsonl")
    out = {}
    for r in log.rows.values():
        if int(r["seed"]) not in SEEDS["replicates"]:
            raise SystemExit("refusing: a non-replicate model is registered for TEST (selection models never touch "
                             "TEST, §2.3)")
        out[(r["model"], int(r["seed"]))] = r["path"]
    return out


TEST_REF_ARMS = ("ORACLE", "STOP", "RAND", "GREEDY", "EXACT-GPI(L)", "BEST-SINGLE(L)", "ENUM-MIS±")


def evaluate_test(ctx, args, extras=False) -> int:
    k, cd, T = learned_test_gate(ctx, args, "extras" if extras else "eval")
    reps = read_json_lines_models(ctx)
    cuts = set(T.rep["G_P5"]["cuts"])
    fams = [f for f in FA.TEST_ORDER if not (f == "H-KN" and "H-KN" in cuts) and not (f == "H-DYN" and "H-DYN" in cuts)]
    specs = []  # (family, parent, arm, tag, policy factory)
    for f in fams:
        for p in T.d.fam[f]:
            for arm in TEST_REF_ARMS:
                specs.append((f, p, arm, Pilot.ref_tag(arm, T.base), "ref"))
            for arm, fs in (("ENUM-H*±", ("H-ID", "H-K", "H-NEG")), ("GNN-ENUM±", ("H-ID", "H-K", "H-NEG"))):
                if f in fs and not (f == "H-NEG" and arm == "GNN-ENUM±" and "GNN-ENUM±@H-NEG" in cuts) and not (
                        f == "H-NEG" and arm == "ENUM-H*±" and "ENUM-H*±@H-NEG-24" in cuts
                        and [q.key for q in T.d.fam[f]].index(p.key) >= 24):
                    specs.append((f, p, arm, "-", "ref"))
            for arm in ("SF-GPI", "SF-noGPI", "SF-L", "SF-GPI-5", "UVFA", "UVFA+aux"):
                for seed in SEEDS["replicates"]:
                    specs.append((f, p, arm, f"{ARM_MODEL[arm]}|s{seed}", ("zs", ARM_MODEL[arm], seed)))
            specs.append((f, p, "SF-shuffled", f"SF-shuffled|s{SEEDS['replicates'][0]}",
                          ("zs", "SF-shuffled", SEEDS["replicates"][0])))
            if f in FA.PRIMARY:
                for r in T.fits.rows.values():
                    if p.key in r["models"] and (r["k"] == 400 or extras):
                        specs.append((f, p, r["kind"], f"{r['cfg']}|k{r['k']}", ("pg", r["models"][p.key])))
    if extras:
        specs = [s for s in specs if s[2] in ("PGV",) and "|k400" not in s[3]]

    def unit(f, p, arm, tag, how):
        def run(sl):
            mark_first_learned(ctx, k, "extras" if extras else "eval")
            d = T.d
            if how == "ref":
                fn = lambda s0: ref_policy(arm, d, p, s0, T.split, T.base)  # noqa: E731
            elif how[0] == "zs":
                pol = learned_policy(arm, load_model(reps[(how[1], how[2])]), d, p, T.base)
                fn = lambda s0: (pol, None)  # noqa: E731
            else:
                model = load_model(how[1])
                stop, w, c = d.arm_stop(p), p.weights(), p.c
                pol = SO.scored_policy(stop, lambda s, a, dsts: value_hat(model, d, dsts, w[None],
                                                                          np.array([c]))[:, 0] - c)
                fn = lambda s0: (pol, None)  # noqa: E731
            return eval_cells(T.cells, d, sl, split=T.split, family=f, p=p, arm=arm, tag=tag, policy_fn=fn,
                              digest=T.digest)
        return run

    def bank(f, p, arm, tag, how):
        def run(sl):
            d = T.d
            if how == "ref":
                pol, _ = ref_policy(arm, d, p, int(p.bank[0]), T.split, T.base) if arm not in ("BEST-SINGLE(L)",
                                                                                                 "RAND") else (None, 0)
                if pol is None:
                    T.banks.add(kj("bank", T.split, arm, tag, p.key), dict(skipped="start-dependent arm"), T.digest)
                    return True
            elif how[0] == "zs":
                pol = learned_policy(arm, load_model(reps[(how[1], how[2])]), d, p, T.base)
            else:
                model = load_model(how[1])
                stop, w, c = d.arm_stop(p), p.weights(), p.c
                pol = SO.scored_policy(stop, lambda s, a, dsts: value_hat(model, d, dsts, w[None],
                                                                          np.array([c]))[:, 0] - c)
            return bank_unit(T.banks, d, sl, split=T.split, p=p, arm=arm, tag=tag, choose=pol, vhat=None,
                             digest=T.digest)
        return run

    def plan():
        U = [Unit(f"eval:{f}:{arm}:{tag}:{p.key}", 16.0, unit(f, p, arm, tag, how)) for f, p, arm, tag, how in specs
             if not cells_done(T.cells, T.split, arm, tag, p)]
        U += [Unit(f"bank:{f}:{arm}:{tag}:{p.key}", 5.0, bank(f, p, arm, tag, how)) for f, p, arm, tag, how in specs
              if kj("bank", T.split, arm, tag, p.key) not in T.banks and not arm.startswith(("ENUM-H", "GNN"))]
        return U
    return drive(ctx, "extras" if extras else "eval", plan, attempt=k, cd=cd, args=vars(args))


# ============================================================================= CLI

def pilot_entry(ctx, args, phase) -> int:
    cd = require_digest(ctx, "t2", args.expect_code_digest)
    if ctx.n != 7:  # the n = 6 fallback only after G-P4(a) failed at 400 and 800 labels at n = 7
        P7 = Pilot(Ctx(runs=ctx.runs, root=ctx.root, n=7, env=ctx.env, decisions=ctx.decisions,
                       exclusions=ctx.exclusions), cd)
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
    rep = pilot_report(P)
    rep["provenance"] = provenance(ctx, "pilot-report", cd)
    sha = write_create_only(ctx.pdir / "PILOT_REPORT.json", rep)
    log_attempt(ctx, "pilot-report", attempt="pilot", sha256=sha)
    print(json.dumps(rep["fill"], indent=1, ensure_ascii=False))
    print(f"PILOT_REPORT.json sha256 {sha}")
    return 0


def main(argv=None) -> int:
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
    args = ap.parse_args(argv)
    if args.cmd == "code-digest":
        out = {s: code_digest(s) for s in ("t2", "t3")}
        print(json.dumps(dict(out[args.stage], tests=test_shas(), reference_shas=LA.reference_shas(),
                              other_stage={k: v["digest"] for k, v in out.items()}), indent=1))
        return 0
    n = getattr(args, "n", 7)
    ctx = Ctx(runs=Path(args.runs), n=n, pilot_dir="pilot" if n == 7 else f"pilot-n{n}")
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
    return evaluate_test(ctx, args, extras=args.cmd == "extras")


if __name__ == "__main__":
    sys.exit(main())
