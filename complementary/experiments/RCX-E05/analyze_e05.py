"""RCX-E05 confirmatory analysis (PREREG_E05_DRAFT.md §6-7, binding; DESIGN.md rev 3 §6.4-6.5). Run once per attempt,
after oracle-test PASS and every pgv / ft / eval / extras phase of that attempt:

  python -B experiments/RCX-E05/analyze_e05.py --attempt K --manifest-sha <T4 sha256> --code-digest <T3 digest>
         [--missing GROUP[,GROUP...]] [--defect TEXT]...

Outputs (create-only; the analysis runs once per attempt): <confirm>/attempt-K/ANALYSIS_E05.started (marker, written after
the custody checks and BEFORE any statistic), ANALYSIS_E05.json and ANALYSIS_E05.md. The run refuses if any of the three
exists (a marker without outputs is a crashed analysis: a defect, never a re-run). Pure Python (rcx.stats only; no numpy,
no torch, no oracle): every number comes from the eval files.

Custody (§2.2, §6.4, L5, B4). Every refusal is a SystemExit raised before anything is written or any statistic computed;
a refusal is not an analysis run:
  1. none of the three outputs exists;
  2. <attempt>/ORACLE_TEST_REPORT.json exists, says status "PASS", and names manifest_sha256 == --manifest-sha and
     code_digest == --code-digest (VOID is declared only by oracle-test; a VOID attempt is never analyzed);
  3. --manifest-sha is 64 hex, equals sha256(<attempt>/TEST_MANIFEST.json), and occurs in the COMMITTED DECISIONS.md
     (`git -C <repo-root> show HEAD:./DECISIONS.md`), not merely in the working copy; the manifest's "attempt" (if
     present) equals --attempt;
  4. the pilot determinations (PILOT_REPORT.json "determinations", or run_e05's pilot-report fields) are complete and
     consistent;
  5. exactly one eval source exists (evals/*.json or run_e05's cells.jsonl, below). Every eval file names
     manifest_sha256 == --manifest-sha and code_digest == --code-digest (cells.jsonl / banks.jsonl: every record's
     digest == --code-digest, its split == "test:<attempt>", and the manifest's code_digest_t3 == --code-digest); arm,
     tag and replicate are well formed, each parent is a manifest TEST parent, each nreg value is finite in [0, 1] (one
     per manifest start), and no (arm, replicate, parent[, start]) cell repeats;
  6. --missing names exactly the protected-core groups (§10.3) that have an absent cell (core_groups below); a declared
     group gives INCONCLUSIVE (missing). --defect records a defect found after the first learned-arm TEST computation
     (§10.4): INCONCLUSIVE (defect).

Input contracts (fields not listed are ignored). Both the exported form below and run_e05.py's native layout (as of
this writing) are read:
  TEST_MANIFEST.json      {"parents": [families_e05.Parent.to_json(): "family", "stratum" ("S"/"U" for H-ID), "key",
                          "starts", "w", "K", "n_neg", "rtype", ...] or run_e05's {family: [Parent.to_json()]},
                          "subset" or run_e05's "pgv_subset": [the 48-parent k-curve subset keys], optional "attempt",
                          optional "code_digest_t3"} (families outside the TEST families are ignored and counted)
  ORACLE_TEST_REPORT.json {"status": "PASS" | "VOID", "manifest_sha256": ..., "code_digest": ..., "checks": {...}}
  PILOT_REPORT.json       {"determinations": {"p1_comparator_invalid": bool, "p2_dropped": bool,
                          "pg400": "PGV_400" | "FT-UVFA_400" | null (null iff P2 dropped), "q9_failed_no_bug": bool,
                          "kstar_descriptive": bool, "r_ext": bool, optional "cut_items": [§10.3 item numbers],
                          optional "kcurve_arms": {"400": arm, "1600": arm, "6400": arm} (default PGV_<k>), "n": 7|6}}
                          or, without "determinations", run_e05's pilot-report fields (native_determinations):
                          p1_comparator_invalid, p2_dropped, pg400 ("PGV" | "FT" | null), kstar ("decision-driving" |
                          "descriptive only"), kcurve_recipe ("original" | "R_ext"; R_ext -> k-curve arms PGV-RX_<k>),
                          G_P1.Q9.ok, G_P5.cuts (run_e05.CUT_ORDER names = §10.3 items 1-7), G_P3.grid.{PGV,FT}.cfg
  eval source, one of:
  evals/*.json            {"manifest_sha256", "code_digest", "arm", "replicate": 20261246-48 | null,
                          "cells": {parent key: {"nreg": [one per start, manifest order], optional numeric lists
                          such as "loop", "regret1", "top1", "value_abs_err", "decisions", "all_tied"}}} (a bare list
                          is read as nreg)
  cells.jsonl (run_e05)   one keyed record per (arm, tag, parent, start): "split", "arm", "tag", "parent", "s0", "nreg",
                          "loop", "capped", "T", optional "diag" {"decisions", "all_tied"}, "digest"; plus optional
                          banks.jsonl records ("arm", "tag", "parent", "rows": [{"regret1", "top1"}], "digest").
                          native_arm maps (arm, tag) to the names below: "<model>|s<seed>" -> replicate <seed>;
                          PGV "<selected cfg>|k400" -> PGV_400, "k<k>|k<k>" -> PGV_<k>, "rext<k>|k<k>" -> PGV-RX_<k>;
                          FT "<selected cfg>|s<seed>|k400" -> FT-UVFA_400 @ <seed>. A (arm, replicate, parent) with
                          fewer records than starts is an absent cell (missing), never a partial mean.
  Arm names (exact strings): replicated (replicate 20261246-48): SF-GPI, UVFA, UVFA+aux, SF-noGPI, SF-L, SF-GPI-5,
  SF-shuffled, FT-UVFA_400, SF-FT_400; deterministic (replicate null): PGV_400, PGV_1600, PGV_6400 (or the pilot's
  kcurve_arms), ENUM-MIS±, ENUM-H*±, GNN-ENUM±, ENUM-MIS (rev 1), ENUM-MIS± (rev 2), GREEDY, STOP, RAND, ORACLE,
  EXACT-GPI(L), BEST-SINGLE(L). Other names are accepted and reported descriptively. Measured cost at confirmation
  scale (synthetic: 480 parents, 16 starts, every arm above, R = 10,000): analyze() about 3 s single-thread, peak
  working set about 110 MB.

Statistics (§6.1, §6.4). A parent's value for one arm is computed in rcx.stats' order: per replicate sum/len over its
starts, then the sum over replicates in seed order / their number (deterministic arms: replicate null). A contrast
D = nreg(X) - nreg(Y) is the per-parent paired difference (rcx.stats' b - a with b = X). Strata: H-ID-S, H-ID-U, H-K,
H-NEG and each secondary family. One index stream per stratum h, random.Random(derive(20261250, "boot:<h>")), with the
stratum's parents in rcx.stats._order (sorted keys) and R resamples of n_h randrange(n_h) draws each (the order of
rcx.stats._boot_means), drawn once and shared by every contrast (common random numbers). A stratum's resampled mean is
the Python sum of the drawn values in draw order / n_h; H-ID = (n_S m_S + n_U m_U) / (n_S + n_U); a single-stratum
family is its stratum; macro = (H-ID + H-K + H-NEG) / 3; the point estimates use the same formulas. Hence on one stratum
the interval and p equal rcx.stats.seed_parent_bootstrap_test(..., resamples=10000, seed=derive(20261250, "boot:<h>"),
p_floor=1e-4, ties='both') exactly. CI: rcx.stats._interval (95% percentile); "upper bound" = its 97.5th percentile.
Two-sided p: rcx.stats.bootstrap_p_two_sided(boot, floor=1/R, ties='both'). Thresholds use the 1e-9 tolerance.

Tests (§6.2-6.3). P1: D1 = nreg(UVFA) - nreg(SF-GPI), intersection-union of P1-M (macro) and P1-ID (H-ID),
p_P1 = max(p_M, p_ID); passes iff Holm-adjusted p_P1 < 0.05, both points >= 0.03, both lower bounds > 0, and for each
matched replicate pair k the macro and H-ID D1 > 0. P2: D2 = nreg(SF-GPI) - nreg(PG400), one-sided
p = max(1/R, share of replicates with D2* >= 0.05); passes iff Holm-adjusted p < 0.05. Holm (rcx.stats.holm) over the
family fixed at the freeze: {P1 unless comparator-invalid at the pilot} + {P2 unless dropped at the pilot}. G1: point
D_G = nreg(SF-GPI) - nreg(ENUM-MIS±) < 0.05; STOP-D condition: D_G >= 0.05 and its lower bound > 0.

Decision (§7.1 determinations, then the §7.2 order; the first that holds decides): [VOID: refusal, never analyzed]
INCONCLUSIVE (defect or missing) > STOP-D > ADVANCE (P1 counts, P2 counts, G1, SF floor) > STOP-M (EXACT-GPI(L) primary
macro >= 0.5 and neither test counts) > STOP-F (P2 does not count and the macro or H-ID D1 upper bound < 0.05) > REVISE
(P1 or P2 counts; every failed ADVANCE condition named) > INCONCLUSIVE. P1 counts iff not comparator-invalid at the pilot,
UVFA's TEST H-ID nreg <= 0.50 and P1 passes; P2 counts iff not dropped, PG400's TEST primary macro <= 0.50 and P2 passes.
Validity never changes the Holm family. Qualifiers change wording only (attribution, GPI, new directions, naming, guard
quality, k*).

Readings this code takes where the prereg leaves a detail open (none changes a threshold):
  - A test outside the Holm family (pilot-invalid P1, dropped P2) is reported with raw p only; its "passes" is null.
  - The P2 replicate count "D2* >= 0.05" uses the 1e-9 tolerance (D2* >= 0.05 - 1e-9; conservative).
  - New-directions qualifier: "H-ID-U D2 upper bound" = the 97.5th percentile (§6.4's definition of "upper bound").
  - "P1-M passes" (REVISE note) = the macro conjunct alone meets conditions 1-4 (its Holm-adjusted p with p_M in place
    of p_P1); Holm is monotone, so P1 passes iff both conjuncts pass.
  - Contrasts restricted to a subset of a stratum (the 48-parent k-curve subset, or parents lost to a missing cell) draw
    their index matrix from the same "boot:<h>" stream with the smaller n.
  - k*: macro over the subset strata; first crossing of nreg(PGV_k) <= nreg(SF-GPI) on log10 k with linear
    interpolation; "<= 400" if PGV_400 already matches, "> 6400" if PGV_6400 does not; censored bootstrap percentiles
    are reported as the censoring bound. FT-UVFA_400 (one budget) gives "<= 400" or "> 400".
  - Conflict stratum: the parent's positive facts are not jointly satisfiable (two supports for one block, two blocks on
    one, on(x, b) with clear(b) or ontable(x), or an on-cycle). K and conflict strata are point estimates only.
  - Binary view: a start is optimal iff nreg <= 1e-9; a parent is "optimal on >= half of the starts" over all its
    (replicate, start) values (rcx.stats.paired_binary_table, threshold 0.5).
  - Protected core for missing-cell detection (§10.3), with "retained" = every TEST family not cut by items 3 or 7:
    SF-GPI, UVFA, UVFA+aux, SF-noGPI, SF-L, SF-GPI-5 x 3 replicates and SF-shuffled (20261246) on every retained family
    (§10.3 gives them no family scope and no cut item); PGV_400 on the primary parents; FT-UVFA_400 x 3 there if it is
    PG400; the k-curve arms on the subset; ENUM-MIS±, GREEDY, STOP, RAND, ORACLE, EXACT-GPI(L), BEST-SINGLE(L) on every
    retained family; ENUM-H*± and GNN-ENUM± on H-ID and H-K. "The one-step banks": a bank result (regret1) for every
    protected cell, except the arms run_e05 does not bank (RAND and BEST-SINGLE(L): start-dependent; ENUM-H*±,
    GNN-ENUM±); their groups are "bank:<arm>[@replicate]".
  - The live code digest is not recomputed here (run_e05.py code-digest owns its definition; oracle-test checked it);
    this file checks the digest named by ORACLE_TEST_REPORT.json and by every eval file against --code-digest.
  - From run_e05's pilot-report fields, "Q9 failed without a bug" = G_P1.Q9.ok is false in the final report (a bug
    found by the audit is fixed and the pilot phase re-run, §9.1).
  - ENUM± all-tied share (§6.5) = all-tied decisions / decisions with >= 2 legal moves, pooled per family.
  - Labels (§6.5): D1 on H-K and H-NEG "includes extrapolation outside UVFA's training weight support"; H-CONJ and
    H-DYN "assumption boundary" (their §4.2 role).
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx import stats as ST  # noqa: E402

EXPERIMENT = "RCX-E05"
BOOT_SEED = 20261250  # App. A: analysis bootstraps
PREREG_RESAMPLES = 10000  # §6.4
RESAMPLES = PREREG_RESAMPLES  # what main() uses (tests may lower it; the report then says it is not the prereg run)
ALPHA = 0.05
TIES = "both"
EPS = 1e-9  # §6.4 tolerance
BAR = 0.50  # competence bar and SF floor (§5.3, §7.1)
P1_MIN = 0.03  # P1 minimum point estimate
PLANNED = 0.05  # planned effect: STOP-F futility bound
DELTA_NI = 0.05  # P2 NI margin
DOMINANCE = 0.05  # G1 / STOP-D margin
CEILING = 0.5  # STOP-M library ceiling
ATTR_MARGIN, GPI_MARGIN = 0.02, 0.01  # qualifiers
NEWDIR_POINT, NEWDIR_D2_HI = 0.03, 0.05
REPLICATES = (20261246, 20261247, 20261248)
PRIMARY = ("H-ID", "H-K", "H-NEG")
SECONDARY = ("H-COST", "H-SPAN", "H-KN", "H-CONJ", "H-DYN")
TEST_ORDER = PRIMARY + SECONDARY
STRATA = {"H-ID": ("H-ID-S", "H-ID-U"), **{f: (f,) for f in TEST_ORDER[1:]}}
CUT_FAMILIES = {3: "H-KN", 7: "H-DYN"}  # §10.3 cut items that remove a family
CUT_NAMES = ("budget-curve", "SF-FT_400", "H-KN", "GNN-ENUM±@H-NEG", "FT-UVFA_400-1rep", "ENUM-H*±@H-NEG-24",
             "H-DYN")  # = run_e05.CUT_ORDER: §10.3 items 1-7
KS = (400, 1600, 6400)
RX = "PGV-RX_{}"  # the k-curve arms under the R_ext escalation (§5.3 G-P3b)
TIE_FIELDS = ("decisions", "all_tied")  # ENUM± all-tied-decision counts (§6.5): shares, not means

SF, UVFA, AUX = "SF-GPI", "UVFA", "UVFA+aux"
NOGPI, SFL, SF5, SHUF = "SF-noGPI", "SF-L", "SF-GPI-5", "SF-shuffled"
PGV, FT, SFFT = "PGV_400", "FT-UVFA_400", "SF-FT_400"
ENUM, ENUM_H, GNN = "ENUM-MIS±", "ENUM-H*±", "GNN-ENUM±"
ENUM_R1, ENUM_R2 = "ENUM-MIS (rev 1)", "ENUM-MIS± (rev 2)"
GREEDY, STOP, RAND, ORACLE, XGPI, BEST = "GREEDY", "STOP", "RAND", "ORACLE", "EXACT-GPI(L)", "BEST-SINGLE(L)"
ZERO_SHOT = (SF, UVFA, AUX, NOGPI, SFL, SF5)
MODEL_OF = {SF: "SF", NOGPI: "SF", SFL: "SF", SF5: "SF", UVFA: "UVFA", AUX: "UVFA+aux", SHUF: "SF-shuffled"}  # run_e05
REPLICATED = frozenset(ZERO_SHOT + (SHUF, FT, SFFT))
DETERMINISTIC = frozenset({PGV, "PGV_1600", "PGV_6400", ENUM, ENUM_H, GNN, ENUM_R1, ENUM_R2, GREEDY, STOP, RAND,
                           ORACLE, XGPI, BEST})
FLOORS = (ENUM, GREEDY, STOP, RAND, ORACLE, XGPI, BEST)
BANK_FIELD = "regret1"  # a cell's one-step bank result (with "top1")
BANK_EXEMPT = frozenset({RAND, BEST, ENUM_H, GNN})  # protected arms without a one-step bank record (run_e05)
ARM_ORDER = (SF, UVFA, AUX, NOGPI, SFL, SF5, SHUF, PGV, FT, SFFT, "PGV_1600", "PGV_6400", ENUM, ENUM_H, GNN, ENUM_R1,
             ENUM_R2, XGPI, BEST, GREEDY, RAND, STOP, ORACLE)
# Descriptive contrasts (§6.5): (label, X, Y) for D = nreg(X) - nreg(Y). D1, D2 and D_G are added by analyze().
DESCRIPTIVE = (("attribution: UVFA+aux - SF-GPI", AUX, SF), ("GPI: SF-noGPI - SF-GPI", NOGPI, SF),
               ("SF-L - SF-GPI", SFL, SF), ("SF-GPI-5 - SF-GPI", SF5, SF),
               ("learnability: SF-L - EXACT-GPI(L)", SFL, XGPI), ("learnability: SF-noGPI - ORACLE", NOGPI, ORACLE),
               ("SF-shuffled - SF-GPI", SHUF, SF), ("SF-FT_400 - SF-GPI", SFFT, SF),
               ("SF-GPI - ENUM-H*±", SF, ENUM_H), ("SF-GPI - GNN-ENUM±", SF, GNN),
               ("BEST-SINGLE(L) - EXACT-GPI(L)", BEST, XGPI), ("SF-GPI - GREEDY", SF, GREEDY),
               ("F5 fix: ENUM-MIS (rev 1) - ENUM-MIS±", ENUM_R1, ENUM), ("B1 fix: ENUM-MIS± (rev 2) - ENUM-MIS±",
                                                                         ENUM_R2, ENUM))
NAMING = "goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets, no TD)"
FAMILY_LABELS = {"H-CONJ": "assumption boundary", "H-DYN": "assumption boundary"}  # §4.2 roles, §6.5
D1_LABELS = {f: "includes extrapolation outside UVFA's training weight support" for f in ("H-K", "H-NEG")}  # §6.5
GUARD_NOTE = "against a guard that failed its DEV sanity check on H-NEG"

INC_DEFECT, INC_MISSING = "INCONCLUSIVE (defect)", "INCONCLUSIVE (missing)"
STOP_D, ADVANCE, STOP_M, STOP_F, REVISE, INCONCLUSIVE = "STOP-D", "ADVANCE", "STOP-M", "STOP-F", "REVISE", "INCONCLUSIVE"
DECISIONS = (INC_DEFECT, INC_MISSING, STOP_D, ADVANCE, STOP_M, STOP_F, REVISE, INCONCLUSIVE)
OUTPUTS = ("ANALYSIS_E05.started", "ANALYSIS_E05.json", "ANALYSIS_E05.md")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DET_KEYS = ("p1_comparator_invalid", "p2_dropped", "pg400", "q9_failed_no_bug", "kstar_descriptive", "r_ext")
_ALL = object()


def _ge(x, t):
    return x >= t - EPS


def _gt(x, t):
    return x > t + EPS


def _lt(x, t):
    return x < t - EPS


def _le(x, t):
    return x <= t + EPS


def derive(seed: int, tag: str) -> int:
    """App. A (= families_e05.derive)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ============================================================================ design (manifest) and parent values

def _facts(n):
    """= rcx.bw.soft.facts(n): on(i, j) for i != j, ontable(i), clear(i)."""
    return [("on", i, j) for i in range(n) for j in range(n) if i != j] + [("ontable", i) for i in range(n)] + \
        [("clear", i) for i in range(n)]


def has_conflict(w):
    """True iff the positive facts of w over facts(n) are not jointly satisfiable (None if w is not a fact vector)."""
    if not w:
        return None
    n = int(round((math.sqrt(1 + 4 * len(w)) - 1) / 2))
    if n * (n + 1) != len(w):
        return None
    pos = [f for f, x in zip(_facts(n), w) if x > 0]
    on = [(f[1], f[2]) for f in pos if f[0] == "on"]
    table, clear = {f[1] for f in pos if f[0] == "ontable"}, {f[1] for f in pos if f[0] == "clear"}
    ups, downs = Counter(a for a, _ in on), Counter(b for _, b in on)
    if max(ups.values(), default=0) > 1 or max(downs.values(), default=0) > 1:
        return True
    if any(a in table or b in clear for a, b in on):
        return True
    sup = dict(on)
    for a in sup:
        x, seen = a, set()
        while x in sup:
            if x in seen:
                return True
            seen.add(x)
            x = sup[x]
    return False


def manifest_parents(manifest):
    """The parent records of TEST_MANIFEST.json: a list, or run_e05's {family: [Parent.to_json()]}."""
    ps = manifest.get("parents")
    if isinstance(ps, dict):
        out = []
        for fam, lst in ps.items():
            if not isinstance(lst, list) or any(not isinstance(p, dict) or p.get("family") != fam for p in lst):
                raise ValueError(f"manifest parents[{fam!r}] is not a list of {fam} parents")
            out += lst
        return out
    if isinstance(ps, list):
        return ps
    raise ValueError("the manifest has no 'parents' list or {family: [parents]} object")


def manifest_subset(manifest):
    a, b = manifest.get("subset"), manifest.get("pgv_subset")
    if a is not None and b is not None and list(a) != list(b):
        raise ValueError("the manifest's 'subset' and 'pgv_subset' differ")
    return list(a if a is not None else b or [])


def make_design(manifest):
    """Strata (keys in rcx.stats._order), parent info and the k-curve subset from TEST_MANIFEST.json."""
    parents, strata, ignored = {}, defaultdict(list), Counter()
    for p in manifest_parents(manifest):
        fam = p["family"]
        if fam not in TEST_ORDER:
            ignored[fam] += 1
            continue
        key = p["key"]
        if not isinstance(key, str) or not key or key in parents:
            raise ValueError(f"manifest parent key {key!r} is empty or repeated")
        if fam == "H-ID":
            if p.get("stratum") not in ("S", "U"):
                raise ValueError(f"H-ID parent {key} has stratum {p.get('stratum')!r} (need 'S' or 'U')")
            h = f"H-ID-{p['stratum']}"
        else:
            h = fam
        starts = p.get("starts")
        if starts is not None and (not isinstance(starts, list) or len(set(starts)) != len(starts)):
            raise ValueError(f"parent {key} has malformed or repeated starts")
        parents[key] = dict(family=fam, stratum=h, n_starts=len(starts) if starts is not None else None,
                            starts=None if starts is None else list(starts), K=p.get("K"), n_neg=p.get("n_neg"),
                            conflict=False if p.get("rtype") == "conj" else has_conflict(p.get("w")))
        strata[h].append(key)
    subset = manifest_subset(manifest)
    bad = [k for k in subset if k not in parents or parents[k]["family"] not in PRIMARY]
    if bad or len(set(subset)) != len(subset):
        raise ValueError(f"subset keys not primary manifest parents or repeated: {bad[:3]}")
    return dict(parents=parents, strata={h: ST._order(ks) for h, ks in strata.items()}, subset=subset,
                ignored=dict(ignored))


def family_keys(design, families):
    return {k for k, v in design["parents"].items() if v["family"] in families}


def _rep_key(r):
    return -1 if r is None else r


def _pv(reps):
    """rcx.stats order: per replicate sum/len over starts, then sum over replicates (seed order) / their number."""
    return sum(sum(v) / len(v) for v in reps) / len(reps)


def arm_values(cells, arm, replicate=_ALL, field="nreg"):
    """{parent key: value} of one arm (all replicates, or one replicate; a deterministic arm serves every replicate)."""
    by_rep = cells.get(arm) or {}
    if replicate is _ALL:
        reps = sorted(by_rep, key=_rep_key)
    elif replicate in by_rep:
        reps = [replicate]
    elif None in by_rep:
        reps = [None]
    else:
        return {}
    out = {}
    for k in {k for r in reps for k in by_rep[r]}:
        vals = [by_rep[r][k][field] for r in reps if k in by_rep[r] and field in by_rep[r][k]]
        if vals:
            out[k] = _pv(vals)
    return out


# ============================================================================ stratified paired cluster bootstrap

_IDX = {}


def index_matrix(h, n, resamples):
    """R x n indices of stratum h from random.Random(derive(20261250, "boot:<h>")), drawn as rcx.stats._boot_means."""
    key = (h, n, resamples)
    if key not in _IDX:
        rng = random.Random(derive(BOOT_SEED, f"boot:{h}"))
        _IDX[key] = [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]
    return _IDX[key]


def estimate(design, values, resamples, keys=None):
    """Stratum, family and primary-macro means of {parent: value} (points; with bootstrap replicates if resamples)."""
    strata = {}
    for h, order in design["strata"].items():
        ks = [k for k in order if k in values and (keys is None or k in keys)]
        if not ks:
            continue
        v = [values[k] for k in ks]
        n = len(v)
        boot = [sum(map(v.__getitem__, idx)) / n for idx in index_matrix(h, n, resamples)] if resamples else None
        strata[h] = dict(n=n, point=sum(v) / n, boot=boot)
    fams = {}
    for f in TEST_ORDER:
        parts = [strata[h] for h in STRATA[f] if h in strata]
        if len(parts) == 1:
            fams[f] = parts[0]
        elif parts:
            N = sum(p["n"] for p in parts)
            boot = [sum(p["n"] * p["boot"][r] for p in parts) / N for r in range(resamples)] if resamples else None
            fams[f] = dict(n=N, point=sum(p["n"] * p["point"] for p in parts) / N, boot=boot)
    macro = None
    if all(f in fams for f in PRIMARY):
        a, b, c = (fams[f] for f in PRIMARY)
        boot = [(x + y + z) / 3 for x, y, z in zip(a["boot"], b["boot"], c["boot"])] if resamples else None
        macro = dict(n=a["n"] + b["n"] + c["n"], point=(a["point"] + b["point"] + c["point"]) / 3, boot=boot)
    return dict(strata=strata, families=fams, macro=macro)


def p_two(boot, resamples):
    return ST.bootstrap_p_two_sided(boot, floor=1.0 / resamples, ties=TIES)


def p_ni(boot, resamples):
    """§6.2 P2: max(1/R, share of replicates with D2* >= delta_NI)."""
    return max(1.0 / resamples, sum(1 for d in boot if _ge(d, DELTA_NI)) / resamples)


def summ(e, resamples, ni=False):
    if e is None:
        return None
    out = dict(n=e["n"], point=e["point"])
    if e.get("boot") is not None:
        lo, hi = ST._interval(e["boot"], ALPHA)
        out.update(lo=lo, hi=hi, p=p_two(e["boot"], resamples), upper95=ST._quantile(e["boot"], 1 - ALPHA))
        if ni:
            out["p_ni"] = p_ni(e["boot"], resamples)
    return out


def summarize(est, resamples, ni=False):
    if est is None:
        return None
    return dict(strata={h: summ(e, resamples, ni) for h, e in est["strata"].items()},
                families={f: summ(e, resamples, ni) for f, e in est["families"].items()},
                macro=summ(est["macro"], resamples, ni))


def contrast(design, cells, x, y, resamples, replicate=_ALL, keys=None):
    """Estimate of D = nreg(x) - nreg(y) (per-parent paired difference; rcx.stats b - a with b = x), or None."""
    vx, vy = arm_values(cells, x, replicate), arm_values(cells, y, replicate)
    d = {k: vx[k] - vy[k] for k in vx if k in vy}
    return estimate(design, d, resamples, keys) if d else None


def level(design, cells, arm, keys=None):
    """Point means of one arm (stratum / family / macro)."""
    v = arm_values(cells, arm)
    return estimate(design, v, 0, keys) if v else None


def _pt(est, where="macro"):
    """Point of an estimate at 'macro', a family or a stratum (None if absent)."""
    if est is None:
        return None
    e = est["macro"] if where == "macro" else est["families"].get(where) or est["strata"].get(where)
    return None if e is None else e["point"]


# ============================================================================ tests, determinations and the rule

def holm_family(det):
    """The confirmatory family fixed at the freeze (§6.2); TEST validity never changes it."""
    return [t for t, off in (("P1", det["p1_comparator_invalid"]), ("P2", det["p2_dropped"])) if not off]


def holm_adjust(family, raw):
    """rcx.stats.holm over the family (a test without data gets p = 1)."""
    return ST.holm({t: 1.0 if raw.get(t) is None else raw[t] for t in family}) if family else {}


def p1_test(d1, d1_reps, family, raw, resamples):
    """P1 (§6.2): IUT of P1-M and P1-ID with conditions 1-4; conjunct flags for the REVISE note."""
    m = d1["macro"] if d1 else None
    i = d1["families"].get("H-ID") if d1 else None
    sm, si = summ(m, resamples), summ(i, resamples)
    reps = {k: dict(macro=_pt(d1_reps.get(k)), hid=_pt(d1_reps.get(k), "H-ID")) for k in REPLICATES}
    out = dict(in_family="P1" in family, macro=sm, hid=si, replicates={str(k): v for k, v in reps.items()},
               p_macro=sm.get("p") if sm else None, p_hid=si.get("p") if si else None, p=None, p_holm=None,
               conditions=None, conjuncts=None, passes=None)
    if sm is None or si is None:
        return out
    out["p"] = max(sm["p"], si["p"])
    adj = holm_adjust(family, raw).get("P1")
    out["p_holm"] = adj

    def conj(s, which, p_c):
        a = holm_adjust(family, {**raw, "P1": p_c}).get("P1")
        flags = dict(holm_p=a is not None and _lt(a, ALPHA), point=_ge(s["point"], P1_MIN), lower_bound=_gt(s["lo"], 0),
                     replicate_signs=all(reps[k][which] is not None and _gt(reps[k][which], 0) for k in REPLICATES))
        return dict(p_holm=a, flags=flags, passes=all(flags.values()) if out["in_family"] else None)

    out["conjuncts"] = dict(M=conj(sm, "macro", sm["p"]), ID=conj(si, "hid", si["p"]))
    c = dict(holm_p=adj is not None and _lt(adj, ALPHA), points=_ge(sm["point"], P1_MIN) and _ge(si["point"], P1_MIN),
             lower_bounds=_gt(sm["lo"], 0) and _gt(si["lo"], 0),
             replicate_signs=all(reps[k]["macro"] is not None and reps[k]["hid"] is not None and _gt(reps[k]["macro"], 0)
                                 and _gt(reps[k]["hid"], 0) for k in REPLICATES))
    out["conditions"] = c
    out["passes"] = all(c.values()) if out["in_family"] else None
    return out


def p2_test(d2, family, raw, resamples):
    m = d2["macro"] if d2 else None
    sm = summ(m, resamples, ni=True)
    out = dict(in_family="P2" in family, macro=sm, p=sm.get("p_ni") if sm else None, p_holm=None, passes=None)
    if sm is None:
        return out
    out["p_holm"] = holm_adjust(family, raw).get("P2")
    if out["in_family"]:
        out["passes"] = out["p_holm"] is not None and _lt(out["p_holm"], ALPHA)
    return out


def decide(f):
    """§7.2 ordered rule on the §7.1 determinations f; returns (decision, reasons). VOID never reaches here."""
    if f["defects"]:
        return INC_DEFECT, [f"defect: {d}" for d in f["defects"]] + [f"missing: {m}" for m in f["missing"]]
    if f["missing"]:
        return INC_MISSING, [f"missing protected-core cells: {m}" for m in f["missing"]]
    if f["stop_d"]:
        return STOP_D, ["STOP-D condition: D_G >= 0.05 and its 95% CI lower bound > 0 (P1/P2 are reported as "
                        "mechanism findings only)"]
    p1c, p2c = f["p1_counts"], f["p2_counts"]
    if p1c and p2c and f["g1"] and f["sf_floor"]:
        return ADVANCE, ["P1 counts, P2 counts, G1 holds, SF floor holds"]
    if f["xgpi"] is not None and _ge(f["xgpi"], CEILING) and not p1c and not p2c:
        return STOP_M, ["EXACT-GPI(L) primary macro nreg >= 0.5 and neither P1 nor P2 counts"]
    hm, hi = f["d1_macro_hi"], f["d1_hid_hi"]
    futile = [w for w, x in (("macro", hm), ("H-ID", hi)) if x is not None and _lt(x, PLANNED)]
    if not p2c and futile:
        return STOP_F, [f"P2 does not count and the {' and '.join(futile)} D1 upper bound < 0.05"]
    if p1c or p2c:
        return REVISE, ["P1 counts" if p1c else "P2 counts"] + (["P2 counts"] if p1c and p2c else [])
    return INCONCLUSIVE, ["no earlier rule holds"]


def revise_notes(f, P1, P2, det):
    """Every failed ADVANCE condition and the §7.2 item 7 notes (listed for any decision; binding wording for REVISE)."""
    n = []
    if not f["g1"] and not f["stop_d"]:
        n.append(f"G1 fails in the middle zone: D_G = {_fmt(f['dg'])} >= 0.05 with a 95% CI that includes 0 "
                 f"[{_fmt(f['dg_lo'])}, {_fmt(f['dg_hi'])}]")
    if not f["sf_floor"]:
        n.append("SF floor fails: SF-GPI nreg > 0.50 on " + ", ".join(f["sf_floor_failed"]))
    if f["p2_counts"] and not f["p1_counts"]:
        n.append("label-cost claim only (P2 counts, P1 does not)")
    if f["p1_counts"] and not f["p2_counts"]:
        n.append("P1 counts, P2 does not")
    if det["p1_comparator_invalid"]:
        n.append("P1 comparator-invalid at the pilot: P1 descriptive, ADVANCE unavailable")
    elif P1["passes"] and not f["p1_valid"]:
        n.append(f"P1 passed but does not count: comparator-invalid on TEST (UVFA H-ID nreg {_fmt(f['uvfa_hid'], 0)} "
                 "> 0.50)")
    if det["p2_dropped"]:
        n.append("P2 dropped at the pilot: ADVANCE unavailable")
    elif P2["passes"] and not f["p2_valid"]:
        n.append(f"P2 passed but does not count: comparator-invalid on TEST (PG400 primary macro nreg "
                 f"{_fmt(f['pg_macro'], 0)} > 0.50)")
    cj = P1.get("conjuncts") or {}
    if cj and cj["M"]["passes"] and not cj["ID"]["passes"] and f["p2_counts"]:
        n.append("P1-M passes, P1-ID does not, P2 counts (follow-up: UVFA-wide/SF-wide arms trained on weights "
                 "covering sign and cardinality)")
    if f["xgpi"] is not None and _ge(f["xgpi"], CEILING):
        n.append("library ceiling >= 0.5: any gain is through the z = w' index (GPI qualifier applies)")
    return n


def qualifiers(design, cells, det, est, resamples):
    """§7.2 qualifiers (wording only)."""
    sf, aux, nog = (_pt(level(design, cells, a)) for a in (SF, AUX, NOGPI))
    attribution = sf is not None and aux is not None and _le(aux, sf + ATTR_MARGIN)
    gpi = sf is not None and nog is not None and _le(nog, sf + GPI_MARGIN)
    u1 = summ(est["D1"]["strata"].get("H-ID-U"), resamples) if est.get("D1") else None
    u2 = summ(est["D2"]["strata"].get("H-ID-U"), resamples) if est.get("D2") else None
    newdir = bool(u1 and u2 and _ge(u1["point"], NEWDIR_POINT) and _gt(u1["lo"], 0) and _lt(u2["hi"], NEWDIR_D2_HI))
    return dict(
        attribution=dict(holds=attribution, uvfa_aux=aux, sf_gpi=sf,
                         wording="gain credited to vector consequence supervision, not to test-time recomposition"
                         if attribution else None),
        gpi=dict(holds=gpi, sf_nogpi=nog, sf_gpi=sf, wording="goal-conditioned endpoint predictor read out linearly"
                 if gpi else "GPI"),
        new_directions=dict(holds=newdir, hid_u_d1=u1, hid_u_d2=u2, wording="new preference directions" if newdir
                            else "re-weighting of seen fact patterns"),
        naming=NAMING,
        guard_quality=dict(holds=bool(det["q9_failed_no_bug"]),
                           wording=GUARD_NOTE if det["q9_failed_no_bug"] else None),
        kstar=dict(descriptive_only=bool(det["kstar_descriptive"])))


# ============================================================================ descriptive (§6.5)

def kstar_point(m_ref, ms, ks):
    """log10 k* of the first crossing nreg(PGV_k) <= m_ref (linear in log10 k); -inf / +inf when censored."""
    if _le(ms[0], m_ref):
        return -math.inf
    for i in range(len(ks) - 1):
        if _le(ms[i + 1], m_ref):
            t = min(1.0, max(0.0, (ms[i] - m_ref) / (ms[i] - ms[i + 1])))
            a, b = math.log10(ks[i]), math.log10(ks[i + 1])
            return a + t * (b - a)
    return math.inf


def _cquantile(vals, q):
    """Percentile with censoring: an infinite neighbour is returned (the bound propagates outward)."""
    v = sorted(vals)
    idx = (len(v) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(v) - 1)
    a, b = v[lo], v[hi]
    if math.isinf(a) and a < 0:
        return a
    if math.isinf(b) and b > 0:
        return b
    return ST._quantile(v, q)


def _klabel(x, ks):
    if x == -math.inf:
        return f"<= {ks[0]}"
    if x == math.inf:
        return f"> {ks[-1]}"
    return f"{10 ** x:.0f}"


def kstar(design, cells, arms_by_k, resamples):
    """k* of a k-curve against SF-GPI on the subset (macro; paired stratum streams), with a censored percentile CI."""
    keys = set(design["subset"])
    ks = tuple(k for k in sorted(arms_by_k) if arm_values(cells, arms_by_k[k]))
    if not keys or not ks or not arm_values(cells, SF):
        return None
    ref = estimate(design, arm_values(cells, SF), resamples, keys)
    curves = [estimate(design, arm_values(cells, arms_by_k[k]), resamples, keys) for k in ks]
    if ref["macro"] is None or any(c["macro"] is None for c in curves):
        return None
    point = kstar_point(ref["macro"]["point"], [c["macro"]["point"] for c in curves], ks)
    boots = [kstar_point(ref["macro"]["boot"][r], [c["macro"]["boot"][r] for c in curves], ks) for r in range(resamples)]
    lo, hi = _cquantile(boots, ALPHA / 2), _cquantile(boots, 1 - ALPHA / 2)
    return dict(ks=list(ks), arms=[arms_by_k[k] for k in ks], n_parents=len(keys), sf_gpi=ref["macro"]["point"],
                curve=[c["macro"]["point"] for c in curves], kstar=_klabel(point, ks), lo=_klabel(lo, ks),
                hi=_klabel(hi, ks), log10=None if math.isinf(point) else point,
                share_censored_low=sum(b == -math.inf for b in boots) / resamples,
                share_censored_high=sum(b == math.inf for b in boots) / resamples)


def binary_view(cells, x, y, keys):
    """Exact McNemar on 'optimal (nreg <= 1e-9) on >= half of the (replicate, start) values' per parent."""
    def opt(arm):
        out = {}
        for rep in (cells.get(arm) or {}).values():
            for k, c in rep.items():
                if k in keys:
                    out.setdefault(k, []).extend(1.0 if v <= EPS else 0.0 for v in c["nreg"])
        return out
    a, b = opt(y), opt(x)
    common = ST._order(set(a) & set(b))
    if len(common) < 1:
        return None
    only_y, only_x = ST.paired_binary_table({k: a[k] for k in common}, {k: b[k] for k in common}, 0.5)
    return dict(n=len(common), only_y=only_y, only_x=only_x, p=ST.mcnemar_exact(only_y, only_x))


def by_attr(design, est_values, attr):
    """Point means of a per-parent contrast by K or conflict within each primary family."""
    out = {}
    for fam in PRIMARY:
        groups = defaultdict(list)
        for k in ST._order(est_values):
            info = design["parents"][k]
            if info["family"] == fam:
                groups[str(info[attr])].append(est_values[k])
        out[fam] = {g: dict(n=len(v), point=sum(v) / len(v)) for g, v in sorted(groups.items())}
    return out


def extras_table(design, cells):
    """Means of every optional per-parent field (loops, bank regret1/top-1, model quality, ...) per family."""
    out = {}
    for arm in arms_sorted(cells):
        fields = sorted({f for rep in cells[arm].values() for c in rep.values() for f in c
                         if f != "nreg" and f not in TIE_FIELDS})
        for fld in fields:
            e = estimate(design, arm_values(cells, arm, field=fld), 0)
            out.setdefault(arm, {})[fld] = dict({f: v["point"] for f, v in e["families"].items()},
                                                primary_macro=_pt(e))
    return out


def tie_shares(design, cells):
    """§6.5 ENUM± diagnostic: per family, (all-tied decisions) / (decisions with >= 2 legal moves), pooled over the
    family's parents, starts and replicates."""
    out = {}
    for arm in arms_sorted(cells):
        tot = defaultdict(lambda: [0.0, 0.0])
        for rep in cells[arm].values():
            for k, c in rep.items():
                if all(t in c for t in TIE_FIELDS):
                    t = tot[design["parents"][k]["family"]]
                    t[0] += sum(c["all_tied"])
                    t[1] += sum(c["decisions"])
        if tot:
            out[arm] = {f: dict(all_tied=a, decisions=d, share=a / d if d else None)
                        for f, (a, d) in sorted(tot.items(), key=lambda kv: TEST_ORDER.index(kv[0]))}
    return out


def arms_sorted(cells):
    return [a for a in ARM_ORDER if a in cells] + sorted(a for a in cells if a not in ARM_ORDER)


def coverage(design, cells):
    out = {}
    for arm in arms_sorted(cells):
        for rep in sorted(cells[arm], key=_rep_key):
            fams = Counter(design["parents"][k]["family"] for k in cells[arm][rep])
            out[arm if rep is None else f"{arm}@{rep}"] = {f: fams[f] for f in TEST_ORDER if fams[f]}
    return out


# ============================================================================ protected core (§10.3)

def kcurve_arms(det):
    return {int(k): v for k, v in (det.get("kcurve_arms") or {str(k): f"PGV_{k}" for k in KS}).items()}


def core_groups(design, det):
    """{group id: parent keys} of the protected core (§10.3); group id = arm, arm@replicate, or bank:<that> for the
    one-step bank of a protected cell (BANK_FIELD present)."""
    cut = {CUT_FAMILIES[i] for i in det.get("cut_items") or () if i in CUT_FAMILIES}
    fams = {v["family"] for v in design["parents"].values()} - cut
    prim, allk, idk = family_keys(design, PRIMARY), family_keys(design, fams), family_keys(design, ("H-ID", "H-K"))
    g = {f"{a}@{k}": allk for a in ZERO_SHOT for k in REPLICATES}
    g[f"{SHUF}@{REPLICATES[0]}"] = allk
    g[PGV] = set(prim)
    if det["pg400"] == FT:
        g.update({f"{FT}@{k}": prim for k in REPLICATES})
    for arm in kcurve_arms(det).values():
        g[arm] = set(g.get(arm, set())) | set(design["subset"])
    g.update({a: allk for a in FLOORS})
    g.update({a: idk for a in (ENUM_H, GNN)})
    g.update({f"bank:{gid}": keys for gid, keys in list(g.items()) if gid.partition("@")[0] not in BANK_EXEMPT})
    return {k: v for k, v in g.items() if v}


def absent_groups(cells, groups):
    """{group id: number of its parents without the cell (bank groups: without BANK_FIELD)}."""
    out = {}
    for gid, keys in groups.items():
        bank = gid.startswith("bank:")
        arm, _, rep = gid[5:].partition("@") if bank else gid.partition("@")
        have = (cells.get(arm) or {}).get(int(rep) if rep else None, {})
        miss = [k for k in keys if k not in have or (bank and BANK_FIELD not in have[k])]
        if miss:
            out[gid] = len(miss)
    return dict(sorted(out.items()))


# ============================================================================ analysis

def analyze(design, cells, det, defects=(), resamples=RESAMPLES, custody=None):
    """Everything of §6-7 from the design, the eval cells {arm: {replicate: {key: {"nreg": [...], ...}}}} and the
    pilot determinations. Pure; the decision is decide() on the §7.1 determinations."""
    family = holm_family(det)
    pg = det["pg400"]
    est = dict(D1=contrast(design, cells, UVFA, SF, resamples), D_G=contrast(design, cells, SF, ENUM, resamples),
               D2=contrast(design, cells, SF, pg, resamples) if pg else None)
    reps = {name: {k: contrast(design, cells, x, y, 0, replicate=k) for k in REPLICATES}
            for name, x, y in (("D1", UVFA, SF), ("D2", SF, pg), ("D_G", SF, ENUM)) if y}
    d1m, d1i = (est["D1"]["macro"], est["D1"]["families"].get("H-ID")) if est["D1"] else (None, None)
    raw = dict(P1=max(p_two(d1m["boot"], resamples), p_two(d1i["boot"], resamples)) if d1m and d1i else None,
               P2=p_ni(est["D2"]["macro"]["boot"], resamples) if est["D2"] and est["D2"]["macro"] else None)
    P1 = p1_test(est["D1"], reps["D1"], family, raw, resamples)
    P2 = p2_test(est["D2"], family, raw, resamples)
    lv = {a: level(design, cells, a) for a in arms_sorted(cells)}
    uvfa_hid, pg_macro = _pt(lv.get(UVFA), "H-ID"), _pt(lv.get(pg)) if pg else None
    sf_fam = {f: _pt(lv.get(SF), f) for f in PRIMARY}
    dg = summ(est["D_G"]["macro"], resamples) if est["D_G"] else None
    absent = absent_groups(cells, core_groups(design, det))
    p1_valid = not det["p1_comparator_invalid"] and uvfa_hid is not None and _le(uvfa_hid, BAR)
    p2_valid = not det["p2_dropped"] and pg_macro is not None and _le(pg_macro, BAR)
    f = dict(defects=list(defects), missing=[f"{g} ({n} parents)" for g, n in absent.items()],
             p1_valid=p1_valid, p2_valid=p2_valid, uvfa_hid=uvfa_hid, pg_macro=pg_macro,
             p1_counts=bool(p1_valid and P1["passes"]), p2_counts=bool(p2_valid and P2["passes"]),
             dg=dg["point"] if dg else None, dg_lo=dg["lo"] if dg else None, dg_hi=dg["hi"] if dg else None,
             g1=bool(dg and _lt(dg["point"], DOMINANCE)), stop_d=bool(dg and _ge(dg["point"], DOMINANCE) and
                                                                       _gt(dg["lo"], 0)),
             sf_floor=all(v is not None and _le(v, BAR) for v in sf_fam.values()),
             sf_floor_failed=[fm for fm, v in sf_fam.items() if v is None or not _le(v, BAR)],
             xgpi=_pt(lv.get(XGPI)), d1_macro_hi=P1["macro"]["hi"] if P1["macro"] else None,
             d1_hid_hi=P1["hid"]["hi"] if P1["hid"] else None)
    decision, reasons = decide(f)
    notes = revise_notes(f, P1, P2, det)
    contrasts = {"D1 = UVFA - SF-GPI": summarize(est["D1"], resamples),
                 "D2 = SF-GPI - PG400": summarize(est["D2"], resamples, ni=True),
                 "D_G = SF-GPI - ENUM-MIS±": summarize(est["D_G"], resamples)}
    others = tuple((f"D2 (non-selected candidate) = SF-GPI - {c}", SF, c) for c in (PGV, FT) if c != pg)
    for label, x, y in others + DESCRIPTIVE:
        c = contrast(design, cells, x, y, resamples)
        if c is not None:
            contrasts[label] = summarize(c, resamples, ni=label.startswith("D2"))
    prim = family_keys(design, PRIMARY)
    per_param = {}
    for name, x, y in (("D1", UVFA, SF), ("D2", SF, pg), ("D_G", SF, ENUM)):
        vx, vy = (arm_values(cells, x), arm_values(cells, y)) if y else ({}, {})
        dv = {k: vx[k] - vy[k] for k in vx if k in vy and k in prim}
        per_param[name] = dict(K=by_attr(design, dv, "K"), conflict=by_attr(design, dv, "conflict"))
    kc = kcurve_arms(det)
    res = dict(
        experiment=EXPERIMENT, resamples=resamples, prereg_resamples=resamples == PREREG_RESAMPLES,
        boot_streams={h: f"random.Random(derive({BOOT_SEED}, 'boot:{h}'))" for h in design["strata"]},
        custody=custody or {}, pilot=det, holm_family=family, raw_p=raw,
        holm_adjusted=holm_adjust(family, raw), P1=P1, P2=P2,
        validity=dict(bar=BAR, p1_valid=p1_valid, p2_valid=p2_valid, uvfa_test_hid=uvfa_hid, pg400=pg,
                      pg400_test_primary_macro=pg_macro, p1_counts=f["p1_counts"], p2_counts=f["p2_counts"],
                      pg400_test_families={fm: _pt(lv.get(pg), fm) for fm in TEST_ORDER} if pg else None,
                      pg400_dev_ood=det.get("pg400_dev_ood")),
        guard=dict(D_G=dg, G1=f["g1"], stop_d_condition=f["stop_d"],
                   statement_suffix=GUARD_NOTE if det["q9_failed_no_bug"] else None),
        sf_floor=dict(holds=f["sf_floor"], family_means=sf_fam, failed=f["sf_floor_failed"]),
        exact_gpi_primary_macro=f["xgpi"], missing=absent, defects=list(defects),
        decision=decision, reasons=reasons, notes=notes, failed_advance=notes if decision == REVISE else [],
        qualifiers=qualifiers(design, cells, det, est, resamples),
        arm_means={a: dict({fm: e["point"] for fm, e in lv[a]["families"].items()},
                           **{h: e["point"] for h, e in lv[a]["strata"].items() if h.startswith("H-ID-")},
                           primary_macro=_pt(lv[a])) for a in lv if lv[a]},
        contrasts=contrasts,
        per_replicate={name: {str(k): dict(macro=_pt(e), **({f: e["families"][f]["point"] for f in e["families"]}
                                                           if e else {})) for k, e in r.items()}
                       for name, r in reps.items()},
        strata_k_conflict=per_param,
        kstar=dict(PGV=kstar(design, cells, kc, resamples), FT=kstar(design, cells, {400: FT}, resamples),
                   descriptive_only=bool(det["kstar_descriptive"])),
        binary_view={f"{x} vs {y}": binary_view(cells, x, y, prim) for x, y in ((SF, UVFA), (SF, pg), (SF, ENUM)) if y},
        extras=extras_table(design, cells), enum_all_tied=tie_shares(design, cells), coverage=coverage(design, cells),
        labels=dict(families=FAMILY_LABELS, D1=D1_LABELS),
        design=dict(strata={h: len(v) for h, v in design["strata"].items()}, subset=len(design["subset"]),
                    ignored_families=design.get("ignored", {})))
    return res


# ============================================================================ custody and loading

def refuse(msg):
    raise SystemExit(f"refusing (nothing written; not an analysis run): {msg}")


def committed_text(root, rel="DECISIONS.md"):
    """The file as committed at HEAD (`git -C root show HEAD:./rel`), or None."""
    git = shutil.which("git")
    if git is None:
        return None
    r = subprocess.run([git, "-C", str(root), "show", f"HEAD:./{rel}"], capture_output=True)
    return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None


def native_determinations(rep):
    """The §7.1 pilot inputs from run_e05.py's pilot-report fields (a report without a 'determinations' object).
    q9_failed_no_bug = not G_P1.Q9.ok: a Q9 failure triggers the enum_e05 audit, and a bug found is fixed and the pilot
    phase re-run (§9.1), so a failure still standing in the final report is a failure without a bug."""
    try:
        pg, cuts, recipe, ks = rep["pg400"], list(rep["G_P5"]["cuts"]), rep["kcurve_recipe"], rep["kstar"]
        det = dict(p1_comparator_invalid=rep["p1_comparator_invalid"], p2_dropped=rep["p2_dropped"],
                   q9_failed_no_bug=not rep["G_P1"]["Q9"]["ok"], n=rep.get("n"), source="run_e05 pilot-report fields")
        grid = (rep.get("G_P3") or {}).get("grid") or {}
        det["native_cfg"] = {kind: (grid.get(kind) or {}).get("cfg") for kind in ("PGV", "FT")}
    except (KeyError, TypeError) as e:
        raise ValueError(f"PILOT_REPORT.json has neither 'determinations' nor run_e05's pilot-report fields ({e!r})")
    if pg not in (None, "PGV", "FT") or recipe not in ("original", "R_ext") or \
            ks not in ("decision-driving", "descriptive only"):
        raise ValueError(f"unknown pg400 {pg!r}, kcurve_recipe {recipe!r} or kstar {ks!r}")
    bad = [c for c in cuts if c not in CUT_NAMES]
    if bad:
        raise ValueError(f"unknown G_P5 cuts {bad} (expected run_e05.CUT_ORDER names)")
    det.update(pg400={None: None, "PGV": PGV, "FT": FT}[pg], kstar_descriptive=ks == "descriptive only",
               r_ext=recipe == "R_ext", cut_items=sorted(CUT_NAMES.index(c) + 1 for c in cuts))
    if det["r_ext"]:
        det["kcurve_arms"] = {str(k): RX.format(k) for k in KS}
    dev = (rep.get("dev_nreg") or {}).get(f"PG400={pg}")
    if dev:
        det["pg400_dev_ood"] = dev
    return det


def pilot_determinations(report):
    det = (report or {}).get("determinations")
    if det is None and isinstance(report, dict):
        det = native_determinations(report)
    if not isinstance(det, dict):
        raise ValueError("PILOT_REPORT.json has no 'determinations' object")
    lost = [k for k in DET_KEYS if k not in det]
    if lost:
        raise ValueError(f"pilot determinations lack {lost}")
    for k in DET_KEYS:
        if k != "pg400" and not isinstance(det[k], bool):
            raise ValueError(f"pilot determination {k} must be a bool, got {det[k]!r}")
    if det["p2_dropped"] != (det["pg400"] is None) or det["pg400"] not in (None, PGV, FT):
        raise ValueError(f"pg400 {det['pg400']!r} is inconsistent with p2_dropped {det['p2_dropped']}")
    cut = det.get("cut_items") or []
    if not all(type(i) is int and 1 <= i <= 7 for i in cut):
        raise ValueError(f"cut_items must be §10.3 item numbers 1-7, got {cut!r}")
    kc = det.get("kcurve_arms")
    if kc is not None and (not isinstance(kc, dict) or sorted(int(k) for k in kc) != list(KS)):
        raise ValueError(f"kcurve_arms must map 400/1600/6400 to arm names, got {kc!r}")
    return dict(det)


def _nums(key, fld, xs, n=None, unit=False):
    if not isinstance(xs, list) or not xs or (n is not None and len(xs) != n):
        raise ValueError(f"{key}: {fld} must be a non-empty list" + (f" of {n} values (one per start)" if n else ""))
    out = []
    for x in xs:
        if isinstance(x, bool):
            x = float(x)
        if not isinstance(x, (int, float)) or not math.isfinite(x) or (unit and not 0.0 <= x <= 1.0):
            raise ValueError(f"{key}: {fld} value {x!r} is not a finite number" + (" in [0, 1]" if unit else ""))
        out.append(float(x))
    return out


def _load(p):
    """A JSON object, or a refusal."""
    try:
        d = read_json(p)
    except (OSError, ValueError) as e:
        refuse(f"cannot read {p}: {e!r}")
    if not isinstance(d, dict):
        refuse(f"{p} is not a JSON object")
    return d


def load_evals(adir, design, manifest_sha, code_digest, det):
    det_arms = DETERMINISTIC | set(kcurve_arms(det).values())
    cells, records = {}, []
    files = sorted((adir / "evals").glob("*.json"))
    for fp in files:
        d = _load(fp)
        if d.get("manifest_sha256") != manifest_sha:
            refuse(f"{fp.name} names manifest {d.get('manifest_sha256')!r}, not {manifest_sha}")
        if d.get("code_digest") != code_digest:
            refuse(f"{fp.name} names code digest {d.get('code_digest')!r}, not {code_digest}")
        arm, rep = d.get("arm"), d.get("replicate")
        if not isinstance(arm, str) or not arm or "@" in arm:
            refuse(f"{fp.name}: malformed arm {arm!r}")
        if rep is not None and type(rep) is not int:
            refuse(f"{fp.name}: replicate {rep!r} is not one of {REPLICATES} or null")
        _check_arm(fp.name, arm, rep, det_arms)
        slot = cells.setdefault(arm, {}).setdefault(rep, {})
        body = d.get("cells")
        if not isinstance(body, dict) or not body:
            refuse(f"{fp.name}: no cells")
        for key, cell in body.items():
            if key not in design["parents"]:
                refuse(f"{fp.name}: parent {key!r} is not a TEST parent of the manifest")
            if key in slot:
                refuse(f"{fp.name}: cell ({arm}, {rep}, {key}) repeats")
            cell = {"nreg": cell} if isinstance(cell, list) else cell
            if not isinstance(cell, dict) or "nreg" not in cell:
                refuse(f"{fp.name}: cell {key} has no nreg")
            try:
                slot[key] = {fld: _nums(key, fld, xs, design["parents"][key]["n_starts"] if fld == "nreg" else None,
                                        unit=fld == "nreg") for fld, xs in cell.items()}
            except ValueError as e:
                refuse(f"{fp.name}: {e}")
        records.append(dict(file=fp.name, sha256=sha(fp), arm=arm, replicate=rep, parents=len(body)))
    if not files:
        refuse(f"no eval files in {adir / 'evals'}")
    return cells, records


_SEED_TAG = re.compile(r"\|s(\d+)(?:\||$)")


def native_arm(arm, tag, cfg=None):
    """(analysis arm, replicate) of a run_e05 cells.jsonl / banks.jsonl record (ValueError if malformed)."""
    if not isinstance(arm, str) or not arm or "@" in arm or not isinstance(tag, str) or not tag:
        raise ValueError(f"malformed arm/tag {arm!r} / {tag!r}")
    cfg = cfg or {}
    m = _SEED_TAG.search(tag)
    rep = int(m.group(1)) if m else None
    if arm == "PGV":
        mk = re.fullmatch(r"(.+)\|k(\d+)", tag)
        if not mk:
            raise ValueError(f"PGV tag {tag!r} is not '<cfg>|k<k>'")
        name, k = mk.group(1), int(mk.group(2))
        if k == 400 and name == cfg.get("PGV"):
            return PGV, None
        if k in KS and name == f"k{k}" and k != 400:
            return f"PGV_{k}", None
        if k in KS and name == f"rext{k}":
            return RX.format(k), None
        return f"PGV[{tag}]", None
    if arm == "FT":
        mk = re.fullmatch(r"(.+)\|s(\d+)\|k400", tag)
        if not mk:
            raise ValueError(f"FT tag {tag!r} is not '<cfg>|s<seed>|k400'")
        return (FT if mk.group(1) == cfg.get("FT") else f"FT[{mk.group(1)}]"), int(mk.group(2))
    if arm in MODEL_OF and tag != f"{MODEL_OF[arm]}|s{rep}":
        raise ValueError(f"{arm} tag {tag!r} is not '{MODEL_OF[arm]}|s<seed>'")
    return arm, rep


def _jsonl(fp):
    """Records of a keyed JSONL log (a torn last line, a crash mid-append, is ignored as run_e05 does)."""
    out, keys = [], set()
    lines = fp.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            if i == len(lines):
                continue
            refuse(f"{fp.name} line {i} is not JSON")
        if not isinstance(r, dict) or r.get("key") in keys:
            refuse(f"{fp.name} line {i}: not a record or a repeated key {r.get('key') if isinstance(r, dict) else r!r}")
        keys.add(r.get("key"))
        out.append(r)
    return out


def _check_arm(where, arm, rep, det_arms):
    if rep is not None and rep not in REPLICATES:
        refuse(f"{where}: replicate {rep!r} is not one of {REPLICATES} or null")
    if (arm in REPLICATED and rep is None) or (arm in det_arms and rep is not None):
        refuse(f"{where}: arm {arm} with replicate {rep!r}")


def load_native(adir, design, code_digest, det, attempt):
    """run_e05.py's keyed logs: <attempt>/cells.jsonl (one record per start) and optional banks.jsonl."""
    det_arms = DETERMINISTIC | set(kcurve_arms(det).values())
    split, cfg = f"test:{attempt}", det.get("native_cfg") or {}
    fp, acc = adir / "cells.jsonl", {}
    for i, r in enumerate(_jsonl(fp), 1):
        where = f"{fp.name} record {i}"
        if r.get("digest") != code_digest:
            refuse(f"{where} names code digest {r.get('digest')!r}, not {code_digest}")
        if r.get("split") != split:
            refuse(f"{where} has split {r.get('split')!r}, not {split!r}")
        try:
            arm, rep = native_arm(r.get("arm"), r.get("tag"), cfg)
        except ValueError as e:
            refuse(f"{where}: {e}")
        _check_arm(where, arm, rep, det_arms)
        key, s0 = r.get("parent"), r.get("s0")
        info = design["parents"].get(key) if isinstance(key, str) else None
        if info is None:
            refuse(f"{where}: parent {key!r} is not a TEST parent of the manifest")
        if info["starts"] is None or s0 not in info["starts"]:
            refuse(f"{where}: start {s0!r} is not a manifest start of {key}")
        slot = acc.setdefault((arm, rep, key), {})
        if s0 in slot:
            refuse(f"{where}: cell ({arm}, {rep}, {key}, start {s0}) repeats")
        slot[s0] = r
    cells, incomplete = {}, Counter()
    for (arm, rep, key), by_s in acc.items():
        starts = design["parents"][key]["starts"]
        if len(by_s) != len(starts):
            incomplete[arm if rep is None else f"{arm}@{rep}"] += 1
            continue
        rows = [by_s[s] for s in starts]
        try:
            cell = {"nreg": _nums(key, "nreg", [x.get("nreg") for x in rows], len(starts), unit=True)}
            for fld in ("loop", "capped", "T"):
                if all(fld in x for x in rows):
                    cell[fld] = _nums(key, fld, [x[fld] for x in rows])
            if all(isinstance(x.get("diag"), dict) and all(t in x["diag"] for t in TIE_FIELDS) for x in rows):
                cell.update({t: _nums(key, t, [x["diag"][t] for x in rows]) for t in TIE_FIELDS})
        except ValueError as e:
            refuse(f"{fp.name}: {e}")
        cells.setdefault(arm, {}).setdefault(rep, {})[key] = cell
    records = [dict(file=fp.name, sha256=sha(fp), records=sum(map(len, acc.values())),
                    cells=sum(len(v) for a in cells.values() for v in a.values()), incomplete=dict(incomplete))]
    bp = adir / "banks.jsonl"
    if bp.is_file():
        used = Counter()
        for i, r in enumerate(_jsonl(bp), 1):
            where = f"{bp.name} record {i}"
            if r.get("digest") != code_digest or r.get("split") != split:
                refuse(f"{where} names digest {r.get('digest')!r} / split {r.get('split')!r}")
            try:
                arm, rep = native_arm(r.get("arm"), r.get("tag"), cfg)
            except ValueError as e:
                refuse(f"{where}: {e}")
            _check_arm(where, arm, rep, det_arms)
            cell = ((cells.get(arm) or {}).get(rep) or {}).get(r.get("parent"))
            rows = r.get("rows")
            if r.get("skipped") or cell is None or not isinstance(rows, list) or not rows:
                used["skipped"] += 1
                continue
            try:
                for fld in ("regret1", "top1"):
                    if all(isinstance(x, dict) and fld in x for x in rows):
                        cell[fld] = _nums(r.get("parent"), fld, [x[fld] for x in rows])
            except ValueError as e:
                refuse(f"{where}: {e}")
            used["attached"] += 1
        records.append(dict(file=bp.name, sha256=sha(bp), **used))
    if not cells:
        refuse(f"no complete eval cell in {fp}")
    return cells, records


def load_attempt(adir, manifest_sha, code_digest, pilot_report, repo_root, missing, attempt=None):
    """Every custody check of the module docstring (refusals only); returns (design, cells, det, custody)."""
    if not HEX64.match(manifest_sha or ""):
        refuse(f"--manifest-sha must be a full lower-case 64-hex sha256, got {manifest_sha!r}")
    if not code_digest:
        refuse("--code-digest is empty")
    rp = adir / "ORACLE_TEST_REPORT.json"
    if not rp.is_file():
        refuse(f"{rp} does not exist (oracle-test has not passed this attempt)")
    orep = _load(rp)
    if orep.get("status") != "PASS":
        refuse(f"{rp.name} says {orep.get('status')!r}, not PASS (a VOID attempt is never analyzed)")
    if orep.get("manifest_sha256") != manifest_sha or orep.get("code_digest") != code_digest:
        refuse(f"{rp.name} names manifest {orep.get('manifest_sha256')!r} / digest {orep.get('code_digest')!r}, not "
               "the --manifest-sha / --code-digest given")
    mp = adir / "TEST_MANIFEST.json"
    if not mp.is_file():
        refuse(f"{mp} does not exist")
    if sha(mp) != manifest_sha:
        refuse(f"sha256({mp.name}) = {sha(mp)} differs from --manifest-sha {manifest_sha}")
    committed = committed_text(repo_root)
    if committed is None:
        refuse(f"cannot read the committed DECISIONS.md (git -C {repo_root} show HEAD:./DECISIONS.md)")
    if manifest_sha not in committed:
        refuse("the TEST_MANIFEST sha256 does not occur in the COMMITTED DECISIONS.md (the working copy does not count)")
    if not Path(pilot_report).is_file():
        refuse(f"{pilot_report} does not exist")
    manifest = _load(mp)
    if attempt is not None and manifest.get("attempt", attempt) != attempt:
        refuse(f"{mp.name} is attempt {manifest.get('attempt')!r}, not --attempt {attempt}")
    if manifest.get("code_digest_t3", code_digest) != code_digest:
        refuse(f"{mp.name} was drawn under code digest {manifest.get('code_digest_t3')!r}, not --code-digest")
    try:
        design = make_design(manifest)
        det = pilot_determinations(_load(pilot_report))
    except (KeyError, TypeError, ValueError) as e:
        refuse(f"malformed manifest or pilot report: {e!r}")
    exported, native = any((adir / "evals").glob("*.json")), (adir / "cells.jsonl").is_file()
    if exported and native:
        refuse("both evals/*.json and cells.jsonl exist: the eval source is ambiguous")
    if native:
        if attempt is None:
            refuse("cells.jsonl needs --attempt (its records carry split 'test:<attempt>')")
        cells, records = load_native(adir, design, code_digest, det, attempt)
    else:
        cells, records = load_evals(adir, design, manifest_sha, code_digest, det)
    absent = absent_groups(cells, core_groups(design, det))
    declared = set(missing)
    if set(absent) != declared:
        refuse(f"--missing must name exactly the protected-core groups with absent cells: absent but undeclared "
               f"{sorted(set(absent) - declared)}, declared but complete {sorted(declared - set(absent))}")
    custody = dict(attempt_dir=str(adir), manifest_sha256=manifest_sha, code_digest=code_digest,
                   oracle_test_report_sha256=sha(rp), oracle_test_status=orep.get("status"),
                   pilot_report=str(pilot_report), pilot_report_sha256=sha(pilot_report),
                   manifest_in_committed_decisions=True, eval_files=records,
                   eval_source="cells.jsonl (run_e05 native)" if native else "evals/*.json",
                   pilot_determinations_source=det.get("source", "determinations"),
                   analyze_e05_sha256=sha(Path(__file__).resolve()), python=sys.version.split()[0])
    return design, cells, det, custody


# ============================================================================ report

def _fmt(x, signed=True):
    if x is None:
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def _ci(s):
    return "n/a" if not s or "lo" not in s else f"[{_fmt(s['lo'])}, {_fmt(s['hi'])}]"


def _p(x):
    return "n/a" if x is None else f"{x:.4g}"


def _yn(b):
    return "n/a" if b is None else ("yes" if b else "no")


def render(res):
    L = [f"# RCX-E05 confirmatory analysis: {res['decision']}", ""]
    cu = res["custody"]
    L += [f"- Attempt dir `{cu.get('attempt_dir', '-')}`; TEST_MANIFEST sha256 `{cu.get('manifest_sha256', '-')}` (in "
          f"the committed DECISIONS.md); code digest `{cu.get('code_digest', '-')}`; ORACLE_TEST_REPORT "
          f"{cu.get('oracle_test_status', '-')}.",
          f"- Bootstrap: stratified paired cluster, R = {res['resamples']}"
          + ("" if res["prereg_resamples"] else " (NOT the prereg's 10,000: not the prereg run)")
          + ", streams random.Random(derive(20261250, 'boot:<stratum>')); ties 'both'; p floor 1/R.",
          f"- Eval files: {len(cu.get('eval_files', []))}; strata sizes {res['design']['strata']}; subset "
          f"{res['design']['subset']} parents.", ""]
    det = res["pilot"]
    L += ["## Pilot determinations (PILOT FILL)", "",
          f"P1 comparator-invalid: {_yn(det['p1_comparator_invalid'])}; P2 dropped: {_yn(det['p2_dropped'])}; PG400 = "
          f"{det['pg400'] or 'none'}; Q9 failed without a bug: {_yn(det['q9_failed_no_bug'])}; k* descriptive only: "
          f"{_yn(det['kstar_descriptive'])}; R_ext: {_yn(det['r_ext'])}; cut items {det.get('cut_items') or []}. "
          f"Holm family: {res['holm_family'] or 'empty'}.", ""]
    L += ["## Decision (§7.2)", "", f"**{res['decision']}**: " + "; ".join(res["reasons"]) + ".", ""]
    if res["notes"]:
        L += ["Failed ADVANCE conditions and notes" + (" (REVISE names these):" if res["decision"] == REVISE else ":")]
        L += [f"- {n}" for n in res["notes"]] + [""]
    P1, P2, v = res["P1"], res["P2"], res["validity"]
    L += ["## Confirmatory tests (§6.2)", "", "| Test | Point | 95% CI | raw p | Holm p | passes | valid | counts |",
          "|---|---|---|---|---|---|---|---|"]
    for name, s, p in (("P1-M (macro D1)", P1["macro"], P1["p_macro"]), ("P1-ID (H-ID D1)", P1["hid"], P1["p_hid"])):
        L.append(f"| {name} | {_fmt(s['point'] if s else None)} | {_ci(s)} | {_p(p)} | | | | |")
    L.append(f"| **P1** (IUT, p = max) | | | {_p(P1['p'])} | {_p(P1['p_holm'])} | {_yn(P1['passes'])} | "
             f"{_yn(v['p1_valid'])} | {_yn(v['p1_counts'])} |")
    s2 = P2["macro"]
    L.append(f"| **P2** (NI, D2 macro) | {_fmt(s2['point'] if s2 else None)} | {_ci(s2)} | {_p(P2['p'])} | "
             f"{_p(P2['p_holm'])} | {_yn(P2['passes'])} | {_yn(v['p2_valid'])} | {_yn(v['p2_counts'])} |")
    L += ["", f"P1 conditions: {P1['conditions']}; per replicate D1: {P1['replicates']}.",
          f"Validity (bar {BAR}): UVFA TEST H-ID nreg {_fmt(v['uvfa_test_hid'], 0)}; PG400 ({v['pg400']}) TEST primary "
          f"macro nreg {_fmt(v['pg400_test_primary_macro'], 0)}"
          + (f"; PG400 per TEST family { {f: round(x, 4) for f, x in v['pg400_test_families'].items() if x is not None} }"
             if v["pg400_test_families"] else "")
          + (f"; PG400 DEV-OOD (pilot) {v['pg400_dev_ood']}" if v["pg400_dev_ood"] else "") + ".", ""]
    g = res["guard"]
    L += ["## Guard, floor and ceiling (§6.3, §7.1)", "",
          f"- D_G = SF-GPI - ENUM-MIS± (macro): {_fmt(g['D_G']['point'] if g['D_G'] else None)} {_ci(g['D_G'])}; G1 "
          f"{_yn(g['G1'])}; STOP-D condition {_yn(g['stop_d_condition'])}"
          + (f" ({g['statement_suffix']})" if g["statement_suffix"] else "") + ".",
          f"- SF floor (SF-GPI <= 0.50 on H-ID, H-K, H-NEG): {_yn(res['sf_floor']['holds'])} "
          f"{ {k: round(x, 4) if x is not None else None for k, x in res['sf_floor']['family_means'].items()} }.",
          f"- EXACT-GPI(L) primary macro nreg: {_fmt(res['exact_gpi_primary_macro'], 0)}.", ""]
    q = res["qualifiers"]
    L += ["## Qualifiers (wording only)", "",
          f"- Naming: {q['naming']}.", f"- Attribution: {_yn(q['attribution']['holds'])}"
          + (f" ({q['attribution']['wording']})" if q["attribution"]["holds"] else "") + ".",
          f"- GPI wording: {q['gpi']['wording']}.", f"- Directions: {q['new_directions']['wording']}.",
          f"- Guard quality: {q['guard_quality']['wording'] or 'no qualifier'}.",
          f"- k*: {'descriptive only' if q['kstar']['descriptive_only'] else 'decision-driving'}.", ""]
    fams = [f for f in TEST_ORDER if any(f in m for m in res["arm_means"].values())]
    L += ["## Mean nreg per family (parent level; descriptive)", "",
          "| Arm | " + " | ".join(fams) + " | H-ID-S | H-ID-U | primary macro |",
          "|---|" + "---|" * (len(fams) + 3)]
    for a, m in res["arm_means"].items():
        L.append(f"| {a} | " + " | ".join(_fmt(m.get(f), 0) for f in fams + ["H-ID-S", "H-ID-U", "primary_macro"])
                 + " |")
    L += ["", "## Contrasts (D = nreg(X) - nreg(Y); stratified bootstrap)", ""]
    for label, c in res["contrasts"].items():
        if not c:
            continue
        L += [f"**{label}**", "", "| Where | n | point | 95% CI | p (two-sided) |" + (" p (NI) |" if "D2" in label
                                                                                     else ""),
              "|---|---|---|---|---|" + ("---|" if "D2" in label else "")]
        rows = [("macro", c["macro"])] + list(c["families"].items()) + [(h, s) for h, s in c["strata"].items()
                                                                         if h.startswith("H-ID-")]
        for where, s in rows:
            if s:
                tag = (D1_LABELS.get(where) if label.startswith("D1") else None) or FAMILY_LABELS.get(where)
                L.append(f"| {where}{f' ({tag})' if tag else ''} | {s['n']} | {_fmt(s['point'])} | {_ci(s)} | "
                         f"{_p(s.get('p'))} |" + (f" {_p(s.get('p_ni'))} |" if "D2" in label else ""))
        L.append("")
    L += ["## Per replicate (points; conditional inference on the 3 replicates, L3)", ""]
    for name, r in res["per_replicate"].items():
        L.append(f"- {name}: " + "; ".join(f"{k}: macro {_fmt(e.get('macro'))}, H-ID {_fmt(e.get('H-ID'))}"
                                            for k, e in r.items()))
    L += ["", "## k* (48-parent subset)", ""]
    for name, k in res["kstar"].items():
        if isinstance(k, dict):
            L.append(f"- {name}: k* = {k['kstar']} (95% CI {k['lo']} to {k['hi']}); curve "
                     f"{[round(x, 4) for x in k['curve']]} at {k['ks']} vs "
                     f"SF-GPI {_fmt(k['sf_gpi'], 0)}" + (" (descriptive only)" if res["kstar"]["descriptive_only"]
                                                        else ""))
    L += ["", "## Binary view (optimal on >= half of the starts; exact McNemar, primary parents)", ""]
    for name, b in res["binary_view"].items():
        if b:
            L.append(f"- {name}: n = {b['n']}, only {name.split(' vs ')[1]} {b['only_y']}, only "
                     f"{name.split(' vs ')[0]} {b['only_x']}, p = {_p(b['p'])}")
    if res["extras"]:
        L += ["", "## Other per-parent fields (primary macro)", ""]
        for a, flds in res["extras"].items():
            L.append(f"- {a}: " + "; ".join(f"{fld} {_fmt(m.get('primary_macro'), 0)}" for fld, m in flds.items()))
    if res["enum_all_tied"]:
        L += ["", "## ENUM± all-tied-decision shares (decisions with >= 2 legal moves)", ""]
        for a, fams in res["enum_all_tied"].items():
            L.append(f"- {a}: " + "; ".join(f"{fm} {_fmt(t['share'], 0)} ({t['all_tied']:.0f}/{t['decisions']:.0f})"
                                            for fm, t in fams.items()))
    L += ["", "## Missing and defects", "", f"- Missing protected-core groups: {res['missing'] or 'none'}",
          f"- Defects: {res['defects'] or 'none'}", ""]
    return "\n".join(L) + "\n"


# ============================================================================ CLI

def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:  # create-only: the analysis runs once per attempt
        f.write(text)


def output_paths(adir):
    return tuple(Path(adir) / name for name in OUTPUTS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RCX-E05 confirmatory analysis (PREREG_E05 §6-7; run once per attempt)")
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--manifest-sha", required=True, help="T4: the committed TEST_MANIFEST.json sha256")
    ap.add_argument("--code-digest", required=True, help="T3: the full code digest")
    ap.add_argument("--confirm-dir", default=str(ROOT / "runs" / "RCX-E05" / "confirm"))
    ap.add_argument("--pilot-report", default=str(ROOT / "runs" / "RCX-E05" / "pilot" / "PILOT_REPORT.json"))
    ap.add_argument("--repo-root", default=str(ROOT), help="directory holding DECISIONS.md (read at git HEAD)")
    ap.add_argument("--missing", default="", help="comma-separated protected-core groups declared missing (§10.4)")
    ap.add_argument("--defect", action="append", default=[], help="a defect found after the first learned-arm TEST "
                    "computation (§10.4); repeatable")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    adir = Path(a.confirm_dir) / f"attempt-{a.attempt}"
    if not adir.is_dir():
        refuse(f"{adir} does not exist")
    marker, jout, mout = output_paths(adir)
    existing = [p.name for p in (marker, jout, mout) if p.exists()]
    if existing:
        refuse(f"{existing} exist in {adir}: the analysis runs once per attempt (a marker without outputs is a crashed "
               "analysis: a defect)")
    missing = [t.strip() for t in a.missing.split(",") if t.strip()]
    defects = [d.strip() for d in a.defect if d.strip()]
    design, cells, det, custody = load_attempt(adir, a.manifest_sha, a.code_digest, Path(a.pilot_report),
                                               Path(a.repo_root), missing, attempt=a.attempt)
    _write_new(marker, json.dumps(dict(started_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), pid=os.getpid(),
                                       argv=list(sys.argv[1:] if argv is None else argv), missing=missing,
                                       defects=defects, manifest_sha256=a.manifest_sha, code_digest=a.code_digest,
                                       analyze_e05_sha256=custody["analyze_e05_sha256"]), indent=1) + "\n")
    res = analyze(design, cells, det, defects=defects, resamples=RESAMPLES, custody=custody)
    txt = render(res)
    _write_new(jout, json.dumps(_jsonable(res), indent=1, ensure_ascii=False) + "\n")
    _write_new(mout, txt)
    print(txt)
    print(f"DECISION: {res['decision']}  (written {jout} and {mout})")
    return res


if __name__ == "__main__":
    main()
