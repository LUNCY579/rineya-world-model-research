"""PREREG-LB analysis (docs/ledger_beam/PREREG_LB_DRAFT.md sections 5-8). CPU only, numpy/json; never loads a model and
never imports torch (refused in the registered path).

  python -B scripts/lbeam_analysis.py --dry-run   # statistics unit tests; the wording self-test (EVERY branch of the
                                                  # decision table through the REAL decide() / wording() on synthetic
                                                  # paired outcomes); void-twin exposure and refused-key flag unit tests;
                                                  # -> docs/ledger_beam/LB_DRYRUN.{json,md} (refused once frozen)
  python -B scripts/lbeam_analysis.py             # the registered analysis of runs/lbeam: refuses unless the PREREG-LB
                                                  # freeze verifies AND the run is finished (DONE.json, or a STOPPED.json
                                                  # the runner marked verdict-eligible) with A0, G1 and C1 recorded and
                                                  # every recorded output hash-verified; writes
                                                  # runs/lbeam/analysis/PREREG_LB_RESULTS.{json,md} as one create-only set

Registered decision rules (verbatim in PREREG_LB_DRAFT.md section 5; thresholds fixed before any confirmation output):
  Success      the ONE submitted plan (plan_beam: the verified plan, else the best node's prefix) reaches the goal
               exactly when executed from init under env.step (lbeam_problems.execute). Unverified submissions count by
               the same rule.
  Co-primary 1 T2 vs OFF, enum W2 E12, paired per instance over ALL N instances: exact two-sided McNemar on the
               discordant pairs (b = T2 only, c = OFF only), p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2).
  Co-primary 2 the same contrast on the needs-lookahead stratum: instances that greedy one-step descent on the TRUE
               unmet hook does not solve (lbeam_problems.greedy_solvable: at most D_MAX = 6 times, move to the first
               successor in vocabulary order with minimal true unmet; a void twin is a true no-op, so it never moves
               uphill: it moves sideways to an equal-unmet state earlier in vocabulary order, or stays put).
  Holm         across the two co-primaries.
  Status       per co-primary, with delta = (b - c) / n:
                 PASS      Holm-adjusted p < 0.01 and delta >= +0.20
                 SHORT     Holm-adjusted p < 0.01, b > c, delta < +0.20
                 REVERSED  Holm-adjusted p < 0.01, c > b
                 NS        Holm-adjusted p >= 0.01
  Key secondary T2/ORACLE success ratio (enum W2 E12, all instances) with a paired percentile bootstrap 95% CI over
               instances (unstratified; 10,000 resamples, numpy default_rng(17)); NEAR iff ratio >= 0.80 and lower
               bound >= 0.70.
  Branch       first match: MIXED_PASS (PASS, REVERSED); MIXED_SHORT (SHORT, REVERSED); MIXED_LOOKAHEAD_BETTER
               (REVERSED, PASS or SHORT); REVERSED (REVERSED, NS or REVERSED); REVERSED_LOOKAHEAD (NS, REVERSED);
               YES_NEAR (PASS, PASS, NEAR); YES (PASS, PASS); GREEDY_ONLY_NEAR (PASS, SHORT or NS, NEAR); GREEDY_ONLY
               (PASS, SHORT or NS); LOOKAHEAD_ONLY (SHORT or NS, PASS); DIRECTIONAL (any SHORT); NO.
  Evaluable    the verdict needs only A0 (first run passed), G1 and C1; report-only stages that are not recorded
               (abandoned after two failed attempts, or never run because the run ended by a deterministic stop) are
               rendered 'not evaluable'.
Report-only (never change the branch): IGNORE_STATUS, select_fixed (T2 / OFF / ORACLE / IGNORE_STATUS), the enum
W4 E24 point (T2, ORACLE), SurfaceAuditF heldout_combined (T2), T2/ORACLE on the needs-lookahead stratum, T2 vs
IGNORE_STATUS, co-primary discordant pairs by d*, verified-plan precision, optimal-length rate, unique F keys and tokens
per instance, refused keys per arm (flag: T2 > 1% of its guarded keys), success by d*, void-twin exposure, the
instrument and deviation record.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import lbeam_problems as P  # noqa: E402  (torch-free)
import lbeam_f as LF  # noqa: E402  (torch-free at import)
import lbeam_run as LR  # noqa: E402  (torch-free at import: stage files, freeze verifier)

# ---------------------------------------------------------------- registered constants (PREREG_LB_DRAFT.md section 5)
DELTA_MIN = 0.20                 # BWM: 'T2 enables planning' requires T2 - OFF >= +0.20 ...
ALPHA_READING = 0.01             # ... with Holm-adjusted p < 0.01
RATIO_MIN, RATIO_LB_MIN = 0.80, 0.70   # BWM: 'near-oracle planning'
BOOT_B, BOOT_SEED = 10000, 17
REFUSED_FLAG_SHARE = 0.01        # RCWM amendment (b): report flag when the guard refuses > 1% of T2's guarded keys
AUDIT_DROP_TOL, AUDIT_ALPHA = 0.05, 0.05   # report-only D-35 audit reading (equivalence margin, significance level)
OUT_NAMES = ("PREREG_LB_RESULTS.json", "PREREG_LB_RESULTS.md")
CP_STATUSES = ("PASS", "SHORT", "NS", "REVERSED")
BRANCHES = ("MIXED_PASS", "MIXED_SHORT", "MIXED_LOOKAHEAD_BETTER", "REVERSED", "REVERSED_LOOKAHEAD", "YES_NEAR", "YES",
            "GREEDY_ONLY_NEAR", "GREEDY_ONLY", "LOOKAHEAD_ONLY", "DIRECTIONAL", "NO")

STATUS_TEXT = {
    "PASS": "passes the pre-set bar (T2 - OFF >= +0.20 and Holm-adjusted p < 0.01)",
    "SHORT": "significant in T2's favour (Holm-adjusted p < 0.01) but below the pre-set +0.20",
    "NS": "not significant (Holm-adjusted p >= 0.01)",
    "REVERSED": "significant in OFF's favour (Holm-adjusted p < 0.01)",
}
HEADLINE = {
    "MIXED_PASS": ("MIXED: T2 ENABLES PLANNING OVER ALL INSTANCES BUT PLANS SIGNIFICANTLY WORSE THAN OFF WHERE LOOKAHEAD "
                   "IS NEEDED (on its training surface)"),
    "MIXED_SHORT": ("MIXED: T2 PLANS BETTER THAN OFF OVER ALL INSTANCES (BELOW THE PRE-SET BAR) BUT SIGNIFICANTLY WORSE "
                    "WHERE LOOKAHEAD IS NEEDED"),
    "MIXED_LOOKAHEAD_BETTER": ("MIXED: T2 PLANS SIGNIFICANTLY WORSE THAN OFF OVER ALL INSTANCES BUT BETTER WHERE "
                               "LOOKAHEAD IS NEEDED"),
    "REVERSED": "NO (REVERSED): T2 PLANS SIGNIFICANTLY WORSE THAN OFF OVER ALL INSTANCES",
    "REVERSED_LOOKAHEAD": ("NO (REVERSED WHERE LOOKAHEAD IS NEEDED): T2 PLANS SIGNIFICANTLY WORSE THAN OFF WHERE "
                           "LOOKAHEAD IS NEEDED"),
    "YES_NEAR": "T2 ENABLES NEAR-ORACLE PLANNING (T2/ORACLE over all instances; on its training surface)",
    "YES": "T2 ENABLES PLANNING, BELOW THE NEAR-ORACLE BAR (on its training surface)",
    "GREEDY_ONLY_NEAR": ("T2 ENABLES NEAR-ORACLE PLANNING OVERALL (T2/ORACLE over all instances), BUT ITS GAIN OVER OFF IS "
                         "NOT SHOWN WHERE LOOKAHEAD IS NEEDED (on its training surface)"),
    "GREEDY_ONLY": ("T2 ENABLES PLANNING ON INSTANCES GREEDY DESCENT ALREADY SOLVES; NOT SHOWN WHERE LOOKAHEAD IS "
                    "NEEDED (on its training surface)"),
    "LOOKAHEAD_ONLY": ("PARTIAL: T2 ENABLES PLANNING WHERE LOOKAHEAD IS NEEDED, BUT ITS GAIN OVER ALL INSTANCES IS NOT "
                       "SHOWN AT THE PRE-SET BAR"),
    "DIRECTIONAL": "PARTIAL: T2 PLANS BETTER THAN OFF, BELOW THE PRE-SET BAR",
    "NO": "NO: T2 IS NOT SHOWN TO PLAN BETTER THAN OFF",
}
READING = {
    "MIXED_PASS": "enables planning, qualified: significantly worse than OFF where lookahead is needed",
    "MIXED_SHORT": "partial (mixed)",
    "MIXED_LOOKAHEAD_BETTER": "partial (mixed)",
    "REVERSED": "no",
    "REVERSED_LOOKAHEAD": "no",
    "YES_NEAR": "near-oracle planning",
    "YES": "enables planning",
    "GREEDY_ONLY_NEAR": "near-oracle planning, qualified: not shown where lookahead is needed",
    "GREEDY_ONLY": "enables planning, qualified: not shown where lookahead is needed",
    "LOOKAHEAD_ONLY": "partial",
    "DIRECTIONAL": "partial",
    "NO": "no",
}
CONCLUSION = {
    "MIXED_PASS": ("Over all instances T2 passes the pre-set bar (co-primary 1: 'T2 enables planning'), but where "
                   "lookahead is needed OFF plans significantly better than T2: the gain over OFF is carried by the "
                   "instances greedy descent already solves and reverses where lookahead is needed."),
    "MIXED_SHORT": ("Over all instances T2 plans significantly better than OFF but below the pre-set +0.20, and where "
                    "lookahead is needed OFF plans significantly better than T2: 'T2 enables planning' is not shown, and "
                    "the gain reverses where lookahead is needed."),
    "MIXED_LOOKAHEAD_BETTER": ("Over all instances OFF plans significantly better than T2, but where lookahead is needed "
                               "T2 plans better than OFF (co-primary 2: {status2}): 'T2 enables planning' is not shown, "
                               "and T2's advantage where lookahead is needed is outweighed by its losses on the instances "
                               "greedy descent already solves."),
    "REVERSED": ("OFF plans significantly better than T2 {where}: the adapter's one-step gain does not turn into "
                 "goal-reaching plans here, and 'T2 enables planning' is not shown."),
    "REVERSED_LOOKAHEAD": ("Where lookahead is needed OFF plans significantly better than T2, and over all instances "
                           "T2's gain is {status1}: 'T2 enables planning' is not shown."),
    "YES_NEAR": ("T2's one-step prediction turns into goal-reaching plans: T2 enables planning, including on the "
                 "instances greedy descent on the true hook cannot solve, and reaches the near-oracle bar against the "
                 "exact simulator over all instances (the T2/ORACLE ratio on the needs-lookahead instances is reported "
                 "separately)."),
    "YES": ("T2 enables planning, including on the instances greedy descent on the true hook cannot solve, but it stays "
            "below the near-oracle bar (T2/ORACLE >= 0.80 with lower bound >= 0.70)."),
    "GREEDY_ONLY_NEAR": ("T2 enables planning and reaches the near-oracle bar over all instances (T2/ORACLE over all "
                         "instances), but on the instances that need lookahead its gain over OFF is {status2}, so the "
                         "planning gain at the pre-set bar is not shown where lookahead is needed."),
    "GREEDY_ONLY": ("T2 enables planning on instances greedy descent already solves; not shown where lookahead is "
                    "needed (the needs-lookahead contrast is {status2})."),
    "LOOKAHEAD_ONLY": ("On the instances that need lookahead T2 passes the pre-set bar, but over all instances its gain "
                       "is {status1}, so 'T2 enables planning' (co-primary 1) is not shown."),
    "DIRECTIONAL": ("T2 plans better than OFF ({which} significant at Holm-adjusted p < 0.01), but no co-primary "
                    "reaches the pre-set +0.20, so 'T2 enables planning' is not shown."),
    "NO": "Neither co-primary is significant at Holm-adjusted p < 0.01: T2 is not shown to plan better than OFF.",
}
WHERE_REVERSED = {"NS": "over all instances", "REVERSED": "over all instances and where lookahead is needed"}
NEAR_ONLY_CLAUSE = ("T2/ORACLE {ratio} (95% CI [{lo}, {hi}]) meets the near-oracle bar, but 'near-oracle planning' also "
                    "requires co-primary 1 ('T2 enables planning'), which is not met.")
NEAR_WITHHELD_CLAUSE = ("T2/ORACLE {ratio} (95% CI [{lo}, {hi}]) meets the near-oracle bar over all instances, but "
                        "'near-oracle planning' is not read because OFF plans significantly better than T2 where "
                        "lookahead is needed.")
NEAR_NOT_EVALUABLE = "The T2/ORACLE ratio is not evaluable (ORACLE success 0)."
REFUSED_FLAG_TEXT = ("Flag (report-only): the input-range guard refused {share} of T2's distinct guarded keys at enum W2 "
                     "E12 (> {thr}); T2's own arithmetic errors carried its imagined states beyond |b| <= 98 often "
                     "enough that the guard, not F, ended part of its search ({n_inst} instances with a refused key). "
                     "Every truly reachable state is in range, so each refusal also pruned a branch whose imagined "
                     "state is impossible under the true dynamics, a signal an F-only planner would not have.")
NOT_EVALUABLE_TEXT = ("Not evaluable (report-only stages that are not recorded; the branch needs only A0, G1 and C1): "
                      "{stages}.")
NOT_EVALUABLE_REASON = {
    "abandoned": "abandoned after {n} failed attempts (registered rule)",
    "stopped": "not run: the run ended by a deterministic stop after A0, G1 and C1 were recorded",
}
LA_RATIO_TEXT = ("Report-only: T2/ORACLE on the {n} needs-lookahead instances {ratio} (95% paired bootstrap CI [{lo}, "
                 "{hi}], {B} resamples, seed {seed}); T2 {t2}, ORACLE {orc}. It never changes the branch.")
LA_RATIO_NOT_EVALUABLE = "Report-only: T2/ORACLE on the needs-lookahead instances is not evaluable (ORACLE success 0 there)."
IGN_TEXT = ("Report-only: T2 vs IGNORE_STATUS (exact arithmetic without the status rule; enum W2 E12): T2 {a}/{n} vs "
            "IGNORE_STATUS {b}/{n} (difference {d}, T2 only {bb}, IGNORE_STATUS only {cc}, exact McNemar p {p}); this "
            "bears on whether T2's status-rule competence, not its arithmetic alone, carries its planning.")
AUDIT_TEXT = {
    "dependent": "surface-dependent (a significant drop off the training surface)",
    "invariant": "surface-invariant within 0.05 (the 95% CI lower bound of the difference is >= -0.05)",
    "inconclusive": "inconclusive (no significant drop, but a drop larger than 0.05 is not excluded)",
}
CP_NAMES = ("Co-primary 1 (all {n} instances)", "Co-primary 2 (the {n} needs-lookahead instances)")
CP_LINE = ("{name}: T2 {k_t2}/{n} ({r_t2}) vs OFF {k_off}/{n} ({r_off}); T2 - OFF {delta} (T2 only {b}, OFF only {c}); "
           "exact McNemar p {p}, Holm-adjusted p {p_holm}: {status}.")
KEY_LINE = ("Key secondary: T2/ORACLE success ratio {ratio} ({ci}); T2 {t2}, ORACLE {orc}; near-oracle bar (>= 0.80, "
            "lower bound >= 0.70) {met}.")
KEY_CI = "95% paired bootstrap CI [{lo}, {hi}], {B} resamples, seed {seed}"
KEY_CI_NONE = "interval not computed"
MET_TEXT = {True: "met", False: "not met"}
W4_TEXT = "Report-only: enum W4 E24 point, T2 {t2} vs ORACLE {orc} (T2/ORACLE {ratio})."
SF_TEXT = ("Report-only: select_fixed (PREREG-2-style selection among the K = 8 stored candidates) success T2 {t2}, OFF "
           "{off}, ORACLE {orc}, IGNORE_STATUS {ign}; IGNORE_STATUS beam (enum W2 E12) {ignb}.")
NOT_EVALUABLE_WORD = "not evaluable"
AUDIT_LINE = ("D-35 audit (report-only): T2 under the held-out combined rendering (reversed keys, word account names, "
              "'void' rendered 'cancelled') {a}/{n} vs the canonical frame {b}/{n} (difference {d}, 95% paired bootstrap "
              "CI [{lo}, {hi}], exact McNemar p {p}): {reading}.")
LIMITS = (
    "Training surface only (trigger-level, PREREG-3): every prompt F answers is Titan's canonical one-step ledger frame "
    "(accounts A-D, statuses posted / void, compact sorted-key JSON). PREREG-3 found T2's void no-op to be a token cue "
    "and its accuracy to drop off this surface, so the result says nothing about held-out renderings (the "
    "SurfaceAuditF heldout_combined arm is report-only).",
    "Problem distribution: generator v2 in-range instances only. Every state reachable within D_CLOSE = 8 vocabulary "
    "steps has |b| <= 98, every amount is one of T2's 21 trained values (-9..12) and every status is posted or void, so "
    "scale actions have factor -1 or 0 only; optimal depth d* 2-6; 8 vocabulary actions per instance, including 1-2 "
    "void twins. This is a subset of PREREG-2's event distribution, fresh at payload level, not out of distribution.",
    "Planner and proposer: BWM's plan_beam (bwmn/blocks_planner.py, sha 18a5f22b...) with the model-free enumerating "
    "proposer (every vocabulary action as a one-step candidate, 0 proposer tokens), in ONE registered configuration as "
    "the primary (enum W2 E12, max_depth 7). The W4 E24 point, select_fixed and IGNORE_STATUS are report-only. "
    "Single-plan setting: no execution feedback while planning.",
    "One adapter: T2 seed 0 (PREREG-2's registered artifact), unmerged on frozen FP32 Titan-Sparse; OFF is the same "
    "PeftModel with the adapter disabled. No other seed, base model, proposer or planner configuration is tested.",
    "F is the first computed answer per (state, action) key, cached per arm, behind the registered input-range guard "
    "(|b| <= 98, trained amounts, posted / void), identical for every arm; a refused key ends its candidate. Every truly "
    "reachable state is in range, so a refusal also prunes a branch whose imagined state is impossible (dev simulation: "
    "about 0.02% of a T2-like F's keys and 0.12% of an OFF-like F's).",
    "Success is exact goal attainment of the one submitted plan (the verified plan, else the best node's prefix) under "
    "env.step. The needs-lookahead stratum is defined by greedy one-step descent on the true unmet hook; it is a "
    "property of the instance, not of any arm.",
    "'Enables planning' is measured against the unadapted base model (OFF), whose F is close to useless on this task "
    "(dev simulation: success 0.12, needs-lookahead 0.04). On its own it does not show that near-oracle prediction "
    "gives near-oracle planning; that reading rests on the T2/ORACLE ratio.",
)


def sha256(p: Path) -> str:
    return LR.sha256(p)


def die(msg: str, errs=None) -> None:
    errs = errs if isinstance(errs, list) else ([str(errs)] if errs else [])
    raise SystemExit(f"lbeam_analysis: {msg}" + ("\n  " + "\n  ".join(errs[:12]) if errs else ""))


def need(cond: bool, msg: str) -> None:
    if not cond:
        die("integrity check failed", [msg])


def read_json(p: Path, what: str):
    if not Path(p).is_file():
        die(f"{what} is missing", [str(p)])
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except ValueError as e:
        die(f"{what} is unreadable", [f"{p}: {e}"])


def load_jsonl(path: Path) -> list:
    out = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            die(f"{path} is not valid JSONL", [f"line {ln + 1}: {e}"])
    return out


def r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


# ---------------------------------------------------------------- statistics
_MC: dict = {}


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar: p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2)."""
    key = (int(b), int(c)) if b <= c else (int(c), int(b))
    if key not in _MC:
        n = key[0] + key[1]
        _MC[key] = 1.0 if n == 0 else float(min(1.0, 2 * sum(math.comb(n, i) for i in range(key[0] + 1)) / 2 ** n))
    return _MC[key]


def holm(ps: dict) -> dict:
    """Monotone Holm step-down adjusted p values (identical to confirm2_analysis.holm)."""
    order = sorted(ps, key=ps.get)
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, ps[k] * (m - i)))
        out[k] = prev
    return out


def paired(t2: np.ndarray, off: np.ndarray) -> dict:
    t2, off = np.asarray(t2, bool), np.asarray(off, bool)
    n = int(t2.size)
    b, c = int((t2 & ~off).sum()), int((~t2 & off).sum())
    return {"n": n, "t2_success": int(t2.sum()), "off_success": int(off.sum()), "b_t2_only": b, "c_off_only": c,
            "t2_rate": (t2.sum() / n) if n else None, "off_rate": (off.sum() / n) if n else None,
            "delta": ((b - c) / n) if n else None, "p_raw": mcnemar_p(b, c)}


def cp_status(cp: dict) -> str:
    if cp["p_holm"] < ALPHA_READING:
        if cp["b_t2_only"] > cp["c_off_only"]:
            return "PASS" if cp["delta"] >= DELTA_MIN - 1e-12 else "SHORT"
        return "REVERSED"
    return "NS"


def boot_diff(a: np.ndarray, b: np.ndarray, B: int = BOOT_B, seed: int = BOOT_SEED, chunk: int = 2000) -> dict:
    """Paired percentile bootstrap (unstratified, over instances) of mean(a) - mean(b) (report-only audit reading)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = a.size
    rng = np.random.default_rng(seed)
    vals, done = [], 0
    while done < B:
        k = min(chunk, B - done)
        idx = rng.integers(0, n, size=(k, n))
        vals.append(a[idx].mean(1) - b[idx].mean(1))
        done += k
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"B": B, "seed": seed, "lo": float(lo), "hi": float(hi)}


def audit_reading(delta: float, p_raw: float, lo: float) -> str:
    """Report-only D-35 reading: a significant drop -> surface-dependent; else the lower 95% bound of the paired
    difference >= -0.05 -> surface-invariant within 0.05 (an equivalence reading); else inconclusive."""
    if p_raw < AUDIT_ALPHA and delta < 0:
        return AUDIT_TEXT["dependent"]
    if lo >= -AUDIT_DROP_TOL - 1e-12:
        return AUDIT_TEXT["invariant"]
    return AUDIT_TEXT["inconclusive"]


def boot_ratio(t2: np.ndarray, orc: np.ndarray, B: int = BOOT_B, seed: int = BOOT_SEED, chunk: int = 2000) -> dict:
    """Paired percentile bootstrap over instances (unstratified) of mean(T2) / mean(ORACLE); resamples whose ORACLE
    mean is 0 are dropped (counted)."""
    t2, orc = np.asarray(t2, np.float64), np.asarray(orc, np.float64)
    n = t2.size
    rng = np.random.default_rng(seed)
    vals, dropped, done = [], 0, 0
    while done < B:
        k = min(chunk, B - done)
        idx = rng.integers(0, n, size=(k, n))
        a, o = t2[idx].mean(1), orc[idx].mean(1)
        ok = o > 0
        dropped += int((~ok).sum())
        vals.append(a[ok] / o[ok])
        done += k
    v = np.concatenate(vals) if vals else np.array([])
    lo, hi = (np.percentile(v, [2.5, 97.5]) if v.size else (float("nan"), float("nan")))
    return {"B": B, "seed": seed, "lo": float(lo), "hi": float(hi), "dropped_zero_oracle": dropped}


def decide(t2, off, orc, la_mask, B: int = BOOT_B, seed: int = BOOT_SEED, lazy_boot: bool = False) -> dict:
    """The registered decision: co-primaries, Holm, statuses, the ratio and its CI, NEAR, the branch.
    lazy_boot (power simulations only): skip the bootstrap when the point ratio is below 0.80 (NEAR is then False
    whatever the interval)."""
    t2, off, orc, la = (np.asarray(x, bool) for x in (t2, off, orc, la_mask))
    cp1, cp2 = paired(t2, off), paired(t2[la], off[la])
    adj = holm({"cp1": cp1["p_raw"], "cp2": cp2["p_raw"]})
    cp1["p_holm"], cp2["p_holm"] = adj["cp1"], adj["cp2"]
    s1, s2 = cp_status(cp1), cp_status(cp2)
    o_rate = float(orc.mean()) if orc.size else 0.0
    ratio = {"t2": float(t2.mean()), "oracle": o_rate, "ratio": (float(t2.mean()) / o_rate) if o_rate > 0 else None}
    near = False
    if ratio["ratio"] is not None:
        if lazy_boot and ratio["ratio"] < RATIO_MIN:
            ratio.update({"lo": None, "hi": None, "B": 0})
        else:
            ratio.update(boot_ratio(t2, orc, B, seed))
            near = ratio["ratio"] >= RATIO_MIN - 1e-12 and ratio["lo"] >= RATIO_LB_MIN - 1e-12
    ratio["near"] = near
    ratio["evaluable"] = ratio["ratio"] is not None
    br = branch(s1, s2, near)
    return {"cp1": cp1, "cp2": cp2, "status1": s1, "status2": s2, "ratio": ratio, "near": near, "branch": br,
            "reading": READING[br], "headline": HEADLINE[br]}


def branch(s1: str, s2: str, near: bool) -> str:
    """First match (PREREG_LB_DRAFT.md section 5). Mixed-sign outcomes (one co-primary significant in T2's favour, the
    other in OFF's) get their own branches, so no rendered sentence contradicts a co-primary."""
    if s1 == "PASS" and s2 == "REVERSED":
        return "MIXED_PASS"
    if s1 == "SHORT" and s2 == "REVERSED":
        return "MIXED_SHORT"
    if s1 == "REVERSED" and s2 in ("PASS", "SHORT"):
        return "MIXED_LOOKAHEAD_BETTER"
    if s1 == "REVERSED":                               # status 2 NS or REVERSED
        return "REVERSED"
    if s2 == "REVERSED":                               # status 1 NS
        return "REVERSED_LOOKAHEAD"
    if s1 == "PASS" and s2 == "PASS":
        return "YES_NEAR" if near else "YES"
    if s1 == "PASS":
        return "GREEDY_ONLY_NEAR" if near else "GREEDY_ONLY"
    if s2 == "PASS":
        return "LOOKAHEAD_ONLY"
    if "SHORT" in (s1, s2):
        return "DIRECTIONAL"
    return "NO"


# ---------------------------------------------------------------- wording (mechanical)
def _cp_line(name: str, cp: dict, status: str) -> str:
    return CP_LINE.format(name=name, k_t2=cp["t2_success"], n=cp["n"], r_t2=r3(cp["t2_rate"]), k_off=cp["off_success"],
                          r_off=r3(cp["off_rate"]), delta=("n/a" if cp["delta"] is None else f"{cp['delta']:+.3f}"),
                          b=cp["b_t2_only"], c=cp["c_off_only"], p=fmt_p(cp["p_raw"]), p_holm=fmt_p(cp["p_holm"]),
                          status=STATUS_TEXT[status])


def wording(D: dict, extra: dict | None = None) -> dict:
    """Headline, reading, sentence and qualifiers, selected mechanically from D = decide(...)."""
    extra = extra or {}
    br, cp1, cp2, R = D["branch"], D["cp1"], D["cp2"], D["ratio"]
    n_la = cp2["n"]
    lines = [_cp_line(CP_NAMES[0].format(n=cp1["n"]), cp1, D["status1"]),
             _cp_line(CP_NAMES[1].format(n=n_la), cp2, D["status2"])]
    if R["evaluable"]:
        ci = (KEY_CI.format(lo=r3(R["lo"]), hi=r3(R["hi"]), B=R["B"], seed=R["seed"]) if R.get("lo") is not None
              else KEY_CI_NONE)
        lines.append(KEY_LINE.format(ratio=r3(R["ratio"]), ci=ci, t2=r3(R["t2"]), orc=r3(R["oracle"]),
                                     met=MET_TEXT[bool(D["near"])]))
    else:
        lines.append(NEAR_NOT_EVALUABLE)
    where = WHERE_REVERSED.get(D["status2"], "")
    which = " and ".join(n for n, s in (("co-primary 1", D["status1"]), ("co-primary 2", D["status2"])) if s == "SHORT")
    concl = CONCLUSION[br].format(where=where, status1=STATUS_TEXT[D["status1"]], status2=STATUS_TEXT[D["status2"]],
                                  which=which)
    quals = []
    if D["near"] and D["status1"] != "PASS":           # BWM: 'near-oracle planning' needs co-primary 1 first
        quals.append(NEAR_ONLY_CLAUSE.format(ratio=r3(R["ratio"]), lo=r3(R["lo"]), hi=r3(R["hi"])))
    elif D["near"] and br == "MIXED_PASS":             # co-primary 1 passes but T2 is significantly worse where needed
        quals.append(NEAR_WITHHELD_CLAUSE.format(ratio=r3(R["ratio"]), lo=r3(R["lo"]), hi=r3(R["hi"])))
    if extra.get("refused_share_t2") is not None and extra["refused_share_t2"] > REFUSED_FLAG_SHARE:
        quals.append(REFUSED_FLAG_TEXT.format(share=f"{extra['refused_share_t2']:.4f}", thr=REFUSED_FLAG_SHARE,
                                              n_inst=extra.get("refused_instances_t2", "?")))
    if extra.get("not_evaluable"):
        quals.append(not_evaluable_sentence(extra["not_evaluable"]))
    for q in extra.get("qualifiers", []):
        quals.append(q)
    return {"branch": br, "headline": HEADLINE[br], "reading": READING[br], "lines": lines, "conclusion": concl,
            "qualifiers": quals, "limits": list(LIMITS)}


STAGE_WHAT = {"G2": "G2 (select_fixed T2 and OFF)", "G3": "G3 (T2 at enum W4 E24)",
              "G4": "G4 (the D-35 audit, T2 under heldout_combined)"}


def not_evaluable_sentence(ne: dict) -> str:
    """ne: {stage: {'kind': 'abandoned' | 'stopped', 'n': failed attempts}} -> the registered sentence."""
    parts = []
    for st in ("G2", "G3", "G4"):
        if st in ne:
            v = ne[st]
            parts.append(f"{STAGE_WHAT[st]}, " + NOT_EVALUABLE_REASON[v["kind"]].format(n=v.get("n", "?")))
    return NOT_EVALUABLE_TEXT.format(stages="; ".join(parts))


def registered_strings() -> list:
    """Every fixed sentence the analysis can render (the verbatim check against the draft, at the freeze and in the
    tests). Templates are checked up to their first placeholder-free stretch."""
    out = list(HEADLINE.values()) + list(STATUS_TEXT.values()) + list(LIMITS) + list(READING.values()) \
        + list(CONCLUSION.values()) + [NEAR_ONLY_CLAUSE, NEAR_WITHHELD_CLAUSE, NEAR_NOT_EVALUABLE, REFUSED_FLAG_TEXT,
                                       NOT_EVALUABLE_TEXT, LA_RATIO_TEXT, LA_RATIO_NOT_EVALUABLE, IGN_TEXT, AUDIT_LINE,
                                       CP_LINE, KEY_LINE, KEY_CI, KEY_CI_NONE, W4_TEXT, SF_TEXT, NOT_EVALUABLE_WORD] \
        + list(CP_NAMES) + list(MET_TEXT.values()) \
        + list(NOT_EVALUABLE_REASON.values()) + list(AUDIT_TEXT.values()) + list(WHERE_REVERSED.values()) \
        + list(STAGE_WHAT.values())
    return out


def registered_strings_missing(draft_text: str) -> list:
    flat = " ".join(draft_text.split())
    return [t[:80] for t in registered_strings() if " ".join(t.split()) not in flat]


# ---------------------------------------------------------------- report-only metrics
def summarise_eps(insts: list, rows: list, la_mask: np.ndarray) -> dict:
    """Per arm / config: success, by d*, by stratum, verified rate, verified-plan precision, optimal-length rate."""
    by = {i["id"]: i for i in insts}
    succ, ver, opt = [], [], []
    for r in rows:
        ex = P.execute(by[r["id"]], r["plan"])
        succ.append(bool(ex["success"]))
        ver.append(bool(r.get("verified")))
        opt.append(bool(ex["success"] and len(r["plan"]) == by[r["id"]]["opt_depth"]))
    s, v, o = np.array(succ), np.array(ver), np.array(opt)
    depth = np.array([by[r["id"]]["opt_depth"] for r in rows])
    out = {"n": len(rows), "success": r3(s.mean()), "success_k": int(s.sum()),
           "success_needs_lookahead": r3(s[la_mask].mean()) if la_mask.any() else None,
           "success_greedy": r3(s[~la_mask].mean()) if (~la_mask).any() else None,
           "verified_rate": r3(v.mean()), "verified_plan_precision": r3(s[v].mean()) if v.any() else None,
           "optimal_length_rate": r3(o.mean()),
           "success_by_depth": {str(d): r3(s[depth == d].mean()) for d in sorted(set(depth.tolist()))},
           "fail_kinds": dict(Counter(str(r.get("fail")) for r, ok in zip(rows, succ) if not ok))}
    return out, s


def f_answer_from_cache(cache_rows: list, frame: str):
    """(state, action) -> predicted next state (or None) from a cumulative cache (the first computed answer)."""
    tab = {}
    for r in cache_rows:
        k = (P._tt(r["s"]), P._tt(r["a"]))
        if k in tab:
            continue
        tab[k] = (LF.parse_answer(r["text"], k[0]) if frame == LF.CANONICAL else LF.audit_parse(r["text"], k[0], frame))
    return lambda s, a: tab.get((s, a), "MISSING")


def f_answer_exact(arm: str):
    if arm == "ORACLE":
        return lambda s, a: P.step_state(s, a)
    if arm == "IGNORE_STATUS":
        return lambda s, a: P.step_ignore_status(s, a)
    raise ValueError(arm)


def void_twin_exposure(insts: list, rows: list, success: np.ndarray, f_answer, includes_distractors: bool = False) -> dict:
    """Among FAILED instances: the share whose submitted plan contains a void action that this arm's F applied as if
    posted (F's predicted next state == the posted application != the input state), read along the plan's own imagined
    trajectory under F. Also: void steps F did not treat as a no-op at all (predicted state != input state, whether
    applied as posted or with an arithmetic error). includes_distractors: select_fixed submissions are chosen among
    stored candidates that include void_swap distractors (labelled, not excluded)."""
    by = {i["id"]: i for i in insts}
    n_fail = n_void = n_applied = n_notnoop = n_void_steps = n_applied_steps = n_notnoop_steps = n_missing = 0
    for r, ok in zip(rows, success):
        if ok:
            continue
        n_fail += 1
        inst = by[r["id"]]
        s = inst["init"]
        has_void = has_applied = has_notnoop = False
        for a in r["plan"]:
            ns = f_answer(s, a)
            if isinstance(ns, str):                # key not in the cache: the guard refused it (it never reached F),
                n_missing += 1                     # e.g. a select_fixed candidate whose prefix left the range
                break
            if a[4] != "posted":
                has_void = True
                n_void_steps += 1
                posted = P.step_ignore_status(s, a)
                if ns is not None and ns != s:
                    has_notnoop = True
                    n_notnoop_steps += 1
                    if ns == posted:
                        has_applied = True
                        n_applied_steps += 1
            if ns is None:
                break
            s = ns
        n_void += has_void
        n_applied += has_applied
        n_notnoop += has_notnoop
    return {"failures": n_fail, "failures_with_void_step": n_void, "failures_with_void_applied_as_posted": n_applied,
            "exposure": r3(n_applied / n_fail) if n_fail else None,
            "failures_with_void_not_noop": n_notnoop, "exposure_not_noop": r3(n_notnoop / n_fail) if n_fail else None,
            "void_steps_in_failed_plans": n_void_steps, "void_steps_applied_as_posted": n_applied_steps,
            "void_steps_not_noop": n_notnoop_steps, "missing_keys_guard_refused": n_missing,
            "includes_stored_distractors": includes_distractors}


def per_instance_stats(acct: dict, ids: list) -> dict:
    def q(xs):
        xs = sorted(xs)
        return {"mean": r3(statistics.fmean(xs)) if xs else None, "median": P._q(xs, 0.5), "p90": P._q(xs, 0.9),
                "max": max(xs) if xs else None, "total": sum(xs)}
    keys = [int((acct.get("f_keys_by_instance") or {}).get(i, 0)) for i in ids]
    toks = [int((acct.get("tokens_by_instance") or {}).get(i, 0)) for i in ids]
    ref = [int((acct.get("refused_by_instance") or {}).get(i, 0)) for i in ids]
    g = acct.get("guard") or {}
    return {"f_keys_per_instance": q(keys), "tokens_per_instance_reencoded": q(toks),
            "refused_unique": g.get("refused_unique"), "guarded_seen_unique": g.get("seen_unique"),
            "refused_share": g.get("oor_key_share"), "instances_with_refused_key": sum(x > 0 for x in ref),
            "generated_tokens_new_exact": acct.get("generated_tokens_new_exact")}


# ---------------------------------------------------------------- inputs (registered or fixture)
class Inputs:
    """Loads and validates one run directory. registered=True: the freeze must verify and every provenance check binds
    (code hashes per turn, runtime, device, adapter hashes at load, fingerprint, BWM planner, draw)."""

    def __init__(self, run: Path, draw: Path, registered: bool, root: Path | None = None):
        self.run, self.draw = Path(run), Path(draw)
        root = Path(root or ROOT)
        self.fz = None
        if registered:
            ok, errs, fz = LR.verify_freeze(root)
            if not ok:
                die("the PREREG-LB freeze does not verify; no analysis", errs)
            self.fz = fz
            need(sha256(self.draw) == fz["inputs"][LR.DRAW_REL], "the confirmation draw differs from the freeze")
        self.insts = P.load_set(self.draw)
        ids = [i["id"] for i in self.insts]
        need(len(set(ids)) == len(ids), "duplicate instance ids in the draw")
        prov = read_json(self.run / "provenance.json", "provenance.json")
        term = terminal_state(self.run)
        errs = provenance_errors(prov, term, self.run, self.fz, registered, root, self.draw)
        if errs:
            die("the run's provenance does not check", errs)
        self.prov, self.term = prov, term
        self.recorded = [st for st in LR.STAGE_ORDER if st in (prov.get("stages") or {})]
        self.not_evaluable = not_evaluable_of(prov, term)
        self.eps, self.caches, self.acct = {}, {}, {}
        errs = []
        for st in self.recorded:
            for name, rel in LR.FILES[st].items():
                p = self.run / rel
                if name.startswith(("eps:", "fixed:")):
                    rows = load_jsonl(p)
                    errs += episode_errors(rows, self.insts, rel)
                    self.eps[name] = [{**r, "plan": [tuple(a) for a in (r.get("plan") or [])]} for r in rows]
                elif name.startswith("cache:"):
                    self.caches[(st, name[6:])] = cache_rows(p, name[6:], errs, rel)
                elif name == "acct":
                    self.acct[st] = read_json(p, rel)
        errs += cache_chain_errors(self.caches, self.insts)
        if errs:
            die("an output file is incomplete or malformed", errs)


def episode_errors(rows: list, insts: list, rel: str) -> list:
    errs = []
    if [r.get("id") for r in rows] != [i["id"] for i in insts]:
        errs.append(f"{rel}: instance ids are not the draw's ids in draw order")
        return errs
    for r, i in zip(rows, insts):
        voc = set(i["vocab"])
        plan = r.get("plan")
        if not isinstance(plan, list) or not all(isinstance(a, list) and len(a) == 6 for a in plan):
            errs.append(f"{rel}: {r['id']}: malformed plan")
        elif any(tuple(a) not in voc for a in plan):
            errs.append(f"{rel}: {r['id']}: plan uses an action outside the instance vocabulary")
        if not isinstance(r.get("verified"), bool):
            errs.append(f"{rel}: {r['id']}: 'verified' is not a bool")
    return errs


def cache_rows(p: Path, key: str, errs: list, rel: str) -> list:
    lines = Path(p).read_text(encoding="utf-8").splitlines()
    if not lines:
        errs.append(f"{rel}: empty cache")
        return []
    head = json.loads(lines[0])
    arm, _, frame = key.partition("@")
    want = {"schema": LF.CACHE_SCHEMA, "class": "SurfaceAuditF" if frame else "LedgerF", "arm": arm,
            "frame": frame or LF.CANONICAL}
    if any(head.get(k) != v for k, v in want.items()):
        errs.append(f"{rel}: cache header {head} is not {want}")
    rows, seen = [], set()
    for ln, line in enumerate(lines[1:]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            errs.append(f"{rel}: line {ln + 2} is not JSON")
            continue
        k = (P._tt(r.get("s")), P._tt(r.get("a")))
        if k in seen:
            errs.append(f"{rel}: duplicate key at line {ln + 2} (F must be the FIRST computed answer)")
        seen.add(k)
        rows.append(r)
    return rows


def cache_chain_errors(caches: dict, insts: list) -> list:
    """Per arm, every later stage's cumulative cache starts with the earlier stage's entries (F = first computed
    answer, never recomputed); every key belongs to a draw instance."""
    errs = []
    own = P.owner_of_action(insts)
    by_arm: dict = {}
    for (st, key) in sorted(caches, key=lambda x: LR.STAGE_ORDER.index(x[0])):
        by_arm.setdefault(key, []).append(st)
    for key, sts in by_arm.items():
        for a, b in zip(sts, sts[1:]):
            ra, rb = caches[(a, key)], caches[(b, key)]
            if rb[:len(ra)] != ra:
                errs.append(f"cache {key}: stage {b} does not extend stage {a}'s entries")
        for st in sts:
            for r in caches[(st, key)]:
                if own.get(P._tt(r.get("a"))) is None:
                    errs.append(f"cache {key} ({st}): a key's action belongs to no draw instance")
                    break
    return errs


REPORT_ONLY_STAGES = ("G2", "G3", "G4")


def terminal_state(run: Path) -> dict:
    """The run's terminal record: DONE.json (every stage recorded or abandoned) or STOPPED.json (a deterministic
    stop; verdict-eligible only if the runner found A0, G1 and C1 recorded and every recorded output unchanged)."""
    done_p, stop_p = Path(run) / "DONE.json", Path(run) / "STOPPED.json"
    if done_p.exists() and stop_p.exists():
        return {"kind": "both", "rec": {}}
    if done_p.exists():
        return {"kind": "done", "rec": read_json(done_p, "DONE.json")}
    if stop_p.exists():
        return {"kind": "stopped", "rec": read_json(stop_p, "STOPPED.json")}
    return {"kind": None, "rec": {}}


def not_evaluable_of(prov: dict, term: dict) -> dict:
    """Report-only stages that are not recorded: abandoned (two failed attempts) or not run (the run stopped)."""
    st, ab, ne = prov.get("stages") or {}, prov.get("abandoned") or {}, {}
    for s in REPORT_ONLY_STAGES:
        if s in st:
            continue
        ne[s] = ({"kind": "abandoned", "n": ab[s].get("failed_attempts")} if s in ab else {"kind": "stopped"})
    return ne


def provenance_errors(prov: dict, term: dict, run: Path, fz: dict | None, registered: bool, root: Path,
                      draw: Path) -> list:
    errs = []
    st = prov.get("stages") or {}
    kind, rec = term.get("kind"), term.get("rec") or {}
    ab = prov.get("abandoned") or {}
    if kind is None:
        errs.append("the run is not finished (neither DONE.json nor STOPPED.json exists)")
    if kind == "both":
        errs.append("DONE.json and STOPPED.json both exist")
    if set(st) - set(LR.STAGE_ORDER):
        errs.append(f"unknown stages recorded: {sorted(set(st) - set(LR.STAGE_ORDER))}")
    miss_core = [s for s in LR.CORE_STAGES if s not in st]
    if miss_core:
        errs.append(f"the verdict needs A0, G1 and C1 recorded; not recorded: {miss_core} (no verdict)")
    if set(ab) - set(REPORT_ONLY_STAGES):
        errs.append(f"a stage other than a report-only one is recorded as abandoned: {sorted(set(ab) - set(REPORT_ONLY_STAGES))}")
    if set(ab) & set(st):
        errs.append(f"a stage is both recorded and abandoned: {sorted(set(ab) & set(st))}")
    if kind == "done":
        unrec = [s for s in LR.STAGE_ORDER if s not in st]
        if [s for s in unrec if s not in ab]:
            errs.append(f"DONE.json, but stages neither recorded nor abandoned: {[s for s in unrec if s not in ab]}")
        if set(rec.get("not_evaluable") or {}) != set(unrec):
            errs.append("DONE.json's not-evaluable stages differ from provenance")
    if kind == "stopped" and rec.get("verdict_eligible") is not True:
        errs.append("STOPPED.json is not verdict-eligible (the run stopped before A0, G1 and C1 were recorded, or a "
                    "recorded output changed): no verdict")
    pv = run / "provenance.json"
    if kind in ("done", "stopped") and rec.get("provenance_sha256") != sha256(pv):
        errs.append("provenance.json changed after the run's terminal record (DONE.json / STOPPED.json)")
    for s in LR.STAGE_ORDER:
        if s not in st:
            continue
        r_ = st[s] or {}
        for p in LR.stage_required(s, run):
            r = LR.rel_out(p, run)
            if not p.is_file():
                errs.append(f"{r} missing")
            elif (r_.get("files") or {}).get(r) != sha256(p):
                errs.append(f"{r}: hash differs from provenance")
            elif kind == "done" and (rec.get("files") or {}).get(s, {}).get(r) != sha256(p):
                errs.append(f"{r}: hash differs from DONE.json")
        if s in ("G1", "G2") and len({v.get("pid") for v in (r_.get("sides") or {}).values()} | {r_.get("pid")}) != 1:
            errs.append(f"stage {s}: T2 and OFF not generated in one process")
    if prov.get("draw_sha256") != sha256(draw):
        errs.append("the run's draw hash differs from the draw analysed")
    anc = LR._read_json(run / "anchor" / "anchor.json") or {}
    bi = LR._read_json(run / "anchor" / "batch_invariance.json") or {}
    if not anc.get("pass") or not (prov.get("anchor") or {}).get("pass"):
        errs.append("the A0 anchor did not pass")
    if not bi.get("pass") or not (prov.get("batch_invariance") or {}).get("pass"):
        errs.append("the A0 batch-invariance check did not pass")
    rf = list((run / "anchor").glob("RECHECK_FAILED_*.json"))
    if rf and kind != "stopped":
        errs.append("a failed A0 re-check is recorded but the run did not end by a stop")
    turns = prov.get("turns") or []
    a0_turn = st.get("A0", {}).get("turn")
    for i, t in enumerate(turns):
        ran = set(t.get("stages_run") or [])
        if i != a0_turn and ran & {"G1", "G2", "G3", "G4"}:
            if not (t.get("anchor_recheck") or {}).get("pass") or not (t.get("batchinv_recheck") or {}).get("pass"):
                errs.append(f"turn {i} ran planning stages without a passing A0 re-check (anchor + batch invariance)")
        if i == a0_turn and ran & {"G1", "G2", "G3", "G4"} and "A0" not in ran:
            errs.append(f"turn {i}: A0 bookkeeping inconsistent")
    for s, r_ in st.items():
        if s in ("G1", "G2", "G3", "G4") and r_.get("turn") is not None and \
                r_["turn"] < (a0_turn if a0_turn is not None else 10 ** 9):
            errs.append(f"stage {s} recorded before the A0 anchor passed")
        if s in ("G1", "G2", "G3", "G4") and r_.get("turn") is not None and r_["turn"] < len(turns) and \
                s not in (turns[r_["turn"]].get("stages_run") or []):
            errs.append(f"stage {s}: not listed in the stages_run of its turn {r_['turn']}")
    if "C1" in st and (st["C1"].get("after_a0") is not True or st["C1"].get("lease_held") is not False):
        errs.append("the CPU stage C1 ran before A0 passed or under the lease")
    if any(c.get("lease_held") is not False for c in prov.get("cpu_runs") or []):
        errs.append("a CPU run is not recorded as outside the lease")
    if any(c.get("cuda_initialized_before") is not False or c.get("cuda_initialized_after") is not False
           for c in prov.get("cpu_runs") or []):
        errs.append("a CPU run is not recorded as CUDA-free (C1 must run before any CUDA use in its process)")
    if prov.get("lease_label") != LR.LEASE_LABEL:
        errs.append(f"lease label {prov.get('lease_label')} != {LR.LEASE_LABEL}")
    if registered:
        allowed = LR.allowed_hashes(fz, root)
        for i, t in enumerate(turns):
            for f, h in (t.get("code_sha256") or {}).items():
                if f in allowed and h not in allowed[f]:
                    errs.append(f"turn {i}: {f} differs from the frozen / amended version")
            rt = t.get("runtime") or {}
            if t.get("stages_run") or t.get("anchor_recheck"):
                for k in ("torch", "transformers"):
                    if rt.get(k) != LR.REGISTERED_RUNTIME[k]:
                        errs.append(f"turn {i}: runtime {k} {rt.get(k)} != registered {LR.REGISTERED_RUNTIME[k]}")
                if (t.get("versions") or {}).get("peft") != LR.REGISTERED_RUNTIME["peft"]:
                    errs.append(f"turn {i}: peft differs from the registered runtime")
                if t.get("device") != "cuda":
                    errs.append(f"turn {i}: not generated by the registered GPU backend (device {t.get('device')})")
                for f, h in (t.get("adapter_sha256_at_load") or {}).items():
                    if h != fz["inputs"].get(f"{LR.ADAPTER_REL}/{f}"):
                        errs.append(f"turn {i}: adapter {f} at load differs from the freeze")
                if not t.get("adapter_sha256_at_load"):
                    errs.append(f"turn {i}: adapter hash at load not recorded")
            if t.get("bwm_planner_sha256") not in (None, LR.BWM_PLANNER_SHA):
                errs.append(f"turn {i}: BWM planner sha differs from the pin")
        for c in prov.get("cpu_runs") or []:
            for f, h in (c.get("code_sha256") or {}).items():
                if f in allowed and h not in allowed[f]:
                    errs.append(f"CPU run: {f} differs from the frozen / amended version")
            if c.get("bwm_planner_sha256") != LR.BWM_PLANNER_SHA:
                errs.append("CPU run: BWM planner sha differs from the pin")
        if not str(prov.get("sparse_fingerprint", "")).startswith(LR.FP["sparse"]):
            errs.append("Titan-Sparse fingerprint")
        if prov.get("registered") is not True or prov.get("freeze_prereg_sha256") != fz.get("prereg_sha256"):
            errs.append("the run was not made under this freeze (provenance 'registered' / prereg sha)")
        if prov.get("draw_sha256") != fz["inputs"].get(LR.DRAW_REL):
            errs.append("provenance draw hash differs from the freeze")
    return errs


# ---------------------------------------------------------------- instrument and deviation record (report-only)
def bwm_drift(prov: dict, fz: dict | None) -> list:
    """BWM modules imported by the planner, per turn and CPU run, against the freeze record (report-only: only
    blocks_planner.py is binding)."""
    ref = dict(((fz or {}).get("bwm") or {}).get("imported_files_recorded_not_binding") or {})
    ref[LR.BWM_PLANNER_REL] = LR.BWM_PLANNER_SHA
    out = []
    for where, recs in (("turn", prov.get("turns") or []), ("cpu_run", prov.get("cpu_runs") or [])):
        for i, t in enumerate(recs):
            for f, h in sorted((t.get("bwm_modules_sha256") or {}).items()):
                if f in ref and h != ref[f]:
                    out.append({"where": f"{where} {i}", "file": f, "sha256": h, "freeze": ref[f]})
                elif f not in ref and fz is not None:
                    out.append({"where": f"{where} {i}", "file": f, "sha256": h, "freeze": None})
    return out


def instrument_record(I) -> dict:
    prov = I.prov
    anc, bi = prov.get("anchor") or {}, prov.get("batch_invariance") or {}

    def anc_rates(a):
        return None if not a else {"pass": a.get("pass"), "off_states": r3((a.get("off") or {}).get("identical_states")),
                                   "on_states": r3((a.get("on") or {}).get("identical_states")),
                                   "off_texts": r3((a.get("off") or {}).get("identical_texts")),
                                   "on_texts": r3((a.get("on") or {}).get("identical_texts"))}

    def bi_rates(b):
        return None if not b else {"pass": b.get("pass"), **{f"{arm}_states": r3(v.get("identical_states"))
                                                             for arm, v in (b.get("arms") or {}).items()},
                                   **{f"{arm}_texts": r3(v.get("identical_texts")) for arm, v in (b.get("arms") or {}).items()}}
    turns = []
    for i, t in enumerate(prov.get("turns") or []):
        turns.append({"turn": i, "stages_run": t.get("stages_run"), "deferred": t.get("deferred"),
                      "no_progress": t.get("no_progress"), "cap_h": t.get("cap_h"), "device": t.get("device"),
                      "anchor_recheck": anc_rates(t.get("anchor_recheck")),
                      "batchinv_recheck": bi_rates(t.get("batchinv_recheck"))})
    od = I.run / "_orphans"
    return {"terminal": I.term["kind"],
            "stop_reason": (I.term.get("rec") or {}).get("reason") if I.term["kind"] == "stopped" else None,
            "anchor": anc_rates(anc), "batch_invariance": bi_rates(bi), "turns": turns,
            "orphaned_stages": prov.get("orphaned") or [],
            "orphan_dirs": sorted(p.name for p in od.iterdir()) if od.is_dir() else [],
            "attempts": prov.get("attempts") or {}, "abandoned": prov.get("abandoned") or {},
            "not_evaluable": I.not_evaluable,
            "recheck_failed_files": sorted(p.name for p in (I.run / "anchor").glob("RECHECK_FAILED_*.json")),
            "bwm_modules_drift": bwm_drift(prov, I.fz)}


# ---------------------------------------------------------------- the analysis
def discordant_by_depth(t2, off, depth, mask) -> dict:
    out = {}
    for d in range(2, 7):
        m = (depth == d) & mask
        out[str(d)] = {"n": int(m.sum()), "t2_only": int((t2 & ~off & m).sum()), "off_only": int((~t2 & off & m).sum())}
    return out


def analyse(I: Inputs) -> dict:
    insts = I.insts
    ids = [i["id"] for i in insts]
    greedy = np.array([P.greedy_solvable(i) for i in insts], bool)
    la = ~greedy
    depth = np.array([i["opt_depth"] for i in insts])
    S, succ = {}, {}
    for name, rows in I.eps.items():
        S[name], succ[name] = summarise_eps(insts, rows, la)
    t2, off, orc = succ["eps:T2:W2E12"], succ["eps:OFF:W2E12"], succ["eps:ORACLE:W2E12"]
    D = decide(t2, off, orc, la)
    # report-only
    acct = {}
    for st, a in I.acct.items():
        for cfg, v in (a.get("configs") or {}).items():
            acct[cfg] = per_instance_stats(v, ids)
    g1 = (I.acct.get("G1") or {}).get("configs", {}).get("T2:W2E12", {})
    ref_share = (g1.get("guard") or {}).get("oor_key_share")
    ref_inst = sum(1 for v in (g1.get("refused_by_instance") or {}).values() if v)
    exposure = {}
    cache_of = {"eps:T2:W2E12": ("G1", "T2"), "eps:OFF:W2E12": ("G1", "OFF"), "fixed:T2": ("G2", "T2"),
                "fixed:OFF": ("G2", "OFF"), "eps:T2:W4E24": ("G3", "T2"),
                "eps:T2@heldout_combined:W2E12": ("G4", "T2@heldout_combined")}
    for name, rows in I.eps.items():
        arm = name.split(":")[1]
        if name in cache_of:
            st, key = cache_of[name]
            frame = key.partition("@")[2] or LF.CANONICAL
            fa = f_answer_from_cache(I.caches[(st, key)], frame)
        else:
            fa = f_answer_exact(arm.split("@")[0])
        exposure[name] = void_twin_exposure(insts, rows, succ[name], fa, includes_distractors=name.startswith("fixed:"))
    quals = []
    if orc[la].sum() > 0:
        bl = boot_ratio(t2[la], orc[la])
        la_ratio = {"n": int(la.sum()), "t2": float(t2[la].mean()), "oracle": float(orc[la].mean()),
                    "ratio": float(t2[la].mean() / orc[la].mean()), **bl}
        quals.append(LA_RATIO_TEXT.format(n=la_ratio["n"], ratio=r3(la_ratio["ratio"]), lo=r3(bl["lo"]), hi=r3(bl["hi"]),
                                          B=bl["B"], seed=bl["seed"], t2=r3(la_ratio["t2"]), orc=r3(la_ratio["oracle"])))
    else:
        la_ratio = {"n": int(la.sum()), "ratio": None}
        quals.append(LA_RATIO_NOT_EVALUABLE)
    ig = paired(t2, succ["eps:IGNORE_STATUS:W2E12"])
    quals.append(IGN_TEXT.format(a=ig["t2_success"], b=ig["off_success"], n=ig["n"], d=f"{ig['delta']:+.3f}",
                                 bb=ig["b_t2_only"], cc=ig["c_off_only"], p=fmt_p(ig["p_raw"])))
    audit_name = "eps:T2@heldout_combined:W2E12"
    au = None
    if audit_name in succ:
        au = paired(succ[audit_name], t2)
        bd = boot_diff(succ[audit_name], t2)
        au.update({"ci_lo": bd["lo"], "ci_hi": bd["hi"], "B": bd["B"], "seed": bd["seed"],
                   "reading": audit_reading(au["delta"], au["p_raw"], bd["lo"]),
                   "note": "t2_* = audit surface, off_* = canonical T2 (paired)"})
        quals.append(AUDIT_LINE.format(a=au["t2_success"], b=au["off_success"], n=au["n"], d=f"{au['delta']:+.3f}",
                                       lo=f"{bd['lo']:+.3f}", hi=f"{bd['hi']:+.3f}", p=fmt_p(au["p_raw"]),
                                       reading=au["reading"]))
    w4 = None
    if "eps:T2:W4E24" in succ:
        w4 = paired(succ["eps:T2:W4E24"], succ["eps:ORACLE:W4E24"])
        quals.append(W4_TEXT.format(t2=r3(w4["t2_rate"]), orc=r3(w4["off_rate"]),
                                    ratio=(r3(w4["t2_rate"] / w4["off_rate"]) if w4["off_rate"] else "n/a")))
    sf = {k: (S[f"fixed:{k}"]["success"] if f"fixed:{k}" in S else NOT_EVALUABLE_WORD)
          for k in ("T2", "OFF", "ORACLE", "IGNORE_STATUS")}
    quals.append(SF_TEXT.format(t2=sf["T2"], off=sf["OFF"], orc=sf["ORACLE"], ign=sf["IGNORE_STATUS"],
                                ignb=S["eps:IGNORE_STATUS:W2E12"]["success"]))
    W = wording(D, {"refused_share_t2": ref_share, "refused_instances_t2": ref_inst, "qualifiers": quals,
                    "not_evaluable": I.not_evaluable})
    res = {"prereg": "PREREG-LB", "draw_sha256": sha256(I.draw), "run_dir": str(I.run), "n": len(insts),
           "n_needs_lookahead": int(la.sum()), "n_greedy": int(greedy.sum()),
           "thresholds": {"DELTA_MIN": DELTA_MIN, "ALPHA_READING": ALPHA_READING, "RATIO_MIN": RATIO_MIN,
                          "RATIO_LB_MIN": RATIO_LB_MIN, "BOOT_B": BOOT_B, "BOOT_SEED": BOOT_SEED,
                          "REFUSED_FLAG_SHARE": REFUSED_FLAG_SHARE, "AUDIT_DROP_TOL": AUDIT_DROP_TOL,
                          "AUDIT_ALPHA": AUDIT_ALPHA},
           "decision": D, "wording": W, "arms": S, "cost": acct,
           "refused_flag_t2": bool(ref_share is not None and ref_share > REFUSED_FLAG_SHARE),
           "void_twin_exposure": exposure, "ratio_needs_lookahead": la_ratio, "t2_vs_ignore_status": ig,
           "audit": au, "w4e24": w4, "not_evaluable": I.not_evaluable,
           "discordant_by_depth": {"cp1_all": discordant_by_depth(t2, off, depth, np.ones_like(la)),
                                   "cp2_needs_lookahead": discordant_by_depth(t2, off, depth, la)},
           "answer_free": {"greedy_true_unmet_rate": r3(greedy.mean())},
           "instrument": instrument_record(I),
           "per_instance": {"ids": ids, "needs_lookahead": la.astype(int).tolist(),
                            "success": {k: v.astype(int).tolist() for k, v in succ.items()}}}
    return res


def render(res: dict, title: str = "PREREG-LB results") -> str:
    W, D = res["wording"], res["decision"]
    L = [f"# {title}", "", "Rendered mechanically by `scripts/lbeam_analysis.py` from the results JSON.", "",
         f"## {W['headline']}", "", f"Reading: **{W['reading']}** (branch {W['branch']}).", ""]
    L += [f"- {x}" for x in W["lines"]]
    L += ["", W["conclusion"], ""]
    if W["qualifiers"]:
        L += ["Qualifiers (report-only; they never change the branch):", ""] + [f"- {q}" for q in W["qualifiers"]] + [""]
    L += ["Limits that always accompany the result:", ""] + [f"- {x}" for x in W["limits"]] + [""]
    L += [f"Instances: {res['n']} ({res['n_needs_lookahead']} needs-lookahead, {res['n_greedy']} greedy-solvable).", ""]
    L += ["## Arms (success of the submitted plan under env.step)", "",
          "| arm / config | success | needs-lookahead | greedy | verified | verified-plan precision | optimal length | "
          "void-twin exposure: applied as posted (failures) | void step not a no-op |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in res["arms"].items():
        ex = res["void_twin_exposure"].get(k, {})
        tag = " (incl. stored distractors)" if ex.get("includes_stored_distractors") else ""
        L.append(f"| {k} | {v['success']} | {v['success_needs_lookahead']} | {v['success_greedy']} | {v['verified_rate']} "
                 f"| {v['verified_plan_precision']} | {v['optimal_length_rate']} | {ex.get('exposure')} "
                 f"({ex.get('failures_with_void_applied_as_posted')}/{ex.get('failures')}){tag} | "
                 f"{ex.get('exposure_not_noop')} |")
    for st in res.get("not_evaluable", {}):
        L.append(f"| {STAGE_WHAT[st]} | not evaluable | | | | | | | |")
    L += ["", "## Success by optimal depth d*", "", "| arm / config | " + " | ".join(f"d*={d}" for d in range(2, 7)) + " |",
          "|---|" + "---:|" * 5]
    for k, v in res["arms"].items():
        L.append(f"| {k} | " + " | ".join(str(v["success_by_depth"].get(str(d))) for d in range(2, 7)) + " |")
    L += ["", "## Co-primary discordant pairs by d* (report-only)", "",
          "| contrast | " + " | ".join(f"d*={d} (n; T2 only / OFF only)" for d in range(2, 7)) + " |", "|---|" + "---|" * 5]
    for k, lab in (("cp1_all", "co-primary 1 (all)"), ("cp2_needs_lookahead", "co-primary 2 (needs lookahead)")):
        row = res["discordant_by_depth"][k]
        L.append(f"| {lab} | " + " | ".join(f"{row[str(d)]['n']}; {row[str(d)]['t2_only']} / {row[str(d)]['off_only']}"
                                            for d in range(2, 7)) + " |")
    L += ["", "## Cost (distinct F keys per instance; tokens re-encoded from the answer texts; refused keys)", "",
          "| config | keys mean | keys p90 | keys max | tokens mean | refused unique | refused share | instances with a "
          "refused key |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in res["cost"].items():
        L.append(f"| {k} | {v['f_keys_per_instance']['mean']} | {v['f_keys_per_instance']['p90']} | "
                 f"{v['f_keys_per_instance']['max']} | {v['tokens_per_instance_reencoded']['mean']} | {v['refused_unique']} "
                 f"| {v['refused_share']} | {v['instances_with_refused_key']} |")
    ins = res["instrument"]
    L += ["", "## Instrument and deviations (report-only)", "",
          f"- Terminal record: {ins['terminal']}" + (f" (stop reason: {ins['stop_reason']})" if ins.get("stop_reason") else "")
          + ".", f"- A0 anchor: {json.dumps(ins['anchor'])}.", f"- A0 batch invariance: {json.dumps(ins['batch_invariance'])}."]
    for t in ins["turns"]:
        L.append(f"- Turn {t['turn']}: stages run {t['stages_run']}, deferred {t['deferred']}, no progress "
                 f"{t['no_progress']}, cap {t['cap_h']} h, device {t['device']}; A0 re-check "
                 f"{json.dumps(t['anchor_recheck'])}; batch-invariance re-check {json.dumps(t['batchinv_recheck'])}.")
    L += [f"- Orphaned stages (moved to _orphans/, regenerated in full): {json.dumps(ins['orphaned_stages'])}.",
          f"- Stage attempts: {json.dumps(ins['attempts'])}; abandoned: {json.dumps(ins['abandoned'])}.",
          f"- Failed A0 re-check records: {ins['recheck_failed_files']}.",
          "- BWM modules differing from the freeze record (report-only; only blocks_planner.py binds): "
          f"{json.dumps(ins['bwm_modules_drift'])}."]
    L += ["", f"Decision record: status 1 {D['status1']}, status 2 {D['status2']}, near {D['near']}.", ""]
    if res.get("inputs"):
        L += ["Inputs: " + json.dumps(res["inputs"]), ""]
    return "\n".join(L) + "\n"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_together(out_dir: Path, files: dict) -> None:
    """Create-only outputs written as one set: every file to <name>.partial first, then each renamed into place."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (out_dir / name).exists() or (out_dir / f"{name}.partial").exists():
            raise SystemExit(f"{out_dir / name} exists: the registered analysis runs once (create-only)")
    tmp = {}
    for name, text in files.items():
        t = out_dir / f"{name}.partial"
        with open(t, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        tmp[name] = t
    for name, t in tmp.items():
        os.rename(t, out_dir / name)


def run_registered(root: Path | None = None, run: Path | None = None) -> dict:
    root = Path(root or ROOT)
    run = Path(run or (root / "runs" / "lbeam"))
    out_dir = run / "analysis"
    for f in OUT_NAMES:
        if (out_dir / f).exists():
            raise SystemExit(f"{out_dir / f} exists: the registered analysis runs once (create-only)")
    t0 = time.time()
    I = Inputs(run, root / LR.DRAW_REL, registered=True, root=root)
    res = analyse(I)
    res["inputs"] = {"code_sha256": {r: sha256(root / r) for r in ("scripts/lbeam_analysis.py", "scripts/lbeam_run.py",
                                                                     "scripts/lbeam_problems.py", "scripts/lbeam_f.py")},
                     "numpy": np.__version__, "torch_imported": "torch" in sys.modules,
                     "seconds": round(time.time() - t0, 1), "freeze_prereg_sha256": I.fz["prereg_sha256"]}
    write_together(out_dir, {OUT_NAMES[0]: json.dumps(clean(res), indent=1) + "\n", OUT_NAMES[1]: render(res)})
    return res


# ---------------------------------------------------------------- dry run: statistics, every wording branch, metrics
def stats_selftest() -> dict:
    cases = OrderedDict()
    rng = np.random.default_rng(5)

    def brute(b, c):
        n = b + c
        if n == 0:
            return 1.0
        pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
        obs = pmf[b]
        return min(1.0, sum(p for p in pmf if p <= obs + 1e-15))
    ok = True
    for b, c in ((0, 0), (5, 0), (0, 5), (10, 3), (3, 10), (20, 20), (37, 9), (60, 2), (1, 1)):
        p1 = mcnemar_p(b, c)
        p2 = brute(b, c)          # two-sided by small probabilities equals the doubled tail for Bin(n, 1/2)
        ok &= abs(p1 - p2) < 1e-12
    cases["exact McNemar == the two-sided binomial test on the discordant pairs (9 cases)"] = ok
    cases["Holm (m=2): 0.004, 0.03 -> 0.008, 0.03; 0.03, 0.004 -> 0.03, 0.008; 0.02, 0.02 -> 0.04, 0.04; "
          "0.6, 0.7 -> 1.0, 1.0"] = (
        holm({"a": 0.004, "b": 0.03}) == {"a": 0.008, "b": 0.03} and holm({"a": 0.03, "b": 0.004}) == {"b": 0.008, "a": 0.03}
        and holm({"a": 0.02, "b": 0.02}) == {"a": 0.04, "b": 0.04} and holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0})
    t2 = rng.random(180) < 0.7
    orc = t2 | (rng.random(180) < 0.5)
    b1, b2 = boot_ratio(t2, orc), boot_ratio(t2, orc)
    cases["bootstrap ratio CI is deterministic for the fixed seed and brackets the point ratio"] = (
        b1 == b2 and b1["lo"] <= t2.mean() / orc.mean() <= b1["hi"])
    cases["delta boundary: exactly +0.20 with Holm p < 0.01 is PASS; +0.1944 is SHORT"] = (
        cp_status({"p_holm": 0.001, "b_t2_only": 40, "c_off_only": 4, "delta": 36 / 180}) == "PASS"
        and cp_status({"p_holm": 0.001, "b_t2_only": 39, "c_off_only": 4, "delta": 35 / 180}) == "SHORT")
    cases["Holm boundary: adjusted p 0.01 is NS; 0.0099 passes"] = (
        cp_status({"p_holm": 0.01, "b_t2_only": 60, "c_off_only": 0, "delta": 0.33}) == "NS"
        and cp_status({"p_holm": 0.0099, "b_t2_only": 60, "c_off_only": 0, "delta": 0.33}) == "PASS")
    # the D-35 audit reading: the reviewer's example (b = 12, c = 18 at n = 180: difference -0.033, p 0.36) is no longer
    # read 'invariant' when the paired CI allows a drop beyond 0.05
    a = np.array([1] * 12 + [0] * 18 + [1] * 60 + [0] * 90, bool)
    b = np.array([0] * 12 + [1] * 18 + [1] * 60 + [0] * 90, bool)
    pr, bd = paired(a, b), boot_diff(a, b)
    cases["audit reading: b 12 / c 18 of 180 (difference -0.033, p 0.36, CI lower bound < -0.05) is 'inconclusive'"] = (
        abs(pr["delta"] + 1 / 30) < 1e-9 and pr["p_raw"] > 0.3 and bd["lo"] < -0.05
        and audit_reading(pr["delta"], pr["p_raw"], bd["lo"]) == AUDIT_TEXT["inconclusive"])
    a2 = np.array([1] * 2 + [0] * 3 + [1] * 100 + [0] * 75, bool)
    b2 = np.array([0] * 2 + [1] * 3 + [1] * 100 + [0] * 75, bool)
    p2, d2 = paired(a2, b2), boot_diff(a2, b2)
    cases["audit reading: b 2 / c 3 of 180 (CI lower bound >= -0.05) is 'surface-invariant within 0.05'"] = (
        d2["lo"] >= -0.05 and audit_reading(p2["delta"], p2["p_raw"], d2["lo"]) == AUDIT_TEXT["invariant"])
    a3 = np.array([0] * 40 + [1] * 2 + [1] * 60 + [0] * 78, bool)
    b3 = np.array([1] * 40 + [0] * 2 + [1] * 60 + [0] * 78, bool)
    p3_, d3 = paired(a3, b3), boot_diff(a3, b3)
    cases["audit reading: a significant drop (b 2 / c 40) is 'surface-dependent'; the CI is deterministic"] = (
        audit_reading(p3_["delta"], p3_["p_raw"], d3["lo"]) == AUDIT_TEXT["dependent"] and boot_diff(a3, b3) == d3)
    return {"ok": all(cases.values()), "cases": {k: bool(v) for k, v in cases.items()}}


def synth_outcomes(n: int, n_la: int, t2_g: float, off_g: float, t2_la: float, off_la: float, orc_g: float,
                   orc_la: float, overlap=False) -> tuple:
    """Deterministic paired outcomes with exact rates per stratum (no randomness): instances 0..n_la-1 need lookahead.
    overlap False places OFF's successes at the END of a block (as little overlap with T2's as the rates allow: many
    discordant pairs); True at the START (T2 and OFF succeed together where both can: few discordant pairs); a pair
    (needs-lookahead block, greedy block) sets it per block. ORACLE >= T2 is not imposed (any pattern is admissible
    input to the rules)."""
    def block(m, p):
        k = int(round(p * m))
        return np.array([1] * k + [0] * (m - k), bool)
    ov_la, ov_g = (overlap, overlap) if isinstance(overlap, bool) else tuple(overlap)
    ng = n - n_la
    la = np.array([True] * n_la + [False] * ng)
    t2 = np.concatenate([block(n_la, t2_la), block(ng, t2_g)])
    o_la, o_g = block(n_la, off_la), block(ng, off_g)
    off = np.concatenate([o_la if ov_la else o_la[::-1], o_g if ov_g else o_g[::-1]])
    orc = np.concatenate([block(n_la, orc_la), block(ng, orc_g)])
    return t2, off, orc, la


# Realistic named cases: (n, n_la, T2 greedy, OFF greedy, T2 lookahead, OFF lookahead, ORACLE greedy, ORACLE lookahead
# [, overlap]) -> expected branch, statuses, near
WORDING_CASES = OrderedDict([
    ("YES_NEAR", ((180, 64, 0.95, 0.25, 0.55, 0.05, 1.0, 0.58), "YES_NEAR", "PASS", "PASS", True)),
    ("YES (ratio below 0.80)", ((180, 64, 0.70, 0.20, 0.40, 0.05, 1.0, 0.58), "YES", "PASS", "PASS", False)),
    ("GREEDY_ONLY_NEAR (lookahead NS)", ((180, 64, 1.0, 0.25, 0.30, 0.25, 1.0, 0.40), "GREEDY_ONLY_NEAR", "PASS", "NS", True)),
    ("GREEDY_ONLY (lookahead NS)", ((180, 64, 0.80, 0.20, 0.15, 0.10, 1.0, 0.58), "GREEDY_ONLY", "PASS", "NS", False)),
    ("GREEDY_ONLY (lookahead SHORT)", ((360, 200, 0.95, 0.35, 0.30, 0.15, 1.0, 0.58), "GREEDY_ONLY", "PASS", "SHORT", False)),
    ("LOOKAHEAD_ONLY (all SHORT)", ((180, 64, 0.40, 0.40, 0.45, 0.05, 1.0, 0.58, True), "LOOKAHEAD_ONLY", "SHORT", "PASS", False)),
    ("LOOKAHEAD_ONLY (all NS, near)", ((180, 64, 0.90, 0.95, 0.50, 0.10, 0.95, 0.55), "LOOKAHEAD_ONLY", "NS", "PASS", True)),
    ("DIRECTIONAL (both SHORT)", ((180, 64, 0.60, 0.45, 0.25, 0.08, 1.0, 0.58, True), "DIRECTIONAL", "SHORT", "SHORT", False)),
    ("DIRECTIONAL (all SHORT, lookahead NS)", ((180, 64, 0.75, 0.60, 0.10, 0.08, 1.0, 0.58, True), "DIRECTIONAL", "SHORT", "NS", False)),
    ("DIRECTIONAL (all NS, lookahead SHORT)", ((180, 64, 0.50, 0.55, 0.25, 0.10, 1.0, 0.58, True), "DIRECTIONAL", "NS", "SHORT", False)),
    ("NO (both NS)", ((180, 64, 0.50, 0.48, 0.10, 0.08, 1.0, 0.58), "NO", "NS", "NS", False)),
    ("NO (both NS, ratio near: OFF also near the oracle)", ((180, 64, 0.95, 0.94, 0.50, 0.48, 1.0, 0.55), "NO", "NS", "NS", True)),
    ("REVERSED (all instances)", ((180, 64, 0.30, 0.80, 0.10, 0.12, 1.0, 0.58), "REVERSED", "REVERSED", "NS", False)),
    ("REVERSED_LOOKAHEAD (lookahead only)", ((180, 64, 0.90, 0.70, 0.00, 0.40, 1.0, 0.58), "REVERSED_LOOKAHEAD", "NS", "REVERSED", False)),
    ("REVERSED (both)", ((180, 64, 0.10, 0.90, 0.00, 0.50, 1.0, 0.58), "REVERSED", "REVERSED", "REVERSED", False)),
    ("MIXED_PASS (reviewer's case: PASS overall, REVERSED where lookahead is needed, NEAR)",
     ((180, 64, 1.0, 0.0, 0.10, 0.60, 1.0, 0.10), "MIXED_PASS", "PASS", "REVERSED", True)),
])
# Every (status 1, status 2, NEAR) combination realised by real synthetic paired outcomes through the real decide()
# (found by a deterministic grid search over synth_outcomes; NEAR cases use ORACLE == T2, so the ratio is 1 in every
# resample; not-NEAR cases use ORACLE = 1 with T2 below 0.8).
COMBO_CASES = OrderedDict([
    ("NS/NS/not near", (180, 64, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, (False, False))),
    ("NS/NS/NEAR", (180, 64, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05, (False, False))),
    ("NS/PASS/not near", (180, 64, 0.0, 0.05, 0.2, 0.0, 1.0, 1.0, (False, False))),
    ("NS/PASS/NEAR", (180, 64, 0.0, 0.05, 0.2, 0.0, 0.0, 0.2, (False, False))),
    ("NS/REVERSED/not near", (180, 64, 0.05, 0.0, 0.0, 0.15, 1.0, 1.0, (False, False))),
    ("NS/REVERSED/NEAR", (180, 64, 0.05, 0.0, 0.0, 0.15, 0.05, 0.0, (False, False))),
    ("NS/SHORT/not near", (180, 64, 0.0, 0.05, 0.15, 0.0, 1.0, 1.0, (False, False))),
    ("NS/SHORT/NEAR", (180, 64, 0.0, 0.05, 0.15, 0.0, 0.0, 0.15, (False, False))),
    ("PASS/NS/not near", (180, 64, 0.15, 0.0, 0.6, 0.3, 1.0, 1.0, (False, False))),
    ("PASS/NS/NEAR", (180, 64, 0.15, 0.0, 0.6, 0.3, 0.15, 0.6, (False, False))),
    ("PASS/PASS/not near", (180, 64, 0.0, 0.0, 0.6, 0.0, 1.0, 1.0, (False, False))),
    ("PASS/PASS/NEAR", (180, 64, 0.0, 0.0, 0.6, 0.0, 0.0, 0.6, (False, False))),
    ("PASS/REVERSED/not near", (180, 64, 0.4, 0.0, 0.0, 0.15, 1.0, 1.0, (False, False))),
    ("PASS/REVERSED/NEAR", (180, 64, 0.4, 0.0, 0.0, 0.15, 0.4, 0.0, (False, False))),
    ("PASS/SHORT/not near", (180, 64, 0.3, 0.0, 0.15, 0.0, 1.0, 1.0, (False, False))),
    ("PASS/SHORT/NEAR", (180, 64, 0.3, 0.0, 0.15, 0.0, 0.3, 0.15, (False, False))),
    ("REVERSED/NS/not near", (180, 64, 0.0, 0.05, 0.0, 0.05, 1.0, 1.0, (False, False))),
    ("REVERSED/NS/NEAR", (180, 64, 0.0, 0.05, 0.05, 0.1, 0.0, 0.05, (True, False))),
    ("REVERSED/PASS/not near", (180, 64, 0.0, 0.3, 0.2, 0.0, 1.0, 1.0, (False, False))),
    ("REVERSED/PASS/NEAR", (180, 64, 0.0, 0.3, 0.2, 0.0, 0.0, 0.2, (False, False))),
    ("REVERSED/REVERSED/not near", (180, 64, 0.0, 0.0, 0.0, 0.15, 1.0, 1.0, (False, False))),
    ("REVERSED/REVERSED/NEAR", (180, 64, 0.0, 0.0, 0.05, 0.2, 0.0, 0.05, (True, False))),
    ("REVERSED/SHORT/not near", (180, 64, 0.0, 0.3, 0.15, 0.0, 1.0, 1.0, (False, False))),
    ("REVERSED/SHORT/NEAR", (180, 64, 0.0, 0.3, 0.15, 0.0, 0.0, 0.15, (False, False))),
    ("SHORT/NS/not near", (180, 64, 0.05, 0.0, 0.05, 0.0, 1.0, 1.0, (False, False))),
    ("SHORT/NS/NEAR", (180, 64, 0.05, 0.0, 0.05, 0.0, 0.05, 0.05, (False, False))),
    ("SHORT/PASS/not near", (180, 64, 0.0, 0.0, 0.2, 0.0, 1.0, 1.0, (False, False))),
    ("SHORT/PASS/NEAR", (180, 64, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, (False, False))),
    ("SHORT/REVERSED/not near", (180, 64, 0.3, 0.0, 0.0, 0.15, 1.0, 1.0, (False, False))),
    ("SHORT/REVERSED/NEAR", (180, 64, 0.3, 0.0, 0.0, 0.15, 0.3, 0.0, (False, False))),
    ("SHORT/SHORT/not near", (180, 64, 0.0, 0.0, 0.15, 0.0, 1.0, 1.0, (False, False))),
    ("SHORT/SHORT/NEAR", (180, 64, 0.0, 0.0, 0.15, 0.0, 0.0, 0.15, (False, False))),
])
DENIALS = ("'T2 enables planning' is not shown", "'T2 enables planning' (co-primary 1) is not shown", "which is not met",
           "T2 is not shown to plan better than OFF", "IS NOT SHOWN TO PLAN BETTER")
POSITIVE = ("PASS", "SHORT")


def truth_errors(D: dict, W: dict) -> list:
    """Mechanical truthfulness rules for the rendered wording of ANY outcome (amendment (a)): no rendered sentence
    contradicts a co-primary's status or the NEAR flag."""
    s1, s2, near = D["status1"], D["status2"], D["near"]
    H, C, Q, Rd = W["headline"], W["conclusion"], W["qualifiers"], W["reading"]
    texts = [H, C] + Q
    cl = C.lower()
    e = []
    if s1 == "PASS":
        if any(d in t for d in DENIALS for t in texts):
            e.append("a sentence denies co-primary 1 although it passes")
        if not Rd.startswith(("enables planning", "near-oracle planning")):
            e.append("co-primary 1 passes but the reading is not 'enables planning' / 'near-oracle planning'")
    elif Rd.startswith(("enables planning", "near-oracle planning")):
        e.append("the reading claims 'enables planning' without co-primary 1")
    worse_all_h = "WORSE THAN OFF OVER ALL INSTANCES" in H
    worse_la_h = "WORSE THAN OFF WHERE LOOKAHEAD IS NEEDED" in H or "WORSE WHERE LOOKAHEAD IS NEEDED" in H
    if worse_all_h != (s1 == "REVERSED"):
        e.append("the headline's 'worse over all instances' does not match status 1")
    if worse_la_h and s2 != "REVERSED":
        e.append("the headline says worse where lookahead is needed but status 2 is not REVERSED")
    c_worse_all = ("off plans significantly better than t2 over all instances" in cl
                   or "over all instances off plans significantly better than t2" in cl)
    c_worse_la = ("where lookahead is needed off plans significantly better than t2" in cl
                  or "off plans significantly better than t2 over all instances and where lookahead is needed" in cl)
    if c_worse_all != (s1 == "REVERSED"):
        e.append("the conclusion's reversal over all instances does not match status 1")
    if c_worse_la != (s2 == "REVERSED"):
        e.append("the conclusion's reversal where lookahead is needed does not match status 2")
    if H.startswith("NO") and (s1 in POSITIVE or s2 in POSITIVE):
        e.append("a 'NO' headline while a co-primary is significant in T2's favour")
    if "ENABLES" in H:
        if "ENABLES PLANNING WHERE LOOKAHEAD IS NEEDED" in H:
            if s2 != "PASS":
                e.append("the headline says enables planning where lookahead is needed without status 2 PASS")
        elif s1 != "PASS":
            e.append("the headline says T2 enables planning without status 1 PASS")
    if "NOT SHOWN WHERE LOOKAHEAD IS NEEDED" in H and s2 == "PASS":
        e.append("the headline says not shown where lookahead is needed although status 2 passes")
    claims_near = "near-oracle" in Rd
    if claims_near != (near and s1 == "PASS" and s2 != "REVERSED"):
        e.append("the near-oracle reading does not match (NEAR, co-primary 1 PASS, no reversal)")
    if ("ENABLES NEAR-ORACLE PLANNING" in H) != claims_near:
        e.append("the headline's near-oracle claim differs from the reading")
    n_only = any(q.startswith("T2/ORACLE") and "also requires co-primary 1" in q for q in Q)
    n_held = any(q.startswith("T2/ORACLE") and "is not read because" in q for q in Q)
    if n_only != (near and s1 != "PASS"):
        e.append("the near-only qualifier does not match (NEAR without co-primary 1)")
    if n_held != (near and s1 == "PASS" and s2 == "REVERSED"):
        e.append("the near-withheld qualifier does not match (NEAR, co-primary 1 PASS, reversal where needed)")
    if STATUS_TEXT[s1] not in W["lines"][0] or STATUS_TEXT[s2] not in W["lines"][1]:
        e.append("a co-primary's status sentence is missing from its result line")
    if any("{" in x for x in [H, C] + W["lines"] + Q):
        e.append("an unfilled placeholder")
    if any(" is passes " in x or " is significant in OFF" in x for x in [C] + Q):
        e.append("a status sentence is spliced ungrammatically ('is passes' / a reversal read as a gain)")
    if H != HEADLINE[W["branch"]] or Rd != READING[W["branch"]] or W["limits"] != list(LIMITS):
        e.append("headline / reading / limits are not the branch's registered strings")
    return e


def spec_branch(s1: str, s2: str, near: bool) -> str:
    """The branch table of PREREG_LB_DRAFT.md section 5, written independently of branch() (as a lookup)."""
    tab = {("PASS", "REVERSED"): "MIXED_PASS", ("SHORT", "REVERSED"): "MIXED_SHORT",
           ("REVERSED", "PASS"): "MIXED_LOOKAHEAD_BETTER", ("REVERSED", "SHORT"): "MIXED_LOOKAHEAD_BETTER",
           ("REVERSED", "NS"): "REVERSED", ("REVERSED", "REVERSED"): "REVERSED", ("NS", "REVERSED"): "REVERSED_LOOKAHEAD",
           ("PASS", "PASS"): "YES_NEAR" if near else "YES", ("PASS", "SHORT"): "GREEDY_ONLY_NEAR" if near else "GREEDY_ONLY",
           ("PASS", "NS"): "GREEDY_ONLY_NEAR" if near else "GREEDY_ONLY", ("SHORT", "PASS"): "LOOKAHEAD_ONLY",
           ("NS", "PASS"): "LOOKAHEAD_ONLY", ("SHORT", "SHORT"): "DIRECTIONAL", ("SHORT", "NS"): "DIRECTIONAL",
           ("NS", "SHORT"): "DIRECTIONAL", ("NS", "NS"): "NO"}
    return tab[(s1, s2)]


def wording_selftest() -> dict:
    out = OrderedDict()
    seen_br, seen_s = set(), set()

    def one(name, args, want):
        t2, off, orc, la = synth_outcomes(*args)
        D = decide(t2, off, orc, la)
        W = wording(D)
        got = [D["branch"], D["status1"], D["status2"], D["near"]]
        te = truth_errors(D, W)
        ok = got == list(want) and not te
        out[name] = {"ok": bool(ok), "want": list(want), "got": got, "truth_errors": te, "headline": W["headline"],
                     "reading": W["reading"], "conclusion": W["conclusion"], "qualifiers": W["qualifiers"],
                     "cp": {k: {kk: r3(vv) if isinstance(vv, float) else vv for kk, vv in D[k].items()} for k in ("cp1", "cp2")},
                     "ratio": {k: r3(v) if isinstance(v, float) else v for k, v in D["ratio"].items()}}
        seen_br.add(D["branch"])
        seen_s.update((D["status1"], D["status2"]))
    for name, (args, br, s1, s2, near) in WORDING_CASES.items():
        one(name, args, (br, s1, s2, near))
    for name, args in COMBO_CASES.items():
        s1, s2, nr = name.split("/")
        near = nr == "NEAR"
        one(f"combination {name} -> {spec_branch(s1, s2, near)}", args, (spec_branch(s1, s2, near), s1, s2, near))
    # the refused-key flag, the not-evaluable ratio, the not-evaluable report-only stages
    t2, off, orc, la = synth_outcomes(*WORDING_CASES["YES_NEAR"][0])
    W = wording(decide(t2, off, orc, la), {"refused_share_t2": 0.012, "refused_instances_t2": 7})
    W0 = wording(decide(t2, off, orc, la), {"refused_share_t2": 0.010, "refused_instances_t2": 7})
    out["refused-key flag above 1% only"] = {"ok": any(q.startswith("Flag (report-only)") for q in W["qualifiers"])
                                             and not any(q.startswith("Flag") for q in W0["qualifiers"])}
    Dz = decide(t2, off, np.zeros_like(orc), la)
    Wz = wording(Dz)
    out["ratio not evaluable when ORACLE success is 0 (NEAR false)"] = {"ok": (not Dz["near"]) and NEAR_NOT_EVALUABLE in Wz["lines"]}
    ne = {"G3": {"kind": "abandoned", "n": 2}, "G4": {"kind": "stopped"}}
    Wn = wording(decide(t2, off, orc, la), {"not_evaluable": ne})
    sent = not_evaluable_sentence(ne)
    out["not-evaluable report-only stages are rendered mechanically (abandoned / stopped) and never change the branch"] = {
        "ok": sent in Wn["qualifiers"] and "G3 (T2 at enum W4 E24), abandoned after 2 failed attempts" in sent
        and "G4 (the D-35 audit, T2 under heldout_combined), not run" in sent and Wn["branch"] == "YES_NEAR",
        "sentence": sent}
    # exhaustive: all 4 x 4 x 2 (status 1, status 2, NEAR) combinations map to exactly one registered branch, equal to
    # the independently written table, and every one of them was realised above with truthful wording
    ok_all, table = True, {}
    for s1 in CP_STATUSES:
        for s2 in CP_STATUSES:
            for near in (False, True):
                b = branch(s1, s2, near)
                ok_all &= b == spec_branch(s1, s2, near) and b in BRANCHES
                table[f"{s1}/{s2}/{'NEAR' if near else 'not near'}"] = b
    realised = {k.split(" -> ")[0].replace("combination ", "") for k in out if k.startswith("combination ")}
    out["exhaustive: every (status 1, status 2, NEAR) combination maps to the registered table (32) and is realised "
        "through decide() / wording() with truthful wording"] = {
        "ok": ok_all and len(table) == 32 and realised == set(table)
        and all(v["ok"] for k, v in out.items() if k.startswith("combination ")), "table": table}
    every = set(BRANCHES) <= seen_br and set(CP_STATUSES) <= seen_s
    return {"ok": all(v["ok"] for v in out.values()) and every, "every_branch_exercised": sorted(seen_br),
            "every_status_exercised": sorted(seen_s), "cases": out}


def metrics_selftest() -> dict:
    """void-twin exposure on hand-built episodes of a dev instance (ORACLE / IGNORE_STATUS exact answers, caches)."""
    cases = OrderedDict()
    dev = P.load_set(ROOT / LR.DEV_REL)
    inst = next(i for i in dev if any(a[4] == "void" and P.step_ignore_status(i["init"], a) != i["init"] for a in i["vocab"]))
    v = next(a for a in inst["vocab"] if a[4] == "void" and P.step_ignore_status(inst["init"], a) != inst["init"])
    rows = [{"id": inst["id"], "plan": [v], "verified": False}]
    ex_ig = void_twin_exposure([inst], rows, np.array([False]), f_answer_exact("IGNORE_STATUS"))
    ex_or = void_twin_exposure([inst], rows, np.array([False]), f_answer_exact("ORACLE"))
    cases["a failed plan with a void step: IGNORE_STATUS applies it as posted (exposure 1), ORACLE does not (0)"] = (
        ex_ig["exposure"] == 1.0 and ex_or["exposure"] == 0.0 and ex_or["failures_with_void_step"] == 1
        and ex_ig["exposure_not_noop"] == 1.0 and ex_or["exposure_not_noop"] == 0.0)
    ok_rows = [{"id": inst["id"], "plan": list(inst["opt_plan"]), "verified": True}]
    s, succ = summarise_eps([inst], ok_rows, np.array([True]))
    cases["the stored optimal plan scores success 1, optimal length 1, precision 1"] = (
        s["success"] == 1.0 and s["optimal_length_rate"] == 1.0 and s["verified_plan_precision"] == 1.0)
    cache = [{"s": inst["init"], "a": v, "text": json.dumps(P.state_to_dict(P.step_ignore_status(inst["init"], v)))}]
    fa = f_answer_from_cache(json.loads(json.dumps(cache)), LF.CANONICAL)
    ex_c = void_twin_exposure([inst], rows, np.array([False]), fa)
    cases["exposure read from a cache whose answer applied the void step"] = ex_c["exposure"] == 1.0
    posted = P.step_ignore_status(inst["init"], v)
    wrong = list(P.vec(posted))
    wrong[1] += 3                                       # an arithmetic error on the posted application
    need(P.from_vec(tuple(wrong)) not in (posted, inst["init"]), "metrics self-test: the perturbed state is degenerate")
    cache_w = [{"s": inst["init"], "a": v, "text": json.dumps(P.state_to_dict(P.from_vec(tuple(wrong))))}]
    ex_w = void_twin_exposure([inst], rows, np.array([False]), f_answer_from_cache(json.loads(json.dumps(cache_w)), LF.CANONICAL))
    cases["a void step applied with an arithmetic error: not 'applied as posted' (0) but 'not a no-op' (1)"] = (
        ex_w["exposure"] == 0.0 and ex_w["exposure_not_noop"] == 1.0)
    ex_m = void_twin_exposure([inst], rows, np.array([False]), lambda s_, a_: "MISSING", includes_distractors=True)
    cases["a guard-refused key (absent from the cache) is counted, and select_fixed exposures are labelled"] = (
        ex_m["missing_keys_guard_refused"] == 1 and ex_m["includes_stored_distractors"] is True)
    return {"ok": all(cases.values()), "cases": {k: bool(x) for k, x in cases.items()}}


def dry_run(out_dir: Path) -> dict:
    t0 = time.time()
    st, ws, ms = stats_selftest(), wording_selftest(), metrics_selftest()
    draft = ROOT / LR.DRAFT_REL
    missing = registered_strings_missing(draft.read_text(encoding="utf-8")) if draft.is_file() else ["draft missing"]
    out = {"ok": st["ok"] and ws["ok"] and ms["ok"] and not missing, "stats_selftest": st, "wording_selftest": ws,
           "metrics_selftest": ms, "registered_strings_missing_from_draft": missing,
           "code_sha256": {r: sha256(ROOT / r) for r in LR.DRYRUN_CODE},
           "draft_sha256": sha256(draft) if draft.is_file() else None,
           "hook_draft_sha256": sha256(ROOT / LR.HOOK_DRAFT_REL) if (ROOT / LR.HOOK_DRAFT_REL).is_file() else None,
           "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1)}
    md = ["# PREREG-LB analysis dry run", "", f"ok: {out['ok']}", "",
          f"Draft sha256 `{out['draft_sha256']}`; hook draft sha256 `{out['hook_draft_sha256']}`; registered sentences "
          f"missing from the draft: {len(missing)}.", "", "## Wording branches", "",
          "| case | want (branch, s1, s2, near) | got | truthful | ok |", "|---|---|---|---|---|"]
    for k, v in ws["cases"].items():
        md.append(f"| {k} | {v.get('want', '')} | {v.get('got', '')} | {not v.get('truth_errors')} | {v['ok']} |")
    md += ["", "## Statistics", ""] + [f"- {k}: {v}" for k, v in st["cases"].items()]
    md += ["", "## Metrics", ""] + [f"- {k}: {v}" for k, v in ms["cases"].items()] + [""]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "LB_DRYRUN.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(clean(out), indent=1) + "\n")
    with open(out_dir / "LB_DRYRUN.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    return out


# ---------------------------------------------------------------- power (numpy; inputs computed by lbeam_test_run.py)
def power_resample(trip: np.ndarray, greedy: np.ndarray, depth: np.ndarray, R: int, B: int, seed: int,
                   n_per_depth: int = 36) -> dict:
    """Monte-Carlo power of the registered rules: resample a confirmation-sized set (n_per_depth per d*) from paired
    per-instance outcome triples (T2, OFF, ORACLE) of a sizing set, apply decide() (bootstrap B, lazy below 0.80)."""
    rng = np.random.default_rng(seed)
    pools = {d: np.where(depth == d)[0] for d in sorted(set(depth.tolist()))}
    br, s1, s2, near, n_la, d1, d2 = Counter(), Counter(), Counter(), 0, [], [], []
    for r in range(R):
        idx = np.concatenate([rng.choice(pools[d], n_per_depth, replace=True) for d in pools])
        D = decide(trip[idx, 0], trip[idx, 1], trip[idx, 2], ~greedy[idx], B=B, seed=int(rng.integers(1 << 30)),
                   lazy_boot=True)
        br[D["branch"]] += 1
        s1[D["status1"]] += 1
        s2[D["status2"]] += 1
        near += D["near"]
        n_la.append(int((~greedy[idx]).sum()))
        d1.append(D["cp1"]["delta"])
        d2.append(D["cp2"]["delta"])
    return {"R": R, "B": B, "seed": seed, "n_per_depth": n_per_depth,
            "P_branch": {k: round(v / R, 4) for k, v in sorted(br.items())},
            "P_status1": {k: round(v / R, 4) for k, v in sorted(s1.items())},
            "P_status2": {k: round(v / R, 4) for k, v in sorted(s2.items())},
            "P_cp1_pass": round(s1["PASS"] / R, 4), "P_cp2_pass": round(s2["PASS"] / R, 4),
            "P_enables_planning_cp1": round(s1["PASS"] / R, 4),
            "P_near_oracle": round(sum(v for k, v in br.items() if k in ("YES_NEAR", "GREEDY_ONLY_NEAR")) / R, 4),
            "P_near_bar_met": round(near / R, 4),
            "n_needs_lookahead": {"mean": round(float(np.mean(n_la)), 1), "p5": int(np.percentile(n_la, 5)),
                                  "p95": int(np.percentile(n_la, 95))},
            "delta1_mean": round(float(np.mean(d1)), 4), "delta2_mean": round(float(np.mean(d2)), 4)}


def exact_pass_prob(n: int, p_t2: float, p_off: float, alpha: float, dmin: float = DELTA_MIN) -> float:
    """P(delta >= dmin and exact McNemar p < alpha) for n paired instances with INDEPENDENT T2 / OFF outcomes
    (b ~ T2 only w.p. p_t2 (1 - p_off), c ~ OFF only w.p. p_off (1 - p_t2)); exact multinomial enumeration."""
    pb, pc = p_t2 * (1 - p_off), p_off * (1 - p_t2)
    pe = max(0.0, 1.0 - pb - pc)
    lg = math.lgamma
    tot = 0.0
    for b in range(n + 1):
        for c in range(n + 1 - b):
            if (b - c) / n < dmin - 1e-12:
                continue
            if mcnemar_p(b, c) >= alpha:
                continue
            e = n - b - c
            lp = lg(n + 1) - lg(b + 1) - lg(c + 1) - lg(e + 1)
            lp += (b * math.log(pb) if b else 0.0) + (c * math.log(pc) if c else 0.0) + (e * math.log(pe) if e else 0.0)
            if (pb == 0 and b) or (pc == 0 and c) or (pe == 0 and e):
                continue
            tot += math.exp(lp)
    return tot


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="statistics, every wording branch, metrics -> LB_DRYRUN")
    ap.add_argument("--out-dir", default=None, help="dry run only: output directory (default docs/ledger_beam)")
    a = ap.parse_args()
    frozen = (ROOT / LR.FREEZE_REL).exists()
    if a.out_dir and not a.dry_run:
        raise SystemExit("--out-dir is for --dry-run only")
    if a.dry_run:
        out_dir = Path(a.out_dir).resolve() if a.out_dir else LR.DOCS
        if frozen and out_dir == LR.DOCS.resolve():
            raise SystemExit("PREREG-LB is frozen: docs/ledger_beam records are frozen inputs; use --out-dir elsewhere")
        out = dry_run(out_dir)
        for k, v in out["wording_selftest"]["cases"].items():
            print(f"  [{'PASS' if v['ok'] else 'FAIL'}] wording: {k}")
        for k, v in out["stats_selftest"]["cases"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] stats: {k}")
        for k, v in out["metrics_selftest"]["cases"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] metrics: {k}")
        print(f"dry run: {'PASS' if out['ok'] else 'FAIL'}; wrote {out_dir / 'LB_DRYRUN.json'} (+ .md)")
        need("torch" not in sys.modules, "torch was imported")
        return 0 if out["ok"] else 1
    need("torch" not in sys.modules, "torch must not be imported by the registered analysis")
    res = run_registered()
    need("torch" not in sys.modules, "torch was imported")
    print(f"wrote {ROOT / 'runs' / 'lbeam' / 'analysis' / OUT_NAMES[0]} (+ .md)")
    print(f"verdict: {res['wording']['headline']} (reading: {res['wording']['reading']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
