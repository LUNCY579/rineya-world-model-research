"""T2b-register screen analysis, revision 5 (RCWM roadmap item 3; joint item 5b). CPU only: numpy / json; never
imports torch, never loads a model. Applies the PRE-STATED go rule (t2br_common.GO_RULE_TEXT, revision 5; verbatim in
docs/t2b/register/GO_RULE.md and DESIGN.md section 7).

  python -B scripts/t2br_analysis.py --selftest    # scorer self-test on the dev bank and its held-out-surface rendering
                                                   # (golds 100%, negatives 0%, floors exact, the 4b constant-fill check,
                                                   # metric and chain-pipeline identities); writes
                                                   # docs/t2b/register/SCORER_SELFTEST.json
  python -B scripts/t2br_analysis.py --dryrun      # synthetic scenarios with known verdicts (GO, decision-point-3 stops,
                                                   # each NO-GO condition, G3 alone, invalid filler, malformed traces, a
                                                   # surface-locked trace arm, a coarse tens-digit comparison, the
                                                   # tens-digit shortcut 'tens(x) > tens(t) or x == t + 1' and its
                                                   # mirror, the off-by-one family, each against a plain arm exactly at
                                                   # chance and against a plain arm that is weak on every out-of-range
                                                   # row, engineered exact-boundary runs incl. the held-out surface and
                                                   # the predicate clause) + exhaustive exact-boundary checks of
                                                   # verdict() + the enumeration of digit shortcuts and of the
                                                   # off-by-one family against G1; writes DRYRUN.json + .md
  python -B scripts/t2br_analysis.py --run-dir runs/t2b_register/s0_sd [--run-dir ...s1_sd ...]
                                                   # the screen analysis (registered: freeze + provenance verified);
                                                   # writes runs/t2b_register/analysis/T2BR_RESULTS_<seeds>_<variant>
                                                   # .{json,md} (create-only)

Scoring. Outcome per one-step row = the exact 'after' state, read by the decision parser of the side (t2br_common.
parse_after: plain / OFF = rcwmt.env.parse_state(lenient); trace / filler = the strict joint parsers, then parse_state on
'after'; malformed = wrong). Mechanism fields for the trace side: format_ok, cmp_ok (predicate inputs + ok, ordered
JSON), ok_ok (branch boolean), calc_ok, after_follows_calc, calc_follows_cmp. Filler side: format_ok only (amendment 4b:
the fill is a constant per action kind, so the emitted fill length is not an outcome; revision 2's fill_exact
diagnostics are dropped in favour of the data-level constant-length check, verified before any output is read). Every
side's branch choice on conditional rows is also read from its 'after' (correct branch / wrong branch / neither).
Two renderings of the same rows, paired by row: IN-SURFACE (E1: Titan's verbatim frame, every dev row) and HELD-OUT
SURFACE (E1H: the primary rows and the identification strata cond_sametens + cond_oor_edge in the combined held-out
surface, data/t2br_dev_hs.jsonl).
Metrics. 'conditional' (PRIMARY) = mean exact over cond_oor_val + cond_oor_thr + cond_far (branch-balanced pairs);
'cond_oor_val' = the boundary pairs alone (G1's second clause); 'cond_oor_edge' = the boundary quadruples alone
(revision 5) and 'cond_sametens' = the same-tens identification pairs alone (revision 4), neither pooled into
'conditional'; the trace side's PREDICATE ACCURACY on cond_oor_edge and on cond_sametens (revision 5: t2br_common.pred_ok,
the pseudo-metrics '<stratum>_pred', G1's absolute clause >= 0.80 with no plain-arm term); 'mix' (PRIMARY) = mix_oor;
'conditional_in' =
branch-balanced mean over cond_xm1 / cond_x0 / cond_xp1 / cond_mid (0.5 x ok-false rows + 0.5 x ok-true rows); 'mix_in'
= mix; controls swap / add_nowrap. The RULE is decided on exact rationals (metric_exact: integer counts,
fractions.Fraction; bars in t2br_common are Fractions). 95% CIs: cluster bootstrap over pairs / triplets (10,000 draws,
seed 17), paired across sides and across the two renderings; Wilson for single accuracies; a rule component whose value
lies within one bootstrap SE ((hi - lo) / 3.92) of its bar is flagged (reported, not gating).
Reported, not gating: the held-out-surface results of every side (OFF and filler included); the surface audit
(held-out minus in-surface per side and primary stratum, paired by row, with the discordant row counts); the filler
residual covariate (the constant filler's tokens minus that row's trace tokens, a property of the data) with the filler
side's accuracy split at its median.
Chains (k = 4 self-rollouts, chain_one frame): exact state at k, normalised regret at k = 1, 2, 4 through the frozen
decision pipeline (rcwmt.data.candidate_table -> rcwmt.decisions.g_only: 256 Gaussian goals seed 1001, tie seed 17,
unparseable -> parent-mean utility), per stratum (CH_IN, CH_X, ALL), parent-paired bootstrap contrasts, the copy-s0
floor; and the per-step -> k4 product law: p(class) from each side's own gold-input chain steps (RESULTS_PILOT R8
convention), predicted k4 exact = mean over chains of the product of p over the chain's 4 gold step classes, against
the observed k4 exact (and the pooled p^4).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if "torch" in sys.modules:
    raise SystemExit("torch must not be imported by the T2b-register analysis")
if "rcwmt.heads" not in sys.modules:          # decisions / fidelity import heads (torch); g_only never uses it
    _m = types.ModuleType("rcwmt.heads")
    _m.__doc__ = "placeholder installed by t2br_analysis.py (as t2_analysis.py does): keeps torch out"
    sys.modules["rcwmt.heads"] = _m
import t2br_common as C  # noqa: E402
from rcwmt import data, decide, decisions, env, fidelity  # noqa: E402,F401

if "torch" in sys.modules:
    raise SystemExit("torch was imported by a dependency of the T2b-register analysis")

FAM = C.FAM
N_BOOT = 10000
BOOT_SEED = 17
Z95 = 1.959963984540054
KS = (1, 2, 4)
DOCS = C.DOCS
CONTRASTS = (("trace", "plain"), ("trace", "filler"), ("filler", "plain"), ("trace", "off"), ("plain", "off"),
             ("filler", "off"))


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[t2br-analysis] {msg}")


def wilson(k: int, n: int):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _write(path: Path, text: str, mode: str = "w") -> None:
    """Text files are written with LF line endings only."""
    with Path(path).open(mode, encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- metrics on one-step rows
# Outcome keys. The PRIMARY outcomes are named primary_conditional / primary_mix here, because the in-range dev
# stratum is itself called 'mix' (a key collision would silently replace the primary by the in-range stratum).
GROUPS = ("primary_conditional", "primary_mix", "conditional_in", "mix_in")
PRIMS = ("primary_conditional", "primary_mix")
METRICS = {**{s: ("mean", (s,)) for s in C.DEV_STRATA},
           "primary_conditional": ("mean", C.PRIMARY["conditional"]), "primary_mix": ("mean", C.PRIMARY["mix"]),
           "conditional_in": ("branch_balanced", C.NONINF["conditional_in"]), "mix_in": ("mean", C.NONINF["mix_in"])}
assert len(METRICS) == len(C.DEV_STRATA) + len(GROUPS), "outcome key collision"
# metrics available on the held-out surface (every one lies inside the held-out-surface strata)
HS_METRICS = ("primary_conditional", "cond_oor_val", "cond_oor_thr", "cond_far", "primary_mix", "mix_oor",
              "cond_sametens", "cond_oor_edge")
assert all(set(METRICS[m][1]) <= set(C.HS_STRATA) for m in HS_METRICS)
# Revision 5: G1's predicate-accuracy clause as pseudo-metrics of the TRACE side only (key -> its stratum), decided
# exactly like every metric (metric_exact on the per-row pred_ok vector), with no plain-arm term
PRED_METRICS = {f"{s}_pred": s for s in C.PRED_STRATA}
PART_LABEL = {"primary_conditional": "conditional", "cond_oor_val": "cond_oor_val", "cond_sametens": "cond_sametens",
              "cond_oor_edge": "cond_oor_edge", **{pm: f"{s} predicate" for pm, s in PRED_METRICS.items()},
              "primary_mix": "mix",
              "swap": "swap", "add_nowrap": "add_nowrap", "conditional_in": "conditional_in", "mix_in": "mix_in"}


class DevIndex:
    def __init__(self, dev: list):
        self.dev = dev
        self.strata = np.array([r["stratum"] for r in dev])
        self.cls = np.array([1 if r["gold_ok"] else 0 for r in dev])
        self.clus = np.array([r["cluster"] for r in dev])
        self.idx = {m: np.where(np.isin(self.strata, list(ss)))[0] for m, (_, ss) in METRICS.items()}
        self.pos = {r["id"]: j for j, r in enumerate(dev)}
        self.hs_rows = np.where(np.isin(self.strata, list(C.HS_STRATA)))[0]


def metric_exact(x: np.ndarray, D: DevIndex, m: str) -> Fraction:
    """The metric as an exact rational from integer counts (x: 0/1 per dev row). The rule is decided on these."""
    kind, _ = METRICS[m]
    i = D.idx[m]
    xi = np.asarray(x)[i]
    if kind == "mean":
        return Fraction(int(round(float(xi.sum()))), len(i))
    c = D.cls[i]
    k0, n0 = int(round(float(xi[c == 0].sum()))), int((c == 0).sum())
    k1, n1 = int(round(float(xi[c == 1].sum()))), int((c == 1).sum())
    return Fraction(k0, 2 * n0) + Fraction(k1, 2 * n1)


def metric_point(x: np.ndarray, D: DevIndex, m: str) -> float:
    return float(metric_exact(x, D, m))


def _cluster_sums(x: np.ndarray, D: DevIndex, m: str):
    kind, _ = METRICS[m]
    i = D.idx[m]
    uniq, inv = np.unique(D.clus[i], return_inverse=True)
    K = 2 if kind == "branch_balanced" else 1
    k = D.cls[i] if K == 2 else np.zeros(len(i), int)
    S = np.zeros((len(uniq), K))
    N = np.zeros((len(uniq), K))
    np.add.at(S, (inv, k), x[i])
    np.add.at(N, (inv, k), 1.0)
    return S, N


def metric_boot_diff(xa: np.ndarray, xb: np.ndarray, D: DevIndex, m: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Cluster bootstrap (pairs / triplets resampled jointly for both vectors) of metric(a) - metric(b). The two vectors
    may be two sides or two renderings of the same rows (paired by row either way)."""
    Sa, N = _cluster_sums(xa, D, m)
    Sb, _ = _cluster_sums(xb, D, m)
    J = len(N)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, J, (n_boot, J))
    cnt = np.zeros((n_boot, J))
    np.add.at(cnt, (np.arange(n_boot)[:, None], draws), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = (cnt @ Sa) / (cnt @ N), (cnt @ Sb) / (cnt @ N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(ma, axis=1) - np.nanmean(mb, axis=1)
    d = d[~np.isnan(d)]
    return [_r(np.quantile(d, 0.025)), _r(np.quantile(d, 0.975))]


def score_side(dev: list, texts: list, side: str) -> list:
    arm = C.SIDE_ARM[side]
    return [C.score_text(r, t, arm) for r, t in zip(dev, texts)]


def branch_of(row: dict, st) -> str:
    """Which conditional branch an answer applied (reduced or unreduced), read from its 'after' state."""
    if st is None:
        return "none"
    b, ev = row["before"], row["event"]
    r, v = ev["register"], ev["value"]
    if any(st[k] != b[k] for k in C.REGS if k != r):
        return "neither"
    y = st[r]
    if y in ((b[r] + v) % C.MOD, b[r] + v):
        return "add"
    if y in ((b[r] - v) % C.MOD, b[r] - v):
        return "sub"
    return "neither"


def _branch_table(dev: list, idx_by_stratum: dict, states: dict) -> dict:
    """states: dev row index -> parsed 'after' (or None)."""
    br = {}
    for s, ii in idx_by_stratum.items():
        cnt = Counter()
        for j in ii:
            row = dev[j]
            got = branch_of(row, states[j])
            want = "add" if row["gold_ok"] else "sub"
            cnt["correct" if got == want else ("wrong" if got in ("add", "sub") else "neither_or_unparsed")] += 1
        n = len(ii)
        chosen = cnt["correct"] + cnt["wrong"]
        br[s] = {"correct_branch": _r(cnt["correct"] / n), "wrong_branch": _r(cnt["wrong"] / n),
                 "neither_or_unparsed": _r(cnt["neither_or_unparsed"] / n),
                 "branch_choice_given_a_branch": _r(cnt["correct"] / chosen) if chosen else None}
    return br


def side_block(dev: list, texts: list, side: str, D: DevIndex) -> dict:
    sc = score_side(dev, texts, side)
    x = np.array([s["after_ok"] for s in sc], float)
    arm = C.SIDE_ARM[side]
    out = {"x": x, "scores": sc, "strata": {}, "metrics": {}}
    if arm == "trace":
        out["x_pred"] = np.array([s["pred_ok"] for s in sc], float)
    for s in C.DEV_STRATA:
        i = D.idx[s]
        k = int(x[i].sum())
        out["strata"][s] = {"n": len(i), "acc": _r(k / len(i)), "wilson95": wilson(k, len(i)),
                            "parsed": _r(np.mean([sc[j]["parsed"] for j in i]))}
    for m in GROUPS:
        out["metrics"][m] = _r(metric_point(x, D, m))
    states = {j: C.parse_after(texts[j], arm) for j in range(len(dev)) if dev[j]["event"]["kind"] == "conditional"}
    out["branch_from_after"] = _branch_table(dev, {s: D.idx[s] for s in C.DEV_STRATA if s.startswith("cond")}, states)
    # unreduced-sum answers on mix / wrap strata (OFF's known failure)
    un = {}
    for s in ("mix", "wrap", "mix_oor", "wrap_oor"):
        hit = 0
        for j in D.idx[s]:
            st = C.parse_after(texts[j], arm)
            h = C.heuristic_after("unreduced", dev[j]["before"], dev[j]["event"])
            hit += st is not None and env.dumps(st) == env.dumps(h) and dev[j]["gold_after"] != env.dumps(h)
        un[s] = _r(hit / len(D.idx[s]))
    out["unreduced_answers"] = un
    if arm in ("trace", "filler"):
        fields = ("format_ok", "pred_ok", "cmp_ok", "ok_ok", "calc_ok", "after_follows_calc", "calc_follows_cmp") \
            if arm == "trace" else ("format_ok",)
        tf = {}
        for s in C.DEV_STRATA:
            i = D.idx[s]
            tf[s] = {f: _r(np.mean([sc[j][f] for j in i])) for f in fields
                     if not (f in ("pred_ok", "ok_ok", "calc_follows_cmp") and not s.startswith("cond"))}
        out["trace_fields"] = tf
        out["format_exact"] = {p: Fraction(sum(bool(sc[j]["format_ok"]) for j in D.idx[p]), len(D.idx[p])) for p in PRIMS}
    else:
        out["format_exact"] = {p: Fraction(1) for p in PRIMS}
    return out


def hs_side_block(dev: list, hs: list, texts: list, side: str, D: DevIndex) -> dict:
    """One side on the held-out-surface rows. Each HS row is scored against ITS dev row (same id, same golds); the
    outcome vector is indexed like the dev bank (rows outside the primary strata stay 0 and enter no HS metric)."""
    arm = C.SIDE_ARM[side]
    x = np.zeros(len(dev))
    xp = np.zeros(len(dev))
    sc, states = {}, {}
    for h, t in zip(hs, texts):
        j = D.pos[h["id"]]
        s = C.score_text(dev[j], t, arm)
        sc[j] = s
        x[j] = float(s["after_ok"])
        if arm == "trace":
            xp[j] = float(s["pred_ok"])
        if dev[j]["event"]["kind"] == "conditional":
            states[j] = C.parse_after(t, arm)
    out = {"x": x, "strata": {}, "metrics": {}}
    if arm == "trace":
        out["x_pred"] = xp
    for s in C.HS_STRATA:
        i = D.idx[s]
        k = int(x[i].sum())
        cell = {"n": len(i), "acc": _r(k / len(i)), "wilson95": wilson(k, len(i)),
                "parsed": _r(np.mean([sc[j]["parsed"] for j in i]))}
        if arm in ("trace", "filler"):
            cell["format_ok"] = _r(np.mean([sc[j]["format_ok"] for j in i]))
        if arm == "trace":
            for f in ("cmp_ok", "calc_ok", "after_follows_calc") + (("pred_ok",) if s.startswith("cond") else ()):
                cell[f] = _r(np.mean([sc[j][f] for j in i]))
        out["strata"][s] = cell
    for m in HS_METRICS:
        out["metrics"][m] = _r(metric_point(x, D, m))
    out["branch_from_after"] = _branch_table(dev, {s: D.idx[s] for s in C.HS_STRATA if s.startswith("cond")}, states)
    return out


def fill_constant_check(dev: list) -> dict:
    """Amendment 4b on the dev bank (data only): ONE fill string per action kind (FILL_UNITS), so every filler target of
    a kind is identical up to 'after', and within every primary stratum the fill length carries zero information about
    the answer (its best single cut equals the majority rate)."""
    fills = defaultdict(set)
    for r in dev:
        fills[r["event"]["kind"]].add(json.loads(r["targets"]["filler"])["fill"])
    one = {k: sorted(len(f) // len(C.FILL_UNIT) for f in v) for k, v in fills.items()}
    ok = all(v == {C.FILL_UNIT * C.FILL_UNITS[k]} for k, v in fills.items()) and \
        all(r["fill_units"] == C.FILL_UNITS[r["event"]["kind"]] for r in dev)
    info = {}
    for s in C.HS_STRATA:
        rs = [r for r in dev if r["stratum"] == s]
        y = [bool(r["gold_ok"]) if r["event"]["kind"] == "conditional" else C.wraps(r["before"], r["event"]) for r in rs]
        maj = max(sum(y), len(y) - sum(y)) / len(y)
        cut = C.best_cut_acc([len(json.loads(r["targets"]["filler"])["fill"]) for r in rs], y)
        info[s] = {"label": "branch" if rs[0]["event"]["kind"] == "conditional" else "wrap", "majority": _r(maj),
                   "fill_length_best_cut": _r(cut), "zero_information": abs(cut - maj) < 1e-12}
        ok &= info[s]["zero_information"]
    return {"ok": bool(ok), "fill_units_by_kind": one, "information_test": info}


def residual_covariate(dev: list, D: DevIndex, x_filler: np.ndarray, x_filler_hs: np.ndarray) -> dict:
    """Reported, not gating: residual = the constant filler target's tokens minus that row's trace tokens (>= 0; a
    property of the data, answer-dependent only through the trace's length). Per primary cell: its distribution and the
    filler side's accuracy at or below / above the cell's median residual (in-surface and held-out surface)."""
    out = {}
    cells = {**{s: D.idx[s] for s in C.HS_STRATA}, **{p: D.idx[p] for p in PRIMS}}
    for name, i in cells.items():
        res = np.array([dev[j]["fill_residual"] for j in i], float)
        med = float(np.median(res))
        lo = [j for j in i if dev[j]["fill_residual"] <= med]
        hi = [j for j in i if dev[j]["fill_residual"] > med]
        out[name] = {"n": len(i), "residual_mean": _r(res.mean()), "residual_median": med,
                     "residual_range": [int(res.min()), int(res.max())],
                     "n_le_median": len(lo), "n_gt_median": len(hi),
                     "filler_acc_le_median": _r(x_filler[lo].mean()) if lo else None,
                     "filler_acc_gt_median": _r(x_filler[hi].mean()) if hi else None,
                     "filler_hs_acc_le_median": _r(x_filler_hs[lo].mean()) if lo else None,
                     "filler_hs_acc_gt_median": _r(x_filler_hs[hi].mean()) if hi else None}
    return out


# ---------------------------------------------------------------- chains
def chain_rows_for(parents: list) -> list:
    return data.candidate_table({"parents": parents})


def regret(rows: list, k: int, gen: dict):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        g = decisions.g_only(rows, FAM, k, {"m": gen}, modes=("m",), split="dev")["m"]
    return g["pids"], np.asarray(g["nregret"], np.float64)


def floor_copy_regret(rows: list, k: int):
    """F_copy_s0 (answer-free floor) through the same goals and tie rule as g_only."""
    gen = {}
    for r in rows:
        if r["k"] == k:
            gen.setdefault(r["cand_id"], {"states": [env.dumps(r["s0"])] * (C.H + 1)})
    return regret(rows, k, gen)


def parent_exact(rows: list, k: int, gen: dict, pids: list) -> np.ndarray:
    acc = defaultdict(list)
    for r in rows:
        if r["k"] == k:
            acc[r["parent_id"]].append(float(gen[r["cand_id"]]["states"][k] == r["gold"]))
    return np.array([np.mean(acc[p]) for p in pids], np.float64)


def product_law(parents: list, gen: dict) -> dict:
    n, c = Counter(), Counter()
    chains = C.flat_chains(parents)
    golds = {}
    for ch in chains:
        gold = env.rollout(FAM, ch["s0"], ch["events"])
        golds[ch["cand_id"]] = gold
        st = gen[ch["cand_id"]]["states"]
        for t, ev in enumerate(ch["events"]):
            if st[t] != env.dumps(gold[t]):
                continue
            cl = C.op_class(gold[t], ev)
            n[cl] += 1
            c[cl] += int(st[t + 1] == env.dumps(gold[t + 1]))
    pooled = sum(c.values()) / sum(n.values()) if sum(n.values()) else float("nan")
    p = {cl: (c[cl] / n[cl]) for cl in n}
    preds, obs = [], []
    for ch in chains:
        gold = golds[ch["cand_id"]]
        pr = 1.0
        for t, ev in enumerate(ch["events"]):
            pr *= p.get(C.op_class(gold[t], ev), pooled)
        preds.append(pr)
        obs.append(float(gen[ch["cand_id"]]["states"][C.H] == env.dumps(gold[C.H])))
    return {"per_class": {cl: {"n_gold_input_steps": n[cl], "acc": _r(p[cl])} for cl in sorted(n)},
            "pooled_step_acc": _r(pooled), "predicted_k4_exact_product": _r(np.mean(preds)),
            "predicted_k4_exact_pooled_pow4": _r(pooled ** C.H) if not math.isnan(pooled) else None,
            "observed_k4_exact": _r(np.mean(obs)), "observed_minus_predicted": _r(np.mean(obs) - np.mean(preds)),
            "n_chains": len(chains)}


def chain_block(parents: list, gens: dict) -> dict:
    out = {}
    strata = {**{s: [p for p in parents if p["stratum"] == s] for s in C.CHAIN_STRATA}, "ALL": parents}
    for sname, ps in strata.items():
        rows = chain_rows_for(ps)
        blk = {"n_parents": len(ps), "n_chains": sum(len(p["candidates"]) for p in ps), "k": {}}
        for k in KS:
            per = {}
            pids = None
            for side in C.SIDES:
                pids, nr = regret(rows, k, gens[side])
                per[side] = {"nregret": nr, "exact": parent_exact(rows, k, gens[side], pids)}
            fp, fr = floor_copy_regret(rows, k)
            need(fp == pids, "floor parents differ")
            kb = {"sides": {s: {"regret": decide.bootstrap_mean(per[s]["nregret"]),
                                "exact": decide.bootstrap_mean(per[s]["exact"])} for s in C.SIDES},
                  "floor_copy_s0_regret": decide.bootstrap_mean(fr), "contrasts": {}}
            for a, b in CONTRASTS:
                kb["contrasts"][f"{a}-{b}"] = {"regret": decide.paired_bootstrap(per[a]["nregret"], per[b]["nregret"]),
                                               "exact": decide.paired_bootstrap(per[a]["exact"], per[b]["exact"])}
            blk["k"][str(k)] = kb
        blk["product_law"] = {s: product_law(ps, gens[s]) for s in C.SIDES}
        out[sname] = blk
    return out


# ---------------------------------------------------------------- the pre-stated rule (exact rationals)
# (condition, side a, side b, metric, bar, op, surface): the component holds iff (a - b) op bar on that surface
# ('in' = Titan's verbatim frame, E1; 'hs' = the combined held-out surface, E1H)
# side b None = an ABSOLUTE component (the value is side a's own metric; revision 5's predicate clause)
COMPONENTS = (
    ("G1", "trace", "plain", "primary_conditional", C.GAIN_CONDITIONAL, ">=", "in"),
    ("G1", "trace", "plain", "cond_oor_val", C.GAIN_CONDITIONAL, ">=", "in"),
    ("G1", "trace", "plain", "cond_oor_edge", C.GAIN_CONDITIONAL, ">=", "in"),        # revision 5
    ("G1", "trace", None, "cond_oor_edge_pred", C.PRED_MIN, ">=", "in"),             # revision 5 (absolute)
    ("G1", "trace", "plain", "cond_sametens", C.GAIN_CONDITIONAL, ">=", "in"),        # revision 4
    ("G1", "trace", None, "cond_sametens_pred", C.PRED_MIN, ">=", "in"),             # revision 5 (absolute)
    ("G1", "trace", "plain", "primary_conditional", C.GAIN_CONDITIONAL, ">=", "hs"),
    ("G1", "trace", "plain", "cond_oor_val", C.GAIN_CONDITIONAL, ">=", "hs"),
    ("G1", "trace", "plain", "cond_oor_edge", C.GAIN_CONDITIONAL, ">=", "hs"),        # revision 5
    ("G1", "trace", None, "cond_oor_edge_pred", C.PRED_MIN, ">=", "hs"),             # revision 5 (absolute)
    ("G1", "trace", "plain", "cond_sametens", C.GAIN_CONDITIONAL, ">=", "hs"),        # revision 4
    ("G1", "trace", None, "cond_sametens_pred", C.PRED_MIN, ">=", "hs"),             # revision 5 (absolute)
    ("G2", "trace", "plain", "primary_mix", C.GAIN_MIX, ">=", "in"),
    ("G2", "trace", "plain", "primary_mix", C.GAIN_MIX, ">=", "hs"),
    ("G3", "trace", "filler", "primary_conditional", C.FILLER_STOP, ">=", "in"),
    ("G3", "trace", "filler", "primary_mix", C.FILLER_STOP, ">=", "in"),
    ("G4", "trace", "off", "swap", -C.CONTROL_LOSS, ">=", "in"),
    ("G4", "trace", "off", "add_nowrap", -C.CONTROL_LOSS, ">=", "in"),
    ("G5", "trace", "plain", "conditional_in", -C.NONINF_MARGIN, ">=", "in"),
    ("G5", "trace", "plain", "mix_in", -C.NONINF_MARGIN, ">=", "in"),
    ("DP3", "trace", "filler", "primary_conditional", C.FILLER_STOP, "<", "in"),
    ("DP3", "trace", "filler", "primary_mix", C.FILLER_STOP, "<", "in"),
)
G_NAMES = ("G1", "G2", "G3", "G4", "G5")
CONDITION_NAMES = {
    "G1": "G1 trace-plain >= +0.25 on conditional (OOR pooled), on cond_oor_val, cond_oor_edge and cond_sametens "
          "alone, and trace predicate accuracy >= 0.80 on cond_oor_edge and cond_sametens, in-surface and held-out "
          "surface",
    "G2": "G2 trace-plain >= +0.30 on mix (OOR), in-surface and held-out surface",
    "G3": "G3 filler below trace: trace-filler >= +0.10 on each in-surface primary (4b filler)",
    "G4": "G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap",
    "G5": "G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in",
    "V": "V filler side valid: filler in-surface format_ok >= 0.95 on each primary",
}


def part_label(metric: str, surface: str) -> str:
    return PART_LABEL[metric] + (" HS" if surface == "hs" else "")


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    raise TypeError(f"the rule is decided on exact rationals; got {type(x).__name__} {x!r}")


def verdict(m: dict) -> dict:
    """m: {'acc': {side: {metric: Fraction}} (in-surface), 'acc_hs': {side: {metric: Fraction}} (held-out surface),
    'format': {'trace' / 'filler': {primary: Fraction}} (in-surface)}. Every comparison is exact (Fractions); floats are
    refused. Returns the verdict with every condition, its failed parts and every component."""
    a = {s: {k: _frac(v) for k, v in d.items()} for s, d in m["acc"].items()}
    h = {s: {k: _frac(v) for k, v in d.items()} for s, d in m["acc_hs"].items()}
    f = {s: {k: _frac(v) for k, v in d.items()} for s, d in m["format"].items()}
    comps = []
    for name, x, y, k, bar, op, surf in COMPONENTS:
        src = a if surf == "in" else h
        val = src[x][k] if y is None else src[x][k] - src[y][k]
        holds = val >= bar if op == ">=" else val < bar
        comps.append({"condition": name, "surface": surf, "part": part_label(k, surf),
                      "contrast": x if y is None else f"{x}-{y}",
                      "metric": k, "op": op, "bar": str(bar), "value": str(val), "value_float": _r(float(val), 6),
                      "holds": bool(holds)})
    failed_parts = {g: [c["part"] for c in comps if c["condition"] == g and not c["holds"]] for g in G_NAMES}
    trace_valid = all(f["trace"][p] >= C.FORMAT_VALID for p in PRIMS)
    filler_valid = all(f["filler"][p] >= C.FORMAT_VALID for p in PRIMS)
    cond = {CONDITION_NAMES[g]: not failed_parts[g] for g in G_NAMES}
    cond[CONDITION_NAMES["V"]] = filler_valid
    dp3 = all(c["holds"] for c in comps if c["condition"] == "DP3")
    neither = all(a["filler"][p] - a["plain"][p] < C.FILLER_STOP for p in PRIMS)
    sub = None
    if dp3:
        v = C.DP3_LABEL
        sub = "trace_format" if not trace_valid else ("filler_invalid" if not filler_valid else
                                                      ("neither" if neither else None))
        if sub:
            v += " - " + C.DP3_SUB[sub]
    elif not any(failed_parts.values()):
        v = "GO" if filler_valid else C.INCONCLUSIVE_FILLER
    else:
        names = [f"{g} [{', '.join(failed_parts[g])}]" for g in G_NAMES if failed_parts[g]]
        if not filler_valid:
            names.append("V")
        v = "NO-GO (failed: " + "; ".join(names) + ")"
    d = lambda x, y, k: float(a[x][k] - a[y][k])          # noqa: E731
    dh = lambda x, y, k: float(h[x][k] - h[y][k])         # noqa: E731
    return {"verdict": v, "decision_point_3_stop": dp3, "dp3_sub_label": sub, "conditions": cond,
            "failed_parts": failed_parts,
            "validity": {"trace_valid": trace_valid, "filler_valid": filler_valid,
                         "format_ok": {s: {p: _r(float(f[s][p])) for p in PRIMS} for s in ("trace", "filler")}},
            "components": comps,
            "values": {"trace-plain|conditional": _r(d("trace", "plain", "primary_conditional")),
                       "trace-plain|cond_oor_val": _r(d("trace", "plain", "cond_oor_val")),
                       "trace-plain|cond_sametens": _r(d("trace", "plain", "cond_sametens")),
                       "trace-plain|cond_oor_edge": _r(d("trace", "plain", "cond_oor_edge")),
                       **{f"trace predicate|{s}": _r(float(a["trace"][pm])) for pm, s in PRED_METRICS.items()},
                       "trace-plain|mix": _r(d("trace", "plain", "primary_mix")),
                       "trace-plain|conditional HS": _r(dh("trace", "plain", "primary_conditional")),
                       "trace-plain|cond_oor_val HS": _r(dh("trace", "plain", "cond_oor_val")),
                       "trace-plain|cond_sametens HS": _r(dh("trace", "plain", "cond_sametens")),
                       "trace-plain|cond_oor_edge HS": _r(dh("trace", "plain", "cond_oor_edge")),
                       **{f"trace predicate|{s} HS": _r(float(h["trace"][pm])) for pm, s in PRED_METRICS.items()},
                       "trace-plain|mix HS": _r(dh("trace", "plain", "primary_mix")),
                       "trace-filler|conditional": _r(d("trace", "filler", "primary_conditional")),
                       "trace-filler|mix": _r(d("trace", "filler", "primary_mix")),
                       "filler-plain|conditional": _r(d("filler", "plain", "primary_conditional")),
                       "filler-plain|mix": _r(d("filler", "plain", "primary_mix")),
                       **{f"{s}-off|{c}": _r(d(s, "off", c)) for s in ("trace", "plain", "filler") for c in C.CONTROLS},
                       **{f"trace-plain|{c}": _r(d("trace", "plain", c)) for c in C.CONTROLS},
                       **{f"trace-plain|{k}": _r(d("trace", "plain", k)) for k in C.NONINF}}}


def flag_near_bar(rule: dict, contr: dict, contr_hs: dict) -> list:
    """Components whose value lies within one bootstrap SE of the bar (SE = CI width / 3.92). Reported, not gating. An
    absolute component (contrast 'trace') uses the bootstrap CI of the trace side's own metric (contr['trace'])."""
    flagged = []
    for c in rule["components"]:
        ci = (contr if c["surface"] == "in" else contr_hs)[c["contrast"]][c["metric"]]["ci95"]
        se = None if None in ci else (ci[1] - ci[0]) / (2 * Z95)
        c["se"] = _r(se, 6) if se is not None else None
        c["within_1se"] = se is not None and abs(float(Fraction(c["value"]) - Fraction(c["bar"]))) < se
        if c["within_1se"]:
            flagged.append(f"{c['condition']} {c['contrast']} {c['part']}")
    rule["within_1se_of_bar"] = flagged
    return flagged


def analyse_seed(dev: list, parents: list, texts: dict, gens: dict, hs: list, hs_texts: dict,
                 n_boot: int = N_BOOT) -> dict:
    D = DevIndex(dev)
    need(sorted(D.pos[h["id"]] for h in hs) == sorted(D.hs_rows.tolist()),
         "the held-out-surface rows are not exactly the dev rows of the held-out-surface strata")
    sides = {s: side_block(dev, texts[s], s, D) for s in C.SIDES}
    hsb = {s: hs_side_block(dev, hs, hs_texts[s], s, D) for s in C.SIDES}
    acc = {s: {m: metric_exact(sides[s]["x"], D, m) for m in METRICS} for s in C.SIDES}
    acc_hs = {s: {m: metric_exact(hsb[s]["x"], D, m) for m in HS_METRICS} for s in C.SIDES}
    for pm, s in PRED_METRICS.items():                     # revision 5: the trace side's predicate accuracy
        acc["trace"][pm] = metric_exact(sides["trace"]["x_pred"], D, s)
        acc_hs["trace"][pm] = metric_exact(hsb["trace"]["x_pred"], D, s)
    contr, contr_hs = {}, {}
    for a, b in CONTRASTS:
        contr[f"{a}-{b}"] = {m: {"diff": _r(float(acc[a][m] - acc[b][m])),
                                 "ci95": metric_boot_diff(sides[a]["x"], sides[b]["x"], D, m, n_boot)}
                             for m in METRICS}
        contr_hs[f"{a}-{b}"] = {m: {"diff": _r(float(acc_hs[a][m] - acc_hs[b][m])),
                                    "ci95": metric_boot_diff(hsb[a]["x"], hsb[b]["x"], D, m, n_boot)}
                                for m in HS_METRICS}
    zeros = np.zeros(len(dev))
    contr["trace"] = {pm: {"value": _r(float(acc["trace"][pm])),
                           "ci95": metric_boot_diff(sides["trace"]["x_pred"], zeros, D, s, n_boot)}
                      for pm, s in PRED_METRICS.items()}
    contr_hs["trace"] = {pm: {"value": _r(float(acc_hs["trace"][pm])),
                              "ci95": metric_boot_diff(hsb["trace"]["x_pred"], zeros, D, s, n_boot)}
                         for pm, s in PRED_METRICS.items()}
    audit = {}
    for s in C.SIDES:
        audit[s] = {}
        for m in HS_METRICS:
            i = D.idx[m]
            xi, xh = sides[s]["x"][i], hsb[s]["x"][i]
            audit[s][m] = {"in_surface": _r(float(acc[s][m])), "held_out": _r(float(acc_hs[s][m])),
                           "held_out_minus_in": _r(float(acc_hs[s][m] - acc[s][m])),
                           "ci95": metric_boot_diff(hsb[s]["x"], sides[s]["x"], D, m, n_boot),
                           "rows_right_in_only": int(((xi == 1) & (xh == 0)).sum()),
                           "rows_right_hs_only": int(((xi == 0) & (xh == 1)).sum())}
    rule = verdict({"acc": acc, "acc_hs": acc_hs,
                    "format": {s: sides[s]["format_exact"] for s in ("trace", "filler")}})
    flag_near_bar(rule, contr, contr_hs)
    res = {"onestep": {s: {k: v for k, v in sides[s].items() if k not in ("x", "x_pred", "scores", "format_exact")}
                       for s in C.SIDES},
           "onestep_heldout_surface": {s: {k: v for k, v in hsb[s].items() if k not in ("x", "x_pred")} for s in C.SIDES},
           "contrasts": contr, "contrasts_heldout_surface": contr_hs, "surface_audit": audit, "rule": rule,
           "filler_residual_covariate": residual_covariate(dev, D, sides["filler"]["x"], hsb["filler"]["x"]),
           "exact_metrics": {s: {m: str(v) for m, v in acc[s].items()
                                 if m in GROUPS + C.CONTROLS + C.G1_ALONE + tuple(PRED_METRICS)} for s in C.SIDES},
           "exact_metrics_heldout_surface": {s: {m: str(v) for m, v in acc_hs[s].items()} for s in C.SIDES}}
    res["chains"] = chain_block(parents, gens)
    return res


# ---------------------------------------------------------------- run-dir loading and validation
def load_run_outputs(run_dir: Path, dev: list, parents: list, hs: list) -> tuple:
    run_dir = Path(run_dir)
    texts, gens, hs_texts = {}, {}, {}
    chains = C.flat_chains(parents)
    for s in C.SIDES:
        rows = C.load_jsonl(run_dir / "onestep" / f"onestep_{s}.jsonl")
        need([r["id"] for r in rows] == [r["id"] for r in dev], f"onestep_{s}: ids differ from the dev bank")
        need(all(isinstance(r["text"], str) for r in rows), f"onestep_{s}: non-string text")
        texts[s] = [r["text"] for r in rows]
        rows = C.load_jsonl(run_dir / "onestep" / f"onestep_hs_{s}.jsonl")
        need([(r["id"], r["hs_index"]) for r in rows] == [(h["id"], h["hs_index"]) for h in hs],
             f"onestep_hs_{s}: ids / hs_index differ from the held-out-surface bank")
        need(all(isinstance(r["text"], str) for r in rows), f"onestep_hs_{s}: non-string text")
        hs_texts[s] = [r["text"] for r in rows]
        crow = C.load_jsonl(run_dir / "chains" / f"chains_{s}.jsonl")
        need([r["cand_id"] for r in crow] == [c["cand_id"] for c in chains], f"chains_{s}: cand_ids differ")
        arm = C.SIDE_ARM[s]
        for r, c in zip(crow, chains):
            st, tx = r["states"], r["texts"]
            need(len(st) == C.H + 1 and st[0] == env.dumps(c["s0"]), f"chains_{s} {c['cand_id']}: bad states")
            broken = False
            for t in range(C.H):
                if broken:
                    need(st[t + 1] is None and len(tx) <= t, f"chains_{s} {c['cand_id']}: continues after a break")
                    continue
                need(t < len(tx), f"chains_{s} {c['cand_id']}: missing text at step {t}")
                p = C.parse_after(tx[t], arm)
                need((None if p is None else env.dumps(p)) == st[t + 1],
                     f"chains_{s} {c['cand_id']}: state {t + 1} is not its text parsed by the {arm} parser")
                broken = p is None
        gens[s] = {r["cand_id"]: r for r in crow}
    return texts, gens, hs_texts


def provenance_check(run_dir: Path, registered: bool) -> dict:
    run_dir = Path(run_dir)
    import t2br_run as R                      # torch-free at import
    done = C.read_json(run_dir / "DONE.json")
    need(done is not None, f"{run_dir}: DONE.json missing (the run is not complete)")
    prov = C.read_json(run_dir / "provenance.json")
    need(prov is not None, "provenance.json unreadable")
    need(C.sha256(run_dir / "provenance.json") == done["provenance_sha256"], "provenance.json changed after DONE.json")
    need(set(done["files"]) == set(R.STAGE_ORDER), "DONE.json does not list every stage")
    for st in R.STAGE_ORDER:
        for rel, h in done["files"][st].items():
            need(C.sha256(run_dir / rel) == h, f"{rel} changed after it was recorded")
    a0 = C.read_json(run_dir / "anchor" / "anchor.json") or {}
    need(bool(a0.get("pass")), "anchor did not pass")
    need(all((t.get("anchor_recheck") or {"pass": True}).get("pass") for t in prov["turns"]), "an anchor re-check failed")
    info = {"seed": prov.get("seed"), "train_variant": prov.get("train_variant", "base"), "registered": prov.get("registered"),
            "turns": len(prov["turns"]), "anchor_identical_states": a0.get("identical_states"),
            "data_sha256": prov.get("data_sha256"),
            "train": {st: {k: prov["stages"][st].get(k) for k in ("final_loss_avg50", "supervised_tokens", "seconds")}
                      for st in R.TRAIN_STAGES}}
    if registered:
        need(prov.get("registered") is True, "the run was not a registered run")
        need(prov.get("train_variant") == C.TRAIN_VARIANT, f"the run did not train the '{C.TRAIN_VARIANT}' build")
        ok, errs, fz = R.verify_freeze(ROOT)
        need(ok, f"the freeze does not verify: {errs[:3]}")
        need(prov.get("freeze_sha256") == C.sha256(ROOT / R.FREEZE_REL), "the run used a different freeze record")
        for t in prov["turns"]:
            for rel, h in t["code_sha256"].items():
                need(fz["inputs"].get(rel) == h, f"turn code {rel} differs from the frozen file")
        for k, rel in (("dev", f"data/{C.DEV.name}"), ("dev_hs", f"data/{C.DEV_HS.name}"),
                       ("chains", f"data/{C.CHAINS.name}")):
            need(prov["data_sha256"][k] == fz["inputs"][rel], f"{k} differs from the freeze")
    return info


def results_stem(seeds: list, variant: str) -> str:
    """T2BR_RESULTS_<seeds>[_<variant>]: a surface-diversity run never collides with a canonical-build result."""
    return "T2BR_RESULTS_" + "_".join(seeds) + ("" if variant == "base" else f"_{variant}")


def analyse_runs(run_dirs: list, registered: bool = True, out_dir: Path | None = None, dev_path: Path = C.DEV,
                 chains_path: Path = C.CHAINS, dev_hs_path: Path = C.DEV_HS, n_boot: int = N_BOOT,
                 write: bool = True) -> dict:
    dev = C.load_dev(dev_path)
    hs = C.load_dev_hs(dev_hs_path)
    parents = C.load_chain_bank(chains_path)
    fcc = fill_constant_check(dev)
    need(fcc["ok"], f"amendment 4b constant-fill check failed on the dev bank: {fcc}")
    per_seed = {}
    variants = set()
    for rd in run_dirs:
        info = provenance_check(rd, registered)
        variants.add(info["train_variant"])
        texts, gens, hs_texts = load_run_outputs(rd, dev, parents, hs)
        res = analyse_seed(dev, parents, texts, gens, hs, hs_texts, n_boot)
        res["provenance"] = info
        per_seed[f"s{info['seed']}"] = res
    need(len(variants) == 1, f"run dirs mix training variants {sorted(variants)}")
    variant = variants.pop()
    seeds = sorted(per_seed)
    vs = {s: per_seed[s]["rule"]["verdict"] for s in seeds}
    overall = ("GO in every analysed seed" if all(v == "GO" for v in vs.values()) else "NOT GO in every analysed seed")
    if seeds == ["s0"]:
        overall = f"SCREEN (seed 0): {vs['s0']}"
    res = {"experiment": "RCWM-T1 T2b-register screen", "go_rule": C.GO_RULE_TEXT, "go_rule_revision": C.GO_RULE_REVISION,
           "registered": registered, "train_variant": variant, "seeds": seeds, "verdicts": vs, "overall": overall,
           "confirmation_note": "the roadmap's confirmation applies the same rule in all 3 seeds, then a sealed R bank",
           "fill_constant_check": fcc, "per_seed": per_seed, "dev_sha256": C.sha256(dev_path),
           "dev_hs_sha256": C.sha256(dev_hs_path), "chains_sha256": C.sha256(chains_path),
           "code_sha256": {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")},
           "torch_imported": "torch" in sys.modules, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write:
        out_dir = Path(out_dir or (ROOT / "runs" / "t2b_register" / "analysis"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = results_stem(seeds, variant)
        _write(out_dir / f"{stem}.json", json.dumps(clean(res), indent=1) + "\n", "x")
        _write(out_dir / f"{stem}.md", render(res), "x")
        print(f"wrote {out_dir / (stem + '.json')} and .md", flush=True)
    return res


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, Fraction):
        return str(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ci(c):
    return "[" + ", ".join("nan" if x is None else f"{x:+.3f}" for x in c) + "]"


def render(res: dict) -> str:
    L = [f"# T2b-register screen results ({', '.join(res['seeds'])}; training variant "
         f"{res.get('train_variant', 'sd')})", "", f"**Overall:** {res['overall']}", "",
         f"Pre-stated rule, revision {res.get('go_rule_revision', C.GO_RULE_REVISION)} (verbatim):", "",
         "> " + res["go_rule"], ""]
    for s in res["seeds"]:
        r = res["per_seed"][s]
        rule = r["rule"]
        L += [f"## Seed {s}: {rule['verdict']}", "", "| condition | holds | failed parts |", "|---|---|---|"]
        for k, v in rule["conditions"].items():
            g = k.split(" ", 1)[0]
            L.append(f"| {k} | {'yes' if v else 'NO'} | {', '.join(rule['failed_parts'].get(g, [])) or '-'} |")
        L += ["", f"Validity (in-surface): trace format_ok {rule['validity']['format_ok']['trace']}, filler format_ok "
                  f"{rule['validity']['format_ok']['filler']} (bar 0.95 on each primary).", "",
              "Trace predicate accuracy (G1's absolute clause, bar 0.80; strict parser, cmp keys x, thr, ok in order, "
              "boolean ok equal to the env's predicate): " + "; ".join(
                  f"{s} {rule['values'][f'trace predicate|{s}']:.3f} in-surface / "
                  f"{rule['values'][f'trace predicate|{s} HS']:.3f} held-out" for s in PRED_METRICS.values()) + ".", "",
              "| component | surface | contrast | outcome | value (exact) | bar | holds | SE | within 1 SE |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in rule["components"]:
            L.append(f"| {c['condition']} | {c['surface']} | {c['contrast']} | {c['metric']} | {c['value_float']:+.4f} "
                     f"({c['value']}) | {c['op']} {float(Fraction(c['bar'])):+.2f} | {'yes' if c['holds'] else 'NO'} | "
                     f"{c.get('se')} | {'FLAG' if c.get('within_1se') else ''} |")
        L += ["", "In-surface (Titan's verbatim frame):", "",
              "| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |",
              "|---|---|---|---|---|---|---|"]
        for m in (*GROUPS, *C.DEV_STRATA):
            vals = []
            for side in C.SIDES:
                v = r["onestep"][side]["metrics"].get(m)
                if v is None:
                    v = r["onestep"][side]["strata"][m]["acc"]
                vals.append(f"{v:.3f}")
            tp, tf = r["contrasts"]["trace-plain"][m], r["contrasts"]["trace-filler"][m]
            L.append(f"| {m} | " + " | ".join(vals) + f" | {tp['diff']:+.3f} {_ci(tp['ci95'])} | "
                     f"{tf['diff']:+.3f} {_ci(tf['ci95'])} |")
        L += ["", "Held-out surface (paraphrase 4 + field names 4 + held-out key orders), the primary rows and the "
                  "identification strata:", "",
              "| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |",
              "|---|---|---|---|---|---|---|"]
        for m in HS_METRICS:
            vals = [f"{r['onestep_heldout_surface'][side]['metrics'][m]:.3f}" for side in C.SIDES]
            tp, tf = r["contrasts_heldout_surface"]["trace-plain"][m], r["contrasts_heldout_surface"]["trace-filler"][m]
            L.append(f"| {m} | " + " | ".join(vals) + f" | {tp['diff']:+.3f} {_ci(tp['ci95'])} | "
                     f"{tf['diff']:+.3f} {_ci(tf['ci95'])} |")
        L += ["", "Surface audit (held-out minus in-surface, paired by row; rows right only in-surface / only held-out):",
              "", "| outcome | " + " | ".join(C.SIDES) + " |", "|---|" + "---|" * len(C.SIDES)]
        for m in HS_METRICS:
            cells = []
            for side in C.SIDES:
                a = r["surface_audit"][side][m]
                cells.append(f"{a['held_out_minus_in']:+.3f} {_ci(a['ci95'])} ({a['rows_right_in_only']}/"
                             f"{a['rows_right_hs_only']})")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", "Controls (a G4 loss shared by every trained arm is a property of the data: every training add wraps):", ""]
        L += [f"- {c}: " + ", ".join(f"{k.split('|')[0]} {rule['values'][k]:+.3f}" for k in rule["values"]
                                    if k.endswith("|" + c)) for c in C.CONTROLS]
        tf = r["onestep"]["trace"].get("trace_fields", {})
        L += ["", "Trace fields (trace side, in-surface):", "",
              "| stratum | format | predicate (pred_ok) | cmp_ok | ok_ok | calc_ok | after follows calc |",
              "|---|---|---|---|---|---|---|"]
        for st, f in tf.items():
            L.append(f"| {st} | {f.get('format_ok')} | {f.get('pred_ok', '-')} | {f.get('cmp_ok')} | {f.get('ok_ok', '-')} | "
                     f"{f.get('calc_ok')} | {f.get('after_follows_calc')} |")
        fc = r["filler_residual_covariate"]
        L += ["", "Filler residual covariate (reported, not gating; amendment 4b: the fill is constant per action kind, so "
                  "its length carries no answer information; residual = filler tokens - that row's trace tokens):", "",
              "| cell | n | residual mean / median / range | filler acc, residual <= median (n) | > median (n) | "
              "held-out: <= median | > median |", "|---|---|---|---|---|---|---|"]
        for cell, f in fc.items():
            L.append(f"| {cell} | {f['n']} | {f['residual_mean']} / {f['residual_median']} / {f['residual_range']} | "
                     f"{f['filler_acc_le_median']} ({f['n_le_median']}) | {f['filler_acc_gt_median']} "
                     f"({f['n_gt_median']}) | {f['filler_hs_acc_le_median']} | {f['filler_hs_acc_gt_median']} |")
        L += ["", "Branch choice read from 'after' (correct / wrong / neither), conditional strata (in-surface):", ""]
        for side in C.SIDES:
            br = r["onestep"][side]["branch_from_after"]
            L.append(f"- {side}: " + "; ".join(f"{k} {v['correct_branch']}/{v['wrong_branch']}/{v['neither_or_unparsed']}"
                                               for k, v in br.items()))
        L += ["", "Chains (k=4; normalised regret / exact; mean [95% CI]):", "",
              "| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |", "|---|---|---|---|---|---|"]
        for cs, blk in r["chains"].items():
            for side in C.SIDES:
                k4 = blk["k"]["4"]["sides"][side]
                pl = blk["product_law"][side]
                L.append(f"| {cs} | {side} | {k4['regret']['mean']:.3f} {_ci(k4['regret']['ci95'])} | "
                         f"{k4['exact']['mean']:.3f} | {pl['predicted_k4_exact_product']} | {pl['observed_k4_exact']} |")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- scorer self-test
def _neg_variants(row: dict) -> dict:
    """Malformed / wrong answers built from a row's golds, with the field each must fail."""
    cmp, calc, after = row["gold_cmp"], row["gold_calc"], json.loads(row["gold_after"])
    tr = row["targets"]["trace"]
    out = {
        "trace sorted keys": (json.dumps(json.loads(tr), separators=(",", ":"), sort_keys=True), "format"),
        "trace order calc,cmp,after": ('{"calc":' + C._dump(calc) + ',"cmp":' + C._dump(cmp) + ',"after":' + env.dumps(after) + "}", "format"),
        "trace extra key": (tr[:-1] + ',"note":1}', "format"),
        "trace duplicate after": (tr[:-1] + ',"after":' + env.dumps(after) + "}", "format"),
        "trace missing after": ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(calc) + "}", "format"),
        "trace truncated": (tr[: int(len(tr) * 0.8)], "format"),
        "trace + trailing text": (tr + " done", "format"),
        "plain text to trace parser": (row["targets"]["plain"], "format"),
        "filler text to trace parser": (row["targets"]["filler"], "format"),
    }
    if env.dumps(row["before"]) != row["gold_after"]:
        out["trace with copied after"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(calc) + ',"after":' +
                                          env.dumps(row["before"]) + "}", "after")
    if calc is not None:
        bad = dict(calc, mod=(calc["mod"] + 1) % C.MOD)
        out["trace wrong calc"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(bad) + ',"after":' + env.dumps(after) + "}",
                                   "calc")
        out["trace calc keys reordered"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' +
                                            C._dump({"mod": calc["mod"], "raw": calc["raw"]}) + ',"after":' + env.dumps(after) + "}",
                                            "calc")
    if cmp is not None:
        # revision 5 (the predicate clause): a cmp whose keys are not exactly x, thr, ok in that order states no
        # predicate (pred_ok false) although the strict top-level parser accepts the trace and its 'after' is right
        out["trace cmp keys reordered"] = ('{"cmp":' + C._dump({"thr": cmp["thr"], "x": cmp["x"], "ok": cmp["ok"]}) +
                                           ',"calc":' + C._dump(calc) + ',"after":' + env.dumps(after) + "}", "pred")
        out["trace cmp extra key"] = ('{"cmp":' + C._dump(dict(cmp, note=1)) + ',"calc":' + C._dump(calc) + ',"after":' +
                                      env.dumps(after) + "}", "pred")
        out["trace cmp without ok"] = ('{"cmp":' + C._dump({"x": cmp["x"], "thr": cmp["thr"]}) + ',"calc":' + C._dump(calc) +
                                       ',"after":' + env.dumps(after) + "}", "pred")
        flip = dict(cmp, ok=not cmp["ok"])
        out["trace flipped ok"] = ('{"cmp":' + C._dump(flip) + ',"calc":' + C._dump(calc) + ',"after":' + env.dumps(after) + "}",
                                   "cmp")
        out["trace ok as 1/0"] = ('{"cmp":' + C._dump(dict(cmp, ok=int(cmp["ok"]))) + ',"calc":' + C._dump(calc) +
                                  ',"after":' + env.dumps(after) + "}", "cmp")
    return out


def _filler_negs(row: dict) -> dict:
    f = row["targets"]["filler"]
    d = json.loads(f)
    after = env.dumps(json.loads(row["gold_after"]))
    return {"filler keys reversed": ('{"after":' + after + ',"fill":"' + d["fill"] + '"}', "format"),
            "filler extra key": (f[:-1] + ',"x":1}', "format"),
            "filler duplicate fill": ('{"fill":"' + d["fill"] + '","fill":"' + d["fill"] + '","after":' + after + "}",
                                      "format"),
            "filler unterminated fill (repetition loop)": ('{"fill":"' + C.FILL_UNIT * (row["fill_units"] + 40), "format"),
            "trace text to filler parser": (row["targets"]["trace"], "format"),
            "filler with a copied after": ('{"fill":"' + d["fill"] + '","after":' + env.dumps(row["before"]) + "}",
                                           "after")}


def _filler_other_lengths(row: dict) -> list:
    """Amendment 4b: the emitted fill's length is not scored. Fills of other lengths with the gold after must parse and
    score right (format_ok and after_ok)."""
    after = env.dumps(json.loads(row["gold_after"]))
    return ['{"fill":"' + C.FILL_UNIT * k + '","after":' + after + "}" for k in (1, max(1, row["fill_units"] - 1),
                                                                              row["fill_units"] + 3)]


def heldout_bank_check(dev: list, hs: list) -> dict:
    """The held-out-surface bank against the dev bank (data only): the primary rows in dev order, same ids and golds;
    every prompt decodes (t2br_common.read_prompt) to its row and uses ONLY held-out factors."""
    D = DevIndex(dev)
    bad = 0
    for j, h in enumerate(hs):
        r = dev[D.pos[h["id"]]] if h["id"] in D.pos else None
        try:
            d = C.read_prompt(h["messages"])
        except ValueError:
            bad += 1
            continue
        hp = C.surface_heldout_parts(d["surface"])
        bad += r is None or r["gold_after"] != h["gold_after"] or r["stratum"] != h["stratum"] \
            or env.dumps(d["before"]) != env.dumps(r["before"]) or env.dumps(d["event"]) != env.dumps(r["event"]) \
            or d["surface"] != C.heldout_surface(h["hs_index"]) or not hp["paraphrase"] \
            or hp["names"] != sorted(C.SD_FIELDS) or hp["orders"] != sorted(C.OBJECT_KEYS)
    order_ok = [h["id"] for h in hs] == [dev[j]["id"] for j in D.hs_rows]
    return {"n": len(hs), "bad_rows": bad, "primary_rows_in_dev_order": order_ok, "ok": bad == 0 and order_ok}


def scorer_selftest(dev_path: Path = C.DEV, chains_path: Path = C.CHAINS, dev_hs_path: Path = C.DEV_HS) -> dict:
    dev = C.load_dev(dev_path)
    hs = C.load_dev_hs(dev_hs_path)
    parents = C.load_chain_bank(chains_path)
    D = DevIndex(dev)
    res = {"golds": {}, "golds_heldout_surface": {}, "negatives": {}, "floors": {}, "metrics": {}, "chains": {}}
    for side in C.SIDES:
        sc = score_side(dev, [r["targets"][C.SIDE_ARM[side]] for r in dev], side)
        fields = ["after_ok", "parsed"] + (["format_ok", "cmp_ok", "calc_ok", "after_follows_calc"] if side == "trace" else []) \
            + (["format_ok"] if side == "filler" else [])
        res["golds"][side] = {f: _r(np.mean([s[f] for s in sc])) for f in fields}
        if side == "trace":
            cond = [i for i, r in enumerate(dev) if r["event"]["kind"] == "conditional"]
            res["golds"][side]["pred_ok|cond"] = _r(np.mean([sc[i]["pred_ok"] for i in cond]))
            res["golds"][side]["ok_ok|cond"] = _r(np.mean([sc[i]["ok_ok"] for i in cond]))
            res["golds"][side]["calc_follows_cmp|cond"] = _r(np.mean([sc[i]["calc_follows_cmp"] for i in cond]))
        if side == "filler":
            res["golds"][side]["fill_units_of == FILL_UNITS[kind]"] = _r(np.mean(
                [C.fill_units_of(r["targets"]["filler"]) == C.FILL_UNITS[r["event"]["kind"]] for r in dev]))
        hb = hs_side_block(dev, hs, [dev[D.pos[h["id"]]]["targets"][C.SIDE_ARM[side]] for h in hs], side, D)
        res["golds_heldout_surface"][side] = {m: hb["metrics"][m] for m in HS_METRICS}
        if side == "trace":
            res["golds_heldout_surface"][side].update({pm: metric_point(hb["x_pred"], D, s)
                                                       for pm, s in PRED_METRICS.items()})
    res["heldout_bank"] = heldout_bank_check(dev, hs)
    res["fill_constant"] = fill_constant_check(dev)
    neg = Counter()
    neg_n = Counter()
    neg_fail = []
    other_len_ok = other_len_n = 0
    for r in dev:
        for name, (text, must), arm in [(k, v, "trace") for k, v in _neg_variants(r).items()] + \
                [(k, v, "filler") for k, v in _filler_negs(r).items()]:
            s = C.score_text(r, text, arm)
            if must == "format":
                bad = s["format_ok"] or s["parsed"] or s.get("pred_ok", False)
            elif must == "after":
                bad = s["after_ok"]
            elif must == "calc":
                bad = s["calc_ok"] or not s["after_ok"]          # the after-only parser still reads the right after
            elif must == "pred":                             # revision 5: parsed, after right, but no stated predicate
                bad = s["pred_ok"] or s["cmp_ok"] or not s["after_ok"] or not s["format_ok"]
            else:                                            # cmp: wrong predicate field, after still read right
                bad = s["cmp_ok"] or s["pred_ok"] or (name == "trace flipped ok" and s["ok_ok"]) or not s["after_ok"]
            neg_n[name] += 1
            neg[name] += int(bool(bad))
            if bad and len(neg_fail) < 5:
                neg_fail.append((r["id"], name, text))
        for text in _filler_other_lengths(r):
            s = C.score_text(r, text, "filler")
            other_len_n += 1
            other_len_ok += bool(s["format_ok"] and s["after_ok"])
    res["negatives"] = {k: {"n": neg_n[k], "wrongly_accepted": neg[k]} for k in sorted(neg_n)}
    res["negative_failures"] = neg_fail
    res["filler_other_lengths_accepted"] = {"n": other_len_n, "accepted": other_len_ok}
    # floors: copy (never correct: every dev row changes the state) and branch heuristics on conditional strata
    for h in ("copy", "always_add", "always_sub", "unreduced"):
        fl = {}
        for s in C.DEV_STRATA:
            rs = [r for r in dev if r["stratum"] == s]
            app = [r for r in rs if C.heuristic_after(h, r["before"], r["event"]) is not None]
            if not app:
                continue
            got = np.mean([C.score_text(r, env.dumps(C.heuristic_after(h, r["before"], r["event"])), "plain")["after_ok"]
                           for r in app])
            if h == "copy":
                exp = np.mean([r["gold_after"] == env.dumps(r["before"]) for r in app])
            elif h == "always_add":
                exp = np.mean([bool(r["gold_ok"]) for r in app])
            elif h == "always_sub":
                exp = np.mean([not r["gold_ok"] for r in app])
            else:
                exp = np.mean([not C.wraps(r["before"], r["event"]) for r in app])
            fl[s] = {"scored": _r(got), "expected": _r(exp), "exact": abs(got - exp) < 1e-12}
        res["floors"][h] = fl
    # metric identities (exact rationals), in-surface and on the held-out-surface index
    ones = np.ones(len(dev))
    sub = np.array([C.score_text(r, env.dumps(C.heuristic_after("always_sub", r["before"], r["event"]) or r["before"]),
                                 "plain")["after_ok"] for r in dev], float)
    sub_hs = np.zeros(len(dev))
    sub_hs[D.hs_rows] = sub[D.hs_rows]
    res["metrics"] = {"perfect": {m: str(metric_exact(ones, D, m)) for m in GROUPS + C.G1_ALONE},
                      "always_sub": {m: str(metric_exact(sub, D, m)) for m in ("primary_conditional", "conditional_in")
                                     + C.G1_ALONE},
                      "always_sub_heldout_index": {m: str(metric_exact(sub_hs, D, m)) for m in ("primary_conditional",)
                                                   + C.G1_ALONE},
                      "boot_identical_sides": metric_boot_diff(ones, ones, D, "primary_conditional", 2000),
                      "n": {m: int(len(D.idx[m])) for m in GROUPS + C.G1_ALONE}}
    # chain pipeline identities: gold chains -> regret 0, exact 1, product law 1; copy chains -> the copy floor
    gold_gen = {}
    copy_gen = {}
    for ch in C.flat_chains(parents):
        g = env.rollout(FAM, ch["s0"], ch["events"])
        gold_gen[ch["cand_id"]] = {"states": [env.dumps(s) for s in g], "texts": [env.dumps(s) for s in g[1:]]}
        copy_gen[ch["cand_id"]] = {"states": [env.dumps(ch["s0"])] * (C.H + 1), "texts": [env.dumps(ch["s0"])] * C.H}
    rows = chain_rows_for(parents)
    pids, gr = regret(rows, 4, gold_gen)
    _, cr = regret(rows, 4, copy_gen)
    _, fr = floor_copy_regret(rows, 4)
    pl = product_law(parents, gold_gen)
    res["chains"] = {"gold_regret_k4_mean": _r(np.nanmean(gr)), "gold_exact_k4": _r(parent_exact(rows, 4, gold_gen, pids).mean()),
                     "copy_regret_equals_floor": bool(np.array_equal(cr, fr, equal_nan=True)),
                     "gold_product_law": {k: pl[k] for k in ("predicted_k4_exact_product", "observed_k4_exact")},
                     "n_parents": len(pids)}
    ok = all(v == 1.0 for side in res["golds"].values() for v in side.values()) \
        and all(v == 1.0 for side in res["golds_heldout_surface"].values() for v in side.values()) \
        and res["heldout_bank"]["ok"] and res["fill_constant"]["ok"] \
        and all(v["wrongly_accepted"] == 0 for v in res["negatives"].values()) \
        and res["filler_other_lengths_accepted"]["accepted"] == res["filler_other_lengths_accepted"]["n"] \
        and all(c["exact"] for fl in res["floors"].values() for c in fl.values()) \
        and all(v == "1" for v in res["metrics"]["perfect"].values()) \
        and res["metrics"]["always_sub"] == {"primary_conditional": "1/2", "conditional_in": "1/2", "cond_oor_val": "1/2",
                                             "cond_oor_edge": "1/2", "cond_sametens": "1/2"} \
        and res["metrics"]["always_sub_heldout_index"] == {"primary_conditional": "1/2", "cond_oor_val": "1/2",
                                                           "cond_oor_edge": "1/2", "cond_sametens": "1/2"} \
        and res["metrics"]["boot_identical_sides"] == [0.0, 0.0] \
        and res["chains"]["gold_regret_k4_mean"] == 0.0 and res["chains"]["gold_exact_k4"] == 1.0 \
        and res["chains"]["copy_regret_equals_floor"] \
        and res["chains"]["gold_product_law"] == {"predicted_k4_exact_product": 1.0, "observed_k4_exact": 1.0}
    res["ok"] = bool(ok)
    res["n_rows"] = len(dev)
    res["n_rows_heldout_surface"] = len(hs)
    res["n_negatives"] = int(sum(neg_n.values()))
    res["dev_sha256"] = C.sha256(dev_path)
    res["dev_hs_sha256"] = C.sha256(dev_hs_path)
    res["chains_sha256"] = C.sha256(chains_path)
    res["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")}
    return res


# ---------------------------------------------------------------- dry run (synthetic sides with known verdicts)
def _coin(row_key: str, salt: str) -> bool:
    return int(hashlib.sha256(f"{salt}|{row_key}".encode()).hexdigest()[:8], 16) % 2 == 0


def _answer(before: dict, ev: dict, how: str, salt: str) -> dict:
    """A synthetic model's next state. how: gold | offlike (unreduced arithmetic, coin-flip branch) | plainlike (mod
    right inside the training ranges, unreduced outside, coin-flip branch) | swapbad (gold except swap is copied) |
    worse_in (gold outside the training ranges, coin-flip branch and unreduced mix inside) | gold_mix (gold on mix,
    plainlike otherwise) | tens (gold arithmetic; the conditional branch by a COARSE comparison of the tens digits,
    other // 10 > threshold // 10: right on every far / out-of-range-threshold pair, wrong on most boundary pairs) |
    rule:<name> (gold arithmetic; the conditional branch by the rule DRY_RULES[name]: e.g. the verifier's shortcut
    tens_or_eq_plus1, right on EVERY primary conditional row, 1/2 on cond_sametens; revision 5: the off-by-one family
    obo_*) | sub_branch (revision 5: a plain arm EXACTLY at chance on every branch-balanced stratum: every conditional
    takes the subtract branch with exact arithmetic; plainlike on every other kind) | weak_oor (revision 5: a plain arm
    that is weak on the out-of-range arithmetic: every row outside the training ranges is answered wrong, the gold
    register + 1; sub_branch inside them).
    'Outside' = t2br_common.is_oor (value, threshold or conditional distance)."""
    key = env.dumps({"b": before, "e": ev})
    v = ev.get("value", 0)
    oor = C.is_oor(before, ev)
    if how == "weak_oor":
        if oor:
            b = dict(env.step(FAM, before, ev))
            b[ev["register"]] = (b[ev["register"]] + 1) % C.MOD
            return b
        how = "sub_branch"
    if how == "sub_branch":
        if ev["kind"] != "conditional":
            how = "plainlike"
        else:
            b = dict(before)
            b[ev["register"]] = (before[ev["register"]] - v) % C.MOD
            return b
    if how == "tens" or how.startswith("rule:"):
        if ev["kind"] != "conditional":
            return env.step(FAM, before, ev)
        b = dict(before)
        rule = DRY_RULES["tens_digit_gt" if how == "tens" else how[5:]]
        add = rule(before[ev["other"]], ev["threshold"])
        b[ev["register"]] = (before[ev["register"]] + (v if add else -v)) % C.MOD
        return b
    if how == "gold" or (how == "worse_in" and oor) or (how == "gold_mix" and ev["kind"] == "mix"):
        return env.step(FAM, before, ev)
    if how == "gold_mix":
        how = "plainlike"
    if how == "swapbad":
        return dict(before) if ev["kind"] == "swap" else env.step(FAM, before, ev)
    k, r = ev["kind"], ev["register"]
    b = dict(before)
    if k == "swap":
        return env.step(FAM, before, ev)
    if k == "conditional":
        add = _coin(key, salt)
        raw = before[r] + (v if add else -v)
    else:
        raw = C.raw_value(before, ev)
    reduce_ = how in ("plainlike",) and not oor or how == "worse_in" and k != "mix"
    b[r] = raw % C.MOD if reduce_ else raw
    return b


# Rules for the synthetic trace sides (Python floor // and %, as t2br_common.heuristic_after). 'true_rule_digitwise' is
# the env's strict predicate computed digit-wise (tens first, units on a tie); it equals other > threshold everywhere.
# Revision 5: the off-by-one family t2br_common.OFF_BY_ONE (the env's 'x > t' with the branch flipped exactly where
# x - t lies in the named subset of {-1, 0, +1}).
DRY_RULES = {
    "tens_digit_gt": lambda x, t: x // 10 > t // 10,
    "tens_or_eq_plus1": lambda x, t: x // 10 > t // 10 or x == t + 1,          # the verifier's shortcut
    "tens_ge_not_eq": lambda x, t: x // 10 >= t // 10 and x != t,              # its mirror (strict '>')
    "true_rule_digitwise": lambda x, t: x // 10 > t // 10 or (x // 10 == t // 10 and x % 10 > t % 10),
    **{n: (lambda x, t, s=s: (x > t) != (x - t in s)) for n, s in C.OFF_BY_ONE.items()},
}


def _branch_add(before: dict, ev: dict, st: dict):
    """Which branch a synthetic conditional answer applied: True = add, False = subtract (reduced or unreduced), None =
    neither (value != 0 mod 97 keeps the two apart)."""
    r, v = ev["register"], ev["value"]
    if any(st[k] != before[k] for k in C.REGS if k != r):
        return None
    y = st[r]
    if y in ((before[r] + v) % C.MOD, before[r] + v):
        return True
    if y in ((before[r] - v) % C.MOD, before[r] - v):
        return False
    return None


def _fmt(before: dict, ev: dict, st: dict, arm: str) -> str:
    """Render a synthetic next state in an arm's format. Trace: cmp carries the gold inputs (x, thr) and, revision 5,
    the predicate the synthetic answer FOLLOWED (ok = its branch; the gold ok when its answer is on neither branch), so
    the predicate clause scores the synthetic model's own comparison; calc from its own value. Filler: the amendment-4b
    constant fill of the event's kind."""
    if arm == "plain":
        return env.dumps(st)
    if arm == "filler_loop":                                       # a greedy ' . . .' loop that never closes
        return '{"fill":"' + C.FILL_UNIT * 70
    if arm == "filler":
        return C.filler_text(C.fill_units(ev["kind"]), st)
    cmp, calc, gold = C.parts(before, ev)
    if ev["kind"] != "swap" and env.dumps(st) != env.dumps(gold):    # a wrong synthetic answer: calc from its own value
        y = st[ev["register"]]
        calc = {"raw": y, "mod": y % C.MOD}
    if ev["kind"] == "conditional":
        br = _branch_add(before, ev, st)
        if br is not None and br != cmp["ok"]:
            cmp = dict(cmp, ok=br)
    tr = C.trace_text(cmp, calc, st)
    if arm == "trace":
        return tr
    if arm == "trace_sorted":
        return json.dumps(json.loads(tr), separators=(",", ":"), sort_keys=True)
    raise ValueError(arm)


def _arm_of(fmt: str) -> str:
    return {"trace_sorted": "trace", "filler_loop": "filler"}.get(fmt, fmt)


def _flip_ok(text: str) -> str:
    """A trace answer with its cmp.ok negated (everything else unchanged): wrong on the predicate clause only."""
    cmp, calc, after = C.parse_trace(text)
    return C.trace_text(dict(cmp, ok=not cmp["ok"]), calc, after)


_OFF, _PLAIN, _FILLER = ("offlike", "plain"), ("plainlike", "plain"), ("plainlike", "filler")
# revision 5: a plain arm EXACTLY at chance on every branch-balanced stratum (subtract branch, exact arithmetic) and a
# plain arm that is WEAK on every out-of-range row (always wrong there), each with a filler side of the same behaviour
_CHANCE, _CHANCE_FILLER = ("sub_branch", "plain"), ("sub_branch", "filler")
_WEAK, _WEAK_FILLER = ("weak_oor", "plain"), ("weak_oor", "filler")


def _obo_expected(n: str, plain: str) -> str:
    """The verdict an off-by-one trace arm must get, derived from the bank's design alone (not from a run): a flip set S
    of x - t in {-1, 0, +1} leaves cond_sametens (|d| >= 2), cond_oor_thr and cond_far untouched; on cond_oor_val (50
    rows at d = 0, 50 at d = +1) the trace scores 1 - |S & {0, 1}| / 2; on the pooled conditional 1 - 50 |S & {0, 1}| /
    300; on cond_oor_edge (20 rows at each of d = -1, 0, +1, +2) after AND predicate accuracy are 1 - |S| / 4 < 4/5 (so
    the predicate clause fails on BOTH surfaces for every S); on conditional_in (cond_xm1 / cond_x0 60 ok-false rows at d
    = -1 / 0, cond_xp1 60 ok-true at +1, cond_mid 50 + 50 at |d| 2..15) the branch-balanced mean is 1/2 (60 [-1 not in S]
    + 60 [0 not in S] + 50) / 170 + 1/2 (60 [1 not in S] + 50) / 110. plain 'chance' = exactly 1/2 on every G1 part and
    on conditional_in (a relative part holds iff the trace reaches 3/4); 'weak' = 0 on every out-of-range G1 part (it
    holds iff the trace reaches 1/4) and 1/2 on conditional_in."""
    s = set(C.OFF_BY_ONE[n])
    p = Fraction(1, 2) if plain == "chance" else Fraction(0)
    k01 = len(s & {0, 1})
    acc = {"conditional": 1 - Fraction(50 * k01, 300), "cond_oor_val": 1 - Fraction(k01, 2),
           "cond_oor_edge": 1 - Fraction(len(s), 4), "cond_sametens": Fraction(1)}
    rel = [m for m in ("conditional", "cond_oor_val", "cond_oor_edge", "cond_sametens") if acc[m] - p < C.GAIN_CONDITIONAL]
    pred = ["cond_oor_edge predicate"] if acc["cond_oor_edge"] < C.PRED_MIN else []
    order = ("conditional", "cond_oor_val", "cond_oor_edge", "cond_oor_edge predicate", "cond_sametens",
             "cond_sametens predicate")
    g1 = [x for x in order if x in rel + pred]
    parts = g1 + [x + " HS" for x in g1]
    ci = Fraction(1, 2) * Fraction(60 * (-1 not in s) + 60 * (0 not in s) + 50, 170) + \
        Fraction(1, 2) * Fraction(60 * (1 not in s) + 50, 110)
    out = "NO-GO (failed: G1 [" + ", ".join(parts) + "]"
    if ci - Fraction(1, 2) < -C.NONINF_MARGIN:
        out += "; G5 [conditional_in]"
    return out + ")"


SCENARIOS = {
    # name: ({side: (how, answer format[, how on the held-out surface])}, (match, expected verdict)); one coin per
    # scenario, shared by its sides and by both renderings. match: exact (the full verdict string).
    "go": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": _FILLER}, ("exact", "GO")),
    "filler_equals_trace": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": ("gold", "filler")},
                            ("exact", C.DP3_LABEL)),
    "nothing_helps": ({"off": _OFF, "plain": _PLAIN, "trace": ("plainlike", "trace"), "filler": _FILLER},
                      ("exact", C.DP3_LABEL + " - " + C.DP3_SUB["neither"])),
    "controls_lost": ({"off": ("gold", "plain"), "plain": _PLAIN, "trace": ("swapbad", "trace"), "filler": _FILLER},
                      ("exact", "NO-GO (failed: G4 [swap])")),
    # the trace side is gold out of range, so the predicate clauses hold; every relative G1 part fails (plain is gold)
    "in_dist_worse": ({"off": _OFF, "plain": ("gold", "plain"), "trace": ("worse_in", "trace"), "filler": ("offlike", "filler")},
                      ("exact", "NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_sametens, "
                                "conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_sametens HS]; G2 [mix, mix HS]; "
                                "G5 [conditional_in, mix_in])")),
    "g3_only": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": ("gold_mix", "filler")},
                ("exact", "NO-GO (failed: G3 [mix])")),
    "filler_invalid": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": ("plainlike", "filler_loop")},
                       ("exact", C.INCONCLUSIVE_FILLER)),
    "malformed_traces": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace_sorted"), "filler": _FILLER},
                         ("exact", C.DP3_LABEL + " - " + C.DP3_SUB["trace_format"])),
    # revision 3: a trace arm that is right in Titan's frame but falls back to the plain arm's behaviour on the held-out
    # surface (the PREREG-3 / BWM key-order failure): G1 and G2 fail on the held-out surface only (revision 5: its
    # held-out cmp.ok is the coin's, so both predicate clauses fail there too)
    "surface_locked": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace", "plainlike"), "filler": _FILLER},
                       ("exact", "NO-GO (failed: G1 [conditional HS, cond_oor_val HS, cond_oor_edge HS, cond_oor_edge "
                                 "predicate HS, cond_sametens HS, cond_sametens predicate HS]; G2 [mix HS])")),
    # revision 3: a coarse tens-digit comparison passes the pooled conditional (far and out-of-range-threshold pairs are
    # always right) but no single boundary / identification stratum (revision 5: plain exactly at chance)
    "coarse_comparison": ({"off": _OFF, "plain": _CHANCE, "trace": ("tens", "trace"), "filler": _CHANCE_FILLER},
                          ("exact", "NO-GO (failed: G1 [cond_oor_val, cond_oor_edge, cond_oor_edge predicate, "
                                    "cond_sametens, cond_sametens predicate, cond_oor_val HS, cond_oor_edge HS, "
                                    "cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS])")),
    # revision 4: the verifier's shortcut 'tens(other) > tens(threshold) or other == threshold + 1' is right on EVERY
    # primary conditional row; cond_sametens catches it (relative and predicate clauses: 1/2 there); on cond_oor_edge it
    # misses the same-decade +2 members (63/80: the relative clause holds, the predicate clause does not)
    "tens_or_eq_plus1": ({"off": _OFF, "plain": _CHANCE, "trace": ("rule:tens_or_eq_plus1", "trace"),
                          "filler": _CHANCE_FILLER},
                         ("exact", "NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens, cond_sametens predicate, "
                                   "cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS])")),
    # revision 4: its mirror for the strict '>' ('tens(other) >= tens(threshold) and other != threshold'): exactly 4/5 on
    # cond_oor_edge (on the predicate bar), 1/2 on cond_sametens
    "tens_ge_not_eq": ({"off": _OFF, "plain": _CHANCE, "trace": ("rule:tens_ge_not_eq", "trace"), "filler": _CHANCE_FILLER},
                       ("exact", "NO-GO (failed: G1 [cond_sametens, cond_sametens predicate, cond_sametens HS, "
                                 "cond_sametens predicate HS])")),
    # revision 4: the true rule computed digit-wise (tens, then units on a tie) passes, as 'go' does
    "true_rule_digitwise": ({"off": _OFF, "plain": _CHANCE, "trace": ("rule:true_rule_digitwise", "trace"),
                             "filler": _CHANCE_FILLER}, ("exact", "GO")),
    # revision 5, decision (1): a plain arm WEAK on the out-of-range arithmetic (0 on every out-of-range row) lets the
    # verifier's shortcut through every RELATIVE G1 part (revision 4's rule would say GO); only the predicate clauses
    # (no plain term) catch it
    "weak_plain_tens_or_eq_plus1": ({"off": _OFF, "plain": _WEAK, "trace": ("rule:tens_or_eq_plus1", "trace"),
                                     "filler": _WEAK_FILLER},
                                    ("exact", "NO-GO (failed: G1 [cond_oor_edge predicate, cond_sametens predicate, "
                                              "cond_oor_edge predicate HS, cond_sametens predicate HS])")),
    "weak_plain_tens_ge_not_eq": ({"off": _OFF, "plain": _WEAK, "trace": ("rule:tens_ge_not_eq", "trace"),
                                   "filler": _WEAK_FILLER},
                                  ("exact", "NO-GO (failed: G1 [cond_sametens predicate, cond_sametens predicate HS])")),
    "weak_plain_go": ({"off": _OFF, "plain": _WEAK, "trace": ("gold", "trace"), "filler": _WEAK_FILLER}, ("exact", "GO")),
    # revision 5, decision (2): every off-by-one variant, against the chance plain and against the weak plain; the
    # expected verdicts are derived from the bank's design (_obo_expected), not from a run
    **{n: ({"off": _OFF, "plain": _CHANCE, "trace": (f"rule:{n}", "trace"), "filler": _CHANCE_FILLER},
           ("exact", _obo_expected(n, "chance"))) for n in C.OFF_BY_ONE},
    **{f"weak_plain_{n}": ({"off": _OFF, "plain": _WEAK, "trace": (f"rule:{n}", "trace"), "filler": _WEAK_FILLER},
                           ("exact", _obo_expected(n, "weak"))) for n in C.OFF_BY_ONE},
}
RULE_SCENARIOS = ("go", "true_rule_digitwise", "tens_or_eq_plus1", "tens_ge_not_eq", "coarse_comparison",
                  "weak_plain_tens_or_eq_plus1", "weak_plain_tens_ge_not_eq", "weak_plain_go",
                  *C.OFF_BY_ONE, *(f"weak_plain_{n}" for n in C.OFF_BY_ONE))

# Engineered exact-boundary runs: per side, the number of WRONG answers per dev stratum (the first rows of the stratum
# in bank order; cond_xm1 rows are all ok-false), in-surface; '_hs' gives the held-out-surface counts (default: the
# in-surface counts of the held-out-surface strata). Revision 5: '_pred' (in-surface) and '_pred_hs' (held-out surface;
# default '_pred') give, per identification stratum, the number of trace answers whose cmp.ok is negated (the LAST rows
# of the stratum: wrong on the predicate clause only). The default side is all correct. Chains are gold for every side.
_BASE_PLAIN = {"cond_oor_val": 50, "cond_oor_thr": 50, "cond_far": 50, "mix_oor": 70, "cond_sametens": 50,
               "cond_oor_edge": 40}
_BASE_TRACE = {"cond_oor_val": 25, "cond_oor_thr": 25, "cond_far": 25, "mix_oor": 40, "swap": 4, "add_nowrap": 4,
               "cond_xm1": 17, "mix": 5, "cond_sametens": 25, "cond_oor_edge": 20,
               "_pred": {"cond_sametens": 20, "cond_oor_edge": 16}}
_BASE_FILLER = {"cond_oor_val": 35, "cond_oor_thr": 35, "cond_far": 35, "mix_oor": 50, "cond_sametens": 35,
                "cond_oor_edge": 28}
_BELOW_TRACE = {**_BASE_TRACE, "cond_oor_val": 26, "mix_oor": 41, "swap": 5, "cond_xm1": 18, "mix": 6, "cond_sametens": 26,
                "cond_oor_edge": 21, "_pred": {"cond_sametens": 21, "cond_oor_edge": 17}}
_HS_AT_BAR = {s: _BASE_TRACE[s] for s in C.HS_STRATA}
_PRED_AT_BAR = dict(_BASE_TRACE["_pred"])
ENGINEERED = {
    # every component exactly ON its bar on both surfaces: G1 +0.25 (conditional, cond_oor_val, cond_oor_edge and
    # cond_sametens) and predicate accuracy 4/5 (cond_oor_edge 64/80, cond_sametens 80/100), G2 +0.30, G3 +0.10 / +0.10,
    # G4 -0.05 / -0.05, G5 -0.05 / -0.05; DP3 does not fire (0.10 is not < 0.10). A float implementation fails G2 here
    # (0.6 - 0.3 < 0.3).
    "boundary_go": ({"plain": _BASE_PLAIN, "trace": _BASE_TRACE, "filler": _BASE_FILLER}, "GO"),
    # every G component one count past its bar on both surfaces (G3 on conditional only, so decision point 3 does not
    # fire); the cond_oor_val count also moves the pooled conditional past its bar
    "boundary_below": ({"plain": _BASE_PLAIN, "trace": _BELOW_TRACE, "filler": {**_BASE_FILLER, "mix_oor": 51}},
                       "NO-GO (failed: G1 [conditional, cond_oor_val, cond_oor_edge, cond_oor_edge predicate, "
                       "cond_sametens, cond_sametens predicate, conditional HS, cond_oor_val HS, cond_oor_edge HS, "
                       "cond_oor_edge predicate HS, cond_sametens HS, cond_sametens predicate HS]; G2 [mix, mix HS]; "
                       "G3 [conditional]; G4 [swap]; G5 [conditional_in, mix_in])"),
    # in-surface exactly on every bar; on the held-out surface cond_oor_val is one count past its bar while the pooled
    # conditional stays exactly on it (one more wrong on cond_oor_val, one fewer on cond_far)
    "boundary_hs_cond_oor_val": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_hs": {**_HS_AT_BAR, "cond_oor_val": 26,
                                                                                         "cond_far": 24}},
                                  "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_val HS])"),
    # revision 4: cond_sametens one count past its bar on the held-out surface only / in-surface only (it is not pooled
    # into 'conditional', so nothing else moves)
    "boundary_hs_cond_sametens": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_hs": {**_HS_AT_BAR, "cond_sametens": 26}},
                                   "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_sametens HS])"),
    "boundary_in_cond_sametens": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "cond_sametens": 26, "_hs": _HS_AT_BAR},
                                   "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_sametens])"),
    # revision 5: cond_oor_edge one count past its relative bar on the held-out surface only / in-surface only
    "boundary_hs_cond_oor_edge": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_hs": {**_HS_AT_BAR, "cond_oor_edge": 21}},
                                   "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_edge HS])"),
    "boundary_in_cond_oor_edge": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "cond_oor_edge": 21, "_hs": _HS_AT_BAR},
                                   "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_edge])"),
    # revision 5: each predicate clause one count below 4/5 on one surface only (every after answer unchanged)
    "boundary_in_pred_cond_oor_edge": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_pred": {**_PRED_AT_BAR, "cond_oor_edge": 17},
                                                                        "_pred_hs": _PRED_AT_BAR},
                                        "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_edge predicate])"),
    "boundary_hs_pred_cond_oor_edge": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE,
                                                                        "_pred_hs": {**_PRED_AT_BAR, "cond_oor_edge": 17}},
                                        "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_edge predicate HS])"),
    "boundary_in_pred_cond_sametens": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_pred": {**_PRED_AT_BAR, "cond_sametens": 21},
                                                                        "_pred_hs": _PRED_AT_BAR},
                                        "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_sametens predicate])"),
    "boundary_hs_pred_cond_sametens": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE,
                                                                        "_pred_hs": {**_PRED_AT_BAR, "cond_sametens": 21}},
                                        "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_sametens predicate HS])"),
    # trace - filler one count below 0.10 on both primaries: decision point 3 (filler - plain >= 0.10: no sub-label)
    "boundary_dp3": ({"plain": _BASE_PLAIN, "trace": _BASE_TRACE,
                      "filler": {"cond_oor_val": 35, "cond_oor_thr": 35, "cond_far": 34, "mix_oor": 49}}, C.DP3_LABEL),
}


def synth_outputs(dev: list, parents: list, hs: list, spec: dict, salt: str) -> tuple:
    texts, gens, hs_texts = {}, {}, {}
    pos = {r["id"]: r for r in dev}
    for side, sp in spec.items():
        how, fmt = sp[0], sp[1]
        how_hs = sp[2] if len(sp) > 2 else how
        arm = _arm_of(fmt)
        texts[side] = [_fmt(r["before"], r["event"], _answer(r["before"], r["event"], how, salt), fmt) for r in dev]
        hs_texts[side] = [_fmt(h["before"], h["event"], _answer(pos[h["id"]]["before"], pos[h["id"]]["event"], how_hs, salt),
                               fmt) for h in hs]
        g = {}
        for ch in C.flat_chains(parents):
            st, tx = [ch["s0"]], []
            for ev in ch["events"]:
                if st[-1] is None:
                    st.append(None)
                    continue
                t = _fmt(st[-1], ev, _answer(st[-1], ev, how, salt), fmt)
                tx.append(t)
                st.append(C.parse_after(t, arm))
            g[ch["cand_id"]] = {"cand_id": ch["cand_id"], "states": [None if s is None else env.dumps(s) for s in st],
                                "texts": tx}
        gens[side] = g
    return texts, gens, hs_texts


def _wrong_texts(rows: list, arm: str, wrong_n: dict, pred_n: dict | None = None) -> list:
    """wrong_n: per stratum, the FIRST rows answered wrong (parseable, the register + 1); pred_n (trace only): per
    stratum, the LAST rows whose cmp.ok is negated."""
    pred_n = pred_n or {}
    tot = Counter(r["stratum"] for r in rows)
    seen = Counter()
    tx = []
    for r in rows:
        st = env.step(FAM, r["before"], r["event"])
        seen[r["stratum"]] += 1
        if seen[r["stratum"]] <= wrong_n.get(r["stratum"], 0):
            st = dict(st)
            st[r["event"]["register"]] = (st[r["event"]["register"]] + 1) % C.MOD   # wrong, parseable
        t = _fmt(r["before"], r["event"], st, arm)
        if arm == "trace" and seen[r["stratum"]] > tot[r["stratum"]] - pred_n.get(r["stratum"], 0):
            t = _flip_ok(t)
        tx.append(t)
    return tx


def engineered_outputs(dev: list, parents: list, hs: list, plan: dict) -> tuple:
    texts, gens, hs_texts = {}, {}, {}
    pos = {r["id"]: r for r in dev}
    for side in C.SIDES:
        arm = C.SIDE_ARM[side]
        spec = dict(plan.get(side, {}))
        pred = spec.pop("_pred", {})
        pred_hs = spec.pop("_pred_hs", pred)
        hs_spec = spec.pop("_hs", {s: spec.get(s, 0) for s in C.HS_STRATA})
        texts[side] = _wrong_texts(dev, arm, spec, pred)
        hs_texts[side] = _wrong_texts([pos[h["id"]] for h in hs], arm, hs_spec, pred_hs)
        g = {}
        for ch in C.flat_chains(parents):
            gold = env.rollout(FAM, ch["s0"], ch["events"])
            g[ch["cand_id"]] = {"cand_id": ch["cand_id"], "states": [env.dumps(s) for s in gold],
                                "texts": [_fmt(gold[t], ev, gold[t + 1], arm) for t, ev in enumerate(ch["events"])]}
        gens[side] = g
    return texts, gens, hs_texts


def _acc_template(value_of) -> tuple:
    """(acc, acc_hs) with every metric of every side (and the trace side's predicate pseudo-metrics) set by
    value_of(side, metric)."""
    acc = {s: {m: value_of(s, m) for m in METRICS} for s in C.SIDES}
    acc_hs = {s: {m: value_of(s, m) for m in HS_METRICS} for s in C.SIDES}
    for pm in PRED_METRICS:
        acc["trace"][pm] = value_of("trace", pm)
        acc_hs["trace"][pm] = value_of("trace", pm)
    return acc, acc_hs


def boundary_checks(dev: list) -> dict:
    """Exhaustive exact-boundary checks of verdict() on the bank's own denominators (the held-out surface has the same
    denominators as the in-surface strata): for every count pair whose difference is EXACTLY a bar, the component
    must take the stated direction ('>=' holds; DP3's '<' does not fire), and one count past the bar the opposite; for
    the absolute predicate clause (revision 5) EVERY count k = 0..N is checked (holds iff k / N >= 4/5). Also counts how
    many of these a float implementation would misjudge."""
    D = DevIndex(dev)
    n = {m: len(D.idx[m]) for m in METRICS}
    ci = D.cls[D.idx["conditional_in"]]
    n0, n1 = int((ci == 0).sum()), int((ci == 1).sum())
    base, base_hs = _acc_template(lambda s, m: {"off": Fraction(1), "plain": Fraction(1, 2), "trace": Fraction(1),
                                                "filler": Fraction(0)}[s])
    fmt = {s: {p: Fraction(1) for p in PRIMS} for s in ("trace", "filler")}
    out = {}
    fbar = {"G1": 0.25, "G2": 0.30, "G3": 0.10, "G4": -0.05, "G5": -0.05, "DP3": 0.10}

    def run(cname, x, y, metric, pairs, at_bar_holds, key, surface="in"):
        bad = fl_bad = tot = 0
        for (ka, kb, ka2) in pairs:
            for kk, want in ((ka, at_bar_holds), (ka2, not at_bar_holds)):
                acc = {s: dict(v) for s, v in base.items()}
                acc_hs = {s: dict(v) for s, v in base_hs.items()}
                tgt = acc if surface == "in" else acc_hs
                tgt[x][metric], tgt[y][metric] = kk, kb
                if cname == "DP3":                     # the other primary far below the bar, so DP3 hinges on this one
                    other = "primary_mix" if metric == "primary_conditional" else "primary_conditional"
                    acc["trace"][other], acc["filler"][other] = Fraction(1, 2), Fraction(1, 2)
                    got = verdict({"acc": acc, "acc_hs": acc_hs, "format": fmt})["decision_point_3_stop"]
                else:
                    got = verdict({"acc": acc, "acc_hs": acc_hs, "format": fmt})["conditions"][CONDITION_NAMES[cname]]
                tot += 1
                bad += got != want
                if kk is ka:                           # what a float implementation would conclude at the bar
                    fv = float(kk) - float(kb)
                    fl_bad += ((fv < fbar[cname]) if cname == "DP3" else (fv >= fbar[cname])) != at_bar_holds
        out[key] = {"cases": tot, "wrong": bad, "at_bar_cases": len(pairs), "float_impl_would_misjudge": fl_bad}

    def run_abs(pm, N, key, surface):
        """The absolute predicate clause: every k = 0..N; holds iff k / N >= PRED_MIN (with every other component
        passing); the at-bar case is k = 4N/5."""
        bad = fl_bad = tot = at_bar = 0
        for k in range(N + 1):
            acc = {s: dict(v) for s, v in base.items()}
            acc_hs = {s: dict(v) for s, v in base_hs.items()}
            (acc if surface == "in" else acc_hs)["trace"][pm] = Fraction(k, N)
            got = verdict({"acc": acc, "acc_hs": acc_hs, "format": fmt})["conditions"][CONDITION_NAMES["G1"]]
            want = Fraction(k, N) >= C.PRED_MIN
            tot += 1
            bad += got != want
            if Fraction(k, N) == C.PRED_MIN:
                at_bar += 1
                fl_bad += ((k / N) >= 0.8) != want
        out[key] = {"cases": tot, "wrong": bad, "at_bar_cases": at_bar, "float_impl_would_misjudge": fl_bad}

    def mean_pairs(N, bar):
        """(a at the bar, b, a one count past the bar) as exact fractions of N."""
        res = []
        num = bar * N
        if num.denominator != 1:
            return res
        step = int(num)
        for kb in range(N + 1):
            ka = kb + step
            if 0 <= ka <= N and 0 <= ka - 1 <= N:
                res.append((Fraction(ka, N), Fraction(kb, N), Fraction(ka - 1, N)))
        return res
    for surf in ("in", "hs"):
        sfx = "" if surf == "in" else " HS"
        for m in ("primary_conditional",) + C.G1_ALONE:
            run("G1", "trace", "plain", m, mean_pairs(n[m], C.GAIN_CONDITIONAL), True, f"G1 {PART_LABEL[m]}{sfx}", surf)
        for pm, s in PRED_METRICS.items():
            run_abs(pm, n[s], f"G1 {PART_LABEL[pm]}{sfx}", surf)
        run("G2", "trace", "plain", "primary_mix", mean_pairs(n["primary_mix"], C.GAIN_MIX), True, "G2 mix" + sfx, surf)
    for p in PRIMS:
        run("G3", "trace", "filler", p, mean_pairs(n[p], C.FILLER_STOP), True, f"G3 {p}")
        # DP3: at exactly 0.10 it must NOT fire; one count below it fires
        run("DP3", "trace", "filler", p, mean_pairs(n[p], C.FILLER_STOP), False, f"DP3 {p}")
    for c in C.CONTROLS:
        run("G4", "trace", "off", c, mean_pairs(n[c], -C.CONTROL_LOSS), True, f"G4 {c}")
    run("G5", "trace", "plain", "mix_in", mean_pairs(n["mix_in"], -C.NONINF_MARGIN), True, "G5 mix_in")
    # G5 conditional_in (branch-balanced): every (k0, k1) count change whose metric change is exactly -0.05
    pairs = []
    for d0 in range(-n0, n0 + 1):
        for d1 in range(-n1, n1 + 1):
            if Fraction(d0, 2 * n0) + Fraction(d1, 2 * n1) == -C.NONINF_MARGIN:
                k0p, k1p = max(0, -d0), max(0, -d1)
                if k0p + d0 - 1 < 0:
                    k0p += 1
                if k0p <= n0 and k1p <= n1 and k0p + d0 <= n0 and k1p + d1 <= n1:
                    a = Fraction(k0p + d0, 2 * n0) + Fraction(k1p + d1, 2 * n1)
                    b = Fraction(k0p, 2 * n0) + Fraction(k1p, 2 * n1)
                    pairs.append((a, b, a - Fraction(1, 2 * n0)))
    run("G5", "trace", "plain", "conditional_in", pairs, True, "G5 conditional_in")
    out["ok"] = all(v["wrong"] == 0 and v["at_bar_cases"] > 0 for k, v in out.items() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------- shortcut enumeration (revisions 4-5)
_CMP = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
G1_ENUM_METRICS = ("primary_conditional",) + C.G1_ALONE
ENUM_DOMAIN = {"other": [0, 96], "threshold": [C.OOR_THR[0][0], C.OOR_THR[1][1]]}   # every (other, threshold) of the bank
OBO_SHIFTS = tuple(range(-3, 4))                    # off-by-one family bases 'x > t + c'
POINTS = (("t-1", -1), ("t", 0), ("t+1", 1))       # the modifier points x == t - 1 / t / t + 1


def _modifier_name(states) -> str:
    return ", ".join(("or x==" if s == 1 else "and x!=") + p for (p, _), s in zip(POINTS, states) if s)


def digit_predicates(x, t) -> dict:
    """Every predicate of the enumerated family, as a boolean array over the (other, threshold) arrays x, t ('add iff
    ...'). Digits use floor semantics (numpy // and % equal Python's, also for negative thresholds). Family:
      atoms        TRUE, FALSE, tens(x) OP tens(t), units(x) OP units(t)         OP in > >= < <= == !=   (14)
      bases        every atom; tens(x) OP1 tens(t) AND/OR units(x) OP2 units(t) (72); the lexicographic form
                   tens(x) OP1 tens(t) or (tens(x) == tens(t) and units(x) OP2 units(t)) (36); the off-by-one family
                   x > t + c for c = -3..+3 (7: x>t-2 = x >= t-1, x>t-1 = x >= t, x>t, x>t+1, and the off-by-two /
                   -three shifts)
      modifiers    at each of x == t-1, x == t, x == t+1 independently: unchanged / forced true ('or x==p') / forced
                   false ('and x!=p'): 3^3 = 27 (revision 4 had the 9 on t and t+1 only)
    3,483 predicates in all (many coincide). With the base x>t the modifiers give the env's rule and every off-by-one
    variant (each non-empty flip set of x - t in {-1, 0, +1})."""
    x, t = np.asarray(x), np.asarray(t)
    tx, tt, ux, ut = x // 10, t // 10, x % 10, t % 10
    atoms = {"TRUE": np.ones(x.shape, bool), "FALSE": np.zeros(x.shape, bool)}
    for o, f in _CMP.items():
        atoms[f"tens(x){o}tens(t)"] = f(tx, tt)
    for o, f in _CMP.items():
        atoms[f"units(x){o}units(t)"] = f(ux, ut)
    bases = dict(atoms)
    for o1, f1 in _CMP.items():
        for o2, f2 in _CMP.items():
            bases[f"tens(x){o1}tens(t) and units(x){o2}units(t)"] = f1(tx, tt) & f2(ux, ut)
            bases[f"tens(x){o1}tens(t) or units(x){o2}units(t)"] = f1(tx, tt) | f2(ux, ut)
    for o1, f1 in _CMP.items():
        for o2, f2 in _CMP.items():
            bases[f"tens(x){o1}tens(t) or (tens(x)==tens(t) and units(x){o2}units(t))"] = f1(tx, tt) | ((tx == tt) & f2(ux, ut))
    for c in OBO_SHIFTS:
        bases["x>t" + (f"{c:+d}" if c else "")] = x > t + c
    eq = {p: x == t + k for p, k in POINTS}
    out = {}
    for states in itertools.product((0, 1, 2), repeat=3):          # 0 unchanged, 1 forced true, 2 forced false
        mn = _modifier_name(states)
        for bn, b in bases.items():
            v = b.copy()
            for (p, _), s in zip(POINTS, states):
                if s == 1:
                    v = v | eq[p]
                elif s == 2:
                    v = v & ~eq[p]
            out[bn + (" | " + mn if mn else "")] = v
    return out


def obo_enum_name(n: str) -> str:
    """The enumeration's name of an off-by-one variant: base x>t, each flipped point forced to the opposite of the
    env's value there (t-1, t: forced true; t+1: forced false)."""
    s = C.OFF_BY_ONE[n]
    return "x>t | " + _modifier_name(tuple((1 if k <= 0 else 2) if k in s else 0 for _, k in POINTS))


def shortcut_enumeration(dev: list, hs: list) -> dict:
    """Which simple digit shortcuts and which variants of the comparison would still pass G1 (revision 5), and which
    passed revision 4's G1 but are now caught. A synthetic trace side answers every conditional with the predicate's
    branch and exact arithmetic, and states that predicate in cmp.ok (so after_ok == pred_ok == the branch is right:
    value != 0 mod 97 makes the two branches differ); the G1 components (relative AND predicate clauses) are then
    decided by verdict() itself, exactly, with every G2-G5 component held at a passing value. Three plain baselines:
    'chance' = exactly 1/2 on every relative G1 part (a relative part holds iff the shortcut reaches 3/4),
    'dryrun_plainlike' = the dry run's coin-flip plain (unreduced outside the training ranges; lower on cond_oor_val,
    so more permissive) and 'weak_plain' = 0 on every relative G1 part (the most permissive: a relative part holds iff
    the shortcut reaches 1/4, so only the predicate clauses can catch it). The held-out surface uses the same rows
    (paired by row), so its parts equal the in-surface ones; both are decided. Predicates equal to the true rule on the
    whole domain (other 0..96, threshold -40..136) compute the comparison and are listed apart. For every predicate
    the set of distances d = other - threshold at which it differs from the rule on the domain is recorded when it is
    a function of d (the comparison family)."""
    D = DevIndex(dev)
    cond_rows = np.array([j for j, r in enumerate(dev) if r["event"]["kind"] == "conditional"])
    x = np.array([dev[j]["before"][dev[j]["event"]["other"]] for j in cond_rows])
    t = np.array([dev[j]["event"]["threshold"] for j in cond_rows])
    gold = np.array([bool(dev[j]["gold_ok"]) for j in cond_rows])
    preds = digit_predicates(x, t)
    X, T = np.meshgrid(np.arange(ENUM_DOMAIN["other"][0], ENUM_DOMAIN["other"][1] + 1),
                       np.arange(ENUM_DOMAIN["threshold"][0], ENUM_DOMAIN["threshold"][1] + 1), indexing="ij")
    Xr, Tr = X.ravel(), T.ravel()
    dom = digit_predicates(Xr, Tr)
    truth_dom = Xr > Tr
    dist_dom = Xr - Tr
    hs_pos = np.array([D.pos[h["id"]] for h in hs])
    # baselines (exact Fractions per relative G1 part; in-surface and held-out surface)
    plain_x = np.array([float(C.score_text(r, env.dumps(_answer(r["before"], r["event"], "plainlike", "shortcut_enum")),
                                           "plain")["after_ok"]) for r in dev])
    plain_hs = np.zeros(len(dev))
    plain_hs[hs_pos] = plain_x[hs_pos]
    base = {"chance": ({m: Fraction(1, 2) for m in G1_ENUM_METRICS}, {m: Fraction(1, 2) for m in G1_ENUM_METRICS}),
            "dryrun_plainlike": ({m: metric_exact(plain_x, D, m) for m in G1_ENUM_METRICS},
                                 {m: metric_exact(plain_hs, D, m) for m in G1_ENUM_METRICS}),
            "weak_plain": ({m: Fraction(0) for m in G1_ENUM_METRICS}, {m: Fraction(0) for m in G1_ENUM_METRICS})}
    other_metrics = ("primary_mix", "swap", "add_nowrap", "conditional_in", "mix_in")

    def g1_failed(tr_in: dict, tr_hs: dict, bl: str) -> list:
        p_in, p_hs = base[bl]
        acc = {"off": {m: Fraction(1) for m in other_metrics}, "plain": {m: Fraction(1, 2) for m in other_metrics},
               "trace": {m: Fraction(1) for m in other_metrics}, "filler": {m: Fraction(0) for m in other_metrics}}
        acc_hs = {s: {"primary_mix": v["primary_mix"]} for s, v in acc.items()}
        for s in C.SIDES:                    # filler exactly 0.10 below trace: G3 holds, DP3 cannot fire (mix)
            for m in G1_ENUM_METRICS:
                acc[s][m] = {"trace": tr_in[m], "plain": p_in[m], "filler": tr_in[m] - C.FILLER_STOP}.get(s, Fraction(0))
                acc_hs[s][m] = {"trace": tr_hs[m], "plain": p_hs[m]}.get(s, Fraction(0))
        for pm in PRED_METRICS:              # the synthetic trace states its own predicate: pred_ok == after_ok
            acc["trace"][pm], acc_hs["trace"][pm] = tr_in[pm], tr_hs[pm]
        fmt = {s: {p: Fraction(1) for p in PRIMS} for s in ("trace", "filler")}
        v = verdict({"acc": acc, "acc_hs": acc_hs, "format": fmt})
        others = [g for g in ("G2", "G3", "G4", "G5") if v["failed_parts"][g]]
        if others:
            raise AssertionError(f"enumeration harness: {others} should hold")
        return v["failed_parts"]["G1"]

    rev5_only = {p + sfx for p in ("cond_oor_edge", "cond_oor_edge predicate", "cond_sametens predicate")
                 for sfx in ("", " HS")}
    rev4_new = {"cond_sametens", "cond_sametens HS"}
    rows_out, classes = [], {}
    for name, b in preds.items():
        right = (b == gold).astype(float)
        xv = np.zeros(len(dev))
        xv[cond_rows] = right
        xh = np.zeros(len(dev))
        xh[hs_pos] = xv[hs_pos]
        tr_in = {m: metric_exact(xv, D, m) for m in G1_ENUM_METRICS}
        tr_hs = {m: metric_exact(xh, D, m) for m in G1_ENUM_METRICS}
        for pm, s in PRED_METRICS.items():
            tr_in[pm], tr_hs[pm] = tr_in[s], tr_hs[s]
        wrong_d = dom[name] != truth_dom
        flips = sorted({int(d) for d in np.unique(dist_dom[wrong_d])}) if wrong_d.any() else []
        # a function of d alone iff, at every flipped distance, it is wrong on the WHOLE domain row of that distance
        d_only = all(bool(wrong_d[dist_dom == d].all()) for d in flips)
        rec = {"name": name, "acc": {m: str(v) for m, v in tr_in.items()}, "acc_hs": {m: str(v) for m, v in tr_hs.items()},
               "true_rule_on_domain": not wrong_d.any(), "flip_distances": flips if d_only else None}
        for bl in base:
            f = g1_failed(tr_in, tr_hs, bl)
            rec[f"g1_failed_{bl}"] = f
            rec[f"passes_g1_rev5_{bl}"] = not f
            rec[f"passed_g1_rev4_{bl}"] = set(f) <= rev5_only
            rec[f"passed_g1_rev3_{bl}"] = set(f) <= rev5_only | rev4_new
        rows_out.append(rec)
        classes.setdefault((b.tobytes(), rec["true_rule_on_domain"]), []).append(rec)

    def summarize(sel) -> list:
        out = []
        for (_, eq), recs in classes.items():
            hit = [r for r in recs if sel(r)]
            if hit:
                rep = min(hit, key=lambda r: (len(r["name"]), r["name"]))
                out.append({"representative": rep["name"], "n_equivalent_on_dev_rows": len(hit),
                            "aliases": sorted(r["name"] for r in hit if r is not rep)[:6], "acc": rep["acc"],
                            "acc_hs": rep["acc_hs"], "true_rule_on_domain": eq,
                            "flip_distances": rep["flip_distances"]})
        return sorted(out, key=lambda c: (c["true_rule_on_domain"], c["representative"]))
    res = {"family": digit_predicates.__doc__.split("Family:", 1)[1].strip(), "n_predicates": len(preds),
           "n_distinct_on_dev_conditional_rows": len({k[0] for k in classes}), "domain": ENUM_DOMAIN,
           "baselines": {"chance": {m: "1/2" for m in G1_ENUM_METRICS},
                         "dryrun_plainlike": {"in": {m: str(v) for m, v in base["dryrun_plainlike"][0].items()},
                                              "hs": {m: str(v) for m, v in base["dryrun_plainlike"][1].items()}},
                         "weak_plain": {m: "0" for m in G1_ENUM_METRICS}},
           "decision": "verdict() on exact counts; G1 parts conditional / cond_oor_val / cond_oor_edge / cond_sametens "
                       "(trace - plain) and the predicate accuracy on cond_oor_edge / cond_sametens, both surfaces"}
    for bl in base:
        res[f"passes_g1_rev5_{bl}"] = {
            "shortcuts": summarize(lambda r, bl=bl: r[f"passes_g1_rev5_{bl}"] and not r["true_rule_on_domain"]),
            "true_rule_equivalents": summarize(lambda r, bl=bl: r[f"passes_g1_rev5_{bl}"] and r["true_rule_on_domain"])}
        res[f"caught_by_rev5_{bl}"] = summarize(
            lambda r, bl=bl: r[f"passed_g1_rev4_{bl}"] and not r[f"passes_g1_rev5_{bl}"])
        res[f"counts_{bl}"] = {
            "passed_rev3_g1": sum(r[f"passed_g1_rev3_{bl}"] for r in rows_out),
            "passed_rev4_g1": sum(r[f"passed_g1_rev4_{bl}"] for r in rows_out),
            "passed_rev4_g1_not_true_rule": sum(r[f"passed_g1_rev4_{bl}"] and not r["true_rule_on_domain"]
                                                for r in rows_out),
            "pass_rev5_g1": sum(r[f"passes_g1_rev5_{bl}"] for r in rows_out),
            "pass_rev5_g1_not_true_rule": sum(r[f"passes_g1_rev5_{bl}"] and not r["true_rule_on_domain"]
                                              for r in rows_out),
            "pass_rev5_g1_not_true_rule_classes": len(res[f"passes_g1_rev5_{bl}"]["shortcuts"]),
            "caught_by_rev5": sum(r[f"passed_g1_rev4_{bl}"] and not r[f"passes_g1_rev5_{bl}"] for r in rows_out),
            "off_by_one_family_passing": sum(r[f"passes_g1_rev5_{bl}"] and not r["true_rule_on_domain"]
                                             and r["flip_distances"] is not None
                                             and set(r["flip_distances"]) <= {-1, 0, 1} for r in rows_out)}
    # reported, NOT gating and NOT in the bank (DESIGN.md section 14): what a cross-digit identification stratum would
    # add. Its rows: the in-range domain (other and threshold 0..96) in DIFFERENT decades, |other - threshold| >= 2,
    # whose units comparison disagrees with the tens comparison (sign of the units difference != sign of the tens
    # difference); a survivor's branch-balanced accuracy there (the true rule scores 1)
    tdx, tdt = Xr // 10, Tr // 10
    cross = (Tr >= 0) & (Tr <= 96) & (tdx != tdt) & (np.abs(Xr - Tr) >= 2) & \
        (np.sign(Xr % 10 - Tr % 10) != np.sign(tdx - tdt))
    c_t, c_f = cross & truth_dom, cross & ~truth_dom

    def cross_acc(name: str) -> Fraction:
        w = dom[name] == truth_dom
        return Fraction(int(w[c_t].sum()), 2 * int(c_t.sum())) + Fraction(int(w[c_f].sum()), 2 * int(c_f.sum()))
    surv = sorted({c["representative"] for bl in base for c in res[f"passes_g1_rev5_{bl}"]["shortcuts"]})
    sacc = {n: cross_acc(n) for n in surv}
    res["cross_digit_option"] = {
        "definition": "other, threshold 0..96; tens(other) != tens(threshold); |other - threshold| >= 2; sign(units(other) "
                      "- units(threshold)) != sign(tens(other) - tens(threshold)); branch-balanced accuracy",
        "rows": int(cross.sum()), "ok_true": int(c_t.sum()), "ok_false": int(c_f.sum()),
        "true_rule_acc": str(cross_acc("tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t))")),
        "survivor_acc": {n: str(v) for n, v in sacc.items()},
        "survivors": len(surv), "survivors_below_4/5": sum(v < C.PRED_MIN for v in sacc.values()),
        "survivors_at_most_1/2": sum(v <= Fraction(1, 2) for v in sacc.values()),
        "survivors_not_caught": sorted(n for n, v in sacc.items() if v >= C.PRED_MIN)}
    by_name = {r["name"]: r for r in rows_out}
    res["named"] = {k: by_name[v] for k, v in (("tens_or_eq_plus1", "tens(x)>tens(t) | or x==t+1"),
                                               ("tens_ge_not_eq", "tens(x)>=tens(t) | and x!=t"),
                                               ("tens_digit_gt", "tens(x)>tens(t)"),
                                               ("true_rule_digitwise",
                                                "tens(x)>tens(t) or (tens(x)==tens(t) and units(x)>units(t))"))}
    # the off-by-one family (decision 2): per variant, the share of each G1-gated stratum's rows that contradicts it,
    # its G1 failures under every baseline
    G1_STRATA = {"conditional": C.PRIMARY["conditional"], **{s: (s,) for s in C.G1_ALONE}}
    obo = {}
    for n in C.OFF_BY_ONE:
        r = by_name[obo_enum_name(n)]
        rule = DRY_RULES[n]
        share = {}
        for g, ss in G1_STRATA.items():
            rows = [q for q in dev if q["stratum"] in ss]
            k = sum(rule(q["before"][q["event"]["other"]], q["event"]["threshold"]) != bool(q["gold_ok"]) for q in rows)
            share[g] = str(Fraction(k, len(rows)))
        obo[n] = {"form": C.OFF_BY_ONE_FORMS[n], "enumeration_name": r["name"], "flip_distances": r["flip_distances"],
                  "contradicted_share": share,
                  "contradicted_ge_1/6_on_a_g1_stratum": any(Fraction(v) >= C.RULE_VARIANT_MIN_CONTRA
                                                             for v in share.values()),
                  "acc": r["acc"], **{f"g1_failed_{bl}": r[f"g1_failed_{bl}"] for bl in base},
                  "fails_g1_under_every_baseline": all(r[f"g1_failed_{bl}"] for bl in base)}
        res["named"][n] = r
    res["off_by_one"] = obo
    return res


def dryrun(n_boot: int = 2000) -> dict:
    dev = C.load_dev()
    hs = C.load_dev_hs()
    parents = C.load_chain_bank()
    out = {"scenarios": {}, "engineered": {}, "n_boot": n_boot}
    go_res = None
    comps_of = {}
    g1_keys = G1_ENUM_METRICS + tuple(PRED_METRICS)
    for name, (spec, want) in SCENARIOS.items():
        t0 = time.time()
        texts, gens, hs_texts = synth_outputs(dev, parents, hs, spec, name)
        res = analyse_seed(dev, parents, texts, gens, hs, hs_texts, n_boot)
        v = res["rule"]["verdict"]
        hit = v == want[1]
        comps_of[name] = {(c["condition"], c["part"]): c["holds"] for c in res["rule"]["components"]}
        out["scenarios"][name] = {"expected": list(want), "verdict": v, "ok": hit,
                                  "failed_parts": res["rule"]["failed_parts"],
                                  "values": res["rule"]["values"], "validity": res["rule"]["validity"],
                                  "seconds": round(time.time() - t0, 1),
                                  "trace_format_ok_mean": _r(np.mean([f["format_ok"] for f in
                                                                      res["onestep"]["trace"]["trace_fields"].values()])),
                                  "surface_audit_trace": {m: res["surface_audit"]["trace"][m]["held_out_minus_in"]
                                                          for m in HS_METRICS},
                                  "product_law_ALL_trace": {k: res["chains"]["ALL"]["product_law"]["trace"][k]
                                                            for k in ("predicted_k4_exact_product", "observed_k4_exact")},
                                  "regret_k4_ALL": {s: res["chains"]["ALL"]["k"]["4"]["sides"][s]["regret"]["mean"]
                                                    for s in C.SIDES},
                                  "exact_g1_trace": {m: res["exact_metrics"]["trace"][m] for m in g1_keys},
                                  "exact_g1_trace_hs": {m: res["exact_metrics_heldout_surface"]["trace"][m]
                                                        for m in g1_keys},
                                  "exact_g1_plain": {m: res["exact_metrics"]["plain"][m] for m in G1_ENUM_METRICS},
                                  "exact_g1_plain_hs": {m: res["exact_metrics_heldout_surface"]["plain"][m]
                                                        for m in G1_ENUM_METRICS}}
        print(f"  [{'PASS' if hit else 'FAIL'}] {name}: {v}", flush=True)
        if name == "go":
            go_res = res
    for name, (plan, want) in ENGINEERED.items():
        t0 = time.time()
        texts, gens, hs_texts = engineered_outputs(dev, parents, hs, plan)
        res = analyse_seed(dev, parents, texts, gens, hs, hs_texts, n_boot)
        v = res["rule"]["verdict"]
        out["engineered"][name] = {"expected": want, "verdict": v, "ok": v == want,
                                   "components": [{k: c[k] for k in ("condition", "surface", "part", "contrast", "metric",
                                                                      "value", "bar", "holds", "within_1se")}
                                                  for c in res["rule"]["components"]],
                                   "within_1se_of_bar": res["rule"]["within_1se_of_bar"],
                                   "seconds": round(time.time() - t0, 1)}
        print(f"  [{'PASS' if v == want else 'FAIL'}] engineered {name}: {v}", flush=True)
    # extra properties
    go = out["scenarios"]["go"]
    bnd = boundary_checks(dev)
    out["boundary_checks"] = bnd
    eg = out["engineered"]["boundary_go"]
    cc, sl = comps_of["coarse_comparison"], comps_of["surface_locked"]
    t0 = time.time()
    enum = shortcut_enumeration(dev, hs)
    enum["seconds"] = round(time.time() - t0, 1)
    sc = out["scenarios"]
    for n in C.OFF_BY_ONE:                                   # the full-pipeline verdicts of the off-by-one scenarios
        enum["off_by_one"][n]["dryrun_verdict_chance_plain"] = sc[n]["verdict"]
        enum["off_by_one"][n]["dryrun_verdict_weak_plain"] = sc[f"weak_plain_{n}"]["verdict"]
    out["shortcut_enumeration"] = enum
    rev3_parts = [("G1", p) for p in ("conditional", "cond_oor_val", "conditional HS", "cond_oor_val HS")]
    rev4_parts = rev3_parts + [("G1", "cond_sametens"), ("G1", "cond_sametens HS")]
    rev5_parts = [("G1", c["part"]) for c in eg["components"] if c["condition"] == "G1"]
    sametens_parts = [("G1", p + s) for p in ("cond_sametens", "cond_sametens predicate") for s in ("", " HS")]

    def only_g1_fails(name):
        fp = sc[name]["failed_parts"]
        return sc[name]["verdict"].startswith("NO-GO") and fp["G1"] and not any(fp[g] for g in ("G2", "G3", "G4"))

    def rev4_g1_holds(name):
        return all(comps_of[name][p] for p in rev4_parts)
    rule_map = (("tens_or_eq_plus1", "tens_or_eq_plus1"), ("tens_ge_not_eq", "tens_ge_not_eq"),
                ("tens_digit_gt", "coarse_comparison"), ("true_rule_digitwise", "true_rule_digitwise"),
                *((n, n) for n in C.OFF_BY_ONE))
    fast_ok = all(enum["named"][k]["acc"] == sc[s]["exact_g1_trace"] and enum["named"][k]["acc_hs"] == sc[s]["exact_g1_trace_hs"]
                  for k, s in rule_map)
    obo = enum["off_by_one"]
    out["checks"] = {
        "tens_or_eq_plus1 (the verifier's shortcut, plain at chance): every revision-3 G1 part holds (conditional, "
        "cond_oor_val; both surfaces: the identification gap); both cond_sametens clauses fail on both surfaces (trace "
        "exactly 1/2 there, after and predicate)":
            all(comps_of["tens_or_eq_plus1"][p] for p in rev3_parts)
            and not any(comps_of["tens_or_eq_plus1"][p] for p in sametens_parts)
            and all(sc["tens_or_eq_plus1"][k]["cond_sametens"] == sc["tens_or_eq_plus1"][k]["cond_sametens_pred"] == "1/2"
                    for k in ("exact_g1_trace", "exact_g1_trace_hs")),
        "tens_ge_not_eq (its mirror for the strict '>'): likewise, both cond_sametens clauses fail on both surfaces":
            all(comps_of["tens_ge_not_eq"][p] for p in rev3_parts)
            and not any(comps_of["tens_ge_not_eq"][p] for p in sametens_parts),
        "decision (1), weak plain (0 on every out-of-range row): the verifier's shortcut and its mirror hold EVERY "
        "relative G1 part and every revision-4 component (revision 4's rule would say GO); the cond_sametens predicate "
        "clause fails both on both surfaces; a gold trace side is still GO":
            all(rev4_g1_holds(n) and only_g1_fails(n) and not sc[n]["failed_parts"]["G5"]
                and all(comps_of[n][("G1", p)] for p in ("conditional", "cond_oor_val", "cond_oor_edge", "cond_sametens",
                                                          "conditional HS", "cond_oor_val HS", "cond_oor_edge HS",
                                                          "cond_sametens HS"))
                and not comps_of[n][("G1", "cond_sametens predicate")]
                and not comps_of[n][("G1", "cond_sametens predicate HS")]
                for n in ("weak_plain_tens_or_eq_plus1", "weak_plain_tens_ge_not_eq"))
            and sc["weak_plain_go"]["verdict"] == "GO",
        "decision (2): every off-by-one variant is contradicted on >= 1/6 of the rows of at least one G1-gated stratum, "
        "and its full dry-run scenario is NO-GO through G1 against the chance plain AND against the weak plain (the "
        "cond_oor_edge predicate clause fails on both surfaces in every case)":
            all(obo[n]["contradicted_ge_1/6_on_a_g1_stratum"] for n in C.OFF_BY_ONE)
            and all(only_g1_fails(s) and not comps_of[s][("G1", "cond_oor_edge predicate")]
                    and not comps_of[s][("G1", "cond_oor_edge predicate HS")]
                    for n in C.OFF_BY_ONE for s in (n, f"weak_plain_{n}")),
        "decision (2): every off-by-one variant fails G1 in the enumeration under all three plain baselines (chance, "
        "dry-run plain, weak plain); no member of the off-by-one family passes":
            all(obo[n]["fails_g1_under_every_baseline"] for n in C.OFF_BY_ONE)
            and all(enum[f"counts_{bl}"]["off_by_one_family_passing"] == 0 for bl in ("chance", "dryrun_plainlike",
                                                                                         "weak_plain")),
        "the true rule (gold, and computed digit-wise) passes every G1 part incl. both identification strata and both "
        "predicate clauses on both surfaces (GO)":
            all(comps_of[n][p] for n in ("go", "true_rule_digitwise") for p in rev5_parts)
            and sc["go"]["verdict"] == sc["true_rule_digitwise"]["verdict"] == "GO",
        "shortcut enumeration: the fast path (branch right <=> after right <=> predicate right) equals the analysis' "
        "exact metrics of the rule scenarios (incl. the 7 off-by-one variants) on every G1 part, both surfaces": fast_ok,
        "shortcut enumeration: the verifier's shortcut and its mirror passed revision 3's G1 and fail revision 5's under "
        "every plain baseline": all(enum["named"][k][f"passed_g1_rev3_{bl}"] and not enum["named"][k][f"passes_g1_rev5_{bl}"]
                                    for k in ("tens_or_eq_plus1", "tens_ge_not_eq")
                                    for bl in ("chance", "dryrun_plainlike", "weak_plain")),
        "go: gold trace side product law predicted == observed == 1": go["product_law_ALL_trace"] ==
        {"predicted_k4_exact_product": 1.0, "observed_k4_exact": 1.0},
        "go: gold trace side regret k4 == 0": go["regret_k4_ALL"]["trace"] == 0.0,
        "malformed: every sorted-key trace rejected (format_ok 0; trace side invalid)":
            out["scenarios"]["malformed_traces"]["trace_format_ok_mean"] == 0.0
            and not out["scenarios"]["malformed_traces"]["validity"]["trace_valid"],
        "filler_invalid: filler side invalid, trace side valid": not out["scenarios"]["filler_invalid"]["validity"]["filler_valid"]
            and out["scenarios"]["filler_invalid"]["validity"]["trace_valid"],
        "go: trace fields of a gold trace all 1.0": all(v == 1.0 for f in go_res["onestep"]["trace"]["trace_fields"].values()
                                                        for v in f.values()),
        "coarse_comparison: the POOLED conditional passes on both surfaces; every G1 part on a single boundary / "
        "identification stratum fails (relative and predicate)": cc[("G1", "conditional")] and cc[("G1", "conditional HS")]
            and not any(cc[("G1", p + s)] for p in ("cond_oor_val", "cond_oor_edge", "cond_oor_edge predicate",
                                                     "cond_sametens", "cond_sametens predicate") for s in ("", " HS"))
            and out["scenarios"]["coarse_comparison"]["values"]["trace-plain|conditional"] >= 0.25,
        "surface_locked: every in-surface G1 / G2 part holds; every held-out-surface part fails (incl. both predicate "
        "clauses); the trace side's surface audit shows the loss": all(sl[p] for p in rev5_parts if not p[1].endswith(" HS"))
            and sl[("G2", "mix")]
            and not any(sl[p] for p in rev5_parts + [("G2", "mix HS")] if p[1].endswith(" HS"))
            and all(v < 0 for v in out["scenarios"]["surface_locked"]["surface_audit_trace"].values()),
        "go: a gold trace side has zero surface-audit loss": all(
            v == 0.0 for v in out["scenarios"]["go"]["surface_audit_trace"].values()),
        "boundary_go: every component value equals its bar exactly (both surfaces, incl. both predicate clauses)": all(
            Fraction(c["value"]) == Fraction(c["bar"]) for c in eg["components"]) and len(eg["components"]) == len(COMPONENTS),
        "boundary_go: every component (all at their bar) is flagged within 1 SE": all(
            c["within_1se"] for c in eg["components"]) and len(eg["within_1se_of_bar"]) == len(COMPONENTS),
        "verdict() exact-boundary checks (exhaustive on the bank's denominators, both surfaces; every count of the "
        "predicate clauses)": bnd["ok"],
        "verdict() refuses floats": _refuses_floats(),
        "results stem carries the training variant": results_stem(["s0"], "base") == "T2BR_RESULTS_s0"
            and results_stem(["s0"], "sd") == "T2BR_RESULTS_s0_sd",
        "amendment 4b: the dev bank's fills are one constant per kind with zero information about the answer":
            fill_constant_check(dev)["ok"],
    }
    for k, v in out["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    # multi-seed overall label (pure function of verdicts)
    vs = {"s0": "GO", "s1": "NO-GO (failed: G1 [conditional])"}
    out["checks"]["multi-seed: one NO-GO seed -> not GO in every seed"] = not all(v == "GO" for v in vs.values())
    out["ok"] = all(s["ok"] for s in out["scenarios"].values()) and all(s["ok"] for s in out["engineered"].values()) \
        and all(out["checks"].values())
    out["n_scenarios"] = len(out["scenarios"]) + len(out["engineered"])
    out["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")}
    out["dev_sha256"] = C.sha256(C.DEV)
    out["dev_hs_sha256"] = C.sha256(C.DEV_HS)
    out["go_rule_revision"] = C.GO_RULE_REVISION
    out["example_report_md"] = render_dryrun_summary(out) + "\n" + render(
        {"seeds": ["s0"], "train_variant": C.TRAIN_VARIANT,
         "overall": "SCREEN (seed 0): " + go_res["rule"]["verdict"] + " [SYNTHETIC DRY RUN]",
         "go_rule": C.GO_RULE_TEXT, "go_rule_revision": C.GO_RULE_REVISION, "per_seed": {"s0": clean(go_res)}})
    return out


def render_dryrun_summary(out: dict) -> str:
    """The dry run's own summary (scenario verdicts, engineered boundary runs, the off-by-one family, the shortcut
    enumeration)."""
    L = [f"# T2b-register dry run, rule revision {C.GO_RULE_REVISION} (synthetic sides; NOT model output)", "",
         "| scenario | expected | verdict | ok |", "|---|---|---|---|"]
    for k, v in out["scenarios"].items():
        L.append(f"| {k} | {v['expected'][1]} | {v['verdict']} | {'yes' if v['ok'] else 'NO'} |")
    for k, v in out["engineered"].items():
        L.append(f"| engineered {k} | {v['expected']} | {v['verdict']} | {'yes' if v['ok'] else 'NO'} |")
    L += ["", "G1 parts of the rule scenarios (exact; plain = the scenario's plain side: 'chance' = exactly 1/2 on every "
              "G1 part, 'weak_plain' = 0 on every out-of-range row, 'go' = the coin-flip plain):", "",
          "| scenario | trace conditional | cond_oor_val | cond_oor_edge | cond_oor_edge predicate | cond_sametens | "
          "cond_sametens predicate | plain cond_oor_val / cond_oor_edge / cond_sametens |", "|---|---|---|---|---|---|---|---|"]
    for k in RULE_SCENARIOS:
        e, p = out["scenarios"][k]["exact_g1_trace"], out["scenarios"][k]["exact_g1_plain"]
        L.append(f"| {k} | {e['primary_conditional']} | {e['cond_oor_val']} | {e['cond_oor_edge']} | "
                 f"{e['cond_oor_edge_pred']} | {e['cond_sametens']} | {e['cond_sametens_pred']} | "
                 f"{p['cond_oor_val']} / {p['cond_oor_edge']} / {p['cond_sametens']} |")
    en = out["shortcut_enumeration"]
    L += ["", "Off-by-one family (decision 2): share of each G1-gated stratum's rows that contradicts the variant, and "
              "its verdict in the full dry run:", "",
          "| variant | form | conditional | cond_oor_val | cond_oor_edge | cond_sametens | verdict, chance plain | "
          "verdict, weak plain |", "|---|---|---|---|---|---|---|---|"]
    for n, o in en["off_by_one"].items():
        s = o["contradicted_share"]
        L.append(f"| {n} | `{o['form']}` | {s['conditional']} | {s['cond_oor_val']} | {s['cond_oor_edge']} | "
                 f"{s['cond_sametens']} | {o['dryrun_verdict_chance_plain']} | {o['dryrun_verdict_weak_plain']} |")
    L += ["", f"Shortcut enumeration: {en['n_predicates']} predicates ({en['n_distinct_on_dev_conditional_rows']} "
              "distinct on the dev conditional rows), G1 decided by verdict() on exact counts, both surfaces.", ""]
    for bl in ("chance", "dryrun_plainlike", "weak_plain"):
        c = en[f"counts_{bl}"]
        L += [f"- plain baseline `{bl}`: passed revision-4 G1 {c['passed_rev4_g1']} ({c['passed_rev4_g1_not_true_rule']} "
              f"not the true rule); pass revision-5 G1 {c['pass_rev5_g1']}, of which NOT the true rule on the domain "
              f"{c['pass_rev5_g1_not_true_rule']} in {c['pass_rev5_g1_not_true_rule_classes']} classes; caught by "
              f"revision 5 {c['caught_by_rev5']}; off-by-one family members passing {c['off_by_one_family_passing']}."]
        for x in en[f"passes_g1_rev5_{bl}"]["shortcuts"]:
            L.append(f"  - still passes: `{x['representative']}` (+{x['n_equivalent_on_dev_rows'] - 1} equivalent on the "
                     f"dev rows; flips at d = {x['flip_distances']}); acc {x['acc']}")
    co = en["cross_digit_option"]
    L += ["", f"Cross-digit option (reported, NOT in the bank, NOT gating): on the {co['rows']} cross-digit rows of the "
              f"in-range domain ({co['ok_true']} ok-true / {co['ok_false']} ok-false; {co['definition']}) the true rule "
              f"scores {co['true_rule_acc']}; of the {co['survivors']} surviving class representatives (all baselines) "
              f"{co['survivors_below_4/5']} score below 4/5 and {co['survivors_at_most_1/2']} at most 1/2; not caught "
              f"there: {co['survivors_not_caught']}."]
    L.append("")
    return "\n".join(L) + "\n"


def _refuses_floats() -> bool:
    fmt = {s: {p: Fraction(1) for p in PRIMS} for s in ("trace", "filler")}
    ok = True
    a_float, h_exact = _acc_template(lambda s, m: 0.5)[0], _acc_template(lambda s, m: Fraction(1, 2))[1]
    a_exact, h_float = _acc_template(lambda s, m: Fraction(1, 2))[0], _acc_template(lambda s, m: 0.5)[1]
    for a, h in ((a_float, h_exact), (a_exact, h_float)):
        try:
            verdict({"acc": a, "acc_hs": h, "format": fmt})
            ok = False
        except TypeError:
            pass
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--dryrun", action="store_true")
    g.add_argument("--run-dir", action="append")
    ap.add_argument("--exploratory", action="store_true", help="analyse an unregistered run (labelled as such)")
    a = ap.parse_args()
    frozen = (ROOT / "docs" / "t2b" / "register" / "SCREEN_FREEZE.json").exists()
    if a.selftest:
        res = scorer_selftest()
        print(json.dumps({k: res[k] for k in ("golds", "golds_heldout_surface", "heldout_bank", "metrics", "chains",
                                              "filler_other_lengths_accepted", "ok", "n_rows", "n_negatives")}, indent=1))
        if frozen:
            print("SCREEN_FREEZE.json exists: SCORER_SELFTEST.json is a frozen input and is not rewritten")
        else:
            DOCS.mkdir(parents=True, exist_ok=True)
            _write(DOCS / "SCORER_SELFTEST.json", json.dumps(clean(res), indent=1) + "\n")
        return 0 if res["ok"] else 1
    if a.dryrun:
        res = dryrun()
        md = res.pop("example_report_md")
        if frozen:
            print("SCREEN_FREEZE.json exists: DRYRUN.json is a frozen input and is not rewritten")
        else:
            DOCS.mkdir(parents=True, exist_ok=True)
            _write(DOCS / "DRYRUN.json", json.dumps(clean(res), indent=1) + "\n")
            _write(DOCS / "DRYRUN.md", "<!-- synthetic dry run: NOT model output -->\n" + md)
        print(f"dry run {'PASS' if res['ok'] else 'FAIL'}")
        return 0 if res["ok"] else 1
    res = analyse_runs([ROOT / d if not Path(d).is_absolute() else Path(d) for d in a.run_dir], registered=not a.exploratory)
    print(res["overall"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
