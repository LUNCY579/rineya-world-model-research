"""T2b-register screen analysis, revision 3 (RCWM roadmap item 3; joint item 5b). CPU only: numpy / json; never
imports torch, never loads a model. Applies the PRE-STATED go rule (t2br_common.GO_RULE_TEXT, revision 3; verbatim in
docs/t2b/register/GO_RULE.md and DESIGN.md section 7).

  python -B scripts/t2br_analysis.py --selftest    # scorer self-test on the dev bank and its held-out-surface rendering
                                                   # (golds 100%, negatives 0%, floors exact, the 4b constant-fill check,
                                                   # metric and chain-pipeline identities); writes
                                                   # docs/t2b/register/SCORER_SELFTEST.json
  python -B scripts/t2br_analysis.py --dryrun      # synthetic scenarios with known verdicts (GO, decision-point-3 stops,
                                                   # each NO-GO condition, G3 alone, invalid filler, malformed traces, a
                                                   # surface-locked trace arm, a coarse tens-digit comparison, engineered
                                                   # exact-boundary runs incl. the held-out surface) + exhaustive
                                                   # exact-boundary checks of verdict(); writes DRYRUN.json + DRYRUN.md
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
SURFACE (E1H: the primary rows in the combined held-out surface, data/t2br_dev_hs.jsonl).
Metrics. 'conditional' (PRIMARY) = mean exact over cond_oor_val + cond_oor_thr + cond_far (branch-balanced pairs);
'cond_oor_val' = the boundary pairs alone (G1's second clause); 'mix' (PRIMARY) = mix_oor; 'conditional_in' =
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
HS_METRICS = ("primary_conditional", "cond_oor_val", "cond_oor_thr", "cond_far", "primary_mix", "mix_oor")
assert all(set(METRICS[m][1]) <= set(C.HS_STRATA) for m in HS_METRICS)
PART_LABEL = {"primary_conditional": "conditional", "cond_oor_val": "cond_oor_val", "primary_mix": "mix",
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
        fields = ("format_ok", "cmp_ok", "ok_ok", "calc_ok", "after_follows_calc", "calc_follows_cmp") if arm == "trace" \
            else ("format_ok",)
        tf = {}
        for s in C.DEV_STRATA:
            i = D.idx[s]
            tf[s] = {f: _r(np.mean([sc[j][f] for j in i])) for f in fields
                     if not (f in ("ok_ok", "calc_follows_cmp") and not s.startswith("cond"))}
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
    sc, states = {}, {}
    for h, t in zip(hs, texts):
        j = D.pos[h["id"]]
        s = C.score_text(dev[j], t, arm)
        sc[j] = s
        x[j] = float(s["after_ok"])
        if dev[j]["event"]["kind"] == "conditional":
            states[j] = C.parse_after(t, arm)
    out = {"x": x, "strata": {}, "metrics": {}}
    for s in C.HS_STRATA:
        i = D.idx[s]
        k = int(x[i].sum())
        cell = {"n": len(i), "acc": _r(k / len(i)), "wilson95": wilson(k, len(i)),
                "parsed": _r(np.mean([sc[j]["parsed"] for j in i]))}
        if arm in ("trace", "filler"):
            cell["format_ok"] = _r(np.mean([sc[j]["format_ok"] for j in i]))
        if arm == "trace":
            for f in ("cmp_ok", "calc_ok", "after_follows_calc"):
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
COMPONENTS = (
    ("G1", "trace", "plain", "primary_conditional", C.GAIN_CONDITIONAL, ">=", "in"),
    ("G1", "trace", "plain", "cond_oor_val", C.GAIN_CONDITIONAL, ">=", "in"),
    ("G1", "trace", "plain", "primary_conditional", C.GAIN_CONDITIONAL, ">=", "hs"),
    ("G1", "trace", "plain", "cond_oor_val", C.GAIN_CONDITIONAL, ">=", "hs"),
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
    "G1": "G1 trace-plain >= +0.25 on conditional (OOR pooled) and on cond_oor_val alone, in-surface and held-out surface",
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
        val = src[x][k] - src[y][k]
        holds = val >= bar if op == ">=" else val < bar
        comps.append({"condition": name, "surface": surf, "part": part_label(k, surf), "contrast": f"{x}-{y}",
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
                       "trace-plain|mix": _r(d("trace", "plain", "primary_mix")),
                       "trace-plain|conditional HS": _r(dh("trace", "plain", "primary_conditional")),
                       "trace-plain|cond_oor_val HS": _r(dh("trace", "plain", "cond_oor_val")),
                       "trace-plain|mix HS": _r(dh("trace", "plain", "primary_mix")),
                       "trace-filler|conditional": _r(d("trace", "filler", "primary_conditional")),
                       "trace-filler|mix": _r(d("trace", "filler", "primary_mix")),
                       "filler-plain|conditional": _r(d("filler", "plain", "primary_conditional")),
                       "filler-plain|mix": _r(d("filler", "plain", "primary_mix")),
                       **{f"{s}-off|{c}": _r(d(s, "off", c)) for s in ("trace", "plain", "filler") for c in C.CONTROLS},
                       **{f"trace-plain|{c}": _r(d("trace", "plain", c)) for c in C.CONTROLS},
                       **{f"trace-plain|{k}": _r(d("trace", "plain", k)) for k in C.NONINF}}}


def flag_near_bar(rule: dict, contr: dict, contr_hs: dict) -> list:
    """Components whose value lies within one bootstrap SE of the bar (SE = CI width / 3.92). Reported, not gating."""
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
         "the held-out-surface rows are not exactly the primary dev rows")
    sides = {s: side_block(dev, texts[s], s, D) for s in C.SIDES}
    hsb = {s: hs_side_block(dev, hs, hs_texts[s], s, D) for s in C.SIDES}
    acc = {s: {m: metric_exact(sides[s]["x"], D, m) for m in METRICS} for s in C.SIDES}
    acc_hs = {s: {m: metric_exact(hsb[s]["x"], D, m) for m in HS_METRICS} for s in C.SIDES}
    contr, contr_hs = {}, {}
    for a, b in CONTRASTS:
        contr[f"{a}-{b}"] = {m: {"diff": _r(float(acc[a][m] - acc[b][m])),
                                 "ci95": metric_boot_diff(sides[a]["x"], sides[b]["x"], D, m, n_boot)}
                             for m in METRICS}
        contr_hs[f"{a}-{b}"] = {m: {"diff": _r(float(acc_hs[a][m] - acc_hs[b][m])),
                                    "ci95": metric_boot_diff(hsb[a]["x"], hsb[b]["x"], D, m, n_boot)}
                                for m in HS_METRICS}
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
    res = {"onestep": {s: {k: v for k, v in sides[s].items() if k not in ("x", "scores", "format_exact")}
                       for s in C.SIDES},
           "onestep_heldout_surface": {s: {k: v for k, v in hsb[s].items() if k != "x"} for s in C.SIDES},
           "contrasts": contr, "contrasts_heldout_surface": contr_hs, "surface_audit": audit, "rule": rule,
           "filler_residual_covariate": residual_covariate(dev, D, sides["filler"]["x"], hsb["filler"]["x"]),
           "exact_metrics": {s: {m: str(acc[s][m]) for m in GROUPS + C.CONTROLS + ("cond_oor_val",)} for s in C.SIDES},
           "exact_metrics_heldout_surface": {s: {m: str(acc_hs[s][m]) for m in HS_METRICS} for s in C.SIDES}}
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
        L += ["", "Held-out surface (paraphrase 4 + field names 4 + held-out key orders), the primary rows:", "",
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
              "| stratum | format | cmp_ok | ok_ok | calc_ok | after follows calc |", "|---|---|---|---|---|---|"]
        for st, f in tf.items():
            L.append(f"| {st} | {f.get('format_ok')} | {f.get('cmp_ok')} | {f.get('ok_ok', '-')} | {f.get('calc_ok')} | "
                     f"{f.get('after_follows_calc')} |")
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
            res["golds"][side]["ok_ok|cond"] = _r(np.mean([sc[i]["ok_ok"] for i in cond]))
            res["golds"][side]["calc_follows_cmp|cond"] = _r(np.mean([sc[i]["calc_follows_cmp"] for i in cond]))
        if side == "filler":
            res["golds"][side]["fill_units_of == FILL_UNITS[kind]"] = _r(np.mean(
                [C.fill_units_of(r["targets"]["filler"]) == C.FILL_UNITS[r["event"]["kind"]] for r in dev]))
        hb = hs_side_block(dev, hs, [dev[D.pos[h["id"]]]["targets"][C.SIDE_ARM[side]] for h in hs], side, D)
        res["golds_heldout_surface"][side] = {m: hb["metrics"][m] for m in HS_METRICS}
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
                bad = s["format_ok"] or s["parsed"]
            elif must == "after":
                bad = s["after_ok"]
            elif must == "calc":
                bad = s["calc_ok"] or not s["after_ok"]          # the after-only parser still reads the right after
            else:                                            # cmp: wrong predicate field, after still read right
                bad = s["cmp_ok"] or (name == "trace flipped ok" and s["ok_ok"]) or not s["after_ok"]
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
    res["metrics"] = {"perfect": {m: str(metric_exact(ones, D, m)) for m in GROUPS + ("cond_oor_val",)},
                      "always_sub": {m: str(metric_exact(sub, D, m)) for m in ("primary_conditional", "conditional_in",
                                                                               "cond_oor_val")},
                      "always_sub_heldout_index": {m: str(metric_exact(sub_hs, D, m)) for m in ("primary_conditional",
                                                                                                 "cond_oor_val")},
                      "boot_identical_sides": metric_boot_diff(ones, ones, D, "primary_conditional", 2000),
                      "n": {m: int(len(D.idx[m])) for m in GROUPS + ("cond_oor_val",)}}
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
        and res["metrics"]["always_sub"] == {"primary_conditional": "1/2", "conditional_in": "1/2", "cond_oor_val": "1/2"} \
        and res["metrics"]["always_sub_heldout_index"] == {"primary_conditional": "1/2", "cond_oor_val": "1/2"} \
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
    other // 10 > threshold // 10: right on every far / out-of-range-threshold pair, wrong on most boundary pairs).
    'Outside' = t2br_common.is_oor (value, threshold or conditional distance)."""
    key = env.dumps({"b": before, "e": ev})
    v = ev.get("value", 0)
    oor = C.is_oor(before, ev)
    if how == "tens":
        if ev["kind"] != "conditional":
            return env.step(FAM, before, ev)
        b = dict(before)
        add = before[ev["other"]] // 10 > ev["threshold"] // 10
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


def _fmt(before: dict, ev: dict, st: dict, arm: str) -> str:
    """Render a synthetic next state in an arm's format (trace fields from the synthetic state itself; the filler is the
    amendment-4b constant fill of the event's kind)."""
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
    tr = C.trace_text(cmp, calc, st)
    if arm == "trace":
        return tr
    if arm == "trace_sorted":
        return json.dumps(json.loads(tr), separators=(",", ":"), sort_keys=True)
    raise ValueError(arm)


def _arm_of(fmt: str) -> str:
    return {"trace_sorted": "trace", "filler_loop": "filler"}.get(fmt, fmt)


_OFF, _PLAIN, _FILLER = ("offlike", "plain"), ("plainlike", "plain"), ("plainlike", "filler")
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
    "in_dist_worse": ({"off": _OFF, "plain": ("gold", "plain"), "trace": ("worse_in", "trace"), "filler": ("offlike", "filler")},
                      ("exact", "NO-GO (failed: G1 [conditional, cond_oor_val, conditional HS, cond_oor_val HS]; "
                                "G2 [mix, mix HS]; G5 [conditional_in, mix_in])")),
    "g3_only": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": ("gold_mix", "filler")},
                ("exact", "NO-GO (failed: G3 [mix])")),
    "filler_invalid": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace"), "filler": ("plainlike", "filler_loop")},
                       ("exact", C.INCONCLUSIVE_FILLER)),
    "malformed_traces": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace_sorted"), "filler": _FILLER},
                         ("exact", C.DP3_LABEL + " - " + C.DP3_SUB["trace_format"])),
    # NEW in revision 3: a trace arm that is right in Titan's frame but falls back to the plain arm's behaviour on the
    # held-out surface (the PREREG-3 / BWM key-order failure): G1 and G2 fail on the held-out surface only
    "surface_locked": ({"off": _OFF, "plain": _PLAIN, "trace": ("gold", "trace", "plainlike"), "filler": _FILLER},
                       ("exact", "NO-GO (failed: G1 [conditional HS, cond_oor_val HS]; G2 [mix HS])")),
    # NEW in revision 3: a coarse tens-digit comparison passes the pooled conditional (far and out-of-range-threshold
    # pairs are always right) but not the boundary pairs alone (G1's cond_oor_val clause)
    "coarse_comparison": ({"off": _OFF, "plain": _PLAIN, "trace": ("tens", "trace"), "filler": _FILLER},
                          ("exact", "NO-GO (failed: G1 [cond_oor_val, cond_oor_val HS])")),
}

# Engineered exact-boundary runs: per side, the number of WRONG answers per dev stratum (the first rows of the stratum
# in bank order; cond_xm1 rows are all ok-false), in-surface; '_hs' gives the held-out-surface counts (default: the
# in-surface counts of the primary strata). The default side is all correct. Chains are gold for every side.
_BASE_PLAIN = {"cond_oor_val": 50, "cond_oor_thr": 50, "cond_far": 50, "mix_oor": 70}
_BASE_TRACE = {"cond_oor_val": 25, "cond_oor_thr": 25, "cond_far": 25, "mix_oor": 40, "swap": 4, "add_nowrap": 4,
               "cond_xm1": 17, "mix": 5}
_BASE_FILLER = {"cond_oor_val": 35, "cond_oor_thr": 35, "cond_far": 35, "mix_oor": 50}
_BELOW_TRACE = {**_BASE_TRACE, "cond_oor_val": 26, "mix_oor": 41, "swap": 5, "cond_xm1": 18, "mix": 6}
ENGINEERED = {
    # every component exactly ON its bar on both surfaces: G1 +0.25 (conditional and cond_oor_val), G2 +0.30, G3 +0.10
    # / +0.10, G4 -0.05 / -0.05, G5 -0.05 / -0.05; DP3 does not fire (0.10 is not < 0.10). A float implementation fails
    # G2 here (0.6 - 0.3 < 0.3).
    "boundary_go": ({"plain": _BASE_PLAIN, "trace": _BASE_TRACE, "filler": _BASE_FILLER}, "GO"),
    # every G component one count past its bar on both surfaces (G3 on conditional only, so decision point 3 does not
    # fire); the cond_oor_val count also moves the pooled conditional past its bar
    "boundary_below": ({"plain": _BASE_PLAIN, "trace": _BELOW_TRACE, "filler": {**_BASE_FILLER, "mix_oor": 51}},
                       "NO-GO (failed: G1 [conditional, cond_oor_val, conditional HS, cond_oor_val HS]; G2 [mix, mix HS]; "
                       "G3 [conditional]; G4 [swap]; G5 [conditional_in, mix_in])"),
    # in-surface exactly on every bar; on the held-out surface cond_oor_val is one count past its bar while the pooled
    # conditional stays exactly on it (one more wrong on cond_oor_val, one fewer on cond_far)
    "boundary_hs_cond_oor_val": ({"plain": _BASE_PLAIN, "trace": {**_BASE_TRACE, "_hs": {**{s: _BASE_TRACE[s] for s in C.HS_STRATA},
                                                                                     "cond_oor_val": 26, "cond_far": 24}},
                                  "filler": _BASE_FILLER}, "NO-GO (failed: G1 [cond_oor_val HS])"),
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


def _wrong_texts(rows: list, arm: str, wrong_n: dict) -> list:
    seen = Counter()
    tx = []
    for r in rows:
        st = env.step(FAM, r["before"], r["event"])
        seen[r["stratum"]] += 1
        if seen[r["stratum"]] <= wrong_n.get(r["stratum"], 0):
            st = dict(st)
            st[r["event"]["register"]] = (st[r["event"]["register"]] + 1) % C.MOD   # wrong, parseable
        tx.append(_fmt(r["before"], r["event"], st, arm))
    return tx


def engineered_outputs(dev: list, parents: list, hs: list, plan: dict) -> tuple:
    texts, gens, hs_texts = {}, {}, {}
    pos = {r["id"]: r for r in dev}
    for side in C.SIDES:
        arm = C.SIDE_ARM[side]
        spec = dict(plan.get(side, {}))
        hs_spec = spec.pop("_hs", {s: spec.get(s, 0) for s in C.HS_STRATA})
        texts[side] = _wrong_texts(dev, arm, spec)
        hs_texts[side] = _wrong_texts([pos[h["id"]] for h in hs], arm, hs_spec)
        g = {}
        for ch in C.flat_chains(parents):
            gold = env.rollout(FAM, ch["s0"], ch["events"])
            g[ch["cand_id"]] = {"cand_id": ch["cand_id"], "states": [env.dumps(s) for s in gold],
                                "texts": [_fmt(gold[t], ev, gold[t + 1], arm) for t, ev in enumerate(ch["events"])]}
        gens[side] = g
    return texts, gens, hs_texts


def boundary_checks(dev: list) -> dict:
    """Exhaustive exact-boundary checks of verdict() on the bank's own denominators (the held-out surface has the same
    denominators as the in-surface primaries): for every count pair whose difference is EXACTLY a bar, the component
    must take the stated direction ('>=' holds; DP3's '<' does not fire), and one count past the bar the opposite. Also
    counts how many of these a float implementation would misjudge."""
    D = DevIndex(dev)
    n = {m: len(D.idx[m]) for m in METRICS}
    ci = D.cls[D.idx["conditional_in"]]
    n0, n1 = int((ci == 0).sum()), int((ci == 1).sum())
    base = {"off": {m: Fraction(1) for m in METRICS}, "plain": {m: Fraction(1, 2) for m in METRICS},
            "trace": {m: Fraction(1) for m in METRICS}, "filler": {m: Fraction(0) for m in METRICS}}
    base_hs = {s: {m: v[m] for m in HS_METRICS} for s, v in base.items()}
    fmt = {s: {p: Fraction(1) for p in PRIMS} for s in ("trace", "filler")}
    out = {}

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
                    bar = {"G1": 0.25, "G2": 0.30, "G3": 0.10, "G4": -0.05, "G5": -0.05, "DP3": 0.10}[cname]
                    fl_bad += ((fv < bar) if cname == "DP3" else (fv >= bar)) != at_bar_holds
        out[key] = {"cases": tot, "wrong": bad, "at_bar_cases": len(pairs), "float_impl_would_misjudge": fl_bad}

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
        run("G1", "trace", "plain", "primary_conditional", mean_pairs(n["primary_conditional"], C.GAIN_CONDITIONAL),
            True, "G1 conditional" + sfx, surf)
        run("G1", "trace", "plain", "cond_oor_val", mean_pairs(n["cond_oor_val"], C.GAIN_CONDITIONAL), True,
            "G1 cond_oor_val" + sfx, surf)
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


def dryrun(n_boot: int = 2000) -> dict:
    dev = C.load_dev()
    hs = C.load_dev_hs()
    parents = C.load_chain_bank()
    out = {"scenarios": {}, "engineered": {}, "n_boot": n_boot}
    go_res = None
    comps_of = {}
    for name, (spec, want) in SCENARIOS.items():
        t0 = time.time()
        texts, gens, hs_texts = synth_outputs(dev, parents, hs, spec, name)
        res = analyse_seed(dev, parents, texts, gens, hs, hs_texts, n_boot)
        v = res["rule"]["verdict"]
        hit = v == want[1]
        comps_of[name] = {(c["condition"], c["part"]): c["holds"] for c in res["rule"]["components"]}
        out["scenarios"][name] = {"expected": list(want), "verdict": v, "ok": hit,
                                  "values": res["rule"]["values"], "validity": res["rule"]["validity"],
                                  "seconds": round(time.time() - t0, 1),
                                  "trace_format_ok_mean": _r(np.mean([f["format_ok"] for f in
                                                                      res["onestep"]["trace"]["trace_fields"].values()])),
                                  "surface_audit_trace": {m: res["surface_audit"]["trace"][m]["held_out_minus_in"]
                                                          for m in HS_METRICS},
                                  "product_law_ALL_trace": {k: res["chains"]["ALL"]["product_law"]["trace"][k]
                                                            for k in ("predicted_k4_exact_product", "observed_k4_exact")},
                                  "regret_k4_ALL": {s: res["chains"]["ALL"]["k"]["4"]["sides"][s]["regret"]["mean"]
                                                    for s in C.SIDES}}
        print(f"  [{'PASS' if hit else 'FAIL'}] {name}: {v}", flush=True)
        if name == "go":
            go_res = res
    for name, (plan, want) in ENGINEERED.items():
        t0 = time.time()
        texts, gens, hs_texts = engineered_outputs(dev, parents, hs, plan)
        res = analyse_seed(dev, parents, texts, gens, hs, hs_texts, n_boot)
        v = res["rule"]["verdict"]
        out["engineered"][name] = {"expected": want, "verdict": v, "ok": v == want,
                                   "components": [{k: c[k] for k in ("condition", "surface", "contrast", "metric",
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
    out["checks"] = {
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
        "coarse_comparison: the POOLED conditional passes on both surfaces; only the cond_oor_val clause fails":
            cc[("G1", "conditional")] and cc[("G1", "conditional HS")] and not cc[("G1", "cond_oor_val")]
            and not cc[("G1", "cond_oor_val HS")] and out["scenarios"]["coarse_comparison"]["values"][
                "trace-plain|conditional"] >= 0.25,
        "surface_locked: every in-surface G1 / G2 part holds; every held-out-surface part fails; the trace side's surface "
        "audit shows the loss": all(sl[("G1", p)] for p in ("conditional", "cond_oor_val")) and sl[("G2", "mix")]
            and not any(sl[(g, p)] for g, p in (("G1", "conditional HS"), ("G1", "cond_oor_val HS"), ("G2", "mix HS"))
                        ) and all(v < 0 for v in out["scenarios"]["surface_locked"]["surface_audit_trace"].values()),
        "go: a gold trace side has zero surface-audit loss": all(
            v == 0.0 for v in out["scenarios"]["go"]["surface_audit_trace"].values()),
        "boundary_go: every component value equals its bar exactly (both surfaces)": all(
            Fraction(c["value"]) == Fraction(c["bar"]) for c in eg["components"]),
        "boundary_go: every component (all at their bar) is flagged within 1 SE": all(
            c["within_1se"] for c in eg["components"]) and len(eg["within_1se_of_bar"]) == len(COMPONENTS),
        "verdict() exact-boundary checks (exhaustive on the bank's denominators, both surfaces)": bnd["ok"],
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
    out["example_report_md"] = render({"seeds": ["s0"], "train_variant": C.TRAIN_VARIANT,
                                       "overall": "SCREEN (seed 0): " + go_res["rule"]["verdict"] + " [SYNTHETIC DRY RUN]",
                                       "go_rule": C.GO_RULE_TEXT, "go_rule_revision": C.GO_RULE_REVISION,
                                       "per_seed": {"s0": clean(go_res)}})
    return out


def _refuses_floats() -> bool:
    acc = {s: {m: 0.5 for m in METRICS} for s in C.SIDES}
    acc_hs = {s: {m: Fraction(1, 2) for m in HS_METRICS} for s in C.SIDES}
    fmt = {s: {p: Fraction(1) for p in PRIMS} for s in ("trace", "filler")}
    ok = True
    for a, h in ((acc, acc_hs), ({s: {m: Fraction(1, 2) for m in METRICS} for s in C.SIDES},
                                 {s: {m: 0.5 for m in HS_METRICS} for s in C.SIDES})):
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
