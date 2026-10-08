"""PREREG-2 analysis (PREREG_2_DRAFT.md; frozen with it). CPU only: numpy/json, never loads a model, never imports torch.

  python scripts/confirm2_analysis.py             # the registered run on runs/bank_confirm2 (refuses unless the PREREG-2
                                                  # freeze verifies; create-only outputs, written together)
  python scripts/confirm2_analysis.py --dry-run [--out-dir DIR]
                                                  # code check on pilot TEST with the T2 seed-0 outputs in runs/t2/t2_s0
                                                  # (NOT confirmatory; default output docs/prereg2/PREREG_2_DRYRUN.*, which
                                                  # is refused once PREREG-2 is frozen because those files are frozen inputs)

Question (PREREG-2 §1). Does the fixed T2 seed-0 LoRA (runs/t2/t2_s0/adapter, sha 92bc7c74...; 1,501 oracle-labelled
one-step rows stratified toward rule cases) lower Titan-Sparse's chain_one k=4 normalised decision regret on LEDGER, versus
the same model with the adapter OFF in the same process, on fresh sealed parents (data/bank_confirm2.json)?

Every contrast is adapter ON minus adapter OFF on the same parents (paired per parent).

Primary (Holm, m = 2; one-sided; 10,000 parent resamples, seed 17):
  P1  ledger chain_one k=4 normalised regret, ON - OFF.      PASS iff Holm p < .05 and point <= -0.15
  P2  ledger chain_one k=4 exact-state accuracy, ON - OFF.   PASS iff Holm p < .05 and point >= +0.25
  Claim confirmed iff P1 and P2 PASS (an intersection-union test: each at .05 already gives a level-.05 test of the
  conjunction, so Holm is conservative here and matters only for the partial outcomes). Strong pass: the UNADJUSTED
  one-sided 95% bound is beyond the threshold (a level-.05 margin test of the conjunction, again by intersection-union).
  Reversal: a primary whose point is in the wrong direction with unadjusted one-sided p < .05 is reported as REVERSED first.
  Qualifiers (wording only, never change PASS): the F_touched decision floor, the copy-chain exact floor (both expected never
  to bind on ledger), increased ledger copying (false-copy ON - OFF CI above 0), and the registered forgetting guard.
  Split halves are descriptive only (random, exchangeable halves; no qualifier).
Forgetting guard (registered; not part of the claim; a re-measurement of quantities partly known before the freeze, §5b):
  ratio r = mean first-token KL(OFF||ON) / mean KL(Supa||Sparse) on the 248 off-task prompts, measured in the same GPU job.
  PASS iff the one-sided 95% upper bound of r < 1; FAIL iff the one-sided 95% lower bound > 1; else INCONCLUSIVE
  (reply-class cluster bootstrap). Only FAIL changes the headline. Descriptive: cumulative KL(Supa||ON) / KL(Supa||Sparse)
  (reading bound 1.25) and 64-token greedy agreement OFF/ON vs Supa/Sparse vs Supa/ON.
Secondary (two-sided p, BH q = 0.10 across all 28): regret and exact-state at k = 1/2/4 in every registered cell (minus the
  two primaries), false-copy rates per cell, and ledger chain_one k=4 exact-state ON - OFF on all-posted and any-void plans.
Pre-declared directional predictions D1-D4 (prerequisite copy shortcut; register no large change) by their stated rules.

Goals: the PREREG-1 / T2 test goals (256 unit Gaussian, seed 1001) and tie-break rng seed 17, through the same code path
as rcwmt.decisions.g_only (asserted equal on every call). A fresh goal set (seed 1004) is a registered sensitivity.

Helper functions shared with the frozen D-19 analysis (chain-file validation, first-token KL, JSON bleed) are imported from
scripts/t2_analysis.py (sha d699dd5a...), which also installs the torch-free rcwmt.heads placeholder.
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
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2_analysis as T2  # noqa: E402  (frozen D-19 analysis; stubs rcwmt.heads so torch is never imported)
import confirm2_gen as G2  # noqa: E402  (freeze verifier and stage list; torch-free at import)
from rcwmt import data, decide, decisions, env, fidelity  # noqa: E402

if "torch" in sys.modules:
    raise SystemExit("torch must not be imported by the PREREG-2 analysis")

# ---------------------------------------------------------------- registered constants
BANK_STEM = "bank_confirm2"
BANK_SHA = "fafb13e573a20eb5045ef204a7cdbf8da4393d21604245e9df26995f5be58aea"
BUILD_LOG_SHA = "0b3a6eaa57fd03e11d95bbb83cae58d1732ed8e2102c66700560f02799d8c040"
PILOT_SHA = "7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a"
ADAPTER_SHA = "92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a"
FP = {"sparse": "818be9c6b6a3", "supa": "a9952c55f5a1"}
N_BOOT = 10000
BOOT_SEED = 17
TIE_SEED = 17
TEST_GOAL_SEED = 1001
SENS_GOAL_SEED = 1004
N_GOALS = 256
ALPHA = 0.05
BH_Q = 0.10
P1_THRESHOLD = -0.15          # ledger regret ON - OFF must be <= this (point) to PASS
P2_THRESHOLD = 0.25           # ledger exact-state ON - OFF must be >= this (point) to PASS
COPY_FLOOR_MARGIN = 0.10      # exact(ON) must exceed the copy-chain floor by this much, else the copy qualifier applies
P_COPY_PREDICTION_MARGIN = 0.10
D4_EXACT_BAND = 0.05          # D4 register "no large change": |exact ON - OFF| <= 0.05 ...
D4_REGRET_BAND = 0.10         # ... and |regret ON - OFF| <= 0.10 (PREREG-1's per-family practical threshold)
GUARD_CUMULATIVE_READING = 1.25   # descriptive reading bound for KL(Supa||ON) / KL(Supa||Sparse)
N_PARENTS = 150               # per family in the sealed bank (checked in the registered run)
HALF_SALT = "RCWM-PREREG2-half-v1"
FULL = {"L": "ledger_replay", "R": "register_machine", "P": "prerequisite_plan"}
CODE = {v: k for k, v in FULL.items()}
# (family, mode) cells that the GPU job generates, OFF and ON. Primary frame = the frame T2 trained (D-19).
CELLS = (("L", "chain_one", "primary"), ("R", "chain_one", "secondary"), ("P", "chain_m1", "secondary"),
         ("L", "chain_m1", "transfer"))
KS = (1, 2, 4)
# Operative-content overlaps found after the build by the engineering review's machine-wide scan (FDS operative_key) in
# banks outside the exclusion union (composition_confirmation_v1 T_BANK_V3X / STAGE_A_T_V3X; composition_confirmation_v2
# T_BANK_V4). Not in docs/prereg2/bank_build_log.json; added to the 'drop any overlap' sensitivity. Zero exact payloads.
EXTRA_OVERLAP_PARENTS = {"R": ("T1R-4e04d7f9361a9fae",),
                         "P": ("T1P-008770db1ca815ee", "T1P-3ba03cc762da0e68", "T1P-c1b0331c2dd65a50", "T1P-e184a5ea0c3eda6f")}
ADAPTER_DESC = ("a LoRA trained on 1,501 oracle-labelled one-step rows stratified toward rule cases "
                "(T2 seed 0, the fixed artifact)")
# PREREG-2 §8: the limits that always accompany the claim (rendered verbatim)
LIMITS = (
    "Ledger only, horizon <= 4, one adapter (T2 seed 0), Titan-Sparse only, the chain_one frame.",
    "The plans and the adapter's training rows come from the same event generator and marginals: the bank is fresh at "
    "payload level, not out-of-distribution.",
    "Rule learning is not separated from frame adaptation (no self-distillation or untargeted-LoRA control); posted-"
    "arithmetic gains count toward the effect, and the targeting of the training rows was not shown to cause the gain.",
    "No comparison with non-LM learners at the same label budget: S@abundant (4,000 parents) reached 0.273 on ledger in "
    "PREREG-1 and S-roll matched Titan on ledger at about 512 transitions (R10); the adapted model used 384 + 1,501 "
    "oracle-labelled rows.",
    "Register (mix, wrap) and prerequisite are not claimed fixed; the prerequisite prediction is a copy shortcut, and the "
    "prerequisite family is fresh at payload level only.",
    "Retention evidence is limited to the first-token guard and 64-token agreement on 248 prompts, a re-measurement of "
    "quantities partly known before the freeze (§5b).",
)
OFFTASK_PROMPTS = ROOT / "data" / "offtask_prompts.jsonl"
RUN_DIR = ROOT / "runs" / BANK_STEM
BANK_PATH = ROOT / "data" / f"{BANK_STEM}.json"
BUILD_LOG = ROOT / "docs" / "prereg2" / "bank_build_log.json"
DRY_RUN_DIR = ROOT / "runs" / "t2" / "t2_s0"           # pilot TEST, T2 seed 0 (exploratory R13 outputs)
DRY_OUT = ROOT / "docs" / "prereg2"
POWER_SIM = {"n": 150, "n_sim": 200, "n_boot": 2000, "seed": 20261008}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(what, errs):
    T2.die(what, errs if isinstance(errs, list) else [str(errs)])


def need(cond: bool, msg: str) -> None:
    """Integrity check that survives `python -O`."""
    if not cond:
        die("integrity check failed", [msg])


def read_json(p: Path, what: str):
    if not Path(p).is_file():
        die(f"{what} is missing", [str(p)])
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except ValueError as e:
        die(f"{what} is unreadable", [f"{p}: {e}"])


# ---------------------------------------------------------------- statistics


def boot_mean(d: np.ndarray, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    """Parent bootstrap of mean(d) (NaN dropped). One-sided p for 'mean < 0' (p_neg) and 'mean > 0' (p_pos),
    as in PREREG-1: p = (#{boot on the null side, inclusive} + 1) / (n_boot + 1). Two-sided p = min(1, 2 min(p_neg, p_pos))."""
    d = np.asarray(d, np.float64)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return {"n": int(len(d)), "point": float("nan"), "p_neg": float("nan"), "p_pos": float("nan"),
                "p_two_sided": float("nan"), "ci95": [float("nan")] * 2, "upper95_one_sided": float("nan"),
                "lower95_one_sided": float("nan"), "sd": float("nan")}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p_neg = float(((bs >= 0).sum() + 1) / (n_boot + 1))
    p_pos = float(((bs <= 0).sum() + 1) / (n_boot + 1))
    return {"n": int(len(d)), "point": float(d.mean()), "sd": float(d.std(ddof=1)),
            "p_neg": p_neg, "p_pos": p_pos, "p_two_sided": float(min(1.0, 2 * min(p_neg, p_pos))),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "upper95_one_sided": float(np.quantile(bs, 0.95)), "lower95_one_sided": float(np.quantile(bs, 0.05))}


def holm(ps: dict, alpha: float = ALPHA) -> dict:
    """Monotone Holm step-down (identical to scripts/confirm_analysis.py)."""
    order = sorted(ps, key=ps.get)
    m, out, prev = len(order), {}, 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, ps[k] * (m - i)))
        out[k] = {"p_holm": prev, "holm_significant": prev < alpha}
    return out


def bh(ps: dict, q: float = BH_Q) -> dict:
    """Benjamini-Hochberg adjusted p (identical to scripts/confirm_analysis.py)."""
    keys = sorted(ps, key=ps.get)
    m = len(keys)
    adj, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ps[keys[i]] * m / (i + 1))
        adj[keys[i]] = prev
    return {k: {"p_bh": adj[k], "bh_significant": adj[k] < q} for k in keys}


def halves(pids: list[str]) -> tuple[list[str], list[str]]:
    """Two fixed halves of the parents: order by sha256(salt|parent_id), split at n//2. Outcome-blind; descriptive only."""
    order = sorted(pids, key=lambda p: hashlib.sha256(f"{HALF_SALT}|{p}".encode()).hexdigest())
    h = len(order) // 2
    return order[:h], order[h:]


def cluster_ratio_boot(num: np.ndarray, den: np.ndarray, clusters: list, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    """Ratio of means sum(num)/sum(den) with whole clusters (reply classes) resampled. The prompts are fixed; the interval
    treats the 62 reply classes as a sample of possible off-task classes."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    _, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    S1, S2 = np.bincount(inv, weights=num), np.bincount(inv, weights=den)
    S3 = np.bincount(inv, weights=num - den)
    N = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S1), (n_boot, len(S1)))
    ratio = S1[idx].sum(1) / np.maximum(S2[idx].sum(1), 1e-300)
    diff = S3[idx].sum(1) / N[idx].sum(1)
    return {"ratio": float(num.sum() / max(den.sum(), 1e-300)), "ratio_ci95": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
            "ratio_upper95_one_sided": float(np.quantile(ratio, 0.95)), "ratio_lower95_one_sided": float(np.quantile(ratio, 0.05)),
            "mean_diff": float((num - den).mean()), "mean_diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "n_prompts": int(len(num)), "n_clusters": int(len(S1))}


def guard_status(rb: dict) -> str:
    """Registered three-outcome rule (§5b)."""
    if rb["ratio_upper95_one_sided"] < 1.0:
        return "PASS"
    if rb["ratio_lower95_one_sided"] > 1.0:
        return "FAIL"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- decision regret (the g_only path, with a goal seed)


def _phi_rows(te, fam, gen_mode, fill):
    dim = len(te[0]["phi"])
    out = []
    for r in te:
        st = fidelity._gen_state({"m": gen_mode}, "m", r)
        if st is None and fill == "last_parsed":
            g = gen_mode.get(r["cand_id"])
            prev = [s for s in (g["states"][: r["k"] + 1] if g else []) if s is not None]
            st = json.loads(prev[-1]) if prev else None
        out.append(np.full(dim, np.nan, np.float32) if st is None else data.phi_from_state(fam, r["s0"], r["events"], st))
    return np.stack(out)


def decision_setup(rows, fam, k, goal_seed):
    te = [r for r in rows if r["family"] == fam and r["k"] == k]
    dim = len(te[0]["phi"])
    G = decide.goals(dim, N_GOALS, goal_seed)
    pids = sorted({r["parent_id"] for r in te})
    K = 1 + max(int(r["cand_id"].rsplit("-c", 1)[1]) for r in te)
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    U_true = decisions._cube(pids, K, k, pos, np.stack([r["phi"] for r in te]) @ G.T)
    return te, G, pids, K, pos, U_true


def gen_regret(rows, fam, k, gen_mode, goal_seed=TEST_GOAL_SEED, fill="parent_mean"):
    """Per-parent normalised regret of a chain set (sorted parent ids). Unparseable -> parent-mean utility (primary)
    or the last parsed chain state (sensitivity). Tie-break rng seed 17, fresh per call (identical for OFF and ON)."""
    te, G, pids, K, pos, U_true = decision_setup(rows, fam, k, goal_seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        Uc = decisions._cube(pids, K, k, pos, _phi_rows(te, fam, gen_mode, fill) @ G.T)
        f = np.nanmean(Uc, axis=1, keepdims=True)
    Uc = np.where(np.isnan(Uc), np.where(np.isnan(f), 0.0, f), Uc)
    nreg = decide.regret_table(U_true, Uc, np.random.default_rng(TIE_SEED))["nregret"]
    if goal_seed == TEST_GOAL_SEED and fill == "parent_mean":        # same code path as the frozen g_only
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            ref = decisions.g_only(rows, fam, k, {"m": gen_mode}, modes=("m",))["m"]
        need(ref["pids"] == pids and np.array_equal(np.asarray(ref["nregret"]), nreg, equal_nan=True), "g_only mismatch")
    return pids, np.asarray(nreg, np.float64)


def floor_regrets(rows, fam, k, goal_seed=TEST_GOAL_SEED):
    """Answer-free floors and rule-ablated diagnostics (PREREG-1 definitions), per parent."""
    te, G, pids, K, pos, U_true = decision_setup(rows, fam, k, goal_seed)
    phis = defaultdict(list)
    for r in te:
        for name, st in fidelity.heuristic_states(fam, r["s0"], r["events"]).items():
            phis[f"F_{name}"].append(data.phi_from_state(fam, r["s0"], r["events"], st))
        td = fidelity.touched_delta(fam, r["s0"], r["events"])
        if td is not None:
            sl = {"ledger_replay": slice(4, 8), "register_machine": slice(6, 9)}[fam]
            ph = data.phi_from_state(fam, r["s0"], r["events"], r["s0"]).copy()
            ph[sl] = td[sl]
            phis["F_touched"].append(ph)
    out = {}
    for name, v in phis.items():
        U = decisions._cube(pids, K, k, pos, np.stack(v) @ G.T)
        out[name] = np.asarray(decide.regret_table(U_true, U, np.random.default_rng(TIE_SEED))["nregret"], np.float64)
    return pids, out


# ---------------------------------------------------------------- exact state, copying, step accuracy


def parent_exact(rows, fam, k, gen_mode, pids):
    acc = defaultdict(list)
    for r in rows:
        if r["family"] == fam and r["k"] == k:
            acc[r["parent_id"]].append(float(gen_mode[r["cand_id"]]["states"][k] == r["gold"]))
    return np.array([np.mean(acc[p]) for p in pids], np.float64)


def parent_copy_floor(rows, fam, k, pids):
    acc = defaultdict(list)
    for r in rows:
        if r["family"] == fam and r["k"] == k:
            acc[r["parent_id"]].append(float(r["gold"] == env.dumps(r["s0"])))
    return np.array([np.mean(acc[p]) for p in pids], np.float64)


def false_copy(cands_f: dict, fam: str, gen_mode: dict, pids: list[str]):
    """Own-input convention: on every chain step whose input (the model's own previous state) parses and whose correct
    next state given that input differs from it, a 'copy' is an output equal to the input. Per-parent rate. The eligible
    steps depend on the arm's own earlier outputs, so ON - OFF is paired by parent, not by step."""
    cnt = defaultdict(lambda: [0, 0])
    for cid, (p, c) in cands_f.items():
        st = gen_mode[cid]["states"]
        for t, ev in enumerate(c["events"]):
            if st[t] is None:
                break
            inp = json.loads(st[t])
            if env.dumps(env.step(fam, inp, ev)) == st[t]:
                continue
            cnt[p["parent_id"]][0] += 1
            cnt[p["parent_id"]][1] += int(st[t + 1] == st[t])
    rate = np.array([cnt[p][1] / cnt[p][0] if cnt[p][0] else np.nan for p in pids], np.float64)
    tot = [sum(cnt[p][0] for p in pids), sum(cnt[p][1] for p in pids)]
    return rate, {"eligible_steps": tot[0], "copies": tot[1], "pooled_rate": tot[1] / tot[0] if tot[0] else float("nan")}


def op_class(fam: str, before, ev, after_gold) -> str:
    if fam == "prerequisite_plan":
        if not env.applicable(before, ev):
            return "inapplicable"
        return "applicable, state-changing" if after_gold != before else "applicable, unchanged"
    return T2.t2_stratum(CODE[fam], before, ev)


def step_accuracy(cands_f: dict, fam: str, gen_mode: dict) -> dict:
    """Accuracy on chain steps whose input equals the gold state (RESULTS_PILOT R8 convention), by operation class.
    The step sets differ between arms (they depend on each arm's earlier correctness): descriptive, not paired."""
    cnt = defaultdict(lambda: [0, 0])
    for cid, (p, c) in cands_f.items():
        st = gen_mode[cid]["states"]
        gold = env.rollout(fam, p["s0"], c["events"])
        for t, ev in enumerate(c["events"]):
            if st[t] != env.dumps(gold[t]):
                continue
            k = op_class(fam, gold[t], ev, gold[t + 1])
            cnt[k][0] += 1
            cnt[k][1] += int(st[t + 1] == env.dumps(gold[t + 1]))
    return {k: {"n": n, "acc": c / n if n else float("nan")} for k, (n, c) in sorted(cnt.items())}


PLAN_TYPES = (("all_posted", lambda evs: all(e["status"] == "posted" for e in evs)),
              ("any_void", lambda evs: any(e["status"] != "posted" for e in evs)))


def void_split_exact(rows, cands_f, gen_off, gen_on, k=4) -> dict:
    """Ledger: k-step exact accuracy for candidates whose first k events include a void event vs all posted (pooled)."""
    out = {}
    for name, pred in PLAN_TYPES:
        rs = [r for r in rows if r["family"] == "ledger_replay" and r["k"] == k and pred(r["events"])]
        if not rs:
            continue
        off = np.array([gen_off[r["cand_id"]]["states"][k] == r["gold"] for r in rs], float)
        on = np.array([gen_on[r["cand_id"]]["states"][k] == r["gold"] for r in rs], float)
        par = [r["parent_id"] for r in rs]
        out[name] = {"n_candidates": len(rs), "off": float(off.mean()), "on": float(on.mean()), "delta": float((on - off).mean()),
                     "delta_ci95_parent_cluster": list(T2.cluster_boot(on - off, par))}
    return out


def void_split_parent(rows, gen_off, gen_on, pids, k=4) -> dict:
    """Per-parent mean of exact(ON) - exact(OFF) over the parent's all-posted (resp. any-void) plans; NaN if it has none."""
    out = {}
    for name, pred in PLAN_TYPES:
        acc = defaultdict(list)
        for r in rows:
            if r["family"] == "ledger_replay" and r["k"] == k and pred(r["events"]):
                acc[r["parent_id"]].append(float(gen_on[r["cand_id"]]["states"][k] == r["gold"])
                                           - float(gen_off[r["cand_id"]]["states"][k] == r["gold"]))
        out[name] = np.array([np.mean(acc[p]) if acc[p] else np.nan for p in pids], np.float64)
    return out


# ---------------------------------------------------------------- inputs


def load_chains(path: Path, cands_f: dict, name: str, filter_family: bool) -> dict:
    """filter_family: only for the dry run, whose T2 files mix the three families. The registered per-family files are
    validated unfiltered, so a foreign or unknown row is an error."""
    if not path.exists():
        die("a chain file is missing", [str(path)])
    rows = T2.load_jsonl_checked(path)
    if filter_family:
        rows = [r for r in rows if r.get("cand_id") in cands_f]
    errs = T2.chain_errors(rows, cands_f, name)
    if errs:
        die(f"chain file {path} is incomplete or malformed", errs)
    return {r["cand_id"]: r for r in rows}


def offtask_reference():
    prompts = T2.load_jsonl(OFFTASK_PROMPTS)
    return [r["id"] for r in prompts], [r["reply_class"] for r in prompts]


def load_logprobs(d: Path, ids, classes, name):
    meta_p, lp_p = d / "offtask_meta.json", d / "offtask_logprobs.npy"
    if not (meta_p.exists() and lp_p.exists()):
        return None
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    errs = []
    if m.get("ids") != ids or m.get("reply_class") != classes:
        errs.append(f"{name}: meta ids/classes differ from data/offtask_prompts.jsonl")
    a = np.load(lp_p, mmap_mode="r")
    if a.shape[0] != len(ids):
        errs.append(f"{name}: {a.shape[0]} rows for {len(ids)} prompts")
    if errs:
        die("off-task capture does not match the prompts", errs)
    return a


def load_greedy(path: Path, ids, keys, name):
    rows = T2.load_jsonl_checked(path)
    if [r.get("id") for r in rows] != ids:
        die(f"{name} does not list the off-task prompts in order", [str(path)])
    if not all(isinstance(r.get(k), str) for r in rows for k in keys):
        die(f"{name} lacks reply texts", [str(path), str(keys)])
    return rows


def provenance_errors(prov: dict, done: dict, fz: dict, run: Path) -> list[str]:
    """Registered instrument checks on runs/bank_confirm2/provenance.json and DONE.json (PREREG-2 §7.3)."""
    errs = []
    if prov.get("bank_sha256") != BANK_SHA:
        errs.append("provenance bank sha")
    if prov.get("freeze_prereg_sha256") != fz.get("prereg_sha256"):
        errs.append("the GPU job ran under a different PREREG_2.md than the frozen one")
    if not str(prov.get("sparse_fingerprint", "")).startswith(FP["sparse"]):
        errs.append("Titan-Sparse fingerprint")
    if not str(prov.get("supa_fingerprint", "")).startswith(FP["supa"]):
        errs.append("Supa fingerprint")
    stages = prov.get("stages") or {}
    for st in G2.STAGE_ORDER:
        rec = stages.get(st)
        if not rec:
            errs.append(f"stage {st} not recorded in provenance")
            continue
        if (done.get("files") or {}).get(st) != rec.get("files"):
            errs.append(f"stage {st}: DONE.json and provenance list different files/hashes")
        for rel, h in (rec.get("files") or {}).items():
            p = run / rel
            if not p.is_file() or sha256(p) != h:
                errs.append(f"stage {st}: {rel} missing or changed after generation")
        pids = {s.get("pid") for s in (rec.get("sides") or {}).values()}
        if len(pids) != 1 or None in pids:
            errs.append(f"stage {st}: sides not recorded in one process ({pids})")
    anchor = prov.get("anchor") or {}
    if not anchor.get("pass"):
        errs.append("anchor check (pilot TEST ledger chains reproduce T2 s0 OFF and ON) did not pass")
    allowed = G2.allowed_hashes(fz)
    sealed = {"S1", "S3", "S4", "S5"}
    for i, t in enumerate(prov.get("turns") or []):
        ran = set(t.get("stages_run") or [])
        if "anchor_recheck" in t and not (t["anchor_recheck"] or {}).get("pass"):
            errs.append(f"turn {i}: anchor re-check failed")
        if ran & sealed and "A0" not in ran and not (t.get("anchor_recheck") or {}).get("pass"):
            errs.append(f"turn {i}: sealed-bank chains generated without an anchor or a passing re-check in that turn")
        for r, h in (t.get("code_sha256") or {}).items():
            if h not in allowed.get(r, set()):
                errs.append(f"turn {i}: {r} sha {h[:12]} is neither the frozen nor an amended version")
        if ran - {"S6"}:
            rt = t.get("runtime") or {}
            for k in ("torch", "transformers"):
                if rt.get(k) != G2.REGISTERED_RUNTIME[k]:
                    errs.append(f"turn {i}: {k} {rt.get(k)} != registered {G2.REGISTERED_RUNTIME[k]}")
            if t.get("peft_version") != G2.REGISTERED_RUNTIME["peft"]:
                errs.append(f"turn {i}: peft {t.get('peft_version')} != registered")
            if t.get("adapter_sha256_at_load") != ADAPTER_SHA:
                errs.append(f"turn {i}: adapter hashed at load differs from the registered artifact")
        if "S6" in ran:
            rt = t.get("supa_runtime") or {}
            for k in ("torch", "transformers"):
                if rt.get(k) != G2.REGISTERED_RUNTIME[k]:
                    errs.append(f"turn {i} (Supa): {k} {rt.get(k)} != registered")
    return errs


class Inputs:
    def __init__(self, dry: bool):
        self.dry = dry
        self.integrity = {}
        if dry:
            bank_path = ROOT / "data" / "bank_pilot.json"
            need(sha256(bank_path) == PILOT_SHA, "pilot bank hash")
            self.bank = json.loads(bank_path.read_text(encoding="utf-8"))
            self.bank["parents"] = [p for p in self.bank["parents"] if p["split"] == "test"]
            run = DRY_RUN_DIR
            self.chain_path = lambda m, f, side: run / f"gen_{m}{'_base' if side == 'off' else ''}.jsonl"
            self.off_dirs = {"off": run / "off", "on": run, "supa": None}
            self.greedy_path, self.greedy_keys = run / "offtask_gen.jsonl", ("base_text", "t2_text")
            self.supa_greedy_path = None
            ad = run / "adapter" / "adapter_model.safetensors"
            self.integrity.update(bank="bank_pilot (TEST split only)", bank_sha256=PILOT_SHA, run_dir=str(run),
                                  adapter_sha256_matches=sha256(ad) == ADAPTER_SHA)
            need(self.integrity["adapter_sha256_matches"], "dry run: the T2 s0 adapter is not the registered artifact")
        else:
            ok, errs, fz = G2.verify_freeze()
            if not ok:
                die("PREREG-2 is not frozen, or the freeze does not verify (confirm2_gen.py --verify)", errs)
            bank_path = BANK_PATH
            need(sha256(bank_path) == BANK_SHA, "confirmation bank hash mismatch")
            self.bank = json.loads(bank_path.read_text(encoding="utf-8"))
            need(all(p["split"] == "test" for p in self.bank["parents"]), "bank split")
            run = RUN_DIR
            done = read_json(run / "DONE.json", "runs/bank_confirm2/DONE.json (the PREREG-2 GPU job is incomplete)")
            prov = read_json(run / "provenance.json", "runs/bank_confirm2/provenance.json")
            errs = provenance_errors(prov, done, fz, run)
            if errs:
                die("provenance does not match the registered instrument", errs)
            self.chain_path = lambda m, f, side: run / "chains" / f"{m}_{f}_{side}.jsonl"
            self.off_dirs = {"off": run / "offtask" / "off", "on": run / "offtask" / "on", "supa": run / "offtask" / "supa"}
            self.greedy_path, self.greedy_keys = run / "offtask" / "offtask_gen.jsonl", ("off_text", "on_text")
            self.supa_greedy_path = run / "offtask" / "supa_gen.jsonl"
            self.integrity.update(bank=BANK_STEM, bank_sha256=BANK_SHA, run_dir=str(run), provenance=prov,
                                  freeze_prereg_sha256=fz["prereg_sha256"])
        self.rows = data.candidate_table(self.bank)
        self.cands = {f: {c["cand_id"]: (p, c) for p in self.bank["parents"] if p["family"] == FULL[f] for c in p["candidates"]}
                      for f in FULL}
        self.gen = {}
        for f, m, _ in CELLS:
            for side in ("off", "on"):
                self.gen[(f, m, side)] = load_chains(self.chain_path(m, f, side), self.cands[f], f"{m}_{f}_{side}", filter_family=dry)
        self.ids, self.classes = offtask_reference()
        self.lp = {k: (None if d is None else load_logprobs(d, self.ids, self.classes, k)) for k, d in self.off_dirs.items()}
        if not self.dry and any(v is None for v in self.lp.values()):
            die("off-task captures missing", [k for k, v in self.lp.items() if v is None])
        if not self.dry and not self.greedy_path.exists():
            die("off-task greedy replies missing", [str(self.greedy_path)])
        self.greedy = load_greedy(self.greedy_path, self.ids, self.greedy_keys, "offtask_gen.jsonl") if self.greedy_path.exists() else None
        self.supa_greedy = None
        if self.supa_greedy_path is not None:
            if not self.supa_greedy_path.exists():
                die("Supa greedy replies missing", [str(self.supa_greedy_path)])
            self.supa_greedy = load_greedy(self.supa_greedy_path, self.ids, ("supa_text",), "supa_gen.jsonl")
        if not dry:
            n = {f: len({p["parent_id"] for p, _ in self.cands[f].values()}) for f in FULL}
            if any(v != N_PARENTS for v in n.values()):
                die("the confirmation bank does not hold the registered parent counts", [str(n)])
        # operative-content overlap of every parent with the adapter's training rows (data/t2_train.jsonl)
        from build_confirm2_bank import content_key   # noqa: E402  (no torch; same definition as the bank build log)
        t2c = {content_key(r["family"], r["before"], r["event"]) for r in T2.load_jsonl(ROOT / "data" / "t2_train.jsonl")}
        self.overlap_t2train = {}
        for f in FULL:
            hit = set()
            for cid, (p, c) in self.cands[f].items():
                st = env.rollout(p["family"], p["s0"], c["events"])
                if any(content_key(p["family"], st[t], ev) in t2c for t, ev in enumerate(c["events"])):
                    hit.add(p["parent_id"])
            self.overlap_t2train[f] = hit
        self.overlap_any = {}
        if not dry:
            need(BUILD_LOG.is_file() and sha256(BUILD_LOG) == BUILD_LOG_SHA, "docs/prereg2/bank_build_log.json missing or changed")
            lg = json.loads(BUILD_LOG.read_text(encoding="utf-8"))
            need(lg.get("sha256") == BANK_SHA, "the build log describes another bank")
            pids_all = {f: {p["parent_id"] for p, _ in self.cands[f].values()} for f in FULL}
            for f in FULL:
                extra = set(EXTRA_OVERLAP_PARENTS.get(f, ()))
                need(extra <= pids_all[f], f"EXTRA_OVERLAP_PARENTS[{f}] not in the bank")
                self.overlap_any[f] = set(lg["content_overlap_parent_ids"][FULL[f]]) | extra
        self.integrity["content_overlap_parents"] = {"t2_train": {f: len(v) for f, v in self.overlap_t2train.items()},
                                                     "any_source_build_log_plus_review_scan": {f: len(v) for f, v in self.overlap_any.items()}}


# ---------------------------------------------------------------- analysis blocks


def forgetting(I: Inputs) -> dict:
    out = {}
    OFF, ON, SU = I.lp["off"], I.lp["on"], I.lp["supa"]
    if OFF is None or ON is None:
        return {"status": "PENDING", "reason": "OFF/ON off-task log-probs missing"}
    kl, a_off, a_on = T2.lp_compare(OFF, ON)
    out["kl_off_on"] = T2._kl_summary(kl)
    out["top1_agree_off_on"] = float((a_off == a_on).mean())
    if SU is None:
        out["kl_ref_supa_sparse"] = "pending (no Supa capture in this run)"
        out["status"] = "PENDING"
    else:
        if SU.shape != OFF.shape:
            die("Supa and Sparse off-task arrays differ in shape", [f"{SU.shape} vs {OFF.shape}"])
        kref, a_su, _ = T2.lp_compare(SU, OFF)            # KL(Supa||Sparse): before || after, like KL(OFF||ON)
        kcum, _, _ = T2.lp_compare(SU, ON)                # KL(Supa||ON): cumulative drift from Supa
        out["kl_ref_supa_sparse"] = T2._kl_summary(kref)
        out["kl_supa_on"] = T2._kl_summary(kcum)
        out["top1_agree_supa_sparse"] = float((a_su == a_off).mean())
        rb = cluster_ratio_boot(kl, kref, I.classes)
        out["ratio_kl_adapter_over_ref"] = rb
        out["status"] = guard_status(rb)
        rc = cluster_ratio_boot(kcum, kref, I.classes)
        out["cumulative_ratio_kl_supa_on_over_supa_sparse"] = rc
        out["cumulative_reading_le_1_25"] = bool(rc["ratio"] <= GUARD_CUMULATIVE_READING)
    if I.greedy is not None:
        ko, kn = I.greedy_keys
        out["greedy64_exact_agree_off_on"] = float(np.mean([r[ko] == r[kn] for r in I.greedy]))
        nonnum = [r for r in I.greedy if r["reply_class"] not in T2.NUMERIC_CLASSES]
        out["json_bleed_nonnumeric"] = {s: float(np.mean([T2.looks_state(r[key]) or T2.int_only(r[key]) for r in nonnum]))
                                        for s, key in (("off", ko), ("on", kn))}
        if I.supa_greedy is not None:
            out["greedy64_exact_agree_supa_sparse"] = float(np.mean([s["supa_text"] == r[ko] for s, r in zip(I.supa_greedy, I.greedy)]))
            out["greedy64_exact_agree_supa_on"] = float(np.mean([s["supa_text"] == r[kn] for s, r in zip(I.supa_greedy, I.greedy)]))
    return out


def guard_selftest() -> dict:
    """Code check of the guard path on synthetic first-token distributions (not data)."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 50))
    small = base + 0.05 * rng.standard_normal((40, 50))
    big = base + 1.0 * rng.standard_normal((40, 50))
    k_small, _, _ = T2.lp_compare(base, small)
    k_big, _, _ = T2.lp_compare(big, base)
    cls = [f"c{i // 4}" for i in range(40)]
    a, b = cluster_ratio_boot(k_small, k_big, cls), cluster_ratio_boot(k_big, k_small, cls)
    c = cluster_ratio_boot(k_small, k_small * (1 + 0.3 * rng.standard_normal(40)), cls)
    return {"small_vs_big": [a["ratio"], guard_status(a)], "big_vs_small": [b["ratio"], guard_status(b)],
            "near_one": [c["ratio"], guard_status(c)],
            "ok": bool(guard_status(a) == "PASS" and guard_status(b) == "FAIL" and guard_status(c) == "INCONCLUSIVE")}


def analyse(I: Inputs) -> dict:
    res = {"dry_run": I.dry, "integrity": I.integrity, "cells": {}, "floors": {},
           "registered_constants": {"P1_threshold": P1_THRESHOLD, "P2_threshold": P2_THRESHOLD, "alpha": ALPHA, "bh_q": BH_Q,
                                    "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "tie_seed": TIE_SEED, "test_goal_seed": TEST_GOAL_SEED,
                                    "sensitivity_goal_seed": SENS_GOAL_SEED, "n_goals": N_GOALS, "half_salt": HALF_SALT,
                                    "copy_floor_margin": COPY_FLOOR_MARGIN, "d4_bands": [D4_EXACT_BAND, D4_REGRET_BAND],
                                    "guard_rule": "PASS iff one-sided 95% upper bound of r < 1; FAIL iff lower bound > 1; else INCONCLUSIVE",
                                    "cells": CELLS, "adapter_sha256": ADAPTER_SHA}}
    per_parent = {}
    pids_by_f = {}
    for f in FULL:
        fam = FULL[f]
        if not any(c[0] == f for c in CELLS):
            continue
        for k in KS:
            pids, fl = floor_regrets(I.rows, fam, k)
            pids_by_f[f] = pids
            for name, v in fl.items():
                per_parent[f"{f}|k{k}|{name}"] = v
            res["floors"][f"{f}|k{k}"] = {name: float(np.nanmean(v)) for name, v in fl.items()}
            cf = parent_copy_floor(I.rows, fam, k, pids)
            per_parent[f"{f}|k{k}|copy_chain_exact_floor"] = cf
            res["floors"][f"{f}|k{k}"]["copy_chain_exact_floor"] = float(cf.mean())
    for f, m, role in CELLS:
        fam = FULL[f]
        cell = {"family": fam, "mode": m, "role": role, "k": {}}
        g_off, g_on = I.gen[(f, m, "off")], I.gen[(f, m, "on")]
        for k in KS:
            pids, r_off = gen_regret(I.rows, fam, k, g_off)
            _, r_on = gen_regret(I.rows, fam, k, g_on)
            e_off, e_on = parent_exact(I.rows, fam, k, g_off, pids), parent_exact(I.rows, fam, k, g_on, pids)
            key = f"{f}|{m}|k{k}"
            per_parent.update({f"{key}|regret_off": r_off, f"{key}|regret_on": r_on, f"{key}|exact_off": e_off,
                               f"{key}|exact_on": e_on})
            cell["k"][k] = {"n_parents": len(pids), "n_nan_regret": int(np.isnan(r_off).sum() + np.isnan(r_on).sum()),
                            "regret_off": float(np.nanmean(r_off)), "regret_on": float(np.nanmean(r_on)),
                            "exact_off": float(e_off.mean()), "exact_on": float(e_on.mean()),
                            "regret_on_minus_off": boot_mean(r_on - r_off), "exact_on_minus_off": boot_mean(e_on - e_off),
                            "parse_rate_off": float(np.mean([g_off[c]["states"][k] is not None for c in I.cands[f]])),
                            "parse_rate_on": float(np.mean([g_on[c]["states"][k] is not None for c in I.cands[f]]))}
        pids = pids_by_f[f]
        fc_off, s_off = false_copy(I.cands[f], fam, g_off, pids)
        fc_on, s_on = false_copy(I.cands[f], fam, g_on, pids)
        per_parent[f"{f}|{m}|false_copy_off"], per_parent[f"{f}|{m}|false_copy_on"] = fc_off, fc_on
        cell["false_copy"] = {"off": s_off, "on": s_on, "on_minus_off": boot_mean(fc_on - fc_off)}
        cell["step_accuracy_correct_input"] = {"off": step_accuracy(I.cands[f], fam, g_off), "on": step_accuracy(I.cands[f], fam, g_on)}
        if (f, m) == ("L", "chain_one"):
            cell["void_split_k4"] = void_split_exact(I.rows, I.cands[f], g_off, g_on)
            vp = void_split_parent(I.rows, g_off, g_on, pids)
            for name, v in vp.items():
                per_parent[f"L|chain_one|k4|exact_on_minus_off|{name}"] = v
                cell["void_split_k4"][name]["parent_paired"] = boot_mean(v)
        # sensitivity (descriptive): last-parsed fill, fresh goals
        _, rl_off = gen_regret(I.rows, fam, 4, g_off, fill="last_parsed")
        _, rl_on = gen_regret(I.rows, fam, 4, g_on, fill="last_parsed")
        _, rg_off = gen_regret(I.rows, fam, 4, g_off, goal_seed=SENS_GOAL_SEED)
        _, rg_on = gen_regret(I.rows, fam, 4, g_on, goal_seed=SENS_GOAL_SEED)
        cell["sensitivity_k4"] = {"fill_last_parsed": boot_mean(rl_on - rl_off), "goals_seed_1004": boot_mean(rg_on - rg_off),
                                  "goals_seed_1004_off_on": [float(np.nanmean(rg_off)), float(np.nanmean(rg_on))]}
        # content-overlap sensitivity: drop parents with any operative-content overlap (bank build log + review scan;
        # registered run) and parents sharing operative content with the adapter's training rows (t2_train; computed here)
        rk = per_parent[f"{f}|{m}|k4|regret_on"] - per_parent[f"{f}|{m}|k4|regret_off"]
        ek = per_parent[f"{f}|{m}|k4|exact_on"] - per_parent[f"{f}|{m}|k4|exact_off"]
        for tag, drop in (("drop_t2_train_overlap", I.overlap_t2train.get(f, set())), ("drop_any_overlap", I.overlap_any.get(f, set()))):
            keep = np.array([p not in drop for p in pids], bool)
            cell["sensitivity_k4"][tag] = {"n_dropped": int((~keep).sum()), "regret_on_minus_off": boot_mean(rk[keep]),
                                           "exact_on_minus_off": boot_mean(ek[keep])}
        res["cells"][f"{f}|{m}"] = cell

    # ---- primary
    L = res["cells"]["L|chain_one"]["k"][4]
    p1, p2 = L["regret_on_minus_off"], L["exact_on_minus_off"]
    prim = {"P1": {"contrast": "ledger chain_one k=4 normalised regret, ON - OFF", "claim": "< 0", "threshold": P1_THRESHOLD,
                   **p1, "p_one_sided": p1["p_neg"], "one_sided_bound": p1["upper95_one_sided"]},
            "P2": {"contrast": "ledger chain_one k=4 exact-state accuracy, ON - OFF", "claim": "> 0", "threshold": P2_THRESHOLD,
                   **p2, "p_one_sided": p2["p_pos"], "one_sided_bound": p2["lower95_one_sided"]}}
    for n, h in holm({n: v["p_one_sided"] for n, v in prim.items()}).items():
        prim[n].update(h)
    prim["P1"]["PASS"] = bool(prim["P1"]["holm_significant"] and prim["P1"]["point"] <= P1_THRESHOLD)
    prim["P2"]["PASS"] = bool(prim["P2"]["holm_significant"] and prim["P2"]["point"] >= P2_THRESHOLD)
    prim["P1"]["strong_pass"] = bool(prim["P1"]["PASS"] and prim["P1"]["upper95_one_sided"] <= P1_THRESHOLD)
    prim["P2"]["strong_pass"] = bool(prim["P2"]["PASS"] and prim["P2"]["lower95_one_sided"] >= P2_THRESHOLD)
    # reversal: wrong-direction point with unadjusted one-sided p < .05 (checked before any pass/partial/directional branch)
    prim["P1"]["reversed"] = bool(prim["P1"]["point"] > 0 and prim["P1"]["p_pos"] < ALPHA)
    prim["P2"]["reversed"] = bool(prim["P2"]["point"] < 0 and prim["P2"]["p_neg"] < ALPHA)
    # split halves (descriptive only: random, exchangeable halves)
    pids = pids_by_f["L"]
    A, B = halves(pids)
    ix = {p: i for i, p in enumerate(pids)}
    d1 = per_parent["L|chain_one|k4|regret_on"] - per_parent["L|chain_one|k4|regret_off"]
    d2 = per_parent["L|chain_one|k4|exact_on"] - per_parent["L|chain_one|k4|exact_off"]
    for n, d in (("P1", d1), ("P2", d2)):
        prim[n]["halves_descriptive"] = {hn: {"n": b["n"], "point": b["point"], "ci95": b["ci95"]}
                                         for hn, hp in (("A", A), ("B", B)) for b in [boot_mean(d[[ix[p] for p in hp]])]}
    # floor qualifiers (expected never to bind on ledger, §5) and the copying qualifier
    fl = res["floors"]["L|k4"]
    prim["P1"]["F_touched_mean"] = fl["F_touched"]
    prim["P1"]["on_beats_F_touched"] = bool(L["regret_on"] < fl["F_touched"])
    prim["P1"]["off_mean"], prim["P1"]["on_mean"] = L["regret_off"], L["regret_on"]
    prim["P1"]["F_ignore_status_mean (rule-ablated diagnostic)"] = fl.get("F_ignore_status")
    prim["P2"]["copy_chain_floor"] = fl["copy_chain_exact_floor"]
    prim["P2"]["on_exceeds_copy_floor_by_margin"] = bool(L["exact_on"] >= fl["copy_chain_exact_floor"] + COPY_FLOOR_MARGIN)
    prim["P2"]["off_mean"], prim["P2"]["on_mean"] = L["exact_off"], L["exact_on"]
    fcL = res["cells"]["L|chain_one"]["false_copy"]["on_minus_off"]
    prim["ledger_copying_increased"] = {"point": fcL["point"], "ci95": fcL["ci95"], "increased": bool(fcL["ci95"][0] > 0)}
    res["primary"] = prim
    res["claim_confirmed"] = bool(prim["P1"]["PASS"] and prim["P2"]["PASS"] and not prim["P1"]["reversed"] and not prim["P2"]["reversed"])

    # ---- forgetting guard (registered; not part of the claim)
    res["forgetting_guard"] = forgetting(I)

    # ---- secondary family (two-sided, BH q = 0.10)
    sec, ps = {}, {}
    for f, m, role in CELLS:
        c = res["cells"][f"{f}|{m}"]
        for k in KS:
            for metric in ("regret_on_minus_off", "exact_on_minus_off"):
                if (f, m, k) == ("L", "chain_one", 4):
                    continue                          # the primaries
                key = f"{f}|{m}|k{k}|{metric}"
                sec[key] = c["k"][k][metric]
                ps[key] = sec[key]["p_two_sided"]
        key = f"{f}|{m}|false_copy_on_minus_off"
        sec[key] = c["false_copy"]["on_minus_off"]
        ps[key] = sec[key]["p_two_sided"]
    for name in ("all_posted", "any_void"):
        key = f"L|chain_one|k4|exact_on_minus_off|{name}_plans"
        sec[key] = res["cells"]["L|chain_one"]["void_split_k4"][name]["parent_paired"]
        ps[key] = sec[key]["p_two_sided"]
    ps = {k: v for k, v in ps.items() if not math.isnan(v)}
    for k, v in bh(ps).items():
        sec[k].update(v)
    res["secondary"] = sec
    res["secondary_note"] = ("Two-sided parent-bootstrap p for 'ON - OFF != 0'; BH q = 0.10 across every listed contrast "
                             f"({len(sec)}). About half are ledger contrasts expected to be near-certain, which loosens the BH "
                             "thresholds for register and prerequisite. False-copy and step-level rates condition on arm-specific "
                             "step sets, so their ON - OFF differences are paired by parent only.")

    # ---- pre-declared directional predictions
    P = res["cells"]["P|chain_m1"]
    fcP = P["false_copy"]["on_minus_off"]
    R = res["cells"]["R|chain_one"]["k"][4]
    sc_off = P["step_accuracy_correct_input"]["off"].get("applicable, state-changing", {}).get("acc")
    sc_on = P["step_accuracy_correct_input"]["on"].get("applicable, state-changing", {}).get("acc")
    res["predictions"] = {
        "D1 prerequisite copy shortcut persists: P chain_m1 false-copy ON - OFF > 0 with the 95% CI above 0":
            {"point": fcP["point"], "ci95": fcP["ci95"], "holds": bool(fcP["ci95"][0] > 0)},
        "D2 prerequisite chain_m1 k=4 exact(ON) <= copy-chain floor + 0.10":
            {"on": P["k"][4]["exact_on"], "copy_floor": res["floors"]["P|k4"]["copy_chain_exact_floor"],
             "holds": bool(P["k"][4]["exact_on"] <= res["floors"]["P|k4"]["copy_chain_exact_floor"] + P_COPY_PREDICTION_MARGIN)},
        "D3 prerequisite chain_m1 applicable state-changing step accuracy ON <= OFF (correct-input steps; descriptive point; "
        "arm-specific step sets)":
            {"off": sc_off, "on": sc_on, "holds": bool(sc_off is not None and sc_on is not None and sc_on <= sc_off)},
        f"D4 register: no large change (chain_one k=4: |exact ON - OFF| <= {D4_EXACT_BAND} and |regret ON - OFF| <= "
        f"{D4_REGRET_BAND}; can hold alongside a small BH-significant change)":
            {"exact_delta": R["exact_on_minus_off"]["point"], "regret_delta": R["regret_on_minus_off"]["point"],
             "regret_delta_bh_significant": res["secondary"]["R|chain_one|k4|regret_on_minus_off"].get("bh_significant"),
             "holds": bool(abs(R["exact_on_minus_off"]["point"]) <= D4_EXACT_BAND
                           and abs(R["regret_on_minus_off"]["point"]) <= D4_REGRET_BAND)},
    }
    res["wording"] = wording(res)
    return res, per_parent


# ---------------------------------------------------------------- registered wording (PREREG-2 §8), mechanical


def _ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def _p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def contrast_line(n: str, p: dict) -> str:
    unit = "normalised regret" if n == "P1" else "exact-state accuracy"
    bound = "upper" if n == "P1" else "lower"
    if p.get("reversed"):
        verdict = "REVERSED (wrong direction, unadjusted one-sided p < .05)"
    elif p["PASS"]:
        verdict = "PASS (strong)" if p["strong_pass"] else "PASS (point beyond the threshold; not a strong pass)"
    elif p["holm_significant"]:
        verdict = "significant, point short of the threshold"
    else:
        verdict = "not significant"
    return (f"{n} ledger chain_one k=4 {unit}, ON - OFF: {p['point']:+.3f} (OFF {p['off_mean']:.3f} -> ON {p['on_mean']:.3f}), "
            f"95% CI {_ci(p['ci95'])}, one-sided 95% {bound} bound {p['one_sided_bound']:+.3f}, Holm one-sided p "
            f"{_p(p['p_holm'])}; threshold {p['threshold']:+.2f}: {verdict}.")


def wording(res: dict) -> dict:
    p1, p2 = res["primary"]["P1"], res["primary"]["P2"]
    g = res["forgetting_guard"]
    n = p1["n"]
    lines, quals = [], []
    sig = {"P1": bool(p1["holm_significant"] and p1["point"] < 0), "P2": bool(p2["holm_significant"] and p2["point"] > 0)}
    rev = [nm for nm, p in (("P1", p1), ("P2", p2)) if p.get("reversed")]
    if rev:
        head = "REVERSED: " + " and ".join(rev) + " significantly in the wrong direction"
        what = {"P1": "makes ledger decisions worse (higher regret)", "P2": "lowers ledger four-step exact-state accuracy"}
        lines.append("On fresh parents the adapter " + " and ".join(what[r] for r in rev) +
                     ": R13's improvement does not replicate. The claim is not confirmed.")
    elif p1["PASS"] and p2["PASS"]:
        strong = p1["strong_pass"] and p2["strong_pass"]
        head = "CONFIRMED (" + ("strong pass: both one-sided 95% bounds beyond the thresholds" if strong else
                                "point estimates beyond the thresholds; not a strong pass: the one-sided 95% bound of " +
                                " and ".join(nm for nm, p in (("P1", p1), ("P2", p2)) if not p["strong_pass"]) +
                                " is not beyond its threshold") + ")"
        lines.append(
            f"On {n} fresh ledger parents, adding {ADAPTER_DESC} to frozen Titan-Sparse lowered the normalised regret of its "
            f"chained four-step self-rollout decisions from {p1['off_mean']:.3f} to {p1['on_mean']:.3f} (ON - OFF "
            f"{p1['point']:+.3f}, 95% CI {_ci(p1['ci95'])}, one-sided 95% upper bound {p1['upper95_one_sided']:+.3f}, Holm "
            f"one-sided p {_p(p1['p_holm'])}; 0 = oracle, 1 = random) and raised four-step exact-state accuracy from "
            f"{p2['off_mean']:.3f} to {p2['on_mean']:.3f} ({p2['point']:+.3f}, {_ci(p2['ci95'])}, one-sided 95% lower bound "
            f"{p2['lower95_one_sided']:+.3f}), against the same model with the adapter off in the same process. Both point "
            f"estimates are beyond the pre-set thresholds (-0.15, +0.25); strong pass: {'yes' if strong else 'no'}.")
    elif p1["PASS"] or p2["PASS"]:
        ok, bad = ("P1", "P2") if p1["PASS"] else ("P2", "P1")
        head = f"PARTIAL: {ok} passes, {bad} does not"
        pb = res["primary"][bad]
        if bad == "P2":
            lines.append("The adapter lowers ledger decision regret beyond -0.15 " +
                         (f"(strong pass: {'yes' if p1['strong_pass'] else 'no'}). ") +
                         (f"Four-step exact-state accuracy rose by {pb['point']:+.3f} {_ci(pb['ci95'])}, below the pre-set +0.25."
                          if sig["P2"] else
                          f"Four-step exact-state accuracy did not rise significantly ({pb['point']:+.3f} {_ci(pb['ci95'])}): "
                          "the decision gain is not accompanied by a significant exact-state gain (consistent with partially "
                          "correct states)."))
        else:
            lines.append("The adapter raises ledger four-step exact-state accuracy beyond +0.25 " +
                         (f"(strong pass: {'yes' if p2['strong_pass'] else 'no'}). ") +
                         (f"Decision regret fell by {-pb['point']:.3f} {_ci(pb['ci95'])}, short of the pre-set -0.15."
                          if sig["P1"] else
                          f"Decision regret did not fall significantly ({pb['point']:+.3f} {_ci(pb['ci95'])}): the exact-state "
                          "gain did not translate into a significant decision gain."))
    else:
        s = [nm for nm in ("P1", "P2") if sig[nm]]
        if s:
            other = [nm for nm in ("P1", "P2") if nm not in s]
            head = f"DIRECTIONALLY CONFIRMED ({' and '.join(s)} only), below the pre-set practical threshold"
            lines.append(f"{' and '.join(s)}: significant in the predicted direction, point estimate short of the pre-set "
                         "threshold" + (f"; {other[0]}: not significant." if other else ".") + " The claim is not confirmed.")
        else:
            head = "NOT CONFIRMED"
            lines.append("The T2 seed-0 ledger effect (R13) did not replicate on fresh parents: neither primary contrast is "
                         "significant.")
    lines += [contrast_line("P1", p1), contrast_line("P2", p2)]
    # qualifiers (wording only)
    if p1["PASS"] and not p1["on_beats_F_touched"]:
        quals.append("The adapted rollout's regret is not below the answer-free touched-entity rule (F_touched).")
    if p2["PASS"] and not p2["on_exceeds_copy_floor_by_margin"]:
        quals.append("Adapted exact-state accuracy does not exceed the copy-chain floor by 0.10 (not better than copying).")
    cp = res["primary"]["ledger_copying_increased"]
    if (p1["PASS"] or p2["PASS"]) and cp["increased"]:
        quals.append(f"With increased copying: ledger chain_one false-copy ON - OFF {cp['point']:+.3f} {_ci(cp['ci95'])}.")
    # forgetting guard (registered; only FAIL changes the headline)
    st = g.get("status")
    if st in ("PASS", "FAIL", "INCONCLUSIVE"):
        rb = g["ratio_kl_adapter_over_ref"]
        rc = g["cumulative_ratio_kl_supa_on_over_supa_sparse"]
        guard = (f"Forgetting guard {st}: off-task first-token KL(OFF||ON) is {rb['ratio']:.2f}x KL(Supa||Sparse), the Titan "
                 f"delta's own drift (95% CI {rb['ratio_ci95'][0]:.2f}-{rb['ratio_ci95'][1]:.2f}; one-sided 95% bounds "
                 f"{rb['ratio_lower95_one_sided']:.2f} / {rb['ratio_upper95_one_sided']:.2f}); cumulative KL(Supa||ON) is "
                 f"{rc['ratio']:.2f}x KL(Supa||Sparse) (reading bound {GUARD_CUMULATIVE_READING}); 64-token greedy agreement "
                 f"OFF/ON {g.get('greedy64_exact_agree_off_on', float('nan')):.2f} vs Supa/Sparse "
                 f"{g.get('greedy64_exact_agree_supa_sparse', float('nan')):.2f} vs Supa/ON {g.get('greedy64_exact_agree_supa_on', float('nan')):.2f}. "
                 "A re-measurement of quantities partly known before the freeze (§5b).")
        if st == "FAIL":
            head += f"; at an off-task first-token drift {rb['ratio']:.2f}x that of the Titan delta itself (forgetting guard FAIL)"
    elif st == "NOT EVALUABLE":
        guard = "No off-task reference was measured; no statement about drift."
    else:
        guard = "Forgetting guard PENDING (no Supa reference in this run)."
    halves_note = "Split halves (random, exchangeable; descriptive only): " + "; ".join(
        f"{nm} A {p['halves_descriptive']['A']['point']:+.3f} / B {p['halves_descriptive']['B']['point']:+.3f}"
        for nm, p in (("P1", p1), ("P2", p2)))
    if res["dry_run"]:
        head = f"DRY RUN ONLY (pilot TEST, not fresh parents; code check, not evidence); the mechanical wording would read: {head}"
    return {"headline": head, "statement": lines, "qualifiers": quals, "guard": guard, "descriptive": [halves_note],
            "limits": list(LIMITS)}


def wording_selftest() -> dict:
    """Every §8 branch on synthetic primaries (code check, not data)."""
    def P(point, holm_sig, passed, strong=False, p_pos=0.5, p_neg=0.5, nm="P1"):
        thr = P1_THRESHOLD if nm == "P1" else P2_THRESHOLD
        rev = (point > 0 and p_pos < ALPHA) if nm == "P1" else (point < 0 and p_neg < ALPHA)
        return {"n": 150, "point": point, "holm_significant": holm_sig, "PASS": passed, "strong_pass": strong, "p_pos": p_pos,
                "p_neg": p_neg, "reversed": rev, "threshold": thr, "on_beats_F_touched": True, "on_exceeds_copy_floor_by_margin": True,
                "off_mean": 0.5, "on_mean": 0.5 + point, "ci95": [point - 0.05, point + 0.05], "p_holm": 0.01 if holm_sig else 0.4,
                "upper95_one_sided": point + 0.04, "lower95_one_sided": point - 0.04, "one_sided_bound": point,
                "halves_descriptive": {"A": {"point": point}, "B": {"point": point}}}

    def guard(status):
        if status == "PENDING":
            return {"status": "PENDING"}
        r = {"PASS": 0.5, "FAIL": 2.0, "INCONCLUSIVE": 1.0}[status]
        rb = {"ratio": r, "ratio_ci95": [r - 0.2, r + 0.2], "ratio_lower95_one_sided": r - 0.15, "ratio_upper95_one_sided": r + 0.15}
        return {"status": status, "ratio_kl_adapter_over_ref": rb, "cumulative_ratio_kl_supa_on_over_supa_sparse": rb,
                "greedy64_exact_agree_off_on": 0.5, "greedy64_exact_agree_supa_sparse": 0.4, "greedy64_exact_agree_supa_on": 0.3}

    cases = {  # name: (P1, P2, guard, copying increased, expected headline prefix, required statement fragment)
        "both pass, strong": (P(-0.4, True, True, True), P(0.6, True, True, True, nm="P2"), "PASS", False,
                              "CONFIRMED (strong pass", "strong pass: yes"),
        "both pass, not strong": (P(-0.17, True, True, False), P(0.6, True, True, True, nm="P2"), "PASS", False,
                                  "CONFIRMED (point estimates beyond", "strong pass: no"),
        "both pass, guard FAIL": (P(-0.4, True, True, True), P(0.6, True, True, True, nm="P2"), "FAIL", False,
                                  "CONFIRMED (strong pass", "Both point estimates"),
        "P1 pass, P2 significant below threshold": (P(-0.3, True, True, True), P(0.22, True, False, nm="P2"), "PENDING", False,
                                                    "PARTIAL: P1 passes, P2 does not", "below the pre-set +0.25"),
        "P1 pass, P2 not significant": (P(-0.3, True, True, True), P(0.02, False, False, nm="P2"), "PENDING", False,
                                        "PARTIAL: P1 passes, P2 does not", "consistent with partially correct states"),
        "P2 pass, P1 significant short": (P(-0.10, True, False), P(0.4, True, True, True, nm="P2"), "PENDING", False,
                                          "PARTIAL: P2 passes, P1 does not", "short of the pre-set -0.15"),
        "P1 reversed, P2 pass": (P(0.08, False, False, p_pos=0.001, p_neg=0.999), P(0.30, True, True, nm="P2"), "PENDING", False,
                                 "REVERSED: P1", "makes ledger decisions worse"),
        "P1 reversed, P2 significant below threshold": (P(0.08, False, False, p_pos=0.001, p_neg=0.999), P(0.10, True, False, nm="P2"),
                                                        "PENDING", False, "REVERSED: P1", "not confirmed"),
        "P2 reversed": (P(-0.02, False, False), P(-0.1, False, False, p_pos=0.999, p_neg=0.001, nm="P2"), "PENDING", False,
                        "REVERSED: P2", "lowers ledger four-step"),
        "P1 n.s., P2 significant below threshold": (P(-0.02, False, False), P(0.10, True, False, nm="P2"), "PENDING", False,
                                                    "DIRECTIONALLY CONFIRMED (P2 only)", "P1: not significant"),
        "neither significant": (P(-0.02, False, False), P(0.01, False, False, nm="P2"), "INCONCLUSIVE", False,
                                "NOT CONFIRMED", "neither primary contrast"),
        "both pass, copying increased": (P(-0.4, True, True, True), P(0.6, True, True, True, nm="P2"), "PASS", True,
                                         "CONFIRMED", "Both point estimates"),
    }
    out, ok = {}, True
    for name, (a, b, gs, cpi, prefix, frag) in cases.items():
        res = {"primary": {"P1": a, "P2": b, "ledger_copying_increased": {"point": 0.05 if cpi else 0.0, "ci95": [0.01, 0.09] if cpi else [-0.01, 0.01],
                                                                         "increased": cpi}},
               "forgetting_guard": guard(gs), "dry_run": False}
        w = wording(res)
        good = w["headline"].startswith(prefix) and any(frag in s for s in w["statement"]) and w["limits"] == list(LIMITS)
        if gs == "FAIL":
            good = good and "forgetting guard FAIL" in w["headline"]
        else:
            good = good and "guard FAIL" not in w["headline"]
        if gs == "INCONCLUSIVE":
            good = good and w["guard"].startswith("Forgetting guard INCONCLUSIVE")
        if cpi:
            good = good and any("increased copying" in q for q in w["qualifiers"])
        out[name] = {"headline": w["headline"], "ok": bool(good)}
        ok = ok and good
    return {"cases": out, "ok": bool(ok)}


# ---------------------------------------------------------------- power (dry run only)


def power_block(per_parent: dict) -> dict:
    """(a) Normal approximation at n = 150 from the pilot per-parent SDs (homogeneous effects).
    (b) Joint power for 'claim confirmed' by simulation with the registered bootstrap and Holm code, resampling the 26 pilot
    parents' paired differences under three effect models at coherent (P1, P2) effect pairs:
      scaled   every parent improves by lambda x its pilot gain (SD shrinks with the effect)
      mixture  each parent is fully 'fixed' with probability lambda, else unchanged (SD inflates: heterogeneous effects)
      sd_upper scaled, plus noise so SD(d1) = 0.433 and SD(d2) = 0.251 (upper 95% chi-square bounds of the pilot SDs)."""
    from statistics import NormalDist
    Phi = NormalDist().cdf
    out = {"normal_approx": {}, "simulation": {}}
    specs = {"P1 (L regret ON-OFF)": ("L|chain_one|k4|regret_on", "L|chain_one|k4|regret_off", P1_THRESHOLD, -1),
             "P2 (L exact ON-OFF)": ("L|chain_one|k4|exact_on", "L|chain_one|k4|exact_off", P2_THRESHOLD, +1),
             "S (R regret ON-OFF)": ("R|chain_one|k4|regret_on", "R|chain_one|k4|regret_off", 0.0, -1),
             "S (P regret ON-OFF, chain_m1)": ("P|chain_m1|k4|regret_on", "P|chain_m1|k4|regret_off", 0.0, -1)}
    for name, (a, b, thr, sgn) in specs.items():
        d = per_parent[a] - per_parent[b]
        d = d[~np.isnan(d)]
        sd = float(d.std(ddof=1))
        eff = float(sgn * d.mean())             # effect in the claimed direction (positive = as claimed)
        rows = {}
        for label, mu in (("pilot effect", eff), ("pilot x 0.5", eff * 0.5), ("threshold + 0.05", abs(thr) + 0.05)):
            for sdm in (1.0, 1.25, 2.0):
                s = sd * sdm
                se150 = s / math.sqrt(150)
                t = abs(thr)
                rows[f"{label} (mu {mu:.3f}), SD x{sdm}"] = {
                    "pass_n150": Phi((mu - max(t, 1.959964 * se150)) / se150) if t > 0 else Phi(mu / se150 - 1.959964),
                    "strong_pass_n150": Phi((mu - t) / se150 - 1.644854) if t > 0 else None, "se_n150": se150}
        out["normal_approx"][name] = {"pilot_n_parents": int(len(d)), "pilot_mean_in_claim_direction": eff, "pilot_sd": sd,
                                      "threshold": thr, "scenarios": rows}
    d1 = per_parent["L|chain_one|k4|regret_on"] - per_parent["L|chain_one|k4|regret_off"]
    d2 = per_parent["L|chain_one|k4|exact_on"] - per_parent["L|chain_one|k4|exact_off"]
    keep = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[keep], d2[keep]
    e1 = -float(d1.mean())
    rng = np.random.default_rng(POWER_SIM["seed"])
    N, NS, NB = POWER_SIM["n"], POWER_SIM["n_sim"], POWER_SIM["n_boot"]
    sd1u, sd2u = 0.433, 0.251
    for model in ("scaled", "mixture", "sd_upper"):
        for target in (e1, 0.5 * e1, 0.20, 0.169):
            lam = target / e1
            r = []
            for _ in range(NS):
                i = rng.integers(0, len(d1), N)
                if model == "scaled":
                    a, b = lam * d1[i], lam * d2[i]
                elif model == "mixture":
                    fix = rng.random(N) < lam
                    a, b = np.where(fix, d1[i], 0.0), np.where(fix, d2[i], 0.0)
                else:
                    a = lam * d1[i] + rng.normal(0, math.sqrt(max(sd1u ** 2 - (lam * d1.std(ddof=1)) ** 2, 0)), N)
                    b = lam * d2[i] + rng.normal(0, math.sqrt(max(sd2u ** 2 - (lam * d2.std(ddof=1)) ** 2, 0)), N)
                q1, q2 = boot_mean(a, seed=int(rng.integers(1 << 30)), n_boot=NB), boot_mean(b, seed=int(rng.integers(1 << 30)), n_boot=NB)
                h = holm({"P1": q1["p_neg"], "P2": q2["p_pos"]})
                s1 = h["P1"]["holm_significant"] and q1["point"] <= P1_THRESHOLD
                s2 = h["P2"]["holm_significant"] and q2["point"] >= P2_THRESHOLD
                r.append((s1, s2, s1 and s2, s1 and s2 and q1["upper95_one_sided"] <= P1_THRESHOLD and q2["lower95_one_sided"] >= P2_THRESHOLD))
            r = np.array(r, float).mean(0)
            out["simulation"][f"{model}: P1 effect {target:.3f}, P2 effect {lam * float(d2.mean()):.3f}"] = {
                "P1_pass": round(float(r[0]), 3), "P2_pass": round(float(r[1]), 3), "claim_confirmed": round(float(r[2]), 3),
                "both_strong": round(float(r[3]), 3)}
    out["simulation_settings"] = dict(POWER_SIM, note="registered boot_mean/holm with n_boot reduced for speed; "
                                                      "pilot parents resampled with replacement")
    return out


# ---------------------------------------------------------------- render


def render(res: dict) -> str:
    f3 = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    sg = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"
    ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    pf = lambda p: "<0.0001" if p < 1e-4 else f"{p:.4f}"
    L = []
    title = "PREREG-2 DRY RUN (code check on pilot TEST, T2 seed 0; NOT confirmatory)" if res["dry_run"] else "PREREG-2 results"
    L.append(f"# {title}\n")
    L.append(f"Rendered mechanically by `scripts/confirm2_analysis.py` from the results JSON. Bank: {res['integrity']['bank']} "
             f"(sha {res['integrity']['bank_sha256'][:12]}...).\n")
    if res["dry_run"]:
        L.append("The dry run uses pilot TEST, which was used to target the failures T2 fixes (R13), so every number below is "
                 "exploratory and only checks the code path. n = 26 ledger parents, not 150.\n")
    w = res["wording"]
    L.append(f"## Verdict: {w['headline']}\n")
    for s in w["statement"]:
        L.append(f"> {s}\n")
    if w["qualifiers"]:
        L.append("Qualifiers:\n")
        L += [f"- {q}" for q in w["qualifiers"]]
        L.append("")
    L.append(f"Off-task: {w['guard']}\n")
    L.append("Limits that always accompany the claim:\n")
    L += [f"- {q}" for q in w["limits"]]
    L.append("")
    L += [f"{q}\n" for q in w["descriptive"]]
    L.append("\n## Primary contrasts (ledger, chain_one, k = 4; Holm m = 2)\n")
    L.append("| | contrast | OFF | ON | ON - OFF | 95% CI | one-sided 95% bound | one-sided p | Holm p | threshold | PASS | strong | reversed | halves A / B (descriptive) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ("P1", "P2"):
        p = res["primary"][n]
        hs = " / ".join(sg(h["point"]) for h in p["halves_descriptive"].values())
        L.append(f"| {n} | {p['contrast']} | {f3(p['off_mean'])} | {f3(p['on_mean'])} | {sg(p['point'])} | {ci(p['ci95'])} | "
                 f"{sg(p['one_sided_bound'])} | {pf(p['p_one_sided'])} | {pf(p['p_holm'])} | {p['threshold']:+.2f} | {p['PASS']} | "
                 f"{p['strong_pass']} | {p['reversed']} | {hs} |")
    p1, p2 = res["primary"]["P1"], res["primary"]["P2"]
    cp = res["primary"]["ledger_copying_increased"]
    L.append(f"\nFloors (expected never to bind on ledger): F_touched {f3(p1['F_touched_mean'])} (ON below it: "
             f"{p1['on_beats_F_touched']}); copy-chain exact floor {f3(p2['copy_chain_floor'])} (ON exceeds it by >= 0.10: "
             f"{p2['on_exceeds_copy_floor_by_margin']}). Rule-ablated F_ignore_status "
             f"{f3(p1['F_ignore_status_mean (rule-ablated diagnostic)'])} (diagnostic). Ledger chain_one false-copy ON - OFF "
             f"{sg(cp['point'])} {ci(cp['ci95'])} (increased: {cp['increased']}).\n")
    g = res["forgetting_guard"]
    L.append("## Forgetting guard (registered; not part of the claim)\n")
    L.append(f"- Status: **{g.get('status')}** (PASS iff the one-sided 95% upper bound of r < 1; FAIL iff the lower bound > 1)")
    if isinstance(g.get("kl_off_on"), dict):
        L.append(f"- KL(OFF||ON) mean {g['kl_off_on']['mean']:.4f} (median {g['kl_off_on']['median']:.4f}); top-1 agreement "
                 f"{f3(g.get('top1_agree_off_on'))}")
    if isinstance(g.get("kl_ref_supa_sparse"), dict):
        r = g["ratio_kl_adapter_over_ref"]
        rc = g["cumulative_ratio_kl_supa_on_over_supa_sparse"]
        L.append(f"- KL(Supa||Sparse) mean {g['kl_ref_supa_sparse']['mean']:.4f}; r = {r['ratio']:.3f} "
                 f"[{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}], one-sided bounds {r['ratio_lower95_one_sided']:.3f} / "
                 f"{r['ratio_upper95_one_sided']:.3f} (reply-class cluster bootstrap, {r['n_clusters']} classes)")
        L.append(f"- Cumulative KL(Supa||ON) mean {g['kl_supa_on']['mean']:.4f}; ratio to KL(Supa||Sparse) {rc['ratio']:.3f} "
                 f"[{rc['ratio_ci95'][0]:.3f}, {rc['ratio_ci95'][1]:.3f}] (reading bound {GUARD_CUMULATIVE_READING}: "
                 f"{'within' if g['cumulative_reading_le_1_25'] else 'beyond'})")
    else:
        L.append(f"- KL(Supa||Sparse): {g.get('kl_ref_supa_sparse')}")
    if "greedy64_exact_agree_off_on" in g:
        L.append(f"- 64-token greedy agreement OFF/ON {f3(g['greedy64_exact_agree_off_on'])}; Supa/Sparse "
                 f"{f3(g.get('greedy64_exact_agree_supa_sparse'))}; Supa/ON {f3(g.get('greedy64_exact_agree_supa_on'))}; JSON bleed "
                 f"(non-numeric classes) OFF {f3(g['json_bleed_nonnumeric']['off'])} ON {f3(g['json_bleed_nonnumeric']['on'])}")
    if "guard_selftest" in res:
        L.append(f"- Guard code self-test on synthetic distributions: ok = {res['guard_selftest']['ok']}")
    if "wording_selftest" in res:
        L.append(f"- Wording self-test ({len(res['wording_selftest']['cases'])} synthetic outcome branches): ok = {res['wording_selftest']['ok']}")
    L.append("\n## Cells (OFF -> ON; per-parent paired bootstrap)\n")
    L.append("| cell | role | k | parents | regret OFF | regret ON | ON - OFF [95% CI] | exact OFF | exact ON | ON - OFF [95% CI] | parse OFF / ON |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, c in res["cells"].items():
        for k, v in c["k"].items():
            L.append(f"| {key} | {c['role']} | {k} | {v['n_parents']} | {f3(v['regret_off'])} | {f3(v['regret_on'])} | "
                     f"{sg(v['regret_on_minus_off']['point'])} {ci(v['regret_on_minus_off']['ci95'])} | {f3(v['exact_off'])} | "
                     f"{f3(v['exact_on'])} | {sg(v['exact_on_minus_off']['point'])} {ci(v['exact_on_minus_off']['ci95'])} | "
                     f"{f3(v['parse_rate_off'])} / {f3(v['parse_rate_on'])} |")
    L.append("\n## False copy (own-input convention) and sensitivity\n")
    L.append("| cell | false-copy OFF | ON | ON - OFF [95% CI] | k4 regret, last-parsed fill | k4 regret, goals seed 1004 | k4 regret, t2_train-overlap parents dropped (n) | k4 regret, any-overlap parents dropped (n) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, c in res["cells"].items():
        fc, s = c["false_copy"], c["sensitivity_k4"]
        L.append(f"| {key} | {f3(fc['off']['pooled_rate'])} | {f3(fc['on']['pooled_rate'])} | {sg(fc['on_minus_off']['point'])} "
                 f"{ci(fc['on_minus_off']['ci95'])} | {sg(s['fill_last_parsed']['point'])} | {sg(s['goals_seed_1004']['point'])} | "
                 f"{sg(s['drop_t2_train_overlap']['regret_on_minus_off']['point'])} ({s['drop_t2_train_overlap']['n_dropped']}) | "
                 f"{sg(s['drop_any_overlap']['regret_on_minus_off']['point'])} ({s['drop_any_overlap']['n_dropped']}) |")
    L.append(f"\n## Secondary contrasts (two-sided, BH q = 0.10)\n\n{res['secondary_note']}\n")
    L.append("| contrast | ON - OFF | 95% CI | p (two-sided) | BH p | BH sig |")
    L.append("|---|---|---|---|---|---|")
    for k, v in res["secondary"].items():
        L.append(f"| {k} | {sg(v['point'])} | {ci(v['ci95'])} | {pf(v['p_two_sided']) if not math.isnan(v['p_two_sided']) else 'n/a'} | "
                 f"{pf(v['p_bh']) if 'p_bh' in v else 'n/a'} | {v.get('bh_significant', 'n/a')} |")
    L.append("\n## Pre-declared directional predictions\n")
    for k, v in res["predictions"].items():
        L.append(f"- {k}: **{'holds' if v['holds'] else 'does not hold'}** ({json.dumps({a: b for a, b in v.items() if a != 'holds'}, default=str)})")
    L.append("\n## Step accuracy on correct-input chain steps (descriptive; arm-specific step sets)\n")
    for key, c in res["cells"].items():
        sa = c["step_accuracy_correct_input"]
        parts = [f"{op} {f3(sa['off'].get(op, {}).get('acc'))}->{f3(sa['on'].get(op, {}).get('acc'))} (n {sa['off'].get(op, {}).get('n', 0)}/{sa['on'].get(op, {}).get('n', 0)})"
                 for op in sorted(set(sa["off"]) | set(sa["on"]))]
        L.append(f"- {key}: " + "; ".join(parts))
    if "void_split_k4" in res["cells"].get("L|chain_one", {}):
        vs = res["cells"]["L|chain_one"]["void_split_k4"]
        L.append("- Ledger chain_one k=4 exact by plan type (pooled; parent-paired ON - OFF): " + "; ".join(
            f"{n}: {f3(v['off'])}->{f3(v['on'])} (n {v['n_candidates']}; {sg(v['parent_paired']['point'])} "
            f"{ci(v['parent_paired']['ci95'])}, {v['parent_paired']['n']} parents)" for n, v in vs.items()))
    L.append("\n## Floors (normalised regret; copy-chain exact floor)\n")
    for key, v in res["floors"].items():
        L.append(f"- {key}: " + ", ".join(f"{a} {f3(b)}" for a, b in sorted(v.items())))
    if "repro_check" in res:
        L.append("\n## Dry-run reproduction of R13 / T2_RESULTS (code check)\n")
        for k, v in res["repro_check"].items():
            L.append(f"- {k}: {v}")
    if "power" in res:
        L.append("\n## Power at the registered sizes\n")
        L.append("(a) Normal approximation from pilot per-parent SDs (homogeneous effects):\n")
        for name, p in res["power"]["normal_approx"].items():
            L.append(f"- {name}: pilot n {p['pilot_n_parents']}, pilot effect {p['pilot_mean_in_claim_direction']:+.3f}, "
                     f"per-parent SD {p['pilot_sd']:.3f}, threshold {p['threshold']:+.2f}")
            for s, v in p["scenarios"].items():
                extra = "" if v["strong_pass_n150"] is None else f", strong {v['strong_pass_n150']:.3f}"
                L.append(f"  - {s}: pass (n 150) {v['pass_n150']:.3f}{extra}; SE {v['se_n150']:.4f}")
        L.append(f"\n(b) Joint power by simulation ({res['power']['simulation_settings']}):\n")
        L.append("| model / effects | P1 PASS | P2 PASS | claim confirmed | both strong |")
        L.append("|---|---|---|---|---|")
        for s, v in res["power"]["simulation"].items():
            L.append(f"| {s} | {v['P1_pass']:.3f} | {v['P2_pass']:.3f} | {v['claim_confirmed']:.3f} | {v['both_strong']:.3f} |")
    L.append(f"\nInputs: {json.dumps(res.get('inputs', {}), default=str)}\n")
    return "\n".join(L) + "\n"


def repro_check(I: Inputs, res: dict, per_parent: dict) -> dict:
    """Dry run: the numbers R13 / T2_RESULTS report, recomputed through this code (and, for prerequisite regret, through
    t2_analysis's two-mode g_only call, whose tie-break stream reaches chain_m1 second)."""
    Lc = res["cells"]["L|chain_one"]["k"][4]
    pb = decide.paired_bootstrap(per_parent["L|chain_one|k4|regret_off"], per_parent["L|chain_one|k4|regret_on"])
    Pc = res["cells"]["P|chain_m1"]["k"][4]
    gP = {m: {s: (I.gen[("P", m, s)] if ("P", m, s) in I.gen else
                  load_chains(I.chain_path(m, "P", s), I.cands["P"], f"{m}_P_{s}", filter_family=True)) for s in ("off", "on")}
          for m in ("chain_one", "chain_m1")}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        two = {s: decisions.g_only(I.rows, "prerequisite_plan", 4, {m: gP[m][s] for m in ("chain_one", "chain_m1")},
                                   modes=("chain_one", "chain_m1"))["chain_m1"] for s in ("off", "on")}
    two_off, two_on = (float(np.nanmean(np.asarray(two[s]["nregret"], float))) for s in ("off", "on"))
    out = {
        "L chain_one k4 regret OFF/ON (R13: 0.507 / 0.038)": [round(Lc["regret_off"], 4), round(Lc["regret_on"], 4)],
        "T2 drop via decide.paired_bootstrap (R13: +0.469 [+0.354, +0.593])": [round(pb["mean_diff"], 4), [round(x, 4) for x in pb["ci95"]]],
        "P1 equals minus the T2 drop, CI mirrored": bool(abs(pb["mean_diff"] + res["primary"]["P1"]["point"]) < 1e-12
                                                         and abs(pb["ci95"][0] + res["primary"]["P1"]["ci95"][1]) < 1e-12),
        "L chain_one k4 exact OFF/ON (R13: 0.024 / 0.683)": [round(Lc["exact_off"], 4), round(Lc["exact_on"], 4)],
        "R chain_one k4 regret OFF/ON (R13: 0.386 / 0.323)": [round(res["cells"]["R|chain_one"]["k"][4]["regret_off"], 4),
                                                              round(res["cells"]["R|chain_one"]["k"][4]["regret_on"], 4)],
        "P chain_m1 k4 exact OFF/ON (R13: 0.018 / 0.330)": [round(Pc["exact_off"], 4), round(Pc["exact_on"], 4)],
        "P chain_m1 k4 regret OFF/ON, t2_analysis two-mode g_only call (R13: 0.878 / 0.685)": [round(two_off, 4), round(two_on, 4)],
        "P chain_m1 k4 regret OFF/ON, registered single-mode call (differs from R13 only by the tie-break stream order)":
            [round(Pc["regret_off"], 4), round(Pc["regret_on"], 4)],
    }
    R4 = res["cells"]["R|chain_one"]["k"][4]
    checks = [(Lc["regret_off"], 0.507), (Lc["regret_on"], 0.038), (Lc["exact_off"], 0.024), (Lc["exact_on"], 0.683),
              (R4["regret_off"], 0.386), (R4["regret_on"], 0.323), (Pc["exact_off"], 0.018), (Pc["exact_on"], 0.330),
              (two_off, 0.878), (two_on, 0.685), (pb["mean_diff"], 0.469)]
    out["reproduces R13 to the reported 3 decimals (L, R, P exact; P regret via the two-mode call)"] = bool(
        all(abs(x - r) <= 5e-4 + 1e-12 for x, r in checks))
    return out


def _write_together(out_dir: Path, files: dict) -> None:
    """Create-only outputs written as a set: every file to <name>.partial first, then each renamed into place
    (os.rename refuses an existing target on Windows; existence is also checked first)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (out_dir / name).exists():
            raise SystemExit(f"{out_dir / name} exists: the registered analysis runs once (create-only)")
    tmp = {}
    for name, writer in files.items():
        t = out_dir / f"{name}.partial"
        writer(t)
        tmp[name] = t
    for name, t in tmp.items():
        os.rename(t, out_dir / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="code check on pilot TEST with runs/t2/t2_s0 (not confirmatory)")
    ap.add_argument("--out-dir", default=None, help="dry run only: output directory (default docs/prereg2, refused once frozen)")
    a = ap.parse_args()
    t0 = time.time()
    if a.out_dir and not a.dry_run:
        raise SystemExit("--out-dir is for the dry run only")
    names = ("PREREG_2_RESULTS.json", "PREREG_2_RESULTS_PER_PARENT.npz", "PREREG_2_RESULTS.md")
    if a.dry_run:
        out_dir = Path(a.out_dir).resolve() if a.out_dir else DRY_OUT
        if out_dir == DRY_OUT.resolve() and (ROOT / G2.FREEZE_REL).exists():
            raise SystemExit("PREREG-2 is frozen: docs/prereg2/PREREG_2_DRYRUN.* are frozen inputs; use --out-dir elsewhere")
    else:
        out_dir = RUN_DIR / "analysis"
        for f in names:
            if (out_dir / f).exists():
                raise SystemExit(f"{out_dir / f} exists: the registered analysis runs once (create-only)")
    I = Inputs(a.dry_run)
    res, per_parent = analyse(I)
    if a.dry_run:
        res["guard_selftest"] = guard_selftest()
        res["wording_selftest"] = wording_selftest()
        need(res["guard_selftest"]["ok"], "guard self-test failed")
        need(res["wording_selftest"]["ok"], f"wording self-test failed: {res['wording_selftest']}")
        res["power"] = power_block(per_parent)
        res["repro_check"] = repro_check(I, res, per_parent)
    res["inputs"] = {"code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in
                                     [Path(__file__), ROOT / "scripts" / "t2_analysis.py", ROOT / "scripts" / "confirm2_gen.py"]
                                     + sorted((ROOT / "rcwmt").glob("*.py"))},
                     "numpy": np.__version__, "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1)}
    need("torch" not in sys.modules, "torch was imported")
    md = render(res)
    js = json.dumps(T2.clean(res), indent=1) + "\n"

    def w_text(s):
        def f(p):
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(s)
        return f

    def w_npz(p):
        with open(p, "wb") as fh:
            np.savez(fh, **per_parent)
    if a.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        w_text(js)(out_dir / "PREREG_2_DRYRUN.json")
        w_npz(out_dir / "PREREG_2_DRYRUN_PER_PARENT.npz")
        w_text(md)(out_dir / "PREREG_2_DRYRUN.md")
        print(f"wrote {out_dir / 'PREREG_2_DRYRUN.json'} (+ .md, per-parent .npz)")
    else:
        _write_together(out_dir, {names[0]: w_text(js), names[1]: w_npz, names[2]: w_text(md)})
        print(f"wrote {out_dir / names[0]} (+ .md, per-parent .npz)")
    p1, p2 = res["primary"]["P1"], res["primary"]["P2"]
    for n, p in (("P1", p1), ("P2", p2)):
        print(f"PRIMARY {n} {p['contrast']}: OFF {p['off_mean']:.3f} ON {p['on_mean']:.3f} diff {p['point']:+.3f} "
              f"CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] p1 {p['p_one_sided']:.4g} holm {p['p_holm']:.4g} -> "
              f"{'PASS' if p['PASS'] else 'FAIL'} (strong {p['strong_pass']}; reversed {p['reversed']}); halves "
              f"{p['halves_descriptive']['A']['point']:+.3f} / {p['halves_descriptive']['B']['point']:+.3f} (descriptive)")
    print(f"forgetting guard: {res['forgetting_guard'].get('status')}")
    print(f"verdict: {res['wording']['headline']}; claim confirmed: {res['claim_confirmed']}")


if __name__ == "__main__":
    main()
