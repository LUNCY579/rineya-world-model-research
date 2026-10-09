"""PREREG-3 analysis (PREREG_3_DRAFT.md §4-§8). CPU only: numpy/json; never loads a model, never imports torch.

  python -B scripts/p3_analysis.py --selftest   # scorer self-test on the sealed bank (no model output is read):
                                                # writes docs/prereg3/SCORER_SELFTEST.json (refused once frozen)
  python -B scripts/p3_analysis.py --dry-run    # statistics unit tests; wording self-test (every branch, through the
                                                # REAL summarise / McNemar / Holm on synthetic per-row classes); synthetic
                                                # fixture runs (8 scenarios) through the full pipeline incl. input
                                                # validation; tamper refusals; power -> docs/prereg3/P3_DRYRUN.*
  python -B scripts/p3_analysis.py              # the registered analysis on runs/bank_p3_oor (refuses unless the
                                                # PREREG-3 freeze verifies and the scorer self-test passes FIRST);
                                                # create-only results set in runs/bank_p3_oor/analysis/

Arms: OFF (Titan-Sparse, adapters disabled), s0 (T2 seed 0, PREREG-2's registered artifact), s1, s2 (replications),
Supa 1.7, COPY (answer = before; no model). One-step rows are scored correct iff the parsed answer equals the gold
balances (p3_common.score_onestep). Chains: exact state at k = 1/2/4 and, through PREREG-2's own code path
(confirm2_analysis.gen_regret == rcwmt.decisions.g_only), normalised decision regret over the 256 seed-1001 goals.

Decision rules (verbatim in PREREG_3_DRAFT.md §5; thresholds fixed before any output). "bar" = accuracy >= 0.80 AND
Wilson 95% lower bound >= 0.70. Unseen-status strata (8): U_pending, U_draft, U_cancelled, U_reversed, U_held,
U_neutral, N_voidcase, N_postedlike. Twin guard (per seed): T_posted accuracy >= 0.80 AND T_posted copied rate <= 0.10.
  V1 STATUS.  RULE    iff for EACH of s0, s1, s2: every unseen-status stratum meets the bar, and the twin guard holds.
              TRIGGER iff for EACH of s0, s1, s2: every unseen-status stratum without a form of 'void' (the five
                      not-applied words, U_neutral, N_postedlike) has accuracy <= 0.30, C_void accuracy >= 0.80, and
                      'applied_as_posted' is more than half of the wrong answers on those seven strata.
              PARTIAL otherwise (per-stratum, per-seed table; named patterns: twins, semantic, posted_substring,
                      trigger_not_interpretable, no_adapter_gain).
              Attribution (wording only; RULE): exact McNemar ON(seed) vs OFF on the 24 unseen-status cells, Holm over
              the 24. base_meets = OFF meets the bar on every unseen-status stratum AND OFF's twins are copied <= 0.10.
              all_sig = all 24 Holm-significant with ON > OFF. strengthened (base_meets, all_sig) / already present
              (base_meets, not all_sig) / learned (not base_meets, all_sig) / mixed (neither).
  V2-V4 GROUPS over (strata, kind class) cells per seed: MAGNITUDE add/transfer (M_bal|at, M_amt|at; 67 rows each),
              MAGNITUDE scale (M_bal+M_amt|sc; 66 rows: multi-digit multiplication), NAMING (A_renamed, A_many),
              FORMAT (F_keyorder). Per cell exact McNemar ON vs OFF, Holm over all 18 cells; floor = the bar. Cell
              classes HARMED / HIGH_GAIN / HIGH_NOGAIN / GAIN_BELOW_FLOOR / LOW_NOGAIN; group verdict HARMED /
              GENERALISES / ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN / DOES NOT GENERALISE[, with a partial gain over
              OFF] / MIXED, in that order.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import confirm2_analysis as C2  # noqa: E402  (PREREG-2's analysis: bootstrap, Holm, gen_regret, floors; torch-free)
import p3_common as C  # noqa: E402
import p3_run as R  # noqa: E402  (freeze verifier, stage files; torch-free at import)
from rcwmt import data, env  # noqa: E402

if "torch" in sys.modules:
    raise SystemExit("torch must not be imported by the PREREG-3 analysis")

# ---------------------------------------------------------------- registered constants
ARMS = ("off", "s0", "s1", "s2", "supa", "copy")
SEEDS = ("s0", "s1", "s2")
ARM_LABEL = {"off": "OFF (Titan-Sparse)", "s0": "T2 s0", "s1": "T2 s1", "s2": "T2 s2", "supa": "Supa 1.7", "copy": "copy floor"}
RULE_ACC, RULE_LB = 0.80, 0.70                 # the bar
TWIN_ACC, TWIN_COPY_MAX = 0.80, 0.10           # the twin guard
TRIGGER_ACC, TRIGGER_CVOID_MIN = 0.30, 0.80
GEN_ACC, GEN_LB = 0.80, 0.70
NOVELTY_COPY_MAX = 0.20                        # novelty-fallback qualifier: ON copied rate over the M/A/F rows
ALPHA = 0.05
E1_VOID_ON, E2_VOID_OFF = 0.95, 0.30
EPS = 1e-12
KS = (1, 2, 4)
PHI_CLIP = 300                                 # rcwmt.data.phi_from_state clips final/100 to [-3, 3]
OUT_NAMES = ("PREREG_3_RESULTS.json", "PREREG_3_RESULTS_PER_ROW.npz", "PREREG_3_RESULTS.md")
DRY_SEED = 20261104
GROUP_TWIN_PART = {"magnitude_at": "at", "magnitude_scale": "sc", "naming": "all", "format": "all"}
ATTR_HEAD = {"learned": "RULE (learned by the adapter)",
             "strengthened": "RULE (present in the base at the bar, strengthened by the adapter)",
             "base_already": "RULE (already present without the adapter)",
             "mixed": "RULE (attribution mixed)"}
WORDS_TEXT = ("five words that mean 'not applied' (pending, draft, cancelled, reversed, held), neutral and nonce words "
              "without that meaning (approved, active, ok, dax, wug), near-miss spellings of 'void' (Void, VOID, voided) "
              "and look-alikes of 'posted' (unposted, not_posted, 'not posted')")
LIMITS = (
    "Ledger only: Titan's one-step frame (1,500 rows: 100 bases x 15 strata) plus 80 four-step chain_one chains; T2 "
    "adapters s0 (PREREG-2's registered artifact), s1 and s2, unmerged on frozen Titan-Sparse.",
    "'Rule' is tested on 16 non-posted strings: five words that mean 'not applied', five neutral or nonce words, three "
    "near-miss spellings of 'void' and three look-alikes of 'posted'. It is not a claim about every possible status "
    "string.",
    "Case variants of 'posted' ('Posted', 'POSTED') and synonyms of it (booked, settled, ...) are not tested: the frame's "
    "rule text does not settle them.",
    "Out-of-range means outside T2's training rows (statuses posted/void, amounts -9..12, |before| <= 98, accounts A-D, "
    "sorted keys); Titan's own 384 training rows share these ranges; Supa's pretraining data are unknown. T_posted and "
    "C_void are fresh in-distribution controls, and CH_S events are in range.",
    "One factor at a time: every one-step stratum is the same 100 bases with one factor changed (status string, "
    "balances, amount, account names, number of accounts, key order), so each out-of-range effect is a paired, "
    "kind-matched contrast with the posted twins; magnitude is judged separately for add/transfer and for scale "
    "(multi-digit multiplication); the CH_X chains mix factors and are descriptive.",
    "No training and no self-distillation or untargeted-LoRA control: a verdict describes what the adapted model does, "
    "not why; PREREG-2's other limits (horizon <= 4, one frame, no non-LM comparator at matched labels) still apply.",
    "Thresholds are absolute (the bar: accuracy 0.80 with Wilson lower bound 0.70; twins copied <= 0.10; trigger 0.30) "
    "and were fixed before any output. Under independence a true accuracy of 0.85 passes one cell with probability "
    "about 0.93 but all 24 unseen-status cells with about 0.19; 0.90 passes all 24 with about 0.98 (power table in the "
    "dry run; the strata share bases, so the joint probabilities are lower bounds).",
    "Chains are descriptive: K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to "
    "PREREG-2's 0.544 -> 0.056; phi clips balances at +-300, so regret is partly blind to large-magnitude errors in "
    "CH_X (exact@k is the primary chain measure there); 10 parents per stratum, so bootstrap intervals are indicative.",
)


def sha256(p: Path) -> str:
    return R.sha256(p)


def die(msg: str, errs=None) -> None:
    errs = errs if isinstance(errs, list) else ([str(errs)] if errs else [])
    raise SystemExit(f"p3_analysis: {msg}" + ("\n  " + "\n  ".join(errs[:12]) if errs else ""))


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


# ---------------------------------------------------------------- statistics
def mcnemar_exact(on: np.ndarray, off: np.ndarray) -> dict:
    """Exact two-sided McNemar (binomial on the discordant pairs, p = min(1, 2 P(X <= min(b, c))))."""
    on, off = np.asarray(on, bool), np.asarray(off, bool)
    b, c = int((on & ~off).sum()), int((~on & off).sum())
    return {"on_only": b, "off_only": c, "p_two_sided": mcnemar_p(b, c),
            "direction": "on>off" if b > c else ("off>on" if c > b else "tie")}


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def cell(k: int, n: int) -> dict:
    lo, hi = C.wilson(k, n)
    return {"n": n, "k": k, "acc": k / n if n else float("nan"), "wilson95": [lo, hi]}


def meets(c: dict, acc: float = RULE_ACC, lb: float = RULE_LB) -> bool:
    return c["n"] > 0 and c["acc"] >= acc - EPS and c["wilson95"][0] >= lb - EPS


def k_min(n: int, acc: float = RULE_ACC, lb: float = RULE_LB) -> int:
    k = math.ceil(acc * n - 1e-9)
    while k <= n and C.wilson(k, n)[0] < lb - EPS:
        k += 1
    return k


def binom_ge(n: int, p: float, k0: int) -> float:
    return float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(k0, n + 1)))


def p_twin_guard(n: int, p_acc: float, p_copy: float) -> float:
    """P(accuracy >= 0.80 AND copied rate <= 0.10) for a multinomial (correct p_acc, copied p_copy, other)."""
    k0, cmax = math.ceil(TWIN_ACC * n - 1e-9), math.floor(TWIN_COPY_MAX * n + 1e-9)
    q = max(0.0, 1 - p_acc - p_copy)
    tot = 0.0
    for k in range(k0, n + 1):
        for j in range(0, min(cmax, n - k) + 1):
            tot += math.comb(n, k) * math.comb(n - k, j) * p_acc ** k * p_copy ** j * q ** (n - k - j)
    return float(tot)


def mcnemar_power(n: int, p_on: float, p_off: float, m_tests: int, sims: int = 20000, seed: int = DRY_SEED) -> float:
    """Monte Carlo power of the exact McNemar at worst-case Holm (alpha / m) under independent ON / OFF errors."""
    tab = np.ones((n + 1, n + 1))
    for b in range(n + 1):
        for c in range(n + 1 - b):
            tab[b, c] = mcnemar_p(b, c)
    rng = np.random.default_rng(seed)
    on = rng.random((sims, n)) < p_on
    off = rng.random((sims, n)) < p_off
    b = (on & ~off).sum(1)
    c = (~on & off).sum(1)
    p = tab[b, c]
    return float(np.mean((p * m_tests < ALPHA) & (b > c)))


def power_table() -> dict:
    """Exact binomial / multinomial probabilities UNDER INDEPENDENCE (the strata share bases and errors are positively
    associated, so the joint probabilities are lower bounds: conservative for RULE detection); McNemar by simulation."""
    n = C.N_PER_STRATUM
    k_rule, k_trig = k_min(n), math.floor(TRIGGER_ACC * n + 1e-9)
    n_cells = len(C.UNSEEN_STATUS_STRATA) * len(SEEDS)
    n_trig = len(C.TRIGGER_STRATA) * len(SEEDS)
    n_gen = sum(len(c) for c in C.GEN_GROUPS.values()) * len(SEEDS)
    rows_of = lambda st, p: len(st) * {"at": 67, "sc": 33, "all": 100}[p]  # noqa: E731
    out = {"assumption": "independent cells (lower bounds for the joint probabilities: the strata share the 100 bases)",
           "n": n, "rule_min_correct": k_rule, "rule_min_correct_note": "accuracy >= 0.80 implies Wilson LB >= 0.70 at n = 100",
           "n_unseen_cells": n_cells, "trigger_max_correct": k_trig, "n_trigger_cells": n_trig, "n_gen_cells": n_gen,
           "floor_min_correct": {str(m): k_min(m) for m in (100, 67, 66, 33, 34, 20)}, "rule": [], "trigger": [], "groups": [],
           "mcnemar": []}
    tw90 = p_twin_guard(n, 0.90, 0.03)
    for p in (0.80, 0.85, 0.90, 0.95, 0.98):
        one = binom_ge(n, p, k_rule)
        twp = p_twin_guard(n, p, min(0.03, 1 - p))
        out["rule"].append({"true_acc_per_unseen_cell": p, "P_cell_meets_bar": round(one, 4),
                            f"P_all_{n_cells}_unseen_cells": round(one ** n_cells, 4),
                            "P_twin_guard_twins_at_0.90_copied_0.03": round(tw90, 4),
                            "P_RULE_twins_at_0.90": round(one ** n_cells * tw90 ** 3, 4),
                            "P_RULE_twins_at_same_acc_copied_min_0.03": round(one ** n_cells * twp ** 3, 4)})
    for p in (0.10, 0.20, 0.25, 0.30):
        one = 1 - binom_ge(n, p, k_trig + 1)
        out["trigger"].append({"true_acc": p, "P_cell_trigger_low": round(one, 4), f"P_all_{n_trig}_cells": round(one ** n_trig, 4)})
    for g, cells in C.GEN_GROUPS.items():
        for p in (0.85, 0.90, 0.95):
            pc = [binom_ge(rows_of(st, pt), p, k_min(rows_of(st, pt))) for st, pt in cells]
            out["groups"].append({"group": g, "rows_per_cell": [rows_of(st, pt) for st, pt in cells],
                                  "cells": len(cells) * len(SEEDS), "min_correct": [k_min(rows_of(st, pt)) for st, pt in cells],
                                  "true_acc": p, "P_cell_meets_floor": [round(x, 4) for x in pc],
                                  "P_all_cells_meet_floor": round(float(np.prod(pc)) ** len(SEEDS), 4)})
    for p in (0.85, 0.90):
        one = binom_ge(33, p, k_min(33))
        out["groups"].append({"group": "unpooled scale cells (NOT registered; why the scale cell pools M_bal and M_amt)",
                              "rows_per_cell": [33, 33], "cells": 6, "min_correct": [k_min(33)] * 2, "true_acc": p,
                              "P_cell_meets_floor": [round(one, 4)] * 2, "P_all_cells_meet_floor": round(one ** 6, 4)})
    for (m, a, b_, k) in ((100, 0.90, 0.60, n_cells), (100, 0.85, 0.70, n_cells), (100, 0.95, 0.80, n_cells),
                          (100, 0.90, 0.60, n_gen), (100, 0.85, 0.70, n_gen), (67, 0.90, 0.60, n_gen),
                          (66, 0.90, 0.60, n_gen), (66, 0.85, 0.60, n_gen)):
        out["mcnemar"].append({"rows": m, "on": a, "off": b_, "holm_worst_case_m": k,
                               "power": round(mcnemar_power(m, a, b_, k), 3)})
    return out


def stats_selftest() -> dict:
    """Unit tests of the exact McNemar and Holm (C2.holm) on known values, incl. the Holm boundary."""
    res = OrderedDict()
    cases = [((10, 0), 2 / 1024, "on>off"), ((9, 1), 2 * 11 / 1024, "on>off"), ((5, 5), 1.0, "tie"), ((0, 0), 1.0, "tie"),
             ((0, 7), 2 / 128, "off>on"), ((15, 0), 2 / 2 ** 15, "on>off"), ((4, 0), 0.125, "on>off")]
    ok = True
    for (b, c), want, d in cases:
        on = np.array([1] * b + [0] * c + [1] * 3 + [0] * 2, bool)
        off = np.array([0] * b + [1] * c + [1] * 3 + [0] * 2, bool)
        m = mcnemar_exact(on, off)
        good = abs(m["p_two_sided"] - want) < 1e-12 and m["direction"] == d and (m["on_only"], m["off_only"]) == (b, c)
        res[f"mcnemar b={b} c={c}"] = {"p": m["p_two_sided"], "want": want, "ok": good}
        ok &= good
    holm_cases = [({"a": 0.012, "b": 0.0126, "c": 0.03, "d": 0.5}, {"a": True, "b": True, "c": False, "d": False}),
                  ({"a": 0.01, "b": 0.011, "c": 0.04}, {"a": True, "b": True, "c": True}),
                  ({"a": 0.02, "b": 0.03}, {"a": True, "b": True}),
                  ({"a": 0.026, "b": 0.03}, {"a": False, "b": False}),
                  ({f"x{i}": 0.05 / 24 * 0.99 for i in range(24)}, {f"x{i}": True for i in range(24)}),
                  ({**{f"x{i}": 1e-6 for i in range(23)}, "y": 0.0499}, {**{f"x{i}": True for i in range(23)}, "y": True}),
                  ({**{f"x{i}": 1e-6 for i in range(23)}, "y": 0.0501}, {**{f"x{i}": True for i in range(23)}, "y": False})]
    for i, (ps, want) in enumerate(holm_cases):
        h = C2.holm(ps, ALPHA)
        got = {k: v["holm_significant"] for k, v in h.items()}
        mono = all(h[a]["p_holm"] <= h[b]["p_holm"] + 1e-15 for a in ps for b in ps if ps[a] <= ps[b])
        res[f"holm case {i}"] = {"got": got, "want": want, "monotone": mono, "ok": got == want and mono}
        ok &= got == want and mono
    return {"ok": bool(ok), "cases": res}


# ---------------------------------------------------------------- scorer self-test (no model output is read)
def _negatives(row: dict) -> list[str]:
    g = json.loads(row["gold"])
    ks = sorted(g)
    plus = dict(g)
    plus[ks[0]] = g[ks[0]] + 1
    miss = {k: g[k] for k in ks[:-1]}
    extra = dict(g)
    extra["zz"] = 0
    strv = dict(g)
    strv[ks[-1]] = str(g[ks[-1]])
    boolv = dict(g)
    boolv[ks[0]] = True
    out = [env.dumps(plus), env.dumps(miss), env.dumps(extra), json.dumps(strv), json.dumps(boolv), "", "I cannot answer that."]
    if row["gold_rule"] == "noop":
        out.append(row["posted_gold"])                  # applying a no-op event
    else:
        out.append(env.dumps(row["before"]))            # copying a posted event
    return out


def _gold_variants(row: dict) -> list[str]:
    g = json.loads(row["gold"])
    ko = row.get("key_order")
    ordered = {k: g[k] for k in (ko["before"] if ko else list(row["before"]))}
    return [row["gold"], json.dumps(g), json.dumps(ordered, separators=(",", ":")), row["gold"] + ".", json.dumps(g, indent=1),
            env.dumps({"before": row["before"], "after": g}), "  " + json.dumps(ordered) + "\n"]


def _parser_corpus(row: dict) -> list[str]:
    g = json.loads(row["gold"])
    ks = sorted(g)
    arith = "{" + ", ".join(f'"{k}": {row["before"][k]} + 0 = {g[k]}' for k in ks) + "}"
    trunc = env.dumps({"before": row["before"], "after": g}) + ' and then {"after": {"A": 1'
    return _gold_variants(row) + _negatives(row) + [arith, trunc, '{"after": 5}', "[1, 2]", "null", row["gold"][:-3]]


def scorer_selftest(bank_path: Path | None = None) -> dict:
    """Registered pre-condition (§5): on the sealed bank, BEFORE any model output is read, the scorer must (1) score every
    gold rendering correct (100%), (2) score the copy answer exactly as the copy floor (1 on every no-op row, 0 on every
    apply row), (3) score every negative wrong (0%), (4) parse A-D texts exactly as rcwmt.env.parse_state, and (5) give
    oracle chains exact 1.0 and regret 0, and copy chains the copy-chain floor and F_copy_s0."""
    rows, chains = C.load_bank(bank_path or C.BANK)
    res = OrderedDict()
    n_ok = n_all = 0
    for r in rows:
        for t in _gold_variants(r):
            n_all += 1
            n_ok += C.score_onestep(r, t)["correct"]
    res["gold renderings scored correct"] = {"n": n_all, "correct": n_ok, "ok": n_ok == n_all}
    per = defaultdict(lambda: [0, 0])
    exact = True
    for r in rows:
        sc = C.score_onestep(r, env.dumps(r["before"]))
        want = r["gold_rule"] == "noop"
        exact &= sc["correct"] == want
        per[r["stratum"]][0] += 1
        per[r["stratum"]][1] += sc["correct"]
    copy_acc = {s: v[1] / v[0] for s, v in per.items()}
    res["copy answer == copy floor (1 on no-op rows, 0 on apply rows)"] = {
        "per_stratum": copy_acc, "ok": exact and all(copy_acc[s] == (1.0 if C.STRATA[s][1] == "noop" else 0.0) for s in C.STRATA)}
    n_neg = n_wrong = 0
    cls_ok = True
    for r in rows:
        negs = _negatives(r)
        for t in negs:
            n_neg += 1
            n_wrong += not C.score_onestep(r, t)["correct"]
        last = C.score_onestep(r, negs[-1])["cls"]
        cls_ok &= last == ("applied_as_posted" if r["gold_rule"] == "noop" else "copied")
    res["negatives scored wrong (perturbed value, missing / extra / string / bool key, empty, prose, applied / copied)"] = {
        "n": n_neg, "wrong": n_wrong, "error_classes_ok": bool(cls_ok), "ok": n_wrong == n_neg and cls_ok}
    mism = n_par = 0
    for r in rows:
        keys = list(r["before"])
        if set(keys) != set(env.ACCOUNTS):
            continue
        for t in _parser_corpus(r):
            n_par += 1
            mism += C.parse_general(t, keys) != env.parse_state(C.LEDGER, t, lenient=True)
    res["parse_general == rcwmt.env.parse_state on A-D texts"] = {"n": n_par, "mismatches": mism, "ok": mism == 0}
    oracle = {c["cand_id"]: {"cand_id": c["cand_id"], "states": list(c["golds"]), "texts": list(c["golds"][1:])} for c in chains}
    copy = {c["cand_id"]: {"cand_id": c["cand_id"], "states": [c["golds"][0]] * (C.H + 1), "texts": [c["golds"][0]] * C.H} for c in chains}
    errs = chain_file_errors(list(oracle.values()), chains, "oracle") + chain_file_errors(list(copy.values()), chains, "copy")
    ch = chain_metrics(chains, {"oracle": oracle, "copy": copy})
    ok_or = all(ch[cs]["exact"]["oracle"][str(k)]["mean"] == 1.0 and np.nanmax(ch[cs]["regret_per_parent"]["oracle"][str(k)]) <= 1e-9
                for cs in C.CHAIN_STRATA for k in KS)
    ok_cp = all(abs(ch[cs]["exact"]["copy"][str(k)]["mean"] - ch[cs]["copy_chain_exact_floor"][str(k)]) < 1e-12 and
                np.allclose(ch[cs]["regret_per_parent"]["copy"][str(k)], ch[cs]["floors_per_parent"][str(k)]["F_copy_s0"],
                            equal_nan=True) for cs in C.CHAIN_STRATA for k in KS)
    res["chains: oracle exact 1.0 and regret 0; copy == copy-chain floor and F_copy_s0; files validate"] = {
        "validation_errors": errs[:5], "ok": not errs and ok_or and ok_cp}
    return {"ok": all(v["ok"] for v in res.values()), "checks": res, "bank_sha256": sha256(bank_path or C.BANK),
            "n_rows": len(rows), "n_chains": len(chains)}


# ---------------------------------------------------------------- input validation
def onestep_file_errors(lines: list, rows: list, name: str) -> list[str]:
    errs = []
    if len(lines) != len(rows):
        return [f"{name}: {len(lines)} lines for {len(rows)} bank rows"]
    for i, (l, r) in enumerate(zip(lines, rows)):
        if not isinstance(l, dict) or l.get("id") != r["id"] or not isinstance(l.get("text"), str) or set(l) != {"id", "text"}:
            errs.append(f"{name} line {i + 1}: not {{id: {r['id']}, text: str}}")
    return errs


def chain_file_errors(lines: list, chains: list, name: str) -> list[str]:
    """Every way a chain file can differ from what p3_run.chain_rows(p3_chain(...)) writes."""
    errs = []
    by = {c["cand_id"]: c for c in chains}
    ids = [l.get("cand_id") for l in lines]
    if len(ids) != len(set(ids)) or set(ids) != set(by) or ids != [c["cand_id"] for c in chains]:
        return [f"{name}: cand_id list differs from the bank's ({len(ids)} lines, {len(by)} chains)"]
    for ln, l in enumerate(lines):
        c = by[l["cand_id"]]
        st, tx = l.get("states"), l.get("texts")
        where = f"{name} line {ln + 1} ({c['cand_id']})"
        if not isinstance(st, list) or not isinstance(tx, list) or len(st) != len(c["events"]) + 1:
            errs.append(f"{where}: states/texts malformed")
            continue
        if st[0] != env.dumps(c["s0"]):
            errs.append(f"{where}: states[0] is not s0")
            continue
        fn = next((i for i, s in enumerate(st) if s is None), len(st))
        if any(s is not None for s in st[fn:]) or not all(isinstance(s, str) for s in st[:fn]):
            errs.append(f"{where}: states must be strings up to the first null")
            continue
        n_try = len(c["events"]) if fn == len(st) else fn
        if len(tx) != n_try or not all(isinstance(t, str) for t in tx):
            errs.append(f"{where}: {len(tx)} texts for {n_try} attempted steps")
            continue
        for t, (txt, s) in enumerate(zip(tx, st[1:])):
            ps = C.parse_ledger(txt, C.chain_keys(c))
            if (None if ps is None else env.dumps(ps)) != s:
                errs.append(f"{where}: state {t + 1} is not the parse of text {t + 1}")
                break
    return errs


def load_jsonl(path: Path) -> list:
    out = []
    for ln, l in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not l.strip():
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError as e:
            die(f"{path} is not valid JSONL", [f"line {ln + 1}: {e}"])
    return out


def provenance_errors(prov: dict, done: dict, run: Path, fz: dict | None, registered: bool, root: Path | None = None) -> list[str]:
    errs = []
    st = prov.get("stages") or {}
    if set(st) != set(R.STAGE_ORDER):
        errs.append(f"stages recorded {sorted(st)} != {list(R.STAGE_ORDER)}")
    for s in R.STAGE_ORDER:
        rec = st.get(s) or {}
        for p in R.stage_required(s, run):
            r = R.rel_out(p, run)
            if not p.is_file():
                errs.append(f"{r} missing")
            elif (rec.get("files") or {}).get(r) != sha256(p) or (done.get("files") or {}).get(s, {}).get(r) != sha256(p):
                errs.append(f"{r}: hash differs from provenance / DONE")
        sides = rec.get("sides") or {}
        pids = {v.get("pid") for v in sides.values() if isinstance(v, dict) and "pid" in v} | ({rec.get("pid")} if rec.get("pid") else set())
        if s != "A0" and len(pids) != 1:
            errs.append(f"stage {s}: sides not generated in one process ({pids})")
    if done.get("provenance_sha256") != sha256(run / "provenance.json"):
        errs.append("provenance.json changed after DONE.json")
    anc = R._read_json(run / "anchor" / "anchor.json") or {}
    if not anc.get("pass") or not (prov.get("anchor") or {}).get("pass"):
        errs.append("the anchor did not pass")
    turns = prov.get("turns") or []
    a0_turn = st.get("A0", {}).get("turn")
    for i, t in enumerate(turns):
        ran = set(t.get("stages_run") or [])
        if i != a0_turn and ran & {"S1", "S2"} and not (t.get("anchor_recheck") or {}).get("pass"):
            errs.append(f"turn {i} generated bank outputs without a passing re-anchor")
    if registered:
        allowed = R.allowed_hashes(fz, root)
        for i, t in enumerate(turns):
            for f, h in (t.get("code_sha256") or {}).items():
                if f in allowed and h not in allowed[f]:
                    errs.append(f"turn {i}: {f} differs from the frozen / amended version")
            for rk in ("runtime", "supa_runtime"):
                rt = t.get(rk) or {}
                for k in ("torch", "transformers"):
                    if rt and rt.get(k) != R.REGISTERED_RUNTIME[k]:
                        errs.append(f"turn {i}: {rk} {k} {rt.get(k)} != registered {R.REGISTERED_RUNTIME[k]}")
            if t.get("device") != "cuda":
                errs.append(f"turn {i}: not generated by the registered GPU backend (device {t.get('device')})")
            if t.get("versions") and t["versions"].get("peft") != R.REGISTERED_RUNTIME["peft"]:
                errs.append(f"turn {i}: peft {t['versions'].get('peft')} != registered")
            for s, fs in (t.get("adapter_sha256_at_load") or {}).items():
                for f, h in fs.items():
                    if h != fz["inputs"].get(f"{R.ADAPTER_RELS[s]}/{f}"):
                        errs.append(f"turn {i}: adapter {s} {f} at load differs from the freeze")
        if not str(prov.get("sparse_fingerprint", "")).startswith(R.FP["sparse"]):
            errs.append("Titan-Sparse fingerprint")
        if not str(prov.get("supa_fingerprint", "")).startswith(R.FP["supa"]):
            errs.append("Supa fingerprint")
        if prov.get("bank_sha256") != fz["inputs"]["data/bank_p3_oor.jsonl"]:
            errs.append("provenance bank hash differs from the freeze")
        if prov.get("registered") is not True or prov.get("freeze_prereg_sha256") != fz.get("prereg_sha256"):
            errs.append("the run was not made under this freeze (provenance 'registered' / prereg sha)")
    return errs


class Inputs:
    """Loads and validates one run directory. registered=True: the freeze must verify and every provenance check binds."""

    def __init__(self, run: Path, bank: Path, registered: bool, root: Path | None = None):
        self.run, self.bank_path = Path(run), Path(bank)
        self.fz = None
        if registered:
            ok, errs, fz = R.verify_freeze(root)
            if not ok:
                die("the PREREG-3 freeze does not verify; no analysis", errs)
            self.fz = fz
            need(sha256(self.bank_path) == fz["inputs"]["data/bank_p3_oor.jsonl"], "bank hash differs from the freeze")
        self.rows, self.chains = C.load_bank(self.bank_path)
        prov = read_json(self.run / "provenance.json", "provenance.json")
        done = read_json(self.run / "DONE.json", "DONE.json")
        errs = provenance_errors(prov, done, self.run, self.fz, registered, root)
        if errs:
            die("the run's provenance does not check", errs)
        self.prov = prov
        self.texts, self.chain_out = {}, {}
        files = {s: (self.run / "onestep" / f"onestep_{s}.jsonl", self.run / "chains" / f"chains_{s}.jsonl") for s in R.SIDES}
        files["supa"] = (self.run / "supa" / "onestep_supa.jsonl", self.run / "supa" / "chains_supa.jsonl")
        errs = []
        for arm, (fo, fc) in files.items():
            lo, lc = load_jsonl(fo), load_jsonl(fc)
            errs += onestep_file_errors(lo, self.rows, fo.name) + chain_file_errors(lc, self.chains, fc.name)
            self.texts[arm] = [l["text"] for l in lo] if len(lo) == len(self.rows) else []
            self.chain_out[arm] = {l["cand_id"]: l for l in lc}
        if errs:
            die("an output file is incomplete or malformed", errs)
        self.texts["copy"] = [env.dumps(r["before"]) for r in self.rows]
        self.chain_out["copy"] = {c["cand_id"]: {"cand_id": c["cand_id"], "states": [c["golds"][0]] * (C.H + 1),
                                                 "texts": [c["golds"][0]] * C.H} for c in self.chains}


# ---------------------------------------------------------------- chain metrics
def chain_metrics(chains: list, outs: dict) -> dict:
    """Per chain stratum: exact@k per arm (per-parent means), regret@k per parent through PREREG-2's gen_regret (the
    g_only path; unparseable -> parent-mean utility), answer-free floors, the copy-chain exact floor, step accuracy on
    correct-input steps by step class, and the number of gold states beyond the phi clip (+-300)."""
    res = {}
    for cs in C.CHAIN_STRATA:
        cc = [c for c in chains if c["stratum"] == cs]
        bank = C.chain_bank_dict(cc)
        rows = data.candidate_table(bank)
        pids = sorted({c["parent_id"] for c in cc})
        r = {"parents": len(pids), "candidates": len(cc), "exact": {}, "regret": {}, "regret_per_parent": {}, "step_accuracy": {},
             "copy_chain_exact_floor": {}, "floors": {}, "floors_per_parent": {},
             "gold_states_beyond_phi_clip": {str(k): sum(any(abs(v) > PHI_CLIP for v in json.loads(c["golds"][k]).values())
                                                         for c in cc) for k in KS}}
        for k in KS:
            rk = [x for x in rows if x["k"] == k]
            r["copy_chain_exact_floor"][str(k)] = float(np.mean([x["gold"] == env.dumps(x["s0"]) for x in rk]))
            fp, fl = C2.floor_regrets(rk, C.LEDGER, k)
            need(fp == pids, "floor parent order")
            r["floors_per_parent"][str(k)] = {n: v for n, v in fl.items()}
            r["floors"][str(k)] = {n: float(np.nanmean(v)) for n, v in fl.items()}
        for arm, out in outs.items():
            r["exact"][arm], r["regret"][arm], r["regret_per_parent"][arm] = {}, {}, {}
            for k in KS:
                rk = [x for x in rows if x["k"] == k]
                ex = C2.parent_exact(rk, C.LEDGER, k, out, pids)
                p2, nreg = C2.gen_regret(rk, C.LEDGER, k, out)
                need(p2 == pids, "regret parent order")
                r["exact"][arm][str(k)] = {"mean": float(np.mean(ex)), "per_parent": ex}
                r["regret"][arm][str(k)] = float(np.nanmean(nreg))
                r["regret_per_parent"][arm][str(k)] = nreg
            cnt = defaultdict(lambda: [0, 0])
            for c in cc:
                st = out[c["cand_id"]]["states"]
                for t, ev in enumerate(c["events"]):
                    if st[t] != c["golds"][t]:
                        continue
                    key = c["meta"]["step_classes"][t]
                    cnt[key][0] += 1
                    cnt[key][1] += int(st[t + 1] == c["golds"][t + 1])
            r["step_accuracy"][arm] = {k2: {"n": n, "acc": (h / n) if n else float("nan")} for k2, (n, h) in sorted(cnt.items())}
        res[cs] = r
    return res


def chain_contrasts(ch: dict) -> dict:
    """Descriptive: per chain stratum and seed, ON - OFF exact@4 and regret@4 (paired parent bootstrap, PREREG-2 form;
    10 parents: indicative only)."""
    out = {}
    for cs, r in ch.items():
        for s in SEEDS + ("supa",):
            if s not in r["exact"] or "off" not in r["exact"]:
                continue
            out[f"{cs}|{s}-off|k4|exact"] = C2.boot_mean(r["exact"][s]["4"]["per_parent"] - r["exact"]["off"]["4"]["per_parent"])
            out[f"{cs}|{s}-off|k4|regret"] = C2.boot_mean(r["regret_per_parent"][s]["4"] - r["regret_per_parent"]["off"]["4"])
    return out


# ---------------------------------------------------------------- one-step summary
def score_all(rows: list, texts: dict) -> tuple[dict, dict]:
    corr, cls = {}, {}
    for arm, tx in texts.items():
        sc = [C.score_onestep(r, t) for r, t in zip(rows, tx)]
        corr[arm] = np.array([s["correct"] for s in sc], bool)
        cls[arm] = [s["cls"] for s in sc]
    return corr, cls


def _cell_rows(rows: list, idx: dict, s: str, part: str) -> np.ndarray:
    return np.array([i for i in idx[s] if part == "all" or C.kind_class(rows[i]["event"]["kind"]) == part], int)


def summarise(rows: list, corr: dict, cls: dict) -> dict:
    idx = {s: np.array([i for i, r in enumerate(rows) if r["stratum"] == s], int) for s in C.STRATA}
    twin_of = {}
    tw = {r["base"]: i for i, r in enumerate(rows) if r["stratum"] == "T_posted"}
    for i, r in enumerate(rows):
        twin_of[i] = tw[r["base"]]
    arms = [a for a in ARMS if a in corr]
    S = {"acc": {}, "cells": {}, "cls": {}, "mcnemar": {"status": OrderedDict(), "gen": OrderedDict(), "descriptive": OrderedDict()},
         "in_vs_out": {}, "retained_gain": {}, "gain_in_out": {}, "pairs": {}, "sensitivity": {}, "breakdown": {},
         "novelty_copy": {}, "trigger_errors": {}, "twin": {}}
    cell_defs = ([(("T_posted",), p) for p in ("all", "at", "sc")] + [c for cells in C.GEN_GROUPS.values() for c in cells]
                 + list(C.DESCRIPTIVE_CELLS))
    cidx = OrderedDict()
    for st, p in cell_defs:
        cidx.setdefault(C.cell_key(st, p), np.concatenate([_cell_rows(rows, idx, s, p) for s in st]))
    desc_keys = [C.cell_key(st, p) for st, p in C.DESCRIPTIVE_CELLS]
    for arm in arms:
        S["acc"][arm] = {s: cell(int(corr[arm][ix].sum()), len(ix)) for s, ix in idx.items()}
        S["cells"][arm] = {k: cell(int(corr[arm][ix].sum()), len(ix)) for k, ix in cidx.items()}
        S["cls"][arm] = {s: dict(Counter(cls[arm][i] for i in ix)) for s, ix in idx.items()}
        tc = S["cls"][arm]["T_posted"]
        n_t = len(idx["T_posted"])
        S["twin"][arm] = {"acc": S["acc"][arm]["T_posted"]["acc"], "copied": tc.get("copied", 0) / n_t,
                          "wrong_value": tc.get("wrong_value", 0) / n_t, "unparsed": tc.get("unparsed", 0) / n_t}
        gen_ix = np.concatenate([idx[s] for s in C.GEN_STRATA])
        S["novelty_copy"][arm] = sum(cls[arm][i] == "copied" for i in gen_ix) / len(gen_ix)
        trig_ix = np.concatenate([idx[s] for s in C.TRIGGER_STRATA])
        S["trigger_errors"][arm] = {"rows": int(len(trig_ix)), "wrong": int(sum(not corr[arm][i] for i in trig_ix)),
                                    "applied": int(sum(cls[arm][i] == "applied_as_posted" for i in trig_ix))}
    # registered families: status (24 unseen-status cells) and gen (18 cells); Holm within each
    for s in C.UNSEEN_STATUS_STRATA:
        for sd in SEEDS:
            if sd in corr and "off" in corr:
                S["mcnemar"]["status"][f"{s}|{sd}"] = mcnemar_exact(corr[sd][idx[s]], corr["off"][idx[s]])
    for _, cells in C.GEN_GROUPS.items():
        for st, p in cells:
            k = C.cell_key(st, p)
            for sd in SEEDS:
                if sd in corr and "off" in corr:
                    S["mcnemar"]["gen"][f"{k}|{sd}"] = mcnemar_exact(corr[sd][cidx[k]], corr["off"][cidx[k]])
    for f in ("status", "gen"):
        if S["mcnemar"][f]:
            for k, v in C2.holm({k: m["p_two_sided"] for k, m in S["mcnemar"][f].items()}, ALPHA).items():
                S["mcnemar"][f][k].update(v)
    for s in ("T_posted", "C_void"):
        for sd in SEEDS + ("supa",):
            if sd in corr and "off" in corr:
                S["mcnemar"]["descriptive"][f"{s}|{sd}"] = mcnemar_exact(corr[sd][idx[s]], corr["off"][idx[s]])
    if "supa" in corr and "off" in corr:
        for s in C.STRATA:
            S["mcnemar"]["descriptive"][f"{s}|supa"] = mcnemar_exact(corr["supa"][idx[s]], corr["off"][idx[s]])
    # in-vs-out within each arm (paired by base): accuracy on a single-stratum GEN cell vs the same bases' posted twins
    for arm in arms:
        S["in_vs_out"][arm] = {}
        for k in desc_keys:
            ix = cidx[k]
            tix = np.array([twin_of[i] for i in ix], int)
            m = mcnemar_exact(corr[arm][ix], corr[arm][tix])
            S["in_vs_out"][arm][k] = {"acc_out": float(corr[arm][ix].mean()), "acc_in": float(corr[arm][tix].mean()),
                                      "out_only": m["on_only"], "in_only": m["off_only"], "p_two_sided": m["p_two_sided"]}
    # retained gain per group and seed: per base, (ON - OFF) on the group's rows minus (ON - OFF) on the base's twin
    if "off" in corr:
        for g, cells in C.GEN_GROUPS.items():
            S["retained_gain"][g], S["gain_in_out"][g] = {}, {}
            for sd in SEEDS + ("supa",):
                if sd not in corr:
                    continue
                per_base = defaultdict(list)
                g_out, g_in = [], []
                for st, p in cells:
                    for i in cidx[C.cell_key(st, p)]:
                        t = twin_of[i]
                        out_g = float(corr[sd][i]) - float(corr["off"][i])
                        in_g = float(corr[sd][t]) - float(corr["off"][t])
                        per_base[rows[i]["base"]].append(out_g - in_g)
                        g_out.append(out_g)
                        g_in.append(in_g)
                d = np.array([np.mean(v) for _, v in sorted(per_base.items())])
                S["retained_gain"][g][sd] = C2.boot_mean(d)
                S["gain_in_out"][g][sd] = {"gain_out": float(np.mean(g_out)), "gain_in_same_bases": float(np.mean(g_in))}
    # pairs and status sensitivity (status bases): unseen row correct and its twin correct / changed (not copied)
    for arm in arms:
        S["pairs"][arm], S["sensitivity"][arm] = {}, {}
        for s in C.UNSEEN_STATUS_STRATA + ("C_void",):
            both = [bool(corr[arm][i] and corr[arm][twin_of[i]]) for i in idx[s]]
            sens = [bool(corr[arm][i] and cls[arm][twin_of[i]] in ("correct", "wrong_value")) for i in idx[s]]
            S["pairs"][arm][s] = cell(int(sum(both)), len(both))
            S["sensitivity"][arm][s] = cell(int(sum(sens)), len(sens))
    # breakdowns (descriptive): registered sub-variants and kinds
    for arm in arms:
        bd = {}
        for s in C.STRATA:
            g = defaultdict(lambda: [0, 0])
            for i in idx[s]:
                k = C.sub_of(rows[i])
                g[k][0] += 1
                g[k][1] += int(corr[arm][i])
            if len(g) > 1:
                bd[s] = {k: cell(h, n) for k, (n, h) in sorted(g.items())}
        g = defaultdict(lambda: [0, 0])
        for i, r in enumerate(rows):
            g[f"{r['stratum']}|{r['event']['kind']}"][0] += 1
            g[f"{r['stratum']}|{r['event']['kind']}"][1] += int(corr[arm][i])
        bd["by_kind"] = {k: cell(h, n) for k, (n, h) in sorted(g.items())}
        S["breakdown"][arm] = bd
    return S


# ---------------------------------------------------------------- verdicts (pure functions of the summary)
def status_class(c: dict) -> str:
    if meets(c):
        return "follows the rule"
    if c["acc"] <= TRIGGER_ACC + EPS:
        return "trigger-like"
    return "intermediate"


def v1_status(S: dict) -> dict:
    A = S["acc"]
    per_seed = {}
    for sd in SEEDS:
        u = {s: A[sd][s] for s in C.UNSEEN_STATUS_STRATA}
        tw, te = S["twin"][sd], S["trigger_errors"][sd]
        per_seed[sd] = {
            "unseen_rule_ok": all(meets(c) for c in u.values()),
            "failing_unseen": [s for s, c in u.items() if not meets(c)],
            "not_applied_ok": all(meets(A[sd][s]) for s in C.NOT_APPLIED_STRATA),
            "neutral_ok": meets(A[sd]["U_neutral"]), "voidcase_ok": meets(A[sd]["N_voidcase"]),
            "postedlike_ok": meets(A[sd]["N_postedlike"]),
            "twin_acc": tw["acc"], "twin_copied": tw["copied"], "twin_wrong_value": tw["wrong_value"], "twin_unparsed": tw["unparsed"],
            "twin_acc_ok": tw["acc"] >= TWIN_ACC - EPS, "twin_copy_ok": tw["copied"] <= TWIN_COPY_MAX + EPS,
            "trigger_low": all(A[sd][s]["acc"] <= TRIGGER_ACC + EPS for s in C.TRIGGER_STRATA),
            "cvoid_acc": A[sd]["C_void"]["acc"], "cvoid_ok": A[sd]["C_void"]["acc"] >= TRIGGER_CVOID_MIN - EPS,
            "trigger_wrong": te["wrong"], "trigger_applied": te["applied"],
            "trigger_applied_majority": 2 * te["applied"] > te["wrong"],
            "voidcase_class": status_class(A[sd]["N_voidcase"])}
        per_seed[sd]["twin_ok"] = per_seed[sd]["twin_acc_ok"] and per_seed[sd]["twin_copy_ok"]
    P = per_seed.values()
    rule = all(v["unseen_rule_ok"] and v["twin_ok"] for v in P)
    trig_low = all(v["trigger_low"] for v in P)
    trig = trig_low and all(v["cvoid_ok"] and v["trigger_applied_majority"] for v in P)
    verdict = "RULE" if rule else ("TRIGGER" if trig else "PARTIAL")
    mc = S["mcnemar"]["status"]
    sig = {k: bool(v.get("holm_significant")) and v["direction"] == "on>off" for k, v in mc.items()}
    all_sig = len(sig) == len(C.UNSEEN_STATUS_STRATA) * len(SEEDS) and all(sig.values())
    off_tw = S["twin"]["off"]
    base_unseen = all(meets(A["off"][s]) for s in C.UNSEEN_STATUS_STRATA)
    base_meets = base_unseen and off_tw["copied"] <= TWIN_COPY_MAX + EPS
    attribution = None
    if verdict == "RULE":
        attribution = ("strengthened" if all_sig else "base_already") if base_meets else ("learned" if all_sig else "mixed")
    patterns = []
    if verdict == "PARTIAL":
        if all(v["unseen_rule_ok"] for v in P):
            patterns.append("twins")
        if all(v["not_applied_ok"] for v in P) and not all(v["neutral_ok"] for v in P):
            patterns.append("semantic")
        if all(v["not_applied_ok"] and v["neutral_ok"] for v in P) and not all(v["postedlike_ok"] for v in P):
            patterns.append("posted_substring")
        if trig_low and not trig:
            patterns.append("trigger_not_interpretable")
        if not any(sig.values()):
            patterns.append("no_adapter_gain")
    subs = []
    for sd in SEEDS:
        for s in ("U_neutral", "N_voidcase", "N_postedlike"):
            if A[sd][s]["acc"] <= TRIGGER_ACC + EPS:
                continue
            for sub, c in S["breakdown"][sd].get(s, {}).items():
                if c["acc"] <= TRIGGER_ACC + EPS:
                    subs.append({"seed": sd, "stratum": s, "sub": sub, "acc": c["acc"], "n": c["n"]})
    novelty = [sd for sd in SEEDS if S["novelty_copy"][sd] > NOVELTY_COPY_MAX + EPS]
    exp = {"E1 C_void ON >= 0.95 in every seed": all(A[sd]["C_void"]["acc"] >= E1_VOID_ON - EPS for sd in SEEDS),
           "E2 C_void OFF <= 0.30": A["off"]["C_void"]["acc"] <= E2_VOID_OFF + EPS}
    return {"verdict": verdict, "attribution": attribution, "patterns": patterns, "per_seed": per_seed,
            "adapter_gain_all_24_holm": all_sig, "adapter_gain_any_holm": any(sig.values()),
            "not_significant_cells": [k for k, v in sig.items() if not v],
            "base_meets_bar_on_every_unseen": base_unseen, "base_twins_copied": off_tw["copied"], "base_meets": base_meets,
            "trigger_low_every_seed": trig_low, "sub_variants_trigger_like": subs, "novelty_fallback_seeds": novelty,
            "expectations": exp}


def gen_cell_class(c: dict, m: dict) -> str:
    sig = bool(m.get("holm_significant"))
    if sig and m["direction"] == "off>on":
        return "HARMED"
    high = meets(c, GEN_ACC, GEN_LB)
    gain = sig and m["direction"] == "on>off"
    if high:
        return "HIGH_GAIN" if gain else "HIGH_NOGAIN"
    return "GAIN_BELOW_FLOOR" if gain else "LOW_NOGAIN"


def v_groups(S: dict) -> dict:
    out = OrderedDict()
    for g, cells in C.GEN_GROUPS.items():
        keys = [C.cell_key(st, p) for st, p in cells]
        cl = OrderedDict((f"{k}|{sd}", gen_cell_class(S["cells"][sd][k], S["mcnemar"]["gen"][f"{k}|{sd}"])) for k in keys for sd in SEEDS)
        v = list(cl.values())
        if "HARMED" in v:
            verdict = "HARMED"
        elif all(c == "HIGH_GAIN" for c in v):
            verdict = "GENERALISES"
        elif all(c.startswith("HIGH_") for c in v):
            verdict = "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN"
        elif not any(c.startswith("HIGH_") for c in v):
            verdict = "DOES NOT GENERALISE" + (", with a partial gain over OFF" if "GAIN_BELOW_FLOOR" in v else "")
        else:
            verdict = "MIXED"
        out[g] = {"verdict": verdict, "cells": cl,
                  "off_below_floor_everywhere": all(not meets(S["cells"]["off"][k]) for k in keys),
                  "off_meets_where_no_seed_does": [k for k in keys if meets(S["cells"]["off"][k])
                                                   and not any(meets(S["cells"][sd][k]) for sd in SEEDS)],
                  "harmed_cells": [k for k, c in cl.items() if c == "HARMED"],
                  "below_floor_cells": [k for k, c in cl.items() if not c.startswith("HIGH_") and c != "HARMED"]}
    return out


# ---------------------------------------------------------------- wording (rendered mechanically)
def _a(c: dict) -> str:
    return f"{c['acc']:.2f} [{c['wilson95'][0]:.2f}, {c['wilson95'][1]:.2f}]"


def _rng(vals) -> str:
    vals = list(vals)
    return f"{min(vals):.2f}-{max(vals):.2f}"


def _seeds_list(seeds) -> str:
    return ", ".join(seeds) if seeds else "none"


def wording(S: dict, V1: dict, VG: dict) -> dict:
    A = S["acc"]
    U = C.UNSEEN_STATUS_STRATA
    on_vals = [A[sd][s]["acc"] for sd in SEEDS for s in U]
    off_vals = [A["off"][s]["acc"] for s in U]
    tw_acc = ", ".join(f"{sd} {V1['per_seed'][sd]['twin_acc']:.2f}" for sd in SEEDS)
    tw_cp = ", ".join(f"{sd} {V1['per_seed'][sd]['twin_copied']:.2f}" for sd in SEEDS)
    v = V1["verdict"]
    vc = {sd: V1["per_seed"][sd]["voidcase_class"] for sd in SEEDS}

    def twin_cause(sd):
        p = V1["per_seed"][sd]
        if p["twin_copied"] >= p["twin_wrong_value"]:
            return (f"{sd}: posted events left unchanged (twin accuracy {p['twin_acc']:.2f}, copied {p['twin_copied']:.2f}, "
                    f"wrong value {p['twin_wrong_value']:.2f})")
        return (f"{sd}: arithmetic errors on posted events (twin accuracy {p['twin_acc']:.2f}, wrong value "
                f"{p['twin_wrong_value']:.2f}, copied {p['twin_copied']:.2f})")
    twin_fail_seeds = [sd for sd in SEEDS if not V1["per_seed"][sd]["twin_ok"]]
    if v == "RULE":
        head = ATTR_HEAD[V1["attribution"]]
        body = (f"In all three T2 seeds, Titan-Sparse with the adapter left unseen-status events unchanged on "
                f"{_rng(on_vals)} of rows in every seed and stratum (each >= 0.80 with Wilson lower bound >= 0.70): "
                f"{WORDS_TEXT}, as the frame's rule states. It applied the same events when posted (twin accuracy "
                f"{tw_acc}; twins copied {tw_cp}). ")
        nsig = V1["not_significant_cells"]
        listed = ", ".join(f"{k} (OFF {A['off'][k.split('|')[0]]['acc']:.2f}, ON {A[k.split('|')[1]][k.split('|')[0]]['acc']:.2f})"
                           for k in nsig[:6]) + (", ..." if len(nsig) > 6 else "")
        body += {"learned": f"Without the adapter the same rows scored {_rng(off_vals)}; ON > OFF in all 24 stratum x seed "
                            "McNemar tests (Holm).",
                 "strengthened": f"Without the adapter Titan-Sparse already met the bar on every unseen status "
                                 f"({_rng(off_vals)}; its posted twins copied {V1['base_twins_copied']:.2f}), and ON > OFF was "
                                 "still Holm-significant in all 24 tests: the rule is present in the base and strengthened "
                                 "by the adapter.",
                 "base_already": f"Without the adapter Titan-Sparse already met the bar on every unseen status "
                                 f"({_rng(off_vals)}; its posted twins copied {V1['base_twins_copied']:.2f}): the adapter "
                                 "preserved a rule the base already follows; P3 does not show that the adapter learned it.",
                 "mixed": f"Without the adapter the same rows scored {_rng(off_vals)}; ON > OFF is not Holm-significant in "
                          f"{len(nsig)} of 24 tests ({listed}), and the base does not meet the bar on every unseen status "
                          "with uncopied twins."}[V1["attribution"]]
    elif v == "TRIGGER":
        head = "TRIGGER"
        tv = [A[sd][s]["acc"] for sd in SEEDS for s in C.TRIGGER_STRATA]
        app = sum(V1["per_seed"][sd]["trigger_applied"] for sd in SEEDS)
        wr = sum(V1["per_seed"][sd]["trigger_wrong"] for sd in SEEDS)
        if all(c == "follows the rule" for c in vc.values()):
            cue = "'void' and its near-miss spellings (Void, VOID, voided)"
        elif all(c == "trigger-like" for c in vc.values()):
            cue = "the exact token 'void' (its near-miss spellings Void, VOID, voided were applied)"
        else:
            cue = "'void' (its near-miss spellings Void, VOID, voided handled inconsistently)"
        body = (f"In all three T2 seeds the adapted model's no-op did not extend beyond 'void': accuracy on the seven "
                f"unseen-status strata that contain no form of 'void' (the five not-applied words, the neutral and nonce "
                f"words, the look-alikes of 'posted') was {_rng(tv)} (each <= 0.30), against C_void "
                + ", ".join(f"{sd} {A[sd]['C_void']['acc']:.2f}" for sd in SEEDS) + f" (each >= 0.80); {app} of the {wr} "
                f"errors on those rows applied the event as if posted. Near-miss spellings of 'void' left unchanged: "
                + ", ".join(f"{sd} {A[sd]['N_voidcase']['acc']:.2f} ("
                            + {"follows the rule": "unchanged", "trigger-like": "applied", "intermediate": "mixed"}[vc[sd]] + ")"
                            for sd in SEEDS)
                + f". The adapted model treats {cue}, not 'any status other than posted', as the no-op cue.")
    else:
        head = "PARTIAL"
        fails = "; ".join(f"{sd}: " + (", ".join(V1['per_seed'][sd]['failing_unseen']) or "none")
                          + ("" if V1["per_seed"][sd]["twin_ok"] else " (twin guard failed)") for sd in SEEDS)
        body = (f"Neither RULE nor TRIGGER: unseen-status accuracy ranged {_rng(on_vals)} across seeds (OFF "
                f"{_rng(off_vals)}); cells below the bar (accuracy 0.80, Wilson LB 0.70) per seed: {fails}.")
        pats = V1["patterns"]
        if "twins" in pats:
            body += (" Every unseen-status cell met the bar; RULE was withheld only by the posted-twin guard (accuracy "
                     ">= 0.80 and copied <= 0.10) in " + _seeds_list(twin_fail_seeds) + ": "
                     + "; ".join(twin_cause(sd) for sd in twin_fail_seeds) + ".")
        if "semantic" in pats:
            bad = [sd for sd in SEEDS if not V1["per_seed"][sd]["neutral_ok"]]
            body += (" In every seed the five words that mean 'not applied' met the bar, but the neutral and nonce words "
                     "(approved, active, ok, dax, wug) did not in " + ", ".join(f"{sd} ({A[sd]['U_neutral']['acc']:.2f})" for sd in bad)
                     + ": consistent with a semantic 'not applied' cue, not the stated rule.")
        if "posted_substring" in pats:
            bad = [sd for sd in SEEDS if not V1["per_seed"][sd]["postedlike_ok"]]
            body += (" Every unseen status word met the bar, but the look-alikes of 'posted' (unposted, not_posted, "
                     "'not posted') did not in " + ", ".join(f"{sd} ({A[sd]['N_postedlike']['acc']:.2f})" for sd in bad)
                     + ": consistent with a 'contains posted' cue.")
        if "trigger_not_interpretable" in pats:
            why = []
            cv = [sd for sd in SEEDS if not V1["per_seed"][sd]["cvoid_ok"]]
            am = [sd for sd in SEEDS if not V1["per_seed"][sd]["trigger_applied_majority"]]
            if cv:
                why.append("C_void was below 0.80 in " + ", ".join(f"{sd} ({A[sd]['C_void']['acc']:.2f})" for sd in cv))
            if am:
                why.append("fewer than half of the errors applied the event in " + ", ".join(
                    f"{sd} ({V1['per_seed'][sd]['trigger_applied']} of {V1['per_seed'][sd]['trigger_wrong']})" for sd in am))
            body += (" Every unseen status without a form of 'void' was at or below 0.30 in every seed, but TRIGGER was "
                     "withheld because " + " and ".join(why) + ": the low accuracy is not interpretable as a void trigger.")
        if "no_adapter_gain" in pats:
            body += " No unseen-status cell shows a Holm-significant adapter gain over OFF."
    quals = []
    if V1["sub_variants_trigger_like"]:
        quals.append("Qualifier: sub-variants at or below 0.30 inside a stratum above 0.30 (adapted seeds): " + "; ".join(
            f"{q['stratum']} '{q['sub']}' {q['seed']} {q['acc']:.2f} (n {q['n']})" for q in V1["sub_variants_trigger_like"][:8])
            + ("; ..." if len(V1["sub_variants_trigger_like"]) > 8 else "") + ".")
    if V1["novelty_fallback_seeds"]:
        quals.append("Qualifier (novelty fallback): the adapted model also copied novel posted inputs: copied rate over the "
                     "magnitude, naming and format rows " + ", ".join(f"{sd} {S['novelty_copy'][sd]:.2f}" for sd in SEEDS)
                     + f" (> {NOVELTY_COPY_MAX:.2f} in {', '.join(V1['novelty_fallback_seeds'])}); its no-op on unseen statuses "
                     "may partly be a fallback to copying on unfamiliar input rather than the stated rule.")
    for k, ok in V1["expectations"].items():
        if not ok:
            quals.append(f"Instrument expectation not met: {k} (" + ", ".join(f"{a} {A[a]['C_void']['acc']:.2f}"
                                                                            for a in ("off",) + SEEDS) + ").")
    sens = lambda arm: [S["sensitivity"][arm][s]["acc"] for s in U]  # noqa: E731
    base = (f"Base semantics (descriptive): unseen-status no-op accuracy OFF {_rng(off_vals)}"
            + (f", Supa 1.7 {_rng(A['supa'][s]['acc'] for s in U)}" if "supa" in A else "")
            + f"; status-sensitive pairs (unseen row unchanged while its posted twin was changed) OFF {_rng(sens('off'))}"
            + (f", Supa 1.7 {_rng(sens('supa'))}" if "supa" in A else "")
            + f", adapted seeds {_rng(x for sd in SEEDS for x in sens(sd))}. Raw no-op accuracy cannot separate status "
            "semantics from copy bias; the paired rate can.")
    quals.append(base)
    groups = []
    for g, r in VG.items():
        keys = [C.cell_key(st, p) for st, p in C.GEN_GROUPS[g]]
        tp = f"T_posted|{GROUP_TWIN_PART[g]}"
        nums = "; ".join(f"{k}: OFF {S['cells']['off'][k]['acc']:.2f}, " + ", ".join(f"{sd} {S['cells'][sd][k]['acc']:.2f}" for sd in SEEDS)
                         for k in keys)
        twin = f"OFF {S['cells']['off'][tp]['acc']:.2f}, " + ", ".join(f"{sd} {S['cells'][sd][tp]['acc']:.2f}" for sd in SEEDS)
        rg = "; ".join(f"{sd} {S['retained_gain'][g][sd]['point']:+.2f} [{S['retained_gain'][g][sd]['ci95'][0]:+.2f}, "
                       f"{S['retained_gain'][g][sd]['ci95'][1]:+.2f}]" for sd in SEEDS)
        line = (f"{C.GROUP_LABEL[g]}: {r['verdict']} ({nums}; the same bases in distribution, {tp}: {twin}; retained gain "
                f"[(ON - OFF) here minus (ON - OFF) on the same bases' twins, base bootstrap]: {rg}).")
        if r["verdict"].startswith("DOES NOT GENERALISE"):
            if r["off_below_floor_everywhere"]:
                line += (" OFF is also below the floor on every cell: P3 cannot separate an in-range adapter shortcut from "
                         "a capacity limit shared with the base.")
            elif r["off_meets_where_no_seed_does"]:
                line += (" OFF meets the floor on " + ", ".join(r["off_meets_where_no_seed_does"]) + " where no adapted "
                         "seed does: the adapter's in-range behaviour does not carry over there.")
        elif r["verdict"] == "HARMED":
            line += (" The adapter makes " + ", ".join(r["harmed_cells"]) + " significantly worse than the base: the "
                     "signature of an in-range shortcut (BWM-001 F-01/F-06).")
        elif r["verdict"] == "MIXED":
            line += " Cells below the floor: " + ", ".join(r["below_floor_cells"]) + "."
        groups.append(line)
    if v == "RULE":
        if V1["novelty_fallback_seeds"]:
            impl = ("For PREREG-2: the adapted model meets the RULE bar, but it also copies novel posted inputs (novelty "
                    "qualifier), so P3 cannot exclude a novelty fallback; PREREG-2's limit 'fresh at payload level, not "
                    "out-of-distribution' stays as registered.")
        else:
            att = {"learned": "", "strengthened": " (present in the base already, strengthened by the adapter)",
                   "base_already": " (but the base already did)", "mixed": " (attribution mixed)"}[V1["attribution"]]
            impl = ("For PREREG-2: within the tested status vocabulary (16 non-posted strings in eight strata) the "
                    f"adapter's status handling follows the stated rule, not the token 'void'{att}; PREREG-2's limit "
                    "'fresh at payload level, not out-of-distribution' can be narrowed for the status rule only, as far as "
                    "the magnitude / naming / format verdicts allow.")
    elif v == "TRIGGER":
        impl = ("For PREREG-2: the void gain is a token cue, not the stated rule; PREREG-2's limit 'fresh at payload level, "
                "not out-of-distribution' binds, and rule data must be surface-diversified before reuse.")
    elif "twins" in V1["patterns"]:
        impl = ("For PREREG-2: status handling met the rule on every unseen status, but RULE was withheld by the "
                "posted-twin guard in " + _seeds_list(twin_fail_seeds) + "; PREREG-2's limit 'fresh at payload level, "
                "not out-of-distribution' stays as registered.")
    elif "semantic" in V1["patterns"]:
        impl = ("For PREREG-2: the adapter's no-op generalises to status words that mean 'not applied' but not to neutral "
                "or nonce words, i.e. along word meaning rather than the stated rule; PREREG-2's limit 'fresh at payload "
                "level, not out-of-distribution' stays as registered, and rule data would need non-posted statuses "
                "without a not-applied meaning.")
    elif "posted_substring" in V1["patterns"]:
        impl = ("For PREREG-2: the adapter's no-op generalises to unseen status words but look-alikes of 'posted' are "
                "applied, consistent with a 'contains posted' cue rather than the stated rule; PREREG-2's limit 'fresh "
                "at payload level, not out-of-distribution' stays as registered, and rule data would need look-alike "
                "statuses.")
    elif "no_adapter_gain" in V1["patterns"]:
        impl = ("For PREREG-2: no unseen-status cell shows an adapter gain over OFF; whatever no-op rate there is on "
                "unseen statuses is the base's, and PREREG-2's limit 'fresh at payload level, not out-of-distribution' "
                "stays as registered.")
    else:
        impl = ("For PREREG-2: the adapter's status handling generalises only partly beyond 'void' (per-stratum table); "
                "PREREG-2's limit 'fresh at payload level, not out-of-distribution' stays as registered.")
    return {"headline": f"STATUS: {head}", "status_sentence": body, "qualifiers": quals, "groups": groups,
            "implication": impl, "limits": list(LIMITS)}


# ---------------------------------------------------------------- synthetic per-row classes (self-tests, fixtures)
def fix_part(row: dict) -> str:
    """The partition the synthetic designs control exactly: the status sub-variant for status strata with a status
    cycle, else the kind class (at / sc)."""
    return row["event"]["status"] if isinstance(C.STRATA[row["stratum"]][2], tuple) else C.kind_class(row["event"]["kind"])


def _alias(s: str) -> list:
    if s in C.NOT_APPLIED_STRATA:
        return ["NA", "U"]
    if s == "U_neutral":
        return ["U"]
    if s.startswith("N_"):
        return ["N"]
    if s in C.GEN_STRATA:
        return ["gen"]
    return {"T_posted": ["T"], "C_void": ["V"]}[s]


def _resolve(spec: dict, s: str, part: str, default=None):
    for k in [f"{s}|{part}", s] + _alias(s) + ["_"]:
        if k in spec:
            return spec[k]
    return default


def _largest_remainder(targets: dict, total: int) -> dict:
    fl = {k: math.floor(v + 1e-9) for k, v in targets.items()}
    rest = total - sum(fl.values())
    for k in sorted(targets, key=lambda k: (-(targets[k] - fl[k]), str(k)))[:max(0, rest)]:
        fl[k] += 1
    return fl


DEFAULT_ERR = {"noop": {"applied": 0.7, "other": 0.2, "unparsed": 0.1}, "apply": {"copy": 0.7, "wrong": 0.2, "unparsed": 0.1}}
MODE_CLS = {"applied": "applied_as_posted", "other": "other_change", "unparsed": "unparsed", "copy": "copied", "wrong": "wrong_value"}


def synth_modes(rows: list, spec: dict, rng: random.Random) -> list:
    """Per row: 'correct' or an error mode, with EXACT counts: per stratum, round(sum of part targets) correct rows,
    split over the parts by largest remainder; errors apportioned the same way over the modes of spec['err']."""
    err_spec = spec.get("err", {})
    modes = [None] * len(rows)
    by_s = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_s[r["stratum"]][fix_part(r)].append(i)
    for s, parts in by_s.items():
        tg = {p: float(_resolve(spec, s, p)) * len(ix) for p, ix in parts.items()}
        k_part = _largest_remainder(tg, int(round(sum(tg.values()) + 1e-9)))
        for p, ix in sorted(parts.items()):
            ix = list(ix)
            rng.shuffle(ix)
            good, bad = ix[:k_part[p]], ix[k_part[p]:]
            for i in good:
                modes[i] = "correct"
            em = _resolve(err_spec, s, p, None) or DEFAULT_ERR[C.STRATA[s][1]]
            cnt = _largest_remainder({m: f * len(bad) for m, f in em.items()}, len(bad))
            j = 0
            for m in sorted(cnt):
                for i in bad[j:j + cnt[m]]:
                    modes[i] = m
                j += cnt[m]
    return modes


def synth_classes(rows: list, specs: dict, seed: int) -> tuple[dict, dict]:
    """(corr, cls) per arm straight from the synthetic modes (no texts); 'copy' is scored from the before-state."""
    rng = random.Random(seed)
    corr, cls = {}, {}
    for arm in ("off", "s0", "s1", "s2", "supa"):
        md = synth_modes(rows, specs[arm], rng)
        cls[arm] = ["correct" if m == "correct" else MODE_CLS[m] for m in md]
        corr[arm] = np.array([c == "correct" for c in cls[arm]], bool)
    sc = [C.score_onestep(r, env.dumps(r["before"])) for r in rows]
    cls["copy"], corr["copy"] = [s["cls"] for s in sc], np.array([s["correct"] for s in sc], bool)
    return corr, cls


DEF_ON = {"U": 0.97, "N": 0.95, "V": 1.0, "T": 0.92, "gen": 0.92}
DEF_OFF = {"U": 0.15, "N": 0.10, "V": 0.08, "T": 0.50, "gen": 0.40}
DEF_SUPA = {"U": 0.20, "N": 0.20, "V": 0.20, "T": 0.20, "gen": 0.10}
ERR_T_WRONG = {"T": {"wrong": 0.95, "copy": 0.05}}


def mk_specs(on=None, off=None, supa=None, seeds=None) -> dict:
    """Arm specs: DEF_* updated by the overrides; seeds: {seed: overrides on top of the ON spec}."""
    onx = {**DEF_ON, **(on or {})}
    sp = {"off": {**DEF_OFF, **(off or {})}, "supa": {**DEF_SUPA, **(supa or {})}}
    for sd in SEEDS:
        sp[sd] = {**onx, **((seeds or {}).get(sd) or {})}
    return sp


def wording_cases() -> OrderedDict:
    Q = 0.80
    return OrderedDict([
        ("RULE learned", (mk_specs(), {"headline": "STATUS: RULE (learned by the adapter)",
                                       "needles": ["can be narrowed for the status rule only", "status-sensitive pairs"]})),
        ("RULE strengthened", (mk_specs(on={"U": 1.0, "N": 1.0}, off={"U": 0.85, "N": 0.85, "err": ERR_T_WRONG}),
                               {"headline": "STATUS: RULE (present in the base at the bar, strengthened by the adapter)"})),
        ("RULE base already", (mk_specs(on={"U": 0.98, "N": 0.98}, off={"U": 0.95, "N": 0.95, "err": ERR_T_WRONG}),
                               {"headline": "STATUS: RULE (already present without the adapter)",
                                "needles": ["(but the base already did)"]})),
        ("RULE mixed: copy-biased base", (mk_specs(on={"U": 0.98, "N": 0.98}, off={"U": 0.95, "N": 0.95}),
                                          {"headline": "STATUS: RULE (attribution mixed)"})),
        ("RULE mixed: weak gain in one seed", (mk_specs(off={"U": 0.78, "N": 0.78}, seeds={"s2": {"U": 0.82, "N": 0.82}}),
                                               {"headline": "STATUS: RULE (attribution mixed)", "needles": ["not Holm-significant in"]})),
        ("TRIGGER exact token", (mk_specs(on={"U": 0.10, "N": 0.10}), {"headline": "STATUS: TRIGGER",
                                                                       "needles": ["the exact token 'void'", "surface-diversified"]})),
        ("TRIGGER with void-like spellings", (mk_specs(on={"U": 0.10, "N_postedlike": 0.10, "N_voidcase": 0.95}),
                                              {"headline": "STATUS: TRIGGER", "needles": ["'void' and its near-miss spellings"]})),
        ("PARTIAL seeds disagree", (mk_specs(seeds={"s2": {"U": 0.10, "N": 0.10}}),
                                    {"headline": "STATUS: PARTIAL", "needles": ["generalises only partly"]})),
        ("PARTIAL intermediate", (mk_specs(on={"U": 0.60, "N": 0.60}), {"headline": "STATUS: PARTIAL"})),
        ("PARTIAL twins: copying", (mk_specs(on={"T": 0.70}), {"headline": "STATUS: PARTIAL", "patterns": ["twins"],
                                                               "needles": ["posted events left unchanged",
                                                                           "withheld by the posted-twin guard"]})),
        ("PARTIAL twins: arithmetic", (mk_specs(on={"T": 0.70, "err": ERR_T_WRONG}),
                                       {"headline": "STATUS: PARTIAL", "patterns": ["twins"],
                                        "needles": ["arithmetic errors on posted events"]})),
        ("PARTIAL twin copy guard", (mk_specs(on={"T": 0.85, "err": {"T": {"copy": 1.0}}}),
                                     {"headline": "STATUS: PARTIAL", "patterns": ["twins"], "needles": ["copied 0.15"]})),
        ("PARTIAL semantic", (mk_specs(on={"U_neutral": 0.15}), {"headline": "STATUS: PARTIAL", "patterns": ["semantic"],
                                                                 "needles": ["semantic 'not applied' cue",
                                                                             "along word meaning rather than the stated rule"]})),
        ("PARTIAL posted substring", (mk_specs(on={"N_postedlike": 0.10}),
                                      {"headline": "STATUS: PARTIAL", "patterns": ["posted_substring"],
                                       "needles": ["'contains posted' cue", "rule data would need look-alike statuses"]})),
        ("PARTIAL near-miss only", (mk_specs(on={"N_voidcase": 0.10}), {"headline": "STATUS: PARTIAL", "patterns": []})),
        ("PARTIAL trigger not interpretable: C_void", (mk_specs(on={"U": 0.10, "N": 0.10, "V": 0.50}),
                                                       {"headline": "STATUS: PARTIAL", "patterns": ["trigger_not_interpretable"],
                                                        "needles": ["C_void was below 0.80"]})),
        ("PARTIAL trigger not interpretable: errors", (mk_specs(on={"U": 0.10, "N": 0.10, "err": {"U": {"unparsed": 1.0},
                                                                                                    "N": {"unparsed": 1.0}}}),
                                                       {"headline": "STATUS: PARTIAL", "patterns": ["trigger_not_interpretable"],
                                                        "needles": ["fewer than half of the errors"]})),
        ("PARTIAL no adapter gain", (mk_specs(on={"U": 0.50, "N": 0.50}, off={"U": 0.50, "N": 0.50}),
                                     {"headline": "STATUS: PARTIAL", "patterns": ["no_adapter_gain"],
                                      "needles": ["no unseen-status cell shows an adapter gain"]})),
        ("boundary 0.80 passes", (mk_specs(on={"U": Q, "N": Q}), {"headline": "STATUS: RULE (learned by the adapter)"})),
        ("boundary 0.79 fails", (mk_specs(on={"U": 0.79}), {"headline": "STATUS: PARTIAL"})),
        ("boundary 0.30 trigger", (mk_specs(on={"U": 0.30, "N": 0.30}, off={"U": 0.30, "N": 0.30}), {"headline": "STATUS: TRIGGER"})),
        ("boundary 0.31 partial", (mk_specs(on={"U": 0.31, "N": 0.31}, off={"U": 0.31, "N": 0.31}), {"headline": "STATUS: PARTIAL"})),
        ("boundary twins copied 0.10 passes", (mk_specs(on={"T": 0.90, "err": {"T": {"copy": 1.0}}}),
                                               {"headline": "STATUS: RULE (learned by the adapter)"})),
        ("boundary twins copied 0.11 fails", (mk_specs(on={"T": 0.89, "err": {"T": {"copy": 1.0}}}),
                                              {"headline": "STATUS: PARTIAL", "patterns": ["twins"]})),
        ("qualifier E1", (mk_specs(on={"V": 0.90}), {"headline": "STATUS: RULE (learned by the adapter)",
                                                     "needles": ["E1 C_void ON >= 0.95"]})),
        ("qualifier E2", (mk_specs(off={"V": 0.50}), {"needles": ["E2 C_void OFF <= 0.30"]})),
        ("qualifier sub-variant", (mk_specs(on={"U_neutral": 1.0, "U_neutral|dax": 0.20}),
                                   {"headline": "STATUS: RULE (learned by the adapter)", "needles": ["U_neutral 'dax'"]})),
        ("qualifier novelty fallback", (mk_specs(on={"gen": 0.70, "err": {"gen": {"copy": 1.0}}}),
                                        {"headline": "STATUS: RULE (learned by the adapter)",
                                         "needles": ["Qualifier (novelty fallback)", "cannot exclude a novelty fallback"]})),
        ("GENERALISES", (mk_specs(), {"groups": {g: "GENERALISES" for g in C.GEN_GROUPS}})),
        ("ABOVE FLOOR", (mk_specs(off={"gen": 0.90}), {"groups": {"magnitude_at": "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN",
                                                                  "naming": "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN"}})),
        ("DOES NOT GENERALISE: shared limit", (mk_specs(on={"gen": 0.35}, off={"gen": 0.35}),
                                               {"groups": {"format": "DOES NOT GENERALISE"},
                                                "needles": ["cannot separate an in-range adapter shortcut"]})),
        ("DOES NOT GENERALISE: partial gain", (mk_specs(on={"gen": 0.60}, off={"gen": 0.20}),
                                               {"groups": {"naming": "DOES NOT GENERALISE, with a partial gain over OFF"}})),
        ("DOES NOT GENERALISE: OFF meets", (mk_specs(on={"F_keyorder": 0.75}, off={"F_keyorder": 0.82}),
                                            {"groups": {"format": "DOES NOT GENERALISE"},
                                             "needles": ["OFF meets the floor on F_keyorder|all"]})),
        ("HARMED", (mk_specs(on={"M_bal": 0.10, "M_amt": 0.10}, off={"M_bal": 0.80, "M_amt": 0.80}),
                    {"groups": {"magnitude_at": "HARMED", "magnitude_scale": "HARMED", "naming": "GENERALISES"},
                     "needles": ["signature of an in-range shortcut"]})),
        ("MIXED", (mk_specs(on={"A_many": 0.60}, off={"gen": 0.20}), {"groups": {"naming": "MIXED"},
                                                                      "needles": ["Cells below the floor: A_many|all"]})),
        ("boundary floor n=100 0.80", (mk_specs(on={"F_keyorder": 0.80}), {"groups": {"format": "GENERALISES"}})),
        ("boundary floor n=100 0.79", (mk_specs(on={"F_keyorder": 0.79}),
                                       {"groups": {"format": "DOES NOT GENERALISE, with a partial gain over OFF"}})),
        ("boundary floor n=67 55", (mk_specs(on={"M_bal|at": 55 / 67, "M_amt|at": 55 / 67}),
                                    {"groups": {"magnitude_at": "GENERALISES"}})),
        ("boundary floor n=67 54", (mk_specs(on={"M_bal|at": 54 / 67, "M_amt|at": 54 / 67}),
                                    {"groups": {"magnitude_at": "DOES NOT GENERALISE, with a partial gain over OFF"}})),
        ("boundary floor n=66 54", (mk_specs(on={"M_bal|sc": 27 / 33, "M_amt|sc": 27 / 33}, off={"gen": 0.10}),
                                    {"groups": {"magnitude_scale": "GENERALISES"}})),
        ("boundary floor n=66 53", (mk_specs(on={"M_bal|sc": 27 / 33, "M_amt|sc": 26 / 33}, off={"gen": 0.10}),
                                    {"groups": {"magnitude_scale": "DOES NOT GENERALISE, with a partial gain over OFF"}})),
    ])


def wording_selftest(bank: Path | None = None) -> dict:
    """Every V1 / attribution / pattern / qualifier / group / implication branch on synthetic per-row classes, through
    the REAL summarise (exact McNemar, Holm, bootstrap) and verdict functions."""
    rows, _ = C.load_bank(bank or C.BANK)
    res, ok = OrderedDict(), True
    for i, (name, (specs, want)) in enumerate(wording_cases().items()):
        corr, cls = synth_classes(rows, specs, DRY_SEED + 101 + i)
        S = summarise(rows, corr, cls)
        V1, VG = v1_status(S), v_groups(S)
        W = wording(S, V1, VG)
        text = " ".join([W["headline"], W["status_sentence"], W["implication"]] + W["qualifiers"] + W["groups"])
        got = {"headline": W["headline"], "patterns": V1["patterns"], "groups": {g: VG[g]["verdict"] for g in want.get("groups", {})}}
        good = (("headline" not in want or got["headline"] == want["headline"])
                and ("patterns" not in want or set(want["patterns"]) <= set(V1["patterns"]) and (want["patterns"] or not
                     set(V1["patterns"]) & {"twins", "semantic", "posted_substring"}))
                and got["groups"] == want.get("groups", {})
                and all(nd in text for nd in want.get("needles", [])))
        res[name] = {"want": want, "got": got, "missing_needles": [nd for nd in want.get("needles", []) if nd not in text],
                     "ok": bool(good)}
        ok &= bool(good)
    return {"ok": bool(ok), "cases": res}


# ---------------------------------------------------------------- analysis and render
def analyse(I: Inputs) -> tuple[dict, dict]:
    corr, cls = score_all(I.rows, I.texts)
    S = summarise(I.rows, corr, cls)
    V1 = v1_status(S)
    VG = v_groups(S)
    W = wording(S, V1, VG)
    ch = chain_metrics(I.chains, I.chain_out)
    chc = chain_contrasts(ch)
    per_row = {f"correct_{a}": corr[a].astype(np.int8) for a in corr}
    per_row["row_index_stratum"] = np.array([list(C.STRATA).index(r["stratum"]) for r in I.rows], np.int16)
    ch_json = {cs: {k: v for k, v in r.items() if k not in ("regret_per_parent", "floors_per_parent")} for cs, r in ch.items()}
    for cs, r in ch_json.items():
        r["exact"] = {a: {k: v["mean"] for k, v in e.items()} for a, e in r["exact"].items()}
    res = {"prereg": "PREREG-3", "bank_sha256": sha256(I.bank_path), "run_dir": str(I.run),
           "thresholds": {"RULE_ACC": RULE_ACC, "RULE_LB": RULE_LB, "TWIN_ACC": TWIN_ACC, "TWIN_COPY_MAX": TWIN_COPY_MAX,
                          "TRIGGER_ACC": TRIGGER_ACC, "TRIGGER_CVOID_MIN": TRIGGER_CVOID_MIN, "GEN_ACC": GEN_ACC,
                          "GEN_LB": GEN_LB, "NOVELTY_COPY_MAX": NOVELTY_COPY_MAX, "ALPHA": ALPHA, "E1": E1_VOID_ON,
                          "E2": E2_VOID_OFF},
           "V1_status": V1, "V_groups": VG, "wording": W, "onestep": S, "chains": ch_json, "chain_contrasts": chc}
    return res, per_row


def render(res: dict, title: str = "PREREG-3 results") -> str:
    S, W, V1 = res["onestep"], res["wording"], res["V1_status"]
    arms = [a for a in ARMS if a in S["acc"]]
    L = [f"# {title}", "", "Rendered mechanically by `scripts/p3_analysis.py` from the results JSON. "
         f"Bank sha256 {res['bank_sha256'][:12]}...", "", f"## Verdict: {W['headline']}", "", f"> {W['status_sentence']}", ""]
    L += [f"> {q}" for q in W["qualifiers"]] + [""]
    L += [f"> {g}" for g in W["groups"]] + ["", f"> {W['implication']}", "", "Limits that always accompany the verdicts:", ""]
    L += [f"- {x}" for x in W["limits"]] + [""]
    L += ["## One-step accuracy by stratum (accuracy [Wilson 95%])", "",
          "| stratum | group | gold | " + " | ".join(ARM_LABEL[a] for a in arms) + " |",
          "|---|---|---|" + "---|" * len(arms)]
    for s, (g, rule, _, _) in C.STRATA.items():
        L.append(f"| {s} | {g} | {rule} | " + " | ".join(_a(S["acc"][a][s]) for a in arms) + " |")
    L += ["", "## V1 status: per seed", "",
          "| seed | unseen strata meet the bar | failing unseen strata | twin acc | twins copied | twin guard | "
          "trigger-set <= 0.30 | C_void | errors applied / wrong (trigger set) |", "|---|---|---|---|---|---|---|---|---|"]
    for sd, v in V1["per_seed"].items():
        L.append(f"| {sd} | {v['unseen_rule_ok']} | {', '.join(v['failing_unseen']) or '-'} | {v['twin_acc']:.2f} | "
                 f"{v['twin_copied']:.2f} | {v['twin_ok']} | {v['trigger_low']} | {v['cvoid_acc']:.2f} | "
                 f"{v['trigger_applied']} / {v['trigger_wrong']} |")
    L += ["", f"Verdict {V1['verdict']}; attribution {V1['attribution']}; PARTIAL patterns {V1['patterns'] or '-'}; adapter "
          f"gain Holm-significant in all 24 cells: {V1['adapter_gain_all_24_holm']} (in any: {V1['adapter_gain_any_holm']}); "
          f"OFF meets the bar on every unseen status: {V1['base_meets_bar_on_every_unseen']} (OFF twins copied "
          f"{V1['base_twins_copied']:.2f}). Expectations: "
          + "; ".join(f"{k}: {'holds' if v else 'does not hold'}" for k, v in V1["expectations"].items()), ""]
    L += ["## McNemar ON vs OFF (exact, two-sided; Holm within each family: status 24, gen 18)", "",
          "| family | cell | ON only | OFF only | p | Holm p | significant | direction |", "|---|---|---|---|---|---|---|---|"]
    for f in ("status", "gen"):
        for k, m in S["mcnemar"][f].items():
            L.append(f"| {f} | {k} | {m['on_only']} | {m['off_only']} | {m['p_two_sided']:.4g} | {m.get('p_holm', float('nan')):.4g} | "
                     f"{m.get('holm_significant')} | {m['direction']} |")
    L += ["", "## Magnitude / naming / format cells", "", "| group | verdict | cell classes |", "|---|---|---|"]
    for g, r in res["V_groups"].items():
        L.append(f"| {g} | {r['verdict']} | " + ", ".join(f"{k} {v}" for k, v in r["cells"].items()) + " |")
    ck = list(S["cells"]["off"])
    L += ["", "### Cell accuracies (kind classes: at = add + transfer, sc = scale)", "",
          "| cell | " + " | ".join(ARM_LABEL[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for k in ck:
        L.append(f"| {k} | " + " | ".join(_a(S["cells"][a][k]) for a in arms) + " |")
    L += ["", "### In distribution vs out of range within each arm (paired by base; exact McNemar, descriptive)", "",
          "| arm | cell | acc out | acc in (twins) | out only | in only | p |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        for k, m in S["in_vs_out"][a].items():
            L.append(f"| {ARM_LABEL[a]} | {k} | {m['acc_out']:.2f} | {m['acc_in']:.2f} | {m['out_only']} | {m['in_only']} | "
                     f"{m['p_two_sided']:.3g} |")
    L += ["", "### Retained adapter gain [(ON - OFF) out of range minus (ON - OFF) on the same bases' twins; base bootstrap]", "",
          "| group | arm | gain out | gain in (same bases) | retained | 95% CI |", "|---|---|---|---|---|---|"]
    for g, d in S["retained_gain"].items():
        for sd, b in d.items():
            gi = S["gain_in_out"][g][sd]
            L.append(f"| {g} | {sd} | {gi['gain_out']:+.3f} | {gi['gain_in_same_bases']:+.3f} | {b['point']:+.3f} | "
                     f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
    L += ["", "## Pairs and status sensitivity (descriptive)", "",
          "Both correct: the unseen row and its posted twin. Sensitive: the unseen row unchanged while the twin was changed "
          "(correct or wrong value, not copied).", "",
          "| stratum | " + " | ".join(f"{ARM_LABEL[a]} both / sensitive" for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for s in C.UNSEEN_STATUS_STRATA + ("C_void",):
        L.append(f"| {s} | " + " | ".join(f"{S['pairs'][a][s]['acc']:.2f} / {S['sensitivity'][a][s]['acc']:.2f}" for a in arms) + " |")
    L += ["", f"Copied rate over the magnitude / naming / format rows: " + ", ".join(f"{a} {S['novelty_copy'][a]:.2f}" for a in arms), ""]
    L += ["## Error classes (descriptive)", "", "| stratum | " + " | ".join(ARM_LABEL[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for s in C.STRATA:
        L.append(f"| {s} | " + " | ".join(", ".join(f"{k} {v}" for k, v in sorted(S["cls"][a][s].items()) if k != "correct") or "-"
                                         for a in arms) + " |")
    L += ["", "## Breakdowns (descriptive; registered sub-variants)", ""]
    for s in C.STRATA:
        if s in S["breakdown"]["off"]:
            L.append(f"- {s}: " + "; ".join(f"{a}: " + ", ".join(f"{k} {v['acc']:.2f}" for k, v in S["breakdown"][a][s].items())
                                            for a in arms))
    L += ["", "## Chains (descriptive)", "",
          "K = 4 candidates per parent (PREREG-2: 8), so normalised regret is not comparable to PREREG-2's; phi clips balances "
          f"at +-{PHI_CLIP}, so regret is partly blind to large-magnitude errors (exact@k is the primary chain measure for "
          "CH_X); 10 parents per stratum, so the bootstrap intervals are indicative only.", ""]
    for cs, r in res["chains"].items():
        L.append(f"### {cs} ({r['parents']} parents x {r['candidates'] // max(r['parents'], 1)} candidates): {C.CHAIN_DESC[cs]}")
        L += ["", f"Gold states beyond the phi clip (|balance| > {PHI_CLIP}): " +
              ", ".join(f"k{k} {v} of {r['candidates']}" for k, v in r["gold_states_beyond_phi_clip"].items()), ""]
        L += ["| arm | exact k1 | exact k2 | exact k4 | regret k1 | regret k2 | regret k4 |", "|---|---|---|---|---|---|---|"]
        for a in [x for x in ARMS if x in r["exact"]]:
            L.append(f"| {ARM_LABEL[a]} | " + " | ".join(f"{r['exact'][a][str(k)]:.3f}" for k in KS) + " | "
                     + " | ".join(f"{r['regret'][a][str(k)]:.3f}" for k in KS) + " |")
        L.append("")
        L.append("Floors (regret): " + "; ".join(f"k{k}: " + ", ".join(f"{n} {v:.3f}" for n, v in sorted(r["floors"][str(k)].items()))
                                                 for k in KS) + "; copy-chain exact floor " +
                 ", ".join(f"k{k} {r['copy_chain_exact_floor'][str(k)]:.3f}" for k in KS))
        L.append("")
        L.append("Step accuracy on correct-input steps: " + "; ".join(
            f"{a}: " + ", ".join(f"{k} {v['acc']:.2f} (n {v['n']})" for k, v in r["step_accuracy"][a].items())
            for a in [x for x in ARMS if x in r["step_accuracy"]]))
        L.append("")
    if res.get("chain_contrasts"):
        L += ["| chain contrast (k4; 10 parents, indicative) | point | 95% CI |", "|---|---|---|"]
        for k, b in res["chain_contrasts"].items():
            L.append(f"| {k} | {b['point']:+.3f} | [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
        L.append("")
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
        return None if math.isnan(x) else x
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------- synthetic fixtures (dry run)
BIG = 100003            # a perturbation that can never turn a wrong answer into the gold, the copy or the applied answer


def _bump(state: dict) -> dict:
    s = dict(state)
    k = sorted(s)[0]
    s[k] += BIG
    return s


def _onestep_text(rng, row, mode) -> str:
    g = json.loads(row["gold"])
    if mode == "correct":
        u = rng.random()
        return row["gold"] if u < 0.7 else (json.dumps(g) if u < 0.9 else row["gold"] + ".")
    if mode == "applied":
        return row["posted_gold"]
    if mode == "copy":
        return env.dumps(row["before"])
    if mode in ("wrong", "other"):
        return env.dumps(_bump(g))
    return "I am not sure."


def _chain_out(rng, c, p_step) -> dict:
    st, tx = [c["s0"]], []
    for t, ev in enumerate(c["events"]):
        cur = st[-1]
        good = C.gold_of(cur, ev)
        u = rng.random()
        if u < p_step:
            nxt = good
        elif u < p_step + (1 - p_step) * 0.1:
            tx.append("no idea")
            st.append(None)
            break
        else:
            nxt = C.gold_of(cur, {**ev, "status": "posted"}) if ev["status"] != "posted" else _bump(good)
        tx.append(env.dumps(nxt))
        st.append(nxt)
    states = [env.dumps(s) if s is not None else None for s in st] + [None] * (len(c["events"]) + 1 - len(st))
    return {"cand_id": c["cand_id"], "states": states, "texts": tx}


SCENARIOS = OrderedDict([
    ("rule_learned", {"specs": mk_specs(), "chain": {"on": 0.95, "off": 0.45, "supa": 0.2},
                      "expect": {"status": "STATUS: RULE (learned by the adapter)", "patterns": [],
                                 "groups": {g: "GENERALISES" for g in C.GEN_GROUPS}}}),
    ("trigger", {"specs": mk_specs(on={"U": 0.08, "N_postedlike": 0.05, "N_voidcase": 0.60,
                                       "M_bal": 0.30, "M_amt": 0.33, "A_renamed": 0.92, "A_many": 0.90, "F_keyorder": 0.60},
                                   off={"M_bal": 0.30, "M_amt": 0.33, "A_renamed": 0.92, "A_many": 0.90, "F_keyorder": 0.20}),
                 "chain": {"on": 0.6, "off": 0.45, "supa": 0.2},
                 "expect": {"status": "STATUS: TRIGGER", "patterns": [],
                            "groups": {"magnitude_at": "DOES NOT GENERALISE", "magnitude_scale": "DOES NOT GENERALISE",
                                       "naming": "ABOVE FLOOR WITHOUT A SHOWN ADAPTER GAIN",
                                       "format": "DOES NOT GENERALISE, with a partial gain over OFF"}}}),
    ("partial_seeds_disagree", {"specs": mk_specs(on={"M_bal": 0.10, "M_amt": 0.92, "A_renamed": 0.92, "A_many": 0.60, "F_keyorder": 0.92},
                                                  off={"M_bal": 0.80, "M_amt": 0.30, "A_renamed": 0.30, "A_many": 0.30, "F_keyorder": 0.30},
                                                  seeds={"s2": {"U_held": 0.50, "U_reversed": 0.45}}),
                                "chain": {"on": 0.8, "off": 0.45, "supa": 0.1},
                                # the pooled scale cell mixes a harmed M_bal (0.10 vs 0.80) with a gained M_amt
                                # (0.92 vs 0.30): ON 33/66 vs OFF 36/66, no significant difference, below the floor
                                "expect": {"status": "STATUS: PARTIAL", "patterns": [],
                                           "groups": {"magnitude_at": "HARMED", "magnitude_scale": "DOES NOT GENERALISE",
                                                      "naming": "MIXED", "format": "GENERALISES"}}}),
    ("rule_base_already", {"specs": mk_specs(on={"U": 0.98, "N": 0.98}, off={"U": 0.95, "N": 0.95, "V": 0.97, "gen": 0.35,
                                                                               "err": ERR_T_WRONG}),
                           "chain": {"on": 0.95, "off": 0.5, "supa": 0.2},
                           "expect": {"status": "STATUS: RULE (already present without the adapter)", "patterns": [],
                                      "groups": {g: "GENERALISES" for g in C.GEN_GROUPS}}}),
    ("rule_strengthened", {"specs": mk_specs(on={"U": 1.0, "N": 1.0}, off={"U": 0.85, "N": 0.85, "err": ERR_T_WRONG}),
                           "chain": {"on": 0.95, "off": 0.5, "supa": 0.2},
                           "expect": {"status": "STATUS: RULE (present in the base at the bar, strengthened by the adapter)",
                                      "patterns": [], "groups": {g: "GENERALISES" for g in C.GEN_GROUPS}}}),
    ("twins_fail", {"specs": mk_specs(on={"U": 0.99, "N": 0.99, "T": 0.55, "gen": 0.50}, off={"gen": 0.50}),
                    "chain": {"on": 0.5, "off": 0.45, "supa": 0.2},
                    "expect": {"status": "STATUS: PARTIAL", "patterns": ["twins"],
                               "groups": {g: "DOES NOT GENERALISE" for g in C.GEN_GROUPS}}}),
    ("semantic", {"specs": mk_specs(on={"U_neutral": 0.15}), "chain": {"on": 0.8, "off": 0.45, "supa": 0.2},
                  "expect": {"status": "STATUS: PARTIAL", "patterns": ["semantic"], "groups": {g: "GENERALISES" for g in C.GEN_GROUPS}}}),
    ("trigger_not_interpretable", {"specs": mk_specs(on={"U": 0.10, "N": 0.10, "V": 0.50}), "chain": {"on": 0.5, "off": 0.45, "supa": 0.2},
                                   "expect": {"status": "STATUS: PARTIAL", "patterns": ["trigger_not_interpretable"],
                                              "groups": {g: "GENERALISES" for g in C.GEN_GROUPS}}}),
])


def make_fixture_run(run: Path, rows: list, chains: list, scen: dict, seed: int) -> None:
    """A complete synthetic run directory in the runner's exact layout (provenance, DONE, every file), so the dry run
    exercises input validation. Exact counts per (arm, stratum, part) from synth_modes; texts rendered per mode."""
    rng = random.Random(seed)
    run.mkdir(parents=True, exist_ok=False)
    for arm in ("off", "s0", "s1", "s2", "supa"):
        md = synth_modes(rows, scen["specs"][arm], rng)
        texts = [_onestep_text(rng, r, m) for r, m in zip(rows, md)]
        p_ch = float(scen["chain"]["off" if arm == "off" else ("supa" if arm == "supa" else "on")])
        ch_rows = [_chain_out(rng, c, p_ch) for c in chains]
        if arm == "supa":
            fo, fc = run / "supa" / "onestep_supa.jsonl", run / "supa" / "chains_supa.jsonl"
        else:
            fo, fc = run / "onestep" / f"onestep_{arm}.jsonl", run / "chains" / f"chains_{arm}.jsonl"
        for f, lines in ((fo, [{"id": r["id"], "text": t} for r, t in zip(rows, texts)]), (fc, ch_rows)):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    anc = {"pass": True, "sides": {s: {"identical_states": 1.0, "identical_texts": 1.0} for s in R.SIDES}, "synthetic": True}
    (run / "anchor").mkdir(parents=True, exist_ok=True)
    for s in R.SIDES:
        (run / "anchor" / f"chain_one_L_{s}.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "anchor" / "anchor.json").write_text(json.dumps(anc) + "\n", encoding="utf-8")
    prov = {"synthetic_fixture": True, "anchor": anc, "turns": [{"stages_run": list(R.STAGE_ORDER)}], "stages": {}}
    for st in R.STAGE_ORDER:
        prov["stages"][st] = {"turn": 0, "pid": 1, "sides": {}, "files": {R.rel_out(p, run): sha256(p) for p in R.stage_required(st, run)}}
    (run / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
    done = {"files": {st: prov["stages"][st]["files"] for st in R.STAGE_ORDER}, "provenance_sha256": sha256(run / "provenance.json")}
    (run / "DONE.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")


def dry_run(bank: Path, out_dir: Path) -> dict:
    t0 = time.time()
    st = scorer_selftest(bank)
    need(st["ok"], f"scorer self-test failed: {json.dumps(clean(st['checks']))[:2000]}")
    ss = stats_selftest()
    ws = wording_selftest(bank)
    rows, chains = C.load_bank(bank)
    tmp = Path(tempfile.mkdtemp(prefix="p3_dryrun_"))
    scen_res, main_res = OrderedDict(), None
    try:
        for i, (name, scen) in enumerate(SCENARIOS.items()):
            run = tmp / name
            make_fixture_run(run, rows, chains, scen, DRY_SEED + i)
            I = Inputs(run, bank, registered=False)
            res, _ = analyse(I)
            got = {"status": res["wording"]["headline"], "patterns": res["V1_status"]["patterns"],
                   "groups": {g: v["verdict"] for g, v in res["V_groups"].items()}}
            ex = scen["expect"]
            ok = (got["status"] == ex["status"] and all(got["groups"][g] == v for g, v in ex["groups"].items())
                  and set(ex["patterns"]) <= set(got["patterns"])
                  and (ex["patterns"] or not set(got["patterns"]) & {"twins", "semantic", "posted_substring", "trigger_not_interpretable"}))
            scen_res[name] = {"expected": ex, "got": got, "ok": ok,
                              "unseen_acc": {a: [round(res["onestep"]["acc"][a][s]["acc"], 2) for s in C.UNSEEN_STATUS_STRATA]
                                             for a in ("off",) + SEEDS},
                              "twins": {a: round(res["onestep"]["acc"][a]["T_posted"]["acc"], 2) for a in ("off",) + SEEDS}}
            if name == "rule_learned":
                main_res = res
        bad = tmp / "tamper"
        make_fixture_run(bad, rows, chains, SCENARIOS["rule_learned"], DRY_SEED)
        p = bad / "onestep" / "onestep_s1.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        refused_changed = _refuses(lambda: Inputs(bad, bank, registered=False))
        bad2 = tmp / "tamper2"
        make_fixture_run(bad2, rows, chains, SCENARIOS["rule_learned"], DRY_SEED)
        p = bad2 / "chains" / "chains_s0.jsonl"
        cl = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        cl[0]["states"][1] = env.dumps({"A": 1, "B": 2, "C": 3, "D": 4})
        p.write_text("".join(json.dumps(x) + "\n" for x in cl), encoding="utf-8")
        prov = json.loads((bad2 / "provenance.json").read_text(encoding="utf-8"))
        prov["stages"]["S2"]["files"]["chains/chains_s0.jsonl"] = sha256(p)
        (bad2 / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
        done = json.loads((bad2 / "DONE.json").read_text(encoding="utf-8"))
        done["files"]["S2"]["chains/chains_s0.jsonl"] = sha256(p)
        done["provenance_sha256"] = sha256(bad2 / "provenance.json")
        (bad2 / "DONE.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        refused_state = _refuses(lambda: Inputs(bad2, bank, registered=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"what": "DRY RUN on synthetic fixtures (NOT model outputs): statistics unit tests; wording self-test through "
                   "the real summary / McNemar / Holm; every scenario through the full pipeline incl. input validation; "
                   "the verdicts must equal the scenario's design", "bank_sha256": sha256(bank),
           "scorer_selftest": st, "stats_selftest": ss, "wording_selftest": ws, "scenarios": scen_res,
           "tamper": {"refuses a truncated one-step file / hash change": refused_changed,
                      "refuses a chain state that is not the parse of its text": refused_state},
           "power": power_table(), "registered_run_refuses_without_freeze": _refuses(lambda: Inputs(R.OUT, bank, registered=True)),
           "code_sha256": {r: sha256(ROOT / r) for r in R.DRYRUN_CODE + ("scripts/p3_run.py", "scripts/confirm2_analysis.py")},
           "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1)}
    out["ok"] = bool(st["ok"] and ss["ok"] and ws["ok"] and all(v["ok"] for v in scen_res.values()) and refused_changed
                     and refused_state and out["registered_run_refuses_without_freeze"] and not out["torch_imported"])
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render(main_res, "PREREG-3 DRY RUN on synthetic fixtures (scenario 'rule_learned'; NOT results)")
    md += "\n## All dry-run scenarios (synthetic; expected = designed verdicts)\n\n| scenario | expected | got | ok |\n|---|---|---|---|\n"
    for n_, v in scen_res.items():
        md += f"| {n_} | {v['expected']['status']}; {v['expected']['patterns']}; {v['expected']['groups']} | {v['got']['status']}; {v['got']['patterns']}; {v['got']['groups']} | {v['ok']} |\n"
    md += "\n## Wording self-test (real summary, McNemar and Holm on synthetic per-row classes)\n\n" + "\n".join(
        f"- {k}: {'ok' if v['ok'] else 'FAIL'}" for k, v in ws["cases"].items()) + "\n"
    md += "\n## Statistics unit tests\n\n" + "\n".join(f"- {k}: {'ok' if v['ok'] else 'FAIL'}" for k, v in ss["cases"].items()) + "\n"
    pw = out["power"]
    md += ("\n## Power (" + pw["assumption"] + ")\n\n| true acc per unseen cell | P(cell meets bar) | P(all 24 cells) | "
           "P(RULE), twins 0.90 copied 0.03 | P(RULE), twins at the same acc, copied min(0.03, 1 - acc) |\n|---|---|---|---|---|\n")
    for r in pw["rule"]:
        md += (f"| {r['true_acc_per_unseen_cell']} | {r['P_cell_meets_bar']} | {r['P_all_24_unseen_cells']} | "
               f"{r['P_RULE_twins_at_0.90']} | {r['P_RULE_twins_at_same_acc_copied_min_0.03']} |\n")
    md += "\n| true acc (trigger set) | P(cell <= 0.30) | P(all 21 cells) |\n|---|---|---|\n"
    for r in pw["trigger"]:
        md += f"| {r['true_acc']} | {r['P_cell_trigger_low']} | {r['P_all_21_cells']} |\n"
    md += "\n| group | rows per cell | cells | min correct | true acc | P(cell meets floor) | P(all cells) |\n|---|---|---|---|---|---|---|\n"
    for r in pw["groups"]:
        md += (f"| {r['group']} | {r['rows_per_cell']} | {r['cells']} | {r['min_correct']} | {r['true_acc']} | "
               f"{r['P_cell_meets_floor']} | {r['P_all_cells_meet_floor']} |\n")
    md += "\n| McNemar rows | ON | OFF | worst-case Holm m | power (20,000 simulations, independent errors) |\n|---|---|---|---|---|\n"
    for r in pw["mcnemar"]:
        md += f"| {r['rows']} | {r['on']} | {r['off']} | {r['holm_worst_case_m']} | {r['power']} |\n"
    md += f"\nDry run ok: {out['ok']}\n"
    (out_dir / "P3_DRYRUN.json").write_text(json.dumps(clean(out), indent=1) + "\n", encoding="utf-8")
    (out_dir / "P3_DRYRUN.md").write_text(md, encoding="utf-8")
    return out


def _refuses(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--selftest", action="store_true", help="scorer self-test on the bank (no model output read)")
    g.add_argument("--dry-run", action="store_true", help="synthetic fixtures through the full pipeline")
    ap.add_argument("--bank", default=None, help="selftest / dry run only: another bank file (rehearsal)")
    ap.add_argument("--out-dir", default=None, help="selftest / dry run only: output directory (default docs/prereg3)")
    a = ap.parse_args()
    frozen = (ROOT / R.FREEZE_REL).exists()
    if (a.bank or a.out_dir) and not (a.selftest or a.dry_run):
        raise SystemExit("--bank / --out-dir are for --selftest / --dry-run only")
    bank = Path(a.bank).resolve() if a.bank else C.BANK
    out_dir = Path(a.out_dir).resolve() if a.out_dir else C.DOCS
    if (a.selftest or a.dry_run) and frozen and out_dir == C.DOCS.resolve():
        raise SystemExit("PREREG-3 is frozen: docs/prereg3 records are frozen inputs; use --out-dir elsewhere")
    if a.selftest:
        st = scorer_selftest(bank)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SCORER_SELFTEST.json").write_text(json.dumps(clean(st), indent=1) + "\n", encoding="utf-8")
        for k, v in st["checks"].items():
            print(f"  [{'PASS' if v['ok'] else 'FAIL'}] {k}")
        print(f"scorer self-test: {'PASS' if st['ok'] else 'FAIL'} ({st['n_rows']} rows, {st['n_chains']} chains); "
              f"wrote {out_dir / 'SCORER_SELFTEST.json'}")
        need("torch" not in sys.modules, "torch was imported")
        return 0 if st["ok"] else 1
    if a.dry_run:
        out = dry_run(bank, out_dir)
        for n_, v in out["scenarios"].items():
            print(f"  scenario {n_}: {'ok' if v['ok'] else 'MISMATCH'} -> {v['got']}")
        bad_w = [k for k, v in out["wording_selftest"]["cases"].items() if not v["ok"]]
        print(f"  wording self-test: {'ok' if out['wording_selftest']['ok'] else 'FAIL'} "
              f"({len(out['wording_selftest']['cases'])} cases){'; failing: ' + str(bad_w) if bad_w else ''}")
        print(f"  statistics unit tests: {'ok' if out['stats_selftest']['ok'] else 'FAIL'} ({len(out['stats_selftest']['cases'])} cases)")
        print(f"  tamper refusals: {out['tamper']}; registered run refuses without a freeze: {out['registered_run_refuses_without_freeze']}")
        print(f"dry run: {'PASS' if out['ok'] else 'FAIL'} in {out['seconds']}s; wrote {out_dir / 'P3_DRYRUN.json'} (+ .md)")
        return 0 if out["ok"] else 1
    # the registered analysis: freeze, then the scorer self-test, THEN the model outputs
    t0 = time.time()
    out_dir = R.OUT / "analysis"
    for f in OUT_NAMES:
        if (out_dir / f).exists():
            raise SystemExit(f"{out_dir / f} exists: the registered analysis runs once (create-only)")
    ok, errs, _ = R.verify_freeze()
    if not ok:
        die("the PREREG-3 freeze does not verify; no analysis", errs)
    st = scorer_selftest(C.BANK)
    need(st["ok"], "the scorer self-test failed: no model output is read")
    st_time = time.time()
    I = Inputs(R.OUT, C.BANK, registered=True)
    res, per_row = analyse(I)
    res["scorer_selftest"] = {"ok": st["ok"], "finished_unix": st_time, "before_outputs_read": True}
    res["inputs"] = {"code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in
                                     [Path(__file__), ROOT / "scripts" / "p3_common.py", ROOT / "scripts" / "p3_run.py",
                                      ROOT / "scripts" / "confirm2_analysis.py"] + sorted((ROOT / "rcwmt").glob("*.py"))},
                     "numpy": np.__version__, "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1)}
    need("torch" not in sys.modules, "torch was imported")
    md = render(res)
    js = json.dumps(clean(res), indent=1) + "\n"

    def w_text(s):
        def f(p):
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(s)
        return f

    def w_npz(p):
        with open(p, "wb") as fh:
            np.savez(fh, **per_row)
    C2._write_together(out_dir, {OUT_NAMES[0]: w_text(js), OUT_NAMES[1]: w_npz, OUT_NAMES[2]: w_text(md)})
    print(f"wrote {out_dir / OUT_NAMES[0]} (+ .md, per-row .npz)")
    print(f"verdict: {res['wording']['headline']}; " + "; ".join(f"{g}: {v['verdict']}" for g, v in res["V_groups"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
