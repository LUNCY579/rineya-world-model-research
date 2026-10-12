"""B4 futility screen runner -- DRAFT (research/PREREG_B4_SCREEN_DRAFT.md; D-75 decision 1; D-76a; D-77; RCX co-sign
review 9dd352ee..., 13 required edits accepted in D-80; canary 9e9ad339... and RCX's low items, D-80 follow-up / D-81).

One arm, a = RCX's hybrid hook `LearnedUnmet.from_file(gnn_s20261276.pt, invalid_policy="fallback", hybrid_lambda=1.0)`
(rcx/bw/b4_hook.py, every executed file sha-pinned, loaded from a sha-verified snapshot taken at the freeze) passed as
`unmet=` to the pinned B2 `plan_beam` (bwmn/blocks_planner.py 18a5f22b..., imported read-only, never patched) with item
7's R0 S-beam arguments (runs/FREEZE_I7_R0_beam.json = FREEZE_B2_beam args) on seeds_fresh-40. The control is item 7's
own R0 beam run (runs/BW_SEEDS/R0_beam): same F, Pa, eval_seed, CRN seeds and shard partition (index % 4); paired per
instance.

  python -B scripts/b4_screen.py --dry-run          # CPU; no model, no torch; stub F/proposer; outputs to scratch
  python -B scripts/b4_screen.py oc [--out DIR]     # CPU; operating characteristics of the futility rule
  python -B scripts/b4_screen.py precheck --out DIR # CPU; GNN train/val goal classes vs seeds_fresh (prereg 4.3)
  python -B scripts/b4_screen.py final-diff --draft D --final F   # CPU; a finalisation may touch only title/Status
  python -B scripts/b4_screen.py freeze --prereg P --addendum A   # CPU; runs/FREEZE_B4S_a_beam.json (once)
  python -B scripts/b4_screen.py canary             # CPU pre-flight (imports torch, 1 thread): RCX golden values
  python -B scripts/b4_screen.py run [--rerun]      # GPU; 4 resumable shards per arm, each a run-shard subprocess
  python -B scripts/b4_screen.py analyze [--rerun]  # CPU; runs/BW_B4_SCREEN/B4S_REPORT{,_RERUN}.{json,md}

Dry run: the pinned planner bytes are executed into a private module whose `bwmn.lm` is a pure-python stand-in (the
two generation functions the planner calls are routed to a scripted model: exact-step F text with F-09-style false
"legal" answers on illegal actions, and a noisy teardown-and-build proposer with per-candidate CRN seeds), so the real
LMForwardBW / LMProposerBW / plan_beam / score code runs with the frozen B2 beam arguments. The hook module is loaded
through a sha-verifying importer (exec of verified bytes; no bytecode); the arm uses LearnedUnmet with ONLY its GNN
forward replaced by 0.0 (so the dry-run arm is "misplaced_blocks + the invalid-state penalty"), never torch. Instances
are generated with a dry-run-only seed (7411); seeds_fresh is hashed but never planned on in a dry run.

Nothing here writes outside --out (dry-run / oc / precheck) or, for the registered stages, outside
runs/FREEZE_B4S_a_beam.json and runs/BW_B4_SCREEN/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import types
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------- pins (prereg section 2)

PLANNER_REL = "bwmn/blocks_planner.py"
PLANNER_SHA256 = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"  # D-57/D-59/D-63
BLOCKS_REL = "bwmn/blocks.py"
BLOCKS_SHA256 = "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83"
B2_BEAM_FREEZE_REL = "runs/FREEZE_B2_beam.json"
B2_BEAM_FREEZE_SHA256 = "412a6ee51af1cbc6f464a7d27d0150526dac90fd32c76b2e370b590084b41c94"
ITEM7_PREREG_REL = "PREREG_ITEM7_SEEDS.md"
ITEM7_PREREG_SHA256 = "956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d"
SEEDS_FRESH_REL = "runs/BW_val_large/instances_seeds_fresh.json"
SEEDS_FRESH_SHA256 = "6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc"
N_FRESH = 40
CODE_PATH_PINS = {  # co-sign edit 6: the rest of the paired code path (R0 ran through these files; so must the arm)
    "bwmn/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",          # empty file
    "bwmn/lm.py": "88a1e3a26b518cebd17e0b8197d1dd10872d549a4a27a69744ed59266895c0ac",                # generation
    "scripts/blocks_eval.py": "b3666424151c78368500535d2571f0eb5b9e546b11764696d99f427a7794acb6",    # R0's driver
    "scripts/bw_run_tests.py": "de7022d1eacebab271f7624e962d40d58aeee435494da079a622a03f80a34542",   # merge
    "scripts/bw_seeds.py": "b783a1916bce655cf93e3d55bedcb276dae3dab2bbdc214e00ff633c828f3b08",       # item 7 runner
}
TESTS_REL = "tests/test_b4_screen.py"
R0_FREEZE_REL = "runs/FREEZE_I7_R0_beam.json"   # written by scripts/bw_seeds.py freeze (not yet at drafting)
R0_OUT_REL = "runs/BW_SEEDS/R0_beam"
BW_SEEDS_REL = "runs/BW_SEEDS"
BW_SEEDS_LOG_REL = "runs/BW_SEEDS.stdout"       # bundleI's i7-* log (outside runs/BW_SEEDS/)
SEEDS_REPORT_REL = "runs/BW_SEEDS/SEEDS_REPORT.json"            # item 7's analysis (bw_seeds.py analyze)
ADDENDUM_A_INPUTS_REL = "runs/BW_SEEDS/ADDENDUM_A_INPUTS.json"  # addendum A.0 (scripts/item7_addendum_a.py hash)
CLOSING_MARKERS = (SEEDS_REPORT_REL, ADDENDUM_A_INPUTS_REL)    # edit 9: once either exists, no screen freeze at all
CLOSING_LOG_MARKER = "release_gate_reading"   # RCX low item 5: bw_seeds.py analyze prints its report into the log
FREEZE_REL = "runs/FREEZE_B4S_a_beam.json"
OUT_REL = "runs/BW_B4_SCREEN"
SNAPSHOT_REL = OUT_REL + "/rcx_snapshot"        # edit 10: sha-verified copies of every RCX file, taken at the freeze
ATTEMPTS_REL = OUT_REL + "/ATTEMPTS.jsonl"      # append-only record of every run attempt
SHARDS = 4                                       # = bw_seeds.BEAM_SHARDS: the same index % 4 partition as R0
F_BATCH_TOKENS, PROP_BATCH_TOKENS = 5000, 6000   # blocks_eval.py defaults; outputs do not depend on them (D-48c)
BLOCKS_EVAL_ARG_KEYS = ("proposer", "pa", "mode", "fwd", "fwd_adapter", "fwd_fmt", "K", "eval_seed", "planner", "W",
                        "E", "max_depth", "step_cap", "R", "M", "P")   # the keys blocks_eval.py checks against a freeze
# arms: the screen's arm a; the pre-registered re-run after an instrument fault (edit 7): c and a in one GPU turn
ARMS = {"a": {"dir": "a_beam", "hook": True},
        "rerun_c": {"dir": "rerun/c_beam", "hook": False},
        "rerun_a": {"dir": "rerun/a_beam", "hook": True}}
REPORT = {False: "B4S_REPORT", True: "B4S_REPORT_RERUN"}

RCX_ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary")
RCX_CODE_PINS = {  # every RCX file the hook import executes (rel. to the RCX root or the snapshot)
    "rcx/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",       # empty file
    "rcx/bw/__init__.py": "4f403acd75709867f8dc7f6952a4e63e702ebaa40001dc2cb7024a6feb1440aa",
    "rcx/bw/b4_hook.py": "a0b4f88c32797feba95115d232f733b8dd47610c9e3018af6b90b297bfdeec81",     # D-76a
    "rcx/bw/bwm_import.py": "84140c7ea91e4066ea3f9ada6a95b6b329b665329b1c52ea38d305e3ee517b9e",
    "rcx/bw/encode.py": "32980f141eaf0fac4757ee253d22c59c55f889ae98c26eddd37eef055b928a8a",      # D-76a
    "rcx/bw/space.py": "29d0b4203156a98521475ce4eda4e7951dcaa2a31bae29b1c3c96c268e015fa4",       # D-76a
    "rcx/bw/models.py": "4c2e3242eadc4ee155d8ddedb5f4188ac3aa5f70a19061145ff0e2fbee118f52",      # D-76a (torch; lazy)
}
RCX_D76_FILES = ("rcx/bw/b4_hook.py", "rcx/bw/models.py", "rcx/bw/encode.py", "rcx/bw/space.py")
GNN_REL = "runs/RCX-E07/main/models/gnn_s20261276.pt"
GNN_SHA256 = "9cc417a3075391e4fc8022f0dddd811e90fb4647c6a88dcd81361e9bb2c13f04"            # D-76a
GNN_SIDECAR_REL = "runs/RCX-E07/main/models/gnn_s20261276.json"
GNN_SIDECAR_SHA256 = "2ccdb2798fc784327c153fe7a6dd5479b2646f0cb29314aad3610eed1aa4a892"    # D-76a
E07_INSTANCES_REL = "runs/RCX-E07/main/data/instances.json"                                # GNN train/val/test
E07_INSTANCES_SHA256 = "1e39f9bbc25a7e857ccbfee97c996638fed63d606c334efef88fc3b7a32f4e31"
CANARY_REL = "experiments/RCX-B4/canary/B4_HOOK_CANARY.json"   # edit 8: RCX's golden-value file (rel. to the RCX root
#   or the snapshot; copied into the snapshot at the freeze). 25 probes from E07 test and E07b TEST-B (never
#   seeds_fresh).
CANARY_SHA256 = "9e9ad3395b3b274db90f96ff47e8fee2e7d7329f045618f88643e0899a11e87a"   # RCX commit 42cbfcc (D-80 f/u)
CANARY_TOL = 1e-5      # |value - expected| per probe, asserted at every hook shard start, before planning
HOOK_CLASS = "rcx.bw.b4_hook.LearnedUnmet"
HOOK_KWARGS = {"invalid_policy": "fallback", "hybrid_lambda": 1.0}                       # D-76a: exactly this call
VERSION_PACKAGES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "numpy")

# ----------------------------------------------------------------------------- the registered rule (prereg section 5)

GO_MIN_NET = 2                        # BWM decision 2026-10-11 (D-77): net >= 2; the sign p is reported, never gating
PENALTY_SHARE_MAX = 0.05              # D-76a attribution rule (call level): GO not credited if share > 0.05
RULE_TEXT = ("GO iff (a_only - R0_only) >= 2; otherwise NO-GO (net <= 1 is always NO-GO). The exact one-sided "
             "sign-test p and the two-sided McNemar p are reported only and do not gate (D-77). If more than 5% of hook "
             "calls return a penalty-band value (invalid state or critic error; value >= PENALTY = 1000), a GO is not "
             "credited to the learned critic alone and c+pen is mandatory in the full B4 (D-76a). The screen is VOID "
             "(no GO/NO-GO) on any hook model/encode/nonfinite/failsafe error, model_evals = 0, a canary failure or R0 "
             "outputs that do not match ADDENDUM_A_INPUTS; it is an INSTRUMENT FAULT (no decision; a and c re-run in "
             "one turn) if the round-1 identity check or the R0 shard-argument check fails.")
REPORTED_ONLY = ("p_one_sided_sign", "p_mcnemar_two_sided")   # edit 3: no p-value is part of the frozen rule
VOID_COUNTERS = ("model_errors", "encode_errors", "nonfinite", "failsafe")                # edit 8
WORDING = {
    "GO": ("B4 screen GO: arm (a), the RCX hybrid hook, passed the pre-set futility rule on seeds_fresh-40, paired vs "
           "item 7's R0 beam. This unlocks the co-signed B4 primary (fresh seed-20261410 set, n = 110). It is not a "
           "claim that the hook improves planning, and the screen's net and p are not effect estimates."),
    "NO-GO": ("B4 arm (a) not supported at screen (seeds_fresh-40, paired vs R0, pre-set futility rule). The full B4 "
              "is not run; B6 follows (D-75.1)."),
    "VOID": ("B4 screen VOID: a registered validity condition failed (see void_reasons). No GO/NO-GO is read; a void "
             "screen needs a new prereg."),
    "INSTRUMENT_FAULT": ("B4 screen INSTRUMENT FAULT: the round-1 identity check failed, so the pairing with R0 is not "
                         "trusted. No decision. Pre-registered consequence: arms c and a are re-run in one GPU turn "
                         "(`run --rerun`) and analysed with `analyze --rerun`."),
    "INSTRUMENT_FAULT_R0_ARGS": ("B4 screen INSTRUMENT FAULT: R0's shard arguments differ from the frozen S-beam "
                                 "configuration, so R0 is not the registered control (see instrument_fault_reasons "
                                 "for whether arm a ran). No decision. Pre-registered consequence: arms c and a are "
                                 "re-run in one GPU turn (`run --rerun`) and analysed with `analyze --rerun`."),
    "INSTRUMENT_FAULT_RERUN": ("B4 screen INSTRUMENT_FAULT_RERUN: the pre-registered re-run failed the round-1 identity "
                               "check as well. Terminal: no decision and no further re-run; the screen closes without "
                               "a reading (a new prereg is needed)."),
}
ATTRIBUTION_NOTE = (" Not credited to the learned critic alone: more than 5% of hook calls returned a penalty-band value "
                    "(invalid state or critic error), so c+pen is mandatory in the full B4 (D-76a).")

DRY_SEED = 7411                          # dry-run only; never a data or evaluation seed
DRY_SIZES = (5, 5, 6, 6, 7, 7, 8, 8)
DRY_OUT = Path("C:/Users/clclc/AppData/Local/Temp/claude/bwm_d75_scratch/C/dryrun")

_MISSING = object()


class PinError(RuntimeError):
    """A pinned file does not match its sha256, or a module would be imported from an unpinned source."""


class CanaryError(RuntimeError):
    """The hook did not reproduce RCX's golden values (edit 8): the shard stops before planning; the screen is VOID."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_pin(path, want: str, what: str) -> str:
    path = Path(path)
    if not path.exists():
        raise PinError(f"{what}: {path} does not exist")
    got = sha256_file(path)
    if got != want:
        raise PinError(f"{what}: {path} sha256 {got} != pinned {want}")
    return got


def rcx_snapshot_files() -> dict:
    """Every RCX file the screen reads: the executed code, the GNN and its sidecar, E07's instances (pre-check) and the
    canary. Refuses while the canary sha is the placeholder."""
    if CANARY_SHA256 is None:
        raise PinError("the RCX canary golden file is not registered yet (CANARY_SHA256 placeholder): refusing (edit 8)")
    return {**RCX_CODE_PINS, GNN_REL: GNN_SHA256, GNN_SIDECAR_REL: GNN_SIDECAR_SHA256,
            E07_INSTANCES_REL: E07_INSTANCES_SHA256, CANARY_REL: CANARY_SHA256}


# ----------------------------------------------------------------------------- statistics (pure python, exact)

def sign_p_one_sided(g: int, l: int) -> Fraction:
    """P[X >= g], X ~ Binomial(g + l, 1/2): the exact one-sided sign (McNemar) test on the discordant pairs."""
    n = g + l
    return Fraction(sum(math.comb(n, i) for i in range(g, n + 1)), 2 ** n) if n else Fraction(1)


def mcnemar_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p = min(1, 2 P[X <= min(b, c)]), X ~ Bin(b + c, 1/2) (= scipy binomtest at p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(Fraction(1), 2 * Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)), 2 ** n)))


def futility_decision(g: int, l: int, go_min_net: int = GO_MIN_NET) -> dict:
    """The registered screen rule on the paired table: g = arm-only successes, l = control-only successes. The p-values
    are reported only (edit 3)."""
    go = (g - l) >= go_min_net
    why = ("net <= 0 (D-76a NO-GO)" if g - l <= 0 else
           f"net {g - l} < {go_min_net}" if not go else f"net {g - l} >= {go_min_net}")
    return {"arm_only": g, "control_only": l, "net": g - l, "p_one_sided_sign": float(sign_p_one_sided(g, l)),
            "p_mcnemar_two_sided": mcnemar_two_sided(g, l), "reported_only": list(REPORTED_ONLY),
            "decision": "GO" if go else "NO-GO", "go": go, "because": why}


def min_gains_for_go(max_losses: int = 15) -> dict:
    return {l: next(g for g in range(l, l + 200) if futility_decision(g, l)["go"]) for l in range(max_losses + 1)}


def binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list:
    """Exact two-sided Clopper-Pearson interval by bisection on the binomial tails (rounded to 4 decimals)."""
    if n == 0:
        return [None, None]

    def bisect(f, lo=0.0, hi=1.0):  # f increasing in p, root in (lo, hi)
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else bisect(lambda p: (1 - binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else bisect(lambda p: alpha / 2 - binom_cdf(k, n, p))
    return [round(lower, 4), round(upper, 4)]


# ----------------------------------------------------------------------------- operating characteristics (oc)

OC_RULES = {
    "registered: net>=2": lambda g, l: futility_decision(g, l)["go"],
    "superseded by D-77: net>=3 & p1<=0.20": lambda g, l: g - l >= 3 and sign_p_one_sided(g, l) <= Fraction(1, 5),
    "panel: net>=3": lambda g, l: g - l >= 3,
    "task: (net>=4 & L<=1) | p2<0.10": lambda g, l: ((g - l >= 4 and l <= 1)
                                                    or (g > l and mcnemar_two_sided(g, l) < 0.10)),
    "net>=4": lambda g, l: g - l >= 4,
    "D-76a minimum: net>=1": lambda g, l: g - l >= 1,
}
OC_SCENARIOS = [  # (label, P(arm-only), P(control-only)) per instance
    ("null, 5% discordant", 0.025, 0.025), ("null, 10% discordant", 0.05, 0.05),
    ("null, 20% discordant", 0.10, 0.10), ("null, 30% discordant", 0.15, 0.15),
    ("harm -5 pp", 0.05, 0.10),
    ("+5 pp, low loss", 0.075, 0.025), ("+5 pp", 0.10, 0.05),
    ("+9 pp, low loss", 0.115, 0.025), ("+9 pp", 0.14, 0.05), ("+9 pp, high loss", 0.165, 0.075),
    ("+13.6 pp (P-14 H2 discordance)", 0.182, 0.045),
    ("+20 pp", 0.25, 0.05),
]
OC_D_GRID = (0.05, 0.10, 0.15, 0.20, 0.227, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0)   # 0.227 = H2's discordance (P-14)
OC_EFFECTS = (("null", 0.0), ("+5 pp", 0.05), ("+9 pp", 0.09), ("+13.6 pp", 0.136))


def _cells(n: int, pg: float, pl: float):
    pn = 1.0 - pg - pl
    for g in range(n + 1):
        for l in range(n - g + 1):
            yield g, l, math.comb(n, g) * math.comb(n - g, l) * pg ** g * pl ** l * pn ** (n - g - l)


_GO_CELLS: dict = {}


def _go_cells(rule, n: int, key) -> frozenset:
    """The (g, l) cells where `rule` says GO at n (computed once per rule and n)."""
    if (key, n) not in _GO_CELLS:
        _GO_CELLS[(key, n)] = frozenset((g, l) for g in range(n + 1) for l in range(n - g + 1) if rule(g, l))
    return _GO_CELLS[(key, n)]


def p_go(rule, n: int, pg: float, pl: float, key=None) -> float:
    cells = _go_cells(rule, n, key if key is not None else id(rule))
    return sum(pr for g, l, pr in _cells(n, pg, pl) if (g, l) in cells)


def primary_power(n: int, pg: float, pl: float, alpha: float) -> float:
    """Context only: P(exact two-sided McNemar p < alpha and gains > losses) for one B4 primary contrast at n."""
    return p_go(lambda g, l: g > l and mcnemar_two_sided(g, l) < alpha, n, pg, pl, key=("primary", alpha))


def oc_dgrid(n: int = 40, ds=OC_D_GRID, effects=OC_EFFECTS) -> dict:
    """Edit 4: P(GO) of the registered rule by discordance d = P(a-only) + P(R0-only) and effect e = P(a-only) -
    P(R0-only) (so P(a-only) = (d + e) / 2); None where the effect does not fit in d. Plus the supremum of the null
    false-GO rate over d in (0, 1] (grid step 0.01)."""
    reg = OC_RULES["registered: net>=2"]
    rows = []
    for d in ds:
        row = {"d": d}
        for name, e in effects:
            pg, pl = (d + e) / 2, (d - e) / 2
            row[name] = round(p_go(reg, n, pg, pl, key="registered: net>=2"), 3) if pl >= 0 and pg + pl <= 1 else None
        rows.append(row)
    nulls = [(round(k / 100, 2), p_go(reg, n, k / 200, k / 200, key="registered: net>=2")) for k in range(1, 101)]
    d_max, p_max = max(nulls, key=lambda x: x[1])
    return {"what": f"P(GO) of 'net >= {GO_MIN_NET}' at n = {n} by discordance d and effect (exact enumeration)",
            "rows": rows, "null_sup_over_d": {"d": d_max, "p_go": round(p_max, 4)},
            "h2_anchor_d": 0.227, "note": "B2 solves 0/110 hard instances in round 1 (mirror 1/110): almost every "
                                          "instance survives round 1 and is exposed to the ranking key"}


def oc_table(ns=(40, 39)) -> dict:
    rows = []
    for name, pg, pl in OC_SCENARIOS:
        row = {"scenario": name, "p_arm_only": pg, "p_control_only": pl}
        for n in ns:
            row[f"n{n}"] = {r: round(p_go(f, n, pg, pl, key=r), 3) for r, f in OC_RULES.items()}
        row["b4_primary_power_n110_alpha0.025"] = round(primary_power(110, pg, pl, 0.025), 3)
        rows.append(row)
    min_gains = {r: {str(l): next((g for g in range(l, 60) if f(g, l)), None) for l in range(8)}
                 for r, f in OC_RULES.items()}
    return {"what": "P(GO) by scenario (exact multinomial enumeration over the paired table)", "rules": list(OC_RULES),
            "rows": rows, "min_arm_only_for_go_by_control_only": min_gains, "d_grid": oc_dgrid(max(ns))}


# ----------------------------------------------------------------------------- the pinned planner

def make_stub_lm():
    """Pure-python stand-in for bwmn.lm (DRY-RUN only): routes the planner's two generation calls to the scripted
    model; anything else that would load or run a model raises."""
    m = types.ModuleType("bwmn.lm")
    m.__doc__ = "b4_screen DRY-RUN stand-in for bwmn.lm (no torch; generation routed to ScriptedModel)"

    def greedy_shared_prefix(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
        return [model.forward_text(p, max_new_tokens, tok) for p in prompts]

    def sample_crn_shared(model, tok, prompts, seeds, *, max_new_tokens=64, temperature=1.0, top_p=0.95,
                          max_batch_tokens=2000):
        if len(prompts) != len(seeds):
            raise ValueError("prompts / seeds length mismatch")
        return [[model.proposer_text(p, s, max_new_tokens, tok) for s in row] for p, row in zip(prompts, seeds)]

    def _no_model(*a, **k):
        raise RuntimeError("b4_screen DRY-RUN: the stub bwmn.lm cannot load or run a model")

    m.greedy_shared_prefix, m.sample_crn_shared = greedy_shared_prefix, sample_crn_shared
    m.load_core = m.sample_crn = m.greedy = _no_model
    m._b4s_stub = True
    return m


def exec_pinned_planner(stub_lm, path=None):
    """(B, BP) for the DRY RUN: the sha-verified bytes of bwmn/blocks_planner.py executed into a private module object
    whose `lm` is `stub_lm`. Nothing is registered in sys.modules and the file is never modified; sys.modules['bwmn.lm']
    and the package attribute are restored afterwards (a real bwmn.lm, if loaded, is untouched)."""
    path = Path(path or ROOT / PLANNER_REL)
    data = path.read_bytes()
    got = sha256_bytes(data)
    if got != PLANNER_SHA256:
        raise PinError(f"{path}: sha256 {got} != pinned {PLANNER_SHA256} (D-63); refusing to plan with it")
    import bwmn
    from bwmn import blocks as B
    check_pin(B.__file__, BLOCKS_SHA256, BLOCKS_REL)
    saved_mod, saved_attr = sys.modules.get("bwmn.lm", _MISSING), vars(bwmn).get("lm", _MISSING)
    sys.modules["bwmn.lm"] = stub_lm
    bwmn.lm = stub_lm
    try:
        mod = types.ModuleType("bwmn.blocks_planner")
        mod.__file__, mod.__package__ = str(path), "bwmn"
        exec(compile(data, str(path), "exec", dont_inherit=True), mod.__dict__)
    finally:
        if saved_mod is _MISSING:
            sys.modules.pop("bwmn.lm", None)
        else:
            sys.modules["bwmn.lm"] = saved_mod
        if saved_attr is _MISSING:
            vars(bwmn).pop("lm", None)
        else:
            bwmn.lm = saved_attr
    if mod.lm is not stub_lm or mod.B is not B:
        raise PinError("the private planner did not bind the stub lm and the pinned blocks module")
    return B, mod


def import_planner_real():
    """(B, BP) for the GPU run: the normal import (as blocks_eval.py does for R0), sha-checked before and after; the
    planner's bwmn.lm must be the pinned file (edit 6)."""
    check_pin(ROOT / PLANNER_REL, PLANNER_SHA256, PLANNER_REL)
    check_pin(ROOT / BLOCKS_REL, BLOCKS_SHA256, BLOCKS_REL)
    from bwmn import blocks as B
    from bwmn import blocks_planner as BP
    if Path(BP.__file__).resolve() != (ROOT / PLANNER_REL).resolve():
        raise PinError(f"imported blocks_planner {BP.__file__} is not {ROOT / PLANNER_REL}")
    check_pin(BP.__file__, PLANNER_SHA256, PLANNER_REL)
    if getattr(BP.lm, "_b4s_stub", False):
        raise PinError("the GPU run imported a planner bound to the dry-run stub lm")
    if Path(BP.lm.__file__).resolve() != (ROOT / "bwmn/lm.py").resolve():
        raise PinError(f"the planner's bwmn.lm is {BP.lm.__file__}, not the pinned bwmn/lm.py")
    check_pin(BP.lm.__file__, CODE_PATH_PINS["bwmn/lm.py"], "bwmn/lm.py")
    return B, BP


# ----------------------------------------------------------------------------- the pinned RCX hook

class RCXPinnedImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta-path importer for the `rcx` package from `root`: every module is executed from bytes whose sha256 equals
    its pin (no bytecode is read or written); importing an unpinned rcx module raises PinError. Installed first on
    sys.meta_path for the life of the process, so the hook's lazy `from . import models` is verified too."""

    def __init__(self, root, pins):
        self.root, self.pins, self.executed = Path(root), dict(pins), {}

    @staticmethod
    def _rel(fullname: str):
        parts = fullname.split(".")
        if parts[0] != "rcx":
            return None
        if len(parts) == 1:
            return "rcx/__init__.py"
        if parts[1] == "bw" and len(parts) == 2:
            return "rcx/bw/__init__.py"
        return "/".join(parts) + ".py"

    def find_spec(self, fullname, path=None, target=None):
        rel = self._rel(fullname)
        if rel is None:
            return None
        if rel not in self.pins:
            raise PinError(f"RCX module {fullname} ({rel}) is not pinned; refusing to import it")
        origin = self.root / rel
        spec = importlib.machinery.ModuleSpec(fullname, self, origin=str(origin),
                                              is_package=rel.endswith("/__init__.py"))
        if spec.submodule_search_locations is not None:
            spec.submodule_search_locations.append(str(origin.parent))
        spec.has_location = True
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        rel = self._rel(module.__name__)
        origin = self.root / rel
        data = origin.read_bytes()
        got = sha256_bytes(data)
        if got != self.pins[rel]:
            raise PinError(f"RCX {rel}: sha256 {got} != pinned {self.pins[rel]}")
        self.executed[rel] = got
        exec(compile(data, str(origin), "exec", dont_inherit=True), module.__dict__)


def unload_rcx(imp=None):
    if imp is not None and imp in sys.meta_path:
        sys.meta_path.remove(imp)
    for k in [k for k in sys.modules if k == "rcx" or k.startswith("rcx.")]:
        del sys.modules[k]


def load_rcx_hook_module(rcx_root=RCX_ROOT, pins=RCX_CODE_PINS):
    """(rcx.bw.b4_hook module, importer). Torch-free: b4_hook imports bwm_import, encode and space (numpy) only; torch
    and models.py are imported lazily by LearnedUnmet's constructor (GPU run only). All pins are checked up front.
    The registered stages pass rcx_root = the freeze's snapshot (edit 10)."""
    if any(k == "rcx" or k.startswith("rcx.") for k in sys.modules):
        raise PinError("an 'rcx' package is already imported in this process; refusing to mix")
    for rel, want in pins.items():
        check_pin(Path(rcx_root) / rel, want, f"RCX {rel}")
    imp = RCXPinnedImporter(rcx_root, pins)
    sys.meta_path.insert(0, imp)
    dwb = sys.dont_write_bytecode
    try:
        mod = importlib.import_module("rcx.bw.b4_hook")
        if mod.BWM_PLANNER_PIN != PLANNER_SHA256:
            raise PinError(f"b4_hook's planner pin {mod.BWM_PLANNER_PIN} != {PLANNER_SHA256}")
    except BaseException:
        unload_rcx(imp)
        raise
    finally:
        sys.dont_write_bytecode = dwb
    return mod, imp


def make_registered_hook(hook_mod, rcx_root):
    """The D-76a call, nothing else (GPU run and canary)."""
    return hook_mod.LearnedUnmet.from_file(str(Path(rcx_root) / GNN_REL), **HOOK_KWARGS)


def make_stub_hook_class(hook_mod):
    """DRY-RUN ONLY: LearnedUnmet with the GNN forward replaced by 0.0 and no torch import. Every other line of the
    pinned hook (consistency analysis, sanitiser, penalty/failsafe bands, hybrid term, clip, LRU cache, counters) runs
    unchanged, so the dry-run arm is 'misplaced_blocks + the invalid-state penalty' (arm a0 of the B4 draft)."""

    class StubModelUnmet(hook_mod.LearnedUnmet):
        def __init__(self, *, invalid_policy="fallback", hybrid_lambda=1.0, cache_size=hook_mod.DEFAULT_CACHE_SIZE):
            if invalid_policy not in hook_mod.INVALID_POLICIES:
                raise ValueError(invalid_policy)
            self._torch = self._M = self.model = None
            self.derived = False
            self.invalid_policy, self.hybrid_lambda, self.round_to = invalid_policy, float(hybrid_lambda), None
            self.cache_size, self._cache = int(cache_size), OrderedDict()
            self._registry, self.registry_conflicts = {}, 0
            self._prov = {"model_file": None, "model_sha256": None, "sidecar": None, "sidecar_sha256": None,
                          "config_source": "DRY-RUN stub: GNN forward replaced by 0.0"}
            self.reset_counters()

        def _forward(self, g) -> float:
            return 0.0

        def describe(self) -> dict:
            return {"hook": "b4_screen DRY-RUN StubModelUnmet(rcx.bw.b4_hook.LearnedUnmet)", **self._prov,
                    "invalid_policy": self.invalid_policy, "hybrid_lambda": self.hybrid_lambda,
                    "round_to": self.round_to,
                    "constants": {"PENALTY": hook_mod.PENALTY, "FAILSAFE": hook_mod.FAILSAFE},
                    "registry": {"goals": len(self._registry), "conflicts": self.registry_conflicts},
                    "counters": dict(self.counters), "reasons": dict(sorted(self.reasons.items())),
                    "bwm_planner_pin": hook_mod.BWM_PLANNER_PIN}

    return StubModelUnmet


def verify_hook(hook, rcx_root=RCX_ROOT) -> dict:
    """GPU run: the constructed hook is exactly the D-76a call on the pinned files (model and sidecar shas, config
    source, the four D-76a code shas, policy, lambda, no rounding, planner pin, the instance registry OFF (the D-76a
    call passes no instances=), truthiness)."""
    d = hook.describe()
    want = {"model_sha256": GNN_SHA256, "sidecar_sha256": GNN_SIDECAR_SHA256, "config_source": "sidecar json 'config'",
            "invalid_policy": HOOK_KWARGS["invalid_policy"], "hybrid_lambda": HOOK_KWARGS["hybrid_lambda"],
            "round_to": None, "bwm_planner_pin": PLANNER_SHA256}
    bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
    for rel in RCX_D76_FILES:
        if (d.get("code_sha256") or {}).get(rel) != RCX_CODE_PINS[rel]:
            bad[rel] = ((d.get("code_sha256") or {}).get(rel), RCX_CODE_PINS[rel])
    if not d.get("model_file") or Path(d["model_file"]).resolve() != (Path(rcx_root) / GNN_REL).resolve():
        bad["model_file"] = (d.get("model_file"), str(Path(rcx_root) / GNN_REL))
    if (d.get("registry") or {}).get("goals", None) != 0:
        bad["registry"] = (d.get("registry"), {"goals": 0})
    if not bool(hook):
        bad["truthiness"] = (False, True)
    if bad:
        raise PinError(f"hook differs from the registered D-76a call: {bad}")
    return d


def loaded_torch_version():
    """torch.__version__ of the torch imported in this process, or None (nothing is imported here)."""
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) if t is not None else None


def load_canary(canary_path, *, want_sha=None, tol=CANARY_TOL) -> dict:
    """Edit 8, torch-free part: the canary file's sha256 must be the registered one, its hook call the D-76a call, its
    tolerance no looser than `tol`, its pins the screen's, and it must carry probes and the expected counters after
    all probes (RCX low item 4: a file without them is refused). Returns the parsed record."""
    want_sha = CANARY_SHA256 if want_sha is None else want_sha
    if want_sha is None:
        raise PinError("the RCX canary golden file is not registered yet (CANARY_SHA256 placeholder): refusing")
    check_pin(canary_path, want_sha, "RCX canary")
    c = json.loads(Path(canary_path).read_text(encoding="utf-8"))
    call = c.get("hook_call") or {}
    if (not str(call.get("path", "")).replace("\\", "/").endswith(GNN_REL)
            or call.get("invalid_policy") != HOOK_KWARGS["invalid_policy"]
            or float(call.get("hybrid_lambda", float("nan"))) != HOOK_KWARGS["hybrid_lambda"]):
        raise CanaryError(f"canary hook_call {call} is not the registered D-76a call")
    if not float(c.get("tolerance_abs", 1.0)) <= tol:
        raise CanaryError(f"canary tolerance {c.get('tolerance_abs')} looser than the registered {tol}")
    pinned = {**RCX_CODE_PINS, GNN_REL: GNN_SHA256, GNN_SIDECAR_REL: GNN_SIDECAR_SHA256,
              E07_INSTANCES_REL: E07_INSTANCES_SHA256}
    off = {rel: (sha, pinned[rel]) for rel, sha in (c.get("pins_sha256") or {}).items() if rel in pinned
           and sha != pinned[rel]}
    if off:
        raise CanaryError(f"canary was made on other RCX files: {off}")
    if not c.get("probes"):
        raise CanaryError("canary has no probes")
    exp_c = c.get("expected_counters_after_all")
    if not isinstance(exp_c, dict) or not exp_c:
        raise CanaryError("canary has no expected_counters_after_all: the counters are part of the check (edit 8)")
    return c


def check_canary(make_hook, canary_path, *, want_sha=None, tol=CANARY_TOL, host_torch_version=None) -> dict:
    """Edit 8: a FRESH hook (the registered call) must reproduce RCX's golden values within `tol` on every probe of the
    sha-pinned canary (25 (state, goal) probes, 20 consistent and 5 flagged-inconsistent, from E07 test and E07b
    TEST-B instances, never seeds_fresh), in file order; its counters after all probes must equal the recorded ones.
    Raises CanaryError (the shard stops before planning; the screen is VOID). Report-only (D-80 follow-up): the torch
    version the canary was made with, the host's (`host_torch_version`, else the torch loaded in this process) and
    whether they match; a mismatch is never a failure, the binding check is |delta| <= tol per probe."""
    want_sha = CANARY_SHA256 if want_sha is None else want_sha
    c = load_canary(canary_path, want_sha=want_sha, tol=tol)
    probes, exp_c = c["probes"], c["expected_counters_after_all"]
    hook = make_hook()
    bad, worst, deltas = [], 0.0, {}
    for k, p in enumerate(probes):
        v = float(hook(frozenset(tuple(f) for f in p["state"]), frozenset(tuple(f) for f in p["goal"])))
        dv = abs(v - float(p["expected"]))
        deltas[str(p.get("id", k))] = dv
        worst = max(worst, dv) if dv == dv else float("inf")
        if not dv <= tol:                       # NaN-safe
            bad.append({"id": p.get("id"), "value": v, "expected": p["expected"]})
    counters = {k: int(v) for k, v in dict(hook.counters).items()}
    counters_ok = counters == {k: int(v) for k, v in exp_c.items()}
    host_tv = host_torch_version if host_torch_version is not None else loaded_torch_version()
    canary_tv = c.get("torch_version")
    out = {"file": CANARY_REL, "sha256": want_sha, "probes": len(probes),
           "consistent_probes": sum(bool(p.get("consistent")) for p in probes), "max_abs_delta": worst,
           "abs_delta_by_probe": deltas, "tolerance_abs": tol, "counters_after_all": counters,
           "counters_match": counters_ok, "failures": bad,
           "canary_torch_version": canary_tv, "canary_torch_threads": c.get("torch_threads"),
           "host_torch_version": host_tv,
           "torch_versions_match": None if canary_tv is None or host_tv is None else canary_tv == host_tv,
           "torch_version_rule": "report-only: a mismatch is flagged, never VOID (D-80 follow-up)"}
    if bad or not counters_ok:
        raise CanaryError(f"hook canary failed: {out}")
    return out


class HookMeter:
    """Truthy pass-through around the unmet hook (plan_beam binds `unmet or default`): returns the hook's value
    unchanged and counts, per goal, the calls, the penalty-band calls (value >= PENALTY: invalid state or critic error)
    and the failsafe-band calls, plus the seconds spent inside the hook. With `classify` (the hook's own is_consistent),
    each penalty-band call below FAILSAFE is split into invalid_state (the state is inconsistent) and critic_error (a
    consistent state that fell back: an encode, model or nonfinite error)."""

    def __init__(self, hook, penalty: float, failsafe: float, instances=(), classify=None):
        self.hook, self.penalty, self.failsafe, self.classify = hook, float(penalty), float(failsafe), classify
        self.owner = {}
        for it in instances:
            g = frozenset(it["goal"])
            self.owner[g] = it["id"] if g not in self.owner else self.owner[g] + "|" + it["id"]
        self.per = {}
        self.calls = self.penalty_calls = self.failsafe_calls = self.invalid_calls = self.error_calls = 0
        self.seconds = 0.0

    def __bool__(self):
        return True

    def __call__(self, state, goal):
        t = time.perf_counter()
        v = self.hook(state, goal)
        self.seconds += time.perf_counter() - t
        try:
            k = self.owner.get(frozenset(goal), "?")
        except Exception:
            k = "?"
        c = self.per.setdefault(k, [0, 0, 0, 0, 0])
        c[0] += 1
        self.calls += 1
        if v >= self.penalty:
            c[1] += 1
            self.penalty_calls += 1
            if v >= self.failsafe:
                c[2] += 1
                self.failsafe_calls += 1
            elif self.classify is not None:
                try:
                    consistent = bool(self.classify(state, goal)[0])
                except Exception:
                    consistent = True
                if consistent:
                    c[4] += 1
                    self.error_calls += 1
                else:
                    c[3] += 1
                    self.invalid_calls += 1
        return v

    def summary(self) -> dict:
        return {"calls": self.calls, "penalty_calls": self.penalty_calls, "failsafe_calls": self.failsafe_calls,
                "penalty_calls_invalid_state": self.invalid_calls, "penalty_calls_critic_error": self.error_calls,
                "penalty_split_classified": self.classify is not None, "seconds": round(self.seconds, 3),
                "per_instance": {k: {"calls": v[0], "penalty_calls": v[1], "failsafe_calls": v[2],
                                     "invalid_state": v[3], "critic_error": v[4]} for k, v in sorted(self.per.items())}}


# ----------------------------------------------------------------------------- dry-run models (no torch)

class FakeTok:
    """1 token per 4 characters (ceil), as tests/test_domain_planner.py's FakeTok."""
    CH = 4

    def n_tokens(self, text: str) -> int:
        return -(-len(text) // self.CH)


def _unit(*parts) -> float:
    return int(sha256_bytes("|".join(map(str, parts)).encode())[:13], 16) / 16 ** 13


class ScriptedModel:
    """DRY-RUN stand-in for the core with the 'fwd' and 'pa' adapters (deterministic; no torch).

    fwd: the exact delta answer (B.forward_target_delta) for legal actions. On illegal actions it answers "legal" with
    the action's nominal effects for a `false_legal` share of keys (the F-09 pattern; parse_forward_delta's support
    check rejects them) and the correct `{"add":[],"del":[],"legal":false}` otherwise. An `inconsistent` share of
    legal keys gets one extra wrong fact (exercises the hook's penalty band; 0 reproduces B2's audited F: no harmful
    errors).
    pa: K candidates from a teardown-and-build plan (valid from any state) with each action replaced by a random
    ground action with probability `noise`; candidate 0 (seed None) uses a seed derived from the prompt (greedy)."""

    def __init__(self, B, false_legal=0.25, inconsistent=0.01, noise=0.15):
        self.B, self.false_legal, self.inconsistent, self.noise = B, false_legal, inconsistent, noise
        self.adapter, self.adapter_calls, self.calls = None, Counter(), Counter()

    def set_adapter(self, name):
        self.adapter = name
        self.adapter_calls[name] += 1

    @staticmethod
    def _user(prompt: str) -> dict:
        return json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])

    @staticmethod
    def _cut(text: str, max_new: int, tok) -> dict:
        n = tok.n_tokens(text) + 1  # + EOS
        if n > max_new:
            return {"text": text[: max_new * tok.CH], "n_tokens": max_new}
        return {"text": text, "n_tokens": n}

    def _nominal(self, s, a):
        op = a[0]
        if op == "pick-up":
            return {("holding", a[1])}, {("clear", a[1]), ("ontable", a[1]), ("handempty",)}
        if op == "put-down":
            return {("clear", a[1]), ("handempty",), ("ontable", a[1])}, {("holding", a[1])}
        if op == "stack":
            return {("handempty",), ("clear", a[1]), ("on", a[1], a[2])}, {("clear", a[2]), ("holding", a[1])}
        return {("holding", a[1]), ("clear", a[2])}, {("on", a[1], a[2]), ("clear", a[1]), ("handempty",)}

    def forward_text(self, prompt, max_new, tok):
        if self.adapter != "fwd":
            raise RuntimeError(f"forward call with adapter {self.adapter!r}")
        self.calls["fwd"] += 1
        B = self.B
        d = self._user(prompt)
        s, a = B.state_from_json(d["state"]), tuple(d["action"].split())
        ns, legal = B.step(s, a)
        if legal:
            add, dele = set(ns - s), set(s - ns)
            if _unit("inc", prompt) < self.inconsistent:
                on = sorted(f for f in ns if f[0] == "on")
                add.add(("ontable", on[0][1]) if on else ("clear", "zz"))
            text = B.canonical({"add": sorted(B.fact_str(f) for f in add), "del": sorted(B.fact_str(f) for f in dele),
                                "legal": True})
        elif _unit("fl", prompt) < self.false_legal:
            add, dele = self._nominal(s, a)
            text = B.canonical({"add": sorted(B.fact_str(f) for f in add), "del": sorted(B.fact_str(f) for f in dele),
                                "legal": True})
        else:
            text = B.forward_target_delta(s, s, False)
        return self._cut(text, max_new, tok)

    def scripted_plan(self, s, goal):
        B, plan, cur = self.B, [], s

        def do(a):
            nonlocal cur
            ns, ok = B.step(cur, a)
            if ok:
                plan.append(a)
                cur = ns

        held = sorted(f[1] for f in cur if f[0] == "holding")
        if held:
            do(("put-down", held[0]))
        for _ in range(200):
            tops = sorted(f for f in cur if f[0] == "on" and ("clear", f[1]) in cur)
            if not tops:
                break
            do(("unstack", tops[0][1], tops[0][2]))
            do(("put-down", tops[0][1]))
        below = {f[1]: f[2] for f in goal if f[0] == "on"}
        placed, pending = set(), sorted(below)
        while pending:
            ready = [x for x in pending if below[x] not in below or below[x] in placed]
            if not ready:
                break
            x = ready[0]
            do(("pick-up", x))
            do(("stack", x, below[x]))
            placed.add(x)
            pending.remove(x)
        return plan

    def proposer_text(self, prompt, seed, max_new, tok):
        if self.adapter != "pa":
            raise RuntimeError(f"proposer call with adapter {self.adapter!r}")
        self.calls["pa"] += 1
        B = self.B
        d = self._user(prompt)
        s = B.state_from_json(d["state"])
        goal = frozenset(("on", x, y) for x, y in d["goal"])
        objs = B.blocks_of(s)
        rng = random.Random(seed if seed is not None else int(sha256_bytes(prompt.encode())[:15], 16))
        acts = B.ground_actions(objs)
        plan = [rng.choice(acts) if rng.random() < self.noise else a for a in self.scripted_plan(s, goal)]
        return self._cut(B.target_actions(plan or [rng.choice(acts)]), max_new, tok)


def dry_instances(B, sizes=DRY_SIZES, seed=DRY_SEED) -> list:
    rng, out = random.Random(seed), []
    for i, n in enumerate(sizes):
        g = None
        while g is None:
            g = B.generate(rng, n, full_goal_frac=1.0)
        objs, init, goal = g
        out.append({"id": f"dry/{i}", "objs": objs, "init": init, "goal": goal, "path": None})
    return out


# ----------------------------------------------------------------------------- one arm (shared by dry-run and GPU)

REQUIRED_ARGS = {"proposer": "lm", "mode": "actions", "fwd": "lm", "fwd_fmt": "delta", "planner": "beam"}


def check_args(args: dict):
    bad = {k: (args.get(k), v) for k, v in REQUIRED_ARGS.items() if args.get(k) != v}
    miss = [k for k in ("K", "W", "E", "R", "M", "P", "max_depth", "eval_seed") if k not in args]
    if bad or miss:
        raise ValueError(f"not an S-beam freeze args dict: {bad} missing {miss}")


def load_json_set(path: Path, prefix: str, want_sha: str | None = None) -> list:
    """Instances exactly as blocks_eval.py's _json_set builds them (id '<prefix>/<index>')."""
    if want_sha is not None:
        check_pin(path, want_sha, str(path))
    return [{"id": f"{prefix}/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"])), "path": None}
            for i, r in enumerate(json.loads(path.read_text()))]


def load_screen_instances(root=ROOT) -> list:
    inst = load_json_set(Path(root) / SEEDS_FRESH_REL, "seeds_fresh", SEEDS_FRESH_SHA256)
    if len(inst) != N_FRESH:
        raise PinError(f"seeds_fresh has {len(inst)} instances, expected {N_FRESH}")
    return inst


def shard_of(iid: str, k: int = SHARDS) -> int:
    return int(iid.rsplit("/", 1)[1]) % k


def run_arm(BP, B, model, tok, inst, args, dst: Path, *, unmet=None, val=True) -> tuple:
    """plan_beam with blocks_eval.py's S-beam construction for `args`, plus `unmet`. Returns (summary, rows); rows are
    BP.score rows (+ 'val', + 'expansions' per instance)."""
    check_args(args)
    dst.mkdir(parents=True, exist_ok=True)
    prop = BP.LMProposerBW(model, tok, "pa", args["mode"], K=args["K"], max_new=args["M"] or None,
                           prefix_ok=bool(args["P"]), max_batch_tokens=PROP_BATCH_TOKENS)
    fwd = BP.LMForwardBW(model, tok, "fwd", fmt=args["fwd_fmt"], cache=dst / "fcache.jsonl",
                         max_batch_tokens=F_BATCH_TOKENS)
    t0 = time.time()
    with open(dst / "progress.txt", "a", encoding="utf-8") as prog:
        def plog(m):
            prog.write(time.strftime("%H:%M:%S ") + m + "\n")
            prog.flush()
        eps = BP.plan_beam(inst, prop, fwd, eval_seed=args["eval_seed"], width=args["W"], max_expansions=args["E"],
                           max_depth=args["max_depth"], max_rollout=args["R"], log=plog, unmet=unmet)
    expansions = {e["id"]: e["expansions"] for e in eps}
    summ, rows = BP.score(inst, eps)
    summ["wall_s"] = round(time.time() - t0, 1)
    summ["f_model_queries"] = getattr(fwd, "model_queries", None)
    summ["f_queries_per_instance"] = round((summ["f_model_queries"] or 0) / max(1, len(inst)), 1)
    summ["proposer_tokens_per_instance"] = round(sum(e["gen_tokens"] for e in eps) / max(1, len(inst)), 1)
    summ["expansions_per_instance"] = round(sum(expansions.values()) / max(1, len(inst)), 2)
    for r in rows:
        r["expansions"] = expansions[r["id"]]
    if val:
        dis = 0
        for it, r in zip(inst, rows):
            if it.get("path") is None:
                p = dst / "tmp_problem.pddl"
                p.write_text(B.write_problem(it["objs"], it["init"], it["goal"]))
            else:
                p = Path(it["path"])
            r["val"] = B.val_check(p, [tuple(x.split()) for x in r["plan"]], domain_path=B.DOMAIN_PDDL)
            dis += (r["val"] != r["success"])
        summ["val_success"] = sum(r["val"] for r in rows) / max(1, len(rows))
        summ["val_disagreements"] = dis
    by = {}
    for it, r in zip(inst, rows):
        by.setdefault(len(it["objs"]), []).append(r["success"])
    summ["success_by_blocks"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())}
    return summ, rows


def write_outputs(dst: Path, summary: dict, rows: list, extra: dict | None = None):
    """plans.jsonl first, summary.json last (summary.json is the completion marker)."""
    with open(dst / "plans.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (dst / "summary.json").write_text(json.dumps({**(extra or {}), "summary": summary}, indent=1))


def _load_script(name: str):
    """Load a BWM script as a private module (read-only; no bytecode written)."""
    spec = importlib.util.spec_from_file_location(f"_b4s_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dwb
    return mod


def merge_hooks(out: Path, k: int, *, require: bool = True) -> dict:
    """The per-shard hook records summed (meter, counters, reasons, canary, executed RCX modules)."""
    hook = {"calls": 0, "penalty_calls": 0, "failsafe_calls": 0, "penalty_calls_invalid_state": 0,
            "penalty_calls_critic_error": 0, "seconds": 0.0, "per_instance": {}, "counters": Counter(),
            "reasons": Counter(), "shards": [], "canary": {}, "rcx_modules_executed": {}, "torch_version": [],
            "model_evals_by_shard": {}}
    for i in range(k):
        f = out / f"shard_{i}of{k}" / "hook.json"
        if not f.exists():
            if require:
                raise PinError(f"{f} missing: every hook shard writes hook.json before its summary.json")
            continue
        h = json.loads(f.read_text())
        m = h["meter"]
        for key in ("calls", "penalty_calls", "failsafe_calls", "penalty_calls_invalid_state",
                    "penalty_calls_critic_error", "seconds"):
            hook[key] += m.get(key, 0)
        hook["per_instance"].update(m["per_instance"])
        hook["counters"].update(h["describe_end"].get("counters", {}))
        hook["reasons"].update(h["describe_end"].get("reasons", {}))
        hook["model_evals_by_shard"][f"shard_{i}of{k}"] = int((h["describe_end"].get("counters") or {})
                                                                 .get("model_evals", 0))
        hook["shards"].append(f"shard_{i}of{k}")
        hook["canary"][f"shard_{i}of{k}"] = h.get("canary")
        hook["rcx_modules_executed"].update(h.get("rcx_modules_executed") or {})
        if h.get("torch_version") not in hook["torch_version"]:
            hook["torch_version"].append(h.get("torch_version"))
    hook["counters"], hook["reasons"] = dict(hook["counters"]), dict(hook["reasons"])
    hook["seconds"] = round(hook["seconds"], 3)
    return hook


def merge_shards(out: Path, k: int, *, with_hook: bool = True):
    """hook_merged.json FIRST, then bw_run_tests.merge (the function item 7 uses for R0), which writes plans.jsonl and
    then summary.json, the arm's completion marker."""
    hook = None
    if with_hook:
        hook = merge_hooks(out, k)
        (out / "hook_merged.json").write_text(json.dumps(hook, indent=1))
    _load_script("bw_run_tests").merge(out, k)
    return hook


# ----------------------------------------------------------------------------- analysis (pure; dry-run and analyze)

def _ok(r) -> bool:
    return bool(r.get("val", r["success"]))


def _id_key(i: str):
    a, _, b = i.rpartition("/")
    return (a, int(b)) if b.isdigit() else (a, 0, b)


def read_cache(path: Path) -> tuple:
    """{(state, action): text} as LMForwardBW loads it (torn lines skipped; last line wins) + counts."""
    out, torn, redef = {}, 0, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            torn += 1
            continue
        k = (frozenset(tuple(f) for f in r["s"]), tuple(r["a"]))
        if k in out and out[k] != r["text"]:
            redef += 1
        out[k] = r["text"]
    return out, {"lines_skipped": torn, "keys_redefined_with_new_text": redef}


def union_caches(paths) -> tuple:
    texts, meta = {}, {}
    for p in paths:
        c, m = read_cache(p)
        meta[str(p)] = {**m, "keys": len(c), "sha256": sha256_file(p)}
        for k, t in c.items():
            texts.setdefault(k, set()).add(t)
    return texts, meta


def certificate_c_eq_cpen(cache_paths, B, hook_mod) -> dict:
    """c+pen == c on a run iff no F-accepted imagined state of the run is inconsistent (the penalty never fires).
    Checked conservatively over every key of the run's caches with the hook's own is_consistent, passing the parent
    state's blocks as objs (verifier fix): the hook is called with the goal, so it flags a goal block that vanished from
    the imagined state (no_position); the cache does not record the goal, and without objs such a state would pass.
    With objs = the parent's blocks every vanished block is caught and a new name is flagged (unknown_block), which is
    at least as strict as the hook for any goal over the instance's blocks."""
    texts, meta = union_caches(cache_paths)
    acc = bad = 0
    reasons = Counter()
    for (s, a), ts in texts.items():
        for t in ts:
            p = B.parse_forward_delta(t, s)
            if p is None or not p[1]:
                continue
            acc += 1
            ok, why = hook_mod.is_consistent(p[0], None, B.blocks_of(s))
            if not ok:
                bad += 1
                reasons.update(why)
    return {"accepted_predictions": acc, "inconsistent": bad, "holds": bad == 0, "reasons": dict(reasons),
            "caches": meta}


def unpenalised_certificate(arm_caches_by_shard: dict, instances, B, hook_mod, k: int = SHARDS) -> dict:
    """REPORT-ONLY (RCX's known hook limitation): with the registry off (the D-76a call), is_consistent(state, goal)
    sees only the state's and the goal's blocks, so a NON-goal block that vanished from an imagined state is not
    flagged (b4_hook.py:206-216) and such a state is scored by the model instead of the penalty. For every F-accepted
    prediction in the arm's caches: strict = is_consistent(pred, None, the instance's blocks) (the parent's blocks when
    no instance of the shard contains them); penalised = not is_consistent(pred, goal) as the hook calls it. Keys whose
    blocks fit several instances of the shard count as penalised or unpenalised only if every candidate goal agrees;
    keys whose parent holds a name that no instance of the shard has (the goal is unknown) are undetermined."""
    by_shard = {}
    for it in instances:
        by_shard.setdefault(shard_of(it["id"], k), []).append(it)
    acc = strict_bad = penalised = unpen = ambiguous = unmatched = 0
    reasons = Counter()
    for si, path in sorted(arm_caches_by_shard.items()):
        if not Path(path).exists():          # a shard with no instance never queried F (dry run only)
            continue
        cache, _ = read_cache(path)
        cands_all = by_shard.get(si, [])
        for (s, a), t in cache.items():
            p = B.parse_forward_delta(t, s)
            if p is None or not p[1]:
                continue
            acc += 1
            parent = set(B.blocks_of(s))
            cands = [it for it in cands_all if parent <= set(it["objs"])]
            exact = [it for it in cands if set(it["objs"]) == parent]
            cands = exact or cands
            if not cands:
                unmatched += 1
                objs = sorted(parent)
            elif len(cands) > 1:
                ambiguous += 1
                objs = sorted(parent)
            else:
                objs = list(cands[0]["objs"])
            ok, why = hook_mod.is_consistent(p[0], None, objs)
            if ok:
                continue
            strict_bad += 1
            if not cands:                    # the parent holds a name no instance has: the goal is unknown
                continue
            pen = [not hook_mod.is_consistent(p[0], it["goal"])[0] for it in cands]
            if all(pen):
                penalised += 1
            elif not any(pen):
                unpen += 1
                reasons.update(why)
    return {"what": "report-only: inconsistent F-accepted keys of the arm that the hook (registry off) did not "
                    "penalise", "accepted_predictions": acc, "strict_inconsistent": strict_bad,
            "penalised_by_hook": penalised, "unpenalised": unpen,
            "undetermined": strict_bad - penalised - unpen, "keys_matching_several_instances": ambiguous,
            "keys_matching_no_instance": unmatched, "unpenalised_reasons": dict(reasons), "registry": "off"}


def cross_cache(ctrl_paths, arm_paths) -> dict:
    """F is frozen and memoised: a key answered in both runs must have the same text (a check of D-48c/D-50)."""
    c, _ = union_caches(ctrl_paths)
    a, _ = union_caches(arm_paths)
    common = set(c) & set(a)
    differ = [k for k in common if c[k] != a[k] or len(c[k]) > 1]
    return {"control_keys": len(c), "arm_keys": len(a), "common_keys": len(common),
            "common_same_text": len(common) - len(differ), "common_different_text": len(differ),
            "intra_run_conflicts": {"control": sum(len(v) > 1 for v in c.values()),
                                    "arm": sum(len(v) > 1 for v in a.values())}}


def unmet_after(B, it, plan_strs):
    ex = B.execute(it["init"], it["goal"], [tuple(x.split()) for x in plan_strs])
    return len(it["goal"] - ex["final"]) if ex["valid"] else None


def round1_identity(ctrl: dict, arm: dict, ids) -> dict:
    """Edit 7: round 0 is identical in both arms by construction (same initial state, proposals, F answers; solutions
    are detected without the hook), so an instance solved (verified) in round 1 by EITHER arm must be solved in round 1
    by both, with the same plan and the same proposer tokens."""
    r1 = lambda r: bool(r["verified"]) and r["rounds"] == 1
    bad = []
    for i in ids:
        c, a = ctrl[i], arm[i]
        if (r1(c) or r1(a)) and not (r1(c) and r1(a) and c["plan"] == a["plan"]
                                     and c["gen_tokens"] == a["gen_tokens"]):
            bad.append(i)
    return {"solved_in_round_1": {"control": sum(r1(ctrl[i]) for i in ids), "arm": sum(r1(arm[i]) for i in ids)},
            "mismatched_ids": bad, "holds": not bad}


def arm_block(rows: dict, ids: list, max_r: int = 12) -> dict:
    rs = [rows[i] for i in ids]
    k, n = sum(map(_ok, rs)), len(rs)
    tax = Counter("solved" if _ok(r) else (r.get("fail") or ("verified_but_invalid" if r["verified"] else "other"))
                  for r in rs)
    tax["verified_but_invalid"] = sum(r["verified"] and not _ok(r) for r in rs)
    by = {}
    for r in rs:
        by.setdefault(r.get("n_blocks"), []).append(_ok(r))
    out = {"success": f"{k}/{n}", "rate": round(k / n, 4) if n else None, "ci95": clopper_pearson(k, n),
           "failure_taxonomy": dict(sorted(tax.items())),
           "verified": sum(r["verified"] for r in rs),
           "verified_precision": round(sum(r["verified"] and _ok(r) for r in rs)
                                       / max(1, sum(r["verified"] for r in rs)), 4),
           "unverified_val_successes": sum(_ok(r) and not r["verified"] for r in rs),
           "val_disagreements": sum(("val" in r) and (bool(r["val"]) != bool(r["success"])) for r in rs),
           "mean_gen_tokens": round(sum(r["gen_tokens"] for r in rs) / max(1, n), 1),
           "mean_rounds": round(sum(r["rounds"] for r in rs) / max(1, n), 2),
           "anytime_success_within_rounds": {str(t): sum(_ok(r) and r["rounds"] <= t for r in rs)
                                             for t in range(1, max_r + 1)},
           "token_budget_success_approx": {str(T): sum(_ok(r) and r["gen_tokens"] <= T for r in rs)
                                           for T in (2000, 4000, 6000, 8000, 10000, 12000, 16000)},
           "success_by_blocks": {str(b): f"{sum(v)}/{len(v)}"
                                 for b, v in sorted(by.items(), key=lambda x: (x[0] is None, x[0]))}}
    if all("expansions" in r for r in rs):
        out["mean_expansions"] = round(sum(r["expansions"] for r in rs) / max(1, n), 2)
    return out


def stalled_block(B, by: dict, fail_rows: dict, other_rows: dict, ids, other_label: str) -> dict:
    """Descriptive, on the TRUE state: unmet goal facts in the executor's state after each failure's submitted prefix
    (an invalid prefix is counted separately), the stalled-at-1 count, and how often the other arm solves them."""
    fails = [i for i in ids if not _ok(fail_rows[i])]
    dist, stalled1 = Counter(), []
    for i in fails:
        u = unmet_after(B, by[i], fail_rows[i]["plan"])
        dist["invalid_prefix" if u is None else str(u)] += 1
        if u == 1:
            stalled1.append(i)
    rest = [i for i in fails if i not in stalled1]
    return {"note": "descriptive; computed on the executor's true state after the submitted prefix",
            "failures": len(fails), "distribution": dict(sorted(dist.items())), "stalled_at_1": len(stalled1),
            f"{other_label}_solves_among_stalled_at_1": sum(_ok(other_rows[i]) for i in stalled1),
            "other_failures": len(rest), f"{other_label}_solves_among_other_failures": sum(_ok(other_rows[i])
                                                                                          for i in rest),
            "ids_stalled_at_1": stalled1}


def hook_report(hook: dict, dec: dict | None, penalty_share_max: float) -> dict:
    calls = hook.get("calls", 0)
    share = hook.get("penalty_calls", 0) / calls if calls else 0.0
    attributable = share <= penalty_share_max
    c = hook.get("counters", {})
    out = {**{k: v for k, v in hook.items() if k != "per_instance"},
           "penalty_band": "invalid state or critic error (value >= PENALTY = 1000)",
           "penalty_share_calls": round(share, 6), "penalty_share_max": penalty_share_max,
           "penalty_split_calls": {"invalid_state": hook.get("penalty_calls_invalid_state"),
                                   "critic_error": hook.get("penalty_calls_critic_error"),
                                   "failsafe": hook.get("failsafe_calls")},
           "go_credited_to_critic": bool(dec is not None and dec["go"] and attributable),
           "attribution_rule_fired": not attributable}
    ev = c.get("evaluations")
    if ev:
        out["penalty_split_evaluations"] = {"invalid_states": c.get("invalid_states", 0),
                                            "critic_errors": sum(c.get(x, 0) for x in ("encode_errors", "model_errors",
                                                                                       "nonfinite")),
                                            "failsafe_incl_analysis_errors": c.get("failsafe", 0)}
        out["penalty_share_evaluations"] = round((c.get("fallbacks", 0) + c.get("analysis_errors", 0)) / ev, 6)
    return out


def analyze_pair(ctrl_rows, arm_rows, *, n_expected: int, drop_ids=(), hook=None, instances=None, B=None,
                 hook_mod=None, ctrl_caches=(), arm_caches=(), arm_caches_by_shard=None, labels=("R0", "a"),
                 go_min_net: int = GO_MIN_NET, penalty_share_max: float = PENALTY_SHARE_MAX,
                 fault_status: str = "INSTRUMENT_FAULT") -> dict:
    """The registered screen analysis (prereg sections 5-6) on two plans.jsonl row lists."""
    c = {r["id"]: r for r in ctrl_rows}
    a = {r["id"]: r for r in arm_rows}
    if len(c) != len(ctrl_rows) or len(a) != len(arm_rows):
        raise ValueError("duplicate instance ids in a plans file")
    if set(c) != set(a):
        raise ValueError(f"the two runs cover different instances: {sorted(set(c) ^ set(a))[:5]}")
    if len(c) != n_expected:
        raise ValueError(f"expected {n_expected} paired instances, got {len(c)}")
    unknown = set(drop_ids) - set(c)
    if unknown:
        raise ValueError(f"pre-check drops unknown ids {sorted(unknown)}")
    ids = sorted(set(c) - set(drop_ids), key=_id_key)
    g = sum(_ok(a[i]) and not _ok(c[i]) for i in ids)
    l = sum(_ok(c[i]) and not _ok(a[i]) for i in ids)
    both = sum(_ok(a[i]) and _ok(c[i]) for i in ids)
    r1 = round1_identity(c, a, sorted(c, key=_id_key))
    dec = futility_decision(g, l, go_min_net) if r1["holds"] else None
    status = dec["decision"] if dec else fault_status
    rep = {"status": status, "n_paired": len(ids), "dropped_by_precheck": sorted(drop_ids, key=_id_key),
           "round1_identity": r1,
           "paired": {f"{labels[1]}_only": g, f"{labels[0]}_only": l, "both": both, "neither": len(ids) - g - l - both},
           "decision": dec, "rule": RULE_TEXT}
    wording = WORDING[status]
    if hook is not None:
        rep["hook"] = hook_report(hook, dec, penalty_share_max)
        if dec is not None and dec["go"] and rep["hook"]["attribution_rule_fired"]:
            wording += ATTRIBUTION_NOTE
    rep["wording"] = wording
    rep["arms"] = {labels[0]: arm_block(c, ids), labels[1]: arm_block(a, ids)}
    strata = {}
    for name, lo, hi in (("6-10 blocks", 0, 10), ("11-15 blocks", 11, 99)):
        sid = [i for i in ids if lo <= (c[i].get("n_blocks") or 0) <= hi]
        sg = sum(_ok(a[i]) and not _ok(c[i]) for i in sid)
        sl = sum(_ok(c[i]) and not _ok(a[i]) for i in sid)
        strata[name] = {"n": len(sid), labels[0]: sum(_ok(c[i]) for i in sid), labels[1]: sum(_ok(a[i]) for i in sid),
                        f"{labels[1]}_only": sg, f"{labels[0]}_only": sl,
                        "p_mcnemar_two_sided": mcnemar_two_sided(sg, sl)}
    rep["strata_report_only"] = strata
    if instances is not None and B is not None:
        by = {it["id"]: it for it in instances}
        rep["control_failures_unmet_at_submitted_prefix"] = stalled_block(B, by, c, a, ids, labels[1])
        rep["arm_failures_unmet_at_submitted_prefix"] = stalled_block(B, by, a, c, ids, labels[0])
    if ctrl_caches and hook_mod is not None and B is not None:
        rep["certificate_c_eq_cpen_on_control"] = certificate_c_eq_cpen(ctrl_caches, B, hook_mod)
    if arm_caches_by_shard and instances is not None and hook_mod is not None and B is not None:
        rep["arm_unpenalised_inconsistent_report_only"] = unpenalised_certificate(arm_caches_by_shard, instances, B,
                                                                                  hook_mod)
    if ctrl_caches and arm_caches:
        rep["f_cache_cross_check"] = cross_cache(ctrl_caches, arm_caches)
    return rep


def void_reasons(hook: dict) -> list:
    """Edit 8: the screen is VOID (not NO-GO) on any hook model/encode/nonfinite/failsafe error, model_evals = 0 (in
    the merge or in any hook shard: RCX low item 7) or a shard without a passed canary. A torch-version mismatch
    between the canary and the host is report-only and never a VOID reason (D-80 follow-up)."""
    c = hook.get("counters", {})
    out = [f"hook {k} = {c.get(k)}" for k in VOID_COUNTERS if c.get(k, 0) > 0]
    if hook.get("failsafe_calls", 0) > 0 and not c.get("failsafe", 0):
        out.append(f"failsafe-band calls = {hook['failsafe_calls']}")
    if c.get("model_evals", 0) == 0:
        out.append("hook model_evals = 0")
    for sh, n in sorted((hook.get("model_evals_by_shard") or {}).items()):
        if not n > 0:
            out.append(f"{sh}: hook model_evals = {n}")
    for sh, can in sorted((hook.get("canary") or {}).items()):
        if not isinstance(can, dict) or can.get("failures") or not can.get("counters_match"):
            out.append(f"{sh}: no passed canary")
    return out


def canary_report(hook: dict) -> dict:
    """Report-only (D-80 follow-up): per hook shard, the canary's max |delta| and the torch versions of the canary and
    of the host process. A mismatch is flagged descriptively; it is never a VOID reason (the binding check is the
    per-probe |delta| <= 1e-5 in the GPU process)."""
    per = {}
    for sh, can in sorted((hook.get("canary") or {}).items()):
        can = can if isinstance(can, dict) else {}
        per[sh] = {"max_abs_delta": can.get("max_abs_delta"), "canary_torch_version": can.get("canary_torch_version"),
                   "host_torch_version": can.get("host_torch_version"),
                   "torch_versions_match": can.get("torch_versions_match")}
    mism = [sh for sh, v in per.items() if v["torch_versions_match"] is False]
    return {"what": "report-only: canary max |delta| per hook shard and the canary's and the host's torch versions",
            "shards": per, "torch_version_mismatch_shards": mism,
            "torch_version_flag": (f"the host's torch differs from the canary's on {mism}; descriptive only, not VOID: "
                                   "every probe still had to match within the registered tolerance") if mism else None}


# ----------------------------------------------------------------------------- pre-check (prereg 4.3)

def goal_chain_key(objs, goal):
    """RCX data.goal_key for on-only goals made of disjoint chains covering every block ('chains:n:h1,h2,...', the
    exact renaming class); None otherwise. Re-implemented (RCX's version calls BI.load(), which refuses inside a BWM
    process); agreement with the goal_key stored in RCX's instance file is checked on every row."""
    goal = [tuple(f) for f in goal]
    if not goal or any(f[0] != "on" for f in goal):
        return None
    on = {f[1]: f[2] for f in goal}
    tops, bottoms = [f[1] for f in goal], [f[2] for f in goal]
    if len(set(tops)) != len(tops) or len(set(bottoms)) != len(bottoms):
        return None
    below, hs = set(on.values()), []
    for top in (b for b in objs if b not in below):
        h, x = 1, top
        while x in on and h <= len(objs):
            x, h = on[x], h + 1
        hs.append(h)
    if sum(hs) != len(objs):
        return None
    return f"chains:{len(objs)}:" + ",".join(map(str, sorted(hs)))


def precheck(seeds_path=None, e07_path=None, *, check_shas=True) -> dict:
    seeds_path = Path(seeds_path or ROOT / SEEDS_FRESH_REL)
    e07_path = Path(e07_path or RCX_ROOT / E07_INSTANCES_REL)
    if check_shas:
        check_pin(seeds_path, SEEDS_FRESH_SHA256, SEEDS_FRESH_REL)
        check_pin(e07_path, E07_INSTANCES_SHA256, E07_INSTANCES_REL)
    e07 = json.loads(e07_path.read_text())
    agree = sum(goal_chain_key(r["objs"], r["goal"]) == r["goal_key"] for r in e07)
    if agree != len(e07):
        raise PinError(f"goal_chain_key disagrees with RCX's stored goal_key on {len(e07) - agree} E07 rows")
    trained = {r["goal_key"] for r in e07 if r["split"] in ("train", "val")}
    tested = {r["goal_key"] for r in e07 if r["split"] == "test"}
    sf = json.loads(seeds_path.read_text())
    rows = []
    for i, r in enumerate(sf):
        k = goal_chain_key(r["objs"], r["goal"])
        rows.append({"id": f"seeds_fresh/{i}", "n": len(r["objs"]), "goal_key": k, "in_gnn_train_or_val": k in trained,
                     "in_e07_test_classes": k in tested})
    drop = [x["id"] for x in rows if x["in_gnn_train_or_val"]]
    return {"rule": "drop a seeds_fresh instance from the screen's paired analysis iff its goal renaming class "
                    "(RCX goal_key) is a goal class of the GNN's training or selection (val) data",
            "seeds_fresh_sha256": sha256_file(seeds_path), "e07_instances_sha256": sha256_file(e07_path),
            "e07_rows_goal_key_agreement": f"{agree}/{len(e07)}",
            "gnn_train_val_goal_classes": len(trained),
            "gnn_train_val_class_sizes": dict(sorted(Counter(int(k.split(":")[1]) for k in trained).items())),
            "seeds_fresh_unkeyed": sum(x["goal_key"] is None for x in rows),
            "seeds_fresh_n_le_8": [x for x in rows if x["n"] <= 8],
            "seeds_fresh_in_e07_test_classes": [x["id"] for x in rows if x["in_e07_test_classes"]],
            "drop_ids": drop, "n_after_drop": len(sf) - len(drop)}


# ----------------------------------------------------------------------------- registration and finalisation

REGISTERED_ROWS = {"runner": "Runner sha256", "tests": "Tests sha256"}
ADDENDUM_PREREG_ROW = "Screen prereg (final) sha256"   # the addendum's header row naming the final screen prereg
FINAL_EDITABLE_ROWS = ("Status", ADDENDUM_PREREG_ROW)


def registered_shas(text: str) -> dict:
    """Edit 2: the runner and test shas registered in the screen prereg's header rows '| Runner sha256 | `<hex>` |'."""
    out = {}
    for key, label in REGISTERED_ROWS.items():
        hits = re.findall(rf"^\|\s*{re.escape(label)}\s*\|\s*`([0-9a-f]{{64}})`", text, flags=re.M)
        if len(hits) != 1:
            raise PinError(f"the prereg must register exactly one '{label}' header row with a full sha256 "
                           f"(found {len(hits)})")
        out[key] = hits[0]
    return out


def check_registered_runner(prereg_text: str, addendum_text: str, root=ROOT) -> dict:
    """Edit 2: refuse unless the runner and tests on disk are the registered bytes (both texts name the runner)."""
    reg = registered_shas(prereg_text)
    if not (Path(root) / TESTS_REL).exists():
        raise PinError(f"{Path(root) / TESTS_REL} missing")
    runner, tests = sha256_file(Path(__file__)), sha256_file(Path(root) / TESTS_REL)
    if reg["runner"] != runner:
        raise PinError(f"scripts/b4_screen.py on disk {runner} != registered {reg['runner']}")
    if reg["tests"] != tests:
        raise PinError(f"{TESTS_REL} on disk {tests} != registered {reg['tests']}")
    if runner not in addendum_text:
        raise PinError(f"the addendum does not name the registered runner sha {runner}")
    return {"runner_sha256": runner, "tests_sha256": tests}


def addendum_prereg_sha(addendum_text: str) -> str:
    """RCX low item 2: the sha256 that the addendum's header row '| Screen prereg (final) sha256 | `<hex>` |' names."""
    hits = re.findall(rf"^\|\s*{re.escape(ADDENDUM_PREREG_ROW)}\s*\|\s*`([0-9a-f]{{64}})`", addendum_text, flags=re.M)
    if len(hits) != 1:
        raise PinError(f"the addendum must name the final screen prereg in exactly one '{ADDENDUM_PREREG_ROW}' header "
                       f"row as a full sha256 in backticks (found {len(hits)})")
    return hits[0]


def finalisation_diff(draft: str, final: str) -> dict:
    """Edit 11: finalising a co-signed text may change only its title line, and only by dropping "(DRAFT)", and the
    header-table rows named in FINAL_EDITABLE_ROWS (Status; the addendum's final screen-prereg sha). RCX re-verifies
    this diff."""
    d, f = draft.splitlines(), final.splitlines()
    if len(d) != len(f):
        return {"ok": False, "why": f"line count {len(d)} -> {len(f)}", "changed_lines": [], "outside_header": []}
    head_end = next((i for i, x in enumerate(d) if x.startswith("## ")), len(d))
    pat = re.compile(r"^\|\s*(" + "|".join(re.escape(x) for x in FINAL_EDITABLE_ROWS) + r")\s*\|")
    changed = [i for i in range(len(d)) if d[i] != f[i]]
    title_ok = bool(d) and f[0] in (d[0].replace(" (DRAFT)", "", 1), d[0].replace("(DRAFT)", "", 1))  # DRAFT only
    bad = [i + 1 for i in changed if not ((i == 0 and title_ok) or (i < head_end and pat.match(d[i])
                                                                     and pat.match(f[i])))]
    return {"ok": not bad, "changed_lines": [i + 1 for i in changed], "outside_header": bad}


# ----------------------------------------------------------------------------- freeze (CPU; once)

def library_versions() -> dict:
    """Installed versions from package metadata only (none of the packages is imported)."""
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for p in VERSION_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def nvidia_driver_version():
    """The NVIDIA driver version through NVML (ctypes; no nvidia-smi process, see bwmn/nvml.py); None if unavailable."""
    try:
        import ctypes
        lib = ctypes.WinDLL("nvml.dll") if hasattr(ctypes, "WinDLL") else ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            return None
        try:
            buf = ctypes.create_string_buffer(96)
            return buf.value.decode() if lib.nvmlSystemGetDriverVersion(buf, 96) == 0 else None
        finally:
            lib.nvmlShutdown()
    except Exception:
        return None


def runtime_versions() -> dict:
    """Edit 6: recorded at the screen freeze (which precedes R0's run in the same bundle); the arm run refuses on any
    change (R0 and arm a run in different turns)."""
    return {**library_versions(), "nvidia_driver": nvidia_driver_version()}


def current_pins(root=ROOT, rcx_root=RCX_ROOT, *, with_snapshot_files: bool = True) -> dict:
    """Check every pinned input; returns {name: sha256}. Raises PinError on any mismatch. The code-path files are
    enforced (edit 6), not only recorded."""
    root, rcx_root = Path(root), Path(rcx_root)
    shas = {PLANNER_REL: check_pin(root / PLANNER_REL, PLANNER_SHA256, PLANNER_REL),
            BLOCKS_REL: check_pin(root / BLOCKS_REL, BLOCKS_SHA256, BLOCKS_REL),
            B2_BEAM_FREEZE_REL: check_pin(root / B2_BEAM_FREEZE_REL, B2_BEAM_FREEZE_SHA256, B2_BEAM_FREEZE_REL),
            ITEM7_PREREG_REL: check_pin(root / ITEM7_PREREG_REL, ITEM7_PREREG_SHA256, ITEM7_PREREG_REL),
            SEEDS_FRESH_REL: check_pin(root / SEEDS_FRESH_REL, SEEDS_FRESH_SHA256, SEEDS_FRESH_REL)}
    for rel, want in CODE_PATH_PINS.items():
        shas[rel] = check_pin(root / rel, want, rel)
    files = rcx_snapshot_files() if with_snapshot_files else {**RCX_CODE_PINS, GNN_REL: GNN_SHA256,
                                                              GNN_SIDECAR_REL: GNN_SIDECAR_SHA256,
                                                              E07_INSTANCES_REL: E07_INSTANCES_SHA256}
    for rel, want in files.items():
        shas["rcx:" + rel] = check_pin(rcx_root / rel, want, "RCX " + rel)
    return shas


def make_snapshot(rcx_root, snap) -> dict:
    """Edit 10: copy every sha-verified RCX file into the screen's own tree (runs/BW_B4_SCREEN/rcx_snapshot/), so run
    and analyze never read RCX's worktree. An existing snapshot (an earlier, interrupted freeze) must already match."""
    files = rcx_snapshot_files()
    snap = Path(snap)
    for rel, want in files.items():
        check_pin(Path(rcx_root) / rel, want, "RCX " + rel)
    for rel, want in files.items():
        dst = snap / rel
        if dst.exists():
            check_pin(dst, want, "snapshot " + rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(rcx_root) / rel, dst)
        check_pin(dst, want, "snapshot " + rel)
    extra = sorted(str(p.relative_to(snap)).replace("\\", "/") for p in snap.rglob("*") if p.is_file()
                   and str(p.relative_to(snap)).replace("\\", "/") not in files)
    if extra:
        raise PinError(f"unexpected files in the snapshot: {extra}")
    return dict(files)


def check_adapters(root, rec: dict):
    """The Pa and F adapter weights named by a freeze record's args have the recorded sha256s."""
    for key, rel in (("pa_sha256", rec["args"]["pa"]), ("fwd_sha256", rec["args"]["fwd_adapter"])):
        check_pin(Path(root) / rel / "adapter_model.safetensors", rec[key], key)


def check_r0_is_b2(r0_rec: dict, b2_rec: dict):
    """R0's beam freeze must be the B2 beam configuration itself (bw_seeds.freeze: B2 args with R0's adapter paths,
    which are B2's), so R0's beam run is arm c of B4."""
    diff = {k: (r0_rec["args"].get(k), b2_rec["args"].get(k)) for k in set(r0_rec["args"]) | set(b2_rec["args"])
            if r0_rec["args"].get(k) != b2_rec["args"].get(k)}
    if diff:
        raise PinError(f"FREEZE_I7_R0_beam args differ from FREEZE_B2_beam: {diff}")
    for k in ("pa_sha256", "fwd_sha256"):
        if r0_rec.get(k) != b2_rec.get(k):
            raise PinError(f"FREEZE_I7_R0_beam {k} {r0_rec.get(k)} != FREEZE_B2_beam {b2_rec.get(k)}")


def build_freeze_record(r0_rec: dict, r0_freeze_sha: str, *, pins: dict, pre: dict, prereg: dict, addendum: dict,
                        seeds_outputs_present: list, runner_sha: str, now: str, registered: dict | None = None,
                        versions: dict | None = None, snapshot: dict | None = None,
                        canary_versions: dict | None = None) -> dict:
    return {"what": "B4 futility screen, arm a: RCX hybrid hook as unmet= in item 7's R0 S-beam configuration",
            "args": dict(r0_rec["args"]),
            "unmet_hook": {"class": HOOK_CLASS, "constructor": "from_file", "model_file": GNN_REL,
                           "kwargs": dict(HOOK_KWARGS), "registry": "off (no instances= in the D-76a call)",
                           "loaded_from": SNAPSHOT_REL},
            "control": {"system": "R0", "freeze": R0_FREEZE_REL, "freeze_sha256": r0_freeze_sha, "out": R0_OUT_REL},
            "set": {"name": "seeds_fresh", "path": SEEDS_FRESH_REL, "sha256": SEEDS_FRESH_SHA256, "n": N_FRESH,
                    "shards": SHARDS},
            "pa_sha256": r0_rec["pa_sha256"], "fwd_sha256": r0_rec["fwd_sha256"],
            "pins": pins, "code_path_pins": dict(CODE_PATH_PINS),
            "precheck": {"drop_ids": pre["drop_ids"], "n_after_drop": pre["n_after_drop"]},
            "rule": {"go_min_net": GO_MIN_NET, "penalty_share_max": PENALTY_SHARE_MAX,
                     "penalty_band": "value >= PENALTY (1000): invalid state or critic error",
                     "reported_only": list(REPORTED_ONLY), "text": RULE_TEXT},
            "void_rule": {"hook_counters_must_be_zero": list(VOID_COUNTERS), "model_evals_must_be_positive": True,
                          "canary_every_hook_shard": True, "r0_outputs_must_match": ADDENDUM_A_INPUTS_REL},
            "canary": {"rel": CANARY_REL, "sha256": CANARY_SHA256, "tolerance_abs": CANARY_TOL,
                       **(canary_versions or {})},
            "rcx_snapshot": {"dir": SNAPSHOT_REL, "files": snapshot or {}},
            "versions": versions or {},
            "prereg": prereg, "addendum": addendum, "item7_prereg_sha256": ITEM7_PREREG_SHA256,
            "runner": "scripts/b4_screen.py", "runner_sha256": runner_sha, "registered": registered or {},
            "late_freeze": bool(seeds_outputs_present),
            "item7_outputs_present_at_freeze": seeds_outputs_present, "written": now}


def bw_seeds_outputs(root=ROOT) -> list:
    """Edit 9: every file under runs/BW_SEEDS/ (R0_direct reveals per-instance outcomes too, and a running shard's
    progress.txt shows per-round solve counts), plus bundleI's log once it shows that an item-7 run has started."""
    root = Path(root)
    d = root / BW_SEEDS_REL
    out = sorted(str(p.relative_to(root)).replace("\\", "/") for p in d.rglob("*") if p.is_file()) if d.exists() else []
    log = root / BW_SEEDS_LOG_REL
    if log.exists() and "[seeds] run" in log.read_text(encoding="utf-8", errors="replace"):
        out.append(BW_SEEDS_LOG_REL)
    return out


def closing_markers(root=ROOT) -> list:
    """Edit 9: what closes the late-freeze window: item 7's report or addendum A.0's table, or bundleI's log once it
    holds item 7's reading (RCX low item 5: `bw_seeds.py analyze` prints the report, incl. release_gate_reading)."""
    root = Path(root)
    out = [m for m in CLOSING_MARKERS if (root / m).exists()]
    log = root / BW_SEEDS_LOG_REL
    if log.exists() and CLOSING_LOG_MARKER in log.read_text(encoding="utf-8", errors="replace"):
        out.append(f"{BW_SEEDS_LOG_REL} (contains {CLOSING_LOG_MARKER!r})")
    return out


def canary_versions_at_freeze(snapshot_root, host_torch) -> dict:
    """D-80 follow-up, report-only: the torch version the canary was made with (from the sha-verified, schema-checked
    snapshot copy; torch-free) beside the host's (package metadata). The freeze warns on a mismatch, never refuses."""
    c = load_canary(Path(snapshot_root) / CANARY_REL)
    tv = c.get("torch_version")
    match = None if tv is None or host_torch is None else tv == host_torch
    return {"canary_torch_version": tv, "canary_torch_threads": c.get("torch_threads"),
            "host_torch_version_at_freeze": host_torch, "torch_versions_match_at_freeze": match,
            "torch_version_rule": "report-only: a mismatch is flagged, never VOID (D-80 follow-up)"}


def stage_freeze(a, root=ROOT, rcx_root=RCX_ROOT) -> int:
    root = Path(root)
    frz = root / FREEZE_REL
    if frz.exists():
        sys.exit(f"{frz} exists (written once)")
    r0f = root / R0_FREEZE_REL
    if not r0f.exists():
        sys.exit(f"{r0f} missing: run `scripts/bw_seeds.py freeze` first (the screen is frozen right after item 7)")
    closed = closing_markers(root)
    if closed:
        sys.exit(f"{closed} exist: the screen can no longer be frozen, not even late (edit 9); the screen is void")
    present = bw_seeds_outputs(root)
    if present and not a.allow_after_r0:
        sys.exit(f"item-7 outputs already exist ({present[:3]}...): the screen must be frozen before any item-7 output; "
                 "the late freeze (--allow-after-r0) is then mandatory and is recorded with the files present")
    docs, texts = {}, {}
    for key, p in (("prereg", a.prereg), ("addendum", a.addendum)):
        if not p:
            sys.exit(f"--{key} is required")
        p = Path(p)
        if "DRAFT" in p.name.upper() and not a.allow_draft:
            sys.exit(f"{p} is a draft: freeze against the final, hash-recorded text (or --allow-draft)")
        docs[key] = {"path": str(p), "sha256": sha256_file(p)}
        texts[key] = p.read_text(encoding="utf-8")
    try:  # edit 2: the runner and tests on disk must be the registered bytes, in the late path as well
        registered = check_registered_runner(texts["prereg"], texts["addendum"], root)
        named = addendum_prereg_sha(texts["addendum"])   # RCX low item 2: the addendum names exactly this prereg
        if named != docs["prereg"]["sha256"]:
            raise PinError(f"the addendum's '{ADDENDUM_PREREG_ROW}' row names {named}, but --prereg "
                           f"{docs['prereg']['path']} has sha256 {docs['prereg']['sha256']}")
        rcx_snapshot_files()
        load_canary(Path(rcx_root) / CANARY_REL)          # sha and schema of the canary (torch-free)
    except (PinError, CanaryError) as e:
        sys.exit(f"freeze refused: {e}")
    pins = current_pins(root, rcx_root)
    r0_rec = json.loads(r0f.read_text())
    check_r0_is_b2(r0_rec, json.loads((root / B2_BEAM_FREEZE_REL).read_text()))
    check_adapters(root, r0_rec)
    pre = precheck(root / SEEDS_FRESH_REL, Path(rcx_root) / E07_INSTANCES_REL)
    versions = runtime_versions()
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    snapshot = make_snapshot(rcx_root, root / SNAPSHOT_REL)
    canary_tv = canary_versions_at_freeze(root / SNAPSHOT_REL, versions.get("torch"))
    if canary_tv["torch_versions_match_at_freeze"] is not True:
        print(f"[b4s] WARNING (report-only, not a refusal): the canary was made with torch "
              f"{canary_tv['canary_torch_version']} and this host has torch {versions.get('torch')}; the binding check "
              "is the per-probe |delta| <= 1e-5 at every hook shard start", flush=True)
    (out / "PRECHECK.json").write_text(json.dumps(pre, indent=1))
    rec = build_freeze_record(r0_rec, sha256_file(r0f), pins=pins, pre=pre, prereg=docs["prereg"],
                              addendum=docs["addendum"], seeds_outputs_present=present,
                              runner_sha=registered["runner_sha256"], registered=registered, versions=versions,
                              snapshot=snapshot, canary_versions=canary_tv,
                              now=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    frz.write_text(json.dumps(rec, indent=1))
    print(f"wrote {frz} (precheck drops {pre['drop_ids']}; n = {pre['n_after_drop']}; late {bool(present)})")
    return 0


def load_and_verify_freeze(root=ROOT) -> dict:
    """Every registered stage re-verifies the freeze: R0's freeze record and args, the runner bytes, the canary sha,
    every pin (RCX files in the SNAPSHOT), the code path, the snapshot files and the adapter shas."""
    root = Path(root)
    rec = json.loads((root / FREEZE_REL).read_text())
    r0f = root / R0_FREEZE_REL
    if sha256_file(r0f) != rec["control"]["freeze_sha256"]:
        raise PinError("FREEZE_I7_R0_beam.json changed since the screen freeze")
    if json.loads(r0f.read_text())["args"] != rec["args"]:
        raise PinError("screen args != R0 beam args")
    if sha256_file(Path(__file__)) != rec["runner_sha256"]:
        raise PinError("scripts/b4_screen.py changed since the freeze")
    if rec["canary"]["sha256"] != CANARY_SHA256 or CANARY_SHA256 is None:
        raise PinError("the canary sha differs from the frozen one")
    now = current_pins(root, root / SNAPSHOT_REL)
    bad = {k: (now.get(k), v) for k, v in rec["pins"].items() if now.get(k) != v}
    bad.update({k: (v, None) for k, v in now.items() if k not in rec["pins"]})
    if bad:
        raise PinError(f"pinned inputs changed since the freeze: {bad}")
    if rec["code_path_pins"] != CODE_PATH_PINS:
        raise PinError("code-path pins differ from the frozen ones")
    if set(rec["rcx_snapshot"]["files"]) != set(rcx_snapshot_files()):
        raise PinError("the frozen snapshot does not list every RCX file")
    for rel, want in rec["rcx_snapshot"]["files"].items():
        check_pin(root / SNAPSHOT_REL / rel, want, "snapshot " + rel)
    check_adapters(root, rec)
    return rec


def expected_r0_shard_args(args: dict, i: int) -> dict:
    """blocks_eval.py's argparse namespace for R0's beam shard i, as bw_seeds.run invokes it."""
    want = {k: args.get(k) for k in BLOCKS_EVAL_ARG_KEYS}
    want.update({"set": "seeds_fresh", "val": True, "freeze": R0_FREEZE_REL, "shard": f"{i}/{SHARDS}", "n": 0,
                 "max_batch_tokens": F_BATCH_TOKENS, "prop_batch_tokens": PROP_BATCH_TOKENS})
    return want


def check_r0_shards(root, rec: dict) -> dict:
    """Edit 7: R0's shard summaries record blocks_eval's arguments; they must be exactly the frozen S-beam
    configuration on seeds_fresh, shard i/4, writing to R0's shard directory. Returns {file: sha256}."""
    out = {}
    for i in range(SHARDS):
        p = Path(root) / R0_OUT_REL / f"shard_{i}of{SHARDS}" / "summary.json"
        if not p.exists():
            raise PinError(f"{p} missing: R0's beam run is incomplete")
        got = json.loads(p.read_text())["args"]
        want = expected_r0_shard_args(rec["args"], i)
        bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if str(got.get("out", "")).replace("\\", "/") != f"{R0_OUT_REL}/shard_{i}of{SHARDS}":
            bad["out"] = (got.get("out"), f"{R0_OUT_REL}/shard_{i}of{SHARDS}")
        extra = set(got) - set(want) - {"out"}
        if extra:
            bad["unexpected_keys"] = sorted(extra)
        if bad:
            raise PinError(f"R0 shard {i} args differ from the frozen S-beam configuration: {bad}")
        out[str(p.relative_to(root)).replace("\\", "/")] = sha256_file(p)
    return out


def r0_args_problems(root, rec: dict) -> list:
    """check_r0_shards as a list of problems (empty = OK). A failure is an INSTRUMENT FAULT (edit 7): run and run-shard
    refuse, and analyze reports the fault so that the pre-registered re-run of c and a can follow."""
    try:
        check_r0_shards(root, rec)
    except (PinError, KeyError, ValueError, TypeError, AttributeError) as e:
        return [f"R0 shard arguments: {e}"]
    return []


def r0_files(root) -> list:
    """Every R0 beam file the screen reads (merged and per shard)."""
    root = Path(root)
    rels = [f"{R0_OUT_REL}/plans.jsonl", f"{R0_OUT_REL}/summary.json"]
    for i in range(SHARDS):
        rels += [f"{R0_OUT_REL}/shard_{i}of{SHARDS}/{f}" for f in ("plans.jsonl", "summary.json", "fcache.jsonl")]
    return rels


def r0_matches_addendum(root) -> list:
    """Edit 9: the screen uses only R0 outputs whose sha256 equals addendum A.0's table; returns the problems
    (empty = OK). Any problem makes the screen VOID."""
    root = Path(root)
    p = root / ADDENDUM_A_INPUTS_REL
    if not p.exists():
        return [f"{ADDENDUM_A_INPUTS_REL} missing (addendum A.0 must hash item 7's outputs first)"]
    table = json.loads(p.read_text())["files"]
    probs = []
    for rel in r0_files(root):
        if rel not in table:
            probs.append(f"{rel} not in {ADDENDUM_A_INPUTS_REL}")
        elif not (root / rel).exists():
            probs.append(f"{rel} missing")
        elif sha256_file(root / rel) != table[rel]["sha256"]:
            probs.append(f"{rel} sha256 differs from {ADDENDUM_A_INPUTS_REL}")
    return probs


def check_versions(frozen: dict, now: dict, torch_mod=None):
    """Edit 6: refuse the arm run on any change of the recorded library or driver versions."""
    diff = {k: (now.get(k), v) for k, v in frozen.items() if now.get(k) != v}
    diff.update({k: (v, None) for k, v in now.items() if k not in frozen})
    if torch_mod is not None and getattr(torch_mod, "__version__", None) != frozen.get("torch"):
        diff["torch (loaded)"] = (getattr(torch_mod, "__version__", None), frozen.get("torch"))
    if diff:
        raise PinError(f"library/driver versions changed since the freeze: {diff}")


def log_attempt(root, **kw):
    """Append-only record of every run attempt (ATTEMPTS.jsonl)."""
    p = Path(root) / ATTEMPTS_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), **kw}) + "\n")


def attempt_events(root) -> list:
    p = Path(root) / ATTEMPTS_REL
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            out.append({"event": "torn_line"})
    return out


def canary_failures(root, arm: str) -> list:
    """Edit 8: a canary failure is final (the screen is VOID); no shard of that arm runs afterwards."""
    return [f"{arm} {e.get('shard')}: canary failed ({str(e.get('error', ''))[:200]})" for e in attempt_events(root)
            if e.get("event") == "canary_failed" and e.get("arm") == arm]


def attempts_problems(root, arm: str) -> list:
    """RCX low item 3: once an arm has run (its directory exists; run-shard creates the shard directory before the
    canary), ATTEMPTS.jsonl must exist, hold a start event for every shard directory of the arm and exactly one end
    event per completed shard (summary.json) and none for an incomplete one. So deleting it cannot erase a canary-failure
    VOID: run-shard and analyze refuse instead (an edit that removes only a canary_failed line is not detected here).
    Returns the problems (empty = OK)."""
    root = Path(root)
    d = root / OUT_REL / ARMS[arm]["dir"]
    if not d.exists():
        return []
    if not (root / ATTEMPTS_REL).exists():
        return [f"{ATTEMPTS_REL} missing although arm {arm} has run ({ARMS[arm]['dir']} exists)"]
    ev = [e for e in attempt_events(root) if e.get("stage") == "run-shard" and e.get("arm") == arm]
    probs = []
    for i in range(SHARDS):
        sd, sh = d / f"shard_{i}of{SHARDS}", f"{i}/{SHARDS}"
        if sd.exists() and not any(e.get("event") == "start" and e.get("shard") == sh for e in ev):
            probs.append(f"{ATTEMPTS_REL}: no start event for arm {arm} shard {sh}, whose directory exists")
        ends = sum(e.get("event") == "end" and e.get("shard") == sh for e in ev)
        want = int((sd / "summary.json").exists())
        if ends != want:
            probs.append(f"{ATTEMPTS_REL}: {ends} end events for arm {arm} shard {sh} (want {want})")
    return probs


# ----------------------------------------------------------------------------- GPU stages

def load_gpu_stack(root, args: dict):
    """(B, BP, model, tok, torch): the pinned planner, then the core and the two adapters exactly as blocks_eval.py
    loads them for R0 (fwd first via PeftModel.from_pretrained, then pa via load_adapter)."""
    B, BP = import_planner_real()
    import torch
    from bwmn import lm
    from peft import PeftModel
    base, tok = lm.load_core("titan-sparse")
    model = PeftModel.from_pretrained(base, str(Path(root) / args["fwd_adapter"]), adapter_name="fwd",
                                      is_trainable=False)
    model.load_adapter(str(Path(root) / args["pa"]), adapter_name="pa", is_trainable=False)
    model.eval()
    return B, BP, model, tok, torch


def _rerun_allowed(root) -> bool:
    p = Path(root) / OUT_REL / (REPORT[False] + ".json")
    return p.exists() and json.loads(p.read_text()).get("status") == "INSTRUMENT_FAULT"


def stage_run(a, root=ROOT) -> int:
    root = Path(root)
    rec = load_and_verify_freeze(root)
    if a.rerun:
        if not _rerun_allowed(root):
            sys.exit("--rerun is the pre-registered consequence of an INSTRUMENT_FAULT report only")
        arms = ("rerun_c", "rerun_a")       # c and a in one GPU turn (edit 7)
    else:
        check_r0_shards(root, rec)          # edit 7: R0's shard args asserted before the GPU run
        probs = r0_matches_addendum(root)   # edit 9: only sha-matched R0 outputs; otherwise the screen is void
        if probs:
            sys.exit(f"screen VOID: {probs}")
        arms = ("a",)
    for arm in arms:
        d = root / OUT_REL / ARMS[arm]["dir"]
        if (d / "summary.json").exists():
            print(f"[b4s] {d}/summary.json exists: done")
            continue
        for i in range(SHARDS):
            if (d / f"shard_{i}of{SHARDS}" / "summary.json").exists():
                continue
            print(f"[b4s] run {arm} shard {i}/{SHARDS}", flush=True)
            rc = subprocess.call([sys.executable, "-B", str(Path(__file__)), "run-shard", "--arm", arm,
                                  "--shard", f"{i}/{SHARDS}"], cwd=str(root))
            log_attempt(root, stage="run", arm=arm, shard=f"{i}/{SHARDS}", event="shard_exit", rc=rc)
            if rc != 0:
                print(f"[b4s] {arm} shard {i} exited {rc}", flush=True)
                return rc
        merge_shards(d, SHARDS, with_hook=ARMS[arm]["hook"])
        print(f"[b4s] done {arm}", (d / "summary.json").read_text()[:400], flush=True)
    return 0


def stage_run_shard(a, root=ROOT) -> int:
    root = Path(root)
    arm = ARMS[a.arm]
    rec = load_and_verify_freeze(root)
    si, sk = map(int, a.shard.split("/"))
    if sk != SHARDS or not 0 <= si < sk:
        sys.exit(f"--shard must be i/{SHARDS}")
    dst = root / OUT_REL / arm["dir"] / f"shard_{si}of{sk}"
    if (dst / "summary.json").exists():
        sys.exit(f"{dst}/summary.json exists: each arm shard runs once")
    if a.arm == "a":
        check_r0_shards(root, rec)
        probs = r0_matches_addendum(root)
        if probs:
            sys.exit(f"screen VOID: {probs}")
    elif not _rerun_allowed(root):
        sys.exit("re-run arms run only after an INSTRUMENT_FAULT report")
    failed = canary_failures(root, a.arm)
    if failed:
        sys.exit(f"screen VOID: {failed}")
    att = attempts_problems(root, a.arm)    # RCX low item 3: a deleted ATTEMPTS.jsonl cannot erase a canary failure
    if att:
        sys.exit(f"refusing: {att}")
    args = rec["args"]
    inst = [it for it in load_screen_instances(root) if shard_of(it["id"], sk) == si]
    fc = dst / "fcache.jsonl"
    warm = fc.exists()
    dst.mkdir(parents=True, exist_ok=True)  # the arm has run from here on: ATTEMPTS.jsonl is required (item 3)
    log_attempt(root, stage="run-shard", arm=a.arm, shard=a.shard, event="start", warm_fcache=warm,
                fcache_lines=len(fc.read_text(encoding="utf-8").splitlines()) if warm else 0)
    B, BP, model, tok, torch = load_gpu_stack(root, args)
    vers = runtime_versions()
    check_versions(rec["versions"], vers, torch)
    snap = root / SNAPSHOT_REL
    unmet = hook = meter = imp = canary = d0 = None
    if arm["hook"]:
        hook_mod, imp = load_rcx_hook_module(snap)
        cuda_before = (torch.cuda.is_available, torch.cuda.device_count, os.environ.get("CUDA_VISIBLE_DEVICES"))
        try:
            canary = check_canary(lambda: make_registered_hook(hook_mod, snap), snap / CANARY_REL,
                                  host_torch_version=torch.__version__)
        except (CanaryError, PinError) as e:
            log_attempt(root, stage="run-shard", arm=a.arm, shard=a.shard, event="canary_failed", error=str(e)[:2000])
            raise
        hook = make_registered_hook(hook_mod, snap)
        if (torch.cuda.is_available, torch.cuda.device_count, os.environ.get("CUDA_VISIBLE_DEVICES")) != cuda_before:
            raise RuntimeError("the hook import changed the host's CUDA view")
        d0 = verify_hook(hook, snap)
        meter = HookMeter(hook, hook_mod.PENALTY, hook_mod.FAILSAFE, inst, classify=hook_mod.is_consistent)
        unmet = meter
    summ, rows = run_arm(BP, B, model, tok, inst, args, dst, unmet=unmet, val=True)
    check_pin(BP.__file__, PLANNER_SHA256, PLANNER_REL)
    if arm["hook"]:
        d1 = verify_hook(hook, snap)
        if d1["counters"].get("calls") != meter.calls:
            raise RuntimeError(f"hook calls {d1['counters'].get('calls')} != meter calls {meter.calls}")
        (dst / "hook.json").write_text(json.dumps({
            "describe_start": d0, "describe_end": d1, "meter": meter.summary(), "canary": canary,
            "rcx_modules_executed": dict(imp.executed), "torch_version": torch.__version__, "versions": vers},
            indent=1, default=str))
        summ["hook_calls"], summ["hook_penalty_calls"] = meter.calls, meter.penalty_calls
    write_outputs(dst, summ, rows, {"args": args, "arm": a.arm, "shard": a.shard, "freeze": FREEZE_REL})
    log_attempt(root, stage="run-shard", arm=a.arm, shard=a.shard, event="end",
                f_model_queries=summ.get("f_model_queries"), hook_calls=summ.get("hook_calls"))
    print(json.dumps(summ), flush=True)
    return 0


def stage_canary(a, root=ROOT) -> int:
    """CPU pre-flight (not part of the registration): the canary on the snapshot, or on RCX's tree before the freeze."""
    snap = Path(root) / SNAPSHOT_REL
    rcx_root = snap if (snap / CANARY_REL).exists() else RCX_ROOT
    hook_mod, imp = load_rcx_hook_module(rcx_root)
    try:
        print(json.dumps(check_canary(lambda: make_registered_hook(hook_mod, rcx_root), rcx_root / CANARY_REL),
                         indent=1))
    finally:
        unload_rcx(imp)
    return 0


# ----------------------------------------------------------------------------- analysis stage

def _rows(p: Path) -> list:
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def require_hook_shards(arm_dir: Path, rec: dict) -> dict:
    """Edit 1: all 4 hook.json, calls > 0 and calls >= the shard's instances, meter calls == hook calls, the arm
    shard's own args, and the hook record of the D-76a call (real model; models.py executed)."""
    out = {}
    for i in range(SHARDS):
        sd = arm_dir / f"shard_{i}of{SHARDS}"
        hj = sd / "hook.json"
        if not hj.exists():
            raise PinError(f"{hj} missing")
        h = json.loads(hj.read_text())
        n = len(_rows(sd / "plans.jsonl"))
        calls = h["meter"]["calls"]
        if not (calls > 0 and calls >= n):
            raise PinError(f"{hj}: hook calls {calls} (need > 0 and >= {n} instances)")
        if h["describe_end"].get("counters", {}).get("calls") != calls:
            raise PinError(f"{hj}: meter calls {calls} != hook counter calls")
        d = h["describe_end"]
        bad = {k: d.get(k) for k, v in (("model_sha256", GNN_SHA256), ("config_source", "sidecar json 'config'"),
                                         ("invalid_policy", HOOK_KWARGS["invalid_policy"]),
                                         ("hybrid_lambda", HOOK_KWARGS["hybrid_lambda"])) if d.get(k) != v}
        if bad or "rcx/bw/models.py" not in (h.get("rcx_modules_executed") or {}):
            raise PinError(f"{hj}: not the registered hook ({bad}; executed {sorted(h.get('rcx_modules_executed') or {})})")
        s = json.loads((sd / "summary.json").read_text())
        if s.get("args") != rec["args"] or s.get("shard") != f"{i}/{SHARDS}":
            raise PinError(f"{sd}/summary.json: args/shard differ from the freeze")
        out[f"shard_{i}of{SHARDS}"] = {"calls": calls, "instances": n}
    return out


def require_val(rows: list, what: str):
    miss = [r["id"] for r in rows if "val" not in r]
    if miss:
        raise PinError(f"{what}: rows without a VAL verdict: {miss[:5]}")


def attempts_report(root) -> dict:
    ev = attempt_events(root)
    if not ev:
        return {"attempts": 0}
    starts = Counter((e.get("arm"), e.get("shard")) for e in ev if e.get("event") == "start")
    warm = sorted({f"{e.get('arm')} {e.get('shard')}" for e in ev if e.get("event") == "start" and e.get("warm_fcache")})
    return {"attempts": len(ev), "starts_per_shard": {f"{k[0]} {k[1]}": v for k, v in sorted(starts.items())},
            "warm_cache_shards": warm,
            "cost_flag": ("f_model_queries of a restarted shard counts only the queries of its last attempt (warm F "
                          "cache): cost figures for these shards are lower bounds") if warm else None}


def _refusal_problems(root, rec: dict, ctrl: Path, arm: Path, rerun: bool, arm_keys) -> tuple:
    """Every refusal check of the analysis (edit 1), collected instead of raised, so that a VOID and then an INSTRUMENT
    FAULT take precedence over a refusal (RCX low item 1: VOID > INSTRUMENT FAULT > refusal). Returns (problems,
    r0_args_problems, shards, hook, ctrl_rows, arm_rows); an output whose check could not run is None."""
    probs, shards, hook, ctrl_rows, arm_rows = [], None, None, None, None
    for k in arm_keys:
        probs += attempts_problems(root, k)                       # RCX low item 3
    r0_args = [] if rerun else r0_args_problems(root, rec)       # edit 7: an INSTRUMENT FAULT, not a refusal
    if rerun:
        for i in range(SHARDS):
            try:
                s = json.loads((ctrl / f"shard_{i}of{SHARDS}" / "summary.json").read_text())
                bad_c = (s.get("args") != rec["args"] or s.get("shard") != f"{i}/{SHARDS}"
                         or s.get("arm") != "rerun_c")
            except (OSError, ValueError, AttributeError) as e:
                probs.append(f"re-run c shard {i}: {e}")
                continue
            if bad_c:
                probs.append(f"re-run c shard {i}: args/shard differ from the freeze")
    malformed = (PinError, OSError, ValueError, KeyError, TypeError, AttributeError)   # collected, never raised
    try:
        shards = require_hook_shards(arm, rec)                    # edit 1
    except malformed as e:
        probs.append(f"hook shard records: {e}")
    try:
        hook = json.loads((arm / "hook_merged.json").read_text())
        if json.loads(json.dumps(merge_hooks(arm, SHARDS))) != hook:   # edits 1/8: VOID is read from the shards
            probs.append(f"{arm / 'hook_merged.json'} differs from the merge of the {SHARDS} shard hook records")
    except malformed as e:
        probs.append(f"hook_merged.json: {e}")
        hook = None
    try:
        ctrl_rows, arm_rows = _rows(ctrl / "plans.jsonl"), _rows(arm / "plans.jsonl")
        require_val(ctrl_rows, "control")
        require_val(arm_rows, "arm")
    except malformed as e:
        probs.append(f"plans: {e}")
    try:
        if hook is not None and shards is not None and arm_rows is not None and (
                hook.get("calls", 0) < len(arm_rows) or hook.get("calls") != sum(v["calls"] for v in shards.values())):
            probs.append("merged hook calls inconsistent with the shard records")
    except malformed:
        probs.append("merged hook calls inconsistent with the shard records")
    return probs, r0_args, shards, hook, ctrl_rows, arm_rows


def _write_report(rep: dict, root, rep_json: Path, rep_md: Path, meta: dict) -> int:
    rep.setdefault("rule", RULE_TEXT)
    rep["attempts"] = attempts_report(root)
    rep["meta"] = meta
    rep_json.write_text(json.dumps(rep, indent=1))
    rep_md.write_text(render_md(rep))
    print(rep["wording"])
    return 0


def stage_analyze(a, root=ROOT, rcx_root=None) -> int:
    """Edit 1: the analysis verifies what the run stage verifies, takes the rule constants and drop_ids from the frozen
    record, asserts R0's shard args, requires the 4 hook records (and hook_merged.json = their merge), ATTEMPTS.jsonl
    once the arm has run, and a VAL verdict on every row, and never overwrites its report. Edits 7-9: VOID and
    INSTRUMENT_FAULT statuses (an R0 shard-argument failure, on which run refuses, is reported as an INSTRUMENT_FAULT so
    that the pre-registered re-run can follow). Precedence (RCX low item 1): VOID > INSTRUMENT FAULT > refusal, so a
    VOID report is written even when a later check would refuse."""
    root = Path(root)
    rerun = bool(getattr(a, "rerun", False))
    out = root / OUT_REL
    rep_json, rep_md = out / (REPORT[rerun] + ".json"), out / (REPORT[rerun] + ".md")
    if rep_json.exists() or rep_md.exists():
        sys.exit(f"{rep_json} exists: the analysis is written once")
    rec = load_and_verify_freeze(root)
    pre = json.loads((out / "PRECHECK.json").read_text())
    if pre["drop_ids"] != rec["precheck"]["drop_ids"] or pre["n_after_drop"] != rec["precheck"]["n_after_drop"]:
        sys.exit("PRECHECK.json differs from the frozen record's pre-check: refusing")
    rule = rec["rule"]
    if rule["go_min_net"] != GO_MIN_NET or rule["penalty_share_max"] != PENALTY_SHARE_MAX or "go_max_p_one_sided" in rule:
        sys.exit(f"frozen rule {rule} != this runner's rule: refusing")
    if rerun and not _rerun_allowed(root):
        sys.exit("--rerun analyses the pre-registered re-run after an INSTRUMENT_FAULT only")
    ctrl = (out / ARMS["rerun_c"]["dir"]) if rerun else (root / R0_OUT_REL)
    arm_key = "rerun_a" if rerun else "a"
    arm_keys = ("rerun_c", "rerun_a") if rerun else ("a",)
    arm = out / ARMS[arm_key]["dir"]
    early = ([] if rerun else r0_matches_addendum(root)) + canary_failures(root, arm_key)
    arm_done = (arm / "summary.json").exists()
    meta = {"freeze_sha256": sha256_file(root / FREEZE_REL), "prereg": rec["prereg"],
            "runner_sha256": sha256_file(Path(__file__)), "rerun": rerun,
            "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if not arm_done:                                              # the run refused or has not finished
        att = [p for k in arm_keys for p in attempts_problems(root, k)]
        fault = [] if rerun else r0_args_problems(root, rec)     # edit 7: R0's shard args (run refuses)
        if early:                                                 # VOID > INSTRUMENT FAULT > refusal
            rep = {"status": "VOID", "void_reasons": early + [f"arm {arm_key} was not completed (run refuses on "
                                                              "these)"], "wording": WORDING["VOID"]}
            if fault:
                rep["r0_args_problems"] = fault
            if att:
                rep["refusal_problems_report_only"] = att
        elif att:                                                 # it could hide a canary-failure VOID
            sys.exit(f"refusing: {att}")
        elif fault:                                               # edit 7: no decision; `run --rerun` is unlocked
            rep = {"status": "INSTRUMENT_FAULT", "decision": None,
                   "instrument_fault_reasons": fault + [f"arm {arm_key} was not run (run refuses on this)"],
                   "wording": WORDING["INSTRUMENT_FAULT_R0_ARGS"]}
        else:
            sys.exit(f"{arm / 'summary.json'} missing: arm {arm_key} is not complete")
        return _write_report(rep, root, rep_json, rep_md, meta)
    files = [ctrl / "plans.jsonl", ctrl / "summary.json", arm / "plans.jsonl", arm / "summary.json",
             arm / "hook_merged.json", out / "PRECHECK.json", root / FREEZE_REL]
    for d in (ctrl, arm):
        for i in range(SHARDS):
            sd = d / f"shard_{i}of{SHARDS}"
            files += [sd / "plans.jsonl", sd / "summary.json", sd / "fcache.jsonl"]
    files += [arm / f"shard_{i}of{SHARDS}" / "hook.json" for i in range(SHARDS)]
    files += [p for p in (root / ADDENDUM_A_INPUTS_REL, root / ATTEMPTS_REL, root / SNAPSHOT_REL / CANARY_REL)
              if p.exists()]
    missing = [str(p.relative_to(root)).replace("\\", "/") for p in files if not p.exists()]
    if missing and not early:
        sys.exit(f"inputs missing: {missing[:5]}")
    inputs = {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in files if p.exists()}  # before parse
    inp_path = out / ("ANALYSIS_INPUTS_RERUN.json" if rerun else "ANALYSIS_INPUTS.json")
    if inp_path.exists() and json.loads(inp_path.read_text()) != inputs:
        sys.exit(f"{inp_path} exists with other shas: the inputs changed since a previous analysis attempt")
    inp_path.write_text(json.dumps(inputs, indent=1))
    meta.update(freeze_sha256=inputs[FREEZE_REL], inputs=inp_path.name)
    void, hook_shards_merge = list(early), None
    try:                                  # edit 8: the VOID counters and canaries are read from the shard records
        hook_shards_merge = json.loads(json.dumps(merge_hooks(arm, SHARDS, require=False)))
        void += void_reasons(hook_shards_merge)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass                              # malformed records: refused below unless a VOID already holds
    probs, r0_args, shards, hook, ctrl_rows, arm_rows = _refusal_problems(root, rec, ctrl, arm, rerun, arm_keys)
    if void:                              # VOID first (RCX low item 1), whatever the later checks say
        rep = {"status": "VOID", "void_reasons": void, "wording": WORDING["VOID"]}
        for k, v in (("inputs_missing", missing), ("r0_args_problems", r0_args),
                     ("refusal_problems_report_only", probs), ("hook_shards", shards)):
            if v:
                rep[k] = v
        if hook_shards_merge is not None:
            rep["canary_report_only"] = canary_report(hook_shards_merge)
        return _write_report(rep, root, rep_json, rep_md, meta)
    if r0_args:                           # edit 7, also after the arm ran: R0 is not the registered control
        rep = {"status": "INSTRUMENT_FAULT", "decision": None, "instrument_fault_reasons": r0_args,
               "wording": WORDING["INSTRUMENT_FAULT_R0_ARGS"]}
        if probs:
            rep["refusal_problems_report_only"] = probs
        if hook_shards_merge is not None:
            rep["canary_report_only"] = canary_report(hook_shards_merge)
        return _write_report(rep, root, rep_json, rep_md, meta)
    if probs:
        sys.exit(f"refusing: {probs}")
    hook_mod, imp = load_rcx_hook_module(rcx_root or root / SNAPSHOT_REL)
    try:
        from bwmn import blocks as B
        check_pin(B.__file__, BLOCKS_SHA256, BLOCKS_REL)
        rep = analyze_pair(ctrl_rows, arm_rows, n_expected=N_FRESH, drop_ids=rec["precheck"]["drop_ids"],
                           hook=hook, instances=load_screen_instances(root), B=B, hook_mod=hook_mod,
                           ctrl_caches=sorted(ctrl.glob("shard_*/fcache.jsonl")),
                           arm_caches=sorted(arm.glob("shard_*/fcache.jsonl")),
                           arm_caches_by_shard={i: arm / f"shard_{i}of{SHARDS}" / "fcache.jsonl"
                                                for i in range(SHARDS)},
                           labels=("c", "a") if rerun else ("R0", "a"), go_min_net=rule["go_min_net"],
                           penalty_share_max=rule["penalty_share_max"],
                           fault_status="INSTRUMENT_FAULT_RERUN" if rerun else "INSTRUMENT_FAULT")
    finally:
        unload_rcx(imp)
    rep["hook_shards"] = shards
    rep["canary_report_only"] = canary_report(hook)
    rep["cost"] = {k: {x: json.loads((d / "summary.json").read_text())["summary"].get(x)
                       for x in ("f_queries_per_instance", "proposer_tokens_per_instance", "wall_s",
                                 "f_model_queries")}
                   for k, d in (("control", ctrl), ("a", arm))}
    return _write_report(rep, root, rep_json, rep_md, meta)


def _canary_md(rep: dict) -> list:
    cr = rep.get("canary_report_only") or {}
    if not cr.get("shards"):
        return []
    L = ["", "Canary per hook shard (the torch versions are report-only; the binding check is |delta| <= 1e-5 per "
             "probe):", "", "| shard | max abs delta | canary torch | host torch | match |", "|---|---|---|---|---|"]
    L += [f"| {sh} | {v['max_abs_delta']} | {v['canary_torch_version']} | {v['host_torch_version']} | "
          f"{v['torch_versions_match']} |" for sh, v in cr["shards"].items()]
    if cr.get("torch_version_flag"):
        L += ["", f"Flag (descriptive, not VOID): {cr['torch_version_flag']}"]
    return L


def render_md(rep: dict, title="B4 futility screen (arm a vs R0, seeds_fresh)") -> str:
    L = [f"# {title}", "", f"**Status: {rep['status']}.** {rep['wording']}", "", f"Rule: {rep['rule']}", ""]
    if rep["status"] == "VOID":
        return "\n".join(L + ["Void reasons:", ""] + [f"- {r}" for r in rep["void_reasons"]] + _canary_md(rep)) + "\n"
    if "paired" not in rep:          # an INSTRUMENT FAULT on R0's shard arguments (edit 7): no paired table
        return "\n".join(L + ["Instrument fault reasons:", ""]
                         + [f"- {r}" for r in rep.get("instrument_fault_reasons", [])] + _canary_md(rep)) + "\n"
    d, p = rep.get("decision"), rep["paired"]
    L += ["| | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in p.items()]
    if d:
        L += [f"| net | {d['net']} |", f"| one-sided sign p (reported only) | {d['p_one_sided_sign']:.4g} |",
              f"| two-sided exact McNemar p (reported only) | {d['p_mcnemar_two_sided']:.4g} |"]
    L += [f"| n paired | {rep['n_paired']} |", f"| round-1 identity holds | {rep['round1_identity']['holds']} |", ""]
    L += ["| arm | success | CP 95% CI | failures | unverified VAL successes | mean tokens | mean rounds |",
          "|---|---|---|---|---|---|---|"]
    for k, v in rep["arms"].items():
        L.append(f"| {k} | {v['success']} | {v['ci95']} | {v['failure_taxonomy']} | {v['unverified_val_successes']} | "
                 f"{v['mean_gen_tokens']} | {v['mean_rounds']} |")
    if "hook" in rep:
        h = rep["hook"]
        L += ["", f"Hook: {h.get('calls')} calls; penalty band (invalid state or critic error) share "
                  f"{h['penalty_share_calls']} (max {h['penalty_share_max']}; split {h['penalty_split_calls']}); "
                  f"GO credited to the critic: {h['go_credited_to_critic']}."]
    return "\n".join(L + _canary_md(rep)) + "\n"


# ----------------------------------------------------------------------------- dry run

def _pycache_snapshot(root: Path) -> dict:
    snap = {}
    for d in root.rglob("__pycache__"):
        snap[str(d)] = sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in d.iterdir())
    return snap


def dry_run(out: Path = DRY_OUT, *, sizes=DRY_SIZES, val: bool = True, inconsistent: float = 0.05,
            shard_check: bool = True, rcx_root=RCX_ROOT) -> dict:
    out = Path(out).resolve()
    if out == (ROOT / "runs").resolve() or (ROOT / "runs").resolve() in out.parents:
        raise ValueError("the dry run never writes under runs/")
    heavy_before = {m: m in sys.modules for m in ("torch", "transformers", "peft")}
    out.mkdir(parents=True, exist_ok=True)
    for d in ("control_beam", "a_stub_beam", "a_stub_beam_unsharded"):
        for f in (out / d).rglob("*") if (out / d).exists() else []:
            if f.is_file() and f.suffix in (".jsonl", ".json", ".txt", ".pddl"):
                f.unlink()
    pins = {PLANNER_REL: check_pin(ROOT / PLANNER_REL, PLANNER_SHA256, PLANNER_REL),
            BLOCKS_REL: check_pin(ROOT / BLOCKS_REL, BLOCKS_SHA256, BLOCKS_REL),
            B2_BEAM_FREEZE_REL: check_pin(ROOT / B2_BEAM_FREEZE_REL, B2_BEAM_FREEZE_SHA256, B2_BEAM_FREEZE_REL),
            ITEM7_PREREG_REL: check_pin(ROOT / ITEM7_PREREG_REL, ITEM7_PREREG_SHA256, ITEM7_PREREG_REL),
            SEEDS_FRESH_REL + " (hashed only; not planned on)": check_pin(ROOT / SEEDS_FRESH_REL, SEEDS_FRESH_SHA256,
                                                                          SEEDS_FRESH_REL)}
    for rel, want in CODE_PATH_PINS.items():
        pins[rel] = check_pin(ROOT / rel, want, rel)
    for rel, want in RCX_CODE_PINS.items():
        pins["rcx:" + rel] = check_pin(rcx_root / rel, want, rel)
    pins["rcx:" + GNN_REL + " (hashed only; not loaded)"] = check_pin(rcx_root / GNN_REL, GNN_SHA256, GNN_REL)
    pins["rcx:" + GNN_SIDECAR_REL] = check_pin(rcx_root / GNN_SIDECAR_REL, GNN_SIDECAR_SHA256, GNN_SIDECAR_REL)
    rcx_pyc_before = _pycache_snapshot(rcx_root / "rcx")
    b2 = json.loads((ROOT / B2_BEAM_FREEZE_REL).read_text())
    args = dict(b2["args"])
    stub = make_stub_lm()
    B, BP = exec_pinned_planner(stub)
    hook_mod, imp = load_rcx_hook_module(rcx_root)
    try:
        Stub = make_stub_hook_class(hook_mod)
        inst = dry_instances(B, sizes)
        tok = FakeTok()
        res = {}
        # control: the planner's default key, exactly as R0 runs (unmet=None)
        model = ScriptedModel(B, inconsistent=inconsistent)
        summ_c, rows_c = run_arm(BP, B, model, tok, inst, args, out / "control_beam", unmet=None, val=val)
        write_outputs(out / "control_beam", summ_c, rows_c, {"args": args, "DRY_RUN": True})
        res["control_adapter_calls"] = dict(model.adapter_calls)
        # arm: the pinned hook class with a stub forward, in SHARDS shards merged by bw_run_tests.merge
        arm_dir = out / "a_stub_beam"
        for i in range(SHARDS):
            sd = arm_dir / f"shard_{i}of{SHARDS}"
            part = [it for j, it in enumerate(inst) if j % SHARDS == i]
            meter = HookMeter(Stub(**HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE, part,
                              classify=hook_mod.is_consistent)
            s, r = run_arm(BP, B, ScriptedModel(B, inconsistent=inconsistent), tok, part, args, sd, unmet=meter,
                           val=val)
            (sd / "hook.json").write_text(json.dumps({"describe_start": {}, "describe_end": meter.hook.describe(),
                                                      "meter": meter.summary(), "canary": None,
                                                      "rcx_modules_executed": dict(imp.executed),
                                                      "torch_version": None}, indent=1))
            write_outputs(sd, s, r, {"args": args, "shard": f"{i}/{SHARDS}", "DRY_RUN": True})
        hook = merge_shards(arm_dir, SHARDS)
        rows_a = [json.loads(x) for x in open(arm_dir / "plans.jsonl", encoding="utf-8")]
        if shard_check:  # D-50 property on this code path: shard composition does not change any outcome
            meter_u = HookMeter(Stub(**HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE, inst)
            _, rows_u = run_arm(BP, B, ScriptedModel(B, inconsistent=inconsistent), tok, inst, args,
                                out / "a_stub_beam_unsharded", unmet=meter_u, val=False)
            ua = {r["id"]: (r["plan"], r["verified"], r["rounds"], r["gen_tokens"]) for r in rows_u}
            sa = {r["id"]: (r["plan"], r["verified"], r["rounds"], r["gen_tokens"]) for r in rows_a}
            res["shard_invariance"] = {"identical": ua == sa, "n": len(ua),
                                       "hook_calls_sharded_vs_unsharded": [hook["calls"], meter_u.calls]}
        # pass-through check of the meter on a few states
        probe = Stub(**HOOK_KWARGS)
        pm = HookMeter(Stub(**HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE, inst)
        res["meter_pass_through"] = all(pm(it["init"], it["goal"]) == probe(it["init"], it["goal"]) for it in inst)
        rep = analyze_pair(rows_c, rows_a, n_expected=len(inst), hook=hook, instances=inst, B=B, hook_mod=hook_mod,
                           ctrl_caches=[out / "control_beam" / "fcache.jsonl"],
                           arm_caches=sorted(arm_dir.glob("shard_*/fcache.jsonl")),
                           arm_caches_by_shard={i: arm_dir / f"shard_{i}of{SHARDS}" / "fcache.jsonl"
                                                for i in range(SHARDS)},
                           labels=("control", "a_stub"))
        rep["void_reasons_if_this_were_the_screen"] = void_reasons(hook)
        # freeze-record preview: what bw_seeds.freeze will write for R0 and what this runner would freeze (in memory)
        r0_like = {"args": dict(b2["args"], pa="runs/BW_Pa_v2_s1/adapter", fwd_adapter="runs/BW_Fdelta_s0/adapter"),
                   "pa_sha256": b2["pa_sha256"], "fwd_sha256": b2["fwd_sha256"]}
        check_r0_is_b2(r0_like, b2)
        preview = build_freeze_record(r0_like, "<sha of FREEZE_I7_R0_beam.json>", pins={}, pre={"drop_ids": [],
                                      "n_after_drop": N_FRESH}, prereg={}, addendum={}, seeds_outputs_present=[],
                                      runner_sha=sha256_file(Path(__file__)), now="DRY-RUN")
        res["freeze_preview_args_equal_r0"] = preview["args"] == r0_like["args"]
    finally:
        unload_rcx(imp)
    check_pin(ROOT / PLANNER_REL, PLANNER_SHA256, PLANNER_REL)
    heavy_after = {m: m in sys.modules for m in ("torch", "transformers", "peft")}
    newly = [m for m in heavy_after if heavy_after[m] and not heavy_before[m]]
    report = {"DRY_RUN": True, "note": "stub F/proposer and a stub GNN forward (0.0): numbers say nothing about B4",
              "args": args, "instances": [{"id": it["id"], "n": len(it["objs"])} for it in inst],
              "pins_verified": pins, "rcx_modules_executed": dict(imp.executed),
              "heavy_modules_newly_imported": newly,
              "rcx_pycache_unchanged": _pycache_snapshot(rcx_root / "rcx") == rcx_pyc_before,
              "control_summary": summ_c, "checks": res, "analysis": rep}
    (out / "DRY_RUN_REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    if newly:
        raise RuntimeError(f"the dry run imported {newly}")
    return report


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", nargs="?", choices=["dry-run", "oc", "precheck", "final-diff", "freeze", "canary", "run",
                                                 "run-shard", "analyze"])
    ap.add_argument("--dry-run", action="store_true", help="same as the 'dry-run' stage")
    ap.add_argument("--out", help="output directory (dry-run, oc, precheck)")
    ap.add_argument("--no-val", action="store_true", help="dry-run: skip VAL")
    ap.add_argument("--inconsistent", type=float, default=0.05,
                    help="dry-run: stub-F share of inconsistent legal answers")
    ap.add_argument("--shard", default="", help="run-shard: i/4")
    ap.add_argument("--arm", default="a", choices=sorted(ARMS), help="run-shard: a, rerun_c or rerun_a")
    ap.add_argument("--rerun", action="store_true", help="run/analyze: the pre-registered re-run after a fault")
    ap.add_argument("--prereg")
    ap.add_argument("--addendum")
    ap.add_argument("--draft")
    ap.add_argument("--final")
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--allow-after-r0", action="store_true")
    a = ap.parse_args(argv)
    stage = "dry-run" if a.dry_run else a.stage
    if stage is None:
        ap.error("give a stage or --dry-run")
    if stage == "dry-run":
        rep = dry_run(Path(a.out) if a.out else DRY_OUT, val=not a.no_val, inconsistent=a.inconsistent)
        d = rep["analysis"]
        print(json.dumps({"out": str(Path(a.out) if a.out else DRY_OUT), "paired": d["paired"],
                          "status": d["status"], "round1_identity": d["round1_identity"]["holds"],
                          "hook": {k: d["hook"][k] for k in ("calls", "penalty_calls", "penalty_share_calls",
                                                             "penalty_split_calls")},
                          "certificate_holds": d["certificate_c_eq_cpen_on_control"]["holds"],
                          "arm_unpenalised": d["arm_unpenalised_inconsistent_report_only"]["unpenalised"],
                          "f_cache_cross_check": d["f_cache_cross_check"], "checks": rep["checks"],
                          "heavy_modules_newly_imported": rep["heavy_modules_newly_imported"],
                          "rcx_pycache_unchanged": rep["rcx_pycache_unchanged"]}, indent=1))
        return 0
    if stage == "oc":
        t = oc_table()
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "b4s_oc.json").write_text(json.dumps(t, indent=1))
        rules = list(OC_RULES)
        print("%-32s" % "scenario (n = 40)" + "".join("%11s" % r.split(":")[0][:10] for r in rules) + "  B4-P1 n110")
        for row in t["rows"]:
            print("%-32s" % row["scenario"] + "".join("%11.3f" % row["n40"][r] for r in rules)
                  + "  %9.3f" % row["b4_primary_power_n110_alpha0.025"])
        print(json.dumps(t["min_arm_only_for_go_by_control_only"]))
        print(json.dumps(t["d_grid"], indent=1))
        return 0
    if stage == "precheck":
        if not a.out:
            ap.error("precheck needs --out (the registered copy is written by the freeze stage)")
        pre = precheck()
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "PRECHECK.json").write_text(json.dumps(pre, indent=1))
        print(json.dumps({k: pre[k] for k in ("e07_rows_goal_key_agreement", "gnn_train_val_goal_classes",
                                              "gnn_train_val_class_sizes", "seeds_fresh_n_le_8",
                                              "seeds_fresh_in_e07_test_classes", "drop_ids", "n_after_drop")},
                         indent=1))
        return 0
    if stage == "final-diff":
        if not (a.draft and a.final):
            ap.error("final-diff needs --draft and --final")
        r = finalisation_diff(Path(a.draft).read_text(encoding="utf-8"), Path(a.final).read_text(encoding="utf-8"))
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    return {"freeze": stage_freeze, "canary": stage_canary, "run": stage_run, "run-shard": stage_run_shard,
            "analyze": stage_analyze}[stage](a)


if __name__ == "__main__":
    sys.exit(main())
