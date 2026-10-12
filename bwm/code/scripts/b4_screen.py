"""B4 futility screen runner -- DRAFT (research/PREREG_B4_SCREEN_DRAFT.md; D-75 decision 1; D-76a).

One arm, a = RCX's hybrid hook `LearnedUnmet.from_file(gnn_s20261276.pt, invalid_policy="fallback", hybrid_lambda=1.0)`
(rcx/bw/b4_hook.py, read-only, every executed file sha-pinned) passed as `unmet=` to the pinned B2 `plan_beam`
(bwmn/blocks_planner.py 18a5f22b..., imported read-only, never patched) with item 7's R0 S-beam arguments
(runs/FREEZE_I7_R0_beam.json = FREEZE_B2_beam args) on seeds_fresh-40. The control is item 7's own R0 beam run
(runs/BW_SEEDS/R0_beam): same F, Pa, eval_seed, CRN seeds and shard partition (index % 4); paired per instance.

  python -B scripts/b4_screen.py --dry-run          # CPU; no model, no torch; stub F/proposer; outputs to scratch
  python -B scripts/b4_screen.py oc [--out DIR]     # CPU; operating characteristics of the futility rule
  python -B scripts/b4_screen.py precheck --out DIR # CPU; GNN train/val goal classes vs seeds_fresh (prereg 4.3)
  python -B scripts/b4_screen.py freeze --prereg P --addendum A   # CPU; runs/FREEZE_B4S_a_beam.json (once)
  python -B scripts/b4_screen.py run                # GPU; 4 resumable shards, each a run-shard subprocess
  python -B scripts/b4_screen.py analyze            # CPU; runs/BW_B4_SCREEN/B4S_REPORT.{json,md}

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
R0_FREEZE_REL = "runs/FREEZE_I7_R0_beam.json"   # written by scripts/bw_seeds.py freeze (not yet at drafting)
R0_OUT_REL = "runs/BW_SEEDS/R0_beam"
FREEZE_REL = "runs/FREEZE_B4S_a_beam.json"
OUT_REL = "runs/BW_B4_SCREEN"
ARM = "a_beam"
SHARDS = 4                                       # = bw_seeds.BEAM_SHARDS: the same index % 4 partition as R0
F_BATCH_TOKENS, PROP_BATCH_TOKENS = 5000, 6000   # blocks_eval.py defaults; outputs do not depend on them (D-48c)

RCX_ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary")
RCX_CODE_PINS = {  # every RCX file the hook import executes (rel. to RCX_ROOT)
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
HOOK_CLASS = "rcx.bw.b4_hook.LearnedUnmet"
HOOK_KWARGS = {"invalid_policy": "fallback", "hybrid_lambda": 1.0}                       # D-76a: exactly this call

# ----------------------------------------------------------------------------- the registered rule (prereg section 5)

GO_MIN_NET = 2                        # BWM decision 2026-10-11 (D-77): net >= 2, sign p reported, not gating
GO_MAX_P_ONE_SIDED = Fraction(1, 5)   # dyadic p-values never equal 1/5 exactly: no boundary ties
PENALTY_SHARE_MAX = 0.05              # D-76a attribution rule (call level)
RULE_TEXT = ("GO iff (a_only - R0_only) >= 2; otherwise NO-GO (net <= 1 is always NO-GO). The exact one-sided "
             "sign-test p and the two-sided McNemar p are reported but do not gate (D-77). If more than 5% of hook calls return a penalty-band value (>= PENALTY), a GO is not credited to "
             "the learned critic alone and c+pen is mandatory in the full B4 (D-76a).")
WORDING = {
    "GO": ("B4 screen GO: arm (a), the RCX hybrid hook, passed the pre-set futility rule on seeds_fresh-40, paired vs "
           "item 7's R0 beam. This unlocks the co-signed B4 primary (fresh seed-20261410 set, n = 110). It is not a "
           "claim that the hook improves planning."),
    "NO-GO": ("B4 arm (a) not supported at screen (seeds_fresh-40, paired vs R0, pre-set futility rule). The full B4 "
              "is not run; B6 follows (D-75.1)."),
}
ATTRIBUTION_NOTE = (" Not credited to the learned critic alone: more than 5% of hook calls hit the invalid-state "
                    "penalty, so c+pen is mandatory in the full B4 (D-76a).")

DRY_SEED = 7411                          # dry-run only; never a data or evaluation seed
DRY_SIZES = (5, 5, 6, 6, 7, 7, 8, 8)
DRY_OUT = Path("C:/Users/clclc/AppData/Local/Temp/claude/bwm_d75_scratch/C/dryrun")

_MISSING = object()


class PinError(RuntimeError):
    """A pinned file does not match its sha256, or a module would be imported from an unpinned source."""


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


def futility_decision(g: int, l: int) -> dict:
    """The registered screen rule on the paired table: g = arm-only successes, l = control-only successes."""
    p1 = sign_p_one_sided(g, l)
    go = (g - l) >= GO_MIN_NET
    why = ("net <= 0 (D-76a NO-GO)" if g - l <= 0 else
           f"net {g - l} < {GO_MIN_NET}" if not go else f"net {g - l} >= {GO_MIN_NET}")
    return {"arm_only": g, "control_only": l, "net": g - l, "p_one_sided_sign": float(p1),
            "p_mcnemar_two_sided": mcnemar_two_sided(g, l), "decision": "GO" if go else "NO-GO", "go": go,
            "because": why}


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
    "earlier draft: net>=3 & p1<=0.20": lambda g, l: g - l >= 3 and sign_p_one_sided(g, l) <= Fraction(1, 5),
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
            "rows": rows, "min_arm_only_for_go_by_control_only": min_gains}


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
    """(B, BP) for the GPU run: the normal import (as blocks_eval.py does for R0), sha-checked before and after."""
    check_pin(ROOT / PLANNER_REL, PLANNER_SHA256, PLANNER_REL)
    check_pin(ROOT / BLOCKS_REL, BLOCKS_SHA256, BLOCKS_REL)
    from bwmn import blocks as B
    from bwmn import blocks_planner as BP
    if Path(BP.__file__).resolve() != (ROOT / PLANNER_REL).resolve():
        raise PinError(f"imported blocks_planner {BP.__file__} is not {ROOT / PLANNER_REL}")
    check_pin(BP.__file__, PLANNER_SHA256, PLANNER_REL)
    if getattr(BP.lm, "_b4s_stub", False):
        raise PinError("the GPU run imported a planner bound to the dry-run stub lm")
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
    and models.py are imported lazily by LearnedUnmet's constructor (GPU run only). All pins are checked up front."""
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
                    "counters": dict(self.counters), "reasons": dict(sorted(self.reasons.items())),
                    "bwm_planner_pin": hook_mod.BWM_PLANNER_PIN}

    return StubModelUnmet


def verify_hook(hook, rcx_root=RCX_ROOT) -> dict:
    """GPU run: the constructed hook is exactly the D-76a call on the pinned files."""
    d = hook.describe()
    want = {"model_sha256": GNN_SHA256, "sidecar_sha256": GNN_SIDECAR_SHA256, "config_source": "sidecar json 'config'",
            "invalid_policy": HOOK_KWARGS["invalid_policy"], "hybrid_lambda": HOOK_KWARGS["hybrid_lambda"],
            "round_to": None, "bwm_planner_pin": PLANNER_SHA256}
    bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
    for rel in RCX_D76_FILES:
        if d["code_sha256"].get(rel) != RCX_CODE_PINS[rel]:
            bad[rel] = (d["code_sha256"].get(rel), RCX_CODE_PINS[rel])
    if Path(d["model_file"]).resolve() != (Path(rcx_root) / GNN_REL).resolve():
        bad["model_file"] = (d["model_file"], str(Path(rcx_root) / GNN_REL))
    if not bool(hook):
        bad["truthiness"] = (False, True)
    if bad:
        raise PinError(f"hook differs from the registered D-76a call: {bad}")
    return d


class HookMeter:
    """Truthy pass-through around the unmet hook (plan_beam binds `unmet or default`): returns the hook's value
    unchanged and counts, per goal, the calls, the penalty-band calls (value >= PENALTY) and failsafe-band calls, plus
    the seconds spent inside the hook."""

    def __init__(self, hook, penalty: float, failsafe: float, instances=()):
        self.hook, self.penalty, self.failsafe = hook, float(penalty), float(failsafe)
        self.owner = {}
        for it in instances:
            g = frozenset(it["goal"])
            self.owner[g] = it["id"] if g not in self.owner else self.owner[g] + "|" + it["id"]
        self.per = {}
        self.calls = self.penalty_calls = self.failsafe_calls = 0
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
        c = self.per.setdefault(k, [0, 0, 0])
        c[0] += 1
        self.calls += 1
        if v >= self.penalty:
            c[1] += 1
            self.penalty_calls += 1
        if v >= self.failsafe:
            c[2] += 1
            self.failsafe_calls += 1
        return v

    def summary(self) -> dict:
        return {"calls": self.calls, "penalty_calls": self.penalty_calls, "failsafe_calls": self.failsafe_calls,
                "seconds": round(self.seconds, 3),
                "per_instance": {k: {"calls": v[0], "penalty_calls": v[1], "failsafe_calls": v[2]}
                                 for k, v in sorted(self.per.items())}}


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


def merge_shards(out: Path, k: int):
    """bw_run_tests.merge (the function item 7 uses for R0) + the per-shard hook records."""
    _load_script("bw_run_tests").merge(out, k)
    hook = {"calls": 0, "penalty_calls": 0, "failsafe_calls": 0, "seconds": 0.0, "per_instance": {},
            "counters": Counter(), "reasons": Counter(), "shards": []}
    for i in range(k):
        f = out / f"shard_{i}of{k}" / "hook.json"
        if not f.exists():
            continue
        h = json.loads(f.read_text())
        m = h["meter"]
        for key in ("calls", "penalty_calls", "failsafe_calls", "seconds"):
            hook[key] += m[key]
        hook["per_instance"].update(m["per_instance"])
        hook["counters"].update(h["describe_end"].get("counters", {}))
        hook["reasons"].update(h["describe_end"].get("reasons", {}))
        hook["shards"].append(f"shard_{i}of{k}")
    hook["counters"], hook["reasons"] = dict(hook["counters"]), dict(hook["reasons"])
    hook["seconds"] = round(hook["seconds"], 3)
    (out / "hook_merged.json").write_text(json.dumps(hook, indent=1))
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


def analyze_pair(ctrl_rows, arm_rows, *, n_expected: int, drop_ids=(), hook=None, instances=None, B=None,
                 hook_mod=None, ctrl_caches=(), arm_caches=(), labels=("R0", "a")) -> dict:
    """The registered screen analysis (prereg section 5-6) on two plans.jsonl row lists."""
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
    dec = futility_decision(g, l)
    rep = {"n_paired": len(ids), "dropped_by_precheck": sorted(drop_ids, key=_id_key),
           "paired": {f"{labels[1]}_only": g, f"{labels[0]}_only": l, "both": both, "neither": len(ids) - g - l - both},
           "decision": dec, "rule": RULE_TEXT}
    wording = WORDING[dec["decision"]]
    if hook is not None:
        calls = hook.get("calls", 0)
        share = hook.get("penalty_calls", 0) / calls if calls else 0.0
        attributable = share <= PENALTY_SHARE_MAX
        rep["hook"] = {**{k: v for k, v in hook.items() if k != "per_instance"}, "penalty_share_calls": round(share, 6),
                       "penalty_share_max": PENALTY_SHARE_MAX,
                       "go_credited_to_critic": bool(dec["go"] and attributable)}
        ev = hook.get("counters", {}).get("evaluations")
        if ev:
            rep["hook"]["penalty_share_evaluations"] = round(hook["counters"].get("fallbacks", 0) / ev, 6)
        if dec["go"] and not attributable:
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
        fails = [i for i in ids if not _ok(c[i])]
        dist = Counter()
        stalled1 = []
        for i in fails:
            u = unmet_after(B, by[i], c[i]["plan"])
            dist["invalid_prefix" if u is None else str(u)] += 1
            if u == 1:
                stalled1.append(i)
        rep["control_failures_unmet_at_submitted_prefix"] = {
            "distribution": dict(sorted(dist.items())), "stalled_at_1": len(stalled1),
            f"{labels[1]}_solves_among_stalled_at_1": sum(_ok(a[i]) for i in stalled1), "ids_stalled_at_1": stalled1}
    if ctrl_caches and hook_mod is not None and B is not None:
        rep["certificate_c_eq_cpen_on_control"] = certificate_c_eq_cpen(ctrl_caches, B, hook_mod)
    if ctrl_caches and arm_caches:
        rep["f_cache_cross_check"] = cross_cache(ctrl_caches, arm_caches)
    return rep


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
            "drop_ids": drop, "n_after_drop": len(sf) - len(drop)}


# ----------------------------------------------------------------------------- freeze (CPU; once)

def current_pins(root=ROOT, rcx_root=RCX_ROOT) -> dict:
    """Check every pinned input; returns {name: sha256}. Raises PinError on any mismatch."""
    shas = {PLANNER_REL: check_pin(root / PLANNER_REL, PLANNER_SHA256, PLANNER_REL),
            BLOCKS_REL: check_pin(root / BLOCKS_REL, BLOCKS_SHA256, BLOCKS_REL),
            B2_BEAM_FREEZE_REL: check_pin(root / B2_BEAM_FREEZE_REL, B2_BEAM_FREEZE_SHA256, B2_BEAM_FREEZE_REL),
            ITEM7_PREREG_REL: check_pin(root / ITEM7_PREREG_REL, ITEM7_PREREG_SHA256, ITEM7_PREREG_REL),
            SEEDS_FRESH_REL: check_pin(root / SEEDS_FRESH_REL, SEEDS_FRESH_SHA256, SEEDS_FRESH_REL)}
    for rel, want in RCX_CODE_PINS.items():
        shas["rcx:" + rel] = check_pin(rcx_root / rel, want, "RCX " + rel)
    shas["rcx:" + GNN_REL] = check_pin(rcx_root / GNN_REL, GNN_SHA256, "RCX " + GNN_REL)
    shas["rcx:" + GNN_SIDECAR_REL] = check_pin(rcx_root / GNN_SIDECAR_REL, GNN_SIDECAR_SHA256, "RCX " + GNN_SIDECAR_REL)
    shas["rcx:" + E07_INSTANCES_REL] = check_pin(rcx_root / E07_INSTANCES_REL, E07_INSTANCES_SHA256,
                                                 "RCX " + E07_INSTANCES_REL)
    for rel in ("bwmn/lm.py", "bwmn/__init__.py", "scripts/blocks_eval.py", "scripts/bw_seeds.py",
                "scripts/bw_run_tests.py"):
        if (root / rel).exists():
            shas["recorded:" + rel] = sha256_file(root / rel)  # recorded, not pinned
    return shas


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
                        r0_outputs_present: list, runner_sha: str, now: str) -> dict:
    return {"what": "B4 futility screen, arm a: RCX hybrid hook as unmet= in item 7's R0 S-beam configuration",
            "args": dict(r0_rec["args"]),
            "unmet_hook": {"class": HOOK_CLASS, "constructor": "from_file", "model_file": GNN_REL,
                           "kwargs": dict(HOOK_KWARGS), "rcx_root": str(RCX_ROOT)},
            "control": {"system": "R0", "freeze": R0_FREEZE_REL, "freeze_sha256": r0_freeze_sha, "out": R0_OUT_REL},
            "set": {"name": "seeds_fresh", "path": SEEDS_FRESH_REL, "sha256": SEEDS_FRESH_SHA256, "n": N_FRESH,
                    "shards": SHARDS},
            "pa_sha256": r0_rec["pa_sha256"], "fwd_sha256": r0_rec["fwd_sha256"],
            "pins": pins, "precheck": {"drop_ids": pre["drop_ids"], "n_after_drop": pre["n_after_drop"]},
            "rule": {"go_min_net": GO_MIN_NET, "go_max_p_one_sided": str(GO_MAX_P_ONE_SIDED),
                     "penalty_share_max": PENALTY_SHARE_MAX, "text": RULE_TEXT},
            "prereg": prereg, "addendum": addendum, "item7_prereg_sha256": ITEM7_PREREG_SHA256,
            "runner": "scripts/b4_screen.py", "runner_sha256": runner_sha,
            "r0_outputs_present_at_freeze": r0_outputs_present, "written": now}


def r0_outputs(root=ROOT) -> list:
    """Every file under R0's beam output directory (verifier fix: not only summary/plans; a running or interrupted
    shard's progress.txt already shows per-round solve counts, and its fcache.jsonl is an R0 output)."""
    d = root / R0_OUT_REL
    if not d.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in d.rglob("*") if p.is_file())


def stage_freeze(a, root=ROOT, rcx_root=RCX_ROOT) -> int:
    frz = root / FREEZE_REL
    if frz.exists():
        sys.exit(f"{frz} exists (written once)")
    r0f = root / R0_FREEZE_REL
    if not r0f.exists():
        sys.exit(f"{r0f} missing: run `scripts/bw_seeds.py freeze` first (the screen is frozen right after item 7)")
    present = r0_outputs(root)
    if present and not a.allow_after_r0:
        sys.exit(f"R0 beam outputs already exist ({present[:3]}...): the screen must be frozen before any R0 output "
                 "(pass --allow-after-r0 to record a late freeze explicitly)")
    docs = {}
    for key, p in (("prereg", a.prereg), ("addendum", a.addendum)):
        if not p:
            sys.exit(f"--{key} is required")
        p = Path(p)
        if "DRAFT" in p.name.upper() and not a.allow_draft:
            sys.exit(f"{p} is a draft: freeze against the final, hash-recorded text (or --allow-draft)")
        docs[key] = {"path": str(p), "sha256": sha256_file(p)}
    pins = current_pins(root, rcx_root)
    r0_rec = json.loads(r0f.read_text())
    check_r0_is_b2(r0_rec, json.loads((root / B2_BEAM_FREEZE_REL).read_text()))
    for key, rel in (("pa_sha256", r0_rec["args"]["pa"]), ("fwd_sha256", r0_rec["args"]["fwd_adapter"])):
        check_pin(root / rel / "adapter_model.safetensors", r0_rec[key], f"R0 {key}")
    pre = precheck(root / SEEDS_FRESH_REL, rcx_root / E07_INSTANCES_REL)
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    (out / "PRECHECK.json").write_text(json.dumps(pre, indent=1))
    rec = build_freeze_record(r0_rec, sha256_file(r0f), pins=pins, pre=pre, prereg=docs["prereg"],
                              addendum=docs["addendum"], r0_outputs_present=present,
                              runner_sha=sha256_file(Path(__file__)), now=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    frz.write_text(json.dumps(rec, indent=1))
    print(f"wrote {frz} (precheck drops {pre['drop_ids']}; n = {pre['n_after_drop']})")
    return 0


def load_and_verify_freeze(root=ROOT, rcx_root=RCX_ROOT) -> dict:
    rec = json.loads((root / FREEZE_REL).read_text())
    r0f = root / R0_FREEZE_REL
    if sha256_file(r0f) != rec["control"]["freeze_sha256"]:
        raise PinError("FREEZE_I7_R0_beam.json changed since the screen freeze")
    if json.loads(r0f.read_text())["args"] != rec["args"]:
        raise PinError("screen args != R0 beam args")
    if sha256_file(Path(__file__)) != rec["runner_sha256"]:
        raise PinError("scripts/b4_screen.py changed since the freeze")
    now = current_pins(root, rcx_root)
    bad = {k: (now.get(k), v) for k, v in rec["pins"].items() if not k.startswith("recorded:") and now.get(k) != v}
    if bad:
        raise PinError(f"pinned inputs changed since the freeze: {bad}")
    for key, rel in (("pa_sha256", rec["args"]["pa"]), ("fwd_sha256", rec["args"]["fwd_adapter"])):
        check_pin(root / rel / "adapter_model.safetensors", rec[key], key)
    return rec


# ----------------------------------------------------------------------------- GPU stages

def stage_run(a) -> int:
    load_and_verify_freeze()
    arm = ROOT / OUT_REL / ARM
    if (arm / "summary.json").exists():
        print(f"[b4s] {arm}/summary.json exists: done")
        return 0
    for i in range(SHARDS):
        if (arm / f"shard_{i}of{SHARDS}" / "summary.json").exists():
            continue
        print(f"[b4s] run shard {i}/{SHARDS}", flush=True)
        rc = subprocess.call([sys.executable, "-B", str(Path(__file__)), "run-shard", "--shard", f"{i}/{SHARDS}"],
                             cwd=str(ROOT))
        if rc != 0:
            print(f"[b4s] shard {i} exited {rc}", flush=True)
            return rc
    merge_shards(arm, SHARDS)
    print("[b4s] done", (arm / "summary.json").read_text()[:400], flush=True)
    return 0


def stage_run_shard(a) -> int:
    rec = load_and_verify_freeze()
    si, sk = map(int, a.shard.split("/"))
    if sk != SHARDS or not 0 <= si < sk:
        sys.exit(f"--shard must be i/{SHARDS}")
    dst = ROOT / OUT_REL / ARM / f"shard_{si}of{sk}"
    if (dst / "summary.json").exists():
        sys.exit(f"{dst}/summary.json exists: the screen arm runs once")
    args = rec["args"]
    inst = load_json_set(ROOT / SEEDS_FRESH_REL, "seeds_fresh", SEEDS_FRESH_SHA256)
    if len(inst) != N_FRESH:
        raise PinError(f"seeds_fresh has {len(inst)} instances, expected {N_FRESH}")
    inst = [it for j, it in enumerate(inst) if j % sk == si]
    B, BP = import_planner_real()
    import torch
    from bwmn import lm
    from peft import PeftModel
    base, tok = lm.load_core("titan-sparse")
    model = PeftModel.from_pretrained(base, str(ROOT / args["fwd_adapter"]), adapter_name="fwd", is_trainable=False)
    model.load_adapter(str(ROOT / args["pa"]), adapter_name="pa", is_trainable=False)
    model.eval()
    hook_mod, _imp = load_rcx_hook_module()
    cuda_before = (torch.cuda.is_available, torch.cuda.device_count, os.environ.get("CUDA_VISIBLE_DEVICES"))
    hook = hook_mod.LearnedUnmet.from_file(str(RCX_ROOT / GNN_REL), **HOOK_KWARGS)
    if (torch.cuda.is_available, torch.cuda.device_count, os.environ.get("CUDA_VISIBLE_DEVICES")) != cuda_before:
        raise RuntimeError("the hook import changed the host's CUDA view")
    d0 = verify_hook(hook)
    meter = HookMeter(hook, hook_mod.PENALTY, hook_mod.FAILSAFE, inst)
    summ, rows = run_arm(BP, B, model, tok, inst, args, dst, unmet=meter, val=True)
    d1 = verify_hook(hook)
    check_pin(BP.__file__, PLANNER_SHA256, PLANNER_REL)
    (dst / "hook.json").write_text(json.dumps({"describe_start": d0, "describe_end": d1, "meter": meter.summary()},
                                              indent=1, default=str))
    summ["hook_calls"], summ["hook_penalty_calls"] = meter.calls, meter.penalty_calls
    write_outputs(dst, summ, rows, {"args": args, "shard": a.shard, "freeze": FREEZE_REL})
    print(json.dumps(summ), flush=True)
    return 0


def stage_analyze(a) -> int:
    rec = json.loads((ROOT / FREEZE_REL).read_text())
    out, arm = ROOT / OUT_REL, ROOT / OUT_REL / ARM
    r0 = ROOT / R0_OUT_REL
    inputs = {}
    for p in [r0 / "plans.jsonl", r0 / "summary.json", arm / "plans.jsonl", arm / "summary.json",
              arm / "hook_merged.json", out / "PRECHECK.json"] + sorted(r0.glob("shard_*/fcache.jsonl")) + \
            sorted(arm.glob("shard_*/fcache.jsonl")):
        inputs[str(p.relative_to(ROOT))] = sha256_file(p)  # hashed before anything is parsed
    (out / "ANALYSIS_INPUTS.json").write_text(json.dumps(inputs, indent=1))
    hook_mod, _imp = load_rcx_hook_module()
    from bwmn import blocks as B
    check_pin(B.__file__, BLOCKS_SHA256, BLOCKS_REL)
    load = lambda p: [json.loads(x) for x in open(p, encoding="utf-8")]
    pre = json.loads((out / "PRECHECK.json").read_text())
    rep = analyze_pair(load(r0 / "plans.jsonl"), load(arm / "plans.jsonl"), n_expected=N_FRESH,
                       drop_ids=pre["drop_ids"], hook=json.loads((arm / "hook_merged.json").read_text()),
                       instances=load_json_set(ROOT / SEEDS_FRESH_REL, "seeds_fresh", SEEDS_FRESH_SHA256), B=B,
                       hook_mod=hook_mod, ctrl_caches=sorted(r0.glob("shard_*/fcache.jsonl")),
                       arm_caches=sorted(arm.glob("shard_*/fcache.jsonl")))
    rep["cost"] = {k: {x: json.loads((d / "summary.json").read_text())["summary"].get(x)
                       for x in ("f_queries_per_instance", "proposer_tokens_per_instance", "wall_s")}
                   for k, d in (("R0", r0), ("a", arm))}
    rep["meta"] = {"freeze_sha256": sha256_file(ROOT / FREEZE_REL), "prereg": rec["prereg"],
                   "runner_sha256": sha256_file(Path(__file__)), "inputs": inputs,
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (out / "B4S_REPORT.json").write_text(json.dumps(rep, indent=1))
    (out / "B4S_REPORT.md").write_text(render_md(rep))
    print(rep["wording"])
    return 0


def render_md(rep: dict, title="B4 futility screen (arm a vs R0, seeds_fresh)") -> str:
    d, p = rep["decision"], rep["paired"]
    L = [f"# {title}", "", f"**Decision: {d['decision']}** ({d['because']}). {rep['wording']}", "",
         f"Rule: {rep['rule']}", "", "| | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in p.items()]
    L += [f"| net | {d['net']} |", f"| one-sided sign p | {d['p_one_sided_sign']:.4g} |",
          f"| two-sided exact McNemar p | {d['p_mcnemar_two_sided']:.4g} |", f"| n paired | {rep['n_paired']} |", ""]
    L += ["| arm | success | CP 95% CI | failures | mean tokens | mean rounds |", "|---|---|---|---|---|---|"]
    for k, v in rep["arms"].items():
        L.append(f"| {k} | {v['success']} | {v['ci95']} | {v['failure_taxonomy']} | {v['mean_gen_tokens']} | "
                 f"{v['mean_rounds']} |")
    if "hook" in rep:
        h = rep["hook"]
        L += ["", f"Hook: {h.get('calls')} calls, penalty-band share {h['penalty_share_calls']} (max "
                  f"{h['penalty_share_max']}); GO credited to the critic: {h['go_credited_to_critic']}."]
    return "\n".join(L) + "\n"


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
            meter = HookMeter(Stub(**HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE, part)
            s, r = run_arm(BP, B, ScriptedModel(B, inconsistent=inconsistent), tok, part, args, sd, unmet=meter,
                           val=val)
            (sd / "hook.json").write_text(json.dumps({"describe_start": {}, "describe_end": meter.hook.describe(),
                                                      "meter": meter.summary()}, indent=1))
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
                           arm_caches=sorted(arm_dir.glob("shard_*/fcache.jsonl")), labels=("control", "a_stub"))
        # freeze-record preview: what bw_seeds.freeze will write for R0 and what this runner would freeze (in memory)
        r0_like = {"args": dict(b2["args"], pa="runs/BW_Pa_v2_s1/adapter", fwd_adapter="runs/BW_Fdelta_s0/adapter"),
                   "pa_sha256": b2["pa_sha256"], "fwd_sha256": b2["fwd_sha256"]}
        check_r0_is_b2(r0_like, b2)
        preview = build_freeze_record(r0_like, "<sha of FREEZE_I7_R0_beam.json>", pins={}, pre={"drop_ids": [],
                                      "n_after_drop": N_FRESH}, prereg={}, addendum={}, r0_outputs_present=[],
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
    ap.add_argument("stage", nargs="?", choices=["dry-run", "oc", "precheck", "freeze", "run", "run-shard", "analyze"])
    ap.add_argument("--dry-run", action="store_true", help="same as the 'dry-run' stage")
    ap.add_argument("--out", help="output directory (dry-run, oc, precheck)")
    ap.add_argument("--no-val", action="store_true", help="dry-run: skip VAL")
    ap.add_argument("--inconsistent", type=float, default=0.05,
                    help="dry-run: stub-F share of inconsistent legal answers")
    ap.add_argument("--shard", default="", help="run-shard: i/4")
    ap.add_argument("--prereg")
    ap.add_argument("--addendum")
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
                          "decision": d["decision"]["decision"], "hook": {k: d["hook"][k] for k in
                          ("calls", "penalty_calls", "penalty_share_calls")},
                          "certificate_holds": d["certificate_c_eq_cpen_on_control"]["holds"],
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
        return 0
    if stage == "precheck":
        if not a.out:
            ap.error("precheck needs --out (the registered copy is written by the freeze stage)")
        pre = precheck()
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "PRECHECK.json").write_text(json.dumps(pre, indent=1))
        print(json.dumps({k: pre[k] for k in ("e07_rows_goal_key_agreement", "gnn_train_val_goal_classes",
                                              "gnn_train_val_class_sizes", "seeds_fresh_n_le_8", "drop_ids",
                                              "n_after_drop")}, indent=1))
        return 0
    return {"freeze": stage_freeze, "run": stage_run, "run-shard": stage_run_shard, "analyze": stage_analyze}[stage](a)


if __name__ == "__main__":
    sys.exit(main())
