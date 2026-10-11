"""Joint RCX-BWM experiment B4, arm (a): an RCX RelGNN value model as the `unmet=` hook of BWM's frozen `plan_beam`.

Hook contract, extracted read-only from BWM's bwmn/blocks_planner.py (sha256 18a5f22b...; see `planner_hook_usage()`,
which re-derives every point below from the file text with `ast` and never imports it):
  * Binding (line 280): `unmet = unmet or (lambda st, g: len(g - st))`. The hook is chosen by TRUTHINESS, so a falsy
    hook object would be silently replaced by the default. LearnedUnmet defines __bool__ -> True and has no __len__.
  * Signature: plan_beam(instances, proposer, fwd, *, ..., goal_met=None, unmet=None); `unmet` is keyword-only.
  * Calls: always positional, two arguments, one state per call, no keywords and no batching:
      line 286  unmet(it["init"], it["goal"])          once per instance, on the true initial state, before search;
      line 319  um = [unmet(st, e["goal"]) for st in traj]
                on every imagined state F produced along every rolled-out candidate (traj = the states after each
                step F called legal; the rollout stops at the first illegal or unparseable step), including the
                candidates that reach the goal.
    `goal` is the instance's goal frozenset (BWM goals are ("on", x, y) facts only); `state` is whatever F's parser
    returned: a frozenset of tuples that may be inconsistent, contain unknown predicates/arity (delta format:
    tuple(fact_str.split())), or names that are not str (state format: JSON values).
  * Use of the return value (lower is closer): min over the trajectory with key (um[t], t) (earliest best state);
    children sorted by (u, prefix length, candidate rank); the running best is replaced when (u, ln) < best[:2]. Only
    comparisons are applied: no arithmetic, no int(), no serialisation. The default returns an int; a float is
    accepted. NaN would make min/sort order undefined, so this hook only returns finite floats.
  * Solution detection does not use the hook (goal_met defaults to B.goal_met, i.e. goal <= state); the hook only ranks
    beam children and picks the best prefix submitted by unsolved instances.

Value of LearnedUnmet(state, goal) (always a finite Python float, never raises for any input):
  objs = sorted union of the block names in the state's and the goal's well-formed facts (plus the instance's blocks
  when a goal -> objs registry is given). For a consistent state this equals B.blocks_of(state), so the value is the
  E07/E08 LearnedH value (models.predict on encode.graph(state, goal, objs)), up to float rounding from batch
  composition and thread count.
  * consistent state      -> model value (+ hybrid_lambda * misplaced blocks), clipped to [-MODEL_CLIP, MODEL_CLIP];
                             optionally rounded to a multiple of `round_to`.
  * inconsistent state    -> invalid_policy "fallback" (default): PENALTY + misplaced_blocks(sanitised copy, goal);
                             invalid_policy "raw": model value on the well-formed facts as they are (encode.graph
                             accepts any edge multiset), with the same clip/round as above.
  * encode/model error, or a non-finite model output -> PENALTY + misplaced_blocks(sanitised copy, goal).
  * the fallback itself fails (e.g. the state is not iterable) -> FAILSAFE.
  Since model values are clipped to |v| <= MODEL_CLIP < PENALTY <= every fallback value < FAILSAFE, every state the
  model scores ranks strictly before every fallback state, and those rank before every failsafe state. Within the
  fallback band, states are ordered by misplaced blocks.

Inconsistency reasons (is_consistent; the state alone, plus the goal's blocks and, if given, the instance's blocks):
  malformed_fact (not a tuple, unknown predicate, wrong arity, non-str or empty name), self_on (on(x, x)),
  multiple_positions (a block with more than one of on/ontable/holding), no_position (a block of the state or goal
  with none), shared_support (two blocks on one block), multiple_held, holding_and_handempty, hand_unspecified (neither
  holding nor handempty), on_held_block, cycle (in 'on'), missing_clear (an uncovered, unheld block without clear),
  wrong_clear (clear on a covered or held block), unknown_block (only when the instance's blocks are known: a name
  outside them). Without a registry, a block name that is not in the goal is NOT an inconsistency: partial goals
  (PB-600 style) and on-only full goals (a goal tower of one block has no on-fact) both leave real blocks out of goal.
  A state is consistent iff it has no reason; this equals space.from_facts acceptance over the same objs, except that
  from_facts accepts a block standing on the held block (on_held_block), which BW dynamics can never reach.

Sanitised copy (deterministic): well-formed facts only; at most one held block (the smallest held name); every other
block, in sorted name order, takes its smallest on-support that is not held, not yet taken and does not close a cycle,
otherwise the table. The result is always a valid Blocksworld state; for a consistent state it is the state itself.

Host safety (the hook runs inside BWM's planner process, next to the GPU forward model F):
  * torch and rcx.bw.models are imported lazily (planner_hook_usage, is_consistent and sanitise need neither).
    models.py patches torch.cuda.is_available/device_count to report no device and may set CUDA_VISIBLE_DEVICES='';
    the import here undoes both (and restores sys.dont_write_bytecode, set by bwm_import), so the host's CUDA view is
    unchanged. The hook only ever creates CPU tensors.
  * Thread count: the hook's forward pass runs with torch.set_num_threads(HOOK_THREADS = 1) and restores the caller's
    value afterwards (the setting is process-wide). One thread makes every value bit-reproducible and keeps the hook
    within the single-CPU-lane rule. Not thread-safe: do not call one hook object from several Python threads.
  * Model construction runs under torch.random.fork_rng(devices=[]), so the host's CPU RNG stream is not consumed; the
    CUDA RNG is never touched.
  * BWM code is never imported or executed (bwm_import.load() is never called; it would refuse anyway when BWM's own
    `bwmn` package is loaded).

Determinism: each state is scored by its own single-state forward pass, so a value never depends on batch
composition, call order or cache history; `batch` is a loop over `__call__` and equals it exactly.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import fields
from pathlib import Path

_DWB = sys.dont_write_bytecode
from . import bwm_import as BI  # noqa: E402  (module only; load() is never called here)
from . import encode as EN  # noqa: E402
from . import space as SP  # noqa: E402

sys.dont_write_bytecode = _DWB  # bwm_import sets it process-wide; restore the host's value

PENALTY = 1000.0      # fallback offset: fallback = PENALTY + misplaced_blocks(sanitised copy, goal) >= 1000
MODEL_CLIP = 500.0    # model values are clipped to [-500, 500]; never binds for plausible h (h* <= 4n)
FAILSAFE = 1.0e6      # returned when even the fallback cannot be computed
HOOK_THREADS = 1      # torch intra-op threads during the hook's forward pass (restored afterwards)
DEFAULT_CACHE_SIZE = 65536
INVALID_POLICIES = ("fallback", "raw")

BWM_PLANNER = BI.BWM_ROOT / "bwmn" / "blocks_planner.py"
BWM_PLANNER_PIN = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"

REASONS = ("malformed_fact", "self_on", "multiple_positions", "no_position", "shared_support", "multiple_held",
           "holding_and_handempty", "hand_unspecified", "on_held_block", "cycle", "missing_clear", "wrong_clear",
           "unknown_block")
COUNTERS = ("calls", "cache_hits", "evaluations", "model_evals", "invalid_states", "fallbacks", "encode_errors",
            "model_errors", "nonfinite", "clipped", "failsafe", "analysis_errors", "bad_goal_facts", "goal_non_on")

_ARITY = {"on": 2, "ontable": 1, "clear": 1, "holding": 1, "handempty": 0}
_MISSING = object()
_TM: dict = {}


# ----------------------------------------------------------------------------- lazy, host-neutral torch import

def _torch_models():
    """(torch, rcx.bw.models), imported once without changing the host's CUDA view or bytecode flag (module doc)."""
    if "M" in _TM:
        return _TM["torch"], _TM["M"]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
    dwb = sys.dont_write_bytecode
    import torch
    cuda = (torch.cuda.is_available, torch.cuda.device_count)
    try:
        from . import models as M
    finally:
        torch.cuda.is_available, torch.cuda.device_count = cuda
        if env is _MISSING:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = env
        sys.dont_write_bytecode = dwb
    _TM.update(torch=torch, M=M)
    return torch, M


# ----------------------------------------------------------------------------- fact checks and sanitising

def _well_formed(f) -> bool:
    return (isinstance(f, tuple) and len(f) >= 1 and isinstance(f[0], str) and f[0] in _ARITY
            and len(f) == 1 + _ARITY[f[0]] and all(isinstance(a, str) and a != "" for a in f[1:]))


def _split(facts):
    """(usable well-formed facts, # malformed, # self-on) of an iterable of facts."""
    good, bad, self_on = set(), 0, 0
    for f in facts:
        if not _well_formed(f):
            bad += 1
        elif f[0] == "on" and f[1] == f[2]:
            self_on += 1
        else:
            good.add(f)
    return frozenset(good), bad, self_on


def _has_cycle(succ: dict) -> bool:
    """Directed cycle in {x: [y, ...]} (iterative DFS, deterministic)."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = defaultdict(int)
    for root in sorted(succ):
        if color[root]:
            continue
        stack = [(root, iter(sorted(succ.get(root, ()))))]
        color[root] = GREY
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = BLACK
                stack.pop()
            elif color[nxt] == GREY:
                return True
            elif color[nxt] == WHITE:
                color[nxt] = GREY
                stack.append((nxt, iter(sorted(succ.get(nxt, ())))))
    return False


def analyse(state, goal=None, objs=None) -> dict:
    """Split state/goal into usable facts and list the inconsistency reasons (raises only if state/goal is not
    iterable). Returns {'reasons': sorted list, 'state': frozenset, 'goal': frozenset, 'objs': sorted list,
    'bad_goal_facts': int, 'goal_non_on': bool}."""
    s_good, s_bad, s_self = _split(state)
    g_good, g_bad, g_self = _split(() if goal is None else goal)
    reasons = set()
    if s_bad:
        reasons.add("malformed_fact")
    if s_self:
        reasons.add("self_on")
    n_pos, on_sup, held, handempty, clear, names = Counter(), defaultdict(list), [], False, set(), set()
    for f in s_good:
        p = f[0]
        if p == "handempty":
            handempty = True
            continue
        names.update(f[1:])
        if p == "on":
            on_sup[f[1]].append(f[2])
            n_pos[f[1]] += 1
        elif p == "ontable":
            n_pos[f[1]] += 1
        elif p == "holding":
            n_pos[f[1]] += 1
            held.append(f[1])
        else:
            clear.add(f[1])
    known = None if objs is None else {o for o in objs if isinstance(o, str) and o != ""}
    goal_names = {a for f in g_good for a in f[1:]}
    blocks = names | goal_names | (known or set())
    if known is not None and names - known:
        reasons.add("unknown_block")
    for b in blocks:
        k = n_pos[b]
        if k == 0:
            reasons.add("no_position")
        elif k > 1:
            reasons.add("multiple_positions")
    covered = Counter(y for ys in on_sup.values() for y in ys)
    if any(v > 1 for v in covered.values()):
        reasons.add("shared_support")
    if len(held) > 1:
        reasons.add("multiple_held")
    if held and handempty:
        reasons.add("holding_and_handempty")
    if not held and not handempty:
        reasons.add("hand_unspecified")
    held_set = set(held)
    if any(y in held_set for y in covered):
        reasons.add("on_held_block")
    if _has_cycle(on_sup):
        reasons.add("cycle")
    for b in blocks:
        should = b not in covered and b not in held_set
        if b in clear and not should:
            reasons.add("wrong_clear")
        elif should and b not in clear:
            reasons.add("missing_clear")
    return {"reasons": sorted(reasons), "state": s_good, "goal": g_good, "objs": sorted(blocks),
            "bad_goal_facts": g_bad + g_self, "goal_non_on": any(f[0] != "on" for f in g_good)}


def is_consistent(state, goal=None, objs=None) -> tuple[bool, list[str]]:
    """(True, []) iff `state` is a valid Blocksworld state over its blocks (plus the goal's blocks and, if given, the
    instance's `objs`); otherwise (False, sorted reason codes; see the module doc). Never raises: a non-iterable state
    or goal gives (False, ['not_iterable'])."""
    try:
        r = analyse(state, goal, objs)["reasons"]
    except Exception:
        return False, ["not_iterable"]
    return not r, r


def _sanitised_pos(s_good: frozenset, objs: list) -> tuple:
    """Compact code (space.py: TABLE = n, HAND = n + 1) of the deterministic sanitised copy (module doc)."""
    n = len(objs)
    idx = {o: i for i, o in enumerate(objs)}
    held = sorted(f[1] for f in s_good if f[0] == "holding")
    h = held[0] if held else None
    sup = defaultdict(set)
    for f in s_good:
        if f[0] == "on":
            sup[f[1]].add(f[2])
    pos = [n] * n
    if h is not None:
        pos[idx[h]] = n + 1
    taken = set()
    for x in objs:
        if x == h:
            continue
        ix = idx[x]
        for y in sorted(sup.get(x, ())):
            if y == h or y in taken:
                continue
            j, steps, cyc = idx[y], 0, False
            while j < n and steps <= n:
                if j == ix:
                    cyc = True
                    break
                j, steps = pos[j], steps + 1
            if not cyc:
                pos[ix] = idx[y]
                taken.add(y)
                break
    return tuple(pos)


def sanitise(state, goal=None, objs=None) -> tuple[frozenset, list]:
    """(valid Blocksworld state, its sorted blocks): the deterministic sanitised copy of `state` (module doc)."""
    a = analyse(state, goal, objs)
    return SP.to_facts(_sanitised_pos(a["state"], a["objs"]), a["objs"]), a["objs"]


def _misplaced(s_good: frozenset, g_good: frozenset, objs: list) -> int:
    """misplaced_blocks on the sanitised copy; equals heuristics.misplaced_blocks for a consistent state."""
    return int(SP.misplaced_compact(_sanitised_pos(s_good, objs), SP.CompiledGoal(objs, g_good))[0])


def misplaced_sanitised(state, goal, objs=None) -> int:
    a = analyse(state, goal, objs)
    return _misplaced(a["state"], a["goal"], a["objs"])


# ----------------------------------------------------------------------------- model loading

def _file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha256(model) -> str:
    """sha256 over the state_dict (key, dtype, shape, raw bytes), in key order: a fingerprint for in-memory models."""
    h = hashlib.sha256()
    for k, t in model.state_dict().items():
        t = t.detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_gnn(path, config: dict | None = None):
    """Load a saved RCX RelGNN: (model in eval mode, provenance dict).

    Same code path as run_e07.load_model (new model, load_state_dict(torch.load(path, weights_only=True)), eval), with
    two additions: the architecture comes from `config`, else from the sidecar `<stem>.json` written next to the
    weights by run_e07.phase_train (key 'config' = models.config_dict), else the RelGNNConfig() default that
    run_e07.new_model uses; and map_location='cpu'. load_state_dict is strict, and a recorded n_params is checked."""
    torch, M = _torch_models()
    path = Path(path).resolve()
    side = path.with_suffix(".json")
    prov = {"model_file": str(path), "model_sha256": _file_sha(path), "sidecar": None, "sidecar_sha256": None,
            "config_source": "default RelGNNConfig() (as run_e07.new_model)"}
    cfg = config
    if cfg is None and side.exists():
        prov.update(sidecar=str(side), sidecar_sha256=_file_sha(side))
        cfg = json.loads(side.read_text(encoding="utf-8")).get("config")
        if cfg is not None:
            prov["config_source"] = "sidecar json 'config'"
    elif cfg is not None:
        prov["config_source"] = "explicit config argument"
    cfg = dict(cfg or {})
    if cfg.get("kind", "gnn") != "gnn":
        raise ValueError(f"only RelGNN ('gnn') models are supported, got kind={cfg.get('kind')!r}")
    names = {f.name for f in fields(M.RelGNNConfig)}
    with torch.random.fork_rng(devices=[]):
        model = M.RelGNN(M.RelGNNConfig(**{k: v for k, v in cfg.items() if k in names}))
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    if "n_params" in cfg and int(cfg["n_params"]) != M.n_params(model):
        raise ValueError(f"n_params mismatch: config {cfg['n_params']} vs model {M.n_params(model)}")
    return model, prov


# ----------------------------------------------------------------------------- the hook

class LearnedUnmet:
    """unmet(state, goal) -> float for BWM plan_beam (lower is closer); see the module doc for the value definition.

    model: an in-memory RelGNN (kind 'gnn'), or use LearnedUnmet.from_file(path, ...).
    invalid_policy: 'fallback' (default; inconsistent states get PENALTY + misplaced on the sanitised copy) or 'raw'
        (inconsistent states are scored by the model on their well-formed facts).
    hybrid_lambda: value = model + hybrid_lambda * misplaced_blocks (the E07b A3 'GNN-hybrid' key; 0 = plain GNN).
    round_to: if set, model-path values are rounded to a multiple of it (restores ties for plan_beam's prefix-length
        tie-break; None = no rounding).
    cache_size: LRU cache over (state, goal); 0 disables. Values do not depend on the cache.
    instances: optional iterable of dicts with 'goal' and 'objs' (plan_beam's instances); enables the unknown_block
        and missing-block checks for those goals (a goal listed with different objs maps to their union).
    """

    def __init__(self, model, *, invalid_policy: str = "fallback", hybrid_lambda: float = 0.0,
                 round_to: float | None = None, cache_size: int = DEFAULT_CACHE_SIZE, instances=None,
                 _provenance: dict | None = None):
        torch, M = _torch_models()
        if getattr(model, "kind", None) != "gnn":
            raise ValueError("LearnedUnmet needs a RelGNN (model.kind == 'gnn')")
        if invalid_policy not in INVALID_POLICIES:
            raise ValueError(f"invalid_policy must be one of {INVALID_POLICIES}")
        if not math.isfinite(float(hybrid_lambda)) or (round_to is not None and not float(round_to) > 0):
            raise ValueError("hybrid_lambda must be finite and round_to > 0")
        self._torch, self._M = torch, M
        self.model = model.eval()
        self.derived = model.cfg.n_rel > len(EN.EDGE_TYPES)
        self.invalid_policy = invalid_policy
        self.hybrid_lambda = float(hybrid_lambda)
        self.round_to = None if round_to is None else float(round_to)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        self._registry, self.registry_conflicts = {}, 0
        for it in instances or ():
            g, o = frozenset(it["goal"]), frozenset(it["objs"])
            if g in self._registry and self._registry[g] != o:
                self.registry_conflicts += 1
                o = o | self._registry[g]
            self._registry[g] = o
        self._prov = dict(_provenance or {"model_file": None, "model_sha256": None, "sidecar": None,
                                          "sidecar_sha256": None, "config_source": "in-memory model"})
        self.reset_counters()

    @classmethod
    def from_file(cls, path, *, config: dict | None = None, **kwargs) -> "LearnedUnmet":
        model, prov = load_gnn(path, config)
        return cls(model, _provenance=prov, **kwargs)

    def __bool__(self) -> bool:  # plan_beam binds `unmet or default`: the hook must be truthy
        return True

    def reset_counters(self):
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.reasons = Counter()

    # -- public API

    def __call__(self, state, goal) -> float:
        try:
            return self._call(state, goal)
        except Exception:  # last line of defence; _call already catches everything it can
            self.counters["failsafe"] += 1
            return FAILSAFE

    def batch(self, states, goal) -> list[float]:
        """[self(s, goal) for s in states]: single-state forwards, so batch values equal individual calls exactly."""
        try:
            states = list(states)
        except Exception:
            self.counters["failsafe"] += 1
            return []
        return [self(s, goal) for s in states]

    def describe(self) -> dict:
        torch, M = self._torch, self._M
        here = Path(__file__).resolve().parent
        return {
            "hook": "rcx.bw.b4_hook.LearnedUnmet",
            **self._prov,
            "weights_sha256": weights_sha256(self.model),
            "config": M.config_dict(self.model),
            "derived_edges": self.derived,
            "invalid_policy": self.invalid_policy, "hybrid_lambda": self.hybrid_lambda, "round_to": self.round_to,
            "constants": {"PENALTY": PENALTY, "MODEL_CLIP": MODEL_CLIP, "FAILSAFE": FAILSAFE,
                          "HOOK_THREADS": HOOK_THREADS},
            "cache": {"size": self.cache_size, "entries": len(self._cache)},
            "registry": {"goals": len(self._registry), "conflicts": self.registry_conflicts},
            "counters": dict(self.counters), "reasons": dict(sorted(self.reasons.items())),
            "code_sha256": {f"rcx/bw/{m}.py": _file_sha(here / f"{m}.py")
                            for m in ("b4_hook", "encode", "models", "space")},
            "torch": torch.__version__, "python": sys.version.split()[0],
            "bwm_planner_pin": BWM_PLANNER_PIN,
        }

    # -- internals

    def _call(self, state, goal) -> float:
        c = self.counters
        c["calls"] += 1
        key = None
        if self.cache_size > 0:
            try:
                key = (state if isinstance(state, frozenset) else frozenset(state),
                       goal if isinstance(goal, frozenset) else frozenset(goal))
                hash(key)
            except Exception:
                key = None
            if key is not None and key in self._cache:
                self._cache.move_to_end(key)
                c["cache_hits"] += 1
                return self._cache[key]
        v = self._evaluate(state, goal)
        if key is not None:
            self._cache[key] = v
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return v

    def _objs_for(self, goal):
        if not self._registry:
            return None
        try:
            return self._registry.get(goal if isinstance(goal, frozenset) else frozenset(goal))
        except Exception:
            return None

    def _evaluate(self, state, goal) -> float:
        c = self.counters
        c["evaluations"] += 1
        try:
            a = analyse(state, goal, self._objs_for(goal))
        except Exception:
            c["analysis_errors"] += 1
            c["failsafe"] += 1
            return FAILSAFE
        if a["bad_goal_facts"]:
            c["bad_goal_facts"] += 1
        if a["goal_non_on"]:
            c["goal_non_on"] += 1
        if a["reasons"]:
            c["invalid_states"] += 1
            self.reasons.update(a["reasons"])
            if self.invalid_policy == "fallback":
                return self._fallback(a)
        try:
            g = EN.graph(a["state"], a["goal"], a["objs"], derived=self.derived)
        except Exception:
            c["encode_errors"] += 1
            return self._fallback(a)
        try:
            v = self._forward(g)
            c["model_evals"] += 1
            if self.hybrid_lambda:
                v += self.hybrid_lambda * _misplaced(a["state"], a["goal"], a["objs"])
        except Exception:
            c["model_errors"] += 1
            return self._fallback(a)
        if not math.isfinite(v):
            c["nonfinite"] += 1
            return self._fallback(a)
        if abs(v) > MODEL_CLIP:
            c["clipped"] += 1
            v = max(-MODEL_CLIP, min(MODEL_CLIP, v))
        if self.round_to is not None:
            v = round(v / self.round_to) * self.round_to
        return float(v)

    def _forward(self, g) -> float:
        torch = self._torch
        prev = torch.get_num_threads()
        try:
            if prev != HOOK_THREADS:
                torch.set_num_threads(HOOK_THREADS)
            return float(self._M.predict(self.model, [g])[0])
        finally:
            if prev != HOOK_THREADS:
                torch.set_num_threads(prev)

    def _fallback(self, a) -> float:
        self.counters["fallbacks"] += 1
        try:
            v = PENALTY + float(_misplaced(a["state"], a["goal"], a["objs"]))
            if math.isfinite(v):
                return v
        except Exception:
            pass
        self.counters["failsafe"] += 1
        return FAILSAFE


# ----------------------------------------------------------------------------- BWM planner compatibility check

# Exact source snippets of plan_beam that consume the hook's values (checked verbatim against the file text).
_VALUE_USES = {
    "trajectory_argmin": "t_best = min(range(len(traj)), key=lambda t: (um[t], t))",
    "child_key": "kids.setdefault(e[\"id\"], []).append((um[t_best], len(path) + t_best + 1, rank, traj[t_best],",
    "child_sort_dedup": "for u, ln, rank, st, p in sorted(kids.get(e[\"id\"], []), key=lambda x: x[:3]):",
    "beam_select": "nxt = sorted(fresh.values(), key=lambda x: x[:3])[:width]",
    "best_update": "if (u, ln) < e[\"best\"][:2]:",
    "best_init": "\"best\": (unmet(it[\"init\"], it[\"goal\"]), 0, it[\"init\"], [])})",
    "best_dropped_before_return": "for k in (\"beam\", \"seen\", \"best\", \"goal\", \"objs\"):",
}
_VALUE_NAMES = ("um", "u")  # names bound to hook values in plan_beam (see _VALUE_USES)


def planner_hook_usage(path=BWM_PLANNER, function: str = "plan_beam", hook: str = "unmet") -> dict:
    """Read BWM's blocks_planner.py as text (never imported or executed) and report how `function` uses `hook`.

    Returns the file sha256 (and whether it matches BWM_PLANNER_PIN), the function signature, the hook's default
    binding, every call site (line, source, positional/keyword arguments, enclosing statement), the docstring contract,
    the value-consuming snippets in _VALUE_USES (found or not, with line numbers), an ast check that no arithmetic or
    numeric conversion is applied to the hook's values, and a 'facts' summary for B4's prereg."""
    path = Path(path)
    data = path.read_bytes()
    text = data.decode("utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if fn is None:
        raise ValueError(f"{function} not found in {path}")
    a = fn.args
    pos_defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    params = [p.arg + ("" if d is None else "=" + ast.unparse(d)) for p, d in zip(a.args, pos_defaults)]
    kwonly = [p.arg + ("" if d is None else "=" + ast.unparse(d)) for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    signature = f"{function}({', '.join(params + (['*'] + kwonly if kwonly else []))})"
    hook_kwonly = hook in [p.arg for p in a.kwonlyargs]
    hook_default = None
    if hook_kwonly:
        d = a.kw_defaults[[p.arg for p in a.kwonlyargs].index(hook)]
        hook_default = None if d is None else ast.unparse(d)

    def seg(node):
        return ast.get_source_segment(text, node)

    bindings = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == hook for t in n.targets):
            bindings.append({"line": n.lineno, "source": seg(n),
                             "by_truthiness": isinstance(n.value, ast.BoolOp) and isinstance(n.value.op, ast.Or)})
    stmts = [n for n in ast.walk(fn) if isinstance(n, ast.stmt) and n is not fn]
    calls = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == hook:
            enclosing = min((s for s in stmts if s.lineno <= n.lineno <= s.end_lineno),
                            key=lambda s: s.end_lineno - s.lineno)
            calls.append({"line": n.lineno, "call": seg(n), "n_positional": len(n.args),
                          "positional_args": [seg(x) for x in n.args], "keywords": [k.arg for k in n.keywords],
                          "statement_lines": [enclosing.lineno, enclosing.end_lineno],
                          "statement": "\n".join(lines[enclosing.lineno - 1:enclosing.end_lineno]).strip()})
    calls.sort(key=lambda r: r["line"])
    doc = ast.get_docstring(fn) or ""
    contract = " ".join(l.strip() for l in doc.splitlines() if re.search(rf"\b{hook}\(", l) or "lower" in l)
    lo, hi = fn.lineno, fn.end_lineno
    uses = {}
    for k, snip in _VALUE_USES.items():
        hits = [i + 1 for i in range(lo - 1, hi) if snip in lines[i]]
        uses[k] = {"snippet": snip, "lines": hits, "found": bool(hits)}

    def mentions(node):
        return any(isinstance(x, ast.Name) and x.id in _VALUE_NAMES for x in ast.walk(node))

    arith = []
    for n in ast.walk(fn):
        if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.AugAssign)) and mentions(n):
            arith.append({"line": n.lineno, "source": seg(n)})
        elif (isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
              and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) in
              ("int", "float", "round", "abs", "sum", "str", "format", "dumps", "dump")
              and any(mentions(x) for x in n.args)):
            arith.append({"line": n.lineno, "source": seg(n)})
    matches = hashlib.sha256(data).hexdigest() == BWM_PLANNER_PIN
    facts = {
        "call_form": f"positional {hook}(state, goal); {len(calls)} call sites; no keywords; one state per call "
                     "(no batch call)",
        "argument_1_state": "frozenset of fact tuples: the instance's init (true state) or an imagined state "
                            "returned by F (fwd.predict -> (state, legal)); may be inconsistent",
        "argument_2_goal": "the instance's goal frozenset (e['goal'] = it['goal']); BWM goals are ('on', x, y) only",
        "lower_is_better": True,
        "return_type": "any totally ordered number; default returns int len(goal - state); float accepted "
                       "(values are only compared: min/sorted keys and tuple '<'); must not be NaN",
        "arithmetic_or_conversion_on_value": bool(arith),
        "hook_selected_by_truthiness": any(b["by_truthiness"] for b in bindings),
        "tie_breaks": "trajectory: earliest t among equal values; children: (value, prefix length, candidate rank); "
                      "best: (value, prefix length) lexicographic",
        "solution_detection_uses_hook": False,
    }
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "pinned_sha256": BWM_PLANNER_PIN,
            "matches_pin": matches, "function": function, "lines": [lo, hi], "signature": signature,
            "hook_keyword_only": hook_kwonly, "hook_default": hook_default, "default_binding": bindings,
            "contract_docstring": contract, "call_sites": calls, "value_uses": uses,
            "arithmetic_or_conversion_on_value": arith, "facts": facts}


if __name__ == "__main__":  # python -B -m rcx.bw.b4_hook  -> prints the planner hook report (no torch import)
    print(json.dumps(planner_hook_usage(), indent=1))
