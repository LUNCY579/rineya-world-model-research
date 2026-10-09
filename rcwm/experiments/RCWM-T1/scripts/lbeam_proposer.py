"""Model-free proposers for BWM's plan_beam on the ledger planning problems, and the PREREG-2-style fixed-candidate
selection baseline. Torch-free; no model calls.

Proposer contract (BWM D-53): propose(items, eval_seed) -> per item a list of K dicts {"plan": [action, ...] or None,
"n_tokens": int}; items are plan_beam's {"id", "objs", "goal", "state", "round"} (plan_beam forwards nothing else).
Candidate j of an item is drawn from its own RNG seeded by crn_seed(eval_seed, id, round, j) (BWM's _seed formula: the
first 15 hex digits of sha256("eval_seed|id|round|j")), so a proposal never depends on which other items share the
batch.

n_tokens = 0 for every proposer here: they are model-free (no generated tokens). Proposer cost is therefore zero by
construction, and every model-call budget in this test is an F budget.

What a proposer may read. The planner-side proposers (RandomProposer, EnumerateProposer) read ONLY the item (objs =
the instance's action vocabulary, state, goal, id, round); they need no instance list at all. Registered runs hand BWM
lbeam_problems.planner_instances(insts) (blind: id / init / goal / objs only). The answer key (optimal plan, seed plan,
vocab roles, fixed candidates) stays on the RCWM side: FixedCandidatesProposer and select_fixed take the full records,
and OracleProposer is a harness-only cheat.

  EnumerateProposer       every vocabulary action sequence of length `depth` (default 1: the V = 8 single actions) as
                          the candidates, in vocabulary order; deterministic (no RNG, so trivially batch-independent);
                          state- and goal-blind. With plan_beam this is best-first / beam search over one-step children
                          ranked by the unmet hook. The recommended default model-free proposer (review 2026-10-09).
  RandomProposer          K random action sequences of length L drawn uniformly (with replacement) from the instance's
                          vocabulary, which includes the void twins and away actions; ignores the state and the goal
  FixedCandidatesProposer the instance's K stored fixed candidates (PREREG-2 style; RCWM records); meaningful only at
                          plan_beam's first expansion (later rounds get no candidates); use with max_expansions=1, width=1
  OracleProposer          K copies of a BFS-shortest plan from the item's (imagined) state over objs (bound: every
                          state on the plan within |b| <= bound); harness tests only
  select_fixed            the fixed-candidate SELECTION baseline (RCWM records): roll out all K candidates under F (one
                          batched F call per step index across all instances), select the candidate whose PREDICTED
                          final state meets the goal, else has the smallest predicted unmet; ties at random (CRN seed per
                          instance); a candidate F cannot support (None) ranks last.

Registered plan_beam keywords: beam_kwargs(proposer, width, max_expansions, max_rollout) returns plan_beam's keyword
arguments with max_depth = lbeam_problems.registered_max_depth(sim_len) = D_CLOSE - sim_len (sim_len = the simulated
candidate length: the proposer's plan length, or max_rollout if smaller). plan_beam's own default (max_depth 80) lets a
long search imagine states beyond the closure depth of a primary instance, i.e. possibly outside T2's verified range;
with beam_kwargs every F input and exact F output is a closure state (|b| <= 98). Use it for every registered run:
    BP.plan_beam(planner_instances(insts), prop, F, eval_seed=s, **beam_kwargs(prop, W, E, R))
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lbeam_problems as P  # noqa: E402


def crn_seed(*parts) -> int:
    """BWM blocks_planner._seed: int(sha256('|'.join(parts))[:15], 16)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def _vocab(item) -> tuple:
    objs = item.get("objs")
    if not objs:
        raise ValueError(f"item {item.get('id')!r} carries no objs (the action vocabulary)")
    return tuple(objs)


class RandomProposer:
    name = "random"

    def __init__(self, instances=None, K: int = 8, length: int = P.D_MAX):
        """instances: unused (kept for call compatibility); the vocabulary is read from each item's objs."""
        self.K, self.length = K, length

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            voc = _vocab(it)
            cands = []
            for j in range(self.K):
                rng = random.Random(crn_seed(eval_seed, it["id"], it["round"], j))
                cands.append({"plan": [voc[rng.randrange(len(voc))] for _ in range(self.length)], "n_tokens": 0})
            out.append(cands)
        return out


class EnumerateProposer:
    name = "enumerate"

    def __init__(self, instances=None, depth: int = 1):
        """Every vocabulary sequence of length `depth` (V ** depth candidates; depth 1 or 2)."""
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        self.depth = depth

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            voc = _vocab(it)
            out.append([{"plan": list(seq), "n_tokens": 0} for seq in itertools.product(voc, repeat=self.depth)])
        return out


class FixedCandidatesProposer:
    name = "fixed"

    def __init__(self, records):
        """records: the RCWM instance records (lbeam_problems.load_set), NOT the blind planner dicts."""
        self.by_id = {}
        for r in records:
            r = r.get("rec", r)
            if "fixed" not in r:
                raise ValueError("FixedCandidatesProposer needs the RCWM records (with 'fixed'), not blind planner dicts")
            self.by_id[r["id"]] = r

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            if str(it["round"]) not in ("0", "0.0"):
                out.append([])
                continue
            out.append([{"plan": list(c), "n_tokens": 0} for c in self.by_id[it["id"]]["fixed"]["cands"]])
        return out


class OracleProposer:
    """BFS-shortest plans from the item's state over its objs (exact dynamics). Harness tests only: perfect proposals.
    bound: None, an int (every state on the plan within |b| <= bound), or a dict id -> int / None."""
    name = "oracle"

    def __init__(self, instances=None, K: int = 1, bound=None):
        self.K, self.bound = K, bound

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            b = self.bound.get(it["id"]) if isinstance(self.bound, dict) else self.bound
            voc = list(_vocab(it))
            r = P.bfs(P.vec(it["state"]), [P.compile_action(a) for a in voc], P._goal_c(it["goal"]), P.D_MAX, bound=b)
            p = None if r is None else [voc[i] for i in r["plan"]]
            out.append([{"plan": p, "n_tokens": 0} for _ in range(self.K)])
        return out


def candidate_len(proposer) -> int:
    """The longest plan a proposer emits (EnumerateProposer: depth; RandomProposer: length; fixed / oracle: D_MAX)."""
    if isinstance(proposer, EnumerateProposer):
        return proposer.depth
    if isinstance(proposer, RandomProposer):
        return proposer.length
    return getattr(proposer, "max_plan_len", P.D_MAX)


def beam_kwargs(proposer, width: int, max_expansions: int, max_rollout: int = 0, d_close: int = P.D_CLOSE) -> dict:
    """plan_beam keyword arguments of a registered run on a primary set (hooks + max_depth from the closure depth)."""
    sim = candidate_len(proposer)
    if max_rollout:
        sim = min(sim, max_rollout)
    return {"width": width, "max_expansions": max_expansions, "max_rollout": max_rollout,
            "max_depth": P.registered_max_depth(sim, d_close), "goal_met": P.goal_met, "unmet": P.unmet}


def rollout_candidates(fwd, starts, plans):
    """Batched rollout (one fwd.predict call per step index across every plan). Per plan: the predicted states after
    each step and whether every step was supported (not None) and legal."""
    n = len(plans)
    states = list(starts)
    traj = [[] for _ in range(n)]
    ok = [True] * n
    alive = [bool(p) for p in plans]
    d = 0
    while True:
        keys, idx = [], []
        for i in range(n):
            if alive[i] and d < len(plans[i]):
                keys.append((states[i], plans[i][d]))
                idx.append(i)
            elif alive[i]:
                alive[i] = False
        if not keys:
            break
        for i, o in zip(idx, fwd.predict(keys)):
            if o is None or not o[1]:
                ok[i] = False
                alive[i] = False
                continue
            states[i] = o[0]
            traj[i].append(o[0])
        d += 1
    return [{"traj": traj[i], "ok": ok[i]} for i in range(n)]


def select_fixed(records, fwd, eval_seed: int = 0, goal_met=P.goal_met, unmet=P.unmet):
    """PREREG-2-style selection among each instance's K fixed candidates under F (records: the RCWM instance records).
    Returns plan_beam-like episode dicts {id, plan, verified, fail, selected, pred_unmet, kinds, expansions, rounds,
    gen_tokens}."""
    recs = [r.get("rec", r) for r in records]
    if any("fixed" not in r for r in recs):
        raise ValueError("select_fixed needs the RCWM records (with 'fixed'), not blind planner dicts")
    starts, plans, owner = [], [], []
    for r in recs:
        for j, c in enumerate(r["fixed"]["cands"]):
            starts.append(r["init"])
            plans.append(list(c))
            owner.append((r["id"], j))
    rolls = rollout_candidates(fwd, starts, plans)
    per = {}
    for (iid, j), plan, ro in zip(owner, plans, rolls):
        full = ro["ok"] and len(ro["traj"]) == len(plan)
        per.setdefault(iid, []).append(ro["traj"][-1] if full else None)
    eps = []
    for r in recs:
        finals = per[r["id"]]
        pu = [math.inf if f is None else unmet(f, r["goal"]) for f in finals]
        pm = [f is not None and goal_met(f, r["goal"]) for f in finals]
        tie = random.Random(crn_seed(eval_seed, r["id"], "fixed", 0)).sample(range(len(finals)), len(finals))
        rank = {j: k for k, j in enumerate(tie)}
        sel = min(range(len(finals)), key=lambda j: (not pm[j], pu[j], rank[j]))
        eps.append({"id": r["id"], "plan": list(r["fixed"]["cands"][sel]), "verified": bool(pm[sel]),
                    "fail": None if pm[sel] else ("no_supported_candidate" if math.isinf(pu[sel]) else "no_candidate_predicted_to_reach"),
                    "selected": sel, "selected_kind": r["fixed"]["kinds"][sel], "pred_unmet": pu,
                    "n_unsupported": sum(f is None for f in finals), "expansions": 1, "rounds": 1, "gen_tokens": 0})
    return eps
