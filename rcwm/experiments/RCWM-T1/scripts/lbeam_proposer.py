"""Model-free proposers for BWM's plan_beam on the ledger planning problems, and the PREREG-2-style fixed-candidate
selection baseline. Torch-free; no model calls.

Proposer contract (BWM D-53): propose(items, eval_seed) -> per item a list of K dicts {"plan": [action, ...] or None,
"n_tokens": int}; items are plan_beam's {"id", "objs", "goal", "state", "round", ...}. Candidate j of an item is drawn
from its own RNG seeded by crn_seed(eval_seed, id, round, j) (BWM's _seed formula: the first 15 hex digits of
sha256("eval_seed|id|round|j")), so a proposal never depends on which other items share the batch.

n_tokens = 0 for every proposer here: they are model-free (no generated tokens). Proposer cost is therefore zero by
construction, and every model-call budget in this test is an F budget.

  RandomProposer          K random action sequences of length L drawn uniformly (with replacement) from the instance's
                          vocabulary, which includes the void twins and away actions; ignores the state and the goal
  FixedCandidatesProposer the instance's K stored fixed candidates (PREREG-2 style); meaningful only at plan_beam's
                          first expansion (later rounds get no candidates); use with max_expansions=1, width=1
  OracleProposer          K copies of a BFS-shortest plan from the item's (imagined) state; harness tests only
  select_fixed            the fixed-candidate SELECTION baseline: roll out all K candidates under F (one batched F call
                          per step index across all instances), select the candidate whose PREDICTED final state meets
                          the goal, else has the smallest predicted unmet; ties at random (CRN seed per instance); a
                          candidate F cannot support (None) ranks last.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
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


class _ByID:
    def __init__(self, instances):
        self.by_id = {}
        for it in instances:
            rec = it.get("rec", it)
            self.by_id[it["id"]] = rec

    def vocab(self, item) -> tuple:
        objs = item.get("objs")
        if objs:
            return tuple(objs)
        return tuple(self.by_id[item["id"]]["vocab"])


class RandomProposer(_ByID):
    name = "random"

    def __init__(self, instances, K: int = 8, length: int = P.D_MAX):
        super().__init__(instances)
        self.K, self.length = K, length

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            voc = self.vocab(it)
            cands = []
            for j in range(self.K):
                rng = random.Random(crn_seed(eval_seed, it["id"], it["round"], j))
                cands.append({"plan": [voc[rng.randrange(len(voc))] for _ in range(self.length)], "n_tokens": 0})
            out.append(cands)
        return out


class FixedCandidatesProposer(_ByID):
    name = "fixed"

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            if str(it["round"]) not in ("0", "0.0"):
                out.append([])
                continue
            out.append([{"plan": list(c), "n_tokens": 0} for c in self.by_id[it["id"]]["fixed"]["cands"]])
        return out


class OracleProposer(_ByID):
    """BFS-shortest plans from the item's state (exact dynamics). Harness tests only: perfect proposals."""
    name = "oracle"

    def __init__(self, instances, K: int = 1):
        super().__init__(instances)
        self.K = K

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            p = P.bfs_plan(it["state"], it["goal"], self.vocab(it))
            out.append([{"plan": p, "n_tokens": 0} for _ in range(self.K)])
        return out


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


def select_fixed(instances, fwd, eval_seed: int = 0, goal_met=P.goal_met, unmet=P.unmet):
    """PREREG-2-style selection among each instance's K fixed candidates under F. Returns plan_beam-like episode dicts
    {id, plan, verified, fail, selected, pred_unmet, kinds, expansions, rounds, gen_tokens}."""
    recs = [it.get("rec", it) for it in instances]
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
