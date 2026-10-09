"""RegisterWorld-v2: RegisterWorld-v1 semantics with tight budgets, several limits and transfer splits.

The transition law is identical to src/rineya_bwm/core.py `exact_step`. The exact executor and the BFS
distance tables are collection/evaluation instruments; no learned component may call them.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

MOVES = ("add1", "sub1", "double", "halve")
ACTIONS = MOVES + ("finish",)
REWARD_ID = "register-v2-success-step-violation"


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exact_step(x: int, action: str, limit: int) -> tuple[int, bool, bool]:
    """(next_x, legal, terminated). Same law as core.exact_step."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action!r}")
    if action == "finish":
        return x, True, True
    if action == "halve" and x % 2:
        return x, False, False
    nx = {"add1": x + 1, "sub1": x - 1, "double": 2 * x, "halve": x // 2}[action]
    if abs(nx) > limit:
        return x, False, False
    return nx, True, False


@lru_cache(maxsize=None)
def dist_to(goal: int, limit: int) -> dict:
    """Shortest number of legal moves from every x in [-limit, limit] to goal (BFS on reversed graph)."""
    states = range(-limit, limit + 1)
    rev = {x: [] for x in states}
    for x in states:
        for a in MOVES:
            nx, legal, _ = exact_step(x, a, limit)
            if legal and nx != x:
                rev[nx].append(x)
    d = {goal: 0}
    q = deque([goal])
    while q:
        y = q.popleft()
        for x in rev[y]:
            if x not in d:
                d[x] = d[y] + 1
                q.append(x)
    return d


def opt_moves(start: int, goal: int, limit: int) -> int | None:
    return dist_to(goal, limit).get(start)


def optimal_next_actions(x: int, goal: int, limit: int) -> list[str]:
    d = dist_to(goal, limit)
    if x not in d:
        return []
    if d[x] == 0:
        return ["finish"]
    out = []
    for a in MOVES:
        nx, legal, _ = exact_step(x, a, limit)
        if legal and d.get(nx, 10**9) == d[x] - 1:
            out.append(a)
    return out


def sample_optimal_plan(x: int, goal: int, limit: int, rng: random.Random) -> list[str]:
    plan = []
    while True:
        acts = optimal_next_actions(x, goal, limit)
        if not acts:
            raise ValueError("unreachable")
        a = rng.choice(acts)
        plan.append(a)
        if a == "finish":
            return plan
        x, _, _ = exact_step(x, a, limit)


@dataclass(frozen=True)
class Task:
    start: int
    goal: int
    limit: int
    budget: int  # max actions including finish

    @property
    def family(self) -> str:
        # The limit never changes the optimal plan in these pools, so limit variants are one family (amendment S2).
        return hashlib.sha256(canonical({"d": "RW2", "s": self.start, "g": self.goal}).encode()).hexdigest()

    @property
    def split(self) -> str:
        return goal_split(self.goal)

    @property
    def opt(self) -> int | None:
        return opt_moves(self.start, self.goal, self.limit)

    @property
    def feasible(self) -> bool:
        o = self.opt
        return o is not None and o + 1 <= self.budget

    def to_json(self) -> dict:
        return {"start": self.start, "goal": self.goal, "limit": self.limit, "budget": self.budget}


def goal_split(goal: int) -> str:
    """Goal-held-out split (amendment A1/SC-01/S2): a goal value belongs to exactly one split, across all limits and
    pools. Suffix relabelling only creates rows whose goal is the episode goal, so no test goal is ever a training goal.
    Proportions 65/10/5/20 (train/validation/calibration/test)."""
    n = int(hashlib.sha256(f"RW2-goal-{goal}".encode()).hexdigest()[:8], 16) % 100
    return "train" if n < 65 else "validation" if n < 75 else "calibration" if n < 80 else "test"


def execute(task: Task, actions: list[str]) -> dict:
    """Trusted executor. Stops at finish or budget."""
    x = task.start
    steps = []
    done = False
    for i, a in enumerate(actions):
        if done or i >= task.budget:
            break
        nx, legal, done = exact_step(x, a, task.limit)
        steps.append({"x": x, "action": a, "next_x": nx, "legal": legal, "terminated": done})
        x = nx
    success = bool(done and x == task.goal)
    ret = (1.0 if success else 0.0) - 0.01 * len(steps) - 0.1 * sum(not s["legal"] for s in steps)
    return {"steps": steps, "success": success, "final_x": x, "return": ret, "terminated": done}


# ----------------------------------------------------------------------------- task pools

ID_LIMITS = (32, 48, 64)
OOD_RANGE_LIMITS = (96, 128)


def _pick_limit(s: int, g: int, limits) -> int | None:
    ok = [L for L in limits if max(abs(s), abs(g)) <= L]
    if not ok:
        return None
    h = int(hashlib.sha256(f"RW2-limit-{s}-{g}".encode()).hexdigest()[:8], 16)
    return ok[h % len(ok)]


def enumerate_families(limits, min_opt, max_opt, cond=None):
    """One task family per (start, goal); its limit is a deterministic nuisance draw among valid limits."""
    fams = []
    R = max(limits)
    for s in range(-R, R + 1):
        for g in range(-R, R + 1):
            if s == g or (cond is not None and not cond(s, g)):
                continue
            L = _pick_limit(s, g, limits)
            if L is None:
                continue
            o = opt_moves(s, g, L)
            if o is None or not (min_opt <= o <= max_opt):
                continue
            fams.append((s, g, L, o))
    return fams


POOL_DEFS = {
    "id": (ID_LIMITS, 1, 8, None),
    "long": (ID_LIMITS, 9, 12, None),
    "range": (OOD_RANGE_LIMITS, 1, 8, lambda s, g: max(abs(s), abs(g)) > 64),
    "limitonly": (OOD_RANGE_LIMITS, 1, 8, lambda s, g: max(abs(s), abs(g)) <= 64),
}


def build_pools(seed: int = 20261005) -> dict:
    """Deterministic task pools split by goal value. Each family appears in exactly one pool."""
    rng = random.Random(seed)
    pools = {}
    for name, (limits, lo, hi, cond) in POOL_DEFS.items():
        pools[name] = {}
        for s, g, L, o in enumerate_families(limits, lo, hi, cond):
            pools[name].setdefault(goal_split(g), []).append((s, g, L, o))
        for sp in pools[name]:
            rng.shuffle(pools[name][sp])
    return pools


def eval_tasks(pools: dict, pool: str, split: str, n: int, seed: int, infeasible: bool = False,
               budget_delta: int = 0) -> list[Task]:
    """One task per family. Feasible: budget = opt+1+slack, slack in {0,1,2}. Infeasible twins use the SAME first
    families with budget = opt+1-d, d in {1,2} (budget >= 1). budget_delta perturbs feasible budgets (control)."""
    rng = random.Random(f"{pool}-{split}-{seed}")
    out = []
    for s, g, L, o in pools[pool][split]:
        if len(out) >= n:
            break
        slack = rng.choice((0, 1, 2))
        d = rng.choice((1, 2))
        if infeasible:
            b = o + 1 - d
            if b < 1:
                b = o
            out.append(Task(s, g, L, b))
        else:
            out.append(Task(s, g, L, o + 1 + slack + budget_delta))
    return out


# ----------------------------------------------------------------------------- rendering

SYS_PROPOSER = (
    "RegisterWorld planner. add1: x+1. sub1: x-1. double: 2x. halve: x/2, illegal if x is odd. "
    "Moves past |x|>limit are illegal. finish ends the episode. Reach the goal, then finish, within budget actions. "
    "Reply with a JSON plan for the requested outcome."
)
SYS_FORWARD = (
    "You simulate RegisterWorld. A register holds an integer x. Actions: add1 (x+1), sub1 (x-1), double (2x), "
    "halve (x/2; illegal when x is odd), finish (ends the episode). A move that would make |x| exceed the limit "
    "is illegal and leaves x unchanged. Given x, the limit and one action, reply with the resulting JSON state."
)


def render_chat(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def proposer_user(x: int, goal: int, limit: int, budget: int, outcome: str = "success") -> str:
    return canonical({"budget": budget, "goal": goal, "limit": limit, "outcome": outcome, "x": x})


def proposer_prompt(x, goal, limit, budget, outcome="success") -> str:
    return render_chat(SYS_PROPOSER, proposer_user(x, goal, limit, budget, outcome))


def target_actions(actions: list[str]) -> str:
    return canonical({"actions": actions})


def target_states(steps: list[dict]) -> str:
    return canonical({"steps": [{"action": s["action"], "x": s["next_x"]} for s in steps]})


def forward_prompt(x: int, action: str, limit: int) -> str:
    return render_chat(SYS_FORWARD, canonical({"action": action, "limit": limit, "x": x}))


def forward_target(nx: int, legal: bool, terminated: bool) -> str:
    return canonical({"legal": legal, "terminated": terminated, "x": nx})


# ----------------------------------------------------------------------------- parsing (model output is untrusted)

MAX_PLAN = 24


def parse_plan(text: str, mode: str):
    """Return (actions, claimed_states|None) or None if malformed. Strict complete JSON."""
    try:
        obj = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if mode == "actions":
        if set(obj) != {"actions"} or not isinstance(obj["actions"], list):
            return None
        acts = obj["actions"]
        states = None
    else:
        if set(obj) != {"steps"} or not isinstance(obj["steps"], list):
            return None
        acts, states = [], []
        for st in obj["steps"]:
            if not isinstance(st, dict) or set(st) != {"action", "x"}:
                return None
            if type(st["x"]) is not int:
                return None
            acts.append(st["action"])
            states.append(st["x"])
    if not acts or len(acts) > MAX_PLAN or any(a not in ACTIONS for a in acts):
        return None
    return acts, states


def parse_forward(text: str):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != {"legal", "terminated", "x"}:
        return None
    if type(obj["legal"]) is not bool or type(obj["terminated"]) is not bool or type(obj["x"]) is not int:
        return None
    return obj["x"], obj["legal"], obj["terminated"]


# ----------------------------------------------------------------------------- training data

def noisy_plan(plan: list[str], rng: random.Random) -> list[str]:
    p = list(plan)
    i = rng.randrange(len(p))
    p[i] = rng.choice([a for a in ACTIONS if a != p[i]])
    return p


def trajectories_for(task: Task, rng: random.Random) -> list[tuple[str, list[str]]]:
    opt = sample_optimal_plan(task.start, task.goal, task.limit, rng)
    rnd = [rng.choice(ACTIONS[:-1]) for _ in range(task.budget - 1)] + ["finish"]
    if rng.random() < 0.5:  # random-length random walk with an early finish
        k = rng.randrange(0, task.budget)
        rnd = [rng.choice(ACTIONS[:-1]) for _ in range(k)] + ["finish"]
    return [("opt", opt), ("noisy", noisy_plan(opt, rng)), ("random", rnd)]


TRAIN_SLACKS = (0, 1, 2, 3, 4, 5, 6)  # amendment SC-04/S6: budgets up to opt+7 appear in training


def build_training_rows(pools: dict, n_families: int, seed: int, pool: str = "id") -> dict:
    """Proposer rows (every suffix) and forward rows (every executed transition) from TRAIN families only."""
    rng = random.Random(seed)
    fams = pools[pool]["train"][:n_families]
    prop, fwd = [], []
    for s, g, L, o in fams:
        task = Task(s, g, L, o + 1 + rng.choice(TRAIN_SLACKS))
        for pol, acts in trajectories_for(task, rng):
            ep = execute(task, acts)
            steps = ep["steps"]
            for t in range(len(steps)):
                suffix = steps[t:]
                x_t = steps[t]["x"]
                rem = task.budget - t
                ok = bool(suffix[-1]["terminated"] and suffix[-1]["next_x"] == g)
                prop.append({
                    "family": task.family, "policy": pol, "t": t,
                    "x": x_t, "goal": g, "limit": L, "budget": rem, "outcome": "success" if ok else "failure",
                    "actions": [st["action"] for st in suffix],
                    "steps": [{"action": st["action"], "next_x": st["next_x"]} for st in suffix],
                })
            for st in steps:
                fwd.append({"family": task.family, "x": st["x"], "action": st["action"], "limit": L,
                            "next_x": st["next_x"], "legal": st["legal"], "terminated": st["terminated"]})
    return {"proposer": prop, "forward": fwd}


def proposer_sft(rows: list[dict], mode: str) -> list[dict]:
    out = []
    for r in rows:
        p = proposer_prompt(r["x"], r["goal"], r["limit"], r["budget"], r["outcome"])
        t = target_actions(r["actions"]) if mode == "actions" else target_states(r["steps"])
        out.append({"prompt": p, "target": t})
    return out


def forward_sft(rows: list[dict]) -> list[dict]:
    return [{"prompt": forward_prompt(r["x"], r["action"], r["limit"]),
             "target": forward_target(r["next_x"], r["legal"], r["terminated"])} for r in rows]


# ----------------------------------------------------------------------------- model-free floors (exact dynamics)

def greedy_episode(task: Task, lookahead: int = 1, step_fn=None) -> dict:
    step_fn = step_fn or (lambda x, a, L: exact_step(x, a, L))
    x = task.start
    acts = []
    for _ in range(task.budget):
        if x == task.goal:
            acts.append("finish")
            break
        best, bd = None, None
        for a in MOVES:
            nx, legal, _ = step_fn(x, a, task.limit)
            if not legal:
                continue
            d = abs(nx - task.goal)
            if lookahead >= 2:
                for b in MOVES:
                    nnx, l2, _ = step_fn(nx, b, task.limit)
                    if l2:
                        d = min(d, abs(nnx - task.goal))
            if bd is None or d < bd:
                best, bd = a, d
        acts.append(best)
        x, _, _ = step_fn(x, best, task.limit)
    return execute(task, acts)
