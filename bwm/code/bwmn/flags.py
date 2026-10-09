"""FlagWorld: STRIPS-style flag planning (the Forge v2 prerequisite_plan family) under BWM contracts.

World generation follows forge_rl/tasks.make_world('prerequisite_plan'): n flags (5-7; 8 for transfer), 2n+2 actions,
each with 0-2 required-on flags, 0-1 required-off flag, one added flag and (60%) one cleared flag; delete before add.
BWM semantics: a plan is a list of action names ending with `finish`. An inapplicable action is illegal and leaves the
state unchanged; every action, including finish, consumes one step of the budget. Success = finish while every goal
flag is on, within the budget. Budgets are tight: shortest-plan steps + 1 + slack. Impossible worlds exist (~1/4).
The exact executor and BFS are collection/evaluation instruments only.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Act:
    name: str
    pre_on: tuple
    pre_off: tuple
    add: int
    delete: int | None


@dataclass(frozen=True)
class World:
    n: int
    actions: tuple  # of Act
    initial: str  # bitstring, index i = flag i
    goal: tuple  # flags that must be on
    budget: int = 0

    @property
    def family(self) -> str:
        return hashlib.sha256(canonical(self.spec()).encode()).hexdigest()

    def spec(self) -> dict:
        return {"n": self.n, "init": self.initial, "goal": list(self.goal),
                "acts": {a.name: render_action(a) for a in self.actions}}

    def act(self, name: str) -> Act | None:
        for a in self.actions:
            if a.name == name:
                return a
        return None


def render_action(a: Act) -> str:
    parts = [f"?{b}" for b in a.pre_on] + [f"!{b}" for b in a.pre_off] + ([f"-{a.delete}"] if a.delete is not None else []) + [f"+{a.add}"]
    return " ".join(parts)


def make_world(rng: random.Random, transfer: bool = False) -> World:
    n = 8 if transfer else rng.choice([5, 6, 7])
    init = "".join(rng.choice("01") for _ in range(n))
    goal = tuple(sorted(rng.sample(range(n), rng.randint(2, min(4, n)))))
    acts = []
    for i in range(2 * n + 2):
        pre_on = tuple(sorted(rng.sample(range(n), rng.randrange(3))))
        pre_off = tuple(b for b in sorted(rng.sample(range(n), rng.randrange(2))) if b not in pre_on)
        add = rng.randrange(n)
        dele = rng.randrange(n) if rng.random() < 0.6 else None
        if dele == add:
            dele = None
        acts.append(Act(f"A{i}", pre_on, pre_off, add, dele))
    return World(n, tuple(acts), init, goal)


def step(state: str, name: str, w: World) -> tuple[str, bool, bool]:
    """(next_state, legal, terminated). Unknown names are illegal no-ops."""
    if name == "finish":
        return state, True, True
    a = w.act(name)
    if a is None or any(state[b] != "1" for b in a.pre_on) or any(state[b] != "0" for b in a.pre_off):
        return state, False, False
    s = list(state)
    if a.delete is not None:
        s[a.delete] = "0"
    s[a.add] = "1"
    return "".join(s), True, False


def is_goal(state: str, w: World) -> bool:
    return all(state[b] == "1" for b in w.goal)


def bfs_dist(w: World, start: str | None = None) -> dict:
    """Shortest number of real (non-finish) moves from `start` to every reachable state."""
    start = w.initial if start is None else start
    d = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        for a in w.actions:
            ns, legal, _ = step(s, a.name, w)
            if legal and ns not in d:
                d[ns] = d[s] + 1
                q.append(ns)
    return d


def opt_moves(w: World, start: str | None = None) -> int | None:
    d = bfs_dist(w, start)
    goals = [v for s, v in d.items() if is_goal(s, w)]
    return min(goals) if goals else None


def dist_to_goal(w: World) -> dict:
    """Moves-to-goal for every state reachable from the initial state (reverse BFS over the reachable graph)."""
    reach = bfs_dist(w)
    rev = {s: [] for s in reach}
    for s in reach:
        for a in w.actions:
            ns, legal, _ = step(s, a.name, w)
            if legal and ns != s:
                rev.setdefault(ns, []).append(s)
    d = {}
    q = deque()
    for s in reach:
        if is_goal(s, w):
            d[s] = 0
            q.append(s)
    while q:
        s = q.popleft()
        for p in rev.get(s, []):
            if p not in d:
                d[p] = d[s] + 1
                q.append(p)
    return d


def optimal_next(w: World, state: str, dtg: dict | None = None) -> list[str]:
    dtg = dtg or dist_to_goal(w)
    if state not in dtg:
        return []
    if dtg[state] == 0:
        return ["finish"]
    out = []
    for a in w.actions:
        ns, legal, _ = step(state, a.name, w)
        if legal and dtg.get(ns, 10**9) == dtg[state] - 1:
            out.append(a.name)
    return out


def sample_optimal_plan(w: World, rng: random.Random, state: str | None = None) -> list[str]:
    dtg = dist_to_goal(w)
    s = w.initial if state is None else state
    plan = []
    while True:
        opts = optimal_next(w, s, dtg)
        if not opts:
            raise ValueError("unreachable")
        a = rng.choice(opts)
        plan.append(a)
        if a == "finish":
            return plan
        s = step(s, a, w)[0]


def execute(w: World, actions: list[str], budget: int) -> dict:
    s = w.initial
    steps, done = [], False
    for i, a in enumerate(actions):
        if done or i >= budget:
            break
        ns, legal, done = step(s, a, w)
        steps.append({"state": s, "action": a, "next": ns, "legal": legal, "terminated": done})
        s = ns
    success = bool(done and is_goal(s, w))
    return {"steps": steps, "success": success, "final": s, "terminated": done,
            "return": (1.0 if success else 0.0) - 0.01 * len(steps) - 0.1 * sum(not x["legal"] for x in steps)}


def split_of(w: World) -> str:
    n = int(w.family[:8], 16) % 100
    return "train" if n < 65 else "validation" if n < 75 else "calibration" if n < 80 else "test"


def world_with_budget(w: World, budget: int) -> World:
    return World(w.n, w.actions, w.initial, w.goal, budget)


def build_worlds(count: int, seed: int, transfer: bool = False, min_opt: int = 1, max_opt: int = 8):
    """Deterministic list of (world, opt) with opt in [min_opt, max_opt] (feasible) plus impossible worlds (opt None)."""
    rng = random.Random(seed)
    feas, imp = [], []
    while len(feas) < count:
        w = make_world(rng, transfer)
        o = opt_moves(w)
        if o is None:
            imp.append(w)
        elif min_opt <= o <= max_opt:
            feas.append((w, o))
    return feas, imp


# ----------------------------------------------------------------------------- rendering

SYS_PROPOSER = (
    "FlagWorld planner. The state is a flag string; character i is flag i (1 on, 0 off). Each action lists "
    "?i (flag i must be on), !i (flag i must be off), -i (clear flag i), +i (set flag i); clear happens before set. "
    "An action whose requirements fail is illegal and changes nothing. finish ends the episode. Turn every goal flag "
    "on, then finish, within budget actions. Reply with a JSON plan for the requested outcome."
)
SYS_FORWARD = (
    "FlagWorld simulator. The state is a flag string; character i is flag i. ?i requires flag i on, !i requires it "
    "off, -i clears it, +i sets it; clear happens before set. If a requirement fails the action is illegal and nothing "
    "changes. finish ends the episode. Given the state and one action, reply with the resulting JSON state."
)


def render_chat(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def proposer_prompt(w: World, state: str, remaining: int, outcome: str = "success") -> str:
    u = canonical({"actions": {a.name: render_action(a) for a in w.actions}, "budget": remaining,
                   "goal": list(w.goal), "outcome": outcome, "state": state})
    return render_chat(SYS_PROPOSER, u)


def forward_prompt(state: str, action_text: str) -> str:
    """Goal-blind: only the current state and the action's own definition (or 'finish')."""
    return render_chat(SYS_FORWARD, canonical({"action": action_text, "state": state}))


def forward_target(ns: str, legal: bool, term: bool) -> str:
    return canonical({"legal": legal, "state": ns, "terminated": term})


def target_actions(actions: list[str]) -> str:
    return canonical({"actions": actions})


def target_states(steps: list[dict]) -> str:
    return canonical({"steps": [{"action": s["action"], "state": s["next"]} for s in steps]})


def parse_plan(text: str, mode: str, w: World):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    names = {a.name for a in w.actions} | {"finish"}
    if mode == "actions":
        if set(obj) != {"actions"} or not isinstance(obj["actions"], list):
            return None
        acts, states = obj["actions"], None
    else:
        if set(obj) != {"steps"} or not isinstance(obj["steps"], list):
            return None
        acts, states = [], []
        for st in obj["steps"]:
            if not isinstance(st, dict) or set(st) != {"action", "state"} or not isinstance(st["state"], str):
                return None
            acts.append(st["action"])
            states.append(st["state"])
    if not acts or len(acts) > 24 or any(not isinstance(a, str) or a not in names for a in acts):
        return None
    return acts, states


def parse_forward(text: str, n: int):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != {"legal", "state", "terminated"}:
        return None
    s = obj["state"]
    if type(obj["legal"]) is not bool or type(obj["terminated"]) is not bool or not isinstance(s, str) \
            or len(s) != n or set(s) - {"0", "1"}:
        return None
    return s, obj["legal"], obj["terminated"]


# ----------------------------------------------------------------------------- pools, data

ID_OPT = (3, 8)


def build_pool(n_worlds: int, seed: int, transfer: bool = False, lo: int = 3, hi: int = 8, imp_frac: float = 0.0):
    """Deterministic worlds split by family hash. Returns {split: [(world, opt)]} and {split: [impossible worlds]}."""
    rng = random.Random(seed)
    feas, imp = {}, {}
    tries = 0
    while sum(len(v) for v in feas.values()) < n_worlds and tries < 50 * n_worlds:
        tries += 1
        w = make_world(rng, transfer)
        o = opt_moves(w)
        sp = split_of(w)
        if o is None:
            if len(imp.get(sp, [])) < imp_frac * n_worlds:
                imp.setdefault(sp, []).append(w)
        elif lo <= o <= hi:
            feas.setdefault(sp, []).append((w, o))
    return feas, imp


def trajectories(w: World, budget: int, rng: random.Random):
    opt = sample_optimal_plan(w, rng)
    noisy = list(opt)
    i = rng.randrange(len(noisy))
    noisy[i] = rng.choice([a.name for a in w.actions] + ["finish"])
    k = rng.randrange(0, budget)
    rnd = [rng.choice(w.actions).name for _ in range(k)] + ["finish"]
    return [("opt", opt), ("noisy", noisy), ("random", rnd)]


TRAIN_SLACKS = (0, 1, 2, 3, 4)


def build_training_rows(worlds, seed: int):
    rng = random.Random(seed)
    prop, fwd = [], []
    for w, o in worlds:
        b = o + 1 + rng.choice(TRAIN_SLACKS)
        for pol, acts in trajectories(w, b, rng):
            ex = execute(w, acts, b)
            st = ex["steps"]
            for t in range(len(st)):
                suf = st[t:]
                ok = bool(suf[-1]["terminated"] and is_goal(suf[-1]["next"], w))
                prop.append({"world": w, "state": st[t]["state"], "budget": b - t,
                             "outcome": "success" if ok else "failure", "policy": pol,
                             "actions": [s["action"] for s in suf], "steps": suf})
            for s in st:
                a = w.act(s["action"])
                fwd.append({"state": s["state"], "action": render_action(a) if a else "finish",
                            "next": s["next"], "legal": s["legal"], "terminated": s["terminated"]})
    return prop, fwd


def proposer_sft(rows, mode: str):
    out = []
    for r in rows:
        p = proposer_prompt(r["world"], r["state"], r["budget"], r["outcome"])
        t = target_actions(r["actions"]) if mode == "actions" else target_states(r["steps"])
        out.append({"prompt": p, "target": t})
    return out


def forward_sft(rows):
    return [{"prompt": forward_prompt(r["state"], r["action"]),
             "target": forward_target(r["next"], r["legal"], r["terminated"])} for r in rows]


def greedy_episode(w: World, budget: int, look: int = 1, stepf=None):
    """Model-free floor: maximise the number of goal flags on (exact dynamics unless stepf is given)."""
    stepf = stepf or (lambda s, name: step(s, name, w))
    s, acts = w.initial, []
    for _ in range(budget):
        if is_goal(s, w):
            acts.append("finish")
            break
        best, bv = None, None
        for a in w.actions:
            ns, legal, _ = stepf(s, a.name)
            if not legal or ns == s:
                continue
            v = sum(ns[b] == "1" for b in w.goal)
            if look >= 2:
                for b2 in w.actions:
                    n2, l2, _ = stepf(ns, b2.name)
                    if l2:
                        v = max(v, sum(n2[b] == "1" for b in w.goal) - 0.5)
            if bv is None or v > bv:
                best, bv = a.name, v
        if best is None:
            break
        acts.append(best)
        s = step(s, best, w)[0]
    return execute(w, acts, budget)
