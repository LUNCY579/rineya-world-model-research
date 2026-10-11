"""Blocksworld (PlanBench 4-op STRIPS domain) under BWM contracts: exact semantics, PDDL I/O, generator, VAL.

State = frozenset of facts: ('clear',x) ('ontable',x) ('handempty',) ('holding',x) ('on',x,y).
Actions: ('pick-up',x) ('put-down',x) ('stack',x,y) ('unstack',x,y). An inapplicable action is illegal and leaves the
state unchanged (BWM convention); PlanBench/VAL scoring requires every action of a plan to be applicable.
The exact simulator, BFS and VAL are data-collection / evaluation instruments only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import subprocess
import tempfile
from collections import deque
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench" / "planbench"
DOMAIN_PDDL = BENCH / "blocksworld" / "generated_domain.pddl"
VAL_EXE = BENCH / "VAL" / "bin" / "validate" / "validate.exe"
NAMES = list("abcdefghijklmno")


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------- semantics

def applicable(s: frozenset, a: tuple) -> bool:
    op = a[0]
    if op == "pick-up":
        x = a[1]
        return ("clear", x) in s and ("ontable", x) in s and ("handempty",) in s
    if op == "put-down":
        return ("holding", a[1]) in s
    if op == "stack":
        x, y = a[1], a[2]
        return x != y and ("clear", y) in s and ("holding", x) in s
    if op == "unstack":
        x, y = a[1], a[2]
        return ("on", x, y) in s and ("clear", x) in s and ("handempty",) in s
    return False


def step(s: frozenset, a: tuple) -> tuple[frozenset, bool]:
    if not applicable(s, a):
        return s, False
    op = a[0]
    s = set(s)
    if op == "pick-up":
        x = a[1]
        s -= {("clear", x), ("ontable", x), ("handempty",)}
        s |= {("holding", x)}
    elif op == "put-down":
        x = a[1]
        s -= {("holding", x)}
        s |= {("clear", x), ("handempty",), ("ontable", x)}
    elif op == "stack":
        x, y = a[1], a[2]
        s -= {("clear", y), ("holding", x)}
        s |= {("handempty",), ("clear", x), ("on", x, y)}
    elif op == "unstack":
        x, y = a[1], a[2]
        s -= {("on", x, y), ("clear", x), ("handempty",)}
        s |= {("holding", x), ("clear", y)}
    return frozenset(s), True


def blocks_of(s: frozenset) -> list[str]:
    out = set()
    for f in s:
        out.update(f[1:])
    return sorted(out)


def ground_actions(objs: list[str]) -> list[tuple]:
    acts = [("pick-up", x) for x in objs] + [("put-down", x) for x in objs]
    acts += [("stack", x, y) for x in objs for y in objs if x != y]
    acts += [("unstack", x, y) for x in objs for y in objs if x != y]
    return acts


def goal_met(s: frozenset, goal: frozenset) -> bool:
    return goal <= s


def bfs_plan(init: frozenset, goal: frozenset, objs: list[str], max_states: int = 200000):
    """Shortest plan (list of actions) or None. Teacher/oracle only."""
    if goal_met(init, goal):
        return []
    acts = ground_actions(objs)
    prev = {init: None}
    q = deque([init])
    while q and len(prev) < max_states:
        s = q.popleft()
        for a in acts:
            if not applicable(s, a):
                continue
            ns, _ = step(s, a)
            if ns in prev:
                continue
            prev[ns] = (s, a)
            if goal_met(ns, goal):
                plan = []
                cur = ns
                while prev[cur] is not None:
                    ps, pa = prev[cur]
                    plan.append(pa)
                    cur = ps
                return plan[::-1]
            q.append(ns)
    return None


def execute(init: frozenset, goal: frozenset, plan: list[tuple]) -> dict:
    """Exact execution; success = every action applicable and goal met at the end (VAL semantics)."""
    s = init
    for i, a in enumerate(plan):
        if not applicable(s, a):
            return {"success": False, "valid": False, "fail_step": i, "final": s}
        s, _ = step(s, a)
    return {"success": goal_met(s, goal), "valid": True, "fail_step": None, "final": s}


# ----------------------------------------------------------------------------- PDDL I/O

def parse_problem(path: Path):
    s = Path(path).read_text().lower()
    objs = re.search(r"\(:objects([^)]*)\)", s).group(1).split()
    init_txt = re.search(r"\(:init(.*?)\)\s*\(:goal", s, re.S).group(1)
    goal_txt = re.search(r"\(:goal\s*\(and(.*)\)\s*\)\s*\)", s, re.S).group(1)
    facts = lambda t: frozenset(tuple(x.split()) for x in re.findall(r"\(([^()]+)\)", t))
    return sorted(objs), facts(init_txt), facts(goal_txt)


def write_problem(objs, init, goal, name="bwm") -> str:
    f = lambda fs: "\n".join("(" + " ".join(x) + ")" for x in sorted(fs))
    return (f"(define (problem {name})\n(:domain blocksworld-4ops)\n(:objects {' '.join(objs)} )\n(:init\n{f(init)}\n)\n"
            f"(:goal\n(and\n{f(goal)})\n)\n)\n")


def plan_to_pddl(plan) -> str:
    return "\n".join("(" + " ".join(a) + ")" for a in plan) + "\n"


def val_check(problem_path: Path, plan, domain_path: Path = DOMAIN_PDDL) -> bool:
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d) / "plan.txt"
        pp.write_text(plan_to_pddl(plan))
        out = subprocess.run([str(VAL_EXE), str(domain_path), str(problem_path), str(pp)], capture_output=True,
                             text=True, timeout=60).stdout
    return "Plan valid" in out


# ----------------------------------------------------------------------------- generator (training only)

def random_towers(objs: list[str], rng: random.Random, p_stack: float = 0.6) -> list[list[str]]:
    o = list(objs)
    rng.shuffle(o)
    towers = []
    for b in o:
        if towers and rng.random() < p_stack:
            rng.choice(towers).append(b)
        else:
            towers.append([b])
    return towers


def towers_to_state(towers) -> frozenset:
    s = {("handempty",)}
    for t in towers:
        s.add(("ontable", t[0]))
        for lo, hi in zip(t, t[1:]):
            s.add(("on", hi, lo))
        s.add(("clear", t[-1]))
    return frozenset(s)


def signature(objs, init, goal) -> str:
    """Canonical form up to renaming of blocks (n <= 7 brute force)."""
    best = None
    for perm in itertools.permutations(range(len(objs))):
        m = {o: f"o{perm[i]}" for i, o in enumerate(objs)}
        ren = lambda fs: sorted(tuple([f[0]] + [m[x] for x in f[1:]]) for f in fs)
        key = canonical([ren(init), ren(goal)])
        if best is None or key < best:
            best = key
    return hashlib.sha256(best.encode()).hexdigest()


def generate(rng: random.Random, n_blocks: int, name_pool=NAMES, full_goal_frac: float = 0.0):
    """full_goal_frac > 0 (proposer data v2): with that probability the goal is every on-pair of the random target
    (the PlanBench Blocksworld-hard goal style; target drawn with p_stack 0.85); otherwise 1-4 random pairs as in v1. At 0 the random stream and the
    output are identical to v1."""
    objs = sorted(rng.sample(name_pool, n_blocks))
    init = towers_to_state(random_towers(objs, rng))
    target = towers_to_state(random_towers(objs, rng))
    ons = sorted(f for f in target if f[0] == "on")
    if not ons:
        return None
    if full_goal_frac > 0 and rng.random() < full_goal_frac:
        # hard-set style: every on-pair of a target built mostly as one or two tall towers
        ons = sorted(f for f in towers_to_state(random_towers(objs, rng, p_stack=0.85)) if f[0] == "on")
        goal = frozenset(ons)
        return (objs, init, goal) if ons and not goal_met(init, goal) else None
    k = rng.randint(1, min(4, len(ons)))
    goal = frozenset(rng.sample(ons, k))
    if goal_met(init, goal):
        return None
    return objs, init, goal


def hard_signatures(max_objs: int = 7) -> set:
    """Renaming signatures of PlanBench Blocksworld-hard instances small enough to canonicalize (6-7 blocks)."""
    sigs = set()
    for p in (BENCH / "blocksworld_hard" / "generated").glob("*.pddl"):
        objs, init, goal = parse_problem(p)
        if len(objs) <= max_objs:
            sigs.add(signature(objs, init, goal))
    return sigs


def test_signatures() -> set:
    sigs = set()
    for sub in ("blocksworld/generated_basic", "blocksworld/generated_basic_3", "unsolvable_blocksworld/generated_basic"):
        for p in (BENCH / sub).glob("*.pddl"):
            try:
                objs, init, goal = parse_problem(p)
                if len(objs) <= 7:
                    sigs.add(signature(objs, init, goal))
            except Exception:
                continue
    return sigs


# ----------------------------------------------------------------------------- rendering for the LM

def state_json(s: frozenset) -> dict:
    hold = [f[1] for f in s if f[0] == "holding"]
    return {"clear": sorted(f[1] for f in s if f[0] == "clear"),
            "hand": hold[0] if hold else "empty",
            "on": sorted([f[1], f[2]] for f in s if f[0] == "on"),
            "table": sorted(f[1] for f in s if f[0] == "ontable")}


def state_from_json(d: dict) -> frozenset | None:
    try:
        s = set()
        for x in d["clear"]:
            s.add(("clear", x))
        if d["hand"] == "empty":
            s.add(("handempty",))
        else:
            s.add(("holding", d["hand"]))
        for x, y in d["on"]:
            s.add(("on", x, y))
        for x in d["table"]:
            s.add(("ontable", x))
        return frozenset(s)
    except Exception:
        return None


def action_str(a: tuple) -> str:
    return " ".join(a)


def parse_action(t: str, objs: list[str]) -> tuple | None:
    p = t.strip().split()
    if not p:
        return None
    if p[0] in ("pick-up", "put-down") and len(p) == 2 and p[1] in objs:
        return tuple(p)
    if p[0] in ("stack", "unstack") and len(p) == 3 and p[1] in objs and p[2] in objs:
        return tuple(p)
    return None


SYS_PROPOSER = (
    "Blocksworld planner. Actions: pick-up X (X clear, on table, hand empty), put-down X (holding X), "
    "stack X Y (holding X, Y clear), unstack X Y (X on Y, X clear, hand empty). Reach every goal pair [X,Y] meaning "
    "X on Y. Reply with a JSON plan."
)
SYS_FORWARD = (
    "Blocksworld simulator. pick-up X needs X clear, X on table, hand empty. put-down X needs holding X. stack X Y "
    "needs holding X and Y clear. unstack X Y needs X on Y, X clear, hand empty. An action whose requirements fail is "
    "illegal and changes nothing. Given the state and one action, reply with the resulting JSON state."
)


def render_chat(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def goal_json(goal: frozenset) -> list:
    return sorted([f[1], f[2]] for f in goal if f[0] == "on")


def proposer_prompt(s: frozenset, goal: frozenset, outcome: str = "success") -> str:
    return render_chat(SYS_PROPOSER, canonical({"goal": goal_json(goal), "outcome": outcome, "state": state_json(s)}))


def forward_prompt(s: frozenset, a: tuple) -> str:
    return render_chat(SYS_FORWARD, canonical({"action": action_str(a), "state": state_json(s)}))


def forward_target(ns: frozenset, legal: bool) -> str:
    return canonical({"legal": legal, "state": state_json(ns)})


def parse_forward(text: str):
    try:
        d = json.loads(text.strip())
        if set(d) != {"legal", "state"} or type(d["legal"]) is not bool:
            return None
        s = state_from_json(d["state"])
        return None if s is None else (s, d["legal"])
    except Exception:
        return None


# Delta format (F-05 follow-up): the model names only the facts the action adds and deletes, so the output length
# does not grow with the number of blocks (state-format errors at 12-15 blocks were copy errors in long fact lists).
SYS_FORWARD_DELTA = (
    "Blocksworld simulator. pick-up X needs X clear, X on table, hand empty. put-down X needs holding X. stack X Y "
    "needs holding X and Y clear. unstack X Y needs X on Y, X clear, hand empty. An action whose requirements fail is "
    "illegal and changes nothing. Given the state and one action, reply with JSON: whether it is legal and the facts "
    "it adds and deletes."
)


def fact_str(f: tuple) -> str:
    return " ".join(f)


def forward_prompt_delta(s: frozenset, a: tuple) -> str:
    return render_chat(SYS_FORWARD_DELTA, canonical({"action": action_str(a), "state": state_json(s)}))


def forward_target_delta(s: frozenset, ns: frozenset, legal: bool) -> str:
    return canonical({"add": sorted(fact_str(f) for f in ns - s), "del": sorted(fact_str(f) for f in s - ns),
                      "legal": legal})


def parse_forward_delta(text: str, s: frozenset):
    """(next_state, legal) or None. A deleted fact that is not in s makes the answer unsupported (None)."""
    try:
        d = json.loads(text.strip())
        if set(d) != {"add", "del", "legal"} or type(d["legal"]) is not bool:
            return None
        add = {tuple(x.split()) for x in d["add"]}
        dele = {tuple(x.split()) for x in d["del"]}
        if not dele <= s:
            return None
        return frozenset((s - dele) | add), d["legal"]
    except Exception:
        return None


def parse_plan_prefix(text: str, objs: list[str]):
    """Leading complete actions of a possibly truncated action-only plan (generation stopped by a token cap).
    Returns (actions, None) with >= 1 action, or None. Used only with the beam planner's rollout cap, where only the
    first R actions of a candidate are ever simulated."""
    full = parse_plan(text, objs, "actions")
    if full is not None:
        return full
    head = text.split('"actions"', 1)
    if len(head) < 2:
        return None
    acts = []
    for m in re.finditer(r'"([^"]*)"', head[1]):
        a = parse_action(m.group(1), objs)
        if a is None:
            break
        acts.append(a)
    return (acts, None) if acts else None


def target_actions(plan) -> str:
    return canonical({"actions": [action_str(a) for a in plan]})


def target_states(steps) -> str:
    """steps: list of (action, next_state)."""
    return canonical({"steps": [{"action": action_str(a), "state": state_json(ns)} for a, ns in steps]})


def parse_plan(text: str, objs: list[str], mode: str = "actions"):
    try:
        d = json.loads(text.strip())
    except Exception:
        return None
    try:
        if mode == "actions":
            acts = [parse_action(t, objs) for t in d["actions"]]
            states = None
        else:
            acts = [parse_action(st["action"], objs) for st in d["steps"]]
            states = [state_from_json(st["state"]) for st in d["steps"]]
    except Exception:
        return None
    if not acts or any(a is None for a in acts) or len(acts) > 60:
        return None
    return acts, states
