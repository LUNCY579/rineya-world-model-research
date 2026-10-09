"""Ledger PLANNING problems for the joint RCWM-T1 x BWM-001 beam test (joint roadmap item 3). Torch-free and model-free:
no model, tokenizer or model output is used anywhere in this module.

  python -B scripts/lbeam_problems.py --make-dev            # writes docs/ledger_beam/dev_set.jsonl + dev_set.SOURCE.json
                                                            # (create-only; full freshness incl. the name-skipping harvest)
  python -B scripts/lbeam_problems.py --verify <set.jsonl>  # re-checks every instance from the file (BFS depth, plans,
                                                            # fixed candidates, oracle); --fresh also re-checks freshness
  python -B scripts/lbeam_problems.py --describe <set.jsonl>

Domain (rcwmt.env, ledger_replay; Titan's one-step semantics):
  state   hashable canonical tuple (("A", a), ("B", b), ("C", c), ("D", d)), accounts sorted, int balances
  action  hashable event tuple (kind, account, other, amount, status, tick) == ACTION_FIELDS; action_to_event() gives the
          exact event dict of env.gen_event (keys account, amount, kind, other, status, tick)
  step    env.step: status 'posted' applies add / scale / transfer; any other status is a no-op. The ledger has NO
          illegal events (every event applies to every state), so legality never fails in this domain.
  goal    hashable tuple of (account, target) pairs, sorted: exact target balances for all four accounts ("all") or for
          a subset of 1-3 accounts ("subset"); goal_met(state, goal) iff every listed account equals its target;
          unmet(state, goal) = sum over listed accounts of |balance - target| (0 iff met; lower = closer)

Instance (one per stratum slot; RNG keyed on (set seed, range, depth, slot, attempt), so slots are independent):
  init     4 balances from env.gen_start's recorded pool restricted to |b| <= 98 (T2's training range)
  vocab    V = 8 actions the planner may choose: 2-5 effective posted CORE actions (a hidden seed plan of length d uses
           every core action at least once), 1-2 VOID TWINS (a core action of the seed plan with status 'void' and a new
           tick: identical arithmetic fields, no effect), and posted RANDOM fill actions (>= 1 of the vocab moves AWAY
           from the goal at init). Kinds and amounts from the recorded marginal (rcwmt.env.load_marginals); posted
           actions are effective (add / transfer amount != 0, scale amount != 1) and perform pairwise different
           arithmetic (signature(): kind, account, amount, + other for transfers); ticks distinct within the instance;
           every action tuple belongs to exactly one instance of a set (so F calls can be attributed to instances).
  goal     the seed plan's end state on all accounts (p .5) or on 1-3 of the accounts it changed
  oracle   BFS over env.step semantics with the instance's vocab (vstep, cross-checked against env.step and the FDS
           oracle on every stored transition): optimal depth d* in 2..6, one optimal plan, the number of optimal plans.
           A slot of depth d accepts a draw only if d* == d (the seed plan has length d, so d* <= d; shortcuts are
           rejected).
  strata   depth d* in 2..6 x range:
           'in'  : init, goal, every vocab amount (the 21 trained values -9..12) and every state on the stored optimal
                   plan inside T2's training range (|b| <= 98); the stored plan is the in-range BFS plan and must be
                   optimal overall (bounded and unbounded BFS agree on d*).
           'oor' : REPORT STRATUM ONLY (PREREG-3 maps whether the rule holds out of range): posted add/transfer amounts
                   +-13..99 w.p. .5 and init balances +-100..999 w.p. .4 per account; the stored optimal plan must touch
                   an out-of-range amount or state. Statuses stay posted / void in both strata.
  fixed    the PREREG-2-style fixed-candidate view: K = 8 candidate plans of length d*, one optimal plus near-miss and
           random distractors (void_swap: one posted step replaced by a void action; order_swap: two adjacent steps
           swapped; substitute: one step replaced by another vocab action; random), each distractor truly NOT reaching
           the goal, all distinct, seeded order. No prefix of length < d* can reach the goal (d* is minimal), so
           selection depends only on final states.

Freshness (binding; load_exclusions()): no (before, event) payload and no FDS operative content of any binding
transition (init x every vocab action, every step of the stored optimal plan, every step of every fixed candidate)
equals one of: the exclusion union of scripts/p3_make_bank.load_all_sources (the historical records harvested by
scripts/l1_make_ood.harvest_historical, which skips files named HSKILL_CONFIRM_V2* by NAME, never opened or hashed;
rcwmt.bank.load_exclusions() is NOT called; the four RCWM banks; the T2 files; SELECT-VIEWS-001; the L1 OOD bank; the
pinned FDS-era banks), every transition of data/bank_p3_oor.jsonl (PREREG-3), and every {before, event} payload of every
other data/*.json[l] file (incl. the concurrent T2b/T2br data; recorded with sha256 at build time). A colliding draw is
redrawn (next attempt of the same slot), so a set is deterministic given (seed, spec, exclusion union) and its sha
equals the no-exclusion build whenever no draw collided (recorded).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rcwmt import env  # noqa: E402  (torch-free)

GEN_VERSION = "lbeam-problems-v1"
LEDGER = "ledger_replay"
ACCOUNTS = env.ACCOUNTS
AIDX = {a: i for i, a in enumerate(ACCOUNTS)}
ACTION_FIELDS = ("kind", "account", "other", "amount", "status", "tick")
KIND_CODE = {"add": 0, "scale": 1, "transfer": 2}
IN_RANGE_MAX_ABS = 98            # T2 training: max |before| = 98 (data/t2_train.jsonl, PREREG_3 §1)
D_MIN, D_MAX = 2, 6
V_DEFAULT = 8
K_FIXED = 8
OOR_AMT = (13, 99)
OOR_BAL = (100, 999)
DEV_SEED = 20261141              # not used by any earlier RCWM-T1 builder, analysis or bootstrap (grep, 2026-10-09)
MAX_ATTEMPTS = 5000
DOCS = ROOT / "docs" / "ledger_beam"
RANGES = ("in", "oor")


# ------------------------------------------------------------------------------------------------ canonical forms
def state_from_dict(d: dict) -> tuple:
    return tuple((a, int(d[a])) for a in sorted(d))


def state_to_dict(s) -> dict:
    return {a: v for a, v in s}


def action_to_event(a) -> dict:
    """The exact event dict Titan's frame carries (keys as env.gen_event: account, amount, kind, other, status, tick)."""
    return dict(zip(ACTION_FIELDS, a))


def event_to_action(ev: dict) -> tuple:
    return tuple(ev[k] for k in ACTION_FIELDS)


def is_state(s) -> bool:
    return (isinstance(s, tuple) and len(s) == len(ACCOUNTS) and all(isinstance(x, tuple) and len(x) == 2 for x in s)
            and tuple(x[0] for x in s) == ACCOUNTS
            and all(isinstance(x[1], int) and not isinstance(x[1], bool) for x in s))


def is_action(a) -> bool:
    return (isinstance(a, tuple) and len(a) == 6 and a[0] in KIND_CODE and a[1] in AIDX and a[2] in AIDX and a[1] != a[2]
            and isinstance(a[3], int) and not isinstance(a[3], bool) and isinstance(a[4], str)
            and isinstance(a[5], int) and not isinstance(a[5], bool))


def step_state(s: tuple, a: tuple) -> tuple:
    """env.step on canonical forms (the registered semantics)."""
    return state_from_dict(env.step(LEDGER, state_to_dict(s), action_to_event(a)))


def step_ignore_status(s: tuple, a: tuple) -> tuple:
    """Rule-ablated simulator (PREREG-2's F_ignore_status): every event applied as if posted. Diagnostic only."""
    return state_from_dict(env.step(LEDGER, state_to_dict(s), {**action_to_event(a), "status": "posted"}))


# fast integer form for BFS (cross-checked against env.step on every stored transition and in the tests)
def vec(s: tuple) -> tuple:
    return tuple(v for _, v in s)


def from_vec(v: tuple) -> tuple:
    return tuple(zip(ACCOUNTS, v))


def compile_action(a: tuple) -> tuple:
    return (KIND_CODE[a[0]], AIDX[a[1]], AIDX[a[2]], a[3], a[4] == "posted")


def vstep(v: tuple, c: tuple) -> tuple:
    k, i, j, amt, posted = c
    if not posted:
        return v
    x = list(v)
    if k == 0:
        x[i] += amt
    elif k == 1:
        x[i] *= amt
    else:
        x[i] -= amt
        x[j] += amt
    return tuple(x)


# ------------------------------------------------------------------------------------------------ hooks (BWM plan_beam)
def goal_met(state, goal) -> bool:
    """goal_met(state, goal): every account listed in the goal holds exactly its target balance."""
    d = dict(state)
    return all(d.get(a) == t for a, t in goal)


def unmet(state, goal) -> int:
    """unmet(state, goal) = sum over goal accounts of |balance - target| (0 iff goal_met; lower = closer)."""
    d = dict(state)
    return sum(abs(d[a] - t) for a, t in goal)


def _goal_c(goal) -> tuple:
    return tuple((AIDX[a], t) for a, t in goal)


def _gm_vec(v, goal_c) -> bool:
    for i, t in goal_c:
        if v[i] != t:
            return False
    return True


# ------------------------------------------------------------------------------------------------ oracle BFS
def bfs(init_v: tuple, comp: list, goal_c: tuple, dmax: int = D_MAX, bound: int | None = None) -> dict | None:
    """Breadth-first search over vstep with the compiled vocabulary. Returns {depth, plan (action indices, the first
    optimal plan in BFS order with the vocab order), n_opt (number of optimal action sequences), n_goal_states,
    explored} or None if no goal state within dmax. bound: every state on a path must have |balance| <= bound."""
    if _gm_vec(init_v, goal_c):
        return {"depth": 0, "plan": [], "n_opt": 1, "n_goal_states": 1, "explored": 1}
    parent = {init_v: None}
    layer = {init_v: 1}
    explored = 1
    for d in range(1, dmax + 1):
        nxt: dict = {}
        for s, c in layer.items():
            for ai, ca in enumerate(comp):
                ns = vstep(s, ca)
                if bound is not None and (abs(ns[0]) > bound or abs(ns[1]) > bound or abs(ns[2]) > bound
                                          or abs(ns[3]) > bound):
                    continue
                if ns in nxt:
                    nxt[ns] += c
                    continue
                if ns in parent:
                    continue
                parent[ns] = (s, ai)
                nxt[ns] = c
        explored += len(nxt)
        goals = [s for s in nxt if _gm_vec(s, goal_c)]
        if goals:
            plan, s = [], goals[0]
            while parent[s] is not None:
                ps, ai = parent[s]
                plan.append(ai)
                s = ps
            plan.reverse()
            return {"depth": d, "plan": plan, "n_opt": sum(nxt[g] for g in goals), "n_goal_states": len(goals),
                    "explored": explored}
        if not nxt:
            return None
        layer = nxt
    return None


def bfs_plan(state: tuple, goal: tuple, vocab, dmax: int = D_MAX) -> list | None:
    """Shortest plan (list of action tuples) from a canonical state, or None (harness proposer / oracle only)."""
    vocab = list(vocab)
    r = bfs(vec(state), [compile_action(a) for a in vocab], _goal_c(goal), dmax)
    return None if r is None else [vocab[i] for i in r["plan"]]


# ------------------------------------------------------------------------------------------------ marginals and draws
_M: list = []


def marg():
    if not _M:
        m = env.load_marginals()                       # recorded SELECT-VIEWS/probe marginals (sha-checked by env)
        _M.append({"kinds": list(m.ledger_kinds), "amounts": list(m.ledger_amounts),
                   "amount_set": frozenset(m.ledger_amounts),
                   "bal_pool": [b for b in m.ledger_balances if abs(b) <= IN_RANGE_MAX_ABS]})
    return _M[0]


def seed_int(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _effective(kind: str, amount: int) -> bool:
    return amount != 1 if kind == "scale" else amount != 0


def _draw_posted(rng, used_ticks: set, oor_amount_p: float = 0.0) -> tuple:
    M = marg()
    while True:
        a = rng.choice(ACCOUNTS)
        other = rng.choice([x for x in ACCOUNTS if x != a])
        kind = rng.choice(M["kinds"])
        amount = rng.choice(M["amounts"])
        if kind != "scale" and oor_amount_p > 0 and rng.random() < oor_amount_p:
            amount = rng.choice((-1, 1)) * rng.randint(*OOR_AMT)
        if _effective(kind, amount):
            break
    tick = rng.choice([t for t in range(1, 100) if t not in used_ticks])
    used_ticks.add(tick)
    return (kind, a, other, amount, "posted", tick)


def signature(a: tuple) -> tuple:
    """The arithmetic an action performs when posted: (kind, account, amount) plus other for transfers."""
    return (a[0], a[1], a[3]) + ((a[2],) if a[0] == "transfer" else ())


def _draw_distinct(rng, used_ticks: set, sigs: set, oor_amount_p: float) -> tuple:
    """A posted, effective action whose arithmetic differs from every posted action already in the vocabulary (void
    twins share their source's arithmetic by design). The tick is drawn only for the accepted action."""
    while True:
        a = _draw_posted(rng, set(used_ticks), oor_amount_p)
        if signature(a) not in sigs:
            sigs.add(signature(a))
            used_ticks.add(a[5])
            return a


def _rollout(s: tuple, plan) -> list:
    out = [s]
    for a in plan:
        out.append(step_state(out[-1], a))
    return out


@dataclass
class SetSpec:
    name: str = "dev"
    seed: int = DEV_SEED
    n_in_per_depth: int = 24
    n_oor_per_depth: int = 6
    depths: tuple = (2, 3, 4, 5, 6)
    V: int = V_DEFAULT
    K: int = K_FIXED


def _fixed_candidates(rng, init, goal, vocab, opt_plan, K) -> dict:
    """K plans of length d*: the optimal plan + distractors that truly do not reach the goal, all distinct."""
    d = len(opt_plan)
    voids = [a for a in vocab if a[4] != "posted"]
    posted = [a for a in vocab if a[4] == "posted"]
    cands, kinds = [list(opt_plan)], ["opt"]

    def ok(c):
        return c not in cands and not goal_met(_rollout(init, c)[-1], goal)

    def try_add(kind, make, n):
        for _ in range(n):
            for _t in range(60):
                c = make()
                if c is not None and ok(c):
                    cands.append(c)
                    kinds.append(kind)
                    break

    def void_swap():
        p = rng.randrange(d)
        if opt_plan[p][4] != "posted" or not voids:
            return None
        twins = [v for v in voids if v[:4] == opt_plan[p][:4]]
        c = list(opt_plan)
        c[p] = rng.choice(twins or voids)
        return c

    def order_swap():
        if d < 2:
            return None
        p = rng.randrange(d - 1)
        if opt_plan[p] == opt_plan[p + 1]:
            return None
        c = list(opt_plan)
        c[p], c[p + 1] = c[p + 1], c[p]
        return c

    def substitute():
        p = rng.randrange(d)
        alt = [a for a in posted if a != opt_plan[p]]
        c = list(opt_plan)
        c[p] = rng.choice(alt)
        return c

    def rand():
        return [rng.choice(vocab) for _ in range(d)]

    try_add("void_swap", void_swap, 2)
    try_add("order_swap", order_swap, 1)
    try_add("substitute", substitute, 2)
    while len(cands) < K:
        before = len(cands)
        try_add("random", rand, 1)
        if len(cands) == before:
            raise RuntimeError("could not fill the fixed candidates")
    cands, kinds = cands[:K], kinds[:K]
    order = list(range(K))
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    kinds = [kinds[i] for i in order]
    finals = [_rollout(init, c)[-1] for c in cands]
    return {"K": K, "cands": cands, "kinds": kinds, "opt_index": kinds.index("opt"),
            "true_final": finals, "true_unmet": [unmet(f, goal) for f in finals]}


def draw_instance(spec: SetSpec, rng_range: str, d: int, slot: int, attempt: int) -> tuple:
    """(instance dict in canonical tuple form, None) or (None, rejection reason)."""
    rng = random.Random(seed_int(GEN_VERSION, spec.seed, rng_range, d, slot, attempt))
    M = marg()
    oor = rng_range == "oor"
    if oor:
        init = tuple((a, rng.choice((-1, 1)) * rng.randint(*OOR_BAL) if rng.random() < 0.4 else rng.choice(M["bal_pool"]))
                     for a in ACCOUNTS)
    else:
        init = tuple((a, rng.choice(M["bal_pool"])) for a in ACCOUNTS)
    used_ticks: set = set()
    sigs: set = set()
    c = rng.randint(2, min(5, d))
    core = [_draw_distinct(rng, used_ticks, sigs, 0.5 if oor else 0.0) for _ in range(c)]
    seed_plan = core + [rng.choice(core) for _ in range(d - c)]
    rng.shuffle(seed_plan)
    states = _rollout(init, seed_plan)
    end = states[-1]
    if end == init:
        return None, "seed plan returns to init"
    changed = [a for a in ACCOUNTS if dict(end)[a] != dict(init)[a]]
    if rng.random() < 0.5:
        goal_type, accs = "all", list(ACCOUNTS)
    else:
        goal_type = "subset"
        accs = sorted(rng.sample(changed, rng.randint(1, min(3, len(changed)))))
    goal = tuple((a, dict(end)[a]) for a in accs)
    n_void = rng.randint(1, 2)
    twin_src = []
    for a in seed_plan:
        if a not in twin_src:
            twin_src.append(a)
    twins = []
    for a in rng.sample(twin_src, min(n_void, len(twin_src))):
        t = rng.choice([x for x in range(1, 100) if x not in used_ticks])
        used_ticks.add(t)
        twins.append((a[0], a[1], a[2], a[3], "void", t))
    fill = []
    while c + len(twins) + len(fill) < spec.V:
        fill.append(_draw_distinct(rng, used_ticks, sigs, 0.5 if oor else 0.0))
    if not fill:
        return None, "no room for a random fill action"
    vocab = core + twins + fill
    roles = ["core"] * len(core) + ["void_twin"] * len(twins) + ["random"] * len(fill)
    order = list(range(len(vocab)))
    rng.shuffle(order)
    vocab = [vocab[i] for i in order]
    roles = [roles[i] for i in order]
    if len(set(vocab)) != len(vocab):
        return None, "duplicate vocab action"
    u0 = unmet(init, goal)
    effect = []
    for a in vocab:
        u1 = unmet(step_state(init, a), goal)
        effect.append("toward" if u1 < u0 else ("away" if u1 > u0 else "neutral"))
    if "away" not in effect:
        return None, "no vocab action moves away from the goal at init"
    comp = [compile_action(a) for a in vocab]
    gc = _goal_c(goal)
    full = bfs(vec(init), comp, gc, d)
    if full is None:
        raise AssertionError("the seed plan reaches the goal, so BFS must")
    if full["depth"] != d:
        return None, f"shortcut: d* {full['depth']} < seed length {d}"
    if not oor:
        if any(abs(v) > IN_RANGE_MAX_ABS for _, v in init) or any(abs(t) > IN_RANGE_MAX_ABS for _, t in goal):
            return None, "in: init or goal out of range"
        if any(a[3] not in M["amount_set"] for a in vocab):
            return None, "in: vocab amount out of range"
        bnd = bfs(vec(init), comp, gc, d, bound=IN_RANGE_MAX_ABS)
        if bnd is None or bnd["depth"] != d:
            return None, "in: no in-range optimal plan"
        plan_idx, n_opt_in = bnd["plan"], bnd["n_opt"]
    else:
        plan_idx, n_opt_in = full["plan"], None
    opt_plan = [vocab[i] for i in plan_idx]
    opt_states = _rollout(init, opt_plan)
    for s, a in zip(opt_states, opt_plan):          # vstep == env.step == the FDS oracle on every stored transition
        env.cross_check(LEDGER, state_to_dict(s), action_to_event(a))
    if [from_vec(v) for v in _vrollout(vec(init), [compile_action(a) for a in opt_plan])] != opt_states:
        raise AssertionError("vstep != env.step on the stored plan")
    if not goal_met(opt_states[-1], goal) or any(goal_met(s, goal) for s in opt_states[:-1]):
        raise AssertionError("stored plan does not reach the goal exactly at d*")
    oor_feat = None
    if oor:
        oor_feat = {"oor_amount_on_plan": any(a[4] == "posted" and a[3] not in M["amount_set"] for a in opt_plan),
                    "oor_state_on_plan": any(abs(v) > IN_RANGE_MAX_ABS for s in opt_states for _, v in s),
                    "oor_amount_in_vocab": any(a[3] not in M["amount_set"] for a in vocab)}
        if not (oor_feat["oor_amount_on_plan"] or oor_feat["oor_state_on_plan"]):
            return None, "oor: the optimal plan stays in range"
    else:
        if any(abs(v) > IN_RANGE_MAX_ABS for s in opt_states for _, v in s):
            raise AssertionError("in-range plan leaves the range")
    fixed = _fixed_candidates(rng, init, goal, vocab, opt_plan, spec.K)
    for cand in fixed["cands"]:
        for s, a in zip(_rollout(init, cand), cand):
            env.cross_check(LEDGER, state_to_dict(s), action_to_event(a))
    inst = {"id": f"lb-{spec.name}-{rng_range}-d{d}-{slot:03d}", "set": spec.name, "range": rng_range, "depth": d,
            "slot": slot, "attempt": attempt, "init": init, "goal": goal, "goal_type": goal_type, "vocab": vocab,
            "vocab_roles": roles, "vocab_effect_at_init": effect, "seed_plan": seed_plan, "opt_depth": d,
            "opt_plan": opt_plan, "opt_states": opt_states, "n_opt_plans": full["n_opt"],
            "n_opt_plans_in_range": n_opt_in, "n_goal_states_at_depth": full["n_goal_states"],
            "bfs_explored": full["explored"], "oor_features": oor_feat, "fixed": fixed,
            "init_unmet": u0}
    return inst, None


def _vrollout(v, comp):
    out = [v]
    for c in comp:
        out.append(vstep(out[-1], c))
    return out


# ------------------------------------------------------------------------------------------------ freshness
def binding_transitions(inst: dict) -> list:
    """(before dict, event dict) of: init x every vocab action, every stored optimal-plan step, every fixed-candidate
    step (deduplicated)."""
    seen, out = set(), []

    def add(s, a):
        if (s, a) not in seen:
            seen.add((s, a))
            out.append((state_to_dict(s), action_to_event(a)))
    for a in inst["vocab"]:
        add(inst["init"], a)
    for s, a in zip(inst["opt_states"], inst["opt_plan"]):
        add(s, a)
    for cand in inst["fixed"]["cands"]:
        for s, a in zip(_rollout(inst["init"], cand), cand):
            add(s, a)
    return out


def _L1():
    return _import_builders().L1                 # l1_make_ood (torch-free), imported with the environment restored


def _import_builders():
    """scripts/p3_make_bank and l1_make_ood set CUDA_VISIBLE_DEVICES="" and thread variables at import; restore them so
    importing the freshness tools never changes a GPU process' environment."""
    keep = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    try:
        import l1_make_ood as L1  # noqa: F401
        import p3_make_bank as MB
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return MB


COVERED_DATA = ("bank_pilot.json", "bank_confirm.json", "bank_confirm2.json", "bank_s4000.json", "t2_train.jsonl",
                "t2_eval.jsonl", "t2_eval_xframe.jsonl", "t2_eval_supp.jsonl", "bank_p3_oor.jsonl")
NON_PAYLOAD_DATA = ("offtask_prompts.jsonl",)


def load_exclusions(historical: bool = True, extra_data: bool = True, log=print) -> dict:
    """The exclusion union (payloads + FDS operative contents) and a per-source record. Never calls
    rcwmt.bank.load_exclusions(); the historical harvest is l1_make_ood.harvest_historical (name-skipping)."""
    MB = _import_builders()
    L1 = MB.L1
    lines: list = []
    t0 = time.time()
    src = MB.load_all_sources(lines, historical=historical)
    rec = OrderedDict()
    pay, ops = set(), set()
    for k, v in src.items():
        rec[k] = {"payloads": len(v["payloads"]), "operative": len(v["ops"]), **({"sha256": v["sha256"]} if v.get("sha256") else {}),
                  **({"skipped_by_name_never_opened": v["skipped_by_name_never_opened"]} if k == "historical" else {})}
        pay |= v["payloads"]
        ops |= v["ops"]
    del src
    # PREREG-3's sealed OOR bank (one-step rows and chain transitions)
    p3 = ROOT / "data" / "bank_p3_oor.jsonl"
    rows, chains = MB.C.load_bank(p3)
    p_pay = {L1.payload(b, e) for b, e in MB.transitions(rows, chains)}
    p_ops = {L1.opkey(LEDGER, b, e) for b, e in MB.transitions(rows, chains)}
    rec["bank_p3_oor.jsonl"] = {"payloads": len(p_pay), "operative": len(p_ops), "sha256": L1.sha256_file(p3)}
    pay |= p_pay
    ops |= p_ops
    del rows, chains, p_pay, p_ops
    if extra_data:
        for p in sorted(list((ROOT / "data").glob("*.jsonl")) + list((ROOT / "data").glob("*.json"))):
            if p.name in COVERED_DATA or p.name in NON_PAYLOAD_DATA or p.name.endswith(".SOURCE.json"):
                continue
            try:
                sha = L1.sha256_file(p)
                e_pay = MB.harvest_file(p)
            except (OSError, ValueError) as e:
                rec[f"data/{p.name}"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            e_ops = L1.op_keys_of_payloads(e_pay)
            rec[f"data/{p.name}"] = {"payloads": len(e_pay), "operative": len(e_ops), "sha256": sha}
            pay |= e_pay
            ops |= e_ops
    log(f"[lbeam] exclusion union: {len(pay)} payloads, {len(ops)} operative contents from {len(rec)} sources "
        f"({time.time() - t0:.0f}s)")
    return {"pay": pay, "ops": ops, "sources": rec, "historical": historical, "build_log": lines,
            "rule": ("p3_make_bank.load_all_sources (l1_make_ood.harvest_historical skips HSKILL_CONFIRM_V2* by name, "
                     "never opened; rcwmt.bank.load_exclusions() NOT called) + data/bank_p3_oor.jsonl transitions + every "
                     "{before, event} payload of the other data/*.json[l] files")}


def fresh_problem(inst: dict, excl: dict | None) -> str | None:
    if excl is None:
        return None
    L1 = _L1()
    for b, e in binding_transitions(inst):
        if L1.payload(b, e) in excl["pay"]:
            return "fresh: payload in an exclusion source"
        if L1.opkey(LEDGER, b, e) in excl["ops"]:
            return "fresh: operative content in an exclusion source"
    return None


def overlap_counts(insts: list, excl: dict) -> dict:
    L1 = _L1()
    n = p = o = 0
    for inst in insts:
        for b, e in binding_transitions(inst):
            n += 1
            p += L1.payload(b, e) in excl["pay"]
            o += L1.opkey(LEDGER, b, e) in excl["ops"]
    return {"binding_transitions": n, "payload_overlap": p, "operative_overlap": o}


def ball_overlap(insts: list, excl: dict, depth: int = 2) -> dict:
    """Descriptive: every transition reachable from init within `depth` steps of the vocab (true dynamics)."""
    L1 = _L1()
    n = p = o = 0
    for inst in insts:
        frontier, seen = [inst["init"]], {inst["init"]}
        for _ in range(depth):
            nxt = []
            for s in frontier:
                for a in inst["vocab"]:
                    b, e = state_to_dict(s), action_to_event(a)
                    n += 1
                    p += L1.payload(b, e) in excl["pay"]
                    o += L1.opkey(LEDGER, b, e) in excl["ops"]
                    ns = step_state(s, a)
                    if ns not in seen:
                        seen.add(ns)
                        nxt.append(ns)
            frontier = nxt
    return {"depth": depth, "transitions": n, "payload_overlap": p, "operative_overlap": o}


# ------------------------------------------------------------------------------------------------ set generation
def make_set(spec: SetSpec, excl: dict | None = None, log=None) -> tuple:
    """(instances, record). Deterministic given (spec, exclusion union): the RNG of every draw is keyed on
    (GEN_VERSION, seed, range, depth, slot, attempt); strata are filled in a fixed order."""
    t0 = time.time()
    insts, rejected, used_actions = [], {}, set()
    attempts_hist = Counter()
    fresh_rejections = []
    for rr in RANGES:
        n = spec.n_in_per_depth if rr == "in" else spec.n_oor_per_depth
        for d in spec.depths:
            rej = rejected.setdefault(f"{rr}-d{d}", Counter())
            for slot in range(n):
                for attempt in range(MAX_ATTEMPTS):
                    inst, why = draw_instance(spec, rr, d, slot, attempt)
                    if inst is not None and any(a in used_actions for a in inst["vocab"]):
                        inst, why = None, "an action tuple is used by another instance"
                    if inst is not None:
                        why = fresh_problem(inst, excl)
                        if why:
                            fresh_rejections.append({"stratum": f"{rr}-d{d}", "slot": slot, "attempt": attempt, "why": why})
                            inst = None
                    if inst is None:
                        rej[why] += 1
                        continue
                    used_actions.update(inst["vocab"])
                    insts.append(inst)
                    attempts_hist[attempt + 1] += 1
                    break
                else:
                    raise SystemExit(f"{rr}-d{d} slot {slot}: no acceptable draw in {MAX_ATTEMPTS} attempts")
            if log:
                log(f"[lbeam] {rr} d{d}: {n} instances; rejected {dict(rej)} ({time.time() - t0:.1f}s)")
    text = dumps_set(insts)
    rec = {"record": "LBEAM_PROBLEM_SET", "generator": GEN_VERSION, "spec": asdict(spec), "n": len(insts),
           "file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
           "per_stratum": dict(Counter(f"{i['range']}-d{i['depth']}" for i in insts)),
           "rejected": {k: dict(v) for k, v in rejected.items()}, "fresh_rejections": fresh_rejections,
           "draws_per_accepted_instance": dict(sorted(attempts_hist.items())),
           "freshness": "not checked" if excl is None else {"historical": excl["historical"], "sources": excl["sources"],
                                                           "rule": excl["rule"]},
           "model_calls": 0, "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1)}
    return insts, rec


# ------------------------------------------------------------------------------------------------ serialisation
def _jsonable(inst: dict) -> dict:
    return json.loads(json.dumps(inst))           # tuples -> lists


def dumps_set(insts: list) -> str:
    return "".join(json.dumps(_jsonable(i), sort_keys=True, separators=(",", ":")) + "\n" for i in insts)


def set_sha(insts: list) -> str:
    return hashlib.sha256(dumps_set(insts).encode("utf-8")).hexdigest()


def _tt(x):
    return tuple(_tt(y) for y in x) if isinstance(x, list) else x


def from_json(r: dict) -> dict:
    """A stored instance with canonical tuples restored (states, goal, actions, plans)."""
    i = dict(r)
    for k in ("init", "goal", "vocab", "seed_plan", "opt_plan", "opt_states"):
        i[k] = _tt(r[k])
    if isinstance(i["vocab"], tuple):
        i["vocab"] = list(i["vocab"])
    for k in ("seed_plan", "opt_plan", "opt_states"):
        i[k] = list(i[k])
    f = dict(r["fixed"])
    f["cands"] = [list(_tt(c)) for c in r["fixed"]["cands"]]
    f["true_final"] = [_tt(s) for s in r["fixed"]["true_final"]]
    i["fixed"] = f
    return i


def load_set(path: Path) -> list:
    return [from_json(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def planner_instances(insts: list) -> list:
    """The dicts BWM's plan_beam takes: id, init (hashable state), goal (hashable), objs (the vocab tuple, forwarded by
    plan_beam to the proposer's items), plus the full record under 'rec'."""
    return [{"id": i["id"], "init": i["init"], "goal": i["goal"], "objs": tuple(i["vocab"]), "rec": i} for i in insts]


def owner_of_action(insts: list) -> dict:
    """action tuple -> instance id (every action tuple belongs to exactly one instance of a set)."""
    own = {}
    for i in insts:
        for a in i["vocab"]:
            if a in own and own[a] != i["id"]:
                raise ValueError(f"action {a} shared by {own[a]} and {i['id']}")
            own[a] = i["id"]
    return own


# ------------------------------------------------------------------------------------------------ scoring
def execute(inst: dict, plan) -> dict:
    """Exact execution from init under env.step. valid = every action is in the instance's vocabulary (the ledger has
    no inapplicable events)."""
    voc = set(inst["vocab"])
    s = inst["init"]
    for t, a in enumerate(plan or []):
        if a not in voc:
            return {"valid": False, "fail_step": t, "final": s, "success": False, "unmet": unmet(s, inst["goal"])}
        s = step_state(s, a)
    return {"valid": True, "fail_step": None, "final": s, "success": goal_met(s, inst["goal"]), "unmet": unmet(s, inst["goal"])}


def _q(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def score(insts: list, eps: list, calls_by_id: dict | None = None) -> tuple:
    """BWM-style summary + per-instance rows. nregret = true unmet of the submitted plan's final state / mean true
    unmet of the K fixed candidates (0 = optimal; ~1 = a random fixed candidate)."""
    by = {i["id"]: i for i in insts}
    rows = []
    for e in eps:
        i = by[e["id"]]
        ex = execute(i, e.get("plan") or [])
        den = statistics.fmean(i["fixed"]["true_unmet"])
        rows.append({"id": i["id"], "range": i["range"], "depth": i["opt_depth"], "success": ex["success"],
                     "valid": ex["valid"], "verified": bool(e.get("verified")), "len": len(e.get("plan") or []),
                     "optimal_len": bool(ex["success"] and len(e.get("plan") or []) == i["opt_depth"]),
                     "unmet": ex["unmet"], "nregret": ex["unmet"] / den if den else None, "fail": e.get("fail"),
                     "expansions": e.get("expansions", e.get("rounds")), "gen_tokens": e.get("gen_tokens", 0),
                     "f_calls": None if calls_by_id is None else calls_by_id.get(i["id"], 0)})
    n = max(1, len(rows))
    ver = [r for r in rows if r["verified"]]
    summ = {"n": len(rows), "success": sum(r["success"] for r in rows) / n,
            "verified_rate": len(ver) / n,
            "verification_precision": (sum(r["success"] for r in ver) / len(ver)) if ver else None,
            "optimal_length_rate": sum(r["optimal_len"] for r in rows) / n,
            "mean_nregret": statistics.fmean(r["nregret"] for r in rows if r["nregret"] is not None),
            "proposer_tokens_per_instance": sum(r["gen_tokens"] for r in rows) / n}
    for key in ("range", "depth"):
        g = {}
        for r in rows:
            g.setdefault(r[key], []).append(r["success"])
        summ[f"success_by_{key}"] = {str(k): round(sum(v) / len(v), 4) for k, v in sorted(g.items())}
    if calls_by_id is not None:
        c = [r["f_calls"] for r in rows]
        summ["f_calls_per_instance"] = {"mean": round(statistics.fmean(c), 2), "median": _q(c, 0.5), "p90": _q(c, 0.9),
                                        "max": max(c), "total": sum(c)}
    return summ, rows


# ------------------------------------------------------------------------------------------------ verification
def verify_instances(insts: list) -> dict:
    """Re-derive every invariant from the stored instances (no generator state)."""
    M = marg()
    bad = Counter()
    seen_actions = {}
    for i in insts:
        voc = i["vocab"]
        if not all(is_action(a) for a in voc) or len(set(voc)) != len(voc) or len({a[5] for a in voc}) != len(voc):
            bad["vocab form / ticks"] += 1
        posted_sigs = [signature(a) for a in voc if a[4] == "posted"]
        if len(set(posted_sigs)) != len(posted_sigs) or not all(_effective(a[0], a[3]) for a in voc if a[4] == "posted"):
            bad["posted actions: effective, distinct arithmetic"] += 1
        if any(a[4] != "posted" and signature(a) not in posted_sigs for a in voc):
            bad["void twin without a posted source"] += 1
        for a in voc:
            if a in seen_actions and seen_actions[a] != i["id"]:
                bad["action shared across instances"] += 1
            seen_actions[a] = i["id"]
        if not is_state(i["init"]) or goal_met(i["init"], i["goal"]):
            bad["init"] += 1
        comp = [compile_action(a) for a in voc]
        r = bfs(vec(i["init"]), comp, _goal_c(i["goal"]), D_MAX)
        if r is None or r["depth"] != i["opt_depth"] or r["n_opt"] != i["n_opt_plans"]:
            bad["BFS depth / count"] += 1
        st = _rollout(i["init"], i["opt_plan"])
        if st != i["opt_states"] or not goal_met(st[-1], i["goal"]) or len(i["opt_plan"]) != i["opt_depth"]:
            bad["stored optimal plan"] += 1
        if not D_MIN <= i["opt_depth"] <= D_MAX:
            bad["depth range"] += 1
        for s, a in zip(st, i["opt_plan"]):
            try:
                env.cross_check(LEDGER, state_to_dict(s), action_to_event(a))
            except AssertionError:
                bad["oracle"] += 1
        if i["range"] == "in":
            if any(abs(v) > IN_RANGE_MAX_ABS for s in st for _, v in s) or any(a[3] not in M["amount_set"] for a in voc) \
                    or any(a[4] not in ("posted", "void") for a in voc):
                bad["in-range definition"] += 1
        f = i["fixed"]
        if len(f["cands"]) != f["K"] or len({tuple(c) for c in f["cands"]}) != f["K"] or f["cands"][f["opt_index"]] != i["opt_plan"]:
            bad["fixed: K / distinct / opt"] += 1
        for j, c in enumerate(f["cands"]):
            fin = _rollout(i["init"], c)[-1]
            if len(c) != i["opt_depth"] or fin != f["true_final"][j] or (goal_met(fin, i["goal"]) != (j == f["opt_index"])):
                bad["fixed: candidate"] += 1
        if "away" not in i["vocab_effect_at_init"] or "void_twin" not in i["vocab_roles"]:
            bad["vocab roles"] += 1
    return {"n": len(insts), "problems": dict(bad), "ok": not bad}


def describe(insts: list) -> dict:
    out = {"n": len(insts), "per_stratum": dict(Counter(f"{i['range']}-d{i['depth']}" for i in insts)),
           "goal_type": dict(Counter(i["goal_type"] for i in insts)),
           "fixed_kinds": dict(Counter(k for i in insts for k in i["fixed"]["kinds"])),
           "vocab_roles": dict(Counter(r for i in insts for r in i["vocab_roles"])),
           "vocab_effect_at_init": dict(Counter(r for i in insts for r in i["vocab_effect_at_init"])),
           "vocab_kinds": dict(Counter(a[0] for i in insts for a in i["vocab"]))}
    for rr in RANGES:
        sub = [i for i in insts if i["range"] == rr]
        if not sub:
            continue
        out[f"{rr}_n_opt_plans_median_by_depth"] = {d: statistics.median([i["n_opt_plans"] for i in sub if i["depth"] == d])
                                                    for d in sorted({i["depth"] for i in sub})}
        out[f"{rr}_max_abs_balance_on_opt_plan"] = max(abs(v) for i in sub for s in i["opt_states"] for _, v in s)
        out[f"{rr}_fixed_mean_true_unmet"] = round(statistics.fmean(statistics.fmean(i["fixed"]["true_unmet"]) for i in sub), 2)
    return out


# ------------------------------------------------------------------------------------------------ CLI
def _nice():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-dev", action="store_true", help="write docs/ledger_beam/dev_set.jsonl + SOURCE (create-only)")
    g.add_argument("--verify", metavar="SET_JSONL")
    g.add_argument("--describe", metavar="SET_JSONL")
    ap.add_argument("--fresh", action="store_true", help="with --verify: re-check freshness (loads the exclusion union)")
    ap.add_argument("--no-historical", action="store_true", help="skip the records-tree harvest (rehearsal only)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _nice()
    if a.describe:
        print(json.dumps(describe(load_set(Path(a.describe))), indent=1))
        return 0
    if a.verify:
        p = Path(a.verify)
        insts = load_set(p)
        res = verify_instances(insts)
        res["file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        res["reserialised_sha256_equal"] = set_sha(insts) == res["file_sha256"]
        src = p.with_name(p.name.replace(".jsonl", ".SOURCE.json"))
        if src.exists():
            rec = json.loads(src.read_text(encoding="utf-8"))
            res["source_record_sha256_equal"] = rec.get("file_sha256") == res["file_sha256"]
            spec = SetSpec(**{**rec["spec"], "depths": tuple(rec["spec"]["depths"])})
            regen, _ = make_set(spec)
            res["regenerated_without_exclusions_sha256"] = set_sha(regen)
            res["regenerated_equal"] = res["regenerated_without_exclusions_sha256"] == res["file_sha256"]
            res["fresh_rejections_at_build"] = len(rec.get("fresh_rejections") or [])
        if a.fresh:
            excl = load_exclusions(historical=not a.no_historical)
            res["overlap"] = overlap_counts(insts, excl)
            res["ball_overlap_depth2_descriptive"] = ball_overlap(insts, excl, 2)
        res["torch_imported"] = "torch" in sys.modules
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] and res.get("reserialised_sha256_equal") and (not a.fresh or (
            res["overlap"]["payload_overlap"] == 0 and res["overlap"]["operative_overlap"] == 0)) else 1
    out, srcp = DOCS / "dev_set.jsonl", DOCS / "dev_set.SOURCE.json"
    if out.exists() or srcp.exists():
        raise SystemExit(f"refusing to overwrite {out} / {srcp} (create-only)")
    excl = load_exclusions(historical=not a.no_historical)
    insts, rec = make_set(SetSpec(), excl, log=print)
    ov = overlap_counts(insts, excl)
    if ov["payload_overlap"] or ov["operative_overlap"]:
        raise SystemExit(f"overlap after the build: {ov}")
    rec["overlap"] = ov
    rec["ball_overlap_depth2_descriptive"] = ball_overlap(insts, excl, 2)
    rec["describe"] = describe(insts)
    rec["verify"] = verify_instances(insts)
    rec["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rec["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["use"] = ("DEV set for the joint beam harness and budget tuning (BWM + RCWM). NOT a confirmation set: the "
                  "confirmation set is drawn at the joint freeze with a fresh seed and the same generator.")
    rec["torch_imported"] = "torch" in sys.modules
    if not rec["verify"]["ok"]:
        raise SystemExit(f"verification failed: {rec['verify']}")
    DOCS.mkdir(parents=True, exist_ok=True)
    text = dumps_set(insts)
    for path, payload in ((out, text), (srcp, json.dumps(rec, indent=1) + "\n")):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    print(json.dumps({"file": str(out), "file_sha256": rec["file_sha256"], "n": rec["n"], "per_stratum": rec["per_stratum"],
                      "overlap": ov, "fresh_rejections": len(rec["fresh_rejections"]),
                      "torch_imported": rec["torch_imported"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
