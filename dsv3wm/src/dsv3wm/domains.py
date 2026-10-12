"""Exact collection/verifier adapters. Learned inference must not call this module.

Only the audited transition functions are reused. Historical generators, marginal
files, frozen evaluation signatures and plan solvers are never invoked.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SOURCES = {
    "register": REPO / "bwm/code/bwmn/rw2.py",
    "blocks": REPO / "bwm/code/bwmn/blocks.py",
    "ledger": REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py",
}
DOMAINS = tuple(SOURCES)
REGISTER_ACTIONS = ("add1", "sub1", "double", "halve", "finish")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in SOURCES.values()}


@lru_cache(maxsize=None)
def _semantics(domain: str):
    # Loading these sources has no I/O side effects; exact_step / step / state_json
    # are the only executable interfaces used below. Register before exec for dataclasses.
    name = f"_dsv3wm_collection_{domain}"
    spec = importlib.util.spec_from_file_location(name, SOURCES[domain])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load collection semantics: {domain}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rules_for(domain: str, *, limit: int = 32, objects: list[str] | None = None) -> dict:
    common = "Malformed or unknown actions are illegal no-ops. Illegal actions leave the state unchanged."
    if domain == "register":
        return {"semantics": "bwm-register-v2", "limit": limit,
                "text": "add1 adds 1; sub1 subtracts 1; double multiplies x by 2; halve divides even x by 2. "
                "Odd halving or exceeding abs(x)<=limit is illegal. finish ends the episode; later actions "
                "are ignored (executed=false, legal=false), with an absorbing state. " + common}
    if domain == "ledger":
        return {"semantics": "rcwm-ledger-replay", "objects": list(objects or ["A", "B", "C", "D"]),
                "text": "Apply events in order. An event with status other than posted is a legal no-op. "
                "add adds amount to account; scale multiplies account by amount; transfer subtracts amount "
                "from account and adds it to other simultaneously, including when both are identical. "
                "Balances are unbounded integers. Events do not terminate the episode. " + common}
    if domain == "blocks":
        return {"semantics": "bwm-blocks-4ops", "objects": list(objects or []),
                "text": "pick-up x requires clear x, ontable x and handempty; removes those and adds holding x. "
                "put-down x requires holding x; removes holding x and adds clear x, ontable x, handempty. "
                "stack x y requires holding x, clear y and x!=y; removes holding x and clear y, adds on x y, "
                "clear x, handempty. unstack x y requires on x y, clear x, handempty; removes those, "
                "adds holding x and clear y. Actions do not terminate the episode. " + common}
    raise ValueError(f"Unknown domain: {domain}")


def block_facts(state: dict) -> frozenset[tuple[str, ...]]:
    facts = {("clear", x) for x in state["clear"]} | {("ontable", x) for x in state["table"]}
    facts |= {("on", x, y) for x, y in state["on"]}
    facts.add(("handempty",) if state["hand"] == "empty" else ("holding", state["hand"]))
    return frozenset(facts)


def exact_transition(domain: str, state: dict, action: Any, rules: dict) -> tuple[dict, bool, bool, bool]:
    """Collection only: return next_state, legal, terminated, invalid syntax.

    Validity wrappers extend the established executor's input contract explicitly.
    A non-posted ledger event is legal even though its status gates the update.
    """
    invalid = False
    if domain == "register":
        invalid = not isinstance(action, str) or action not in REGISTER_ACTIONS
        if not invalid:
            x, legal, done = _semantics(domain).exact_step(state["x"], action, rules["limit"])
            return {"x": x}, legal, done, False
    elif domain == "ledger":
        invalid = (not isinstance(action, dict) or set(action) - {"kind", "account", "other", "amount", "status"}
                   or action.get("kind") not in ("add", "scale", "transfer")
                   or action.get("account") not in rules["objects"]
                   or type(action.get("amount")) is not int or not isinstance(action.get("status"), str)
                   or (action.get("kind") == "transfer" and action.get("other") not in rules["objects"]))
        if not invalid:
            return _semantics(domain).step("ledger_replay", state, action), True, False, False
    elif domain == "blocks":
        parts = action.split() if isinstance(action, str) else []
        arities = {"pick-up": 2, "put-down": 2, "stack": 3, "unstack": 3}
        invalid = (not parts or len(parts) != arities.get(parts[0], -1)
                   or any(x not in rules["objects"] for x in parts[1:]))
        if not invalid:
            module = _semantics(domain)
            ns, legal = module.step(block_facts(state), tuple(parts))
            return module.state_json(ns), legal, False, False
    else:
        raise ValueError(domain)
    return copy.deepcopy(state), False, False, bool(invalid)


def rollout_labels(domain: str, s0: dict, actions: list, rules: dict) -> dict:
    labels = {key: [] for key in ("states", "legal", "terminated", "failed", "invalid", "executed")}
    state, done, failed = copy.deepcopy(s0), False, False
    for action in actions:
        executed = not done
        if executed:
            state, legal, done, invalid = exact_transition(domain, state, action, rules)
            failed = failed or not legal
        else:
            legal, invalid = False, False
        for key, value in (("states", copy.deepcopy(state)), ("legal", legal), ("terminated", done),
                           ("failed", failed), ("invalid", invalid), ("executed", executed)):
            labels[key].append(value)
    return labels


def delta_state(domain: str, before: dict, after: dict) -> dict:
    if domain == "blocks":
        old, new = block_facts(before), block_facts(after)
        return {"add": sorted(" ".join(f) for f in new - old), "del": sorted(" ".join(f) for f in old - new)}
    return {key: after[key] - before[key] for key in before}


def apply_delta(domain: str, before: dict, delta: dict) -> dict:
    """Pure representation utility: no exact dynamics are consulted."""
    if domain != "blocks":
        if set(delta) != set(before) or any(type(v) is not int for v in delta.values()):
            raise ValueError("Invalid numeric delta")
        return {key: before[key] + delta[key] for key in before}
    if set(delta) != {"add", "del"} or not all(isinstance(delta[k], list) for k in delta):
        raise ValueError("Invalid fact delta")
    old = block_facts(before)
    if any(not isinstance(f, str) for key in ("add", "del") for f in delta[key]):
        raise ValueError("Fact deltas must contain strings")
    add, delete = ({tuple(f.split()) for f in delta[key]} for key in ("add", "del"))
    objects = {x for fact in old for x in fact[1:]}
    arities = {"clear": 2, "ontable": 2, "handempty": 1, "holding": 2, "on": 3}
    if any(not fact or len(fact) != arities.get(fact[0], -1) or any(x not in objects for x in fact[1:])
           for fact in add | delete):
        raise ValueError("Unknown or malformed fact delta")
    if not delete <= old or add & delete:
        raise ValueError("Unsupported or contradictory fact delta")
    facts = (old - delete) | add
    holds = sorted(f[1] for f in facts if f[0] == "holding")
    if len(holds) > 1 or bool(holds) == (("handempty",) in facts):
        raise ValueError("Inconsistent hand facts")
    locations = {obj: [] for obj in objects}
    above: dict[str, str] = {}
    for fact in facts:
        if fact[0] in ("ontable", "holding", "on"):
            locations[fact[1]].append(fact)
        if fact[0] == "on":
            upper, lower = fact[1:]
            if upper == lower or lower in above:
                raise ValueError("Self-support or multiple blocks on one support")
            above[lower] = upper
    if any(len(location) != 1 for location in locations.values()):
        raise ValueError("Missing or contradictory block locations")
    clear = {f[1] for f in facts if f[0] == "clear"}
    if clear != objects - set(above) - set(holds) or any(obj in above for obj in holds):
        raise ValueError("Inconsistent clear/held block facts")
    for obj in objects:
        seen, current = set(), obj
        while current in above:
            if current in seen:
                raise ValueError("Cyclic block support")
            seen.add(current)
            current = above[current]
    return {"clear": sorted(f[1] for f in facts if f[0] == "clear"), "hand": holds[0] if holds else "empty",
            "on": sorted([f[1], f[2]] for f in facts if f[0] == "on"),
            "table": sorted(f[1] for f in facts if f[0] == "ontable")}


def _rename(domain: str, state: dict, actions: list, rules: dict, mapping: dict) -> tuple:
    nrules = copy.deepcopy(rules)
    nrules["objects"] = sorted(mapping.values())
    if domain == "ledger":
        nstate = {mapping[k]: v for k, v in state.items()}
        nactions = copy.deepcopy(actions)
        for action in nactions:
            if isinstance(action, dict):
                for key in ("account", "other"):
                    if action.get(key) in mapping:
                        action[key] = mapping[action[key]]
    else:
        nstate = {"clear": sorted(mapping[x] for x in state["clear"]),
                  "table": sorted(mapping[x] for x in state["table"]),
                  "on": sorted([mapping[x], mapping[y]] for x, y in state["on"]),
                  "hand": "empty" if state["hand"] == "empty" else mapping[state["hand"]]}
        nactions = [" ".join(mapping.get(token, token) for token in a.split()) if isinstance(a, str) else a
                    for a in actions]
    return nstate, nactions, nrules


def semantic_key(domain: str, s0: dict, actions: list, rules: dict) -> str:
    """Canonical input identity, invariant to keys/fact order and object renaming.

    Exact permutation minimization is bounded to five objects. Larger cases are
    rejected rather than silently advertised as having renaming-safe deduplication.
    """
    # Prose/key order are presentation choices, not distinct parent worlds.
    rules = {key: copy.deepcopy(rules[key]) for key in ("semantics", "limit", "objects") if key in rules}
    if domain == "register":
        return sha256({"domain": domain, "s0": s0, "actions": actions, "rules": rules})
    objects = sorted(rules["objects"])
    if len(objects) > 5:
        raise ValueError("Renaming-safe canonicalization supports <=5 objects")
    best = None
    for permutation in itertools.permutations(range(len(objects))):
        mapping = {name: f"o{permutation[i]}" for i, name in enumerate(objects)}
        state, acts, rs = _rename(domain, s0, actions, rules, mapping)
        representation = canonical({"domain": domain, "s0": state, "actions": acts, "rules": rs})
        best = representation if best is None or representation < best else best
    return hashlib.sha256((best or "").encode()).hexdigest()


def parent_world_key(domain: str, s0: dict, rules: dict) -> str:
    return semantic_key(domain, s0, [], rules)


def sample_world(domain: str, rng: random.Random, *, stress: str = "id") -> tuple[dict, dict]:
    if domain == "register":
        limit = rng.choice((64, 80, 96)) if stress == "unseen_range" else rng.choice((24, 32, 40))
        x = rng.choice((-1, 1)) * rng.randint(41, limit) if stress == "unseen_range" else rng.randint(-limit, limit)
        return {"x": x}, rules_for(domain, limit=limit)
    if domain == "ledger":
        names = ["u", "v", "w", "z"] if stress == "object_rename" else ["A", "B", "C", "D"]
        span = 300 if stress == "unseen_range" else 20
        return {name: rng.randint(-span, span) for name in names}, rules_for(domain, objects=names)
    names = ["gem", "reed", "stone", "ash", "fern"] if stress == "object_rename" else list("abcde")
    count = 5 if stress == "unseen_object_count" else rng.choice((3, 4))
    objects = names[:count]
    # Fresh unconstrained tower construction: no old generator/goal/signatures.
    ordered = list(objects)
    rng.shuffle(ordered)
    towers: list[list[str]] = []
    for obj in ordered:
        if towers and rng.random() < .6:
            rng.choice(towers).append(obj)
        else:
            towers.append([obj])
    state = {"clear": sorted(t[-1] for t in towers), "hand": "empty",
             "on": sorted([high, low] for t in towers for low, high in zip(t, t[1:])),
             "table": sorted(t[0] for t in towers)}
    # Reachable held states increase parent-world coverage.
    if rng.random() < .3:
        clear = rng.choice(state["clear"])
        on = next((pair for pair in state["on"] if pair[0] == clear), None)
        action = f"unstack {clear} {on[1]}" if on else f"pick-up {clear}"
        state, _, _, _ = exact_transition(domain, state, action, rules_for(domain, objects=objects))
    return state, rules_for(domain, objects=objects)


def sample_actions(domain: str, s0: dict, rules: dict, rng: random.Random, *, length: int = 4,
                   stress: str = "id") -> list:
    state, actions = copy.deepcopy(s0), []
    for index in range(length):
        if domain == "register":
            action = rng.choice(REGISTER_ACTIONS[:-1])
            if index == length - 1 and rng.random() < .2:
                action = "finish"
        elif domain == "ledger":
            objects = rules["objects"]
            action = {"kind": rng.choice(("add", "scale", "transfer")), "account": rng.choice(objects),
                      "other": rng.choice(objects), "amount": rng.randint(-5, 5),
                      "status": rng.choice(("posted", "posted", "pending", "cancelled"))}
        else:
            objects = rules["objects"]
            candidates = [f"{op} {x}" for op in ("pick-up", "put-down") for x in objects]
            candidates += [f"{op} {x} {y}" for op in ("stack", "unstack") for x in objects for y in objects if x != y]
            legal = [a for a in candidates if exact_transition(domain, state, a, rules)[1]]
            action = rng.choice(legal if legal and rng.random() < .7 else candidates)
        actions.append(action)
        state, _, _, _ = exact_transition(domain, state, action, rules)
    if stress == "invalid_action":
        actions[rng.randrange(length)] = ("teleport" if domain != "ledger" else
                                         {"kind": "add", "account": "missing", "amount": 1, "status": "posted"})
    return actions
