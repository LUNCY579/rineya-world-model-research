"""Bounded learned planning, followed by a separate exact training verifier.

Search/ranking never executes exact transitions. The exact executor is imported
only by verify_plan; test fixtures may inject predictors but there is no default
predictor or oracle fallback. Goals are separate from dynamics conditioning.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

Predictor = Callable[[dict], dict]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PlanningBudget:
    max_model_calls: int = 64
    max_depth: int = 4
    beam_width: int = 4
    max_candidates: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def validate_state(domain: str, state: Any, rules: dict) -> bool:
    """Strict representation validation, without transition simulation/repair."""
    if not isinstance(state, dict):
        return False
    if domain == "register":
        return set(state) == {"x"} and type(state["x"]) is int
    if domain == "ledger":
        return (set(state) == set(rules["objects"])
                and all(type(value) is int for value in state.values()))
    if domain != "blocks" or set(state) != {"clear", "hand", "on", "table"}:
        return False
    objects = set(rules["objects"])
    if not isinstance(state["hand"], str) or state["hand"] not in objects | {"empty"}:
        return False
    for key in ("clear", "table"):
        items = state[key]
        if (not isinstance(items, list) or not all(isinstance(x, str) and x in objects for x in items)
                or len(set(items)) != len(items)):
            return False
    if not isinstance(state["on"], list):
        return False
    pairs = []
    for pair in state["on"]:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(x, str) or x not in objects for x in pair) or pair[0] == pair[1]):
            return False
        pairs.append(tuple(pair))
    if len(set(pairs)) != len(pairs):
        return False
    tops, supports = [x for x, _ in pairs], [y for _, y in pairs]
    held = [] if state["hand"] == "empty" else [state["hand"]]
    positions = tops + state["table"] + held
    if (len(set(tops)) != len(tops) or len(set(supports)) != len(supports)
            or len(positions) != len(objects) or set(positions) != objects
            or set(state["clear"]) != objects - set(supports) - set(held)
            or set(held) & set(supports)):
        return False
    below = dict(pairs)
    for obj in objects:
        seen = set()
        while obj in below:
            if obj in seen:
                return False
            seen.add(obj)
            obj = below[obj]
    return True


def _facts(state: dict) -> set[tuple]:
    facts = {("clear", x) for x in state["clear"]} | {("ontable", x) for x in state["table"]}
    facts |= {("on", x, y) for x, y in state["on"]}
    facts.add(("handempty",) if state["hand"] == "empty" else ("holding", state["hand"]))
    return facts


def validate_problem(conditioning: dict, goal: dict) -> None:
    if set(conditioning) != {"domain", "rules", "s0", "actions"} or conditioning["actions"]:
        raise ValueError("planning requires initial conditioning with empty actions")
    domain, rules = conditioning["domain"], conditioning["rules"]
    if domain not in {"register", "ledger", "blocks"} or not isinstance(rules, dict):
        raise ValueError("unknown domain/rules")
    if domain == "register":
        if type(rules.get("limit")) is not int or rules["limit"] <= 0:
            raise ValueError("invalid limit")
    else:
        objects = rules.get("objects")
        if (not isinstance(objects, list) or not objects or not all(isinstance(x, str) for x in objects)
                or len(set(objects)) != len(objects) or "empty" in objects):
            raise ValueError("invalid objects")
    if not validate_state(domain, conditioning["s0"], rules):
        raise ValueError("invalid initial state")
    if not isinstance(goal, dict):
        raise ValueError("goal object required")
    if set(goal) == {"kind", "state"} and goal["kind"] == "state":
        if not validate_state(domain, goal["state"], rules):
            raise ValueError("invalid target state")
    elif domain == "blocks" and set(goal) == {"kind", "facts"} and goal["kind"] == "facts":
        arities = {"clear": 2, "ontable": 2, "on": 3, "holding": 2, "handempty": 1}
        if not isinstance(goal["facts"], list) or not goal["facts"]:
            raise ValueError("nonempty fact goal required")
        for fact in goal["facts"]:
            if (not isinstance(fact, list) or not fact or not isinstance(fact[0], str)
                    or len(fact) != arities.get(fact[0], -1)
                    or any(not isinstance(x, str) or x not in rules["objects"] for x in fact[1:])):
                raise ValueError("invalid goal fact")
    else:
        raise ValueError("unsupported goal semantics")


def goal_distance(domain: str, state: dict, goal: dict) -> float:
    if goal["kind"] == "facts":
        return float(len({tuple(f) for f in goal["facts"]} - _facts(state)))
    target = goal["state"]
    if domain == "blocks":
        return float(len(_facts(state) ^ _facts(target)))
    return float(sum(abs(state[key] - target[key]) for key in target))


def goal_reached(domain: str, state: dict, goal: dict) -> bool:
    return goal_distance(domain, state, goal) == 0


def action_library(conditioning: dict, *, ledger_amounts=(-2, -1, 1, 2)) -> list:
    """Static syntax library: does not test legality or execute an action."""
    domain, rules = conditioning["domain"], conditioning["rules"]
    if domain == "register":
        return ["add1", "sub1", "double", "halve", "finish"]
    objects = rules["objects"]
    if domain == "blocks":
        return ([f"{op} {x}" for op in ("pick-up", "put-down") for x in objects]
                + [f"{op} {x} {y}" for op in ("stack", "unstack")
                   for x in objects for y in objects if x != y])
    if domain == "ledger":
        if not ledger_amounts or any(type(x) is not int for x in ledger_amounts):
            raise ValueError("integer ledger amount library required")
        return [{"kind": kind, "account": account, "other": other, "amount": amount, "status": "posted"}
                for kind in ("add", "scale", "transfer") for account in objects
                for other in (objects if kind == "transfer" else [account]) for amount in ledger_amounts]
    raise ValueError("unknown domain")


def _step_prediction(predictor: Predictor, conditioning: dict, state: dict, action: Any) -> dict | None:
    value = predictor({"domain": conditioning["domain"], "rules": copy.deepcopy(conditioning["rules"]),
                       "s0": copy.deepcopy(state), "actions": [copy.deepcopy(action)]})
    if (not isinstance(value, dict) or value.get("malformed")
            or any(type(value.get(key)) is not bool for key in ("legal", "terminated", "failed"))
            or not validate_state(conditioning["domain"], value.get("state"), conditioning["rules"])):
        return None
    return value


def _rank_key(row: dict):
    return (not row["predicted_success"], row["distance"], len(row["plan"]), _json(row["plan"]))


def rank_candidates(conditioning: dict, goal: dict, candidates: list[list], predictor: Predictor,
                    budget: PlanningBudget = PlanningBudget()) -> dict:
    """Rank complete given plans using learned rollouts; never exact feedback."""
    validate_problem(conditioning, goal)
    ranked, calls, malformed, skipped = [], 0, 0, 0
    for plan in candidates[:budget.max_candidates]:
        if not isinstance(plan, list) or len(plan) > budget.max_depth:
            skipped += 1
            continue
        state, valid, complete, terminated = copy.deepcopy(conditioning["s0"]), True, True, False
        for action in plan:
            if calls >= budget.max_model_calls:
                complete = False
                break
            if terminated:
                valid = False  # Plans after a predicted finish are not complete legal trajectories.
                break
            calls += 1
            value = _step_prediction(predictor, conditioning, state, action)
            if value is None:
                malformed += 1
                valid = False
                break
            state, terminated = value["state"], value["terminated"]
            if not value["legal"] or value["failed"]:
                valid = False
                break
        if valid and complete:
            distance = goal_distance(conditioning["domain"], state, goal)
            ranked.append({"plan": copy.deepcopy(plan), "predicted_state": state,
                           "distance": distance, "predicted_success": distance == 0})
        else:
            skipped += 1
    ranked.sort(key=_rank_key)
    return {"choice": ranked[0]["plan"] if ranked else None, "ranked": ranked,
            "model_calls": calls, "malformed_predictions": malformed, "skipped_candidates": skipped,
            "budget": asdict(budget), "oracle_feedback": False}


def learned_beam_search(conditioning: dict, goal: dict, predictor: Predictor,
                        budget: PlanningBudget = PlanningBudget(), actions: list | None = None) -> dict:
    validate_problem(conditioning, goal)
    library = action_library(conditioning) if actions is None else copy.deepcopy(actions)
    if not library:
        raise ValueError("nonempty action library required")
    frontier = [{"plan": [], "predicted_state": copy.deepcopy(conditioning["s0"]),
                 "distance": goal_distance(conditioning["domain"], conditioning["s0"], goal),
                 "predicted_success": goal_reached(conditioning["domain"], conditioning["s0"], goal)}]
    finished, calls, malformed = [], 0, 0
    if frontier[0]["predicted_success"]:
        finished.extend(frontier)
        frontier = []
    for _ in range(budget.max_depth):
        next_frontier = []
        for node in frontier:
            for action in library:
                if calls >= budget.max_model_calls:
                    break
                calls += 1
                value = _step_prediction(predictor, conditioning, node["predicted_state"], action)
                if value is None:
                    malformed += 1
                    continue
                if not value["legal"] or value["failed"]:
                    continue
                distance = goal_distance(conditioning["domain"], value["state"], goal)
                child = {"plan": node["plan"] + [copy.deepcopy(action)], "predicted_state": value["state"],
                         "distance": distance, "predicted_success": distance == 0}
                if child["predicted_success"] or value["terminated"]:
                    finished.append(child)
                else:
                    next_frontier.append(child)
            if calls >= budget.max_model_calls:
                break
        next_frontier.sort(key=_rank_key)
        frontier = next_frontier[:budget.beam_width]
        if not frontier or calls >= budget.max_model_calls:
            break
    ranked = sorted(finished + frontier, key=_rank_key)[:budget.max_candidates]
    return {"choice": ranked[0]["plan"] if ranked else None, "ranked": ranked,
            "model_calls": calls, "malformed_predictions": malformed,
            "budget": asdict(budget), "oracle_feedback": False}


def verify_plan(conditioning: dict, goal: dict, plan: list) -> dict:
    """Independent exact scoring, invoked only AFTER learned ranking/selection."""
    validate_problem(conditioning, goal)
    if not isinstance(plan, list):
        raise ValueError("plan list required")
    from .domains import rollout_labels, source_hashes
    labels = rollout_labels(conditioning["domain"], conditioning["s0"], plan, conditioning["rules"])
    final = labels["states"][-1] if plan else copy.deepcopy(conditioning["s0"])
    all_legal = all(labels["legal"]) and all(labels["executed"])
    success = all_legal and goal_reached(conditioning["domain"], final, goal)
    return {"verified": success, "success": success, "all_legal": all_legal,
            "final_state": final, "labels": labels, "verifier_sha256": source_hashes()}


def planning_prompt(conditioning: dict, goal: dict) -> str:
    validate_problem(conditioning, goal)
    return ("Return only a JSON array of ordered actions that legally achieves the goal.\n"
            + _json({"domain": conditioning["domain"], "rules": conditioning["rules"],
                     "s0": conditioning["s0"], "goal": goal}) + "\nPlan:")


def distill_ranked(training_row: dict, goal: dict, ranked_result: dict, *, keep: int = 1) -> dict:
    """Verify ranked training candidates after all learned model calls finish."""
    if training_row.get("split") != "train" or type(keep) is not int or keep <= 0:
        raise ValueError("only TRAIN records and positive retention count allowed")
    from .domains import sha256
    conditioning = {key: copy.deepcopy(training_row[key]) for key in ("domain", "rules", "s0")}
    conditioning["actions"] = []
    if ranked_result.get("oracle_feedback") is not False:
        raise ValueError("learned ranking without oracle feedback required")
    positive, negative = [], []
    for candidate in ranked_result["ranked"]:
        plan = candidate["plan"]
        verification = verify_plan(conditioning, goal, plan)
        common = {"split": "train", "domain": conditioning["domain"], "rules": conditioning["rules"],
                  "s0": conditioning["s0"], "goal": copy.deepcopy(goal), "plan": copy.deepcopy(plan),
                  "parent_world_id": training_row["parent_world_id"], "verified": verification["verified"],
                  "verification": verification, "provenance": {"training_input_sha256": sha256(training_row),
                    "candidate_sha256": sha256(candidate), "model_calls": ranked_result["model_calls"],
                    "budget": ranked_result["budget"], "selection": "learned-ranked-then-independent-exact"}}
        if verification["verified"] and len(positive) < keep:
            common.update(prompt=planning_prompt(conditioning, goal), target=_json(plan))
            positive.append(common)
        elif not verification["verified"]:
            negative.append(common)
    return {"verified": positive, "hard_negatives": negative}


def verify_distillation_row(row: dict) -> bool:
    """Fail-closed re-execution gate for native A3/A4 training input."""
    try:
        if row.get("split") != "train" or row.get("verified") is not True:
            return False
        conditioning = {key: row[key] for key in ("domain", "rules", "s0")}
        conditioning["actions"] = []
        if not isinstance(row.get("parent_world_id"), str) or not row["parent_world_id"]:
            return False
        if row["prompt"] != planning_prompt(conditioning, row["goal"]) or row["target"] != _json(row["plan"]):
            return False
        verification = verify_plan(conditioning, row["goal"], row["plan"])
        return (verification["verified"] and row["verification"] == verification
                and row["provenance"]["selection"] == "learned-ranked-then-independent-exact")
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
