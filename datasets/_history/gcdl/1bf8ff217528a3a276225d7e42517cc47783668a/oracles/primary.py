"""Authoritative deterministic simulation with exact rational aggregation."""
from copy import deepcopy
from fractions import Fraction
import json
from ..environments.core import get_environment

ORACLE_VERSION = "primary-v1"
UTILITY_DEFINITION = {'comparison': 'lexicographic',
    'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
    'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
    'all_infeasible': 'abstain; no best candidate'}


def _rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _matches(observation, world):
    if observation is None:
        return True
    if type(observation) is dict:
        return type(world) is dict and set(observation) == set(world) and all(_matches(observation[k], world[k]) for k in observation)
    if type(observation) is list:
        return type(world) is list and len(observation) == len(world) and all(_matches(a, b) for a, b in zip(observation, world))
    return type(observation) is type(world) and observation == world


def _unknown(value):
    return value is None or (type(value) is dict and any(_unknown(v) for v in value.values())) or (type(value) is list and any(_unknown(v) for v in value))


def _input(task):
    if type(task) is not dict or not {"environment", "initial_state", "rules", "goal", "constraints", "candidates"} <= set(task):
        raise ValueError("task requires environment/initial_state/rules/goal/constraints/candidates")
    if "utility_definition" in task and task["utility_definition"] != UTILITY_DEFINITION:
        raise ValueError("unsupported utility_definition")
    environment = get_environment(task["environment"])
    info = task.get("information", "fully_observed")
    if info not in ("fully_observed", "missing", "ambiguous"):
        raise ValueError("unsupported information kind")
    if info == "fully_observed":
        if "possible_worlds" in task or task.get("probability_assumption", "deterministic") != "deterministic":
            raise ValueError("fully observed task must have one deterministic initial state")
        worlds = [task["initial_state"]]
        assumption = "deterministic"
    else:
        worlds = task.get("possible_worlds")
        assumption = task.get("probability_assumption")
        if type(worlds) is not list or len(worlds) < 2 or assumption != "uniform_over_possible_worlds" or not _unknown(task["initial_state"]):
            raise ValueError("uncertainty requires unknown observation and >=2 explicit uniformly weighted worlds")
        if not all(_matches(task["initial_state"], world) for world in worlds):
            raise ValueError("possible world contradicts observation")
        if len({json.dumps(w, sort_keys=True, separators=(",", ":")) for w in worlds}) != len(worlds):
            raise ValueError("duplicate possible worlds")
    candidates = task["candidates"]
    if type(candidates) is not list or not candidates:
        raise ValueError("nonempty candidates required")
    ids = []
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {"id", "plan"} or type(candidate["id"]) is not str or not candidate["id"] or type(candidate["plan"]) is not list:
            raise ValueError("invalid candidate")
        ids.append(candidate["id"])
        for action in candidate["plan"]:
            environment.validate_action(action)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    for world in worlds:
        environment.validate(world, task["rules"], task["goal"], task["constraints"])
    # A hidden entity name/world topology must not masquerade as missing numeric data.
    if any(_structure(world) != _structure(worlds[0]) for world in worlds[1:]):
        raise ValueError("possible worlds must share state field/entity structure")
    return environment, info, assumption, worlds


def _structure(value):
    if type(value) is dict:
        return {k: _structure(v) for k, v in value.items()}
    if type(value) is list:
        return [_structure(v) for v in value]
    return "leaf"


def evaluate(task):
    env, information, assumption, worlds = _input(task)
    results = []
    for candidate in task["candidates"]:
        outcomes = []
        for world_index, initial in enumerate(worlds):
            current = deepcopy(initial)
            trajectory = [deepcopy(current)]
            steps, violations = [], []
            cost, failure = 0, None
            for label in env.violations(current, initial, task["constraints"], cost):
                violations.append({"constraint": label, "step": 0})
            for step, action in enumerate(candidate["plan"], 1):
                following, error = env.transition(current, action, task["rules"])
                step_cost = 0 if error else task["rules"]["action_costs"][action["op"]]
                steps.append({"step": step, "action": deepcopy(action), "legal": error is None, "state": deepcopy(following), "cost": step_cost, "error": error})
                if error:
                    failure = step
                    break
                current = following
                cost += step_cost
                trajectory.append(deepcopy(current))
                for label in env.violations(current, initial, task["constraints"], cost):
                    violations.append({"constraint": label, "step": step})
            score, goal_satisfied = env.score(current, task["goal"])
            legal = failure is None
            outcomes.append({"world_index": world_index, "legal": legal, "failure_step": failure, "steps": steps, "trajectory": trajectory, "terminal_state": deepcopy(current), "cost": cost, "goal_satisfied": goal_satisfied, "goal_score": score, "constraint_violations": violations, "utility": [int(legal and not violations), score, -cost]})
        count = len(outcomes)
        sums = [sum(o["utility"][i] for o in outcomes) for i in range(3)]
        utility = [_rational(Fraction(value, count)) for value in sums]
        results.append({"id": candidate["id"], "outcomes": outcomes, "legal_probability": _rational(Fraction(sum(o["legal"] for o in outcomes), count)), "feasible_probability": utility[0], "success_probability": _rational(Fraction(sum(o["legal"] and not o["constraint_violations"] and o["goal_satisfied"] for o in outcomes), count)), "expected_goal_score": utility[1], "expected_cost": _rational(Fraction(sum(o["cost"] for o in outcomes), count)), "utility": utility, "executed_steps": sum(len(o["steps"]) for o in outcomes)})
    return {"environment": task["environment"], "world_count": len(worlds), "information": information, "probability_assumption": assumption, "candidates": results}
