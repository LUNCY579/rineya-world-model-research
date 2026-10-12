"""Batched learned-only W1 RegisterWorld planning with delayed exact scoring.

No exact transition, solver or legality checker is imported by search. Each
model call conditions on its own predicted state, and tasks share batching only.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from .domains import canonical, parent_world_key

ACTIONS = ("add1", "sub1", "double", "halve", "finish")


@dataclass(frozen=True)
class RecoveryPlanningBudget:
    max_depth: int = 6
    max_model_calls: int = 30
    beam_width: int = 1
    max_processed_positions: int = 120000

    def __post_init__(self):
        if type(self.beam_width) is not int or self.beam_width != 1:
            raise ValueError("Recovery planning binds beam width 1")
        for key in ("max_depth", "max_model_calls", "max_processed_positions"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("Planning budget fields must be positive integers")
        if self.max_depth > 6 or self.max_model_calls > 30:
            raise ValueError("Recovery planning cannot exceed depth 6 or 30 queries")


def validate_tasks(tasks: list[dict], *, expected_split: str = "dev") -> None:
    if expected_split not in {"train", "dev"}:
        raise PermissionError("Learned recovery planning accepts explicit TRAIN/DEV splits only")
    seen_ids, seen_parents = set(), set()
    expected = {"id", "split", "parent_world_id", "domain", "rules", "s0", "goal", "provenance"}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != expected:
            raise ValueError("Planning task must contain explicit inputs/provenance only")
        if task["domain"] != "register" or task["split"] != expected_split:
            raise PermissionError(f"Recovery planning requires {expected_split} RegisterWorld tasks")
        if not isinstance(task["rules"], dict) or not isinstance(task["id"], str):
            raise ValueError("Explicit rule object and string task ID required")
        limit = task["rules"].get("limit")
        if type(limit) is not int or limit <= 0 or task["rules"].get("semantics") != "bwm-register-v2":
            raise ValueError("Invalid RegisterWorld rule specification")
        goal = task["goal"]
        if not isinstance(goal, dict) or set(goal) != {"kind", "state"} or goal["kind"] != "state":
            raise ValueError("Explicit desired state goal required")
        for state in (task["s0"], goal["state"]):
            if not isinstance(state, dict) or set(state) != {"x"} or type(state["x"]) is not int or abs(state["x"]) > limit:
                raise ValueError("Invalid initial/goal register state")
        if parent_world_key("register", task["s0"], task["rules"]) != task["parent_world_id"]:
            raise ValueError("Canonical planning parent identity mismatch")
        if task["id"] in seen_ids or task["parent_world_id"] in seen_parents:
            raise ValueError("Planning tasks require unique IDs and independent parent worlds")
        seen_ids.add(task["id"])
        seen_parents.add(task["parent_world_id"])
    if not tasks:
        raise ValueError(f"At least one {expected_split} planning task is required")


def _rank(node: dict) -> tuple:
    return (not node["predicted_success"], node["distance"], len(node["plan"]), canonical(node["plan"]))


def _valid_prediction(prediction: dict) -> bool:
    return (isinstance(prediction, dict) and not prediction.get("malformed")
            and isinstance(prediction.get("state"), dict) and set(prediction["state"]) == {"x"}
            and type(prediction["state"]["x"]) is int
            and all(type(prediction.get(key)) is bool for key in ("legal", "terminated", "failed")))


def _token_length(predictor, inputs: dict) -> int:
    if hasattr(predictor, "token_length"):
        length = predictor.token_length(inputs)
    else:
        from .recovery import legacy_prompt
        length = len(predictor.tokenizer(legacy_prompt(inputs), add_special_tokens=False).input_ids)
    if type(length) is not int or length <= 0:
        raise ValueError("A positive real prompt token count is required for preflight")
    return length


def batched_search(tasks: list[dict], predictor, *, budget: RecoveryPlanningBudget = RecoveryPlanningBudget(),
                   max_prompt_tokens: int = 512, generation_tokens: int = 32, expected_split: str = "dev") -> dict:
    """Batch frontier children across tasks; never inspect exact scoring metadata.

    Every native query consumes at most 512+32 padded processed positions. Thus
    the default 30-query cap bounds each task to 16320 positions before inference,
    below 120000. Lower custom budgets are enforced by conservative preflight.
    Token metrics require the LegacyPredictor raw traces, not inferred counts.
    """
    validate_tasks(tasks, expected_split=expected_split)
    if (type(max_prompt_tokens) is not int or not 1 <= max_prompt_tokens <= 512
            or type(generation_tokens) is not int or not 1 <= generation_tokens <= 32):
        raise ValueError("Planning readout binds prompt<=512 and generation<=32 tokens")
    sessions = []
    for task in tasks:
        distance = abs(task["s0"]["x"] - task["goal"]["state"]["x"])
        initial = {"plan": [], "state": copy.deepcopy(task["s0"]), "distance": distance,
                   "predicted_success": distance == 0}
        sessions.append({"frontier": None if distance == 0 else initial,
                         "finished": [initial] if distance == 0 else [],
                         "metrics": {"query_attempts": 0, "model_queries": 0, "input_tokens": 0,
                                     "output_tokens": 0, "processed_positions_upper_bound": 0,
                                     "unpadded_processed_positions_upper_bound": 0,
                                     "malformed_predictions": 0, "illegal_predictions": 0,
                                     "overlength_refusals": 0, "budget_refusals": 0}})
    provenance = []
    for depth in range(1, budget.max_depth + 1):
        requests = []
        for index, (task, session) in enumerate(zip(tasks, sessions)):
            node, metrics = session["frontier"], session["metrics"]
            if node is None:
                continue
            for action in ACTIONS:
                if metrics["query_attempts"] >= budget.max_model_calls:
                    break
                metrics["query_attempts"] += 1
                inputs = {"domain": "register", "rules": copy.deepcopy(task["rules"]),
                          "s0": copy.deepcopy(node["state"]), "actions": [action]}
                length = _token_length(predictor, inputs)
                if length > max_prompt_tokens:
                    metrics["overlength_refusals"] += 1
                    continue
                requests.append({"task_index": index, "task_id": task["id"], "depth": depth,
                                 "parent_plan": copy.deepcopy(node["plan"]), "action": action,
                                 "inputs": inputs, "prompt_tokens": length})
        # Use the broad batch padding envelope before accepting any request. Even
        # requests later refused contribute to this conservative envelope.
        batch_max_prompt = max((request["prompt_tokens"] for request in requests), default=0)
        reserve = batch_max_prompt + generation_tokens
        counts, accepted = {}, []
        for request in requests:
            index = request["task_index"]
            metrics = sessions[index]["metrics"]
            pending = counts.get(index, 0)
            if metrics["processed_positions_upper_bound"] + (pending + 1) * reserve > budget.max_processed_positions:
                metrics["budget_refusals"] += 1
                continue
            counts[index] = pending + 1
            accepted.append(request)
        active_indices = {request["task_index"] for request in accepted}
        for index, session in enumerate(sessions):
            # Budget/length refusal freezes the last nonempty predicted prefix.
            # An unexpanded initial state never becomes a fabricated empty plan.
            if index not in active_indices and session["frontier"] is not None and session["frontier"]["plan"]:
                session["finished"].append(session["frontier"])
            session["frontier"] = None
        if not accepted:
            break
        batch_inputs = [copy.deepcopy(request["inputs"]) for request in accepted]
        before = len(predictor.traces)
        outputs = predictor.predict_batch(batch_inputs)
        traces = predictor.traces[before:]
        if len(outputs) != len(accepted) or len(traces) != len(accepted):
            raise RuntimeError("Predictor batch/trace cardinality mismatch")
        generated_counts = [trace["generated"]["n_tokens"] for trace in traces]
        if any(type(count) is not int or count < 0 or count > generation_tokens for count in generated_counts):
            raise RuntimeError("Predictor exceeded the frozen generation cap")
        actual_max_prompt = max(request["prompt_tokens"] for request in accepted)
        padded_upper = actual_max_prompt + max(generated_counts, default=0)
        candidates: dict[int, list] = {}
        for request, output, trace, generated in zip(accepted, outputs, traces, generated_counts):
            index, metrics = request["task_index"], sessions[request["task_index"]]["metrics"]
            metrics["model_queries"] += 1
            metrics["input_tokens"] += request["prompt_tokens"]
            metrics["output_tokens"] += generated
            metrics["processed_positions_upper_bound"] += padded_upper
            metrics["unpadded_processed_positions_upper_bound"] += request["prompt_tokens"] + generated
            if trace.get("conditioning") != request["inputs"]:
                raise RuntimeError("Predictor trace/input provenance mismatch")
            provenance.append({"task_id": request["task_id"], "parent_world_id": tasks[index]["parent_world_id"],
                               "depth": depth, "batch_index": depth - 1, "batch_size": len(accepted),
                               "parent_plan": request["parent_plan"], "action": request["action"],
                               "prompt_tokens": request["prompt_tokens"], "output_tokens": generated,
                               "padded_processed_positions_upper_bound": padded_upper, "raw": copy.deepcopy(trace)})
            if not _valid_prediction(output):
                metrics["malformed_predictions"] += 1
                continue
            if not output["legal"] or output["failed"]:
                metrics["illegal_predictions"] += 1
                continue
            distance = abs(output["state"]["x"] - tasks[index]["goal"]["state"]["x"])
            node = {"plan": request["parent_plan"] + [request["action"]], "state": copy.deepcopy(output["state"]),
                    "distance": distance, "predicted_success": distance == 0}
            if output["terminated"] or node["predicted_success"]:
                sessions[index]["finished"].append(node)
            else:
                candidates.setdefault(index, []).append(node)
        for index, nodes in candidates.items():
            sessions[index]["frontier"] = min(nodes, key=_rank)
    choices = []
    for task, session in zip(tasks, sessions):
        nodes = [*session["finished"]]
        if session["frontier"] is not None:
            nodes.append(session["frontier"])
        chosen = min(nodes, key=_rank) if nodes else None
        choices.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"],
                        "choice": copy.deepcopy(chosen["plan"]) if chosen is not None else None,
                        "predicted_state": copy.deepcopy(chosen["state"]) if chosen is not None else None,
                        "predicted_success": chosen["predicted_success"] if chosen is not None else False,
                        **session["metrics"]})
    return {"choices": choices, "budget": asdict(budget), "oracle_feedback": False,
            "all_choices_fixed_before_scoring": True, "traces": provenance,
            "preflight_proof": {"max_prompt_tokens": max_prompt_tokens, "generation_tokens": generation_tokens,
                                "max_task_positions": budget.max_model_calls * (max_prompt_tokens + generation_tokens),
                                "padding_policy": "every row charged the broad batch prompt/generation maxima"}}


def score_choices(tasks: list[dict], search_result: dict, scoring_metadata: list[dict], *, verifier=None) -> list[dict]:
    """Independent post-hoc scoring; all learned searches must finish first."""
    validate_tasks(tasks)
    if search_result.get("oracle_feedback") is not False or search_result.get("all_choices_fixed_before_scoring") is not True:
        raise ValueError("Completed learned-only search required before scoring")
    choices = copy.deepcopy(search_result["choices"])
    if len(choices) != len(tasks):
        raise ValueError("Search/task cardinality mismatch")
    metadata = {row["task_id"]: row for row in scoring_metadata if row["split"] == "dev"}
    if set(metadata) != {task["id"] for task in tasks}:
        raise ValueError("Development scoring metadata/task identities mismatch")
    if verifier is None:
        from .planning import verify_plan
        verifier = verify_plan
    # These collection/verifier interfaces are imported only after every task's
    # learned search has ended. Distances never influence candidate selection.
    from .domains import _semantics, exact_transition
    executor = _semantics("register")
    scored = []
    for task, choice in zip(tasks, choices):
        label = metadata[task["id"]]
        if choice["task_id"] != task["id"] or choice["parent_world_id"] != task["parent_world_id"] or label["parent_world_id"] != task["parent_world_id"]:
            raise ValueError("Parent identity mismatch at scoring boundary")
        optimum = label["optimal_moves"]
        if type(optimum) is not int or not 3 <= optimum <= 5:
            raise ValueError("Frozen recovery development distance must be 3..5")
        verified_optimum = executor.opt_moves(task["s0"]["x"], task["goal"]["state"]["x"], task["rules"]["limit"])
        if verified_optimum != optimum:
            raise ValueError("Independent post-choice shortest distance disagrees with frozen scoring metadata")
        inputs = {"domain": "register", "rules": copy.deepcopy(task["rules"]), "s0": copy.deepcopy(task["s0"]), "actions": []}
        plan = choice["choice"]
        verification = verifier(inputs, copy.deepcopy(task["goal"]), copy.deepcopy(plan)) if plan is not None else None
        success = verification is not None and verification["success"]
        cost = sum(action != "finish" for action in plan) if success else None
        first_action, first_legal, remaining_distance = None, None, None
        if plan:
            first_action = plan[0]
            after, first_legal, _, invalid = exact_transition("register", task["s0"], first_action, task["rules"])
            if first_legal and not invalid:
                remaining_distance = executor.opt_moves(after["x"], task["goal"]["state"]["x"], task["rules"]["limit"])
        first_optimal = bool(plan and first_legal and remaining_distance == verified_optimum - 1)
        scored.append({**choice, "cohort": label["cohort"], "optimal_moves": optimum,
                       "success": bool(success), "all_legal": bool(verification and verification["all_legal"]),
                       "actual_move_cost": cost, "optimal_success": bool(success and cost == optimum),
                       "independently_verified_optimal_moves": verified_optimum,
                       "first_action": first_action, "first_action_present": bool(plan),
                       "first_action_legal": first_legal, "first_action_remaining_distance": remaining_distance,
                       "first_action_optimal": first_optimal,
                       "first_action_scoring_policy": "post-choice exact legality and shortest-distance decrease; missing proposals count false",
                       "normalized_excess_cost": (cost - optimum) / optimum if success else None,
                       "regret_failure_policy": "unsolved cases have null cost/regret and remain failures",
                       "verification": verification})
    return scored
