"""Generator-side exact computation for public finite proxy contracts.

Authored by /root/relational_critic_producer. Never imports the independent oracle.
"""
from itertools import product
import json
from pathlib import Path
import re


def scalar_key(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def scalar_equal(left, right):
    return type(left) is type(right) and left == right


def contracts():
    document = (Path(__file__).resolve().parents[2] / "schemas/EXTENDED_SEMANTICS.md").read_text(encoding="utf-8")
    parts = re.split(r"^## `kind=([^`]+)`\n", document, flags=re.MULTILINE)
    shared = ("JSON scalar values in this task are null, boolean, integer (excluding boolean), or string. "
              "No floating point or coercion is used. Scalar equality is type-sensitive. "
              "Malformed tasks are outside the valid-task domain.\n")
    return {parts[index]: shared + parts[index + 1].strip() + "\n" for index in range(1, len(parts), 2)}


def skills(task):
    state = dict(task["state"])
    calls, conditions = [], []
    primitive_count = 0

    def execute(instructions):
        nonlocal primitive_count
        for instruction in instructions:
            op = instruction["op"]
            if op == "call":
                calls.append(instruction["name"])
                execute(task["procedures"][instruction["name"]])
            elif op == "if":
                test = instruction["test"]
                a, b = state[test["register"]], test["value"]
                result = {"lt": lambda: a < b, "eq": lambda: a == b, "ge": lambda: a >= b}[test["cmp"]]()
                conditions.append(result)
                execute(instruction["then"] if result else instruction["else"])
            else:
                primitive_count += 1
                if op == "add":
                    state[instruction["register"]] += instruction["value"]
                elif op == "scale":
                    state[instruction["register"]] *= instruction["value"]
                elif op == "swap":
                    state["x"], state["y"] = state["y"], state["x"]
                else:
                    raise ValueError(op)
    execute(task["main"])
    return dict(after=state, executed_primitives=primitive_count, call_order=calls, conditions=conditions)


def modules(task):
    memory, routes = dict(task["memory"]), []
    protected = set(task["protected"])
    for request in task["requests"]:
        eligible = [module for module in task["modules"] if set(request["needs"]) <= set(module["capabilities"])]
        selected = min(eligible, key=lambda module: (module["priority"], module["module_id"])) if eligible else None
        blocked = []
        if selected is not None:
            for index, write in enumerate(request["writes"]):
                if write["key"] in protected:
                    blocked.append(index)
                else:
                    memory[write["key"]] = write["value"]
        routes.append(dict(request_id=request["request_id"], selected_module=selected["module_id"] if selected else None,
                           eligible_modules=sorted(module["module_id"] for module in eligible), blocked_writes=blocked))
    return dict(routes=routes, memory=memory)


def episodic(task):
    results = []
    for query in task["queries"]:
        eligible = [(event["version"], event["at"], index, event) for index, event in enumerate(task["events"])
                    if event["key"] == query["key"] and event["at"] <= query["as_of"]]
        chosen = max(eligible, key=lambda item: item[:3]) if eligible else None
        if chosen is None:
            status, value, version, event_index = "absent", None, None, None
        else:
            version, _, event_index, event = chosen
            status = "deleted" if event["op"] == "delete" else "present"
            value = None if status == "deleted" else event["value"]
        results.append(dict(query_id=query["query_id"], key=query["key"], status=status, value=value,
                            chosen_version=version, chosen_event=event_index))
    return dict(results=results)


def belief(task):
    survivors = list(task["worlds"])
    sizes, queries = [], []
    for observation in task["observations"]:
        survivors = [world for world in survivors if scalar_equal(world["facts"][observation["fact"]], observation["equals"])]
        sizes.append(len(survivors))
    for query in task["queries"]:
        counts = {}
        for world in survivors:
            value = world["facts"][query["fact"]]
            key = scalar_key(value)
            if key not in counts:
                counts[key] = dict(value=value, count=0)
            counts[key]["count"] += 1
        entries = [counts[key] for key in sorted(counts)]
        entailed = len(entries) == 1
        queries.append(dict(query_id=query["query_id"], counts=entries, distinct_values=len(entries), entailed=entailed,
                            guaranteed=entries[0]["value"] if entailed else None, can_reduce_uncertainty=len(entries) > 1))
    return dict(surviving_worlds=sorted(world["world_id"] for world in survivors), belief_sizes=sizes, queries=queries)


def predictive_abstraction(task):
    def partition(signatures):
        groups = {}
        for state in task["states"]:
            groups.setdefault(signatures[state], []).append(state)
        return sorted(sorted(group) for group in groups.values())
    groups = partition({state: scalar_key(task["observable"][state]) for state in task["states"]})
    actions = sorted(task["actions"])
    while True:
        ids = {state: index for index, group in enumerate(groups) for state in group}
        refined = partition({state: (scalar_key(task["observable"][state]),
                                     tuple(ids[task["transition"][state][action]] for action in actions)) for state in task["states"]})
        if groups == refined:
            break
        groups = refined
    ids = {state: index for index, group in enumerate(groups) for state in group}
    transitions = [dict(class_id=index, action=action, next_class=ids[task["transition"][group[0]][action]])
                   for index, group in enumerate(groups) for action in actions]
    observable = [dict(class_id=index, value=task["observable"][group[0]]) for index, group in enumerate(groups)]
    return dict(classes=groups, state_to_class=ids, abstract_transitions=transitions, abstract_observable=observable)


def constraint_repair(task):
    def satisfies(assignment):
        for constraint in task["constraints"]:
            kind = constraint["kind"]
            if kind in ("equal", "different"):
                i, j = constraint["positions"]
                equal = assignment[i] == assignment[j]
                if equal != (kind == "equal"):
                    return False
            elif kind == "allowed":
                if assignment[constraint["position"]] not in constraint["values"]:
                    return False
            elif kind == "count":
                if not constraint["min"] <= assignment.count(constraint["value"]) <= constraint["max"]:
                    return False
            else:
                raise ValueError(kind)
        return True
    feasible = [assignment for assignment in product(sorted(task["alphabet"]), repeat=task["slots"]) if satisfies(assignment)]
    feasible_set = set(feasible)
    repairs = []
    for candidate in task["candidates"]:
        original = candidate["assignment"]
        if feasible:
            distance, repaired = min((sum(a != b for a, b in zip(original, assignment)), assignment) for assignment in feasible)
            repaired = list(repaired)
        else:
            distance, repaired = None, None
        repairs.append(dict(candidate_id=candidate["candidate_id"], original_valid=tuple(original) in feasible_set,
                            edit_distance=distance, repaired=repaired))
    accepted, selected = [], []
    for repair in sorted((row for row in repairs if row["repaired"] is not None), key=lambda row: (row["edit_distance"], row["candidate_id"])):
        if all(sum(a != b for a, b in zip(repair["repaired"], prior)) >= task["min_distance"] for prior in accepted):
            accepted.append(repair["repaired"])
            selected.append(repair["candidate_id"])
    return dict(feasible_count=len(feasible), repairs=repairs, diverse_selection=selected)


def agent_history(task):
    actual = dict(task["initial_locations"])
    beliefs = {agent["agent_id"]: dict(agent["initial_beliefs"]) for agent in task["agents"]}
    for event in task["events"]:
        actual[event["object"]] = event["to"]
        for agent_id in event["observed_by"]:
            beliefs[agent_id][event["object"]] = event["to"]
    output = []
    for agent in task["agents"]:
        state = beliefs[agent["agent_id"]]
        chosen = next((obj for obj in agent["object_priority"] if state[obj] is not None), None)
        action = dict(type="search", object=chosen, place=state[chosen]) if chosen is not None else dict(type="abstain")
        output.append(dict(agent_id=agent["agent_id"], beliefs=state,
                           false_belief_objects=sorted(obj for obj in task["objects"] if state[obj] is not None and state[obj] != actual[obj]),
                           unknown_objects=sorted(obj for obj in task["objects"] if state[obj] is None), action=action))
    return dict(actual_locations=actual, agents=output)


def solve(task):
    functions = dict(skills=skills, modules=modules, episodic=episodic, belief=belief,
                     predictive_abstraction=predictive_abstraction, constraint_repair=constraint_repair, agent_history=agent_history)
    return functions[task["kind"]](task)
