"""An independently authored executable interpretation of SEMANTICS.md.

This module deliberately does not import a generator, stored answer, or model.
Search states use immutable tuples, and optimal plans are found by breadth-first
enumeration of legal successors in the contract's lexical action order.
"""

from collections import deque


class OracleError(ValueError):
    """The supplied task is outside the executable semantic contract."""


def _require(test, message):
    if not test:
        raise OracleError(message)


def _mapping(value, label):
    _require(type(value) is dict, label + " must be an object")
    return value


def _list(value, label):
    _require(type(value) is list, label + " must be a list")
    return value


def _integer(value, label, minimum=None):
    _require(type(value) is int, label + " must be an integer, excluding bool")
    if minimum is not None:
        _require(value >= minimum, label + " is below its minimum")
    return value


def _fields(value, required, label, exact=False):
    _mapping(value, label)
    _require(set(required).issubset(value), label + " lacks required fields")
    if exact:
        _require(set(value) == set(required), label + " has unsupported fields")


_VARS = frozenset(("event.status", "event.amount", "state.x", "state.y"))
_COMPARISONS = frozenset(("eq", "ne", "ge", "gt", "le", "lt"))


def _scalar(value):
    return value is None or type(value) in (bool, int, str)


def _validate_ast(node, active=None):
    """Validate the whole tree, including branches that may short-circuit."""
    _mapping(node, "AST node")
    if active is None:
        active = set()
    _require(id(node) not in active, "AST contains a reference cycle")
    active.add(id(node))
    try:
        keys = set(node)
        if keys == {"var"}:
            _require(type(node["var"]) is str and node["var"] in _VARS,
                     "AST variable is unsupported")
            return None if node["var"] == "event.status" else int
        if keys == {"const"}:
            _require(_scalar(node["const"]), "AST constant must be a supported JSON scalar")
            return type(node["const"])
        _require(keys == {"op", "args"}, "AST must have exactly one recognized node shape")
        operation = node["op"]
        _require(type(operation) is str and operation in _COMPARISONS | {"all", "any", "not"},
                 "AST operation is unsupported")
        children = _list(node["args"], "AST args")
        if operation in _COMPARISONS:
            _require(len(children) == 2, "comparison needs exactly two arguments")
        elif operation == "not":
            _require(len(children) == 1, "not needs exactly one argument")
        child_types = [_validate_ast(child, active) for child in children]
        if operation in {"all", "any", "not"}:
            _require(all(kind is bool or kind is None for kind in child_types),
                     "boolean AST operators require boolean operands")
        elif operation in {"ge", "gt", "le", "lt"}:
            known = [kind for kind in child_types if kind is not None]
            _require(all(kind in (int, str) for kind in known) and len(set(known)) <= 1,
                     "ordered comparison needs same-type integer or string operands")
        return bool
    finally:
        active.remove(id(node))


def _eval_ast(node, state, event):
    if "const" in node:
        return node["const"]
    if "var" in node:
        scope, name = node["var"].split(".")
        return (event if scope == "event" else state)[name]
    operation = node["op"]
    values = [_eval_ast(child, state, event) for child in node["args"]]
    if operation == "all":
        _require(all(type(value) is bool for value in values), "all operands must be boolean")
        return all(values)
    if operation == "any":
        _require(all(type(value) is bool for value in values), "any operands must be boolean")
        return any(values)
    if operation == "not":
        _require(type(values[0]) is bool, "not operand must be boolean")
        return not values[0]
    left, right = values
    if operation in {"eq", "ne"}:
        equal = type(left) is type(right) and left == right
        return equal if operation == "eq" else not equal
    _require(type(left) is type(right) and type(left) in (int, str),
             "ordered comparison needs same-type integer or string operands")
    if operation == "ge":
        return left >= right
    if operation == "gt":
        return left > right
    if operation == "le":
        return left <= right
    return left < right


def _causal(task):
    _fields(task, ("state", "rule", "precondition", "events"), "causal task")
    _fields(task["state"], ("x", "y"), "causal state", exact=True)
    state = {name: _integer(task["state"][name], "state." + name) for name in ("x", "y")}
    for name in ("rule", "precondition"):
        _require(_validate_ast(task[name]) in (bool, None), name + " must have a boolean root")
    events = _list(task["events"], "events")
    for event in events:
        _fields(event, ("status", "amount", "op", "source", "destination"), "event", exact=True)
        _require(_scalar(event["status"]), "event.status must be a supported JSON scalar")
        _integer(event["amount"], "event.amount", 1)
        _require(type(event["op"]) is str and event["op"] in {"add", "scale", "transfer", "swap"},
                 "event.op is unsupported")
        for name in ("source", "destination"):
            _require(type(event[name]) is str and event[name] in {"x", "y"},
                     "event " + name + " must identify x or y")
        if event["op"] in {"transfer", "swap"}:
            _require(event["source"] != event["destination"], "transfer/swap endpoints must differ")
    steps = []
    for event in events:
        rule = _eval_ast(task["rule"], state, event)
        precondition = _eval_ast(task["precondition"], state, event)
        _require(type(rule) is bool and type(precondition) is bool, "gates must evaluate to bool")
        operation = "identity"
        if rule and precondition:
            operation = event["op"]
            source, destination, amount = event["source"], event["destination"], event["amount"]
            if operation == "add":
                state[source] = state[source] + amount
            elif operation == "scale":
                state[source] = state[source] * amount
            elif operation == "transfer":
                old_source, old_destination = state[source], state[destination]
                state[source], state[destination] = old_source - amount, old_destination + amount
            else:
                old_source, old_destination = state[source], state[destination]
                state[source], state[destination] = old_destination, old_source
        steps.append({"cmp": {"rule": rule, "precondition": precondition},
                      "calc": {"operation": operation}, "after": dict(state)})
    return {"after": dict(state), "steps": steps}


def _graph_world(task):
    _fields(task, ("nodes", "edges", "start", "goal", "fuel"), "graph world")
    nodes = _list(task["nodes"], "nodes")
    _require(all(type(node) is str for node in nodes), "nodes must be strings")
    _require(len(set(nodes)) == len(nodes), "nodes must be distinct")
    node_set = set(nodes)
    for name in ("start", "goal"):
        _require(type(task[name]) is str and task[name] in node_set, name + " must be a world node")
    fuel = _integer(task["fuel"], "fuel", 0)
    edges = {}
    for edge in _list(task["edges"], "edges"):
        _require(type(edge) is list and len(edge) == 3, "edge must be a three-element list")
        origin, destination, cost = edge
        _require(type(origin) is str and origin in node_set and type(destination) is str
                 and destination in node_set, "edge endpoints must be world nodes")
        _integer(cost, "edge cost", 1)
        _require((origin, destination) not in edges, "parallel ordered edges are unsupported")
        edges[(origin, destination)] = cost
    adjacent = {node: [] for node in nodes}
    for (origin, destination), cost in edges.items():
        adjacent[origin].append((destination, cost))
    for neighbors in adjacent.values():
        neighbors.sort()
    return (task["start"], fuel), task["goal"], node_set, edges, adjacent


def _graph_plan(plan):
    _list(plan, "graph plan")
    _require(all(type(destination) is str for destination in plan), "graph actions must be strings")


def _graph_run(initial, edges, plan):
    current = initial
    steps = []
    error = None
    for index, destination in enumerate(plan):
        node, fuel = current
        cost = edges.get((node, destination))
        legal = cost is not None and cost <= fuel
        if legal:
            current = (destination, fuel - cost)
        steps.append({"legal": legal, "after": {"node": current[0], "fuel": current[1]}})
        if not legal:
            error = index
            break
    return current, steps, error


def _graph_search(initial, goal, adjacent):
    frontier = deque([(initial, ())])
    visited = {initial}
    while frontier:
        state, path = frontier.popleft()
        node, fuel = state
        if node == goal:
            return list(path)
        for destination, cost in adjacent[node]:
            following = (destination, fuel - cost)
            if cost <= fuel and following not in visited:
                visited.add(following)
                frontier.append((following, path + (destination,)))
    return None


def _graph(task):
    _fields(task, ("plan",), "graph task")
    initial, goal, _, edges, adjacent = _graph_world(task)
    _graph_plan(task["plan"])
    state, steps, error = _graph_run(initial, edges, task["plan"])
    valid = error is None
    return {"steps": steps, "after": {"node": state[0], "fuel": state[1]},
            "valid": valid, "reaches_goal": valid and state[0] == goal,
            "shortest_plan": _graph_search(initial, goal, adjacent)}


def _blocks_world(task):
    _fields(task, ("on", "goal_on", "plan"), "blocks task")
    on = _mapping(task["on"], "on")
    _require(all(type(block) is str and block != "TABLE" for block in on),
             "block names must be strings other than TABLE")
    names = tuple(sorted(on))
    allowed = set(names) | {"TABLE"}
    supported = []
    for block, support in on.items():
        _require(type(support) is str and support in allowed and support != block,
                 "initial block support is invalid")
        if support != "TABLE":
            supported.append(support)
    _require(len(supported) == len(set(supported)), "multiple blocks occupy one support")
    for block in names:
        seen = set()
        cursor = block
        while cursor != "TABLE":
            _require(cursor not in seen, "initial block state contains a cycle")
            seen.add(cursor)
            cursor = on[cursor]
    goal = _mapping(task["goal_on"], "goal_on")
    for block, support in goal.items():
        _require(type(block) is str and block in on and type(support) is str and support in allowed
                 and support != block,
                 "goal support must refer to an existing block or TABLE")
    goal_supports = [support for support in goal.values() if support != "TABLE"]
    _require(len(goal_supports) == len(set(goal_supports)), "goal assigns multiple blocks to one support")
    for block in goal:
        seen = set()
        cursor = block
        while cursor in goal:
            _require(cursor not in seen, "goal contains a support cycle")
            seen.add(cursor)
            cursor = goal[cursor]
    plan = _list(task["plan"], "blocks plan")
    for action in plan:
        _fields(action, ("block", "to"), "block action", exact=True)
        _require(type(action["block"]) is str and type(action["to"]) is str,
                 "block action names must be strings")
    return names, tuple(on[name] for name in names), goal, plan


def _blocks_successor(names, state, block, destination):
    positions = dict(zip(names, state))
    occupied = set(state) - {"TABLE"}
    if block not in positions or block in occupied or destination == block:
        return None
    if destination != "TABLE" and (destination not in positions or destination in occupied):
        return None
    if destination == positions[block]:
        return None
    return tuple(destination if name == block else support for name, support in zip(names, state))


def _blocks_goal(names, state, goal):
    positions = dict(zip(names, state))
    return all(positions[block] == support for block, support in goal.items())


def _blocks_search(names, initial, goal):
    frontier = deque([(initial, ())])
    visited = {initial}
    destinations = sorted((*names, "TABLE"))
    while frontier:
        state, path = frontier.popleft()
        if _blocks_goal(names, state, goal):
            return [{"block": block, "to": destination} for block, destination in path]
        for block in names:
            for destination in destinations:
                following = _blocks_successor(names, state, block, destination)
                if following is not None and following not in visited:
                    visited.add(following)
                    frontier.append((following, path + ((block, destination),)))
    return None


def _blocks(task):
    names, initial, goal, plan = _blocks_world(task)
    state = initial
    steps = []
    valid = True
    for action in plan:
        following = _blocks_successor(names, state, action["block"], action["to"])
        legal = following is not None
        if legal:
            state = following
        steps.append({"legal": legal, "after": dict(zip(names, state))})
        if not legal:
            valid = False
            break
    return {"steps": steps, "after": dict(zip(names, state)), "valid": valid,
            "reaches_goal": valid and _blocks_goal(names, state, goal),
            "shortest_plan": _blocks_search(names, initial, goal)}


def _critic(task):
    _fields(task, ("world", "candidates"), "critic task")
    world = _mapping(task["world"], "critic world")
    _require(world.get("kind") == "graph", "critic world must be kind=graph")
    initial, goal, node_set, edges, adjacent = _graph_world(world)
    repair = _graph_search(initial, goal, adjacent)
    initial_distance = None if repair is None else len(repair)
    candidates = _list(task["candidates"], "candidates")
    ids = set()
    for candidate in candidates:
        _fields(candidate, ("candidate_id", "plan", "claimed_after"), "candidate", exact=True)
        identity = candidate["candidate_id"]
        _require(type(identity) is str and identity not in ids, "candidate IDs must be distinct strings")
        ids.add(identity)
        _graph_plan(candidate["plan"])
        claim = candidate["claimed_after"]
        _fields(claim, ("node", "fuel"), "claimed_after", exact=True)
        _require(type(claim["node"]) is str and claim["node"] in node_set,
                 "claimed node must be a world node")
        _integer(claim["fuel"], "claimed fuel", 0)
    evaluations = []
    scores = {}
    for candidate in candidates:
        state, _, first_illegal = _graph_run(initial, edges, candidate["plan"])
        valid = first_illegal is None
        claim_correct = candidate["claimed_after"] == {"node": state[0], "fuel": state[1]}
        reaches_goal = valid and state[0] == goal
        remaining_plan = _graph_search(state, goal, adjacent)
        remaining = None if remaining_plan is None else len(remaining_plan)
        progress = (initial_distance - remaining if valid and initial_distance is not None
                    and remaining is not None else None)
        usefulness = (-1 if not valid else 2 if reaches_goal else
                      1 if remaining is not None and progress is not None and progress > 0 else 0)
        first_error = first_illegal if not valid else "claim" if not claim_correct else None
        evaluations.append({"candidate_id": candidate["candidate_id"], "valid": valid,
                            "claim_correct": claim_correct, "reaches_goal": reaches_goal,
                            "remaining_moves": remaining, "progress": progress,
                            "usefulness": usefulness, "first_error": first_error,
                            "repair_plan": None if repair is None else list(repair)})
        distance_score = float("-inf") if remaining is None else -remaining
        scores[candidate["candidate_id"]] = (valid and claim_correct, reaches_goal, usefulness,
                                            distance_score, -len(candidate["plan"]))
    ordered = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    groups = []
    last_score = None
    for identity in ordered:
        if not groups or scores[identity] != last_score:
            groups.append([])
            last_score = scores[identity]
        groups[-1].append(identity)
    for group in groups:
        group.sort()
    return {"evaluations": evaluations, "ranking": groups}


def solve(task):
    """Return the semantic target for an original or extended task payload.

    Malformed structures raise OracleError. Syntactically valid illegal actions
    are represented in the target and stop execution; they are not parse errors.
    Extra task-level metadata is ignored. AST/action/state shapes are strict.
    """
    _mapping(task, "task")
    kind = task.get("kind")
    _require(type(kind) is str and kind in {"causal", "graph", "blocks", "critic", "skills", "modules",
                                          "episodic", "belief", "predictive_abstraction", "constraint_repair",
                                          "agent_history", "option_resume"},
             "task.kind is unsupported or absent")
    if kind == "option_resume":
        from factory.rsm_controls import option_resume_solve
        return option_resume_solve(task)
    original = {"causal": _causal, "graph": _graph, "blocks": _blocks, "critic": _critic}
    if kind in original:
        return original[kind](task)
    from .extended_independent import solve as solve_extended
    return solve_extended(task)
