"""Independent finite interpretation of EXTENDED_SEMANTICS.md.

This module imports only the original oracle's validation primitives. No
producer reference, generator, stored target, or review is an input.
Predictive equivalence is decided by exhaustive paired-state reachability,
independently of the contract's partition-refinement presentation.
"""

from collections import deque
from itertools import product
import json

from .independent import OracleError, _fields, _integer, _list, _mapping, _require, _scalar


def _strings(value, label, nonempty=False):
    _list(value, label)
    _require(all(type(item) is str for item in value), label + " must contain strings")
    _require(len(value) == len(set(value)), label + " must contain distinct strings")
    _require(not nonempty or bool(value), label + " must be nonempty")
    return value


def _value(value, label):
    _require(_scalar(value), label + " must be a JSON scalar without floats")
    return value


def _token(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _same(left, right):
    return type(left) is type(right) and left == right


def _name(value, label):
    _require(type(value) is str, label + " must be a string")
    return value


def _skills(task):
    _fields(task, ("state", "procedures", "main"), "skills task")
    _fields(task["state"], ("x", "y"), "state", exact=True)
    state = {key: _integer(task["state"][key], "state register") for key in ("x", "y")}
    procedures = _mapping(task["procedures"], "procedures")
    _require(all(type(name) is str for name in procedures), "procedure names must be strings")

    def validate(instructions, active=None):
        _list(instructions, "instructions")
        if active is None:
            active = set()
        _require(id(instructions) not in active, "instruction lists contain a cycle")
        active.add(id(instructions))
        calls = set()
        try:
            for instruction in instructions:
                _mapping(instruction, "instruction")
                operation = instruction.get("op")
                _require(type(operation) is str, "instruction op must be a string")
                if operation in ("add", "scale"):
                    _fields(instruction, ("op", "register", "value"), "arithmetic instruction", exact=True)
                    _require(instruction["register"] in ("x", "y"), "unsupported register")
                    _integer(instruction["value"], "instruction value")
                elif operation == "swap":
                    _fields(instruction, ("op",), "swap instruction", exact=True)
                elif operation == "call":
                    _fields(instruction, ("op", "name"), "call instruction", exact=True)
                    _require(type(instruction["name"]) is str and instruction["name"] in procedures,
                             "call must name an existing procedure")
                    calls.add(instruction["name"])
                elif operation == "if":
                    _fields(instruction, ("op", "test", "then", "else"), "if instruction", exact=True)
                    test = instruction["test"]
                    _fields(test, ("register", "cmp", "value"), "if test", exact=True)
                    _require(test["register"] in ("x", "y") and test["cmp"] in ("lt", "eq", "ge"),
                             "if test register or comparison is unsupported")
                    _integer(test["value"], "comparison value")
                    calls.update(validate(instruction["then"], active))
                    calls.update(validate(instruction["else"], active))
                else:
                    raise OracleError("unsupported skills instruction")
        finally:
            active.remove(id(instructions))
        return calls

    dependencies = {name: validate(body) for name, body in procedures.items()}
    validate(task["main"])
    # Check all declared procedures, including unreachable calls and both branches.
    finished = set()
    for start in procedures:
        pending = [(start, False)]
        ancestors = set()
        while pending:
            name, returning = pending.pop()
            if returning:
                ancestors.remove(name)
                finished.add(name)
            elif name not in finished:
                _require(name not in ancestors, "procedure call graph contains a cycle")
                ancestors.add(name)
                pending.append((name, True))
                pending.extend((child, False) for child in sorted(dependencies[name], reverse=True))
    result = {"after": state, "executed_primitives": 0, "call_order": [], "conditions": []}

    def execute(instructions):
        for instruction in instructions:
            operation = instruction["op"]
            if operation == "call":
                result["call_order"].append(instruction["name"])
                execute(procedures[instruction["name"]])
            elif operation == "if":
                test = instruction["test"]
                left, right = state[test["register"]], test["value"]
                condition = left < right if test["cmp"] == "lt" else left == right if test["cmp"] == "eq" else left >= right
                result["conditions"].append(condition)
                execute(instruction["then"] if condition else instruction["else"])
            else:
                result["executed_primitives"] += 1
                if operation == "swap":
                    state["x"], state["y"] = state["y"], state["x"]
                elif operation == "add":
                    state[instruction["register"]] += instruction["value"]
                else:
                    state[instruction["register"]] *= instruction["value"]

    execute(task["main"])
    return result


def _modules(task):
    _fields(task, ("modules", "memory", "protected", "requests"), "modules task")
    modules = _list(task["modules"], "modules")
    ids = set()
    for module in modules:
        _fields(module, ("module_id", "capabilities", "priority"), "module", exact=True)
        identity = _name(module["module_id"], "module_id")
        _require(identity not in ids, "duplicate module ID")
        ids.add(identity)
        _strings(module["capabilities"], "capabilities")
        _integer(module["priority"], "priority")
    memory = dict(_mapping(task["memory"], "memory"))
    for key, value in memory.items():
        _name(key, "memory key")
        _value(value, "memory value")
    protected = set(_strings(task["protected"], "protected"))
    _require(protected <= set(memory), "protected keys must exist in initial memory")
    requests = _list(task["requests"], "requests")
    ids = set()
    for request in requests:
        _fields(request, ("request_id", "needs", "writes"), "request", exact=True)
        identity = _name(request["request_id"], "request_id")
        _require(identity not in ids, "duplicate request ID")
        ids.add(identity)
        _strings(request["needs"], "needs")
        for write in _list(request["writes"], "writes"):
            _fields(write, ("key", "value"), "write", exact=True)
            _name(write["key"], "write key")
            _value(write["value"], "write value")
    routes = []
    for request in requests:
        eligible = [module for module in modules if set(request["needs"]) <= set(module["capabilities"])]
        selected = min(eligible, key=lambda module: (module["priority"], module["module_id"])) if eligible else None
        blocked = []
        if selected is not None:
            for index, write in enumerate(request["writes"]):
                if write["key"] in protected:
                    blocked.append(index)
                else:
                    memory[write["key"]] = write["value"]
        routes.append({"request_id": request["request_id"],
                       "selected_module": None if selected is None else selected["module_id"],
                       "eligible_modules": sorted(module["module_id"] for module in eligible),
                       "blocked_writes": blocked})
    return {"routes": routes, "memory": memory}


def _episodic(task):
    _fields(task, ("events", "queries"), "episodic task")
    events = _list(task["events"], "events")
    for event in events:
        _fields(event, ("key", "version", "at", "op", "value"), "event", exact=True)
        _name(event["key"], "event key")
        _integer(event["version"], "version", 0)
        _integer(event["at"], "at", 0)
        _require(event["op"] in ("put", "delete"), "unsupported event operation")
        _value(event["value"], "event value")
        _require(event["op"] != "delete" or event["value"] is None, "delete value must be null")
    queries = _list(task["queries"], "queries")
    ids = set()
    for query in queries:
        _fields(query, ("query_id", "key", "as_of"), "query", exact=True)
        identity = _name(query["query_id"], "query_id")
        _require(identity not in ids, "duplicate query ID")
        ids.add(identity)
        _name(query["key"], "query key")
        _integer(query["as_of"], "as_of", 0)
    results = []
    for query in queries:
        eligible = [(event["version"], event["at"], index) for index, event in enumerate(events)
                    if event["key"] == query["key"] and event["at"] <= query["as_of"]]
        if eligible:
            version, _, index = max(eligible)
            event = events[index]
            status = "deleted" if event["op"] == "delete" else "present"
            value = event["value"]
        else:
            version, index, value, status = None, None, None, "absent"
        results.append({"query_id": query["query_id"], "key": query["key"], "status": status,
                        "value": value, "chosen_version": version, "chosen_event": index})
    return {"results": results}


def _belief(task):
    _fields(task, ("worlds", "observations", "queries"), "belief task")
    worlds = _list(task["worlds"], "worlds")
    ids, fact_keys = set(), None
    for world in worlds:
        _fields(world, ("world_id", "facts"), "world", exact=True)
        identity = _name(world["world_id"], "world_id")
        _require(identity not in ids, "duplicate world ID")
        ids.add(identity)
        facts = _mapping(world["facts"], "facts")
        for key, value in facts.items():
            _name(key, "fact name")
            _value(value, "fact value")
        _require(fact_keys is None or fact_keys == set(facts), "world fact keys must agree")
        fact_keys = set(facts)
    if fact_keys is None:
        fact_keys = set()
    observations = _list(task["observations"], "observations")
    for observation in observations:
        _fields(observation, ("fact", "equals"), "observation", exact=True)
        _require(type(observation["fact"]) is str and observation["fact"] in fact_keys,
                 "observation must name an existing fact")
        _value(observation["equals"], "observation value")
    queries = _list(task["queries"], "queries")
    ids = set()
    for query in queries:
        _fields(query, ("query_id", "fact"), "query", exact=True)
        identity = _name(query["query_id"], "query_id")
        _require(identity not in ids, "duplicate query ID")
        ids.add(identity)
        _require(type(query["fact"]) is str and query["fact"] in fact_keys, "query must name an existing fact")
    possible = list(worlds)
    sizes = []
    for observation in observations:
        possible = [world for world in possible if _same(world["facts"][observation["fact"]], observation["equals"])]
        sizes.append(len(possible))
    rows = []
    for query in queries:
        counted = {}
        for world in possible:
            value = world["facts"][query["fact"]]
            token = _token(value)
            if token not in counted:
                counted[token] = {"value": value, "count": 0}
            counted[token]["count"] += 1
        counts = [counted[token] for token in sorted(counted)]
        entailed = bool(possible) and len(counts) == 1
        rows.append({"query_id": query["query_id"], "counts": counts,
                     "distinct_values": len(counts), "entailed": entailed,
                     "guaranteed": counts[0]["value"] if entailed else None,
                     "can_reduce_uncertainty": len(counts) > 1})
    return {"surviving_worlds": sorted(world["world_id"] for world in possible),
            "belief_sizes": sizes, "queries": rows}


def _predictive(task):
    _fields(task, ("states", "actions", "observable", "transition"), "predictive task")
    states = _strings(task["states"], "states", nonempty=True)
    actions = _strings(task["actions"], "actions", nonempty=True)
    observable = _mapping(task["observable"], "observable")
    transition = _mapping(task["transition"], "transition")
    _require(set(observable) == set(states) and set(transition) == set(states), "state maps must be total and exact")
    for state in states:
        _value(observable[state], "observable")
        successors = _mapping(transition[state], "state transitions")
        _require(set(successors) == set(actions), "action transitions must be total and exact")
        _require(all(type(value) is str and value in states for value in successors.values()),
                 "transition destination must be a state")

    def equivalent(left, right):
        # Every reachable pair represents the observations after one possible
        # common action sequence. Revisiting a pair adds no new information.
        frontier = deque([(left, right)])
        seen = {(left, right)}
        while frontier:
            first, second = frontier.popleft()
            if not _same(observable[first], observable[second]):
                return False
            for action in actions:
                following = (transition[first][action], transition[second][action])
                if following not in seen:
                    seen.add(following)
                    frontier.append(following)
        return True

    classes = []
    for state in sorted(states):
        for group in classes:
            if equivalent(state, group[0]):
                group.append(state)
                break
        else:
            classes.append([state])
    classes.sort()
    indices = {state: index for index, group in enumerate(classes) for state in group}
    abstract_transitions = []
    abstract_observable = []
    for index, group in enumerate(classes):
        representative = group[0]
        for action in sorted(actions):
            abstract_transitions.append({"class_id": index, "action": action,
                                         "next_class": indices[transition[representative][action]]})
        abstract_observable.append({"class_id": index, "value": observable[representative]})
    return {"classes": classes, "state_to_class": indices,
            "abstract_transitions": abstract_transitions, "abstract_observable": abstract_observable}


def _repair(task):
    _fields(task, ("alphabet", "slots", "constraints", "candidates", "min_distance"), "repair task")
    alphabet = _strings(task["alphabet"], "alphabet", nonempty=True)
    slots = _integer(task["slots"], "slots", 1)
    _integer(task["min_distance"], "min_distance", 0)
    constraints = _list(task["constraints"], "constraints")

    def position(value):
        _integer(value, "constraint position", 0)
        _require(value < slots, "constraint position is out of range")

    for constraint in constraints:
        _mapping(constraint, "constraint")
        kind = constraint.get("kind")
        if kind in ("equal", "different"):
            _fields(constraint, ("kind", "positions"), "pair constraint", exact=True)
            positions = _list(constraint["positions"], "positions")
            _require(len(positions) == 2, "pair constraint must have two positions")
            for value in positions:
                position(value)
        elif kind == "allowed":
            _fields(constraint, ("kind", "position", "values"), "allowed constraint", exact=True)
            position(constraint["position"])
            _list(constraint["values"], "allowed values")
            _require(all(type(value) is str and value in alphabet for value in constraint["values"]),
                     "allowed values must be alphabet symbols")
        elif kind == "count":
            _fields(constraint, ("kind", "value", "min", "max"), "count constraint", exact=True)
            _require(type(constraint["value"]) is str and constraint["value"] in alphabet,
                     "count value must be an alphabet symbol")
            _integer(constraint["min"], "count minimum")
            _integer(constraint["max"], "count maximum")
        else:
            raise OracleError("unsupported constraint kind")
    candidates = _list(task["candidates"], "candidates")
    ids = set()
    for candidate in candidates:
        _fields(candidate, ("candidate_id", "assignment"), "candidate", exact=True)
        identity = _name(candidate["candidate_id"], "candidate_id")
        _require(identity not in ids, "duplicate candidate ID")
        ids.add(identity)
        assignment = _list(candidate["assignment"], "assignment")
        _require(len(assignment) == slots and all(type(value) is str and value in alphabet for value in assignment),
                 "candidate assignment must have slots alphabet symbols")

    def feasible(assignment):
        for constraint in constraints:
            kind = constraint["kind"]
            if kind in ("equal", "different"):
                first, second = constraint["positions"]
                equal = assignment[first] == assignment[second]
                if equal != (kind == "equal"):
                    return False
            elif kind == "allowed":
                if assignment[constraint["position"]] not in constraint["values"]:
                    return False
            else:
                count = assignment.count(constraint["value"])
                if not constraint["min"] <= count <= constraint["max"]:
                    return False
        return True

    def distance(left, right):
        return sum(first != second for first, second in zip(left, right))

    possible = [assignment for assignment in product(sorted(alphabet), repeat=slots) if feasible(assignment)]
    possible_set = set(possible)
    repairs = []
    for candidate in candidates:
        original = tuple(candidate["assignment"])
        repaired = min(possible, key=lambda value: (distance(value, original), value)) if possible else None
        repairs.append({"candidate_id": candidate["candidate_id"], "original_valid": original in possible_set,
                        "edit_distance": None if repaired is None else distance(original, repaired),
                        "repaired": None if repaired is None else list(repaired)})
    accepted = []
    selected = []
    for repair in sorted((row for row in repairs if row["repaired"] is not None),
                         key=lambda row: (row["edit_distance"], row["candidate_id"])):
        if all(distance(repair["repaired"], previous) >= task["min_distance"] for previous in accepted):
            accepted.append(repair["repaired"])
            selected.append(repair["candidate_id"])
    return {"feasible_count": len(possible), "repairs": repairs, "diverse_selection": selected}


def _history(task):
    _fields(task, ("objects", "places", "initial_locations", "agents", "events"), "history task")
    objects = _strings(task["objects"], "objects", nonempty=True)
    places = _strings(task["places"], "places", nonempty=True)
    initial = _mapping(task["initial_locations"], "initial_locations")
    _require(set(initial) == set(objects), "initial locations must contain every object exactly")
    _require(all(type(value) is str and value in places for value in initial.values()), "initial location must be a place")
    agents = _list(task["agents"], "agents")
    ids = set()
    for agent in agents:
        _fields(agent, ("agent_id", "initial_beliefs", "object_priority"), "agent", exact=True)
        identity = _name(agent["agent_id"], "agent_id")
        _require(identity not in ids, "duplicate agent ID")
        ids.add(identity)
        beliefs = _mapping(agent["initial_beliefs"], "initial_beliefs")
        _require(set(beliefs) == set(objects), "beliefs must contain every object exactly")
        _require(all(value is None or type(value) is str and value in places for value in beliefs.values()),
                 "beliefs must be places or null")
        priority = _strings(agent["object_priority"], "object_priority")
        _require(set(priority) <= set(objects), "priority objects must exist")
    events = _list(task["events"], "events")
    for event in events:
        _fields(event, ("object", "to", "observed_by"), "history event", exact=True)
        _require(type(event["object"]) is str and event["object"] in objects and
                 type(event["to"]) is str and event["to"] in places, "event object/place must exist")
        _require(set(_strings(event["observed_by"], "observed_by")) <= ids, "observers must be existing agents")
    locations = dict(initial)
    beliefs = {agent["agent_id"]: dict(agent["initial_beliefs"]) for agent in agents}
    for event in events:
        locations[event["object"]] = event["to"]
        for observer in event["observed_by"]:
            beliefs[observer][event["object"]] = event["to"]
    rows = []
    for agent in agents:
        known = beliefs[agent["agent_id"]]
        action = {"type": "abstain"}
        for object_name in agent["object_priority"]:
            if known[object_name] is not None:
                action = {"type": "search", "object": object_name, "place": known[object_name]}
                break
        rows.append({"agent_id": agent["agent_id"], "beliefs": known,
                     "false_belief_objects": sorted(name for name in objects if known[name] is not None and known[name] != locations[name]),
                     "unknown_objects": sorted(name for name in objects if known[name] is None), "action": action})
    return {"actual_locations": locations, "agents": rows}


_KINDS = {"skills": _skills, "modules": _modules, "episodic": _episodic, "belief": _belief,
          "predictive_abstraction": _predictive, "constraint_repair": _repair, "agent_history": _history}


def solve(task):
    """Return an extended semantic target; reject malformed inputs with OracleError."""
    _mapping(task, "task")
    kind = task.get("kind")
    _require(type(kind) is str and kind in _KINDS, "unsupported or absent extended kind")
    return _KINDS[kind](task)
