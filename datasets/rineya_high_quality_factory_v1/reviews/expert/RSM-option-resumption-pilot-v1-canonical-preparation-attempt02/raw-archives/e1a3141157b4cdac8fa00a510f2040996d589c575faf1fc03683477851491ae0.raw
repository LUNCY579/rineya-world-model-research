"""Independent producer-side option_resume interpreter: preparation only.

Authored by /root/relational_critic_producer from frozen public contract v1.
No imports or dispatch to factory references, independent oracles or checkers.
This module is not a production runner and writes no files.
"""
from copy import deepcopy

SOURCE_VERSION = "rsm-option-preparation-reference-0.1.0"
TASK_FIELDS = {"kind", "version", "initial_state", "options", "entry", "mode",
               "history", "checkpoint", "event_budget"}
TERMINAL = {"completed", "initiation_failed", "success_failed"}


class BadProgram(ValueError):
    pass


def _require(value):
    if not value:
        raise BadProgram()


def _integer(value):
    return type(value) is int


def _object(value, keys):
    return type(value) is dict and set(value) == set(keys)


def typed_equal(left, right):
    """JSON equality with bool distinct from integer, and ordered lists."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            typed_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _name(value):
    return type(value) is str and bool(value)


def _validate_expression(expr, parameters):
    _require(type(expr) is dict and len(expr) == 1)
    if "const" in expr:
        _require(_integer(expr["const"]))
    elif "reg" in expr:
        _require(type(expr["reg"]) is str and expr["reg"] in ("x", "y"))
    elif "param" in expr:
        _require(_name(expr["param"]) and expr["param"] in parameters)
    else:
        raise BadProgram()


def _validate_predicate(predicate, parameters):
    _require(_object(predicate, {"all"}) and type(predicate["all"]) is list)
    for atom in predicate["all"]:
        _require(_object(atom, {"left", "cmp", "right"}))
        _require(type(atom["cmp"]) is str and atom["cmp"] in ("lt", "eq", "ge"))
        _validate_expression(atom["left"], parameters)
        _validate_expression(atom["right"], parameters)


def _validate_arguments(arguments, callee, parameters, options):
    _require(_name(callee) and callee in options)
    _require(_object(arguments, options[callee]["params"]))
    for value in arguments.values():
        _validate_expression(value, parameters)


def _validate_body(body, parameters, options, edges):
    _require(type(body) is list)
    for instruction in body:
        _require(type(instruction) is dict and type(instruction.get("op")) is str)
        operation = instruction["op"]
        if operation in ("add", "scale"):
            _require(_object(instruction, {"op", "register", "value"}))
            _require(type(instruction["register"]) is str and instruction["register"] in ("x", "y"))
            _validate_expression(instruction["value"], parameters)
        elif operation == "swap":
            _require(_object(instruction, {"op"}))
        elif operation == "call":
            _require(_object(instruction, {"op", "option", "args"}))
            _validate_arguments(instruction["args"], instruction["option"], parameters, options)
            edges.add(instruction["option"])
        elif operation == "if":
            _require(_object(instruction, {"op", "test", "then", "else"}))
            _validate_predicate(instruction["test"], parameters)
            _validate_body(instruction["then"], parameters, options, edges)
            _validate_body(instruction["else"], parameters, options, edges)
        else:
            raise BadProgram()


def validate_program(task):
    """Validate stage one only; history and checkpoint are later stages."""
    _require(_object(task, TASK_FIELDS))
    _require(type(task["kind"]) is str and task["kind"] == "option_resume")
    _require(_integer(task["version"]) and task["version"] == 1)
    _require(_object(task["initial_state"], {"x", "y"}))
    _require(all(_integer(value) for value in task["initial_state"].values()))
    _require(type(task["mode"]) is str and task["mode"] in ("fresh", "resume", "restart"))
    _require(_integer(task["event_budget"]) and task["event_budget"] >= 0)
    options = task["options"]
    _require(type(options) is dict and bool(options) and all(_name(name) for name in options))
    for option in options.values():
        _require(_object(option, {"params", "initiation", "success", "body"}))
        parameters = option["params"]
        _require(type(parameters) is list and all(_name(name) for name in parameters))
        _require(len(set(parameters)) == len(parameters))
    calls = {}
    for name, option in options.items():
        parameters = set(option["params"])
        _validate_predicate(option["initiation"], parameters)
        _validate_predicate(option["success"], parameters)
        calls[name] = set()
        _validate_body(option["body"], parameters, options, calls[name])
    _require(_object(task["entry"], {"option", "args"}))
    _validate_arguments(task["entry"]["args"], task["entry"]["option"], set(), options)
    pending, checked = set(), set()

    def visit(name):
        _require(name not in pending)
        if name in checked:
            return
        pending.add(name)
        for destination in sorted(calls[name]):
            visit(destination)
        pending.remove(name)
        checked.add(name)

    for name in sorted(options):
        visit(name)


def _read(expr, state, bindings):
    if "const" in expr:
        return expr["const"]
    if "reg" in expr:
        return state[expr["reg"]]
    return bindings[expr["param"]]


def _test(predicate, state, bindings):
    outcomes = []
    for atom in predicate["all"]:
        left, right = _read(atom["left"], state, bindings), _read(atom["right"], state, bindings)
        comparison = atom["cmp"]
        outcomes.append(left < right if comparison == "lt" else left == right if comparison == "eq" else left >= right)
    return all(outcomes)


def _new_machine(state):
    return {"state": deepcopy(state), "frames": [], "next_instance_id": 0,
            "root_pending": True, "phase": "running"}


def _instruction_list(frame, options):
    body = options[frame["option"]]["body"]
    path = frame["body_path"]
    for index in range(0, len(path), 2):
        body = body[path[index]][path[index + 1]]
    return body


def _normalize(machine, options):
    while machine["frames"]:
        frame = machine["frames"][-1]
        if frame["frame_kind"] != "branch" or frame["next_index"] < len(_instruction_list(frame, options)):
            break
        machine["frames"].pop()


def _enter(machine, options, option_name, bindings, call_site):
    instance = machine["next_instance_id"]
    machine["next_instance_id"] += 1
    allowed = _test(options[option_name]["initiation"], machine["state"], bindings)
    event = {"event": "enter", "instance_id": instance, "option": option_name,
             "bindings": deepcopy(bindings), "call_site": deepcopy(call_site), "initiation_ok": allowed}
    if allowed:
        machine["frames"].append({"frame_kind": "option", "instance_id": instance,
                                  "option": option_name, "bindings": deepcopy(bindings),
                                  "body_path": [], "next_index": 0})
    else:
        machine["phase"] = "initiation_failed"
    return event


def _step(machine, task):
    """Commit precisely one event to a validated running machine."""
    options = task["options"]
    _normalize(machine, options)
    if machine["root_pending"]:
        request = task["entry"]
        bindings = {key: _read(value, machine["state"], {}) for key, value in request["args"].items()}
        machine["root_pending"] = False
        event = _enter(machine, options, request["option"], bindings, None)
    else:
        frame = machine["frames"][-1]
        body = _instruction_list(frame, options)
        index = frame["next_index"]
        if index == len(body):
            success = _test(options[frame["option"]]["success"], machine["state"], frame["bindings"])
            event = {"event": "return", "instance_id": frame["instance_id"],
                     "option": frame["option"], "success_ok": success}
            machine["frames"].pop()
            _normalize(machine, options)
            if not success:
                machine["phase"] = "success_failed"
            elif not machine["frames"]:
                machine["phase"] = "completed"
        else:
            instruction = body[index]
            operation = instruction["op"]
            path = deepcopy(frame["body_path"])
            frame["next_index"] += 1
            if operation == "call":
                bindings = {key: _read(value, machine["state"], frame["bindings"])
                            for key, value in instruction["args"].items()}
                site = {"caller_instance": frame["instance_id"], "body_path": path, "index": index}
                event = _enter(machine, options, instruction["option"], bindings, site)
            elif operation == "if":
                condition = _test(instruction["test"], machine["state"], frame["bindings"])
                event = {"event": "branch", "instance_id": frame["instance_id"],
                         "option": frame["option"], "body_path": path, "index": index, "condition": condition}
                machine["frames"].append({"frame_kind": "branch", "instance_id": frame["instance_id"],
                                          "option": frame["option"], "bindings": deepcopy(frame["bindings"]),
                                          "body_path": path + [index, "then" if condition else "else"], "next_index": 0})
            else:
                event = {"event": "effect", "instance_id": frame["instance_id"],
                         "option": frame["option"], "body_path": path, "index": index, "op": operation}
                state = machine["state"]
                if operation == "swap":
                    state["x"], state["y"] = state["y"], state["x"]
                else:
                    register = instruction["register"]
                    operand = _read(instruction["value"], state, frame["bindings"])
                    event.update(register=register, value=operand)
                    state[register] = state[register] + operand if operation == "add" else state[register] * operand
                event["state"] = deepcopy(state)
    _normalize(machine, options)
    return event


def _invalid(code):
    return {"status": "invalid", "error": code, "checkpoint": None, "new_events": [],
            "previous_effects": None, "new_effects": 0, "next_event": None}


def solve(task):
    """Return the unique typed public result; never trust a supplied stack."""
    try:
        validate_program(task)
    except (BadProgram, RecursionError):
        return _invalid("invalid_task")
    machine = _new_machine(task["initial_state"])
    history = task["history"]
    if type(history) is not list:
        return _invalid("invalid_history")
    for supplied in history:
        if machine["phase"] != "running":
            return _invalid("invalid_history")
        expected = _step(machine, task)
        if not typed_equal(supplied, expected):
            return _invalid("invalid_history")
    if task["mode"] == "fresh":
        if history or task["checkpoint"] is not None:
            return _invalid("invalid_checkpoint")
    elif not typed_equal(task["checkpoint"], machine):
        return _invalid("invalid_checkpoint")
    if task["mode"] == "restart":
        machine = _new_machine(machine["state"])
    events = []
    for _ in range(task["event_budget"]):
        if machine["phase"] != "running":
            break
        events.append(_step(machine, task))
    paused = machine["phase"] == "running"
    predicted = _step(deepcopy(machine), task) if paused else None
    return {"status": "paused" if paused else machine["phase"], "error": None,
            "checkpoint": deepcopy(machine), "new_events": events,
            "previous_effects": sum(event["event"] == "effect" for event in history),
            "new_effects": sum(event["event"] == "effect" for event in events), "next_event": predicted}


def trusted_machine_probe(task, machine, event_budget):
    """Diagnostic only: run a copied, already trusted machine, without validation.

    This separate API never changes solve's history/checkpoint rejection policy.
    It permits measuring whether a hand-mutated saved binding is active.
    """
    copied = deepcopy(machine)
    events = []
    for _ in range(event_budget):
        if copied["phase"] != "running":
            break
        events.append(_step(copied, task))
    return {"checkpoint": copied, "events": events}
