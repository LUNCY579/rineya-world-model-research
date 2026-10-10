"""Expert-owned option_resume v1 interpreter derived from six public sources.

Pure import: no filesystem reads, candidate access, or test execution. Cohort
bounds/metadata are separate from task semantics. Objects are type-sensitive;
this source does not borrow any producer/formal implementation or hand gold.
"""
import copy
import json
import re

VERSION = "expert-option-resume-0.1.0"
TASK_KEYS = {"kind", "version", "initial_state", "options", "entry", "mode",
             "history", "checkpoint", "event_budget"}


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(typed_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "wrong object shape")


def name(value):
    require(type(value) is str and bool(value), "nonempty string required")


def integer(value):
    require(type(value) is int, "integer excluding boolean required")


def state_shape(state):
    keys(state, {"x", "y"})
    for value in state.values():
        integer(value)


def expression_shape(expr, parameters):
    require(type(expr) is dict and len(expr) == 1, "one expression form required")
    key, value = next(iter(expr.items()))
    if key == "const":
        integer(value)
    elif key == "reg":
        require(type(value) is str and value in ("x", "y"), "unknown register")
    elif key == "param":
        name(value)
        require(value in parameters, "unknown parameter")
    else:
        raise ValueError("unknown expression")


def predicate_shape(pred, parameters):
    keys(pred, {"all"})
    require(type(pred["all"]) is list, "predicate atoms must be list")
    for atom in pred["all"]:
        keys(atom, {"left", "cmp", "right"})
        require(type(atom["cmp"]) is str and atom["cmp"] in ("lt", "eq", "ge"), "unknown comparison")
        expression_shape(atom["left"], parameters)
        expression_shape(atom["right"], parameters)


def argument_shape(args, callee, parameters, options):
    keys(args, options[callee]["params"])
    for expr in args.values():
        expression_shape(expr, parameters)


def validate_task(task):
    # history and checkpoint deliberately await their own validation stages.
    keys(task, TASK_KEYS)
    require(task["kind"] == "option_resume" and type(task["kind"]) is str, "wrong kind")
    require(type(task["version"]) is int and task["version"] == 1, "wrong version")
    state_shape(task["initial_state"])
    require(type(task["options"]) is dict and bool(task["options"]), "nonempty options map")
    options = task["options"]
    for option, definition in options.items():
        name(option)
        keys(definition, {"params", "initiation", "success", "body"})
        parameters = definition["params"]
        require(type(parameters) is list, "parameter declaration must be list")
        for parameter in parameters:
            name(parameter)
        require(len(parameters) == len(set(parameters)), "duplicate parameter")
    edges = {option: set() for option in options}

    def block_shape(block, owner):
        require(type(block) is list, "instruction block must be list")
        parameters = options[owner]["params"]
        for instruction in block:
            require(type(instruction) is dict and type(instruction.get("op")) is str, "instruction op")
            op = instruction["op"]
            if op in ("add", "scale"):
                keys(instruction, {"op", "register", "value"})
                require(type(instruction["register"]) is str and instruction["register"] in ("x", "y"), "bad effect register")
                expression_shape(instruction["value"], parameters)
            elif op == "swap":
                keys(instruction, {"op"})
            elif op == "call":
                keys(instruction, {"op", "option", "args"})
                callee = instruction["option"]
                name(callee)
                require(callee in options, "missing callee")
                argument_shape(instruction["args"], callee, parameters, options)
                edges[owner].add(callee)
            elif op == "if":
                keys(instruction, {"op", "test", "then", "else"})
                predicate_shape(instruction["test"], parameters)
                block_shape(instruction["then"], owner)
                block_shape(instruction["else"], owner)
            else:
                raise ValueError("unknown instruction")

    for option, definition in options.items():
        predicate_shape(definition["initiation"], definition["params"])
        predicate_shape(definition["success"], definition["params"])
        block_shape(definition["body"], option)
    visiting, finished = set(), set()

    def visit(option):
        require(option not in visiting, "recursive call graph")
        if option in finished:
            return
        visiting.add(option)
        for child in edges[option]:
            visit(child)
        visiting.remove(option)
        finished.add(option)

    for option in options:  # includes all unused options and both branches
        visit(option)
    keys(task["entry"], {"option", "args"})
    name(task["entry"]["option"])
    require(task["entry"]["option"] in options, "missing entry option")
    argument_shape(task["entry"]["args"], task["entry"]["option"], [], options)
    require(type(task["mode"]) is str and task["mode"] in ("fresh", "resume", "restart"), "bad mode")
    integer(task["event_budget"])
    require(task["event_budget"] >= 0, "negative event budget")


def evaluate(expr, state, bindings):
    if "const" in expr:
        return expr["const"]
    if "reg" in expr:
        return state[expr["reg"]]
    return bindings[expr["param"]]


def test(pred, state, bindings):
    values = []
    for atom in pred["all"]:
        left = evaluate(atom["left"], state, bindings)
        right = evaluate(atom["right"], state, bindings)
        values.append(left < right if atom["cmp"] == "lt" else left == right if atom["cmp"] == "eq" else left >= right)
    return all(values)


def initial_machine(state):
    return {"state": copy.deepcopy(state), "frames": [], "next_instance_id": 0,
            "root_pending": True, "phase": "running"}


def body_at(task, frame):
    block = task["options"][frame["option"]]["body"]
    path = frame["body_path"]
    for offset in range(0, len(path), 2):
        block = block[path[offset]][path[offset + 1]]
    return block


def normalize(machine, task):
    while machine["frames"]:
        frame = machine["frames"][-1]
        if frame["frame_kind"] != "branch" or frame["next_index"] != len(body_at(task, frame)):
            break
        machine["frames"].pop()
    return machine


def step(machine, task):
    normalize(machine, task)
    require(machine["phase"] == "running", "terminal has no next event")

    def enter(option, args, caller_bindings, call_site):
        # All operands read the same pre-call state, before initiation/push.
        bound = {parameter: evaluate(expr, machine["state"], caller_bindings)
                 for parameter, expr in args.items()}
        instance = machine["next_instance_id"]
        machine["next_instance_id"] += 1
        ok = test(task["options"][option]["initiation"], machine["state"], bound)
        event = {"event": "enter", "instance_id": instance, "option": option,
                 "bindings": copy.deepcopy(bound), "call_site": copy.deepcopy(call_site), "initiation_ok": ok}
        if ok:
            machine["frames"].append({"frame_kind": "option", "instance_id": instance,
                "option": option, "bindings": bound, "body_path": [], "next_index": 0})
        else:
            machine["phase"] = "initiation_failed"
        return event

    if machine["root_pending"]:
        machine["root_pending"] = False
        event = enter(task["entry"]["option"], task["entry"]["args"], {}, None)
    else:
        frame = machine["frames"][-1]
        block = body_at(task, frame)
        index = frame["next_index"]
        option = frame["option"]
        instance = frame["instance_id"]
        if index == len(block):
            require(frame["frame_kind"] == "option", "exhausted branches must normalize")
            ok = test(task["options"][option]["success"], machine["state"], frame["bindings"])
            machine["frames"].pop()
            normalize(machine, task)
            event = {"event": "return", "instance_id": instance, "option": option, "success_ok": ok}
            if not ok:
                machine["phase"] = "success_failed"
            elif not machine["frames"]:
                machine["phase"] = "completed"
        else:
            instruction = block[index]
            path = copy.deepcopy(frame["body_path"])
            frame["next_index"] += 1
            op = instruction["op"]
            if op == "call":
                event = enter(instruction["option"], instruction["args"], frame["bindings"],
                    {"caller_instance": instance, "body_path": path, "index": index})
            elif op == "if":
                condition = test(instruction["test"], machine["state"], frame["bindings"])
                machine["frames"].append({"frame_kind": "branch", "instance_id": instance,
                    "option": option, "bindings": copy.deepcopy(frame["bindings"]),
                    "body_path": path + [index, "then" if condition else "else"], "next_index": 0})
                event = {"event": "branch", "instance_id": instance, "option": option,
                         "body_path": path, "index": index, "condition": condition}
            else:
                event = {"event": "effect", "instance_id": instance, "option": option,
                         "body_path": path, "index": index, "op": op}
                if op == "swap":
                    machine["state"]["x"], machine["state"]["y"] = machine["state"]["y"], machine["state"]["x"]
                else:
                    value = evaluate(instruction["value"], machine["state"], frame["bindings"])
                    register = instruction["register"]
                    if op == "add":
                        machine["state"][register] += value
                    else:
                        machine["state"][register] *= value
                    event.update({"register": register, "value": value})
                event["state"] = copy.deepcopy(machine["state"])
    normalize(machine, task)
    return event


def invalid(code):
    return {"status": "invalid", "error": code, "checkpoint": None, "new_events": [],
            "previous_effects": None, "new_effects": 0, "next_event": None}


def solve(task):
    try:
        validate_task(task)
    except (ValueError, TypeError, KeyError):
        return invalid("invalid_task")
    machine = initial_machine(task["initial_state"])
    if type(task["history"]) is not list:
        return invalid("invalid_history")
    for supplied in task["history"]:
        if machine["phase"] != "running":
            return invalid("invalid_history")
        expected = step(machine, task)
        if not typed_equal(supplied, expected):
            return invalid("invalid_history")
    normalize(machine, task)
    if task["mode"] == "fresh":
        if task["history"] or task["checkpoint"] is not None:
            return invalid("invalid_checkpoint")
    elif not typed_equal(task["checkpoint"], machine):
        return invalid("invalid_checkpoint")
    if task["mode"] == "restart":
        machine = initial_machine(machine["state"])
    events = []
    for _ in range(task["event_budget"]):
        if machine["phase"] != "running":
            break
        events.append(step(machine, task))
    normalize(machine, task)
    terminal = machine["phase"] != "running"
    upcoming = None if terminal else step(copy.deepcopy(machine), task)
    return {"status": machine["phase"] if terminal else "paused", "error": None,
            "checkpoint": copy.deepcopy(machine), "new_events": events,
            "previous_effects": sum(event["event"] == "effect" for event in task["history"]),
            "new_effects": sum(event["event"] == "effect" for event in events), "next_event": upcoming}


def strict_json(text):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key")
            output[key] = value
        return output

    def no_constant(value):
        raise ValueError("nonfinite JSON number " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=no_constant)


def parse_prompt(rendered_input, semantic_task, contract_text):
    require(type(rendered_input) is str, "prompt must be string")
    marker = "Task JSON:\n"
    require(rendered_input.count(marker) == 1, "one actual task marker required")
    lead, body = rendered_input.split(marker)
    require(contract_text.strip() in lead, "full public task contract absent")
    # Public metadata requires these private coordinates/claims outside input.
    # Check labelled hints outside the contract, not ordinary contract prose.
    outside_contract = lead.replace(contract_text.strip(), "")
    private = ("controlled_context", "difficulty", "structural_difficulty", "verified_target",
               "expected_status", "source_family_key", "source_family_index", "split",
               "property_witnesses", "family_control_claims", "renderer")
    require(not re.search(r"(?i)(?<![\w])(?:" + "|".join(private) + r")[\"']?\s*[:=]", outside_contract),
            "private labelled input hint")
    parsed = strict_json(body)
    require(typed_equal(parsed, semantic_task), "visible task differs from semantic task")
    return parsed
