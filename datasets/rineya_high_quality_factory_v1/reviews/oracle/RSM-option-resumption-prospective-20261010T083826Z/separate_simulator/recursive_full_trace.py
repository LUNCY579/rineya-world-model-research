"""NONBINDING recursive full-event simulation of valid RSM v1 programs.

Authored independently from RSM_OPTION_RESUMPTION_V1.md only by
/root/formal_oracle/rsm_recursive_simulator on 2026-10-10. This role does not
approve a release or supply binding adjudication. This module does not validate
task envelopes, history, checkpoints, modes, or budgets. ``full_trace`` starts
a new complete execution from the initial state (or the supplied start_state).
It deliberately uses recursive instruction-list and option traversal, with no
explicit runtime frame stack or resumable interpreter.
"""

SPEC_SHA256 = "44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f"
AUTHOR_IDENTITY = "/root/formal_oracle/rsm_recursive_simulator"
REVIEW_ROLE = "nonbinding independently authored recursive trace simulator"


def typed_equal(left, right):
    """JSON equality preserving types, object-key independence and list order."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            typed_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            typed_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class _Halt(Exception):
    """A failed entry or exit unwinds every active recursive invocation."""

    def __init__(self, phase):
        self.phase = phase


def full_trace(task, start_state=None):
    """Return exactly {phase, state, events} for a valid finite task program.

    Only options, entry and initial_state are used. ``start_state`` overrides
    initial_state for a new complete run, including fresh root bindings/guards
    and invocation numbering from zero. Every output event has contract fields
    only. Input data is never modified.
    """
    supplied_state = task["initial_state"] if start_state is None else start_state
    state = {"x": supplied_state["x"], "y": supplied_state["y"]}
    options = task["options"]
    events = []
    next_instance = 0

    def expression(expr, bindings, registers):
        if "const" in expr:
            return expr["const"]
        if "reg" in expr:
            return registers[expr["reg"]]
        return bindings[expr["param"]]

    def predicate(pred, bindings):
        # Copy once: all atoms observe the same register values.
        registers = dict(state)
        answers = []
        for atom in pred["all"]:
            lhs = expression(atom["left"], bindings, registers)
            rhs = expression(atom["right"], bindings, registers)
            comparison = atom["cmp"]
            if comparison == "lt":
                answers.append(lhs < rhs)
            elif comparison == "eq":
                answers.append(lhs == rhs)
            else:  # Valid program: ge is the only remaining comparator.
                answers.append(lhs >= rhs)
        return all(answers)

    def bind_arguments(target_name, arguments, caller_bindings):
        # Evaluate into a separate dictionary before invoking the callee.
        registers = dict(state)
        return {
            parameter: expression(arguments[parameter], caller_bindings, registers)
            for parameter in options[target_name]["params"]
        }

    def visit_list(instructions, name, instance, bindings, body_path):
        for index, instruction in enumerate(instructions):
            op = instruction["op"]
            locator = {
                "instance_id": instance,
                "option": name,
                "body_path": list(body_path),
                "index": index,
            }
            if op == "if":
                condition = predicate(instruction["test"], bindings)
                events.append({"event": "branch", **locator, "condition": condition})
                selected = "then" if condition else "else"
                visit_list(
                    instruction[selected], name, instance, bindings,
                    body_path + [index, selected],
                )
            elif op == "call":
                target = instruction["option"]
                arguments = bind_arguments(target, instruction["args"], bindings)
                site = {
                    "caller_instance": instance,
                    "body_path": list(body_path),
                    "index": index,
                }
                visit_option(target, arguments, site)
            elif op == "swap":
                state["x"], state["y"] = state["y"], state["x"]
                events.append({"event": "effect", **locator, "op": "swap", "state": dict(state)})
            else:
                register = instruction["register"]
                value = expression(instruction["value"], bindings, state)
                if op == "add":
                    state[register] = state[register] + value
                else:  # Valid program: scale is the remaining primitive.
                    state[register] = state[register] * value
                events.append({
                    "event": "effect", **locator, "op": op,
                    "register": register, "value": value, "state": dict(state),
                })

    def visit_option(name, supplied_bindings, call_site):
        nonlocal next_instance
        option = options[name]
        bindings = dict(supplied_bindings)
        instance = next_instance
        next_instance += 1
        initiation_ok = predicate(option["initiation"], bindings)
        events.append({
            "event": "enter", "instance_id": instance, "option": name,
            "bindings": dict(bindings), "call_site": call_site,
            "initiation_ok": initiation_ok,
        })
        if not initiation_ok:
            raise _Halt("initiation_failed")
        visit_list(option["body"], name, instance, bindings, [])
        success_ok = predicate(option["success"], bindings)
        events.append({
            "event": "return", "instance_id": instance, "option": name,
            "success_ok": success_ok,
        })
        if not success_ok:
            raise _Halt("success_failed")

    entry = task["entry"]
    root_name = entry["option"]
    root_bindings = bind_arguments(root_name, entry["args"], {})
    phase = "completed"
    try:
        visit_option(root_name, root_bindings, None)
    except _Halt as halt:
        phase = halt.phase
    return {"phase": phase, "state": dict(state), "events": events}
