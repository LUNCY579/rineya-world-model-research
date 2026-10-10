"""Own source-frozen hand calibration and meaningful option-resume probes.

No imports of a production implementation. A separate recursive fresh-run
simulator is authored here, sharing no expression/step/frame helper with the
stack solver. Both remain same-author/common-model evidence, not independent
model or corpus approval. Handmade targets/snapshots precede both executions.
"""
import collections
import copy
import json


class CalibrationFailure(AssertionError):
    def __init__(self, name, completed, evidence):
        super().__init__(name)
        self.completed_checks = copy.deepcopy(completed)
        self.evidence = evidence


def recursive_fresh_trace(task):
    state = copy.deepcopy(task["initial_state"])
    events, counter = [], 0

    class End(Exception):
        def __init__(self, phase):
            self.phase = phase

    def number(expr, env):
        form = next(iter(expr))
        if form == "const":
            return expr[form]
        if form == "param":
            return env[expr[form]]
        return state[expr[form]]

    def guard(pred, env):
        for atom in pred["all"]:
            a, b = number(atom["left"], env), number(atom["right"], env)
            if atom["cmp"] == "eq" and a != b:
                return False
            if atom["cmp"] == "ge" and a < b:
                return False
            if atom["cmp"] == "lt" and a >= b:
                return False
        return True

    def invoke(option, bound, site):
        nonlocal counter
        instance = counter
        counter += 1
        definition = task["options"][option]
        ok = guard(definition["initiation"], bound)
        events.append({"event": "enter", "instance_id": instance, "option": option,
                       "bindings": copy.deepcopy(bound), "call_site": copy.deepcopy(site), "initiation_ok": ok})
        if not ok:
            raise End("initiation_failed")

        def walk(block, path):
            for index, instruction in enumerate(block):
                op = instruction["op"]
                if op == "call":
                    child_args = {p: number(e, bound) for p, e in instruction["args"].items()}
                    invoke(instruction["option"], child_args,
                           {"caller_instance": instance, "body_path": path, "index": index})
                elif op == "if":
                    condition = guard(instruction["test"], bound)
                    events.append({"event": "branch", "instance_id": instance, "option": option,
                                   "body_path": copy.deepcopy(path), "index": index, "condition": condition})
                    chosen = "then" if condition else "else"
                    walk(instruction[chosen], path + [index, chosen])
                else:
                    event = {"event": "effect", "instance_id": instance, "option": option,
                             "body_path": copy.deepcopy(path), "index": index, "op": op}
                    if op == "swap":
                        x_before, y_before = state["x"], state["y"]
                        state.update(x=y_before, y=x_before)
                    else:
                        reg, operand = instruction["register"], number(instruction["value"], bound)
                        state[reg] = state[reg] + operand if op == "add" else state[reg] * operand
                        event.update(register=reg, value=operand)
                    event["state"] = copy.deepcopy(state)
                    events.append(event)
        walk(definition["body"], [])
        ok = guard(definition["success"], bound)
        events.append({"event": "return", "instance_id": instance, "option": option, "success_ok": ok})
        if not ok:
            raise End("success_failed")

    phase = "completed"
    try:
        args = {p: number(expr, {}) for p, expr in task["entry"]["args"].items()}
        invoke(task["entry"]["option"], args, None)
    except End as failure:
        phase = failure.phase
    return {"phase": phase, "state": state, "events": events}


def reversed_fields(value):
    if type(value) is dict:
        return {key: reversed_fields(value[key]) for key in reversed(list(value))}
    if type(value) is list:
        return [reversed_fields(item) for item in value]
    return value


def alpha(value, option_map, parameter_map):
    if type(value) is list:
        return [alpha(item, option_map, parameter_map) for item in value]
    if type(value) is not dict:
        return value
    output = {}
    for key, item in value.items():
        if key == "options":
            output[key] = {option_map[o]: alpha(d, option_map, parameter_map) for o, d in item.items()}
        elif key in ("state", "initial_state"):
            output[key] = {"y" if k == "x" else "x": v for k, v in item.items()}
        elif key in ("args", "bindings"):
            output[key] = {parameter_map[p]: alpha(v, option_map, parameter_map) for p, v in item.items()}
        elif key == "params":
            output[key] = [parameter_map[p] for p in item]
        elif key == "option":
            output[key] = option_map[item]
        elif key in ("reg", "register"):
            output[key] = "y" if item == "x" else "x"
        elif key == "param":
            output[key] = parameter_map[item]
        else:
            output[key] = alpha(item, option_map, parameter_map)
    return output


def run(solver, fixtures, contract_text):
    checks, outcomes = [], {}

    def check(name, condition, evidence=None):
        if not condition:
            raise CalibrationFailure(name, checks, evidence)
        checks.append({"name": name, "passed": True, "evidence": evidence})

    for case in fixtures["cases"]:
        actual = solver.solve(case["task"])
        check("hand full typed target: " + case["name"], solver.typed_equal(actual, case["expected"]))
        outcomes[case["name"]] = actual
        if case["task"]["mode"] == "fresh" and case["expected"]["status"] != "invalid":
            simulated = recursive_fresh_trace(case["task"])
            if case["expected"]["status"] != "paused":
                check("separate recursive terminal trace: " + case["name"],
                    solver.typed_equal(simulated["events"], actual["new_events"]) and
                    solver.typed_equal(simulated["state"], actual["checkpoint"]["state"]) and
                    simulated["phase"] == actual["status"])
            else:
                check("separate recursive prefix trace: " + case["name"],
                    solver.typed_equal(simulated["events"][:len(actual["new_events"])], actual["new_events"]))
        changed = reversed_fields(case["task"])
        check("recursive field-order invariant: " + case["name"], solver.typed_equal(solver.solve(changed), actual))
        prompt = contract_text + "\nTask JSON:\n" + json.dumps(changed, ensure_ascii=False)
        parsed = solver.parse_prompt(prompt, case["task"], contract_text)
        check("visible full-contract parser field order: " + case["name"], solver.typed_equal(parsed, case["task"]))
        opts = {o: "opt_" + str(i) for i, o in enumerate(case["task"]["options"])}
        names = sorted({p for option in case["task"]["options"].values() for p in option["params"]})
        params = {p: "arg_" + str(i) for i, p in enumerate(names)}
        transformed = alpha(case["task"], opts, params)
        expected = alpha(case["expected"], opts, params)
        check("option/parameter/register alpha equivariance: " + case["name"],
              solver.typed_equal(solver.solve(transformed), expected))

    cases = {case["name"]: case for case in fixtures["cases"]}
    base = cases["handcrafted-nested-F"]["task"]
    resume = cases["handcrafted-nested-R"]["task"]
    full_trace = fixtures["nested_full_manual_events"]
    snapshots = fixtures["nested_all_manual_snapshots"]
    check("ten handmade snapshots include every full-trace event boundary", len(snapshots) == len(full_trace) + 1 == 10)
    for cut in range(10):
        task = copy.deepcopy(base)
        task["event_budget"] = cut
        fresh = solver.solve(task)
        terminal = cut == len(full_trace)
        expected = {"status": "completed" if terminal else "paused", "error": None,
                    "checkpoint": snapshots[cut], "new_events": full_trace[:cut],
                    "previous_effects": 0, "new_effects": sum(e["event"] == "effect" for e in full_trace[:cut]),
                    "next_event": None if terminal else full_trace[cut]}
        check("hand boundary and noncommitting peek " + str(cut), solver.typed_equal(fresh, expected))
        for budget in range(12):
            task = copy.deepcopy(base)
            task.update(mode="resume", history=copy.deepcopy(full_trace[:cut]),
                        checkpoint=copy.deepcopy(snapshots[cut]), event_budget=budget)
            stop = min(cut + budget, len(full_trace))
            expected = {"status": "completed" if stop == len(full_trace) else "paused", "error": None,
                        "checkpoint": snapshots[stop], "new_events": full_trace[cut:stop],
                        "previous_effects": sum(e["event"] == "effect" for e in full_trace[:cut]),
                        "new_effects": sum(e["event"] == "effect" for e in full_trace[cut:stop]),
                        "next_event": None if stop == len(full_trace) else full_trace[stop]}
            check("manual prefix/suffix all cuts/budgets " + str(cut) + "/" + str(budget),
                  solver.typed_equal(solver.solve(task), expected))
        restart = copy.deepcopy(base)
        restart.update(mode="restart", history=copy.deepcopy(full_trace[:cut]), checkpoint=copy.deepcopy(snapshots[cut]))
        task_for_second_simulator = copy.deepcopy(base)
        task_for_second_simulator["initial_state"] = copy.deepcopy(snapshots[cut]["state"])
        simulated = recursive_fresh_trace(task_for_second_simulator)
        actual = solver.solve(restart)
        check("restart independently simulated from checkpoint state " + str(cut),
              solver.typed_equal(actual["new_events"], simulated["events"]) and
              solver.typed_equal(actual["checkpoint"]["state"], simulated["state"]) and actual["status"] == simulated["phase"])

    checkpoint = resume["checkpoint"]
    child, branch = checkpoint["frames"][1:]
    definition = base["options"]["inner"]
    check("retained child previously initiated but current initiation false",
          full_trace[1]["initiation_ok"] is True and
          solver.test(definition["initiation"], checkpoint["state"], child["bindings"]) is False)
    check("selected branch guard becomes false while its pending body is retained",
          full_trace[2]["condition"] is True and
          solver.test(definition["body"][0]["test"], checkpoint["state"], branch["bindings"]) is False and
          outcomes["handcrafted-nested-R"]["new_events"][0]["value"] == 2)
    check("R retains two owning option frames and a selected branch with pending caller",
          [frame["frame_kind"] for frame in checkpoint["frames"]] == ["option", "option", "branch"] and
          checkpoint["frames"][0]["next_index"] == 1 and child["next_index"] == 1 and branch["next_index"] == 1)
    f, r, t, p, u, a, i, s = [outcomes["handcrafted-nested-" + c] for c in "FRTPUAIS"]
    check("F/R full terminal equality and exact suffix", solver.typed_equal(f["checkpoint"], r["checkpoint"]) and solver.typed_equal(r["new_events"], f["new_events"][4:]))
    check("T root reenters successfully and repeats actually committed instruction",
          t["new_events"][0]["initiation_ok"] is True and
          full_trace[3]["option"] == t["new_events"][3]["option"] and
          full_trace[3]["body_path"] == t["new_events"][3]["body_path"] and
          full_trace[3]["index"] == t["new_events"][3]["index"] and
          full_trace[3]["value"] == t["new_events"][3]["value"] == 1 and
          not solver.typed_equal(t["checkpoint"]["state"], r["checkpoint"]["state"]))
    check("replayed primitive g(g(s)) differs from g(s)", {"x": 4, "y": 0} != {"x": 3, "y": 0},
          {"pre_state": {"x": 2, "y": 0}, "once": {"x": 3, "y": 0}, "twice": {"x": 4, "y": 0}, "primitive": "add x 1"})
    check("P/U second actual checkpoint after effect and pending work", p["status"] == "paused" and p["new_effects"] == 1 and
          p["next_event"]["event"] == "effect" and solver.typed_equal(p["new_events"], r["new_events"][:1]) and
          solver.typed_equal(u["new_events"], r["new_events"][1:]) and solver.typed_equal(u["checkpoint"], f["checkpoint"]))
    check("A used root parameter changes later actual effect while both complete", a["status"] == f["status"] == "completed" and
          f["new_events"][4]["value"] == 2 and a["new_events"][4]["value"] == 3 and not solver.typed_equal(a["checkpoint"]["state"], f["checkpoint"]["state"]))
    check("I only decisive initiation atom false and no callee effects", i["status"] == "initiation_failed" and i["new_effects"] == 0 and
          solver.test({"all": [base["options"]["outer"]["initiation"]["all"][1]]}, base["initial_state"], {"p": 0}) is True)
    effect_events = lambda output: [event for event in output["new_events"] if event["event"] == "effect"]
    check("S effects committed unchanged and only exit success differs", s["status"] == "success_failed" and
          solver.typed_equal(effect_events(s), effect_events(f)) and solver.typed_equal(s["checkpoint"]["state"], f["checkpoint"]["state"]))

    for kind in ("state", "cursor", "binding"):
        task = copy.deepcopy(resume)
        if kind == "state":
            task["checkpoint"]["state"]["x"] += 1
        elif kind == "cursor":
            task["checkpoint"]["frames"][2]["next_index"] += 1
        else:
            task["checkpoint"]["frames"][2]["bindings"]["k"] += 1
        check("X active " + kind + " exact history rejects before new event", solver.typed_equal(solver.solve(task), solver.invalid("invalid_checkpoint")))
        diagnostic = copy.deepcopy(task["checkpoint"])
        diagnostic_events = []
        while diagnostic["phase"] == "running":
            diagnostic_events.append(solver.step(diagnostic, base))
        check("X trusted-machine diagnostic changes observable pending continuation " + kind,
              not solver.typed_equal(diagnostic["state"], r["checkpoint"]["state"]),
              {"diagnostic_only_not_valid_input": True, "changed_state": diagnostic["state"], "new_events": diagnostic_events})

    mutations = []
    def malformed(name, code, change, origin=base):
        task = copy.deepcopy(origin)
        change(task)
        mutations.append((name, code, task))
    malformed("extra envelope field", "invalid_task", lambda t: t.update(extra=1))
    malformed("missing envelope field", "invalid_task", lambda t: t.pop("mode"))
    malformed("boolean version", "invalid_task", lambda t: t.update(version=True))
    malformed("boolean budget", "invalid_task", lambda t: t.update(event_budget=False))
    malformed("negative budget", "invalid_task", lambda t: t.update(event_budget=-1))
    malformed("boolean register", "invalid_task", lambda t: t["initial_state"].update(x=True))
    malformed("float register", "invalid_task", lambda t: t["initial_state"].update(y=0.0))
    malformed("boolean constant", "invalid_task", lambda t: t["entry"]["args"].update(p={"const": True}))
    malformed("root parameter reference", "invalid_task", lambda t: t["entry"]["args"].update(p={"param": "p"}))
    malformed("unknown called argument", "invalid_task", lambda t: t["options"]["outer"]["body"][0]["args"].update(extra={"const": 1}))
    malformed("unreachable unknown parameter", "invalid_task", lambda t: t["options"]["inner"]["body"][0]["else"][0].update(value={"param": "absent"}))
    malformed("duplicate parameter", "invalid_task", lambda t: t["options"]["inner"]["params"].append("k"))
    malformed("unknown comparison", "invalid_task", lambda t: t["options"]["outer"]["initiation"]["all"][0].update(cmp="le"))
    malformed("malformed predicate atoms", "invalid_task", lambda t: t["options"]["outer"]["success"].update(all={}))
    def add_unused_cycle(task):
        task["options"]["unused"] = {"params": [], "initiation": {"all": []}, "success": {"all": []},
            "body": [{"op": "if", "test": {"all": []}, "then": [], "else": [{"op": "call", "option": "unused", "args": {}}]}]}
    malformed("unused recursive call in untaken branch", "invalid_task", add_unused_cycle)
    malformed("history non-list", "invalid_history", lambda t: t.update(history={}))
    malformed("false committed effect state", "invalid_history", lambda t: t["history"][3]["state"].update(x=4), resume)
    malformed("history extra event field", "invalid_history", lambda t: t["history"][0].update(extra=0), resume)
    malformed("typed branch condition integer", "invalid_history", lambda t: t["history"][2].update(condition=1), resume)
    malformed("typed history instance bool", "invalid_history", lambda t: t["history"][1].update(instance_id=True), resume)
    malformed("invalid task precedes bad history/checkpoint", "invalid_task", lambda t: t.update(version=True, history={}, checkpoint={}))
    malformed("invalid history precedes bad checkpoint", "invalid_history", lambda t: t.update(history=[{}], checkpoint={}))
    malformed("valid nonempty fresh history forbidden", "invalid_checkpoint", lambda t: t.update(mode="fresh", checkpoint=None), resume)
    malformed("nonnull fresh checkpoint forbidden", "invalid_checkpoint", lambda t: t.update(checkpoint=copy.deepcopy(snapshots[0])))
    malformed("checkpoint missing", "invalid_checkpoint", lambda t: t.update(checkpoint=None), resume)
    malformed("typed checkpoint cursor boolean", "invalid_checkpoint", lambda t: t["checkpoint"]["frames"][2].update(next_index=True), resume)
    malformed("wrong invocation counter", "invalid_checkpoint", lambda t: t["checkpoint"].update(next_instance_id=9), resume)
    paused = cases["handcrafted-nested-P"]["expected"]["checkpoint"]
    after_first = copy.deepcopy(resume)
    after_first.update(history=copy.deepcopy(full_trace[:5]), checkpoint=copy.deepcopy(paused))
    def exhausted_branch(task):
        task["checkpoint"]["frames"].append({"frame_kind": "branch", "instance_id": 1,
            "option": "inner", "bindings": {"k": 2}, "body_path": [0, "then"], "next_index": 2})
    malformed("unnormalized exhausted branch is not repaired", "invalid_checkpoint", exhausted_branch, after_first)
    terminal = copy.deepcopy(base)
    terminal.update(mode="resume", history=copy.deepcopy(full_trace), checkpoint=copy.deepcopy(snapshots[-1]))
    malformed("event after terminal history", "invalid_history", lambda t: t["history"].append(copy.deepcopy(full_trace[-1])), terminal)
    for label, code, task in mutations:
        check("typed malformed precedence: " + label, solver.typed_equal(solver.solve(task), solver.invalid(code)))

    failed = cases["handcrafted-nested-S"]
    continuation = copy.deepcopy(failed["task"])
    continuation.update(mode="resume", history=copy.deepcopy(failed["expected"]["new_events"]),
                        checkpoint=copy.deepcopy(failed["expected"]["checkpoint"]), event_budget=0)
    check("terminal failure resumes with no new event", solver.typed_equal(solver.solve(continuation),
          {"status": "success_failed", "error": None, "checkpoint": failed["expected"]["checkpoint"],
           "new_events": [], "previous_effects": 4, "new_effects": 0, "next_event": None}))
    continuation["mode"] = "restart"
    continuation["event_budget"] = 48
    restarted = solver.solve(continuation)
    check("failed checkpoint may restart with fresh invocation IDs", restarted["new_events"][0]["instance_id"] == 0 and restarted["new_events"][0]["initiation_ok"] is True and restarted["status"] == "completed")

    # Success is checked on exhaustion, not when an intermediate state satisfies it.
    early = copy.deepcopy(base)
    early["options"]["outer"]["success"] = {"all": []}
    check("already-true success does not skip the body", solver.solve(early)["new_effects"] == 4)

    def must_reject(name, fn):
        try:
            fn()
        except (ValueError, json.JSONDecodeError):
            check(name, True)
            return
        raise AssertionError(name + " was accepted")
    must_reject("parser duplicate object keys", lambda: solver.strict_json('{"version":1,"version":1}'))
    must_reject("parser nonfinite numeric scalar", lambda: solver.strict_json('{"x":NaN}'))
    must_reject("prompt without full contract", lambda: solver.parse_prompt("Task JSON:\n" + json.dumps(base), base, contract_text))
    must_reject("prompt/semantic task mismatch", lambda: solver.parse_prompt(contract_text + "\nTask JSON:\n" + json.dumps(cases["handcrafted-nested-A"]["task"]), base, contract_text))
    for label in ("controlled_context: R", "Difficulty = 4", '"verified_target": {}', "source_family_index: 12"):
        must_reject("private labelled prompt hint " + label, lambda label=label:
            solver.parse_prompt(contract_text + "\n" + label + "\nTask JSON:\n" + json.dumps(base), base, contract_text))
    check("typed equality separates bool and integer", not solver.typed_equal(True, 1) and not solver.typed_equal(False, 0))
    check("typed equality separates float and integer", not solver.typed_equal(1.0, 1))
    check("ordered lists remain semantic", not solver.typed_equal([1, 2], [2, 1]))
    counts = collections.Counter(name["name"].split(":")[0] for name in checks)
    return {"checks": checks, "passed": len(checks), "failed": 0, "handmade_full_target_cases": len(fixtures["cases"]),
            "manual_event_cuts": len(snapshots), "all_cut_budget_resume_checks": 120,
            "restart_recursive_crosschecks": 10, "malformed_stage_checks": len(mutations),
            "handmade_outcomes": outcomes, "check_name_prefix_counts": dict(counts),
            "feasibility_scope": "One handmade nested public example satisfies all nine control semantics; no claim of12 novel future programs or cohort metadata/novelty approval",
            "implementations": "Fresh stack solver plus separately authored recursive complete-run simulator; same author/common-model, independent of producer/formal code"}
