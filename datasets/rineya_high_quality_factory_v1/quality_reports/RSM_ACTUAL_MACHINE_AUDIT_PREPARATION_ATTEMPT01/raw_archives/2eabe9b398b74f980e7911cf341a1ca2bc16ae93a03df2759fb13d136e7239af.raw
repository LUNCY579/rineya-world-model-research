"""Producer's own handmade calibration, frozen before first import/test.

No official interpreter/checker, private fixture or production records used.
The recursive trace simulator below has no stack/checkpoint implementation.
It is an internal cross-check, not an independently staffed oracle or approval.
"""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
checks = []


def check(name, condition):
    checks.append({"name": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def reversed_objects(value):
    if type(value) is dict:
        return {key: reversed_objects(value[key]) for key in reversed(list(value))}
    if type(value) is list:
        return [reversed_objects(item) for item in value]
    return value


def recursive_trace(task):
    """Separate direct-recursive fresh simulator, using its own arithmetic.

    This validates fresh traces without using reference frame/event helpers.
    It does not validate input/checkpoints and supplies no production labels.
    """
    registers = dict(task["initial_state"])
    events = []
    counter = 0

    class Halt(Exception):
        pass

    def expression(value, local):
        tag = next(iter(value))
        if tag == "reg":
            return registers[value[tag]]
        if tag == "param":
            return local[value[tag]]
        return value[tag]

    def predicate(value, local):
        for atom in value["all"]:
            left = expression(atom["left"], local)
            right = expression(atom["right"], local)
            if atom["cmp"] == "lt" and not left < right:
                return False
            if atom["cmp"] == "eq" and not left == right:
                return False
            if atom["cmp"] == "ge" and not left >= right:
                return False
        return True

    def instructions(body, name, iid, local, path):
        for index, instruction in enumerate(body):
            op = instruction["op"]
            if op == "call":
                actuals = {key: expression(value, local) for key, value in instruction["args"].items()}
                invoke(instruction["option"], actuals,
                       {"caller_instance": iid, "body_path": list(path), "index": index})
            elif op == "if":
                selected = predicate(instruction["test"], local)
                events.append({"event": "branch", "instance_id": iid, "option": name,
                               "body_path": list(path), "index": index, "condition": selected})
                branch = "then" if selected else "else"
                instructions(instruction[branch], name, iid, local, path + [index, branch])
            else:
                event = {"event": "effect", "instance_id": iid, "option": name,
                         "body_path": list(path), "index": index, "op": op}
                if op == "swap":
                    old_x, old_y = registers["x"], registers["y"]
                    registers["x"], registers["y"] = old_y, old_x
                else:
                    operand = expression(instruction["value"], local)
                    register = instruction["register"]
                    before = registers[register]
                    registers[register] = before + operand if op == "add" else before * operand
                    event.update(register=register, value=operand)
                event["state"] = dict(registers)
                events.append(event)

    def invoke(name, local, site):
        nonlocal counter
        iid, counter = counter, counter + 1
        option = task["options"][name]
        enters = predicate(option["initiation"], local)
        events.append({"event": "enter", "instance_id": iid, "option": name,
                       "bindings": dict(local), "call_site": site, "initiation_ok": enters})
        if not enters:
            raise Halt("initiation_failed")
        instructions(option["body"], name, iid, local, [])
        succeeds = predicate(option["success"], local)
        events.append({"event": "return", "instance_id": iid, "option": name, "success_ok": succeeds})
        if not succeeds:
            raise Halt("success_failed")

    entry = task["entry"]
    try:
        invoke(entry["option"], {key: expression(value, {}) for key, value in entry["args"].items()}, None)
        phase = "completed"
    except Halt as failure:
        phase = str(failure)
    return {"status": phase, "state": registers, "events": events}


def main():
    spec = importlib.util.spec_from_file_location("producer_rsm_reference_attempt01", HERE / "reference_attempt01.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve, same = module.solve, module.typed_equal
    fixture = json.loads((HERE / "hand_fixtures_attempt01.json").read_text())
    F = fixture["task"]
    full = fixture["manually_enumerated_full_events"]
    complete = solve(F)
    check("manual_full_event_trace_19_events", same(complete["new_events"], full))
    check("manual_full_state_and_terminal_snapshot", same(complete["checkpoint"],
          {"state": {"x": 15, "y": 5}, "frames": [], "next_instance_id": 4,
           "root_pending": False, "phase": "completed"}))
    check("manual_complete_typed_output_counts", complete["status"] == "completed" and complete["error"] is None
          and complete["previous_effects"] == 0 and complete["new_effects"] == 9 and complete["next_event"] is None)
    direct = recursive_trace(F)
    check("direct_recursive_simulator_matches_manual_trace", same(direct["events"], full) and direct["state"] == {"x": 15, "y": 5})
    R = deepcopy(F)
    R.update(mode="resume", history=deepcopy(full[:5]), checkpoint=deepcopy(fixture["manually_enumerated_first_checkpoint"]))
    resumed = solve(R)
    check("manual_first_checkpoint_and_resume_suffix", same(resumed["new_events"], full[5:]) and same(resumed["checkpoint"], complete["checkpoint"]))
    check("resume_previous_new_effect_accounting", resumed["previous_effects"] == 2 and resumed["new_effects"] == 7)
    check("retained_child_initiation_now_false", R["checkpoint"]["state"]["x"] != R["checkpoint"]["frames"][1]["bindings"]["r"])
    check("retained_selected_branch_guard_now_false", R["checkpoint"]["state"]["x"] != R["checkpoint"]["frames"][2]["bindings"]["r"])
    check("retained_caller_cursor_advanced_before_call", R["checkpoint"]["frames"][0]["next_index"] == 2)
    check("retained_bindings_consumed_without_reentry_or_branch_retest", resumed["new_events"][0]["value"] == 2
          and resumed["new_events"][0]["state"] == {"x": 3, "y": 4}
          and all(not (event["event"] == "enter" and event["option"] == "work") for event in resumed["new_events"]))
    T = deepcopy(R)
    T["mode"] = "restart"
    restarted = solve(T)
    check("restart_original_checkpoint_validated_then_root_rebound", restarted["new_events"][0]["bindings"] == {"p": 3}
          and restarted["new_events"][0]["initiation_ok"] is True and restarted["new_events"][0]["instance_id"] == 0)
    check("restart_nonidempotent_committed_effect_actually_replayed", restarted["new_events"][1]["option"] == full[1]["option"]
          and restarted["new_events"][1]["body_path"] == full[1]["body_path"]
          and restarted["new_events"][1]["index"] == full[1]["index"]
          and restarted["new_events"][1]["state"] == {"x": 4, "y": 2} and (1 + 1) + 1 != 1 + 1)
    check("restart_manual_terminal_difference", restarted["status"] == "completed"
          and restarted["checkpoint"]["state"] == {"x": 53, "y": 63}
          and not same(restarted["new_events"], resumed["new_events"]))
    P = deepcopy(R)
    P["event_budget"] = 1
    paused = solve(P)
    manual_P = deepcopy(R["checkpoint"])
    manual_P["state"]["y"] = 4
    manual_P["frames"][2]["next_index"] = 2
    check("short_budget_pauses_after_new_effect_at_exact_second_checkpoint", paused["status"] == "paused"
          and paused["new_effects"] == 1 and same(paused["checkpoint"], manual_P)
          and same(paused["new_events"], full[5:6]) and same(paused["next_event"], full[6]))
    U = deepcopy(R)
    U.update(history=deepcopy(full[:6]), checkpoint=deepcopy(manual_P))
    twice_resumed = solve(U)
    check("second_checkpoint_suffix_and_completion", same(twice_resumed["new_events"], full[6:])
          and same(twice_resumed["checkpoint"], complete["checkpoint"]))
    A = deepcopy(F)
    A["entry"]["args"]["p"] = {"const": 2}
    argument_changed = solve(A)
    check("positive_argument_counterfactual_used_later", argument_changed["status"] == "completed"
          and argument_changed["checkpoint"]["state"] == {"x": 26, "y": 20}
          and argument_changed["new_events"][4]["value"] == 2 and full[4]["value"] == 1)
    I = deepcopy(F)
    I["entry"]["args"]["p"] = {"const": 0}
    initiation_failed = solve(I)
    check("one_decisive_initiation_atom_other_true_zero_effects", initiation_failed["status"] == "initiation_failed"
          and initiation_failed["new_effects"] == 0 and len(initiation_failed["new_events"]) == 1
          and initiation_failed["checkpoint"]["next_instance_id"] == 1
          and initiation_failed["checkpoint"]["state"] == F["initial_state"])
    S = deepcopy(F)
    S["options"]["start"]["success"]["all"][0]["right"]["const"] = 100000
    success_failed = solve(S)
    check("root_success_literal_same_effects_no_rollback", success_failed["status"] == "success_failed"
          and success_failed["checkpoint"]["state"] == {"x": 15, "y": 5}
          and same(success_failed["new_events"][:-1], full[:-1])
          and success_failed["new_events"][-1]["success_ok"] is False)
    X = deepcopy(R)
    X["checkpoint"]["frames"][2]["bindings"]["r"] = 3
    invalid = solve(X)
    check("active_saved_binding_tamper_exact_checkpoint_rejected_before_events", invalid == {
        "status": "invalid", "error": "invalid_checkpoint", "checkpoint": None, "new_events": [],
        "previous_effects": None, "new_effects": 0, "next_event": None})
    modified_trust = module.trusted_machine_probe(R, X["checkpoint"], 1)
    check("saved_binding_tamper_consequential_in_separate_trusted_probe", modified_trust["events"][0]["value"] == 3
          and modified_trust["events"][0]["state"] == {"x": 3, "y": 5})
    for context_name, task in [("F", F), ("R", R), ("T", T), ("P", P), ("U", U), ("A", A), ("I", I), ("S", S), ("X", X)]:
        check("recursive_object_order_" + context_name, same(solve(reversed_objects(task)), solve(task)))
    for cut in range(len(full) + 1):
        fresh = deepcopy(F)
        fresh["event_budget"] = cut
        prefix = solve(fresh)
        check(f"fresh_exact_budget_boundary_{cut}", same(prefix["new_events"], full[:cut])
              and prefix["status"] == ("completed" if cut == len(full) else "paused")
              and same(prefix["next_event"], None if cut == len(full) else full[cut]))
        continuation = deepcopy(F)
        continuation.update(mode="resume", history=deepcopy(full[:cut]), checkpoint=deepcopy(prefix["checkpoint"]))
        for budget in range(len(full) - cut + 2):
            continuation["event_budget"] = budget
            result = solve(continuation)
            stop = min(len(full), cut + budget)
            check(f"resume_cut_{cut}_budget_{budget}", same(result["new_events"], full[cut:stop])
                  and result["status"] == ("completed" if stop == len(full) else "paused")
                  and same(result["next_event"], None if stop == len(full) else full[stop])
                  and result["previous_effects"] == sum(event["event"] == "effect" for event in full[:cut]))
    # Own direct-recursive simulator covers successful and failed fresh modes.
    for name, task in [("F", F), ("A", A), ("I", I), ("S", S)]:
        direct = recursive_trace(task)
        actual = solve(task)
        check("direct_simulator_" + name, direct["status"] == actual["status"]
              and same(direct["events"], actual["new_events"])
              and same(direct["state"], actual["checkpoint"]["state"]))

    def bad(name, task, stage):
        result = solve(task)
        check("malformed_" + name, result["status"] == "invalid" and result["error"] == stage
              and result["checkpoint"] is None and not result["new_events"] and result["previous_effects"] is None)

    mutations = []
    def mutation(name, stage, edit, origin=F):
        value = deepcopy(origin)
        edit(value)
        mutations.append((name, value, stage))

    mutation("boolean_version", "invalid_task", lambda t: t.update(version=True))
    mutation("boolean_initial_state", "invalid_task", lambda t: t["initial_state"].update(x=True))
    mutation("boolean_budget", "invalid_task", lambda t: t.update(event_budget=False))
    mutation("negative_budget", "invalid_task", lambda t: t.update(event_budget=-1))
    mutation("float_constant", "invalid_task", lambda t: t["options"]["start"]["body"][0].update(value={"const": 1.0}))
    mutation("bool_constant", "invalid_task", lambda t: t["options"]["start"]["body"][0].update(value={"const": True}))
    mutation("extra_envelope_field", "invalid_task", lambda t: t.update(private_hint="completed"))
    mutation("missing_envelope_field", "invalid_task", lambda t: t.pop("entry"))
    mutation("unknown_mode", "invalid_task", lambda t: t.update(mode="repair"))
    mutation("missing_call_arg", "invalid_task", lambda t: t["options"]["start"]["body"][1]["args"].pop("r"))
    mutation("extra_call_arg", "invalid_task", lambda t: t["options"]["start"]["body"][1]["args"].update(extra={"const": 2}))
    mutation("root_param_reference", "invalid_task", lambda t: t["entry"]["args"].update(p={"param": "p"}))
    mutation("unknown_parameter_in_unselected_branch", "invalid_task", lambda t: t["options"]["work"]["body"][0]["else"][0].update(value={"param": "z"}))
    mutation("duplicate_params", "invalid_task", lambda t: t["options"]["work"].update(params=["q", "q"]))
    mutation("unknown_instruction_field", "invalid_task", lambda t: t["options"]["finish"]["body"][0].update(cursor=0))
    mutation("cycle_in_unselected_branch", "invalid_task", lambda t: t["options"]["work"]["body"][0]["else"].append({"op": "call", "option": "start", "args": {"p": {"const": 1}}}))
    mutation("cycle_in_unused_option", "invalid_task", lambda t: t["options"].update(unused={"params": [], "initiation": {"all": []}, "success": {"all": []}, "body": [{"op": "call", "option": "unused", "args": {}}]}))
    mutation("history_not_list", "invalid_history", lambda t: t.update(history={}))
    mutation("false_history_state", "invalid_history", lambda t: t["history"][1]["state"].update(x=999), R)
    mutation("bool_history_instance", "invalid_history", lambda t: t["history"][0].update(instance_id=False), R)
    mutation("extra_history_event_field", "invalid_history", lambda t: t["history"][0].update(outcome="good"), R)
    mutation("reordered_history_events", "invalid_history", lambda t: t["history"].reverse(), R)
    mutation("history_after_terminal", "invalid_history", lambda t: t.update(history=deepcopy(full + full[:1])))
    mutation("fresh_correct_history_forbidden", "invalid_checkpoint", lambda t: t.update(history=deepcopy(full[:1])))
    mutation("fresh_nonnull_checkpoint", "invalid_checkpoint", lambda t: t.update(checkpoint={}))
    mutation("resume_missing_checkpoint", "invalid_checkpoint", lambda t: t.update(checkpoint=None), R)
    mutation("checkpoint_bool_counter", "invalid_checkpoint", lambda t: t["checkpoint"].update(next_instance_id=True), R)
    mutation("checkpoint_extra_field", "invalid_checkpoint", lambda t: t["checkpoint"].update(padding=0), R)
    mutation("checkpoint_wrong_phase", "invalid_checkpoint", lambda t: t["checkpoint"].update(phase="completed"), R)
    mutation("checkpoint_wrong_branch_path", "invalid_checkpoint", lambda t: t["checkpoint"]["frames"][2].update(body_path=[0, "else"]), R)
    mutation("checkpoint_active_cursor", "invalid_checkpoint", lambda t: t["checkpoint"]["frames"][2].update(next_index=2), R)
    mutation("checkpoint_active_state", "invalid_checkpoint", lambda t: t["checkpoint"]["state"].update(x=4), R)
    mutation("stage_one_before_history_and_checkpoint", "invalid_task", lambda t: t.update(version=True, history={}, checkpoint={}))
    mutation("stage_two_before_checkpoint", "invalid_history", lambda t: t.update(history={}, checkpoint={}), R)
    mutation("restart_does_not_bypass_checkpoint_validation", "invalid_checkpoint", lambda t: t.update(mode="restart", checkpoint={}), R)
    for name, task, stage in mutations:
        bad(name, task, stage)

    def small(body, success=None, initial=None):
        return {"kind": "option_resume", "version": 1, "initial_state": initial or {"x": 1, "y": 2},
                "options": {"a": {"params": [], "initiation": {"all": []},
                                    "success": success or {"all": []}, "body": body}},
                "entry": {"option": "a", "args": {}}, "mode": "fresh", "history": [], "checkpoint": None, "event_budget": 48}

    empty = small([])
    empty_result = solve(empty)
    check("empty_body_entry_and_return_are_distinct_events", len(empty_result["new_events"]) == 2 and empty_result["status"] == "completed")
    empty["event_budget"] = 1
    empty_paused = solve(empty)
    check("empty_body_budget_one_keeps_option_return_pending", empty_paused["status"] == "paused"
          and empty_paused["checkpoint"]["frames"][0]["next_index"] == 0 and empty_paused["next_event"]["event"] == "return")
    empty_branch = small([{"op": "if", "test": {"all": []}, "then": [], "else": []}, {"op": "swap"}])
    empty_branch["event_budget"] = 2
    branch_pause = solve(empty_branch)
    check("empty_branch_normalized_without_extra_event", len(branch_pause["checkpoint"]["frames"]) == 1
          and branch_pause["next_event"]["op"] == "swap" and branch_pause["next_event"]["state"] == {"x": 2, "y": 1})
    unnormalized = deepcopy(empty_branch)
    unnormalized.update(mode="resume", history=branch_pause["new_events"], checkpoint=deepcopy(branch_pause["checkpoint"]))
    unnormalized["checkpoint"]["frames"].append({"frame_kind": "branch", "instance_id": 0,
                                               "option": "a", "bindings": {}, "body_path": [0, "then"], "next_index": 0})
    bad("unnormalized_exhausted_empty_branch", unnormalized, "invalid_checkpoint")
    not_early = small([{"op": "add", "register": "x", "value": {"const": 1}},
                       {"op": "add", "register": "x", "value": {"const": -3}}],
                      {"all": [{"left": {"reg": "x"}, "cmp": "ge", "right": {"const": 0}}]})
    no_early = solve(not_early)
    check("success_test_only_at_return_not_early_stop", no_early["status"] == "success_failed"
          and no_early["new_effects"] == 2 and no_early["checkpoint"]["state"]["x"] == -1)
    simultaneous = small([{"op": "call", "option": "b", "args": {"u": {"reg": "x"}, "v": {"reg": "y"}}}])
    simultaneous["options"]["b"] = {"params": ["u", "v"], "initiation": {"all": []}, "success": {"all": []},
                                     "body": [{"op": "add", "register": "x", "value": {"param": "v"}},
                                              {"op": "add", "register": "y", "value": {"param": "u"}}]}
    bound = solve(simultaneous)
    check("simultaneous_call_args_immutable_after_register_mutation", bound["checkpoint"]["state"] == {"x": 3, "y": 3}
          and bound["new_events"][1]["bindings"] == {"u": 1, "v": 2})
    for phase in ["initiation", "success"]:
        child_failure = deepcopy(simultaneous)
        child_failure["options"]["b"][phase] = {"all": [{"left": {"const": 0}, "cmp": "eq", "right": {"const": 1}}]}
        result = solve(child_failure)
        check("nested_" + phase + "_failure_keeps_advanced_caller", result["status"] == phase + "_failed"
              and len(result["checkpoint"]["frames"]) == 1 and result["checkpoint"]["frames"][0]["next_index"] == 1
              and result["new_effects"] == (0 if phase == "initiation" else 2))
        terminal = deepcopy(child_failure)
        terminal.update(mode="resume", history=deepcopy(result["new_events"]), checkpoint=deepcopy(result["checkpoint"]), event_budget=0)
        check("terminal_" + phase + "_resume_no_events", solve(terminal)["status"] == phase + "_failed" and not solve(terminal)["new_events"])
    terminal_restart = deepcopy(I)
    terminal_restart.update(mode="restart", history=deepcopy(initiation_failed["new_events"]), checkpoint=deepcopy(initiation_failed["checkpoint"]))
    check("failed_checkpoint_restart_new_entry_allocation", solve(terminal_restart)["status"] == "initiation_failed"
          and solve(terminal_restart)["new_events"][0]["instance_id"] == 0)
    huge = small([{"op": "scale", "register": "x", "value": {"const": 2 ** 100}}])
    check("exact_unbounded_integer_arithmetic_not_cohort_range_gate", solve(huge)["checkpoint"]["state"]["x"] == 2 ** 100)
    live_read = small([{"op": "add", "register": "x", "value": {"reg": "x"}},
                       {"op": "scale", "register": "y", "value": {"reg": "x"}}, {"op": "swap"}])
    check("live_register_reads_and_simultaneous_swap", solve(live_read)["checkpoint"]["state"] == {"x": 4, "y": 2})
    for name, task in [("empty", small([])), ("empty_branch", empty_branch), ("not_early", not_early), ("simultaneous", simultaneous), ("live_read", live_read), ("huge", huge)]:
        direct, actual = recursive_trace(task), solve({**task, "event_budget": 48})
        check("second_simulator_boundary_" + name, same(direct["events"], actual["new_events"])
              and same(direct["state"], actual["checkpoint"]["state"]) and direct["status"] == actual["status"])
    return {"scope": "PRODUCER_HAND_CALIBRATION_ONLY", "checks": checks, "passed": len(checks), "failed": 0,
            "manual_trace_events": 19, "legal_prefix_cuts": 20, "hand_contexts": 9,
            "malformed_cases": len(mutations) + 1, "candidate_records_created": 0, "production_semantic_parents_created": 0,
            "independent_staffed_oracle_or_review": False, "production_feasibility_and_novelty": "PROSPECTIVE_NOT_EXECUTED"}


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
