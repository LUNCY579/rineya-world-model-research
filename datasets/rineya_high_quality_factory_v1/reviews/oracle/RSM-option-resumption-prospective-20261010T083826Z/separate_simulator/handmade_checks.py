"""Hand-derived full-event checks; run only after source/test preexecution freeze."""

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

AGENT = "/root/formal_oracle/rsm_recursive_simulator"
SPEC_SHA = "44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f"
TRUE = {"all": []}


def atom(left, cmp, right):
    return {"all": [{"left": left, "cmp": cmp, "right": right}]}


def option(params, body, initiation=None, success=None):
    return {
        "params": params, "body": body,
        "initiation": copy.deepcopy(TRUE if initiation is None else initiation),
        "success": copy.deepcopy(TRUE if success is None else success),
    }


def task(options, args=None, state=None):
    return {
        "kind": "option_resume", "version": 1,
        "initial_state": {"x": 0, "y": 0} if state is None else state,
        "options": options,
        "entry": {"option": "root", "args": {} if args is None else args},
        "mode": "fresh", "history": [], "checkpoint": None, "event_budget": 1000,
    }


def enter(instance, name, bindings, site=None, ok=True):
    return {"event": "enter", "instance_id": instance, "option": name,
            "bindings": bindings, "call_site": site, "initiation_ok": ok}


def effect(instance, name, path, index, op, x, y, register=None, value=None):
    event = {"event": "effect", "instance_id": instance, "option": name,
             "body_path": path, "index": index, "op": op}
    if register is not None:
        event.update(register=register, value=value)
    event["state"] = {"x": x, "y": y}
    return event


def branch(instance, name, path, index, condition):
    return {"event": "branch", "instance_id": instance, "option": name,
            "body_path": path, "index": index, "condition": condition}


def ret(instance, name, ok=True):
    return {"event": "return", "instance_id": instance, "option": name, "success_ok": ok}


def expected(phase, x, y, events):
    return {"phase": phase, "state": {"x": x, "y": y}, "events": events}


def build_cases():
    cases = []
    root = option(["p", "q"], [
        {"op": "add", "register": "x", "value": {"param": "p"}},
        {"op": "scale", "register": "y", "value": {"param": "q"}},
        {"op": "swap"},
        {"op": "add", "register": "x", "value": {"param": "q"}},
    ], success={"all": [
        {"left": {"reg": "x"}, "cmp": "eq", "right": {"const": 12}},
        {"left": {"reg": "y"}, "cmp": "eq", "right": {"const": 4}},
    ]})
    cases.append(("simultaneous_root_bindings_and_retained_values",
        task({"root": root}, {"q": {"reg": "y"}, "p": {"reg": "x"}}, {"x": 2, "y": 3}),
        None, expected("completed", 12, 4, [
            enter(0, "root", {"p": 2, "q": 3}),
            effect(0, "root", [], 0, "add", 4, 3, "x", 2),
            effect(0, "root", [], 1, "scale", 4, 9, "y", 3),
            effect(0, "root", [], 2, "swap", 9, 4),
            effect(0, "root", [], 3, "add", 12, 4, "x", 3), ret(0, "root"),
        ])))

    root = option(["p"], [
        {"op": "add", "register": "x", "value": {"param": "p"}},
        {"op": "call", "option": "child", "args": {
            "c": {"param": "p"}, "b": {"reg": "y"}, "a": {"reg": "x"}}},
        {"op": "scale", "register": "y", "value": {"param": "p"}},
    ])
    child = option(["a", "b", "c"], [
        {"op": "add", "register": "x", "value": {"param": "a"}},
        {"op": "add", "register": "y", "value": {"param": "b"}},
        {"op": "add", "register": "x", "value": {"param": "c"}},
    ])
    cases.append(("nested_simultaneous_arguments_and_caller_binding",
        task({"root": root, "child": child}, {"p": {"const": 5}}, {"x": 2, "y": 7}),
        None, expected("completed", 19, 70, [
            enter(0, "root", {"p": 5}), effect(0, "root", [], 0, "add", 7, 7, "x", 5),
            enter(1, "child", {"a": 7, "b": 7, "c": 5}, {"caller_instance": 0, "body_path": [], "index": 1}),
            effect(1, "child", [], 0, "add", 14, 7, "x", 7),
            effect(1, "child", [], 1, "add", 14, 14, "y", 7),
            effect(1, "child", [], 2, "add", 19, 14, "x", 5), ret(1, "child"),
            effect(0, "root", [], 2, "scale", 19, 70, "y", 5), ret(0, "root"),
        ])))

    root = option([], [
        {"op": "if", "test": atom({"reg": "x"}, "lt", {"reg": "y"}), "then": [
            {"op": "swap"},
            {"op": "if", "test": atom({"reg": "x"}, "eq", {"const": 8}),
             "then": [{"op": "call", "option": "child", "args": {}}], "else": []},
        ], "else": [{"op": "add", "register": "x", "value": {"const": 100}}]},
        {"op": "add", "register": "y", "value": {"const": 1}},
    ])
    child = option([], [{"op": "if", "test": TRUE, "then": [], "else": [{"op": "swap"}]}])
    cases.append(("nested_branch_paths_call_sites_order_and_empty_branch",
        task({"root": root, "child": child}, state={"x": 1, "y": 8}), None,
        expected("completed", 8, 2, [
            enter(0, "root", {}), branch(0, "root", [], 0, True),
            effect(0, "root", [0, "then"], 0, "swap", 8, 1),
            branch(0, "root", [0, "then"], 1, True),
            enter(1, "child", {}, {"caller_instance": 0, "body_path": [0, "then", 1, "then"], "index": 0}),
            branch(1, "child", [], 0, True), ret(1, "child"),
            effect(0, "root", [], 1, "add", 8, 2, "y", 1), ret(0, "root"),
        ])))

    cases.append(("empty_body_and_empty_predicates", task({"root": option([], [])}), None,
        expected("completed", 0, 0, [enter(0, "root", {}), ret(0, "root")])))

    root = option([], [{"op": "swap"}], initiation=atom({"reg": "x"}, "lt", {"const": 0}))
    cases.append(("failed_root_entry_has_no_body_or_return", task({"root": root}, state={"x": 0, "y": 2}), None,
        expected("initiation_failed", 0, 2, [enter(0, "root", {}, ok=False)])))

    root = option([], [
        {"op": "add", "register": "x", "value": {"const": 2}},
        {"op": "call", "option": "child", "args": {}}, {"op": "swap"},
    ])
    child = option([], [{"op": "swap"}], initiation=atom({"reg": "x"}, "lt", {"const": 1}))
    cases.append(("failed_nested_entry_halts_caller_and_retains_effect", task({"root": root, "child": child}), None,
        expected("initiation_failed", 2, 0, [enter(0, "root", {}),
            effect(0, "root", [], 0, "add", 2, 0, "x", 2),
            enter(1, "child", {}, {"caller_instance": 0, "body_path": [], "index": 1}, False),
        ])))

    root = option([], [{"op": "call", "option": "child", "args": {}}, {"op": "swap"}])
    child = option([], [{"op": "add", "register": "x", "value": {"const": 1}}],
                   success=atom({"reg": "x"}, "eq", {"const": 2}))
    cases.append(("failed_nested_exit_halts_caller_without_rollback", task({"root": root, "child": child}), None,
        expected("success_failed", 1, 0, [enter(0, "root", {}),
            enter(1, "child", {}, {"caller_instance": 0, "body_path": [], "index": 0}),
            effect(1, "child", [], 0, "add", 1, 0, "x", 1), ret(1, "child", False),
        ])))

    root = option([], [
        {"op": "add", "register": "x", "value": {"const": 1}},
        {"op": "add", "register": "x", "value": {"const": 1}},
    ], success=atom({"reg": "x"}, "eq", {"const": 1}))
    cases.append(("success_is_checked_at_exhaustion_and_not_early", task({"root": root}, state={"x": 1, "y": 0}), None,
        expected("success_failed", 3, 0, [enter(0, "root", {}),
            effect(0, "root", [], 0, "add", 2, 0, "x", 1),
            effect(0, "root", [], 1, "add", 3, 0, "x", 1), ret(0, "root", False),
        ])))

    root = option(["p"], [{"op": "add", "register": "x", "value": {"param": "p"}}],
                  initiation=atom({"param": "p"}, "ge", {"const": 10}))
    cases.append(("start_state_rebinds_root_and_restarts_invocation_ids",
        task({"root": root}, {"p": {"reg": "x"}}, {"x": 1, "y": 0}), {"x": 10, "y": 4},
        expected("completed", 20, 4, [enter(0, "root", {"p": 10}),
            effect(0, "root", [], 0, "add", 20, 4, "x", 10), ret(0, "root"),
        ])))

    root = option([], [
        {"op": "add", "register": "x", "value": {"reg": "x"}},
        {"op": "scale", "register": "x", "value": {"const": -3}},
        {"op": "scale", "register": "y", "value": {"const": 0}},
    ])
    cases.append(("large_exact_integer_self_operand_and_zero_negative_scaling",
        task({"root": root}, state={"x": 36893488147419103232, "y": -3}), None,
        expected("completed", -221360928884514619392, 0, [enter(0, "root", {}),
            effect(0, "root", [], 0, "add", 73786976294838206464, -3, "x", 36893488147419103232),
            effect(0, "root", [], 1, "scale", -221360928884514619392, -3, "x", -3),
            effect(0, "root", [], 2, "scale", -221360928884514619392, 0, "y", 0), ret(0, "root"),
        ])))

    root = option([], [{"op": "if", "test": {"all": [
        {"left": {"reg": "x"}, "cmp": "ge", "right": {"const": 0}},
        {"left": {"reg": "y"}, "cmp": "lt", "right": {"const": 0}},
    ]}, "then": [{"op": "swap"}], "else": []}, {"op": "swap"}])
    cases.append(("false_conjunction_empty_else_and_following_instruction",
        task({"root": root}, state={"x": 2, "y": 1}), None,
        expected("completed", 1, 2, [enter(0, "root", {}), branch(0, "root", [], 0, False),
            effect(0, "root", [], 1, "swap", 1, 2), ret(0, "root"),
        ])))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    source = pathlib.Path(args.source).resolve()
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    test_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    assert source_sha == manifest["files"]["recursive_full_trace.py"]["sha256"]
    assert test_sha == manifest["files"]["handmade_checks.py"]["sha256"]
    assert manifest["spec_sha256"] == SPEC_SHA
    assert manifest["source_was_executed_before_freeze"] is False
    module_spec = importlib.util.spec_from_file_location("frozen_recursive_trace", source)
    simulator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(simulator)
    records = []
    for name, supplied_task, start_state, wanted in build_cases():
        original_task = copy.deepcopy(supplied_task)
        original_start = copy.deepcopy(start_state)
        actual = simulator.full_trace(supplied_task, start_state)
        passed = simulator.typed_equal(actual, wanted)
        inputs_unmodified = simulator.typed_equal(supplied_task, original_task) and simulator.typed_equal(start_state, original_start)
        records.append({"name": name, "task": supplied_task, "start_state": start_state,
                        "expected": wanted, "actual": actual, "passed": passed,
                        "inputs_unmodified": inputs_unmodified})
    equality_inputs = [
        ({"a": 1, "b": [True, 2]}, {"b": [True, 2], "a": 1}, True),
        ([1, 2], [2, 1], False), (True, 1, False), (False, 0, False),
        (1, 1.0, False), ({"v": [False]}, {"v": [0]}, False),
        (None, False, False), ({"a": 1}, {"a": 1, "b": 2}, False),
    ]
    equality_records = []
    for left, right, wanted in equality_inputs:
        actual = simulator.typed_equal(left, right)
        equality_records.append({"left": left, "right": right, "expected": wanted,
                                 "actual": actual, "passed": actual is wanted})
    passed = all(r["passed"] and r["inputs_unmodified"] for r in records) and all(r["passed"] for r in equality_records)
    evidence = {
        "schema": "nonbinding_recursive_trace_execution_evidence_v1",
        "actual_execution_agent": AGENT,
        "execution_role": "nonbinding simulator author and hand-test executor",
        "approval": False,
        "binding_adjudication": False,
        "public_design_author": "/root/research_architect",
        "direct_guidance_from": "/root/formal_oracle",
        "only_external_source_read": "High-Quality_Dataset/schemas/RSM_OPTION_RESUMPTION_V1.md",
        "spec_sha256": SPEC_SHA,
        "source_sha256": source_sha, "test_source_sha256": test_sha,
        "frozen_source_path": str(source),
        "execution_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_executable": sys.executable, "python_version": sys.version,
        "process_id": os.getpid(),
        "recursive_implementation": True,
        "explicit_runtime_frame_stack": False,
        "supports_resume_checkpoint_validation": False,
        "full_trace_case_count": len(records), "typed_equality_case_count": len(equality_records),
        "cases": records, "typed_equality_cases": equality_records, "all_passed": passed,
    }
    output = pathlib.Path(args.evidence)
    with output.open("x") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"all_passed": passed, "full_trace_cases": len(records),
                      "typed_equality_cases": len(equality_records), "source_sha256": source_sha,
                      "evidence": str(output)}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
