"""Frozen prospective production source; DO NOT execute without root grant.

Only the producer's frozen e1 reference is injected for targets. No oracle
dispatch. Twelve code choices are prospective until actual novelty/control
checks pass. All student prompts contain the complete public task contract.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "rsm-option-resumption-pilot-1.0.0"
CONTEXTS = ("F", "R", "T", "P", "U", "A", "I", "S", "X")
CONTROL_CONTEXTS = {
    "uninterrupted_resume_suffix": ("F", "R"),
    "resume_restart_replay": ("R", "T"),
    "budget_pause_after_effect": ("R", "P"),
    "second_checkpoint_suffix": ("P", "U"),
    "positive_bound_argument_counterfactual": ("F", "A"),
    "initiation_argument_counterfactual": ("F", "I"),
    "success_contract_counterfactual": ("F", "S"),
    "active_checkpoint_tamper_rejection": ("R", "X"),
}
STRATA = {12: "saved_argument_after_mutation", 13: "saved_argument_after_mutation", 18: "saved_argument_after_mutation",
          14: "nested_return_caller_continuation", 15: "nested_return_caller_continuation", 19: "nested_return_caller_continuation",
          16: "selected_branch_after_guard_reversal", 17: "selected_branch_after_guard_reversal", 28: "selected_branch_after_guard_reversal",
          20: "noncommuting_calls_and_exit_contract", 21: "noncommuting_calls_and_exit_contract", 29: "noncommuting_calls_and_exit_contract"}


class NeedNewPublicAmendment(RuntimeError):
    """No weakened criteria, automatic replacement or quota padding allowed."""


def require(condition, message):
    if not condition:
        raise NeedNewPublicAmendment(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reverse_fields(value):
    if type(value) is dict:
        return {key: reverse_fields(value[key]) for key in reversed(list(value))}
    if type(value) is list:
        return [reverse_fields(item) for item in value]
    return value


def differences(left, right, path=()):
    if type(left) is not type(right):
        return [(list(path), left, right)]
    if type(left) is dict and set(left) == set(right):
        return [item for key in sorted(left) for item in differences(left[key], right[key], path + (key,))]
    if type(left) is list and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, path + (index,))]
    return [] if left == right else [(list(path), left, right)]


def read(expr, state, bindings):
    if "reg" in expr:
        return state[expr["reg"]]
    return bindings[expr["param"]] if "param" in expr else expr["const"]


def atoms(predicate, state, bindings):
    outcomes = []
    for atom in predicate["all"]:
        left, right = read(atom["left"], state, bindings), read(atom["right"], state, bindings)
        outcomes.append(left < right if atom["cmp"] == "lt" else left == right if atom["cmp"] == "eq" else left >= right)
    return outcomes


def body_at(task, option, path):
    body = task["options"][option]["body"]
    for offset in range(0, len(path), 2):
        body = body[path[offset]][path[offset + 1]]
    return body


def primitive(instruction, state, bindings):
    after = deepcopy(state)
    if instruction["op"] == "swap":
        after["x"], after["y"] = state["y"], state["x"]
    else:
        reg = instruction["register"]
        value = read(instruction["value"], state, bindings)
        after[reg] = state[reg] + value if instruction["op"] == "add" else state[reg] * value
    return after


def declared_metrics(task):
    def instructions(body, depth=0):
        count, maximum = len(body), depth
        for instruction in body:
            if instruction["op"] == "if":
                for branch in ("then", "else"):
                    additional, subdepth = instructions(instruction[branch], depth + 1)
                    count += additional
                    maximum = max(maximum, subdepth)
        return count, maximum
    measurements = [instructions(option["body"]) for option in task["options"].values()]
    return {"declared_instructions": sum(item[0] for item in measurements),
            "syntactic_conditional_depth": max(item[1] for item in measurements)}


def build_family(index, fixture):
    """Instantiate one actual graph only when the production runner is granted."""
    task = deepcopy(fixture["task"])
    # Captured p=2 makes the shared scale instructions actively non-idempotent.
    task["initial_state"] = {"x": 2, "y": 2}
    root = task["options"]["start"]["body"]
    selected = task["options"]["work"]["body"][0]["then"]
    parameter = lambda name: {"param": name}
    scalar = lambda value: {"const": value}
    add = lambda reg, operand: {"op": "add", "register": reg, "value": operand}
    scale = lambda reg, operand: {"op": "scale", "register": reg, "value": operand}
    call = lambda name, arguments: {"op": "call", "option": name, "args": arguments}
    if index == 12:
        pass
    elif index == 13:
        root.insert(2, {"op": "swap"})
    elif index == 18:
        root.insert(2, add("x", {"reg": "y"}))
    elif index == 14:
        root.insert(2, scale("x", parameter("p")))
    elif index in (15, 20, 29):
        body = [scale("x" if index == 15 else "y", parameter("u"))]
        if index == 20:
            body.append(add("x", parameter("u")))
        task["options"]["multiply"] = {"params": ["u"], "initiation": {"all": []}, "success": {"all": []}, "body": body}
        if index == 29:
            selected.append(call("multiply", {"u": scalar(2)}))
        else:
            root.insert(2, call("multiply", {"u": scalar(2)}))
    elif index == 19:
        root.insert(3, call("work", {"q": parameter("p"), "r": {"reg": "x"}}))
    elif index == 16:
        selected[2]["then"].insert(0, add("y", parameter("q")))
    elif index == 17:
        selected.append({"op": "if", "test": {"all": [{"left": {"reg": "y"}, "cmp": "ge", "right": parameter("q")}]},
                         "then": [add("x", parameter("q"))], "else": [scale("x", parameter("q"))]})
    elif index == 28:
        root.insert(2, {"op": "if", "test": {"all": [{"left": {"reg": "y"}, "cmp": "ge", "right": parameter("p")}]},
                        "then": [call("finish", {"t": parameter("p")})], "else": [add("x", parameter("p"))]})
    elif index == 21:
        root[2:2] = [scale("y", parameter("p")), {"op": "swap"}, scale("x", parameter("p"))]
    else:
        raise ValueError(index)
    return task


def _checkpoint_candidate(task, full, reference):
    for cut in range(1, len(full["new_events"])):
        interrupted = deepcopy(task)
        interrupted["event_budget"] = cut
        prefix = reference.solve(interrupted)
        machine = prefix["checkpoint"]
        if prefix["status"] != "paused" or prefix["new_effects"] < 1:
            continue
        frames = machine["frames"]
        option_frames = [frame for frame in frames if frame["frame_kind"] == "option"]
        branches = [(position, frame) for position, frame in enumerate(frames) if frame["frame_kind"] == "branch"]
        if len(option_frames) < 2 or not branches:
            continue
        root_id = option_frames[0]["instance_id"]
        failed_current_entry = [frame for frame in option_frames if frame["instance_id"] != root_id
                                and not all(atoms(task["options"][frame["option"]]["initiation"], machine["state"], frame["bindings"]))]
        if not failed_current_entry:
            continue
        for position, frame in branches:
            path = frame["body_path"]
            original_if = body_at(task, frame["option"], path[:-2])[path[-2]]
            if path[-1] != "then" or all(atoms(original_if["test"], machine["state"], frame["bindings"])):
                continue
            baseline = reference.trusted_machine_probe(task, machine, 48)
            for parameter in sorted(frame["bindings"]):
                # A direct live-register call actual, copied into this exact frame.
                entered = next(event for event in prefix["new_events"] if event["event"] == "enter" and event["instance_id"] == frame["instance_id"])
                call_site = entered["call_site"]
                if call_site is None:
                    continue
                caller = next(event for event in prefix["new_events"] if event["event"] == "enter" and event["instance_id"] == call_site["caller_instance"])
                instruction = body_at(task, caller["option"], call_site["body_path"])[call_site["index"]]
                if "reg" not in instruction["args"][parameter]:
                    continue
                changed = deepcopy(machine)
                changed["frames"][position]["bindings"][parameter] += 1
                altered = reference.trusted_machine_probe(task, changed, 48)
                if not reference.typed_equal(baseline, altered):
                    return prefix, cut, {"saved_binding_locator": ["frames", position, "bindings", parameter],
                                         "entered_child_current_initiation_false": [frame["instance_id"] for frame in failed_current_entry],
                                         "selected_branch_guard_now_false": {"frame_index": position, "body_path": path},
                                         "active_binding_probe_differences": differences(baseline, altered)}
    raise NeedNewPublicAmendment("No simultaneously valid active R boundary")


def _replay_witness(task, history, restarted, reference):
    require(restarted["new_events"] and restarted["new_events"][0]["event"] == "enter"
            and restarted["new_events"][0]["initiation_ok"] is True, "Restart blocked before replay")
    state = deepcopy(task["initial_state"])
    bindings = {}
    for position, event in enumerate(history):
        if event["event"] == "enter":
            bindings[event["instance_id"]] = event["bindings"]
        elif event["event"] == "effect":
            instruction = body_at(task, event["option"], event["body_path"])[event["index"]]
            once = primitive(instruction, state, bindings[event["instance_id"]])
            twice = primitive(instruction, once, bindings[event["instance_id"]])
            matches = [(i, item) for i, item in enumerate(restarted["new_events"]) if item["event"] == "effect"
                       and item["option"] == event["option"] and item["body_path"] == event["body_path"] and item["index"] == event["index"]]
            if not reference.typed_equal(once, twice) and matches:
                return {"replay_locator": {"option": event["option"], "body_path": event["body_path"], "index": event["index"]},
                        "history_event_index": position, "restart_event_index": matches[0][0],
                        "original_pre_state": state, "original_once_state": once, "original_twice_state": twice,
                        "restart_actual_effect": matches[0][1]}
            state = deepcopy(event["state"])
    raise NeedNewPublicAmendment("No actually replayed non-idempotent committed effect")


def derive_contexts(task, index, reference):
    """Build actual controls from the exact finite producer execution."""
    F = deepcopy(task)
    F["event_budget"] = 48
    f = reference.solve(F)
    require(f["status"] == "completed", "F did not complete successfully")
    prefix, cut, retention = _checkpoint_candidate(F, f, reference)
    R = deepcopy(F)
    R.update(mode="resume", history=deepcopy(prefix["new_events"]), checkpoint=deepcopy(prefix["checkpoint"]))
    r = reference.solve(R)
    require(reference.typed_equal(r["new_events"], f["new_events"][cut:])
            and reference.typed_equal(r["checkpoint"], f["checkpoint"]), "F-R suffix mismatch")
    T = deepcopy(R)
    T["mode"] = "restart"
    t = reference.solve(T)
    replay = _replay_witness(F, R["history"], t, reference)
    require(not reference.typed_equal(t["new_events"], r["new_events"])
            and (t["status"] != r["status"] or not reference.typed_equal(t["checkpoint"]["state"], r["checkpoint"]["state"])), "Restart replay did not alter trace and terminal result")
    for budget in range(1, len(r["new_events"])):
        P = deepcopy(R)
        P["event_budget"] = budget
        p = reference.solve(P)
        if p["status"] == "paused" and p["new_effects"] >= 1:
            break
    else:
        raise NeedNewPublicAmendment("No second checkpoint after a new effect")
    require(reference.typed_equal(p["new_events"], r["new_events"][:budget]), "P not actual R prefix")
    U = deepcopy(R)
    U.update(history=deepcopy(R["history"] + p["new_events"]), checkpoint=deepcopy(p["checkpoint"]))
    u = reference.solve(U)
    require(reference.typed_equal(u["new_events"], f["new_events"][len(U["history"]):])
            and reference.typed_equal(u["checkpoint"], f["checkpoint"]), "Second suffix mismatch")
    A = deepcopy(F)
    A["entry"]["args"]["p"] = {"const": 3}
    a = reference.solve(A)
    require(a["status"] == "completed" and a["new_events"][0]["initiation_ok"] is True, "Positive argument did not complete")
    behavior_f = [event for event in f["new_events"] if event["event"] != "enter"]
    behavior_a = [event for event in a["new_events"] if event["event"] != "enter"]
    require(not reference.typed_equal(behavior_f, behavior_a), "Argument is unused beyond enter bindings")
    # Actual immutable p passed to q and consumed by the work add operand.
    dependent_f = next(event for event in f["new_events"] if event["event"] == "effect" and event["option"] == "work" and event["body_path"] == [0, "then"] and event["index"] == 0)
    dependent_a = next(event for event in a["new_events"] if event["event"] == "effect" and event["option"] == "work" and event["body_path"] == [0, "then"] and event["index"] == 0)
    require(dependent_f["value"] != dependent_a["value"], "Changed bound argument did not affect later evaluated operand")
    I = deepcopy(F)
    I["entry"]["args"]["p"] = {"const": 0}
    i = reference.solve(I)
    root = F["entry"]["option"]
    init_before = atoms(F["options"][root]["initiation"], F["initial_state"], f["new_events"][0]["bindings"])
    init_after = atoms(I["options"][root]["initiation"], I["initial_state"], i["new_events"][0]["bindings"])
    require(all(init_before) and init_after.count(False) == 1 and i["status"] == "initiation_failed" and i["new_effects"] == 0, "I is not a sole-guard, no-effects failure")
    S = deepcopy(F)
    S["options"][root]["success"]["all"][0]["right"]["const"] = f["checkpoint"]["state"]["x"] + 1
    s = reference.solve(S)
    exit_after = atoms(S["options"][root]["success"], s["checkpoint"]["state"], f["new_events"][0]["bindings"])
    require(s["status"] == "success_failed" and exit_after.count(False) == 1
            and reference.typed_equal(s["new_events"][:-1], f["new_events"][:-1])
            and reference.typed_equal(s["checkpoint"]["state"], f["checkpoint"]["state"]), "S changes more than exit contract success")
    X = deepcopy(R)
    tamper_kind = ("binding", "cursor", "state")[sorted(STRATA).index(index) % 3]
    locator = list(retention["saved_binding_locator"])
    if tamper_kind == "cursor":
        locator = ["frames", locator[1], "next_index"]
    elif tamper_kind == "state":
        locator = ["state", "x"]
    container = X["checkpoint"]
    for key in locator[:-1]:
        container = container[key]
    container[locator[-1]] += 1
    x = reference.solve(X)
    original_probe = reference.trusted_machine_probe(R, R["checkpoint"], 48)
    changed_probe = reference.trusted_machine_probe(R, X["checkpoint"], 48)
    require(len(differences(R, X)) == 1 and x["error"] == "invalid_checkpoint" and not x["new_events"]
            and not reference.typed_equal(original_probe, changed_probe), "X not exactly one consequential checkpoint scalar")
    tasks = dict(zip(CONTEXTS, (F, R, T, P, U, A, I, S, X)))
    targets = dict(zip(CONTEXTS, (f, r, t, p, u, a, i, s, x)))
    evidence = {"first_cut_events": cut, "second_cut_events": len(U["history"]), "short_budget": budget,
                "retention": retention, "replay": replay,
                "argument_operand_counterfactual": {"F_effect": dependent_f, "A_effect": dependent_a},
                "initiation_atom_truth": {"F": init_before, "I": init_after}, "success_atom_truth_S": exit_after,
                "tamper_kind": tamper_kind, "checkpoint_tamper_locator": locator,
                "tamper_trusted_probe_differences": differences(original_probe, changed_probe),
                "actual_pair_input_differences": {name: differences(tasks[left], tasks[right]) for name, (left, right) in CONTROL_CONTEXTS.items()}}
    for name, pair in CONTROL_CONTEXTS.items():
        left, right = tasks[pair[0]], tasks[pair[1]]
        if name in ("resume_restart_replay", "budget_pause_after_effect"):
            required_field = "mode" if name == "resume_restart_replay" else "event_budget"
            require(len(differences(left, right)) == 1 and differences(left, right)[0][0] == [required_field], "Unexpected control difference")
    return tasks, targets, evidence


def validate_bounds(tasks, targets, reference):
    F = tasks["F"]
    metrics = declared_metrics(F)
    require(2 <= len(F["options"]) <= 4 and all(1 <= len(option["params"]) <= 3 for option in F["options"].values()), "Option/parameter cohort bounds")
    require(metrics["declared_instructions"] <= 24 and metrics["syntactic_conditional_depth"] <= 2, "Declared instruction/branch bounds")
    require(all(-6 <= value <= 6 for value in F["initial_state"].values()), "Initial range bounds")
    require(len(targets["F"]["new_events"]) <= 48 and targets["F"]["new_effects"] >= 3, "F event/effect bounds")
    require({event["option"] for event in targets["F"]["new_events"] if event["event"] == "enter"} == set(F["options"]), "Unused option padding cannot add a family")
    maximum_options = maximum_branches = 0
    for cut in range(len(targets["F"]["new_events"]) + 1):
        task = deepcopy(F)
        task["event_budget"] = cut
        frames = reference.solve(task)["checkpoint"]["frames"]
        maximum_options = max(maximum_options, sum(frame["frame_kind"] == "option" for frame in frames))
        maximum_branches = max(maximum_branches, sum(frame["frame_kind"] == "branch" for frame in frames))
    require(maximum_options <= 3, "Maximum simultaneous option depth")
    for context, target in targets.items():
        if target["status"] != "invalid":
            require(target["status"] != "paused" or context == "P", "Full continuation budget insufficient")
            states = [tasks[context]["initial_state"], target["checkpoint"]["state"]]
            states += [event["state"] for event in tasks[context]["history"] + target["new_events"] if event["event"] == "effect"]
            require(all(-4096 <= value <= 4096 for state in states for value in state.values()), "Actual valid-run intermediate range")
    return {"maximum_option_depth": maximum_options, "maximum_selected_branch_depth": maximum_branches,
            "declared_instructions": metrics["declared_instructions"], "total_F_events": len(targets["F"]["new_events"]),
            "total_F_effects": targets["F"]["new_effects"],
            "history_effects_at_R": sum(event["event"] == "effect" for event in tasks["R"]["history"]),
            "caller_frames_at_R": sum(frame["frame_kind"] == "option" for frame in tasks["R"]["checkpoint"]["frames"]) - 1,
            "selected_branch_frames_at_R": sum(frame["frame_kind"] == "branch" for frame in tasks["R"]["checkpoint"]["frames"])}


def verify_actual_relation(name, left, right, reference):
    """Reconstruct a supplied relation from actual member tasks/own targets."""
    a, b = left["semantic_task"], right["semantic_task"]
    ta, tb = left["verified_target"], right["verified_target"]
    common_program = all(reference.typed_equal(a[field], b[field]) for field in ("initial_state", "entry", "options"))
    if name == "uninterrupted_resume_suffix":
        require(common_program and a["mode"] == "fresh" and b["mode"] == "resume" and b["history"]
                and any(event["event"] == "effect" for event in b["history"])
                and b["checkpoint"]["phase"] == "running" and ta["status"] == tb["status"] == "completed"
                and reference.typed_equal(b["history"], ta["new_events"][:len(b["history"])])
                and reference.typed_equal(tb["new_events"], ta["new_events"][len(b["history"]):])
                and reference.typed_equal(ta["checkpoint"], tb["checkpoint"]), "Actual F-R relation")
    elif name == "resume_restart_replay":
        require(len(differences(a, b)) == 1 and differences(a, b)[0][0] == ["mode"] and a["mode"] == "resume" and b["mode"] == "restart", "Actual R-T typed diff")
        _replay_witness(a, a["history"], tb, reference)
        require(not reference.typed_equal(ta["new_events"], tb["new_events"])
                and (ta["status"] != tb["status"] or not reference.typed_equal(ta["checkpoint"]["state"], tb["checkpoint"]["state"])), "Actual R-T terminal/trace distinction")
    elif name == "budget_pause_after_effect":
        require(len(differences(a, b)) == 1 and differences(a, b)[0][0] == ["event_budget"] and 0 < b["event_budget"] < a["event_budget"]
                and tb["status"] == "paused" and tb["new_effects"] > 0 and tb["next_event"] is not None
                and reference.typed_equal(tb["new_events"], ta["new_events"][:len(tb["new_events"])]), "Actual R-P relation")
    elif name == "second_checkpoint_suffix":
        fresh = deepcopy(a)
        fresh.update(mode="fresh", history=[], checkpoint=None, event_budget=48)
        full = reference.solve(fresh)
        require(common_program and b["mode"] == "resume" and b["event_budget"] == 48
                and reference.typed_equal(b["history"], a["history"] + ta["new_events"])
                and reference.typed_equal(b["checkpoint"], ta["checkpoint"])
                and reference.typed_equal(tb["new_events"], full["new_events"][len(b["history"]):])
                and reference.typed_equal(tb["checkpoint"], full["checkpoint"]), "Actual P-U second suffix")
    elif name in ("positive_bound_argument_counterfactual", "initiation_argument_counterfactual"):
        changed_keys = [key for key in a["entry"]["args"] if not reference.typed_equal(a["entry"]["args"][key], b["entry"]["args"].get(key))]
        patched = deepcopy(a)
        patched["entry"]["args"] = deepcopy(b["entry"]["args"])
        require(len(changed_keys) == 1 and reference.typed_equal(patched, b), "Actual single root-argument counterfactual")
        if name == "positive_bound_argument_counterfactual":
            require(ta["status"] == tb["status"] == "completed" and ta["new_events"][0]["initiation_ok"] and tb["new_events"][0]["initiation_ok"]
                    and not reference.typed_equal([event for event in ta["new_events"] if event["event"] != "enter"], [event for event in tb["new_events"] if event["event"] != "enter"]), "Actual used positive bound argument")
        else:
            root = a["entry"]["option"]
            truth = atoms(b["options"][root]["initiation"], b["initial_state"], tb["new_events"][0]["bindings"])
            require(ta["new_events"][0]["initiation_ok"] and tb["status"] == "initiation_failed" and tb["new_effects"] == 0 and truth.count(False) == 1, "Actual sole initiation failure")
    elif name == "success_contract_counterfactual":
        diff = differences(a, b)
        root = a["entry"]["option"]
        require(len(diff) == 1 and diff[0][0][:3] == ["options", root, "success"] and diff[0][0][-1] == "const"
                and ta["status"] == "completed" and tb["status"] == "success_failed"
                and reference.typed_equal(ta["new_events"][:-1], tb["new_events"][:-1])
                and reference.typed_equal(ta["checkpoint"]["state"], tb["checkpoint"]["state"])
                and atoms(b["options"][root]["success"], tb["checkpoint"]["state"], ta["new_events"][0]["bindings"]).count(False) == 1, "Actual sole success-literal change/no rollback")
    elif name == "active_checkpoint_tamper_rejection":
        diff = differences(a, b)
        require(len(diff) == 1 and diff[0][0][0] == "checkpoint" and type(diff[0][1]) is int and type(diff[0][2]) is int, "Actual single scalar checkpoint tamper")
        path = diff[0][0][1:]
        supported = (path in (["state", "x"], ["state", "y"]) or
                     (len(path) == 3 and path[0] == "frames" and type(path[1]) is int and path[2] == "next_index") or
                     (len(path) == 4 and path[0] == "frames" and type(path[1]) is int and path[2] == "bindings"))
        require(supported and tb["error"] == "invalid_checkpoint" and not tb["new_events"]
                and not reference.typed_equal(reference.trusted_machine_probe(a, a["checkpoint"], 48), reference.trusted_machine_probe(b, b["checkpoint"], 48)), "Actual active tamper rejected before continuation")


def manifest_validate(rows, manifest, candidate_raw, reference, source_bindings):
    require(manifest["metadata_version"] == 1 and type(manifest["metadata_version"]) is int, "Manifest metadata version")
    require(manifest["batch"] == "option-resumption-pilot-v1", "Public literal batch mismatch")
    require(manifest["candidate_sha256"] == sha(candidate_raw), "Whole candidate SHA mismatch")
    require(manifest["source_bindings"] == source_bindings, "Public source binding mismatch")
    samples = {row["sample_id"]: row for row in rows}
    require(len(samples) == len(rows) == 216, "Duplicate/missing candidate IDs")
    def identifier(item):
        name = item.get("name", item.get("property"))
        require(type(name) is str, "Nonstring control identifier")
        if "name" in item and "property" in item:
            require(type(item["property"]) is str and item["name"] == item["property"], "Mismatched control identifiers")
        return name

    resolver = {}
    for entry in manifest["control_pairs"]:
        require(type(entry["control_id"]) is str and bool(entry["control_id"]), "Empty/nonstring control ID")
        pair = entry["pair_sample_ids"]
        require(type(pair) is list and len(pair) == 2 and all(type(item) is str for item in pair) and pair[0] != pair[1], "Malformed pair IDs")
        require(all(identifier in samples for identifier in pair), "Missing actual pair member")
        key = (entry["control_id"], identifier(entry), tuple(pair))
        require(key not in resolver, "Duplicate complete resolver key")
        resolver[key] = entry
        require(entry["candidate_sha256"] == sha(candidate_raw), "Named pair whole candidate binding")
        if "pair_record_hashes" in entry:
            require(type(entry["pair_record_hashes"]) is list and len(entry["pair_record_hashes"]) == 2
                    and entry["pair_record_hashes"] == [samples[item]["record_hash"] for item in pair]
                    and all(type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value) for value in entry["pair_record_hashes"]), "Optional external member hashes")
        if identifier(entry) != "object_field_order_invariance":
            verify_actual_relation(identifier(entry), samples[pair[0]], samples[pair[1]], reference)
    require(len(resolver) == 300, "Missing actual named controls")
    all_names = set(CONTROL_CONTEXTS) | {"object_field_order_invariance"}
    for owner in rows:
        require(type(owner["source_family_index"]) is int and type(owner["sample_id"]) is str, "Owner coordinate types")
        require(reference.typed_equal(reference.solve(owner["semantic_task"]), owner["verified_target"]), "Actual task target mismatch")
        require(reference.typed_equal(json.loads(owner["rendered_input"].split("Task JSON:\n", 1)[1]), owner["semantic_task"]), "Rendered typed task mismatch")
        require("family_claims" not in owner and len(owner["family_control_claims"]) == 9, "Claim container/count")
        require(owner["record_hash"] == sha(canonical({key: value for key, value in owner.items() if key != "record_hash"}).encode()), "Full record hash")
        names = []
        for claim in owner["family_control_claims"]:
            require("pair_record_hashes" not in claim, "Forbidden circular canonical member hash")
            allowed = {"name", "property", "scope", "evidence_manifest", "control_id", "pair_sample_ids", "source_family_key", "source_family_index", "controlled_context", "renderer", "diagnostics"}
            require(set(claim) <= allowed, "Unsupported claim field")
            name = identifier(claim)
            require(type(name) is str and name in all_names, "Claim identifier")
            if "name" in claim and "property" in claim:
                require(claim["name"] == claim["property"], "Mismatched identifiers")
            require(claim["scope"] == "corpus_variant" and type(claim["control_id"]) is str and bool(claim["control_id"]), "Claim scope/control ID")
            relative = Path(claim["evidence_manifest"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe evidence path")
            pair = claim["pair_sample_ids"]
            entry = resolver.get((claim["control_id"], name, tuple(pair)))
            require(entry is not None, "Claim does not resolve once")
            members = [samples[item] for item in pair]
            require(all(member["source_family_key"] == owner["source_family_key"] and member["source_family_index"] == owner["source_family_index"] and member["split"] == owner["split"] for member in members), "Foreign-family/split claim")
            require(entry["source_family_key"] == owner["source_family_key"] and entry["source_family_index"] == owner["source_family_index"]
                    and entry["controlled_contexts"] == [member["controlled_context"] for member in members] and entry["renderers"] == [member["renderer"] for member in members], "Manifest coordinates")
            require(type(entry["source_family_index"]) is int and entry["scope"] == "corpus_variant", "Manifest typed coordinate/scope")
            for coordinate in ("source_family_key", "source_family_index", "controlled_context", "renderer"):
                if coordinate in claim:
                    require(type(claim[coordinate]) is type(owner[coordinate]) and claim[coordinate] == owner[coordinate], "Optional claim provenance")
            if "diagnostics" in claim:
                require(type(claim["diagnostics"]) is dict, "Optional diagnostics object")
            if name == "object_field_order_invariance":
                require(all(member["controlled_context"] == owner["controlled_context"] for member in members) and [member["renderer"] for member in members] == ["forward", "reverse"], "Field-order owner/pair")
                require(reference.typed_equal(members[0]["semantic_task"], members[1]["semantic_task"])
                        and reference.typed_equal(members[0]["verified_target"], members[1]["verified_target"]), "Field-order task/target")
            else:
                require([member["controlled_context"] for member in members] == list(CONTROL_CONTEXTS[name])
                        and all(member["renderer"] == owner["renderer"] for member in members), "Ordered context/style binding")
            names.append(name)
        require(set(names) == all_names and len(names) == len(set(names)), "Nine unique required names")


def diagnostics(rows, family_evidence):
    histograms = {}
    for axes in (("split",), ("controlled_context",), ("source_family_key", "controlled_context"), ("split", "controlled_context")):
        name = "/".join(axes)
        histograms[name] = {}
        for row in rows:
            key = "/".join(str(row[axis]) for axis in axes)
            counts = histograms[name].setdefault(key, {"status": Counter(), "new_effects": Counter(), "history_effects": Counter(), "new_event_horizon": Counter(), "difficulty": Counter(),
                                                      "maximum_option_depth": Counter(), "maximum_selected_branch_depth": Counter(), "caller_frames_at_R": Counter(), "selected_branch_frames_at_R": Counter()})
            target = row["verified_target"]
            counts["status"][target["status"]] += 1
            counts["new_effects"][str(target["new_effects"])] += 1
            counts["history_effects"]["null" if target["previous_effects"] is None else str(target["previous_effects"])] += 1
            counts["new_event_horizon"][str(len(target["new_events"]))] += 1
            counts["difficulty"][str(row["difficulty"])] += 1
            for measurement in ("maximum_option_depth", "maximum_selected_branch_depth", "caller_frames_at_R", "selected_branch_frames_at_R"):
                counts[measurement][str(row["property_witnesses"]["dependency_metrics"][measurement])] += 1
    baselines = {}
    training = [row for row in rows if row["split"] == "train"]
    for feature in ("majority", "mode", "history_length", "mode_history_length"):
        def bucket(row):
            task = row["semantic_task"]
            if feature == "majority": return "all"
            if feature == "mode": return task["mode"]
            if feature == "history_length": return str(len(task["history"]))
            return canonical([task["mode"], len(task["history"])])
        buckets = {}
        for row in training:
            buckets.setdefault(bucket(row), Counter())[row["verified_target"]["status"]] += 1
        majority = Counter(row["verified_target"]["status"] for row in training)
        fallback = sorted(majority, key=lambda value: (-majority[value], value))[0]
        learned = {key: sorted(counts, key=lambda value: (-counts[value], value))[0] for key, counts in sorted(buckets.items())}
        baselines[feature] = {"train_derived_lookup": learned, "majority_fallback": fallback, "metric": "status_only_not_full_typed_target"}
        for split in ("train", "dev_challenge"):
            subset = [row for row in rows if row["split"] == split]
            baselines[feature][split] = {"rows": len(subset), "correct": sum(learned.get(bucket(row), fallback) == row["verified_target"]["status"] for row in subset),
                                         "unknown_buckets": sum(bucket(row) not in learned for row in subset)}
    return {"histograms": json.loads(canonical(histograms)), "train_derived_baselines": baselines,
            "retention_replay_parameter_caller_branch_and_tamper_evidence": family_evidence,
            "limits": ["finite_program_memorization_remains_plausible", "all_four_strata_exposed_on_dev", "no_model_training_or_efficacy", "hand_calibration_shared_motif_reuse", "common_model_and_root_guidance_correlation"]}


def prepare_corpus(reference, normalizer, common, configuration, fixture, legacy_rows, contract, source_bindings, progress_hook=None):
    """Compute in memory; write nothing until ALL actual gates pass."""
    require(configuration["proposed_batch"] == "option-resumption-pilot-v1", "Batch differs from public configuration")
    indices = sorted(STRATA)
    require(indices == sorted(index for group in configuration["source_family_indices_by_stratum"].values() for index in group), "Prospective family indices differ")
    programs = {index: build_family(index, fixture) for index in indices}
    if progress_hook is not None:
        progress_hook("actual_source_inputs_before_gates", programs)
    novelty = normalizer.compare_actual(programs, legacy_rows)
    if progress_hook is not None:
        progress_hook("actual_novelty_comparison", novelty)
    for view, report in novelty.items():
        require(report["new_unique_signatures"] == 12 and not report["mutual_collision_groups"] and not report["legacy_colliding_indices"], "NEED_NEW_PUBLIC_AMENDMENT: actual novelty failed: " + view)
    tasks_by_index, targets_by_index, family_evidence = {}, {}, {}
    for index, program in programs.items():
        tasks, targets, evidence = derive_contexts(program, index, reference)
        evidence["dependency_metrics"] = validate_bounds(tasks, targets, reference)
        evidence["split"] = common.split_for(index)
        evidence["source_family_key"] = f"RSM-seed20261009-p{index:04d}"
        tasks_by_index[index], targets_by_index[index], family_evidence[index] = tasks, targets, evidence
        if progress_hook is not None:
            progress_hook(f"actual_family_{index:04d}_control_diagnostics", evidence)
    batch = configuration["proposed_batch"]
    manifest_path = f"data/RSM-001/manifests/{batch}.family_controls.json"
    # Allocate every actual ID and source split before field rendering.
    ids = {(index, context, style): f"RSM-001-{batch}-p{index:04d}-{context}-{style}"
           for index in indices for context in CONTEXTS for style in ("forward", "reverse")}
    rows, parents = [], []
    for index in indices:
        key = family_evidence[index]["source_family_key"]
        evidence = family_evidence[index]
        for context in CONTEXTS:
            task = tasks_by_index[index][context]
            target = targets_by_index[index][context]
            family = f"RSM-001-{batch}-p{index:04d}"
            parent = family + "-" + context
            parents.append({"semantic_parent_id": parent, "source_family_key": key, "source_family_index": index,
                            "semantic_family_id": family, "split": common.split_for(index), "controlled_context": context,
                            "semantic_task": task, "status": "PENDING", "generator_version": VERSION})
            for style in ("forward", "reverse"):
                view = task if style == "forward" else reverse_fields(task)
                rendered = contract.rstrip() + "\n\nTask JSON:\n" + json.dumps(view, ensure_ascii=False, separators=(",", ":"))
                require(reference.typed_equal(json.loads(rendered.split("Task JSON:\n", 1)[1]), task), "Prompt does not encode actual typed task")
                difficulty = 2 if context == "I" else 3
                row = common.record("RSM-001", batch, index, context, task, rendered, target,
                                    ["finite_synthetic_proxy", "supplied_option_execution", "versioned_checkpoint_validation"], difficulty, renderer=style)
                row.update(semantic_parent_id=parent, source_family_key=key, source_family_index=index, controlled_context=context,
                           generator_version=VERSION, oracle_version="producer-reference-rsm-option-0.1.0-unreviewed", structural_difficulty=difficulty)
                row["source_provenance"].update(actual_author_execution_id="/root/relational_critic_producer", source_stratum=STRATA[index],
                    hand_calibration_motif_reused=True, common_model_prior_context=True,
                    curriculum_difficulty_explanation=("Validate a decisive argument-bound initiation atom before any effects." if context == "I" else
                    "Execute or validate the actual nested option trace while retaining an advanced caller, selected branch and immutable register-derived argument consumed by pending arithmetic; entry and exit predicates have distinct event boundaries."))
                row["property_witnesses"] = {"saved_binding_locator": evidence["retention"]["saved_binding_locator"],
                                             "checkpoint_tamper_locator": evidence["checkpoint_tamper_locator"],
                                             "replay_locator": evidence["replay"]["replay_locator"], "dependency_metrics": evidence["dependency_metrics"]}
                claims = []
                for name, (left, right) in CONTROL_CONTEXTS.items():
                    claims.append({"name": name, "property": name, "scope": "corpus_variant", "evidence_manifest": manifest_path,
                                   "control_id": f"{key}:{style}:{name}", "pair_sample_ids": [ids[index, left, style], ids[index, right, style]]})
                claims.append({"name": "object_field_order_invariance", "property": "object_field_order_invariance", "scope": "corpus_variant",
                               "evidence_manifest": manifest_path, "control_id": f"{key}:{context}:fields",
                               "pair_sample_ids": [ids[index, context, "forward"], ids[index, context, "reverse"]]})
                row["family_control_claims"] = claims
                require(row["sample_id"] == ids[index, context, style], "Actual allocated ID mismatch")
                row["record_hash"] = common.record_hash(row)
                rows.append(row)
    candidate_raw = ("".join(common.canonical(row) + "\n" for row in rows)).encode()
    candidate_hash = sha(candidate_raw)
    samples = {row["sample_id"]: row for row in rows}
    pairs = {}
    for row in rows:
        for claim in row["family_control_claims"]:
            key = (claim["control_id"], claim["name"], tuple(claim["pair_sample_ids"]))
            if key not in pairs:
                members = [samples[item] for item in claim["pair_sample_ids"]]
                pairs[key] = {"name": claim["name"], "scope": "corpus_variant", "control_id": claim["control_id"],
                              "pair_sample_ids": claim["pair_sample_ids"], "source_family_key": row["source_family_key"],
                              "source_family_index": row["source_family_index"], "controlled_contexts": [member["controlled_context"] for member in members],
                              "renderers": [member["renderer"] for member in members], "candidate_sha256": candidate_hash,
                              "pair_record_hashes": [member["record_hash"] for member in members]}
    manifest = {"metadata_version": 1, "batch": batch, "candidate_sha256": candidate_hash,
                "source_bindings": source_bindings, "control_pairs": [pairs[key] for key in sorted(pairs)]}
    manifest_validate(rows, manifest, candidate_raw, reference, source_bindings)
    require(len(parents) == 108 and len(rows) == 216 and sum(row["split"] == "train" for row in rows) == 144, "Exact cohort counts/splits")
    require(all(row["review_status"] == "PENDING" for row in rows), "Production approval forbidden")
    return {"rows": rows, "parents": parents, "candidate_raw": candidate_raw, "manifest": manifest,
            "novelty": novelty, "diagnostics": diagnostics(rows, family_evidence),
            "counts": {"source_families_after_actual_novelty": 12, "typed_tasks": 108, "views": 216, "named_pairs": 300,
                       "owner_claims": sum(len(row["family_control_claims"]) for row in rows), "train_rows": 144, "exposed_dev_rows": 72}}
