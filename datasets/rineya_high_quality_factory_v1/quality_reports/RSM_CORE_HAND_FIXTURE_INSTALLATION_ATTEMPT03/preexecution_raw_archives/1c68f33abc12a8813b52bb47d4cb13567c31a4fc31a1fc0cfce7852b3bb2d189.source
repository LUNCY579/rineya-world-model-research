"""Own prospective hand tests. Not imported/executed during source preparation."""
from copy import deepcopy
import json


def calibrate_before_generation(reference, normalizer, generator, fixture):
    checks = []
    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)
    original = deepcopy(fixture["task"])
    original_trace = reference.solve(original)
    check("own_frozen_manual_full_trace", reference.typed_equal(original_trace["new_events"], fixture["manually_enumerated_full_events"]))
    check("own_frozen_manual_terminal", original_trace["checkpoint"]["state"] == fixture["manual_terminal_state"])
    literal_changed = deepcopy(original)
    def literals(value):
        if type(value) is dict:
            if set(value) == {"const"}:
                value["const"] += 7
            else:
                for item in value.values(): literals(item)
        elif type(value) is list:
            for item in value: literals(item)
    literals(literal_changed)
    alpha = deepcopy(original)
    option_map = {name: f"renamed_option_{position}" for position, name in enumerate(reversed(list(original["options"])))}
    param_maps = {name: {parameter: f"renamed_parameter_{position}" for position, parameter in enumerate(option["params"])} for name, option in original["options"].items()}
    reg_map = {"x": "y", "y": "x"}
    def expression(value, owner):
        if "reg" in value: return {"reg": reg_map[value["reg"]]}
        if "param" in value: return {"param": param_maps[owner][value["param"]]}
        return deepcopy(value)
    def predicate(value, owner):
        return {"all": [{"left": expression(atom["left"], owner), "cmp": atom["cmp"], "right": expression(atom["right"], owner)} for atom in value["all"]]}
    def body(instructions, owner):
        result = []
        for instruction in instructions:
            op = instruction["op"]
            if op in ("add", "scale"):
                result.append({"op": op, "register": reg_map[instruction["register"]], "value": expression(instruction["value"], owner)})
            elif op == "swap": result.append({"op": "swap"})
            elif op == "if": result.append({"op": "if", "test": predicate(instruction["test"], owner), "then": body(instruction["then"], owner), "else": body(instruction["else"], owner)})
            else:
                callee = instruction["option"]
                result.append({"op": "call", "option": option_map[callee], "args": {param_maps[callee][key]: expression(value, owner) for key, value in instruction["args"].items()}})
        return result
    alpha["options"] = {option_map[name]: {"params": [param_maps[name][parameter] for parameter in option["params"]],
        "initiation": predicate(option["initiation"], name), "success": predicate(option["success"], name), "body": body(option["body"], name)} for name, option in original["options"].items()}
    root = original["entry"]["option"]
    alpha["entry"] = {"option": option_map[root], "args": {param_maps[root][key]: expression(value, root) for key, value in original["entry"]["args"].items()}}
    alpha["initial_state"] = {reg_map[key]: value for key, value in original["initial_state"].items()}
    operator_changed = deepcopy(original)
    operator_changed["options"]["start"]["body"][0]["op"] = "scale"
    order_changed = deepcopy(original)
    order_changed["options"]["start"]["body"][0:2] = list(reversed(order_changed["options"]["start"]["body"][0:2]))
    unused = deepcopy(original)
    unused["options"]["unused"] = {"params": ["u"], "initiation": {"all": []}, "success": {"all": []}, "body": [{"op": "swap"}]}
    views = ("full_option", "shared_primitive_declared", "shared_primitive_reachable")
    for view in views:
        base = normalizer.signature(original, view)
        check(view + "_numeric_literals_erased", base == normalizer.signature(literal_changed, view))
        check(view + "_option_parameter_register_alpha_invariance", base == normalizer.signature(alpha, view))
        check(view + "_recursive_object_field_order_invariance", base == normalizer.signature(generator.reverse_fields(original), view))
        check(view + "_operator_preserved", base != normalizer.signature(operator_changed, view))
        check(view + "_ordered_call_effect_preserved", base != normalizer.signature(order_changed, view))
        check(view + "_unused_scope", (base == normalizer.signature(unused, view)) == (view == "shared_primitive_reachable"))
    scalarized = deepcopy(original)
    def scalar_parameters(value):
        if type(value) is dict:
            if set(value) == {"param"}:
                value.clear(); value["const"] = 2
            else:
                for item in value.values(): scalar_parameters(item)
        elif type(value) is list:
            for item in value: scalar_parameters(item)
    scalar_parameters(scalarized)
    check("full_preserves_parameter_def_use", normalizer.signature(original, "full_option") != normalizer.signature(scalarized, "full_option"))
    for view in views[1:]:
        check(view + "_constant_and_parameter_scalar_collapse", normalizer.signature(original, view) == normalizer.signature(scalarized, view))
    legacy = {"kind": "skills", "state": {"x": 1, "y": 2}, "procedures": {"b": [{"op": "add", "register": "x", "value": 2}]},
              "main": [{"op": "call", "name": "b"}, {"op": "if", "test": {"register": "y", "cmp": "ge", "value": 0}, "then": [{"op": "swap"}], "else": []}]}
    lifted = normalizer.lift_legacy(legacy)
    for view in views:
        check(view + "_legacy_lift_equivalence", normalizer.signature(legacy, view) == normalizer.signature(lifted, view))
    collision = normalizer.compare_actual({12: original, 13: alpha}, [])
    for view in views:
        check(view + "_mutual_alpha_collision_reported", collision[view]["new_unique_signatures"] == 1 and collision[view]["mutual_collision_groups"] == [[12, 13]])
    tasks, targets, evidence = generator.derive_contexts(original, 12, reference)
    check("hand_nine_contexts_all_required", set(tasks) == set(generator.CONTEXTS))
    check("hand_R_valid_suffix", reference.typed_equal(targets["R"]["new_events"], original_trace["new_events"][evidence["first_cut_events"]:]))
    check("hand_restart_committed_replay_nonidempotence", evidence["replay"]["original_once_state"] != evidence["replay"]["original_twice_state"])
    check("hand_A_actual_later_operand_dependency", evidence["argument_operand_counterfactual"]["F_effect"]["value"] != evidence["argument_operand_counterfactual"]["A_effect"]["value"])
    check("hand_single_atom_I_and_S", evidence["initiation_atom_truth"]["I"].count(False) == 1 and evidence["success_atom_truth_S"].count(False) == 1)
    check("hand_X_active_scalar_tamper_rejected", targets["X"]["error"] == "invalid_checkpoint" and len(generator.differences(tasks["R"], tasks["X"])) == 1)
    check("hand_P_additional_effect_pending_and_U_suffix", targets["P"]["status"] == "paused" and targets["P"]["new_effects"] >= 1 and targets["U"]["status"] == "completed")
    return {"scope": "OWN_HAND_CALIBRATION_NOT_COHORT_APPROVAL", "checks": checks, "passed": len(checks), "failed": 0,
            "reuse_disclosure": "Own frozen e1 manual feasibility example reused; this does not establish12actual novel families."}


def calibrate_actual_interface(corpus, reference, generator, common, source_bindings):
    """Future own positive/negative metadata probes before writing candidate."""
    checks = []
    def run(name, edit=None, valid=True):
        rows, manifest = deepcopy(corpus["rows"]), deepcopy(corpus["manifest"])
        if edit is not None: edit(rows, manifest)
        # Refresh acyclic custody after explicit hand metadata mutations.
        for row in rows: row["record_hash"] = common.record_hash(row)
        raw = ("".join(common.canonical(row) + "\n" for row in rows)).encode()
        manifest["candidate_sha256"] = generator.sha(raw)
        lookup = {row["sample_id"]: row for row in rows}
        for entry in manifest["control_pairs"]:
            entry["candidate_sha256"] = generator.sha(raw)
            if entry["pair_sample_ids"] and all(item in lookup for item in entry["pair_sample_ids"]):
                if "pair_record_hashes" in entry: entry["pair_record_hashes"] = [lookup[item]["record_hash"] for item in entry["pair_sample_ids"]]
        try:
            generator.manifest_validate(rows, manifest, raw, reference, source_bindings)
            accepted = True
        except (generator.NeedNewPublicAmendment, KeyError, ValueError, TypeError):
            accepted = False
        result = accepted == valid
        checks.append({"name": name, "passed": result, "mutant_accepted": accepted})
        if not result: raise AssertionError(name)
    run("actual_positive_all216_nonmember_owners_and300_pairs")
    def property_only(rows, manifest):
        for row in rows:
            for claim in row["family_control_claims"]: claim.pop("name")
        for entry in manifest["control_pairs"]: entry["property"] = entry.pop("name")
    run("property_only_identifiers_positive", property_only)
    run("external_member_hashes_optional", lambda rows, manifest: [entry.pop("pair_record_hashes") for entry in manifest["control_pairs"]])
    def shared_ids(rows, manifest):
        mapping = {}
        for entry in manifest["control_pairs"]:
            if entry["name"] in ("resume_restart_replay", "budget_pause_after_effect"):
                replacement = entry["source_family_key"] + ":" + entry["renderers"][0] + ":shared"
                mapping[entry["control_id"]] = replacement; entry["control_id"] = replacement
        for row in rows:
            for claim in row["family_control_claims"]: claim["control_id"] = mapping.get(claim["control_id"], claim["control_id"])
    run("shared_control_id_across_distinct_names_positive", shared_ids)
    def empty_sample(rows, manifest):
        previous = rows[0]["sample_id"]; rows[0]["sample_id"] = ""
        for row in rows:
            for claim in row["family_control_claims"]: claim["pair_sample_ids"] = ["" if value == previous else value for value in claim["pair_sample_ids"]]
        for entry in manifest["control_pairs"]: entry["pair_sample_ids"] = ["" if value == previous else value for value in entry["pair_sample_ids"]]
    run("distinct_existing_empty_sample_id_is_allowed", empty_sample)
    run("canonical_pair_record_hashes_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0].update(pair_record_hashes=["0" * 64, "1" * 64]), False)
    run("ignored_family_claims_alias_forbidden", lambda rows, manifest: rows[0].update(family_claims=[]), False)
    run("boolean_source_index_forbidden", lambda rows, manifest: rows[0].update(source_family_index=True), False)
    run("empty_control_id_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0].update(control_id=""), False)
    run("nonstring_control_id_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0].update(control_id=1), False)
    run("reversed_canonical_pair_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0]["pair_sample_ids"].reverse(), False)
    run("missing_claim_forbidden", lambda rows, manifest: rows[0]["family_control_claims"].pop(), False)
    run("missing_named_entry_forbidden", lambda rows, manifest: manifest["control_pairs"].pop(), False)
    run("duplicate_complete_resolver_key_forbidden", lambda rows, manifest: manifest["control_pairs"].append(deepcopy(manifest["control_pairs"][0])), False)
    run("wrong_owner_renderer_forbidden", lambda rows, manifest: rows[0].update(renderer="reverse"), False)
    run("wrong_manifest_contexts_forbidden", lambda rows, manifest: manifest["control_pairs"][0].update(controlled_contexts=["X", "F"]), False)
    run("wrong_optional_claim_coordinates_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0].update(source_family_index=999), False)
    run("parent_traversal_evidence_forbidden", lambda rows, manifest: rows[0]["family_control_claims"][0].update(evidence_manifest="../hidden.json"), False)
    run("target_mutation_forbidden", lambda rows, manifest: rows[0]["verified_target"].update(new_effects=0), False)
    # Direct probes leave asserted custody intentionally wrong (no refresh).
    raw = corpus["candidate_raw"]
    def direct(name, edit):
        manifest = deepcopy(corpus["manifest"])
        edit(manifest)
        try:
            generator.manifest_validate(corpus["rows"], manifest, raw, reference, source_bindings)
            rejected = False
        except (generator.NeedNewPublicAmendment, KeyError, ValueError, TypeError): rejected = True
        checks.append({"name": name, "passed": rejected})
        if not rejected: raise AssertionError(name)
    direct("wrong_whole_candidate_SHA_forbidden", lambda manifest: manifest.update(candidate_sha256="0" * 64))
    direct("wrong_public_source_SHA_forbidden", lambda manifest: manifest["source_bindings"].update({next(iter(source_bindings)): "0" * 64}))
    direct("missing_public_source_forbidden", lambda manifest: manifest["source_bindings"].pop(next(iter(source_bindings))))
    direct("optional_external_member_SHA_wrong_forbidden", lambda manifest: manifest["control_pairs"][0].update(pair_record_hashes=["0" * 64, "1" * 64]))
    direct("optional_external_member_SHA_reordered_forbidden", lambda manifest: manifest["control_pairs"][0]["pair_record_hashes"].reverse())
    direct("optional_external_member_SHA_boolean_forbidden", lambda manifest: manifest["control_pairs"][0].update(pair_record_hashes=[True, False]))
    direct("optional_external_member_SHA_missing_element_forbidden", lambda manifest: manifest["control_pairs"][0]["pair_record_hashes"].pop())
    return {"scope": "PRODUCER_ACTUAL_INTERFACE_MUTATION_CALIBRATION_NO_APPROVAL", "checks": checks, "passed": len(checks), "failed": 0}
