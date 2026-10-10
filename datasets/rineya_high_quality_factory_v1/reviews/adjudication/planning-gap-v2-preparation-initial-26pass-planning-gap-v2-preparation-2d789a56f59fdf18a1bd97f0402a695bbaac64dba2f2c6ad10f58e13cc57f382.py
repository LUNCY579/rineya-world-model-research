"""Public-only V2 adjudication preparation helpers; never reads production or emits decisions."""
from collections import Counter
import datetime
import hashlib
import json
import re

PUBLIC_VERSION = "rgm-planning-controls-2.0.0"
PUBLIC_DOCS = (
    "schemas/SEMANTICS.md",
    "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md",
    "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md",
    "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md",
    "schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md",
    "research/curricula/RGM-PLANNING-GAP-V2-PROSPECTIVE.md",
    "schemas/RGM_PLANNING_V2_RENDERER_CONTROL_ENCODING_ADDENDUM.md",
)
GUARDS = ("source_exists", "source_clear", "destination_exists_or_table",
          "source_destination_distinct", "destination_clear_or_table", "changed_support")
PAIRS = {
    "A": [("fuel_only_counterfactual", "A0", "A1")],
    "B": [("directed_edge_deletion_counterfactual", "B0", "B1"), ("same_world_valid_goal_detour", "B0", "B3")],
    "C": [("adjacent_action_order_counterfactual", "C0", "C1"), ("same_world_valid_goal_detour", "C0", "C2"),
          ("isolated_noop_append_counterfactual", "C3", "C4")],
    "D": [("partial_goal_extension_counterfactual", "D0", "D1"), ("same_world_valid_goal_detour", "D2", "D3")],
}


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def typed_equal(left, right):
    return canonical(left) == canonical(right)


def digest(raw):
    require(type(raw) is bytes, "hash input must be bytes")
    return hashlib.sha256(raw).hexdigest()


def verify_bytes(raw, expected):
    require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "invalid expected hash")
    require(digest(raw) == expected, "exact bytes changed")


def record_digest(record):
    return digest(canonical({key: value for key, value in record.items() if key != "record_hash"}).encode())


def pending_record_index(records, expected_count):
    require(type(expected_count) is int and expected_count >= 1, "invalid expected count")
    require(type(records) is list and len(records) == expected_count, "record count")
    result = {}
    for record in records:
        sid = record.get("sample_id")
        require(type(sid) is str and sid and sid not in result, "duplicate/invalid sample ID")
        require(record.get("review_status") == "PENDING", "producer candidate status must remain PENDING")
        require(record.get("record_hash") == record_digest(record), "full canonical record hash")
        result[sid] = record
    return result


def exact_review_join(index, receipts, expected_role, expected_execution_id):
    """Caller supplies trusted actual role identity after completed evidence is authorized."""
    require(type(expected_role) is str and expected_role, "missing expected role")
    require(type(expected_execution_id) is str and expected_execution_id, "missing actual reviewer identity")
    require(type(receipts) is list and len(receipts) == len(index), "receipt completeness")
    result, receipt_ids = {}, set()
    for receipt in receipts:
        sid, rid = receipt.get("sample_id"), receipt.get("receipt_id")
        require(sid in index and sid not in result, "unknown/duplicate reviewed sample")
        require(type(rid) is str and rid and rid not in receipt_ids, "duplicate/invalid receipt ID")
        require(receipt.get("record_hash") == index[sid]["record_hash"], "review hash drift")
        require(receipt.get("reviewer_role") == expected_role, "wrong reviewer role")
        require(receipt.get("reviewer_execution_id") == expected_execution_id, "wrong actual reviewer execution")
        require(expected_execution_id != index[sid].get("producer_agent_id"), "producer cannot review own record")
        require(receipt.get("decision") in ["ACCEPT", "REVISE", "REJECT", "QUARANTINE"], "unfinished/unknown final decision")
        confidence = receipt.get("confidence")
        require(type(confidence) in [int, float] and 0 <= confidence <= 1, "typed confidence")
        receipt_ids.add(rid)
        result[sid] = receipt
    require(set(result) == set(index), "exact sample coverage")
    return result


def completed_execution_attestation(attestation, candidate_hash, manifest_hash, execution_id):
    """Internal normalized evidence, not a new public/private report-key requirement.

    Later actual runner must inspect each report's real shape and preserve its raw binding
    before constructing this object. This preparation has no file parser or discovery.
    """
    require(attestation.get("completed_execution") is True, "execution incomplete")
    require(attestation.get("actual_execution_id") == execution_id, "execution identity drift")
    require(attestation.get("candidate_sha256") == candidate_hash, "execution candidate drift")
    require(attestation.get("manifest_sha256") == manifest_hash, "execution manifest drift")
    require(type(attestation.get("raw_receipt_sha256")) is str and
            re.fullmatch(r"[0-9a-f]{64}", attestation["raw_receipt_sha256"]) is not None, "raw receipt binding missing")
    require(attestation.get("first_predictions_frozen_before_target_comparison") is True, "first-freeze custody not established")
    # Deliberately returns no approval/decision. Counts and assertions are never target evidence.


def first_freeze_before_comparison(first_freeze_utc, comparison_start_utc):
    first, comparison = [datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                         for value in [first_freeze_utc, comparison_start_utc]]
    require(first.tzinfo is not None and comparison.tzinfo is not None, "timezone-aware actual execution timestamps required")
    require(first <= comparison, "comparison preceded first input freeze")


def public_binding_check(manifest, expected):
    require(manifest.get("version") == PUBLIC_VERSION and type(manifest.get("version")) is str, "V2 exact version")
    require(set(expected) == set(PUBLIC_DOCS), "seven authoritative public documents")
    bindings = manifest.get("public_specification_bindings")
    require(type(bindings) is list, "public binding list")
    indexed = {}
    for binding in bindings:
        require(type(binding) is dict, "binding object")
        path, value = binding.get("path"), binding.get("sha256")
        require(type(path) is str and path and path not in indexed, "duplicate/invalid public path")
        require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "typed public hash")
        indexed[path] = value
    require(all(indexed.get(path) == value for path, value in expected.items()), "missing/stale public hash")
    # Optional extra bindings are retained for later inspection, not rejected by a private closed shape.
    return indexed


def expected_allocation(index):
    require(type(index) is int and 180 <= index <= 227, "typed source index")
    stratum = "ABCD"[(index - 180) // 12]
    a_sizes = [5, 5, 5, 5, 6, 6, 7, 6, 6, 7, 7, 7]
    b_sizes = [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]
    return {"index": index, "stratum": stratum, "split": "dev_challenge" if index % 10 in [8, 9] else "train",
            "objects": a_sizes[index - 180] if stratum == "A" else b_sizes[index - 192] if stratum == "B" else 5 if index % 2 == 0 else 6,
            "a_subtype": ("fresh_infeasible" if index % 2 == 0 else "fuel_lengthened") if stratum == "A" else None,
            "c1_reason": ("destination_not_clear" if index in [207, 209, 210, 212, 213, 214] else "source_not_clear") if stratum == "C" else None,
            "c4_prefix": 1 + (index - 204) % 3 if stratum == "C" else None,
            "contexts": 5 if stratum == "C" else 4}


def guard_vector(state, action):
    block, destination = action["block"], action["to"]
    require(type(block) is str and type(destination) is str, "typed action names")
    clear = set(state) - set(state.values())
    return {"source_exists": block in state, "source_clear": block in clear,
            "destination_exists_or_table": destination == "TABLE" or destination in state,
            "source_destination_distinct": block != destination,
            "destination_clear_or_table": destination == "TABLE" or destination in clear,
            "changed_support": block in state and destination != state[block]}


def block_prefix_replay(task, omit_guard=None):
    """Public guard/prefix facts only; no shortest-target generator and no output writing."""
    require(omit_guard is None or omit_guard in GUARDS, "unknown omitted guard")
    state = dict(task["on"])
    states, steps = [dict(state)], []
    for action in task["plan"]:
        guards = guard_vector(state, action)
        legal = all(truth for guard, truth in guards.items() if guard != omit_guard)
        if legal:
            state[action["block"]] = action["to"]
        steps.append({"legal": legal, "guards": guards, "after": dict(state)})
        states.append(dict(state))
        if not legal:
            break
    return {"states": states, "steps": steps, "valid": all(step["legal"] for step in steps)}


def clearance(state, block):
    return block in state and block not in state.values()


def clearance_evidence_check(task, evidence):
    required = {"action_index", "precondition", "enabling_action_index", "obstructing_block"}
    require(type(evidence) is dict and required <= set(evidence), "four required V2 witness fields")
    i, j, code, obstruction = [evidence[key] for key in ["action_index", "enabling_action_index", "precondition", "obstructing_block"]]
    require(type(i) is int and type(j) is int and 0 <= j < i < len(task["plan"]) and i >= 1, "typed witness indices")
    require(type(code) is str and code in ["source_not_clear", "destination_not_clear"], "exact V2 witness enum")
    require(type(obstruction) is str and obstruction in task["on"], "existing obstruction")
    replay = block_prefix_replay(task)
    require(i < len(replay["steps"]) and all(step["legal"] for step in replay["steps"][:i + 1]), "executed legal witness prefix/action")
    action = task["plan"][i]
    expected = "source_not_clear" if not clearance(task["on"], action["block"]) else "destination_not_clear" if action["to"] != "TABLE" and not clearance(task["on"], action["to"]) else None
    require(code == expected, "initial canonical source-first witness selection")
    selected = action["block"] if code == "source_not_clear" else action["to"]
    require(task["on"][obstruction] == selected and task["plan"][j]["block"] == obstruction, "actual initial obstruction/enabler binding")
    require(not clearance(replay["states"][j], selected) and clearance(replay["states"][j + 1], selected), "actual false-to-true enabling transition")
    require(all(clearance(state, selected) for state in replay["states"][j + 1:i + 1]), "no intervening reblocking")
    return {"selected_object": selected, "initial_choice": code, "actual_enabler_index": j}


def selective_failure_check(task, evidence):
    require(type(evidence["swap_index"]) is int and 0 <= evidence["swap_index"] < len(task["plan"]) - 1, "typed swap index")
    require(type(evidence["first_failure_index"]) is int and evidence["first_failure_index"] >= 0, "typed failure index")
    reason = evidence["first_failure_reason"]
    require(type(reason) is str and reason in ["source_not_clear", "destination_not_clear"], "typed C1 reason")
    declared = evidence["guard_truth"]
    require(type(declared) is dict and set(declared) == set(GUARDS) and all(type(x) is bool for x in declared.values()), "exact six typed boolean guard values")
    strict = block_prefix_replay(task)
    first = next((i for i, step in enumerate(strict["steps"]) if not step["legal"]), None)
    require(first == evidence["first_failure_index"], "actual first-failure index")
    actual = strict["steps"][first]["guards"]
    require(typed_equal(declared, actual), "false guard annotation")
    selected = "source_clear" if reason == "source_not_clear" else "destination_clear_or_table"
    require([guard for guard, truth in actual.items() if not truth] == [selected], "failure not selective")
    changed = block_prefix_replay(task, selected)
    require(changed["steps"][first]["legal"] and not typed_equal(strict["steps"][first], changed["steps"][first]), "single-guard omission did not change executed step")
    return {"first_index": first, "selected_guard": selected, "mutated_step_executed": True}


def isolated_noop_check(task, evidence, source_index):
    k = expected_allocation(source_index)["c4_prefix"]
    require(k is not None and type(evidence["action_index"]) is int and evidence["action_index"] == k, "fixed C4 prefix index")
    block = evidence["block"]
    require(type(block) is str and evidence["current_support"] == "TABLE", "typed C4 source/support")
    require(len(task["plan"]) == k + 1 and task["plan"][k] == {"block": block, "to": "TABLE"}, "exact final appended no-op")
    strict = block_prefix_replay(task)
    require(len(strict["steps"]) == k + 1 and all(step["legal"] for step in strict["steps"][:k]), "legal complete C4 prefix")
    before = strict["states"][k]
    require(not all(before[b] == support for b, support in task["goal_on"].items()), "C4 prefix must be non-goal")
    guards = strict["steps"][k]["guards"]
    require([guard for guard, truth in guards.items() if not truth] == ["changed_support"], "no-op co-failure")
    require(typed_equal(strict["states"][k], strict["states"][k + 1]), "failed no-op changed state")
    require(block_prefix_replay(task, "changed_support")["valid"], "no-op omission did not change validity")
    for clearance_guard in ["source_clear", "destination_clear_or_table"]:
        require(typed_equal(strict, block_prefix_replay(task, clearance_guard)), "clearance omission changed isolated no-op")
    return {"first_index": k, "only_false_guard": "changed_support", "no_op_omission_changes_validity": True}


def control_renderer_bindings(families, index):
    """Only joins/coverage; semantic pair truth and actual renderer differences remain separate."""
    seen_ids, declaration_count = set(), 0
    for family in families:
        stratum = family["primary_stratum"]
        require(stratum in PAIRS, "unknown family stratum")
        context_map, styles = {}, set()
        for context in family["contexts"]:
            context_id = context["context_id"]
            require(context_id not in context_map, "duplicate context")
            ids = context["sample_ids"]
            require(type(ids) is list and len(ids) == len(set(ids)) == 2, "two renderer sample IDs per context")
            by_style = {}
            for sid in ids:
                require(sid in index, "unknown context sample")
                row = index[sid]
                require(row["source_family_key"] == family["source_family_key"] and row["split"] == family["split"], "context lineage/split drift")
                renderer = row["renderer"]
                require(type(renderer) is str and renderer and renderer not in by_style, "distinct nonempty context renderer values")
                by_style[renderer] = sid
            context_map[context_id] = by_style
            styles.update(by_style)
        require(len(styles) == 2 and all(set(values) == styles for values in context_map.values()), "style/context completeness")
        required = {(name, style, context_map[left][style], context_map[right][style])
                    for name, left, right in PAIRS[stratum] for style in styles}
        observed = set()
        for control in family["controls"]:
            cid = control["control_id"]
            require(type(cid) is str and cid and cid not in seen_ids, "globally distinct nonempty control ID")
            seen_ids.add(cid)
            sids = control["sample_ids"]
            require(type(sids) is list and len(sids) == len(set(sids)) == 2, "exact two-ID declaration")
            require(all(sid in index for sid in sids), "unknown control sample")
            left, right = [index[sid] for sid in sids]
            require(all(row["source_family_key"] == family["source_family_key"] and row["split"] == family["split"] for row in [left, right]), "control lineage/split drift")
            require(left["renderer"] == right["renderer"], "mixed renderer pair")
            require(control["scope"] == "corpus_variant" and control["provenance"] == "supplied_tasks", "supplied pair scope")
            observed.add((control["name"], left["renderer"], sids[0], sids[1]))
            declaration_count += 1
        require(required <= observed, "missing required per-renderer pair")
    return {"verified_pair_binding_declarations": declaration_count, "semantic_truth_checked": False, "decision_emitted": False}


if __name__ == "__main__":
    raise SystemExit("Preparation library only. Run its synthetic tests; actual adjudication requires a later authorized evidence-bundle execution.")
