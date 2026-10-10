#!/opt/anaconda3/bin/python3
"""New blind memory-gap-v1 audit using only prompt-derived expert code.

Imports the fresh preparation module, no producer/reference/oracle modules.
This first phase writes independent facts; judgments are frozen only after
the actual agent has inspected the resulting substantive examples.
"""
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from independent_memory_gap_v1 import (
    BASE, EXECUTION_ID, encode, exact_equal, solve_scan, solve_sorted,
    public_prompt_task, result_from_event, metamorphic_checks, query_diagnostics,
)

BATCH = BASE / "data/REM-001/candidate_renderings/memory-gap-v1.jsonl"
OUT = BASE / "reviews/expert"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def typed_value(value):
    return "null" if value is None else {bool: "boolean", int: "integer", str: "string"}[type(value)]


def answer_payload(result):
    return (result["status"], typed_value(result["value"]), encode(result["value"]))


def event_payload(event):
    return ("deleted", "null", "null") if event["op"] == "delete" else ("present", typed_value(event["value"]), encode(event["value"]))


def rank(events, index):
    return (events[index]["version"], events[index]["at"], index)


def witnesses(task, answer):
    events, queries = task["events"], task["queries"]
    names = ["decisive_delayed_boundary", "version_over_recency", "equal_version_time_conflict",
             "equal_version_time_input_conflict", "authoritative_tombstone", "present_null_distinction",
             "future_event_filter_control", "cross_key_filter_control", "absence_control"]
    found = {name: [] for name in names}
    for qi, (query, result) in enumerate(zip(queries, answer["results"])):
        eligible = [i for i, e in enumerate(events) if e["key"] == query["key"] and e["at"] <= query["as_of"]]
        winner = result["chosen_event"]
        base = {"query_position": qi, "query_id": query["query_id"], "signature": [query["key"], query["as_of"]],
                "winner_index": winner, "answer": list(answer_payload(result))}
        if winner is None:
            exists = [i for i, e in enumerate(events) if e["key"] == query["key"]]
            found["absence_control"].append(dict(base, absence_kind="future_only" if exists else "missing_key", key_event_indices=exists))
        else:
            chosen = events[winner]
            latest = max(eligible, key=lambda i: (events[i]["at"], i))
            if chosen["version"] > events[latest]["version"] and chosen["at"] < events[latest]["at"] and event_payload(events[latest]) != answer_payload(result):
                found["version_over_recency"].append(dict(base, winner_tuple=list(rank(events, winner)), latest_index=latest,
                                                        latest_tuple=list(rank(events, latest)), incorrect_latest_answer=list(event_payload(events[latest]))))
            older = [i for i in eligible if events[i]["version"] == chosen["version"] and events[i]["at"] < chosen["at"]
                     and event_payload(events[i]) != answer_payload(result)]
            if older:
                found["equal_version_time_conflict"].append(dict(base, winner_tuple=list(rank(events, winner)), older_competitors=older))
            tied = [i for i in eligible if events[i]["version"] == chosen["version"] and events[i]["at"] == chosen["at"]
                    and i != winner and event_payload(events[i]) != answer_payload(result)]
            if tied:
                found["equal_version_time_input_conflict"].append(dict(base, winner_tuple=list(rank(events, winner)), conflicting_ties=tied))
            lower_puts = [i for i in eligible if events[i]["op"] == "put" and events[i]["version"] < chosen["version"]]
            if chosen["op"] == "delete" and lower_puts:
                found["authoritative_tombstone"].append(dict(base, winner_tuple=list(rank(events, winner)), eligible_lower_version_puts=lower_puts))
        all_key = [i for i, e in enumerate(events) if e["key"] == query["key"]]
        if all_key:
            unfiltered = max(all_key, key=lambda i: rank(events, i))
            if events[unfiltered]["at"] > query["as_of"] and event_payload(events[unfiltered]) != answer_payload(result):
                found["future_event_filter_control"].append(dict(base, unfiltered_index=unfiltered, unfiltered_tuple=list(rank(events, unfiltered)),
                                                                 incorrect_answer=list(event_payload(events[unfiltered]))))
        all_eligible = [i for i, e in enumerate(events) if e["at"] <= query["as_of"]]
        if all_eligible:
            unkeyed = max(all_eligible, key=lambda i: rank(events, i))
            if events[unkeyed]["key"] != query["key"] and event_payload(events[unkeyed]) != answer_payload(result):
                found["cross_key_filter_control"].append(dict(base, unkeyed_index=unkeyed, unkeyed_tuple=list(rank(events, unkeyed)),
                                                              incorrect_answer=list(event_payload(events[unkeyed]))))
        for before_index, before_query in enumerate(queries):
            if query["as_of"] >= 1 and before_query["key"] == query["key"] and before_query["as_of"] == query["as_of"] - 1 and winner is not None:
                if events[winner]["at"] == query["as_of"] and answer_payload(result) != answer_payload(answer["results"][before_index]):
                    found["decisive_delayed_boundary"].append(dict(base, before_query_position=before_index,
                        before_answer=list(answer_payload(answer["results"][before_index])), boundary=query["as_of"], winner_tuple=list(rank(events, winner))))
    nulls = [i for i, r in enumerate(answer["results"]) if r["status"] == "present" and r["value"] is None]
    other_nulls = [i for i, r in enumerate(answer["results"]) if r["status"] in ("absent", "deleted") and r["value"] is None]
    if nulls and other_nulls:
        found["present_null_distinction"].append({"put_null_query_positions": nulls, "absent_or_deleted_query_positions": other_nulls})
    return found


def role_map(record, task):
    assignments = record["query_role_assignments"]
    assert len(assignments) == len(task["queries"])
    roles = {}
    for pos, (query, assignment) in enumerate(zip(task["queries"], assignments)):
        assert assignment["position"] == pos and assignment["query_id"] == query["query_id"]
        assert assignment["role"] not in roles
        roles[assignment["role"]] = (pos, query)
    expected = {"absent_key", "delayed_older_later", "delayed_before", "index_tie_at", "delayed_at", "put_null_at", "tombstone_at", "lifecycle_before"}
    assert set(roles) == expected
    groups = record["key_role_assignments"]
    for name, (pos, query) in roles.items():
        group = "absent" if name == "absent_key" else "delayed" if name.startswith("delayed") else "index_tie" if name == "index_tie_at" else "lifecycle"
        assert query["key"] == groups[group]
    assert roles["delayed_at"][1]["as_of"] == roles["delayed_before"][1]["as_of"] + 1
    assert roles["tombstone_at"][1]["as_of"] == roles["lifecycle_before"][1]["as_of"] + 1
    assert roles["put_null_at"][1]["as_of"] == roles["tombstone_at"][1]["as_of"] + 1
    return roles


def controlled_probes(task, answer):
    probes = []
    for qi, (query, result) in enumerate(zip(task["queries"], answer["results"])):
        winner = result["chosen_event"]
        if winner is None:
            continue
        def evaluate(name, changed, expect_change, changed_field, event_index=None):
            new = solve_scan(changed)[0]["results"][qi]
            assert exact_equal(solve_scan(changed)[0], solve_sorted(changed))
            payload_changed = answer_payload(new) != answer_payload(result)
            passed = payload_changed if expect_change else exact_equal(new, result)
            if not passed:
                raise AssertionError(name)
            probes.append({"scope": "audit_probe", "control": name, "query_position": qi,
                           "query_id": query["query_id"], "changed_field": changed_field,
                           "event_index": event_index, "old_result": result, "new_result": new,
                           "semantic_answer_changed": payload_changed, "passed": passed})
        if result["chosen_version"] > 0:
            changed = copy.deepcopy(task)
            changed["events"].append({"key": query["key"], "version": result["chosen_version"] - 1,
                                      "at": query["as_of"], "op": "put", "value": "expert-appended-low"})
            evaluate("lower_version_append_invariance", changed, False, "append eligible lower-version same-key event")
        changed = copy.deepcopy(task)
        other_indices = [i for i,e in enumerate(task["events"]) if e["key"] != query["key"]]
        if other_indices:
            for i in other_indices:
                changed["events"][i]["version"] += 1000
                changed["events"][i]["op"], changed["events"][i]["value"] = "put", "expert-other-key"
            evaluate("other_key_edit_invariance", changed, False, "other-key version/op/value edits")
        future_indices = [i for i,e in enumerate(task["events"]) if e["at"] > query["as_of"]]
        if future_indices:
            changed = copy.deepcopy(task)
            for i in future_indices:
                changed["events"][i]["version"] += 1000
                changed["events"][i]["op"], changed["events"][i]["value"] = "put", "expert-future"
            evaluate("future_event_edit_invariance", changed, False, "strictly-future version/op/value edits")
        competitors = [i for i,e in enumerate(task["events"]) if e["key"] == query["key"] and e["at"] <= query["as_of"]
                       and event_payload(e) != answer_payload(result)]
        if competitors:
            index = competitors[0]
            changed = copy.deepcopy(task)
            changed["events"][index]["version"] = max(e["version"] for e in task["events"] if e["key"] == query["key"] and e["at"] <= query["as_of"]) + 1
            evaluate("version_priority_counterfactual", changed, True, "one eligible competing event version", index)
        ties = [i for i in competitors if task["events"][i]["version"] == task["events"][winner]["version"]
                and task["events"][i]["at"] == task["events"][winner]["at"]]
        if ties:
            index = ties[0]
            changed = copy.deepcopy(task)
            changed["events"][winner], changed["events"][index] = changed["events"][index], changed["events"][winner]
            evaluate("active_input_tie_swap_counterfactual", changed, True, "swap only two top tuple events", index)
        if task["events"][winner]["value"] is None:
            changed = copy.deepcopy(task)
            changed["events"][winner]["op"] = "put" if changed["events"][winner]["op"] == "delete" else "delete"
            evaluate("null_lifecycle_counterfactual", changed, True, "one winning null-valued event op", winner)
        for key in sorted({q["key"] for q in task["queries"]} - {query["key"]}):
            changed = copy.deepcopy(task)
            changed["queries"][qi]["key"] = key
            new = solve_scan(changed)[0]["results"][qi]
            if answer_payload(new) != answer_payload(result):
                evaluate("query_key_counterfactual", changed, True, "only queried key")
                break
    return probes


def hist_add(bucket, role, result):
    bucket["role"][role] += 1
    bucket["status"][result["status"]] += 1
    bucket["type"][typed_value(result["value"])] += 1
    bucket["present_null"][str(result["status"] == "present" and result["value"] is None)] += 1


def empty_hist():
    return {kind: Counter() for kind in ("role", "status", "type", "present_null")}


def run():
    public_text = (BASE / "schemas/EXTENDED_SEMANTICS.md").read_text()
    properties_path = BASE / "schemas/MEMORY_CONTROL_PROPERTIES.md"
    # Bind the exact public design without importing any producer/report code.
    properties_path.read_text()
    records = [json.loads(line) for line in BATCH.read_text().splitlines()]
    evidence, by_id, history_groups, renderer_groups = [], {}, defaultdict(list), defaultdict(list)
    for record in records:
        sample = record["sample_id"]
        assert sample not in by_id
        computed_hash = hashlib.sha256(encode({k:v for k,v in record.items() if k != "record_hash"}).encode()).hexdigest()
        assert computed_hash == record["record_hash"]
        task = public_prompt_task(record["rendered_input"], public_text)
        # Derive before canonical gold comparison.
        answer, traces = solve_scan(task)
        secondary = solve_sorted(task)
        checks = {"canonical_hash": True,
                  "full_contract_and_valid_task": True,
                  "prompt_payload_exact": exact_equal(task, record["semantic_task"]),
                  "own_solver_agreement": exact_equal(answer, secondary),
                  "full_target_typed_exact": exact_equal(answer, record["verified_target"])}
        roles = role_map(record, task)
        row = {"sample_id": sample, "record_hash": computed_hash, "source_family_key": record["source_family_key"],
               "controlled_variant": record["controlled_variant"], "renderer": record["renderer"],
               "query_rotation": record["query_rotation"], "checks": checks,
               "own_answer": answer, "selection_traces": traces,
               "public_property_witnesses": witnesses(task, answer),
               "query_diagnostics": query_diagnostics(task),
               "metamorphics": metamorphic_checks(task),
               "audit_probes": controlled_probes(task, answer),
               "status_histogram": dict(Counter(r["status"] for r in answer["results"])),
               "present_null_queries": sum(r["status"] == "present" and r["value"] is None for r in answer["results"]),
               "put_event_count": sum(e["op"] == "put" for e in task["events"]),
               "delete_event_count": sum(e["op"] == "delete" for e in task["events"])}
        row["role_results"] = {role:{"query": query, "result": answer["results"][pos]} for role,(pos,query) in roles.items()}
        evidence.append(row)
        by_id[sample] = (record, task, answer, row)
        history_groups[(record["source_family_key"], record["controlled_variant"], record["renderer"])].append((record, task, answer, row))
        renderer_groups[(record["source_family_key"], record["controlled_variant"], record["query_rotation"])].append((record,task,answer,row))
    assert all(all(row["checks"].values()) for row in evidence), "Exact target/hash/contract check failed"
    crossings = []
    key_hist, family_key_hist = defaultdict(empty_hist), defaultdict(lambda: defaultdict(empty_hist))
    for (family, variant, style), group in sorted(history_groups.items()):
        ids, positions = defaultdict(empty_hist), defaultdict(empty_hist)
        signatures = defaultdict(set)
        assert len(group) == 8 and {r[0]["query_rotation"] for r in group} == set(range(8))
        for record,task,answer,row in group:
            for role,(pos,query) in role_map(record, task).items():
                result = answer["results"][pos]
                hist_add(ids[query["query_id"]], role, result)
                hist_add(positions[str(pos)], role, result)
                hist_add(key_hist[query["key"]], role, result)
                hist_add(family_key_hist[family][query["key"]], role, result)
                signatures[role].add((query["key"],query["as_of"]))
        assert all(len(s) == 1 for s in signatures.values())
        id_values, pos_values = list(ids.values()), list(positions.values())
        ids_equal = all(exact_equal(h,id_values[0]) for h in id_values)
        positions_equal = all(exact_equal(h,pos_values[0]) for h in pos_values)
        all_roles_once = all(set(h["role"]) == set(signatures) and set(h["role"].values()) == {1} for h in id_values + pos_values)
        crossing = {"source_family_key": family, "controlled_variant": variant, "field_order": style,
                    "scope": "corpus_variant", "id_histograms": ids, "position_histograms": positions,
                    "role_signatures": {k: sorted(v) for k,v in signatures.items()},
                    "ids_equal": ids_equal, "positions_equal": positions_equal, "each_role_once": all_roles_once,
                    "all_three_statuses": set(id_values[0]["status"]) == {"present","deleted","absent"},
                    "query_id_position_mutually_independent": False}
        crossings.append(crossing)
    renderer_checks = []
    for key, group in sorted(renderer_groups.items()):
        assert len(group) == 2 and {r[0]["renderer"] for r in group} == {"forward","reverse"}
        forward = next(r for r in group if r[0]["renderer"] == "forward")
        reverse = next(r for r in group if r[0]["renderer"] == "reverse")
        assert exact_equal(forward[1], reverse[1]) and exact_equal(forward[2], reverse[2])
        prefix_a, encoded_a = forward[0]["rendered_input"].split("Task JSON:\n")
        prefix_b, encoded_b = reverse[0]["rendered_input"].split("Task JSON:\n")
        assert prefix_a == prefix_b and encoded_a != encoded_b
        renderer_checks.append({"samples": [forward[0]["sample_id"], reverse[0]["sample_id"]],
                                "scope": "corpus_variant", "control": "field_order_invariance", "passed": True})
    key_assignments, pairs = [], []
    families = sorted({r["source_family_key"] for r in records})
    for family in families:
        controlled = {r["controlled_variant"]:(r,t,a,e) for r,t,a,e in by_id.values() if r["source_family_key"] == family and r["query_rotation"] == 0 and r["renderer"] == "forward"}
        assert set(controlled) == {"baseline","delayed_tombstone","tombstone_to_put_null","index_tie_swap"}
        assignments = {v:r[0]["key_role_assignments"] for v,r in controlled.items()}
        strings = sorted({s for roles in assignments.values() for s in roles.values()})
        role_by_key = defaultdict(Counter)
        for roles in assignments.values():
            for role,key in roles.items():
                role_by_key[key][role] += 1
        full_key_cross = len(strings) == 4 and all(set(h) == {"delayed","lifecycle","index_tie","absent"} and set(h.values()) == {1} for h in role_by_key.values())
        key_assignments.append({"source_family_key": family, "scope": "corpus_variant", "four_histories": assignments,
                                "role_by_key": role_by_key, "full_four_role_crossing": full_key_cross,
                                "outcome_histograms_by_key": family_key_hist[family]})
        base = controlled["baseline"]
        for variant in ("delayed_tombstone","tombstone_to_put_null","index_tie_swap"):
            other = controlled[variant]
            diffs = [i for i,(a,b) in enumerate(zip(base[1]["events"], other[1]["events"])) if not exact_equal(a,b)]
            assert len(base[1]["events"]) == len(other[1]["events"])
            assert exact_equal(base[1]["queries"], other[1]["queries"])
            changed_roles = [role for role in base[3]["role_results"] if answer_payload(base[3]["role_results"][role]["result"]) != answer_payload(other[3]["role_results"][role]["result"])]
            pairs.append({"source_family_key": family, "scope": "corpus_variant", "baseline_sample": base[0]["sample_id"],
                          "variant_sample": other[0]["sample_id"], "variant": variant,
                          "event_differences": [{"index":i,"before":base[1]["events"][i],"after":other[1]["events"][i]} for i in diffs],
                          "changed_answer_roles": changed_roles})
    unique_history = {encode(t["events"]) for _,t,_,_ in by_id.values()}
    unique_observations = {(encode(t["events"]),q["key"],q["as_of"]) for _,t,_,_ in by_id.values() for q in t["queries"]}
    report = {"reviewer_execution_id": EXECUTION_ID, "stage": "INDEPENDENT_FACTS_BEFORE_FIRST_JUDGMENTS",
              "source_file": str(BATCH), "source_sha256": sha(BATCH), "records":len(records),
              "source_families":len(families), "controlled_histories":len(unique_history),
              "base_history_query_observations":len(unique_observations), "rotation_level_observations":len(records)//2*8,
              "rendered_observations":len(records)*8,
              "split_record_counts":dict(Counter(r["split"] for r in records)),
              "family_split_counts":dict(Counter(next(r["split"] for r in records if r["source_family_key"] == f) for f in families)),
              "source_family_splits_consistent":all(len({r["split"] for r in records if r["source_family_key"] == f}) == 1 for f in families),
              "difficulty_counts":dict(Counter(r["difficulty"] for r in records)),
              "event_count_counts":dict(Counter(len(t["events"]) for _,t,_,_ in by_id.values())),
              "full_target_hash_contract_checks":len(evidence),
              "property_row_counts":{name:sum(bool(e["public_property_witnesses"][name]) for e in evidence) for name in evidence[0]["public_property_witnesses"]},
              "status_query_counts":dict(Counter(result["status"] for e in evidence for result in e["own_answer"]["results"])),
              "type_query_counts":dict(Counter(typed_value(result["value"]) for e in evidence for result in e["own_answer"]["results"])),
              "missing_key_absence_queries":sum(w["absence_kind"] == "missing_key" for e in evidence for w in e["public_property_witnesses"]["absence_control"]),
              "future_only_absence_queries":sum(w["absence_kind"] == "future_only" for e in evidence for w in e["public_property_witnesses"]["absence_control"]),
              "all_three_status_history_style_counts":dict(Counter(c["all_three_statuses"] for c in crossings)),
              "query_crossings_complete":all(c["ids_equal"] and c["positions_equal"] and c["each_role_once"] for c in crossings),
              "key_crossings_complete":all(c["full_four_role_crossing"] for c in key_assignments),
              "crossings":crossings, "key_role_crossings":key_assignments, "global_key_histograms":key_hist,
              "renderer_checks":renderer_checks, "corpus_counterfactual_pairs":pairs,
              "metamorphic_checks":sum(len(e["metamorphics"]) for e in evidence),
              "audit_probe_count":sum(len(e["audit_probes"]) for e in evidence),
              "audit_probe_control_counts":dict(Counter(p["control"] for e in evidence for p in e["audit_probes"])),
              "scripts_sha256":{str(Path(__file__).resolve()):sha(__file__), str(OUT/"independent_memory_gap_v1.py"):sha(OUT/"independent_memory_gap_v1.py")},
              "public_contract_sha256":sha(BASE/"schemas/EXTENDED_SEMANTICS.md"),
              "public_properties_sha256":sha(properties_path),
              "read_scope":"Candidate memory-gap-v1 only, the two public contracts, and new own prepared/audit code. No other verdicts, prior candidates, producer/generator/reference/oracles/coverage reports.",
              "row_evidence":evidence}
    with (OUT/"memory-gap-v1.independent_analysis.json").open("x") as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    summary = {k:v for k,v in report.items() if k not in ("crossings","key_role_crossings","global_key_histograms","renderer_checks","corpus_counterfactual_pairs","row_evidence")}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    run()
