#!/opt/anaconda3/bin/python3
"""Corrective-topic-informed, independent expert audit for new REM v2 bytes.

Only new candidate rows, public contracts and the expert's own earlier solvers
are read.  Producer witness metadata is verified after prompt-only derivation;
no producer control manifest, generator, reference, oracle or other verdict is
opened.  No v1 output or code is modified.
"""
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path

from independent_memory_gap_v1 import (
    BASE, EXECUTION_ID, encode, exact_equal, solve_scan, solve_sorted,
    public_prompt_task, metamorphic_checks, query_diagnostics,
)
from review_memory_gap_v1 import (
    witnesses, rank, event_payload, answer_payload, typed_value,
    controlled_probes, decode_key_roles,
)

OUT = BASE / "reviews/expert"
BATCH = BASE / "data/REM-001/candidate_renderings/memory-gap-v2.jsonl"
ROLES = {"delayed_before","delayed_later","delayed_at","lifecycle_present_null",
         "lifecycle_at","absent","same_version_time_index","lifecycle_delete"}
PROPERTIES = {"absence_control","authoritative_tombstone","cross_key_filter_control",
              "decisive_delayed_boundary","equal_version_time_conflict","equal_version_time_input_conflict",
              "future_event_filter_control","present_null_distinction","version_over_recency"}
SCOPES = {"query_role_id_position_crossing":"corpus_variant", "key_role_crossing":"corpus_variant",
          "field_order_invariance":"corpus_variant", "version_priority_counterfactual":"corpus_variant",
          "active_input_tie_swap_counterfactual":"corpus_variant", "null_lifecycle_counterfactual":"corpus_variant",
          "lower_version_append_invariance":"audit_probe", "other_key_edit_invariance":"audit_probe",
          "future_event_edit_invariance":"audit_probe"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_map(record,task):
    roles = {}
    for pos,(query,assignment) in enumerate(zip(task["queries"],record["query_role_assignments"])):
        assert assignment["position"] == pos and assignment["query_id"] == query["query_id"]
        assert assignment["role"] not in roles
        roles[assignment["role"]] = (pos,query)
    assert set(roles) == ROLES and len(task["queries"]) == len(record["query_role_assignments"]) == 8
    groups = record["key_role_assignments"]
    assert set(groups) == {"delayed","lifecycle","index_tie","absent"} and len(set(groups.values())) == 4
    for role,(pos,query) in roles.items():
        group = "absent" if role == "absent" else "index_tie" if role == "same_version_time_index" else "delayed" if role.startswith("delayed") else "lifecycle"
        assert query["key"] == groups[group]
    assert roles["delayed_at"][1]["as_of"] == roles["delayed_before"][1]["as_of"] + 1
    assert roles["lifecycle_present_null"][1]["as_of"] == roles["lifecycle_at"][1]["as_of"] + 1
    assert roles["lifecycle_delete"][1]["as_of"] < roles["lifecycle_at"][1]["as_of"]
    return roles


def validate_metadata_witnesses(record,task,answer,own_witnesses):
    declared = record["property_witnesses"]
    tagged = set(record["skill_tags"]) - {"finite_synthetic_proxy","versioned_retrieval"}
    assert tagged == set(declared) and tagged <= PROPERTIES
    assert all(own_witnesses[name] for name in tagged), "A tagged property lacks any independent witness"
    events,queries = task["events"],task["queries"]
    validated = []

    def atomic(item):
        matching = [i for i,q in enumerate(queries) if q["query_id"] == item["query_id"] and
                    exact_equal({"key":q["key"],"as_of":q["as_of"]},item["query_signature"])]
        assert len(matching) == 1
        qi = matching[0]
        query,result = queries[qi],answer["results"][qi]
        assert exact_equal(item["answer"],{k:result[k] for k in ("status","value")})
        winner = result["chosen_event"]
        assert exact_equal(item["winner_index"],winner)
        assert exact_equal(item["winner_tuple"],None if winner is None else list(rank(events,winner)))
        assert len(item["competitor_indices"]) == len(item["competitor_tuples"]) == len(item["competitor_answers"])
        for ci,ct,ca in zip(item["competitor_indices"],item["competitor_tuples"],item["competitor_answers"]):
            assert type(ci) is int and 0 <= ci < len(events)
            assert exact_equal(ct,list(rank(events,ci)))
            event = events[ci]
            payload = {"status":"deleted" if event["op"] == "delete" else "present",
                       "value":None if event["op"] == "delete" else event["value"]}
            assert exact_equal(payload,ca)
        return qi,query,result,winner,item["competitor_indices"]

    for name,item in sorted(declared.items()):
        if name == "decisive_delayed_boundary":
            before = atomic(item["before"])
            at = atomic(item["at"])
            boundary = item["boundary"]
            assert type(boundary) is int and boundary >= 1
            assert before[1]["key"] == at[1]["key"] and before[1]["as_of"] == boundary-1 and at[1]["as_of"] == boundary
            assert at[3] is not None and events[at[3]]["at"] == boundary
            assert answer_payload(before[2]) != answer_payload(at[2])
        elif name == "present_null_distinction":
            present = atomic(item["present_null"])
            contrast = atomic(item["null_contrast"])
            assert present[2]["status"] == "present" and present[2]["value"] is None
            assert contrast[2]["status"] in ("absent","deleted") and contrast[2]["value"] is None
        else:
            qi,query,result,winner,competitors = atomic(item)
            eligible = [i for i,e in enumerate(events) if e["key"] == query["key"] and e["at"] <= query["as_of"]]
            if name == "absence_control":
                assert not eligible and winner is None
                kind = "future_only" if any(e["key"] == query["key"] for e in events) else "missing_key"
                assert item["absence_kind"] == kind
            else:
                assert competitors
                chosen = events[winner] if winner is not None else None
                for ci in competitors:
                    competitor = events[ci]
                    if name == "authoritative_tombstone":
                        assert chosen["op"] == "delete" and ci in eligible and competitor["op"] == "put" and competitor["version"] < chosen["version"]
                    elif name == "version_over_recency":
                        latest = max(eligible,key=lambda i:(events[i]["at"],i))
                        assert ci == latest and chosen["version"] > competitor["version"] and chosen["at"] < competitor["at"]
                        assert event_payload(competitor) != answer_payload(result)
                    elif name == "equal_version_time_conflict":
                        assert ci in eligible and chosen["version"] == competitor["version"] and chosen["at"] > competitor["at"]
                        assert event_payload(competitor) != answer_payload(result)
                    elif name == "equal_version_time_input_conflict":
                        assert ci in eligible and chosen["version"] == competitor["version"] and chosen["at"] == competitor["at"] and ci < winner
                        assert event_payload(competitor) != answer_payload(result)
                    elif name == "future_event_filter_control":
                        assert ci == max((i for i,e in enumerate(events) if e["key"] == query["key"]),key=lambda i:rank(events,i))
                        assert competitor["at"] > query["as_of"] and event_payload(competitor) != answer_payload(result)
                    elif name == "cross_key_filter_control":
                        assert ci == max((i for i,e in enumerate(events) if e["at"] <= query["as_of"]),key=lambda i:rank(events,i))
                        assert competitor["key"] != query["key"] and event_payload(competitor) != answer_payload(result)
                    else:
                        raise AssertionError("Unknown public property")
        validated.append(name)
    controls = record["family_control_claims"]
    assert len(controls) == len(SCOPES) and {c["name"] for c in controls} == set(SCOPES)
    assert all(c["scope"] == SCOPES[c["name"]] and c["source_family_key"] == record["source_family_key"] for c in controls)
    assert all(type(c["evidence_manifest"]) is str and c["evidence_manifest"].endswith("memory-gap-v2.family_controls.json") for c in controls)
    return {"validated_properties":validated,"declared_scopes_exact":True,
            "manifest_read":False,"independent_support":"Own corpus reconstruction and own query-local audit probes; producer report pointer is not evidence read"}


def empty_hist():
    return {name:Counter() for name in ("role","status","type","typed_value","present_null","chosen_version","chosen_event","decoded_winner")}


def hist_add(hist,role,result,record,task):
    inverse = {key:group for group,key in record["key_role_assignments"].items()}
    hist["role"][role] += 1
    hist["status"][result["status"]] += 1
    hist["type"][typed_value(result["value"])] += 1
    hist["typed_value"][encode(result["value"])] += 1
    hist["present_null"][str(result["status"] == "present" and result["value"] is None)] += 1
    hist["chosen_version"][encode(result["chosen_version"])] += 1
    hist["chosen_event"][encode(result["chosen_event"])] += 1
    chosen = result["chosen_event"]
    decoded = None
    if chosen is not None:
        decoded = dict(task["events"][chosen])
        decoded["key"] = inverse[decoded["key"]]
    hist["decoded_winner"][encode(decoded)] += 1


def run():
    public_text = (BASE / "schemas/EXTENDED_SEMANTICS.md").read_text()
    properties = BASE / "schemas/MEMORY_CONTROL_PROPERTIES.md"
    properties.read_text()
    rows = [json.loads(line) for line in BATCH.read_text().splitlines()]
    facts,by_id = [],{}
    for record in rows:
        sample = record["sample_id"]
        assert sample not in by_id
        recomputed = hashlib.sha256(encode({k:v for k,v in record.items() if k != "record_hash"}).encode()).hexdigest()
        assert recomputed == record["record_hash"]
        task = public_prompt_task(record["rendered_input"],public_text)
        answer,traces = solve_scan(task)
        assert exact_equal(answer,solve_sorted(task))
        # Compare canonical material only after independent rendered-prompt solve.
        assert exact_equal(task,record["semantic_task"])
        assert exact_equal(answer,record["verified_target"])
        roles = role_map(record,task)
        own_witnesses = witnesses(task,answer)
        metadata = validate_metadata_witnesses(record,task,answer,own_witnesses)
        result = {"sample_id":sample,"record_hash":recomputed,"source_family_key":record["source_family_key"],
                  "controlled_variant":record["controlled_variant"],"renderer":record["renderer"],"query_rotation":record["query_rotation"],
                  "own_answer":answer,"selection_traces":traces,"public_property_witnesses":own_witnesses,
                  "metadata_witness_validation":metadata,"query_diagnostics":query_diagnostics(task),
                  "metamorphics":metamorphic_checks(task),"audit_probes":controlled_probes(task,answer),
                  "role_results":{role:{"query":q,"result":answer["results"][pos]} for role,(pos,q) in roles.items()},
                  "status_histogram":dict(Counter(r["status"] for r in answer["results"])),
                  "delete_event_count":sum(e["op"] == "delete" for e in task["events"])}
        assert set(result["status_histogram"]) == {"present","deleted","absent"}
        facts.append(result)
        by_id[sample] = (record,task,answer,result)
    assert len(rows) == 768
    by_history,by_render,by_family = defaultdict(list),defaultdict(list),defaultdict(list)
    for item in by_id.values():
        record = item[0]
        by_history[(record["source_family_key"],record["controlled_variant"],record["renderer"])].append(item)
        by_render[(record["source_family_key"],record["controlled_variant"],record["query_rotation"])].append(item)
        by_family[record["source_family_key"]].append(item)
    crossings,conditional = [],[]
    for (family,history,style),group in sorted(by_history.items()):
        assert len(group) == 8 and {r[0]["query_rotation"] for r in group} == set(range(8))
        ids,positions,keys = defaultdict(empty_hist),defaultdict(empty_hist),defaultdict(empty_hist)
        key_roles = defaultdict(Counter)
        role_signatures = defaultdict(set)
        canonical_history = None
        unconditional = Counter()
        joint = {name:defaultdict(Counter) for name in ("id_and_key","position_and_key","id_and_position")}
        schedule = []
        for record,task,answer,evidence in group:
            normalized = decode_key_roles(record,task)
            if canonical_history is None:
                canonical_history = normalized["events"]
            assert exact_equal(canonical_history,normalized["events"])
            inverse = {key:group for group,key in record["key_role_assignments"].items()}
            schedule.append({"history":history,"query_rotation":record["query_rotation"],"field_order":style,
                             "key_role_assignments":record["key_role_assignments"],"query_role_assignments":record["query_role_assignments"]})
            for role,key in record["key_role_assignments"].items():
                key_roles[key][role] += 1
            for role,(pos,query) in role_map(record,task).items():
                result = answer["results"][pos]
                hist_add(ids[query["query_id"]],role,result,record,task)
                hist_add(positions[str(pos)],role,result,record,task)
                hist_add(keys[query["key"]],role,result,record,task)
                role_signatures[role].add((inverse[query["key"]],query["as_of"]))
                unconditional[result["status"]] += 1
                joint["id_and_key"][(query["query_id"],query["key"])][result["status"]] += 1
                joint["position_and_key"][(str(pos),query["key"])][result["status"]] += 1
                joint["id_and_position"][(query["query_id"],str(pos))][result["status"]] += 1
        assert all(len(s) == 1 for s in role_signatures.values())
        for hist in ids.values():
            assert set(hist["role"]) == ROLES and set(hist["role"].values()) == {1}
        for hist in positions.values():
            assert set(hist["role"]) == ROLES and set(hist["role"].values()) == {1}
        assert len(keys) == 4 and all(set(c) == {"delayed","lifecycle","index_tie","absent"} and set(c.values()) == {2} for c in key_roles.values())
        assert all(exact_equal(h,list(ids.values())[0]) for h in ids.values())
        assert all(exact_equal(h,list(positions.values())[0]) for h in positions.values())
        assert all(exact_equal(h,list(keys.values())[0]) for h in keys.values())
        crossings.append({"source_family_key":family,"history":history,"field_order":style,"scope":"corpus_variant",
                          "id_histograms":ids,"position_histograms":positions,"key_histograms":keys,
                          "key_group_counts":key_roles,"role_signatures":{k:sorted(v) for k,v in role_signatures.items()},
                          "assignment_schedule":schedule,"typed_and_provenance_histograms_equal":True,"all_three_statuses":True})
        measured = {"family":family,"history":history,"field_order":style,"scope":"corpus_variant",
                    "observations":sum(unconditional.values()),"status_counts":unconditional,
                    "marginal_majority_fraction":max(unconditional.values())/sum(unconditional.values()),
                    "interpretation":"Empirical label association; no model, trained predictor, held-out result or efficacy claim"}
        for name,buckets in joint.items():
            measured[name] = {"majority_fraction":sum(max(c.values()) for c in buckets.values())/sum(unconditional.values()),
                              "tables":[{"bucket":list(k),"status_counts":v} for k,v in sorted(buckets.items())]}
        conditional.append(measured)
    rendering_pairs = []
    for key,group in sorted(by_render.items()):
        assert len(group) == 2
        forward = next(r for r in group if r[0]["renderer"] == "forward")
        reverse = next(r for r in group if r[0]["renderer"] == "reverse")
        assert exact_equal(forward[1],reverse[1]) and exact_equal(forward[2],reverse[2])
        prefix_a,serialization_a = forward[0]["rendered_input"].split("Task JSON:\n")
        prefix_b,serialization_b = reverse[0]["rendered_input"].split("Task JSON:\n")
        assert prefix_a == prefix_b and serialization_a != serialization_b
        rendering_pairs.append({"scope":"corpus_variant","control":"field_order_invariance",
                                "samples":[forward[0]["sample_id"],reverse[0]["sample_id"]],"passed":True})
    paired = []
    for family,group in sorted(by_family.items()):
        assert len(group) == 64 and len({r[0]["split"] for r in group}) == 1
        assert {r[0]["controlled_variant"] for r in group} == {"baseline","version_priority_raise","lifecycle_op_toggle","index_tie_swap"}
        for rotation in range(8):
            for style in ("forward","reverse"):
                controlled = {r[0]["controlled_variant"]:r for r in group if r[0]["query_rotation"] == rotation and r[0]["renderer"] == style}
                base = controlled["baseline"]
                for variant in ("version_priority_raise","lifecycle_op_toggle","index_tie_swap"):
                    other = controlled[variant]
                    assert exact_equal(base[1]["queries"],other[1]["queries"])
                    assert len(base[1]["events"]) == len(other[1]["events"])
                    changed_indices = [i for i,(a,b) in enumerate(zip(base[1]["events"],other[1]["events"])) if not exact_equal(a,b)]
                    differences = [{"index":i,"before":base[1]["events"][i],"after":other[1]["events"][i]} for i in changed_indices]
                    changed_roles = [role for role in ROLES if answer_payload(base[3]["role_results"][role]["result"]) != answer_payload(other[3]["role_results"][role]["result"])]
                    if variant == "version_priority_raise":
                        assert len(differences) == 1 and set(changed_roles) == {"delayed_later"}
                        diff = differences[0]
                        assert exact_equal({k:v for k,v in diff["before"].items() if k != "version"},
                                           {k:v for k,v in diff["after"].items() if k != "version"})
                        qi = role_map(base[0],base[1])["delayed_later"][0]
                        query = base[1]["queries"][qi]
                        eligible = [e for e in base[1]["events"] if e["key"] == query["key"] and e["at"] <= query["as_of"]]
                        assert diff["after"]["version"] > max(e["version"] for e in eligible)
                        assert other[2]["results"][qi]["chosen_event"] == diff["index"]
                        control = "version_priority_counterfactual"
                    elif variant == "lifecycle_op_toggle":
                        assert len(differences) == 1 and set(changed_roles) == {"lifecycle_at"}
                        diff = differences[0]
                        assert exact_equal({k:v for k,v in diff["before"].items() if k != "op"},
                                           {k:v for k,v in diff["after"].items() if k != "op"})
                        assert diff["before"]["value"] is None and {diff["before"]["op"],diff["after"]["op"]} == {"put","delete"}
                        control = "null_lifecycle_counterfactual"
                    else:
                        assert len(differences) == 2 and set(changed_roles) == {"same_version_time_index"}
                        first,second = differences
                        assert exact_equal(first["before"],second["after"]) and exact_equal(second["before"],first["after"])
                        qi = role_map(base[0],base[1])["same_version_time_index"][0]
                        winner = base[2]["results"][qi]["chosen_event"]
                        assert winner in changed_indices
                        assert rank(base[1]["events"],first["index"])[:2] == rank(base[1]["events"],second["index"])[:2]
                        control = "active_input_tie_swap_counterfactual"
                    paired.append({"family":family,"scope":"corpus_variant","control":control,
                                   "baseline_sample":base[0]["sample_id"],"changed_sample":other[0]["sample_id"],
                                   "event_differences":differences,"changed_roles":changed_roles,"passed":True})
    normalized_histories = {encode(decode_key_roles(r,t)["events"]) for r,t,_,_ in by_id.values()}
    observations = {(encode(decode_key_roles(r,t)["events"]),q["key"],q["as_of"])
                    for r,t,_,_ in by_id.values() for q in decode_key_roles(r,t)["queries"]}
    report = {"reviewer_execution_id":EXECUTION_ID,"stage":"INDEPENDENT_CORRECTIVE_FACTS_BEFORE_FIRST_JUDGMENTS",
              "guidance_scope":"Own v1 metadata/coverage findings informed the topic; no other reviewer verdicts or producer/source/control reports read",
              "source_file":str(BATCH),"source_sha256":sha(BATCH),"records":len(rows),"source_families":len(by_family),
              "base_controlled_histories":len(normalized_histories),
              "distinct_named_event_histories":len({encode(t["events"]) for _,t,_,_ in by_id.values()}),
              "base_history_query_observations":len(observations),"rotation_observations":len(rows)//2*8,"rendered_observations":len(rows)*8,
              "split_record_counts":dict(Counter(r["split"] for r in rows)),
              "split_family_counts":dict(Counter(group[0][0]["split"] for group in by_family.values())),
              "event_count_counts":dict(Counter(len(t["events"]) for _,t,_,_ in by_id.values())),
              "difficulty_counts":dict(Counter(r["difficulty"] for r in rows)),
              "all_hash_prompt_target_typed_checks_pass":True,"all_candidate_witnesses_validated":True,
              "all_family_scope_declarations_validated":True,"producer_control_manifests_read":False,
              "property_row_counts":{name:sum(bool(row["public_property_witnesses"][name]) for row in facts) for name in sorted(PROPERTIES)},
              "status_query_counts":dict(Counter(r["status"] for row in facts for r in row["own_answer"]["results"])),
              "type_query_counts":dict(Counter(typed_value(r["value"]) for row in facts for r in row["own_answer"]["results"])),
              "missing_key_absence_queries":sum(w["absence_kind"] == "missing_key" for row in facts for w in row["public_property_witnesses"]["absence_control"]),
              "future_only_absence_queries":sum(w["absence_kind"] == "future_only" for row in facts for w in row["public_property_witnesses"]["absence_control"]),
              "all_three_status_rows":sum(set(row["status_histogram"]) == {"present","deleted","absent"} for row in facts),
              "full_typed_and_provenance_histogram_crossing_sets":len(crossings),
              "rendering_pairs":len(rendering_pairs),"corpus_counterfactual_pairs":len(paired),
              "metamorphic_checks":sum(len(row["metamorphics"]) for row in facts),
              "audit_probe_count":sum(len(row["audit_probes"]) for row in facts),
              "audit_probe_control_counts":dict(Counter(p["control"] for row in facts for p in row["audit_probes"])),
              "conditional_majority_fractions":{name:sorted({r[name]["majority_fraction"] for r in conditional}) for name in ("id_and_key","position_and_key","id_and_position")},
              "conditional_marginal_majority_fractions":sorted({r["marginal_majority_fraction"] for r in conditional}),
              "scripts_sha256":{str(Path(__file__).resolve()):sha(__file__),
                                 str(OUT/"independent_memory_gap_v1.py"):sha(OUT/"independent_memory_gap_v1.py"),
                                 str(OUT/"review_memory_gap_v1.py"):sha(OUT/"review_memory_gap_v1.py")},
              "public_contract_sha256":sha(BASE/"schemas/EXTENDED_SEMANTICS.md"),"public_properties_sha256":sha(properties),
              "read_scope":"Only memory-gap-v2 candidates, public contracts and own code. Producer witness fields inside candidates independently checked after own solve; referenced producer control manifest never read. Old receipts/reviews untouched.",
              "row_evidence":facts,"crossings":crossings,"field_order_pairs":rendering_pairs,
              "supplied_counterfactuals":paired,"conditional_shortcut_tables":conditional}
    raw_path = OUT/"memory-gap-v2.independent_analysis.json"
    raw = (json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode()
    with raw_path.open("xb") as stream:
        stream.write(raw)
    compressed = gzip.compress(raw,compresslevel=9,mtime=0)
    compressed_path = Path(str(raw_path)+".gz")
    with compressed_path.open("xb") as stream:
        stream.write(compressed)
    assert gzip.decompress(compressed) == raw and len(compressed) < 25*1024*1024
    manifest = {"reviewer_execution_id":EXECUTION_ID,"raw_path":str(raw_path),"raw_sha256":hashlib.sha256(raw).hexdigest(),"raw_bytes":len(raw),
                "gzip_path":str(compressed_path),"gzip_sha256":hashlib.sha256(compressed).hexdigest(),"gzip_bytes":len(compressed),
                "deterministic_gzip_mtime":0,"decompressed_bytes_equal":True,"decompressed_sha256":hashlib.sha256(gzip.decompress(compressed)).hexdigest(),
                "raw_preserved":raw_path.read_bytes() == raw}
    with (OUT/"memory-gap-v2.independent_analysis.compression_manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest,indent=2)+"\n")
    summary = {k:v for k,v in report.items() if k not in ("row_evidence","crossings","field_order_pairs","supplied_counterfactuals","conditional_shortcut_tables")}
    with (OUT/"memory-gap-v2.independent_facts_summary.json").open("x") as stream:
        stream.write(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    run()
