#!/opt/anaconda3/bin/python3
"""Prospective exposed-release RBM gap measurement, no generation or verdicts.

Own public-contract-derived recomputation. Reads approved canonical/review
evidence only, no producer/reference/oracle or future batch material.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
EXECUTION_ID = "/root/expert_reviewer"


def encode(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))


def scalar_key(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":"))


def exact(left,right):
    return encode(left) == encode(right)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(task):
    assert task["kind"] == "belief" and set(task) == {"kind","worlds","observations","queries"}
    worlds = task["worlds"]
    assert worlds and len({w["world_id"] for w in worlds}) == len(worlds)
    facts = set(worlds[0]["facts"])
    assert all(set(w) == {"world_id","facts"} and type(w["world_id"]) is str and set(w["facts"]) == facts for w in worlds)
    assert all(v is None or type(v) in (bool,int,str) for w in worlds for v in w["facts"].values())
    for observation in task["observations"]:
        assert set(observation) == {"fact","equals"} and observation["fact"] in facts
        assert observation["equals"] is None or type(observation["equals"]) in (bool,int,str)
    assert len({q["query_id"] for q in task["queries"]}) == len(task["queries"])
    assert all(set(q) == {"query_id","fact"} and q["fact"] in facts and type(q["query_id"]) is str for q in task["queries"])


def answer_queries(task,survivors):
    queries = []
    for query in task["queries"]:
        counts = Counter(scalar_key(task["worlds"][i]["facts"][query["fact"]]) for i in survivors)
        entries = [{"value":json.loads(key),"count":counts[key]} for key in sorted(counts)]
        entailed = bool(survivors) and len(entries) == 1
        queries.append({"query_id":query["query_id"],"counts":entries,"distinct_values":len(entries),
                        "entailed":entailed,"guaranteed":entries[0]["value"] if entailed else None,
                        "can_reduce_uncertainty":len(entries) > 1})
    return queries


def solve_filter(task):
    validate(task)
    survivors = list(range(len(task["worlds"])))
    sizes,traces = [],[]
    for observation in task["observations"]:
        before = survivors[:]
        survivors = [i for i in survivors if scalar_key(task["worlds"][i]["facts"][observation["fact"]]) == scalar_key(observation["equals"])]
        sizes.append(len(survivors))
        traces.append({"observation":observation,"before":[task["worlds"][i]["world_id"] for i in before],
                       "after":[task["worlds"][i]["world_id"] for i in survivors]})
    return {"surviving_worlds":sorted(task["worlds"][i]["world_id"] for i in survivors),
            "belief_sizes":sizes,"queries":answer_queries(task,survivors)},traces


def solve_conjunction(task):
    """Independent prefix enumeration rather than incremental update."""
    validate(task)
    sizes = []
    for prefix_length in range(1,len(task["observations"])+1):
        sizes.append(sum(all(scalar_key(w["facts"][o["fact"]]) == scalar_key(o["equals"]) for o in task["observations"][:prefix_length]) for w in task["worlds"]))
    final = [i for i,w in enumerate(task["worlds"]) if all(scalar_key(w["facts"][o["fact"]]) == scalar_key(o["equals"]) for o in task["observations"])]
    # Compute count entries directly by typed equivalence classes of values.
    queries = []
    for query in task["queries"]:
        values = [task["worlds"][i]["facts"][query["fact"]] for i in final]
        unique = {scalar_key(v):v for v in values}
        entries = [{"value":unique[key],"count":sum(scalar_key(v) == key for v in values)} for key in sorted(unique)]
        entailed = len(final) > 0 and len(entries) == 1
        queries.append({"query_id":query["query_id"],"counts":entries,"distinct_values":len(entries),
                        "entailed":entailed,"guaranteed":entries[0]["value"] if entailed else None,
                        "can_reduce_uncertainty":len(entries) > 1})
    return {"surviving_worlds":sorted(task["worlds"][i]["world_id"] for i in final),"belief_sizes":sizes,"queries":queries}


def run():
    started = datetime.now(timezone.utc).isoformat()
    contracts = BASE/"schemas/EXTENDED_SEMANTICS.md"
    curriculum = BASE/"research/curricula/RBM-001.md"
    canonical = BASE/"data/RBM-001/releases/v0001/canonical.jsonl"
    reviews = BASE/"data/RBM-001/releases/v0001/review_receipts.jsonl"
    manifest = BASE/"data/RBM-001/releases/v0001/MANIFEST.json"
    public_text = contracts.read_text()
    curriculum.read_text()
    section = public_text.split("## `kind=belief`\n",1)[1].split("\n## ",1)[0].strip()
    records = [json.loads(line) for line in canonical.read_text().splitlines()]
    receipts = [json.loads(line) for line in reviews.read_text().splitlines()]
    receipt_lookup = {r["receipt_id"]:r for r in receipts}
    manifest_data = json.loads(manifest.read_text())
    assert sha(canonical) == manifest_data["files"]["canonical.jsonl"]["sha256"]
    assert sha(reviews) == manifest_data["files"]["review_receipts.jsonl"]["sha256"]
    facts,parent_rows,query_rows = [],[],[]
    support_buckets = defaultdict(Counter)
    id_buckets = defaultdict(Counter)
    mutant_correct = Counter()
    for record in records:
        assert hashlib.sha256(encode({k:v for k,v in record.items() if k != "record_hash"}).encode()).hexdigest() == record["record_hash"]
        prefix,encoded_task = record["rendered_input"].split("Task JSON:\n")
        assert section in prefix
        task = json.loads(encoded_task)
        answer,traces = solve_filter(task)
        assert exact(answer,solve_conjunction(task)) and exact(task,record["semantic_task"]) and exact(answer,record["verified_target"])
        assert record["review_status"] == "ACCEPTED" and len(record["review_receipt_ids"]) == 3
        for receipt_id in record["review_receipt_ids"]:
            receipt = receipt_lookup[receipt_id]
            assert receipt["sample_id"] == record["sample_id"] and receipt["record_hash"] == record["reviewed_candidate_hash"] and receipt["decision"] == "ACCEPT"
        restored = {k:v for k,v in record.items() if k not in ("record_hash","reviewed_candidate_hash")}
        restored["review_status"],restored["review_receipt_ids"] = "PENDING",[]
        assert hashlib.sha256(encode(restored).encode()).hexdigest() == record["reviewed_candidate_hash"]
        support = len(answer["surviving_worlds"])
        observed_facts = {o["fact"] for o in task["observations"]}
        row = {"sample_id":record["sample_id"],"record_hash":record["record_hash"],"reviewed_candidate_hash":record["reviewed_candidate_hash"],
               "semantic_parent_id":record["semantic_parent_id"],"renderer":record["renderer"],"split":record["split"],
               "world_count":len(task["worlds"]),"observation_count":len(task["observations"]),"query_count":len(task["queries"]),
               "belief_sizes":answer["belief_sizes"],"final_support":support,"observation_trace":traces,"query_outputs":answer["queries"],
               "observations":task["observations"],"query_mapping":task["queries"],"world_table_hash":hashlib.sha256(encode(task["worlds"]).encode()).hexdigest(),
               "difficulty":record["difficulty"],"challenge_axes":record["challenge_axes"],"skill_tags":record["skill_tags"]}
        for pos,(query,result) in enumerate(zip(task["queries"],answer["queries"])):
            useful_shortcut = support > 1 and query["fact"] not in observed_facts
            entailed_shortcut = support > 0 and (support == 1 or query["fact"] in observed_facts)
            mutant_correct["support_plus_observed_fact_usefulness"] += useful_shortcut == result["can_reduce_uncertainty"]
            mutant_correct["support_plus_observed_fact_entailment"] += entailed_shortcut == result["entailed"]
            support_buckets[str(support)][str(result["can_reduce_uncertainty"])] += 1
            id_buckets[query["query_id"]][str(result["can_reduce_uncertainty"])] += 1
            query_rows.append({"sample_id":record["sample_id"],"parent":record["semantic_parent_id"],"query_id":query["query_id"],
                               "position":pos,"fact":query["fact"],"observed_fact":query["fact"] in observed_facts,
                               "support":support,"distinct_values":result["distinct_values"],"entailed":result["entailed"],
                               "guaranteed":result["guaranteed"],"can_reduce_uncertainty":result["can_reduce_uncertainty"],"counts":result["counts"]})
        facts.append(row)
        if record["renderer"] == "forward":
            parent_rows.append(row)
    assert len(records) == 24 and len(parent_rows) == 12
    pairs = defaultdict(list)
    for record in records:
        pairs[record["semantic_parent_id"]].append(record)
    assert all(len(group) == 2 and exact(group[0]["semantic_task"],group[1]["semantic_task"]) and exact(group[0]["verified_target"],group[1]["verified_target"]) for group in pairs.values())
    assert scalar_key(True) != scalar_key(1) and scalar_key(False) != scalar_key(0)
    actual_mechanisms = Counter()
    for row in parent_rows:
        task = next(r["semantic_task"] for r in records if r["sample_id"] == row["sample_id"])
        support = row["final_support"]
        observations = task["observations"]
        if not observations:
            actual_mechanisms["unfiltered_counts"] += 1
        if support == 0:
            actual_mechanisms["typed_same_fact_contradiction_empty"] += 1
        if support > 1 and any(q["can_reduce_uncertainty"] for q in row["query_outputs"]) and any(q["entailed"] for q in row["query_outputs"]):
            actual_mechanisms["mixed_useful_entailed_multiworld_queries"] += 1
        if len(observations) >= 2 and support > 0:
            actual_mechanisms["conjunctive_positive_updates"] += 1
        if any(q["entailed"] and q["guaranteed"] is None for q in row["query_outputs"]):
            actual_mechanisms["entailed_null"] += 1
        if any(o["equals"] == 1 and type(o["equals"]) is int for o in observations) and support > 0:
            actual_mechanisms["integer_one_observation_excludes_boolean_true"] += 1
    summary = {
        "current_release_records":len(records),"declared_source_parents":len(parent_rows),"unique_parsed_tasks":len({encode(r["semantic_task"]) for r in records}),
        "field_order_only_pairs":len(pairs),"distinct_exact_world_tables":len({encode(r["semantic_task"]["worlds"]) for r in records}),
        "observed_mode_count":len({encode(row["observations"]) for row in parent_rows}),
        "observed_abstract_mode_count":6,"actual_mechanism_parent_counts":actual_mechanisms,
        "split_records":dict(Counter(r["split"] for r in records)),"split_parents":dict(Counter(r["split"] for r in parent_rows)),
        "world_count_records":dict(Counter(r["world_count"] for r in facts)),"observation_count_records":dict(Counter(r["observation_count"] for r in facts)),
        "support_size_records":dict(Counter(r["final_support"] for r in facts)),"difficulty_records":dict(Counter(r["difficulty"] for r in facts)),
        "belief_trajectory_records":dict(Counter(encode(r["belief_sizes"]) for r in facts)),
        "queries_total":len(query_rows),"can_reduce_true_queries":sum(q["can_reduce_uncertainty"] for q in query_rows),
        "entailed_true_queries":sum(q["entailed"] for q in query_rows),"empty_belief_queries":sum(q["support"] == 0 for q in query_rows),
        "distinct_value_query_counts":dict(Counter(q["distinct_values"] for q in query_rows)),
        "useful_after_nonempty_observations_queries":sum(q["can_reduce_uncertainty"] and any(r["observation_count"] > 0 for r in facts if r["sample_id"] == q["sample_id"]) for q in query_rows),
        "useful_after_multiple_observations_queries":sum(q["can_reduce_uncertainty"] and any(r["observation_count"] >= 2 for r in facts if r["sample_id"] == q["sample_id"]) for q in query_rows),
        "mixed_useful_and_entailed_multiworld_records":sum(r["final_support"] > 1 and any(q["can_reduce_uncertainty"] for q in r["query_outputs"]) and any(q["entailed"] for q in r["query_outputs"]) for r in facts),
        "unobserved_entailed_multiworld_queries":sum(q["support"] > 1 and not q["observed_fact"] and q["entailed"] for q in query_rows),
        "entailed_null_queries":sum(q["entailed"] and q["guaranteed"] is None for q in query_rows),
        "entailed_null_multiworld_queries":sum(q["support"] > 1 and q["entailed"] and q["guaranteed"] is None for q in query_rows),
        "nonempty_multiworld_all_redundant_records":sum(r["final_support"] > 1 and not any(q["can_reduce_uncertainty"] for q in r["query_outputs"]) for r in facts),
        "observations_after_first_empty":sum(sum(size == 0 for size in r["belief_sizes"][:-1]) for r in facts),
        "nonempty_redundant_filter_steps":sum(bool(t["before"]) and t["before"] == t["after"] for r in facts for t in r["observation_trace"]),
        "filter_step_count":sum(len(r["observation_trace"]) for r in facts),
        "useful_query_multivalue_count_entries_unequal":sum(q["can_reduce_uncertainty"] and len({entry["count"] for entry in q["counts"]}) > 1 for q in query_rows),
        "typed_false_zero_world_pairs":0,
        "id_role_mapping":{qid:sorted({(q["position"],q["fact"]) for q in query_rows if q["query_id"] == qid}) for qid in sorted(id_buckets)},
        "id_usefulness_histograms":id_buckets,"support_usefulness_histograms":support_buckets,
        "incomplete_contract_heuristic_matches":dict(mutant_correct),
        "support_only_empirical_majority_matches":sum(max(c.values()) for c in support_buckets.values()),
        "shortcut_interpretation":"Empirical release label associations/mutant compatibility only; not trained-model accuracy, data leakage proof or held-out robustness result",
        "released_hashes_exact":True,"restored_candidate_hashes_exact":True,"own_solver_full_targets_exact":True,
        "approved_receipts_bound":len(receipts),"approved_receipt_roles":dict(Counter(r["reviewer_role"] for r in receipts)),
        "approved_receipt_decisions":dict(Counter(r["decision"] for r in receipts)),
        "prior_expert_scope_issue_categories":dict(Counter(i["category"] for r in receipts if r["reviewer_role"] == "expert_reviewer" for i in r["issues"])),
        "no_new_dataset_verdict":True,"no_future_samples_or_targets":True,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = BASE/f"reviews/expert/RBM-gap-analysis-{timestamp}"
    output = Path(str(stem)+".json")
    receipt_path = Path(str(stem)+".execution.json")
    report = {"analysis_id":"RBM-PROSPECTIVE-"+timestamp,"reviewer_execution_id":EXECUTION_ID,"started_utc":started,
              "finished_utc":datetime.now(timezone.utc).isoformat(),"task_scope":"Prospective gap analysis of exposed approved RBM v0001, not generation/review/approval of a future batch",
              "read_inputs_sha256":{str(p):sha(p) for p in (contracts,curriculum,canonical,reviews,manifest)},
              "summary":summary,"parent_examples":parent_rows,"per_record_evidence":facts,"query_evidence":query_rows}
    with output.open("x") as stream:
        stream.write(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    execution = {"receipt_id":"EXPERT-RBM-PROSPECTIVE-"+timestamp,"reviewer_role":"expert_reviewer","reviewer_execution_id":EXECUTION_ID,
                 "status":"COMPLETED_PROSPECTIVE_MEASUREMENT","decision":"NO_DATASET_VERDICT","started_utc":started,"finished_utc":report["finished_utc"],
                 "script_path":str(Path(__file__).resolve()),"script_sha256":sha(__file__),"report_path":str(output),"report_sha256":sha(output),
                 "input_hashes_before_after_equal":all(sha(p) == h for p,h in report["read_inputs_sha256"].items()),
                 "exact_existing_records_recomputed":len(records),"exact_approved_receipts_bound":len(receipts),
                 "no_future_targets_or_records_authored":True,"no_generation_or_contract_release_training_edits":True,
                 "no_sealed_material":True,"no_producer_reference_or_oracle_source_read":True,"REM_v1_v2_outputs_not_modified":True}
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(execution,indent=2)+"\n")
    print(json.dumps({"report":str(output),"receipt":str(receipt_path),"summary":summary},indent=2))


if __name__ == "__main__":
    run()
