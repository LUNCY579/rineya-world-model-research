#!/opt/anaconda3/bin/python3
"""Own post-prediction expert checks; preserve all first-stage artifacts.
Only public design, frozen own predictions and subsequently authorized canonical
rows are read. No producer/reference/oracle/checker or other review imports.
"""
import collections
import copy
import datetime
import gzip
import hashlib
import itertools
import json
import pathlib
import runpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
OWN = runpy.run_path(str(ROOT / "reviews/expert/belief-gap-v1.prepared_source.py"))
canon, equal, scalar, solve = [OWN[k] for k in ("canon", "equal", "scalar", "solve")]
sha = lambda b: hashlib.sha256(b).hexdigest()
CANONICAL = ROOT / "data/RBM-001/candidate_renderings/belief-gap-v1.jsonl"
PREDICTIONS = ROOT / "reviews/expert/belief-gap-v1.input_only_predictions.json"
COMPARISON = ROOT / "reviews/expert/belief-gap-v1.post_freeze_comparison.json"


def key_order_tree(value):
    if isinstance(value, dict):
        return {"object_keys": list(value), "children": [key_order_tree(v) for v in value.values()]}
    if isinstance(value, list):
        return {"list": [key_order_tree(v) for v in value]}
    return {"scalar": value}


def reverse_objects(value):
    if isinstance(value, dict):
        return {k: reverse_objects(value[k]) for k in reversed(value)}
    if isinstance(value, list):
        return [reverse_objects(v) for v in value]
    return value


def typed_corpus_pairs(records):
    bykey = {(r["metadata"]["source_family_key"], r["metadata"]["controlled_context"], r["metadata"]["schedule_index"], r["metadata"]["renderer"]): r for r in records}
    found = []
    for key, A in sorted(bykey.items()):
        fam, context, schedule, renderer = key
        if context != "A":
            continue
        B = bykey.get((fam, "B", schedule, renderer))
        if B is None:
            continue
        at, bt = A["task"], B["task"]
        change = OWN["single_equals_change"](at["observations"], bt["observations"])
        if change is None:
            continue
        old, new = at["observations"][change]["equals"], bt["observations"][change]["equals"]
        if not any(equal(v, new) for v in OWN["aliases"](old)):
            continue
        p, q = A["facts"]["prefix_survivors"][change + 1], B["facts"]["prefix_survivors"][change + 1]
        if not equal(at["worlds"], bt["worlds"]) or not equal(at["queries"], bt["queries"]) or not p or not q or equal(p, q):
            continue
        found.append({"family": fam, "pair_sample_ids": [A["sample_id"], B["sample_id"]], "scope": "corpus_variant",
                      "name": "typed_equals_counterfactual", "observation_index": change, "old": old, "new": new,
                      "old_changed_prefix_survivors": p, "new_changed_prefix_survivors": q,
                      "old_final_survivors": A["facts"]["expected"]["surviving_worlds"],
                      "new_final_survivors": B["facts"]["expected"]["surviving_worlds"]})
    return found


def invariant_hist(records, axes):
    buckets = collections.defaultdict(collections.Counter)
    for r in records:
        md = r["metadata"]
        for pos, (q, a) in enumerate(zip(r["task"]["queries"], r["facts"]["expected"]["queries"])):
            role = next(k for k, v in md["fact_role_assignments"].items() if v == q["fact"])
            known = {"query_id": q["query_id"], "position": pos, "fact_name": q["fact"], "support": len(r["facts"]["expected"]["surviving_worlds"]),
                     "observed": q["fact"] in {o["fact"] for o in r["task"]["observations"]}, "decoded_role": role,
                     "world_count": len(r["task"]["worlds"]), "context": md["controlled_context"]}
            buckets[canon([known[k] for k in axes])][canon([a["entailed"], a["can_reduce_uncertainty"]])] += 1
    return {"axes": axes, "buckets": {k: dict(v) for k, v in sorted(buckets.items())},
            "empirical_joint_flag_majority_matches": sum(max(c.values()) for c in buckets.values()),
            "query_observations": sum(sum(c.values()) for c in buckets.values()),
            "interpretation": "in-sample corpus compatibility only; no model trained/evaluated"}


def main():
    assert sha(CANONICAL.read_bytes()) == "423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e"
    assert sha(PREDICTIONS.read_bytes()) == "f732c60d2ae3646bb3cc92bc166dc05a867c94bbc3b364558630e6e1f647d4fe"
    assert sha(COMPARISON.read_bytes()) == "3f84bd5289a79a8e07d109d84d8dc72e7a07973049c833fc68c467765c2f1937"
    d = json.loads(COMPARISON.read_text())
    records = d["post_freeze_audit_records"]
    rows = {r["sample_id"]: r for r in map(json.loads, CANONICAL.read_text().splitlines())}
    errors = []
    corpus_typed = typed_corpus_pairs(records)
    witnessed_typed_families = {p["family"] for p in corpus_typed}
    implementation_warnings_resolved = []
    for e in d["family_audits"]["errors"]:
        if e["error"] == "family claim has no reconstructed witness at declared scope: typed_equals_counterfactual":
            record = next(r for r in records if r["sample_id"] == e["sample_id"])
            claims = [c for c in record["metadata"]["family_control_claims"] if c.get("name", c.get("property")) == "typed_equals_counterfactual"]
            if record["metadata"]["source_family_key"] in witnessed_typed_families and claims and all(c["scope"] == "corpus_variant" for c in claims):
                implementation_warnings_resolved.append(e["sample_id"])
            else:
                errors.append(e)
        else:
            errors.append(e)
    fields, probes, identity = [], [], []
    for r in records:
        row, md, task = rows[r["sample_id"]], r["metadata"], r["task"]
        if not equal(row["semantic_task"], task):
            errors.append({"sample_id": r["sample_id"], "error": "semantic task/prompt mismatch"})
        if row["difficulty"] != row["structural_difficulty"] or type(row["difficulty"]) is not int or row["difficulty"] not in (2, 3):
            errors.append({"sample_id": r["sample_id"], "error": "unsupported difficulty"})
        expected_split = "train" if md["source_family_index"] in (12, 13) else "dev_challenge"
        if row["split"] != expected_split or row["semantic_parent_id"] != row["semantic_family_id"]:
            errors.append({"sample_id": r["sample_id"], "error": "family/split custody mismatch"})
        if row["review_status"] != "PENDING" or row["review_receipt_ids"] != []:
            errors.append({"sample_id": r["sample_id"], "error": "candidate carries review status"})
        if row["track_id"] != "RBM-001" or row["target_format"] != "structured":
            errors.append({"sample_id": r["sample_id"], "error": "wrong track/output format"})
        if r["facts"]["metrics"]["duplicate_visible_payloads"]:
            errors.append({"sample_id": r["sample_id"], "error": "duplicated complete world payload"})
        if md["renderer"] == "forward":
            counterpart_id = r["sample_id"].removesuffix("forward") + "reverse"
            other = rows[counterpart_id]
            lead, body = row["rendered_input"].split("Task JSON:\n")
            olead, obody = other["rendered_input"].split("Task JSON:\n")
            t, ot = json.loads(body), json.loads(obody)
            exact_reverse = equal(key_order_tree(reverse_objects(t)), key_order_tree(ot))
            okay = lead == olead and row["rendered_input"] != other["rendered_input"] and equal(t, ot) and exact_reverse
            fields.append({"pair_sample_ids": [r["sample_id"], counterpart_id], "pass": okay,
                           "same_prompt_contract": lead == olead, "actual_recursive_object_key_reversal": exact_reverse,
                           "typed_values_and_all_list_orders_preserved": equal(t, ot)})
            if not okay:
                errors.append({"sample_id": r["sample_id"], "error": "field renderer not exact object-order pair"})
        for claim in md["family_control_claims"]:
            if claim["scope"] == "audit_probe" and claim.get("name", claim.get("property")) in ("world_order_invariance", "world_id_rename_equivariance"):
                name = claim.get("name", claim.get("property"))
                executed = next(p for p in r["probes"]["invariance_probes"] if p["name"] == name)
                probes.append({"sample_id": r["sample_id"], "family": md["source_family_key"], "declared_probe_id": claim.get("probe_id"),
                               "independently_executed_control": executed,
                               "producer_manifest_not_read": True, "producer_pointer_not_witness": True})
    representative = [r for r in records if r["sample_id"].endswith("s00-forward")]
    own_prior_path = ROOT / "reviews/expert/RBM-gap-analysis-20261010T015614Z.json"
    own_prior = json.loads(own_prior_path.read_text())
    old_world_counts = sorted({p["world_count"] for p in own_prior["per_record_evidence"]})
    new_world_counts = sorted({len(r["task"]["worlds"]) for r in representative})
    assert set(old_world_counts).isdisjoint(new_world_counts)
    # Number of world vertices is isomorphism invariant. This proves freshness
    # against the prior corpus using frozen own measurements, without reopening
    # historical reviews or relying on arbitrary IDs/name changes.
    freshness = {"own_prior_analysis_path": str(own_prior_path), "own_prior_analysis_sha256": sha(own_prior_path.read_bytes()),
                 "prior_world_counts": old_world_counts, "new_world_counts": new_world_counts,
                 "nonisomorphic_to_every_prior_world_table": True,
                 "proof": "Different world-vertex cardinality4/6 versus8/12 cannot be removed by world/fact/value renaming.",
                 "mutual_new_table_nonisomorphism": d["family_audits"]["exact_table_structure_comparisons"],
                 "shared_submotifs": "Uniform and weighted cyclic families share unweighted topology; null/opal invariant and linen auxiliary motif reused in all four."}
    counts = {
        "records": len(records), "declared_source_families": len({r["metadata"]["source_family_key"] for r in records}),
        "base_contexts": len({(r["metadata"]["source_family_key"], r["metadata"]["controlled_context"]) for r in records}),
        "unique_typed_tasks": len({canon(r["task"]) for r in records}),
        "literal_typed_world_tables": len({canon(r["task"]["worlds"]) for r in records}),
        "decoded_world_tables": len({canon(OWN["decoded_task"](r["task"], r["metadata"])["worlds"]) for r in records}),
        "rendered_prompt_strings": len({rows[r["sample_id"]]["rendered_input"] for r in records}),
        "supplied_typed_counterfactual_pairs": len(corpus_typed), "field_pairs": len(fields),
        "executed_invariance_probes": sum(len(r["probes"]["invariance_probes"]) for r in records),
        "executed_typed_equals_probes": sum(len(r["probes"]["typed_equals_counterfactual_probes"]) for r in records),
        "qualifying_typed_equals_probes": sum(p["qualifying_changed_prefix"] for r in records for p in r["probes"]["typed_equals_counterfactual_probes"]),
        "query_observations": sum(len(r["task"]["queries"]) for r in records),
        "split": dict(collections.Counter(rows[r["sample_id"]]["split"] for r in records)),
        "contexts": dict(collections.Counter(r["metadata"]["controlled_context"] for r in records)),
        "per_record_property_counts": {p: sum(bool(r["facts"]["properties"][p]) for r in records) for p in sorted(OWN["PROPERTIES"])},
        "skill_tag_counts": dict(collections.Counter(tag for r in records for tag in r["metadata"]["skill_tags"])),
        "incomplete_useful_heuristic_errors": sum(len(r["facts"]["incomplete_heuristic_error_query_ids"]["useful"]) for r in records),
        "incomplete_entailed_heuristic_errors": sum(len(r["facts"]["incomplete_heuristic_error_query_ids"]["entailed"]) for r in records),
        "incomplete_heuristic_complete_task_errors": sum(bool(r["facts"]["incomplete_heuristic_error_query_ids"]["useful"]) or bool(r["facts"]["incomplete_heuristic_error_query_ids"]["entailed"]) for r in records),
        "histogram_context_renderer_groups": len(d["family_audits"]["histograms"]),
        "typed_payload_provenance_histogram_axes_pass": sum(t["identical"] for h in d["family_audits"]["histograms"].values() for t in h["outcome_histograms"].values())}
    query_types, outcomes, null_contexts = collections.Counter(), collections.Counter(), collections.Counter()
    for r in records:
        for q in r["facts"]["expected"]["queries"]:
            category = "empty" if not q["counts"] else "entailed" if q["entailed"] else "ambiguous"
            outcomes[category] += 1
            if q["guaranteed"] is None:
                null_contexts[category] += 1
            for b in q["counts"]:
                query_types[type(b["value"]).__name__] += 1
    counts.update({"query_categories": dict(outcomes), "guaranteed_null_interpretations": dict(null_contexts), "positive_count_bin_types": dict(query_types)})
    report = {"actual_author": "/root/expert_reviewer", "report_type": "OWN_POST_FREEZE_SEMANTIC_INSPECTION_AUDIT",
              "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "input_prediction_sha256": sha(PREDICTIONS.read_bytes()), "initial_comparison_sha256": sha(COMPARISON.read_bytes()),
              "original_candidate_sha256": sha(CANONICAL.read_bytes()), "source_sha256": sha(pathlib.Path(__file__).read_bytes()),
              "resolved_initial_audit_implementation_warning_count": len(implementation_warnings_resolved),
              "warning_resolution": "Initial program had implemented typed_equals_counterfactual only as probe, not supplied A/B relationship. New exact same-world/query single-typed-equals pairs establish the corpus witness in13,18,19. No target or first judgment changed; initial diagnostic is retained.",
              "supplied_typed_pairs": corpus_typed, "field_order_evidence": fields,
              "declared_probe_relationships_independently_executed": probes,
              "freshness": freshness, "counts": counts, "remaining_errors": errors,
              "empirical_conditional_flag_tables": [invariant_hist(records, axes) for axes in [
                  ["query_id"], ["position"], ["fact_name"], ["query_id", "position", "fact_name"],
                  ["support"], ["support", "observed"], ["decoded_role", "context"], ["world_count", "observed"]]],
              "substantive_inspection_ids": [r["sample_id"] for r in representative] + [r["sample_id"] for r in records if r["sample_id"].endswith("A-s26-reverse")],
              "substantive_examples": [{"sample_id": r["sample_id"], "record_hash": r["record_hash"], "observations": r["task"]["observations"],
                                        "expected": r["facts"]["expected"], "metrics": r["facts"]["metrics"]} for r in representative],
              "no_dataset_verdict_yet": True}
    output = ROOT / "reviews/expert/belief-gap-v1.supplemental_inspection.json"
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(canon({"path": str(output), "sha256": sha(output.read_bytes()), "remaining_errors": len(errors), "counts": counts}))
    # Preserve exact large evidence; deterministic gzip and actual round-trip.
    compression = []
    for p in (PREDICTIONS, COMPARISON, output):
        data = p.read_bytes()
        if len(data) >= 20 * 1024 * 1024:
            compressed = gzip.compress(data, compresslevel=9, mtime=0)
            cp = pathlib.Path(str(p) + ".gz")
            assert not cp.exists()
            cp.write_bytes(compressed)
            assert gzip.decompress(cp.read_bytes()) == data and len(compressed) <= 25 * 1024 * 1024
            compression.append({"raw_path": str(p), "raw_sha256": sha(data), "raw_length": len(data),
                                "compressed_path": str(cp), "compressed_sha256": sha(compressed), "compressed_length": len(compressed),
                                "gzip_mtime": 0, "roundtrip_byte_equal": True, "decoded_sha256": sha(gzip.decompress(cp.read_bytes()))})
    manifest = ROOT / "reviews/expert/belief-gap-v1.evidence_compression_manifest.json"
    assert not manifest.exists()
    manifest.write_text(json.dumps({"actual_author": "/root/expert_reviewer", "artifacts": compression, "raw_evidence_preserved": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
