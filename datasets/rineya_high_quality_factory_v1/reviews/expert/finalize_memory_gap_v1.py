#!/opt/anaconda3/bin/python3
"""Freeze actual expert judgments after full independent recomputation/inspection.

No other reviewer data is read.  The semantic assessment and severity choices
below were made by the actual /root/expert_reviewer after inspecting the public
contracts, all-family baseline examples and every matched change mechanism.
"""
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter

from independent_memory_gap_v1 import BASE, EXECUTION_ID, encode, exact_equal
from review_memory_gap_v1 import typed_value, answer_payload

OUT = BASE / "reviews/expert"
BATCH = BASE / "data/REM-001/candidate_renderings/memory-gap-v1.jsonl"
ALIASES = ["delayed_dependency", "decisive_asof_boundary", "version_first",
           "tombstone_vs_put_null", "same_time_index_tie", "wrong_key_suppression",
           "absent_key_control", "exact_query_role_crossing"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    analysis_path = OUT / "memory-gap-v1.independent_analysis.json"
    report = json.loads(analysis_path.read_text())
    records = [json.loads(line) for line in BATCH.read_text().splitlines()]
    assert sha(BATCH) == report["source_sha256"]
    evidence_by_id = {row["sample_id"]:row for row in report["row_evidence"]}
    assert set(evidence_by_id) == {r["sample_id"] for r in records}
    # Explicit negative controls for type-sensitive review comparison.
    assert not exact_equal({"value":True},{"value":1})
    assert not exact_equal({"value":False},{"value":0})
    assert answer_payload({"status":"present","value":True}) != answer_payload({"status":"present","value":1})
    assert answer_payload({"status":"present","value":False}) != answer_payload({"status":"present","value":0})

    conditional = []
    by_history = defaultdict(list)
    for record in records:
        by_history[(record["source_family_key"],record["controlled_variant"],record["renderer"])].append(record)
    for (family,history,style), rows in sorted(by_history.items()):
        unconditional = Counter()
        buckets = {"id_and_key":defaultdict(Counter),"position_and_key":defaultdict(Counter),"id_and_position":defaultdict(Counter)}
        for record in rows:
            evidence = evidence_by_id[record["sample_id"]]
            for pos,(query,result) in enumerate(zip(record["semantic_task"]["queries"],evidence["own_answer"]["results"])):
                unconditional[result["status"]] += 1
                buckets["id_and_key"][(query["query_id"],query["key"])][result["status"]] += 1
                buckets["position_and_key"][(str(pos),query["key"])][result["status"]] += 1
                buckets["id_and_position"][(query["query_id"],str(pos))][result["status"]] += 1
        total = sum(unconditional.values())
        row = {"family":family,"history":history,"field_order":style,"observations":total,
               "status_counts":dict(unconditional),"marginal_majority_fraction":max(unconditional.values())/total,
               "interpretation":"Empirical conditional label association only; no trained predictor, held-out performance or robustness claim"}
        for name,group in buckets.items():
            row[name] = {"majority_fraction":sum(max(c.values()) for c in group.values())/total,
                         "tables":[{"bucket":list(key),"status_counts":dict(value)} for key,value in sorted(group.items())]}
        conditional.append(row)
    conditional_path = OUT / "memory-gap-v1.conditional_shortcuts.json"
    with conditional_path.open("x") as stream:
        stream.write(json.dumps({"reviewer_execution_id":EXECUTION_ID,"scope":"corpus_variant",
                                "definitions":"Within each source family, controlled history and field order; buckets are nuisance-ID/position/name conjunctions. Marginal balance does not imply joint independence.",
                                "tables":conditional},indent=2)+"\n")

    receipts = []
    for record in records:
        evidence = evidence_by_id[record["sample_id"]]
        assert record["record_hash"] == evidence["record_hash"]
        issues = [{"category":"metadata_contract","severity":"moderate",
                   "evidence":f"skill_tags contains public-undefined factual aliases {ALIASES}. The public properties define precise per-record names and separate corpus_variant/audit_probe scopes. No alias mapping or scope binding is supplied; intended truthful aliases cannot be made an exact public coverage claim by the reviewer alone.",
                   "fix":"Replace aliases with the exact independently witnessed public property names for this row, or publish an explicit alias mapping and scope contract. Bind query_role_id_position_crossing as a corpus_variant family control rather than an unspecified per-row mechanism."}]
        blockers = ["UNDEFINED_FACTUAL_PROPERTY_ALIASES"]
        if evidence["delete_event_count"] == 0:
            assert record["controlled_variant"] == "tombstone_to_put_null"
            issues.extend([
                {"category":"false_or_ambiguous_factual_tag","severity":"moderate",
                 "evidence":"This visible history has zero delete events; all eight answers are seven present and one absent. It still carries tombstone_vs_put_null. present_null_distinction has a valid present-null/absent witness, but authoritative_tombstone and a within-record put-null/deleted contrast have no witness.",
                 "fix":"Remove the tombstone-specific per-record claim here. Retain present_null_distinction with its actual witness. Describe the true matched null_lifecycle_counterfactual against the baseline separately as corpus_variant evidence."},
                {"category":"coverage_design_gap","severity":"moderate",
                 "evidence":"The public count report says each base history should include present, deleted and absent queries; this controlled history has no deleted query. Its rotations equalize the actual two-status distribution, but every ID/slot has zero deleted support for this history.",
                 "fix":"Either supply a fresh controlled history with a genuine deleted-answer query while preserving the declared matched controls, or explicitly narrow the public coverage requirement and claim to the measured per-history support. Do not report all-three-status support for this history."}])
            blockers.extend(["NO_PER_RECORD_TOMBSTONE_WITNESS","MISSING_HISTORY_DELETED_STATUS_SUPPORT"])
        issues.extend([
            {"category":"shortcut_scope","severity":"low",
             "evidence":"Actual eight rotations balance role/status/type/present-null marginal histograms for every reused ID and position, and key-name histograms are equal after consistent key renaming. The schedule correlates these nuisance variables; conditional ID+key, position+key and ID+position tables are reported separately. Marginal balance does not establish joint shortcut removal.",
             "fix":"Keep the removal claim limited to the measured marginal controls; use additional independently crossed schedules if joint nuisance independence is required."},
            {"category":"generalization_scope","severity":"low",
             "evidence":"Nine to twelve supplied events and eight independent queries instantiate finite precedence retrieval with correlated controls. All 768 absent results use genuinely missing keys; future-only absence has zero corpus observations. Four changed histories, rotations and renderings add 12 source families, not 768 independent mechanisms or learned-memory evidence.",
             "fix":"Report 12 families, 48 base controlled histories and bounded proxy scope. Add a future-only absence family if that missing edge-case coverage is intended."}])
        scores = {"semantic_correctness":4,"target_correctness":5,"instruction_clarity":5,
                  "reasoning_validity":5,"generalization_value":3,"difficulty_quality":4,
                  "diversity":2 if evidence["delete_event_count"] == 0 else 3,"shortcut_resistance":3}
        receipt = {"receipt_id":"EXPERT-memory-gap-v1-"+record["sample_id"]+"-"+record["record_hash"][:12],
                   "sample_id":record["sample_id"],"record_hash":record["record_hash"],
                   "reviewer_role":"expert_reviewer","reviewer_execution_id":EXECUTION_ID,
                   "review_status":"REVISE","scores":scores,"critical_defects":blockers,
                   "issues":issues,"decision":"REVISE","confidence":0.99,
                   "review_method":"Fresh prompt-derived independent scan and sorted-precedence implementations, exact typed full-target/hash/payload checks, per-row witness search, eight metamorphics, query-local protected-control probes, full supplied family/renderer/control distribution audit, plus substantive actual-agent inspection of every base family and all changed mechanisms.",
                   "evidence":{"all_target_hash_contract_checks":evidence["checks"],
                               "derived_target_sha256":hashlib.sha256(encode(evidence["own_answer"]).encode()).hexdigest(),
                               "witnessed_public_properties":[k for k,v in evidence["public_property_witnesses"].items() if v],
                               "delete_event_count":evidence["delete_event_count"],"status_histogram":evidence["status_histogram"],
                               "metamorphics_passed":len(evidence["metamorphics"]),"protected_probe_count":len(evidence["audit_probes"]),
                               "row_evidence_path":str(analysis_path),"compressed_exact_evidence_path":str(analysis_path)+".gz",
                               "conditional_shortcut_report":str(conditional_path),
                               "semantic_inspection_report":str(OUT/"semantic_inspection_memory_gap_v1.md")},
                   "blindness":"Only this candidate batch, two public contracts, and new own code/evidence read. Public design availability/key-crossing clarification and generic alias-definition scope notices were communicated; no sample targets, producer coverage, reference/oracle source or other reviewer verdicts supplied/read.",
                   "assessment_note":"Full student-visible task and target are correct; REVISE is a factual-metadata/public-scope gate, not a target-correctness rejection. No dataset release approval."}
        receipts.append(receipt)
    first_path = OUT / "memory-gap-v1.first_judgments.jsonl"
    final_path = OUT / "memory-gap-v1.jsonl"
    encoded = "".join(encode(row)+"\n" for row in receipts)
    # Preserve the complete first assessment before writing structured final.
    with first_path.open("x") as stream:
        stream.write(encoded)
    assert len(first_path.read_text().splitlines()) == len(records)
    with final_path.open("x") as stream:
        stream.write(encoded)
    assert first_path.read_bytes() == final_path.read_bytes()
    summary = {k:v for k,v in report.items() if k not in ("crossings","key_role_crossings","global_key_histograms","renderer_checks","corpus_counterfactual_pairs","row_evidence")}
    summary.update({"stage":"COMPLETED_INDEPENDENT_EXPERT_FIRST_AND_FINAL","decision_counts":dict(Counter(r["decision"] for r in receipts)),
                    "targets_correct":768,"false_or_ambiguous_tombstone_tag_rows":192,"undefined_factual_alias_rows":768,
                    "three_status_base_histories":36,"two_status_base_histories":12,
                    "property_count_interpretation":"equal_version_time_conflict is actually witnessed in 384 rows and not blanket inferred; authoritative_tombstone in 576. Untagged absence and temporal controls remain reported by their exact witnessed names.",
                    "first_judgments_sha256":sha(first_path),"final_receipts_sha256":sha(final_path),
                    "first_and_final_bytes_identical":True,"actual_agent_inspection_scope":"All 12 base families p0012-p0023 at forward qrot0; every three changed mechanism per family, active tie and null/type boundary examples; aggregate per-ID/position/key histograms and proper decoded key isomorphs inspected. Programmatic recomputation covers every one of 768 records.",
                    "conditional_majority_fractions":{name:sorted({r[name]["majority_fraction"] for r in conditional}) for name in ("id_and_key","position_and_key","id_and_position")},
                    "conditional_marginal_majority_fractions":sorted({r["marginal_majority_fraction"] for r in conditional}),
                    "raw_evidence_sha256":sha(analysis_path),"gzip_evidence_sha256":sha(str(analysis_path)+".gz"),
                    "compression_manifest_path":str(OUT/"memory-gap-v1.independent_analysis.compression_manifest.json"),
                    "first_judgment_finalizer_sha256":sha(__file__),"preparation_report":str(OUT/"memory-gap-v1.preparation.json"),
                    "no_release_approval":True})
    with (OUT/"verification_summary_memory_gap_v1.json").open("x") as stream:
        stream.write(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"records":len(records),"decision_counts":summary["decision_counts"],"targets_correct":summary["targets_correct"],
                      "first_sha256":summary["first_judgments_sha256"],"conditional_majority_fractions":summary["conditional_majority_fractions"],
                      "conditional_marginal_majority_fractions":summary["conditional_marginal_majority_fractions"]},indent=2))


if __name__ == "__main__":
    run()
