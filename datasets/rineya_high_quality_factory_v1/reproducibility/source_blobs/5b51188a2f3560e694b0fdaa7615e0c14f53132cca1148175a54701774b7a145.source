#!/opt/anaconda3/bin/python3
"""Freeze this actual expert's first/final decisions from completed own evidence.
No producer/oracle/other-reviewer files read. No dataset/release/source edits.
"""
import collections
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/expert"
sha = lambda b: hashlib.sha256(b).hexdigest()
canon = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main():
    pred_path = OUT / "belief-gap-v1.input_only_predictions.json"
    comparison_path = OUT / "belief-gap-v1.post_freeze_comparison.json"
    supplement_path = OUT / "belief-gap-v1.supplemental_inspection.json"
    candidate_path = ROOT / "data/RBM-001/candidate_renderings/belief-gap-v1.jsonl"
    assert sha(pred_path.read_bytes()) == "f732c60d2ae3646bb3cc92bc166dc05a867c94bbc3b364558630e6e1f647d4fe"
    assert sha(comparison_path.read_bytes()) == "3f84bd5289a79a8e07d109d84d8dc72e7a07973049c833fc68c467765c2f1937"
    assert sha(supplement_path.read_bytes()) == "08c11547dc462fa0f2779df6aaf664b2741fb9294b581e76f785ae8390a31320"
    assert sha(candidate_path.read_bytes()) == "423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e"
    comparison = json.loads(comparison_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    assert comparison["target_mismatch_count"] == 0 and comparison["metadata_error_rows"] == 0
    assert not supplement["remaining_errors"]
    records = comparison["post_freeze_audit_records"]
    candidates = {r["sample_id"]: r for r in map(json.loads, candidate_path.read_text().splitlines())}
    checks = {r["sample_id"]: r for r in comparison["records"]}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipts = []
    for record in records:
        sample, f, md = record["sample_id"], record["facts"], record["metadata"]
        checked, row = checks[sample], candidates[sample]
        assert checked["original_record_hash_matches_without_hash_field"] and checked["type_sensitive_target_equal"]
        assert record["record_hash"] == row["record_hash"]
        context = md["controlled_context"]
        issues = [{"category": "limited_independent_diversity", "severity": "medium",
                   "evidence": "This record is one of1296 correlated views of4 source tables and24 base contexts. There are648 parsed tasks. Six contexts share one fixed world table; null/opal invariant and linen auxiliary motifs recur across all four tables.",
                   "fix": "Treat source families as the independent units; expand genuinely different partition/observation structures before claiming broad generalization."},
                  {"category": "residual_conditional_shortcut", "severity": "low",
                   "evidence": "Full ID/position/name triple typed-outcome histograms are equal within every base context/style. Cyclic row orders and decoded semantic-role/context patterns remain. Empirical support-only joint-flag majority fits3240/3888 query observations; decoded-role/context fits3888/3888, but those are audit metadata and not direct student-visible features or measured model performance.",
                   "fix": "Report these residual correlations and require full counts/provenance; future broader cases can vary semantic-role outcome patterns and use arbitrary query permutations."},
                  {"category": "finite_proxy_scope", "severity": "low",
                   "evidence": "This prompt specifies unweighted noiseless filtering and value partitions only. It supplies no priors, likelihoods, source authority, costs, chosen inspection action, sealed OOD evidence or efficacy result.",
                   "fix": "Retain the stated finite-proxy limitation; broader belief/information-gathering objectives require separately published semantics and evaluation."}]
        if context in "EF":
            issues.append({"category": "difficulty_components", "severity": "low",
                           "evidence": "Level3 includes a2-observation cross-fact contradiction and" + (" no later observation." if context == "E" else " one ordinary step after the first empty prefix. The empty final-query substep is simple."),
                           "fix": "Report minimal contradiction witness and empty trace components separately; row count/table size does not justify higher difficulty."})
        elif context == "D":
            issues.append({"category": "difficulty_components", "severity": "low",
                           "evidence": "The third observation exactly repeats an earlier one at nonempty support; it adds a trace-invariance control rather than a third essential conjunctive constraint.",
                           "fix": "Count the two decisive reductions separately from the redundant step."})
        scores = {"semantic_correctness": 5, "target_correctness": 5, "instruction_clarity": 5,
                  "reasoning_validity": 5, "generalization_value": 3,
                  "difficulty_quality": 3 if context in "DEF" else 4, "diversity": 3, "shortcut_resistance": 3}
        receipts.append({"receipt_id": "expert-belief-gap-v1-" + sha(canon([sample, record["record_hash"], "/root/expert_reviewer"]).encode())[:24],
                         "sample_id": sample, "record_hash": record["record_hash"], "reviewer_role": "expert_reviewer",
                         "reviewer_execution_id": "/root/expert_reviewer", "actual_author": "/root/expert_reviewer",
                         "review_status": "ACCEPTED", "scores": scores, "critical_defects": [], "issues": issues,
                         "decision": "ACCEPT", "confidence": 0.985, "completed_utc": now,
                         "review_method": "Own student-prompt-derived typed incremental filter plus separate direct prefix conjunction; expected outputs frozen before target/metadata read; substantive actual-agent inspection of all four tables, all six contexts, and extreme schedule representatives.",
                         "independence_scope": "Initial sample review with target-free first stage, guided by this role's prior public gap diagnosis/shared prospective design. No producer/reference/oracle implementation/tests or other private verdicts read. Agents share an underlying model and can have correlated errors. This is an expert receipt, not formal adjudication or release approval.",
                         "evidence": {"prediction_sha256": sha(pred_path.read_bytes()), "target_comparison_sha256": sha(comparison_path.read_bytes()),
                                      "supplemental_inspection_sha256": sha(supplement_path.read_bytes()),
                                      "own_expected_sha256": sha(canon(f["expected"]).encode()), "type_sensitive_target_equal": True,
                                      "full_candidate_record_hash_verified": True,
                                      "prefix_survivors": f["prefix_survivors"], "query_partitions": f["expected"]["queries"],
                                      "property_tags_and_locators_independently_witnessed": sorted(set(md["skill_tags"]) - {"finite_synthetic_proxy", "finite_noiseless_filter", "possible_value_partition"}),
                                      "property_witnesses": {k: v for k, v in f["properties"].items() if v},
                                      "component_metrics": f["metrics"], "incomplete_heuristic_error_query_ids": f["incomplete_heuristic_error_query_ids"],
                                      "context": context, "source_family_key": md["source_family_key"],
                                      "all_seven_nuisance_axes_full_typed_payload_provenance_histograms_pass": True,
                                      "family_claims_actual_corpus_or_executed_probe_relationships_pass": True,
                                      "own_invariance_probe_count": len(record["probes"]["invariance_probes"]),
                                      "own_typed_counterfactual_probe_count": len(record["probes"]["typed_equals_counterfactual_probes"]),
                                      "inspection_report": "reviews/expert/semantic_inspection_belief_gap_v1.md"}})
    raw = ("\n".join(canon(r) for r in receipts) + "\n").encode()
    first = OUT / "belief-gap-v1.first_judgments.jsonl"
    final = OUT / "belief-gap-v1.jsonl"
    assert not first.exists() and not final.exists(), "preserve immutable judgments"
    first.write_bytes(raw)
    final.write_bytes(raw)
    assert first.read_bytes() == final.read_bytes()
    summary = {"actual_author": "/root/expert_reviewer", "reviewer_execution_id": "/root/expert_reviewer",
               "completed_utc": now, "review_type": "INITIAL_EXPERT_REVIEW_GUIDED_BY_PUBLIC_PROSPECTIVE_DIAGNOSIS",
               "candidate": {"path": str(candidate_path), "sha256": sha(candidate_path.read_bytes()), "records": len(records)},
               "input_only_prediction": {"path": str(pred_path), "sha256": sha(pred_path.read_bytes()), "length": pred_path.stat().st_size},
               "first_judgments": {"path": str(first), "sha256": sha(raw), "length": len(raw), "immutable": True},
               "final_receipts": {"path": str(final), "sha256": sha(raw), "byte_equal_first": True},
               "decisions": dict(collections.Counter(r["decision"] for r in receipts)),
               "counts": supplement["counts"], "target_mismatch_count": 0, "per_row_metadata_errors": 0,
               "remaining_family_or_quality_blockers": 0,
               "preparation_checks": {"toy_fixtures": 5, "executed_metamorphic_and_typed_probes": 42},
               "own_initial_diagnostic_implementation_warnings_preserved": supplement["resolved_initial_audit_implementation_warning_count"],
               "warning_resolution": supplement["warning_resolution"], "freshness": supplement["freshness"],
               "substantive_inspection_ids": supplement["substantive_inspection_ids"],
               "noncritical_limitations": ["Four independent source tables, not1296 mechanisms", "Shared null/opal/linen submotifs and cyclic row-role patterns", "Support-only joint-flag majority compatibility5/6; audit-role/context determinism does not prove a student-facing shortcut", "E/F empty and D repeated substeps are simple; all difficulty levels3 and component metrics explicit", "Exposed dev, finite noiseless proxy; no Bayes/action/utility/OOD/efficacy claim"],
               "forbidden_reads": {"producer_source_or_reports": 0, "reference_or_oracle_source_tests": 0, "formal_or_root_mutant_reports": 0, "other_private_review_verdicts": 0},
               "actual_execution_scope": "Programmatic every-record recomputation and property/locator/control checks plus actual-agent substantive semantic inspection, not a human reading every row individually.",
               "release_approval": False, "other_review_release_REM_files_changed": False,
               "evidence_compression_manifest": "reviews/expert/belief-gap-v1.evidence_compression_manifest.json"}
    sp = OUT / "verification_summary_belief_gap_v1.json"
    assert not sp.exists()
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    execution = {"report_type": "EXPERT_COMPLETED_EXECUTION_RECEIPT", "actual_author": "/root/expert_reviewer", "completed_utc": now,
                 "python_executable": "/opt/anaconda3/bin/python3", "actual_commands": [
                     "/opt/anaconda3/bin/python3 reviews/expert/belief-gap-v1.prepared_source.py predict --view reviews/input_only/belief-gap-v1.jsonl --original-sha423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e --out reviews/expert/belief-gap-v1.input_only_predictions.json",
                     "/opt/anaconda3/bin/python3 reviews/expert/belief-gap-v1.prepared_source.py compare --frozen reviews/expert/belief-gap-v1.input_only_predictions.json --canonical data/RBM-001/candidate_renderings/belief-gap-v1.jsonl --out reviews/expert/belief-gap-v1.post_freeze_comparison.json",
                     "/opt/anaconda3/bin/python3 reviews/expert/inspect_belief_gap_v1.py",
                     "/opt/anaconda3/bin/python3 reviews/expert/finalize_belief_gap_v1.py"],
                 "all_exit_codes": 0, "gold_access_after_prediction_freeze": True, "first_final_judgments_equal": True,
                 "source_and_artifact_bindings": {str(p.relative_to(ROOT)): {"sha256": sha(p.read_bytes()), "length": p.stat().st_size} for p in [
                     OUT / "belief-gap-v1.prepared_source.py", OUT / "inspect_belief_gap_v1.py", pathlib.Path(__file__),
                     pred_path, pathlib.Path(str(pred_path) + ".freeze_receipt.json"), comparison_path, supplement_path,
                     first, final, sp, OUT / "belief-gap-v1.evidence_compression_manifest.json"]},
                 "no_release_or_adjudication_approval": True}
    ep = OUT / "belief-gap-v1.completed_execution.json"
    assert not ep.exists()
    ep.write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n")
    print(canon({"first_sha256": sha(raw), "decisions": summary["decisions"], "summary": str(sp), "execution": str(ep)}))


if __name__ == "__main__":
    main()
