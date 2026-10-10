#!/opt/anaconda3/bin/python3
"""Actual corrective expert post-freeze quality audit and immutable first verdicts.
Uses only own frozen evidence, authorized canonical rows and public guidance.
"""
import collections
import datetime
import hashlib
import json
import pathlib
import runpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/expert"
OWN = runpy.run_path(str(OUT / "belief_gap_v2_metadata_support.py"))
canon, equal = OWN["canon"], OWN["equal"]
sha = lambda b: hashlib.sha256(b).hexdigest()


def reverse_objects(v):
    if isinstance(v, dict):
        return {k: reverse_objects(v[k]) for k in reversed(v)}
    if isinstance(v, list):
        return [reverse_objects(i) for i in v]
    return v


def order_tree(v):
    if isinstance(v, dict):
        return [list(v), [order_tree(i) for i in v.values()]]
    if isinstance(v, list):
        return [order_tree(i) for i in v]
    return v


def subgroup(record):
    ident = record["visible_identity"]
    family, split = ident["semantic_family_id"], ident["split"]
    m = len(record["task"]["observations"])
    return ["all", canon(["family", family]), canon(["split", split]), canon(["length", m]),
            canon(["family_length", family, m]), canon(["split_length", split, m])]


def diagnostics(records):
    rules = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    hists = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)))
    scalars = sorted({v for r in records for v in r["input_only_frozen_diagnostics"]["input_features"]["scalar_occurrences"]})
    position_values = sorted({canon(v) for r in records for v in r["input_only_frozen_diagnostics"]["input_features"]["position_values"]})
    baselines = {"single_scalar:" + v: collections.defaultdict(collections.Counter) for v in scalars}
    baselines.update({"scalar_position:" + v: collections.defaultdict(collections.Counter) for v in position_values})
    for name in ("first_prefix_ids", "first_prefix_count", "coarse_counts_first_prefix", "ordered_scalar_values_lookup", "ordered_fact_scalar_values_lookup"):
        baselines[name] = collections.defaultdict(collections.Counter)
    for r in records:
        f = r["input_only_frozen_diagnostics"]["input_features"]
        empty = not r["facts"]["expected"]["surviving_worlds"]
        outcome = "empty" if empty else "nonempty"
        md = r["metadata"]
        inverse = {v: k for k, v in md["fact_role_assignments"].items()}
        obs_roles = [[j, inverse[o["fact"]], OWN["scalar"](o["equals"])] for j, o in enumerate(r["task"]["observations"])]
        full_pred = r["input_only_frozen_diagnostics"]["abstaining_full_output_predictions"]
        for scope in subgroup(r):
            for name, pred in r["input_only_frozen_diagnostics"]["empty_predictions"].items():
                c = rules[name][scope]
                c["records"] += 1
                c["classification_matches"] += pred == empty
                c["classification_errors"] += pred != empty
                if full_pred[name] is not None:
                    c["full_target_covered"] += 1
                    exact = equal(full_pred[name], r["facts"]["expected"])
                    c["full_target_exact"] += exact
                    c["full_target_errors"] += not exact
                    c["query_rows_covered"] += len(r["task"]["queries"])
                    c["query_rows_exact"] += sum(equal(a, b) for a, b in zip(full_pred[name]["queries"], r["facts"]["expected"]["queries"]))
            cells = {
                "scalar_anywhere": f["scalar_occurrences"],
                "scalar_position": [canon(v) for v in f["position_values"]],
                "fact_name_anywhere": sorted({o["fact"] for o in r["task"]["observations"]}),
                "fact_role_anywhere": sorted({v[1] for v in obs_roles}),
                "role_position_value": [canon(v) for v in obs_roles],
                "name_position_value": [canon(v) for v in f["position_fact_values"]],
                "world_count": [canon(f["world_count"])], "fact_count": [canon(f["fact_count"])],
                "query_count": [canon(f["query_count"])], "observation_length": [canon(f["observation_length"])],
                "first_prefix_ids": [canon(f["first_prefix_survivors"])], "first_prefix_count": [canon(f["first_prefix_count"])],
                "coarse_counts_first_prefix": [canon([f[k] for k in ("world_count", "fact_count", "query_count", "observation_length", "first_prefix_survivors")])],
                "repeated_flags": [canon([i for i, o in enumerate(r["task"]["observations"]) if any(equal(o, p) for p in r["task"]["observations"][:i])])],
                "strict_filter_count_component_metric": [canon(len(r["facts"]["metrics"]["strict_nonempty_reduction_indices"]))],
                "final_payload_count_component_metric_not_input_baseline": [canon(r["facts"]["metrics"]["distinct_surviving_payloads"])]}
            for axis, buckets in cells.items():
                for bucket in buckets:
                    hists[scope][axis][bucket][outcome] += 1
        for token in scalars:
            baselines["single_scalar:" + token][canon(token in f["scalar_occurrences"])][outcome] += 1
        for token in position_values:
            baselines["scalar_position:" + token][canon(token in {canon(v) for v in f["position_values"]})][outcome] += 1
        baseline_cells = {"first_prefix_ids": f["first_prefix_survivors"], "first_prefix_count": f["first_prefix_count"],
                          "coarse_counts_first_prefix": [f[k] for k in ("world_count", "fact_count", "query_count", "observation_length", "first_prefix_survivors")],
                          "ordered_scalar_values_lookup": [v[1] for v in f["position_values"]],
                          "ordered_fact_scalar_values_lookup": f["position_fact_values"]}
        for name, cell in baseline_cells.items():
            baselines[name][canon(cell)][outcome] += 1
    fits = [{"name": name, "empirical_in_sample_status_majority_matches": sum(max(v.values()) for v in table.values()),
             "records": len(records), "bucket_count": len(table), "input_only_feature": True,
             "gold_final_support_used_as_feature": False, "outcome_buckets": {k: dict(v) for k, v in sorted(table.items())}} for name, table in sorted(baselines.items())]
    return {"frozen_public_rules": {name: {scope: dict(c) for scope, c in sorted(groups.items())} for name, groups in sorted(rules.items())},
            "actual_scalar_role_position_coarse_histograms": {scope: {axis: {bucket: dict(c) for bucket, c in sorted(cells.items())} for axis, cells in sorted(axes.items())} for scope, axes in sorted(hists.items())},
            "tested_input_baselines": fits, "interpretation": "Corpus diagnostic compatibility, not learned model performance; exploratory lookup fit measured after fixed public rules were frozen."}


def main():
    cp = OUT / "belief-gap-v2.post_freeze_comparison.json"
    pp = OUT / "belief-gap-v2.input_only_predictions.json"
    assert sha(cp.read_bytes()) == "9659bb63d9ba2e3c93c7d65809a25fc6fb957766b98422ba75cdcbe5bfab92d0"
    assert sha(pp.read_bytes()) == "282295962fd76052db303e012e88967d82db97c5f94919a9b2070ab80e2e1be5"
    d = json.loads(cp.read_text())
    rs = d["records"]
    assert d["target_mismatches"] == d["original_hash_mismatches"] == d["metadata_error_rows"] == 0
    assert not d["family_audits"]["errors"]
    candidate_path = ROOT / "data/RBM-001/candidate_renderings/belief-gap-v2.jsonl"
    rows = {r["sample_id"]: r for r in map(json.loads, candidate_path.read_text().splitlines())}
    old = json.loads((OUT / "belief-gap-v1.post_freeze_comparison.json").read_text())
    old_records = {r["sample_id"]: r for r in old["post_freeze_audit_records"]}
    changes = []
    field_pairs, extra_errors = [], []
    for r in rs:
        md, row = r["metadata"], rows[r["sample_id"]]
        allowed_levels = {2} if md["controlled_context"] in "GH" else {2, 3}
        if type(r["declared_difficulty"]) is not int or r["declared_difficulty"] not in allowed_levels or r["structural_difficulty"] != r["declared_difficulty"]:
            extra_errors.append({"sample_id": r["sample_id"], "error": "difficulty does not match revised component level"})
        split = "train" if md["source_family_index"] in (12, 13) else "dev_challenge"
        if row["split"] != split or row["semantic_parent_id"] != row["semantic_family_id"]:
            extra_errors.append({"sample_id": r["sample_id"], "error": "split/family custody mismatch"})
        if r["facts"]["metrics"]["duplicate_visible_payloads"]:
            extra_errors.append({"sample_id": r["sample_id"], "error": "duplicate complete world payload"})
        if md["renderer"] == "forward":
            other = rows[r["sample_id"].removesuffix("forward") + "reverse"]
            a, at = row["rendered_input"].split("Task JSON:\n")
            b, bt = other["rendered_input"].split("Task JSON:\n")
            okay = a == b and row["rendered_input"] != other["rendered_input"] and equal(order_tree(reverse_objects(json.loads(at))), order_tree(json.loads(bt)))
            field_pairs.append({"sample_ids": [r["sample_id"], other["sample_id"]], "pass": okay})
            if not okay:
                extra_errors.append({"sample_id": r["sample_id"], "error": "field renderer not recursive key-order-only pair"})
        previous_id = r["sample_id"].replace("belief-gap-v2", "belief-gap-v1")
        if previous_id in old_records:
            prev = old_records[previous_id]
            assert equal(prev["task"]["worlds"], r["task"]["worlds"])
            changes.append({"new_sample_id": r["sample_id"], "old_sample_id": previous_id, "typed_task_changed": not equal(prev["task"], r["task"]),
                            "old_observations": prev["task"]["observations"], "new_observations": r["task"]["observations"]})
    audit = diagnostics(rs)
    fixed = audit["frozen_public_rules"]
    compound = fixed["prospective_marker_and_auxiliary_token_empty"]["all"]
    original = fixed["v1_original_scalar_marker_empty"]["all"]
    assert compound["classification_matches"] == 1728 and compound["full_target_covered"] == compound["full_target_exact"] == 432
    assert original["classification_errors"] == 432
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stats = {"records": len(rs), "source_families": len({r["metadata"]["source_family_key"] for r in rs}),
             "base_contexts": len({(r["metadata"]["source_family_key"], r["metadata"]["controlled_context"]) for r in rs}),
             "unique_typed_tasks": len({canon(r["task"]) for r in rs}), "contexts": dict(collections.Counter(r["metadata"]["controlled_context"] for r in rs)),
             "splits": dict(collections.Counter(r["visible_identity"]["split"] for r in rs)),
             "empty_records": sum(not r["facts"]["expected"]["surviving_worlds"] for r in rs),
             "nonempty_records": sum(bool(r["facts"]["expected"]["surviving_worlds"]) for r in rs),
             "field_pairs": len(field_pairs), "all_field_pairs_pass": all(p["pass"] for p in field_pairs),
             "nuisance_context_renderer_groups": len(d["family_audits"]["histograms"]),
             "actual_typed_payload_provenance_histogram_axis_passes": sum(t["identical"] for h in d["family_audits"]["histograms"].values() for t in h["outcome_histograms"].values()),
             "matched_empty_nonempty_pairs": sum(len(c["same_prefix_empty_nonempty_counterfactual"]) for c in d["family_audits"]["family_control_witnesses"].values()),
             "new_families_relative_v1": 0, "new_G_H_records": 432,
             "existing_A_F_task_changes": sum(c["typed_task_changed"] for c in changes),
             "property_tag_counts": dict(collections.Counter(t for r in rs for t in r["metadata"]["skill_tags"])),
             "invariance_probes": sum(len(r["probes"]["invariance_probes"]) for r in rs),
             "typed_equals_probes": sum(len(r["probes"]["typed_equals_counterfactual_probes"]) for r in rs),
             "G_H_actual_final_support_distribution": dict(collections.Counter(str(r["facts"]["metrics"]["final_support"]) for r in rs if r["metadata"]["controlled_context"] in "GH")),
             "G_H_strict_reduction_counts": dict(collections.Counter(str(len(r["facts"]["metrics"]["strict_nonempty_reduction_indices"])) for r in rs if r["metadata"]["controlled_context"] in "GH"))}
    report = {"report_type": "CORRECTIVE_V2_EXPERT_SHORTCUT_AND_QUALITY_EVIDENCE", "actual_author": "/root/expert_reviewer", "completed_utc": now,
              "prediction_sha256": sha(pp.read_bytes()), "comparison_sha256": sha(cp.read_bytes()), "candidate_sha256": sha(candidate_path.read_bytes()),
              "source_sha256": sha(pathlib.Path(__file__).read_bytes()), "statistics": stats,
              "diagnostics": audit, "actual_changes_relative_frozen_own_v1": changes,
              "field_order_pairs": field_pairs, "remaining_correctness_metadata_errors": extra_errors,
              "substantive_ids": [r["sample_id"] for r in rs if r["sample_id"].endswith("s00-forward") or r["sample_id"].endswith("G-s26-reverse")],
              "blocker": {"category": "compound_scalar_marker_shortcut", "affected_contexts": ["E", "F"], "affected_rows": 432,
                          "fixed_public_rule": "any exact string stone/unknown observation equality AND any exact string linen equality",
                          "classification_matches": 1728, "full_correct_empty_targets": 432,
                          "uses_world_fact_payloads": False, "uses_gold": False,
                          "scope": "Perfect emptiness classifier on this frozen corpus and exact full-output shortcut on all E/F. No claim of learned model behavior or general-validity."},
              "no_crossreview_private_verdict_read": True, "no_release_adjudication_approval": True}
    report["own_pre_judgment_failed_audit_attempt"] = "belief-gap-v2.inspect_attempt1_failure.json; unsupported all-A-F-Level3 assumption removed, preserving source. Lower Level2 for source13 B is truthful. No prediction/verdict changed."
    assert not extra_errors
    ep = OUT / "belief-gap-v2.semantic_quality_evidence.json"
    assert not ep.exists()
    ep.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    evidence_hash = sha(ep.read_bytes())
    pred_hash, comparison_hash = sha(pp.read_bytes()), sha(cp.read_bytes())
    receipts = []
    for r in rs:
        md, f = r["metadata"], r["facts"]
        bad = md["controlled_context"] in "EF"
        decision = "REVISE" if bad else "ACCEPT"
        issues = [{"category": "limited_independent_diversity", "severity": "medium",
                   "evidence": "1,728 correlated renderings derive from four unchanged source tables and 32 base contexts; query shifts and G/H add zero families. Shared null/opal/linen motifs and cyclic row patterns persist.",
                   "fix": "Report four source families and retain bounded proxy scope; future coverage should vary structural partitions and semantic-role outcomes."}]
        if bad:
            issues.append({"category": "compound_scalar_marker_shortcut", "severity": "high",
                           "evidence": "The pre-data public marker-plus-linen rule perfectly classifies emptiness on all 1,728 rows. On this E/F row it plus world count, observation count and supplied query IDs reproduces the entire verified target without examining any world facts. All 432 E/F targets are reproduced this way.",
                           "fix": "Publish and generate a further controlled revision that crosses this compound literal association, preserving v2. Additional table-consistent scalar/semantic variations are needed; adding the same unchanged marker/linen conjunction within the same table cannot make its truth differ. Re-review exact new hashes."})
        else:
            issues.append({"category": "batch_shortcut_limit", "severity": "medium",
                           "evidence": "The G/H pairs break the original single-marker rule, but a frozen compound marker-plus-linen rule still perfectly classifies batch emptiness. It abstains from this nonempty row's full counts/provenance, which independently recompute correctly. This ACCEPT does not approve the batch's empty-control distribution.",
                           "fix": "Keep this row's honest scope, and correct the E/F discriminative defect before claiming the whole corrective batch passes release."})
        issues.append({"category": "finite_proxy_scope", "severity": "low", "evidence": "Exact unweighted noiseless filtering and possible-value partitions; no Bayes, authority, chosen inspection/utility, sealed OOD or efficacy evidence.", "fix": "Retain the public scope limitation."})
        scores = {"semantic_correctness": 5, "target_correctness": 5, "instruction_clarity": 5, "reasoning_validity": 5,
                  "generalization_value": 2 if bad else 3, "difficulty_quality": 2 if bad else 3 if md["controlled_context"] == "D" else 4,
                  "diversity": 3, "shortcut_resistance": 1 if bad else 3}
        receipts.append({"receipt_id": "expert-belief-gap-v2-" + sha(canon([r["sample_id"], r["record_hash"], "/root/expert_reviewer"]).encode())[:24],
                         "sample_id": r["sample_id"], "record_hash": r["record_hash"], "reviewer_role": "expert_reviewer", "reviewer_execution_id": "/root/expert_reviewer",
                         "review_status": "REVISE" if bad else "ACCEPTED", "scores": scores,
                         "critical_defects": ["compound_scalar_marker_bypasses_empty_belief_reasoning"] if bad else [],
                         "issues": issues, "decision": decision, "confidence": 0.985, "completed_utc": now,
                         "review_method": "New own direct and sequential typed solvers on eight-field stripped view, expected outputs frozen before gold/metadata; all row/property/locator/control checks and substantive actual-agent inspection.",
                         "independence_scope": "Corrective issue-guided review with disclosed public v1 finding and root diagnostic hypotheses; not fully blind. No other private reviewer verdicts or producer/reference/oracle/checker files read. Same underlying-model correlation risk remains.",
                         "evidence": {"prediction_sha256": pred_hash, "comparison_sha256": comparison_hash, "semantic_quality_evidence_sha256": evidence_hash,
                                      "type_sensitive_target_equal": True, "full_original_candidate_hash_verified": True,
                                      "prefix_survivors": f["prefix_survivors"], "query_partitions": f["expected"]["queries"],
                                      "property_tags_locators_and_family_claims_pass": True, "row_component_metrics": f["metrics"],
                                      "declared_difficulty": r["declared_difficulty"], "context": md["controlled_context"],
                                      "frozen_input_diagnostic_predictions": r["input_only_frozen_diagnostics"]["empty_predictions"],
                                      "compound_full_target_prediction_equals_own_expected": bad and equal(r["input_only_frozen_diagnostics"]["abstaining_full_output_predictions"]["prospective_marker_and_auxiliary_token_empty"], f["expected"]),
                                      "inspection_report": "reviews/expert/semantic_inspection_belief_gap_v2.md"}})
    raw = ("\n".join(canon(r) for r in receipts) + "\n").encode()
    first, final = OUT / "belief-gap-v2.first_judgments.jsonl", OUT / "belief-gap-v2.jsonl"
    assert not first.exists() and not final.exists()
    first.write_bytes(raw)
    final.write_bytes(raw)
    summary = {"actual_author": "/root/expert_reviewer", "completed_utc": now, "review_mode": "CORRECTIVE_PUBLIC_ISSUE_AND_DIAGNOSTIC_GUIDED",
               "candidate_sha256": sha(candidate_path.read_bytes()), "prediction_sha256": pred_hash, "comparison_sha256": comparison_hash,
               "quality_evidence_sha256": evidence_hash, "first_final_sha256": sha(raw), "first_final_byte_equal": True,
               "decisions": dict(collections.Counter(r["decision"] for r in receipts)), "statistics": stats,
               "frozen_rule_results": {name: scopes["all"] for name, scopes in audit["frozen_public_rules"].items()},
               "correctness_target_hash_property_metadata_errors": 0, "quality_blocker": report["blocker"],
               "substantive_inspection_ids": report["substantive_ids"], "no_release_adjudication_approval": True,
               "old_v1_first_judgments_preserved_sha256": sha((OUT / "belief-gap-v1.first_judgments.jsonl").read_bytes()),
               "old_v1_final_receipts_preserved_sha256": sha((OUT / "belief-gap-v1.jsonl").read_bytes()),
               "source_sha256": sha(pathlib.Path(__file__).read_bytes())}
    sp = OUT / "verification_summary_belief_gap_v2.json"
    assert not sp.exists()
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(canon({"first_final_sha256": sha(raw), "decisions": summary["decisions"], "quality_evidence_sha256": evidence_hash,
                 "statistics": stats, "baseline_fits": [{k:v for k,v in b.items() if k != "outcome_buckets"} for b in audit["tested_input_baselines"]],
                 "substantive_examples": [{"id":r["sample_id"], "obs":r["task"]["observations"], "prefix_counts":r["facts"]["expected"]["belief_sizes"],
                                           "bins":[q["counts"] for q in r["facts"]["expected"]["queries"]], "difficulty":r["declared_difficulty"]} for r in rs if r["sample_id"].endswith("s00-forward")]}))


if __name__ == "__main__":
    main()
