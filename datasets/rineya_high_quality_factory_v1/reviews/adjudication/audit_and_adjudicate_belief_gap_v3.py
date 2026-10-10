"""Actual V3 adjudication: own public-derived solver and independent manifest audit."""
from pathlib import Path
from collections import Counter, defaultdict
import copy
import datetime
import gzip
import hashlib
import json

from audit_belief_gap_v1 import BASE, canonical, scalar, solve, property_audit, ids

OUT = BASE / "reviews/adjudication"
NAMES = ("same_prefix_empty_nonempty_counterfactual", "contradiction_equals_mixed_outcomes")
EXPECTED_CANDIDATE = "7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c"
bindings = {}

def read(name):
    raw = (BASE / name).read_bytes()
    bindings[name] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return raw

def json_file(name):
    return json.loads(read(name))

def json_rows(name):
    return [json.loads(line) for line in read(name).splitlines()]

def new_json(name, value):
    path = BASE / name
    assert not path.exists(), name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}

def transform(task, control):
    out = copy.deepcopy(task)
    y, b, g = control["fact"], control["baseline_value"], control["supported_value"]
    def swap(value):
        return g if scalar(value) == scalar(b) else b if scalar(value) == scalar(g) else value
    for world in out["worlds"]:
        world["facts"][y] = swap(world["facts"][y])
    for observation in out["observations"]:
        if observation["fact"] == y:
            observation["equals"] = swap(observation["equals"])
    return out

def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    code = read("reviews/adjudication/audit_and_adjudicate_belief_gap_v3.py")
    read("reviews/adjudication/audit_belief_gap_v1.py")
    public_names = ["schemas/EXTENDED_SEMANTICS.md", "schemas/BELIEF_CONTROL_PROPERTIES.md",
        "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md", "schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md",
        "schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json", "research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md",
        "research/curricula/RBM-EVIDENCE-GAP-REVISION-V3-ROOT_FREEZE.json"]
    for name in public_names:
        read(name)
    candidate_path = "data/RBM-001/candidate_renderings/belief-gap-v3.jsonl"
    candidates = json_rows(candidate_path)
    assert bindings[candidate_path]["sha256"] == EXPECTED_CANDIDATE
    assert len(candidates) == 3456
    by_id = {r["sample_id"]: r for r in candidates}
    assert len(by_id) == len(candidates)
    receipts = {}
    receipt_paths = {
        "expert_original": "reviews/expert/belief-gap-v3.jsonl",
        "expert_first": "reviews/expert/belief-gap-v3.first_judgments.jsonl",
        "expert_supplement": "reviews/expert/belief-gap-v3.inherited_claim_interface_supplement.jsonl",
        "expert_supplement_first": "reviews/expert/belief-gap-v3.inherited_claim_interface_supplement.first_judgments.jsonl",
        "red_original": "reviews/adversarial/belief-gap-v3.jsonl",
        "red_first": "reviews/adversarial/belief-gap-v3.first_judgments.jsonl",
        "red_supplement_index": "audits/execution/belief-gap-v3-inherited-interface-affected-records.jsonl",
        "oracle": "reviews/oracle/belief-gap-v3.jsonl"}
    for role, path in receipt_paths.items():
        values = json_rows(path)
        assert len(values) == 3456 and {r["sample_id"] for r in values} == set(by_id)
        receipts[role] = {r["sample_id"]: r for r in values}
        for r in values:
            assert r["record_hash"] == by_id[r["sample_id"]]["record_hash"]
    assert bindings[receipt_paths["expert_original"]]["sha256"] == bindings[receipt_paths["expert_first"]]["sha256"]
    assert bindings[receipt_paths["expert_supplement"]]["sha256"] == bindings[receipt_paths["expert_supplement_first"]]["sha256"]
    completed_paths = ["reviews/expert/belief-gap-v3.completed_execution.json",
        "reviews/expert/belief-gap-v3.inherited_claim_interface_supplement.execution.json",
        "reviews/expert/belief-gap-v3.inherited_claim_interface_supplement.json",
        "reviews/expert/semantic_inspection_belief_gap_v3.md",
        "reviews/expert/semantic_inspection_belief_gap_v3_inherited_claim_interface_supplement.md",
        "reviews/expert/belief-gap-v3.frequency_counterexamples.json",
        "audits/execution/belief-gap-v3-adversarial.json",
        "audits/execution/belief-gap-v3-supplement-adversarial.json",
        "audits/execution/belief-gap-v3-inherited-interface-adversarial.json",
        "reviews/oracle/RBM-v3-public-interface-interpretation-20261010T040233Z.json"]
    for name in completed_paths:
        read(name)
    machine = json_file("audits/diversity/belief-gap-v3-belief-controls.json")
    assert bindings["audits/diversity/belief-gap-v3-belief-controls.json"]["sha256"] == "f49dda4edacbc12ed35bb0929bdc12073808bb4912863573f56f11cbe73390fe"
    assert machine["checker_sha256"] == "ff1a72153e985810d2771fecf299944b39cc8afa52effbd833feac056f143271"
    leakage = json_file("audits/leakage/belief-gap-v3.json")
    assert leakage["oracle_failures"] == 0 and leakage["metadata_failures"] == 3456
    assert leakage["belief_control_error_count"] == len(machine["report"]["errors"]) == 3484
    read("audits/diversity/belief-gap-v3-belief-controls.compression_manifest.json")
    compressed_path = "audits/diversity/belief-gap-v3-belief-controls.json.gz"
    read(compressed_path)
    decompressed_hash, decompressed_size = hashlib.sha256(), 0
    with gzip.open(BASE / compressed_path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            decompressed_hash.update(chunk)
            decompressed_size += len(chunk)
    assert decompressed_hash.hexdigest() == bindings["audits/diversity/belief-gap-v3-belief-controls.json"]["sha256"]
    assert decompressed_size == bindings["audits/diversity/belief-gap-v3-belief-controls.json"]["bytes"]
    manifest_path = "data/RBM-001/manifests/belief-gap-v3.family_controls.json"
    manifest_raw = read(manifest_path)
    manifest = json.loads(manifest_raw)
    assert manifest["exact_candidate_sha256"] == EXPECTED_CANDIDATE
    prior = json_rows("data/RBM-001/candidate_renderings/belief-gap-v2.jsonl")
    prior_index = {(r["source_family_index"], r["controlled_context"], r["schedule_index"], r["renderer"]): r for r in prior}
    index, targets, prefixes, details = {}, {}, {}, []
    properties, status, rules, missing = Counter(), Counter(), defaultdict(Counter), Counter()
    decoded_tables, scalar_tables, visible_tables = set(), set(), set()
    for row in candidates:
        sid, task = row["sample_id"], row["semantic_task"]
        assert hashlib.sha256(canonical({k: v for k, v in row.items() if k != "record_hash"}).encode()).hexdigest() == row["record_hash"]
        assert canonical(json.loads(row["rendered_input"].split("Task JSON:\n", 1)[1])) == canonical(task)
        assert all(key not in row["rendered_input"] for key in ["auxiliary_scalar_variant", "family_control_claims", "property_witnesses", "controlled_context"])
        target, prefix = solve(task)
        assert canonical(target) == canonical(row["verified_target"])
        targets[sid], prefixes[sid] = target, prefix
        tags = property_audit(row, task, target, prefix)
        properties.update(tags)
        family, context, variant, schedule, renderer = (row[k] for k in ["source_family_index", "controlled_context", "auxiliary_scalar_variant", "schedule_index", "renderer"])
        key = (family, context, variant, schedule, renderer)
        assert key not in index
        index[key] = row
        assert family in (12, 13, 18, 19) and row["split"] == ("train" if family in (12, 13) else "dev_challenge")
        assert variant in ("identity", "transposed")
        control = row["auxiliary_scalar_control"]
        assert set(control) == {"fact", "baseline_value", "supported_value"}
        y, b, g = control["fact"], control["baseline_value"], control["supported_value"]
        assert type(b) is type(g) is str and b == "linen" and g not in ("linen", "stone", "unknown")
        assert y == row["fact_role_assignments"]["observation_auxiliary"] and all(q["fact"] != y for q in task["queries"])
        assert any(w["facts"][y] == b for w in task["worlds"]) and any(w["facts"][y] == g for w in task["worlds"])
        assert canonical(transform(transform(task, control), control)) == canonical(task)
        if variant == "identity":
            assert canonical(task) == canonical(prior_index[(family, context, schedule, renderer)]["semantic_task"])
        if context in ("G", "H"):
            assert row["structural_difficulty"] == 2
        empty = not prefix[-1]
        status["empty" if empty else "nonempty"] += 1
        old_marker = any(type(o["equals"]) is str and o["equals"] in ("stone", "unknown") for o in task["observations"])
        compound = old_marker and any(type(o["equals"]) is str and o["equals"] == "linen" for o in task["observations"])
        for name, predicted in [("original_marker", old_marker), ("marker_and_linen", compound)]:
            rules[name]["status_errors"] += predicted != empty
            rules[name]["train_errors" if row["split"] == "train" else "dev_errors"] += predicted != empty
        inherited = [claim for claim in row["family_control_claims"] if claim.get("name", claim.get("property")) in NAMES]
        assert len(inherited) == 2
        for claim in inherited:
            assert claim["scope"] == "corpus_variant" and claim["source_family_key"] == row["source_family_key"]
            assert claim["evidence_manifest"] == manifest_path and "pair_sample_ids" not in claim
            missing[claim.get("name", claim.get("property"))] += 1
        visible_tables.add(canonical(task["worlds"]))
        name_map = {value: role for role, value in row["fact_role_assignments"].items()}
        normalized = transform(task, control) if variant == "transposed" else task
        def decode(worlds):
            return canonical([{ "world_id": w["world_id"], "facts": {name_map[k]: v for k, v in w["facts"].items()}} for w in worlds])
        decoded_tables.add(decode(normalized["worlds"]))
        scalar_tables.add(decode(task["worlds"]))
        details.append({"sample_id": sid, "record_hash": row["record_hash"], "context": context, "variant": variant,
                        "prefix_survivor_ids": [ids(p) for p in prefix], "property_tags": sorted(tags),
                        "inherited_missing_inline_pair_claims": [claim.get("name", claim.get("property")) for claim in inherited]})
    assert (len(decoded_tables), len(scalar_tables), len(visible_tables)) == (4, 8, 24)
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["source_family_index"], row["controlled_context"], row["auxiliary_scalar_variant"], row["renderer"])].append(row)
    for group in groups.values():
        assert len(group) == 27 and {r["schedule_index"] for r in group} == set(range(27))
        hist = defaultdict(list)
        for row in group:
            for rindex, role in enumerate(("invariant_query", "variation_query", "observed_control")):
                loc = next(x for x in row["query_role_assignments"] if x["role"] == role)
                assert loc["position"] == (rindex + row["position_shift"]) % 3
                assert loc["query_id"] == row["query_id_pool"][(rindex + row["id_shift"]) % 3]
                q = row["semantic_task"]["queries"][loc["position"]]
                assert q == {"query_id": loc["query_id"], "fact": row["queried_fact_name_pool"][(rindex + row["fact_name_shift"]) % 3]}
                answer = copy.deepcopy(targets[row["sample_id"]]["queries"][loc["position"]])
                del answer["query_id"]
                hist[(q["query_id"], loc["position"], q["fact"])].append(canonical({"role": role, "answer": answer,
                    "survivors": targets[row["sample_id"]]["surviving_worlds"], "prefixes": [ids(p) for p in prefixes[row["sample_id"]]]}))
        assert len(hist) == 27 and len({canonical(sorted(values)) for values in hist.values()}) == 1
    scalar_count, field_count, cross_count = 0, 0, 0
    same_table_pairs = {}
    for family in (12, 13, 18, 19):
        for schedule in range(27):
            for renderer in ("forward", "reverse"):
                for context in "ABCDEFGH":
                    a = index[family, context, "identity", schedule, renderer]
                    b = index[family, context, "transposed", schedule, renderer]
                    assert canonical(transform(a["semantic_task"], a["auxiliary_scalar_control"])) == canonical(b["semantic_task"])
                    assert canonical(targets[a["sample_id"]]) == canonical(targets[b["sample_id"]])
                    assert [ids(p) for p in prefixes[a["sample_id"]]] == [ids(p) for p in prefixes[b["sample_id"]]]
                    scalar_count += 1
                for variant in ("identity", "transposed"):
                    a = {c: index[family, c, variant, schedule, renderer] for c in "ABCDEFGH"}
                    t = {c: a[c]["semantic_task"] for c in a}
                    assert all(canonical(t[c]["worlds"]) == canonical(t["A"]["worlds"]) and canonical(t[c]["queries"]) == canonical(t["A"]["queries"]) for c in a)
                    assert len(prefixes[a["A"]["sample_id"]][-1]) == len(prefixes[a["B"]["sample_id"]][-1])
                    assert targets[a["A"]["sample_id"]]["queries"] != targets[a["B"]["sample_id"]]["queries"]
                    assert t["C"]["observations"] == list(reversed(t["A"]["observations"]))
                    assert targets[a["A"]["sample_id"]]["surviving_worlds"] == targets[a["C"]["sample_id"]]["surviving_worlds"]
                    assert targets[a["A"]["sample_id"]]["belief_sizes"] != targets[a["C"]["sample_id"]]["belief_sizes"]
                    for short, repeat in [("A", "D"), ("E", "F"), ("G", "H")]:
                        assert t[repeat]["observations"] == t[short]["observations"] + [t[short]["observations"][0]]
                        assert targets[a[short]["sample_id"]]["queries"] == targets[a[repeat]["sample_id"]]["queries"]
                    for left, right in [("E", "G"), ("F", "H")]:
                        lt, rt = t[left], t[right]
                        assert len(lt["observations"]) == len(rt["observations"])
                        assert [i for i, (x, y) in enumerate(zip(lt["observations"], rt["observations"])) if canonical(x) != canonical(y)] == [1]
                        assert lt["observations"][1]["fact"] == rt["observations"][1]["fact"]
                        pair = (a[left]["sample_id"], a[right]["sample_id"])
                        assert ids(prefixes[pair[0]][1]) == ids(prefixes[pair[1]][1])
                        assert not prefixes[pair[0]][-1] and prefixes[pair[1]][-1]
                        same_table_pairs[pair] = len(lt["observations"])
                for left, right in [("E", "G"), ("F", "H")]:
                    for lvariant, rvariant in [("identity", "transposed"), ("transposed", "identity")]:
                        a = index[family, left, lvariant, schedule, renderer]
                        b = index[family, right, rvariant, schedule, renderer]
                        lt, rt = a["semantic_task"], b["semantic_task"]
                        assert canonical(lt["observations"]) == canonical(rt["observations"])
                        assert canonical(lt["queries"]) == canonical(rt["queries"])
                        assert canonical(transform(lt, a["auxiliary_scalar_control"])["worlds"]) == canonical(rt["worlds"])
                        assert ids(prefixes[a["sample_id"]][1]) == ids(prefixes[b["sample_id"]][1])
                        assert not prefixes[a["sample_id"]][-1] and prefixes[b["sample_id"]][-1]
                        cross_count += 1
                if renderer == "forward":
                    for context in "ABCDEFGH":
                        for variant in ("identity", "transposed"):
                            a, b = index[family, context, variant, schedule, "forward"], index[family, context, variant, schedule, "reverse"]
                            assert canonical(a["semantic_task"]) == canonical(b["semantic_task"])
                            assert a["rendered_input"] != b["rendered_input"]
                            field_count += 1
    assert (scalar_count, field_count, cross_count, len(same_table_pairs)) == (1728, 1728, 864, 864)
    entries = manifest["supplied_same_table_v2_pairs"]
    assert len(entries) == 1728
    named_pair_set = set()
    for entry in entries:
        pair = tuple(entry["pair_sample_ids"])
        assert pair in same_table_pairs and entry["name"] in NAMES
        assert (entry["name"], pair) not in named_pair_set
        named_pair_set.add((entry["name"], pair))
        assert entry["exact_candidate_sha256"] == EXPECTED_CANDIDATE
        assert entry["source_family_key"] == by_id[pair[0]]["source_family_key"]
        assert entry["auxiliary_scalar_variant"] == by_id[pair[0]]["auxiliary_scalar_variant"]
        assert entry["scope"] == "corpus_variant"
        assert entry["changed_fields"] == ["observations[1].equals"]
        assert entry["observation_length"] == same_table_pairs[pair]
        assert canonical(entry["typed_first_observation"]) == canonical(by_id[pair[0]]["semantic_task"]["observations"][0])
        assert entry["first_prefix_survivor_ids"] == ids(prefixes[pair[0]][1])
        assert entry["left_prefix_survivor_ids"] == [ids(p) for p in prefixes[pair[0]]]
        assert entry["right_prefix_survivor_ids"] == [ids(p) for p in prefixes[pair[1]]]
        assert entry["left_prefix_counts"] == [len(p) for p in prefixes[pair[0]]]
        assert entry["right_prefix_counts"] == [len(p) for p in prefixes[pair[1]]]
        assert canonical(entry["left_complete_query_payloads"]) == canonical(targets[pair[0]]["queries"])
        assert canonical(entry["right_complete_query_payloads"]) == canonical(targets[pair[1]]["queries"])
    assert len(named_pair_set) == 1728
    manifest_ids = {e["control_id"] for e in entries}
    row_claim_ids = {claim["control_id"] for row in candidates for claim in row["family_control_claims"] if claim.get("name", claim.get("property")) in NAMES}
    assert not row_claim_ids & manifest_ids
    assert all(identifier.encode() not in manifest_raw for identifier in row_claim_ids)
    executed_probe_count = 0
    for probe in manifest["executed_audit_probes"]:
        target, _ = solve(probe["probe_task"])
        assert canonical(target) == canonical(probe["producer_reference_target"])
        source = by_id[probe["source_sample_id"]]
        expected = copy.deepcopy(targets[source["sample_id"]])
        if probe["name"] == "world_id_rename_equivariance":
            before, after = source["semantic_task"]["worlds"], probe["probe_task"]["worlds"]
            assert [w["facts"] for w in before] == [w["facts"] for w in after]
            mapping = {a["world_id"]: b["world_id"] for a, b in zip(before, after)}
            expected["surviving_worlds"] = sorted(mapping[value] for value in expected["surviving_worlds"])
        else:
            assert probe["name"] == "world_order_invariance"
        assert canonical(target) == canonical(expected)
        executed_probe_count += 1
    for name in rules:
        assert dict(rules[name]) == {"status_errors": 864, "train_errors": 432, "dev_errors": 432}
    interpretations = {
        "best_cumulative_reading": "V2 specialized actual same-family pair binding remains applicable to its two named controls. The documented inline encoding is pair_sample_ids; V3 expressly preserves original/v2 fields and gives precedence for grouping only. Its new-control section does not explicitly repeal those specializations.",
        "explicit_vs_inferred": "Actual pair identities and length-specific evidence are explicit V2 requirements. Requiring the inline pair_sample_ids field is a supported encoding inference from the closed field vocabulary, not a verbatim must-contain-field sentence.",
        "strongest_alternative": "V3 line48 is later and says pair-specific evidence may include pair_sample_ids without an explicit identifier qualifier. Read broadly, it permits external aggregate bindings for inherited claims too. The actual manifest supplies every named length-specific pair and is independently reconstructable by public source family/variant/name/schedule/renderer grouping. Under that broad reading, semantics and aggregate evidence are sufficient; do not assert missing pairs.",
        "actual_manifest_assessment": "1,728 exact-candidate-bound named inherited-control entries cover 864 true pairs at both lengths. Generic inline control_id strings have no literal manifest entry or group alias; no exact control_id format was prescribed, so this mismatch is recorded as a custody limitation rather than an invented independent hard failure. Public grouping reconstructs the intended evidence.",
        "ambiguity_resolution": "A release cannot silently choose the broad waiver over the retained specialization. Hold these exact records and publish a prospective explicit version/encoding clarification; prefer a fresh metadata-corrected candidate retaining visible tasks/targets with exact inherited pair IDs and corresponding named length-specific evidence. Preserve old hashes/reviews/specifications.",
        "author_responsibility": "This actual adjudicator authored the ambiguous V3 sentence. That ambiguity is a design defect; it explains producer/reviewer behavior and cannot be attributed solely to production. This nonblind interpretation is not an independent-model result."}
    audit = {"status": "COMPLETED_INPUT_DERIVED_V3_ADJUDICATOR_AUDIT", "actual_execution_id": "/root/research_architect", "completed_at_utc": now,
        "source_bindings": bindings, "records": 3456, "exact_target_hash_prompt_and_property_pass": 3456,
        "independent_incremental_filter_direct_prefix_agreement": 3456, "property_counts": dict(properties),
        "conditional_full_typed_triple_histogram_groups": len(groups), "scalar_pairs": scalar_count, "field_pairs": field_count,
        "opposite_status_same_observation_pairs": cross_count, "inherited_same_table_pairs": len(same_table_pairs),
        "actual_manifest_named_inherited_pair_entries": len(entries), "actual_manifest_named_pair_entries_pass": len(named_pair_set),
        "same_table_pair_length_distribution": dict(Counter(str(value) for value in same_table_pairs.values())),
        "inherited_claims_missing_inline_pair_ids": dict(missing), "affected_rows": 3456,
        "distinct_generic_inherited_claim_control_ids": len(row_claim_ids), "literal_manifest_entry_id_matches": 0,
        "decoded_source_tables": len(decoded_tables), "scalar_label_tables": len(scalar_tables), "visible_renamed_tables": len(visible_tables),
        "unchanged_identity_visible_tasks_relative_to_v2": 1728, "source_families": 4, "new_source_families": 0,
        "scalar_variant_base_cases": 64, "semantic_base_contexts": 32, "distinct_parsed_tasks": len({canonical(r["semantic_task"]) for r in candidates}),
        "actual_status_counts": dict(status), "frozen_rule_errors": {name: dict(values) for name, values in rules.items()},
        "own_executed_stored_manifest_probes": executed_probe_count, "probe_corpus_records_added": 0,
        "root_completed_machine_counts": machine["report"]["counts"], "root_completed_machine_error_count": 3484,
        "root_machine_gzip_own_decode_hash_and_size_equal": True,
        "interpretation": interpretations, "per_row_measurements": details,
        "private_producer_reference_or_oracle_implementation_read": False,
        "actual_review_scope": "Programmatic recomputation of every row and actual manifest pair, public-text inspection and concrete completed reviewer evidence; not individually hand-read 3,456 rows. Nonblind adjudication after completed reviews.",
        "design_author_and_later_adjudicator_overlap": True, "same_underlying_model_correlation_risk": True}
    audit_binding = new_json("reviews/adjudication/belief-gap-v3.independent_adjudicator_audit.json", audit)
    verdict_path = OUT / "belief-gap-v3.jsonl"
    assert not verdict_path.exists()
    with verdict_path.open("w") as stream:
        for row in candidates:
            sid = row["sample_id"]
            family, variant, schedule, renderer = (row[k] for k in ["source_family_index", "auxiliary_scalar_variant", "schedule_index", "renderer"])
            left, right = ("E", "G") if len(row["semantic_task"]["observations"]) == 2 else ("F", "H")
            pair = [index[family, context, variant, schedule, renderer]["sample_id"] for context in (left, right)]
            incoming = {role: receipts[role][sid]["decision"] for role in ["expert_original", "expert_first", "expert_supplement", "red_original", "red_first"]}
            assert incoming["expert_original"] == incoming["red_original"] == incoming["expert_first"] == incoming["red_first"] == "ACCEPT"
            assert incoming["expert_supplement"] == "REVISE"
            assert receipts["red_supplement_index"][sid]["supplemental_metadata_status"] == "REVISE_METADATA_UNDER_INHERITED_V2_REQUIREMENT"
            receipt = {"receipt_id": "adj-belief-gap-v3-" + hashlib.sha256(canonical([sid, row["record_hash"], "/root/research_architect", "inherited_binding_ambiguity"]).encode()).hexdigest()[:24],
                "sample_id": sid, "record_hash": row["record_hash"], "reviewer_role": "independent_adjudicator", "reviewer_execution_id": "/root/research_architect",
                "scores": {"semantic_correctness": 5, "target_correctness": 5, "reasoning_validity": 5, "instruction_clarity": 4, "difficulty_quality": 3, "diversity": 3, "shortcut_resistance": 3, "generalization_value": 3},
                "critical_defects": ["unresolved_inherited_corpus_pair_binding"],
                "issues": [{"severity": "medium", "category": "inherited_pair_encoding_and_public_scope_ambiguity",
                    "evidence": "Both inherited corpus_variant claims omit inline pair_sample_ids. Actual manifest supplies true length-specific pairs, but broad V3 optional encoding conflicts with retained specialized V2 binding. No target/pair-truth defect.",
                    "fix": "Freeze an explicit prospective clarification and a fresh metadata-corrected candidate with exact inherited pair IDs and resolvable length-specific evidence; obtain fresh exact-hash review joins."}],
                "decision": "REVISE", "confidence": 0.94,
                "reason": "Hold the exact metadata under the best cumulative frozen contract interpretation and genuine unresolved alternative reading; original/supplemental evidence remains distinct. Correct targets do not resolve encoding ambiguity.",
                "evidence": {"candidate_sha256": EXPECTED_CANDIDATE, "own_audit_sha256": audit_binding["sha256"], "source_sha256": hashlib.sha256(code).hexdigest(),
                    "own_exact_target_property_prompt_hash_checks": "PASS", "actual_manifest_pair_truth": "PASS", "actual_length_specific_pair_sample_ids": pair,
                    "pair_record_hashes": [by_id[value]["record_hash"] for value in pair], "inline_pair_ids_absent_for": list(NAMES),
                    "original_and_supplemental_decisions": incoming, "red_supplemental_metadata_status": receipts["red_supplement_index"][sid]["supplemental_metadata_status"],
                    "original_expert_red_disagreement": False, "current_supplement_expert_red_disagreement": False,
                    "original_to_supplement_recommendation_change": True, "best_encoding_reading_confidence": 0.8,
                    "actual_absence_and_manifest_pair_factuality_confidence": 1.0, "spec_design_author_adjudication_overlap": True}}
            stream.write(canonical(receipt) + "\n")
    verdict_binding = {"path": str(verdict_path.relative_to(BASE)), "sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest(), "bytes": verdict_path.stat().st_size}
    ticket = {"ticket_id": "RBM-INTERFACE-012", "status": "OPEN_FOR_FRESH_METADATA_REVISION", "actual_author_execution_id": "/root/research_architect", "created_at_utc": now,
        "candidate_sha256": EXPECTED_CANDIDATE, "affected_records": 3456, "affected_claims": 6912,
        "semantic_targets_or_pair_truth_defects": 0, "actual_manifest_pairs_pass": 864, "actual_manifest_named_pair_entries_pass": 1728,
        "public_interpretation": interpretations, "evidence_bindings": [bindings[manifest_path], audit_binding, verdict_binding],
        "required_fix": "Publish NEW prospective clarification naming the two inherited controls and their inline/manifest encoding. Fresh metadata revision must carry exact same-family/variant/schedule/style E/G or F/H pairs with length-specific evidence, preserve visible tasks/targets if unchanged, recompute record hashes and obtain completed new expert/red/adjudication receipts. Do not edit V3 candidate/spec/first reviews or silently relax its gate.",
        "recommended_unambiguous_encoding": "Both inherited corpus_variant claim objects contain pair_sample_ids of two actual distinct same-family supplied IDs. Use a length-specific manifest entry whose exact control_id resolves with the same name, pair IDs and row hashes. The same pair/control ID may support both identifiers; no globally unique-ID-per-property assumption. A nonmember context row may carry a family claim referring to its matched E/G or F/H evidence; it is not a per-row truth tag.",
        "residual_quality_limits": "Four reused families, 32 contexts, 64 scalar cases and correlated 27x2 nuisance views;75% nonempty/87.5% query-usefulness-false majority; fitted finite table/frequency/length associations persist. Same-model/guided correction and design/adjudication overlap; no complete shortcut elimination or efficacy.",
        "data_or_release_approval": False}
    ticket_binding = new_json("feedback/improvement_requests/RBM-INTERFACE-012.json", ticket)
    report_path = OUT / "ADJUDICATION_BELIEF_GAP_V3.md"
    assert not report_path.exists()
    report_path.write_text(f"""# Belief-gap-v3 exact adjudication

Actual adjudicator `/root/research_architect`. Candidate SHA256 `{EXPECTED_CANDIDATE}`. **All3,456 REVISE**, release eligibility zero. Targets, row properties and actual semantic pairs are correct; the hold concerns inherited evidence encoding and genuine public-contract ambiguity.

Both original completed reviewers accepted all3,456, with zero original disagreement. Expert first/final hashes are identical; red first/final were separately preserved. Their separately completed guided supplements recommend revision for all3,456 and6,912 inherited claims. These new recommendations do not rewrite original ACCEPT receipts. All original, supplemental and oracle record hashes join exactly to this candidate. Adjudication preserves the recommendation change and both interpretations.

## Independent manifest finding

Unlike the initial reviewers, this adjudicator inspected `data/RBM-001/manifests/belief-gap-v3.family_controls.json` SHA `{bindings[manifest_path]['sha256']}`. Its `supplied_same_table_v2_pairs` contains1,728 named entries:864 actual E/G or F/H pairs multiplied by the two inherited identifiers. Every entry binds this exact candidate, actual pair IDs and observation length, true same-table/query/first-prefix controls and independently recomputed prefix/query evidence. All432 length-two and432 length-three pairs pass. It would be false to claim these pairs are absent or that the manifest is merely an empty pointer.

Each row nevertheless omits `pair_sample_ids` in both inherited claim objects. Their16 generic family/variant/identifier control_id strings have no literal manifest entry/group alias. Full relationships are reconstructable from public family/variant/name/schedule/renderer grouping. No exact control-ID format was prescribed, so the mismatch is a disclosed custody limitation, not an invented additional hard requirement.

## Frozen text and the genuine alternative reading

V2 `BELIEF_CONTROL_PROPERTIES_REVISION_V2.md:42` specifically requires supplied E/G and F/H claims with actual same-family pair IDs and length-specific evidence. The original closed field vocabulary documents `pair_sample_ids` as the inline container. Requiring that field is a supported encoding inference; the public sentence does not literally say must contain that exact field name.

V3 `BELIEF_CONTROL_PROPERTIES_REVISION_V3.md:3` gives precedence for grouping only; line52 retains original/v2 canonical fields; line59 describes three new identifiers. Line48, following the new-control table, says pair-specific evidence may include pair_sample_ids. The best cumulative reading limits that optional encoding to the new controls and retains the inherited specialization. But the sentence lacks an explicit identifier qualifier. The strongest broad later-amendment reading permits aggregates for every family claim; under that reading the actual rich manifest establishes the semantic witnesses. This is genuine ambiguity, not irrational production behavior.

I authored that ambiguous V3 wording. The design defect must be disclosed along with the missing inline encoding; it cannot be blamed solely on producers/reviewers. I will not silently waive the inherited specialization to approve this frozen candidate. A NEW prospective clarification and fresh metadata candidate with exact inherited IDs resolves both readings without changing filtering semantics or rewriting history. See `feedback/improvement_requests/RBM-INTERFACE-012.json`. Interpretation confidence0.8 is separate from exact missing-field/pair-truth confidence1.0.

## Actual exact checks and quality boundary

Own public-derived typed filtering and separate direct-prefix conjunction agree on all3,456 full targets. Hash/prompt read-back and every attached property/locator pass. All128 conditional full typed nuisance-triple histogram groups,1,728 field-order pairs,1,728 fact-local scalar/involution pairs and864 opposite-status same-observation pairs pass. All1,728 identity tasks remain unchanged from V2; decoded counts are four source tables/eight scalar tables/24 visible renamed tables. Sixteen stored manifest probe tasks were independently executed by this adjudicator; they add no corpus rows/families. This is programmatic every-row inspection plus public-text/manifest/evidence assessment, not a claim all rows were manually read or input predictions were frozen before gold in this adjudication.

Both frozen marker rules now make864 status errors,432 per split; the75% status fit and864 matched-feature lower bound are established. The scope is still four reused families,32 contexts,64 scalar cases and1,728 parsed tasks behind3,456 correlated rows; zero new families. Nonempty status majority is75%; usefulness-false majority is87.5%. Expert's actual fitted column-frequency/observation lookup100% in-sample and length/scalar lookup96.875% are residual corpus associations; both expose75% dev fit with zero seen dev buckets. Those are reviewer diagnostics, not this adjudicator's trained model experiment. Actual expert counterexamples preserve restricted column/observation features while changing table correspondence and status. No broad shortcut immunity, Bayesian reasoning, authority, chosen inspection utility, sealed OOD, architecture or efficacy result follows.

Root's completed ff1 checker report SHA `{bindings['audits/diversity/belief-gap-v3-belief-controls.json']['sha256']}` has zero target errors but3,456 metadata failures and3,484 aggregate control errors. This adjudicator independently decoded its existing gzip exactly; no large raw copy was added. The formal public-only interpretation, exact source hashes, completed first/final/supplemental executions and actual root runner remain separately attributed. No producer/reference/oracle/reviewer implementations were imported; only this adjudicator's own frozen public-derived helper was reused. Earlier reviewer failures/interruption histories remain preserved and bound through their completed receipts.

This actual agent designed V1/V2/V3 and now adjudicates after completed initial reviews. Nonblind common guidance and the shared underlying model can produce correlated errors. This execution edits no candidate, original review, contract, generator or release, and grants no partial release approval. RGM prospective design remains a separate frozen task.
""")
    report_binding = {"path": str(report_path.relative_to(BASE)), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(), "bytes": report_path.stat().st_size}
    for name, binding in bindings.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == binding["sha256"], ("source changed during adjudication", name)
    result = {"actual_execution_id": "/root/research_architect", "status": "COMPLETED_FROZEN_V3_ADJUDICATION", "completed_at_utc": now,
        "decision_counts": {"REVISE": 3456}, "exact_hash_joins_per_evidence_stream": 3456,
        "original_expert_red_disagreements": 0, "supplemental_expert_red_disagreements": 0,
        "original_to_current_recommendation_changes": 3456, "source_bindings": bindings,
        "output_bindings": [audit_binding, verdict_binding, ticket_binding, report_binding],
        "actual_source_sha256": hashlib.sha256(code).hexdigest(), "private_implementation_imports": [],
        "own_public_derived_helper": "reviews/adjudication/audit_belief_gap_v1.py", "data_or_release_edits": False,
        "release_eligible_records": 0, "design_author_adjudicator_overlap": True, "common_model_and_guided_exposure": True,
        "independent_public_counterread_delegation": "Attempted separate public-only text counterread; tool rejected because four-agent thread limit reached. No additional execution is claimed.",
        "new_raw_files_exceeding25MiB": []}
    result_binding = new_json("reviews/adjudication/execution-belief-gap-v3.json", result)
    print(json.dumps({"decision_counts": result["decision_counts"], "outputs": result["output_bindings"] + [result_binding]}, indent=2))

if __name__ == "__main__":
    main()
