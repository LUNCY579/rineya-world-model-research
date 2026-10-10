#!/opt/anaconda3/bin/python3
"""Own V4 final receipt custody after actual independent corrective assessment.

Freeze this exact source and all previously completed dependencies before
execution. No solver, external review, gate, producer or formal code is imported.
This writes new final aliases/compact reports only, never any historical file.
"""
import collections
import datetime
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/expert"
ROLE = "/root/expert_reviewer"
PREFIX = "belief-gap-v4-canonical-finalize-attempt01"
BIND = {
    "reviews/expert/belief-gap-v4-canonical-attempt03.execution.json": "0da3142a23c77c156cc24adeff973052e7c0c22a32cc287474cf037e50e9645a",
    "reviews/expert/belief-gap-v4-canonical-attempt03.analysis.json": "098ac542d42fd38bd89826303e81f45bddc99d3c265e93cc27f07e283fec5d59",
    "reviews/expert/belief-gap-v4.canonical-corrective-attempt03.jsonl": "155f65a1766addc6322b55ff51c6c5bf71f9c32cd8f536935ca479eeccc99d4a",
    "reviews/expert/belief-gap-v4-canonical-review-attempt03.py": "c23633c0a7a2e528543f5f93121c9362b28b847e3acacacecabd22208022f6cc",
    "reviews/expert/belief-gap-v4-canonical-attempt03.source.preexecution.py": "c23633c0a7a2e528543f5f93121c9362b28b847e3acacacecabd22208022f6cc",
    "reviews/expert/belief-gap-v4-canonical-attempt03.preexecution.json": "665560876b1957006d651a27d7645c0081fc54d26ba8b8262a3db715e00f4719",
    "reviews/expert/belief-gap-v4-canonical-inspection-attempt01.evidence.json": "4da6f65fdf6e7ce8a060abd2e1c0332f1ea4f90d4aa0ba7714db6a926005916b",
    "reviews/expert/belief-gap-v4-canonical-inspection-attempt01.execution.json": "2a7e49fe047c5dbe90751597dec894b7d1a929d675410ad5742367840da12984",
    "reviews/expert/belief-gap-v4-canonical-inspection-attempt01.py": "cbacc5aa409db14362dbd457864bd1a09fcf45715efd8744834382cbb80a4d50",
    "reviews/expert/belief-gap-v4-canonical-inspection-attempt01.source.preexecution.py": "cbacc5aa409db14362dbd457864bd1a09fcf45715efd8744834382cbb80a4d50",
    "reviews/expert/belief-gap-v4.canonical-semantic-inspection.md": "595709da153f2168b008eb63c4978a549316121ad9f650b00772885c7296374c",
}
SCORES = {"semantic_correctness", "target_correctness", "instruction_clarity",
          "reasoning_validity", "generalization_value", "difficulty_quality",
          "diversity", "shortcut_resistance"}


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def load(text):
    def object_pairs(items):
        output = {}
        for key, value in items:
            assert key not in output, "duplicate JSON key"
            output[key] = value
        return output
    return json.loads(text, object_pairs_hook=object_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def hp(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "length_bytes": size}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canon(value) + "\n")
    return hp(path)


def bindings():
    current = {path: hp(ROOT / path) for path in BIND}
    assert all(current[path]["sha256"] == sha for path, sha in BIND.items())
    previous = load((OUT / "belief-gap-v4-canonical-attempt03.execution.json").read_text())
    assert previous["bindings_before"] == previous["bindings_after"]
    for path, record in previous["bindings_before"].items():
        current[path] = hp(ROOT / path)
        assert current[path] == record, ("changed prior binding", path)
    return current


def freeze():
    source = pathlib.Path(__file__).resolve()
    archive = OUT / (PREFIX + ".source.preexecution.py")
    with archive.open("xb") as stream:
        stream.write(source.read_bytes())
    pre = {
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "created_utc": now(),
        "source": hp(source), "archive": hp(archive), "bindings_before": bindings(),
        "python_binary": hp(pathlib.Path(sys.executable).resolve()),
        "scope": "Finalize already actual own corrective judgments and agent inspection; no additional private evidence, model score or publication approval",
        "preserve": "Original failed attempt, first canonical REVISE receipts, prior input-first predictions/judgments and all frozen evidence remain unchanged",
        "output_layout": "Every new individual raw output below 25 MiB; lossless compression not needed; no new oversized evidence",
    }
    path = OUT / (PREFIX + ".preexecution.json")
    print(canon({"source": hp(source), "preexecution": save(path, pre)}))


def readrows(path):
    output = {}
    for line in path.open(encoding="utf-8"):
        row = load(line)
        assert row["sample_id"] not in output
        output[row["sample_id"]] = row
    return output


def run():
    started = now()
    timer = time.monotonic()
    source = pathlib.Path(__file__).resolve()
    prepath = OUT / (PREFIX + ".preexecution.json")
    pre = load(prepath.read_text())
    before = bindings()
    assert before == pre["bindings_before"]
    assert hp(source) == pre["source"] == hp(OUT / (PREFIX + ".source.preexecution.py"))
    assert hp(pathlib.Path(sys.executable).resolve()) == pre["python_binary"]
    execution = load((OUT / "belief-gap-v4-canonical-attempt03.execution.json").read_text())
    analysis = load((OUT / "belief-gap-v4-canonical-attempt03.analysis.json").read_text())
    inspection = load((OUT / "belief-gap-v4-canonical-inspection-attempt01.evidence.json").read_text())
    assert execution["exit_code"] == 0 and execution["row_error_count"] == execution["batch_error_count"] == 0
    assert not analysis["row_errors"] and not analysis["batch_errors"]
    assert len(analysis["decoder_component_tests"]) == 49
    assert all(test["passed"] is True for test in analysis["decoder_component_tests"])
    assert len(inspection["exact_actual_inspected_ids"]) == 32
    first = readrows(OUT / "belief-gap-v4.canonical-first-judgments.jsonl")
    current = readrows(OUT / "belief-gap-v4.canonical-corrective-attempt03.jsonl")
    predictions = readrows(OUT / "belief-gap-v4-input-attempt01.predictions.jsonl")
    assert len(first) == len(current) == len(predictions) == 3456
    assert set(first) == set(current) == set(predictions)
    initial_counts = collections.Counter()
    final_counts = collections.Counter()
    receipt_ids = set()
    ordered_bindings = []
    for sample_id, row in current.items():
        initial, prediction = first[sample_id], predictions[sample_id]
        assert row["sample_id"] == initial["sample_id"] == prediction["sample_id"]
        assert row["record_hash"] == initial["record_hash"] == prediction["record_hash"]
        for receipt in (initial, row):
            assert receipt["reviewer_execution_id"] == ROLE and receipt["reviewer_role"] == "expert"
        assert initial["decision"] == "REVISE"
        assert initial["critical_defects"] == ["conditional typed histogram/control assertion failed"]
        assert row["decision"] == "ACCEPT" and not row["critical_defects"]
        assert set(row["scores"]) == SCORES and all(type(score) is int and 1 <= score <= 5 for score in row["scores"].values())
        assert row["scores"]["generalization_value"] == row["scores"]["diversity"] == row["scores"]["shortcut_resistance"] == 2
        assert row["receipt_id"] not in receipt_ids
        receipt_ids.add(row["receipt_id"])
        assert row["receipt_id"] != initial["receipt_id"]
        evidence = row["evidence"]
        assert evidence["full_record_hash_recomputed"] is True and evidence["whole_typed_target_matches"] is True
        assert evidence["canonical_read_after_input_FINAL"] is True
        assert len(evidence["owner_scoped_inherited_claims"]) == 2
        assert evidence["own_complete_target_sha256"] == hashlib.sha256(canon(prediction["expected_target"]).encode()).hexdigest()
        assert evidence["own_prefix_survivors_sha256"] == hashlib.sha256(canon(prediction["prefix_survivors"]).encode()).hexdigest()
        initial_counts[initial["decision"]] += 1
        final_counts[row["decision"]] += 1
        ordered_bindings.append([sample_id, row["record_hash"], row["receipt_id"]])
    assert dict(final_counts) == execution["decisions"] == {"ACCEPT": 3456}
    src = OUT / "belief-gap-v4.canonical-corrective-attempt03.jsonl"
    final = OUT / "belief-gap-v4.jsonl"
    with src.open("rb") as left, final.open("xb") as right:
        for block in iter(lambda: left.read(1048576), b""):
            right.write(block)
    assert hp(final) == hp(src)
    assert final.read_bytes() == src.read_bytes()
    after = bindings()
    assert before == after and hp(source) == pre["source"]
    summary = {
        "report_type": "ACTUAL_EXPERT_BELIEF_V4_FINAL_CANONICAL_SUMMARY",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "created_utc": now(),
        "candidate": before["data/RBM-001/candidate_renderings/belief-gap-v4.jsonl"],
        "manifest": before["data/RBM-001/manifests/belief-gap-v4.family_controls.json"],
        "final_receipts_path": str(final.relative_to(ROOT)), "final_receipts": hp(final),
        "final_decisions": dict(final_counts), "original_first_decisions": dict(initial_counts),
        "original_first_receipts": before["reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl"],
        "first_correction": {
            "cause": "Own histogram decoder treated textual diagnostic position labels as typed task scalars; actual supplied labels are strings 0/1/2 and independent histogram labels integers",
            "disposition": "Withdraw first histogram defect conclusion after own source-frozen corrective execution; original first/failure evidence preserved unchanged",
            "no_candidate_or_target_change": True, "no_external_reviewer_feedback_used": True,
            "normalization_boundary": "Only textual diagnostic position coordinates; preserve typed full outcomes, bins, counts, prefix IDs, null/false/zero separation; reject malformed coordinates and decoded collisions",
        },
        "input_freeze_preceded_canonical_access": True,
        "pregold_predictions": before["reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl"],
        "pregold_first_input_judgments": before["reviews/expert/belief-gap-v4-input-attempt01.first-input-judgments.jsonl"],
        "input_FINAL": before["reviews/expert/belief-gap-v4-input-attempt01.FINAL.json"],
        "actual_counts": analysis["counts"],
        "preserved_field_counts": analysis["preservation_counts"],
        "changed_top_level_field_counts": analysis["actual_changed_top_level_fields_vs_v3"],
        "added_semantic_families": 0, "added_semantic_tasks": 0, "added_student_inputs": 0,
        "actual_named_entry_usage_histogram": analysis["manifest_named_entry_usage_histogram"],
        "actual_property_counts": analysis["actual_attached_property_tag_counts"],
        "actual_scalar_rule_results": analysis["frozen_public_rule_actual_counts"],
        "actual_fitted_component_status_lookups": analysis["fitted_input_status_lookups"],
        "quality": {"remaining_candidate_critical_defects": [], "moderate_limitations": [
            "Same four pre-exposed V3 families;32 contexts,64 scalar cases and nuisance rotations;zero mechanism/family growth",
            "75% nonempty and87.5% usefulness-false label majorities;simple G/H one effective filter",
            "Finite fitted full column-frequency plus observation status map100%;scalar plus length96.875%;no model/full-target or out-of-family score implied",
            "Two exposed dev families;train fitted lookup often abstains entirely on dev;coverage counts retained",
            "Same underlying model,public design overlap,own solver reuse and prior V3 gold exposure;not unseen/model-independent"]},
        "actual_semantic_inspection": before["reviews/expert/belief-gap-v4.canonical-semantic-inspection.md"],
        "actual_semantic_inspection_path": "reviews/expert/belief-gap-v4.canonical-semantic-inspection.md",
        "compact_inspection_evidence": before["reviews/expert/belief-gap-v4-canonical-inspection-attempt01.evidence.json"],
        "compact_inspection_evidence_path": "reviews/expert/belief-gap-v4-canonical-inspection-attempt01.evidence.json",
        "actual_corrective_source": before["reviews/expert/belief-gap-v4-canonical-review-attempt03.py"],
        "actual_corrective_execution": before["reviews/expert/belief-gap-v4-canonical-attempt03.execution.json"],
        "actual_corrective_analysis": before["reviews/expert/belief-gap-v4-canonical-attempt03.analysis.json"],
        "ordered_exact_sample_hash_receipt_join_sha256": hashlib.sha256(canon(ordered_bindings).encode()).hexdigest(),
        "exposure": analysis["exposure"], "no_other_review_results_read": True,
        "new_raw_output_layout": "Every new file under25MiB;no oversized duplicated raw;historical files untouched",
        "publication_approval": False, "no_release_authority": True,
    }
    summary_path = OUT / "belief-gap-v4.canonical-final-summary.json"
    save(summary_path, summary)
    final_execution = {
        "report_type": "ACTUAL_EXPERT_V4_CANONICAL_FINALIZATION_EXECUTION",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "start_utc": started,
        "finish_utc": now(), "elapsed_seconds": round(time.monotonic() - timer, 6),
        "source_before_after": hp(source), "preexecution": hp(prepath),
        "bindings_before": before, "bindings_after": after,
        "actual_final_receipts": hp(final), "actual_corrective_receipts": hp(src),
        "exact_final_byte_equality_to_completed_corrective_receipts": True,
        "rows_joined": len(ordered_bindings), "unique_receipt_ids": len(receipt_ids),
        "original_first_REVISE_cause_verified_all_rows": True,
        "actual_first_files_unchanged": True, "final_decisions": dict(final_counts),
        "summary": hp(summary_path), "exit_code": 0, "publication_approval": False,
    }
    execution_path = OUT / (PREFIX + ".execution.json")
    save(execution_path, final_execution)
    final_manifest_path = OUT / "belief-gap-v4.canonical-FINAL.json"
    final_manifest = {
        "report_type": "ACTUAL_EXPERT_V4_CANONICAL_COMPLETED_FINAL",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "completed_utc": now(),
        "status": "COMPLETED_EXPERT_CANONICAL_REVIEW", "final_decisions": dict(final_counts),
        "candidate": summary["candidate"], "manifest": summary["manifest"],
        "source": hp(source), "source_archive": hp(OUT / (PREFIX + ".source.preexecution.py")),
        "preexecution": hp(prepath), "finalization_execution": hp(execution_path),
        "files": {str(p.relative_to(ROOT)): hp(p) for p in [final, summary_path,
            OUT / "belief-gap-v4.canonical-semantic-inspection.md",
            OUT / "belief-gap-v4-canonical-attempt03.execution.json",
            OUT / "belief-gap-v4-canonical-attempt03.analysis.json",
            OUT / "belief-gap-v4.canonical-first-judgments.jsonl",
            OUT / "belief-gap-v4.canonical-corrective-attempt03.jsonl",
            OUT / "belief-gap-v4.canonical-histogram-inspect.execution.json",
            OUT / "belief-gap-v4-canonical-attempt01.failure.json",
            OUT / "belief-gap-v4-canonical-inspection-attempt01.evidence.json",
            OUT / "belief-gap-v4-canonical-inspection-attempt01.execution.json"]},
        "all_listed_new_outputs_below_25MiB": True, "first_evidence_immutable": True,
        "exposure": analysis["exposure"], "publication_approval": False,
        "no_dataset_release_or_adjudication_authority": True,
    }
    assert all(record["length_bytes"] <= 25 * 1024 * 1024 for record in final_manifest["files"].values())
    save(final_manifest_path, final_manifest)
    print(canon({"FINAL": hp(final_manifest_path), "summary": hp(summary_path),
                 "execution": hp(execution_path), "final_receipts": hp(final),
                 "decisions": dict(final_counts), "original_first_receipts_unchanged": True}))


if __name__ == "__main__":
    {"freeze": freeze, "run": run}[sys.argv[1]]()
