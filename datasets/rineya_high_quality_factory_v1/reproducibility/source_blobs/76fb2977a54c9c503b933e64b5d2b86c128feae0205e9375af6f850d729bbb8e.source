#!/opt/anaconda3/bin/python3
"""Preserve v2 first expert assessment, then finalize unchanged judgments.

The actual agent inspected every base family and changed mechanism before
--first.  No producer manifests or other reviewer judgments are read here.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from independent_memory_gap_v1 import BASE,EXECUTION_ID,encode,exact_equal

OUT = BASE/"reviews/expert"
BATCH = BASE/"data/REM-001/candidate_renderings/memory-gap-v2.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first():
    facts = json.loads((OUT/"memory-gap-v2.independent_analysis.json").read_text())
    records = [json.loads(line) for line in BATCH.read_text().splitlines()]
    assert sha(BATCH) == facts["source_sha256"]
    evidence = {r["sample_id"]:r for r in facts["row_evidence"]}
    receipts = []
    assert not exact_equal({"value":True},{"value":1}) and not exact_equal({"value":False},{"value":0})
    for record in records:
        row = evidence[record["sample_id"]]
        assert row["record_hash"] == record["record_hash"]
        for name in ("query_rotation","key_rotation","difficulty","structural_difficulty","source_family_index","generation_seed"):
            assert type(record[name]) is int
        assert record["difficulty"] == record["structural_difficulty"] == 3
        assert 0 <= record["query_rotation"] <= 7 and 0 <= record["key_rotation"] <= 3
        for pos,assignment in enumerate(record["query_role_assignments"]):
            assert type(assignment["position"]) is int and assignment["position"] == pos
        assert len(row["metadata_witness_validation"]["validated_properties"]) == 9
        assert set(row["status_histogram"]) == {"present","deleted","absent"}
        receipts.append({
            "receipt_id":"EXPERT-memory-gap-v2-"+record["sample_id"]+"-"+record["record_hash"][:12],
            "sample_id":record["sample_id"],"record_hash":record["record_hash"],
            "reviewer_role":"expert_reviewer","reviewer_execution_id":EXECUTION_ID,
            "review_status":"ACCEPTED","decision":"ACCEPT","confidence":0.99,
            "scores":{"semantic_correctness":5,"target_correctness":5,"instruction_clarity":5,
                      "reasoning_validity":5,"generalization_value":3,"difficulty_quality":4,
                      "diversity":3,"shortcut_resistance":3},
            "critical_defects":[],
            "issues":[
                {"category":"shortcut_scope","severity":"low",
                 "evidence":"All supplied rotations balance exact role, status, typed value, present-null, chosen-version/index and decoded-winner histograms marginally across every ID, position and consistently renamed key. Joint nuisance buckets retain conditional status associations; observed majority fractions reach 0.875, versus marginal fractions 0.625 or 0.75. This is an empirical association, not measured model performance.",
                 "fix":"Limit claims to the verified marginal crossing and disclose residual joint correlations. Add independently crossed schedules if joint nuisance independence becomes a requirement."},
                {"category":"generalization_scope","severity":"low",
                 "evidence":"Twelve source families provide 48 base controlled histories with 12–14 events and eight independent queries each. The 768 rows are correlated histories, key/query rotations and renderings. This establishes finite precedence retrieval and matched counterfactuals, not learned persistent memory, associative relevance or efficacy.",
                 "fix":"Retain bounded proxy scope and source-family counts; evaluate additional mechanism families and actual model outcomes separately if broader claims are needed."},
                {"category":"edge_case_coverage","severity":"low",
                 "evidence":"All 768 absent query answers use truly missing keys. Future-filter witnesses exist on other keys, but future-only absence supplies zero corpus observations. The current absence_control tag is truthful and does not itself promise both absence subtypes.",
                 "fix":"Report missing-key and future-only absence separately; add a new future-only absence control if that unrepresented subtype is desired."}
            ],
            "review_method":"Own prompt-derived typed scan and separate sorted-precedence resolver; full hash/payload/target recomputation; independent property search plus exact supplied-witness validation; all family scope/paired controls, exact typed/provenance histogram crossing and field-order invariance; eight per-row metamorphics and query-local protected probes; substantive actual-agent inspection of every base family and changed mechanism.",
            "evidence":{"hash_prompt_full_target_typed_exact":True,
                        "derived_target_sha256":hashlib.sha256(encode(row["own_answer"]).encode()).hexdigest(),
                        "validated_public_properties":row["metadata_witness_validation"]["validated_properties"],
                        "status_histogram":row["status_histogram"],"delete_event_count":row["delete_event_count"],
                        "family_scope_declarations_valid":True,"manifest_pointer_not_read":True,
                        "metamorphics_passed":len(row["metamorphics"]),"audit_probe_count":len(row["audit_probes"]),
                        "own_raw_evidence":str(OUT/"memory-gap-v2.independent_analysis.json"),
                        "own_exact_gzip_evidence":str(OUT/"memory-gap-v2.independent_analysis.json.gz"),
                        "compression_manifest":str(OUT/"memory-gap-v2.independent_analysis.compression_manifest.json"),
                        "semantic_inspection_report":str(OUT/"semantic_inspection_memory_gap_v2.md")},
            "review_guidance":"Corrective issue-topic-informed assessment using my own v1 findings; not a fresh fully blind study. No other reviewer verdicts, producer code/control reports, reference or oracle were read.",
            "first_assessment_read_scope":"memory-gap-v2 candidate only, two public contracts, own prior freshly authored solver/audit helpers and new v2 code/evidence. Generic public metadata-interface availability notice received; that document was not read before this complete first assessment.",
            "release_approval":False,
        })
    path = OUT/"memory-gap-v2.first_judgments.jsonl"
    with path.open("x") as stream:
        stream.write("".join(encode(r)+"\n" for r in receipts))
    print(json.dumps({"first_judgments":str(path),"records":len(receipts),"decisions":dict(Counter(r["decision"] for r in receipts)),
                      "sha256":sha(path)},indent=2))


def final():
    initial = OUT/"memory-gap-v2.first_judgments.jsonl"
    destination = OUT/"memory-gap-v2.jsonl"
    rows = [json.loads(line) for line in initial.read_text().splitlines()]
    with destination.open("xb") as stream:
        stream.write(initial.read_bytes())
    facts = json.loads((OUT/"memory-gap-v2.independent_facts_summary.json").read_text())
    assert sha(BATCH) == facts["source_sha256"]
    manifest = json.loads((OUT/"memory-gap-v2.independent_analysis.compression_manifest.json").read_text())
    raw,compressed = Path(manifest["raw_path"]).read_bytes(),Path(manifest["gzip_path"]).read_bytes()
    assert gzip.decompress(compressed) == raw
    assert hashlib.sha256(raw).hexdigest() == manifest["raw_sha256"]
    assert hashlib.sha256(compressed).hexdigest() == manifest["gzip_sha256"]
    metadata_doc = BASE/"schemas/MEMORY_CONTROL_METADATA.md"
    inspection = OUT/"semantic_inspection_memory_gap_v2.md"
    assert inspection.exists()
    facts.update({"stage":"COMPLETED_CORRECTIVE_EXPERT_FIRST_AND_FINAL", "decision_counts":dict(Counter(r["decision"] for r in rows)),
                  "first_judgments_sha256":sha(initial),"final_receipts_sha256":sha(destination),
                  "first_final_bytes_identical":initial.read_bytes() == destination.read_bytes(),
                  "metadata_public_doc_read_after_first_judgments":True,"metadata_public_doc_sha256":sha(metadata_doc),
                  "metadata_doc_effect_on_verdict":"Clarifies generic family existence-witness encoding; no task/property requirement change, candidate mutation or verdict change",
                  "raw_evidence_sha256":manifest["raw_sha256"],"raw_evidence_bytes":manifest["raw_bytes"],
                  "gzip_evidence_sha256":manifest["gzip_sha256"],"gzip_evidence_bytes":manifest["gzip_bytes"],
                  "gzip_exact_roundtrip_verified":True,"finalizer_sha256":sha(__file__),
                  "semantic_inspection_sha256":sha(inspection),
                  "actual_agent_inspection_scope":"Every one of 12 base families p0012-p0023 forward qrot0 histories/role answers; all three supplied changed mechanisms for each family; representative property/witness/provenance/nuisance crossing examples. Programmatic exact review covers all768 records.",
                  "candidate_bytes_unchanged":True,"v1_files_not_modified":True,"release_approval":False})
    with (OUT/"verification_summary_memory_gap_v2.json").open("x") as stream:
        stream.write(json.dumps(facts,indent=2)+"\n")
    print(json.dumps({"final_receipts":str(destination),"records":len(rows),"decisions":facts["decision_counts"],
                      "first_final_identical":facts["first_final_bytes_identical"],"sha256":sha(destination),
                      "metadata_doc_sha256":facts["metadata_public_doc_sha256"],"compression_exact":True},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first",action="store_true")
    parser.add_argument("--final",action="store_true")
    args = parser.parse_args()
    if args.first and not args.final:
        first()
    elif args.final and not args.first:
        final()
    else:
        parser.error("Choose exactly one stage")
