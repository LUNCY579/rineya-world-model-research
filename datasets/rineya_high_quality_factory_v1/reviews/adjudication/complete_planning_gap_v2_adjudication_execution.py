"""Complete missing execution receipt after owned finalization KeyError; no verdict edits."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "/root/research_architect"
BASE = "reviews/adjudication/planning-gap-v2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    raw = (ROOT/path).read_bytes()
    return {"path":path,"sha256":sha(raw),"bytes":len(raw)}


def read(path):
    return json.loads((ROOT/path).read_text())


def complete():
    old_source = binding("reviews/adjudication/audit_and_adjudicate_planning_gap_v2.py")
    assert old_source["sha256"] == "da2b2af07a419c01551d37dcc90515645e6655cdf83d57c2327bc6d04ad45a8f"
    audit_binding = binding(BASE+".audit.json")
    assert audit_binding["sha256"] == "1132df81bb6e4c884fecaf53136b2dea7a31595b55550e4420a4e3bb2b42677e"
    audit = read(BASE+".audit.json")
    for path, evidence in audit["bindings"].items():
        assert binding(path) == evidence, ("frozen audit evidence changed",path)
    candidate_path = "data/RGM-001/candidate_renderings/planning-gap-v2.jsonl"
    manifest_path = "data/RGM-001/manifests/planning-gap-v2.family_controls.json"
    candidate = [json.loads(line) for line in (ROOT/candidate_path).read_text().splitlines()]
    manifest = read(manifest_path)
    red_path = "audits/execution/planning-gap-v2-adversarial.json"
    assert binding(red_path)["sha256"] == "ecf3795076f8ecb53493ea5b09d3fdff2aea2040069e50af5cf81690c7262ca3"
    red_execution = read(red_path)
    assert red_execution["completed"] is True and red_execution["actual_execution_id"] == "/root/adversarial_auditor" and red_execution["rows"] == 408
    originals = {r["sample_id"]:r for r in candidate}
    verdicts = [json.loads(line) for line in (ROOT/(BASE+".jsonl")).read_text().splitlines()]
    assert len(verdicts) == len(originals) == 408
    assert len({v["receipt_id"] for v in verdicts}) == 408 and {v["sample_id"] for v in verdicts} == set(originals)
    for verdict in verdicts:
        assert verdict["record_hash"] == originals[verdict["sample_id"]]["record_hash"]
        assert verdict["reviewer_execution_id"] == IDENTITY and verdict["reviewer_role"] == "independent_adjudicator"
        assert verdict["decision"] == "QUARANTINE" and verdict["evidence"]["publication_eligible"] is False
        assert verdict["evidence"]["own_audit"] == audit_binding
        assert verdict["evidence"]["adjudicator_source_sha256"] == old_source["sha256"]
        assert verdict["evidence"]["final_red_custody"] == binding(red_path)
    ticket_path = "feedback/improvement_requests/RGM-RENDERER-008.json"
    ticket = read(ticket_path)
    assert ticket["decision_counts"] == {"QUARANTINE":408} and ticket["verdicts"] == binding(BASE+".jsonl")
    assert ticket["candidate"] == binding(candidate_path) and ticket["manifest"] == binding(manifest_path)
    failure_path = BASE+".owned-failure-dd52f689928fcec3.json"
    failure = read(failure_path)
    assert failure["source_sha256"] == old_source["sha256"] and "KeyError" in failure["exception"]
    assert "family_controls.json" in failure["exception"]
    # Additional root-input custody read is authorized; no producer implementation/fixtures.
    input_freeze_path = "reviews/input_only/planning-gap-v2.freeze.json"
    assert binding(input_freeze_path)["sha256"] == "11843c6fe93fc3b6547082ee7f49c0eecd65aff04151103980c1e5783401e5aa"
    execution = {"actual_execution_id":IDENTITY,"completed_execution":True,
                 "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "phase":"COMPLETED_EXISTING_FINAL_VERDICTS_AFTER_PRESERVED_OWNED_RECEIPT_CONSTRUCTION_FAILURE",
                 "candidate":binding(candidate_path),"manifest":binding(manifest_path),"audit":audit_binding,
                 "verdicts":binding(BASE+".jsonl"),"ticket":binding(ticket_path),"report":binding(BASE+".report.md"),
                 "original_adjudicator_source":old_source,"completion_helper_source":binding("reviews/adjudication/complete_planning_gap_v2_adjudication_execution.py"),
                 "original_finalization_preexecution":binding(BASE+".finalization-preexecution.json"),
                 "root_input_only_custody":binding(input_freeze_path),"final_red_custody_receipt":binding(red_path),
                 "original_audit_actual_execution":{"session_id":50606,"start_chunk":"0d2292","final_chunk":"5e912d","exit_code":0},
                 "original_finalization_actual_execution":{"chunk_id":"e86eca","exit_code":1,"failure":binding(failure_path),
                      "effect":"Existing verdicts/ticket/report were completely written; no execution receipt existed. Missing manifest in local BINDINGS caused receipt construction failure. No target or decision was incorrect or changed by completion."},
                 "all_bound_audit_inputs_and_existing_decisions_unchanged":True,
                 "decision_counts":{"QUARANTINE":408},"publication_eligible":False,"whole_cohort_hold":True,
                 "completion_creates_no_new_verdicts_or_targets":True,"fresh_renderer_r1_not_read":True,
                 "guidance":audit["independence"]}
    output = ROOT/(BASE+".execution.json")
    assert not output.exists(), "existing frozen execution must not be overwritten"
    output.write_text(json.dumps(execution,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps({"execution":binding(BASE+".execution.json"),"verdicts":binding(BASE+".jsonl"),
                      "report":binding(BASE+".report.md"),"ticket":binding(ticket_path),"decision_counts":{"QUARANTINE":408},"publication_eligible":False},sort_keys=True))


if __name__ == "__main__":
    complete()
