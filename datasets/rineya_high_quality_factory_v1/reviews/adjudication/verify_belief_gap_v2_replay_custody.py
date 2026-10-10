"""Independent exact custody comparison, without rewriting frozen adjudication."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OLD_SHA = "0b5fc58730ce02f7c2319735fe52a86cdc1a724a4930fe15b0cb34b51042c338"
NEW_SHA = "1bd7d07a1bb69db6a7d6efb5fb58c9e7c921d4bcc31273a41cf8e4c722e08700"
old_path = "audits/diversity/history/belief-gap-v2-belief-controls-" + OLD_SHA + ".json"
new_path = "audits/diversity/belief-gap-v2-belief-controls.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": digest(raw), "bytes": len(raw)}


def changes(a, b, path=""):
    if type(a) is not type(b):
        return [{"path": path, "before": a, "after": b}]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("Changed object keys: " + path)
        return [entry for key in sorted(a) for entry in changes(a[key], b[key], path + "/" + key)]
    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError("Changed list length: " + path)
        return [entry for idx, (left, right) in enumerate(zip(a, b))
                for entry in changes(left, right, path + "/" + str(idx))]
    return [] if a == b else [{"path": path, "before": a, "after": b}]


old_raw = (ROOT / old_path).read_bytes()
new_raw = (ROOT / new_path).read_bytes()
assert digest(old_raw) == OLD_SHA
assert digest(new_raw) == NEW_SHA
diff = changes(json.loads(old_raw), json.loads(new_raw))
claimed = json.loads((ROOT / "quality_reports/BELIEF_GAP_V2_AUDIT_REPLAY_DIFFERENCE.json").read_text())
assert sorted(diff, key=lambda item: item["path"]) == sorted(claimed["exact_value_changes"], key=lambda item: item["path"])
assert len(diff) == 6
assert all(item["path"].endswith(("/strongest_typed_scalar_anywhere/feature",
                                  "/strongest_typed_scalar_anywhere/present_predicts_empty")) for item in diff)
archives = []
for path, raw in [(old_path, old_raw), (new_path, new_raw)]:
    gzip_path = path + ".gz"
    assert gzip.decompress((ROOT / gzip_path).read_bytes()) == raw
    archives.append({"raw": binding(path), "gzip": binding(gzip_path), "own_decode_byte_equality": True})

expected = {
    "data/RBM-001/candidate_renderings/belief-gap-v2.jsonl": "589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df",
    "reviews/adjudication/belief-gap-v2.jsonl": "af6f96da80c5cbb715ba52ad7f0e2106b03134ac96968f5867baa05f625da960",
    "reviews/expert/belief-gap-v2.jsonl": "91b8c18ed026f6d296d7d4ace73b35420e5b99100c25790e47c815b5479bd5e1",
    "reviews/adversarial/belief-gap-v2.jsonl": "21ec9d792759d6f793eb346c893b25b44bb9e27ede560028d46ffb7397fbdde0",
    "reviews/adjudication/belief-gap-v2.custody_supplement.json": "7fa5d9a825c04817f308371e42b7c7b51708617d65187e68fd291c7ec0098420",
}
frozen_bindings = [binding(path) for path in expected]
assert all(item["sha256"] == expected[item["path"]] for item in frozen_bindings)
proof_paths = ["quality_reports/BELIEF_GAP_V2_AUDIT_REPLAY_DIFFERENCE.json",
               "feedback/improvement_requests/ORACLE-REPRO-011.json",
               "quality_reports/BELIEF_GAP_V2_QUARANTINE_EXECUTION.json"]
report = {
    "status": "COMPLETED_NEW_REPLAY_CUSTODY_SUPPLEMENT",
    "actual_verifier_execution_id": "/root/research_architect",
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "actual_source": binding(str(Path(__file__).relative_to(ROOT))),
    "original_adjudication_machine_report_sha256": OLD_SHA,
    "current_root_replay_machine_report_sha256": NEW_SHA,
    "archives": archives,
    "actual_independent_recursive_value_changes": diff,
    "parsed_equal": False,
    "scoped_equivalence": "Exactly six diagnostic representative/polarity fields changed in three length-three cohorts; every other parsed report field is identical, including semantic outputs, probes, histograms, pair checks, all diagnostic scores and error counts. This is metric/semantic equivalence, not report byte reproducibility.",
    "frozen_candidate_and_all_three_review_receipts_unchanged": frozen_bindings,
    "root_execution_evidence": [binding(path) for path in proof_paths],
    "quarantine_execution_attribution": "/root; independently read and hash-bound receipt, not executed by adjudicator",
    "data_decisions": {"REVISE": 1728},
    "direct_defects": 432,
    "dependency_holds": 1296,
    "original_adjudication_receipts_rewritten": False,
    "checker_implementation_read_or_rerun_by_adjudicator": False,
    "reproduction_root_cause": "Root ticket proposes unordered equal-score tie selection; exact six-field report difference is independently verified. Checker source cause was not inspected by adjudicator.",
    "release_approval_granted": False,
}
output = ROOT / "reviews/adjudication/belief-gap-v2.replay_custody_supplement.json"
assert not output.exists()
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": digest(output.read_bytes()), "changed_fields": len(diff), "archives_verified": len(archives)}))
