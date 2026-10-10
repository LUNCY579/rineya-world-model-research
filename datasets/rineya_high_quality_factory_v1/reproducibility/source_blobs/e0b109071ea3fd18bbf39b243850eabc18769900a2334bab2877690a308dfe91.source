#!/opt/anaconda3/bin/python3
"""Post-first-judgment public-interface check, independent of producer checker."""
import hashlib
import json
from pathlib import Path
from independent_memory_gap_v1 import BASE,EXECUTION_ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    out = BASE/"reviews/expert"
    first = out/"memory-gap-v2.first_judgments.jsonl"
    assert first.exists()
    first_hash = sha(first)
    document = BASE/"schemas/MEMORY_CONTROL_METADATA.md"
    document.read_text()
    public = (BASE/"schemas/MEMORY_CONTROL_PROPERTIES.md").read_text()
    family_section = public.split("## Family controls, distinct from per-record tags",1)[1].split("For any added",1)[0]
    names = {line.split("`",2)[1] for line in family_section.splitlines() if line.startswith("| `")}
    candidate = BASE/"data/REM-001/candidate_renderings/memory-gap-v2.jsonl"
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    claims_checked = 0
    for row in rows:
        seen = set()
        for claim in row["family_control_claims"]:
            assert set(claim) <= {"name","property","scope","source_family_key","evidence_manifest"}
            assert len({"name","property"} & set(claim)) == 1
            name = claim.get("name",claim.get("property"))
            assert name in names and name not in seen
            seen.add(name)
            assert claim["scope"] in ("corpus_variant","audit_probe")
            assert "source_family_key" not in claim or claim["source_family_key"] == row["source_family_key"]
            assert "evidence_manifest" not in claim or type(claim["evidence_manifest"]) is str
            claims_checked += 1
    assert sha(first) == first_hash
    report = {"reviewer_execution_id":EXECUTION_ID,"stage":"PUBLIC_INTERFACE_READ_AFTER_FROZEN_FIRST_JUDGMENTS",
              "first_judgments_sha256_before_after":first_hash,"candidate_sha256":sha(candidate),
              "public_metadata_doc_sha256":sha(document),"public_properties_sha256":sha(BASE/"schemas/MEMORY_CONTROL_PROPERTIES.md"),
              "actual_interface_claims_checked":claims_checked,"records":len(rows),"all_passed":True,
              "verdict_change":False,"pointer_contents_read":False,
              "family_scope_interpretation":"Existence of an independently established same-family witness at declared corpus_variant/audit_probe scope; not every row required to instantiate that relationship",
              "read_scope":"New public interface and factual definitions, new candidate batch, new own code, frozen own first receipts for integrity only",
              "script_sha256":sha(__file__)}
    with (out/"memory-gap-v2.post_first_metadata_interface.json").open("x") as stream:
        stream.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    run()
