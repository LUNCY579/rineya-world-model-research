"""Fail-closed validation, evidence joins and immutable release assembly."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from factory.common import ROOT, VERSION, canonical, digest, atomic, write_json, write_jsonl, load_jsonl, record_hash, candidate_paths
from factory.reference import solve as reference_solve

REQUIRED = {"sample_id", "semantic_parent_id", "semantic_family_id", "track_id", "skill_tags", "difficulty", "generator_version", "oracle_version", "generation_seed", "source_provenance", "semantic_task", "rendered_input", "verified_target", "target_format", "split", "review_status", "review_receipt_ids", "record_hash"}


def expected_target(row, oracle):
    result = oracle(row["semantic_task"])
    return {"after": result["after"]} if row["target_format"] == "answer_only" else result


def schema_validate(row):
    from jsonschema import Draft202012Validator
    schema=json.loads((ROOT/"schemas/canonical/v1.schema.json").read_text())
    Draft202012Validator(schema).validate(row)
    assert REQUIRED <= row.keys(), "missing canonical fields"
    assert row["record_hash"] == record_hash(row), "record hash mismatch"
    assert row["track_id"] in [f"{t}-001" for t in ["RCM","RGM","RVM","RSM","RMM","REM","RBM","RJP","RDP","ROM"]]
    assert 1 <= row["difficulty"] <= 7 and row["split"] in ["train","dev_challenge"]
    assert row["source_provenance"]["sealed_evaluation_used"] is False
    payload=row["rendered_input"].split("Task JSON:\n",1)[1]
    assert json.loads(payload)==row["semantic_task"], "rendering semantic mismatch"
    for key in ["sample_id","semantic_parent_id","record_hash","review_receipt_ids","verified_target"]:
        assert key not in payload, "privileged metadata in student payload"


def audit(batch):
    from oracles.independent import solve as independent_solve
    rows=[r for p in candidate_paths(batch) for r in load_jsonl(p)]
    checks=[]
    lineage=collections.defaultdict(set)
    payloads=collections.defaultdict(list)
    for row in rows:
        errors=[]
        try:
            schema_validate(row)
            gold=expected_target(row,independent_solve)
            assert gold==row["verified_target"], "independent oracle mismatch"
            assert expected_target(row,reference_solve)==gold, "reference mismatch"
        except Exception as e: errors.append(f"{type(e).__name__}: {e}")
        lineage[row["semantic_family_id"]].add(row["split"])
        payloads[digest(row["semantic_task"])].append(row["sample_id"])
        checks.append({"sample_id":row["sample_id"],"record_hash":row["record_hash"],"pass":not errors,"errors":errors,"oracle_author_execution":"/root/formal_oracle","check_runner_execution":"/root","verification":"independently authored oracle recomputation"})
    leakage=[k for k,v in lineage.items() if len(v)>1]
    exact_prompts=collections.Counter(r["rendered_input"] for r in rows)
    # Semantic repeats across renderers are intentional; cross-parent repeats are defects.
    sem=collections.defaultdict(set)
    for r in rows:sem[digest(r["semantic_task"])].add(r["semantic_parent_id"])
    duplicates=[sorted(v) for v in sem.values() if len(v)>1]
    stats={"batch":batch,"records":len(rows),"unique_parents":len(lineage),"oracle_failures":sum(not c["pass"] for c in checks),"split_leakage":leakage,"cross_parent_semantic_duplicates":duplicates,"exact_prompt_duplicates":sum(n-1 for n in exact_prompts.values() if n>1),"difficulty":dict(collections.Counter(str(r["difficulty"]) for r in rows)),"tracks":dict(collections.Counter(r["track_id"] for r in rows)),"structural_template_holdout":False,"protected_benchmark_inspected":False,"scope":"fresh generated development data; not a sealed generalization instrument"}
    write_jsonl(ROOT/f"reviews/oracle/{batch}.jsonl",checks)
    write_json(ROOT/f"audits/leakage/{batch}.json",stats)
    return stats


def load_receipts(role,batch):
    paths=sorted((ROOT/f"reviews/{role}").glob(f"{batch}*.jsonl"))
    records=[r for p in paths for r in load_jsonl(p)]
    by=collections.defaultdict(list)
    for r in records:by[r["sample_id"]].append(r)
    return by


def release(batch,version):
    stats=audit(batch)
    assert stats["oracle_failures"]==0 and not stats["split_leakage"] and not stats["cross_parent_semantic_duplicates"] and stats["exact_prompt_duplicates"]==0,stats
    evidence={role:load_receipts(role,batch) for role in ["expert","adversarial","adjudication"]}
    created=[]
    for path in candidate_paths(batch):
        track=path.parent.parent.name
        accepted=[];quarantine=[]
        for r in load_jsonl(path):
            receipts=[];fail=[]
            for role,by in evidence.items():
                matches=by.get(r["sample_id"],[])
                valid=[x for x in matches if x.get("record_hash")==r["record_hash"] and x.get("decision")=="ACCEPT"]
                if not valid:fail.append(role)
                else:receipts.append(valid[-1])
            if fail:
                quarantine.append({"sample_id":r["sample_id"],"pending_or_failed_roles":fail,"record":r})
                continue
            assert len({x["reviewer_execution_id"] for x in receipts})==3,"reviewer/adjudicator must differ"
            assert all(x["reviewer_execution_id"]!="/root" for x in receipts),"generator cannot approve itself"
            assert not any(x.get("critical_defects") or any(i.get("severity") in ["high","critical"] for i in x.get("issues",[])) for x in receipts)
            r=dict(r);r["review_status"]="ACCEPTED";r["review_receipt_ids"]=[x["receipt_id"] for x in receipts];r["record_hash"]=record_hash(r)
            accepted.append(r)
        write_jsonl(ROOT/f"data/{track}/quarantine/{batch}.jsonl",quarantine)
        if not accepted:continue
        dest=ROOT/f"data/{track}/releases/{version}"
        if dest.exists():raise FileExistsError(f"Immutable release already exists: {dest}")
        dest.mkdir(parents=True)
        write_jsonl(dest/"canonical.jsonl",accepted)
        for split in ["train","dev_challenge"]:
            subset=[r for r in accepted if r["split"]==split]
            export=[{"messages":[{"role":"user","content":r["rendered_input"]},{"role":"assistant","content":canonical(r["verified_target"])}]} for r in subset]
            write_jsonl(dest/f"{split}-sft.jsonl",export)
            write_jsonl(dest/f"{split}-structured.jsonl",[{"input":r["rendered_input"],"target":r["verified_target"]} for r in subset])
        receipts=[x for role in evidence.values() for r in accepted for x in role[r["sample_id"]] ]
        write_jsonl(dest/"review_receipts.jsonl",receipts)
        write_jsonl(dest/"oracle_receipts.jsonl",[x for x in load_jsonl(ROOT/f"reviews/oracle/{batch}.jsonl") if x["sample_id"] in {r["sample_id"] for r in accepted}])
        manifest={"release":version,"batch":batch,"track":track,"status":"QUALITY_APPROVED_LOCAL", "records":len(accepted),"semantic_parents":len({r["semantic_parent_id"] for r in accepted}),"splits":dict(collections.Counter(r["split"] for r in accepted)),"generator_version":VERSION,"oracle_version":"independent-1.0.0","source":"original synthetic; repository-private research; no third-party examples","research_effectiveness":"untested; no model training performed","reviewer_correlation":"Independent executions use the same underlying model; correlated semantic errors remain possible.","data_custody":"exposed train/dev data; no sealed evaluation","files":{}}
        atomic(dest/"QUALITY_REPORT.md",f"# {track} {version}\n\n{len(accepted)} approved records from {manifest['semantic_parents']} semantic parents. {len(quarantine)} quarantined/pending records. Exact oracle, schema, family split and three independent evidence joins passed. Refer to review_receipts.jsonl for scope and limitations. Structural template overlap is intentionally disclosed; this is development data, not evidence of OOD efficacy. Model benefits are untested. Remote status is recorded separately after byte verification.\n")
        for f in sorted(dest.iterdir()):manifest["files"][f.name]={"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
        write_json(dest/"MANIFEST.json",manifest)
        atomic(dest/"CHECKSUMS.sha256","".join(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+f.name+"\n" for f in sorted(dest.iterdir()) if f.name!="CHECKSUMS.sha256"))
        write_jsonl(ROOT/f"data/{track}/accepted/{batch}.jsonl",accepted)
        created.append(str(dest.relative_to(ROOT)))
    return created


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["audit","release"]);p.add_argument("--batch",default="pilot-v1");p.add_argument("--version",default="v0001")
    a=p.parse_args();print(json.dumps(audit(a.batch) if a.command=="audit" else release(a.batch,a.version),indent=2))


if __name__=="__main__":main()
