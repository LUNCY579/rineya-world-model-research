"""Rebuild living reports from current bytes, receipts and manifests."""
import collections
import json
from factory.common import ROOT, atomic, load_jsonl, write_json


def update():
    releases=[]
    for p in sorted(ROOT.glob("data/*/releases/*/MANIFEST.json")):
        m=json.loads(p.read_text());m["path"]=str(p.parent.relative_to(ROOT));releases.append(m)
    backups=[]
    for p in sorted((ROOT/"backup_logs").glob("*_RECEIPT.json")):
        b=json.loads(p.read_text());backups.append(b)
    candidates=[]
    for p in sorted(ROOT.glob("data/*/candidate_renderings/*.jsonl")):
        rows=load_jsonl(p)
        candidates.append({"batch":p.stem,"track":p.parent.parent.name,"records":len(rows),"parents":len({r["semantic_parent_id"] for r in rows}),"difficulty":dict(collections.Counter(r["difficulty"] for r in rows))})
    oracle_reports=[json.loads(p.read_text()) for p in sorted((ROOT/"audits/leakage").glob("*.json"))]
    review_counts={role:{"records":0,"decisions":collections.Counter()} for role in ["expert","adversarial","adjudication"]}
    for role in review_counts:
        for p in sorted((ROOT/f"reviews/{role}").glob("*.jsonl")):
            if "calibration" in p.name:continue
            rows=load_jsonl(p);review_counts[role]["records"]+=len(rows);review_counts[role]["decisions"].update(r["decision"] for r in rows)
        review_counts[role]["decisions"]=dict(review_counts[role]["decisions"])
    snapshot={"candidates":candidates,"releases":releases,"backups":[{k:b.get(k) for k in ["checkpoint_id","status","commit_sha","remote_commit_sha","approval"]} for b in backups],"reviews":review_counts,"audits":oracle_reports}
    write_json(ROOT/"quality_reports/CURRENT.json",snapshot)
    table="| Track | Batch | Candidate records | Independent semantic parents |\n|---|---|---:|---:|\n"+"".join(f"| {x['track']} | {x['batch']} | {x['records']} | {x['parents']} |\n" for x in candidates)
    release_lines="\n".join(f"- {m['track']} {m['release']}: {m['records']} records, {m['semantic_parents']} parents; {m['status']}." for m in releases) or "No quality-approved release yet."
    backup_lines="\n".join(f"- {b['checkpoint_id']}: {b['status']}; commit {b.get('commit_sha','pending')}." for b in backups) or "No remotely verified checkpoint yet."
    atomic(ROOT/"PROJECT_STATUS.md",f"# Project status\n\nCPU-only dataset production is active. Existing research is read-only. No model loading, training or weights/architecture changes.\n\n{table}\n\nApproved release inventory:\n\n{release_lines}\n\nBackups:\n\n{backup_lines}\n\nAll ten curricula exist. P0 pilot correctness/review precedes expansion. Counterfactual/rendering counts do not inflate parent counts. Exposed dev challenges are not sealed tests. Exact state correctness is a data property; model benefit remains untested.\n")
    atomic(ROOT/"DATASET_QUALITY_REPORT.md",f"# Dataset quality report\n\nMachine evidence: quality_reports/CURRENT.json. Candidate batches are retained even when superseded. Review decisions are bound to exact candidate hashes and independent execution identities.\n\n{table}\n\nReview decision counts:\n\n```json\n{json.dumps(review_counts,indent=2)}\n```\n\nExact oracle, schema, duplicates and family split audit summaries:\n\n```json\n{json.dumps(oracle_reports,indent=2)}\n```\n\nTemplate overlap remains a known scope limitation. All three P0 task families require independent expert/adversarial assessment and adjudication before release. Oracle equivalence and mechanized review do not establish native model performance. Reviewers share the underlying model, with correlated-error risk. Generation uses local CPU arithmetic/BFS, no paid external generation service; agent-token costs are not metered per sample.\n")
    atomic(ROOT/"CURRICULUM_COVERAGE.md",f"# Actual curriculum coverage\n\n{table}\n\nDifficulty counts by batch:\n\n```json\n{json.dumps({x['track']+'/'+x['batch']:x['difficulty'] for x in candidates},indent=2)}\n```\n\nUse research/curricula cards for all ten proposed programs. Actual mechanisms and split axes are in semantic_parents files, not inferred from prompt length. Unsupported latent representation, hidden-causal identification, unrestricted social inference and sealed OOD evidence remain gaps. Surface twins do not create new independent concepts.\n")
    atomic(ROOT/"TRAINING_READINESS.md",f"# Training readiness\n\n{release_lines}\n\nOnly independently approved release bytes may be considered for a separately authorized future training experiment. Pending candidates and quarantine files are excluded from student exports. Claims are limited to the stated exact synthetic domains. The factory does not establish transfer, retention, larger-world success or intelligence improvement. Structural template overlap, symbolic AST presentation and same-model reviewer correlation limit interpretation. Train/dev records are exposed; separate custody is needed for a future sealed evaluation.\n")
    atomic(ROOT/"GITHUB_BACKUP_LOG.md",f"# Verified GitHub backups\n\nRepository LUNCY579/rineya-world-model-research is private; dedicated branch research/rineya-high-quality-dataset-v1; tracked prefix datasets/rineya_high_quality_factory_v1. Original checkout remains on main.\n\n{backup_lines}\n\nDetailed exact inventories and actual remote blob SHA256 evidence are in backup_logs/*_RECEIPT.json. Each receipt is written after its checkpoint, so the receipt itself is local until a later checkpoint. A commit alone is not remote verification. Release directories retain immutable local approval state; remote backup status is separately certified by inventory.\n")
    atomic(ROOT/"NEXT_SESSION_HANDOFF.md",f"# Next-session handoff\n\nRead PROJECT_STATUS.md, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout.\n\n{release_lines}\n\n{backup_lines}\n\nRun `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory. Audit each candidate batch with `-m factory.pipeline audit --batch NAME`. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Revisions require new batches and regression evidence; do not rewrite existing releases.\n\nExpand P0 toward 150–250 reviewed parents by semantic gaps after pilots pass. Remaining seven tracks need small exact-verifiable pilots and independent reviews. Create no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. No background execution is promised.\n")
    atomic(ROOT/"handoff/NEXT_SESSION_HANDOFF.md",(ROOT/"NEXT_SESSION_HANDOFF.md").read_text())
    return snapshot


if __name__=="__main__":
    report=update();print(json.dumps({"candidate_batches":len(report["candidates"]),"approved_releases":len(report["releases"]),"backups":len(report["backups"])}))
