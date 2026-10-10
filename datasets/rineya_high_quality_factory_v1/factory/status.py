"""Rebuild living reports from current bytes, receipts and manifests."""
import collections
import json
import hashlib
from factory.common import ROOT, atomic, load_jsonl, write_json
from factory.lineage import source_key
from factory.catalog import write_catalog


def update():
    releases=[]
    for p in sorted(ROOT.glob("data/*/releases/*/MANIFEST.json")):
        m=json.loads(p.read_text());m["path"]=str(p.parent.relative_to(ROOT));releases.append(m)
    backups=[]
    for p in sorted((ROOT/"backup_logs").glob("*_RECEIPT.json")):
        b=json.loads(p.read_text());backups.append(b)
    active_rows,catalog=write_catalog()
    catalog_by={(m['track'],m['release']):m for m in catalog['releases']}
    totals=collections.defaultdict(lambda:{'records':0,'source_families':set(),'splits':collections.Counter()})
    for m in releases:
        path=ROOT/m['path']; rows=load_jsonl(path/'canonical.jsonl')
        m['active_status']=catalog_by[(m['track'],m['release'])]['status']
        m['active_records']=catalog_by[(m['track'],m['release'])]['active_records']
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
        matches=[]
        for backup in backups:
            inv={x['path']:x['sha256'] for x in backup.get('inventory',[])}
            if backup.get('status')=='BACKED_UP' and all(inv.get(k)==v for k,v in files.items()):matches.append(backup)
        m['remote_backup_status']='BACKED_UP' if matches else 'LOCAL_ONLY'
        m['remote_checkpoint']=matches[-1]['checkpoint_id'] if matches else None
    for row in active_rows:
        total=totals[row['track_id']];total['records']+=1;total['source_families'].add(source_key(row));total['splits'].update([row['split']])
    totals={k:{'records':v['records'],'source_families':len(v['source_families']),'splits':dict(v['splits'])} for k,v in totals.items()}
    active_difficulty={track:dict(collections.Counter(r['difficulty'] for r in active_rows if r['track_id']==track)) for track in sorted(totals)}
    current_evidence=[]
    for relative in ['quality_reports/ALL_RELEASES_INTEGRITY.json','quality_reports/FINAL_SESSION_VALIDATION.json','audits/shortcut_detection/claim-gap-v2-controls.json','research/research_gap_analysis/PRODUCTION_MILESTONE.md',
                     'quality_reports/MEMORY_GAP_V2_TEST_EXECUTION.json','quality_reports/EXPORT_COMPLETENESS_TEST_EXECUTION.json',
                     'quality_reports/EXPORT_COMPLETENESS_HISTORICAL_RELEASE_VALIDATION.json','quality_reports/MEMORY_GAP_V2_RELEASE_VALIDATION.json',
                     'research/research_gap_analysis/MEMORY_HISTORY_MILESTONE.md']:
        p=ROOT/relative
        if p.exists():current_evidence.append({'path':relative,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    candidates=[]
    for p in sorted(ROOT.glob("data/*/candidate_renderings/*.jsonl")):
        rows=load_jsonl(p)
        candidates.append({"batch":p.stem,"track":p.parent.parent.name,"records":len(rows),"parents":len({r["semantic_parent_id"] for r in rows}),"difficulty":dict(collections.Counter(r["difficulty"] for r in rows))})
    oracle_reports=[json.loads(p.read_text()) for p in sorted((ROOT/"audits/leakage").glob("*.json"))]
    review_counts={role:{"records":0,"decisions":collections.Counter()} for role in ["expert","adversarial","adjudication"]}
    batches={x['batch'] for x in candidates}
    for role in review_counts:
        for p in sorted((ROOT/f"reviews/{role}").glob("*.jsonl")):
            if p.stem not in batches:continue
            rows=load_jsonl(p);review_counts[role]["records"]+=len(rows);review_counts[role]["decisions"].update(r["decision"] for r in rows)
        review_counts[role]["decisions"]=dict(review_counts[role]["decisions"])
    snapshot={"candidates":candidates,"releases":releases,'accepted_totals':totals,'accepted_records':sum(v['records'] for v in totals.values()),'accepted_source_families':sum(v['source_families'] for v in totals.values()),'active_difficulty':active_difficulty,'current_evidence':current_evidence,"backups":[{k:b.get(k) for k in ["checkpoint_id","status","commit_sha","remote_commit_sha","approval"]} for b in backups],"reviews":review_counts,"audits":oracle_reports,'review_count_scope':'Designated completed batch judgments only; preserved first judgments excluded. Counts span historical revisions and must not be added to accepted totals.'}
    write_json(ROOT/"quality_reports/CURRENT.json",snapshot)
    table="| Track | Batch | Candidate records | Source-family IDs |\n|---|---|---:|---:|\n"+"".join(f"| {x['track']} | {x['batch']} | {x['records']} | {x['parents']} |\n" for x in candidates)
    release_lines="\n".join(f"- {m['track']} {m['release']}: {m['records']} historical records, {m['active_records']} active; {m['active_status']}, {m['remote_backup_status']} ({m['remote_checkpoint'] or 'verification pending'})." for m in releases) or "No quality-approved release yet."
    total_lines='| Track | Approved records | Distinct source families |\n|---|---:|---:|\n'+''.join(f"| {track} | {v['records']} | {v['source_families']} |\n" for track,v in sorted(totals.items()))
    backup_lines="\n".join(f"- {b['checkpoint_id']}: {b['status']}; commit {b.get('commit_sha','pending')}." for b in backups) or "No remotely verified checkpoint yet."
    source_note='Source bytes are preserved under reproducibility/source_blobs and indexed in reproducibility/SOURCE_ARCHIVE_INDEX.json. Three original broad source snapshots for historical RCM v0002 could not be recovered; v0003 supersedes it with identical reviewed data, independently reproduced generation and available source bytes. The historical limitation remains documented in quality_reports/reproducibility/RCM-001-v0002.json. Machine audit reruns preserve prior bytes when results change.'
    milestone_reached=all(totals.get(t,{}).get('source_families',0)>=150 for t in ['RCM-001','RGM-001','RVM-001']) and len(totals)==10
    milestone_text='The first P0 production milestone is reached: each priority track has at least 150 reviewed source families, and all seven other tracks have independently reviewed exact-verifiable pilots. Further generation should follow diagnosed capability gaps.' if milestone_reached else 'The first P0 production milestone remains in progress. Expand toward 150–250 reviewed source families after quality gates pass, then establish exact-verifiable pilots for the remaining tracks.'
    next_actions='Prioritize research/research_gap_analysis/PRODUCTION_MILESTONE.md and unresolved research gaps. Add new semantic mechanisms before further surface variants. Critic matched controls still have a 2:1 global overclaim/underclaim mix, limited dev scope and residual plan-length correlation. Expand memory relevance/history conflicts, skill protection/composition and belief information-gathering coverage with fresh exact contracts and independent review.' if milestone_reached else 'Complete P0 review and expand toward 150–250 reviewed source families. Establish small exact-verifiable pilots in remaining tracks.'
    if (ROOT/'research/research_gap_analysis/MEMORY_HISTORY_MILESTONE.md').is_file():
        next_actions+=' Read research/research_gap_analysis/MEMORY_HISTORY_MILESTONE.md for the newer memory coverage and remaining controlled retrieval, composition and belief gaps.'
    atomic(ROOT/"PROJECT_STATUS.md",f"# Project status\n\nCPU-only dataset production is active. Existing research is read-only. No model loading, training or weights/architecture changes.\n\n{snapshot['accepted_records']} approved canonical records from {snapshot['accepted_source_families']} distinct source families. A family is one generated world/program instance, not a new mechanism class.\n\n{total_lines}\n\nHistorical candidate inventory (revisions must not be summed):\n\n{table}\n\nApproved release inventory:\n\n{release_lines}\n\nBackups:\n\n{backup_lines}\n\nAll ten curricula exist. P0 pilot correctness/review precedes expansion. Counterfactual/rendering counts do not inflate parent counts. Exposed dev challenges are not sealed tests. Exact state correctness is a data property; model benefit remains untested.\n")
    atomic(ROOT/"DATASET_QUALITY_REPORT.md",f"# Dataset quality report\n\nMachine evidence: quality_reports/CURRENT.json. Candidate batches are retained even when superseded. Review decisions are bound to exact candidate hashes and independent execution identities.\n\n{table}\n\nReview decision counts:\n\n```json\n{json.dumps(review_counts,indent=2)}\n```\n\nExact oracle, schema, duplicates and family split audit summaries:\n\n```json\n{json.dumps(oracle_reports,indent=2)}\n```\n\nTemplate overlap remains a known scope limitation. All three P0 task families require independent expert/adversarial assessment and adjudication before release. Oracle equivalence and mechanized review do not establish native model performance. Reviewers share the underlying model, with correlated-error risk. Generation uses local CPU arithmetic/BFS, no paid external generation service; agent-token costs are not metered per sample.\n")
    atomic(ROOT/"CURRICULUM_COVERAGE.md",f"# Actual curriculum coverage\n\n{milestone_text}\n\nActive approved inventory:\n\n{total_lines}\n\nActive difficulty counts (canonical records, including render variants):\n\n```json\n{json.dumps(active_difficulty,indent=2)}\n```\n\nHistorical candidate inventory; revisions and renderings must not be added to active family counts:\n\n{table}\n\nDifficulty counts by historical batch:\n\n```json\n{json.dumps({x['track']+'/'+x['batch']:x['difficulty'] for x in candidates},indent=2)}\n```\n\nUse research/curricula cards for all ten proposed programs. Actual mechanisms and split axes are in semantic_parents files, not inferred from prompt length. Unsupported latent representation, hidden-causal identification, unrestricted social inference and sealed OOD evidence remain gaps. Surface twins do not create new independent concepts.\n")
    atomic(ROOT/"TRAINING_READINESS.md",f"# Training readiness\n\nAuthoritative active inventory: manifests/ACTIVE_RELEASE_CATALOG.json. Load through factory.catalog.catalog(), which excludes exact-hash errata and superseded versions; do not concatenate historical release directories. Original release bytes remain preserved.\n\n{release_lines}\n\n{source_note}\n\nOnly independently approved active release bytes may be considered for a separately authorized future training experiment. Pending candidates and quarantine files are excluded from student exports. Claims are limited to the stated exact synthetic domains. The factory does not establish transfer, retention, larger-world success or intelligence improvement. Structural template overlap, symbolic AST presentation and same-model reviewer correlation limit interpretation. Train/dev records are exposed; separate custody is needed for a future sealed evaluation.\n")
    atomic(ROOT/"GITHUB_BACKUP_LOG.md",f"# Verified GitHub backups\n\nRepository LUNCY579/rineya-world-model-research is private; dedicated branch research/rineya-high-quality-dataset-v1; tracked prefix datasets/rineya_high_quality_factory_v1. Original checkout remains on main.\n\n{backup_lines}\n\nDetailed exact inventories and actual remote blob SHA256 evidence are in backup_logs/*_RECEIPT.json. Each receipt is written after its checkpoint, so the receipt itself is local until a later checkpoint. A commit alone is not remote verification. Release directories retain immutable local approval state; remote backup status is separately certified by inventory.\n")
    atomic(ROOT/"NEXT_SESSION_HANDOFF.md",f"# Next-session handoff\n\n{milestone_text}\n\n{total_lines}\n\nRead PROJECT_STATUS.md, manifests/ACTIVE_RELEASE_CATALOG.json, quality_reports/CURRENT.json and backup_logs receipts first. Inspect original repo/worktree status; never switch or clean the research checkout. Load approved rows through factory.catalog.catalog(); the old RVM pilot is superseded and retained solely as historical evidence.\n\n{release_lines}\n\n{backup_lines}\n\nRun `/opt/anaconda3/bin/python3 -m unittest discover -s tests -p '*.py'` from the factory, and `/opt/anaconda3/bin/python3 backup_logs/test_checkpoint_backup.py`. Audit fresh candidate batches with `-m factory.pipeline audit --batch NAME`. Full state and independent factual metadata checks must pass. Reviewers must see no prior verdict before their first judgement. New hashes require new review receipts. Assemble only ACCEPT-gated immutable releases with `-m factory.pipeline release --batch NAME --version vNNNN`. Corrections require fresh batches and an explicit supersedes argument through factory.pipeline.release; do not rewrite existing releases.\n\n{next_actions}\n\nThe 80 expansion competency-tag errors and 6 historical pilot errors were fixed through fresh reviews and a superseding pilot; use feedback/resolved_issues/META-FIX-001.json. Claim-gap-v1 failed 12 renderer hashes and remains quarantined; v2 fixes the alias and has fresh review evidence. Historical review decisions and failed candidate bytes remain preserved and are excluded from active training counts. Initial reviews were blind; correction-topic guidance and same-model correlation are disclosed in research/ROLE_REGISTRY.json.\n\nCreate no sealed evaluation under current exposed custody. Back up exact file inventories with checkpoint_backup.py; failed pushes preserve local commits and support --resume-commit. Each post-push receipt is local until a later checkpoint. No model training or background execution is promised.\n")
    atomic(ROOT/"handoff/NEXT_SESSION_HANDOFF.md",(ROOT/"NEXT_SESSION_HANDOFF.md").read_text())
    for relative in ['NEXT_SESSION_HANDOFF.md','handoff/NEXT_SESSION_HANDOFF.md','DATASET_QUALITY_REPORT.md']:
        path=ROOT/relative
        atomic(path,path.read_text()+'\n'+source_note+'\n')
    return snapshot


if __name__=="__main__":
    report=update();print(json.dumps({"candidate_batches":len(report["candidates"]),"approved_releases":len(report["releases"]),"backups":len(report["backups"])}))
