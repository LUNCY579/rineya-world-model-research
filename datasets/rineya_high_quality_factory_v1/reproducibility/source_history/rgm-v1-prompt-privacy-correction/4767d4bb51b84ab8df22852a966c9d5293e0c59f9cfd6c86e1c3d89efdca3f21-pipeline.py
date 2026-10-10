"""Fail-closed validation, evidence joins and immutable release assembly."""
import argparse
import collections
import hashlib
import importlib
import json
from pathlib import Path
from factory.common import ROOT, VERSION, canonical, digest, atomic, write_json, write_jsonl, load_jsonl, record_hash, candidate_paths
from factory.reference import solve as reference_solve
from factory.planning_controls import planning_control_rows, planning_control_report

REQUIRED = {"sample_id", "semantic_parent_id", "semantic_family_id", "track_id", "skill_tags", "difficulty", "generator_version", "oracle_version", "generation_seed", "source_provenance", "semantic_task", "rendered_input", "verified_target", "target_format", "split", "review_status", "review_receipt_ids", "record_hash"}
APPROVAL_ACTORS = {'expert': '/root/expert_reviewer', 'adversarial': '/root/adversarial_auditor',
                   'adjudication': '/root/research_architect'}


def expected_target(row, oracle):
    result = oracle(row["semantic_task"])
    return {"after": result["after"]} if row["target_format"] == "answer_only" else result


def memory_control_rows(rows):
    return [row for row in rows if row.get('semantic_task',{}).get('kind')=='episodic'
            and (row.get('controlled_variant') is not None or row.get('query_role_assignments') is not None)]


def memory_control_report(rows):
    selected=memory_control_rows(rows)
    if not selected:return None
    from oracles.memory_controls_independent import check_batch
    return check_batch(selected)


BELIEF_PROPERTY_TAGS = {
    'multiworld_unobserved_invariant', 'conjunction_required_invariant',
    'useful_after_conjunctive_evidence', 'multiworld_all_queries_redundant',
    'multiworld_entailed_null', 'ambiguous_null_partition',
    'cross_fact_inconsistent_conjunction', 'post_empty_absorption',
    'nonempty_repeated_observation', 'typed_observation_separator',
    'typed_query_partition_separator', 'skewed_possible_value_partition',
}
BELIEF_V2_VERSION = 'rbm-belief-gap-2.0.0'
BELIEF_V2_CLAIMS = {'same_prefix_empty_nonempty_counterfactual',
                    'contradiction_equals_mixed_outcomes'}
BELIEF_V3_VERSION = 'rbm-belief-gap-3.0.0'
BELIEF_V3_CLAIMS = {'auxiliary_scalar_transposition_invariance',
                    'same_observations_auxiliary_table_counterfactual',
                    'context_auxiliary_scalar_variant_crossing'}


def belief_checker_name(rows):
    is_v3 = any(row.get('generator_version') == BELIEF_V3_VERSION
                or 'auxiliary_scalar_variant' in row or 'auxiliary_scalar_control' in row
                or any(isinstance(claim, dict)
                       and (claim.get('name') or claim.get('property')) in BELIEF_V3_CLAIMS
                       for claim in row.get('family_control_claims', []))
                for row in rows)
    if is_v3:
        return 'oracles.belief_controls_v3_independent'
    # Normal revision provenance survives filtering out all G/H rows.
    # The context/claim checks also prevent a changed version field from
    # hiding the published v2 obligations.
    is_v2 = any(row.get('generator_version') == BELIEF_V2_VERSION
                or row.get('controlled_context') in {'G', 'H'}
                or any(isinstance(claim, dict)
                       and (claim.get('name') or claim.get('property')) in BELIEF_V2_CLAIMS
                       for claim in row.get('family_control_claims', []))
                for row in rows)
    return 'oracles.belief_controls_v2_independent' if is_v2 else 'oracles.belief_controls_independent'


def belief_checker(rows):
    return importlib.import_module(belief_checker_name(rows))


def belief_control_rows(rows):
    markers = {'controlled_context', 'structural_family_profile',
               'fact_role_assignments', 'query_role_assignments',
               'property_witnesses', 'family_control_claims',
               'auxiliary_scalar_variant', 'auxiliary_scalar_control'}
    tags = BELIEF_PROPERTY_TAGS | {'finite_noiseless_filter', 'possible_value_partition'}
    return [row for row in rows if row.get('semantic_task', {}).get('kind') == 'belief'
            and (row.get('generator_version') in {BELIEF_V2_VERSION, BELIEF_V3_VERSION}
                 or markers.intersection(row) or tags.intersection(row.get('skill_tags', [])))]


def belief_control_report(rows):
    selected = belief_control_rows(rows)
    if not selected:
        return None
    return belief_checker(selected).check_batch(selected)


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
    # Python container equality conflates bools with integers (True == 1).
    # Evidence joins must preserve the exact public JSON scalar types.
    assert canonical(json.loads(payload))==canonical(row["semantic_task"]), "rendering semantic mismatch"
    for key in ["sample_id","semantic_parent_id","record_hash","review_receipt_ids","verified_target"]:
        assert key not in row["rendered_input"], "privileged metadata in student input"


def preserve_prior_audit(path, new_bytes):
    """Keep the exact earlier machine evidence if a rerun changes its bytes."""
    if path.exists() and path.read_bytes()!=new_bytes:
        previous=path.read_bytes()
        sha=hashlib.sha256(previous).hexdigest()
        snapshot=path.parent/'history'/f'{path.stem}-{sha}{path.suffix}'
        if snapshot.exists():assert snapshot.read_bytes()==previous,'Audit snapshot mismatch'
        else:atomic(snapshot,previous.decode('utf-8'))


def preserve_source_bytes(paths):
    """Snapshot source before publishing a manifest that references its hash."""
    hashes={}
    for source in paths:
        content=source.read_bytes();sha=hashlib.sha256(content).hexdigest()
        blob=ROOT/'reproducibility/source_blobs'/f'{sha}.source'
        if blob.exists():assert blob.read_bytes()==content,'Source archive mismatch'
        else:atomic(blob,content.decode('utf-8'))
        hashes[str(source.relative_to(ROOT))]=sha
    return hashes


def audit(batch):
    from oracles.independent import solve as independent_solve
    rows=[r for p in candidate_paths(batch) for r in load_jsonl(p)]
    control_source=ROOT/'oracles/memory_controls_independent.py'
    control_source_sha=hashlib.sha256(control_source.read_bytes()).hexdigest() if memory_control_rows(rows) else None
    selected_belief_rows=belief_control_rows(rows)
    belief_module=belief_checker(selected_belief_rows) if selected_belief_rows else None
    belief_source=Path(belief_module.__file__) if belief_module else None
    belief_source_sha=hashlib.sha256(belief_source.read_bytes()).hexdigest() if belief_source else None
    belief_dependencies={}
    if belief_module and belief_module.__name__.endswith('belief_controls_v2_independent'):
        source=ROOT/'oracles/belief_controls_independent.py'
        belief_dependencies[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    elif belief_module and belief_module.__name__.endswith('belief_controls_v3_independent'):
        for name in ['belief_controls_independent.py', 'belief_controls_v2_independent.py']:
            source=ROOT/'oracles'/name
            belief_dependencies[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    checks=[]
    lineage=collections.defaultdict(set)
    payloads=collections.defaultdict(list)
    for row in rows:
        errors=[]
        try:
            schema_validate(row)
            gold=expected_target(row,independent_solve)
            assert canonical(gold)==canonical(row["verified_target"]), "independent oracle mismatch"
            assert canonical(expected_target(row,reference_solve))==canonical(gold), "reference mismatch"
            if planning_control_rows([row], batch):
                from oracles.rgm_planning_controls_independent import solve as planning_solve
                assert canonical(expected_target(row,planning_solve))==canonical(gold), "planning independent exact oracle mismatch"
        except Exception as e: errors.append(f"{type(e).__name__}: {e}")
        from oracles.metadata_independent import check as metadata_check
        metadata_errors=metadata_check(row)
        if memory_control_rows([row]):
            from oracles.memory_controls_independent import check as memory_check
            metadata_errors.extend(memory_check(row))
        if belief_control_rows([row]):
            metadata_errors.extend(belief_module.check(row))
        if planning_control_rows([row], batch):
            from oracles.rgm_planning_controls_independent import check as planning_check
            metadata_errors.extend(planning_check(row))
        errors.extend('Metadata: '+e for e in metadata_errors)
        lineage[row["semantic_family_id"]].add(row["split"])
        payloads[digest(row["semantic_task"])].append(row["sample_id"])
        checks.append({"sample_id":row["sample_id"],"record_hash":row["record_hash"],"pass":not errors,"errors":errors,'metadata_errors':metadata_errors,"oracle_author_execution":"/root/formal_oracle","check_runner_execution":"/root","verification":"independently authored state and factual metadata oracle recomputation"})
    leakage=[k for k,v in lineage.items() if len(v)>1]
    exact_prompts=collections.Counter(r["rendered_input"] for r in rows)
    # Semantic repeats across renderers are intentional; cross-parent repeats are defects.
    sem=collections.defaultdict(set)
    for r in rows:sem[digest(r["semantic_task"])].add(r["semantic_parent_id"])
    duplicates=[sorted(v) for v in sem.values() if len(v)>1]
    parent_tasks=collections.defaultdict(set)
    for r in rows:parent_tasks[r["semantic_parent_id"]].add(digest(r["semantic_task"]))
    distinct_parent_signatures={digest(sorted(tasks)) for tasks in parent_tasks.values()}
    stats={"batch":batch,"records":len(rows),"unique_parents":len(lineage),"distinct_parent_semantic_signatures":len(distinct_parent_signatures),"oracle_failures":sum(not c["pass"] for c in checks),"split_leakage":leakage,"cross_parent_semantic_duplicates":duplicates,"exact_prompt_duplicates":sum(n-1 for n in exact_prompts.values() if n>1),"difficulty":dict(collections.Counter(str(r["difficulty"]) for r in rows)),"tracks":dict(collections.Counter(r["track_id"] for r in rows)),"structural_template_holdout":False,"protected_benchmark_inspected":False,"scope":"fresh generated development data; not a sealed generalization instrument"}
    stats['metadata_failures']=sum(bool(c['metadata_errors']) for c in checks)
    stats['validation_failures']=stats['oracle_failures']
    stats['oracle_failures']=sum(any(not e.startswith('Metadata: ') for e in c['errors']) for c in checks)
    controls=memory_control_report(rows)
    if controls:
        assert hashlib.sha256(control_source.read_bytes()).hexdigest()==control_source_sha,'Memory checker source changed during audit; rerun after source is stable'
    stats['memory_control_error_count']=len(controls['errors']) if controls else 0
    if controls:
        stats['memory_control_counts']=controls['counts']
        control_path=ROOT/f'audits/diversity/{batch}-memory-controls.json'
        payload={'oracle_author_execution':'/root/formal_oracle','check_runner_execution':'/root',
                 'checker_sha256':control_source_sha,
                 'batch':batch,'report':controls}
        content=(json.dumps(payload,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        preserve_prior_audit(control_path,content)
        atomic(control_path,content.decode('utf-8'))
    belief_controls=belief_control_report(rows)
    stats['belief_control_error_count']=len(belief_controls['errors']) if belief_controls else 0
    if belief_controls:
        assert hashlib.sha256(belief_source.read_bytes()).hexdigest()==belief_source_sha,'Belief checker source changed during audit; rerun after source is stable'
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in belief_dependencies.items()),'Belief checker dependency changed during audit'
        stats['belief_control_counts']=belief_controls['counts']
        control_path=ROOT/f'audits/diversity/{batch}-belief-controls.json'
        payload={'oracle_author_execution':'/root/formal_oracle','check_runner_execution':'/root',
                 'checker_sha256':belief_source_sha,'batch':batch,'report':belief_controls}
        if belief_dependencies:payload['checker_dependency_sha256']=belief_dependencies
        content=(json.dumps(payload,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        preserve_prior_audit(control_path,content)
        atomic(control_path,content.decode('utf-8'))
    planning_controls=planning_control_report(rows,batch,ROOT)
    stats['planning_control_error_count']=len(planning_controls['errors']) if planning_controls else 0
    if planning_controls:
        stats['planning_control_counts']=planning_controls['counts']
        control_path=ROOT/f'audits/diversity/{batch}-planning-controls.json'
        payload={'oracle_author_execution':'/root/formal_oracle','check_runner_execution':'/root',
                 'batch':batch,'report':planning_controls}
        content=(json.dumps(payload,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        preserve_prior_audit(control_path,content)
        atomic(control_path,content.decode('utf-8'))
    stats['batch_control_failure_count']=sum(stats[name] for name in [
        'memory_control_error_count','belief_control_error_count','planning_control_error_count'])
    stats['all_machine_quality_gates_pass']=not (
        stats['validation_failures'] or stats['batch_control_failure_count']
        or stats['split_leakage'] or stats['cross_parent_semantic_duplicates']
        or stats['exact_prompt_duplicates'])
    receipt_path=ROOT/f"reviews/oracle/{batch}.jsonl"
    report_path=ROOT/f"audits/leakage/{batch}.json"
    preserve_prior_audit(receipt_path,(''.join(canonical(c)+'\n' for c in checks)).encode('utf-8'))
    preserve_prior_audit(report_path,(json.dumps(stats,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    write_jsonl(receipt_path,checks)
    write_json(report_path,stats)
    return stats


def load_receipts(role,batch):
    # Initial blind drafts remain audit evidence, but only the designated
    # completed judgement file controls approval.
    final=ROOT/f"reviews/{role}/{batch}.jsonl"
    paths=[final] if final.exists() else []
    records=[r for p in paths for r in load_jsonl(p)]
    by=collections.defaultdict(list)
    for r in records:by[r["sample_id"]].append(r)
    return by


def approval_evidence(row,evidence):
    receipts=[];fail=[]
    for role,by in evidence.items():
        matches=by.get(row["sample_id"],[])
        current=matches[-1] if matches else None
        if not current or current.get("record_hash")!=row["record_hash"] or current.get("decision")!="ACCEPT":fail.append(role)
        else:receipts.append(current)
    if fail:return receipts,fail
    ids={x["reviewer_execution_id"] for x in receipts}
    if (set(evidence)!=set(APPROVAL_ACTORS) or len(ids)!=3
            or any(by[row['sample_id']][-1]['reviewer_execution_id']!=APPROVAL_ACTORS.get(role)
                   for role,by in evidence.items())):
        fail.append("nonindependent_approval")
    if any(x.get("critical_defects") or any(i.get("severity") in ["high","critical"] for i in x.get("issues",[])) for x in receipts):fail.append("unresolved_serious_issue")
    return receipts,fail


def exporter_approval():
    expert=json.loads((ROOT/'reviews/expert/exporter_revised_CODE_REVIEW.json').read_text())
    red=json.loads((ROOT/'reviews/adversarial/exports-code-review.json').read_text())
    sha=hashlib.sha256((ROOT/'factory/exports.py').read_bytes()).hexdigest()
    assert expert['decision']=='ACCEPT' and red['decision']=='ACCEPT_CODE'
    assert expert['reviewer_execution_id']=='/root/expert_reviewer' and red['reviewer_execution_id']=='/root/adversarial_auditor'
    assert expert['artifact_sha256']==sha==red['source_sha256'],'stale exporter code review'
    for evidence in [expert['artifact_hashes'],red['schema_sha256']]:
        for relative,expected in evidence.items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,('stale exporter artifact',relative)
    assert not expert.get('critical_defects') and not any(i.get('severity') in ['high','critical'] for r in [expert,red] for i in r.get('issues',[]))
    return {'exporter_sha256':sha,'expert_receipt':expert['receipt_id'],'adversarial_evidence':'reviews/adversarial/exports-code-review.json'}


def release(batch,version,tracks=None,supersedes=None):
    stats=audit(batch)
    # Failed diagnostic tracks in a mixed batch do not prevent a separately
    # approved selected track from release; validate each selected row below.
    selected_ids={r['sample_id'] for p in candidate_paths(batch) for r in load_jsonl(p) if not tracks or r['track_id'] in tracks}
    selected_checks=[c for c in load_jsonl(ROOT/f'reviews/oracle/{batch}.jsonl') if c['sample_id'] in selected_ids]
    assert selected_ids and len(selected_checks)==len(selected_ids),'Selected batch missing records or duplicate sample IDs'
    assert all(c['pass'] for c in selected_checks) and not stats["split_leakage"] and not stats["cross_parent_semantic_duplicates"] and stats["exact_prompt_duplicates"]==0,stats
    from factory.lineage import cross_release
    new_rows=[r for p in candidate_paths(batch) for r in load_jsonl(p) if not tracks or r['track_id'] in tracks]
    prior=[]
    from factory.lineage import source_key
    if supersedes:
        assert tracks and len(tracks)==1,'Supersession must name exactly one track'
        replaced=ROOT/f'data/{tracks[0]}/releases/{supersedes}/canonical.jsonl'
        assert replaced.is_file() and supersedes!=version
        assert {source_key(r) for r in load_jsonl(replaced)}=={source_key(r) for r in new_rows},'Supersession must cover exactly the same source families'
    for p in ROOT.glob('data/*/releases/*/canonical.jsonl'):
        if supersedes and p==replaced:continue
        prior.extend(load_jsonl(p))
    lineage=cross_release(new_rows,prior)
    write_json(ROOT/f'audits/leakage/{batch}_cross_release.json',lineage)
    assert not any(lineage.values()), 'Cross-release duplicate family or split reuse: '+str(lineage)
    if any(r['track_id']=='RVM-001' for r in new_rows):exporter_evidence=exporter_approval()
    else:exporter_evidence=None
    evidence={role:load_receipts(role,batch) for role in ["expert","adversarial","adjudication"]}
    created=[]
    for path in candidate_paths(batch):
        track=path.parent.parent.name
        if tracks and track not in tracks:continue
        accepted=[];quarantine=[]
        for r in load_jsonl(path):
            receipts,fail=approval_evidence(r,evidence)
            if fail:
                quarantine.append({"sample_id":r["sample_id"],"pending_or_failed_roles":fail,"record":r})
                continue
            r=dict(r);r["reviewed_candidate_hash"]=r["record_hash"];r["review_status"]="ACCEPTED";r["review_receipt_ids"]=[x["receipt_id"] for x in receipts];r["record_hash"]=record_hash(r)
            accepted.append(r)
        write_jsonl(ROOT/f"data/{track}/quarantine/{batch}.jsonl",quarantine)
        if supersedes:assert not quarantine,'Full supersession requires every replacement record accepted'
        if not accepted:continue
        # Complete reviewed family controls must survive approval filtering.
        # A missing approved rotation cannot silently break the release's claim.
        controls=memory_control_report(accepted)
        assert not controls or not controls['errors'],'Released memory family controls incomplete or false'
        belief_controls=belief_control_report(accepted)
        assert not belief_controls or not belief_controls['errors'],'Released belief family controls incomplete or false'
        planning_controls=planning_control_report(accepted,batch,ROOT)
        assert not planning_controls or not planning_controls['errors'],'Released planning family controls incomplete or false'
        dest=ROOT/f"data/{track}/releases/{version}"
        if dest.exists():raise FileExistsError(f"Immutable release already exists: {dest}")
        dest.mkdir(parents=True)
        write_jsonl(dest/"canonical.jsonl",accepted)
        for split in ["train","dev_challenge"]:
            subset=[r for r in accepted if r["split"]==split]
            export=[{"messages":[{"role":"user","content":r["rendered_input"]},{"role":"assistant","content":canonical(r["verified_target"])}]} for r in subset]
            write_jsonl(dest/f"{split}-sft.jsonl",export)
            write_jsonl(dest/f"{split}-structured.jsonl",[{"input":r["rendered_input"],"target":r["verified_target"]} for r in subset])
        from factory.exports import write_specialized
        write_specialized(dest,accepted)
        receipts=[x for role in evidence.values() for r in accepted for x in role[r["sample_id"]] ]
        write_jsonl(dest/"review_receipts.jsonl",receipts)
        write_jsonl(dest/"oracle_receipts.jsonl",[x for x in load_jsonl(ROOT/f"reviews/oracle/{batch}.jsonl") if x["sample_id"] in {r["sample_id"] for r in accepted}])
        code_files=sorted((ROOT/"factory").glob("*.py"))+sorted((ROOT/"oracles").glob("*.py"))+sorted((ROOT/"schemas").glob("*.md"))+sorted((ROOT/"schemas").rglob('*.schema.json'))+sorted((ROOT/"generators").rglob("*.py"))
        source_hashes=preserve_source_bytes(code_files)
        manifest={"release":version,"batch":batch,"track":track,"status":"QUALITY_APPROVED_LOCAL", "records":len(accepted),"semantic_parents":len({r["semantic_parent_id"] for r in accepted}),"splits":dict(collections.Counter(r["split"] for r in accepted)),"generator_versions":sorted({r["generator_version"] for r in accepted}),"framework_version":VERSION,"oracle_version":"independent-1.0.0","reproducibility_code_sha256":source_hashes,"source_archive":"reproducibility/source_blobs/{sha256}.source","batch_configuration":f"manifests/{batch}_GENERATION.json","source":"original synthetic; repository-private research; no third-party examples","research_effectiveness":"untested; no model training performed","reviewer_correlation":"Independent executions use the same underlying model; correlated semantic errors remain possible.","data_custody":"exposed train/dev data; no sealed evaluation","files":{}}
        manifest['export_profile']='specialized-1.0.0'
        if supersedes:manifest['supersedes']=[supersedes]
        if track=='RVM-001':manifest['exporter_review']=exporter_evidence
        atomic(dest/"QUALITY_REPORT.md",f"# {track} {version}\n\n{len(accepted)} approved records from {manifest['semantic_parents']} semantic parents. {len(quarantine)} quarantined/pending records. Exact oracle, schema, family split and three independent evidence joins passed. Refer to review_receipts.jsonl for scope and limitations. Structural template overlap is intentionally disclosed; this is development data, not evidence of OOD efficacy. Model benefits are untested. Remote status is recorded separately after byte verification.\n")
        for f in sorted(dest.iterdir()):manifest["files"][f.name]={"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
        write_json(dest/"MANIFEST.json",manifest)
        atomic(dest/"CHECKSUMS.sha256","".join(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+f.name+"\n" for f in sorted(dest.iterdir()) if f.name!="CHECKSUMS.sha256"))
        write_jsonl(ROOT/f"data/{track}/accepted/{batch}.jsonl",accepted)
        created.append(str(dest.relative_to(ROOT)))
    return created


def validate_release(path):
    from oracles.independent import solve as oracle
    path=Path(path)
    manifest=json.loads((path/"MANIFEST.json").read_text())
    checksum_lines=(path/"CHECKSUMS.sha256").read_text().splitlines()
    checks={name:sha for sha,name in (line.split("  ",1) for line in checksum_lines)}
    files={p.name for p in path.iterdir() if p.is_file() and p.name!="CHECKSUMS.sha256"}
    assert set(checks)==files,"checksum coverage mismatch"
    for name,sha in checks.items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==sha,(name,"checksum mismatch")
    for name,meta in manifest["files"].items():
        f=path/name
        assert f.stat().st_size==meta["bytes"] and hashlib.sha256(f.read_bytes()).hexdigest()==meta["sha256"],name
    if manifest.get('source_archive'):
        for sha in set(manifest['reproducibility_code_sha256'].values()):
            blob=ROOT/'reproducibility/source_blobs'/f'{sha}.source'
            assert blob.is_file() and hashlib.sha256(blob.read_bytes()).hexdigest()==sha,'Missing exact release source bytes'
    rows=load_jsonl(path/"canonical.jsonl")
    assert len(rows)==manifest["records"]
    if manifest.get('export_profile'):
        expected_suffixes={'RGM-001':['trajectories'],'RVM-001':['preference','process_reward'],'RSM-001':['skill_demonstrations'],'REM-001':['memory_updates'],'RMM-001':['memory_updates'],'RBM-001':['belief_updates'],'ROM-001':['belief_updates']}.get(manifest['track'],[])
        assert all((path/f'{split}-{suffix}.jsonl').is_file() for split in ['train','dev_challenge'] for suffix in expected_suffixes),'missing declared specialized export'
    receipt_by={r["receipt_id"]:r for r in load_jsonl(path/"review_receipts.jsonl")}
    for r in rows:
        schema_validate(r)
        assert canonical(expected_target(r,oracle))==canonical(r["verified_target"]),"released target mismatch"
        from oracles.metadata_independent import check as metadata_check
        metadata_failures=metadata_check(r)
        if memory_control_rows([r]):
            from oracles.memory_controls_independent import check as memory_check
            metadata_failures.extend(memory_check(r))
        if belief_control_rows([r]):
            metadata_failures.extend(belief_checker([r]).check(r))
        if planning_control_rows([r],manifest['batch']):
            from oracles.rgm_planning_controls_independent import check as planning_check
            metadata_failures.extend(planning_check(r))
        if metadata_failures:
            from factory.catalog import catalog
            active,_=catalog()
            assert r['record_hash'] not in {x['record_hash'] for x in active},'Factual metadata defect remains active'
        for receipt_id in r["review_receipt_ids"]:
            receipt=receipt_by[receipt_id]
            assert receipt["record_hash"]==r["reviewed_candidate_hash"] and receipt["decision"]=="ACCEPT"
        receipts=[receipt_by[x] for x in r['review_receipt_ids']]
        assert len(receipts)==3 and len(set(r['review_receipt_ids']))==3
        assert {x['reviewer_execution_id'] for x in receipts}==set(APPROVAL_ACTORS.values()),'Released approval identities are not independent'
        assert not any(x.get('critical_defects') or any(i.get('severity') in ['high','critical'] for i in x.get('issues',[])) for x in receipts),'Serious release issue unresolved'
        candidate=dict(r);candidate.pop("reviewed_candidate_hash");candidate["review_status"]="PENDING";candidate["review_receipt_ids"]=[]
        assert record_hash(candidate)==r["reviewed_candidate_hash"],"candidate-to-release lineage mismatch"
    for split in ["train","dev_challenge"]:
        export=load_jsonl(path/f"{split}-sft.jsonl")
        expected=[r for r in rows if r["split"]==split]
        assert len(export)==len(expected)
        for item,r in zip(export,expected):
            assert canonical(item)==canonical({"messages":[{"role":"user","content":r["rendered_input"]},{"role":"assistant","content":canonical(r["verified_target"])}]})
        structured=load_jsonl(path/f'{split}-structured.jsonl')
        assert canonical(structured)==canonical([{'input':r['rendered_input'],'target':r['verified_target']} for r in expected]),'structured export mismatch'
    from factory.export_validation import validate_specialized
    specialized_counts=validate_specialized(path,rows)
    controls=memory_control_report(rows)
    assert not controls or not controls['errors'],'Released memory family controls incomplete or false'
    belief_controls=belief_control_report(rows)
    assert not belief_controls or not belief_controls['errors'],'Released belief family controls incomplete or false'
    planning_controls=planning_control_report(rows,manifest['batch'],ROOT)
    assert not planning_controls or not planning_controls['errors'],'Released planning family controls incomplete or false'
    return {"path":str(path.relative_to(ROOT)),"records":len(rows),"checksum_files":len(files),"specialized_exports":specialized_counts,"pass":True}


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["audit","release"]);p.add_argument("--batch",default="pilot-v2");p.add_argument("--version",default="v0001");p.add_argument("--tracks",nargs="+")
    a=p.parse_args();print(json.dumps(audit(a.batch) if a.command=="audit" else release(a.batch,a.version,a.tracks),indent=2))


if __name__=="__main__":main()
