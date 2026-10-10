#!/usr/bin/env python3
"""Fresh read-only existing-checkpoint and completed-review custody preflight."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
FACTORY = Path(__file__).resolve().parent.parent
LOGS = FACTORY / 'backup_logs'
AUTHOR = '/root/backup_manager'
PARENT = 'bc5198e11273cb6ea69659b6d3a8b5dc312b6f00'
FREEZE = LOGS / 'RGM_RENDERER_R1_RELEASE_CHECKPOINT_PREFLIGHT_SOURCE_FREEZE.json'
OUTPUT = LOGS / 'RGM_RENDERER_R1_RELEASE_CHECKPOINT_PREFLIGHT.json'
FAILURE = LOGS / 'RGM_RENDERER_R1_RELEASE_CHECKPOINT_PREFLIGHT_EXECUTION_FAILURE.json'
RUNNER_SHA = 'df03bdebc7d9cb6b24c65f82f1e17b42e8f857a106fafc3bfd2ac3ac341a0897'
RECEIPTS = [
    'quality_reports/RGM_RENDERER_R1_FROZEN_REVIEW_EVIDENCE_JOIN.json',
    'reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json',
    'reviews/adjudication/planning-gap-v2-renderer-r1.actual-preexecution.json',
    'reviews/adjudication/planning-gap-v2-renderer-r1.execution.json',
    'reviews/expert/planning-gap-v2-renderer-r1.expert_input.freeze.json',
    'reviews/expert/planning-gap-v2-renderer-r1.expert_input_inspection.actual_final.freeze.json',
    'reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.freeze.json',
    'reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.execution.json',
    'reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.preexecution.json',
    'audits/execution/planning-gap-v2-renderer-r1-adversarial.json',
    'audits/execution/planning-gap-v2-renderer-r1-adversarial-comparison.preexec.json',
    'audits/execution/planning-gap-v2-renderer-r1-adversarial-input-freeze.json',
    'reviews/adversarial/planning-gap-v2-renderer-r1.input_freeze.json',
    'quality_reports/RGM_V2_RENDERER_R1_ROOT_SOURCE_FREEZE.json',
    'reviews/oracle/RGM-v2-renderer-public-assessment-receipt-20261010T062719Z.json',
]
MANUAL_BINDINGS = {
    'audits/execution/planning-gap-v2-renderer-r1-adversarial.json': {
        'source_sha256': 'audits/rgm_planning_gap_v2_renderer_r1_adversarial_comparison.py',
        'preexec_sha256': 'audits/execution/planning-gap-v2-renderer-r1-adversarial-comparison.preexec.json',
        'canonical_audit_sha256': 'audits/diversity/planning-gap-v2-renderer-r1-adversarial-canonical.json',
        'final_judgment_sha256': 'reviews/adversarial/planning-gap-v2-renderer-r1.jsonl',
        'first_canonical_judgment_sha256': 'reviews/adversarial/planning-gap-v2-renderer-r1.first.jsonl',
    },
    'audits/execution/planning-gap-v2-renderer-r1-adversarial-input-freeze.json': {
        'source_sha256': 'audits/rgm_planning_gap_v2_renderer_r1_adversarial_input.py',
        'preexec_sha256': 'audits/execution/planning-gap-v2-renderer-r1-adversarial-input.preexec.json',
        'fresh_view_freeze_sha256_before_after': 'reviews/input_only/planning-gap-v2-renderer-r1.freeze.json',
    },
    'reviews/adversarial/planning-gap-v2-renderer-r1.input_freeze.json': {
        'source_sha256': 'audits/rgm_planning_gap_v2_renderer_r1_adversarial_input_custody.py',
        'preexec_sha256': 'audits/execution/planning-gap-v2-renderer-r1-adversarial-input-custody.preexec.json',
    },
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    require(not OUTPUT.exists() and not FAILURE.exists(), 'Fresh output paths required')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    freeze = json.loads(FREEZE.read_bytes())
    require(freeze['source']['sha256'] == source_sha, 'Preflight source was not frozen before execution')
    runner = LOGS / 'run_provisional_archival_checkpoint_v2.py'
    require(hashlib.sha256(runner.read_bytes()).hexdigest() == RUNNER_SHA, 'Prepared v2 runner changed')
    sp = importlib.util.spec_from_file_location('frozen_prospective_archival_v2', runner)
    m = importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    original = FACTORY.parent / 'rineya-world-model-research'
    worktree = FACTORY.parent / '.rineya-dataset-git-worktree'
    def artifact(name):
        proof = m.proof(m.safe(FACTORY, name), False)
        return {'path': name, 'size': proof['size'], 'sha256': proof['sha256']}
    for item in freeze['frozen_inputs']:
        require(artifact(item['path']) == item, 'Frozen metadata input changed: ' + item['path'])
    joined = json.loads((LOGS/'RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json').read_bytes())
    m.original_custody(original, joined)
    private = m.dedicated_custody(worktree, PARENT)
    spec = json.loads((LOGS/'rgm-v2-renderer-review-milestone_SPEC.json').read_bytes())
    full = json.loads((LOGS/'rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json').read_bytes())
    expected = {x['path']: x for x in spec['files']}
    coverage = {path: b for b in full['remote_verification']['blob_receipts'] for path in b['paths']}
    tree = m.tree(worktree, PARENT)
    require(len(expected) == 2095 and sum(x['size'] for x in expected.values()) == 886553408, 'Prior frozen scope changed')
    require(set(tree) == set(expected) == set(coverage) and m.index(worktree) == tree, 'Prior exact paths/index changed')
    changed = []
    release_count = 0
    for name,item in expected.items():
        require(tree[name] == coverage[name]['git_blob'], 'Prior tree blob changed: ' + name)
        target = m.proof(m.safe(worktree,m.PREFIX+'/'+name), False)
        require(m.equal(item,target) and target['git_blob'] == tree[name], 'Prior checkout bytes differ: ' + name)
        current = artifact(name)
        if current != item:
            changed.append({'frozen':item, 'current':current, 'current_factory_is_not_claimed_in_old_checkpoint':True})
        if '/releases/' in '/'+name:
            release_count += 1
            require(current == item, 'Historical immutable local release changed: ' + name)
    require(release_count == 198, 'Historical release count changed')
    bindings = []
    metadata = {}
    def validate(path, sha, size=None, context=''):
        current = artifact(path)
        require(current['sha256'] == sha and (size is None or current['size'] == size), 'Review binding mismatch: ' + path)
        bindings.append({'receipt':context, **current})
    def walk(obj, context):
        if isinstance(obj, dict):
            if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
                validate(obj['path'],obj['sha256'],obj.get('bytes',obj.get('size')),context)
                for key in ('archive','preexecution_archive'):
                    if key in obj:
                        validate(obj[key],obj['sha256'],obj.get('bytes',obj.get('size')),context)
            for key,value in obj.items():
                if '/' in key and isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value):
                    validate(key,value,context=context)
                walk(value,context)
        elif isinstance(obj, list):
            if len(obj)==2 and isinstance(obj[0],str) and '/' in obj[0] and isinstance(obj[1],str) and len(obj[1])==64:
                validate(obj[0],obj[1],context=context)
            else:
                for item in obj:walk(item,context)
    for name in RECEIPTS:
        obj = json.loads(m.safe(FACTORY,name).read_bytes());metadata[name]=obj
        walk(obj,name)
        for field,path in MANUAL_BINDINGS.get(name,{}).items():
            validate(path,obj[field],context=name)
    expert = metadata['reviews/expert/planning-gap-v2-renderer-r1.expert_comparison.execution.json']
    adversarial = metadata['audits/execution/planning-gap-v2-renderer-r1-adversarial.json']
    adjudication = metadata['reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json']
    require(expert['actual_exit_code']==0 and expert['decision_counts']=={'ACCEPT':408}, 'Expert completion receipt mismatch')
    require(adversarial['completed'] is True and adversarial['decisions']=={'ACCEPT':408}, 'Adversarial completion receipt mismatch')
    require(adjudication['completed_execution'] is True and adjudication['actual_adjudication']['exit_code']==0 and adjudication['actual_decision_counts']=={'ACCEPT':408}, 'Architect completion receipt mismatch')
    require(adjudication['actual_execution_id']=='/root/research_architect', 'Actual adjudicator identity differs')
    readiness = json.loads((LOGS/'PROVISIONAL_ARCHIVAL_RUNNER_V2_READINESS.json').read_bytes())
    require(readiness['actual_tests_passed']==26 and readiness['source']['sha256']==RUNNER_SHA, 'Prepared runner readiness differs')
    core = json.loads(m.run(['gh','api','rate_limit']))['resources']['core']
    core = {key:core[key] for key in ('limit','remaining','used','reset')}
    completed_v4 = 'reviews/oracle/RBM-v4-prospective-20261010T071308Z/'
    formal_names=[];expert_names=[]
    for base in ('reviews/oracle','reviews/expert'):
        for p in sorted((FACTORY/base).rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            name=p.relative_to(FACTORY).as_posix()
            if '/RBM-v4-' in name:formal_names.append(name)
            elif '/belief-v4-prospective-' in name:expert_names.append(name)
    unfinished_formal=[n for n in formal_names if not n.startswith(completed_v4)]
    final_r1_names=sorted(p.relative_to(FACTORY).as_posix() for base in ('reviews','audits','quality_reports') for p in (FACTORY/base).rglob('*') if p.is_file() and not p.is_symlink() and ('renderer-r1' in p.name or 'renderer_r1' in p.name) and '__pycache__' not in p.parts)
    new_release = FACTORY/'data/RGM-001/releases/v0003'
    new_release_names=sorted(p.relative_to(FACTORY).as_posix() for p in new_release.rglob('*') if p.is_file()) if new_release.exists() else []
    m.original_custody(original,joined);m.dedicated_custody(worktree,PARENT)
    for item in freeze['frozen_inputs']:
        require(artifact(item['path']) == item, 'Frozen metadata changed during preflight')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_sha, 'Preflight source changed')
    report={
        'status':'PASS_READ_ONLY_PROSPECTIVE_RELEASE_CHECKPOINT_CUSTODY_PREFLIGHT',
        'author_execution':AUTHOR, 'actual_runner_execution':AUTHOR, 'finished_at_utc':now(),
        'model_disclosure':'Inherited same model; separate execution does not imply model diversity',
        'source':artifact(Path(__file__).relative_to(FACTORY).as_posix()), 'source_freeze':artifact(FREEZE.relative_to(FACTORY).as_posix()),
        'expected_dedicated_remote_parent':PARENT, 'private_admin':private,
        'original_main':{'head':m.ORIGINAL_HEAD,'branch':'main','clean':True,'six_protected_byte_existence_baselines_verified_before_after':m.normalize_baselines(original,joined)},
        'dedicated':{'head':PARENT,'branch':m.BRANCH,'clean':True,'exact_prior_tree_index_checkout_sha256_sha1_verified':True,'sparse_scope':['tools',m.PREFIX],'full_index':True},
        'prior_checkpoint':{'spec':artifact('backup_logs/rgm-v2-renderer-review-milestone_SPEC.json'),'actual_full_proof':artifact('backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json'),'actual_join':artifact('backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json'),'files':2095,'bytes':886553408,'all_prior_paths_present_in_original_factory':True,'all_prior_paths_preserved_in_dedicated_commit':True,'historical_release_files_actual_bytes_preserved':198},
        'current_factory_existing_path_changes_since_prior_spec':changed,
        'current_final_review_receipt_evidence':[artifact(name) for name in RECEIPTS],
        'current_final_review_exact_binding_checks':bindings,
        'completed_review_actor_receipt_counts':{'expert':{'actor':'/root/expert_reviewer','ACCEPT':408},'adversarial':{'actor':'/root/adversarial_auditor','ACCEPT':408},'adjudicator':{'actor':'/root/research_architect','ACCEPT':408}},
        'review_custody_interpretation':'Hashes and completion attestations verified; no semantic data-quality decision by backup manager. Prior original canonical exposure, common model/public design and same-task reuse remain disclosed in immutable reviewer receipts.',
        'historical_root_join_state':'Earlier frozen root review join states PENDING_ACTUAL_FRESH_DECISIONS; retained unchanged and now followed by separately completed architect execution receipt.',
        'new_v2_runner':artifact('backup_logs/run_provisional_archival_checkpoint_v2.py'),'new_v2_runner_readiness':artifact('backup_logs/PROVISIONAL_ARCHIVAL_RUNNER_V2_READINESS.json'),'source_frozen_26_tests_passed':True,
        'original_helper_bytes_preserved':[artifact('backup_logs/'+n) for n in ['run_provisional_archival_checkpoint.py','checkpoint_backup.py','recover_remote_checkpoint.py','verify_archival_remote_lossless.py']],
        'disk_free_bytes_at_finish':shutil.disk_usage(worktree).free,
        'disk_boundary':'Existing 886553408-byte custody revalidated; no new production inventory constructed. Future runner must enforce 2x actual new inventory +512 MiB reserve before copy.',
        'actual_api_core_counts_only':core,
        'new_rgm_v0003_path_names_observed_only':new_release_names,
        'next_checkpoint_must_include_all_completed_renderer_r1_artifacts_names_only':final_r1_names,
        'next_checkpoint_must_include_completed_belief_v4_preparation_names_only':[n for n in formal_names if n.startswith(completed_v4)],
        'unfinished_belief_v4_official_checker_names_only':unfinished_formal,
        'unfinished_belief_v4_expert_names_only':expert_names,
        'belief_v4_boundary':'Completed reviews/oracle/RBM-v4-prospective-20261010T071308Z/ preparation is included next time. Only unfinished NEW official-checker and reviews/expert/belief-v4-prospective-* work may be deferred by final explicit names after root confirms status; empty observed lists do not imply completion. No private expectation contents inspected or forwarded. Exact active formal prefix still pending root.',
        'future_deferral_rule':'Do not reuse stale renderer-r1 deferral patterns. All completed r1 reviews, sources, firsts, failures, freezes, adjudication, current bc519 proofs and new owned custody evidence belong in the future full inventory. No previously tracked path may be absent.',
        'production_inventory_created':False,'copy_stage_commit_push_executed':False,'grants_data_quality_or_release_approval':False,
        'concrete_custody_blockers':[], 'self_inventory_boundary':'This fresh source/freeze/receipt are post-bc519 local evidence until included in a later exact checkpoint. Prospective release/living report bytes are outside this preflight and are not claimed backed up.',
    }
    exclusive(OUTPUT,report)
    print(json.dumps({'status':report['status'],'receipt':artifact(OUTPUT.relative_to(FACTORY).as_posix()),'review_binding_checks':len(bindings),'current_prior_path_changes':len(changed),'historical_release_files':release_count,'disk_free_bytes':report['disk_free_bytes_at_finish'],'api_core':core}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if not FAILURE.exists():
            exclusive(FAILURE,{'status':'FAILED_READ_ONLY_PREFLIGHT_EVIDENCE_PRESERVED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'at_utc':now(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'error_type':type(error).__name__,'error':str(error),'no_copy_stage_commit_push':True})
        raise
