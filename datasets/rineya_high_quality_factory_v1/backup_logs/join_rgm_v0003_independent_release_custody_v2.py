#!/usr/bin/env python3
"""New read-only local preflight and fail-closed proof join for commit 82787689."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
FACTORY=Path(__file__).resolve().parent.parent
LOGS=FACTORY/'backup_logs'
ORIGINAL=FACTORY.parent/'rineya-world-model-research'
WORKTREE=FACTORY.parent/'.rineya-dataset-git-worktree'
AUTHOR='/root/backup_manager'
COMMIT='82787689fef4e50f7a35cfee505c290b2f6a6d78'
PARENT='bc5198e11273cb6ea69659b6d3a8b5dc312b6f00'
SPEC='backup_logs/rgm-v0003-independent-release_SPEC.json'
SPEC_SHA='caf6e0e4d0a0569206cec8c725130cbb4150160eb79f74be8113f18f8d496708'
STAGE='backup_logs/rgm-v0003-independent-release_STAGE.json'
STAGE_SHA='9ddf03eecbcc13d603659b951f6404b267ce7795bac392ebab22004af840f97d'
FULL='backup_logs/rgm-v0003-independent-release_RECOVERY_VERIFIED_RECEIPT.json'
GZIP='quality_reports/RGM_V0003_REMOTE_LOSSLESS.json'
GZIP_LEDGER='backup_logs/rgm-v0003-independent-release_REMOTE_LOSSLESS_LEDGER.jsonl'
FREEZE='backup_logs/RGM_V0003_JOIN_CUSTODY_V2_SOURCE_FREEZE.json'
DEPENDENCIES={
 'backup_logs/run_provisional_archival_checkpoint_v2.py':'df03bdebc7d9cb6b24c65f82f1e17b42e8f857a106fafc3bfd2ac3ac341a0897',
 'backup_logs/recover_remote_checkpoint.py':'3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1',
 'backup_logs/verify_archival_remote_lossless.py':'7bab2c46dd1e7a52520a90617a2d910b5d6987bad8ec2489b05097045505a0aa',
 'backup_logs/ARCHIVAL_REMOTE_LOSSLESS_SOURCE_FREEZE.json':'ad9e0d0c8b27df84f2e113f4fe2777c2fc48ec3958e6cf2789eaebe85c85fa29',
 'backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json':'70d4dbd4441cc35df22920a428e804e87af3d0b3bd34814b4760fa025b6befa0',
 'backup_logs/checkpoint_backup.py':'07ef424f92d903c265c6cc06622733dcdeabf5bd03d3c13d98f793c2ddcc2d4b',
}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value,message):
    if not value:raise ValueError(message)


def sha64(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit lowercase SHA256 required')
    return value


def validate_stage(spec,stage,commit,parent,spec_sha):
    files=spec['files'];names={x['path'] for x in files}
    require(spec['validation_status']=='passed' and len(names)==len(files),'Unvalidated or duplicate exact inventory')
    require(stage['status']=='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','Stage is not a completed push')
    require(stage['commit_sha']==stage['remote_commit_sha']==commit and stage['expected_parent_sha']==parent,'Stage commit/sole-parent binding differs')
    require(stage['spec_sha256']==spec_sha and stage['files']==len(files) and stage['total_bytes']==sum(x['size'] for x in files),'Stage frozen inventory differs')
    require(stage['strict_diff_check']=='PASS' and stage['all_source_target_index_sha256_and_sha1_verified'] is True and stage['prior_release_guard']=='PASS' and stage['protected_original_baselines_verified'] is True,'Missing successful stage safeguards')
    require(stage['runner_execution']=='/root' and stage['grants_data_approval'] is False,'Actual stage role/approval boundary differs')
    require(stage['repository']=='LUNCY579/rineya-world-model-research' and stage['branch']=='research/rineya-high-quality-dataset-v1','Stage repository/branch differs')


def validate_proofs(spec,stage,full,gzip,commit,parent,spec_sha,stage_sha,full_sha,tree_sha):
    validate_stage(spec,stage,commit,parent,spec_sha)
    files={x['path']:x for x in spec['files']}
    require(full['status']=='BACKED_UP' and full['commit_sha']==full['remote_commit_sha']==commit,'Completed actual full BACKED_UP proof required')
    require(full['spec_sha256']==spec_sha and full['inventory']==spec['files'] and full['files']==len(files) and full['total_bytes']==sum(x['size'] for x in files.values()),'Full exact inventory differs')
    require(full['verification_source_sha256']==DEPENDENCIES['backup_logs/recover_remote_checkpoint.py'] and full['runner_execution']=='/root','Full verifier source/runner mismatch')
    require(full['repository']=='LUNCY579/rineya-world-model-research' and full['branch']=='research/rineya-high-quality-dataset-v1','Full proof repository/branch differs')
    private=full['private_repository_evidence']
    require(private['isPrivate'] is True and private['viewerPermission']=='ADMIN' and private['nameWithOwner']==full['repository'],'Full proof private ADMIN differs')
    require(not full['current_source_evolution'] and not full['current_source_evolution_at_finish'],'Unexpected evolved inventoried sources')
    remote=full['remote_verification'];blobs=remote['blob_receipts']
    require(remote['verified_files']==len(files) and remote['verified_unique_blobs']==len(blobs),'Full actual fetch count differs')
    require(len({b['git_blob'] for b in blobs})==len(blobs),'Duplicate actual unique blob proofs')
    require(remote['remote_tree_sha']==tree_sha,'Full actual remote tree differs')
    require(remote['http_accept']=='application/vnd.github.raw+json' and remote['cache_policy']=='no gh --cache; Cache-Control: no-cache; every blob fetched this run','Full actual raw/no-cache transport differs')
    coverage={}
    for blob in blobs:
        require(blob['verification_source']=='actual_remote_raw_GitHub_blob','Cached/local-only blob evidence rejected')
        for name in blob['paths']:
            require(name in files and name not in coverage,'Missing/duplicate/unrelated actual remote path')
            require((blob['size'],blob['sha256'])==(files[name]['size'],files[name]['sha256']),'Actual remote bytes differ from frozen spec')
            coverage[name]=blob
    require(set(coverage)==set(files),'Incomplete actual remote path coverage')
    require(gzip['status']=='PASS_ACTUAL_REMOTE_GZIP_DECODE' and gzip['commit_sha']==commit and gzip['spec_sha256']==spec_sha,'Completed fresh actual gzip proof required')
    require(gzip['stage_receipt_sha256']==stage_sha and gzip['verified_inventory_receipt_sha256']==full_sha,'Gzip receipt SHA bindings differ')
    require(gzip['source_sha256']==DEPENDENCIES['backup_logs/verify_archival_remote_lossless.py'] and gzip['source_freeze_sha256']==DEPENDENCIES['backup_logs/ARCHIVAL_REMOTE_LOSSLESS_SOURCE_FREEZE.json'] and gzip['dependency_sha256']==DEPENDENCIES['backup_logs/recover_remote_checkpoint.py'],'Frozen gzip source dependencies differ')
    require(gzip['actual_runner_execution']==AUTHOR and gzip['grants_data_approval'] is False and gzip['private_admin_confirmed'] is True,'Actual gzip actor/custody boundary differs')
    require(gzip['actual_remote_tree_sha']==tree_sha,'Gzip actual remote tree differs')
    require(gzip['stream_chunk_bytes']==65536 and gzip['bounded_spool_memory_bytes']==8388608,'Frozen bounded gzip execution settings differ')
    raw={x['path']:x for x in spec['raw_size_exclusions']};proofs=gzip['actual_remote_gzip_proofs']
    require(len(raw)==len(spec['raw_size_exclusions'])==24 and len(proofs)==24,'Every frozen oversized raw artifact requires exactly one proof')
    require(gzip['actual_remote_gzip_fetches']==gzip['oversized_raw_artifacts']==24,'24 fresh actual gzip fetches required')
    require(gzip['raw_bytes_exactly_recovered']==1188634377==sum(x['size'] for x in raw.values()),'Raw recovered byte total differs')
    require(gzip['actual_decoded_bytes_including_multiple_representations']==1188634377,'Decoded byte total differs')
    require({p['raw_path'] for p in proofs}==set(raw),'Incomplete/duplicate raw gzip coverage')
    for proof in proofs:
        require(proof['actual_remote_fetch_this_run'] is True and proof['decoded_byte_for_byte_equal_to_preserved_raw'] is True,'Missing new actual remote byte-for-byte proof')
        entry=raw[proof['raw_path']]
        require((proof['raw_bytes'],proof['raw_sha256'])==(entry['size'],entry['sha256']),'Frozen raw byte binding differs')
        allowed={r['gzip']['path'] for r in entry['lossless_representations']}
        require(proof['remote_gzip_path'] in allowed and proof['remote_gzip_path'] in coverage,'Gzip not an inventoried exact representation')
        blob=coverage[proof['remote_gzip_path']]
        require((proof['remote_git_blob'],proof['remote_gzip_bytes'],proof['remote_gzip_sha256'])==(blob['git_blob'],blob['size'],blob['sha256']),'Full/gzip actual blob binding differs')
    local_raw=full['oversized_raw_local_verification']
    require(len(local_raw)==24 and {p['path'] for p in local_raw}==set(raw),'Full local raw verification missing')
    for entry in local_raw:
        require(entry['lossless_local_roundtrip'] is True and (entry['size'],entry['sha256'])==(raw[entry['path']]['size'],raw[entry['path']]['sha256']),'Full local raw verification differs')
    return coverage


def validate_ledgers(full,gzip,full_lines,gzip_lines):
    actual=[x for x in full_lines if x['phase']=='actual_remote_blob_verified']
    blobs=full['remote_verification']['blob_receipts']
    require(len(actual)==len(blobs) and full_lines[-1]['phase']=='verification_complete' and full_lines[-1]['status']=='BACKED_UP','Incomplete actual full execution ledger')
    require({(x['git_blob'],x['size'],x['sha256']) for x in actual}=={(x['git_blob'],x['size'],x['sha256']) for x in blobs},'Full actual ledger/receipt mismatch')
    decoded=[x for x in gzip_lines if x['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF']
    require(len(decoded)==24 and gzip_lines[-1]['phase']=='COMPLETE_ACTUAL_REMOTE_GZIP_PROOF' and gzip_lines[-1]['status']=='PASS_ACTUAL_REMOTE_GZIP_DECODE','Incomplete actual gzip execution ledger')
    for key in ('raw_path','raw_bytes','raw_sha256','remote_gzip_path','remote_gzip_bytes','remote_gzip_sha256','remote_git_blob'):
        require({x[key] for x in decoded}=={x[key] for x in gzip['actual_remote_gzip_proofs']},'Gzip actual ledger/receipt mismatch: '+key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','join'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--handoff',type=Path)
    parser.add_argument('--expected-full-proof-sha256',type=sha64)
    parser.add_argument('--expected-gzip-proof-sha256',type=sha64)
    parser.add_argument('--expected-source-freeze-sha256',type=sha64,required=True)
    args=parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink() and args.output.resolve().is_relative_to(LOGS),'Fresh owned receipt in backup_logs required')
    if args.phase=='join':
        require(args.expected_full_proof_sha256 and args.expected_gzip_proof_sha256 and args.handoff,'Explicit completed proof hashes and fresh handoff required')
        require(not args.handoff.exists() and not args.handoff.is_symlink() and args.handoff.resolve().is_relative_to(FACTORY/'handoff'),'Fresh owned handoff required')
    source={'path':Path(__file__).relative_to(FACTORY).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    freeze_path=FACTORY/FREEZE
    require(hashlib.sha256(freeze_path.read_bytes()).hexdigest()==args.expected_source_freeze_sha256,'Join source freeze SHA differs')
    freeze=json.loads(freeze_path.read_bytes());require(freeze['source']['sha256']==source['sha256'] and freeze['author_execution']==AUTHOR,'Join source was not frozen before execution')
    for name,sha in DEPENDENCIES.items():require(hashlib.sha256((FACTORY/name).read_bytes()).hexdigest()==sha,'Frozen dependency changed: '+name)
    sp=importlib.util.spec_from_file_location('frozen_v2_runner_for_join',FACTORY/'backup_logs/run_provisional_archival_checkpoint_v2.py')
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    sp=importlib.util.spec_from_file_location('frozen_remote_stream_for_join',FACTORY/'backup_logs/recover_remote_checkpoint.py')
    h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
    def artifact(name,sha=None):
        proof=m.proof(m.safe(FACTORY,name),False)
        require(sha is None or sha==proof['sha256'],'Exact evidence checksum differs: '+name)
        return {'path':name,'size':proof['size'],'sha256':proof['sha256']}
    spec_artifact=artifact(SPEC,SPEC_SHA);stage_artifact=artifact(STAGE,STAGE_SHA)
    spec=json.loads((FACTORY/SPEC).read_bytes());stage=json.loads((FACTORY/STAGE).read_bytes())
    validate_stage(spec,stage,COMMIT,PARENT,SPEC_SHA)
    require(stage['source_sha256']==DEPENDENCIES['backup_logs/run_provisional_archival_checkpoint_v2.py'] and stage['source_freeze_sha256']=='e182c1fbd65f5d83edcf288555bf44a2b2b67f3264d2c1073ef086166c4444fc','Actual v2 stage source/freeze differs')
    require(len(spec['files'])==2303 and sum(x['size'] for x in spec['files'])==928914914,'Checkpoint-specific frozen scope differs')
    previous=json.loads((FACTORY/'backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json').read_bytes())
    m.original_custody(ORIGINAL,previous);private=m.dedicated_custody(WORKTREE,COMMIT)
    checked={}
    for entry in spec['files']:
        actual=m.proof(m.safe(FACTORY,entry['path']),False)
        require(m.equal(entry,actual),'Inventoried current source changed: '+entry['path'])
        require(m.proof(m.safe(WORKTREE,m.PREFIX+'/'+entry['path']),False)==actual,'Inventoried target bytes changed')
        checked[entry['path']]=actual
    m.verify_committed_parent_tree(WORKTREE,COMMIT,PARENT,checked)
    require(m.index(WORKTREE)=={n:v['git_blob'] for n,v in checked.items()},'Exact index blob IDs differ')
    groups={}
    for name,value in checked.items():groups.setdefault(value['git_blob'],[]).append(name)
    require(len(groups)==1895,'Expected actual unique local blobs differ')
    for blob,names in groups.items():
        expected=checked[names[0]]
        actual=h.process_stream(['git','cat-file','blob',blob],lambda stream:m.stream_proof(stream,expected['size'],False),WORKTREE)
        require(actual==expected,'Actual local Git index/commit object bytes differ')
    prior=m.tree(WORKTREE,PARENT);m.guard_previous(prior,checked)
    require(len(prior)==2095,'Previous exact parent path count differs')
    releases=[n for n in checked if '/releases/' in '/'+n];require(len(releases)==210,'Expected all210 releases absent')
    require(m.git(WORKTREE,'diff','--check',PARENT,COMMIT)==b'','Strict actual commit diff check failed')
    changed=[n.decode() for n in m.git(WORKTREE,'diff','--name-only','-z',PARENT,COMMIT).split(b'\0') if n]
    require(all(n.startswith(m.PREFIX+'/') and n[len(m.PREFIX)+1:] in checked for n in changed),'Unrelated commit path changes')
    remote=json.loads(m.run(['gh','api','repos/'+m.REPOSITORY+'/git/commits/'+COMMIT]));tree_sha=remote['tree']['sha']
    require([p['sha'] for p in remote['parents']]==[PARENT],'Actual remote sole parent differs')
    report={'status':'INDEPENDENT_LOCAL_PUSHED_CUSTODY_AUDIT_PASS_FULL_REMOTE_GZIP_JOIN_PENDING','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'actual_copy_stage_commit_push_runner':'/root','at_utc':now(),'commit_sha':COMMIT,'sole_parent_sha':PARENT,'spec':spec_artifact,'stage':stage_artifact,'frozen_files':2303,'frozen_bytes':928914914,'actual_local_git_unique_blobs_verified':1895,'source_target_index_commit_sha256_git_sha1_match':True,'all_parent2095_paths_preserved':True,'immutable_release_files_actual_bytes_verified':210,'strict_diff_check':'PASS','exact_scoped_commit_changes':len(changed),'private_admin':private,'actual_remote_tree_metadata_sha':tree_sha,'original_main_clean':True,'dedicated_clean':True,'six_protected_git_baselines':m.normalize_baselines(ORIGINAL,previous),'source':source,'source_freeze_sha256':args.expected_source_freeze_sha256,'dependency_checksums':DEPENDENCIES,'model_disclosure':'Inherited same model; separate execution does not imply model diversity','actual_remote_payload_fetches_by_this_preflight_or_join':0,'grants_data_approval':False,'disk_free_bytes':shutil.disk_usage(WORKTREE).free}
    evidence=[]
    if args.phase=='join':
        full_artifact=artifact(FULL,args.expected_full_proof_sha256);gzip_artifact=artifact(GZIP,args.expected_gzip_proof_sha256)
        full=json.loads((FACTORY/FULL).read_bytes());gzip=json.loads((FACTORY/GZIP).read_bytes())
        coverage=validate_proofs(spec,stage,full,gzip,COMMIT,PARENT,SPEC_SHA,STAGE_SHA,args.expected_full_proof_sha256,tree_sha)
        require(all(coverage[n]['git_blob']==v['git_blob'] for n,v in checked.items()),'Full actual remote/local commit blob bindings differ')
        require(full['remote_verification']['verified_unique_blobs']==1895,'Full actual unique fetch count differs')
        full_ledger=m.safe(FACTORY,Path(full['ledger']).relative_to(FACTORY).as_posix())
        gzip_ledger=m.safe(FACTORY,GZIP_LEDGER)
        full_ledger_artifact=artifact(full_ledger.relative_to(FACTORY).as_posix());gzip_ledger_artifact=artifact(GZIP_LEDGER)
        with full_ledger.open() as stream:full_lines=[json.loads(line) for line in stream]
        with gzip_ledger.open() as stream:gzip_lines=[json.loads(line) for line in stream]
        validate_ledgers(full,gzip,full_lines,gzip_lines)
        evidence=[full_artifact,gzip_artifact,full_ledger_artifact,gzip_ledger_artifact]
        report.update(status='BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED',actual_full_inventory_proof=full_artifact,actual_full_inventory_runner='/root',actual_full_unique_remote_fetches=1895,actual_full_ledger=full_ledger_artifact,actual_new_gzip_proof=gzip_artifact,actual_new_gzip_runner=AUTHOR,actual_new_gzip_payload_fetches=24,raw_bytes_exactly_recovered=1188634377,actual_gzip_ledger=gzip_ledger_artifact)
    excluded={n for n in checked}|{e['path'] for e in spec['raw_size_exclusions']}
    report['postinventory_local_path_names_only']=sorted(p.relative_to(FACTORY).as_posix() for p in FACTORY.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store' and p.relative_to(FACTORY).as_posix() not in excluded)
    report['snapshot_deferrals_preserved']=spec['deferred_incomplete_new_v4_paths']
    report['postinventory_boundary']='Only2303 exact frozen files in82787689 are within backup claim. Spec, current stage/full/gzip proofs and ledgers, this helper/freezes/tests/preflight/join/handoff and official/adversarial FINALs or newborn prospective work excluded at scanstart remain local until a later exact inventory. No retroactive insertion; all previous receipt statuses remain unchanged historical evidence.'
    report['self_inventory_boundary']='This new output/handoff cannot be in an earlier frozen spec or self-hash its final bytes; include actual final bytes in a later checkpoint.'
    for item in [spec_artifact,stage_artifact,*evidence]:require(artifact(item['path'],item['sha256'])==item,'Evidence changed during preflight/join')
    m.original_custody(ORIGINAL,previous);m.dedicated_custody(WORKTREE,COMMIT)
    for name,sha in DEPENDENCIES.items():require(hashlib.sha256((FACTORY/name).read_bytes()).hexdigest()==sha,'Frozen dependency changed at finish: '+name)
    require(hashlib.sha256(freeze_path.read_bytes()).hexdigest()==args.expected_source_freeze_sha256,'Source freeze changed at finish')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source['sha256'],'Frozen join source changed')
    m.exclusive(args.output,report)
    output_artifact=artifact(args.output.relative_to(FACTORY).as_posix())
    if args.phase=='join':
        text=f'''# RGM v0003 archival backup completed\n\nCommit `{COMMIT}`, sole parent `{PARENT}`, preserves **2303 files /928,914,914 bytes** from spec `{SPEC_SHA}`. Root's completed actual remote proof `{args.expected_full_proof_sha256}` verified **1895 unique GitHub blobs**. Backup manager's separate actual gzip proof `{args.expected_gzip_proof_sha256}` fetched **24 new remote payloads**, decoded byte-for-byte to **1,188,634,377 preserved raw bytes**.\n\nJoined receipt `{output_artifact['path']}` SHA256 `{output_artifact['sha256']}`. Fresh read-only source/target/index/commit checks, private ADMIN, sole parent/tree/branch and six original Git baselines passed. All2095 parent paths and210 immutable release files remain preserved. Root ran copy/stage/push/full verification; `/root/backup_manager` authored the join and ran the separate gzip verification. Same inherited model; no model diversity or data-quality approval claim.\n\nOnly the frozen2303-file scope is claimed backed up. Postinventory proofs/ledgers/spec/this addendum and deferred or newborn BELIEF V4 official/adversarial/architect work remain local for the next checkpoint. Historical pending/failure receipts and exact frozen spec remain unchanged. No Git mutation by this join.\n'''
        with args.handoff.open('x') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':report['status'],'output':output_artifact,'handoff':artifact(args.handoff.relative_to(FACTORY).as_posix()) if args.phase=='join' else None}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        failure=LOGS/('RGM_V0003_JOIN_NEW_EXECUTION_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with failure.open('x') as stream:json.dump({'status':'FAILED_NEW_READ_ONLY_CUSTODY_EXECUTION_PRESERVED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'at_utc':now(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'error_type':type(error).__name__,'error':str(error),'no_git_mutation':True},stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        raise
