#!/usr/bin/env python3
"""NEW read-only exact r2 proof join; preserve five frozen nonzero whitespace warnings."""
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
COMMIT='4d0b5eff1af3184fedf7fa25ada729a6cc909268'
PARENT='82787689fef4e50f7a35cfee505c290b2f6a6d78'
SPEC='backup_logs/belief-v4-validated-code-candidate-r2_SPEC.json'
SPEC_SHA='6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5'
STAGE='backup_logs/belief-v4-validated-code-candidate-r2_RESUMED_STAGE.json'
STAGE_SHA='095e42ce7adf1df5afaba67a49500f9bf1dd880abbab6b7b6882d4b5483c41a1'
FULL='backup_logs/belief-v4-validated-code-candidate-r2_RECOVERY_VERIFIED_RECEIPT.json'
GZIP='backup_logs/BELIEF_V4_R2_REMOTE_LOSSLESS.json'
GZIP_LEDGER='backup_logs/belief-v4-validated-code-candidate-r2_REMOTE_LOSSLESS_LEDGER.jsonl'
FREEZE='backup_logs/BELIEF_V4_R2_JOIN_SOURCE_FREEZE.json'
SCOPE_CONFIG='backup_logs/BELIEF_V4_R2_JOIN_SCOPE_CONFIG.json'
RESUMPTION=None
DEPENDENCIES={
 'backup_logs/run_provisional_archival_checkpoint_v2.py':'df03bdebc7d9cb6b24c65f82f1e17b42e8f857a106fafc3bfd2ac3ac341a0897',
 'backup_logs/recover_remote_checkpoint.py':'3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1',
 'backup_logs/verify_archival_remote_lossless.py':'7bab2c46dd1e7a52520a90617a2d910b5d6987bad8ec2489b05097045505a0aa',
 'backup_logs/ARCHIVAL_REMOTE_LOSSLESS_SOURCE_FREEZE.json':'ad9e0d0c8b27df84f2e113f4fe2777c2fc48ec3958e6cf2789eaebe85c85fa29',
 'backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json':'3ec1e2f75e42e827eff483fff190f349ceedee2d6b20dcc4619a315f16fa0bb1',
 'backup_logs/join_rgm_v0003_independent_release_custody_v2.py':'11326dcb65fa8421e601b757780e5d0e8df672027f094e98a4b4994a99f280d4',
 'backup_logs/resume_belief_v4_r2_frozen_stage.py':'8dd342c2394b125b2284f3975556db49735b50a150e58f5c0909a6acb7e7ac92',
 'backup_logs/BELIEF_V4_R2_RESUMPTION_SOURCE_FREEZE.json':'c8e8356e8f2feb24060fa04ccae2a5aca1d696040d14ca001742268a02762158',
 'backup_logs/BELIEF_V4_R2_FROZEN_WHITESPACE_EXCEPTION.json':'71abc77279f45a864b7d493840d07cd9eb4515076521e850bdd3a5d8209640f4',
 'backup_logs/BELIEF_V4_CODE_CANDIDATE_CUSTODY_ADAPTER.json':'2e604f1dc2dc36c29ccc7968988a6abdc08a71c63e8ebf596b39faf856fb1e23',
 'backup_logs/belief-v4-validated-code-candidate-r2_STAGE.json':'37cb266ec809b27c42ec3a4b992ef66cf992cbdf63d4d0156ae5603371c28436',
 'backup_logs/belief-v4-validated-code-candidate-r2_STAGE_LEDGER.jsonl':'d4d3beef10861e4a50c5611ac1deba227dd0577a8023c52cf819c4f814985e21',
 'backup_logs/belief-v4-validated-code-candidate-r2_RESUMED_STAGE_LEDGER.jsonl':'3858f9cf2b532c96a520a92dcdfcab0722605ebe39bdaa2a93f3dc84cb821e3a',
 'backup_logs/belief-v4-validated-code-candidate-r2_RESUMED_LOCAL_COMMIT.json':'22781e1fb4e7a73bd28d5c98505e13d2154b42c36cc5fc37e8defcd4859c1d90',
 'backup_logs/BELIEF_V4_R2_JOIN_SCOPE_CONFIG.json':'d709258ba61716aa8ca60e2fcc7c43d1e158d86c00e74d410819e4ab667bb9c4',

 'backup_logs/checkpoint_backup.py':'07ef424f92d903c265c6cc06622733dcdeabf5bd03d3c13d98f793c2ddcc2d4b',
}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value,message):
    if not value:raise ValueError(message)


def sha64(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Explicit lowercase SHA256 required')
    return value


def validate_stage(spec,stage,commit,parent,spec_sha):
    require(RESUMPTION is not None,'Frozen resumption guard is not loaded')
    RESUMPTION.validate_spec(spec)
    files=spec['files'];names={x['path'] for x in files}
    require(len(names)==len(files) and spec['validation_status']=='passed','Duplicate/unvalidated exact inventory')
    require(stage['status']=='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','Actual completed push required')
    require(stage['commit_sha']==stage['remote_commit_sha']==commit==COMMIT and stage['expected_parent_sha']==parent==PARENT,'Actual stage commit/parent differs')
    require(stage['spec_sha256']==spec_sha==SPEC_SHA and stage['files']==2685 and stage['total_bytes']==1080978676,'Actual stage frozen scope differs')
    require(stage['unique_actual_index_blob_groups']==2153 and stage['preserved_raw_bytes']==1353229114,'Actual local blob/raw totals differ')
    for key in ('full_index_exact_scope_verified','outside_dataset_parent_index_unchanged','prior2303_paths_preserved','immutable210_release_bytes_preserved','protected_original_six_baselines_verified','all_source_target_index_sha256_and_sha1_verified'):
        require(stage.get(key) is True,'Missing actual stage safeguard: '+key)
    require(stage['runner_execution']=='/root' and stage['author_execution']==AUTHOR and stage['grants_data_approval'] is False,'Actual stage roles/data boundary differ')
    require(stage['source_sha256']==DEPENDENCIES['backup_logs/resume_belief_v4_r2_frozen_stage.py'] and stage['source_freeze_sha256']==DEPENDENCIES['backup_logs/BELIEF_V4_R2_RESUMPTION_SOURCE_FREEZE.json'],'Actual frozen resumption source differs')
    require(stage['exception_configuration_sha256']==DEPENDENCIES['backup_logs/BELIEF_V4_R2_FROZEN_WHITESPACE_EXCEPTION.json'] and stage['prior_failed_stage_sha256']==DEPENDENCIES['backup_logs/belief-v4-validated-code-candidate-r2_STAGE.json'] and stage['prior_failed_ledger_sha256']==DEPENDENCIES['backup_logs/belief-v4-validated-code-candidate-r2_STAGE_LEDGER.jsonl'],'Actual preserved failure/exception binding differs')
    require(stage['custody_adapter_sha256']==DEPENDENCIES['backup_logs/BELIEF_V4_CODE_CANDIDATE_CUSTODY_ADAPTER.json'] and stage['immutable_df03_helper_sha256']==DEPENDENCIES['backup_logs/run_provisional_archival_checkpoint_v2.py'],'Actual df03/custody binding differs')
    expected=RESUMPTION.validate_diagnostics({'returncode':2,'stdout':RESUMPTION.expected_diagnostics(RESUMPTION.canonical_exception()),'stderr':b''},RESUMPTION.canonical_exception())
    require(stage['actual_diff_check']==expected and stage['precommit_actual_diff_check']==expected,'Untruthful/missing exact nonzero five-warning exception')
    require('strict_diff_check' not in stage or stage['strict_diff_check']!='PASS','Unqualified strict-diff PASS is forbidden')
    staged=stage['staged_paths'];require(len(staged)==393 and len(set(staged))==393 and all(p.startswith(RESUMPTION.PREFIX+'/') and p[len(RESUMPTION.PREFIX)+1:] in names for p in staged),'Actual staged exact scope differs')
    restricted={'other_staged_paths':388,'restricted_strict_diff_check':'PASS'}
    require(stage['restricted_other_paths_check']==restricted and stage['precommit_restricted_other_paths_check']==restricted,'Actual other388 restricted checks differ')
    for key in ('private_repository_evidence','actual_prepush_private_repository_evidence'):
        require(stage[key]=={'isPrivate':True,'nameWithOwner':'LUNCY579/rineya-world-model-research','viewerPermission':'ADMIN'},'Actual private ADMIN stage evidence differs')
    raw={x['path']:x for x in spec['raw_size_exclusions']};local=stage['raw_lossless_proofs']
    require(len(local)==26 and len({x['path'] for x in local})==26 and {x['path'] for x in local}==set(raw),'Stage actual raw coverage differs')
    for x in local:
        require(x['secret_scan']=='PASS' and x['byte_for_byte_lossless'] is True and (x['size'],x['sha256'])==(raw[x['path']]['size'],raw[x['path']]['sha256']),'Stage actual raw scan/roundtrip differs')


def validate_stage_ledger(spec,stage,records,local_commit):
    require(records and records[-1]['phase']=='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING' and records[-1]['commit_sha']==COMMIT,'Completed actual stage ledger required')
    files={e['path']:e for e in spec['files']}
    sources=[e for e in records if e['phase']=='source_target_reverified_no_copy']
    blobs=[e for e in records if e['phase']=='actual_index_blob_reverified']
    require(len(sources)==2685 and len({e['path'] for e in sources})==2685 and {e['path'] for e in sources}==set(files),'Actual stage source/target ledger coverage differs')
    coverage={};seen_blobs=set()
    for e in blobs:
        require(type(e['paths']) is list and e['paths'] and len(set(e['paths']))==len(e['paths']),'Empty/duplicate stage blob paths')
        require(e['git_blob'] not in seen_blobs,'Duplicated stage blob ledger group');seen_blobs.add(e['git_blob'])
        for name in e['paths']:
            require(name in files and name not in coverage and (e['size'],e['sha256'])==(files[name]['size'],files[name]['sha256']),'Actual stage index ledger bytes/path coverage differs')
            coverage[name]=e
    require(len(blobs)==2153 and set(coverage)==set(files),'Actual2153 stage index groups required')
    for e in sources:
        require((e['size'],e['sha256'],e['git_blob'])==(coverage[e['path']]['size'],coverage[e['path']]['sha256'],coverage[e['path']]['git_blob']),'Actual stage source/target/index cross-join differs')
    warning=stage['exception_actual_five_source_target_index_proofs']
    require(len(warning)==5 and [e['path'] for e in warning]==list(RESUMPTION.WARNING_PATHS),'Actual five source payload coverage differs')
    for e in warning:
        require(e['size']==20749 and e['sha256']==RESUMPTION.TEST_SHA and e['actual_line212_base64']==RESUMPTION.LINE_B64 and e['git_blob']==coverage[e['path']]['git_blob'],'Actual five exception source/index bytes differ')
    require(local_commit['status']=='LOCAL_COMMIT_PRESERVED' and local_commit['commit_sha']==COMMIT and local_commit['spec_sha256']==SPEC_SHA and local_commit['expected_parent_sha']==PARENT,'Immutable local commit receipt differs')
    for key in ('source_sha256','source_freeze_sha256','files','total_bytes','unique_actual_index_blob_groups','raw_lossless_proofs','all_source_target_index_sha256_and_sha1_verified','actual_diff_check','precommit_actual_diff_check'):
        require(local_commit[key]==stage[key],'Local commit/stage safeguard differs: '+key)
    return coverage


def validate_proofs(spec,stage,full,gzip,commit,parent,spec_sha,stage_sha,full_sha,tree_sha):
    validate_stage(spec,stage,commit,parent,spec_sha)
    files={x['path']:x for x in spec['files']}
    require(full['status']=='BACKED_UP' and full['commit_sha']==full['remote_commit_sha']==commit,'Completed actual full BACKED_UP proof required')
    require(full['spec_sha256']==spec_sha and full['inventory']==spec['files'] and full['files']==len(files) and full['total_bytes']==sum(x['size'] for x in files.values()),'Full exact inventory differs')
    require(full['verification_source_sha256']==DEPENDENCIES['backup_logs/recover_remote_checkpoint.py'] and full['runner_execution']=='/root' and full['local_git_verified_unique_blobs']==2153 and full['current_source_matching_files']==2685,'Full verifier source/runner/local blob count mismatch')
    require(any(e['sha256']==stage_sha and Path(e['path']).name==Path(STAGE).name for e in full['prior_evidence']),'Full actual proof must bind the new actual resumption stage')
    require(full['repository']=='LUNCY579/rineya-world-model-research' and full['branch']=='research/rineya-high-quality-dataset-v1','Full proof repository/branch differs')
    private=full['private_repository_evidence']
    require(private['isPrivate'] is True and private['viewerPermission']=='ADMIN' and private['nameWithOwner']==full['repository'],'Full proof private ADMIN differs')
    require(not full['current_source_evolution'] and not full['current_source_evolution_at_finish'],'Unexpected evolved inventoried sources')
    remote=full['remote_verification'];blobs=remote['blob_receipts']
    require(remote['verified_files']==2685 and remote['verified_unique_blobs']==len(blobs)==2153,'Full actual fresh2153 fetch count differs')
    require(len({b['git_blob'] for b in blobs})==len(blobs),'Duplicate actual unique blob proofs')
    require(remote['remote_tree_sha']==tree_sha,'Full actual remote tree differs')
    require(remote['http_accept']=='application/vnd.github.raw+json' and remote['cache_policy']=='no gh --cache; Cache-Control: no-cache; every blob fetched this run','Full actual raw/no-cache transport differs')
    coverage={}
    for blob in blobs:
        require(type(blob['paths']) is list and blob['paths'] and len(set(blob['paths']))==len(blob['paths']),'Empty/duplicate remote blob paths')
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
    require(len(raw)==len(spec['raw_size_exclusions'])==26 and len(proofs)==26,'Every frozen oversized raw artifact requires exactly one proof')
    require(gzip['actual_remote_gzip_fetches']==gzip['oversized_raw_artifacts']==26,'26 fresh actual gzip fetches required')
    require(gzip['raw_bytes_exactly_recovered']==1353229114==sum(x['size'] for x in raw.values()),'Raw recovered byte total differs')
    require(gzip['actual_decoded_bytes_including_multiple_representations']==1353229114,'Decoded byte total differs')
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
    require(len(local_raw)==26 and {p['path'] for p in local_raw}==set(raw),'Full local raw verification missing')
    for entry in local_raw:
        require(entry['lossless_local_roundtrip'] is True and (entry['size'],entry['sha256'])==(raw[entry['path']]['size'],raw[entry['path']]['sha256']),'Full local raw verification differs')
    return coverage


def validate_local_remote_blob_coverage(coverage,checked):
    require(set(coverage)==set(checked),'Remote/local exact coverage differs')
    require(all(coverage[n]['git_blob']==v['git_blob'] and (coverage[n]['size'],coverage[n]['sha256'])==(v['size'],v['sha256']) for n,v in checked.items()),'Full actual remote/local commit byte and blob bindings differ')


def validate_ledgers(full,gzip,full_lines,gzip_lines):
    actual=[x for x in full_lines if x['phase']=='actual_remote_blob_verified']
    blobs=full['remote_verification']['blob_receipts']
    require(len(actual)==len(blobs)==2153 and full_lines[-1]['phase']=='verification_complete' and full_lines[-1]['status']=='BACKED_UP','Incomplete actual full execution ledger')
    def blobkey(x):return (x['git_blob'],x['size'],x['sha256'],tuple(sorted(x['paths'])))
    require(len({blobkey(x) for x in actual})==len(actual) and {blobkey(x) for x in actual}=={blobkey(x) for x in blobs},'Full actual ledger/receipt path/hash grouping differs')
    decoded=[x for x in gzip_lines if x['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF']
    require(len(decoded)==26 and gzip_lines[-1]['phase']=='COMPLETE_ACTUAL_REMOTE_GZIP_PROOF' and gzip_lines[-1]['status']=='PASS_ACTUAL_REMOTE_GZIP_DECODE' and gzip_lines[-1]['raw_count']==gzip_lines[-1]['actual_fetches']==26,'Incomplete actual gzip execution ledger')
    keys=('raw_path','raw_bytes','raw_sha256','remote_gzip_path','remote_gzip_bytes','remote_gzip_sha256','remote_git_blob','actual_remote_fetch_this_run','decoded_byte_for_byte_equal_to_preserved_raw')
    def rawkey(x):return tuple(x[k] for k in keys)
    require(len({rawkey(x) for x in decoded})==26 and {rawkey(x) for x in decoded}=={rawkey(x) for x in gzip['actual_remote_gzip_proofs']},'Gzip actual complete tuple ledger/receipt differs')


def import_frozen_resumption():
    path=FACTORY/'backup_logs/resume_belief_v4_r2_frozen_stage.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest()==DEPENDENCIES['backup_logs/resume_belief_v4_r2_frozen_stage.py'],'Frozen resumption dependency differs')
    sp=importlib.util.spec_from_file_location('frozen_r2_resumption_for_join',path)
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    return module


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
        require(not args.handoff.exists() and not args.handoff.is_symlink() and args.handoff.resolve().is_relative_to(LOGS),'Fresh owned backup_logs addendum required')
    source={'path':Path(__file__).relative_to(FACTORY).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    freeze_path=FACTORY/FREEZE
    require(hashlib.sha256(freeze_path.read_bytes()).hexdigest()==args.expected_source_freeze_sha256,'Join source freeze SHA differs')
    freeze=json.loads(freeze_path.read_bytes());require(freeze['source']['sha256']==source['sha256'] and freeze['author_execution']==AUTHOR,'Join source was not frozen before execution')
    for item in freeze['frozen_inputs']:require(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],'New join preparation frozen input differs')
    for name,sha in DEPENDENCIES.items():require(hashlib.sha256((FACTORY/name).read_bytes()).hexdigest()==sha,'Frozen dependency changed: '+name)
    global RESUMPTION
    RESUMPTION=import_frozen_resumption()
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
    stage_lines=[json.loads(line) for line in (FACTORY/'backup_logs/belief-v4-validated-code-candidate-r2_RESUMED_STAGE_LEDGER.jsonl').read_text().splitlines()]
    local_commit=json.loads((FACTORY/'backup_logs/belief-v4-validated-code-candidate-r2_RESUMED_LOCAL_COMMIT.json').read_bytes())
    stage_coverage=validate_stage_ledger(spec,stage,stage_lines,local_commit)
    require(len(spec['files'])==2685 and sum(x['size'] for x in spec['files'])==1080978676,'Checkpoint-specific frozen scope differs')
    previous=json.loads((FACTORY/'backup_logs/BELIEF_V4_CODE_CANDIDATE_CUSTODY_ADAPTER.json').read_bytes())
    m.original_custody(ORIGINAL,previous);private=m.dedicated_custody(WORKTREE,COMMIT)
    checked={}
    for entry in spec['files']:
        actual=m.proof(m.safe(FACTORY,entry['path']),True)
        require(m.equal(entry,actual),'Inventoried current source changed: '+entry['path'])
        require(m.proof(m.safe(WORKTREE,m.PREFIX+'/'+entry['path']),False)==actual,'Inventoried target bytes changed')
        checked[entry['path']]=actual
    m.verify_committed_parent_tree(WORKTREE,COMMIT,PARENT,checked)
    require(m.index(WORKTREE)=={n:v['git_blob'] for n,v in checked.items()},'Exact index blob IDs differ')
    groups={}
    for name,value in checked.items():groups.setdefault(value['git_blob'],[]).append(name)
    require(len(groups)==2153,'Expected actual unique local blobs differ')
    for blob,names in groups.items():
        expected=checked[names[0]]
        actual=h.process_stream(['git','cat-file','blob',blob],lambda stream:m.stream_proof(stream,expected['size'],False),WORKTREE)
        require(actual==expected,'Actual local Git index/commit object bytes differ')
    prior=m.tree(WORKTREE,PARENT);m.guard_previous(prior,checked)
    require(len(prior)==2303,'Previous exact parent path count differs')
    releases=[n for n in checked if '/releases/' in '/'+n];require(len(releases)==210,'Expected all210 releases absent')
    actual_commit_diff=RESUMPTION.validate_diagnostics(RESUMPTION.bounded_process(['git','diff','--check',PARENT,COMMIT],WORKTREE),RESUMPTION.canonical_exception())
    actual_commit_diff['command']=['git','diff','--check',PARENT,COMMIT]
    changed=[n.decode() for n in m.git(WORKTREE,'diff','--name-only','-z',PARENT,COMMIT).split(b'\0') if n]
    require(all(n.startswith(m.PREFIX+'/') and n[len(m.PREFIX)+1:] in checked for n in changed) and changed==stage['staged_paths'],'Unrelated/missing commit path changes')
    parent_full=RESUMPTION.parse_tree(m.git(WORKTREE,'ls-tree','-r','-z',PARENT));index_full=RESUMPTION.parse_index(m.git(WORKTREE,'ls-files','--stage','-z'))
    RESUMPTION.validate_whole_index(index_full,parent_full,checked)
    require(RESUMPTION.parse_tree(m.git(WORKTREE,'ls-tree','-r','-z',COMMIT))==index_full,'Full commit/index/outside-parent tree differs')
    for name,value in checked.items():require(stage_coverage[name]['git_blob']==value['git_blob'],'Actual stage/local payload blob differs')
    others=sorted(set(changed)-{m.PREFIX+'/'+p for p in RESUMPTION.WARNING_PATHS})
    for i in range(0,len(others),100):
        result=RESUMPTION.bounded_process(['git','--literal-pathspecs','diff','--check',PARENT,COMMIT,'--',*others[i:i+100]],WORKTREE)
        require(result=={'returncode':0,'stdout':b'','stderr':b''},'Restricted strict check failed outside frozen warnings')
    local_lossless=m.lossless(FACTORY,spec,checked);require(len(local_lossless)==26,'Own local actual26 lossless proof absent')
    remote=json.loads(m.run(['gh','api','repos/'+m.REPOSITORY+'/git/commits/'+COMMIT]));tree_sha=remote['tree']['sha']
    require([p['sha'] for p in remote['parents']]==[PARENT],'Actual remote sole parent differs')
    report={'status':'INDEPENDENT_LOCAL_PUSHED_CUSTODY_AUDIT_PASS_FULL_REMOTE_GZIP_JOIN_PENDING','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'actual_copy_stage_commit_push_runner':'/root','at_utc':now(),'commit_sha':COMMIT,'sole_parent_sha':PARENT,'spec':spec_artifact,'stage':stage_artifact,'frozen_files':2685,'frozen_bytes':1080978676,'actual_local_git_unique_blobs_verified':2153,'source_target_index_commit_sha256_git_sha1_match':True,'all_parent2303_paths_preserved':True,'immutable_release_files_actual_bytes_verified':210,'strict_diff_check':'NONZERO_EXIT_2_EXACT_FIVE_FROZEN_WARNINGS_EXCEPTED','actual_commit_diff_check':actual_commit_diff,'restricted_commit_other_paths_check':{'paths':len(others),'status':'PASS'},'local_actual26_lossless_proofs':local_lossless,'exact_scoped_commit_changes':len(changed),'private_admin':private,'actual_remote_tree_metadata_sha':tree_sha,'original_main_clean':True,'dedicated_clean':True,'six_protected_git_baselines':m.normalize_baselines(ORIGINAL,previous),'source':source,'source_freeze_sha256':args.expected_source_freeze_sha256,'dependency_checksums':DEPENDENCIES,'model_disclosure':'Inherited same model; separate execution does not imply model diversity','actual_remote_payload_fetches_by_this_preflight_or_join':0,'grants_data_approval':False,'approved_catalog_records':3748,'approved_catalog_families':696,'v4_data_approval':False,'disk_free_bytes':shutil.disk_usage(WORKTREE).free}
    evidence=[]
    if args.phase=='join':
        full_artifact=artifact(FULL,args.expected_full_proof_sha256);gzip_artifact=artifact(GZIP,args.expected_gzip_proof_sha256)
        full=json.loads((FACTORY/FULL).read_bytes());gzip=json.loads((FACTORY/GZIP).read_bytes())
        coverage=validate_proofs(spec,stage,full,gzip,COMMIT,PARENT,SPEC_SHA,STAGE_SHA,args.expected_full_proof_sha256,tree_sha)
        validate_local_remote_blob_coverage(coverage,checked)
        require(full['remote_verification']['verified_unique_blobs']==2153,'Full actual unique fetch count differs')
        full_ledger=m.safe(FACTORY,Path(full['ledger']).relative_to(FACTORY).as_posix())
        gzip_ledger=m.safe(FACTORY,GZIP_LEDGER)
        full_ledger_artifact=artifact(full_ledger.relative_to(FACTORY).as_posix());gzip_ledger_artifact=artifact(GZIP_LEDGER)
        with full_ledger.open() as stream:full_lines=[json.loads(line) for line in stream]
        with gzip_ledger.open() as stream:gzip_lines=[json.loads(line) for line in stream]
        validate_ledgers(full,gzip,full_lines,gzip_lines)
        evidence=[full_artifact,gzip_artifact,full_ledger_artifact,gzip_ledger_artifact]
        report.update(status='BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED',actual_full_inventory_proof=full_artifact,actual_full_inventory_runner='/root',actual_full_unique_remote_fetches=2153,actual_full_ledger=full_ledger_artifact,actual_new_gzip_proof=gzip_artifact,actual_new_gzip_runner=AUTHOR,actual_new_gzip_payload_fetches=26,raw_bytes_exactly_recovered=1353229114,actual_gzip_ledger=gzip_ledger_artifact)
    excluded={n for n in checked}|{e['path'] for e in spec['raw_size_exclusions']}
    report['postinventory_local_path_names_only']=sorted(p.relative_to(FACTORY).as_posix() for p in FACTORY.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store' and p.relative_to(FACTORY).as_posix() not in excluded)
    report['snapshot_deferrals_preserved']=spec['deferred_incomplete_new_v4_paths']
    report['postinventory_boundary']='Only2685 exact frozen files in4d0b5eff are within this backup claim. Current r2 SPEC/report, current failed/resumed-stage/full/gzip proofs and ledgers, NEW resumption/join helpers/freezes/tests/config/fixtures/preflight/join/addendum and expert canonical outputs that completed after the immutable r2 freeze remain local until a later exact inventory. Snapshot deferrals are historical point-in-time scope, not a current unfinished-review claim. No retroactive insertion; all previous receipt statuses remain unchanged historical evidence.'
    report['self_inventory_boundary']='This new output/handoff cannot be in an earlier frozen spec or self-hash its final bytes; include actual final bytes in a later checkpoint.'
    for item in [spec_artifact,stage_artifact,*evidence]:require(artifact(item['path'],item['sha256'])==item,'Evidence changed during preflight/join')
    m.original_custody(ORIGINAL,previous);m.dedicated_custody(WORKTREE,COMMIT)
    for name,sha in DEPENDENCIES.items():require(hashlib.sha256((FACTORY/name).read_bytes()).hexdigest()==sha,'Frozen dependency changed at finish: '+name)
    require(hashlib.sha256(freeze_path.read_bytes()).hexdigest()==args.expected_source_freeze_sha256,'Source freeze changed at finish')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source['sha256'],'Frozen join source changed')
    m.exclusive(args.output,report)
    output_artifact=artifact(args.output.relative_to(FACTORY).as_posix())
    if args.phase=='join':
        text=f'''# BELIEF V4 r2 archival checkpoint completed\n\nCommit `{COMMIT}`, sole parent `{PARENT}`, preserves **2685 files /1,080,978,676 bytes** from spec `{SPEC_SHA}`. Root's completed actual remote proof `{args.expected_full_proof_sha256}` verified **2153 unique GitHub blobs**. Backup manager's separate actual gzip proof `{args.expected_gzip_proof_sha256}` fetched **26 new remote payloads**, decoded byte-for-byte to **1,353,229,114 preserved raw bytes**.\n\nJoined receipt `{output_artifact['path']}` SHA256 `{output_artifact['sha256']}`. Fresh read-only source/target/index/commit checks, private ADMIN, sole parent/tree/branch and six original Git baselines passed. All2303 parent paths and210 immutable release files remain preserved. Exact five archived line212 whitespace warnings retain their actual nonzero exit2; restricted other-path checks passed. Root ran copy/stage/push/full verification; `/root/backup_manager` authored the join and ran the separate gzip verification. Same inherited model; no model diversity or data-quality approval claim.\n\nOnly the frozen2685-file scope is claimed backed up. Postinventory proofs/ledgers/spec/this addendum and expert canonical outputs deferred at freeze and later completed, plus newborn adjudication/prospective work remain local for the next checkpoint. Historical pending/failure receipts and exact frozen spec remain unchanged. No Git mutation by this join.\n'''
        with args.handoff.open('x') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':report['status'],'output':output_artifact,'handoff':artifact(args.handoff.relative_to(FACTORY).as_posix()) if args.phase=='join' else None}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        failure=LOGS/('BELIEF_V4_R2_JOIN_NEW_EXECUTION_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with failure.open('x') as stream:json.dump({'status':'FAILED_NEW_READ_ONLY_CUSTODY_EXECUTION_PRESERVED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'at_utc':now(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'error_type':type(error).__name__,'error':str(error),'no_git_mutation':True},stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        raise
