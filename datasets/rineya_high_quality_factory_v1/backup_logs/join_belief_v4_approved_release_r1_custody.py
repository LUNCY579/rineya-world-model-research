#!/usr/bin/env python3
"""NEW read-only653 approved-release custody join; no implicit remote payload proof.
Main requires separately completed actual full+gzip receipts and hash-bound ledgers.
Metadata exceptions and nonzero whitespace diagnostics stay exact and truthful.
"""
from __future__ import annotations
import argparse,base64,datetime,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,threading
from pathlib import Path
sys.dont_write_bytecode=True
AUTHOR='/root/backup_manager';FACTORY=Path(__file__).resolve().parent.parent;LOGS=FACTORY/'backup_logs'
WORKTREE=FACTORY.parent/'.rineya-dataset-git-worktree';ORIGINAL=FACTORY.parent/'rineya-world-model-research'
COMMIT='653ba70f1f7035a5aed137102c14a1ff29d1cf99';PARENT='4d0b5eff1af3184fedf7fa25ada729a6cc909268'
SPEC='backup_logs/belief-v4-approved-release-r1_SPEC.json';SPEC_SHA='6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'
STAGE='backup_logs/belief-v4-approved-release-r1_STAGE.json';STAGE_SHA='68e764dc40fad0d2b28c351cb0fd5bfea2dff4f628323b48e14b406f281b530e'
LOCAL='backup_logs/belief-v4-approved-release-r1_LOCAL_COMMIT.json';LOCAL_SHA='ca8902559856f61552609ff8fd3d024a52b5f4fbd5150600e66aace7a8075976'
STAGE_LEDGER='backup_logs/belief-v4-approved-release-r1_STAGE_LEDGER.jsonl';STAGE_LEDGER_SHA='b79917cb31f677a0da447d4f84a683728e4daa9a755cae797abb0bb8024e90f3'
RECOVERY='backup_logs/recover_remote_checkpoint_exact_public_metadata_v3.py';RECOVERY_SHA='a9544fd6e18daffea15be3bc65d45a8ab83239aa7632ed65618d0e7eadcd3135'
GZIP_SOURCE='backup_logs/verify_archival_remote_lossless_exact_public_metadata_v3.py';GZIP_SOURCE_SHA='29e27b8d87facdc5afbc8df57c4259515977e7b65895b4134966ad153e99f3d4'
RECOVERY_FREEZE='backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_RECOVERY_SOURCE_FREEZE.json';RECOVERY_FREEZE_SHA='325309feb9d0af45bdab6e6702e0c543a443fbb9044fa8c5d37de76a91fea6af'
GZIP_FREEZE='backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_GZIP_SOURCE_FREEZE.json';GZIP_FREEZE_SHA='7918c9b153ac221dae7197be74c60e2480c8e19b84c92fc66cda312dc368dc28'
CONFIG='backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_SCOPE_CONFIG.json'
FREEZE='backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_SOURCE_FREEZE.json'
B=M=G=None
FIXED={'commit_sha':COMMIT,'sole_parent_sha':PARENT,'spec_sha256':SPEC_SHA,'stage_sha256':STAGE_SHA,'local_commit_sha256':LOCAL_SHA,
 'stage_ledger_sha256':STAGE_LEDGER_SHA,'files':3158,'bytes':1263090770,'unique_git_blobs':2474,'raw_paths':31,'raw_bytes':1724728576,
 'local_decode_iterations':32,'local_decoded_bytes':1766951239,'manifest_references':32,'global_manifest_paths':31,'remote_raw_gzip_pairs':31,
 'all_release_files_capped_raw_union':222,'research_references':65,'original_git_guards':6,'changed_paths':484,'strict_other_paths':482,
 'unrestricted_diff_exit':2,'exact_whitespace_warnings':6,'exact_whitespace_files':2,'atcut_observed_deferred_paths':26,'postcut_completed_adjudication_files':29,
 'author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'grants_data_approval':False}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(ok,message):
    if not ok:raise ValueError(message)
def typed(a,b):
    if type(a) is not type(b):return False
    if type(b) is dict:return set(a)==set(b) and all(typed(a[k],b[k]) for k in b)
    if type(b) is list:return len(a)==len(b) and all(typed(x,y) for x,y in zip(a,b))
    return a==b

CONFIG_SHA='44e73e8efbed8ac5485dd22ed12a17b0fc01dc35936cda553ece2361ab752d50'
TESTS='backup_logs/test_belief_v4_approved_release_r1_join_custody.py'
REQUIRED_PREPARATION_HASHES={'backup_logs/recover_remote_checkpoint_exact_public_metadata_v3.py': 'a9544fd6e18daffea15be3bc65d45a8ab83239aa7632ed65618d0e7eadcd3135', 'backup_logs/verify_archival_remote_lossless_exact_public_metadata_v3.py': '29e27b8d87facdc5afbc8df57c4259515977e7b65895b4134966ad153e99f3d4', 'backup_logs/test_remote_checkpoint_exact_public_metadata_v3.py': 'e3b06fc91241c9e8dea8c3322ef3c49b6b1ec34118dccd412e53b389d7926f39', 'backup_logs/test_archival_remote_lossless_exact_public_metadata_v3.py': '3d4c58c2af170585da3ac8ad1a586a9aa024db793d9ad9ce9778ef6262d07de6', 'backup_logs/author_remote_exact_v3_record.py': '6e7d6f365eac861b19ad875838b345a7cd2e2a2ae308bd3e1cc7e37e448e5ddb', 'backup_logs/finalize_remote_exact_v3_record.py': '8a94f8bfc49690605e459c98a252b30b6778fb1f45a46f7f3f5e903519ffcba6', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_CONFIG.json': '168c5338000efb25e47969a28318da35dea29f8b054ac0146b98d8a8bc544680', 'backup_logs/recover_remote_checkpoint.py': '3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1', 'backup_logs/verify_archival_remote_lossless.py': '7bab2c46dd1e7a52520a90617a2d910b5d6987bad8ec2489b05097045505a0aa', 'backup_logs/exact_four_public_metadata_secret_scan_attempt02.py': 'a79963d367f47c54074714d24bc2a5f372f09bf725a4b2046006cbdb586e4b1d', 'backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json': '2e18c2c156aaefcee75ad6ad92fe51c66abf864150924d5ac1cd26dbeed5aa05', 'backup_logs/run_provisional_archival_checkpoint_v4_attempt02.py': 'f0561eb5f0e2ebe16bc46f51ed1fcbb1ad74236e90fba4d9b33da166c6767715', 'backup_logs/PROSPECTIVE_ARCHIVAL_V4_ATTEMPT02_SOURCE_FREEZE.json': '3c312ce1c2295b78b565191571e2103f4070f92694aeaae6d6953f80e34648ee', 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_EXECUTION_CONFIG.json': '85cfe47a1cabb952d2f00e03d5d3e10d37134a7510c7653fbc05366100a27a72', 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json': '7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023', 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json': 'bd6639e10289bcbd7c27e3bf68241e83dbf7f0d9985400b7abf9df9ae9b9fa4b', 'backup_logs/BELIEF_V4_R2_RESUMPTION_FIXTURES/exact_official_test_13990.source': '13990b79681d988f66eca74562aa67e07f33d23bd2d8d55cd9021726221ae95c', 'backup_logs/BELIEF_V4_R2_RESUMPTION_FIXTURES/exact_diff_check_stdout.txt': '256b3790a01671d8b33647213342d1256f95fd776dcd738065db160efd60d55e', 'backup_logs/belief-v4-validated-code-candidate-r2_SPEC.json': '6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5', 'backup_logs/belief-v4-approved-release-r1_SPEC.json': '6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9', 'backup_logs/belief-v4-approved-release-r1_STAGE.json': '68e764dc40fad0d2b28c351cb0fd5bfea2dff4f628323b48e14b406f281b530e', 'backup_logs/belief-v4-approved-release-r1_LOCAL_COMMIT.json': 'ca8902559856f61552609ff8fd3d024a52b5f4fbd5150600e66aace7a8075976', 'backup_logs/belief-v4-approved-release-r1_STAGE_LEDGER.jsonl': 'b79917cb31f677a0da447d4f84a683728e4daa9a755cae797abb0bb8024e90f3', 'backup_logs/belief-v4-approved-release-r1_INVENTORY_REPORT.json': 'e28d558b6fbc0a5427faf5ad1e114c27ad054718029f103ae9e0c02901e89a40', 'backup_logs/prepare_belief_v4_approved_release_inventory_r4.py': '77abc32c421ad9677406158d95dca9a90c1227a5fcb84518abfe63b60efb82e9', 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_INVENTORY_R4_SOURCE_FREEZE.json': '183d005c3063eb9e4ccac963db3fce63e768aa139fadd8d70c440bb0455eca1d', 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_HISTORICAL_RAW_CONFIG.json': '3a563a5e9b816d120c688d425fa8b634f16baa3456633090c347c7caf11ea27d', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V2_CONFIG.json': '7eac610853157b2c6fe300cdcd87986925f33a68ac92c536ddea1c7adba5fb61', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V2_RECOVERY_SOURCE_FREEZE.json': '50164a64ed99c0635ce816575942587b26c13edf9806e056c65c76d68a249812', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V2_GZIP_SOURCE_FREEZE.json': '333567500a91a3b91a30aabcd59a008c22df77b6c7276282c5cd6550f548b391', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V2_TEST_EXECUTION.json': 'c5a888660c41e6db6528bf02bf9deafbdecb228625adcbeef676ca5452dc9dda', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V2_INDEPENDENT_REVIEW_FINAL.json': '5403d3b9ac65be106a3054ac6a45b9e2cf24e7bd46cd94fe3da0c2bbc0637047', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_SOURCE_ARCHIVES/a9544fd6e18daffea15be3bc65d45a8ab83239aa7632ed65618d0e7eadcd3135.source': 'a9544fd6e18daffea15be3bc65d45a8ab83239aa7632ed65618d0e7eadcd3135', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_SOURCE_ARCHIVES/29e27b8d87facdc5afbc8df57c4259515977e7b65895b4134966ad153e99f3d4.source': '29e27b8d87facdc5afbc8df57c4259515977e7b65895b4134966ad153e99f3d4', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_SOURCE_ARCHIVES/e3b06fc91241c9e8dea8c3322ef3c49b6b1ec34118dccd412e53b389d7926f39.source': 'e3b06fc91241c9e8dea8c3322ef3c49b6b1ec34118dccd412e53b389d7926f39', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_SOURCE_ARCHIVES/3d4c58c2af170585da3ac8ad1a586a9aa024db793d9ad9ce9778ef6262d07de6.source': '3d4c58c2af170585da3ac8ad1a586a9aa024db793d9ad9ce9778ef6262d07de6', 'quality_reports/RGM_RENDERER_ARCHIVAL_BOUNDARY_PREFLIGHT.json': '03ec34f4f778698c9dc52b3bd4624f324abc525aaca1e33417d4c6042d8168a1', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_RECOVERY_SOURCE_FREEZE.json': '325309feb9d0af45bdab6e6702e0c543a443fbb9044fa8c5d37de76a91fea6af', 'backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_GZIP_SOURCE_FREEZE.json': '7918c9b153ac221dae7197be74c60e2480c8e19b84c92fc66cda312dc368dc28', 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_SCOPE_CONFIG.json': '44e73e8efbed8ac5485dd22ed12a17b0fc01dc35936cda553ece2361ab752d50'}

REQUIRED_PREPARATION_HASHES.update({'backup_logs/belief-v4-approved-release-r1_EXACT_METADATA_V3_RECOVERY_VERIFIED_RECEIPT.json': '9facf85590fc6fe1d86065b7927f92de7162fade0016469b7ca6f85123e491b7', 'backup_logs/belief-v4-approved-release-r1_EXACT_METADATA_V3_RECOVERY_VERIFICATION_LEDGER.jsonl': '1fcc48129bef66a143457e1f623a2989026239bc2ec32a072808954ea4ab7974', 'quality_reports/BELIEF_V4_APPROVED_RELEASE_R1_REMOTE_LOSSLESS.json': '0ad1f63b5f322856d8abda203199ae142a6d5f18dd017d2aea199e77f8a846ea', 'backup_logs/belief-v4-approved-release-r1_REMOTE_LOSSLESS_LEDGER.jsonl': 'c58294b492bdc37359a0ec956b9b5406d84131c8c80fcc4cea58ee50eb6dd72a', 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_EXECUTION_CONFIG.json': '84fd5932cf599e9f236f95975eb096159551082fc5305cbd5c46cbae84c60295'})
EXECUTION_CONFIG='backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_JOIN_EXECUTION_CONFIG.json'

def literal_path(name):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name,'Literal relative path required')
    parts=name.split('/');require(all(v not in ('','.','..') for v in parts),'Path aliases forbidden')
    p=FACTORY
    for v in parts:
        p=p/v;require(not p.is_symlink(),'Symlink path forbidden')
    require(p.is_file() and p.absolute().relative_to(FACTORY).as_posix()==name,'Regular literal owned source required')
    return p

def preparation_artifact(name):
    p=literal_path(name);before=p.stat();h=hashlib.sha256();size=0
    with p.open('rb') as stream:
        while data:=stream.read(65536):
            size+=len(data);require(size<=25*1024*1024,'Preparation artifact exceeds cap');h.update(data)
    after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and size==after.st_size,'Preparation bytes changed during read')
    return {'path':name,'size':size,'sha256':h.hexdigest()}

def validate_preparation_freeze(freeze):
    require(type(freeze) is dict and freeze.get('author_execution')==AUTHOR and freeze.get('frozen_before_first_import_test_execution') is True,'Actual preparation freeze required')
    source=Path(__file__).relative_to(FACTORY).as_posix();own=preparation_artifact(source);test=preparation_artifact(TESTS);scope=preparation_artifact(CONFIG)
    require(typed(freeze.get('source'),own) and typed(freeze.get('tests'),test) and typed(freeze.get('scope_config'),scope),'Own source/test/scope bytes missing or substituted')
    require(scope['sha256']==CONFIG_SHA,'Fixed scope config substituted')
    rows=freeze.get('frozen_inputs');require(type(rows) is list and rows,'Mandatory preparation list required')
    observed={}
    for row in rows:
        require(type(row) is dict and set(row)=={'path','size','sha256'} and type(row['path']) is str and type(row['size']) is int and 0<=row['size']<=25*1024*1024 and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Typed preparation row required')
        require(row['path'] not in observed,'Duplicate preparation path forbidden')
        require(typed(row,preparation_artifact(row['path'])),'Frozen preparation bytes differ');observed[row['path']]=row
    required=dict(REQUIRED_PREPARATION_HASHES);required.update({source:own['sha256'],TESTS:test['sha256']})
    require(set(required)<=set(observed),'Mandatory dependency/source/test/config omitted')
    for name,sha in required.items():require(observed[name]['sha256']==sha,'Fixed dependency hash substituted')
    require(typed(observed[source],own) and typed(observed[TESTS],test) and typed(observed[CONFIG],scope),'Required source/test/scope membership differs')
    return True

def load_dependencies():
    global B,M,G
    p=FACTORY/RECOVERY;h=hashlib.sha256(p.read_bytes()).hexdigest();require(h==RECOVERY_SHA and not p.is_symlink(),'Frozen NEW recovery changed')
    s=importlib.util.spec_from_file_location('frozen_approved_release_join_recovery',p);B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
    M=B.import_bound(FACTORY/'backup_logs/run_provisional_archival_checkpoint_v4_attempt02.py',B.STAGE_SOURCE_SHA,'frozen_join_stage')
    G=B.import_bound(FACTORY/GZIP_SOURCE,GZIP_SOURCE_SHA,'frozen_join_gzip')
    require(B.hash_file(p)['sha256']==RECOVERY_SHA,'Recovery changed during import')
    return B,M,G

def artifact(name,expected=None):
    p=B.source_path(FACTORY,name);row={'path':name,**B.hash_file(p)}
    require(row['size']<=25*1024*1024,'Bound metadata exceeds capped bytes')
    if expected is not None:require(row['sha256']==expected,'Bound artifact SHA differs: '+name)
    return row

def read_json(name,expected=None):artifact(name,expected);return json.loads(B.source_path(FACTORY,name).read_bytes())
def lines(name):
    artifact(name)
    with B.source_path(FACTORY,name).open() as s:return [json.loads(line) for line in s]

def validate_stage(spec,stage):
    B.validate_exact_scope(spec);B.validate_stage_observation(stage,M)
    require(stage.get('author_execution')==AUTHOR and stage.get('runner_execution')=='/root' and stage.get('grants_data_approval') is False,'Stage actual roles/boundary differ')
    wanted={'prior_tracked_paths':2685,'prior_release_files':210,'all_release_files_including_oversized_raws':222,'prior_raw_paths_retained':26,'all_raw_paths':31}
    require(typed(stage.get('execution_scope_guards'),wanted),'Actual stage prior/release/raw scope differs')
    require(stage.get('private_repository_evidence')=={'isPrivate':True,'nameWithOwner':B.REPOSITORY,'viewerPermission':'ADMIN'},'Actual stage private ADMIN differs')
    names={e['path'] for e in spec['files']};changed=stage.get('staged_paths')
    require(type(changed) is list and len(changed)==len(set(changed))==484 and all(type(p) is str and p.startswith(B.PREFIX+'/') and p[len(B.PREFIX)+1:] in names for p in changed),'Exact484 stage changes differ')
    raw={e['path']:e for e in spec['raw_size_exclusions']};proofs=stage.get('raw_lossless_proofs',[])
    require(len(proofs)==31 and len({e['path'] for e in proofs})==31 and {e['path'] for e in proofs}==set(raw),'Stage31 raw coverage differs')
    for e in proofs:require(e['secret_scan']=='PASS' and e['byte_for_byte_lossless'] is True and (e['size'],e['sha256'])==(raw[e['path']]['size'],raw[e['path']]['sha256']),'Stage raw secret/byte equality differs')
    return True

def stage_coverage(spec,stage,records,local):
    validate_stage(spec,stage)
    require(records and records[-1].get('phase')=='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING' and records[-1].get('commit_sha')==COMMIT,'Completed actual stage ledger required')
    files={e['path']:e for e in spec['files']};source=[e for e in records if e['phase']=='source_target_verified'];blobs=[e for e in records if e['phase']=='actual_index_blob_verified']
    require(len(source)==3158 and len({e['path'] for e in source})==3158 and {e['path'] for e in source}==set(files),'All actual source-target ledger paths required')
    require(len(blobs)==2474 and len({e['git_blob'] for e in blobs})==2474,'All actual2474 unique index blobs required')
    coverage={}
    for row in blobs:
        require(type(row.get('paths')) is list and row['paths'] and len(row['paths'])==len(set(row['paths'])) and re.fullmatch('[0-9a-f]{40}',row['git_blob']),'Actual index group path/id schema differs')
        for name in row['paths']:
            require(name in files and name not in coverage and (row['size'],row['sha256'])==(files[name]['size'],files[name]['sha256']),'Actual index path/hash differs');coverage[name]=row
    require(set(coverage)==set(files),'Actual index path coverage missing')
    for e in source:require((e['size'],e['sha256'],e['git_blob'])==(coverage[e['path']]['size'],coverage[e['path']]['sha256'],coverage[e['path']]['git_blob']),'Actual source-target-index body crossjoin differs')
    require(local.get('status')=='LOCAL_COMMIT_PRESERVED' and local.get('remote_commit_sha') is None and local.get('commit_sha')==COMMIT,'Local commit preserved status differs')
    for key in ('author_execution','runner_execution','source_sha256','source_freeze_sha256','spec_sha256','files','total_bytes','expected_parent_sha','secret_scan','execution_scope_guards','staged_paths','raw_lossless_proofs','all_source_target_index_sha256_and_sha1_verified','actual_exit_code','actual_stdout_base64','warning_count','exception_files','restricted_strict_diff_check','restricted_changed_path_count','whole_index_and_outside_parent_guard'):
        require(typed(local.get(key),stage.get(key)),'Actual local-stage common binding differs: '+key)
    return coverage

def validate_proofs(spec,stage,full,gz,full_sha,tree_sha):
    validate_stage(spec,stage);inventory,coverage=G.validate_bindings(spec,stage,full,COMMIT,SPEC_SHA,B)
    require(full.get('runner_execution')=='/root' and full.get('repository')==B.REPOSITORY and full.get('branch')==B.BRANCH,'Full actual runner/repository/branch differs')
    require(full.get('source_freeze_sha256')==RECOVERY_FREEZE_SHA and full.get('original_verifier_source_sha256')==B.ORIGINAL_SOURCE_SHA and full.get('metadata_scanner_source_sha256')==B.METADATA_SOURCE_SHA and full.get('metadata_exception_config_sha256')==B.METADATA_CONFIG_SHA,'Full actual source/scanner/freezes differ')
    require(full.get('local_git_verified_unique_blobs')==2474 and full.get('current_source_matching_files')==3158 and full.get('secret_scan_truth')=='PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','Full local proof/report truth differs')
    prior={e['path']:e for e in full.get('prior_evidence',[])}
    for name,sha in ((STAGE,STAGE_SHA),(LOCAL,LOCAL_SHA),(STAGE_LEDGER,STAGE_LEDGER_SHA)):
        require(prior.get(str(FACTORY/name),{}).get('sha256')==sha,'Full proof actual stage/local/ledger evidence not bound')
    private=full.get('private_repository_evidence',{})
    require(private.get('nameWithOwner')==B.REPOSITORY and private.get('isPrivate') is True and private.get('viewerPermission')=='ADMIN','Full private ADMIN differs')
    for key in ('checkpoint_custody_guards_before','checkpoint_custody_guards_after'):
        expected={'all3158_paths_full_tree_index':True,'all2685_parent_paths_preserved':True,'all222_immutable_release_files':True,'all65_readonly_reference_hashes':True,'original_six_git_byte_guards':True,'original_main_and_dedicated_clean':True,'unrestricted_diff_exit':2,'frozen_exact_trailing_whitespace_warnings':6,'restricted_strict_paths':482,'sole_parent_sha':PARENT,'stage_source_sha256':B.STAGE_SOURCE_SHA,'stage_config_sha256':B.STAGE_CONFIG_SHA}
        require(typed(full.get(key),expected),'Full before/after current custody guard differs')
    require(full.get('local_raw_representation_decode_iterations')==32 and full.get('local_raw_representation_decoded_bytes')==1766951239,'Full32 actual local representation coverage differs')
    remote=full['remote_verification'];require(remote['verified_unique_blobs']==2474 and len(remote['blob_receipts'])==2474 and remote['remote_tree_sha']==tree_sha,'Actual full2474 unique object proof/tree differs')
    require(remote.get('http_accept')=='application/vnd.github.raw+json' and remote.get('cache_policy')=='no gh --cache; Cache-Control: no-cache; every blob fetched this run','Actual raw/no-cache full transport differs')
    fixed={'status':'PASS_ACTUAL_REMOTE_GZIP_DECODE','commit_sha':COMMIT,'spec_sha256':SPEC_SHA,'stage_receipt_sha256':STAGE_SHA,
        'verified_inventory_receipt_sha256':full_sha,'source_sha256':GZIP_SOURCE_SHA,'source_freeze_sha256':GZIP_FREEZE_SHA,'dependency_sha256':RECOVERY_SHA,
        'actual_runner_execution':AUTHOR,'runner_execution':AUTHOR,'grants_data_approval':False,'private_admin_confirmed':True,'overall_backup_status_claimed_by_this_gzip_proof':False,
        'actual_remote_tree_sha':tree_sha,'stream_chunk_bytes':65536,'bounded_spool_memory_bytes':8388608,'actual_remote_gzip_fetches':31,'oversized_raw_artifacts':31,
        'raw_bytes_exactly_recovered':1724728576,'actual_decoded_bytes_including_multiple_representations':1724728576,'frozen_representation_reference_count':32,
        'unique_manifest_path_count':31,'unique_raw_gzip_pair_count':31,'local_fullproof_decode_iterations':32,'local_fullproof_decoded_bytes':1766951239}
    for key,value in fixed.items():require(typed(gz.get(key),value),'Actual gzip binding/coverage differs: '+key)
    raw={e['path']:e for e in spec['raw_size_exclusions']};expected_pairs={(e['path'],rep['gzip']['path']) for e in raw.values() for rep in e['lossless_representations']}
    proofs=gz.get('actual_remote_gzip_proofs',[]);require(len(proofs)==31 and len({(e['raw_path'],e['remote_gzip_path']) for e in proofs})==31 and {(e['raw_path'],e['remote_gzip_path']) for e in proofs}==expected_pairs,'All exact31 unique raw-gzip pairs required')
    for p in proofs:
        e=raw[p['raw_path']];blob=coverage[p['remote_gzip_path']]
        require(p['actual_remote_fetch_this_run'] is True and p['decoded_byte_for_byte_equal_to_preserved_raw'] is True,'New actual remote byte comparison required')
        require((p['raw_bytes'],p['raw_sha256'])==(e['size'],e['sha256']) and (p['remote_git_blob'],p['remote_gzip_bytes'],p['remote_gzip_sha256'])==(blob['git_blob'],blob['size'],blob['sha256']),'Actual gzip/raw blob/byte binding differs')
    refs=[]
    for e in spec['raw_size_exclusions']:
        for rep in e['lossless_representations']:
            record=rep['manifest'];refs.append({'raw_path':e['path'],'gzip_path':rep['gzip']['path'],'manifest':record,'actual_remote_git_blob':coverage[record['path']]['git_blob'],
                'actual_payload_proof_source':'Bound full inventory actual remote-byte proof; not a new gzip execution manifest fetch'})
    require(typed(gz.get('full_frozen_lossless_manifest_reference_coverage'),refs),'All32 ordered perraw manifest references must crossjoin actual full proof')
    return coverage

def validate_ledgers(full,gz,full_lines,gzip_lines):
    require(full_lines and full_lines[-1].get('phase')=='verification_complete' and full_lines[-1].get('status')=='BACKED_UP','Actual full ledger completion required')
    def key(e):return (e['git_blob'],e['size'],e['sha256'],tuple(sorted(e['paths'])))
    actual=[e for e in full_lines if e['phase']=='actual_remote_blob_verified'];expected=full['remote_verification']['blob_receipts']
    require(len(actual)==2474 and len({key(e) for e in actual})==2474 and {key(e) for e in actual}=={key(e) for e in expected},'Full ledger actual group proof differs')
    local=[e for e in full_lines if e['phase']=='local_git_blob_verified'];require(len(local)==2474 and len({e['git_blob'] for e in local})==2474,'Full ledger actual local2474 payloads required')
    for e in local:
        match=next((v for v in expected if v['git_blob']==e['git_blob']),None)
        require(match is not None and (e['size'],e['sha256'])==(match['size'],match['sha256']),'Actual local/remote full ledger body differs')
    require(gzip_lines and gzip_lines[-1].get('phase')=='COMPLETE_ACTUAL_REMOTE_GZIP_PROOF' and gzip_lines[-1].get('status')=='PASS_ACTUAL_REMOTE_GZIP_DECODE' and gzip_lines[-1].get('raw_count')==31 and gzip_lines[-1].get('actual_fetches')==31,'Actual31gzip ledger completion required')
    fields=('raw_path','raw_bytes','raw_sha256','remote_gzip_path','remote_gzip_bytes','remote_gzip_sha256','remote_git_blob','actual_remote_fetch_this_run','decoded_byte_for_byte_equal_to_preserved_raw')
    def rk(e):return tuple(e[k] for k in fields)
    actual=[e for e in gzip_lines if e['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF'];expected=gz['actual_remote_gzip_proofs']
    require(len(actual)==31 and len({rk(e) for e in actual})==31 and {rk(e) for e in actual}=={rk(e) for e in expected},'Actual gzip ledger exact tuple coverage differs')

def crossjoin(coverage,checked):
    require(set(coverage)==set(checked) and all((coverage[n]['size'],coverage[n]['sha256'],coverage[n]['git_blob'])==(v['size'],v['sha256'],v['git_blob']) for n,v in checked.items()),'Stage/full/current source-target-index-commit exact payload groups differ')

def validate_scope(config,spec):
    for key,value in FIXED.items():require(typed(config.get(key),value),'Frozen join scope/type differs: '+key)
    require(typed(config.get('historical_observed_pending_atcut'),spec.get('deferred_observed_pending_paths')) and len(config['historical_observed_pending_atcut'])==26,'Historical26 deferrals differ')
    rows=config.get('completed_postcut_adjudication_artifacts',[]);require(type(rows) is list and len(rows)==29 and len({e['path'] for e in rows})==29,'Actual29 later completed artifacts required')
    names={e['path'] for e in spec['files']}|{e['path'] for e in spec['raw_size_exclusions']}
    require(all(e['path'].startswith('reviews/adjudication/RSM-option-resumption-public-preparation-20261010T1048Z/') and e['path'] not in names for e in rows),'Postcut29 must remain outside exact checkpoint')
    return True

def validate_execution_config(config):
    require(config.get('author_execution')==AUTHOR and config.get('actual_runner_execution')==AUTHOR and config.get('actual_completed_proofs_bound') is True and config.get('grants_data_approval') is False,'Separate actual final-proof config required')
    require(config.get('commit_sha')==COMMIT and config.get('spec_sha256')==SPEC_SHA,'Execution exact checkpoint differs')
    bindings=config.get('bindings');require(type(bindings) is dict and set(bindings)=={'full','full_ledger','gzip','gzip_ledger'},'Exact four final evidence bindings required')
    require(len({e['path'] for e in bindings.values()})==4,'Distinct final proof/ledger paths required')
    for key,row in bindings.items():
        require(type(row) is dict and set(row)=={'path','size','sha256'} and type(row['size']) is int and 0<row['size']<=25*1024*1024 and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Actual final evidence row typing differs')
        require(row['path'].startswith('backup_logs/') or (key=='gzip' and row['path'].startswith('quality_reports/')),'Final proof outside owned evidence scope')
    return bindings

def bounded_commit_diff(paths=None):
    argv=['git','--literal-pathspecs','diff','--check',PARENT,COMMIT]
    if paths is not None:require(paths,'Empty restricted path list forbidden');argv+=['--',*paths]
    proc=subprocess.Popen(argv,cwd=WORKTREE,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=M.ENV)
    out={'stdout':bytearray(),'stderr':bytearray()};errors=[]
    def read(k):
        try:
            while c:=getattr(proc,k).read(65536):
                require(len(out[k])+len(c)<=2*1024*1024,'Diagnostics cap exceeded');out[k].extend(c)
        except Exception as e:errors.append(e);proc.kill()
        finally:getattr(proc,k).close()
    threads=[threading.Thread(target=read,args=(k,),daemon=True) for k in out]
    for t in threads:t.start()
    try:code=proc.wait(timeout=180)
    finally:
        if proc.poll() is None:proc.kill()
        proc.wait()
        for t in threads:t.join()
    if errors:raise errors[0]
    return {'returncode':code,**{k:bytes(v) for k,v in out.items()}}

def verify_current(spec,stage,stage_paths,coverage,scope):
    checked={}
    for e in spec['files']:
        result=B.metadata_helper().proof(B.source_path(FACTORY,e['path']),B.SECRET_PATTERNS)
        require((result['size'],result['sha256'])==(e['size'],e['sha256']),'Current exact inventoried source differs');checked[e['path']]=result
    source_before=B.validate_metadata_report(B.metadata_helper().report(require_complete=True));crossjoin(stage_paths,checked);crossjoin(coverage,checked)
    for name,value in checked.items():require(M.proof(M.safe(WORKTREE,B.PREFIX+'/'+name),False)==value,'Fully scanned source/SPEC/target hash equality differs')
    for context in B._STREAM_REPORTS:B._STREAM_REPORTS[context].clear()
    groups={}
    for name,value in checked.items():groups.setdefault(value['git_blob'],[]).append(name)
    require(len(groups)==2474,'Current exact2474 local Git groups required')
    for blob,names in groups.items():
        e=checked[names[0]];actual=B.process_stream(['git','cat-file','blob',blob],lambda h:B.hash_stream(h,e['size'],blob,True,names,'local_git'),WORKTREE)
        require((actual['size'],actual['sha256'])==(e['size'],e['sha256']),'Actual local Git payload differs')
    local_report=B.complete_metadata_stream_report('local_git')
    local_paths={B.PREFIX+'/'+n:{'sha':v['git_blob']} for n,v in checked.items()}
    guards=B.full_checkpoint_guards(WORKTREE,COMMIT,spec,local_paths)
    changed=[e.decode() for e in M.git(WORKTREE,'diff','--name-only','-z',PARENT,COMMIT).split(b'\0') if e]
    require(changed==stage['staged_paths'],'Exact committed changed paths differ from actual stage')
    warning=read_json('backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json',B.WARNING_CONFIG_SHA)
    M.verify_warning_sources(FACTORY,WORKTREE,checked,M.index(WORKTREE),warning)
    diagnostics=M.validate_diagnostics(bounded_commit_diff(),warning);excepted={B.PREFIX+'/'+e['path'] for e in warning['files']};others=sorted(set(changed)-excepted)
    require(len(others)==482 and len(excepted)==2 and excepted<=set(changed),'Restricted482 source paths differ')
    for start in range(0,len(others),100):require(bounded_commit_diff(others[start:start+100])=={'returncode':0,'stdout':b'','stderr':b''},'Restricted strict diff failed')
    lossless=M.lossless(FACTORY,spec,checked);require(len(lossless)==31,'Own full31raw lossless proof required')
    for e in scope['completed_postcut_adjudication_artifacts']:require(artifact(e['path'],e['sha256'])==e,'Completed local29 artifact changed')
    for e in spec['files']:require(B.check_source(FACTORY,e,False) is None,'Held source evolved after current byte checks')
    source_after=B.validate_metadata_report(B.metadata_helper().report(require_complete=True))
    for name,value in checked.items():require(M.proof(M.safe(WORKTREE,B.PREFIX+'/'+name),False)==value,'Target changed at finish')
    require(B.full_checkpoint_guards(WORKTREE,COMMIT,spec,local_paths)==guards,'Current custody guards changed at finish')
    return {'source_target_index_commit_equal':True,'actual_local_git_payload_groups_verified_this_join':2474,'source_metadata_before':source_before,'source_metadata_after':source_after,
        'actual_local_git_metadata':local_report,'current_checkpoint_guards':guards,'actual_commit_unrestricted_diff':diagnostics,'restricted_strict_paths':482,
        'actual_raw_byte_comparisons':lossless,'secret_skip_on_target_adapter':'Only after full originalSPEC source regex scan +size/SHA256/GitSHA1 equality; current targets and source equality checked both ends',
        'private_admin':M.dedicated_custody(WORKTREE,COMMIT),'six_protected_git_baselines':M.normalize_baselines(ORIGINAL,read_json('backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json',B.CUSTODY_SHA))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source-freeze','execution-config','output','handoff'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('expected-freeze-sha256','expected-execution-config-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    freeze_name=a.source_freeze.absolute().relative_to(FACTORY).as_posix();freeze_row=preparation_artifact(freeze_name)
    require(freeze_row["sha256"]==a.expected_freeze_sha256,"Explicit preparation freeze SHA differs")
    freeze=json.loads(literal_path(freeze_name).read_bytes());validate_preparation_freeze(freeze)
    load_dependencies()
    for path in (a.output,a.handoff):require(not path.exists() and not path.is_symlink() and path.absolute().is_relative_to(LOGS) and B.source_path(FACTORY,path.absolute().relative_to(FACTORY).as_posix())==path.absolute(),'Fresh literal owned output required')
    require(a.output!=a.handoff,'Distinct output/addendum required')
    validate_preparation_freeze(freeze);inputs=freeze['frozen_inputs']
    execution_name=a.execution_config.absolute().relative_to(FACTORY).as_posix();require(execution_name==EXECUTION_CONFIG,'Exact frozen execution-config path required')
    require(any(e==artifact(execution_name,a.expected_execution_config_sha256) for e in inputs),'Execution config missing from preparation freeze')
    execution=read_json(execution_name,a.expected_execution_config_sha256);bindings=validate_execution_config(execution)
    for e in bindings.values():require(artifact(e['path'],e['sha256'])==e,'Completed actual evidence binding differs')
    spec=read_json(SPEC,SPEC_SHA);stage=read_json(STAGE,STAGE_SHA);local=read_json(LOCAL,LOCAL_SHA);scope=read_json(CONFIG,CONFIG_SHA);validate_scope(scope,spec)
    stage_paths=stage_coverage(spec,stage,lines(STAGE_LEDGER),local)
    full=read_json(bindings['full']['path'],bindings['full']['sha256']);gz=read_json(bindings['gzip']['path'],bindings['gzip']['sha256'])
    require(full.get('ledger')==str(FACTORY/bindings['full_ledger']['path']),'Full actual absolute ledger path must bind configured bytes')
    tree=M.text(WORKTREE,'rev-parse',COMMIT+'^{tree}');coverage=validate_proofs(spec,stage,full,gz,bindings['full']['sha256'],tree)
    validate_ledgers(full,gz,lines(bindings['full_ledger']['path']),lines(bindings['gzip_ledger']['path']));crossjoin(coverage,{n:{k:e[k] for k in ('size','sha256','git_blob')} for n,e in stage_paths.items()})
    current=verify_current(spec,stage,stage_paths,coverage,scope)
    remote=B.api('repos/'+B.REPOSITORY+'/git/commits/'+COMMIT);require(remote['sha']==COMMIT and remote['tree']['sha']==tree and [e['sha'] for e in remote['parents']]==[PARENT],'Actual remote commit/tree/sole-parent differs')
    excluded={e['path'] for e in spec['files']}|{e['path'] for e in spec['raw_size_exclusions']}
    postlocal=sorted(p.relative_to(FACTORY).as_posix() for p in FACTORY.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='.DS_Store' and p.relative_to(FACTORY).as_posix() not in excluded)
    report={'status':'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,
        'copy_stage_commit_push_full_verification_runner_execution':'/root','gzip_runner_execution':AUTHOR,'commit_sha':COMMIT,'sole_parent_sha':PARENT,'tree_sha':tree,'at_utc':now(),
        'exact_scope':FIXED,'spec':artifact(SPEC,SPEC_SHA),'stage':artifact(STAGE,STAGE_SHA),'local_commit':artifact(LOCAL,LOCAL_SHA),'stage_ledger':artifact(STAGE_LEDGER,STAGE_LEDGER_SHA),
        'actual_completed_proofs':bindings,'current_actual_custody':current,'actual_unique_remote_blob_groups_proved_by_root':2474,
        'actual_root_http_attempt_count_not_asserted':'Retries may occur;2474 counts unique objects with completed actual byte proofs',
        'actual_remote_gzip_fetches_in_separate_proof':31,'remote_decoded_bytes':1724728576,'local_fullproof_decode_iterations':32,'local_fullproof_decoded_bytes':1766951239,
        'perraw_manifest_refs':32,'global_manifest_paths':31,'actual_remote_payload_fetches_this_join':0,'current_approved_catalog_records':7204,'current_approved_source_families':700,
        'grants_data_approval':False,'same_model_disclosure':'Inherited same model; independent execution does not imply model diversity',
        'historical_observed_pending_atcut':scope['historical_observed_pending_atcut'],'completed_postcut_adjudication_artifacts':scope['completed_postcut_adjudication_artifacts'],
        'current_postinventory_local_path_names_only':postlocal,'self_inventory_boundary':'New join/source/test/freeze/config/proof/ledger/receipt/addendum bytes are postcutLOCAL, cannot selfhash theirfinalbytes or be retroinserted in3158SPEC.',
        'postcut_boundary':'Only exact3158cappedfiles +31rawsrecoverablethroughfrozenlosslessbytes are claimed.26observedarchitectpaths wereRUNNINGatcut;29latercompletedartifacts remainLOCAL. LaterpublicRSM preps/rootproofs/new files areoutside653.',
        'source':artifact(Path(__file__).relative_to(FACTORY).as_posix()),'source_freeze':artifact(freeze_name,a.expected_freeze_sha256),'execution_config':artifact(execution_name,a.expected_execution_config_sha256),'disk_free_bytes':shutil.disk_usage(FACTORY).free}
    for e in inputs+list(bindings.values()):require(artifact(e['path'],e['sha256'])==e,'Bound evidence/source changed before finalwrite')
    require(artifact(freeze_name,a.expected_freeze_sha256)=={'path':freeze_name,**B.hash_file(a.source_freeze)},'Source freeze changed')
    validate_preparation_freeze(freeze)
    B.write_exclusive(a.output,report);out=artifact(a.output.relative_to(FACTORY).as_posix())
    text=f'''# BELIEF V4 approved-release archival checkpoint completed\n\nCommit `{COMMIT}` (sole parent `{PARENT}`) preserves3158files/1,263,090,770bytes under spec `{SPEC_SHA}`. Root's separate actual full proof verified2474unique GitHub objects. Backup manager's separate actual31gzip fetches recovered1,724,728,576preservedrawbytes;32perraw manifestreferences cover31globalmanifestpaths. Local fullverification performed32decodeiterations/1,766,951,239bytes.\n\nJoined custody `{out['path']}` SHA256 `{out['sha256']}` rechecked all source/target/index/localGit payloads,222releasefiles via capped/rawunion,65readonlyreferences andsix originalGit byte/existence guards; bothcheckouts clean/privateADMIN. Unrestricted diff retains exit2/exact6frozenwhitespacewarnings;482otherchangedpaths pass strict checks. No Git mutation or dataapproval by this join. Root ran stage/push/fullverification; backup_manager authored and ran this join and separategzip proof. Same inheritedmodel.\n\nOnly frozen3158scope is claimed.26observedarchitectpaths deferredatcut and29subsequentlycompletedfiles, newproofs/ledgers/helpers/thisaddendum and laterRSMpreparations remain local untila laterexactcheckpoint. Historical failure/pending receipts remain unchanged.\n'''
    with a.handoff.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':report['status'],'output':out,'handoff':artifact(a.handoff.relative_to(FACTORY).as_posix())}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        p=LOGS/('BELIEF_V4_APPROVED_RELEASE_R1_JOIN_NEW_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with p.open('x') as f:json.dump({'status':'FAILED_NEW_READ_ONLY_JOIN_EVIDENCE_PRESERVED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'at_utc':now(),'error_type':type(e).__name__,'error':str(e),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_Git_mutation':True},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        raise
