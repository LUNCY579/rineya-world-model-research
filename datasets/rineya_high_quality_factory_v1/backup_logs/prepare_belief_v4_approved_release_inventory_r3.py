#!/usr/bin/env python3
"""NEW streaming derivative of preserved2db inventory: approved-release byte custody.
Reads only factory/Git/references and writes fresh backup_logs outputs. No Git mutation.
All source/raw/secret and gzip comparisons use64KiB streams. Exact stale279 manifest
is bound to preserved279 histories; current149 is independently bound to newpackage.
"""
from __future__ import annotations
import argparse,datetime as dt,gzip,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
AUTHOR='/root/backup_manager'
FACTORY=Path(__file__).resolve().parent.parent
LOGS=FACTORY/'backup_logs'
PARENT='4d0b5eff1af3184fedf7fa25ada729a6cc909268'
CHECKPOINT='belief-v4-approved-release-r1'
FREEZE=LOGS/'BELIEF_V4_APPROVED_RELEASE_R1_INVENTORY_R3_SOURCE_FREEZE.json'
CONFIG=LOGS/'BELIEF_V4_APPROVED_RELEASE_R1_HISTORICAL_RAW_CONFIG.json'
PRIOR_SHA='6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5'
RUNNER_SHA='f0561eb5f0e2ebe16bc46f51ed1fcbb1ad74236e90fba4d9b33da166c6767715'
JOIN_SHA='32c1fbeb9cef515b532e4097a750cd325470010768f594c413439a5720db85c9'
ADAPTER_SHA='7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023'
CATALOG_SHA='5d90edad2e9af72e4ed5be382efac03447a83bbb6d79aba08103038a24d3bbff'
PENDING_PREFIXES=('reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/','reviews/expert/RSM-option-resumption-public-preparation-20261010T1021Z/','reviews/adjudication/RSM-option-resumption-public-preparation-20261010T1048Z/')
CHUNK=65536
R=None


def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def checksum(p):
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as stream:
        while b:=stream.read(CHUNK):h.update(b)
    after=p.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('Source changed during checksum')
    return h.hexdigest()

def runner():
    global R
    p=LOGS/'run_provisional_archival_checkpoint_v4_attempt02.py'
    if p.is_symlink() or checksum(p)!=RUNNER_SHA:raise ValueError('Frozen v3 helper differs before import')
    if R is None:
        sp=importlib.util.spec_from_file_location('frozen_v3_inventory_helpers',p);R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)
    if checksum(p)!=RUNNER_SHA:raise ValueError('Frozen v3 helper changed during import')
    return R

def binding(p):
    return {'path':p.relative_to(FACTORY).as_posix(),'size':p.stat().st_size,'sha256':checksum(p)}

def write(p,value):
    with p.open('x',encoding='utf8') as h:json.dump(value,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())

def check_source_freeze():
    freeze=json.loads(FREEZE.read_bytes())
    if freeze['source']!=binding(Path(__file__)) or freeze['source_frozen_before_first_import_test_execution'] is not True:raise ValueError('Exact independent source freeze missing')
    for row in freeze['frozen_inputs']:
        if binding(runner().safe(FACTORY,row['path']))!=row:raise ValueError('Frozen inventory input changed: '+row['path'])
    return freeze

def snapshot(path,scan=True):
    actual=runner().proof(path,scan)
    return {'path':path.relative_to(FACTORY).as_posix(),'size':actual['size'],'sha256':actual['sha256']},actual['git_blob']

def normalized_record(record):
    x=dict(record)
    for target,alternate in [('raw_bytes','raw_length'),('gzip_path','compressed_path'),('gzip_bytes','compressed_length'),('gzip_sha256','compressed_sha256')]:
        if target not in x and alternate in x:x[target]=x[alternate]
    for key in ('raw_path','raw_sha256','raw_bytes','gzip_path','gzip_sha256','gzip_bytes'):
        if key not in x:raise ValueError('Unsupported lossless manifest schema')
    return x

def declared(value):
    p=Path(value)
    if p.is_absolute():value=p.relative_to(FACTORY).as_posix()
    return runner().safe(FACTORY,value)

def same(entry,size,sha):return type(size) is int and entry['size']==size and type(sha) is str and entry['sha256']==sha

def compare_gzip(raw,gz):
    with raw.open('rb') as a,gzip.open(gz,'rb') as b:
        while True:
            x,y=a.read(CHUNK),b.read(CHUNK)
            if x!=y:raise ValueError('Lossless full-byte gzip comparison failed')
            if not x:break
    return True

def expected_historical_config(prior,pack):
    old=next(e for e in prior['raw_size_exclusions'] if e['path']==runner().MUTABLE_AUDIT)
    current=next(e for e in pack['artifacts'] if e['raw_path']==old['path'])
    archives=[e for e in pack['artifacts'] if e['raw']['sha256']==old['sha256']]
    return {'schema':'belief-v4-approved-release-r1-bound-historical-audit-v1','author_execution':AUTHOR,'expected_parent':PARENT,'prior_spec_sha256':PRIOR_SHA,
            'old_raw':{k:old[k] for k in ('path','size','sha256')},'old_representation':old['lossless_representations'][0],
            'current_pack_record':current,'archive_pack_records':archives,'root_publication_sha256':'9712178453a4c0730edef626eee173555c0cb48a958fc35ec69da00833158051',
            'root_publication_readback_sha256':'5c376b65f3630fbe060a27c8bb707d2d3e466a80c13092e60d790a81426af72b',
            'root_packing_receipt_sha256':'c22acd6710839c8072a072668b9c09bfbae59891ca029bdc1ccb8dac27403e2e',
            'root_living_pack_readback_sha256':'f13c945dc74377ad58453df1757b8928d92bd7cfcac92469c4295ba173da7269','grants_data_approval':False,'exception_count':1}

def validate_historical_config(config,prior,pack):
    if not runner().exact_types(config,expected_historical_config(prior,pack)):raise ValueError('Only exact prior279/current149 historical rule permitted')
    if (config['old_raw']['sha256']!=runner().HISTORICAL_AUDIT_SHA or config['old_raw']['size']!=123141333
            or config['current_pack_record']['raw']['sha256']!=runner().CURRENT_AUDIT_SHA
            or config['current_pack_record']['raw']['bytes']!=123141333
            or {e['raw_path'] for e in config['archive_pack_records']}!=set(runner().HISTORICAL_AUDIT_PATHS)):
        raise ValueError('Exact published audit/history identities differ')
    return config

def record_entry(pack,role):
    return {'path':pack[role+'_path'],'size':pack[role]['bytes'],'sha256':pack[role]['sha256']}

def verify_historical_mismatch(manifest,record,raw_entry,entries,config,compare=compare_gzip):
    old=config['old_raw'];rep=config['old_representation'];current=config['current_pack_record']
    if (manifest!=rep['manifest']['path'] or entries.get(manifest)!=rep['manifest']
            or raw_entry['path']!=old['path'] or not same(raw_entry,current['raw']['bytes'],current['raw']['sha256'])
            or declared(record['raw_path']).relative_to(FACTORY).as_posix()!=old['path']
            or record['raw_bytes']!=old['size'] or record['raw_sha256']!=old['sha256']
            or declared(record['gzip_path']).relative_to(FACTORY).as_posix()!=rep['gzip']['path']
            or record['gzip_bytes']!=rep['gzip']['size'] or record['gzip_sha256']!=rep['gzip']['sha256']
            or entries.get(rep['gzip']['path'])!=rep['gzip']):raise ValueError('Unbound stale manifest/path/hash/bytes mismatch')
    for role in ('gzip','manifest'):
        e=record_entry(current,role)
        if entries.get(e['path'])!=e:raise ValueError('Current149 exact new standard package missing or changed')
    proofs=[]
    for archive in config['archive_pack_records']:
        raw=record_entry(archive,'raw')
        if entries.get(raw['path'])!=raw or not same(raw,old['size'],old['sha256']):raise ValueError('Archived279 raw changed or missing')
        for role in ('gzip','manifest'):
            e=record_entry(archive,role)
            if entries.get(e['path'])!=e:raise ValueError('Archived279 standard recovery package changed or missing')
        compare(declared(raw['path']),declared(rep['gzip']['path']))
        proofs.append({'archived_raw':raw,'original_old_gzip':rep['gzip'],'actual_full_byte_old_gzip_decode_compare':True})
    return {'status':'EXACT_BOUND_HISTORICAL279_CURRENT149_EXCEPTION_PASS','old_mutable_raw':old,
            'old_manifest':rep['manifest'],'old_gzip':rep['gzip'],'current_mutable_raw':record_entry(current,'raw'),
            'current_standard_manifest':record_entry(current,'manifest'),'archived_old_gzip_decode_proofs':proofs,
            'old_manifest_is_historical_not_current_raw_mapping':True,'grants_data_approval':False}

def validate_prior_scope(previous,names,deferred):
    required={e['path'] for e in previous['files']}
    if len(required)!=2685 or not required<=names or required&deferred:raise ValueError('All2685 prior tracked paths must be included; no prior path deferral')
    return required

def observe_deferred(paths,prefix_status,previous):
    prior={e['path'] for e in previous['files']};deferred=[]
    for path in paths:
        name=path.relative_to(FACTORY).as_posix()
        prefix=next((p for p in PENDING_PREFIXES if name.startswith(p) and prefix_status[p]=='RUNNING'),None)
        if prefix is None:continue
        if name in prior:raise ValueError('Prior tracked path cannot be deferred')
        if '__pycache__' in path.parts or path.suffix=='.pyc':continue
        before=path.stat();h=hashlib.sha256()
        with path.open('rb') as s:
            while b:=s.read(CHUNK):h.update(b)
        after=path.stat();stable=all(getattr(before,k)==getattr(after,k) for k in ('st_size','st_mtime_ns','st_ctime_ns'))
        deferred.append({'path':name,'reason':'actual_observed_unfinished_NEW_public_RSM_preparation_at_cut','prefix':prefix,'actor_status_at_cut':'RUNNING',
                         'observed_size':after.st_size,'observed_sha256':h.hexdigest() if stable else None,'point_in_time_digest_stable':stable,'frozen_hash_claim':False})
    return deferred

def custody_guard(release):
    check_source_freeze();m=runner();original=FACTORY.parent/'rineya-world-model-research';worktree=FACTORY.parent/'.rineya-dataset-git-worktree'
    join=json.loads((LOGS/'BELIEF_V4_R2_JOINED_CUSTODY.json').read_bytes());adapter=json.loads((LOGS/'PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json').read_bytes())
    if (checksum(LOGS/'BELIEF_V4_R2_JOINED_CUSTODY.json')!=JOIN_SHA or checksum(LOGS/'PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json')!=ADAPTER_SHA
            or join['commit_sha']!=PARENT or adapter['expected_parent_sha']!=PARENT):raise ValueError('Actual joined parent/adapter differs')
    baseline=m.normalize_baselines(original,adapter)
    if baseline!=m.normalize_baselines(original,join['six_protected_git_baselines']):raise ValueError('Six original guards changed through adapter')
    m.original_custody(original,adapter);private=m.dedicated_custody(worktree,PARENT)
    previous=json.loads((LOGS/'belief-v4-validated-code-candidate-r2_SPEC.json').read_bytes())
    tree=m.tree(worktree,PARENT)
    if set(tree)!={e['path'] for e in previous['files']} or len(tree)!=2685 or m.index(worktree)!=tree or m.whole_index(worktree)!=m.whole_tree(worktree,PARENT):raise ValueError('Clean parent2685 full tree/index differs')
    boundary=json.loads((FACTORY/'quality_reports/RGM_RENDERER_ARCHIVAL_BOUNDARY_PREFLIGHT.json').read_bytes())
    reference_root=FACTORY.parent/'rineya_complementary_research_2026-10-09';refs=[]
    for e in boundary['reference_checks']:
        p=m.safe(reference_root,e['path']);actual=m.proof(p,False)
        if actual['sha256']!=e['expected_sha256']:raise ValueError('Protected reference changed')
        refs.append({'path':str(p),'size':actual['size'],'sha256':actual['sha256']})
    m.verify_readonly_references({'readonly_reference_files':refs})
    hashes=dict(release['historical_release_bindings'])
    for name,e in release['result']['new_release_files'].items():hashes['data/RBM-001/releases/v0002/'+name]=e['sha256']
    actual_releases={p.relative_to(FACTORY).as_posix() for p in (FACTORY/'data').rglob('*') if p.is_file() and '/releases/' in '/'+p.relative_to(FACTORY).as_posix()}
    if len(hashes)!=222 or set(hashes)!=actual_releases:raise ValueError('All222 release files must be exact actual scope')
    for name,sha in hashes.items():
        if m.proof(m.safe(FACTORY,name),False)['sha256']!=sha:raise ValueError('Immutable release changed:'+name)
    catalog=json.loads((FACTORY/'manifests/ACTIVE_RELEASE_CATALOG.json').read_bytes())
    if checksum(FACTORY/'manifests/ACTIVE_RELEASE_CATALOG.json')!=CATALOG_SHA or (catalog['active_records'],catalog['active_source_families'])!=(7204,700):raise ValueError('Actual published catalog7204/700 differs')
    core=json.loads(m.run(['gh','api','rate_limit']))['resources']['core']
    return {'expected_parent':PARENT,'private_admin':private,'original_main_head':m.ORIGINAL_HEAD,'original_main_clean':True,'dedicated_clean':True,'six_protected_baselines':baseline,
            'prior_tracked_paths':2685,'readonly_reference_files':refs,'immutable_release_files':222,'active_records':7204,'active_source_families':700,
            'disk_free_bytes':shutil.disk_usage(FACTORY).free,'api_core_counts_only':{k:core[k] for k in ('limit','remaining','used','reset')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cut-config',type=Path,required=True);p.add_argument('--expected-cut-config-sha256',required=True);a=p.parse_args()
    freeze=check_source_freeze();m=runner()
    if a.cut_config.is_symlink() or not a.cut_config.resolve().is_relative_to(LOGS) or checksum(a.cut_config)!=a.expected_cut_config_sha256:raise ValueError('Actual cut status config must be exact fresh bound bytes')
    cut=json.loads(a.cut_config.read_bytes())
    if (cut.get('checkpoint_id')!=CHECKPOINT or cut.get('author_execution')!=AUTHOR or cut.get('expected_parent')!=PARENT
            or type(cut.get('prefix_status')) is not dict or set(cut['prefix_status'])!=set(PENDING_PREFIXES)
            or any(v not in ('RUNNING','FINAL_COMPLETE') for v in cut['prefix_status'].values())):raise ValueError('Only exact confirmed prefix statuses permitted')
    specpath=LOGS/(CHECKPOINT+'_SPEC.json');reportpath=LOGS/(CHECKPOINT+'_INVENTORY_REPORT.json')
    if specpath.exists() or reportpath.exists():raise ValueError('Fresh spec/report paths required')
    prior=json.loads((LOGS/'belief-v4-validated-code-candidate-r2_SPEC.json').read_bytes());pack=json.loads((LOGS/'BELIEF_V4_APPROVED_RELEASE_LOSSLESS_RECEIPT_ATTEMPT01.json').read_bytes())
    config=validate_historical_config(json.loads(CONFIG.read_bytes()),prior,pack)
    release=json.loads((FACTORY/'quality_reports/BELIEF_V4_RELEASE_ATTEMPT01_EXECUTION.json').read_bytes());before=custody_guard(release)
    # A single path-list cut. New names born after this list are truthful postcut local artifacts.
    paths=sorted(path for path in FACTORY.rglob('*') if path.is_file() and path not in (specpath,reportpath))
    deferred=observe_deferred(paths,cut['prefix_status'],prior);deferred_names={e['path'] for e in deferred}
    entries={};blobs={};generated=[];secret_flags=[]
    for i,path in enumerate(paths,1):
        name=path.relative_to(FACTORY).as_posix()
        if name in deferred_names:continue
        if '__pycache__' in path.parts or path.suffix=='.pyc' or path.name=='.DS_Store':
            e,_=snapshot(path,False);generated.append({**e,'reason':'generated_cache_or_platform_metadata'});continue
        m.safe(FACTORY,name)
        try:e,blob=snapshot(path)
        except ValueError as error:
            if 'secret' in str(error).lower():secret_flags.append(name)
            raise ValueError('Frozen scan rejected path '+name+'; contents withheld; '+str(error)) from error
        entries[name]=e;blobs[name]=blob
        if i%250==0:print(json.dumps({'phase':'actual_streaming_snapshot','observed_paths':i,'included_or_raw':len(entries)}),flush=True)
    validate_prior_scope(prior,set(entries),deferred_names)
    raw={name:e for name,e in entries.items() if e['size']>m.MAX_BYTES}
    if len(raw)!=31:raise ValueError('Expected exact current31 oversized raw evidence paths')
    manifests={};historical=[]
    for name,e in entries.items():
        if not (name.endswith('compression_manifest.json') or name.endswith('.lossless.json')):continue
        if e['size']>m.MAX_BYTES:raise ValueError('Manifest cannot exceed cap')
        data=json.loads(m.safe(FACTORY,name).read_bytes());records=data.get('artifacts',data.get('items',[data]))
        for rec in records:
            rec=normalized_record(rec);rawname=declared(rec['raw_path']).relative_to(FACTORY).as_posix();rawentry=entries.get(rawname)
            if rawentry is None:raise ValueError('Lossless manifest raw missing from this exact cut')
            if not same(rawentry,rec['raw_bytes'],rec['raw_sha256']):
                historical.append(verify_historical_mismatch(name,rec,rawentry,entries,config));continue
            gzname=declared(rec['gzip_path']).relative_to(FACTORY).as_posix();gzentry=entries.get(gzname)
            if gzentry is None or not same(gzentry,rec['gzip_bytes'],rec['gzip_sha256']) or gzentry['size']>m.MAX_BYTES:raise ValueError('Gzip manifest bytes/path mismatch')
            manifests.setdefault(rawname,[]).append((name,gzname))
    if len(historical)!=1:raise ValueError('Require exactly one bound historical stale279/current149 mapping; no other stale manifest allowed')
    exclusions=[]
    for name,e in sorted(raw.items()):
        representations=[]
        for manifest,gz in manifests.get(name,[]):
            if '/releases/' in '/'+gz or '/releases/' in '/'+manifest:raise ValueError('Recovery package must remain outside immutable release dirs')
            compare_gzip(m.safe(FACTORY,name),m.safe(FACTORY,gz))
            representations.append({'gzip':entries[gz],'manifest':entries[manifest],'exact_raw_roundtrip_verified':True,'uncompressed_secret_scan':'PASS'})
        if not representations:raise ValueError('Oversized raw has no actual verified recovery package:'+name)
        exclusions.append({**e,'reason':'raw_exceeds_repository_25MiB_cap_preserved_local','lossless_representations':representations})
        print(json.dumps({'phase':'actual_lossless_verified','raw_path':name,'bytes':e['size']}),flush=True)
    files=[e for name,e in entries.items() if name not in raw]
    preliminary={'files':files,'raw_size_exclusions':exclusions}
    m.verify_actual_release_coverage(FACTORY,preliminary,prior)
    relocation=[{'old_raw_path':m.MUTABLE_AUDIT,'old_raw_bytes':123141333,'old_raw_sha256':m.HISTORICAL_AUDIT_SHA,'current_raw_sha256':m.CURRENT_AUDIT_SHA,'historical_raw_paths':list(m.HISTORICAL_AUDIT_PATHS)}]
    m.validate_historical_raw_custody(prior['raw_size_exclusions'],{e['path']:e for e in exclusions},{n:{**entries[n],'git_blob':blobs[n]} for n in entries if n not in raw},{'historical_raw_relocations':relocation})
    after=custody_guard(release)
    required=2*sum(e['size'] for e in files)+m.RESERVE
    if before['disk_free_bytes']<required or after['disk_free_bytes']<required:raise ValueError('Actual2x full inventory+512MiB reserve unavailable')
    # Full second streamed pass; excluded unfinished files are never mistaken for immutable input.
    for name,e in entries.items():
        current,blob=snapshot(m.safe(FACTORY,name),False)
        if current!=e or blob!=blobs[name]:raise ValueError('Frozen included file changed before final cut:'+name)
    if checksum(a.cut_config)!=a.expected_cut_config_sha256:raise ValueError('Cut config changed during execution')
    check_source_freeze()
    report={'schema':'belief-v4-approved-release-r1-exact-inventory-report-v1','author_execution':AUTHOR,'actual_preparer_execution':AUTHOR,'eventual_checkpoint_runner_execution':'/root','at_utc':now(),'checkpoint_id':CHECKPOINT,'packing_validation':'PASS','secret_flags':secret_flags,'blockers':[],
            'custody_preflight_before':before,'custody_preflight_after':after,'historical_audit_mapping':historical,'historical_raw_relocations':relocation,'raw_size_exclusions':exclusions,'oversized_raw_artifacts':31,'raw_bytes_exactly_recoverable':sum(e['size'] for e in exclusions),
            'deferred_observed_pending_paths':deferred,'declared_prefix_status_at_cut':cut['prefix_status'],'deferred_absent_paths_invented':False,'cut_config':binding(a.cut_config),'excluded_generated_files':generated,'prior2685_paths_preserved':True,'immutable_release_capped_raw_union_count':222,
            'active_catalog_records':7204,'active_source_families':700,'source_freeze':binding(FREEZE),'inventory_source':binding(Path(__file__)),'full_uncompressed_secret_scan':m.metadata_helper().report(require_complete=True),'all_raw_actual_local_gzip_comparisons':'PASS','required_free_bytes_without_own_report':required,
            'grants_data_approval':False,'approval_scope':'Actual independently adjudicated publication is bound as evidence; backup role supplies no new data approval','model_disclosure':'Inherited same-model independent execution; no model diversity claim','self_inventory_boundary':'Report is included; specification and later execution/stage/fullremote/gzip/join receipts/configs and newborn files remain local until next exact cut'}
    write(reportpath,report);reportentry,reportblob=snapshot(reportpath);files.append(reportentry);blobs[reportentry['path']]=reportblob
    required=2*sum(e['size'] for e in files)+m.RESERVE
    free=shutil.disk_usage(FACTORY).free
    if free<required:raise ValueError('Including report exceeds2x full inventory+512MiB reserve')
    spec={'checkpoint_id':CHECKPOINT,'approval':'reviewed_release','approved_by':'/root','runner_execution':'/root','inventory_preparer_execution':AUTHOR,'inventory_script_author_execution':AUTHOR,'validation_status':'passed','expected_parent_sha':PARENT,
          'files':sorted(files,key=lambda e:e['path']),'raw_size_exclusions':exclusions,'excluded_generated_files':generated,'inventory_report':reportentry,'raw_lossless_required_count':31,'inventory_unique_git_blob_groups':len({blobs[e['path']] for e in files}),
          'deferred_observed_pending_paths':deferred,'prefix_status_at_cut':cut['prefix_status'],'historical_raw_relocations':relocation,'prior2685_paths_required_and_preserved':True,'immutable222_release_files_preserved_via_capped_raw_union':True,'post_read_custody_guard_passed':True,
          'inventory_builder_source_sha256':checksum(Path(__file__)),'inventory_source_freeze_sha256':checksum(FREEZE),'source_bound_historical_rule_config':binding(CONFIG),'secret_scan_status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','exact_public_metadata_exception_files':4,'unbound_secret_detections':0,'metadata_exception_config_sha256':m.METADATA_CONFIG_SHA,'cut_config':binding(a.cut_config),'actual_approved_catalog':{'records':7204,'source_families':700,'sha256':CATALOG_SHA},
          'self_inventory_boundary':{'spec':specpath.relative_to(FACTORY).as_posix(),'reason':'Specification and subsequent receipts/configs cannot inventory their own final bytes; carry in next exact checkpoint'},'grants_data_approval':False}
    write(specpath,spec)
    print(json.dumps({'status':'PASS_EXACT_INVENTORY_ONLY','spec':binding(specpath),'report':reportentry,'files':len(files),'bytes':sum(e['size'] for e in files),'unique_git_blob_groups':spec['inventory_unique_git_blob_groups'],'raw_paths':31,'raw_bytes':sum(e['size'] for e in exclusions),'secret_scan_status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','exact_public_metadata_exception_files':4,'unbound_secret_flags':0,'blockers':0,'deferred_actual_observed_names':len(deferred),'generated_exclusions':len(generated),'release_files_capped_raw_union':222,'disk_free_bytes':free,'required_free_bytes':required,'api_core_counts_only':after['api_core_counts_only']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        p=LOGS/('BELIEF_V4_APPROVED_RELEASE_R1_INVENTORY_R3_FAILURE_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        write(p,{'status':'FAILED_INVENTORY_ACTUAL_EVIDENCE_PRESERVED','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'source':binding(Path(__file__)),'at_utc':now(),'error_type':type(e).__name__,'error':str(e),'no_copy_stage_commit_push':True});raise
