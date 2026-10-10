"""Exact46 immutable failed-stage evidence only. Source preparation does not grant execution."""
from pathlib import Path
import argparse, datetime, gzip, hashlib, json, os, re, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
CHUNK=65536
CAP=25*1024*1024
POLICY_SHA='037cb6b5f910e1fa4217a47bcaad4a3b381082c492e4216893812a1ff3cfc525'

def safe(value):
    if type(value) is not str or not value or any(c in value for c in ('\\','\0','\n','\r')):raise ValueError('Literal factory path required')
    p=Path(value)
    if p.is_absolute() or p.as_posix()!=value or any(c in ('.','..','.git') for c in p.parts):raise ValueError('Alias/traversal rejected')
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Symlink rejected')
    return FACTORY/p

def key(p):
    s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def digest(p):
    before=key(p);h=hashlib.sha256()
    with p.open('rb') as stream:
        while block:=stream.read(CHUNK):h.update(block)
    if before!=key(p):raise ValueError('Source changed while hashing')
    return {'size':before[2],'sha256':h.hexdigest()}

def binding(p):return {'path':p.relative_to(FACTORY).as_posix(),**digest(p)}
def check(row):
    if (type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0
            or type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):raise ValueError('Closed typed byte binding required')
    p=safe(row['path'])
    if digest(p)!={k:row[k] for k in ('size','sha256')}:raise ValueError('Exact planned bytes changed: '+row['path'])
    return p

def exclusive(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def full_compare(raw,gz):
    before=key(raw),key(gz);total=0
    with raw.open('rb') as original,gzip.open(gz,'rb') as decoded:
        while block:=decoded.read(CHUNK):
            if original.read(len(block))!=block:raise ValueError('Actual decoded/raw bytes differ')
            total+=len(block)
        if original.read(1) or total!=before[0][2]:raise ValueError('Complete EOF/CRC/raw length differs')
    if before!=(key(raw),key(gz)):raise ValueError('Source or gzip changed during roundtrip')
    with gz.open('rb') as stream:
        if stream.read(8)!=b'\x1f\x8b\x08\x00'+b'\0'*4:raise ValueError('Deterministic gzip header differs')
    return total

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--preexecution',required=True,type=Path);p.add_argument('--preexecution-sha256',required=True);p.add_argument('--source-sha256',required=True);a=p.parse_args()
    if digest(Path(__file__))['sha256']!=a.source_sha256 or digest(a.plan)['sha256']!=a.plan_sha256 or digest(a.preexecution)['sha256']!=a.preexecution_sha256:raise ValueError('External source/plan/preexecution differs')
    cfg=json.loads(a.plan.read_bytes());pre=json.loads(a.preexecution.read_bytes())
    if (type(cfg) is not dict or set(cfg)!={'schema','checkpoint_id','expected_parent','source','policy','base_mapping_contract','planned_mappings','mapping_contract_output','receipt_output','failure_output','actual_runner_execution','new_gzip_content_groups','reuse_existing_content_groups','grants_data_approval'}
            or cfg['schema']!='next653-exact46-failed-stage-archival-evidence-pack-plan-v1' or cfg['actual_runner_execution']!='/root'
            or cfg['expected_parent']!='653ba70f1f7035a5aed137102c14a1ff29d1cf99' or cfg['grants_data_approval'] is not False):raise ValueError('Closed exact ROOT pack plan required')
    if (pre.get('schema')!='next653-exact46-pack-root-preexecution-v1' or pre.get('actual_runner_execution')!='/root'
            or pre.get('authorizer_execution')!='/root' or pre.get('root_authorizes_this_exact_pack') is not True
            or pre.get('source')!=cfg['source'] or pre.get('plan')!=binding(a.plan)):raise ValueError('ROOT exact new packing grant required')
    check(cfg['source']);check(pre['raw_source'])
    if cfg['source']!=binding(Path(__file__)) or pre['raw_source']['sha256']!=a.source_sha256:raise ValueError('Current/raw pack source differs')
    policy=json.loads(check(cfg['policy']).read_bytes());base=json.loads(check(cfg['base_mapping_contract']).read_bytes())
    if (cfg['policy']['sha256']!=POLICY_SHA or cfg['checkpoint_id']!=policy['checkpoint_id'] or cfg['base_mapping_contract']!=policy['base_mapping_contract']
            or cfg['planned_mappings']!=policy['raw_evidence'] or len(policy['raw_evidence'])!=46):raise ValueError('Fixed literal46 policy/plan mismatch')
    # All authority metadata is frozen; no generic original class or alias lookup is admitted.
    for k in ('failed_spec','failed_stage_failure','failed_stage_custody','fresh_worktree_completion'):check(policy[k])
    prior=json.loads(check({'path':'backup_logs/belief-v4-approved-release-r1_SPEC.json','size':Path(safe('backup_logs/belief-v4-approved-release-r1_SPEC.json')).stat().st_size,'sha256':'6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'}).read_bytes())
    previous={r['path'] for r in prior['files']+prior['raw_size_exclusions']};rows=cfg['planned_mappings'];dest=set();new_groups={};reuse_groups=set()
    outputs=[safe(cfg[k]) for k in ('mapping_contract_output','receipt_output','failure_output')]
    if any(x.exists() or x.parent!=HERE for x in outputs) or len(set(outputs))!=3:raise ValueError('Fresh owned exclusive outputs required')
    for row in rows:
        raw,origin=row['raw'],row['origin'];check(raw);check(origin)
        if (raw['path'] in previous or raw['size']>CAP or '/releases/' in '/'+raw['path'] or raw['sha256']!=origin['sha256'] or raw['size']!=origin['size']
                or row['origin_is_same_literal_immutable_source']!=(raw==origin)):raise ValueError('Exact immutable new raw/origin authority differs')
        manifest=safe(row['manifest_path'])
        if manifest.exists() or manifest in dest or not row['manifest_path'].startswith('manifests/lossless_archives/rsm-stage-warning-evidence-r1/'):raise ValueError('Exact new manifest destination differs')
        dest.add(manifest)
        package=row['existing_package']
        if package is not None:
            for role in ('raw','gzip','manifest'):check(package[role])
            value=json.loads(safe(package['manifest']['path']).read_bytes())
            expected={'raw_path':package['raw']['path'],'raw_bytes':raw['size'],'raw_sha256':raw['sha256'],'gzip_path':package['gzip']['path'],'gzip_bytes':package['gzip']['size'],'gzip_sha256':package['gzip']['sha256']}
            if any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in expected.items()) or row['gzip_path']!=package['gzip']['path']:raise ValueError('Existing exact package relation differs')
            reuse_groups.add(raw['sha256'])
        else:
            if row['gzip_path']!='reproducibility/lossless_archives/rsm-stage-warning-evidence-r1/'+raw['sha256']+'.gz' or safe(row['gzip_path']).exists():raise ValueError('Exact new content-addressed gzip must be absent')
            new_groups.setdefault(raw['sha256'],row)
    if len(new_groups)!=7 or len(reuse_groups)!=7 or cfg['new_gzip_content_groups']!=7 or cfg['reuse_existing_content_groups']!=7:raise ValueError('Actual derived seven reused/seven new group scope differs')
    for row in base['raw_mappings']:
        for role in ('raw','origin','gzip','manifest'):check(row[role])
    results=[];written=[]
    try:
        for row in new_groups.values():
            raw=check(row['raw']);packed=safe(row['gzip_path']);packed.parent.mkdir(parents=True,exist_ok=True)
            with raw.open('rb') as original,packed.open('xb') as target:
                with gzip.GzipFile(filename='',fileobj=target,mode='wb',compresslevel=9,mtime=0) as encoded:
                    while block:=original.read(CHUNK):encoded.write(block)
                target.flush();os.fsync(target.fileno())
            if packed.stat().st_size>CAP:raise ValueError('New gzip exceeds cap')
            written.append(binding(packed))
        # All46 actual byte roundtrips and EOF/CRC checks complete before any recognized new manifest is published.
        for row in rows:full_compare(check(row['raw']),safe(row['gzip_path']));check(row['origin'])
        for row in rows:
            packed=binding(safe(row['gzip_path']));metadata={'raw_path':row['raw']['path'],'raw_bytes':row['raw']['size'],'raw_sha256':row['raw']['sha256'],'gzip_path':packed['path'],'gzip_bytes':packed['size'],'gzip_sha256':packed['sha256'],'codec':'gzip','parameters':{'compresslevel':9,'mtime':0,'filename':''},'actual_full_byte_roundtrip_equal':True,'raw_preserved_unchanged':True,'policy':cfg['policy'],'source':cfg['source'],'plan_sha256':a.plan_sha256,'actual_author_execution':'/root/backup_manager','actual_runner_execution':'/root','grants_data_approval':False}
            exclusive(safe(row['manifest_path']),metadata)
            results.append({'raw':row['raw'],'origin':row['origin'],'gzip':packed,'manifest':binding(safe(row['manifest_path'])),'whitespace_evidence':row['whitespace_evidence']})
        for row in rows:check(row['raw']);check(row['origin'])
        for row in base['raw_mappings']:
            for role in ('raw','origin','gzip','manifest'):check(row[role])
        for k in ('source','policy','base_mapping_contract'):check(cfg[k])
        for k in ('failed_spec','failed_stage_failure','failed_stage_custody','fresh_worktree_completion'):check(policy[k])
        contract={'schema':base['schema'],'checkpoint_id':cfg['checkpoint_id'],'expected_parent':cfg['expected_parent'],'raw_mappings':sorted(base['raw_mappings']+results,key=lambda r:r['raw']['path']),'grants_data_approval':False}
        exclusive(outputs[0],contract);exclusive(outputs[1],{'status':'PASS_EXACT46_LOCAL_LOSSLESS_EVIDENCE_ONLY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_author_execution':'/root/backup_manager','actual_runner_execution':'/root','source':cfg['source'],'plan_sha256':a.plan_sha256,'preexecution_sha256':a.preexecution_sha256,'mapping_contract':binding(outputs[0]),'new_raw_mappings':46,'retained_existing_mapping_rows':13,'actual_full_local_byte_roundtrips':46,'actual_new_gzip_content_groups':len(written),'reused_existing_content_groups':7,'raw_bytes':sum(r['raw']['size'] for r in rows),'original_actor_bytes_preserved':True,'data_approval':False,'git_network_payload_execution':False})
        print(json.dumps({'status':'PASS_EXACT46_LOCAL_LOSSLESS_EVIDENCE_ONLY','receipt':binding(outputs[1]),'mapping_contract':binding(outputs[0])}))
    except BaseException as error:
        exclusive(outputs[2],{'status':'FAILED_ACTUAL_PACK_EVIDENCE_PRESERVED','error_type':type(error).__name__,'error':str(error),'completed_manifests':results,'new_gzip_bytes_written':written,'source':cfg['source'],'plan_sha256':a.plan_sha256,'actual_runner_execution':'/root','grants_data_approval':False});raise

if __name__=='__main__':main()
