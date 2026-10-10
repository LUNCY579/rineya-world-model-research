"""ROOT-authorized deterministic gzip preservation of exact diagnosed NEW raw copies only."""
from pathlib import Path
import argparse, datetime, gzip, hashlib, json, os, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
CHUNK=65536
CAP=25*1024*1024

def safe(name):
    if type(name) is not str or not name or any(c in name for c in ('\\','\0','\n','\r')):raise ValueError('Literal path required')
    p=Path(name)
    if p.is_absolute() or p.as_posix()!=name or any(x in ('.','..','.git') for x in p.parts):raise ValueError('Alias/traversal rejected')
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Symlink rejected')
    return FACTORY/p

def key(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def digest(p):
    before=key(p);h=hashlib.sha256()
    with p.open('rb') as s:
        while b:=s.read(CHUNK):h.update(b)
    if key(p)!=before:raise ValueError('Source changed during hashing')
    return {'size':before[2],'sha256':h.hexdigest()}

def check(row):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Closed exact byte binding required')
    p=safe(row['path'])
    if digest(p)!={k:row[k] for k in ('size','sha256')}:raise ValueError('Exact planned bytes changed: '+row['path'])
    return p

def binding(p):return {'path':p.relative_to(FACTORY).as_posix(),**digest(p)}

def exclusive(p,value):
    with p.open('x') as s:json.dump(value,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--preexecution',required=True,type=Path);p.add_argument('--preexecution-sha256',required=True);p.add_argument('--source-sha256',required=True);a=p.parse_args()
    if digest(Path(__file__))['sha256']!=a.source_sha256 or digest(a.plan)['sha256']!=a.plan_sha256 or digest(a.preexecution)['sha256']!=a.preexecution_sha256:raise ValueError('External packing source/config/preexecution differs')
    cfg=json.loads(a.plan.read_bytes());pre=json.loads(a.preexecution.read_bytes())
    if (set(cfg)!={'schema','checkpoint_id','expected_parent','source','diagnosis','diagnosis_config','admission','preparation','planned_mappings','mapping_contract_output','receipt_output','failure_output','runner_execution','grants_data_approval'}
            or cfg['schema']!='next653-exact-named-whitespace-pack-plan-v1' or cfg['expected_parent']!='653ba70f1f7035a5aed137102c14a1ff29d1cf99'
            or cfg['runner_execution']!='/root' or cfg['grants_data_approval'] is not False):raise ValueError('Closed ROOT exact packing plan required')
    if (pre.get('schema')!='next653-exact-named-whitespace-pack-root-preexecution-v1' or pre.get('authorizer_execution')!='/root'
            or pre.get('actual_runner_execution')!='/root' or pre.get('root_authorizes_this_exact_pack') is not True
            or pre.get('plan')!=binding(a.plan) or pre.get('source')!=cfg['source']):raise ValueError('Fresh ROOT exact packing grant required')
    check(cfg['source']);check(pre['raw_source'])
    if pre['raw_source']['sha256']!=cfg['source']['sha256'] or cfg['source']['sha256']!=a.source_sha256:raise ValueError('Raw/current packing source differs')
    for name in ('diagnosis','diagnosis_config','admission','preparation'):check(cfg[name])
    diagnosed=json.loads(safe(cfg['diagnosis']['path']).read_bytes());named=json.loads(safe(cfg['diagnosis_config']['path']).read_bytes())
    if (diagnosed.get('status')!='EXACT_NAMED_BYTE_DIAGNOSIS_NO_PRODUCTION_INVENTORY' or diagnosed.get('checkpoint_id')!=cfg['checkpoint_id']
            or diagnosed.get('config_sha256')!=cfg['diagnosis_config']['sha256'] or diagnosed.get('unmappable_new_whitespace_paths')
            or diagnosed.get('all_named_bytes_rechecked_after') is not True or cfg['planned_mappings']!=diagnosed.get('proposed_mappings')):raise ValueError('Complete exact diagnosed mapping scope required')
    rows=cfg['planned_mappings'];admitted={r['path']:r for r in named['admitted_bindings']};previous=set(named['prior_paths'])
    if not rows or len({r['raw']['path'] for r in rows})!=len(rows):raise ValueError('Nonempty unique exact raw plan required')
    outputs=[safe(cfg[k]) for k in ('mapping_contract_output','receipt_output','failure_output')]
    if any(p.exists() or p.parent!=HERE for p in outputs) or len(set(outputs))!=3:raise ValueError('Fresh exclusive owned contract/receipt/failure outputs required')
    destinations=set();components={'raw_archives','raw_source_archives','original_byte_archives','preexecution_raw_archives','raw_preexecution_archives_attempt03','source_blobs','runtime_raw_archives','execution_raw_archives','raw-archives','runtime-archives','byte_archives'}
    for row in rows:
        if set(row)!={'raw','origin','whitespace_evidence','gzip_path','manifest_path'}:raise ValueError('Exact diagnosed row shape required')
        raw,origin=row['raw'],row['origin'];check(raw);check(origin)
        if (admitted.get(raw['path'])!=raw or admitted.get(origin['path'])!=origin or raw['path'] in previous or raw['path']==origin['path']
                or raw['size']>CAP or not set(Path(raw['path']).parts)&components or '/releases/' in '/'+raw['path']
                or (raw['size'],raw['sha256'])!=(origin['size'],origin['sha256'])):raise ValueError('Only bound NEW archive copies of admitted originals may pack')
        expected_key=hashlib.sha256(raw['path'].encode()).hexdigest()+'-'+raw['sha256']
        expected_gzip='reproducibility/lossless_archives/rsm-runtime-whitespace-r1/'+expected_key+'.gz'
        expected_manifest='manifests/lossless_archives/rsm-runtime-whitespace-r1/'+expected_key+'.lossless.json'
        if row['gzip_path']!=expected_gzip or row['manifest_path']!=expected_manifest:raise ValueError('Exact new standard binary destinations required')
        for name in ('gzip_path','manifest_path'):
            path=safe(row[name])
            if path.exists() or path in destinations:raise ValueError('Packing destination exists or duplicates')
            destinations.add(path)
    results=[]
    try:
        # Recheck the whole named immutable admitted set immediately before any package writes.
        for row in named['named_metadata']+named['admitted_bindings']:check(row)
        for row in rows:
            raw=check(row['raw']);origin=check(row['origin']);packed=safe(row['gzip_path']);manifest=safe(row['manifest_path'])
            packed.parent.mkdir(parents=True,exist_ok=True);manifest.parent.mkdir(parents=True,exist_ok=True)
            observed=hashlib.sha256();size=0
            with raw.open('rb') as source,packed.open('xb') as target:
                with gzip.GzipFile(filename='',fileobj=target,mode='wb',compresslevel=9,mtime=0) as stream:
                    while block:=source.read(CHUNK):observed.update(block);size+=len(block);stream.write(block)
                target.flush();os.fsync(target.fileno())
            if {'size':size,'sha256':observed.hexdigest()}!={k:row['raw'][k] for k in ('size','sha256')}:raise ValueError('Raw changed during packing')
            packed_row=binding(packed)
            if packed_row['size']>CAP:raise ValueError('Compressed artifact exceeds cap; preserve failure and bytes')
            decoded=0
            with raw.open('rb') as source,gzip.open(packed,'rb') as stream:
                while block:=stream.read(CHUNK):
                    if source.read(len(block))!=block:raise ValueError('Decoded gzip/raw byte mismatch')
                    decoded+=len(block)
                if source.read(1) or decoded!=row['raw']['size']:raise ValueError('Complete raw byte roundtrip differs')
            check(row['raw']);check(row['origin'])
            metadata={'raw_path':row['raw']['path'],'raw_bytes':row['raw']['size'],'raw_sha256':row['raw']['sha256'],
                      'gzip_path':packed_row['path'],'gzip_bytes':packed_row['size'],'gzip_sha256':packed_row['sha256'],
                      'codec':'gzip','parameters':{'compresslevel':9,'mtime':0,'filename':''},'stream_chunk_bytes':CHUNK,
                      'actual_full_byte_roundtrip_equal':True,'raw_preserved_unchanged':True,'source':cfg['source'],
                      'plan_sha256':a.plan_sha256,'actual_author_execution':'/root/backup_manager','actual_runner_execution':'/root','grants_data_approval':False}
            exclusive(manifest,metadata);manifest_row=binding(manifest)
            results.append({'raw':row['raw'],'origin':row['origin'],'gzip':packed_row,'manifest':manifest_row,'whitespace_evidence':row['whitespace_evidence']})
        for row in named['named_metadata']+named['admitted_bindings']:check(row)
        for name in ('source','diagnosis','diagnosis_config','admission','preparation'):check(cfg[name])
        contract={'schema':'next653-new-runtime-raw-binary-mapping-v2','checkpoint_id':cfg['checkpoint_id'],'expected_parent':cfg['expected_parent'],
                  'raw_mappings':sorted(results,key=lambda r:r['raw']['path']),'grants_data_approval':False}
        exclusive(outputs[0],contract)
        exclusive(outputs[1],{'status':'PASS_EXACT_LOCAL_NEW_WHITESPACE_RAW_BINARY_ROUNDTRIPS','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             'actual_author_execution':'/root/backup_manager','actual_runner_execution':'/root','source':cfg['source'],'plan_sha256':a.plan_sha256,
                             'preexecution_sha256':a.preexecution_sha256,'mapping_contract':binding(outputs[0]),'mapping_count':len(results),
                             'raw_bytes':sum(r['raw']['size'] for r in results),'gzip_bytes':sum(r['gzip']['size'] for r in results),
                             'all_named_source_and_raw_bytes_unchanged':True,'actual_full_local_byte_roundtrips':len(results),'git_network_data_approval':False})
        print(json.dumps({'status':'PASS_EXACT_LOCAL_NEW_WHITESPACE_RAW_BINARY_ROUNDTRIPS','mapping_count':len(results),'contract':binding(outputs[0]),'receipt':binding(outputs[1])}))
    except BaseException as error:
        exclusive(outputs[2],{'status':'FAILED_EXACT_WHITESPACE_PACK_BYTES_PRESERVED','actual_runner_execution':'/root','error_type':type(error).__name__,
                              'error':str(error),'source':cfg['source'],'plan_sha256':a.plan_sha256,'completed_rows':results,'grants_data_approval':False});raise

if __name__=='__main__':main()
