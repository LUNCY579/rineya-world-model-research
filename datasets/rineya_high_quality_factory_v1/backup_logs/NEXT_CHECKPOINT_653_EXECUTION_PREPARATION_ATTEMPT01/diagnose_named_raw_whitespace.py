"""One authorized stdlib-only hash/opaque-line diagnosis of explicitly named frozen closures."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
CHUNK=65536
CAP=25*1024*1024

def safe(name):
    p=Path(name)
    if type(name) is not str or not name or p.is_absolute() or p.as_posix()!=name or any(x in ('.','..','.git') for x in p.parts) or any(c in name for c in ('\\','\0','\n','\r')):raise ValueError('Literal named factory path required')
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Symlink rejected')
    return FACTORY/p

def stat(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def proof(p,whitespace=False):
    before=stat(p);whole=hashlib.sha256();linehash=hashlib.sha256();tail=b'';length=0;line=1;findings=[];last_blank=None;text=True
    with p.open('rb') as stream:
        while block:=stream.read(CHUNK):
            whole.update(block)
            if not whitespace:continue
            if b'\0' in block:text=False
            parts=block.split(b'\n')
            for i,part in enumerate(parts):
                linehash.update(part);length+=len(part);tail=(tail+part)[-2:]
                if i==len(parts)-1:continue
                linehash.update(b'\n');end=tail[:-1] if tail.endswith(b'\r') else tail
                if end.endswith((b' ',b'\t')):findings.append({'line':line,'type':'trailing_whitespace','line_sha256':linehash.hexdigest()})
                last_blank={'line':line,'type':'blank_line_at_eof','line_sha256':linehash.hexdigest()} if length==0 or (length==1 and tail==b'\r') else None
                line+=1;linehash=hashlib.sha256();tail=b'';length=0
                if len(findings)>100000:raise ValueError('Bounded diagnosis finding cap exceeded')
    if length:
        if tail.endswith((b' ',b'\t')):findings.append({'line':line,'type':'trailing_whitespace','line_sha256':linehash.hexdigest()})
        last_blank=None
    if last_blank is not None:findings.append(last_blank)
    if stat(p)!=before:raise ValueError('Frozen path changed during diagnosis')
    return {'size':before[2],'sha256':whole.hexdigest()},findings if text else []

def check(row):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or len(row['sha256'])!=64:raise ValueError('Closed explicit byte binding required')
    p=safe(row['path']);observed,_=proof(p)
    if observed!={k:row[k] for k in ('size','sha256')}:raise ValueError('Named frozen bytes differ: '+row['path'])
    return p

def exclusive(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--config-sha256',required=True);p.add_argument('--preexecution',required=True,type=Path);p.add_argument('--preexecution-sha256',required=True);p.add_argument('--source-sha256',required=True);a=p.parse_args()
    if proof(Path(__file__))[0]['sha256']!=a.source_sha256 or proof(a.config)[0]['sha256']!=a.config_sha256 or proof(a.preexecution)[0]['sha256']!=a.preexecution_sha256:raise ValueError('External source/config/preexecution binding differs')
    cfg=json.loads(a.config.read_bytes());pre=json.loads(a.preexecution.read_bytes())
    if set(cfg)!={'schema','checkpoint_id','source','named_metadata','admitted_bindings','candidate_raw_paths','prior_paths','output','failure','actor','grants_data_approval'} or cfg['schema']!='next653-named-raw-whitespace-diagnosis-v1' or cfg['actor']!='/root/backup_manager' or cfg['grants_data_approval'] is not False:raise ValueError('Exact scoped diagnosis configuration required')
    if pre.get('authorizer_execution')!='/root' or pre.get('metadata_hash_opaque_line_diagnosis_only') is not True or pre.get('no_helper_import_tests_inventory_git_network_pack') is not True:raise ValueError('Bounded diagnosis preexecution required')
    check(cfg['source']);check(pre['raw_source'])
    if cfg['source']['sha256']!=pre['raw_source']['sha256']:raise ValueError('Raw source archive differs')
    output,failure=safe(cfg['output']),safe(cfg['failure'])
    if output.exists() or failure.exists() or output.parent!=HERE or failure.parent!=HERE:raise ValueError('Fresh exclusive owned outputs required')
    try:
        rows=cfg['admitted_bindings'];named={r['path']:r for r in rows}
        if len(named)!=len(rows) or cfg['candidate_raw_paths']!=sorted(set(cfg['candidate_raw_paths'])) or not set(cfg['candidate_raw_paths']).issubset(named):raise ValueError('Closed explicit path membership required')
        for r in cfg['named_metadata']:check(r)
        for r in rows:check(r)
        byhash={}
        for r in rows:byhash.setdefault((r['size'],r['sha256']),[]).append(r)
        proposed=[];oversized=[];unmappable=[]
        for name in cfg['candidate_raw_paths']:
            r=named[name]
            if name in cfg['prior_paths']:continue
            if r['size']>CAP:oversized.append(r);continue
            measured,warnings=proof(safe(name),True)
            if measured!={k:r[k] for k in ('size','sha256')}:raise ValueError('Raw changed during opaque line diagnosis')
            if not warnings:continue
            origins=sorted((x for x in byhash[(r['size'],r['sha256'])] if x['path']!=name),key=lambda x:x['path'])
            if not origins:unmappable.append({'raw':r,'whitespace_evidence':warnings});continue
            key=hashlib.sha256(name.encode()).hexdigest()
            proposed.append({'raw':r,'origin':origins[0],'whitespace_evidence':warnings,'gzip_path':'reproducibility/lossless_archives/rsm-runtime-whitespace-r1/'+key+'-'+r['sha256']+'.gz','manifest_path':'manifests/lossless_archives/rsm-runtime-whitespace-r1/'+key+'-'+r['sha256']+'.lossless.json'})
        for r in cfg['named_metadata']+rows:check(r)
        result={'schema':'next653-named-raw-whitespace-diagnosis-result-v1','status':'EXACT_NAMED_BYTE_DIAGNOSIS_NO_PRODUCTION_INVENTORY','actual_actor':'/root/backup_manager','checkpoint_id':cfg['checkpoint_id'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':cfg['source'],'config_sha256':a.config_sha256,'preexecution_sha256':a.preexecution_sha256,'named_admitted_paths_checked':len(rows),'candidate_raw_paths_checked':len(cfg['candidate_raw_paths']),'proposed_mappings':proposed,'oversized_new_named_raw_paths':oversized,'unmappable_new_whitespace_paths':unmappable,'all_named_bytes_rechecked_after':True,'contents_or_match_text_emitted':False,'helper_imports_tests_inventory_git_network_pack':0,'grants_data_approval':False}
        exclusive(output,result);print(json.dumps({'status':result['status'],'named_paths':len(rows),'mapping_paths':len(proposed),'oversized_new_raw':len(oversized),'unmappable':len(unmappable)}))
    except BaseException as error:
        exclusive(failure,{'status':'FAILED_NAMED_METADATA_DIAGNOSIS_PRESERVED','error_type':type(error).__name__,'error':str(error),'source':cfg['source'],'config_sha256':a.config_sha256,'preexecution_sha256':a.preexecution_sha256,'grants_data_approval':False});raise

if __name__=='__main__':main()
