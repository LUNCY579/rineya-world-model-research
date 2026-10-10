"""Read-only actual Git added-file whitespace checks; no helper imports or inventory publication."""
from pathlib import Path
import argparse, base64, datetime, hashlib, json, os, re, subprocess, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
CHUNK=65536
CAP=25*1024*1024

def key(p):
    s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def digest(p):
    before=key(p);h=hashlib.sha256()
    with p.open('rb') as s:
        while b:=s.read(CHUNK):h.update(b)
    if before!=key(p):raise ValueError('Actual source changed while hashing')
    return{'size':before[2],'sha256':h.hexdigest()}
def binding(p):return{'path':str(p.relative_to(FACTORY)),**digest(p)}
def check(row):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('Closed byte binding required')
    v=row['path'];p=Path(v)
    if type(v) is not str or p.is_absolute() or p.as_posix()!=v or any(c in v for c in ('\\','\0','\n','\r')) or any(c in ('.','..','.git') for c in p.parts):raise ValueError('Literal factory name required')
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Symlink rejected')
    q=FACTORY/p
    if digest(q)!={k:row[k] for k in('size','sha256')}:raise ValueError('Exact current bytes changed: '+v)
    return q
def publish(p,v):
    with p.open('x') as s:json.dump(v,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--expected-config-sha256',required=True);p.add_argument('--expected-source-sha256',required=True);p.add_argument('--preexecution',type=Path,required=True);p.add_argument('--expected-preexecution-sha256',required=True);a=p.parse_args()
    if digest(Path(__file__))['sha256']!=a.expected_source_sha256 or digest(a.config)['sha256']!=a.expected_config_sha256 or digest(a.preexecution)['sha256']!=a.expected_preexecution_sha256:raise ValueError('Externally bound source/config/preexecution differs')
    cfg=json.loads(a.config.read_bytes());pre=json.loads(a.preexecution.read_bytes())
    if (set(cfg)!={'schema','source','raw_source','prior_spec','failed_spec','exact_policy','actual_original_report','output','failure','actual_reader_execution','grants_data_approval'} or cfg['schema']!='next653-readonly-new-capped-git-whitespace-preflight-v1' or cfg['actual_reader_execution']!='/root/backup_manager' or cfg['grants_data_approval'] is not False):raise ValueError('Exact read-only preflight contract required')
    if pre.get('source')!=cfg['source'] or pre.get('config')!=binding(a.config) or pre.get('root_grant')!='actual_readonly_all_new_capped_git_diff_no_index_check' or pre.get('actual_reader_execution')!='/root/backup_manager':raise ValueError('Preexecution read-only grant binding differs')
    deps={k:check(cfg[k]) for k in('source','raw_source','prior_spec','failed_spec','exact_policy','actual_original_report')}
    if cfg['raw_source']['sha256']!=a.expected_source_sha256 or cfg['source']!=binding(Path(__file__)):raise ValueError('Current/raw reader source differs')
    prior=json.loads(deps['prior_spec'].read_bytes());failed=json.loads(deps['failed_spec'].read_bytes());policy=json.loads(deps['exact_policy'].read_bytes())
    priornames={r['path'] for r in prior['files']};logical={r['path']:r for r in failed['raw_size_exclusions']}
    logical.update({r['raw']['path']:r['raw'] for r in policy['raw_evidence']})
    for r in logical.values():check({k:r[k] for k in('path','size','sha256')})
    original_report=json.loads(deps['actual_original_report'].read_bytes())
    inherited=original_report['observed_new_file_bindings']
    if len(inherited)!=original_report['actual_new_capped_paths_checked'] or any(r['actual_exit_code'] not in (0,1) or r['stdout_bytes'] or r['stderr_bytes'] for r in inherited):raise ValueError('Actual original diagnostics contain a real blocker')
    for row in inherited:check(row['raw'])
    inherited_names={r['raw']['path'] for r in inherited}
    if len(inherited_names)!=len(inherited):raise ValueError('Original scope has duplicate literal names')
    out=FACTORY/cfg['output'];failure=FACTORY/cfg['failure']
    if out.parent!=HERE or failure.parent!=HERE or out.exists() or failure.exists() or out==failure:raise ValueError('Fresh own JSON report/failure aliases required')
    observed=[];ignored=[];findings=[];checked=0
    try:
        paths=sorted(q for q in FACTORY.rglob('*') if q.is_file())
        for q in paths:
            rel=q.relative_to(FACTORY).as_posix()
            if q.is_symlink():raise ValueError('Observed source symlink rejected')
            if '__pycache__' in q.parts or q.suffix=='.pyc' or q.name=='.DS_Store':ignored.append(rel);continue
            if rel in priornames or rel in logical or rel in inherited_names:continue
            r=binding(q)
            if r['size']>CAP:raise ValueError('NEW unrepresented oversized path: '+rel)
            before=key(q)
            # Native Git semantics on all added bytes; no index, attributes, or whitespace-policy override.
            v=subprocess.run(['git','diff','--no-index','--check','--','/dev/null',str(q.absolute())],cwd=FACTORY,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            if before!=key(q) or binding(q)!=r:raise ValueError('Source changed during read-only native Git check')
            checked+=1;row={'raw':r,'actual_exit_code':v.returncode,'stdout_bytes':len(v.stdout),'stdout_sha256':hashlib.sha256(v.stdout).hexdigest(),'stderr_bytes':len(v.stderr),'stderr_sha256':hashlib.sha256(v.stderr).hexdigest()}
            if v.returncode not in (0,1) or v.stdout or v.stderr:
                warning_headers=[]
                for line in v.stdout.decode('utf-8').splitlines():
                    m=re.fullmatch(re.escape(str(q.absolute()))+r':(\d+): (.+)\.',line)
                    if m:warning_headers.append({'line':int(m[1]),'type':m[2]})
                row.update({'warning_headers':warning_headers,'stdout_base64':base64.b64encode(v.stdout).decode(),'stderr_base64':base64.b64encode(v.stderr).decode(),'raw_context_printed':False});findings.append(row)
            observed.append(row)
            if checked%1000==0:print(json.dumps({'actual_checked_new_capped_paths':checked,'actual_warning_paths_so_far':len(findings)}),flush=True)
        for row in observed+inherited:check(row['raw'])
        for k in deps:check(cfg[k])
        report={'schema':'next653-readonly-added-file-whitespace-preflight-result-v1','status':'PASS_NO_NEW_UNREPRESENTED_GIT_WHITESPACE_WARNINGS' if not findings else 'BLOCKED_EXACT_NEW_WHITESPACE_MAPPING_REQUIRED','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_reader_execution':'/root/backup_manager','source':cfg['source'],'config':binding(a.config),'preexecution':binding(a.preexecution),'prior3158_paths_not_reclassified':len(priornames),'explicit_logical_raws_not_proposed_as_git_text':len(logical),'actual_new_capped_paths_checked':checked,'findings':findings,'checked_bytes':sum(x['raw']['size'] for x in observed),'observed_new_file_bindings':observed,'actual_original_report':cfg['actual_original_report'],'inherited_actual_git_checked_bindings_rehashed':len(inherited),'original_report_false_BLOCKED_label_preserved':True,'native_no_index_0_or_1_empty_diagnostics_correctly_classified':True,'generated_cache_names':ignored,'all_observed_checked_bytes_rehashed_after':True,'git_index_worktree_mutation':False,'new_inventory_spec_published':False,'helper_imports_tests_pack_network':0,'grants_data_approval':False}
        publish(out,report);print(json.dumps({'status':report['status'],'actual_checked':checked,'warning_paths':len(findings),'report':binding(out)}))
    except BaseException as error:
        publish(failure,{'status':'FAILED_READONLY_PREFLIGHT_PRESERVED','error_type':type(error).__name__,'error':str(error),'actual_checked':checked,'findings':findings,'source':cfg['source'],'config':binding(a.config),'grants_data_approval':False});raise
if __name__=='__main__':main()
