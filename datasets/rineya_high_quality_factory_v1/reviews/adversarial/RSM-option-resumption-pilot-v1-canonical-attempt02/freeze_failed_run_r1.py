"""Failure custody metadata only; no semantic helper or target parsing/import.
Prepared and byte-hashed before this small stdlib-only custody execution.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    sys.dont_write_bytecode=True
    freeze_raw=(OUT/'SOURCE_FREEZE.json').read_bytes()
    if sha(freeze_raw)!='8fe4fe30284584956b08ced910281b1f2650851bcd3e44c95ebcfc360b56c29e':raise ValueError('sourcefreeze drift')
    original=json.loads(freeze_raw);pairs=[]
    for row in original['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p
        a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if len(a)!=row['size_bytes'] or len(b)!=row['size_bytes'] or a!=b or sha(a)!=row['sha256'] or sha(b)!=row['sha256']:raise ValueError('postfailure exactpair drift')
        pairs.append(row)
    paths=[OUT/name for name in ['FACTUAL_RUN_STDOUT.txt','FACTUAL_RUN_STDERR.txt','FACTUAL_RUN_EXIT_STATUS.txt']]
    files=[]
    for p in paths:
        b=p.read_bytes();archive=OUT/'failed_run_raw'/(p.name+'.raw')
        if archive.read_bytes()!=b:raise ValueError('failure raw byte drift')
        files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(b),'size_bytes':len(b),'archive':str(archive.relative_to(ROOT))})
    if paths[2].read_text()!='1\n' or paths[0].read_bytes()!=b'':raise ValueError('actual stdout/exit mismatch')
    if 'ValueError: new/changed runtime module identity _frozen_importlib' not in paths[1].read_text():raise ValueError('actual failure marker')
    if any((OUT/name).exists() for name in ['FACTUAL_FIRST.jsonl','FACTUAL_EVIDENCE.json','COMPARISON_EXECUTION.json','FACTUAL_FIRST_FREEZE.json','QUALITY_MAP.json']):raise ValueError('unexpected completed stage output')
    p=OUT/'FAILED_FACTUAL_EXECUTION_CUSTODY.json'
    result={'actual_author_execution':'/root/adversarial_auditor','stage':'ACTUAL_FACTUAL_INVOCATION_FAILED_BEFORE_OWN_HELPERS',
        'invocations':1,'exit_code':1,'failure':'new/changed runtime module identity _frozen_importlib',
        'source_freeze_sha256':sha(freeze_raw),'source_sha256':sha((OUT/'factual_comparison_r2.py').read_bytes()),
        'config_sha256':sha((OUT/'RUN_CONFIG.json').read_bytes()),'grant_sha256':sha((ROOT/'quality_reports/RSM_ADVERSARIAL_CANONICAL_FACTUAL_ROOT_GRANT_ATTEMPT02.json').read_bytes()),
        'actual_command_flags':['-I','-B','-S'],'failure_files':files,'all_operative_current_raw_pairs_postverified':pairs,
        'operative_pair_count':len(pairs),'helper_imports_target_comparison_quality_votes':0,'factual_first_produced':False,
        'no_source_edit_guard_waiver_or_retry':True,'source_prep_runtime_identity_profile_not_accepted_under_authorized_flags':True,
        'failure_custody_source_sha256':sha(Path(__file__).read_bytes()),
        'model_disclosure':'same model/common public design/own earlier source and predictions reused; actual guard failure gives no dataset quality verdict',
        'stop_writes':True}
    with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'failure_custody_sha256':sha(p.read_bytes()),'postverified_pairs':len(pairs),'actual_exit_code':1,'no_retry':True,'stop_writes':True}))
if __name__=='__main__':main()
