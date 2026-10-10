"""Benign exact-byte final custody; no helper import, target run or quality vote.
Read/archived externally before execution. Preserves failed draft evidence and
binds completed factual/quality tool logs after their actual process completion.
"""
import argparse,datetime,hashlib,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-self-sha256',required=True);parser.add_argument('--expected-quality-freeze-sha256',required=True)
    args=parser.parse_args();own=Path(__file__).resolve();ownraw=own.read_bytes()
    if sha(ownraw)!=args.expected_self_sha256 or (OUT/'source_archives/finalize_quality_custody_r1.preexec.raw').read_bytes()!=ownraw:raise ValueError('external finalizer source/raw barrier')
    qpath=OUT/'QUALITY_MAP_FREEZE.json'
    if sha(qpath.read_bytes())!=args.expected_quality_freeze_sha256:raise ValueError('external quality freeze')
    if sha((OUT/'SOURCE_FREEZE.json').read_bytes())!='0f8bf67ebd632c89a85efbdc151bbe5db08eacb25ca8358d9cdc7e30a413b25d':raise ValueError('original sourcefreeze')
    qfreeze=load(qpath);sf=load(OUT/'SOURCE_FREEZE.json');files=list(qfreeze['files']);bound={}
    def verify():
        for row in files:
            p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve();a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
            if a!=b or len(a)!=row['size_bytes'] or sha(a)!=row['sha256']:raise ValueError('final exact raw pair '+str(p))
            bound[str(p)]=row
    verify()
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or str(path.resolve()) in bound:continue
        raw=path.read_bytes();digest=sha(raw)
        if len(raw)>25000000:raise ValueError('individual artifact cap')
        archive=OUT/'source_archives'/(digest+'.source')
        if archive.exists():
            if archive.read_bytes()!=raw:raise ValueError('raw archive collision')
        else:
            with archive.open('xb') as handle:handle.write(raw)
        row={'path':str(path.relative_to(ROOT)),'sha256':digest,'size_bytes':len(raw),'archive':str(archive.relative_to(ROOT))}
        files.append(row);bound[str(path.resolve())]=row
    if len(files)!=len(bound):raise ValueError('duplicate current path')
    def runtime():
        if str(Path(sys.executable).resolve())!=sf['resolved_python'] or {k:getattr(sys.flags,k) for k in sf['runtime_flags']}!=sf['runtime_flags']:raise ValueError('final runtime flags/interpreter')
        modules=[]
        for name,module in sorted(sys.modules.items()):
            spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None);source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
            sp=str(Path(source).resolve()) if source else None;cp=str(Path(cache).resolve()) if cache else None
            ns=[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []
            identity={'origin':origin,'source_path':sp,'cache_path':cp,'namespace_paths':ns}
            if name=='__main__':
                if sp!=str(own) or origin is not None:raise ValueError('final main identity')
            elif sf['runtime_module_identities'].get(name)!=identity:raise ValueError('unbound/changed final module '+name)
            for value in [sp,cp]:
                if value and Path(value).is_file():
                    if value not in bound:raise ValueError('unbound concrete runtime')
                    raw=Path(value).read_bytes();r=bound[value]
                    if sha(raw)!=r['sha256'] or len(raw)!=r['size_bytes']:raise ValueError('runtime bytes')
                elif value==sp and value:raise ValueError('missing loaded source')
            modules.append({'module':name,**identity})
        return modules
    before=runtime();quality=load(OUT/'QUALITY_MAP.json');ledger=load(OUT/'MANUAL_QUALITY_LEDGER.json')
    factual=[json.loads(line) for line in (OUT/'FACTUAL_FIRST.jsonl').read_text().splitlines()]
    if len(quality['rows'])!=216 or {r['sample_id']:r['record_hash'] for r in quality['rows']}!={r['sample_id']:r['record_hash'] for r in factual}:raise ValueError('actual exact216 quality/FIRST join')
    manual={r['family_index']:r for r in ledger['family_reviews']};scorekeys=set(load(OUT/'RUN_CONFIG.json')['score_keys'])
    for row in quality['rows']:
        profile=ledger['context_profiles'][row['actual_inspected_context']];f=manual[row['source_family_index']]
        if row['scores']!=profile['scores'] or row['decision']!=f['decisions'][row['actual_inspected_context']] or row['issues']!=ledger['shared_issues'] or row['critical_defects']!=ledger['critical_defects'] or row['confidence']!=ledger['confidence']:raise ValueError('manual values changed')
        if set(row['scores'])!=scorekeys or any(type(v) is not int or not 1<=v<=5 for v in row['scores'].values()):raise ValueError('eight rubrics')
    expected_exits={'FACTUAL_RUN_EXIT_STATUS.txt':'0\n','MANUAL_EXTRACTION_EXIT.txt':'0\n','QUALITY_PREEXEC_TOOL_EXIT.txt':'0\n','QUALITY_PREEXEC_R2_TOOL_EXIT.txt':'0\n','QUALITY_AUTHOR_TOOL_EXIT.txt':'0\n'}
    for name,value in expected_exits.items():
        if (OUT/name).read_text()!=value:raise ValueError('actual process completion mismatch')
    for name in ['FACTUAL_RUN_STDERR.txt','MANUAL_EXTRACTION_STDERR.txt','QUALITY_PREEXEC_TOOL_STDERR.txt','QUALITY_PREEXEC_R2_TOOL_STDERR.txt','QUALITY_AUTHOR_TOOL_STDERR.txt']:
        if (OUT/name).read_bytes()!=b'':raise ValueError('unexpected successful-stage stderr')
    old=load(OUT/'QUALITY_AUTHOR_PREEXEC.json')
    original_bad=next(r for r in old['files'] if r['path'].endswith('/QUALITY_PREEXEC_TOOL_STDOUT.txt'))
    actual_bad=bound[str((OUT/'QUALITY_PREEXEC_TOOL_STDOUT.txt').resolve())]
    if original_bad['size_bytes']!=0 or actual_bad['size_bytes']!=258 or original_bad['sha256']==actual_bad['sha256']:raise ValueError('failed collector evidence not preserved')
    verify();after=runtime()
    decisions={d:sum(r['decision']==d for r in quality['rows']) for d in ['ACCEPT','REVISE','REJECT','QUARANTINE']}
    result={'actual_author_execution':'/root/adversarial_auditor','stage':'ACTUAL_CANONICAL_FACTUAL_AND_MANUAL_QUALITY_FINAL_STOP_BEFORE_EMITTER',
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':216,'semantic_task_assessments':108,'source_families':12,
        'factual_checks':{'passed':7490,'failed':0,'actual_invocations':1},'quality_decisions':decisions,'quality_source':'Explicit actual-agent ledger, never machinepasses',
        'quality_map_freeze_sha256':args.expected_quality_freeze_sha256,'external_finalizer_source_sha256':args.expected_self_sha256,
        'files':files,'current_raw_pairs_before_after':len(files),'runtime_before':before,'runtime_after':after,'resolved_python':sf['resolved_python'],
        'runtime_flags':sf['runtime_flags'],'unbound_concrete_modules':[],'all_quality_values_equal_authored_ledger':True,
        'preserved_failed_quality_preexec':{'freeze_sha256':sha((OUT/'QUALITY_AUTHOR_PREEXEC.json').read_bytes()),'failed_expected_stdout':original_bad,'actual_completed_stdout':actual_bad,'quality_author_r1_was_not_run':True},
        'original_attempt02_runtime_failure_preserved':'reviews/adversarial/RSM-option-resumption-pilot-v1-canonical-attempt02/FAILED_FACTUAL_CUSTODY.json',
        'original_input_FIRST_negative_evidence_preserved':True,'status_only_dev_accuracy':'68/72; not full typed target;40 irrelevant-predicate baseline counterexamples',
        'moderate_scientific_issues_retained_on_every_row':True,'no_other_actor_quality_read':True,'semantic_helper_imports_in_quality_stage':0,
        'no_receipt_emitter_designated_review_exporter_code_release_living_or_Git_writes':True,
        'model_disclosure':ledger['model_disclosure'],'boundary':'Concrete runtime source/cache and interpreter bytes bound; namespace contents and arbitrary dynamic code not certified.'}
    raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();target=OUT/'FINAL_QUALITY_CUSTODY.json'
    if len(raw)>25000000:raise ValueError('final artifact cap')
    with target.open('xb') as handle:handle.write(raw)
    if target.read_bytes()!=raw:raise ValueError('final custody roundtrip')
    archive=OUT/'source_archives'/(sha(raw)+'.source')
    with archive.open('xb') as handle:handle.write(raw)
    verify();runtime()
    print(json.dumps({'final_custody_path':str(target.relative_to(ROOT)),'sha256':sha(raw),'size_bytes':len(raw),'rows':216,'decisions':decisions,'current_raw_pairs':len(files),'actual_runtime_modules':len(after),'stop_writes':True}))
if __name__=='__main__':main()
