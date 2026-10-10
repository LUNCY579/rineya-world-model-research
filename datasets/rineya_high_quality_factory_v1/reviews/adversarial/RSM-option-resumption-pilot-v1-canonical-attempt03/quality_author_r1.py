"""Mechanical expansion of an explicitly authored actual-agent quality ledger.
This source neither executes task semantics nor derives quality votes from checks.
External preexec/source/config anchors, exact raw pairs and runtime identities guard
every read. No designated receipt is written in this separately authorized stage.
"""
import argparse,datetime,hashlib,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ACTOR='/root/adversarial_auditor'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def exact_binding(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),'size_bytes':len(raw),'sha256':sha(raw)}
def verify_pairs(rows):
    bound={}
    for row in rows:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
        current=p.read_bytes();stored=(ROOT/row['archive']).read_bytes()
        if current!=stored or len(current)!=row['size_bytes'] or sha(current)!=row['sha256']:raise ValueError('quality exact current/raw pair '+str(p))
        if str(p) in bound:raise ValueError('duplicate current path')
        bound[str(p)]=row
    return bound
def verify_runtime(freeze,bound,entry):
    if str(Path(sys.executable).resolve())!=freeze['resolved_python']:raise ValueError('resolvedPython mismatch')
    if {k:getattr(sys.flags,k) for k in freeze['runtime_flags']}!=freeze['runtime_flags']:raise ValueError('runtime flags')
    actual=[]
    for name,module in sorted(sys.modules.items()):
        spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
        source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
        sp=str(Path(source).resolve()) if source else None;cp=str(Path(cache).resolve()) if cache else None
        ns=[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []
        identity={'origin':origin,'source_path':sp,'cache_path':cp,'namespace_paths':ns}
        if name=='__main__':
            if sp!=str(entry.resolve()) or origin is not None:raise ValueError('exact main identity')
        elif freeze['runtime_module_identities'].get(name)!=identity:raise ValueError('changed/new runtime identity '+name)
        for value in [sp,cp]:
            if value and Path(value).is_file():
                if value not in bound:raise ValueError('unbound concrete runtime '+name)
                b=bound[value];raw=Path(value).read_bytes()
                if len(raw)!=b['size_bytes'] or sha(raw)!=b['sha256']:raise ValueError('changed concrete runtime '+name)
            elif value==sp and value:raise ValueError('missing loaded source')
        actual.append({'module':name,**identity,'classification':'namespace' if ns else 'builtin' if origin=='built-in' else 'frozen' if origin=='frozen' else 'file-backed' if sp else 'no-concrete-file'})
    if str(Path(sys.executable).resolve()) not in bound:raise ValueError('unbound interpreter')
    return {'resolved_python':freeze['resolved_python'],'runtime_flags':freeze['runtime_flags'],'loaded_module_identities':actual,'unbound_concrete_modules':[],
        'boundary':'Reported concrete source/cache bytes exact; builtin/frozen code bound through interpreter. Namespace locations only; arbitrary dynamic in-memory code is not certified.'}
def archive_binding(path,prior):
    binding=exact_binding(path);p=Path(path).resolve()
    if str(p) in prior:
        row=prior[str(p)]
        if any(binding[k]!=row[k] for k in ['sha256','size_bytes']):raise ValueError('archive reuse drift')
        binding['archive']=row['archive'];binding['archive_reuse']='Existing exact own raw archive pointer'
    else:
        target=OUT/'source_archives'/(binding['sha256']+'.source');raw=p.read_bytes()
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('archive collision')
        else:
            with target.open('xb') as handle:handle.write(raw)
        binding['archive']=str(target.relative_to(ROOT))
    return binding
def write(name,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>25000000:raise ValueError('artifact size cap')
    with (OUT/name).open('xb') as handle:handle.write(raw)
    if (OUT/name).read_bytes()!=raw:raise ValueError('output roundtrip')
    return exact_binding(OUT/name)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-preexec-sha256',required=True)
    parser.add_argument('--expected-config-sha256',required=True);parser.add_argument('--expected-source-freeze-sha256',required=True)
    parser.add_argument('--expected-author-config-sha256',required=True)
    args=parser.parse_args();prepath=OUT/'QUALITY_AUTHOR_PREEXEC.json'
    if sha(prepath.read_bytes())!=args.expected_preexec_sha256:raise ValueError('external quality preexec')
    if sha((OUT/'RUN_CONFIG.json').read_bytes())!=args.expected_config_sha256:raise ValueError('external config')
    if sha((OUT/'SOURCE_FREEZE.json').read_bytes())!=args.expected_source_freeze_sha256:raise ValueError('external sourcefreeze')
    if sha((OUT/'QUALITY_AUTHOR_CONFIG.json').read_bytes())!=args.expected_author_config_sha256:raise ValueError('external author config')
    pre=load(prepath);sf=load(OUT/'SOURCE_FREEZE.json');config=load(OUT/'QUALITY_AUTHOR_CONFIG.json')
    before=verify_pairs(pre['files']);runtime_before=verify_runtime(sf,before,Path(__file__))
    if config['config_sha256']!=args.expected_config_sha256 or config['source_freeze_sha256']!=args.expected_source_freeze_sha256:raise ValueError('quality config anchors')
    for name,key in [('FACTUAL_FIRST.jsonl','factual_first_sha256'),('COMPARISON_EXECUTION.json','comparison_execution_sha256'),('FACTUAL_FIRST_FREEZE.json','factual_first_freeze_sha256')]:
        if sha((OUT/name).read_bytes())!=config[key]:raise ValueError('immutable factual binding')
    grant=ROOT/config['grant_path']
    if sha(grant.read_bytes())!=config['grant_sha256']:raise ValueError('original quality-stage authorization custody')
    ledger=load(OUT/config['ledger']);inspection=load(OUT/'MANUAL_ROW_INSPECTION.json')
    factual=[json.loads(line) for line in (OUT/'FACTUAL_FIRST.jsonl').read_text().splitlines()]
    if ledger['actual_author_execution']!=ACTOR or ledger['stage']!='MANUALLY_AUTHORED_AFTER_ACTUAL_CANONICAL_INSPECTION':raise ValueError('actual manually authored ledger')
    family={row['family_index']:row for row in ledger['family_reviews']}
    contexts=set('FRTPUAISX');scorekeys=set(load(OUT/'RUN_CONFIG.json')['score_keys'])
    if len(family)!=12 or set(ledger['context_profiles'])!=contexts:raise ValueError('complete manual family/context assessment')
    for value in family.values():
        if set(value['decisions'])!=contexts or any(d not in {'ACCEPT','REVISE','REJECT','QUARANTINE'} for d in value['decisions'].values()):raise ValueError('explicit manual votes')
    fj={row['sample_id']:row['record_hash'] for row in factual};ij={row['sample_id']:row['record_hash'] for row in inspection}
    if len(factual)!=216 or len(inspection)!=216 or len(fj)!=216 or ij!=fj:raise ValueError('actual exact ID/hash coverage')
    seen=set();manual=[]
    for row in inspection:
        f=family[row['family_index']];c=row['context'];profile=ledger['context_profiles'][c]
        if row['split']!=f['split']:raise ValueError('manually inspected split drift')
        key=(row['family_index'],c,row['renderer'])
        if key in seen or row['renderer'] not in {'forward','reverse'}:raise ValueError('style twin coverage')
        seen.add(key)
        if set(profile['scores'])!=scorekeys or any(type(x) is not int or not 1<=x<=5 for x in profile['scores'].values()):raise ValueError('explicit eight manual rubrics')
        manual.append({'sample_id':row['sample_id'],'record_hash':row['record_hash'],'scores':profile['scores'],
            'decision':f['decisions'][c],'issues':ledger['shared_issues'],'critical_defects':ledger['critical_defects'],'confidence':ledger['confidence'],
            'actual_assessment_rationale':profile['rationale'],'family_inspection_evidence':f['observed_program'],
            'assessment_scope':ledger['decision_policy'],'actual_inspected_context':c,'source_family_index':row['family_index'],'renderer':row['renderer']})
    if len(seen)!=216 or set((f,c) for f,c,s in seen)!={(f,c) for f in family for c in contexts}:raise ValueError('complete108 semantic task assessments')
    evidence=['ACTUAL_QUALITY_INSPECTION.md','MANUAL_QUALITY_LEDGER.json','MANUAL_ROW_INSPECTION.json','MANUAL_FAMILY_PROGRAMS.json',
        'OPERATIONAL_ARGUMENT_INSPECTION.json','MANUAL_EXTRACTION_EXECUTION.json','FACTUAL_EVIDENCE.json']
    evidence=[str((OUT/p).relative_to(ROOT)) for p in evidence]
    evidence += ['reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01/INPUT_DIVERSITY_SHORTCUT_AUDIT.json',
        'reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01/NUMERIC_SHORTCUT_EXECUTION.json']
    q=write('QUALITY_MAP.json',{'actual_author_execution':ACTOR,'quality_stage':'ACTUAL_AGENT_INSPECTION',
        'factual_first_sha256':config['factual_first_sha256'],'comparison_execution_sha256':config['comparison_execution_sha256'],
        'candidate':load(OUT/'FACTUAL_FIRST_FREEZE.json')['candidate'],'inspection_evidence':evidence,'model_disclosure':ledger['model_disclosure'],
        'decision_policy':ledger['decision_policy'],'actual_inspection_method':ledger['method'],'rows':manual})
    after=verify_pairs(pre['files']);runtime_after=verify_runtime(sf,after,Path(__file__))
    decisions={d:sum(row['decision']==d for row in manual) for d in ['ACCEPT','REVISE','REJECT','QUARANTINE']}
    execution=write('QUALITY_AUTHOR_EXECUTION.json',{'actor':ACTOR,'stage':'MANUAL_LEDGER_EXACT_ROW_EXPANSION_COMPLETED','rows':216,'semantic_tasks':108,'families':12,
        'manually_authored_decisions':decisions,'manual_scores_per_row':8,'quality_map':q,'preexec':exact_binding(prepath),
        'external_config_sha256':args.expected_config_sha256,'external_source_freeze_sha256':args.expected_source_freeze_sha256,
        'explicit_manual_ledger':exact_binding(OUT/config['ledger']),'all_exact_raw_pairs_before_after':True,'raw_pair_count':len(before),
        'runtime_before':runtime_before,'runtime_after':runtime_after,'new_semantic_helper_imports':0,
        'decisions_from_machinepasses':False,'designated_receipts_or_data_release_writes':False,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    files=list(pre['files'])
    for p in [prepath,OUT/'QUALITY_MAP.json',OUT/'QUALITY_AUTHOR_EXECUTION.json']:
        files.append(archive_binding(p,before))
    if len({str((Path(p['path']) if Path(p['path']).is_absolute() else ROOT/p['path']).resolve()) for p in files})!=len(files):raise ValueError('quality freeze duplicate paths')
    freeze=write('QUALITY_MAP_FREEZE.json',{'actual_author_execution':ACTOR,'stage':'ACTUAL_AGENT_QUALITY_MAP_FROZEN',
        'external_config_sha256':args.expected_config_sha256,'external_source_freeze_sha256':args.expected_source_freeze_sha256,
        'quality_map_path':q['path'],'quality_author_source_path':str(Path(__file__).relative_to(ROOT)),
        'quality_author_source_freeze_path':str(prepath.relative_to(ROOT)),'files':files,'no_designated_receipt_emission':True,
        'factual_first_freeze_sha256':config['factual_first_freeze_sha256'],'original_negative_evidence_preserved':True})
    verify_pairs(files);verify_runtime(sf,verify_pairs(files),Path(__file__))
    print(json.dumps({'quality_map':q,'execution':execution,'quality_map_freeze':freeze,'decisions':decisions,'raw_pairs':len(files),'no_designated_receipt':True}))
if __name__=='__main__':main()
