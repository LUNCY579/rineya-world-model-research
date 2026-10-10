"""UNEXECUTED prepared later receipt emitter: manual quality votes only.
Own stage source, not imported in source-only preparation. External grant and
quality/FIRST freezes are mandatory before any designated receipt can be written.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ACTOR='/root/adversarial_auditor'
"""Owned source block copied into both entry points; never imported as a helper.
Source-only preparation may inspect this text but executes no stage functions.
"""
import hashlib,json,sys
from pathlib import Path

def anchored_closure(root,out,args,phase):
    """External SHA anchors precede all owned semantic-helper imports.
    Every operative current/raw pair, actual interpreter, loaded concrete source
    and existing cache is byte/size bound. Names and reported origins are exact.
    Frozen/builtin code has a binary-bound boundary, not source execution proof.
    """
    def rawhash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if rawhash(out/'RUN_CONFIG.json')!=args.expected_config_sha256:raise ValueError('external config SHA mismatch')
    if rawhash(out/'SOURCE_FREEZE.json')!=args.expected_source_freeze_sha256:raise ValueError('external sourcefreeze SHA mismatch')
    if rawhash(args.grant)!=args.expected_grant_sha256:raise ValueError('external grant SHA mismatch')
    grant=json.loads(Path(args.grant).read_text())
    if grant.get('authorized_by')!='/root' or grant.get('phase')!=phase:raise ValueError('explicit stage execution grant')
    if grant.get('config_sha256')!=args.expected_config_sha256 or grant.get('source_freeze_sha256')!=args.expected_source_freeze_sha256:raise ValueError('grant anchors')
    freeze=json.loads((out/'SOURCE_FREEZE.json').read_text())
    actual_python=str(Path(sys.executable).resolve())
    if actual_python!=freeze['resolved_python']:raise ValueError('resolvedPython path mismatch')
    bound={};pairs=[]
    for row in freeze['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else root/p
        p=p.resolve();archive=root/row['archive'];current=p.read_bytes();stored=archive.read_bytes()
        if len(current)!=row['size_bytes'] or len(stored)!=row['size_bytes'] or current!=stored:raise ValueError('current/raw exactpair '+row['path'])
        if hashlib.sha256(current).hexdigest()!=row['sha256'] or hashlib.sha256(stored).hexdigest()!=row['sha256']:raise ValueError('current/raw SHA '+row['path'])
        if str(p) in bound:raise ValueError('duplicate operative resolved path')
        bound[str(p)]={'sha256':row['sha256'],'size_bytes':row['size_bytes']}
        pairs.append({'path':row['path'],'sha256':row['sha256'],'size_bytes':row['size_bytes'],'archive':row['archive']})
    if actual_python not in bound:raise ValueError('resolvedPython bytes unbound')
    expected=freeze['runtime_module_identities'];modules=[]
    for name,module in sorted(sys.modules.items()):
        spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
        source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
        source_path=str(Path(source).resolve()) if source else None
        cache_path=str(Path(cache).resolve()) if cache else None
        namespace=[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []
        identity={'origin':origin,'source_path':source_path,'cache_path':cache_path,'namespace_paths':namespace}
        if name=='__main__':
            if source_path not in freeze['allowed_entrypoint_paths'] or origin is not None:raise ValueError('unbound main entrypoint origin')
        elif name not in expected or expected[name]!=identity:raise ValueError('new/changed runtime module identity '+name)
        concrete=[]
        for kind,value in [('source_or_extension',source_path),('cache',cache_path)]:
            if not value:continue
            path=Path(value)
            if not path.is_file():
                if kind=='source_or_extension':raise ValueError('loaded file missing '+name)
                continue
            if value not in bound:raise ValueError('new/unbound concrete runtime '+name+' '+kind+' '+value)
            raw=path.read_bytes();actual={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
            if actual!=bound[value]:raise ValueError('loaded runtime SHA/size mismatch '+name+' '+kind)
            concrete.append({'kind':kind,'path':value,**actual})
        classification='namespace' if namespace else 'builtin' if origin=='built-in' else 'frozen' if origin=='frozen' else 'file-backed' if concrete else 'no-concrete-file'
        modules.append({'module':name,'classification':classification,**identity,'concrete_bindings':concrete,
            'boundary':'directory locations only; contents not certified' if namespace else 'bound interpreter handles builtin/frozen origin; reported concrete fields independently bound'})
    return {'external_config_sha256':args.expected_config_sha256,'external_source_freeze_sha256':args.expected_source_freeze_sha256,
        'external_grant_sha256':args.expected_grant_sha256,'resolved_python':actual_python,'all_exact_raw_pairs_verified':pairs,
        'loaded_runtime_modules':modules,'unbound_concrete_file_modules':[],'module_origin_identity_checks':True}

def anchor_arguments(parser):
    parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--expected-source-freeze-sha256',required=True)
    parser.add_argument('--grant',required=True)
    parser.add_argument('--expected-grant-sha256',required=True)
# Prepared separate emitter. No quality map exists or is authorized during prep.
def read_exact_json(path):
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=nonfinite)

def verify_later_freeze(root,path,external_sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=external_sha:raise ValueError('external later freeze SHA mismatch')
    freeze=read_exact_json(path);bindings={}
    for item in freeze['files']:
        current=Path(item['path']);current=current if current.is_absolute() else root/current
        archive=root/item['archive'];a=current.read_bytes();b=archive.read_bytes()
        if len(a)!=item['size_bytes'] or len(b)!=item['size_bytes'] or a!=b or hashlib.sha256(a).hexdigest()!=item['sha256'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('later exact raw pair mismatch')
        key=str(current.resolve())
        if key in bindings:raise ValueError('duplicate later current path')
        bindings[key]=item
    return freeze,bindings

def require_binding(bindings,path):
    key=str(Path(path).resolve())
    if key not in bindings:raise ValueError('later operative current path omitted '+key)
    return bindings[key]

def main():
    import argparse,datetime
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description='Later authorized copy of actual agent quality assessment only')
    anchor_arguments(parser)
    parser.add_argument('--expected-factual-first-freeze-sha256',required=True)
    parser.add_argument('--quality-map-freeze',required=True)
    parser.add_argument('--expected-quality-map-freeze-sha256',required=True)
    args=parser.parse_args()
    before=anchored_closure(ROOT,OUT,args,'RSM_ADVERSARIAL_RECEIPT_EMISSION')
    config=read_exact_json(OUT/'RUN_CONFIG.json');grant=read_exact_json(args.grant)
    if grant.get('factual_first_freeze_sha256')!=args.expected_factual_first_freeze_sha256 or grant.get('quality_map_freeze_sha256')!=args.expected_quality_map_freeze_sha256:raise ValueError('grant later anchors mismatch')
    # QUALITY_MAP_FREEZE provides full raw archive pairs for factual FIRST and
    # its execution, plus inspection evidence/manual quality map and author code.
    qfreeze,qbound=verify_later_freeze(ROOT,args.quality_map_freeze,args.expected_quality_map_freeze_sha256)
    if qfreeze.get('actual_author_execution')!=ACTOR or qfreeze.get('stage')!='ACTUAL_AGENT_QUALITY_MAP_FROZEN':raise ValueError('actual agent quality freeze identity')
    if qfreeze.get('external_config_sha256')!=args.expected_config_sha256 or qfreeze.get('external_source_freeze_sha256')!=args.expected_source_freeze_sha256:raise ValueError('quality source/config anchors')
    require_binding(qbound,ROOT/qfreeze['quality_author_source_path'])
    require_binding(qbound,ROOT/qfreeze['quality_author_source_freeze_path'])
    firstfreeze_path=OUT/'FACTUAL_FIRST_FREEZE.json'
    if hashlib.sha256(firstfreeze_path.read_bytes()).hexdigest()!=args.expected_factual_first_freeze_sha256:raise ValueError('immutable factual FIRST freeze mismatch')
    require_binding(qbound,firstfreeze_path);firstfreeze=read_exact_json(firstfreeze_path)
    if firstfreeze.get('actor')!=ACTOR or firstfreeze.get('stage')!='FACTUAL_FIRST_FROZEN_QUALITY_UNASSESSED':raise ValueError('factual FIRST identity')
    if firstfreeze.get('external_config_sha256')!=args.expected_config_sha256 or firstfreeze.get('external_source_freeze_sha256')!=args.expected_source_freeze_sha256:raise ValueError('factual source/config anchors')
    for item in firstfreeze['files']:
        p=ROOT/item['path'];later=require_binding(qbound,p)
        if any(later[k]!=item[k] for k in ['path','sha256','size_bytes']):raise ValueError('immutable FIRST file drift')
    for key in ['frame_first_binding','recursive_first_binding','candidate','original_negative_evidence']:
        item=firstfreeze[key];raw=(ROOT/item['path']).read_bytes()
        if len(raw)!=item['size_bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('immutable original binding drift')
    factual= [read_exact_json_line(line) for line in (OUT/'FACTUAL_FIRST.jsonl').read_text().splitlines()]
    comparison=read_exact_json(OUT/'COMPARISON_EXECUTION.json')
    if len(factual)!=216 or any(row.get('quality_assessment')!='UNASSESSED_PENDING_ACTUAL_AGENT_INSPECTION' or any(k in row for k in ['scores','decision']) for row in factual):raise ValueError('factual FIRST must remain unassessed')
    if comparison.get('actor')!=ACTOR or comparison.get('stage')!='FACTUAL_COMPARISON_ONLY' or comparison.get('no_quality_scores_decisions_or_designated_receipts') is not True:raise ValueError('comparison must remain factual')
    quality_path=ROOT/qfreeze['quality_map_path'];require_binding(qbound,quality_path);quality=read_exact_json(quality_path)
    if quality.get('actual_author_execution')!=ACTOR or quality.get('quality_stage')!='ACTUAL_AGENT_INSPECTION':raise ValueError('manual quality identity')
    if quality.get('factual_first_sha256')!=require_binding(qbound,OUT/'FACTUAL_FIRST.jsonl')['sha256'] or quality.get('comparison_execution_sha256')!=require_binding(qbound,OUT/'COMPARISON_EXECUTION.json')['sha256']:raise ValueError('quality does not bind actual FIRST/comparison')
    if type(quality.get('model_disclosure')) is not str or not quality['model_disclosure'].strip():raise ValueError('quality model/exposure disclosure')
    if type(quality.get('inspection_evidence')) is not list or not quality['inspection_evidence']:raise ValueError('actual inspection evidence required')
    for path in quality['inspection_evidence']:require_binding(qbound,ROOT/path)
    candidate=[read_exact_json_line(line) for line in (ROOT/config['candidate_path']).read_text().splitlines()]
    gold_join={row['sample_id']:row['record_hash'] for row in candidate};fact_join={row['sample_id']:row['record_hash'] for row in factual}
    if len(gold_join)!=216 or len(fact_join)!=216 or gold_join!=fact_join:raise ValueError('complete candidate/FIRST ID/hash join')
    manual=quality['rows'];manual_join={row['sample_id']:row['record_hash'] for row in manual}
    if len(manual)!=216 or len(manual_join)!=216 or manual_join!=gold_join:raise ValueError('complete actual quality ID/hash join')
    result=[]
    for row in manual:
        if set(row['scores'])!=set(config['score_keys']) or any(type(v) is not int or not 1<=v<=5 for v in row['scores'].values()):raise ValueError('eight actual rubric scores')
        if row['decision'] not in config['quality_decisions'] or type(row['issues']) is not list or type(row['critical_defects']) is not list:raise ValueError('actual quality fields')
        for issue in row['issues']:
            if type(issue) is not dict or not {'category','severity','evidence','recommended_fix'}<=set(issue):raise ValueError('actionable issue fields')
            if issue['severity'] not in {'low','moderate','high','critical'}:raise ValueError('actual issue severity')
            if not issue['evidence'] or not issue['recommended_fix']:raise ValueError('actual issue evidence/fix')
        confidence=row['confidence']
        if not ((type(confidence) in {int,float} and 0<=confidence<=1) or (type(confidence) is str and confidence in {'low','medium','high'})):raise ValueError('quality confidence')
        # No heuristic interpretation of machine checks determines this decision.
        receipt_id='adversarial-RSM-pilot-v1-'+row['record_hash'][:20]
        result.append({'receipt_id':receipt_id,'sample_id':row['sample_id'],'record_hash':row['record_hash'],
            'reviewer_role':'independent_adversarial','reviewer_execution_id':ACTOR,'scores':row['scores'],
            'critical_defects':row['critical_defects'],'issues':row['issues'],'decision':row['decision'],'confidence':confidence,
            'quality_map_sha256':require_binding(qbound,quality_path)['sha256'],'factual_first_freeze_sha256':args.expected_factual_first_freeze_sha256,
            'model_disclosure':quality['model_disclosure']})
    if len({row['receipt_id'] for row in result})!=216:raise ValueError('unique exact row receipt IDs')
    # Exact complete manual values are the only source of quality votes.
    payload=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in result).encode()
    if len(payload)>25000000:raise ValueError('receipt cap')
    before_write=anchored_closure(ROOT,OUT,args,'RSM_ADVERSARIAL_RECEIPT_EMISSION')
    verify_later_freeze(ROOT,args.quality_map_freeze,args.expected_quality_map_freeze_sha256)
    target=ROOT/config['final_receipt_path']
    with target.open('xb') as handle:handle.write(payload)
    after=anchored_closure(ROOT,OUT,args,'RSM_ADVERSARIAL_RECEIPT_EMISSION')
    verify_later_freeze(ROOT,args.quality_map_freeze,args.expected_quality_map_freeze_sha256)
    if target.read_bytes()!=payload:raise ValueError('receipt roundtrip')
    execution={'actor':ACTOR,'stage':'SEPARATELY_AUTHORIZED_MANUAL_RECEIPT_EMISSION','completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'rows':len(result),'receipt_path':config['final_receipt_path'],'receipt_sha256':hashlib.sha256(payload).hexdigest(),
        'closure_before':before,'closure_before_write':before_write,'closure_after':after,
        'quality_map_freeze_sha256':args.expected_quality_map_freeze_sha256,'no_automatic_quality_votes':True}
    with (OUT/'EMISSION_EXECUTION.json').open('x') as handle:json.dump(execution,handle,indent=2);handle.write('\n')
    print(json.dumps({'rows':len(result),'receipt_sha256':execution['receipt_sha256'],'manual_quality_only':True}))

def read_exact_json_line(line):
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(line,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

if __name__=='__main__':main()
