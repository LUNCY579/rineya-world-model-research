"""Serialize the actual agent's already-authored rubric; no semantic solver/vote inference."""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

ROOT = Path.cwd().resolve()
PREFIX = Path('reviews/adjudication/RSM-option-resumption-pilot-v1-canonical-baseline-preparation-attempt01')
ROLE = '/root/research_architect'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def raw(path):
    return (ROOT / path).read_bytes()

def write_new(path, value, lines=False):
    data = (''.join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n' for x in value)
            if lines else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
    if len(data) >= 25*1024*1024:
        raise ValueError('output exceeds compact-file bound')
    with (ROOT/path).open('xb') as handle:
        handle.write(data)
    return {'path':str(path),'sha256':sha(data),'bytes':len(data)}

def guard(bindings):
    checked=[]
    for item in bindings:
        actual=raw(item['path']); archived=raw(item['archive'])
        if len(actual)!=item['bytes'] or sha(actual)!=item['sha256'] or actual!=archived:
            raise ValueError('current/raw binding changed: '+item['path'])
        checked.append({'path':item['path'],'archive':item['archive'],'sha256':item['sha256'],'bytes':item['bytes'],'matches':True})
    return checked

def runtime_guard(bindings, resolved):
    if str(Path(sys.executable).resolve()) != resolved:
        raise ValueError('resolved interpreter mismatch')
    paths={str((ROOT/i['path']).resolve()):i for i in bindings}
    out=[]
    for name,module in sorted(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        if not origin:
            continue
        path=str(Path(origin).resolve())
        if path not in paths:
            raise ValueError('unfrozen registered file module: '+name+' '+path)
        item=paths[path]
        if sha(Path(path).read_bytes())!=item['sha256']:
            raise ValueError('registered module bytes changed: '+name)
        out.append({'module':name,'path':path,'sha256':item['sha256'],'bytes':item['bytes']})
    return out

def run():
    started=now()
    if len(sys.argv)!=5 or sys.argv[1]!='--expected-freeze' or sys.argv[3]!='--expected-policy':
        raise ValueError('mandatory external freeze/policy hashes missing')
    freeze_path=PREFIX/'QUALITY_SOURCE_PREEXECUTION-r1.json'
    fb=raw(freeze_path)
    if sha(fb)!=sys.argv[2]:
        raise ValueError('external freeze hash mismatch')
    freeze=json.loads(fb)
    policy_path=PREFIX/'ACTUAL_AGENT_QUALITY_POLICY-r1.json'
    pb=raw(policy_path)
    if sha(pb)!=sys.argv[4] or sha(pb)!=freeze['policy_sha256']:
        raise ValueError('external authored-policy hash mismatch')
    base_bytes=raw(freeze['base_source_freeze_path'])
    if sha(base_bytes)!=freeze['base_source_freeze_sha256']:
        raise ValueError('original source freeze changed')
    base=json.loads(base_bytes)
    bindings=base['bindings']+freeze['bindings']
    before=guard(bindings)
    runtime_before=runtime_guard(bindings,freeze['resolved_python'])
    policy=json.loads(pb)
    if policy['actual_execution_id']!=ROLE or policy['stage']!='ACTUAL_AGENT_CANONICAL_FIRST_QUALITY_POLICY':
        raise ValueError('authored policy role/stage mismatch')
    config=json.loads(raw(PREFIX/'quality-serializer-config-r1.json'))
    if sha(raw(PREFIX/'quality-serializer-config-r1.json'))!=freeze['config_sha256']:
        raise ValueError('serializer config mismatch')
    rows=[json.loads(x) for x in raw(config['candidate']).splitlines()]
    joins={x['sample_id']:x for x in [json.loads(line) for line in raw(config['factual_join']).splitlines()]}
    initials={x['sample_id']:x for x in [json.loads(line) for line in raw(config['input_initials']).splitlines()]}
    if len(rows)!=216 or len(joins)!=216 or len(initials)!=216 or len({r['sample_id'] for r in rows})!=216:
        raise ValueError('exact 216 row/first/join coverage mismatch')
    factual=json.loads(raw(PREFIX/'FACTUAL_FIRST-r1.json'))
    if factual['exact_TYPED_target_matches']!=216 or not factual['original_INPUT_FIRST_predictions_unchanged']:
        raise ValueError('own factual FIRST incomplete')
    by_family={}
    for row in rows:
        by_family.setdefault((row['source_family_key'],row['renderer']),{})[row['controlled_context']]=row
    evidence=[]; judgments=[]
    rubric={'semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','difficulty_quality','diversity','shortcut_resistance'}
    for n,row in enumerate(rows,1):
        sid=row['sample_id']; join=joins[sid]; initial=initials[sid]
        if not join['typed_target_match'] or not join['FIRST_prediction_unchanged'] or join['record_hash']!=row['record_hash'] or initial['record_hash']!=row['record_hash']:
            raise ValueError('immutable exact row join mismatch: '+sid)
        context=row['controlled_context']
        if initial['evidence']['input_role_derived_from_semantics']!=context:
            raise ValueError('canonical context differs from own frozen semantic role: '+sid)
        authored=policy['context_assessments'][context]
        scores=authored['scores']
        if set(scores)!=rubric or any(type(v) is not int or v<1 or v>5 for v in scores.values()):
            raise ValueError('authored eight-rubric score shape invalid')
        fam=by_family[(row['source_family_key'],row['renderer'])]
        target=row['verified_target']; cp=target['checkpoint']; state=cp['state'] if cp else None
        ev={'sample_id':sid,'record_hash':row['record_hash'],'source_family_key':row['source_family_key'],'source_family_index':row['source_family_index'],'semantic_family_id':row['semantic_family_id'],'semantic_parent_id':row['semantic_parent_id'],'context':context,'renderer':row['renderer'],'split':row['split'],'difficulty':row['difficulty'],'typed_gold_sha256':join['actual_gold_sha256'],'unchanged_FIRST_expected_sha256':join['FIRST_expected_sha256'],'target_status':target['status'],'terminal_or_paused_state':state,'new_event_count':len(target['new_events']),'new_effect_count':target['new_effects'],'previous_effect_count':target['previous_effects'],'family_F_sample_id':fam['F']['sample_id'],'family_R_sample_id':fam['R']['sample_id'],'family_T_sample_id':fam['T']['sample_id'],'F_state':fam['F']['verified_target']['checkpoint']['state'],'T_state':fam['T']['verified_target']['checkpoint']['state'],'program_motif':policy['family_motifs'][str(row['source_family_index'])],'property_witnesses':row['property_witnesses'],'resolved_claim_identifiers':[x.get('name',x.get('property')) for x in row['family_control_claims']],'prior_INITIAL_decision':initial['decision']}
        evidence.append(ev)
        judgments.append({'receipt_id':'research_architect-own-canonical-quality-r1:'+sid,'sample_id':sid,'record_hash':row['record_hash'],'reviewer_role':'independent_adjudicator_own_canonical_baseline','reviewer_execution_id':ROLE,'stage':'OWN_CANONICAL_FIRST_QUALITY_BASELINE','scores':scores,'critical_defects':policy['critical_defects'],'issues':policy['issues'],'decision':authored['decision'],'confidence':authored['confidence'],'reason':authored['reason']+' Actual program: '+ev['program_motif']+'. Status '+str(target['status'])+'; state '+str(state)+'; '+str(len(target['new_events']))+' new events/'+str(target['new_effects'])+' effects. Scope limited by explicit cohort issues.','evidence':{'ledger':str(PREFIX/'quality-row-evidence-r1.jsonl'),'line':n,'inspection':str(PREFIX/'ACTUAL_AGENT_CANONICAL_INSPECTION-r1.md'),'policy_sha256':sha(pb),'factual_join':config['factual_join'],'factual_FIRST':str(PREFIX/'FACTUAL_FIRST-r1.json')},'target_correctness':'VERIFIED_AGAINST_IMMUTABLE_INPUT_FIRST','actual_agent_authored_quality':True,'design_author_adjudicator_overlap':True,'same_model_correlation_risk':True,'other_reviews_read':False,'data_approval':False,'publication_eligible':False})
    outputs=[write_new(PREFIX/'quality-row-evidence-r1.jsonl',evidence,True),write_new(PREFIX/'canonical-first-quality-judgments-r1.jsonl',judgments,True)]
    counts=dict(Counter(x['decision'] for x in judgments))
    runtime_after=runtime_guard(bindings,freeze['resolved_python'])
    after=guard(bindings)
    if before!=after:
        raise ValueError('custody changed during serialization')
    execution={'actual_execution_id':ROLE,'stage':'ACTUAL_AGENT_AUTHORED_QUALITY_SERIALIZATION','completed_execution':True,'started_utc':started,'completed_utc':now(),'semantic_solver_or_calibration_or_target_comparison_rerun':False,'decision_counts':counts,'all_216_quality_assessments_authored_before_other_review_access':True,'machine_PASS_did_not_choose_votes':True,'source_freeze_sha256':sha(fb),'policy_sha256':sha(pb),'config_sha256':freeze['config_sha256'],'bindings_before':before,'bindings_after':after,'all_current_and_raw_unchanged':True,'runtime_modules_before':runtime_before,'runtime_modules_after':runtime_after,'other_reviews_or_private_formal_evidence_read':False,'data_approval':False,'publication_eligible':False,'whole_cohort_publication_eligible':False,'outputs':outputs}
    execution_binding=write_new(PREFIX/'QUALITY_EXECUTION-r1.json',execution)
    outputs.append(execution_binding)
    first={'actual_execution_id':ROLE,'stage':'IMMUTABLE_OWN_CANONICAL_QUALITY_FIRST','completed_execution':True,'at_utc':now(),'rows':216,'decision_counts':counts,'eight_rubric_scores_per_record':True,'original_INPUT_FIRST_and_INITIALs_unchanged':True,'source_freeze_sha256':sha(fb),'authored_policy_sha256':sha(pb),'bindings':outputs,'other_review_predictions_verdicts_findings_read':False,'data_approval':False,'publication_eligible':False,'whole_cohort_publication_eligible':False,'reason':'Own quality baseline is frozen before other actual quality votes and final evidence adjudication.'}
    first_binding=write_new(PREFIX/'QUALITY_FIRST-r1.json',first)
    print(json.dumps({'completed_execution':True,'decision_counts':counts,'quality_FIRST':first_binding,'current_raw_bindings':len(bindings),'publication_eligible':False},sort_keys=True))

if __name__=='__main__':
    try:
        run()
    except Exception as exc:
        failure={'actual_execution_id':ROLE,'completed_execution':False,'at_utc':now(),'error_type':type(exc).__name__,'error':str(exc),'data_approval':False,'publication_eligible':False,'no_silent_retry':True}
        write_new(PREFIX/'QUALITY_SERIALIZATION_FIRST_FAILURE-r1.json',failure)
        raise
