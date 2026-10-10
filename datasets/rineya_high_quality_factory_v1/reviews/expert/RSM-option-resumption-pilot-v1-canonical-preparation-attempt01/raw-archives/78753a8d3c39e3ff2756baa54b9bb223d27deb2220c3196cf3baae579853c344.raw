"""Hand-constructed review-evidence calibrations. Pure import, no execution.
No producer/formal fixtures or expected targets. Future execution only after
root's separate comparison grant and source/runtime/raw-data freeze.
"""
import copy
import json
from pathlib import Path

def run(solver,review,norm):
    checks=[]
    def check(name,holds,evidence=None):
        checks.append({'name':name,'passed':bool(holds),'evidence':evidence})
        if not holds:raise AssertionError(name)
    true={'all':[]}
    task={'kind':'option_resume','version':1,'initial_state':{'x':2,'y':2},'mode':'resume','event_budget':48,'entry':{'option':'start','args':{'p':{'reg':'x'}}},'history':[],
      'options':{'start':{'params':['p'],'initiation':copy.deepcopy(true),'success':copy.deepcopy(true),'body':[{'op':'call','option':'child','args':{'used':{'reg':'x'},'unused':{'const':9}}},{'op':'add','register':'y','value':{'param':'p'}}]},
       'child':{'params':['used','unused'],'initiation':copy.deepcopy(true),'success':copy.deepcopy(true),'body':[{'op':'if','test':copy.deepcopy(true),'then':[{'op':'add','register':'x','value':{'param':'used'}},{'op':'add','register':'y','value':{'const':1}}],'else':[]}]}}}
    # HAND snapshot: three committed events enter-start, enter-child, branch.
    cp={'state':{'x':2,'y':2},'next_instance_id':2,'root_pending':False,'phase':'running','frames':[
      {'frame_kind':'option','instance_id':0,'option':'start','bindings':{'p':2},'body_path':[],'next_index':1},
      {'frame_kind':'option','instance_id':1,'option':'child','bindings':{'used':2,'unused':9},'body_path':[],'next_index':1},
      {'frame_kind':'branch','instance_id':1,'option':'child','bindings':{'used':2,'unused':9},'body_path':[0,'then'],'next_index':0}]}
    task['checkpoint']=copy.deepcopy(cp)
    a=review.trusted_future(solver,task,cp)
    check('hand trusted continuation ends x4y5, five events',a['terminal_state']=={'x':4,'y':5} and a['terminal_phase']=='completed' and len(a['events'])==5,a)
    unused=review.future_dependency(solver,task,['frames',2,'bindings','unused'])
    check('NEGATIVE unused selected-frame binding not future dependency',not unused['holds'] and unused['future_differences']==[],unused)
    used=review.future_dependency(solver,task,['frames',2,'bindings','used'])
    check('POSITIVE saved operand changes pending first add and terminal x',used['holds'] and used['starting_snapshot_omitted'],used)
    altered=review.changed_checkpoint(cp,['frames',2,'bindings','used']);b=review.trusted_future(solver,task,altered)
    check('hand changed used2to3 first effect value3 statex5 and terminalx5y5',b['events'][0]['value']==3 and b['events'][0]['state']=={'x':5,'y':2} and b['terminal_state']=={'x':5,'y':5},b)
    check('negative demonstrably defeats old snapshot-inclusive comparator',cp!=review.changed_checkpoint(cp,['frames',2,'bindings','unused']) and not unused['holds'])
    over=copy.deepcopy(task);over['options']['child']['body'][0]['then'][0]={'op':'scale','register':'x','value':{'const':0}}
    statechange=review.future_dependency(solver,over,['state','x'])
    check('NEGATIVE state overwritten before first observed event not dependency',not statechange['holds'],statechange)
    cursor=review.future_dependency(solver,task,['frames',2,'next_index'])
    check('POSITIVE advanced cursor omits pending primitive',cursor['holds'],cursor)
    try:review.locate(cp,['frames',False,'next_index']);ok=False
    except ValueError:ok=True
    check('boolean locator index is rejected',ok)
    # Actual changed-pending-work calibration is derived only from the allowed
    # old input-only row, with original pending value3 manually established in
    # own input inspection; no stored production target is consulted.
    root=review.ROOT;inp=root/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl'
    actual=next(json.loads(x)['semantic_task'] for x in inp.read_text().splitlines() if json.loads(x)['sample_id']=='RSM-001-option-resumption-pilot-v1-p0012-R-forward')
    future=review.trusted_future(solver,actual,actual['checkpoint']);changed=review.changed_checkpoint(actual['checkpoint'],['frames',2,'bindings','r']);cf=review.trusted_future(solver,actual,changed)
    check('POSITIVE actual pending R r3to4 changes first add y5to6',future['events'][0]['value']==3 and future['events'][0]['state']=={'x':5,'y':5} and cf['events'][0]['value']==4 and cf['events'][0]['state']=={'x':5,'y':6} and not solver.typed_equal(future,cf))
    base={'control_id':'same id','name':'uninterrupted_resume_suffix','scope':'corpus_variant','pair_sample_ids':['left','right']}
    n,k=review.identifier(base);other=copy.deepcopy(base);other['name']='resume_restart_replay';_,k2=review.identifier(other)
    check('shared descriptive ID across different names remains distinct resolver key',k!=k2)
    alias=copy.deepcopy(base);alias.pop('name');alias['property']='uninterrupted_resume_suffix'
    check('property-only identifier resolves equal key',review.identifier(alias)==(n,k))
    both=copy.deepcopy(base);both['property']=both['name'];check('equal name/property legal',review.identifier(both)==(n,k))
    for name,change in [('boolean ID',{'control_id':True}),('unequal name/property',{'property':'resume_restart_replay'}),('duplicate member',{'pair_sample_ids':['left','left']})]:
        x=copy.deepcopy(base);x.update(change)
        try:review.identifier(x);ok=False
        except ValueError:ok=True
        check('negative '+name,ok)
    legacy={'kind':'skills','state':{'x':1,'y':2},'procedures':{'bump':[{'op':'add','register':'x','value':2}]},'main':[{'op':'call','name':'bump'}]}
    number=copy.deepcopy(legacy);number['procedures']['bump'][0]['value']=999;number['state']={'x':6,'y':-5}
    for view in ('full','coarse_all','coarse_reachable'):check('literal/state erasure '+view,norm.signature(legacy,view)['sha256']==norm.signature(number,view)['sha256'])
    padded=copy.deepcopy(legacy);padded['procedures']['unused']=[{'op':'scale','register':'y','value':3}]
    check('reachable view ignores unused padding',norm.signature(legacy,'coarse_reachable')['sha256']==norm.signature(padded,'coarse_reachable')['sha256'])
    check('all-declared view retains actual unused procedure graph',norm.signature(legacy,'coarse_all')['sha256']!=norm.signature(padded,'coarse_all')['sha256'])
    r={'sample_id':'toy','record_hash':'ignored','family_control_claims':[{'nested_hash':'old'}]};h=review.record_hash(r);r['record_hash']='changed';check('top-level record hash excluded',review.record_hash(r)==h)
    r['family_control_claims'][0]['nested_hash']='new';check('nested hash is never excluded from full-record hash',review.record_hash(r)!=h)
    return {'actual_author_execution_id':'/root/expert_reviewer','checks':checks,'passed':len(checks),'failed':0,'handmade_and_input_derived':True,'production_targets_read':False,'common_model_same_author':True}
