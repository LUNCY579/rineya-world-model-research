"""Actual Expert canonical comparison, source-prepared only until root grant.
Compare immutable input-first outputs; never replace predictions using gold.
No producer/reference/formal source import. Claims are proved from real pairs.
Final receipts need an explicitly authored/frozen actual-agent quality map;
absence never becomes automatic ACCEPT. Software checks alone are insufficient.
"""
import argparse
import collections
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import resource
import sys
import time
import traceback

VERSION='expert-rsm-canonical-1.0.1'
ACTOR='/root/expert_reviewer'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CONTEXTS=('F','R','T','P','U','A','I','S','X')
PAIRS={
'uninterrupted_resume_suffix':('F','R'),'resume_restart_replay':('R','T'),
'budget_pause_after_effect':('R','P'),'second_checkpoint_suffix':('P','U'),
'positive_bound_argument_counterfactual':('F','A'),'initiation_argument_counterfactual':('F','I'),
'success_contract_counterfactual':('F','S'),'active_checkpoint_tamper_rejection':('R','X')}
NAMES=set(PAIRS)|{'object_field_order_invariance'}
SCORES=('semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','difficulty_quality','diversity','shortcut_resistance')

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def record_hash(r):return sha(canonical({k:v for k,v in r.items() if k!='record_hash'}).encode())
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def safe_relative(name):
    if type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe factory-relative path')
    p=ROOT
    for part in Path(name).parts:
        p=p/part
        if p.is_symlink():raise ValueError('symlink source component')
    return p

def identifier(entry):
    ns=[entry[k] for k in ('name','property') if k in entry]
    if not ns or any(type(n) is not str or n not in NAMES for n in ns) or len(set(ns))!=1:raise ValueError('invalid name/property identifier')
    if type(entry.get('control_id')) is not str or not entry['control_id']:raise ValueError('control_id is not nonempty string')
    ids=entry.get('pair_sample_ids')
    if type(ids) is not list or len(ids)!=2 or any(type(x) is not str for x in ids) or ids[0]==ids[1]:raise ValueError('ordered distinct real member IDs required')
    return ns[0],(entry['control_id'],ns[0],tuple(ids))

def diffs(solver,a,b,path=()):
    if type(a) is not type(b):return [{'path':list(path),'left':a,'right':b}]
    if type(a) is dict:
        if set(a)!=set(b):return [{'path':list(path),'left':a,'right':b}]
        return sum((diffs(solver,a[k],b[k],path+(k,)) for k in a),[])
    if type(a) is list:
        if len(a)!=len(b):return [{'path':list(path),'left':a,'right':b}]
        return sum((diffs(solver,x,y,path+(i,)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [{'path':list(path),'left':a,'right':b}]

def trusted_future(solver,task,checkpoint):
    """Diagnostic only; deliberately bypass snapshot validation. OMIT starting
    snapshots/frames/bindings, so unused field changes are not credited.
    Returns future semantic events, effect after-states and terminal status.
    """
    m=copy.deepcopy(checkpoint);events=[]
    for _ in range(256):
        if m['phase']!='running':break
        events.append(solver.step(m,task))
    if m['phase']=='running':raise ValueError('trusted diagnostic cap exceeded')
    return {'events':events,'terminal_state':m['state'],'terminal_phase':m['phase']}

def locate(value,path):
    for k in path:
        if type(value) is list:
            if type(k) is not int or not 0<=k<len(value):raise ValueError('typed list locator required')
        elif type(value) is dict:
            if type(k) is not str or k not in value:raise ValueError('literal actual object-key locator required')
        else:raise ValueError('locator descends into scalar')
        value=value[k]
    return value

def changed_checkpoint(cp,path):
    old=locate(cp,path)
    if type(old) is not int:raise ValueError('active cohort locator must resolve exact integer scalar')
    out=copy.deepcopy(cp);parent=locate(out,path[:-1]);parent[path[-1]]=old+1;return out

def future_dependency(solver,task,path):
    cp=task['checkpoint'];changed=changed_checkpoint(cp,path)
    a=trusted_future(solver,task,cp);b=trusted_future(solver,task,changed)
    return {'holds':not solver.typed_equal(a,b),'locator':path,'before':locate(cp,path),'after':locate(changed,path),
            'future_differences':diffs(solver,a,b),'starting_snapshot_omitted':True}

def saved_origin(solver,task,trace,locator):
    """Trace actual retained argument to a register evaluation, including
    transitively passed caller parameters. No private naming assumption.
    """
    if len(locator)!=4 or locator[0]!='frames' or locator[2]!='bindings':return False
    frame=task['checkpoint']['frames'][locator[1]];parameter=locator[3]
    enters={e['instance_id']:e for e in trace['events'] if e['event']=='enter'}
    def origin(instance,p):
        enter=enters[instance];site=enter['call_site']
        if site is None:expr=task['entry']['args'][p]
        else:
            caller=enters[site['caller_instance']]
            ins=instruction_at(solver,task,caller['option'],site['body_path'],site['index'])
            expr=ins['args'][p]
        if 'reg' in expr:return True
        if 'const' in expr:return False
        return site is not None and origin(site['caller_instance'],expr['param'])
    return origin(frame['instance_id'],parameter)

def instruction_at(solver,task,option,path,index):return solver.body_at(task,{'option':option,'body_path':path})[index]

def replay_witnesses(solver,task,original_trace,restart_events,locator=None):
    out=[]
    for k,e in enumerate(original_trace['events']):
        if e['event']!='effect':continue
        loc={'option':e['option'],'body_path':e['body_path'],'index':e['index']}
        if locator is not None and not solver.typed_equal(loc,locator):continue
        actual=[n for n in restart_events if n['event']=='effect' and all(solver.typed_equal(n[q],loc[q]) for q in loc)]
        if not actual:continue
        snap=original_trace['snapshots'][k];bindings=next(f['bindings'] for f in reversed(snap['frames']) if f['instance_id']==e['instance_id'])
        ins=instruction_at(solver,task,e['option'],e['body_path'],e['index'])
        def g(s):
            v=copy.deepcopy(s)
            if ins['op']=='swap':v={'x':s['y'],'y':s['x']}
            else:
                a=solver.evaluate(ins['value'],s,bindings);reg=ins['register'];v[reg]=s[reg]+a if ins['op']=='add' else s[reg]*a
            return v
        a=g(snap['state']);b=g(a)
        if not solver.typed_equal(a,b):out.append({'locator':loc,'original_before':snap['state'],'retained_original_bindings':bindings,'once':a,'twice':b,'actual_T_effects':actual})
    return out

def relation(solver,name,left,right,pred,facts,calibration):
    a,b=left['semantic_task'],right['semantic_task'];pa,pb=pred[left['sample_id']]['expected_output'],pred[right['sample_id']]['expected_output']
    ds=diffs(solver,a,b);detail={'typed_task_differences':ds};holds=False
    if name=='object_field_order_invariance':
        ao=solver.strict_json(left['rendered_input'].split('Task JSON:\n')[1]);bo=solver.strict_json(right['rendered_input'].split('Task JSON:\n')[1])
        holds=solver.typed_equal(ao,bo) and solver.typed_equal(pa,pb) and canonical(calibration.reversed_fields(ao))==canonical(bo)
        # canonical equality alone discards order; compare raw parsed insertion order serialization as well.
        holds=holds and json.dumps(calibration.reversed_fields(ao),ensure_ascii=False,separators=(',',':'))==json.dumps(bo,ensure_ascii=False,separators=(',',':'))
    elif name=='uninterrupted_resume_suffix':
        allowed={'mode','history','checkpoint'};same=solver.typed_equal({k:v for k,v in a.items() if k not in allowed},{k:v for k,v in b.items() if k not in allowed})
        holds=same and a['mode']=='fresh' and b['mode']=='resume' and len(b['history'])>0 and pb['previous_effects']>=1 and len(pb['new_events'])>0 and pb['status']==pa['status']=='completed' and solver.typed_equal(b['history']+pb['new_events'],pa['new_events']) and solver.typed_equal(pa['checkpoint'],pb['checkpoint'])
    elif name=='resume_restart_replay':
        w=replay_witnesses(solver,a,facts[left['sample_id']]['original_full_run'],pb['new_events']);detail['actual_nonidempotent_replay_witnesses']=w
        holds=len(ds)==1 and ds[0]['path']==['mode'] and a['mode']=='resume' and b['mode']=='restart' and bool(pb['new_events']) and pb['new_events'][0]['initiation_ok'] and bool(w) and not solver.typed_equal(pa['new_events'],pb['new_events']) and (pa['status']!=pb['status'] or not solver.typed_equal(pa['checkpoint']['state'],pb['checkpoint']['state']))
    elif name=='budget_pause_after_effect':holds=len(ds)==1 and ds[0]['path']==['event_budget'] and 0<b['event_budget']<a['event_budget'] and pb['status']=='paused' and pb['new_effects']>=1 and pb['next_event'] is not None and solver.typed_equal(pb['new_events'],pa['new_events'][:len(pb['new_events'])])
    elif name=='second_checkpoint_suffix':
        same=solver.typed_equal({k:v for k,v in a.items() if k not in ('event_budget','history','checkpoint')},{k:v for k,v in b.items() if k not in ('event_budget','history','checkpoint')})
        original=facts[left['sample_id']]['original_full_run'];holds=same and b['mode']=='resume' and b['event_budget']==48 and solver.typed_equal(b['history'],a['history']+pa['new_events']) and solver.typed_equal(b['checkpoint'],pa['checkpoint']) and solver.typed_equal(b['history']+pb['new_events'],original['events']) and solver.typed_equal(pb['checkpoint'],original['checkpoint'])
    elif name=='positive_bound_argument_counterfactual':
        # Later actual primitive/branch/return events must differ, not merely a new enter binding.
        later_a=[e for e in pa['new_events'] if e['event']!='enter'];later_b=[e for e in pb['new_events'] if e['event']!='enter']
        detail['downstream_nonenter_differences']=diffs(solver,later_a,later_b)
        holds=len(ds)==1 and ds[0]['path'][:2]==['entry','args'] and pa['status']==pb['status']=='completed' and bool(detail['downstream_nonenter_differences'])
    elif name=='initiation_argument_counterfactual':
        root=a['entry']['option'];definition=a['options'][root]
        ba={p:solver.evaluate(e,a['initial_state'],{}) for p,e in a['entry']['args'].items()};bb={p:solver.evaluate(e,b['initial_state'],{}) for p,e in b['entry']['args'].items()}
        va=[solver.test({'all':[x]},a['initial_state'],ba) for x in definition['initiation']['all']];vb=[solver.test({'all':[x]},b['initial_state'],bb) for x in definition['initiation']['all']]
        detail.update(F_initiation_atoms=va,I_initiation_atoms=vb)
        holds=len(ds)==1 and ds[0]['path'][:2]==['entry','args'] and pa['status']=='completed' and pb['status']=='initiation_failed' and pb['new_effects']==0 and all(va) and sum(not x for x in vb)==1
    elif name=='success_contract_counterfactual':
        root=a['entry']['option'];fe=[e for e in pa['new_events'] if e['event']=='effect'];se=[e for e in pb['new_events'] if e['event']=='effect'];bind=pa['new_events'][0]['bindings']
        va=[solver.test({'all':[x]},pa['checkpoint']['state'],bind) for x in a['options'][root]['success']['all']];vb=[solver.test({'all':[x]},pb['checkpoint']['state'],bind) for x in b['options'][root]['success']['all']]
        detail.update(F_success_atoms=va,S_success_atoms=vb)
        holds=len(ds)==1 and ds[0]['path'][:3]==['options',root,'success'] and ds[0]['path'][-1]=='const' and pa['status']=='completed' and pb['status']=='success_failed' and pa['new_events'][-1]['instance_id']==pb['new_events'][-1]['instance_id']==0 and solver.typed_equal(fe,se) and all(va) and sum(not x for x in vb)==1
    elif name=='active_checkpoint_tamper_rejection':
        observable_a=trusted_future(solver,a,a['checkpoint']);observable_b=trusted_future(solver,b,b['checkpoint'])
        detail['future_observable_differences']=diffs(solver,observable_a,observable_b);detail['starting_snapshot_omitted']=True
        path=ds[0]['path'] if len(ds)==1 else [];allowed=path[:2]==['checkpoint','state'] or (path[:2]==['checkpoint','frames'] and (path[-1]=='next_index' or 'bindings' in path))
        holds=len(ds)==1 and allowed and type(ds[0]['left']) is int and type(ds[0]['right']) is int and pb['error']=='invalid_checkpoint' and pb['new_events']==[] and bool(detail['future_observable_differences'])
    return {'name':name,'sample_ids':[left['sample_id'],right['sample_id']],'holds':bool(holds),'evidence':detail}

def write_json(name,data):
    p=HERE/name
    with p.open('x') as f:json.dump(data,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')
    return {'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
def write_jsonl(path,rs):
    with path.open('x') as f:
        for r in rs:f.write(canonical(r)+'\n')
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'rows':len(rs)}

def execute_compare():
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter()
    cfg=json.loads((HERE/'config.json').read_text());freeze=json.loads((HERE/'SOURCE-PREPARATION-FREEZE.json').read_text())
    for item in freeze['bindings']:
        p=Path(item['absolute_path']);assert sha(p.read_bytes())==item['sha256'],item['path']
        assert (HERE/item['archive_path']).read_bytes()==p.read_bytes()
    for item in freeze['runtime']['file_bindings']:assert sha(Path(item['absolute_path']).read_bytes())==item['sha256']
    solver=load_module('expert_own_rsm_solver',ROOT/cfg['own_solver'])
    cal=load_module('expert_own_rsm_calibration',ROOT/cfg['own_calibration'])
    norms=load_module('expert_own_rsm_normalization',HERE/'own_normalization.py')
    toys=load_module('expert_own_new_review_calibration',HERE/'new_calibrations.py')
    toy_results=toys.run(solver,sys.modules[__name__],norms)
    toy_output=write_json('new-calibration-results.json',toy_results)
    input_rows=[solver.strict_json(x) for x in (ROOT/cfg['input']).read_text().splitlines()]
    rows=[solver.strict_json(x) for x in (ROOT/cfg['candidate']).read_text().splitlines()]
    pred={x['sample_id']:x for x in map(solver.strict_json,(ROOT/cfg['first_predictions']).read_text().splitlines())}
    facts={x['sample_id']:x for x in map(solver.strict_json,(ROOT/cfg['first_facts']).read_text().splitlines())}
    parents=[solver.strict_json(x) for x in (ROOT/cfg['parents']).read_text().splitlines()]
    legacy=[solver.strict_json(x) for x in (ROOT/cfg['legacy']).read_text().splitlines()]
    manifest=solver.strict_json((ROOT/cfg['manifest']).read_text());spec=solver.strict_json((ROOT/cfg['public_config']).read_text());contract=(ROOT/cfg['contract']).read_text()
    candidate_sha=sha((ROOT/cfg['candidate']).read_bytes());by_id={r['sample_id']:r for r in rows};inputs={r['sample_id']:r for r in input_rows};parentmap={r['semantic_parent_id']:r for r in parents}
    batch_checks=[];rowchecks={sid:[] for sid in by_id};evidence=[]
    def check(name,holds,detail=None,sids=None,severity='high',category='semantic_correctness'):
        x={'name':name,'holds':bool(holds),'evidence':detail,'severity':severity,'category':category};batch_checks.append(x)
        for sid in (list(by_id) if sids is None else sids):rowchecks[sid].append(x)
        return bool(holds)
    check('216 actual exact unique candidate and first IDs',len(rows)==216 and len(by_id)==216 and set(by_id)==set(inputs)==set(pred)==set(facts))
    check('108 unique semantic parents',len(parents)==len(parentmap)==108)
    schema=solver.strict_json((ROOT/cfg['canonical_schema']).read_text())
    families=collections.defaultdict(dict)
    required_pub=set(cfg['public_sources'])
    check('manifest metadata version strict integer1',type(manifest.get('metadata_version')) is int and manifest['metadata_version']==1)
    check('manifest exact pending candidate/batch binding',manifest.get('batch')=='option-resumption-pilot-v1' and manifest.get('candidate_sha256')==candidate_sha)
    bindings=manifest.get('source_bindings',{})
    check('six operative public source hashes supplied',type(bindings) is dict and required_pub<=set(bindings))
    opaque=set(cfg['opaque_provenance_paths'])
    for name,claimed in bindings.items():
        try:p=safe_relative(name);actual=sha(p.read_bytes());h=type(claimed) is str and bool(re.fullmatch('[0-9a-f]{64}',claimed)) and actual==claimed
        except (ValueError,OSError):actual=None;h=False
        check('truthful actual manifest source binding '+name,h,{'asserted':claimed,'actual':actual,'scope':'OPAQUE_NOT_SEMANTICALLY_INSPECTED_NOT_IMPORTED' if name in opaque else 'public/own/nonprivate supplied source'})
    for r in rows:
        sid=r['sample_id'];sids=[sid];t=r['semantic_task'];check('canonical required fields',set(schema['required'])<=set(r),sids=sids)
        check('full top-level-only record hash exclusion',record_hash(r)==r['record_hash'],{'actual':record_hash(r),'asserted':r['record_hash']},sids)
        check('exact eight-field stripped projection',solver.typed_equal({k:r[k] for k in input_rows[0]},inputs[sid]),sids=sids)
        check('exact immutable typed first prediction versus canonical gold',solver.typed_equal(pred[sid]['expected_output'],r['verified_target']),{'typed_differences':diffs(solver,pred[sid]['expected_output'],r['verified_target'])},sids,category='target_correctness')
        try:solver.parse_prompt(r['rendered_input'],t,contract);ok=True
        except (ValueError,TypeError,KeyError):ok=False
        check('full public contract and actual typed prompt',ok,sids=sids,category='instruction_clarity')
        check('PENDING status empty review IDs structured RSM record',r['track_id']=='RSM-001' and r['target_format']=='structured' and r['review_status']=='PENDING' and r['review_receipt_ids']==[],sids=sids,category='custody')
        prov=r.get('source_provenance',{});check('sealed evaluation explicitly false',type(prov.get('sealed_evaluation_used')) is bool and prov['sealed_evaluation_used'] is False,sids=sids,category='provenance')
        check('origin/license and actual generation source metadata typed',prov.get('origin')=='original_synthetic' and type(prov.get('license')) is str and bool(prov['license']) and type(r['generator_version']) is str and type(r['oracle_version']) is str and type(r['generation_seed']) is int,sids=sids,category='provenance')
        ix=r.get('source_family_index');ctx=r.get('controlled_context');style=r.get('renderer');fam=r.get('source_family_key')
        check('typed public family/context/style coordinates',type(ix) is int and type(fam) is str and bool(fam) and ctx in CONTEXTS and style in ('forward','reverse'),sids=sids,category='metadata')
        expected_stratum=next((k for k,v in spec['source_family_indices_by_stratum'].items() if ix in v),None)
        check('actual configured stratum index and family split',expected_stratum is not None and prov.get('source_stratum')==expected_stratum and r['split']==('dev_challenge' if ix%10>=8 else 'train'),sids=sids,category='lineage')
        key=(ctx,style);check('unique actual family/context/renderer member',key not in families[fam],sids=sids,category='lineage');families[fam][key]=r
        p=parentmap.get(r['semantic_parent_id']);check('exact semantic parent task/family/source coordinates',p is not None and solver.typed_equal(p['semantic_task'],t) and all(p.get(k)==r.get(k) for k in ('semantic_family_id','source_family_key','source_family_index','controlled_context','split')),sids=sids,category='lineage')
        lev=r.get('difficulty');struct=r.get('structural_difficulty');ex=prov.get('curriculum_difficulty_explanation')
        check('strict difficulty2..4 equal plus explanation present',type(lev) is int and 2<=lev<=4 and type(struct) is int and struct==lev and type(ex) is str and bool(ex.strip()),sids=sids,category='difficulty_quality')
        check('string skill descriptors',type(r['skill_tags']) is list and all(type(x) is str and x for x in r['skill_tags']),sids=sids,category='competency')
        check('superseded family_claims alias absent','family_claims' not in r,sids=sids,category='metadata')
    entries=manifest.get('control_pairs',[]);resolver=collections.defaultdict(list);entry_evidence={}
    check('300 supplied named control entries',type(entries) is list and len(entries)==300)
    for en in entries:
        try:name,key=identifier(en);resolver[key].append(en)
        except (ValueError,KeyError,TypeError) as exc:check('valid named entry resolver',False,{'error':str(exc)});continue
        ids=en['pair_sample_ids'];members=[by_id.get(x) for x in ids]
        h=all(m is not None for m in members)
        if not h:check('actual control members exist',False,ids);continue
        left,right=members;coords=[m['controlled_context'] for m in members];styles=[m['renderer'] for m in members]
        h=en.get('scope')=='corpus_variant' and en.get('candidate_sha256')==candidate_sha and type(en.get('source_family_index')) is int and en['source_family_index']==left['source_family_index']==right['source_family_index'] and en.get('source_family_key')==left['source_family_key']==right['source_family_key'] and en.get('controlled_contexts')==coords and en.get('renderers')==styles and left['split']==right['split']
        if name=='object_field_order_invariance':h=h and coords[0]==coords[1] and styles==['forward','reverse']
        else:h=h and coords==list(PAIRS[name]) and styles[0]==styles[1]
        if 'pair_record_hashes' in en:h=h and solver.typed_equal(en['pair_record_hashes'],[m['record_hash'] for m in members])
        check('truthful named entry exact coordinates and member hashes',h,{'key':key},ids)
        try:proof=relation(solver,name,left,right,pred,facts,cal)
        except (KeyError,ValueError,TypeError,StopIteration,IndexError) as exc:proof={'name':name,'sample_ids':ids,'holds':False,'evidence':{'error':str(exc)}}
        evidence.append(proof);entry_evidence[key]=proof;check('actual semantic relation '+name,proof['holds'],proof['evidence'],ids)
    check('unique full named resolver keys',all(len(es)==1 for es in resolver.values()))
    owner_count=0;nonmember=0
    required_claim={'scope','evidence_manifest','control_id','pair_sample_ids'}
    allowed_claim=required_claim|{'name','property','source_family_key','source_family_index','controlled_context','renderer','diagnostics'}
    for r in rows:
        sid=r['sample_id'];claims=r.get('family_control_claims',[]);names=[]
        check('exactly nine actual typed owner claims',type(claims) is list and len(claims)==9,sids=[sid])
        for c in claims:
            owner_count+=1
            try:
                name,key=identifier(c);names.append(name);members=[by_id[x] for x in c['pair_sample_ids']]
                h=required_claim<=set(c)<=allowed_claim and c['scope']=='corpus_variant' and 'pair_record_hashes' not in c and str(safe_relative(c['evidence_manifest']).relative_to(ROOT))==cfg['manifest'] and len(resolver[key])==1
                if name=='object_field_order_invariance':h=h and sid in c['pair_sample_ids'] and all(m['controlled_context']==r['controlled_context'] for m in members)
                else:h=h and all(m['renderer']==r['renderer'] for m in members)
                h=h and all(m['source_family_key']==r['source_family_key'] and m['source_family_index']==r['source_family_index'] and m['split']==r['split'] for m in members)
                for k in ('source_family_key','source_family_index','controlled_context','renderer'):
                    if k in c:h=h and type(c[k]) is type(r[k]) and c[k]==r[k]
                if 'diagnostics' in c:h=h and type(c['diagnostics']) is dict
                h=h and entry_evidence[key]['holds'];nonmember+=sid not in c['pair_sample_ids']
                detail={'name':name,'resolver_key':key,'owner_is_pair_member':sid in c['pair_sample_ids']}
            except (KeyError,ValueError,TypeError,IndexError) as exc:h=False;detail={'error':str(exc),'claim':c}
            check('owner-scoped unique actual claim resolution',h,detail,[sid])
        check('one claim each required public name',len(names)==9 and set(names)==NAMES,sids=[sid])
    check('actual cohort counts 12families1944claims1344nonmemberowners',len(families)==12 and owner_count==1944 and nonmember==1344)
    signatures=[];legacy_signatures={v:collections.defaultdict(list) for v in ('full','coarse_all','coarse_reachable')}
    for l in legacy:
        for v in legacy_signatures:legacy_signatures[v][norms.signature(l['semantic_task'],v)['sha256']].append(l['sample_id'])
    for fam,members in families.items():
        sids=[r['sample_id'] for r in members.values()]
        check('complete nine contexts both field styles',set(members)==set(itertools.product(CONTEXTS,('forward','reverse'))),sids=sids)
        F,R,T=[members[(x,'forward')] for x in ('F','R','T')];ft=F['semantic_task'];rt=R['semantic_task'];f=facts[F['sample_id']];rf=facts[R['sample_id']];b=f['cohort_bounds_observations'];mt=f['metrics']
        check('all public finite-cohort bounds',2<=b['options']<=4 and all(1<=x<=3 for x in b['parameter_counts']) and b['max_option_depth']<=3 and mt['syntactic_if_depth']<=2 and mt['declared_instruction_count']<=24 and b['original_full_events']<=48 and b['initial_abs_max']<=6 and all(facts[r['sample_id']]['cohort_bounds_observations']['run_abs_max']<=4096 for r in members.values()) and b['original_full_effects']>=3,sids=sids)
        active=rf['actual_start_cut_facts'];pending_callers=[]
        frames=rt['checkpoint']['frames']
        for i,frame in enumerate(frames[:-1]):
            if frame['frame_kind']!='option':continue
            later=[x for x in frames[i+1:] if x['frame_kind']=='option']
            block=solver.body_at(rt,frame)
            if later and 0<frame['next_index']<len(block):pending_callers.append({'option':frame['option'],'next_index':frame['next_index'],'pending_instruction':block[frame['next_index']]})
        check('actual R entered nonroot-initiation and selected-guard reversal plus advanced pending caller',active['option_depth']>=2 and active['branch_depth']>=1 and any(x['instance_id']!=0 for x in active['entered_initiation_now_false']) and bool(active['selected_guard_reversals']) and bool(pending_callers),{'active':active,'pending_callers':pending_callers},sids)
        for v in legacy_signatures:
            sg=norms.signature(ft,v);sg.update(source_family_key=fam,sample_id=F['sample_id'],legacy_collisions=legacy_signatures[v].get(sg['sha256'],[]));signatures.append(sg)
            check('no actual legacy graph collision '+v,not sg['legacy_collisions'],sg,sids)
        metrics={'maximum_option_depth':b['max_option_depth'],'maximum_selected_branch_depth':b['max_branch_depth'],'declared_instructions':mt['declared_instruction_count'],'total_F_events':b['original_full_events'],'total_F_effects':b['original_full_effects'],'history_effects_at_R':sum(e['event']=='effect' for e in rt['history']),'caller_frames_at_R':active['option_depth']-1,'selected_branch_frames_at_R':active['branch_depth']}
        for r in members.values():
            sid=r['sample_id'];w=r.get('property_witnesses');check('optional property witnesses diagnostic object',type(w) is dict,sids=[sid])
            if type(w) is not dict:continue
            if 'dependency_metrics' in w:
                dm=w['dependency_metrics'];h=type(dm) is dict and set(dm)<=set(metrics) and all(type(x) is int and x>=0 and x==metrics[k] for k,x in dm.items());check('supplied dependency metric facts',h,{'actual':metrics,'supplied':dm},[sid])
            for tag in ('saved_binding_locator','checkpoint_tamper_locator'):
                if tag not in w:continue
                try:
                    loc=w[tag];h=type(loc) is list and bool(loc) and all(type(x) in (str,int) for x in loc)
                    proof=future_dependency(solver,rt,loc);h=h and proof['holds']
                    if tag=='saved_binding_locator':h=h and len(loc)==4 and loc[0]=='frames' and loc[2]=='bindings' and saved_origin(solver,rt,rf['original_full_run'],loc)
                    else:h=h and solver.typed_equal(loc,[x for x in diffs(solver,rt,members[('X',r['renderer'])]['semantic_task'])[0]['path'][1:]])
                except (ValueError,TypeError,IndexError,KeyError) as exc:h=False;proof={'error':str(exc)}
                check('actual active future-use locator '+tag,h,proof,[sid])
            if 'replay_locator' in w:
                loc=w['replay_locator'];h=type(loc) is dict and set(loc)=={'option','body_path','index'} and type(loc.get('option')) is str and type(loc.get('body_path')) is list and type(loc.get('index')) is int
                proof=replay_witnesses(solver,rt,rf['original_full_run'],pred[T['sample_id']]['expected_output']['new_events'],loc) if h else []
                check('supplied locator actual nonidempotent replay',h and bool(proof),proof,[sid])
            check('private witness diagnostics typed if supplied','diagnostics' not in w or type(w['diagnostics']) is dict,sids=[sid])
    for v in legacy_signatures:
        groups=collections.defaultdict(list)
        for s in signatures:
            if s['view']==v:groups[s['sha256']].append(s['source_family_key'])
        for sh,fs in groups.items():
            involved=[r['sample_id'] for fam in fs for r in families[fam].values()]
            check('actual mutual graph novelty '+v,len(fs)==1,{'signature':sh,'families':fs},involved)
    # No verdict is inferred from a positive check ledger. Actual semantic and
    # shortcut/difficulty assessment must supply a separately frozen quality map.
    comparison_rows=[]
    for r in rows:
        sid=r['sample_id'];checks=rowchecks[sid]
        comparison_rows.append({'sample_id':sid,'record_hash':r['record_hash'],'actual_author_execution_id':ACTOR,
          'frozen_first_prediction_sha256':pred[sid]['typed_task_sha256'],'target_equal_to_immutable_first':solver.typed_equal(pred[sid]['expected_output'],r['verified_target']),
          'checks':checks,'failures':[x for x in checks if not x['holds']],'quality_decision':'UNASSESSED_PENDING_ACTUAL_AGENT_INSPECTION'})
    outputs={'new_calibration':toy_output,'per_record':write_jsonl(HERE/'canonical-comparison.first.jsonl',comparison_rows),
      'controls':write_json('actual-control-evidence.json',evidence),'novelty':write_json('own-novelty-evidence.json',{'new_signatures':signatures,'legacy_unique_signatures':{k:len(v) for k,v in legacy_signatures.items()},'legacy_rows':len(legacy),'all_three_views':'full/coarse_all/coarse_reachable; no producer normalizer read'}),
      'checks':write_json('canonical-check-ledger.json',{'checks':batch_checks,'passed':sum(x['holds'] for x in batch_checks),'failed':sum(not x['holds'] for x in batch_checks)})}
    assert all(sha(Path(item['absolute_path']).read_bytes())==item['sha256'] for item in freeze['bindings'])
    execution={'actual_author_execution_id':ACTOR,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-timer,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'argv':sys.argv,'exit_code':0,'all_bound_bytes_unchanged':True,'source_version':VERSION,'source_freeze_sha256':sha((HERE/'SOURCE-PREPARATION-FREEZE.json').read_bytes()),'frozen_first_predictions_unchanged':True,'outputs':outputs,'dataset_approval':False}
    ex=write_json('COMPARISON-EXECUTION.json',execution);print(canonical({'execution':ex,'outputs':outputs,'checks_failed':sum(not x['holds'] for x in batch_checks),'verdicts':'pending actual agent quality inspection'}))

def emit_receipts(quality_path):
    """Later authorized phase: source-frozen actual-agent quality map, no fake
    automatic acceptance. This phase does not recompute/edit gold/predictions.
    """
    quality_path=Path(quality_path);q=json.loads(quality_path.read_text());auth=json.loads((HERE/'QUALITY-MAP-FREEZE.json').read_text())
    assert q['actual_author_execution_id']==ACTOR and auth['quality_map_sha256']==sha(quality_path.read_bytes())
    compared=[json.loads(x) for x in (HERE/'canonical-comparison.first.jsonl').read_text().splitlines()]
    by_id={x['sample_id']:x for x in compared};qualities=q['per_sample'];assert set(qualities)==set(by_id)
    rs=[]
    for sid,item in by_id.items():
        quality=qualities[sid];decision=quality['decision'];assert decision in ('ACCEPT','REVISE','REJECT','QUARANTINE')
        assert set(quality['scores'])==set(SCORES) and all(type(x) is int and 1<=x<=5 for x in quality['scores'].values())
        assert type(quality['confidence']) in (int,float) and 0<=quality['confidence']<=1
        assert type(quality['issues']) is list and type(quality['critical_defects']) is list
        failed=item['failures'];assert decision!='ACCEPT' or (not failed and not quality['critical_defects'])
        if not item['target_equal_to_immutable_first']:assert quality['scores']['target_correctness']<=2 and decision!='ACCEPT'
        issues=list(quality['issues'])+[{'category':x['category'],'severity':x['severity'],'evidence':x['evidence'],'fix':'Correct a fresh version while preserving original candidate and immutable first reviewer evidence.','check':x['name']} for x in failed]
        rs.append({'receipt_id':'expert-RSM-option-resumption-pilot-v1-canonical:'+sid,'sample_id':sid,'record_hash':item['record_hash'],'reviewer_role':'Expert Reviewer','reviewer_execution_id':ACTOR,'review_status':'COMPLETED','scores':quality['scores'],'critical_defects':quality['critical_defects'],'issues':issues,'decision':decision,'confidence':quality['confidence'],'evidence':{'comparison_first_path':str((HERE/'canonical-comparison.first.jsonl').relative_to(ROOT)),'quality_map_sha256':sha(quality_path.read_bytes()),'prediction_before_gold_preserved':True,'target_equal_to_immutable_first':item['target_equal_to_immutable_first'],'source_version':VERSION,'disclosures':q['disclosures']}})
    first=write_jsonl(HERE/'canonical-judgments.first.jsonl',rs)
    output=ROOT/'reviews/expert/option-resumption-pilot-v1.jsonl';final=write_jsonl(output,rs)
    print(canonical({'first':first,'final':final,'decisions':dict(collections.Counter(r['decision'] for r in rs)),'publication_approval':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['compare','receipts']);parser.add_argument('--quality-map');args=parser.parse_args()
    try:
        if args.phase=='compare':execute_compare()
        elif args.quality_map:emit_receipts(args.quality_map)
        else:raise ValueError('actual agent authored/frozen quality map required')
    except BaseException:
        p=HERE/('FAILURE-'+args.phase+'.txt')
        with p.open('x') as f:f.write(traceback.format_exc())
        raise
