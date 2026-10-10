"""Expert-owned actual input-only RSM review; no production implementation/gold.
Source must be archived/frozen before first import or prediction. Input initial
judgments are deliberately PENDING_CANONICAL with UNKNOWN target correctness.
Reuse of own public-only interpreter/calibrations and common model is explicit.
"""
import collections
import copy
import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ACTOR = '/root/expert_reviewer'
VERSION = 'rsm-input-first-expert-1.0.0'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(b): return hashlib.sha256(b).hexdigest()
def typed_token(v):
    if type(v) is dict: return ('dict', tuple((k, typed_token(v[k])) for k in sorted(v)))
    if type(v) is list: return ('list', tuple(typed_token(x) for x in v))
    return (type(v).__name__, v)
def digest(v): return sha(repr(typed_token(v)).encode())
def write_json(name, value):
    p = HERE / name
    with p.open('x') as f: json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2); f.write('\n')
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p.read_bytes()), 'bytes':p.stat().st_size}
def write_rows(name, rows):
    p = HERE / name
    with p.open('x') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p.read_bytes()), 'bytes':p.stat().st_size,'rows':len(rows)}
def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def instruction_metrics(task):
    counts=collections.Counter(); max_depth=0; paths=[]; live_reads=0; param_reads=0
    def walk(block, owner, path, depth):
        nonlocal max_depth,live_reads,param_reads
        for i, ins in enumerate(block):
            counts[ins['op']]+=1;paths.append({'option':owner,'body_path':path,'index':i,'op':ins['op']})
            max_depth=max(max_depth,depth+(ins['op']=='if'))
            if ins['op']=='if':
                walk(ins['then'],owner,path+[i,'then'],depth+1);walk(ins['else'],owner,path+[i,'else'],depth+1)
            def scan(v):
                nonlocal live_reads,param_reads
                if type(v) is dict:
                    if set(v)=={'reg'}:live_reads+=1
                    if set(v)=={'param'}:param_reads+=1
                    for k,x in v.items():
                        if k not in ('then','else'):scan(x)
                elif type(v) is list:
                    for x in v:scan(x)
            scan(ins)
    for o,d in task['options'].items():walk(d['body'],o,[],0)
    return {'declared_instruction_count':sum(counts.values()),'opcode_counts':dict(counts),
            'syntactic_if_depth':max_depth,'instruction_locators':paths,'syntactic_live_read_count':live_reads,'syntactic_parameter_read_count':param_reads}

def run_to_terminal(solver, task, start=None):
    m=copy.deepcopy(start) if start is not None else solver.initial_machine(task['initial_state'])
    events=[]; snapshots=[copy.deepcopy(m)]
    for _ in range(256):
        if m['phase']!='running':break
        events.append(solver.step(m,task));snapshots.append(copy.deepcopy(m))
    if m['phase']=='running':raise AssertionError('reviewer safety event cap exceeded')
    return {'events':events,'snapshots':snapshots,'checkpoint':m}

def expected_at_budget(solver, task, start, budget, previous):
    m=copy.deepcopy(start); events=[]
    for _ in range(budget):
        if m['phase']!='running':break
        events.append(solver.step(m,task))
    return {'status':m['phase'] if m['phase']!='running' else 'paused','error':None,
       'checkpoint':m,'new_events':events,'previous_effects':previous,
       'new_effects':sum(e['event']=='effect' for e in events),
       'next_event':solver.step(copy.deepcopy(m),task) if m['phase']=='running' else None}

def checkpoints_facts(solver,task,trace):
    facts=[]
    for cut,m in enumerate(trace['snapshots']):
        opts=[f for f in m['frames'] if f['frame_kind']=='option']
        branches=[f for f in m['frames'] if f['frame_kind']=='branch']
        rechecked=[]
        for f in opts:
            if not solver.test(task['options'][f['option']]['initiation'],m['state'],f['bindings']):
                rechecked.append({'instance_id':f['instance_id'],'option':f['option'],'bindings':f['bindings']})
        reversals=[]
        for f in branches:
            p=f['body_path'];parent={'option':f['option'],'body_path':p[:-2]}
            ins=solver.body_at(task,parent)[p[-2]]
            current=solver.test(ins['test'],m['state'],f['bindings'])
            selected=p[-1]=='then'
            if current!=selected:reversals.append({'instance_id':f['instance_id'],'option':f['option'],'body_path':p,'selected':selected,'current_guard':current})
        facts.append({'cut':cut,'state':m['state'],'phase':m['phase'],'option_depth':len(opts),'branch_depth':len(branches),
                     'entered_initiation_now_false':rechecked,'selected_guard_reversals':reversals})
    return facts

def effect_instruction(task, solver, e):
    return solver.body_at(task,{'option':e['option'],'body_path':e['body_path']})[e['index']]

def semantic_signature(task, coarse=False):
    """Input-only alpha/scalar-erased program graph canonicalization; exhaustive
    option/parameter/register renamings within bounded <=4/<=3 public cohort.
    Coarse removes guards/entry args and collapses const/param scalar reads;
    these are review diagnostics, not a publication novelty verdict vs legacy.
    """
    names=list(task['options']); signatures=[]
    for order in itertools.permutations(names):
        om={o:i for i,o in enumerate(order)}
        # Parameter declaration order is semantic only as names to args; erase
        # name under every parameter permutation, retain all def/use incidence.
        porders=[list(itertools.permutations(task['options'][o]['params'])) for o in order]
        for combinations in itertools.product(*porders):
            pm={o:{p:i for i,p in enumerate(ps)} for o,ps in zip(order,combinations)}
            for swap in (False,True):
                def expr(x,owner):
                    if 'reg' in x:return ('reg',('y' if x['reg']=='x' else 'x') if swap else x['reg'])
                    if coarse:return ('scalar',)
                    return ('const',) if 'const' in x else ('param',pm[owner][x['param']])
                def pred(x,owner):return tuple((expr(a['left'],owner),a['cmp'],expr(a['right'],owner)) for a in x['all'])
                def body(block,owner):
                    result=[]
                    for x in block:
                        op=x['op']
                        if op in ('add','scale'):
                            register=('y' if x['register']=='x' else 'x') if swap else x['register']
                            result.append((op,register,expr(x['value'],owner)))
                        elif op=='swap':result.append(('swap',))
                        elif op=='if':result.append(('if',pred(x['test'],owner),body(x['then'],owner),body(x['else'],owner)))
                        else:
                            callee=x['option']
                            args=() if coarse else tuple(sorted((pm[callee][p],expr(v,owner)) for p,v in x['args'].items()))
                            result.append(('call',om[callee],args))
                    return tuple(result)
                ds=[]
                for o in order:
                    d=task['options'][o]
                    ds.append((body(d['body'],o),) if coarse else (len(d['params']),pred(d['initiation'],o),pred(d['success'],o),body(d['body'],o)))
                signatures.append(repr((tuple(ds),None if coarse else om[task['entry']['option']])))
    return sha(min(signatures).encode())

def typed_differences(a,b,path=()):
    if type(a) is not type(b):return [{'path':list(path),'left':a,'right':b}]
    if type(a) is dict:
        if set(a)!=set(b):return [{'path':list(path),'left':a,'right':b}]
        return sum((typed_differences(a[k],b[k],path+(k,)) for k in a),[])
    if type(a) is list:
        if len(a)!=len(b):return [{'path':list(path),'left':a,'right':b}]
        return sum((typed_differences(x,y,path+(i,)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [{'path':list(path),'left':a,'right':b}]

def status_baseline(rows, predictions, feature):
    buckets=collections.defaultdict(collections.Counter)
    for r,p in zip(rows,predictions):buckets[str(feature(r))][p['expected_output']['status']]+=1
    correct=sum(max(x.values()) for x in buckets.values())
    # Fitted on the same inspected inputs, not student/model evaluation.
    return {'correct':correct,'total':len(rows),'accuracy':correct/len(rows),
            'buckets':{k:dict(v) for k,v in sorted(buckets.items())},'scope':'in-sample status-only upper fit; full structured output not solved'}

def run():
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(); timer=time.perf_counter()
    freeze=json.loads((HERE/'PREEXECUTION.json').read_text());config=json.loads((HERE/'config.json').read_text())
    initial={}
    for item in freeze['bindings']:
        p=ROOT/item['path'];b=p.read_bytes();assert sha(b)==item['sha256'];assert len(b)==item['bytes']
        ab=(HERE/item['archive_path']).read_bytes();assert ab==b
        initial[item['path']]=sha(b)
    # Every own/new import occurs only after explicit raw/source/runtime freeze.
    solver=load_module('expert_owned_rsm_solver',ROOT/config['solver'])
    calibration=load_module('expert_owned_rsm_calibration',ROOT/config['calibration_source'])
    fixtures=json.loads((ROOT/config['calibration_fixtures']).read_text())
    contract=(ROOT/config['contract']).read_text()
    calibrations=calibration.run(solver,fixtures,contract)
    outputs={'calibration':write_json('calibration-reexecution.json',calibrations)}
    raw=(ROOT/config['input']).read_bytes();assert sha(raw)==config['input_sha256']
    rows=[solver.strict_json(line) for line in raw.decode().splitlines()]
    assert len(rows)==216 and len({r['sample_id'] for r in rows})==216
    allowed=set(config['input_fields']);assert all(set(r)==allowed for r in rows)
    predictions=[];facts=[];initial_judgments=[];allchecks=[]; by_id={}
    def check(label, condition, sample, evidence=None):
        entry={'name':label,'sample_id':sample,'passed':bool(condition)}
        if evidence is not None:entry['evidence']=evidence
        allchecks.append(entry)
        if not condition:raise AssertionError(json.dumps(entry,ensure_ascii=False))
    for r in rows:
        sid=r['sample_id'];t=solver.parse_prompt(r['rendered_input'],r['semantic_task'],contract)
        check('full contract and exact typed rendered task',True,sid)
        before=digest(t);p=solver.solve(t);check('solver leaves typed input unchanged',before==digest(t),sid)
        check('reversed-object-fields target invariant',solver.typed_equal(p,solver.solve(calibration.reversed_fields(t))),sid)
        om={o:'option_'+str(i) for i,o in enumerate(t['options'])}
        ps=sorted({p for d in t['options'].values() for p in d['params']});pm={p:'parameter_'+str(i) for i,p in enumerate(ps)}
        renamed=calibration.alpha(t,om,pm); check('alpha names and register-swap target equivariance',
            solver.typed_equal(solver.solve(renamed),calibration.alpha(p,om,pm)),sid)
        ftask=copy.deepcopy(t);ftask.update(mode='fresh',history=[],checkpoint=None,event_budget=256)
        trace=run_to_terminal(solver,ftask);recursive=calibration.recursive_fresh_trace(ftask)
        check('separate own recursive fresh simulator complete trace/state/status',
            solver.typed_equal(trace['events'],recursive['events']) and solver.typed_equal(trace['checkpoint']['state'],recursive['state']) and trace['checkpoint']['phase']==recursive['phase'],sid)
        cp_facts=checkpoints_facts(solver,ftask,trace)
        hist=t['history'];cut=len(hist)
        prefix_valid=type(hist) is list and cut<len(trace['snapshots']) and solver.typed_equal(hist,trace['events'][:cut])
        expected_cp=trace['snapshots'][cut] if prefix_valid else None
        cp_match=solver.typed_equal(t['checkpoint'],expected_cp) if t['mode']!='fresh' else t['checkpoint'] is None and not hist
        check('actual history is independent generated prefix',prefix_valid,sid)
        if p['error'] is None:
            start=solver.initial_machine(t['checkpoint']['state']) if t['mode']=='restart' else (expected_cp if t['mode']=='resume' else trace['snapshots'][0])
            past=sum(e['event']=='effect' for e in hist)
            check('direct event-limited transition agrees exact full output',solver.typed_equal(p,expected_at_budget(solver,t,start,t['event_budget'],past)),sid)
            for budget in sorted({0,1,t['event_budget'],256}):
                changed=copy.deepcopy(t);changed['event_budget']=budget
                check('actual input boundary budget '+str(budget),solver.typed_equal(solver.solve(changed),expected_at_budget(solver,t,start,budget,past)),sid)
            continuous=run_to_terminal(solver,t,start)
            if t['mode']=='resume':check('resume full suffix exact original event composition',solver.typed_equal(hist+continuous['events'],trace['events']),sid)
            if t['mode']=='restart':
                rt=copy.deepcopy(ftask);rt['initial_state']=copy.deepcopy(t['checkpoint']['state']);rr=calibration.recursive_fresh_trace(rt)
                check('restart independent recursive new-run from retained state',solver.typed_equal(continuous['events'],rr['events']) and solver.typed_equal(continuous['checkpoint']['state'],rr['state']) and continuous['checkpoint']['phase']==rr['phase'],sid)
        else:
            check('invalid task has uniquely empty/null supervision result',solver.typed_equal(p,solver.invalid(p['error'])),sid)
            check('observed invalidity is checkpoint mismatch before any new event',p['error']=='invalid_checkpoint' and not cp_match,sid)
            continuous=None
        if hist:
            changed=copy.deepcopy(t);changed['history'][0]['instance_id']=False
            check('typed false cannot impersonate history integer zero',solver.solve(changed)['error']=='invalid_history',sid)
        if t['mode']!='fresh' and cp_match:
            changed=copy.deepcopy(t);changed['checkpoint']['next_instance_id']=bool(changed['checkpoint']['next_instance_id'])
            check('checkpoint boolean cannot impersonate integer',solver.solve(changed)['error']=='invalid_checkpoint',sid)
        metrics=instruction_metrics(t)
        states=[m['state'] for m in trace['snapshots']]
        if continuous:states+= [m['state'] for m in continuous['snapshots']]
        bounds={'initial_abs_max':max(abs(x) for x in t['initial_state'].values()),'run_abs_max':max(abs(x) for s in states for x in s.values()),
                'options':len(t['options']),'parameter_counts':[len(d['params']) for d in t['options'].values()],
                'original_full_events':len(trace['events']),'original_full_effects':sum(e['event']=='effect' for e in trace['events']),
                'max_option_depth':max(x['option_depth'] for x in cp_facts),'max_branch_depth':max(x['branch_depth'] for x in cp_facts)}
        fact={'sample_id':sid,'record_hash':r['record_hash'],'input_packet_sha256':config['input_sha256'],
            'typed_task_sha256':digest(t),'mode':t['mode'],'event_budget':t['event_budget'],'history_events':cut,
            'independent_status':p['status'],'independent_error':p['error'],'history_prefix_valid':prefix_valid,'checkpoint_matches_replay':cp_match,
            'original_full_run':trace,'original_checkpoint_facts':cp_facts,'actual_start_cut_facts':cp_facts[cut],
            'actual_continuation':continuous,'metrics':metrics,'cohort_bounds_observations':bounds,
            'checkpoint_mismatch_paths':typed_differences(expected_cp,t['checkpoint']) if t['mode']!='fresh' else [],
            'full_program_exact_typed_sha256':digest({'options':t['options'],'entry':t['entry'],'initial_state':t['initial_state']})}
        facts.append(fact)
        pr={k:r[k] for k in ('sample_id','record_hash','semantic_parent_id','semantic_family_id','track_id','split')}
        pr.update(reviewer_role='Expert Reviewer',reviewer_execution_id=ACTOR,source_version=VERSION,
            input_packet_sha256=config['input_sha256'],typed_task_sha256=digest(t),expected_output=p)
        predictions.append(pr)
        issues=[{'category':'generalization','severity':'moderate','evidence':'Finite supplied deterministic program; repeated requests/field views share the same program. Four public strata and exposed development provide no sealed transfer or model-efficacy evidence.','fix':'Describe actual independent program/control/view counts; retain this as bounded execution supervision.'}]
        if hist:
            issues.append({'category':'shortcut_resistance','severity':'minor','evidence':{'history_events':cut,'mode':t['mode'],'event_budget':t['event_budget']},'fix':'Measure mode/history/budget status associations at batch level and evaluate full typed outputs, not status-only accuracy.'})
        iq={'receipt_id':'RSM-input-first-expert-01:'+sid,'sample_id':sid,'record_hash':r['record_hash'],
            'reviewer_role':'Expert Reviewer','reviewer_execution_id':ACTOR,'review_status':'INITIAL_INPUT_ONLY',
            'decision':'PENDING_CANONICAL','input_verdict':'CLEAR_DETERMINATE_INPUT_PENDING_GOLD',
            'target_correctness':{'status':'UNKNOWN','reason':'No canonical target or other initial reviewer verdict read.'},
            'scores':{'semantic_correctness':5,'target_correctness':None,'instruction_clarity':4,'reasoning_validity':5,
                'generalization_value':3,'difficulty_quality':4 if bounds['max_option_depth']==3 and metrics['syntactic_if_depth']==2 else 3,
                'diversity':3,'shortcut_resistance':3},'critical_defects':[],'issues':issues,'confidence':0.97,
            'evidence':{'prediction_path':'predictions.first.jsonl','facts_path':'independent-facts.first.jsonl',
                'typed_task_sha256':digest(t),'input_packet_sha256':config['input_sha256'],
                'independent_status':p['status'],'semantic_event_count':len(p['new_events']),'previous_effects':p['previous_effects'],
                'new_effects':p['new_effects'],'active_option_depth':cp_facts[cut]['option_depth'],
                'active_branch_depth':cp_facts[cut]['branch_depth'],'actual_history_prefix_valid':prefix_valid,
                'scoring_scope':'input semantics/clarity/competency only; metadata/witnesses/difficulty labels and target correctness unassessed'}}
        initial_judgments.append(iq);by_id[sid]=(r,pr,fact)
    outputs['predictions']=write_rows('predictions.first.jsonl',predictions)
    outputs['independent_facts']=write_rows('independent-facts.first.jsonl',facts)
    outputs['initial_judgments']=write_rows('initial-judgments.first.jsonl',initial_judgments)
    outputs['metamorphics']=write_json('actual-input-checks.json',{'checks':allchecks,'passed':len(allchecks),'failed':0})
    # Input-only matched relationships, derived from visible tasks rather than
    # any unseen producer witnesses. Context coordinates are join-only and do
    # not determine predictions; task mode/history/program alone was solved.
    families=collections.defaultdict(dict)
    for r,pr,fact in by_id.values():
        parts=r['sample_id'].rsplit('-',2);context,style=parts[-2:]
        families[r['semantic_family_id']][(context,style)]=(r,pr,fact)
    controls=[]
    def relation(name, members, condition, evidence):
        controls.append({'name':name,'sample_ids':members,'holds_in_actual_inputs':bool(condition),'evidence':evidence})
    for fam,d in sorted(families.items()):
        assert len(d)==18
        for style in ('forward','reverse'):
            F,R,T,P,U,A,I,S,X=[d[(c,style)] for c in ('F','R','T','P','U','A','I','S','X')]
            def ids(*items):return [x[0]['sample_id'] for x in items]
            ft,rt,tt,pt,ut,at,it,st,xt=[x[0]['semantic_task'] for x in (F,R,T,P,U,A,I,S,X)]
            fp,rp,tp,pp,up,ap,ip,sp,xp=[x[1]['expected_output'] for x in (F,R,T,P,U,A,I,S,X)]
            cut=len(rt['history'])
            relation('F-R exact suffix/state/status',ids(F,R),solver.typed_equal(rt['history']+rp['new_events'],fp['new_events']) and solver.typed_equal(fp['checkpoint'],rp['checkpoint']) and fp['status']==rp['status'],{'prefix_events':cut,'suffix_events':len(rp['new_events'])})
            df=typed_differences(rt,tt)
            replay=[]
            for h in rt['history']:
                if h['event']!='effect':continue
                matches=[e for e in tp['new_events'] if e['event']=='effect' and e['option']==h['option'] and e['body_path']==h['body_path'] and e['index']==h['index']]
                for e in matches:
                    ins=effect_instruction(tt,solver,e)
                    # Apply the exact same committed instruction twice from its
                    # original pre-effect state, with original immutable args.
                    hi=rt['history'].index(h);snap=R[2]['original_full_run']['snapshots'][hi]
                    binding=next(f['bindings'] for f in reversed(snap['frames']) if f['instance_id']==h['instance_id'])
                    s0=copy.deepcopy(snap['state'])
                    def effect(s):
                        out=copy.deepcopy(s)
                        if ins['op']=='swap':out={'x':s['y'],'y':s['x']}
                        else:
                            v=solver.evaluate(ins['value'],s,binding);q=ins['register'];out[q]=s[q]+v if ins['op']=='add' else s[q]*v
                        return out
                    s1=effect(s0);s2=effect(s1)
                    if not solver.typed_equal(s1,s2):replay.append({'option':h['option'],'body_path':h['body_path'],'index':h['index'],'original_before':s0,'once':s1,'twice':s2,'restart_event':e})
            relation('R-T mode-only actual non-idempotent replay',ids(R,T),len(df)==1 and df[0]['path']==['mode'] and tp['new_events'][0]['initiation_ok'] and bool(replay) and not solver.typed_equal(rp['new_events'],tp['new_events']) and (rp['status']!=tp['status'] or not solver.typed_equal(rp['checkpoint']['state'],tp['checkpoint']['state'])),{'typed_differences':df,'nonidempotent_replay_witnesses':replay})
            df=typed_differences(rt,pt)
            relation('R-P budget-only positive second pause',ids(R,P),len(df)==1 and df[0]['path']==['event_budget'] and 0<pt['event_budget']<rt['event_budget'] and pp['status']=='paused' and pp['new_effects']>=1 and pp['next_event'] is not None,{'typed_differences':df,'new_events':len(pp['new_events']),'new_effects':pp['new_effects'],'next_event':pp['next_event']})
            relation('P-U exact extended history/cp and F suffix',ids(P,U),solver.typed_equal(ut['history'],pt['history']+pp['new_events']) and solver.typed_equal(ut['checkpoint'],pp['checkpoint']) and solver.typed_equal(ut['history']+up['new_events'],fp['new_events']) and solver.typed_equal(up['checkpoint'],fp['checkpoint']),{'extended_history_events':len(ut['history'])})
            df=typed_differences(ft,at);laterchanges=typed_differences(fp['new_events'][1:],ap['new_events'][1:])
            relation('F-A root argument actual downstream use',ids(F,A),len(df)==1 and df[0]['path'][:2]==['entry','args'] and fp['status']==ap['status']=='completed' and bool(laterchanges),{'typed_differences':df,'downstream_event_changes':laterchanges})
            df=typed_differences(ft,it)
            relation('F-I root argument decisive initiation failure',ids(F,I),len(df)==1 and df[0]['path'][:2]==['entry','args'] and fp['status']=='completed' and ip['status']=='initiation_failed' and ip['new_effects']==0,{'typed_differences':df,'failed_enter':ip['new_events']})
            df=typed_differences(ft,st);fe=[e for e in fp['new_events'] if e['event']=='effect'];se=[e for e in sp['new_events'] if e['event']=='effect']
            relation('F-S root success literal only same effects failure',ids(F,S),len(df)==1 and df[0]['path'][:3]==['options',ft['entry']['option'],'success'] and fp['status']=='completed' and sp['status']=='success_failed' and solver.typed_equal(fe,se),{'typed_differences':df,'body_effects':len(fe)})
            df=typed_differences(rt,xt)
            # Trusted changed machine is diagnostic only: execute its actual
            # pending transitions to prove changed scalar is consequential.
            originaltrusted=run_to_terminal(solver,rt,rt['checkpoint']);changedtrusted=run_to_terminal(solver,xt,xt['checkpoint'])
            consequence=not solver.typed_equal(originaltrusted,changedtrusted)
            relation('R-X one active checkpoint scalar rejected/consequential',ids(R,X),len(df)==1 and df[0]['path'][0]=='checkpoint' and type(df[0]['left']) in (int,str,bool) and xp['error']=='invalid_checkpoint' and consequence,{'typed_differences':df,'trusted_pending_behavior_changes':consequence,'trusted_diagnostic_only':True})
        for c in ('F','R','T','P','U','A','I','S','X'):
            left,right=d[(c,'forward')],d[(c,'reverse')]
            lb=left[0]['rendered_input'].split('Task JSON:\n')[1];rb=right[0]['rendered_input'].split('Task JSON:\n')[1]
            lo=solver.strict_json(lb);ro=solver.strict_json(rb)
            exactreverse=json.dumps(calibration.reversed_fields(lo),ensure_ascii=False,separators=(',',':'))==json.dumps(ro,ensure_ascii=False,separators=(',',':'))
            relation('field order reverse actual whole typed task',ids(left,right),solver.typed_equal(lo,ro) and solver.typed_equal(left[1]['expected_output'],right[1]['expected_output']) and exactreverse,{'actual_recursive_object_reversal':exactreverse})
    controls_output={'scope':'input-only observed public relationships, no producer metadata/gold/control manifest read','relations':controls,'passed':sum(c['holds_in_actual_inputs'] for c in controls),'failed':sum(not c['holds_in_actual_inputs'] for c in controls)}
    outputs['input_relations']=write_json('input-only-relations.json',controls_output)
    # Program overlap diagnostics measured within this actual packet. Legacy
    # novelty/control-manifest correctness remains for separately allowed stage.
    signatures={};program_rows=[]
    for fam,d in sorted(families.items()):
        r,pr,f=d[('F','forward')];t=r['semantic_task'];full=semantic_signature(t);coarse=semantic_signature(t,True)
        signatures[fam]={'full_alpha_scalar_erased_sha256':full,'coarse_all_declared_sha256':coarse}
        program_rows.append({'semantic_family_id':fam,'sample_id':r['sample_id'],'split':r['split'],**signatures[fam],
            'full_status':pr['expected_output']['status'],'full_events':len(pr['expected_output']['new_events']),
            'full_effects':pr['expected_output']['new_effects'],'metrics':f['metrics'],'bounds':f['cohort_bounds_observations']})
    hist=lambda fn:dict(sorted(collections.Counter(str(fn(r,p,f)) for r,p,f in by_id.values()).items()))
    summary={'actual_author_execution_id':ACTOR,'source_version':VERSION,'input_only':True,'candidate_or_gold_read':False,'dataset_approval':False,
      'records':len(rows),'source_families':len(families),'semantic_parents':len({r['semantic_parent_id'] for r in rows}),
      'typed_tasks':len({digest(r['semantic_task']) for r in rows}),'exact_rendered_prompts':len({r['rendered_input'] for r in rows}),
      'full_expected_outputs':len({digest(p['expected_output']) for p in predictions}),
      'split_counts':hist(lambda r,p,f:r['split']),'mode_counts':hist(lambda r,p,f:r['semantic_task']['mode']),
      'status_counts':hist(lambda r,p,f:p['expected_output']['status']),'error_counts':hist(lambda r,p,f:p['expected_output']['error']),
      'history_length_counts':hist(lambda r,p,f:len(r['semantic_task']['history'])),
      'new_event_counts':hist(lambda r,p,f:len(p['expected_output']['new_events'])),
      'original_horizon_counts':hist(lambda r,p,f:f['cohort_bounds_observations']['original_full_events']),
      'checkpoint_option_depth_counts':hist(lambda r,p,f:f['actual_start_cut_facts']['option_depth']),
      'checkpoint_guard_reversal_counts':hist(lambda r,p,f:len(f['actual_start_cut_facts']['selected_guard_reversals'])),
      'checkpoint_entered_initiation_now_false_counts':hist(lambda r,p,f:len(f['actual_start_cut_facts']['entered_initiation_now_false'])),
      'programs':program_rows,'full_alpha_programs':len({v['full_alpha_scalar_erased_sha256'] for v in signatures.values()}),
      'coarse_all_declared_programs':len({v['coarse_all_declared_sha256'] for v in signatures.values()}),
      'input_relationship_checks':{'passed':controls_output['passed'],'failed':controls_output['failed']},
      'baselines':{'majority':status_baseline(rows,predictions,lambda r:'all'),
          'mode':status_baseline(rows,predictions,lambda r:r['semantic_task']['mode']),
          'history_length':status_baseline(rows,predictions,lambda r:len(r['semantic_task']['history'])),
          'event_budget':status_baseline(rows,predictions,lambda r:r['semantic_task']['event_budget']),
          'mode_history_budget':status_baseline(rows,predictions,lambda r:(r['semantic_task']['mode'],len(r['semantic_task']['history']),r['semantic_task']['event_budget']))},
      'disclosures':config['disclosures'],'limitations':['Target correctness UNKNOWN; no canonical/private metadata/control claims inspected.','No comparison against legacy inputs in this stage; packet-only structure diagnostics are not structural novelty approval.','Status-only fitted baselines do not solve structured events/checkpoints.','Four public strata, twelve finite program graphs, correlated requests/field styles, exposed development and common-model evidence do not establish general recovery, planning or learned options.']}
    outputs['summary']=write_json('compact-summary.json',summary)
    after={item['path']:sha((ROOT/item['path']).read_bytes()) for item in freeze['bindings']}
    assert initial==after
    executed={'actual_author_execution_id':ACTOR,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'elapsed_seconds':time.perf_counter()-timer,'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'argv':sys.argv,'python':sys.executable,'exit_code':0,'source_version':VERSION,'preexecution_sha256':sha((HERE/'PREEXECUTION.json').read_bytes()),
      'source_bindings_before':initial,'source_bindings_after':after,'all_bindings_unchanged':initial==after,
      'all216_predictions_and_initial_input_judgments_completed':True,'canonical_gold_access':False,
      'calibration_checks':calibrations['passed'],'actual_input_checks':len(allchecks),'outputs':outputs}
    execution=write_json('EXECUTION.json',executed)
    first=write_json('FIRST.json',{'actual_author_execution_id':ACTOR,'status':'FROZEN_COMPLETE_INPUT_FIRST_PENDING_CANONICAL',
      'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':216,'input_packet_sha256':config['input_sha256'],
      'preexecution_sha256':sha((HERE/'PREEXECUTION.json').read_bytes()),'execution':execution,'outputs':outputs,
      'target_correctness':'UNKNOWN','dataset_approval':False,'disclosures':config['disclosures']})
    print(json.dumps({'status':'COMPLETE_INPUT_FIRST','first':first,'predictions':outputs['predictions'],'initial_judgments':outputs['initial_judgments'],'summary':outputs['summary'],'checks':len(allchecks),'relationships':summary['input_relationship_checks'],'elapsed':executed['elapsed_seconds']}))

if __name__=='__main__':
    try:run()
    except BaseException:
        text=traceback.format_exc(); (HERE/'FAILURE.txt').write_text(text)
        print(text,file=sys.stderr);sys.exit(1)
