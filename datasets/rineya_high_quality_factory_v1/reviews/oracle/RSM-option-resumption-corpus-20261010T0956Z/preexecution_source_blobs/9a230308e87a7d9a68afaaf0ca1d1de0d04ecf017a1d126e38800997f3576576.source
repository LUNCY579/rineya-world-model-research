"""Independent public-only option-resumption corpus checker, R2.

Own frozen task interpreter/graph normalizations are reused by exact hash.
No targets are used as semantic evidence. No ambient candidate/manifest reads.
The caller supplies raw PENDING candidate, public-source, manifest and legacy
bytes; accepted record projections are joined to preserved candidate records.
Authorship is /root/formal_oracle, not production approval.
"""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = {
 'schemas/RSM_OPTION_RESUMPTION_V1.md': '44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f',
 'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md': '8bda187009cc91756248ddd86792c28aabe0793a3f6be304bb9433bfdf48be34',
 'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json': '422f67c836639d0accbf411c3080ef4fc7695ad8630120a140488c90a19061fa',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md': '1612b14d7549b7fb406cbef8e127440dcfddaf30548ef29d217bc49aefb6c854',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md': '5f7f6a2c45dd6a0b51c7a912b28fd49badd9aa3aa2eac8632e9b9781c201b57b',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md': 'b62663c3082c9e2ee2bf34aadc1fdd10f263a6497396b6768f1475b12f568fe4',
}
DEPENDENCIES = {
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/option_resume_solver_v1.py': '42726260ec8a1d713c257f09efdd0910a9f30a836058ab0d7f662e95d1112684',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/legacy_contrast_v1.py': 'bb628fd918aeb82c3422a029b6ca671ff694090749f9d23d85a8a1d473cdf566',
}
LEGACY_SHA = 'ba62d973531521bfc57a5b52ee630a61f3384d9e8dfd41becea6a1eb63ca86c1'
CONTEXTS = tuple('FRTPUAISX')
STYLES = ('forward', 'reverse')
RELATIONS = {
 'uninterrupted_resume_suffix': ('F','R'), 'resume_restart_replay': ('R','T'),
 'budget_pause_after_effect': ('R','P'), 'second_checkpoint_suffix': ('P','U'),
 'positive_bound_argument_counterfactual': ('F','A'), 'initiation_argument_counterfactual': ('F','I'),
 'success_contract_counterfactual': ('F','S'), 'active_checkpoint_tamper_rejection': ('R','X'),
 'object_field_order_invariance': None,
}
INDICES = {12,13,18,14,15,19,16,17,28,20,21,29}
METRICS = {'maximum_option_depth', 'maximum_selected_branch_depth','declared_instructions',
 'total_F_events','total_F_effects','history_effects_at_R','caller_frames_at_R','selected_branch_frames_at_R'}
CLAIM_FIELDS = {'name','property','scope','evidence_manifest','control_id','pair_sample_ids',
 'source_family_key','source_family_index','controlled_context','renderer','diagnostics'}
REVIEW_FIELDS = {'record_hash','review_status','review_receipt_ids','reviewed_candidate_hash'}

def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
def digest(x): return sha256(canonical(x).encode()).hexdigest()
def record_hash(x): return digest({k:v for k,v in x.items() if k != 'record_hash'})
def is_sha(x): return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def nonempty(x): return type(x) is str and bool(x)
def need(ok, message):
    if not ok: raise ValueError(message)
def load_dependency(path):
    p=ROOT/path; need(sha256(p.read_bytes()).hexdigest()==DEPENDENCIES[path], 'dependency hash mismatch')
    spec=importlib.util.spec_from_file_location('rsm_own_'+p.stem,p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
SOLVER=load_dependency(next(iter(DEPENDENCIES)))
GRAPH=load_dependency(list(DEPENDENCIES)[1])
eq=SOLVER.typed_equal
solve=SOLVER.solve

def strict_json(raw):
    need(type(raw) is bytes, 'raw bytes required')
    def object_pairs(pairs):
        result={}
        for k,v in pairs:
            need(k not in result,'duplicate JSON key'); result[k]=v
        return result
    def bad(v): raise ValueError('nonfinite JSON number')
    return json.loads(raw.decode('utf-8'),object_pairs_hook=object_pairs,parse_constant=bad)

def safe_path(value):
    need(nonempty(value) and '\\' not in value, 'unsafe evidence path')
    p=PurePosixPath(value)
    need(not p.is_absolute() and all(c not in ('..','.') for c in value.split('/')) and '' not in value.split('/'), 'unsafe evidence path')
    current=ROOT
    for c in p.parts:
        current=current/c; need(not current.is_symlink(),'symlink evidence path')
    return value

def identifier(value):
    need(type(value) is dict,'claim/entry object required')
    names=[value[k] for k in ('name','property') if k in value]
    need(names and all(type(n) is str for n in names) and len(set(names))==1,'missing/conflicting name/property')
    need(names[0] in RELATIONS,'unknown control name'); return names[0]

def pair_ids(value):
    p=value.get('pair_sample_ids')
    need(type(p) is list and len(p)==2 and all(nonempty(x) for x in p) and p[0]!=p[1],'two distinct actual pair IDs required')
    return tuple(p)

def claims(row):
    need('family_claims' not in row,'superseded family_claims alias')
    cc=row.get('family_control_claims'); need(type(cc) is list and len(cc)==9,'exactly nine family controls required')
    out={}
    for c in cc:
        n=identifier(c); need(n not in out,'duplicate required control name')
        need(set(c)<=CLAIM_FIELDS,'unsupported canonical claim field')
        need(c.get('scope')=='corpus_variant' and type(c.get('scope')) is str,'control scope')
        safe_path(c.get('evidence_manifest')); need(nonempty(c.get('control_id')),'nonempty string control_id')
        pair_ids(c)
        for k in ('source_family_key','controlled_context','renderer'):
            if k in c: need(type(c[k]) is str and eq(c[k],row.get(k)),'claim owner '+k)
        if 'source_family_index' in c: need(type(c['source_family_index']) is int and eq(c['source_family_index'],row.get('source_family_index')),'claim owner index')
        if 'diagnostics' in c: need(type(c['diagnostics']) is dict,'claim diagnostics object')
        out[n]=c
    need(set(out)==set(RELATIONS),'missing required control'); return out

def locator_shape(p):
    need(type(p) is list and all(type(x) is str or type(x) is int and x>=0 for x in p),'typed locator path')
def at(value,p):
    locator_shape(p)
    for k in p:
        if type(value) is list: need(type(k) is int and 0<=k<len(value),'list locator index')
        elif type(value) is dict: need(type(k) is str and k in value,'object locator key')
        else: raise ValueError('locator through scalar')
        value=value[k]
    return value

def prompt_task(row):
    text=row.get('rendered_input'); need(type(text) is str and '\nTask JSON:\n' in text,'complete public prompt required')
    prefix,raw=text.rsplit('\nTask JSON:\n',1)
    public=(ROOT/'schemas/RSM_OPTION_RESUMPTION_V1.md').read_bytes()
    need(sha256(public).hexdigest()==PUBLIC['schemas/RSM_OPTION_RESUMPTION_V1.md'],'task public bytes changed')
    need(prefix.strip()==public.decode().strip(),'complete unchanged contract prefix required')
    parsed=strict_json(raw.encode()); need(eq(parsed,row.get('semantic_task')),'rendered task typed mismatch'); return parsed

def check(row):
    errors=[]
    def part(label,fn):
        try: fn()
        except (ValueError,TypeError,KeyError,IndexError,RecursionError) as e: errors.append(label+': '+str(e))
    def base():
        need(type(row) is dict,'record object')
        for k in ('sample_id','semantic_parent_id','semantic_family_id','source_family_key'): need(nonempty(row.get(k)),k)
        need(row.get('track_id')=='RSM-001','track')
        need(type(row.get('source_family_index')) is int and row['source_family_index'] in INDICES,'source index')
        need(row.get('split')==('train' if row['source_family_index']%10<8 else 'dev_challenge'),'split')
        need(type(row.get('controlled_context')) is str and row['controlled_context'] in CONTEXTS,'context')
        need(type(row.get('renderer')) is str and row['renderer'] in STYLES,'renderer')
        need(type(row.get('skill_tags')) is list and all(type(x) is str for x in row['skill_tags']),'descriptor tags')
        need(type(row.get('difficulty')) is int and row['difficulty'] in (2,3,4) and type(row.get('structural_difficulty')) is int and row['difficulty']==row['structural_difficulty'],'typed difficulty')
        prov=row.get('source_provenance'); need(type(prov) is dict and type(prov.get('curriculum_difficulty_explanation')) is str and bool(prov['curriculum_difficulty_explanation'].strip()),'difficulty explanation')
        need(is_sha(row.get('record_hash')) and row['record_hash']==record_hash(row),'full record hash')
    def task():
        SOLVER.validate_program(row['semantic_task']); result=solve(row['semantic_task'])
        expected='invalid_checkpoint' if row.get('controlled_context')=='X' else None
        need(result['error']==expected,'task/history/checkpoint stage: '+str(result['error']))
    def witnesses():
        w=row.get('property_witnesses'); need(type(w) is dict,'property_witnesses object')
        need(set(w)<={'saved_binding_locator','checkpoint_tamper_locator','replay_locator','dependency_metrics','diagnostics'},'unsupported property witness')
        for k in ('saved_binding_locator','checkpoint_tamper_locator'):
            if k in w: locator_shape(w[k])
        if 'replay_locator' in w:
            p=w['replay_locator']; need(type(p) is dict and set(p)=={'option','body_path','index'},'replay locator fields')
            path=p['body_path']
            need(nonempty(p['option']) and type(path) is list and len(path)%2==0 and all(type(v) is int and v>=0 if j%2==0 else type(v) is str and v in ('then','else') for j,v in enumerate(path)) and type(p['index']) is int and p['index']>=0,'replay locator types')
        if 'dependency_metrics' in w:
            mm=w['dependency_metrics']; need(type(mm) is dict and set(mm)<=METRICS and all(type(v) is int and v>=0 for v in mm.values()),'typed dependency metrics')
        if 'diagnostics' in w: need(type(w['diagnostics']) is dict,'witness diagnostics object')
    part('record',base); part('claim',lambda:claims(row)); part('task',task); part('prompt',lambda:prompt_task(row)); part('witness',witnesses)
    return errors

def differences(a,b,path=()):
    if eq(a,b): return []
    if type(a) is dict and type(b) is dict and set(a)==set(b):
        return [p for k in a for p in differences(a[k],b[k],path+(k,))]
    if type(a) is list and type(b) is list and len(a)==len(b):
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,path+(i,))]
    return [path]

def only_fields(a,b,allowed):
    need(all(p and p[0] in allowed for p in differences(a,b)),'unexpected task differences')
def atoms(p,state,bindings):
    return [SOLVER.predicate({'all':[a]},state,bindings) for a in p['all']]
def declared(task):
    count=0; maxdepth=0; todo=[(o['body'],0) for o in task['options'].values()]
    while todo:
        body,depth=todo.pop(); maxdepth=max(maxdepth,depth); count+=len(body)
        for i in body:
            if i['op']=='if': todo.extend(((i['then'],depth+1),(i['else'],depth+1)))
    return count,maxdepth
def original_snapshots(task,events):
    m=SOLVER.initial(task); snapshots=[deepcopy(m)]
    for e in events:
        need(m['phase']=='running' and eq(SOLVER.transition(task,m),e),'original event replay mismatch'); snapshots.append(deepcopy(m))
    return snapshots

def token(f,param): return (f['frame_kind'],f['instance_id'],f['option'],tuple(f['body_path']),param)
def observe_next(task,m):
    """Record actual pending reads, distinguishing each retained binding copy."""
    SOLVER.normalize(task,m); reads=[]
    def expr(e,f):
        v=SOLVER.eval_expr(e,m['state'],{} if f is None else f['bindings'])
        if 'reg' in e: reads.append(('register',e['reg'],v))
        if 'param' in e and f is not None: reads.append(('binding',token(f,e['param']),v))
        return v
    def pred(p,f):
        for a in p['all']: expr(a['left'],f); expr(a['right'],f)
    if m['root_pending']:
        for e in task['entry']['args'].values(): expr(e,None)
    else:
        f=m['frames'][-1]; b=SOLVER.selected_body(task,f); k=f['next_index']
        need(type(k) is int and 0<=k<=len(b),'trusted counterfactual cursor out of range')
        if k==len(b): pred(task['options'][f['option']]['success'],f)
        else:
            i=b[k]
            if i['op'] in ('add','scale'): expr(i['value'],f); reads.append(('register',i['register'],m['state'][i['register']]))
            elif i['op']=='swap': reads.extend(('register',r,m['state'][r]) for r in ('x','y'))
            elif i['op']=='if': pred(i['test'],f)
            else:
                for e in i['args'].values(): expr(e,f)
    return reads
def trusted(task,m):
    """Diagnostic continuation, never accepts an invalid supplied checkpoint."""
    m=deepcopy(m); events=[]; reads=[]
    # Finite configured programs have <=48 complete events. A retained suffix
    # cannot create additional invocations; 48 is the published cohort bound.
    for _ in range(49):
        if m['phase']!='running': return events,reads,m
        reads.extend(observe_next(task,m)); events.append(SOLVER.transition(task,m))
    raise ValueError('UNKNOWN: trusted continuation exceeds configured horizon')

def origins(task,history):
    by_id={}
    for e in history:
        if e['event']!='enter' or not e['initiation_ok']: continue
        if e['call_site'] is None: aa=task['entry']['args']; parent={}
        else:
            site=e['call_site']; caller=next(x for x in history if x['event']=='enter' and x['instance_id']==site['caller_instance'])
            f={'option':caller['option'],'body_path':site['body_path']}; i=SOLVER.selected_body(task,f)[site['index']]
            aa=i['args']; parent=by_id[site['caller_instance']]
        by_id[e['instance_id']]={p: ('reg' in x or 'param' in x and parent.get(x['param'],False)) for p,x in aa.items()}
    return by_id
def primitive(task,e):
    return SOLVER.selected_body(task,{'option':e['option'],'body_path':e['body_path']})[e['index']]
def locator(e): return {'option':e['option'],'body_path':e['body_path'],'index':e['index']}
def apply_effect(i,state,bindings):
    out=deepcopy(state)
    if i['op']=='swap': out['x'],out['y']=out['y'],out['x']
    else:
        v=SOLVER.eval_expr(i['value'],out,bindings); r=i['register']; out[r]=out[r]+v if i['op']=='add' else out[r]*v
    return out
def replay_witnesses(F,R,T,targets):
    f=targets['F']; t=targets['T']; snapshots=original_snapshots(F,f['new_events']); result=[]
    bindings={e['instance_id']:e['bindings'] for e in f['new_events'] if e['event']=='enter' and e['initiation_ok']}
    for j,e in enumerate(R['history']):
        if e['event']!='effect': continue
        i=primitive(F,e); s=snapshots[j]['state']; one=apply_effect(i,s,bindings[e['instance_id']]); two=apply_effect(i,one,bindings[e['instance_id']])
        if eq(one,two): continue
        if any(q['event']=='effect' and eq(locator(e),locator(q)) for q in t['new_events']): result.append(locator(e))
    return result

def family_analysis(tasks):
    F,R,T,P,U,A,I,S,X=(tasks[c] for c in CONTEXTS)
    targets={c:solve(t) for c,t in tasks.items()}
    f,r,t,p,u,a,i,s,x=(targets[c] for c in CONTEXTS)
    need(all(targets[c]['error'] is None for c in CONTEXTS if c!='X') and x['error']=='invalid_checkpoint','family validation stages')
    need(F['mode']=='fresh' and not F['history'] and F['checkpoint'] is None,'F fresh')
    for c in ('F','R','T','U','A','I','S','X'): need(tasks[c]['event_budget']==48 and type(tasks[c]['event_budget']) is int,'full budget '+c)
    need(f['status']=='completed' and f['new_effects']>=3,'F complete/minimum effects')
    need(2<=len(F['options'])<=4 and all(1<=len(o['params'])<=3 for o in F['options'].values()),'option/parameter bounds')
    n,cond=declared(F); need(n<=24 and cond<=2,'instruction/conditional bounds')
    snapshots=original_snapshots(F,f['new_events']); need(len(f['new_events'])<=48,'complete event bound')
    need(all(-6<=v<=6 for v in F['initial_state'].values()),'initial range')
    for c,z in targets.items():
        states=[tasks[c]['initial_state']]+[e['state'] for e in tasks[c]['history']+z['new_events'] if e['event']=='effect']
        need(all(type(v) is int and -4096<=v<=4096 for st in states for v in st.values()),'run state range '+c)
    depth=max(sum(v['frame_kind']=='option' for v in z['frames']) for z in snapshots)
    branch_depth=max(sum(v['frame_kind']=='branch' for v in z['frames']) for z in snapshots)
    need(depth<=3,'simultaneous option depth')
    only_fields(F,R,{'mode','history','checkpoint'}); need(R['mode']=='resume' and R['history'],'R prefix')
    cut=len(R['history']); need(cut<len(f['new_events']) and eq(R['history'],f['new_events'][:cut]) and eq(R['checkpoint'],snapshots[cut]),'R original prefix/snapshot')
    need(any(e['event']=='effect' for e in R['history']) and r['new_events'] and r['status']=='completed' and eq(r['new_events'],f['new_events'][cut:]) and eq(r['checkpoint'],f['checkpoint']),'uninterrupted suffix')
    cp=R['checkpoint']; opts=[v for v in cp['frames'] if v['frame_kind']=='option']; branches=[v for v in cp['frames'] if v['frame_kind']=='branch']
    need(len(opts)>=2 and branches,'nested callers and selected branch at R')
    need(any(v['next_index']<len(SOLVER.selected_body(R,v)) for v in opts[:-1]),'caller pending work/cursor')
    need(any(not SOLVER.predicate(R['options'][v['option']]['initiation'],cp['state'],v['bindings']) for v in opts[1:]),'entered nonroot current initiation false')
    guard_reversed=[]
    for v in branches:
        parent={'option':v['option'],'body_path':v['body_path'][:-2]}; instruction=SOLVER.selected_body(R,parent)[v['body_path'][-2]]
        if v['body_path'][-1]=='then' and not SOLVER.predicate(instruction['test'],cp['state'],v['bindings']): guard_reversed.append(v['body_path'])
    need(guard_reversed,'selected true branch current guard false')
    future,reads,_=trusted(R,cp); origin=origins(R,R['history']); saved=[]
    for j,v in enumerate(cp['frames']):
        for param in v['bindings']:
            if origin.get(v['instance_id'],{}).get(param) and any(q[0]=='binding' and q[1]==token(v,param) for q in reads): saved.append(['frames',j,'bindings',param])
    need(saved,'particular retained register-derived argument read by pending work')
    only_fields(R,T,{'mode'}); need(T['mode']=='restart' and differences(R,T)==[('mode',)],'mode-only restart')
    need(t['new_events'] and t['new_events'][0]['event']=='enter' and t['new_events'][0]['initiation_ok'],'restart root initiation')
    replay=replay_witnesses(F,R,T,targets); need(replay,'committed non-idempotent effect actually reexecuted')
    need(not eq(t['new_events'],r['new_events']) and (t['status']!=r['status'] or not eq(t['checkpoint']['state'],r['checkpoint']['state'])),'restart trace and terminal difference')
    only_fields(R,P,{'event_budget'}); need(type(P['event_budget']) is int and 0<P['event_budget']<len(r['new_events']),'positive shorter P budget')
    need(p['status']=='paused' and p['new_effects']>=1 and p['next_event'] is not None and eq(p['new_events'],r['new_events'][:len(p['new_events'])]),'P commits effect and leaves semantic work')
    only_fields(R,U,{'history','checkpoint'}); need(U['mode']=='resume' and eq(U['history'],R['history']+p['new_events']) and eq(U['checkpoint'],p['checkpoint']),'second checkpoint history/snapshot')
    need(u['status']==f['status'] and eq(u['new_events'],f['new_events'][len(U['history']):]) and eq(u['checkpoint'],f['checkpoint']),'second exact suffix')
    def argument_change(z):
        only_fields(F,z,{'entry'}); need(z['entry']['option']==F['entry']['option'],'root same')
        changed=[k for k in F['entry']['args'] if not eq(F['entry']['args'][k],z['entry']['args'][k])]
        need(len(changed)==1,'one root argument expression'); return changed[0]
    changed=argument_change(A); need(a['status']=='completed','positive A completes')
    need(not eq(f['new_events'][0]['bindings'][changed],a['new_events'][0]['bindings'][changed]),'A binding actually changes')
    qualified=lambda z:[e for e in z['new_events'] if e['event']!='enter']
    need(not eq(qualified(f),qualified(a)),'A later behavior difference, not unused enter binding')
    start=snapshots[1]; changed_start=deepcopy(start); changed_start['frames'][0]['bindings'][changed]=a['new_events'][0]['bindings'][changed]
    altered,argument_reads,_=trusted(F,changed_start)
    need(eq(altered,a['new_events'][1:]) and any(q[0]=='binding' and q[1][-1]==changed for q in argument_reads),'actual root argument causal continuation')
    argument_change(I); need(i['status']=='initiation_failed' and i['new_effects']==0 and len(i['new_events'])==1,'I decisive initiation failure')
    root=F['entry']['option']; fa=atoms(F['options'][root]['initiation'],F['initial_state'],f['new_events'][0]['bindings']); ia=atoms(I['options'][root]['initiation'],I['initial_state'],i['new_events'][0]['bindings'])
    need(all(fa) and sum(not v for v in ia)==1,'I one false atom, all others true')
    diff=differences(F,S); need(len(diff)==1 and diff[0][:4]==('options',root,'success','all') and diff[0][-1]=='const' and type(at(S,list(diff[0]))) is int,'one root success literal')
    need(s['status']=='success_failed' and eq([e for e in f['new_events'] if e['event']=='effect'],[e for e in s['new_events'] if e['event']=='effect']) and eq(s['checkpoint']['state'],f['checkpoint']['state']),'S committed effects unchanged')
    need(s['new_events'][-1]['event']=='return' and s['new_events'][-1]['instance_id']==0 and not s['new_events'][-1]['success_ok'],'opposite root return')
    sa=atoms(S['options'][root]['success'],s['checkpoint']['state'],f['new_events'][0]['bindings']); need(sum(not z for z in sa)==1,'S all other success atoms true')
    only_fields(R,X,{'checkpoint'}); xd=differences(R['checkpoint'],X['checkpoint']); need(len(xd)==1,'one checkpoint leaf')
    xp=list(xd[0]); need(type(at(cp,xp)) is int and type(at(X['checkpoint'],xp)) is int,'integer active checkpoint scalar')
    active=False; mutant,mutant_reads,mutant_cp=trusted(R,X['checkpoint'])
    if len(xp)==2 and xp[0]=='state' and xp[1] in ('x','y'):
        reg=xp[1]; active=any(q[0]=='register' and q[1]==reg and q not in reads for q in mutant_reads) or not eq(mutant_cp['state'],r['checkpoint']['state'])
    elif len(xp)>=3 and xp[0]=='frames' and type(xp[1]) is int:
        j=xp[1]
        if xp[2:] == ['next_index']: active=not eq(mutant,r['new_events'])
        elif len(xp)==4 and xp[2]=='bindings':
            tok=token(cp['frames'][j],xp[3]); active=any(q[0]=='binding' and q[1]==tok and q not in reads for q in mutant_reads)
    need(active,'particular checkpoint tamper affects pending computation')
    metrics={'maximum_option_depth':depth,'maximum_selected_branch_depth':branch_depth,'declared_instructions':n,
             'total_F_events':len(f['new_events']),'total_F_effects':f['new_effects'],'history_effects_at_R':r['previous_effects'],
             'caller_frames_at_R':len(opts)-1,'selected_branch_frames_at_R':len(branches)}
    return {'targets':targets,'metrics':metrics,'saved_binding_locators':saved,'checkpoint_tamper_locator':xp,
            'replay_locators':replay,'cut':cut,'statuses':{c:z['status'] for c,z in targets.items()},
            'effects':{c:z['new_effects'] for c,z in targets.items()},'horizons':{c:len(z['new_events']) for c,z in targets.items()}}

def reverse_presentation(a,b):
    if type(a) is not type(b): return False
    if type(a) is dict: return list(a)==list(reversed(b)) and all(reverse_presentation(a[k],b[k]) for k in a)
    if type(a) is list: return len(a)==len(b) and all(reverse_presentation(x,y) for x,y in zip(a,b))
    return a==b

def check_batch(rows,manifest_sources=None,source_bytes=None,candidate_bytes=None,legacy_input_bytes=None):
    report={'errors':[],'counts':{},'custody':{},'families':[],'controls':[],'novelty':None,'production_data_approval':False}
    errors=report['errors']
    def fail(label,e): errors.append(label+': '+str(e))
    if type(rows) is not list: fail('UNKNOWN','records list required'); return report
    for j,row in enumerate(rows):
        for e in check(row): fail('row '+str(j),e)
    report['counts']['records']=len(rows)
    current={}
    for row in rows:
        if type(row) is not dict or not nonempty(row.get('sample_id')): continue
        if row['sample_id'] in current: fail('coverage','duplicate sample ID')
        current[row['sample_id']]=row
    pending={}; manifests={}; entries={}; candidate_sha=None
    try:
        need(type(candidate_bytes) is bytes,'UNKNOWN: exact raw PENDING candidate bytes required')
        original=[strict_json(line) for line in candidate_bytes.splitlines() if line.strip()]; candidate_sha=sha256(candidate_bytes).hexdigest()
        need(len(original)==len(rows),'complete candidate/current projection membership')
        for row in original:
            need(type(row) is dict and nonempty(row.get('sample_id')) and row['sample_id'] not in pending,'pending duplicate/invalid sample')
            need(row.get('review_status')=='PENDING' and row.get('record_hash')==record_hash(row),'PENDING full record custody')
            pending[row['sample_id']]=row
        need(set(current)==set(pending),'complete actual sample join')
        for sid,row in current.items():
            source=pending[sid]
            if row.get('review_status')=='ACCEPTED':
                need(row.get('reviewed_candidate_hash')==source['record_hash'],'accepted original record lineage')
                need(eq({k:v for k,v in row.items() if k not in REVIEW_FIELDS},{k:v for k,v in source.items() if k not in REVIEW_FIELDS}),'accepted typed immutable projection')
            else: need(eq(row,source),'current PENDING row differs from exact source')
        report['custody']['candidate_sha256']=candidate_sha
    except (ValueError,TypeError,KeyError) as e: fail('candidate',e)
    try:
        need(type(source_bytes) is dict,'UNKNOWN: raw public-source byte ledger required')
        for path,sha in PUBLIC.items(): need(type(source_bytes.get(path)) is bytes and sha256(source_bytes[path]).hexdigest()==sha,'UNKNOWN/mismatched operative source '+path)
        report['custody']['public_sources']={p:sha256(source_bytes[p]).hexdigest() for p in PUBLIC}
    except (ValueError,TypeError,KeyError) as e: fail('sources',e)
    try:
        need(type(manifest_sources) is dict and manifest_sources,'UNKNOWN: raw manifest byte ledger required')
        for path,raw in manifest_sources.items():
            safe_path(path); m=strict_json(raw); need(type(m) is dict,'manifest object')
            need(type(m.get('metadata_version')) is int and m['metadata_version']==1 and m.get('batch')=='option-resumption-pilot-v1','manifest version/batch')
            need(is_sha(m.get('candidate_sha256')) and m['candidate_sha256']==candidate_sha,'whole PENDING candidate SHA')
            sb=m.get('source_bindings'); need(type(sb) is dict and set(PUBLIC)<=set(sb),'six public source bindings')
            for p,sh in sb.items(): safe_path(p); need(is_sha(sh) and type(source_bytes) is dict and type(source_bytes.get(p)) is bytes and sha256(source_bytes[p]).hexdigest()==sh,'asserted raw source binding '+p)
            ee=m.get('control_pairs'); need(type(ee) is list,'manifest control_pairs list')
            local={}
            for entry in ee:
                n=identifier(entry); ids=pair_ids(entry); need(nonempty(entry.get('control_id')) and entry.get('scope')=='corpus_variant','entry control ID/scope')
                key=(entry['control_id'],n,ids); need(key not in local,'duplicate resolver key')
                need(entry.get('candidate_sha256')==candidate_sha,'entry complete candidate SHA')
                need(all(sid in current for sid in ids),'entry actual members')
                left,right=(current[sid] for sid in ids)
                need(type(entry.get('source_family_index')) is int and entry['source_family_index']==left['source_family_index'] and entry.get('source_family_key')==left['source_family_key'],'entry source coordinates')
                need(eq(entry.get('controlled_contexts'),[left['controlled_context'],right['controlled_context']]) and eq(entry.get('renderers'),[left['renderer'],right['renderer']]),'entry actual ordered coordinates')
                if 'pair_record_hashes' in entry:
                    pp=entry['pair_record_hashes']; need(type(pp) is list and len(pp)==2 and all(is_sha(x) for x in pp) and eq(pp,[pending[sid]['record_hash'] for sid in ids]),'external actual ordered full member hashes')
                local[key]=entry
            manifests[path]=m; entries[path]=local
        report['custody']['manifests']={p:sha256(raw).hexdigest() for p,raw in manifest_sources.items()}
    except (ValueError,TypeError,KeyError,IndexError) as e: fail('manifest',e)
    groups={}; indexes={}; countclaims=0; nonmember=0; resolved=set()
    for row in rows:
        if type(row) is not dict: continue
        try:
            k=row['source_family_key']; idx=row['source_family_index']; coord=(row['controlled_context'],row['renderer'])
            need(k not in indexes or indexes[k]==idx,'family index changes'); indexes[k]=idx
            g=groups.setdefault(k,{}); need(coord not in g,'duplicate context/style'); g[coord]=row
            for n,c in claims(row).items():
                countclaims+=1; ids=pair_ids(c); need(all(v in current for v in ids),'claim missing actual member')
                left,right=(current[v] for v in ids)
                need(all(v['source_family_key']==k and type(v['source_family_index']) is int and v['source_family_index']==idx and v['split']==row['split'] for v in (left,right)),'claim foreign source/index/split')
                if n=='object_field_order_invariance':
                    need([left['controlled_context'],right['controlled_context']]==[row['controlled_context']]*2 and [left['renderer'],right['renderer']]==list(STYLES) and row['sample_id'] in ids,'owner field-order pair')
                    need(left['semantic_parent_id']==right['semantic_parent_id'] and eq(left['semantic_task'],right['semantic_task']) and reverse_presentation(prompt_task(left),prompt_task(right)),'field-order typed/list/parent/presentation invariant')
                else:
                    need((left['controlled_context'],right['controlled_context'])==RELATIONS[n] and left['renderer']==right['renderer']==row['renderer'],'ordered control context/style')
                    nonmember+=row['sample_id'] not in ids
                path=c['evidence_manifest']; key=(c['control_id'],n,ids); need(path in entries and key in entries[path],'missing exact named resolver entry')
                resolved.add((path,key))
        except (ValueError,TypeError,KeyError,IndexError,RecursionError) as e: fail('owner '+str(row.get('sample_id')),e)
    need_coords={(c,s) for c in CONTEXTS for s in STYLES}
    if len(groups)!=12 or set(indexes.values())!=INDICES or len(indexes.values())!=len(set(indexes.values())): fail('coverage','twelve configured family/index mapping')
    for family,g in groups.items():
        try:
            need(set(g)==need_coords,'all nine contexts/two renderers')
            facts=family_analysis({c:g[c,'forward']['semantic_task'] for c in CONTEXTS})
            need(all(eq(g[c,'forward']['semantic_task'],g[c,'reverse']['semantic_task']) for c in CONTEXTS),'style task equality')
            for row in g.values():
                w=row['property_witnesses']
                for n,v in w.get('dependency_metrics',{}).items(): need(v==facts['metrics'][n],'false dependency metric '+n)
                if 'saved_binding_locator' in w: need(w['saved_binding_locator'] in facts['saved_binding_locators'],'inactive/unrelated saved frame locator')
                if 'checkpoint_tamper_locator' in w: need(eq(w['checkpoint_tamper_locator'],facts['checkpoint_tamper_locator']),'false tamper locator')
                if 'replay_locator' in w: need(any(eq(w['replay_locator'],v) for v in facts['replay_locators']),'false/blocked/idempotent replay locator')
            report['families'].append({'family':family,'index':indexes[family],**{k:v for k,v in facts.items() if k!='targets'}})
            for name,contexts in RELATIONS.items():
                for style in (STYLES if contexts else CONTEXTS):
                    ids=[g[c,style]['sample_id'] for c in contexts] if contexts else [g[style,s]['sample_id'] for s in STYLES]
                    report['controls'].append({'family':family,'property':name,'pair_sample_ids':ids,'proved':True})
        except (ValueError,TypeError,KeyError,IndexError,RecursionError,AssertionError) as e: fail('family '+str(family),e)
    proposed=[{'source_family_index':indexes[k],'task':g['F','forward']['semantic_task']} for k,g in groups.items() if ('F','forward') in g]
    try:
        need(type(legacy_input_bytes) is bytes and sha256(legacy_input_bytes).hexdigest()==LEGACY_SHA,'UNKNOWN: exact target-free frozen legacy input bytes required')
        legacy=[strict_json(line) for line in legacy_input_bytes.splitlines() if line.strip()]
        report['novelty']=GRAPH.compare(proposed,legacy)
        for n,v in report['novelty']['graph_views'].items(): need(v['proposed_distinct_signatures']==12 and not v['proposed_duplicate_pairs'] and not v['exact_new_legacy_collisions'],'graph novelty '+n)
        report['custody']['legacy_sha256']=sha256(legacy_input_bytes).hexdigest()
    except (ValueError,TypeError,KeyError,RecursionError) as e: fail('novelty',e)
    semantic={canonical(r['semantic_task']) for r in rows if type(r) is dict and 'semantic_task' in r}
    allentries=sum(len(v) for v in entries.values())
    report['counts'].update(source_families=len(groups),semantic_tasks=len(semantic),owner_claims=countclaims,nonmember_first8_claims=nonmember,
                            resolved_named_pairs=len(resolved),manifest_named_entries=allentries,proved_named_pairs=len(report['controls']),
                            train_records=sum(r.get('split')=='train' for r in rows if type(r) is dict),dev_challenge_records=sum(r.get('split')=='dev_challenge' for r in rows if type(r) is dict))
    expected={'records':216,'source_families':12,'semantic_tasks':108,'owner_claims':1944,'nonmember_first8_claims':1344,'resolved_named_pairs':300,'proved_named_pairs':300,'train_records':144,'dev_challenge_records':72}
    for n,v in expected.items():
        if report['counts'].get(n)!=v: fail('counts',n+' expected '+str(v))
    report['counts']['additional_manifest_entries']=allentries-len(resolved)
    report['histograms']={'status_by_context':{},'event_horizon_by_context':{},'effects_by_context':{},'mode_status':{},'history_length_status':{}}
    statuses=[]
    for row in rows:
        if type(row) is not dict or type(row.get('semantic_task')) is not dict: continue
        z=solve(row['semantic_task']); status=z['status']; statuses.append(status)
        for field,value in (('status_by_context',status),('event_horizon_by_context',str(len(z['new_events']))),('effects_by_context',str(z['new_effects']))):
            slot=report['histograms'][field].setdefault(row.get('controlled_context','UNKNOWN'),{}); slot[value]=slot.get(value,0)+1
        for field,key in (('mode_status',str(row['semantic_task'].get('mode'))),('history_length_status',str(len(row['semantic_task'].get('history',[])))):
            slot=report['histograms'][field].setdefault(key,{}); slot[status]=slot.get(status,0)+1
    counts={z:statuses.count(z) for z in sorted(set(statuses))}
    report['input_shortcut_diagnostics']={'records':len(statuses),'global_majority_correct':max(counts.values(),default=0),
      'best_mode_bucket_correct':sum(max(v.values()) for v in report['histograms']['mode_status'].values()),
      'best_history_length_bucket_correct':sum(max(v.values()) for v in report['histograms']['history_length_status'].values()),
      'fit_scope':'descriptive in-corpus best bucket labels, no trained model or held-out efficacy',
      'finite_program_exposure':'12 hand programs; contexts/styles are correlated supplied-policy traces, no sealed transfer'}
    return report
