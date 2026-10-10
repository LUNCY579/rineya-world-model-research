"""Own authorized packet only. Source/public/runtime freeze precedes helper imports.
No canonical gold, producer, formal, other reviewer or private metadata reads.
"""
import argparse,collections,copy,datetime,hashlib,importlib.util,io,json,os,pathlib,re,sys,time,traceback
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def typed(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key '+k)
        d[k]=v
    return d
def loads(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def need(v,msg):
    if not v:raise AssertionError(msg)
def newjson(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
def newjsonl(p,values):
    with p.open('x',encoding='utf-8') as f:
        for v in values:f.write(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
def load_helper(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify(freeze):
    checked=[]
    for v in freeze['bindings']:
        p=pathlib.Path(v['path']) if v['path'].startswith('/') else BASE/v['path'];raw=p.read_bytes();arch=(BASE/v['archive']).read_bytes()
        need(len(raw)==v['bytes'] and sha(raw)==v['sha256'] and raw==arch,'changed frozen dependency '+v['path']);checked.append({'path':v['path'],'sha256':v['sha256'],'bytes':v['bytes'],'archive_sha256':sha(arch)})
    need(str(pathlib.Path(sys.executable).resolve())==freeze['resolved_python'],'wrong resolved Python')
    return checked

def code_body(t,opt,path):
    items=t['options'][opt]['body']
    for i in range(0,len(path),2):items=items[path[i]][path[i+1]]
    return items

def fold_events(t,events):
    """Reconstruct prescribed snapshots from separate recursive events, not Machine."""
    state=copy.deepcopy(t['initial_state']);frames=[];counter=0;pending=True;phase='running'
    def normalize():
        while frames and frames[-1]['frame_kind']=='branch' and frames[-1]['next_index']==len(code_body(t,frames[-1]['option'],frames[-1]['body_path'])):frames.pop()
    for e in events:
        need(phase=='running','event beyond terminal phase');normalize();kind=e['event']
        if kind=='enter':
            need(e['instance_id']==counter,'counter mismatch');counter+=1
            if e['call_site'] is None:need(pending,'root enter twice');pending=False
            else:
                f=frames[-1];site=e['call_site'];need(f['instance_id']==site['caller_instance'] and f['next_index']==site['index'] and typed(f['body_path'],site['body_path']),'call site mismatch');f['next_index']+=1
            if e['initiation_ok']:frames.append({'frame_kind':'option','instance_id':e['instance_id'],'option':e['option'],'bindings':copy.deepcopy(e['bindings']),'body_path':[],'next_index':0})
            else:phase='initiation_failed'
        elif kind=='branch':
            f=frames[-1];need(f['instance_id']==e['instance_id'] and f['next_index']==e['index'] and typed(f['body_path'],e['body_path']),'branch locator mismatch');f['next_index']+=1
            frames.append({'frame_kind':'branch','instance_id':e['instance_id'],'option':e['option'],'bindings':copy.deepcopy(f['bindings']),'body_path':e['body_path']+[e['index'],'then' if e['condition'] else 'else'],'next_index':0})
        elif kind=='effect':
            f=frames[-1];need(f['instance_id']==e['instance_id'] and f['next_index']==e['index'] and typed(f['body_path'],e['body_path']),'effect locator mismatch');f['next_index']+=1;state=copy.deepcopy(e['state'])
        elif kind=='return':
            f=frames[-1];need(f['frame_kind']=='option' and f['instance_id']==e['instance_id'] and f['next_index']==len(code_body(t,f['option'],[])),'return before exhaustion');frames.pop();normalize()
            if not e['success_ok']:phase='success_failed'
            elif not frames:phase='completed'
        else:raise AssertionError('unknown recursive event')
        normalize()
    return {'state':state,'frames':frames,'next_instance_id':counter,'root_pending':pending,'phase':phase}

def recursive_prediction(t,s,r):
    """All output fields from recursive full traces and event folds; parser shared."""
    try:s.parse_task(t)
    except (s.InvalidTask,TypeError,KeyError,RecursionError):return s.invalid('invalid_task'),None
    rec=r.simulate(t);events=rec['events'];history=t['history']
    if type(history) is not list or len(history)>len(events) or not typed(history,events[:len(history)]):return s.invalid('invalid_history'),rec
    start=fold_events(t,history)
    if t['mode']=='fresh':
        if history or t['checkpoint'] is not None:return s.invalid('invalid_checkpoint'),rec
    elif not typed(start,t['checkpoint']):return s.invalid('invalid_checkpoint'),rec
    if t['mode']=='restart':
        run=copy.deepcopy(t);run['initial_state']=copy.deepcopy(start['state']);fresh=r.simulate(run);all_events=fresh['events'];new_events=all_events[:t['event_budget']];snap=fold_events(run,new_events)
    else:
        all_events=events[len(history):];new_events=all_events[:t['event_budget']];snap=fold_events(t,history+new_events)
    paused=snap['phase']=='running'
    need(not paused or len(new_events)<len(all_events),'finite recursive run unexpectedly incomplete')
    return {'status':'paused' if paused else snap['phase'],'error':None,'checkpoint':snap,'new_events':new_events,'previous_effects':sum(e['event']=='effect' for e in history),'new_effects':sum(e['event']=='effect' for e in new_events),'next_event':all_events[len(new_events)] if paused else None},rec

def primitives(t):
    nodes=0;depth=0
    def visit(items,d):
        nonlocal nodes,depth
        for i in items:
            nodes+=1
            if i['op']=='if':depth=max(depth,d+1);visit(i['then'],d+1);visit(i['else'],d+1)
    for opt in t['options'].values():visit(opt['body'],0)
    return {'options':len(t['options']),'parameters_per_option':{k:len(v['params']) for k,v in t['options'].items()},'declared_instructions':nodes,'syntactic_conditional_depth':depth}

def diff(a,b,path=()):
    if type(a)!=type(b):return [{'path':list(path),'before':a,'after':b}]
    if type(a) is dict:
        if set(a)!=set(b):return [{'path':list(path),'before':a,'after':b}]
        return [d for k in a for d in diff(a[k],b[k],path+(k,))]
    if type(a) is list:
        if len(a)!=len(b):return [{'path':list(path),'before':a,'after':b}]
        return [d for k,(x,y) in enumerate(zip(a,b)) for d in diff(x,y,path+(k,))]
    return [] if a==b else [{'path':list(path),'before':a,'after':b}]

def trusted_continuation(t,s,snapshot):
    m=s.Machine(t,snapshot=snapshot);events=[]
    for _ in range(48):
        e=m.step()
        if e is None:break
        events.append(e)
    return {'events':events,'checkpoint':m.snapshot()}

def frame_facts(t,s):
    cp=t['checkpoint']
    if type(cp) is not dict or cp.get('phase')!='running':return {}
    active=[];base=None
    for fi,f in enumerate(cp['frames']):
        fact={'frame_index':fi,'frame_kind':f['frame_kind'],'option':f['option'],'body_path':f['body_path'],'next_index':f['next_index'],'bindings':f['bindings']}
        if f['frame_kind']=='option' and f['instance_id']!=0:fact['current_initiation_is_false']=not s.test(t['options'][f['option']]['initiation'],cp['state'],f['bindings'])
        if f['frame_kind']=='branch':
            owner_path=f['body_path'][:-2];ins=code_body(t,f['option'],owner_path)[f['body_path'][-2]];now=s.test(ins['test'],cp['state'],f['bindings']);fact['current_condition']=now;fact['selected_branch_would_change_now']=now!=(f['body_path'][-1]=='then')
        fact['active_binding_probes']=[]
        for param,value in f['bindings'].items():
            try:
                if base is None:base=trusted_continuation(t,s,cp)
                mutated=copy.deepcopy(cp);mutated['frames'][fi]['bindings'][param]=value+1;probe=trusted_continuation(t,s,mutated)
                if not typed(base,probe):fact['active_binding_probes'].append({'parameter':param,'changed_value':value+1,'trusted_pending_continuation_changes':True})
            except (AssertionError,KeyError,IndexError):fact['trusted_probe_error']=True
        active.append(fact)
    return {'active_frames':active,'active_option_frames':sum(f['frame_kind']=='option' for f in cp['frames']),'active_branch_frames':sum(f['frame_kind']=='branch' for f in cp['frames']),'probe_scope':'diagnostic trusted machine only, not permission to accept changed checkpoints'}

def locate_roles(parents,outputs):
    roles={};pairs=[]
    for pid,row in parents.items():
        t=row['semantic_task'];o=outputs[row['sample_id']]
        if t['mode']=='restart':roles['T']=pid
        elif t['mode']=='fresh' and o['status']=='initiation_failed':roles['I']=pid
        elif t['mode']=='fresh' and o['status']=='success_failed':roles['S']=pid
        elif t['mode']=='resume' and o['status']=='paused':roles['P']=pid
        elif t['mode']=='resume' and o['status']=='invalid':roles['X']=pid
        elif t['mode']=='resume' and o['status']=='completed':pairs.append(pid)
    if len(pairs)==2:
        pairs.sort(key=lambda p:len(parents[p]['semantic_task']['history']));roles['R'],roles['U']=pairs
        rt=parents[roles['R']]['semantic_task']
        for pid,row in parents.items():
            t=row['semantic_task'];o=outputs[row['sample_id']]
            if t['mode']=='fresh' and o['status']=='completed':
                same=typed({k:t[k] for k in ('initial_state','options','entry')},{k:rt[k] for k in ('initial_state','options','entry')});roles['F' if same else 'A']=pid
    return roles

def family_controls(parents,roles,outputs,s):
    if set(roles)!=set('FRTPUAISX'):return {'roles_complete':False}
    tasks={k:parents[p]['semantic_task'] for k,p in roles.items()};out={k:outputs[parents[p]['sample_id']] for k,p in roles.items()}
    F,R,T,P,U,A,I,S,X=(tasks[k] for k in 'FRTPUAISX')
    checks={}
    checks['F_R_prefix_suffix']=typed(R['history'],out['F']['new_events'][:len(R['history'])]) and typed(R['history']+out['R']['new_events'],out['F']['new_events']) and typed(out['R']['checkpoint'],out['F']['checkpoint'])
    checks['R_T_only_mode']=len(diff(R,T))==1 and diff(R,T)[0]['path']==['mode']
    checks['R_T_root_pass_and_actual_difference']=bool(out['T']['new_events']) and out['T']['new_events'][0]['event']=='enter' and out['T']['new_events'][0]['initiation_ok'] and not typed(out['R']['new_events'],out['T']['new_events']) and (out['R']['status']!=out['T']['status'] or not typed(out['R']['checkpoint']['state'],out['T']['checkpoint']['state']))
    replay=[];bindings={};state=copy.deepcopy(R['initial_state'])
    for e in R['history']:
        if e['event']=='enter':bindings[e['instance_id']]=e['bindings']
        if e['event']=='effect':
            opt=e['option'];path=e['body_path'];idx=e['index'];ins=code_body(R,opt,path)[idx];saved=bindings[e['instance_id']]
            def transform(st):
                z=copy.deepcopy(st)
                if ins['op']=='swap':z['x'],z['y']=z['y'],z['x']
                else:
                    value=s.evaluate(ins['value'],z,saved);reg=ins['register'];z[reg]=z[reg]+value if ins['op']=='add' else z[reg]*value
                return z
            once=transform(state);twice=transform(once)
            hits=[j for j,v in enumerate(out['T']['new_events']) if v['event']=='effect' and (v['option'],v['body_path'],v['index'])==(opt,path,idx)]
            if not typed(once,twice) and hits:replay.append({'original_instance':e['instance_id'],'option':opt,'body_path':path,'index':idx,'original_prestate':state,'once':once,'twice':twice,'T_event_indices':hits})
            state=e['state']
    checks['actual_nonidempotent_replay_witness']=bool(replay)
    checks['R_P_only_budget']=len(diff(R,P))==1 and diff(R,P)[0]['path']==['event_budget'] and out['P']['status']=='paused' and out['P']['new_effects']>=1
    checks['P_U_exact_extended_history_snapshot']=typed(U['history'],P['history']+out['P']['new_events']) and typed(U['checkpoint'],out['P']['checkpoint']) and out['U']['status']=='completed' and typed(out['U']['checkpoint'],out['F']['checkpoint'])
    da=diff(F,A);di=diff(F,I);ds=diff(F,S);dx=diff(R,X)
    checks['F_A_one_root_argument_positive_behavior']=len(da)==1 and da[0]['path'][:2]==['entry','args'] and out['A']['status']=='completed' and any(not typed(x,y) for x,y in zip([e for e in out['F']['new_events'] if e['event']!='enter'],[e for e in out['A']['new_events'] if e['event']!='enter']))
    checks['F_I_one_root_argument_no_effects']=len(di)==1 and di[0]['path'][:2]==['entry','args'] and out['I']['status']=='initiation_failed' and out['I']['new_effects']==0
    checks['F_S_one_success_literal_same_effects']=len(ds)==1 and ds[0]['path'][:3]==['options',F['entry']['option'],'success'] and typed([e for e in out['F']['new_events'] if e['event']=='effect'],[e for e in out['S']['new_events'] if e['event']=='effect']) and out['S']['status']=='success_failed'
    checks['R_X_one_checkpoint_scalar_rejected']=len(dx)==1 and dx[0]['path'][0]=='checkpoint' and not isinstance(dx[0]['before'],(dict,list)) and out['X']['status']=='invalid' and out['X']['error']=='invalid_checkpoint' and not out['X']['new_events']
    facts=frame_facts(R,s)
    checks['R_nonroot_entry_now_false']=any(f.get('current_initiation_is_false',False) for f in facts.get('active_frames',[]))
    checks['R_selected_branch_now_opposite']=any(f.get('selected_branch_would_change_now',False) for f in facts.get('active_frames',[]))
    checks['R_binding_copy_pending_dependency']=any(f['active_binding_probes'] for f in facts.get('active_frames',[]))
    return {'roles_complete':True,'checks':checks,'one_field_diffs':{'F_A':da,'F_I':di,'F_S':ds,'R_X':dx},'R_frame_facts':facts,'replay_witnesses':replay,'status_by_derived_role':{k:out[k]['status'] for k in roles},'role_scope':'derived from typed task relationships and own outputs, never private ID grammar or metadata'}

def cue_baseline(rows,outputs,fn):
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev_challenge'];counts=collections.defaultdict(collections.Counter)
    for row in train:counts[str(fn(row['semantic_task']))][outputs[row['sample_id']]['status']]+=1
    chosen={k:sorted(v,key=lambda s:(-v[s],s))[0] for k,v in counts.items()}
    majority=collections.Counter(outputs[r['sample_id']]['status'] for r in train);fallback=sorted(majority,key=lambda s:(-majority[s],s))[0]
    return {'train_counts':{k:dict(v) for k,v in counts.items()},'train_accuracy':sum(chosen[str(fn(r['semantic_task']))]==outputs[r['sample_id']]['status'] for r in train)/len(train),'dev_accuracy':sum(chosen.get(str(fn(r['semantic_task'])),fallback)==outputs[r['sample_id']]['status'] for r in dev)/len(dev),'labels':'OWN predicted status only; not canonical correctness or student/model evaluation','unseen_dev_feature_values':sum(str(fn(r['semantic_task'])) not in chosen for r in dev)}

def run(a):
    start=utc();startclock=time.monotonic();fr=(HERE/'source-freeze-r1.json').read_bytes();need(sha(fr)==a.expected_freeze,'freeze hash');freeze=loads(fr);before=verify(freeze);cfg=loads((HERE/'config-r1.json').read_bytes())
    for name in cfg['outputs']:need(not (HERE/name).exists(),'would overwrite output '+name)
    dep=BASE/cfg['own_prior_preparation'];import_at=utc();s=load_helper('own_rsm_input_frame_r1',dep/'solver-r1.py');r=load_helper('own_rsm_input_recursive_r1',dep/'recursive-simulator-r1.py')
    hand=loads((dep/'hand-calibration-r1.json').read_bytes());need(typed(fold_events(hand['rich']['task'],hand['rich']['events'][:5]),hand['rich']['checkpoint_after_5']),'manual fold checkpoint5');need(typed(fold_events(hand['rich']['task'],hand['rich']['events'][:6]),hand['rich']['checkpoint_after_6']),'manual fold checkpoint6');need(typed(fold_events(hand['rich']['task'],hand['rich']['events']),hand['rich']['terminal']),'manual fold terminal')
    raw=(BASE/cfg['packet']).read_bytes();need(sha(raw)==cfg['packet_sha256'] and len(raw)==cfg['packet_bytes'],'packet mismatch');lines=raw.splitlines();need(len(lines)==cfg['expected_rows'],'row count');rows=[loads(line) for line in lines];need(len({row['sample_id'] for row in rows})==len(rows),'duplicate sample ID');need(len({row['record_hash'] for row in rows})==len(rows),'duplicate provided full record hash')
    contract=(BASE/cfg['public_contract']).read_text();predictions=[];inspections=[];outputs={};records={};families=collections.defaultdict(list);crosschecks=0
    for row,line in zip(rows,lines):
        sid=row['sample_id'];need(set(row)==set(cfg['packet_fields']),'input extra/missing fields '+sid);need(row['track_id']=='RSM-001' and row['split'] in ('train','dev_challenge'),'track/split');need(type(row['record_hash']) is str and re.fullmatch('[0-9a-f]{64}',row['record_hash']) is not None,'full record join hash')
        for field in ('sample_id','semantic_parent_id','semantic_family_id'):need(type(row[field]) is str and row[field],'coordinate type')
        text=row['rendered_input'];need(type(text) is str and text.count('\nTask JSON:\n')==1,'prompt delimiter '+sid);prefix,tail=text.split('\nTask JSON:\n');need(prefix.rstrip()==contract.rstrip(),'contract prefix '+sid);student=loads(tail);need(typed(student,row['semantic_task']),'student/task typed mismatch '+sid)
        out=s.solve(student);ind,rec=recursive_prediction(student,s,r);need(typed(out,ind),'independent full-output discrepancy '+sid);crosschecks+=1;outputs[sid]=out;records[sid]=row;families[row['semantic_family_id']].append(row)
        evidence={'sample_id':sid,'record_hash':row['record_hash'],'packet_line_sha256':sha(line),'task_sha256':sha(canonical(student)),'rendered_input_sha256':sha(text.encode()),'row_hash_scope':'provided original FULL PENDING candidate hash, not stripped row hash','history_prefix_matches_recursive_trace':type(student['history']) is list and typed(student['history'],rec['events'][:len(student['history'])]) if rec else None,'independent_full_output_matches':True,'original_trace_events':len(rec['events']) if rec else None,'original_trace_effects':sum(e['event']=='effect' for e in rec['events']) if rec else None,'history_events':len(student['history']) if type(student['history']) is list else None,'mode':student['mode'],'event_budget':student['event_budget'],'expected_status':out['status'],'expected_error':out['error'],'previous_effects':out['previous_effects'],'new_effects':out['new_effects'],'pending_work':out['status']=='paused','program_metrics':primitives(student),'input_clarity':'complete exact public contract plus unambiguous typed task','gold_correctness':'UNKNOWN','private_metadata_and_source_novelty':'UNREVIEWED'}
        predictions.append({'prediction_id':'research_architect-input-first:'+sid,'sample_id':sid,'semantic_parent_id':row['semantic_parent_id'],'semantic_family_id':row['semantic_family_id'],'split':row['split'],'record_hash':row['record_hash'],'task_sha256':evidence['task_sha256'],'rendered_input_sha256':evidence['rendered_input_sha256'],'expected_output':out,'actual_execution_id':'/root/research_architect','crosscheck':'own independent recursive traversal plus event fold, shared public parser','canonical_gold_read':False});inspections.append(evidence)
    family_ledger=[];parent_count=0;style_pairs=0;row_initials=[]
    for family,members in sorted(families.items()):
        need(len({row['split'] for row in members})==1,'family split leak');byparent=collections.defaultdict(list)
        for row in members:byparent[row['semantic_parent_id']].append(row)
        parent_count+=len(byparent);parents={};fieldpairs=[]
        for parent,pair in sorted(byparent.items()):
            need(len(pair)==2 and typed(pair[0]['semantic_task'],pair[1]['semantic_task']),'parent typed pair mismatch');need(typed(outputs[pair[0]['sample_id']],outputs[pair[1]['sample_id']]),'field pair output difference');need(pair[0]['rendered_input']!=pair[1]['rendered_input'],'field pair identical prompt');style_pairs+=1;parents[parent]=pair[0];fieldpairs.append([row['sample_id'] for row in pair])
        roles=locate_roles(parents,outputs);controls=family_controls(parents,roles,outputs,s);issues=[]
        if not controls['roles_complete']:issues.append('PUBLIC_INPUT_ROLES_NOT_FULLY_RESOLVED')
        for key,ok in controls.get('checks',{}).items():
            if not ok:issues.append('INPUT_CONTROL_PROBE_FALSE:'+key)
        role_by_parent={v:k for k,v in roles.items()};ledger={'semantic_family_id':family,'split':members[0]['split'],'rows':len(members),'typed_parents':len(parents),'derived_roles':roles,'field_pairs':fieldpairs,'controls':controls,'issues':issues,'actual_novelty_not_measured':True,'source_family_index_not_in_stripped_view':True,'programs':{role:{'task':parents[pid]['semantic_task'],'expected_output':outputs[parents[pid]['sample_id']],'sample_id':parents[pid]['sample_id']} for role,pid in roles.items()}}
        family_ledger.append(ledger)
        for row in members:
            sid=row['sample_id'];role=role_by_parent.get(row['semantic_parent_id'],'UNRESOLVED');row_initials.append({'receipt_id':'research_architect-input-initial:'+sid,'sample_id':sid,'record_hash':row['record_hash'],'reviewer_role':'independent_adjudicator_input_first','reviewer_execution_id':'/root/research_architect','stage':'INITIAL_INPUT_ONLY','decision':'PENDING_CANONICAL_COMPARISON','target_correctness':'UNKNOWN','data_approval':False,'publication_eligible':None,'scores':{'input_clarity':4,'typed_feasibility':4,'target_correctness':None,'competency':3 if issues else 4},'critical_defects':[],'issues':issues,'confidence_expected_output':0.99,'reason':'Own typed frame execution and separately recursive full-output/event-fold agree. Gold, canonical metadata, novelty and reviewer evidence remain unread. Derived input role '+role+'. Bounded explicit procedure; correlated control/style views, exposed development and same-model/design overlap limit scope.','evidence':{'task_sha256':sha(canonical(row['semantic_task'])),'rendered_input_sha256':sha(row['rendered_input'].encode()),'family':family,'input_role_derived_from_semantics':role,'family_ledger':'family-input-ledger-r1.json','prediction_ledger':'predictions-r1.jsonl'},'review_design_author_overlap':True,'same_model_correlation_risk':True})
    need(len(rows)==216 and len(row_initials)==216,'first completeness');newjsonl(HERE/'predictions-r1.jsonl',predictions);newjsonl(HERE/'initial-judgments-r1.jsonl',row_initials);newjsonl(HERE/'row-input-inspections-r1.jsonl',inspections);newjson(HERE/'family-input-ledger-r1.json',family_ledger)
    statuscounts=collections.Counter(o['status'] for o in outputs.values());audit={'rows':len(rows),'semantic_families_observed':len(families),'semantic_parents_observed':parent_count,'field_view_pairs':style_pairs,'source_family_independence_or_novelty':'NOT_ESTABLISHED_FROM_PACKET','split_rows':dict(collections.Counter(row['split'] for row in rows)),'split_families':dict(collections.Counter(f['split'] for f in family_ledger)),'prediction_status_counts':dict(statuscounts),'prediction_status_by_split':{split:dict(collections.Counter(outputs[row['sample_id']]['status'] for row in rows if row['split']==split)) for split in ('train','dev_challenge')},'full_output_recursive_fold_crosschecks':crosschecks,'family_input_probe_failures':{f['semantic_family_id']:f['issues'] for f in family_ledger if f['issues']},'input_role_statuses':{f['semantic_family_id']:f['controls'].get('status_by_derived_role') for f in family_ledger},'baseline_probes':{'mode':cue_baseline(rows,outputs,lambda t:t['mode']),'history_length':cue_baseline(rows,outputs,lambda t:len(t['history'])),'budget':cue_baseline(rows,outputs,lambda t:t['event_budget']),'mode_history_budget':cue_baseline(rows,outputs,lambda t:(t['mode'],len(t['history']),t['event_budget']))},'history_validation_warnings':sum(not v['history_prefix_matches_recursive_trace'] for v in inspections),'record_hash_recomputed':False,'record_hash_scope':'original full candidate join token, stripped record has not enough fields to verify it','future_full_candidate_and_manifest_required':True,'target_correctness':'UNKNOWN','data_approval':False,'publication_eligible':None,'limitations':['source novelty/control manifest/hash claims unavailable','tiny finite supplied programs and correlated views','all broad strata exposed on development','public-design author/common-model/root-guidance correlation','visible modes/history/budgets may predict status; exact full outputs still require transitions']};newjson(HERE/'input-audit-r1.json',audit)
    after=verify(freeze);runtime_paths={v['path'] for v in freeze['bindings'] if v['scope']=='runtime'};loaded=[]
    for name,m in sorted(sys.modules.items()):
        origin=getattr(m,'__file__',None)
        if origin and ('/lib/python' in origin) and pathlib.Path(origin).exists():
            p=str(pathlib.Path(origin).resolve());need(p in runtime_paths,'new unfrozen loaded stdlib '+name+' '+p);loaded.append({'module':name,'path':p,'sha256':sha(pathlib.Path(p).read_bytes())})
    receipt={'completed_execution':True,'actual_execution_id':'/root/research_architect','stage':'INPUT_FIRST_NO_CANONICAL_ACCESS','started_utc':start,'first_own_helper_import_utc':import_at,'completed_utc':utc(),'elapsed_seconds':time.monotonic()-startclock,'source_freeze_sha256':sha(fr),'bindings_before_import':before,'bindings_after_execution':after,'frozen_bindings_unchanged':before==after,'loaded_stdlib_files_after_execution':loaded,'hand_event_fold_checks':3,'rows_predicted':len(predictions),'initial_judgments':len(row_initials),'independent_full_output_agreements':crosschecks,'runtime':{'resolved_python':freeze['resolved_python'],'python_version':sys.version},'all_outputs':[],'gold_target_correctness':'UNKNOWN','gold_or_other_first_verdicts_read':False,'data_approval':False,'publication_eligible':None,'completed_agent_inspection_required_before_FIRST':True}
    for name in cfg['outputs']:
        if name!='execution-r1.json':
            data=(HERE/name).read_bytes();receipt['all_outputs'].append({'path':str((HERE/name).relative_to(BASE)),'sha256':sha(data),'bytes':len(data)})
    newjson(HERE/'execution-r1.json',receipt);print(json.dumps({'completed_execution':True,'rows':216,'full_output_agreements':crosschecks,'families':len(families),'parents':parent_count,'field_pairs':style_pairs,'status_counts':dict(statuscounts),'family_probe_failures':audit['family_input_probe_failures'],'data_approval':False}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-freeze',required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        failure={'actual_execution_id':'/root/research_architect','at_utc':utc(),'status':'FAILED_PRESERVED','completed_execution':False,'traceback':traceback.format_exc(),'data_approval':False};newjson(HERE/'failure-execution-r1.json',failure);print(failure['traceback']);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
