#!/usr/bin/env python3
"""Independent input-only planning review, actual author /root/adversarial_auditor.
Imports only this reviewer's frozen public-only preparation, never factory code.
"""
import collections,copy,datetime,hashlib,heapq,importlib.util,itertools,json,pathlib,re,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
PREP=ROOT/'audits/rgm_planning_gap_v2_adversarial_preparation_r4.py'
PREP_HASH='b4f1bf2bf60cce998f92d0a1f8cb629ec25413074dc644d7559dc1222b2fb0ea'
INPUT=ROOT/'reviews/input_only/planning-gap-v2.jsonl'
INPUT_HASH='00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c'
EXEC='/root/adversarial_auditor'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,msg):
    if not v:raise ValueError(msg)
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():require(p.read_bytes()==data,'immutable artifact differs '+str(p))
    else:p.write_bytes(data)
def jsbytes(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def lines(xs):return ''.join(canon(x)+'\n' for x in xs).encode()
require(sha(PREP)==PREP_HASH,'frozen preparation hash')
spec=importlib.util.spec_from_file_location('own_public_preparation',PREP);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def independent_simple_graph(t):
    # Positive fuel/move costs imply any minimum-move solution has no repeated node.
    adj=collections.defaultdict(list)
    for a,b,c in t['edges']:adj[a].append((b,c))
    paths=[]
    def visit(node,seen,path,cost):
        if node==t['goal']:paths.append((path,cost));return
        for dest,c in adj[node]:
            if dest not in seen:visit(dest,seen|{dest},path+[dest],cost+c)
    visit(t['start'],{t['start']},[],0)
    feasible=[p for p,c in paths if c<=t['fuel']]
    minimum=min(feasible,key=lambda p:(len(p),p)) if feasible else None
    top=min((len(p) for p,c in paths),default=None)
    cheapest=min((c for p,c in paths),default=None)
    count=sum(len(p)==len(minimum) for p in feasible) if minimum is not None else 0
    return minimum,top,cheapest,count

def graph_signature(t,weighted=True,mark=True):
    names=t['nodes'];fixed=([t['start']] if t['start']==t['goal'] else [t['start'],t['goal']]) if mark else []
    remaining=[n for n in names if n not in fixed];best=None
    for tail in itertools.permutations(remaining):
        label={n:i for i,n in enumerate(fixed+list(tail))}
        edges=tuple(sorted((label[a],label[b],c if weighted else 1) for a,b,c in t['edges']))
        key=(len(names),t['start']==t['goal'] if mark else None,edges)
        if best is None or key<best:best=key
    return hashlib.sha256(canon(best).encode()).hexdigest()

def blocks_signature(t,goal=True):
    names=list(t['on']);best=None
    for order in itertools.permutations(names):
        label={n:i for i,n in enumerate(order)};label['TABLE']=-1
        on=tuple(label[t['on'][b]] for b in order)
        goals=tuple(sorted((label[b],label[d]) for b,d in t['goal_on'].items())) if goal else ()
        key=(len(names),on,goals)
        if best is None or key<best:best=key
    return hashlib.sha256(canon(best).encode()).hexdigest()

def get_features(r):
    t=r['semantic_task'];names=t.get('nodes',list(t.get('on',{})));ns=sorted(names)
    out={'kind':t['kind'],'size':len(names),'plan_length':len(t['plan']),'prompt_length':len(r['rendered_input']),
         'name_total_chars':sum(map(len,names)),'name_lengths':tuple(sorted(map(len,names))),
         'name_initials':tuple(sorted(n[0:1] for n in names)),'first_action':canon(t['plan'][0]) if t['plan'] else 'EMPTY',
         'last_action':canon(t['plan'][-1]) if t['plan'] else 'EMPTY','payload_length':len(r['rendered_input'].split('Task JSON:\n')[1])}
    if t['kind']=='graph':
        counts=collections.Counter(a for a,b,c in t['edges']);inc=collections.Counter(b for a,b,c in t['edges'])
        out.update(fuel=t['fuel'],edge_count=len(t['edges']),cost_sum=sum(c for a,b,c in t['edges']),
                   cost_multiset=tuple(sorted(c for a,b,c in t['edges'])),outdegrees=tuple(sorted(counts[n] for n in names)),
                   indegrees=tuple(sorted(inc[n] for n in names)),start_rank=ns.index(t['start']),goal_rank=ns.index(t['goal']),
                   first_destination_rank=ns.index(t['plan'][0]) if t['plan'] and t['plan'][0] in ns else -1)
    else:
        depths=[]
        for b in names:
            d=0
            while b!='TABLE':d+=1;b=t['on'][b]
            depths.append(d)
        out.update(goal_count=len(t['goal_on']),initial_true_goal_count=sum(t['on'][b]==d for b,d in t['goal_on'].items()),
                   stack_count=sum(d=='TABLE' for d in t['on'].values()),depth_multiset=tuple(sorted(depths)),
                   goal_TABLE_count=sum(d=='TABLE' for d in t['goal_on'].values()),
                   first_block_rank=ns.index(t['plan'][0]['block']) if t['plan'] and t['plan'][0]['block'] in ns else -1,
                   plan_TABLE_count=sum(m['to']=='TABLE' for m in t['plan']))
    return out

def lookup_diagnostics(units,preds):
    results=[]
    keys=[(k,) for k in sorted(set().union(*(get_features(r) for r in units)))]
    keys += [('kind','size','plan_length'),('kind','size','plan_length','fuel'),('kind','edge_count','cost_sum','fuel'),
             ('kind','goal_count','initial_true_goal_count','plan_length'),('kind','depth_multiset','goal_count','plan_length'),
             ('kind','name_lengths','plan_length'),('kind','start_rank','goal_rank','first_destination_rank'),
             ('kind','size','plan_length','payload_length'),('kind','stack_count','goal_count','plan_TABLE_count')]
    train=[r for r in units if r['split']=='train'];dev=[r for r in units if r['split']!='train']
    for label in ('valid','reaches_goal','fresh_feasible'):
        def y(r):return preds[r['sample_id']]['expected_target'][label] if label!='fresh_feasible' else preds[r['sample_id']]['expected_target']['shortest_plan'] is not None
        maj=collections.Counter(y(r) for r in train).most_common(1)[0][0]
        results.append({'feature_keys':['TRAIN_MAJORITY'],'label':label,'dev_correct':sum(y(r)==maj for r in dev),'dev_rows':len(dev),'train_rows':len(train)})
        for key in keys:
            def x(r):return canon([get_features(r).get(k,'NOT_APPLICABLE') for k in key])
            table=collections.defaultdict(collections.Counter)
            for r in train:table[x(r)][y(r)]+=1
            guesses={v:counts.most_common(1)[0][0] for v,counts in table.items()}
            correct=sum(y(r)==guesses.get(x(r),maj) for r in dev);covered=sum(x(r) in guesses for r in dev)
            results.append({'feature_keys':list(key),'label':label,'dev_correct':correct,'dev_rows':len(dev),'covered_dev_rows':covered,
                            'train_rows':len(train),'train_keys':len(table),'mixed_train_keys':sum(len(c)>1 for c in table.values()),
                            'examples_wrong':[r['sample_id'] for r in dev if y(r)!=guesses.get(x(r),maj)][:3]})
    return sorted(results,key=lambda r:(r['label'],-r['dev_correct'],r['feature_keys']))

def graph_mutant(t,mode):
    edges={(a,b):c for a,b,c in t['edges']};state={'node':t['start'],'fuel':t['fuel']};steps=[];okay=True
    for dest in t['plan']:
        cost=edges.get((state['node'],dest))
        if cost is None and mode=='undirected':cost=edges.get((dest,state['node']))
        legal=cost is not None and (mode=='ignore_fuel' or cost<=state['fuel'])
        if legal:state={'node':dest,'fuel':state['fuel']-cost}
        steps.append({'legal':legal,'after':copy.deepcopy(state)});okay &= legal
        if not legal and mode!='continue_after_failure':break
    answer,_=P.graph_solve(t);answer.update(steps=steps,after=state,valid=okay,reaches_goal=okay and state['node']==t['goal'])
    if mode=='fresh_from_final':
        tmp=copy.deepcopy(t);tmp.update(start=state['node'],fuel=state['fuel']);answer['shortest_plan']=P.graph_minimum(tmp)[0]
    elif mode=='cheapest_fuel_objective':
        heap=[(0,(),t['start'])];seen=set();answer['shortest_plan']=None
        while heap:
            cost,path,node=heapq.heappop(heap)
            if node in seen:continue
            seen.add(node)
            if node==t['goal']:answer['shortest_plan']=list(path);break
            for (a,b),c in edges.items():
                if a==node and cost+c<=t['fuel']:heapq.heappush(heap,(cost+c,path+(b,),b))
    return answer

def blocks_mutant(t,mode):
    state=copy.deepcopy(t['on']);steps=[];okay=True
    for m in t['plan']:
        legal,after,_=P.block_move(state,m)
        if mode=='noop_allowed' and m['block'] in state and state[m['block']]==m['to']:legal=True;after=copy.deepcopy(state)
        if mode=='ignore_destination_clear' and m['block'] in state and m['block'] not in state.values() and m['to'] in state and m['to']!=m['block'] and state[m['block']]!=m['to']:
            legal=True;after=copy.deepcopy(state);after[m['block']]=m['to']
        if legal:state=after
        steps.append({'legal':legal,'after':copy.deepcopy(state)});okay &= legal
        if not legal and mode!='continue_after_failure':break
    answer,_=P.blocks_solve(t);answer.update(steps=steps,after=state,valid=okay,reaches_goal=okay and all(state[b]==d for b,d in t['goal_on'].items()))
    if mode=='infer_unspecified_TABLE':answer['reaches_goal']=okay and all(state[b]==t['goal_on'].get(b,'TABLE') for b in state)
    if mode=='fresh_from_final':
        tmp=copy.deepcopy(t);tmp['on']=state;answer['shortest_plan']=P.blocks_minimum(tmp)[0]
    return answer


WHITELIST={'sample_id','record_hash','source_family_key','split','semantic_task','rendered_input'}
def normalized_task(t):
    P.graph_validate(t) if t['kind']=='graph' else P.blocks_validate(t)
    out=copy.deepcopy(t)
    if t['kind']=='graph':out['nodes']=sorted(t['nodes']);out['edges']=sorted(t['edges'])
    # Node/edge array order and mapping serialization are presentation; action sequence isn't.
    return out

def canonical_clearance_witnesses(t,answer):
    if t['kind']!='blocks':return []
    out=[]
    for i,a in enumerate(t['plan'][:len(answer['steps'])]):
        if i<1 or not answer['steps'][i]['legal']:continue
        initial=P.guard_truth(t['on'],a)
        code='source_not_clear' if not initial['source_clear'] else 'destination_not_clear' if a['to']!='TABLE' and not initial['destination_clear_or_table'] else None
        if code is None:continue
        selected=a['block'] if code=='source_not_clear' else a['to'];obstruction=next((b for b,d in t['on'].items() if d==selected),None)
        for j in range(i):
            e={'action_index':i,'precondition':code,'enabling_action_index':j,'obstructing_block':obstruction}
            try:P.validate_clearance_v2(t,e)
            except ValueError:continue
            out.append(e);break
    return out

def six_guard_trace(t,answer):
    if t['kind']!='blocks':return None
    strict,guards,first=P.diagnostic_replay(t)
    require(canon(strict)==canon({k:v for k,v in answer.items() if k!='shortest_plan'}),'own six-guard strict replay agreement')
    return {'guards':guards,'first_failure_index':first,'first_failure_guards':None if first is None else guards[first],
            'sole_failure_guard':None if first is None or sum(not v for v in guards[first].values())!=1 else next(k for k,v in guards[first].items() if not v)}

def extended_features(r):
    f=get_features(r);t=r['semantic_task'];payload=r['rendered_input'].split('Task JSON:\n')[1];lead=r['rendered_input'].split('Task JSON:\n')[0]
    f.update(first_payload_key=next(iter(json.loads(payload))),first_node=t['nodes'][0] if t['kind']=='graph' else 'NA',
             first_edge=canon(t['edges'][:1]) if t['kind']=='graph' else 'NA',
             lead_length=len(lead),prompt_whitespace_tokens=len(r['rendered_input'].split()),
             lowercase_count=sum(c.islower() for c in payload),uppercase_count=sum(c.isupper() for c in payload),
             digit_count=sum(c.isdigit() for c in payload),TABLE_word_count=payload.count('TABLE'),
             last_destination_rank=sorted(t['nodes']).index(t['plan'][-1]) if t['kind']=='graph' and t['plan'] and t['plan'][-1] in t['nodes'] else -1,
             first_destination_is_goal=bool(t['kind']=='graph' and t['plan'] and t['plan'][0]==t['goal']),
             last_destination_is_goal=bool(t['kind']=='graph' and t['plan'] and t['plan'][-1]==t['goal']))
    if t['kind']=='blocks':
        f['last_to_TABLE']=bool(t['plan'] and t['plan'][-1]['to']=='TABLE')
        f['last_noop_against_initial']=bool(t['plan'] and t['on'].get(t['plan'][-1]['block'])==t['plan'][-1]['to'])
        f['goal_support_name_lengths']=tuple(sorted(map(len,t['goal_on'].values())))
    return f

def stronger_lookups(units,preds):
    features={r['sample_id']:extended_features(r) for r in units};train=[r for r in units if r['split']=='train'];dev=[r for r in units if r['split']=='dev_challenge']
    keys=[(k,) for k in sorted(set().union(*(f for f in features.values())))]+[
        ('kind','size','plan_length'),('kind','fuel','edge_count','plan_length'),('kind','size','prompt_length'),
        ('kind','cost_sum','edge_count','fuel'),('kind','stack_count','depth_multiset','goal_count','plan_length'),
        ('kind','initial_true_goal_count','goal_TABLE_count','plan_TABLE_count'),('kind','first_block_rank','plan_length'),
        ('kind','first_action','last_action'),('kind','name_lengths','payload_length','plan_length'),
        ('kind','first_payload_key','prompt_whitespace_tokens'),('kind','last_to_TABLE','last_noop_against_initial','plan_length'),
        ('kind','first_destination_rank','last_destination_rank','start_rank','goal_rank')]
    def label(r,name):
        p=preds[r['sample_id']];a=p['expected_target'];s=p['search_evidence']
        if name in ('valid','reaches_goal'):return a[name]
        if name=='fresh_feasible':return a['shortest_plan'] is not None
        if name=='minimum_moves':return s['fresh_minimum_moves']
        if name=='first_error_index':return None if s['first_failure'] is None else s['first_failure']['index']
        if name=='first_error_reason':
            if s['first_failure'] is None:return None
            if r['semantic_task']['kind']=='graph':return s['first_failure']['reasons'][0]
            return p['input_derived_six_guard_trace']['sole_failure_guard'] or 'MULTIPLE'
    results=[]
    for name in ('valid','reaches_goal','fresh_feasible','minimum_moves','first_error_index','first_error_reason'):
        counts=collections.Counter(canon(label(r,name)) for r in train);majority=counts.most_common(1)[0][0]
        results.append({'label':name,'feature_keys':['TRAIN_MAJORITY'],'train_rows':len(train),'dev_rows':len(dev),'dev_correct':sum(canon(label(r,name))==majority for r in dev),'train_label_histogram':dict(counts)})
        for key in keys:
            x=lambda r:canon([features[r['sample_id']].get(k,'NOT_APPLICABLE') for k in key])
            tables=collections.defaultdict(collections.Counter)
            for r in train:tables[x(r)][canon(label(r,name))]+=1
            guesses={k:c.most_common(1)[0][0] for k,c in tables.items()}
            right=[r for r in dev if canon(label(r,name))==guesses.get(x(r),majority)]
            results.append({'label':name,'feature_keys':list(key),'train_rows':len(train),'dev_rows':len(dev),'dev_correct':len(right),'covered_dev_rows':sum(x(r) in guesses for r in dev),'mixed_train_keys':sum(len(c)>1 for c in tables.values()),'train_keys':len(tables),'actual_wrong_ids':[r['sample_id'] for r in dev if r not in right][:3]})
    return sorted(results,key=lambda x:(x['label'],-x['dev_correct'],x['feature_keys']))

def derived_family_controls(units,preds):
    families=collections.defaultdict(list)
    for r in units:families[r['source_family_key']].append(r)
    controls=[];errors=[];base_rows=[]
    for family,rs in families.items():
        successful=[r for r in rs if preds[r['sample_id']]['expected_target']['reaches_goal']]
        require(bool(successful),'some independently successful base in '+family)
        base=min(successful,key=lambda r:(len(r['semantic_task']['plan']),r['sample_id']));base_rows.append(base)
        for l,r in itertools.combinations(rs,2):
            lt,rt=l['semantic_task'],r['semantic_task'];changed=[k for k in lt if canon(normalized_task(lt)[k])!=canon(normalized_task(rt)[k])]
            name=None;evidence=None
            if changed==['fuel']:
                lo,hi=sorted((l,r),key=lambda z:z['semantic_task']['fuel']);low,high=lo['semantic_task'],hi['semantic_task'];lp,hp=preds[lo['sample_id']],preds[hi['sample_id']]
                name='fuel_only_counterfactual';left,right=hi,lo
                evidence={'low_fuel':low['fuel'],'high_fuel':high['fuel'],'minimum_required_fuel':lp['search_evidence']['cheapest_required_fuel'],'low_fresh_minimum':lp['search_evidence']['fresh_minimum_moves'],'high_fresh_minimum':hp['search_evidence']['fresh_minimum_moves'],'low_topology_minimum':lp['search_evidence']['topology_minimum_moves'],'subtype':'INFEASIBLE' if lp['expected_target']['shortest_plan'] is None else 'LENGTHENED'}
                require(hp['expected_target']['reaches_goal'] and not lp['expected_target']['valid'],'decisive actual fuel-only submitted legality')
            elif changed==['edges']:
                larger,smaller=sorted((l,r),key=lambda z:len(z['semantic_task']['edges']),reverse=True);big={tuple(e) for e in larger['semantic_task']['edges']};small={tuple(e) for e in smaller['semantic_task']['edges']}
                if len(big-small)==1 and small<=big:
                    name='directed_edge_deletion_counterfactual';left,right=larger,smaller;evidence={'removed_edge':list(next(iter(big-small))),'smaller_first_failure':preds[smaller['sample_id']]['search_evidence']['first_failure'],'both_fresh_feasible':all(preds[z['sample_id']]['expected_target']['shortest_plan'] is not None for z in (larger,smaller))}
                    require(preds[larger['sample_id']]['expected_target']['reaches_goal'] and not preds[smaller['sample_id']]['expected_target']['valid'] and evidence['both_fresh_feasible'],'decisive edge deletion and remaining fresh feasibility')
            elif changed==['goal_on']:
                weak,strong=sorted((l,r),key=lambda z:len(z['semantic_task']['goal_on']));wg,sg=weak['semantic_task']['goal_on'],strong['semantic_task']['goal_on']
                if len(wg)<len(sg) and all(sg.get(k)==v for k,v in wg.items()):
                    name='partial_goal_extension_counterfactual';left,right=weak,strong;evidence={'weak_goal_count':len(wg),'strong_goal_count':len(sg),'weak_reaches_goal':preds[weak['sample_id']]['expected_target']['reaches_goal'],'strong_reaches_goal':preds[strong['sample_id']]['expected_target']['reaches_goal'],'strong_fresh_minimum':preds[strong['sample_id']]['search_evidence']['fresh_minimum_moves']}
                    require(evidence['weak_reaches_goal'] and not evidence['strong_reaches_goal'] and preds[strong['sample_id']]['expected_target']['valid'],'goal extension affects goal satisfaction with same legal plan')
            elif changed==['plan']:
                a,b=preds[l['sample_id']],preds[r['sample_id']]
                for left,right in [(l,r),(r,l)]:
                    aa,bb=preds[left['sample_id']],preds[right['sample_id']];la,rb=left['semantic_task'],right['semantic_task']
                    if aa['expected_target']['reaches_goal'] and bb['expected_target']['reaches_goal'] and len(la['plan'])==aa['search_evidence']['fresh_minimum_moves'] and len(rb['plan'])>bb['search_evidence']['fresh_minimum_moves']:
                        name='valid_goal_detour_counterfactual';evidence={'minimum_moves':aa['search_evidence']['fresh_minimum_moves'],'submitted_detour_moves':len(rb['plan'])};break
                    if lt['kind']=='blocks' and aa['expected_target']['reaches_goal'] and len(la['plan'])==aa['search_evidence']['fresh_minimum_moves'] and not bb['expected_target']['valid'] and len(la['plan'])==len(rb['plan']):
                        for i in range(len(la['plan'])-1):
                            swap=copy.deepcopy(la['plan']);swap[i],swap[i+1]=swap[i+1],swap[i]
                            if canon(swap)!=canon(rb['plan']):continue
                            g=bb['input_derived_six_guard_trace'];reason='source_not_clear' if g['sole_failure_guard']=='source_clear' else 'destination_not_clear' if g['sole_failure_guard']=='destination_clear_or_table' else None
                            if reason is not None:
                                ev={'swap_index':i,'first_failure_index':g['first_failure_index'],'first_failure_reason':reason,'guard_truth':g['first_failure_guards']};w=P.validate_selective_pair(la,rb,ev)
                                name='adjacent_action_order_counterfactual';evidence={**ev,'mutant_name':w['mutant_name'],'actual_mutated_action_legal':w['mutant']['steps'][g['first_failure_index']]['legal']};break
                        if name:break
                    if lt['kind']=='blocks' and aa['expected_target']['valid'] and not aa['expected_target']['reaches_goal'] and not bb['expected_target']['valid'] and len(rb['plan'])==len(la['plan'])+1 and canon(rb['plan'][:-1])==canon(la['plan']):
                        action=rb['plan'][-1]
                        if action['to']=='TABLE':
                            ev={'prefix_length':len(la['plan']),'block':action['block'],'current_support':'TABLE'}
                            try:w=P.validate_noop_pair(base['semantic_task'],la,rb,ev)
                            except ValueError:continue
                            name='isolated_noop_append_counterfactual';evidence={**ev,'guard_truth':w['guard_truth'],'no_op_mutant_valid':w['mutant']['valid'],'clearance_mutants_agree':True};break
            if name:controls.append({'name':name,'source_family_key':family,'split':l['split'],'sample_ids':[left['sample_id'],right['sample_id']],'record_hashes':[left['record_hash'],right['record_hash']],'actual_changed_task_fields':changed,'input_derived_evidence':evidence,'scope':'Input-derived semantic pair only; supplied manifest declarations/control IDs have not been read.'})
        if len(rs) not in (4,5):errors.append({'family':family,'semantic_tasks':len(rs)})
    return controls,base_rows,errors

def main():
    started=time.perf_counter();require(sha(INPUT)==INPUT_HASH,'exact frozen input SHA')
    source=pathlib.Path(__file__);source_hash=sha(source);freeze(ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_input-{source_hash}.py',source.read_bytes())
    rootfreeze=ROOT/P.ROOT_FREEZE;require(sha(rootfreeze)==P.ROOT_FREEZE_SHA,'seven-document root freeze')
    public=json.loads(rootfreeze.read_text())['public_specification_bindings']
    for rel,h in public.items():require(sha(ROOT/rel)==h,'public hash '+rel)
    viewfreeze=ROOT/'reviews/input_only/planning-gap-v2.freeze.json'
    pre={'actual_execution_id':EXEC,'stage':'V2_INPUT_ONLY_SOURCE_FREEZE_BEFORE_TARGET_EXECUTION','source_sha256':source_hash,'preparation_sha256':PREP_HASH,'input_sha256':INPUT_HASH,'input_view_freeze_sha256':sha(viewfreeze),'root_public_freeze_sha256':P.ROOT_FREEZE_SHA,'public_sources_sha256':public,'read_allowlist':[str(INPUT.relative_to(ROOT)),str(viewfreeze.relative_to(ROOT)),str(PREP.relative_to(ROOT)),P.ROOT_FREEZE]+list(public),'disclosure':'Public-guided prospective correction and common underlying model/design; reuse own V1 CPU solvers and own frozen V2 hand fixtures. No canonical gold/manifest/private producer/formal/other reviewer expectations read. No dataset approval.'}
    prepath=ROOT/'audits/execution/planning-gap-v2-adversarial-input.preexec.json';freeze(prepath,jsbytes(pre))
    print(canon({'stage':'source_frozen_before_target_execution','source_sha256':source_hash,'preexec_sha256':sha(prepath)}),flush=True)
    rows=[json.loads(s) for s in INPUT.open()];require(len(rows)==408 and all(set(r)==WHITELIST for r in rows),'408 exact six-field views')
    require(len({r['sample_id'] for r in rows})==408 and len({r['record_hash'] for r in rows})==408,'distinct custody IDs/hashes')
    predictions=[];judgments=[];cache={};units=[];independent=collections.Counter();normal_groups=collections.defaultdict(list)
    private=['sample_id','semantic_parent_id','semantic_family_id','track_id','skill_tags','difficulty','generator_version','oracle_version','generation_seed','source_provenance','target_format','split','review_status','review_receipt_ids','record_hash','verified_target','source_family_key','source_index','source_family_index','controlled_context','primary_stratum','property_witnesses','family_control_claims','producer_role','producer_agent_id']
    label=re.compile(r'(?<![A-Za-z0-9_])(?:'+'|'.join(private)+r')(?![A-Za-z0-9_])["\x27`*_]*\s*[:=]',re.I)
    for r in rows:
        t=r['semantic_task'];parts=r['rendered_input'].split('Task JSON:\n');require(len(parts)==2,'single payload marker '+r['sample_id'])
        lead,payload=parts;require(canon(json.loads(payload))==canon(t),'exact typed prompt-payload equality '+r['sample_id'])
        require(not label.search(lead),'explicit private metadata labels excluded from contract lead')
        require(all(r[k] not in r['rendered_input'] for k in ('sample_id','source_family_key','record_hash')),'actual custody values excluded from prompt')
        normalized=normalized_task(t);key=canon(normalized);normal_groups[key].append(r)
        require((t['kind']=='graph' and 5<=len(t['nodes'])<=7 and t['fuel']<=30) or (t['kind']=='blocks' and 5<=len(t['on'])<=6),'public object/fuel bounds')
        require(len(t['plan'])<=12,'public submitted plan bound')
        if key not in cache:
            answer,info=P.solve(t);cache[key]=(answer,info);units.append(r)
            if t['kind']=='graph':
                alt,top,cheapest,count=independent_simple_graph(t)
                require(canon(alt)==canon(answer['shortest_plan']) and top==info['topology_minimum_moves'] and cheapest==info['cheapest_required_fuel'] and count==info['minimum_route_multiplicity']['route_count'],'alternate positive-cost simple-path enumerator')
                independent['graph_simple_path_fresh_plan_topology_fuel_multiplicity_comparisons']+=1
            else:
                alternate=P.alternate_blocks_minimum(t);require(canon(alternate)==canon(answer['shortest_plan']),'own alternate relation-set optimal search')
                state=frozenset(t['on'].items())
                for action,step in zip(t['plan'],answer['steps']):
                    legal,state=P.alternate_relation_move(state,action);require(legal==step['legal'] and canon(dict(state))==canon(step['after']),'independent relation transition')
                    independent['block_relation_transition_comparisons']+=1
                independent['block_relation_fresh_plan_comparisons']+=1
        answer,info=cache[key];properties=P.factual_properties(t,answer,info);guard=six_guard_trace(t,answer);clearance=canonical_clearance_witnesses(t,answer)
        if guard and guard['first_failure_index'] is not None and guard['first_failure_index']>=1 and guard['sole_failure_guard']=='changed_support' and t['plan'][guard['first_failure_index']]['to']=='TABLE':
            properties['isolated_unchanged_support_failure_after_legal_prefix']={'action_index':guard['first_failure_index'],'block':t['plan'][guard['first_failure_index']]['block'],'current_support':'TABLE','full_guard_truth':guard['first_failure_guards']}
        predictions.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],'reviewer_execution_id':EXEC,'stage':'INPUT_ONLY_FROZEN_EXPECTATION','expected_target':answer,'independent_row_properties':properties,'search_evidence':info,'input_derived_v2_clearance_witnesses':clearance,'input_derived_six_guard_trace':guard})
        judgments.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'reviewer_execution_id':EXEC,'reviewer_role':'adversarial_input_only','decision':'INPUT_WELL_FORMED_AND_EXACTLY_SOLVED','dataset_decision':None,'gold_not_read':True,'instruction_sufficient':True,'exact_typed_prompt_payload_match':True,'critical_input_defects':[],'scope_limits':['Public-guided V2 corrective design, same underlying model and own V1/V2 solver/fixture reuse are correlated.','Custody IDs/family/split are excluded from student prompt and never fed into the target solver.','No actual metadata witness/control manifest/canonical targets read; no acceptance of dataset or supplied claims.']})
    predpath=ROOT/'reviews/adversarial/planning-gap-v2.input_predictions.jsonl';judgepath=ROOT/'reviews/adversarial/planning-gap-v2.input_first_judgments.jsonl'
    freeze(predpath,lines(predictions));freeze(judgepath,lines(judgments));preds={p['sample_id']:p for p in predictions}
    print(canon({'stage':'ALL_FIRST_INPUT_PREDICTIONS_AND_JUDGMENTS_FROZEN','rows':len(rows),'predictions_sha256':sha(predpath),'first_judgments_sha256':sha(judgepath)}),flush=True)
    require(len(units)==204 and all(len(rs)==2 for rs in normal_groups.values()),'independent204 semantic units each with2 field presentations')
    controls,bases,layout_errors=derived_family_controls(units,preds)
    metamorphic=collections.Counter();mutants={};examples={};signatures=collections.defaultdict(lambda:collections.defaultdict(set));splitstruct=collections.defaultdict(set)
    for r in units:
        t=r['semantic_task'];answer=preds[r['sample_id']]['expected_target']
        modes=['undirected','ignore_fuel','continue_after_failure','fresh_from_final','cheapest_fuel_objective'] if t['kind']=='graph' else ['source_clear_ignored','destination_clear_ignored','no_op_allowed','continue_after_failure','fresh_from_final','infer_unspecified_TABLE']
        sigs={'weighted_start_goal':graph_signature(t),'unweighted_start_goal':graph_signature(t,False),'unweighted_topology':graph_signature(t,False,False)} if t['kind']=='graph' else {'joint_initial_goal':blocks_signature(t),'initial_shape':blocks_signature(t,False)}
        for name,sig in sigs.items():signatures[name][sig].add(r['source_family_key']);splitstruct[(name,sig)].add(r['split'])
        for mode in modes:
            name=t['kind']+':'+mode
            if t['kind']=='graph':mutant=graph_mutant(t,mode)
            elif mode in P.MUTANTS:mutant=copy.deepcopy(answer);mutant.update(P.diagnostic_replay(t,mode)[0])
            else:mutant=blocks_mutant(t,mode)
            tally=mutants.setdefault(name,{'executed':0,'differs_from_correct':0,'affected_ids':[],'affected_splits':collections.Counter(),'affected_families':set()});tally['executed']+=1
            if canon(mutant)!=canon(answer):
                tally['differs_from_correct']+=1;tally['affected_ids'].append(r['sample_id']);tally['affected_splits'][r['split']]+=1;tally['affected_families'].add(r['source_family_key'])
                if name not in examples:examples[name]={'sample_id':r['sample_id'],'record_hash':r['record_hash'],'actual_input_task':t,'input_derived_expected_target':answer,'actual_mutant_target':mutant,'first_differing_step':next((i for i,(a,b) in enumerate(zip(answer['steps'],mutant['steps'])) if canon(a)!=canon(b)),None),'minimality_limit':'First exact actual sample counterexample; global minimality not asserted. Minimal standalone source/destination/no-op toys are frozen in own public preparation.'}
        reversed_task=copy.deepcopy(t)
        if t['kind']=='graph':reversed_task['nodes'].reverse();reversed_task['edges'].reverse()
        else:reversed_task['on']=dict(reversed(list(t['on'].items())));reversed_task['goal_on']=dict(reversed(list(t['goal_on'].items())))
        require(canon(P.solve(reversed_task)[0])==canon(answer),'actual field/list order invariance');metamorphic['field_or_list_order_invariance']+=1
        names=sorted(t.get('nodes',t.get('on',{})));mp={n:('A' if n<'TABLE' else 'z')+str(i).zfill(2) for i,n in enumerate(names)} if t['kind']=='blocks' else {n:'x'+str(i).zfill(2) for i,n in enumerate(names)};mp['TABLE']='TABLE'
        remap=lambda x:mp.get(x,x)
        if t['kind']=='graph':renamed={'kind':'graph','nodes':[mp[n] for n in t['nodes']],'edges':[[mp[a],mp[b],c] for a,b,c in t['edges']],'start':mp[t['start']],'goal':mp[t['goal']],'fuel':t['fuel'],'plan':[remap(n) for n in t['plan']]}
        else:renamed=P.blocks({mp[b]:remap(d) for b,d in t['on'].items()},{mp[b]:remap(d) for b,d in t['goal_on'].items()},[P.move(remap(m['block']),remap(m['to'])) for m in t['plan']])
        def rename_obj(v):
            if type(v) is str:return remap(v)
            if type(v) is list:return [rename_obj(x) for x in v]
            if type(v) is dict:return {remap(k):rename_obj(x) for k,x in v.items()}
            return v
        require(canon(P.solve(renamed)[0])==canon(rename_obj(answer)),'lexical-order-preserving names including TABLE');metamorphic['lexical_preserving_name_equivariance']+=1
        if not answer['valid']:
            suffix=copy.deepcopy(t);suffix['plan'] += [t['start']] if t['kind']=='graph' else [P.move(names[0],'TABLE')]
            require(canon(P.solve(suffix)[0])==canon(answer),'first failure suffix absorption');metamorphic['first_failure_suffix_absorption']+=1
    for tally in mutants.values():tally['affected_families']=sorted(tally['affected_families']);tally['affected_splits']=dict(tally['affected_splits'])
    lookup=stronger_lookups(units,preds);features={r['sample_id']:extended_features(r) for r in units}
    label_counts=lambda rs:{k:dict(collections.Counter(str(preds[r['sample_id']]['expected_target'][k]) for r in rs)) for k in ('valid','reaches_goal')}
    family_splits=collections.defaultdict(set)
    for r in rows:family_splits[r['source_family_key']].add(r['split'])
    rawgroups=collections.defaultdict(list);promptgroups=collections.defaultdict(list)
    for r in rows:rawgroups[canon(r['semantic_task'])].append(r);promptgroups[r['rendered_input']].append(r)
    first_payload_hist=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for r in rows:
        key=next(iter(json.loads(r['rendered_input'].split('Task JSON:\n')[1])))
        for name in ('valid','reaches_goal'):first_payload_hist[key][name][str(preds[r['sample_id']]['expected_target'][name])]+=1
    base_signatures=collections.defaultdict(set)
    for r in bases:
        t=r['semantic_task']
        if t['kind']=='graph':base_signatures['weighted_start_goal'].add(graph_signature(t));base_signatures['unweighted_topology'].add(graph_signature(t,False,False))
        else:base_signatures['joint_initial_goal'].add(blocks_signature(t));base_signatures['initial_shape'].add(blocks_signature(t,False))
    pairs_path=ROOT/'audits/diversity/planning-gap-v2-input-derived-controls.json'
    controls_artifact={'actual_execution_id':EXEC,'input_sha256':INPUT_HASH,'predictions_sha256':sha(predpath),'semantic_pair_count':len(controls),'pair_counts':dict(collections.Counter(c['name'] for c in controls)),'actual_input_derived_semantic_pairs':controls,'family_layout_errors':layout_errors,'inferred_family_base_ids':[r['sample_id'] for r in bases],'base_inference_rule':'Shortest submitted successful plan per family. This uses independently solved inputs, never ID context names or manifest annotations.','supplied_manifest_not_read':True,'remaining_verification':'Actual192per-renderer control declarations, unique control_ids and metadata locators require later authorized manifest comparison; input-derived pairs do not establish them.'}
    freeze(pairs_path,jsbytes(controls_artifact))
    diversity={'actual_execution_id':EXEC,'input_sha256':INPUT_HASH,'prediction_sha256':sha(predpath),'rows':len(rows),'raw_typed_task_variants':len(rawgroups),'normalized_semantic_inputs':len(units),'normalized_field_order_groups':len(normal_groups),'all_field_order_group_sizes':dict(collections.Counter(len(v) for v in normal_groups.values())),'source_families':len(family_splits),'source_family_split_collisions':sum(len(s)>1 for s in family_splits.values()),'views_by_split':dict(collections.Counter(r['split'] for r in rows)),'semantic_inputs_by_split':dict(collections.Counter(r['split'] for r in units)),'family_semantic_input_counts':dict(collections.Counter(sum(r['source_family_key']==f for r in units) for f in family_splits)),'exact_prompt_duplicate_groups':sum(len(rs)>1 for rs in promptgroups.values()),'raw_exact_task_duplicate_groups':sum(len(rs)>1 for rs in rawgroups.values()),'cross_split_raw_exact_task_groups':sum(len({r['split'] for r in rs})>1 for rs in rawgroups.values()),'cross_split_normalized_task_groups':sum(len({r['split'] for r in rs})>1 for rs in normal_groups.values()),'cross_split_prompt_groups':sum(len({r['split'] for r in rs})>1 for rs in promptgroups.values()),'all_input_structural_classes':{k:len(v) for k,v in signatures.items()},'base_structural_classes_inferred_from_inputs':{k:len(v) for k,v in base_signatures.items()},'same_signature_multifamily_groups':{k:sum(len(fs)>1 for fs in v.values()) for k,v in signatures.items()},'cross_split_structural_groups':dict(collections.Counter(k for (k,sig),ss in splitstruct.items() if len(ss)>1)),'labels_views':label_counts(rows),'labels_semantic_units':label_counts(units),'by_split_labels':{s:label_counts([r for r in units if r['split']==s]) for s in ('train','dev_challenge')},'fresh_minimum_moves_by_split':{s:dict(collections.Counter(str(preds[r['sample_id']]['search_evidence']['fresh_minimum_moves']) for r in units if r['split']==s)) for s in ('train','dev_challenge')},'first_payload_key_label_histogram':{k:{name:dict(c) for name,c in v.items()} for k,v in first_payload_hist.items()},'independent_row_property_counts':dict(collections.Counter(k for r in units for k in preds[r['sample_id']]['independent_row_properties'])),'input_feature_rows':[{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'split':r['split'],'features':features[r['sample_id']]} for r in units],'limits':['48 revised source lineages with zero growth relative to pendingV1 is public prospective allocation, not204/408 independent worlds.','Structural classes are representation overlaps, not independent mechanisms; two action grammars and exposed development remain.','Active catalog/source collision and earlier-release novelty are not independently certified under this input-only read scope.','No tags/producer control claims/difficulty labels/verified targets have been read.']}
    divpath=ROOT/'audits/diversity/planning-gap-v2-input-only.json';freeze(divpath,jsbytes(diversity))
    shortcut={'actual_execution_id':EXEC,'input_sha256':INPUT_HASH,'prediction_sha256':sha(predpath),'first_judgment_sha256':sha(judgepath),'stage':'OWN_FIRST_TARGETS_FROZEN_BEFORE_BATCH_DIAGNOSTICS','independent_crosschecks':dict(independent),'metamorphic_counts':dict(metamorphic),'actual_mutation_tests':mutants,'actual_counterexamples':examples,'semantic_unit_train_to_exposed_dev_lookups':lookup,'baseline_scope':'CPU counts/lookup/majority only; no training. Custody IDs/familykeys/split excluded from predictive features. Dedup normalized units avoids double weighting renderer views.','crossings_scope':'Per-row input facts and per-family supplied-task pair inference; manifest192declarations still unobserved.','label_length_limit':'Correct target trace/shortest-plan lengths follow semantics; their length differences are not input leakage. Input/prompt length associations are measured separately.','common_model_design_exposure_disclosed':True}
    shortpath=ROOT/'audits/shortcut_detection/planning-gap-v2-input-only.json';freeze(shortpath,jsbytes(shortcut))
    require(sha(INPUT)==INPUT_HASH and sha(PREP)==PREP_HASH,'input and solver bytes unchanged after actual execution')
    for rel,h in public.items():require(sha(ROOT/rel)==h,'public source bytes unchanged after actual execution')
    execution={'actual_execution_id':EXEC,'completed':True,'stage':'ALL_INPUT_FIRST_PREDICTIONS_JUDGMENTS_AND_AUDITS_FROZEN_AWAITING_COMPARISON_AUTHORIZATION','source_sha256':source_hash,'preexec_sha256':sha(prepath),'input_sha256_before_and_after':INPUT_HASH,'input_view_freeze_sha256':sha(viewfreeze),'preparation_sha256_before_and_after':PREP_HASH,'public_sources_sha256_before_and_after':public,'actual_duration_seconds':time.perf_counter()-started,'rows':408,'normalized_semantic_inputs':len(units),'raw_typed_task_variants':len(rawgroups),'families':len(family_splits),'independent_crosschecks':dict(independent),'metamorphics':dict(metamorphic),'mutant_test_count':sum(m['executed'] for m in mutants.values()),'artifacts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (predpath,judgepath,pairs_path,divpath,shortpath)},'read_allowlist':pre['read_allowlist'],'canonical_gold_manifest_producer_formal_test_otherreviewer_not_read':True,'dataset_approval':False,'review_mode':pre['disclosure'],'limitations':diversity['limits']}
    expath=ROOT/'audits/execution/planning-gap-v2-adversarial-input-freeze.json';freeze(expath,jsbytes(execution))
    print(canon({'completed':True,'execution_sha256':sha(expath),'source_sha256':source_hash,'predictions_sha256':sha(predpath),'first_judgments_sha256':sha(judgepath),'diversity_sha256':sha(divpath),'shortcut_sha256':sha(shortpath),'controls_sha256':sha(pairs_path),'normalized_units':len(units),'raw_variants':len(rawgroups),'pairs':dict(collections.Counter(c['name'] for c in controls)),'independent_checks':dict(independent),'metamorphics':dict(metamorphic),'actual_mutation_summary':{k:{n:v[n] for n in ('executed','differs_from_correct','affected_splits')} for k,v in mutants.items()}}),flush=True)

if __name__=='__main__':main()
