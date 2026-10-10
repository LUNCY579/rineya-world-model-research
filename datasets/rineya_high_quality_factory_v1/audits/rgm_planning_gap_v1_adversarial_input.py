#!/usr/bin/env python3
"""Independent input-only planning review, actual author /root/adversarial_auditor.
Imports only this reviewer's frozen public-only preparation, never factory code.
"""
import collections,copy,datetime,hashlib,heapq,importlib.util,itertools,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
PREP=ROOT/'audits/rgm_planning_gap_adversarial_preparation.py'
PREP_HASH='2330e383f2a8e1ec608435ee7eacfd647e072e8715bb35d583c12798eaaa1ad3'
INPUT=ROOT/'reviews/input_only/planning-gap-v1.jsonl'
INPUT_HASH='2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654'
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

def main():
    require(sha(INPUT)==INPUT_HASH,'input SHA mismatch')
    rows=[json.loads(s) for s in INPUT.open()];require(len(rows)==384,'384 views')
    require(all(set(r)=={'sample_id','original_record_hash','source_family_key','split','semantic_task','rendered_input'} for r in rows),'six input whitelist fields')
    source=pathlib.Path(__file__);source_hash=sha(source)
    freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_input-{source_hash}.py',source.read_bytes())
    predictions=[];judgments=[];cache={};independent_counts=collections.Counter();units=[];first_seen=set()
    for r in rows:
        t=r['semantic_task'];payload=r['rendered_input'].split('Task JSON:\n');require(len(payload)==2,'one payload delimiter')
        require(canon(json.loads(payload[1]))==canon(t),'prompt task equality '+r['sample_id'])
        require(all(s not in r['rendered_input'] for s in (r['sample_id'],r['source_family_key'],r['original_record_hash'])) and ('"'+r['split']+'"') not in r['rendered_input'],'internal metadata excluded')
        key=canon(t)
        if key not in cache:
            answer,info=P.solve(t);cache[key]=(answer,info);units.append(r)
            if t['kind']=='graph':
                alt,top,cheapest,count=independent_simple_graph(t)
                require(canon(alt)==canon(answer['shortest_plan']) and top==info['topology_minimum_moves'] and cheapest==info['cheapest_required_fuel'] and count==info['minimum_route_multiplicity']['route_count'],'alternate graph exact check')
                independent_counts['graph_simple_path_comparisons']+=1
            else:
                alt=P.alternate_blocks_minimum(t);require(canon(alt)==canon(answer['shortest_plan']),'alternate relation-set fresh plan')
                s=frozenset(t['on'].items())
                for m,step in zip(t['plan'],answer['steps']):
                    legal,s=P.alternate_relation_move(s,m);require(legal==step['legal'] and canon(dict(s))==canon(step['after']),'alternate replay')
                    independent_counts['block_relation_transition_comparisons']+=1
                independent_counts['block_relation_fresh_plan_comparisons']+=1
        answer,info=cache[key];properties=P.factual_properties(t,answer,info)
        predictions.append({'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],
                            'reviewer_execution_id':EXEC,'stage':'INPUT_ONLY_FROZEN_EXPECTATION','expected_target':answer,'independent_row_properties':properties,'search_evidence':info})
        judgments.append({'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'reviewer_execution_id':EXEC,'reviewer_role':'adversarial_input_only',
                          'decision':'INPUT_WELL_FORMED_AND_EXACTLY_SOLVED','dataset_decision':None,'gold_not_read':True,
                          'instruction_sufficient':True,'exact_prompt_payload_match':True,'critical_input_defects':[],
                          'scope_limits':['Same model family and public-guided preparation; prior belief/planning exposure disclosed.',
                                          'Source family and ID are join keys excluded from student prompt, not shortcut features.',
                                          'No canonical targets, metadata claims or control manifest read; no dataset acceptance.']})
    predpath=ROOT/'reviews/adversarial/planning-gap-v1.input_predictions.jsonl';judgepath=ROOT/'reviews/adversarial/planning-gap-v1.input_first_judgments.jsonl'
    freeze(predpath,lines(predictions));freeze(judgepath,lines(judgments));preds={r['sample_id']:r for r in predictions}
    print(canon({'stage':'first_predictions_frozen','rows':384,'predictions_sha256':sha(predpath),'initial_input_judgments_sha256':sha(judgepath),'source_sha256':source_hash}),flush=True)
    # Only own input-derived expectations now feed these input-only attacks.
    metamorphic=collections.Counter();mutants={};examples={};family_groups=collections.defaultdict(list);struct=collections.defaultdict(lambda:collections.defaultdict(set));cross=collections.defaultdict(set)
    for r in units:
        t=r['semantic_task'];answer=preds[r['sample_id']]['expected_target'];family_groups[r['source_family_key']].append(r)
        if t['kind']=='graph':
            signatures={'weighted_start_goal':graph_signature(t),'unweighted_start_goal':graph_signature(t,False),'unweighted_topology':graph_signature(t,False,False)}
            modes=['undirected','ignore_fuel','continue_after_failure','fresh_from_final','cheapest_fuel_objective']
        else:
            signatures={'joint_initial_goal':blocks_signature(t),'initial_shape':blocks_signature(t,False)}
            modes=['noop_allowed','ignore_destination_clear','continue_after_failure','fresh_from_final','infer_unspecified_TABLE']
        for name,sig in signatures.items():struct[name][sig].add(r['source_family_key']);cross[(name,sig)].add(r['split'])
        for mode in modes:
            name=t['kind']+':'+mode;mutant=graph_mutant(t,mode) if t['kind']=='graph' else blocks_mutant(t,mode)
            count=mutants.setdefault(name,{'executed':0,'differs_from_correct':0});count['executed']+=1
            if canon(mutant)!=canon(answer):
                count['differs_from_correct']+=1
                if name not in examples:examples[name]={'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'actual_task':t,'independent_expected_target':answer,'mutant_target':mutant,'counterexample_scope':'actual input, no gold; first sample encountered, not claimed globally minimal'}
        # Key order must not affect answer; lexical-order-preserving rename checks replay and fresh canonical sequence.
        rev=copy.deepcopy(t)
        if t['kind']=='graph':rev['nodes']=list(reversed(rev['nodes']));rev['edges']=list(reversed(rev['edges']))
        else:rev['on']=dict(reversed(list(rev['on'].items())));rev['goal_on']=dict(reversed(list(rev['goal_on'].items())))
        require(canon(P.solve(rev)[0])==canon(answer),'ordering invariance');metamorphic['node_edge_or_map_order']+=1
        names=sorted(t.get('nodes',t.get('on',{})));rename={n:f'x{i:02}' for i,n in enumerate(names)};rename['TABLE']='TABLE'
        renamed=copy.deepcopy(t)
        if t['kind']=='graph':
            renamed['nodes']=[rename[n] for n in t['nodes']];renamed['edges']=[[rename[a],rename[b],c] for a,b,c in t['edges']]
            renamed.update(start=rename[t['start']],goal=rename[t['goal']],plan=[rename[n] for n in t['plan']])
        else:
            renamed['on']={rename[b]:rename[d] for b,d in t['on'].items()};renamed['goal_on']={rename[b]:rename[d] for b,d in t['goal_on'].items()};renamed['plan']=[{'block':rename[m['block']],'to':rename[m['to']]} for m in t['plan']]
        a2,_=P.solve(renamed)
        def remap(v):
            if type(v) is str:return rename.get(v,v)
            if type(v) is list:return [remap(x) for x in v]
            if type(v) is dict:return {rename.get(k,k):remap(x) for k,x in v.items()}
            return v
        require(canon(a2)==canon(remap(answer)),'order-preserving name equivariance');metamorphic['lexical_preserving_name_equivariance']+=1
        if not answer['valid']:
            suffix=copy.deepcopy(t);suffix['plan'] += [t['start']] if t['kind']=='graph' else [{'block':names[0],'to':'TABLE'}]
            require(canon(P.solve(suffix)[0])==canon(answer),'illegal absorption');metamorphic['unevaluated_suffix_absorption']+=1
    lookup=lookup_diagnostics(units,preds)
    family_splits=collections.defaultdict(set);prompt_sets=collections.defaultdict(set);task_sets=collections.defaultdict(set)
    for r in rows:family_splits[r['source_family_key']].add(r['split']);prompt_sets[r['rendered_input']].add(r['split']);task_sets[canon(r['semantic_task'])].add(r['split'])
    artifact=ROOT/'audits/shortcut_detection/planning-gap-v1-input-only.json'
    evidence={'actual_author_execution':EXEC,'stage':'INPUT_ONLY_AFTER_EXPECTATIONS_FROZEN','input_sha256':INPUT_HASH,'prediction_sha256':sha(predpath),'initial_judgments_sha256':sha(judgepath),
              'independent_crosschecks':dict(independent_counts),'metamorphic_checks':dict(metamorphic),'mutants':mutants,'actual_mutant_counterexamples':examples,
              'train_to_dev_lookup_tests_semantic_units':lookup,'lookup_scope':'No model training; majority fallback on unseen keys; semantic unit dedup prevents field-renderer double weighting. Join IDs and oracle-derived replay/search properties excluded.',
              'semantic_units':len(units),'views':len(rows),'families':len(family_splits),'split_views':dict(collections.Counter(r['split'] for r in rows)),
              'split_semantic_units':dict(collections.Counter(r['split'] for r in units)),
              'label_counts_views':{k:dict(collections.Counter(str(p['expected_target'][k]) for p in predictions)) for k in ('valid','reaches_goal')},
              'fresh_minimum_moves_distribution':dict(collections.Counter(str(p['search_evidence']['fresh_minimum_moves']) for p in predictions)),
              'inferred_row_property_counts_views':dict(collections.Counter(k for p in predictions for k in p['independent_row_properties'])),
              'per_family_semantic_units':dict(collections.Counter(len(v) for v in family_groups.values())),
              'exact_prompt_duplicate_groups':sum(len([r for r in rows if r['rendered_input']==p])>1 for p in prompt_sets),
              'exact_semantic_task_duplicate_groups':len(rows)-len(task_sets),'cross_split_exact_prompt_groups':sum(len(s)>1 for s in prompt_sets.values()),
              'cross_split_exact_task_groups':sum(len(s)>1 for s in task_sets.values()),'cross_split_family_groups':sum(len(s)>1 for s in family_splits.values()),
              'current_structural_signature_counts':{k:len(v) for k,v in struct.items()},
              'same_signature_multifamily_groups':{k:sum(len(fs)>1 for fs in v.values()) for k,v in struct.items()},
              'cross_split_structural_signature_groups':dict(collections.Counter(k for (k,sig),splits in cross.items() if len(splits)>1)),
              'scope_limits':['No comparison with earlier release inputs: public-only read boundary forbids them in this stage.',
                              'World/context structural overlap is measured honestly and may be intentional; field-order pairs are not independent source families.',
                              'Feature associations are input-only corpus diagnostics, not efficacy evidence; correct semantic features naturally determine answers.',
                              'Same-model/common-public-design and prior reviewer knowledge remain correlated review risks.']}
    freeze(artifact,jsbytes(evidence))
    report={'execution_id':EXEC,'completed':True,'stage':'INPUT_PREDICTION_AND_INITIAL_INPUT_JUDGMENT_FREEZE','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'read_allowlist':['reviews/input_only/planning-gap-v1.jsonl','reviews/input_only/planning-gap-v1.execution.json']+P.PUBLIC+['own frozen preparation source'],
            'canonical_candidate_gold_and_manifest_not_read':True,'producer_formal_expert_and_prior_verdicts_not_read':True,'dataset_approval':False,
            'input':{'path':str(INPUT.relative_to(ROOT)),'sha256':INPUT_HASH,'bytes':INPUT.stat().st_size,'rows':384},
            'public_sources':[{'path':p,'sha256':sha(ROOT/p)} for p in P.PUBLIC],
            'preparation_sha256':PREP_HASH,'stage_source_sha256':source_hash,
            'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in (predpath,judgepath,artifact)],
            'independent_crosschecks':dict(independent_counts),'metamorphic_counts':dict(metamorphic),'semantic_units':len(units),'families':len(family_splits),'mutant_test_count':sum(v['executed'] for v in mutants.values()),
            'residual_scope':'Input validity/solutions only. Future targets/claims/control/provenance comparisons require separate authorization; no dataset verdict.'}
    out=ROOT/'audits/execution/planning-gap-v1-adversarial-input-freeze.json';freeze(out,jsbytes(report));print(canon({'completed':True,'execution_sha256':sha(out),'attack_sha256':sha(artifact),'source_sha256':source_hash,'independent_checks':dict(independent_counts),'metamorphics':dict(metamorphic)}),flush=True)
if __name__=='__main__':main()
