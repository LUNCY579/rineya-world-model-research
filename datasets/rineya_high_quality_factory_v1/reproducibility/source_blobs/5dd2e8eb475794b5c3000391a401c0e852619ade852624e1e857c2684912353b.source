#!/usr/bin/env python3
"""Independent blind audit for finite extended contracts, without producer imports."""
import copy
import argparse
import itertools
import json
import pathlib
from collections import Counter, defaultdict
from adversarial_audit import ROOT, EXECUTION, SCORE_KEYS, canon, digest, integer, scalar, require, reorder, issue, groups_by, lookup_baseline


def unique_strings(xs, label, nonempty=False):
    require(type(xs) is list and all(type(x) is str for x in xs) and len(xs)==len(set(xs)) and (xs or not nonempty), label)


def skills(t):
    state=copy.deepcopy(t['state']); procedures=t['procedures']
    require(set(state)=={'x','y'} and all(integer(x) for x in state.values()),'integer register state')
    require(type(procedures) is dict and all(type(x) is str for x in procedures),'procedure map')
    dependencies=defaultdict(set)
    def validate(body,parent):
        require(type(body) is list,'instruction list')
        for ins in body:
            require(type(ins) is dict and 'op' in ins,'instruction object')
            op=ins['op']
            if op in ('add','scale'):
                require(set(ins)=={'op','register','value'} and ins['register'] in ('x','y') and integer(ins['value']),'primitive shape/type')
            elif op=='swap':require(set(ins)=={'op'},'swap exact shape')
            elif op=='call':
                require(set(ins)=={'op','name'} and ins['name'] in procedures,'existing procedure call')
                dependencies[parent].add(ins['name'])
            elif op=='if':
                require(set(ins)=={'op','test','then','else'},'if shape')
                test=ins['test']
                require(set(test)=={'register','cmp','value'} and test['register'] in ('x','y') and test['cmp'] in ('lt','eq','ge') and integer(test['value']),'if test')
                validate(ins['then'],parent);validate(ins['else'],parent)
            else:raise ValueError('unknown instruction')
    for name,body in procedures.items():validate(body,name)
    validate(t['main'],'__main__')
    def acyclic(name,stack):
        require(name not in stack,'cyclic calls including inactive branches')
        for child in dependencies[name]:acyclic(child,stack|{name})
    for name in procedures:acyclic(name,set())
    calls,conditions,primitives=[],[],0
    def execute(body):
        nonlocal primitives
        for ins in body:
            op=ins['op']
            if op=='add':state[ins['register']]+=ins['value'];primitives+=1
            elif op=='scale':state[ins['register']]*=ins['value'];primitives+=1
            elif op=='swap':state['x'],state['y']=state['y'],state['x'];primitives+=1
            elif op=='call':calls.append(ins['name']);execute(procedures[ins['name']])
            else:
                v=state[ins['test']['register']];q=ins['test']['value'];cmp=ins['test']['cmp']
                result=v<q if cmp=='lt' else v==q if cmp=='eq' else v>=q
                conditions.append(result);execute(ins['then'] if result else ins['else'])
    execute(t['main'])
    return {'after':state,'executed_primitives':primitives,'call_order':calls,'conditions':conditions}


def modules(t):
    ms=t['modules'];ids=[m['module_id'] for m in ms];unique_strings(ids,'module IDs')
    for m in ms:
        require(set(m)=={'module_id','capabilities','priority'} and integer(m['priority']),'module shape/priority')
        unique_strings(m['capabilities'],'capabilities')
    memory=t['memory'].copy()
    require(type(memory) is dict and all(type(k) is str and scalar(v) for k,v in memory.items()),'scalar memory')
    unique_strings(t['protected'],'protected keys');require(set(t['protected'])<=set(memory),'protected keys exist')
    unique_strings([r['request_id'] for r in t['requests']],'request IDs')
    routes=[]
    for request in t['requests']:
        require(set(request)=={'request_id','needs','writes'},'request shape');unique_strings(request['needs'],'needs')
        eligible=[m for m in ms if set(request['needs'])<=set(m['capabilities'])]
        chosen=min(eligible,key=lambda m:(m['priority'],m['module_id'])) if eligible else None
        blocked=[]
        for i,w in enumerate(request['writes']):
            require(set(w)=={'key','value'} and type(w['key']) is str and scalar(w['value']),'write scalar shape')
            if chosen is None:continue
            if w['key'] in t['protected']:blocked.append(i)
            else:memory[w['key']]=w['value']
        routes.append({'request_id':request['request_id'],'selected_module':None if chosen is None else chosen['module_id'],
                       'eligible_modules':sorted(m['module_id'] for m in eligible),'blocked_writes':blocked})
    return {'routes':routes,'memory':memory}


def episodic(t):
    for e in t['events']:
        require(set(e)=={'key','version','at','op','value'} and type(e['key']) is str and integer(e['version']) and e['version']>=0 and integer(e['at']) and e['at']>=0 and e['op'] in ('put','delete') and scalar(e['value']),'event shape/type')
        require(e['op']!='delete' or e['value'] is None,'delete value null')
    unique_strings([q['query_id'] for q in t['queries']],'query IDs')
    results=[]
    for q in t['queries']:
        require(set(q)=={'query_id','key','as_of'} and type(q['key']) is str and integer(q['as_of']) and q['as_of']>=0,'query shape/type')
        eligible=[(e['version'],e['at'],i,e) for i,e in enumerate(t['events']) if e['key']==q['key'] and e['at']<=q['as_of']]
        if not eligible:status,value,version,index='absent',None,None,None
        else:
            version,_,index,e=max(eligible,key=lambda x:x[:3])
            status='deleted' if e['op']=='delete' else 'present';value=e['value']
        results.append(dict(query_id=q['query_id'],key=q['key'],status=status,value=value,chosen_version=version,chosen_event=index))
    return {'results':results}


def belief(t):
    worlds=t['worlds'];unique_strings([w['world_id'] for w in worlds],'world IDs')
    keys=set(worlds[0]['facts']) if worlds else set()
    for w in worlds:
        require(set(w)=={'world_id','facts'} and type(w['facts']) is dict and set(w['facts'])==keys and all(type(k) is str and scalar(v) for k,v in w['facts'].items()),'world facts')
    remaining=worlds[:];sizes=[]
    for o in t['observations']:
        require(set(o)=={'fact','equals'} and o['fact'] in keys and scalar(o['equals']),'observation')
        remaining=[w for w in remaining if canon(w['facts'][o['fact']])==canon(o['equals'])];sizes.append(len(remaining))
    unique_strings([q['query_id'] for q in t['queries']],'query IDs')
    queries=[]
    for q in t['queries']:
        require(set(q)=={'query_id','fact'} and q['fact'] in keys,'query fact')
        counts=Counter(canon(w['facts'][q['fact']]) for w in remaining)
        entailed=bool(remaining) and len(counts)==1
        values=[{'value':json.loads(v),'count':counts[v]} for v in sorted(counts)]
        queries.append({'query_id':q['query_id'],'counts':values,'distinct_values':len(values),'entailed':entailed,
                        'guaranteed':values[0]['value'] if entailed else None,'can_reduce_uncertainty':len(values)>1})
    return {'surviving_worlds':sorted(w['world_id'] for w in remaining),'belief_sizes':sizes,'queries':queries}


def abstraction(t):
    states=t['states'];actions=t['actions'];unique_strings(states,'states',True);unique_strings(actions,'actions',True)
    require(set(t['observable'])==set(states) and all(scalar(v) for v in t['observable'].values()),'total scalar observables')
    require(set(t['transition'])==set(states),'total transitions')
    for s in states:require(set(t['transition'][s])==set(actions) and all(x in states for x in t['transition'][s].values()),'transition action/state domains')
    # Independent pair-distinguishability closure, rather than the producer partition algorithm.
    distinct={(a,b) for a in states for b in states if canon(t['observable'][a])!=canon(t['observable'][b])}
    while True:
        new={(a,b) for a in states for b in states if any((t['transition'][a][u],t['transition'][b][u]) in distinct for u in actions)}
        if new<=distinct:break
        distinct|=new
    left=set(states);classes=[]
    while left:
        first=min(left);group=sorted(s for s in left if (first,s) not in distinct)
        classes.append(group);left-=set(group)
    classes.sort();ids={s:i for i,g in enumerate(classes) for s in g}
    return {'classes':classes,'state_to_class':ids,
            'abstract_transitions':[{'class_id':i,'action':u,'next_class':ids[t['transition'][g[0]][u]]} for i,g in enumerate(classes) for u in sorted(actions)],
            'abstract_observable':[{'class_id':i,'value':t['observable'][g[0]]} for i,g in enumerate(classes)]}


def constraint_repair(t):
    alphabet=t['alphabet'];unique_strings(alphabet,'alphabet',True);n=t['slots'];require(integer(n) and n>0 and integer(t['min_distance']) and t['min_distance']>=0,'slots/min_distance')
    require(len(alphabet)**n<=100000,'explicit exhaustive repair bound exceeded')
    for c in t['constraints']:
        k=c['kind']
        if k in ('equal','different'):require(set(c)=={'kind','positions'} and len(c['positions'])==2 and all(integer(i) and 0<=i<n for i in c['positions']),'position constraint')
        elif k=='allowed':require(set(c)=={'kind','position','values'} and integer(c['position']) and 0<=c['position']<n and type(c['values']) is list and all(v in alphabet for v in c['values']),'allowed constraint')
        elif k=='count':require(set(c)=={'kind','value','min','max'} and c['value'] in alphabet and integer(c['min']) and integer(c['max']),'count constraint')
        else:raise ValueError('unknown constraint')
    unique_strings([c['candidate_id'] for c in t['candidates']],'candidate IDs')
    def valid(a):
        for c in t['constraints']:
            k=c['kind']
            if k=='equal' and a[c['positions'][0]]!=a[c['positions'][1]]:return False
            if k=='different' and a[c['positions'][0]]==a[c['positions'][1]]:return False
            if k=='allowed' and a[c['position']] not in c['values']:return False
            if k=='count' and not c['min']<=a.count(c['value'])<=c['max']:return False
        return True
    feasible=[list(a) for a in itertools.product(sorted(alphabet),repeat=n) if valid(a)]
    hamming=lambda a,b:sum(x!=y for x,y in zip(a,b))
    repairs=[]
    for c in t['candidates']:
        a=c['assignment'];require(set(c)=={'candidate_id','assignment'} and type(a) is list and len(a)==n and all(x in alphabet for x in a),'candidate assignment')
        chosen=min(feasible,key=lambda f:(hamming(a,f),f)) if feasible else None
        repairs.append({'candidate_id':c['candidate_id'],'original_valid':valid(a),'edit_distance':hamming(a,chosen) if chosen is not None else None,'repaired':chosen})
    selected=[]
    for r in sorted((r for r in repairs if r['repaired'] is not None),key=lambda r:(r['edit_distance'],r['candidate_id'])):
        if all(hamming(r['repaired'],s['repaired'])>=t['min_distance'] for s in selected):selected.append(r)
    return {'feasible_count':len(feasible),'repairs':repairs,'diverse_selection':[r['candidate_id'] for r in selected]}


def agent_history(t):
    objects=t['objects'];places=t['places'];unique_strings(objects,'objects',True);unique_strings(places,'places',True)
    actual=t['initial_locations'].copy();require(set(actual)==set(objects) and all(v in places for v in actual.values()),'initial locations')
    unique_strings([a['agent_id'] for a in t['agents']],'agent IDs');agents=copy.deepcopy(t['agents'])
    for a in agents:
        require(set(a)=={'agent_id','initial_beliefs','object_priority'} and set(a['initial_beliefs'])==set(objects) and all(v is None or v in places for v in a['initial_beliefs'].values()),'explicit initial beliefs')
        unique_strings(a['object_priority'],'priority');require(set(a['object_priority'])<=set(objects),'priority objects')
    ids={a['agent_id'] for a in agents}
    for e in t['events']:
        require(set(e)=={'object','to','observed_by'} and e['object'] in objects and e['to'] in places,'history event')
        unique_strings(e['observed_by'],'observers');require(set(e['observed_by'])<=ids,'existing observer IDs')
        actual[e['object']]=e['to']
        for a in agents:
            if a['agent_id'] in e['observed_by']:a['initial_beliefs'][e['object']]=e['to']
    output=[]
    for a in agents:
        beliefs=a['initial_beliefs'];action={'type':'abstain'}
        for obj in a['object_priority']:
            if beliefs[obj] is not None:action={'type':'search','object':obj,'place':beliefs[obj]};break
        output.append({'agent_id':a['agent_id'],'beliefs':beliefs,'false_belief_objects':sorted(o for o in objects if beliefs[o] is not None and beliefs[o]!=actual[o]),
                       'unknown_objects':sorted(o for o in objects if beliefs[o] is None),'action':action})
    return {'actual_locations':actual,'agents':output}


SOLVERS={'skills':skills,'modules':modules,'episodic':episodic,'belief':belief,'predictive_abstraction':abstraction,'constraint_repair':constraint_repair,'agent_history':agent_history}


def solve(t):return SOLVERS[t['kind']](t)


def metamorphic(t,a):
    require(solve(reorder(t))==a,'JSON key order changed output');tests=['object_key_order']
    altered=copy.deepcopy(t);k=t['kind']
    if k=='skills':
        altered['procedures']['unused_arbitrary']=[{'op':'add','register':'x','value':9999}]
        require(solve(altered)==a,'unused procedure affects output');tests.append('unused_procedure')
        mapping={name:'procedure_%03d'%i for i,name in enumerate(sorted(t['procedures']))}
        def names(ins):
            if ins['op']=='call':ins['name']=mapping[ins['name']]
            if ins['op']=='if':
                for branch in ('then','else'):
                    for item in ins[branch]:names(item)
        altered=copy.deepcopy(t)
        for body in list(altered['procedures'].values())+[altered['main']]:
            for ins in body:names(ins)
        altered['procedures']={mapping[n]:body for n,body in altered['procedures'].items()}
        expected=copy.deepcopy(a);expected['call_order']=[mapping[n] for n in expected['call_order']]
        require(solve(altered)==expected,'procedure naming shortcut');tests.append('procedure_renaming')
    elif k=='modules':
        altered['modules'].reverse();altered['protected'].reverse()
        for m in altered['modules']:m['capabilities'].reverse()
        require(solve(altered)==a,'module/protection order');tests.append('module_fact_order')
    elif k=='episodic':
        altered['events'].append({'key':'__irrelevant_unqueried_key__','version':9999,'at':0,'op':'put','value':True})
        require(solve(altered)==a,'irrelevant key event');tests.append('unqueried_event')
        altered=copy.deepcopy(t)
        for e in altered['events']:e['at']+=7
        for q in altered['queries']:q['as_of']+=7
        require(solve(altered)==a,'uniform time shift');tests.append('chronology_shift')
    elif k=='belief':
        altered['worlds'].reverse();require(solve(altered)==a,'world list order');tests.append('world_order')
        if altered['observations']:
            altered['observations']+=copy.deepcopy(altered['observations'])
            expected=copy.deepcopy(a);expected['belief_sizes']+=[len(a['surviving_worlds'])]*len(t['observations'])
            require(solve(altered)==expected,'repeated observations');tests.append('observation_idempotence')
    elif k=='predictive_abstraction':
        altered['states'].reverse();altered['actions'].reverse();require(solve(altered)==a,'state/action order');tests.append('state_action_order')
        action=sorted(t['actions'])[0];new='ZZ_redundant_action'
        require(new not in t['actions'],'fresh action collision');altered=copy.deepcopy(t);altered['actions'].append(new)
        for s in altered['states']:altered['transition'][s][new]=altered['transition'][s][action]
        expected=copy.deepcopy(a)
        nextclass={(r['class_id'],r['action']):r['next_class'] for r in a['abstract_transitions']}
        expected['abstract_transitions']=[{'class_id':i,'action':u,'next_class':nextclass[i,action if u==new else u]} for i in range(len(a['classes'])) for u in sorted(altered['actions'])]
        require(solve(altered)==expected,'redundant action changes equivalence');tests.append('redundant_action_equivalence')
    elif k=='constraint_repair':
        altered['alphabet'].reverse();altered['constraints'].reverse();require(solve(altered)==a,'constraint/alphabet order');tests.append('constraint_alphabet_order')
        altered['candidates'].reverse();expected=copy.deepcopy(a);expected['repairs'].reverse()
        require(solve(altered)==expected,'candidate order affects greedy ID ordering');tests.append('candidate_order_greedy_invariance')
    else:
        altered['objects'].reverse();altered['places'].reverse()
        for e in altered['events']:e['observed_by'].reverse()
        require(solve(altered)==a,'history object/observer order');tests.append('history_fact_order')
        altered['agents'].reverse();expected=copy.deepcopy(a);expected['agents'].reverse()
        require(solve(altered)==expected,'agent input order');tests.append('agent_output_order_equivariance')
    # Output identifiers are allowed references, not quality indicators. Preserve lexical
    # ordering only where the public tie rule explicitly makes identifier order meaningful.
    altered=copy.deepcopy(t);expected=copy.deepcopy(a)
    if k=='modules':
        mapping={m['module_id']:'opaque_module_%03d'%i for i,m in enumerate(sorted(t['modules'],key=lambda m:m['module_id']))}
        for m in altered['modules']:m['module_id']=mapping[m['module_id']]
        for r in expected['routes']:
            r['selected_module']=None if r['selected_module'] is None else mapping[r['selected_module']]
            r['eligible_modules']=[mapping[m] for m in r['eligible_modules']]
        require(solve(altered)==expected,'module names convey competence');tests.append('module_id_renaming')
    elif k in ('episodic','belief'):
        mapping={q['query_id']:'opaque_query_%03d'%i for i,q in enumerate(t['queries'])}
        for q in altered['queries']:q['query_id']=mapping[q['query_id']]
        for q in expected['results' if k=='episodic' else 'queries']:q['query_id']=mapping[q['query_id']]
        require(solve(altered)==expected,'query ID conveys answer');tests.append('query_id_renaming')
    elif k=='constraint_repair':
        mapping={c['candidate_id']:'opaque_candidate_%03d'%i for i,c in enumerate(sorted(t['candidates'],key=lambda c:c['candidate_id']))}
        for c in altered['candidates']:c['candidate_id']=mapping[c['candidate_id']]
        for r in expected['repairs']:r['candidate_id']=mapping[r['candidate_id']]
        expected['diverse_selection']=[mapping[c] for c in expected['diverse_selection']]
        require(solve(altered)==expected,'order-preserving candidate rename');tests.append('candidate_id_renaming')
    elif k=='agent_history':
        mapping={agent['agent_id']:'opaque_agent_%03d'%i for i,agent in enumerate(t['agents'])}
        for agent in altered['agents']:agent['agent_id']=mapping[agent['agent_id']]
        for event in altered['events']:event['observed_by']=[mapping[agent] for agent in event['observed_by']]
        for agent in expected['agents']:agent['agent_id']=mapping[agent['agent_id']]
        require(solve(altered)==expected,'agent names imply psychological traits');tests.append('agent_id_renaming')
    return tests


def stripped(x):
    # Conservative numeric/identifier normalization. Entity equality patterns remain explicit.
    if type(x) is dict:
        return {k:('ID' if k.endswith('_id') else stripped(v)) for k,v in x.items()}
    if type(x) is list:return [stripped(v) for v in x]
    if integer(x):return 'INTEGER'
    return x


def duplicates(rs,key):
    return [{'fingerprint':digest(k),'samples':[r['sample_id'] for r in group],'families':sorted({r['semantic_family_id'] for r in group}),
             'splits':sorted({r['split'] for r in group})} for k,group in groups_by(rs,key).items() if len(group)>1]


def audit_metrics(rows,answers):
    units=list({(r['semantic_family_id'],canon(r['semantic_task'])):r for r in rows}.values());tracks={};findings=[]
    for track in sorted({r['track_id'] for r in units}):
        rs=[r for r in units if r['track_id']==track];examples=[];coverage=Counter()
        for r in rs:
            t=r['semantic_task'];a=answers[r['sample_id']];common={'split':r['split'],'family':r['semantic_family_id'],'task_length':len(canon(t)),'sample':r['sample_id']}
            k=t['kind']
            if k=='skills':
                examples.append(dict(common,label=any(a['conditions']),feature=len(t['main']),identity=a['after']==t['state']))
                coverage.update({'primitive_count_'+str(a['executed_primitives']):1,'condition_count_'+str(len(a['conditions'])):1})
            elif k=='modules':
                for i,e in enumerate(a['routes']):examples.append(dict(common,label=e['selected_module'] is None,feature=i,request_id=e['request_id']))
                coverage.update({'abstaining_routes':sum(e['selected_module'] is None for e in a['routes']),'blocked_writes':sum(len(e['blocked_writes']) for e in a['routes'])})
            elif k=='episodic':
                for i,e in enumerate(a['results']):examples.append(dict(common,label=e['status'],feature=i,query_id=e['query_id']))
                coverage.update(e['status'] for e in a['results']);coverage['present_null']+=sum(e['status']=='present' and e['value'] is None for e in a['results'])
            elif k=='belief':
                examples.append(dict(common,label=bool(a['surviving_worlds']),feature=len(t['observations'])))
                coverage.update({'surviving_'+str(len(a['surviving_worlds'])):1,'uncertainty_reducing_queries':sum(q['can_reduce_uncertainty'] for q in a['queries'])})
            elif k=='predictive_abstraction':
                initial=len({canon(v) for v in t['observable'].values()})
                examples.append(dict(common,label=len(a['classes']),feature=initial,initial_classes=initial,states=len(t['states'])))
                coverage.update({'class_count_'+str(len(a['classes'])):1,'needs_transition_refinement':int(initial<len(a['classes']))})
            elif k=='constraint_repair':
                examples.append(dict(common,label=a['feasible_count']>0,feature=len(t['constraints'])))
                coverage.update({'feasible_count_'+str(a['feasible_count']):1,'diverse_selection_size_'+str(len(a['diverse_selection'])):1,'null_repairs':sum(q['repaired'] is None for q in a['repairs'])})
            else:
                for i,e in enumerate(a['agents']):examples.append(dict(common,label=e['action']['type']=='abstain',feature=i,agent_id=e['agent_id']))
                coverage.update({'abstaining_agents':sum(e['action']['type']=='abstain' for e in a['agents']),'false_belief_objects':sum(len(e['false_belief_objects']) for e in a['agents']),'unknown_objects':sum(len(e['unknown_objects']) for e in a['agents'])})
        tracks[track]={'families':len({r['semantic_family_id'] for r in rs}),'logical_variants':len(rs),'coverage_counts':dict(coverage),'label_counts':dict(Counter(e['label'] for e in examples)),
                       'length_bucket_baseline':lookup_baseline(examples,lambda e:e['task_length']//50,lambda e:e['label']),
                       'stripped_feature_baseline':lookup_baseline(examples,lambda e:e['feature'],lambda e:e['label'])}
        if track=='RMM-001':
            byid=groups_by(examples,lambda e:e['request_id'])
            stable={identifier:sorted({e['label'] for e in es}) for identifier,es in byid.items()}
            tracks[track]['request_id_to_abstention_values']=stable
            tracks[track]['within_family_request_id_support']={family:{identifier:sorted({e['label'] for e in group}) for identifier,group in groups_by(es,lambda e:e['request_id']).items()} for family,es in groups_by(examples,lambda e:e['family']).items()}
            tracks[track]['within_family_position_support']={family:{str(position):sorted({e['label'] for e in group}) for position,group in groups_by(es,lambda e:e['feature']).items()} for family,es in groups_by(examples,lambda e:e['family']).items()}
            tracks[track]['every_family_ID_and_position_crosses_abstention']=all(set(labels)=={False,True} for field in ('within_family_request_id_support','within_family_position_support') for data in tracks[track][field].values() for labels in data.values())
            tracks[track]['joint_ID_position_baseline']=lookup_baseline(examples,lambda e:(e['request_id'],e['feature']),lambda e:e['label'])
            if all(len(v)==1 for v in stable.values()):findings.append({'code':'fixed_request_id_abstention','track':track,'severity':'high','evidence':stable})
        elif track=='REM-001':
            tracks[track]['query_id_status_baseline']=lookup_baseline(examples,lambda e:e['query_id'],lambda e:e['label'])
            tracks[track]['query_id_status_support']={identifier:dict(Counter(e['label'] for e in es)) for identifier,es in groups_by(examples,lambda e:e['query_id']).items()}
            tracks[track]['within_family_query_id_status_support']={family:{identifier:dict(Counter(e['label'] for e in group)) for identifier,group in groups_by(es,lambda e:e['query_id']).items()} for family,es in groups_by(examples,lambda e:e['family']).items()}
            tracks[track]['within_family_query_position_status_support']={family:{str(position):dict(Counter(e['label'] for e in group)) for position,group in groups_by(es,lambda e:e['feature']).items()} for family,es in groups_by(examples,lambda e:e['family']).items()}
            tracks[track]['every_family_ID_and_position_has_identical_status_distribution']=all(len({canon(counts) for counts in data.values()})==1 for field in ('within_family_query_id_status_support','within_family_query_position_status_support') for data in tracks[track][field].values())
            tracks[track]['joint_ID_position_baseline']=lookup_baseline(examples,lambda e:(e['query_id'],e['feature']),lambda e:e['label'])
            accuracy=tracks[track]['query_id_status_baseline']['dev_accuracy']
            if accuracy is not None and accuracy>=0.9:findings.append({'code':'fixed_query_id_retrieval_status','track':track,'severity':'high','evidence':tracks[track]['query_id_status_support']})
        elif track=='RJP-001':
            tracks[track]['observable_only_partition_exact_fraction']=sum(e['initial_classes']==e['label'] for e in examples)/len(examples)
            tracks[track]['observation_count_to_final_class_counts']={str(initial):dict(Counter(e['label'] for e in es)) for initial,es in groups_by(examples,lambda e:e['initial_classes']).items()}
        # Only two development families per track: these tables demonstrate attacks, not efficacy.
        tracks[track]['limitations']='Twelve finite families, two exposed development families. Row labels within one world are dependent. No trained baseline or sealed evaluation.'
    exact=duplicates(rows,lambda r:canon(r['semantic_task']))
    near=duplicates(rows,lambda r:canon(stripped(r['semantic_task'])))
    def shape(x):
        if type(x) is dict:return {k:shape(v) for k,v in x.items()}
        if type(x) is list:return sorted({canon(shape(v)) for v in x})
        return type(x).__name__
    schema_templates=duplicates(rows,lambda r:canon(shape(r['semantic_task'])))
    div={'records':len(rows),'families':len({r['semantic_family_id'] for r in rows}),'logical_variants':len(units),'exact_task_duplicates':exact,'exact_duplicates_across_distinct_families':[g for g in exact if len(g['families'])>1],
         'numeric_and_id_stripped_structure_groups':near,'cross_split_structure_groups':[g for g in near if len(g['splits'])>1],
         'schema_container_template_groups':schema_templates,
         'family_split_violations':[family for family,rs in groups_by(rows,lambda r:r['semantic_family_id']).items() if len({r['split'] for r in rs})>1],
         'provenance_counts':dict(Counter(canon(r['source_provenance']) for r in rows)),
         'source_family_key_presence_counts':dict(Counter(canon([r['track_id'],'source_family_key' in r]) for r in rows)),
         'scope_limitation_counts':dict(Counter(r.get('scope_limitation','MISSING') for r in rows)),
         'limitations':['Structure normalization strips integer magnitudes and *_id fields, but retains entity names and list order; it is conservative and may miss isomorphic templates.',
                        'Schema/container templates strip scalar values and list multiplicity; overlap indicates reusable public contract shape, not semantic or outcome identity.',
                        'Same-family forward/reverse duplicates are intended renderings. A new protected copied scalar or shuffled order alone is not a new semantic mechanism.',
                        'Original synthetic/license provenance is asserted by records. No external origin/license or sealed custody is established.']}
    return div,{'scope':'Independent CPU count/lookups and exhaustive finite solvers; no training or producer/reference imports.','tracks':tracks,'findings':findings},findings


def attack_probes(rows,answers,batch):
    probes=[]
    m={'kind':'modules','modules':[{'module_id':'b','priority':0,'capabilities':['known']},{'module_id':'a','priority':0,'capabilities':['known']}],
       'memory':{'protected':None},'protected':['protected'],'requests':[{'request_id':'opaque','needs':[],'writes':[{'key':'protected','value':1},{'key':'new','value':True}]},
       {'request_id':'another','needs':['unknown'],'writes':[{'key':'protected','value':1}]}]}
    ma=modules(m);require(ma['routes'][0]['selected_module']=='a' and ma['routes'][1]['selected_module'] is None and ma['routes'][1]['blocked_writes']==[] and ma['memory']['protected'] is None,'module tie/abstention/protection')
    probes.append({'attack':'module_empty_needs_tie_and_abstention_no_writes','task':m,'correct':ma})
    b={'kind':'belief','worlds':[{'world_id':'a','facts':{'f':True}},{'world_id':'b','facts':{'f':1}}],'observations':[],'queries':[{'query_id':'q','fact':'f'}]}
    ba=belief(b);require(ba['queries'][0]['distinct_values']==2,'boolean/int merged')
    b['observations']=[{'fact':'f','equals':True},{'fact':'f','equals':1}];empty=belief(b);require(empty['surviving_worlds']==[] and not empty['queries'][0]['entailed'],'empty belief vacuous entailment')
    probes.append({'attack':'type_sensitive_beliefs_and_empty_entailment','task':b,'correct':empty,'unfiltered_correct':ba})
    e={'kind':'episodic','events':[{'key':'k','version':3,'at':5,'op':'delete','value':None},{'key':'k','version':2,'at':9,'op':'put','value':'older'},
       {'key':'k','version':3,'at':5,'op':'put','value':None}], 'queries':[{'query_id':'q','key':'k','as_of':9}]}
    ea=episodic(e);require(ea['results'][0]['status']=='present' and ea['results'][0]['value'] is None and ea['results'][0]['chosen_event']==2,'version/tie/present-null semantics')
    e['events'][0],e['events'][2]=e['events'][2],e['events'][0];deleted=episodic(e);require(deleted['results'][0]['status']=='deleted','input tie ignored')
    probes.append({'attack':'same_version_time_input_index_and_authoritative_tombstone','task':e,'correct':deleted,'opposite_tie_order_correct':ea})
    p={'kind':'predictive_abstraction','states':['a','b','c','d'],'actions':['step'],'observable':{'a':0,'b':0,'c':1,'d':0},
       'transition':{'a':{'step':'c'},'b':{'step':'b'},'c':{'step':'c'},'d':{'step':'c'}}}
    pa=abstraction(p);require(pa['classes']==[['a','d'],['b'],['c']],'observable-only partition accepted')
    probes.append({'attack':'transition_distinguishes_equal_observables','task':p,'correct':pa})
    c={'kind':'constraint_repair','alphabet':['a','b'],'slots':3,'constraints':[],'min_distance':2,
       'candidates':[{'candidate_id':'A','assignment':['a','a','a']},{'candidate_id':'B','assignment':['a','a','b']},{'candidate_id':'C','assignment':['a','b','a']}]}
    ca=constraint_repair(c);require(ca['diverse_selection']==['A'],'greedy incorrectly replaced by maximum packing')
    probes.append({'attack':'greedy_selection_is_not_maximum_packing','task':c,'correct':ca,'larger_feasible_packing':['B','C']})
    h={'kind':'agent_history','objects':['book'],'places':['drawer','shelf'],'initial_locations':{'book':'shelf'},
       'agents':[{'agent_id':'A','initial_beliefs':{'book':'drawer'},'object_priority':['book']},{'agent_id':'B','initial_beliefs':{'book':'shelf'},'object_priority':[]}],'events':[]}
    ha=agent_history(h);require(ha['agents'][0]['action']['place']=='drawer' and ha['agents'][0]['false_belief_objects']==['book'] and ha['agents'][1]['action']=={'type':'abstain'},'explicit false belief/policy overwritten')
    probes.append({'attack':'explicit_false_initial_belief_and_empty_policy','task':h,'correct':ha})
    for track,field,outfield in [('RMM-001','requests','routes'),('REM-001','queries','results')]:
        rs=[r for r in rows if r['track_id']==track]
        if not rs:continue
        r=rs[0];altered=copy.deepcopy(r['semantic_task']);idfield='request_id' if field=='requests' else 'query_id'
        altered[field][0][idfield],altered[field][1][idfield]=altered[field][1][idfield],altered[field][0][idfield]
        recomputed=solve(altered)
        require(recomputed[outfield][0][idfield]!=answers[r['sample_id']][outfield][0][idfield],'ID intervention missing')
        probes.append({'attack':'identifier_only_status_or_abstention_baseline','source_sample_id':r['sample_id'],'source_record_hash':r['record_hash'],
                       'intervention':'Swap only the first two row identifiers, retaining all world/needs/as-of/write information.','task':altered,'correct':recomputed})
    out=ROOT/'audits'/'shortcut_detection';out.mkdir(parents=True,exist_ok=True)
    (out/('counterexamples-'+batch+'.json')).write_text(json.dumps(probes,indent=2,ensure_ascii=False)+'\n')
    return len(probes)


def main(batch='extended-pilot-v2'):
    paths=sorted((ROOT/'data').glob('*/candidate_renderings/'+batch+'.jsonl'))
    rows=[json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    receipts=[];answers={};errors=[]
    for r in rows:
        issues=[];critical=[];scores={k:5 for k in SCORE_KEYS};scores.update(generalization_value=4,difficulty_quality=4,diversity=4,shortcut_resistance=4)
        try:
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'candidate hash mismatch')
            answer=solve(r['semantic_task']);answers[r['sample_id']]=answer
            require(json.loads(r['rendered_input'].split('Task JSON:\n',1)[1])==r['semantic_task'],'rendered payload mismatch')
            require(all(r[field] not in r['rendered_input'] for field in ('sample_id','semantic_parent_id','semantic_family_id','record_hash')),'metadata leaked into input')
            if canon(answer)!=canon(r['verified_target']):
                critical.append('target_mismatch');issues.append(issue('target_mismatch','critical',{'computed':answer,'observed':r['verified_target']},'Correct independently recomputed target; add regression.'));scores['target_correctness']=scores['reasoning_validity']=1
            tests=metamorphic(r['semantic_task'],answer)
            require('scope_limitation' in r and r['scope_limitation'],'scope metadata missing')
        except Exception as e:
            critical.append('public_contract_failure');issues.append(issue('public_contract_failure','critical',str(e),'Repair task/target/rendering and replay independent checks.'));scores['semantic_correctness']=1;tests=[];errors.append({'sample_id':r['sample_id'],'error':str(e)})
        receipts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'receipt_id':'adv-'+batch+'-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:20],
                         'reviewer_role':'Adversarial Red-Team','reviewer_execution_id':EXECUTION,'additional_audit_role':'Diversity and leakage auditor','scores':scores,
                         'critical_defects':critical,'issues':issues,'decision':'REJECT' if critical else 'ACCEPT','confidence':0.95,
                         'evidence':{'independent_recomputation':True,'metamorphic_checks':tests,'private_generator_expectations_or_reviewer_verdicts_read':False,
                                     'scope':'Public finite proxy behavior only; record-level correctness is not proof of hierarchy, memory, Bayesian inference, latent learning, diffusion sampling, or theory of mind.'}})
    out=ROOT/'reviews'/'adversarial';out.mkdir(parents=True,exist_ok=True)
    first=out/(batch+'.first_judgments.jsonl')
    if not first.exists():first.write_text(''.join(canon(r)+'\n' for r in receipts))
    (out/(batch+'.jsonl')).write_text(''.join(canon(r)+'\n' for r in receipts))
    print(canon({'phase':'independent_first_judgments_saved','records':len(rows),'decisions':dict(Counter(r['decision'] for r in receipts)),'errors':errors}))
    if errors:return
    div,short,findings=audit_metrics(rows,answers)
    for directory,data in (('diversity',div),('shortcut_detection',short)):
        p=ROOT/'audits'/directory;p.mkdir(parents=True,exist_ok=True);(p/(batch+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for receipt,r in zip(receipts,rows):
        for f in findings:
            if r['track_id']==f['track']:
                receipt['issues'].append(issue(f['code'],f['severity'],f['evidence'],'Cross query/request IDs and input positions with both outcome classes; retain same-family matched controls.'))
                receipt['scores']['shortcut_resistance']=2;receipt['scores']['generalization_value']=3
                if not receipt['critical_defects']:receipt['decision']='REVISE'
        if r['track_id']=='RDP-001':
            receipt['issues'].append(issue('constraint_count_feasibility_association','low',
                'A constraint-count lookup predicts feasibility for both exposed development families; the single-constraint template is always an empty allowed set. Full repair, minimal Hamming distance and greedy selection still require the contract.',
                'Before measuring generalization, cross constraint count with satisfiable and unsatisfiable cases and evaluate complete repair outputs.'))
    (out/(batch+'.jsonl')).write_text(''.join(canon(r)+'\n' for r in receipts))
    execution={'reviewer_execution_id':EXECUTION,'batch':batch,'records':len(rows),'decisions':dict(Counter(r['decision'] for r in receipts)),
               'candidate_file_hashes':{str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in paths},
               'script_hash':__import__('hashlib').sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               'contract_hash':__import__('hashlib').sha256((ROOT/'schemas'/'EXTENDED_SEMANTICS.md').read_bytes()).hexdigest(),
               'first_judgments_preserved':True,'independent_solver_source':'This script, authored from public contracts; abstraction uses pair-distinguishability closure.',
               'metamorphic_counts':dict(Counter(test for r in receipts for test in r['evidence']['metamorphic_checks'])),
               'findings':findings,'exact_cross_family_duplicates':len(div['exact_duplicates_across_distinct_families'])}
    execution['minimal_attack_counterexamples']=attack_probes(rows,answers,batch)
    (out/('execution-'+batch+'.json')).write_text(json.dumps(execution,indent=2)+'\n')
    print(canon({'final_decisions':execution['decisions'],'findings':findings}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='extended-pilot-v2');args=parser.parse_args();main(args.batch)
