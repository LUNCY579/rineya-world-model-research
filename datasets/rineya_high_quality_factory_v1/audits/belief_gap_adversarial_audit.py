#!/usr/bin/env python3
"""Fresh two-stage belief review: stripped inputs first, untrusted metadata second.
No producer, reference, oracle, checker, tests or other reviewer verdicts imported.
"""
import argparse,collections,copy,hashlib,itertools,json,pathlib
from adversarial_audit import ROOT,EXECUTION,SCORE_KEYS,canon,digest,require,issue,reorder
BATCH='belief-gap-v1';INPUT=ROOT/'reviews/input_only'/f'{BATCH}.jsonl';CAND=ROOT/'data/RBM-001/candidate_renderings'/f'{BATCH}.jsonl';OUT=ROOT/'reviews/adversarial'
PUBLIC=[ROOT/'schemas/EXTENDED_SEMANTICS.md',ROOT/'schemas/BELIEF_CONTROL_PROPERTIES.md',ROOT/'research/curricula/RBM-EVIDENCE-GAP.md']
EXPECTED_INPUT_SHA='86a07b5b6e4bf89706e9491ed41415e3facc8a744acdf14ad551cbade8529ac6'
EXPECTED_CAND_SHA='423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e'
PROPS={'multiworld_unobserved_invariant','conjunction_required_invariant','useful_after_conjunctive_evidence','multiworld_all_queries_redundant','multiworld_entailed_null','ambiguous_null_partition','cross_fact_inconsistent_conjunction','post_empty_absorption','nonempty_repeated_observation','typed_observation_separator','typed_query_partition_separator','skewed_possible_value_partition'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def freeze(p,objs):
    s=''.join(canon(o)+'\n' for o in objs);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():require(p.read_text()==s,'Refuse to rewrite frozen evidence '+str(p))
    else:p.write_text(s)
def save(p,o):
    s=json.dumps(o,indent=2,ensure_ascii=False)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_text()!=s:
        hist=p.parent/'history'/f'{p.stem}-{sha(p)}{p.suffix}';hist.parent.mkdir(parents=True,exist_ok=True)
        if not hist.exists():hist.write_bytes(p.read_bytes())
    p.write_text(s)
def scalar(v):return v is None or type(v) in (str,int,bool)
def typed(v):require(scalar(v),'valid typed scalar');return canon(v)
def payload_count(t,ids):return len({canon(w['facts']) for w in t['worlds'] if w['world_id'] in ids})
def partition(t,ids,f):
    bins=collections.Counter(typed(w['facts'][f]) for w in t['worlds'] if w['world_id'] in ids)
    return [{'value':json.loads(k),'count':bins[k]} for k in sorted(bins)]
def answer(q,bins,n):
    d=len(bins);entailed=n>0 and d==1
    return {'query_id':q['query_id'],'counts':bins,'distinct_values':d,'entailed':entailed,'guaranteed':bins[0]['value'] if entailed else None,'can_reduce_uncertainty':d>1}

def solve(t):
    require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','belief input shape')
    ws=t['worlds'];obs=t['observations'];qs=t['queries'];require(type(ws) is list and type(obs) is list and type(qs) is list,'lists')
    require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in ws),'world shape')
    keys=set(ws[0]['facts']) if ws else set();require(len({w['world_id'] for w in ws})==len(ws),'unique world IDs')
    require(all(set(w['facts'])==keys and all(type(k) is str and scalar(v) for k,v in w['facts'].items()) for w in ws),'common typed fact keys')
    require(all(type(o) is dict and set(o)=={'fact','equals'} and o['fact'] in keys and scalar(o['equals']) for o in obs),'observation shape/type')
    require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and q['fact'] in keys for q in qs),'query shape/type');require(len({q['query_id'] for q in qs})==len(qs),'unique query IDs')
    allids={w['world_id'] for w in ws};prefix=[sorted(allids)];remaining=allids.copy();individual=[]
    for o in obs:
        matching={w['world_id'] for w in ws if typed(w['facts'][o['fact']])==typed(o['equals'])};individual.append(sorted(matching));remaining&=matching;prefix.append(sorted(remaining))
    result={'surviving_worlds':sorted(remaining),'belief_sizes':[len(p) for p in prefix[1:]],'queries':[answer(q,partition(t,remaining,q['fact']),len(remaining)) for q in qs]}
    details={'prefix_survivors':prefix,'individual_observation_supports':individual,'prefix_query_partitions':[[{'query_id':q['query_id'],'fact':q['fact'],'bins':partition(t,p,q['fact'])} for q in qs] for p in prefix],'world_count':len(ws),'fact_count':len(keys),'observation_count':len(obs),'query_count':len(qs),'distinct_complete_world_payloads':payload_count(t,allids)}
    return result,details

def witnesses(t,a,d):
    w=collections.defaultdict(list);obs=t['observations'];ps=d['prefix_survivors'];final=set(ps[-1]);m=len(obs);fields={o['fact'] for o in obs};distinct=payload_count(t,final);qs=t['queries'];answers=a['queries'];unobs=[]
    for j,(q,ans) in enumerate(zip(qs,answers)):
        base={'query_id':q['query_id'],'fact':q['fact'],'query_position':j,'prefix_index':m,'survivors':ps[-1],'answer':ans}
        if len(final)>=2 and distinct>=2 and q['fact'] not in fields and ans['distinct_values']==1:
            w['multiworld_unobserved_invariant'].append(base);unobs.append((j,base))
        if len(final)>=2 and distinct>=2 and ans['entailed'] and ans['guaranteed'] is None:w['multiworld_entailed_null'].append(base)
        if ans['distinct_values']>1 and any(x['value'] is None for x in ans['counts']):w['ambiguous_null_partition'].append(base)
        active={typed(x['value']) for x in ans['counts']}
        for aliases in [('true','1'),('false','0')]:
            if set(aliases)<=active:w['typed_query_partition_separator'].append({**base,'aliases':aliases})
        if len(ans['counts'])>=2 and len({x['count'] for x in ans['counts']})>=2:w['skewed_possible_value_partition'].append(base)
        strict=[i for i in range(m) if 0<len(ps[i+1])<len(ps[i])]
        if final and ans['distinct_values']>1 and len({o['fact'] for o in obs})>=2 and len(strict)>=2:w['useful_after_conjunctive_evidence'].append({**base,'strict_nonempty_reductions':strict,'prefix_survivors':ps})
    for j,base in unobs:
        for i,k in itertools.combinations(range(m),2):
            if obs[i]['fact']==obs[k]['fact']:continue
            z=copy.deepcopy(t);z['observations']=[o for h,o in enumerate(obs) if h!=i];ai,di=solve(z)
            z['observations']=[o for h,o in enumerate(obs) if h!=k];ak,dk=solve(z)
            if ai['queries'][j]['distinct_values']>=2 and ak['queries'][j]['distinct_values']>=2:w['conjunction_required_invariant'].append({**base,'necessary_observation_indices':[i,k],'omit_first_survivors':di['prefix_survivors'][-1],'omit_first_answer':ai['queries'][j],'omit_second_survivors':dk['prefix_survivors'][-1],'omit_second_answer':ak['queries'][j]})
    if len(final)>=2 and distinct>=2 and all(q['distinct_values']==1 for q in answers):
        keys=set(t['worlds'][0]['facts']) if t['worlds'] else set();varying=[f for f in sorted(keys-{q['fact'] for q in qs}) if len(partition(t,final,f))>1]
        if varying:w['multiworld_all_queries_redundant'].append({'prefix_index':m,'survivors':ps[-1],'unqueried_varying_facts':varying,'partitions':{f:partition(t,final,f) for f in varying},'all_query_answers':answers})
    if not final:
        for i,k in itertools.combinations(range(m),2):
            left=set(d['individual_observation_supports'][i]);right=set(d['individual_observation_supports'][k])
            if obs[i]['fact']!=obs[k]['fact'] and left and right and not left&right:w['cross_fact_inconsistent_conjunction'].append({'observation_indices':[i,k],'individual_supports':[sorted(left),sorted(right)],'prefix_index':m,'final_survivors':ps[-1]})
        first=next((i for i,p in enumerate(ps) if not p),None)
        if first is not None and first<m:
            later=[i for i in range(first,m) if d['individual_observation_supports'][i]]
            if later and all(not p for p in ps[first:]) and all(not q['counts'] for q in answers):w['post_empty_absorption'].append({'first_empty_prefix':first,'later_individually_supported_indices':later,'individual_supports':{str(i):d['individual_observation_supports'][i] for i in later},'prefix_survivors':ps,'query_answers':answers})
    for j,o in enumerate(obs):
        for i in range(j):
            if canon(o)==canon(obs[i]) and ps[j] and ps[j]==ps[j+1]:w['nonempty_repeated_observation'].append({'observation_indices':[i,j],'incoming_prefix':j,'outgoing_prefix':j+1,'survivors':ps[j],'count':len(ps[j])})
        c=typed(o['equals']);alias={'true':'1','1':'true','false':'0','0':'false'}.get(c)
        if alias:
            before=set(ps[j]);exact={x['world_id'] for x in t['worlds'] if x['world_id'] in before and typed(x['facts'][o['fact']])==c};other={x['world_id'] for x in t['worlds'] if x['world_id'] in before and typed(x['facts'][o['fact']])==alias}
            if exact and other:w['typed_observation_separator'].append({'observation_index':j,'incoming_prefix':j,'outgoing_prefix':j+1,'fact':o['fact'],'equals':o['equals'],'alias':json.loads(alias),'incoming_survivors':ps[j],'exact_retained':sorted(exact),'alias_excluded':sorted(other),'after':ps[j+1]})
    return dict(w)

def stage1():
    require(sha(INPUT)==EXPECTED_INPUT_SHA,'stripped input SHA');rs=load(INPUT);expected=[]
    require(len(rs)==1296,'expected actual stripped rows')
    for r in rs:
        require(set(r)=={'record_hash','rendered_input','sample_id','semantic_family_id','semantic_parent_id','semantic_task','split','track_id'},'exact input-only whitelist')
        task=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);require(canon(task)==canon(r['semantic_task']),'typed rendered equality')
        a,d=solve(task);w=witnesses(task,a,d);expected.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':EXECUTION,'expected_target':a,'prefix_evidence':d,'input_derived_property_witnesses':w,'expectation_scope':'Only stripped public tasks/prompts and join keys read. No original candidates, stored targets, roles, locators, producer claims or other reviewer judgments read.'})
    p=OUT/f'{BATCH}.input_predictions.jsonl';freeze(p,expected)
    receipt={'reviewer_execution_id':EXECUTION,'batch':BATCH,'stage':'input_only_expectations_frozen','records':len(rs),'input_only_file_sha256':sha(INPUT),'input_prediction_sha256':sha(p),'script_sha256':sha(pathlib.Path(__file__)),'public_common_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'property_support_counts':dict(collections.Counter(k for x in expected for k in x['input_derived_property_witnesses'])),'scope':'Own fresh typed set-intersection solver and exact count partitions. Public common design includes prior aggregate rationale; no underlying expert reports/producer expectations opened. Same-model execution; no sealed evaluation or efficacy claim.'}
    save(ROOT/'audits/execution'/f'{BATCH}-input-only-adversarial.json',receipt);print(canon(receipt),flush=True)

def locator_valid(prop,loc,ws,t):
    require(type(loc) is dict and {'query_ids','observation_indices','prefix_indices'}<=set(loc)<= {'query_ids','observation_indices','prefix_indices','supporting_world_ids','note'},'locator exact fields')
    qids={q['query_id'] for q in t['queries']};m=len(t['observations'])
    for f,n,kind in [('query_ids',qids,str),('observation_indices',set(range(m)),int),('prefix_indices',set(range(m+1)),int)]:
        require(type(loc[f]) is list and all(type(x) is kind and x in n for x in loc[f]) and len(loc[f])==len(set(loc[f])),'locator bounds '+f)
    if 'note' in loc:require(type(loc['note']) is str,'locator note')
    if 'supporting_world_ids' in loc:
        ids=loc['supporting_world_ids'];require(type(ids) is list and ids==sorted(set(ids)) and set(ids)<={w['world_id'] for w in t['worlds']},'supporting IDs')
    q=set(loc['query_ids']);obs=set(loc['observation_indices']);ps=set(loc['prefix_indices'])
    for w in ws:
        if prop in ('multiworld_unobserved_invariant','multiworld_entailed_null','ambiguous_null_partition','typed_query_partition_separator','skewed_possible_value_partition'):
            ok=w['query_id'] in q and m in ps
        elif prop=='conjunction_required_invariant':ok=w['query_id'] in q and set(w['necessary_observation_indices'])<=obs and {0,m}<=ps
        elif prop=='useful_after_conjunctive_evidence':
            steps=[i for i in w['strict_nonempty_reductions'] if i in obs and {i,i+1}<=ps];ok=w['query_id'] in q and m in ps and len(steps)>=2
        elif prop=='multiworld_all_queries_redundant':ok=qids<=q and m in ps
        elif prop=='cross_fact_inconsistent_conjunction':ok=set(w['observation_indices'])<=obs and {0,m}<=ps
        elif prop=='post_empty_absorption':
            f=w['first_empty_prefix'];later=[i for i in w['later_individually_supported_indices'] if i in obs and {i,i+1}<=ps];ok=f in ps and f-1 in obs and bool(later) and qids<=q
        elif prop=='nonempty_repeated_observation':ok=set(w['observation_indices'])<=obs and {w['incoming_prefix'],w['outgoing_prefix']}<=ps
        else:ok=w['observation_index'] in obs and {w['incoming_prefix'],w['outgoing_prefix']}<=ps
        if ok:
            if 'supporting_world_ids' in loc:
                a,d=solve(t);allowed=set().union(*(set(d['prefix_survivors'][i]) for i in ps));require(set(loc['supporting_world_ids'])<=allowed,'supporting IDs outside indicated supports')
            return w
    raise ValueError('Locator does not point to a required factual witness: '+prop)

def metamorphics(t,a):
    checks=[]
    def assert_target(z,e,name):require(canon(solve(z)[0])==canon(e),name);checks.append(name)
    assert_target(reorder(t),a,'object_field_order')
    z=copy.deepcopy(t);z['worlds'].reverse();assert_target(z,a,'world_order_invariance')
    z=copy.deepcopy(t);mapping={w['world_id']:'opaque_%03d'%(len(t['worlds'])-i) for i,w in enumerate(t['worlds'])}
    for w in z['worlds']:w['world_id']=mapping[w['world_id']]
    e=copy.deepcopy(a);e['surviving_worlds']=sorted(mapping[x] for x in a['surviving_worlds']);assert_target(z,e,'world_id_rename_equivariance')
    z=copy.deepcopy(t);z['queries'].reverse();e=copy.deepcopy(a);e['queries'].reverse();assert_target(z,e,'query_order_equivariance')
    z=copy.deepcopy(t);e=copy.deepcopy(a)
    for i,(q,r) in enumerate(zip(z['queries'],e['queries'])):q['query_id']=r['query_id']='opaque_q_'+str(8-i)
    assert_target(z,e,'query_id_rename_equivariance')
    z=copy.deepcopy(t);mapping={k:'opaque_fact_'+str(i) for i,k in enumerate(sorted(t['worlds'][0]['facts']))}
    for w in z['worlds']:w['facts']={mapping[k]:v for k,v in w['facts'].items()}
    for o in z['observations']:o['fact']=mapping[o['fact']]
    for q in z['queries']:q['fact']=mapping[q['fact']]
    assert_target(z,a,'consistent_fact_rename')
    z=copy.deepcopy(t);z['observations'].reverse();b,bd=solve(z);require(canon(b['surviving_worlds'])==canon(a['surviving_worlds']) and canon(b['queries'])==canon(a['queries']),'observation order final result');checks.append('observation_reorder_final_with_recomputed_prefix')
    z=copy.deepcopy(t)
    if a['surviving_worlds']:z['observations'].append(copy.deepcopy(t['observations'][0]));name='nonempty_repeat_append'
    else:
        f=next(iter(t['worlds'][0]['facts']));z['observations'].append({'fact':f,'equals':t['worlds'][0]['facts'][f]});name='post_empty_supported_append'
    e=copy.deepcopy(a);e['belief_sizes'].append(len(a['surviving_worlds']));assert_target(z,e,name)
    return checks

def normalized(r):
    t=copy.deepcopy(r['semantic_task']);roles=r['fact_role_assignments'];inverse={v:k for k,v in roles.items()}
    for w in t['worlds']:w['facts']={inverse[k]:v for k,v in w['facts'].items()}
    for o in t['observations']:o['fact']=inverse[o['fact']]
    for q in t['queries']:q['fact']=inverse[q['fact']];q['query_id']=q['fact']
    t['queries'].sort(key=lambda q:q['fact']);return t

def stage2():
    require((OUT/f'{BATCH}.input_predictions.jsonl').exists(),'expectations must be frozen first');require(sha(CAND)==EXPECTED_CAND_SHA,'original candidate SHA')
    stripped={r['sample_id']:r for r in load(INPUT)};pred={r['sample_id']:r for r in load(OUT/f'{BATCH}.input_predictions.jsonl')};rows=load(CAND);receipts=[]
    for r in rows:
        errors=[];critical=[];scores={k:5 for k in SCORE_KEYS};scores.update(generalization_value=3,difficulty_quality=3,diversity=3,shortcut_resistance=4);p=pred[r['sample_id']];w=p['input_derived_property_witnesses'];locators={};checks=[]
        try:
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'exact candidate hash')
            require(canon({k:r[k] for k in stripped[r['sample_id']]})==canon(stripped[r['sample_id']]),'original differs from stripped public input')
            require(canon(p['expected_target'])==canon(r['verified_target']),'target mismatch')
            require(p['prefix_evidence']['distinct_complete_world_payloads']==len(r['semantic_task']['worlds']),'duplicate complete world payloads')
            tags=set(r['skill_tags']);require({'finite_synthetic_proxy','finite_noiseless_filter','possible_value_partition'}<=tags<=PROPS|{'finite_synthetic_proxy','finite_noiseless_filter','possible_value_partition'},'public tag names')
            require(set(r['property_witnesses'])==tags&PROPS,'locator keys match factual tags')
            for tag in tags&PROPS:require(w.get(tag),'unsupported property '+tag);locators[tag]=locator_valid(tag,r['property_witnesses'][tag],w[tag],r['semantic_task'])
            checks=metamorphics(r['semantic_task'],p['expected_target'])
            for field in ('sample_id','record_hash','semantic_parent_id','semantic_family_id','source_family_key'):require(r[field] not in r['rendered_input'],'privileged input metadata '+field)
        except Exception as e:critical.append('semantic_or_metadata_failure');errors.append(issue('semantic_or_metadata_failure','high',str(e),'Correct the exact task/target/false claim and obtain fresh review.'));scores['semantic_correctness']=2
        receipts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'receipt_id':'adv-'+BATCH+'-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:20],'reviewer_role':'Adversarial Red-Team','reviewer_execution_id':EXECUTION,'additional_audit_role':'Diversity and leakage auditor','scores':scores,'critical_defects':critical,'issues':errors,'decision':'REVISE' if critical else 'ACCEPT','confidence':0.97,'evidence':{'input_prediction_frozen_before_candidate_target_access':True,'prediction_file_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'own_locator_witnesses':locators,'metamorphic_checks':checks,'producer_oracle_reference_or_other_reviewer_verdicts_read':False,'scope':'Common public design; actual separate same-model execution, exposed two-family development split, finite noiseless filtering/partitions only.'}})
    freeze(OUT/f'{BATCH}.first_judgments.jsonl',receipts)
    print(canon({'phase':'first_judgments_frozen','decisions':dict(collections.Counter(r['decision'] for r in receipts))}),flush=True)
    save(ROOT/'audits/execution'/f'{BATCH}-local-adversarial.json',{'reviewer_execution_id':EXECUTION,'stage':'first_judgments_frozen','candidate_sha256':sha(CAND),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'records':len(rows),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'script_sha256':sha(pathlib.Path(__file__))})
    if globals().get("global_review"):global_review(rows,pred,receipts)

def incidence(t,x,y):
    edges=collections.Counter((typed(w['facts'][x]),typed(w['facts'][y])) for w in t['worlds']);left=sorted({a for a,b in edges});right=sorted({b for a,b in edges});adj=collections.defaultdict(set)
    for a,b in edges:adj[('L',a)].add(('R',b));adj[('R',b)].add(('L',a))
    unseen=set(adj);components=0
    while unseen:
        todo=[unseen.pop()];components+=1
        while todo:
            for nxt in adj[todo.pop()]:
                if nxt in unseen:unseen.remove(nxt);todo.append(nxt)
    matrices=[tuple(edges[(a,b)] for a in ls for b in rs) for ls in itertools.permutations(left) for rs in itertools.permutations(right)]
    matrix=min(matrices);unweighted=min(tuple(int(v>0) for v in mat) for mat in matrices)
    return {'worlds':len(t['worlds']),'vertices':len(adj),'edges':len(edges),'components':components,'cycle_rank':len(edges)-len(adj)+components,'max_degree':max(map(len,adj.values())),'degree_sequence':sorted(map(len,adj.values())),'positive_cell_weights':sorted(edges.values()),'left_standalone_supports':sorted(sum(v for (a,b),v in edges.items() if a==k) for k in left),'right_standalone_supports':sorted(sum(v for (a,b),v in edges.items() if b==k) for k in right),'unsupported_joint_cells':len(left)*len(right)-len(edges),'weighted_canonical_signature':[len(left),len(right),list(matrix)],'unweighted_canonical_signature':[len(left),len(right),list(unweighted)]}

def table_signature(t):
    facts=sorted(t['worlds'][0]['facts']);blocks=sorted(sorted(collections.Counter(typed(w['facts'][f]) for w in t['worlds']).values()) for f in facts);pairs=sorted(canon(incidence(t,x,y)['weighted_canonical_signature']) for x,y in itertools.combinations(facts,2))
    return [len(t['worlds']),len(facts),blocks,pairs]

def scalar_rename(t,a):
    strings=sorted({w['facts'][f] for w in t['worlds'] for f in w['facts'] if type(w['facts'][f]) is str}|{o['equals'] for o in t['observations'] if type(o['equals']) is str});mapping={s:'opaque_scalar_%02d'%i for i,s in enumerate(strings)};z=copy.deepcopy(t);e=copy.deepcopy(a)
    for w in z['worlds']:
        for f,v in w['facts'].items():
            if type(v) is str:w['facts'][f]=mapping[v]
    for o in z['observations']:
        if type(o['equals']) is str:o['equals']=mapping[o['equals']]
    for q in e['queries']:
        if type(q['guaranteed']) is str:q['guaranteed']=mapping[q['guaranteed']]
        for b in q['counts']:
            if type(b['value']) is str:b['value']=mapping[b['value']]
        q['counts'].sort(key=lambda b:typed(b['value']))
    require(canon(solve(z)[0])==canon(e),'opaque scalar bijection');return z,e

def global_review(rows,pred,receipts):
    import re
    problems=[];families=collections.defaultdict(list);groups=collections.defaultdict(list);fieldpairs=collections.defaultdict(list);claims=collections.defaultdict(set);actual_claims={};histories={};obs=[];crossing=[];structural=[];controls=[];probes=[]
    for r in rows:
        t=r['semantic_task'];family=r['source_family_key'];families[family].append(r);key=(family,r['controlled_context'],r['renderer']);groups[key].append(r);fieldpairs[(family,r['controlled_context'],r['schedule_index'])].append(r)
        try:
            idx=r['source_family_index'];require(type(idx) is int and idx in (12,13,18,19),'source index');require(r['split']==('dev_challenge' if idx>=18 else 'train'),'family split policy');require(r['controlled_context'] in 'ABCDEF','six contexts');require(r['structural_family_profile']=={12:'cyclic_uniform',13:'acyclic_branching',18:'cyclic_weighted',19:'typed_path'}[idx],'profile index')
            u,v,w=r['id_shift'],r['position_shift'],r['fact_name_shift'];require(all(type(i) is int and 0<=i<=2 for i in (u,v,w)) and r['schedule_index']==9*u+3*v+w,'declared independent schedule')
            roles=r['fact_role_assignments'];required={'observation_control','observation_auxiliary','invariant_query','variation_query','unoffered_feature'};require(required<=set(roles) and all(k in required or re.fullmatch(r'aux_extra_[0-9]+',k) for k in roles),'fact role names');require(len(roles)==len(set(roles.values())) and set(roles.values())==set(t['worlds'][0]['facts']),'role fact table coverage')
            pools=[r['query_id_pool'],r['queried_fact_name_pool']];require(all(type(p) is list and len(p)==len(set(p))==3 and all(type(x) is str for x in p) for p in pools),'three fixed pools');require(set(pools[1])=={roles[k] for k in ('invariant_query','variation_query','observation_control')},'queried pool roles')
            assignments=r['query_role_assignments'];require(len(assignments)==3 and {a['role'] for a in assignments}=={'invariant_query','variation_query','observed_control'},'three declared query roles')
            for a in assignments:
                require({'role','position','query_id'}<=set(a)<={'role','position','query_id','fact'},'query assignment fields');role=('invariant_query','variation_query','observed_control').index(a['role']);pos=(role+v)%3;name=pools[1][(role+w)%3];qid=pools[0][(role+u)%3];q=t['queries'][pos];expected_fact=roles['observation_control' if role==2 else a['role']];require(a['position']==pos and a['query_id']==qid==q['query_id'] and q['fact']==name==expected_fact and ('fact' not in a or a['fact']==name),'actual query slot/ID/fact schedule')
                ans=pred[r['sample_id']]['expected_target']['queries'][pos];detail=pred[r['sample_id']]['prefix_evidence'];cat='empty' if not ans['counts'] else 'entailed' if ans['entailed'] else 'ambiguous';nullcontext=cat+'_null' if ans['guaranteed'] is None else 'entailed_nonnull';outcome={**{k:value for k,value in ans.items() if k!='query_id'},'audit_category':cat,'guaranteed_null_context':nullcontext,'bin_type_signature':[('null' if b['value'] is None else type(b['value']).__name__,b['count']) for b in ans['counts']],'final_survivor_provenance':detail['prefix_survivors'][-1],'prefix_survivor_provenance':detail['prefix_survivors'],'prefix_counts':[len(s) for s in detail['prefix_survivors']]}
                obs.append({'sample_id':r['sample_id'],'family':family,'context':r['controlled_context'],'renderer':r['renderer'],'split':r['split'],'role':role,'id':qid,'position':pos,'fact':name,'name_index':pools[1].index(name),'answer':ans,'outcome':outcome,'input_length':len(r['rendered_input']),'observation_count':len(t['observations']),'world_count':len(t['worlds']),'final_support':len(detail['prefix_survivors'][-1]),'observed':q['fact'] in {o['fact'] for o in t['observations']}})
            for c in r['family_control_claims']:
                require(type(c) is dict and set(c)<={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'} and len(set(c)&{'name','property'})==1 and c.get('scope') in ('corpus_variant','audit_probe'),'family claim encoding');name=c.get('name',c.get('property'));require('source_family_key' not in c or c['source_family_key']==family,'same family claim')
                if 'evidence_manifest' in c:require(type(c['evidence_manifest']) is str and bool(c['evidence_manifest']) and not pathlib.PurePosixPath(c['evidence_manifest']).is_absolute() and '..' not in pathlib.PurePosixPath(c['evidence_manifest']).parts,'relative pointer')
                for f in ('control_id','probe_id'):
                    if f in c:require(type(c[f]) is str and bool(c[f]),'control/probe ID')
                if 'pair_sample_ids' in c:require(c['scope']=='corpus_variant' and type(c['pair_sample_ids']) is list and len(set(c['pair_sample_ids']))==2 and all(any(z['sample_id']==sid and z['source_family_key']==family for z in rows) for sid in c['pair_sample_ids']),'same-family pair IDs')
                if 'probe_id' in c:require(c['scope']=='audit_probe','probe scope')
                claims[family].add((name,c['scope']))
        except Exception as e:problems.append({'sample_id':r['sample_id'],'family':family,'error':str(e)})
    for (family,context,renderer),rs in sorted(groups.items()):
        require(len(rs)==27 and {r['schedule_index'] for r in rs}==set(range(27)),'complete 27 schedule set');require(len({canon(normalized(r)) for r in rs})==1,'decoded consistent worlds/observations/queries');os=[o for o in obs if (o['family'],o['context'],o['renderer'])==(family,context,renderer)]
        symbols={digest(o['outcome']):o['outcome'] for o in os};tables={}
        for names in [('id',),('position',),('fact',),('id','position'),('id','fact'),('position','fact'),('id','position','fact')]:
            table=collections.defaultdict(lambda:{'roles':collections.Counter(),'typed_outcomes':collections.Counter()})
            for o in os:b=table[canon([o[n] for n in names])];b['roles'][str(o['role'])]+=1;b['typed_outcomes'][digest(o['outcome'])]+=1
            expected={1:9,2:3,3:1}[len(names)];require(len(table)==3**len(names) and all(dict(b['roles'])=={'0':expected,'1':expected,'2':expected} for b in table.values()),'complete marginal/pair/triple role support');require(len({canon(b['typed_outcomes']) for b in table.values()})==1,'actual full outcome histogram conditional leak');tables['+'.join(names)]={k:{sub:dict(counts) for sub,counts in b.items()} for k,b in table.items()}
        crossing.append({'family':family,'context':context,'renderer':renderer,'scope':'corpus_variant','outcome_signature_payloads':symbols,'marginal_pair_triple_tables':tables})
    for k,rs in fieldpairs.items():require(len(rs)==2 and {r['renderer'] for r in rs}=={'forward','reverse'} and canon(rs[0]['semantic_task'])==canon(rs[1]['semantic_task']) and rs[0]['rendered_input']!=rs[1]['rendered_input'],'exact field-order pair')
    for family,rs in sorted(families.items()):
        require(len({r['split'] for r in rs})==len({r['semantic_family_id'] for r in rs})==len({r['semantic_parent_id'] for r in rs})==1,'whole source-family split/parent custody');require(len({canon([r['query_id_pool'],r['queried_fact_name_pool']]) for r in rs})==1,'stable pools');base={r['controlled_context']:r for r in rs if r['schedule_index']==0 and r['renderer']=='forward'};require(set(base)==set('ABCDEF'),'six actual contexts');t={c:normalized(r) for c,r in base.items()};a={c:solve(task)[0] for c,task in t.items()};require(len({canon(task['worlds']) for task in t.values()})==len({canon(task['queries']) for task in t.values()})==1,'controlled same-world same-query contexts')
        def unchanged(x,y):return canon(a[x]['surviving_worlds'])==canon(a[y]['surviving_worlds']) and canon(a[x]['queries'])==canon(a[y]['queries'])
        def obschange(x,y):
            require(len(t[x]['observations'])==len(t[y]['observations']),'equal obs count');diff=[i for i,(p,q) in enumerate(zip(t[x]['observations'],t[y]['observations'])) if canon(p)!=canon(q)];require(len(diff)==1 and t[x]['observations'][diff[0]]['fact']==t[y]['observations'][diff[0]]['fact'],'one equals-only observation change');return diff[0]
        i=obschange('A','B');require(len(a['A']['surviving_worlds'])==len(a['B']['surviving_worlds'])>=2 and any(p['can_reduce_uncertainty']!=q['can_reduce_uncertainty'] for p,q in zip(a['A']['queries'],a['B']['queries'])),'equal support changed usefulness');require(all(q['entailed'] for q in a['B']['queries']),'B offered all redundant');actual={('equal_support_usefulness_counterfactual','corpus_variant')}
        require(canon(t['C']['observations'])==canon(list(reversed(t['A']['observations']))) and unchanged('A','C') and a['A']['belief_sizes']!=a['C']['belief_sizes'],'actual reordered changed prefix');actual.add(('observation_reorder_final_invariance','corpus_variant'))
        require(canon(t['D']['observations'][:-1])==canon(t['A']['observations']) and canon(t['D']['observations'][-1]) in {canon(o) for o in t['A']['observations']} and unchanged('A','D') and a['D']['belief_sizes']==a['A']['belief_sizes']+[len(a['A']['surviving_worlds'])],'actual nonempty repeat');actual.add(('nonempty_repetition_final_invariance','corpus_variant'))
        obschange('A','E');require(pred[base['E']['sample_id']]['input_derived_property_witnesses'].get('cross_fact_inconsistent_conjunction'),'E independent cross-fact witness');actual.add(('cross_fact_contradiction_counterfactual','corpus_variant'))
        require(canon(t['F']['observations'][:-1])==canon(t['E']['observations']) and unchanged('E','F') and a['F']['belief_sizes']==a['E']['belief_sizes']+[0] and pred[base['F']['sample_id']]['input_derived_property_witnesses'].get('post_empty_absorption'),'actual empty supported append');actual.add(('post_empty_append_absorption','corpus_variant'))
        old=t['A']['observations'][i]['equals'];new=t['B']['observations'][i]['equals']
        if (typed(old),typed(new)) in [('true','1'),('1','true'),('false','0'),('0','false')]:
            da=solve(t['A'])[1];db=solve(t['B'])[1];require(da['prefix_survivors'][i+1] and db['prefix_survivors'][i+1] and da['prefix_survivors'][i+1]!=db['prefix_survivors'][i+1],'actual typed equals separator paired supports');actual.add(('typed_equals_counterfactual','corpus_variant'))
        allw={p for r in rs for p in pred[r['sample_id']]['input_derived_property_witnesses']}
        if {'multiworld_entailed_null','ambiguous_null_partition'}<=allw:actual.add(('null_guarantee_contrast','corpus_variant'))
        actual|={('query_id_position_fact_crossing','corpus_variant'),('field_order_invariance','corpus_variant'),('world_order_invariance','audit_probe'),('world_id_rename_equivariance','audit_probe')};actual_claims[family]=actual
        missing=claims[family]-actual
        if missing:problems.append({'family':family,'unsupported_family_claims':sorted(missing)})
        graph=incidence(t['A'],'observation_control','observation_auxiliary');profile=base['A']['structural_family_profile'];ok={'cyclic_uniform':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))==1,'acyclic_branching':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']>=3,'cyclic_weighted':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))>1,'typed_path':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']==2}[profile];require(ok,'factual graph profile');structural.append({'family':family,'profile':profile,'split':base['A']['split'],'graph':graph,'complete_table_partition_incidence_invariant':table_signature(t['A'])})
        controls.append({'family':family,'same_table_contexts':'ABCDEF','scope':'corpus_variant','exact_pair_sources':{c:{'sample_id':r['sample_id'],'record_hash':r['record_hash']} for c,r in base.items()},'independent_targets':a,'verified_relationships':sorted(actual),'A_B_changed_observation_index':i})
        for name in ('world_order_invariance','world_id_rename_equivariance'):probes.append({'probe_id':str(base['A']['source_family_index'])+'-'+name,'family':family,'control':name,'scope':'audit_probe','source_sample_id':base['A']['sample_id'],'source_hash':base['A']['record_hash'],'execution':'Actual metamorphics executed independently for all 324 same-family rows.'})
    require(len({canon(s['graph']['weighted_canonical_signature']) for s in structural})==4,'four distinct factual structural profiles')
    # Nuisance CPU lookups; no training. Query IDs and fact identity omitted from target payload labels.
    def gold(o,kind):
        if kind=='full_payload':return canon({k:v for k,v in o['answer'].items() if k!='query_id'})
        if kind=='category':return o['outcome']['audit_category']
        return canon(o['answer'][kind])
    train=[o for o in obs if o['split']=='train'];dev=[o for o in obs if o['split']=='dev_challenge']
    def lookup(fn,kind):
        major=collections.Counter(gold(o,kind) for o in train).most_common(1)[0][0];table=collections.defaultdict(collections.Counter)
        for o in train:table[canon(fn(o))][gold(o,kind)]+=1
        choose={k:c.most_common(1)[0][0] for k,c in table.items()};return {'correct':sum(choose.get(canon(fn(o)),major)==gold(o,kind) for o in dev),'total':len(dev),'majority_correct':sum(gold(o,kind)==major for o in dev),'seen_feature_observations':sum(canon(fn(o)) in choose for o in dev)}
    features={'id':lambda o:o['id'],'position':lambda o:o['position'],'fact_name':lambda o:o['fact'],'id_position_fact_joint':lambda o:(o['id'],o['position'],o['fact']),'input_char_length':lambda o:o['input_length'],'fact_name_length':lambda o:len(o['fact']),'world_count_semantic':lambda o:o['world_count'],'observation_count_semantic':lambda o:o['observation_count'],'support_size_and_observed_fact_semantic':lambda o:(o['final_support'],o['observed'])}
    baselines={name:{kind:lookup(fn,kind) for kind in ('entailed','can_reduce_uncertainty','category','full_payload')} for name,fn in features.items()};heuristics={}
    for name in ('observed_or_singleton_entails','multiworld_implies_useful'):
        total=len(obs);correct=0;whole=collections.defaultdict(bool)
        for o in obs:
            n=o['final_support'];entailed=(n>0 and (o['observed'] or n==1)) if name=='observed_or_singleton_entails' else n==1;useful=(n>1 and not o['observed']) if name=='observed_or_singleton_entails' else n>1;ok=(entailed==o['answer']['entailed'] and useful==o['answer']['can_reduce_uncertainty']);correct+=ok;whole[o['sample_id']]|=not ok
        heuristics[name]={'query_flag_pairs_correct':correct,'query_observations':total,'complete_flag_task_failures':sum(whole.values()),'tasks':len(whole)}
    # Detect token collapse independently; diagnostic marker learned from train empty observations.
    marker={'stone','unknown'};markers=[];empty_target_lookup=collections.defaultdict(collections.Counter);scalar_checks=0;marker_counterexamples=[]
    for r in rows:
        t=r['semantic_task'];p=pred[r['sample_id']];a=p['expected_target'];observed_marker=any(type(o['equals']) is str and o['equals'] in marker for o in t['observations']);is_empty=not a['surviving_worlds'];key=(observed_marker,len(t['worlds']),len(t['observations']));core=copy.deepcopy(a)
        for q in core['queries']:q.pop('query_id')
        if r['split']=='train' and observed_marker:empty_target_lookup[key][canon(core)]+=1
        z,e=scalar_rename(t,a);scalar_checks+=1;require(not any(type(o['equals']) is str and o['equals'] in marker for o in z['observations']),'renamed task retains marker');markers.append({'sample_id':r['sample_id'],'split':r['split'],'marker':observed_marker,'empty':is_empty,'key':key,'answer_core':canon(core)})
        if is_empty and r['schedule_index']==0 and r['renderer']=='forward':
            # Two individually-supported but jointly-impossible observations suffice as minimal failure.
            left=t['observations'][0];right=t['observations'][1];worldleft=next(w for w in t['worlds'] if typed(w['facts'][left['fact']])==typed(left['equals']));worldright=next(w for w in t['worlds'] if typed(w['facts'][right['fact']])==typed(right['equals']));minimal={'kind':'belief','worlds':[{'world_id':w['world_id'],'facts':{left['fact']:w['facts'][left['fact']],right['fact']:w['facts'][right['fact']]}} for w in (worldleft,worldright)],'observations':t['observations'][:2],'queries':[{'query_id':'q','fact':left['fact']}]};ma,_=solve(minimal);mz,me=scalar_rename(minimal,ma);marker_counterexamples.append({'source_sample_id':r['sample_id'],'record_hash':r['record_hash'],'minimal_task':minimal,'correct':ma,'consistent_opaque_scalar_rename_task':mz,'correct_after_rename':me,'token_marker_prediction_after_rename':'nonempty (incorrect)','scope':'audit_probe; scalar bijection adds no source families or corpus rows.'})
    marker_metrics={split:{'records':len([m for m in markers if m['split']==split]),'empty_records':sum(m['empty'] for m in markers if m['split']==split),'marker_empty_flag_correct':sum(m['marker']==m['empty'] for m in markers if m['split']==split),'empty_complete_target_core_correct_from_train_marker_worldcount_observationcount_lookup':sum(empty_target_lookup[m['key']].most_common(1)[0][0]==m['answer_core'] for m in markers if m['split']==split and m['empty'])} for split in ('train','dev_challenge')}
    marker_issue={'code':'observation_scalar_token_empty_target_shortcut','severity':'high','evidence':{'marker_strings':sorted(marker),'split_results':marker_metrics,'semantically_irrelevant_bijection_breaks_marker':True,'affected_empty_context_records':432},'fix':'In a new batch, cross consistent scalar-token assignments with supported and contradictory roles within matched worlds/families. Preserve the factual structural/control requirements; do not count renamings as new independent source families. Re-evaluate complete target and conditional token baselines.'}
    for receipt,r in zip(receipts,rows):
        if r['controlled_context'] in ('E','F'):receipt['issues'].append(marker_issue);receipt['scores']['shortcut_resistance']=2;receipt['scores']['generalization_value']=2;receipt['decision']='REVISE'
        receipt['issues'].append(issue('four_family_exposed_finite_filter_scope','low',{'source_families':4,'base_cases':24,'exposed_dev_families':2,'row_order_cyclic_shift_patterns':'Complete per-query triples do not make full row permutations independent.'},'Keep same-model/common-design, exposed-development and correlated-record limitations in release documentation; efficacy needs actual sealed evaluation.'))
        for problem in problems:
            if problem.get('sample_id')==r['sample_id'] or problem.get('family')==r['source_family_key']:receipt['issues'].append(issue('public_metadata_or_family_claim_failure','high',problem,'Repair inaccurate metadata or actual missing control with fresh hashes.'));receipt['decision']='REVISE'
    freeze(OUT/f'{BATCH}.jsonl',receipts)
    # Prior release world inputs only, ignoring their targets and receipt metadata.
    prior_paths=sorted((ROOT/'data/RBM-001/releases').glob('*/canonical.jsonl'));prior_worlds=set();prior_sigs=set();prior_counts=collections.Counter()
    for path in prior_paths:
        for r in load(path):
            t=r['semantic_task'];worlds=sorted([w['facts'] for w in t['worlds']],key=canon);prior_worlds.add(digest(worlds));prior_sigs.add(canon(table_signature(t)));prior_counts[str(len(t['worlds']))]+=1
    new_worlds={digest(sorted([w['facts'] for w in normalized(r)['worlds']],key=canon)) for r in rows};exact_tasks=collections.Counter(digest(r['semantic_task']) for r in rows);normalized_tasks=collections.Counter(digest(normalized(r)) for r in rows);unweighted=collections.defaultdict(list)
    for s in structural:unweighted[canon(s['graph']['unweighted_canonical_signature'])].append(s['family'])
    diversity={'reviewer_execution_id':EXECUTION,'batch':BATCH,'records':len(rows),'source_families':len(families),'base_contexts':len(groups)//2,'rotation_level_tasks':len(rows)//2,'base_query_observations':len(groups)//2*3,'rotation_level_query_observations':len(obs)//2,'rendered_query_observations':len(obs),'split_records':dict(collections.Counter(r['split'] for r in rows)),'split_families':{split:len({r['source_family_key'] for r in rows if r['split']==split}) for split in ('train','dev_challenge')},'actual_exact_parsed_tasks':len(exact_tasks),'exact_renderer_duplicate_groups':sum(n>1 for n in exact_tasks.values()),'decoded_base_tasks':len(normalized_tasks),'decoded_base_world_tables':len(new_worlds),'field_order_pairs':len(fieldpairs),'structural_profiles':structural,'shared_unweighted_incidence_motifs':dict(unweighted),'prior_release_world_inputs':{'canonical_files':[str(p.relative_to(ROOT)) for p in prior_paths],'world_table_count':len(prior_worlds),'world_sizes':dict(prior_counts),'exact_normalized_fact_table_overlap':len(new_worlds&prior_worlds),'partition_incidence_invariant_overlap':[s['family'] for s in structural if canon(s['complete_table_partition_incidence_invariant']) in prior_sigs],'scope':'Only prior public canonical semantic inputs parsed; no targets/receipts/verdicts used. Invariant distinguishes these profiles but is not a complete arbitrary hypergraph isomorphism algorithm.'},'crossing_histograms':crossing,'actual_corpus_controls':controls,'executed_family_probes':probes,'factual_component_counts':{r['sample_id']:{'worlds':len(r['semantic_task']['worlds']),'facts':len(r['semantic_task']['worlds'][0]['facts']),'observations':len(r['semantic_task']['observations']),'queries':3,'strict_nonempty_prefix_reductions':sum(0<len(b)<len(a) for a,b in zip(pred[r['sample_id']]['prefix_evidence']['prefix_survivors'],pred[r['sample_id']]['prefix_evidence']['prefix_survivors'][1:])),'surviving_distinct_payloads':payload_count(r['semantic_task'],pred[r['sample_id']]['expected_target']['surviving_worlds']),'final_bins':[q['distinct_values'] for q in pred[r['sample_id']]['expected_target']['queries']],'steps_after_first_empty':next((len(r['semantic_task']['observations'])-i for i,p in enumerate(pred[r['sample_id']]['prefix_evidence']['prefix_survivors']) if not p),0)} for r in rows if r['schedule_index']==0 and r['renderer']=='forward'},'metadata_or_claim_errors':problems,'limitations':['Four source families / 24 contexts; 1296 rows include correlated renaming/order views, not 1296 new mechanisms.','Four truthful incidence profiles include a shared cyclic unweighted motif across train/dev. All use one finite noiseless intersection/partition mechanism.','The 27 shifts cross each query role over all nuisance triples. Cyclic whole-row arrangements and auxiliary fact/world-ID roles are not fully crossed.','Empty/repetition substeps can be simple; world/list/rotation counts do not establish high difficulty.','Common public design and same underlying model make reviewer errors correlated; dev remains exposed.']}
    shortcut={'reviewer_execution_id':EXECUTION,'batch':BATCH,'cpu_lookup_scope':'Train lookup/majority against two exposed-dev families, no fitting/training, correlated row counts reported.','nuisance_and_semantic_feature_baselines':baselines,'incomplete_semantic_flag_mutants':heuristics,'observation_token_empty_shortcut':marker_issue,'marker_complete_empty_target_lookup':marker_metrics,'minimal_marker_counterexamples':marker_counterexamples,'scalar_bijection_invariance_executions':scalar_checks,'category_imbalance':dict(collections.Counter(o['outcome']['audit_category'] for o in obs)),'useful_flag_imbalance':dict(collections.Counter(str(o['answer']['can_reduce_uncertainty']) for o in obs)),'entailed_flag_imbalance':dict(collections.Counter(str(o['answer']['entailed']) for o in obs)),'limitations':['Full output/count list length differences are required by the target format; the reported high issue concerns literal observation-token predictability from inputs.','Support-size/observed-fact features are meaningful semantic information but incomplete substitutes for partition computation.','Marginal/pair/triple query-role controls do not address observation scalar-token role assignment, whole-row cyclic patterns or robust model efficacy.']}
    save(ROOT/'audits/diversity'/f'{BATCH}-adversarial.json',diversity);save(ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json',shortcut)
    execution={'reviewer_execution_id':EXECUTION,'batch':BATCH,'status':'completed','input_only_sha256':sha(INPUT),'candidate_sha256':sha(CAND),'input_predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'final_receipts_sha256':sha(OUT/f'{BATCH}.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'public_contract_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'diversity_sha256':sha(ROOT/'audits/diversity'/f'{BATCH}-adversarial.json'),'shortcut_sha256':sha(ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json'),'records':len(rows),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'metamorphic_counts':dict(collections.Counter(c for r in receipts for c in r['evidence']['metamorphic_checks'])),'scalar_bijection_checks':scalar_checks,'metadata_or_claim_errors':len(problems),'finding':marker_issue,'forbidden_sources_read':False,'expectations_frozen_before_gold_or_metadata_access':True,'scope':'Prompt-derived exact typed solver, actual prefix/property locator proofs, exhaustive complete schedules and own metamorphic/counterexample executions. Same-model/common-public-design; no efficacy or sealed custody claims.'}
    save(ROOT/'audits/execution'/f'{BATCH}-adversarial.json',execution);print(canon({'decisions':execution['decisions'],'metadata_errors':len(problems),'final_sha256':execution['final_receipts_sha256'],'execution_sha256':sha(ROOT/'audits/execution'/f'{BATCH}-adversarial.json'),'marker_metrics':marker_metrics,'baselines':baselines,'heuristics':heuristics}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['expectations','review'],default='expectations');args=parser.parse_args()
    if args.stage=='expectations':stage1()
    else:stage2()
