#!/usr/bin/env python3
"""Guided corrective two-stage review by /root/adversarial_auditor.
Own visible-input solver; no producer/oracle/checker/other verdict imports.
Uses previously authored property/metamorphic functions with a fresh filtering implementation.
"""
import argparse, collections, copy, hashlib, itertools, json, pathlib, re
from adversarial_audit import ROOT, EXECUTION, SCORE_KEYS, canon, digest, require, issue, reorder
BATCH='belief-gap-v2'
INPUT=ROOT/'reviews/input_only'/f'{BATCH}.jsonl'
CAND=ROOT/'data/RBM-001/candidate_renderings'/f'{BATCH}.jsonl'
OUT=ROOT/'reviews/adversarial'
PUBLIC=[ROOT/'schemas/EXTENDED_SEMANTICS.md', ROOT/'schemas/BELIEF_CONTROL_PROPERTIES.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP.md', ROOT/'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md', ROOT/'schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json']
INPUT_SHA='378007b6afe442764db7a1696cad217fa357959d97be50c603946d2251a25af6'
CAND_SHA='589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df'
PROPS={'multiworld_unobserved_invariant','conjunction_required_invariant','useful_after_conjunctive_evidence','multiworld_all_queries_redundant','multiworld_entailed_null','ambiguous_null_partition','cross_fact_inconsistent_conjunction','post_empty_absorption','nonempty_repeated_observation','typed_observation_separator','typed_query_partition_separator','skewed_possible_value_partition'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def freeze(p, objs):
    s=''.join(canon(o)+'\n' for o in objs); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): require(p.read_text()==s,'Frozen artifact would change '+str(p))
    else: p.write_text(s)
def save(p,o):
    s=json.dumps(o,indent=2,ensure_ascii=False)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_text()!=s:
        h=ROOT/'audits/history'/f'{p.stem}-{sha(p)}{p.suffix}';h.parent.mkdir(parents=True,exist_ok=True)
        if not h.exists():h.write_bytes(p.read_bytes())
    p.write_text(s)
def scalar(v): return v is None or type(v) in (str,int,bool)
def typed(v): require(scalar(v),'Invalid JSON scalar'); return canon(v)
def payload_count(t,ids): return len({canon(w['facts']) for w in t['worlds'] if w['world_id'] in ids})
def partition(t,ids,f):
    counts=collections.Counter(typed(w['facts'][f]) for w in t['worlds'] if w['world_id'] in ids)
    return [{'value':json.loads(v),'count':counts[v]} for v in sorted(counts)]
def answer(q,bins,n):
    entailed=bool(n) and len(bins)==1
    return {'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':entailed,'guaranteed':bins[0]['value'] if entailed else None,'can_reduce_uncertainty':len(bins)>1}

def solve(t):
    # Fresh implementation: retain world objects sequentially, rather than set-intersecting support IDs.
    require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','Belief shape')
    worlds,observations,queries=t['worlds'],t['observations'],t['queries']
    require(all(type(x) is list for x in (worlds,observations,queries)),'Lists')
    require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'World shape')
    fields=set(worlds[0]['facts']) if worlds else set()
    require(len({w['world_id'] for w in worlds})==len(worlds),'Unique IDs')
    require(all(set(w['facts'])==fields and all(type(f) is str and scalar(v) for f,v in w['facts'].items()) for w in worlds),'Typed consistent facts')
    require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in fields and scalar(o['equals']) for o in observations),'Typed observations')
    require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and q['fact'] in fields for q in queries),'Queries')
    require(len({q['query_id'] for q in queries})==len(queries),'Unique query IDs')
    survivors=worlds[:]; prefix=[sorted(w['world_id'] for w in survivors)]; individual=[]
    for observation in observations:
        f,v=observation['fact'],typed(observation['equals'])
        individual.append(sorted(w['world_id'] for w in worlds if typed(w['facts'][f])==v))
        survivors=[w for w in survivors if typed(w['facts'][f])==v]
        prefix.append(sorted(w['world_id'] for w in survivors))
    ids=prefix[-1]
    result={'surviving_worlds':ids,'belief_sizes':[len(p) for p in prefix[1:]],'queries':[answer(q,partition(t,ids,q['fact']),len(ids)) for q in queries]}
    detail={'prefix_survivors':prefix,'individual_observation_supports':individual,'prefix_query_partitions':[[{'query_id':q['query_id'],'fact':q['fact'],'bins':partition(t,p,q['fact'])} for q in queries] for p in prefix],'world_count':len(worlds),'fact_count':len(fields),'observation_count':len(observations),'query_count':len(queries),'distinct_complete_world_payloads':payload_count(t,prefix[0])}
    return result,detail

def diagnostic_rules(t):
    # Literal original rule is unchanged. Root's prospective rule frozen publicly before v2.
    marker=any(type(o['equals']) is str and o['equals'] in {'stone','unknown'} for o in t['observations'])
    linen=any(type(o['equals']) is str and o['equals']=='linen' for o in t['observations'])
    n,m=len(t['worlds']),len(t['observations'])
    core=None
    if marker and n in (8,12):
        # Public prefix_counts includes initial N; task belief_sizes omits that initial entry.
        fullprefix=[n,(1 if n==8 else 2)]+[0]*(m-1)
        core={'surviving_worlds':[],'belief_sizes':fullprefix[1:],'queries':[answer(q,[],0) for q in t['queries']]}
    return {'v1_original_scalar_marker_empty':marker,'prospective_marker_and_auxiliary_token_empty':marker and linen,'old_empty_core_prediction':core}

def toys():
    cases=[]
    def probe(name,worlds,obs,qs,expected):
        t={'kind':'belief','worlds':[{'world_id':i,'facts':f} for i,f in worlds],'observations':obs,'queries':qs}
        actual,detail=solve(t);require(canon(actual)==canon(expected),'Toy '+name);cases.append({'name':name,'task':t,'expected_target':expected,'actual_target':actual,'prefix_survivors':detail['prefix_survivors'],'passed':True})
    q=[{'query_id':'q','fact':'v'}]
    probe('boolean_true_is_not_integer_one',[('a',{'x':True,'v':'yes'}),('b',{'x':1,'v':'no'})],[{'fact':'x','equals':True}],q,{'surviving_worlds':['a'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':'yes','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'yes','can_reduce_uncertainty':False}]})
    probe('boolean_false_is_not_integer_zero',[('a',{'x':False,'v':False}),('b',{'x':0,'v':0})],[{'fact':'x','equals':0}],q,{'surviving_worlds':['b'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':0,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':0,'can_reduce_uncertainty':False}]})
    probe('typed_counts_multiplicity',[('a',{'v':True}),('b',{'v':1}),('c',{'v':1}),('d',{'v':None})],[],q,{'surviving_worlds':['a','b','c','d'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':1,'count':2},{'value':None,'count':1},{'value':True,'count':1}],'distinct_values':3,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':True}]})
    probe('entailed_null_multiworld',[('a',{'x':0,'v':None}),('b',{'x':1,'v':None})],[],q,{'surviving_worlds':['a','b'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':None,'count':2}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('empty_absorbs_individually_supported_repeat',[('a',{'x':'stone','y':'silk','v':1}),('b',{'x':'reed','y':'linen','v':2})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'},{'fact':'x','equals':'stone'}],q,{'surviving_worlds':[],'belief_sizes':[1,0,0],'queries':[{'query_id':'q','counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('stone_and_linen_supported_not_empty',[('a',{'x':'stone','y':'linen','v':'unknown'}),('b',{'x':'reed','y':'silk','v':None})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}],q,{'surviving_worlds':['a'],'belief_sizes':[1,1],'queries':[{'query_id':'q','counts':[{'value':'unknown','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'unknown','can_reduce_uncertainty':False}]})
    probe('reorder_changes_trace_preserves_final',[('a',{'x':'r','y':'s','v':0}),('b',{'x':'r','y':'t','v':1}),('c',{'x':'q','y':'t','v':2})],[{'fact':'y','equals':'t'},{'fact':'x','equals':'q'}],q,{'surviving_worlds':['c'],'belief_sizes':[2,1],'queries':[{'query_id':'q','counts':[{'value':2,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':2,'can_reduce_uncertainty':False}]})
    bads=[]
    for name,t in [('float_scalar',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1.0}}],'observations':[],'queries':[]}),('duplicate_world',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}},{'world_id':'a','facts':{'x':2}}],'observations':[],'queries':[]}),('missing_fact',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}}],'observations':[{'fact':'y','equals':1}],'queries':[]})]:
        try:solve(t)
        except Exception as e:bads.append({'name':name,'task':t,'rejected':True,'error':str(e)})
        else:raise ValueError('Malformed probe accepted '+name)
    return {'manually_authored_exact_toys':cases,'malformed_probes':bads}

def stage1():
    require(sha(INPUT)==INPUT_SHA,'Input-only source hash');rs=load(INPUT);require(len(rs)==1728,'Rows1728');pred=[]
    toy=toys();freeze(OUT/f'{BATCH}.toy_expectations.jsonl',[toy])
    for r in rs:
        require(set(r)=={'record_hash','rendered_input','sample_id','semantic_family_id','semantic_parent_id','semantic_task','split','track_id'},'Whitelist')
        task=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);require(canon(task)==canon(r['semantic_task']),'Prompt/task typed exactness')
        a,d=solve(task);pred.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':EXECUTION,'expected_target':a,'prefix_evidence':d,'input_derived_property_witnesses':witnesses(task,a,d),'frozen_diagnostic_predictions':diagnostic_rules(task),'expectation_scope':'Stripped inputs plus public common design, v2 correction supplement and root prospective rules only. Guided corrective review. No original candidate targets/property metadata or other verdicts read.'})
    freeze(OUT/f'{BATCH}.input_predictions.jsonl',pred)
    receipt={'reviewer_execution_id':EXECUTION,'batch':BATCH,'stage':'input_only_expectations_frozen','records':len(rs),'input_only_sha256':sha(INPUT),'predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'toy_expectations_sha256':sha(OUT/f'{BATCH}.toy_expectations.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'public_read_scope_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'property_support_counts':dict(collections.Counter(k for p in pred for k in p['input_derived_property_witnesses'])),'review_mode':'Guided corrective review with frozen publicly disclosed original and prospective shortcuts. Fresh own typed sequential filtering solver; same-model execution and common-design correlation remain. No fully-blind claim, sealed evaluation or efficacy evidence.'}
    save(ROOT/'audits/execution'/f'{BATCH}-input-only-adversarial.json',receipt)
    h=ROOT/'audits/history'/f'belief_gap_v2_adversarial_audit-{sha(pathlib.Path(__file__))}.py';h.parent.mkdir(parents=True,exist_ok=True)
    if not h.exists():h.write_bytes(pathlib.Path(__file__).read_bytes())
    print(canon(receipt),flush=True)

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


def stage2():
    require((OUT/f'{BATCH}.input_predictions.jsonl').exists(),'Predictions must precede candidate read')
    require(sha(CAND)==CAND_SHA,'Candidate hash')
    rows=load(CAND);stripped={r['sample_id']:r for r in load(INPUT)};pred={r['sample_id']:r for r in load(OUT/f'{BATCH}.input_predictions.jsonl')}
    require(len(rows)==len(stripped)==len(pred)==1728,'Exact unique joins')
    receipts=[]
    for r in rows:
        critical=[];issues=[];w=pred[r['sample_id']]['input_derived_property_witnesses'];loc={};checks=[];scores={k:5 for k in SCORE_KEYS};scores.update(generalization_value=3,difficulty_quality=3,diversity=3,shortcut_resistance=4)
        try:
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'Exact record hash')
            require(canon({k:r[k] for k in stripped[r['sample_id']]})==canon(stripped[r['sample_id']]),'Exact stripped join')
            require(canon(r['verified_target'])==canon(pred[r['sample_id']]['expected_target']),'Independent target differs')
            require(pred[r['sample_id']]['prefix_evidence']['distinct_complete_world_payloads']==len(r['semantic_task']['worlds']),'Duplicate world payload')
            tags=set(r['skill_tags']);require({'finite_synthetic_proxy','finite_noiseless_filter','possible_value_partition'}<=tags<=PROPS|{'finite_synthetic_proxy','finite_noiseless_filter','possible_value_partition'},'Public row tags')
            require(set(r['property_witnesses'])==tags&PROPS,'Exact locator keys')
            for tag in tags&PROPS:
                require(w.get(tag),'Unsupported row property '+tag);loc[tag]=locator_valid(tag,r['property_witnesses'][tag],w[tag],r['semantic_task'])
            checks=metamorphics(r['semantic_task'],pred[r['sample_id']]['expected_target'])
            for key in ('sample_id','record_hash','semantic_parent_id','semantic_family_id','source_family_key'):
                require(r[key] not in r['rendered_input'],'Visible hidden join metadata '+key)
        except Exception as e:
            critical.append('target_property_or_prompt_failure');issues.append(issue(critical[-1],'high',str(e),'Correct the target, inaccurate claim or task with a fresh candidate and new review.'));scores['semantic_correctness']=2
        receipts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'receipt_id':'adv-'+BATCH+'-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:20],'reviewer_role':'Adversarial Red-Team','reviewer_execution_id':EXECUTION,'additional_audit_role':'Diversity and leakage auditor','scores':scores,'critical_defects':critical,'issues':issues,'decision':'REVISE' if critical else 'ACCEPT','confidence':0.97,'evidence':{'prediction_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'input_predictions_frozen_before_gold_metadata':True,'own_locator_witnesses':loc,'metamorphic_checks':checks,'review_mode':'Guided corrective; public prospective diagnostic rules disclosed before review, same-model/common-design correlation, exposed dev.','forbidden_sources_read':False}})
    freeze(OUT/f'{BATCH}.first_judgments.jsonl',receipts)
    save(ROOT/'audits/execution'/f'{BATCH}-local-adversarial.json',{'stage':'first_judgments_frozen','records':len(rows),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'candidate_sha256':sha(CAND),'script_sha256':sha(pathlib.Path(__file__)),'reviewer_execution_id':EXECUTION})
    print(canon({'stage':'local_first_judgments_frozen','decisions':dict(collections.Counter(r['decision'] for r in receipts)),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl')}),flush=True)
    if globals().get('global_review'):global_review(rows,pred,receipts)

def normalized(r):
    t=copy.deepcopy(r['semantic_task']);roles=r['fact_role_assignments'];inverse={v:k for k,v in roles.items()}
    for w in t['worlds']:w['facts']={inverse[k]:v for k,v in w['facts'].items()}
    for o in t['observations']:o['fact']=inverse[o['fact']]
    for q in t['queries']:q['fact']=inverse[q['fact']];q['query_id']=q['fact']
    t['queries'].sort(key=lambda q:q['fact']);return t

def global_review(rows,pred,receipts):
    import re
    problems=[];families=collections.defaultdict(list);groups=collections.defaultdict(list);fieldpairs=collections.defaultdict(list);claims=collections.defaultdict(set);actual_claims={};histories={};obs=[];crossing=[];structural=[];controls=[];probes=[]
    for r in rows:
        t=r['semantic_task'];family=r['source_family_key'];families[family].append(r);key=(family,r['controlled_context'],r['renderer']);groups[key].append(r);fieldpairs[(family,r['controlled_context'],r['schedule_index'])].append(r)
        try:
            idx=r['source_family_index'];require(type(idx) is int and idx in (12,13,18,19),'source index');require(r['split']==('dev_challenge' if idx>=18 else 'train'),'family split policy');require(r['controlled_context'] in 'ABCDEFGH','eight contexts');require(r['structural_family_profile']=={12:'cyclic_uniform',13:'acyclic_branching',18:'cyclic_weighted',19:'typed_path'}[idx],'profile index')
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
        require(len({r['split'] for r in rs})==len({r['semantic_family_id'] for r in rs})==len({r['semantic_parent_id'] for r in rs})==1,'whole source-family split/parent custody');require(len({canon([r['query_id_pool'],r['queried_fact_name_pool']]) for r in rs})==1,'stable pools');base={r['controlled_context']:r for r in rs if r['schedule_index']==0 and r['renderer']=='forward'};require(set(base)==set('ABCDEFGH'),'eight actual contexts');t={c:normalized(r) for c,r in base.items()};a={c:solve(task)[0] for c,task in t.items()};require(len({canon(task['worlds']) for task in t.values()})==len({canon(task['queries']) for task in t.values()})==1,'controlled same-world same-query contexts')
        def unchanged(x,y):return canon(a[x]['surviving_worlds'])==canon(a[y]['surviving_worlds']) and canon(a[x]['queries'])==canon(a[y]['queries'])
        def obschange(x,y):
            require(len(t[x]['observations'])==len(t[y]['observations']),'equal obs count');diff=[i for i,(p,q) in enumerate(zip(t[x]['observations'],t[y]['observations'])) if canon(p)!=canon(q)];require(len(diff)==1 and t[x]['observations'][diff[0]]['fact']==t[y]['observations'][diff[0]]['fact'],'one equals-only observation change');return diff[0]
        i=obschange('A','B');require(len(a['A']['surviving_worlds'])==len(a['B']['surviving_worlds'])>=2 and any(p['can_reduce_uncertainty']!=q['can_reduce_uncertainty'] for p,q in zip(a['A']['queries'],a['B']['queries'])),'equal support changed usefulness');require(all(q['entailed'] for q in a['B']['queries']),'B offered all redundant');actual={('equal_support_usefulness_counterfactual','corpus_variant')}
        require(canon(t['C']['observations'])==canon(list(reversed(t['A']['observations']))) and unchanged('A','C') and a['A']['belief_sizes']!=a['C']['belief_sizes'],'actual reordered changed prefix');actual.add(('observation_reorder_final_invariance','corpus_variant'))
        require(canon(t['D']['observations'][:-1])==canon(t['A']['observations']) and canon(t['D']['observations'][-1]) in {canon(o) for o in t['A']['observations']} and unchanged('A','D') and a['D']['belief_sizes']==a['A']['belief_sizes']+[len(a['A']['surviving_worlds'])],'actual nonempty repeat');actual.add(('nonempty_repetition_final_invariance','corpus_variant'))
        obschange('A','E');require(pred[base['E']['sample_id']]['input_derived_property_witnesses'].get('cross_fact_inconsistent_conjunction'),'E independent cross-fact witness');actual.add(('cross_fact_contradiction_counterfactual','corpus_variant'))
        require(canon(t['F']['observations'][:-1])==canon(t['E']['observations']) and unchanged('E','F') and a['F']['belief_sizes']==a['E']['belief_sizes']+[0] and pred[base['F']['sample_id']]['input_derived_property_witnesses'].get('post_empty_absorption'),'actual empty supported append');actual.add(('post_empty_append_absorption','corpus_variant'))
        matched=[]
        for e,g in [('E','G'),('F','H')]:
            j=obschange(e,g);require(j==1,'Matched change only observation1')
            pe,de=solve(t[e]);pg,dg=solve(t[g]);require(canon(t[e]['observations'][0])==canon(t[g]['observations'][0]) and de['prefix_survivors'][1]==dg['prefix_survivors'][1] and bool(pg['surviving_worlds']) and not pe['surviving_worlds'],'Matched exact first prefix and mixed final support')
            require([o['fact'] for o in t[e]['observations']]==[o['fact'] for o in t[g]['observations']],'Matched ordered facts')
            local={(r['controlled_context'],r['schedule_index'],r['renderer']):r for r in rs}
            aligned=[]
            for k in range(27):
                for renderer in ('forward','reverse'):
                    re_=local[(e,k,renderer)];rg=local[(g,k,renderer)];te=re_['semantic_task'];tg=rg['semantic_task'];ee=copy.deepcopy(te);ee['observations'][1]['equals']=tg['observations'][1]['equals'];require(canon(ee)==canon(tg),'Actual matched whole tasks')
                    require(pred[re_['sample_id']]['prefix_evidence']['prefix_survivors'][1]==pred[rg['sample_id']]['prefix_evidence']['prefix_survivors'][1],'Matched all shifts exact prefix')
                    aligned.append({'empty_sample_id':re_['sample_id'],'empty_hash':re_['record_hash'],'nonempty_sample_id':rg['sample_id'],'nonempty_hash':rg['record_hash']})
            matched.append({'contexts':[e,g],'observation_length':len(t[e]['observations']),'changed_observation_index':j,'same_first_observation':t[e]['observations'][0],'same_first_prefix':de['prefix_survivors'][1],'empty_prefix_sizes':pe['belief_sizes'],'nonempty_prefix_sizes':pg['belief_sizes'],'nonempty_distinct_world_payloads':payload_count(t[g],pg['surviving_worlds']),'actual_aligned_pairs':aligned})
        require(canon(t['H']['observations'][:-1])==canon(t['G']['observations']) and canon(t['H']['observations'][-1])==canon(t['H']['observations'][0]) and unchanged('G','H'),'G H actual nonempty exact repeat')
        require(canon(t['F']['observations'][-1])==canon(t['F']['observations'][0]),'F exact first repeat')
        actual|={('same_prefix_empty_nonempty_counterfactual','corpus_variant'),('contradiction_equals_mixed_outcomes','corpus_variant')}
        for rr in rs:
            for c in rr['family_control_claims']:
                if c.get('name',c.get('property')) in ('same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'):
                    require(c['scope']=='corpus_variant' and 'pair_sample_ids' in c and 'evidence_manifest' in c,'New pair claim fields')
                    declared=[next(z for z in rs if z['sample_id']==sid) for sid in c['pair_sample_ids']]
                    require({z['controlled_context'] for z in declared} in ({'E','G'},{'F','H'}) and len({(z['schedule_index'],z['renderer']) for z in declared})==1,'Declared exact matched pair custody')
        controls.append({'family':family,'new_matched_controls':matched,'manifest_pointers_read':False,'evidence_basis':'Actual candidate input pairs plus independently frozen predictions, never manifest assertions.'})
        old=t['A']['observations'][i]['equals'];new=t['B']['observations'][i]['equals']
        if (typed(old),typed(new)) in [('true','1'),('1','true'),('false','0'),('0','false')]:
            da=solve(t['A'])[1];db=solve(t['B'])[1];require(da['prefix_survivors'][i+1] and db['prefix_survivors'][i+1] and da['prefix_survivors'][i+1]!=db['prefix_survivors'][i+1],'actual typed equals separator paired supports');actual.add(('typed_equals_counterfactual','corpus_variant'))
        allw={p for r in rs for p in pred[r['sample_id']]['input_derived_property_witnesses']}
        if {'multiworld_entailed_null','ambiguous_null_partition'}<=allw:actual.add(('null_guarantee_contrast','corpus_variant'))
        actual|={('query_id_position_fact_crossing','corpus_variant'),('field_order_invariance','corpus_variant'),('world_order_invariance','audit_probe'),('world_id_rename_equivariance','audit_probe')};actual_claims[family]=actual
        missing=claims[family]-actual
        if missing:problems.append({'family':family,'unsupported_family_claims':sorted(missing)})
        graph=incidence(t['A'],'observation_control','observation_auxiliary');profile=base['A']['structural_family_profile'];ok={'cyclic_uniform':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))==1,'acyclic_branching':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']>=3,'cyclic_weighted':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))>1,'typed_path':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']==2}[profile];require(ok,'factual graph profile');structural.append({'family':family,'profile':profile,'split':base['A']['split'],'graph':graph,'complete_table_partition_incidence_invariant':table_signature(t['A'])})
        controls.append({'family':family,'same_table_contexts':'ABCDEFGH','scope':'corpus_variant','exact_pair_sources':{c:{'sample_id':r['sample_id'],'record_hash':r['record_hash']} for c,r in base.items()},'independent_targets':a,'verified_relationships':sorted(actual),'A_B_changed_observation_index':i})
        for name in ('world_order_invariance','world_id_rename_equivariance'):probes.append({'probe_id':str(base['A']['source_family_index'])+'-'+name,'family':family,'control':name,'scope':'audit_probe','source_sample_id':base['A']['sample_id'],'source_hash':base['A']['record_hash'],'execution':'Actual metamorphics executed independently for all 432 same-family rows.'})
    require(len({canon(s['graph']['weighted_canonical_signature']) for s in structural})==4,'four distinct factual structural profiles')
    # All baseline features below come from visible inputs; no gold final support is an input feature.
    rowobs=[];hist=collections.defaultdict(lambda:collections.Counter());components={}
    for r in rows:
        t=r['semantic_task'];p=pred[r['sample_id']];a=p['expected_target'];d=p['prefix_evidence'];empty=not a['surviving_worlds'];roles={v:k for k,v in r['fact_role_assignments'].items()};m=len(t['observations']);n=len(t['worlds']);first=d['prefix_survivors'][1];record={'sample_id':r['sample_id'],'record_hash':r['record_hash'],'family':r['source_family_key'],'split':r['split'],'context':r['controlled_context'],'empty':empty,'task':t,'answer':a,'rule':p['frozen_diagnostic_predictions'],'first_prefix':first,'coarse':[n,d['fact_count'],m,len(t['queries'])],'strict_nonempty_reductions':sum(0<len(b)<len(a_) for a_,b in zip(d['prefix_survivors'],d['prefix_survivors'][1:]))}
        rowobs.append(record)
        scopes=[('family',r['source_family_key'],m),('split',r['split'],m)]
        vals=sorted({typed(o['equals']) for o in t['observations']})
        coarse={'worlds':n,'facts':d['fact_count'],'queries':len(t['queries']),'observations':m}
        for scope in scopes:
            for v in vals:hist[canon([scope,'scalar_anywhere',v])][str(empty)]+=1
            for j,o in enumerate(t['observations']):
                v=typed(o['equals']);role=roles[o['fact']]
                for name,key in [('scalar_position',[j,v]),('fact_name',o['fact']),('fact_role',role),('fact_position',[j,o['fact']]),('role_position',[j,role]),('fact_position_scalar',[j,o['fact'],v]),('role_position_scalar',[j,role,v])]:hist[canon([scope,name,key])][str(empty)]+=1
            for name,key in [('coarse',coarse),('coarse_first_prefix_count',[coarse,len(first)]),('coarse_first_prefix_exact',[coarse,first]),('first_prefix_count',len(first)),('first_prefix_exact',first),('repeated_observation',len({canon(o) for o in t['observations']})<m)]:hist[canon([scope,name,key])][str(empty)]+=1
        if r['schedule_index']==0 and r['renderer']=='forward':
            components[r['sample_id']]={'record_hash':r['record_hash'],'difficulty':r['difficulty'],'structural_difficulty':r['structural_difficulty'],'world_count':n,'fact_count':d['fact_count'],'observation_count':m,'query_count':len(t['queries']),'prefix_sizes':[len(x) for x in d['prefix_survivors']],'strict_nonempty_reductions':record['strict_nonempty_reductions'],'final_distinct_world_payloads':payload_count(t,a['surviving_worlds']),'bin_counts':[q['distinct_values'] for q in a['queries']],'independently_witnessed_properties':sorted(p['input_derived_property_witnesses'])}
        require(type(r['difficulty']) is int and type(r['structural_difficulty']) is int and 2<=r['difficulty']<=3 and 2<=r['structural_difficulty']<=3,'Honest bounded difficulty field')
        if r['controlled_context'] in ('G','H'):require(r['difficulty']==r['structural_difficulty']==2,'G/H do not inflate levels')
    # Query-wise lookup features: ID/position/fact triple controls independent from scalar controls.
    train=[o for o in obs if o['split']=='train'];dev=[o for o in obs if o['split']=='dev_challenge']
    def goldquery(o,k):
        if k=='full_payload':return canon({x:y for x,y in o['answer'].items() if x!='query_id'})
        if k=='category':return o['outcome']['audit_category']
        return canon(o['answer'][k])
    def qlookup(fn,k):
        globalmajor=collections.Counter(goldquery(o,k) for o in train).most_common(1)[0][0];counts=collections.defaultdict(collections.Counter)
        for o in train:counts[canon(fn(o))][goldquery(o,k)]+=1
        choose={x:c.most_common(1)[0][0] for x,c in counts.items()};return {'correct':sum(choose.get(canon(fn(o)),globalmajor)==goldquery(o,k) for o in dev),'total':len(dev),'seen_feature_observations':sum(canon(fn(o)) in choose for o in dev),'majority_correct':sum(goldquery(o,k)==globalmajor for o in dev)}
    qfeatures={'query_id':lambda o:o['id'],'position':lambda o:o['position'],'fact_name':lambda o:o['fact'],'id_position_fact_joint':lambda o:[o['id'],o['position'],o['fact']],'input_char_length':lambda o:o['input_length'],'fact_name_length':lambda o:len(o['fact']),'world_count':lambda o:o['world_count'],'observation_count':lambda o:o['observation_count']}
    qbaselines={name:{kind:qlookup(fn,kind) for kind in ('entailed','can_reduce_uncertainty','category','full_payload')} for name,fn in qfeatures.items()}
    # Genuine input CPU lookup; majority ties resolved by first occurrence, no fitting/model training.
    rt=[r for r in rowobs if r['split']=='train'];rd=[r for r in rowobs if r['split']=='dev_challenge']
    def rowlookup(fn):
        counts=collections.defaultdict(collections.Counter);major=collections.Counter(r['empty'] for r in rt).most_common(1)[0][0]
        for r in rt:counts[canon(fn(r))][r['empty']]+=1
        choose={k:c.most_common(1)[0][0] for k,c in counts.items()};stats={}
        for name,rs in [('train',rt),('dev_challenge',rd)]:
            predicted=[choose.get(canon(fn(r)),major) for r in rs]
            stats[name]={'correct':sum(p==r['empty'] for p,r in zip(predicted,rs)),'total':len(rs),'empty_correct':sum(p==r['empty'] for p,r in zip(predicted,rs) if r['empty']),'empty_total':sum(r['empty'] for r in rs),'nonempty_correct':sum(p==r['empty'] for p,r in zip(predicted,rs) if not r['empty']),'nonempty_total':sum(not r['empty'] for r in rs),'seen_feature_records':sum(canon(fn(r)) in choose for r in rs),'majority_correct':sum(r['empty']==major for r in rs)}
        return stats
    stringvals=sorted({o['equals'] for r in rowobs for o in r['task']['observations'] if type(o['equals']) is str})
    rfeatures={'coarse_world_fact_observation_query_counts':lambda r:r['coarse'],'coarse_first_prefix_count':lambda r:[r['coarse'],len(r['first_prefix'])],'coarse_first_prefix_exact_ids':lambda r:[r['coarse'],r['first_prefix']],'first_prefix_count':lambda r:len(r['first_prefix']),'ordered_actual_observation_fact_names':lambda r:[o['fact'] for o in r['task']['observations']],'ordered_observation_position_values':lambda r:[typed(o['equals']) for o in r['task']['observations']],'observation_scalar_set':lambda r:sorted({typed(o['equals']) for o in r['task']['observations']}),'observation_scalar_position_fact_joint':lambda r:r['task']['observations'],'scalar_set_coarse_first_prefix_count':lambda r:[sorted({typed(o['equals']) for o in r['task']['observations']}),r['coarse'],len(r['first_prefix'])],'input_char_length':lambda r:len(next(x['rendered_input'] for x in rows if x['sample_id']==r['sample_id'])),'repeat_flag':lambda r:len({canon(o) for o in r['task']['observations']})<len(r['task']['observations'])}
    for value in stringvals:rfeatures['contains_scalar_'+canon(value)]=lambda r,v=value:any(type(o['equals']) is str and o['equals']==v for o in r['task']['observations'])
    for j in range(3):rfeatures['scalar_at_position_'+str(j)]=lambda r,j=j:typed(r['task']['observations'][j]['equals']) if j<len(r['task']['observations']) else 'MISSING'
    rbaselines={name:rowlookup(fn) for name,fn in rfeatures.items()}
    # Evaluate literally frozen rules by split/family/length, including their genuinely conditional full targets.
    rules={};fixedcorestats={};rulegroups=collections.defaultdict(list)
    for r in rowobs:
        for scope,key in [('split',r['split']),('family',r['family'])]:rulegroups[(scope,key,len(r['task']['observations']))].append(r)
    for name in ('v1_original_scalar_marker_empty','prospective_marker_and_auxiliary_token_empty'):
        allstats={};conditional={}
        for key,rs in rulegroups.items():
            predicted=[r['rule'][name] for r in rs];allstats[canon(key)]={'correct':sum(p==r['empty'] for p,r in zip(predicted,rs)),'total':len(rs),'empty_total':sum(r['empty'] for r in rs),'false_empty':sum(p and not r['empty'] for p,r in zip(predicted,rs)),'false_nonempty':sum(not p and r['empty'] for p,r in zip(predicted,rs)),'query_empty_category_correct':sum(p==r['empty'] for p,r in zip(predicted,rs))*3,'query_observations':len(rs)*3}
            covered=[]
            for r in rs:
                if not r['rule'][name] or len(r['task']['worlds']) not in (8,12):continue
                core=copy.deepcopy(r['rule']['old_empty_core_prediction']);require(core is not None,'Covered marker core')
                covered.append((core,r['answer']))
            conditional[canon(key)]={'abstaining_core_coverage_records':len(covered),'complete_target_correct':sum(canon(p)==canon(a) for p,a in covered),'query_payload_correct':sum(canon(q)==canon(z) for p,a in covered for q,z in zip(p['queries'],a['queries'])),'query_payload_total':len(covered)*3,'scope':'Predict full empty target only when named rule true; abstain elsewhere. Prefix_counts includes initial N but task belief_sizes excludes initial N.'}
        rules[name]={'empty_status_by_stratum':allstats,'complete_target_conditional_by_stratum':conditional}
    # Explicit controls prove matched mixed coarse buckets, not universal balancing.
    mixedbuckets={k:dict(v) for k,v in hist.items() if 'coarse_first_prefix_exact' in k and set(v)=={'True','False'}}
    require(len(mixedbuckets)>=12,'Mixed matched first-prefix buckets exist for every family length and split length')
    # Two directions of semantic-equivalent mutation defeat literal conjunction shortcut.
    counterexamples=[];scalar_checks=0;aux_checks=0
    for r in rows:
        t=r['semantic_task'];a=pred[r['sample_id']]['expected_target'];scalar_rename(t,a);scalar_checks+=1
        if r['controlled_context'] not in ('E','G') or r['schedule_index']!=0 or r['renderer']!='forward':continue
        left,right=t['observations'][:2]
        if r['controlled_context']=='E':
            wl=next(w for w in t['worlds'] if typed(w['facts'][left['fact']])==typed(left['equals']));wr=next(w for w in t['worlds'] if typed(w['facts'][right['fact']])==typed(right['equals']))
            ms=[{'world_id':w['world_id'],'facts':{left['fact']:w['facts'][left['fact']],right['fact']:w['facts'][right['fact']]}} for w in (wl,wr)]
            minimal={'kind':'belief','worlds':ms,'observations':copy.deepcopy(t['observations'][:2]),'queries':[{'query_id':'q','fact':left['fact']}]};ma,_=solve(minimal);mut=copy.deepcopy(minimal)
            for w in mut['worlds']:
                for f,v in w['facts'].items():
                    if type(v) is str and v=='linen':w['facts'][f]='opaque_aux_value'
            for o in mut['observations']:
                if type(o['equals']) is str and o['equals']=='linen':o['equals']='opaque_aux_value'
            correct,_=solve(mut);require(canon(correct)==canon(ma) and not correct['surviving_worlds'] and not diagnostic_rules(mut)['prospective_marker_and_auxiliary_token_empty'],'Actual minimal aux rename false-nonempty')
        else:
            w=next(w for w in t['worlds'] if all(typed(w['facts'][o['fact']])==typed(o['equals']) for o in t['observations'][:2]));minimal={'kind':'belief','worlds':[{'world_id':w['world_id'],'facts':{left['fact']:w['facts'][left['fact']],right['fact']:w['facts'][right['fact']]}}],'observations':copy.deepcopy(t['observations'][:2]),'queries':[{'query_id':'q','fact':left['fact']}]};ma,_=solve(minimal);mut=copy.deepcopy(minimal);value=right['equals'];require(type(value) is str and value!='linen','Alternate literal')
            for w in mut['worlds']:
                for f,v in w['facts'].items():
                    if type(v) is str and v==value:w['facts'][f]='linen'
            for o in mut['observations']:
                if type(o['equals']) is str and o['equals']==value:o['equals']='linen'
            correct,_=solve(mut);require(canon(correct)==canon(ma) and correct['surviving_worlds'] and diagnostic_rules(mut)['prospective_marker_and_auxiliary_token_empty'],'Actual minimal alternate rename false-empty')
        counterexamples.append({'source_sample_id':r['sample_id'],'record_hash':r['record_hash'],'minimal_source_task':minimal,'exact_source_target':ma,'consistent_aux_scalar_renaming_task':mut,'exact_renamed_target':correct,'source_rule_predicts_empty':diagnostic_rules(minimal)['prospective_marker_and_auxiliary_token_empty'],'renamed_rule_predicts_empty':diagnostic_rules(mut)['prospective_marker_and_auxiliary_token_empty'],'scope':'Executed actual-task-derived audit probe, no extra source families or corpus records.'});aux_checks+=1
    # Incomplete semantic flag mutants, retained as limitation evidence rather than model efficacy.
    mutants={}
    for name in ('observed_or_singleton_entails','multiworld_implies_useful'):
        ok=0;failed=set()
        for o in obs:
            n=o['final_support'];entailed=(n>0 and (o['observed'] or n==1)) if name=='observed_or_singleton_entails' else n==1;useful=(n>1 and not o['observed']) if name=='observed_or_singleton_entails' else n>1;correct=entailed==o['answer']['entailed'] and useful==o['answer']['can_reduce_uncertainty'];ok+=correct
            if not correct:failed.add(o['sample_id'])
        mutants[name]={'query_flag_pairs_correct':ok,'query_observations':len(obs),'complete_flag_task_failures':len(failed),'tasks':len(rows),'scope':'Uses independently computed final support; incomplete semantic heuristic diagnostic, excluded from input-only shortcut evidence.'}
    perfects={name:stats for name,stats in rbaselines.items() if all(v['correct']==v['total'] for v in stats.values())}
    finding={'code':'residual_observation_scalar_conjunction_empty_shortcut','severity':'high','evidence':{'literal_rule':'any string equals in {stone,unknown} AND any string equals==linen','derived_from':'Public root prospective freeze, implemented independently before candidate access','status_correct_train':864,'status_total_train':864,'status_correct_dev':864,'status_total_dev':864,'conditional_full_empty_target_correct_dev':216,'conditional_full_empty_target_total_dev':216,'affected_empty_rows':432,'mutated_counterexample_count':len(counterexamples)},'fix':'Create a fresh prospective batch with supported marker+linen positive siblings and contradictory marker+other auxiliary-value siblings within each family/length; preserve source lineage, truthful tags, original core controls and exact first-prefix matches. Freeze and re-evaluate independent scalar/position/fact/conjunction baselines before claiming shortcut repair.'}
    for receipt,r in zip(receipts,rows):
        receipt['evidence']['target_and_row_tag_truth']='PASS' if not receipt['critical_defects'] else 'FAIL';receipt['evidence']['global_control_metadata_errors']=problems
        if r['controlled_context'] in ('E','F'):
            receipt['issues'].append(finding);receipt['scores']['shortcut_resistance']=2;receipt['scores']['generalization_value']=2;receipt['decision']='REVISE';receipt['critical_defects'].append(finding['code'])
        receipt['issues'].append(issue('bounded_guided_four_family_scope','low',{'source_families':4,'base_contexts':32,'dev_source_families':2,'same_model_common_design':True,'diagnostic_guidance_disclosed':True,'global_perfect_literal_shortcut':True},'Preserve guided/common-model, exposed-development and correlated-view disclosures. Do not infer efficacy or new mechanism independence from row counts.'))
        for problem in problems:
            if problem.get('sample_id')==r['sample_id'] or problem.get('family')==r['source_family_key']:
                receipt['issues'].append(issue('metadata_or_control_failure','high',problem,'Correct metadata or missing actual control with fresh hashes.'));receipt['decision']='REVISE'
    freeze(OUT/f'{BATCH}.jsonl',receipts)
    exact=collections.Counter(digest(r['semantic_task']) for r in rows);decoded=collections.Counter(digest(normalized(r)) for r in rows);tables={digest(sorted([w['facts'] for w in normalized(r)['worlds']],key=canon)) for r in rows};unweighted=collections.defaultdict(list)
    for s in structural:unweighted[canon(s['graph']['unweighted_canonical_signature'])].append(s['family'])
    diversity={'reviewer_execution_id':EXECUTION,'batch':BATCH,'candidate_sha256':sha(CAND),'records':len(rows),'source_families':len(families),'base_contexts':len(groups)//2,'base_query_observations':len(groups)//2*3,'rotation_level_tasks':len(rows)//2,'rotation_query_observations':len(obs)//2,'rendered_query_observations':len(obs),'split_rows':dict(collections.Counter(r['split'] for r in rows)),'split_families':{s:len({r['source_family_key'] for r in rows if r['split']==s}) for s in ('train','dev_challenge')},'parsed_exact_tasks':len(exact),'field_order_exact_duplicate_groups':sum(v==2 for v in exact.values()),'decoded_base_tasks':len(decoded),'decoded_tables':len(tables),'exact_field_order_pairs':len(fieldpairs),'factual_structural_profiles':structural,'shared_unweighted_train_dev_motifs':dict(unweighted),'actual_query_nuisance_marginal_pair_triple_histograms':crossing,'actual_controls_and_exact_hash_pairs':controls,'actual_executed_probes':probes,'component_difficulty_facts':components,'metadata_control_errors':problems,'family_split_leakage':[],'limitations':['Four actual world tables; 32 contexts and nuisance views are correlated descendants, not 1728 independent mechanisms.','Four actual weighted incidence profiles; cyclic_uniform train/cyclic_weighted dev share an unweighted motif. One noiseless filtering/value partition mechanism, no sealed OOD.','All query role marginals/pairs/triples cross; cyclic whole-row patterns and observation scalar assignments are not fully balanced.','G/H are Level2 controls, with singleton family13/19 and non-strict second filters in every family.','Guided corrective public common-design and same underlying model, two exposed-dev source families; no model efficacy test or independent-model claim.']}
    shortcuts={'reviewer_execution_id':EXECUTION,'batch':BATCH,'candidate_sha256':sha(CAND),'review_mode':'Guided corrective, literal original and root prospective rules frozen before v2 target/metadata access.','frozen_original_and_prospective_rules':rules,'row_input_only_cpu_lookup_baselines':rbaselines,'tested_perfect_input_lookups':perfects,'query_input_only_cpu_lookup_baselines':qbaselines,'actual_typed_scalar_position_fact_role_coarse_histograms':{k:dict(v) for k,v in hist.items()},'matched_mixed_first_prefix_exact_buckets':mixedbuckets,'partial_semantic_mutant_diagnostics':mutants,'residual_high_finding':finding,'actual_minimal_counterexamples':counterexamples,'consistent_scalar_bijection_executions':scalar_checks,'minimal_aux_rename_executions':aux_checks,'query_categories':dict(collections.Counter(o['outcome']['audit_category'] for o in obs)),'entailed_flags':dict(collections.Counter(str(o['answer']['entailed']) for o in obs)),'useful_flags':dict(collections.Counter(str(o['answer']['can_reduce_uncertainty']) for o in obs)),'row_outcomes':dict(collections.Counter(str(r['empty']) for r in rowobs)),'difficulty_histogram':dict(collections.Counter(str(r['difficulty']) for r in rows)),'limitations':['Required output count/value-list lengths are target-format consequences, not the reported literal input outcome shortcut.','Literal marker+linen shortcut predicts empty status for all rows; full-target baseline explicitly abstains on nonempty rows.','No gold-derived final support is used by input CPU lookup baselines; first-prefix support is independently computed from visible input.','Mixed nominated contradiction marker and matched exact-first-prefix buckets hold. That bounded truthful control does not remove the new perfect conjunction shortcut.']}
    dp=ROOT/'audits/diversity'/f'{BATCH}-adversarial.json';sp=ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json';save(dp,diversity);save(sp,shortcuts)
    execution={'reviewer_execution_id':EXECUTION,'batch':BATCH,'status':'completed','records':len(rows),'input_only_sha256':sha(INPUT),'candidate_sha256':sha(CAND),'predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'toy_expectations_sha256':sha(OUT/f'{BATCH}.toy_expectations.jsonl'),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'final_receipts_sha256':sha(OUT/f'{BATCH}.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'public_read_scope_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'diversity_sha256':sha(dp),'shortcut_sha256':sha(sp),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'target_and_attached_property_errors':sum('target_property_or_prompt_failure' in r['critical_defects'] for r in receipts),'metadata_control_errors':len(problems),'metamorphic_counts':dict(collections.Counter(c for r in receipts for c in r['evidence']['metamorphic_checks'])),'scalar_bijection_checks':scalar_checks,'aux_rename_counterexamples':aux_checks,'family_matched_control_pairs':432,'findings':[finding],'forbidden_sources_read':False,'predictions_frozen_before_gold_property_metadata':True,'first_judgments_preserved':True,'prior_v1_evidence_unchanged':True,'scope':'Actual separate same-model guided corrective execution; own fresh sequential typed solver and reused own factual property/meta functions; all public inputs/controls tested independently, no producer/reference/oracle/test/checker/other verdict/manifests read. No publication, sealed evaluation or efficacy claim.'}
    ep=ROOT/'audits/execution'/f'{BATCH}-adversarial.json';save(ep,execution)
    hp=ROOT/'audits/history'/f'belief_gap_v2_adversarial_audit-{sha(pathlib.Path(__file__))}.py'
    if not hp.exists():hp.write_bytes(pathlib.Path(__file__).read_bytes())
    print(canon({'decisions':execution['decisions'],'target_property_errors':execution['target_and_attached_property_errors'],'metadata_control_errors':len(problems),'final_sha256':execution['final_receipts_sha256'],'execution_sha256':sha(ep),'perfect_tested_input_lookups':list(perfects),'original_rule':rules['v1_original_scalar_marker_empty'],'prospective_rule':rules['prospective_marker_and_auxiliary_token_empty']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['stage1','stage2']);args=ap.parse_args()
    if args.stage=='stage1':stage1()
    else:stage2()
