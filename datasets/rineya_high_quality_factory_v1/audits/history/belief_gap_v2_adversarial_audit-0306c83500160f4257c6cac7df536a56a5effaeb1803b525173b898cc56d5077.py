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

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['stage1','stage2']);args=ap.parse_args()
    if args.stage=='stage1':stage1()
    else:stage2()
