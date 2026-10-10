#!/usr/bin/env python3
"""Guided corrective two-stage review by /root/adversarial_auditor.
Own visible-input solver; no producer/oracle/checker/other verdict imports.
Uses previously authored property/metamorphic functions with a fresh bitmask implementation.
"""
import argparse, collections, copy, hashlib, itertools, json, pathlib, re
from adversarial_audit import ROOT, EXECUTION, SCORE_KEYS, canon, digest, require, issue, reorder
BATCH='belief-gap-v3'
INPUT=ROOT/'reviews/input_only'/f'{BATCH}.jsonl'
CAND=ROOT/'data/RBM-001/candidate_renderings'/f'{BATCH}.jsonl'
OUT=ROOT/'reviews/adversarial'
PUBLIC=[ROOT/'schemas/EXTENDED_SEMANTICS.md', ROOT/'schemas/BELIEF_CONTROL_PROPERTIES.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP.md', ROOT/'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md', ROOT/'schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json', ROOT/'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md', ROOT/'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3-ROOT_FREEZE.json']
INPUT_SHA='50b3f47bce29e8a639f298ece104744be1583d04c04d3041d940219d6998b694'
CAND_SHA='7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c'
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
    # Own third implementation uses typed equality keys and a prefix bitmask over world positions.
    require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','Belief shape')
    worlds,obs,queries=t['worlds'],t['observations'],t['queries']
    require(all(type(x) is list for x in (worlds,obs,queries)),'Lists')
    require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'World shape')
    fields=set(worlds[0]['facts']) if worlds else set();require(len({w['world_id'] for w in worlds})==len(worlds),'Unique worlds')
    require(all(set(w['facts'])==fields and all(type(f) is str and scalar(v) for f,v in w['facts'].items()) for w in worlds),'Typed consistent facts')
    require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in fields and scalar(o['equals']) for o in obs),'Observations')
    require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and q['fact'] in fields for q in queries),'Queries');require(len({q['query_id'] for q in queries})==len(queries),'Unique queries')
    def equal(v,u):return (type(v),v)==(type(u),u)
    mask=(1<<len(worlds))-1
    def ids(mask):return sorted(w['world_id'] for j,w in enumerate(worlds) if mask&(1<<j))
    prefixes=[ids(mask)];individual=[]
    for o in obs:
        support=sum(1<<j for j,w in enumerate(worlds) if equal(w['facts'][o['fact']],o['equals']))
        individual.append(ids(support));mask&=support;prefixes.append(ids(mask))
    def count_at(mask,f):
        bins=collections.Counter(canon(w['facts'][f]) for j,w in enumerate(worlds) if mask&(1<<j))
        return [{'value':json.loads(k),'count':bins[k]} for k in sorted(bins)]
    final=ids(mask);outputs=[]
    for q in queries:
        bins=count_at(mask,q['fact']);ent=bool(mask) and len(bins)==1
        outputs.append({'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':ent,'guaranteed':bins[0]['value'] if ent else None,'can_reduce_uncertainty':len(bins)>1})
    result={'surviving_worlds':final,'belief_sizes':[len(x) for x in prefixes[1:]],'queries':outputs}
    detail={'prefix_survivors':prefixes,'individual_observation_supports':individual,'prefix_query_partitions':[[{'query_id':q['query_id'],'fact':q['fact'],'bins':partition(t,p,q['fact'])} for q in queries] for p in prefixes],'world_count':len(worlds),'fact_count':len(fields),'observation_count':len(obs),'query_count':len(queries),'distinct_complete_world_payloads':payload_count(t,prefixes[0])}
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



def stage1():
    require(sha(INPUT)==INPUT_SHA,'Input-only source hash');rs=load(INPUT);require(len(rs)==3456,'Rows3456');pred=[]
    toy=calibration();freeze(OUT/f'{BATCH}.toy_expectations.jsonl',[toy])
    for r in rs:
        require(set(r)=={'record_hash','rendered_input','sample_id','semantic_family_id','semantic_parent_id','semantic_task','split','track_id'},'Whitelist')
        task=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);require(canon(task)==canon(r['semantic_task']),'Prompt/task typed exactness')
        a,d=solve(task);pred.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':EXECUTION,'expected_target':a,'prefix_evidence':d,'input_derived_property_witnesses':witnesses(task,a,d),'frozen_diagnostic_predictions':diagnostic_rules(task),'expectation_scope':'Stripped inputs plus public common design, v2/v3 correction supplements and root prospective rules only. Guided corrective review. No original candidate targets/property metadata or other verdicts read.'})
    freeze(OUT/f'{BATCH}.input_predictions.jsonl',pred)
    receipt={'reviewer_execution_id':EXECUTION,'batch':BATCH,'stage':'input_only_expectations_frozen','records':len(rs),'input_only_sha256':sha(INPUT),'predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'toy_expectations_sha256':sha(OUT/f'{BATCH}.toy_expectations.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'public_read_scope_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'property_support_counts':dict(collections.Counter(k for p in pred for k in p['input_derived_property_witnesses'])),'review_mode':'Guided corrective review with frozen publicly disclosed original and prospective shortcuts. Fresh own typed bitmask filtering solver; same-model execution and common-design correlation remain. No fully-blind claim, sealed evaluation or efficacy evidence.'}
    save(ROOT/'audits/execution'/f'{BATCH}-input-only-adversarial.json',receipt)
    h=ROOT/'audits/history'/f'belief_gap_v3_adversarial_audit-{sha(pathlib.Path(__file__))}.py';h.parent.mkdir(parents=True,exist_ok=True)
    if not h.exists():h.write_bytes(pathlib.Path(__file__).read_bytes())
    print(canon(receipt),flush=True)

def calibration():
    result=toys()
    task={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':'marker','y':'b','q':None}},{'world_id':'b','facts':{'x':'other','y':'g','q':'unknown'}}],'observations':[{'fact':'x','equals':'marker'},{'fact':'y','equals':'b'}],'queries':[{'query_id':'q','fact':'q'}]}
    expected={'surviving_worlds':['a'],'belief_sizes':[1,1],'queries':[{'query_id':'q','counts':[{'value':None,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]}
    actual,d=solve(task);require(canon(actual)==canon(expected),'Manual auxiliary calibration')
    transposed=copy.deepcopy(task)
    for w in transposed['worlds']:w['facts']['y']={'b':'g','g':'b'}[w['facts']['y']]
    transposed['observations'][1]['equals']='g';at,dt=solve(transposed);require(canon(at)==canon(expected) and d['prefix_survivors']==dt['prefix_survivors'],'Unqueried fact transposition exact outputs')
    unchanged_obs=copy.deepcopy(transposed);unchanged_obs['observations']=copy.deepcopy(task['observations']);empty,ed=solve(unchanged_obs);ee={'surviving_worlds':[],'belief_sizes':[1,0],'queries':[{'query_id':'q','counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False}]};require(canon(empty)==canon(ee),'Same observation changed table status')
    result['fact_local_transposition_and_same_observation_manual_probes']={'source_task':task,'source_expected':expected,'transposed_task':transposed,'transposed_expected':expected,'same_observations_changed_table_task':unchanged_obs,'same_observations_changed_table_expected':ee,'passed':True}
    return result

def stage2():
    require((OUT/f'{BATCH}.input_predictions.jsonl').exists(),'Predictions must precede candidate read')
    require(sha(CAND)==CAND_SHA,'Candidate hash')
    rows=load(CAND);stripped={r['sample_id']:r for r in load(INPUT)};pred={r['sample_id']:r for r in load(OUT/f'{BATCH}.input_predictions.jsonl')}
    require(len(rows)==len(stripped)==len(pred)==3456,'Exact unique joins')
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
        receipts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'receipt_id':'adv-'+BATCH+'-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:20],'reviewer_role':'Adversarial Red-Team','reviewer_execution_id':EXECUTION,'additional_audit_role':'Diversity and leakage auditor','scores':scores,'critical_defects':critical,'issues':issues,'decision':'REVISE' if critical else 'ACCEPT','confidence':0.97,'evidence':{'prediction_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'input_predictions_frozen_before_gold_metadata':True,'own_locator_witnesses':loc,'metamorphic_checks':checks,'review_mode':'Guided corrective v3 with prior v1/v2 own findings; public prospective diagnostic rules disclosed before review, same-model/common-design correlation, exposed dev.','forbidden_sources_read':False}})
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

def transposition(t,c):
    z=copy.deepcopy(t);f,b,g=c['fact'],c['baseline_value'],c['supported_value']
    def exchange(v):return g if type(v) is str and v==b else b if type(v) is str and v==g else v
    for w in z['worlds']:w['facts'][f]=exchange(w['facts'][f])
    for o in z['observations']:
        if o['fact']==f:o['equals']=exchange(o['equals'])
    return z

def decoded_source(r):
    z=copy.deepcopy(r)
    if r['auxiliary_scalar_variant']=='transposed':z['semantic_task']=transposition(r['semantic_task'],r['auxiliary_scalar_control'])
    return normalized(z)

def global_review(rows,pred,receipts):
    byid={r['sample_id']:r for r in rows};groups=collections.defaultdict(list);families=collections.defaultdict(list);fieldpairs=collections.defaultdict(list);scalarpairs=collections.defaultdict(list);obs=[];errors=[];claims=collections.defaultdict(set)
    known={'equal_support_usefulness_counterfactual','observation_reorder_final_invariance','nonempty_repetition_final_invariance','cross_fact_contradiction_counterfactual','post_empty_append_absorption','query_id_position_fact_crossing','field_order_invariance','null_guarantee_contrast','typed_equals_counterfactual','same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes','world_order_invariance','world_id_rename_equivariance','auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
    for r in rows:
        t=r['semantic_task'];f=r['source_family_key'];variant=r['auxiliary_scalar_variant'];context=r['controlled_context'];renderer=r['renderer'];key=(f,context,variant,renderer);groups[key].append(r);families[f].append(r);fieldpairs[(f,context,variant,r['schedule_index'])].append(r);scalarpairs[(f,context,r['schedule_index'],renderer)].append(r)
        try:
            idx=r['source_family_index'];require(type(idx) is int and idx in (12,13,18,19),'Source index');require(r['split']==('dev_challenge' if idx>=18 else 'train'),'Source split');require(context in 'ABCDEFGH' and variant in ('identity','transposed') and renderer in ('forward','reverse'),'Public conditional grouping fields');require(r['structural_family_profile']=={12:'cyclic_uniform',13:'acyclic_branching',18:'cyclic_weighted',19:'typed_path'}[idx],'Profile index')
            u,v,w=r['id_shift'],r['position_shift'],r['fact_name_shift'];require(all(type(i) is int and 0<=i<3 for i in (u,v,w)) and r['schedule_index']==9*u+3*v+w,'27 independent shifts')
            roles=r['fact_role_assignments'];required={'observation_control','observation_auxiliary','invariant_query','variation_query','unoffered_feature'};require(required<=set(roles) and all(k in required or re.fullmatch(r'aux_extra_[0-9]+',k) for k in roles),'Fact roles');require(len(roles)==len(set(roles.values())) and set(roles.values())==set(t['worlds'][0]['facts']),'Complete fact role bijection')
            control=r['auxiliary_scalar_control'];require(type(control) is dict and set(control)=={'fact','baseline_value','supported_value'} and control['fact']==roles['observation_auxiliary'],'Exact scalar-control public interface');b,g=control['baseline_value'],control['supported_value'];require(type(b) is type(g) is str and b=='linen' and g not in {'stone','unknown','linen'},'Distinct public diagnostic values');require(all(q['fact']!=control['fact'] for q in t['queries']),'Unqueried auxiliary');require({b,g}<={w['facts'][control['fact']] for w in t['worlds'] if type(w['facts'][control['fact']]) is str},'Both strings supported')
            pools=[r['query_id_pool'],r['queried_fact_name_pool']];require(all(type(p) is list and len(p)==len(set(p))==3 and all(type(x) is str for x in p) for p in pools),'Fixed pools');require(set(pools[1])=={roles[k] for k in ('invariant_query','variation_query','observation_control')},'Queried role pool');assignments=r['query_role_assignments'];require(len(assignments)==3 and {a['role'] for a in assignments}=={'invariant_query','variation_query','observed_control'},'Three query roles')
            for a in assignments:
                require({'role','position','query_id'}<=set(a)<={'role','position','query_id','fact'},'Assignment fields');role=('invariant_query','variation_query','observed_control').index(a['role']);pos=(role+v)%3;name=pools[1][(role+w)%3];qid=pools[0][(role+u)%3];q=t['queries'][pos];expectedfact=roles['observation_control' if role==2 else a['role']];require(a['position']==pos and a['query_id']==qid==q['query_id'] and q['fact']==name==expectedfact and ('fact' not in a or a['fact']==name),'Actual schedule not merely annotation')
                ans=pred[r['sample_id']]['expected_target']['queries'][pos];detail=pred[r['sample_id']]['prefix_evidence'];cat='empty' if not ans['counts'] else 'entailed' if ans['entailed'] else 'ambiguous';outcome={**{k:value for k,value in ans.items() if k!='query_id'},'category':cat,'null_context':cat+'_null' if ans['guaranteed'] is None else 'entailed_nonnull','bin_type_counts':[('null' if x['value'] is None else type(x['value']).__name__,x['count']) for x in ans['counts']],'all_prefix_survivor_ids':detail['prefix_survivors'],'all_prefix_counts':[len(p) for p in detail['prefix_survivors']]}
                obs.append({'sample_id':r['sample_id'],'family':f,'context':context,'variant':variant,'renderer':renderer,'split':r['split'],'role':role,'id':qid,'position':pos,'fact':name,'answer':ans,'outcome':outcome,'input_length':len(r['rendered_input']),'world_count':len(t['worlds']),'observation_count':len(t['observations'])})
            for c in r['family_control_claims']:
                require(type(c) is dict and set(c)<={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'} and len(set(c)&{'name','property'})==1 and c.get('scope') in ('corpus_variant','audit_probe'),'Closed family claim shape');name=c.get('name',c.get('property'));require(name in known,'Known claim');require('source_family_key' not in c or c['source_family_key']==f,'Claim same lineage family')
                if 'evidence_manifest' in c:require(type(c['evidence_manifest']) is str and bool(c['evidence_manifest']) and not pathlib.PurePosixPath(c['evidence_manifest']).is_absolute() and '..' not in pathlib.PurePosixPath(c['evidence_manifest']).parts,'Relative pointer only')
                for field in ('control_id','probe_id'):
                    if field in c:require(type(c[field]) is str and bool(c[field]),'Nonempty claim IDs')
                if 'probe_id' in c:require(c['scope']=='audit_probe','Probe scope')
                if 'pair_sample_ids' in c:
                    require(c['scope']=='corpus_variant' and type(c['pair_sample_ids']) is list and len(set(c['pair_sample_ids']))==2 and all(s in byid and byid[s]['source_family_key']==f for s in c['pair_sample_ids']),'Actual same-family pair IDs')
                    if name=='auxiliary_scalar_transposition_invariance':
                        pair=[byid[s] for s in c['pair_sample_ids']];require({z['auxiliary_scalar_variant'] for z in pair}=={'identity','transposed'} and len({(z['controlled_context'],z['schedule_index'],z['renderer']) for z in pair})==1,'Actual nominated scalar pair')
                if name in ('auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'):require(c['scope']=='corpus_variant','Required supplied gate cannot be a probe')
                claims[f].add((name,c['scope']))
            require(type(r['difficulty']) is int and type(r['structural_difficulty']) is int and 2<=r['difficulty']<=3 and 2<=r['structural_difficulty']<=3,'Honest levels2/3')
            if context in ('G','H'):require(r['difficulty']==r['structural_difficulty']==2,'G/H simple honest levels')
        except Exception as e:errors.append({'sample_id':r['sample_id'],'family':f,'error':str(e)})
    require(len(families)==4 and len(groups)==128 and all(len(rs)==27 and {r['schedule_index'] for r in rs}==set(range(27)) for rs in groups.values()),'All conditional groups complete')
    crossing=[]
    for key,rs in sorted(groups.items()):
        require(len({canon(normalized(r)) for r in rs})==1,'All decoded cases same within conditional group')
        os=[o for o in obs if (o['family'],o['context'],o['variant'],o['renderer'])==key];tables={};symbols={digest(o['outcome']):o['outcome'] for o in os}
        for names in [('id',),('position',),('fact',),('id','position'),('id','fact'),('position','fact'),('id','position','fact')]:
            table=collections.defaultdict(lambda:{'roles':collections.Counter(),'full_typed_outcomes':collections.Counter()})
            for o in os:bucket=table[canon([o[n] for n in names])];bucket['roles'][str(o['role'])]+=1;bucket['full_typed_outcomes'][digest(o['outcome'])]+=1
            count={1:9,2:3,3:1}[len(names)];require(len(table)==3**len(names) and all(dict(b['roles'])=={'0':count,'1':count,'2':count} for b in table.values()),'Complete actual role nuisance buckets');require(len({canon(b['full_typed_outcomes']) for b in table.values()})==1,'Full outcomes/provenance histogram conditional leakage')
            tables['+'.join(names)]={k:{label:dict(value) for label,value in v.items()} for k,v in table.items()}
        crossing.append({'conditional_group':list(key),'independent_outcome_payloads':symbols,'actual_histograms':tables})
    field_evidence=[];scalar_evidence=[]
    for key,pair in sorted(fieldpairs.items()):
        require(len(pair)==2 and {r['renderer'] for r in pair}=={'forward','reverse'} and canon(pair[0]['semantic_task'])==canon(pair[1]['semantic_task']) and pair[0]['rendered_input']!=pair[1]['rendered_input'],'Exact field-order pair');require(canon(pred[pair[0]['sample_id']]['expected_target'])==canon(pred[pair[1]['sample_id']]['expected_target']),'Field targets invariant')
        field_evidence.append({'pair_ids':[r['sample_id'] for r in pair],'record_hashes':[r['record_hash'] for r in pair]})
    for key,pair in sorted(scalarpairs.items()):
        require(len(pair)==2 and {r['auxiliary_scalar_variant'] for r in pair}=={'identity','transposed'},'Complete scalar pair');identity=next(r for r in pair if r['auxiliary_scalar_variant']=='identity');tr=next(r for r in pair if r['auxiliary_scalar_variant']=='transposed');c=identity['auxiliary_scalar_control'];require(canon(c)==canon(tr['auxiliary_scalar_control']),'Fixed identity-coordinate control');require(canon(transposition(identity['semantic_task'],c))==canon(tr['semantic_task']) and canon(transposition(tr['semantic_task'],c))==canon(identity['semantic_task']),'Actual exact nontrivial involution');require(canon(identity['semantic_task'])!=canon(tr['semantic_task']),'Actual table changed');pi,pt=pred[identity['sample_id']],pred[tr['sample_id']];require(canon(pi['expected_target'])==canon(pt['expected_target']) and pi['prefix_evidence']['prefix_survivors']==pt['prefix_evidence']['prefix_survivors'],'All exact outputs/prefixes invariant')
        scalar_evidence.append({'identity_id':identity['sample_id'],'identity_hash':identity['record_hash'],'transposed_id':tr['sample_id'],'transposed_hash':tr['record_hash'],'changed_fact':c['fact'],'identity_coordinate_values':[c['baseline_value'],c['supported_value']],'full_target_hash':digest(pi['expected_target']),'all_prefix_survivors':pi['prefix_evidence']['prefix_survivors']})
    require(len(field_evidence)==len(scalar_evidence)==1728,'Actual field/scalar pairs1728')
    controls=[];countertable=[];structural=[];components={};actual_claims={};probes=[]
    for f,rs in sorted(families.items()):
        require(len(rs)==864 and len({r['split'] for r in rs})==len({r['source_family_index'] for r in rs})==len({r['semantic_parent_id'] for r in rs})==len({r['semantic_family_id'] for r in rs})==1,'Whole-family source/parent/split custody');require(len({canon([r['query_id_pool'],r['queried_fact_name_pool'],r['auxiliary_scalar_control']]) for r in rs})==1,'Fixed pools and control');require(len({canon(decoded_source(r)['worlds']) for r in rs})==1,'Single decoded source table across all contexts/views')
        actual={('query_id_position_fact_crossing','corpus_variant'),('field_order_invariance','corpus_variant'),('auxiliary_scalar_transposition_invariance','corpus_variant'),('same_observations_auxiliary_table_counterfactual','corpus_variant'),('context_auxiliary_scalar_variant_crossing','corpus_variant'),('world_order_invariance','audit_probe'),('world_id_rename_equivariance','audit_probe')}
        for variant in ('identity','transposed'):
            base={r['controlled_context']:r for r in rs if r['auxiliary_scalar_variant']==variant and r['schedule_index']==0 and r['renderer']=='forward'};require(set(base)==set('ABCDEFGH'),'Complete contexts within variant');t={c:normalized(r) for c,r in base.items()};a={c:solve(z)[0] for c,z in t.items()};require(len({canon(z['worlds']) for z in t.values()})==len({canon(z['queries']) for z in t.values()})==1,'Same table/queries within variant')
            def same_final(c,d):return a[c]['surviving_worlds']==a[d]['surviving_worlds'] and canon(a[c]['queries'])==canon(a[d]['queries'])
            def one_equals(c,d):
                require(len(t[c]['observations'])==len(t[d]['observations']),'Equal lengths');ds=[i for i,(x,y) in enumerate(zip(t[c]['observations'],t[d]['observations'])) if canon(x)!=canon(y)];require(len(ds)==1 and t[c]['observations'][ds[0]]['fact']==t[d]['observations'][ds[0]]['fact'],'One equality only');return ds[0]
            changed=one_equals('A','B');require(len(a['A']['surviving_worlds'])==len(a['B']['surviving_worlds'])>=2 and any(x['can_reduce_uncertainty']!=y['can_reduce_uncertainty'] for x,y in zip(a['A']['queries'],a['B']['queries'])) and all(q['entailed'] for q in a['B']['queries']),'Actual equal support usefulness contrast');actual.add(('equal_support_usefulness_counterfactual','corpus_variant'))
            require(canon(t['C']['observations'])==canon(t['A']['observations'][::-1]) and same_final('A','C') and a['A']['belief_sizes']!=a['C']['belief_sizes'],'Actual reorder prefix change');actual.add(('observation_reorder_final_invariance','corpus_variant'))
            for c,d in [('A','D'),('G','H')]:require(canon(t[d]['observations'][:-1])==canon(t[c]['observations']) and canon(t[d]['observations'][-1])==canon(t[d]['observations'][0]) and same_final(c,d) and a[d]['belief_sizes']==a[c]['belief_sizes']+[len(a[c]['surviving_worlds'])],'Actual nonempty repetition')
            actual.add(('nonempty_repetition_final_invariance','corpus_variant'))
            one_equals('A','E');require(pred[base['E']['sample_id']]['input_derived_property_witnesses'].get('cross_fact_inconsistent_conjunction'),'Actual E contradiction');actual.add(('cross_fact_contradiction_counterfactual','corpus_variant'))
            require(canon(t['F']['observations'][:-1])==canon(t['E']['observations']) and canon(t['F']['observations'][-1])==canon(t['F']['observations'][0]) and same_final('E','F') and pred[base['F']['sample_id']]['input_derived_property_witnesses'].get('post_empty_absorption'),'Actual empty repeat absorption');actual.add(('post_empty_append_absorption','corpus_variant'))
            for c,d in [('E','G'),('F','H')]:
                require(one_equals(c,d)==1 and not a[c]['surviving_worlds'] and a[d]['surviving_worlds'] and pred[base[c]['sample_id']]['prefix_evidence']['prefix_survivors'][1]==pred[base[d]['sample_id']]['prefix_evidence']['prefix_survivors'][1],'Same prefix mixed equals');actual|={('same_prefix_empty_nonempty_counterfactual','corpus_variant'),('contradiction_equals_mixed_outcomes','corpus_variant')}
            if (typed(t['A']['observations'][changed]['equals']),typed(t['B']['observations'][changed]['equals'])) in [('true','1'),('1','true'),('false','0'),('0','false')]:actual.add(('typed_equals_counterfactual','corpus_variant'))
            w={k for r in rs if r['auxiliary_scalar_variant']==variant for k in pred[r['sample_id']]['input_derived_property_witnesses']}
            if {'multiworld_entailed_null','ambiguous_null_partition'}<=w:actual.add(('null_guarantee_contrast','corpus_variant'))
            controls.append({'family':f,'variant':variant,'all_context_base_ids_hashes':{c:{'sample_id':r['sample_id'],'record_hash':r['record_hash']} for c,r in base.items()},'independent_full_targets':a,'actual_core_relationships':sorted(actual),'manifest_pointers_read':False})
            for name in ('world_order_invariance','world_id_rename_equivariance'):probes.append({'family':f,'variant':variant,'name':name,'scope':'audit_probe','actual_execution':EXECUTION,'source_sample_id':base['A']['sample_id'],'source_hash':base['A']['record_hash'],'coverage_records':432,'evidence':'Own per-row metamorphics in exact-hash first receipts.'})
            for c,r in base.items():
                p=pred[r['sample_id']];prefix=p['prefix_evidence']['prefix_survivors'];components[r['sample_id']]={'record_hash':r['record_hash'],'difficulty':r['difficulty'],'structural_difficulty':r['structural_difficulty'],'worlds':len(t[c]['worlds']),'facts':len(t[c]['worlds'][0]['facts']),'observations':len(t[c]['observations']),'queries':len(t[c]['queries']),'prefix_sizes':[len(x) for x in prefix],'strict_nonempty_reductions':sum(0<len(b)<len(a_) for a_,b in zip(prefix,prefix[1:])),'final_distinct_payloads':payload_count(t[c],a[c]['surviving_worlds']),'final_bin_counts':[q['distinct_values'] for q in a[c]['queries']],'actual_properties':sorted(p['input_derived_property_witnesses'])}
            if variant=='identity':
                graph=incidence(t['A'],'observation_control','observation_auxiliary');profile=base['A']['structural_family_profile'];require({'cyclic_uniform':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))==1,'acyclic_branching':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']>=3,'cyclic_weighted':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))>1,'typed_path':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']==2}[profile],'Factual incidence profile');structural.append({'family':f,'profile':profile,'split':base['A']['split'],'actual_graph':graph,'table_partition_incidence_invariant':table_signature(t['A'])})
        actual_claims[f]=actual
        require(not claims[f]-actual,'Unsupported actual family control')
        aligned={(r['controlled_context'],r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer']):r for r in rs}
        for c,v,d,u in [('E','identity','G','transposed'),('E','transposed','G','identity'),('F','identity','H','transposed'),('F','transposed','H','identity')]:
            for schedule in range(27):
                for renderer in ('forward','reverse'):
                    left=aligned[(c,v,schedule,renderer)];right=aligned[(d,u,schedule,renderer)];lt,rt=left['semantic_task'],right['semantic_task'];lp,rp=pred[left['sample_id']],pred[right['sample_id']];require(canon(lt['observations'])==canon(rt['observations']) and canon(lt['queries'])==canon(rt['queries']),'Same observation/query typed input');require(lp['prefix_evidence']['prefix_survivors'][1]==rp['prefix_evidence']['prefix_survivors'][1] and not lp['expected_target']['surviving_worlds'] and rp['expected_target']['surviving_worlds'],'Matched exact first prefix opposite status');z=transposition(lt,left['auxiliary_scalar_control']);z['observations']=copy.deepcopy(lt['observations']);require(canon(z)==canon(rt),'Full same-observation worlds differ only declared y transposition')
                    countertable.append({'family':f,'pair_type':[c,v,d,u],'observation_length':len(lt['observations']),'left_id':left['sample_id'],'left_hash':left['record_hash'],'right_id':right['sample_id'],'right_hash':right['record_hash'],'same_observations':lt['observations'],'same_queries':lt['queries'],'same_first_prefix':lp['prefix_evidence']['prefix_survivors'][1],'left_prefix_sizes':lp['expected_target']['belief_sizes'],'right_prefix_sizes':rp['expected_target']['belief_sizes']})
    require(len(countertable)==864 and len({p['left_id'] for p in countertable}|{p['right_id'] for p in countertable})==1728,'864 disjoint paired rows')
    require(len({canon(s['actual_graph']['weighted_canonical_signature']) for s in structural})==4,'Four weighted structure profiles')
    global_diagnostics(rows,pred,receipts,groups,families,obs,crossing,field_evidence,scalar_evidence,controls,countertable,structural,components,probes,errors)

def global_diagnostics(rows,pred,receipts,groups,families,obs,crossing,field_evidence,scalar_evidence,controls,countertable,structural,components,probes,errors):
    # Features are computed only from actual visible inputs. Outcome labels use own frozen predictions.
    records=[];hist=collections.defaultdict(collections.Counter);rulegroups=collections.defaultdict(list)
    for r in rows:
        t=r['semantic_task'];p=pred[r['sample_id']];a=p['expected_target'];d=p['prefix_evidence'];empty=not a['surviving_worlds'];m=len(t['observations']);n=len(t['worlds']);roles={v:k for k,v in r['fact_role_assignments'].items()};counts=[sum(type(w['facts'][o['fact']]) is type(o['equals']) and w['facts'][o['fact']]==o['equals'] for w in t['worlds']) for o in t['observations']];value_lengths=[len(canon(o['equals'])) for o in t['observations']];tablecounts=collections.Counter(typed(v) for w in t['worlds'] for v in w['facts'].values());perobscolumn=[sorted(collections.Counter(typed(w['facts'][o['fact']]) for w in t['worlds']).items()) for o in t['observations']];prefix=d['prefix_survivors'][1];coarse=[n,d['fact_count'],m,len(t['queries'])]
        x={'sample_id':r['sample_id'],'record_hash':r['record_hash'],'family':r['source_family_key'],'split':r['split'],'context':r['controlled_context'],'variant':r['auxiliary_scalar_variant'],'renderer':r['renderer'],'task':t,'empty':empty,'answer':a,'coarse':coarse,'first_prefix':prefix,'input_chars':len(r['rendered_input']),'world_json_chars':len(canon(t['worlds'])),'all_fact_json_chars':sum(len(canon(w['facts'])) for w in t['worlds']),'world_scalar_chars':sum(len(canon(v)) for w in t['worlds'] for v in w['facts'].values()),'observation_chars':len(canon(t['observations'])),'observation_value_chars':value_lengths,'observation_full_table_value_frequencies':counts,'observed_column_typed_frequencies':perobscolumn,'all_table_value_frequencies':sorted(tablecounts.items()),'string_marker_table_counts':[tablecounts[canon(v)] for v in ('stone','unknown','linen','silk','cotton','opal')],'root_rules':p['frozen_diagnostic_predictions'],'full_table_key':digest(sorted(t['worlds'],key=lambda w:canon(w)))}
        records.append(x)
        for scope in [('family',r['source_family_key'],m),('split',r['split'],m),('family_variant',r['source_family_key'],r['auxiliary_scalar_variant'],m),('split_variant',r['split'],r['auxiliary_scalar_variant'],m)]:
            for v in sorted({typed(o['equals']) for o in t['observations']}):hist[canon([scope,'scalar_anywhere',v])][str(empty)]+=1
            for j,o in enumerate(t['observations']):
                for name,key in [('scalar_position',[j,typed(o['equals'])]),('fact_name',o['fact']),('fact_role',roles[o['fact']]),('position_fact_value',[j,o['fact'],typed(o['equals'])]),('position_role_value',[j,roles[o['fact']],typed(o['equals'])])]:hist[canon([scope,name,key])][str(empty)]+=1
            for name,key in [('scalar_pair_set',sorted({typed(o['equals']) for o in t['observations']})),('coarse_counts',coarse),('coarse_first_prefix_count',[coarse,len(prefix)]),('coarse_first_prefix_exact',[coarse,prefix]),('standalone_observation_support_counts',counts),('observed_scalar_encoding_lengths',value_lengths),('world_scalar_encoding_chars',x['world_scalar_chars']),('input_chars',x['input_chars'])]:hist[canon([scope,name,key])][str(empty)]+=1
        for scope,key in [('split',r['split']),('family',r['source_family_key'])]:rulegroups[canon([scope,key,r['controlled_context'],r['auxiliary_scalar_variant'],m])].append(x)
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='dev_challenge'];values=sorted({typed(o['equals']) for r in records for o in r['task']['observations']})
    def lookup(feature,units,train_units,dev_units,label):
        major=collections.Counter(label(o) for o in train_units).most_common(1)[0][0];table=collections.defaultdict(collections.Counter)
        for o in train_units:table[canon(feature(o))][label(o)]+=1
        choose={k:v.most_common(1)[0][0] for k,v in table.items()};result={}
        for split,rs in [('train',train_units),('dev_challenge',dev_units)]:
            predictions=[choose.get(canon(feature(o)),major) for o in rs];result[split]={'correct':sum(v==label(o) for v,o in zip(predictions,rs)),'total':len(rs),'seen_feature_units':sum(canon(feature(o)) in choose for o in rs),'majority_correct':sum(label(o)==major for o in rs)}
        result['fit_scope']='Train CPU majority lookup, exposed-dev transfer; no model training, no hidden metadata/final support in features.'
        return result
    features={
        'coarse_counts':lambda r:r['coarse'],'first_prefix_count':lambda r:len(r['first_prefix']),'coarse_first_prefix_count':lambda r:[r['coarse'],len(r['first_prefix'])],'coarse_exact_first_prefix':lambda r:[r['coarse'],r['first_prefix']],
        'observation_scalar_set':lambda r:sorted({typed(o['equals']) for o in r['task']['observations']}),'ordered_observation_scalars':lambda r:[typed(o['equals']) for o in r['task']['observations']], 'ordered_actual_observation_facts':lambda r:[o['fact'] for o in r['task']['observations']], 'observation_scalar_position_fact_joint':lambda r:r['task']['observations'],
        'input_char_length':lambda r:r['input_chars'],'world_json_char_length':lambda r:r['world_json_chars'],'fact_json_char_length':lambda r:r['all_fact_json_chars'],'world_scalar_encoding_char_length':lambda r:r['world_scalar_chars'],'observation_char_length':lambda r:r['observation_chars'],'observation_value_encoding_lengths':lambda r:r['observation_value_chars'],
        'standalone_observation_value_support_counts':lambda r:r['observation_full_table_value_frequencies'],'coarse_standalone_observation_support_counts':lambda r:[r['coarse'],r['observation_full_table_value_frequencies']],'observed_column_typed_value_frequencies':lambda r:r['observed_column_typed_frequencies'],'all_table_typed_value_frequencies':lambda r:r['all_table_value_frequencies'],'literal_marker_auxiliary_table_counts':lambda r:r['string_marker_table_counts'],
        'observation_scalars_and_standalone_frequencies':lambda r:[[typed(o['equals']) for o in r['task']['observations']],r['observation_full_table_value_frequencies']], 'scalar_set_coarse_first_prefix_count':lambda r:[sorted({typed(o['equals']) for o in r['task']['observations']}),r['coarse'],len(r['first_prefix'])],
        'full_visible_table_fingerprint':lambda r:r['full_table_key'],'full_table_and_observation_conjunction':lambda r:[r['full_table_key'],r['task']['observations']],'all_typed_table_frequencies_and_observation_scalars':lambda r:[r['all_table_value_frequencies'],[typed(o['equals']) for o in r['task']['observations']]],'literal_table_counts_and_scalar_set':lambda r:[r['string_marker_table_counts'],sorted({typed(o['equals']) for o in r['task']['observations']})],'world_scalar_chars_and_scalar_set':lambda r:[r['world_scalar_chars'],sorted({typed(o['equals']) for o in r['task']['observations']})],'repeat_flag':lambda r:len({canon(o) for o in r['task']['observations']})<len(r['task']['observations'])
    }
    for v in values:features['contains_typed_scalar_'+v]=lambda r,v=v:any(typed(o['equals'])==v for o in r['task']['observations'])
    for j in range(3):features['scalar_at_position_'+str(j)]=lambda r,j=j:typed(r['task']['observations'][j]['equals']) if j<len(r['task']['observations']) else 'MISSING'
    rowbaselines={name:lookup(fn,records,train,dev,lambda r:r['empty']) for name,fn in features.items()}
    # Cross-validation for table-sensitive fits must show custody and feature coverage, not an OOD label.
    qtrain=[o for o in obs if o['split']=='train'];qdev=[o for o in obs if o['split']=='dev_challenge']
    def querylabel(o,k):
        if k=='full_payload':return canon({x:y for x,y in o['answer'].items() if x!='query_id'})
        if k=='category':return o['outcome']['category']
        return canon(o['answer'][k])
    qfeatures={'query_id':lambda o:o['id'],'query_position':lambda o:o['position'],'actual_fact_name':lambda o:o['fact'],'id_position_fact_joint':lambda o:[o['id'],o['position'],o['fact']],'input_char_length':lambda o:o['input_length'],'world_count':lambda o:o['world_count'],'observation_count':lambda o:o['observation_count']}
    qbaselines={name:{kind:lookup(fn,obs,qtrain,qdev,lambda o,k=kind:querylabel(o,k)) for kind in ('entailed','can_reduce_uncertainty','category','full_payload')} for name,fn in qfeatures.items()}
    frozenrules={}
    for name in ('v1_original_scalar_marker_empty','prospective_marker_and_auxiliary_token_empty'):
        metrics={};complete={}
        for group,rs in rulegroups.items():
            predicted=[r['root_rules'][name] for r in rs];metrics[group]={'correct':sum(p==r['empty'] for p,r in zip(predicted,rs)),'total':len(rs),'false_empty':sum(p and not r['empty'] for p,r in zip(predicted,rs)),'false_nonempty':sum(not p and r['empty'] for p,r in zip(predicted,rs))}
            covered=[r for r in rs if r['root_rules'][name]];complete[group]={'conditional_covered_records':len(covered),'conditional_exact_target_correct':sum(canon(r['root_rules']['old_empty_core_prediction'])==canon(r['answer']) for r in covered),'query_payload_correct':sum(canon(q)==canon(z) for r in covered for q,z in zip(r['root_rules']['old_empty_core_prediction']['queries'],r['answer']['queries'])),'query_payload_total':len(covered)*3,'abstentions':len(rs)-len(covered)}
        for split in ('train','dev_challenge'):
            rs=[r for r in records if r['split']==split];require(sum(r['root_rules'][name]!=r['empty'] for r in rs)==432,'Actual frozen rule432 errors split')
        frozenrules[name]={'conditional_family_split_context_variant_length_status':metrics,'conditional_full_empty_target_payloads':complete,'overall_status_correct':sum(r['root_rules'][name]==r['empty'] for r in records),'overall_status_total':len(records),'errors':sum(r['root_rules'][name]!=r['empty'] for r in records),'full_target_scope':'Core predicts empty target when predicate true, abstains otherwise. Public full prefix sequence initial N omitted from output belief_sizes as required.'}
    # Matched opposite-status pairs establish hard lower bound for all table-free feature functions.
    recbyid={r['sample_id']:r for r in records};paired_features=['observation_scalar_set','ordered_observation_scalars','ordered_actual_observation_facts','observation_scalar_position_fact_joint','coarse_counts','coarse_first_prefix_count','coarse_exact_first_prefix','observation_char_length','observation_value_encoding_lengths','repeat_flag']
    for pair in countertable:
        l,r=recbyid[pair['left_id']],recbyid[pair['right_id']]
        for name in paired_features:require(canon(features[name](l))==canon(features[name](r)),'Actual matched-feature equality '+name)
    # Fresh actual-source minimal controls show why literal predictions cannot stand in for table evidence.
    minimalpairs=[]
    for pair in countertable:
        left=next(r for r in rows if r['sample_id']==pair['left_id']);right=next(r for r in rows if r['sample_id']==pair['right_id'])
        if left['schedule_index']!=0 or left['renderer']!='forward':continue
        lt,rt=left['semantic_task'],right['semantic_task'];x,y=lt['observations'][0],lt['observations'][1];first=next(w for w in lt['worlds'] if typed(w['facts'][x['fact']])==typed(x['equals']));second=next(w for w in lt['worlds'] if typed(w['facts'][y['fact']])==typed(y['equals']));q=lt['queries'][0];facts={x['fact'],y['fact'],q['fact']};ids=[first['world_id'],second['world_id']]
        def reduced(t):return {'kind':'belief','worlds':[{'world_id':w['world_id'],'facts':{f:v for f,v in w['facts'].items() if f in facts}} for w in t['worlds'] if w['world_id'] in ids],'observations':copy.deepcopy(t['observations']),'queries':[copy.deepcopy(q)]}
        ml,mr=reduced(lt),reduced(rt);al,dl=solve(ml);ar,dr=solve(mr);require(not al['surviving_worlds'] and ar['surviving_worlds'] and ml['observations']==mr['observations'] and dl['prefix_survivors'][1]==dr['prefix_survivors'][1],'Minimal opposite-status actual inputs')
        minimalpairs.append({'left_source_id':left['sample_id'],'left_record_hash':left['record_hash'],'right_source_id':right['sample_id'],'right_record_hash':right['record_hash'],'left_minimal_input':ml,'left_exact_target':al,'right_minimal_input':mr,'right_exact_target':ar,'same_first_prefix':dl['prefix_survivors'][1],'frozen_rules_same_predictions':{name:diagnostic_rules(ml)[name] for name in frozenrules},'scope':'Actually executed two-world, one-query reduced pair retains individually supported contradictory observations. Audit probe, no corpus rows or source families added.'})
    require(len(minimalpairs)==16,'Four pairtypes each family minimalcounterexamples')
    scalarchecks=0
    for r in rows:scalar_rename(r['semantic_task'],pred[r['sample_id']]['expected_target']);scalarchecks+=1
    # CPU unrestricted in-corpus lookup shows correlated fitting potential, separately from exposed-dev transfer.
    corpus=collections.defaultdict(set)
    for r in records:corpus[canon(features['full_table_and_observation_conjunction'](r))].add(r['empty'])
    require(all(len(labels)==1 for labels in corpus.values()),'Visible table+conjunction uniquely determines status')
    corpusfit={'feature':'Actual visible table fingerprint plus ordered observation conjunction','unique_feature_buckets':len(corpus),'in_corpus_status_fit':len(records),'records':len(records),'exposed_dev_seen_train_table_buckets':rowbaselines['full_table_and_observation_conjunction']['dev_challenge']['seen_feature_units'],'scope':'Unrestricted corpus lookup is a finite-space fit demonstration, not independently held-out model generalization or a hidden-metadata shortcut.'}
    perfect={k:v for k,v in rowbaselines.items() if all(v[split]['correct']==v[split]['total'] for split in ('train','dev_challenge'))}
    # Strongest single-coordinate thresholds trained from data, supplementary to majority bucket lookup.
    thresholdfeatures={'first_prefix_count':lambda r:len(r['first_prefix']),'input_chars':lambda r:r['input_chars'],'world_scalar_chars':lambda r:r['world_scalar_chars'],'world_json_chars':lambda r:r['world_json_chars']}
    thresholdbaselines={}
    for name,fn in thresholdfeatures.items():
        levels=sorted({fn(r) for r in train});candidates=[(sum((fn(r)<=cut)==r['empty'] for r in train),cut,True) for cut in levels]+[(sum((fn(r)>cut)==r['empty'] for r in train),cut,False) for cut in levels];_,cut,le=max(candidates,key=lambda v:(v[0],-v[1],v[2]));thresholdbaselines[name]={'train_chosen_threshold':cut,'predict_empty_iff':'feature<=threshold' if le else 'feature>threshold','train_correct':sum(((fn(r)<=cut) if le else (fn(r)>cut))==r['empty'] for r in train),'dev_correct':sum(((fn(r)<=cut) if le else (fn(r)>cut))==r['empty'] for r in dev),'train_total':len(train),'dev_total':len(dev)}
    highfinding=None
    if perfect:
        highfinding={'code':'remaining_perfect_input_only_lookup','severity':'high','evidence':{'tested_perfect_cpu_features':perfect},'fix':'Design fresh matched counterfactuals crossing those input features and outcome within source families, preserving all existing controls and prospective scope.'}
    for receipt,r in zip(receipts,rows):
        receipt['evidence']['truth_and_attached_properties']='PASS' if not receipt['critical_defects'] else 'FAIL';receipt['evidence']['global_control_metadata_errors']=errors;receipt['evidence']['same_observation_table_required_lower_bound_errors']=864
        receipt['issues'].append(issue('bounded_guided_four_source_scope','low',{'source_families':4,'semantic_base_contexts':32,'scalar_base_cases':64,'exposed_dev_families':2,'same_model_common_design_and_prior_own_v1_v2_findings':True,'class_nonempty_majority':0.75,'useful_false_majority':0.875,'table_context_in_corpus_fit':True},'Keep truthful bounded control, correlated-view/common-model and exposed-dev disclosures; no wholebatch shortcut removal, sealed OOD, efficacy or Levels4–7 claim.'))
        if highfinding and r['controlled_context'] in ('E','F'):
            receipt['critical_defects'].append(highfinding['code']);receipt['issues'].append(highfinding);receipt['scores']['shortcut_resistance']=2;receipt['decision']='REVISE'
        for error in errors:
            if error.get('sample_id')==r['sample_id']:
                receipt['critical_defects'].append('public_metadata_failure');receipt['issues'].append(issue('public_metadata_failure','high',error,'Correct actual interface/claim with fresh hashes and review.'));receipt['decision']='REVISE'
    freeze(OUT/f'{BATCH}.jsonl',receipts)
    exact=collections.Counter(digest(r['semantic_task']) for r in rows);decoded=collections.Counter(digest(normalized(r)) for r in rows);decodedtables={digest(normalized(r)['worlds']) for r in rows};sources={digest(decoded_source(r)['worlds']) for r in rows};visibletables={digest(r['semantic_task']['worlds']) for r in rows};unweighted=collections.defaultdict(list)
    for s in structural:unweighted[canon(s['actual_graph']['unweighted_canonical_signature'])].append(s['family'])
    identitysets={}
    for split in ('train','dev_challenge'):
        xs=[r for r in rows if r['split']==split];identitysets[split]={'source_family_keys':{r['source_family_key'] for r in xs},'semantic_parent_ids':{r['semantic_parent_id'] for r in xs},'semantic_family_ids':{r['semantic_family_id'] for r in xs},'world_ids':{w['world_id'] for r in xs for w in r['semantic_task']['worlds']},'exact_tasks':{digest(r['semantic_task']) for r in xs},'decoded_sources':{digest(decoded_source(r)['worlds']) for r in xs}}
    overlap={k:sorted(identitysets['train'][k]&identitysets['dev_challenge'][k]) for k in identitysets['train']};require(all(not v for v in overlap.values()),'No unexpected exact source/input identity split leakage');require(len(exact)==1728 and len(decoded)==64 and len(decodedtables)==8 and len(sources)==4 and len(visibletables)==24,'Actual view/table/source counts')
    # Original v1/v2 public stripped tasks only for lineage verification, never prior verdicts/golds/manifests.
    lineage={}
    for prior in ('belief-gap-v1','belief-gap-v2'):
        path=ROOT/'reviews/input_only'/f'{prior}.jsonl';priorinputs=load(path);tables={digest(sorted(t['semantic_task']['worlds'],key=canon)) for t in priorinputs};currentidentity={digest(sorted(r['semantic_task']['worlds'],key=canon)) for r in rows if r['auxiliary_scalar_variant']=='identity'};lineage[prior]={'input_only_sha256':sha(path),'prior_visible_tables':len(tables),'current_identity_visible_tables':len(currentidentity),'exact_visible_table_overlap':len(tables&currentidentity),'source_families_added':0,'scope':'Only previously reviewed public stripped task world tables used, no prior candidate target/receipt or verdict opened.'}
    diversity={'reviewer_execution_id':EXECUTION,'batch':BATCH,'candidate_sha256':sha(CAND),'records':len(rows),'source_families':len(families),'semantic_base_contexts':32,'scalar_base_cases':len(groups)//2,'conditional_crossing_groups':len(groups),'rotation_level_tasks':len(rows)//2,'base_query_observations':len(groups)//2*3,'rendered_query_observations':len(obs),'actual_parsed_tasks':len(exact),'decoded_variant_tasks':len(decoded),'decoded_variant_tables':len(decodedtables),'decoded_source_tables':len(sources),'visible_fact_name_variant_tables':len(visibletables),'field_order_pairs':len(field_evidence),'scalar_pairs':len(scalar_evidence),'same_observation_opposite_status_pairs':len(countertable),'split_rows':dict(collections.Counter(r['split'] for r in rows)),'source_split_identity_overlap':overlap,'lineage_from_prior_public_stripped_inputs':lineage,'actual_structural_profiles':structural,'shared_unweighted_motifs':dict(unweighted),'conditional_query_full_typed_outcome_histograms':crossing,'actual_field_pair_hashes':field_evidence,'actual_scalar_pair_hashes_prefixes':scalar_evidence,'actual_opposite_status_pair_hashes_inputs':countertable,'actual_core_controls':controls,'actual_executed_probes':probes,'component_difficulty_evidence':components,'metadata_control_errors':errors,'limitations':['One finite noiseless filter/partition mechanism, four source tables and32 semanticcontexts.64 scalarcases and3456 views add no independent source mechanisms.','Train/dev source IDs, world IDs and exact tables/tasks separate; cyclic_uniform/cyclic_weighted share an unweighted incidence motif. Dev is exposed, no sealed efficacy.','Query triples fully crossed conditional on variant. Whole row query rotations remain cyclic; observation-fact/scalar/table-frequency roles are not globally balanced.','G/H are Level2, singleton families13/19 and non-strict second filters allfamilies.','Prior own v1/v2 findings and public correction guidance plus same underlying model correlate review errors.']}
    shortcut={'reviewer_execution_id':EXECUTION,'batch':BATCH,'candidate_sha256':sha(CAND),'review_mode':'Guided corrective v3; new input-only solver/calibration freeze, prior own v1/v2 knowledge and public literal diagnostic guidance disclosed.','frozen_rule_results':frozenrules,'input_cpu_lookups':rowbaselines,'single_feature_threshold_baselines':thresholdbaselines,'query_input_cpu_lookups':qbaselines,'typed_scalar_position_fact_role_frequency_length_histograms':{k:dict(v) for k,v in hist.items()},'actual_matched_feature_lower_bound':{'identical_features':paired_features,'disjoint_opposite_status_pairs':len(countertable),'paired_records':1728,'required_errors_at_least':864,'scope':'Deterministic status predictors using only these jointly identical features, excludes differing world payloads. Tested actual equality on every supplied pair.'},'strongest_tested_perfect_input_lookups':perfect,'in_corpus_table_conjunction_lookup':corpusfit,'actual_minimal_two_world_counterexamples':minimalpairs,'consistent_scalar_bijection_executions':scalarchecks,'class_counts':dict(collections.Counter(str(r['empty']) for r in records)),'query_category_counts':dict(collections.Counter(o['outcome']['category'] for o in obs)),'query_entailed_counts':dict(collections.Counter(str(o['answer']['entailed']) for o in obs)),'query_useful_counts':dict(collections.Counter(str(o['answer']['can_reduce_uncertainty']) for o in obs)),'difficulty_counts':dict(collections.Counter(str(r['difficulty']) for r in rows)),'high_finding':highfinding,'limits':['No learned model or efficacy evaluation; CPUs counts/majority/finite lookup only, all labels independently recomputed.','Standalone full-table value counts are input features; no gold final support is an input shortcut feature.','Exact table intersection legitimately determines final status. Its perfect performance cannot establish dataset usefulness or hidden shortcut leakage.','Required target length/count-bin differences are semantic output consequences. Input character features measured separately.','No statement that every scalar/fact/position frequency is balanced or that all potential shortcuts are excluded.']}
    dp=ROOT/'audits/diversity'/f'{BATCH}-adversarial.json';sp=ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json';save(dp,diversity);save(sp,shortcut)
    execution={'reviewer_execution_id':EXECUTION,'batch':BATCH,'status':'completed','records':len(rows),'input_only_sha256':sha(INPUT),'candidate_sha256':sha(CAND),'predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'toy_expectations_sha256':sha(OUT/f'{BATCH}.toy_expectations.jsonl'),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'final_receipts_sha256':sha(OUT/f'{BATCH}.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'public_read_scope_hashes':{str(p.relative_to(ROOT)):sha(p) for p in PUBLIC},'diversity_sha256':sha(dp),'shortcut_sha256':sha(sp),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'target_attached_property_errors':sum('target_property_or_prompt_failure' in r['critical_defects'] for r in receipts),'metadata_control_errors':len(errors),'metamorphic_counts':dict(collections.Counter(c for r in receipts for c in r['evidence']['metamorphic_checks'])),'scalar_bijection_checks':scalarchecks,'source_families':4,'conditional_crossing_groups':128,'field_pairs':1728,'scalar_pairs':1728,'same_observation_opposite_status_pairs':864,'first_predictions_and_judgments_preserved':True,'prior_v1_v2_artifacts_unchanged':True,'forbidden_sources_read':False,'review_mode':'Guided corrective actual independent same-model execution, own bitmask exact solver with manual calibration, prior own findings and publicrootdiagnostic guidance disclosed, no independent-model or sealed efficacy claim.','high_finding':highfinding}
    ep=ROOT/'audits/execution'/f'{BATCH}-adversarial.json';save(ep,execution);hp=ROOT/'audits/history'/f'belief_gap_v3_adversarial_audit-{sha(pathlib.Path(__file__))}.py'
    if not hp.exists():hp.write_bytes(pathlib.Path(__file__).read_bytes())
    print(canon({'decisions':execution['decisions'],'target_property_errors':execution['target_attached_property_errors'],'metadata_control_errors':len(errors),'final_sha256':execution['final_receipts_sha256'],'execution_sha256':sha(ep),'perfect_input_lookups':perfect,'row_baseline_dev_correct':{k:v['dev_challenge']['correct'] for k,v in rowbaselines.items()},'frozen_rule_errors':{k:v['errors'] for k,v in frozenrules.items()}}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['stage1','stage2']);args=ap.parse_args()
    if args.stage=='stage1':stage1()
    else:stage2()
