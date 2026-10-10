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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['expectations','review'],default='expectations');args=parser.parse_args()
    if args.stage=='expectations':stage1()
    else:stage2()
