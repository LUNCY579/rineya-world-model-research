#!/usr/bin/env python3
"""Own actual V4 canonical comparison, after input-only expectations are frozen.
Author /root/adversarial_auditor. Own solver/prospective resolver reuse disclosed.
No private producer/formal/checker/other-review imports or reads.
"""
import collections,copy,hashlib,importlib.util,itertools,json,pathlib,re,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];ACTOR='/root/adversarial_auditor'
def ownmodule(name,relative):
 spec=importlib.util.spec_from_file_location(name,ROOT/relative);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
# These are OWN immutable sources; this file is frozen before its first import/execution.
I=ownmodule('own_v4_input','audits/belief_v4_adversarial_input_r2.py')
P=ownmodule('own_v4_prospective','audits/belief_v4_adversarial_prospective_r1.py')
canon,digest,sha,require,freeze,solve,partition,typed,payload_count=I.canon,I.digest,I.sha,I.require,I.freeze,I.solve,I.partition,I.typed,I.payload_count
jsonbytes,jlbytes,loadjson,rows=I.jsonbytes,I.jlbytes,I.loadjson,I.rows
DISCLOSURE=I.DISCLOSURE+' Canonical comparison is a separately authorized later stage after exact fresh input predictions and initial judgments frozen; all per-row targets are compared with those expectations. Public raw manifest assertions are untrusted and independently derived; no other reviewer/private checker/producer implementation or root machine-gate results read.'
CAND='data/RBM-001/candidate_renderings/belief-gap-v4.jsonl'
MAN='data/RBM-001/manifests/belief-gap-v4.family_controls.json'
OLDC='data/RBM-001/candidate_renderings/belief-gap-v3.jsonl'
PRED='reviews/adversarial/belief-gap-v4.input_predictions.jsonl'
PRE='audits/execution/belief-gap-v4-canonical-adversarial-r2.preexec.json'
INHERITED={'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}
AGGREGATE={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
PROPERTIES={'multiworld_unobserved_invariant','conjunction_required_invariant','useful_after_conjunctive_evidence','multiworld_all_queries_redundant','multiworld_entailed_null','ambiguous_null_partition','cross_fact_inconsistent_conjunction','post_empty_absorption','nonempty_repeated_observation','typed_observation_separator','typed_query_partition_separator','skewed_possible_value_partition'}
GENERIC={'finite_synthetic_proxy','finite_noiseless_filter','possible_value_partition'}
ROLES=('invariant_query','variation_query','observed_control')
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

def entry_key(e):return (e.get('control_id'),e.get('name'),canon(e.get('pair_sample_ids')))
def inherited_pair(left,right,pred):
 lt,rt=left['semantic_task'],right['semantic_task'];a,b=pred[left['sample_id']],pred[right['sample_id']]
 require(canon(lt['worlds'])==canon(rt['worlds']) and canon(lt['queries'])==canon(rt['queries']),'same typed worlds and ordered queries')
 require(len(lt['observations'])==len(rt['observations']) in (2,3),'same actual pair length')
 diffs=[(i,k) for i,(x,y) in enumerate(zip(lt['observations'],rt['observations'])) for k in set(x)|set(y) if canon(x.get(k))!=canon(y.get(k))]
 require(diffs==[(1,'equals')],'actual difference only observation1 typed equality')
 ad,bd=a['prefix_supports'],b['prefix_supports'];require(ad['prefix_survivors'][1]==bd['prefix_survivors'][1],'same first prefix exact world IDs')
 require(not a['expected_target']['surviving_worlds'] and b['expected_target']['surviving_worlds'],'empty first nonempty second')
 require(lt['observations'][0]['fact']!=lt['observations'][1]['fact'] and all(ad['individual_observation_supports'][i] for i in (0,1)),'individually supported different-fact contradiction')
 return {'pair_sample_ids':[left['sample_id'],right['sample_id']],'record_hashes':[left['record_hash'],right['record_hash']],'actual_length':len(lt['observations']),'changed_fields':diffs,'prefix_survivors':[ad['prefix_survivors'],bd['prefix_survivors']],'prefix_counts':[[len(s) for s in ad['prefix_survivors']],[len(s) for s in bd['prefix_survivors']]],'targets_sha256':[digest(a['expected_target']),digest(b['expected_target'])]}
def validate_entry(e,byid,pred,ch):
 require(type(e) is dict and P.ENTRY_FIELDS<=set(e),'published eight minimum entry fields')
 require(e['name'] in INHERITED and e['scope']=='corpus_variant' and e['exact_candidate_sha256']==ch,'entry name/scope/exact whole PENDING candidate SHA')
 ids=e['pair_sample_ids'];require(type(ids) is list and len(ids)==2 and all(type(x) is str and x in byid for x in ids) and len(set(ids))==2,'actual ordered entry pair')
 left,right=[byid[x] for x in ids];proof=inherited_pair(left,right,pred)
 require(all(r['source_family_key']==e['source_family_key'] and r['auxiliary_scalar_variant']==e['auxiliary_scalar_variant'] for r in (left,right)),'entry actual family and scalar variant')
 require(type(e['observation_length']) is int and e['observation_length']==proof['actual_length'],'entry actual typed length')
 for k in ('schedule_index','renderer'):
  if k in e:require(all(type(r[k]) is type(e[k]) and r[k]==e[k] for r in (left,right)),'optional entry coordinate factual '+k)
 # No closed evidence-field shape invented. Validate every present recognized assertion.
 assertions={'first_prefix_survivor_ids':proof['prefix_survivors'][0][1],'left_prefix_survivor_ids':proof['prefix_survivors'][0],'right_prefix_survivor_ids':proof['prefix_survivors'][1],'left_prefix_counts':proof['prefix_counts'][0],'right_prefix_counts':proof['prefix_counts'][1],'left_complete_query_payloads':pred[left['sample_id']]['expected_target']['queries'],'right_complete_query_payloads':pred[right['sample_id']]['expected_target']['queries'],'typed_first_observation':left['semantic_task']['observations'][0],'changed_fields':['observations[1].equals']}
 present=[]
 for k,v in assertions.items():
  if k in e:require(canon(e[k])==canon(v),'provided parsed evidence inaccurate '+k);present.append(k)
 require(bool(present),'no parsed comparison/prefix evidence supplied')
 return {**proof,'name':e['name'],'control_id':e['control_id'],'checked_present_evidence_fields':present,'entry_sha256':digest(e)}
def resolve(owner,c,byid,index,pred,ch):
 require(type(c) is dict and set(c)<=P.CLAIM_FIELDS and len(set(c)&{'name','property'})==1 and 'scope' in c,'closed canonical family claim shape')
 name=c.get('name',c.get('property'));require(name in INHERITED and c['scope']=='corpus_variant' and 'probe_id' not in c,'inherited supplied scope')
 require({'pair_sample_ids','evidence_manifest','control_id'}<=set(c),'inherited inline pair/pointer/ID mandatory')
 require(c['evidence_manifest']==MAN and type(c['control_id']) is str and bool(c['control_id']),'actual supplied manifest and descriptive control ID')
 require('source_family_key' not in c or c['source_family_key']==owner['source_family_key'],'claim optional owner family')
 ids=c['pair_sample_ids'];require(type(ids) is list and len(ids)==2 and all(type(x) is str and x in byid for x in ids) and len(set(ids))==2,'actual distinct evidence pair IDs')
 left,right=[byid[x] for x in ids]
 for r in (left,right):
  for k in ('source_family_key','auxiliary_scalar_variant','schedule_index','renderer'):require(type(r[k]) is type(owner[k]) and r[k]==owner[k],'owner coordinate '+k)
 length=len(owner['semantic_task']['observations']);want=('E','G') if length==2 else ('F','H') if length==3 else None
 require(want is not None and (left['controlled_context'],right['controlled_context'])==want,'owner actual length chooses ordered E/G or F/H')
 require(all(len(r['semantic_task']['observations'])==length for r in (left,right)),'actual owner and pair lengths agree')
 key=(c['control_id'],name,canon(ids));matches=index.get(key,[]);require(len(matches)==1,'exact one control ID+name+ordered pair entry')
 e=matches[0];proof=validate_entry(e,byid,pred,ch)
 return {'name':name,'control_id':c['control_id'],'pair_sample_ids':ids,'pair_record_hashes':[left['record_hash'],right['record_hash']],'owner_not_pair_member':owner['sample_id'] not in ids,'actual_owner_length':length,'entry_sha256':digest(e),'derived_pair_evidence_sha256':digest(proof)}

def family_semantics(family,rs,pred):
 require(len(rs)==864 and len({r['semantic_family_id'] for r in rs})==len({r['semantic_parent_id'] for r in rs})==len({r['source_family_index'] for r in rs})==len({r['split'] for r in rs})==1,'whole source-family/split/parent custody')
 require(len({canon([r['query_id_pool'],r['queried_fact_name_pool'],r['auxiliary_scalar_control']]) for r in rs})==1,'fixed ordered pools/identity scalar controls')
 require(len({canon(decoded_source(r)['worlds']) for r in rs})==1,'one actual source table under declared fact/scalar decoding')
 aligned={(r['controlled_context'],r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer']):r for r in rs};require(len(aligned)==864,'unique actual declared coordinates')
 actual=set(INHERITED)|set(AGGREGATE)|{'query_id_position_fact_crossing','field_order_invariance','world_order_invariance','world_id_rename_equivariance'};controls=[];components=[]
 for variant in ('identity','transposed'):
  base={c:aligned[(c,variant,0,'forward')] for c in 'ABCDEFGH'};tasks={c:normalized(r) for c,r in base.items()};answers={c:solve(t)[0] for c,t in tasks.items()}
  require(len({canon(t['worlds']) for t in tasks.values()})==len({canon(t['queries']) for t in tasks.values()})==1,'same table/query slots across eight contexts')
  def equal_final(a,b):return answers[a]['surviving_worlds']==answers[b]['surviving_worlds'] and canon(answers[a]['queries'])==canon(answers[b]['queries'])
  def one_equality(a,b):
   x,y=tasks[a]['observations'],tasks[b]['observations'];require(len(x)==len(y),'core contrast actual equal lengths');diff=[(i,k) for i,(u,v) in enumerate(zip(x,y)) for k in set(u)|set(v) if canon(u.get(k))!=canon(v.get(k))];require(len(diff)==1 and diff[0][1]=='equals','core contrast only one typed equality changed');return diff[0][0]
  changed=one_equality('A','B');require(len(answers['A']['surviving_worlds'])==len(answers['B']['surviving_worlds'])>=2 and any(a['can_reduce_uncertainty']!=b['can_reduce_uncertainty'] for a,b in zip(answers['A']['queries'],answers['B']['queries'])),'actual equal-support usefulness contrast');actual.add('equal_support_usefulness_counterfactual')
  require(canon(tasks['C']['observations'])==canon(tasks['A']['observations'][::-1]) and equal_final('A','C') and answers['A']['belief_sizes']!=answers['C']['belief_sizes'],'actual reorder preserves final output but changes trace');actual.add('observation_reorder_final_invariance')
  for a,b in [('A','D'),('G','H')]:require(canon(tasks[b]['observations'][:-1])==canon(tasks[a]['observations']) and canon(tasks[b]['observations'][-1])==canon(tasks[b]['observations'][0]) and equal_final(a,b) and answers[b]['belief_sizes']==answers[a]['belief_sizes']+[len(answers[a]['surviving_worlds'])] and bool(answers[a]['surviving_worlds']),'actual nonempty repetition')
  actual.add('nonempty_repetition_final_invariance');one_equality('A','E');require(witnesses(tasks['E'],*solve(tasks['E'])).get('cross_fact_inconsistent_conjunction'),'actual different-fact E contradiction');actual.add('cross_fact_contradiction_counterfactual')
  require(canon(tasks['F']['observations'][:-1])==canon(tasks['E']['observations']) and answers['F']['belief_sizes']==answers['E']['belief_sizes']+[0] and witnesses(tasks['F'],*solve(tasks['F'])).get('post_empty_absorption'),'actual post-empty absorption');actual.add('post_empty_append_absorption')
  allprops={p for c,t in tasks.items() for p in witnesses(t,*solve(t))}
  if {'multiworld_entailed_null','ambiguous_null_partition'}<=allprops and not answers['E']['surviving_worlds']:actual.add('null_guarantee_contrast')
  if {typed(tasks['A']['observations'][changed]['equals']),typed(tasks['B']['observations'][changed]['equals'])} in ({'true','1'},{'false','0'}):
   da,db=solve(tasks['A'])[1],solve(tasks['B'])[1];require(bool(da['prefix_survivors'][changed+1]) and bool(db['prefix_survivors'][changed+1]) and da['prefix_survivors'][changed+1]!=db['prefix_survivors'][changed+1],'actual typed alternative supports');actual.add('typed_equals_counterfactual')
  for c,r in base.items():
   a,d=solve(tasks[c]);components.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'context':c,'variant':variant,'difficulty':r['difficulty'],'structural_difficulty':r['structural_difficulty'],'initial_worlds':len(tasks[c]['worlds']),'prefix_counts':[len(x) for x in d['prefix_survivors']],'strict_nonempty_steps':sum(0<len(y)<len(x) for x,y in zip(d['prefix_survivors'],d['prefix_survivors'][1:])),'distinct_final_payloads':payload_count(tasks[c],a['surviving_worlds']),'query_distinct_values':[q['distinct_values'] for q in a['queries']],'properties':sorted(witnesses(tasks[c],a,d))})
  controls.append({'family':family,'variant':variant,'base_sample_hashes':{c:[r['sample_id'],r['record_hash']] for c,r in base.items()},'core_controls':sorted(actual),'target_sha256_by_context':{c:digest(a) for c,a in answers.items()}})
 base=aligned[('A','identity',0,'forward')];t=normalized(base);graph=incidence(t,'observation_control','observation_auxiliary');profile=base['structural_family_profile']
 valid={'cyclic_uniform':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))==1,'acyclic_branching':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']>=3,'cyclic_weighted':graph['cycle_rank']>0 and len(set(graph['positive_cell_weights']))>1,'typed_path':graph['components']==1 and graph['cycle_rank']==0 and graph['max_degree']==2}
 require(profile in valid and valid[profile],'actual public profile factuality')
 return actual,{'family':family,'profile':profile,'split':base['split'],'actual_graph':graph,'controls':controls,'component_metrics':components}

def probe_evidence(e,byid,pred):
 require(e['scope']=='audit_probe' and e['name'] in ('world_order_invariance','world_id_rename_equivariance') and e['source_sample_id'] in byid,'actual declared probe scope/source/name')
 source=byid[e['source_sample_id']];t=source['semantic_task'];u=e['probe_task'];expected=copy.deepcopy(pred[source['sample_id']]['expected_target']);a,d=solve(u);b,bd=P.own_prior_bitmask_solve(u);require(canon(a)==canon(b),'actual probe typed recomputation')
 require(source['source_family_key']==e['source_family_key'] and source['auxiliary_scalar_variant']==e['auxiliary_scalar_variant'],'probe actual family/scalar variant')
 require(canon(t['observations'])==canon(u['observations']) and canon(t['queries'])==canon(u['queries']),'probe fixed observations/queries')
 if e['name']=='world_order_invariance':
  require(sorted(canon(w) for w in t['worlds'])==sorted(canon(w) for w in u['worlds']) and canon(t['worlds'])!=canon(u['worlds']),'probe actual nonidentity world permutation')
 else:
  require([canon(w['facts']) for w in t['worlds']]==[canon(w['facts']) for w in u['worlds']],'ID probe payload/list order');mp={a['world_id']:b['world_id'] for a,b in zip(t['worlds'],u['worlds'])};require(len(mp)==len(set(mp.values()))==len(t['worlds']) and any(k!=v for k,v in mp.items()),'actual world-ID bijection');expected['surviving_worlds']=sorted(mp[w] for w in expected['surviving_worlds'])
  require(d['prefix_survivors']==[sorted(mp[w] for w in p) for p in pred[source['sample_id']]['prefix_supports']['prefix_survivors']],'probe exact prefix ID equivariance')
 require(canon(a)==canon(expected),'actual probe full expected output');require(canon(e['producer_reference_target'])==canon(a),'untrusted stored probe target checked independently')
 require(e.get('added_records')==e.get('added_families')==0,'executed probe adds no supplied corpus/families')
 return {'actual_executor':ACTOR,'probe_id':e['probe_id'],'name':e['name'],'source_sample_id':source['sample_id'],'source_hash':source['record_hash'],'probe_input_sha256':digest(u),'actual_expected_output_sha256':digest(a),'actual_prefix_ids':d['prefix_survivors'],'supplied_corpus_credit':0}

def payload(q,a,d):
 full={k:v for k,v in a.items() if k!='query_id'};cat='empty' if not a['counts'] else 'entailed' if a['entailed'] else 'ambiguous'
 return {'audit_category':cat,'final_survivors':d['prefix_survivors'][-1],'full_query_payload':full,'guaranteed_null_context':cat if a['guaranteed'] is None else 'nonnull','prefix_counts':[len(x) for x in d['prefix_survivors']],'prefix_survivors':d['prefix_survivors'],'value_types':['null' if b['value'] is None else {'bool':'boolean','int':'integer','str':'string'}[type(b['value']).__name__] for b in a['counts']]}
def verify_histograms(records,pred,manifest):
 groups=collections.defaultdict(list)
 for r in records:groups[(r['source_family_key'],r['controlled_context'],r['auxiliary_scalar_variant'],r['renderer'])].append(r)
 indexed={(e['source_family_key'],e['controlled_context'],e['auxiliary_scalar_variant'],e['renderer']):e for e in manifest['actual_conditional_query_histograms']};require(len(indexed)==len(groups)==128,'actual conditional group count128')
 reports=[]
 for key,rs in groups.items():
  require(len(rs)==27 and {r['schedule_index'] for r in rs}==set(range(27)),'actual complete27 schedules');e=indexed[key];require(e['actual_schedule_indices']==list(range(27)),'manifest actual schedule list')
  computed={}
  for names in [('id',),('position',),('fact',),('id','position'),('id','fact'),('position','fact'),('id','position','fact')]:
   roles=collections.defaultdict(collections.Counter);outs=collections.defaultdict(collections.Counter)
   for r in rs:
    for assignment in r['query_role_assignments']:
     i=assignment['position'];q=r['semantic_task']['queries'][i];values={'id':q['query_id'],'position':str(i),'fact':q['fact']};bucket=canon([values[n] for n in names]);roles[bucket][assignment['role']]+=1;outs[bucket][canon(payload(q,pred[r['sample_id']]['expected_target']['queries'][i],pred[r['sample_id']]['prefix_supports']))]+=1
   count={1:9,2:3,3:1}[len(names)];require(len(roles)==3**len(names) and all(dict(v)=={role:count for role in ROLES} for v in roles.values()),'actual marginal/pair/triple role counts');require(len({canon(v) for v in outs.values()})==1,'actual full typed outcome/provenance nuisance balance')
   computed[','.join(names)]={'actual_role_histograms':{k:dict(v) for k,v in roles.items()},'actual_full_typed_outcome_provenance_histograms':{k:dict(v) for k,v in outs.items()}}
  require(canon(computed)==canon(e['histograms']) and e['complete_positive_triple_support'] is True,'all untrusted manifest conditional histograms independently exact')
  reports.append({'group':list(key),'records':27,'actual_histograms_sha256':digest(computed),'manifest_histogram_entry_sha256':digest(e),'actual_full_typed_provenance_balance':True})
 return reports

def regressions(byid,index,pred,ch):
 owner=next(r for r in byid.values() if r['controlled_context']=='A' and r['auxiliary_scalar_variant']=='identity' and r['schedule_index']==0 and r['renderer']=='forward')
 original=next(c for c in owner['family_control_claims'] if c.get('name',c.get('property')) in INHERITED);names=sorted(INHERITED);receipts=[]
 for name in names:
  claim=copy.deepcopy(original);claim['name']=name;valid=resolve(owner,claim,byid,index,pred,ch);receipts.append({'name':'legitimate_nonmember_owner_'+name,'accepted':True,'resolved':valid})
  # Shared control ID across two truthful names is intentionally legitimate.
  require(owner['sample_id'] not in claim['pair_sample_ids'],'actual chosen nonmember owner')
  def negative(title,c=claim,o=owner,idx=index,expected=True):
   try:resolve(o,c,byid,idx,pred,ch)
   except Exception as error:receipts.append({'name':title+'_'+name,'rejected':True,'owner_sample_id':o['sample_id'],'claim':c,'actual_error':str(error)})
   else:raise ValueError('negative resolver mutation accepted '+title+' '+name)
  for key in ('pair_sample_ids','evidence_manifest','control_id'):
   c=copy.deepcopy(claim);del c[key];negative('missing_'+key,c)
  c=copy.deepcopy(claim);c['pair_sample_ids'].reverse();negative('reversed_pair',c)
  c=copy.deepcopy(claim);c['scope']='audit_probe';c['probe_id']='unexecuted';negative('probe_substitution',c)
  c=copy.deepcopy(claim);c['evidence_manifest']='missing.json';negative('missing_actual_manifest',c)
  c=copy.deepcopy(claim);c['source_family_key']='foreign';negative('foreign_claim_family',c)
  c=copy.deepcopy(claim);c['property']=c['name'];negative('both_name_and_property',c)
  c=copy.deepcopy(claim);c['pair_record_hashes']=['optional_not_canonical'];negative('extra_canonical_field',c)
  for field in ('auxiliary_scalar_variant','schedule_index','renderer','source_family_key'):
   foreign=next(r for r in byid.values() if r[field]!=owner[field] and r['controlled_context']=='E' and len(r['semantic_task']['observations'])==2)
   foreign2=next(r for r in byid.values() if r['controlled_context']=='G' and all(r[k]==foreign[k] for k in ('source_family_key','auxiliary_scalar_variant','schedule_index','renderer')))
   c=copy.deepcopy(claim);c['pair_sample_ids']=[foreign['sample_id'],foreign2['sample_id']];negative('foreign_owner_'+field,c)
  foreign=next(r for r in byid.values() if r['controlled_context']=='F' and all(r[k]==owner[k] for k in ('source_family_key','auxiliary_scalar_variant','schedule_index','renderer')));partner=next(r for r in byid.values() if r['controlled_context']=='H' and all(r[k]==owner[k] for k in ('source_family_key','auxiliary_scalar_variant','schedule_index','renderer')));c=copy.deepcopy(claim);c['pair_sample_ids']=[foreign['sample_id'],partner['sample_id']];negative('wrong_actual_length_pair',c)
  key=(claim['control_id'],name,canon(claim['pair_sample_ids']));entry=copy.deepcopy(index[key][0])
  for title,changed in [('missing_named_entry',[]),('duplicate_lookup',[entry,entry])]:
   ix=dict(index);ix[key]=changed;negative(title,idx=ix)
  for field,value in [('exact_candidate_sha256','0'*64),('observation_length',3),('observation_length',True),('source_family_key','foreign'),('auxiliary_scalar_variant','transposed'),('left_prefix_counts',[999]),('right_complete_query_payloads',[])]:
   e=copy.deepcopy(entry);e[field]=value;ix=dict(index);ix[key]=[e];negative('tampered_'+field+'_'+type(value).__name__,idx=ix)
  e=copy.deepcopy(entry);del e['exact_candidate_sha256'];ix=dict(index);ix[key]=[e];negative('missing_entry_candidate_binding',idx=ix)
  c=copy.deepcopy(claim);c['property']=c.pop('name');receipts.append({'name':'permitted_property_alias_'+name,'accepted':True,'resolved':resolve(owner,c,byid,index,pred,ch)})
 return receipts

def main():
 started=time.time();pre=loadjson((ROOT/PRE).read_bytes());before={p:sha(ROOT/p) for p in pre['file_bindings']};require(before==pre['file_bindings'],'all exact source/data/public/own prediction bytes before execution');rb={p:sha(pathlib.Path(p)) for p in pre['runtime_dependencies']};require(rb==pre['runtime_dependencies'],'actual runtime before')
 data=rows(ROOT/CAND);manifest=loadjson((ROOT/MAN).read_bytes());old=rows(ROOT/OLDC);view={r['sample_id']:r for r in rows(ROOT/I.VIEW)};pred={r['sample_id']:r for r in rows(ROOT/PRED)};byid={r['sample_id']:r for r in data};oldlookup={I.input_key(r):r for r in old};ch=before[CAND]
 require(len(data)==len(byid)==len(pred)==len(view)==3456 and manifest['exact_candidate_sha256']==ch,'actual cohorts/exact raw PENDING candidate binding')
 defects=collections.defaultdict(list);derived={};resolved={};families=collections.defaultdict(list);preservation=collections.Counter();property_counts=collections.Counter();row_assertions={};all_diff_fields=collections.Counter()
 for r in data:
  sid=r['sample_id'];families[r['source_family_key']].append(r)
  try:
   require(P.record_hash(r)==r['record_hash'],'actual full canonical record hash');require(canon({k:r[k] for k in I.FIELDS})==canon(view[sid]),'exact fresh stripped candidate join');I.parse_prompt(r)
   require(canon(r['verified_target'])==canon(pred[sid]['expected_target']),'exact typed stored target vs own pre-gold expectation');a,d=solve(r['semantic_task']);require(canon(a)==canon(pred[sid]['expected_target']) and d['prefix_survivors']==pred[sid]['prefix_supports']['prefix_survivors'],'canonical-stage own target/prefix recomputation')
   P.decode(r);require(type(r['source_family_index']) is int and r['source_family_index'] in (12,13,18,19),'public source index');idx=r['source_family_index'];require(r['split']==('train' if idx in (12,13) else 'dev_challenge'),'source family split custody');require(r['structural_family_profile']=={12:'cyclic_uniform',13:'acyclic_branching',18:'cyclic_weighted',19:'typed_path'}[idx],'public index/profile binding')
   require(type(r['difficulty']) is type(r['structural_difficulty']) is int and r['difficulty'] in (2,3) and r['structural_difficulty'] in (2,3),'honest bounded difficulty fields');require(r['controlled_context'] not in 'GH' or r['difficulty']==r['structural_difficulty']==2,'G/H component difficulty')
   require(r['review_status']=='PENDING' and r['review_receipt_ids']==[] and r['target_format']=='structured','unapproved original candidate/status/render format')
   require(type(r['generator_version']) is str and bool(r['generator_version']) and type(r['producer_agent_id']) is str and type(r['producer_role']) is str,'descriptive production/version strings')
   o=oldlookup[I.input_key(r)];unchanged=['semantic_task','rendered_input','verified_target','split','track_id','difficulty','structural_difficulty','skill_tags','property_witnesses','source_family_index','source_family_key','structural_family_profile','source_provenance','generation_seed','oracle_version','target_format','scope_limitation']
   for k in unchanged:require(canon(r[k])==canon(o[k]),'actual unchanged V3 semantic/skill/difficulty/source '+k);preservation[k]+=1
   changed=[k for k in set(r)|set(o) if canon(r.get(k))!=canon(o.get(k))];all_diff_fields.update(changed);require(set(changed)<={'sample_id','semantic_family_id','semantic_parent_id','generator_version','record_hash','family_control_claims'},'unexpected canonical V3/V4 metadata change')
   require(r['generator_version']!=o['generator_version'] and r['sample_id']!=o['sample_id'] and r['record_hash']!=o['record_hash'],'fresh version/ID/full hash')
   ws=witnesses(r['semantic_task'],a,d);tags=r['skill_tags'];require(type(tags) is list and len(tags)==len(set(tags)) and GENERIC<=set(tags)<=GENERIC|PROPERTIES,'public generic/property tags only');attached=set(tags)-GENERIC;require(set(r['property_witnesses'])==attached,'exact tag/locator key equality');locs={p:locator_valid(p,r['property_witnesses'][p],ws.get(p,[]),r['semantic_task']) for p in attached};property_counts.update(attached);derived[sid]=locs
   require(payload_count(r['semantic_task'],{w['world_id'] for w in r['semantic_task']['worlds']})==len(r['semantic_task']['worlds']),'actual distinct full fact payloads, not world-ID duplicates')
   row_assertions[sid]={'target_sha256':digest(a),'prefix_supports_sha256':digest(d),'actual_properties':sorted(attached),'own_locator_evidence_sha256':digest(locs),'prior_v3_sample_id':o['sample_id'],'prior_v3_record_hash':o['record_hash'],'all_semantic_skill_difficulty_fields_unchanged':True}
  except Exception as e:defects[sid].append({'severity':'high','type':'row_semantic_or_metadata_defect','evidence':str(e),'fix':'Correct this fresh candidate in a new immutable version; preserve existing hashes and first judgments.'})
 index=collections.defaultdict(list);entryproofs=[]
 for e in manifest['supplied_same_table_v2_pairs']:index[entry_key(e)].append(e)
 for e in manifest['supplied_same_table_v2_pairs']:entryproofs.append(validate_entry(e,byid,pred,ch))
 require(len(entryproofs)==1728 and len({tuple(x['pair_sample_ids']) for x in entryproofs})==864,'1728 truthful named entries covering864 supplied actual pairs')
 structure=[];actualnames={};global_errors=[]
 for f,rs in families.items():
  try:names,report=family_semantics(f,rs,pred);actualnames[f]=names;structure.append(report)
  except Exception as e:
   global_errors.append({'family':f,'error':str(e)})
   for r in rs:defects[r['sample_id']].append({'severity':'high','type':'family_control_defect','evidence':str(e),'fix':'Supply complete factual same-family controls in a new immutable candidate.'})
 probes=[probe_evidence(e,byid,pred) for e in manifest['executed_audit_probes']];probeindex={(e['probe_id'],e['name']):e for e in manifest['executed_audit_probes']};require(len(probes)==len(probeindex)==16 and manifest['probes_add_corpus_records']==manifest['probes_add_source_families']==0,'actual16 separately executed zero-credit probes')
 inherited_count=nonmembers=aggregate_count=0
 for r in data:
  sid=r['sample_id'];resolved[sid]=[]
  try:
   names=[]
   for c in r['family_control_claims']:
    require(type(c) is dict and set(c)<=P.CLAIM_FIELDS and len(set(c)&{'name','property'})==1 and c.get('scope') in ('corpus_variant','audit_probe'),'all closed canonical claims');name=c.get('name',c.get('property'));names.append(name);require(name in actualnames[r['source_family_key']],'actual independently reconstructed family relationship, not pointer proof');require(c.get('source_family_key',r['source_family_key'])==r['source_family_key'] and c.get('evidence_manifest')==MAN,'actual same-family supplied manifest pointer')
    if name in INHERITED:
     evidence=resolve(r,c,byid,index,pred,ch);resolved[sid].append(evidence);inherited_count+=1;nonmembers+=evidence['owner_not_pair_member']
    elif c['scope']=='audit_probe':
     e=probeindex.get((c.get('probe_id'),name));require(e is not None and e['source_family_key']==r['source_family_key'] and e['auxiliary_scalar_variant']==r['auxiliary_scalar_variant'],'actual named executed probe same family/variant');require('pair_sample_ids' not in c,'probe cannot be supplied pair')
    else:
     require('probe_id' not in c,'supplied claim cannot be unexecuted probe');aggregate_count+=name in AGGREGATE
     if 'pair_sample_ids' in c:
      ids=c['pair_sample_ids'];require(type(ids) is list and len(ids)==2 and len(set(ids))==2 and all(x in byid and byid[x]['source_family_key']==r['source_family_key'] for x in ids),'actual nominated same-family supplied pair')
      if name=='auxiliary_scalar_transposition_invariance':
       left,right=[byid[x] for x in ids];require({left['auxiliary_scalar_variant'],right['auxiliary_scalar_variant']}=={'identity','transposed'} and all(left[k]==right[k]==r[k] for k in ('controlled_context','schedule_index','renderer')),'nominated scalar pair owner coordinates');require(canon(transposition(left['semantic_task'],left['auxiliary_scalar_control']))==canon(right['semantic_task']),'nominated pair actual T')
   require(len(names)==len(set(names)) and INHERITED|AGGREGATE<=set(names),'required specialized/aggregate identifiers once per row')
  except Exception as e:defects[sid].append({'severity':'high','type':'claim_resolution_defect','evidence':str(e),'fix':'Repair actual pair/owner/manifest binding in a fresh candidate; do not substitute annotations or probe coverage.'})
 require(inherited_count==6912 and aggregate_count==10368,'actual6912 inherited and10368 aggregate row claims')
 hist=verify_histograms(data,pred,manifest);reg=regressions(byid,index,pred,ch)
 # Input-only controls were frozen from inputs before metadata. Bind exact actual pair sets now.
 ownco=loadjson((ROOT/'audits/diversity/belief-gap-v4-input-only-adversarial-r2.json').read_bytes())['cohort_reconstruction']
 def pairset(xs,key):return {tuple(sorted(x[key])) for x in xs}
 require(pairset(manifest['field_order_pairs'],'pair_sample_ids')==pairset(ownco['field_order_pairs'],'sample_ids'),'all1728 actual manifest field-pair joins')
 for e in manifest['field_order_pairs']:
  l,r=[byid[x] for x in e['pair_sample_ids']];require(canon(l['semantic_task'])==canon(r['semantic_task']) and l['rendered_input']!=r['rendered_input'] and e['typed_task_sha256']==digest(l['semantic_task']) and e['complete_target_sha256']==digest(pred[l['sample_id']]['expected_target']),'manifest typed field pair/hash facts')
 require(pairset(manifest['within_context_scalar_pairs'],'pair_sample_ids')==pairset(ownco['scalar_invariance_pairs'],'sample_ids'),'all1728 actual scalar-pair joins')
 for e in manifest['within_context_scalar_pairs']:
  l,r=[byid[x] for x in e['pair_sample_ids']];a=pred[l['sample_id']]['expected_target'];d=pred[l['sample_id']]['prefix_supports'];require(e['candidate_sha256']==ch and e['auxiliary_scalar_control']==l['auxiliary_scalar_control'] and canon(transposition(l['semantic_task'],e['auxiliary_scalar_control']))==canon(r['semantic_task']) and canon(transposition(r['semantic_task'],e['auxiliary_scalar_control']))==canon(l['semantic_task']),'actual manifest involution/control binding');require(canon(e['prefix_and_target_evidence'])==canon({'complete_target':a,'prefix_counts':[len(x) for x in d['prefix_survivors']],'prefix_survivor_ids':d['prefix_survivors']}),'actual scalar parsed prefix/target evidence')
 require(pairset(manifest['same_observation_opposite_status_pairs'],'pair_sample_ids')==pairset(ownco['same_observations_opposite_status_pairs'],'sample_ids'),'all864 same-observation opposite-status joins')
 for e in manifest['same_observation_opposite_status_pairs']:
  l,r=[byid[x] for x in e['pair_sample_ids']];require(e['candidate_sha256']==ch and canon(e['ordered_identical_observations'])==canon(l['semantic_task']['observations'])==canon(r['semantic_task']['observations']) and canon(e['ordered_identical_queries'])==canon(l['semantic_task']['queries'])==canon(r['semantic_task']['queries']),'actual opposite pair input/candidate binding')
  for side,record in [('left',l),('right',r)]:
   p=pred[record['sample_id']];d=p['prefix_supports'];require(canon(e[side])==canon({'complete_target':p['expected_target'],'prefix_counts':[len(x) for x in d['prefix_survivors']],'prefix_survivor_ids':d['prefix_survivors']}),'actual opposite parsed prefix/target evidence')
 require(len({canon(e['actual_graph']['weighted_canonical_signature']) for e in structure})==4,'four actual distinct weighted source profiles')
 unweighted=collections.defaultdict(list)
 for e in structure:unweighted[canon(e['actual_graph']['unweighted_canonical_signature'])].append(e['family'])
 component_report={'actual_profiles':structure,'shared_unweighted_motifs':dict(unweighted),'difficulty_levels':dict(collections.Counter(r['difficulty'] for r in data)),'semantics_preservation_counts':dict(preservation),'actual_changed_metadata_fields':dict(all_diff_fields),'source_growth':0,'semantic_task_growth':0,'target_changes':0,'split_changes':0,'difficulty_changes':0,'skill_tag_or_locator_changes':0}
 limits=['Only four original source families,32 base contexts/64 scalar cases; all3456 original student inputs/targets were already exposed.','Two development families are exposed; cyclic uniform/weighted profiles share an unweighted motif; nuisance/scalar/style views are correlated.','Status majority75percent and usefulness-false87.5percent persist; finite whole-table/observation lookup fits reported in own input-only audit.','G/H remain honest Level2, singleton in two families, with non-strict filtering; repeats/empty substeps do not establish higher difficulty.','Published strict ID+name+orderedpair lookup applies to the two inherited identifiers; descriptive IDs of other aggregate controls are not subject to an invented exact closed manifest/global-unique-ID rule.','No arbitrary encoded-alias metadata privacy, noisy/Bayesian authority, inspection utility, learned shortcut, sealed OOD, architecture or efficacy certification.','Producer/version/provenance fields are untrusted assertions. This source-limited review does not reproduce private generation or authenticate claimed producer execution.','Common model/public-guided revision and own original gold/solver/prediction/resolver reuse correlate errors.']
 evidence_report={'actual_execution_id':ACTOR,'candidate_sha256':ch,'manifest_sha256':before[MAN],'semantic_contexts':32,'scalar_base_cases':64,'schedules_per_conditional_group':27,'renderer_styles':2,'all_exact_targets_hashes_prompt_joins':3456,'inherited_claims_resolved':inherited_count,'legitimate_nonmember_claims_resolved':nonmembers,'named_entries_checked':len(entryproofs),'actual_same_table_pair_count':864,'unique_named_lookup_count':len(index),'control_IDs_may_share_names':len({e['control_id'] for e in manifest['supplied_same_table_v2_pairs']}),'entry_proofs':entryproofs,'row_properties_count':dict(property_counts),'all3_aggregate_controls_per_row':aggregate_count,'conditional_groups_exact':hist,'actual_executed_probes':probes,'resolver_regressions':reg,'original_core_control_and_difficulty_evidence':component_report,'global_errors':global_errors,'disclosure':DISCLOSURE,'residual_limits':limits,'dataset_or_publication_authority':False}
 ep='audits/diversity/belief-gap-v4-canonical-adversarial-r2.json';freeze(ROOT/ep,jsonbytes(evidence_report));eh=sha(ROOT/ep)
 receipts=[]
 for r in data:
  sid=r['sample_id'];issues=defects[sid];decision='REVISE' if issues else 'ACCEPT';receipt={'receipt_id':'adv-belief-gap-v4-canonical-'+digest([sid,r['record_hash'],pre['new_source_sha256']])[:24],'sample_id':sid,'record_hash':r['record_hash'],'reviewer_role':'Adversarial Red-Team','additional_audit_role':'Diversity and leakage auditor','reviewer_execution_id':ACTOR,'review_stage':'canonical comparison after all input-only first predictions/judgments frozen','scores':{'semantic_correctness':5 if not issues else 2,'target_correctness':5 if not issues else 2,'instruction_clarity':5,'reasoning_validity':5 if not issues else 2,'generalization_value':3,'difficulty_quality':3,'diversity':3,'shortcut_resistance':3},'critical_defects':[x['type'] for x in issues],'issues':issues,'decision':decision,'confidence':0.97,'evidence':{'candidate_sha256':ch,'manifest_sha256':before[MAN],'own_before_gold_predictions_sha256':before[PRED],'own_input_first_judgments_sha256':before['reviews/adversarial/belief-gap-v4.input_first_judgments.jsonl'],'row_checks':row_assertions.get(sid,{}),'owner_scoped_claim_resolutions':resolved.get(sid,[]),'canonical_audit_path':ep,'canonical_audit_sha256':eh,'input_only_shortcut_audit_sha256':before['audits/shortcut_detection/belief-gap-v4-input-only-adversarial-r2.json'],'stored_target_approval_alone':False,'producer_or_formal_private_or_other_verdict_reads':False},'scope_limitation':DISCLOSURE+' ACCEPT is scoped to actual finite semantics, factual tags and supplied interface; it is not a publication, unseen evaluation or efficacy approval.'}
  receipts.append(receipt)
 first='reviews/adversarial/belief-gap-v4.canonical_first_judgments.jsonl';final='reviews/adversarial/belief-gap-v4.jsonl';raw=jlbytes(receipts);freeze(ROOT/first,raw);freeze(ROOT/final,raw);print('FIRST_CANONICAL_FROZEN '+canon({'first_sha256':sha(ROOT/first),'final_sha256':sha(ROOT/final),'decisions':dict(collections.Counter(r['decision'] for r in receipts))}),flush=True)
 after={p:sha(ROOT/p) for p in pre['file_bindings']};ra={p:sha(pathlib.Path(p)) for p in pre['runtime_dependencies']};require(before==after and rb==ra,'source/data/public/own artifacts/runtime unchanged after actual full canonical execution')
 execution={'actual_author_and_execution_id':ACTOR,'completed':True,'phase':'FINAL_CANONICAL_COMPLETE','source_sha256':pre['new_source_sha256'],'preexec_sha256':sha(ROOT/PRE),'file_bindings_before':before,'file_bindings_after':after,'runtime_dependencies_before':rb,'runtime_dependencies_after':ra,'outputs':{p:sha(ROOT/p) for p in (ep,first,final)},'actual_rows_reviewed':3456,'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'actual_inherited_claims':inherited_count,'actual_named_entries':1728,'actual_supplied_same_table_pairs':864,'actual_regression_rejects':sum(x.get('rejected',False) for x in reg),'actual_regression_legitimate_accepts':sum(x.get('accepted',False) for x in reg),'actual16_probes_reexecuted':16,'actual_typed_target_hash_prompt_checks':3456,'actual_row_property_locators':sum(property_counts.values()),'actual_conditional_full27_groups':128,'prior_semantic_skill_difficulty_target_preservation':component_report,'wall_seconds':time.time()-started,'source_limited_review_not_generation_reproduction':True,'private_or_other_reviewer_reads':False,'disclosure':DISCLOSURE,'residual_limits':limits,'publication_approval':False}
 xp='audits/execution/belief-gap-v4-canonical-adversarial-r2.json';freeze(ROOT/xp,jsonbytes(execution));print('FINAL '+canon({'execution_sha256':sha(ROOT/xp),'outputs':execution['outputs'],'counts':{k:execution[k] for k in ('actual_rows_reviewed','decisions','actual_inherited_claims','actual_regression_rejects','actual_regression_legitimate_accepts','actual16_probes_reexecuted','actual_row_property_locators')}}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  error={'actual_actor':ACTOR,'source_sha256':sha(pathlib.Path(__file__)),'completed':False,'actual_failure_type':type(e).__name__,'actual_failure':str(e),'no_other_reviewer_or_private_checker_read':True}
  freeze(ROOT/'audits/execution/belief-gap-v4-canonical-adversarial-r2.failure.json',jsonbytes(error));print('FAILURE '+canon(error),flush=True);raise
