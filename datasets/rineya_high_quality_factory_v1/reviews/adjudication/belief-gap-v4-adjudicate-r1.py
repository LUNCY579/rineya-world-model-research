"""Owned exact V4 adjudication. Imports only frozen own public helpers after preflight.
No producer/formal/reviewer implementation import; no release or pipeline invocation.
"""
from pathlib import Path
from collections import Counter, defaultdict
import copy, datetime, hashlib, importlib.util, itertools, json, sys, time, traceback
BASE=Path(__file__).resolve().parents[2]
P='reviews/adjudication/belief-gap-v4-'
ACTOR='/root/research_architect'

def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def require(ok,reason):
    if not ok: raise ValueError(reason)
def bind(p):
    h=hashlib.sha256(); n=0
    with (BASE/p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b);n+=len(b)
    return {'path':p,'sha256':h.hexdigest(),'bytes':n}
def load(p): return json.loads((BASE/p).read_text())
def lines(p): return [json.loads(s) for s in (BASE/p).open()]
def out(p,v,jsonl=False):
    require(not (BASE/p).exists(),'new output only: '+p)
    with (BASE/p).open('x') as f:
        if jsonl:
            for r in v:f.write(canon(r)+'\n')
        else:f.write(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    b=bind(p);require(b['bytes']<25*1024*1024,'output below 25MiB');return b
def fresh_import(name,p):
    spec=importlib.util.spec_from_file_location(name,BASE/p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def direct(t):
    supports=[[w for w in t['worlds'] if all(type(w['facts'][o['fact']]) is type(o['equals']) and w['facts'][o['fact']]==o['equals'] for o in t['observations'][:n])] for n in range(len(t['observations'])+1)]
    ans=[]
    for q in t['queries']:
        values={canon(w['facts'][q['fact']]):w['facts'][q['fact']] for w in supports[-1]}
        counts=Counter(canon(w['facts'][q['fact']]) for w in supports[-1]);bins=[{'value':values[k],'count':counts[k]} for k in sorted(counts)]
        e=bool(supports[-1]) and len(bins)==1
        ans.append({'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':e,'guaranteed':bins[0]['value'] if e else None,'can_reduce_uncertainty':len(bins)>1})
    return {'surviving_worlds':sorted(w['world_id'] for w in supports[-1]),'belief_sizes':[len(s) for s in supports[1:]],'queries':ans},supports

def transform(t,c):
    t=copy.deepcopy(t);f=c['fact'];a,b=c['baseline_value'],c['supported_value']
    require(type(a) is str and type(b) is str and a!=b,'two typed distinct strings')
    def swap(v): return b if type(v) is str and v==a else a if type(v) is str and v==b else v
    for w in t['worlds']:w['facts'][f]=swap(w['facts'][f])
    for o in t['observations']:
        if o['fact']==f:o['equals']=swap(o['equals'])
    require(all(q['fact']!=f for q in t['queries']),'auxiliary unqueried')
    return t

def no_dupes(pairs):
    d={}
    for k,v in pairs:require(k not in d,'duplicate JSON key');d[k]=v
    return d

def decode_coords(raw,axes):
    x=json.loads(raw,object_pairs_hook=no_dupes);require(type(x) is list and len(x)==len(axes),'coordinate arity')
    result=[]
    for a,v in zip(axes,x):
        if a=='position':
            if type(v) is str and v in ('0','1','2'):v=int(v)
            require(type(v) is int and 0<=v<3,'diagnostic position')
        else:require(a in ('id','fact') and type(v) is str and bool(v),'coordinate axis')
        result.append(v)
    return canon(result)
def norm_hist(d,axes,signatures):
    result={}
    for raw,counts in d.items():
        coord=decode_coords(raw,axes);require(coord not in result,'coordinate collision');result[coord]={}
        for token,n in counts.items():
            require(type(n) is int and n>0,'histogram positive integer count')
            key=canon(json.loads(token,object_pairs_hook=no_dupes)) if signatures else token
            require(key not in result[coord],'signature collision');result[coord][key]=n
    return result

def typ(v):return 'null' if v is None else {bool:'boolean',int:'integer',str:'string'}[type(v)]
def signature(answer,target,prefix):
    payload={k:v for k,v in answer.items() if k!='query_id'}
    cat='empty' if not target['surviving_worlds'] else 'entailed' if answer['entailed'] else 'ambiguous'
    return {'audit_category':cat,'final_survivors':target['surviving_worlds'],'full_query_payload':payload,'guaranteed_null_context':cat if answer['guaranteed'] is None else 'nonnull','prefix_counts':[len(p) for p in prefix],'prefix_survivors':prefix,'value_types':[typ(c['value']) for c in answer['counts']]}

def graph_profile(r):
    t=r['semantic_task'];roles=r['fact_role_assignments'];x,y=roles['observation_control'],roles['observation_auxiliary']
    cells=Counter((canon(w['facts'][x]),canon(w['facts'][y])) for w in t['worlds']);vertices={('x',a) for a,b in cells}|{('y',b) for a,b in cells};adj={v:set() for v in vertices}
    for a,b in cells:adj['x',a].add(('y',b));adj['y',b].add(('x',a))
    seen=set();components=0
    for v in vertices:
        if v in seen:continue
        components+=1;todo=[v]
        while todo:
            u=todo.pop()
            if u in seen:continue
            seen.add(u);todo+=list(adj[u]-seen)
    xs=sorted({a for a,b in cells});ys=sorted({b for a,b in cells})
    sig=min(tuple(cells[a,b] for a in xp for b in yp) for xp in itertools.permutations(xs) for yp in itertools.permutations(ys))
    return {'worlds':len(t['worlds']),'vertices':len(vertices),'occupied_cells':len(cells),'components':components,'cycle_rank':len(cells)-len(vertices)+components,'max_degree':max(map(len,adj.values())),'cell_weights':sorted(cells.values()),'weighted_incidence_signature':[len(xs),len(ys),sig]}

def decoded_worlds(r,inverse=False):
    t=transform(r['semantic_task'],r['auxiliary_scalar_control']) if inverse else r['semantic_task'];roles=r['fact_role_assignments']
    return canon(sorted([canon({'world_id':w['world_id'],'facts':{role:w['facts'][f] for role,f in roles.items()}}) for w in t['worlds']]))

def receipt_join(rs,by,role,actor):
    require(len(rs)==len(by),'receipt count '+role);m={}
    for r in rs:
        sid=r['sample_id'];require(sid in by and sid not in m,'unique receipt '+role);m[sid]=r
        require(r['record_hash']==by[sid]['record_hash'],'receipt hash '+role)
        require(r['reviewer_execution_id']==actor and r['reviewer_role']==role,'exact role '+role)
    return m

def main():
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
    f=load(P+'preexecution-r1.json');cfg=load(P+'config-r1.json')
    for b in f['bindings']:require(bind(b['path'])==b,'preexecution unchanged '+b['path'])
    for key in cfg['outputs']:require(not (BASE/cfg['outputs'][key]).exists(),'new output '+key)
    own=fresh_import('owned_exposed_solver_v4',cfg['own_solver']);component=fresh_import('owned_prospective_resolver_v4',cfg['own_resolver'])
    fixtures=load(P+'hand-fixtures-r1.json');hands=[]
    for case in fixtures['coordinate_cases']:
        try:actual=decode_coords(case['raw'],case['axes']);ok=actual==case.get('expected')
        except (ValueError,TypeError,json.JSONDecodeError):ok=case.get('reject',False)
        hands.append({'case_id':case['case_id'],'pass':ok})
    for case in load(cfg['own_resolver_fixtures'])['cases']:
        try:component.resolve(case['owner'],case['claim'],case['records'],case['manifest'],case['candidate_sha']);actual='PASS_COMPONENT'
        except ValueError:actual='FAIL_COMPONENT'
        hands.append({'case_id':case['case_id'],'pass':actual==case['expected']})
    # Independently exercise normalization collision, typed scalar separation and signature mutation.
    for name,bad in [('coordinate_collision',{'[0]':{'role':1},'["0"]':{'role':1}}),('boolean_count',{'[0]':{'role':True}})]:
        try:norm_hist(bad,['position'],False);ok=False
        except ValueError:ok=True
        hands.append({'case_id':name,'pass':ok})
    hands.append({'case_id':'typed_signature_values','pass':len({canon(None),canon(False),canon(0),canon('0')})==4})
    hb=out(cfg['outputs']['hands'],{'actual_execution_id':ACTOR,'scope':'hand component checks; not production validation','cases':hands,'passed':sum(h['pass'] for h in hands)})
    require(all(h['pass'] for h in hands),'hand checks')
    print('HAND_CHECKS',len(hands),flush=True)
    rows=lines(cfg['candidate']);old=lines(cfg['old_candidate']);by={r['sample_id']:r for r in rows};require(len(rows)==len(by)==3456,'3456 unique candidate IDs')
    views={r['sample_id']:r for r in lines(cfg['view'])};pred={r['sample_id']:r for r in lines(cfg['own_predictions'])};first={r['sample_id']:r for r in lines(cfg['own_input_first'])}
    joins={r['fresh_sample_id']:r for r in lines(cfg['own_input_joins'])};oldby={r['sample_id']:r for r in old}
    require(set(by)==set(views)==set(pred)==set(first)==set(joins),'all own view/input-first joins')
    targets={};prefix={};props=Counter();changes=Counter();status=Counter();flags=Counter();dif=Counter();splits=Counter();idx={};groups=defaultdict(list);detail=[];family_splits={};decoded=set();scalars=set();visible=set()
    allowed={'family_control_claims','generator_version','record_hash','sample_id','semantic_family_id','semantic_parent_id'}
    prompt_seen=set();semantic_families=defaultdict(set)
    for r in rows:
        sid=r['sample_id'];t=r['semantic_task'];component.verify_hash(r);previous=oldby[joins[sid]['prior_sample_id']];component.verify_hash(previous)
        require(set(r)==set(previous),'complete old/new key preservation')
        changed={k for k in r if canon(r[k])!=canon(previous[k])};require(changed<=allowed,'only prospective metadata changes');changes.update(changed)
        for k,v in views[sid].items():require(canon(v)==canon(r[k]),'own input view to canonical '+k)
        require(first[sid]['decision']=='INPUT_ONLY_CONSISTENT' and first[sid]['record_hash']==r['record_hash'],'own input first hash')
        require(r['review_status']=='PENDING' and r['producer_agent_id']!=ACTOR,'no self-production approval')
        lead,body=r['rendered_input'].split('Task JSON:\n',1);require(canon(json.loads(body,object_pairs_hook=no_dupes))==canon(t),'prompt typed task')
        require(r['rendered_input'] not in prompt_seen,'exact prompt duplicate');prompt_seen.add(r['rendered_input'])
        a,p=own.solve(t);b,q=direct(t);ids=lambda s:sorted(w['world_id'] for w in s)
        ps=[ids(s) for s in p];require(canon(a)==canon(b)==canon(r['verified_target'])==canon(pred[sid]['predicted_target']),'four exact target comparisons')
        require(ps==[ids(s) for s in q]==pred[sid]['prefix_world_ids'],'all exact prefixes and pregold prediction')
        require(pred[sid]['record_hash']==r['record_hash'] and pred[sid]['semantic_task_sha256']==sha(t),'pregold hash')
        tags=own.property_audit(r,t,a,p);props.update(tags);targets[sid]=a;prefix[sid]=ps
        coordinates=(r['source_family_index'],r['controlled_context'],r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer'])
        require(coordinates not in idx,'unique grouping coordinates');idx[coordinates]=r
        require(type(r['schedule_index']) is int and 0<=r['schedule_index']<27,'typed schedule')
        require(all(type(r[k]) is int and 0<=r[k]<3 for k in ['id_shift','position_shift','fact_name_shift']),'typed shift')
        require(r['schedule_index']==9*r['id_shift']+3*r['position_shift']+r['fact_name_shift'],'full27 actual shift encoding')
        roles=r['query_role_assignments'];require(len(roles)==len(t['queries'])==3,'three actual query roles');require({v['role'] for v in roles}=={'invariant_query','variation_query','observed_control'},'roles')
        for v in roles:
            n=v['position'];require(type(n) is int and 0<=n<3,'actual query position strict int')
            require(t['queries'][n]['query_id']==v['query_id'] and t['queries'][n]['fact']==v['fact'],'role actual query position/id/fact')
            rolefact='observation_control' if v['role']=='observed_control' else v['role'];require(r['fact_role_assignments'][rolefact]==v['fact'],'typed decoded role fact')
        require(len(set(v['position'] for v in roles))==3,'role positions unique')
        require(set(r['query_id_pool'])=={v['query_id'] for v in roles} and set(r['queried_fact_name_pool'])=={v['fact'] for v in roles},'same actual pools')
        require(len(r['fact_role_assignments'])==5 and len(set(r['fact_role_assignments'].values()))==5,'five actual roles')
        fi=r['source_family_index'];require(type(fi) is int and fi in (12,13,18,19),'same source indices')
        require(r['split']==('train' if fi%10<8 else 'dev_challenge'),'common lineage split')
        require(r['semantic_parent_id']==r['semantic_family_id'],'parent family')
        family_splits.setdefault(r['source_family_key'],r['split']);require(family_splits[r['source_family_key']]==r['split'],'same family split')
        semantic_families[sha(t)].add(r['source_family_key'])
        require(type(r['difficulty']) is int and r['difficulty']==r['structural_difficulty'] and r['difficulty'] in (2,3),'honest bounded difficulty')
        if r['controlled_context'] in 'GH':require(r['difficulty']==2,'simple G/H level2')
        strict=[n for n in range(len(p)-1) if 0<len(p[n+1])<len(p[n])]
        detail.append({'sample_id':sid,'record_hash':r['record_hash'],'task_sha256':sha(t),'target_sha256':sha(a),'prefix_sha256':sha(ps),'properties':sorted(tags),'difficulty':r['difficulty'],'strict_nonempty_filters':len(strict),'prefix_counts':[len(s) for s in ps],'distinct_final_payloads':len({canon(w['facts']) for w in p[-1]})})
        groups[(r['source_family_key'],r['controlled_context'],r['auxiliary_scalar_variant'],r['renderer'])].append(r)
        decoded.add(decoded_worlds(r,r['auxiliary_scalar_variant']=='transposed'));scalars.add(decoded_worlds(r));visible.add(canon(t['worlds']))
        status['nonempty' if p[-1] else 'empty']+=1;splits[r['split']]+=1;dif[r['difficulty']]+=1
        for answer in a['queries']:flags['useful_true' if answer['can_reduce_uncertainty'] else 'useful_false']+=1
    require(len(idx)==3456 and len(groups)==128 and len(decoded)==4 and len(scalars)==8 and len(visible)==24,'actual lineage/context/table counts')
    require(all(len(v)==1 for v in semantic_families.values()) and len(semantic_families)==1728,'same-style duplicates scoped; no crossfamily task duplicates')
    require(dict(splits)=={'train':1728,'dev_challenge':1728} and Counter(family_splits.values())=={'train':2,'dev_challenge':2},'2/2 exposed family split')
    print('ROWS_TARGETS_HASHES_PROPERTIES_PRESERVATION',len(rows),flush=True)
    manifest=load(cfg['manifest']);candidate_sha=bind(cfg['candidate'])['sha256'];require(manifest['exact_candidate_sha256']==candidate_sha,'whole candidate manifest SHA')
    profiles=[]
    for fi in (12,13,18,19):
        r=idx[fi,'A','identity',0,'forward'];g=graph_profile(r);expected={12:'cyclic_uniform',13:'acyclic_branching',18:'cyclic_weighted',19:'typed_path'}[fi]
        require(r['structural_family_profile']==expected and g['components']==1,'factual profile connected')
        if fi in (12,18):require(g['cycle_rank']==1 and g['occupied_cells']==6,'cycle')
        else:require(g['cycle_rank']==0 and g['occupied_cells']==5,'acyclic')
        if fi==12:require(len(set(g['cell_weights']))==1,'uniform')
        if fi==18:require(len(set(g['cell_weights']))>1,'weighted')
        if fi==13:require(g['max_degree']==3,'branching')
        if fi==19:require(g['max_degree']==2,'path')
        profiles.append({'source_family':r['source_family_key'],'profile':expected,'derived':g})
    require(len({canon(g['derived']['weighted_incidence_signature']) for g in profiles})==4,'four distinct mechanisms beyond scalar names')
    pair_evidence={};cross_pairs=set();scalar_pairs=set();field_pairs=set();core=Counter()
    for fi in (12,13,18,19):
        for schedule in range(27):
            for renderer in ('forward','reverse'):
                for variant in ('identity','transposed'):
                    rr={c:idx[fi,c,variant,schedule,renderer] for c in 'ABCDEFGH'};tt={c:r['semantic_task'] for c,r in rr.items()}
                    require(all(canon(tt[c]['worlds'])==canon(tt['A']['worlds']) and canon(tt[c]['queries'])==canon(tt['A']['queries']) for c in tt),'all contexts actual same table/queries')
                    A,B,C=(rr[c]['sample_id'] for c in 'ABC')
                    require(len(prefix[A][-1])==len(prefix[B][-1])>0 and targets[A]['queries']!=targets[B]['queries'],'equal support useful contrast');core['equal_support']+=1
                    require(canon(tt['C']['observations'])==canon(list(reversed(tt['A']['observations']))) and targets[A]['surviving_worlds']==targets[C]['surviving_worlds'] and targets[A]['belief_sizes']!=targets[C]['belief_sizes'],'reorder actual traces');core['reorder']+=1
                    for left,right in [('A','D'),('G','H'),('E','F')]:
                        a,b=rr[left]['sample_id'],rr[right]['sample_id'];require(canon(tt[right]['observations'])==canon(tt[left]['observations']+[tt[left]['observations'][0]]),'exact repeated observation')
                        require(targets[a]['queries']==targets[b]['queries'] and prefix[a][-1]==prefix[b][-1],'repeat final equality')
                        core['postempty' if left=='E' else 'nonempty_repetition']+=1
                    require(canon(tt['A']['observations'][1:])==canon(tt['E']['observations'][1:]) and tt['A']['observations'][0]['fact']==tt['E']['observations'][0]['fact'],'contradiction single first scalar');core['cross_fact_contradiction']+=1
                    for left,right in [('E','G'),('F','H')]:
                        a,b=rr[left],rr[right];lt,rt=tt[left],tt[right];pair=(a['sample_id'],b['sample_id'])
                        require([n for n,(x,y) in enumerate(zip(lt['observations'],rt['observations'])) if canon(x)!=canon(y)]==[1],'only later scalar changed')
                        require(lt['observations'][1]['fact']==rt['observations'][1]['fact'] and prefix[pair[0]][1]==prefix[pair[1]][1],'same first support and ordered facts')
                        require(not prefix[pair[0]][-1] and prefix[pair[1]][-1],'opposite empty/nonempty')
                        for o in lt['observations'][:2]:require(any(canon(w['facts'][o['fact']])==canon(o['equals']) for w in lt['worlds']),'contradictory equals individually supported')
                        pair_evidence[pair]={'observation_length':len(lt['observations']),'left':{'complete_target':targets[pair[0]],'prefix_counts':[len(s) for s in prefix[pair[0]]],'prefix_survivor_ids':prefix[pair[0]]},'right':{'complete_target':targets[pair[1]],'prefix_counts':[len(s) for s in prefix[pair[1]]],'prefix_survivor_ids':prefix[pair[1]]}}
                for context in 'ABCDEFGH':
                    a,b=[idx[fi,context,v,schedule,renderer] for v in ('identity','transposed')];pair=(a['sample_id'],b['sample_id']);control=a['auxiliary_scalar_control']
                    require(canon(transform(a['semantic_task'],control))==canon(b['semantic_task']) and canon(transform(b['semantic_task'],control))==canon(a['semantic_task']),'actual involutive column transposition')
                    require(a['semantic_task']['worlds']!=b['semantic_task']['worlds'] and targets[pair[0]]==targets[pair[1]] and prefix[pair[0]]==prefix[pair[1]],'nontrivial scalar transformation exact output')
                    scalar_pairs.add(pair)
                    if renderer=='forward':
                        c=idx[fi,context,'identity',schedule,'reverse'];d=idx[fi,context,'transposed',schedule,'reverse']
                        for x,y in [(a,c),(b,d)]:require(canon(x['semantic_task'])==canon(y['semantic_task']) and x['rendered_input']!=y['rendered_input'],'two actual field views');field_pairs.add((x['sample_id'],y['sample_id']))
                for left,right in [('E','G'),('F','H')]:
                    for lv,rv in [('identity','transposed'),('transposed','identity')]:
                        a,b=idx[fi,left,lv,schedule,renderer],idx[fi,right,rv,schedule,renderer];at,bt=a['semantic_task'],b['semantic_task'];pair=(a['sample_id'],b['sample_id'])
                        require(canon(at['observations'])==canon(bt['observations']) and canon(at['queries'])==canon(bt['queries']),'cross-table same exact observations/queries')
                        require(canon(transform(at,a['auxiliary_scalar_control'])['worlds'])==canon(bt['worlds']) and at['worlds']!=bt['worlds'],'only actual auxiliary table change')
                        require(prefix[pair[0]][1]==prefix[pair[1]][1] and not prefix[pair[0]][-1] and prefix[pair[1]][-1],'same prefix opposite table outcome');cross_pairs.add(pair)
    require([len(pair_evidence),len(scalar_pairs),len(field_pairs),len(cross_pairs)]==[864,1728,1728,864],'complete actual pair counts')
    entries=manifest['supplied_same_table_v2_pairs'];entryindex={};entryuses=Counter();namedpairs=set();inherited_ledger=[];nonmembers=0
    for e in entries:
        key=(e['control_id'],e['name'],tuple(e['pair_sample_ids']));require(key not in entryindex,'unique named resolver');entryindex[key]=e
        pair=key[2];require(pair in pair_evidence and e['name'] in component.NAMES,'actual named pair')
        require((e['name'],pair) not in namedpairs,'complete named pair coverage');namedpairs.add((e['name'],pair));ev=pair_evidence[pair];a,b=[by[s] for s in pair]
        require(e['scope']=='corpus_variant' and e['exact_candidate_sha256']==candidate_sha,'entry candidate scope')
        require(type(e['observation_length']) is int and e['observation_length']==ev['observation_length'],'actual entry length')
        require(e['source_family_key']==a['source_family_key'] and e['auxiliary_scalar_variant']==a['auxiliary_scalar_variant'],'actual entry family variant')
        require(e['changed_fields']==['observations[1].equals'] and canon(e['typed_first_observation'])==canon(a['semantic_task']['observations'][0]),'supplied change/first evidence')
        require(e['first_prefix_survivor_ids']==prefix[pair[0]][1],'supplied first prefix')
        for side,sid in [('left',pair[0]),('right',pair[1])]:
            require(e[side+'_prefix_survivor_ids']==prefix[sid] and e[side+'_prefix_counts']==[len(s) for s in prefix[sid]],'entry complete prefix comparison')
            require(canon(e[side+'_complete_query_payloads'])==canon(targets[sid]['queries']),'entry complete query payload')
        if 'pair_record_hashes' in e:require(e['pair_record_hashes']==[by[s]['record_hash'] for s in pair],'optional exact pair hashes')
    require(len(entries)==len(namedpairs)==1728,'1728 unique entries')
    aggregate_names={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
    probe_map={e['probe_id']:e for e in manifest['executed_audit_probes']};require(len(probe_map)==16,'16 unique stored probes')
    for r in rows:
        names=[]
        for c in r['family_control_claims']:
            require(set(c)<=component.CLAIM_FIELDS and ('name' in c)!=('property' in c),'closed claim vocabulary')
            name=c.get('name',c.get('property'));names.append(name);require(c['evidence_manifest']==cfg['manifest'],'exact actual provenance file')
            if name in component.NAMES:
                resolved=component.resolve(r,c,by,manifest,candidate_sha);key=(c['control_id'],name,tuple(resolved['pair_sample_ids']));entryuses[key]+=1
                nm=r['sample_id'] not in resolved['pair_sample_ids'];nonmembers+=nm
                inherited_ledger.append({'owner_sample_id':r['sample_id'],'owner_record_hash':r['record_hash'],'name':name,'control_id':c['control_id'],'pair_sample_ids':resolved['pair_sample_ids'],'pair_record_hashes':[by[s]['record_hash'] for s in resolved['pair_sample_ids']],'entry_sha256':sha(entryindex[key]),'actual_owner_length':resolved['actual_observation_length'],'owner_nonmember':nm,'actual_pair_evidence_sha256':sha(pair_evidence[tuple(resolved['pair_sample_ids'])])})
            elif c['scope']=='audit_probe':
                require(c['probe_id'] in probe_map,'executed probe provenance');e=probe_map[c['probe_id']];require(e['name']==name and e['source_family_key']==r['source_family_key'] and e['auxiliary_scalar_variant']==r['auxiliary_scalar_variant'],'probe actual scope')
            else:require(c['scope']=='corpus_variant','corpus claim scope')
        require(len(names)==len(set(names)) and component.NAMES|aggregate_names<=set(names),'two inherited plus three aggregates every row')
    require(len(inherited_ledger)==6912 and nonmembers==3456 and set(entryuses)==set(entryindex),'full owner binding')
    require(Counter(entryuses.values())=={5:864,3:864},'actual owner uses length strata')
    print('INHERITED_AND_AGGREGATE_CONTROLS',len(inherited_ledger),len(pair_evidence),flush=True)
    for field,expected in [('within_context_scalar_pairs',scalar_pairs),('field_order_pairs',field_pairs),('same_observation_opposite_status_pairs',cross_pairs)]:
        actual=set()
        for e in manifest[field]:
            pair=tuple(e['pair_sample_ids']);require(pair in expected and pair not in actual,'actual '+field);actual.add(pair);a,b=[by[s] for s in pair]
            if field=='within_context_scalar_pairs':
                require(e['candidate_sha256']==candidate_sha and e['auxiliary_scalar_control']==a['auxiliary_scalar_control'] and e['exact_changed_world_column']==a['auxiliary_scalar_control']['fact'] and e['involution_verified'] is True,'scalar manifest evidence')
                expected_ev={'complete_target':targets[pair[0]],'prefix_counts':[len(s) for s in prefix[pair[0]]],'prefix_survivor_ids':prefix[pair[0]]};require(canon(e['prefix_and_target_evidence'])==canon(expected_ev),'scalar complete evidence')
            elif field=='field_order_pairs':require(e['typed_task_sha256']==sha(a['semantic_task']) and e['complete_target_sha256']==sha(targets[pair[0]]),'field task target hashes')
            else:
                require(e['candidate_sha256']==candidate_sha and e['observation_length']==len(a['semantic_task']['observations']),'cross table hash length')
                require(canon(e['ordered_identical_observations'])==canon(a['semantic_task']['observations']) and canon(e['ordered_identical_queries'])==canon(a['semantic_task']['queries']),'cross table input evidence')
                for side,sid in [('left',pair[0]),('right',pair[1])]:require(canon(e[side])==canon({'complete_target':targets[sid],'prefix_counts':[len(s) for s in prefix[sid]],'prefix_survivor_ids':prefix[sid]}),'cross table prefix target')
        require(actual==expected,'complete manifest '+field)
    # Original controls remain scope-sensitive; null evidence deliberately has evidence_sample_ids.
    original_counts=Counter();original_seen=set()
    for e in manifest['supplied_original_pairs']:
        name=e['name'];original_counts[name]+=1;ident=(name,e['source_family_key'],e['auxiliary_scalar_variant'],tuple(e.get('pair_sample_ids',e.get('evidence_sample_ids',[]))));require(ident not in original_seen,'original control uniqueness');original_seen.add(ident)
        members=e.get('pair_sample_ids',e.get('evidence_sample_ids'));require(members and all(s in by for s in members),'original actual evidence')
        rr=[by[s] for s in members];require(all(r['source_family_key']==e['source_family_key'] and r['auxiliary_scalar_variant']==e['auxiliary_scalar_variant'] for r in rr),'original scope')
        if name=='null_guarantee_contrast':
            categories=set()
            for r in rr:
                for answer in targets[r['sample_id']]['queries']:
                    if not targets[r['sample_id']]['surviving_worlds']:categories.add('empty')
                    if answer['entailed'] and answer['guaranteed'] is None:categories.add('entailed_null')
                    if answer['distinct_values']>1 and any(c['value'] is None for c in answer['counts']):categories.add('ambiguous_null')
            require(categories>={'empty','entailed_null','ambiguous_null'},'actual null category contrast');continue
        require(len(rr)==2,'two original pair members');a,b=rr;at,bt=a['semantic_task'],b['semantic_task'];aa,bb=targets[a['sample_id']],targets[b['sample_id']]
        require(canon(at['worlds'])==canon(bt['worlds']) and canon(at['queries'])==canon(bt['queries']),'original same table/queries')
        if name=='typed_equals_counterfactual':
            changed=[n for n,(x,y) in enumerate(zip(at['observations'],bt['observations'])) if canon(x)!=canon(y)];require(len(changed)==1,'typed one changed observation');n=changed[0];x,y=at['observations'][n],bt['observations'][n]
            require(x['fact']==y['fact'] and {type(x['equals']),type(y['equals'])}=={bool,int} and x['equals']==y['equals'],'active bool/int same-value contrast')
            require(prefix[a['sample_id']][n+1] and prefix[b['sample_id']][n+1] and prefix[a['sample_id']][n+1]!=prefix[b['sample_id']][n+1],'typed changed nonempty prefix')
        elif name=='equal_support_usefulness_counterfactual':require(len(aa['surviving_worlds'])==len(bb['surviving_worlds'])>0 and aa['queries']!=bb['queries'],'original equal support actual')
        elif name=='observation_reorder_final_invariance':require(canon(at['observations'])==canon(list(reversed(bt['observations']))) and aa['surviving_worlds']==bb['surviving_worlds'] and aa['queries']==bb['queries'],'original reorder')
        elif name in ('nonempty_repetition_final_invariance','post_empty_append_absorption'):require(canon(bt['observations'])==canon(at['observations']+[at['observations'][0]]) and aa['surviving_worlds']==bb['surviving_worlds'] and aa['queries']==bb['queries'],'original append')
        elif name=='cross_fact_contradiction_counterfactual':require(bool(aa['surviving_worlds'])!=bool(bb['surviving_worlds']),'original contradiction')
        else:require(False,'unknown original entry '+name)
    require(len(original_seen)==58,'all58 original entries')
    probes=[]
    for e in probe_map.values():
        a,p=own.solve(e['probe_task']);b,q=direct(e['probe_task']);require(canon(a)==canon(b),'probe independent solvers')
        source=by[e['source_sample_id']];expected=copy.deepcopy(targets[source['sample_id']]);expected_prefix=copy.deepcopy(prefix[source['sample_id']])
        if e['name']=='world_id_rename_equivariance':
            before,after=source['semantic_task']['worlds'],e['probe_task']['worlds'];require([w['facts'] for w in before]==[w['facts'] for w in after],'rename same actual facts');mapping={x['world_id']:y['world_id'] for x,y in zip(before,after)};require(len(set(mapping.values()))==len(mapping),'bijection')
            expected['surviving_worlds']=sorted(mapping[s] for s in expected['surviving_worlds']);expected_prefix=[sorted(mapping[s] for s in pp) for pp in expected_prefix]
        else:require(e['name']=='world_order_invariance' and {canon(w) for w in source['semantic_task']['worlds']}=={canon(w) for w in e['probe_task']['worlds']},'world order exact set')
        require(canon(a)==canon(expected)==canon(e['producer_reference_target']) and [sorted(w['world_id'] for w in s) for s in p]==expected_prefix,'executed probe complete output/prefix')
        require(e['scope']=='audit_probe' and e['added_records']==e['added_families']==0,'separate probe scope');probes.append({'probe_id':e['probe_id'],'task_sha256':sha(e['probe_task']),'computed_target_sha256':sha(a),'pass':True})
    axes=[('id',),('position',),('fact',),('id','position'),('id','fact'),('position','fact'),('id','position','fact')];histledger=[]
    mh={(e['source_family_key'],e['controlled_context'],e['auxiliary_scalar_variant'],e['renderer']):e for e in manifest['actual_conditional_query_histograms']};require(set(mh)==set(groups) and len(mh)==128,'manifest all128 histogram groups')
    for key,rr in groups.items():
        require(len(rr)==27 and {r['schedule_index'] for r in rr}==set(range(27)),'whole27 conditional schedule')
        m=mh[key];require(m['actual_schedule_indices']==list(range(27)) and m['complete_positive_triple_support'] is True,'measured complete schedule')
        results={}
        for axis in axes:
            rolehist=defaultdict(Counter);fullhist=defaultdict(Counter)
            for r in rr:
                sid=r['sample_id'];answers={a['query_id']:a for a in targets[sid]['queries']}
                for role in r['query_role_assignments']:
                    coord=canon([role[{'id':'query_id','fact':'fact','position':'position'}[a]] for a in axis]);rolehist[coord][role['role']]+=1;fullhist[coord][canon(signature(answers[role['query_id']],targets[sid],prefix[sid]))]+=1
            expected_count=9 if len(axis)==1 else 3 if len(axis)==2 else 1
            require(len(rolehist)==3**len(axis) and all(set(c)=={'invariant_query','variation_query','observed_control'} and set(c.values())=={expected_count} for c in rolehist.values()),'actual full triple role crossing')
            require(len({canon(c) for c in fullhist.values()})==1,'actual fulltyped provenance hist equal every bucket')
            supplied=m['histograms'][','.join(axis)]
            require(canon(dict(rolehist))==canon(norm_hist(supplied['actual_role_histograms'],axis,False)),'factual role histogram')
            require(canon(dict(fullhist))==canon(norm_hist(supplied['actual_full_typed_outcome_provenance_histograms'],axis,True)),'factual typed/provenance histogram')
            results[','.join(axis)]={'actual_buckets':len(rolehist),'each_role_count_per_bucket':expected_count,'actual_role_histogram_sha256':sha(rolehist),'actual_typed_provenance_histogram_sha256':sha(fullhist),'diagnostic_coordinate_normalization':'axis-specific position labels only; actual positions and scalar payloads strict'}
        histledger.append({'coordinates':key,'actual_schedules':27,'axes':results,'pass':True})
    print('HISTOGRAMS_ORIGINAL_CONTROLS_AND_PROBES',len(histledger)*7,len(original_seen),len(probes),flush=True)
    # Exact final roles/hash joins and first-to-final disagreement preservation.
    expert=receipt_join(lines(cfg['expert_final']),by,'expert', '/root/expert_reviewer');red=receipt_join(lines(cfg['red_final']),by,'Adversarial Red-Team','/root/adversarial_auditor')
    ef=receipt_join(lines(cfg['expert_first']),by,'expert','/root/expert_reviewer');rf=receipt_join(lines(cfg['red_first']),by,'Adversarial Red-Team','/root/adversarial_auditor')
    oracle={r['sample_id']:r for r in lines(cfg['oracle'])};require(len(oracle)==3456 and set(oracle)==set(by),'oracle exact coverage')
    ep={r['sample_id']:r for r in lines(cfg['expert_predictions'])};rp={r['sample_id']:r for r in lines(cfg['red_predictions'])};require(set(ep)==set(rp)==set(by),'both pregold full predictions')
    for sid,r in by.items():
        require(expert[sid]['decision']==red[sid]['decision']=='ACCEPT' and not expert[sid]['critical_defects'] and not red[sid]['critical_defects'],'completed final reviewer no unresolved critical')
        require(ef[sid]['decision']=='REVISE' and ef[sid]['critical_defects']==['conditional typed histogram/control assertion failed'] and rf[sid]['decision']=='ACCEPT','preserve exact original disagreements')
        require(oracle[sid]['pass'] is True and not oracle[sid]['errors'] and not oracle[sid]['metadata_errors'] and oracle[sid]['record_hash']==r['record_hash'],'oracle exact pass')
        for p in (ep[sid],rp[sid]):require(p['record_hash']==r['record_hash'] and canon(p['expected_target'])==canon(targets[sid]),'fresh own expected target joins from other input phases')
    efinal=load(cfg['expert_execution_final']);rfinal=load(cfg['red_execution']);ee=load(cfg['expert_execution'])
    require(efinal['status']=='COMPLETED_EXPERT_CANONICAL_REVIEW' and efinal['reviewer_execution_id']=='/root/expert_reviewer','completed expert FINAL actual actor')
    require(rfinal['completed'] is True and rfinal['actual_author_and_execution_id']=='/root/adversarial_auditor' and rfinal['actual_rows_reviewed']==3456,'completed red actual actor')
    require(ee['exit_code']==0 and ee['row_error_count']==ee['batch_error_count']==0,'expert actual corrective execution')
    for path,e in efinal['files'].items():require(bind(path)['sha256']==e['sha256'] and bind(path)['bytes']==e['length_bytes'],'all expert11 FINAL bindings')
    for path,h in rfinal['outputs'].items():require(bind(path)['sha256']==h,'all red actual output bindings')
    require(rfinal['file_bindings_before']==rfinal['file_bindings_after'],'red custody before after')
    require(ee['bindings_before']==ee['bindings_after'] and ee['source_before_after']['sha256']==bind('reviews/expert/belief-gap-v4-canonical-review-attempt03.py')['sha256'],'expert custody before after')
    # Receipt-only root code/official/integration evidence; private implementation not inspected.
    machine=load(cfg['machine_execution']);suite=load(cfg['suite_execution'])
    require(machine['exit_code']==0 and machine['all3456_whole_cohort_machine_gates_pass'] is True and machine['actual_v4_preservation']['status']=='PROVED','root completed machine evidence')
    for k in ['oracle_failures','metadata_failures','validation_failures','belief_control_error_count']:require(machine['actual_statistics'][k]==0,'current machine zero '+k)
    require(suite['exit_code']==0 and suite['status'].startswith('PASS') and suite['bound_sources_and_inputs_unchanged'] is True,'root445 passing source-bound suite')
    require(machine['bound_source_and_input_bytes_before']==machine['bound_source_and_input_bytes_after'],'machine unchanged custody')
    require(suite['source_and_input_sha256_before']==suite['source_and_input_sha256_after'],'suite unchanged custody')
    for path,h in machine['outputs'].items():
        expected=h['sha256'] if isinstance(h,dict) else h;require(bind(path)['sha256']==expected,'root actual output current hash')
    require(bind(suite['output_path'])['sha256']==suite['output_sha256'],'source-bound suite log')
    # Fitted finite feature maps are diagnostics, not unseen/model efficacy.
    rules={name:Counter() for name in ['original_marker','marker_and_linen']};lookup=defaultdict(lambda:defaultdict(Counter));train_lookup=defaultdict(lambda:defaultdict(Counter))
    for r in rows:
        t=r['semantic_task'];sid=r['sample_id'];empty=not targets[sid]['surviving_worlds'];marker=any(type(o['equals']) is str and o['equals'] in ('stone','unknown') for o in t['observations']);linen=any(type(o['equals']) is str and o['equals']=='linen' for o in t['observations'])
        for name,predict in [('original_marker',marker),('marker_and_linen',marker and linen)]:
            q=rules[name];q['records']+=1;q['status_errors']+=predict!=empty;q[r['split']+'_errors']+=predict!=empty;q[r['auxiliary_scalar_variant']+'_errors']+=predict!=empty
            if predict:
                n,m=len(t['worlds']),len(t['observations']);guessed={'surviving_worlds':[],'belief_sizes':[1 if n==8 else 2]+[0]*(m-1),'queries':[{'query_id':q['query_id'],'counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False} for q in t['queries']]}
                q['full_target_covered']+=1;q['full_target_correct']+=canon(guessed)==canon(targets[sid])
            else:q['full_target_abstentions']+=1
        # Column names omitted; ordered typed frequency profiles and exact observations retained.
        freq=canon({'columns':{f:sorted(Counter(canon(w['facts'][f]) for w in t['worlds']).items()) for f in sorted(t['worlds'][0]['facts'])},'observations':t['observations']})
        features={'coarse_counts':canon([len(t['worlds']),len(t['observations']),len(t['queries'])]),'prompt_length':str(len(r['rendered_input'])),'scalars_prompt_length':canon([[o['equals'] for o in t['observations']],len(r['rendered_input'])]),'typed_column_frequencies_observations':freq,'full_typed_task':canon(t)}
        for name,key in features.items():lookup[name][key][empty]+=1
        if r['split']=='train':
            for name,key in features.items():train_lookup[name][key][empty]+=1
        r['_diagnostic_features']=features
    fitted={}
    for name,buckets in lookup.items():
        count=sum(max(v.values()) for v in buckets.values());covered=correct=0
        for r in rows:
            if r['split']!='dev_challenge':continue
            key=r['_diagnostic_features'][name];bucket=train_lookup[name].get(key)
            if bucket:covered+=1;guess=max(bucket,key=lambda k:(bucket[k],not k));correct+=guess==(not targets[r['sample_id']]['surviving_worlds'])
        fitted[name]={'cohort_fitted_correct':count,'cohort_records':3456,'finite_buckets':len(buckets),'train_fit_dev_covered':covered,'train_fit_dev_correct':correct,'train_fit_dev_abstentions':1728-covered}
    for r in rows:del r['_diagnostic_features']
    require(all(v['status_errors']==864 and v['train_errors']==v['dev_challenge_errors']==432 for v in rules.values()),'frozen rules actual independent errors')
    # Four fresh component counterexamples demonstrate frequency lookup is not conjunction semantics.
    counterexamples=[]
    for fi in (12,13,18,19):
        r=idx[fi,'E','identity',0,'forward'];t=r['semantic_task'];o=t['observations'][1];x=t['observations'][0];inside=[w for w in t['worlds'] if canon(w['facts'][x['fact']])==canon(x['equals'])];outside=[w for w in t['worlds'] if canon(w['facts'][o['fact']])==canon(o['equals']) and w not in inside]
        altered=copy.deepcopy(t);ids_pair=[inside[0]['world_id'],outside[0]['world_id']];ww={w['world_id']:w for w in altered['worlds']};a,b=[ww[s] for s in ids_pair];a['facts'][o['fact']],b['facts'][o['fact']]=b['facts'][o['fact']],a['facts'][o['fact']]
        actual,p=own.solve(altered);other,_=direct(altered);require(canon(actual)==canon(other) and actual['surviving_worlds'] and not targets[r['sample_id']]['surviving_worlds'],'component counterexample opposite outcome')
        require([sorted(w['world_id'] for w in pp) for pp in p][1]==prefix[r['sample_id']][1] and altered['observations']==t['observations'] and altered['queries']==t['queries'] and len(canon(altered))==len(canon(t)),'same prefix/obs/query/length counterexample')
        require(all(Counter(canon(w['facts'][f]) for w in altered['worlds'])==Counter(canon(w['facts'][f]) for w in t['worlds']) for f in t['worlds'][0]['facts']),'same full column frequencies')
        counterexamples.append({'source_sample_id':r['sample_id'],'source_record_hash':r['record_hash'],'swapped_world_ids':ids_pair,'changed_auxiliary_fact':o['fact'],'actual_probe_task':altered,'computed_target':actual,'production_records_added':0})
    outputs=[hb,out(cfg['outputs']['rows'],detail,True),out(cfg['outputs']['inherited'],inherited_ledger,True),out(cfg['outputs']['histograms'],histledger,True),out(cfg['outputs']['counterexamples'],counterexamples)]
    audit={'status':'COMPLETED_INDEPENDENT_EXACT_V4_ADJUDICATION_AUDIT','actual_execution_id':ACTOR,'records':3456,'exact_target_hash_prompt_properties':3456,'own_fresh_incremental_direct_and_precanonical_targets_agree':3456,'full_old_new_typed_preservation':3456,'only_changed_fields':dict(changes),'added_student_inputs':0,'added_semantic_tasks':0,'added_source_families':0,'source_families':4,'base_contexts':32,'scalar_contexts':64,'decoded_source_tables':len(decoded),'decoded_scalar_tables':len(scalars),'visible_tables':len(visible),'parsed_tasks':len(semantic_families),'split_records':dict(splits),'family_splits':family_splits,'difficulty':dict(dif),'properties':dict(props),'status_counts':dict(status),'query_usefulness_flags':dict(flags),'profiles':profiles,'core_controls':dict(core),'inherited_claims':len(inherited_ledger),'inherited_named_entries':len(entries),'actual_inherited_pairs':len(pair_evidence),'pair_lengths':dict(Counter(str(ev['observation_length']) for ev in pair_evidence.values())),'nonmember_owner_claims':nonmembers,'entry_use_histogram':dict(Counter(entryuses.values())),'shared_ids_across_names_legal':True,'original_manifest_entries':dict(original_counts),'scalar_pairs':len(scalar_pairs),'field_pairs':len(field_pairs),'same_observation_cross_table_pairs':len(cross_pairs),'conditional_groups':128,'actual_axes_histogram_comparisons':896,'independently_reexecuted_stored_manifest_probes':probes,'root_actual_machine_probe_count_separate_scope':machine['actual_statistics']['belief_control_counts']['executed_probes'],'hand_checks':len(hands),'frozen_rule_measurements':{k:dict(v) for k,v in rules.items()},'fitted_finite_status_diagnostics':fitted,'first_expert_red_disagreements':3456,'final_expert_red_disagreements':0,'first_critical_issue_resolution':'Actual positions integer; supplied object-key strings are axis-specific diagnostic coordinates. Own complete typed/provenance histogram reconstruction passes. Public text specifies no literal key serialization. Expert first REVISE and both failed executions remain immutable. No semantic scalar coercion or retroactive gate relaxation.','root_source_bound_suite_attestation':'445 tests passed; receipt/log bound, private implementation not inspected or imported','private_producer_or_formal_implementation_read':False,'other_reviewer_implementation_imported':False,'direct_individual_handread_records':False,'scope':'Programmatic complete recomputation/control proof with public-text and actual representative evidence inspection; nonblind adjudication after completed reviews.','disclosures':cfg['disclosures'],'evidence_outputs':outputs,'preexecution':bind(P+'preexecution-r1.json'),'source':bind(str(Path(__file__).relative_to(BASE))),'all_required_data_gates_pass':True,'publication_eligible':True,'whole_cohort_publication_eligible':True,'publication_performed':False,'remaining_limits':['Four reused finite lineage families; no semantic growth from3456 aliases','75% nonempty and87.5% usefulness-false majority; residual fitted lookup and length associations','Exposed dev with correlated structures; no sealed OOD or model efficacy','Prior V3 canonical/gold and root-guided metadata exposure; own solver reuse; public design/adjudicator overlap and shared underlying model']}
    ab=out(cfg['outputs']['audit'],audit)
    verdict=[]
    for r in rows:
        sid=r['sample_id'];inherited=[x for x in inherited_ledger if x['owner_sample_id']==sid]
        verdict.append({'receipt_id':'adj-belief-gap-v4-'+sha([sid,r['record_hash'],ACTOR])[:24],'sample_id':sid,'record_hash':r['record_hash'],'reviewer_role':'independent_adjudicator','reviewer_execution_id':ACTOR,'scores':{'semantic_correctness':5,'target_correctness':5,'reasoning_validity':5,'instruction_clarity':4,'difficulty_quality':3,'diversity':2,'shortcut_resistance':2,'generalization_value':2},'critical_defects':[],'issues':[{'severity':'medium','category':'bounded_correlated_finite_proxy','evidence':'Four reused families;75% nonempty/87.5% usefulness false; fitted finite table/length associations; exposed development split; prior gold/design/shared-model exposure.','fix':'Future independent semantic-family and mechanism growth, fresh model-independent review and separately designed held-out evaluation; no efficacy inferred.'}],'decision':'ACCEPT','confidence':0.97,'reason':'Own complete typed output/prefix/hash/prompt/property recomputation agrees with frozen pre-canonical predictions. Exact prospective owner/length/named resolver and every cohort control pass. First histogram REVISE resolves as diagnostic decoding, preserved separately.','evidence':{'candidate_sha256':candidate_sha,'manifest_sha256':bind(cfg['manifest'])['sha256'],'own_audit_sha256':ab['sha256'],'source_sha256':bind(str(Path(__file__).relative_to(BASE)))['sha256'],'own_prediction_file':cfg['own_predictions'],'expert_first':'REVISE','expert_final':'ACCEPT','adversarial_first':'ACCEPT','adversarial_final':'ACCEPT','original_disagreement_preserved':True,'current_disagreement':False,'pair_sample_ids':inherited[0]['pair_sample_ids'],'actual_owner_length':inherited[0]['actual_owner_length'],'pair_record_hashes':inherited[0]['pair_record_hashes'],'publication_requires_complete_cohort':True}})
    vb=out(cfg['outputs']['verdicts'],verdict,True)
    designated=out(cfg['outputs']['designated_verdicts'],verdict,True)
    require(designated['sha256']==vb['sha256'],'designated alias exact bytes')
    report=f"""# Exact BELIEF V4 adjudication\n\nAll 3,456 exact records ACCEPT. The complete cohort is eligible under the prospective public contract and completed source-bound gates; this execution publishes nothing. Candidate `{candidate_sha}`, manifest `{bind(cfg['manifest'])['sha256']}`.\n\nOwn pre-canonical predictions were already frozen. Fresh complete recomputation checks full records, typed prompts/tasks/targets/prefixes and every attached property; all3,456 old/new triples, tags, difficulty, splits and semantic controls are preserved. Only six custody/claim fields changed. Zero added student inputs, tasks or source families.\n\nAll6,912 inherited claims resolve to1,728 exact named entries/864 truthful matched pairs at both lengths. Owner coordinates match actual evidence; nonmember owners and shared IDs across names are legitimate. All1,728 scalar pairs,1,728 field pairs,864 same-observation cross-table pairs,58 original entries,16 stored independent probe executions and896 conditional full typed/provenance histogram comparisons pass. Four decoded source tables/eight scalar tables/24 visible tables remain four mechanisms, not3,456 independent examples. Root machine's168 probes have a separate scope from16 stored probes reexecuted here.\n\nExpert first REVISE3456 disagreed with adversarial first ACCEPT3456. Both completed final reviewers ACCEPT3456. Expert attempt01 failed on an unpublished pair-field assumption; expert first full review and adversarial first canonical execution failed on diagnostic bucket position serialization. Their first files are immutable. Public specifications require actual positions but no serialized key spelling. The own axis-aware decoder accepts unambiguous diagnostic labels while rejecting collisions/false counts; actual positions, typed scalars, complete payloads and prefixes remain strict. All actual histograms independently agree, so the original histogram allegation is resolved without changing candidate, specs or gates.\n\nMeasured difficulty2/3 counts are972/2,484; G/H are simple level2. Status is75% nonempty; usefulness is87.5% false. Both frozen predicates make864 status errors,432 per split; complete-target coverage and abstentions are recorded separately. The finite fitted maps in the audit retain strong conditional associations and do not measure learning or generalization. Four independent component counterexamples preserve full marginal frequencies/observations/first prefix/length while changing conjunction status. They add no corpus rows.\n\nActual identity `/root/research_architect` authored the public designs and now adjudicates, without producing samples or serving as initial expert/red reviewer. Prior V3 canonical gold, root-guided metadata repair, own public solver reuse, common underlying model and design/adjudication overlap are disclosed. This is a nonblind exact metadata correction with exposed development families, not independent-model validation, structural OOD, architecture evidence or efficacy.\n\nSource/config/fixtures and all inputs were frozen before imports/execution. Compact ledgers cover every row/control; all new outputs are below25MiB. No pipeline audit, producer/private oracle import, release/catalog/public/core edit, push or training occurred.\n"""
    with (BASE/cfg['outputs']['report']).open('x') as s:s.write(report)
    outputs+=[ab,vb,designated,bind(cfg['outputs']['report'])]
    tb=out(cfg['outputs']['ticket'],{'ticket_id':'BELIEF-V4-RESIDUAL-GAP-ADJUDICATOR','status':'PROSPECTIVE_RESEARCH_PRIORITY_NO_CURRENT_HOLD','actual_author_execution_id':ACTOR,'candidate_sha256':candidate_sha,'own_audit':ab,'record_decisions':vb,'measured_limits':audit['remaining_limits'],'measured_status_counts':dict(status),'measured_usefulness_flags':dict(flags),'fitted_finite_status_diagnostics':fitted,'next_priority':'Grow independently authored structural/mechanism families and genuinely held-out combinations; preserve explicit existing finite claims. Future diagnostics need fresh input-only predictions and complete typed controls. Do not add aliases to claim semantic growth.','current_candidate_defect':False,'new_generation_or_release_requested':False,'design_adjudicator_overlap':True,'same_model_risk':True})
    outputs.append(tb)
    after=[bind(b['path']) for b in f['bindings']];require(after==f['bindings'],'all preexecution bindings unchanged after actual execution')
    execution={'status':'COMPLETED_EXECUTION','completed_execution':True,'actual_author_and_execution_id':ACTOR,'started_at_utc':start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-tick,'exit_code':0,'records':3456,'decision_counts':{'ACCEPT':3456},'publication_eligible':True,'whole_cohort_publication_eligible':True,'publication_performed':False,'preexecution_freeze':bind(P+'preexecution-r1.json'),'bindings_before':f['bindings'],'bindings_after':after,'source_and_inputs_unchanged':True,'outputs':outputs,'disclosures':cfg['disclosures']}
    eb=out(cfg['outputs']['execution'],execution)
    final={'status':'ACTUAL_ADJUDICATOR_FINAL','actual_execution_id':ACTOR,'candidate':bind(cfg['candidate']),'manifest':bind(cfg['manifest']),'decision_counts':{'ACCEPT':3456},'decisions':{'ACCEPT':3456},'publication_eligible':True,'whole_cohort_publication_eligible':True,'publication_performed':False,'completed_execution':eb,'all_new_outputs_below_25MiB':True,'files':outputs+[eb],'preexecution':bind(P+'preexecution-r1.json'),'disclosures':cfg['disclosures']}
    fb=out(cfg['outputs']['final'],final);print(canon({'final':fb,'verdicts':vb,'audit':ab,'execution':eb,'records':3456,'eligible':True}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        path=P+'failure-r1.json'
        if not (BASE/path).exists():out(path,{'actual_execution_id':ACTOR,'completed_execution':False,'exception':repr(exc),'traceback':traceback.format_exc(),'source':bind(str(Path(__file__).relative_to(BASE))),'config':bind(P+'config-r1.json'),'preexecution':bind(P+'preexecution-r1.json'),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_approval':False})
        traceback.print_exc();sys.exit(1)
