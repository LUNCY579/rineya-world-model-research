"""Actual adjudicator's independent public-contract checks; no producer imports.

This writes evidence only. Final decisions require separately completed reviewers.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import copy, hashlib, json, os
import jsonschema

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reviews/adjudication'
ACTOR = '/root/research_architect'
BATCH = 'memory-gap-v2'
EXPECTED = 'ca17f8a460f9a8909522901d2ef39b5f096b27a2cc56e0e3fb49ca465c8e96b1'
PROPS = ['absence_control', 'authoritative_tombstone', 'cross_key_filter_control',
         'decisive_delayed_boundary', 'equal_version_time_conflict',
         'equal_version_time_input_conflict', 'future_event_filter_control',
         'present_null_distinction', 'version_over_recency']
CORPUS = {'query_role_id_position_crossing', 'key_role_crossing', 'field_order_invariance',
          'version_priority_counterfactual', 'active_input_tie_swap_counterfactual',
          'null_lifecycle_counterfactual'}
PROBES = {'lower_version_append_invariance', 'other_key_edit_invariance',
          'future_event_edit_invariance'}

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def typed_equal(a, b):
    return canon(a) == canon(b)

def scalar(v):
    return v is None or type(v) in (bool, int, str)

def tuple_of(events, i):
    e = events[i]
    return e['version'], e['at'], i

def event_answer(e):
    return {'status': 'deleted' if e['op'] == 'delete' else 'present',
            'value': None if e['op'] == 'delete' else e['value']}

def semantic(r):
    return {k: r[k] for k in ('status', 'value')}

def resolve(events, q):
    indices = [i for i, e in enumerate(events)
               if e['key'] == q['key'] and e['at'] <= q['as_of']]
    i = max(indices, key=lambda i: tuple_of(events, i)) if indices else None
    a = {'status': 'absent', 'value': None} if i is None else event_answer(events[i])
    return dict(query_id=q['query_id'], key=q['key'], **a,
                chosen_version=None if i is None else events[i]['version'], chosen_event=i)

def solve(task):
    return {'results': [resolve(task['events'], q) for q in task['queries']]}

def solve_sorted(task):
    order = sorted(range(len(task['events'])),
                   key=lambda i: tuple_of(task['events'], i), reverse=True)
    result = []
    for q in task['queries']:
        i = next((i for i in order if task['events'][i]['key'] == q['key']
                  and task['events'][i]['at'] <= q['as_of']), None)
        a = {'status': 'absent', 'value': None} if i is None else event_answer(task['events'][i])
        result.append(dict(query_id=q['query_id'], key=q['key'], **a,
                           chosen_version=None if i is None else task['events'][i]['version'],
                           chosen_event=i))
    return {'results': result}

def validate_task(t):
    assert set(t) == {'kind', 'events', 'queries'} and t['kind'] == 'episodic'
    assert isinstance(t['events'], list) and isinstance(t['queries'], list)
    for e in t['events']:
        assert set(e) == {'key', 'version', 'at', 'op', 'value'}
        assert type(e['key']) is str and type(e['version']) is int and e['version'] >= 0
        assert type(e['at']) is int and e['at'] >= 0 and e['op'] in ('put', 'delete')
        assert scalar(e['value']) and (e['op'] != 'delete' or e['value'] is None)
    for q in t['queries']:
        assert set(q) == {'query_id', 'key', 'as_of'}
        assert type(q['key']) is str and type(q['query_id']) is str
        assert type(q['as_of']) is int and q['as_of'] >= 0
    assert len({q['query_id'] for q in t['queries']}) == len(t['queries']) == 8

def find_properties(t):
    es, qs = t['events'], t['queries']
    ans = [resolve(es, q) for q in qs]
    witnesses = {}
    for p, (q, a) in enumerate(zip(qs, ans)):
        w = a['chosen_event']
        eligible = [i for i, e in enumerate(es) if e['key'] == q['key'] and e['at'] <= q['as_of']]
        if w is None:
            witnesses.setdefault('absence_control', {'query_position': p,
                'subtype': 'future_only' if any(e['key'] == q['key'] for e in es) else 'missing_key'})
        else:
            for i in eligible:
                if typed_equal(event_answer(es[i]), semantic(a)):
                    continue
                if es[w]['op'] == 'delete' and es[i]['op'] == 'put' and es[i]['version'] < es[w]['version']:
                    witnesses.setdefault('authoritative_tombstone', {'query_position': p, 'winner': w, 'competitor': i})
                if es[i]['version'] == es[w]['version'] and es[i]['at'] < es[w]['at']:
                    witnesses.setdefault('equal_version_time_conflict', {'query_position': p, 'winner': w, 'competitor': i})
                if tuple_of(es, i)[:2] == tuple_of(es, w)[:2] and i < w:
                    witnesses.setdefault('equal_version_time_input_conflict', {'query_position': p, 'winner': w, 'competitor': i})
            recency = max(eligible, key=lambda i: (es[i]['at'], i))
            if (es[recency]['at'] > es[w]['at'] and es[recency]['version'] < es[w]['version']
                    and not typed_equal(event_answer(es[recency]), semantic(a))):
                witnesses.setdefault('version_over_recency', {'query_position': p, 'winner': w, 'competitor': recency})
        all_key = [i for i, e in enumerate(es) if e['key'] == q['key']]
        no_cutoff = max(all_key, key=lambda i: tuple_of(es, i)) if all_key else None
        if (no_cutoff is not None and es[no_cutoff]['at'] > q['as_of']
                and not typed_equal(event_answer(es[no_cutoff]), semantic(a))):
            witnesses.setdefault('future_event_filter_control', {'query_position': p, 'competitor': no_cutoff})
        all_eligible = [i for i, e in enumerate(es) if e['at'] <= q['as_of']]
        no_key = max(all_eligible, key=lambda i: tuple_of(es, i)) if all_eligible else None
        if (no_key is not None and es[no_key]['key'] != q['key']
                and not typed_equal(event_answer(es[no_key]), semantic(a))):
            witnesses.setdefault('cross_key_filter_control', {'query_position': p, 'competitor': no_key})
        for j, (q2, a2) in enumerate(zip(qs, ans)):
            if (q2['key'] == q['key'] and q2['as_of'] == q['as_of'] + 1
                    and a2['chosen_event'] is not None
                    and es[a2['chosen_event']]['at'] == q2['as_of']
                    and not typed_equal(semantic(a), semantic(a2))):
                witnesses.setdefault('decisive_delayed_boundary', {'before_position': p, 'at_position': j,
                                     'boundary': q2['as_of'], 'winner': a2['chosen_event']})
            if (a['status'] == 'present' and a['value'] is None and
                    a2['status'] in ('absent', 'deleted') and a2['value'] is None):
                witnesses.setdefault('present_null_distinction', {'present_position': p, 'contrast_position': j})
    return witnesses

def validate_supplied_witnesses(row, t):
    """Independently verify supplied factual evidence after our own truth search."""
    es, qs = t['events'], t['queries']
    def local(w):
        sig = w['query_signature']
        matches = [q for q in qs if q['query_id'] == w['query_id'] and
                   typed_equal(sig, {'key': q['key'], 'as_of': q['as_of']})]
        assert len(matches) == 1
        q = matches[0]; a = resolve(es, q); i = a['chosen_event']
        assert typed_equal(w['answer'], semantic(a)) and w['winner_index'] == i
        assert typed_equal(w['winner_tuple'], None if i is None else list(tuple_of(es, i)))
        indices = w['competitor_indices']
        assert len(indices) == len(w['competitor_tuples']) == len(w['competitor_answers'])
        for j, tup, payload in zip(indices, w['competitor_tuples'], w['competitor_answers']):
            assert type(j) is int and 0 <= j < len(es)
            assert typed_equal(tup, list(tuple_of(es, j))) and typed_equal(payload, event_answer(es[j]))
        return q, a, indices
    ws = row['property_witnesses']; assert set(ws) == set(PROPS)
    for name, w in ws.items():
        if name == 'decisive_delayed_boundary':
            before, a, _ = local(w['before']); at, z, _ = local(w['at']); T = w['boundary']
            assert type(T) is int and T >= 1 and before['key'] == at['key']
            assert before['as_of'] == T-1 and at['as_of'] == T
            assert es[z['chosen_event']]['at'] == T and not typed_equal(semantic(a), semantic(z))
            continue
        if name == 'present_null_distinction':
            _, a, _ = local(w['present_null']); _, z, _ = local(w['null_contrast'])
            assert a['status'] == 'present' and a['value'] is None
            assert z['status'] in ('absent', 'deleted') and z['value'] is None
            continue
        q, a, cs = local(w); i = a['chosen_event']
        if name == 'absence_control':
            assert i is None
            subtype = 'future_only' if any(e['key'] == q['key'] for e in es) else 'missing_key'
            assert w['absence_kind'] == subtype
            continue
        assert cs
        for j in cs:
            e = es[j]
            assert not typed_equal(event_answer(e), semantic(a))
            if name == 'future_event_filter_control':
                assert e['key'] == q['key'] and e['at'] > q['as_of']
                assert j == max([k for k, v in enumerate(es) if v['key'] == q['key']], key=lambda k: tuple_of(es, k))
            elif name == 'cross_key_filter_control':
                assert e['key'] != q['key'] and e['at'] <= q['as_of']
                assert j == max([k for k, v in enumerate(es) if v['at'] <= q['as_of']], key=lambda k: tuple_of(es, k))
            else:
                assert e['key'] == q['key'] and e['at'] <= q['as_of'] and i is not None
                if name == 'version_over_recency':
                    assert e['version'] < es[i]['version'] and e['at'] > es[i]['at']
                    latest = max([k for k,v in enumerate(es) if v['key']==q['key'] and v['at']<=q['as_of']], key=lambda k:(es[k]['at'],k))
                    assert not typed_equal(event_answer(es[latest]), semantic(a))
                elif name == 'authoritative_tombstone':
                    assert es[i]['op'] == 'delete' and e['op'] == 'put' and e['version'] < es[i]['version']
                elif name == 'equal_version_time_conflict':
                    assert e['version'] == es[i]['version'] and e['at'] < es[i]['at']
                elif name == 'equal_version_time_input_conflict':
                    assert tuple_of(es,j)[:2] == tuple_of(es,i)[:2] and j < i
                else:
                    raise AssertionError(name)

def decode(row, task):
    keys = row['key_role_assignments']
    assert set(keys) == {'delayed', 'lifecycle', 'index_tie', 'absent'}
    assert len(set(keys.values())) == 4 and all(type(v) is str for v in keys.values())
    inverse = {v:k for k,v in keys.items()}
    assert set(e['key'] for e in task['events']) <= set(inverse)
    assignments = row['query_role_assignments']
    assert len(assignments) == 8 and len({a['role'] for a in assignments}) == 8
    assert {a['position'] for a in assignments} == set(range(8))
    role_by_position = {}
    for a in assignments:
        assert set(a) == {'position','query_id','role'}
        assert a['query_id'] == task['queries'][a['position']]['query_id']
        role_by_position[a['position']] = a['role']
    events = [dict(e, key=inverse[e['key']]) for e in task['events']]
    signatures = {role_by_position[p]:[inverse[q['key']],q['as_of']] for p,q in enumerate(task['queries'])}
    assert len({canon(v) for v in signatures.values()}) == 8
    return events, signatures, role_by_position, inverse

def own_family_probes(t):
    es, qs = t['events'], t['queries']; own = find_properties(t); results = {}
    q = qs[own['version_over_recency']['query_position']]; before = resolve(es,q)
    maximum = max(e['version'] for e in es)
    wrong_value = '__adjudicator_contrast__'
    assert not typed_equal(semantic(before),{'status':'present','value':wrong_value})
    low = dict(key=q['key'],version=before['chosen_version']-1,at=q['as_of'],op='put',value=wrong_value)
    assert low['version'] >= 0
    es2 = copy.deepcopy(es)+[low]
    assert typed_equal(resolve(es2,q),before)
    results['lower_version_append_invariance']={'protected_query':q,'appended_event':low,'same_exact_result':True}
    other = dict(key='__adjudicator_unused_key__',version=maximum+1,at=q['as_of'],op='put',value=wrong_value)
    assert all(e['key'] != other['key'] for e in es)
    assert typed_equal(resolve(es+[other],q),before)
    results['other_key_edit_invariance']={'protected_query':q,'appended_event':other,'same_exact_result':True}
    future = dict(key=q['key'],version=maximum+1,at=q['as_of']+1,op='put',value=wrong_value)
    assert typed_equal(resolve(es+[future],q),before)
    results['future_event_edit_invariance']={'protected_query':q,'appended_event':future,'same_exact_result':True}
    j=own['version_over_recency']['competitor']; es2=copy.deepcopy(es);es2[j]['version']=maximum+1
    z=resolve(es2,q);assert z['chosen_event']==j and not typed_equal(semantic(z),semantic(before))
    results['version_priority_counterfactual']={'protected_query':q,'changed_event':j,'changed_field':'version','answer_changed':True}
    wit=own['equal_version_time_input_conflict'];q=qs[wit['query_position']];es2=copy.deepcopy(es)
    i,j=wit['winner'],wit['competitor'];es2[i],es2[j]=es2[j],es2[i]
    assert not typed_equal(semantic(resolve(es2,q)),semantic(resolve(es,q)))
    results['active_input_tie_swap_counterfactual']={'protected_query':q,'swapped_indices':[i,j],'answer_changed':True}
    q=qs[own['present_null_distinction']['present_position']];i=resolve(es,q)['chosen_event'];es2=copy.deepcopy(es)
    es2[i]['op']='delete';z=resolve(es2,q);assert z['status']=='deleted' and z['value'] is None
    results['null_lifecycle_counterfactual']={'protected_query':q,'changed_event':i,'changed_field':'op','status_changed':True}
    q2=dict(q,key='__adjudicator_unused_key__');z=resolve(es,q2)
    assert z['status']=='absent' and not typed_equal(semantic(z),semantic(resolve(es,q)))
    results['query_key_counterfactual']={'before_query':q,'after_query':q2,'answer_changed':True}
    return results

def write_new(p,x):
    payload=json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+'\n'
    with p.open('x',encoding='utf-8') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())

def main():
    out=OUT/'memory-gap-v2.independent_adjudicator_audit.json'
    if out.exists():
        print('PRESERVED_EXISTING_INDEPENDENT_AUDIT');return
    candidate=ROOT/'data/REM-001/candidate_renderings/memory-gap-v2.jsonl'
    assert sha(candidate)==EXPECTED
    rows=[json.loads(l) for l in candidate.read_text().splitlines()]
    assert len(rows)==len({r['sample_id'] for r in rows})==768
    schema=ROOT/'schemas/canonical/v1.schema.json'
    validator=jsonschema.Draft202012Validator(json.loads(schema.read_text()))
    groups=defaultdict(list); facts=[]; split_counts=Counter(); status_counts=Counter(); type_counts=Counter()
    property_counts=Counter(); absence_counts=Counter(); event_counts=Counter(); task_digests=defaultdict(list)
    for row in rows:
        sid=row['sample_id']; validator.validate(row)
        assert digest({k:v for k,v in row.items() if k!='record_hash'})==row['record_hash']
        assert row['review_status']=='PENDING' and not row['review_receipt_ids']
        assert row['track_id']=='REM-001' and row['difficulty']==row['structural_difficulty']==3
        prefix, raw=row['rendered_input'].split('Task JSON:\n',1)
        assert 'greatest tuple (version,at,zero_based_input_event_index)' in prefix
        assert 'Scalar equality is type-sensitive' in prefix
        task=json.loads(raw);validate_task(task);assert typed_equal(task,row['semantic_task'])
        target=solve(task); assert typed_equal(target,solve_sorted(task))
        assert typed_equal(target,row['verified_target'])
        props=find_properties(task);assert set(props)==set(PROPS)
        assert set(row['skill_tags'])==set(PROPS)|{'finite_synthetic_proxy','versioned_retrieval'}
        validate_supplied_witnesses(row,task)
        claims=row['family_control_claims'];assert len(claims)==9
        actual=set()
        for claim in claims:
            assert len({'name','property'}&set(claim))==1
            assert set(claim)<= {'name','property','scope','source_family_key','evidence_manifest'}
            name=claim.get('name',claim.get('property'));scope=claim['scope'];actual.add((name,scope))
            assert claim.get('source_family_key',row['source_family_key'])==row['source_family_key']
            if 'evidence_manifest' in claim:assert type(claim['evidence_manifest']) is str and claim['evidence_manifest']
        assert actual=={(n,'corpus_variant') for n in CORPUS}|{(n,'audit_probe') for n in PROBES}
        events,sigs,roles,inverse=decode(row,task)
        results=target['results'];sc=Counter(a['status'] for a in results)
        assert set(sc)=={'present','deleted','absent'}
        property_counts.update(props.keys());status_counts.update(sc)
        for a in results:
            v=a['value'];type_counts.update(['null' if v is None else {bool:'boolean',int:'integer',str:'string'}[type(v)]])
        absence_counts.update(props[p]['subtype'] for p in ['absence_control'])
        split_counts[row['split']]+=1;event_counts[len(events)]+=1
        groups[(row['source_family_key'],row['controlled_variant'],row['renderer'])].append(row)
        task_digests[digest(task)].append(sid)
        facts.append({'sample_id':sid,'record_hash':row['record_hash'],'derived_target_sha256':digest(target),
                      'all_hash_schema_prompt_target_typed_checks':True,'supplied_witnesses_verified':True,
                      'properties':props,'status_counts':dict(sc),'history_event_count':len(events),
                      'family_claims_scope_valid':True,'source_family_key':row['source_family_key']})
    families={r['source_family_key'] for r in rows}
    assert len(families)==12 and {r['source_family_index'] for r in rows}==set(range(12,24))
    for fam in families:
        g=[r for r in rows if r['source_family_key']==fam]
        assert len(g)==64 and len({r['split'] for r in g})==1
        assert len({r['semantic_parent_id'] for r in g})==len({r['semantic_family_id'] for r in g})==1
    crossings=[]; joint_fractions=defaultdict(set); marginal_fractions=set()
    for (fam,variant,renderer),g in sorted(groups.items()):
        assert len(g)==8 and {r['query_rotation'] for r in g}==set(range(8))
        roleids=defaultdict(Counter);rolepos=defaultdict(Counter);keygroups=defaultdict(Counter)
        hist=defaultdict(lambda:defaultdict(Counter));joint=defaultdict(lambda:defaultdict(Counter))
        history=None; signatures=None
        for row in g:
            t=row['semantic_task']; events,sigs,roles,inverse=decode(row,t)
            if history is None:history=events;signatures=sigs
            assert typed_equal(events,history) and typed_equal(sigs,signatures)
            for group,key in row['key_role_assignments'].items():keygroups[key][group]+=1
            for p,(q,a) in enumerate(zip(t['queries'],solve(t)['results'])):
                role=roles[p];roleids[q['query_id']][role]+=1;rolepos[p][role]+=1
                observed=canon({'semantic':semantic(a),'chosen_version':a['chosen_version'],'chosen_event':a['chosen_event']})
                hist['query_id'][q['query_id']][observed]+=1
                hist['position'][p][observed]+=1;hist['key'][q['key']][observed]+=1
                for name,key in [('id_and_key',(q['query_id'],q['key'])),('position_and_key',(p,q['key'])),('id_and_position',(q['query_id'],p))]:
                    joint[name][key][a['status']]+=1
        assert len(roleids)==len(rolepos)==8
        assert all(set(c.values())=={1} and len(c)==8 for c in [*roleids.values(),*rolepos.values()])
        assert len(keygroups)==4 and all(len(c)==4 and set(c.values())=={2} for c in keygroups.values())
        for name,h in hist.items():assert len({canon(c) for c in h.values()})==1,name
        for name,h in joint.items():joint_fractions[name].add(sum(max(c.values()) for c in h.values())/sum(sum(c.values()) for c in h.values()))
        all_sc=Counter(a['status'] for r in g for a in solve(r['semantic_task'])['results'])
        marginal_fractions.add(max(all_sc.values())/sum(all_sc.values()))
        crossings.append({'source_family_key':fam,'controlled_variant':variant,'renderer':renderer,
                          'decoded_history_sha256':digest(history),'stable_role_signatures':signatures,
                          'role_id_counts':{k:dict(c) for k,c in roleids.items()},
                          'role_position_counts':{str(k):dict(c) for k,c in rolepos.items()},
                          'key_group_counts':{k:dict(c) for k,c in keygroups.items()},
                          'full_typed_outcome_and_provenance_histograms_equal_by_id_position_key':True,
                          'status_counts':dict(all_sc)})
    assert len(crossings)==96
    by_schedule={(r['source_family_key'],r['controlled_variant'],r['query_rotation'],r['renderer']):r for r in rows}
    pairs=[]; fieldpairs=[]
    mapping={'version_priority_raise':'version_priority_counterfactual','lifecycle_op_toggle':'null_lifecycle_counterfactual','index_tie_swap':'active_input_tie_swap_counterfactual'}
    for row in rows:
        fam,v,rot,style=row['source_family_key'],row['controlled_variant'],row['query_rotation'],row['renderer']
        if style=='forward':
            reverse=by_schedule[(fam,v,rot,'reverse')]
            assert row['rendered_input']!=reverse['rendered_input']
            assert typed_equal(row['semantic_task'],reverse['semantic_task'])
            assert typed_equal(solve(row['semantic_task']),solve(reverse['semantic_task']))
            fieldpairs.append([row['sample_id'],reverse['sample_id']])
        if v=='baseline':continue
        base=by_schedule[(fam,'baseline',rot,style)]
        t0,t1=base['semantic_task'],row['semantic_task']
        assert typed_equal(t0['queries'],t1['queries'])
        e0,e1=t0['events'],t1['events'];assert len(e0)==len(e1)
        changed=[i for i,(a,z) in enumerate(zip(e0,e1)) if not typed_equal(a,z)]
        changed_answers=[p for p,(a,z) in enumerate(zip(solve(t0)['results'],solve(t1)['results'])) if not typed_equal(semantic(a),semantic(z))]
        assert changed_answers
        if v=='version_priority_raise':
            assert len(changed)==1;i=changed[0]
            assert {k for k in e0[i] if not typed_equal(e0[i][k],e1[i][k])}=={'version'}
            assert e1[i]['version']>e0[i]['version']
            for p in changed_answers:
                q=t0['queries'][p];assert e1[i]['key']==q['key'] and e1[i]['at']<=q['as_of']
                assert e1[i]['version']>max(e['version'] for e in e0 if e['key']==q['key'] and e['at']<=q['as_of'])
                assert resolve(e1,q)['chosen_event']==i
        elif v=='lifecycle_op_toggle':
            assert len(changed)==1;i=changed[0]
            assert {k for k in e0[i] if not typed_equal(e0[i][k],e1[i][k])}=={'op'}
            assert e0[i]['value'] is None and e1[i]['value'] is None
            assert {e0[i]['op'],e1[i]['op']}=={'put','delete'}
            for p in changed_answers:
                a,z=resolve(e0,t0['queries'][p]),resolve(e1,t1['queries'][p])
                assert a['chosen_event']==z['chosen_event']==i and a['value'] is z['value'] is None
                assert {a['status'],z['status']}=={'present','deleted'}
        elif v=='index_tie_swap':
            assert len(changed)==2;i,j=changed
            assert typed_equal(e0[i],e1[j]) and typed_equal(e0[j],e1[i])
            assert tuple_of(e0,i)[:2]==tuple_of(e0,j)[:2]
            for p in changed_answers:
                assert resolve(e0,t0['queries'][p])['chosen_event']==max(i,j)
                assert resolve(e1,t1['queries'][p])['chosen_event']==max(i,j)
        else:raise AssertionError(v)
        pairs.append({'property':mapping[v],'scope':'corpus_variant','source_family_key':fam,
                      'baseline_sample_id':base['sample_id'],'changed_sample_id':row['sample_id'],
                      'changed_event_indices':changed,'semantic_answer_changed_positions':changed_answers})
    assert len(fieldpairs)==384 and len(pairs)==576
    probes=[]
    for (fam,v,style),g in sorted(groups.items()):
        if style!='forward':continue
        r=next(r for r in g if r['query_rotation']==0)
        p=own_family_probes(r['semantic_task'])
        assert PROBES<=set(p)
        probes.append({'source_family_key':fam,'controlled_variant':v,'base_sample_id':r['sample_id'],
                       'scope':'audit_probe','checks':p,'adds_training_records':0,'adds_source_families':0})
    assert len(probes)==48
    prior=sorted(ROOT.glob('data/*/releases/*/canonical.jsonl'))
    prior_source_keys=set(); prior_sids=set(); prior_events=set(); prior_tasks=set(); prior_hashes={}
    for p in prior:
        prior_hashes[str(p.relative_to(ROOT))]=sha(p)
        for line in p.read_text().splitlines():
            r=json.loads(line)
            prior_source_keys.add(r.get('source_family_key',r['semantic_family_id']))
            prior_sids.add(r['sample_id'])
            prior_tasks.add(digest(r['semantic_task']))
            if r['semantic_task'].get('kind')=='episodic':prior_events.add(digest(r['semantic_task']['events']))
    assert not families&prior_source_keys
    assert not {r['sample_id'] for r in rows}&prior_sids
    assert not set(task_digests)&prior_tasks
    assert not {digest(r['semantic_task']['events']) for r in rows}&prior_events
    assert all(len(g)==2 for g in task_digests.values()) and len(task_digests)==384
    assert sha(candidate)==EXPECTED
    source_paths=[candidate,schema,Path(__file__).resolve(),ROOT/'schemas/EXTENDED_SEMANTICS.md',
                  ROOT/'schemas/MEMORY_CONTROL_PROPERTIES.md',ROOT/'schemas/MEMORY_CONTROL_METADATA.md']
    report={'execution_id':ACTOR,'role':'independent_quality_adjudicator','completed_independent_check':True,
            'observed_utc':datetime.now(timezone.utc).isoformat(),'batch':BATCH,
            'candidate_sha256':EXPECTED,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
            'counts':{'records':len(rows),'source_families':len(families),'base_controlled_histories':48,
                      'distinct_named_event_histories':len({digest(r['semantic_task']['events']) for r in rows}),
                      'unique_correlated_task_instances':len(task_digests),'base_history_query_observations':384,
                      'rotation_level_query_observations':3072,'rendered_query_observations':6144,
                      'crossing_sets':len(crossings),'field_order_pairs':len(fieldpairs),
                      'supplied_counterfactual_pairs_all_rotations_styles':len(pairs),
                      'base_counterfactual_pairs':36,'own_probe_base_histories':len(probes),
                      'own_probe_checks':sum(len(p['checks']) for p in probes)},
            'split_record_counts':dict(split_counts),'split_source_family_counts':dict(Counter(next(r['split'] for r in rows if r['source_family_key']==f) for f in families)),
            'history_event_counts':dict(event_counts),'status_counts':dict(status_counts),
            'type_counts':dict(type_counts),'property_row_counts':dict(property_counts),'absence_subtype_counts':dict(absence_counts),
            'all_three_status_rows':len(rows),'conditional_empirical_status_majority_fractions':{k:sorted(v) for k,v in joint_fractions.items()},
            'marginal_status_majority_fractions':sorted(marginal_fractions),'rows':facts,'family_crossings':crossings,
            'field_order_pairs':fieldpairs,'supplied_counterfactual_pairs':pairs,'own_audit_probes':probes,
            'cross_release_lineage':{'historical_release_files':len(prior),'historical_release_sha256':prior_hashes,
                                     'source_family_overlap':[],'sample_id_overlap':[],'exact_task_overlap':[],
                                     'exact_named_event_history_overlap':[],
                                     'scope':'Exact typed task/history and source-family overlap; familiar precedence mechanisms still recur. No sealed OOD claim.'},
            'read_scope':'Public contracts/schema, current complete candidates, and synthetic historical canonical releases for lineage. No producer, reference or oracle implementation imported.',
            'search_scope_disclosure':'A preliminary workspace hash-definition search also returned producer hash-assertion lines, without opening or using producer semantic implementations.',
            'prior_design_role':'Actual same identity wrote REM-GAP-003 public design; it generated no samples and authored no initial expert/adversarial verdicts.',
            'release_approval':False,'final_review_join_pending':True,
            'limitations':['12 source families and correlated control/rendering views; no independent mechanism count from rotations.',
                           'Absent corpus queries all target missing keys; future-only absence is not supplied.',
                           'Marginal role/outcome crossings do not establish joint nuisance independence; empirical associations remain.',
                           'Each query is independent finite argmax; histories have12-14 events, with no persistent learned memory, semantic relevance, trust/unresolved-conflict or efficacy evidence.',
                           'Corrective expert review was informed by its own v1 criticism. Actual separate executions share an underlying model.']}
    write_new(out,report)
    print(json.dumps({k:report[k] for k in ['counts','status_counts','property_row_counts','conditional_empirical_status_majority_fractions','cross_release_lineage'] if k!='cross_release_lineage'},indent=2))
    print('AUDIT_SHA256',sha(out))

if __name__=='__main__':main()
