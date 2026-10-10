#!/usr/bin/env python3
"""Prompt-derived typed REM audit. No producer, reference, oracle, or reviewer imports.
Only candidate rendered inputs + public contract determine predictions and witnesses.
Producer property_witnesses/role assignments/counterfactual expectations are not read.
"""
import copy,json,pathlib,hashlib,collections,statistics
from adversarial_audit import ROOT, EXECUTION, SCORE_KEYS, canon, digest, require, issue, reorder
from extended_adversarial_audit import episodic
BATCH='memory-gap-v2'
P=ROOT/'data/REM-001/candidate_renderings'/f'{BATCH}.jsonl'
OUT=ROOT/'reviews/adversarial'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);new=json.dumps(obj,indent=2,ensure_ascii=False)+'\n'
    if p.exists() and p.read_text()!=new:
        old=p.read_bytes();h=hashlib.sha256(old).hexdigest();hist=p.parent/'history'/f'{p.stem}-{h}{p.suffix}';hist.parent.mkdir(parents=True,exist_ok=True)
        if not hist.exists():hist.write_bytes(old)
    p.write_text(new)
def freeze(p,items):
    text=''.join(canon(x)+'\n' for x in items)
    if p.exists():require(p.read_text()==text,'Refuse to change frozen initial evidence '+str(p))
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def event_answer(e):return ('deleted',canon(None)) if e['op']=='delete' else ('present',canon(e['value']))
def answer_sem(a):return (a['status'],canon(a['value']))
def eligible(t,q):return [(e['version'],e['at'],i) for i,e in enumerate(t['events']) if e['key']==q['key'] and e['at']<=q['as_of']]
def winner(t,q):return max(eligible(t,q),default=None)
def qs(t,i):return {'query_id':t['queries'][i]['query_id'],'position':i,'key':t['queries'][i]['key'],'as_of':t['queries'][i]['as_of']}
def describe(t,i):return {'index':i,'tuple':[t['events'][i]['version'],t['events'][i]['at'],i],'event':t['events'][i]}
def witnesses(t,a):
    ws=collections.defaultdict(list);rs=a['results']
    for j,q in enumerate(t['queries']):
        es=eligible(t,q);w=winner(t,q);actual=answer_sem(rs[j]);info={'query':qs(t,j),'actual':rs[j]}
        if w is None:
            ws['absence_control'].append({**info,'absence_class':'future_only' if any(e['key']==q['key'] for e in t['events']) else 'never_key'})
        else:
            we=t['events'][w[2]];wa=event_answer(we)
            latest=max(es,key=lambda z:z[1:])
            comps=[x for x in es if x[0]<w[0] and x[1]>w[1]]
            if comps and event_answer(t['events'][latest[2]])!=wa:
                ws['version_over_recency'].append({**info,'winner':describe(t,w[2]),'later_lower_version':describe(t,latest[2])})
            older=[x for x in es if x[0]==w[0] and x[1]<w[1] and event_answer(t['events'][x[2]])!=wa]
            if older:ws['equal_version_time_conflict'].append({**info,'winner':describe(t,w[2]),'older_top_version_conflict':describe(t,older[0][2])})
            ties=[x for x in es if x[:2]==w[:2] and event_answer(t['events'][x[2]])!=wa]
            if ties:ws['equal_version_time_input_conflict'].append({**info,'winner':describe(t,w[2]),'active_conflicting_tie':describe(t,ties[0][2])})
            puts=[x for x in es if x[0]<w[0] and t['events'][x[2]]['op']=='put']
            if we['op']=='delete' and puts:ws['authoritative_tombstone'].append({**info,'winner':describe(t,w[2]),'lower_version_put':describe(t,puts[0][2])})
        allsame=[(e['version'],e['at'],i) for i,e in enumerate(t['events']) if e['key']==q['key']]
        fw=max(allsame,default=None)
        if fw and t['events'][fw[2]]['at']>q['as_of'] and event_answer(t['events'][fw[2]])!=actual:
            ws['future_event_filter_control'].append({**info,'wrong_if_future_eligible':describe(t,fw[2])})
        allkeys=[(e['version'],e['at'],i) for i,e in enumerate(t['events']) if e['at']<=q['as_of']]
        kw=max(allkeys,default=None)
        if kw and t['events'][kw[2]]['key']!=q['key'] and event_answer(t['events'][kw[2]])!=actual:
            ws['cross_key_filter_control'].append({**info,'wrong_without_key_filter':describe(t,kw[2])})
        for k,r in enumerate(t['queries']):
            if q['key']==r['key'] and q['as_of']>=1 and r['as_of']==q['as_of']-1 and w is not None and w[1]==q['as_of'] and actual!=answer_sem(rs[k]):
                ws['decisive_delayed_boundary'].append({'before_query':qs(t,k),'at_query':qs(t,j),'before':rs[k],'at':rs[j],'winner_at_boundary':describe(t,w[2])})
    pnull=[i for i,r in enumerate(rs) if r['status']=='present' and r['value'] is None]
    others=[i for i,r in enumerate(rs) if r['status'] in ('deleted','absent') and r['value'] is None]
    if pnull and others:ws['present_null_distinction'].append({'present_query':qs(t,pnull[0]),'other_query':qs(t,others[0]),'present':rs[pnull[0]],'other':rs[others[0]]})
    return dict(ws)

def probes(t,a,w):
    checks=[];evidence=[]
    def test(name,task,expected,protected=None):
        actual=episodic(task)
        if protected is None:require(canon(actual)==canon(expected),name)
        else:
            for j in protected:require(canon(actual['results'][j])==canon(expected['results'][j]),name+str(j))
        checks.append(name);return actual
    test('object_field_order',reorder(t),a)
    z=copy.deepcopy(t);z['events'].append({'key':'__audit_unqueried__','version':99999,'at':0,'op':'put','value':True});test('irrelevant_key_append',z,a)
    z=copy.deepcopy(t)
    for e in z['events']:e['at']+=13
    for q in z['queries']:q['as_of']+=13
    test('uniform_time_translation',z,a)
    z=copy.deepcopy(t);z['queries'].reverse();test('query_order',z,{'results':list(reversed(a['results']))})
    z=copy.deepcopy(t);expected=copy.deepcopy(a)
    for j,(q,r) in enumerate(zip(z['queries'],expected['results'])):q['query_id']=r['query_id']='opaque_'+str(19-j)
    test('query_identifier_renaming',z,expected)
    z=copy.deepcopy(t);keys=sorted({e['key'] for e in t['events']}|{q['key'] for q in t['queries']});mapping={k:'opaque_key_'+str(i) for i,k in enumerate(keys)};expected=copy.deepcopy(a)
    for e in z['events']:e['key']=mapping[e['key']]
    for q,r in zip(z['queries'],expected['results']):q['key']=mapping[q['key']];r['key']=mapping[r['key']]
    test('consistent_key_renaming',z,expected)
    # Invariance is explicitly scoped to a protected query, never all queries.
    for j,q in enumerate(t['queries']):
        ww=winner(t,q)
        if ww and ww[0]>0:
            z=copy.deepcopy(t);z['events'].append({'key':q['key'],'version':ww[0]-1,'at':q['as_of'],'op':'put','value':'audit_low'});test('lower_version_append_protected',z,a,[j]);break
    j=0;q=t['queries'][j];z=copy.deepcopy(t)
    for e in z['events']:
        if e['key']!=q['key']:e.update(version=12345,at=0,op='put',value=True)
    test('other_key_edit_protected',z,a,[j])
    for j,q in enumerate(t['queries']):
        future=[i for i,e in enumerate(t['events']) if e['key']==q['key'] and e['at']>q['as_of']]
        if future:
            z=copy.deepcopy(t)
            for i in future:z['events'][i].update(version=99999,op='put',value=False)
            test('future_event_edit_protected',z,a,[j]);break
    # Construct independent decisive counterfactuals from actual prompt witnesses.
    for prop,name in [('version_over_recency','version_priority_counterfactual'),('equal_version_time_input_conflict','active_input_tie_swap_counterfactual')]:
        witness=w[prop][0];j=witness['query']['position'];z=copy.deepcopy(t)
        if prop=='version_over_recency':
            i=witness['later_lower_version']['index'];z['events'][i]['version']=max(x[0] for x in eligible(t,t['queries'][j]))+1
            expected_event=i
        else:
            i=witness['active_conflicting_tie']['index'];v=witness['winner']['index'];z['events'][i],z['events'][v]=z['events'][v],z['events'][i];expected_event=v
        after=episodic(z)['results'][j];require(answer_sem(after)!=answer_sem(a['results'][j]) and after['chosen_event']==expected_event,name)
        checks.append(name);evidence.append({'control':name,'scope':'audit_probe','protected_query_position':j,'before':a['results'][j],'after':after})
    j=next(i for i,r in enumerate(a['results']) if r['status']=='present' and r['value'] is None);i=a['results'][j]['chosen_event'];z=copy.deepcopy(t);z['events'][i]['op']='delete';after=episodic(z)['results'][j]
    require(after['status']=='deleted' and after['value'] is None,'null lifecycle');checks.append('null_lifecycle_counterfactual');evidence.append({'control':'null_lifecycle_counterfactual','scope':'audit_probe','protected_query_position':j,'before':a['results'][j],'after':after})
    j=next(i for i,r in enumerate(a['results']) if r['status']=='absent');q=t['queries'][j]
    key=next(k for k in keys if winner(t,{**q,'key':k}) and event_answer(t['events'][winner(t,{**q,'key':k})[2]])!=answer_sem(a['results'][j]))
    z=copy.deepcopy(t);z['queries'][j]['key']=key;after=episodic(z)['results'][j];require(answer_sem(after)!=answer_sem(a['results'][j]),'query-key cf');checks.append('query_key_counterfactual');evidence.append({'control':'query_key_counterfactual','scope':'audit_probe','protected_query_position':j,'before':a['results'][j],'after':after})
    return checks,evidence

def decode(t):
    # Semantic groups inferred from visible null lifecycle/active tie, not producer role labels.
    counts=collections.Counter(e['key'] for e in t['events']);allkeys=set(counts)|{q['key'] for q in t['queries']}
    absent={k for k in allkeys if not counts[k]}
    lifecycle={e['key'] for e in t['events'] if e['value'] is None}
    tied=set()
    for q in t['queries']:
        w=winner(t,q)
        if w and any(x[:2]==w[:2] and event_answer(t['events'][x[2]])!=event_answer(t['events'][w[2]]) for x in eligible(t,q)):tied.add(q['key'])
    require(len(allkeys)==4 and len(absent)==len(lifecycle)==len(tied)==1 and not (absent&lifecycle or absent&tied or lifecycle&tied),'unambiguous input-derived key groups')
    priority=allkeys-absent-lifecycle-tied;require(len(priority)==1,'one version-recency group')
    group={k:'absence' for k in absent};group.update({k:'null_lifecycle' for k in lifecycle});group.update({k:'active_index_tie' for k in tied});group.update({k:'version_recency' for k in priority})
    cutoff={k:sorted({q['as_of'] for q in t['queries'] if q['key']==k}) for k in allkeys}
    roles=[group[q['key']]+'-cutoff-'+str(cutoff[q['key']].index(q['as_of'])) for q in t['queries']]
    require(len(roles)==len(set(roles))==8,'eight distinct input-derived roles')
    z=copy.deepcopy(t)
    for e in z['events']:e['key']=group[e['key']]
    for q in z['queries']:q['key']=group[q['key']];q.pop('query_id')
    z['queries'].sort(key=lambda q:(q['key'],q['as_of']))
    return group,roles,z

def exact_diff(a,b):
    ds=[]
    for i,(e,f) in enumerate(zip(a['events'],b['events'])):
        for k in e:
            if canon(e[k])!=canon(f[k]):ds.append({'event_index':i,'field':k,'before':e[k],'after':f[k]})
    require(canon(a['queries'])==canon(b['queries']),'matched corpus queries differ after key decoding')
    return ds

def typename(v):return 'null' if v is None else {bool:'boolean',int:'integer',str:'string'}[type(v)]
def label(r,kind):
    if kind=='status':return r['status']
    if kind=='type':return typename(r['value'])
    if kind=='present_null':return str(r['status']=='present' and r['value'] is None)
    return canon([r['status'],r['value'],r['chosen_version'],r['chosen_event']])
def hist(os,key,kind):
    out=collections.defaultdict(collections.Counter)
    for o in os:out[str(key(o))][label(o['answer'],kind)]+=1
    return {k:dict(sorted(v.items())) for k,v in sorted(out.items())}
def baseline(train,dev,key,kind):
    majority=collections.Counter(label(o['answer'],kind) for o in train).most_common(1)[0][0];table=collections.defaultdict(collections.Counter)
    for o in train:table[str(key(o))][label(o['answer'],kind)]+=1
    pred={k:c.most_common(1)[0][0] for k,c in table.items()}
    hit=sum(pred.get(str(key(o)),majority)==label(o['answer'],kind) for o in dev)
    return {'correct':hit,'total':len(dev),'accuracy':hit/len(dev),'seen_feature_count':sum(str(key(o)) in pred for o in dev),'majority_accuracy':sum(label(o['answer'],kind)==majority for o in dev)/len(dev)}

def main():
    rows=[json.loads(s) for s in P.read_text().splitlines() if s.strip()];data=[];predictions=[];receipts=[]
    for r in rows:
        errors=[];t=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);a=episodic(t);w=witnesses(t,a)
        try:checks,cf=probes(t,a,w)
        except Exception as e:checks=[];cf=[];errors.append(str(e))
        require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'candidate hash mismatch')
        require(canon(t)==canon(r['semantic_task']),'rendered task differs typed semantic payload')
        for field in ('sample_id','record_hash','semantic_parent_id','semantic_family_id','source_family_key'):
            require(r[field] not in r['rendered_input'],'metadata leakage '+field)
        group,roles,norm=decode(t)
        predictions.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':EXECUTION,'input_derived_answer':a,'property_witnesses':w,'metamorphic_checks':checks,'audit_probe_counterfactuals':cf,'errors':errors})
        data.append({'r':r,'t':t,'a':a,'w':w,'checks':checks,'cf':cf,'errors':errors,'group':group,'roles':roles,'normalized':norm})
    # Predictions + own witnesses frozen BEFORE the first verified_target read below.
    freeze(OUT/f'{BATCH}.input_predictions.jsonl',predictions)
    for d in data:
        r=d['r'];issues=[];critical=[];scores={k:5 for k in SCORE_KEYS};scores.update(generalization_value=3,difficulty_quality=3,diversity=3,shortcut_resistance=4)
        if canon(d['a'])!=canon(r['verified_target']):critical.append('target_mismatch');scores['target_correctness']=scores['reasoning_validity']=1;issues.append(issue('target_mismatch','critical',{'independent':d['a'],'observed':r['verified_target']},'Correct target then obtain fresh exact-hash independent reviews.'))
        for tag in r['skill_tags']:
            if tag not in ('finite_synthetic_proxy','versioned_retrieval') and not d['w'].get(tag):critical.append('unsupported_'+tag);scores['semantic_correctness']=2;issues.append(issue('unsupported_property_tag','high',{'tag':tag,'witness_search':'All visible queries and eligible tuple conflicts exhausted.'},'Remove false factual tag or supply a fresh task with a semantic witness.'))
        if d['errors']:critical.append('metamorphic_failure');issues.append(issue('metamorphic_failure','high',d['errors'],'Fix violated public contract or missing coverage and re-review.'))
        receipts.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'receipt_id':'adv-'+BATCH+'-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:20],'reviewer_role':'Adversarial Red-Team','additional_audit_role':'Diversity and leakage auditor','reviewer_execution_id':EXECUTION,'scores':scores,'critical_defects':critical,'issues':issues,'decision':'REVISE' if critical else 'ACCEPT','confidence':0.97,'evidence':{'prediction_frozen_before_gold_comparison':True,'input_only_solver':'Own typed episodic solver; exact key, cutoff, tuple(version,time,input index).','property_witnesses':{k:v[0] for k,v in d['w'].items()},'metamorphic_checks':d['checks'],'private_producer_witnesses_or_other_reviewer_verdicts_read':False,'scope':'Finite explicit-history retrieval. Same-model independent execution; exposed development families, correlated variants, no sealed custody or learned memory efficacy.'}})
    freeze(OUT/f'{BATCH}.first_judgments.jsonl',receipts)
    print(canon({'phase':'first_judgments_frozen','records':len(rows),'decisions':dict(collections.Counter(x['decision'] for x in receipts))}),flush=True)
    # Independently reconstruct crossings and instantiated corpus counterfactuals.
    controls=[];problems=[];by=collections.defaultdict(list);obs=[]
    for d in data:
        r=d['r'];by[(r['source_family_key'],r['controlled_variant'],r['renderer'])].append(d)
        for j,(q,a,role) in enumerate(zip(d['t']['queries'],d['a']['results'],d['roles'])):
            obs.append({'sample_id':r['sample_id'],'family':r['source_family_key'],'history':r['controlled_variant'],'renderer':r['renderer'],'split':r['split'],'id':q['query_id'],'position':j,'key':q['key'],'keygroup':d['group'][q['key']],'role':role,'as_of':q['as_of'],'query_text_length':len(canon(q)),'input_length':len(r['rendered_input']),'answer':a})
    for (family,history,style),ds in sorted(by.items()):
        require(len(ds)==8 and len({canon(d['normalized']) for d in ds})==1,'eight consistent rotations for history')
        os=[o for o in obs if (o['family'],o['history'],o['renderer'])==(family,history,style)]
        role_id=hist(os,lambda o:canon([o['role'],o['id']]),'status');role_pos=hist(os,lambda o:canon([o['role'],o['position']]),'status')
        require(len(role_id)==len(role_pos)==64 and all(sum(v.values())==1 for v in role_id.values()) and all(sum(v.values())==1 for v in role_pos.values()),'role ID/position crossing')
        assignment=collections.Counter((d['group'][key],key) for d in ds for key in d['group'])
        require(len(assignment)==16 and set(assignment.values())=={2},'four-way key crossing')
        histograms={};
        for feature in ('id','position','key'):
            histograms[feature]={kind:hist(os,lambda o:o[feature],kind) for kind in ('status','type','present_null')}
            for kind,table in histograms[feature].items():
                if len({canon(v) for v in table.values()})!=1:problems.append({'family':family,'history':history,'style':style,'feature':feature,'kind':kind,'histograms':table})
        require({'present','deleted','absent'}<=set(o['answer']['status'] for o in os),'each history three statuses')
        controls.append({'family':family,'history':history,'field_order':style,'scope':'corpus_variant','input_derived_roles':sorted({o['role'] for o in os}),'keygroup_assignment_counts':{canon(k):v for k,v in sorted(assignment.items())},'role_by_id':role_id,'role_by_position':role_pos,'nuisance_outcome_histograms':histograms})
    field_groups=collections.defaultdict(list)
    for d in data:field_groups[(d['r']['source_family_key'],d['r']['controlled_variant'],d['r']['query_rotation'])].append(d)
    for ds in field_groups.values():
        require(len(ds)==2 and {d['r']['renderer'] for d in ds}=={'forward','reverse'},'field-order pair')
        require(canon(ds[0]['t'])==canon(ds[1]['t']) and canon(ds[0]['a'])==canon(ds[1]['a']),'typed field-order invariance')
        require(ds[0]['r']['rendered_input']!=ds[1]['r']['rendered_input'],'different actual field orders')
    actual_cf=[]
    byfamily=collections.defaultdict(dict)
    for d in data:
        if d['r']['query_rotation']==0 and d['r']['renderer']=='forward':byfamily[d['r']['source_family_key']][d['r']['controlled_variant']]=d
    for family,ds in sorted(byfamily.items()):
        base=ds['baseline'];bt=base['normalized']
        for variant in ('version_priority_raise','lifecycle_op_toggle','index_tie_swap'):
            v=ds[variant];diff=exact_diff(bt,v['normalized']);bans=episodic({**bt,'queries':[{**q,'query_id':str(i)} for i,q in enumerate(bt['queries'])]})['results'];vans=episodic({**v['normalized'],'queries':[{**q,'query_id':str(i)} for i,q in enumerate(v['normalized']['queries'])]})['results'];changed=[i for i,(x,y) in enumerate(zip(bans,vans)) if answer_sem(x)!=answer_sem(y)]
            require(changed,'corpus counterfactual not semantically decisive')
            if variant=='version_priority_raise':require(len(diff)==1 and diff[0]['field']=='version' and diff[0]['after']>diff[0]['before'],'corpus version-only change')
            if variant=='lifecycle_op_toggle':require(len(diff)==1 and diff[0]['field']=='op' and {diff[0]['before'],diff[0]['after']}=={'put','delete'} and bt['events'][diff[0]['event_index']]['value'] is None,'corpus null-op-only change')
            if variant=='index_tie_swap':
                indices=sorted({x['event_index'] for x in diff});require(len(indices)==2,'corpus exactly two-event swap');i,j=indices;require(canon(bt['events'][i])==canon(v['normalized']['events'][j]) and canon(bt['events'][j])==canon(v['normalized']['events'][i]),'corpus exact event swap');require(bt['events'][i]['version']==bt['events'][j]['version'] and bt['events'][i]['at']==bt['events'][j]['at'],'corpus active tuple tie')
            actual_cf.append({'family':family,'control':variant,'scope':'corpus_variant','base_sample_id':base['r']['sample_id'],'variant_sample_id':v['r']['sample_id'],'base_record_hash':base['r']['record_hash'],'variant_record_hash':v['r']['record_hash'],'changes_after_consistent_key_decoding':diff,'semantically_changed_query_signatures':[bt['queries'][i] for i in changed],'before_after':[[bans[i],vans[i]] for i in changed]})
    # Only marginal/conditionally nuisance correlations count as shortcuts; temporal signals are public semantics.
    train=[o for o in obs if o['split']=='train'];dev=[o for o in obs if o['split']!='train'];features={'query_id':lambda o:o['id'],'position':lambda o:o['position'],'query_key_string':lambda o:o['key'],'query_key_length':lambda o:len(o['key']),'query_id_position':lambda o:(o['id'],o['position']),'input_char_length':lambda o:o['input_length'],'input_char_length_query_id':lambda o:(o['input_length'],o['id']),'query_serialized_length':lambda o:o['query_text_length'],'as_of_time_semantic':lambda o:o['as_of'],'input_derived_semantic_keygroup_requires_pattern_decoding':lambda o:o['keygroup']}
    baselines={name:{kind:baseline(train,dev,fn,kind) for kind in ('status','type','present_null','full_query_target')} for name,fn in features.items()}
    wrong_resolver={name:collections.Counter() for name in ('latest_time','version_then_index','earliest_top_tuple_index','ignore_cutoff','ignore_key','put_null_as_deleted','latest_input_index')};examples={}
    for d in data:
        t=d['t']
        for j,q in enumerate(t['queries']):
            es=eligible(t,q);orig=d['a']['results'][j]
            tuples={'latest_time':max(es,key=lambda x:x[1:],default=None),'version_then_index':max(es,key=lambda x:(x[0],x[2]),default=None),'earliest_top_tuple_index':max(es,key=lambda x:(x[0],x[1],-x[2]),default=None),'ignore_cutoff':max([(e['version'],e['at'],i) for i,e in enumerate(t['events']) if e['key']==q['key']],default=None),'ignore_key':max([(e['version'],e['at'],i) for i,e in enumerate(t['events']) if e['at']<=q['as_of']],default=None),'put_null_as_deleted':max(es,default=None),'latest_input_index':max(es,key=lambda x:x[2],default=None)}
            for name,z in tuples.items():
                ans={'query_id':q['query_id'],'key':q['key'],'status':'absent','value':None,'chosen_version':None,'chosen_event':None}
                if z:
                    e=t['events'][z[2]];ans.update(status='deleted' if e['op']=='delete' else 'present',value=e['value'],chosen_version=e['version'],chosen_event=z[2])
                if name=='put_null_as_deleted' and ans['status']=='present' and ans['value'] is None:ans['status']='deleted'
                wrong_resolver[name]['observations']+=1;wrong_resolver[name]['full_target_correct']+=canon(ans)==canon(orig);wrong_resolver[name]['semantic_answer_correct']+=answer_sem(ans)==answer_sem(orig)
                if answer_sem(ans)!=answer_sem(orig) and name not in examples:
                    # Reduce to protected query and decisive selected events; recompute all new indices.
                    keep=sorted({x for x in (orig['chosen_event'],ans['chosen_event']) if x is not None});mt={'kind':'episodic','events':[t['events'][i] for i in keep],'queries':[q]}
                    examples[name]={'source_sample_id':d['r']['sample_id'],'source_record_hash':d['r']['record_hash'],'minimal_task':mt,'correct':episodic(mt),'incorrect_original_answer':ans,'original_correct':orig,'warning':'Minimal tasks renumber events; incorrect_original_answer retains original index and demonstrates semantic discrepancy.'}
    # Type probes are diagnostic synthetic tasks, not extra supplied coverage.
    typed={'kind':'episodic','events':[{'key':'k','version':1,'at':0,'op':'put','value':True},{'key':'k','version':1,'at':0,'op':'put','value':1}],'queries':[{'query_id':'q','key':'k','as_of':0}]};ta=episodic(typed);require(type(ta['results'][0]['value']) is int,'boolean/int conflation');typed_swap=copy.deepcopy(typed);typed_swap['events'].reverse();require(type(episodic(typed_swap)['results'][0]['value']) is bool,'typed index tie')
    malformed=[]
    for loc,val in [('version',True),('at',True),('as_of',True),('delete_value',1)]:
        z=copy.deepcopy(typed)
        if loc=='as_of':z['queries'][0]['as_of']=val
        elif loc=='delete_value':z['events'][0].update(op='delete',value=val)
        else:z['events'][0][loc]=val
        try:episodic(z)
        except ValueError:malformed.append(loc)
        else:raise ValueError('accepted malformed '+loc)
    # Exact duplicate and structural overlap counts use only actual rendered tasks.
    duplicate=collections.defaultdict(list);normalized=collections.defaultdict(list);parents=collections.defaultdict(list)
    for d in data:duplicate[digest(d['t'])].append(d);normalized[digest(d['normalized'])].append(d);parents[d['r']['source_family_key']].append(d)
    family_split_violations=[k for k,ds in parents.items() if len({d['r']['split'] for d in ds})!=1]
    require(not family_split_violations,'source-family split leakage')
    def structure(d):
        z=copy.deepcopy(d['normalized']);version_ranks={v:i for i,v in enumerate(sorted({e['version'] for e in z['events']}))};time_ranks={v:i for i,v in enumerate(sorted({e['at'] for e in z['events']}|{q['as_of'] for q in z['queries']}))};value_symbols={};count=collections.Counter()
        for e in z['events']:
            e['version']=version_ranks[e['version']];e['at']=time_ranks[e['at']]
            if e['value'] is not None:
                typ=typename(e['value']);v=canon(e['value'])
                if v not in value_symbols:value_symbols[v]=typ+'_'+str(count[typ]);count[typ]+=1
                e['value']=value_symbols[v]
        for q in z['queries']:q['as_of']=time_ranks[q['as_of']]
        return z
    shapes=collections.defaultdict(list)
    for d in data:shapes[digest(structure(d))].append(d)
    # Prompt vocabulary is held out by two families; structure is exposed and correlated by design.
    div={'batch':BATCH,'reviewer_execution_id':EXECUTION,'records':len(rows),'rendered_query_observations':len(obs),'rotation_level_query_observations':len(obs)//2,'base_history_query_observations':sum(len(ds[0]['t']['queries']) for ds in by.values() if ds[0]['r']['renderer']=='forward'),'source_families':len(parents),'base_controlled_histories':len(by)//2,'split_records':dict(collections.Counter(r['split'] for r in rows)),'split_source_families':{s:len({r['source_family_key'] for r in rows if r['split']==s}) for s in {r['split'] for r in rows}},'field_order_pairs':len(field_groups),'exact_task_duplicate_groups':len([ds for ds in duplicate.values() if len(ds)>1]),'exact_task_duplicates_cross_family':[{'sample_ids':[d['r']['sample_id'] for d in ds]} for ds in duplicate.values() if len({d['r']['source_family_key'] for d in ds})>1],'normalized_base_history_groups':len(normalized),'family_split_violations':family_split_violations,'mechanism_motif_count_rank_names_values_and_query_id_stripped':len(shapes),'mechanism_motifs_cross_split':sum(len({d['r']['split'] for d in ds})>1 for ds in shapes.values()),'motif_groups':[{'shape_hash':h,'records':len(ds),'families':sorted({d['r']['source_family_key'] for d in ds}),'splits':sorted({d['r']['split'] for d in ds})} for h,ds in shapes.items()],'source_provenance_assertion_counts':dict(collections.Counter(canon(r['source_provenance']) for r in rows)),'property_support':dict(collections.Counter(k for d in data for k,v in d['w'].items() if v)),'absence_witness_counts':dict(collections.Counter(v['absence_class'] for d in data for v in d['w'].get('absence_control',[]))),'role_definition':'Inferred from absent key, null-valued lifecycle group, active index-tie group, remaining version-recency group, and within-key query cutoff rank. Producer role maps and witnesses never read.','family_crossings':controls,'corpus_counterfactuals':actual_cf,'audit_probe_scope':{'lower_version_append_invariance':'One existing winner per row protected; original indices preserved.','other_key_edit_invariance':'First query protected only; own-key history untouched.','future_event_edit_invariance':'One query cutoff per row protected; strictly future same-key events modified; all indices fixed.','query_key_counterfactual':'Absent query key replaced by eligible other key at same cutoff.','counts':'Audit probes do not add any families or supplied records.'},'limitations':['768 correlated renderings are 12 source families / 48 controlled histories, not 768 independent memory mechanisms.','Rank normalization retains event order; near-isomorphic shuffled templates may therefore be undercounted. Cross-split motifs are exposed, not sealed structural OOD.','Absent queries all have never-used keys; future-only absence is not supplied.','All supplied queries independent; no query result changes another query. No persistent learned-memory, associative retrieval, source trust, unresolved-conflict or efficacy claim.','Reviewers share underlying model; separate actual executions do not make model errors independent.']}
    shortcut={'batch':BATCH,'reviewer_execution_id':EXECUTION,'cpu_baseline_scope':'Train lookup and majority on exposed dev observations; correlated renderings and tiny two-family dev sample, no model training.','conditional_nuisance_histogram_failures':problems,'baselines':baselines,'global_outcome_histograms':{kind:hist(obs,lambda o:'all',kind)['all'] for kind in ('status','type','present_null')},'split_outcome_histograms':{kind:hist(obs,lambda o:o['split'],kind) for kind in ('status','type','present_null')},'incorrect_resolver_accuracy':{k:dict(v) for k,v in wrong_resolver.items()},'ID_position_coupling':sorted({(o['id'],o['position']) for o in obs}),'ID_position_scope':'IDs and positions need not be mutually independent; each is crossed with all eight roles per complete supplied history set.','typed_synthetic_probe':{'task':typed,'correct':ta,'swapped_correct':episodic(typed_swap),'scope':'audit_probe','malformed_rejected':malformed},'minimal_counterexamples':examples,'limitations':['Output length depends on exact status/value/provenance format by design; only input nuisance features assessed as outcome leakage.','Meaningful key-history multiplicity and time/version features are part of public semantics; partial-label prediction from them does not establish complete-target leakage.','Equal ID/position/key histograms control these nuisance marginals within complete history sets; higher-order semantic mechanisms remain simple finite tuple retrieval.']}
    for receipt in receipts:
        receipt['issues'].append(issue('correlated_finite_history_scope','low',{'source_families':12,'controlled_histories':48,'dev_source_families':2,'property_scope':'Finite explicit-history deterministic retrieval; same-model exposed-dev review.'},'Use sealed new mechanism families and actual model evaluations before efficacy claims; count correlated variants separately.'))
        if problems:receipt['issues'].append(issue('conditional_nuisance_leakage','high',problems,'Cross all conditional outcomes or narrow claims.'));receipt['decision']='REVISE';receipt['scores']['shortcut_resistance']=2
    # Only own final artifacts are written. All first judgments remain preserved.
    freeze(OUT/f'{BATCH}.jsonl',receipts)
    save(ROOT/'audits/diversity'/f'{BATCH}-adversarial.json',div);save(ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json',shortcut)
    execution={'batch':BATCH,'reviewer_execution_id':EXECUTION,'status':'completed','candidate_file':str(P.relative_to(ROOT)),'candidate_file_sha256':sha(P),'records':len(rows),'decisions':dict(collections.Counter(r['decision'] for r in receipts)),'input_predictions_sha256':sha(OUT/f'{BATCH}.input_predictions.jsonl'),'first_judgments_sha256':sha(OUT/f'{BATCH}.first_judgments.jsonl'),'final_receipts_sha256':sha(OUT/f'{BATCH}.jsonl'),'script_sha256':sha(pathlib.Path(__file__)),'own_solver_sha256':sha(ROOT/'audits/extended_adversarial_audit.py'),'own_helpers_sha256':sha(ROOT/'audits/adversarial_audit.py'),'public_contract_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'schemas/EXTENDED_SEMANTICS.md',ROOT/'schemas/MEMORY_CONTROL_PROPERTIES.md']},'diversity_audit_sha256':sha(ROOT/'audits/diversity'/f'{BATCH}-adversarial.json'),'shortcut_audit_sha256':sha(ROOT/'audits/shortcut_detection'/f'{BATCH}-adversarial.json'),'metamorphic_check_counts':dict(collections.Counter(c for d in data for c in d['checks'])),'target_mismatches':sum('target_mismatch' in r['critical_defects'] for r in receipts),'unsupported_property_rows':sum(any(c.startswith('unsupported_') for c in r['critical_defects']) for r in receipts),'conditional_histogram_failures':len(problems),'actual_corpus_counterfactual_pairs':len(actual_cf),'first_predictions_and_judgments_preserved':True,'forbidden_sources_read':False,'scope':'Own prompt-derived typed recomputation, all property searches, complete family controls, synthetic metamorphic probes. Same underlying model, exposed dev, no independent reviewer efficacy or sealed OOD claim.'}
    save(ROOT/'audits/execution'/f'{BATCH}-adversarial.json',execution);print(canon(execution),flush=True)

if __name__=='__main__':main()
