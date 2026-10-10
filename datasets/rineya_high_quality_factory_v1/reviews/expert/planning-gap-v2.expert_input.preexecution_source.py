#!/opt/anaconda3/bin/python3
"""Actual expert V2 input-only prediction/first-input-judgment stage.

Read allowlist: stripped input view, its public custody freeze, own V2 prep,
own frozen V1 solver, seven public specs/root public freezes. No stored gold,
producer controls/source, formal implementation, other private reviews.
"""
import collections, copy, datetime, hashlib, importlib.util, itertools, json, pathlib, re, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/'reviews/expert'; ROLE='/root/expert_reviewer'
VIEW='reviews/input_only/planning-gap-v2.jsonl'
VIEW_SHA='00bd1c2de5d0d4719957bb2ded44e2342940de4d368c65b646817474fe79b73c'
PREP='reviews/expert/prepare_rgm_planning_gap_v2.py'
PREP_SHA='7c74fcebba1057bcdb0313c7c658b57a0752d041558f82534fb44a573d98fc53'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((ROOT/PREP).read_bytes())==PREP_SHA
s=importlib.util.spec_from_file_location('own_public_v2_preparation',ROOT/PREP); prep=importlib.util.module_from_spec(s);s.loader.exec_module(prep)
v1=prep.v1;eq=prep.eq

def bind(path):
    b=(ROOT/path).read_bytes();return {'path':path,'sha256':sha(b),'bytes':len(b)}
def normalized(task):
    t=copy.deepcopy(task)
    if t['kind']=='graph':t['nodes']=sorted(t['nodes']);t['edges']=sorted(t['edges'])
    return t
def signature(task):return v1.canon(normalized(task))
def differences(a,b):
    aa,bb=normalized(a),normalized(b)
    return [k for k in sorted(set(aa)|set(bb)) if not eq(aa.get(k),bb.get(k))]
def family_index(key):
    m=re.fullmatch(r'RGM-001-seed20261009-p(\d+)',key)
    if not m:raise ValueError('unexpected public family key')
    return int(m.group(1))
def stratum(index):return 'A' if index<192 else 'B' if index<204 else 'C' if index<216 else 'D'
def block_shape(task,joint=True):
    names=sorted(task['on']);best=None
    for permutation in itertools.permutations(range(len(names))):
        ids=dict(zip(names,permutation));ids['TABLE']=-1
        initial=tuple(sorted((ids[k],ids[d]) for k,d in task['on'].items()))
        goal=tuple(sorted((ids[k],ids[d]) for k,d in task['goal_on'].items())) if joint else ()
        value=(initial,goal)
        if best is None or value<best:best=value
    return repr(best)
def graph_shape(task,weighted=True):
    names=sorted(task['nodes']);best=None
    for permutation in itertools.permutations(range(len(names))):
        ids=dict(zip(names,permutation))
        edges=tuple(sorted((ids[a],ids[b],c) if weighted else (ids[a],ids[b]) for a,b,c in task['edges']))
        value=(ids[task['start']],ids[task['goal']],edges) if weighted else edges
        if best is None or value<best:best=value
    return repr(best)

def input_contexts(items,index):
    """Context identities inferred from solved tasks, not sample-ID spelling."""
    group={};kind=stratum(index)
    def one(context,criterion):
        possible=[i for i in items if criterion(i)]
        if len(possible)!=1:raise ValueError('nonunique inferred '+context+': '+str(len(possible)))
        group[context]=possible[0]
    def target(i):return i['facts']['target']
    def error(i):return i['facts']['measurements']['first_error']
    if kind=='A':
        one('A0',lambda i:target(i)['valid'] and target(i)['reaches_goal'])
        a=group['A0']['task']
        one('A1',lambda i:differences(a,i['task'])==['fuel'] and i['task']['fuel']<a['fuel'])
        one('A2',lambda i:target(i)['valid'] and not target(i)['reaches_goal'])
        one('A3',lambda i:error(i) is not None and error(i)['reason']=='missing_edge')
    elif kind=='B':
        candidates=[]
        for left in items:
            for right in items:
                lt,rt=left['task'],right['task']
                if target(left)['valid'] and target(left)['reaches_goal'] and differences(lt,rt)==['edges']:
                    if len(lt['edges'])==len(rt['edges'])+1 and all(e in lt['edges'] for e in rt['edges']):candidates.append((left,right))
        if len(candidates)!=1:raise ValueError('nonunique B0/B1 input edge pair')
        group['B0'],group['B1']=candidates[0]
        one('B3',lambda i:target(i)['valid'] and target(i)['reaches_goal'] and len(i['task']['plan'])>len(target(i)['shortest_plan']))
        remain=[i for i in items if i not in group.values()]
        if len(remain)!=1:raise ValueError('nonunique B2')
        group['B2']=remain[0]
    elif kind=='C':
        one('C0',lambda i:target(i)['valid'] and target(i)['reaches_goal'] and eq(i['task']['plan'],target(i)['shortest_plan']))
        one('C2',lambda i:target(i)['valid'] and target(i)['reaches_goal'] and len(i['task']['plan'])>len(target(i)['shortest_plan']))
        one('C3',lambda i:target(i)['valid'] and not target(i)['reaches_goal'])
        one('C4',lambda i:i['facts'].get('v2_isolated_noop_witness') is not None)
        remain=[i for i in items if i not in group.values()]
        if len(remain)!=1:raise ValueError('nonunique C1')
        group['C1']=remain[0]
    else:
        one('D1',lambda i:target(i)['valid'] and not target(i)['reaches_goal'])
        d1=group['D1']['task']
        one('D0',lambda i:differences(i['task'],d1)==['goal_on'] and len(i['task']['goal_on'])<len(d1['goal_on']))
        one('D2',lambda i:eq(i['task']['goal_on'],d1['goal_on']) and target(i)['valid'] and target(i)['reaches_goal'] and eq(i['task']['plan'],target(i)['shortest_plan']))
        one('D3',lambda i:eq(i['task']['goal_on'],d1['goal_on']) and target(i)['valid'] and target(i)['reaches_goal'] and len(i['task']['plan'])>len(target(i)['shortest_plan']))
    if len({id(i) for i in group.values()})!=len(items):raise ValueError('input context overlap/omission')
    return group

def audit_family(items,index):
    checks=[];contexts=input_contexts(items,index);kind=stratum(index)
    def check(name,truth,evidence=None):checks.append({'name':name,'pass':bool(truth),'evidence':evidence})
    def t(c):return contexts[c]['task']
    def f(c):return contexts[c]['facts']
    def y(c):return f(c)['target']
    def e(c):return f(c)['measurements']['first_error']
    expected_split='dev_challenge' if index%10 in (8,9) else 'train'
    check('all-family-row-split-custody',all(r['split']==expected_split for i in items for r in i['rows']),expected_split)
    check('exact-context-count',len(contexts)==(5 if kind=='C' else 4))
    for context,item in contexts.items():
        check(context+':exact-two-equivalent-rows',len(item['rows'])==2)
        task=item['task'];size=len(task['nodes']) if kind in 'AB' else len(task['on'])
        check(context+':public-task-bound',len(task['plan'])<=12 and (5<=size<=7 and task['fuel']<=30 if kind in 'AB' else size in (5,6)))
    if kind=='A':
        check('A-fuel-only',differences(t('A0'),t('A1'))==['fuel'])
        check('A0-legal-goal',y('A0')['valid'] and y('A0')['reaches_goal'])
        check('A1-submitted-route-blocked',not y('A1')['valid'] and e('A1')['reason']=='insufficient_fuel')
        subtype='I' if y('A1')['shortest_plan'] is None else 'L'
        expected_I={180,182,184,186,188,190}
        check('A-exact-subtype-allocation',(subtype=='I')==(index in expected_I),subtype)
        if subtype=='I':
            check('A-exact-fuel-boundary',f('A0')['measurements']['cheapest_required_fuel']==t('A0')['fuel'] and t('A1')['fuel']==t('A0')['fuel']-1)
            check('A1-topologically-reachable',f('A1')['measurements']['topology_shortest_plan'] is not None)
        else:
            check('A1-fuel-lengthens-fresh-minimum',len(y('A1')['shortest_plan'])>len(y('A0')['shortest_plan']) and len(y('A1')['shortest_plan'])>len(f('A1')['measurements']['topology_shortest_plan']))
        check('A2-same-world-nonempty-nongoal',differences(t('A0'),t('A2'))==['plan'] and bool(t('A2')['plan']) and y('A2')['valid'] and not y('A2')['reaches_goal'])
        check('A3-missing-edge-after-legal-prefix',differences(t('A0'),t('A3'))==['plan'] and e('A3')['index']>=1 and e('A3')['reason']=='missing_edge')
        expected_sizes=[5,5,5,5,6,6,7,6,6,7,7,7]
        check('A-exact-public-size-allocation',len(t('A0')['nodes'])==expected_sizes[index-180])
    elif kind=='B':
        check('B0-legal-goal-minimum-three-moves',y('B0')['valid'] and y('B0')['reaches_goal'] and len(t('B0')['plan'])>=3)
        check('B1-submitted-missing-edge-after-prefix-fresh-feasible',e('B1') is not None and e('B1')['index']>=1 and e('B1')['reason']=='missing_edge' and y('B1')['shortest_plan'] is not None)
        edge_set={(a,b) for a,b,_ in t('B2')['edges']};error=e('B2')
        pre=f('B2')['measurements']['prefix_states'][error['index']] if error else {}
        dest=t('B2')['plan'][error['index']] if error else None
        check('B2-attempted-absent-directed-reverse-after-prefix',differences(t('B0'),t('B2'))==['plan'] and error is not None and error['index']>=1 and error['reason']=='missing_edge' and (dest,pre.get('node')) in edge_set and (pre.get('node'),dest) not in edge_set)
        check('B3-valid-same-world-detour',differences(t('B0'),t('B3'))==['plan'] and 'valid_goal_detour' in f('B3')['row_properties'])
        check('B-exact-public-size-allocation',len(t('B0')['nodes'])==[5,6,7,5,6,7,5,6,7,5,6,7][index-192])
    elif kind=='C':
        check('C0-canonical-four-to-eight-minimum',eq(t('C0')['plan'],y('C0')['shortest_plan']) and 4<=len(t('C0')['plan'])<=8)
        check('C0-actual-changed-clearance',bool(f('C0')['v2_canonical_clearance_witnesses']))
        swaps=[]
        for i in range(len(t('C0')['plan'])-1):
            candidate=copy.deepcopy(t('C0')['plan']);candidate[i:i+2]=reversed(candidate[i:i+2])
            if eq(candidate,t('C1')['plan']):swaps.append(i)
        check('C1-exact-adjacent-only-plan-swap',differences(t('C0'),t('C1'))==['plan'] and bool(swaps))
        error=e('C1');idx=error['index'];guards=f('C1')['v2_guard_facts']['guard_vectors'][idx]
        reason=error['reason'];required_reason='destination_not_clear' if index in {207,209,210,212,213,214} else 'source_not_clear'
        omission='source_clear_ignored' if reason=='source_not_clear' else 'destination_clear_ignored'
        selected=prep.OMISSION[omission]
        check('C1-exact-clearance-class-allocation',reason==required_reason,{'reason':reason,'first_index':idx,'swap_indices':swaps})
        check('C1-sole-named-false-guard',[k for k in prep.GUARDS if not guards[k]]==[selected],guards)
        strict={k:y('C1')[k] for k in ('steps','after','valid','reaches_goal')};mutant=f('C1')['diagnostic_mutant_outputs'][omission]
        check('C1-executed-single-guard-mutation-kill',mutant['steps'][idx]['legal'] and not eq(strict,mutant))
        check('C2-valid-same-world-detour',differences(t('C0'),t('C2'))==['plan'] and 'valid_goal_detour' in f('C2')['row_properties'])
        k=1+(index-204)%3
        check('C3-exact-C0-prefix-k-nongoal',differences(t('C0'),t('C3'))==['plan'] and eq(t('C3')['plan'],t('C0')['plan'][:k]) and y('C3')['valid'] and not y('C3')['reaches_goal'])
        fact=f('C4')['v2_isolated_noop_witness']
        check('C4-exact-isolated-append',differences(t('C3'),t('C4'))==['plan'] and eq(t('C4')['plan'][:-1],t('C3')['plan']) and fact is not None and fact['action_index']==k and len(t('C4')['plan'])==k+1,fact)
        strict4={key:y('C4')[key] for key in ('steps','after','valid','reaches_goal')}
        mutants=f('C4')['diagnostic_mutant_outputs']
        check('C4-noop-mutant-changes-validity',not strict4['valid'] and mutants['no_op_allowed']['valid'] and mutants['no_op_allowed']['steps'][k]['legal'])
        check('C4-clearance-mutants-unchanged',all(eq(strict4,mutants[m]) for m in ('source_clear_ignored','destination_clear_ignored')))
        check('C-block-size-even-five-odd-six',len(t('C0')['on'])==(5 if index%2==0 else 6))
    else:
        check('D0-proper-partial-minimum-at-least-two',0<len(t('D0')['goal_on'])<len(t('D0')['on']) and eq(t('D0')['plan'],y('D0')['shortest_plan']) and len(t('D0')['plan'])>=2)
        check('D1-only-valid-goal-extension',differences(t('D0'),t('D1'))==['goal_on'] and len(t('D1')['goal_on'])>len(t('D0')['goal_on']) and all(t('D1')['goal_on'].get(k)==value for k,value in t('D0')['goal_on'].items()))
        check('D1-same-valid-plan-fails-goal-fresh-longer',y('D1')['valid'] and not y('D1')['reaches_goal'] and len(y('D1')['shortest_plan'])>len(y('D0')['shortest_plan']))
        check('D2-canonical-goal-extension-plan',differences(t('D1'),t('D2'))==['plan'] and eq(t('D2')['plan'],y('D2')['shortest_plan']))
        check('D3-same-strengthened-world-valid-detour',differences(t('D2'),t('D3'))==['plan'] and 'valid_goal_detour' in f('D3')['row_properties'])
        check('D-block-size-even-five-odd-six',len(t('D0')['on'])==(5 if index%2==0 else 6))
    return contexts,checks

def baselines(units):
    train=[i for i in units if i['rows'][0]['split']=='train'];dev=[i for i in units if i['rows'][0]['split']=='dev_challenge']
    def feature(i,fields):
        t=i['task'];size=len(t['nodes']) if t['kind']=='graph' else len(t['on'])
        allf={'kind':t['kind'],'size':size,'plan_length':len(t['plan']),'fuel':t.get('fuel'),'edge_count':len(t.get('edges',[])),'character_length':len(i['rows'][0]['rendered_input'])}
        return tuple(allf[k] for k in fields)
    result=[]
    for label in ('valid','reaches_goal','fresh_feasible','first_error_index'):
        def label_value(i):
            if label=='fresh_feasible':return i['facts']['target']['shortest_plan'] is not None
            if label=='first_error_index':
                e=i['facts']['measurements']['first_error'];return e['index'] if e else None
            return i['facts']['target'][label]
        global_counts=collections.Counter(v1.canon(label_value(i)) for i in train)
        majority=sorted(global_counts,key=lambda y:(-global_counts[y],y))[0]
        for fields in [('kind',),('size',),('plan_length',),('fuel',),('edge_count',),('character_length',),('kind','size','plan_length'),('kind','size','plan_length','fuel','edge_count')]:
            groups=collections.defaultdict(collections.Counter)
            for i in train:groups[feature(i,fields)][v1.canon(label_value(i))]+=1
            right=0;known=0
            for i in dev:
                values=groups.get(feature(i,fields));prediction=sorted(values,key=lambda y:(-values[y],y))[0] if values else majority
                right+=prediction==v1.canon(label_value(i));known+=bool(values)
            result.append({'label':label,'features':fields,'train_semantic_inputs':len(train),'dev_semantic_inputs':len(dev),'dev_correct':right,'dev_total':len(dev),'dev_feature_values_supported_in_train':known,'majority_dev_correct':sum(v1.canon(label_value(i))==majority for i in dev),'scope':'Lookup baseline from own inferred truth on exposed semantic inputs; not trained-model or full-target performance.'})
    return result

def run():
    started=time.monotonic();source_binding=bind(str(pathlib.Path(__file__).relative_to(ROOT)))
    before=bind(VIEW);assert before['sha256']==VIEW_SHA
    public_prep=prep.preparation() # Actual hand calibration rerun, no new source/gold reads.
    raw=(ROOT/VIEW).read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
    expected_fields={'sample_id','record_hash','source_family_key','split','semantic_task','rendered_input'}
    assert len(rows)==408 and all(set(r)==expected_fields for r in rows)
    assert len({r['sample_id'] for r in rows})==408 and len({r['record_hash'] for r in rows})==408
    cache={};by_family=collections.defaultdict(dict);predictions=[];row_failures={};max_search=collections.Counter();readback_checks=0;alternative_checks=0
    for r in rows:
        sample=r['sample_id'];task=r['semantic_task'];key=signature(task)
        try:
            # Every rendered input gets typed exact read-back and lead-label checking.
            lead,payload=r['rendered_input'].split('Task JSON:\n',1)
            assert r['rendered_input'].count('Task JSON:\n')==1 and eq(json.loads(payload),task)
            for label in prep.PRIVILEGED_LABELS:
                assert not re.search(r'(?<!\w)[\"\']?'+re.escape(label)+r'[\"\']?\s*[:=]',lead)
            v1.validate(task);readback_checks+=1
            if key not in cache:
                cache[key]=prep.derive_visible(r)
                if task['kind']=='graph':
                    assert eq(v1.graph_minimum_alternative(task),cache[key]['target']['shortest_plan']);alternative_checks+=1
            facts=cache[key]
            replay,error,states=v1.replay(task)
            assert eq(replay,{k:facts['target'][k] for k in replay}) and eq(states,facts['measurements']['prefix_states']) and eq(error,facts['measurements']['first_error'])
            for name in ('search','topology_search','preserve_search'):
                counts=facts['measurements'].get(name)
                if counts:
                    assert counts['complete']
                    for k in ('visited','expanded','maximum_frontier'):max_search[name+'_'+k]=max(max_search[name+'_'+k],counts[k])
            predicted={'sample_id':sample,'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],'reviewer_execution_id':ROLE,'stage':'INPUT_DERIVED_EXPECTATION_BEFORE_GOLD','semantic_task_normalized_sha256':sha(key.encode()),'rendered_input_sha256':sha(r['rendered_input'].encode()),'expected_target':facts['target'],'independent_input_facts':facts,'target_source':'Own public-contract implementation; no canonical target read.'}
        except Exception as error:
            row_failures[sample]={'category':'input_or_independent_solver_failure','severity':'high','evidence':type(error).__name__+': '+str(error),'fix':'Resolve the exact input or own solver failure before comparison; do not infer unreachable from incomplete execution.'}
            predicted={'sample_id':sample,'record_hash':r['record_hash'],'source_family_key':r['source_family_key'],'split':r['split'],'reviewer_execution_id':ROLE,'stage':'INPUT_DERIVED_EXPECTATION_UNRESOLVED','expected_target':None,'actual_error':row_failures[sample]}
            facts=None
        predictions.append(predicted)
        if facts is not None:
            item=by_family[r['source_family_key']].setdefault(key,{'task':normalized(task),'facts':facts,'rows':[]});item['rows'].append(r)
    families=[];row_context={};family_issues={};contexts_all=[];units=[];global_checks=[]
    for family,items in sorted(by_family.items()):
        index=family_index(family);values=list(items.values());units+=values
        try:
            contexts,checks=audit_family(values,index)
            family_issues[family]=[{'category':'public_input_context_obligation','severity':'high','evidence':c,'fix':'Fresh versioned correction of the concrete required family context.'} for c in checks if not c['pass']]
            contexts_all.append((family,index,contexts))
            for context,item in contexts.items():
                for row in item['rows']:row_context[row['sample_id']]=context
            families.append({'source_family_key':family,'source_index':index,'inferred_stratum':stratum(index),'contexts':{c:{'sample_ids':[r['sample_id'] for r in i['rows']],'normalized_task_sha256':sha(signature(i['task']).encode()),'submitted_length':len(i['task']['plan']),'minimum_length':len(i['facts']['target']['shortest_plan']) if i['facts']['target']['shortest_plan'] is not None else None,'first_error':i['facts']['measurements']['first_error'],'target_valid':i['facts']['target']['valid'],'reaches_goal':i['facts']['target']['reaches_goal']} for c,i in contexts.items()},'checks':checks})
        except Exception as error:
            family_issues[family]=[{'category':'input_context_inference_failure','severity':'high','evidence':type(error).__name__+': '+str(error),'fix':'Inspect all visible tasks and public context requirements before control comparison.'}]
            families.append({'source_family_key':family,'source_index':index,'actual_error':family_issues[family]})
    def global_check(name,truth,evidence):global_checks.append({'name':name,'pass':bool(truth),'evidence':evidence})
    global_check('exact48publiccohort',len(by_family)==48 and {family_index(k) for k in by_family}==set(range(180,228)),len(by_family))
    global_check('normalized204semanticinputs',len(cache)==204,len(cache))
    global_check('408typedexpectations',len(predictions)==408 and not row_failures,len(row_failures))
    c1_hist=collections.Counter();c4_hist=collections.Counter();reason_positions=collections.defaultdict(set);temporary=[];bties=[];hist=collections.Counter();graphweighted=[];graphunweighted=[];blockjoint=[];sixshapes=[]
    for family,index,contexts in contexts_all:
        kind=stratum(index);base=contexts[kind+'0']['task'];size=len(base['nodes']) if kind in 'AB' else len(base['on'])
        if kind in 'AB':graphweighted.append(graph_shape(base,True));graphunweighted.append(graph_shape(base,False))
        else:
            blockjoint.append(block_shape(base,True))
            if size==6:sixshapes.append(block_shape(base,False))
        if kind=='B' and contexts['B0']['facts']['measurements']['minimum_feasible_route_count']>=2:bties.append(family)
        if kind=='C':
            c1=contexts['C1'];e=c1['facts']['measurements']['first_error'];split=c1['rows'][0]['split'];reason_positions[e['reason']].add(e['index']);c1_hist[(e['reason'],size,split)]+=1
            c4=contexts['C4'];c4_hist[(c4['facts']['measurements']['first_error']['index'],c4['rows'][0]['split'])]+=1
            if 'required_temporary_goal_displacement' in contexts['C0']['facts']['row_properties']:temporary.append(family)
        for context,item in contexts.items():
            e=item['facts']['measurements']['first_error'];target=item['facts']['target'];split=item['rows'][0]['split']
            hist[(kind,context,split,size,len(item['task']['plan']),target['valid'],target['reaches_goal'],target['shortest_plan'] is not None,e['reason'] if e else None,e['index'] if e else None)]+=1
    global_check('sixBminimumroute-tie-families',len(bties)>=6,bties)
    global_check('fourCrequiredtemporarygoal-families',len(temporary)>=4,temporary)
    global_check('C1firsterror-atleast-three-total-two-per-reason-nonempty-each',len(set.union(*reason_positions.values()))>=3 and all(len(s)>=2 and any(i>=1 for i in s) for s in reason_positions.values()),{k:sorted(v) for k,v in reason_positions.items()})
    for reason in ('source_not_clear','destination_not_clear'):
        global_check('C1-reason-size-split:'+reason,sum(v for (r,_,_),v in c1_hist.items() if r==reason)==6 and all(sum(v for (r,n,_),v in c1_hist.items() if r==reason and n==size)==3 for size in (5,6)) and sum(v for (r,_,split),v in c1_hist.items() if r==reason and split=='dev_challenge')==1,{repr(k):v for k,v in c1_hist.items() if k[0]==reason})
    global_check('24distinctweightedgraphbases',len(set(graphweighted))==24,len(set(graphweighted)))
    global_check('atleast16unweightedgraphclasses',len(set(graphunweighted))>=16,len(set(graphunweighted)))
    global_check('24distinctjointblockbases',len(set(blockjoint))==24,len(set(blockjoint)))
    global_check('atleast8six-blockinitialshapes',len(set(sixshapes))>=8,len(set(sixshapes)))
    failed_global=[c for c in global_checks if not c['pass']]
    baselines_result=baselines(units)
    first=[]
    for row in rows:
        issues=[]
        if row['sample_id'] in row_failures:issues.append(row_failures[row['sample_id']])
        issues+=family_issues.get(row['source_family_key'],[])
        issues += [{'category':'cohort_input_obligation','severity':'high','evidence':c,'fix':'Resolve the actual cohort-level input requirement before any candidate approval.'} for c in failed_global]
        context=row_context.get(row['sample_id']);is_B=context is not None and context.startswith('B')
        issues.append({'category':'coverage_and_shortcut_scope','severity':'moderate','evidence':'48 revised source families and two existing grammars; renderers/controls are correlated views. Exposed-dev input lookup and full conditional tables are measured separately; no structural OOD, model independence, new-family growth relative V1 or learning benefit established.','fix':'Retain honest scope and conditional tables; assess residual cues against actual independently inferred labels.'})
        first.append({'receipt_id':'expert-planning-gap-v2-input-'+row['sample_id'],'sample_id':row['sample_id'],'record_hash':row['record_hash'],'reviewer_role':'expert','reviewer_execution_id':ROLE,'review_status':'INITIAL_INPUT_STAGE_FROZEN_BEFORE_GOLD','decision':'REVISE_INPUT_PENDING_COMPARISON' if any(i['severity']=='high' for i in issues) else 'INPUT_CLEAR_PENDING_TARGET_METADATA_COMPARISON','scores':{'semantic_correctness':5 if row['sample_id'] not in row_failures else 1,'target_correctness':None,'instruction_clarity':5,'reasoning_validity':5 if row['sample_id'] not in row_failures else 1,'generalization_value':3,'difficulty_quality':3,'diversity':3,'shortcut_resistance':2 if is_B else 3},'scores_scope':'Input-only provisional assessment. Stored target correctness, producer tags, difficulty metadata and control declarations are intentionally unassessed. No dataset ACCEPT.','critical_defects':[i for i in issues if i['severity']=='high'],'issues':issues,'confidence':0.96 if not any(i['severity']=='high' for i in issues) else 0.9,'evidence':{'source_family_key':row['source_family_key'],'split':row['split'],'inferred_context':context,'input_view_sha256':VIEW_SHA,'own_prediction_index':next(i for i,p in enumerate(predictions) if p['sample_id']==row['sample_id']),'independent_method':'Full finite prompt-derived planning search per normalized task; every row typed read-back/replay; renderer equivalence and public input context relations audited. Own V1 solver/public V2 guided correction; shared underlying model disclosed.'}})
    assert bind(VIEW)==before
    paths=[]
    def write_jsonl(name,values):
        p=OUT/name
        with p.open('x') as f:
            for value in values:f.write(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n')
        paths.append(bind(str(p.relative_to(ROOT))))
    write_jsonl('planning-gap-v2.expert_input_predictions.jsonl',predictions)
    write_jsonl('planning-gap-v2.expert_first_input_judgments.jsonl',first)
    analysis={'actual_author':ROLE,'stage':'INPUT_ONLY_COMPLETE_BEFORE_GOLD','source':source_binding,'input_binding_before':before,'input_binding_after':bind(VIEW),'actual_rows':len(rows),'raw_typed_task_variants':len({v1.canon(r['semantic_task']) for r in rows}),'normalized_semantic_inputs':len(cache),'source_families':len(by_family),'distinct_prompt_leads':len({r['rendered_input'].split('Task JSON:\n')[0] for r in rows}),'all_rows_typed_readback':readback_checks,'own_graph_heap_alternative_minimum_checks':alternative_checks,'every_row_replayed':len(rows)-len(row_failures),'own_prep_calibration_rerun_checks':public_prep['actual_check_entries'],'row_failures':row_failures,'family_audits':families,'global_checks':global_checks,'all_input_requirements_pass':not row_failures and not failed_global and not any(family_issues.values()),'C1_reason_size_split':{repr(k):v for k,v in c1_hist.items()},'C4_error_position_split':{repr(k):v for k,v in c4_hist.items()},'semantic_conditional_histogram':{repr(k):v for k,v in hist.items()},'own_input_only_lookup_baselines':baselines_result,'search_completion_and_maxima':dict(max_search),'within_batch_classes':{'weighted_start_goal_graph':len(set(graphweighted)),'unweighted_graph':len(set(graphunweighted)),'joint_initial_goal_blocks':len(set(blockjoint)),'six_block_initial_shapes':len(set(sixshapes))},'prior_corpus_novelty_comparison':'Not performed in this stage; prior active data/producer novelty manifests not read. No prior novelty acceptance claim.','actual_elapsed_seconds':time.monotonic()-started,'scope':'408 input expectations/first judgments only, not canonical target/metadata approval. Public-guided corrective same-family cohort, own frozen V1 solver reuse, same underlying model; no family growth or model independence.'}
    p=OUT/'planning-gap-v2.expert_input_analysis.json'
    with p.open('x') as f:json.dump(analysis,f,indent=2);f.write('\n')
    paths.append(bind(str(p.relative_to(ROOT))))
    execution={'actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'INPUT_ONLY_PREDICTIONS_FIRST_JUDGMENTS_FROZEN_BEFORE_GOLD','actual_command':[sys.executable,'-B',str(pathlib.Path(__file__).relative_to(ROOT))],'actual_exit_code':0,'source_before_after_unchanged':bind(source_binding['path'])==source_binding,'bound_source':source_binding,'bound_preparation_source':bind(PREP),'bound_own_V1_solver':bind('reviews/expert/prepare_rgm_planning_gap_v1.py'),'input_view_before':before,'input_view_after':bind(VIEW),'input_view_custody_freeze':bind('reviews/input_only/planning-gap-v2.freeze.json'),'artifacts':paths,'actual_rows':len(rows),'predictions':len(predictions),'typed_expectation_rows':len(predictions)-len(row_failures),'first_input_judgments':len(first),'all_input_requirements_pass':analysis['all_input_requirements_pass'],'actual_failure':None,'gold_control_private_code_other_reviews_read':False,'elapsed_seconds':analysis['actual_elapsed_seconds'],'source_and_prediction_order':'Source was frozen before actual execution; predictions and first input judgments are now frozen, before any gold or private control comparison.'}
    p=OUT/'planning-gap-v2.expert_input_execution.json'
    with p.open('x') as f:json.dump(execution,f,indent=2);f.write('\n')
    print(json.dumps({'execution':bind(str(p.relative_to(ROOT))),'artifacts':paths,'rows':len(rows),'normalized_tasks':len(cache),'all_input_requirements_pass':analysis['all_input_requirements_pass'],'elapsed_seconds':analysis['actual_elapsed_seconds']}))

if __name__=='__main__':run()
