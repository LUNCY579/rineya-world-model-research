"""Actual input-only independent RSM predictions and adversarial cohort audit.
Gold/producer/formal/other reviewer paths never opened. Reuses own frozen solvers.
"""
import copy, datetime, hashlib, itertools, json, math, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PREP=ROOT/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'
CORRECTED=ROOT/'reviews/adversarial/RSM-option-resumption-public-interface-correction-20261010/attempt01'
PACKET=ROOT/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl'
sys.path[:0]=[str(PREP),str(CORRECTED)]

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def bind(path):
    p=Path(path);raw=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(name,value,jsonl=False):
    p=OUT/name
    with p.open('x') as f:
        if jsonl:
            for row in value:f.write(canonical(row)+'\n')
        else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    if p.stat().st_size>25000000:raise ValueError('output exceeds declared artifact cap')
    return bind(p)
def frozen_bindings(freeze):
    return [bind(ROOT/r['path'] if not Path(r['path']).is_absolute() else r['path']) for r in freeze['source_bindings']+freeze['runtime_bindings']]

def rename(value, mapping):
    if type(value) is dict:return {mapping.get(k,k):rename(v,mapping) for k,v in value.items()}
    if type(value) is list:return [rename(v,mapping) for v in value]
    return mapping.get(value,value) if type(value) is str else value
def lookups(rows,output,feature):
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']!='train']
    hist=Counter(output[r['sample_id']]['status'] for r in train)
    major=min(hist,key=lambda k:(-hist[k],k));table=defaultdict(Counter)
    for row in train:table[canonical(feature(row))][output[row['sample_id']]['status']]+=1
    correct=0;seen=0;errors=[];byclass=defaultdict(lambda:[0,0]);counts={}
    for row in dev:
        key=canonical(feature(row));h=table.get(key);pred=min(h,key=lambda k:(-h[k],k)) if h else major
        actual=output[row['sample_id']]['status'];seen+=bool(h);correct+=pred==actual
        byclass[actual][1]+=1;byclass[actual][0]+=pred==actual
        if pred!=actual:errors.append({'sample_id':row['sample_id'],'record_hash':row['record_hash'],'feature':feature(row),'prediction':pred,'own_expected_status':actual})
    examples=defaultdict(list)
    for row in rows:examples[canonical(feature(row))].append(row)
    collision=None
    for key,group in sorted(examples.items()):
        for a,b in itertools.combinations(group,2):
            if output[a['sample_id']]['status']!=output[b['sample_id']]['status']:
                collision={'sample_ids':[a['sample_id'],b['sample_id']],'feature':feature(a),'own_statuses':[output[a['sample_id']]['status'],output[b['sample_id']]['status']]};break
        if collision:break
    return {'train_rows':len(train),'dev_rows':len(dev),'dev_correct':correct,'dev_accuracy':correct/len(dev),
            'dev_seen_feature_rows':seen,'train_majority':major,'train_status_histogram':dict(hist),
            'dev_macro_accuracy':sum(a/b for a,b in byclass.values())/len(byclass),
            'training_method':'train-only frequency lookup, unseen→train majority; deterministic lexical tie',
            'first_errors':errors[:4],'minimal_feature_collision':collision,'distinct_train_keys':len(table)}

def geometry(task,frame):
    ins_count=0;max_cond=0;calls=defaultdict(set);read=Counter();ops=Counter()
    def expr(node):
        if 'reg' in node:read[node['reg']]+=1
    def guard(pred):
        for a in pred['all']:expr(a['left']);expr(a['right'])
    def walk(body,owner,depth):
        nonlocal ins_count,max_cond
        for ins in body:
            ins_count+=1;ops[ins['op']]+=1
            if ins['op']=='if':
                max_cond=max(max_cond,depth+1);guard(ins['test']);walk(ins['then'],owner,depth+1);walk(ins['else'],owner,depth+1)
            elif ins['op']=='call':
                calls[owner].add(ins['option'])
                for node in ins['args'].values():expr(node)
            elif ins['op'] in {'add','scale'}:expr(ins['value'])
    for owner,opt in task['options'].items():guard(opt['initiation']);guard(opt['success']);walk(opt['body'],owner,0)
    return {'options':len(task['options']),'parameters':[len(o['params']) for o in task['options'].values()],
            'declared_instructions':ins_count,'conditional_depth':max_cond,'ops':dict(ops),'declared_register_reads':dict(read)}

def saved_binding_evidence(task,out,frame,controls):
    """Track live-register provenance at actual invocation sites, then future uses.
    This is actual input-derived dependency evidence, not a supplied tag.
    """
    origins={};bindings={};pre=copy.deepcopy(task['initial_state'])
    for event in task['history']:
        if event['event']=='enter':
            if event['call_site'] is None:args=task['entry']['args'];caller=None
            else:
                site=event['call_site'];caller=site['caller_instance'];owner=next(e['option'] for e in task['history'] if e['event']=='enter' and e['instance_id']==caller)
                ins=controls.instruction_at(task,{'option':owner,'body_path':site['body_path'],'index':site['index']});args=ins['args']
            local={}
            for p,expr in args.items():
                if 'reg' in expr:local[p]=expr['reg']
                elif 'param' in expr:local[p]=origins.get(caller,{}).get(expr['param'])
                else:local[p]=None
            origins[event['instance_id']]=local;bindings[event['instance_id']]=event['bindings']
        elif event['event']=='effect':pre=event['state']
    saved=[];checkpoint=task['checkpoint']
    for event in out['new_events']:
        if event['event'] not in {'effect','branch','return'}:continue
        iid=event['instance_id']
        if iid not in origins:continue
        if event['event']=='return':nodes=task['options'][event['option']]['success']['all']
        else:
            ins=controls.instruction_at(task,event)
            nodes=[ins['value']] if event['event']=='effect' and ins['op']!='swap' else ins['test']['all'] if event['event']=='branch' else []
        def params(node):
            if type(node) is dict:
                if 'param' in node:yield node['param']
                else:
                    for value in node.values():yield from params(value)
            elif type(node) is list:
                for value in node:yield from params(value)
        for p in params(nodes):
            register=origins[iid].get(p)
            if register and checkpoint['state'][register]!=bindings[iid][p]:
                saved.append({'instance_id':iid,'option':event['option'],'parameter':p,'original_register':register,
                              'saved_value':bindings[iid][p],'checkpoint_live_value':checkpoint['state'][register],
                              'actual_future_use':event})
    return saved

def main():
    started=time.monotonic();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    freeze=json.loads((OUT/'PREEXECUTION_FREEZE.json').read_text());expected=[{k:r[k] for k in ('path','size_bytes','sha256')} for r in freeze['source_bindings']+freeze['runtime_bindings']]
    before=frozen_bindings(freeze)
    if before!=expected:raise RuntimeError('preexec bytes differ')
    # First imports of own solvers occur only after complete raw source/input freeze.
    import frame_solver_r1 as frame
    import recursive_simulator_r1 as recursive
    import control_probes_r2 as controls
    raw=PACKET.read_bytes();rows=[frame.strict_json(x.decode()) for x in raw.splitlines()]
    assert hashlib.sha256(raw).hexdigest()=='d99074c6cd18d1bd9709a09f833bfe9fea50c0c3981ece532c3951aac220f6b1'
    config=json.loads((OUT/'RUN_CONFIG.json').read_text());assert len(rows)==config['expected_rows']
    checks=[];failures=[]
    def check(name, condition, evidence=None):
        okay=bool(condition);item={'check':name,'passed':okay}
        if evidence is not None:item['evidence']=evidence
        checks.append(item)
        if not okay:failures.append(item)
        return okay
    def dual(task,name):
        a,b=frame.solve(task),recursive.solve(task)
        check(name,frame.typed_equal(a,b),None if frame.typed_equal(a,b) else {'task':task,'frame':a,'recursive':b})
        return a
    hand=json.loads((PREP/'hand_fixtures_r1.json').read_text())
    for context,task in hand['contexts'].items():dual(task,'own_hand_dual_'+context)
    check('own_hand_F_exact_trace',frame.typed_equal(frame.solve(hand['contexts']['F'])['new_events'],hand['hand_F_trace']))
    check('own_hand_R_exact_checkpoint',frame.typed_equal(hand['contexts']['R']['checkpoint'],hand['hand_R_checkpoint']))
    preds=[];other=[];outputs={};initial=[];prompt_info=[];groups=defaultdict(list)
    doc=(ROOT/'schemas/RSM_OPTION_RESUMPTION_V1.md').read_text();ids=set()
    allowed={'sample_id','semantic_parent_id','semantic_family_id','track_id','split','semantic_task','rendered_input','record_hash'}
    for row in rows:
        sid=row['sample_id'];task=row['semantic_task'];groups[row['semantic_family_id']].append(row)
        issues=[]
        check('packet_fields_'+sid,set(row)==allowed)
        check('unique_sample_id_'+sid,sid not in ids);ids.add(sid)
        check('record_hash_join_type_'+sid,type(row['record_hash']) is str and re.fullmatch('[0-9a-f]{64}',row['record_hash']) is not None)
        a,b=frame.solve(task),recursive.solve(task);outputs[sid]=a
        same=check('cohort_dual_'+sid,frame.typed_equal(a,b),None if frame.typed_equal(a,b) else {'task':task,'frame':a,'recursive':b})
        try:
            lead,rendered=row['rendered_input'].rsplit('Task JSON:\n',1)
            pt=frame.strict_json(rendered);prompt_equal=frame.typed_equal(pt,task)
            exact_lead=lead.rstrip()==doc.rstrip()
        except Exception as error:
            lead='';prompt_equal=False;exact_lead=False
            issues.append({'severity':'HIGH','code':'prompt_not_parseable','evidence':str(error)})
        check('typed_prompt_equals_task_'+sid,prompt_equal)
        check('full_public_contract_lead_'+sid,exact_lead)
        if not same:issues.append({'severity':'HIGH','code':'own_solvers_disagree','evidence':'full two outputs preserved in predictions and failure artifact'})
        if not prompt_equal:issues.append({'severity':'HIGH','code':'typed_prompt_task_mismatch','evidence':'rendered typed task differs from semantic_task'})
        if not exact_lead:issues.append({'severity':'HIGH','code':'contract_lead_mismatch','evidence':'lead differs from frozen public contract'})
        if a['error']=='invalid_task':issues.append({'severity':'HIGH','code':'unexpected_program_malformed','evidence':'input-derived program validation failed'})
        prompt_info.append({'sample_id':sid,'prompt_chars':len(row['rendered_input']),'prompt_bytes':len(row['rendered_input'].encode()),'task_bytes':len(canonical(task).encode()),'lead_sha256':hashlib.sha256(lead.encode()).hexdigest()})
        joint={'sample_id':sid,'record_hash':row['record_hash'],'semantic_parent_id':row['semantic_parent_id'],
               'semantic_family_id':row['semantic_family_id'],'split':row['split'],'typed_task_sha256':digest(task)}
        preds.append({**joint,'expected_output':a,'prediction_method':'own explicit frame machine','actor':'/root/adversarial_auditor'})
        other.append({**joint,'expected_output':b,'prediction_method':'own independent recursive trace','actor':'/root/adversarial_auditor'})
        initial.append({**joint,'receipt_id':'red-rsm-input-first-r1-'+digest([sid,row['record_hash']])[:24],
            'reviewer_role':'adversarial_input_only','reviewer_execution_id':'/root/adversarial_auditor',
            'scores':{'semantic_correctness':4 if not issues else 1,'target_correctness':'UNKNOWN',
                      'instruction_clarity':4 if prompt_equal and exact_lead else 1,'reasoning_validity':4 if same else 1,
                      'generalization_value':2,'difficulty_quality':'UNKNOWN_WITHOUT_METADATA','diversity':2,'shortcut_resistance':'PENDING_COHORT_AUDIT'},
            'critical_defects':[i['code'] for i in issues if i['severity']=='HIGH'],'issues':issues,
            'decision':'INITIAL_INPUT_ONLY' if not issues else 'REVISE_INPUT','confidence':0.92 if same else 0.5,
            'gold_access':False,'canonical_metadata_access':False,'data_approval':False,
            'evidence':{'dual_typed_agreement':same,'full_contract_and_typed_prompt':prompt_equal and exact_lead,
                        'own_status':a['status'],'own_error':a['error'],'semantic_output_hash':digest(a)},
            'scope_limits':['No stored target/full record/control metadata read; record_hash is opaque join key',
                            'same-model and shared own grammar; previous handmade/public guidance exposure',
                            'finite supplied procedures, tiny exposed families; no efficacy or sealed OOD claim']})
    prediction_file=write('frame_predictions.jsonl',preds,True)
    recursive_file=write('recursive_predictions.jsonl',other,True)
    initial_file=write('INITIAL_INPUT_JUDGMENTS.jsonl',initial,True)
    early=write('PREDICTION_FREEZE.json',{'actor':'/root/adversarial_auditor','gold_access':False,'rows':len(rows),
        'predictions':[prediction_file,recursive_file],'initial_judgments':initial_file,'input':bind(PACKET),
        'preexecution':bind(OUT/'PREEXECUTION_FREEZE.json'),'scope':'first full predictions and input judgments frozen before cohort guided statistics; no gold approval'})

    metamorphic=[];family_evidence=[];contexts_status=defaultdict(Counter);family_signatures={};geometries=[]
    cut_count=0;corruption_count=0
    for row in rows:
        sid=row['sample_id'];task=row['semantic_task'];actual=outputs[sid]
        reversed_task=controls.reverse_objects(task)
        rev=dual(reversed_task,'object_order_dual_'+sid)
        check('object_order_output_'+sid,frame.typed_equal(rev,actual))
        option_names=list(task['options']);param_names=sorted({p for o in task['options'].values() for p in o['params']})
        mapping={name:'opaque_option_'+str(i) for i,name in enumerate(sorted(option_names))}
        mapping.update({name:'opaque_saved_'+str(i) for i,name in enumerate(param_names)})
        renamed=rename(task,mapping);renamed_output=dual(renamed,'alpha_rename_dual_'+sid)
        check('alpha_rename_covariance_'+sid,frame.typed_equal(renamed_output,rename(actual,mapping)))
        swapped=rename(task,{'x':'y','y':'x'});swapped_output=dual(swapped,'register_swap_dual_'+sid)
        check('register_swap_covariance_'+sid,frame.typed_equal(swapped_output,rename(actual,{'x':'y','y':'x'})))
        malformed=copy.deepcopy(task);malformed['event_budget']=True
        out=dual(malformed,'typed_bool_budget_dual_'+sid);check('typed_bool_budget_stage_'+sid,out==frame.invalid('invalid_task'));corruption_count+=1
        malformed['history']=[{'event':'wrong'}];malformed['checkpoint']=False
        out=dual(malformed,'stage_order_program_first_'+sid);check('stage_order_program_expected_'+sid,out==frame.invalid('invalid_task'));corruption_count+=1
        broken=copy.deepcopy(task);broken['history']=[{'event':'wrong'}];broken['checkpoint']=False
        out=dual(broken,'stage_order_history_dual_'+sid);check('stage_order_history_expected_'+sid,out==frame.invalid('invalid_history'));corruption_count+=1
        if actual['status']!='invalid':
            broken=copy.deepcopy(task)
            if task['mode']=='fresh':broken['checkpoint']={}
            else:broken['checkpoint']['root_pending']=int(broken['checkpoint']['root_pending'])
            out=dual(broken,'typed_checkpoint_dual_'+sid);check('typed_checkpoint_expected_'+sid,out==frame.invalid('invalid_checkpoint'));corruption_count+=1
        metamorphic.append({'sample_id':sid,'object_order':True,'option_parameter_rename':True,'register_rename':True,
                            'scope':'actual task descendants, not additional independent families; results bound by named checks'})

    config_public=json.loads((ROOT/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json').read_text())
    index_stratum={i:k for k,values in config_public['source_family_indices_by_stratum'].items() for i in values}
    for family,group in sorted(groups.items()):
        coordinate={};indices=set();splits=set()
        for row in group:
            m=re.fullmatch(r'RSM-001-option-resumption-pilot-v1-p(\d+)-([FRTPUAISX])-(forward|reverse)',row['sample_id'])
            check('public_id_diagnostic_shape_'+row['sample_id'],m is not None)
            if m is None:continue
            index=int(m[1]);context=m[2];style=m[3];indices.add(index);splits.add(row['split'])
            coordinate[(context,style)]=row
            contexts_status[context][outputs[row['sample_id']]['status']]+=1
        check('family_18_views_'+family,len(group)==18 and len(coordinate)==18)
        check('family_one_split_'+family,len(splits)==1 and len(indices)==1)
        if len(coordinate)!=18:continue
        index=next(iter(indices));expected_split='train' if index%10<=7 else 'dev_challenge'
        check('family_public_split_policy_'+family,splits=={expected_split})
        ftask=coordinate[('F','forward')]['semantic_task'];family_signatures[family]=controls.program_signatures(ftask)
        geometries.append({'family':family,'index':index,'stratum_public_id_assertion':index_stratum.get(index),'geometry':geometry(ftask,frame)})
        for style in ('forward','reverse'):
            tasks={c:coordinate[(c,style)]['semantic_task'] for c in 'FRTPUAISX'}
            results={c:outputs[coordinate[(c,style)]['sample_id']] for c in 'FRTPUAISX'}
            try:
                control=controls.check_controls(tasks,results)
                check('eight_actual_semantic_relations_'+family+'_'+style,True)
            except Exception as error:
                control={'error':type(error).__name__+': '+str(error)}
                check('eight_actual_semantic_relations_'+family+'_'+style,False,control)
            saved=saved_binding_evidence(tasks['R'],results['R'],frame,controls)
            check('actual_saved_live_binding_future_use_'+family+'_'+style,bool(saved),saved)
            # Demonstrate actual mistakes if a retained frame is spuriously rebound
            # or a selected branch guard is spuriously rerun. Trusted-machine probes
            # do not replace canonical checkpoint rejection or corpus witnesses.
            attack_evidence=[]
            for attack in ('rebind_saved_argument','erase_retained_branch','rewind_caller_cursor'):
                machine=copy.deepcopy(tasks['R']['checkpoint'])
                if attack=='rebind_saved_argument' and saved:
                    witness=saved[0]
                    for fr in machine['frames']:
                        if fr['instance_id']==witness['instance_id']:fr['bindings'][witness['parameter']]=machine['state'][witness['original_register']]
                elif attack=='erase_retained_branch':machine['frames']=[fr for fr in machine['frames'] if fr['frame_kind']!='branch']
                elif attack=='rewind_caller_cursor':
                    machine['frames'][0]['next_index']=max(0,machine['frames'][0]['next_index']-1)
                changed=[];error=None
                try:
                    while machine['phase']=='running' and len(changed)<64:changed.append(frame.event_step(tasks['R'],machine))
                except Exception as e:error=type(e).__name__+': '+str(e)
                changed_result=not frame.typed_equal(changed,results['R']['new_events'])
                check('actual_trusted_mutant_'+attack+'_'+family+'_'+style,changed_result)
                attack_evidence.append({'mutation':attack,'changes_suffix':changed_result,'diagnostic_error':error,
                    'actual_diagnostic_events':changed,'actual_diagnostic_final_snapshot':machine})
            family_evidence.append({'family':family,'index':index,'style_diagnostic_from_id':style,
                'actual_context_sample_ids':{c:coordinate[(c,style)]['sample_id'] for c in tasks},
                'independently_executed_control_evidence':control,'eight_semantic_relations_plus_separate_field_order':True,
                'actual_saved_binding_future_uses':saved,'trusted_machine_mutants':attack_evidence,
                'minimal_pairs':[{'contexts':[a,b],'sample_ids':[coordinate[(a,style)]['sample_id'],coordinate[(b,style)]['sample_id']],
                                  'typed_input_differences':[list(p) for p in controls.differences(tasks[a],tasks[b])],
                                  'own_statuses':[results[a]['status'],results[b]['status']]} for a,b in controls.PAIRS.values()]})
        for context in 'FRTPUAISX':
            a,b=coordinate[(context,'forward')],coordinate[(context,'reverse')]
            check('actual_field_pair_task_'+a['sample_id'],frame.typed_equal(a['semantic_task'],b['semantic_task']))
            check('actual_field_pair_result_'+a['sample_id'],frame.typed_equal(outputs[a['sample_id']],outputs[b['sample_id']]))
            check('actual_field_pair_presentation_changed_'+a['sample_id'],a['rendered_input']!=b['rendered_input'])
        events,snapshots=recursive.full_execution(ftask)
        check('F_full_trace_vs_prediction_'+family,frame.typed_equal(events,outputs[coordinate[('F','forward')]['sample_id']]['new_events']))
        geom=geometry(ftask,frame)
        check('public_bounds_syntax_'+family,2<=geom['options']<=4 and all(1<=p<=3 for p in geom['parameters']) and geom['declared_instructions']<=24 and geom['conditional_depth']<=2)
        check('public_bounds_actual_F_'+family,len(events)<=48 and sum(e['event']=='effect' for e in events)>=3 and max(sum(fr['frame_kind']=='option' for fr in snap['frames']) for snap in snapshots)<=3 and all(-4096<=value<=4096 for snap in snapshots for value in snap['state'].values()))
        for cut in range(len(events)+1):
            for mode in ('resume','restart'):
                for budget in (0,1,3,48):
                    task=copy.deepcopy(ftask);task.update(mode=mode,history=events[:cut],checkpoint=snapshots[cut],event_budget=budget)
                    result=dual(task,'all_F_cut_budget_'+family+'_'+str(cut)+'_'+mode+'_'+str(budget));cut_count+=1
                    if mode=='resume' and budget==48:
                        check('actual_cut_suffix_'+family+'_'+str(cut),frame.typed_equal(result['new_events'],events[cut:]) and frame.typed_equal(result['checkpoint'],snapshots[-1]))
        reordered=copy.deepcopy(ftask);reordered['options'][ftask['entry']['option']]['body'].reverse()
        altered=dual(reordered,'instruction_list_order_dual_'+family)
        check('actual_list_order_not_invariant_'+family,not frame.typed_equal(altered,outputs[coordinate[('F','forward')]['sample_id']]))

    # Input-only distributions and train-derived outcome baselines. Output features
    # below are own predictions, never stored gold. Metadata IDs are separated.
    feature_set={
      'student_mode':lambda r:r['semantic_task']['mode'],
      'student_history_length':lambda r:len(r['semantic_task']['history']),
      'student_mode_history_budget':lambda r:[r['semantic_task']['mode'],len(r['semantic_task']['history']),r['semantic_task']['event_budget']],
      'student_initial_xy':lambda r:list(r['semantic_task']['initial_state'].values()),
      'student_root_argument_forms_values':lambda r:sorted(canonical(v) for v in r['semantic_task']['entry']['args'].values()),
      'student_program_option_names':lambda r:sorted(r['semantic_task']['options']),
      'student_option_count':lambda r:len(r['semantic_task']['options']),
      'student_checkpoint_frame_depth':lambda r:len(r['semantic_task']['checkpoint']['frames']) if r['semantic_task']['checkpoint'] else 0,
      'student_mode_history_effects_budget':lambda r:[r['semantic_task']['mode'],sum(e.get('event')=='effect' for e in r['semantic_task']['history']),r['semantic_task']['event_budget']],
      'student_prompt_char_100_bins':lambda r:len(r['rendered_input'])//100,
      'student_prompt_length_exact':lambda r:len(r['rendered_input']),
      'student_task_length_exact':lambda r:len(canonical(r['semantic_task'])),
      'student_first_option_object_key':lambda r:next(iter(r['semantic_task']['options'])),
      'external_join_ID_context_not_student_prompt':lambda r:r['sample_id'].split('-')[-2],
      'external_join_ID_style_not_student_prompt':lambda r:r['sample_id'].split('-')[-1],
    }
    baselines={name:lookups(rows,outputs,feature) for name,feature in feature_set.items()}
    hist=Counter(o['status'] for o in outputs.values());train_hist=Counter(outputs[r['sample_id']]['status'] for r in rows if r['split']=='train')
    exact_tasks=defaultdict(list);exact_prompts=defaultdict(list);parents=defaultdict(list)
    for row in rows:
        exact_tasks[digest(row['semantic_task'])].append(row);exact_prompts[hashlib.sha256(row['rendered_input'].encode()).hexdigest()].append(row);parents[row['semantic_parent_id']].append(row)
    def duplication(groups):return [{'sample_ids':[r['sample_id'] for r in group],'splits':sorted({r['split'] for r in group})} for group in groups.values() if len(group)>1]
    sig_groups={}
    for view in ('full','coarse_all','coarse_reachable'):
        table=defaultdict(list)
        for fam,signatures in family_signatures.items():table[signatures[view]].append(fam)
        sig_groups[view]={'unique_graphs':len(table),'duplicate_groups':[v for v in table.values() if len(v)>1]}
        check('12_distinct_'+view,len(table)==12)
    output_lengths=[{'sample_id':r['sample_id'],'status':outputs[r['sample_id']]['status'],'prompt_chars':len(r['rendered_input']),
                    'response_chars':len(canonical(outputs[r['sample_id']])),'history_events':len(r['semantic_task']['history']),
                    'new_events':len(outputs[r['sample_id']]['new_events']),'new_effects':outputs[r['sample_id']]['new_effects']} for r in rows]
    capability_hist=defaultdict(Counter)
    for row in rows:
        t=row['semantic_task'];o=outputs[row['sample_id']];cp=t['checkpoint']
        capability_hist['split_status'][row['split']+'|'+o['status']]+=1
        capability_hist['mode_status'][t['mode']+'|'+o['status']]+=1
        capability_hist['history_events'][str(len(t['history']))]+=1
        capability_hist['new_events'][str(len(o['new_events']))]+=1
        capability_hist['new_effects'][str(o['new_effects'])]+=1
        if cp:capability_hist['input_checkpoint_frame_depth'][str(len(cp['frames']))]+=1
    limitations=[
      'Gold targets, actual canonical tags/difficulty/provenance/control claims and full record hashes not accessed; all remain UNKNOWN pending authorized comparison.',
      '216 views are108 typed tasks in12 input-family lineages, including matched interventions; transforms/cuts are audit descendants, no family growth.',
      'Only four broad strata and two registers, finite deterministic supplied procedures; all strata exposed on dev. No recursion/policy discovery/concurrency/rollback/general retry.',
      'Graph distinctness measured within new cohort using own public normalization; no authorized legacy target-free packet read, so prior-active novelty remains unverified.',
      'Same underlying model/common public-guided design, own handmade and grammar reuse. Two execution algorithms share own validation; not model diversity or unseen-blind novelty.',
      'Context/style/family labels parsed from join IDs only as untrusted assertions, separately checked through actual semantic relations; IDs are absent from student task/prompt.',
      'Status-only majority/lookup baseline cannot produce exact event/state outputs. No trained model or efficacy evidence. Input/output lengths include intended trace/error-format differences.',
    ]
    audits=write('INPUT_DIVERSITY_SHORTCUT_AUDIT.json',{'actor':'/root/adversarial_auditor','gold_access':False,
        'rows':len(rows),'families':len(groups),'typed_tasks':len(exact_tasks),'prompts':len(exact_prompts),'semantic_parents':len(parents),
        'split_views':dict(Counter(r['split'] for r in rows)),'status_histogram_own':dict(hist),'contexts_status_ID_assertion_checked_by_actual_inputs':{k:dict(v) for k,v in contexts_status.items()},
        'typed_task_duplicate_groups':duplication(exact_tasks),'exact_prompt_duplicate_groups':duplication(exact_prompts),
        'within_cohort_structural_views':sig_groups,'family_geometries':geometries,'capability_histograms':{k:dict(v) for k,v in capability_hist.items()},
        'status_baselines':baselines,'input_response_length_records':output_lengths,'prompt_binding':prompt_info,
        'shortcuts_scope':'student features separately from external join IDs; train-only exact frequency lookups, not trained models',
        'intended_length_scope':'unique output includes variable event traces and short invalid outputs. No target-length leakage assertion from that fact alone.',
        'prior_active_novelty':'UNKNOWN: no current authorized legacy packet read; no sealed-OOD/effectiveness claim',
        'limits':limitations})
    relation_file=write('ACTUAL_CONTROL_AND_ATTACK_EVIDENCE.json',{'actor':'/root/adversarial_auditor','gold_access':False,
        'actual_family_renderer_groups':len(family_evidence),'expected_control_scope':'eight semanticrelations/group, field order separately108pairs; no manifest read',
        'actual_evidence':family_evidence,'metamorphic_actual_row_inventory':metamorphic,
        'cut_budget_cases':cut_count,'malformed_descendants':corruption_count,'audit_descendants_are_new_families':False})
    failure_file=write('FIRST_FAILURES.json',{'actor':'/root/adversarial_auditor','failures':failures,'failed_count':len(failures),'preservation':'first actual execution; no silent repair'})
    after=frozen_bindings(freeze);same_bytes=check('all_preexec_bindings_before_after',before==after)
    execution=write('EXECUTION.json',{'actor':'/root/adversarial_auditor','started_at_utc':now,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'check_count':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
        'checks':checks,'input':bind(PACKET),'preexecution':bind(OUT/'PREEXECUTION_FREEZE.json'),
        'all_before_after_equal':same_bytes,'before_bindings':before,'after_bindings':after,
        'rows':len(rows),'cut_budget_cases':cut_count,'malformed_descendants':corruption_count,
        'calibration_checks':11,'original1568_and63_reused_not_rerun':True,
        'outputs':[prediction_file,recursive_file,initial_file,early,audits,relation_file,failure_file],
        'no_gold_or_other_verdict_access':True,'data_approval':False,'scope_limits':limitations})
    print(canonical({'execution':execution,'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
                     'rows':len(rows),'prediction_freeze':early,'audits':audits,'controls':relation_file}))

if __name__=='__main__':main()
