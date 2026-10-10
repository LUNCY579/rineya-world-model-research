"""Actual owned handmade calibration/attack execution; freeze checked before imports."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREEXEC=HERE/'PREEXEC_R1.json'
RESULT=HERE/'EXECUTION_R1.json'


def binding(path):
    raw=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    freeze=json.loads(PREEXEC.read_bytes())
    for row in freeze['bindings']:
        if binding(ROOT/row['path'])!=row:raise ValueError('Preexecution binding differs before any owned solver import')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    sys.path.insert(0,str(HERE))
    import frame_solver_r1 as frame
    import recursive_simulator_r1 as recursive
    import control_probes_r1 as controls
    fixtures=json.loads((HERE/'hand_fixtures_r1.json').read_bytes())
    tasks=fixtures['contexts'];results=[];outputs={};control_evidence=None
    def check(name, fn, expects_rejection=False):
        try:
            value=fn()
            passed=not expects_rejection;error=None
            digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        except Exception as exception:
            passed=expects_rejection;error={'type':type(exception).__name__,'message':str(exception)};digest=None
        results.append({'name':name,'passed':passed,'expected_rejection':expects_rejection,'error':error,'returned_value_sha256':digest})
    def equal(a,b):
        if not frame.typed_equal(a,b):raise AssertionError('Exact typed result mismatch')
        return a
    def hand_check(task, expected):
        a=frame.solve(task);b=recursive.solve(task);equal(a,b)
        for key in ('status','error','previous_effects','new_effects'):
            if key in expected:equal(a[key],expected[key])
        if 'state' in expected:equal(a['checkpoint']['state'],expected['state'])
        if 'event_count' in expected:equal(len(a['new_events']),expected['event_count'])
        if 'next_event' in expected:equal(a['next_event'],expected['next_event'])
        if 'retained_caller_next_index' in expected:equal(a['checkpoint']['frames'][0]['next_index'],expected['retained_caller_next_index'])
        if 'leaf_bindings' in expected:equal(a['new_events'][1]['bindings'],expected['leaf_bindings'])
        return a
    for context,task in tasks.items():
        check('hand_context_'+context,lambda context=context,task=task:hand_check(task,fixtures['hand_expected'][context]))
        outputs[context]=frame.solve(task)
    for fixture in fixtures['boundaries']:
        check('hand_boundary_'+fixture['name'],lambda fixture=fixture:hand_check(fixture['task'],fixture['expected']))
    check('hand_exact_F_trace',lambda:equal(outputs['F']['new_events'],fixtures['hand_F_trace']))
    check('hand_exact_R_checkpoint',lambda:equal(recursive.full_execution(tasks['F'])[1][5],fixtures['hand_R_checkpoint']))
    check('simultaneous_nine_controls',lambda:controls.check_controls(tasks,outputs))
    try:control_evidence=controls.check_controls(tasks,outputs)
    except Exception:pass
    programs=[('simultaneous_controls',tasks['F'])]+[(fixture['name'],fixture['task']) for fixture in fixtures['boundaries']]
    cuts=0;budget_cases=0;second_interruptions=0
    for name,task in programs:
        events,snapshots=recursive.full_execution(task)
        n=len(events)
        for budget in range(n+3):
            variant=copy.deepcopy(task);variant['event_budget']=budget
            check('fresh_budget_'+name+'_'+str(budget),lambda variant=variant:equal(frame.solve(variant),recursive.solve(variant)))
        for cut in range(n+1):
            cuts+=1
            for mode in ('resume','restart'):
                for budget in range(n+3):
                    budget_cases+=1
                    variant=copy.deepcopy(task);variant.update(mode=mode,history=copy.deepcopy(events[:cut]),checkpoint=copy.deepcopy(snapshots[cut]),event_budget=budget)
                    check(f'all_cut_budget_{name}_{cut}_{mode}_{budget}',lambda variant=variant:equal(frame.solve(variant),recursive.solve(variant)))
            if cut==n:continue
            for budget in range(1,n-cut):
                variant=copy.deepcopy(task);variant.update(mode='resume',history=copy.deepcopy(events[:cut]),checkpoint=copy.deepcopy(snapshots[cut]),event_budget=budget)
                paused=frame.solve(variant)
                if paused['status']!='paused':continue
                second_interruptions+=1
                resumed=copy.deepcopy(variant);resumed.update(history=variant['history']+paused['new_events'],checkpoint=paused['checkpoint'],event_budget=48)
                check(f'two_interruptions_{name}_{cut}_{budget}',lambda resumed=resumed,cut=cut,budget=budget,events=events,snapshots=snapshots:equal(
                    (frame.solve(resumed)['new_events'],frame.solve(resumed)['checkpoint']),
                    (events[cut+budget:],snapshots[-1])))
    for budget in range(12):
        variant=copy.deepcopy(tasks['F']);variant['event_budget']=budget
        expected_trace=fixtures['hand_F_trace'][:budget]
        check('event_budget_is_not_effect_count_'+str(budget),lambda variant=variant,expected_trace=expected_trace,budget=budget:equal(
            (frame.solve(variant)['new_events'],frame.solve(variant)['new_effects'],frame.solve(variant)['status']),
            (expected_trace,sum(e['event']=='effect' for e in expected_trace),'completed' if budget==11 else 'paused')))
    malformed=[]
    def mutation(name,stage,mutate,base='F'):
        task=copy.deepcopy(tasks[base]);mutate(task);malformed.append((name,stage,task))
    mutation('extra_envelope','invalid_task',lambda t:t.update(private_label='hidden'))
    mutation('missing_checkpoint_key','invalid_task',lambda t:t.pop('checkpoint'))
    mutation('bool_version','invalid_task',lambda t:t.update(version=True))
    mutation('float_budget','invalid_task',lambda t:t.update(event_budget=1.0))
    mutation('bool_budget','invalid_task',lambda t:t.update(event_budget=True))
    mutation('negative_budget','invalid_task',lambda t:t.update(event_budget=-1))
    mutation('bool_initial_integer','invalid_task',lambda t:t['initial_state'].update(x=True))
    mutation('unknown_initial_field','invalid_task',lambda t:t['initial_state'].update(z=0))
    mutation('duplicate_parameter','invalid_task',lambda t:t['options']['main']['params'].append('p'))
    mutation('bool_constant','invalid_task',lambda t:t['options']['main']['body'][0].update(value={'const':True}))
    mutation('float_constant','invalid_task',lambda t:t['options']['main']['body'][0].update(value={'const':1.0}))
    mutation('expression_extra_form','invalid_task',lambda t:t['options']['main']['body'][0].update(value={'const':1,'reg':'x'}))
    mutation('wrong_register','invalid_task',lambda t:t['options']['main']['body'][0].update(register='z'))
    mutation('missing_callee_arg','invalid_task',lambda t:t['options']['main']['body'][1]['args'].pop('a'))
    mutation('unknown_callee','invalid_task',lambda t:t['options']['main']['body'][1].update(option='absent'))
    mutation('root_parameter_ref','invalid_task',lambda t:t['entry']['args'].update(p={'param':'p'}))
    mutation('nested_undeclared_parameter','invalid_task',lambda t:t['options']['child']['body'][0]['else'][0].update(value={'param':'missing'}))
    mutation('bad_comparison','invalid_task',lambda t:t['options']['main']['initiation']['all'][0].update(cmp='le'))
    mutation('hidden_branch_cycle','invalid_task',lambda t:t['options']['child']['body'][0]['else'].append({'op':'call','option':'main','args':{'p':{'const':1},'q':{'const':2}}}))
    mutation('unused_option_cycle','invalid_task',lambda t:t['options'].update(unused={'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':[{'op':'call','option':'unused','args':{}}]}))
    mutation('history_nonlist','invalid_history',lambda t:t.update(history={}))
    mutation('history_boolean_as_ID','invalid_history',lambda t:t['history'][2].update(instance_id=True),'R')
    mutation('history_int_as_boolean','invalid_history',lambda t:t['history'][0].update(initiation_ok=1),'R')
    mutation('history_event_extra_field','invalid_history',lambda t:t['history'][0].update(extra=0),'R')
    mutation('history_events_reordered','invalid_history',lambda t:t['history'].reverse(),'R')
    mutation('history_after_terminal','invalid_history',lambda t:t.update(history=copy.deepcopy(fixtures['hand_F_trace'])+[copy.deepcopy(fixtures['hand_F_trace'][-1])]))
    mutation('fresh_valid_history_forbidden','invalid_checkpoint',lambda t:t.update(history=copy.deepcopy(fixtures['hand_F_trace'][:1])))
    mutation('fresh_nonnull_checkpoint','invalid_checkpoint',lambda t:t.update(checkpoint={}))
    mutation('checkpoint_missing','invalid_checkpoint',lambda t:t.update(checkpoint=None),'R')
    mutation('checkpoint_wrong_state','invalid_checkpoint',lambda t:t['checkpoint']['state'].update(x=8),'R')
    mutation('checkpoint_float_state','invalid_checkpoint',lambda t:t['checkpoint']['state'].update(x=7.0),'R')
    mutation('checkpoint_int_root_pending','invalid_checkpoint',lambda t:t['checkpoint'].update(root_pending=0),'R')
    mutation('checkpoint_bool_ID','invalid_checkpoint',lambda t:t['checkpoint']['frames'][1].update(instance_id=True),'R')
    mutation('checkpoint_wrong_cursor','invalid_checkpoint',lambda t:t['checkpoint']['frames'][0].update(next_index=1),'R')
    mutation('checkpoint_wrong_binding_copy','invalid_checkpoint',lambda t:t['checkpoint']['frames'][2]['bindings'].update(a=3),'R')
    mutation('checkpoint_unnormalized_branch','invalid_checkpoint',lambda t:t['checkpoint']['frames'][2].update(next_index=3),'R')
    mutation('task_error_precedes_history_checkpoint','invalid_task',lambda t:(t.update(version=True,history='wrong',checkpoint='wrong')))
    mutation('history_error_precedes_checkpoint','invalid_history',lambda t:t.update(history='wrong',checkpoint='wrong'),'R')
    for name,stage,task in malformed:
        check('malformed_'+name,lambda task=task,stage=stage:equal((frame.solve(task),recursive.solve(task)),(frame.invalid(stage),frame.invalid(stage))))
    for name,raw in [('duplicate','{"x":1,"x":2}'),('nonfinite','{"x":NaN}')]:
        check('typed_JSON_reject_'+name,lambda raw=raw:frame.strict_json(raw),True)
    for context,task in tasks.items():
        reversed_task=controls.reverse_objects(task)
        check('recursive_object_order_'+context,lambda task=task,reversed_task=reversed_task:equal(frame.solve(frame.strict_json(json.dumps(reversed_task))),frame.solve(task)))
    def rename(value,mapping):
        if type(value) is dict:return {mapping.get(k,k):rename(v,mapping) for k,v in value.items()}
        if type(value) is list:return [rename(v,mapping) for v in value]
        return mapping.get(value,value) if type(value) is str else value
    for mapping in ({'main':'opaque-root','child':'opaque-leaf','p':'saved-one','q':'scale-factor','a':'captured-x','b':'factor'}, {'x':'y','y':'x'}):
        for context,task in tasks.items():
            check('alpha_rename_'+str(len(mapping))+'_'+context,lambda mapping=mapping,task=task:equal(frame.solve(rename(task,mapping)),rename(frame.solve(task),mapping)))
        check('three_structural_views_alpha_'+str(len(mapping)),lambda mapping=mapping:equal(controls.program_signatures(tasks['F']),controls.program_signatures(rename(tasks['F'],mapping))))
    moved=copy.deepcopy(tasks['F']);moved['options']['main']['body'].reverse()
    check('list_order_semantically_significant',lambda:equal(frame.typed_equal(frame.solve(moved),outputs['F']),False))
    padded=copy.deepcopy(tasks['F']);padded['options']['unused']={'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':[{'op':'swap'}]}
    check('unused_padding_not_reachable_novelty',lambda:equal(controls.program_signatures(padded)['coarse_reachable'],controls.program_signatures(tasks['F'])['coarse_reachable']))
    inactive=copy.deepcopy(tasks);inactive['X']['checkpoint']['frames'][1]['bindings']['a']=3;inactive['X']['checkpoint']['frames'][2]['bindings']['a']=2
    check('unused_saved_binding_copy_cannot_witness_activity',lambda:controls.check_controls(inactive,{k:frame.solve(v) for k,v in inactive.items()}),True)
    blocked=copy.deepcopy(tasks);blocked['T']['options']['main']['initiation']['all'][0]['right']['const']=1000
    check('blocked_restart_cannot_witness_replay',lambda:controls.check_controls(blocked,{k:frame.solve(v) for k,v in blocked.items()}),True)
    source_bytes={path:(ROOT/path).read_bytes() for path in controls.PUBLIC_PATHS};candidate_sha=hashlib.sha256(b'owned toy candidate bytes; not production').hexdigest()
    members={};entries=[];manifest_path='own-fixtures/public-controls.json'
    for context in tasks:
        for style in ('forward','reverse'):
            sample=f'opaque-{style}-{context}'
            row={'sample_id':sample,'source_family_key':'own-hand-family','source_family_index':12,'split':'train','controlled_context':context,'renderer':style}
            row['record_hash']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest();members[sample]=row
    for name in controls.NAMES:
        pairs=[([f'opaque-{style}-{c}' for c in controls.PAIRS[name]],style) for style in ('forward','reverse')] if name in controls.PAIRS else [([f'opaque-forward-{c}',f'opaque-reverse-{c}'],None) for c in tasks]
        for pair,style in pairs:
            first,second=(members[x] for x in pair)
            entries.append({'name':name,'scope':'corpus_variant','control_id':'shared arbitrary ID','pair_sample_ids':pair,'source_family_key':'own-hand-family','source_family_index':12,'controlled_contexts':[first['controlled_context'],second['controlled_context']],'renderers':[first['renderer'],second['renderer']],'candidate_sha256':candidate_sha,'pair_record_hashes':[first['record_hash'],second['record_hash']]})
    for owner in members.values():
        claims=[]
        for name in controls.NAMES:
            pair=[f"opaque-{owner['renderer']}-{c}" for c in controls.PAIRS[name]] if name in controls.PAIRS else [f"opaque-forward-{owner['controlled_context']}",f"opaque-reverse-{owner['controlled_context']}"]
            claim={'property':name,'scope':'corpus_variant','evidence_manifest':manifest_path,'control_id':'shared arbitrary ID','pair_sample_ids':pair,'diagnostics':{}}
            claims.append(claim)
        owner['family_control_claims']=claims
        check('nine_claims_'+owner['sample_id'],lambda owner=owner:controls.owner_claims(owner))
    # Canonical R2 order in these toy records: attach ID-only claims, hash complete
    # records excluding only their top-level hash, then construct external hashes.
    for row in members.values():
        content={k:v for k,v in row.items() if k!='record_hash'}
        row['record_hash']=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for entry in entries:entry['pair_record_hashes']=[members[v]['record_hash'] for v in entry['pair_sample_ids']]
    for owner in members.values():
        for claim in owner['family_control_claims']:
            check('synthetic_resolver_'+owner['sample_id']+'_'+controls.identifier(claim),lambda owner=owner,claim=claim:controls.resolve_claim(owner,claim,entries,members,candidate_sha))
    owner=members['opaque-forward-X'];claim=copy.deepcopy(owner['family_control_claims'][0]);pair=claim['pair_sample_ids']
    attacks=[('reversed_pair',lambda c:c['pair_sample_ids'].reverse()),('forbidden_canonical_hashes',lambda c:c.update(pair_record_hashes=['0'*64]*2)),('empty_control_id',lambda c:c.update(control_id='')),('bool_control_id',lambda c:c.update(control_id=True)),('mismatched_name_property',lambda c:c.update(name=controls.NAMES[1])),('wrong_owner_index',lambda c:c.update(source_family_index=13)),('boolean_owner_index',lambda c:c.update(source_family_index=True)),('absolute_manifest',lambda c:c.update(evidence_manifest='/tmp/file')),('parent_manifest',lambda c:c.update(evidence_manifest='../file')),('ignored_claim_field',lambda c:c.update(probe_id='synthetic'))]
    for name,mutate in attacks:
        changed=copy.deepcopy(claim);mutate(changed)
        check('metadata_attack_'+name,lambda changed=changed:controls.resolve_claim(owner,changed,entries,members,candidate_sha),True)
    omitted_hashes=copy.deepcopy(entries)
    for e in omitted_hashes:e.pop('pair_record_hashes')
    check('optional_external_member_hashes_absent_valid',lambda:controls.resolve_claim(owner,claim,omitted_hashes,members,candidate_sha))
    named=copy.deepcopy(claim);named['name']=named.pop('property')
    check('name_only_identifier_valid',lambda:controls.resolve_claim(owner,named,entries,members,candidate_sha))
    both=copy.deepcopy(claim);both['name']=both['property']
    check('matching_name_property_valid',lambda:controls.resolve_claim(owner,both,entries,members,candidate_sha))
    matching=next(e for e in entries if e['pair_sample_ids']==pair and e['name']==controls.NAMES[0])
    check('duplicate_complete_lookup_key',lambda:controls.resolve_claim(owner,claim,entries+[copy.deepcopy(matching)],members,candidate_sha),True)
    for name,alter in [('wrong_candidate',lambda e:e.update(candidate_sha256='0'*64)),('reordered_external_hashes',lambda e:e['pair_record_hashes'].reverse()),('wrong_external_hash',lambda e:e['pair_record_hashes'].__setitem__(0,'0'*64)),('wrong_entry_context',lambda e:e.update(controlled_contexts=['R','F']))]:
        changed=copy.deepcopy(entries);selected=next(e for e in changed if e['pair_sample_ids']==pair and e['name']==controls.NAMES[0]);alter(selected)
        check('entry_attack_'+name,lambda changed=changed:controls.resolve_claim(owner,claim,changed,members,candidate_sha),True)
    missing=copy.deepcopy(members);missing.pop(pair[0])
    check('missing_actual_member',lambda:controls.resolve_claim(owner,claim,entries,missing,candidate_sha),True)
    foreign=copy.deepcopy(members);foreign[pair[0]]['source_family_key']='foreign'
    check('foreign_actual_member',lambda:controls.resolve_claim(owner,claim,entries,foreign,candidate_sha),True)
    alias=copy.deepcopy(owner);alias['family_claims']=[]
    check('ignored_parallel_claim_alias',lambda:controls.owner_claims(alias),True)
    header={'metadata_version':1,'batch':'option-resumption-pilot-v1','candidate_sha256':candidate_sha,'source_bindings':{p:hashlib.sha256(raw).hexdigest() for p,raw in source_bytes.items()},'control_pairs':entries,'extra_diagnostics':{}}
    check('open_manifest_header_six_actual_sources',lambda:controls.manifest_header(header,source_bytes,candidate_sha,header['batch']))
    for name,mutate in [('bool_version',lambda m:m.update(metadata_version=True)),('missing_source',lambda m:m['source_bindings'].pop(controls.PUBLIC_PATHS[-1])),('false_source_hash',lambda m:m['source_bindings'].__setitem__(controls.PUBLIC_PATHS[0],'0'*64)),('wrong_candidate',lambda m:m.update(candidate_sha256='0'*64))]:
        changed=copy.deepcopy(header);mutate(changed)
        check('header_attack_'+name,lambda changed=changed:controls.manifest_header(changed,source_bytes,candidate_sha,header['batch']),True)
    for row in freeze['bindings']:
        if binding(ROOT/row['path'])!=row:raise ValueError('Frozen source/fixture/public/archive bytes changed')
    statuses={};mode_status={}
    for context,output in outputs.items():
        status=output['status'];statuses[status]=statuses.get(status,0)+1;mode=tasks[context]['mode'];mode_status.setdefault(mode,{})[status]=mode_status.setdefault(mode,{}).get(status,0)+1
    report={'schema':'public-only-rsm-adversarial-preparation-execution-v1','actor':'/root/adversarial_auditor','started_at_utc':started,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preexecution':binding(PREEXEC),'all_frozen_bindings_before_after_equal':True,'check_count':len(results),'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),'checks':results,'calibration_counts':{'hand_contexts':9,'additional_boundary_programs':9,'independent_fresh_programs':len(programs),'all_original_event_cuts':cuts,'resume_restart_cut_budget_cases':budget_cases,'two_interruptions_cases':second_interruptions,'synthetic_named_manifest_entries':len(entries),'synthetic_owner_claims':len(members)*9},'hand_control_evidence':control_evidence,'prospective_shortcut_probe':{'hand_status_histogram':statuses,'hand_mode_by_status':mode_status,'majority_correct':max(statuses.values()),'denominator':9,'limitation':'Hand fixture associations only; no production cohort input or outcome distribution read'},'read_scope':'Only frozen public documents and own newly authored fixtures/code. No legacy target-free inputs needed; no production inputs/gold/private checker/other preparations or verdicts. Earlier backup task accessed four RSM-named metadata files as opaque hash/span bytes only.','model_disclosure':'Same model and common public design/root guidance; independent algorithms and handmade expectations, shared grammar parser. No blind novelty, model diversity, production graph novelty or efficacy claim.','data_approval':False,'production_inspection':False}
    with RESULT.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({'check_count':report['check_count'],'passed':report['passed'],'failed':report['failed'],'counts':report['calibration_counts'],'execution':binding(RESULT)}))


if __name__=='__main__':main()
