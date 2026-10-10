#!/usr/bin/env python3
"""Root-guided interface discrepancy follow-up AFTER frozen first judgments.
Author /root/adversarial_auditor. Extracts only authorized checker _witness;
never imports its solver, producer, expert or reference implementations.
"""
import collections,copy,hashlib,importlib.util,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
def b(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
COMPARISON=ROOT/'audits/rgm_planning_gap_v1_adversarial_comparison.py'
assert sha(COMPARISON)=='930f0c971b0da353f07a8f814aeb4ece372d962c41dce6f3b150d5184fe79b8e'
spec=importlib.util.spec_from_file_location('own_frozen_canonical',COMPARISON);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);P=C.P
CHECKER=ROOT/'oracles/rgm_planning_controls_independent.py'
assert sha(CHECKER)=='cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3'
class EncodingError(Exception):pass
def need(ok,msg):
    if not ok:raise EncodingError(msg)
def forbidden_locator(*args):raise AssertionError('No locator callback should be used by actual v1 witness shapes')

def own_result(t,p):
    a=p['expected_target'];info=p['search_evidence'];fail=info['first_failure'];clearance=[]
    if t['kind']=='blocks':
        # Own transition/facts; only the code vocabulary and last-clearing selection
        # are matched to the now-authorized checker for this guided encoding test.
        for i,(m,step) in enumerate(zip(t['plan'],a['steps'])):
            if not step['legal'] or i<1:continue
            _,_,reasons=P.block_move(t['on'],m);reason=reasons[0] if reasons else None
            if reason not in ('source_not_clear','destination_not_clear'):continue
            relevant=m['block'] if reason=='source_not_clear' else m['to'];changes=[]
            for j in range(i):
                before=t['on'] if j==0 else a['steps'][j-1]['after'];after=a['steps'][j]['after']
                if relevant in before.values() and relevant not in after.values():changes.append(j)
            if changes:clearance.append({'action_index':i,'precondition':reason,'relevant_block':relevant,'changed_by_action_index':changes[-1]})
    search={'minimum_moves':info['fresh_minimum_moves'],'topology_distance':info.get('topology_minimum_moves'),'cheapest_fuel':info.get('cheapest_required_fuel')}
    return {'target':a,'search':search,'properties':{k:k in p['independent_row_properties'] for k in C.PROPS},
            'first_error':fail['index'] if fail else None,'first_error_reason':fail['reasons'][0] if fail else None,
            'input_sha256':hashlib.sha256(canon(t).encode()).hexdigest(),'clearance':clearance}

def main():
    candidate=ROOT/C.FILES['candidate'][0];manifest=ROOT/C.FILES['manifest'][0];predpath=ROOT/C.FILES['predictions'][0]
    for role in ('candidate','manifest','predictions'):
        name,h=C.FILES[role];assert sha(ROOT/name)==h
    first=ROOT/'reviews/adversarial/planning-gap-v1.first.jsonl';final=ROOT/'reviews/adversarial/planning-gap-v1.jsonl'
    frozen_verdict='404c54fc35dc86ab4670abd9ab4dc2d666a6912d2bf1962a6dce1109efbda557';assert sha(first)==sha(final)==frozen_verdict
    source=pathlib.Path(__file__);freeze(ROOT/f'audits/history/rgm_planning_gap_v1_adversarial_interface_supplement-{sha(source)}.py',source.read_bytes())
    # Executable read scope is ONLY this already-authorized metadata function.
    text=CHECKER.read_text();start=text.index('def _witness(');end=text.index('\ndef _context_rules(',start);excerpt=text[start:end]
    env={'_need':need,'_json':canon,'OracleError':EncodingError,'_input_locator':forbidden_locator}
    exec(compile(excerpt,str(CHECKER)+':_witness-only','exec'),env);fn=env['_witness']
    rs={r['sample_id']:r for r in map(json.loads,candidate.open())};ps={r['sample_id']:r for r in map(json.loads,predpath.open())};m=json.loads(manifest.read_text())
    failures=[];normalized_failures=[];factual_errors=[];values=collections.Counter();total=0;actual_examples={}
    for f in m['families']:
        for c in f['contexts']:
            for w in c['row_witnesses']:
                total+=1;r=rs[w['sample_id']];p=ps[w['sample_id']];truth=C.witness_errors(r,p,w)
                if truth:factual_errors.append({'sample_id':r['sample_id'],'errors':truth})
                result=own_result(r['semantic_task'],p)
                try:fn(w['evidence'],w['property'],r['semantic_task'],result)
                except EncodingError as e:failures.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'family':f['source_family_key'],'property':w['property'],'evidence':w['evidence'],'error':str(e)})
                normalized=copy.deepcopy(w['evidence'])
                if 'precondition' in normalized:
                    values[normalized['precondition']]+=1;actual_examples.setdefault(normalized['precondition'],{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'actual_task':r['semantic_task'],'actual_witness':w,'own_clearance_facts':p['search_evidence']['changed_clearance_witnesses'],'guided_checker_vocabulary_facts':result['clearance']})
                    normalized['precondition']={'source':'source_not_clear','destination':'destination_not_clear'}.get(normalized['precondition'],normalized['precondition'])
                try:fn(normalized,w['property'],r['semantic_task'],result)
                except EncodingError as e:normalized_failures.append({'sample_id':r['sample_id'],'error':str(e),'normalized_evidence':normalized})
    # Actual minimal three-block examples: only encoding alias changes, truth unchanged.
    toys=[{'name':'minimal_source_clearance','task':{'kind':'blocks','on':{'A':'B','B':'TABLE','C':'TABLE'},'goal_on':{'B':'C'},'plan':[{'block':'A','to':'TABLE'},{'block':'B','to':'C'}]},'precondition':'source'},
          {'name':'minimal_destination_clearance','task':{'kind':'blocks','on':{'A':'TABLE','B':'TABLE','C':'B'},'goal_on':{'A':'B'},'plan':[{'block':'C','to':'TABLE'},{'block':'A','to':'B'}]},'precondition':'destination'}]
    for toy in toys:
        a,info=P.solve(toy['task']);p={'expected_target':a,'search_evidence':info,'independent_row_properties':P.factual_properties(toy['task'],a,info)};res=own_result(toy['task'],p)
        w={'sample_id':'own_toy','property':'later_action_requires_changed_block_clearance','scope':'row','evidence':{'action_index':1,'changed_by_action_index':0,'precondition':toy['precondition']}}
        assert not C.witness_errors({'sample_id':'own_toy','semantic_task':toy['task']},p,w)
        try:fn(w['evidence'],w['property'],toy['task'],res)
        except EncodingError as e:toy['actual_original_checker_error']=str(e)
        else:raise AssertionError('Original vocabulary unexpectedly accepted in toy')
        norm=copy.deepcopy(w['evidence']);norm['precondition'] += '_not_clear';fn(norm,w['property'],toy['task'],res)
        toy.update(original_evidence=w['evidence'],normalized_evidence=norm,own_exact_target=a,own_clearance_facts=info['changed_clearance_witnesses'],normalized_checker_passed=True,minimality_scope='Three blocks isolate one clearing move followed by a different source moved onto another block. Hand-derived fixture; no new source family.')
    guidance=['quality_reports/RGM_PLANNING_V1_WITNESS_INTERFACE_DISCREPANCY.json','quality_reports/RGM_PLANNING_V1_FIRST_AUDIT_EXECUTION.json','audits/diversity/planning-gap-v1-planning-controls.json']
    expected_guidance=['b0764005552787401015a11fb54ca2f2c4d5868881a24172e93cf333a4bd356a','4ac9d30168cba306cb3f1a154074490ca516a3d9aa1c659876dccd2752e7ebcb','f351c652dd68bd3b4504af5a0399d9c59a4517b276e604924c89af491b09d332']
    for path,h in zip(guidance,expected_guidance):assert sha(ROOT/path)==h
    rootreport=json.loads((ROOT/guidance[2]).read_text())['report'];direct=[e for e in rootreport['errors'] if 'incorrect clearance witness: precondition' in e];downstream=[e for e in rootreport['errors'] if e not in direct]
    strictpath=ROOT/'audits/execution/planning-gap-v1-adversarial-final-checks-v3.json';assert sha(strictpath)=='0c3212d14ed0385d0b57d8ef8e53bbd9a371520297420777da6f4d4efd983d16';strict=json.loads(strictpath.read_text())
    isolated=strict['old_guard_isolated_failure_counts'];allguards=strict['old_guard_all_violated_counts']
    correction={'supersedes_only':'guard_count_interpretation prose in planning-gap-v1-adversarial-final-checks-v3.json, not measurements or verdicts',
                'unchanged_support':{'all_violations':allguards['unchanged_support'],'isolated':isolated.get('unchanged_support',0)},
                'destination_not_clear':{'all_violations':allguards['destination_not_clear'],'isolated':isolated.get('destination_not_clear',0)},
                'interpretation':'Prior unchanged-support has11 isolated failures among34 total violations. Destination-not-clear has12 total violations and0 isolated failures under own all-guard evaluation. Aggregate violated-guard counts must not imply selective ablation coverage. Earlier combined assertion failure did not identify which conjunct failed; interim no-op inference was too broad.',
                'preserved_failed_executions':strict['preserved_failed_executions'],'strict_input_matches':strict['input_matches'],'strict_allocation_checks':strict['allocation_policy_checks_completed'],'conditional_histograms_sha256':strict['conditional_histograms_sha256']}
    out={'actual_author_execution':'/root/adversarial_auditor','completed':True,'review_mode':'ROOT_GUIDED_INTERFACE_FOLLOWUP_AFTER_FIRST_CANONICAL_FREEZE','source_sha256':sha(source),
         'candidate_sha256':sha(candidate),'manifest_sha256':sha(manifest),'first_and_final_verdict_sha256_preserved':sha(first),'own_input_prediction_sha256':sha(predpath),
         'authorized_checker_sha256':sha(CHECKER),'authorized_witness_function_excerpt_sha256':hashlib.sha256(excerpt.encode()).hexdigest(),
         'actual_execution_scope':'Extracted checker _witness only; target/search/clearance facts from own frozen input predictions and own preparation transitions. Checker solver module never imported. Guidance and metadata code read only after own first384 judgments froze.',
         'guided_sources':[{'path':p,'sha256':sha(ROOT/p)} for p in guidance],
         'public_sources':[{'path':p,'sha256':sha(ROOT/p)} for p in P.PUBLIC],
         'witnesses_executed':total,'own_input_factual_or_locator_errors':factual_errors,'original_encoding_errors':failures,'original_encoding_error_count':len(failures),
         'affected_source_families':len({e['family'] for e in failures}),'precondition_literal_counts':dict(values),
         'after_alias_normalization_errors':normalized_failures,'after_alias_normalization_error_count':len(normalized_failures),'alias_normalization_is_a_probe_only_not_candidate_mutation':True,
         'minimal_hand_derived_encoding_counterexamples':toys,'actual_exact_hash_sample_examples':actual_examples,
         'root_first_gate_direct_family_errors':direct,'root_first_gate_downstream_aggregate_errors':downstream,
         'early_abort_interpretation':{'direct_failed_families':len(direct),'downstream_errors':len(downstream),'root_complete_family_count':rootreport['counts']['complete_families'],
             'own_completed_families':48,'own_supplied_controls':84,'own_six_block_initial_shapes':11,'own_C0_temporary_displacement_bases':7,'own_C1_index_support':[0,1,2],
             'conclusion':'Nine aggregate coverage/shape errors are downstream of21 family validation aborts before diagnostics/families are appended. They do not demonstrate missing supplied families, controls, varied C1 indices or missing six-block arrangements. Own exact-input checks establish these actual facts.'},
         'classification':{'decision':'INTERFACE_ALIGNMENT_REQUIRED_FOR_PUBLICATION','severity':'medium','wrong_target_demonstrated':False,'false_input_derived_property_demonstrated':bool(factual_errors),
             'unsupported_encoding_under_frozen_checker':True,'frozen_public_enum_declared':False,
             'interpretation':'Frozen public contracts require a truthful row-bound clearance precondition and prior clearing action, but do not declare source/destination versus source_not_clear/destination_not_clear values. The actual source/destination values denote the independently proved guard. Checker equality against its internal failure-reason vocabulary introduces an undeclared encoding convention. Classify as unresolved machine-interface alignment, not a false property or absent supplied family.',
             'publication_requirements':['Preserve the frozen v1 candidate, first judgments, original checker failure and this guided disagreement. Do not retroactively relax v1 gating or silently rewrite evidence.',
                 'Publish an explicit prospective witness vocabulary and normalization/locator policy before a fresh versioned candidate/checker; independently review those new bytes.',
                 'Keep task truth, metadata encoding acceptance, and derived incomplete-gate coverage counts separate. Existing formal gate remains blocked until the discrepancy is explicitly resolved by the authorized publication process.',
                 'Prospective dev infeasibility and selective block-guard controls are useful coverage repairs; no new v2 data is reviewed or accepted here.']},
         'guard_count_interpretation_correction':correction,'original_first_final_receipts_unchanged':True,'no_dataset_receipt_rewrites':True,'no_release_or_pipeline_writes':True,
         'scope_limits':['This supplement is guided by root diagnostic disclosure and authorized frozen checker metadata code; it is not a second blind independent review.',
                         'Original first input predictions and canonical judgments remain independently executed and frozen before this exposure. Common underlying model/shared public design correlation persists.']}
    path=ROOT/'audits/execution/planning-gap-v1-adversarial-interface-supplement.json';freeze(path,b(out));print(canon({'source_sha256':sha(source),'supplement_sha256':sha(path),'original_encoding_errors':len(failures),'normalized_errors':len(normalized_failures),'factual_errors':len(factual_errors),'downstream_aggregate_errors':len(downstream),'guard_correction':correction['interpretation']}))
if __name__=='__main__':main()
