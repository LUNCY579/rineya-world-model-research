#!/opt/anaconda3/bin/python3
"""NEW root-guided witness/code supplement after immutable first data receipts.
Real frozen checker _witness/_block_step on own pre-gold facts; bounded tiny
formal analyze fixture only. No root production invocation, oracle mutation,
producer/private reviewer verdict or rewritten old data review.
"""
import collections,copy,datetime,hashlib,json,pathlib,runpy,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert'
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
bind={
 'oracles/rgm_planning_controls_independent.py':'cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3',
 'quality_reports/RGM_PLANNING_V1_WITNESS_INTERFACE_DISCREPANCY.json':'b0764005552787401015a11fb54ca2f2c4d5868881a24172e93cf333a4bd356a',
 'reviews/expert/planning-gap-v1.expert_input_predictions.jsonl':'5fe3101f6ca6b6dcda5518b92fe659ffe254bf7862992609bccf9e7fc018f5b3',
 'reviews/expert/planning-gap-v1.first.jsonl':'9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72',
 'reviews/expert/planning-gap-v1.jsonl':'9b7ff4e888279ee152c065a978231f7ffd3a2c4cf46be3f6733477ce0d952d72',
 'data/RGM-001/candidate_renderings/planning-gap-v1.jsonl':'a01f88e1e11f23478b65d9825c0850975f03f0ca8fffb1345c0af672ee7a5c27',
 'data/RGM-001/manifests/planning-gap-v1.family_controls.json':'405e97f28c81e791e8614b369e8af89fa08b74da49a66114315a217bcf083233',
 'schemas/RGM_PLANNING_CONTROL_PROPERTIES.md':'f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a',
 'schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md':'02613aa2333f401117def89da224f169d564b527bacc554a0b872877aaac42fe',
 'reviews/expert/prepare_rgm_planning_gap_v1.py':'82ae6d000d5289bf439599579eb6437de5a09468388ead4fb1cef95f9dd3a987'}
data={};binding={}
for f,h in bind.items():
 b=(ROOT/f).read_bytes();assert sha(b)==h,(f,sha(b));data[f]=b;binding[f]={'sha256':h,'bytes':len(b)}
own=pathlib.Path(__file__).read_bytes();arch=OUT/'planning-gap-v1.expert_witness_guided.preexecution_source.py';assert arch.read_bytes()==own
oracle=ROOT/'oracles/rgm_planning_controls_independent.py';C={'__name__':'own_guided_review_frozen_checker','__file__':str(oracle)};exec(compile(data['oracles/rgm_planning_controls_independent.py'],str(oracle),'exec'),C)
pred={r['sample_id']:r for r in map(json.loads,data['reviews/expert/planning-gap-v1.expert_input_predictions.jsonl'].splitlines())};rows={r['sample_id']:r for r in map(json.loads,data['data/RGM-001/candidate_renderings/planning-gap-v1.jsonl'].splitlines())};manifest=json.loads(data['data/RGM-001/manifests/planning-gap-v1.family_controls.json']);P=runpy.run_path(str(OUT/'prepare_rgm_planning_gap_v1.py'))
records=[];transition_checks=0;formal_clearance_by={};prefix_agreements=0;start=time.perf_counter()
for sid,r in rows.items():
 t=r['semantic_task'];p=pred[sid];fs=p['independent_input_facts']['prefix_states'];target=p['expected_target'];clearance=[]
 if t['kind']=='blocks':
  initial=frozenset(t['on'].items())
  for i,step in enumerate(target['steps']):
   following,reason=C['_block_step'](frozenset(fs[i].items()),t['plan'][i]);transition_checks+=1
   nextstate=dict(following) if following is not None else fs[i]
   assert (following is not None)==step['legal'] and canon(nextstate)==canon(step['after']);prefix_agreements+=1
   if step['legal'] and i>=1:
    _,initial_reason=C['_block_step'](initial,t['plan'][i])
    if initial_reason in ('source_not_clear','destination_not_clear'):
     relevant=t['plan'][i]['block'] if initial_reason=='source_not_clear' else t['plan'][i]['to']
     clearing=[j for j in range(i) if relevant in fs[j].values() and relevant not in fs[j+1].values()]
     if clearing:clearance.append({'action_index':i,'precondition':initial_reason,'relevant_block':relevant,'changed_by_action_index':clearing[-1]})
 formal_clearance_by[sid]=clearance
def adapter(sid):
 t=rows[sid]['semantic_task'];p=pred[sid];f=p['independent_input_facts'];target=p['expected_target'];err=f['first_error'];search={'complete':True,'minimum_moves':None if target['shortest_plan'] is None else len(target['shortest_plan'])}
 if t['kind']=='graph':search.update({'topology_distance':None if f['topology_shortest_plan'] is None else len(f['topology_shortest_plan']),'cheapest_fuel':f['cheapest_required_fuel']})
 else:search.update({'initially_true_goal_relations':dict(f['initially_true_goal_relations']),'held_goal_reachable':f['preserve_initial_goal_minimum'] is not None})
 return {'search':search,'first_error':None if err is None else err['index'],'first_error_reason':None if err is None else err['reason'],'input_sha256':sha(canon(t).encode()),'target':target,'properties':{k:k in p['independently_witnessed_property_names'] for k in C['PROPERTIES']},'clearance':formal_clearance_by[sid]}
def attempt(e,p,t,result):
 try:C['_witness'](e,p,t,result);return None
 except C['OracleError'] as exc:return str(exc)
for f in manifest['families']:
 for cx in f['contexts']:
  for w in cx['row_witnesses']:
   sid=w['sample_id'];p=w['property'];t=rows[sid]['semantic_task'];result=adapter(sid);error=attempt(w['evidence'],p,t,result);normalized=copy.deepcopy(w['evidence'])
   if normalized.get('precondition') in ('source','destination'):normalized['precondition']+='_'+('not_clear')
   normerror=attempt(normalized,p,t,result)
   records.append({'sample_id':sid,'record_hash':rows[sid]['record_hash'],'source_family_key':f['source_family_key'],'property':p,'original_evidence':w['evidence'],'independent_pre_gold_fact_true':result['properties'][p],'original_frozen_checker_witness_error':error,'canonical_reason_alias_probe_error':normerror,'scope':'Real frozen _witness code on own independently frozen measurement/target/prefix facts translated to its search/result interface. Clearance reproduction uses actual frozen _block_step on every executed actual block prefix and the unchanged checker-source last-clearing algorithm. No full-batch BFS/signature/probe re-execution.'})
# Public property permits either genuinely changed blocked precondition. This
# authored tiny task has both source and destination initially blocked at i2.
toy={'kind':'blocks','on':{'a':'TABLE','b':'a','c':'TABLE','d':'c'},'goal_on':{'a':'c'},'plan':[{'block':'b','to':'TABLE'},{'block':'d','to':'TABLE'},{'block':'a','to':'c'}]}
hand= P['measurements'](toy);actual=C['analyze'](toy);assert canon(hand['target'])==canon(actual['target'])
public_source={'action_index':2,'changed_by_action_index':0,'precondition':'source_not_clear'};public_destination={'action_index':2,'changed_by_action_index':1,'precondition':'destination_not_clear'}
source_error=attempt(public_source,'later_action_requires_changed_block_clearance',toy,actual);destination_error=attempt(public_destination,'later_action_requires_changed_block_clearance',toy,actual)
assert source_error is None and destination_error is not None
assert actual['target']['steps'][2]['legal'] and 'a' in toy['on'].values() and 'c' in toy['on'].values() and 'a' not in actual['target']['steps'][0]['after'].values() and 'c' not in actual['target']['steps'][1]['after'].values()
errors=collections.Counter(r['original_frozen_checker_witness_error'] for r in records if r['original_frozen_checker_witness_error']);normerrors=collections.Counter(r['canonical_reason_alias_probe_error'] for r in records if r['canonical_reason_alias_probe_error']);failed=[r for r in records if r['original_frozen_checker_witness_error']];remaining=[r for r in records if r['canonical_reason_alias_probe_error']]
report={'review_type':'GUIDED_CODE_PUBLIC_INTERFACE_SUPPLEMENT','actual_author':'/root/expert_reviewer','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'elapsed_seconds':time.perf_counter()-start,'guidance':'Root supplied exact witness-encoding discrepancy after own first/final dataset receipts were frozen and then authorized original frozen checker/root diagnostic read. Shared underlying model and public prospective guidance disclosed. This supplement is corrective/root-guided, not a fresh blind initial dataset review. No other private reviewer verdict or producer/reference source read.','source_bindings':binding,'own_new_source_sha256':sha(own),'preexecution_archive_sha256':sha(arch.read_bytes()),'decision':'REVISE_CODE_INTERFACE_BEFORE_PUBLICATION','dataset_first_final_decisions_preserved':{'ACCEPT':384},'dataset_first_final_changed':False,'publication_approval':False,'publication_status':'Original frozen gate is unresolved; do not bypass it or relabel code/suite pass as batch admission. Fresh explicit interface/checker/generator revision with independent review is recommended.','actual_real_code_block_transition_checks':transition_checks,'all_prefix_transitions_agree':transition_checks==prefix_agreements,'actual_witness_checks':len(records),'all_independent_factual_truths_true':all(r['independent_pre_gold_fact_true'] for r in records),'original_checker_encoding_errors':dict(errors),'affected_rows':len({r['sample_id'] for r in failed}),'affected_families':len({r['source_family_key'] for r in failed}),'original_precondition_value_counts':dict(collections.Counter(r['original_evidence'].get('precondition') for r in failed)),'canonical_reason_alias_remaining_errors':dict(normerrors),'alias_probe_remaining_sample_ids':[r['sample_id'] for r in remaining],'records':records,'actual_toy_both_initial_preconditions_blocked':{'task':toy,'hand_minimum':hand['target']['shortest_plan'],'actual_frozen_checker_target':actual['target'],'actual_checker_clearance':actual['clearance'],'truthful_source_witness':public_source,'source_error':source_error,'truthful_destination_witness':public_destination,'destination_error':destination_error,'interpretation':'Even using checker-internal canonical reason strings, it discards a valid destination-clearance witness whenever source was also initially blocked, because initial_reason records one prioritized failure. Public property permits source OR destination witness, and both are factually supported here.'},'findings':[{'category':'unpublished_exclusive_precondition_enum','severity':'high','file':'oracles/rgm_planning_controls_independent.py','line':407,'evidence':'Frozen public property and encoding supplement require which source/destination precondition and actual changed clearance, but do not enumerate exact strings. source/destination are unambiguous and factually witnessed; strict equality to internal source_not_clear/destination_not_clear rejects them. Treat this as encoding/interface incompatibility, not demonstrated false public metadata or wrong targets.','fix':'Prospectively expose the exact witness interface and freeze a fresh checker/generator version; preserve original candidate and rejected gate evidence. Do not silently relax a frozen original gate or change old review receipts.'},{'category':'single_prioritized_failure_discards_other_true_clearance','severity':'high','file':'oracles/rgm_planning_controls_independent.py','line':213,'evidence':'Actual own hand-derived four-block fixture independently succeeds, and both initial source/destination clearances are changed by real preceding moves. Frozen analyze records only the source-first reason, so _witness rejects truthful destination indices even with canonical strings.','fix':'Fresh implementation must assess each source/destination condition independently at the named action and verify the actual named antecedent transition; retain lexical solver and all unrelated admission obligations.'}], 'root_report_comparison':'Root diagnostic was read as disclosed guidance only. Actual direct checker-function executions above independently reproduce witness errors on pre-gold facts. Whole production check_batch/oracle suite was not re-executed here; root reported 316 tests separately, which is not independent batch acceptance.','no_factory_or_oracle_source_modifications':True,'no_new_candidate_generation_targets_or_release_writes':True}
for f,h in bind.items():assert sha((ROOT/f).read_bytes())==h,'Protected input/review bytes changed during guided supplement'
op=OUT/'planning-gap-v1.expert_witness_interface_guided_supplement.json';assert not op.exists();op.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(canon({k:v for k,v in report.items() if k not in ['records','source_bindings','actual_toy_both_initial_preconditions_blocked','findings']}));print('supplement_sha256',sha(op.read_bytes()),'bytes',op.stat().st_size)
