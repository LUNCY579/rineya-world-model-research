#!/opt/anaconda3/bin/python3
"""Actual expert compact inspection extraction after own corrective audit freeze.
No external verdict or implementation. Read only own completed analysis and
actual authorized canonical/manifest for bounded human inspection evidence.
"""
import datetime,hashlib,json,pathlib,sys,collections
R=pathlib.Path(__file__).resolve().parents[2];O=R/'reviews/expert';ROLE='/root/expert_reviewer';P='belief-gap-v4-canonical-inspection-attempt01'
B={
 'reviews/expert/belief-gap-v4-canonical-attempt03.analysis.json':'098ac542d42fd38bd89826303e81f45bddc99d3c265e93cc27f07e283fec5d59',
 'reviews/expert/belief-gap-v4-canonical-attempt03.execution.json':'0da3142a23c77c156cc24adeff973052e7c0c22a32cc287474cf037e50e9645a',
 'reviews/expert/belief-gap-v4.canonical-corrective-attempt03.jsonl':'155f65a1766addc6322b55ff51c6c5bf71f9c32cd8f536935ca479eeccc99d4a',
 'reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl':'b1cd43c4c0e54acce5964e2010bb8ae431dfd53e6b7971d842770a7579c6b6fc',
 'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl':'38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c',
 'data/RBM-001/manifests/belief-gap-v4.family_controls.json':'6b91a54b60dcf5a702889452a29ec5348acb4eb8c0ef62b1a03408f04fad2631',
 'reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl':'d9a55125b127edc6ab691f4981ea61edfa60b03b333e90ad4fc7c865b1ccc9ed'}
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def hp(p):
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
 return {'sha256':h.hexdigest(),'length_bytes':n}
def save(p,v):
 with p.open('x') as f:f.write(canon(v)+'\n')
 return hp(p)
def bindings():
 b={p:hp(R/p) for p in B};assert all(b[p]['sha256']==h for p,h in B.items());return b
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def freeze():
 p=pathlib.Path(__file__).resolve();a=O/(P+'.source.preexecution.py')
 with a.open('xb') as f:f.write(p.read_bytes())
 v={'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':now(),'source':hp(p),'archive':hp(a),'bindings_before':bindings(),'scope':'Own completed facts/actual metadata extraction for actual-agent inspection; no solver expectations from manifest','python_executable':sys.executable}
 print(canon(save(O/(P+'.preexecution.json'),v)))
def run():
 p=pathlib.Path(__file__).resolve();pre=json.loads((O/(P+'.preexecution.json')).read_text());before=bindings();assert before==pre['bindings_before'];assert hp(p)==pre['source']==hp(O/(P+'.source.preexecution.py'))
 a=json.loads((O/'belief-gap-v4-canonical-attempt03.analysis.json').read_text());m=json.loads((R/'data/RBM-001/manifests/belief-gap-v4.family_controls.json').read_text())
 rows={}
 for line in (R/'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl').open():
  r=json.loads(line)
  if r['auxiliary_scalar_variant']=='identity' and r['schedule_index']==0 and r['renderer']=='forward':rows[r['sample_id']]=r
 assert len(rows)==32
 predictions={}
 for line in (O/'belief-gap-v4-input-attempt01.predictions.jsonl').open():
  p1=json.loads(line)
  if p1['sample_id'] in rows:predictions[p1['sample_id']]=p1
 resolver={(e['control_id'],e['name'],tuple(e['pair_sample_ids'])):e for e in m['supplied_same_table_v2_pairs']};examples=[]
 for sid,r in rows.items():
  claims=[c for c in r['family_control_claims'] if c.get('name',c.get('property')) in {'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}]
  resolved=[resolver[(c['control_id'],c.get('name',c.get('property')),tuple(c['pair_sample_ids']))] for c in claims]
  examples.append({'sample_id':sid,'record_hash':r['record_hash'],'family':r['source_family_key'],'context':r['controlled_context'],'observations':r['semantic_task']['observations'],'queries':r['semantic_task']['queries'],'actual_complete_target':r['verified_target'],'difficulty':r['difficulty'],'structural_difficulty':r['structural_difficulty'],'tags':r['skill_tags'],'property_locators':r['property_witnesses'],'claims':claims,'actual_named_entries':resolved,'scalar_control':r['auxiliary_scalar_control']})
 result={k:a[k] for k in ['counts','preservation_counts','actual_changed_top_level_fields_vs_v3','added_semantic_families','added_student_inputs','row_errors','batch_errors','actual_attached_property_tag_counts','profiles','actual_null_family_category_support','manifest_named_entry_usage_histogram','frozen_public_rule_actual_counts','fitted_input_status_lookups','independently_solved_frequency_counterexamples']}
 result.update({'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':now(),'source':hp(p),'exact_actual_inspected_ids':list(rows),'actual_representative_records_and_resolved_entries':examples,'own_input_prediction_hash':B['reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl'],'conditional_first_group_supplied_position_labels':list(m['actual_conditional_query_histograms'][0]['histograms']['position']['actual_role_histograms']),'not_an_external_review_or_release_approval':True})
 out=O/(P+'.evidence.json');save(out,result);after=bindings();assert before==after
 ep=O/(P+'.execution.json');save(ep,{'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':now(),'source_before_after':hp(p),'preexecution':hp(O/(P+'.preexecution.json')),'bindings_before':before,'bindings_after':after,'evidence':hp(out),'exit_code':0,'row_examples':32,'no_external_results_read':True})
 print(canon({'evidence':hp(out),'execution':hp(ep),'counts':result['counts'],'preservation':result['preservation_counts'],'changed_metadata':result['actual_changed_top_level_fields_vs_v3'],'lookups':result['fitted_input_status_lookups'],'rules':result['frozen_public_rule_actual_counts'],'profiles':result['profiles'],'null_support':result['actual_null_family_category_support'],'named_entry_usage':result['manifest_named_entry_usage_histogram']}))
if __name__=='__main__':
 {'freeze':freeze,'run':run}[sys.argv[1]]()
