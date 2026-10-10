#!/opt/anaconda3/bin/python3
"""Own valid-task minimal counterexamples to table-frequency/length lookups."""
import copy,json,pathlib,runpy,datetime,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';V=runpy.run_path(str(OUT/'review_belief_gap_v3.py'));Q=runpy.run_path(str(OUT/'inspect_belief_gap_v3.py'));equal,canon,sha=V['equal'],V['canon'],V['sha']
def main():
 d=json.loads((OUT/'belief-gap-v3.post_freeze_comparison.json').read_text());rows={r['sample_id']:r for r in map(json.loads,(ROOT/'data/RBM-001/candidate_renderings/belief-gap-v3.jsonl').read_text().splitlines())};probes=[]
 for r in d['records']:
  md=r['metadata']
  if md['controlled_context']!='E' or md['auxiliary_scalar_variant']!='identity' or md['schedule_index']!=0 or md['renderer']!='forward':continue
  t=r['task'];x,y=md['fact_role_assignments']['observation_control'],md['fact_role_assignments']['observation_auxiliary'];e,b=t['observations'][0]['equals'],t['observations'][1]['equals']
  left=next(w for w in t['worlds'] if equal(w['facts'][x],e));right=next(w for w in t['worlds'] if equal(w['facts'][y],b) and not equal(w['facts'][x],e))
  changed=copy.deepcopy(t);iw={w['world_id']:w for w in changed['worlds']};lid,rid=left['world_id'],right['world_id'];lv,rv=left['facts'][y],right['facts'][y];iw[lid]['facts'][y],iw[rid]['facts'][y]=rv,lv
  out,pref=V['solve'](changed);assert not r['facts']['expected']['surviving_worlds'] and len(out['surviving_worlds'])==1 and equal(pref[1],r['facts']['prefix_survivors'][1])
  lead=rows[r['sample_id']]['rendered_input'].split('Task JSON:\n')[0];mock={**rows[r['sample_id']],'rendered_input':lead+'Task JSON:\n'+json.dumps(changed,ensure_ascii=False,separators=(',',':'))};origmock={**mock,'rendered_input':lead+'Task JSON:\n'+json.dumps(t,ensure_ascii=False,separators=(',',':'))}
  rr={**r,'task':changed};old=Q['features'](r,origmock);new=Q['features'](rr,mock)
  keep=['ordered_typed_observation_scalars','first_prefix_ids','coarse_counts_first_prefix','observation_full_column_support_counts','typed_scalars_column_support_counts','observed_column_frequency_tables','all_column_frequency_tables','task_character_length','scalars_task_character_length','column_frequency_observation_lookup','prompt_character_length','scalars_prompt_character_length']
  assert all(equal(old[k],new[k]) for k in keep)
  probes.append({'probe_id':'expert-v3-frequency-counterexample-'+str(md['source_family_index']),'source_sample_id':r['sample_id'],'source_record_hash':r['record_hash'],'actual_author':V['ROLE'],'corpus_variant':False,'new_dataset_unit':False,'changed_task':changed,'changed_task_sha256':sha(canon(changed).encode()),'operation':'Exchange only the two auxiliary y values between two worlds; all per-column typed frequencies, scalar equalities, query list and first-prefix set remain unchanged. This is a valid finite world counterexample, not a purported invariance.','changed_world_ids':[lid,rid],'before_aux_values':[lv,rv],'after_aux_values':[rv,lv],'source_expected':r['facts']['expected'],'counterfactual_expected':out,'counterfactual_prefix_ids':pref,'invariant_input_feature_names':keep,'invariant_feature_values':{k:old[k] for k in keep},'legitimate_changed_conjunction_feature':[old['actual_pair_cell_intersection_counts'],new['actual_pair_cell_intersection_counts']]})
 assert len(probes)==4
 report={'report_type':'OWN_V3_POSTFREEZE_RESTRICTED_FEATURE_COUNTEREXAMPLES','actual_author':V['ROLE'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(pathlib.Path(__file__).read_bytes()),'candidate_sha256':d['candidate_sha256'],'prediction_sha256':d['prediction_sha256'],'actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'probes':probes,'probes_passed':4,'no_dataset_target_or_bytes_changed':True,'interpretation':'Perfect in-sample column-frequency lookup is finite corpus correlation. Four independently solved minimal column-swaps preserve its input but reverse status. Existing V3 paired controls establish the declared matched-observation gates, not universal shortcut immunity.'}
 h=V['save'](OUT/'belief-gap-v3.frequency_counterexamples.json',report);print(canon({'sha256':h,'passed':4}))
if __name__=='__main__':main()
