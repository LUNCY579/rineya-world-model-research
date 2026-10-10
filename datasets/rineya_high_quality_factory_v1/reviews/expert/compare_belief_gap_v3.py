#!/opt/anaconda3/bin/python3
"""Separate post-prediction-freeze exact targets/property/control audit; own author."""
import collections,copy,datetime,itertools,json,pathlib,runpy,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert'
V=runpy.run_path(str(OUT/'review_belief_gap_v3.py'));OWN=V['OWN'];canon,equal,sha=V['canon'],V['equal'],V['sha'];ROLE=V['ROLE'];NEW=V['NEW']
def reverse(v):
 if isinstance(v,dict):return {k:reverse(v[k]) for k in reversed(v)}
 if isinstance(v,list):return [reverse(x) for x in v]
 return v
def order(v):
 if isinstance(v,dict):return [list(v),[order(x) for x in v.values()]]
 if isinstance(v,list):return [order(x) for x in v]
 return v
def main():
 p=OUT/'belief-gap-v3.input_only_predictions.json';b=p.read_bytes();pr=json.loads(pathlib.Path(str(p)+'.freeze_receipt.json').read_text());assert sha(b)==pr['prediction_sha256']
 d=json.loads(b);path=ROOT/'data/RBM-001/candidate_renderings/belief-gap-v3.jsonl';cb=path.read_bytes();assert sha(cb)==V['CAND_SHA'] and len(cb)==40541020
 candidates={r['sample_id']:r for r in map(json.loads,cb.splitlines())};assert len(candidates)==3456
 OWN['FAMILY_CONTROLS'].update(NEW)
 records=[];errors=[]
 for r in d['records']:
  row=candidates[r['sample_id']];t=OWN['parse_prompt'](row)
  assert equal(t,r['task']) and equal(t,row['semantic_task']) and row['record_hash']==r['record_hash']
  md,es=OWN['metadata_audit'](row,t,r['facts'])
  for k in ['auxiliary_scalar_variant','auxiliary_scalar_control']:md[k]=OWN['locate'](row,k)
  variant,control=md['auxiliary_scalar_variant'],md['auxiliary_scalar_control'];y=md['fact_role_assignments']['observation_auxiliary']
  if variant not in ('identity','transposed'):es.append('unknown auxiliary_scalar_variant')
  if type(control) is not dict or set(control)!={'fact','baseline_value','supported_value'}:es.append('incorrect auxiliary_scalar_control exact shape')
  else:
   base,supported=control['baseline_value'],control['supported_value']
   if control['fact']!=y or y in [q['fact'] for q in t['queries']]:es.append('auxiliary is queried or wrong role')
   if type(base) is not str or type(supported) is not str or base!='linen' or supported in {'stone','unknown','linen'}:es.append('typed diagnostic-control values wrong')
   if not all(any(equal(w['facts'][y],v) for w in t['worlds']) for v in [base,supported]):es.append('b/g absent on actual full-table auxiliary column')
  target_equal=equal(r['facts']['expected'],row['verified_target']);hash_equal=sha(canon({k:v for k,v in row.items() if k!='record_hash'}).encode())==row['record_hash']
  if not target_equal:es.append('frozen independently derived typed target mismatch')
  if not hash_equal:es.append('canonical record hash mismatch')
  allowed={2} if md['controlled_context'] in 'GH' else {2,3}
  if type(row['difficulty']) is not int or row['difficulty'] not in allowed or not equal(row['structural_difficulty'],row['difficulty']):es.append('unsupported structural component difficulty')
  if r['facts']['metrics']['duplicate_visible_payloads']:es.append('duplicate complete world payload')
  expected_split='train' if md['source_family_index'] in (12,13) else 'dev_challenge'
  if row['split']!=expected_split or row['semantic_parent_id']!=row['semantic_family_id']:es.append('family split/parent custody mismatch')
  if row['review_status']!='PENDING' or row['review_receipt_ids']:es.append('producer presents approved review metadata')
  for k in ['auxiliary_scalar_variant','auxiliary_scalar_control','property_witnesses','family_control_claims','verified_target']:
   if k in t or k in row['rendered_input']:es.append('nonstudent metadata leaked into prompt '+k)
  records.append({**r,'metadata':md,'metadata_errors':es,'target_equal':target_equal,'original_hash_equal':hash_equal,'declared_difficulty':row['difficulty'],'structural_difficulty':row['structural_difficulty']})
 lookup={(r['metadata']['source_family_key'],r['metadata']['controlled_context'],r['metadata']['auxiliary_scalar_variant'],r['metadata']['schedule_index'],r['metadata']['renderer']):r for r in records};assert len(lookup)==3456
 audits={}
 for variant in ['identity','transposed']:
  subset=[]
  for r in records:
   if r['metadata']['auxiliary_scalar_variant']==variant:
    z={**r,'metadata':{**r['metadata'],'family_control_claims':[c for c in r['metadata']['family_control_claims'] if c.get('name',c.get('property')) not in NEW]}}
    subset.append(z)
  audits[variant]=OWN['family_audits'](subset)
  errors += [{'scalar_variant':variant,**e} for e in audits[variant]['errors']]
 scalarpairs=[];opposite=[];fieldpairs=[];families=sorted({r['metadata']['source_family_key'] for r in records});newwitness=collections.defaultdict(lambda:collections.defaultdict(list));bysample={r['sample_id']:r for r in records}
 tables=set();identity_tables=set();visible_tables=set();probes=[]
 for r in records:
  md,t,f=r['metadata'],r['task'],r['facts'];fam,c,v,s,ren=(md[k] for k in ['source_family_key','controlled_context','auxiliary_scalar_variant','schedule_index','renderer']);control=md['auxiliary_scalar_control'];y,b,g=control['fact'],control['baseline_value'],control['supported_value']
  inv=OWN['decoded_task'](t,md);tables.add(canon(inv['worlds']));norm=V['transpose'](t,y,b,g) if v=='transposed' else t;identity_tables.add(canon(OWN['decoded_task'](norm,md)['worlds']));visible_tables.add(canon(t['worlds']))
  if v=='identity':
   other=lookup[(fam,c,'transposed',s,ren)];tt=V['transpose'](t,y,b,g)
   okay=equal(tt,other['task']) and equal(t,V['transpose'](tt,y,b,g)) and not equal(t['worlds'],tt['worlds']) and equal(f['expected'],other['facts']['expected']) and equal(f['prefix_survivors'],other['facts']['prefix_survivors']) and equal(control,other['metadata']['auxiliary_scalar_control'])
   pair={'name':'auxiliary_scalar_transposition_invariance','pair_sample_ids':[r['sample_id'],other['sample_id']],'pass':okay,'all_prefix_ids_equal':equal(f['prefix_survivors'],other['facts']['prefix_survivors']),'full_output_equal':equal(f['expected'],other['facts']['expected'])};scalarpairs.append(pair);newwitness[fam][pair['name']].append(pair)
   if not okay:errors.append({'pair':pair['pair_sample_ids'],'error':'actual scalar transposition/output/prefix invariance failed'})
  if c in 'EF':
   other=lookup[(fam,{'E':'G','F':'H'}[c],{'identity':'transposed','transposed':'identity'}[v],s,ren)]
   transformed_worlds=V['transpose'](t,y,b,g)['worlds']
   okay=equal(t['observations'],other['task']['observations']) and equal(t['queries'],other['task']['queries']) and equal(transformed_worlds,other['task']['worlds']) and not equal(t['worlds'],other['task']['worlds']) and equal(f['prefix_survivors'][1],other['facts']['prefix_survivors'][1]) and not f['expected']['surviving_worlds'] and bool(other['facts']['expected']['surviving_worlds'])
   pair={'name':'same_observations_auxiliary_table_counterfactual','pair_sample_ids':[r['sample_id'],other['sample_id']],'type':[c,v],'pass':okay,'exact_observations':t['observations'],'first_prefix_world_ids':f['prefix_survivors'][1],'final_supports':[len(f['expected']['surviving_worlds']),len(other['facts']['expected']['surviving_worlds'])]};opposite.append(pair);newwitness[fam][pair['name']].append(pair)
   if not okay:errors.append({'pair':pair['pair_sample_ids'],'error':'same-observation opposite-status table counterfactual failed'})
  if ren=='forward':
   other=lookup[(fam,c,v,s,'reverse')];a,at=candidates[r['sample_id']]['rendered_input'].split('Task JSON:\n');bb,bt=candidates[other['sample_id']]['rendered_input'].split('Task JSON:\n')
   okay=a==bb and at!=bt and equal(order(reverse(json.loads(at))),order(json.loads(bt)))
   fieldpairs.append({'pair_sample_ids':[r['sample_id'],other['sample_id']],'pass':okay})
   if not okay:errors.append({'sample_id':r['sample_id'],'error':'actual renderer not recursive field-key-only reversal'})
  for claim in md['family_control_claims']:
   if claim['scope']=='audit_probe':
    name=claim.get('name',claim.get('property'));w=[p for p in r['probes']['invariance_probes']+r['probes']['typed_equals_counterfactual_probes'] if p.get('name')==name]
    if not w:errors.append({'sample_id':r['sample_id'],'error':'advertised audit probe not independently executed: '+name})
    else:probes.append({'sample_id':r['sample_id'],'advertised_probe_id':claim.get('probe_id'),'expert_execution':ROLE,'own_executed_probe':w[0],'interpretation':'Own execution proves relationship; no inference about producer execution from pointer'})
 for fam in families:
  fr=[r for r in records if r['metadata']['source_family_key']==fam];controls={canon(r['metadata']['auxiliary_scalar_control']) for r in fr};splits={r['visible_identity']['split'] for r in fr};fids={r['visible_identity']['semantic_family_id'] for r in fr}
  okay=len(fr)==864 and len(controls)==len(splits)==len(fids)==1 and all(not audits[v]['errors'] for v in audits)
  newwitness[fam]['context_auxiliary_scalar_variant_crossing'].append({'pass':okay,'actual_rows':len(fr),'conditional_groups':32,'same_lineage_split_and_identity_controls':len(controls)==len(splits)==len(fids)==1})
  if not okay:errors.append({'family':fam,'error':'complete scalar-context family custody/crossing failed'})
  for r in fr:
   md=r['metadata'];variant=md['auxiliary_scalar_variant'];s=md['schedule_index'];ren=md['renderer'];E=lookup[(fam,'E','identity',s,ren)];G=lookup[(fam,'G','identity',s,ren)];control=md['auxiliary_scalar_control']
   if not equal(E['task']['observations'][1]['equals'],control['baseline_value']) or not equal(G['task']['observations'][1]['equals'],control['supported_value']):errors.append({'sample_id':r['sample_id'],'error':'identity-coordinate scalar control does not equal actual E/G values'})
   newclaims=[c for c in md['family_control_claims'] if c.get('name',c.get('property')) in NEW]
   if {c.get('name',c.get('property')) for c in newclaims}!=NEW:errors.append({'sample_id':r['sample_id'],'error':'missing required v3 family claim'})
   for claim in newclaims:
    name=claim.get('name',claim.get('property'));w=newwitness[fam][name]
    if claim.get('scope')!='corpus_variant':errors.append({'sample_id':r['sample_id'],'error':'v3 supplied control not corpus variant'})
    if 'pair_sample_ids' in claim:w=[p for p in w if set(p.get('pair_sample_ids',[]))==set(claim['pair_sample_ids'])]
    if not any(p['pass'] for p in w):errors.append({'sample_id':r['sample_id'],'error':'v3 exact claim has no reconstructed matching witness '+name})
 report={'report_type':'V3_POST_FREEZE_OWN_TARGET_METADATA_CONTROL_AUDIT','actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prediction_sha256':pr['prediction_sha256'],'candidate_sha256':sha(cb),'source_sha256':sha(pathlib.Path(__file__).read_bytes()),'guidance':V['GUIDANCE'],'records':records,'within_variant_family_audits':audits,'new_control_witnesses':dict(newwitness),'scalar_pairs':scalarpairs,'same_observations_opposite_status_pairs':opposite,'actual_field_order_pairs':fieldpairs,'own_actual_probe_claim_executions':probes,'errors':errors,'target_mismatches':sum(not r['target_equal'] for r in records),'record_hash_mismatches':sum(not r['original_hash_equal'] for r in records),'metadata_error_rows':sum(bool(r['metadata_errors']) for r in records),'counts':{'records':len(records),'families':len(families),'conditional_groups':sum(len(a['histograms']) for a in audits.values()),'histogram_axis_passes':sum(h['identical'] for a in audits.values() for g in a['histograms'].values() for h in g['outcome_histograms'].values()),'decoded_scalar_tables':len(tables),'decoded_identity_tables':len(identity_tables),'visible_tables':len(visible_tables),'scalar_pairs':len(scalarpairs),'field_pairs':len(fieldpairs),'opposite_status_pairs':len(opposite),'all_pairs_pass':all(p['pass'] for p in scalarpairs+fieldpairs+opposite),'opposite_pairs_disjoint_record_count':len({i for p in opposite for i in p['pair_sample_ids']})},'no_judgments_yet':True}
 out=OUT/'belief-gap-v3.post_freeze_comparison.json';h=V['save'](out,report);cp=V['compress'](out);V['save'](pathlib.Path(str(out)+'.compression.json'),cp)
 print(canon({'comparison_sha256':h,'counts':report['counts'],'target_mismatches':report['target_mismatches'],'record_hash_mismatches':report['record_hash_mismatches'],'metadata_error_rows':report['metadata_error_rows'],'errors':collections.Counter(e['error'] for e in errors),'compression':cp}))
if __name__=='__main__':main()
