"""Independent hash-bound adjudication of the new claim-control batch.

Uses public contracts, completed reviewers and the independently authored oracle.
Own manual graph replay and matrix/ranking assertions add an independent check.
Writes only review artifacts, never samples or releases.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,collections,sys,os,runpy,re
import jsonschema
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/adjudication'
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
ACTOR='/root/research_architect';BATCH='claim-gap-v2'
SCORES=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
def atomic(p,s):
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def replay(world,plan):
 edges={(a,b):c for a,b,c in world['edges']};node=world['start'];fuel=world['fuel']
 for index,dest in enumerate(plan):
  cost=edges.get((node,dest))
  if cost is None or cost>fuel:return {'node':node,'fuel':fuel},False,index
  node=dest;fuel-=cost
 return {'node':node,'fuel':fuel},True,None
def world_key(t):
 w=t['world'] if t['kind']=='critic' else t
 return digest(dict(nodes=sorted(w['nodes']),edges=sorted(w['edges']),start=w['start'],goal=w['goal'],fuel=w['fuel']))
cp=ROOT/f'data/RVM-001/candidate_renderings/{BATCH}.jsonl'
rps={role:ROOT/f'reviews/{role}/{BATCH}.jsonl' for role in ['expert','adversarial','oracle']}
ep=ROOT/'reviews/expert/verification_summary_claim_gap_v2.json'
er=ROOT/'reviews/expert/semantic_inspection_claim_gap_v2.md'
ap=ROOT/'reviews/adversarial/execution-claim-gap-v2.json'
control=ROOT/'audits/shortcut_detection/claim-gap-v2-controls.json'
lineagep=OUT/'claim-gap-v2-lineage-audit.json'
required=[cp,*rps.values(),ep,er,ap,control,lineagep,ROOT/'schemas/SEMANTICS.md',ROOT/'schemas/COMPETENCY_TAGS.md',ROOT/'schemas/canonical/v1.schema.json',ROOT/'oracles/independent.py']
if (OUT/f'{BATCH}.jsonl').exists():
 print('PRESERVED_EXISTING_FINAL');sys.exit(0)
if not all(p.exists() for p in required):
 print(json.dumps({'status':'PENDING','missing':[str(p.relative_to(ROOT)) for p in required if not p.exists()]}));sys.exit(2)
prior_paths=sorted(ROOT.glob('data/*/releases/*/canonical.jsonl'))
inputs=required+prior_paths+[ROOT/f'audits/{d}/{BATCH}.json' for d in ['leakage','diversity','shortcut_detection']]+[ROOT/'audits/shortcut_detection/claim-gap-v2-mechanism-tags.json',Path(__file__).resolve(),OUT/'adjudicate_metadata_revisions.py']
snapshots={str(p.relative_to(ROOT)):sha(p) for p in inputs}
rows=read(cp);assert len(rows)==len({r['sample_id'] for r in rows})==24
ids={r['sample_id'] for r in rows};reviews={role:{r['sample_id']:r for r in read(p)} for role,p in rps.items()}
assert all(len(v)==24 and set(v)==ids for v in reviews.values())
expert=json.loads(ep.read_text());adv=json.loads(ap.read_text());controls=json.loads(control.read_text())
assert expert['rows']==24 and expert['source_sha256']==sha(cp) and expert['all_sourcefamilies_substantively_inspected']==12
assert not expert['candidate_failures'] and adv['completed'] is True and adv['record_count']==24
assert adv['candidate_file_hashes'][str(cp.relative_to(ROOT))]==sha(cp) and adv['final_receipts_sha256']==sha(rps['adversarial'])
assert adv['semantics_sha256']==sha(ROOT/'schemas/SEMANTICS.md') and adv['matched_claim_controls_checked'] is True
assert controls['input_sha256']==sha(cp) and not controls['new_source_key_overlap'] and not controls['exact_world_overlap_prior']
assert adv['claim_control_evidence_sha256']==sha(control)
for k,h in controls['prior_release_sha256'].items():assert sha(ROOT/k)==h
groups=collections.defaultdict(list)
for r in rows:groups[r['source_family_key']].append(r)
assert len(groups)==12 and all(len(g)==2 and {r['renderer'] for r in g}=={'forward','reverse'} for g in groups.values())
assert all(len({r['semantic_family_id'] for r in g})==len({r['semantic_parent_id'] for r in g})==len({r['split'] for r in g})==1 for g in groups.values())
assert {int(re.search(r'-p(\d+)-',r['sample_id']).group(1)) for r in rows}==set(range(180,192))
assert all(canon(g[0]['semantic_task'])==canon(g[1]['semantic_task']) and canon(g[0]['verified_target'])==canon(g[1]['verified_target']) for g in groups.values())
assert len({world_key(g[0]['semantic_task']) for g in groups.values()})==12
prior_worlds=collections.defaultdict(list);prior_sourcekeys=set()
for p in prior_paths:
 for r in read(p):
  t=r['semantic_task']
  if t['kind'] in ['graph','critic']:prior_worlds[world_key(t)].append((str(p.relative_to(ROOT)),r['sample_id']))
  if r['track_id']=='RVM-001':
   key=r.get('source_family_key')
   if key:prior_sourcekeys.add(key)
   else:prior_sourcekeys.add('RVM-001-seed20261009-p'+re.search(r'-p(\d+)-',r['sample_id']).group(1))
assert not set(groups)&prior_sourcekeys
assert not {world_key(g[0]['semantic_task']) for g in groups.values()}&set(prior_worlds)
metadata=runpy.run_path(str(OUT/'adjudicate_metadata_revisions.py'),run_name='claim_metadata_helpers')['metadata_checks']
validator=jsonschema.Draft202012Validator(json.loads((ROOT/'schemas/canonical/v1.schema.json').read_text()))
receipt_rows=[];family_details={};unique_counts=collections.Counter();signed_magnitudes=collections.Counter();observations=0
for row in rows:
 sid=row['sample_id'];h=row['record_hash'];validator.validate(row)
 assert digest({k:v for k,v in row.items() if k!='record_hash'})==h
 e,a,o=[reviews[k][sid] for k in ['expert','adversarial','oracle']]
 assert all(r['record_hash']==h for r in [e,a,o])
 assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor' and o['oracle_author_execution']=='/root/formal_oracle'
 assert e['review_phase']=='completed_initial_blind_claim_gap_review'
 payload=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1]);assert canon(payload)==canon(row['semantic_task'])
 result=solve(payload);assert canon(result)==canon(row['verified_target']) and o['pass'] is True and not o['errors']
 checks,tag_details=metadata(row,result);assert all(checks.values())
 assert all(s not in row['rendered_input'] for s in [sid,row['source_family_key'],row['semantic_parent_id']])
 world=payload['world'];by_id={r['candidate_id']:r for r in result['evaluations']};rank={cid:i for i,g in enumerate(result['ranking']) for cid in g}
 plan_groups=collections.defaultdict(list);pool=collections.Counter();deltas=collections.Counter()
 for candidate in payload['candidates']:
  after,valid,first_illegal=replay(world,candidate['plan']);evaluation=by_id[candidate['candidate_id']]
  correct=canon(after)==canon(candidate['claimed_after']);delta=candidate['claimed_after']['fuel']-after['fuel'];location=after['node']==candidate['claimed_after']['node']
  expected_error=first_illegal if not valid else None if correct else 'claim'
  assert evaluation['valid'] is valid and evaluation['claim_correct'] is correct and evaluation['first_error']==expected_error
  assert evaluation['reaches_goal'] is (valid and after['node']==world['goal'])
  category='invalid' if not valid else 'wrong_location' if not location else 'over' if delta>0 else 'under' if delta<0 else 'exact'
  pool[category]+=1;observations+=1
  if valid and location and delta:deltas[str(delta)]+=1
  plan_groups[canon(candidate['plan'])].append(dict(candidate_id=candidate['candidate_id'],plan=candidate['plan'],actual_after=after,claim=candidate['claimed_after'],category=category,delta=delta,evaluation=evaluation,rank_group=rank[candidate['candidate_id']]))
 matrices=[]
 for plan,items in plan_groups.items():
  if len(json.loads(plan)) not in [0,1] or not all(i['evaluation']['valid'] for i in items):continue
  supported={i['category'] for i in items}
  if not {'exact','over','under','wrong_location'}<=supported:continue
  truth=[i for i in items if i['category']=='exact'];wrongs=[i for i in items if i['category']!='exact']
  assert all(i['delta']==0 for i in items if i['category']=='wrong_location')
  assert len({i['rank_group'] for i in wrongs})==1 and max(i['rank_group'] for i in truth)<min(i['rank_group'] for i in wrongs)
  fields=['valid','reaches_goal','remaining_moves','progress','usefulness','repair_plan']
  assert len({canon({k:i['evaluation'][k] for k in fields}) for i in items})==1
  over={i['delta'] for i in items if i['category']=='over'};under={-i['delta'] for i in items if i['category']=='under'}
  assert over&under,'Signed same-plan controls lack a matched absolute-error magnitude'
  matrices.append(dict(plan=json.loads(plan),actual_after=items[0]['actual_after'],claim_support={kind:[i['candidate_id'] for i in items if i['category']==kind] for kind in ['exact','over','under','wrong_location']},matched_signed_absolute_magnitudes=sorted(over&under),wrong_claims_share_exact_ranking_group=True,truth_strictly_outranks_wrong_claims=True,claim_independent_dynamics_fields=fields))
 assert len(matrices)==2 and {len(m['plan']) for m in matrices}=={0,1}
 if row['source_family_key'] not in family_details:
  unique_counts.update(pool);signed_magnitudes.update(deltas)
  family_details[row['source_family_key']]=dict(sample_id=sid,split=row['split'],independent_world_sha256=world_key(payload),candidate_count=len(payload['candidates']),valid_claim_type_counts=dict(pool),matrices=matrices)
 issues=[dict(i,source_role=role,source_receipt_id=r['receipt_id']) for role,r in [('expert',e),('adversarial',a)] for i in r.get('issues',[])]
 defects=[dict(d,source_role=role) if isinstance(d,dict) else dict(defect=d,source_role=role) for role,r in [('expert',e),('adversarial',a)] for d in r.get('critical_defects',[])]
 if defects or 'QUARANTINE' in [e['decision'],a['decision']]:decision='QUARANTINE'
 elif 'REJECT' in [e['decision'],a['decision']]:decision='REJECT'
 elif 'REVISE' in [e['decision'],a['decision']] or any(i.get('severity') in ['high','critical'] for i in issues):decision='REVISE'
 else:decision='ACCEPT'
 issues.append(dict(category='correlated_claim_matrix_scope',severity='low',evidence='Both renderer rows and all candidate labels share the family world. The batch has12 world families,199 distinct semantic candidates and24 renderings; only2 development families. Presence of matched signs/location controls does not create independent trials or new dynamics/objective classes.',fix='Use family-aware and corruption-stratified evaluation; report repeated motifs and2:1 full-pool over/under frequency.',source_role='independent_adjudicator'))
 scores={k:min(e['scores'][k],a['scores'][k]) for k in SCORES};scores['generalization_value']=min(scores['generalization_value'],3)
 receipt_rows.append(dict(receipt_id='adj-'+BATCH+'-'+digest([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores=scores,critical_defects=defects,issues=issues,decision=decision,confidence=0.98,reason='Both completed actual-execution reviewers ACCEPT this exact record. Hash/schema/prompt/full target/oracle, factual tags, independent graph replay and matched truth/over/under/location ranking invariants agree. Source families and exact worlds are new relative to prior releases. Targeted coverage closes missing signed/location support; pool imbalance, correlated family/candidate observations and familiar mechanisms remain disclosed.',evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=e['decision']!=a['decision'],oracle_receipt_file=f'reviews/oracle/{BATCH}.jsonl',oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],typed_record_hash_schema_prompt_target_checks=True,adjudicator_factual_tag_checks=checks,tag_details=tag_details,independent_candidate_replay_count=len(payload['candidates']),matched_same_plan_claim_matrices=matrices,source_family_key=row['source_family_key'],source_family_variants_same_parent_split=True,cross_release_source_family_overlap=0,cross_release_exact_graph_world_overlap_all_tracks=0,independent_lineage_execution='/root/research_architect/claim_lineage_audit',lineage_audit='reviews/adjudication/claim-gap-v2-lineage-audit.json'),scope_limitations=['Same underlying model across actual independent agents leaves correlated-error risk.','Adjudicator previously wrote curriculum specifications but never generated these samples.','Original sealed datasets/instruments/model artifacts were not opened.','claim-gap-v1 is quarantined historical material, not approval input.']))
assert observations==398 and len(family_details)==12
assert unique_counts=={'exact':71,'over':48,'under':24,'wrong_location':24,'invalid':32}
assert sum(d['candidate_count'] for d in family_details.values())==199
assert all(snapshots[str(p.relative_to(ROOT))]==sha(p) for p in inputs),'Evidence changed during adjudication'
counts=dict(collections.Counter(r['decision'] for r in receipt_rows));assert counts=={'ACCEPT':24}
output=OUT/f'{BATCH}.jsonl';atomic(output,''.join(canon(r)+'\n' for r in receipt_rows))
execution=dict(schema_version='1.0.0',reviewer_execution_id=ACTOR,role='Independent Quality Adjudicator',batch=BATCH,completed_at_utc=datetime.now(timezone.utc).isoformat(),records=24,new_source_families=12,source_family_indices=[180,191],distinct_semantic_candidates=199,rendered_candidate_observations=398,split_counts=dict(collections.Counter(r['split'] for r in rows)),difficulty_counts=dict(collections.Counter(str(r['structural_difficulty']) for r in rows)),decision_counts=counts,reviewer_disagreements_preserved=0,all_final_receipts_complete=True,typed_hash_schema_prompt_target_and_factual_tags_verified=True,independent_graph_replay_and_claim_matrix_checks=True,distinct_valid_claim_type_counts=dict(unique_counts),pure_valid_same_location_fuel_error_delta_counts=dict(signed_magnitudes),full_pool_sign_ratio='48 overclaims:24 underclaims,2:1;matched both-sign presence is not global frequency balance',matched_plan_groups_per_world=2,claim_independent_dynamics_and_exact_ranking_invariants=True,cross_release_source_family_overlap=0,cross_release_exact_graph_world_overlap_all_tracks=0,adversarial_normalized_motif_overlap_families=4,independent_lineage_execution='/root/research_architect/claim_lineage_audit',family_details=family_details,inputs_sha256=snapshots,output_sha256={str(output.relative_to(ROOT)):sha(output)},producer_or_release_files_modified=False,previous_quarantine_and_reviews_preserved=True,role_reuse_disclosure='Actual architect/adjudicator authored specifications, not candidates; reviewers and lineage auditor are actual separate executions. Same underlying model correlation remains.',review_blinding_disclosure='Expert initial claim-gap review reports no prior batch verdict read; adversarial executes public-contract/tag/control audits independently. Adjudicator consumes finalized reviews by design.')
execution['guided_followup_review_disclosure']='Expert completed initial blind claim-gap review without prior batch verdicts. Adversarial factual-tag/control followup is guided by previously disclosed tag/control topics; its exact facts are independently recomputed and no other reviewer verdict files are reported read. This is not represented as a new entirely blind correction execution.'
atomic(OUT/f'execution-{BATCH}.json',json.dumps(execution,indent=2)+'\n')
report='''# Claim-gap-v2 independent adjudication

Actual execution `/root/research_architect`:24 ACCEPT records after completed expert/adversarial reviews and independent lineage inspection. Exact canonical hashes, schema, visible payload/full target, oracle and factual tags agree. The adjudicator independently replayed all398 rendered candidate observations and checked both same-plan correctness matrices in every family.

There are12 new source families p0180–p0191,12 distinct exact graph worlds,199 distinct semantic candidates and24 forward/reverse renderings. Source keys and exact worlds do not overlap earlier generated releases across graph/critic tracks. Every family's variants keep one parent/split. Training has10 families/20 rows; exposed development2 families/4 rows. These are fresh world instances under familiar graph dynamics and critic objectives, not12 new reasoning mechanisms. Four normalized graph motifs are already exposed under the adversarial auditor's documented stripping convention.

Every family includes exact, fuel-overclaim, fuel-underclaim and exact-fuel wrong-location claims for the same empty and legal one-step plans, with matching absolute fuel-error magnitudes. Changing a claim leaves validity, actual goal reach, remaining distance, progress, usefulness and fresh initial repair unchanged. Wrong claims share exact ranking groups, while exact claims strictly outrank them; illegal action indices still take precedence over wrong claims.

The199 distinct candidate pool includes71 valid exact claims,48 valid same-location overclaims,24 underclaims,24 valid wrong-location claims and32 invalid candidates. The full pure-fuel mix remains2:1 over-to-under, despite both directions appearing in every matched stratum. Renderer/candidate labels remain correlated within12 worlds, development has only2 families, and the fixed corruption matrix still limits broad unseen-producer claims. No efficacy claim follows without separate authorized training/evaluation.

Both finalized reviewers ACCEPT all24 exact records; no disagreement is omitted. Their low issues and independent audit limits remain attached. Quarantined claim-gap-v1 was not used as approval evidence. This architect/adjudicator produced no samples and changed no producer or release files. Actual agent executions are separate, while same-model correlated error remains a limitation.

Review context is disclosed: expert initial claim-gap review reports no prior batch verdicts read; adversarial factual-tag/control followup is informed by previously disclosed tag/control topics. The facts were independently recomputed, with no other reviewer verdict files reported read. Guided followup is not presented as an entirely blind new correction execution.
'''
atomic(OUT/'ADJUDICATION_CLAIM_GAP_V2.md',report)
print(json.dumps({k:execution[k] for k in ['records','new_source_families','distinct_semantic_candidates','rendered_candidate_observations','decision_counts','output_sha256']}))
