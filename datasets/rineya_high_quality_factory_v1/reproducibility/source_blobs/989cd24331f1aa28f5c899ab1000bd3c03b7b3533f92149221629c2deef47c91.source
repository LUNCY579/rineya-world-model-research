"""Independent final adjudication. Requires both final reviewers before writing.
Only writes reviews/adjudication. Uses no producer implementation.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,sys,os
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/adjudication'
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
ACTOR='/root/research_architect';BATCH='expansion-v1'
SCORES=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def dig(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
if (OUT/f'{BATCH}.jsonl').exists():
 print('PRESERVED_EXISTING_FINAL: expansion-v1 adjudication already exists. Consult supplemental execution-input-scope erratum.');sys.exit(0)
def atomic(p,s):
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
paths=[ROOT/f'data/{t}/candidate_renderings/{BATCH}.jsonl' for t in ['RCM-001','RGM-001','RVM-001']]
rpaths={r:ROOT/f'reviews/{r}/{BATCH}.jsonl' for r in ['expert','adversarial','oracle']}
if not all(p.exists() for p in rpaths.values()):
 print('PENDING: no adjudication written; both independent final reviewer receipts must exist.');sys.exit(2)
rows=[r for p in paths for r in read(p)]
assert len(rows)==1860 and len({r['sample_id'] for r in rows})==1860
reviews={role:{r['sample_id']:r for r in read(p)} for role,p in rpaths.items()}
review_snapshots={role:sha(p) for role,p in rpaths.items()}
assert all(len(v)==1860 and set(v)=={r['sample_id'] for r in rows} for v in reviews.values())
advp=ROOT/f'reviews/adversarial/execution-{BATCH}.json';adv=json.loads(advp.read_text())
assert adv['record_count']==1860 and all(adv['candidate_file_hashes'][str(p.relative_to(ROOT))]==sha(p) for p in paths)
assert adv['semantics_sha256']==sha(ROOT/'schemas/SEMANTICS.md')
assert adv['final_receipt_coverage_exact_hash_verified'] is True and adv['decision_counts']=={'ACCEPT':1780,'REVISE':80}
expertp=ROOT/'reviews/expert/verification_summary_expansion_v1.json';expert=json.loads(expertp.read_text())
assert expert['reviewer_execution_id']=='/root/expert_reviewer' and expert['rows']==1860
assert all(expert['source_sha256'][str(p.relative_to(ROOT))]==sha(p) for p in paths)
scopep=OUT/'expansion-v1_scope_review.json';scope=json.loads(scopep.read_text())
assert scope['records']==1860 and scope['independent_source_families']==450
assert all(scope['inputs_sha256'][str(p.relative_to(ROOT))]==sha(p) for p in paths)
assert all(scope['inputs_sha256'][f'data/{t}/releases/v0001/canonical.jsonl']==sha(ROOT/f'data/{t}/releases/v0001/canonical.jsonl') for t in ['RCM-001','RGM-001','RVM-001'])
lineage=json.loads((ROOT/f'audits/leakage/{BATCH}_cross_release.json').read_text())
assert not any(lineage.values())
leakage=json.loads((ROOT/f'audits/leakage/{BATCH}.json').read_text())
assert not leakage['split_leakage'] and not leakage['cross_parent_semantic_duplicates'] and leakage['oracle_failures']==0
family_groups=collections.defaultdict(list)
for r in rows:family_groups[r['semantic_family_id']].append(r)
assert len(family_groups)==450 and all(len({r['split'] for r in g})==len({r['semantic_parent_id'] for r in g})==1 for g in family_groups.values())
def equal_routes(world):
 # Iterative length-by-length path multiplicities over node/spent fuel.
 # Capping at two is sufficient to prove or refute a distinct equal-length pair.
 adjacency=collections.defaultdict(list)
 for a,b,c in world['edges']:adjacency[a].append((b,c))
 frontier={(world['start'],0):1};maximum=world['fuel']//min(c for a,b,c in world['edges'])
 for horizon in range(1,maximum+1):
  nxt=collections.defaultdict(int)
  for (node,spent),count in frontier.items():
   for dest,cost in adjacency[node]:
    if spent+cost<=world['fuel']:
     key=(dest,spent+cost);nxt[key]=min(2,nxt[key]+count)
  terminal_count=sum(count for (node,spent),count in nxt.items() if node==world['goal'])
  if terminal_count>1:return True
  frontier=nxt
  if not frontier:break
 return False
receipts=[];oracle_failures=[];metadata_defects=collections.Counter();route_cache={}
for row in rows:
 sid=row['sample_id'];h=row['record_hash'];e,a,o=[reviews[k][sid] for k in ['expert','adversarial','oracle']]
 assert dig({k:v for k,v in row.items() if k!='record_hash'})==h
 assert all(r['record_hash']==h for r in [e,a,o])
 assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor'
 assert o['oracle_author_execution']=='/root/formal_oracle'
 assert e['reviewer_execution_id']!=a['reviewer_execution_id']!=ACTOR
 assert e['decision'] in ['ACCEPT','REVISE','REJECT','QUARANTINE'] and a['decision'] in ['ACCEPT','REVISE','REJECT','QUARANTINE']
 assert e['review_phase']=='completed_initial_blind_semantic_review' and e['review_status']==('ACCEPTED' if e['decision']=='ACCEPT' else e['decision'])
 issues=[dict(i,source_role=role,source_receipt_id=r['receipt_id']) for role,r in [('expert',e),('adversarial',a)] for i in r.get('issues',[])]
 defects=[dict(d,source_role=role) if isinstance(d,dict) else {'defect':d,'source_role':role} for role,r in [('expert',e),('adversarial',a)] for d in r.get('critical_defects',[])]
 payload=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1]);prompt_matches=canon(payload)==canon(row['semantic_task'])
 result=solve(payload);target={'after':result['after']} if row['target_format']=='answer_only' else result
 target_matches=canon(target)==canon(row['verified_target'])
 oracle_pass=o['pass'] is True and not o['errors']
 metadata_issue=None
 if row['track_id']=='RGM-001' and 'branched_resource_graph' in row['skill_tags']:
  degrees=collections.Counter(a for a,b,c in payload['edges'])
  if max(degrees.values(),default=0)<=1:
   metadata_issue={'claimed_mechanism':'branched_resource_graph','actual_outdegree_branch_nodes':0,'outdegree_multiset':sorted(degrees.get(n,0) for n in payload['nodes']),'affected_v1_records':26,'affected_source_families':13}
 if row['track_id']=='RGM-001' and 'multi_stack_goal_composition' in row['skill_tags']:
  initial_roots=sum(v=='TABLE' for v in payload['on'].values());goal_roots=sum(v=='TABLE' for v in payload['goal_on'].values())
  if initial_roots<2 and goal_roots<2:
   metadata_issue={'claimed_mechanism':'multi_stack_goal_composition','initial_table_roots':initial_roots,'explicit_goal_table_roots':goal_roots,'affected_v1_records':14,'affected_source_families':7}
 if row['track_id']=='RVM-001':
  if any('initially_unreachable' in tag for tag in row['skill_tags']) and result['evaluations'][0]['repair_plan'] is not None:
   metadata_issue={'claimed_mechanism':'initially_unreachable','verified_initial_goal_plan':result['evaluations'][0]['repair_plan'],'affected_v1_records':22,'affected_source_families':11}
  if any(tag.endswith(':equal_length_routes') for tag in row['skill_tags']):
   key=dig(payload['world'])
   if key not in route_cache:route_cache[key]=equal_routes(payload['world'])
   if not route_cache[key]:
    metadata_issue={'claimed_mechanism':'equal_length_routes','distinct_feasible_same_length_terminal_routes':False,'check':'Iterative saturated route count over every move horizon through fuel/minimum positive edge cost, aggregating every fuel use.','affected_v1_records':18,'affected_source_families':9}
 if metadata_issue:
  metadata_defects[metadata_issue['claimed_mechanism']]+=1
  issues.append(dict(category='incorrect_competency_metadata',severity='moderate',evidence=metadata_issue,fix='Correct the factual tag in a fresh version and obtain exact-hash final reviews; retain this v1 record as REVISE.',source_role='independent_adjudicator'))
 if not(prompt_matches and target_matches and oracle_pass):
  decision='QUARANTINE';defects.append({'defect':'independent_payload_or_target_mismatch','source_role':'independent_adjudicator'});oracle_failures.append(sid)
 elif defects or 'QUARANTINE' in [e['decision'],a['decision']]:decision='QUARANTINE'
 elif 'REJECT' in [e['decision'],a['decision']]:decision='REJECT'
 elif metadata_issue or 'REVISE' in [e['decision'],a['decision']] or any(i.get('severity') in ['high','critical'] for i in issues):decision='REVISE'
 else:decision='ACCEPT'
 own=[dict(category='independent_family_vs_mechanism',severity='low',evidence='Each source family is a fresh generated world instance; many families share fixed primitives, base predicate classes or corruption templates. Surface/twin variants are not new independent concepts.',fix='Report150 new families per track and mechanism-level coverage separately; avoid claiming150 unique reasoning mechanisms.',source_role='independent_adjudicator')]
 if row['track_id']=='RCM-001' and row['split']=='dev_challenge':
  root=row['semantic_task']['rule'];wrapper=root if root['op']=='all' else root['args'][0];boundary=wrapper['args'][1]['args'][0]['args'][1]['const']
  if all(ev['amount']>boundary for ev in row['semantic_task']['events']):
   own.append(dict(category='redundant_nested_control',severity='low',evidence='The added any amount clause is true for every supplied event, so the wrapper leaves base rule applicability unchanged.16 of30 dev families serve these redundant-shape controls.',fix='Count separately from14 gate-varying dev families; add cases requiring both sides of the new conjunction for broader composition coverage.',source_role='independent_adjudicator'))
  own.append(dict(category='bounded_nested_predicate_shift',severity='low',evidence='Across60 dev twin tasks, wrapper removal changes25 rule traces/21 final states in14/12 families. This supports a bounded nested-control challenge, with concurrent larger magnitudes/first-status shift.',fix='Report factors separately and retain no broad causal discovery or general natural-language rule claim.',source_role='independent_adjudicator'))
 if row['track_id']=='RGM-001':
  t=row['semantic_task']
  if t['kind']=='graph' and not any(sum(e[0]==n for e in t['edges'])>1 for n in t['nodes']):
   own.append(dict(category='nonbranching_control_scope',severity='low',evidence='13 of90 source graph families are simpler controls. The accompanying false branched skill tag is separately treated as a factual revision defect.',fix='Report nonbranching controls separately and correct the skill tag in a new version.',source_role='independent_adjudicator'))
 if row['track_id']=='RVM-001':
  own.append(dict(category='critic_shift_scope',severity='low',evidence='Branched worlds and shuffled IDs vary outcomes, but the scoring objective and familiar positive-offset/self-loop/repeated-node error families remain shared.',fix='Expand independently held-out error producers and goal functions before robust critic/goal-transfer claims.',source_role='independent_adjudicator'))
 issues+=own
 issues.append(dict(category='cross_release_mechanism_overlap',severity='low',evidence='The completed adversarial scope audit identifies214 current training logical variants sharing AST motifs with previously exposed development. Exact tasks and source families are disjoint, but mechanism novelty and prior-development isolation are not established.',fix='Keep exposed-development and repeated-mechanism claims explicit; future sealed custody must use independently allocated mechanism families.',source_role='independent_adjudicator'))
 scores={k:min(e['scores'][k],a['scores'][k]) for k in SCORES}
 # Coverage limits cap broad generalization/diversity claims even with perfect exact labels.
 scores['generalization_value']=min(scores['generalization_value'],3)
 if decision=='ACCEPT':
  reason='Both final distinct-execution reviewers ACCEPT this exact record; oracle evidence and adjudicator typed prompt/target reexecution agree. Cross-release source indices and exact semantic-task identities do not overlap accepted pilots; family splits remain intact. Low coverage/mechanism/proxy limitations are attached. Acceptance is for bounded original synthetic behavior and does not establish new model efficacy or broad OOD mastery.'
 else:
  reason=f'Final evidence requires {decision}. Expert decision {e["decision"]} and adversarial decision {a["decision"]} are preserved alongside exact oracle/payload checks and issues; unresolved serious evidence prevents acceptance. Corrections require fresh hashes and final reviews.'
 receipts.append(dict(receipt_id='adj-'+BATCH+'-'+dig([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores=scores,critical_defects=defects,issues=issues,decision=decision,confidence=0.95 if decision=='ACCEPT' else 0.98,reason=reason,evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=e['decision']!=a['decision'],oracle_receipt_file=f'reviews/oracle/{BATCH}.jsonl',oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],oracle_pass=oracle_pass,adjudicator_execution=ACTOR,typed_record_hash_matches=True,typed_prompt_payload_matches=prompt_matches,typed_target_oracle_recheck=target_matches,family_split_consistent=True,cross_release_source_index_overlap=0,cross_release_exact_task_overlap=0,source_family_key=row['source_family_key'],source_family_independence_unit=row['semantic_family_id'],scope_audit='reviews/adjudication/expansion-v1_scope_review.json'),review_method='Actual independent final adjudication after complete final reviewers, exact typed identity/target recheck and cross-release scope/lineage audit.',scope_limitations=['1860 renderings represent450 source families,150 per track; mechanism classes repeat.','Exposed development; no final sealed instrument created/inspected.','Same underlying model across agents leaves correlated-error risk.','Earlier architect wrote specifications only and produced no training candidates.']))
counts=dict(collections.Counter(r['decision'] for r in receipts));disagreements=sum(r['evidence']['disagreement_preserved'] for r in receipts)
assert len(receipts)==1860
assert metadata_defects=={'branched_resource_graph':26,'multi_stack_goal_composition':14,'initially_unreachable':22,'equal_length_routes':18}
assert counts=={'ACCEPT':1780,'REVISE':80} and disagreements==14
assert all(review_snapshots[role]==sha(p) for role,p in rpaths.items()),'Final reviewer receipts changed during adjudication'
atomic(OUT/f'{BATCH}.jsonl',''.join(canon(r)+'\n' for r in receipts))
extra=[advp,scopep,ROOT/f'audits/leakage/{BATCH}_cross_release.json',ROOT/f'audits/leakage/{BATCH}.json',ROOT/f'audits/diversity/{BATCH}.json',ROOT/f'audits/diversity/{BATCH}-mechanisms-and-lineage.json',ROOT/f'audits/shortcut_detection/{BATCH}.json',ROOT/f'audits/shortcut_detection/{BATCH}-mechanism-tags.json',ROOT/'schemas/SEMANTICS.md',ROOT/'oracles/independent.py',ROOT/'research/ARCHITECT_EXECUTION.json']
# Snapshot all completed expert batch evidence without reading producer implementations.
extra += sorted(set((ROOT/'reviews/expert').glob('*expansion_v1*'))|set((ROOT/'reviews/expert').glob('expansion-v1*')))
inputs=paths+list(rpaths.values())+extra
execution=dict(schema_version='1.0.0',reviewer_execution_id=ACTOR,role='independent Quality Adjudicator',batch=BATCH,completed_at_utc=datetime.now(timezone.utc).isoformat(),records=1860,independent_source_families=450,decision_counts=counts,track_decisions={t:dict(collections.Counter(r['decision'] for r in receipts if r['sample_id'].startswith(t))) for t in ['RCM-001','RGM-001','RVM-001']},reviewer_disagreements_preserved=disagreements,oracle_recheck_failures=oracle_failures,independently_confirmed_metadata_defects=dict(metadata_defects),independent_metadata_method='Direct outdegree count, oracle feasible initial plan, and independently authored iterative equal-length path multiplicity DP across all feasible horizons.',scope_interpretation_correction='Preliminary audit described nonbranching worlds as controls under a generator-family tag; final review establishes that the actual skill tag is factually false and must be revised.',all_final_receipts_complete=True,typed_hash_and_payload_checks=True,cross_release_source_and_exact_task_overlap=0,role_reuse_disclosure='Actual /root/research_architect previously authored research specifications only; distinct from /root production and /root/expert_reviewer and /root/adversarial_auditor reviewers. Same underlying model error correlation remains.',scope_findings=scope['tracks'],inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()},output_sha256={f'reviews/adjudication/{BATCH}.jsonl':sha(OUT/f'{BATCH}.jsonl')},producer_or_release_files_modified=False,previous_reviews_preserved=True)
atomic(OUT/f'execution-{BATCH}.json',json.dumps(execution,indent=2)+'\n')
report=f'''# Expansion-v1 final adjudication

Execution `/root/research_architect`: {json.dumps(counts)} across 1,860 renderings and 450 fresh source families (150 per P0 track). Both final expert and final adversarial receipts were complete before any decision was written. All {disagreements} expert ACCEPT/adversarial REVISE disagreements are preserved; the reviewers agree REVISE on the other 66 metadata defects. RCM: 1,200 ACCEPT; RGM: 320 ACCEPT/40 REVISE; RVM: 260 ACCEPT/40 REVISE. Exact hashes, typed student payloads and independent-oracle target reexecution were checked for every record.

All 80 revisions concern factual competency metadata: 26 nonbranching graph rows tagged branched, 14 block rows labeled multi-stack despite having one initial TABLE root and fewer than two explicit goal TABLE roots, 22 reachable critic rows tagged initially_unreachable, and 18 critic rows lacking two distinct feasible equal-length terminal routes. I independently checked graph outdegrees, block initial/explicit-goal roots, feasible initial repair plans, and an iterative path multiplicity dynamic program over every fuel-feasible move horizon. Correct target labels do not cure incorrect competency metadata. The preliminary scope report treated nonbranching worlds as controls under a generator-family label; final adjudication corrects that interpretation and treats the actual skill tag as false. Controls remain useful once accurately labeled. New metadata versions need fresh exact-hash final reviews. The adversarial initial all-ACCEPT checkpoint is preserved separately; its later independently recomputed tag challenge is the final receipt version used here.

Source indices30–179 are disjoint from accepted pilot indices0–29, with no exact task overlap even after irrelevant causal annotation removal. Every source family's variants retain one parent/split. Prior review/releases remain untouched. Existing source families versus mechanism classes are reported separately:150 worlds do not equal150 independently new mechanisms.

The completed adversarial audit reports 214 current training logical variants sharing AST motifs with previously exposed development. Thus exact-task/source-family separation does not establish prior-development mechanism isolation. Development is already exposed, and no sealed OOD or independent mechanism generalization claim is made.

The preliminary scope audit remains the detailed evidence. Causal nested control is meaningful in14 development families and redundant in16; wrapper removal changes25 of60 twin rule traces and21 final states, so diversity must distinguish active combinations from controls. Relational data has90 graph families (77 with actual branching) and60 block families (40 full/20 partial goals), with both development goal labels. Critic worlds vary while scoring objectives and familiar corruption families persist. All difficulty labels remain Levels1–4; no broad composition or sealed OOD claim is inferred.

Individual receipts retain both reviewer issues and adjudicator scope limits. Target correctness supports these bounded finite tasks; native learning, assistance removal, retention and broad intelligence remain future authorized-study questions. Same underlying model family can yield correlated errors even across actual independent executions. This adjudicator earlier wrote curriculum/evidence specifications only and did not generate these candidates. Only adjudication files were written.
'''
atomic(OUT/'ADJUDICATION_EXPANSION_V1.md',report)
print(json.dumps({k:execution[k] for k in ['records','independent_source_families','decision_counts','track_decisions','reviewer_disagreements_preserved','output_sha256']},indent=2))
