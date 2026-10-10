"""Final adjudication of corrected relational and extended finite-proxy pilots.
Only reviews/adjudication files are written. Never imports producer code.
"""
from pathlib import Path
import collections, hashlib, json, os, sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
ACTOR='/root/research_architect'
OUT=ROOT/'reviews/adjudication'
SCORES=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def atomic(p,text):
 q=p.with_name(p.name+'.tmp')
 with q.open('w') as f: f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(q,p)

def adjudicate(batch,tracks,count):
 paths=[ROOT/f'data/{t}/candidate_renderings/{batch}.jsonl' for t in tracks]
 review_paths={r:ROOT/f'reviews/{r}/{batch}.jsonl' for r in ['expert','adversarial','oracle']}
 rows=[r for p in paths for r in read(p)]
 assert len(rows)==count and len({r['sample_id'] for r in rows})==count
 reviews={role:{r['sample_id']:r for r in read(p)} for role,p in review_paths.items()}
 assert all(len(rs)==count and set(rs)=={r['sample_id'] for r in rows} for rs in reviews.values())
 adv_execution_path=ROOT/f'reviews/adversarial/execution-{batch}.json'
 adv_execution=json.loads(adv_execution_path.read_text())
 assert all(adv_execution['candidate_file_hashes'][str(p.relative_to(ROOT))]==sha(p) for p in paths)
 families=collections.defaultdict(list)
 for r in rows:families[r['semantic_family_id']].append(r)
 assert all(len({r['split'] for r in group})==1 for group in families.values())
 special={}
 if batch=='pilot-v3':
  dev_families={f:g for f,g in families.items() if g[0]['split']=='dev_challenge'}
  assert len(dev_families)==6
  for f,g in dev_families.items():
   assert len(g)==4 and collections.Counter(r['verified_target']['reaches_goal'] for r in g)=={False:2,True:2}
   assert len({r['semantic_parent_id'] for r in g})==1
  special=dict(relational_dev_families={f:dict(records=len(g),reaches_goal_false=2,reaches_goal_true=2,shared_parent_and_split=True,optimal_horizons=sorted({len(r['verified_target']['shortest_plan']) for r in g if r['verified_target']['shortest_plan'] is not None})) for f,g in dev_families.items()},source_families=len(families),task_variants=len({canon(r['semantic_task']) for r in rows}),custody_note='36 task variants across30 source families; matched dev variants share parent/family IDs and split. Do not count them as36 independent parents.')
 else:
  routing_id=collections.defaultdict(set);routing_position=collections.defaultdict(set)
  memory_id=collections.defaultdict(set);memory_position=collections.defaultdict(set)
  for r in rows:
   if r['renderer']!='forward':continue
   if r['track_id']=='RMM-001':
    for i,a in enumerate(r['verified_target']['routes']):
     value=a['selected_module'] is None;routing_id[a['request_id']].add(value);routing_position[i].add(value)
   if r['track_id']=='REM-001':
    for i,a in enumerate(r['verified_target']['results']):
     memory_id[a['query_id']].add(a['status']);memory_position[i].add(a['status'])
  assert routing_id['r1']=={True} and all(v=={False} for k,v in routing_id.items() if k!='r1')
  assert memory_id['q0']==memory_id['q4']=={'absent'} and memory_id['q1']==memory_id['q3']=={'present'}
  special=dict(routing_abstention_by_id={k:sorted(v) for k,v in routing_id.items()},routing_abstention_by_position={str(k):sorted(v) for k,v in routing_position.items()},memory_status_by_id={k:sorted(v) for k,v in memory_id.items()},memory_status_by_position={str(k):sorted(v) for k,v in memory_position.items()})
 receipts=[]
 for row in rows:
  sid=row['sample_id'];h=row['record_hash'];e,a,o=[reviews[x][sid] for x in ['expert','adversarial','oracle']]
  assert digest({k:v for k,v in row.items() if k!='record_hash'})==h
  assert all(r['record_hash']==h for r in [e,a,o])
  assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor'
  assert o['oracle_author_execution']=='/root/formal_oracle' and o['pass'] is True and not o['errors']
  assert e['decision']=='ACCEPT' and a['decision'] in ['ACCEPT','REVISE'] and not e['critical_defects'] and not a['critical_defects']
  payload=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1])
  assert canon(payload)==canon(row['semantic_task'])
  assert canon(solve(payload))==canon(row['verified_target'])
  issues=[dict(i,source_role=role,source_receipt_id=r['receipt_id']) for role,r in [('expert',e),('adversarial',a)] for i in r.get('issues',[])]
  if batch=='extended-pilot-v2' and row['track_id'] in ['RMM-001','REM-001']:
   assert a['decision']=='REVISE' and any(i.get('severity')=='high' for i in a['issues'])
   decision='REVISE'
   defect='request ID and position perfectly determine abstention for the recurring request scaffold' if row['track_id']=='RMM-001' else 'query IDs and fixed positions determine several retrieval-status classes without consulting the event history'
   reason=f'Gold is correct, but final adversarial evidence shows a serious shortcut: {defect}. Expert ACCEPT is retained as a disagreement. Correctness alone does not satisfy quality gates. Cross IDs/positions with alternative outcomes under matched same-family tasks and obtain fresh exact-hash reviews before acceptance; this v2 record remains withheld.'
  else:
   assert a['decision']=='ACCEPT' and not any(i.get('severity') in ['high','critical'] for i in issues)
   decision='ACCEPT'
   reason='Final independent expert and adversarial reviewers both ACCEPT this exact current record, and independently authored oracle receipts pass. Adjudicator verified typed canonical identity, complete student-visible payload and target equality by rerunning the oracle. Low scope/diversity/difficulty limitations remain attached. This is acceptance for the stated finite synthetic behavior, not a model architecture, efficacy or unrestricted generalization claim.'
   if batch=='pilot-v3':reason+=' The former development-goal collapse is repaired by paired reaching/non-reaching variants in every dev source family; family custody remains intact.'
  receipts.append(dict(receipt_id='adj-'+batch+'-'+digest([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores={k:min(e['scores'][k],a['scores'][k]) for k in SCORES},critical_defects=[],issues=issues,decision=decision,confidence=0.98 if decision=='REVISE' else 0.95,reason=reason,evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=e['decision']!=a['decision'],oracle_receipt_file=f'reviews/oracle/{batch}.jsonl',oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],oracle_pass=True,adjudicator_execution=ACTOR,typed_canonical_record_hash_matches=True,typed_prompt_payload_matches=True,typed_target_oracle_recheck=True,family_split_consistent=True,batch_issue_evidence=special if decision=='REVISE' or batch=='pilot-v3' else None),review_method='Actual independent adjudication after final reviewer judgments; exact typed hashes and oracle recheck, not a blind initial reviewer.',scope_limitations=['Earlier architect wrote specifications only and generated no samples.','Same underlying model family across agents carries correlated-error risk.','Extended-pilot-v3 remains unapproved here until fresh final reviews complete.']))
 counts=dict(collections.Counter(r['decision'] for r in receipts))
 assert counts==({'ACCEPT':72} if batch=='pilot-v3' else {'ACCEPT':120,'REVISE':48})
 atomic(OUT/f'{batch}.jsonl',''.join(canon(r)+'\n' for r in receipts))
 inputs=paths+list(review_paths.values())+[adv_execution_path,ROOT/'schemas/SEMANTICS.md',ROOT/'schemas/EXTENDED_SEMANTICS.md',ROOT/'oracles/independent.py',ROOT/'oracles/extended_independent.py',ROOT/'reviews/expert/semantic_inspection_v3_extended_v2.md',ROOT/'reviews/expert/verification_summary_v3_extended_v2.json',ROOT/'reviews/oracle/extended_formal_oracle_execution.json',ROOT/'research/ARCHITECT_EXECUTION.json',ROOT/'CURRICULUM.md']
 execution=dict(schema_version='1.0.0',reviewer_execution_id=ACTOR,role='independent Quality Adjudicator',completed_at_utc=datetime.now(timezone.utc).isoformat(),batch=batch,records=count,decision_counts=counts,track_decisions={t:dict(collections.Counter(r['decision'] for r in receipts if r['sample_id'].startswith(t))) for t in tracks},reviewer_disagreements_preserved=sum(r['evidence']['disagreement_preserved'] for r in receipts),source_families=len(families),batch_issue_evidence=special,independence='Same actual /root/research_architect execution previously wrote curriculum/evidence specifications only, no generation or reviewer decisions. Distinct from /root and /root/relational_critic_producer producers and /root/expert_reviewer and /root/adversarial_auditor reviewers; same underlying model risk disclosed.',inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},output_sha256={f'reviews/adjudication/{batch}.jsonl':sha(OUT/f'{batch}.jsonl')},no_producer_or_release_writes=True,excluded_batch='extended-pilot-v3',excluded_batch_status='Fresh final receipts not yet complete; no approval inferred.')
 atomic(OUT/f'execution-{batch}.json',json.dumps(execution,indent=2)+'\n')
 print(json.dumps({k:execution[k] for k in ['batch','records','decision_counts','track_decisions','source_families','reviewer_disagreements_preserved','output_sha256']},indent=2))
 return execution

rgm=adjudicate('pilot-v3',['RGM-001'],72)
ext=adjudicate('extended-pilot-v2',['RBM-001','RDP-001','REM-001','RJP-001','RMM-001','ROM-001','RSM-001'],168)
report='''# Corrected relational and extended pilot adjudication

Actual execution `/root/research_architect` completed these adjudications. Its earlier Dataset Research Architect role wrote evidence/curriculum specifications only; it neither produced samples nor acted as expert/red reviewer. The role reuse is disclosed, and same underlying model risk remains. Adjudication intentionally reads final verdicts after independent initial judgments; it is not a blind expert review.

Relational pilot-v3:72 ACCEPT across30 source families. There are36 semantic task variants, including six added matched development variants, and two field-order renderings. Every development source family has two reaching and two non-reaching renderings. All variants in each family retain one parent ID and split. Fresh shortest plans and structural difficulty are checked using the oracle; horizons are0,1,5 or6 for dev counterparts, with unreachable cases represented as null. This fixes v2's universal false goal label without inflating parent counts. Low conservative difficulty/repeated-motif limits remain attached.

Extended-pilot-v2:120 ACCEPT and48 REVISE. RSM/RBM/RJP/RDP/ROM each have24 accepted renderings from12 parents. RMM and REM each have24 REVISE receipts. Their exact gold is correct, but routing abstention is encoded by fixed request ID/position and retrieval status by fixed query ID/position. Expert ACCEPT versus adversarial REVISE is explicitly preserved on all48 records. Corrected IDs/positions must be crossed with alternative outcomes under same-family tasks, with fresh exact-hash independent reviews.

Every record was rehashed, every final expert/adversarial/oracle receipt identity matched, and every full student-visible payload was parsed and checked with type-sensitive canonical equality. The adjudicator reran the independently authored oracle under its actual identity; the original oracle runner and author identities remain distinct in evidence. Input/output SHA-256 snapshots are in each execution JSON. No producer or release bytes were modified.

These are bounded finite-task approvals. Supplied-procedure execution is not learned skill hierarchy; declared capability routing is not measured competence/retention; versioned retrieval is not learned persistent memory; possible-world filtering is not calibrated probabilistic inference; exact quotienting is not trained latent dynamics; minimum Hamming repair is not trained diffusion; explicit fictional observations/policies are not general human theory of mind. Shared templates, proxy scope and modest diversity limit stronger future claims.

Extended-pilot-v3 remains pending outside these decisions; current correctness/uniqueness cannot replace incomplete fresh independent final review.
'''
atomic(OUT/'ADJUDICATION_NEXT_PILOTS.md',report)
