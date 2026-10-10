"""Exact-record adjudication. Writes only adjudication evidence; creates no data or release."""
from pathlib import Path
import collections, hashlib, json, os, sys
from datetime import datetime, timezone

FACTORY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(FACTORY))
from oracles.independent import solve
ACTOR='/root/research_architect'
BATCH='pilot-v2'
OUT=FACTORY/'reviews/adjudication'
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def atomic(p,text):
    q=p.with_name(p.name+'.tmp')
    with q.open('w') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(q,p)

candidate_paths=[FACTORY/f'data/{t}/candidate_renderings/{BATCH}.jsonl' for t in ['RCM-001','RGM-001','RVM-001']]
review_paths={r:FACTORY/f'reviews/{r}/{BATCH}.jsonl' for r in ['expert','adversarial','oracle']}
rows=[r for p in candidate_paths for r in read(p)]
assert len(rows)==360 and len({r['sample_id'] for r in rows})==360
reviews={role:{r['sample_id']:r for r in read(path)} for role,path in review_paths.items()}
assert all(len(rs)==360 for rs in reviews.values())
assert all(set(rs)=={r['sample_id'] for r in rows} for rs in reviews.values())
required_scores=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
dev=[r for r in rows if r['track_id']=='RGM-001' and r['split']=='dev_challenge']
assert len(dev)==12 and len({r['semantic_family_id'] for r in dev})==6
assert all(r['verified_target']['reaches_goal'] is False for r in dev)
collapse_ids={r['sample_id'] for r in dev}

receipts=[]
for row in rows:
 sid=row['sample_id'];h=row['record_hash']
 assert digest({k:v for k,v in row.items() if k!='record_hash'})==h
 e,a,o=[reviews[x][sid] for x in ['expert','adversarial','oracle']]
 assert all(r['record_hash']==h for r in [e,a,o])
 assert e['reviewer_execution_id']=='/root/expert_reviewer'
 assert a['reviewer_execution_id']=='/root/adversarial_auditor'
 assert e['reviewer_execution_id']!=a['reviewer_execution_id']!=ACTOR
 assert o['oracle_author_execution']=='/root/formal_oracle' and o['pass'] is True and not o['errors']
 assert e['decision']=='ACCEPT' and a['decision'] in ['ACCEPT','REVISE']
 assert not e['critical_defects'] and not a['critical_defects']
 payload=json.loads(row['rendered_input'].split('\nTask JSON:\n',1)[1])
 assert payload==row['semantic_task']
 result=solve(payload)
 expected={'after':result['after']} if row['target_format']=='answer_only' else result
 assert expected==row['verified_target']
 issues=[]
 for role,review in [('expert',e),('adversarial',a)]:
  for issue in review.get('issues',[]):
   issues.append(dict(issue,source_role=role,source_receipt_id=review['receipt_id']))
 disagreement=e['decision']!=a['decision']
 if sid in collapse_ids:
  assert a['decision']=='REVISE'
  decision='REVISE'
  reason=('Independent target correctness is supported, but the final adversarial reviewer identifies a serious dataset-level development-goal shortcut: all 12 renderings from six relational dev families have reaches_goal=false. Expert ACCEPT is preserved as a disagreement; correctness alone does not satisfy shortcut/diversity gates. Add matched valid reaching and non-reaching plans within the same held-out source families and obtain fresh exact-hash reviews. This v2 record is withheld from acceptance pending correction and review.')
 else:
  assert a['decision']=='ACCEPT'
  assert not any(i.get('severity') in ['high','critical'] for i in issues)
  decision='ACCEPT'
  reason=('The final blind expert and final adversarial reviewer both ACCEPT this exact record; independently authored oracle evidence passes. The adjudicator rechecked record identity, student-visible payload equivalence and oracle target equality. Remaining low-severity coverage/difficulty/diagnostic limitations are preserved and restrict claims, but do not establish an incorrect target or unresolved serious record defect. Acceptance is for this bounded original synthetic training/development example, not broad OOD performance or learning efficacy.')
 scores={k:min(e['scores'][k],a['scores'][k]) for k in required_scores}
 receipt=dict(receipt_id='adj-pilot-v2-'+digest([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores=scores,critical_defects=[],issues=issues,decision=decision,confidence=0.98 if decision=='REVISE' else 0.95,reason=reason,evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=disagreement,oracle_receipt_file='reviews/oracle/pilot-v2.jsonl',oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],oracle_pass=o['pass'],adjudicator_recheck_execution=ACTOR,adjudicator_target_recheck=True,student_visible_payload_matches=True,canonical_record_hash_matches=True,adjudicator_oracle_source_hash=sha(FACTORY/'oracles/independent.py'),development_label_distribution={'relational_dev_renderings':12,'relational_dev_families':6,'reaches_goal_true':0,'reaches_goal_false':12} if decision=='REVISE' else None),review_method='Independent adjudication of final distinct-execution receipts, low/high severity evidence, current public contract and exact-record recheck; not a blind initial reviewer.',scope_limitations=['Same underlying model family across independent executions can yield correlated errors.','Architect role previously wrote research specifications only and generated or approved no samples.','No approval is issued for pilot-v3 or extended-pilot-v2; missing final reviews cannot be inferred from oracle passes.'])
 receipts.append(receipt)
counts=dict(collections.Counter(r['decision'] for r in receipts))
assert counts=={'ACCEPT':348,'REVISE':12}
assert sum(r['evidence']['disagreement_preserved'] for r in receipts)==12
text=''.join(canonical(r)+'\n' for r in receipts)
atomic(OUT/'pilot-v2.jsonl',text)
input_paths=candidate_paths+list(review_paths.values())+[
 FACTORY/'schemas/SEMANTICS.md',FACTORY/'CURRICULUM.md',
 FACTORY/'reviews/expert/semantic_inspection_v2.md',
 FACTORY/'reviews/expert/verification_summary_v2.json',
 FACTORY/'reviews/adversarial/execution-pilot-v2.json',
 FACTORY/'reviews/oracle/formal_oracle_execution.json',
 FACTORY/'oracles/independent.py',
 FACTORY/'research/ARCHITECT_EXECUTION.json']
execution=dict(schema_version='1.0.0',role='independent Quality Adjudicator',reviewer_execution_id=ACTOR,completed_at_utc=datetime.now(timezone.utc).isoformat(),batch=BATCH,records=360,decision_counts=counts,track_decisions={t:dict(collections.Counter(r['decision'] for r in receipts if r['sample_id'].startswith(t))) for t in ['RCM-001','RGM-001','RVM-001']},reviewer_disagreements_preserved=12,independence=dict(actual_execution_identity=ACTOR,previous_role='Dataset Research Architect',prior_actions='Research source review and curriculum specifications only; no sample generation, expert review, adversarial review or approval.',producer_identities=['/root','/root/relational_critic_producer'],expert_identity='/root/expert_reviewer',adversarial_identity='/root/adversarial_auditor',oracle_author_identity='/root/formal_oracle',distinct_from_producers_and_reviewers=True,blindness='Adjudication intentionally reads final reviewer verdicts after independent first judgments; this is not a blind expert review.',correlated_error_risk='All agents share the underlying model family. Distinct execution/context does not ensure statistically independent errors.'),checks={'all_exact_record_hashes':True,'all_review_hashes_match':True,'all_oracle_passes':True,'all_student_payloads_match':True,'all_adjudicator_oracle_rechecks_pass':True,'all_12_rgm_dev_labels_false':True},inputs_sha256={str(p.relative_to(FACTORY)):sha(p) for p in input_paths},output_sha256={'reviews/adjudication/pilot-v2.jsonl':sha(OUT/'pilot-v2.jsonl')},excluded_batches=['pilot-v3','extended-pilot-v2'],excluded_batch_policy='Independent final receipts were not complete for these batches at task assignment; no decision or acceptance is made here.',writes=['reviews/adjudication/pilot-v2.jsonl','reviews/adjudication/execution-pilot-v2.json','reviews/adjudication/ADJUDICATION_PILOT_V2.md','reviews/adjudication/adjudicate_pilot_v2.py'],producer_or_release_mutations=False,protected_instruments_opened=False)
atomic(OUT/'execution-pilot-v2.json',json.dumps(execution,indent=2)+'\n')
report='''# Independent pilot-v2 adjudication

Actual execution identity: `/root/research_architect`. This same execution earlier served as Dataset Research Architect, writing only evidence inventories and curriculum specifications. It generated no semantic parents, renderings, targets or reviewer verdicts. It now serves as Quality Adjudicator, distinct from the causal producer `/root`, relational/critic producer `/root/relational_critic_producer`, expert `/root/expert_reviewer`, adversarial reviewer `/root/adversarial_auditor` and oracle author `/root/formal_oracle`. This role reuse is explicit; no new identity or blind-review claim is invented. All executions share the underlying model family and may share semantic mistakes.

All 360 v2 candidate identities, reviewer hashes and independent-oracle passes were checked. The adjudicator parsed every student-visible Task JSON, checked equality with the canonical task, and reran the independently authored oracle under this actual execution. The root's original oracle-run identity is retained in per-record evidence; using that oracle is not a claim that this adjudicator independently authored it. Exact source/input/output hashes are recorded in `execution-pilot-v2.json`.

The decisions are 348 ACCEPT and 12 REVISE: causal 240 ACCEPT; relational 48 ACCEPT and 12 REVISE; critic 60 ACCEPT. Acceptance means exact-record dataset quality approval within declared semantics and known limitations, not evidence that training improves a model.

The 12 relational development records retain the expert ACCEPT / adversarial REVISE disagreement. Their targets are correct, yet all six development families (12 renderings) have `reaches_goal=false`; an always-false prediction scores 100% on that field. This high-severity shortcut blocks acceptance. They require matched reaching/non-reaching plans at the same held-out names/sizes, grouped under the same family, and fresh exact-hash independent review. No v2 receipt is transferred to corrected v3 bytes.

Low-severity limitations remain visible in every affected receipt: partial visible-AST/no-op correlation, answer-only copy controls, repeated versus truly dependent transitions, conservative replay/optimal-plan difficulty distinctions, narrow critic corruption producers, and shared templates across exposed magnitude/vocabulary development challenges. These constrain coverage and future claims. Surface/number changes are not structural OOD evidence; correctness is not unrestricted semantic generalization.

No decisions are made for relational pilot-v3 or the seven-track extended-pilot-v2. Oracle passes and uniqueness do not replace missing final independent expert/adversarial receipts. No producer, accepted-data or release file was written by this execution; only adjudication artifacts were produced. Existing research test instruments and model artifacts were not opened.
'''
atomic(OUT/'ADJUDICATION_PILOT_V2.md',report)
print(json.dumps({'records':len(receipts),'decisions':counts,'track_decisions':execution['track_decisions'],'disagreements_preserved':12,'output_sha256':execution['output_sha256']},indent=2))
