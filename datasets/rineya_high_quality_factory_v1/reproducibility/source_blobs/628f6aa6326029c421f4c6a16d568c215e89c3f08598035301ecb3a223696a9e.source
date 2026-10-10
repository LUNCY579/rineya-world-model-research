"""Independent final adjudication of corrected finite routing/retrieval variants.
Writes only reviews/adjudication; no producer or release imports/mutations.
"""
from pathlib import Path
from datetime import datetime, timezone
import collections, hashlib, json, os, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
ACTOR='/root/research_architect';BATCH='extended-pilot-v3';OUT=ROOT/'reviews/adjudication'
SCORES=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def dig(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def atomic(p,s):
 q=p.with_name(p.name+'.tmp')
 with q.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
paths=[ROOT/f'data/{t}/candidate_renderings/{BATCH}.jsonl' for t in ['RMM-001','REM-001']]
rows=[r for p in paths for r in read(p)]
assert len(rows)==196 and len({r['sample_id'] for r in rows})==196
rp={k:ROOT/f'reviews/{k}/{BATCH}.jsonl' for k in ['expert','adversarial','oracle']}
rv={k:{r['sample_id']:r for r in read(p)} for k,p in rp.items()}
assert all(len(v)==196 and set(v)=={r['sample_id'] for r in rows} for v in rv.values())
advp=ROOT/f'reviews/adversarial/execution-{BATCH}.json';adv=json.loads(advp.read_text())
assert adv['records']==196 and adv['decisions']=={'ACCEPT':196}
assert all(adv['candidate_file_hashes'][str(p.relative_to(ROOT))]==sha(p) for p in paths)
expertp=ROOT/'reviews/expert/verification_summary_extended_v3.json';expert=json.loads(expertp.read_text())
assert expert['records']==196 and expert['decisions']=={'ACCEPT':196}
assert expert['files']['reviews/expert/extended-pilot-v3.jsonl']==sha(rp['expert'])
assert sha(ROOT/'schemas/EXTENDED_SEMANTICS.md')==adv['contract_hash']
families=collections.defaultdict(list)
for r in rows:families[r['semantic_family_id']].append(r)
assert len(families)==24
family_evidence={};track_stats={}
for f,g in families.items():
 assert len({r['semantic_parent_id'] for r in g})==1 and len({r['split'] for r in g})==1
 forward=[r for r in g if r['renderer']=='forward'];reverse=[r for r in g if r['renderer']=='reverse']
 assert len(forward)==len(reverse)
 byrotation=collections.defaultdict(list)
 for r in g:byrotation[r['sample_id'].rsplit('-',2)[-2]].append(r)
 assert all(len(x)==2 and canon(x[0]['semantic_task'])==canon(x[1]['semantic_task']) for x in byrotation.values())
 track=g[0]['track_id'];field='requests' if track=='RMM-001' else 'queries';idkey='request_id' if track=='RMM-001' else 'query_id'
 width=len(forward[0]['semantic_task'][field]);assert len(forward)==width
 ids={x[idkey] for x in forward[0]['semantic_task'][field]}
 signatures=collections.defaultdict(lambda:dict(positions=[],ids=[]))
 statuses=collections.defaultdict(list)
 for r in forward:
  entries=r['semantic_task'][field]
  assert len(entries)==width and {x[idkey] for x in entries}==ids
  labels=r['verified_target']['routes' if track=='RMM-001' else 'results']
  for pos,x in enumerate(entries):
   signature=canon({k:v for k,v in x.items() if k!=idkey})
   signatures[signature]['positions'].append(pos);signatures[signature]['ids'].append(x[idkey])
   label=labels[pos]['selected_module'] is None if track=='RMM-001' else labels[pos]['status']
   statuses[x[idkey]].append(label)
 assert len(signatures)==width
 assert all(sorted(v['positions'])==list(range(width)) and set(v['ids'])==ids and len(v['ids'])==width for v in signatures.values())
 if track=='RMM-001':
  assert all(set(v)=={False,True} for v in statuses.values())
  for r in forward:
   assert all(r['verified_target']['memory'][key]==value for key,value in r['semantic_task']['memory'].items() if key in r['semantic_task']['protected'])
   assert all(a['eligible_modules']==[] and a['blocked_writes']==[] for a in r['verified_target']['routes'] if a['selected_module'] is None)
 else:
  # Query results must follow semantic (key,as_of), not assigned ID/position.
  query_maps=[]
  for r in forward:
   query_maps.append({(x['key'],x['as_of']):canon({k:v for k,v in y.items() if k!='query_id'}) for x,y in zip(r['semantic_task']['queries'],r['verified_target']['results'])})
  assert all(m==query_maps[0] for m in query_maps)
  assert all({'absent','present'}<=set(v) for v in statuses.values())
 family_evidence[f]=dict(track=track,source_parent_count=1,renderings=len(g),semantic_rotations=width,split=g[0]['split'],every_signature_crosses_every_id_and_position=True,labels_by_id={k:dict(collections.Counter(str(x) for x in v)) for k,v in statuses.items()},semantic_rotations_are_matched_variants_not_independent_parents=True)
for track in ['RMM-001','REM-001']:
 g=[r for r in rows if r['track_id']==track];stat=collections.defaultdict(collections.Counter)
 for r in g:
  if r['renderer']!='forward':continue
  for pos,x in enumerate(r['verified_target']['routes' if track=='RMM-001' else 'results']):
   stat[x['request_id' if track=='RMM-001' else 'query_id']][str(x['selected_module'] is None) if track=='RMM-001' else x['status']]+=1
 track_stats[track]=dict(renderings=len(g),independent_families=len({r['semantic_family_id'] for r in g}),labels_by_id={k:dict(v) for k,v in stat.items()})
receipts=[]
for row in rows:
 sid=row['sample_id'];h=row['record_hash'];e,a,o=[rv[k][sid] for k in ['expert','adversarial','oracle']]
 assert dig({k:v for k,v in row.items() if k!='record_hash'})==h
 assert all(r['record_hash']==h for r in [e,a,o])
 assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor'
 assert e['decision']==a['decision']=='ACCEPT' and not e['critical_defects'] and not a['critical_defects']
 assert o['oracle_author_execution']=='/root/formal_oracle' and o['pass'] is True and not o['errors']
 payload=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1]);assert canon(payload)==canon(row['semantic_task'])
 assert canon(solve(payload))==canon(row['verified_target'])
 issues=[dict(i,source_role=k,source_receipt_id=r['receipt_id']) for k,r in [('expert',e),('adversarial',a)] for i in r.get('issues',[])]
 assert not any(i.get('severity') in ['high','critical'] for i in issues)
 reason='Both final independent reviewers ACCEPT this exact corrected record; target and hash binding pass independent-oracle evidence and adjudicator typed reexecution. Every semantic request/query within its source family now occupies every ID and position once across rotations; outcome classes cross those nuisance features, removing the prior deterministic ID/position shortcut. Variants remain one source parent/family/split. Remaining narrow fixtures, modest diversity and finite proxy limitations stay attached and restrict generalization claims.'
 receipts.append(dict(receipt_id='adj-'+BATCH+'-'+dig([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores={k:min(e['scores'][k],a['scores'][k]) for k in SCORES},critical_defects=[],issues=issues,decision='ACCEPT',confidence=0.95,reason=reason,evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=False,oracle_receipt_file=f'reviews/oracle/{BATCH}.jsonl',oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],oracle_pass=True,adjudicator_execution=ACTOR,typed_canonical_record_hash_matches=True,typed_prompt_payload_matches=True,typed_target_oracle_recheck=True,family_crossing=family_evidence[row['semantic_family_id']]),review_method='Actual independent final adjudication of distinct-execution final receipts, typed target recheck and per-family nuisance/outcome crossing.',scope_limitations=['24 underlying independent families;196 renderings do not represent196 independent parents.','Same underlying model family carries correlated-error risk.','Explicit routing/protection is not empirical competence/retention; versioned retrieval is not learned persistent memory.','Earlier v2 REVISE receipts and disagreements remain historical evidence; new acceptance applies only to corrected exact hashes.']))
assert len(receipts)==196
atomic(OUT/f'{BATCH}.jsonl',''.join(canon(r)+'\n' for r in receipts))
inputs=paths+list(rp.values())+[advp,expertp,ROOT/'reviews/expert/semantic_inspection_extended_v3.md',ROOT/'schemas/EXTENDED_SEMANTICS.md',ROOT/'oracles/independent.py',ROOT/'oracles/extended_independent.py',ROOT/'research/ARCHITECT_EXECUTION.json']
execution=dict(reviewer_execution_id=ACTOR,role='independent Quality Adjudicator',batch=BATCH,completed_at_utc=datetime.now(timezone.utc).isoformat(),records=196,decision_counts={'ACCEPT':196},track_statistics=track_stats,independent_families=24,role_reuse_disclosure='Actual /root/research_architect previously wrote curriculum/evidence specifications only; no sample generation, expert review or red-team role. Distinct from producers and final reviewers; same underlying model risk.',family_crossing_checks=family_evidence,all_hash_payload_target_checks_pass=True,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},output_sha256={f'reviews/adjudication/{BATCH}.jsonl':sha(OUT/f'{BATCH}.jsonl')},previous_versions_preserved=True,producer_or_release_files_modified=False)
atomic(OUT/f'execution-{BATCH}.json',json.dumps(execution,indent=2)+'\n')
report='''# Corrected routing and retrieval adjudication

Actual execution `/root/research_architect`:196 ACCEPT, comprising76 RMM renderings and120 REM renderings. The independent count remains12 source families per track,24 total. Within-family semantic rotations and forward/reverse renderings are matched variants, not independent-parent expansion. Earlier v2 REVISE receipts and48 reviewer disagreements remain untouched; these approvals bind only to new v3 hashes.

Both final reviewers ACCEPT all196 records. Their exact record hashes, candidate-file snapshots and oracle passes match. The adjudicator parsed every complete student-visible task, used type-sensitive canonical equality, and reran the independently authored oracle under its actual identity. No producer reference was imported and no producer/release file was changed.

Per-family audit confirms every underlying request/query signature ignoring its assigned ID occupies every ID and every input position exactly once. Each family retains one parent ID and split. Routing families have three rotations (four for the two larger development families); retrieval families have five. Every routing ID/position now sees supported and unsupported requests; every retrieval ID/position sees absent and present cases, with deleted cases where the family's ledger supplies a tombstone. Across forward records, each q0–q4 has24 absent,31 present and5 deleted results. r0–r2 each have12 abstentions and26 supported results; r3, available only in larger dev families, has2 abstentions and6 supported results. This removes the previous deterministic outcome mapping while retaining legitimate class imbalance.

Request order can change final unprotected writes and was reexecuted rather than treated as meaning-preserving. Protected memory remains initial, abstaining requests attempt no writes, and retrieval results indexed by (key,as_of) stay identical across query rotations. IDs still name displayed positions, so nuisance-ID/position independence itself is not claimed; outcomes now cross both.

Remaining limits: small repeated capability/write/ledger fixtures, moderate shortcut resistance, and retrieval development history whose added at19 event is not queried before supersession. An as_of19 case would add a decisive dependency. Declared routing/protection is not measured retention or learned routing; visible versioned retrieval is not a persistent learned memory system. Same underlying model family across agents remains a correlated-error risk.

The architect/adjudicator dual role remains auditable: its prior work was research specifications only, with no generation or expert/adversarial review. This is final adjudication after final reviewers, not an invented blind review identity.
'''
atomic(OUT/'ADJUDICATION_EXTENDED_V3.md',report)
print(json.dumps({k:execution[k] for k in ['records','decision_counts','independent_families','track_statistics','output_sha256']},indent=2))
