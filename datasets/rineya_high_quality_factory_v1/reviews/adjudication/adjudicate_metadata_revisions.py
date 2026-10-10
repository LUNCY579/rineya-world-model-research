"""Fail-closed adjudication of fresh, immutable factual-metadata revisions.

No producer code imported; no samples or releases written. Parent assigns actual
/root/research_architect identity. Repeated worlds do not become new parents.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,sys,os,re
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/adjudication'
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
ACTOR='/root/research_architect'
SCORES=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity','generalization_value','shortcut_resistance','diversity','difficulty_quality']
CONFIG={'expansion-v2':('RVM-001',300,150,'expansion-v1'),'expansion-v3':('RGM-001',360,150,'expansion-v1'),'pilot-v4':('RVM-001',60,30,'pilot-v2')}
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def atomic(p,s):
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def equal_routes(world):
 adjacency=collections.defaultdict(list)
 for a,b,c in world['edges']:adjacency[a].append((b,c))
 frontier={(world['start'],0):1}
 for length in range(1,world['fuel']//min(c for a,b,c in world['edges'])+1):
  nxt=collections.defaultdict(int)
  for (node,spent),count in frontier.items():
   for dest,cost in adjacency[node]:
    if spent+cost<=world['fuel']:
     key=(dest,spent+cost);nxt[key]=min(2,nxt[key]+count)
  if sum(count for (node,spent),count in nxt.items() if node==world['goal'])>1:return True
  frontier=nxt
  if not frontier:break
 return False
def metadata_checks(row,result):
 t=row['semantic_task'];w=t['world'] if t['kind']=='critic' else t
 checks={};details={}
 degree=collections.Counter(a for a,b,c in w['edges']) if 'edges' in w else {}
 branched=bool(degree and max(degree.values())>1)
 for tag in row['skill_tags']:
  leaf=tag.split(':')[-1]
  if tag.startswith('branched_resource_graph'):checks['branched_resource_graph']=branched
  if leaf=='single_successor_resource_graph':checks[leaf]=not branched
  if leaf=='multiple_initial_table_stacks':checks[leaf]=sum(v=='TABLE' for v in t['on'].values())>=2
  if leaf=='multiple_explicit_goal_table_roots':checks[leaf]=sum(v=='TABLE' for v in t['goal_on'].values())>=2
  if leaf=='block_goal_composition':checks[leaf]=t['kind']=='blocks'
  if leaf in ['equal_length_goal_routes','no_equal_length_goal_routes']:
   equal=equal_routes(w);checks[leaf]=equal if leaf=='equal_length_goal_routes' else not equal
   details['has_distinct_fuel_feasible_equal_move_terminal_goal_routes']=equal
  if leaf in ['initial_goal_unreachable','initially_unreachable','initial_goal_reachable']:
   reachable=result['evaluations'][0]['repair_plan'] is not None
   checks[leaf]=reachable if leaf=='initial_goal_reachable' else not reachable
   details['initial_repair_plan']=result['evaluations'][0]['repair_plan']
  if leaf=='initial_goal':checks[leaf]=w['start']==w['goal']
  if leaf=='unreachable':checks[leaf]=result['shortest_plan'] is None
 if t['kind']=='blocks':details.update(initial_table_roots=sum(v=='TABLE' for v in t['on'].values()),explicit_goal_table_roots=sum(v=='TABLE' for v in t['goal_on'].values()))
 return checks,details
def run(batch):
 tid,count,nfamilies,prior=CONFIG[batch]
 saved=OUT/f'{batch}.jsonl';saved_execution=OUT/f'execution-{batch}.json'
 if saved.exists():
  assert saved_execution.exists(),'Existing adjudication lacks execution receipt; preserve and investigate'
  receipt=json.loads(saved_execution.read_text())
  assert receipt['reviewer_execution_id']==ACTOR and receipt['records']==count
  assert receipt['output_sha256'][str(saved.relative_to(ROOT))]==sha(saved)
  inputs=receipt['inputs_sha256']
  supplement=OUT/'metadata-revisions-contract-documentation-supplement.json'
  if supplement.exists():
   s=json.loads(supplement.read_text())
   if batch in s['batches']:
    entry=s['batches'][batch];assert entry['original_execution_sha256']==sha(saved_execution)
    inputs=entry['corrected_immutable_input_paths_sha256']
  assert all(sha(ROOT/p)==h for p,h in inputs.items()),'Historical inputs changed; preserve adjudication and issue a new batch'
  print(json.dumps({'batch':batch,'status':'PRESERVED_EXISTING_FINAL','decision_counts':receipt['decision_counts'],'output_sha256':receipt['output_sha256']}));return
 cp=ROOT/f'data/{tid}/candidate_renderings/{batch}.jsonl';oldp=ROOT/f'data/{tid}/candidate_renderings/{prior}.jsonl'
 rps={role:ROOT/f'reviews/{role}/{batch}.jsonl' for role in ['expert','adversarial','oracle']}
 reportp=ROOT/'reviews/expert'/('semantic_inspection_expansion_v2.md' if batch=='expansion-v2' else 'semantic_inspection_metadata_revisions.md')
 summaryp=ROOT/'reviews/expert'/('verification_summary_expansion_v2.json' if batch=='expansion-v2' else f'verification_summary_{batch}.json')
 advp=ROOT/f'reviews/adversarial/execution-{batch}.json'
 required=[cp,oldp,*rps.values(),reportp,summaryp,advp,ROOT/'schemas/SEMANTICS.md',ROOT/'schemas/COMPETENCY_TAGS.md',ROOT/'oracles/independent.py']
 if not all(p.exists() for p in required):
  print(json.dumps({'batch':batch,'status':'PENDING','missing':[str(p.relative_to(ROOT)) for p in required if not p.exists()]}));return
 snapshots={str(p.relative_to(ROOT)):sha(p) for p in required}
 rows=read(cp);old={r['sample_id']:r for r in read(oldp)};ids={r['sample_id'] for r in rows}
 assert len(rows)==len(ids)==count
 reviews={role:{r['sample_id']:r for r in read(p) if r['sample_id'] in ids} for role,p in rps.items()}
 assert all(len(v)==count and set(v)==ids for v in reviews.values())
 summary=json.loads(summaryp.read_text());adv=json.loads(advp.read_text())
 source_hash=summary['source_sha256'] if isinstance(summary['source_sha256'],str) else summary['source_sha256'][str(cp.relative_to(ROOT))]
 assert summary['rows']==count and source_hash==sha(cp)
 assert adv['candidate_file_hashes'][str(cp.relative_to(ROOT))]==sha(cp)
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_family_key']].append(r)
 assert len(groups)==nfamilies and all(len({r['split'] for r in g})==len({r['semantic_parent_id'] for r in g})==len({r['semantic_family_id'] for r in g})==1 for g in groups.values())
 output=[];tag_counts=collections.Counter();tag_changes=0;failures=[];differing_keys=collections.Counter()
 for row in rows:
  sid=row['sample_id'];h=row['record_hash'];previous_id=sid.replace(batch,prior);previous=old[previous_id]
  assert digest({k:v for k,v in row.items() if k!='record_hash'})==h
  assert digest({k:v for k,v in previous.items() if k!='record_hash'})==previous['record_hash']
  assert all(canon(row[k])==canon(previous[k]) for k in ['semantic_task','rendered_input','verified_target','split','structural_difficulty','target_format'])
  if previous.get('source_family_key'):assert row['source_family_key']==previous['source_family_key']
  else:
   index=re.search(r'-p(\d+)-',previous_id).group(1)
   assert row['source_family_key']==f'{tid}-seed20261009-p{index}'
  differing_keys.update(k for k in row if canon(row.get(k))!=canon(previous.get(k)))
  tag_changes+=row['skill_tags']!=previous['skill_tags']
  e,a,o=[reviews[role][sid] for role in ['expert','adversarial','oracle']]
  assert all(r['record_hash']==h for r in [e,a,o])
  assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor' and o['oracle_author_execution']=='/root/formal_oracle'
  assert e['review_phase']=='completed_fresh_metadata_revision_review'
  payload=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1]);result=solve(payload)
  matches=canon(payload)==canon(row['semantic_task']) and canon(result)==canon(row['verified_target']) and o['pass'] is True and not o['errors']
  checks,details=metadata_checks(row,result);tag_counts.update(checks)
  issues=[dict(i,source_role=role,source_receipt_id=r['receipt_id']) for role,r in [('expert',e),('adversarial',a)] for i in r.get('issues',[])]
  defects=[dict(d,source_role=role) if isinstance(d,dict) else dict(defect=d,source_role=role) for role,r in [('expert',e),('adversarial',a)] for d in r.get('critical_defects',[])]
  if not matches or defects or 'QUARANTINE' in [e['decision'],a['decision']]:decision='QUARANTINE'
  elif 'REJECT' in [e['decision'],a['decision']]:decision='REJECT'
  elif not all(checks.values()) or 'REVISE' in [e['decision'],a['decision']] or any(i.get('severity') in ['high','critical'] for i in issues):decision='REVISE'
  else:decision='ACCEPT'
  if not matches or not all(checks.values()):failures.append(sid)
  issues.append(dict(category='metadata_revision_not_new_data',severity='low',evidence='Semantic task, rendered input, gold, difficulty and split are byte-equivalent under typed canonical comparison to the prior batch. This correction contributes zero new source families relative to that batch.',fix='Supersede the prior source-family version and avoid counting both as independent training data.',source_role='independent_adjudicator'))
  issues.append(dict(category='bounded_repeated_mechanism_scope',severity='low',evidence='Fresh metadata restores factual competency descriptions; fixed move rules, goal/rank objective and recurring error patterns remain. Development is exposed; no efficacy or broad unseen-mechanism claim is established.',fix='Report world families separately from mechanisms and retain future independent sealed evaluation custody.',source_role='independent_adjudicator'))
  scores={k:min(e['scores'][k],a['scores'][k]) for k in SCORES};scores['generalization_value']=min(scores['generalization_value'],3)
  output.append(dict(receipt_id='adj-'+batch+'-'+digest([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,scores=scores,critical_defects=defects,issues=issues,decision=decision,confidence=0.98 if decision=='ACCEPT' else 0.99,reason='Fresh exact-record expert and adversarial final reviews, independent oracle reexecution and literal metadata checks agree. Prior task/prompt/gold/split are unchanged; this is a factual metadata correction with zero new underlying families.' if decision=='ACCEPT' else 'Unresolved final reviewer or independently verified evidence requires '+decision+'. Prior versions remain unchanged.',evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=e['decision']!=a['decision'],oracle_author_execution=o['oracle_author_execution'],oracle_original_runner_execution=o['check_runner_execution'],oracle_receipt_key={'sample_id':sid,'record_hash':h},typed_hash_prompt_target_checks=matches,adjudicator_metadata_checks=checks,metadata_details=details,prior_sample_id=previous_id,prior_record_hash=previous['record_hash'],prior_semantic_prompt_target_split_unchanged=True,source_family_key=row['source_family_key'],same_source_family_variants_same_parent_split=True,new_source_family_count_relative_to_prior=0,metadata_revision_fresh_reviews=True),scope_limitations=['Actual architect/adjudicator execution never produced training candidates.','Same underlying model across independent agents can leave correlated errors.','Original protected tests and model artifacts were not opened.','Fresh review is independently recomputed, but not fully blind: root supplied prior tag topics, and adjudicator supplied finalized v1 aggregate counts after v1 initial reviews.']))
 assert all(snapshots[str(p.relative_to(ROOT))]==sha(p) for p in required),'Review evidence changed during execution'
 counts=dict(collections.Counter(r['decision'] for r in output));disagreements=sum(r['evidence']['disagreement_preserved'] for r in output)
 atomic(OUT/f'{batch}.jsonl',''.join(canon(r)+'\n' for r in output))
 execution=dict(schema_version='1.0.0',reviewer_execution_id=ACTOR,role='independent Quality Adjudicator',batch=batch,track_id=tid,completed_at_utc=datetime.now(timezone.utc).isoformat(),records=count,source_families=nfamilies,new_source_families_relative_to_prior_batch=0,prior_batch=prior,tag_changed_records=tag_changes,changed_canonical_keys=dict(differing_keys),decision_counts=counts,reviewer_disagreements_preserved=disagreements,independently_verified_factual_tag_counts=dict(tag_counts),verification_failures=failures,all_final_receipts_complete=True,typed_prior_task_prompt_target_split_unchanged=True,inputs_sha256=snapshots,output_sha256={f'reviews/adjudication/{batch}.jsonl':sha(OUT/f'{batch}.jsonl')},producer_or_release_files_modified=False,previous_reviews_preserved=True,review_context_disclosure='Fresh review independently recomputes facts, but is not fully blind: root disclosed prior tag topics; adjudicator sent finalized v1 aggregate counts after initial v1 reviewers completed. No reviewer verdict files were provided. This limits independence beyond separate actual executions.',role_reuse_disclosure='Actual /root/research_architect authored research specifications, not candidates; separate from producers and both reviewers. Same underlying model correlation remains.')
 atomic(OUT/f'execution-{batch}.json',json.dumps(execution,indent=2)+'\n')
 report=f'''# {batch} metadata revision adjudication

Actual execution `/root/research_architect`: {counts} across {count} {tid} records and {nfamilies} source families. Both completed fresh reviewer receipts were required before decisions. {disagreements} disagreements are preserved. Exact canonical hashes, typed visible payloads, independent-oracle targets and the corrected literal competency tags were checked. Metadata changes affect {tag_changes} records.

This batch contributes **zero new source families relative to {prior}**. Semantic task, student input, gold, structural difficulty and split are unchanged. Repeated worlds must be superseded rather than counted twice. Family variants retain one parent and split; earlier decisions and candidate bytes remain intact. The metadata corrections require fresh hashes and have received them.

The independently authored iterative path multiplicity audit checks every fuel-feasible length for claimed equal routes; it does not confuse equal routes with optimal-route ties. Reachability follows a feasible fresh initial plan. Relational tags follow actual outdegree or initial/explicit-goal TABLE root counts. Existing bounded dynamics, repeated mechanisms/corruption producers, exposed development and same-model error correlation remain limitations. This adjudicator generated no candidates and modified no release files.

These fresh metadata reviews are independently recomputed but not fully blind. Root disclosed prior tag topics, and the adjudicator sent finalized v1 aggregate decision counts after the v1 initial reviews. No other reviewer verdict files were supplied. The limited prior exposure is disclosed rather than represented as a new blind-review execution.
'''
 atomic(OUT/f'ADJUDICATION_{batch.upper().replace("-","_")}.md',report)
 print(json.dumps({k:execution[k] for k in ['batch','records','source_families','new_source_families_relative_to_prior_batch','decision_counts','output_sha256']}))
if __name__=='__main__':
 for batch in sys.argv[1:] or CONFIG:run(batch)
