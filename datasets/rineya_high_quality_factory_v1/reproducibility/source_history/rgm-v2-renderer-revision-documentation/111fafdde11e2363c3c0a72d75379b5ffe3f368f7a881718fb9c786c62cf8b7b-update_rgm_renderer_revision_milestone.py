"""Preserve prior living bytes, then document actual quarantine and renderer repair."""
import datetime,hashlib,json,pathlib
from factory.common import ROOT
from factory.catalog import catalog
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 evidence_names=['reviews/adjudication/planning-gap-v2.execution.json','reviews/adjudication/planning-gap-v2.jsonl','reviews/adjudication/planning-gap-v2.audit.json','quality_reports/RGM_PLANNING_V2_QUARANTINE_EXECUTION.json','data/RGM-001/quarantine/planning-gap-v2.jsonl','reviews/expert/planning-gap-v2.jsonl','reviews/adversarial/planning-gap-v2.jsonl','quality_reports/RGM_V2_FROZEN_REVIEW_EVIDENCE_JOIN.json','quality_reports/RGM_V2_FULL_EXECUTION.json','quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json','audits/leakage/planning-gap-v2-renderer-r1.json','audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json','manifests/planning-gap-v2-renderer-r1_PRODUCER_FINAL_CUSTODY.json','manifests/planning-gap-v2-renderer-r1_REPRODUCTION.json','reviews/input_only/planning-gap-v2-renderer-r1.freeze.json','data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl','data/RGM-001/manifests/planning-gap-v2-renderer-r1.family_controls.json','backup_logs/RGM_V1_FINAL_REVIEW_JOINED_CUSTODY.json','schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md','factory/pipeline.py','factory/planning_controls.py']
 evidence={n:sha(ROOT/n) for n in evidence_names}
 def obj(n):return json.loads((ROOT/n).read_bytes())
 adjud=obj(evidence_names[0]);assert adjud['completed_execution'] and adjud['decision_counts']=={'QUARANTINE':408} and not adjud['publication_eligible']
 q=obj('quality_reports/RGM_PLANNING_V2_QUARANTINE_EXECUTION.json');assert q['pass'] and q['whole_quarantine_records']==408
 audit=obj('audits/leakage/planning-gap-v2-renderer-r1.json');assert audit['all_machine_quality_gates_pass'] and audit['planning_control_error_count']==0 and audit['oracle_failures']==0
 assert audit['planning_control_counts']['complete_v2_families']==48 and audit['planning_control_counts']['required_renderer_control_declarations']==192
 producer=obj('manifests/planning-gap-v2-renderer-r1_PRODUCER_FINAL_CUSTODY.json');assert producer['all_eight_core_artifacts_reproduced_byte_exactly'] and producer['new_semantic_tasks']==producer['new_source_families']==0
 active,report=catalog();assert len(active)==3340 and report['active_source_families']==648
 section='''# Current renderer revision and original V2 adjudication — 2026-10-10 UTC

The approved catalog remains **3,340 records / 648 source families across all ten tracks**. No new release or training has occurred. Actual independent adjudication now quarantines the complete original planning V2 cohort: **408 records**, with exact original rows preserved under `data/RGM-001/quarantine/planning-gap-v2.jsonl`. Planning V1 and belief V1/V2/V3 remain quarantined. All prior candidate/review/specification/checker/failure bytes are retained.

Both original expert and adversarial canonical reviews completed with408 scoped ACCEPT judgments each, after all their own input predictions and first judgments froze. The independent adjudicator recomputed408 exact targets,510 witnesses and192 renderer declarations across48 complete families/204 tasks. No false gold or factual control was found. The original frozen gate still fails11 D families:66 projected rows change shared goal-key order. Formal and adjudicator assessments find that exact projection was not explicitly prescribed by the public wording. The required admission gate remains unresolved; the whole original cohort is QUARANTINE, with no partial publication and no retrospective gate relaxation. `RGM-RENDERER-008` records the remedy. Same underlying model and public-design/adjudicator overlap remain disclosed.

A fresh prospective presentation revision, `planning-gap-v2-renderer-r1`, now contains408 PENDING records:340 train/68 exposed-development, reusing all204 semantic tasks and48 source families, with **zero new semantic tasks or families**. Only22 visible inputs change goal-map key order. Root exact joins independently preserve every408 typed input semantic task/target, split, context, difficulty and skill tags. New custody IDs/hashes are explicit. Producer version `rgm-planning-gap-2.0.5`, source attempt03, was frozen before generation; all8 core artifacts reproduce byte-for-byte across separate seeds17/971. Attempt01 failed a protected-path substitution check before generation; attempt02 passed20 own probes then failed a lineage-availability guard before emitting rows. Both failure sources/receipts remain preserved.

The fresh actual whole-cohort machine audit passes the **unchanged frozen V2 checker and seven public bindings**:408 schema/target/metadata checks,48 complete families,96 semantic control pairs,192 renderer declarations and72 mutant executions, with zero measured oracle/control/metadata/split/exact-duplicate failures. Its source-bound execution receipt is `quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json`. The current unchanged integration previously passed358 full tests and51 focused tests; the fresh data audit does not imply independent quality approval. Difficulty remains234 level3/32 level2/142 level4 and uncalibrated to a trained model. Public novelty gates concern prior active structures; all revised tasks reuse original V2 semantics and existing mechanism grammars. Correlated context/rendering views, tiny-development class imbalance and scoped conditional associations remain documented risks.

Fresh independent review and adjudication are **pending**. A six-field problem-only view is frozen under `reviews/input_only/planning-gap-v2-renderer-r1.jsonl`. Each reviewer must freeze all408 own fresh predictions/first judgments before fresh canonical access, after exact semantic joins if reusing their own original predictions. Prior original canonical exposure and same-task reuse must be disclosed; this is not an unseen blind semantic review. No producer private expectations, formal private implementation or another first verdict may enter that first phase. No release may precede complete fresh role receipts, whole-cohort admission, student-export validation and independent adjudication.

The last actual remotely verified archival commit remains **448c36af9370460782da3c0a4ba7c5ca4d8ba71a**:1,801 frozen paths/849,246,497 bytes/1,497 actual remote blobs, plus24 separate actual remote gzip-decode proofs preserving1,188,634,377 raw bytes. New V2/revision/review/recovery/current-report artifacts are local until the next exact verified checkpoint. A successful commit/push is insufficient; actual GitHub payload and lossless-decode checks must complete. The next checkpoint is archival and must preserve every prior tracked path and historical release; pending/quarantined data receives no approval through backup. Protected research main/index and65 read-only reference hashes remain unchanged in the latest actual boundary receipt.

Next: complete fresh reviewer phases and independent adjudication; publish only if every existing gate passes. Preserve this important review/repair milestone on the dedicated private branch with actual remote-byte verification. Continue dataset-only CPU work on diagnosed gaps afterward. No Rineya model training, GPU job, sealed evaluation tuning, architecture change or learning-efficacy claim has occurred.

---

'''
 changes=[]
 living=['PROJECT_STATUS.md','RESEARCH_GAPS.md','DATASET_QUALITY_REPORT.md','GENERATOR_IMPROVEMENTS.md','CURRICULUM_COVERAGE.md','TRAINING_READINESS.md','GITHUB_BACKUP_LOG.md','NEXT_SESSION_HANDOFF.md','handoff/NEXT_SESSION_HANDOFF.md','research/role_registry.json']
 for n in living:
  p=ROOT/n;old=p.read_bytes();h=hashlib.sha256(old).hexdigest();a=ROOT/'handoff/report_history/rgm-v2-renderer-revision-milestone'/(h+'-'+p.name);a.parent.mkdir(parents=True,exist_ok=True)
  if a.exists():assert a.read_bytes()==old
  else:
   with a.open('xb') as f:f.write(old)
  if n.endswith('.json'):
   registry=json.loads(old);registry['rgm_planning_wave'].update({'stage':'ORIGINAL_V2_WHOLE_QUARANTINE_RENDERER_R1_MACHINE_PASS_FRESH_REVIEWS_PENDING','v2_stage':'ACTUAL408_QUARANTINE_NO_FALSE_GOLD_FINDING','v2_quarantine_records':408,'v2_machine_quality_gates_pass':False,'renderer_revision':'planning-gap-v2-renderer-r1','renderer_revision_generator':'rgm-planning-gap-2.0.5','renderer_revision_records':408,'renderer_revision_reused_tasks':204,'renderer_revision_reused_families':48,'renderer_revision_new_semantic_tasks':0,'renderer_revision_new_source_families':0,'renderer_revision_machine_quality_gates_pass':True,'renderer_revision_fresh_review_status':'PENDING','latest_verified_backup_commit':'448c36af9370460782da3c0a4ba7c5ca4d8ba71a','newer_milestone_artifacts_local_only':True});content=(json.dumps(registry,indent=2,sort_keys=True)+'\n').encode()
  else:content=section.encode()+old
  p.write_bytes(content);changes.append({'path':n,'before_sha256':h,'preserved_path':str(a.relative_to(ROOT)),'after_sha256':sha(p)})
 receipt={'runner_execution':'/root','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(pathlib.Path(__file__)),'approved_records':3340,'active_source_families':648,'created_releases':0,'original_v2_quarantine_records':408,'renderer_revision_pending_records':408,'renderer_revision_machine_quality_gates_pass':True,'new_semantic_tasks':0,'new_source_families':0,'fresh_reviews_pending':True,'evidence_sha256':evidence,'living_report_updates':changes,'latest_verified_backup_commit':'448c36af9370460782da3c0a4ba7c5ca4d8ba71a','newer_artifacts_local_only':True}
 new(ROOT/'handoff/RGM_RENDERER_REVISION_MILESTONE.json',receipt);new(ROOT/'quality_reports/RGM_RENDERER_REVISION_LIVING_REPORT_UPDATE.json',receipt)
 print(json.dumps({'receipt_sha256':sha(ROOT/'handoff/RGM_RENDERER_REVISION_MILESTONE.json'),'updated_reports':10,'created_releases':0}))
if __name__=='__main__':main()
