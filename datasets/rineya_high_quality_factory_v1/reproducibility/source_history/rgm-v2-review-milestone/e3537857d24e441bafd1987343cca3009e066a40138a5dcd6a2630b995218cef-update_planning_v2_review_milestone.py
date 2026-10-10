"""Preserve living report bytes, then record the actual V2 review/audit milestone."""
import datetime,hashlib,json,pathlib
from factory.common import ROOT
from factory.catalog import catalog

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def new(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
 output=ROOT/'handoff/RGM_V2_REVIEW_MILESTONE.json';assert not output.exists()
 proof=ROOT/'backup_logs/rgm-v1-final-review_RECOVERY_VERIFIED_RECEIPT.json';decode=ROOT/'quality_reports/RGM_V1_FINAL_REVIEW_REMOTE_LOSSLESS.json';joined=ROOT/'backup_logs/RGM_V1_FINAL_REVIEW_JOINED_CUSTODY.json'
 assert joined.exists(),'Wait for actual manager joined-custody FINAL before executing'
 assert sha(proof)=='e797ac4f8fa4fee850ada684680d75a11e8f9f28dec4d5afec5a65d74216d83e' and sha(decode)=='f6cde6564b6f1a5767b2b4fb274fefd76ff545a19afde9dff834e73e478ce55a'
 backup=json.loads(proof.read_text());gz=json.loads(decode.read_text());assert backup['status']=='BACKED_UP' and backup['files']==1801 and backup['total_bytes']==849246497
 assert gz['status']=='PASS_ACTUAL_REMOTE_GZIP_DECODE' and len(gz['actual_remote_gzip_proofs'])==24
 raw_bytes=sum(x['raw_bytes'] for x in gz['actual_remote_gzip_proofs']);assert raw_bytes==1188634377
 suite=json.loads((ROOT/'quality_reports/RGM_V2_FULL_EXECUTION.json').read_text());assert suite['exit_code']==0 and suite['bound_sources_unchanged'] and any('Ran 358 tests' in s for s in suite['actual_output_tail'])
 for name,h in suite['source_sha256_before'].items():assert sha(ROOT/name)==h,name
 audit=json.loads((ROOT/'audits/leakage/planning-gap-v2.json').read_text());assert audit['records']==408 and audit['oracle_failures']==audit['metadata_failures']==0 and audit['planning_control_error_count']==11 and not audit['all_machine_quality_gates_pass']
 assert sha(ROOT/'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl')=='57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23'
 rows,active=catalog();assert len(rows)==3340 and active['active_source_families']==648
 names=['backup_logs/rgm-v1-final-review_RECOVERY_VERIFIED_RECEIPT.json','quality_reports/RGM_V1_FINAL_REVIEW_REMOTE_LOSSLESS.json','backup_logs/RGM_V1_FINAL_REVIEW_JOINED_CUSTODY.json','quality_reports/RGM_V2_FULL_EXECUTION.json','quality_reports/RGM_V2_FOCUSED_EXECUTION.json','quality_reports/RGM_V2_AUDIT_EXECUTION.json','quality_reports/RGM_V2_RENDERER_CONTROL_FAILURE_DIAGNOSTIC.json','quality_reports/RGM_V2_REVIEW_BOUNDARY_PREFLIGHT.json','reviews/expert/planning-gap-v2.expert_input.freeze.json','reviews/adversarial/planning-gap-v2.input_freeze.json','manifests/planning-gap-v2_REPRODUCTION.json','manifests/planning-gap-v2_PRODUCER_FINAL_CUSTODY.json','data/RGM-001/candidate_renderings/planning-gap-v2.jsonl','data/RGM-001/manifests/planning-gap-v2.family_controls.json','factory/pipeline.py','factory/planning_controls.py']
 evidence={n:sha(ROOT/n) for n in names}
 section='''# Current planning V2 review milestone — 2026-10-10 UTC

This newest status supersedes the older in-progress summaries preserved below. The approved catalog remains **3,340 records / 648 source families across all ten tracks**. No new release has been created. Planning V1's 384 records and belief V1/V2/V3 remain wholly quarantined; their original candidate, first-review, failure and final-adjudication evidence remains immutable.

The fresh planning V2 producer completed **408 pending candidates / 204 normalized tasks / the same 48 revised families**, with340 train and68 exposed-development rows. Two independently executed stripped-input reviews each froze all 408 own exact predictions and first input judgments before canonical access. Both independently confirm 204 normalized tasks; the two ordered graph renderings produce 300 distinct raw typed task encodings. First predictions/judgments are immutable. Canonical target/declaration comparisons are separately authorized and remain pending at this milestone. Actual roles inherit the same model; producer/reviewer separation does not imply model diversity. The public-design author later adjudicates, with that overlap disclosed.

The successful generator is fresh version `rgm-planning-gap-2.0.2`, source attempt04. All8 core artifacts reproduce byte-for-byte under two separate hash seeds. Attempts 01 and 03 failed bounded searches before emitting rows; attempt 02 was frozen but not executed after version steering. Every attempt source, receipt and failure is retained. Revised rows add zero source families relative to V1. These facts establish reproducibility, not quality approval or model benefit.

The integrated source-bound suite passes **358 actual tests**; **51 focused tests** pass. Two initial integration errors concerned the V1 report attribute and test import path. Failed source/logs and the runner's failed first invocation are preserved. Neither frozen independent oracle nor its tests was edited. Current root integration changes are newer than the previous verified checkpoint.

The actual whole 408 candidate audit recomputes all targets using the independent, reference and frozen planning checkers. All408 schema/target/row-metadata checks pass, with zero measured split leakage, cross-parent exact semantic duplicates or duplicate prompts. **11 family-control errors remain, and `all_machine_quality_gates_pass=false`.** The frozen checker completes 37 of 48 families and verifies 74 semantic pairs / 148 required renderer declarations before the 11 D-family errors interrupt those families. These partial counts do not establish complete 192-declaration verification. Post-audit localization finds 66 D rows whose shared goal-map key order differs from their base presentation, while top-level/on/action-object ordering matches exactly one base style. This is a guided diagnostic; whether that exact key-order projection is a published requirement awaits independent contract assessment. No checker/data/criterion fix or publication has been applied. See the actual audit and diagnostic; do not treat successful test execution or a process exit0 as a passing dataset gate.

The authoritative frozen archival checkpoint is **448c36af9370460782da3c0a4ba7c5ca4d8ba71a** on the private dedicated branch. Actual raw GitHub blob checks verify 1,801 frozen paths / 849,246,497 bytes / 1,497 unique blobs. A separate actual backup-manager run fetches and decodes all 24 remote gzip artifacts byte-for-byte to 1,188,634,377 preserved raw bytes. Earlier staging/preflight/whitespace/first decode failures remain FAILED; the new recovery VERIFIED receipt, manager lossless proof and joined custody are authoritative. The exceptional raw Git packed-refs snapshot's original trailing space was preserved exactly; strict whitespace checks passed all other paths. The original research main/index and 65 read-only reference hashes remain unchanged. The dedicated worktree is clean at 448.

That checkpoint binds the completed V1 scope and public V2 documents, not later V2candidate/checker/reviewer/core integration changes or postinventory recovery receipts. All new milestone artifacts/current documentation and the remote-verification receipts themselves remain **local-only until the next exact verified checkpoint**. A backup receipt cannot include its own completed bytes; carry it forward explicitly. Existing approved releases remain remotely preserved. Storage permits bounded CPU work; no unrelated project was cleaned.

Next: finish each independent canonical comparison without sharing other verdicts/formal private evidence, obtain a separate formal contract assessment of the 11 renderer failures, then obtain actual independent adjudication. Preserve the entire candidate and failed audit. Release only if every existing quality gate and exact-role approval passes; otherwise quarantine and issue a fresh prospective revision. Prepare the next exact archival checkpoint after completed review/adjudication or a meaningful 20–30 minute milestone. No Rineya training, GPU job, model/architecture change, sealed-test tuning or efficacy claim is authorized or performed.

---

'''
 changes=[]
 living=['PROJECT_STATUS.md','RESEARCH_GAPS.md','DATASET_QUALITY_REPORT.md','GENERATOR_IMPROVEMENTS.md','CURRICULUM_COVERAGE.md','TRAINING_READINESS.md','GITHUB_BACKUP_LOG.md','NEXT_SESSION_HANDOFF.md','handoff/NEXT_SESSION_HANDOFF.md','research/role_registry.json']
 for n in living:
  p=ROOT/n;old=p.read_bytes();h=hashlib.sha256(old).hexdigest();archive=ROOT/'handoff/report_history/rgm-v2-review-milestone'/(h+'-'+p.name);archive.parent.mkdir(parents=True,exist_ok=True)
  if archive.exists():assert archive.read_bytes()==old
  else:archive.write_bytes(old)
  if n.endswith('.json'):
   registry=json.loads(old);wave=registry['rgm_planning_wave'];wave.update({'stage':'V2_ALL408_TARGETS_TRUE_11_RENDERER_FAMILY_HOLDS_CANONICAL_REVIEWS_PENDING','current_tests':{'count':358,'execution':'quality_reports/RGM_V2_FULL_EXECUTION.json'},'v2_stage':'ACTUAL408_FIRST_PREDICTIONS_FROZEN_BEFORE_GOLD_COMPARISONS_PENDING','v2_records':408,'v2_normalized_tasks':204,'v2_raw_typed_inputs':300,'v2_revised_families':48,'v2_family_growth_relative_v1':0,'v2_machine_quality_gates_pass':False,'v2_control_errors':11,'v2_audit':'audits/leakage/planning-gap-v2.json','backup':'448c36af9370460782da3c0a4ba7c5ca4d8ba71a actual 1,801 path / 1,497 blob bytes and 24 lossless proofs; newer V2/local receipts not yet backed up'})
   content=(json.dumps(registry,indent=2,sort_keys=True)+'\n').encode()
  else:content=section.encode()+old
  p.write_bytes(content);changes.append({'path':n,'before_sha256':h,'preserved_path':str(archive.relative_to(ROOT)),'after_sha256':sha(p)})
 report={'runner_execution':'/root','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(pathlib.Path(__file__)),'approved_records':3340,'active_source_families':648,'new_releases':0,'v2_pending_records':408,'v2_normalized_tasks':204,'v2_family_control_failures':11,'v2_publication_allowed':False,'production_test_count':358,'focused_test_count':51,'verified_backup_commit':backup['commit_sha'],'postinventory_artifacts_local_only':True,'evidence_sha256':evidence,'living_report_updates':changes}
 new(output,report);new(ROOT/'quality_reports/RGM_V2_REVIEW_LIVING_REPORT_UPDATE.json',report);print(json.dumps({'report_sha256':sha(output),'updated_reports':len(changes),'new_releases':0}))

if __name__=='__main__':main()
