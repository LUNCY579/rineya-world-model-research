"""Preserve exact original records after actual completed whole-cohort adjudication."""
import datetime,hashlib,json,pathlib,sys
from factory.catalog import catalog
from factory.common import ROOT, canonical
def sha(n):return hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
def new(n,x):
 with (ROOT/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 execution_name=sys.argv[1];execution_sha=sys.argv[2]
 assert sha(execution_name)==execution_sha
 execution=json.loads((ROOT/execution_name).read_bytes())
 assert execution.get('completed_execution') is True and execution['actual_execution_id']=='/root/research_architect'
 names=['data/RGM-001/candidate_renderings/planning-gap-v2.jsonl','reviews/adjudication/planning-gap-v2.jsonl',execution_name,'reviews/adjudication/planning-gap-v2.audit.json','reviews/expert/planning-gap-v2.jsonl','reviews/adversarial/planning-gap-v2.jsonl','reviews/oracle/planning-gap-v2.jsonl','audits/diversity/planning-gap-v2-planning-controls.json']
 before={n:sha(n) for n in names}
 assert before[names[0]]=='57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23'
 rows=[json.loads(l) for l in (ROOT/names[0]).read_bytes().splitlines()]
 decisions=[json.loads(l) for l in (ROOT/names[1]).read_bytes().splitlines()]
 byid={r['sample_id']:r for r in decisions};assert len(byid)==len(rows)==408
 assert len({r['receipt_id'] for r in decisions})==408
 assert set(byid)=={r['sample_id'] for r in rows}
 for r in rows:
  d=byid[r['sample_id']]
  assert d['record_hash']==r['record_hash'] and d['decision']=='QUARANTINE' and d['reviewer_role']=='independent_adjudicator' and d['reviewer_execution_id']=='/root/research_architect'
  assert d['evidence']['publication_eligible'] is False
 pre={'actual_execution_id':'/root','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':before,'runner_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'independent_adjudicator_completed':True,'whole_quarantine_records':408,'scope':'Exact original nested records; unresolved required renderer admission hold, no false-target finding. No data approval.'}
 new('quality_reports/RGM_PLANNING_V2_QUARANTINE_PREEXECUTION.json',pre)
 output='data/RGM-001/quarantine/planning-gap-v2.jsonl'
 payload=''.join(canonical({'sample_id':r['sample_id'],'record':r,'adjudication_receipt_id':byid[r['sample_id']]['receipt_id'],'adjudication_decision':'QUARANTINE','pending_or_failed_roles':['adjudication_required_renderer_admission_hold']})+'\n' for r in rows).encode()
 with (ROOT/output).open('xb') as f:f.write(payload)
 readback=[json.loads(l) for l in (ROOT/output).read_bytes().splitlines()]
 assert all(canonical(x['record'])==canonical(r) for x,r in zip(readback,rows)) and len(readback)==408
 assert before=={n:sha(n) for n in names}
 active,report=catalog();assert len(active)==3340 and report['active_source_families']==648
 result={**pre,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pass':True,'quarantine_path':output,'quarantine_sha256':sha(output),'quarantine_bytes':len(payload),'all_candidate_evidence_unchanged':True,'created_release_paths':[],'active_approved_records':3340,'active_source_families':648}
 new('quality_reports/RGM_PLANNING_V2_QUARANTINE_EXECUTION.json',result)
 print(json.dumps({'quarantine_sha256':sha(output),'records':408,'created_releases':0,'execution_sha256':sha('quality_reports/RGM_PLANNING_V2_QUARANTINE_EXECUTION.json')}))
if __name__=='__main__':main()
