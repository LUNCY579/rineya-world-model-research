"""Actual root revision audit with immutable source/contract/data bindings."""
import datetime,hashlib,json,os,pathlib,subprocess,time
from oracles.rgm_planning_controls_v2_independent import SPECIFICATIONS
ROOT=pathlib.Path(__file__).resolve().parents[1]
PREFIX='RGM_V2_RENDERER_R1_AUDIT'
names=set(p.relative_to(ROOT).as_posix() for d in ('factory','oracles') for p in (ROOT/d).glob('*.py'))
names.update(SPECIFICATIONS)
names.update(['quality_reports/run_rgm_renderer_r1_audit.py','data/RGM-001/candidate_renderings/planning-gap-v2-renderer-r1.jsonl','data/RGM-001/manifests/planning-gap-v2-renderer-r1.family_controls.json','research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl','reviews/input_only/planning-gap-v1.jsonl','data/RGM-001/candidate_renderings/planning-gap-v2.jsonl','data/RGM-001/manifests/planning-gap-v2.family_controls.json','reviews/input_only/planning-gap-v2-renderer-r1.jsonl','reviews/input_only/planning-gap-v2-renderer-r1.freeze.json','schemas/RGM_PLANNING_V2_RENDERER_REVISION_POLICY.md'])
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def new(n,x):
 with (ROOT/'quality_reports'/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
before={p:sha(p) for p in sorted(names)}
cmd=['/opt/anaconda3/bin/python3','-m','factory.pipeline','audit','--batch','planning-gap-v2-renderer-r1']
pre={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runner_execution':'/root','command':cmd,'environment_overrides':{'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:tests'},'source_sha256_before':before,'scope':'Fresh complete408-row renderer revision audit against unchanged frozenV2 gate. Same204semantic tasks/48families already exposed. No publication authorization.'}
new(PREFIX+'_PREEXECUTION.json',pre)
log=ROOT/'quality_reports'/(PREFIX+'_OUTPUT.txt');start=time.monotonic()
with log.open('xb') as f:r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:tests'},stdout=f,stderr=subprocess.STDOUT)
after={p:sha(p) for p in sorted(names)};raw=log.read_bytes()
outputs={p:sha(p) for p in ['reviews/oracle/planning-gap-v2-renderer-r1.jsonl','audits/leakage/planning-gap-v2-renderer-r1.json','audits/diversity/planning-gap-v2-renderer-r1-planning-controls.json'] if (ROOT/p).exists()}
receipt={**pre,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'source_sha256_after':after,'bound_sources_unchanged':before==after,'output_path':str(log.relative_to(ROOT)),'output_sha256':hashlib.sha256(raw).hexdigest(),'outputs':outputs,'actual_output_tail':raw.decode().splitlines()[-15:]}
new(PREFIX+'_EXECUTION.json',receipt)
print(json.dumps({'exit_code':r.returncode,'sources_unchanged':before==after,'wall_seconds':receipt['wall_seconds'],'receipt_sha256':sha('quality_reports/'+PREFIX+'_EXECUTION.json')}));print(raw.decode()[-1800:])
assert r.returncode==0 and before==after
