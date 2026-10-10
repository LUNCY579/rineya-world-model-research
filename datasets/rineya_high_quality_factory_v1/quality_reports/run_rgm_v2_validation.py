"""Root real validation runner; immutable mode receipts and before/after bindings."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
mode=sys.argv[1]
assert mode in {'focused','full','audit'}
prefix='RGM_V2_'+mode.upper()
names=set(p.relative_to(ROOT).as_posix() for directory in ('factory','oracles') for p in (ROOT/directory).glob('*.py'))
if mode=='full':
 old=json.loads((ROOT/'quality_reports/RGM_V1_CASE_BOUNDARY_FULL_SUITE_PREEXECUTION.json').read_text())
 modules=old['command'][3:-1]+['tests.test_planning_gap_v2_gate','tests.oracle_rgm_planning_v2_test']
 cmd=['/opt/anaconda3/bin/python3','-m','unittest',*modules,'-v']
elif mode=='focused':
 modules=['tests.test_planning_gap_v2_gate','tests.test_planning_control_gate','tests.test_planning_review_corrections','tests.oracle_rgm_planning_v2_test']
 cmd=['/opt/anaconda3/bin/python3','-m','unittest',*modules,'-v']
else:
 modules=[];cmd=['/opt/anaconda3/bin/python3','-m','factory.pipeline','audit','--batch','planning-gap-v2']
for module in modules:names.add(module.replace('.','/')+'.py')
names.add('tests/oracle_rgm_planning_controls_test.py')
names.add('quality_reports/run_rgm_v2_validation.py')
for name in ['data/RGM-001/candidate_renderings/planning-gap-v2.jsonl','data/RGM-001/manifests/planning-gap-v2.family_controls.json','research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl','reviews/input_only/planning-gap-v1.jsonl']:
 names.add(name)
from oracles.rgm_planning_controls_v2_independent import SPECIFICATIONS
names.update(SPECIFICATIONS)
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def write_new(name,payload):
 with (ROOT/'quality_reports'/name).open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
before={name:sha(name) for name in sorted(names)}
pre={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runner_execution':'/root','command':cmd,'environment_overrides':{'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:tests'},'source_sha256_before':before,'scope':'Actual frozen source tests' if modules else 'Actual whole408candidate audit with all target/schema/metadata/manifest/novelty controls; no publication authorization'}
write_new(prefix+'_PREEXECUTION.json',pre)
log=ROOT/'quality_reports'/(prefix+'_OUTPUT.txt'); started=time.monotonic()
with log.open('xb') as f:result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:tests'},stdout=f,stderr=subprocess.STDOUT)
after={name:sha(name) for name in sorted(names)};raw=log.read_bytes()
receipt={**pre,'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,'source_sha256_after':after,'bound_sources_unchanged':after==before,'output_path':str(log.relative_to(ROOT)),'output_sha256':hashlib.sha256(raw).hexdigest(),'output_bytes':len(raw),'actual_output_tail':raw.decode().splitlines()[-15:]}
if mode=='audit':receipt['outputs']={name:sha(name) for name in ['reviews/oracle/planning-gap-v2.jsonl','audits/leakage/planning-gap-v2.json','audits/diversity/planning-gap-v2-planning-controls.json'] if (ROOT/name).exists()}
write_new(prefix+'_EXECUTION.json',receipt)
print(json.dumps({'mode':mode,'exit_code':result.returncode,'bound_sources_unchanged':after==before,'wall_seconds':receipt['wall_seconds'],'output_sha256':receipt['output_sha256']}));print(raw.decode()[-1400:])
assert result.returncode==0 and after==before
