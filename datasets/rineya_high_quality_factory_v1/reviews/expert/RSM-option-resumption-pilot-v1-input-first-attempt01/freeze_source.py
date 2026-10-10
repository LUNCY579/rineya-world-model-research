"""Stdlib-only preparation; no interpreter import or execution of predictions."""
from pathlib import Path
import datetime,hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def h(b):return hashlib.sha256(b).hexdigest()
expected={
 'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl':'d99074c6cd18d1bd9709a09f833bfe9fea50c0c3981ece532c3951aac220f6b1',
 'reviews/input_only/RSM-option-resumption-pilot-v1.INPUT_CUSTODY.json':'231250e9819e72608a9a34bc32b3efea22f2a7943b5e4a0eacf8560015457e76',
 'quality_reports/RSM_INPUT_ONLY_PACKET_ROOT_COMPLETION_ATTEMPT01.json':'ae18870448d846d2d13027b55e617305c9ddc1ce1d017da368f052c6492af266',
 'schemas/RSM_OPTION_RESUMPTION_V1.md':'44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f',
 'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md':'8bda187009cc91756248ddd86792c28aabe0793a3f6be304bb9433bfdf48be34',
 'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json':'422f67c836639d0accbf411c3080ef4fc7695ad8630120a140488c90a19061fa',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md':'1612b14d7549b7fb406cbef8e127440dcfddaf30548ef29d217bc49aefb6c854',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md':'5f7f6a2c45dd6a0b51c7a912b28fd49badd9aa3aa2eac8632e9b9781c201b57b',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md':'b62663c3082c9e2ee2bf34aadc1fdd10f263a6497396b6768f1475b12f568fe4',
 'research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R2_ROOT_FREEZE.json':'3d82da83535c6e8c0a0884d870945d064605e1f58ae2072153c5da8c8d8daeaa',
 'reviews/expert/RSM-option-resumption-public-preparation-20261010T1021Z/solver-attempt01.py':'aefd06ae9de6d661fb7d7e61e0d9721fdac090d69064a6a79fec06d7ac081967',
 'reviews/expert/RSM-option-resumption-public-preparation-20261010T1021Z/tests-attempt01.py':'e034e30d73264f48dd3e40d9c8cbfeeafa3c9acf8e4347514f8227b1d433cf09',
 'reviews/expert/RSM-option-resumption-public-preparation-20261010T1021Z/hand-calibrations-attempt01.json':'c7638941ba2c1fae6c93e0cf0543f6f68f4c0fe0ddab5eacd65b098522a9ac9d'
}
paths=list(expected)+[str((HERE/n).relative_to(ROOT)) for n in ('config.json','input_review_runner.py','freeze_source.py')]
archive=HERE/'source-input-archive';archive.mkdir()
bindings=[]
for name in paths:
 b=(ROOT/name).read_bytes();hs=h(b)
 if name in expected:assert hs==expected[name],name
 ap=archive/(hs+'.blob')
 if not ap.exists():ap.write_bytes(b)
 assert ap.read_bytes()==b
 bindings.append({'path':name,'sha256':hs,'bytes':len(b),'archive_path':str(ap.relative_to(HERE))})
python=Path(sys.executable).resolve();pb=python.read_bytes();runtime={'python_executable':str(python),'python_executable_sha256':h(pb),'python_executable_bytes':len(pb),'sys_version':sys.version,'stdlib_only_dependency':'Python standard library; frozen own interpreter/tests import only copy/json/re/collections'}
runtimearchive=archive/(h(pb)+'.python-binary.blob');runtimearchive.write_bytes(pb);assert runtimearchive.read_bytes()==pb
runtime['binary_archive_path']=str(runtimearchive.relative_to(HERE))
f={'actual_author_execution_id':'/root/expert_reviewer','stage':'BEFORE_FIRST_NEW_INTERPRETER_IMPORT_TEST_OR_PREDICTION','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'runtime':runtime,
 'freeze_command':'/opt/anaconda3/bin/python3 -B reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/freeze_source.py',
 'planned_actual_command':'/opt/anaconda3/bin/python3 -B reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01/input_review_runner.py',
 'no_candidate_gold_read':True,'previous_source_preserved':True,'archive_verified_byte_equal':True}
p=HERE/'PREEXECUTION.json'
with p.open('x') as s:json.dump(f,s,sort_keys=True,indent=2);s.write('\n')
print(json.dumps({'preexecution':str(p.relative_to(ROOT)),'sha256':h(p.read_bytes()),'runner_sha256':h((HERE/'input_review_runner.py').read_bytes()),'runtime':runtime,'bindings':len(bindings)}))
