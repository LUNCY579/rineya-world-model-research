"""Verify frozen own/public bytes BEFORE importing any own solver/test source."""
import argparse,datetime,hashlib,importlib.util,io,json,pathlib,sys,time,unittest
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parents[2]
def sha(data):return hashlib.sha256(data).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump_new(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
def verify(freeze):
    checks=[]
    for binding in freeze['bindings']:
        path=BASE/binding['path'];archive=BASE/binding['archive'];data=path.read_bytes();ad=archive.read_bytes()
        if sha(data)!=binding['sha256'] or len(data)!=binding['bytes'] or data!=ad:raise RuntimeError('changed frozen bytes '+binding['path'])
        checks.append({'path':binding['path'],'sha256':sha(data),'bytes':len(data),'archive_sha256':sha(ad),'unchanged':True})
    runtime=pathlib.Path(freeze['runtime']['real_path']);raw=runtime.read_bytes()
    if sha(raw)!=freeze['runtime']['sha256'] or len(raw)!=freeze['runtime']['bytes']:raise RuntimeError('runtime bytes changed')
    if pathlib.Path(sys.executable).resolve()!=runtime:raise RuntimeError('wrong Python runtime')
    return checks
class Result(unittest.TextTestResult):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);self.executed_tests=[];self.subtest_executions=0
    def startTest(self,test):self.executed_tests.append(test.id());super().startTest(test)
    def addSubTest(self,test,subtest,outcome):self.subtest_executions+=1;super().addSubTest(test,subtest,outcome)

def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-freeze',required=True);a=p.parse_args()
    started=utc();clock=time.monotonic();freeze_raw=(HERE/'source-freeze-r1.json').read_bytes()
    if sha(freeze_raw)!=a.expected_freeze:raise RuntimeError('unexpected pre-execution freeze')
    freeze=json.loads(freeze_raw);before=verify(freeze);config=json.loads((HERE/'config-r1.json').read_text())
    for filename in config['outputs'].values():
        if (HERE/filename).exists():raise RuntimeError('refusing to overwrite observed output '+filename)
    stream=io.StringIO();spec=importlib.util.spec_from_file_location('own_rsm_preparation_tests_r1',HERE/config['tests']);mod=importlib.util.module_from_spec(spec)
    source_import_started=utc();spec.loader.exec_module(mod)
    suite=unittest.defaultTestLoader.loadTestsFromModule(mod);result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
    after=verify(freeze)
    log=stream.getvalue();logpath=HERE/config['outputs']['log']
    with logpath.open('x',encoding='utf-8') as f:f.write(log)
    receipt={'completed_execution':True,'actual_execution_id':'/root/research_architect','stage':config['stage'],'attempt':config['attempt'],'started_utc':started,'source_import_started_utc':source_import_started,'completed_utc':utc(),'elapsed_seconds':time.monotonic()-clock,'source_freeze_sha256':sha(freeze_raw),'bindings_before_import':before,'bindings_after_execution':after,'all_frozen_bindings_unchanged':before==after,'runtime':dict(freeze['runtime'],actual_python_version=sys.version),'tests_run':result.testsRun,'tests_passed':result.testsRun-len(result.failures)-len(result.errors),'subtest_executions':result.subtest_executions,'failures':[{'test':str(t),'traceback':tb} for t,tb in result.failures],'errors':[{'test':str(t),'traceback':tb} for t,tb in result.errors],'executed_tests':result.executed_tests,'hand_fixture_name':'hand-calibration-r1.json','hand_program_count':5,'log':{'path':str(logpath.relative_to(BASE)),'sha256':sha(log.encode()),'bytes':len(log.encode())},'passed':result.wasSuccessful(),'production_rows_read':0,'data_approval':False,'publication_eligible':None,'future_input_stage_authorized':False,'disclosures':config['disclosures']}
    dump_new(HERE/config['outputs']['execution'],receipt)
    print(json.dumps({k:receipt[k] for k in ['completed_execution','attempt','tests_run','tests_passed','subtest_executions','passed','data_approval','production_rows_read']}))
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
