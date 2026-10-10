"""SOURCE ONLY now. Later calibration execution needs a separate root grant.
External wrapper/config SHA and complete raw runtime guards precede own imports.
"""
import argparse,contextlib,datetime,hashlib,importlib.util,io,json,pathlib,sys,traceback,types,unittest
D=pathlib.Path(__file__).resolve().parent;B=D.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def newjson(path,value):
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def newtext(path,value):
 with path.open('x',encoding='utf-8') as f:f.write(value)
def rawbindings(items):
 verified=[]
 for item in items:
  path=pathlib.Path(item['path']) if item['path'].startswith('/') else B/item['path'];raw=path.read_bytes();archive=(B/item['archive']).read_bytes()
  if len(raw)!=item['bytes'] or sha(raw)!=item['sha256'] or raw!=archive:raise ValueError('raw frozen byte drift '+item['path'])
  verified.append({'path':item['path'],'sha256':item['sha256'],'bytes':item['bytes'],'archive':item['archive']})
 return verified
def runtime_guard(items,resolved):
 if str(pathlib.Path(sys.executable).resolve())!=resolved:raise ValueError('resolved Python differs')
 bypath={str((pathlib.Path(v['path']) if v['path'].startswith('/') else B/v['path']).resolve()):v for v in items};checked=[]
 for name,module in sorted(sys.modules.items()):
  source=getattr(module,'__file__',None)
  if source is None:continue
  p=pathlib.Path(source)
  if p.suffix=='.pyc':
   try:p=pathlib.Path(importlib.util.source_from_cache(str(p)))
   except ValueError:pass
  if not p.is_file():raise ValueError('registered file-backed module lacks raw origin '+name)
  p=p.resolve();item=bypath.get(str(p))
  if item is None:raise ValueError('registered module outside frozen closure '+name+':'+str(p))
  raw=p.read_bytes()
  if sha(raw)!=item['sha256'] or len(raw)!=item['bytes']:raise ValueError('registered module bytes changed '+name)
  checked.append({'module':name,'path':str(p),'sha256':item['sha256'],'bytes':item['bytes']})
 return checked
def load_registered(name,path):
 if name in sys.modules:raise ValueError('would replace registered module '+name)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def register_exposed_owned_helpers(module,allowed):
 registered=[]
 for value in vars(module).values():
  if not isinstance(value,types.ModuleType) or not getattr(value,'__file__',None):continue
  p=pathlib.Path(value.__file__).resolve()
  if str(p) not in allowed:continue
  name=value.__name__
  if name in sys.modules and sys.modules[name] is not value:raise ValueError('conflicting owned module registration '+name)
  sys.modules[name]=value;registered.append({'module':name,'path':str(p)})
 return registered
def run(args):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();cfgraw=(D/'calibration-runner-config-r1.json').read_bytes();wrapraw=(D/'CALIBRATION_RUNNER_SOURCE_FREEZE-r1.json').read_bytes()
 if sha(cfgraw)!=args.expected_config:raise ValueError('external runner config SHA differs')
 if sha(wrapraw)!=args.expected_freeze:raise ValueError('external runner freeze SHA differs')
 cfg=json.loads(cfgraw);wrapper=json.loads(wrapraw)
 if wrapper['config_sha256']!=args.expected_config:raise ValueError('wrapper config binding differs')
 basepath=B/cfg['base_source_freeze'];baseraw=basepath.read_bytes()
 if sha(baseraw)!=cfg['base_source_freeze_sha256'] or sha(baseraw)!=wrapper['base_source_freeze_sha256']:raise ValueError('base source freeze differs')
 base=json.loads(baseraw)
 if len(base['bindings'])!=453:raise ValueError('base raw binding completeness differs')
 items=base['bindings']+wrapper['bindings'];before=rawbindings(items);modules_before=runtime_guard(items,base['resolved_python'])
 # Every source/config/raw/runtime check above occurs before the first owned
 # semantic module import. The invocation flag is not itself authorization.
 for name in cfg['output_paths']:
  if (D/name).exists():raise ValueError('would overwrite prior calibration output '+name)
 stdout=io.StringIO();stderr=io.StringIO();runnerlog=io.StringIO();helper_names=[]
 with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
  tests=load_registered('owned_rsm_canonical_calibration_r1',D/cfg['tests_source'])
  allowed={str((B/path).resolve()) for path in cfg['exposed_owned_helpers']};helper_names=register_exposed_owned_helpers(tests,allowed)
  modules_after_import=runtime_guard(items,base['resolved_python'])
  suite=unittest.defaultTestLoader.loadTestsFromModule(tests);result=unittest.TextTestRunner(stream=runnerlog,verbosity=2).run(suite)
 modules_after=runtime_guard(items,base['resolved_python']);after=rawbindings(items)
 newtext(D/'CALIBRATION_RAW_STDOUT-r1.txt',stdout.getvalue());newtext(D/'CALIBRATION_RAW_STDERR-r1.txt',stderr.getvalue());newtext(D/'CALIBRATION_UNITTEST_LOG-r1.txt',runnerlog.getvalue())
 receipt={'actual_execution_id':'/root/research_architect','completed_execution':True,'stage':'HANDMADE_CALIBRATION_ONLY_NO_PRODUCTION_COMPARISON','started_utc':start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_wrapper_sha256':sha(wrapraw),'runner_config_sha256':sha(cfgraw),'base_source_freeze_sha256':sha(baseraw),'base_binding_count':453,'total_binding_count':len(items),'before_raw_bindings':before,'after_raw_bindings':after,'all_raw_SHA_size_archive_bytes_unchanged':before==after,'registered_file_module_checks_before':modules_before,'registered_file_module_checks_after_import':modules_after_import,'registered_file_module_checks_after_tests':modules_after,'owned_loads_registered':helper_names,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),'skips':len(result.skipped),'production_targets_compared':0,'machine_baseline_executed':False,'quality_votes':0,'data_approval':False,'registry_scope':'Registered module origins plus explicitly exposed owned helper objects; all frozen file bytes independently checked regardless of registration.'}
 newjson(D/'CALIBRATION_EXECUTION-r1.json',receipt)
 print(runnerlog.getvalue(),end='');print(json.dumps({'calibration_completed':True,'tests_run':result.testsRun,'success':result.wasSuccessful(),'data_approval':False}))
 if not result.wasSuccessful():raise SystemExit(1)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-config',required=True);parser.add_argument('--expected-freeze',required=True);parser.add_argument('--execute-after-separate-root-grant',action='store_true');args=parser.parse_args()
 if not args.execute_after_separate_root_grant:raise SystemExit('Source prepared only; separate root execution grant required.')
 try:run(args)
 except Exception:
  newjson(D/'CALIBRATION_RUNNER_FAILED-r1.json',{'actual_execution_id':'/root/research_architect','completed_execution':False,'traceback':traceback.format_exc(),'data_approval':False,'production_target_comparison_completed':False});raise
if __name__=='__main__':main()
