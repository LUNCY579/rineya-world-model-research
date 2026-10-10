"""Future root-run installed-source runner. PREPARATION ONLY until grant/freeze.

No staged adapter is imported from this prefix. Verify installed/runtime byte
archives BEFORE first core import, then run HAND gates or root-authorized suite.
The separate reference's opaque raw archive is frozen/verified by root, never
inspected by the formal author while preparing this source.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path,PurePosixPath
import sys
import time
import traceback
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
def safe_path(relative):
    if type(relative) is not str or not relative or '\\' in relative or '\x00' in relative:raise ValueError('safe runtime relative path required')
    p=PurePosixPath(relative)
    if p.is_absolute() or any(v in ('','..','.') for v in relative.split('/')):raise ValueError('safe runtime relative path required')
    current=ROOT
    for part in p.parts:
        current=current/part
        if current.is_symlink():raise ValueError('symlink runtime evidence')
    return current
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    with path.open('x') as f:f.write(json.dumps(value,sort_keys=True,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-freeze',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--execution-id',required=True)
    parser.add_argument('--mode',choices=('hand','full'),default='hand')
    args=parser.parse_args()
    output=safe_path(args.output_dir)
    if not output.is_relative_to(ROOT/'quality_reports'):raise ValueError('new quality_reports output prefix required')
    output.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();start=utc();clock=time.monotonic()
    receipt={'status':'FAIL_BEFORE_IMPORT','author_execution_id':'/root/formal_oracle',
        'actual_runner_execution_id':args.execution_id,'pid':os.getpid(),'started_at':start,
        'python':sys.version,'executable':sys.executable,'mode':args.mode,
        'scope':'Installed-source HAND/core suite only; no data approval or production validation.'}
    freeze=None
    try:
        raw=safe_path(args.runtime_freeze).read_bytes()
        if sha256(raw).hexdigest()!=args.expected_freeze_sha256:raise ValueError('expected runtime freeze SHA mismatch before imports')
        freeze=json.loads(raw)
        receipt['runtime_freeze_sha256']=sha256(raw).hexdigest()
        bindings=freeze['bindings']
        if type(bindings) is not dict:raise ValueError('runtime binding mapping required')
        required={'factory/rsm_controls.py','factory/pipeline.py','factory/exports.py','factory/export_validation.py',
            'oracles/independent.py','tests/rsm_core_integration_test.py',
            'factory/option_resume_reference.py','factory/reference.py',
            'generators/skills/prospective_option_resumption_attempt01/reference_attempt01.py',
            'tests/rsm_reference_routing_test.py',
            'reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/corpus_checker_attempt04.py',
            'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/option_resume_solver_v1.py',
            'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/legacy_contrast_v1.py',
            'schemas/RSM_OPTION_RESUMPTION_V1.md',
            'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md',
            'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md',
            'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md',
            'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md',
            'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json',
            'schemas/canonical/v1.schema.json','factory/common.py',
            'reviews/input_only/RSM-legacy-v0001.jsonl',
            'reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/actual-hand-candidate-attempt04.jsonl',
            'reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/hand-control-manifest.json',
            'quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/run_installed_suite_attempt01.py',
            'quality_reports/BELIEF_V4_FULL_SUITE_SOURCE_FREEZE_ATTEMPT01.json'}
        if not required<=set(bindings):raise ValueError('complete installed independent/reference/core freeze required')
        # Both exact staged proposals and installed destinations must be held.
        required.update('quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/'+p for p in ('factory/rsm_controls.py','factory/pipeline.py','factory/exports.py','factory/export_validation.py','oracles/independent.py','tests/rsm_core_integration_test.py'))
        required.update('quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/'+p for p in ('option_resume_reference.py','factory_reference_proposed.py','rsm_reference_routing_test.py'))
        if not required<=set(bindings):raise ValueError('staged proposals plus installed closure required before import')
        parent_raw=safe_path('quality_reports/BELIEF_V4_FULL_SUITE_SOURCE_FREEZE_ATTEMPT01.json').read_bytes()
        if sha256(parent_raw).hexdigest()!='ffb751bc7456f0a7df8d92f03d81798df97e91749d9e8a258ca84e1d0b2bd1c5':raise ValueError('legacy suite source freeze changed')
        parent=json.loads(parent_raw)
        if args.mode=='full':
            allowed_changes={'factory/pipeline.py','factory/exports.py','factory/export_validation.py','oracles/independent.py','factory/reference.py'}
            for path,b in parent['bindings'].items():
                if path not in bindings:raise ValueError('missing legacy runtime binding: '+path)
                if path not in allowed_changes and (bindings[path]['sha256']!=b['sha256'] or bindings[path]['bytes']!=b['bytes']):raise ValueError('held legacy source/data changed: '+path)
            actual_tests={str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('*.py') if p.name.endswith('_test.py') or p.name.startswith('test_')}
            actual_core={str(p.relative_to(ROOT)) for folder in ('factory','oracles') for p in (ROOT/folder).glob('*.py')}
            required.update(actual_tests|actual_core)
            closure=freeze.get('suite_source_paths')
            if type(closure) is not list or not closure or not all(type(v) is str for v in closure):raise ValueError('explicit complete suite source closure required')
            required.update(closure)
            if not required<=set(bindings):raise ValueError('unfrozen current test/core/suite dependency')
        for path,b in bindings.items():
            if type(b) is not dict or type(b.get('bytes')) is not int or b['bytes']<0:raise ValueError('typed runtime byte binding required')
            current=safe_path(path).read_bytes();archive=safe_path(b['archive']).read_bytes()
            if len(current)!=b['bytes'] or sha256(current).hexdigest()!=b['sha256'] or current!=archive:raise ValueError('runtime byte/archive mismatch: '+path)
        pairs=[('factory/'+p,'quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/factory/'+p) for p in ('rsm_controls.py','pipeline.py','exports.py','export_validation.py')]
        pairs += [('oracles/independent.py','quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/oracles/independent.py'),('tests/rsm_core_integration_test.py','quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/tests/rsm_core_integration_test.py')]
        pairs += [('factory/option_resume_reference.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/option_resume_reference.py'),('factory/reference.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/factory_reference_proposed.py'),('tests/rsm_reference_routing_test.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/rsm_reference_routing_test.py')]
        for installed,proposed in pairs:
            if safe_path(installed).read_bytes()!=safe_path(proposed).read_bytes():raise ValueError('installed source differs from prepared proposal: '+installed)
        receipt['verified_preimport_bindings']=len(bindings)
        sys.path.insert(0,str(ROOT))
        if args.mode=='hand':
            filename=ROOT/'tests/rsm_core_integration_test.py'
            spec=importlib.util.spec_from_file_location('frozen_rsm_core_hand_tests',filename)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            suite=unittest.defaultTestLoader.loadTestsFromModule(module)
            ref_filename=ROOT/'tests/rsm_reference_routing_test.py'
            ref_spec=importlib.util.spec_from_file_location('frozen_root_rsm_reference_hand_tests',ref_filename)
            ref_module=importlib.util.module_from_spec(ref_spec);ref_spec.loader.exec_module(ref_module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(ref_module))
        else:
            modules=parent['modules']+['tests.rsm_core_integration_test','tests.rsm_reference_routing_test']
            if len(parent['modules'])!=38 or len(set(modules))!=len(modules):raise ValueError('exact distinct legacy38 plus RSM2 module list required')
            if actual_tests!={name.replace('.','/')+'.py' for name in modules}:raise ValueError('actual test files differ from explicit complete suite list')
            receipt['actual_test_modules']=modules
            suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        receipt.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),status='PASS' if result.wasSuccessful() else 'FAIL_TESTS')
        if args.mode=='hand':save(output/'HAND_CASES.json',{'formal_module':module.RESULTS,'root_reference_module':getattr(ref_module,'RESULTS',{'available':False,'per_test_results':'FULL_OUTPUT.txt'})})
    except BaseException:
        traceback.print_exc(file=stream)
        if receipt.get('status')=='PASS':receipt['status']='FAIL_POST_TEST_EVIDENCE'
    finally:
        receipt['finished_at']=utc();receipt['elapsed_seconds']=round(time.monotonic()-clock,6)
        changed=[]
        if type(freeze) is dict and type(freeze.get('bindings')) is dict:
            for path,b in freeze['bindings'].items():
                try:
                    raw=safe_path(path).read_bytes()
                    if len(raw)!=b['bytes'] or sha256(raw).hexdigest()!=b['sha256']:changed.append(path)
                except Exception:changed.append(path)
        receipt['changed_runtime_bindings']=changed
        if changed:receipt['status']='FAIL_SOURCE_CHANGED'
        raw=stream.getvalue().encode()
        with (output/'FULL_OUTPUT.txt').open('xb') as f:f.write(raw)
        receipt['full_output_sha256']=sha256(raw).hexdigest();receipt['full_output_bytes']=len(raw)
        save(output/'EXECUTION.json',receipt)
        print(json.dumps({k:receipt.get(k) for k in ('status','actual_runner_execution_id','pid','tests_run','failures','errors','elapsed_seconds','runtime_freeze_sha256')},sort_keys=True))
    return 0 if receipt['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
