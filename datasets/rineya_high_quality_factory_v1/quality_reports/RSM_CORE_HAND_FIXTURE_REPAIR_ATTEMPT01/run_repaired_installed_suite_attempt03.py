"""Repaired future root-run installed-source runner. PREPARATION ONLY until grant/freeze.

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
        'scope':'Installed-source schema-complete derived HAND/core suite only; no data approval or production validation.'}
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
            'quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/run_installed_suite_attempt03.py',
            'quality_reports/BELIEF_V4_FULL_SUITE_SOURCE_FREEZE_ATTEMPT01.json'}
        if not required<=set(bindings):raise ValueError('complete installed independent/reference/core freeze required')
        # Both exact staged proposals and installed destinations must be held.
        required.update('quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/'+p for p in ('factory/rsm_controls.py','factory/pipeline.py','factory/exports.py','factory/export_validation.py','oracles/independent.py','tests/rsm_core_integration_test.py'))
        required.update('quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/'+p for p in ('option_resume_reference.py','factory_reference_proposed.py','rsm_reference_routing_test.py'))
        if not required<=set(bindings):raise ValueError('staged proposals plus installed closure required before import')
        repair='quality_reports/RSM_CORE_HAND_FIXTURE_REPAIR_ATTEMPT01/'
        parent_install_path='quality_reports/RSM_CORE_INSTALLATION_ATTEMPT01/INSTALLED_RUNTIME_SOURCE_FREEZE.json'
        required.update({parent_install_path,repair+'run_repaired_installed_suite_attempt03.py',
            repair+'staged_attempt03/tests/rsm_core_integration_test.py',
            repair+'HAND_SCHEMA_COMPLETE_PENDING_ATTEMPT01.jsonl',
            repair+'HAND_SCHEMA_COMPLETE_CONTROL_MANIFEST_ATTEMPT01.json',
            repair+'HAND_FIXTURE_TRANSFORMATION_RECEIPT_ATTEMPT01.json',
            repair+'SOURCE_PREPARED_CONFIG_ATTEMPT03.json',repair+'SOURCE_PREEXECUTION_FREEZE_ATTEMPT03.json'})
        parent_install_raw=safe_path(parent_install_path).read_bytes()
        if sha256(parent_install_raw).hexdigest()!='e8db5c7569076e03d2a67f2616fe2e0e4db47c35ac8a8050f1ff80210204ec60':
            raise ValueError('exact failed-run463 installed parent freeze required before imports')
        installed_parent=json.loads(parent_install_raw)
        if len(installed_parent['bindings'])!=463:raise ValueError('exact installed parent463 closure required')
        required.update(installed_parent['bindings'])
        if not required<=set(bindings):raise ValueError('complete parent463 and repair closure required in both modes')
        # The sole replacement is the actual installed formal test. Every other
        # installed parent binding remains exact in HAND as well as FULL mode.
        for path,b in installed_parent['bindings'].items():
            old_archive=safe_path(b['archive']).read_bytes()
            if len(old_archive)!=b['bytes'] or sha256(old_archive).hexdigest()!=b['sha256']:
                raise ValueError('old installed parent archive changed: '+path)
            if path=='tests/rsm_core_integration_test.py':
                if {k:bindings[path][k] for k in ('sha256','bytes')}!={'sha256':'8895cacd679ccfadcba7edf553fc37e6263df8b6ed330dc26d049dda5636b23b','bytes':24621}:
                    raise ValueError('only exact source-prepared repaired formal test is allowed')
            elif bindings[path]['sha256']!=b['sha256'] or bindings[path]['bytes']!=b['bytes']:
                raise ValueError('held installed parent463 source changed: '+path)
        if freeze.get('legacy_modules')!=installed_parent['legacy_modules'] or freeze.get('new_modules')!=installed_parent['new_modules']:
            raise ValueError('exact previous38 plus2 module inventory must remain unchanged')
        receipt['installed_parent_freeze_sha256']=sha256(parent_install_raw).hexdigest()
        receipt['installed_parent_bindings']=463
        receipt['only_parent_binding_replaced']='tests/rsm_core_integration_test.py'
        repaired_parent_path='quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/INSTALLED_RUNTIME_SOURCE_FREEZE.json'
        required.add(repaired_parent_path)
        repaired_parent_raw=safe_path(repaired_parent_path).read_bytes()
        if sha256(repaired_parent_raw).hexdigest()!='a94f4a79ca03f6367ad6320e9f09fc2e81018ed8ca0edbc81bd68c03286b5d95':
            raise ValueError('exact executed repaired parent489 freeze required before imports')
        repaired_parent=json.loads(repaired_parent_raw)
        if len(repaired_parent['bindings'])!=489:raise ValueError('exact executed parent489 closure required')
        required.update(repaired_parent['bindings'])
        if not required<=set(bindings):raise ValueError('complete executed parent489 closure required in both modes')
        for path,b in repaired_parent['bindings'].items():
            old_archive=safe_path(b['archive']).read_bytes()
            if len(old_archive)!=b['bytes'] or sha256(old_archive).hexdigest()!=b['sha256']:
                raise ValueError('executed repaired parent archive changed: '+path)
            if path!='tests/rsm_core_integration_test.py' and (bindings[path]['sha256']!=b['sha256'] or bindings[path]['bytes']!=b['bytes']):
                raise ValueError('held executed parent489 source changed: '+path)
        receipt['executed_repaired_parent_freeze_sha256']=sha256(repaired_parent_raw).hexdigest()
        receipt['executed_repaired_parent_bindings']=489
        parent_raw=safe_path('quality_reports/BELIEF_V4_FULL_SUITE_SOURCE_FREEZE_ATTEMPT01.json').read_bytes()
        if sha256(parent_raw).hexdigest()!='ffb751bc7456f0a7df8d92f03d81798df97e91749d9e8a258ca84e1d0b2bd1c5':raise ValueError('legacy suite source freeze changed')
        parent=json.loads(parent_raw)
        allowed_changes={'factory/pipeline.py','factory/exports.py','factory/export_validation.py','oracles/independent.py','factory/reference.py'}
        for path,b in parent['bindings'].items():
            if path not in bindings:raise ValueError('missing legacy runtime binding: '+path)
            if path=='manifests/ACTIVE_RELEASE_CATALOG.json':
                old_catalog={'sha256':'85a6b220aa0996717886fca76794aa2e1298a89047effb98d787c03094b2a852','bytes':6672}
                new_catalog={'sha256':'5d90edad2e9af72e4ed5be382efac03447a83bbb6d79aba08103038a24d3bbff','bytes':7001}
                if b!=old_catalog or {k:bindings[path][k] for k in ('sha256','bytes')}!=new_catalog:raise ValueError('unapproved catalog evolution')
            elif path not in allowed_changes and (bindings[path]['sha256']!=b['sha256'] or bindings[path]['bytes']!=b['bytes']):raise ValueError('held legacy source/data changed: '+path)
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
        pairs += [('oracles/independent.py','quality_reports/RSM_CORE_SOURCE_PREPARATION_20261010T1210Z/staged/oracles/independent.py'),('tests/rsm_core_integration_test.py','quality_reports/RSM_CORE_HAND_FIXTURE_REPAIR_ATTEMPT01/staged_attempt03/tests/rsm_core_integration_test.py')]
        pairs += [('factory/option_resume_reference.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/option_resume_reference.py'),('factory/reference.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/factory_reference_proposed.py'),('tests/rsm_reference_routing_test.py','quality_reports/RSM_ROOT_REFERENCE_ROUTING_PREPARATION_20261010/rsm_reference_routing_test.py')]
        for installed,proposed in pairs:
            if safe_path(installed).read_bytes()!=safe_path(proposed).read_bytes():raise ValueError('installed source differs from prepared proposal: '+installed)
        if parent['modules']!=installed_parent['legacy_modules']:raise ValueError('parent legacy38 lists disagree')
        all_modules=parent['modules']+installed_parent['new_modules']
        if len(parent['modules'])!=38 or len(all_modules)!=40 or len(set(all_modules))!=40:
            raise ValueError('exact distinct legacy38 plus unchanged RSM2 list required before imports')
        if actual_tests!={name.replace('.','/')+'.py' for name in all_modules}:
            raise ValueError('actual test files differ from explicit complete40 suite list before imports')
        receipt['actual_full_suite_module_inventory']=all_modules
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
            modules=all_modules
            if len(parent['modules'])!=38 or len(set(modules))!=len(modules):raise ValueError('exact distinct legacy38 plus RSM2 module list required')
            if actual_tests!={name.replace('.','/')+'.py' for name in modules}:raise ValueError('actual test files differ from explicit complete suite list')
            receipt['actual_test_modules']=modules
            suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        receipt.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),status='PASS' if result.wasSuccessful() else 'FAIL_TESTS')
        if args.mode=='hand':
            save(output/'HAND_CASES.json',{'formal_module':module.RESULTS,'root_reference_module':getattr(ref_module,'RESULTS',{'available':False,'per_test_results':'FULL_OUTPUT.txt'})})
            receipt['actual_test_modules']=['tests.rsm_core_integration_test','tests.rsm_reference_routing_test']
            receipt['per_module_test_counts']={
                'tests.rsm_core_integration_test':unittest.defaultTestLoader.loadTestsFromModule(module).countTestCases(),
                'tests.rsm_reference_routing_test':unittest.defaultTestLoader.loadTestsFromModule(ref_module).countTestCases()}
        else:
            # Reuses already imported modules; this does not run HAND tests twice.
            save(output/'FULL_MODULE_RESULTS.json',{name:getattr(sys.modules[name],'RESULTS',{'available':False,'per_test_results':'FULL_OUTPUT.txt'})
                for name in installed_parent['new_modules']})
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
