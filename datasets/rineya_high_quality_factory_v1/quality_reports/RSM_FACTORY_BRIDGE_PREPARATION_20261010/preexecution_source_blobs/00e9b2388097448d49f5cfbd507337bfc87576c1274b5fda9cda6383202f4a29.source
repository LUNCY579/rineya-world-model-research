"""Frozen isolated HAND runner; captures even pre-test import failures."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def utc(): return datetime.now(timezone.utc).isoformat()
def write_json(path,value):
    with path.open('x') as f: f.write(json.dumps(value,sort_keys=True,indent=2)+'\n')

start=utc(); clock=time.monotonic(); output=io.StringIO(); result=None; module=None
receipt={'author_execution_id':'/root/formal_oracle','runner_execution_id':'/root/formal_oracle',
 'pid':os.getpid(),'python':sys.version,'python_executable':sys.executable,'started_at':start,
 'scope':'Isolated original HAND216 only; no production, shared-core installation, approval, or remote proof.',
 'reference_agreement':'NOT_RUN_SEPARATE_AUTHOR_REQUIRED','status':'FAIL_BEFORE_TESTS'}
try:
    freeze_path=HERE/'SOURCE_FREEZE_ATTEMPT01.json'; raw=freeze_path.read_bytes()
    receipt['preexecution_freeze_sha256']=sha256(raw).hexdigest(); freeze=json.loads(raw)
    for path,b in freeze['bindings'].items():
        current=(ROOT/path).read_bytes(); archived=(HERE/b['archive']).read_bytes()
        assert len(current)==b['bytes'] and sha256(current).hexdigest()==b['sha256'],('source changed before execution',path)
        assert archived==current,('source archive mismatch',path)
    receipt['verified_preimport_bindings']=len(freeze['bindings'])
    spec=importlib.util.spec_from_file_location('rsm_bridge_hand_tests_attempt01',HERE/'bridge_tests_attempt01.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    receipt.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    receipt['status']='PASS' if result.wasSuccessful() else 'FAIL_TESTS'
    write_json(HERE/'ACTUAL_HAND_CASE_RESULTS_ATTEMPT01.json',module.RESULTS)
    if result.wasSuccessful():
        package_path=HERE/'actual-source-package-attempt01'
        package=module.B.package_sources(package_path)
        proof=module.B.verify_package(package_path)
        receipt['actual_package_proof']=proof
        receipt['actual_package_ledger_sha256']=sha256((package_path/'SOURCE_PACKAGE.json').read_bytes()).hexdigest()
        receipt['actual_package_bindings']=package['source_bindings']
except BaseException:
    traceback.print_exc(file=output)
    if receipt.get('status')=='PASS': receipt['status']='FAIL_POST_TEST_PACKAGE'
finally:
    receipt['finished_at']=utc(); receipt['elapsed_seconds']=round(time.monotonic()-clock,6)
    post={}
    if 'freeze' in globals():
        for path,b in freeze['bindings'].items():
            try:
                current=(ROOT/path).read_bytes(); h=sha256(current).hexdigest()
                post[path]={'sha256':h,'bytes':len(current),'unchanged':h==b['sha256'] and len(current)==b['bytes']}
            except OSError as e: post[path]={'unchanged':False,'error':str(e)}
        receipt['postexecution_bindings']=post
        if any(not b['unchanged'] for b in post.values()): receipt['status']='FAIL_SOURCE_CHANGED'
    full=output.getvalue().encode()
    with (HERE/'FULL_EXECUTION_OUTPUT_ATTEMPT01.txt').open('xb') as f: f.write(full)
    receipt['full_output_sha256']=sha256(full).hexdigest(); receipt['full_output_bytes']=len(full)
    write_json(HERE/'EXECUTION_ATTEMPT01.json',receipt)
    print(json.dumps({k:receipt.get(k) for k in ('status','pid','tests_run','failures','errors','elapsed_seconds','preexecution_freeze_sha256','full_output_sha256')},sort_keys=True))
sys.exit(0 if receipt['status']=='PASS' else 1)
