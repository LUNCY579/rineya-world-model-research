"""Frozen first execution harness. Verify bytes before loading any new source."""
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import unittest

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
def now(): return datetime.now(timezone.utc).isoformat()
def h(p): return sha256(p.read_bytes()).hexdigest()
start=now(); tick=time.monotonic(); freeze=json.loads((HERE/'PREEXECUTION_FREEZE_ATTEMPT03.json').read_bytes())
for b in freeze['bindings']:
    p=ROOT/b['path']
    if p.stat().st_size!=b['bytes'] or h(p)!=b['sha256']: raise RuntimeError('pre-import binding mismatch '+b['path'])
spec=importlib.util.spec_from_file_location('rsm_corpus_public_tests',HERE/'corpus_tests_attempt03.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
stream=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(HERE/'unittest-output-attempt03.txt').write_text(stream.getvalue())
(HERE/'test-results-attempt03.json').write_text(json.dumps(module.RESULTS,sort_keys=True,indent=2)+'\n')
unchanged=all((ROOT/b['path']).stat().st_size==b['bytes'] and h(ROOT/b['path'])==b['sha256'] for b in freeze['bindings'])
receipt={'actual_author_execution_id':'/root/formal_oracle','actual_runner_execution_id':'/root/formal_oracle','pid':os.getpid(),
 'started_utc':start,'finished_utc':now(),'elapsed_seconds':time.monotonic()-tick,'preexecution_freeze_sha256':h(HERE/'PREEXECUTION_FREEZE_ATTEMPT03.json'),
 'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),
 'all_preexecution_bindings_unchanged':unchanged,'production_inputs_read':False,'data_approval':False,
 'dependency_reuse':'Own427262 task solver, ownbb628 graph views, owncdc1 hand programs. Common model/author correlation disclosed.',
 'execution_scope':'Handwritten complete216 fixture only, exact target-free24 legacy bytes for novelty; no producer or canonical production inputs.',
 'outputs':[{ 'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h(p)} for p in sorted(HERE.glob('actual-*-attempt03.*')) if p.is_file()]+[
 {'path':str((HERE/n).relative_to(ROOT)),'bytes':(HERE/n).stat().st_size,'sha256':h(HERE/n)} for n in ('unittest-output-attempt03.txt','test-results-attempt03.json')]}
(HERE/'EXECUTION_ATTEMPT03.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(stream.getvalue()); print(json.dumps(receipt,sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and unchanged else 1)
