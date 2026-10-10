"""Prospective hand counterexamples, frozen before targeted execution.
No changed semantics: malformed inputs must return errors, and an unused
callee's enter binding alone cannot prove a saved pending argument dependency.
"""
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import unittest

HERE=Path(__file__).parent; ROOT=HERE.parents[2]
freeze=json.loads((HERE/'TARGETED_SELF_AUDIT_FREEZE01.json').read_bytes())
for b in freeze['bindings']:
    p=ROOT/b['path']; assert sha256(p.read_bytes()).hexdigest()==b['sha256']
def local(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
C=local('corpus_checker_attempt02'); H=local('hand_fixture_recipe_attempt01')
start=datetime.now(timezone.utc).isoformat(); tick=time.monotonic()
class FinalSelfAudit(unittest.TestCase):
    def test_nonobject_row_returns_failures(self): self.assertIsInstance(C.check(None),list)
    def test_unhashable_coordinate_returns_failures(self):
        b=H.build(C); row=deepcopy(b['rows'][0]); row['source_family_index']={}; row['record_hash']=C.record_hash(row)
        self.assertIsInstance(C.check_batch([row])['errors'],list)
    def test_unused_callee_argument_does_not_prove_pending_saved_dependency(self):
        b=H.build(C); base=deepcopy(b['rows'][54]['semantic_task'])
        base['options']['leaf']['body'][0]['value']={'const':1}
        base['options']['root']['body'][-1]['value']={'const':2}
        tasks=H.make_contexts(C,base)
        with self.assertRaisesRegex(ValueError,'pending work'): C.family_analysis(tasks)

stream=io.StringIO(); r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalSelfAudit))
(HERE/'targeted-self-audit-output-attempt01.txt').write_text(stream.getvalue())
receipt={'actual_author_execution_id':'/root/formal_oracle','actual_runner_execution_id':'/root/formal_oracle','pid':os.getpid(),'started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-tick,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),'source_sha256':sha256((HERE/'corpus_checker_attempt02.py').read_bytes()).hexdigest(),'test_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'freeze_sha256':sha256((HERE/'TARGETED_SELF_AUDIT_FREEZE01.json').read_bytes()).hexdigest(),'output_sha256':sha256(stream.getvalue().encode()).hexdigest(),'production_data_approval':False}
(HERE/'TARGETED_SELF_AUDIT_EXECUTION01.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(stream.getvalue()); print(json.dumps(receipt,sort_keys=True)); raise SystemExit(0 if r.wasSuccessful() else 1)
