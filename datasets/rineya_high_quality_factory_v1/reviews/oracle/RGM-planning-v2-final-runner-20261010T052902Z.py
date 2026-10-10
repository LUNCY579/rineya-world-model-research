"""Actual prospective public-only formal runner; /root/formal_oracle."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import unittest

BASE = Path(__file__).resolve().parents[2]
os.chdir(BASE)
sys.path[:0] = [str(BASE), str(BASE/'tests')]
import oracle_rgm_planning_v2_test as fixtures
from oracles import rgm_planning_controls_v2_independent as oracle

def sha(data): return hashlib.sha256(data).hexdigest()
dependencies = ('oracles/rgm_planning_controls_v2_independent.py',
    'tests/oracle_rgm_planning_v2_test.py', 'oracles/rgm_planning_controls_independent.py',
    'tests/oracle_rgm_planning_controls_test.py', 'oracles/independent.py', 'oracles/__init__.py')
hashes = {p:sha((BASE/p).read_bytes()) for p in dependencies}
runner_path = str(Path(__file__).resolve().relative_to(BASE))
hashes[runner_path] = sha(Path(__file__).read_bytes())
for p,digest in hashes.items():
    blob = BASE/'reproducibility/source_blobs'/(digest+'.source')
    raw = (BASE/p).read_bytes()
    if blob.exists(): assert blob.read_bytes() == raw
    else: blob.write_bytes(raw)
public = {}
for p,expected in oracle.SPECIFICATIONS.items():
    public[p] = sha((BASE/p).read_bytes())
    assert public[p] == expected, (p, public[p], expected)
for p,expected in {
    'research/curricula/RGM-PLANNING-GAP-V2-ROOT_FREEZE.json':'35259b023c0fa70cf4d044a4d7e1a0d5143f26ed5bab5c1f6646ce1ce49402b9',
    'research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json':'7759940fcb0412f67f02c620cc40f52216ca7cbfb695e4f2f5690e99b12f2dbd'}.items():
    public[p] = sha((BASE/p).read_bytes()); assert public[p] == expected
stream = io.StringIO()
started = time.monotonic()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromModule(fixtures))
elapsed = time.monotonic()-started
code = 'import sys,json;sys.path.insert(0,"tests");from oracle_rgm_planning_v2_test import family;from oracles.rgm_planning_controls_v2_independent import check_batch;rows,m=family(204);print(json.dumps(check_batch(rows,m,[],[]),sort_keys=True,separators=(",",":")))'
replays = [subprocess.check_output([sys.executable,'-c',code],
    env=dict(os.environ,PYTHONHASHSEED=seed,PYTHONDONTWRITEBYTECODE='1')) for seed in ('1','987')]
assert replays[0] == replays[1]
raw_path = BASE/'reviews/oracle/RGM-planning-v2-handmade-diagnostic-20261010T052902Z.json'
assert not raw_path.exists()
raw_path.write_bytes(replays[0])
assert all(sha((BASE/p).read_bytes()) == digest for p,digest in hashes.items())
receipt = {'actual_execution_id':'/root/formal_oracle',
    'role':'Independent prospective formal author/runner; no data approval',
    'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python_executable':sys.executable,'python_version':sys.version,
    'runner_path':runner_path,'source_sha256':hashes,'public_sha256':public,
    'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'elapsed_seconds':elapsed,'counts':fixtures.COUNTS,'output':stream.getvalue(),
    'source_unchanged_after_run':True,
    'diagnostic_replays':{'seeds':['1','987'],'full_json_byte_equal':True,
        'raw_path':str(raw_path.relative_to(BASE)),'raw_sha256':sha(replays[0]),'raw_bytes':len(replays[0])},
    'generator_version_interface':'Nonempty strict string, one coherent cohort version; no prescribed spelling. Compare freshness against parent generator_version only when present in supplied public parent metadata. Explicit source/version/hash custody belongs in caller execution receipts.',
    'shared_implementation':'V1 finite search, validation, isomorphism, control and novelty gates; internal immutable input-only V1 projection. New V2 guard replay, clearance, selective mutants, no-op controls and actual renderer pair checks.',
    'read_boundary':'Frozen public V1/V2 docs/addendum and own frozen V1 source/tests only. No producer/candidate/gold/root gates/private verdicts or actual parent/prior input views read.',
    'limitations':'Handmade partial family batches retain global cohort/novelty failures. These executions establish tested mechanics and fail-closed controls; they do not establish production408/204/48 closure, actual novelty, actual input custody or approve data.',
    'history':'Prior18-test and24-test sources/execution receipts remain immutable; version-string boundary correction is new final prospective source.'}
output = BASE/'reviews/oracle/RGM-planning-v2-final-execution-20261010T052902Z.json'
assert not output.exists()
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('tests','failures','errors','elapsed_seconds','counts','diagnostic_replays')},indent=2))
print('receipt',str(output.relative_to(BASE)),sha(output.read_bytes()))
if not result.wasSuccessful(): raise SystemExit(1)
