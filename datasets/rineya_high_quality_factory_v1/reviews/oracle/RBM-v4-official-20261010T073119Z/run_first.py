"""Actual official handmade test runner, frozen before execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
source=here/'belief_controls_v4_independent.py'; tests=here/'oracle_belief_controls_v4_test.py'
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='17')
command=[sys.executable,str(tests)]
started=datetime.now(timezone.utc).isoformat(); timer=time.perf_counter()
process=subprocess.run(command,cwd=here.parents[2],env=env,capture_output=True)
raw=process.stdout+b'\n--- ACTUAL STDERR ---\n'+process.stderr
(here/'first-test-output.txt').write_bytes(raw)
counts=None
for line in process.stdout.decode('utf8').splitlines():
    if line.startswith('ACTUAL_V4_HANDMADE_COUNTS '):counts=json.loads(line.partition(' ')[2])
receipt={'execution_id':'RBM-v4-official-first-20261010T073119Z','actual_author_and_runner':'/root/formal_oracle','role':'independent formal checker author; no approval','runner_pid':os.getpid(),'actual_test_process_pid_available':False,'actual_command':command,'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-timer,'exit_code':process.returncode,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tests_sha256':hashlib.sha256(tests.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(raw).hexdigest(),'observed_counts':counts,'production_data_run':False,'common_model_and_own_prior_implementation_reuse':True}
(here/'first-execution.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
sys.exit(process.returncode)
