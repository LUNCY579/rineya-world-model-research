"""Frozen actual focused handmade integration runner; no production validation."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
here=Path(__file__).resolve().parent;base=here.parents[2]
paths=['factory/belief_controls.py','factory/pipeline.py','tests/test_belief_v4_factory_bridge.py','oracles/belief_controls_v4_independent.py','tests/oracle_belief_controls_v4_test.py']
before={path:hashlib.sha256((base/path).read_bytes()).hexdigest() for path in paths}
started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='17')
command=[sys.executable,str(base/'tests/test_belief_v4_factory_bridge.py')]
process=subprocess.run(command,cwd=base,env=env,capture_output=True)
raw=process.stdout+b'\n--- ACTUAL STDERR ---\n'+process.stderr
(here/'focused-attempt01-output.txt').write_bytes(raw)
counts=None
for line in process.stdout.decode('utf8').splitlines():
    if line.startswith('ACTUAL_V4_BRIDGE_HANDMADE_COUNTS '):counts=json.loads(line.partition(' ')[2])
after={path:hashlib.sha256((base/path).read_bytes()).hexdigest() for path in paths}
receipt={'execution_id':'RBM-v4-factory-integration-attempt01-20261010T080702Z','actual_author_and_runner':'/root/formal_oracle','role':'formal checker/factory integration author, no approval','actual_runner_pid':os.getpid(),'actual_command':command,'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-timer,'exit_code':process.returncode,'source_before':before,'source_after':after,'all_source_bytes_unchanged':before==after,'actual_output_sha256':hashlib.sha256(raw).hexdigest(),'observed_handmade_counts':counts,'production_validation_run':False,'full_suite_run':False,'official25_repeated':False,'scope':'New focused handmade integration tests only; root separately authorized full suite after FINAL.'}
(here/'focused-attempt01-execution.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True));sys.exit(process.returncode or (before!=after))
