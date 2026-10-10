"""Actual runner: source freeze verification occurs before test imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
frozen = json.loads((HERE / 'preexecution-freeze-attempt01.json').read_text())
for row in frozen['files']:
    raw = (ROOT / row['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
    assert raw == (ROOT / row['archive']).read_bytes(), row['archive']
started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
command = ['/opt/anaconda3/bin/python3', str(HERE / 'task_solver_tests_attempt01.py')]
result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='17'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(HERE / 'attempt01-output.txt').write_bytes(result.stdout)
unchanged = all(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in frozen['files'])
receipt = {'actual_author_and_runner': '/root/formal_oracle', 'actual_runner_pid': os.getpid(),
           'execution_id': 'RSM-official-task-solver-attempt01-20261010T0932Z',
           'role': 'Official isolated task semantics author; corpus controls pending, no approval',
           'actual_command': command, 'started_at_utc': started,
           'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - clock,
           'exit_code': result.returncode, 'all_frozen_sources_and_inputs_unchanged': unchanged,
           'output_sha256': hashlib.sha256(result.stdout).hexdigest(),
           'future_production_data_or_gold_read': False, 'actual_target_free_legacy_inputs_read': True,
           'data_approval': None}
(HERE / 'attempt01-execution.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, indent=2, sort_keys=True)); print(result.stdout.decode())
raise SystemExit(result.returncode if unchanged else 2)
