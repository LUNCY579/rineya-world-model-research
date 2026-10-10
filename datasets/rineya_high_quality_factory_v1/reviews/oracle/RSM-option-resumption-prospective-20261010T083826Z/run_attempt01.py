"""Actual isolated runner; requires the preexecution byte freeze before import."""
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
    actual = (ROOT / row['path']).read_bytes()
    assert hashlib.sha256(actual).hexdigest() == row['sha256'], row['path']
started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
command = ['/opt/anaconda3/bin/python3', str(HERE / 'hand_tests_attempt01.py')]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='17')
result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output = HERE / 'attempt01-output.txt'; output.write_bytes(result.stdout)
unchanged = all(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in frozen['files'])
receipt = {'actual_author_and_runner': '/root/formal_oracle', 'role': 'Nonbinding public-only RSM feasibility author',
           'execution_id': 'RSM-option-resumption-hand-attempt01-20261010T083826Z',
           'actual_runner_pid': os.getpid(), 'started_at_utc': started,
           'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': time.monotonic() - clock,
           'actual_command': command, 'exit_code': result.returncode,
           'all_source_and_public_bytes_unchanged': unchanged,
           'output_sha256': hashlib.sha256(result.stdout).hexdigest(),
           'production_data_read': False, 'data_approval': None}
(HERE / 'attempt01-execution.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, indent=2, sort_keys=True)); print(result.stdout.decode())
raise SystemExit(result.returncode if unchanged else 2)
