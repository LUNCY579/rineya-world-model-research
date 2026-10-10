"""Additive actual runner with independent preexecution byte freeze."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
frozen = json.loads((HERE / 'preexecution-freeze-attempt02.json').read_text())
for row in frozen['files']:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
start = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
command = ['/opt/anaconda3/bin/python3', str(HERE / 'hand_tests_attempt02.py')]
result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='9001'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(HERE / 'attempt02-output.txt').write_bytes(result.stdout)
unchanged = all(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in frozen['files'])
receipt = {'actual_author_and_runner': '/root/formal_oracle', 'actual_runner_pid': os.getpid(),
           'execution_id': 'RSM-option-resumption-additive-attempt02-20261010T083826Z',
           'actual_command': command, 'started_at_utc': start, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
           'wall_seconds': time.monotonic() - clock, 'exit_code': result.returncode,
           'all_source_and_first_evidence_bytes_unchanged': unchanged,
           'output_sha256': hashlib.sha256(result.stdout).hexdigest(),
           'role': 'Nonbinding prospective formal feasibility author', 'production_data_read': False, 'approval': None}
(HERE / 'attempt02-execution.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, indent=2, sort_keys=True)); print(result.stdout.decode())
raise SystemExit(result.returncode if unchanged else 2)
