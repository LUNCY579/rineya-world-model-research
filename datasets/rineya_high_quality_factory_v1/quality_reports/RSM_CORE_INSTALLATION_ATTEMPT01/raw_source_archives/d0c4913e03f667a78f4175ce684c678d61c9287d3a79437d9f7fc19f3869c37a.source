"""Execute the complete CPU dataset test suite with pre/post source custody."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes(bindings):
    result = {}
    for name in bindings:
        path = ROOT / name
        data = path.read_bytes()
        result[name] = {'bytes': len(data), 'sha256': sha(data)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    fb = (ROOT / args.freeze).read_bytes()
    assert sha(fb) == args.expected_freeze_sha256, 'Unbound execution freeze'
    frozen = json.loads(fb)
    before = hashes(frozen['bindings'])
    assert before == frozen['bindings'], 'Frozen source/input bytes changed before execution'
    assert before[str(Path(__file__).relative_to(ROOT))]['sha256'] == sha(Path(__file__).read_bytes())
    output = ROOT / frozen['output']
    receipt = ROOT / frozen['receipt']
    assert not output.exists() and not receipt.exists(), 'Prior execution evidence must be preserved'
    command = ['/opt/anaconda3/bin/python3', '-m', 'unittest', *frozen['modules'], '-v']
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '.:tests'})
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    with output.open('xb') as out:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    after = hashes(frozen['bindings'])
    data = output.read_bytes()
    record = {
        'actual_runner_execution': '/root', 'started_at_utc': started_at,
        'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Actual full CPU dataset suite after V4 factory integration; no data approval or model training',
        'source_sha256': sha(Path(__file__).read_bytes()), 'freeze_sha256': sha(fb),
        'command': command, 'environment_overrides': {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '.:tests'},
        'source_and_input_sha256_before': before, 'source_and_input_sha256_after': after,
        'bound_sources_and_inputs_unchanged': before == after,
        'exit_code': process.returncode, 'wall_seconds': time.monotonic() - started,
        'output_path': frozen['output'], 'output_bytes': len(data), 'output_sha256': sha(data),
        'actual_output_tail': data.decode(errors='replace')[-7000:],
        'status': 'PASS' if process.returncode == 0 and before == after else 'FAILED_PRESERVED_EXECUTION',
        'data_approval': False, 'model_training': False,
    }
    with receipt.open('x') as out:
        json.dump(record, out, sort_keys=True, indent=2)
        out.write('\n')
    print(json.dumps({k: record[k] for k in ['status', 'exit_code', 'wall_seconds', 'output_sha256', 'bound_sources_and_inputs_unchanged']}))
    if record['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
