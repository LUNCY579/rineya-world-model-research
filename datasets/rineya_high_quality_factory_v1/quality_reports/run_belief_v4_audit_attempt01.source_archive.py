"""Run the frozen whole-cohort V4 machine audit after actual full-suite PASS."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def binding(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    fb = (ROOT / args.freeze).read_bytes()
    assert hashlib.sha256(fb).hexdigest() == args.expected_freeze_sha256
    frozen = json.loads(fb)
    before = {name: binding(ROOT / name) for name in frozen['bindings']}
    assert before == frozen['bindings'], 'Source or required input changed before audit'
    assert binding(Path(__file__)) == before[str(Path(__file__).relative_to(ROOT))]
    suite = json.loads((ROOT / frozen['passed_full_suite_receipt']).read_bytes())
    assert suite['status'] == 'PASS' and suite['bound_sources_and_inputs_unchanged'] is True
    for name in [frozen['output'], frozen['receipt'], *frozen['audit_outputs']]:
        assert not (ROOT / name).exists(), 'Refusing to overwrite earlier audit evidence: ' + name
    command = ['/opt/anaconda3/bin/python3', '-m', 'factory.pipeline', 'audit', '--batch', 'belief-gap-v4']
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '.:tests'})
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    with (ROOT / frozen['output']).open('xb') as out:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    after = {name: binding(ROOT / name) for name in frozen['bindings']}
    outputs = {name: binding(ROOT / name) for name in frozen['audit_outputs'] if (ROOT / name).exists()}
    gates = False
    stats = None
    preservation = None
    errors = []
    try:
        stats = json.loads((ROOT / 'audits/leakage/belief-gap-v4.json').read_bytes())
        report = json.loads((ROOT / 'audits/diversity/belief-gap-v4-belief-controls.json').read_bytes())['report']
        rows = [json.loads(line) for line in (ROOT / 'reviews/oracle/belief-gap-v4.jsonl').read_bytes().splitlines()]
        preservation = report['v4_preservation']
        gates = (stats['records'] == 3456 and stats['unique_parents'] == 4
                 and stats['all_machine_quality_gates_pass'] is True
                 and len(rows) == 3456 and len({row['sample_id'] for row in rows}) == 3456
                 and all(row['pass'] is True for row in rows)
                 and not report['errors'] and preservation['status'] == 'PROVED')
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    record = {
        'actual_runner_execution': '/root', 'started_at_utc': started_at,
        'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': binding(Path(__file__)), 'source_freeze_sha256': args.expected_freeze_sha256,
        'command': command, 'exit_code': proc.returncode, 'wall_seconds': time.monotonic() - start,
        'bound_source_and_input_bytes_before': before, 'bound_source_and_input_bytes_after': after,
        'bound_source_and_input_bytes_unchanged': before == after,
        'outputs': outputs, 'output_log': {'path': frozen['output'], **binding(ROOT / frozen['output'])},
        'all3456_whole_cohort_machine_gates_pass': gates,
        'actual_statistics': stats, 'actual_v4_preservation': preservation, 'inspection_errors': errors,
        'status': 'PASS_WHOLE3456_FROZEN_MACHINE_AUDIT' if gates and proc.returncode == 0 and before == after else 'FAILED_PRESERVED_MACHINE_AUDIT',
        'data_approval': False, 'scope': 'Machine evidence for independent reviews/adjudication; no publication or approval',
    }
    with (ROOT / frozen['receipt']).open('x') as out:
        json.dump(record, out, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps({k: record[k] for k in ['status', 'exit_code', 'wall_seconds', 'all3456_whole_cohort_machine_gates_pass']}))
    if record['status'] != 'PASS_WHOLE3456_FROZEN_MACHINE_AUDIT':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
