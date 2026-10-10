"""ROOT execution of independently authored actual-cohort machine checks.

Unexecuted preparation. Full integration and exact primary/environment raw
closures are mandatory before importing any factory or semantic oracle helper.
No reviewer vote, release, candidate mutation, model work, or Git mutation.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BATCH = 'option-resumption-pilot-v1'
OUTPUTS = ['reviews/oracle/' + BATCH + '.jsonl',
           'audits/diversity/' + BATCH + '-rsm-controls.json',
           'audits/leakage/' + BATCH + '.json']


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(relative):
    need(type(relative) is str and relative and chr(0) not in relative and '\\' not in relative,
         'literal relative path required')
    parts = PurePosixPath(relative)
    need(not parts.is_absolute() and all(v not in ('', '.', '..')
         for v in relative.split('/')), 'normalized relative path required')
    current = ROOT
    for part in parts.parts:
        current = current / part
        need(not current.is_symlink(), 'symlink rejected: ' + relative)
    return current


def read(path):
    path = Path(path)
    need(path.is_absolute(), 'absolute read path required')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        need(not current.is_symlink(), 'resolved regular input path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), 'regular input required')
        raw = stream.read()
        after = os.fstat(stream.fileno())
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    need(tuple(getattr(before, k) for k in keys) ==
         tuple(getattr(after, k) for k in keys), 'input changed during read')
    return raw


def binding(path):
    raw = read(safe(path))
    return {'path': path, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def verify(bindings, external=False):
    need(type(bindings) is dict and bindings, 'nonempty exact closure required')
    for path, row in bindings.items():
        need(type(row) is dict and type(row['bytes']) is int and row['bytes'] >= 0,
             'typed raw binding required')
        location = Path(path) if external else safe(path)
        raw = read(location)
        need(len(raw) == row['bytes'] and sha256(raw).hexdigest() == row['sha256'],
             'frozen input mismatch: ' + path)
        need(read(safe(row['archive'])) == raw, 'exact raw archive mismatch: ' + path)


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    args = parser.parse_args()
    raw = read(safe(args.config))
    need(sha256(raw).hexdigest() == args.expected_config_sha256, 'configuration hash')
    config = json.loads(raw)
    need(config['schema'] == 'rsm-root-actual-machine-audit-v1'
         and config['actual_runner_execution_id'] == '/root'
         and config['batch'] == BATCH and config['data_approval'] is False,
         'exact actual-cohort machine scope required')
    output = safe(config['execution_output'])
    need(output.is_relative_to(ROOT / 'quality_reports'), 'own report prefix required')
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    receipt = {'status': 'FAIL_BEFORE_SEMANTIC_IMPORT',
               'actual_runner_execution_id': '/root',
               'independent_oracle_source_author_execution_id': '/root/formal_oracle',
               'root_wrapper_author_execution_id': '/root',
               'started_at_utc': started, 'pid': os.getpid(),
               'config_sha256': sha256(raw).hexdigest(),
               'data_approval': False, 'release_run': False}
    try:
        verify(config['primary_bindings'])
        verify(config['environment_bindings'], external=True)
        need(str(Path(sys.executable).resolve()) == config['resolved_python'],
             'resolved Python mismatch')
        need(str(Path(__file__).relative_to(ROOT)) in config['primary_bindings'],
             'actual wrapper source absent from frozen closure')
        core = json.loads(read(safe(config['full_suite_execution'])))
        completion = json.loads(read(safe(config['full_suite_root_completion'])))
        need(core['status'] == 'PASS' and core['mode'] == 'full'
             and core['actual_runner_execution_id'] == '/root'
             and type(core['tests_run']) is int and core['tests_run'] > 0
             and all(type(core[k]) is int and core[k] == 0
                     for k in ('failures', 'errors', 'skipped'))
             and core['changed_runtime_bindings'] == [], 'actual successful full suite required')
        need(len(core['actual_test_modules']) == 40
             and len(set(core['actual_test_modules'])) == 40
             and core['actual_test_modules'] == core['actual_full_suite_module_inventory'],
             'complete measured 40-module execution required')
        need(completion['actual_process_exit_code'] == 0
             and completion['full40_suite_ran'] is True
             and completion['tests_run'] == core['tests_run']
             and completion['runtime_freeze']['sha256'] == core['runtime_freeze_sha256'],
             'exact actual root completion required')
        for relative in OUTPUTS:
            need(not safe(relative).exists(), 'first actual audit destination exists: ' + relative)
        # Benign schema/runtime imports are covered by the external preparation
        # freeze; no factory or semantic helper has been imported above.
        import jsonschema
        sys.path.insert(0, str(ROOT))
        from factory import pipeline
        stats = pipeline.audit(BATCH)
        receipt['actual_audit_stats'] = stats
        receipt['outputs'] = [binding(relative) for relative in OUTPUTS]
        checks = [json.loads(line) for line in read(safe(OUTPUTS[0])).splitlines()]
        need(len(checks) == 216 and len({v['sample_id'] for v in checks}) == 216,
             'whole actual 216 oracle receipt cohort required')
        need(all(v['oracle_author_execution'] == '/root/formal_oracle'
                 and v['check_runner_execution'] == '/root' for v in checks),
             'actual source author and runner roles required')
        verify(config['primary_bindings'])
        verify(config['environment_bindings'], external=True)
        runtime_files = []
        for name, module in sorted(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if origin and '/lib/python' in origin and Path(origin).exists():
                resolved = str(Path(origin).resolve())
                need(resolved in config['environment_bindings'],
                     'new unfrozen runtime module: ' + name)
                runtime_files.append({'module': name, 'path': resolved})
        receipt['actual_loaded_environment_files'] = runtime_files
        receipt['runtime_boundary'] = config['runtime_boundary']
        receipt['verified_primary_raw_pairs'] = len(config['primary_bindings'])
        receipt['verified_environment_raw_pairs'] = len(config['environment_bindings'])
        receipt['candidate_unchanged'] = True
        need(stats['records'] == 216 and stats['all_machine_quality_gates_pass'] is True
             and all(v['pass'] is True for v in checks), 'actual machine checks failed')
        receipt['status'] = 'PASS_ACTUAL_216_MACHINE_CHECKS_ONLY'
    except BaseException:
        receipt['traceback'] = traceback.format_exc()
        receipt['status'] = 'FAIL_PRESERVED'
    receipt['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    receipt['elapsed_seconds'] = round(time.monotonic() - clock, 6)
    save(output / 'EXECUTION.json', receipt)
    print(json.dumps({k: receipt.get(k) for k in
                      ('status', 'actual_runner_execution_id', 'elapsed_seconds',
                       'verified_primary_raw_pairs', 'verified_environment_raw_pairs')}, sort_keys=True))
    return 0 if receipt['status'] == 'PASS_ACTUAL_216_MACHINE_CHECKS_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
