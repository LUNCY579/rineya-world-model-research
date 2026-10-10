"""ROOT bounded documentation transition; no dataset approval or Git mutation.

All ten original streams and all proposed streams are frozen before execution.
The barrier verifies every old stream/archive before the first atomic rewrite.
All prior semantic runtime sources remain unchanged; old execution receipts
keep their own dated documentation snapshot, preserved here verbatim.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LIVING = ('PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
          'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md', 'TRAINING_READINESS.md',
          'GITHUB_BACKUP_LOG.md', 'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
          'research/role_registry.json')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def path(name):
    need(type(name) is str and name and chr(0) not in name and '\\' not in name,
         'literal path required')
    parts = PurePosixPath(name)
    need(not parts.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
         'normalized relative path required')
    current = ROOT
    for part in parts.parts:
        current /= part
        need(not current.is_symlink(), 'no symlink components')
    return current


def read(name):
    p = path(name)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), 'regular file required')
        raw = stream.read()
        after = os.fstat(stream.fileno())
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'changed during read')
    return raw


def binding(name):
    raw = read(name)
    return {'path': name, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def verified(row):
    need(binding(row['path']) == {k: row[k] for k in ('path', 'sha256', 'bytes')},
         'exact source binding mismatch: ' + row['path'])
    raw = read(row['path'])
    need(read(row['archive']) == raw, 'exact archive mismatch: ' + row['path'])
    return raw


def write_json(p, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with p.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    args = parser.parse_args()
    need(sha256(Path(__file__).read_bytes()).hexdigest() == args.expected_source_sha256,
         'external source hash mismatch')
    raw = read(args.config)
    need(sha256(raw).hexdigest() == args.expected_config_sha256, 'external config hash mismatch')
    config = json.loads(raw)
    need(config['actual_author_execution_id'] == '/root' and config['data_approval'] is False,
         'bounded documentation-only scope')
    need(set(config['old']) == set(config['proposed']) == set(LIVING), 'exact ten document set')
    old = {name: verified(row) for name, row in config['old'].items()}
    proposed = {name: verified(row) for name, row in config['proposed'].items()}
    for row in config['evidence_bindings']:
        verified(row)
    frozen = json.loads(read(config['runtime_freeze']['path']))
    need(binding(config['runtime_freeze']['path']) == config['runtime_freeze'], 'runtime freeze hash')
    need(len(frozen['bindings']) == 507, 'historical runtime size')
    for name, row in frozen['bindings'].items():
        need(binding(name) == {'path': name, 'sha256': row['sha256'], 'bytes': row['bytes']} and
             read(row['archive']) == read(name), 'runtime changed before documentation refresh')
        if name in LIVING:
            need(old[name] == read(row['archive']), 'ten document prior snapshot mismatch')
    for name in LIVING[:-1]:
        need(proposed[name].endswith(old[name]), 'verbatim historical report suffix required')
    registry = json.loads(proposed[LIVING[-1]])
    wave = registry['rsm_option_resumption_prospective_wave']
    need(wave['new_approved_records'] == 0 and wave['production_data_approval'] is False and
         wave['actual_generated_counts']['PENDING_rows'] == 216 and
         registry['current_backup_custody']['commit_sha'] == '653ba70f1f7035a5aed137102c14a1ff29d1cf99',
         'no new approval or remote claim')
    barrier = {'status': 'ALL10_OLD_AND_PROPOSED_STREAMS_RAW_VERIFIED_BEFORE_FIRST_WRITE',
               'actual_author_execution_id': '/root', 'at_utc': datetime.now(timezone.utc).isoformat(),
               'config_sha256': sha256(raw).hexdigest(), 'old': config['old'],
               'proposed': config['proposed'], 'data_approval': False}
    write_json(HERE / 'ALL10_PREWRITE_PRESERVATION.json', barrier)
    for name in LIVING:
        need(read(name) == old[name], 'concurrent living document mutation')
        target = path(name)
        temporary = target.parent / ('.' + target.name + '.rsm_living_refresh_attempt01.tmp')
        with temporary.open('xb') as stream:
            stream.write(proposed[name])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    for name in LIVING:
        need(read(name) == proposed[name] and read(config['old'][name]['archive']) == old[name],
             'post-write current/archive verification')
    for name, row in frozen['bindings'].items():
        if name not in LIVING:
            need(binding(name) == {'path': name, 'sha256': row['sha256'], 'bytes': row['bytes']},
                 'unrelated runtime changed')
    for row in config['evidence_bindings']:
        verified(row)
    completion = {'status': 'PASS_ALL10_CURRENT_TRANSITIONS_VERIFIED_497_OTHER_RUNTIME_BINDINGS_UNCHANGED',
                  'actual_author_execution_id': '/root', 'actual_runner_execution_id': '/root',
                  'at_utc': datetime.now(timezone.utc).isoformat(), 'config': binding(args.config),
                  'source_sha256': args.expected_source_sha256, 'old': config['old'],
                  'new': {name: binding(name) for name in LIVING},
                  'all10_prior_streams_archived_before_any_write': True,
                  'nine_markdown_prior_bytes_preserved_as_verbatim_suffix': True,
                  'historical_runtime_freeze': config['runtime_freeze'],
                  'only_historical_runtime_paths_evolved': list(LIVING),
                  'other_runtime_current_bindings_unchanged': 497, 'data_approval': False,
                  'remote_backup_claim': False, 'new_living_bytes_local_post653': True}
    write_json(HERE / 'EXECUTION.json', completion)
    print(json.dumps({'status': completion['status'], 'documents': 10, 'data_approval': False}))


if __name__ == '__main__':
    main()
