"""Root-owned reviewed installation and preimport runtime preservation.

Standard-library-only; does not import the factory, run tests, generate rows,
publish datasets, or mutate Git. Every old destination is archived before any
replacement. Configuration and this source must be frozen before execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INSTALL_DESTINATIONS = {'tests/rsm_core_integration_test.py'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(relative):
    need(type(relative) is str and relative and '\\' not in relative and '\x00' not in relative, 'relative path required')
    path = PurePosixPath(relative)
    need(not path.is_absolute() and all(part not in ('', '.', '..') for part in relative.split('/')), 'safe relative path required')
    result = ROOT
    for part in path.parts:
        result = result / part
        need(not result.is_symlink(), 'symlink rejected: ' + relative)
    return result


def read(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), 'regular file required: ' + str(path))
        raw = stream.read()
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'file changed during read')
    return raw


def binding(relative):
    raw = read(safe(relative))
    return {'sha256': sha(raw), 'bytes': len(raw)}


def verify(relative, expected):
    actual = binding(relative)
    need(actual == expected, 'exact byte binding differs: ' + relative)


def new_bytes(path, raw):
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), 'immutable output already exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def new_json(path, value):
    new_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def archive(raw):
    relative = str((HERE / 'raw_source_archives' / (sha(raw) + '.source')).relative_to(ROOT))
    path = safe(relative)
    if path.exists():
        need(read(path) == raw, 'existing immutable raw archive differs')
    else:
        new_bytes(path, raw)
    return relative


def git(repository, arguments):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    result = subprocess.run(['git', '-C', str(repository), *arguments], env=environment,
                            check=True, capture_output=True)
    return result.stdout.decode().strip()


def guards(config):
    releases = config['immutable_release_files']
    need(len(releases) == 222 and len({row['path'] for row in releases}) == 222, 'release guard cardinality')
    actual_paths = {str(path.relative_to(ROOT)) for folder in ROOT.glob('data/*/releases')
                    for path in folder.rglob('*') if path.is_file()}
    need(actual_paths == {row['path'] for row in releases}, 'release file membership changed')
    for row in releases:
        verify(row['path'], {'sha256': row['sha256'], 'bytes': row['size']})
    references = config['readonly_reference_files']
    need(len(references) == 65 and len({row['path'] for row in references}) == 65, 'reference guard cardinality')
    for row in references:
        raw = read(row['path'])
        need(sha(raw) == row['sha256'] and len(raw) == row['size'], 'protected reference changed')
    git_guards = config['original_git_guards']
    need(len(git_guards) == 6 and len({row['path'] for row in git_guards}) == 6, 'Git guard cardinality')
    for row in git_guards:
        path = Path(row['path'])
        need(path.exists() == row['expected_exists'], 'protected Git existence changed')
        if row['expected_exists']:
            need(sha(read(path)) == row['sha256'], 'protected Git bytes changed')
    repositories = config['repositories']
    need(len(repositories) == 2, 'two repository guards required')
    for row in repositories:
        need(git(row['path'], ['rev-parse', 'HEAD']) == row['head'], 'repository HEAD changed')
        need(git(row['path'], ['status', '--porcelain']) == '', 'repository no longer clean')
    need(git(repositories[1]['path'], ['show', '-s', '--format=%P', 'HEAD']) ==
         '4d0b5eff1af3184fedf7fa25ada729a6cc909268', 'dedicated sole parent changed')
    need(git(repositories[1]['path'], ['rev-parse', 'refs/remotes/origin/research/rineya-high-quality-dataset-v1']) ==
         '653ba70f1f7035a5aed137102c14a1ff29d1cf99', 'last verified remote ref changed')
    for relative, expected in config['held_evidence'].items():
        verify(relative, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    arguments = parser.parse_args()
    config_raw = read(safe(arguments.config))
    need(sha(config_raw) == arguments.expected_config_sha256, 'configuration hash mismatch')
    config = json.loads(config_raw)
    need(config['status'] == 'ROOT_REVIEWED_INSTALLATION_ONLY_NO_DATA_APPROVAL', 'installation authorization scope')
    verify(str(Path(__file__).relative_to(ROOT)), config['installer_source'])
    plan = config['installation_plan']
    need(set(plan) == INSTALL_DESTINATIONS, 'exact installation scope required')
    need(not (HERE / 'EXECUTION.json').exists() and not (HERE / 'ALL_OLD_DESTINATIONS_PRESERVED.json').exists(), 'first installation attempt already exists')
    guards(config)
    before, prepared = {}, {}
    for destination, row in plan.items():
        target = safe(destination)
        need(target.exists() == (row['before'] is not None), 'destination existence differs: ' + destination)
        if row['before'] is not None:
            verify(destination, row['before'])
            raw = read(target)
            before[destination] = {**row['before'], 'archive': archive(raw)}
        else:
            before[destination] = {'exists': False}
        verify(row['prepared_path'], row['prepared_binding'])
        prepared[destination] = read(safe(row['prepared_path']))
        archive(prepared[destination])
    # All originals and prepared bytes are durable before ANY replacement.
    new_json(HERE / 'ALL_OLD_DESTINATIONS_PRESERVED.json', {
        'at_utc': datetime.now(timezone.utc).isoformat(), 'actual_runner_execution': '/root',
        'config_sha256': sha(config_raw), 'before': before,
        'all_old_sources_preserved_before_any_replacement': True,
    })
    replaced = []
    try:
        for destination, row in plan.items():
            if row['before'] is not None:
                verify(destination, row['before'])
            else:
                need(not safe(destination).exists(), 'new destination appeared')
            target = safe(destination)
            temporary = target.with_name(target.name + '.rsm-install-attempt02.tmp')
            new_bytes(temporary, prepared[destination])
            os.replace(temporary, target)
            replaced.append(destination)
        paths = set(config['runtime_paths'])
        need(INSTALL_DESTINATIONS <= paths, 'new runtime source closure incomplete')
        need(set(config['legacy_unmodified_bindings']) <= paths, 'legacy source/input closure omitted')
        runtime_bindings = {}
        for relative in sorted(paths):
            if relative in config['legacy_unmodified_bindings']:
                verify(relative, config['legacy_unmodified_bindings'][relative])
            raw = read(safe(relative))
            runtime_bindings[relative] = {'sha256': sha(raw), 'bytes': len(raw), 'archive': archive(raw)}
        for destination, row in plan.items():
            need(runtime_bindings[destination]['sha256'] == row['prepared_binding']['sha256'], 'installed source differs')
        freeze = {'status': 'INSTALLED_RUNTIME_FROZEN_BEFORE_FIRST_CORE_IMPORT',
                  'at_utc': datetime.now(timezone.utc).isoformat(), 'actual_author_execution': '/root',
                  'bindings': runtime_bindings, 'legacy_modules': config['legacy_modules'],
                  'suite_source_paths': sorted(config['runtime_paths']),
                  'new_modules': ['tests.rsm_core_integration_test', 'tests.rsm_reference_routing_test'],
                  'python_executable': sys.executable, 'python_version': sys.version,
                  'installation_config_sha256': sha(config_raw),
                  'data_approval': False, 'core_import_or_test': 'NOT_RUN'}
        new_json(HERE / 'INSTALLED_RUNTIME_SOURCE_FREEZE.json', freeze)
        for relative, expected in runtime_bindings.items():
            verify(relative, {'sha256': expected['sha256'], 'bytes': expected['bytes']})
            need(read(safe(expected['archive'])) == read(safe(relative)), 'runtime archive equality')
        guards(config)
        new_json(HERE / 'EXECUTION.json', {'status': 'INSTALLED_AND_FROZEN_NO_CORE_IMPORT',
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'actual_runner_execution': '/root',
            'installed_destinations': sorted(replaced), 'runtime_bindings': len(runtime_bindings),
            'runtime_freeze_sha256': sha(read(HERE / 'INSTALLED_RUNTIME_SOURCE_FREEZE.json')),
            'all_originals_preserved': True, 'all_guards_unchanged': True, 'data_approval': False})
    except BaseException as error:
        # Make the first failure durable before attempting any restoration.
        new_json(HERE / 'FIRST_FAILURE.json', {'status': 'INSTALLATION_FAILED_PRESERVED',
            'at_utc': datetime.now(timezone.utc).isoformat(), 'error': type(error).__name__ + ': ' + str(error),
            'attempted_replacements': replaced, 'rollback': 'PENDING',
            'core_import_or_test': 'NOT_RUN', 'data_approval': False})
        # Restore each own replacement independently; preserve external conflicts.
        conflicts = []
        for destination in reversed(replaced):
            try:
                target = safe(destination)
                if not target.exists() or read(target) != prepared[destination]:
                    conflicts.append({'path': destination, 'reason': 'external source changed'})
                    continue
                old = before[destination]
                if old.get('exists') is False:
                    target.unlink()
                else:
                    temporary = target.with_name(target.name + '.rsm-rollback-attempt02.tmp')
                    new_bytes(temporary, read(safe(old['archive'])))
                    os.replace(temporary, target)
            except BaseException as restoration_error:
                conflicts.append({'path': destination, 'reason': type(restoration_error).__name__ + ': ' + str(restoration_error)})
        new_json(HERE / 'ROLLBACK.json', {'status': 'OWN_RESTORATION_ATTEMPT_COMPLETED',
            'at_utc': datetime.now(timezone.utc).isoformat(), 'external_conflicts_preserved': conflicts,
            'core_import_or_test': 'NOT_RUN', 'data_approval': False})
        raise


if __name__ == '__main__':
    main()
