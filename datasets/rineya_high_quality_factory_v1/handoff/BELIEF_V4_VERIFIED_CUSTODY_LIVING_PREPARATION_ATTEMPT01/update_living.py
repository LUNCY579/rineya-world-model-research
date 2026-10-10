#!/usr/bin/env python3
"""Gated ten-document update. Production execution requires new root-bound custody.

This source is a preparation only until root supplies the completed joined receipt,
its completed readback, a READY config and a freeze made before first import/run.
It does not import dataset code, fetch remote payloads or grant data approval.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import traceback

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[1]
AUTHOR = '/root/living_custody_updater'
RUNNER = '/root'
COMMIT = '653ba70f1f7035a5aed137102c14a1ff29d1cf99'
PARENT = '4d0b5eff1af3184fedf7fa25ada729a6cc909268'
PREFIX = 'handoff/BELIEF_V4_VERIFIED_CUSTODY_LIVING_PREPARATION_ATTEMPT01/'
SNAPSHOT = PREFIX + 'PREPARATION_INPUT_SNAPSHOT.json'
DRAFT = PREFIX + 'EXECUTION_CONFIG_DRAFT.json'
PROPOSAL = PREFIX + 'PROPOSED_HEADER.md'
SOURCE = PREFIX + 'update_living.py'
HISTORY = 'handoff/report_history/belief-v4-verified-custody-living-attempt01'
LIVING = ('PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
          'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md', 'TRAINING_READINESS.md',
          'GITHUB_BACKUP_LOG.md', 'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
          'research/role_registry.json')
COUNTS = {'RBM-001': 3480, 'RCM-001': 1440, 'RDP-001': 24, 'REM-001': 888,
          'RGM-001': 840, 'RJP-001': 24, 'RMM-001': 76, 'ROM-001': 24,
          'RSM-001': 24, 'RVM-001': 384}
REQUIRED_SCOPE = {'commit_sha': COMMIT, 'sole_parent_sha': PARENT, 'files': 3158,
                  'bytes': 1263090770, 'unique_git_blobs': 2474, 'raw_paths': 31,
                  'raw_bytes': 1724728576, 'local_decode_iterations': 32,
                  'local_decoded_bytes': 1766951239, 'manifest_references': 32,
                  'global_manifest_paths': 31, 'remote_raw_gzip_pairs': 31,
                  'all_release_files_capped_raw_union': 222, 'research_references': 65,
                  'original_git_guards': 6, 'changed_paths': 484, 'strict_other_paths': 482,
                  'unrestricted_diff_exit': 2, 'exact_whitespace_warnings': 6,
                  'exact_whitespace_files': 2, 'atcut_observed_deferred_paths': 26,
                  'postcut_completed_adjudication_files': 29, 'grants_data_approval': False}
SHA = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def typed(a, b):
    if type(a) is not type(b):
        return False
    if type(b) is dict:
        return set(a) == set(b) and all(typed(a[k], b[k]) for k in b)
    if type(b) in (list, tuple):
        return len(a) == len(b) and all(typed(x, y) for x, y in zip(a, b))
    return a == b


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def literal(name, must_exist=True):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name,
            'Literal factory-relative path required')
    parts = name.split('/')
    require(all(p not in ('', '.', '..') for p in parts), 'Path aliases forbidden')
    p = ROOT
    for part in parts:
        p = p / part
        require(not p.is_symlink(), 'Symlink component forbidden: ' + name)
    if must_exist:
        require(p.is_file(), 'Existing regular file required: ' + name)
    return p


def read_regular(path):
    """Hold an inode while reading; reject aliases and concurrent replacement."""
    require(path.is_absolute() and not path.is_symlink(), 'Regular absolute path required')
    for parent in path.parents:
        require(not parent.is_symlink(), 'Symlink ancestor forbidden')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'Nonregular file forbidden')
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(identity(before) == identity(after) == identity(current),
                'File changed during read: ' + str(path))
        return b''.join(chunks)
    finally:
        os.close(fd)


def proof(name):
    p = literal(name)
    # Large release bytes are streamed rather than accumulated in memory.
    fd = os.open(p, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'Nonregular binding forbidden')
        digest, size = hashlib.sha256(), 0
        while block := os.read(fd, 65536):
            digest.update(block)
            size += len(block)
        after, current = os.fstat(fd), p.stat(follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(identity(before) == identity(after) == identity(current) and size == current.st_size,
                'Binding changed during read: ' + name)
        return {'path': name, 'size': size, 'sha256': digest.hexdigest()}
    finally:
        os.close(fd)


def validate_row(row):
    require(type(row) is dict and set(row) == {'path', 'size', 'sha256'} and
            type(row['path']) is str and type(row['size']) is int and row['size'] >= 0 and
            type(row['sha256']) is str and SHA.fullmatch(row['sha256']), 'Typed byte binding required')
    return row


def bound_json(row):
    validate_row(row)
    require(typed(proof(row['path']), row), 'JSON binding differs: ' + row['path'])
    return json.loads(read_regular(literal(row['path'])))


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive(path, data):
    require(not path.exists() and not path.is_symlink(), 'Fresh output required')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(path.parent)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def atomic_replace(path, content, serial):
    temporary = path.with_name(path.name + '.verified-custody-living-' + str(serial) + '.tmp')
    require(not temporary.exists() and not temporary.is_symlink(), 'Fresh transaction temp required')
    try:
        exclusive(temporary, content)
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_dir(path.parent)


def git(worktree, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run(['git', *args], cwd=worktree, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
    require(p.returncode == 0 and not p.stderr and len(p.stdout) <= 128 * 1024,
            'Read-only Git check failed')
    return p.stdout


def check_immutable(snapshot):
    releases = snapshot['all222_release_files']
    require(type(releases) is list and len(releases) == 222, 'All222 release bindings required')
    expected_paths = {validate_row(v)['path'] for v in releases}
    actual_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'data').glob('*/releases/**/*')
                    if p.is_file() and not p.is_symlink()}
    require(len(expected_paths) == 222 and actual_paths == expected_paths, 'Release membership changed')
    for row in releases:
        require(typed(proof(row['path']), row), 'Immutable release changed: ' + row['path'])
    references = snapshot['all65_readonly_reference_files']
    require(type(references) is list and len(references) == 65 and
            len({r['path'] for r in references}) == 65, 'All65 reference bindings required')
    for row in references:
        validate_row(row)
        path = Path(row['path'])
        require(path.is_absolute() and any(path.is_relative_to(ROOT.parent / n) for n in
                ('rineya_complementary_research_2026-10-09', 'rineya-world-model-research')),
                'Reference outside protected roots')
        data = read_regular(path)
        require(len(data) == row['size'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                'Readonly reference changed: ' + str(path))
    guards = snapshot['six_original_git_guards']
    require(type(guards) is list and len(guards) == 6 and len({g['label'] for g in guards}) == 6,
            'Exact six Git guards required')
    for row in guards:
        path = Path(row['path'])
        require(path.is_absolute() and path.is_relative_to(ROOT.parent / 'rineya-world-model-research/.git'),
                'Git guard outside original repository')
        require(type(row['expected_exists']) is bool and path.exists() is row['expected_exists']
                and not path.is_symlink(), 'Original Git existence guard changed')
        if row['expected_exists']:
            require(hashlib.sha256(read_regular(path)).hexdigest() == row['sha256'],
                    'Original Git byte guard changed')
    original = ROOT.parent / 'rineya-world-model-research'
    worktree = ROOT.parent / '.rineya-dataset-git-worktree'
    require(git(original, 'rev-parse', 'HEAD').strip().decode() == snapshot['original_head'],
            'Original branch HEAD changed')
    require(not git(original, 'status', '--porcelain') and not git(worktree, 'status', '--porcelain'),
            'Both protected checkouts must remain clean')
    require(git(worktree, 'rev-parse', 'HEAD').strip().decode() == COMMIT and
            git(worktree, 'rev-parse', 'HEAD^').strip().decode() == PARENT and
            git(worktree, 'rev-parse', 'origin/research/rineya-high-quality-dataset-v1').strip().decode() == COMMIT,
            'Dedicated checkpoint/sole parent/remote ref changed')


def preflight(config_row, freeze_row):
    config, freeze = bound_json(config_row), bound_json(freeze_row)
    draft = json.loads(read_regular(literal(DRAFT)))
    allowed_changes = {'status', 'root_execution_authorized', 'completed_joined_custody', 'completed_root_join_readback'}
    require(set(config) == set(draft) and all(typed(config[k], draft[k]) for k in draft if k not in allowed_changes),
            'Completed config differs from prepared scope')
    require(config['status'] == 'READY_ROOT_EXECUTION_AFTER_COMPLETE_JOIN_READBACK' and
            config['root_execution_authorized'] is True and config['actual_runner_execution'] == RUNNER,
            'Root execution is not authorized')
    join_row = validate_row(config['completed_joined_custody'])
    readback_row = validate_row(config['completed_root_join_readback'])
    joined, readback = bound_json(join_row), bound_json(readback_row)
    require(joined['status'] == 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED' and
            joined['commit_sha'] == COMMIT and joined['sole_parent_sha'] == PARENT and
            joined['grants_data_approval'] is False, 'Completed exact653 join required')
    for key, expected in REQUIRED_SCOPE.items():
        require(key in joined['exact_scope'] and typed(joined['exact_scope'][key], expected),
                'Joined scope differs: ' + key)
    require(readback.get('status') == 'PASS_COMPLETED_APPROVED_RELEASE_CUSTODY_JOIN_READBACK' and
            readback.get('actual_reader_execution') == RUNNER and readback.get('commit_sha') == COMMIT and
            typed(readback.get('joined_custody'), join_row) and readback.get('data_approval') is False,
            'Completed root readback with exact joined receipt required')
    for key in ('all_bound_inputs_current', 'joined_scope_exact', 'all222_release_files_unchanged',
                'all65_readonly_references_unchanged', 'all6_git_guards_unchanged', 'postcut_boundary_preserved'):
        require(readback.get(key) is True, 'Root readback gate missing: ' + key)
    source = proof(SOURCE)
    require(typed(freeze.get('source'), source) and typed(freeze.get('execution_config'), config_row) and
            freeze.get('actual_runner_execution') == RUNNER and
            freeze.get('frozen_before_first_import_execution') is True,
            'Own production source/config pre-import freeze required')
    snapshot = json.loads(read_regular(literal(SNAPSHOT)))
    required_rows = [proof(n) for n in (SOURCE, SNAPSHOT, DRAFT, PROPOSAL)] + snapshot['evidence_bindings'] + [join_row, readback_row, config_row]
    postcut = joined['completed_postcut_adjudication_artifacts']
    require(type(postcut) is list and len(postcut) == 29 and
            len(joined['historical_observed_pending_atcut']) == 26, 'Actual26/29 postcut boundary required')
    required_rows += [validate_row(v) for v in postcut]
    rows = freeze.get('frozen_inputs')
    require(type(rows) is list and rows, 'Complete typed freeze list required')
    frozen_map = {}
    for row in rows:
        validate_row(row)
        require(row['path'] not in frozen_map and typed(proof(row['path']), row), 'Duplicate/changed frozen input')
        frozen_map[row['path']] = row
    for row in required_rows:
        require(typed(frozen_map.get(row['path']), row), 'Mandatory frozen dependency omitted or changed: ' + row['path'])
    for key in ('living_before', 'all222_release_files', 'all65_readonly_reference_files', 'six_original_git_guards'):
        require(typed(freeze.get(key), snapshot[key]), 'Protected freeze collection omitted or substituted: ' + key)
    require(set(snapshot['living_before']) == set(LIVING), 'Exact ten living bindings required')
    for name in LIVING:
        require(typed(proof(name), snapshot['living_before'][name]), 'Living bytes changed since preparation')
    evidence = {v['path']: v for v in snapshot['evidence_bindings']}
    stage_config = bound_json(evidence['backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_EXECUTION_CONFIG.json'])
    adapter = bound_json(evidence['backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json'])
    require(typed(stage_config['readonly_reference_files'], snapshot['all65_readonly_reference_files']) and
            typed(adapter['protected_git_byte_baselines'], snapshot['six_original_git_guards']),
            'Original exact65/six source baselines substituted')
    spec = bound_json(evidence['backup_logs/belief-v4-approved-release-r1_SPEC.json'])
    require(len(spec['files']) == 3158 and sum(v['size'] for v in spec['files']) == 1263090770 and
            spec['expected_parent_sha'] == PARENT and spec['grants_data_approval'] is False,
            'Frozen3158 inventory scope differs')
    inventory = {v['path']: {k: v[k] for k in ('path', 'size', 'sha256')}
                 for v in spec['files'] + spec['raw_size_exclusions']}
    for row in snapshot['all222_release_files']:
        require(typed(inventory.get(row['path']), row), 'Release binding outside frozen capped/raw union')
    catalog = bound_json(evidence['manifests/ACTIVE_RELEASE_CATALOG.json'])
    tracks = collections.Counter()
    for item in catalog['releases']:
        tracks[item['track']] += item['active_records']
    require(catalog['active_records'] == 7204 and catalog['active_source_families'] == 700 and
            dict(tracks) == COUNTS and len(catalog['releases']) == 19 and
            sum(v['active_records'] > 0 for v in catalog['releases']) == 17, 'Actual catalog changed')
    completed = joined['actual_completed_proofs']
    for key, path in config['completed_proof_paths'].items():
        require(typed(completed.get(key), evidence[path]), 'Actual full/gzip proof/ledger join differs')
    check_immutable(snapshot)
    return config, freeze, snapshot, source, joined


def registry_content(old, source, config, started):
    registry = json.loads(old)
    registry['backup_execution_note'] = ('Root ran stage/push/full remote proof and this living update. '
        'Backup manager authored and ran the fresh31gzip proof and completed custody join. '
        'The new living updater author is /root/living_custody_updater; actual roles are receipt-bound.')
    registry['belief_evidence_wave'].update({
        'stage': 'V4_ALL3456_INDEPENDENT_ACCEPT_VALIDATED_V0002_REMOTELY_VERIFIED_FROZEN_653',
        'v4_data_approval': True, 'v4_approved_records': 3456, 'v4_production_records': 3456,
        'v4_production_record_count_claim': 'Actual completed3456 independently accepted records; no new semantic tasks or source families relative to V3 quarantine.',
        'v4_release_remotely_backed_up': True, 'v4_release_verified_checkpoint': COMMIT,
        'v4_release_joined_custody': config['completed_joined_custody'],
        'v4_added_semantic_tasks_relative_v3_quarantine': 0, 'v4_added_source_families_relative_v3_quarantine': 0,
        'v4_added_student_inputs_relative_v3_quarantine': 0,
        'v4_adversarial_and_architect_preparation_local_only': False,
        'v4_official_checker_final_status': 'ACTUAL_COMPLETED_VALIDATED_FROZEN_653_NO_DATA_APPROVAL',
        'v4_status_evidence_boundary': 'Independent data approval predates publication; exact frozen653 backup custody grants no approval. New proof/living bytes remain postcutLOCAL.'})
    registry['current_backup_custody'] = {
        'commit_sha': COMMIT, 'sole_parent_sha': PARENT, 'frozen_files': 3158, 'frozen_bytes': 1263090770,
        'actual_full_unique_remote_fetches': 2474, 'actual_new_gzip_fetches': 31,
        'raw_bytes_exactly_recovered': 1724728576, 'local_decode_iterations': 32,
        'local_decoded_bytes_including_multiple_representations': 1766951239,
        'perraw_manifest_references': 32, 'global_manifest_paths': 31,
        'joined_custody': config['completed_joined_custody'], 'root_join_readback': config['completed_root_join_readback'],
        'v4_release_frozen_bytes_remotely_verified': True,
        'new_proofs_helpers_rsm_preparations_and_living_bytes_local_only': True,
        'historical_observed_deferred_atcut_paths': 26, 'actual_completed_postcut_adjudication_files': 29,
        'unrestricted_diff_exit': 2, 'exact_whitespace_warnings': 6, 'exact_whitespace_paths': 2,
        'restricted_strict_changed_paths_passed': 482, 'backup_grants_data_approval': False}
    registry['rsm_option_resumption_prospective_wave'] = {
        'stage': 'R2_PUBLIC_AND_ISOLATED_HAND_PREPARATIONS_COMPLETED_SHARED_FACTORY_INTEGRATION_PENDING',
        'production_authorized': False, 'new_approved_records': 0, 'actual_generated_counts': None,
        'proposed_source_families': 12, 'proposed_typed_tasks': 108, 'proposed_views': 216,
        'legacy_lineage_families': 12, 'legacy_structural_signatures': 8,
        'operative_metadata_amendment': 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md',
        'completed_preparation_checks': {'producer_each_seed': 337, 'producer_seeds': [17, 991],
            'expert': 290, 'adjudicator_tests': 24, 'adjudicator_subtests': 365,
            'adversarial_initial': 1568, 'adversarial_interface_correction': 63, 'isolated_bridge_hand_tests': 20},
        'shared_factory_integration': 'NOT_RUN', 'actual_production_graph_novelty': 'NOT_RUN',
        'independent_production_input_first_review': 'NOT_RUN', 'production_data_approval': False,
        'later_preparation_artifacts_postcut_local': True,
        'exact_preparation_backup_boundary': 'Some earlier public/checker preparations are included in frozen653; later producer/adversarial/adjudicator/bridge/root-preparation bytes remain LOCAL. Only explicit SPEC paths are claimed.',
        'limitations': 'Bounded public/HAND preparation; common model/public guidance/design-adjudication overlap; no new producer graphs, learned hierarchy, sealed transfer or efficacy evidence.'}
    registry['living_report_refresh_execution'] = {
        'actual_author': AUTHOR, 'actual_runner': RUNNER, 'at_utc': started, 'source': source,
        'all10_prior_byte_streams_archived_before_any_write': True,
        'new_living_bytes_local_postinventory': True, 'data_approval_by_updater': False}
    return json_bytes(registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-config', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    for digest in (args.expected_config_sha256, args.expected_freeze_sha256):
        require(type(digest) is str and SHA.fullmatch(digest), 'Explicit full SHA256 required')
    config_row, freeze_row = proof(args.execution_config), proof(args.source_freeze)
    require(config_row['sha256'] == args.expected_config_sha256 and
            freeze_row['sha256'] == args.expected_freeze_sha256, 'Explicit config/freeze bytes differ')
    config, freeze, snapshot, source, joined = preflight(config_row, freeze_row)
    started = now()
    history = literal(HISTORY, False)
    require(not history.exists(), 'Fresh history required')
    for name in ('ALL10_PREWRITE_PRESERVATION.json', 'EXECUTION.json', 'FAILED_EXECUTION.json'):
        require(not (HERE / name).exists(), 'Fresh execution receipt paths required')
    old = {name: read_regular(literal(name)) for name in LIVING}
    header = read_regular(literal(PROPOSAL)).decode()
    require(header.count('{{AT_UTC}}') == 1 and header.count('{{JOIN_PATH}}') == 1 and
            header.count('{{JOIN_SHA256}}') == 1, 'Exact prepared header substitution contract')
    header = header.replace('{{AT_UTC}}', started).replace('{{JOIN_PATH}}', config['completed_joined_custody']['path']).replace(
        '{{JOIN_SHA256}}', config['completed_joined_custody']['sha256']).encode()
    contents = {name: registry_content(old[name], source, config, started) if name.endswith('.json') else header + old[name]
                for name in LIVING}
    history.mkdir()
    fsync_dir(history.parent)
    archives = {}
    # This global barrier is completed for ALL ten documents before any replacement.
    for name in LIVING:
        archive = history / (snapshot['living_before'][name]['sha256'] + '-' + name.replace('/', '__'))
        exclusive(archive, old[name])
        archives[name] = proof(archive.relative_to(ROOT).as_posix())
        require(archives[name]['size'] == snapshot['living_before'][name]['size'] and
                archives[name]['sha256'] == snapshot['living_before'][name]['sha256'], 'Prior living archive differs')
    exclusive(HERE / 'ALL10_PREWRITE_PRESERVATION.json', json_bytes({
        'status': 'ALL10_EXACT_PRIOR_LIVING_BYTES_ARCHIVED_BEFORE_ANY_WRITE', 'at_utc': now(),
        'actual_author': AUTHOR, 'actual_runner': RUNNER, 'source': source,
        'living_before': snapshot['living_before'], 'archives': archives, 'data_approval': False}))
    replaced = []
    try:
        # Catch drift after archival, while all living documents still have their original bytes.
        preflight(config_row, freeze_row)
        for serial, name in enumerate(LIVING):
            require(typed(proof(name), snapshot['living_before'][name]), 'Living file changed before its replacement')
            replaced.append(name)
            atomic_replace(literal(name), contents[name], serial)
            actual = read_regular(literal(name))
            require(actual == contents[name], 'Committed living content differs')
            if name.endswith('.md'):
                require(actual[len(header):] == old[name], 'Exact historical Markdown suffix changed')
        for row in freeze['frozen_inputs']:
            require(typed(proof(row['path']), row), 'Held evidence/source changed after writing')
        require(typed(proof(freeze_row['path']), freeze_row), 'Production source freeze changed after writing')
        check_immutable(snapshot)
        for row in archives.values():
            require(typed(proof(row['path']), row), 'Archived original changed')
        result = {'status': 'COMPLETED_ALL10_PRESERVED_VERIFIED_FROZEN_CUSTODY_LIVING_UPDATE',
            'actual_author': AUTHOR, 'actual_runner': RUNNER, 'started_at_utc': started, 'finished_at_utc': now(),
            'source': source, 'source_freeze': freeze_row, 'execution_config': config_row,
            'completed_joined_custody': config['completed_joined_custody'],
            'completed_root_join_readback': config['completed_root_join_readback'],
            'all10_prior_byte_streams_archived_before_any_write': True,
            'living_changes': [{'path': n, 'before': snapshot['living_before'][n], 'archive': archives[n], 'after': proof(n)} for n in LIVING],
            'all9_markdown_exact_old_suffixes_preserved': True, 'all222_release_files_unchanged': True,
            'all65_readonly_references_unchanged': True, 'all6_original_git_guards_unchanged': True,
            'both_git_checkouts_clean': True, 'catalog_records': 7204, 'source_families': 700,
            'historical_releases': 19, 'active_releases': 17, 'verified_frozen_checkpoint': COMMIT,
            'new_living_and_execution_bytes_postcut_local': True, 'data_approval_by_updater': False,
            'new_rsm_approved_records': 0, 'rsm_production_authorized': False}
        exclusive(HERE / 'EXECUTION.json', json_bytes(result))
        print(json.dumps({'status': result['status'], 'living_documents': 10, 'data_approval': False}))
    except BaseException as error:
        primary_traceback = traceback.format_exc()
        rollback, conflicts, rollback_errors = [], [], []
        for serial, name in enumerate(reversed(replaced)):
            try:
                current = read_regular(literal(name))
                if current == contents[name]:
                    atomic_replace(literal(name), old[name], 'rollback-' + str(serial))
                    require(read_regular(literal(name)) == old[name], 'Rollback content differs')
                    rollback.append(name)
                elif current != old[name]:
                    # Preserve an independently changed document instead of overwriting it.
                    conflicts.append(name)
            except BaseException as rollback_error:
                rollback_errors.append({'path': name, 'error_type': type(rollback_error).__name__,
                                        'error': str(rollback_error)})
        exclusive(HERE / 'FAILED_EXECUTION.json', json_bytes({
            'status': 'FAILED_UPDATE_WITH_ORIGINAL_ARCHIVES_AND_ROLLBACK_EVIDENCE_PRESERVED',
            'at_utc': now(), 'actual_author': AUTHOR, 'actual_runner': RUNNER, 'source': source,
            'error_type': type(error).__name__, 'error': str(error), 'traceback': primary_traceback,
            'replacement_attempted_before_failure': replaced, 'rolled_back': rollback,
            'external_write_conflicts': conflicts, 'rollback_errors': rollback_errors,
            'archives': archives, 'data_approval': False}))
        raise


if __name__ == '__main__':
    main()
