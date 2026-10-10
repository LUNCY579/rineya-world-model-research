#!/usr/bin/env python3
"""Read-only exact evidence join for the completed bc519 archival checkpoint."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/Users/chengli/Downloads/Rineya_World_Model')
FACTORY = ROOT / 'High-Quality_Dataset'
ORIGINAL = ROOT / 'rineya-world-model-research'
WORKTREE = ROOT / '.rineya-dataset-git-worktree'
PREFIX = 'datasets/rineya_high_quality_factory_v1/'
REPOSITORY = 'LUNCY579/rineya-world-model-research'
BRANCH = 'research/rineya-high-quality-dataset-v1'
COMMIT = 'bc5198e11273cb6ea69659b6d3a8b5dc312b6f00'
PARENT = '448c36af9370460782da3c0a4ba7c5ca4d8ba71a'
ORIGINAL_HEAD = 'a223884d55ff47baf3890c7d4a82054d07223215'
AUTHOR = '/root/backup_manager'
JOIN_PATH = 'backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json'
ADDENDUM_PATH = 'handoff/RGM_V2_RENDERER_BACKUP_COMPLETED_ADDENDUM.md'
FREEZE_PATH = 'backup_logs/RGM_V2_RENDERER_JOIN_SOURCE_FREEZE.json'
EXPECTED = {
    'backup_logs/rgm-v2-renderer-review-milestone_SPEC.json': 'ee1c503b896b5e705c181f0fadcfe92268fd9859bb55a348eab70e90d94c24f1',
    'backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json': '146cb0b19bc1fda50ae810823457b0e3fce70dc4dddb0a1a779d790f0e173937',
    'quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json': '0fcc58c0453a7942b22a6b585bad1f6ddecbe460a3139f7b6ea4aac06bb4be16',
    'backup_logs/rgm-v2-renderer-review-milestone_STAGE_R2.json': 'c27836578aae3b512b632f869b55f140df974e4275526d90b40de74dbd91ce3c',
    'backup_logs/RGM_RENDERER_MILESTONE_CUSTODY_ADAPTER.json': '8a1b0f1af45128ef8cf5ce36e3831cb12595b40ecb9e858609720675fe2a0fa7',
    'backup_logs/RGM_V2_RENDERER_PUSHED_INDEPENDENT_CUSTODY_AUDIT.json': '3a4c2073351260a8d9c1e17dd204583f61bf3dba2e25a77cb808a88d5ab7faa1',
    'backup_logs/rgm-v2-renderer-review-milestone_STAGE.json': 'cbcd3bf662ff03bdd01fef9263ff407592c4b430ef6f51aaaeadc8f4baf2a795',
    'backup_logs/rgm-v2-renderer-review-milestone_STAGE_LEDGER.jsonl': '36c510a59cbfac7ceda719d44bdf293c35d07aba87f089c0624461359e3ec8e5',
    'backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LEDGER.jsonl': 'cd963ce9095f7bf9ae57ca22bc2fbf58b7ae944d5fb3da80c24d129fee732238',
    'backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LOSSLESS_LEDGER.jsonl': '6fb9565ac18315560b9cc76e7802ab9a934ab363fbf4a0227a93d81aca32b751',
    'backup_logs/recover_remote_checkpoint.py': '3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1',
    'backup_logs/verify_archival_remote_lossless.py': '7bab2c46dd1e7a52520a90617a2d910b5d6987bad8ec2489b05097045505a0aa',
    'backup_logs/ARCHIVAL_REMOTE_LOSSLESS_SOURCE_FREEZE.json': 'ad9e0d0c8b27df84f2e113f4fe2777c2fc48ec3958e6cf2789eaebe85c85fa29',
    'backup_logs/run_provisional_archival_checkpoint.py': 'a77e440f29afc8aca9131d3b5906fb483987acf432654f5e41d00fbcc2235ec9',
    'backup_logs/checkpoint_backup.py': '07ef424f92d903c265c6cc06622733dcdeabf5bd03d3c13d98f793c2ddcc2d4b',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(name, expected=None):
    p = FACTORY / name
    digest = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    result = {'path': name, 'size': p.stat().st_size, 'sha256': digest.hexdigest()}
    require(expected is None or result['sha256'] == expected, 'Changed evidence: ' + name)
    return result


def data(name):
    return json.loads((FACTORY / name).read_text())


def command(args, cwd=None):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(args, cwd=cwd, env=env, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=60).stdout


def git(cwd, *args):
    return command(['git', '-C', str(cwd), *args])


def baseline_check(adapter):
    items = adapter['protected_git_byte_baselines']
    require(len(items) == 6, 'Six protected Git baseline entries required')
    output = []
    for item in items:
        p = Path(item['path'])
        require(p.parent == ORIGINAL / '.git' or p == ORIGINAL / '.git/refs/heads/main', 'Unexpected protected path')
        require(p.exists() is item['expected_exists'], 'Protected existence changed')
        record = dict(item, actual_exists=p.exists())
        if p.exists():
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            require(actual == item['sha256'], 'Protected Git bytes changed: ' + item['label'])
            record['actual_sha256'] = actual
        record['matches_frozen_baseline'] = True
        output.append(record)
    require(len({x['label'] for x in items}) == 6, 'Duplicate baseline labels')
    return output


def tree(commit):
    result = {}
    for line in git(WORKTREE, 'ls-tree', '-rlz', commit, '--', PREFIX).split(b'\0'):
        if not line:
            continue
        info, path = line.split(b'\t', 1)
        mode, kind, blob, size = info.split()
        name = path.decode()
        require(name.startswith(PREFIX) and kind == b'blob', 'Unexpected committed tree entry')
        result[name[len(PREFIX):]] = {'mode': mode.decode(), 'blob': blob.decode(), 'size': int(size)}
    return result


def ledger(name):
    with (FACTORY / name).open() as stream:
        return [json.loads(line) for line in stream]


def exclusive(path, value):
    with (FACTORY / path).open('x') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    require(not (FACTORY / JOIN_PATH).exists() and not (FACTORY / ADDENDUM_PATH).exists(), 'Fresh output paths required')
    source = read(str(Path(__file__).relative_to(FACTORY)))
    freeze = data(FREEZE_PATH)
    require(freeze['source_sha256'] == source['sha256'] and freeze['author_execution'] == AUTHOR, 'Source was not frozen before execution')
    evidence = {name: read(name, expected) for name, expected in EXPECTED.items()}
    spec = data('backup_logs/rgm-v2-renderer-review-milestone_SPEC.json')
    full = data('backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json')
    gzip = data('quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json')
    stage = data('backup_logs/rgm-v2-renderer-review-milestone_STAGE_R2.json')
    adapter = data('backup_logs/RGM_RENDERER_MILESTONE_CUSTODY_ADAPTER.json')
    audit = data('backup_logs/RGM_V2_RENDERER_PUSHED_INDEPENDENT_CUSTODY_AUDIT.json')
    baseline_before = baseline_check(adapter)
    files = {x['path']: x for x in spec['files']}
    require(len(files) == 2095 and sum(x['size'] for x in files.values()) == 886553408, 'Frozen scope changed')
    require(full['status'] == 'BACKED_UP' and full['commit_sha'] == full['remote_commit_sha'] == COMMIT, 'Full proof not complete')
    require(full['inventory'] == spec['files'] and full['files'] == 2095 and full['total_bytes'] == 886553408, 'Full inventory mismatch')
    require(full['spec_sha256'] == EXPECTED['backup_logs/rgm-v2-renderer-review-milestone_SPEC.json'], 'Spec binding mismatch')
    require(not full['current_source_evolution'] and not full['current_source_evolution_at_finish'], 'Unexpected evolved sources')
    require(full['runner_execution'] == '/root' and full['verification_source_sha256'] == EXPECTED['backup_logs/recover_remote_checkpoint.py'], 'Full verifier identity mismatch')
    require(stage['commit_sha'] == COMMIT and stage['expected_parent_sha'] == PARENT and stage['strict_diff_check'] == 'PASS', 'Stage custody mismatch')
    require(stage['status'] == 'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING' and stage['all_source_target_index_sha256_and_sha1_verified'] is True, 'Historical stage state changed')
    require(stage['runner_execution'] == '/root' and audit['commit_sha'] == COMMIT and audit['immutable_prior_release_files_verified'] == 198, 'Stage/audit roles mismatch')
    require(gzip['status'] == 'PASS_ACTUAL_REMOTE_GZIP_DECODE' and gzip['commit_sha'] == COMMIT, 'Gzip proof not complete')
    require(gzip['actual_remote_gzip_fetches'] == gzip['oversized_raw_artifacts'] == 24 and gzip['raw_bytes_exactly_recovered'] == 1188634377, 'Gzip counts mismatch')
    require(gzip['verified_inventory_receipt_sha256'] == EXPECTED['backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json'] and gzip['stage_receipt_sha256'] == EXPECTED['backup_logs/rgm-v2-renderer-review-milestone_STAGE_R2.json'], 'Gzip proof bindings mismatch')
    require(gzip['source_sha256'] == EXPECTED['backup_logs/verify_archival_remote_lossless.py'] and gzip['actual_runner_execution'] == AUTHOR and gzip['grants_data_approval'] is False, 'Gzip source/role mismatch')
    blobs = full['remote_verification']['blob_receipts']
    require(len(blobs) == full['remote_verification']['verified_unique_blobs'] == 1740 and full['remote_verification']['verified_files'] == 2095, 'Actual full fetch count mismatch')
    coverage = {}
    for blob in blobs:
        require(blob['verification_source'] == 'actual_remote_raw_GitHub_blob', 'Non-actual blob provenance')
        for name in blob['paths']:
            require(name not in coverage and name in files, 'Duplicate/unexpected remote path')
            require((blob['size'], blob['sha256']) == (files[name]['size'], files[name]['sha256']), 'Remote byte binding mismatch')
            coverage[name] = blob
    require(set(coverage) == set(files), 'Incomplete exact remote coverage')
    raw = {x['path']: x for x in spec['raw_size_exclusions']}
    proofs = gzip['actual_remote_gzip_proofs']
    require(len(raw) == len(proofs) == 24 and {p['raw_path'] for p in proofs} == set(raw), 'Raw gzip coverage mismatch')
    for proof in proofs:
        require(proof['actual_remote_fetch_this_run'] is True and proof['decoded_byte_for_byte_equal_to_preserved_raw'] is True, 'Missing actual gzip byte proof')
        require((proof['raw_bytes'], proof['raw_sha256']) == (raw[proof['raw_path']]['size'], raw[proof['raw_path']]['sha256']), 'Raw frozen bytes mismatch')
        blob = coverage[proof['remote_gzip_path']]
        require((proof['remote_git_blob'], proof['remote_gzip_bytes'], proof['remote_gzip_sha256']) == (blob['git_blob'], blob['size'], blob['sha256']), 'Full/gzip remote bytes mismatch')
    full_ledger = ledger('backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LEDGER.jsonl')
    actual_events = [x for x in full_ledger if x['phase'] == 'actual_remote_blob_verified']
    require(len(actual_events) == 1740 and full_ledger[-1]['status'] == 'BACKED_UP', 'Full ledger incomplete')
    require({(x['git_blob'], x['size'], x['sha256']) for x in actual_events} == {(x['git_blob'], x['size'], x['sha256']) for x in blobs}, 'Full receipt/ledger mismatch')
    gzip_ledger = ledger('backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LOSSLESS_LEDGER.jsonl')
    require(sum(x['phase'] == 'ACTUAL_REMOTE_GZIP_BYTE_PROOF' for x in gzip_ledger) == 24 and gzip_ledger[-1]['status'] == gzip['status'], 'Gzip ledger incomplete')
    require(git(ORIGINAL, 'rev-parse', 'HEAD').decode().strip() == ORIGINAL_HEAD and git(ORIGINAL, 'branch', '--show-current').decode().strip() == 'main', 'Original main custody changed')
    require(git(WORKTREE, 'rev-parse', 'HEAD').decode().strip() == COMMIT and git(WORKTREE, 'branch', '--show-current').decode().strip() == BRANCH, 'Dedicated custody changed')
    require(not git(ORIGINAL, 'status', '--porcelain=v1', '-z') and not git(WORKTREE, 'status', '--porcelain=v1', '-z'), 'Checkout not clean')
    require(git(WORKTREE, 'rev-list', '--parents', '-n', '1', COMMIT).decode().split() == [COMMIT, PARENT], 'Commit sole parent changed')
    committed = tree(COMMIT)
    require(set(committed) == set(files), 'Committed exact tree mismatch')
    for name, item in committed.items():
        require(item == {'mode': '100644', 'blob': coverage[name]['git_blob'], 'size': files[name]['size']}, 'Tree/full receipt mismatch')
    index = {}
    for line in git(WORKTREE, 'ls-files', '-s', '-z', '--', PREFIX).split(b'\0'):
        if line:
            info, path = line.split(b'\t', 1)
            mode, blob, stage_number = info.split()
            require(stage_number == b'0', 'Unmerged index')
            index[path.decode()[len(PREFIX):]] = (mode.decode(), blob.decode())
    require(index == {n: (x['mode'], x['blob']) for n,x in committed.items()}, 'Index differs from frozen commit')
    prior = tree(PARENT)
    release_names = [n for n in prior if '/releases/' in n]
    require(len(prior) == 1801 and set(prior).issubset(committed), 'Previously tracked paths absent')
    require(len(release_names) == 198 and all(prior[n] == committed[n] for n in release_names), 'Historical release bytes changed')
    require(git(WORKTREE, 'diff', '--check', PARENT, COMMIT) == b'', 'Strict commit whitespace check failed')
    sparse = git(WORKTREE, 'sparse-checkout', 'list').decode().splitlines()
    require(set(sparse) == {'tools', 'datasets/rineya_high_quality_factory_v1'}, 'Sparse scope changed')
    require(git(WORKTREE, 'config', '--worktree', '--get', 'index.sparse').decode().strip() == 'false', 'Unexpected sparse index')
    private = json.loads(command(['gh', 'repo', 'view', REPOSITORY, '--json', 'nameWithOwner,isPrivate,viewerPermission']))
    require(private == {'nameWithOwner': REPOSITORY, 'isPrivate': True, 'viewerPermission': 'ADMIN'}, 'Private ADMIN custody unavailable')
    ref = json.loads(command(['gh', 'api', 'repos/' + REPOSITORY + '/git/ref/heads/' + BRANCH]))
    require(ref['object']['sha'] == COMMIT and ref['object']['type'] == 'commit', 'Actual remote branch changed')
    remote_commit = json.loads(command(['gh', 'api', 'repos/' + REPOSITORY + '/git/commits/' + COMMIT]))
    require(remote_commit['tree']['sha'] == full['remote_verification']['remote_tree_sha'] == gzip['actual_remote_tree_sha'] and [x['sha'] for x in remote_commit['parents']] == [PARENT], 'Actual remote tree/parent changed')
    baseline_after = baseline_check(adapter)
    require(baseline_after == baseline_before, 'Custody changed during read-only join')
    require(all(read(name, expected)['sha256'] == expected for name,expected in EXPECTED.items()), 'Evidence changed during join')
    local_names = sorted(str(p.relative_to(FACTORY)) for base in ['backup_logs', 'handoff', 'quality_reports', 'reviews', 'audits'] for p in (FACTORY/base).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and str(p.relative_to(FACTORY)) not in files and (('renderer-r1' in p.name or 'renderer_r1' in p.name) or base in ['backup_logs','handoff']))
    joined = {
        'status': 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED',
        'author_execution': AUTHOR, 'runner_execution': AUTHOR, 'joined_at_utc': now(),
        'model_disclosure': 'Inherited same model; independent execution does not imply model diversity',
        'checkpoint_id': spec['checkpoint_id'], 'repository': REPOSITORY, 'branch': BRANCH,
        'commit_sha': COMMIT, 'remote_commit_sha': COMMIT, 'sole_parent_sha': PARENT,
        'approval_type': 'Provisional archival checkpoint only', 'grants_data_approval': False,
        'frozen_scope': {'spec': evidence['backup_logs/rgm-v2-renderer-review-milestone_SPEC.json'], 'files': 2095, 'bytes': 886553408, 'exact_path_coverage': True, 'all_previous_1801_paths_preserved': True, 'historical_release_files_unchanged': 198},
        'actual_remote_inventory_verification': evidence['backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json'],
        'actual_remote_inventory_runner': '/root', 'actual_remote_unique_blob_fetches_by_root': 1740,
        'actual_remote_inventory_ledger': evidence['backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LEDGER.jsonl'],
        'actual_remote_lossless_verification': evidence['quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json'],
        'actual_remote_lossless_runner': AUTHOR, 'new_actual_gzip_fetches_by_backup_manager': 24,
        'raw_artifacts_exactly_recovered': 24, 'raw_bytes_exactly_recovered': 1188634377,
        'actual_remote_lossless_ledger': evidence['backup_logs/rgm-v2-renderer-review-milestone_REMOTE_LOSSLESS_LEDGER.jsonl'],
        'root_stage_r2_receipt': evidence['backup_logs/rgm-v2-renderer-review-milestone_STAGE_R2.json'],
        'independent_stage_review': evidence['backup_logs/RGM_V2_RENDERER_PUSHED_INDEPENDENT_CUSTODY_AUDIT.json'],
        'preserved_prior_failed_stage': evidence['backup_logs/rgm-v2-renderer-review-milestone_STAGE.json'],
        'preserved_prior_failed_stage_ledger': evidence['backup_logs/rgm-v2-renderer-review-milestone_STAGE_LEDGER.jsonl'],
        'schema_adapter': evidence['backup_logs/RGM_RENDERER_MILESTONE_CUSTODY_ADAPTER.json'],
        'preserved_prior_status_boundary': 'Original FAILED and PUSHED_REMOTE_BYTES_VERIFICATION_PENDING receipts remain byte-identical historical execution evidence; this new receipt joins completed proofs without rewriting them',
        'private_admin': private, 'actual_remote_ref_sha': ref['object']['sha'], 'actual_remote_tree_sha': remote_commit['tree']['sha'],
        'original_custody': {'head': ORIGINAL_HEAD, 'branch': 'main', 'clean': True, 'optional_git_locks_disabled': True},
        'dedicated_custody': {'head': COMMIT, 'branch': BRANCH, 'clean': True, 'sparse_paths': sparse, 'full_index': True, 'exact_index_tree_verified': True},
        'protected_git_byte_baselines': baseline_after,
        'exact_helpers': {n: e for n,e in evidence.items() if n.endswith('.py') or n.endswith('SOURCE_FREEZE.json')},
        'join_source': source, 'join_source_freeze': read(FREEZE_PATH),
        'snapshot_deferred_paths': spec['deferred_incomplete_review_paths'],
        'post_inventory_local_names_only_observed': local_names,
        'post_inventory_boundary': 'Only the 2095 exact files in the bound spec are claimed in bc519. Current checkpoint spec, failure/adapter/stage/remote verification records, this new gzip proof/ledger/preexecution, this join/source/freeze/addendum, and newly born renderer reviewer artifacts remain local until a later exact checkpoint. No current unlisted bytes are claimed backed up.',
        'prospective_runner_v2_state': 'New postinventory runner/test drafts are unfinished and untested; no readiness or execution claim',
        'verification_execution_boundary': 'Root separately fetched and verified 1740 actual remote blob groups. Backup manager separately fetched 24 actual remote gzip payloads and compared decoded bytes to preserved local raw. This join inspected immutable receipts, ledgers, local tree/index object IDs, custody and actual GitHub metadata; it did not repeat payload fetches or infer remote success from local Git.',
        'self_inventory_boundary': 'This new receipt cannot contain its own final digest or be part of its earlier frozen spec; preserve actual receipt/addendum bytes in a later checkpoint',
        'new_handoff_addendum': ADDENDUM_PATH, 'data_or_release_approval': False,
        'concrete_current_custody_blockers': [],
    }
    exclusive(JOIN_PATH, json.dumps(joined, indent=2, sort_keys=True) + '\n')
    joined_hash = read(JOIN_PATH)['sha256']
    text = f'''# RGM V2 renderer milestone backup completed\n\nActual joined custody: `backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json` SHA256 `{joined_hash}`.\n\nCommit `{COMMIT}` on private `{REPOSITORY}` branch `{BRANCH}` contains the exact frozen **2095 files / 886,553,408 bytes**, spec SHA256 `{EXPECTED['backup_logs/rgm-v2-renderer-review-milestone_SPEC.json']}`. Its sole parent is `{PARENT}`. Root completed actual raw GitHub verification of **1740 unique blobs**, proof SHA256 `{EXPECTED['backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json']}`.\n\nIndependent backup manager `/root/backup_manager` ran the frozen generic verifier and fetched **24 new actual GitHub gzip payloads**. Every decoded stream matched preserved local raw byte-for-byte, recovering **1,188,634,377 raw bytes**; proof SHA256 `{EXPECTED['quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json']}`. SHA256, size and Git blob SHA1 bindings match the completed full proof. The verifier used 64 KiB streaming and an 8 MiB memory spool. Same inherited model; independence is by execution, with no model diversity claim.\n\nFresh read-only checks confirmed private ADMIN, actual remote branch/tree and sole parent, exact committed/index paths, all prior 1801 tracked paths and 198 historical release files preserved, clean dedicated checkout, clean original main `{ORIGINAL_HEAD}`, and all six protected original Git byte/existence baselines unchanged. No Git mutation was performed by this proof or join. Original FAILED stage and PUSHED_PENDING receipts, the actual schema adapter and independent stage audit remain unchanged historical evidence.\n\nOnly the bound 2095-file inventory is claimed backed up. Current postinventory stage/recovery/proof/ledger/spec/join/addendum records and newborn renderer reviewer artifacts remain local pending a later exact checkpoint. The prospective v2 runner/test drafts remain unfinished and untested. This archival result grants no candidate approval, release approval or training authorization. Root owns any subsequent living-document refresh.\n'''
    exclusive(ADDENDUM_PATH, text)
    print(json.dumps({'status': joined['status'], 'receipt': read(JOIN_PATH), 'addendum': read(ADDENDUM_PATH), 'actual_new_gzip_fetches': 24, 'raw_bytes': 1188634377}))


if __name__ == '__main__':
    main()
