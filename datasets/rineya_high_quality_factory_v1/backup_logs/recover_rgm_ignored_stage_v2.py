"""One frozen checkpoint recovery: explicit14ignored audit bytes, no recopy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parent / '.rineya-dataset-git-worktree'
ORIGINAL = ROOT.parent / 'rineya-world-model-research'
PREFIX = 'datasets/rineya_high_quality_factory_v1/'
BRANCH = 'research/rineya-high-quality-dataset-v1'
PARENT = 'b46d96a5db0f1cc94e894f15d61942af49e61819'
SPEC_SHA = 'd669b0b151a26e51892c7b69281cf1215a924835a8a735018d2ef5adfa553ed5'
NAMES = ['common_HEAD', 'common_config', 'dedicated_HEAD', 'dedicated_commondir',
         'dedicated_git_pointer', 'dedicated_gitdir', 'dedicated_index', 'dedicated_ref',
         'dedicated_sparse_rules', 'dedicated_worktree_config', 'main_ref', 'original_index',
         'packed_refs', 'remote_ref']
IGNORED = {f'backup_logs/SPARSE_DATASET_RESTORE_20261010T050841Z/pre_bytes/{name}.bin' for name in NAMES}
RECEIPT = ROOT / 'backup_logs/rgm-v1-final-review_STAGE_RECOVERY.json'


def run(args, cwd=WORKTREE, allowed=(0,), stdin=None):
    result = subprocess.run(args, cwd=cwd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in allowed:
        raise RuntimeError(f'{args[:3]}: {result.stderr.decode()}')
    return result.stdout


def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def persist(report):
    temporary = RECEIPT.with_suffix('.new.json')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(RECEIPT)


def main():
    assert not RECEIPT.exists(), 'Preserve any earlier recovery; use a new source/version'
    spec_path = ROOT / 'backup_logs/rgm-v1-final-review_SPEC.json'
    assert digest(spec_path)[1] == SPEC_SHA
    spec = json.loads(spec_path.read_text())
    assert spec['validation_status'] == 'passed' and spec['approval'] == 'provisional_checkpoint'
    entries = spec['files']
    inventory = {entry['path']: entry for entry in entries}
    assert len(entries) == len(inventory) == 1801 and IGNORED <= inventory.keys()
    failed_path = ROOT / 'backup_logs/rgm-v1-final-review_RECEIPT.json'
    failed = json.loads(failed_path.read_text())
    assert failed['status'] == 'FAILED_LOCAL_ARTIFACTS_PRESERVED' and 'git --literal-pathspecs add failed' in failed['error']
    assert failed.get('commit_sha') is None
    failed_sha = digest(failed_path)[1]
    original = {'head': run(['git', 'rev-parse', 'HEAD'], ORIGINAL).decode().strip(),
                'branch': run(['git', 'branch', '--show-current'], ORIGINAL).decode().strip(),
                'status': run(['git', 'status', '--porcelain=v1'], ORIGINAL).decode().strip()}
    assert original == {'head': 'a223884d55ff47baf3890c7d4a82054d07223215', 'branch': 'main', 'status': ''}
    assert run(['git', 'rev-parse', 'HEAD']).decode().strip() == PARENT
    assert run(['git', 'branch', '--show-current']).decode().strip() == BRANCH
    remote = json.loads(run(['gh', 'repo', 'view', 'LUNCY579/rineya-world-model-research', '--json', 'nameWithOwner,isPrivate,viewerPermission']))
    assert remote['nameWithOwner'] == 'LUNCY579/rineya-world-model-research' and remote['isPrivate'] and remote['viewerPermission'] == 'ADMIN'
    remote_before = run(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH]).decode().split()[0]
    assert remote_before == PARENT
    status = run(['git', 'status', '--porcelain=v1', '-z'])
    for item in filter(None, status.split(b'\0')):
        state, path = item[:2].decode(), item[3:].decode()
        assert state in {'A ', 'M ', '??'} and path.startswith(PREFIX) and path[len(PREFIX):] in inventory, (state, path)
    paths = [PREFIX + name for name in sorted(inventory)]
    ignored_actual = {path.decode()[len(PREFIX):] for path in run(['git', 'check-ignore', '--no-index', '-z', '--stdin'], allowed=(0, 1), stdin=('\0'.join(paths)+'\0').encode()).split(b'\0') if path}
    assert ignored_actual == IGNORED, ignored_actual
    for name, entry in inventory.items():
        assert digest(ROOT / name) == (entry['size'], entry['sha256']), name
        assert digest(WORKTREE / PREFIX / name) == (entry['size'], entry['sha256']), name
    report = {'actual_runner_execution': '/root', 'source_sha256': digest(Path(__file__))[1],
              'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'VERIFIED_COPY_STAGE_RECOVERY_STARTED',
              'spec_sha256': SPEC_SHA, 'original_failed_receipt_sha256': failed_sha,
              'failed_receipt_unchanged': True, 'inventory_files': 1801,
              'inventory_bytes': sum(entry['size'] for entry in entries), 'private_repository_evidence': remote,
              'parent_sha': PARENT, 'original_before': original, 'ignored_exact_paths': sorted(IGNORED),
              'initial_status_sha256': hashlib.sha256(status).hexdigest(),
              'scope': 'Recover already copied frozen archival bytes using exact ignored inventory paths; no recopy, ignore-rule edit, release approval, main change, reset, removal, or force push',
              'self_inventory_boundary': 'This recovery code/receipt and original failed receipt arose after frozen inventory and require a subsequent checkpoint'}
    persist(report)
    try:
        run(['git', '--literal-pathspecs', 'add', '--force', '--', *[PREFIX + name for name in sorted(IGNORED)]])
        staged = run(['git', 'diff', '--cached', '--name-only', '-z'])
        assert all(path.decode().startswith(PREFIX) and path.decode()[len(PREFIX):] in inventory for path in staged.split(b'\0') if path)
        index = {}
        for item in filter(None, run(['git', 'ls-files', '-s', '-z']).split(b'\0')):
            metadata, path = item.split(b'\t', 1)
            mode, blob, stage = metadata.decode().split()
            if path.decode().startswith(PREFIX):
                assert stage == '0'
                index[path.decode()[len(PREFIX):]] = (mode, blob)
        for name, entry in inventory.items():
            assert name in index and index[name][0] == '100644', name
            h = hashlib.sha1(('blob ' + str(entry['size']) + '\0').encode())
            with (ROOT / name).open('rb') as handle:
                for chunk in iter(lambda: handle.read(65536), b''): h.update(chunk)
            assert h.hexdigest() == index[name][1], name
        run(['git', 'diff', '--cached', '--check'])
        report['staged_paths'] = [path.decode() for path in staged.split(b'\0') if path]
        report['all1801_index_blobs_match_frozen_bytes'] = True
        report['diff_stat'] = run(['git', 'diff', '--cached', '--stat']).decode()
        run(['git', 'commit', '-m', 'Preserve RGM planning quarantine and reviewed publication corrections'])
        commit = run(['git', 'rev-parse', 'HEAD']).decode().strip()
        report.update(status='LOCAL_COMMIT_PRESERVED', commit_sha=commit)
        persist(report)
        run(['git', 'push', '--set-upstream', 'origin', 'HEAD:refs/heads/' + BRANCH])
        assert run(['git', 'ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH]).decode().split()[0] == commit
        run(['git', 'fetch', 'origin', 'refs/heads/' + BRANCH + ':refs/remotes/origin/' + BRANCH])
        assert not run(['git', 'status', '--porcelain=v1']).strip()
        assert digest(failed_path)[1] == failed_sha and digest(spec_path)[1] == SPEC_SHA
        assert run(['git', 'rev-parse', 'HEAD'], ORIGINAL).decode().strip() == original['head']
        assert not run(['git', 'status', '--porcelain=v1'], ORIGINAL).strip()
        report.update(status='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING', remote_commit_sha=commit,
                      original_after_unchanged=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
        persist(report)
        print(json.dumps({'status': report['status'], 'commit_sha': commit, 'files': 1801, 'receipt': str(RECEIPT)}))
    except Exception as exc:
        report.update(error=str(exc), status='FAILED_RECOVERY_LOCAL_STATE_PRESERVED')
        persist(report)
        raise


if __name__ == '__main__':
    main()
