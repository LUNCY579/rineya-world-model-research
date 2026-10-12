"""Independent compaction preflight: local Git only, all network/auth forbidden.

Every fixture has a disposable GCDL host with a primary repository and its
dedicated linked worktree. Small opaque bytes are hand specified; there is no
dataset generation, production access, sealed-label parse or actual eviction.
Custody control helpers are mocked only to reach later local preflight guards.
Each refusal must preserve every existing file byte, inode and symlink target.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.scripts import compact_git as compact


class IndependentCompactionPreflight(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-independent-compact-guards-')
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.host = self.base / 'gcdl001-host'
        self.repository = self.host / 'repository'
        self.repository.mkdir(parents=True)
        self.environment = {
            'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0',
            'GIT_ALLOW_PROTOCOL': '', 'GIT_NO_LAZY_FETCH': '1',
        }
        environment = patch.dict(os.environ, self.environment)
        environment.start(); self.addCleanup(environment.stop)
        self.git(self.repository, 'init', '-q', '-b', 'main')
        self.git(self.repository, 'config', 'user.name', 'independent-local-fixture')
        self.git(self.repository, 'config', 'user.email', 'independent@example.invalid')
        self.original_root = self.repository / archive.PROGRAMME
        self.original_root.mkdir()
        (self.repository / '.gitignore').write_text(archive.PROGRAMME + '/state/\n')
        (self.original_root / 'code.py').write_bytes(b'hand specified retained fixture source\n')
        self.relative = 'data/accepted/train-730001.jsonl'
        self.raw = b'opaque hand specified nonsealed fixture bytes\n'
        source = self.original_root / self.relative
        source.parent.mkdir(parents=True); source.write_bytes(self.raw)
        self.git(self.repository, 'add', '.')
        self.git(self.repository, 'commit', '-qm', 'local fixture original bytes')
        self.primary_head = self.git(self.repository, 'rev-parse', 'HEAD').strip()
        self.git(self.repository, 'branch', compact.BRANCH)
        self.worktree = self.host / 'worktree'
        self.git(self.repository, 'worktree', 'add', '-q', str(self.worktree), compact.BRANCH)
        self.root = self.worktree / archive.PROGRAMME
        self.git(self.worktree, 'rm', '-q', archive.PROGRAMME + '/' + self.relative)
        self.git(self.worktree, 'commit', '-qm', 'local fixture active branch excludes cold bytes')
        self.head = self.git(self.worktree, 'rev-parse', 'HEAD').strip()
        self.git(self.repository, 'remote', 'add', 'origin', compact.REMOTE)
        self.git(self.repository, 'update-ref', 'refs/remotes/origin/main', self.primary_head)
        self.git(self.repository, 'update-ref', 'refs/remotes/origin/' + compact.BRANCH, self.head)
        self.state = self.root / 'state'; self.state.mkdir()
        (self.state / 'PAUSED.json').write_text('{"reason":"independent local fixture"}\n')
        (self.state / 'backup.json').write_text(json.dumps({
            'status': 'confirmed', 'commit': self.head, 'remote_commit': self.head,
        }))
        (self.state / 'worker.lock').touch()
        (self.state / 'queue.db').write_bytes(b'fixed fixture control bytes\n')
        catalog_path = 'state/fixture_catalog.json'
        (self.root / catalog_path).write_text('{"independent_fixture":true}\n')
        self.index = {
            'files': {self.relative: {
                'git_blob_sha1': archive.git_blob_sha1(self.raw),
                'sha256': hashlib.sha256(self.raw).hexdigest(), 'bytes': len(self.raw),
                'commit': self.primary_head, 'catalog_id': '1' * 64, 'receipt_id': '2' * 64,
            }},
            'catalogs': {'1' * 64: {'catalog_path': catalog_path, 'catalog_commit': self.primary_head}},
        }
        self.forbidden = {}
        for name, owner in (
            ('_owner_environment', compact), ('_server_heads', compact),
            ('_remote_proof', archive), ('GitHubTransport', archive),
        ):
            hook = patch.object(owner, name, side_effect=AssertionError(
                'authentication/network must not run in local preflight: ' + name))
            self.forbidden[name] = hook.start(); self.addCleanup(hook.stop)
        original_command = compact._command

        def local_command(directory, command, env, timeout=45, max_output=8 * 1024 ** 2):
            self.assertEqual(command[0], 'git', 'preflight attempted credential or external command')
            self.assertFalse({'clone', 'fetch', 'push', 'pull', 'ls-remote'} & set(command),
                             'preflight attempted Git transport')
            self.assertEqual(env.get('GIT_ALLOW_PROTOCOL'), '', 'local Git must forbid protocols')
            self.assertEqual(env.get('GIT_NO_LAZY_FETCH'), '1', 'local Git must forbid lazy fetch')
            return original_command(directory, command, env, timeout, max_output)

        for hook in (
            patch.object(compact, '_command', side_effect=local_command),
            patch.object(archive, 'load_index', side_effect=lambda root: deepcopy(self.index)),
            patch.object(archive, 'deletion_custody', side_effect=lambda root, relative: self.index['files'][relative]),
        ):
            hook.start(); self.addCleanup(hook.stop)
        review = patch('goal_conditioned_decision_data.auditors.release.current_independent_review', return_value=True)
        self.review = review.start(); self.addCleanup(review.stop)

    def git(self, directory, *args):
        # This setup helper executes only local metadata/worktree commands.
        self.assertFalse({'clone', 'fetch', 'push', 'pull', 'ls-remote'} & set(args))
        return subprocess.run(['git', *args], cwd=directory, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def snapshot(self):
        result = {}
        for path in sorted(self.base.rglob('*')):
            details = path.lstat()
            identity = (details.st_dev, details.st_ino, details.st_mode, details.st_nlink)
            if path.is_symlink():
                content = ('symlink', os.readlink(path))
            elif path.is_file():
                content = ('file', path.read_bytes())
            else:
                content = ('directory',)
            result[str(path.relative_to(self.base))] = (identity, content)
        return result

    def refuse(self, message, root=None):
        before = self.snapshot()
        try:
            with self.assertRaisesRegex(RuntimeError, message):
                compact.compact(self.root if root is None else root, seconds=15)
        finally:
            self.assertEqual(self.snapshot(), before, 'preflight changed local bytes or ownership')
            for mock in self.forbidden.values():
                mock.assert_not_called()

    def test_foreign_host_refuses_before_review_auth_or_network(self):
        foreign = self.base / 'unrelated-host' / 'worktree' / archive.PROGRAMME
        foreign.mkdir(parents=True); (foreign / 'sentinel').write_bytes(b'unrelated local bytes')
        self.refuse('restricted to the owned GCDL host', root=foreign)
        self.review.assert_not_called()

    def test_symlink_root_and_repository_refuse_without_following_external_bytes(self):
        alias = self.base / 'programme-alias'; alias.symlink_to(self.root, target_is_directory=True)
        self.refuse('root cannot be a symlink', root=alias)
        primary = self.host / 'saved-primary'; self.repository.rename(primary)
        self.repository.symlink_to(primary, target_is_directory=True)
        self.refuse('unsafe owned Git repository path')

    def test_stop_missing_pause_and_missing_review_are_local_refusals(self):
        stop = self.state / 'STOP'; stop.write_bytes(b'authorized stop sentinel\n')
        self.refuse('STOP prevents'); stop.unlink()
        pause = self.state / 'PAUSED.json'; raw = pause.read_bytes(); pause.unlink()
        self.refuse('requires the worker pause'); pause.write_bytes(raw)
        self.review.return_value = False
        self.refuse('independent review required')

    def test_wrong_branch_and_wrong_remote_refuse_locally(self):
        self.git(self.worktree, 'checkout', '-qb', 'unrelated-fixture-branch')
        self.refuse('branch guard failed')
        self.git(self.worktree, 'checkout', '-q', compact.BRANCH)
        self.git(self.repository, 'remote', 'set-url', 'origin', 'https://example.invalid/unrelated.git')
        self.refuse('remote guard failed')

    def test_dirty_active_branch_and_existing_pointer_collision_preserve_bytes(self):
        code = self.root / 'code.py'; raw = code.read_bytes(); code.write_bytes(b'unpublished active bytes')
        self.refuse('clean confirmed code branch'); code.write_bytes(raw)
        collision = self.worktree / '.git.compact-tmp'; collision.write_bytes(b'unrelated existing pointer bytes')
        self.refuse('clean confirmed code branch')

    def test_dirty_untracked_and_ignored_primary_preserve_every_byte(self):
        for kind in ('tracked', 'untracked', 'ignored'):
            with self.subTest(kind=kind):
                relative = {'tracked': 'code.py', 'untracked': 'unpublished.bin', 'ignored': 'state/ignored.bin'}[kind]
                path = self.original_root / relative
                raw = path.read_bytes() if path.exists() else None
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'unpublished primary ' + kind.encode())
                self.refuse('primary Git checkout is dirty/untracked')
                if raw is None:
                    path.unlink()
                else:
                    path.write_bytes(raw)
                if kind == 'ignored':
                    path.parent.rmdir()

    def test_primary_store_symlink_or_shared_hardlink_never_deletes_external_bytes(self):
        outside = self.base / 'external-sentinel'; outside.write_bytes(b'external user bytes stay exact')
        linked = self.repository / '.git' / 'foreign-link'; linked.symlink_to(outside)
        self.refuse('symlink/shared hardlink'); linked.unlink()
        os.link(outside, linked)
        self.refuse('symlink/shared hardlink')

    def test_extra_real_worktree_refuses_and_preserves_all_worktree_files(self):
        other = self.host / 'other-worktree'
        self.git(self.repository, 'worktree', 'add', '-q', '-b', 'independent-third-fixture', str(other), self.primary_head)
        (other / 'unpublished.bin').write_bytes(b'third worktree belongs to a user')
        self.refuse('other worktrees must be preserved')

    def test_tag_and_stash_refs_refuse_before_archive_proof_or_owner_auth(self):
        self.git(self.repository, 'tag', 'independent-unpublished-tag')
        self.refuse('stash/tag/unpublished reference')
        self.git(self.repository, 'tag', '-d', 'independent-unpublished-tag')
        self.git(self.repository, 'update-ref', 'refs/stash', self.primary_head)
        self.refuse('stash/tag/unpublished reference')

    def test_divergent_local_origin_mirror_refuses_before_archive_proof_or_owner_auth(self):
        self.git(self.repository, 'update-ref', 'refs/remotes/origin/' + compact.BRANCH, self.primary_head)
        self.refuse('not mirrored by origin')

    def test_extra_unmirrored_local_head_refuses_before_archive_proof_or_owner_auth(self):
        self.git(self.repository, 'branch', 'independent-unpublished-local-head', self.primary_head)
        self.refuse('not mirrored by origin|unpublished reference')

    def test_unconfirmed_backup_and_remaining_local_payload_refuse_without_transport(self):
        backup = self.state / 'backup.json'; raw = backup.read_bytes()
        backup.write_text(json.dumps({'status': 'confirmed', 'commit': self.primary_head, 'remote_commit': self.head}))
        self.refuse('confirmed GitHub code backup'); backup.write_bytes(raw)
        # Use an ignored state report, so the clean-code guard remains satisfied.
        relative = 'state/fixture_catalog.json'; original = self.index
        self.index = {'files': {relative: next(iter(original['files'].values()))}, 'catalogs': original['catalogs']}
        self.refuse('local archived payload still exists')


if __name__ == '__main__':
    unittest.main()
