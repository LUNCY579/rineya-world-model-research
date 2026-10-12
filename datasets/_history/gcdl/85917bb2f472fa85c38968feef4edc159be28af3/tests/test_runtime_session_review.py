"""Independent runtime/session review using only small owned local fixtures.

No test reads production data, executes the H pilot, contacts GitHub, or
implements the runtime under review. Expectations are stated at policy bounds.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors import release
from goal_conditioned_decision_data.pipelines.common import canonical, digest
from goal_conditioned_decision_data.pipelines import remote_archive as archive, worker
from goal_conditioned_decision_data.tests.test_archive_read_session import ColdCustody, FakeGitHubApi


class IndependentRuntimeSessionReview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-independent-runtime-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve() / archive.PROGRAMME
        self.root.mkdir()
        self.clear_raw_cache()
        self.addCleanup(self.clear_raw_cache)

    def clear_raw_cache(self):
        archive._CACHE.clear()
        archive._CACHE_SIZE = 0

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def local_files(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def test_explicit_transport_deadline_tightens_a_longer_live_session(self):
        # An explicit five-second caller cannot receive a twenty-second provider.
        with patch.object(archive.time, 'monotonic', return_value=100.0), \
                patch.object(archive, '_run', side_effect=AssertionError('no authentication permitted')):
            with archive.read_session(self.root, deadline=120.0) as session:
                provider = archive.read_transport(self.root, deadline=105.0)
                self.assertLessEqual(provider.deadline, 105.0)
                self.assertEqual(session.deadline, 120.0)
                self.assertIs(provider, session.transport)

    def test_expired_explicit_transport_deadline_refuses_without_creating_provider(self):
        with patch.object(archive.time, 'monotonic', return_value=100.0), \
                patch.object(archive, 'GitHubTransport') as provider:
            with archive.read_session(self.root, deadline=120.0) as session:
                with self.assertRaises(archive.ReadDeadlineExceeded):
                    archive.read_transport(self.root, deadline=99.0)
                self.assertIsNone(session._transport)
                provider.assert_not_called()

    def test_opaque_sealed_bytes_remain_explicit_even_after_an_allowed_session_read(self):
        raw = b'opaque fixture, deliberately not JSON or gold\x00\xff\n'
        relative = 'data/sealed/opaque.bin'
        self.write(relative, raw)
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=AssertionError('sealed local fixture needs no remote')):
            with archive.read_session(self.root):
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(self.root, relative)
                self.assertEqual(archive.read_bytes(self.root, relative, allow_sealed=True), raw)
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(self.root, relative)
        self.assertEqual(self.local_files(), before)

    def test_warmed_read_catalog_never_authorizes_destructive_proof_reuse(self):
        fixture = ColdCustody(self.root, count=2)
        self.write('state/worker.lock', b'')
        api = FakeGitHubApi(fixture)
        first = next(iter(fixture.raws))
        receipt_path = next(iter(fixture.index['receipts'].values()))['path']
        receipt = json.loads((self.root / receipt_path).read_text())
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=api):
            with archive.read_session(self.root) as session:
                self.assertEqual(archive.read_bytes(self.root, first), fixture.raws[first])
                self.assertEqual(api.counts['catalog_blob'], 1)
                self.assertEqual(len(session.catalogs), 1)
                # Keep the immutable tree/OID metadata; corrupt only the fresh
                # raw server body after a successful session read.
                oid = fixture.index['catalogs'][fixture.catalog['catalog_id']]['catalog_git_blob_sha1']
                api.blobs[oid] = b'x' * len(api.blobs[oid])
                with self.assertRaises(archive.ArchiveError):
                    archive.evict_verified(self.root, receipt, transport=session.transport)
                self.assertEqual(api.counts['catalog_blob'], 2)
        self.assertEqual(self.local_files(), before)
        self.assertFalse(any((self.root / relative).exists() for relative in fixture.raws))

    def test_complete_review_evidence_map_cannot_drop_inactive_H_custody(self):
        # This fixed reviewer statement names the full map exactly once. A new
        # receipt cannot weaken it by removing evidence for an inactive module.
        import hashlib
        paths = ['manifests/V2_EXPERIMENT_FREEZE.json',
                 'reports/V2_CONTROLLED_EXPERIMENT.json',
                 'manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json',
                 'manifests/H_COMPOSITION_EXPERIMENT_STARTED.json',
                 'reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json']
        evidence = {}
        for number, relative in enumerate(paths):
            raw = ('opaque custody fixture %d\n' % number).encode()
            self.write(relative, raw)
            evidence[relative] = hashlib.sha256(raw).hexdigest()
        code = {'fixture.py': 'a' * 64}
        review = ('FINAL_VERDICT: PASS\n' + digest(code) + '\n' + canonical(evidence)).encode()
        self.write('reports/INDEPENDENT_REVIEW.md', review)
        self.write('configs/production.json', b'{"active_generation_profile":"v2"}\n')
        receipt = {'code_hashes': code, 'source_digest': digest(code),
                   'independent_review_sha256': hashlib.sha256(review).hexdigest(),
                   'required_evidence': evidence}
        self.write('manifests/VALIDATED_CODE.json', canonical(receipt).encode())
        with patch.object(release, 'ROOT', self.root), patch.object(release, 'source_hashes', return_value=code), \
                patch.object(archive, '_run', side_effect=AssertionError('fixture evidence stays local')):
            self.assertTrue(release.current_independent_review())
            for removed in ([p] for p in paths[2:]):
                with self.subTest(removed=removed):
                    altered = {**receipt, 'required_evidence':
                               {p: value for p, value in evidence.items() if p not in removed}}
                    self.write('manifests/VALIDATED_CODE.json', canonical(altered).encode())
                    self.assertFalse(release.current_independent_review())
            altered = {**receipt, 'required_evidence': {p: evidence[p] for p in paths[:2]}}
            self.write('manifests/VALIDATED_CODE.json', canonical(altered).encode())
            self.assertFalse(release.current_independent_review())

    def worker_config(self, seconds=180):
        config = {'archive_enabled': False, 'cycle_timeout_seconds': seconds,
                  'groups_per_cycle': 1, 'seed_start': 0, 'production_splits': ['train'],
                  'max_consecutive_failures': 1, 'backup_interval_seconds': 1800}
        self.write('configs/production.json', json.dumps(config).encode())
        self.write('state/checkpoint.json', b'{"seed":7,"fixture":"immutable"}\n')
        return config

    @staticmethod
    def args():
        return SimpleNamespace(no_backup=False, seal=False, render_only=True, groups=1)

    def test_expiry_before_session_entry_yields_without_database_or_generation(self):
        self.worker_config(seconds=10)
        clock = [100.0]

        def elapsed_preflight(*args, **kwargs):
            clock[0] = 111.0
            return None

        checkpoint = (self.root / 'state/checkpoint.json').read_bytes()
        with patch.object(worker, 'ROOT', self.root), \
                patch.object(worker.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(worker, '_archive_for_resource_pressure', side_effect=elapsed_preflight), \
                patch.object(worker, 'connection') as connection, \
                patch.object(worker, 'process') as process, \
                patch.object(worker, 'backup_if_due') as backup, \
                redirect_stdout(StringIO()):
            self.assertEqual(worker.run(self.args()), 0)
        connection.assert_not_called()
        process.assert_not_called()
        backup.assert_not_called()
        self.assertFalse((self.root / 'state/PAUSED.json').exists())
        self.assertEqual((self.root / 'state/checkpoint.json').read_bytes(), checkpoint)
        heartbeat = json.loads((self.root / 'state/heartbeat.json').read_text())
        self.assertEqual(heartbeat['status'], 'yielded')
        self.assertEqual(heartbeat['reason'], 'bounded_cycle_time')

    def run_failure_case(self, *, approved, recovery_error=None):
        self.worker_config()
        self.write('manifests/VALIDATED_CODE.json', b'{}\n')
        self.write('manifests/SEALED_FREEZE.json', b'{"fixture":"metadata only"}\n')
        with patch.object(worker, 'ROOT', self.root):
            db = worker.connection()
            db.execute('INSERT INTO jobs(seed,split,status,accepted,quarantined,report,profile) '
                       'VALUES(7,"train","done",7,0,"{}","v1")')
            db.commit()
            db.close()
        checkpoint = (self.root / 'state/checkpoint.json').read_bytes()
        with patch.object(worker, 'ROOT', self.root), patch.object(worker, 'guard'), \
                patch.object(worker, '_archive_for_resource_pressure', return_value=None), \
                patch.object(release, 'current_independent_review', return_value=approved), \
                patch.object(worker, 'recover_inventory', side_effect=recovery_error) as recovery, \
                patch.object(worker, 'process') as process, patch.object(worker, 'backup_if_due') as backup, \
                patch.object(archive, '_run', side_effect=AssertionError('no remote permitted')), \
                redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(worker.run(self.args()), 1)
        process.assert_not_called()
        backup.assert_not_called()
        self.assertEqual(recovery.call_count, int(approved))
        self.assertEqual((self.root / 'state/checkpoint.json').read_bytes(), checkpoint)
        heartbeat = json.loads((self.root / 'state/heartbeat.json').read_text())
        self.assertEqual(heartbeat['status'], 'paused')
        reason = json.loads((self.root / 'state/PAUSED.json').read_text())['reason']
        with sqlite3.connect(self.root / 'state/queue.sqlite') as db:
            self.assertEqual(db.execute('SELECT status,accepted,profile FROM jobs WHERE seed=7').fetchone(),
                             ('done', 7, 'v1'))
            kinds = [row[0] for row in db.execute('SELECT kind FROM events')]
        self.assertNotIn('cycle_budget_yield', kinds)
        return reason

    def test_invalid_source_approval_still_pauses_instead_of_budget_yield(self):
        reason = self.run_failure_case(approved=False)
        self.assertIn('independent review required', reason)

    def test_custody_failure_still_pauses_and_preserves_committed_counts(self):
        reason = self.run_failure_case(approved=True,
                                       recovery_error=archive.ArchiveError('fixture custody mismatch'))
        self.assertEqual(reason, 'fixture custody mismatch')


if __name__ == '__main__':
    unittest.main()
