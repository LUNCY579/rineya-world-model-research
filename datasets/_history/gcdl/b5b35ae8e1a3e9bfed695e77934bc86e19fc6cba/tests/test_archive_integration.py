"""Archive read-through integration; only tiny hand-specified temp fixtures.

No generator call, production mutation, network request, or sealed label read.
The archive transport/custody implementation has separately owned tests.
"""
import argparse
from copy import deepcopy
import fnmatch
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import ANY, patch

from goal_conditioned_decision_data.auditors import release
from goal_conditioned_decision_data.auditors.integrity import semantic_fingerprint
from goal_conditioned_decision_data.decision_labels import rendering
from goal_conditioned_decision_data.generators.profiles import resolve_profile
from goal_conditioned_decision_data.pipelines import remote_archive, rendered, worker
from goal_conditioned_decision_data.pipelines.common import canonical, digest
from goal_conditioned_decision_data.tests.test_surface_review import manual_record


class MemoryArchive:
    """Mock verified payload commitments and a bounded RAM-only transport."""
    def __init__(self, root):
        self.root = root
        self.payloads = {}
        self.reads = []
        self.offline = False

    def archive(self, relative, keep_local=False):
        self.payloads[relative] = (self.root / relative).read_bytes()
        if not keep_local:
            (self.root / relative).unlink()

    def read_bytes(self, root, relative, *, max_bytes=8 * 1024 ** 2,
                   allow_sealed=False):
        self.assert_root(root)
        self.reads.append((relative, max_bytes, allow_sealed))
        if relative.startswith('data/sealed/') and not allow_sealed:
            raise ValueError('sealed raw read refused')
        path = self.root / relative
        if path.is_file():
            raw = path.read_bytes()
        elif relative in self.payloads:
            if self.offline:
                raise OSError('mock archive offline')
            raw = self.payloads[relative]
        else:
            raise FileNotFoundError(relative)
        if len(raw) > max_bytes:
            raise ValueError('bounded payload read exceeded')
        return raw

    def read_text(self, *args, **kwargs):
        return self.read_bytes(*args, **kwargs).decode('utf-8')

    def payload_info(self, root, relative):
        self.assert_root(root)
        path = self.root / relative
        if path.is_file():
            raw, archived = path.read_bytes(), False
        elif relative in self.payloads:
            raw, archived = self.payloads[relative], True
        else:
            raise FileNotFoundError(relative)
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                'archived': archived, 'commit': 'a' * 40}

    def exists(self, root, relative):
        self.assert_root(root)
        return (self.root / relative).is_file() or relative in self.payloads

    def is_archived(self, root, relative):
        self.assert_root(root)
        return relative in self.payloads

    def iter_paths(self, root, pattern):
        self.assert_root(root)
        paths = {str(p.relative_to(self.root)) for p in self.root.glob(pattern)
                 if p.is_file()}
        paths.update(p for p in self.payloads if fnmatch.fnmatchcase(p, pattern))
        return sorted(paths)

    def assert_root(self, root):
        if Path(root).resolve() != self.root:
            raise AssertionError('archive reads escaped fixture root')


class ArchiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / 'goal_conditioned_decision_data'
        self.root.mkdir()
        (self.root / 'manifests').mkdir()
        (self.root / 'configs').mkdir()
        self.config = json.loads((worker.ROOT / 'configs/production.json').read_text())
        self.config.update(active_generation_profile='v1', archive_enabled=False,
                           seed_start=100, production_splits=['train'], groups_per_cycle=1,
                           rendered_views=['listwise'], rendered_variant_budget=3)
        (self.root / 'configs/production.json').write_text(canonical(self.config))
        self.row = manual_record(seed=730001)
        self.code = {'manual_archive_fixture': '1' * 64}
        self.row['verification']['code_hashes'] = deepcopy(self.code)
        self.raw = (canonical(self.row) + '\n').encode()
        self.shard = 'data/accepted/train-730001.jsonl'
        (self.root / self.shard).parent.mkdir(parents=True)
        (self.root / self.shard).write_bytes(self.raw)
        profile = resolve_profile('v1')
        provenance = self.row['provenance']
        self.batch = {'version': 'gcdl-manifest-2', 'seed': 730001, 'split': 'train',
                      'shard': self.shard, 'sha256': hashlib.sha256(self.raw).hexdigest(),
                      'bytes': len(self.raw), 'accepted': 1, 'quarantined': 0,
                      'audit': {'verdict': 'PASS', 'accepted': 1, 'quarantined': 0,
                                'family_counts': {'A': 1}},
                      'generation_profile': 'v1',
                      'generation_profile_evidence': {key: profile[key] for key in
                          ('id', 'generator_version', 'environment_version',
                           'config_version', 'config_hash', 'config_file_sha256')},
                      'protected_index': [[self.row['id'], semantic_fingerprint(self.row),
                          provenance['parent_world_id'], provenance['template_group_id'], 'train']],
                      'code_hashes': deepcopy(self.code), 'reproducibility': 'PASS'}
        self.manifest = 'manifests/batch-train-730001.json'
        (self.root / self.manifest).write_text(canonical(self.batch) + '\n')
        self.archive = MemoryArchive(self.root)
        for name in ('read_bytes', 'read_text', 'payload_info', 'exists', 'iter_paths','is_archived'):
            hook = patch.object(remote_archive, name, side_effect=getattr(self.archive, name))
            hook.start()
            self.addCleanup(hook.stop)
        root_hook = patch.object(worker, 'ROOT', self.root)
        root_hook.start()
        self.addCleanup(root_hook.stop)
        disk_hook = patch.object(worker.shutil, 'disk_usage',
                                 return_value=type('Disk', (), {'free': 1 << 40})())
        disk_hook.start()
        self.addCleanup(disk_hook.stop)
        self.db = worker.connection()
        self.addCleanup(self.db.close)

    def publish(self):
        with patch.object(rendered, 'source_hashes', return_value=self.code), \
             patch.object(rendering, 'source_hashes', return_value=self.code):
            rendered.enqueue(self.db, self.root, self.config)
            return rendered.drain(self.db, self.root, self.config, time.monotonic() + 30)

    def test_archived_accepted_recovery_preserves_counts_profile_lineage_and_max_seed(self):
        self.archive.archive(self.shard)
        worker.recover_inventory(self.db)
        worker.recover_inventory(self.db)
        self.assertEqual(self.db.execute('SELECT seed,split,status,accepted,profile FROM jobs').fetchone(),
                         (730001, 'train', 'done', 1, 'v1'))
        self.assertEqual(tuple(self.batch['protected_index'][0]),
                         self.db.execute('SELECT * FROM seen').fetchone())
        self.assertFalse((self.root / self.shard).exists())
        self.assertFalse(any(p == self.shard for p, _, _ in self.archive.reads))
        with patch.object(worker, 'process', side_effect=RuntimeError('fixture stop')) as process:
            self.assertEqual(worker.run(argparse.Namespace(no_backup=True, seal=False,
                                                          groups=1, render_only=False)), 1)
        self.assertEqual(process.call_args.args[:2], (730002, 'train'))

    def test_local_plus_archive_inventory_counts_once(self):
        self.archive.archive(self.shard, keep_local=True)
        self.archive.archive(self.manifest, keep_local=True)
        worker.recover_inventory(self.db)
        self.assertEqual(self.db.execute('SELECT SUM(accepted) FROM jobs').fetchone()[0], 1)
        self.assertEqual(rendering._source_paths(self.root, 'train', None),
                         [self.root / self.shard])

    def test_missing_unarchived_canonical_shard_refuses_recovery_and_export(self):
        (self.root / self.shard).unlink()
        with self.assertRaises(FileNotFoundError):
            worker.recover_inventory(self.db)
        with self.assertRaises(FileNotFoundError):
            rendering._batch_source(self.root, self.root / self.shard, 'train')

    def test_archived_shard_wrong_hash_refuses_recovery(self):
        self.archive.archive(self.shard)
        self.archive.payloads[self.shard] += b'corrupt'
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            worker.recover_inventory(self.db)

    def test_archived_canonical_payload_cannot_reset_inventory_after_control_loss(self):
        self.archive.archive(self.shard)
        (self.root / self.manifest).unlink()
        with self.assertRaisesRegex(ValueError, 'missing its immutable batch'):
            worker.recover_inventory(self.db)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)

    def test_existing_archived_batch_refuses_replay_before_generator_call(self):
        self.archive.archive(self.manifest)
        with patch('goal_conditioned_decision_data.generators.build.generate_group') as generate:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                worker.process(730001, 'train', self.config, self.db)
            generate.assert_not_called()

    def sealed_metadata(self):
        relative = 'data/sealed/sealed-880000.jsonl'
        raw = b'OPAQUE SEAL SENTINEL: MUST NEVER BE PARSED OR READ\n'
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        batch = {'seed': 880000, 'split': 'sealed', 'shard': relative, 'sha256': sha,
                 'bytes': len(raw), 'accepted': 1, 'quarantined': 0, 'audit': {},
                 'reproducibility': 'PASS',
                 'protected_index': [['sealed-id', 'sealed-semantic', 'sealed-parent',
                                      'cost_cap_and_five_block', 'sealed']]}
        (self.root / 'manifests/batch-sealed-880000.json').write_text(canonical(batch))
        (self.root / 'manifests/SEALED_FREEZE.json').write_text(canonical({'shards': {relative: sha}}))
        self.archive.archive(relative)
        return relative

    def test_archived_sealed_recovery_is_metadata_only_even_offline(self):
        relative = self.sealed_metadata()
        self.archive.offline = True
        worker.recover_inventory(self.db)
        self.assertEqual(self.db.execute('SELECT accepted FROM jobs WHERE split="sealed"').fetchone()[0], 1)
        self.assertFalse(any(p == relative for p, _, _ in self.archive.reads))

    def test_archived_sealed_freeze_mismatch_refuses_without_label_read(self):
        relative = self.sealed_metadata()
        (self.root / 'manifests/SEALED_FREEZE.json').write_text(canonical({'shards': {relative: '0' * 64}}))
        with self.assertRaisesRegex(ValueError, 'local freeze'):
            worker.recover_inventory(self.db)
        self.assertFalse(any(p == relative for p, _, _ in self.archive.reads))

    def test_nonsealed_release_audit_reads_archived_rows_and_keeps_seal_metadata_only(self):
        sealed = self.sealed_metadata()
        config = dict(self.config, sealed_seeds=[880000])
        (self.root / 'configs/production.json').write_text(canonical(config))
        self.archive.archive(self.shard)
        with patch.object(release, 'ROOT', self.root), \
             patch.object(release, 'current_independent_review', return_value=True):
            result = release.audit(False)
        self.assertEqual(result['accepted_verified'], 1)
        self.assertEqual(result['counts_by_split'], {'train': 1, 'sealed_hash_verified_records': 1})
        self.assertEqual(result['errors'], [])
        self.assertFalse(any(p == sealed for p, _, _ in self.archive.reads))
        self.assertFalse((self.root / self.shard).exists())

    def test_archived_source_exports_identical_views_without_restoring_source(self):
        before = self.publish()
        receipt = rendered.validate_receipt(self.root, before[0])[0]
        manifest_path = receipt['datasets'][0]['manifest_path']
        manifest_before = (self.root / manifest_path).read_bytes()
        self.archive.archive(self.shard)
        with patch.object(rendering, 'source_hashes', return_value=self.code):
            result = rendering.export_rendered_dataset(self.root, 'train', 'listwise', 1, 3)
        self.assertEqual((canonical(result) + '\n').encode(), manifest_before)
        self.assertFalse((self.root / self.shard).exists())
        self.assertTrue(any(p == self.shard for p, _, _ in self.archive.reads))
        self.assertTrue(all(bound <= 8 * 1024 ** 2 and not seal
                            for _, bound, seal in self.archive.reads))

    def test_archived_source_export_still_checks_physical_free_space_first(self):
        self.archive.archive(self.shard)
        with patch.object(worker.shutil, 'disk_usage',
                          return_value=type('Disk', (), {'free': 1})()):
            with self.assertRaisesRegex(RuntimeError, '10 GiB'):
                rendering.export_rendered_dataset(self.root, 'train', 'listwise', 1, 3)
        self.assertFalse(any(p == self.shard for p, _, _ in self.archive.reads))

    def test_record_limit_does_not_fetch_later_archived_shard(self):
        self.archive.archive(self.shard)
        later='data/accepted/train-730002.jsonl'
        self.archive.payloads[later]=b'UNSELECTED PAYLOAD MUST NOT BE FETCHED'
        with patch.object(rendering, 'source_hashes', return_value=self.code):
            result=rendering.export_rendered_dataset(self.root,'train','listwise',1,3)
        self.assertEqual(result['source_records_examined'],1)
        self.assertFalse(any(p==later for p,_,_ in self.archive.reads))

    def test_archived_unaccepted_or_split_tamper_never_exports(self):
        for field, value in [('quality_gate', {'status': 'quarantined'}), ('split', 'dev')]:
            row = deepcopy(self.row)
            row[field] = value
            self.archive.payloads[self.shard] = (canonical(row) + '\n').encode()
            (self.root / self.shard).unlink(missing_ok=True)
            with self.assertRaisesRegex(ValueError, 'unaccepted|mismatched-split'):
                rendering.export_rendered_dataset(self.root, 'train', 'listwise', 1, 3)
        self.assertFalse((self.root / 'data/rendered').exists())

    def test_archived_derived_payloads_preserve_completed_queue_and_counts(self):
        receipts = self.publish()
        receipt, counts = rendered.validate_receipt(self.root, receipts[0])
        self.assertEqual(counts, {'listwise': 3})
        manifest_relative = receipt['datasets'][0]['manifest_path']
        manifest = json.loads((self.root / manifest_relative).read_text())
        for relative in [self.shard, manifest_relative] + [p['path'] for p in manifest['shards']]:
            self.archive.archive(relative)
        self.db.execute('DELETE FROM rendered_jobs')
        self.db.commit()
        rendered.recover(self.db, self.root)
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'done')
        inventory = rendered.inventory(self.root)
        self.assertEqual(inventory['completed_source_group_exports'], 1)
        self.assertEqual(inventory['rendered_rows_by_split_view'], {'train/listwise': 3})
        self.assertEqual(inventory['eligible_source_records_by_split_view'], {'train/listwise': 1})
        self.assertFalse((self.root / manifest_relative).exists())

    def test_missing_unarchived_derived_payload_refuses_completed_inventory(self):
        receipts = self.publish()
        receipt = json.loads((self.root / receipts[0]).read_text())
        manifest = json.loads((self.root / receipt['datasets'][0]['manifest_path']).read_text())
        (self.root / manifest['shards'][0]['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            rendered.inventory(self.root)

    def review_fixture(self):
        config = dict(self.config, active_generation_profile='v2')
        (self.root / 'configs/production.json').write_text(canonical(config))
        evidence = {}
        for relative, raw in [('manifests/V2_EXPERIMENT_FREEZE.json', b'{}\n'),
                              ('reports/V2_CONTROLLED_EXPERIMENT.json', b'{"PASS":true}\n')]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            evidence[relative] = hashlib.sha256(raw).hexdigest()
        review = ('FINAL_VERDICT: PASS\n' + digest(self.code) + '\n' + canonical(evidence)).encode()
        (self.root / 'reports/INDEPENDENT_REVIEW.md').write_bytes(review)
        receipt = {'required_evidence': evidence, 'code_hashes': self.code,
                   'source_digest': digest(self.code),
                   'independent_review_sha256': hashlib.sha256(review).hexdigest()}
        (self.root / 'manifests/VALIDATED_CODE.json').write_text(canonical(receipt))
        for relative in list(evidence) + ['reports/INDEPENDENT_REVIEW.md']:
            self.archive.archive(relative)
        return receipt

    def test_archived_review_and_mandatory_v2_evidence_are_read_and_hash_bound(self):
        self.review_fixture()
        with patch.object(release, 'ROOT', self.root), \
             patch.object(release, 'source_hashes', return_value=self.code):
            self.assertTrue(release.current_independent_review())
            self.archive.offline = True
            self.assertFalse(release.current_independent_review())
        self.assertFalse(any(seal for _, _, seal in self.archive.reads))

    def test_archived_evidence_cannot_be_removed_or_rehashed_to_broaden_review(self):
        receipt = self.review_fixture()
        path = self.root / 'manifests/VALIDATED_CODE.json'
        evidence_path = 'reports/V2_CONTROLLED_EXPERIMENT.json'
        self.archive.payloads[evidence_path] += b'changed'
        with patch.object(release, 'ROOT', self.root), \
             patch.object(release, 'source_hashes', return_value=self.code):
            self.assertFalse(release.current_independent_review())
            receipt['required_evidence'][evidence_path] = hashlib.sha256(self.archive.payloads[evidence_path]).hexdigest()
            path.write_text(canonical(receipt))
            self.assertFalse(release.current_independent_review())
            del receipt['required_evidence'][evidence_path]
            path.write_text(canonical(receipt))
            self.assertFalse(release.current_independent_review())

    def test_active_low_disk_cannot_evict_without_review_or_generate_below_floor(self):
        self.config['archive_enabled'] = True
        (self.root / 'configs/production.json').write_text(canonical(self.config))
        from goal_conditioned_decision_data.scripts import archive_cycle
        cycle_root = self.root / remote_archive.PROGRAMME
        (cycle_root / 'state').mkdir(parents=True)
        policy = dict(self.config, archive_authorization=remote_archive.AUTHORIZATION,
                      archive_trigger_free_bytes=10 * 1024 ** 3,
                      min_free_bytes=10 * 1024 ** 3)
        with patch.object(worker.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()), \
             patch.object(worker, 'ROOT', cycle_root), \
             patch.object(archive_cycle, 'current_independent_review', return_value=False) as review, \
             patch.object(archive_cycle, '_command', side_effect=AssertionError('unreviewed cycle reached Git')) as command, \
             patch.object(worker, '_run_archive_cycle', wraps=worker._run_archive_cycle) as archive:
            with self.assertRaisesRegex(RuntimeError, 'review required'):
                worker._archive_for_resource_pressure(policy)
            archive.assert_called_once_with(policy, None)
            review.assert_called_once_with(deadline=ANY)
            command.assert_not_called()
            self.assertEqual((self.root / self.shard).read_bytes(), self.raw)
        with patch.object(worker.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()), \
             patch.object(release, 'current_independent_review', return_value=True), \
             patch.object(worker, '_run_archive_cycle', return_value={'verified': True}) as archive, \
             patch.object(worker, 'process') as process:
            self.assertEqual(worker.run(argparse.Namespace(no_backup=True, seal=False, groups=1)), 1)
            archive.assert_called_once_with(self.config,ANY)
            process.assert_not_called()

    def test_disk_pause_archives_under_review_and_preserves_cause_without_generation(self):
        self.config['archive_enabled'] = True
        (self.root / 'configs/production.json').write_text(canonical(self.config))
        marker = self.root / 'state/PAUSED.json'
        raw = b'{"reason":"resource gate: fewer than 10 GiB free","original":17}\n'
        marker.write_bytes(raw)
        with patch.object(worker.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()), \
             patch.object(release, 'current_independent_review', return_value=True), \
             patch.object(worker, '_run_archive_cycle', return_value=None) as archive, \
             patch.object(worker, 'connection') as connect, patch.object(worker, 'process') as process:
            self.assertEqual(worker.run(argparse.Namespace(no_backup=True, seal=False, groups=1)), 0)
            archive.assert_called_once()
            connect.assert_not_called()
            process.assert_not_called()
        self.assertEqual(marker.read_bytes(), raw)

    def test_stop_and_migration_pause_never_archive_or_rewrite_state(self):
        self.config['archive_enabled'] = True
        (self.root / 'configs/production.json').write_text(canonical(self.config))
        for filename, content in [('PAUSED.json', '{"reason":"archive migration requires fresh review"}'),
                                  ('STOP', 'operator stop')]:
            path = self.root / 'state' / filename
            path.write_text(content)
            before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
            with patch.object(worker, '_run_archive_cycle') as archive, \
                 patch.object(worker, 'connection') as connect:
                self.assertEqual(worker.run(argparse.Namespace(no_backup=True, seal=False, groups=1)), 0)
                archive.assert_not_called()
                connect.assert_not_called()
            self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_stop_does_not_require_parseable_config_or_archive_inventory(self):
        (self.root/'state/STOP').write_text('operator stop')
        (self.root/'configs/production.json').write_text('MALFORMED CONFIG NEVER READ')
        with patch.object(worker,'_run_archive_cycle') as archive:
            self.assertEqual(worker.run(argparse.Namespace(no_backup=True,seal=False,groups=1)),0)
            archive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
