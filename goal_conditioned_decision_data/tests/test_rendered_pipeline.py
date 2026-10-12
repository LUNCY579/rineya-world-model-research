"""End-to-end queue recovery, derived-view custody, and publication failures."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import worker, rendered
from goal_conditioned_decision_data.decision_labels import rendering
from goal_conditioned_decision_data.pipelines.common import canonical, file_hash


class RenderedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config = json.loads((worker.ROOT / 'configs/production.json').read_text())
        self.config.update(rendered_views=['listwise', 'computation', 'distillation'],
                           rendered_variant_budget=3, rendered_groups_per_cycle=12)
        self.patch_root = patch.object(worker, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.patch_disk = patch.object(worker.shutil, 'disk_usage',
                                      return_value=type('Disk', (), {'free': 100 * 1024 ** 3})())
        self.patch_disk.start()
        self.addCleanup(self.patch_disk.stop)
        self.db = worker.connection()
        self.addCleanup(self.db.close)
        self.db.execute('INSERT INTO jobs (seed,split,status) VALUES (1000000,"train","pending")')
        self.db.commit()
        self.batch = worker.process(1000000, 'train', self.config, self.db)

    def publish(self):
        rendered.enqueue(self.db, self.root, self.config)
        return rendered.drain(self.db, self.root, self.config, time.monotonic() + 60)

    def test_real_batch_publishes_all_views_and_replay_keeps_gold_bytes(self):
        shard = self.root / self.batch['shard']
        before = file_hash(shard)
        receipts = self.publish()
        self.assertEqual(len(receipts), 1)
        receipt, counts = rendered.validate_receipt(self.root, receipts[0])
        self.assertEqual(counts, {'listwise': 90, 'computation': 15, 'distillation': 18})
        self.assertEqual(file_hash(shard), before)
        self.assertEqual(len(receipt['datasets']), 3)
        rendered.recover(self.db, self.root)
        self.assertEqual(self.publish(), [])
        stats = rendered.inventory(self.root)
        self.assertEqual(stats['completed_source_group_exports'], 1)
        self.assertEqual(stats['rendered_rows_by_split_view']['train/listwise'], 90)

    def test_completed_receipt_recovers_database_loss_and_missing_receipt_fails(self):
        receipts = self.publish()
        self.db.execute('DELETE FROM rendered_jobs')
        self.db.commit()
        rendered.recover(self.db, self.root)
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'done')
        (self.root / receipts[0]).unlink()
        with self.assertRaises(OSError):
            rendered.recover(self.db, self.root)

    def test_interrupted_views_remain_pending_and_resume_same_artifacts(self):
        rendered.enqueue(self.db, self.root, self.config)
        real_export = rendering.export_rendered_dataset
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args[2])
            if len(calls) == 2:
                raise RuntimeError('injected interruption after first view')
            return real_export(*args, **kwargs)

        with patch.object(rendered, 'export_rendered_dataset', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                rendered.drain(self.db, self.root, self.config, time.monotonic() + 60)
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'pending')
        self.assertEqual(list((self.root / 'manifests').glob('render-batch-*.json')), [])
        existing = {str(path): file_hash(path) for path in (self.root / 'data/rendered').rglob('*') if path.is_file()}
        self.assertEqual(len(self.publish()), 1)
        self.assertTrue(all(file_hash(Path(path)) == sha for path, sha in existing.items()))

    def test_corrupted_derived_shard_stops_recovery(self):
        receipts = self.publish()
        receipt = json.loads((self.root / receipts[0]).read_text())
        dataset = json.loads((self.root / receipt['datasets'][0]['manifest_path']).read_text())
        shard = self.root / dataset['shards'][0]['path']
        shard.write_bytes(shard.read_bytes() + b'corruption\n')
        with self.assertRaisesRegex(ValueError, 'digest/size/path'):
            rendered.recover(self.db, self.root)

    def test_expired_deadline_and_group_budget_are_respected(self):
        rendered.enqueue(self.db, self.root, self.config)
        self.assertEqual(rendered.drain(self.db, self.root, self.config, time.monotonic() - 1), [])
        self.assertFalse((self.root / 'data/rendered').exists())
        invalid = dict(self.config, rendered_groups_per_cycle=17)
        with self.assertRaisesRegex(ValueError, 'group budget'):
            rendered.settings(invalid)

    def test_source_drift_keeps_rendered_receipt_unpublished(self):
        rendered.enqueue(self.db, self.root, self.config)
        code = rendered.source_hashes()
        with patch.object(rendered, 'source_hashes', side_effect=[code, {'drift': 'changed'}]):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                rendered.drain(self.db, self.root, self.config, time.monotonic() + 60)
        self.assertFalse(list((self.root / 'manifests').glob('render-batch-*.json')))

    def test_sealed_labels_are_never_opened_for_render_queue(self):
        sealed = self.root / 'data/sealed/sealed-880000.jsonl'
        sealed.parent.mkdir(parents=True)
        sealed.write_text('INVALID SEALED GOLD THAT MUST NEVER BE PARSED')
        (self.root / 'manifests/batch-sealed-880000.json').write_text(canonical({
            'seed':880000, 'split':'sealed', 'accepted':27, 'shard':'data/sealed/sealed-880000.jsonl'}))
        self.assertEqual(rendered.enqueue(self.db, self.root, self.config), 1)
        self.assertEqual(len(self.publish()), 1)

    def test_partial_quarantine_batch_exports_only_accepted_shard(self):
        path = self.root / 'manifests/batch-train-1000000.json'
        batch = json.loads(path.read_text())
        batch['quarantined'] = 1
        batch['audit']['verdict'] = 'PARTIAL_QUARANTINE'
        batch['audit']['quarantined'] = 1
        path.write_text(canonical(batch) + '\n')
        quarantine = self.root / 'data/quarantined/train-1000000.jsonl'
        quarantine.parent.mkdir(parents=True)
        quarantine.write_text('POISON QUARANTINED ROW NEVER READ')
        receipts = self.publish()
        self.assertEqual(len(receipts), 1)
        self.assertEqual(rendered.validate_receipt(self.root, receipts[0])[1]['listwise'], 90)

    def test_partial_ordinary_only_subset_commits_zero_auxiliary_views(self):
        path = self.root / 'manifests/batch-train-1000000.json'
        batch = json.loads(path.read_text())
        shard = self.root / batch['shard']
        first = shard.read_bytes().splitlines(keepends=True)[0]
        shard.write_bytes(first)
        batch.update(accepted=1, quarantined=29, bytes=len(first), sha256=file_hash(shard),
                     protected_index=batch['protected_index'][:1])
        batch['audit'].update(verdict='PARTIAL_QUARANTINE', accepted=1, quarantined=29)
        path.write_text(canonical(batch) + '\n')
        receipts = self.publish()
        self.assertEqual(len(receipts), 1)
        self.assertEqual(rendered.validate_receipt(self.root, receipts[0])[1],
                         {'listwise':3, 'computation':0, 'distillation':0})


if __name__ == '__main__':
    unittest.main()
