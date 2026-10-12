"""Persistence/resource safeguards independent of generated labels."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from goal_conditioned_decision_data.pipelines import worker, common


class OperationsTests(unittest.TestCase):
    def test_three_all_quarantine_batches_checkpoint_then_pause(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir();(root/'state').mkdir();(root/'manifests').mkdir()
            config=json.loads((worker.ROOT/'configs/production.json').read_text())
            config.update(groups_per_cycle=3,seed_start=100,production_splits=['train'])
            (root/'configs/production.json').write_text(json.dumps(config))
            with patch.object(worker,'ROOT',root),patch.object(worker.shutil,'disk_usage',return_value=type('Disk',(),{'free':1<<40})()),patch('goal_conditioned_decision_data.generators.build.generate_group',return_value=[{'invalid_fixture':True}]):
                self.assertEqual(worker.run(argparse.Namespace(no_backup=True,seal=False,groups=3)),1)
                db=worker.connection()
                self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs WHERE status="done"').fetchone()[0],3)
                db.close()
            self.assertIn('unproductive',(root/'state/PAUSED.json').read_text())
            self.assertEqual(json.loads((root/'state/checkpoint.json').read_text())['seed'],102)

    def test_mid_batch_source_drift_stops_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'state').mkdir()
            config=json.loads((worker.ROOT/'configs/production.json').read_text())
            with patch.object(worker,'ROOT',root),patch.object(worker.shutil,'disk_usage',return_value=type('Disk',(),{'free':1<<40})()),patch.object(worker,'source_hashes',side_effect=[{'code':'before'},{'code':'after'}]),patch('goal_conditioned_decision_data.generators.build.generate_group',return_value=[]),patch('goal_conditioned_decision_data.auditors.integrity.audit_group',return_value=([],[],{})):
                db=worker.connection()
                with self.assertRaisesRegex(RuntimeError,'changed during batch'):worker.process(101,'train',config,db)
                db.close()
            self.assertFalse((root/'data/accepted').exists())

    def test_immutable_publish_conflict_and_same_content_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'accepted.jsonl'
            common.immutable_bytes(p,b'one\n');common.immutable_bytes(p,b'one\n')
            with self.assertRaises(ValueError):common.immutable_bytes(p,b'two\n')
            self.assertEqual(p.read_bytes(),b'one\n');self.assertFalse(p.with_name(p.name+'.tmp').exists())

    def test_paused_interval_does_not_grow_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir();(root/'state').mkdir()
            (root/'configs/production.json').write_text('{}')
            marker=root/'state/PAUSED.json';marker.write_text('{"reason":"original"}')
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(worker,'ROOT',root):
                for _ in range(3):self.assertEqual(worker.run(argparse.Namespace(no_backup=True,seal=False,groups=1)),0)
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_publication_guard_reserves_space(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'existing').write_bytes(b'12345')
            with patch.object(worker,'ROOT',root),patch.object(worker.shutil,'disk_usage',return_value=type('Disk',(),{'free':100})()):
                with self.assertRaisesRegex(RuntimeError,'storage'):worker.publication_guard({'max_programme_bytes':7,'min_free_bytes':10},3)
                with self.assertRaisesRegex(RuntimeError,'free'):worker.publication_guard({'max_programme_bytes':100,'min_free_bytes':99},2)

    def test_manifest_recovery_retains_complete_index_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifests').mkdir();(root/'data/sealed').mkdir(parents=True)
            shard=root/'data/sealed/one.jsonl';shard.write_bytes(b'sealed bytes never parsed by producer\n')
            index=['id1','semantic1','parent1','template1','sealed']
            m={'seed':880000,'split':'sealed','shard':'data/sealed/one.jsonl','sha256':common.file_hash(shard),'accepted':1,'quarantined':0,'audit':{},'protected_index':[index]}
            (root/'manifests/batch-sealed-880000.json').write_text(json.dumps(m))
            with patch.object(worker,'ROOT',root):
                db=worker.connection();worker.recover_inventory(db);worker.recover_inventory(db)
                self.assertEqual(tuple(index),db.execute('SELECT * FROM seen').fetchone())
                db.execute('UPDATE seen SET split="train"');db.commit()
                with self.assertRaisesRegex(ValueError,'conflict'):worker.recover_inventory(db)
                db.close()

if __name__=='__main__':unittest.main()
