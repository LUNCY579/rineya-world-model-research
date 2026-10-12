"""Clock-driven cold recovery/publication/backup limits; no real transport."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import worker, rendered, remote_archive
from goal_conditioned_decision_data.decision_labels import rendering
from goal_conditioned_decision_data.auditors.integrity import semantic_fingerprint
from goal_conditioned_decision_data.generators.profiles import resolve_profile
from goal_conditioned_decision_data.pipelines.common import canonical
from goal_conditioned_decision_data.tests.test_surface_review import manual_record


class Clock:
    def __init__(self):self.now=1000.0
    def monotonic(self):return self.now


class CycleBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()/remote_archive.PROGRAMME;self.root.mkdir()
        for folder in ('state','configs','manifests'):(self.root/folder).mkdir()
        self.config=json.loads((worker.ROOT/'configs/production.json').read_text())
        self.config.update(active_generation_profile='v1',archive_enabled=False,cycle_timeout_seconds=30,
                           groups_per_cycle=1,seed_start=100,production_splits=['train'],rendered_views=[])
        (self.root/'configs/production.json').write_text(canonical(self.config))
        self.code={'fixture.py':'1'*64}
        (self.root/'manifests/VALIDATED_CODE.json').write_text(canonical({'code_hashes':self.code}))
        (self.root/'manifests/SEALED_FREEZE.json').write_text('{}')
        self.clock=Clock()
        for hook in (patch.object(worker,'ROOT',self.root),patch.object(worker.time,'monotonic',side_effect=self.clock.monotonic),
                     patch.object(worker,'source_hashes',return_value=self.code),patch.object(worker,'guard'),
                     patch.object(worker,'_archive_for_resource_pressure',return_value=None),
                     patch.object(worker,'recover_inventory'),patch.object(rendered,'recover'),
                     patch.object(rendered,'enqueue'),patch.object(rendered,'drain',return_value=[]),
                     patch.object(worker,'summary',return_value={'accepted':0,'splits':{}}),
                     patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)),
                     patch('goal_conditioned_decision_data.auditors.release.current_independent_review',return_value=True)):
            mock=hook.start();self.addCleanup(hook.stop)
            name=hook.attribute
            setattr(self,'mock_'+name,mock)
        self.args=argparse.Namespace(no_backup=True,seal=False,groups=1,render_only=False)

    def run_worker(self):
        with redirect_stdout(io.StringIO()):return worker.run(self.args)

    def test_cold_recovery_time_counts_before_generation_and_yields_without_science_pause(self):
        self.mock_recover_inventory.side_effect=lambda db:setattr(self.clock,'now',1025.0)
        with patch.object(worker,'process') as generate:
            self.assertEqual(self.run_worker(),0)
        generate.assert_not_called();self.assertFalse((self.root/'state/PAUSED.json').exists())
        heartbeat=json.loads((self.root/'state/heartbeat.json').read_text())
        self.assertEqual(heartbeat['status'],'yielded')
        self.mock_summary.assert_not_called()

    def test_render_budget_is_anchored_before_cold_recovery(self):
        self.mock_recover_inventory.side_effect=lambda db:setattr(self.clock,'now',1010.0)
        with patch.object(worker,'process'):
            self.assertEqual(self.run_worker(),0)
        self.assertEqual(self.mock_drain.call_args.args[-1],1024.0)

    def test_archive_action_yields_without_opening_database_or_generation(self):
        self.mock__archive_for_resource_pressure.return_value={'status':'archived'}
        with patch.object(worker,'connection',side_effect=AssertionError('archive must yield before DB')):
            self.assertEqual(self.run_worker(),0)
        self.assertEqual(json.loads((self.root/'state/heartbeat.json').read_text())['status'],'archived')

    def test_summary_deadline_expiry_never_publishes_healthy_checkpoint(self):
        def slow_summary(db):
            self.clock.now=1031.0
            return {'accepted':0,'splits':{}}
        self.mock_summary.side_effect=slow_summary
        with patch.object(worker,'process'):
            self.assertEqual(self.run_worker(),0)
        self.assertEqual(json.loads((self.root/'state/heartbeat.json').read_text())['status'],'yielded')
        self.assertFalse((self.root/'state/PAUSED.json').exists())

    def test_backup_parent_and_child_receive_only_remaining_reserved_budget(self):
        self.clock.now=1120.0
        database=sqlite3.connect(':memory:');self.addCleanup(database.close)
        with patch.object(worker.subprocess,'run',return_value=SimpleNamespace(returncode=0,stderr='')) as command:
            worker.backup_if_due(self.config,database,deadline=1130.0)
        self.assertEqual(command.call_args.kwargs['timeout'],10.0)
        self.assertEqual(float(command.call_args.kwargs['env']['GCDL_BACKUP_TIMEOUT_SECONDS']),9.0)

    def test_backup_deadline_expiry_yields_instead_of_healthy_or_science_pause(self):
        self.args.no_backup=False
        def spent_backup(config,db,*,deadline):
            self.assertEqual(deadline,1170.0)
            self.clock.now=1171.0
        with patch.object(worker,'process'),patch.object(worker,'backup_if_due',side_effect=spent_backup):
            self.assertEqual(self.run_worker(),0)
        self.assertEqual(json.loads((self.root/'state/heartbeat.json').read_text())['status'],'yielded')
        self.assertFalse((self.root/'state/PAUSED.json').exists())


class RenderBudgetTests(unittest.TestCase):
    def test_timed_out_second_view_keeps_job_pending_and_replays_first_view_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()/remote_archive.PROGRAMME;root.mkdir();(root/'manifests').mkdir()
            row=manual_record(seed=730001);code={'fixture.py':'1'*64}
            row['verification']['code_hashes']=deepcopy(code)
            relative='data/accepted/train-730001.jsonl';shard=root/relative;shard.parent.mkdir(parents=True)
            raw=(canonical(row)+'\n').encode();shard.write_bytes(raw)
            profile=resolve_profile('v1');provenance=row['provenance']
            batch={'version':'gcdl-manifest-2','seed':730001,'split':'train','shard':relative,
                   'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'accepted':1,'quarantined':0,
                   'audit':{'verdict':'PASS','accepted':1,'quarantined':0},'generation_profile':'v1',
                   'generation_profile_evidence':{key:profile[key] for key in ('id','generator_version','environment_version','config_version','config_hash','config_file_sha256')},
                   'protected_index':[[row['id'],semantic_fingerprint(row),provenance['parent_world_id'],provenance['template_group_id'],'train']],
                   'code_hashes':code}
            (root/'manifests/batch-train-730001.json').write_text(canonical(batch)+'\n')
            db=sqlite3.connect(':memory:');self.addCleanup(db.close)
            config={'rendered_views':['listwise','computation'],'rendered_variant_budget':3,'rendered_groups_per_cycle':1}
            exporter=rendering.export_rendered_dataset;calls=[]
            def interrupt(*args,**kwargs):
                calls.append(args[2])
                if args[2]=='computation':raise remote_archive.ReadDeadlineExceeded('fixture view budget')
                return exporter(*args,**kwargs)
            with patch.object(rendered,'source_hashes',return_value=code),patch.object(rendering,'source_hashes',return_value=code),\
                 patch.object(rendering.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                rendered.enqueue(db,root,config)
                import time
                with patch.object(rendered,'export_rendered_dataset',side_effect=interrupt):
                    self.assertEqual(rendered.drain(db,root,config,time.monotonic()+30),[])
                self.assertEqual(db.execute('SELECT status FROM rendered_jobs').fetchone()[0],'pending')
                self.assertFalse(list((root/'manifests').glob('render-batch-*.json')))
                first={str(p.relative_to(root)):p.read_bytes() for p in (root/'data/rendered').rglob('*') if p.is_file()}
                self.assertTrue(first)
                self.assertEqual(len(rendered.drain(db,root,config,time.monotonic()+30)),1)
                self.assertTrue(all((root/path).read_bytes()==contents for path,contents in first.items()))
                self.assertEqual(shard.read_bytes(),raw)


if __name__=='__main__':unittest.main()
