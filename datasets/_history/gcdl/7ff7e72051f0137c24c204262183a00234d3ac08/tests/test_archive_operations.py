"""Independent storage facts for runtime snapshots and Git payload absence."""
import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import Mock,patch
from pathlib import Path

from goal_conditioned_decision_data.scripts.snapshot_state import snapshot,MAX_BYTES

class RuntimeSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/'state').mkdir()
        (self.root/'manifests').mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_wal_snapshot_is_consistent_and_live_controls_remain(self):
        database=self.root/'state/queue.sqlite'
        connection=sqlite3.connect(database)
        self.addCleanup(connection.close)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE facts(value INTEGER)')
        connection.execute('INSERT INTO facts VALUES (42)');connection.commit()
        pause=self.root/'state/PAUSED.json';pause.write_text('{"reason":"test"}')
        result=snapshot(self.root)
        connection.execute('INSERT INTO facts VALUES (99)');connection.commit()
        with tarfile.open(fileobj=io.BytesIO((self.root/result['payload']).read_bytes())) as archive:
            raw=archive.extractfile('state/queue.sqlite').read()
            self.assertEqual(archive.extractfile('state/PAUSED.json').read(),b'{"reason":"test"}')
        recovered=sqlite3.connect(':memory:')
        try:
            recovered.deserialize(raw)
            self.assertEqual(recovered.execute('SELECT value FROM facts').fetchall(),[(42,)])
        finally:recovered.close()
        self.assertTrue(database.exists());self.assertTrue(pause.exists())
        self.assertEqual(connection.execute('SELECT COUNT(*) FROM facts').fetchone()[0],2)

    def test_same_json_snapshot_has_identical_payload_and_corruption_refuses(self):
        (self.root/'state/checkpoint.json').write_text('{"seed":7}')
        first=snapshot(self.root);second=snapshot(self.root)
        self.assertEqual(first,second)
        (self.root/first['payload']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'immutable conflict'):snapshot(self.root)

    def test_secret_shaped_content_is_not_published(self):
        (self.root/'state/checkpoint.json').write_text(json.dumps({'secret':'ghp_'+'A'*32}))
        with self.assertRaisesRegex(ValueError,'credential-shaped'):snapshot(self.root)
        self.assertFalse((self.root/'data').exists())

    def test_archive_symlink_cannot_write_outside_programme(self):
        (self.root/'state/checkpoint.json').write_text('{"seed":7}')
        (self.root/'data').mkdir()
        outside=self.root/'outside';outside.mkdir()
        (self.root/'data/backups').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'unsafe runtime archive'):snapshot(self.root)
        self.assertEqual(list(outside.iterdir()),[])

    def test_hardlinked_live_control_is_rejected(self):
        import os
        control=self.root/'state/checkpoint.json';control.write_text('{"seed":7}')
        os.link(control,self.root/'alias.json')
        with self.assertRaisesRegex(ValueError,'unsafe runtime control'):snapshot(self.root)
        self.assertTrue(control.exists())

    def test_oversized_json_is_rejected_before_reading(self):
        path=self.root/'state/checkpoint.json'
        with path.open('wb') as output:output.truncate(MAX_BYTES+1)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('oversized file must not be read')):
            with self.assertRaisesRegex(ValueError,'control exceeds'):snapshot(self.root)

    def test_database_page_budget_is_checked_before_temporary_backup(self):
        (self.root/'state/queue.sqlite').touch()
        source=Mock()
        source.execute.side_effect=lambda query:Mock(fetchone=lambda:(4096,) if 'page_size' in query else (100000,))
        with patch('goal_conditioned_decision_data.scripts.snapshot_state.sqlite3.connect',return_value=source),patch('goal_conditioned_decision_data.scripts.snapshot_state.tempfile.TemporaryDirectory',side_effect=AssertionError('must not allocate oversized backup')):
            with self.assertRaisesRegex(ValueError,'database exceeds'):snapshot(self.root)
        source.backup.assert_not_called()

if __name__=='__main__':unittest.main()
