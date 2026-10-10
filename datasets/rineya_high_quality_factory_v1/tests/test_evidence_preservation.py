import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.pipeline import preserve_prior_audit, preserve_source_bytes


class EvidencePreservation(unittest.TestCase):
    def test_changed_machine_report_keeps_original_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'batch.jsonl'
            original=b'{"pass":true}\n'
            path.write_bytes(original)
            preserve_prior_audit(path,b'{"pass":false}\n')
            saved=path.parent/'history'/f'batch-{hashlib.sha256(original).hexdigest()}.jsonl'
            self.assertEqual(saved.read_bytes(),original)
            self.assertEqual(path.read_bytes(),original)
            saved.write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):preserve_prior_audit(path,b'new')

    def test_source_snapshot_survives_later_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'generator.py'
            original=b'answer = 1\n';source.write_bytes(original)
            with patch('factory.pipeline.ROOT',root):hashes=preserve_source_bytes([source])
            source.write_bytes(b'answer = 2\n')
            blob=root/'reproducibility/source_blobs'/f'{hashes["generator.py"]}.source'
            self.assertEqual(blob.read_bytes(),original)
            source.write_bytes(original);blob.write_bytes(b'corrupted')
            with patch('factory.pipeline.ROOT',root),self.assertRaises(AssertionError):preserve_source_bytes([source])


if __name__=='__main__':unittest.main()
