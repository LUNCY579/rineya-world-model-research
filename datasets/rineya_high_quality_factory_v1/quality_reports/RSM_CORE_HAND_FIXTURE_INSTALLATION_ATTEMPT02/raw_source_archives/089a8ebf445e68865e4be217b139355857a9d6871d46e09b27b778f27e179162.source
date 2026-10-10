"""Valid individual labels cannot excuse omitted or duplicated export records."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from factory.common import ROOT, load_jsonl, write_jsonl
from factory.export_validation import validate_specialized
from factory.pipeline import validate_release


class ExportCompletenessTests(unittest.TestCase):
    def source(self, track, version='v0001'):
        return ROOT / 'data' / track / 'releases' / version

    def reject(self, source, suffix, transform):
        rows = load_jsonl(source / 'canonical.jsonl')
        file = source / ('train-' + suffix + '.jsonl')
        items = load_jsonl(file)
        self.assertTrue(items, 'Fixture must contain a populated export')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_jsonl(path / file.name, transform(items))
            with self.assertRaises(AssertionError):
                validate_specialized(path, rows)

    def test_empty_one_per_row_exports_rejected(self):
        for track, suffix in [('REM-001','memory_updates'), ('RMM-001','memory_updates'),
                              ('RSM-001','skill_demonstrations'), ('RBM-001','belief_updates'),
                              ('ROM-001','belief_updates')]:
            with self.subTest(track=track):
                self.reject(self.source(track), suffix, lambda items: [])

    def test_duplicate_one_per_row_export_rejected(self):
        self.reject(self.source('REM-001'), 'memory_updates', lambda items: items + [items[0]])

    def test_same_count_missing_input_replaced_with_valid_duplicate_rejected(self):
        self.reject(self.source('REM-001'), 'memory_updates', lambda items: [items[0]] + items[:-1])

    def test_omitted_or_duplicated_selected_trajectory_rejected(self):
        for transform in [lambda items: items[1:], lambda items: items + [items[0]]]:
            self.reject(self.source('RGM-001'), 'trajectories', transform)

    def test_omitted_or_duplicated_process_candidate_rejected(self):
        for transform in [lambda items: items[1:], lambda items: items + [items[0]]]:
            self.reject(self.source('RVM-001','v0003'), 'process_reward', transform)

    def test_empty_strict_preference_bucket_coverage_rejected(self):
        self.reject(self.source('RVM-001','v0003'), 'preference', lambda items: [])

    def test_empty_memory_export_with_updated_checksums_rejected(self):
        source = self.source('REM-001')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'release'
            shutil.copytree(source, path)
            file = path / 'train-memory_updates.jsonl'
            write_jsonl(file, [])
            manifest = json.loads((path / 'MANIFEST.json').read_text())
            manifest['files'][file.name] = {'bytes':file.stat().st_size,
                                          'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
            (path / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (path / 'CHECKSUMS.sha256').write_text(''.join(
                hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n'
                for f in sorted(path.iterdir()) if f.name != 'CHECKSUMS.sha256'))
            with self.assertRaisesRegex(AssertionError, 'one-per-row coverage mismatch'):
                validate_release(path)


if __name__ == '__main__':
    unittest.main()
