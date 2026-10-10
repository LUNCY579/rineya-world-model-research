#!/usr/bin/env python3
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
FACTORY = LOGS.parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, LOGS / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


a = module('v4_adapter_tests', 'prepare_belief_v4_code_candidate_custody_adapter.py')
r = module('frozen_archival_v2_for_adapter_tests', 'run_provisional_archival_checkpoint_v2.py')
ORIGINAL = FACTORY.parent / 'rineya-world-model-research'


class ExactMapping(unittest.TestCase):
    def setUp(self):
        data = (FACTORY / a.JOINED).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), a.JOINED_SHA)
        self.joined = json.loads(data)

    def test_authoritative_six_rows_map_to_supported_schema(self):
        rows = a.normalized_joined(self.joined, ORIGINAL, r)
        self.assertEqual(len(rows), 6)
        self.assertEqual(r.normalize_baselines(ORIGINAL, {'protected_git_byte_baselines': rows}), rows)

    def test_wrong_parent_or_pending_join_rejected(self):
        for key, value in [('commit_sha', '0' * 40), ('status', 'PUSHED_PENDING')]:
            joined = copy.deepcopy(self.joined)
            joined[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.normalized_joined(joined, ORIGINAL, r)

    def test_missing_duplicate_or_nonlist_rows_rejected(self):
        rows = self.joined['six_protected_git_baselines']
        for bad in (rows[:-1], rows[:5] + [rows[0]], None, {'invented': rows}):
            joined = copy.deepcopy(self.joined)
            joined['six_protected_git_baselines'] = bad
            with self.subTest(bad_type=type(bad).__name__), self.assertRaises(ValueError):
                a.normalized_joined(joined, ORIGINAL, r)

    def test_path_hash_existence_type_or_observation_mismatch_rejected(self):
        for field, value in [('path', '/tmp/unrelated'), ('sha256', '0' * 64),
                             ('expected_exists', 1), ('actual_exists', False),
                             ('actual_sha256', '0' * 64), ('matches_frozen_baseline', False)]:
            joined = copy.deepcopy(self.joined)
            joined['six_protected_git_baselines'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.normalized_joined(joined, ORIGINAL, r)

    def test_absent_worktree_config_rejects_invented_hash(self):
        joined = copy.deepcopy(self.joined)
        next(row for row in joined['six_protected_git_baselines']
             if row['label'] == 'original_worktree_config')['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            a.normalized_joined(joined, ORIGINAL, r)


if __name__ == '__main__':
    unittest.main(verbosity=2)
