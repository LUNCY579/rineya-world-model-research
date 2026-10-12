import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('recovery_retention', Path(__file__).parents[1] / 'scripts/evaluate_recovery_retention.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryRetentionTests(unittest.TestCase):
    def test_duplicate_keys_and_wrong_numeric_types_fail(self):
        fixture = {'id': 'x', 'domain': 'register', 'target': '{"x":1,"legal":true,"terminated":false}'}
        for text in ('{"x":1,"x":1,"legal":true,"terminated":false}',
                     '{"x":true,"legal":true,"terminated":false}',
                     '{"x":1.0,"legal":true,"terminated":false}',
                     '{"x":1,"legal":true,"terminated":false} explanation'):
            self.assertFalse(module.score_reply(fixture, {'text': text})['correct'])

    def test_flags_and_complete_ledger_state_are_scored(self):
        register = {'id': 'x', 'domain': 'register', 'target': '{"x":1,"legal":true,"terminated":false}'}
        self.assertFalse(module.score_reply(register, {'text': '{"x":1,"legal":false,"terminated":false}'})['correct'])
        ledger = {'id': 'y', 'domain': 'ledger', 'target': '{"A":1,"B":2,"C":3,"D":4}'}
        self.assertTrue(module.score_reply(ledger, {'text': '{"D":4,"C":3,"B":2,"A":1}'})['correct'])
        self.assertFalse(module.score_reply(ledger, {'text': '{"A":1,"B":2,"C":3}'})['schema_valid'])


if __name__ == '__main__':
    unittest.main()
