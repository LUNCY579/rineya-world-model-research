"""Additive checks; all attempt01 sources/results remain immutable."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_frozen_hand_checks', HERE / 'hand_tests_attempt01.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
COUNTS = {'additional_phase_cut_budget_compositions': 0, 'additional_phase_cuts': 0,
          'register_equivariance_pairs': 0, 'restart_reduction_pairs': 0}


class AdditiveBoundaries(unittest.TestCase):
    def test_all_entry_exit_restart_phase_cuts(self):
        templates = []
        for family in old.fixtures.hand_programs():
            c, _, _ = old.fixtures.contexts(family['task'], old.solve)
            templates.extend([c['A'], c['I'], c['S']])
            restart = old.solve(c['T'])
            new_root = deepcopy(c['F']); new_root['initial_state'] = deepcopy(c['T']['checkpoint']['state'])
            old.equal(old.solve(new_root)['new_events'], restart['new_events'])
            old.equal(old.solve(new_root)['checkpoint'], restart['checkpoint'])
            COUNTS['restart_reduction_pairs'] += 1
            templates.append(new_root)
        failure = old.fixtures.hand_programs()[0]['task']
        failure['options']['child']['success'] = old.fixtures.pred(old.fixtures.atom(old.fixtures.R('x'), 'lt', old.fixtures.C(0)))
        templates.append(failure)
        failure = deepcopy(failure)
        failure['options']['child']['initiation'] = old.fixtures.pred(old.fixtures.atom(old.fixtures.R('x'), 'lt', old.fixtures.C(0)))
        templates.append(failure)
        for base in templates:
            complete = old.solve(base); events = complete['new_events']; n = len(events)
            recursive = old.reference.full_trace(base)
            old.equal(events, recursive['events']); self.assertEqual(complete['status'], recursive['phase'])
            for cut in range(n + 1):
                prefix = deepcopy(base); prefix['event_budget'] = cut; at = old.solve(prefix)
                resumed = deepcopy(base); resumed.update(mode='resume', history=events[:cut], checkpoint=at['checkpoint'])
                for budget in range(n - cut + 2):
                    resumed['event_budget'] = budget; out = old.solve(resumed)
                    old.equal(out['new_events'], events[cut:cut + budget])
                    self.assertEqual(out['status'], complete['status'] if cut + budget >= n else 'paused')
                    if cut + budget < n: old.equal(out['next_event'], events[cut + budget])
                    else: self.assertIsNone(out['next_event'])
                    prefix['event_budget'] = min(n, cut + budget)
                    old.equal(out['checkpoint'], old.solve(prefix)['checkpoint'])
                    COUNTS['additional_phase_cut_budget_compositions'] += 1
                COUNTS['additional_phase_cuts'] += 1

    def test_register_equivariance_all_families(self):
        rename = {'x': 'y', 'y': 'x'}
        def exchange(value):
            if type(value) is dict: return {rename.get(k, k): exchange(v) for k, v in value.items()}
            if type(value) is list: return [exchange(v) for v in value]
            if type(value) is str: return rename.get(value, value)
            return value
        for family in old.fixtures.hand_programs():
            task = family['task']; changed = exchange(task)
            old.equal(exchange(old.solve(task)), old.solve(changed))
            self.assertEqual(old.fixtures.normalized_signature(task), old.fixtures.normalized_signature(changed))
            old.equal(old.reference.full_trace(changed)['events'], old.solve(changed)['new_events'])
            COUNTS['register_equivariance_pairs'] += 1


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdditiveBoundaries))
    report = {'actual_role': '/root/formal_oracle', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'counts': COUNTS,
              'first_attempt_reexecuted': False, 'new_production_data_read': False, 'approval': None}
    (HERE / 'hand-results-attempt02.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
