"""Official task semantics only; all source frozen before importing this file."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / 'reviews/oracle/RSM-option-resumption-prospective-20261010T083826Z'
def load(name, file):
    s = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m
solver = load('official_option_resume', HERE / 'option_resume_solver_v1.py')
contrast = load('official_rsm_legacy_contrast', HERE / 'legacy_contrast_v1.py')
hand = load('frozen_hand_tests_for_official', PRIOR / 'hand_tests_attempt01.py')
additive = load('frozen_additive_tests_for_official', PRIOR / 'hand_tests_attempt02.py')
# In-memory dependency replacement only: original files/results remain unchanged.
hand.machine = solver; hand.solve = solver.solve
additive.old.machine = solver; additive.old.solve = solver.solve
fx, reference = hand.fixtures, hand.reference
COUNTS = Counter()
CONTRAST_REPORT = None


def eq(a, b):
    if not solver.typed_equal(a, b): raise AssertionError(f'typed unequal: {a!r} vs {b!r}')


def reject(task, stage):
    eq(solver.solve(task), solver.invalid(stage)); COUNTS['new_invalid_controls'] += 1


def rename(value, names):
    if type(value) is dict: return {names.get(k, k): rename(v, names) for k, v in value.items()}
    if type(value) is list: return [rename(v, names) for v in value]
    if type(value) is str: return names.get(value, value)
    return value


class OfficialTaskCases(unittest.TestCase):
    def test_complete_option_parameter_object_metamorphics(self):
        names = {'root': 'opt-Z', 'child': 'opt-A', 'leaf': 'opt-M', 'tail': 'opt-N',
                 'a': 'pa-Z', 'b': 'pb-A', 'c': 'pc-M', 'd': 'pd-N'}
        for family in fx.hand_programs():
            c, _, _ = fx.contexts(family['task'], solver.solve)
            # Fresh/history/checkpoint names all transform together, including invalid X mutation.
            for context in c:
                if context == 'X': c[context]['checkpoint']['state']['x'] += 1
                t = c[context]; changed = rename(t, names)
                eq(rename(solver.solve(t), names), solver.solve(changed))
                eq(solver.solve(t), solver.solve(fx.reverse_objects(t)))
                COUNTS['option_parameter_object_metamorphics'] += 1

    def test_simultaneous_bindings_and_parameter_declaration_order(self):
        t = fx.hand_programs()[0]['task']
        t['options']['root']['params'] = ['a', 'b']
        t['entry']['args'] = {'a': fx.R('x'), 'b': fx.R('y')}
        t['options']['root']['body'] = [fx.add('x', fx.P('b')), fx.add('y', fx.P('a')), fx.swap(), fx.add('x', fx.P('a'))]
        out = solver.solve(t)
        eq(out['new_events'][0]['bindings'], {'a': 2, 'b': 1})
        eq(out['checkpoint']['state'], {'x': 5, 'y': 3})
        changed = deepcopy(t); changed['options']['root']['params'].reverse()
        changed['entry']['args'] = dict(reversed(list(changed['entry']['args'].items())))
        eq(solver.solve(changed), out)
        eq(reference.full_trace(t)['events'], out['new_events']); COUNTS['new_manual_reference_cases'] += 1

    def test_init_once_success_only_on_exit(self):
        t = fx.hand_programs()[0]['task']
        t['options']['root']['initiation'] = fx.pred(fx.atom(fx.R('x'), 'eq', fx.C(2)))
        t['options']['root']['success'] = fx.pred(fx.atom(fx.R('x'), 'eq', fx.C(2)))
        t['options']['root']['body'] = [fx.add('x', fx.C(1)), fx.add('x', fx.C(1))]
        out = solver.solve(t)
        self.assertEqual(out['status'], 'success_failed'); self.assertEqual(out['new_effects'], 2)
        eq(out['checkpoint']['state'], {'x': 4, 'y': 1})
        self.assertEqual([e['event'] for e in out['new_events']], ['enter', 'effect', 'effect', 'return'])
        eq(reference.full_trace(t)['events'], out['new_events']); COUNTS['new_manual_reference_cases'] += 1

    def test_preview_atomic_and_exact_arithmetic(self):
        t = fx.hand_programs()[0]['task']; t['event_budget'] = 1
        out = solver.solve(t)
        self.assertEqual(out['status'], 'paused'); self.assertEqual(out['new_effects'], 0)
        eq(out['checkpoint']['state'], {'x': 2, 'y': 1})
        self.assertEqual(out['checkpoint']['frames'][0]['next_index'], 0)
        eq(out['next_event']['state'], {'x': 4, 'y': 1})
        eq(solver.solve(t), out)
        t['options']['root']['body'] = [fx.add('x', fx.C(2 ** 150)), fx.scale('x', fx.C(2 ** 90))]
        t['event_budget'] = 48
        eq(solver.solve(t)['checkpoint']['state'], {'x': (2 + 2 ** 150) * 2 ** 90, 'y': 1})
        COUNTS['exact_arithmetic_controls'] += 1

    def test_task_types_and_all_program_branches(self):
        t = fx.hand_programs()[0]['task']
        for field in ('version', 'event_budget'):
            for value in (True, False, 1.0, '1', None):
                q = deepcopy(t); q[field] = value; reject(q, 'invalid_task')
        for value in (True, False, 2.0, '2', None):
            q = deepcopy(t); q['initial_state']['y'] = value; reject(q, 'invalid_task')
            q = deepcopy(t); q['entry']['args']['a'] = fx.C(value); reject(q, 'invalid_task')
        mutations = [
            lambda q: q['options']['root']['params'].append('a'),
            lambda q: q['entry']['args'].update(unused=fx.C(2)),
            lambda q: q['options']['child']['body'][0]['else'].append(fx.add('x', fx.C(True))),
            lambda q: q['options']['child']['body'][0]['else'].append(fx.call('missing', 'u', fx.C(1))),
            lambda q: q['options']['child']['body'][0]['else'].append(fx.call('root', 'a', fx.P('b'))),
            lambda q: q['options']['root']['success']['all'][0].update(cmp='gt'),
            lambda q: q['options']['root']['success']['all'][0]['right'].update(extra=1),
            lambda q: q['options'].update(unused=fx.opt('u', [fx.call('unused', 'u', fx.P('u'))])),
        ]
        for mutation in mutations:
            q = deepcopy(t); mutation(q); q['history'] = {'invalid': 'history'}; q['checkpoint'] = 'bad'
            reject(q, 'invalid_task')

    def test_history_types_fields_and_original_restart_replay(self):
        c, _, _ = fx.contexts(fx.hand_programs()[0]['task'], solver.solve)
        alterations = [
            lambda q: q['history'][0].update(instance_id=False),
            lambda q: q['history'][0].update(initiation_ok=1),
            lambda q: q['history'][0].update(call_site={}),
            lambda q: q['history'][1].update(index=False),
            lambda q: q['history'][1].update(value=2.0),
            lambda q: q['history'][1]['state'].update(y=True),
            lambda q: q['history'][2]['call_site'].update(caller_instance=False),
            lambda q: q['history'][3].update(condition=1),
            lambda q: q['history'][3].update(condition=False),
            lambda q: q['history'].reverse(),
        ]
        for mutation in alterations:
            q = deepcopy(c['T']); mutation(q); q['checkpoint'] = None
            reject(q, 'invalid_history')
        q = deepcopy(c['T']); q['history'][0]['bindings']['a'] = 6
        reject(q, 'invalid_history')  # Replay validates original entry x2, not restart x6.

    def test_checkpoint_exact_types_fields_and_normalization(self):
        c, _, _ = fx.contexts(fx.hand_programs()[0]['task'], solver.solve)
        changes = [
            lambda q: q['checkpoint'].update(next_instance_id=2.0),
            lambda q: q['checkpoint'].update(root_pending=0),
            lambda q: q['checkpoint'].update(phase='completed'),
            lambda q: q['checkpoint']['state'].update(y=True),
            lambda q: q['checkpoint']['frames'][0].update(next_index=1),
            lambda q: q['checkpoint']['frames'][1].update(instance_id=True),
            lambda q: q['checkpoint']['frames'][-1].update(next_index=True),
            lambda q: q['checkpoint']['frames'][-1]['body_path'].__setitem__(0, False),
            lambda q: q['checkpoint']['frames'][-1]['body_path'].__setitem__(1, 'else'),
            lambda q: q['checkpoint']['frames'][-1]['bindings'].update(b=2.0),
            lambda q: q['checkpoint']['frames'][-1].update(extra=None),
            lambda q: q['checkpoint']['frames'].reverse(),
        ]
        for mutation in changes:
            q = deepcopy(c['R']); mutation(q); reject(q, 'invalid_checkpoint')
        # First-stage and second-stage faults dominate a bad canonical snapshot.
        q = deepcopy(c['R']); q['checkpoint'] = None; q['history'][0]['option'] = 'wrong'; reject(q, 'invalid_history')
        q['event_budget'] = True; reject(q, 'invalid_task')

    def test_iterative_deep_valid_syntax_and_dag(self):
        # Stress public finite semantics outside cohort bounds; no private semantic depth cap.
        t = fx.hand_programs()[0]['task']; b = [fx.add('x', fx.C(1))]
        for _ in range(1100): b = [fx.branch(fx.pred(), b, [])]
        t['options']['root']['body'] = b; t['event_budget'] = 1200
        out = solver.solve(t)
        self.assertEqual(out['status'], 'completed'); self.assertEqual(len(out['new_events']), 1103)
        self.assertEqual(out['new_effects'], 1); COUNTS['deep_finite_branch_depth'] = 1100
        opts = {}
        for n in range(1100):
            body = [] if n == 1099 else [{'op': 'call', 'option': f'p{n + 1}', 'args': {}}]
            opts[f'p{n}'] = {'params': [], 'initiation': fx.pred(), 'success': fx.pred(), 'body': body}
        t = {'kind': 'option_resume', 'version': 1, 'initial_state': {'x': 0, 'y': 0},
             'options': opts, 'entry': {'option': 'p0', 'args': {}},
             'mode': 'fresh', 'history': [], 'checkpoint': None, 'event_budget': 2300}
        out = solver.solve(t)
        self.assertEqual(out['status'], 'completed'); self.assertEqual(len(out['new_events']), 2200)
        self.assertEqual(out['checkpoint']['next_instance_id'], 1100); COUNTS['deep_finite_call_depth'] = 1100

    def test_actual_legacy_graph_contrast_and_erasure_metamorphics(self):
        global CONTRAST_REPORT
        f = ROOT / 'reviews/input_only/RSM-legacy-v0001.jsonl'; raw = f.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), 'ba62d973531521bfc57a5b52ee630a61f3384d9e8dfd41becea6a1eb63ca86c1')
        self.assertEqual(len(raw), 71432)
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        CONTRAST_REPORT = contrast.compare(fx.hand_programs(), rows)
        self.assertEqual(CONTRAST_REPORT['legacy_records'], 24)
        self.assertEqual(CONTRAST_REPORT['legacy_source_families'], 12)
        for view in CONTRAST_REPORT['graph_views'].values():
            self.assertEqual(view['proposed_distinct_signatures'], 12)
            self.assertEqual(view['exact_new_legacy_collisions'], [])
            self.assertEqual(view['proposed_duplicate_pairs'], [])
            self.assertEqual(view['new_legacy_pair_comparisons'], 144)
        task = fx.hand_programs()[0]['task']
        names = {'root': 'different-main', 'child': 'different-callee', 'a': 'q0', 'b': 'q1', 'x': 'y', 'y': 'x'}
        changed = rename(task, names); changed['kind'] = 'spelling-does-not-prove-novelty'
        changed['history'] = [{'unused-label': True}]; changed['event_budget'] = 999
        def change_constants(x):
            if type(x) is dict:
                return {k: (900 if k == 'const' else change_constants(v)) for k, v in x.items()}
            if type(x) is list: return [change_constants(v) for v in x]
            return x
        changed = change_constants(changed)
        for norm in (contrast.full_option_signature, contrast.shared_primitive_signature,
                     lambda q: contrast.shared_primitive_signature(q, True)):
            self.assertEqual(norm(task), norm(changed)); COUNTS['graph_erasure_metamorphics'] += 1
        # Shared projection must ignore adding/removing guards/binding names.
        changed = deepcopy(task); changed['options']['root']['initiation'] = fx.pred(); changed['options']['root']['success'] = fx.pred()
        self.assertEqual(contrast.shared_primitive_signature(task), contrast.shared_primitive_signature(changed))


if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(hand.HandFeasibility),
                               unittest.defaultTestLoader.loadTestsFromTestCase(additive.AdditiveBoundaries),
                               unittest.defaultTestLoader.loadTestsFromTestCase(OfficialTaskCases)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'actual_role': '/root/formal_oracle', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'reused_hand_tests_executed_against_official_solver': True,
              'reused_hand_counts': dict(hand.COUNTS), 'reused_additive_counts': dict(additive.COUNTS),
              'new_counts': dict(COUNTS), 'legacy_graph_contrast': CONTRAST_REPORT,
              'future_production_data_or_gold_read': False, 'actual_target_free_legacy_inputs_read': True,
              'whole_corpus_control_gate': 'PENDING_PUBLIC_ENCODING_NOT_IMPLEMENTED', 'approval': None}
    (HERE / 'task-solver-results-attempt01.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
