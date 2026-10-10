"""Isolated hand tests; imported/executed only after the source freeze."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import unittest

HERE = Path(__file__).resolve().parent
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
machine = load('rsm_hand_stack', HERE / 'interpreter_attempt01.py')
fixtures = load('rsm_hand_fixtures', HERE / 'fixtures_attempt01.py')
simulator_path = HERE / 'separate_simulator/preexecution_freeze/recursive_full_trace.py'
assert hashlib.sha256(simulator_path.read_bytes()).hexdigest() == '4be10076b3419d8868f32603f44ea809245732f52ea14ddd22b592b86913696a'
reference = load('rsm_separate_recursive', simulator_path)
solve = machine.solve
COUNTS = Counter()
WITNESSES = []


def equal(a, b):
    if not machine.typed_equal(a, b): raise AssertionError(f'typed mismatch: {a!r} != {b!r}')


def paths(a, b, path=()):
    if type(a) is not type(b): return [path]
    if type(a) is dict:
        if set(a) != set(b): return [path]
        return [p for k in a for p in paths(a[k], b[k], path + (k,))]
    if type(a) is list:
        if len(a) != len(b): return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in paths(x, y, path + (i,))]
    return [] if a == b else [path]


def declaration_counts(task):
    def count(bb, depth=0):
        total, maximum = len(bb), depth
        for i in bb:
            if i['op'] == 'if':
                for side in ('then', 'else'):
                    n, d = count(i[side], depth + 1); total += n; maximum = max(maximum, d)
        return total, maximum
    sizes = [count(o['body']) for o in task['options'].values()]
    return sum(n for n, _ in sizes), max(d for _, d in sizes)


class HandFeasibility(unittest.TestCase):
    def test_all_twelve_simultaneous_nine_contexts_and_bounds(self):
        families = fixtures.hand_programs()
        self.assertEqual(len(families), 12)
        self.assertEqual(Counter(f['split'] for f in families), {'train': 8, 'dev_challenge': 4})
        signatures = set()
        for ordinal, family in enumerate(families):
            base = family['task']; c, cut, pbudget = fixtures.contexts(base, solve)
            rframe = c['R']['checkpoint']['frames'][-1]
            # Exactly one actually consequential state/cursor/binding mutation.
            mutation = ordinal % 3
            if mutation == 0: c['X']['checkpoint']['state']['x'] += 1
            elif mutation == 1: c['X']['checkpoint']['frames'][-1]['next_index'] += 1
            else: c['X']['checkpoint']['frames'][-1]['bindings']['b'] += 1
            out = {k: solve(t) for k, t in c.items()}
            self.assertEqual(set(c), set('FRTPUAISX'))
            for context in 'FRUA': self.assertEqual(out[context]['status'], 'completed')
            self.assertEqual(out['T']['status'], 'initiation_failed')
            self.assertEqual(out['P']['status'], 'paused')
            self.assertEqual(out['I']['status'], 'initiation_failed')
            self.assertEqual(out['S']['status'], 'success_failed')
            self.assertEqual(out['X']['error'], 'invalid_checkpoint')
            self.assertEqual(out['X']['new_events'], [])
            equal(out['R']['new_events'], out['F']['new_events'][cut:])
            equal(out['U']['new_events'], out['F']['new_events'][cut + pbudget:])
            equal(out['R']['checkpoint'], out['F']['checkpoint'])
            equal(out['U']['checkpoint'], out['F']['checkpoint'])
            self.assertEqual(paths(c['R'], c['T']), [('mode',)])
            self.assertEqual(paths(c['R'], c['P']), [('event_budget',)])
            self.assertEqual(len(paths(c['F'], c['A'])), 1)
            self.assertEqual(len(paths(c['F'], c['I'])), 1)
            self.assertEqual(len(paths(c['F'], c['S'])), 1)
            self.assertEqual(len(paths(c['R'], c['X'])), 1)
            checkpoint = c['R']['checkpoint']
            self.assertGreaterEqual(sum(f['frame_kind'] == 'option' for f in checkpoint['frames']), 2)
            self.assertTrue(any(f['frame_kind'] == 'branch' for f in checkpoint['frames']))
            self.assertEqual(checkpoint['frames'][0]['next_index'], 2)
            self.assertEqual(rframe['bindings']['b'], 2)
            self.assertFalse(machine.predicate(base['options']['child']['initiation'], checkpoint['state'], {'b': 2}))
            self.assertTrue(machine.predicate(base['options']['root']['initiation'], checkpoint['state'], {'a': checkpoint['state']['x']}))
            self.assertTrue(any(e['event'] == 'effect' and e.get('value') == 2 for e in out['R']['new_events']))
            self.assertGreater(out['P']['new_effects'], 0)
            self.assertTrue(out['U']['new_events'])
            self.assertTrue(any(e['event'] == 'effect' for e in out['U']['new_events']))
            original = next(e for e in c['R']['history'] if e['event'] == 'effect' and e['option'] == 'root')
            replay = next(e for e in out['T']['new_events'] if e['event'] == 'effect' and e['option'] == 'root')
            equal({k: original[k] for k in ('option', 'body_path', 'index', 'op', 'register')},
                  {k: replay[k] for k in ('option', 'body_path', 'index', 'op', 'register')})
            self.assertNotEqual(replay['state']['x'], checkpoint['state']['x'])
            self.assertEqual(replay['state']['x'], checkpoint['state']['x'] + replay['value'])
            self.assertNotEqual(out['T']['checkpoint']['state'], out['R']['checkpoint']['state'])
            self.assertNotEqual(out['T']['new_events'], out['R']['new_events'])
            self.assertNotEqual(out['A']['checkpoint']['state'], out['F']['checkpoint']['state'])
            ai = base['options']['root']['initiation']['all']
            self.assertEqual([machine.predicate({'all': [a]}, base['initial_state'], {'a': 0}) for a in ai], [False, True])
            self.assertEqual(out['I']['new_effects'], 0)
            self.assertEqual(len(out['I']['new_events']), 1)
            equal([e for e in out['S']['new_events'] if e['event'] == 'effect'],
                  [e for e in out['F']['new_events'] if e['event'] == 'effect'])
            self.assertEqual(out['S']['new_events'][-1]['success_ok'], False)
            self.assertGreaterEqual(out['F']['new_effects'], 3)
            self.assertLessEqual(len(out['F']['new_events']), 48)
            size, depth = declaration_counts(base)
            self.assertLessEqual(size, 24); self.assertLessEqual(depth, 2)
            self.assertTrue(2 <= len(base['options']) <= 4)
            self.assertTrue(all(1 <= len(o['params']) <= 3 for o in base['options'].values()))
            for name in 'FRT PUAIS'.replace(' ', ''):
                for e in out[name]['new_events']:
                    if e['event'] == 'effect': self.assertTrue(all(-4096 <= v <= 4096 for v in e['state'].values()))
            n = len(out['F']['new_events'])
            maxdepth = 0
            for budget in range(n + 1):
                t = deepcopy(base); t['event_budget'] = budget
                maxdepth = max(maxdepth, sum(f['frame_kind'] == 'option' for f in solve(t)['checkpoint']['frames']))
            self.assertLessEqual(maxdepth, 3)
            signature = fixtures.normalized_signature(base)
            self.assertNotIn(signature, signatures); signatures.add(signature)
            for name, task in c.items():
                equal(solve(task), solve(fixtures.reverse_objects(task)))
                COUNTS['field_order_pairs'] += 1
            for name in ('F', 'A', 'I', 'S'):
                recursive = reference.full_trace(c[name])
                equal(recursive['events'], out[name]['new_events'])
                equal(recursive['state'], out[name]['checkpoint']['state'])
                self.assertEqual(recursive['phase'], out[name]['status'])
                COUNTS['separate_recursive_full_runs'] += 1
            restart = reference.full_trace(base, start_state=checkpoint['state'])
            equal(restart['events'], out['T']['new_events'])
            equal(restart['state'], out['T']['checkpoint']['state'])
            self.assertEqual(restart['phase'], out['T']['status'])
            COUNTS['separate_recursive_full_runs'] += 1
            COUNTS.update({'families': 1, 'semantic_context_tasks': 9, 'rendered_equivalents': 18, 'replay_witnesses': 1, 'second_checkpoints': 1})
            WITNESSES.append({'source_family_index': family['source_family_index'], 'split': family['split'],
                              'stratum': family['stratum'], 'cut': cut, 'P_budget': pbudget,
                              'instruction_count': size, 'conditional_depth': depth, 'option_depth': maxdepth,
                              'signature_sha256': hashlib.sha256(signature.encode()).hexdigest(),
                              'statuses': {k: v['status'] for k, v in out.items()},
                              'F_events': n, 'F_effects': out['F']['new_effects'],
                              'R_checkpoint': checkpoint, 'P_checkpoint': out['P']['checkpoint'],
                              'original_committed_effect': original, 'replayed_effect': replay,
                              'R_terminal_state': out['R']['checkpoint']['state'],
                              'T_terminal_state': out['T']['checkpoint']['state'],
                              'suffix_equal': True, 'second_suffix_equal': True, 'X_changed_path': list(paths(c['R'], c['X'])[0])})

    def test_every_cut_and_continuation_budget(self):
        for family in fixtures.hand_programs():
            task = family['task']; events = solve(task)['new_events']; n = len(events)
            for cut in range(n + 1):
                t = deepcopy(task); t['event_budget'] = cut
                paused = solve(t)
                r = deepcopy(task); r.update(mode='resume', history=deepcopy(events[:cut]), checkpoint=paused['checkpoint'])
                for budget in range(n - cut + 2):
                    r['event_budget'] = budget; out = solve(r)
                    equal(out['new_events'], events[cut:cut + budget])
                    self.assertEqual(out['previous_effects'], sum(e['event'] == 'effect' for e in events[:cut]))
                    self.assertEqual(out['new_effects'], sum(e['event'] == 'effect' for e in events[cut:cut + budget]))
                    if cut + budget < n:
                        self.assertEqual(out['status'], 'paused'); equal(out['next_event'], events[cut + budget])
                    else: self.assertEqual(out['status'], 'completed'); self.assertIsNone(out['next_event'])
                    composed = deepcopy(task); composed['event_budget'] = min(cut + budget, n)
                    equal(out['checkpoint'], solve(composed)['checkpoint'])
                    COUNTS['cut_budget_compositions'] += 1
                COUNTS['legal_event_cuts'] += 1

    def test_manual_immutable_and_simultaneous_arguments(self):
        f = fixtures
        t = {'kind': 'option_resume', 'version': 1, 'initial_state': {'x': 2, 'y': 1},
             'options': {'root': f.opt('a', [f.add('x', f.P('a')), {'op': 'call', 'option': 'child', 'args': {'u': f.R('x'), 'v': f.P('a')}}]),
                         'child': {'params': ['u', 'v'], 'initiation': f.pred(), 'success': f.pred(),
                                   'body': [f.add('x', f.P('v')), f.add('y', f.P('u'))]}},
             'entry': {'option': 'root', 'args': {'a': f.R('x')}}, 'mode': 'fresh', 'history': [], 'checkpoint': None, 'event_budget': 48}
        out = solve(t)
        equal(out['new_events'][2]['bindings'], {'u': 4, 'v': 2})
        equal(out['checkpoint']['state'], {'x': 6, 'y': 5})
        self.assertEqual(out['new_effects'], 3)
        equal(reference.full_trace(t)['events'], out['new_events']); COUNTS['manual_reference_cases'] += 1

    def test_manual_empty_frames_zero_and_terminal_budgets(self):
        t = fixtures.hand_programs()[0]['task']; t['options']['root']['body'] = []
        t['event_budget'] = 0; z = solve(t)
        self.assertEqual(z['status'], 'paused'); self.assertEqual(z['checkpoint']['root_pending'], True)
        self.assertEqual(z['next_event']['event'], 'enter'); self.assertEqual(z['checkpoint']['next_instance_id'], 0)
        t['event_budget'] = 1; e = solve(t)
        self.assertEqual(e['status'], 'paused'); self.assertEqual(e['next_event']['event'], 'return')
        t['event_budget'] = 2; end = solve(t)
        self.assertEqual(end['status'], 'completed')
        r = deepcopy(t); r.update(mode='resume', history=end['new_events'], checkpoint=end['checkpoint'], event_budget=0)
        self.assertEqual(solve(r)['status'], 'completed'); self.assertEqual(solve(r)['new_events'], [])
        r['mode'] = 'restart'; self.assertEqual(solve(r)['status'], 'paused')
        t['options']['root']['body'] = [fixtures.branch(fixtures.pred(), [], [])]
        t['event_budget'] = 2; b = solve(t)
        self.assertEqual([x['frame_kind'] for x in b['checkpoint']['frames']], ['option'])
        self.assertEqual(b['next_event']['event'], 'return')
        t['event_budget'] = 48; equal(reference.full_trace(t)['events'], solve(t)['new_events'])
        COUNTS['manual_reference_cases'] += 1

    def test_nested_failure_retained_caller_and_no_rollback(self):
        t = fixtures.hand_programs()[0]['task']
        t['options']['child']['success'] = fixtures.pred(fixtures.atom(fixtures.R('x'), 'lt', fixtures.C(0)))
        o = solve(t); self.assertEqual(o['status'], 'success_failed')
        self.assertEqual(len(o['checkpoint']['frames']), 1)
        self.assertEqual(o['checkpoint']['frames'][0]['next_index'], 2)
        equal(o['checkpoint']['state'], {'x': 6, 'y': 3})
        equal(reference.full_trace(t)['events'], o['new_events']); COUNTS['manual_reference_cases'] += 1
        r = deepcopy(t); r.update(mode='resume', history=o['new_events'], checkpoint=o['checkpoint'])
        equal(solve(r)['checkpoint'], o['checkpoint']); self.assertEqual(solve(r)['new_events'], [])
        t['options']['child']['initiation'] = fixtures.pred(fixtures.atom(fixtures.R('x'), 'lt', fixtures.C(0)))
        o = solve(t); self.assertEqual(o['status'], 'initiation_failed'); self.assertEqual(o['new_effects'], 1)
        self.assertEqual(o['checkpoint']['next_instance_id'], 2)
        self.assertEqual(o['checkpoint']['frames'][0]['next_index'], 2)
        equal(reference.full_trace(t)['events'], o['new_events']); COUNTS['manual_reference_cases'] += 1

    def test_malformed_validation_precedence_and_typed_history(self):
        base = fixtures.hand_programs()[0]['task']; c, _, _ = fixtures.contexts(base, solve)
        def rejects(t, stage):
            equal(solve(t), machine.invalid(stage)); COUNTS['malformed_cases'] += 1
        for key, value in [('version', True), ('event_budget', False), ('event_budget', -1), ('mode', 'other')]:
            t = deepcopy(base); t[key] = value; t['history'] = 'bad'; t['checkpoint'] = 'bad'; rejects(t, 'invalid_task')
        for value in (True, 2.0, '2', None):
            t = deepcopy(base); t['initial_state']['x'] = value; rejects(t, 'invalid_task')
        t = deepcopy(base); t['options']['root']['body'][0]['value']['param'] = 'missing'; rejects(t, 'invalid_task')
        t = deepcopy(base); t['entry']['args']['a'] = fixtures.P('a'); rejects(t, 'invalid_task')
        t = deepcopy(base); t['options']['child']['params'].append('b'); rejects(t, 'invalid_task')
        t = deepcopy(base); t['options']['child']['body'][0]['else'] = [fixtures.call('root', 'a', fixtures.P('b'))]; rejects(t, 'invalid_task')
        t = deepcopy(base); t['options']['unused'] = fixtures.opt('u', [fixtures.call('unused', 'u', fixtures.P('u'))]); rejects(t, 'invalid_task')
        t = deepcopy(base); t['options']['root']['body'][0]['unexpected'] = 1; rejects(t, 'invalid_task')
        for hist in ({}, None, True):
            t = deepcopy(base); t['history'] = hist; t['checkpoint'] = 'bad'; rejects(t, 'invalid_history')
        for bad in ('wrong binding', 'extra key', 'boolean outcome', 'float effect', 'event order'):
            t = deepcopy(c['R']); t['checkpoint'] = 'bad'
            if bad == 'wrong binding': t['history'][0]['bindings']['a'] = 3
            if bad == 'extra key': t['history'][0]['unexpected'] = None
            if bad == 'boolean outcome': t['history'][0]['initiation_ok'] = 1
            if bad == 'float effect': t['history'][1]['value'] = 2.0
            if bad == 'event order': t['history'][0], t['history'][1] = t['history'][1], t['history'][0]
            rejects(t, 'invalid_history')
        full = solve(base)
        t = deepcopy(base); t.update(mode='resume', history=full['new_events'] + [full['new_events'][-1]], checkpoint=full['checkpoint']); rejects(t, 'invalid_history')
        for key, value in [('root_pending', 0), ('next_instance_id', True), ('phase', 'completed'), ('state', {'x': 6.0, 'y': 1})]:
            t = deepcopy(c['R']); t['checkpoint'][key] = value; rejects(t, 'invalid_checkpoint')
        t = deepcopy(c['R']); t['checkpoint']['frames'][-1]['bindings']['b'] = True; rejects(t, 'invalid_checkpoint')
        t = deepcopy(c['R']); t['checkpoint']['frames'][-1]['body_path'][0] = False; rejects(t, 'invalid_checkpoint')
        t = deepcopy(c['R']); t['checkpoint']['frames'][-1]['next_index'] = True; rejects(t, 'invalid_checkpoint')
        t = deepcopy(c['R']); t['mode'] = 'fresh'; rejects(t, 'invalid_checkpoint')
        t = deepcopy(base); t['checkpoint'] = machine.initial(t); rejects(t, 'invalid_checkpoint')
        # Correct exhausted branch must have been removed; a superficially legal redundant frame is invalid.
        t = deepcopy(base); t['event_budget'] = 6; p = solve(t)
        self.assertEqual(p['next_event']['event'], 'return')
        t.update(mode='resume', history=p['new_events'], checkpoint=p['checkpoint'])
        child = deepcopy(t['checkpoint']['frames'][-1]); child.update(frame_kind='branch', body_path=[0, 'then'], next_index=2)
        t['checkpoint']['frames'].append(child); rejects(t, 'invalid_checkpoint')

    def test_name_order_constant_erasure_signature(self):
        t = fixtures.hand_programs()[0]['task']; original = fixtures.normalized_signature(t)
        renames = {'root': 'R99', 'child': 'C99', 'a': 'alpha', 'b': 'beta'}
        def rename(x):
            if type(x) is dict: return {renames.get(k, k): rename(v) for k, v in x.items()}
            if type(x) is list: return [rename(v) for v in x]
            if type(x) is str: return renames.get(x, x)
            return x
        changed = rename(t)
        self.assertEqual(fixtures.normalized_signature(changed), original)
        out, after = solve(t), solve(changed)
        equal(rename(out), after)
        self.assertEqual(fixtures.normalized_signature(fixtures.reverse_objects(t)), original)
        changed['options']['R99']['success']['all'][0]['right'] = fixtures.C(-1)
        self.assertEqual(fixtures.normalized_signature(changed), original)
        COUNTS['rename_metamorphics'] += 1


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HandFeasibility)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'actual_role': '/root/formal_oracle', 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'counts': dict(sorted(COUNTS.items())), 'family_witnesses': WITNESSES,
              'production_data_read': False, 'production_approval': None,
              'legacy_input_graph_comparison': 'NOT_RUN; no legacy canonical body read; public root measurement documents absence of option argument/guard/checkpoint grammar.'}
    (HERE / 'hand-results-attempt01.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
