"""Independent adversarial examples with specified, hand-calculated outcomes.

These tests were authored by E/F, who did not author environments, oracles,
generators, scoring, or operations. No generated expected labels are used.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import argparse
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import runpy
import signal
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


UTILITY_DEFINITION = {'comparison': 'lexicographic',
    'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
    'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
    'all_infeasible': 'abstain; no best candidate'}


def ledger_task():
    return {
        'environment': 'ledger',
        'initial_state': {'accounts': {'oak': 10, 'pine': 4},
                          'entries': {'bill': {'account': 'oak', 'amount': 3, 'status': 'waiting'}}},
        'rules': {'statuses': {'pending': 'waiting', 'settled': 'recorded', 'void': 'void'},
                  'void_effect': 'reverse', 'allow_overdraft': False,
                  'balance_bounds': [-100, 100],
                  'action_costs': {'settle': 1, 'void': 2, 'transfer': 1, 'noop': 0}},
        'goal': {'balances': {'oak': 13}}, 'constraints': {},
        'utility_definition': deepcopy(UTILITY_DEFINITION),
        'candidates': [
            {'id': 'option_x', 'plan': [{'op': 'settle', 'entry': 'bill'}]},
            {'id': 'option_y', 'plan': [{'op': 'settle', 'entry': 'bill'}, {'op': 'void', 'entry': 'bill'}]},
            {'id': 'option_z', 'plan': []}],
    }


def blocks_task():
    return {'environment': 'blocksworld',
            'initial_state': {'on': {'oak': 'table', 'pine': 'oak', 'elm': 'table'}},
            'rules': {'table': 'table', 'max_height': 3, 'action_costs': {'move': 2, 'noop': 0}},
            'goal': {'on': {'oak': 'elm'}}, 'constraints': {},
            'utility_definition': deepcopy(UTILITY_DEFINITION),
            'candidates': [
                {'id': 'alpha', 'plan': [{'op': 'move', 'block': 'oak', 'destination': 'elm'}]},
                {'id': 'beta', 'plan': [{'op': 'move', 'block': 'pine', 'destination': 'table'},
                                       {'op': 'move', 'block': 'oak', 'destination': 'elm'}]},
                {'id': 'gamma', 'plan': []}]}


def exact(value):
    return Fraction(value['numerator'], value['denominator']) if isinstance(value, dict) else Fraction(value)


def schema_fixture():
    """A manual structural record; deliberately no derived outcome fixtures."""
    from goal_conditioned_decision_data.pipelines.common import digest
    from goal_conditioned_decision_data.generators.build import record_id
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/curriculum.json').read_text())
    task = ledger_task()
    for index, candidate in enumerate(task['candidates']): candidate['id'] = 'c_' + format(index, '024x')
    template = config['split_templates']['train']['template']
    signature = digest({'environment': 'ledger', 'world_size': [2, 1],
                        'rule_keys': sorted(task['rules']), 'constraint_keys': []})
    record = {'schema_version': 'gcdl-1.0', 'family': 'A', 'difficulty': 'adversarial', 'split': 'train',
              'provenance': {'seed': 610700, 'generator_version': config['generator_version'],
                             'environment_version': 'environments-1.0', 'config_version': config['version'],
                             'config_hash': digest(config), 'template_id': template,
                             'template_group_id': template, 'lineage_id': template,
                             'parent_world_id': 'independent_manual_parent', 'record_index': 0},
              'input': task, 'targets': {'oracle': {}, 'decision': {}},
              'metadata': {'tags': ['independent_schema_fixture'], 'counterfactual_group_id': 'independent_manual_parent',
                           'protected_template_signature': signature,
                           'perturbation': {'kind': 'none', 'label_relation': 'base'}}}
    record['id'] = record_id(record)
    return record


class IndependentSemanticTests(unittest.TestCase):
    def evaluate_both(self, task):
        from goal_conditioned_decision_data.oracles.primary import evaluate as first
        from goal_conditioned_decision_data.oracles.independent import evaluate as second
        a, b = first(deepcopy(task)), second(deepcopy(task))
        self.assertEqual(a, b)
        return {c['id']: c for c in a['candidates']}

    def test_reverse_and_retain_void_rules_change_actual_balances(self):
        task = ledger_task()
        results = self.evaluate_both(task)
        self.assertEqual(results['option_x']['outcomes'][0]['terminal_state']['accounts']['oak'], 13)
        self.assertEqual(results['option_y']['outcomes'][0]['terminal_state']['accounts']['oak'], 10)
        self.assertEqual(exact(results['option_x']['success_probability']), 1)
        self.assertEqual(exact(results['option_y']['success_probability']), 0)
        task['rules']['void_effect'] = 'retain'
        results = self.evaluate_both(task)
        self.assertEqual(results['option_y']['outcomes'][0]['terminal_state']['accounts']['oak'], 13)
        self.assertEqual(exact(results['option_y']['success_probability']), 1)

    def test_void_spelling_can_mean_settled_and_new_word_can_mean_void(self):
        task = ledger_task()
        task['rules']['statuses'] = {'pending': 'fresh', 'settled': 'void', 'void': 'withdrawn'}
        task['initial_state']['entries']['bill']['status'] = 'void'
        task['initial_state']['accounts']['oak'] = 13
        task['goal'] = {'balances': {'oak': 10}}
        task['candidates'] = [{'id': 'x', 'plan': [{'op': 'void', 'entry': 'bill'}]}]
        result = self.evaluate_both(task)['x']
        self.assertEqual(exact(result['legal_probability']), 1)
        self.assertEqual(exact(result['success_probability']), 1)
        self.assertEqual(result['outcomes'][0]['terminal_state']['entries']['bill']['status'], 'withdrawn')
        task['initial_state']['entries']['bill']['status'] = 'withdrawn'
        result = self.evaluate_both(task)['x']
        self.assertEqual(exact(result['legal_probability']), 0)

    def test_negative_entry_settle_and_reverse_are_sign_symmetric(self):
        task = ledger_task()
        task['initial_state']['entries']['bill']['amount'] = -3
        task['goal'] = {'balances': {'oak': 7}}
        result = self.evaluate_both(task)
        self.assertEqual(result['option_x']['outcomes'][0]['terminal_state']['accounts']['oak'], 7)
        self.assertEqual(result['option_y']['outcomes'][0]['terminal_state']['accounts']['oak'], 10)
        self.assertEqual(exact(result['option_x']['success_probability']), 1)

    def test_unknown_status_and_invalid_initial_state_are_rejected(self):
        from goal_conditioned_decision_data.oracles.primary import evaluate as first
        from goal_conditioned_decision_data.oracles.independent import evaluate as second
        bad_tasks = []
        task = ledger_task(); task['initial_state']['entries']['bill']['status'] = 'new_undeclared_word'; bad_tasks.append(task)
        task = ledger_task(); task['initial_state']['accounts']['oak'] = True; bad_tasks.append(task)
        task = ledger_task(); task['initial_state']['accounts']['oak'] = 101; bad_tasks.append(task)
        task = ledger_task(); task['rules']['statuses']['void'] = 'waiting'; bad_tasks.append(task)
        task = blocks_task(); task['initial_state']['on'] = {'oak': 'pine', 'pine': 'oak', 'elm': 'table'}; bad_tasks.append(task)
        task = blocks_task(); task['initial_state']['on']['elm'] = 'oak'; bad_tasks.append(task)
        task = ledger_task(); task['information'] = 'missing'; task['probability_assumption'] = 'uniform_over_possible_worlds'
        world = deepcopy(task['initial_state']); task['initial_state']['entries']['bill']['amount'] = None
        task['possible_worlds'] = [world, deepcopy(world)]; bad_tasks.append(task)
        task = ledger_task(); task['information'] = 'missing'; task['probability_assumption'] = 'uniform_over_possible_worlds'
        world_a = deepcopy(task['initial_state']); world_b = deepcopy(world_a); world_b['accounts']['oak'] = 11
        task['initial_state']['entries']['bill']['amount'] = None
        task['possible_worlds'] = [world_a, world_b]; bad_tasks.append(task)
        for invalid in bad_tasks:
            for evaluate in (first, second):
                with self.subTest(task=invalid, checker=evaluate.__module__):
                    with self.assertRaises((ValueError, TypeError)):
                        evaluate(invalid)

    def test_blocked_block_illegal_and_cleared_prefix_legal(self):
        result = self.evaluate_both(blocks_task())
        self.assertEqual(exact(result['alpha']['legal_probability']), 0)
        self.assertEqual(result['alpha']['outcomes'][0]['terminal_state']['on']['oak'], 'table')
        self.assertEqual(exact(result['beta']['legal_probability']), 1)
        self.assertEqual(exact(result['beta']['success_probability']), 1)
        self.assertEqual(result['beta']['outcomes'][0]['terminal_state']['on'], {'oak': 'elm', 'pine': 'table', 'elm': 'table'})

    def test_exact_uniform_possible_world_success(self):
        task = ledger_task()
        world_a = deepcopy(task['initial_state'])
        world_b = deepcopy(world_a); world_b['entries']['bill']['amount'] = 4
        world_c = deepcopy(world_a); world_c['entries']['bill']['amount'] = 5
        task['initial_state']['entries']['bill']['amount'] = None
        task['possible_worlds'] = [world_a, world_b, world_c]
        task['information'] = 'ambiguous'
        task['probability_assumption'] = 'uniform_over_possible_worlds'
        result = self.evaluate_both(task)
        self.assertEqual(exact(result['option_x']['success_probability']), Fraction(1, 3))
        self.assertEqual(exact(result['option_x']['legal_probability']), 1)
        self.assertEqual(exact(result['option_y']['success_probability']), 0)

    def test_goal_reversal_changes_selection_without_changing_dynamics(self):
        from goal_conditioned_decision_data.generators.build import decision_labels
        from goal_conditioned_decision_data.oracles.primary import evaluate
        task = ledger_task(); forward = evaluate(task)
        task['goal'] = {'balances': {'oak': 10}}; reverse = evaluate(task)
        self.assertEqual([c['outcomes'][0]['terminal_state'] for c in forward['candidates']],
                         [c['outcomes'][0]['terminal_state'] for c in reverse['candidates']])
        self.assertEqual(decision_labels(forward)['best_candidate_ids'], ['option_x'])
        self.assertEqual(decision_labels(reverse)['best_candidate_ids'], ['option_z'])

    def test_empty_plan_and_zero_cost_noop_tie(self):
        from goal_conditioned_decision_data.generators.build import decision_labels
        from goal_conditioned_decision_data.oracles.primary import evaluate
        task = ledger_task(); task['goal'] = {'balances': {'oak': 10}}
        task['candidates'] = [{'id': 'u', 'plan': []}, {'id': 'v', 'plan': [{'op': 'noop'}]}]
        labels = decision_labels(evaluate(task))
        self.assertEqual(labels['best_candidate_ids'], ['u', 'v'])
        self.assertEqual(labels['uncertainty']['label'], 'tied_best')

    def test_all_candidates_infeasible_requires_abstention(self):
        from goal_conditioned_decision_data.generators.build import decision_labels
        from goal_conditioned_decision_data.oracles.primary import evaluate
        # Settle repairs the final balance, but cannot erase the initial violation.
        task = ledger_task(); task['constraints'] = {'minimum_balances': {'oak': 12}}
        labels = decision_labels(evaluate(task))
        self.assertEqual(labels['best_candidate_ids'], [])
        self.assertEqual(labels['uncertainty']['abstention'], 'no_feasible_candidate')

    def test_illegal_prefix_stops_and_charges_only_successful_actions(self):
        task = ledger_task()
        task['candidates'] = [{'id': 'prefix', 'plan': [{'op': 'settle', 'entry': 'bill'},
                                                      {'op': 'settle', 'entry': 'bill'},
                                                      {'op': 'void', 'entry': 'bill'}]}]
        candidate = self.evaluate_both(task)['prefix']; outcome = candidate['outcomes'][0]
        self.assertEqual(outcome['terminal_state']['accounts']['oak'], 13)
        self.assertEqual(outcome['terminal_state']['entries']['bill']['status'], 'recorded')
        self.assertEqual(outcome['failure_step'], 2)
        self.assertEqual(outcome['cost'], 1)
        self.assertEqual(len(outcome['steps']), 2)
        self.assertEqual(len(outcome['trajectory']), 2)
        self.assertEqual([exact(v) for v in candidate['utility']], [0, 1, -1])
        self.assertEqual(exact(candidate['success_probability']), 0)

    def test_repaired_intermediate_constraint_violation_still_counts(self):
        task = ledger_task(); task['goal'] = {'balances': {'oak': 10}}
        task['constraints'] = {'minimum_balances': {'oak': 8}}
        task['candidates'] = [
            {'id': 'repair', 'plan': [{'op': 'transfer', 'source': 'oak', 'destination': 'pine', 'amount': 5},
                                     {'op': 'transfer', 'source': 'pine', 'destination': 'oak', 'amount': 5}]},
            {'id': 'safe', 'plan': []}]
        result = self.evaluate_both(task)
        self.assertEqual(result['repair']['outcomes'][0]['terminal_state']['accounts']['oak'], 10)
        self.assertEqual(exact(result['repair']['legal_probability']), 1)
        self.assertEqual(exact(result['repair']['feasible_probability']), 0)
        self.assertEqual(exact(result['repair']['success_probability']), 0)
        self.assertEqual(result['repair']['outcomes'][0]['constraint_violations'], [{'constraint': 'minimum_balance:oak', 'step': 1}])
        self.assertEqual(exact(result['safe']['success_probability']), 1)

    def test_candidate_ids_and_order_do_not_affect_outcomes(self):
        task = ledger_task()
        base = self.evaluate_both(task)
        renamed = deepcopy(task)
        mapping = {'option_x': 'z9', 'option_y': 'a1', 'option_z': 'void'}
        for candidate in renamed['candidates']: candidate['id'] = mapping[candidate['id']]
        renamed['candidates'].reverse()
        result = self.evaluate_both(renamed)
        for old, new in mapping.items():
            left, right = deepcopy(base[old]), deepcopy(result[new])
            left.pop('id'); right.pop('id')
            self.assertEqual(left, right)

    def test_register_add_multiply_copy_swap_and_set_controls(self):
        task = {'environment': 'register', 'initial_state': {'registers': {'r': 3, 's': -2}},
                'rules': {'value_bounds': [-10, 10], 'overflow': 'reject',
                          'operations': ['add', 'multiply', 'set', 'copy', 'swap', 'noop'],
                          'action_costs': {'add': 1, 'multiply': 1, 'set': 1, 'copy': 1, 'swap': 1, 'noop': 0}},
                'goal': {'values': {'r': 6}}, 'constraints': {},
                'utility_definition': deepcopy(UTILITY_DEFINITION),
                'candidates': [
                    {'id': 'a', 'plan': [{'op': 'add', 'register': 'r', 'value': 3}]},
                    {'id': 'm', 'plan': [{'op': 'multiply', 'register': 'r', 'value': 2}]},
                    {'id': 'c', 'plan': [{'op': 'copy', 'source': 'r', 'destination': 's'}]},
                    {'id': 's', 'plan': [{'op': 'swap', 'left': 'r', 'right': 's'}]},
                    {'id': 't', 'plan': [{'op': 'set', 'register': 'r', 'value': 6}]},
                    {'id': 'bad', 'plan': [{'op': 'add', 'register': 'r', 'value': 8}]}]}
        result = self.evaluate_both(task)
        expected = {'a': {'r': 6, 's': -2}, 'm': {'r': 6, 's': -2},
                    'c': {'r': 3, 's': 3}, 's': {'r': -2, 's': 3},
                    't': {'r': 6, 's': -2}, 'bad': {'r': 3, 's': -2}}
        for candidate, terminal in expected.items():
            self.assertEqual(result[candidate]['outcomes'][0]['terminal_state']['registers'], terminal)
        self.assertEqual(exact(result['bad']['legal_probability']), 0)
        self.assertEqual(exact(result['a']['success_probability']), 1)

    def test_register_overflow_rule_reversal_changes_endpoint(self):
        task = {'environment': 'register', 'initial_state': {'registers': {'r': 5}},
                'rules': {'value_bounds': [-5, 5], 'overflow': 'clamp', 'operations': ['add', 'noop'],
                          'action_costs': {'add': 1, 'noop': 0}},
                'goal': {'values': {'r': -5}}, 'constraints': {},
                'utility_definition': deepcopy(UTILITY_DEFINITION),
                'candidates': [{'id': 'x', 'plan': [{'op': 'add', 'register': 'r', 'value': 1}]}]}
        clamp = self.evaluate_both(task)['x']
        self.assertEqual(clamp['outcomes'][0]['terminal_state']['registers']['r'], 5)
        self.assertEqual(exact(clamp['success_probability']), 0)
        task['rules']['overflow'] = 'wrap'
        wrapped = self.evaluate_both(task)['x']
        self.assertEqual(wrapped['outcomes'][0]['terminal_state']['registers']['r'], -5)
        self.assertEqual(exact(wrapped['success_probability']), 1)


class IndependentOperationsTests(unittest.TestCase):
    def test_immutable_publish_replay_and_conflict_preserve_original(self):
        from goal_conditioned_decision_data.pipelines.common import immutable_bytes
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            path = Path(directory)/'shard.jsonl'
            immutable_bytes(path, b'original\n')
            immutable_bytes(path, b'original\n')
            with self.assertRaises(ValueError): immutable_bytes(path, b'changed\n')
            self.assertEqual(path.read_bytes(), b'original\n')
            self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_guard_stops_low_disk_storage_stop_and_parallelism(self):
        from goal_conditioned_decision_data.pipelines import worker
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            root = Path(directory); (root/'state').mkdir()
            db = sqlite3.connect(':memory:'); db.execute('CREATE TABLE jobs(accepted INTEGER)')
            config = {'min_free_bytes': 10, 'max_programme_bytes': 1000, 'max_records': 10, 'concurrency': 1,
                      'groups_per_cycle': 1, 'cycle_timeout_seconds': 180, 'max_consecutive_failures': 3}
            with patch.object(worker, 'ROOT', root), patch.object(worker.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 9})()):
                with self.assertRaises(RuntimeError): worker.guard(config, db)
            with patch.object(worker, 'ROOT', root), patch.object(worker.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 100})()):
                (root/'state/STOP').write_text('stop')
                with self.assertRaises(RuntimeError): worker.guard(config, db)
                (root/'state/STOP').unlink()
                with self.assertRaises(RuntimeError): worker.guard({**config, 'concurrency': 2}, db)
                (root/'large').write_bytes(b'x'*1001)
                with self.assertRaises(RuntimeError): worker.guard(config, db)
            db.close()

    def test_recovery_checks_shard_hash_before_inventory(self):
        from goal_conditioned_decision_data.pipelines import worker
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            root = Path(directory); (root/'manifests').mkdir(); (root/'data').mkdir()
            (root/'data/sample.jsonl').write_bytes(b'tampered\n')
            (root/'manifests/batch-train-1.json').write_text(json.dumps({'shard': 'data/sample.jsonl', 'sha256': '0'*64}))
            db = sqlite3.connect(':memory:')
            with patch.object(worker, 'ROOT', root):
                with self.assertRaises(ValueError): worker.recover_inventory(db)
            db.close()

    def test_publication_reserves_space_before_writes(self):
        from goal_conditioned_decision_data.pipelines import worker
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            root = Path(directory); (root/'existing').write_bytes(b'x'*900)
            config = {'max_programme_bytes': 1000, 'min_free_bytes': 100}
            with patch.object(worker, 'ROOT', root), patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000)):
                with self.assertRaises(RuntimeError): worker.publication_guard(config, 101)
            with patch.object(worker, 'ROOT', root), patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=200)):
                with self.assertRaises(RuntimeError): worker.publication_guard({**config, 'max_programme_bytes': 2000}, 101)

    def test_metadata_only_recovery_replays_and_rejects_conflicts(self):
        from goal_conditioned_decision_data.pipelines import worker, common
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            root = Path(directory); (root/'manifests').mkdir(); (root/'data').mkdir()
            shard = root/'data/not-json-sealed.bytes'; shard.write_bytes(b'opaque bytes; recovery must not parse labels\n')
            index = ['independent_id', 'independent_semantic', 'independent_parent', 'independent_template', 'sealed']
            manifest = {'shard': 'data/not-json-sealed.bytes', 'sha256': common.file_hash(shard),
                        'seed': 1, 'split': 'sealed', 'accepted': 1, 'quarantined': 0,
                        'protected_index': [index], 'audit': {'verdict': 'PASS'}}
            (root/'manifests/batch-sealed-1.json').write_text(json.dumps(manifest))
            db = sqlite3.connect(':memory:')
            db.execute('CREATE TABLE seen(id TEXT PRIMARY KEY, semantic TEXT, parent TEXT, template TEXT, split TEXT)')
            db.execute('CREATE TABLE jobs(seed INTEGER, split TEXT, status TEXT, accepted INTEGER, quarantined INTEGER, report TEXT, PRIMARY KEY(seed,split))')
            with patch.object(worker, 'ROOT', root):
                worker.recover_inventory(db); worker.recover_inventory(db)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM seen').fetchone()[0], 1)
                self.assertEqual(db.execute('SELECT status,accepted FROM jobs').fetchone(), ('done', 1))
                db.execute('UPDATE seen SET split="train"')
                with self.assertRaises(ValueError): worker.recover_inventory(db)
            db.close()

    def test_watchdog_kills_remaining_descendants_after_parent_exits(self):
        from goal_conditioned_decision_data.pipelines import common
        class TimedOutParent:
            pid = 812345
            calls = 0
            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1: raise subprocess.TimeoutExpired('fake-worker', timeout)
                return 0
        process = TimedOutParent()
        script = Path(common.ROOT)/'scripts/run_cycle.py'
        with patch.object(subprocess, 'Popen', return_value=process) as popen, patch.object(os, 'killpg') as killpg, patch.object(common, 'atomic_json') as persist:
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(script), run_name='__main__')
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertIn(((process.pid, signal.SIGTERM),), [tuple(c)[:1] for c in killpg.call_args_list])
        self.assertIn(((process.pid, signal.SIGKILL),), [tuple(c)[:1] for c in killpg.call_args_list])
        self.assertEqual(persist.call_count, 2)

    def test_backup_branch_and_unrelated_file_guards_precede_credentials(self):
        from goal_conditioned_decision_data.pipelines import common
        namespace = runpy.run_path(str(Path(common.ROOT)/'scripts/backup.py'))
        main_globals = namespace['main'].__globals__
        for branch, status in [('main', ''), ('research/goal-conditioned-decision-data-v1', ' M bwm/owned.py\x00')]:
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=branch if command[:2] == ['git', 'branch'] else status, stderr='')
            with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
                main_globals['ROOT'] = Path(directory)
                main_globals['execute'] = execute
                main_globals['atomic_json'] = Mock()
                self.assertEqual(namespace['main'](), 1)
            self.assertFalse(any(command[0] == 'gh' for command in calls))
            self.assertFalse(any('add' in command or 'push' in command for command in calls))

    def test_backup_errors_redact_known_credentials(self):
        from goal_conditioned_decision_data.pipelines import common
        namespace = runpy.run_path(str(Path(common.ROOT)/'scripts/backup.py'))
        main_globals = namespace['main'].__globals__
        fake_token = 'gcdl_fake_token_only_for_redaction_123'
        def execute(command, **kwargs):
            if command[:2] == ['git', 'branch']:
                return SimpleNamespace(returncode=0, stdout='research/goal-conditioned-decision-data-v1', stderr='')
            if command[0] == 'gh': return SimpleNamespace(returncode=0, stdout=fake_token, stderr='')
            if 'ls-remote' in command: return SimpleNamespace(returncode=0, stdout='a'*40+'\trefs/heads/research/goal-conditioned-decision-data-v1', stderr='')
            if 'push' in command: raise RuntimeError('mock remote accidentally echoed '+fake_token)
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        captured = io.StringIO(); receipt = Mock()
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            main_globals['ROOT'] = Path(directory); main_globals['execute'] = execute; main_globals['atomic_json'] = receipt
            with redirect_stdout(captured), redirect_stderr(captured): self.assertEqual(namespace['main'](), 1)
        self.assertNotIn(fake_token, captured.getvalue())
        self.assertNotIn(fake_token, json.dumps(receipt.call_args.args[1]))

    def test_all_quarantine_sealed_batch_cannot_create_freeze(self):
        from goal_conditioned_decision_data.pipelines import worker
        from goal_conditioned_decision_data.generators import build
        config = json.loads((worker.ROOT/'configs/production.json').read_text())
        config['sealed_seeds'] = [1]
        with tempfile.TemporaryDirectory(prefix='gcdl-independent-') as directory:
            root = Path(directory); (root/'configs').mkdir()
            (root/'configs/production.json').write_text(json.dumps(config))
            captured = io.StringIO()
            with patch.object(worker, 'ROOT', root), patch.object(worker, 'source_hashes', return_value={'fixed': 'hash'}), patch.object(build, 'generate_group', return_value=[{'original_candidate': 'invalid schema'}]), patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*1024**3)), redirect_stderr(captured), redirect_stdout(captured):
                result = worker.run(argparse.Namespace(no_backup=True, seal=True, groups=1))
            self.assertEqual(result, 1)
            self.assertFalse((root/'manifests/SEALED_FREEZE.json').exists())
            self.assertTrue((root/'state/PAUSED.json').exists())


class IndependentIntegrityTests(unittest.TestCase):
    def test_schema_rejects_false_provenance_and_utility_declaration(self):
        from goal_conditioned_decision_data.auditors.integrity import validate_schema
        from goal_conditioned_decision_data.generators.build import record_id
        record = schema_fixture(); validate_schema(record)
        bad = deepcopy(record); bad['provenance']['config_hash'] = '0'*64; bad['id'] = record_id(bad)
        with self.assertRaises(ValueError): validate_schema(bad)
        bad = deepcopy(record); bad['input']['utility_definition']['comparison'] = 'minimize_cost'; bad['id'] = record_id(bad)
        with self.assertRaises(ValueError): validate_schema(bad)

    def test_semantic_fingerprint_preserves_status_semantic_keys(self):
        from goal_conditioned_decision_data.generators.build import semantic_fingerprint
        task = ledger_task()
        renamed = deepcopy(task)
        renamed['rules']['statuses'] = {'pending': 'fresh', 'settled': 'void', 'void': 'withdrawn'}
        renamed['initial_state']['entries']['bill']['status'] = 'fresh'
        self.assertEqual(semantic_fingerprint({'input': task}), semantic_fingerprint({'input': renamed}))

    def test_entity_names_colliding_with_structural_keys_are_normalized(self):
        from goal_conditioned_decision_data.generators.build import semantic_fingerprint
        task = ledger_task(); renamed = deepcopy(task)
        renamed['initial_state'] = {'accounts': {'goal': 10, 'amount': 4},
                                   'entries': {'rules': {'account': 'goal', 'amount': 3, 'status': 'waiting'}}}
        renamed['goal'] = {'balances': {'goal': 13}}
        for candidate in renamed['candidates']:
            for action in candidate['plan']:
                if 'entry' in action: action['entry'] = 'rules'
        self.assertEqual(semantic_fingerprint({'input': task}), semantic_fingerprint({'input': renamed}))

    def test_finite_world_order_does_not_change_semantic_fingerprint(self):
        from goal_conditioned_decision_data.generators.build import semantic_fingerprint
        task = ledger_task(); world_a = deepcopy(task['initial_state']); world_b = deepcopy(world_a)
        world_b['entries']['bill']['amount'] = 4
        task['initial_state']['entries']['bill']['amount'] = None
        task['possible_worlds'] = [world_a, world_b]
        task['information'] = 'ambiguous'; task['probability_assumption'] = 'uniform_over_possible_worlds'
        reverse = deepcopy(task); reverse['possible_worlds'].reverse()
        self.assertEqual(semantic_fingerprint({'input': task}), semantic_fingerprint({'input': reverse}))

    def test_corrupted_uncertainty_is_rejected(self):
        from goal_conditioned_decision_data.oracles.primary import evaluate
        from goal_conditioned_decision_data.generators.build import decision_labels
        from goal_conditioned_decision_data.auditors.integrity import validate_decision
        oracle = evaluate(ledger_task())
        labels = decision_labels(oracle)
        labels['uncertainty']['label'] = 'tied_best'
        with self.assertRaises(ValueError): validate_decision(labels, oracle)
        labels = decision_labels(oracle)
        labels['uncertainty']['abstention'] = 'no_candidate_can_succeed'
        with self.assertRaises(ValueError): validate_decision(labels, oracle)

    def test_proof_scope_and_witness_must_respect_declared_bound(self):
        from goal_conditioned_decision_data.oracles.primary import evaluate
        from goal_conditioned_decision_data.auditors.integrity import validate_auxiliary
        task = ledger_task(); task['goal'] = {'balances': {'oak': 10}}
        task['candidates'] = [{'id': 'zero', 'plan': []}]
        proof = {'scope': 'finite_declared_action_alphabet_and_horizon', 'proof': 'exhaustive_execution',
                 'action_alphabet': [{'op': 'noop'}], 'horizon': 0, 'enumerated_plan_count': 1,
                 'best_utility': evaluate(task)['candidates'][0]['utility'],
                 'solution_exists_in_bound': True, 'witness_plan': [], 'global_unbounded_optimality_claim': False}
        record = {'input': task, 'targets': {'bounded_proof': proof}}
        validate_auxiliary(record)
        for field, value in [('scope', 'globally_optimal'), ('proof', 'model_guess'),
                             ('global_unbounded_optimality_claim', True), ('witness_plan', None),
                             ('witness_plan', [{'op': 'noop'}])]:
            mutated = deepcopy(record); mutated['targets']['bounded_proof'][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError): validate_auxiliary(mutated)

    def test_search_universe_and_available_budget_are_checked(self):
        from goal_conditioned_decision_data.auditors.integrity import validate_auxiliary, search_result
        task = ledger_task()
        task['search_request'] = {'cheap_candidate_budget': 1, 'expanded_candidate_budget': 3,
                                  'same_candidate_universe': True}
        cheap, expanded = search_result(task, 1), search_result(task, 3)
        delta = [exact(b)-exact(a) for a, b in zip(cheap['best_utility'], expanded['best_utility'])]
        search = {'matched_candidate_ids': ['option_x', 'option_y', 'option_z'],
                  'cheap': cheap, 'expanded': expanded,
                  'realized_utility_delta': [{'numerator': x.numerator, 'denominator': x.denominator} for x in delta],
                  'additional_candidate_evaluations': 2,
                  'additional_transition_node_evaluations': expanded['transition_node_evaluations']-cheap['transition_node_evaluations'],
                  'benefit_is_oracle_only': True}
        record = {'input': task, 'targets': {'search': search}}
        validate_auxiliary(record)
        bad = deepcopy(record); bad['targets']['search']['matched_candidate_ids'] = ['invented_candidate']
        with self.assertRaises(ValueError): validate_auxiliary(bad)
        bad = deepcopy(record); bad['input']['search_request']['cheap_candidate_budget'] = 2
        with self.assertRaises(ValueError): validate_auxiliary(bad)
        bad = deepcopy(record); bad['targets']['search']['benefit_is_oracle_only'] = False
        with self.assertRaises(ValueError): validate_auxiliary(bad)

    def test_nested_oracle_leak_and_quarantine_exports_are_rejected(self):
        from goal_conditioned_decision_data.auditors.integrity import recursive_input_check
        from goal_conditioned_decision_data.decision_labels.views import training_views
        with self.assertRaises(ValueError): recursive_input_check({'context': [{'best_candidate_ids': ['x']}]})
        with self.assertRaises(ValueError): training_views({'quality_gate': {'status': 'quarantined'}})


if __name__ == '__main__':
    unittest.main()
