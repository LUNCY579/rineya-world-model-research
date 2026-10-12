"""Semantic and differential properties, independent of generated labels.

This author also wrote both oracles; this suite is engineering evidence, not
independent review. tests/test_adversarial_review.py supplies separate review.
"""
from copy import deepcopy
from itertools import product
import random
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.environments.core import Ledger
from goal_conditioned_decision_data.oracles.primary import evaluate
from goal_conditioned_decision_data.oracles.independent import evaluate as check


def ledger():
    return {'environment': 'ledger', 'initial_state': {'accounts': {'a': 8, 'b': 5},
             'entries': {'e': {'account': 'a', 'amount': 3, 'status': 'queued'}}},
            'rules': {'statuses': {'pending': 'queued', 'settled': 'void', 'void': 'discarded'},
                      'void_effect': 'reverse', 'allow_overdraft': False, 'balance_bounds': [-20, 20],
                      'action_costs': {'settle': 2, 'void': 3, 'transfer': 1, 'noop': 0}},
            'goal': {'balances': {'a': 11}, 'statuses': {'e': 'void'}}, 'constraints': {},
            'candidates': [{'id': 'credit', 'plan': [{'op': 'settle', 'entry': 'e'}]},
                           {'id': 'undo', 'plan': [{'op': 'settle', 'entry': 'e'}, {'op': 'void', 'entry': 'e'}]},
                           {'id': 'stay', 'plan': []}]}


def register(overflow='reject'):
    return {'environment': 'register', 'initial_state': {'registers': {'x': 1, 'y': -1}},
            'rules': {'value_bounds': [-2, 2], 'overflow': overflow,
                      'operations': ['add', 'multiply', 'set', 'copy', 'swap', 'noop'],
                      'action_costs': {'add': 1, 'multiply': 2, 'set': 3, 'copy': 2, 'swap': 2, 'noop': 0}},
            'goal': {'values': {'x': -2}}, 'constraints': {},
            'candidates': [{'id': 'add', 'plan': [{'op': 'add', 'register': 'x', 'value': 2}]},
                           {'id': 'multiply', 'plan': [{'op': 'multiply', 'register': 'y', 'value': 3}]},
                           {'id': 'swap', 'plan': [{'op': 'swap', 'left': 'x', 'right': 'y'}]}]}


def block_task(on):
    return {'environment': 'blocksworld', 'initial_state': {'on': on},
            'rules': {'table': 'ground', 'max_height': 3, 'action_costs': {'move': 2, 'noop': 0}},
            'goal': {'on': {'a': 'b', 'b': 'ground'}}, 'constraints': {},
            'candidates': [{'id': 'empty', 'plan': []}]}


class OracleProperties(unittest.TestCase):
    def same(self, task):
        before = deepcopy(task)
        primary, secondary = evaluate(task), check(task)
        self.assertEqual(primary, secondary)
        self.assertEqual(task, before, 'oracle mutated input')
        for candidate in primary['candidates']:
            for outcome in candidate['outcomes']:
                successes = [step for step in outcome['steps'] if step['legal']]
                self.assertEqual(outcome['cost'], sum(step['cost'] for step in successes))
                self.assertEqual(len(outcome['trajectory']), len(successes) + 1)
                self.assertEqual(outcome['terminal_state'], outcome['trajectory'][-1])
                self.assertEqual(outcome['legal'], outcome['failure_step'] is None)
                self.assertEqual(outcome['utility'][0], int(outcome['legal'] and not outcome['constraint_violations']))
                if not outcome['legal']:
                    self.assertFalse(outcome['steps'][-1]['legal'])
                    self.assertEqual(outcome['steps'][-1]['cost'], 0)
                    self.assertEqual(outcome['steps'][-1]['state'], outcome['terminal_state'])
        return primary

    def test_random_ledger_signed_entries_rules_horizons(self):
        rng = random.Random(81313)
        actions = [{'op': 'settle', 'entry': 'e'}, {'op': 'void', 'entry': 'e'}, {'op': 'noop'},
                   {'op': 'transfer', 'source': 'a', 'destination': 'b', 'amount': 3},
                   {'op': 'transfer', 'source': 'b', 'destination': 'a', 'amount': 9},
                   {'op': 'settle', 'entry': 'missing'}, {'op': 'unknown'}]
        for case in range(180):
            task = ledger()
            task['initial_state']['accounts'] = {'a': rng.randint(0, 20), 'b': rng.randint(0, 20)}
            entry = task['initial_state']['entries']['e']
            entry['amount'] = rng.choice([i for i in range(-21, 22) if i])
            entry['status'] = rng.choice(['queued', 'void', 'discarded'])
            task['rules']['void_effect'] = rng.choice(['reverse', 'retain'])
            task['rules']['allow_overdraft'] = bool(case % 2)
            task['constraints'] = {'minimum_balances': {'a': rng.randint(-5, 22)}, 'max_cost': rng.randint(0, 12),
                                   'forbidden_statuses': rng.sample(['queued', 'void', 'discarded'], rng.randint(0, 3))}
            task['candidates'] = [{'id': 'p' + str(i), 'plan': rng.choices(actions, k=rng.randint(0, 8))} for i in range(8)]
            with self.subTest(case=case): self.same(task)

    def test_exhaustive_three_block_forests_and_two_step_actions(self):
        names = ['a', 'b', 'c']
        actions = [{'op': 'move', 'block': a, 'destination': b} for a in names for b in ['ground'] + names]
        valid_states = 0
        for assignments in product(['ground'] + names, repeat=3):
            task = block_task(dict(zip(names, assignments)))
            try: evaluate(task)
            except ValueError: continue
            valid_states += 1
            task['candidates'] = [{'id': str(i), 'plan': list(plan)} for i, plan in enumerate(product(actions, repeat=2))]
            task['constraints'] = {'forbidden_on': [['a', 'c']], 'protected_blocks': ['b'], 'max_cost': 3}
            self.same(task)
        self.assertEqual(valid_states, 13)

    def test_register_overflow_expected_endpoints(self):
        reject = self.same(register())['candidates']
        self.assertFalse(reject[0]['outcomes'][0]['legal'])
        clamp = self.same(register('clamp'))['candidates']
        self.assertEqual(clamp[0]['outcomes'][0]['terminal_state']['registers']['x'], 2)
        self.assertEqual(clamp[1]['outcomes'][0]['terminal_state']['registers']['y'], -2)
        wrap = self.same(register('wrap'))['candidates']
        self.assertEqual(wrap[0]['outcomes'][0]['terminal_state']['registers']['x'], -2)
        self.assertEqual(wrap[1]['outcomes'][0]['terminal_state']['registers']['y'], 2)

    def test_random_register_values_large_integer_operands(self):
        rng = random.Random(661)
        for case in range(120):
            task = register(rng.choice(['reject', 'wrap', 'clamp']))
            task['initial_state']['registers'] = {'x': rng.randint(-2, 2), 'y': rng.randint(-2, 2)}
            task['constraints'] = {'minimum_values': {'x': -1}, 'maximum_values': {'y': 1}, 'protected_registers': ['y'], 'max_cost': 5}
            actions = [{'op': 'add', 'register': 'x', 'value': rng.randint(-100000, 100000)},
                       {'op': 'multiply', 'register': 'y', 'value': rng.randint(-50, 50)},
                       {'op': 'set', 'register': 'x', 'value': rng.randint(-9, 9)},
                       {'op': 'copy', 'source': 'x', 'destination': 'y'},
                       {'op': 'swap', 'left': 'x', 'right': 'y'}, {'op': 'noop'}]
            task['candidates'] = [{'id': str(i), 'plan': rng.choices(actions, k=rng.randint(0, 7))} for i in range(8)]
            with self.subTest(case=case): self.same(task)

    def test_goal_changes_cannot_change_transition(self):
        task = ledger()
        a = self.same(task)
        task['goal'] = {'balances': {'b': 5}, 'statuses': {'e': 'discarded'}}
        b = self.same(task)
        for old, new in zip(a['candidates'], b['candidates']):
            for first, second in zip(old['outcomes'], new['outcomes']):
                for field in ['steps', 'trajectory', 'terminal_state', 'cost', 'legal', 'constraint_violations']:
                    self.assertEqual(first[field], second[field])

    def test_unknown_support_is_exact_and_observation_enforced(self):
        task = ledger(); base = deepcopy(task['initial_state'])
        task['initial_state']['entries']['e']['amount'] = None
        task['information'] = 'missing'; task['probability_assumption'] = 'uniform_over_possible_worlds'
        task['possible_worlds'] = [deepcopy(base) for _ in range(3)]
        for world, amount in zip(task['possible_worlds'], [2, 3, 4]): world['entries']['e']['amount'] = amount
        result = self.same(task)
        self.assertEqual(result['candidates'][0]['success_probability'], {'numerator': 1, 'denominator': 3})
        self.assertEqual(result['candidates'][0]['expected_goal_score'], {'numerator': 4, 'denominator': 3})
        for invalid in ['duplicate', 'contradiction', 'assumption']:
            altered = deepcopy(task)
            if invalid == 'duplicate': altered['possible_worlds'][1] = deepcopy(altered['possible_worlds'][0])
            if invalid == 'contradiction': altered['possible_worlds'][0]['accounts']['a'] = 7
            if invalid == 'assumption': altered['probability_assumption'] = 'model_confidence'
            for oracle in [evaluate, check]:
                with self.assertRaises(ValueError): oracle(altered)

    def test_strict_malformed_rule_state_goal_and_action_rejection(self):
        mutations = [lambda t: t['rules'].update({'hidden_bonus': 1}),
                     lambda t: t['rules']['action_costs'].update({'void': True}),
                     lambda t: t['initial_state']['accounts'].update({'a': True}),
                     lambda t: t['goal'].update({'unexecuted_instruction': 'good'}),
                     lambda t: t['goal']['balances'].update({'unknown': 2}),
                     lambda t: t['constraints'].update({'maximum_balances': {'a': 4}}),
                     lambda t: t['candidates'][0]['plan'][0].update({'explanation': 'pretend legal'}),
                     lambda t: t.update({'utility_definition': {'comparison': 'minimize_cost'}})]
        for mutate in mutations:
            task = ledger(); mutate(task)
            for oracle in [evaluate, check]:
                with self.subTest(mutation=mutate, oracle=oracle.__module__):
                    with self.assertRaises(ValueError): oracle(task)

    def test_deliberately_wrong_signed_transition_detected(self):
        class WrongLedger(Ledger):
            def transition(self, state, action, rules):
                result, error = super().transition(state, action, rules)
                if error is None and action['op'] == 'settle':
                    entry = state['entries'][action['entry']]
                    result['accounts'][entry['account']] -= 2 * entry['amount']
                return result, error
        expected = check(ledger())
        with patch('goal_conditioned_decision_data.oracles.primary.get_environment', return_value=WrongLedger()):
            self.assertNotEqual(evaluate(ledger()), expected)

    def test_deliberately_wrong_scoring_and_cost_detected(self):
        class WrongScore(Ledger):
            def score(self, state, goal): return 10, True
        expected = check(ledger())
        with patch('goal_conditioned_decision_data.oracles.primary.get_environment', return_value=WrongScore()):
            self.assertNotEqual(evaluate(ledger()), expected)
        altered = deepcopy(expected)
        altered['candidates'][0]['outcomes'][0]['cost'] += 1
        self.assertNotEqual(evaluate(ledger()), altered)


if __name__ == '__main__': unittest.main()
