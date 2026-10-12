"""Reproducible, protected-template curriculum for GCDL-001.

All labels originate in an executable oracle. This module never reads accepted
shards or sealed targets. Re-evaluation and acceptance belong to the auditor.
"""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import random

GENERATOR_VERSION = 'generator-1.0'
SCHEMA_VERSION = 'gcdl-1.0'
ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = sorted(['settle', 'noop', 'transfer_out', 'invalid', 'credit_then_reverse',
    'credit_only', 'neutral', 'harmful', 'cheap_credit', 'costly_credit',
    'constructed_path', 'partial_path', 'reversed_path', 'random_alternative',
    'void', 'no_change', 'noop_a', 'noop_b', 'credit_then_transfer', 'add', 'subtract'])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def curriculum(config=None):
    if config is None:
        return json.loads((ROOT / 'configs' / 'curriculum.json').read_text())
    # Production budget config is separate; only a curriculum override changes tasks.
    if 'split_templates' not in config:
        return curriculum()
    return deepcopy(config)


def record_id(record):
    """Stable identity excludes audit decisions and runtime measurements."""
    clean = deepcopy(record)
    for key in ('id', 'verification', 'quality_gate', 'quality', 'runtime', 'audit'):
        clean.pop(key, None)
    metadata = clean.get('metadata', {})
    for key in ('runtime_measurements', 'timing', 'source_hashes', 'code_hashes'):
        metadata.pop(key, None)
    return 'gcdl_' + _hash(clean)[:32]


def protected_template_signature(inputs):
    state = inputs.get('initial_state', inputs.get('possible_worlds', [{}])[0])
    if inputs['environment'] == 'ledger':
        size = [len(state['accounts']), len(state['entries'])]
    elif inputs['environment'] == 'blocksworld':
        size = [len(state['on'])]
    elif inputs['environment'] == 'register':
        size = [len(state['registers'])]
    else:
        size = [len(state)]
    return _hash({'environment': inputs['environment'], 'world_size': size,
                  'rule_keys': sorted(inputs['rules']),
                  'constraint_keys': sorted(inputs.get('constraints', {}))})


def _rename_tree(value, mapping):
    if isinstance(value, dict):
        return {mapping.get(k, k): _rename_tree(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename_tree(v, mapping) for v in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _normalize_task(inputs, first_map, second_map=None):
    """Rename only entity/status namespaces, never structural/semantic keys."""
    # JSON round-trip removes Python object aliasing between shared candidate
    # action dictionaries, matching the persisted canonical record semantics.
    task = json.loads(_canonical(inputs))
    env = task['environment']
    if env == 'ledger':
        account_map, entry_map = first_map, second_map
        status_map = {surface: semantic for semantic, surface in task['rules']['statuses'].items()}
        def state(value):
            return {'accounts': {account_map.get(k, k): v for k, v in value['accounts'].items()},
                    'entries': {entry_map.get(k, k): dict(v, account=account_map.get(v['account'], v['account']),
                                 status=status_map.get(v['status'], v['status'])) for k, v in value['entries'].items()}}
        task['rules']['statuses'] = {key: key for key in ('pending', 'settled', 'void')}
        if 'balances' in task['goal']:
            task['goal']['balances'] = {account_map.get(k, k): v for k, v in task['goal']['balances'].items()}
        if 'statuses' in task['goal']:
            task['goal']['statuses'] = {entry_map.get(k, k): status_map.get(v, v) for k, v in task['goal']['statuses'].items()}
        if 'minimum_balances' in task['constraints']:
            task['constraints']['minimum_balances'] = {account_map.get(k, k): v for k, v in task['constraints']['minimum_balances'].items()}
        if 'forbidden_statuses' in task['constraints']:
            task['constraints']['forbidden_statuses'] = [status_map.get(v, v) for v in task['constraints']['forbidden_statuses']]
        for c in task['candidates']:
            for action in c['plan']:
                if action['op'] in ('settle', 'void'):
                    action['entry'] = entry_map.get(action['entry'], {'unknown_entry': action['entry']})
                elif action['op'] == 'transfer':
                    for key in ('source', 'destination'):
                        action[key] = account_map.get(action[key], {'unknown_account': action[key]})
    elif env == 'blocksworld':
        block_map = first_map
        table = task['rules']['table']
        def support(value):
            if value == table:
                return '__table__'
            if value is None:
                return None
            return block_map.get(value, {'unknown_block': value})
        def state(value):
            return {'on': {block_map.get(k, k): support(v) for k, v in value['on'].items()}}
        task['rules']['table'] = '__table__'
        task['goal']['on'] = {block_map.get(k, k): support(v) for k, v in task['goal']['on'].items()}
        if 'forbidden_on' in task['constraints']:
            task['constraints']['forbidden_on'] = [[block_map.get(a, a), support(b)] for a, b in task['constraints']['forbidden_on']]
        if 'protected_blocks' in task['constraints']:
            task['constraints']['protected_blocks'] = [block_map.get(v, v) for v in task['constraints']['protected_blocks']]
        for c in task['candidates']:
            for action in c['plan']:
                if action['op'] == 'move':
                    action['block'] = block_map.get(action['block'], {'unknown_block': action['block']})
                    action['destination'] = support(action['destination'])
    elif env == 'register':
        register_map = first_map
        def state(value):
            return {'registers': {register_map.get(k, k): v for k, v in value['registers'].items()}}
        task['goal']['values'] = {register_map.get(k, k): v for k, v in task['goal']['values'].items()}
        for key in ('minimum_values', 'maximum_values'):
            if key in task['constraints']:
                task['constraints'][key] = {register_map.get(k, k): v for k, v in task['constraints'][key].items()}
        if 'protected_registers' in task['constraints']:
            task['constraints']['protected_registers'] = [register_map.get(v, v) for v in task['constraints']['protected_registers']]
        for c in task['candidates']:
            for action in c['plan']:
                for key in ('register', 'source', 'destination', 'left', 'right'):
                    if key in action:
                        action[key] = register_map.get(action[key], {'unknown_register': action[key]})
    else:
        return task
    if 'initial_state' in task:
        task['initial_state'] = state(task['initial_state'])
    if 'possible_worlds' in task:
        task['possible_worlds'] = sorted((state(w) for w in task['possible_worlds']), key=_canonical)
    task['candidates'] = sorted(task['candidates'], key=_canonical)
    return task


def semantic_fingerprint(record):
    """Bounded entity/status namespace normalization for semantic duplicates.

    This ignores instruction prose, option IDs/order and entity spelling. It
    preserves actual rules, supplied possible worlds and candidate actions.
    The declared curriculum bounds entities to six blocks/four accounts and
    entries; arbitrary environment logical equivalence is not claimed.
    """
    inputs = deepcopy(record['input'])
    for key in ('instruction', 'context', 'observations', 'search_request'):
        inputs.pop(key, None)
    for candidate in inputs['candidates']:
        candidate.pop('id', None)
    state = inputs.get('initial_state', inputs.get('possible_worlds', [{}])[0])
    env = inputs['environment']
    choices = []
    if env == 'ledger':
        for aa in permutations(state['accounts']):
            account_map = {name: 'account_' + str(i) for i, name in enumerate(aa)}
            for ee in permutations(state['entries']):
                entry_map = {name: 'entry_' + str(i) for i, name in enumerate(ee)}
                choices.append(_canonical(_normalize_task(inputs, account_map, entry_map)))
    elif env in ('blocksworld', 'register'):
        names = state['on'] if env == 'blocksworld' else state['registers']
        prefix = 'block_' if env == 'blocksworld' else 'register_'
        for order in permutations(names):
            choices.append(_canonical(_normalize_task(inputs, {name: prefix + str(i) for i, name in enumerate(order)})))
    else:
        inputs['candidates'] = sorted(inputs['candidates'], key=_canonical)
        choices.append(_canonical(inputs))
    return hashlib.sha256(min(choices).encode()).hexdigest()


def _fraction(value):
    if isinstance(value, dict):
        return Fraction(value['numerator'], value['denominator'])
    return Fraction(value)


def _rational(value):
    value = Fraction(value)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def _oracle(inputs):
    from ..oracles.primary import evaluate
    return evaluate(inputs)


def decision_labels(oracle):
    """Project exact authoritative outputs; no heuristic confidence labels."""
    items = oracle['candidates']
    projected = []
    for candidate in items:
        projected.append({'id': candidate['id'], 'utility': deepcopy(candidate['utility']),
                          'success_probability': deepcopy(candidate['success_probability']),
                          'feasible_probability': deepcopy(candidate['feasible_probability'])})
    def score(item):
        return tuple(_fraction(v) for v in item['utility'])
    levels = sorted({score(x) for x in projected}, reverse=True)
    ranking = [{'rank': i + 1, 'candidate_ids': sorted(x['id'] for x in projected if score(x) == level)}
               for i, level in enumerate(levels)]
    all_infeasible = levels[0][0] == 0
    best = [] if all_infeasible else ranking[0]['candidate_ids']
    success = [_fraction(x['success_probability']) for x in projected]
    any_success = any(p > 0 for p in success)
    uncertainty = ('no_feasible_candidate' if all_infeasible else
                   'impossible_in_candidate_set' if not any_success else
                   'tied_best' if len(best) > 1 else
                   'partially_observed' if oracle.get('world_count', 1) > 1 else 'determined')
    return {'candidates': projected, 'best_candidate_ids': best, 'ranking': ranking,
            'optimality': {'scope': 'supplied_candidates', 'proof': 'exhaustive_candidate_execution'},
            'uncertainty': {'label': uncertainty,
                            'abstention': 'no_feasible_candidate' if all_infeasible else
                            'no_candidate_can_succeed' if not any_success else 'not_required',
                            'probability_definition': 'Exact uniform fraction of supplied possible worlds in which legal execution satisfies every goal atom and all constraints.'}}


def _record(seed, split, cfg, family, difficulty, inputs, index, parent_id, tags=None, perturbation=None):
    from ..environments.core import ENVIRONMENT_VERSION
    inputs = deepcopy(inputs)
    id_rng = random.Random(int(_hash({'seed': seed, 'parent_world_id': parent_id})[:16], 16))
    role_ids = {role: 'c_' + format(id_rng.getrandbits(96), '024x') for role in ROLE_NAMES}
    aliases = {}
    for candidate in inputs['candidates']:
        role = candidate['id']
        candidate['id'] = role_ids[role]
        aliases[candidate['id']] = role
    inputs.setdefault('utility_definition', {'comparison': 'lexicographic',
        'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
        'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
        'all_infeasible': 'abstain; no best candidate'})
    oracle = _oracle(inputs)
    result = {'schema_version': SCHEMA_VERSION, 'family': family, 'difficulty': difficulty,
              'split': split, 'provenance': {'seed': seed, 'generator_version': GENERATOR_VERSION,
                  'environment_version': ENVIRONMENT_VERSION, 'config_version': cfg['version'],
                  'config_hash': _hash(cfg), 'template_id': cfg['split_templates'][split]['template'],
                  'template_group_id': cfg['split_templates'][split]['template'],
                  'parent_world_id': parent_id, 'lineage_id': cfg['split_templates'][split]['template'],
                  'record_index': index},
              'input': deepcopy(inputs), 'targets': {'oracle': oracle, 'decision': decision_labels(oracle)},
              'metadata': {'tags': list(tags or []), 'counterfactual_group_id': parent_id,
                  'protected_template_signature': protected_template_signature(inputs),
                  'candidate_roles': aliases,
                  'perturbation': perturbation or {'kind': 'none', 'label_relation': 'base'}}}
    result['id'] = record_id(result)
    return result


def _constraints(spec, a, status, balance):
    composition = spec['constraint_composition']
    if composition == 'none':
        return {}
    if composition == 'minimum':
        return {'minimum_balances': {a: balance}}
    if composition == 'cost':
        return {'max_cost': 1}
    return {'minimum_balances': {a: balance}, 'max_cost': 1,
            'forbidden_statuses': [status['void']]}


def _ledger(rng, spec):
    magnitude = rng.randint(spec['magnitude_min'], spec['magnitude_max'])
    accounts = {'a': rng.randint(magnitude + 1, magnitude * 9 + 3),
                'b': rng.randint(magnitude + 1, magnitude * 11 + 7)}
    for i in range(2, spec['accounts']):
        accounts['account' + str(i)] = magnitude * (i + 1) + rng.randint(0, 5)
    entries = {'e': {'account': 'a', 'amount': magnitude, 'status': 'pending'}}
    for i in range(1, spec['entries']):
        entries['entry' + str(i)] = {'account': list(accounts)[i % len(accounts)],
                                   'amount': -rng.randint(1, magnitude * 2) - i, 'status': 'pending'}
    rules = {'statuses': {'pending': 'pending', 'settled': 'settled', 'void': 'void'},
             'void_effect': 'reverse', 'allow_overdraft': False,
             'balance_bounds': [-magnitude * 100, magnitude * 100],
             'action_costs': {'settle': 1, 'void': 2, 'transfer': 1, 'noop': 0}}
    return {'environment': 'ledger', 'initial_state': {'accounts': accounts, 'entries': entries},
            'rules': rules, 'goal': {'balances': {'a': accounts['a'] + magnitude}},
            'constraints': _constraints(spec, 'a', rules['statuses'], accounts['a']),
            'candidates': [{'id': 'settle', 'plan': [{'op': 'settle', 'entry': 'e'}]},
                           {'id': 'noop', 'plan': [{'op': 'noop'}]},
                           {'id': 'transfer_out', 'plan': [{'op': 'transfer', 'source': 'a', 'destination': 'b', 'amount': magnitude}]},
                           {'id': 'invalid', 'plan': [{'op': 'settle', 'entry': 'absent'}]}],
            'instruction': 'Choose the plan with lexicographically highest feasible goal utility; preserve ties.'}


def _rotate(record, slot):
    """Move sole best candidate to requested slot without altering any semantics."""
    best = record['targets']['decision']['best_candidate_ids']
    candidates = record['input']['candidates']
    if len(best) == 1:
        current = next(i for i, c in enumerate(candidates) if c['id'] == best[0])
        shift = (current - slot) % len(candidates)
        record['input']['candidates'] = candidates[shift:] + candidates[:shift]
        record['targets']['oracle'] = _oracle(record['input'])
        record['targets']['decision'] = decision_labels(record['targets']['oracle'])
    record['id'] = record_id(record)
    return record


def _search(inputs, budgets):
    """Run matched searches on the SAME supplied candidate universe."""
    runs = []
    for budget in budgets:
        chosen = deepcopy(inputs)
        chosen['candidates'] = chosen['candidates'][:budget]
        oracle = _oracle(chosen)
        labels = decision_labels(oracle)
        best_ids = labels['best_candidate_ids']
        top_id = best_ids[0] if best_ids else labels['ranking'][0]['candidate_ids'][0]
        best = next(c for c in labels['candidates'] if c['id'] == top_id)
        nodes = sum(len(outcome['steps']) for candidate in oracle['candidates'] for outcome in candidate['outcomes'])
        runs.append({'candidate_budget': len(chosen['candidates']),
                     'candidate_evaluations': len(chosen['candidates']),
                     'transition_node_evaluations': nodes,
                     'best_candidate_ids': best_ids, 'best_utility': best['utility']})
    cheap, expanded = runs
    delta = [_rational(_fraction(b) - _fraction(a)) for a, b in zip(cheap['best_utility'], expanded['best_utility'])]
    return {'matched_candidate_ids': [c['id'] for c in inputs['candidates']],
            'cheap': cheap, 'expanded': expanded, 'realized_utility_delta': delta,
            'additional_candidate_evaluations': expanded['candidate_evaluations'] - cheap['candidate_evaluations'],
            'additional_transition_node_evaluations': expanded['transition_node_evaluations'] - cheap['transition_node_evaluations'],
            'benefit_is_oracle_only': True,
            'cost_definition': 'Number of actual candidate executions and attempted transition nodes; wall-clock timings are pipeline runtime evidence, excluded from reproducible records.'}


def _bounded_proof(inputs, actions, horizon):
    plans = [{'id': 'proof_' + str(i), 'plan': list(plan)} for i, plan in
             enumerate(plan for h in range(horizon + 1) for plan in product(actions, repeat=h))]
    task = deepcopy(inputs)
    task['candidates'] = plans
    oracle = _oracle(task)
    labels = decision_labels(oracle)
    successful = [c['id'] for c in labels['candidates'] if _fraction(c['success_probability']) == 1]
    return {'scope': 'finite_declared_action_alphabet_and_horizon', 'proof': 'exhaustive_execution',
            'action_alphabet': deepcopy(actions), 'horizon': horizon,
            'enumerated_plan_count': len(plans), 'best_utility': next(c['utility'] for c in labels['candidates'] if c['id'] in labels['ranking'][0]['candidate_ids']),
            'solution_exists_in_bound': bool(successful),
            'witness_plan': next((c['plan'] for c in plans if c['id'] in successful), None),
            'global_unbounded_optimality_claim': False}


def generate_group(seed, config=None, split='train'):
    """Generate related A-H candidate records; acceptance is a separate stage."""
    cfg = curriculum(config)
    if split not in cfg['split_templates']:
        raise ValueError('split must be train, dev, sealed, or ood')
    rng = random.Random(seed)
    spec = cfg['split_templates'][split]
    base = _ledger(rng, spec)
    parent = 'world_' + _hash({'seed': seed, 'template': spec['template'], 'state': base['initial_state'], 'rules': base['rules']})[:24]
    records = []
    def emit(family, difficulty, inputs, tags=(), perturbation=None):
        record = _record(seed, split, cfg, family, difficulty, inputs, len(records), parent,
                         tags, perturbation)
        records.append(record)
        return record
    balance = base['initial_state']['accounts']['a']
    amount = base['initial_state']['entries']['e']['amount']
    # A: identical transition problem, three genuinely different goals.
    for goal in ({'balances': {'a': balance + amount}}, {'balances': {'a': balance}},
                 {'balances': {'b': base['initial_state']['accounts']['b'] + amount}}):
        task = deepcopy(base)
        task['goal'] = goal
        emit('A', 'easy', task, ['same_world_multi_goal', 'goal_change_only'])
    # B: the same initial credit looks useful but an explicit later reversal
    # removes it; the oracle records every intermediate state.
    delayed = deepcopy(base)
    delayed['candidates'] = [
        {'id': 'credit_then_reverse', 'plan': [{'op': 'settle', 'entry': 'e'}, {'op': 'void', 'entry': 'e'}]},
        {'id': 'credit_only', 'plan': [{'op': 'settle', 'entry': 'e'}]},
        {'id': 'neutral', 'plan': [{'op': 'noop'}]},
        {'id': 'harmful', 'plan': [{'op': 'transfer', 'source': 'a', 'destination': 'b', 'amount': amount}]}]
    emit('B', 'medium', delayed, ['delayed_negative_consequence', 'clear_cut'])
    close = deepcopy(base)
    close['candidates'] = [{'id': 'cheap_credit', 'plan': [{'op': 'settle', 'entry': 'e'}]},
                           {'id': 'costly_credit', 'plan': [{'op': 'noop'}, {'op': 'settle', 'entry': 'e'}]}]
    close['rules']['action_costs']['noop'] = 1
    if 'max_cost' in close['constraints']:
        close['constraints']['max_cost'] = 3
    emit('B', 'medium', close, ['close_call', 'same_goal_score_one_cost_gap'])
    # D: explicit semantics-preserving surfaces plus rule-changing control.
    for kind, mapping in [('entity_rename', {'a': 'cedar', 'b': 'river', 'e': 'tally'}),
                          ('status_vocabulary', {'pending': 'awaiting', 'settled': 'recorded', 'void': 'withdrawn'})]:
        task = deepcopy(base) if kind == 'status_vocabulary' else _rename_tree(deepcopy(base), mapping)
        # Status surfaces are a separate namespace from action op/rule keys.
        if kind == 'status_vocabulary':
            task['rules']['statuses'] = {k: mapping[v] for k, v in base['rules']['statuses'].items()}
            for entry in task['initial_state']['entries'].values():
                entry['status'] = mapping[entry['status']]
            if 'statuses' in task['goal']:
                task['goal']['statuses'] = {k: mapping[v] for k, v in task['goal']['statuses'].items()}
            if 'forbidden_statuses' in task['constraints']:
                task['constraints']['forbidden_statuses'] = [mapping[v] for v in task['constraints']['forbidden_statuses']]
        emit('D', 'adversarial', task, [kind], {'kind': kind, 'label_relation': 'invariant', 'base_record_id': records[0]['id']})
    ordered = deepcopy(base)
    ordered['candidates'] = list(reversed(ordered['candidates']))
    ordered = dict(reversed(list(ordered.items())))
    ordered['instruction'] = 'Compare the consequences of the listed plans using the stated goals and constraints.'
    ordered['context'] = 'Irrelevant bulletin: the painted wall is blue. This changes no rule or balance.'
    emit('D', 'adversarial', ordered, ['candidate_order', 'field_order', 'paraphrase', 'distractor'],
         {'kind': 'order_paraphrase_distractor', 'label_relation': 'invariant', 'base_record_id': records[0]['id']})
    reversal = deepcopy(delayed)
    reversal['rules']['void_effect'] = 'retain'
    emit('D', 'adversarial', reversal, ['rule_reversal'],
         {'kind': 'void_rule_reversal', 'label_relation': 'recompute', 'base_record_id': records[3]['id']})
    # An initially settled credit makes the governing void rule determine
    # the actual best choice, subject to explicitly supplied constraints.
    settled = deepcopy(base)
    settled['initial_state']['accounts']['a'] = balance + amount
    settled['initial_state']['entries']['e']['status'] = 'settled'
    settled['goal'] = {'balances': {'a': balance}}
    settled['candidates'] = [{'id': 'void', 'plan': [{'op': 'void', 'entry': 'e'}]},
                             {'id': 'noop', 'plan': [{'op': 'noop'}]}]
    flip_base = emit('D', 'adversarial', settled, ['governing_rule_decision_control'],
                     {'kind': 'settled_rule_sensitivity_base', 'label_relation': 'base'})
    flip_base['provenance']['parent_world_id'] = parent + '_settled'
    flip_base['metadata']['counterfactual_group_id'] = parent + '_settled'
    flip_base['id'] = record_id(flip_base)
    flip_task = deepcopy(settled)
    flip_task['rules']['void_effect'] = 'retain'
    flip = emit('D', 'adversarial', flip_task, ['governing_rule_decision_reversal'],
                {'kind': 'settled_rule_sensitivity_reversal', 'label_relation': 'recompute',
                 'base_record_id': flip_base['id']})
    flip['provenance']['parent_world_id'] = parent + '_settled'
    flip['metadata']['counterfactual_group_id'] = parent + '_settled'
    flip['metadata']['rule_sensitivity'] = {'decision_flip_expected': spec['constraint_composition'] in ('none', 'minimum'),
        'constraint_blocked_in_other_templates': True}
    flip['id'] = record_id(flip)
    # C: random valid initial/goal forests and actual candidate paths. Entity
    # permutations alone do not count as task diversity.
    blocks = ['block' + str(i) for i in range(spec['blocks'])]
    on = {b: 'table' for b in blocks}
    def propose_moves(state):
        clear = [b for b in blocks if b not in state.values()]
        actions = []
        for source in clear:
            for destination in ['table'] + clear:
                if source != destination and state[source] != destination:
                    actions.append({'op': 'move', 'block': source, 'destination': destination})
        return actions
    for _ in range(rng.randint(0, len(blocks))):
        action = rng.choice(propose_moves(on))
        on[action['block']] = action['destination']
    terminal = deepcopy(on)
    stack_plan = []
    for _ in range(spec['horizon']):
        action = rng.choice(propose_moves(terminal))
        terminal[action['block']] = action['destination']
        stack_plan.append(action)
    if terminal == on:
        terminal = deepcopy(on)
        for action in stack_plan[:-1]:
            terminal[action['block']] = action['destination']
        stack_plan[-1] = {'op': 'noop'}
    alternate_state = deepcopy(on)
    alternate = []
    for _ in range(spec['horizon']):
        action = rng.choice(propose_moves(alternate_state))
        alternate_state[action['block']] = action['destination']
        alternate.append(action)
    bw = {'environment': 'blocksworld', 'initial_state': {'on': on},
          'rules': {'table': 'table', 'max_height': len(blocks),
                    'action_costs': {'move': rng.randint(1, 9), 'noop': rng.randint(0, 5)}},
          'goal': {'on': terminal}, 'constraints': {},
          'candidates': [{'id': 'constructed_path', 'plan': stack_plan},
                         {'id': 'partial_path', 'plan': stack_plan[:-1]},
                         {'id': 'reversed_path', 'plan': list(reversed(stack_plan))},
                         {'id': 'random_alternative', 'plan': alternate},
                         {'id': 'no_change', 'plan': []}],
          'instruction': 'Execute plans under the stated Blocksworld rules and compare feasible goal utility.'}
    c = emit('C', 'hard', bw, ['multi_step_planning', 'horizon_' + str(len(stack_plan))])
    c['provenance']['parent_world_id'] = parent + '_blocks'
    c['metadata']['counterfactual_group_id'] = parent + '_blocks'
    c['id'] = record_id(c)
    if split == 'ood':
        task = deepcopy(bw)
        task['candidates'][0]['plan'] = [{'op': 'noop'}] * 2 + stack_plan
        ood = emit('D', 'adversarial', task, ['ood_size', 'ood_horizon'],
                   {'kind': 'ood_structural', 'label_relation': 'recompute',
                    'training_limits': {'ledger_magnitude_max': cfg['split_templates']['train']['magnitude_max'],
                                        'blocks': cfg['split_templates']['train']['blocks'], 'horizon': cfg['split_templates']['train']['horizon']}})
        ood['provenance']['parent_world_id'] = parent + '_blocks'
        ood['metadata']['counterfactual_group_id'] = parent + '_blocks'
        ood['id'] = record_id(ood)
    if split == 'ood':
        emit('D', 'adversarial', deepcopy(base), ['ood_magnitude', 'ood_size', 'heldout_rule_conjunction'],
             {'kind': 'ood_numeric_and_ledger_size', 'label_relation': 'recompute',
              'observed_entry_amount': amount,
              'training_entry_amount_max': cfg['split_templates']['train']['magnitude_max'],
              'world_size': [spec['accounts'], spec['entries']],
              'training_world_size': [cfg['split_templates']['train']['accounts'], cfg['split_templates']['train']['entries']]})
    negative = deepcopy(base)
    negative['initial_state']['entries']['e']['amount'] = -amount
    negative['goal'] = {'balances': {'a': balance - amount}}
    emit('D', 'adversarial', negative, ['negative_entry_amount', 'goal_reversal'],
         {'kind': 'signed_amount_and_goal_reversal', 'label_relation': 'recompute'})
    # E: exact finite-world uncertainty, ties, impossible candidate goal.
    uncertain = deepcopy(base)
    state1 = deepcopy(uncertain['initial_state'])
    state2 = deepcopy(state1)
    state2['entries']['e']['amount'] = amount + 1
    uncertain['initial_state']['entries']['e']['amount'] = None
    uncertain['possible_worlds'] = [state1, state2]
    uncertain['information'] = 'missing'
    uncertain['probability_assumption'] = 'uniform_over_possible_worlds'
    uncertain['observations'] = {'missing': ['entries.e.amount'], 'known_possible_values': [amount, amount + 1],
                                'prior_definition': 'uniform conditional on the two explicitly supplied finite worlds'}
    emit('E', 'medium', uncertain, ['missing_information', 'exact_finite_world_probability'])
    ambiguous = deepcopy(uncertain)
    state3 = deepcopy(state1)
    state3['entries']['e']['amount'] = amount + 2
    ambiguous['possible_worlds'].append(state3)
    ambiguous['information'] = 'ambiguous'
    ambiguous['observations']['known_possible_values'] = [amount, amount + 1, amount + 2]
    ambiguous['observations']['prior_definition'] = 'uniform conditional on three declared possible worlds'
    emit('E', 'medium', ambiguous, ['ambiguous_outcome', 'uniform_three_world_probability'])
    tie = deepcopy(base)
    tie['goal'] = {'balances': {'a': balance}}
    tie['candidates'] = [{'id': 'noop_a', 'plan': [{'op': 'noop'}]}, {'id': 'noop_b', 'plan': []}]
    emit('E', 'easy', tie, ['tied_best', 'binary_deterministic_success'])
    impossible = deepcopy(base)
    impossible['goal'] = {'balances': {'a': balance + amount * 9 + 1}}
    emit('E', 'medium', impossible, ['no_candidate_solution', 'abstention'])
    conflict = deepcopy(base)
    conflict['goal'] = {'statuses': {'e': 'settled'}}
    conflict['constraints'] = {'forbidden_statuses': ['settled']}
    emit('E', 'hard', conflict, ['conflicting_goal_constraint', 'no_feasible_solution'])
    infeasible = deepcopy(base)
    infeasible['constraints'] = {'minimum_balances': {'a': balance + amount * 100}}
    emit('E', 'hard', infeasible, ['all_candidates_infeasible', 'initial_constraint_conflict'])
    # F/G: matched cheap/expanded searches and bounded prover cases.
    repair = deepcopy(base)
    repair['candidates'] = repair['candidates'][1:] + repair['candidates'][:1]
    for task, tag in ((repair, 'expanded_improves'), (base, 'cheap_matches_expanded')):
        task = deepcopy(task)
        task['search_request'] = {'cheap_candidate_budget': 1, 'expanded_candidate_budget': len(task['candidates']),
                                  'same_candidate_universe': True}
        f = emit('F', 'medium', task, [tag, 'matched_search'])
        f['targets']['search'] = _search(f['input'], (1, len(task['candidates'])))
        f['id'] = record_id(f)
    for task, tag in ((base, 'direct_succeeds'), (repair, 'search_repairs_direct'),
                      (impossible, 'search_fails')):
        g = emit('G', 'hard', deepcopy(task), [tag, 'decision_distillation'])
        g['targets']['search'] = _search(g['input'], (1, len(task['candidates'])))
        g['id'] = record_id(g)
    omitted = deepcopy(base)
    omitted['constraints'] = deepcopy(base['constraints'])
    # Goal reversal to settled avoids forbidden-void constraints in OOD;
    # the proposer intentionally omits settle while the bounded oracle includes it.
    omitted['goal'] = {'statuses': {'e': 'settled'}}
    omitted['candidates'] = [c for c in omitted['candidates'] if c['id'] != 'settle']
    g = emit('G', 'hard', omitted, ['proposer_omitted_solution'])
    alphabet = [{'op': 'noop'}, {'op': 'settle', 'entry': 'e'}, {'op': 'void', 'entry': 'e'},
                {'op': 'transfer', 'source': 'a', 'destination': 'b', 'amount': amount},
                {'op': 'transfer', 'source': 'b', 'destination': 'a', 'amount': amount}]
    g['targets']['bounded_proof'] = _bounded_proof(g['input'], alphabet, cfg['exhaustive_search']['max_horizon'])
    g['id'] = record_id(g)
    # H: simultaneous goals under the reserved constraint composition. The
    # held-out conjunction is explicitly OOD, not a renamed training sample.
    composed = deepcopy(base)
    composed['goal'] = {'balances': {'a': balance + amount}, 'statuses': {'e': 'settled'}}
    composed['candidates'].append({'id': 'credit_then_transfer', 'plan': [{'op': 'settle', 'entry': 'e'},
                                     {'op': 'transfer', 'source': 'a', 'destination': 'b', 'amount': amount}]})
    if split == 'train':
        for constraint in ({'minimum_balances': {'a': balance}}, {'max_cost': 1},
                           {'forbidden_statuses': ['void']}):
            support = deepcopy(composed)
            support['constraints'] = constraint
            support['candidates'].append({'id': 'credit_then_reverse',
                'plan': [{'op': 'settle', 'entry': 'e'}, {'op': 'void', 'entry': 'e'}]})
            emit('H', 'medium', support, ['individual_constraint_primitive', 'composition_support'])
    h = emit('H', 'hard', composed, ['simultaneous_goals', 'priority_feasibility_then_goal_atoms_then_cost',
         'heldout_composition' if split == 'ood' else 'primitive_composition_support'])
    h['metadata']['transfer'] = {'experiment_role': 'heldout_combination' if split == 'ood' else 'support',
                               'primitive_rules': ['minimum_balances', 'max_cost', 'forbidden_statuses'],
                               'heldout_combination': ['minimum_balances', 'max_cost', 'forbidden_statuses'],
                               'claim': 'evaluation_design_only_no_model_transfer_result'}
    h['id'] = record_id(h)
    # A cross-family probe represents the same additive consequence in a
    # register world. It establishes evaluation pairs, not learned transfer.
    register_constraints = {}
    if spec['constraint_composition'] == 'minimum':
        register_constraints = {'minimum_values': {'a': balance}}
    elif spec['constraint_composition'] == 'cost':
        register_constraints = {'max_cost': 1}
    elif spec['constraint_composition'] == 'minimum_cost_status':
        register_constraints = {'minimum_values': {'a': balance}, 'max_cost': 1, 'protected_registers': []}
    register = {'environment': 'register',
                'initial_state': {'registers': deepcopy(base['initial_state']['accounts'])},
                'rules': {'value_bounds': [-amount * 100, amount * 100], 'overflow': 'reject',
                          'operations': ['add', 'multiply', 'set', 'copy', 'swap', 'noop'],
                          'action_costs': {'add': 1, 'multiply': 1, 'set': 1, 'copy': 1, 'swap': 1, 'noop': 0}},
                'goal': {'values': {'a': balance + amount}}, 'constraints': register_constraints,
                'candidates': [{'id': 'add', 'plan': [{'op': 'add', 'register': 'a', 'value': amount}]},
                               {'id': 'noop', 'plan': [{'op': 'noop'}]},
                               {'id': 'subtract', 'plan': [{'op': 'add', 'register': 'a', 'value': -amount}]},
                               {'id': 'invalid', 'plan': [{'op': 'add', 'register': 'absent', 'value': amount}]}],
                'instruction': 'Compare the register plans by feasibility, satisfied goal atoms and negative cost.'}
    transfer = emit('H', 'medium', register, ['cross_environment_transfer_probe', 'same_additive_primitive'])
    transfer['provenance']['parent_world_id'] = parent + '_register'
    transfer['metadata']['counterfactual_group_id'] = parent + '_register'
    transfer['metadata']['transfer'] = {'experiment_role': 'cross_family_probe',
           'paired_parent_world_id': parent, 'paired_family': 'A',
           'candidate_correspondence': {'add': 'settle', 'noop': 'noop', 'subtract': 'transfer_out', 'invalid': 'invalid'},
           'claim': 'aligned_numeric_goal_and_action_ranking_only_no_model_transfer_result'}
    transfer['id'] = record_id(transfer)
    if cfg['candidate_order'] == 'balanced':
        changed_ids = {}
        for record in records:
            old_id = record['id']
            # Prefix-budget searches have an order-sensitive decision-time
            # policy. Preserve that policy while balancing action-only views.
            if 'search' not in record['targets']:
                _rotate(record, (seed + record['provenance']['record_index']) % len(record['input']['candidates']))
            changed_ids[old_id] = record['id']
        for record in records:
            perturbation = record['metadata']['perturbation']
            if perturbation.get('base_record_id') in changed_ids:
                perturbation['base_record_id'] = changed_ids[perturbation['base_record_id']]
                record['id'] = record_id(record)
    return records


def measure_search_runtime(inputs, repeats=3):
    """Optional runtime evidence; NEVER included by deterministic generation.

    Time both matched prefix configurations separately using a monotonic clock.
    Tiny CPU timings are noisy, so retain raw samples and their scope instead
    of treating elapsed time as a scientific utility or stable target.
    """
    import time
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('runtime repetitions must be bounded to 1..10')
    samples = {'cheap': [], 'expanded': []}
    for _ in range(repeats):
        for name, budget in (('cheap', 1), ('expanded', len(inputs['candidates']))):
            task = deepcopy(inputs)
            task['candidates'] = task['candidates'][:budget]
            start = time.perf_counter_ns()
            _oracle(task)
            samples[name].append(time.perf_counter_ns() - start)
    return {'clock': 'perf_counter_ns', 'unit': 'nanoseconds', 'repeats': repeats,
            'candidate_budgets': {'cheap': 1, 'expanded': len(inputs['candidates'])},
            'duration_samples': samples,
            'measurement_scope': 'Actual primary.evaluate calls; input-copy time excluded. Counters and semantic benefit remain separate exact targets.',
            'timing_is_nondeterministic': True}
