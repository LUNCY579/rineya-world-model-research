"""Once-only, inactive DEV register-composition prototype. Never publishes shards.

Developmental hand witnesses preceded this experiment; fresh fixed DEV parents
test implementation/diagnostic properties, not learned model generalization.
The production registry has no prototype profile and remains v1/v2 only.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import re
import time

from ..auditors.integrity import audit_record, recursive_input_check, validate_decision
from ..oracles import primary, independent
from ..pipelines.common import ROOT, canonical, digest, file_hash, immutable_bytes, source_hashes
from .build import decision_labels
from .version2 import generate_group

VERSION = 'h-composition-prototype-1'
SPEC_PATH = 'configs/h_composition_dev_experiment.json'
FREEZE_PATH = 'manifests/H_COMPOSITION_EXPERIMENT_FREEZE.json'
RESULT_PATH = 'reports/H_COMPOSITION_CONTROLLED_EXPERIMENT.json'
STARTED_PATH = 'manifests/H_COMPOSITION_EXPERIMENT_STARTED.json'
OPERATIONS = ('add', 'multiply', 'set', 'copy', 'swap', 'noop')
RESULT_COUNTS = {'tasks': 800, 'baseline_tasks': 32, 'baseline_correct': 32,
                 'goal_only_equality_checks': 32, 'constraint_only_equality_checks': 32,
                 'three_step_goal_winners': 96, 'conjunction_exact_two_way_ties': 32,
                 'goal_atom_ablations_change_best': 544,
                 'constraint_blind_best_membership': 64, 'terminal_only_constraint_best_membership': 64,
                 'terminal_only_constraint_exact_set': 64}


def _action(op, **fields):
    return {'op': op, **fields}


def _base(seed, arm, state, plans):
    # Neither IDs nor order depend on goal, constraints, or expected labels.
    ids = {name: 'c_' + digest([VERSION, seed, arm, i])[:24]
           for i, name in enumerate(plans)}
    candidates = [{'id': ids[name], 'plan': deepcopy(plan)} for name, plan in plans.items()]
    candidates.sort(key=lambda c: digest([seed, arm, 'candidate_order', c['id']]))
    task = {'environment': 'register', 'initial_state': {'registers': state},
            'rules': {'value_bounds': [-256, 256], 'overflow': 'reject',
                      'operations': list(OPERATIONS),
                      'action_costs': {op: ((1 if arm == 'G' else 4) if op == 'set' else 0 if op == 'noop' else 1)
                                       for op in OPERATIONS}},
            'goal': {'values': {}}, 'constraints': {}, 'candidates': candidates,
            'utility_definition': deepcopy(primary.UTILITY_DEFINITION)}
    return task, ids


def make_bundles(seed, index):
    """Pure construction. Index arithmetic is frozen; no search or rejection."""
    if type(seed) is not int or seed < 0 or type(index) is not int or not 0 <= index < 32:
        raise ValueError('prototype requires a nonnegative seed and index in [0,32)')
    x, y, m = 2 + index % 16, 21 + index, 2 + index // 16
    swap = _action('swap', left='a', right='b')
    multiply = _action('multiply', register='a', value=m)
    copy = _action('copy', source='a', destination='c')
    goals = [{'a': m*y, 'b': x, 'c': m*y}, {'a': y, 'b': m*x, 'c': y},
             {'a': m*y, 'b': x, 'c': x}]
    plans = {'P': [swap, multiply, copy], 'Q': [multiply, swap, copy],
             'R': [copy, swap, multiply]}
    partials = set()
    for goal in goals:
        for omitted in ('a', 'b', 'c'):
            plan = [_action('set', register=k, value=v) for k, v in goal.items() if k != omitted]
            key = canonical(plan)
            if key not in partials:
                plans['partial_' + str(len(partials))] = plan
                partials.add(key)
    task, ids = _base(seed, 'G', {'a': x, 'b': y, 'c': 0}, plans)
    goal_rows = []
    for i, role in enumerate(('P', 'Q', 'R')):
        row = deepcopy(task); row['goal'] = {'values': goals[i]}
        goal_rows.append({'arm': 'G', 'variant': i, 'input': row,
                          'expected_best': [ids[role]], 'role_ids': deepcopy(ids)})

    a, b, c = 2 + index % 8, 3 + index // 8, 6 + index
    target_a, target_b = 2*c-2, 2*c
    p = [_action('add', register='a', value=target_b-a),
         _action('copy', source='a', destination='b'), _action('add', register='a', value=-2)]
    r = [_action('multiply', register='c', value=2),
         _action('copy', source='c', destination='b'), _action('set', register='a', value=target_a)]
    q = [_action('set', register='a', value=target_a), _action('set', register='b', value=target_b)]
    plans = {'P': p, 'R': r, 'Q': q, 'Q_reverse': list(reversed(q)),
             'F': p[:-1], 'N': [_action('noop')],
             'partial_a': [_action('add', register='a', value=target_a-a)],
             'partial_b_maximum': r[:-1],
             'partial_b_conjunction': [_action('set', register='b', value=target_b)]}
    task, ids = _base(seed, 'X', {'a': a, 'b': b, 'c': c}, plans)
    task['goal'] = {'values': {'a': target_a, 'b': target_b}}
    constraints = [{}, {'maximum_values': {'a': target_a}}, {'protected_registers': ['c']},
                   {'maximum_values': {'a': target_a}, 'protected_registers': ['c']}]
    expected = [('P',), ('R',), ('P',), ('Q', 'Q_reverse')]
    constraint_rows = []
    for i, roles in enumerate(expected):
        row = deepcopy(task); row['constraints'] = constraints[i]
        constraint_rows.append({'arm': 'X', 'variant': i, 'input': row,
                                'expected_best': sorted(ids[role] for role in roles),
                                'role_ids': deepcopy(ids)})
    return goal_rows, constraint_rows


def _dual(task):
    recursive_input_check(task)
    first, second = primary.evaluate(task), independent.evaluate(task)
    if canonical(first) != canonical(second):
        raise ValueError('prototype oracle disagreement')
    labels = decision_labels(first)
    validate_decision(labels, second)
    return first, labels


def _fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def _blind_choices(task):
    candidates = task['candidates']
    names = task['initial_state']['registers']
    adds = [c for c in candidates if len(c['plan']) == 1 and c['plan'][0].get('op') == 'add'
            and c['plan'][0].get('register') in names and c['plan'][0].get('value', 0) > 0]
    return {'valid_positive_singleton_add_then_ID': min(c['id'] for c in (adds or candidates)),
            'first_option': candidates[0]['id'],
            'shortest_plan_then_ID': min(candidates, key=lambda c: (len(c['plan']), c['id']))['id'],
            'longest_plan_then_ID': min(candidates, key=lambda c: (-len(c['plan']), c['id']))['id'],
            'lowest_opaque_ID': min(c['id'] for c in candidates)}


def _terminal_only_best(task, oracle):
    # Deliberately flawed diagnostic: ignore constraints at intermediate states.
    values = {}
    initial = task['initial_state']['registers']
    constraints = task['constraints']
    for candidate in oracle['candidates']:
        o = candidate['outcomes'][0]; end = o['terminal_state']['registers']
        broken = any(end[k] > v for k, v in constraints.get('maximum_values', {}).items())
        broken |= any(end[k] != initial[k] for k in constraints.get('protected_registers', []))
        values[candidate['id']] = (int(o['legal'] and not broken), o['goal_score'], -o['cost'])
    top = max(values.values())
    return sorted(k for k, v in values.items() if v == top) if top[0] else []


def _bounded_json(relative, limit):
    path = Path(ROOT).resolve() / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('prototype control must be a regular nonsymlink file')
    if path.stat().st_size > limit:
        raise ValueError('bounded prototype control exceeds byte limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('bounded prototype control exceeds byte limit')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate prototype JSON key')
            value[key] = item
        return value

    def finite(value):
        raise ValueError('nonfinite prototype JSON value')

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    if type(value) is not dict:
        raise ValueError('prototype control must be a JSON object')
    return value


def _spec():
    spec = _bounded_json(SPEC_PATH, 65536)
    # Deliberately fixed experiment, rather than a user-configurable search API.
    if (spec.get('version') != 'h-composition-controlled-dev-1' or spec.get('prototype_version') != VERSION
            or spec.get('split') != 'dev' or spec.get('seed_start') != 430000 or spec.get('parents') != 32
            or spec.get('goal_tasks_per_parent') != 3 or spec.get('constraint_tasks_per_parent') != 4
            or spec.get('baseline_tasks_per_parent') != 1
            or spec.get('max_tasks') != 800 or spec.get('max_wall_seconds') != 300
            or spec.get('max_result_bytes') != 1048576 or spec.get('max_horizon') != 3
            or spec.get('goal_candidate_count') != 11 or spec.get('constraint_candidate_count') != 9
            or spec.get('workers') != 1
            or spec.get('replacement_seeds') is not False or spec.get('publication') is not False
            or spec.get('model_training') is not False):
        raise ValueError('unsupported frozen DEV experiment')
    return spec


def freeze():
    spec = _spec(); code = source_hashes()
    # Input commitments are computed without invoking either oracle or baseline.
    commitments = []
    for i in range(spec['parents']):
        seed = spec['seed_start'] + i
        g, x = make_bundles(seed, i)
        commitments.append({'seed': seed, 'goal_inputs': [digest(row['input']) for row in g],
                            'constraint_inputs': [digest(row['input']) for row in x]})
    receipt = {'version': 'h-composition-experiment-freeze-1', 'specification': spec,
               'specification_sha256': file_hash(ROOT / SPEC_PATH), 'code_hashes': code,
               'source_digest': digest(code), 'input_commitments': commitments,
               'publication': False, 'training_launched': False,
               'developmental_witnesses': 'Hand arithmetic and separate exploratory witnesses preceded freeze; these are mechanical confirmation parents, not a learned generalization test'}
    immutable_bytes(ROOT / FREEZE_PATH, (canonical(receipt) + '\n').encode())
    return receipt


def _existing_result(result, frozen, freeze_hash, code_digest):
    """Authenticate complete recorded custody without executing the pilot again.

    The independent review additionally pins the exact result file SHA256.
    This structural/integrity check alone is not a reviewer signature.
    """
    started = _bounded_json(STARTED_PATH, 4096)
    if (set(started) != {'version', 'freeze_sha256', 'source_digest', 'started_at_ns'}
            or started['version'] != 'h-composition-experiment-started-1'
            or started['freeze_sha256'] != freeze_hash or started['source_digest'] != code_digest
            or type(started['started_at_ns']) is not int or started['started_at_ns'] <= 0):
        raise ValueError('prototype started receipt custody mismatch')
    if (result.get('version') != 'h-composition-controlled-result-1' or result.get('prototype_version') != VERSION
            or result.get('freeze_sha256') != freeze_hash or result.get('source_digest') != code_digest
            or result.get('verdict') not in ('PASS', 'FAIL')
            or result.get('started_sha256') != file_hash(ROOT / STARTED_PATH)
            or result.get('canonical_records_published') != 0 or result.get('training_launched') is not False):
        raise ValueError('existing prototype result custody mismatch')
    integrity = result.get('integrity_sha256')
    body = {k: v for k, v in result.items() if k != 'integrity_sha256'}
    if integrity != digest(body):
        raise ValueError('prototype result integrity commitment mismatch')
    if result['verdict'] == 'FAIL' and 'error_class' in result:
        return result
    counts = result.get('counts', {})
    if ((result['verdict'] == 'PASS' and counts != RESULT_COUNTS)
            or type(counts) is not dict or set(counts) != set(RESULT_COUNTS)
            or any(type(v) is not int or not 0 <= v <= RESULT_COUNTS[k] for k, v in counts.items())
            or result.get('expected_counts') != RESULT_COUNTS
            or result.get('oracle_or_label_mismatches') != 0 or result.get('reproduction_mismatches') != 0
            or result.get('supplied_operations') != sorted(OPERATIONS)
            or len(result.get('parent_receipts', [])) != 32
            or [p.get('seed') for p in result['parent_receipts']] != list(range(430000, 430032))
            or set(result.get('goal_blind_best_membership', {})) != set(_blind_choices(make_bundles(111, 0)[0][0]['input']))
            or any(type(n) is not int or not 0 <= n <= 32 for n in result['goal_blind_best_membership'].values())):
        raise ValueError('completed prototype result lacks frozen full experiment evidence')

    def sha(value):
        return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

    for i, parent in enumerate(result['parent_receipts']):
        rows = make_bundles(parent['seed'], i)
        commitment = frozen['input_commitments'][i]
        for arm, checks, tasks, hashes in (
                ('G', parent.get('goal'), rows[0], commitment['goal_inputs']),
                ('X', parent.get('constraints'), rows[1], commitment['constraint_inputs'])):
            if type(checks) is not list or len(checks) != len(tasks):
                raise ValueError('prototype per-parent receipt inventory mismatch')
            for j, (check, task_row, input_hash) in enumerate(zip(checks, tasks, hashes)):
                task = task_row['input']; best = task_row['expected_best']
                plans = {c['id']: c['plan'] for c in task['candidates']}
                if (check.get('arm') != arm or check.get('variant') != j
                        or check.get('input_sha256') != input_hash or digest(task) != input_hash
                        or not sha(check.get('oracle_sha256')) or not sha(check.get('decision_sha256'))
                        or check.get('best_candidate_ids') != best
                        or check.get('winning_plan_lengths') != [len(plans[k]) for k in best]
                        or type(check.get('candidate_summary')) is not list
                        or [c.get('id') for c in check['candidate_summary']] != [c['id'] for c in task['candidates']]):
                    raise ValueError('prototype per-task receipt custody mismatch')
                for summary in check['candidate_summary']:
                    if (set(summary) != {'id', 'utility', 'terminal_state', 'constraint_violations'}
                            or type(summary['utility']) is not list or len(summary['utility']) != 3
                            or type(summary['terminal_state']) is not dict
                            or set(summary['terminal_state']) != {'registers'}
                            or set(summary['terminal_state']['registers']) != {'a', 'b', 'c'}
                            or any(type(v) is not int for v in summary['terminal_state']['registers'].values())
                            or type(summary['constraint_violations']) is not list):
                        raise ValueError('prototype candidate summary incomplete')
                ablations = check.get('goal_atom_ablations')
                atoms = list(task['goal']['values'])
                if type(ablations) is not list or [a.get('omitted_atom') for a in ablations] != atoms:
                    raise ValueError('prototype per-atom evidence inventory mismatch')
                for ablation, omitted in zip(ablations, atoms):
                    reduced = deepcopy(task); del reduced['goal']['values'][omitted]
                    ids = ablation.get('best_candidate_ids')
                    if (ablation.get('input_sha256') != digest(reduced)
                            or not sha(ablation.get('oracle_sha256')) or not sha(ablation.get('decision_sha256'))
                            or type(ids) is not list or not ids or ids != sorted(set(ids))
                            or not set(ids) <= set(plans) or set(ids) & set(best)):
                        raise ValueError('prototype per-atom evidence custody mismatch')
        baseline = parent.get('baseline', {})
        if (not all(sha(baseline.get(k)) for k in ('input_sha256', 'target_sha256'))
                or type(baseline.get('record_id')) is not str or not baseline['record_id']
                or baseline.get('best_candidate_ids') != [baseline.get('selected_candidate_id')]):
            raise ValueError('prototype baseline receipt incomplete')
    return result


def measure():
    frozen = _bounded_json(FREEZE_PATH, 1048576); spec = _spec(); code = source_hashes()
    freeze_hash = file_hash(ROOT / FREEZE_PATH)
    if (frozen['source_digest'] != digest(code) or frozen['code_hashes'] != code
            or frozen['specification'] != spec or frozen['specification_sha256'] != file_hash(ROOT / SPEC_PATH)):
        raise ValueError('frozen source/specification changed; no adaptive rerun permitted')
    result_path = ROOT / RESULT_PATH
    if result_path.exists():
        result = _bounded_json(RESULT_PATH, spec['max_result_bytes'])
        return _existing_result(result, frozen, freeze_hash, digest(code))
    if (ROOT / STARTED_PATH).exists():
        raise ValueError('prior DEV attempt interrupted; preserve evidence, no adaptive rerun permitted')
    immutable_bytes(ROOT / STARTED_PATH, (canonical({'version': 'h-composition-experiment-started-1',
                    'freeze_sha256': freeze_hash, 'source_digest': digest(code),
                    'started_at_ns': time.time_ns()}) + '\n').encode())
    started = time.monotonic(); deadline = started + spec['max_wall_seconds']
    receipts, counts, goal_choices = [], Counter(), Counter()
    baseline_generated_rows = 0
    operations = set()

    def bounded():
        if time.monotonic() >= deadline:
            raise TimeoutError('frozen DEV wall budget exhausted')
        if counts['tasks'] > spec['max_tasks']:
            raise ValueError('frozen task budget exceeded')

    def checked(row):
        bounded()
        task = row['input']; oracle, labels = _dual(task)
        best = labels['best_candidate_ids']
        if best != row['expected_best']:
            raise ValueError('hand-formula best-set mismatch')
        if any(not c['outcomes'][0]['legal'] for c in oracle['candidates']):
            raise ValueError('prototype supplied an illegal candidate')
        for candidate in oracle['candidates']:
            outcome = candidate['outcomes'][0]
            if candidate['id'] in best and (_fraction(candidate['success_probability']) != 1
                                           or _fraction(candidate['feasible_probability']) != 1):
                raise ValueError('best candidate does not feasibly achieve every goal atom')
        plans = {c['id']: c['plan'] for c in task['candidates']}
        expected_count = 11 if row['arm'] == 'G' else 9
        if max(map(len, plans.values())) > 3 or len(plans) != expected_count or len({canonical(p) for p in plans.values()}) != expected_count:
            raise ValueError('prototype plan/candidate/duplicate bounds')
        for plan in plans.values():
            operations.update(a['op'] for a in plan)
        if any(len(plans[k]) <= 1 for k in best):
            raise ValueError('one-action prototype winner')
        counts['tasks'] += 1
        ablations = []
        for omitted in task['goal']['values']:
            bounded()
            ablated = deepcopy(task); del ablated['goal']['values'][omitted]
            ablated_oracle, ablated_labels = _dual(ablated)
            if (not ablated_labels['best_candidate_ids']
                    or set(ablated_labels['best_candidate_ids']) & set(best)):
                raise ValueError('goal atom redundant for complete best-set recovery')
            counts['goal_atom_ablations_change_best'] += 1
            counts['tasks'] += 1
            ablations.append({'omitted_atom': omitted, 'input_sha256': digest(ablated),
                              'oracle_sha256': digest(ablated_oracle), 'decision_sha256': digest(ablated_labels),
                              'best_candidate_ids': ablated_labels['best_candidate_ids']})
        return oracle, labels, {'arm': row['arm'], 'variant': row['variant'], 'input_sha256': digest(task),
                                'oracle_sha256': digest(oracle), 'decision_sha256': digest(labels),
                                'best_candidate_ids': best, 'winning_plan_lengths': [len(plans[k]) for k in best],
                                'goal_atom_ablations': ablations,
                                'candidate_summary': [{'id': c['id'], 'utility': c['utility'],
                                                       'terminal_state': c['outcomes'][0]['terminal_state'],
                                                       'constraint_violations': c['outcomes'][0]['constraint_violations']}
                                                      for c in oracle['candidates']]}

    for i in range(spec['parents']):
        bounded(); seed = spec['seed_start'] + i
        g, x = make_bundles(seed, i)
        if canonical((g, x)) != canonical(make_bundles(seed, i)):
            raise ValueError('prototype reproduction mismatch')
        expected = frozen['input_commitments'][i]
        if expected != {'seed': seed, 'goal_inputs': [digest(r['input']) for r in g],
                         'constraint_inputs': [digest(r['input']) for r in x]}:
            raise ValueError('frozen input commitment mismatch')
        for rows, axis in ((g, 'goal'), (x, 'constraints')):
            residuals = []
            for row in rows:
                residual = deepcopy(row['input']); residual.pop(axis); residuals.append(canonical(residual))
            if len(set(residuals)) != 1:
                raise ValueError('paired prototype changes more than declared axis')
        parent = {'seed': seed, 'goal': [], 'constraints': []}
        choices = _blind_choices(g[0]['input'])
        bests = []
        for row in g:
            oracle, labels, check = checked(row); bests.append(set(labels['best_candidate_ids']))
            if _blind_choices(row['input']) != choices:
                raise ValueError('goal-blind choice changed within fixed triplet')
            for name, chosen in choices.items():
                goal_choices[name] += int(chosen in labels['best_candidate_ids'])
            if check['winning_plan_lengths'] != [3]:
                raise ValueError('goal winner does not execute all three steps')
            counts['three_step_goal_winners'] += 1; parent['goal'].append(check)
        if any(bests[a] & bests[b] for a in range(3) for b in range(a)):
            raise ValueError('goal triplet optimal sets overlap')
        counts['goal_only_equality_checks'] += 1
        constraint_blind = None
        for row in x:
            oracle, labels, check = checked(row)
            if constraint_blind is None:
                constraint_blind = labels['best_candidate_ids']
            counts['constraint_blind_best_membership'] += int(bool(set(constraint_blind) & set(labels['best_candidate_ids'])))
            terminal = _terminal_only_best(row['input'], oracle)
            counts['terminal_only_constraint_best_membership'] += int(bool(set(terminal) & set(labels['best_candidate_ids'])))
            counts['terminal_only_constraint_exact_set'] += int(terminal == labels['best_candidate_ids'])
            if row['variant'] == 3:
                if len(labels['best_candidate_ids']) != 2:
                    raise ValueError('conjunction must retain both equal-cost distinct-order plans')
                counts['conjunction_exact_two_way_ties'] += 1
            parent['constraints'].append(check)
        counts['constraint_only_equality_checks'] += 1
        bounded()
        # Same fixed DEV seeds; fresh generation, never reading old accepted gold.
        baseline_group = generate_group(seed, split='dev')
        baseline_generated_rows += len(baseline_group)
        old = [r for r in baseline_group
               if r['family'] == 'H' and r['input']['environment'] == 'register']
        if len(old) != 1:
            raise ValueError('baseline must contain exactly one current v2 H register record')
        audit_record(old[0])
        choice = _blind_choices(old[0]['input'])['valid_positive_singleton_add_then_ID']
        counts['baseline_correct'] += int(choice in old[0]['targets']['decision']['best_candidate_ids'])
        counts['baseline_tasks'] += 1; counts['tasks'] += 1
        parent['baseline'] = {'record_id': old[0]['id'], 'input_sha256': digest(old[0]['input']),
                              'target_sha256': digest(old[0]['targets']), 'selected_candidate_id': choice,
                              'best_candidate_ids': old[0]['targets']['decision']['best_candidate_ids']}
        receipts.append(parent); bounded()
    if source_hashes() != code:
        raise ValueError('source changed during frozen measurement')
    required = RESULT_COUNTS
    passed = all(counts[k] == v for k, v in required.items()) and operations == set(OPERATIONS)
    passed &= all(n <= 32 for n in goal_choices.values())
    result = {'version': 'h-composition-controlled-result-1', 'prototype_version': VERSION,
              'verdict': 'PASS' if passed else 'FAIL', 'source_digest': digest(code),
              'freeze_sha256': freeze_hash, 'started_sha256': file_hash(ROOT / STARTED_PATH),
              'counts': dict(counts), 'expected_counts': required,
              'goal_blind_best_membership': dict(goal_choices), 'supplied_operations': sorted(operations),
              'baseline_generation_work': {'full_v2_groups_generated': 32,
                                            'all_family_rows_generated': baseline_generated_rows,
                                            'selected_H_register_rows_evaluated': 32,
                                            'unused_family_rows_not_counted_as_pilot_cases': baseline_generated_rows - 32,
                                            'note': 'Full v2 group construction executes oracles on unused family rows; the 800-case cap counts evaluated diagnostic cases, while the 300-second wall bound covers all this work'},
              'oracle_or_label_mismatches': 0, 'reproduction_mismatches': 0,
              'elapsed_seconds': time.monotonic() - started, 'parent_receipts': receipts,
              'canonical_records_published': 0, 'training_launched': False,
              'interpretation': spec['interpretation'],
              'limits': ['Prototype uses fixed arithmetic templates and bare validated tasks, not an allowlisted canonical profile.',
                         'Numeric worlds derive from index, while seed changes IDs/order; developmental witness indices overlap some confirmation dynamics, so this is not an unseen semantic holdout or IID evidence.',
                         'Baseline and prototype change goal cardinality, plan length and candidates; only within-bundle contrasts isolate goal/constraint effects.',
                         'Formula-aware goal/constraint-sensitive shortcuts remain possible; no learned model was evaluated.',
                         'Ledger H coverage and TRAIN/OOD reservations require separate implementation and review.']}
    result['integrity_sha256'] = digest(result)
    _existing_result(result, frozen, freeze_hash, digest(code))
    raw = (canonical(result) + '\n').encode()
    if len(raw) > spec['max_result_bytes']:
        raise ValueError('bounded research result exceeds one MiB')
    immutable_bytes(result_path, raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    try:
        result = freeze() if args.freeze else measure()
    except Exception as error:
        if not args.freeze and (ROOT / STARTED_PATH).is_file() and not (ROOT / RESULT_PATH).exists():
            failure = {'version': 'h-composition-controlled-result-1', 'prototype_version': VERSION,
                       'verdict': 'FAIL', 'error_class': type(error).__name__, 'error': str(error),
                       'source_digest': digest(source_hashes()), 'freeze_sha256': file_hash(ROOT / FREEZE_PATH),
                       'started_sha256': file_hash(ROOT / STARTED_PATH),
                       'canonical_records_published': 0, 'training_launched': False}
            failure['integrity_sha256'] = digest(failure)
            immutable_bytes(ROOT / RESULT_PATH, (canonical(failure) + '\n').encode())
        raise
    print(canonical({k: v for k, v in result.items() if k not in ('code_hashes', 'specification', 'input_commitments', 'parent_receipts')}))
    return 0 if result.get('verdict', 'PASS') == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
