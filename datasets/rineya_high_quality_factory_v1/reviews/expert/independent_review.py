"""Blind expert verifier written from public SEMANTICS and visible candidate prompts.

No generator, reference implementation, oracle, calibration expectation, or other
reviewer's receipts are imported or read. Manual agent inspection is documented
separately in semantic_inspection.md; this file independently recomputes each row.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'reviews/expert'
ROLE = 'expert_reviewer'
EXECUTION = '/root/expert_reviewer'
SCORE_KEYS = ('semantic_correctness', 'target_correctness', 'instruction_clarity',
              'reasoning_validity', 'generalization_value', 'difficulty_quality',
              'diversity', 'shortcut_resistance')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def expr(node, event, state):
    require(isinstance(node, dict), 'AST node must be an object')
    if set(node) == {'const'}:
        value = node['const']
        require(value is None or type(value) in (str, int, bool), 'nonscalar AST constant')
        return value
    if set(node) == {'var'}:
        mapping = {'event.status': event['status'], 'event.amount': event['amount'],
                   'state.x': state['x'], 'state.y': state['y']}
        require(node['var'] in mapping, 'unknown AST variable')
        return mapping[node['var']]
    require(set(node) == {'op', 'args'}, 'wrong AST node keys')
    op, args = node['op'], node['args']
    require(isinstance(args, list), 'AST args must be a list')
    require(op in ('eq', 'ne', 'ge', 'gt', 'le', 'lt', 'all', 'any', 'not'), 'unknown AST op')
    require(op in ('all', 'any') or len(args) == (1 if op == 'not' else 2), 'wrong AST arity')
    # Eagerly inspect every branch, including ones ordinary short-circuiting skips.
    values = [expr(a, event, state) for a in args]
    if op in ('all', 'any', 'not'):
        require(all(type(v) is bool for v in values), 'boolean op received nonboolean')
        return all(values) if op == 'all' else any(values) if op == 'any' else not values[0]
    a, b = values
    if op == 'eq':
        return same(a, b)
    if op == 'ne':
        return not same(a, b)
    require(type(a) is type(b) and type(a) in (int, str), 'invalid ordered comparison')
    return {'ge': lambda: a >= b, 'gt': lambda: a > b,
            'le': lambda: a <= b, 'lt': lambda: a < b}[op]()


def causal(task):
    state = copy.deepcopy(task['state'])
    require(set(state) == {'x', 'y'} and all(integer(v) for v in state.values()), 'invalid causal state')
    steps = []
    for event in task['events']:
        require(integer(event['amount']) and event['amount'] > 0, 'invalid amount')
        require(event['status'] is None or type(event['status']) in (str, int, bool), 'invalid status')
        op, src, dst, amount = (event[k] for k in ('op', 'source', 'destination', 'amount'))
        require(op in ('add', 'scale', 'transfer', 'swap') and src in state and dst in state, 'invalid event')
        require(op not in ('transfer', 'swap') or src != dst, 'identical source and destination')
        rule = expr(task['rule'], event, state)
        pre = expr(task['precondition'], event, state)
        require(type(rule) is bool and type(pre) is bool, 'gate root is not boolean')
        calc = 'identity'
        if rule and pre:
            calc = op
            if op == 'add':
                state[src] += amount
            elif op == 'scale':
                state[src] *= amount
            elif op == 'transfer':
                state[src], state[dst] = state[src] - amount, state[dst] + amount
            else:
                state[src], state[dst] = state[dst], state[src]
        steps.append({'cmp': {'rule': rule, 'precondition': pre},
                      'calc': {'operation': calc}, 'after': state.copy()})
    return {'after': state, 'steps': steps}


def graph_setup(task):
    nodes = task['nodes']
    require(isinstance(nodes, list) and all(type(n) is str for n in nodes) and len(set(nodes)) == len(nodes), 'invalid graph nodes')
    require(task['start'] in nodes and task['goal'] in nodes, 'unknown start or goal')
    require(integer(task['fuel']) and task['fuel'] >= 0, 'invalid initial fuel')
    costs = {}
    for edge in task['edges']:
        require(isinstance(edge, list) and len(edge) == 3, 'invalid edge shape')
        a, b, cost = edge
        require(a in nodes and b in nodes and integer(cost) and cost > 0, 'invalid edge')
        require((a, b) not in costs, 'duplicate ordered edge')
        costs[a, b] = cost
    return costs


def graph_optimum(task, state=None):
    costs = graph_setup(task)
    start = state or {'node': task['start'], 'fuel': task['fuel']}
    queue = collections.deque([(start['node'], start['fuel'], [])])
    seen = {(start['node'], start['fuel'])}
    while queue:
        node, fuel, path = queue.popleft()
        if node == task['goal']:
            return path
        for destination in sorted(task['nodes']):
            cost = costs.get((node, destination))
            if cost is None or cost > fuel:
                continue
            key = destination, fuel - cost
            if key not in seen:
                seen.add(key)
                queue.append((*key, path + [destination]))
    return None


def graph_replay(task, plan):
    costs = graph_setup(task)
    state = {'node': task['start'], 'fuel': task['fuel']}
    steps = []
    valid = True
    for destination in plan:
        cost = costs.get((state['node'], destination))
        legal = cost is not None and cost <= state['fuel']
        if legal:
            state = {'node': destination, 'fuel': state['fuel'] - cost}
        steps.append({'legal': legal, 'after': state.copy()})
        if not legal:
            valid = False
            break
    return {'steps': steps, 'after': state, 'valid': valid,
            'reaches_goal': valid and state['node'] == task['goal']}


def graph(task):
    result = graph_replay(task, task['plan'])
    result['shortest_plan'] = graph_optimum(task)
    return result


def block_setup(task):
    on = task['on']
    names = sorted(on)
    require(all(type(n) is str and n != 'TABLE' for n in names), 'invalid block name')
    for label, mapping in [('initial', on), ('goal', task['goal_on'])]:
        supports = []
        for block, support in mapping.items():
            require(block in on and (support in on or support == 'TABLE') and support != block, 'invalid ' + label + ' support')
            if support != 'TABLE':
                supports.append(support)
            chain, current = {block}, support
            while current != 'TABLE' and current in mapping:
                require(current not in chain, 'cyclic ' + label + ' support')
                chain.add(current)
                current = mapping[current]
        require(len(set(supports)) == len(supports), 'multiple blocks directly on same ' + label + ' support')
    return names


def block_move(on, block, dest):
    clear = set(on) - set(on.values())
    legal = block in clear and (dest == 'TABLE' or dest in clear) and dest != block and on.get(block) != dest
    result = on.copy()
    if legal:
        result[block] = dest
    return legal, result


def block_optimum(task):
    names = block_setup(task)
    state = task['on'].copy()
    queue = collections.deque([(state, [])])
    seen = {tuple(state[n] for n in names)}
    choices = sorted((b, d) for b in names for d in names + ['TABLE'] if b != d)
    while queue:
        on, path = queue.popleft()
        if all(on[b] == d for b, d in task['goal_on'].items()):
            return path
        for block, dest in choices:
            legal, after = block_move(on, block, dest)
            key = tuple(after[n] for n in names)
            if legal and key not in seen:
                seen.add(key)
                queue.append((after, path + [{'block': block, 'to': dest}]))
    return None


def blocks(task):
    block_setup(task)
    on = task['on'].copy()
    steps, valid = [], True
    for move in task['plan']:
        require(set(move) == {'block', 'to'}, 'invalid move shape')
        legal, on = block_move(on, move['block'], move['to'])
        steps.append({'legal': legal, 'after': on.copy()})
        if not legal:
            valid = False
            break
    return {'steps': steps, 'after': on, 'valid': valid,
            'reaches_goal': valid and all(on[b] == d for b, d in task['goal_on'].items()),
            'shortest_plan': block_optimum(task)}


def critic(task):
    world = task['world']
    graph_setup(world)
    candidates = task['candidates']
    ids = [c['candidate_id'] for c in candidates]
    require(all(type(i) is str for i in ids) and len(set(ids)) == len(ids), 'invalid candidate IDs')
    fresh = graph_optimum(world)
    initial_distance = len(fresh) if fresh is not None else None
    evaluations, ranks = [], []
    for candidate in candidates:
        claimed = candidate['claimed_after']
        require(set(claimed) == {'node', 'fuel'} and claimed['node'] in world['nodes'] and integer(claimed['fuel']) and claimed['fuel'] >= 0, 'malformed claimed state')
        replay = graph_replay(world, candidate['plan'])
        remaining = graph_optimum(world, replay['after'])
        distance = len(remaining) if remaining is not None else None
        progress = initial_distance - distance if replay['valid'] and initial_distance is not None and distance is not None else None
        usefulness = -1 if not replay['valid'] else 2 if replay['reaches_goal'] else 1 if progress is not None and progress > 0 else 0
        correct = same(claimed, replay['after'])
        error = next((i for i, s in enumerate(replay['steps']) if not s['legal']), None)
        if replay['valid'] and not correct:
            error = 'claim'
        item = {'candidate_id': candidate['candidate_id'], 'valid': replay['valid'],
                'claim_correct': correct, 'reaches_goal': replay['reaches_goal'],
                'remaining_moves': distance, 'progress': progress, 'usefulness': usefulness,
                'first_error': error, 'repair_plan': fresh}
        evaluations.append(item)
        rank = (replay['valid'] and correct, replay['reaches_goal'], usefulness,
                -distance if distance is not None else -math.inf, -len(candidate['plan']))
        ranks.append((rank, candidate['candidate_id']))
    groups = collections.defaultdict(list)
    for rank, identity in ranks:
        groups[rank].append(identity)
    ranking = [sorted(groups[key]) for key in sorted(groups, reverse=True)]
    return {'evaluations': evaluations, 'ranking': ranking}


def normalized_tree(node):
    if 'const' in node:
        return ('const', type(node['const']).__name__)
    if 'var' in node:
        return ('var', node['var'])
    return node['op'], tuple(normalized_tree(n) for n in node['args'])


def add_issue(issues, category, severity, evidence, fix):
    issues.append({'category': category, 'severity': severity, 'evidence': evidence, 'fix': fix})


def evidence_summary(task, result):
    kind = task['kind']
    if kind == 'causal':
        return {'horizon': len(task['events']), 'gate_values': [s['cmp'] for s in result['steps']],
                'executed_operations': [s['calc']['operation'] for s in result['steps']],
                'independent_after': result['after']}
    if kind in ('graph', 'blocks'):
        return {'supplied_plan_length': len(task['plan']), 'legal_steps': [s['legal'] for s in result['steps']],
                'independent_after': result['after'], 'shortest_plan': result['shortest_plan'],
                'valid': result['valid'], 'reaches_goal': result['reaches_goal']}
    return {'independent_evaluations': result['evaluations'], 'independent_ranking': result['ranking']}


def review(row, peers):
    issues = []
    scores = dict(zip(SCORE_KEYS, (5, 5, 5, 5, 3, 4, 3, 4)))
    summary = {}
    required_revision = False
    try:
        require('Task JSON:' in row['rendered_input'], 'visible prompt lacks parseable Task JSON marker')
        task = json.loads(row['rendered_input'].split('Task JSON:', 1)[1].strip())
        if not same(task, row['semantic_task']):
            add_issue(issues, 'prompt_payload_mismatch', 'high', 'Visible Task JSON differs from semantic_task.', 'Make canonical task and the complete student-visible payload identical.')
            required_revision = True
        result = {'causal': causal, 'graph': graph, 'blocks': blocks, 'critic': critic}[task['kind']](task)
        expected = {'after': result['after']} if row['target_format'] == 'answer_only' else result
        summary = evidence_summary(task, result)
        if 'structural_difficulty' in row:
            summary['structural_difficulty'] = row['structural_difficulty']
            summary['challenge_axes'] = row.get('challenge_axes', [])
            if row['structural_difficulty'] != row['difficulty']:
                add_issue(issues, 'inconsistent_difficulty_metadata', 'medium',
                          'difficulty differs from structural_difficulty.', 'Use one structural difficulty assessment consistently.')
                required_revision = True
        summary['target_match'] = same(expected, row['verified_target'])
        if not summary['target_match']:
            scores['target_correctness'] = 1
            scores['reasoning_validity'] = 2
            add_issue(issues, 'target_correctness', 'high',
                      {'independent_target': expected, 'candidate_target': row['verified_target']},
                      'Replace the target with the independently recomputed result; verify final-state and trace agreement.')
            required_revision = True
        if task['kind'] == 'causal':
            scores['shortcut_resistance'] = 3
            add_issue(issues, 'gate_balance', 'low',
                      'In the initial 24 training families, top-level not appears in 28 forward rows and 19 of those execute no event. This visible syntactic imbalance leaves a partial no-op shortcut despite paired rule reversals.',
                      'Balance apply/no-op within visible AST-root strata and assess counterfactual pairs jointly; do not rely on aggregate answer accuracy alone.')
            train = [p for p in peers if p['split'] == 'train' and p['semantic_task'].get('kind') == 'causal']
            if row['difficulty'] == 7:
                shape = normalized_tree(task['rule']), normalized_tree(task['precondition'])
                matching = [p['sample_id'] for p in train if (normalized_tree(p['semantic_task']['rule']), normalized_tree(p['semantic_task']['precondition'])) == shape]
                if matching:
                    scores['difficulty_quality'] = 2
                    scores['generalization_value'] = 2
                    add_issue(issues, 'difficulty_inflation', 'medium',
                              {'assigned_difficulty': 7, 'shared_training_rule_shape_examples': matching[:2],
                               'renderer': row['renderer'], 'horizon': len(task['events']),
                               'assessment': 'Larger magnitudes and vocabulary are exposed development shifts, but predicate structure and renderer are reused from training; the curriculum requires held-out predicate trees, magnitude ranges and renderer combinations for level 7.'},
                              'Use difficulty 4 for multistep rows (or an appropriate lower structural level), retain dev_challenge as a magnitude-shift label, and describe the limited generalization claim explicitly.')
                    required_revision = True
            if row['target_format'] == 'answer_only' and result['after'] == task['state']:
                scores['shortcut_resistance'] = 3
                add_issue(issues, 'diagnostic_limit', 'low',
                          'The final state equals the initial state, so an answer-only copy can pass without demonstrating gate evaluation. The linked explicit trace distinguishes the gates.',
                          'Evaluate paired trace accuracy and rule-flipped families jointly; avoid interpreting this answer-only row alone as evidence of applicability reasoning.')
            if len(task['events']) > 1 and all(s['calc']['operation'] == 'identity' for s in result['steps']):
                scores['difficulty_quality'] = min(scores['difficulty_quality'], 3)
                add_issue(issues, 'competency_coverage', 'low',
                          'Every event is gated off; this row checks repeated condition evaluation but exercises no changing intermediate state.',
                          'Retain as a control, and assess genuine state dependency on rows with executed intermediate updates.')
            add_issue(issues, 'generalization_scope', 'low',
                      'Pilot uses only x/y state, fixed arithmetic primitives and a small recurring AST vocabulary; exact execution does not establish hidden causal discovery or unrestricted language-rule reasoning.',
                      'Report scope as supplied-rule conditional execution; expand genuinely held-out AST compositions before making stronger generalization claims.')
        else:
            scores['generalization_value'] = 3
            scores['diversity'] = 3
            if task['kind'] in ('graph', 'blocks') and result['shortest_plan'] is not None and len(result['shortest_plan']) > 2 and row['difficulty'] <= 3:
                scores['difficulty_quality'] = 3
                add_issue(issues, 'difficulty_boundary', 'low',
                          {'structural_difficulty': row['difficulty'], 'optimal_plan_moves': len(result['shortest_plan']),
                           'assessment': 'The fresh optimum is a genuine dependent multiaction sequence, even when the submitted plan is short. Its search/replay complexity is richer than entity renaming alone.'},
                          'Keep the conservative label if difficulty measures submitted-plan replay, but record optimal-goal distance separately and calibrate task difficulty against the full required output.')
            if row['difficulty'] == 7:
                scores['difficulty_quality'] = 2
                scores['generalization_value'] = 2
                if task['kind'] in ('graph', 'blocks'):
                    assessment = 'Larger held-out object counts are useful, but graph motifs/block plan modes and both renderers recur in training; the curriculum level 7 requires held-out graph motifs/object range crossed with a held-out renderer.'
                    fix = 'Relabel as structural difficulty 4 (or 3 for an initially satisfied partial goal) while retaining exposed size-shift dev_challenge status; reserve level 7 for the declared crossed structural/renderer holdout.'
                else:
                    assessment = 'Larger worlds retain the same candidate corruption patterns, ranking objective and six plan modes as training; no held-out corruption producer, goal function or mechanism composition is demonstrated.'
                    fix = 'Relabel difficulty 5 (or 4 where appropriate) and retain a size-shift development label; create new held-out corruption producers and explicit goal-function/composition shifts before using level 7.'
                add_issue(issues, 'difficulty_inflation', 'medium',
                          {'assigned_difficulty': 7, 'renderer': row['renderer'],
                           'object_count': len(task.get('nodes', task.get('on', task.get('world', {}).get('nodes', [])))),
                           'assessment': assessment}, fix)
                required_revision = True
            if task['kind'] == 'critic':
                scores['shortcut_resistance'] = 3
                add_issue(issues, 'corruption_diversity', 'low',
                          'Candidates use recurring positive fuel-claim offsets, start-node self-moves and repeated intermediate-node moves. Shuffled order, equal-length good/bad claims and explicit ties reduce simple style cues, but corruption signatures remain narrow.',
                          'Retain matched controls and expand to underclaims, wrong-node claims, later missing edges and non-self-loop failures before reporting robust critic generalization.')
            if task['kind'] == 'graph' and 'lexicographic_tie' in row['skill_tags'] and len(task['edges']) == len(task['nodes']) - 1:
                add_issue(issues, 'competency_tag', 'low',
                          'The graph is a single directed chain and has no alternative shortest sequence, although lexicographic_tie is listed in skill_tags.',
                          'Remove the tie tag from this chain row or add a feasible equal-length alternative; use the genuine tie rows to measure tie breaking.')
            add_issue(issues, 'generalization_scope', 'low',
                      'Exact bounded dynamics and explicit objectives support local simulation and plan criticism; broader relational or value-model generalization needs separate held-out mechanism evidence.',
                      'Report bounded task scope and mechanism-specific validation; do not infer generic planning or successor-feature guarantees.')
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        scores['semantic_correctness'] = 1
        scores['instruction_clarity'] = 2
        scores['target_correctness'] = 1
        add_issue(issues, 'malformed_or_underspecified_task', 'high', str(error), 'Repair task and visible contract, then regenerate and independently review the target.')
        required_revision = True
    decision = 'REVISE' if required_revision else 'ACCEPT'
    return {'receipt_id': 'expert-' + row['sample_id'] + '-' + row['record_hash'][:12],
            'sample_id': row['sample_id'], 'record_hash': row['record_hash'],
            'reviewer_role': ROLE, 'reviewer_execution_id': EXECUTION,
            'review_status': 'REVISE' if required_revision else 'ACCEPTED',
            'scores': scores,
            'critical_defects': [i for i in issues if i['severity'] in ('high', 'critical')],
            'issues': issues, 'decision': decision, 'confidence': 0.99 if summary.get('target_match') is False else 0.97,
            'evidence': summary, 'review_method': 'Independent visible-prompt recomputation plus actual-agent semantic inspection of documented mechanism representatives.'}


def main():
    OUT.mkdir(exist_ok=True)
    version = sys.argv[1] if len(sys.argv) > 1 else 'pilot-v1'
    require(version in ('pilot-v1', 'pilot-v2'), 'unsupported review version')
    paths = [BASE / f'data/{track}/candidate_renderings/{version}.jsonl' for track in ('RCM-001', 'RGM-001', 'RVM-001')]
    rows = [json.loads(line) for path in paths if path.exists() for line in path.read_text().splitlines() if line]
    receipts = [review(row, rows) for row in rows]
    (OUT / (version + '.jsonl')).write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in receipts))
    calibration_path = BASE / 'reviews/reviewer_calibration/blind_candidates.jsonl'
    calibration = [json.loads(line) for line in calibration_path.read_text().splitlines() if line]
    calibration_receipts = [review(row, rows) for row in calibration]
    # Keep first blind calibration verdicts immutable after initial completion.
    initial = OUT / 'calibration_initial.jsonl'
    if not initial.exists():
        initial.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in calibration_receipts))
    counts = collections.Counter((r['sample_id'].split('-pilot')[0], r['decision']) for r in receipts)
    print(json.dumps({'pilot_rows': len(receipts), 'counts': {str(k): v for k, v in counts.items()},
                      'calibration_initial': [{'sample_id': r['sample_id'], 'decision': r['decision']} for r in calibration_receipts]}, indent=2))


if __name__ == '__main__':
    main()
