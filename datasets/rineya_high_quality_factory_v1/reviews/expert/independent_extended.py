"""Independent implementations authored from EXTENDED_SEMANTICS and visible prompts.

Reads public contracts and candidate rows only. No producer, reference, or formal
oracle is imported. The only shared code is this same reviewer's type-equality,
receipt utilities and graph/block verifier from its earlier independent review.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from independent_review import BASE, OUT, EXECUTION, ROLE, SCORE_KEYS, require, integer, same, add_issue, review

TRACKS = ('RSM-001', 'RMM-001', 'REM-001', 'RBM-001', 'RJP-001', 'RDP-001', 'ROM-001')


def scalar(value):
    return value is None or type(value) in (str, int, bool)


def strings(values, nonempty=False):
    return isinstance(values, list) and (not nonempty or bool(values)) and all(type(v) is str for v in values) and len(set(values)) == len(values)


def scalar_key(value):
    require(scalar(value), 'invalid scalar value')
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def skills(task):
    state = task['state'].copy()
    require(set(state) == {'x', 'y'} and all(integer(v) for v in state.values()), 'invalid registers')
    procedures = task['procedures']
    require(isinstance(procedures, dict) and all(type(n) is str for n in procedures), 'invalid procedure registry')
    called_by = collections.defaultdict(set)

    def inspect(instructions, caller):
        require(isinstance(instructions, list), 'instruction body must be a list')
        for instruction in instructions:
            op = instruction['op']
            if op in ('add', 'scale'):
                require(set(instruction) == {'op', 'register', 'value'} and instruction['register'] in state and integer(instruction['value']), 'invalid arithmetic instruction')
            elif op == 'swap':
                require(set(instruction) == {'op'}, 'invalid swap')
            elif op == 'call':
                require(set(instruction) == {'op', 'name'} and instruction['name'] in procedures, 'invalid call')
                called_by[caller].add(instruction['name'])
            elif op == 'if':
                require(set(instruction) == {'op', 'test', 'then', 'else'}, 'invalid conditional')
                test = instruction['test']
                require(set(test) == {'register', 'cmp', 'value'} and test['register'] in state and test['cmp'] in ('lt', 'eq', 'ge') and integer(test['value']), 'invalid conditional test')
                inspect(instruction['then'], caller)
                inspect(instruction['else'], caller)
            else:
                raise ValueError('unknown instruction')

    inspect(task['main'], None)
    for name, body in procedures.items():
        inspect(body, name)
    seen, active = set(), set()

    def acyclic(name):
        require(name not in active, 'cyclic calls, including unused branches')
        if name in seen:
            return
        active.add(name)
        for callee in called_by[name]:
            acyclic(callee)
        active.remove(name)
        seen.add(name)
    for name in procedures:
        acyclic(name)
    calls, conditions = [], []
    primitive_count = 0

    def execute(instructions):
        nonlocal primitive_count
        for instruction in instructions:
            op = instruction['op']
            if op == 'call':
                calls.append(instruction['name'])
                execute(procedures[instruction['name']])
            elif op == 'if':
                test = instruction['test']
                actual, threshold = state[test['register']], test['value']
                passes = actual < threshold if test['cmp'] == 'lt' else actual == threshold if test['cmp'] == 'eq' else actual >= threshold
                conditions.append(passes)
                execute(instruction['then'] if passes else instruction['else'])
            else:
                primitive_count += 1
                if op == 'add':
                    state[instruction['register']] += instruction['value']
                elif op == 'scale':
                    state[instruction['register']] *= instruction['value']
                else:
                    state['x'], state['y'] = state['y'], state['x']
    execute(task['main'])
    return {'after': state, 'executed_primitives': primitive_count, 'call_order': calls, 'conditions': conditions}


def modules(task):
    registry = task['modules']
    require(strings([m['module_id'] for m in registry]), 'invalid module IDs')
    for m in registry:
        require(set(m) == {'module_id', 'capabilities', 'priority'} and strings(m['capabilities']) and integer(m['priority']), 'invalid module')
    memory = task['memory'].copy()
    require(all(type(k) is str and scalar(v) for k, v in memory.items()), 'invalid memory')
    require(strings(task['protected']) and all(k in memory for k in task['protected']), 'invalid protected keys')
    protected = set(task['protected'])
    requests = task['requests']
    require(strings([r['request_id'] for r in requests]), 'invalid request IDs')
    routes = []
    for request in requests:
        require(set(request) == {'request_id', 'needs', 'writes'} and strings(request['needs']), 'invalid request')
        for write in request['writes']:
            require(set(write) == {'key', 'value'} and type(write['key']) is str and scalar(write['value']), 'invalid write')
        eligible = [m for m in registry if set(request['needs']) <= set(m['capabilities'])]
        selected = min(eligible, key=lambda m: (m['priority'], m['module_id']))['module_id'] if eligible else None
        blocked = []
        if selected is not None:
            for index, write in enumerate(request['writes']):
                if write['key'] in protected:
                    blocked.append(index)
                else:
                    memory[write['key']] = write['value']
        routes.append({'request_id': request['request_id'], 'selected_module': selected,
                       'eligible_modules': sorted(m['module_id'] for m in eligible), 'blocked_writes': blocked})
    return {'routes': routes, 'memory': memory}


def episodic(task):
    events, queries = task['events'], task['queries']
    require(strings([q['query_id'] for q in queries]), 'invalid query IDs')
    for event in events:
        require(set(event) == {'key', 'version', 'at', 'op', 'value'} and type(event['key']) is str and integer(event['version']) and event['version'] >= 0 and integer(event['at']) and event['at'] >= 0 and event['op'] in ('put', 'delete') and scalar(event['value']), 'invalid event')
        require(event['op'] != 'delete' or event['value'] is None, 'delete value must be null')
    results = []
    for query in queries:
        require(set(query) == {'query_id', 'key', 'as_of'} and type(query['key']) is str and integer(query['as_of']) and query['as_of'] >= 0, 'invalid retrieval query')
        eligible = [(e['version'], e['at'], index, e) for index, e in enumerate(events) if e['key'] == query['key'] and e['at'] <= query['as_of']]
        picked = max(eligible, key=lambda e: e[:3]) if eligible else None
        results.append({'query_id': query['query_id'], 'key': query['key'],
                        'status': 'absent' if picked is None else 'deleted' if picked[3]['op'] == 'delete' else 'present',
                        'value': None if picked is None or picked[3]['op'] == 'delete' else picked[3]['value'],
                        'chosen_version': None if picked is None else picked[0],
                        'chosen_event': None if picked is None else picked[2]})
    return {'results': results}


def belief(task):
    worlds = task['worlds']
    require(strings([w['world_id'] for w in worlds]), 'invalid possible-world IDs')
    keys = set(worlds[0]['facts']) if worlds else set()
    for world in worlds:
        require(set(world) == {'world_id', 'facts'} and set(world['facts']) == keys and all(type(k) is str and scalar(v) for k, v in world['facts'].items()), 'invalid possible-world facts')
    possible, sizes = worlds[:], []
    for observation in task['observations']:
        require(set(observation) == {'fact', 'equals'} and observation['fact'] in keys and scalar(observation['equals']), 'invalid observation')
        possible = [w for w in possible if same(w['facts'][observation['fact']], observation['equals'])]
        sizes.append(len(possible))
    queries = task['queries']
    require(strings([q['query_id'] for q in queries]), 'invalid belief query IDs')
    results = []
    for query in queries:
        require(set(query) == {'query_id', 'fact'} and query['fact'] in keys, 'invalid belief query')
        encoded = collections.Counter(scalar_key(w['facts'][query['fact']]) for w in possible)
        values = [{'value': json.loads(k), 'count': encoded[k]} for k in sorted(encoded)]
        entailed = bool(possible) and len(values) == 1
        results.append({'query_id': query['query_id'], 'counts': values, 'distinct_values': len(values),
                        'entailed': entailed, 'guaranteed': values[0]['value'] if entailed else None,
                        'can_reduce_uncertainty': len(values) > 1})
    return {'surviving_worlds': sorted(w['world_id'] for w in possible), 'belief_sizes': sizes, 'queries': results}


def predictive_abstraction(task):
    states, actions = task['states'], task['actions']
    require(strings(states, True) and strings(actions, True), 'invalid states or actions')
    require(set(task['observable']) == set(states) and all(scalar(v) for v in task['observable'].values()), 'invalid observable map')
    require(set(task['transition']) == set(states), 'invalid transition domain')
    for state in states:
        require(set(task['transition'][state]) == set(actions) and all(v in states for v in task['transition'][state].values()), 'transition not total')
    def partition(signature):
        grouped = collections.defaultdict(list)
        for state in sorted(states):
            grouped[signature(state)].append(state)
        return sorted(grouped.values())
    classes = partition(lambda s: scalar_key(task['observable'][s]))
    while True:
        index = {s: n for n, members in enumerate(classes) for s in members}
        refined = partition(lambda s: (scalar_key(task['observable'][s]), tuple(index[task['transition'][s][a]] for a in sorted(actions))))
        if refined == classes:
            break
        classes = refined
    transitions, observables = [], []
    for n, members in enumerate(classes):
        representative = members[0]
        observables.append({'class_id': n, 'value': task['observable'][representative]})
        for action in sorted(actions):
            transitions.append({'class_id': n, 'action': action, 'next_class': index[task['transition'][representative][action]]})
    return {'classes': classes, 'state_to_class': index, 'abstract_transitions': transitions, 'abstract_observable': observables}


def hamming(left, right):
    return sum(a != b for a, b in zip(left, right))


def constraint_repair(task):
    alphabet, slots = task['alphabet'], task['slots']
    require(strings(alphabet, True) and integer(slots) and slots > 0 and integer(task['min_distance']) and task['min_distance'] >= 0, 'invalid assignment universe')
    candidates = task['candidates']
    require(strings([c['candidate_id'] for c in candidates]), 'invalid repair candidate IDs')
    for candidate in candidates:
        require(set(candidate) == {'candidate_id', 'assignment'} and isinstance(candidate['assignment'], list) and len(candidate['assignment']) == slots and all(v in alphabet for v in candidate['assignment']), 'invalid candidate assignment')
    for c in task['constraints']:
        kind = c['kind']
        if kind in ('equal', 'different'):
            require(set(c) == {'kind', 'positions'} and len(c['positions']) == 2 and all(integer(v) and 0 <= v < slots for v in c['positions']), 'invalid binary constraint')
        elif kind == 'allowed':
            require(set(c) == {'kind', 'position', 'values'} and integer(c['position']) and 0 <= c['position'] < slots and all(v in alphabet for v in c['values']), 'invalid membership constraint')
        elif kind == 'count':
            require(set(c) == {'kind', 'value', 'min', 'max'} and c['value'] in alphabet and integer(c['min']) and integer(c['max']), 'invalid count constraint')
        else:
            raise ValueError('unknown constraint')
    def feasible(assignment):
        for c in task['constraints']:
            kind = c['kind']
            if kind == 'equal' and assignment[c['positions'][0]] != assignment[c['positions'][1]]:
                return False
            if kind == 'different' and assignment[c['positions'][0]] == assignment[c['positions'][1]]:
                return False
            if kind == 'allowed' and assignment[c['position']] not in c['values']:
                return False
            if kind == 'count' and not c['min'] <= assignment.count(c['value']) <= c['max']:
                return False
        return True
    universe = [a for a in itertools.product(sorted(alphabet), repeat=slots) if feasible(a)]
    repairs = []
    for candidate in candidates:
        original = candidate['assignment']
        selected = min(universe, key=lambda a: (hamming(a, original), a)) if universe else None
        repairs.append({'candidate_id': candidate['candidate_id'], 'original_valid': feasible(original),
                        'edit_distance': None if selected is None else hamming(selected, original),
                        'repaired': None if selected is None else list(selected)})
    accepted = []
    for repair in sorted((r for r in repairs if r['repaired'] is not None), key=lambda r: (r['edit_distance'], r['candidate_id'])):
        if all(hamming(repair['repaired'], previous['repaired']) >= task['min_distance'] for previous in accepted):
            accepted.append(repair)
    return {'feasible_count': len(universe), 'repairs': repairs, 'diverse_selection': [r['candidate_id'] for r in accepted]}


def agent_history(task):
    objects, places = task['objects'], task['places']
    require(strings(objects, True) and strings(places, True), 'invalid objects or places')
    actual = task['initial_locations'].copy()
    require(set(actual) == set(objects) and all(v in places for v in actual.values()), 'invalid initial locations')
    agents = task['agents']
    require(strings([a['agent_id'] for a in agents]), 'invalid agent IDs')
    beliefs = {}
    for agent in agents:
        require(set(agent) == {'agent_id', 'initial_beliefs', 'object_priority'} and set(agent['initial_beliefs']) == set(objects) and all(v is None or v in places for v in agent['initial_beliefs'].values()) and strings(agent['object_priority']) and all(v in objects for v in agent['object_priority']), 'invalid supplied agent state')
        beliefs[agent['agent_id']] = agent['initial_beliefs'].copy()
    for event in task['events']:
        require(set(event) == {'object', 'to', 'observed_by'} and event['object'] in objects and event['to'] in places and strings(event['observed_by']) and all(a in beliefs for a in event['observed_by']), 'invalid observation event')
        actual[event['object']] = event['to']
        for observer in event['observed_by']:
            beliefs[observer][event['object']] = event['to']
    results = []
    for agent in agents:
        belief_map = beliefs[agent['agent_id']]
        search = next((o for o in agent['object_priority'] if belief_map[o] is not None), None)
        results.append({'agent_id': agent['agent_id'], 'beliefs': belief_map,
                        'false_belief_objects': sorted(o for o in objects if belief_map[o] is not None and belief_map[o] != actual[o]),
                        'unknown_objects': sorted(o for o in objects if belief_map[o] is None),
                        'action': {'type': 'abstain'} if search is None else {'type': 'search', 'object': search, 'place': belief_map[search]}})
    return {'actual_locations': actual, 'agents': results}


SOLVERS = {f.__name__: f for f in (skills, modules, episodic, belief, predictive_abstraction, constraint_repair, agent_history)}
SCOPE = {
    'skills': 'Measures execution of supplied acyclic procedures in shared registers; no learned hierarchy, initiation/termination discovery, argument learning or interruption recovery is demonstrated.',
    'modules': 'Measures routing from declared capabilities and protecting explicit memory keys; no empirical expert competence, learned routing or continual-learning retention is measured.',
    'episodic': 'Measures versioned as-of retrieval from a visible finite ledger; no persistent memory storage, learned write policy or compression is measured.',
    'belief': 'Measures exact deterministic possible-world filtering and counts; no Bayes posterior, introspective confidence, entropy or inspection utility is measured.',
    'predictive_abstraction': 'Measures canonical deterministic transition equivalence for future observables; no learned latent encoder, training loss, masked-feature model or energy function is measured.',
    'constraint_repair': 'Measures finite Hamming-minimal assignment repair and a declared greedy diversity rule; no trained diffusion sampler, maximum packing or plan-prefix preservation is measured.',
    'agent_history': 'Measures separate observation-conditioned beliefs and a fully stated search policy; no hidden intentions, emotions, promises, cooperation objective or second-order belief is inferred.'}
PATTERN = {
    'skills': 'A small named procedure library and recurring main templates limit composition breadth; unused supplied procedures and scale-by-zero paths can create behavioral similarity across distinct worlds.',
    'modules': 'Requests repeatedly contain an unsupported middle request and an empty-needs request; write key/value patterns and the protected anchor structure recur. Unique initial protected values do not by themselves establish new routing mechanisms.',
    'episodic': 'Query cutoffs, version numbers and value tokens recur across reordered histories; input-index evidence makes ordering relevant but broader temporal/conflict patterns are not tested.',
    'belief': 'A small repeated fact/value palette is crossed with a few observation modes; explicit true-versus-1 and entailed-null controls are useful but not evidence of general uncertainty reasoning.',
    'predictive_abstraction': 'A small deterministic automaton library covers constant classes, typed observables and refinement; state-name shuffles improve canonicalization coverage but training of a representation is outside scope.',
    'constraint_repair': 'Constraint types and initial assignment patterns recur; candidate-ID/order changes can make exact worlds unique while leaving the same underlying repair problem.',
    'agent_history': 'A repeated finite object-movement scaffold and first-known-object policy limit social breadth; names and event counts vary but no underdetermined human behavior is labeled.'}


def receipt(row, contract_sections):
    issues, result = [], None
    scores = dict(zip(SCORE_KEYS, (5, 5, 5, 5, 3, 4, 3, 3)))
    revision = False
    try:
        prefix, payload = row['rendered_input'].split('Task JSON:', 1)
        task = json.loads(payload.strip())
        require(task['kind'] in SOLVERS, 'unknown extended kind')
        if not same(task, row['semantic_task']):
            add_issue(issues, 'prompt_payload_mismatch', 'high', 'Student-visible task differs from the canonical semantic_task.', 'Align prompt and canonical task, then reverify the target.')
            revision = True
        if contract_sections[task['kind']] not in prefix:
            scores['instruction_clarity'] = 2
            add_issue(issues, 'incomplete_visible_contract', 'high', 'Full public mechanism contract is absent from the visible prefix.', 'Include every output, tie-breaking and scalar-typing rule in the prompt.')
            revision = True
        require('type-sensitive' in prefix and 'No floating point or coercion is used.' in prefix, 'missing visible scalar typing rule')
        result = SOLVERS[task['kind']](task)
        if not same(result, row['verified_target']):
            scores['target_correctness'] = 1
            scores['reasoning_validity'] = 2
            add_issue(issues, 'target_correctness', 'high', {'independent_target': result, 'candidate_target': row['verified_target']}, 'Replace the target with the independent execution result and re-review.')
            revision = True
        if row['difficulty'] == 7:
            scores['difficulty_quality'] = 2
            add_issue(issues, 'difficulty_inflation', 'medium', 'Finite repeated proxy mechanisms do not demonstrate the declared level-7 holdouts.', 'Use measured structural difficulty and separate exposed challenge-axis metadata.')
            revision = True
        if 'structural_difficulty' in row and row['difficulty'] != row['structural_difficulty']:
            add_issue(issues, 'inconsistent_difficulty_metadata', 'medium', 'difficulty and structural_difficulty differ.', 'Use a consistent structural assessment.')
            revision = True
        kind = task['kind']
        add_issue(issues, 'proxy_scope', 'low', SCOPE[kind], 'Keep evaluation claims bounded to the supplied executable behavior and run separate learning/system experiments for broader hypotheses.')
        rotation = '-rotation' in row['sample_id']
        pattern = PATTERN[kind]
        if rotation and kind == 'modules':
            pattern = 'Unsupported, empty-needs and declared-capability requests now occupy every request ID/position through within-family rotations. The fixed-position shortcut is addressed, while capability/write/protection fixtures remain narrow.'
        elif rotation and kind == 'episodic':
            pattern = 'Every semantic query now occupies every query ID/position through within-family rotations. The fixed-position shortcut is addressed, while cutoff/version/value patterns recur and no new retrieval mechanism is added.'
        add_issue(issues, 'template_diversity', 'low', pattern, 'Report family-level rather than rendering-level evidence and expand structurally different compositions/edge cases before stronger generalization claims.')
        if kind in ('modules', 'episodic', 'agent_history'):
            scores['diversity'] = 2
        if kind in ('modules', 'episodic'):
            scores['shortcut_resistance'] = 3 if rotation else 2
        if rotation:
            add_issue(issues, 'lineage_discipline', 'low',
                      'Cyclic variants share an underlying family. Request rotations can change write chronology; query rotations change output positions/IDs while preserving the per-query resolution rule. These are matched variants, not new independent semantic parents.',
                      'Keep all rotations and renderers in the same split and aggregate outcomes by the underlying family.')
        if kind == 'predictive_abstraction':
            scores['shortcut_resistance'] = 4
        if kind == 'episodic' and row['split'] == 'dev_challenge':
            scores['generalization_value'] = 2
            add_issue(issues, 'added_history_relevance', 'low',
                      'The added queried-key event at19/version2 is excluded by as_of14 and is superseded by the at20/version4 event for as_of21. The new other-key events also cannot affect the supplied query results. Longer history is present, but these added records do not create a new decisive retrieval dependency.',
                      'Include an as_of19 query to test that the older version cannot displace the higher-version tombstone/put, and vary decisive update patterns before claiming history-depth generalization.')
        if kind == 'skills' and len(result['conditions']) > 1 and row['difficulty'] <= 3:
            scores['difficulty_quality'] = 3
            add_issue(issues, 'difficulty_boundary', 'low', 'Several procedure calls/conditions must use updated shared state, beyond a single conditional initiation.', 'Calibrate complexity using executed primitive count, call depth and state dependencies; retain conservative labels if explicitly defined.')
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        scores['semantic_correctness'] = 1
        scores['target_correctness'] = 1
        scores['instruction_clarity'] = 2
        add_issue(issues, 'malformed_or_underspecified_task', 'high', str(error), 'Repair the visible task/contract and regenerate the target before re-review.')
        revision = True
    return {'receipt_id': 'expert-' + row['sample_id'] + '-' + row['record_hash'][:12],
            'sample_id': row['sample_id'], 'record_hash': row['record_hash'],
            'reviewer_role': ROLE, 'reviewer_execution_id': EXECUTION,
            'review_status': 'REVISE' if revision else 'ACCEPTED', 'scores': scores,
            'critical_defects': [i for i in issues if i['severity'] in ('high', 'critical')],
            'issues': issues, 'decision': 'REVISE' if revision else 'ACCEPT', 'confidence': 0.97,
            'evidence': {'independent_target': result, 'target_match': result is not None and same(result, row['verified_target']),
                         'mechanism': row['semantic_task'].get('kind'), 'challenge_axes': row.get('challenge_axes', [])},
            'review_method': 'Own prompt-derived implementation plus actual-agent semantic mechanism/edge-case inspection; no external oracle used.'}


def main():
    public = (BASE / 'schemas/EXTENDED_SEMANTICS.md').read_text()
    sections = {m.group(1): m.group(2).strip() for m in re.finditer(r'## `kind=([^`]+)`\n\n(.*?)(?=\n## |\Z)', public, re.S)}
    version = sys.argv[1] if len(sys.argv) > 1 else 'extended-pilot-v2'
    require(version in ('extended-pilot-v2', 'extended-pilot-v3'), 'unsupported extended review version')
    tracks = TRACKS if version == 'extended-pilot-v2' else ('RMM-001', 'REM-001')
    extended = [json.loads(line) for track in tracks for line in (BASE / f'data/{track}/candidate_renderings/{version}.jsonl').read_text().splitlines() if line]
    receipts = [receipt(row, sections) for row in extended]
    (OUT / (version + '.jsonl')).write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in receipts))
    if version == 'extended-pilot-v3':
        print(json.dumps({'extended_count': len(receipts), 'extended_decisions': dict(collections.Counter(r['decision'] for r in receipts))}, indent=2))
        return
    relational = [json.loads(line) for line in (BASE / 'data/RGM-001/candidate_renderings/pilot-v3.jsonl').read_text().splitlines() if line]
    rgm_receipts = [review(row, relational) for row in relational]
    for row, r in zip(relational, rgm_receipts):
        if '-matched-' in row['sample_id'] and row['difficulty'] == 3 and not row['semantic_task']['plan']:
            r['scores']['difficulty_quality'] = 3
            add_issue(r['issues'], 'difficulty_boundary', 'low', 'The matched initially satisfied partial goal has an empty supplied and optimal plan, making it a simple boundary control despite structural difficulty3.', 'Treat as a control and calibrate empty-plan difficulty independently of object count.')
    (OUT / 'pilot-v3.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rgm_receipts))
    print(json.dumps({'extended_count': len(receipts), 'extended_decisions': dict(collections.Counter(r['decision'] for r in receipts)),
                      'relational_count': len(rgm_receipts), 'relational_decisions': dict(collections.Counter(r['decision'] for r in rgm_receipts))}, indent=2))


if __name__ == '__main__':
    main()
