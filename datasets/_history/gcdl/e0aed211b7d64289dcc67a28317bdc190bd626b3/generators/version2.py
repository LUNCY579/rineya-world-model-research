"""Bounded v2 goal triplets and explicit namespace surfaces; no publication.

V1 core code/configuration stay immutable. This adapter preserves A-H dynamics,
replaces C's one-goal construction, and never orders candidates using gold.
"""
from copy import deepcopy
import json
from pathlib import Path
import random

from . import build as v1
from ..environments.core import Blocksworld
from ..oracles.primary import evaluate as primary, UTILITY_DEFINITION
from ..oracles.independent import evaluate as independent
from ..pipelines.common import digest

GENERATOR_VERSION = 'generator-2.0'
CONFIG_VERSION = 'curriculum-2.0'
CONFIG_HASH = '1fa733236552daae035fc844be2b09381005d6343aa732483e19185f61ec92d6'
NAMESPACES = ('accounts', 'entries', 'blocks', 'registers', 'statuses', 'table')
ROOT = Path(__file__).resolve().parents[1]


class GenerationError(ValueError):
    """Explicit failed construction; diagnostics never disappear as exclusions."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = deepcopy(diagnostics)


def curriculum(config=None):
    cfg = json.loads((ROOT / 'configs/curriculum-v2.json').read_text()) if config is None else deepcopy(config)
    if digest(cfg) != CONFIG_HASH or cfg.get('version') != CONFIG_VERSION or cfg.get('generator_version') != GENERATOR_VERSION:
        raise ValueError('unsupported or changed v2 curriculum profile')
    return cfg


def _check_seed(seed, split, cfg):
    if type(seed) is not int or seed < 0 or split not in cfg['split_templates']:
        raise ValueError('nonnegative integer seed and declared split required')


def bundle_identity(task):
    """The goal-independent input bundle identity used by v2 rendering."""
    return digest({key: task[key] for key in ('environment', 'initial_state', 'rules',
                                             'constraints', 'candidates', 'utility_definition')})


def _dual(task):
    first, second = primary(task), independent(task)
    if first != second:
        raise GenerationError('v2 primary/independent disagreement', {'status': 'FAIL', 'reason': 'oracle_disagreement', 'task': deepcopy(task), 'primary': first, 'independent': second})
    return {'oracle': first, 'decision': v1.decision_labels(first)}


def _rename_action(action, maps, environment):
    result = deepcopy(action)
    fields = {'ledger': {'entry': 'entries', 'source': 'accounts', 'destination': 'accounts'},
              'register': {key: 'registers' for key in ('register', 'source', 'destination', 'left', 'right')},
              'blocksworld': {'block': 'blocks', 'destination': 'blocks'}}[environment]
    for field, namespace in fields.items():
        if field in result:
            value = result[field]
            if environment == 'blocksworld' and field == 'destination' and value in maps['table']:
                result[field] = _mapped(maps, 'table', value)
            else:
                result[field] = _mapped(maps, namespace, value)
    return result


def _mapped(maps, namespace, value):
    if value is None:
        return None
    if not isinstance(value, str) or value not in maps[namespace]:
        raise ValueError('unmapped ' + namespace + ' namespace reference')
    return maps[namespace][value]


def rename_task(task, maps):
    """Rename only explicitly identified namespaces, including invalid refs.

    Structural keys, opcodes, candidate IDs, numbers and prose are untouched.
    The same API with inverse maps restores exact source task values.
    """
    if set(maps) != set(NAMESPACES):
        raise ValueError('exact declared namespace map set required')
    result = deepcopy(task)
    environment = result['environment']
    def named(values, namespace):
        return {_mapped(maps, namespace, key): deepcopy(value) for key, value in values.items()}
    def support(value):
        return _mapped(maps, 'table' if value in maps['table'] else 'blocks', value)
    def state(value):
        value = deepcopy(value)
        if environment == 'ledger':
            value['accounts'] = named(value['accounts'], 'accounts')
            value['entries'] = {_mapped(maps, 'entries', key): {
                **entry, 'account': _mapped(maps, 'accounts', entry['account']),
                'status': _mapped(maps, 'statuses', entry['status'])} for key, entry in value['entries'].items()}
        elif environment == 'blocksworld':
            value['on'] = {_mapped(maps, 'blocks', key): support(destination) for key, destination in value['on'].items()}
        elif environment == 'register':
            value['registers'] = named(value['registers'], 'registers')
        else:
            raise ValueError('unsupported v2 namespace environment')
        return value
    if 'initial_state' in result:
        result['initial_state'] = state(result['initial_state'])
    if 'possible_worlds' in result:
        result['possible_worlds'] = [state(world) for world in result['possible_worlds']]
    goal, constraints = result['goal'], result['constraints']
    if environment == 'ledger':
        result['rules']['statuses'] = {role: _mapped(maps, 'statuses', value) for role, value in result['rules']['statuses'].items()}
        if 'balances' in goal:
            goal['balances'] = named(goal['balances'], 'accounts')
        if 'statuses' in goal:
            goal['statuses'] = {_mapped(maps, 'entries', key): _mapped(maps, 'statuses', value) for key, value in goal['statuses'].items()}
        if 'minimum_balances' in constraints:
            constraints['minimum_balances'] = named(constraints['minimum_balances'], 'accounts')
        if 'forbidden_statuses' in constraints:
            constraints['forbidden_statuses'] = [_mapped(maps, 'statuses', value) for value in constraints['forbidden_statuses']]
    elif environment == 'blocksworld':
        result['rules']['table'] = _mapped(maps, 'table', result['rules']['table'])
        goal['on'] = {_mapped(maps, 'blocks', key): support(value) for key, value in goal['on'].items()}
        if 'forbidden_on' in constraints:
            constraints['forbidden_on'] = [[_mapped(maps, 'blocks', block), support(destination)] for block, destination in constraints['forbidden_on']]
        if 'protected_blocks' in constraints:
            constraints['protected_blocks'] = [_mapped(maps, 'blocks', value) for value in constraints['protected_blocks']]
    elif environment == 'register':
        goal['values'] = named(goal['values'], 'registers')
        for field in ('minimum_values', 'maximum_values'):
            if field in constraints:
                constraints[field] = named(constraints[field], 'registers')
        if 'protected_registers' in constraints:
            constraints['protected_registers'] = [_mapped(maps, 'registers', value) for value in constraints['protected_registers']]
    for candidate in result['candidates']:
        candidate['plan'] = [_rename_action(action, maps, environment) for action in candidate['plan']]
    # The generated observation path grammar is explicit, rather than a
    # recursive replacement of arbitrary strings or semantic field names.
    for index, path in enumerate(result.get('observations', {}).get('missing', [])):
        parts = path.split('.')
        path_namespace = {'accounts': 'accounts', 'entries': 'entries', 'on': 'blocks', 'registers': 'registers'}
        if len(parts) >= 2 and parts[0] in path_namespace:
            parts[1] = _mapped(maps, path_namespace[parts[0]], parts[1])
            result['observations']['missing'][index] = '.'.join(parts)
    return result


def _namespace_names(task, extra_actions=()):
    refs = {namespace: set() for namespace in NAMESPACES}
    active = {namespace: set() for namespace in NAMESPACES}
    def add(namespace, values, is_active=False):
        values = [value for value in values if value is not None]
        if any(type(value) is not str for value in values):
            raise ValueError('namespace references must be strings')
        refs[namespace].update(values)
        if is_active:
            active[namespace].update(values)
    env = task['environment']
    worlds = [task['initial_state']] + task.get('possible_worlds', [])
    if env == 'ledger':
        add('statuses', task['rules']['statuses'].values(), True)
        for world in worlds:
            add('accounts', world['accounts'], True); add('entries', world['entries'], True)
            add('accounts', [entry['account'] for entry in world['entries'].values()])
            add('statuses', [entry['status'] for entry in world['entries'].values()])
        add('accounts', task['goal'].get('balances', {})); add('entries', task['goal'].get('statuses', {}))
        add('statuses', task['goal'].get('statuses', {}).values())
        add('accounts', task['constraints'].get('minimum_balances', {})); add('statuses', task['constraints'].get('forbidden_statuses', []))
    elif env == 'blocksworld':
        table = task['rules']['table']; add('table', [table], True)
        for world in worlds:
            add('blocks', world['on'], True); add('blocks', [value for value in world['on'].values() if value != table])
        add('blocks', task['goal']['on']); add('blocks', [value for value in task['goal']['on'].values() if value != table])
        for block, destination in task['constraints'].get('forbidden_on', []):
            add('blocks', [block] + ([] if destination == table else [destination]))
        add('blocks', task['constraints'].get('protected_blocks', []))
    elif env == 'register':
        for world in worlds:
            add('registers', world['registers'], True)
        add('registers', task['goal']['values'])
        for field in ('minimum_values', 'maximum_values', 'protected_registers'):
            add('registers', task['constraints'].get(field, {}))
    else:
        raise ValueError('unsupported v2 namespace environment')
    fields = {'ledger': {'entry': 'entries', 'source': 'accounts', 'destination': 'accounts'},
              'register': {key: 'registers' for key in ('register', 'source', 'destination', 'left', 'right')},
              'blocksworld': {'block': 'blocks', 'destination': 'blocks'}}[env]
    actions = [action for candidate in task['candidates'] for action in candidate['plan']] + list(extra_actions)
    for action in actions:
        for field, namespace in fields.items():
            if field in action and not (env == 'blocksworld' and field == 'destination' and action[field] == task['rules']['table']):
                add(namespace, [action[field]])
    for path in task.get('observations', {}).get('missing', []):
        parts = path.split('.')
        path_namespace = {'accounts': 'accounts', 'entries': 'entries', 'on': 'blocks', 'registers': 'registers'}
        if len(parts) >= 2 and parts[0] in path_namespace:
            add(path_namespace[parts[0]], [parts[1]])
    return refs, active


def _maps_for(task, parent, lexicon_split, presentation, cfg, extra_actions=()):
    refs, active = _namespace_names(task, extra_actions)
    maps = {}
    aliases = {'accounts': {'cedar': 'a', 'river': 'b'}, 'entries': {'tally': 'e'}}
    for namespace in NAMESPACES:
        varied = (presentation == 'entity_rename' and namespace in ('accounts', 'entries')) or (presentation == 'status_vocabulary' and namespace == 'statuses')
        salt = presentation if varied else 'base'
        pool = sorted(cfg['lexicons'][lexicon_split][namespace], key=lambda word: digest([parent, namespace, salt, word]))
        if namespace == 'statuses' and task['environment'] == 'ledger':
            known = [task['rules']['statuses'][role] for role in ('pending', 'settled', 'void')]
        else:
            known = sorted(active[namespace], key=lambda name: (aliases.get(namespace, {}).get(name, name), name))
        names = known + sorted(refs[namespace] - set(known))
        if len(names) > len(pool):
            raise ValueError('declared namespace pool exhausted')
        if varied and names:
            base_pool = sorted(cfg['lexicons'][lexicon_split][namespace], key=lambda word: digest([parent, namespace, 'base', word]))
            if pool[:len(names)] == base_pool[:len(names)]:
                pool = pool[1:] + pool[:1]
        maps[namespace] = dict(zip(names, pool))
    return maps


def restore_source_names(record, config=None):
    """Validate declared vocabulary custody and return exact source task.

    Unknown refs are restored too, retaining v1 semantic-fingerprint markers.
    No mapping is inferred from a target or passed to executable module text.
    """
    cfg = curriculum(config)
    vocabulary = record.get('metadata', {}).get('vocabulary', {})
    split, stratum = record['split'], vocabulary.get('stratum')
    lexicon_split, presentation = vocabulary.get('lexicon_split'), vocabulary.get('presentation')
    if vocabulary.get('version') != cfg['vocabulary_policy']['version'] or presentation not in ('base', 'entity_rename', 'status_vocabulary'):
        raise ValueError('unsupported vocabulary policy/presentation')
    if presentation != 'base' and (record['family'] != 'D' or record['metadata']['perturbation'].get('kind') != presentation):
        raise ValueError('vocabulary perturbation presentation requires its declared D control')
    if stratum == 'familiar_control':
        if (split != 'dev' or lexicon_split != 'train' or presentation != 'base'
                or record['family'] != 'D' or record['input']['environment'] != 'ledger'):
            raise ValueError('familiar vocabulary control is a separate DEV-only stratum')
    elif stratum != 'reserved_' + split or lexicon_split != split:
        raise ValueError('reserved vocabulary split/stratum mismatch')
    maps = vocabulary.get('maps', {})
    if set(maps) != set(NAMESPACES):
        raise ValueError('missing declared vocabulary namespace maps')
    inverse = {}
    for namespace, mapping in maps.items():
        if (not isinstance(mapping, dict) or len(set(mapping.values())) != len(mapping)
                or any(type(key) is not str or type(value) is not str for key, value in mapping.items())
                or not set(mapping.values()) <= set(cfg['lexicons'][lexicon_split][namespace])):
            raise ValueError('nonbijective or out-of-pool vocabulary mapping')
        inverse[namespace] = {value: key for key, value in mapping.items()}
    restored = rename_task(record['input'], inverse)
    if digest(restored) != vocabulary.get('source_namespace_task_sha256'):
        raise ValueError('vocabulary source-task round-trip hash mismatch')
    proof = record['targets'].get('bounded_proof')
    extra_actions = [] if not proof else [_rename_action(action, inverse, restored['environment']) for action in proof['action_alphabet']]
    expected = _maps_for(restored, record['provenance']['parent_world_id'], lexicon_split, presentation, cfg, extra_actions)
    if maps != expected:
        raise ValueError('vocabulary mapping not total or not declared parent assignment')
    return restored


def validate_vocabulary(record, config=None):
    restore_source_names(record, config)


def _parent(seed, source_parent):
    return 'worldv2_' + digest([GENERATOR_VERSION, seed, source_parent])[:24]


def _final_record(source, cfg, index, parent, lexicon_split=None, stratum=None,
                  presentation='base', preserve_order=False):
    record = deepcopy(source)
    split = record['split']
    task = deepcopy(record['input'])
    if not preserve_order and 'search' not in source['targets']:
        task['candidates'].sort(key=lambda candidate: digest([parent, 'candidate_order', candidate['id']]))
        if source['metadata']['perturbation'].get('kind') == 'order_paraphrase_distractor':
            task['candidates'].reverse()
    lexicon_split = lexicon_split or split
    proof = source['targets'].get('bounded_proof')
    extra = [] if not proof else proof['action_alphabet']
    maps = _maps_for(task, parent, lexicon_split, presentation, cfg, extra)
    renamed = rename_task(task, maps)
    targets = _dual(renamed)
    before = {item['id']: item for item in source['targets']['decision']['candidates']}
    after = {item['id']: item for item in targets['decision']['candidates']}
    if before != after:
        raise GenerationError('namespace mapping changed source exact labels', {'status': 'FAIL', 'reason': 'vocabulary_label_change', 'source': source})
    if 'search' in source['targets']:
        search = source['targets']['search']
        targets['search'] = v1._search(renamed, (search['cheap']['candidate_budget'], search['expanded']['candidate_budget']))
        if targets['search'] != search:
            raise GenerationError('namespace mapping changed matched search', {'status': 'FAIL', 'reason': 'search_change', 'source': source})
    if proof:
        actions = [_rename_action(action, maps, task['environment']) for action in proof['action_alphabet']]
        targets['bounded_proof'] = v1._bounded_proof(renamed, actions, proof['horizon'])
    spec = cfg['split_templates'][split]
    record['provenance'].update(generator_version=GENERATOR_VERSION, config_version=CONFIG_VERSION,
                                config_hash=digest(cfg), template_id=spec['template'],
                                template_group_id=spec['template'], lineage_id=spec['template'],
                                parent_world_id=parent, record_index=index)
    record['input'], record['targets'] = renamed, targets
    record['metadata'].update(counterfactual_group_id=parent,
                              protected_template_signature=v1.protected_template_signature(renamed),
                              vocabulary={'version': cfg['vocabulary_policy']['version'],
                                          'stratum': stratum or 'reserved_' + split,
                                          'lexicon_split': lexicon_split, 'presentation': presentation,
                                          'maps': maps, 'source_namespace_task_sha256': digest(task)})
    record['id'] = v1.record_id(record)
    validate_vocabulary(record, cfg)
    return record


def _legal_actions(on, rules):
    env = Blocksworld()
    actions = []
    for block in sorted(on):
        for destination in [rules['table']] + sorted(on):
            action = {'op': 'move', 'block': block, 'destination': destination}
            _, error = env.transition({'on': on}, action, rules)
            if error is None:
                actions.append(action)
    return actions


def _path_proposal(initial, rules, horizon, rng):
    state, path = deepcopy(initial), []
    for _ in range(horizon):
        actions = _legal_actions(state['on'], rules)
        if not actions:
            return None, None
        action = deepcopy(rng.choice(actions))
        state, error = Blocksworld().transition(state, action, rules)
        if error is not None:
            raise AssertionError('declared legal-path proposal became illegal')
        path.append(action)
    return path, state


def _goal_triplet(source, cfg, seed, split, index_start=0):
    base = source['input']
    horizon = cfg['split_templates'][split]['horizon']
    rng = random.Random(int(digest([GENERATOR_VERSION, seed, 'C-legal-path-proposals'])[:16], 16))
    chosen, seen, attempts = [], set(), 0
    for attempts in range(1, cfg['c_bundle']['max_proposals'] + 1):
        path, terminal = _path_proposal(base['initial_state'], base['rules'], horizon, rng)
        if path is None or terminal == base['initial_state'] or digest(terminal) in seen:
            continue
        seen.add(digest(terminal)); chosen.append((path, terminal, attempts))
        if len(chosen) == 3:
            break
    diagnostic = {'seed': seed, 'split': split, 'proposals_attempted': attempts,
                  'max_proposals': cfg['c_bundle']['max_proposals'], 'horizon': horizon,
                  'distinct_noninitial_endpoints': len(chosen), 'training_launched': False,
                  'publication': False}
    if len(chosen) < 3:
        return [], {**diagnostic, 'status': 'EXCLUDED', 'reason': 'insufficient_distinct_noninitial_endpoints'}
    roles = [('path_0', chosen[0][0]), ('path_1', chosen[1][0]), ('path_2', chosen[2][0]),
             ('prefix_0', chosen[0][0][:-1]), ('reverse_0', list(reversed(chosen[0][0]))), ('empty', [])]
    world_key = digest({'version': GENERATOR_VERSION, 'seed': seed, 'template': cfg['split_templates'][split]['template'],
                        'initial_state': base['initial_state'], 'rules': base['rules'], 'roles_and_plans': roles})
    candidates = [{'id': 'c_' + digest([world_key, role])[:24], 'plan': deepcopy(plan)} for role, plan in roles]
    candidates.sort(key=lambda candidate: digest([world_key, 'candidate_order', candidate['id']]))
    task = deepcopy(base)
    task['candidates'], task['utility_definition'] = candidates, deepcopy(UTILITY_DEFINITION)
    source_bundle = bundle_identity(task)
    parent = 'worldv2_' + source_bundle[:24]
    records = []
    for goal_index, (_, terminal, proposal_index) in enumerate(chosen):
        raw = deepcopy(source)
        raw['input'] = deepcopy(task); raw['input']['goal'] = deepcopy(terminal)
        raw['targets'] = _dual(raw['input'])
        raw['metadata'].update(tags=['multi_step_planning', 'goal_change_only', 'fixed_candidate_goal_triplet', 'horizon_' + str(horizon)],
                               perturbation={'kind': 'fixed_bundle_goal', 'label_relation': 'recompute'},
                               candidate_roles={'c_' + digest([world_key, role])[:24]: role for role, _ in roles},
                               source_goal_bundle_id=source_bundle, goal_index=goal_index,
                               construction={'proposal_index': proposal_index, 'proposals_attempted': attempts, 'max_proposals': 24})
        record = _final_record(raw, cfg, index_start + goal_index, parent, preserve_order=True)
        record['metadata']['goal_bundle_id'] = bundle_identity(record['input'])
        record['id'] = v1.record_id(record)
        records.append(record)
    best = [set(record['targets']['decision']['best_candidate_ids']) for record in records]
    if any(not values for values in best) or any(best[a] & best[b] for a in range(3) for b in range(a)):
        raise GenerationError('full-goal best sets overlap or lack feasible plans', {**diagnostic, 'status': 'FAIL', 'reason': 'goal_bundle_invariant', 'quarantine_records': records})
    return records, {**diagnostic, 'status': 'VALID', 'reason': None,
                     'parent_world_id': parent, 'goal_bundle_id': records[0]['metadata']['goal_bundle_id']}


def generate_goal_triplet(seed, split='dev', config=None):
    cfg = curriculum(config); _check_seed(seed, split, cfg)
    source = next(record for record in v1.generate_group(seed, config=cfg, split=split) if record['family'] == 'C')
    return _goal_triplet(source, cfg, seed, split)


def generate_vocabulary_pair(seed, split='dev', config=None):
    cfg = curriculum(config); _check_seed(seed, split, cfg)
    if split != 'dev':
        raise ValueError('matched familiar/reserved vocabulary control is DEV only')
    source = v1.generate_group(seed, config=cfg, split=split)[0]
    source['family'], source['difficulty'] = 'D', 'adversarial'
    parent = _parent(seed, source['provenance']['parent_world_id'])
    records = []
    for index, (lexicon, stratum) in enumerate((('train', 'familiar_control'), ('dev', 'reserved_dev'))):
        raw = deepcopy(source)
        raw['metadata'].update(tags=['vocabulary_matched_control', stratum],
                               perturbation={'kind': 'vocabulary_matched_control', 'label_relation': 'invariant'})
        records.append(_final_record(raw, cfg, index, parent, lexicon, stratum))
    return records, {'seed': seed, 'split': split, 'status': 'VALID', 'parent_world_id': parent,
                     'strata': ['familiar_control', 'reserved_dev'], 'publication': False, 'training_launched': False}


def generate_group(seed, config=None, split='train'):
    """Preserve core A-H dynamics, replace C, and apply v2 namespaces to all."""
    cfg = curriculum(config); _check_seed(seed, split, cfg)
    sources = v1.generate_group(seed, config=cfg, split=split)
    records, identity_map, parent_map = [], {}, {}
    for source in sources:
        source_parent = source['provenance']['parent_world_id']
        parent_map.setdefault(source_parent, _parent(seed, source_parent))
        if source['family'] == 'C':
            triplet, diagnostic = _goal_triplet(source, cfg, seed, split, len(records))
            if not triplet:
                raise GenerationError('v2 C parent construction inadequate', diagnostic)
            records.extend(triplet)
            continue
        kind = source['metadata']['perturbation'].get('kind')
        presentation = kind if kind in ('entity_rename', 'status_vocabulary') else 'base'
        record = _final_record(source, cfg, len(records), parent_map[source_parent], presentation=presentation)
        identity_map[source['id']] = record['id']
        records.append(record)
    for record in records:
        perturbation = record['metadata']['perturbation']
        if 'base_record_id' in perturbation:
            if perturbation['base_record_id'] not in identity_map:
                raise GenerationError('removed core task has a dangling control reference', {'status': 'FAIL', 'reason': 'dangling_control_reference'})
            perturbation['base_record_id'] = identity_map[perturbation['base_record_id']]
        transfer = record['metadata'].get('transfer', {})
        if 'paired_parent_world_id' in transfer:
            transfer['paired_parent_world_id'] = parent_map[transfer['paired_parent_world_id']]
        record['id'] = v1.record_id(record)
    return records
