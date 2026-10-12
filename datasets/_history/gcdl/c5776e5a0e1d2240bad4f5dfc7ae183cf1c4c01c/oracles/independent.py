"""Differential checker implemented separately from the primary simulator.

Does not import environments or primary. Blocksworld uses bottom-to-top stack
lists rather than support-edge transitions. Ledger uses entry tuples and balance
deltas. Exact aggregation and constraint/goal evaluation are duplicated here.
Both implementations share an engineer; agreement is differential evidence,
not engineering independence. External hard-coded review tests audit both.
"""
import copy
import json
from fractions import Fraction

ORACLE_VERSION = "independent-v1"
UTILITY = {'comparison': 'lexicographic',
    'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
    'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
    'all_infeasible': 'abstain; no best candidate'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(data, mandatory, allowed=None):
    require(type(data) is dict, "mapping required")
    allowed = mandatory if allowed is None else allowed
    require(set(mandatory) <= set(data) <= set(allowed), "missing/unknown keys")


def int_value(value, floor=None):
    require(type(value) is int and (floor is None or value >= floor), "integer required")


def named(data, nonempty=False):
    require(type(data) is dict and (bool(data) or not nonempty), "named dictionary required")
    require(all(type(k) is str and k != '' for k in data), "invalid entity name")


def declared_list(data, universe):
    require(type(data) is list and all(type(v) is str for v in data), "string list required")
    require(len(data) == len(set(data)) and set(data) <= set(universe), "duplicate/unknown name")


def interval(data):
    require(type(data) is list and len(data) == 2, "two bounds required")
    int_value(data[0]); int_value(data[1])
    require(data[0] <= data[1], "reversed bounds")


def numeric_map(data, names):
    named(data)
    require(set(data) <= set(names), "unknown goal/constraint entity")
    for value in data.values(): int_value(value)


def check_action(action, family):
    require(type(action) is dict and type(action.get('op')) is str, "action/op required")
    op = action['op']
    shapes = {
        'ledger': {'settle': ('op', 'entry'), 'void': ('op', 'entry'), 'transfer': ('op', 'source', 'destination', 'amount'), 'noop': ('op',)},
        'blocksworld': {'move': ('op', 'block', 'destination'), 'noop': ('op',)},
        'register': {'add': ('op', 'register', 'value'), 'multiply': ('op', 'register', 'value'), 'set': ('op', 'register', 'value'), 'copy': ('op', 'source', 'destination'), 'swap': ('op', 'left', 'right'), 'noop': ('op',)}}
    keys(action, shapes[family].get(op, ('op',)))
    for name, value in action.items():
        if name in ('amount', 'value'): int_value(value)
        else: require(type(value) is str and value != '', "action names must be nonempty strings")


def check_world(state, rule, goal, constraint, family):
    if family == 'ledger':
        keys(rule, ('statuses', 'void_effect', 'allow_overdraft', 'balance_bounds', 'action_costs'))
        keys(rule['statuses'], ('pending', 'settled', 'void'))
        status_values = list(rule['statuses'].values())
        require(all(type(v) is str and v for v in status_values) and len(set(status_values)) == 3, "invalid statuses")
        require(rule['void_effect'] in ('reverse', 'retain') and type(rule['allow_overdraft']) is bool, "invalid ledger rule")
        interval(rule['balance_bounds'])
        ops = ('settle', 'void', 'transfer', 'noop')
        keys(state, ('accounts', 'entries')); named(state['accounts'], True); named(state['entries'])
        lo, hi = rule['balance_bounds']
        for balance in state['accounts'].values():
            int_value(balance)
            require(lo <= balance <= hi and (rule['allow_overdraft'] or balance >= 0), "invalid starting balance")
        for entry in state['entries'].values():
            keys(entry, ('account', 'amount', 'status'))
            require(type(entry['account']) is str and entry['account'] in state['accounts'], "unknown entry account")
            require(type(entry['status']) is str and entry['status'] in status_values, "unknown status")
            int_value(entry['amount']); require(entry['amount'] != 0, "zero amount")
        keys(goal, (), ('balances', 'statuses')); require(bool(goal) and any(goal.values()), "empty goal")
        numeric_map(goal.get('balances', {}), state['accounts'])
        named(goal.get('statuses', {}))
        require(set(goal.get('statuses', {})) <= set(state['entries']) and all(type(v) is str and v in status_values for v in goal.get('statuses', {}).values()), "invalid status goal")
        keys(constraint, (), ('minimum_balances', 'forbidden_statuses', 'max_cost'))
        numeric_map(constraint.get('minimum_balances', {}), state['accounts'])
        declared_list(constraint.get('forbidden_statuses', []), status_values)
    elif family == 'register':
        keys(rule, ('value_bounds', 'overflow', 'operations', 'action_costs'))
        interval(rule['value_bounds'])
        require(rule['overflow'] in ('reject', 'clamp', 'wrap'), "invalid overflow")
        declared_list(rule['operations'], ('add', 'multiply', 'set', 'copy', 'swap', 'noop'))
        require(bool(rule['operations']), "empty operations")
        ops = rule['operations']
        keys(state, ('registers',)); named(state['registers'], True)
        lo, hi = rule['value_bounds']
        for value in state['registers'].values():
            int_value(value); require(lo <= value <= hi, "starting value outside bounds")
        keys(goal, ('values',)); numeric_map(goal['values'], state['registers']); require(bool(goal['values']), "empty goal")
        keys(constraint, (), ('minimum_values', 'maximum_values', 'protected_registers', 'max_cost'))
        numeric_map(constraint.get('minimum_values', {}), state['registers'])
        numeric_map(constraint.get('maximum_values', {}), state['registers'])
        declared_list(constraint.get('protected_registers', []), state['registers'])
    else:
        keys(rule, ('table', 'max_height', 'action_costs'))
        require(type(rule['table']) is str and rule['table'], "invalid table")
        int_value(rule['max_height'], 1)
        ops = ('move', 'noop')
        keys(state, ('on',)); named(state['on'], True)
        table = rule['table']; require(table not in state['on'], "table/block collision")
        require(all(type(v) is str and (v == table or v in state['on']) for v in state['on'].values()), "unknown support")
        support_used = [v for v in state['on'].values() if v != table]
        require(len(set(support_used)) == len(support_used), "branching stack")
        remaining = set(state['on'])
        for stack in stacks_from_edges(state['on'], table):
            require(len(stack) <= rule['max_height'], "height violation")
            remaining.difference_update(stack)
        require(not remaining, "cyclic support graph")
        keys(goal, ('on',)); named(goal['on'], True)
        require(set(goal['on']) <= set(state['on']), "unknown goal block")
        require(all(type(v) is str and v in set(state['on']) | {table} and k != v for k, v in goal['on'].items()), "invalid goal support")
        keys(constraint, (), ('forbidden_on', 'protected_blocks', 'max_cost'))
        pairs = constraint.get('forbidden_on', [])
        require(type(pairs) is list, "forbidden pairs must be list")
        require(all(type(p) is list and len(p) == 2 and type(p[0]) is str and type(p[1]) is str and p[0] in state['on'] and p[1] in set(state['on']) | {table} for p in pairs), "invalid forbidden pair")
        require(len(set(map(tuple, pairs))) == len(pairs), "duplicate pairs")
        declared_list(constraint.get('protected_blocks', []), state['on'])
    keys(rule['action_costs'], ops)
    for cost in rule['action_costs'].values(): int_value(cost, 0)
    if 'max_cost' in constraint: int_value(constraint['max_cost'], 0)


def stacks_from_edges(edges, table):
    """Enumerate anchored stacks; disconnected cycles remain unvisited."""
    above = {lower: block for block, lower in edges.items() if lower != table}
    stacks = []
    for base in sorted(block for block in edges if edges[block] == table):
        stack, seen = [base], {base}
        while stack[-1] in above:
            next_block = above[stack[-1]]
            require(next_block not in seen, "cyclic anchored stack")
            seen.add(next_block); stack.append(next_block)
        stacks.append(stack)
    return stacks


def encoded(state, family, rule):
    if family == 'ledger':
        return (dict(state['accounts']), {k: (v['account'], v['amount'], v['status']) for k, v in state['entries'].items()})
    if family == 'register': return dict(state['registers'])
    return stacks_from_edges(state['on'], rule['table'])


def decoded(internal, family, rule):
    if family == 'ledger':
        return {'accounts': dict(internal[0]), 'entries': {key: {'account': data[0], 'amount': data[1], 'status': data[2]} for key, data in internal[1].items()}}
    if family == 'register': return {'registers': dict(internal)}
    on = {}
    for stack in internal:
        for index, block in enumerate(stack): on[block] = rule['table'] if index == 0 else stack[index-1]
    return {'on': on}


def advance(internal, action, rule, family):
    """Return replacement private representation and canonical error code."""
    op = action['op']
    work = copy.deepcopy(internal)
    if family == 'ledger':
        if op not in ('settle', 'void', 'transfer', 'noop'): return internal, 'unsupported_operation'
        if op == 'noop': return work, None
        accounts, entries = work
        delta = {key: 0 for key in accounts}
        if op == 'transfer':
            source, destination, amount = action['source'], action['destination'], action['amount']
            if source not in accounts or destination not in accounts: return internal, 'unknown_account'
            if source == destination or amount <= 0: return internal, 'invalid_transfer'
            delta[source] = -amount; delta[destination] = amount
        else:
            key = action['entry']
            if key not in entries: return internal, 'unknown_entry'
            account, amount, status = entries[key]
            pending, settled, voided = (rule['statuses'][x] for x in ('pending', 'settled', 'void'))
            if op == 'settle':
                if status != pending: return internal, 'entry_not_pending'
                delta[account] = amount; entries[key] = (account, amount, settled)
            else:
                if status == voided: return internal, 'entry_already_void'
                delta[account] = -amount if status == settled and rule['void_effect'] == 'reverse' else 0
                entries[key] = (account, amount, voided)
        values = {key: balance + delta[key] for key, balance in accounts.items()}
        if not all(rule['balance_bounds'][0] <= value <= rule['balance_bounds'][1] for value in values.values()): return internal, 'balance_out_of_bounds'
        if rule['allow_overdraft'] is False and min(values.values()) < 0: return internal, 'overdraft_forbidden'
        return (values, entries), None
    if family == 'register':
        if op not in rule['operations']: return internal, 'unsupported_operation'
        if op == 'noop': return work, None
        operands = {key: value for key, value in action.items() if key not in ('op', 'value')}
        if not set(operands.values()) <= set(work): return internal, 'unknown_register'
        if op == 'add': work[action['register']] += action['value']
        elif op == 'multiply': work[action['register']] *= action['value']
        elif op == 'set': work[action['register']] = action['value']
        elif op == 'copy': work[action['destination']] = internal[action['source']]
        elif op == 'swap':
            work[action['left']] = internal[action['right']]
            work[action['right']] = internal[action['left']]
        low, high = rule['value_bounds']
        for register, value in list(work.items()):
            if value < low or value > high:
                if rule['overflow'] == 'reject': return internal, 'value_out_of_bounds'
                if rule['overflow'] == 'clamp': work[register] = low if value < low else high
                else:
                    width = high-low+1
                    work[register] = value - ((value-low)//width)*width
        return work, None
    if op == 'noop': return work, None
    if op != 'move': return internal, 'unsupported_operation'
    block, destination, table = action['block'], action['destination'], rule['table']
    positions = {b: (si, bi) for si, stack in enumerate(internal) for bi, b in enumerate(stack)}
    if block not in positions or (destination != table and destination not in positions): return internal, 'unknown_block'
    if block == destination: return internal, 'self_support'
    source_index, block_index = positions[block]
    old_support = table if block_index == 0 else internal[source_index][block_index-1]
    if old_support == destination: return internal, 'already_at_destination'
    if block_index != len(internal[source_index])-1: return internal, 'source_not_clear'
    target_index = None if destination == table else positions[destination][0]
    if target_index is not None and positions[destination][1] != len(internal[target_index])-1: return internal, 'destination_not_clear'
    if target_index is not None and len(internal[target_index])+1 > rule['max_height']: return internal, 'height_exceeded'
    work[source_index].pop()
    if target_index is None: work.append([block])
    else: work[target_index].append(block)
    return [stack for stack in work if stack], None


def goal_count(state, goal, family):
    if family == 'ledger':
        wanted = [(state['accounts'][name], value) for name, value in goal.get('balances', {}).items()]
        wanted += [(state['entries'][name]['status'], value) for name, value in goal.get('statuses', {}).items()]
    elif family == 'register': wanted = [(state['registers'][name], value) for name, value in goal['values'].items()]
    else: wanted = [(state['on'][name], value) for name, value in goal['on'].items()]
    count = len([pair for pair in wanted if pair[0] == pair[1]])
    return count, count == len(wanted)


def broken_constraints(state, start, limits, total, family):
    labels = []
    if family == 'ledger':
        for name, limit in limits.get('minimum_balances', {}).items():
            if state['accounts'][name] < limit: labels.append('minimum_balance:' + name)
        for name, entry in state['entries'].items():
            if entry['status'] in limits.get('forbidden_statuses', []): labels.append('forbidden_status:' + name)
    elif family == 'register':
        for name, limit in limits.get('minimum_values', {}).items():
            if state['registers'][name] < limit: labels.append('minimum_value:' + name)
        for name, limit in limits.get('maximum_values', {}).items():
            if state['registers'][name] > limit: labels.append('maximum_value:' + name)
        for name in limits.get('protected_registers', []):
            if state['registers'][name] != start['registers'][name]: labels.append('protected_register:' + name)
    else:
        for upper, lower in limits.get('forbidden_on', []):
            if state['on'][upper] == lower: labels.append('forbidden_on:' + upper + ':' + lower)
        for name in limits.get('protected_blocks', []):
            if state['on'][name] != start['on'][name]: labels.append('protected_block:' + name)
    if 'max_cost' in limits and total > limits['max_cost']: labels.append('max_cost')
    return sorted(labels)


def observation_check(observation, world):
    """Compare known leaves via an iterative queue, without primary recursion."""
    pending, unknown = [(observation, world)], False
    while pending:
        seen, actual = pending.pop()
        if seen is None: unknown = True; continue
        require(type(seen) is type(actual), "observation type differs")
        if type(seen) is dict:
            require(set(seen) == set(actual), "observation fields differ")
            pending.extend((seen[key], actual[key]) for key in seen)
        elif type(seen) is list:
            require(len(seen) == len(actual), "observation list length differs")
            pending.extend(zip(seen, actual))
        else: require(seen == actual, "possible world contradicts observation")
    return unknown


def structure_signature(state):
    if type(state) is dict: return ('dict', tuple((k, structure_signature(v)) for k, v in sorted(state.items())))
    if type(state) is list: return ('list', tuple(structure_signature(v) for v in state))
    return ('leaf',)


def ratio(number, denominator=1):
    value = Fraction(number, denominator)
    return {'numerator': value.numerator, 'denominator': value.denominator}


def evaluate(task):
    require(type(task) is dict and {'environment', 'initial_state', 'rules', 'goal', 'constraints', 'candidates'} <= set(task), "missing task fields")
    family = task['environment']
    require(type(family) is str and family in ('ledger', 'register', 'blocksworld'), "unknown environment")
    if 'utility_definition' in task: require(task['utility_definition'] == UTILITY, "unsupported utility definition")
    information = task.get('information', 'fully_observed')
    require(information in ('fully_observed', 'missing', 'ambiguous'), "invalid information kind")
    if information == 'fully_observed':
        require('possible_worlds' not in task and task.get('probability_assumption', 'deterministic') == 'deterministic', "invalid deterministic universe")
        worlds, assumption = [task['initial_state']], 'deterministic'
    else:
        worlds, assumption = task.get('possible_worlds'), task.get('probability_assumption')
        require(type(worlds) is list and len(worlds) >= 2 and assumption == 'uniform_over_possible_worlds', "explicit uniform universe required")
        for world in worlds: require(observation_check(task['initial_state'], world), "uncertainty requires unknown field")
        serial = [json.dumps(w, sort_keys=True, separators=(',', ':')) for w in worlds]
        require(len(set(serial)) == len(serial), "duplicate world")
    for world in worlds: check_world(world, task['rules'], task['goal'], task['constraints'], family)
    require(all(structure_signature(world) == structure_signature(worlds[0]) for world in worlds), "world structures differ")
    candidates = task['candidates']
    require(type(candidates) is list and candidates, "candidates required")
    identifiers = []
    for candidate in candidates:
        keys(candidate, ('id', 'plan'))
        require(type(candidate['id']) is str and candidate['id'] and type(candidate['plan']) is list, "invalid candidate")
        identifiers.append(candidate['id'])
        for action in candidate['plan']: check_action(action, family)
    require(len(set(identifiers)) == len(identifiers), "duplicate candidate id")
    result = []
    for candidate in candidates:
        outcomes = []
        for index, world in enumerate(worlds):
            current = encoded(world, family, task['rules'])
            states = [copy.deepcopy(world)]
            history, violations, total, failure = [], [], 0, None
            violations.extend({'constraint': label, 'step': 0} for label in broken_constraints(world, world, task['constraints'], 0, family))
            for step_index, action in enumerate(candidate['plan'], 1):
                next_internal, error = advance(current, action, task['rules'], family)
                snapshot = decoded(next_internal, family, task['rules'])
                step_cost = task['rules']['action_costs'][action['op']] if error is None else 0
                history.append({'step': step_index, 'action': copy.deepcopy(action), 'legal': error is None, 'state': copy.deepcopy(snapshot), 'cost': step_cost, 'error': error})
                if error is not None: failure = step_index; break
                current = next_internal; total += step_cost; states.append(copy.deepcopy(snapshot))
                violations.extend({'constraint': label, 'step': step_index} for label in broken_constraints(snapshot, world, task['constraints'], total, family))
            terminal = decoded(current, family, task['rules'])
            score, achieved = goal_count(terminal, task['goal'], family)
            legal = failure is None
            outcomes.append({'world_index': index, 'legal': legal, 'failure_step': failure, 'steps': history, 'trajectory': states, 'terminal_state': terminal, 'cost': total, 'goal_satisfied': achieved, 'goal_score': score, 'constraint_violations': violations, 'utility': [int(legal and len(violations) == 0), score, -total]})
        n = len(worlds)
        feasible = sum(outcome['utility'][0] for outcome in outcomes)
        goal_sum = sum(outcome['goal_score'] for outcome in outcomes)
        cost_sum = sum(outcome['cost'] for outcome in outcomes)
        result.append({'id': candidate['id'], 'outcomes': outcomes, 'legal_probability': ratio(sum(1 for o in outcomes if o['legal']), n), 'feasible_probability': ratio(feasible, n), 'success_probability': ratio(sum(1 for o in outcomes if o['utility'][0] == 1 and o['goal_satisfied']), n), 'expected_goal_score': ratio(goal_sum, n), 'expected_cost': ratio(cost_sum, n), 'utility': [ratio(feasible, n), ratio(goal_sum, n), ratio(-cost_sum, n)], 'executed_steps': sum(len(o['steps']) for o in outcomes)})
    return {'environment': family, 'world_count': len(worlds), 'information': information, 'probability_assumption': assumption, 'candidates': result}
