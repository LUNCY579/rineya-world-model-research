"""Twelve hand-authored prospective source graphs, no production sample source."""
from copy import deepcopy
from itertools import permutations
import json


def C(n): return {'const': n}
def R(n): return {'reg': n}
def P(n): return {'param': n}
def atom(l, cmp, r): return {'left': l, 'cmp': cmp, 'right': r}
def pred(*aa): return {'all': list(aa)}
def add(r, v): return {'op': 'add', 'register': r, 'value': v}
def scale(r, v): return {'op': 'scale', 'register': r, 'value': v}
def swap(): return {'op': 'swap'}
def call(n, p, v): return {'op': 'call', 'option': n, 'args': {p: v}}
def branch(test, yes, no): return {'op': 'if', 'test': test, 'then': yes, 'else': no}
def opt(p, body, initiation=None, success=None):
    return {'params': [p], 'initiation': pred() if initiation is None else initiation,
            'success': pred() if success is None else success, 'body': body}


def hand_programs():
    # Each variant changes an ordered control/call/read-write graph, not a constant or name.
    variants = [
        (12, 'saved_argument_after_mutation', [add('y', P('b'))], {}, []),
        (13, 'saved_argument_after_mutation', [add('y', P('b')), scale('y', P('b'))], {}, []),
        (18, 'saved_argument_after_mutation', [add('y', P('b')), swap()], {}, []),
        (14, 'nested_return_caller_continuation', [call('leaf', 'c', P('b'))],
         {'leaf': opt('c', [add('y', P('c'))])}, []),
        (15, 'nested_return_caller_continuation', [call('leaf', 'c', P('b')), add('y', P('b'))],
         {'leaf': opt('c', [scale('y', P('c')), add('y', P('c'))])}, []),
        (19, 'nested_return_caller_continuation', [call('leaf', 'c', P('b'))],
         {'leaf': opt('c', [swap(), add('x', P('c'))])}, []),
        (16, 'selected_branch_after_guard_reversal',
         [branch(pred(atom(R('x'), 'ge', P('b'))), [add('y', P('b'))], [add('y', C(0))])], {}, []),
        (17, 'selected_branch_after_guard_reversal',
         [branch(pred(atom(R('x'), 'ge', P('b'))), [add('y', P('b')), scale('y', P('b'))], [swap()])],
         {}, [branch(pred(atom(R('y'), 'ge', P('a'))), [add('x', P('a'))], [add('y', P('a'))])]),
        (28, 'selected_branch_after_guard_reversal',
         [branch(pred(atom(R('y'), 'lt', P('b'))), [scale('y', P('b')), add('y', P('b'))], [add('y', P('b'))])], {}, []),
        (20, 'noncommuting_calls_and_exit_contract', [add('y', P('b'))],
         {'leaf': opt('c', [scale('y', P('c')), add('y', P('c'))])}, [call('leaf', 'c', P('a'))]),
        (21, 'noncommuting_calls_and_exit_contract', [add('y', P('b'))],
         {'leaf': opt('c', [scale('y', P('c')), add('y', P('c'))])},
         [call('leaf', 'c', P('a')), call('leaf', 'c', P('a'))]),
        (29, 'noncommuting_calls_and_exit_contract', [add('y', P('b'))],
         {'leaf': opt('c', [scale('y', P('c')), call('tail', 'd', P('c'))]),
          'tail': opt('d', [add('y', P('d'))])}, [call('leaf', 'c', P('a'))]),
    ]
    result = []
    for idx, stratum, pending, extra, root_middle in variants:
        child = opt('b', [branch(pred(atom(R('x'), 'lt', C(6))),
                                    [add('x', P('b'))] + pending, [scale('y', C(1))])],
                    pred(atom(R('x'), 'ge', C(4)), atom(R('x'), 'lt', C(6))), pred(atom(R('x'), 'ge', C(0))))
        root = opt('a', [add('x', P('a')), call('child', 'b', P('a'))] + root_middle + [add('y', P('a'))],
                   pred(atom(P('a'), 'ge', C(1)), atom(R('x'), 'ge', C(-4096))),
                   pred(atom(R('x'), 'ge', C(0)), atom(R('y'), 'ge', C(-4096))))
        task = {'kind': 'option_resume', 'version': 1, 'initial_state': {'x': 2, 'y': 1},
                'options': {'root': root, 'child': child, **extra},
                'entry': {'option': 'root', 'args': {'a': R('x')}},
                'mode': 'fresh', 'history': [], 'checkpoint': None, 'event_budget': 48}
        result.append({'source_family_index': idx, 'stratum': stratum,
                       'split': 'train' if idx % 10 < 8 else 'dev_challenge', 'task': task})
    return result


def contexts(base, solve):
    F = deepcopy(base)
    f = solve(F)
    if f['status'] != 'completed': raise AssertionError('hand F not complete')
    cut = next(i + 1 for i, e in enumerate(f['new_events']) if e['event'] == 'effect' and e['option'] == 'child')
    prefix = deepcopy(F); prefix['event_budget'] = cut
    at = solve(prefix)
    Rtask = deepcopy(F)
    Rtask.update(mode='resume', history=deepcopy(f['new_events'][:cut]), checkpoint=at['checkpoint'])
    r = solve(Rtask)
    T = deepcopy(Rtask); T['mode'] = 'restart'
    pbudget = next(k for k in range(1, len(r['new_events']))
                   if any(e['event'] == 'effect' for e in r['new_events'][:k]))
    Ptask = deepcopy(Rtask); Ptask['event_budget'] = pbudget
    p = solve(Ptask)
    U = deepcopy(Rtask); U.update(history=Rtask['history'] + p['new_events'], checkpoint=p['checkpoint'])
    A = deepcopy(F); A['entry']['args']['a'] = C(3)
    I = deepcopy(F); I['entry']['args']['a'] = C(0)
    S = deepcopy(F); S['options']['root']['success']['all'][0]['right'] = C(100000)
    X = deepcopy(Rtask)
    return dict(F=F, R=Rtask, T=T, P=Ptask, U=U, A=A, I=I, S=S, X=X), cut, pbudget


def normalized_signature(task):
    # Canonical ordered call/control/def-use syntax under all option/register renamings.
    # Constants erase to one marker; each parameter maps to its declaration position.
    names = list(task['options']); possibilities = []
    for order in permutations(names):
        om = {n: i for i, n in enumerate(order)}
        for regs in [('x', 'y'), ('y', 'x')]:
            rm = dict(zip(regs, ['r0', 'r1']))
            def e(x, ps):
                if 'const' in x: return ['const']
                if 'reg' in x: return ['reg', rm[x['reg']]]
                return ['param', ps.index(x['param'])]
            def pp(p, ps): return [[e(a['left'], ps), a['cmp'], e(a['right'], ps)] for a in p['all']]
            def body(bb, ps):
                out = []
                for i in bb:
                    op = i['op']
                    if op in ('add', 'scale'): out.append([op, rm[i['register']], e(i['value'], ps)])
                    elif op == 'swap': out.append([op])
                    elif op == 'call':
                        cps = task['options'][i['option']]['params']
                        out.append([op, om[i['option']], [e(i['args'][p], ps) for p in cps]])
                    else: out.append([op, pp(i['test'], ps), body(i['then'], ps), body(i['else'], ps)])
                return out
            graph = [[len(task['options'][n]['params']), pp(task['options'][n]['initiation'], task['options'][n]['params']),
                      pp(task['options'][n]['success'], task['options'][n]['params']),
                      body(task['options'][n]['body'], task['options'][n]['params'])] for n in order]
            possibilities.append(json.dumps([om[task['entry']['option']], graph], separators=(',', ':')))
    return min(possibilities)


def reverse_objects(value):
    if type(value) is dict: return {k: reverse_objects(v) for k, v in reversed(list(value.items()))}
    if type(value) is list: return [reverse_objects(v) for v in value]
    return value
