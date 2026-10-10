"""Public-only official task solver for option_resume v1.

Independent explicit stack transitions reused from frozen feasibility source
32217364392f2c2b4b944fc276c3a4f84ff522e445f2b37b4c807100e6da35bb.
Validation and typed equality below use iterative traversal. No corpus controls,
factory integration, candidate generation, or approval mechanism.
"""
from copy import deepcopy


def typed_equal(a, b):
    pending = [(a, b)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right): return False
        if type(left) is dict:
            if len(left) != len(right) or set(left) != set(right): return False
            pending.extend((v, right[k]) for k, v in left.items())
        elif type(left) is list:
            if len(left) != len(right): return False
            pending.extend(zip(left, right))
        elif left != right: return False
    return True


def exact(obj, keys):
    if type(obj) is not dict or set(obj) != set(keys):
        raise ValueError("invalid shape")


def integer(x):
    if type(x) is not int:
        raise ValueError("integer required")


def validate_program(task):
    exact(task, "kind version initial_state options entry mode history checkpoint event_budget".split())
    if task['kind'] != 'option_resume' or type(task['kind']) is not str:
        raise ValueError("kind")
    integer(task['version'])
    if task['version'] != 1 or task['mode'] not in ('fresh', 'resume', 'restart'):
        raise ValueError("version or mode")
    integer(task['event_budget'])
    if task['event_budget'] < 0:
        raise ValueError("budget")
    exact(task['initial_state'], ('x', 'y'))
    for v in task['initial_state'].values(): integer(v)
    opts = task['options']
    if type(opts) is not dict or not opts:
        raise ValueError("options")
    for n, o in opts.items():
        if type(n) is not str or not n: raise ValueError("option name")
        exact(o, ('params', 'initiation', 'success', 'body'))
        ps = o['params']
        if type(ps) is not list or any(type(p) is not str or not p for p in ps):
            raise ValueError("params")
        if len(set(ps)) != len(ps): raise ValueError("duplicate param")

    def expr(e, ps):
        if type(e) is not dict or len(e) != 1: raise ValueError("expression")
        if set(e) == {'const'}: integer(e['const'])
        elif set(e) == {'reg'}:
            if type(e['reg']) is not str or e['reg'] not in ('x', 'y'): raise ValueError("register")
        elif set(e) == {'param'}:
            if type(e['param']) is not str or e['param'] not in ps: raise ValueError("param ref")
        else: raise ValueError("expression form")

    def pred(p, ps):
        exact(p, ('all',))
        if type(p['all']) is not list: raise ValueError("all")
        for atom in p['all']:
            exact(atom, ('left', 'cmp', 'right'))
            if type(atom['cmp']) is not str or atom['cmp'] not in ('lt', 'eq', 'ge'): raise ValueError("cmp")
            expr(atom['left'], ps); expr(atom['right'], ps)

    edges = {n: set() for n in opts}

    def args(name, aa, ps):
        if type(name) is not str or name not in opts: raise ValueError("callee")
        exact(aa, opts[name]['params'])
        for e in aa.values(): expr(e, ps)

    def body(bb, ps, owner):
        pending = [bb]
        while pending:
            current = pending.pop()
            if type(current) is not list: raise ValueError("body")
            for i in current:
                if type(i) is not dict or type(i.get('op')) is not str: raise ValueError("instruction")
                op = i['op']
                if op in ('add', 'scale'):
                    exact(i, ('op', 'register', 'value'))
                    if type(i['register']) is not str or i['register'] not in ('x', 'y'): raise ValueError("register")
                    expr(i['value'], ps)
                elif op == 'swap': exact(i, ('op',))
                elif op == 'call':
                    exact(i, ('op', 'option', 'args')); args(i['option'], i['args'], ps)
                    edges[owner].add(i['option'])
                elif op == 'if':
                    exact(i, ('op', 'test', 'then', 'else')); pred(i['test'], ps)
                    pending.extend((i['then'], i['else']))
                else: raise ValueError("opcode")

    for n, o in opts.items():
        pred(o['initiation'], o['params']); pred(o['success'], o['params']); body(o['body'], o['params'], n)
    exact(task['entry'], ('option', 'args')); args(task['entry']['option'], task['entry']['args'], ())
    indegree = {n: 0 for n in opts}
    for children in edges.values():
        for child in children: indegree[child] += 1
    ready = [n for n in opts if indegree[n] == 0]
    count = 0
    while ready:
        n = ready.pop(); count += 1
        for child in edges[n]:
            indegree[child] -= 1
            if indegree[child] == 0: ready.append(child)
    if count != len(opts): raise ValueError("cycle")


def eval_expr(e, state, bindings):
    if 'const' in e: return e['const']
    if 'reg' in e: return state[e['reg']]
    return bindings[e['param']]


def predicate(p, state, bindings):
    values = [(eval_expr(a['left'], state, bindings), a['cmp'], eval_expr(a['right'], state, bindings)) for a in p['all']]
    return all((x < y if op == 'lt' else x == y if op == 'eq' else x >= y) for x, op, y in values)


def initial(task, state=None):
    return {'state': deepcopy(task['initial_state'] if state is None else state), 'frames': [],
            'next_instance_id': 0, 'root_pending': True, 'phase': 'running'}


def selected_body(task, frame):
    b = task['options'][frame['option']]['body']
    path = frame['body_path']
    for i in range(0, len(path), 2): b = b[path[i]][path[i + 1]]
    return b


def normalize(task, machine):
    while machine['frames']:
        top = machine['frames'][-1]
        if top['frame_kind'] != 'branch' or top['next_index'] < len(selected_body(task, top)): break
        machine['frames'].pop()


def transition(task, machine):
    normalize(task, machine)
    if machine['phase'] != 'running': raise ValueError("terminal")
    s, fs = machine['state'], machine['frames']

    def enter(name, aa, caller_bindings, site):
        # All values are read before any transition state or binding mutation.
        bindings = {p: eval_expr(aa[p], s, caller_bindings) for p in task['options'][name]['params']}
        iid = machine['next_instance_id']; machine['next_instance_id'] += 1
        ok = predicate(task['options'][name]['initiation'], s, bindings)
        e = {'event': 'enter', 'instance_id': iid, 'option': name, 'bindings': deepcopy(bindings),
             'call_site': deepcopy(site), 'initiation_ok': ok}
        if ok:
            fs.append({'frame_kind': 'option', 'instance_id': iid, 'option': name, 'bindings': bindings,
                       'body_path': [], 'next_index': 0})
        else: machine['phase'] = 'initiation_failed'
        return e

    if machine['root_pending']:
        machine['root_pending'] = False
        e = enter(task['entry']['option'], task['entry']['args'], {}, None)
    else:
        f = fs[-1]; body = selected_body(task, f); idx = f['next_index']
        if idx == len(body):
            assert f['frame_kind'] == 'option'
            ok = predicate(task['options'][f['option']]['success'], s, f['bindings'])
            e = {'event': 'return', 'instance_id': f['instance_id'], 'option': f['option'], 'success_ok': ok}
            fs.pop(); normalize(task, machine)
            if not ok: machine['phase'] = 'success_failed'
            elif not fs: machine['phase'] = 'completed'
        else:
            instr = body[idx]; op = instr['op']; f['next_index'] += 1
            loc = {'instance_id': f['instance_id'], 'option': f['option'], 'body_path': deepcopy(f['body_path']), 'index': idx}
            if op == 'call':
                e = enter(instr['option'], instr['args'], f['bindings'],
                          {'caller_instance': f['instance_id'], 'body_path': deepcopy(f['body_path']), 'index': idx})
            elif op == 'if':
                cond = predicate(instr['test'], s, f['bindings'])
                e = {'event': 'branch', **loc, 'condition': cond}
                fs.append({'frame_kind': 'branch', 'instance_id': f['instance_id'], 'option': f['option'],
                           'bindings': deepcopy(f['bindings']), 'body_path': f['body_path'] + [idx, 'then' if cond else 'else'], 'next_index': 0})
            else:
                e = {'event': 'effect', **loc, 'op': op}
                if op == 'swap': s['x'], s['y'] = s['y'], s['x']
                else:
                    v = eval_expr(instr['value'], s, f['bindings']); r = instr['register']
                    s[r] = s[r] + v if op == 'add' else s[r] * v
                    e.update(register=r, value=v)
                e['state'] = deepcopy(s)
    normalize(task, machine)
    return e


def invalid(stage):
    return {'status': 'invalid', 'error': stage, 'checkpoint': None, 'new_events': [],
            'previous_effects': None, 'new_effects': 0, 'next_event': None}


def solve(task):
    try: validate_program(task)
    except (ValueError, TypeError, KeyError, RecursionError): return invalid('invalid_task')
    m = initial(task)
    if type(task['history']) is not list: return invalid('invalid_history')
    for supplied in task['history']:
        if m['phase'] != 'running': return invalid('invalid_history')
        generated = transition(task, m)
        if not typed_equal(supplied, generated): return invalid('invalid_history')
    if task['mode'] == 'fresh':
        if task['history'] or task['checkpoint'] is not None: return invalid('invalid_checkpoint')
    elif not typed_equal(task['checkpoint'], m): return invalid('invalid_checkpoint')
    if task['mode'] == 'restart': m = initial(task, m['state'])
    events = []
    for _ in range(task['event_budget']):
        if m['phase'] != 'running': break
        events.append(transition(task, m))
    normalize(task, m)
    status = 'paused' if m['phase'] == 'running' else m['phase']
    preview = transition(task, deepcopy(m)) if status == 'paused' else None
    return {'status': status, 'error': None, 'checkpoint': deepcopy(m), 'new_events': events,
            'previous_effects': sum(e['event'] == 'effect' for e in task['history']),
            'new_effects': sum(e['event'] == 'effect' for e in events), 'next_event': preview}
