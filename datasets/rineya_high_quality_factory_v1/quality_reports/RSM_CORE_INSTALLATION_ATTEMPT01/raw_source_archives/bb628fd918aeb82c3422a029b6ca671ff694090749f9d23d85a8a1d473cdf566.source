"""Input-only graph diagnostics, independent of gold, kind/tag/ID spelling.

Full view preserves guards and parameter/read dependencies. Shared primitive
view drops option guards and arguments; parameter/constant operands collapse
to scalar operands, retaining live-register reads, ordered primitive writes,
conditions, branch edges and call/return structure. It reports all-declared and
root-reachable views separately rather than calling a grammar change novelty.
"""
from itertools import permutations
from hashlib import sha256
import json


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def as_options(task):
    if 'options' in task:
        return task['options'], task['entry']['option']
    name = '#legacy-main#'
    options = {n: {'params': [], 'initiation': {'all': []}, 'success': {'all': []}, 'body': b}
               for n, b in task['procedures'].items()}
    options[name] = {'params': [], 'initiation': {'all': []}, 'success': {'all': []}, 'body': task['main']}
    return options, name


def full_option_signature(task):
    options, root = as_options(task); signatures = []
    for order in permutations(options):
        names = {n: i for i, n in enumerate(order)}
        for rr in [('x', 'y'), ('y', 'x')]:
            regs = dict(zip(rr, ('r0', 'r1')))
            def expr(e, params):
                if type(e) is int or 'const' in e: return ['scalar']
                if 'reg' in e: return ['read-register', regs[e['reg']]]
                return ['read-binding', params.index(e['param'])]
            def predicate(p, params):
                if 'all' not in p:
                    return [[['read-register', regs[p['register']]], p['cmp'], ['scalar']]]
                return [[expr(a['left'], params), a['cmp'], expr(a['right'], params)] for a in p['all']]
            def body(bb, params):
                out = []
                for i in bb:
                    op = i['op']
                    if op in ('add', 'scale'): out.append([op, regs[i['register']], expr(i['value'], params)])
                    elif op == 'swap': out.append([op])
                    elif op == 'call':
                        child = i.get('option', i.get('name'))
                        aa = i.get('args', {})
                        out.append([op, names[child], [expr(aa[p], params) for p in options[child]['params']]])
                    else: out.append([op, predicate(i['test'], params), body(i['then'], params), body(i['else'], params)])
                return out
            graph = [[len(options[n]['params']), predicate(options[n]['initiation'], options[n]['params']),
                      predicate(options[n]['success'], options[n]['params']), body(options[n]['body'], options[n]['params'])] for n in order]
            signatures.append(canonical([names[root], graph]))
    return min(signatures)


def shared_primitive_signature(task, reachable_only=False):
    if 'options' in task:
        root = task['entry']['option']; main = task['options'][root]['body']
        procedures = {n: o['body'] for n, o in task['options'].items() if n != root}
    else:
        main, procedures = task['main'], dict(task['procedures'])
    def calls(bb):
        found = set()
        for i in bb:
            if i['op'] == 'call': found.add(i.get('option', i.get('name')))
            elif i['op'] == 'if': found.update(calls(i['then'])); found.update(calls(i['else']))
        return found
    if reachable_only:
        seen, pending = set(), list(calls(main))
        while pending:
            n = pending.pop()
            if n in seen: continue
            seen.add(n); pending.extend(calls(procedures[n]))
        procedures = {n: b for n, b in procedures.items() if n in seen}
    signatures = []
    for order in permutations(procedures):
        names = {n: i for i, n in enumerate(order)}
        for rr in [('x', 'y'), ('y', 'x')]:
            regs = dict(zip(rr, ('r0', 'r1')))
            def operand(e):
                return ['read-register', regs[e['reg']]] if type(e) is dict and 'reg' in e else ['scalar']
            def condition(p):
                if 'all' not in p: return [[['read-register', regs[p['register']]], p['cmp'], ['scalar']]]
                return [[operand(a['left']), a['cmp'], operand(a['right'])] for a in p['all']]
            def body(bb):
                out = []
                for i in bb:
                    op = i['op']
                    if op in ('add', 'scale'): out.append([op, regs[i['register']], operand(i['value'])])
                    elif op == 'swap': out.append([op])
                    elif op == 'call': out.append([op, names[i.get('option', i.get('name'))]])
                    else: out.append([op, condition(i['test']), body(i['then']), body(i['else'])])
                return out
            signatures.append(canonical([body(main), [body(procedures[n]) for n in order]]))
    return min(signatures)


def compare(proposed, legacy_rows):
    allowed = {'sample_id', 'semantic_parent_id', 'semantic_family_id', 'track_id', 'semantic_task', 'rendered_input', 'split', 'record_hash'}
    if any(set(row) != allowed for row in legacy_rows): raise ValueError('Target-free row whitelist mismatch')
    groups = {}
    for row in legacy_rows:
        n = row['semantic_family_id']; task = row['semantic_task']
        if n in groups and canonical(groups[n]['semantic_task']) != canonical(task): raise ValueError('Legacy field pair differs')
        groups[n] = row
    views = [('full_option', full_option_signature),
             ('shared_primitive_all_declared', shared_primitive_signature),
             ('shared_primitive_reachable', lambda t: shared_primitive_signature(t, True))]
    report = {'legacy_records': len(legacy_rows), 'legacy_source_families': len(groups), 'proposed_hand_families': len(proposed),
              'targets_or_golds_used': False, 'graph_views': {}}
    for name, normalization in views:
        current = [(f['source_family_index'], normalization(f['task'])) for f in proposed]
        prior = [(k, normalization(v['semantic_task'])) for k, v in sorted(groups.items())]
        collisions = [{'proposed_family': p, 'legacy_family': q} for p, a in current for q, b in prior if a == b]
        duplicate_pairs = [[p, q] for i, (p, a) in enumerate(current) for q, b in current[i + 1:] if a == b]
        report['graph_views'][name] = {'proposed_distinct_signatures': len({a for _, a in current}),
                                     'legacy_distinct_signatures': len({a for _, a in prior}),
                                     'exact_new_legacy_collisions': collisions, 'proposed_duplicate_pairs': duplicate_pairs,
                                     'new_legacy_pair_comparisons': len(current) * len(prior),
                                     'proposed': [{'family': n, 'sha256': sha256(s.encode()).hexdigest(), 'signature': s} for n, s in current],
                                     'legacy': [{'family': n, 'sha256': sha256(s.encode()).hexdigest(), 'signature': s} for n, s in prior]}
    return report
