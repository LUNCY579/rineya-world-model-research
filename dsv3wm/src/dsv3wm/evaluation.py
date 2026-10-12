"""Prediction-only rollout instruments; exact labels used only after outputs exist."""
import copy
import time


def free_rollout(predict, inputs, horizon):
    """No oracle imports, no hidden labels, no teacher state in recurrent calls."""
    state = copy.deepcopy(inputs['s0'])
    outputs = []
    terminated = False
    for action in inputs['actions'][:horizon]:
        if terminated:
            outputs.append({'state': copy.deepcopy(state), 'legal': False, 'terminated': True, 'skipped': True})
            continue
        result = predict({'domain': inputs['domain'], 'rules': copy.deepcopy(inputs['rules']),
                          's0': copy.deepcopy(state), 'actions': [copy.deepcopy(action)]})
        outputs.append(result)
        if result.get('malformed') or 'state' not in result:
            return {'outputs': outputs, 'malformed': True, 'state': None}
        state = copy.deepcopy(result['state'])
        terminated = result.get('terminated') is True
    return {'outputs': outputs, 'malformed': False, 'state': state}


def score_dynamics(predict, records, horizons=(1, 2, 4)):
    from .data import conditioning
    results = []
    for record in records:
        result = {'id': record['id'], 'parent_world_id': record['parent_world_id'], 'domain': record['domain'], 'stress': record['stress'],
                  'horizons': {}}
        for h in horizons:
            if len(record['actions']) < h:
                continue
            inputs = conditioning(record, h)
            started = time.perf_counter()
            direct = predict(inputs)
            recursive = free_rollout(predict, inputs, h)
            label = record['labels']['states'][h - 1]
            result['horizons'][str(h)] = {
                'direct_correct': not direct.get('malformed') and direct.get('state') == label,
                'rollout_correct': not recursive['malformed'] and recursive['state'] == label,
                'legality_correct': type(direct.get('legal')) is bool and direct['legal'] == record['labels']['legal'][h - 1],
                'direct_malformed': bool(direct.get('malformed')), 'rollout_malformed': recursive['malformed'],
                'intermediate_correct': [out.get('state') == record['labels']['states'][j] and not out.get('malformed')
                                         for j, out in enumerate(recursive['outputs'])],
                'copy_correct': record['s0'] == label,
                'elapsed_s': time.perf_counter() - started,
            }
        results.append(result)
    summary = {}
    for h in horizons:
        rows = [r['horizons'][str(h)] for r in results if str(h) in r['horizons']]
        summary[str(h)] = {'n': len(rows)}
        for field in ('direct_correct', 'rollout_correct', 'legality_correct', 'direct_malformed', 'rollout_malformed', 'copy_correct'):
            count = sum(r[field] for r in rows)
            summary[str(h)][field] = {'count': count, 'rate': count / len(rows) if rows else None}
    return {'summary': summary, 'per_parent': results}
