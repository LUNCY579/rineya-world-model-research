"""Training exports carry inference inputs and oracle-only labels separately."""
from copy import deepcopy
from itertools import combinations


def training_views(record):
    """Return pointwise, pairwise and listwise examples, never audit metadata.

    Input comes only from canonical ``input``. Targets retain exact rational
    utilities; optional explanatory text is deliberately absent.
    """
    gate = record.get('quality_gate', {})
    if gate.get('status') in ('quarantined', 'rejected'):
        raise ValueError('quarantined or rejected records cannot become training views')
    inputs = record['input']
    labels = record['targets']['decision']
    by_id = {x['id']: x for x in labels['candidates']}
    pointwise = []
    for candidate in inputs['candidates']:
        item_input = deepcopy(inputs)
        item_input['candidates'] = [deepcopy(candidate)]
        pointwise.append({'record_id': record['id'], 'input': item_input,
                          'target': deepcopy(by_id[candidate['id']])})
    pairwise = []
    for left, right in combinations(inputs['candidates'], 2):
        item_input = deepcopy(inputs)
        item_input['candidates'] = [deepcopy(left), deepcopy(right)]
        a, b = by_id[left['id']], by_id[right['id']]
        pairwise.append({'record_id': record['id'], 'input': item_input,
                         'target': {'left_id': left['id'], 'right_id': right['id'],
                                    'preference': _compare(a['utility'], b['utility']),
                                    'left_utility': deepcopy(a['utility']),
                                    'right_utility': deepcopy(b['utility'])}})
    return {'pointwise': pointwise, 'pairwise': pairwise,
            'listwise': [{'record_id': record['id'], 'input': deepcopy(inputs),
                          'target': deepcopy(labels)}]}


def _compare(left, right):
    from fractions import Fraction
    def parts(value):
        return tuple(Fraction(v['numerator'], v['denominator'])
                     if isinstance(v, dict) else Fraction(v) for v in value)
    return (parts(left) > parts(right)) - (parts(left) < parts(right))
