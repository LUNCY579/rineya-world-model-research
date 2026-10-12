"""A bounded development-only comparison, never reading sealed answers."""
from collections import Counter
from copy import deepcopy
import math
from .build import curriculum, generate_group, semantic_fingerprint


def run_position_experiment(groups=None, seed_start=None):
    config = curriculum()
    experiment = config['controlled_experiment']
    if experiment['split'] != 'dev':
        raise ValueError('controlled optimization must use development tasks only')
    groups = groups or experiment['groups']
    seed_start = seed_start if seed_start is not None else experiment['seed_start']
    before_config = deepcopy(config)
    before_config['candidate_order'] = 'fixed'
    after_config = deepcopy(config)
    after_config['candidate_order'] = 'balanced'
    counts = {'before': Counter(), 'after': Counter()}
    semantic_changes = 0
    oracle_label_changes = 0
    evaluated = 0
    id_only_correct = 0
    id_lexical_rank_counts = Counter()
    for seed in range(seed_start, seed_start + groups):
        # Each group's first A task has the same four candidate identities and
        # a unique best action; sampling it avoids variable action-count bias.
        before = generate_group(seed, before_config, 'dev')[0]
        after = generate_group(seed, after_config, 'dev')[0]
        sorted_ids = sorted(c['id'] for c in after['input']['candidates'])
        winner = after['targets']['decision']['best_candidate_ids'][0]
        id_only_correct += sorted_ids[0] == winner
        id_lexical_rank_counts[sorted_ids.index(winner)] += 1
        semantic_changes += semantic_fingerprint(before) != semantic_fingerprint(after)
        def canonical_labels(record):
            return {c['id']: c for c in record['targets']['decision']['candidates']}
        oracle_label_changes += canonical_labels(before) != canonical_labels(after)
        for name, record in (('before', before), ('after', after)):
            best = record['targets']['decision']['best_candidate_ids']
            if len(best) != 1:
                raise ValueError('development fixture must have one best candidate')
            index = next(i for i, c in enumerate(record['input']['candidates']) if c['id'] == best[0])
            counts[name][index] += 1
        evaluated += 1
    def measure(count):
        probabilities = [count[i] / evaluated for i in range(4)]
        return {'best_option_position_counts': {str(i): count[i] for i in range(4)},
                'maximum_position_fraction': max(probabilities),
                'position_entropy_bits': -sum(p * math.log2(p) for p in probabilities if p)}
    before_metric, after_metric = measure(counts['before']), measure(counts['after'])
    no_regression = semantic_changes == oracle_label_changes == 0
    supported = no_regression and after_metric['maximum_position_fraction'] < before_metric['maximum_position_fraction']
    return {'experiment_version': 'position-experiment-1.0',
            'failure': 'Fixed candidate generation makes the first option always correct in the goal-A fixture.',
            'hypothesis': experiment['hypothesis'], 'split': 'dev',
            'seed_start': seed_start, 'groups': groups, 'records_compared': evaluated,
            'before': before_metric, 'after': after_metric,
            'semantic_changes': semantic_changes, 'oracle_label_changes': oracle_label_changes,
            'candidate_identifier_audit': {'ids_are_opaque': True,
                'lowest_lexical_id_choice_accuracy': id_only_correct / evaluated,
                'winner_lexical_rank_counts': {str(i): id_lexical_rank_counts[i] for i in range(4)},
                'baseline_scope': 'ID-only lowest lexical identifier heuristic; no guarantee against arbitrary shortcuts'},
            'proposed_decision': 'promote' if supported else 'reject',
            'independent_review_required': True,
            'limitations': ['No evidence of model learning or model generalization.',
                            'This comparison removes one option-position shortcut; other shortcut audits remain necessary.',
                            'Search-prefix examples preserve option order because permutation changes compute policy.']}
