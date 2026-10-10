"""Actual input-only predictions. Own exposed solver plus separate conjunction, no canonical reads."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import json

BASE = Path(__file__).resolve().parents[2]
PREFIX = 'reviews/adjudication/belief-v4-input-only-'

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def bind(path):
    raw = (BASE / path).read_bytes()
    return {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def new_json(path, value):
    with (BASE / path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    return bind(path)

def new_rows(path, values):
    with (BASE / path).open('x') as stream:
        for value in values: stream.write(canonical(value) + '\n')
    return bind(path)

def direct(task):
    """Independently express every prefix as a conjunction over the original full table."""
    worlds, observations = task['worlds'], task['observations']
    supports = [[w for w in worlds if all(type(w['facts'][o['fact']]) is type(o['equals'])
                  and w['facts'][o['fact']] == o['equals'] for o in observations[:end])]
                for end in range(len(observations) + 1)]
    final = supports[-1]
    answers = []
    for query in task['queries']:
        counts, values = Counter(), {}
        for world in final:
            value = world['facts'][query['fact']]
            token = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            key = (type(value).__name__, token)
            counts[key] += 1; values[key] = value
        keys = sorted(counts, key=lambda key: key[1])
        bins = [{'value': values[key], 'count': counts[key]} for key in keys]
        entailed = len(final) > 0 and len(keys) == 1
        answers.append({'query_id': query['query_id'], 'counts': bins, 'distinct_values': len(keys),
                        'entailed': entailed, 'guaranteed': bins[0]['value'] if entailed else None,
                        'can_reduce_uncertainty': len(keys) > 1})
    return {'surviving_worlds': sorted(w['world_id'] for w in final),
            'belief_sizes': [len(s) for s in supports[1:]], 'queries': answers}, supports

def input_key(row):
    return canonical({'track_id': row['track_id'], 'split': row['split'],
                      'semantic_task': row['semantic_task'], 'rendered_input': row['rendered_input']})

def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen_path = PREFIX + 'preexecution-freeze.json'
    frozen = json.loads((BASE / frozen_path).read_text())
    for item in frozen['bindings']:
        assert bind(item['path']) == item, item['path']
    config = json.loads((BASE / (PREFIX + 'config.json')).read_text())
    # This exact own prior helper has only standard imports, definitions and a guarded main.
    # Freeze/archive read-back above occurs before importing it. Do not call its main.
    dependency = BASE / config['own_prior_solver_archive']
    spec = importlib.util.spec_from_file_location('own_exposed_belief_public_solver', dependency)
    own = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(own)
    hands = json.loads((BASE / config['hand_fixtures']).read_text())
    hand_results = []
    for case in hands['cases']:
        try:
            target, supports = own.solve(case['task'])
            other, other_supports = direct(case['task'])
            assert canonical(target) == canonical(other)
            assert [[w['world_id'] for w in s] for s in supports] == [[w['world_id'] for w in s] for s in other_supports]
            assert canonical(target) == canonical(case['expected'])
            passed, reason = not case.get('invalid', False), None
        except AssertionError as error:
            passed, reason = case.get('invalid', False), type(error).__name__
        hand_results.append({'case_id': case['case_id'], 'passed': bool(passed), 'reason': reason})
    hand_binding = new_json(PREFIX + 'hand-results.json', {'scope': 'HAND_PUBLIC_FILTER_CASES',
        'cases': hand_results, 'passed': sum(r['passed'] for r in hand_results), 'production_approval': False})
    assert all(r['passed'] for r in hand_results), 'first hand failure preserved in result'

    fresh = [json.loads(line) for line in (BASE / config['fresh_view']).read_bytes().splitlines()]
    old = [json.loads(line) for line in (BASE / config['prior_target_free_view']).read_bytes().splitlines()]
    view_manifest = json.loads((BASE / config['fresh_view_manifest']).read_text())
    supplied_lineage = json.loads((BASE / config['input_lineage']).read_text())
    assert len(fresh) == len(old) == config['expected_records'] == 3456
    fields = set(config['allowed_view_fields'])
    for row in fresh + old:
        assert set(row) == fields and row['track_id'] == 'RBM-001'
        assert all(type(row[k]) is str and bool(row[k]) for k in fields - {'semantic_task'})
        assert len(row['record_hash']) == 64 and all(c in '0123456789abcdef' for c in row['record_hash'])
        assert row['split'] in ('train', 'dev_challenge')
        assert row['semantic_parent_id'] == row['semantic_family_id']
        visible = row['rendered_input'].split('Task JSON:\n', 1)
        assert len(visible) == 2 and canonical(json.loads(visible[1])) == canonical(row['semantic_task'])
        assert set(row['semantic_task']) == {'kind', 'worlds', 'observations', 'queries'}
        assert row['semantic_task']['kind'] == 'belief'
    assert set(view_manifest['view_fields']) == fields and view_manifest['records'] == 3456
    assert view_manifest['view_sha256'] == bind(config['fresh_view'])['sha256']
    assert view_manifest['input_lineage_sha256'] == bind(config['input_lineage'])['sha256']
    fresh_ids = {r['sample_id']: r for r in fresh}; old_ids = {r['sample_id']: r for r in old}
    assert len(fresh_ids) == len(old_ids) == 3456
    old_index = {input_key(r): r for r in old}
    assert len(old_index) == 3456, 'complete student-visible input/split joins must be unique'
    root_lineage = {r['fresh_sample_id']: r for r in supplied_lineage['joins']}
    assert len(root_lineage) == 3456 and set(root_lineage) == set(fresh_ids)
    assert {r['prior_sample_id'] for r in root_lineage.values()} == set(old_ids)

    predictions, judgments, joins = [], [], []
    families, family_mapping, family_splits = Counter(), {}, {}
    split_counts, status_counts, query_flags, type_counts = Counter(), Counter(), Counter(), Counter()
    semantic_style_groups, targets_by_input, prefix_sizes = defaultdict(list), {}, Counter()
    for row in fresh:
        sid, task = row['sample_id'], row['semantic_task']
        key = input_key(row)
        assert key in old_index
        previous = old_index[key]
        joined = root_lineage[sid]
        assert joined['prior_sample_id'] == previous['sample_id']
        assert joined['same_typed_student_input_and_split'] is True
        assert row['rendered_input'] == previous['rendered_input']
        assert canonical(task) == canonical(previous['semantic_task']) and row['split'] == previous['split']
        family = row['semantic_family_id']
        assert family not in family_mapping or family_mapping[family] == previous['semantic_family_id']
        family_mapping[family] = previous['semantic_family_id']
        assert family not in family_splits or family_splits[family] == row['split']
        family_splits[family] = row['split']; families[family] += 1; split_counts[row['split']] += 1
        target, supports = own.solve(task)  # Full fresh recomputation; no prior prediction/target read.
        independently_formed, direct_supports = direct(task)
        assert canonical(target) == canonical(independently_formed)
        prefix_ids = [sorted(w['world_id'] for w in support) for support in supports]
        assert prefix_ids == [sorted(w['world_id'] for w in support) for support in direct_supports]
        ih = hashlib.sha256(key.encode()).hexdigest()
        th = digest(task); prompt_hash = hashlib.sha256(row['rendered_input'].encode()).hexdigest()
        predictions.append({'sample_id': sid, 'record_hash': row['record_hash'],
            'reviewer_execution_id': '/root/research_architect', 'input_sha256': ih,
            'semantic_task_sha256': th, 'rendered_input_sha256': prompt_hash,
            'predicted_target': target, 'prefix_world_ids': prefix_ids,
            'prediction_scope': 'fresh_full_input_recomputation_before_canonical_metadata',
            'prior_V3_semantics_and_gold_exposure': True})
        judgments.append({'receipt_id': 'belief-v4-input-only-' + sid, 'sample_id': sid,
            'record_hash': row['record_hash'], 'reviewer_role': 'independent_adjudicator_input_phase',
            'reviewer_execution_id': '/root/research_architect', 'decision': 'INPUT_ONLY_CONSISTENT',
            'critical_defects': [], 'issues': ['canonical_metadata_unseen', 'prior_V3_gold_design_and_same_model_exposure'],
            'reason': 'Own typed filter and direct original-table conjunction agree on complete output/prefixes; exact typed student-input/split join to authorized old target-free view.',
            'evidence': {'input_sha256': ih, 'prediction_file': PREFIX + 'predictions.jsonl'},
            'data_approval': False})
        joins.append({'fresh_sample_id': sid, 'prior_sample_id': previous['sample_id'],
            'fresh_record_hash_opaque': row['record_hash'], 'prior_record_hash_opaque': previous['record_hash'],
            'input_sha256': ih, 'semantic_task_sha256': th, 'rendered_input_sha256': prompt_hash,
            'supplier_lineage_hash_not_interpreted': joined['input_sha256'],
            'same_typed_student_input_and_split': True})
        targets_by_input[sid] = digest(target)
        semantic_style_groups[(family, th)].append({'sid': sid, 'target_sha': digest(target), 'prompt': row['rendered_input']})
        status_counts['nonempty' if supports[-1] else 'empty'] += 1
        prefix_sizes[str([len(s) for s in supports])] += 1
        for answer in target['queries']:
            query_flags['entailed_true' if answer['entailed'] else 'entailed_false'] += 1
            query_flags['useful_true' if answer['can_reduce_uncertainty'] else 'useful_false'] += 1
            for bucket in answer['counts']: type_counts[type(bucket['value']).__name__] += 1
    assert len(families) == len(set(family_mapping.values())) == 4 and set(families.values()) == {864}
    assert Counter(family_splits.values()) == {'train': 2, 'dev_challenge': 2}
    assert split_counts == {'train': 1728, 'dev_challenge': 1728}
    assert len(semantic_style_groups) == 1728
    for group in semantic_style_groups.values():
        assert len(group) == 2 and len({r['target_sha'] for r in group}) == 1
        assert len({r['prompt'] for r in group}) == 2

    outputs = [hand_binding, new_rows(PREFIX + 'predictions.jsonl', predictions),
               new_rows(PREFIX + 'first-input-judgments.jsonl', judgments),
               new_rows(PREFIX + 'typed-input-joins.jsonl', joins)]
    audit = {'status': 'COMPLETED_INPUT_ONLY_RECOMPUTATION_NO_ADJUDICATION',
        'actual_execution_id': '/root/research_architect', 'started_at_utc': started,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'records_recomputed': 3456, 'direct_conjunction_comparisons': 3456,
        'exact_typed_input_and_split_joins': 3456, 'unique_fresh_and_prior_ids': [3456, 3456],
        'family_counts': dict(families), 'family_mapping': family_mapping, 'family_splits': family_splits,
        'split_records': dict(split_counts), 'parsed_tasks': 1728, 'two_rendered_views_per_task': True,
        'same_task_views_have_identical_predicted_outputs': True,
        'expected_status_counts': dict(status_counts), 'expected_query_flags': dict(query_flags),
        'expected_count_bin_types': dict(type_counts), 'prefix_size_histogram': dict(prefix_sizes),
        'hand_checks': {'cases': len(hand_results), 'passed': sum(r['passed'] for r in hand_results)},
        'outputs': outputs, 'preexecution_freeze': bind(frozen_path),
        'opaque_full_candidate_record_hashes_verified': False,
        'input_view_manifest_canonical_candidate_attestation_unverified': view_manifest['bindings']['data/RBM-001/candidate_renderings/belief-gap-v4.jsonl'],
        'supplier_lineage_hash_field_scope': 'Retained verbatim; unpublished input-hash recipe not inferred. Independent canonical student-input+split hash and actual complete equality checked for every join.',
        'old_targets_or_prior_predictions_loaded': False,
        'fresh_canonical_metadata_full_manifest_gold_private_code_formal_or_other_reviews_read': False,
        'data_approval': False, 'metadata_adjudication': 'NOT_AUTHORIZED_OR_PERFORMED',
        'disclosures': config['disclosures'],
        'remaining': 'Wait for root authorization after own predictions/judgments freeze and completed fresh reviewer executions; then inspect actual canonical metadata/full record hashes/manifest/control truth and publication eligibility.'}
    audit_binding = new_json(PREFIX + 'audit.json', audit)
    print(canonical({'audit': audit_binding, 'records': 3456, 'joins': 3456,
                     'hand_checks': audit['hand_checks'], 'outputs': outputs,
                     'status_counts': dict(status_counts), 'data_approval': False}))

if __name__ == '__main__': main()
