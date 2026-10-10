"""Hand-only V4 reference/custody component; no semantic solver or production reads."""
import hashlib
import json
from pathlib import Path
import sys

NAMES = {'same_prefix_empty_nonempty_counterfactual', 'contradiction_equals_mixed_outcomes'}
COORDINATES = ('source_family_key', 'auxiliary_scalar_variant', 'schedule_index', 'renderer')
CLAIM_FIELDS = {'name', 'property', 'scope', 'source_family_key', 'evidence_manifest',
                'control_id', 'pair_sample_ids', 'probe_id'}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def typed_equal(left, right):
    return canonical(left) == canonical(right)

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def verify_hash(row):
    actual = hashlib.sha256(canonical({k: v for k, v in row.items() if k != 'record_hash'}).encode()).hexdigest()
    require(actual == row['record_hash'], 'full_record_hash')

def resolve(owner, claim, records, manifest, candidate_sha):
    """Resolve only the public inherited reference obligations; prove no target truth."""
    verify_hash(owner)
    require(set(claim) <= CLAIM_FIELDS, 'claim_vocabulary')
    require(('name' in claim) != ('property' in claim), 'exactly_one_identifier')
    name = claim.get('name', claim.get('property'))
    require(name in NAMES and claim.get('scope') == 'corpus_variant', 'inherited_supplied_scope')
    for key in ('evidence_manifest', 'control_id'):
        require(type(claim.get(key)) is str and bool(claim[key]), key)
    provenance = Path(claim['evidence_manifest'])
    require(not provenance.is_absolute() and '..' not in provenance.parts, 'relative_provenance')
    require('probe_id' not in claim, 'probe_substitution')
    if 'source_family_key' in claim:
        require(typed_equal(claim['source_family_key'], owner['source_family_key']), 'claim_family')
    pair = claim.get('pair_sample_ids')
    require(type(pair) is list and len(pair) == 2 and all(type(s) is str and bool(s) for s in pair)
            and pair[0] != pair[1], 'ordered_two_distinct_ids')
    require(all(s in records for s in pair), 'actual_pair_members')
    members = [records[s] for s in pair]
    length = len(owner['semantic_task']['observations'])
    require(length in (2, 3), 'actual_owner_length')
    expected = ('E', 'G') if length == 2 else ('F', 'H')
    require(tuple(r['controlled_context'] for r in members) == expected, 'length_specific_order')
    for member in members:
        verify_hash(member)
        require(len(member['semantic_task']['observations']) == length, 'actual_member_length')
        for coordinate in COORDINATES:
            require(typed_equal(member[coordinate], owner[coordinate]), 'owner_' + coordinate)
    require(type(candidate_sha) is str and len(candidate_sha) == 64
            and all(c in '0123456789abcdef' for c in candidate_sha), 'candidate_sha_shape')
    matches = [entry for entry in manifest['supplied_same_table_v2_pairs']
               if entry.get('control_id') == claim['control_id'] and entry.get('name') == name
               and typed_equal(entry.get('pair_sample_ids'), pair)]
    require(len(matches) == 1, 'unique_named_ordered_resolver')
    entry = matches[0]
    minimum = {'name', 'scope', 'source_family_key', 'auxiliary_scalar_variant',
               'observation_length', 'control_id', 'pair_sample_ids', 'exact_candidate_sha256'}
    require(minimum <= set(entry), 'entry_minimum_fields')
    require(entry['scope'] == 'corpus_variant', 'entry_scope')
    require(entry['exact_candidate_sha256'] == candidate_sha, 'exact_candidate_sha')
    require(type(entry['observation_length']) is int and entry['observation_length'] == length,
            'entry_actual_length')
    for coordinate in ('source_family_key', 'auxiliary_scalar_variant'):
        require(typed_equal(entry[coordinate], owner[coordinate]), 'entry_' + coordinate)
    return {'name': name, 'pair_sample_ids': pair, 'actual_observation_length': length}

def main():
    fixtures_path, output_path = map(Path, sys.argv[1:])
    require(not output_path.exists(), 'new_output_only')
    fixtures = json.loads(fixtures_path.read_text())
    results = []
    for case in fixtures['cases']:
        try:
            resolved = resolve(case['owner'], case['claim'], case['records'], case['manifest'], case['candidate_sha'])
            actual, reason = 'PASS_COMPONENT', None
        except ValueError as exc:
            resolved, actual, reason = None, 'FAIL_COMPONENT', str(exc)
        passed = actual == case['expected'] and (case.get('expected_reason') is None or reason == case['expected_reason'])
        results.append({'case_id': case['case_id'], 'actual': actual, 'reason': reason,
                        'expected': case['expected'], 'test_passed': passed, 'resolved': resolved})
    report = {'actual_execution_id': '/root/research_architect', 'scope': 'HAND_REFERENCE_COMPONENT_ONLY',
              'production_inputs_read': False, 'semantic_pairs_or_targets_validated': False,
              'production_approval': False, 'fixture_sha256': hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tests': len(results), 'passed': sum(r['test_passed'] for r in results), 'results': results}
    with output_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('scope', 'tests', 'passed', 'production_approval')}))
    require(all(r['test_passed'] for r in results), 'hand_test_failed')

if __name__ == '__main__':
    main()
