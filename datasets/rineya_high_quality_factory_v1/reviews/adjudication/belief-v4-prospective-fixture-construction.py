"""Explicit handmade reference cases; these contain no worlds, targets or production IDs."""
import copy
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).with_name('belief-v4-prospective-hand-fixtures.json')
SHA = '1' * 64  # Hand sentinel; never an actual candidate assertion.
N1 = 'same_prefix_empty_nonempty_counterfactual'
N2 = 'contradiction_equals_mixed_outcomes'

def seal(row):
    value = {k: v for k, v in row.items() if k != 'record_hash'}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    row['record_hash'] = hashlib.sha256(raw).hexdigest()
    return row

def base(length=2, variant='identity'):
    contexts = ('E', 'G') if length == 2 else ('F', 'H')
    def row(sid, context):
        return seal({'sample_id': sid, 'source_family_key': 'hand-family',
                     'auxiliary_scalar_variant': variant, 'schedule_index': 0, 'renderer': 'forward',
                     'controlled_context': context,
                     'semantic_task': {'observations': [{'fact': 'hand-x', 'equals': 'hand-value'}] * length}})
    members = [row('hand-left', contexts[0]), row('hand-right', contexts[1])]
    owner = row('hand-owner', 'A' if length == 2 else 'D')
    claim = {'name': N1, 'scope': 'corpus_variant', 'control_id': 'shared-hand-id',
             'evidence_manifest': 'hand/path.json', 'pair_sample_ids': ['hand-left', 'hand-right']}
    entries = [{'name': name, 'scope': 'corpus_variant', 'source_family_key': 'hand-family',
                'auxiliary_scalar_variant': variant, 'observation_length': length,
                'control_id': 'shared-hand-id', 'pair_sample_ids': ['hand-left', 'hand-right'],
                'exact_candidate_sha256': SHA} for name in (N1, N2)]
    return {'owner': owner, 'claim': claim, 'records': {r['sample_id']: r for r in members},
            'manifest': {'supplied_same_table_v2_pairs': entries, 'allowed_extra_diagnostic': True},
            'candidate_sha': SHA}

cases = []
def add(name, value, reason=None, reseal=True):
    if reseal:
        seal(value['owner'])
        for row in value['records'].values(): seal(row)
    value.update({'case_id': name, 'expected': 'PASS_COMPONENT' if reason is None else 'FAIL_COMPONENT'})
    if reason is not None: value['expected_reason'] = reason
    cases.append(value)

for length in (2, 3):
    for variant in ('identity', 'transposed'):
        add(f'valid-nonmember-{length}-{variant}', base(length, variant))
v = base(); v['claim']['name'] = N2
add('shared-id-different-name-is-legal', v)
v = base(); v['claim']['property'] = v['claim'].pop('name')
add('property-identifier-alias-is-legal', v)
v = base(); v['claim']['evidence_manifest'] = '../hand/path.json'
add('relative-path-no-extra-private-normalization', v)
v = base(); v['manifest']['supplied_same_table_v2_pairs'][0]['pair_record_hashes'] = ['extra', 'diagnostic']
add('optional-manifest-diagnostics-not-authority', v)
v = base(); v['owner'] = copy.deepcopy(v['records']['hand-left'])
add('member-owner-also-legal', v)
for coordinate, replacement in [('source_family_key', 'foreign-family'),
                                 ('auxiliary_scalar_variant', 'transposed'),
                                 ('schedule_index', 1), ('renderer', 'reverse')]:
    v = base(); v['records']['hand-left'][coordinate] = replacement
    add('foreign-owner-' + coordinate, v, 'owner_' + coordinate)
v = base(); v['records']['hand-left']['schedule_index'] = False
add('bool-does-not-alias-integer-coordinate', v, 'owner_schedule_index')
v = base(); v['claim']['pair_sample_ids'].reverse()
add('reversed-pair-invalid', v, 'length_specific_order')
v = base(3); v['owner']['semantic_task']['observations'].pop()
add('actual-owner-length-over-annotation', v, 'length_specific_order')
v = base(); v['records']['hand-left']['semantic_task']['observations'].append({'fact': 'hand-x', 'equals': None})
add('actual-member-length-invalid', v, 'actual_member_length')
v = base(); v['manifest']['supplied_same_table_v2_pairs'].append(copy.deepcopy(v['manifest']['supplied_same_table_v2_pairs'][0]))
add('duplicate-identical-named-key-invalid', v, 'unique_named_ordered_resolver')
v = base(); v['manifest']['supplied_same_table_v2_pairs'] = v['manifest']['supplied_same_table_v2_pairs'][1:]
add('other-name-cannot-substitute', v, 'unique_named_ordered_resolver')
v = base(); v['manifest']['supplied_same_table_v2_pairs'][0]['exact_candidate_sha256'] = '2' * 64
add('wrong-whole-candidate-sha-invalid', v, 'exact_candidate_sha')
v = base(); v['manifest']['supplied_same_table_v2_pairs'][0]['observation_length'] = True
add('typed-entry-length-invalid', v, 'entry_actual_length')
v = base(); del v['manifest']['supplied_same_table_v2_pairs'][0]['scope']
add('missing-public-entry-field-invalid', v, 'entry_minimum_fields')
v = base(); del v['claim']['pair_sample_ids']
add('missing-inline-pair-invalid', v, 'ordered_two_distinct_ids')
v = base(); v['claim']['pair_sample_ids'][1] = 'unknown-hand-id'
add('absent-actual-record-invalid', v, 'actual_pair_members')
v = base(); v['records']['hand-left']['record_hash'] = '0' * 64
add('bad-actual-full-record-hash-invalid', v, 'full_record_hash', reseal=False)
v = base(); v['claim']['probe_id'] = 'unexecuted-hand-probe'
add('probe-cannot-substitute-for-supplied-pair', v, 'probe_substitution')
v = base(); v['claim']['name'] = N1; v['claim']['property'] = N1
add('two-identifiers-invalid', v, 'exactly_one_identifier')
v = base(); v['claim']['unpublished_key'] = 'hand'
add('unknown-claim-field-invalid', v, 'claim_vocabulary')

payload = {'actual_author_execution_id': '/root/research_architect',
           'scope': 'HAND_REFERENCE_ONLY_NO_TARGETS', 'production_data': False,
           'semantic_pair_truth_not_asserted': True, 'candidate_sha_is_hand_sentinel': True,
           'cases': cases}
with OUT.open('x') as stream:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps({'path': str(OUT), 'cases': len(cases), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))
