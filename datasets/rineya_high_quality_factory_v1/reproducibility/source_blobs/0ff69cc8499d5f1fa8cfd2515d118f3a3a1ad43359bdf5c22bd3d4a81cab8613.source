"""Official prospective V4 input-only belief controls, formal_oracle authored.

Own immutable V3 mechanisms supply all finite semantics, profiles, scoped
probes and complete conditional cohorts. This module adds the published V4
owner/ordered-pair/manifest resolver and exact-byte custody. Producer diagnostic
annotations are not a required private proof language: the proof here is derived
again from actual supplied tasks. No target is used to derive an answer.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
from . import belief_controls_v3_independent as v3

v1, v2 = v3.v1, v3.v2
BeliefControlError = v1.BeliefControlError
_need, _json, _meta = v1._need, v1._json, v1._meta
INHERITED = v2.NEW_FAMILY
DEPENDENCIES = {
    'oracles/belief_controls_independent.py': 'ec9f387485f2a2d58ab557db0a2a7d95c24e77cb3108afa503a80c575e04b03e',
    'oracles/belief_controls_v2_independent.py': '5b97b73df0b9248b1ad155307e75e2a73b995dff64e81232382a6c05a2fe4984',
    'oracles/belief_controls_v3_independent.py': 'ff1a72153e985810d2771fecf299944b39cc8afa52effbd833feac056f143271',
    'factory/common.py': '8502429a270c0a2b0ee74ce89118b28c780b7242c58d8d9c6e858ebcd43ef7c8',
    'schemas/canonical/v1.schema.json': 'f4d8276fe000ceafb0f70c69dbfd1608b823767487d7c0a79623f82cf0daf25f',
}
PUBLIC_V4 = {
    'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md': '40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce',
    'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md': '925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c',
}
REVIEW_FIELDS = frozenset(['record_hash', 'review_status', 'review_receipt_ids', 'reviewed_candidate_hash'])
ENTRY_MINIMUM = frozenset(['name', 'scope', 'source_family_key', 'auxiliary_scalar_variant', 'observation_length', 'control_id', 'pair_sample_ids', 'exact_candidate_sha256'])


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record_hash(row):
    """Reimplement the published canonical full-record formula independently."""
    _need(type(row) is dict, 'record object required')
    return _sha(_json({k: value for k, value in row.items() if k != 'record_hash'}).encode('utf8'))


def _valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _record(row):
    _need(type(row) is dict and _valid_hash(row.get('record_hash')), 'canonical full record_hash required')
    _need(row['record_hash'] == record_hash(row), 'full canonical record_hash mismatch')


def _path(value):
    _need(type(value) is str and bool(value), 'evidence_manifest nonempty path required')
    _need(not value.startswith(('/', '\\')) and not re.match('^[A-Za-z]:', value) and '..' not in value.replace('\\', '/').split('/'), 'evidence_manifest repository-relative file required')
    _need(str(PurePosixPath(value)) not in ('', '.'), 'evidence_manifest must name a file')
    return value


def _claims(row):
    # Original closed claim shape is enforced by the immutable V3 parser.
    claims = v3._claims(row)
    for claim in claims:
        if claim['property'] in INHERITED and claim['scope'] == 'corpus_variant':
            _need(all(key in claim for key in ('pair_sample_ids', 'control_id', 'evidence_manifest')), 'V4 inherited corpus claim requires inline pair_sample_ids, control_id, evidence_manifest')
            _path(claim['evidence_manifest'])
    return claims


def _coordinate(row):
    _need(type(row) is dict, 'coordinate row object required')
    family, variant = _meta(row, 'source_family_key'), _meta(row, 'auxiliary_scalar_variant')
    schedule, renderer = _meta(row, 'schedule_index'), _meta(row, 'renderer')
    context = _meta(row, 'controlled_context')
    _need(type(family) is str and bool(family), 'actual source_family_key required')
    _need(type(variant) is str and variant in v3.VARIANTS, 'actual scalar variant required')
    _need(type(schedule) is int and 0 <= schedule <= 26, 'strict integer schedule_index 0..26 required')
    _need(type(renderer) is str and renderer in ('forward', 'reverse'), 'actual renderer required')
    _need(type(context) is str and context in tuple(v2.CONTEXTS), 'actual A-H context required')
    task = row.get('semantic_task'); actual = v1.analyze(task)
    encoding = v1._encoding(v2._projection(row), actual)
    decoded = v1._decode(row, {**encoding, 'context': context})
    length = len(task['observations'])
    _need(type(length) is int and length in (2, 3), 'actual observation length 2 or3 required')
    return {'family': family, 'variant': variant, 'schedule': schedule, 'renderer': renderer, 'context': context, 'length': length, 'decoded': decoded}


def check(row):
    """Return row errors; manifest/pair existence is resolved only in check_batch."""
    errors = v3.check(row)
    try:
        _record(row); _coordinate(row); _claims(row)
    except (BeliefControlError, KeyError, TypeError, ValueError) as error:
        errors.append('belief_control_v4_row: ' + str(error))
    return errors


def _strict_json(raw, label):
    _need(type(raw) is bytes, label + ' requires actual bytes')
    def pairs(items):
        output = {}
        for key, value in items:
            _need(key not in output, label + ': duplicate JSON key ' + key)
            output[key] = value
        return output
    def constant(value):
        raise BeliefControlError(label + ': nonfinite JSON scalar ' + value)
    return json.loads(raw.decode('utf8'), object_pairs_hook=pairs, parse_constant=constant)


def _candidate(raw):
    _need(type(raw) is bytes and bool(raw), 'UNKNOWN: preserved complete candidate_bytes required')
    lines = raw.splitlines(); _need(all(line.strip() for line in lines), 'candidate JSONL contains an empty record')
    records = [_strict_json(line, 'candidate record') for line in lines]
    ids = {}
    for row in records:
        _record(row); sample = row.get('sample_id')
        _need(type(sample) is str and sample and sample not in ids, 'candidate requires distinct actual sample IDs')
        ids[sample] = row
    return records, ids, _sha(raw)


def _bind_rows(rows, candidate_ids, candidate_sha):
    _need(type(rows) is list, 'batch requires a list')
    seen = set(); bindings = []
    for row in rows:
        _record(row); sample = row.get('sample_id')
        _need(type(sample) is str and sample not in seen and sample in candidate_ids, 'actual row must resolve uniquely in preserved candidate')
        seen.add(sample); source = candidate_ids[sample]
        if _json(row) != _json(source):
            _need(row.get('review_status') == 'ACCEPTED' and row.get('reviewed_candidate_hash') == source['record_hash'], 'nonidentical release row must bind its preserved candidate record')
            _need(type(row.get('review_receipt_ids')) is list and all(type(value) is str for value in row['review_receipt_ids']), 'review receipt list requires string IDs')
            _need(_json({k: v for k, v in row.items() if k not in REVIEW_FIELDS}) == _json({k: v for k, v in source.items() if k not in REVIEW_FIELDS}), 'release mutation exceeds documented review fields')
        bindings.append({'sample_id': sample, 'actual_record_hash': row['record_hash'], 'candidate_record_hash': source['record_hash'], 'candidate_sha256': candidate_sha})
    _need(seen == set(candidate_ids), 'supplied rows do not cover exact complete preserved candidate')
    return bindings


def _manifest(raw, path):
    parsed = _strict_json(raw, 'manifest ' + path)
    _need(type(parsed) is dict, 'manifest object required')
    return parsed, {'path': path, 'sha256': _sha(raw), 'bytes': len(raw)}


def _pair(owner, claim, actual_ids):
    coordinates = _coordinate(owner); pair = claim['pair_sample_ids']
    _need(all(sample in actual_ids for sample in pair), 'inline pair IDs do not resolve to supplied records')
    left, right = (actual_ids[sample] for sample in pair)
    lc, rc = _coordinate(left), _coordinate(right)
    for coordinate in (lc, rc):
        _need(all(coordinate[key] == coordinates[key] and type(coordinate[key]) is type(coordinates[key]) for key in ('family', 'variant', 'schedule', 'renderer', 'length')), 'evidence pair does not match actual owner family/variant/schedule/renderer/length')
        _need(_json(coordinate['decoded']['worlds']) == _json(coordinates['decoded']['worlds']), 'owner and evidence decoded source tables differ')
    expected = ('E', 'G') if coordinates['length'] == 2 else ('F', 'H')
    _need((lc['context'], rc['context']) == expected, 'inline pair requires ordered E/G or F/H at actual owner length')
    proof = v2.matched_pair(left['semantic_task'], right['semantic_task'])
    return coordinates, left, right, proof


def resolve_claim(owner, claim, actual_ids, manifest, candidate_sha):
    """Public resolver primitive for focused handmade tests and auditable proof."""
    _need(type(claim) is dict and claim.get('scope') == 'corpus_variant', 'inherited supplied claim required')
    names = set(claim) & {'name', 'property'}
    _need(len(names) == 1, 'exactly one inherited name/property required')
    name = claim[next(iter(names))]
    _need(type(name) is str and name in INHERITED, 'unknown inherited name')
    # Normalize through the original closed parser, including source-family check.
    normalized_owner = deepcopy(owner)
    v2._setmeta(normalized_owner, 'family_control_claims', [claim])
    checked = _claims(normalized_owner)[0]
    _need(type(manifest) is dict and type(manifest.get('supplied_same_table_v2_pairs')) is list, 'published manifest collection required')
    _need(_valid_hash(candidate_sha), 'exact complete candidate SHA256 required')
    coordinates, left, right, detail = _pair(owner, checked, actual_ids)
    for row in (owner, left, right):
        _record(row)
    matches = [entry for entry in manifest['supplied_same_table_v2_pairs'] if type(entry) is dict and entry.get('control_id') == checked['control_id'] and entry.get('name') == name and _json(entry.get('pair_sample_ids')) == _json(checked['pair_sample_ids'])]
    _need(len(matches) == 1, 'named ID/name/ordered-pair resolver requires exactly one entry')
    entry = matches[0]
    _need(ENTRY_MINIMUM <= set(entry), 'resolved named entry missing published minimum fields')
    _need(type(entry['scope']) is str and entry['scope'] == 'corpus_variant', 'resolved entry scope must be corpus_variant')
    for field, value in [('source_family_key', coordinates['family']), ('auxiliary_scalar_variant', coordinates['variant']), ('observation_length', coordinates['length'])]:
        _need(type(entry[field]) is type(value) and entry[field] == value, 'resolved entry typed '+field+' mismatch')
    _need(_valid_hash(entry['exact_candidate_sha256']) and entry['exact_candidate_sha256'] == candidate_sha, 'resolved entry exact candidate SHA256 mismatch')
    # Annotation dictionaries are intentionally open. This receipt is the actual
    # parsed comparison/prefix evidence, rather than acceptance of a pointer.
    return {'owner_sample_id': owner['sample_id'], 'property': name, 'scope': 'corpus_variant', 'control_id': checked['control_id'], 'evidence_manifest': checked['evidence_manifest'], 'pair_sample_ids': checked['pair_sample_ids'], 'owner_is_pair_member': owner['sample_id'] in checked['pair_sample_ids'], 'owner_coordinate': {k: coordinates[k] for k in ('family', 'variant', 'schedule', 'renderer', 'length')}, 'pair_record_hashes': [left.get('record_hash'), right.get('record_hash')], 'parsed_input_sha256': [_sha(_json(r['semantic_task']).encode('utf8')) for r in (left, right)], 'exact_candidate_sha256': candidate_sha, 'derived_comparison_prefix_evidence': detail, 'satisfied': True}


def preservation(rows, previous_rows):
    """Optional exact typed V4/V3 triples proof; never substitutes cohort checks."""
    if previous_rows is None:
        return {'status': 'NOT_RUN', 'errors': [], 'reason': 'No previous_rows supplied; zero semantic growth must not be claimed from this report.'}
    errors = []; maps = []
    try:
        for records in (previous_rows, rows):
            _need(type(records) is list, 'previous/current rows list required'); mapping = {}
            for row in records:
                coordinate = _coordinate(row)
                key = tuple(coordinate[k] for k in ('family', 'context', 'variant', 'schedule', 'renderer'))
                _need(key not in mapping, 'duplicate stable comparison coordinate')
                mapping[key] = row
            maps.append(mapping)
        _need(set(maps[0]) == set(maps[1]), 'stable previous/current coordinate coverage differs')
        for key in sorted(maps[0]):
            a, b = (mapping[key] for mapping in maps)
            _need(_json(a['semantic_task']) == _json(b['semantic_task']) and a['rendered_input'] == b['rendered_input'] and _json(a['verified_target']) == _json(b['verified_target']), 'previous/current task/rendered/typed-target triple differs at '+str(key))
            expected = v1.derive(b['semantic_task'])['target']
            _need(_json(a['verified_target']) == _json(expected), 'preserved supplied target differs from independently derived target')
        return {'status': 'PROVED', 'errors': [], 'matched_triples': len(rows), 'added_semantic_tasks': 0, 'added_source_families': 0}
    except (BeliefControlError, KeyError, TypeError, ValueError) as error:
        errors.append('V4 preservation: '+str(error))
        return {'status': 'FAILED', 'errors': errors}


def check_batch(rows, manifest_sources=None, candidate_bytes=None, previous_rows=None):
    """Return errors/counts plus derived evidence, using explicit raw-byte ledger.

    manifest_sources maps actual repository-relative file names to exact bytes.
    Missing bytes fail closed as UNKNOWN; filesystem pointers alone are not proof.
    Accepted rows may differ from their preserved candidate only in documented
    review fields, and their full hashes must independently remain valid.
    """
    if type(rows) is not list:
        return {'errors': ['V4 batch requires a list'], 'counts': {'records': 0}, 'approval_verdict': None}
    try:
        report = v3.check_batch(rows)
    except Exception as error:
        report = {'errors': ['UNKNOWN: inherited V3 execution failed: '+type(error).__name__+': '+str(error)], 'counts': {'records': len(rows)}}
    errors = report['errors']; proofs = []; manifests = {}; manifest_bindings = []; row_bindings = []; actual_ids = {}; candidate_sha = None
    for row in rows:
        errors.extend(str(row.get('sample_id', '?') if type(row) is dict else '?')+': '+error for error in check(row))
    try:
        _, actual_ids, candidate_sha = _candidate(candidate_bytes)
        row_bindings = _bind_rows(rows, actual_ids, candidate_sha)
        _need(type(manifest_sources) is dict, 'UNKNOWN: manifest_sources actual-byte ledger required')
        for path, raw in manifest_sources.items():
            _path(path); parsed, binding = _manifest(raw, path); manifests[path] = parsed; manifest_bindings.append(binding)
    except (BeliefControlError, KeyError, TypeError, ValueError, UnicodeError) as error:
        errors.append('V4 byte custody: '+str(error))
    for row in rows:
        try:
            for claim in _meta(row, 'family_control_claims'):
                name = claim.get('name', claim.get('property')) if type(claim) is dict else None
                if name not in INHERITED or claim.get('scope') != 'corpus_variant':
                    continue
                _need(candidate_sha is not None and bool(row_bindings), 'UNKNOWN: actual complete candidate custody not proved')
                path = claim.get('evidence_manifest'); _path(path)
                _need(path in manifests, 'UNKNOWN: evidence_manifest has no actual supplied bytes')
                proofs.append(resolve_claim(row, claim, actual_ids, manifests[path], candidate_sha))
        except (BeliefControlError, KeyError, TypeError, ValueError) as error:
            errors.append(str(row.get('sample_id', '?') if type(row) is dict else '?')+': V4 resolver: '+str(error))
    retained = preservation(rows, previous_rows); errors.extend(retained['errors'])
    distinct = {(p['property'], p['control_id'], tuple(p['pair_sample_ids']), p['evidence_manifest']) for p in proofs}
    length_counts = Counter(str(p['owner_coordinate']['length']) for p in proofs)
    report['counts'].update(v4_resolved_inherited_claims=len(proofs), v4_distinct_named_pair_entries=len(distinct), v4_manifest_files=len(manifest_bindings), v4_bound_records=len(row_bindings), v4_nonmember_owner_claims=sum(not p['owner_is_pair_member'] for p in proofs))
    report.update(v4_resolver_proofs=proofs, v4_claim_length_counts=dict(sorted(length_counts.items())), v4_manifest_byte_bindings=sorted(manifest_bindings, key=lambda item: item['path']), v4_record_byte_bindings=row_bindings, v4_exact_candidate_sha256=candidate_sha, v4_preservation=retained, v4_dependency_sha256=DEPENDENCIES, v4_public_sha256=PUBLIC_V4, v4_proof_annotation_policy='No private annotation shape is required. Every resolved pair has an independently recomputed typed difference, exact prefix sets and complete query partitions in this report.', approval_verdict=None)
    return report
