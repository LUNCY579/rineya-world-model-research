"""Versioned, reversible model-visible surfaces over immutable accepted gold.

JSON object order, JSON whitespace, fixed framing and gold-independent candidate
order change. Search prefixes and all other arrays, values and ops stay fixed.
This module exports data and never starts a model or training process.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil

from ..auditors.integrity import (audit_record, recursive_input_check, semantic_fingerprint, validate_auxiliary,
                                  validate_decision, validate_schema)
from ..oracles.primary import evaluate as primary, ORACLE_VERSION as PRIMARY_VERSION
from ..oracles.independent import evaluate as independent, ORACLE_VERSION as INDEPENDENT_VERSION
from ..pipelines.common import canonical, digest, file_hash, immutable_bytes, source_hashes

RENDERING_VERSION = 'gcdl-rendering-1.0'
V2_RENDERING_VERSION = 'gcdl-rendering-2.0'
RENDERING_POLICY_SET_VERSION = 'gcdl-rendering-policies-2.0'
SUPPORTED_RENDERING_VERSIONS = (RENDERING_VERSION, V2_RENDERING_VERSION)
EXPORT_VERSION = 'gcdl-rendered-export-2.0'
VIEWS = ('pointwise', 'pairwise', 'listwise', 'computation', 'distillation')
VARIANTS = (
    {'name': 'canonical_compact', 'key_order': 'canonical', 'layout': 'compact', 'framing': 0},
    {'name': 'reverse_indented', 'key_order': 'reverse', 'layout': 'indented', 'framing': 1},
    {'name': 'seeded_spaced', 'key_order': 'seeded', 'layout': 'spaced', 'framing': 2},
)
MAX_SHARD_BYTES = 8 * 1024 ** 2 - 1  # Every shard is strictly below 8 MiB.
MAX_PROGRAMME_BYTES = 512 * 1024 ** 2
MIN_FREE_BYTES = 10 * 1024 ** 3
MAX_RECORDS = 12000
BEGIN = 'TASK_JSON_BEGIN\n'
END = '\nTASK_JSON_END\n'
_FRAMING = (
    'Use the task JSON below to assess the supplied candidate plans under its stated rules, goals, constraints and utility definition.\n',
    'Assess the supplied candidate plans using the rules, goals, constraints and utility definition in this JSON task.\n',
    'The following JSON specifies a decision task. Apply its rules, goals, constraints and utility definition to the supplied plans.\n',
)
_REQUEST = {
    'pointwise': 'Return the exact utility and success and feasibility probabilities for the supplied candidate.\n',
    'pairwise': 'Compare the first and second supplied candidates using the declared utility ordering.\n',
    'listwise': 'Compare all supplied candidates using the declared utility ordering.\n',
    'computation': 'Compare the supplied candidates. The evaluation policies consider only the first candidate or every supplied candidate.\n',
    'distillation': 'Compare the supplied alternatives and consider a plan under the stated task.\n',
}


def _bounds(view, max_records=None, variant_budget=3):
    if view not in VIEWS:
        raise ValueError('unsupported rendered view')
    if type(variant_budget) is not int or not 1 <= variant_budget <= len(VARIANTS):
        raise ValueError('variant_budget must be an integer from 1 to 3')
    if max_records is not None and (type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS):
        raise ValueError('max_records must be an integer from 1 to 12000')


def _ordered(value, policy, seed, path=()):
    if type(value) is dict:
        keys = sorted(value)
        if policy == 'reverse':
            keys.reverse()
        elif policy == 'seeded':
            keys.sort(key=lambda key: digest([seed, list(path), key]))
            ascending = sorted(value)
            # At least three keys admit a third order. Guarantee one instead
            # of accidentally drawing canonical/reverse from the finite set.
            if len(keys) >= 3 and keys in (ascending, list(reversed(ascending))):
                keys = ascending[1:] + ascending[:1]
        return {key: _ordered(value[key], policy, seed, path + (key,)) for key in keys}
    if type(value) is list:
        return [_ordered(item, policy, seed, path + (index,)) for index, item in enumerate(value)]
    return value


def _json_surface(task, variant, seed):
    ordered = _ordered(task, variant['key_order'], seed)
    options = {'ensure_ascii': False, 'allow_nan': False, 'sort_keys': False}
    if variant['layout'] == 'indented':
        options['indent'] = 2
    elif variant['layout'] == 'spaced':
        options['separators'] = (', ', ': ')
    else:
        options['separators'] = (',', ':')
    return json.dumps(ordered, **options)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate task JSON key')
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError('nonfinite task JSON constant: ' + value)


def parse_input_text(input_text):
    """Decode only a registered complete prompt; reject extra text/duplicate keys.

    JSON strings may themselves contain delimiter-looking text. Fixed framing
    and suffix matching avoid treating such strings as task boundaries.
    """
    if type(input_text) is not str:
        raise ValueError('input_text must be a string')
    for framing in _FRAMING:
        prefix = framing + BEGIN
        if not input_text.startswith(prefix):
            continue
        for request in _REQUEST.values():
            suffix = END + request
            if input_text.endswith(suffix):
                task = json.loads(input_text[len(prefix):-len(suffix)],
                                  object_pairs_hook=_unique_object, parse_constant=_bad_constant)
                if type(task) is not dict:
                    raise ValueError('task JSON must be an object')
                recursive_input_check(task)
                canonical(task)
                return task
    raise ValueError('unregistered or incomplete prompt framing')


def _accepted_source(record):
    if record.get('quality_gate', {}).get('status') != 'accepted':
        raise ValueError('rendering requires quality_gate.status=accepted')
    if record.get('split') not in ('train', 'dev', 'ood'):
        raise ValueError('sealed or unknown split cannot become rendered exports')
    evidence = audit_record(record)
    if record.get('verification', {}).get('evidence_sha256') != evidence['evidence_sha256']:
        raise ValueError('stored accepted-source verification evidence hash mismatch')
    return deepcopy(record['targets']['oracle'])


def _projection(candidate):
    return {key: deepcopy(candidate[key]) for key in
            ('id', 'utility', 'success_probability', 'feasible_probability')}


def _score(candidate):
    return tuple(Fraction(part['numerator'], part['denominator']) for part in candidate['utility'])


def _subviews(record, view):
    task, targets = record['input'], record['targets']
    by_id = {candidate['id']: candidate for candidate in targets['decision']['candidates']}
    if view == 'computation' and 'search' not in targets:
        return
    if view == 'computation' and (targets['search']['cheap']['candidate_budget'] != 1
                                  or targets['search']['expanded']['candidate_budget'] != len(task['candidates'])):
        raise ValueError('computation framing requires the fixed first-versus-all candidate policies')
    if view == 'distillation' and not {'search', 'bounded_proof'}.intersection(targets):
        return
    if view in ('listwise', 'computation', 'distillation'):
        target = deepcopy(targets['decision']) if view == 'listwise' else {
            key: deepcopy(targets[key]) for key in ('decision', 'search', 'bounded_proof')
            if key in targets and (view == 'distillation' or key != 'bounded_proof')}
        yield deepcopy(task), target
        return
    subsets = ((candidate,) for candidate in task['candidates']) if view == 'pointwise' else combinations(task['candidates'], 2)
    for subset in subsets:
        item_input = deepcopy(task)
        item_input['candidates'] = deepcopy(list(subset))
        if view == 'pointwise':
            target = deepcopy(by_id[subset[0]['id']])
        else:
            left, right = (by_id[candidate['id']] for candidate in subset)
            target = {'left_id': left['id'], 'right_id': right['id'],
                      'preference': (_score(left) > _score(right)) - (_score(left) < _score(right)),
                      'left_utility': deepcopy(left['utility']), 'right_utility': deepcopy(right['utility'])}
        yield item_input, target


def _verify_target(task, target, view, oracle):
    if view == 'pointwise':
        if target != _projection(oracle['candidates'][0]):
            raise ValueError('pointwise target changed')
    elif view == 'pairwise':
        left, right = oracle['candidates']
        expected = {'left_id': left['id'], 'right_id': right['id'],
                    'preference': (_score(left) > _score(right)) - (_score(left) < _score(right)),
                    'left_utility': left['utility'], 'right_utility': right['utility']}
        if target != expected:
            raise ValueError('pairwise target changed')
    else:
        validate_decision(target if view == 'listwise' else target['decision'], oracle)
        if view in ('computation', 'distillation'):
            validate_auxiliary({'input': task, 'targets': target})


def _rendering_identity(record):
    """Preserve v1 exactly; v2 goal triplets share a gold-independent surface."""
    provenance = record['provenance']
    if provenance['generator_version'] != 'generator-2.0':
        return RENDERING_VERSION, provenance['record_index'], False
    if record['family'] == 'C':
        from ..generators.version2 import bundle_identity
        bundle = record['metadata'].get('goal_bundle_id')
        if bundle != bundle_identity(record['input']):
            raise ValueError('v2 goal-bundle surface identity mismatch')
        return V2_RENDERING_VERSION, bundle, True
    return V2_RENDERING_VERSION, provenance['record_index'], False


def _variant_task(record, task, target, view, variant):
    rendered, projected = deepcopy(task), deepcopy(target)
    permutable = view in ('pairwise', 'listwise', 'distillation') and 'search' not in record['targets']
    if permutable:
        provenance = record['provenance']
        _, order_identity, _ = _rendering_identity(record)
        # No decision labels, optimal choice, record_id (which hashes targets),
        # family or split enters this ordering function.
        rendered['candidates'].sort(key=lambda candidate: digest([
            provenance['seed'], order_identity, variant['name'],
            'candidate_order', candidate['id']]))
        ids = [candidate['id'] for candidate in rendered['candidates']]
        by_id = {candidate['id']: candidate for candidate in record['targets']['decision']['candidates']}
        if view == 'pairwise':
            left, right = (by_id[name] for name in ids)
            projected = {'left_id': left['id'], 'right_id': right['id'],
                         'preference': (_score(left) > _score(right)) - (_score(left) < _score(right)),
                         'left_utility': deepcopy(left['utility']), 'right_utility': deepcopy(right['utility'])}
        else:
            decision = projected if view == 'listwise' else projected['decision']
            decision['candidates'] = [deepcopy(by_id[name]) for name in ids]
    return rendered, projected, permutable


def _source_equivalent(parsed, source_task):
    """Recover the exact canonical subset after undoing candidate order only."""
    restored = deepcopy(parsed)
    candidates = {candidate['id']: candidate for candidate in parsed['candidates']}
    source_ids = [candidate['id'] for candidate in source_task['candidates']]
    if len(candidates) != len(parsed['candidates']) or set(candidates) != set(source_ids):
        return False
    restored['candidates'] = [candidates[name] for name in source_ids]
    return canonical(restored) == canonical(source_task)


def render_training_views(record, view='listwise', variant_budget=3):
    """Yield accepted-only, independently rechecked prompt/target examples.

    Each candidate subset keeps canonical gold from the original record. Both
    oracles execute every parsed prompt and must match its original subset.
    Computation/distillation filter to records with the relevant supervision.
    """
    _bounds(view, variant_budget=variant_budget)
    original = _accepted_source(record)
    version, order_identity, bundled = _rendering_identity(record)
    original_by_id = {candidate['id']: candidate for candidate in original['candidates']}
    for task, target in _subviews(record, view):
        candidate_ids = [candidate['id'] for candidate in task['candidates']]
        seed = digest([version, record['provenance']['seed'],
                       order_identity, view, candidate_ids])
        for variant in VARIANTS[:variant_budget]:
            rendered, rendered_target, permutable = _variant_task(record, task, target, view, variant)
            expected = deepcopy(original)
            expected['candidates'] = [deepcopy(original_by_id[candidate['id']]) for candidate in rendered['candidates']]
            input_text = _FRAMING[variant['framing']] + BEGIN + _json_surface(rendered, variant, seed) + END + _REQUEST[view]
            parsed = parse_input_text(input_text)
            if canonical(parsed) != canonical(rendered) or not _source_equivalent(parsed, task):
                raise ValueError('rendered JSON changed canonical task beyond permitted candidate order')
            a, b = primary(parsed), independent(parsed)
            if a != b or a != expected:
                raise ValueError('rendered task changed original candidate-subset outcomes')
            _verify_target(parsed, rendered_target, view, b)
            yield {'record_id': record['id'],
                   'view_id': 'gcdlv_' + digest([version, record['id'], view, candidate_ids, variant['name']]),
                   'view': view, 'input_text': input_text, 'target': rendered_target,
                   'split': record['split'], 'provenance': deepcopy(record['provenance']),
                   'surface': {'rendering_version': version, **variant,
                               'candidate_order': ('goal_bundle_sha256' if bundled else 'gold_independent_sha256') if permutable else 'source_order_preserved'},
                   'verification': {'input_round_trip': 'PASS', 'original_subset_oracle_match': 'PASS',
                                    'differential_oracle': 'PASS', 'unchanged_target': 'PASS',
                                    'source_equivalence': 'exact_except_candidate_permutation',
                                    'canonical_source_subset_sha256': digest(task),
                                    'canonical_input_sha256': digest(parsed), 'target_sha256': digest(rendered_target),
                                    'primary_version': PRIMARY_VERSION, 'independent_version': INDEPENDENT_VERSION}}


def _storage_guard(root, additional_bytes=0):
    current = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    if current + additional_bytes > MAX_PROGRAMME_BYTES:
        raise RuntimeError('rendered export would exceed 512 MiB programme storage ceiling')
    if shutil.disk_usage(root).free - additional_bytes < MIN_FREE_BYTES:
        raise RuntimeError('rendered export would leave fewer than 10 GiB free')


def _publish(root, path, raw):
    base = root / 'data' / 'rendered'
    if path.resolve() != path or base not in path.parents or path.is_symlink():
        raise ValueError('rendered destination symlink or path escape')
    if path.with_name(path.name + '.tmp').is_symlink():
        raise ValueError('rendered temporary destination cannot be a symlink')
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('immutable artifact conflict: ' + str(path))
        return
    _storage_guard(root, len(raw))
    immutable_bytes(path, raw)


def _source_paths(root, split, source_paths):
    accepted = (root / 'data' / 'accepted').resolve()
    if accepted != root / 'data' / 'accepted':
        raise ValueError('data/accepted must not resolve to a different source directory')
    paths = sorted(accepted.glob(split + '-*.jsonl')) if source_paths is None else sorted(Path(path) for path in source_paths)
    checked = []
    for path in paths:
        if not path.is_absolute():
            path = root / path
        if path.is_symlink() or path.resolve().parent != accepted or not path.name.startswith(split + '-') or path.suffix != '.jsonl':
            raise ValueError('source shards must be selected split JSONL directly in data/accepted')
        if path.resolve() in checked:
            raise ValueError('duplicate selected source shard')
        checked.append(path.resolve())
    return checked


def _batch_source(root, path, split):
    """Authenticate source bytes to an existing immutable acceptance receipt."""
    relative = str(path.relative_to(root))
    seed_text = path.stem[len(split) + 1:]
    if not seed_text.isdecimal():
        raise ValueError('source shard name must identify its accepted batch seed')
    manifest_path = root / 'manifests' / ('batch-%s-%s.json' % (split, seed_text))
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.resolve() != manifest_path:
        raise ValueError('accepted source batch manifest is missing or unmanaged')
    batch = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    count = len(raw.splitlines())
    if len(raw) > MAX_SHARD_BYTES or any(not line.strip() for line in raw.splitlines()):
        raise ValueError('accepted source shard size or blank-row mismatch')
    audit = batch.get('audit', {})
    verdict = audit.get('verdict')
    quarantined = batch.get('quarantined', 0)
    if (verdict not in ('PASS', 'PARTIAL_QUARANTINE')
            or type(quarantined) is not int or quarantined < 0
            or (verdict == 'PASS' and quarantined != 0)
            or (verdict == 'PARTIAL_QUARANTINE' and quarantined == 0)
            or ('accepted' in audit and (type(audit['accepted']) is not int or audit['accepted'] != count))
            or ('quarantined' in audit and (type(audit['quarantined']) is not int or audit['quarantined'] != quarantined))):
        raise ValueError('accepted source batch audit acceptance/quarantine count mismatch')
    if (batch.get('shard') != relative or batch.get('sha256') != actual_hash
            or batch.get('accepted') != count or batch.get('bytes') != len(raw)
            or batch.get('split') != split or batch.get('seed') != int(seed_text)):
        raise ValueError('accepted source batch manifest path/hash/row-count mismatch')
    for line in raw.splitlines():
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get('split') != split
                or row.get('quality_gate', {}).get('status') != 'accepted'):
            raise ValueError('accepted source shard contains unaccepted or mismatched-split record')
    code = batch.get('code_hashes')
    from ..pipelines.worker import batch_profile
    profile = batch_profile(batch)
    index = batch.get('protected_index')
    if not isinstance(code, dict) or not code or not isinstance(index, list) or len(index) != count:
        raise ValueError('accepted source manifest lacks code or protected provenance')
    if any(not isinstance(row, list) or len(row) != 5 or row[-1] != split for row in index):
        raise ValueError('accepted source protected index mismatch')
    if len({row[0] for row in index}) != count:
        raise ValueError('duplicate canonical record in source protected index')
    return {'path': relative, 'sha256': actual_hash, 'selected_records': 0,
            'source_generation_profile':profile['id'],
            'source_seed': batch['seed'],
            'source_batch_manifest': str(manifest_path.relative_to(root)),
            'source_batch_manifest_sha256': file_hash(manifest_path),
            'source_code_hashes': code, 'protected_index': index}


def _source_record_provenance(record, source):
    index = next((row for row in source['protected_index'] if row[0] == record['id']), None)
    provenance = record['provenance']
    from ..generators.profiles import resolve_record_profile
    if resolve_record_profile(provenance)['id']!=source['source_generation_profile']:
        raise ValueError('accepted source record/batch generation profile mismatch')
    if index is None or index[2:] != [provenance['parent_world_id'], provenance['template_group_id'], record['split']]:
        raise ValueError('accepted source canonical ID/parent/split provenance mismatch')
    if provenance['seed'] != source['source_seed'] or index[1] != semantic_fingerprint(record):
        raise ValueError('accepted source seed/semantic protected index mismatch')
    if record.get('verification', {}).get('code_hashes') != source['source_code_hashes']:
        raise ValueError('accepted source verification code hashes disagree with batch manifest')


def _records(paths, split, max_records):
    count = 0
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if count >= max_records:
                    return
                if len(line.encode('utf-8')) > MAX_SHARD_BYTES:
                    raise ValueError('source line exceeds bounded shard size')
                record = json.loads(line)
                if record.get('split') != split or record.get('quality_gate', {}).get('status') != 'accepted':
                    raise ValueError('selected source contains unaccepted or mismatched-split record')
                count += 1
                yield path, record


def export_rendered_dataset(source_root, split, view, max_records, variant_budget,
                            output_dir=None, source_paths=None):
    """Publish deterministic immutable shards plus a hash/version manifest.

    max_records caps accepted source records examined, including records that
    lack computation/distillation targets. source_paths can select a new batch.
    Source paths can only reference accepted train/dev/ood shards. Exports stay
    beneath data/rendered, so this function cannot overwrite canonical gold.
    """
    _bounds(view, max_records, variant_budget)
    if split not in ('train', 'dev', 'ood'):
        raise ValueError('sealed or unknown split cannot be exported')
    root = Path(source_root).resolve()
    base = (root / 'data' / 'rendered').resolve()
    output = Path(output_dir) if output_dir is not None else base
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output != base and base not in output.parents:
        raise ValueError('output_dir must be beneath source_root/data/rendered')
    if base != root / 'data' / 'rendered':
        raise ValueError('rendered destination cannot be a symlink outside programme root')
    _storage_guard(root)
    paths = _source_paths(root, split, source_paths)
    selected, sources = [], {}
    rendering_versions = set()
    seen_ids = set()
    for path, record in _records(paths, split, max_records):
        if record['id'] in seen_ids:
            raise ValueError('duplicate selected canonical record id')
        seen_ids.add(record['id'])
        relative = str(path.relative_to(root))
        if relative not in sources:
            sources[relative] = _batch_source(root, path, split)
        _source_record_provenance(record, sources[relative])
        rendering_versions.add(_rendering_identity(record)[0])
        sources[relative]['selected_records'] += 1
        selected.append({'record_id': record['id'], 'record_sha256': digest(record),
                         'parent_world_id': record['provenance']['parent_world_id'],
                         'provenance': deepcopy(record['provenance']),
                         'split': split, 'source_shard': relative})
    if not selected:
        raise ValueError('no selected accepted records')
    code = source_hashes()
    versions = sorted(rendering_versions)
    specification = {'export_version': EXPORT_VERSION,
                     'rendering_version': versions[0] if len(versions)==1 else 'mixed',
                     'rendering_versions': versions, 'rendering_policy_set_version': RENDERING_POLICY_SET_VERSION,
                     'split': split, 'view': view, 'max_records': max_records, 'variant_budget': variant_budget,
                     'selected_records': selected, 'sources': list(sources.values()), 'code_sha256': digest(code)}
    dataset_id = 'gcdl_rendered_' + digest(specification)[:32]
    destination = output / dataset_id
    shards, buffer, row_count, eligible_count = [], bytearray(), 0, 0
    buffered_rows = 0

    def flush():
        nonlocal buffer, buffered_rows
        if not buffer:
            return
        path = destination / ('shard-%05d.jsonl' % len(shards))
        raw = bytes(buffer)
        _publish(root, path, raw)
        shards.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(raw).hexdigest(),
                       'bytes': len(raw), 'rows': buffered_rows})
        buffer, buffered_rows = bytearray(), 0

    examined = 0
    for index, (path, record) in enumerate(_records(paths, split, max_records)):
        if index >= len(selected) or digest(record) != selected[index]['record_sha256']:
            raise ValueError('accepted source changed during rendered export')
        examined += 1
        produced = False
        for item in render_training_views(record, view, variant_budget):
            raw = (canonical(item) + '\n').encode('utf-8')
            if len(raw) > MAX_SHARD_BYTES:
                raise ValueError('single rendered example exceeds bounded shard size')
            if len(buffer) + len(raw) > MAX_SHARD_BYTES:
                flush()
            # Reserve the unpublished buffer too; fail before growing it past
            # the shared programme/free-space limits.
            _storage_guard(root, len(buffer) + len(raw))
            buffer.extend(raw)
            buffered_rows += 1
            row_count += 1
            produced = True
        eligible_count += int(produced)
    if examined != len(selected):
        raise ValueError('accepted source record count changed during rendered export')
    flush()
    for source in sources.values():
        if file_hash(root / source['path']) != source['sha256']:
            raise ValueError('accepted source hash changed during rendered export')
        if file_hash(root / source['source_batch_manifest']) != source['source_batch_manifest_sha256']:
            raise ValueError('accepted source batch manifest changed during rendered export')
    if source_hashes() != code:
        raise ValueError('executable code changed during rendered export')
    manifest = {**specification, 'dataset_id': dataset_id, 'code_hashes': code,
                'versions': {'primary': PRIMARY_VERSION, 'independent': INDEPENDENT_VERSION},
                'source_records_examined': len(selected), 'eligible_source_records': eligible_count,
                'rendered_rows': row_count, 'shards': shards,
                'model_fields': ['input_text'], 'target_fields': ['target'],
                'nonmodel_fields': ['record_id', 'view_id', 'view', 'split', 'provenance', 'surface', 'verification'],
                'resource_limits': {'max_shard_bytes': MAX_SHARD_BYTES, 'max_programme_bytes': MAX_PROGRAMME_BYTES,
                                    'min_free_bytes': MIN_FREE_BYTES},
                'verification': 'Exact parsed input and original candidate-subset outcomes checked by both executable oracles; targets independently rechecked.',
                'training_launched': False}
    manifest_path = destination / 'manifest.json'
    manifest['manifest_path'] = str(manifest_path.relative_to(root))
    _publish(root, manifest_path, (canonical(manifest) + '\n').encode('utf-8'))
    return manifest
