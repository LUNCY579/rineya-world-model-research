#!/usr/bin/env python3
"""Root-owned administrative DATA projection; no training targets or admissions."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import collections, datetime, gzip, hashlib, importlib.util, json, os, re

W = Path('/Users/chengli/Downloads/Rineya_Autonomous_Knowledge_Learning_Bundle_v1.0')
RAW = W/'output/revision54_research/cc0_source_byte_only_science_collection/v1.0'
OUT = W/'output/revision55_research/root_source_claim_knowledge_projection/v1.0'
spec = importlib.util.spec_from_file_location('root_owned_backup_helpers', W/'backup_control/OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.6.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def enc(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(x).hexdigest()

def pointer(x, p):
    for part in p.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x

def validate(v, s, path='$'):
    allowed = {'$schema', 'title', 'type', 'additionalProperties', 'required', 'properties', 'const', 'enum', 'minLength', 'items', 'uniqueItems', 'format', 'pattern'}
    assert set(s) <= allowed, ('unsupported schema keyword', path, set(s)-allowed)
    if 'const' in s:
        assert type(v) is type(s['const']) and v == s['const'], ('const', path)
    if 'enum' in s:
        assert any(type(v) is type(a) and v == a for a in s['enum']), ('enum', path)
    if 'type' in s:
        types = s['type'] if isinstance(s['type'], list) else [s['type']]
        checks = {'object': type(v) is dict, 'array': type(v) is list, 'string': type(v) is str, 'null': v is None, 'boolean': type(v) is bool}
        assert all(t in checks for t in types), ('unsupported schema type', path, types)
        assert any(checks[t] for t in types), ('type', path)
    if isinstance(v, dict):
        assert set(s.get('required', [])) <= set(v), ('required', path)
        props = s.get('properties', {})
        if s.get('additionalProperties') is False:
            assert set(v) <= set(props), ('additionalProperties', path)
        for k, a in v.items():
            if k in props:
                validate(a, props[k], path+'/'+k)
    if isinstance(v, list):
        if s.get('uniqueItems'):
            assert len({enc(a) for a in v}) == len(v), ('uniqueItems', path)
        for i, a in enumerate(v):
            if 'items' in s:
                validate(a, s['items'], path+'/'+str(i))
    if isinstance(v, str):
        assert len(v) >= s.get('minLength', 0), ('minLength', path)
        if 'pattern' in s:
            assert re.search(s['pattern'], v), ('pattern', path)
        if 'format' in s:
            assert s['format'] == 'date-time', ('unsupported format', path)
            assert re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})', v), ('date-time lexical', path)
            assert datetime.datetime.fromisoformat(v.replace('Z', '+00:00')).tzinfo is not None, ('date-time zone', path)

def safe_snak(snak):
    dtype = snak.get('datatype')
    if dtype in ('quantity', 'wikibase-item', 'time'):
        return True
    return dtype == 'string' and snak.get('property') == 'P1545' and snak.get('snaktype') == 'value' and re.fullmatch(r'[0-9]+', snak.get('datavalue', {}).get('value', '')) is not None

assert not OUT.exists(), 'fresh projection output required'
inputs = [W/'schemas/knowledge_record.schema.json', W/'schemas/task_record.schema.json', W/'schemas/README.md', RAW/'administration/item_science_property_scope.DATA.json']
scope_path = inputs[-1]
scope = json.loads(scope_path.read_bytes())
for entity in scope['entities']:
    inputs.append(RAW/entity['raw_source_relative_path'])
data_before = {str(p): helpers.native(p) for p in inputs}
_, sources_before, _, _ = helpers.tree_inventory()
strict_before = {str(p): helpers.native(p) for p in helpers.STRICT}
knowledge_schema = json.loads(inputs[0].read_bytes())
task_schema = json.loads(inputs[1].read_bytes())
assert knowledge_schema['$schema'] == task_schema['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
assert helpers.sha(inputs[0]) == 'fb7b66bfe577d764a0ef3ac2a5cc94d94d5e21adbf3ed56cf4bdd2b14f4b46c4'
assert helpers.sha(inputs[1]) == '9f772dfe10f5f7a693b057ce0a586c10a45f83f1f40bc9a7aa5c31da4a1048a6'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows, bindings, withheld = [], [], []
counts = collections.Counter()
for entity in scope['entities']:
    p = RAW/entity['raw_source_relative_path']
    body = p.read_bytes()
    assert len(body) == entity['raw_source_size_bytes'] and digest(body) == entity['raw_source_sha256']
    raw = json.loads(body)
    qid, rev = entity['entity_id'], entity['revision_id']
    item = raw['entities'][qid]
    assert item['id'] == qid and item['lastrevid'] == rev and item['ns'] == 0 and item['type'] == 'item'
    for prop in entity['candidate_science_property_scope']:
        for candidate in prop['statements']:
            ptr = candidate['raw_json_pointer']
            st = pointer(raw, ptr)
            main = st['mainsnak']
            assert st['id'] == candidate['statement_id'] and st['rank'] == candidate['rank'] and main['property'] == prop['property_id']
            if main.get('datatype') not in ('quantity', 'wikibase-item'):
                continue
            counts['narrow_statements'] += 1
            counts['main_'+main['datatype']] += 1
            assert main.get('snaktype') in ('value', 'novalue', 'somevalue')
            counts['snaktype_'+main['snaktype']] += 1
            if main['snaktype'] != 'value':
                withheld.append({'entity_id': qid, 'revision_id': rev, 'statement_id': st['id'], 'raw_json_pointer': ptr, 'main_datatype': main['datatype'], 'main_snaktype': main['snaktype'], 'statement_semantic_json_sha256': digest(enc(st)), 'disposition': 'RETAIN_RAW_ONLY_NO_NUMERIC_OR_RELATION_FACT_CREATED'})
                continue
            if main['datatype'] == 'quantity':
                value = main['datavalue']['value']
                assert type(value['amount']) is str and type(value['unit']) is str
                for k in ('lowerBound', 'upperBound'):
                    assert k not in value or value[k] is None or type(value[k]) is str
            qualifiers, excluded = {}, []
            for qp, snaks in st.get('qualifiers', {}).items():
                kept = []
                for i, snak in enumerate(snaks):
                    sp = ptr+'/qualifiers/'+qp+'/'+str(i)
                    if safe_snak(snak):
                        kept.append({'raw_json_pointer': sp, 'snak': snak})
                        counts['retained_qualifier_snaks'] += 1
                        if snak.get('datatype') == 'string':
                            counts['digit_only_P1545_ordinals'] += 1
                    else:
                        excluded.append({'raw_json_pointer': sp, 'datatype': snak.get('datatype'), 'snaktype': snak.get('snaktype'), 'semantic_json_sha256': digest(enc(snak)), 'disposition': 'RAW_ONLY_LITERAL_OR_OTHER_FIELD_NOT_PROJECTED'})
                        counts['withheld_qualifier_snaks'] += 1
                if kept:
                    qualifiers[qp] = kept
            references = [{'raw_json_pointer': ptr+'/references/'+str(i), 'reference_source_hash_field': ref.get('hash'), 'semantic_json_sha256': digest(enc(ref))} for i, ref in enumerate(st.get('references', []))]
            counts['reference_blocks_pointer_hash_only'] += len(references)
            projection = {'entity_id': qid, 'revision_id': rev, 'statement_id': st['id'], 'property_id': prop['property_id'], 'rank': st['rank'], 'mainsnak': main, 'qualifiers': qualifiers, 'source_qualifiers_order': st.get('qualifiers-order', []), 'withheld_context': excluded, 'reference_blocks': references}
            canonical = enc(projection).decode()
            kid = 'WD_SOURCE_CLAIM_'+qid+'_R'+str(rev)+'_'+st['id'].split('$', 1)[1]
            source_id = 'wikidata:'+qid+':revision:'+str(rev)+':'+ptr
            row = {'schema_version': '1.0', 'id': kid, 'kind': 'fact', 'status': 'proposed', 'canonical_form': canonical, 'scope': 'Exact pinned Wikidata structured source-claim projection. This is not adjudicated current scientific truth. Reference text is withheld; source rank, typed conditions and literal numeric ordinals are retained. Rights, scientific intake, human review and training admission remain pending.', 'source_ids': [source_id], 'dependency_ids': [], 'verification_ids': [], 'created_at': now, 'payload_sha256': digest(canonical.encode())}
            validate(row, knowledge_schema)
            rows.append(row)
            bindings.append({'knowledge_proposal_id': kid, 'source_id': source_id, 'entity_id': qid, 'revision_id': rev, 'raw_source_path': str(p.relative_to(W)), 'raw_source_bytes': len(body), 'raw_source_sha256': digest(body), 'raw_json_pointer': ptr, 'statement_id': st['id'], 'statement_semantic_json_sha256': digest(enc(st)), 'canonical_projection_sha256': row['payload_sha256'], 'rank': st['rank'], 'main_datatype': main['datatype'], 'main_snaktype': main['snaktype'], 'reference_blocks': references, 'withheld_context': excluded, 'rights': 'HOLD_FINAL_UNRESTRICTED_CLEARANCE_UNPROVEN', 'scientific_intake': 'PENDING_SOURCE_SPECIFIC_REVIEW_BINDING', 'training_admission': False, 'role': None, 'independent_parent_id': None, 'model_authored_target': None})
assert counts['narrow_statements'] == 336 and counts['main_quantity'] == 186 and counts['main_wikibase-item'] == 150
assert len(rows) == len(bindings) == 335 and len(withheld) == 1 and counts['snaktype_novalue'] == 1 and counts['digit_only_P1545_ordinals'] == 33
assert len({x['id'] for x in rows}) == len(rows)
schema_rejection_checks = []
for label, change in [('training_target_extra_field', {'target': 'not allowed in knowledge schema'}), ('admitted_status', {'status': 'TRAIN'}), ('wrong_schema_version', {'schema_version': '2.0'}), ('duplicated_source_id', {'source_ids': rows[0]['source_ids']*2})]:
    bad = dict(rows[0]); bad.update(change)
    try:
        validate(bad, knowledge_schema)
    except AssertionError as ex:
        schema_rejection_checks.append({'case': label, 'rejected': True, 'validator_reason': str(ex)})
    else:
        raise AssertionError(('failed to reject contract violation', label))
OUT.mkdir(parents=True)
outputs = {}
for name, payload in [('proposed_source_claim_knowledge_records.DATA.jsonl', b'\n'.join(enc(x) for x in rows)+b'\n'), ('source_claim_bindings.DATA.jsonl', b'\n'.join(enc(x) for x in bindings)+b'\n'), ('nonvalue_raw_only_exclusions.DATA.json', enc(withheld)+b'\n')]:
    helpers.atomic(OUT/name, payload)
    outputs[name] = {'bytes': len(payload), 'sha256': digest(payload)}
disk_rows = [json.loads(x) for x in (OUT/'proposed_source_claim_knowledge_records.DATA.jsonl').read_bytes().splitlines()]
disk_bindings = [json.loads(x) for x in (OUT/'source_claim_bindings.DATA.jsonl').read_bytes().splitlines()]
assert disk_rows == rows and disk_bindings == bindings
for row, binding in zip(disk_rows, disk_bindings):
    validate(row, knowledge_schema)
    assert row['id'] == binding['knowledge_proposal_id'] and row['payload_sha256'] == digest(row['canonical_form'].encode())
    assert json.loads(row['canonical_form'])['statement_id'] == binding['statement_id']
data_after = {str(p): helpers.native(p) for p in inputs}
ordinary_diff = [p for p in data_before if helpers.ordinary(data_before[p]) != helpers.ordinary(data_after[p])]
_, sources_after, _, _ = helpers.tree_inventory()
strict_after = {str(p): helpers.native(p) for p in helpers.STRICT}
source_diff = [p for p in set(sources_before)|set(sources_after) if sources_before.get(p) != sources_after.get(p)]
assert not ordinary_diff and not source_diff and strict_before == strict_after
receipt = {'schema': 'r55.source-claim-projection-native-receipt.v1.0', 'ordinary_DATA_excluded_fields': ['st_atime', 'st_atime_ns'], 'input_native19_before': data_before, 'input_native19_after': data_after, 'ordinary_DATA_differences': ordinary_diff, 'source_native19_before': sources_before, 'source_native19_after': sources_after, 'source_ALL19_differences': source_diff, 'strict7_native19_before': strict_before, 'strict7_native19_after': strict_after, 'schema_inputs_restored_new_native_epoch': True}
payload = gzip.compress(enc(receipt)+b'\n', mtime=0)
helpers.atomic(OUT/'native_integrity_receipt.DATA.json.gz', payload)
outputs['native_integrity_receipt.DATA.json.gz'] = {'bytes': len(payload), 'sha256': digest(payload)}
report = {'schema': 'r55.source-claim-knowledge-schema-compatibility.v1.0', 'utc': now, 'knowledge_schema': {'path': str(inputs[0].relative_to(W)), 'bytes': inputs[0].stat().st_size, 'sha256': helpers.sha(inputs[0]), 'draft': '2020-12', 'used_unchanged': True}, 'task_schema': {'path': str(inputs[1].relative_to(W)), 'sha256': helpers.sha(inputs[1]), 'used_unchanged': True, 'new_task_records': 0}, 'validator': {'owner': 'Root', 'scope': 'dependency-free validation of all keywords present in this exact knowledge schema; unknown schema keywords rejected', 'general_2020_12_validator_claim': False}, 'schema_rejection_checks': schema_rejection_checks, 'counts': dict(counts), 'knowledge_source_claim_proposals': len(rows), 'nonvalue_statements_kept_raw_only': len(withheld), 'structural_validation_all_written_records': True, 'complete_written_record_roundtrip_and_binding_checks': True, 'source_entity_count': len(scope['entities']), 'new_parent_proposals': 0, 'new_student_views': 0, 'new_training_targets': 0, 'new_training_admissions': 0, 'source_programs_stat_only_ALL19_count': len(sources_before), 'source_programs_ALL19_differences': 0, 'strict7_ALL19_unchanged': True, 'ordinary_DATA_native19_differences': 0, 'rights': 'HOLD_FINAL_UNRESTRICTED_CLEARANCE_UNPROVEN', 'scientific_review': 'PENDING_BINDING_TO_CLOSED_PRIMARY_REVIEWS', 'human_review': 'PENDING', 'source_claim_projection_is_not_scientific_truth_or_training_admission': True, 'reference_text_quotes_titles_pages_sections_not_copied': True, 'numbers_units_bounds_not_float_coerced_or_normalized': True, 'source_entity_and_claim_counts_are_not_independent_parents': True, 'preserved_root_telemetry_failures': [{'chunk': '1ca69c', 'failure': 'manifest candidate output truncated; exact selected path plan later persisted'}, {'chunk': '3b015f', 'failure': '1,317,546-byte scope output truncated; no complete prose read claimed; complete machine parse and bounded key projection 5dcc3b subsequently used'}], 'outputs': outputs}
payload = enc(report)+b'\n'
helpers.atomic(OUT/'schema_compatibility_report.DATA.json', payload)
outputs['schema_compatibility_report.DATA.json'] = {'bytes': len(payload), 'sha256': digest(payload)}
manifest = {'schema': 'r55.closed-source-claim-projection-files.v1.0', 'writer': 'Root', 'writer_closed': True, 'source_program': {'owner': 'Root', 'path': str(Path(__file__).relative_to(W)), 'excluded_from_DATA_backup': True}, 'files': [{'path': name, **ref} for name, ref in outputs.items()], 'new_training_admissions': 0}
helpers.atomic(OUT/'CLOSED_DATA_FILES.DATA.json', enc(manifest)+b'\n')
print(json.dumps({'output': str(OUT), 'proposed_source_claim_records': len(rows), 'narrow_statements': counts['narrow_statements'], 'raw_only_nonvalue': len(withheld), 'digit_only_ordinals': counts['digit_only_P1545_ordinals'], 'typed_qualifiers': counts['retained_qualifier_snaks'], 'withheld_qualifiers': counts['withheld_qualifier_snaks'], 'reference_blocks_hash_pointer_only': counts['reference_blocks_pointer_hash_only'], 'ordinary_DATA_differences': 0, 'Source_ALL19_differences': 0, 'strict7_unchanged': True, 'new_targets_parents_admissions': 0}))
