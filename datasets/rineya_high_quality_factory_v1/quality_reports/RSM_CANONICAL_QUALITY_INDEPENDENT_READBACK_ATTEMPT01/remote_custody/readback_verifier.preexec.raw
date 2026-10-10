"""Own read-only technical byte readback; never imports or executes reviewed code.

Only declared metadata JSON is parsed. All listed source/runtime/candidate and raw
archives are read opaquely for SHA256, size and byte equality. Outputs are owned
technical evidence; no quality votes, target receipts or data approval are emitted.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset')
BASE = ROOT / 'reviews/adversarial/RSM-option-resumption-pilot-v1-canonical-attempt03'
OWN = Path(__file__).absolute().parent
ANCHORS = {
    'receipt_emitter_r3.py': '423c581183004824cd5072ce71d44cdfe22c4ee451ccd5573e9f7141ca3f6030',
    'quality_author_r2.py': '06d7ae3529370e65a3ad1133e5cc5f6b17e86c928a77722acba3d35615dddd39',
    'freeze_quality_author_r2.py': 'ba56320a98f659b4490cad358e1005802bc20f07bbc934faf0de0c799e6c534a',
    'finalize_quality_custody_r1.py': '931689fe224b733704bd3c603a9f1ee097c9aeceab8d0292092625c97edf9ed3',
    'SOURCE_FREEZE.json': '0f8bf67ebd632c89a85efbdc151bbe5db08eacb25ca8358d9cdc7e30a413b25d',
    'RUN_CONFIG.json': '61a9bffd524f8246fa4ecba8dfabdb60eadb9b3c0a3aa57029616f4e17ea5e26',
    'FACTUAL_FIRST.jsonl': '42d8f13dc96e3f8c65fa9d92142e8851646a15bf599216da37e72672ca7ab98c',
    'FACTUAL_FIRST_FREEZE.json': 'eaa3791551ff779ee8cef0b6d44a1202cf17ca5ea2a4d945f234358fff100a19',
    'COMPARISON_EXECUTION.json': '3e022c86806a53eba169bd5d2ff15c7038d0aea5023e76ce62a5ca702d4f2db7',
    'QUALITY_AUTHOR_CONFIG.json': '1736cc9d6ce0aa93913fab8c5c3b3d478f4e0c7aa2b4000c387e6ca6f034b64b',
    'QUALITY_AUTHOR_PREEXEC_R2.json': '078a5cfdec45cb5599e6d3e4edeb30ad0bc13d9f0742bf59357da91a4f64677b',
    'QUALITY_AUTHOR_EXECUTION.json': 'f7c705f22db7d87dbcd42f310d500d5ee65027a7afc10746839c2d3f8246171e',
    'QUALITY_MAP.json': 'a9e8e42965ca3c22117898a7ab5a7ceeb1e2fbeb6cb8207269a2dd23cad8b704',
    'QUALITY_MAP_FREEZE.json': '094c804d3801c0e3ad558cca0211732c7052ad2877d55ea9abebbc5f0abd23aa',
    'FINAL_QUALITY_CUSTODY.json': 'b78d6967cd3f4930c226ee9c29194c5fab9228651c89a6da4147815d79142cc0',
    'FINAL_CUSTODY_TOOL_COMPLETION.json': 'ae1fd1ec16aa9dfca58deb4a324d64a1a778aa18b2df0169b65206a8e0b5387a',
}
errors = []
stats = {}
digests = {}

def check(condition, message):
    if not condition:
        errors.append(message)

def statkey(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def filehash(path):
    path = Path(path)
    before = statkey(path)
    if str(path) in digests:
        check(stats[str(path)] == before, 'changed stat since read: ' + str(path))
        return digests[str(path)]
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(65536):
            h.update(chunk)
            size += len(chunk)
    check(statkey(path) == before, 'changed during hash: ' + str(path))
    stats[str(path)] = before
    value = {'sha256': h.hexdigest(), 'size_bytes': size}
    digests[str(path)] = value
    return value

def strict_json(path):
    def pairs(values):
        result = {}
        for k, v in values:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=nonfinite)

def current_path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()

def verify_rows(label, rows):
    bound = {}
    checked = 0
    for row in rows:
        p = current_path(row['path'])
        archive = ROOT / row['archive']
        check(str(p) not in bound, label + ': duplicate resolved current path ' + str(p))
        bound[str(p)] = row
        try:
            a, b = filehash(p), filehash(archive)
            expected = {k: row[k] for k in ('sha256', 'size_bytes')}
            check(a == expected and b == expected, label + ': size/hash drift ' + str(p))
            same = True
            with p.open('rb') as x, archive.open('rb') as y:
                while True:
                    left, right = x.read(65536), y.read(65536)
                    if left != right:
                        same = False
                        break
                    if not left:
                        break
            check(same, label + ': current/raw byte inequality ' + str(p))
            checked += 1
        except OSError as e:
            errors.append(label + ': missing/unreadable pair ' + str(p) + ': ' + str(e))
    return bound, {'listed_rows': len(rows), 'unique_resolved_current_paths': len(bound), 'pairs_read': checked}

def runtime_check(label, modules, bound, sf, main):
    seen = set()
    concrete = 0
    for r in modules:
        name = r['module']
        check(name not in seen, label + ': duplicate module ' + name)
        seen.add(name)
        identity = {k: r[k] for k in ('origin', 'source_path', 'cache_path', 'namespace_paths')}
        if name == '__main__':
            check(identity['source_path'] == str(main) and identity['origin'] is None,
                  label + ': main identity mismatch')
        else:
            check(sf['runtime_module_identities'].get(name) == identity,
                  label + ': module identity outside SOURCE_FREEZE ' + name)
        for field in ('source_path', 'cache_path'):
            value = identity[field]
            if value and Path(value).is_file():
                concrete += 1
                check(str(Path(value).resolve()) in bound,
                      label + ': unbound actual-loaded concrete path ' + name + ':' + value)
            elif value and field == 'source_path':
                check(False, label + ': missing source ' + name + ':' + value)
    check(sf['resolved_python'] in bound, label + ': interpreter byte binding omitted')
    return {'reported_modules': len(modules), 'reported_concrete_path_observations': concrete}

def main():
    pre = strict_json(OWN / 'PREEXEC.json')
    source = Path(__file__).read_bytes()
    check(hashlib.sha256(source).hexdigest() == pre['verifier_sha256'], 'own source preexec hash')
    check((OWN / 'readback_verifier.preexec.raw').read_bytes() == source, 'own source preexec raw')
    if errors:
        raise ValueError(errors)
    anchored = {}
    for name, digest in ANCHORS.items():
        actual = filehash(BASE / name)
        anchored[name] = actual
        check(actual['sha256'] == digest, 'external readback anchor drift ' + name)
    documents = {name: strict_json(BASE / name) for name in ANCHORS if name.endswith('.json')}
    sf = documents['SOURCE_FREEZE.json']
    groups = [
        ('SOURCE_FREEZE', sf['operative_bindings']),
        ('QUALITY_AUTHOR_PREEXEC_R2', documents['QUALITY_AUTHOR_PREEXEC_R2.json']['files']),
        ('FACTUAL_FIRST_FREEZE', documents['FACTUAL_FIRST_FREEZE.json']['files']),
        ('QUALITY_MAP_FREEZE', documents['QUALITY_MAP_FREEZE.json']['files']),
        ('FINAL_QUALITY_CUSTODY', documents['FINAL_QUALITY_CUSTODY.json']['files']),
    ]
    sets, summaries = {}, {}
    for label, rows in groups:
        sets[label], summaries[label] = verify_rows(label, rows)
    runtime = {}
    qe = documents['QUALITY_AUTHOR_EXECUTION.json']
    final = documents['FINAL_QUALITY_CUSTODY.json']
    runtime['quality_preexec'] = runtime_check('quality_preexec', documents['QUALITY_AUTHOR_PREEXEC_R2.json']['actual_loaded_stdlib_identities'], sets['QUALITY_AUTHOR_PREEXEC_R2'], sf, BASE / 'freeze_quality_author_r2.py')
    for phase in ('before', 'after'):
        runtime['quality_author_' + phase] = runtime_check('quality_author_' + phase, qe['runtime_' + phase]['loaded_module_identities'], sets['QUALITY_AUTHOR_PREEXEC_R2'], sf, BASE / 'quality_author_r2.py')
        runtime['finalizer_' + phase] = runtime_check('finalizer_' + phase, final['runtime_' + phase], sets['FINAL_QUALITY_CUSTODY'], sf, BASE / 'finalize_quality_custody_r1.py')
    comparison = documents['COMPARISON_EXECUTION.json']
    for phase in ('closure_before_helpers', 'closure_after_helpers_before_targets', 'closure_after_comparison'):
        runtime['factual_' + phase] = runtime_check('factual_' + phase, comparison[phase]['loaded_runtime_modules'], sets['SOURCE_FREEZE'], sf, BASE / 'factual_comparison_r3.py')
    for r in documents['FACTUAL_FIRST_FREEZE.json']['files']:
        later = sets['QUALITY_MAP_FREEZE'].get(str(current_path(r['path'])))
        check(later is not None and all(later[k] == r[k] for k in ('path', 'sha256', 'size_bytes')), 'FIRST omitted/crosswired from quality freeze: ' + r['path'])
    for name in ('QUALITY_MAP.json', 'QUALITY_AUTHOR_EXECUTION.json', 'QUALITY_AUTHOR_PREEXEC_R2.json', 'receipt_emitter_r3.py'):
        check(str(BASE / name) in sets['QUALITY_MAP_FREEZE'], 'quality freeze missing operative evidence ' + name)
    qf = documents['QUALITY_MAP_FREEZE.json']
    check(qf['quality_map_path'] == qe['quality_map']['path'] and qe['quality_map']['sha256'] == ANCHORS['QUALITY_MAP.json'], 'quality map crosslink')
    check(final['quality_map_freeze_sha256'] == ANCHORS['QUALITY_MAP_FREEZE.json'], 'final map freeze crosslink')
    completion = documents['FINAL_CUSTODY_TOOL_COMPLETION.json']
    check(completion['final_custody_sha256'] == ANCHORS['FINAL_QUALITY_CUSTODY.json'], 'completed final hash crosslink')
    erratum = completion['descriptive_pointer_erratum']
    corrected = sets['FINAL_QUALITY_CUSTODY'].get(str(current_path(erratum['correct_actual_path'])))
    check(corrected is not None and corrected['sha256'] == erratum['correct_actual_sha256'], 'descriptive failure pointer erratum binding')
    excluded = [str(BASE / ('QUALITY_PREEXEC_R2_TOOL_' + suffix + '.txt')) for suffix in ('STDOUT', 'STDERR', 'EXIT')]
    check(all(p not in sets['QUALITY_AUTHOR_PREEXEC_R2'] for p in excluded), 'live r2 log was included in preexec')
    check(all(p in sets['FINAL_QUALITY_CUSTODY'] for p in excluded), 'completed r2 logs absent from final custody')
    for suffix in ('STDOUT', 'STDERR', 'EXIT'):
        check(str(BASE / ('QUALITY_AUTHOR_TOOL_' + suffix + '.txt')) in sets['FINAL_QUALITY_CUSTODY'], 'completed quality author log absent ' + suffix)
    factual = [json.loads(line) for line in (BASE / 'FACTUAL_FIRST.jsonl').read_text().splitlines()]
    quality = documents['QUALITY_MAP.json']
    facts = {r['sample_id']: r['record_hash'] for r in factual}
    qualities = {r['sample_id']: r['record_hash'] for r in quality['rows']}
    check(len(factual) == len(facts) == len(quality['rows']) == len(qualities) == 216 and facts == qualities, 'FIRST/quality exact216 ID/hash join')
    check(all(r.get('quality_assessment') == 'UNASSESSED_PENDING_ACTUAL_AGENT_INSPECTION' and 'scores' not in r and 'decision' not in r for r in factual), 'FIRST was quality assessed')
    check(qe['rows'] == final['rows'] == 216 and qe['raw_pair_count'] == summaries['QUALITY_AUTHOR_PREEXEC_R2']['listed_rows'], 'finished row/pair counts')
    check(final['current_raw_pairs_before_after'] == summaries['FINAL_QUALITY_CUSTODY']['listed_rows'], 'final pair count')
    for p, before in stats.items():
        check(statkey(Path(p)) == before, 'observed file changed by final readback: ' + p)
    own_preexec = filehash(OWN / 'PREEXEC.json')
    result = {
        'status': 'PASS_STATIC_BYTE_READBACK' if not errors else 'STATIC_BYTE_READBACK_GAPS',
        'same_model_disclosure': 'Same model; independent technical readback only; no semantic execution or data approval.',
        'verifier_source': {'path': str(Path(__file__)), 'sha256': pre['verifier_sha256']},
        'preexec': own_preexec, 'reviewed_byte_anchors': anchored,
        'freeze_pair_summaries': summaries, 'actual_loaded_runtime_membership': runtime,
        'unique_opaque_files_hashed': len(digests), 'errors': errors,
        'factual_quality_exact_id_hash_join': len(facts),
        'designated_receipt_exists_metadata_only': current_path(documents['RUN_CONFIG.json']['final_receipt_path']).exists(),
        'emission_execution_exists_metadata_only': (BASE / 'EMISSION_EXECUTION.json').exists(),
        'target_modules_imported_or_executed': 0, 'target_compiles_tests_receipts_votes_approvals': 0,
        'resolved_own_verifier_python': str(Path(sys.executable).resolve()),
        'own_verifier_flags': {k: getattr(sys.flags, k) for k in ('isolated', 'dont_write_bytecode', 'no_site', 'ignore_environment')},
        'limitation': 'Metadata-reported membership verified against frozen paths; this does not inspect a live target process or certify historical memory, namespace contents, arbitrary dynamic code or subjective quality judgments.',
    }
    raw = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    target = OWN / 'READBACK.json'
    with target.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'readback_path': str(target), 'sha256': hashlib.sha256(raw).hexdigest(), 'status': result['status'], 'errors': errors, 'pair_summaries': summaries, 'runtime_modules': {k: v['reported_modules'] for k, v in runtime.items()}}))

if __name__ == '__main__':
    main()
