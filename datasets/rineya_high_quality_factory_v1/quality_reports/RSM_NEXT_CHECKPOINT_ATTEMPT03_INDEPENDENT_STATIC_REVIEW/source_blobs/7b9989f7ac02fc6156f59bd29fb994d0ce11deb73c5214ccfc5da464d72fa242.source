"""Own stdlib-only explicit-ledger byte readback. No target imports/execution.

This reads declared metadata and hashes exactly named sources/raw archives;
it does not discover a dataset/tree inventory or parse production semantic data.
Freeze this source and its raw archive before the first invocation.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).absolute().parent
OLD = 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT02/'
NEW = 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT03/'
EXPECTED = {
    OLD+'SOURCE_PREPARATION_FREEZE.json':'4d7bebfdb6e1d353504db18c6294735a1168a809fc88775bacc8c20984d02b00',
    NEW+'SOURCE_PREPARATION_FREEZE.json':'4f4781a897941e28a827da09cad81a23e9588296d2861a093a46fd53ddda917b',
    NEW+'SOURCE_READY.json':'76d7b358d716bf7b8ad846acdabb4b6329739b5b7d7289d05914cb59bb8ffda9',
    NEW+'SOURCE_CHANGES_MINIMAL.diff':'c39b0970542386e486882d935afce440c0710367db6ca46f42cc252ca557b263',
    NEW+'common.py':'1da770c74bc70a0a45672965fec36b384dd7778cc997b391504fba37392a32a4',
    NEW+'tests.py':'ccfe50b78dc5494f829505360b3a4252ef9d7b2dca8b7bb894f3006c26184825',
}
OBSERVATIONS = {}

def literal(name, absolute=False):
    if type(name) is not str or not name or any(c in name for c in ('\\','\0','\n','\r')):
        raise ValueError('typed literal declared path required')
    p = Path(name)
    if p.is_absolute() != absolute or p.as_posix() != name or any(c in ('.','..') for c in p.parts):
        raise ValueError('declared path alias rejected')
    p = p if absolute else ROOT/p
    if p.is_symlink() or not p.is_file():
        raise ValueError('declared file/symlink differs: '+name)
    current = p
    while current != current.parent:
        if current.is_symlink():raise ValueError('parent symlink rejected: '+name)
        current = current.parent
    return p

def hashed(name, absolute=False):
    p = literal(name, absolute)
    before = tuple(getattr(p.stat(), k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    h = sha256()
    with p.open('rb') as stream:
        while part := stream.read(65536):h.update(part)
    after = tuple(getattr(p.stat(), k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    if before != after:raise ValueError('declared bytes evolved while reading')
    result = {'path':name,'size':before[2],'sha256':h.hexdigest()}
    OBSERVATIONS[name] = result
    return result

def bound(row, absolute=False):
    if type(row) is not dict or set(row) != {'path','size','sha256'}:
        raise ValueError('closed binding required')
    if type(row['size']) is not int or row['size'] < 0 or type(row['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None:
        raise ValueError('typed size/hash required')
    if hashed(row['path'], absolute) != row:raise ValueError('declared bytes mismatch: '+row['path'])
    return row

def metadata(path):
    record = hashed(path)
    if path in EXPECTED and record['sha256'] != EXPECTED[path]:raise ValueError('external fixed hash mismatch')
    return json.loads(literal(path).read_bytes())

def rows(freeze, expected_count):
    listed = freeze['frozen_inputs']
    if type(listed) is not list or len(listed) != expected_count:raise ValueError('declared preparation count mismatch')
    if len({r['path'] for r in listed}) != expected_count:raise ValueError('duplicate declared preparation path')
    for row in listed:bound(row)
    return {r['path']:r for r in listed}

def main():
    # This executes own byte-reading code only. Reviewed modules remain inert.
    for path, expected in EXPECTED.items():
        if hashed(path)['sha256'] != expected:raise ValueError('fixed source/metadata hash mismatch')
    old = metadata(OLD+'SOURCE_PREPARATION_FREEZE.json')
    new = metadata(NEW+'SOURCE_PREPARATION_FREEZE.json')
    old_rows = rows(old,277);new_rows = rows(new,276)
    runtime = metadata(NEW+'RUNTIME_SOURCE_FREEZE.json')
    pairs = runtime['runtime_bindings']
    if len(pairs) != 206 or len({r['current']['path'] for r in pairs}) != 206:
        raise ValueError('exact declared runtime206 membership differs')
    for row in pairs:
        bound(row['current'],True);bound(row['raw_archive'])
        if {k:row['current'][k] for k in ('size','sha256')} != {k:row['raw_archive'][k] for k in ('size','sha256')}:
            raise ValueError('runtime current/raw pair differs')
    sources = runtime['source_code_bindings']
    if len(sources) != 18 or sorted(r['path'] for r in sources) != new['operative_source_paths']:
        raise ValueError('exact operative18 source membership differs')
    config = metadata(NEW+'RUNTIME_COLLECTOR_CONFIG.json')
    if sources != config['source_code_bindings']:raise ValueError('collector/runtime source bindings cross-wired')
    for row in sources:
        bound(row)
        if new_rows.get(row['path']) != row:raise ValueError('operative source absent from frozen preparation')
    raw_sources = new['raw_source_archives']
    if len(raw_sources) != 12 or len({r['original_path'] for r in raw_sources}) != 12:
        raise ValueError('exact own12 raw mapping differs')
    for row in raw_sources:
        original = new_rows[row['original_path']];archive = new_rows[row['archive_path']]
        if original['sha256'] != row['sha256'] or archive['sha256'] != row['sha256'] or original['size'] != archive['size']:
            raise ValueError('own source/raw archive differs')
        bound(original);bound(archive)
    own_names = sorted(Path(r['original_path']).name for r in raw_sources)
    changes = []
    for name in own_names:
        prior = hashed(OLD+name);current = hashed(NEW+name)
        if {k:prior[k] for k in ('sha256','size')} != {k:current[k] for k in ('sha256','size')}:changes.append(name)
    if changes != ['common.py','tests.py']:raise ValueError('unexpected Python source change')
    old_common = literal(OLD+'common.py').read_text();new_common = literal(NEW+'common.py').read_text()
    replacement = "    if phase=='test':\n        if value['fields']!={}:raise ValueError('Completed test phase accepts no fields')\n    elif phase=='inventory':"
    if new_common.count(replacement)!=1 or new_common.replace(replacement,"    if phase=='inventory':") != old_common:
        raise ValueError('common delta exceeds stated narrow fix')
    old_tests = literal(OLD+'tests.py').read_text();new_tests = literal(NEW+'tests.py').read_text()
    start = new_tests.index('    def test_completed_test_empty_fields_and_receipt_provenance(self):')
    end = new_tests.index('    def test_completed_proof_config_freeze_output_ledger_and_dependency_substitutions(self):',start)
    if new_tests[:start]+new_tests[end:] != old_tests:raise ValueError('test delta exceeds the single new regression')
    methods = re.findall(r'^    def (test_[A-Za-z0-9_]+)\(self\):',new_tests,re.M)
    if len(methods) != 72 or len(set(methods)) != 72:raise ValueError('declared72 unique test source methods differ')
    own = hashed(str(Path(__file__).absolute().relative_to(ROOT)))
    output = {'status':'PASS_EXPLICIT_METADATA_BYTE_READBACK_ONLY','author_execution':'/root/formal_oracle',
        'actual_runner_execution':'/root/formal_oracle','at_utc':datetime.now(timezone.utc).isoformat(),
        'own_reader':own,'old_preparation_bindings_unchanged':len(old_rows),'new_preparation_bindings_valid':len(new_rows),
        'runtime_current_raw_pairs':len(pairs),'operative_sources':len(sources),'own_source_raw_pairs':len(raw_sources),
        'changed_python_sources':changes,'unchanged_python_sources':10,'unique_test_methods_source_only':len(methods),
        'observations':dict(sorted(OBSERVATIONS.items())),
        'reviewed_imports_compiles_tests_phases_inventory_git_network':0,'production_semantic_rerun':False,'data_approval':False}
    p = OUT/'STATIC_BYTE_READBACK.json'
    with p.open('x') as stream:json.dump(output,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:output[k] for k in ('status','old_preparation_bindings_unchanged','new_preparation_bindings_valid','runtime_current_raw_pairs','operative_sources','own_source_raw_pairs','changed_python_sources','unique_test_methods_source_only')},sort_keys=True))

if __name__ == '__main__':main()
