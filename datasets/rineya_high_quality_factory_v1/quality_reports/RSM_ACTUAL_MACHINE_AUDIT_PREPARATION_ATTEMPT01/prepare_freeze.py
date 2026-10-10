"""ROOT benign runtime discovery and exact-byte preparation only.

No factory, generator, semantic oracle, reviewer or test helper imports.
No actual candidate target inspection or comparisons. Bootstrap/runtime package
discovery is disclosed; the resulting closure precedes the actual audit import.
"""
import argparse, ast, bz2, collections, copy, datetime, gzip, hashlib
import importlib.util, itertools, json, linecache, locale, lzma, math, os
from pathlib import Path
import re, resource, stat, subprocess, sys, tempfile, time, tokenize, traceback
import jsonschema
import attr, attrs, referencing, rpds

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    walk = Path(path.anchor)
    for part in path.parts[1:]:
        walk /= part
        assert not walk.is_symlink(), str(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode)
        raw = stream.read()
        after = os.fstat(stream.fileno())
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(raw), 'bytes': len(raw)}


def main():
    assert not any(name == 'factory' or name.startswith(('factory.', 'oracles.', 'generators.')) for name in sys.modules)
    # Generic schema work only discovers the benign validator import closure.
    schema = {'type': 'object', 'required': ['n'], 'properties': {'n': {'type': 'integer', 'minimum': 0}, 's': {'type': 'string', 'pattern': '^x'}}}
    jsonschema.Draft202012Validator.check_schema(schema)
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors({'n': 1, 's': 'x'}))
    full = 'quality_reports/RSM_REPAIRED_FULL_SUITE_EXECUTION_ATTEMPT03/EXECUTION.json'
    full_root = 'quality_reports/RSM_REPAIRED_FULL_SUITE_ROOT_COMPLETION_ATTEMPT03.json'
    assert digest(read(full_root)) == '71d9ce4f3cfefc84210da977d682caae892b7dfb938ac5c26f9d3f06e1c4235e'
    execution = json.loads(read(full))
    assert execution['status'] == 'PASS' and execution['mode'] == 'full' and execution['tests_run'] == 468
    runtime_path = 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/INSTALLED_RUNTIME_SOURCE_FREEZE.json'
    assert digest(read(runtime_path)) == '415cb144658406925f7d2939edcdc907a17f8dc1e9e62616ba6808202fb5a6ea'
    runtime = json.loads(read(runtime_path))
    primary = dict(runtime['bindings'])
    assert len(primary) == 507
    for path, row in primary.items():
        raw = read(path)
        assert len(raw) == row['bytes'] and digest(raw) == row['sha256'] and read(row['archive']) == raw
    final_path = 'generators/skills/option_resumption_pilot_v1_attempt01/executions/main-17/PRODUCER_FINAL_CUSTODY.json'
    assert digest(read(final_path)) == '23be24a2d6bbafd67bfdc74cf4f3ae83dc8770aeb64c0d4b1f987d8e363f1d16'
    producer = json.loads(read(final_path))
    assert producer['artifact_count_excluding_this_receipt'] == 240 and producer['writes_stop_after_this_receipt'] is True
    paths = {final_path, runtime_path, full, full_root,
             'quality_reports/RSM_REPAIRED_FULL_SUITE_EXECUTION_ATTEMPT03/FULL_OUTPUT.txt',
             'quality_reports/RSM_REPAIRED_FULL_SUITE_EXECUTION_ATTEMPT03/FULL_MODULE_RESULTS.json',
             'quality_reports/RSM_PRODUCER_ACTUAL_COMPLETION_ROOT_READBACK_ATTEMPT01.json',
             str(Path(__file__).relative_to(ROOT)), str((HERE / 'run_actual_audit.py').relative_to(ROOT))}
    for row in producer['all_owned_preparation_execution_output_inventory_excluding_this_receipt']:
        raw = read(row['path'])
        assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
        paths.add(row['path'])
    # Existing bytecode is executable import input even with -B. Archive it.
    for folder in ['factory', 'oracles', 'generators/other_tracks', 'generators/skills/prospective_option_resumption_attempt01']:
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*.pyc') if p.is_file())
    archive = HERE / 'raw_archives'
    archive.mkdir(exist_ok=False)

    def frozen(path, external=False):
        raw = read(path)
        assert len(raw) < 25 * 1024 * 1024, 'new raw artifact cap'
        h = digest(raw)
        target = archive / (h + '.raw')
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        assert read(target) == raw
        return {'sha256': h, 'bytes': len(raw), 'archive': str(target.relative_to(ROOT))}

    for path in sorted(paths):
        assert 'c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e' not in path
        assert path not in {'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/FINAL_RECEIPT.json', 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-execution.json', 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/run_task_solver_attempt01.py'}
        if path not in primary:
            primary[path] = frozen(path)
    modules = []
    environment_paths = {Path(sys.executable).resolve()}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            modules.append({'module': name, 'kind': 'BUILTIN_OR_FROZEN', 'origin': getattr(getattr(module, '__spec__', None), 'origin', None)})
            continue
        path = Path(filename).resolve()
        if path == Path(__file__).resolve():
            continue
        assert str(path).startswith('/opt/anaconda3/'), str(path)
        environment_paths.add(path)
        cached = getattr(module, '__cached__', None)
        if cached and Path(cached).is_file():
            environment_paths.add(Path(cached).resolve())
        modules.append({'module': name, 'kind': 'FILE_BACKED_BENIGN_RUNTIME', 'path': str(path)})
    # Freeze package source/cache/extension inventories for benign lazy imports.
    for module in [jsonschema, attr, attrs, referencing, rpds]:
        for folder in module.__path__:
            environment_paths.update(p.resolve() for p in Path(folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.pyc', '.so', '.dylib'})
    links = []
    pending = list(environment_paths)
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix not in {'.so', '.dylib'} and path != Path(sys.executable).resolve():
            continue
        result = subprocess.run(['/usr/bin/otool', '-L', str(path)], capture_output=True, text=True, check=True)
        links.append({'path': str(path), 'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode})
        for line in result.stdout.splitlines()[1:]:
            value = line.strip().split(' (', 1)[0]
            linked = None
            if value.startswith('/opt/anaconda3/'):
                linked = Path(value)
            elif value.startswith('@rpath/'):
                linked = Path('/opt/anaconda3/lib') / value[len('@rpath/'):]
            elif value.startswith('@loader_path/'):
                linked = path.parent / value[len('@loader_path/'):]
            if linked and linked.exists() and linked.resolve() not in environment_paths:
                environment_paths.add(linked.resolve())
                pending.append(linked.resolve())
    environment = {str(path): frozen(path, external=True) for path in sorted(environment_paths)}
    for module in sys.modules.values():
        origin = getattr(module, '__file__', None)
        if origin and Path(origin).resolve() != Path(__file__).resolve():
            assert str(Path(origin).resolve()) in environment, origin
    config = {'schema': 'rsm-root-actual-machine-audit-v1', 'actual_runner_execution_id': '/root', 'batch': 'option-resumption-pilot-v1', 'data_approval': False,
              'primary_bindings': primary, 'environment_bindings': environment, 'resolved_python': str(Path(sys.executable).resolve()),
              'full_suite_execution': full, 'full_suite_root_completion': full_root,
              'execution_output': 'quality_reports/RSM_ACTUAL_MACHINE_AUDIT_EXECUTION_ATTEMPT01',
              'runtime_boundary': {'preparation': 'Benign stdlib/schema package discovery and metadata/raw custody only; no semantic helper import or new predictions. All recorded concrete file-backed runtime and local linked-library bytes freeze before actual audit import. Apple shared-cache/system libraries are disclosed OS dependencies, not extracted byte archives.', 'python': sys.version, 'modules': modules, 'linked_libraries': links, 'cached_bytecode_included': True}}
    config_binding = write(HERE / 'CONFIG.json', config)
    write(HERE / 'SOURCE_PREEXECUTION_FREEZE.json', {'status': 'ROOT_SOURCE_PREPARED_NOT_EXECUTED', 'actual_author_execution_id': '/root', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'config': config_binding, 'primary_pairs': len(primary), 'environment_pairs': len(environment), 'semantic_import_or_execution': False, 'data_approval': False})
    print(json.dumps({'config': config_binding, 'primary_pairs': len(primary), 'environment_pairs': len(environment), 'no_semantic_execution': True}, sort_keys=True))


if __name__ == '__main__':
    main()
