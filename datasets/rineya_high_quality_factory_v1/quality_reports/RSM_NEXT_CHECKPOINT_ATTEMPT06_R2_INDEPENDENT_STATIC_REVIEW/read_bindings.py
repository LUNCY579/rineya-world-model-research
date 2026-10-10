"""Own metadata/opaque-byte reader; never imports reviewed sources or runs phases."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset')
NEW = ROOT / 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT06_R2'
OLD = ROOT / 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT05'
OUT = ROOT / 'quality_reports/RSM_NEXT_CHECKPOINT_ATTEMPT06_R2_INDEPENDENT_STATIC_REVIEW'

def digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            h.update(block)
            size += len(block)
    return size, h.hexdigest()

checked = {}
def bound(item):
    rel = item['path']
    path = Path(rel) if rel.startswith('/') else ROOT / rel
    assert not path.is_symlink(), rel
    actual = digest(path)
    assert actual == (item['size'], item['sha256']), (rel, actual, item)
    checked[rel] = {'path': rel, 'size': actual[0], 'sha256': actual[1]}

def load(path):
    return json.loads(path.read_bytes())

def recursive_bindings(value):
    if isinstance(value, dict):
        if {'path', 'size', 'sha256'} <= value.keys():
            bound(value)
        for child in value.values():
            recursive_bindings(child)
    elif isinstance(value, list):
        for child in value:
            recursive_bindings(child)

def test_bodies(path):
    source = path.read_text()
    tree = ast.parse(source)
    bodies = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    bodies[(cls.name, node.name)] = ast.get_source_segment(source, node)
    return bodies

ready = load(NEW / 'SOURCE_READY.json')
freeze = load(NEW / 'SOURCE_PRECOLLECTION_FINAL_FREEZE.json')
assert digest(NEW / 'SOURCE_READY.json') == (8497, '8903beca389841bd3f41bbff73e345bf5d74a5ee2102261b274bcfecca537b0f')
assert digest(NEW / 'SOURCE_PRECOLLECTION_FINAL_FREEZE.json') == (17727, '5a284554e88ad4cbe75ec13c01c0825420c2f9ffd34f8fa6ed7b3c8bc0c1d0c9')
recursive_bindings(ready)
recursive_bindings(freeze)
policy = load(NEW / 'EXACT_FAILED_STAGE_ARCHIVAL_EVIDENCE_POLICY.json')
pack_plan = load(ROOT / 'backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT04/EXACT46_ARCHIVAL_EVIDENCE_PACK_PLAN_R2.json')
assert pack_plan['planned_mappings'] == policy['raw_evidence']
recursive_bindings(policy)
recursive_bindings(pack_plan)
assert len(policy['raw_evidence']) == 46
base = load(ROOT / policy['base_mapping_contract']['path'])
assert len(base['mappings']) == 13
source_tree = ast.parse((NEW / 'archive_mapping.py').read_text())
literal = None
for statement in source_tree.body:
    if isinstance(statement, ast.Assign):
        if any(isinstance(target, ast.Name) and target.id == 'EXACT_FAILED_STAGE_EVIDENCE' for target in statement.targets):
            literal = ast.literal_eval(statement.value)
projected = {row['raw']['path']: {'raw': row['raw'], 'origin': row['origin']} for row in policy['raw_evidence']}
assert literal == projected
old_tests = test_bodies(OLD / 'tests.py')
new_tests = test_bodies(NEW / 'tests.py')
assert len(old_tests) == 76 and len(new_tests) == 79
assert all(new_tests.get(name) == body for name, body in old_tests.items())
same, changed = [], []
for row in ready['source_files']:
    name = Path(row['path']).name
    (same if digest(NEW / name) == digest(OLD / name) else changed).append(name)
assert set(changed) == {'common.py', 'bootstrap.py', 'archive_mapping.py', 'tests.py'}
assert len(same) == 8
failed_policy = load(NEW / 'FAILED_WORKTREE_PRESERVATION_POLICY.json')
recursive_bindings(failed_policy)
assert ready['collector_not_executed'] and ready['tests_not_executed'] and ready['inventory_not_executed']
assert not freeze['fresh_runtime_complete'] and not freeze['execution_authorized']
report = {
    'schema': 'own-source-only-metadata-readback-v1',
    'reader_execution': '/root/formal_oracle',
    'reviewed_helper_imports_phase_test_git_network_execution': False,
    'source_ready_sha256': digest(NEW / 'SOURCE_READY.json')[1],
    'source_precollection_freeze_sha256': digest(NEW / 'SOURCE_PRECOLLECTION_FINAL_FREEZE.json')[1],
    'source_only_assertions_passed': True,
    'checked_unique_byte_bindings': len(checked),
    'bindings': sorted(checked.values(), key=lambda row: row['path']),
    'exact_new_mapping_rows': 46,
    'retained_base_mapping_rows': 13,
    'policy_plan_literal_exact': True,
    'own_sources_unchanged_vs05': sorted(same),
    'own_sources_changed_vs05': sorted(changed),
    'old_test_method_bodies_exact': len(old_tests),
    'new_test_method_bodies_prepared_not_run': len(new_tests),
    'runtime_execution_approval': False,
    'data_approval': False,
}
(OUT / 'READBACK.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'bindings'}, sort_keys=True))
