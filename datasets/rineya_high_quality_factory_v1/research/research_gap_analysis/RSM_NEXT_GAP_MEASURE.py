"""Measure the accepted RSM input grammar without invoking a producer or oracle."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'research/research_gap_analysis/RSM_NEXT_GAP_MEASURE_SOURCE_FREEZE.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    freeze = json.loads(FREEZE.read_bytes())
    for name, item in freeze['bindings'].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], name
    assert freeze['bindings'][str(Path(__file__).relative_to(ROOT))]['sha256'] == sha(Path(__file__).read_bytes())
    path = ROOT / 'data/RSM-001/releases/v0001/canonical.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert len(rows) == 24 and all(row['review_status'] == 'ACCEPTED' for row in rows)
    parents = {}
    for row in rows:
        family = row['semantic_family_id']
        task = row['semantic_task']
        if family in parents:
            assert task == parents[family]['semantic_task']
        else:
            parents[family] = row

    summaries = []
    for family, row in sorted(parents.items()):
        task = row['semantic_task']
        assert set(task) == {'kind', 'state', 'procedures', 'main'}
        assert task['kind'] == 'skills' and set(task['state']) == {'x', 'y'}
        counts = collections.Counter()
        graph = {name: set() for name in task['procedures']}
        max_branch_depth = 0

        def walk(instructions, owner=None, depth=0):
            nonlocal max_branch_depth
            max_branch_depth = max(max_branch_depth, depth)
            for op in instructions:
                counts[op['op']] += 1
                if op['op'] == 'call' and owner is not None:
                    graph[owner].add(op['name'])
                if op['op'] == 'if':
                    walk(op['then'], owner, depth + 1)
                    walk(op['else'], owner, depth + 1)

        walk(task['main'])
        for name, instructions in task['procedures'].items():
            walk(instructions, name)

        def call_depth(name, stack=()):
            assert name not in stack, 'Accepted procedure graph must be acyclic'
            return 1 + max([call_depth(child, stack + (name,)) for child in graph[name]] or [0])

        main_calls = [op['name'] for op in task['main'] if op['op'] == 'call']
        summaries.append({
            'semantic_family_id': family,
            'representative_sample_id': row['sample_id'],
            'split': row['split'], 'difficulty': row['difficulty'],
            'source_task_sha256': sha(json.dumps(task, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()),
            'declared_procedures': len(task['procedures']),
            'syntactic_operations_including_both_branches': dict(sorted(counts.items())),
            'maximum_syntactic_conditional_depth': max_branch_depth,
            'maximum_declared_call_depth': max(call_depth(name) for name in graph),
            'nested_procedure_call_edges': sum(map(len, graph.values())),
            'repeated_top_level_direct_calls': len(main_calls) != len(set(main_calls)),
            'target_fields': sorted(row['verified_target']),
        })
    report = {
        'actual_author_execution': '/root', 'actual_runner_execution': '/root',
        'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'COMPLETED_ACCEPTED_INPUT_GRAMMAR_COVERAGE_MEASUREMENT',
        'source_sha256': sha(Path(__file__).read_bytes()),
        'source_freeze_sha256': sha(FREEZE.read_bytes()),
        'frozen_inputs': freeze['bindings'],
        'approved_catalog_snapshot_records': 3748, 'approved_catalog_snapshot_source_families': 696,
        'RSM_accepted_records': len(rows), 'RSM_source_families': len(parents),
        'record_splits': dict(collections.Counter(row['split'] for row in rows)),
        'family_splits': dict(collections.Counter(row['split'] for row in parents.values())),
        'difficulty_records': dict(collections.Counter(str(row['difficulty']) for row in rows)),
        'skill_tags': sorted({tag for row in rows for tag in row['skill_tags']}),
        'source_family_input_grammar_measurements': summaries,
        'all_task_fields': sorted({key for row in rows for key in row['semantic_task']}),
        'all_target_fields': sorted({key for row in rows for key in row['verified_target']}),
        'observed_grammar_boundaries': {
            'state': 'Exactly two integer registers x and y',
            'procedures': 'Acyclic named instruction lists sharing the same register state',
            'execution': 'add, scale, swap, call and conditionals; full run from main',
            'formal_arguments_or_local_bindings': 'Absent from this task grammar',
            'option_initiation_or_success_contract': 'Absent from this task grammar',
            'interruption_checkpoint_or_resume_stack': 'Absent from this task grammar',
            'recover_without_replaying_completed_effects': 'Cannot be supervised explicitly by this task grammar',
        },
        'next_curriculum_hypothesis': 'Add explicit initiation, argument binding, success conditions and interrupted execution checkpoints, with non-idempotent effects that expose duplicate replay and paired resume-versus-restart tasks.',
        'prospective_requirements': [
            'Freeze a public typed task/output contract and controls before producer/oracle implementation',
            'Use a new task kind and version; preserve the original accepted release and old checker',
            'Independently implement resumption and validate checkpoints against supplied execution history',
            'Keep completed and pending work separate in structured supervision',
            'Family-split interrupted and uninterrupted counterparts before rendering',
            'Measure procedural structure, failure modes and matched controls separately from rendering count',
        ],
        'limitations': 'Static accepted input-grammar measurement; no learner results, replay correctness certification, unseen transfer claim or new data approval. Gold was read only to enumerate the existing output field names.',
        'new_candidate_records': 0, 'data_approval': False, 'model_training': False,
    }
    output = ROOT / 'research/research_gap_analysis/RSM_NEXT_GAP_MEASURE.json'
    for name, item in freeze['bindings'].items():
        assert sha((ROOT / name).read_bytes()) == item['sha256'], name
    with output.open('x') as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'records': len(rows), 'source_families': len(parents),
                      'report_sha256': sha(output.read_bytes())}))


if __name__ == '__main__':
    main()
