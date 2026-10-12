"""Prospectively extend the fixed TRAIN planning allowlist to 32 parents."""
import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def build(train, original, seed=202610123):
    from dsv3wm.recovery_data import _task
    from dsv3wm.domains import parent_world_key
    parents = {}
    for row in train:
        if row.get('split') != 'train' or row.get('domain') != 'register':
            raise PermissionError('TRAIN-only RegisterWorld source required')
        parent = parent_world_key('register', row['s0'], row['rules'])
        if parent != row['parent_world_id'] or parent in parents:
            raise ValueError('unique authentic TRAIN parents required')
        parents[parent] = row
    if len(original) != 16 or len({task['parent_world_id'] for task in original}) != 16:
        raise ValueError('existing fixed sixteen-task TRAIN allowlist required')
    tasks = copy.deepcopy(original)
    for task in tasks:
        if task.get('split') != 'train' or task['parent_world_id'] not in parents:
            raise PermissionError('original task is outside the TRAIN allowlist')
        row = parents[task['parent_world_id']]
        if task['s0'] != row['s0'] or task['rules'] != row['rules']:
            raise ValueError('original planning source mismatch')
    used = {task['parent_world_id'] for task in tasks}
    rng = random.Random(seed)
    for row in train:
        if row['parent_world_id'] in used:
            continue
        task, _ = _task(row, rng, 2, 4)
        task['provenance']['generator'] = 'dsv3wm-recovery-h3-prospective32-v1'
        tasks.append(task)
        used.add(task['parent_world_id'])
        if len(tasks) == 32:
            break
    if len(tasks) != 32:
        raise ValueError('insufficient TRAIN parents')
    return tasks


def main():
    from dsv3wm.data import load_records
    from dsv3wm.domains import canonical
    from dsv3wm.native import sha256
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'original-tasks', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('teacher task directory exists')
    train = load_records(args.train, expected_split='train')
    original = [json.loads(line) for line in args.original_tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    tasks = build(train, original)
    args.out.mkdir(parents=True, exist_ok=False)
    task_file = args.out / 'planning_train.jsonl'
    with task_file.open('x', encoding='utf-8') as stream:
        stream.write(''.join(canonical(task) + '\n' for task in tasks))
    record = {'schema': 'dsv3wm-recovery-h3-tasks-v1', 'seed': 202610123,
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'prospective TRAIN-only inputs; no native teacher outcomes',
              'selection': 'keep original sixteen; take next sixteen eligible parents in frozen TRAIN row order',
              'parents': 32, 'limits': sorted({task['rules']['limit'] for task in tasks}),
              'confirmation_access': False, 'solver_plans_generated': False,
              'goal_collection': 'new goals at exact distance 2..4; distances never appear in task inputs',
              'files': {str(path.resolve()): {'sha256': sha256(path), 'bytes': path.stat().st_size}
                        for path in (args.train, args.original_tasks, task_file, Path(__file__),
                                     ROOT / 'src/dsv3wm/recovery_data.py')}}
    with (args.out / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
