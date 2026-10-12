"""Freeze the agreed 32-parent B0 learned teacher collection before outcomes."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from dsv3wm.native import sha256
    from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA
    from dsv3wm.recovery_planning import validate_tasks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('teacher freeze exists')
    tasks = ROOT / 'datasets/recovery-h3-v1/planning_train.jsonl'
    rows = [json.loads(line) for line in tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    validate_tasks(rows, expected_split='train')
    if len(rows) != 32 or sha256(INITIAL_F / 'adapter_model.safetensors') != INITIAL_F_SHA:
        raise ValueError('32 authentic TRAIN parents and registered B0 identity required')
    dependencies = ('__init__', 'config', 'data', 'domains', 'native', 'planning',
                    'provenance', 'recovery', 'recovery_data', 'recovery_planning')
    paths = [*[ROOT / f'src/dsv3wm/{name}.py' for name in dependencies],
             ROOT / 'configs/recovery_A0.json', ROOT / 'scripts/collect_recovery_teacher.py',
             ROOT / 'scripts/build_recovery_teacher_tasks.py', ROOT / 'scripts/run_measured.py', Path(__file__),
             ROOT / 'datasets/recovery-v1/train.jsonl', tasks,
             ROOT / 'datasets/recovery-h3-v1/manifest.json', ROOT / 'datasets/recovery-v1/parent_registry.json',
             ROOT / 'prereg/RECOVERY_H3_DRAFT_V1.md', ROOT / 'prereg/RECOVERY_H3_TEACHER_V1.md',
             ROOT / 'prereg/RECOVERY_RESOURCE_INSTRUMENT_V1.md',
             ROOT / 'prereg/RECOVERY_RESOURCE_CLARIFICATION_2.md',
             ROOT / 'results/recovery-v1/B0_qualification.json',
             INITIAL_F / 'adapter_model.safetensors', INITIAL_F / 'adapter_config.json',
             REPO / 'bwm/code/bwmn/rw2.py', REPO / 'bwm/code/bwmn/lm.py',
             REPO / 'bwm/code/bwmn/gpu_lease.py', REPO / 'bwm/code/scripts/planbench_qwen.py']
    record = {'instrument': 'recovery-H3-B0-teacher-v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(path.resolve()).replace('\\', '/'): sha256(path) for path in sorted(set(paths))},
              'config': str((ROOT / 'configs/recovery_A0.json').resolve()),
              'teacher_adapter': str(INITIAL_F.resolve()), 'teacher_adapter_sha256': INITIAL_F_SHA,
              'confirmation_access': False, 'native_student_training_authorized_by_this_freeze': False,
              'scope': 'TRAIN teacher collection only; student recipe remains a separate draft'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
