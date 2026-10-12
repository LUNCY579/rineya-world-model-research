"""Freeze a qualified shared teacher corpus and actual A0/A2 warm artifacts."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    from dsv3wm.native import sha256
    from dsv3wm.data import load_records
    from dsv3wm.recovery import INITIAL_F
    from dsv3wm.recovery_student import StudentConfig, load_teacher_corpus, admit_readout_lineage
    from freeze_recovery_student_readout import validate_warm_history
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('student freeze exists')
    train = ROOT / 'datasets/recovery-v1/train.jsonl'
    tasks = ROOT / 'datasets/recovery-h3-v1/planning_train.jsonl'
    rows = load_records(train, expected_split='train')
    task_rows = [json.loads(line) for line in tasks.read_text(encoding='utf-8').splitlines() if line.strip()]
    positives, manifest, gate = load_teacher_corpus(args.teacher_dir, rows, task_rows)
    if manifest['train_file_sha256'] != sha256(train) or manifest['tasks_file_sha256'] != sha256(tasks):
        raise ValueError('qualified teacher inputs differ from the frozen pools')
    dependencies = ('__init__', 'recovery_student', 'recovery', 'recovery_data', 'recovery_planning',
                    'native', 'provenance', 'data', 'domains', 'planning', 'objectives', 'config', 'proposer')
    paths = [*[ROOT / f'src/dsv3wm/{name}.py' for name in dependencies], train, tasks,
             ROOT / 'datasets/recovery-v1/planning_dev.jsonl', ROOT / 'datasets/recovery-v1/dev.jsonl',
             ROOT / 'scripts/train_recovery_student.py', ROOT / 'scripts/evaluate_recovery_student.py',
             ROOT / 'scripts/freeze_recovery_student_readout.py',
             ROOT / 'scripts/run_measured.py', Path(__file__),
             ROOT / 'prereg/RECOVERY_H3_DRAFT_V1.md', ROOT / 'prereg/RECOVERY_H3_CONTROL_CLARIFICATION_V1.md',
             ROOT / 'prereg/RECOVERY_H3_TEACHER_V1.md', ROOT / 'prereg/RECOVERY_RESOURCE_CLARIFICATION_2.md',
             ROOT / 'results/recovery-v1/B0_qualification.json', args.teacher_dir / 'manifest.json',
             *[args.teacher_dir / name for name in manifest['output_sha256']],
             INITIAL_F / 'adapter_config.json', INITIAL_F / 'adapter_model.safetensors',
             REPO / 'bwm/code/bwmn/rw2.py', REPO / 'bwm/code/bwmn/lm.py',
             REPO / 'bwm/code/bwmn/gpu_lease.py', REPO / 'bwm/code/scripts/planbench_qwen.py']
    for kind in ('A3', 'A4', 'C3', 'C4'):
        config_path = ROOT / f'configs/recovery_student_{kind}.json'
        cfg = StudentConfig.read(config_path)
        expected_arm = 'A3' if kind.endswith('3') else 'A4'
        if (cfg.arm != expected_arm or cfg.seed != 0 or cfg.plan_weight != (1.0 if kind.startswith('A') else 0.0)
                or cfg.control_extra_one_step != kind.startswith('C')):
            raise ValueError('student/control filename does not match its fixed recipe')
        warm = ROOT / f'artifacts/recovery-v1/{cfg.warm_arm}_s0'
        _, warm_sources, _ = admit_readout_lineage(warm / 'adapter', warm / 'run.json', cfg,
                                                  warm_baseline=True, train_path=train)
        warm_sources += validate_warm_history(warm, cfg, train)
        paths += [config_path, warm / 'run.json', warm / 'adapter/adapter_model.safetensors',
                  warm / 'adapter/adapter_config.json', ROOT / f'configs/recovery_{cfg.warm_arm}.json',
                  ROOT / 'prereg/RECOVERY_TRAIN_FREEZE_V2.json', *warm_sources]
    record = {'instrument': 'warm-verified-H3-students-controls-v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'corpus_gate': gate,
              'shared_verified_parent_count': len(positives), 'confirmation_access': False,
              'files': {str(path.resolve()).replace('\\', '/'): sha256(path) for path in sorted(set(paths))},
              'scope': 'prospective native A3/A4 and C3/C4 development; no efficacy confirmation'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
