"""Freeze prospective recovery readouts, binding exact trained or inherited adapters."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))


def validate_run(directory, train):
    from dsv3wm.native import EXPECTED_CORE_SHA, sha256, verify_freeze
    from dsv3wm.recovery import INITIAL_F_SHA, RecoveryConfig
    directory, train = Path(directory).resolve(), Path(train).resolve()
    record = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if record.get('status') != 'pilot-trained' or record.get('initial_adapter_sha256') != INITIAL_F_SHA:
        raise ValueError('completed warm-start recovery training required')
    command = record['command']
    config = (REPO / command[command.index('--config') + 1]).resolve()
    training_freeze = (REPO / command[command.index('--freeze') + 1]).resolve()
    qualification = (REPO / command[command.index('--qualification') + 1]).resolve()
    if (sha256(config) != record['config_sha256'] or RecoveryConfig.read(config).dict() != record['config']
            or sha256(train) != record['data_sha256'] or sha256(qualification) != record['qualification_sha256']
            or verify_freeze(training_freeze) != record['freeze_sha256'] or record['core_sha256'] != EXPECTED_CORE_SHA):
        raise ValueError('recovery training provenance drift')
    adapter = directory / 'adapter'
    if sha256(adapter / 'adapter_model.safetensors') != record['adapter_sha256']:
        raise ValueError('trained adapter weight drift')
    peft = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if (peft['r'] != 16 or peft['lora_alpha'] != 32 or peft['lora_dropout'] != 0.05
            or set(peft['target_modules']) != targets):
        raise ValueError('registered capacity/configuration drift')
    return record['config']['arm'], [directory / 'run.json', config, training_freeze, qualification,
                                   adapter / 'adapter_model.safetensors', adapter / 'adapter_config.json']


def main():
    from dsv3wm.native import sha256
    from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--run', action='append', type=Path, default=[])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('readout freeze exists; use a new output')
    dataset = args.dataset.resolve()
    required = ('train.jsonl', 'dev.jsonl', 'dev_invalid.jsonl', 'planning_dev.jsonl',
                'planning_scoring.json', 'manifest.json', 'parent_registry.json')
    paths = [dataset / name for name in required]
    dependencies = ('__init__', 'config', 'data', 'domains', 'native', 'objectives',
                    'planning', 'provenance', 'recovery', 'recovery_data', 'recovery_planning')
    paths += [*[ROOT / f'src/dsv3wm/{name}.py' for name in dependencies], *ROOT.glob('configs/recovery_*.json'),
              ROOT / 'scripts/evaluate_recovery.py', ROOT / 'scripts/evaluate_recovery_planning.py',
              ROOT / 'scripts/evaluate_recovery_retention.py', ROOT / 'results/retention_core_v1.fixtures.json',
              ROOT / 'scripts/analyze_recovery_traces.py', Path(__file__),
              ROOT / 'scripts/run_measured.py',
              ROOT / 'prereg/RECOVERY_G3_V1.md', ROOT / 'prereg/RECOVERY_CLARIFICATION_1.md',
              ROOT / 'prereg/RECOVERY_RESOURCE_INSTRUMENT_V1.md',
              INITIAL_F / 'adapter_model.safetensors', INITIAL_F / 'adapter_config.json',
              REPO / 'bwm/code/bwmn/lm.py', REPO / 'bwm/code/bwmn/rw2.py',
              REPO / 'bwm/code/bwmn/gpu_lease.py', REPO / 'bwm/code/scripts/planbench_qwen.py']
    if sha256(INITIAL_F / 'adapter_model.safetensors') != INITIAL_F_SHA:
        raise ValueError('inherited adapter identity drift')
    runs = {}
    for directory in args.run:
        arm, bound = validate_run(directory, dataset / 'train.jsonl')
        if arm in runs:
            raise ValueError('duplicate arm in readout freeze')
        runs[arm] = str(directory.resolve())
        paths += bound
    record = {'instrument': 'registered-F-recovery-readout-v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(path.resolve()).replace('\\', '/'): sha256(path)
                        for path in sorted(set(paths))},
              'runs': runs, 'inherited_adapter_sha256': INITIAL_F_SHA,
              'confirmation_access': False,
              'scope': 'prospective development readout; original B0 and training freezes unchanged'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
