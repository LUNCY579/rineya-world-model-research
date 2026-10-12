"""Create prospective Ledger qualification, training or adapted readout freezes."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    from dsv3wm.native import sha256
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    from train_ledger_recovery import required_sources, qualified_baseline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--run', action='append', type=Path, default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Ledger freeze exists')
    data = ROOT / 'datasets/ledger-recovery-v1'
    if (sha256(T2_ADAPTER / 'adapter_model.safetensors') != T2_SHA
            or sha256(T2_ADAPTER / 'adapter_config.json') != T2_CONFIG_SHA):
        raise ValueError('registered T2 identity drift')
    paths = [*required_sources(), *ROOT.glob('configs/ledger_A*.json'), *data.glob('*'),
             ROOT / 'scripts/build_ledger_recovery.py', ROOT / 'scripts/run_measured.py', Path(__file__),
             ROOT / 'prereg/LEDGER_G1_G3_V1.md', ROOT / 'prereg/RECOVERY_RESOURCE_INSTRUMENT_V1.md',
             ROOT / 'prereg/RECOVERY_RESOURCE_CLARIFICATION_2.md',
             T2_ADAPTER / 'adapter_model.safetensors', T2_ADAPTER / 'adapter_config.json']
    record = {'instrument': 'tick-compatible-ledger-development-v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'confirmation_access': False}
    if args.qualification:
        qualification = json.loads(args.qualification.read_text(encoding='utf-8'))
        original = qualification.get('freeze_sha256')
        if not qualified_baseline(qualification, original, sha256(data / 'dev.jsonl')):
            raise ValueError('qualified native Ledger B0 required')
        record['qualification_freeze_sha256'] = original
        paths += [args.qualification, args.qualification.with_suffix('.traces.json')]
    for directory in args.run:
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        if (run.get('status') != 'pilot-trained' or run.get('initial_adapter_sha256') != T2_SHA
                or sha256(directory / 'adapter/adapter_model.safetensors') != run.get('adapter_sha256')
                or sha256(directory / 'adapter/adapter_config.json') != run.get('adapter_config_sha256')):
            raise ValueError('completed Ledger run/artifact lineage required')
        paths += [directory / 'run.json', directory / 'adapter/adapter_model.safetensors', directory / 'adapter/adapter_config.json']
    record['files'] = {str(path.resolve()).replace('\\', '/'): sha256(path)
                       for path in sorted(set(paths)) if path.is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
