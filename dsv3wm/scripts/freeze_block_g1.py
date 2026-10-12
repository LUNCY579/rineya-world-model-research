"""Freeze constructed original-delta G1 inputs and private initializer."""
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
    from build_block_g1 import required_sources, BW_ADAPTER, BW_SHA, BW_CONFIG_SHA
    from evaluate_block_g1 import validate_cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Block G1 freeze exists')
    data = ROOT / 'datasets/block-g1-v1'
    rows = [json.loads(line) for line in (data / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    validate_cases(rows)
    if (sha256(BW_ADAPTER / 'adapter_model.safetensors') != BW_SHA
            or sha256(BW_ADAPTER / 'adapter_config.json') != BW_CONFIG_SHA):
        raise ValueError('registered Block initializer drift')
    paths = [*required_sources(), *data.glob('*'), Path(__file__), ROOT / 'scripts/run_measured.py',
             ROOT / 'prereg/BLOCK_G1_EXECUTION_V1.md',
             BW_ADAPTER / 'adapter_model.safetensors', BW_ADAPTER / 'adapter_config.json']
    record = {'instrument': 'constructed-original-delta-Block-G1-v1',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(path.resolve()).replace('\\', '/'): sha256(path)
                        for path in sorted(set(paths)) if path.is_file()},
              'confirmation_access': False, 'scope': 'original-functionality qualification only; raw/filter outcomes separate'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
