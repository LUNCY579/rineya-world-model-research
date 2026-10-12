"""Freeze original-format T2 R/P retention before baseline/adapted behavior."""
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
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--adapter', action='append', type=Path, default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('retention freeze exists')
    if (sha256(T2_ADAPTER / 'adapter_model.safetensors') != T2_SHA
            or sha256(T2_ADAPTER / 'adapter_config.json') != T2_CONFIG_SHA):
        raise ValueError('registered T2 identity drift')
    paths = [ROOT / 'configs/rcwm_retention_v1.json', ROOT / 'datasets/rcwm-retention-v1/fixtures.json',
             ROOT / 'scripts/build_rcwm_retention.py', ROOT / 'scripts/evaluate_rcwm_retention.py',
             ROOT / 'scripts/run_measured.py', Path(__file__), ROOT / 'prereg/RCWM_RETENTION_V1.md',
             ROOT / 'src/dsv3wm/native.py', ROOT / 'src/dsv3wm/provenance.py',
             ROOT / 'src/dsv3wm/config.py', ROOT / 'src/dsv3wm/domains.py',
             REPO / 'rcwm/experiments/RCWM-T1/rcwmt/env.py', REPO / 'bwm/code/bwmn/lm.py',
             REPO / 'bwm/code/bwmn/gpu_lease.py', REPO / 'bwm/code/scripts/planbench_qwen.py']
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
        if baseline.get('status') != 'development-evaluated' or baseline.get('adapter_sha256') != T2_SHA:
            raise ValueError('completed unchanged T2 baseline required')
        paths.append(args.baseline)
    for directory in [T2_ADAPTER, *args.adapter]:
        paths += [directory / 'adapter_model.safetensors', directory / 'adapter_config.json']
    record = {'instrument': 'rcwm-RP-retention-v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': {str(path.resolve()).replace('\\', '/'): sha256(path) for path in sorted(set(paths))},
              'confirmation_access': False, 'scope': 'supplementary per-case behavior; no broad retention pass'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
