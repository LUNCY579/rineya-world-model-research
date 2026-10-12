"""Seal the operational pilot's register-only projection and source/config hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError('freeze already exists')
    dataset = Path(a.dataset)
    rows = [json.loads(line) for line in (dataset/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    register = [r for r in rows if r['domain']=='register']
    projection = dataset/'train_register.jsonl'
    if projection.exists() or not register:
        raise ValueError('projection exists/empty')
    projection.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in register), encoding='utf-8')
    paths = [*ROOT.glob('configs/*pilot.json'), ROOT/'prereg/PILOT_V1.md', projection,
             dataset/'train.jsonl', dataset/'dev.jsonl', *ROOT.rglob('*.py')]
    record = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'command':sys.argv, 'files':{str(p.resolve()):sha(p) for p in sorted(set(paths))},
              'register_training_parents':len(register), 'confirmation_labels_accessed':False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record,indent=2), encoding='utf-8')
    print(json.dumps({'frozen':str(out),'register_training_parents':len(register)}))


if __name__=='__main__':
    main()
