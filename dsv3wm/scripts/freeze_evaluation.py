"""Freeze corrected pilot source/instruments before any dev generation results."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
out = ROOT/'prereg/EVALUATION_FREEZE_V2.json'
if out.exists():
    raise ValueError('evaluation freeze already exists')
paths = [*ROOT.rglob('*.py'), *ROOT.glob('configs/*pilot.json'), ROOT/'prereg/PILOT_V1.md',
         ROOT/'prereg/PILOT_AMENDMENT_1.md', ROOT/'datasets/pilot-v1/dev.jsonl',
         ROOT/'datasets/pilot-v1/train_register.jsonl', REPO/'bwm/code/bwmn/lm.py',
         REPO/'bwm/code/bwmn/gpu_lease.py', REPO/'bwm/code/scripts/planbench_qwen.py']
value = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'head':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
         'files':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
         'dev_cohort':{'domain':'register','first_records':2,'horizons':[1,2,4]},
         'cache_policy':'disabled','decoding':'greedy, max64 tokens','confirmation_access':False}
out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
print(str(out))
