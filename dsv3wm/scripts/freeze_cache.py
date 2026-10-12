"""Freeze cached instrument/trained artifacts before cached native comparisons."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
out=ROOT/'prereg/CACHE_FREEZE_V3.json'
if out.exists():raise ValueError('cache freeze exists')
paths=[*ROOT.rglob('*.py'),*ROOT.glob('configs/*pilot.json'),ROOT/'prereg/PILOT_AMENDMENT_2_CACHE.md',
       ROOT/'datasets/pilot-v1/dev.jsonl',ROOT/'datasets/pilot-v1/train_register.jsonl',REPO/'bwm/code/bwmn/lm.py',
       REPO/'bwm/code/bwmn/gpu_lease.py',REPO/'bwm/code/scripts/planbench_qwen.py']
for arm in ('A0','A1','A2','A5'):
    paths.extend([ROOT/'artifacts'/f'{arm}_s0/adapter/adapter_model.safetensors',ROOT/'artifacts'/f'{arm}_s0/run.json'])
record={'instrument':'native-standard-kv-v3','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
        'prompt_styles':{'plain':'matched comparison','chat':'post hoc diagnostic only'},
        'dev_cohort':{'domain':'register','first_records':2,'horizons':[1,2,4]},'confirmation_access':False}
out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n');print(str(out))
