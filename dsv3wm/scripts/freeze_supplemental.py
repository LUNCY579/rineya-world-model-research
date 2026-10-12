"""Create once, before supplemental native measurements."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
paths=[ROOT/'scripts/planning_evaluate.py',ROOT/'scripts/audit_retention.py',ROOT/'scripts/measure_pilot.py',
       ROOT/'prereg/SUPPLEMENTAL_AUDITS_V1.md',ROOT/'src/dsv3wm/planning.py',ROOT/'src/dsv3wm/native.py',
       ROOT/'datasets/pilot-v1/dev.jsonl']
out=ROOT/'prereg/SUPPLEMENTAL_FREEZE_V1.json'
if out.exists():
    raise ValueError('supplemental freeze exists')
value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
       'files':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
       'confirmation_access':False}
out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
print(str(out))
