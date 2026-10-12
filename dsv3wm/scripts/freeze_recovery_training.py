"""Bind clarified recovery source and successful B0, before adapted outcomes."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from dsv3wm.native import sha256
from dsv3wm.recovery import INITIAL_F,INITIAL_F_SHA
ap=argparse.ArgumentParser();ap.add_argument('--qualification',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out)
if out.exists():raise ValueError('training freeze exists')
q=json.loads(Path(a.qualification).read_text(encoding='utf-8'))
if q['status']!='development-evaluated' or not q['baseline_qualified'] or q['adapter_sha256']!=INITIAL_F_SHA:
    raise ValueError('qualified B0 required')
paths=[*ROOT.glob('configs/recovery*.json'),*ROOT.glob('src/dsv3wm/*.py'),ROOT/'scripts/train_recovery.py',
       ROOT/'scripts/evaluate_recovery.py',ROOT/'scripts/freeze_recovery_training.py',
       ROOT/'prereg/RECOVERY_G3_V1.md',ROOT/'prereg/RECOVERY_CLARIFICATION_1.md',Path(a.qualification),
       *ROOT.glob('datasets/recovery-v1/*'),INITIAL_F/'adapter_model.safetensors',INITIAL_F/'adapter_config.json',
       REPO/'bwm/code/bwmn/lm.py',REPO/'bwm/code/bwmn/rw2.py',REPO/'bwm/code/bwmn/gpu_lease.py',REPO/'bwm/code/scripts/planbench_qwen.py']
record={'instrument':'registered-F-recovery-training-v2','qualification_freeze_sha256':q['freeze_sha256'],
        'qualification_sha256':sha256(a.qualification),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.resolve()).replace('\\','/'):sha256(p) for p in sorted(set(paths)) if p.is_file()},'confirmation_access':False}
out.write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n');print(str(out))
