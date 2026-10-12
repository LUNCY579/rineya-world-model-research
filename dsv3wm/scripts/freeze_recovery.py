"""Prospective recovery protocol/source/data/model freeze; no model execution."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import sys
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from dsv3wm.recovery import INITIAL_F,INITIAL_F_SHA
from dsv3wm.native import sha256
out=ROOT/'prereg/RECOVERY_FREEZE_V1.json'
if out.exists():raise ValueError('recovery freeze exists')
if sha256(INITIAL_F/'adapter_model.safetensors')!=INITIAL_F_SHA:raise ValueError('initial model identity mismatch')
paths=[ROOT/'prereg/RECOVERY_G3_V1.md',*ROOT.glob('configs/recovery*.json'),*ROOT.glob('src/dsv3wm/*.py'),
       ROOT/'scripts/train_recovery.py',ROOT/'scripts/evaluate_recovery.py',ROOT/'scripts/build_recovery_data.py',
       ROOT/'scripts/freeze_recovery.py',REPO/'bwm/code/bwmn/lm.py',REPO/'bwm/code/bwmn/rw2.py',
       REPO/'bwm/code/bwmn/gpu_lease.py',REPO/'bwm/code/scripts/planbench_qwen.py',
       INITIAL_F/'adapter_model.safetensors',INITIAL_F/'adapter_config.json',*ROOT.glob('datasets/recovery-v1/*')]
record={'instrument':'registered-F-recovery-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.resolve()).replace('\\','/'):sha256(p) for p in sorted(set(paths)) if p.is_file()},
        'confirmation_access':False,'initial_adapter_sha256':INITIAL_F_SHA,
        'registered_parent_sha256':'a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284'}
out.write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n');print(str(out))
