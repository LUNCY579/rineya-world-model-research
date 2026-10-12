"""Create a new cached-instrument freeze for an independently reproduced run."""
import argparse,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
ap=argparse.ArgumentParser()
ap.add_argument('--dataset',required=True)
ap.add_argument('--run',action='append',required=True,help='ARM=run_directory, repeated for A0/A1/A2/A5')
ap.add_argument('--out',required=True)
ap.add_argument('--probe',required=True)
a=ap.parse_args();out=Path(a.out).resolve()
if out.exists():raise ValueError('replica freeze exists')
runs={}
for spec in a.run:
    arm,path=spec.split('=',1)
    if arm in runs or arm not in {'A0','A1','A2','A5'}:raise ValueError('duplicate/unknown arm')
    runs[arm]=Path(path).resolve()
if set(runs)!={'A0','A1','A2','A5'}:raise ValueError('all four trained arms required')
dataset=Path(a.dataset).resolve()
paths=[*ROOT.rglob('*.py'),ROOT/'prereg/PILOT_AMENDMENT_2_CACHE.md',dataset/'train_register.jsonl',dataset/'dev.jsonl',
       REPO/'bwm/code/bwmn/lm.py',REPO/'bwm/code/bwmn/gpu_lease.py',REPO/'bwm/code/scripts/planbench_qwen.py']
sys.path.insert(0,str(ROOT/'src'))
from dsv3wm.provenance import validate_training_lineage
from dsv3wm.native import EXPECTED_CORE_SHA
probe=Path(a.probe).resolve();qualification=json.loads(probe.read_text(encoding='utf-8'))
if (qualification.get('status')!='passed' or qualification.get('parent_sha256')!=EXPECTED_CORE_SHA
    or qualification.get('script_sha256')!=hashlib.sha256((ROOT/'scripts/cache_probe.py').read_bytes()).hexdigest()):
    raise ValueError('qualified unchanged native cache probe required')
paths.extend([probe,REPO/'bwm/code/bwmn/rw2.py',REPO/'bwm/code/bwmn/blocks.py',REPO/'rcwm/experiments/RCWM-T1/rcwmt/env.py'])
for arm,directory in runs.items():
    config,adapter_config=validate_training_lineage(directory,dataset,arm)
    paths.extend([directory/'run.json',directory/'adapter/adapter_model.safetensors',adapter_config,config])
files={str(p.resolve()).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
record={'instrument':'native-standard-kv-v4','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'files':files,'replica_runs':{arm:str(path) for arm,path in runs.items()},'confirmation_access':False,
        'scope':'new independently timed run; never overwrite original freeze/results'}
# Required source-membership identities use repo-relative keys; external run/data
# paths remain absolute and are resolved natively by verify_freeze on Windows.
for p in (ROOT/'src/dsv3wm/cached_native.py',ROOT/'scripts/cached_pilot.py',ROOT/'scripts/cache_probe.py',
          ROOT/'prereg/PILOT_AMENDMENT_2_CACHE.md',REPO/'bwm/code/bwmn/lm.py'):
    record['files'][p.relative_to(REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
p=ROOT/'src/dsv3wm/provenance.py'
record['files'][p.relative_to(REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(record,indent=2),encoding='utf-8',newline='\n');print(str(out))
