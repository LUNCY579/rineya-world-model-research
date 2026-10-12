"""Sequential current-session native measurements. Never schedules or spawns peers."""
from pathlib import Path
import subprocess
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dsv3wm.native import sha256, verify_freeze
from dsv3wm.config import Config

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def run(script, args, result):
    if result.exists():
        record=json.loads(result.read_text(encoding='utf-8'))
        status=record['status']
        if status in {'pilot-evaluated','complete','completed','passed','loss-only-audit'}:
            for option, field in (('--config','config_sha256'),('--data','data_sha256'),('--baseline','baseline_file_sha256')):
                if option in args and record.get(field)!=sha256(args[args.index(option)+1]):
                    raise RuntimeError('Existing result provenance mismatch: '+option)
            if '--adapter' in args:
                adapter=Path(args[args.index('--adapter')+1])
                expected=sha256(adapter/'adapter_model.safetensors')
                actual=record.get('adapter_sha256') or record.get('adapter_files_sha256',{}).get('adapter_model.safetensors')
                if actual!=expected:
                    raise RuntimeError('Existing adapter provenance mismatch')
            if script=='evaluate.py' and record.get('freeze_sha256')!=verify_freeze(ROOT/'prereg/EVALUATION_FREEZE_V2.json'):
                raise RuntimeError('Existing dynamics freeze mismatch')
            if script=='planning_evaluate.py' and record.get('script_sha256')!=sha256(ROOT/'scripts'/script):
                raise RuntimeError('Existing planning source mismatch')
            if script=='audit_retention.py':
                sources=record.get('fixture_source_sha256',{})
                if sources.get('dsv3wm/scripts/audit_retention.py')!=sha256(ROOT/'scripts'/script):
                    raise RuntimeError('Existing retention source mismatch')
            print(f'Already recorded: {result.name} ({status})',flush=True)
            return
        raise RuntimeError(f'Previous run must be audited before retry: {result}, {status}')
    command=[sys.executable,str(ROOT/'scripts'/script),*args]
    print('Starting '+result.name,flush=True)
    subprocess.run(command,cwd=REPO,check=True)


def main():
    verify_freeze(ROOT/'prereg/SUPPLEMENTAL_FREEZE_V1.json')
    data=ROOT/'datasets/pilot-v1/dev.jsonl'
    for arm in ('A0','A1','A2','A5'):
        config=ROOT/'configs'/f'{arm}_pilot.json'
        adapter=ROOT/'artifacts'/f'{arm}_s0/adapter'
        result=ROOT/'results'/f'{arm}_dev.json'
        run('evaluate.py',['--config',str(config),'--data',str(data),'--domain','register','--limit','2',
                           '--adapter',str(adapter),'--out',str(result)],result)
    core=ROOT/'results/retention_core_v1.json'
    run('audit_retention.py',['--config',str(ROOT/'configs/A0_pilot.json'),'--out',str(core)],core)
    for arm in ('A0','A1','A2','A5'):
        config=ROOT/'configs'/f'{arm}_pilot.json'
        adapter=ROOT/'artifacts'/f'{arm}_s0/adapter'
        result=ROOT/'results'/f'retention_{arm}_v1.json'
        run('audit_retention.py',['--config',str(config),'--adapter',str(adapter),'--baseline',str(core),'--out',str(result)],result)
        planning=ROOT/'results'/f'{arm}_planning_dev.json'
        run('planning_evaluate.py',['--config',str(config),'--adapter',str(adapter),'--data',str(data),'--out',str(planning)],planning)


if __name__=='__main__':
    main()
