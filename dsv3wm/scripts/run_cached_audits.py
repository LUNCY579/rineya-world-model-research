"""Sequential native supplemental audits under the current authorized session."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent


def call(script,args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=REPO,check=True)


def main():
    core=ROOT/'results/retention_core_v1.json'
    call('audit_retention.py',['--config',ROOT/'configs/A0_pilot.json','--out',core])
    for arm in ('A0','A1','A2','A5'):
        cfg=ROOT/'configs'/f'{arm}_pilot.json';adapter=ROOT/'artifacts'/f'{arm}_s0/adapter'
        call('audit_retention.py',['--config',cfg,'--adapter',adapter,'--baseline',core,
                                  '--out',ROOT/'results'/f'retention_{arm}_v1.json'])
        call('cached_pilot.py',['--config',cfg,'--adapter',adapter,'--data',ROOT/'datasets/pilot-v1/dev.jsonl',
            '--out',ROOT/'results'/f'{arm}_cached_planning.json','--freeze',ROOT/'prereg/CACHE_FREEZE_V3.json',
            '--probe',ROOT/'results/cache_probe_cpu.json','--planning'])
    call('cached_pilot.py',['--config',ROOT/'configs/A0_pilot.json','--adapter',ROOT/'artifacts/A0_s0/adapter',
         '--data',ROOT/'datasets/pilot-v1/dev.jsonl','--out',ROOT/'results/A0_chat_diagnostic.json',
         '--freeze',ROOT/'prereg/CACHE_FREEZE_V3.json','--probe',ROOT/'results/cache_probe_cpu.json','--prompt-style','chat'])
    call('registered_forward_probe.py',['--adapter',
        REPO.parent/'Rineya_BWM_001_Research_Bundle/native_v1/runs/F_titan_main/adapter',
        '--out',ROOT/'results/registered_forward_probe_cpu.json'])


if __name__=='__main__':main()
