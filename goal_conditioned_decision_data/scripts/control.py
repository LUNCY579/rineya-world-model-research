#!/usr/bin/env python3
"""macOS launchd lifecycle for bounded GCDL production. Never starts training."""
import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
LABEL='com.rineya.gcdl001'
DOMAIN=f'gui/{os.getuid()}'
PLIST=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
sys.path.insert(0,str(REPO))
from goal_conditioned_decision_data.pipelines.common import atomic_json

def command(*args,check=True):
    return subprocess.run(args,capture_output=True,text=True,check=check)

def install():
    config=json.loads((ROOT/'configs/production.json').read_text())
    expected=[REPO/'goal_conditioned_decision_data',ROOT/'scripts/run_cycle.py']
    if not all(p.exists() for p in expected):raise RuntimeError('installation missing programme')
    content={'Label':LABEL,'ProgramArguments':[sys.executable,str(ROOT/'scripts/run_cycle.py')],
        'WorkingDirectory':str(REPO),'StartInterval':config['interval_seconds'],'RunAtLoad':True,
        'ProcessType':'Background','LowPriorityIO':True,'Nice':10,'StandardOutPath':'/dev/null','StandardErrorPath':'/dev/null',
        'EnvironmentVariables':{'PYTHONUNBUFFERED':'1'},'ExitTimeOut':10}
    PLIST.parent.mkdir(parents=True,exist_ok=True)
    if PLIST.exists() and plistlib.loads(PLIST.read_bytes()).get('WorkingDirectory')!=str(REPO):
        raise RuntimeError('runner label owned by different checkout')
    PLIST.write_bytes(plistlib.dumps(content))
    atomic_json(ROOT/'state/runner.json',{'label':LABEL,'plist':str(PLIST),'python':sys.executable,'interval':config['interval_seconds'],'host_limit':'runs while user logged in and host awake; no AI agent continuity; no wake from sleep'})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','status','resume','once','seal'])
    args=p.parse_args();(ROOT/'state').mkdir(exist_ok=True)
    if args.action=='status':
        for name in ['runner','heartbeat','metrics','checkpoint','backup','PAUSED']:
            f=ROOT/'state'/f'{name}.json'
            if f.exists():print(name+': '+f.read_text().strip())
        r=command('launchctl','print',DOMAIN+'/'+LABEL,check=False)
        print(r.stdout or r.stderr);return 0
    if args.action=='stop':
        (ROOT/'state/STOP').write_text('operator requested stop\n')
        command('launchctl','bootout',DOMAIN+'/'+LABEL,check=False)
        print('Stopped; accepted data and queue retained.');return 0
    if args.action in ['start','resume']:
        if args.action=='start' and (ROOT/'state/PAUSED.json').exists():
            raise RuntimeError('resolve pause gate and use resume')
        if args.action=='resume':
            (ROOT/'state/PAUSED.json').unlink(missing_ok=True)
        (ROOT/'state/STOP').unlink(missing_ok=True)
        install()
        r=command('launchctl','print',DOMAIN+'/'+LABEL,check=False)
        if r.returncode:command('launchctl','bootstrap',DOMAIN,str(PLIST))
        else:command('launchctl','kickstart',DOMAIN+'/'+LABEL)
        print('Runner active: '+str(PLIST));return 0
    flags=['--seal','--no-backup'] if args.action=='seal' else []
    return subprocess.run([sys.executable,str(ROOT/'scripts/run_cycle.py'),*flags],cwd=REPO).returncode
if __name__=='__main__':raise SystemExit(main())
