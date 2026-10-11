#!/usr/bin/env python3
"""Scoped, bounded Git backup; uses owner's existing credentials per process only."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
BRANCH='research/goal-conditioned-decision-data-v1'
DEADLINE=None
sys.path.insert(0,str(REPO))
from goal_conditioned_decision_data.pipelines.common import atomic_json

def execute(command,env=None,timeout=60,check=True):
    if DEADLINE is not None:
        timeout=min(timeout,max(0.1,DEADLINE-time.monotonic()))
        if DEADLINE-time.monotonic()<=0:raise RuntimeError('backup deadline exceeded')
    result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('command failed: '+command[0]+' '+result.stderr[-1500:])
    return result

def remote_command(command,env,timeout=45):
    """Retry only transport failures; auth/rebase/conflict errors require review."""
    attempts=json.loads((ROOT/'configs/production.json').read_text())['backup_transport_attempts']
    if type(attempts) is not int or not 1<=attempts<=3:raise RuntimeError('invalid backup retry ceiling')
    for attempt in range(attempts):
        try:return execute(command,env=env,timeout=timeout)
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            message=str(error).lower()
            transport=isinstance(error,subprocess.TimeoutExpired) or any(s in message for s in ['could not resolve','failed to connect','connection timed out','connection reset','remote end hung up','rpc failed'])
            if not transport or attempt==attempts-1 or DEADLINE-time.monotonic()<5:raise
            time.sleep(min(2**(attempt+1),max(0,DEADLINE-time.monotonic()-1)))

def main():
    global DEADLINE
    DEADLINE=time.monotonic()+140
    token=None
    state=ROOT/'state/backup.json'
    old=json.loads(state.read_text()) if state.exists() else {}
    try:
        if execute(['git','branch','--show-current']).stdout.strip()!=BRANCH:
            raise RuntimeError('backup branch guard failed')
        status=execute(['git','status','--porcelain','-z']).stdout.split('\x00')
        if any(row and not row[3:].startswith('goal_conditioned_decision_data/') for row in status):
            raise RuntimeError('unrelated changes present; refusing backup')
        # Never echo the credential, place it on disk, or change global auth state.
        token=execute(['gh','auth','token','--user','LUNCY579']).stdout.strip()
        env=os.environ.copy();env['GH_TOKEN']=token
        git=['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
        execute(['git','add','--','goal_conditioned_decision_data'])
        changed=execute(['git','diff','--cached','--quiet'],check=False).returncode
        if changed:
            stats=ROOT/'state/metrics.json'
            s=json.loads(stats.read_text()) if stats.exists() else {}
            accepted=s.get('accepted',0); quarantine=s.get('quarantined',0)
            execute(['git','commit','-m',f'GCDL A-H verified production: {accepted} accepted, {quarantine} quarantined; differential and integrity gates'])
        remote=remote_command(git+['ls-remote','--heads','origin',BRANCH],env).stdout.strip()
        if remote:
            remote_command(git+['pull','--rebase','origin',BRANCH],env)
        else:
            remote_command(git+['fetch','origin','main'],env)
            execute(['git','rebase','origin/main'])
        remote_command(git+['push','--set-upstream','origin','HEAD:refs/heads/'+BRANCH],env,timeout=60)
        local=execute(['git','rev-parse','HEAD']).stdout.strip()
        remote=remote_command(git+['ls-remote','--heads','origin',BRANCH],env).stdout.split()[0]
        if local!=remote: raise RuntimeError('post-push remote HEAD mismatch')
        atomic_json(state,{'status':'confirmed','branch':BRANCH,'commit':local,'remote_commit':remote,'last_success_epoch':time.time(),'failures':0})
        print(json.dumps({'status':'confirmed','commit':local,'branch':BRANCH}))
        return 0
    except Exception as error:
        detail=str(error)
        if token:detail=detail.replace(token,'[REDACTED]')
        detail=re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1[REDACTED]@',detail)
        detail=re.sub(r'\b(?:gho_|ghp_|github_pat_)[A-Za-z0-9_]+','[REDACTED]',detail)
        preserved={k:old[k] for k in ['branch','commit','remote_commit','last_success_epoch'] if k in old}
        atomic_json(state,{**preserved,'status':'failed','last_attempt_epoch':time.time(),'failures':old.get('failures',0)+1,'reason':detail})
        print(detail,file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
