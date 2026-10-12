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
REMOTE='https://github.com/LUNCY579/rineya-world-model-research.git'
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

def deleted_paths(status):
    result=[]
    for row in status:
        if not row:continue
        if len(row)<4 or row[2]!=' ':
            raise RuntimeError('malformed Git status; refusing backup')
        if any(flag in row[:2] for flag in ('R','C')):
            raise RuntimeError('rename/copy needs explicit provenance review before backup')
        if 'D' in row[:2]:result.append(row[3:])
    return result

def local_deletion_custody(paths):
    """An absent local file must never silently delete its only remote copy."""
    from goal_conditioned_decision_data.pipelines.remote_archive import deletion_custody
    result={}
    for path in paths:
        if not path.startswith('goal_conditioned_decision_data/'):
            raise RuntimeError('unrelated local deletion; refusing backup')
        relative=path.removeprefix('goal_conditioned_decision_data/')
        try:entry=deletion_custody(ROOT,relative)
        except Exception as error:
            raise RuntimeError('unverified local payload deletion: '+path) from error
        if not entry:
            raise RuntimeError('unverified local payload deletion: '+path)
        result[path]=entry
    return result

def revalidate_deletion_custody(custody,git,env):
    """The active code tree may shrink only while an immutable snapshot survives."""
    trees={};refs={}
    for path,entry in custody.items():
        commit=entry['catalog_commit'];ref=entry['archive_ref']
        if not re.fullmatch(r'[0-9a-f]{40}',commit) or ref!='refs/heads/archive/gcdl-001/'+commit:
            raise RuntimeError('invalid archive snapshot deletion custody')
        if ref not in refs:
            row=remote_command(git+['ls-remote','--heads','origin',ref],env).stdout.split()
            if len(row)!=2 or row[0]!=commit or row[1]!=ref:
                raise RuntimeError('archive snapshot missing or moved; deletion refused')
            refs[ref]=commit
        old_oid=execute(['git','rev-parse','HEAD:'+path]).stdout.strip()
        if old_oid!=entry['git_blob_sha1']:
            raise RuntimeError('archive custody does not match deleted HEAD payload')
        if commit not in trees:
            data=json.loads(execute(['gh','api','repos/LUNCY579/rineya-world-model-research/git/commits/'+commit],env=env).stdout)
            tree=json.loads(execute(['gh','api','repos/LUNCY579/rineya-world-model-research/git/trees/'+data['tree']['sha']+'?recursive=1'],env=env).stdout)
            if tree.get('truncated'):
                raise RuntimeError('truncated archive tree; deletion refused')
            trees[commit]={item['path']:item for item in tree['tree']}
        item=trees[commit].get(path)
        if not item or item['type']!='blob' or item['sha']!=old_oid or item.get('size')!=entry['bytes']:
            raise RuntimeError('remote snapshot payload mismatch; deletion refused')

def main():
    global DEADLINE
    budget=float(os.environ.get('GCDL_BACKUP_TIMEOUT_SECONDS','140'))
    if not 0<budget<=140:raise RuntimeError('invalid bounded backup deadline')
    DEADLINE=time.monotonic()+budget
    token=None
    state=ROOT/'state/backup.json'
    old=json.loads(state.read_text()) if state.exists() else {}
    try:
        if execute(['git','branch','--show-current']).stdout.strip()!=BRANCH:
            raise RuntimeError('backup branch guard failed')
        status=execute(['git','status','--porcelain','-z']).stdout.split('\x00')
        custody=local_deletion_custody(deleted_paths(status))
        if any(row and not row[3:].startswith('goal_conditioned_decision_data/') for row in status):
            raise RuntimeError('unrelated changes present; refusing backup')
        if execute(['git','remote','get-url','origin']).stdout.strip()!=REMOTE:
            raise RuntimeError('backup remote guard failed')
        # Never echo the credential, place it on disk, or change global auth state.
        auth_env=os.environ.copy()
        for key in ('GH_TOKEN','GITHUB_TOKEN','GH_ENTERPRISE_TOKEN','GITHUB_ENTERPRISE_TOKEN','GH_DEBUG','GH_HOST'):
            auth_env.pop(key,None)
        token=execute(['gh','auth','token','--user','LUNCY579','--hostname','github.com'],env=auth_env).stdout.strip()
        if not re.fullmatch('[A-Za-z0-9_]+',token):raise RuntimeError('owner credential unavailable')
        env=auth_env;env['GH_TOKEN']=token;env['GH_PROMPT_DISABLED']='1';env['GIT_ALLOW_PROTOCOL']='https'
        git=['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
        revalidate_deletion_custody(custody,git,env)
        execute(['git','add','--sparse','--','goal_conditioned_decision_data'])
        staged=execute(['git','diff','--cached','--name-only','--diff-filter=D','-z']).stdout.split('\x00')
        staged=[path for path in staged if path]
        if set(staged)!=set(custody):
            raise RuntimeError('staged deletion inventory changed; backup refused')
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
