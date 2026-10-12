#!/usr/bin/env python3
"""Replace only the owned Git store after verified payload eviction."""
import argparse
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.pipelines import remote_archive as archive

BRANCH='research/goal-conditioned-decision-data-v1'
REMOTE='https://github.com/LUNCY579/rineya-world-model-research.git'
DEADLINE=ContextVar('gcdl_git_compact_deadline',default=None)
LIVE_ROOT=ContextVar('gcdl_git_compact_root',default=None)
INITIAL_PAUSE=ContextVar('gcdl_git_compact_initial_pause',default=None)

def _remaining(timeout=45):
    limit=DEADLINE.get()
    value=timeout if limit is None else min(timeout,limit-time.monotonic())
    if value<=0:raise RuntimeError('owned Git compaction deadline exceeded')
    return value

def _paused(root):
    archive._check_stop(root)
    if not archive._path(root,'state/PAUSED.json').exists():
        raise RuntimeError('Git store replacement requires the worker pause')
    raw=archive._local(root,'state/PAUSED.json',archive.MAX_READ_BYTES)[0]
    if INITIAL_PAUSE.get() is not None and raw!=INITIAL_PAUSE.get():
        raise RuntimeError('operator pause changed during Git compaction; original bytes retained')
    return raw

def _command(directory,command,env,timeout=45,max_output=8*1024*1024):
    deadline=time.monotonic()+_remaining(timeout)
    process=None;selector=selectors.DefaultSelector()
    try:
        process=subprocess.Popen(command,cwd=directory,env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,stdin=subprocess.DEVNULL)
        selector.register(process.stdout,selectors.EVENT_READ)
        chunks=[];total=0
        while selector.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise RuntimeError('owned Git compaction deadline exceeded')
            for key,_ in selector.select(remaining):
                raw=os.read(key.fileobj.fileno(),min(65536,max_output+1-total))
                if not raw:selector.unregister(key.fileobj);continue
                total+=len(raw)
                if total>max_output:raise RuntimeError('owned Git response exceeds bounded bytes')
                chunks.append(raw)
        if process.wait(timeout=max(0.001,deadline-time.monotonic())):
            raise RuntimeError('owned Git compaction command failed')
        return b''.join(chunks)
    except (OSError,subprocess.TimeoutExpired):
        raise RuntimeError('owned Git compaction transport/deadline failure') from None
    finally:
        selector.close()
        if process is not None:
            if process.poll() is None:process.kill();process.wait(timeout=2)
            process.stdout.close()

def _git(directory,args,environment=None,timeout=45):
    env=os.environ.copy() if environment is None else environment.copy()
    env['GIT_TERMINAL_PROMPT']='0'
    for key in list(env):
        if key.startswith('GIT_TRACE') or key in ('GH_DEBUG','GIT_CURL_VERBOSE'):env.pop(key,None)
    if environment is None:
        env['GIT_ALLOW_PROTOCOL']='';env['GIT_NO_LAZY_FETCH']='1';env['GIT_PROTOCOL_FROM_USER']='0'
    return _command(directory,['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential',*args],env,timeout)

def _owner_environment():
    env=os.environ.copy()
    for key in ('GH_TOKEN','GITHUB_TOKEN','GH_ENTERPRISE_TOKEN','GITHUB_ENTERPRISE_TOKEN','GH_DEBUG','GH_HOST'):
        env.pop(key,None)
    for key in list(env):
        if key.startswith('GIT_TRACE') or key=='GIT_CURL_VERBOSE':env.pop(key,None)
    raw=_command(ROOT.parent,['gh','auth','token','--user','LUNCY579','--hostname','github.com'],env,15,4096)
    token=raw.decode().strip()
    if not re.fullmatch('[A-Za-z0-9_]+',token):
        raise RuntimeError('owner credential unavailable for Git compaction')
    env['GH_TOKEN']=token;env['GH_PROMPT_DISABLED']='1'
    env['GIT_ALLOW_PROTOCOL']='https'
    return env

def _object_ids(repository):
    env=os.environ.copy();env['GIT_ALLOW_PROTOCOL']='';env['GIT_NO_LAZY_FETCH']='1'
    raw=_git(repository,['cat-file','--batch-all-objects','--batch-check=%(objectname)','--unordered'],env)
    return set(raw.decode().splitlines())

def _file_hash(path,limit=archive.MAX_READ_BYTES):
    before=path.stat()
    if before.st_size>limit:raise RuntimeError('retained source/runtime file exceeds bounded bytes')
    sha=hashlib.sha256();total=0
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(65536),b''):
            _remaining();total+=len(chunk)
            if total>limit:raise RuntimeError('retained file exceeded bounded bytes')
            sha.update(chunk)
    if archive._identity(before)!=archive._identity(path.stat()) or total!=before.st_size:
        raise RuntimeError('retained file changed while hashing')
    return sha.hexdigest()

def _tracked_files(repo):
    paths=_git(repo,['ls-files','-z','--','goal_conditioned_decision_data']).decode().split('\0')
    result={}
    for relative in paths:
        if not relative:continue
        path=repo/relative
        if path.is_symlink() or not path.is_file() or path.resolve()!=path or path.stat().st_nlink!=1:
            raise RuntimeError('retained source/control is not a regular owned file')
        result[relative]=_file_hash(path)
    return result

def _state_files(root):
    result={};total=0
    for path in (root/'state').rglob('*'):
        if path.is_symlink():raise RuntimeError('unsafe retained runtime control')
        if not path.is_file():continue
        if path.resolve()!=path or path.stat().st_nlink!=1:
            raise RuntimeError('unsafe retained runtime control')
        total+=path.stat().st_size
        if total>archive.MAX_BLOB_BYTES:raise RuntimeError('retained runtime exceeds bounded bytes')
        result[str(path.relative_to(root))]=(_file_hash(path,archive.MAX_BLOB_BYTES),archive._identity(path.stat()))
    return result

def _directory(path):
    if path.is_symlink() or path.resolve()!=path or not path.is_dir():
        raise RuntimeError('unsafe or foreign owned Git directory')
    value=path.stat();return [value.st_dev,value.st_ino]

def _primary_clean(repository):
    _directory(repository);_directory(repository/'.git')
    children=list(repository.iterdir())
    metadata_only=len(children)==1 and children[0].name=='.git' and not (repository/'.git/index').exists()
    if not metadata_only and _git(repository,['status','--porcelain','-z','--untracked-files=all','--ignored=matching']).strip():
        raise RuntimeError('primary Git checkout is dirty/untracked; every original byte retained')
    for path in repository.rglob('*'):
        _remaining()
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink!=1):
            raise RuntimeError('primary Git store contains a symlink/shared hardlink')

def _local_refs(repository):
    result={};heads={};mirrors={}
    for row in _git(repository,['for-each-ref','--format=%(refname) %(objectname)']).decode().splitlines():
        reference,oid=row.split()
        if reference.startswith('refs/heads/'):
            key=reference;heads[key]=oid
        elif reference.startswith('refs/remotes/origin/') and reference!='refs/remotes/origin/HEAD':
            key='refs/heads/'+reference[len('refs/remotes/origin/'):];mirrors[key]=oid
        elif reference=='refs/remotes/origin/HEAD':continue
        else:raise RuntimeError('stash/tag/unpublished reference refuses Git cache removal')
        if key in result and result[key]!=oid:raise RuntimeError('local branch is not mirrored by origin')
        result[key]=oid
    if any(mirrors.get(reference)!=oid for reference,oid in heads.items()):
        raise RuntimeError('local branch has no exact origin mirror; unpublished reference retained')
    return result

def _server_heads(repository,env):
    result={}
    for row in _git(repository,['ls-remote','--heads','origin'],env).decode().splitlines():
        oid,reference=row.split()
        if not re.fullmatch('[0-9a-f]{40}',oid) or not reference.startswith('refs/heads/') or reference in result:
            raise RuntimeError('invalid remote Git branch inventory')
        result[reference]=oid
    return result

def _primary_reachable(repository,head,refs,server):
    if head in set(server.values()):return True
    # Existing commit/tree metadata is sufficient: the exact confirmed remote
    # branch retains every ancestor, and no payload blob is read or fetched.
    for reference,tip in refs.items():
        if server.get(reference)!=tip:continue
        try:
            ancestor=_git(repository,['merge-base',head,tip]).decode().strip()
        except RuntimeError:
            continue
        if ancestor==head:return True
    return False

def _cleanup_tree(path,identity):
    _remaining()
    if not path.exists() and not path.is_symlink():return
    if _directory(path)!=identity:raise RuntimeError('compaction path replaced; unrelated bytes retained')
    _remove_tree(path)

def _remove_tree(path):
    """Stream directory entries and check the deadline before every removal."""
    _remaining()
    if LIVE_ROOT.get() is not None:_paused(LIVE_ROOT.get())
    with os.scandir(path) as entries:
        for entry in entries:
            _remaining()
            if LIVE_ROOT.get() is not None:_paused(LIVE_ROOT.get())
            if entry.is_symlink():raise RuntimeError('retired store acquired a foreign symlink')
            target=path/entry.name
            if entry.is_dir(follow_symlinks=False):_remove_tree(target)
            else:
                details=entry.stat(follow_symlinks=False)
                if details.st_nlink!=1:raise RuntimeError('retired store acquired a shared hardlink')
                target.unlink()
    _remaining();path.rmdir()

def _write_receipt(root,record):
    archive._write(root,'state/git_compaction.json',(archive._canonical(record)+'\n').encode(),immutable=False)

def _pending_cleanup_record(root,host,head,index_digest):
    """Validate a pending retired-store receipt without removing any bytes."""
    if not archive._path(root,'state/git_compaction.json').exists():return None
    record=json.loads(archive.read_text(root,'state/git_compaction.json'))
    if record.get('status')!='compacted_cleanup_pending':return None
    if (record.get('repository')!=archive.REPOSITORY or record.get('programme')!=archive.PROGRAMME
            or record.get('commit')!=head or record.get('archive_index_digest')!=index_digest
            or not re.fullmatch(r'\.git-compaction-[0-9a-f]{32}',record.get('transaction',''))):
        raise RuntimeError('invalid retired-store cleanup receipt')
    transaction=host/record['transaction']
    if _directory(transaction)!=record.get('transaction_identity'):
        raise RuntimeError('retired-store transaction directory replaced')
    if set(path.name for path in transaction.iterdir())-{'retired','worktree'}:
        raise RuntimeError('unexpected files in retired-store transaction')
    for name,key in (('retired','retired_identity'),('worktree','temporary_identity')):
        path=transaction/name
        if (path.exists() or path.is_symlink()) and _directory(path)!=record.get(key):
            raise RuntimeError('pending retired-store path replaced; unrelated bytes retained')
    return record

def _cleanup_pending(root,host,head,index_digest):
    record=_pending_cleanup_record(root,host,head,index_digest)
    if record is None:return None
    transaction=host/record['transaction']
    _paused(root)
    try:
        _cleanup_tree(transaction/'retired',record['retired_identity'])
        _cleanup_tree(transaction/'worktree',record['temporary_identity'])
        transaction.rmdir()
    except (OSError,RuntimeError):return record
    record.update(status='compacted',archived_blob_objects_remaining=0)
    _write_receipt(root,record);return record

def compact(root=ROOT,*,lock_fd=None,transport=None,seconds=180):
    if type(seconds) not in (int,float) or not 0<seconds<=180:raise RuntimeError('invalid bounded compaction deadline')
    token=DEADLINE.set(time.monotonic()+seconds)
    root_token=LIVE_ROOT.set(Path(root).resolve())
    pause_token=INITIAL_PAUSE.set(None)
    try:return _compact(root,lock_fd=lock_fd,transport=transport)
    finally:
        INITIAL_PAUSE.reset(pause_token);LIVE_ROOT.reset(root_token);DEADLINE.reset(token)

def _compact(root,*,lock_fd,transport):
    original_root=Path(root).absolute()
    root=archive._root(root);repo=root.parent;host=repo.parent
    if original_root.resolve()!=original_root or repo.name!='worktree' or host.name!='gcdl001-host':
        raise RuntimeError('Git compaction is restricted to the owned GCDL host')
    old_repository=host/'repository'
    if old_repository.is_symlink() or old_repository.resolve()!=old_repository:
        raise RuntimeError('unsafe owned Git repository path')
    host_identity=_directory(host);repo_identity=_directory(repo);root_identity=_directory(root)
    old_identity=_directory(old_repository);_directory(old_repository/'.git')
    with archive._lock(root,lock_fd):
        INITIAL_PAUSE.set(_paused(root))
        from goal_conditioned_decision_data.auditors.release import current_independent_review
        if not current_independent_review(deadline=DEADLINE.get()):raise RuntimeError('current independent review required')
        if _git(repo,['branch','--show-current']).decode().strip()!=BRANCH:
            raise RuntimeError('Git compaction branch guard failed')
        if _git(repo,['status','--porcelain','-z','--untracked-files=all']).strip():
            raise RuntimeError('Git compaction requires a clean confirmed code branch')
        common=Path(_git(repo,['rev-parse','--git-common-dir']).decode().strip())
        if not common.is_absolute():common=(repo/common).resolve()
        if common!=old_repository/'.git':raise RuntimeError('Git common store is not owned')
        if _git(repo,['remote','get-url','origin']).decode().strip()!=REMOTE:
            raise RuntimeError('Git compaction remote guard failed')
        worktrees=_git(repo,['worktree','list','--porcelain']).decode().splitlines()
        locations={Path(line[9:]).resolve() for line in worktrees if line.startswith('worktree ')}
        if locations!={old_repository,repo}:
            raise RuntimeError('other worktrees must be preserved/retired before store replacement')
        _primary_clean(old_repository)
        backup=json.loads(archive.read_text(root,'state/backup.json'))
        head=_git(repo,['rev-parse','HEAD']).decode().strip()
        if backup.get('status')!='confirmed' or backup.get('commit')!=head or backup.get('remote_commit')!=head:
            raise RuntimeError('confirmed GitHub code backup required before compaction')
        index=archive.load_index(root)
        if not index['files']:raise RuntimeError('verified cold custody required before compaction')
        for relative in index['files']:
            archive.deletion_custody(root,relative)
            if archive._path(root,relative).exists():
                raise RuntimeError('local archived payload still exists; compaction refused')
        cold_ids={entry['git_blob_sha1'] for entry in index['files'].values()}
        head_blobs={row.split()[1] for row in _git(repo,['ls-tree','-r','--format=%(objecttype) %(objectname)','HEAD']).decode().splitlines() if row.startswith('blob ')}
        if cold_ids.intersection(head_blobs):
            raise RuntimeError('archived byte alias remains required by active HEAD; full eviction impossible')
        before_objects=_object_ids(repo)
        refs=_local_refs(repo)  # Locally decidable refusals precede authentication.
        index_digest=index.get('digest')
        if not isinstance(index_digest,str) or not re.fullmatch('[0-9a-f]{64}',index_digest):
            raise RuntimeError('invalid archive index digest for Git compaction')
        # The current path index can point at a newer archived version. Older
        # catalogued bytes still belong to custody and may remain in Git history.
        for descriptor in index['catalogs'].values():
            catalog=archive._catalog(json.loads(archive.read_text(root,descriptor['catalog_path'])))
            cold_ids.update(meta['git_blob_sha1'] for meta in catalog['files'].values())
        if cold_ids.intersection(head_blobs):
            raise RuntimeError('archived byte alias remains required by active HEAD; full eviction impossible')
        pending_record=_pending_cleanup_record(root,host,head,index_digest)
        present_cold_ids=cold_ids.intersection(before_objects)
        remote=transport or archive.GitHubTransport(root,seconds=_remaining(180))
        for identifier,descriptor in index['catalogs'].items():
            catalog=json.loads(archive.read_text(root,descriptor['catalog_path']))
            if pending_record is not None:
                # A partially removed retired store may no longer have usable
                # Git metadata. Never infer that its last payload bytes vanished
                # from the empty replacement inventory: keep full raw proof.
                archive._remote_proof(root,catalog,descriptor['catalog_commit'],remote)
            else:
                required={meta['git_blob_sha1'] for meta in catalog['files'].values()} & present_cold_ids
                archive._remote_proof(root,catalog,descriptor['catalog_commit'],remote,required_blob_ids=required)
        _paused(root)
        env=_owner_environment();server=_server_heads(repo,env)
        primary_head=_git(old_repository,['rev-parse','HEAD']).decode().strip()
        if (server.get('refs/heads/'+BRANCH)!=head or not _primary_reachable(repo,primary_head,refs,server)
                or any(server.get(reference)!=oid for reference,oid in refs.items())):
            raise RuntimeError('local branch/detached HEAD is not confirmed on GitHub')
        if cold_ids.intersection(_object_ids(repo))!=present_cold_ids:
            raise RuntimeError('archived Git objects changed during proof; original store retained')
        if _pending_cleanup_record(root,host,head,index_digest)!=pending_record:
            raise RuntimeError('retired-store cleanup receipt changed during proof; remaining bytes retained')
        pending_cleanup=_cleanup_pending(root,host,head,index_digest)
        if pending_cleanup is not None and pending_cleanup['status']=='compacted_cleanup_pending':return pending_cleanup
        if not cold_ids.intersection(before_objects):
            result=pending_cleanup or {'status':'already_compact','repository':archive.REPOSITORY,'programme':archive.PROGRAMME,
                    'archive_index_digest':index_digest,'archived_blob_objects':0,'archived_blob_objects_remaining':0,
                    'commit':head,'PAUSED_preserved':True,'utc_epoch':time.time()}
            _paused(root);_write_receipt(root,result);return result
        retained=_tracked_files(repo);runtime=_state_files(root)
        if (repo/'.git').is_symlink() or (repo/'.git').stat().st_nlink!=1:
            raise RuntimeError('unsafe active Git pointer')
        original_git=(repo/'.git').read_bytes()
        suffix=uuid.uuid4().hex
        transaction=host/('.git-compaction-'+suffix)
        transaction.mkdir(mode=0o700)  # Existing sibling collision never belongs to us.
        transaction_identity=_directory(transaction)
        replacement=transaction/'repository';temporary=transaction/'worktree';retired=transaction/'retired'
        installed=False;retired_moved=False;committed=False
        replacement_identity=None;temporary_identity=None;pointer_identity=None
        pending=transaction/'pointer'
        try:
            replacement.mkdir(mode=0o700);replacement_identity=_directory(replacement)
            _git(host,['clone','--no-checkout','--depth','1','--single-branch','--filter=blob:none','--branch',BRANCH,REMOTE,str(replacement)],env,90)
            _git(replacement,['update-ref','--no-deref','HEAD',head])
            _git(replacement,['worktree','add','--no-checkout',str(temporary),BRANCH])
            temporary_identity=_directory(temporary)
            _git(temporary,['sparse-checkout','set','--cone','goal_conditioned_decision_data'],env)
            _git(temporary,['read-tree','-mu','HEAD'],env)
            _git(replacement,['remote','set-url','origin',REMOTE])
            if _git(temporary,['rev-parse','HEAD']).decode().strip()!=head or _tracked_files(temporary)!=retained:
                raise RuntimeError('replacement retained source/control bytes do not match')
            if cold_ids.intersection(_object_ids(temporary)):
                raise RuntimeError('replacement still stores archived payload blobs')
            if (_state_files(root)!=runtime or _tracked_files(repo)!=retained or _directory(host)!=host_identity
                    or _directory(repo)!=repo_identity or _directory(root)!=root_identity or _directory(old_repository)!=old_identity):
                raise RuntimeError('source/runtime changed during Git store preparation')
            _primary_clean(old_repository)
            if _local_refs(repo)!=refs or _server_heads(repo,env)!=server:
                raise RuntimeError('Git references changed during preparation')
            _paused(root)
            _remaining()
            pointer=(temporary/'.git').read_text().strip()
            if not pointer.startswith('gitdir: '):raise RuntimeError('invalid replacement Git pointer')
            meta=Path(pointer[8:])
            if not meta.is_relative_to(replacement/'.git/worktrees') or meta.resolve()!=meta:
                raise RuntimeError('replacement worktree metadata is outside owned store')
            relative_meta=meta.relative_to(replacement)
            (meta/'gitdir').write_text(str(repo/'.git')+'\n')
            if (_directory(transaction)!=transaction_identity or _directory(replacement)!=replacement_identity
                    or retired.exists() or retired.is_symlink()):
                raise RuntimeError('Git replacement collision; every original byte retained')
            if cold_ids.intersection(_object_ids(repo))!=present_cold_ids:
                raise RuntimeError('archived Git objects changed during preparation; original store retained')
            os.rename(old_repository,retired);retired_moved=True
            os.rename(replacement,old_repository)
            fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            details=os.fstat(fd);pointer_identity=[details.st_dev,details.st_ino]
            with os.fdopen(fd,'wb') as stream:
                stream.write(('gitdir: '+str(old_repository/relative_meta)+'\n').encode())
                stream.flush();os.fsync(stream.fileno())
            os.replace(pending,repo/'.git');installed=True
            _git(repo,['worktree','repair',str(repo)])
            if (_git(repo,['rev-parse','HEAD']).decode().strip()!=head or _tracked_files(repo)!=retained or _state_files(root)!=runtime
                    or _directory(repo)!=repo_identity or _directory(root)!=root_identity):
                raise RuntimeError('installed Git store/runtime verification failed')
            if cold_ids.intersection(_object_ids(repo)):
                raise RuntimeError('installed Git store contains archived payload objects')
            if not current_independent_review(deadline=DEADLINE.get()):raise RuntimeError('review guard failed after Git replacement')
            _paused(root)
            # The active worktree/state inode never moved; the shared lock remains live.
            removed_objects=len(cold_ids.intersection(before_objects))
            result={'status':'compacted_cleanup_pending','repository':archive.REPOSITORY,'programme':archive.PROGRAMME,
                    'archive_index_digest':index_digest,
                    'transaction':transaction.name,'transaction_identity':transaction_identity,
                    'retired_identity':old_identity,'temporary_identity':temporary_identity,
                    'commit':head,'archived_blob_objects_removed':removed_objects,
                    'archived_blob_objects_remaining':removed_objects,'retained_tracked_files':len(retained),
                    'retained_runtime_files':len(runtime),'PAUSED_preserved':True,'utc_epoch':time.time()}
            _write_receipt(root,result)
            # Once cleanup starts, retired may be partially removed. A durable
            # healthy replacement must never be rolled back to that old cache.
            committed=True
            return _cleanup_pending(root,host,head,index_digest)
        except BaseException:
            if retired_moved and not committed:
                if (_directory(retired)!=old_identity or _directory(repo)!=repo_identity
                        or _directory(root)!=root_identity or _directory(host)!=host_identity):
                    raise RuntimeError('rollback path replaced; original and replacement stores retained')
                if old_repository.exists() and _directory(old_repository)!=replacement_identity:
                    raise RuntimeError('rollback store replaced; unrelated bytes retained')
                if installed:
                    pointer=repo/'.git'
                    if pointer.is_symlink() or pointer.resolve()!=pointer or not pointer.is_file() or pointer.stat().st_nlink!=1:
                        raise RuntimeError('rollback pointer replaced; both stores retained')
                    pointer.write_bytes(original_git)
                if old_repository.exists():os.replace(old_repository,replacement)
                os.replace(retired,old_repository)
            raise
        finally:
            # Only unsuccessful replacement copies; originals remain on any failure.
            if not committed:
                if replacement_identity is not None:_cleanup_tree(replacement,replacement_identity)
                if temporary_identity is not None:_cleanup_tree(temporary,temporary_identity)
                if pointer_identity is not None and pending.exists():
                    value=pending.lstat()
                    if pending.is_symlink() or [value.st_dev,value.st_ino]!=pointer_identity or value.st_nlink!=1:
                        raise RuntimeError('private pointer file replaced; bytes retained')
                    pending.unlink()
                if transaction.exists() and _directory(transaction)==transaction_identity and not list(transaction.iterdir()):transaction.rmdir()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    print(json.dumps(compact(),sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
