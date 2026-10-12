#!/usr/bin/env python3
"""One bounded GCDL-only GitHub backup, verification, eviction and Git cleanup."""
import argparse
import json
import os
from pathlib import Path
import shutil
import hashlib
import stat
import uuid
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.pipelines.common import atomic_json, file_hash, immutable_bytes
from goal_conditioned_decision_data.scripts import snapshot_state, compact_git
from goal_conditioned_decision_data.auditors.release import current_independent_review

BRANCH=compact_git.BRANCH
MAX_FILES=12
MAX_BYTES=128*1024*1024
SECONDS=280
DISK_REASONS={'resource gate: fewer than 10 GiB free',
 'resource gate: publication would leave fewer than 10 GiB free',
 'rendered export would leave fewer than 10 GiB free'}
MIGRATION='archive migration: verified GitHub custody and reviewed archive-aware code required'
CLEANUP='archive Git cleanup pending'
RENDER_MIGRATION='cold rendered metadata recovery exceeds cycle budget; reviewed operational metadata index required'

def cleanup_pending(root):
    root=archive._root(root)
    index=archive.load_index(root)
    path=archive._path(root,'state/git_compaction.json')
    record=json.loads(archive.read_text(root,'state/git_compaction.json')) if path.exists() else {}
    git_pending=record.get('status')=='compacted_cleanup_pending'
    incomplete=bool(index['files']) and (record.get('status') not in ('compacted','already_compact')
                                         or record.get('archive_index_digest')!=index['digest'])
    legacy=archive._path(root,'state/development_evacuation.json')
    legacy_pending=legacy.exists() and json.loads(archive.read_text(root,'state/development_evacuation.json')).get('pending') is not None
    return git_pending or incomplete or legacy_pending or bool(_pending_receipts(root,index))

def _command(root,args,deadline,environment=None):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise RuntimeError('archive cycle deadline exceeded')
    result=subprocess.run(args,cwd=root.parent,env=environment,capture_output=True,
                          timeout=min(90,remaining))
    if len(result.stdout)>8*1024*1024:raise RuntimeError('archive command response exceeds budget')
    if result.returncode:raise RuntimeError('archive command failed: '+args[0])
    return result.stdout

def _backup(root,deadline):
    archive._check_stop(root)
    _command(root,[sys.executable,'-B',str(root/'scripts/backup.py')],deadline)
    info=json.loads((root/'state/backup.json').read_text())
    head=_command(root,['git','rev-parse','HEAD'],deadline).decode().strip()
    if info.get('status')!='confirmed' or info.get('commit')!=head or info.get('remote_commit')!=head:
        raise RuntimeError('archive requires confirmed GitHub backup')
    return head

def _publish_snapshot(root,commit,deadline,transport):
    reference=archive.archive_ref(commit)
    env=transport._environment()
    env=dict(env);env['GIT_ALLOW_PROTOCOL']='https';env['GIT_TERMINAL_PROMPT']='0'
    git=['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
    rows=_command(root,git+['ls-remote','--heads','origin',reference],deadline,env).decode().split()
    if rows and rows!=[commit,reference]:raise RuntimeError('immutable archive snapshot already moved')
    if not rows:
        archive._check_stop(root)
        # POST creates a missing ref; it cannot fast-forward a concurrently created ref.
        # https://docs.github.com/en/rest/git/refs#create-a-reference
        raw=_command(root,['gh','api','--hostname','github.com','--method','POST',
                          'repos/'+archive.REPOSITORY+'/git/refs',
                          '-f','ref='+reference,'-f','sha='+commit],deadline,env)
        created=json.loads(raw)
        if created.get('ref')!=reference or created.get('object',{}).get('type')!='commit' or created.get('object',{}).get('sha')!=commit:
            raise RuntimeError('archive snapshot creation identity mismatch')
    rows=_command(root,git+['ls-remote','--heads','origin',reference],deadline,env).decode().split()
    if rows!=[commit,reference]:raise RuntimeError('archive snapshot remote confirmation failed')
    return reference

def _pending_receipts(root,index):
    pending=[]
    directory=root/'state/archive_evacuations'
    if directory.is_symlink() or directory.resolve()!=directory:
        raise RuntimeError('unsafe archive pending journal directory')
    for path in sorted(directory.glob('*.json')):
        relative=str(path.relative_to(root));identifier=path.stem
        record=json.loads(archive.read_text(root,relative,max_bytes=archive.MAX_INDEX_BYTES))
        if record.get('receipt_id')!=identifier:
            raise RuntimeError('archive pending journal custody mismatch')
        if record.get('pending') is not None:
            if identifier not in index['receipts']:
                raise RuntimeError('orphan archive pending journal; retained for reviewed recovery')
            pending.append(identifier)
    return pending

def _local_payloads(root):
    found=[]
    for path in root.rglob('*'):
        relative=str(path.relative_to(root))
        if path.is_file() and archive.is_payload_path(relative):
            archive._path(root,relative)
            found.append(relative)
    return sorted(found)

def _select_payloads(root,index,priority):
    """Keep completed rendering cohorts together so their controls precede eviction."""
    present=set(_local_payloads(root))
    fresh=present-set(index['files'])
    units=[];managed=set()
    for relative in archive.iter_paths(root,'manifests/render-batch-*.json'):
        receipt=archive._json(archive.read_bytes(root,relative))
        datasets=receipt['datasets']
        folders=[]
        for item in datasets:
            logical=archive._relative(item['manifest_path'])
            path=Path(logical)
            if (len(path.parts)!=4 or path.parts[:2]!=('data','rendered')
                    or path.name!='manifest.json' or path.parent.name!=item['dataset_id']):
                raise RuntimeError('unmanaged completed rendering cohort')
            archive._path(root,logical)
            folders.append(path.parent.as_posix()+'/')
        local={path for path in present if any(path.startswith(folder) for folder in folders)}
        if not local:continue
        cohort=set()
        for item in datasets:
            logical=item['manifest_path']
            if archive.payload_info(root,logical)['sha256']!=item['manifest_sha256']:
                raise RuntimeError('completed rendering cohort manifest changed')
            manifest=archive._json(archive.read_bytes(root,logical))
            cohort.add(logical)
            for shard in manifest['shards']:
                path=archive._relative(shard['path'])
                if Path(path).parent!=Path(logical).parent:
                    raise RuntimeError('completed rendering cohort shard path changed')
                info=archive.payload_info(root,path)
                if info['sha256']!=shard['sha256'] or info['bytes']!=shard['bytes']:
                    raise RuntimeError('completed rendering cohort shard changed')
                cohort.add(path)
        if local-cohort:
            raise RuntimeError('uncommitted payload in completed rendering cohort')
        managed.update(cohort)
        pending=sorted(cohort & fresh)
        if pending:
            # Finish a previously partial snapshot before opening a new cohort.
            units.append((0 if cohort & set(index['files']) else 1,relative,pending))
    ordered=[[path] for path in priority if path in fresh]
    ordered.extend(unit for _,_,unit in sorted(units))
    ordered.extend([path] for path in sorted(fresh-managed-set(priority)))
    selected=[];total=0
    for unit in ordered:
        sizes=[archive._path(root,path).stat().st_size for path in unit]
        if any(size>archive.MAX_BLOB_BYTES for size in sizes):
            raise RuntimeError('payload exceeds GitHub archive object budget')
        if len(unit)>MAX_FILES or sum(sizes)>MAX_BYTES:
            raise RuntimeError('completed rendering cohort exceeds archive batch bounds')
        if len(selected)+len(unit)>MAX_FILES or total+sum(sizes)>MAX_BYTES:continue
        selected.extend(unit);total+=sum(sizes)
    return selected

def _publish_render_controls(root,deadline,transport,lock_fd,*,max_receipts=None):
    """Qualify bounded label-free recovery controls before their raw headers leave."""
    from ..pipelines import render_controls
    with archive.read_session(root,deadline=deadline):
        archive._check_stop(root)
        pause=archive._path(root,'state/PAUSED.json')
        expected=archive._local(root,'state/PAUSED.json',archive.MAX_INDEX_BYTES)[0] if pause.exists() else None
        reason=archive._json(expected).get('reason') if expected else None
        if reason not in DISK_REASONS|{None,CLEANUP,MIGRATION,RENDER_MIGRATION,'archive maintenance in progress'}:
            raise RuntimeError('render control publication cannot override another pause')
        pending=render_controls.pending(root,transport=transport)
        if not pending:return None
        def gate():
            archive._check_stop(root)
            now=archive._local(root,'state/PAUSED.json',archive.MAX_INDEX_BYTES)[0] if pause.exists() else None
            if now!=expected:raise RuntimeError('pause changed during render control publication')
            if not current_independent_review(deadline=deadline):
                archive.check_read_deadline(root)
                raise RuntimeError('independent review changed during render control publication')
        gate()
        if max_receipts is not None and (type(max_receipts) is not int
                or not 1<=max_receipts<=render_controls.MAX_NEW_RECEIPTS):
            raise RuntimeError('invalid render control publication bound')
        if max_receipts is None and len(pending)>render_controls.MAX_NEW_RECEIPTS:
            raise RuntimeError('render control backfill required before automatic eviction')
        pending=pending[:max_receipts or render_controls.MAX_NEW_RECEIPTS]
        capsules=render_controls.prepare(root,pending,deadline=deadline,
                                        transport=transport,lock_fd=lock_fd)
        plan=render_controls.plan_bundle(root,capsules,transport=transport,lock_fd=lock_fd)
        gate()
        commit=_backup(root,deadline)
        gate()
        _publish_snapshot(root,commit,deadline,transport)
        gate()
        authority=render_controls.activate(root,plan['bundle_path'],commit,
                  transport=transport,deadline=deadline,lock_fd=lock_fd,
                  publication_guard=gate)
        gate()
        _backup(root,deadline)
        gate()
        return authority

def _require_render_controls(root,receipt,deadline):
    """An interrupted partial cohort cannot evade qualification through recovery."""
    from ..pipelines import rendered,render_controls
    candidates=[path for path in receipt['files'] if path.startswith('data/rendered/')]
    if not candidates:return
    with archive.read_session(root,deadline=deadline):
        for relative in archive.iter_paths(root,'manifests/render-batch-*.json'):
            completed=archive._json(archive.read_bytes(root,relative))
            folders=[str(Path(item['manifest_path']).parent)+'/' for item in completed['datasets']]
            if not any(path.startswith(folder) for path in candidates for folder in folders):continue
            batch=rendered._batch(root,completed['source_manifest'])
            if render_controls.try_projection(root,relative,completed,batch) is None:
                raise RuntimeError('completed rendering cohort requires qualified controls before eviction')
            rendered.validate_receipt(root,relative)

def _development_copy(root):
    source=root/'state/development_archive.tar.gz'
    if not source.exists():return None
    archive._path(root,'state/development_archive.tar.gz')
    manifest=json.loads((root/'manifests/DEVELOPMENT_ARCHIVE.json').read_text())
    if source.stat().st_size!=manifest['bytes'] or file_hash(source)!=manifest['sha256']:
        raise RuntimeError('development archive commitment mismatch')
    relative='data/backups/development_archive.tar.gz'
    destination=archive._path(root,relative)
    if source.stat().st_size>archive.MAX_BLOB_BYTES:raise RuntimeError('development archive exceeds object budget')
    raw=source.read_bytes()
    if destination.exists():
        if destination.read_bytes()!=raw:raise RuntimeError('development backup immutable conflict')
    else:immutable_bytes(destination,raw)
    return relative

def _remove_development_original(root,transport=None,deletion_guard=None):
    source_relative='state/development_archive.tar.gz'
    relative='data/backups/development_archive.tar.gz'
    source=archive._path(root,source_relative)
    journal='state/development_evacuation.json'
    custody='state/development_eviction_custody.json'
    if not archive.is_archived(root,relative):return 0
    if not source.exists() and archive._path(root,journal).exists():
        if json.loads(archive.read_text(root,journal)).get('status')=='evacuated':return 0
    index=archive.load_index(root);entry=index['files'][relative]
    info=archive.payload_info(root,relative)
    descriptor=index['catalogs'][entry['catalog_id']]
    catalog=json.loads(archive.read_text(root,descriptor['catalog_path']))
    archive._remote_proof(root,catalog,descriptor['catalog_commit'],transport or archive.GitHubTransport(root))
    raw=archive.read_bytes(root,relative,max_bytes=archive.MAX_BLOB_BYTES,transport=transport)
    if hashlib.sha256(raw).hexdigest()!=info['sha256']:
        raise RuntimeError('development archive remote raw-byte mismatch')
    # Reuse the reviewed anchored holding/journal protocol for this exact duplicate.
    if archive._path(root,custody).exists():
        alias=json.loads(archive.read_text(root,custody))
        if (alias.get('remote_receipt_id')!=entry['receipt_id']
                or alias.get('files')!={source_relative:{key:entry[key] for key in archive._META}}
                or alias.get('receipt_id')!=archive._digest({key:value for key,value in alias.items() if key!='receipt_id'})):
            raise RuntimeError('development duplicate custody changed')
    elif source.exists():
        _,meta,identity,parents=archive._local(root,source_relative,archive.MAX_BLOB_BYTES,False)
        if any(meta[key]!=entry[key] for key in ('sha256','bytes','git_blob_sha1')):
            raise RuntimeError('development original does not match verified remote payload')
        body={'remote_receipt_id':entry['receipt_id'],
              'files':{source_relative:{key:entry[key] for key in archive._META}},
              'local_identities':{source_relative:identity},
              'local_ancestry':{source_relative:parents}}
        alias={**body,'receipt_id':archive._digest(body)}
        atomic_json(root/custody,alias)
    else:return 0
    archive._check_stop(root)
    recovered=archive._recover_pending(root,alias,journal)
    deleted_bytes=0
    if source.exists():
        _,meta,identity,parents=archive._local(root,source_relative,archive.MAX_BLOB_BYTES,False)
        if identity!=recovered.get(source_relative,alias['local_identities'][source_relative]) or parents!=alias['local_ancestry'][source_relative]:
            raise RuntimeError('development duplicate identity or ancestry changed')
        archive._move_then_unlink(root,source_relative,alias,identity,parents,journal,[],deletion_guard=deletion_guard)
        deleted_bytes=info['bytes']
    archive._journal(root,journal,{'receipt_id':alias['receipt_id'],'status':'evacuated','verified_remote':True,'deleted':[source_relative]})
    manifest=root/'manifests/DEVELOPMENT_ARCHIVE.json'
    record=json.loads(manifest.read_text())
    record['archive']=relative
    record['custody']='manifests/REMOTE_ARCHIVE.json'
    record['local_payload_evicted']=True
    atomic_json(manifest,record)
    return deleted_bytes

def _drop_from_active_tree(root,paths,deadline):
    tracked=_command(root,['git','ls-files','-z','--','goal_conditioned_decision_data'],deadline).decode().split('\0')
    candidates={'goal_conditioned_decision_data/'+path for path in paths}
    selected=sorted(candidates.intersection(tracked))
    for relative in selected:
        path=relative.removeprefix('goal_conditioned_decision_data/')
        if archive._path(root,path).exists() or not archive.deletion_custody(root,path):
            raise RuntimeError('active-tree removal lacks verified absent-payload custody')
    if selected:
        archive._check_stop(root)
        _command(root,['git','rm','--cached','--sparse','--',*selected],deadline)
    return selected

def _deletion_gate(root,expected_pause,deadline):
    archive._check_stop(root)
    pause=root/'state/PAUSED.json'
    if not pause.exists() or pause.read_bytes()!=expected_pause:
        raise RuntimeError('pause changed during archive operation; deletion refused')
    if not current_independent_review(deadline=deadline):
        raise RuntimeError('review changed during archive operation; deletion refused')

def run(root=ROOT,config=None,*,lock_fd=None,automatic=True,transport=None,compact=True):
    deadline=time.monotonic()+SECONDS
    root=archive._root(root)
    config=config or json.loads((root/'configs/production.json').read_text())
    if config.get('archive_enabled') is not True or config.get('archive_authorization')!=archive.AUTHORIZATION:
        return None
    trigger=config.get('archive_trigger_free_bytes')
    if type(trigger) is not int or trigger!=10*1024**3 or config.get('min_free_bytes')!=trigger:
        raise RuntimeError('archive resource policy changed; review required')
    if (root/'state/STOP').exists():return None
    paused=root/'state/PAUSED.json'
    original=paused.read_bytes() if paused.exists() else None
    reason=json.loads(original).get('reason') if original else None
    if automatic and (reason is not None and reason not in DISK_REASONS|{CLEANUP}):return None
    if not automatic and reason not in DISK_REASONS|{CLEANUP,MIGRATION,None}:
        raise RuntimeError('manual archive cannot override another pause')
    if automatic and shutil.disk_usage(root).free>=trigger and not cleanup_pending(root):return None
    if lock_fd is None:
        path=archive._path(root,'state/worker.lock')
        path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:return run(root,config,lock_fd=fd,automatic=automatic,transport=transport,compact=compact)
        finally:os.close(fd)
    with archive._lock(root,lock_fd):
        archive._check_stop(root)
        original=paused.read_bytes() if paused.exists() else None
        reason=json.loads(original).get('reason') if original else None
        if automatic and reason is not None and reason not in DISK_REASONS|{CLEANUP}:return None
        if not automatic and reason not in DISK_REASONS|{CLEANUP,MIGRATION,None}:
            raise RuntimeError('manual archive cannot override another pause')
        if not current_independent_review(deadline=deadline):raise RuntimeError('independent review required before archival')
        if _command(root,['git','branch','--show-current'],deadline).decode().strip()!=BRANCH:
            raise RuntimeError('archive branch guard failed')
        if original is None:
            atomic_json(paused,{'reason':'archive maintenance in progress','utc_epoch':time.time()})
        expected_pause=paused.read_bytes()
        try:
            index=archive.load_index(root)
            present=_local_payloads(root)
            cold_present=[path for path in present if path in index['files']]
            pending=_pending_receipts(root,index)
            new=[path for path in present if path not in index['files']]
            receipt=None;eviction={'deleted':[],'deleted_bytes':0}
            remote=transport
            if pending or cold_present:
                # Finish a verified interrupted batch before starting a fresh snapshot.
                identifier=pending[0] if pending else index['files'][cold_present[0]]['receipt_id']
                receipt=json.loads(archive.read_text(root,index['receipts'][identifier]['path']))
                if pending:
                    _deletion_gate(root,expected_pause,deadline)
                    archive._recover_pending(root,receipt,'state/archive_evacuations/'+identifier+'.json')
                _backup(root,deadline)  # A prior custody push may have failed; never assume it survived.
            elif new or (root/'state/development_archive.tar.gz').exists():
                development=_development_copy(root)
                runtime=snapshot_state.snapshot(root)
                priority=[runtime['payload']]
                if development:priority.append(development)
                selected=_select_payloads(root,index,priority)
                _backup(root,deadline)  # Commit and confirm every new payload before planning.
                catalog=archive.plan_archive(root,selected,lock_fd=lock_fd)
                commit=_backup(root,deadline)  # Catalog and exact payload tree on GitHub.
                remote=remote or archive.GitHubTransport(root,seconds=min(180,deadline-time.monotonic()))
                _publish_snapshot(root,commit,deadline,remote)
                receipt=archive.verify_archive(root,catalog,commit,transport=remote,lock_fd=lock_fd)
                _backup(root,deadline)  # Persist custody/index remotely before local eviction.
            if receipt:
                remote=remote or archive.GitHubTransport(root,seconds=min(180,deadline-time.monotonic()))
                _publish_render_controls(root,deadline,remote,lock_fd)
                # Qualification has its own bounded transport. Destructive
                # proof gets a fresh provider within the original cycle limit.
                if transport is None:
                    remote=archive.GitHubTransport(root,seconds=min(180,deadline-time.monotonic()))
                _require_render_controls(root,receipt,deadline)
                _deletion_gate(root,expected_pause,deadline)
                eviction=archive.evict_verified(root,receipt,transport=remote,lock_fd=lock_fd,deletion_guard=lambda:_deletion_gate(root,expected_pause,deadline))
            index=archive.load_index(root)
            absent=[path for path in index['files'] if not archive._path(root,path).exists()]
            removed=_drop_from_active_tree(root,absent,deadline)
            _deletion_gate(root,expected_pause,deadline)
            development_removed=_remove_development_original(root,remote,deletion_guard=lambda:_deletion_gate(root,expected_pause,deadline))
            pending_status=_command(root,['git','status','--porcelain','-z'],deadline)
            confirmation=json.loads((root/'state/backup.json').read_text()) if (root/'state/backup.json').exists() else {}
            if receipt or removed or development_removed or pending_status or confirmation.get('status')!='confirmed':
                _backup(root,deadline)
            # Compaction only after every local indexed payload is absent and active-tree removals are pushed.
            compact_result=None
            if compact and index['files'] and not any(archive._path(root,path).exists() for path in index['files']):
                compact_result=compact_git.compact(root,lock_fd=lock_fd,transport=remote,seconds=min(180,deadline-time.monotonic()))
            _deletion_gate(root,expected_pause,deadline)
            result={'status':'archived' if receipt or removed or development_removed else 'no_local_payloads',
                    'deleted_files':len(eviction['deleted']),'deleted_bytes':eviction['deleted_bytes']+development_removed,
                    'active_tree_removed':len(removed),'remaining_local_payloads':len(_local_payloads(root)),
                    'archive_ref':receipt['archive_ref'] if receipt else None,'git_cleanup':compact_result,
                    'free_bytes':shutil.disk_usage(root).free,'utc_epoch':time.time()}
            if compact_result and compact_result.get('status')=='compacted_cleanup_pending':
                result['status']='archive_cleanup_pending'
                if original is None or reason==CLEANUP:
                    atomic_json(paused,{'reason':CLEANUP,'utc_epoch':time.time()})
                atomic_json(root/'state/archive_cycle.json',result)
                return result
            atomic_json(root/'state/archive_cycle.json',result)
            if original is None or reason==CLEANUP:
                archive._check_stop(root)
                if paused.read_bytes()!=expected_pause:
                    raise RuntimeError('pause changed during archive maintenance')
                paused.unlink()
            return result
        except BaseException:
            # Preserve original disk/migration cause. A new maintenance failure is binding.
            if original is None and paused.exists() and paused.read_bytes()==expected_pause:
                atomic_json(paused,{'reason':'archive operation incomplete; reviewed recovery required','utc_epoch':time.time()})
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual',action='store_true',help='explicit initial migration, preserves pause')
    parser.add_argument('--defer-git-cleanup',action='store_true',help='preserve proposal until reviewed promotion')
    args=parser.parse_args()
    if args.defer_git_cleanup and not args.manual:parser.error('--defer-git-cleanup requires --manual')
    result=run(automatic=not args.manual,compact=not args.defer_git_cleanup)
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
