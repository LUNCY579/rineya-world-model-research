#!/usr/bin/env python3
"""Create an immutable, bounded runtime backup; never evict live controls."""
import argparse
import gzip
import hashlib
import io
import json
import re
import sqlite3
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.pipelines.common import atomic_json,immutable_bytes,file_hash

JSON_CONTROLS=('checkpoint.json','metrics.json','heartbeat.json','backup.json','PAUSED.json',
               'runner.json','proposal_retirement.json')
MAX_BYTES=8*1024*1024
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})')

def snapshot(root=ROOT):
    """Caller holds worker.lock; SQLite backup additionally ensures one DB view."""
    root=Path(root).resolve();state=root/'state'
    if state.is_symlink() or state.resolve()!=state:
        raise ValueError('unsafe runtime state directory')
    members={}
    for name in JSON_CONTROLS:
        path=state/name
        if path.exists():
            if path.is_symlink() or path.resolve()!=path or path.stat().st_nlink!=1:
                raise ValueError('unsafe runtime control snapshot path')
            if path.stat().st_size>MAX_BYTES-sum(map(len,members.values())):
                raise ValueError('runtime control exceeds snapshot input budget')
            raw=path.read_bytes();json.loads(raw)
            members['state/'+name]=raw
    queue=state/'queue.sqlite'
    if queue.exists():
        if queue.is_symlink() or queue.resolve()!=queue or queue.stat().st_nlink!=1:
            raise ValueError('unsafe runtime database path')
        source=sqlite3.connect(queue.as_uri()+'?mode=ro',uri=True,timeout=5)
        try:
            page_size=source.execute('PRAGMA page_size').fetchone()[0]
            page_count=source.execute('PRAGMA page_count').fetchone()[0]
            remaining=MAX_BYTES-sum(map(len,members.values()))
            if page_size*page_count>remaining:
                raise ValueError('runtime database exceeds snapshot input budget')
            with tempfile.TemporaryDirectory(prefix='gcdl-snapshot-',dir=state) as temporary:
                destination=Path(temporary)/'queue.sqlite'
                target=sqlite3.connect(destination)
                deadline=time.monotonic()+10
                def progress(status,remaining,total):
                    if time.monotonic()>deadline:raise RuntimeError('runtime snapshot database deadline exceeded')
                    if total*page_size>MAX_BYTES-sum(map(len,members.values())):
                        raise ValueError('runtime database grew beyond snapshot input budget')
                try:
                    source.backup(target,pages=64,progress=progress,sleep=0.01)
                    if target.execute('PRAGMA journal_mode=DELETE').fetchone()[0]!='delete':
                        raise ValueError('runtime snapshot must be standalone without WAL sidecars')
                    if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                        raise ValueError('runtime snapshot database integrity failure')
                finally:target.close()
                if destination.stat().st_size>remaining:
                    raise ValueError('runtime backup exceeds snapshot input budget')
                members['state/queue.sqlite']=destination.read_bytes()
        finally:source.close()
    if not members:raise ValueError('no runtime controls to snapshot')
    if sum(map(len,members.values()))>MAX_BYTES:
        raise ValueError('runtime snapshot exceeds bounded input budget')
    if any(SECRET.search(raw) for raw in members.values()):
        raise ValueError('credential-shaped content in runtime snapshot; refusing publication')
    contents={name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for name,raw in members.items()}
    identity=hashlib.sha256(json.dumps(contents,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name,raw in sorted(members.items()):
                item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o600;item.mtime=0
                archive.addfile(item,io.BytesIO(raw))
    raw=buffer.getvalue()
    if len(raw)>MAX_BYTES:raise ValueError('runtime snapshot archive exceeds budget')
    relative='data/backups/runtime-'+identity+'.tar.gz'
    path=root/relative
    if path.resolve()!=path or path.is_symlink() or (path.exists() and path.stat().st_nlink!=1):
        raise ValueError('unsafe runtime archive destination')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('runtime archive immutable conflict')
    else:immutable_bytes(path,raw)
    receipt={'version':'runtime-snapshot-1','members':contents,'payload':relative,
             'sha256':file_hash(path),'bytes':path.stat().st_size,
             'policy':'consistent SQLite backup; live controls remain minimal and are not eviction candidates'}
    manifest=root/'manifests'/('runtime-snapshot-'+identity+'.json')
    if manifest.resolve()!=manifest or manifest.is_symlink() or (manifest.exists() and manifest.stat().st_nlink!=1):
        raise ValueError('unsafe runtime snapshot manifest destination')
    if manifest.exists():
        if json.loads(manifest.read_text())!=receipt:raise ValueError('runtime snapshot manifest conflict')
    else:atomic_json(manifest,receipt)
    return receipt

def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    with (ROOT/'state/worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=snapshot()
    print(json.dumps(receipt,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
