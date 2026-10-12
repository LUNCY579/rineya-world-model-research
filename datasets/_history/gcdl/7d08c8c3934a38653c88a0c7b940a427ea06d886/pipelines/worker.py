"""One bounded, resumable production cycle. launchd invokes this every five minutes."""
import argparse
import collections
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .common import ROOT, REPO, atomic_json, canonical, digest, file_hash, immutable_bytes, source_hashes
from . import remote_archive


def ensure_job_profiles(db):
    db.execute('CREATE TABLE IF NOT EXISTS jobs (seed INTEGER, split TEXT, status TEXT, accepted INTEGER DEFAULT 0, quarantined INTEGER DEFAULT 0, report TEXT, PRIMARY KEY(seed,split))')
    if 'profile' not in {row[1] for row in db.execute('PRAGMA table_info(jobs)')}:
        db.execute('ALTER TABLE jobs ADD COLUMN profile TEXT NOT NULL DEFAULT "v1"')
    db.commit()


def connection():
    (ROOT / 'state').mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'state/queue.sqlite', timeout=5)
    db.execute('PRAGMA journal_mode=WAL')
    ensure_job_profiles(db)
    db.execute('CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, semantic TEXT, parent TEXT, template TEXT, split TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS events (at REAL, kind TEXT, detail TEXT)')
    db.commit()
    return db


def guard(config, db):
    from ..generators.profiles import resolve_profile
    resolve_profile(config.get('active_generation_profile', 'v1'))
    from .rendered import settings
    settings(config)
    if not 1<=config['groups_per_cycle']<=32 or not 10<=config['cycle_timeout_seconds']<=600:
        raise RuntimeError('invalid bounded cycle configuration')
    if not 1<=config['max_consecutive_failures']<=10:
        raise RuntimeError('invalid failure ceiling')
    if shutil.disk_usage(ROOT).free < config['min_free_bytes']:
        raise RuntimeError('resource gate: fewer than 10 GiB free')
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if size >= config['max_programme_bytes']:
        raise RuntimeError('resource gate: programme storage ceiling reached')
    count = db.execute('SELECT COALESCE(SUM(accepted),0) FROM jobs').fetchone()[0]
    if count >= config['max_records']:
        raise RuntimeError('coverage gate: bounded record budget reached; human-reviewed generator improvement required')
    if config['concurrency'] != 1:
        raise RuntimeError('only single-worker production supported')
    if (ROOT / 'state/STOP').exists():
        raise RuntimeError('operator stop marker present')
    if (ROOT / 'state/PAUSED.json').exists():
        raise RuntimeError('pause gate present; use explicit resume after resolving cause')


def publication_guard(config,additional_bytes):
    current=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if current+additional_bytes>config['max_programme_bytes']:
        raise RuntimeError('resource gate: publication would exceed storage ceiling')
    if shutil.disk_usage(ROOT).free-additional_bytes<config['min_free_bytes']:
        raise RuntimeError('resource gate: publication would leave fewer than 10 GiB free')


def event(db, kind, detail):
    db.execute('INSERT INTO events VALUES (?,?,?)', (time.time(), kind, canonical(detail)))
    db.commit()


def heartbeat(status, **kwargs):
    atomic_json(ROOT / 'state/heartbeat.json', {'status': status, 'utc_epoch': time.time(), **kwargs})


def pause(db, reason):
    atomic_json(ROOT / 'state/PAUSED.json', {'reason': reason, 'utc_epoch': time.time()})
    heartbeat('paused', reason=reason)
    event(db, 'paused', {'reason': reason})


def batch_profile(manifest):
    from ..generators.profiles import resolve_profile
    name=manifest.get('generation_profile','v1')
    resolved=resolve_profile(name)
    declared=manifest.get('generation_profile_evidence')
    version=manifest.get('version','gcdl-manifest-1')
    if version not in ('gcdl-manifest-1','gcdl-manifest-2'):
        raise ValueError('unsupported canonical batch manifest version')
    if version=='gcdl-manifest-2' and ('generation_profile' not in manifest or declared is None):
        raise ValueError('versioned batch missing explicit generation profile commitment')
    expected={k:resolved[k] for k in ('id','generator_version','environment_version','config_version','config_hash','config_file_sha256')}
    if (name=='v2' and declared is None) or (declared is not None and declared!=expected):
        raise ValueError('stored generation profile commitment mismatch')
    return resolved


def recover_inventory(db):
    """Files become authoritative only after immutable shard + manifest are complete."""
    ensure_job_profiles(db)
    from ..auditors.integrity import semantic_fingerprint
    from .rendered import _path
    root=ROOT.resolve()
    recovered=set()
    committed_shards=set()
    for relative in remote_archive.iter_paths(root,'manifests/batch-*.json'):
        path=_path(root,relative,('manifests',))
        manifest = json.loads(remote_archive.read_text(root,relative))
        if not {'seed','split','protected_index','accepted','quarantined','audit','shard','sha256'}<=set(manifest):
            raise ValueError('missing canonical inventory receipt field')
        seed, split = manifest['seed'], manifest['split']
        if type(seed) is not int or seed<0 or split not in ('train','dev','ood','sealed'):
            raise ValueError('invalid canonical inventory seed/split')
        if path.parent!=root/'manifests' or path.name!=f'batch-{split}-{seed}.json':
            raise ValueError('canonical inventory receipt filename mismatch')
        resolved=batch_profile(manifest)
        profile=resolved['id']
        previous=db.execute('SELECT profile FROM jobs WHERE seed=? AND split=?',(seed,split)).fetchone()
        if previous and previous[0]!=profile:
            raise ValueError('recovery cannot reinterpret job generation profile')
        folder='sealed' if split=='sealed' else 'accepted'
        shard=_path(root,manifest['shard'],('data',folder))
        if shard.parent!=root/'data'/folder:
            raise ValueError('canonical shard must be directly in its split directory')
        committed_shards.add(manifest['shard'])
        info=remote_archive.payload_info(root,manifest['shard'])
        if info['sha256'] != manifest['sha256']:
            raise ValueError('stored shard digest mismatch: ' + str(shard))
        if 'bytes' in manifest and info['bytes']!=manifest['bytes']:
            raise ValueError('stored shard byte count mismatch: ' + str(shard))
        if split=='sealed' and info['archived']:
            freeze=json.loads((root/'manifests/SEALED_FREEZE.json').read_text())
            if freeze['shards'].get(manifest['shard'])!=info['sha256']:
                raise ValueError('archived sealed shard disagrees with local freeze')
        # Recovery reads only manifest indices; never opens sealed target labels.
        if len(manifest['protected_index'])!=manifest['accepted']:
            raise ValueError('manifest protected index count mismatch')
        for index in manifest['protected_index']:
            if len(index)!=5 or index[-1]!=split:
                raise ValueError('manifest split/index mismatch')
            if index[3]!=resolved['config']['split_templates'][split]['template']:
                raise ValueError('protected template disagrees with generation profile')
            existing=db.execute('SELECT id,semantic,parent,template,split FROM seen WHERE id=?',(index[0],)).fetchone()
            if existing and tuple(index)!=existing:
                raise ValueError('recovery database/index conflict')
            overlaps=db.execute('SELECT split FROM seen WHERE semantic=? OR parent=? OR template=?',tuple(index[1:4])).fetchall()
            if any(found[0]!=split for found in overlaps):
                raise ValueError('recovery protected split overlap')
            db.execute('INSERT OR IGNORE INTO seen VALUES (?,?,?,?,?)', tuple(index))
        db.execute('INSERT OR REPLACE INTO jobs (seed,split,status,accepted,quarantined,report,profile) VALUES (?,?,?,?,?,?,?)', (seed,split,'done',manifest['accepted'],manifest['quarantined'],canonical(manifest['audit']),profile))
        recovered.add((seed,split))
    for pattern in ('data/accepted/*.jsonl','data/sealed/*.jsonl'):
        for relative in remote_archive.iter_paths(root,pattern):
            if remote_archive.is_archived(root,relative) and relative not in committed_shards:
                raise ValueError('archived canonical shard missing its immutable batch receipt')
    if any((seed,split) not in recovered for seed,split in db.execute('SELECT seed,split FROM jobs WHERE status="done"')):
        raise ValueError('completed canonical job missing its immutable batch receipt')
    db.commit()


def process(seed, split, config, db):
    ensure_job_profiles(db)
    before_source=source_hashes()
    from ..generators.profiles import resolve_profile
    job=db.execute('SELECT profile FROM jobs WHERE seed=? AND split=?',(seed,split)).fetchone()
    profile=resolve_profile(job[0] if job else config.get('active_generation_profile','v1'))
    from .rendered import _path
    root=ROOT.resolve()
    existing=root/'manifests'/f'batch-{split}-{seed}.json'
    relative=str(existing.relative_to(root))
    if remote_archive.exists(root,relative) or existing.is_symlink():
        existing=_path(root,relative,('manifests',))
        stored=batch_profile(json.loads(remote_archive.read_text(root,relative)))
        if stored['id']!=profile['id']:
            raise ValueError('existing batch profile cannot be reinterpreted')
        raise ValueError('completed immutable batch already exists; recover inventory before replay')
    if profile['id']=='v1':
        from ..generators.build import generate_group
    else:
        from ..generators.version2 import generate_group
    from ..auditors.integrity import audit_group, semantic_fingerprint
    started = time.monotonic()
    candidates = generate_group(seed, config=profile['config'], split=split)
    rerun = generate_group(seed, config=profile['config'], split=split)
    if canonical(candidates) != canonical(rerun):
        raise ValueError('reproducibility mismatch')
    raw_candidates = ('\n'.join(canonical(r) for r in candidates) + '\n').encode()
    candidate_hash = __import__('hashlib').sha256(raw_candidates).hexdigest()
    publication_guard(config,len(raw_candidates))
    immutable_bytes(ROOT / 'data/candidates' / f'{split}-{seed}.jsonl', raw_candidates)
    accepted, rejected, report = audit_group(candidates, db)
    from ..generators.profiles import resolve_record_profile
    if any(resolve_record_profile(row['provenance'])['id']!=profile['id'] for row in accepted):
        raise RuntimeError('generator adapter emitted a different generation profile')
    if profile['id']=='v2':
        group_error=None
        if not rejected:
            from ..auditors.integrity import validate_v2_group
            try:
                validate_v2_group(accepted,require_full_curriculum=True)
            except Exception as error:
                group_error=type(error).__name__+': '+str(error)
        if rejected or group_error:
            reason='v2 parent atomicity: '+(group_error or 'a sibling failed authoritative acceptance')
            original_report=json.loads(canonical(report))
            moved=len(accepted)
            for row in accepted:
                row['quality_gate']={'status':'quarantined','rejection_reasons':[reason]}
            rejected.extend(accepted)
            accepted=[]
            report.update(verdict='PARTIAL_QUARANTINE',accepted=0,quarantined=len(rejected),
                          oracle_agreement_on_accepted=None,schema_pass_rate_on_accepted=None,
                          family_counts={},difficulty_counts={},perturbation_tags={},unique_task_fingerprints=0,
                          original_record_audit=original_report,atomic_parent_gate='FAIL')
            report['causes'][reason]=moved
    if split=='sealed' and (not accepted or rejected):
        immutable_bytes(ROOT/'data/quarantined'/f'failed-seal-{seed}.jsonl', ('\n'.join(canonical(r) for r in rejected)+'\n').encode())
        raise RuntimeError('sealed quality gate failed; no freeze publication permitted')
    from ..generators.build import measure_search_runtime
    runtime_evidence={r['id']:measure_search_runtime(r['input']) for r in accepted if 'search' in r['targets']}
    if time.monotonic() - started > config['cycle_timeout_seconds']:
        raise TimeoutError('batch time budget exceeded before acceptance')
    source = source_hashes()
    if source!=before_source:
        raise RuntimeError('code/config changed during batch; acceptance halted')
    for record in accepted:
        record['verification']['code_hashes'] = source
    if len(accepted) + db.execute('SELECT COALESCE(SUM(accepted),0) FROM jobs').fetchone()[0] > config['max_records']:
        raise RuntimeError('record ceiling reached before shard publication')
    raw = ('\n'.join(canonical(r) for r in accepted) + ('\n' if accepted else '')).encode()
    if len(raw) > config['max_shard_bytes']:
        raise RuntimeError('shard size exceeds configured limit')
    publication_guard(config,len(raw)+sum(len(canonical(r).encode())+1 for r in rejected)+1048576)
    shard = ROOT / ('data/sealed' if split == 'sealed' else 'data/accepted') / f'{split}-{seed}.jsonl'
    immutable_bytes(shard, raw)
    rejected_raw = ('\n'.join(canonical(r) for r in rejected) + ('\n' if rejected else '')).encode()
    if rejected:
        immutable_bytes(ROOT / 'data/quarantined' / f'{split}-{seed}.jsonl', rejected_raw)
    manifest = {'version':'gcdl-manifest-2','seed':seed,'split':split,'shard':str(shard.relative_to(ROOT)),
        'generation_profile':profile['id'],
        'generation_profile_evidence':{k:profile[k] for k in ('id','generator_version','environment_version','config_version','config_hash','config_file_sha256')},
        'sha256':file_hash(shard),'bytes':len(raw),'candidate_sha256':candidate_hash,'accepted':len(accepted),
        'quarantined':len(rejected),'audit':report,'code_hashes':source,'config':config,
        'runtime_evidence':runtime_evidence,'runtime_is_canonical_target':False,
        'protected_index':[[r['id'],semantic_fingerprint(r),r['provenance']['parent_world_id'],r['provenance']['template_group_id'],split] for r in accepted],
        'reproducibility':'PASS','protected_groups':sorted({r['provenance']['parent_world_id'] for r in accepted})}
    immutable_bytes(ROOT / 'manifests' / f'batch-{split}-{seed}.json', (canonical(manifest)+'\n').encode())
    # Candidate cache is reproducible; its seed/config/source + digest persist.
    (ROOT / 'data/candidates' / f'{split}-{seed}.jsonl').unlink(missing_ok=True)
    for row in accepted:
        p = row['provenance']
        db.execute('INSERT OR IGNORE INTO seen VALUES (?,?,?,?,?)', (row['id'],semantic_fingerprint(row),p['parent_world_id'],p['template_group_id'],split))
    db.execute('UPDATE jobs SET status=?,accepted=?,quarantined=?,report=? WHERE seed=? AND split=?', ('done',len(accepted),len(rejected),canonical(report),seed,split))
    db.commit()
    atomic_json(ROOT / 'state/checkpoint.json', {'seed':seed,'split':split,'generation_profile':profile['id'],'manifest':str((ROOT/'manifests'/f'batch-{split}-{seed}.json').relative_to(ROOT)), 'utc_epoch':time.time()})
    event(db,'batch',{'seed':seed,'split':split,'accepted':len(accepted),'quarantined':len(rejected),'elapsed_seconds':time.monotonic()-started})
    return manifest


def summary(db):
    by_split = {r[0]: {'groups':r[1],'accepted':r[2],'quarantined':r[3]} for r in db.execute('SELECT split,COUNT(*),SUM(accepted),SUM(quarantined) FROM jobs WHERE status="done" GROUP BY split')}
    events = db.execute('SELECT SUM(json_extract(detail,"$.elapsed_seconds")) FROM events WHERE kind="batch"').fetchone()[0] or 0
    accepted = sum(r['accepted'] for r in by_split.values())
    families=collections.Counter();difficulty=collections.Counter();causes=collections.Counter();tags=collections.Counter()
    profiles={r[0]:{'groups':r[1],'accepted':r[2]} for r in db.execute('SELECT profile,COUNT(*),SUM(accepted) FROM jobs WHERE status="done" GROUP BY profile')}
    for (raw,) in db.execute('SELECT report FROM jobs WHERE status="done"'):
        report=json.loads(raw);families.update(report.get('family_counts',{}));difficulty.update(report.get('difficulty_counts',{}));causes.update(report.get('causes',{}));tags.update(report.get('perturbation_tags',{}))
    result = {'version':'production-stats-1','splits':by_split,'accepted':accepted,'quarantined':sum(r['quarantined'] for r in by_split.values()),'active_generation_seconds':events,
        'accepted_per_active_hour':accepted*3600/events if events else None,'queue_size':db.execute('SELECT COUNT(*) FROM jobs WHERE status!="done"').fetchone()[0],
        'disk_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),'free_bytes':shutil.disk_usage(ROOT).free,
        'quality_classification':'experimental synthetic pilot; no model generalization demonstrated',
        'family_counts':dict(families),'difficulty_counts':dict(difficulty),'quarantine_causes':dict(causes),'perturbation_tags':dict(tags),'generation_profiles':profiles,
        'oracle_agreement_on_accepted':1.0 if accepted else None,'schema_pass_rate_on_accepted':1.0 if accepted else None,
        'audit_seconds_per_accepted_record':events/accepted if accepted else None,
        'last_healthy_checkpoint':json.loads((ROOT/'state/checkpoint.json').read_text()) if (ROOT/'state/checkpoint.json').exists() else None}
    from .rendered import inventory, ensure_queue
    ensure_queue(db)
    result['rendered_datasets'] = inventory(ROOT)
    result['rendered_datasets']['pending_source_groups'] = db.execute('SELECT COUNT(*) FROM rendered_jobs WHERE status="pending"').fetchone()[0]
    atomic_json(ROOT/'state/metrics.json',result)
    return result


def backup_if_due(config, db):
    p = ROOT/'state/backup.json'
    last=json.loads(p.read_text()) if p.exists() else {}
    if time.time()-last.get('last_success_epoch',0)<config['backup_interval_seconds']:
        return
    r=subprocess.run([sys.executable,str(ROOT/'scripts/backup.py')],capture_output=True,text=True,timeout=150)
    if r.returncode:
        event(db,'backup_failure',{'exit_code':r.returncode,'detail':r.stderr[-2000:]})
        raise RuntimeError('remote backup failed; checkpoint retained; bounded retry via resume')


def _run_archive_cycle(config,lock_fd):
    from ..scripts import archive_cycle
    return archive_cycle.run(root=ROOT,config=config,lock_fd=lock_fd,automatic=True)


def _archive_for_resource_pressure(config,*,lock_fd=None):
    """Policy-authorized eviction never approves code or relaxes publication."""
    if config.get('archive_enabled') is not True:
        return None
    if (ROOT/'state/STOP').exists():
        return None
    from ..scripts.archive_cycle import cleanup_pending
    if shutil.disk_usage(ROOT).free>=config['min_free_bytes'] and not cleanup_pending(ROOT):
        return None
    # The composed coordinator checks review under this live lock and at each deletion.
    if (ROOT/'state/STOP').exists():
        return None
    return _run_archive_cycle(config,lock_fd)


def _disk_pause(reason):
    return reason in ('resource gate: fewer than 10 GiB free',
                      'resource gate: publication would leave fewer than 10 GiB free',
                      'rendered export would leave fewer than 10 GiB free')


def run(args):
    if (ROOT/'state/STOP').exists():
        return 0
    config=json.loads((ROOT/'configs/production.json').read_text())
    (ROOT/'state').mkdir(exist_ok=True)
    # A launchd interval can fire while paused. Preserve the original cause
    # without logging repeated errors or expanding a resource-limited database.
    if (ROOT/'state/STOP').exists():
        return 0
    marker=ROOT/'state/PAUSED.json'
    if marker.exists() and (config.get('archive_enabled') is not True
                            or not (_disk_pause(json.loads(marker.read_text()).get('reason')) or json.loads(marker.read_text()).get('reason')=='archive Git cleanup pending')):
        return 0
    with (ROOT/'state/worker.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 0
        paused=ROOT/'state/PAUSED.json'
        if paused.exists():
            reason=json.loads(paused.read_text()).get('reason')
            if _disk_pause(reason) or reason=='archive Git cleanup pending':
                try: _archive_for_resource_pressure(config,lock_fd=lock.fileno())
                except Exception as error:
                    print(str(error),file=sys.stderr)
                    return 1
            return 0  # Preserve every pause marker/cause; no implicit resume.
        try: _archive_for_resource_pressure(config,lock_fd=lock.fileno())
        except Exception as error:
            print(str(error),file=sys.stderr)
            return 1
        db=connection()
        try:
            guard(config,db)
            approved=ROOT/'manifests/VALIDATED_CODE.json'
            if not args.no_backup:
                from ..auditors.release import current_independent_review
                if not approved.exists() or not current_independent_review():
                    raise RuntimeError('code/config changed since final validation; independent review required')
                if not (ROOT/'manifests/SEALED_FREEZE.json').exists():
                    raise RuntimeError('sealed evaluation freeze required before unattended operation')
            heartbeat('recovering')
            recover_inventory(db)
            from . import rendered
            rendered.recover(db, ROOT)
            if args.seal:
                if (ROOT/'manifests/SEALED_FREEZE.json').exists():
                    heartbeat('sealed_already_frozen'); return 0
                seeds=[(s,'sealed') for s in config['sealed_seeds']]
            elif getattr(args, 'render_only', False):
                seeds=[]
            else:
                existing = db.execute('SELECT MAX(seed) FROM jobs WHERE split!="sealed"').fetchone()[0]
                start=max(config['seed_start'],(existing+1) if existing is not None else 0)
                seeds=[(start+i,config['production_splits'][(start-config['seed_start']+i)%len(config['production_splits'])]) for i in range(args.groups or config['groups_per_cycle'])]
            for seed,split in seeds:
                profile='v1' if args.seal else config.get('active_generation_profile','v1')
                db.execute('INSERT OR IGNORE INTO jobs (seed,split,status,profile) VALUES (?,?,?,?)',(seed,split,'pending',profile))
            db.commit()
            jobs=[] if getattr(args, 'render_only', False) else db.execute('SELECT seed,split FROM jobs WHERE status!="done" AND '+('split="sealed"' if args.seal else 'split!="sealed"')+' ORDER BY seed').fetchall()
            started=time.monotonic()
            for seed,split in jobs:
                if time.monotonic()-started>config['cycle_timeout_seconds']*.8: break
                guard(config,db)
                if not args.no_backup and json.loads(approved.read_text())['code_hashes']!=source_hashes():
                    raise RuntimeError('reviewed source/config changed between batches')
                heartbeat('generating',seed=seed,split=split)
                process(seed,split,config,db)
                recent=db.execute('SELECT accepted,quarantined,report FROM jobs WHERE status="done" AND split!="sealed" ORDER BY seed DESC LIMIT ?', (config['max_consecutive_failures'],)).fetchall()
                if len(recent)==config['max_consecutive_failures'] and all(a==0 and q>0 for a,q,_ in recent):
                    raise RuntimeError('novelty/quality gate: repeated unproductive batches; coverage/validator improvement needed')
                if recent and any('oracle disagreement' in cause for cause in json.loads(recent[0][2]).get('causes',{})):
                    raise RuntimeError('oracle disagreement quarantined; investigation required')
            if not args.seal:
                if not args.no_backup and not current_independent_review():
                    raise RuntimeError('independent source approval changed before rendered publication')
                rendered.enqueue(db, ROOT, config)
                heartbeat('rendering')
                published=rendered.drain(db, ROOT, config, started+config['cycle_timeout_seconds']*.8)
                if published:
                    event(db, 'rendered_batch', {'receipts':published})
                if not args.no_backup and not current_independent_review():
                    raise RuntimeError('independent source approval changed after rendered publication')
            if args.seal:
                frozen = {str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'data/sealed').glob('*.jsonl'))}
                expected={f'data/sealed/sealed-{s}.jsonl' for s in config['sealed_seeds']}
                if set(frozen)!=expected or any(db.execute('SELECT accepted,quarantined FROM jobs WHERE seed=? AND split="sealed"',(s,)).fetchone() is None or db.execute('SELECT accepted,quarantined FROM jobs WHERE seed=? AND split="sealed"',(s,)).fetchone()[0]==0 or db.execute('SELECT accepted,quarantined FROM jobs WHERE seed=? AND split="sealed"',(s,)).fetchone()[1]!=0 for s in config['sealed_seeds']):
                    raise RuntimeError('seal incomplete/failed; resume seal before production')
                immutable_bytes(ROOT/'manifests/SEALED_FREEZE.json',(canonical({'version':'seal-1','shards':frozen,'seeds':config['sealed_seeds'],'code_hashes':source_hashes(),'policy':'producer never reads sealed labels; same-user filesystem access is procedural isolation'})+'\n').encode())
                os.chmod(ROOT/'data/sealed',0o700)
            metrics=summary(db)
            if not args.no_backup: backup_if_due(config,db)
            heartbeat('healthy',metrics=metrics)
            print(canonical({'status':'healthy','accepted':metrics['accepted'],'splits':metrics['splits']}))
            return 0
        except Exception as error:
            event(db,'error',{'class':type(error).__name__,'message':str(error),'diagnostics':getattr(error,'diagnostics',None),'traceback':traceback.format_exc()})
            pause(db,str(error))
            print(str(error),file=sys.stderr)
            return 1
        finally: db.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--groups',type=int)
    parser.add_argument('--seal',action='store_true')
    parser.add_argument('--render-only',action='store_true',help='bounded recovery/backfill of committed nonsealed training views')
    parser.add_argument('--no-backup',action='store_true',help='development/initial pre-backup cycles only')
    args=parser.parse_args()
    if args.groups is not None and not 1<=args.groups<=32: parser.error('groups must be 1..32')
    if args.seal and args.render_only: parser.error('--seal and --render-only are incompatible')
    return run(args)

if __name__=='__main__': raise SystemExit(main())
