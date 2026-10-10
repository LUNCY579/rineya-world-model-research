#!/usr/bin/env python3
"""Authorized generic stdlib-only custody collection; never imports a verifier/helper/test."""
from __future__ import annotations
import argparse, ast, datetime, hashlib, importlib, json, os, shutil, sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
PRELOAD=('argparse','ast','base64','concurrent.futures','concurrent.futures.thread','copy','datetime',
         'difflib','gzip','hashlib','importlib.util','io','json','os','pathlib','re','shutil',
         'subprocess','tempfile','threading','time','types','unittest')

def digest(p):
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as s:
        while b:=s.read(65536):h.update(b)
    after=p.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Runtime changed while hashing')
    return {'size':before.st_size,'sha256':h.hexdigest()}

def relative(value):
    if type(value) is not str or not value or any(c in value for c in ('\\','\0','\n','\r')):raise ValueError('Literal factory-relative path required')
    p=Path(value)
    if p.is_absolute() or p.as_posix()!=value or any(x in ('.','..','.git') for x in p.parts):raise ValueError('Factory path alias rejected')
    target=FACTORY/p
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Factory symlink rejected')
    return target

def check(row):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Closed typed collector binding required')
    p=relative(row['path'])
    if not p.is_file() or digest(p)!={k:row[k] for k in ('size','sha256')}:raise ValueError('Frozen collector source/config bytes differ')
    return p

def binding(p):return {'path':p.relative_to(FACTORY).as_posix(),**digest(p)}
def exclusive(p,value):
    with p.open('x') as s:json.dump(value,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True,type=Path);a.add_argument('--expected-config-sha256',required=True);a.add_argument('--preexecution',required=True,type=Path);a.add_argument('--expected-preexecution-sha256',required=True);a.add_argument('--expected-source-sha256',required=True);args=a.parse_args()
    if digest(Path(__file__))['sha256']!=args.expected_source_sha256 or digest(args.config)['sha256']!=args.expected_config_sha256 or digest(args.preexecution)['sha256']!=args.expected_preexecution_sha256:raise ValueError('External collector source/config/preexecution hash differs')
    cfg=json.loads(args.config.read_bytes());pre=json.loads(args.preexecution.read_bytes())
    keys={'schema','collector_source','guard_source','source_code_bindings','archive_catalogs','requested_python','max_new_raw_bytes','output','failure','raw_archive_prefix','actor_execution','grants_data_approval'}
    if set(cfg)!=keys or cfg['schema']!='next653-stdlib-only-collector-config-v2' or cfg['actor_execution']!='/root/backup_manager' or cfg['grants_data_approval'] is not False or cfg['requested_python']!=sys.executable or type(cfg['max_new_raw_bytes']) is not int or not 0<=cfg['max_new_raw_bytes']<=16*1024*1024:raise ValueError('Closed authorized stdlib collector configuration required')
    if pre.get('schema')!='next653-stdlib-collector-preexecution-v2' or pre.get('authorizer_execution')!='/root' or pre.get('generic_stdlib_collection_only') is not True or pre.get('helper_imports_tests_inventory_git_network_allowed') is not False or pre.get('collector_source')!=cfg['collector_source'] or pre.get('collector_config')!=binding(args.config):raise ValueError('Actual bounded collector preexecution differs')
    check(cfg['collector_source']);check(cfg['guard_source']);check(pre['raw_collector_source'])
    if pre['raw_collector_source']['sha256']!=cfg['collector_source']['sha256']:raise ValueError('Collector raw source archive differs')
    sources=cfg['source_code_bindings']
    if type(sources) is not list or not sources or sources!=sorted(sources,key=lambda r:r['path']) or len({r['path'] for r in sources})!=len(sources):raise ValueError('Full sorted source-only closure required')
    for r in sources:check(r)
    catalog={}
    for item in cfg['archive_catalogs']:
        if set(item)!={'binding','format','base'} or item['format'] not in ('producer_prepared','producer_runtime'):raise ValueError('Exact inherited archive catalog type required')
        value=json.loads(check(item['binding']).read_bytes());base=item['base']
        for r in value['bindings']:
            archive=base+'/'+r['raw_archive'];catalog.setdefault((r['bytes'],r['sha256']),[]).append(archive)
            if item['format']=='producer_runtime':catalog.setdefault((r['bytes'],r['sha256']),[]).append(base+'/'+r['runtime_archive'])
    prefix=cfg['raw_archive_prefix']
    if prefix!=str((HERE/'runtime_raw_archives').relative_to(FACTORY)):raise ValueError('Only fresh own runtime raw archives allowed')
    output,failure=(relative(cfg[k]) for k in ('output','failure'))
    if output.exists() or failure.exists() or output.parent!=HERE or failure.parent!=HERE:raise ValueError('Fresh own exclusive collector output aliases required')
    created=0
    try:
        for name in PRELOAD:importlib.import_module(name)
        if any(n.startswith('next653_') for n in sys.modules):raise ValueError('No helper or verifier module import permitted during collection')
        current={}
        for name,module in sorted(sys.modules.copy().items()):
            if module is None:continue
            for kind,key in (('source','__file__'),('cache','__cached__')):
                value=getattr(module,key,None)
                if type(value) is not str:continue
                p=Path(value)
                if not p.is_file():
                    if kind=='cache':continue
                    raise ValueError('Loaded runtime source is absent')
                p=p.resolve()
                if p.is_relative_to(FACTORY):
                    if name!='__main__' or p!=Path(__file__).resolve():raise ValueError('Factory helper import prohibited')
                    continue
                row=current.setdefault(str(p),{'module_names':set(),'kinds':set()});row['module_names'].add(name);row['kinds'].add(kind)
        executable=Path(sys.executable).resolve();record=current.setdefault(str(executable),{'module_names':set(),'kinds':set()});record['module_names'].add('<resolved_python>');record['kinds'].add('executable')
        # Inspect cache paths statically; never import any source under the factory.
        for source in sources:
            cache=Path(importlib.util.cache_from_source(str(relative(source['path']))))
            if cache.is_file():
                row=current.setdefault(str(cache.resolve()),{'module_names':set(),'kinds':set()})
                row['module_names'].add('<static_source_cache:'+source['path']+'>');row['kinds'].add('factory_source_cache')
        rows=[]
        for path,meta in sorted(current.items()):
            p=Path(path);proof=digest(p);raw=None
            for alias in sorted(set(catalog.get((proof['size'],proof['sha256']),[]))):
                q=relative(alias)
                if q.is_file() and digest(q)==proof:raw=binding(q);break
            if raw is None:
                if proof['size']>25*1024*1024 or created+proof['size']>cfg['max_new_raw_bytes'] or shutil.disk_usage(HERE).free<proof['size']+512*1024*1024:raise ValueError('Bounded raw runtime archive storage unavailable')
                q=relative(prefix+'/'+proof['sha256']+'.source');q.parent.mkdir(exist_ok=True)
                if q.exists():
                    if digest(q)!=proof:raise ValueError('Existing raw runtime archive differs')
                else:
                    with p.open('rb') as a,q.open('xb') as b:
                        while chunk:=a.read(65536):b.write(chunk)
                        b.flush();os.fsync(b.fileno())
                    created+=proof['size']
                if digest(q)!=proof or digest(p)!=proof:raise ValueError('Actual current/raw runtime bytes differ after archive')
                raw=binding(q)
            rows.append({'current':{'path':path,**proof},'raw_archive':raw,'module_names':sorted(meta['module_names']),'kinds':sorted(meta['kinds'])})
        for r in sources:check(r)
        for row in rows:
            if digest(Path(row['current']['path']))!={k:row['current'][k] for k in ('size','sha256')}:raise ValueError('Runtime changed after collection')
            check(row['raw_archive'])
        receipt={'schema':'next653-operative-python-runtime-v2','status':'COLLECTED_STDLIB_RUNTIME_NO_HELPER_IMPORT','collector_source':cfg['collector_source'],'collector_config':binding(args.config),'collector_preexecution':binding(args.preexecution),'source_code_bindings':sources,'requested_python':sys.executable,'resolved_python':{'path':str(executable),**digest(executable)},'python_version':sys.version,'python_cache_tag':sys.implementation.cache_tag,'preload_modules':list(PRELOAD),'runtime_bindings':rows,'runtime_archive_bytes_created_this_collection':created,'actual_collector_execution':'/root/backup_manager','helper_imports_tests_inventory_git_network':0,'grants_data_approval':False}
        exclusive(output,receipt);print(json.dumps({'status':receipt['status'],'output':binding(output),'runtime_paths':len(rows),'new_raw_bytes':created}),flush=True)
    except BaseException as error:
        exclusive(failure,{'status':'FAILED_STDLIB_COLLECTION_ACTUAL_BYTES_PRESERVED','error_type':type(error).__name__,'error':str(error),'collector_source':cfg['collector_source'],'config':binding(args.config),'preexecution':binding(args.preexecution),'created_raw_bytes':created,'helper_imports_tests_inventory_git_network':0,'grants_data_approval':False});raise

if __name__=='__main__':main()
