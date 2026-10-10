"""Complete operative Python/source/cache/raw checks before helper imports and after a phase."""
from __future__ import annotations
import hashlib, importlib, json, os, sys
from pathlib import Path

HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
SCHEMA='next653-operative-python-runtime-v2'
PRELOAD=('argparse','ast','base64','concurrent.futures','concurrent.futures.thread','copy','datetime',
         'difflib','gzip','hashlib','importlib.util','io','json','os','pathlib','re','shutil',
         'subprocess','tempfile','threading','time','types','unittest')

def digest(path):
    before=tuple(getattr(path.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(65536):h.update(b)
    if before!=tuple(getattr(path.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('Operative runtime evolved while hashing')
    return {'size':before[2],'sha256':h.hexdigest()}

def relative(value):
    if type(value) is not str or not value or any(c in value for c in ('\\','\0','\n','\r')):raise ValueError('Literal raw archive path required')
    p=Path(value)
    if p.is_absolute() or p.as_posix()!=value or any(x in ('.','..','.git') for x in p.parts):raise ValueError('Raw archive alias rejected')
    result=FACTORY/p
    if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Raw archive symlink rejected')
    return result

def check_binding(row,absolute=False):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Typed closed operative byte binding required')
    p=Path(row['path']) if absolute else relative(row['path'])
    if absolute and (not p.is_absolute() or str(p)!=row['path'] or str(p.resolve())!=row['path']):raise ValueError('Canonical resolved runtime identity required')
    if not p.is_file() or p.is_symlink() or digest(p)!={k:row[k] for k in ('size','sha256')}:raise ValueError('Current/runtime raw bytes differ: '+row['path'])
    return p

def preload():
    for name in PRELOAD:importlib.import_module(name)

def observed():
    """External module source/cache paths only; factory module admission is separate."""
    result={}
    for name,module in sorted(sys.modules.copy().items()):
        if module is None:continue
        for kind,key in (('source','__file__'),('cache','__cached__')):
            value=getattr(module,key,None)
            if type(value) is not str:continue
            p=Path(value)
            if not p.is_file():
                if kind=='cache':continue
                raise ValueError('Loaded operative file absent')
            p=p.resolve()
            if p.is_relative_to(FACTORY):continue
            rec=result.setdefault(str(p),{'module_names':set(),'kinds':set()})
            rec['module_names'].add(name);rec['kinds'].add(kind)
    return {p:{'module_names':sorted(r['module_names']),'kinds':sorted(r['kinds'])} for p,r in sorted(result.items())}

def validate_value(value,expected_sources):
    keys={'schema','status','collector_source','collector_config','collector_preexecution','source_code_bindings',
          'requested_python','resolved_python','python_version','python_cache_tag','preload_modules','runtime_bindings',
          'runtime_archive_bytes_created_this_collection','actual_collector_execution','helper_imports_tests_inventory_git_network','grants_data_approval'}
    if type(value) is not dict or set(value)!=keys or value['schema']!=SCHEMA or value['status']!='COLLECTED_STDLIB_RUNTIME_NO_HELPER_IMPORT':raise ValueError('Exact completed runtime collection schema required')
    if value['actual_collector_execution']!='/root/backup_manager' or type(value['helper_imports_tests_inventory_git_network']) is not int or value['helper_imports_tests_inventory_git_network']!=0 or value['grants_data_approval'] is not False:raise ValueError('Collector role/scope differs')
    if value['preload_modules']!=list(PRELOAD) or value['python_version']!=sys.version or value['python_cache_tag']!=sys.implementation.cache_tag or value['requested_python']!=sys.executable:raise ValueError('Operative interpreter/preload identity differs')
    if not value['source_code_bindings'] or value['source_code_bindings']!=expected_sources:raise ValueError('Complete source collection closure substituted')
    for row in expected_sources:check_binding(row)
    for label in ('collector_source','collector_config','collector_preexecution'):check_binding(value[label])
    check_binding(value['resolved_python'],True)
    if value['resolved_python']['path']!=str(Path(sys.executable).resolve()):raise ValueError('Resolved actual Python changed')
    rows=value['runtime_bindings'];seen=set()
    if type(rows) is not list or not rows:raise ValueError('Nonempty complete operative source/cache/raw membership required')
    for row in rows:
        if type(row) is not dict or set(row)!={'current','raw_archive','module_names','kinds'}:raise ValueError('Closed runtime source/cache/raw row required')
        p=row['current']['path']
        if p in seen or type(row['module_names']) is not list or not row['module_names'] or row['module_names']!=sorted(set(row['module_names'])) or type(row['kinds']) is not list or not set(row['kinds'])<={'source','cache','executable','factory_source_cache'} or not row['kinds']:raise ValueError('Runtime duplicate/member/kind type differs')
        seen.add(p);check_binding(row['current'],True);check_binding(row['raw_archive'])
        if (row['current']['size'],row['current']['sha256'])!=(row['raw_archive']['size'],row['raw_archive']['sha256']):raise ValueError('Raw runtime archive differs from operative bytes')
    return value

def validate(cfg,context):
    if context not in ('preimport','post_phase'):raise ValueError('Exact operative runtime phase context required')
    row=cfg['inputs']['runtime_freeze'];check_binding(row);value=json.loads(relative(row['path']).read_bytes())
    expected_freeze=next((r for r in cfg['_preparation']['frozen_inputs'] if r['path']==str((HERE/'RUNTIME_SOURCE_FREEZE.json').relative_to(FACTORY))),None)
    if row!=expected_freeze:raise ValueError('Runtime input must equal the exact actual completed preparation-bound collection')
    sources=sorted([r for r in cfg['_preparation']['frozen_inputs'] if r['path'] in cfg['_preparation']['operative_source_paths']],key=lambda r:r['path'])
    validate_value(value,sources);preload();actual=observed()
    expected={r['current']['path']:{'module_names':r['module_names'],'kinds':[k for k in r['kinds'] if k!='executable']} for r in value['runtime_bindings'] if set(r['kinds'])&{'source','cache'}}
    if actual!=expected:raise ValueError('Loaded operative source/cache membership differs before import or after phase')
    # Factory module identity is restricted to exact fully checked preparation inputs.
    admitted={r['path']:r for r in cfg['_preparation']['frozen_inputs']}
    for name,module in sys.modules.copy().items():
        p=getattr(module,'__file__',None)
        if type(p) is not str:continue
        path=Path(p).absolute()
        if path.is_relative_to(FACTORY):
            rel=path.relative_to(FACTORY).as_posix()
            if rel not in admitted or (name!='__main__' and not name.startswith('next653_')):raise ValueError('Unexpected factory module imported into custody runtime')
            check_binding(admitted[rel])
            cache=getattr(module,'__cached__',None)
            if type(cache) is str and Path(cache).exists():
                current=next((r for r in value['runtime_bindings'] if r['current']['path']==str(Path(cache).resolve()) and r['kinds']==['factory_source_cache']),None)
                if current is None:raise ValueError('Unadmitted factory bytecode cache exists')
                check_binding(current['current'],True)
    recorded={r['current']['path'] for r in value['runtime_bindings'] if r['kinds']==['factory_source_cache']}
    actual_caches={str(Path(importlib.util.cache_from_source(str(relative(r['path'])))).resolve()) for r in sources if Path(importlib.util.cache_from_source(str(relative(r['path'])))).is_file()}
    if recorded!=actual_caches:raise ValueError('Static operative helper source/cache membership changed')
    # All current and raw bytes are rehashed after membership/preload checks as well.
    validate_value(value,sources)
    return {'status':'PASS_EXACT_OPERATIVE_RUNTIME_CURRENT_AND_RAW','context':context,'runtime_freeze':row,
            'external_source_cache_executable_paths':len(value['runtime_bindings']),'loaded_external_membership':actual,
            'requested_python':value['requested_python'],'resolved_python':value['resolved_python'],'grants_data_approval':False}
