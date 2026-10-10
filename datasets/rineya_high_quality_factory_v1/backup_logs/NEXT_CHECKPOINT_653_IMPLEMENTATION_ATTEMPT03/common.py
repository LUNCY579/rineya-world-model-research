"""Bounded archival primitives with local session state and a complete typed closure."""
from __future__ import annotations
import base64, concurrent.futures, datetime as dt, gzip, hashlib, importlib.util, json, os, re
import shutil, subprocess, sys, tempfile, threading
from pathlib import Path

HERE = Path(__file__).absolute().parent
FACTORY = HERE.parent.parent
ORIGINAL = FACTORY.parent/'rineya-world-model-research'
WORKTREE = FACTORY.parent/'.rineya-dataset-git-worktree'
PARENT = '653ba70f1f7035a5aed137102c14a1ff29d1cf99'
PREFIX = 'datasets/rineya_high_quality_factory_v1'
REPOSITORY = 'LUNCY579/rineya-world-model-research'
BRANCH = 'research/rineya-high-quality-dataset-v1'
CAP, CHUNK, RESERVE = 25*1024*1024, 65536, 512*1024*1024
FIXED = {
 'manager': ('backup_logs/run_provisional_archival_checkpoint_v4_attempt02.py','f0561eb5f0e2ebe16bc46f51ed1fcbb1ad74236e90fba4d9b33da166c6767715'),
 'scanner': ('backup_logs/exact_four_public_metadata_secret_scan_attempt02.py','a79963d367f47c54074714d24bc2a5f372f09bf725a4b2046006cbdb586e4b1d'),
 'gzip': ('backup_logs/verify_archival_remote_lossless_exact_public_metadata_v3.py','29e27b8d87facdc5afbc8df57c4259515977e7b65895b4134966ad153e99f3d4'),
 'prior': ('backup_logs/belief-v4-approved-release-r1_SPEC.json','6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'),
 'custody': ('backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_FINAL_JOINED_CUSTODY.json','cf621c9e6625de9ea2ff7af30437cb638e9b76964df1d342d3e003e5ebd8950e'),
 'history': ('backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_HISTORICAL_RAW_CONFIG.json','3a563a5e9b816d120c688d425fa8b634f16baa3456633090c347c7caf11ea27d'),
 'references': ('quality_reports/RGM_RENDERER_ARCHIVAL_BOUNDARY_PREFLIGHT.json','03ec34f4f778698c9dc52b3bd4624f324abc525aaca1e33417d4c6042d8168a1'),
 'metadata_config': ('backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json','2e18c2c156aaefcee75ad6ad92fe51c66abf864150924d5ac1cd26dbeed5aa05')}
ADMISSION_KEYS = {'preparation_tests','admission_contract','archive_mapping_contract','runtime_freeze'}
PHASE_KEYS = {
 'test': {'admission_contract','archive_mapping_contract','runtime_freeze'},
 'inventory': ADMISSION_KEYS | {'catalog'},
 'stage': ADMISSION_KEYS | {'spec','inventory_report','inventory_execution_config'},
 'recovery': ADMISSION_KEYS | {'spec','inventory_report','inventory_execution_config','stage','local_commit','stage_ledger','stage_execution_config'},
 'lossless': ADMISSION_KEYS | {'spec','inventory_report','inventory_execution_config','stage','local_commit','stage_ledger','stage_execution_config','full_proof','full_ledger','recovery_execution_config'},
 'join': ADMISSION_KEYS | {'spec','inventory_report','inventory_execution_config','stage','local_commit','stage_ledger','stage_execution_config','full_proof','full_ledger','recovery_execution_config','gzip_proof','gzip_ledger','lossless_execution_config'}}

OUTPUT_KEYS = {
 'test': {'receipt','failure'}, 'inventory': {'spec','report','receipt','failure'},
 'stage': {'receipt','local_commit','ledger','message','failure'},
 'recovery': {'receipt','ledger','failure'}, 'lossless': {'receipt','ledger','failure'},
 'join': {'receipt','addendum','failure'}}
_MODULES, _LOAD_LOCK = {}, threading.RLock()
METADATA_GIT_BLOBS={
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/FINAL_RECEIPT.json':'7e9d12e54b4235d20653fdbccf74e689097ed84f',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-execution.json':'b40be3a0a775a8118f45fe51ba339b3161452ad8',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-source-blobs/c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e.source':'62ccac42b959c7cead65ddf689f298de6c18b0bb',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/run_task_solver_attempt01.py':'62ccac42b959c7cead65ddf689f298de6c18b0bb'}

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def typed_equal(a,b):
    if type(a) is not type(b):return False
    if type(b) is dict:return set(a)==set(b) and all(typed_equal(a[k],b[k]) for k in b)
    if type(b) is list:return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b
def sha(value, length=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value): raise ValueError('Invalid complete hash')
    return value
def number(value):
    if type(value) is not int or value < 0: raise ValueError('Nonnegative integer required; booleans rejected')
    return value
def name(value):
    if type(value) is not str or not value or any(v in value for v in ('\\','\x00','\n','\r')):
        raise ValueError('Invalid literal relative path')
    p = Path(value)
    if p.is_absolute() or p.as_posix()!=value or any(x in ('.','..','.git') for x in p.parts):
        raise ValueError('Path alias/traversal rejected')
    if any(x == '__pycache__' for x in p.parts) or p.suffix == '.pyc' or p.name in ('.DS_Store','.env') or p.suffix in ('.pem','.key'):
        raise ValueError('Protected/generated path rejected')
    return value
def safe(root, value, exists=True):
    value = name(value); p = root/value
    if any(root.joinpath(*Path(value).parts[:i]).is_symlink() for i in range(1,len(Path(value).parts)+1)):
        raise ValueError('Symlink path rejected')
    if not p.resolve().is_relative_to(root.resolve()) or (exists and not p.is_file()): raise ValueError('File custody/path differs')
    return p
def safe_output(value):
    p = safe(FACTORY,value,False)
    if not value.startswith('backup_logs/') or p.exists(): raise ValueError('Fresh backup-owned output alias required')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
def stat_key(p):
    s=p.stat(); return tuple(getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
def hash_stream(handle,size):
    number(size); h=hashlib.sha256(); g=hashlib.sha1(f'blob {size}\0'.encode()); count=0
    while b:=handle.read(CHUNK):
        count+=len(b)
        if count>size: raise ValueError('Frozen size exceeded')
        h.update(b);g.update(b)
    if count!=size: raise ValueError('Frozen size incomplete')
    return {'size':count,'sha256':h.hexdigest(),'git_blob':g.hexdigest()}
def hash_file(p):
    before=stat_key(p)
    with p.open('rb') as s: result=hash_stream(s,before[2])
    if stat_key(p)!=before: raise ValueError('File evolved while hashing')
    return result
def entry(value, cap=False, git_required=False):
    if type(value) is not dict: raise ValueError('Entry object required')
    name(value.get('path')); number(value.get('size')); sha(value.get('sha256'))
    if cap and value['size']>CAP: raise ValueError('Repository file exceeds cap')
    if git_required: sha(value.get('git_blob'),40)
    return value
def equal(row,actual):
    if any(actual.get(k)!=row.get(k) or type(actual.get(k)) is not type(row.get(k)) for k in ('size','sha256')):
        raise ValueError('Exact bytes differ: '+row.get('path','<stream>'))
    if 'git_blob' in row and actual.get('git_blob')!=row['git_blob']: raise ValueError('Git blob bytes differ')
    return actual
def projection(row): return {k:row[k] for k in ('path','size','sha256')}
def file_binding(p):
    proof=hash_file(p)
    return {'path':str(p.relative_to(FACTORY)),'size':proof['size'],'sha256':proof['sha256']}
def validate_binding(row):
    if type(row) is not dict or set(row)!={'path','size','sha256'}: raise ValueError('Closed input binding required')
    entry(row);equal(row,hash_file(safe(FACTORY,row['path'])));return row
def bound_json(row): validate_binding(row);return json.loads(safe(FACTORY,row['path']).read_bytes())
def fixed_json(key):
    path,expected=FIXED[key];p=safe(FACTORY,path)
    if hash_file(p)['sha256']!=expected: raise ValueError('Immutable dependency changed')
    return json.loads(p.read_bytes())
def library(key):
    with _LOAD_LOCK:
        path,expected=FIXED[key];p=safe(FACTORY,path)
        if hash_file(p)['sha256']!=expected: raise ValueError('Frozen library changed before import')
        if key not in _MODULES:
            s=importlib.util.spec_from_file_location('next653_frozen_'+key,p);m=importlib.util.module_from_spec(s)
            sys.modules[s.name]=m;s.loader.exec_module(m);_MODULES[key]=m
        if hash_file(p)['sha256']!=expected: raise ValueError('Frozen library changed after import')
        return _MODULES[key]
def write_exclusive(p,value):
    with p.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return file_binding(p)
def process_stream(argv,consume,cwd=None,timeout=180):
    environment=dict(os.environ, GIT_OPTIONAL_LOCKS='0', PYTHONDONTWRITEBYTECODE='1')
    process=subprocess.Popen(argv,cwd=cwd,env=environment,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    timer=threading.Timer(timeout,process.kill);timer.daemon=True;timer.start()
    try:
        result=consume(process.stdout)
        if process.wait(timeout=10)!=0: raise ValueError('Process failed; payload/error body withheld')
        return result
    finally:
        timer.cancel();process.stdout.close()
        if process.poll() is None:process.kill()
        process.wait()
def command(argv,cwd=None):
    def consume(s):
        output=bytearray()
        while b:=s.read(CHUNK):
            output.extend(b)
            if len(output)>16*1024*1024:raise ValueError('Metadata exceeded bound')
        return bytes(output)
    return process_stream(argv,consume,cwd)
def git(*args): return command(['git',*args],WORKTREE)
def text(*args):return git(*args).decode().strip()
def api(endpoint):return json.loads(command(['gh','api','-H','Cache-Control: no-cache',endpoint]))
def raw_blob(blob,consume):
    sha(blob,40)
    return process_stream(['gh','api','-H','Accept: application/vnd.github.raw+json','-H','Cache-Control: no-cache',f'repos/{REPOSITORY}/git/blobs/{blob}'],consume)
def ledger(p,event):
    with p.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
def new_ledger(value):
    p=safe_output(value)
    with p.open('x'):pass
    return p
def ledger_rows(row,event):
    validate_binding(row);rows=[]
    with safe(FACTORY,row['path']).open() as stream:
        for line in stream:
            if len(line)>1024*1024:raise ValueError('Oversized ledger row')
            r=json.loads(line)
            if r.get('event')==event:rows.append(r)
    return rows

class SecretSession:
    """Per-execution/context state; no mutation of old module globals or regexes."""
    def __init__(self):
        self.helper=library('scanner');self.helper.validate_config()
        self.patterns=[re.compile(p,f) for p,f in self.helper.EXPECTED_PATTERN_SPECS]
        self.expected={r['path']:r for r in self.helper.EXPECTED_FILES};self.observations={};self.lock=threading.RLock()
    def stream(self,handle,size,paths,context,expected_blob=None):
        if context not in ('source_before','source_after','target','local_git','actual_remote'):raise ValueError('Actual provenance context required')
        if type(paths) is not list or not paths or len(set(paths))!=len(paths):raise ValueError('Unique exact original paths required')
        for p in paths:name(p)
        rows=[self.expected.get(p) for p in paths]
        known=[r for r in rows if r is not None]
        if known and len(known)!=len(paths):raise ValueError('Exception object cannot be admitted at another path')
        row=known[0] if known else None
        if known and any((r['size'],r['sha256'],r['matches'])!=(row['size'],row['sha256'],row['matches']) for r in known):
            raise ValueError('Cross-wired metadata object group')
        actual,matches=self.helper.scan_stream(handle,size,self.patterns,row,CHUNK)
        if expected_blob is not None and actual['git_blob']!=sha(expected_blob,40):raise ValueError('Git object bytes differ')
        if row is not None:
            if any(actual['git_blob']!=METADATA_GIT_BLOBS[p] for p in paths):raise ValueError('Original exact metadata Git object identity differs')
            with self.lock:
                for p in paths:self.observations.setdefault(context,{})[p]={'path':p,**actual,'matches':matches,'full_size_sha256_gitsha1_verified_before_exception_acceptance':True}
        return actual
    def file(self,p,original_name,context):
        name(original_name);before=stat_key(p)
        with p.open('rb') as s: result=self.stream(s,before[2],[original_name],context)
        if before!=stat_key(p):raise ValueError('Source/target evolved during complete scan')
        return result
    def complete(self,context):
        with self.lock:rows=dict(self.observations.get(context,{}))
        if set(rows)!=set(self.expected):raise ValueError('All four unique actual metadata observations required in '+context)
        for p,r in self.expected.items():
            v=rows[p]
            if (type(v['size']) is not int or v['size']!=r['size'] or v['sha256']!=r['sha256']
                    or not typed_equal(v['matches'],r['matches']) or v['git_blob']!=METADATA_GIT_BLOBS[p]
                    or v.get('full_size_sha256_gitsha1_verified_before_exception_acceptance') is not True):raise ValueError('Metadata report full-byte/span mismatch')
        return {'status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','context':context,
                'allowed_exact_match_detections':4,'unbound_secret_detections':0,'existing_regexes_unchanged':True,
                'scanner_source_sha256':FIXED['scanner'][1],'configuration_sha256':FIXED['metadata_config'][1],
                'four_exception_files':[rows[n] for n in sorted(rows)]}

def validate_metadata_report(report,context):
    session=SecretSession()
    if type(report) is not dict or report.get('context')!=context or report.get('status')!='PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS':raise ValueError('Metadata provenance/report truth differs')
    if (type(report.get('allowed_exact_match_detections')) is not int or report['allowed_exact_match_detections']!=4
        or type(report.get('unbound_secret_detections')) is not int or report['unbound_secret_detections']!=0
        or report.get('scanner_source_sha256')!=FIXED['scanner'][1] or report.get('configuration_sha256')!=FIXED['metadata_config'][1]
        or report.get('existing_regexes_unchanged') is not True):raise ValueError('Metadata immutable scanner/config differs')
    rows=report.get('four_exception_files')
    if type(rows) is not list or len(rows)!=4 or {r.get('path') for r in rows if type(r) is dict}!=set(session.expected):raise ValueError('Metadata report coverage incomplete/duplicate')
    for row in rows:
        r=session.expected[row['path']]
        if type(row.get('size')) is not int or row['size']!=r['size'] or row.get('sha256')!=r['sha256'] or not typed_equal(row.get('matches'),r['matches']) or row.get('full_size_sha256_gitsha1_verified_before_exception_acceptance') is not True:raise ValueError('Metadata actual byte/span evidence differs')
        # Git object IDs are independently derived from the immutable original bytes.
        if row.get('git_blob')!=METADATA_GIT_BLOBS[row['path']] or row.get('git_blob')!=hash_file(safe(FACTORY,row['path']))['git_blob']:raise ValueError('Metadata actual Git object evidence differs')
    return report

def load_execution_config(a,preparation):
    if a.execution_config.is_symlink() or hash_file(a.execution_config)['sha256']!=sha(a.expected_execution_config_sha256):raise ValueError('Execution config changed')
    cfg=json.loads(a.execution_config.read_bytes())
    if set(cfg)!={'schema','phase','checkpoint_id','runner_execution','expected_parent','authorization','source_freeze','inputs','fields','outputs'} or cfg['schema']!='next-checkpoint-653-execution-v2' or cfg['phase']!=a.phase:
        raise ValueError('Closed execution config schema differs')
    if cfg['expected_parent']!=PARENT or type(cfg['checkpoint_id']) is not str or not re.fullmatch('[a-z0-9][a-z0-9-]{0,95}',cfg['checkpoint_id']):raise ValueError('Explicit expected parent/checkpoint differs')
    if cfg['runner_execution'] not in ('/root','/root/backup_manager') or not typed_equal(cfg['authorization'],{'authorizer_execution':'/root','phase_granted':a.phase,'source_only_preparation_is_not_grant':True}):raise ValueError('Separate exact phase grant required')
    validate_runner(a.phase,cfg['runner_execution'])
    inputs=cfg['inputs']
    validate_runtime_membership(a.phase,inputs)
    names=[]
    for r in inputs.values():validate_binding(r);names.append(r['path'])
    if len(names)!=len(set(names)):raise ValueError('Duplicate runtime dependency path')
    if type(cfg['fields']) is not dict or type(cfg['outputs']) is not dict or set(cfg['outputs'])!=OUTPUT_KEYS[a.phase]:raise ValueError('Closed phase output aliases required')
    if len(set(cfg['outputs'].values()))!=len(cfg['outputs']):raise ValueError('Output aliases collide')
    for value in cfg['outputs'].values():safe_output(value)
    if any(v in names for v in cfg['outputs'].values()):raise ValueError('Outputs cannot overwrite immutable inputs')
    cfg['_config']=file_binding(a.execution_config);cfg['_source_freeze']=file_binding(a.source_freeze);cfg['_preparation']=preparation
    if not typed_equal(cfg['source_freeze'],cfg['_source_freeze']):raise ValueError('Execution config must bind exact actual preparation freeze')
    cfg['_inputs_before']=dict(inputs)
    if a.phase!='test':validate_test_receipt(cfg)
    return cfg
def validate_runtime_membership(phase,inputs):
    if phase not in PHASE_KEYS or type(inputs) is not dict or PHASE_KEYS[phase] != set(inputs):raise ValueError('Mandatory phase dependency omitted')
    if any(type(key) is not str or not key for key in inputs):raise ValueError('Runtime input labels must be unique nonempty strings')
    names=[r.get('path') if type(r) is dict else None for r in inputs.values()]
    if None in names or len(names)!=len(set(names)):raise ValueError('Runtime dependency path duplication/type differs')
    return True
def recheck_inputs(cfg):
    if cfg['phase'] != 'test':
        admission=validate_admission(cfg)
        validate_archive_mappings(cfg,admission)
    for row in cfg['_inputs_before'].values():validate_binding(row)
    validate_binding(cfg['_config']);validate_binding(cfg['_source_freeze'])
    for row in cfg['_preparation']['frozen_inputs']:validate_binding(row)
    import next653_runtime_guard
    next653_runtime_guard.validate(cfg,'post_phase')
def validate_runner(phase,runner):
    expected={'test':'/root/backup_manager','inventory':'/root/backup_manager','stage':'/root',
              'recovery':'/root','lossless':'/root/backup_manager','join':'/root/backup_manager'}
    if phase not in expected or runner!=expected[phase]:raise ValueError('Actual runner role differs from explicitly assigned phase')
    return True

def validate_admission(cfg,spec=None):
    # Own module is imported only after complete preparation/config checking.
    import next653_admission
    return next653_admission.validate(cfg,spec)

def validate_archive_mappings(cfg,admission,spec=None):
    import next653_archive_mapping
    return next653_archive_mapping.validate(cfg,admission,spec)

def validate_test_receipt(cfg):
    binding=cfg['inputs']['preparation_tests'];tested=bound_json(binding)
    if (tested.get('status')!='PASS' or type(tested.get('tests_run')) is not int or tested['tests_run']<=0
            or any(type(tested.get(k)) is not int or tested[k]!=0 for k in ('failures','errors','skipped'))
            or not typed_equal(tested.get('source_freeze'),cfg['_source_freeze'])
            or tested.get('actual_runner_execution')!='/root/backup_manager'
            or tested.get('no_git_or_network_execution') is not True
            or type(tested.get('output')) is not str
            or hashlib.sha256(tested['output'].encode()).hexdigest()!=tested.get('output_sha256')):
        raise ValueError('Actual successful measured preparation test result differs')
    value=bound_json(tested['execution_config'])
    validate_completed_config_value(value,'test',cfg['checkpoint_id'],cfg['_source_freeze'],cfg['inputs'])
    if value['outputs']['receipt']!=binding['path']:raise ValueError('Actual preparation test output/config cross-wired')
    return tested

def validate_completed_config_value(value,phase,checkpoint,freeze,inputs):
    keys={'schema','phase','checkpoint_id','runner_execution','expected_parent','authorization',
          'source_freeze','inputs','fields','outputs'}
    if type(value) is not dict or set(value)!=keys or value['schema']!='next-checkpoint-653-execution-v2':
        raise ValueError('Actual completed phase config schema differs')
    if value['phase']!=phase or value['checkpoint_id']!=checkpoint or value['expected_parent']!=PARENT:
        raise ValueError('Completed config phase/checkpoint/parent cross-wired')
    validate_runner(phase,value['runner_execution'])
    expected={'authorizer_execution':'/root','phase_granted':phase,'source_only_preparation_is_not_grant':True}
    if not typed_equal(value['authorization'],expected) or not typed_equal(value['source_freeze'],freeze):
        raise ValueError('Completed config grant or actual preparation freeze differs')
    validate_runtime_membership(phase,value['inputs'])
    if type(value['fields']) is not dict or type(value['outputs']) is not dict or set(value['outputs'])!=OUTPUT_KEYS[phase]:
        raise ValueError('Completed config fields/outputs differ')
    if len(set(value['outputs'].values()))!=len(value['outputs']):raise ValueError('Completed outputs duplicate')
    for output in value['outputs'].values():
        name(output)
        if not output.startswith('backup_logs/'):raise ValueError('Completed output outside backup ownership')
    if any(output in {r['path'] for r in value['inputs'].values()} for output in value['outputs'].values()):
        raise ValueError('Completed output/input aliases collide')
    for label,row in value['inputs'].items():
        if type(row) is not dict or set(row)!={'path','size','sha256'}:raise ValueError('Closed completed input binding required')
        entry(row)
        if label in inputs and not typed_equal(row,inputs[label]):
            raise ValueError('Completed phase actual dependency hash differs: '+label)
    if phase=='test':
        if value['fields']!={}:raise ValueError('Completed test phase accepts no fields')
    elif phase=='inventory':
        if set(value['fields'])!={'pending_observed','byte_evolution','raw_relocations','stable_root_hold'} or value['fields']['stable_root_hold'] is not True:
            raise ValueError('Completed inventory field contract differs')
    elif phase=='stage':
        f=value['fields']
        if (set(f)!={'allow_ignored_exact_paths','message'} or type(f['message']) is not str or not f['message'].strip()
                or type(f['allow_ignored_exact_paths']) is not list
                or len(set(f['allow_ignored_exact_paths']))!=len(f['allow_ignored_exact_paths'])):
            raise ValueError('Completed stage field contract differs')
        for path in f['allow_ignored_exact_paths']:name(path)
    elif phase=='recovery':
        f=value['fields']
        if set(f)!={'commit_sha','workers','allow_evolving_source'} or type(f['workers']) is not int or f['workers']!=2 or f['allow_evolving_source'] is not False:
            raise ValueError('Completed bounded unchanged recovery contract differs')
        sha(f['commit_sha'],40)
    elif phase in ('lossless','join'):
        if set(value['fields'])!={'commit_sha'}:raise ValueError('Completed proof exact commit field missing')
        sha(value['fields']['commit_sha'],40)
    return value

def validate_completed_config(cfg,phase,binding):
    value=bound_json(binding)
    validate_completed_config_value(value,phase,cfg['checkpoint_id'],cfg['_source_freeze'],cfg['inputs'])
    # Recheck full actual source/test/helper/raw preparation, rather than trusting a relabeled binding.
    import next653_bootstrap
    next653_bootstrap.validate_freeze(safe(FACTORY,cfg['_source_freeze']['path']),cfg['_source_freeze']['sha256'])
    for row in value['inputs'].values():validate_binding(row)
    if phase!='test':validate_test_receipt(cfg)
    return value

def validate_completed_proof(cfg,phase,proof,proof_binding,spec,local_commit=False):
    label={'stage':'stage_execution_config','recovery':'recovery_execution_config','lossless':'lossless_execution_config'}[phase]
    if (not typed_equal(proof.get('source_freeze'),cfg['_source_freeze'])
            or not typed_equal(proof.get('execution_config'),cfg['inputs'][label])):
        raise ValueError('Proof must bind actual active preparation and correct completed phase config')
    value=validate_completed_config(cfg,phase,cfg['inputs'][label])
    output='local_commit' if local_commit else 'receipt'
    if value['outputs'][output]!=proof_binding['path'] or value['runner_execution']!=proof.get('actual_runner_execution'):
        raise ValueError('Actual proof runner/output alias differs from completed config')
    if proof.get('spec')!=cfg['inputs']['spec'] or not typed_equal(proof.get('exact_scope'),scope(spec)):
        raise ValueError('Proof completed SPEC/scope differs')
    if phase!='stage' and value['fields']['commit_sha']!=proof.get('commit_sha'):
        raise ValueError('Completed config actual commit differs')
    if phase=='stage':
        if not local_commit and proof.get('stage_ledger')!=cfg['inputs']['stage_ledger']:
            raise ValueError('Actual stage ledger differs')
        if value['outputs']['ledger']!=cfg['inputs']['stage_ledger']['path']:
            raise ValueError('Completed stage ledger output cross-wired')
        if not local_commit and value['outputs']['local_commit']!=cfg['inputs']['local_commit']['path']:
            raise ValueError('Completed preserved local-commit output cross-wired')
    else:
        ledger_label='full_ledger' if phase=='recovery' else 'gzip_ledger'
        if proof.get('ledger')!=cfg['inputs'][ledger_label] or value['outputs']['ledger']!=cfg['inputs'][ledger_label]['path']:
            raise ValueError('Actual completed proof/ledger/output differs')
        dependencies=('stage','local_commit','stage_ledger') if phase=='recovery' else ('stage','full_proof','full_ledger')
        for key in dependencies:
            if proof.get(key)!=cfg['inputs'][key]:raise ValueError('Actual completed proof dependency differs: '+key)
    return value

def keyed(rows,cap=False,git_required=False):
    if type(rows) is not list:raise ValueError('Inventory list required')
    result={}
    for row in rows:
        entry(row,cap,git_required)
        if row['path'] in result:raise ValueError('Duplicate inventory path')
        result[row['path']]=row
    return result
def all_entries(spec):
    files=keyed(spec['files'],True,'schema' in spec and spec['schema'] in ('next-checkpoint-653-exact-inventory-v1','next-checkpoint-653-exact-inventory-v2'))
    raw=keyed(spec['raw_size_exclusions'])
    mapped_reason='new_immutable_runtime_raw_preserved_as_binary_gzip_for_strict_zero_delta'
    if set(files)&set(raw) or any(r['size']<=CAP and (spec.get('schema')!='next-checkpoint-653-exact-inventory-v2' or r.get('reason')!=mapped_reason) for r in raw.values()):raise ValueError('Capped/raw scope overlap or unbound classification')
    return files,raw
def groups(files):
    result={}
    for path,row in sorted(files.items()):
        sha(row.get('git_blob'),40); key=row['git_blob']
        if key not in result:result[key]={'git_blob':key,'size':row['size'],'sha256':row['sha256'],'paths':[]}
        if (result[key]['size'],result[key]['sha256'])!=(row['size'],row['sha256']):raise ValueError('One Git object cross-wired to distinct bytes')
        result[key]['paths'].append(path)
    return result
def release_rows(entries):return {p:projection(r) for p,r in entries.items() if '/releases/' in '/'+p}
def preserve_prior(prior,entries,deferred=()):
    oldfiles,oldraw=all_entries(prior);old={**oldfiles,**oldraw}
    if set(old)-set(entries) or set(old)&set(deferred):raise ValueError('Previously tracked/raw path absent/deferred')
    if any(entries[p]['size']>CAP for p in oldfiles):raise ValueError('Prior tracked path cannot silently become an omitted oversized raw')
    for p,row in release_rows(old).items():
        if p not in entries or projection(entries[p])!=row:raise ValueError('Immutable historical release changed/missing')
    return {'previous_capped_paths_preserved':len(oldfiles),'prior_immutable_release_bytes_preserved':len(release_rows(old))}
def scope(spec):
    files,raw=all_entries(spec);g=groups(files)
    pairs={};manifest_paths=set();refs=0;local_bytes=0
    for p,r in raw.items():
        reps=r.get('lossless_representations')
        if type(reps) is not list or not reps:raise ValueError('Every oversized raw requires actual recovery representation')
        seen=set()
        for rep in reps:
            packed,manifest=rep['gzip'],rep['manifest'];entry(packed,True);entry(manifest,True)
            if packed['path'] not in files or manifest['path'] not in files or projection(files[packed['path']])!=projection(packed) or projection(files[manifest['path']])!=projection(manifest):raise ValueError('Raw recovery bytes not in capped tracked scope')
            if '/releases/' in '/'+packed['path'] or '/releases/' in '/'+manifest['path']:raise ValueError('Recovery package inside immutable release')
            key=(p,packed['path'],manifest['path'])
            if key in seen:raise ValueError('Duplicate representation row')
            seen.add(key);refs+=1;local_bytes+=r['size'];manifest_paths.add(manifest['path'])
            pairs[(p,packed['path'])]={'raw':r,'gzip':files[packed['path']]}
    return {'files':len(files),'bytes':sum(r['size'] for r in files.values()),'unique_git_blob_groups':len(g),
            'raw_paths':len(raw),'raw_bytes':sum(r['size'] for r in raw.values()),'local_representation_iterations':refs,
            'local_decoded_bytes':local_bytes,'unique_raw_gzip_pairs':len(pairs),'remote_decoded_bytes':sum(v['raw']['size'] for v in pairs.values()),
            'global_manifest_paths':len(manifest_paths),'release_union_files':len(release_rows({**files,**raw}))}

def validate_spec_closure(cfg,spec):
    if (spec.get('schema')!='next-checkpoint-653-exact-inventory-v2' or spec.get('expected_parent_sha')!=PARENT
            or spec.get('checkpoint_id')!=cfg['checkpoint_id'] or not typed_equal(spec.get('exact_scope'),scope(spec))
            or spec.get('source_freeze')!=cfg['_source_freeze'] or spec.get('inventory_report')!=cfg['inputs']['inventory_report']
            or spec.get('execution_config')!=cfg['inputs']['inventory_execution_config'] or spec.get('grants_data_approval') is not False):raise ValueError('Complete inventory source/config/report/scope closure differs')
    inventory_config=validate_completed_config(cfg,'inventory',cfg['inputs']['inventory_execution_config'])
    admission=validate_admission(cfg,spec)
    mappings=validate_archive_mappings(cfg,admission,spec)
    files,raw=all_entries(spec);oldfiles,_=all_entries(fixed_json('prior'))
    if set(oldfiles)-set(files):raise ValueError('Every prior3158 tracked path must remain directly tracked')
    if inventory_config.get('phase')!='inventory' or inventory_config.get('expected_parent')!=PARENT or inventory_config.get('checkpoint_id')!=cfg['checkpoint_id']:raise ValueError('Inventory execution config is cross-wired')
    report=bound_json(cfg['inputs']['inventory_report'])
    if report.get('status')!='PASS_EXACT_INVENTORY_ONLY' or report.get('source_freeze')!=cfg['_source_freeze'] or report.get('execution_config')!=cfg['inputs']['inventory_execution_config']:raise ValueError('Actual inventory report closure differs')
    for ctx in ('source_before','source_after'):validate_metadata_report(report['metadata_reports'][ctx],ctx)
    if report.get('actual_core_producer_admission') != spec.get('actual_core_producer_admission'):raise ValueError('Inventory report admission snapshot differs')
    if report.get('new_runtime_raw_binary_mapping_contract')!=spec.get('new_runtime_raw_binary_mapping_contract') or not typed_equal(report.get('new_runtime_raw_binary_mappings'),spec.get('new_runtime_raw_binary_mappings')):raise ValueError('Inventory report binary raw mapping differs')
    if inventory_config['outputs']['spec'] != cfg['inputs']['spec']['path'] or inventory_config['outputs']['report'] != cfg['inputs']['inventory_report']['path']:raise ValueError('Inventory completed outputs cross-wired')
    return True

def custody(expected,clean=True,remote_expected=None):
    m=library('manager'); prior_join=fixed_json('custody')
    rows=prior_join['current_actual_custody']['six_protected_git_baselines']
    adapter={'protected_git_byte_baselines':rows}
    baselines=m.normalize_baselines(ORIGINAL,adapter);m.original_custody(ORIGINAL,adapter)
    if remote_expected is None:
        private=m.dedicated_custody(WORKTREE,expected,clean)
    else:
        m.require_head(WORKTREE,expected)
        if m.text(WORKTREE,'branch','--show-current')!=BRANCH or m.text(WORKTREE,'remote','get-url','origin')!=f'https://github.com/{REPOSITORY}.git':raise ValueError('Dedicated branch/remote changed')
        if clean and m.text(WORKTREE,'status','--porcelain=v1','--untracked-files=all'):raise ValueError('Dedicated dirty; preserve state')
        if m.text(WORKTREE,'ls-remote','--heads','origin','refs/heads/'+BRANCH).split()!=[remote_expected,'refs/heads/'+BRANCH]:raise ValueError('Remote branch commit differs')
        private=json.loads(command(['gh','repo','view',REPOSITORY,'--json','isPrivate,nameWithOwner,viewerPermission']))
        if private!={'isPrivate':True,'nameWithOwner':REPOSITORY,'viewerPermission':'ADMIN'}:raise ValueError('Private ADMIN required')
        if m.text(WORKTREE,'config','--get','index.sparse')!='false' or set(m.text(WORKTREE,'sparse-checkout','list').splitlines())!={'tools',PREFIX}:raise ValueError('Full index/restored scope differs')
    reference=fixed_json('references');reference_root=FACTORY.parent/'rineya_complementary_research_2026-10-09'
    checks=reference['reference_checks']
    if type(checks) is not list or len(checks)!=65 or len({r['path'] for r in checks})!=65:raise ValueError('Exact 65 reference baseline required')
    refs=[]
    for r in checks:
        p=safe(reference_root,r['path']);actual=hash_file(p)
        if actual['sha256']!=r['expected_sha256']:raise ValueError('Protected readonly reference changed')
        refs.append({'path':str(p),'size':actual['size'],'sha256':actual['sha256']})
    core=api('rate_limit')['resources']['core']
    if any(type(core.get(k)) is not int for k in ('limit','remaining','used','reset')):raise ValueError('API core budget unavailable')
    return {'head':expected,'remote_expected':remote_expected or expected,'private_admin':private,'six_protected_git_baselines':baselines,
            'readonly_reference_files':refs,'readonly_reference_count':65,'disk_free_bytes':shutil.disk_usage(FACTORY).free,
            'api_core_counts_only':{k:core[k] for k in ('limit','remaining','used','reset')}}
def capacity(spec,free):
    needed=2*scope(spec)['bytes']+RESERVE
    if free<needed:raise ValueError('Actual 2x capped inventory plus reserve unavailable')
    return needed
def verify_sources(spec,session,context):
    files,raw=all_entries(spec)
    for p,row in {**files,**raw}.items():equal(row,session.file(safe(FACTORY,p),p,context))
    return session.complete(context)
def current_oversized_names():
    return {p.relative_to(FACTORY).as_posix() for p in FACTORY.rglob('*') if p.is_file() and p.stat().st_size>CAP}
def compare_gzip(raw,packed):
    before=(stat_key(raw),stat_key(packed));total=0
    with raw.open('rb') as a,gzip.open(packed,'rb') as b:
        while True:
            x,y=a.read(CHUNK),b.read(CHUNK)
            if x!=y:raise ValueError('Lossless decoded bytes differ')
            total+=len(x)
            if not x:break
    if before!=(stat_key(raw),stat_key(packed)):raise ValueError('Raw/package changed during decode')
    return total
def verify_local_lossless(spec):
    files,raw=all_entries(spec);vector=scope(spec);rows=[]
    if current_oversized_names()!={p for p,r in raw.items() if r['size']>CAP}:raise ValueError('Current oversized scope differs, including postcut files')
    for p,r in raw.items():
        for rep in r['lossless_representations']:
            for role in ('gzip','manifest'):equal(rep[role],hash_file(safe(FACTORY,rep[role]['path'])))
            count=compare_gzip(safe(FACTORY,p),safe(FACTORY,rep['gzip']['path']))
            if count!=r['size']:raise ValueError('Local decode size differs')
            rows.append({'raw_path':p,'gzip_path':rep['gzip']['path'],'manifest_path':rep['manifest']['path'],'raw_bytes':count,'byte_for_byte_equal':True})
    if len(rows)!=vector['local_representation_iterations'] or sum(r['raw_bytes'] for r in rows)!=vector['local_decoded_bytes']:raise ValueError('Actual local decode accounting differs')
    return rows
def verify_git_scope(spec,commit,staged=False):
    m=library('manager');files,_=all_entries(spec);expected=files
    if m.text(WORKTREE,'rev-list','--parents','-n','1',commit).split()!=[commit,PARENT] and not staged:raise ValueError('Commit sole parent differs')
    parent_tree=m.whole_tree(WORKTREE,PARENT);wanted=m.expected_whole_tree(parent_tree,expected)
    actual=m.whole_index(WORKTREE) if staged else m.whole_tree(WORKTREE,commit)
    if actual!=wanted:raise ValueError('Whole tree/index or outside-prefix bytes differ')
    if staged:
        return actual
    if m.whole_index(WORKTREE)!=wanted:raise ValueError('Current index differs from frozen committed full tree')
    return actual
def local_groups(spec,session,ledger_path=None,staged=False):
    files,_=all_entries(spec);verified=[]
    for blob,row in groups(files).items():
        actual=process_stream(['git','cat-file','blob',blob],lambda s,r=row:session.stream(s,r['size'],r['paths'],'local_git',r['git_blob']),WORKTREE)
        equal(row,actual);v={**row,'actual_local_git_stream_this_run':True};verified.append(v)
        if ledger_path:ledger(ledger_path,{'event':'local_git_blob_verified',**v})
    return verified,session.complete('local_git')
def assert_strict_diff(observation):
    if (type(observation) is not dict or set(observation)!={'returncode','stdout','stderr'} or type(observation['returncode']) is not int
            or observation['returncode']!=0 or type(observation['stdout']) is not bytes or observation['stdout']!=b''
            or type(observation['stderr']) is not bytes or observation['stderr']!=b''):raise ValueError('New parent653 delta must strictly pass with zero warnings')
    return {'strict_diff_check':'PASS','unrestricted_exit_code':0,'warning_count':0,'historical_old_six_warning_exception_reused':False}
def strict_diff_cached():
    m=library('manager');result=m.bounded_diff(WORKTREE)
    try:return assert_strict_diff(result)
    except ValueError:raise DiffRejected(result)
class DiffRejected(ValueError):
    def __init__(self,observation):
        super().__init__('New parent653 delta contains warnings; no historical exception is authorized')
        self.observation=library('manager').diff_observation(observation)
def strict_diff_commit(commit):
    # Read both pipes concurrently with a 2MiB bound per pipe before accumulation.
    p=subprocess.Popen(['git','diff','--check',PARENT,sha(commit,40)],cwd=WORKTREE,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    data={'stdout':bytearray(),'stderr':bytearray()};errors=[]
    def read(key):
        try:
            while chunk:=getattr(p,key).read(CHUNK):
                if len(data[key])+len(chunk)>2*1024*1024:raise ValueError('Diff metadata bound exceeded')
                data[key].extend(chunk)
        except BaseException as error:
            errors.append(error)
            if p.poll() is None:p.kill()
        finally:getattr(p,key).close()
    threads=[threading.Thread(target=read,args=(key,),daemon=True) for key in data]
    for thread in threads:thread.start()
    try:
        code=p.wait(timeout=180)
    finally:
        if p.poll() is None:p.kill();p.wait()
        for thread in threads:thread.join()
    if errors:raise errors[0]
    result={'returncode':code,**{k:bytes(v) for k,v in data.items()}}
    try:return assert_strict_diff(result)
    except ValueError:raise DiffRejected(result)
def validate_actual_groups(rows,expected,flag):
    if type(rows) is not list or len(rows)!=len(expected):raise ValueError('Actual object proof coverage differs')
    seen=set()
    for row in rows:
        key=row.get('git_blob')
        if key in seen or key not in expected or row.get(flag) is not True:raise ValueError('Duplicate/cached/inherited object proof')
        seen.add(key);e=expected[key]
        if any(type(row.get(k)) is not type(e[k]) or row.get(k)!=e[k] for k in ('size','sha256','paths')):raise ValueError('Actual object path/size/hash grouping differs')
    if seen!=set(expected):raise ValueError('Missing actual object proof')
    return True
