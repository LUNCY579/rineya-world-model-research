"""Own frozen stdlib byte/metadata readback; no reviewed program execution.
Literal hashing/set routines reuse own immutable attempt04 reader a057f035...
Targets, helper imports, AST compilation, packing and tree inventories are absent.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
ROOT=HERE.parents[1]
OLD='backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT04/'
NEW='backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT05/'
MAPPING='backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT02/ARCHIVE_MAPPING_CONTRACT.json'
ADMISSION='backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT01/ADMISSION_CONTRACT.json'
OBS={}
FIXED={
 OLD+'SOURCE_PREPARATION_FREEZE.json':'07fecaa4b0e938142d22878fa6396e16e6cf0a79798d676d0f87809a0f47d952',
 NEW+'SOURCE_READY.json':'863b686fd5dcfa249f7b123ef72665fe0c4d61fa16b7de77c5b5f7d549fcde27',
 NEW+'SOURCE_PREPARATION_FREEZE.json':'586ec1524c16420ba4269de5bc1cf191f3669db4eb611480090c27d284c5cd9e',
 NEW+'SOURCE_CHANGES_MINIMAL.diff':'14fe98a3671c69892bc45adc3a8686e515b30015582db10f0650f4d5aa7f0f8e',
 NEW+'EXACT_ROLE_REGISTRY_PHYSICAL_PATH_POLICY.json':'7f417a1dab171a720ed041d30475454ef8a3d86ae2a31d34eee52011cd8dc82c',
 MAPPING:'e26896c5764b57313cc8bd79748ee63162adefa0a6ada01bce12ea5ceedd2676',
 ADMISSION:'8f74d5d915c5d061efd5b20fe5efb9103d6d11689c3d44a0dd49fdfbdcf719bd',
 'quality_reports/RSM_MACHINE_PENDING_REVIEW_LIVING_ROOT_COMPLETION_ATTEMPT01.json':'e92965afdbb3bcca577717dd065a9a51abc7cba4869847a5c0926b78bbe78026',
 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/INSTALLED_RUNTIME_SOURCE_FREEZE.json':'415cb144658406925f7d2939edcdc907a17f8dc1e9e62616ba6808202fb5a6ea',
 'backup_logs/belief-v4-approved-release-r1_SPEC.json':'6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'}
def literal(name,absolute=False):
    if type(name) is not str or not name or any(c in name for c in ('\\','\0','\n','\r')):raise ValueError('Literal declared path required')
    p=Path(name)
    if p.is_absolute()!=absolute or p.as_posix()!=name or any(c in ('.','..') for c in p.parts):raise ValueError('Alias rejected')
    p=p if absolute else ROOT/p
    if not p.is_file():raise ValueError('Declared file missing')
    q=p
    while q!=q.parent:
        if q.is_symlink():raise ValueError('Symlink rejected')
        q=q.parent
    return p
def hashed(name,absolute=False):
    p=literal(name,absolute);before=tuple(getattr(p.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'));h=sha256()
    with p.open('rb') as f:
        while block:=f.read(65536):h.update(block)
    if before!=tuple(getattr(p.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('Declared bytes evolved')
    r={'path':name,'size':before[2],'sha256':h.hexdigest()};OBS[name]=r
    if name in FIXED and r['sha256']!=FIXED[name]:raise ValueError('External hash mismatch')
    return r
def bound(row,absolute=False):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None:raise ValueError('Typed closed binding differs')
    if hashed(row['path'],absolute)!=row:raise ValueError('Declared tuple differs')
def meta(path):hashed(path);return json.loads(literal(path).read_bytes())
def text(path):hashed(path);return literal(path).read_text()
def freeze(path,count):
    v=meta(path);rows=v['frozen_inputs']
    if len(rows)!=count or len({r['path'] for r in rows})!=count:raise ValueError('Preparation count differs')
    for r in rows:bound(r)
    return v,{r['path']:r for r in rows}
def main():
    for p in FIXED:hashed(p)
    old,oldrows=freeze(OLD+'SOURCE_PREPARATION_FREEZE.json',276)
    for r in old['supplemental_custody_inputs']:bound(r)
    new,rows=freeze(NEW+'SOURCE_PREPARATION_FREEZE.json',277)
    for r in new['supplemental_custody_inputs']:bound(r)
    runtime=meta(NEW+'RUNTIME_SOURCE_FREEZE.json');pairs=runtime['runtime_bindings']
    if len(pairs)!=206 or len({r['current']['path'] for r in pairs})!=206:raise ValueError('Runtime206 differs')
    for r in pairs:
        bound(r['current'],True);bound(r['raw_archive'])
        if (r['current']['size'],r['current']['sha256'])!=(r['raw_archive']['size'],r['raw_archive']['sha256']):raise ValueError('Runtime/raw differs')
    sources=runtime['source_code_bindings']
    if len(sources)!=18 or sorted(r['path'] for r in sources)!=new['operative_source_paths'] or sources!=meta(NEW+'RUNTIME_COLLECTOR_CONFIG.json')['source_code_bindings']:raise ValueError('Operative source closure differs')
    changes=[]
    for r in new['raw_source_archives']:
        a,b=rows[r['original_path']],rows[r['archive_path']]
        if (a['size'],a['sha256'])!=(b['size'],b['sha256']) or a['sha256']!=r['sha256']:raise ValueError('Own source/raw differs')
        name=Path(r['original_path']).name
        if hashed(OLD+name)['sha256']!=a['sha256']:changes.append(name)
    if len(new['raw_source_archives'])!=12 or sorted(changes)!=['admission.py','bootstrap.py','tests.py']:raise ValueError('Unexpected Python changes')
    admission=text(NEW+'admission.py');a=admission.index("REGISTRY_LOGICAL='research/role_registry.json'");b=admission.index('def add(',a)
    inverse=admission[:a]+admission[b:]
    inserted="    if r['path']==REGISTRY_LOGICAL:\n        # Logical snapshots keep their exact old names. Only the current physical\n        # closure key projects to the one canonical path already in parent653.\n        r=exact_registry_alias(r)['canonical_physical_binding']\n"
    result="            'exact_physical_path_aliases':[exact_registry_alias(registry_policy()['logical_current'])],\n"
    if inverse.count(inserted)!=1 or inverse.count(result)!=1 or inverse.replace(inserted,'').replace(result,'')!=text(OLD+'admission.py'):raise ValueError('Admission delta not closed')
    bootstrap=text(NEW+'bootstrap.py');addition=", 'EXACT_ROLE_REGISTRY_PHYSICAL_PATH_POLICY.json'"
    if bootstrap.count(addition)!=1 or bootstrap.replace(addition,'')!=text(OLD+'bootstrap.py'):raise ValueError('Bootstrap delta not one new dependency')
    tests=text(NEW+'tests.py');a=tests.index('class ExactRegistryPhysicalPathTests(');b=tests.index('def execute(cfg):',a)
    if tests[:a]+tests[b:]!=text(OLD+'tests.py'):raise ValueError('Old74 tests changed')
    methods=re.findall(r'^    def (test_[A-Za-z0-9_]+)\(self\):',tests,re.M)
    if len(methods)!=76 or len(set(methods))!=76:raise ValueError('Source76 methods differ')
    own=set(re.findall(r"'([^']+)'",bootstrap.split('OWN_NAMES = ',1)[1].split('\nFIXED = {',1)[0]))
    fixed=dict(re.findall(r"^\s*'([^']+)'\s*:\s*'([0-9a-f]{64})',?$",bootstrap.split('\nFIXED = {',1)[1].split('\n}',1)[0],re.M))
    policy=meta(NEW+'ADMISSION_POLICY.json');deps=[r for k in ('core_fixed','core_result_fixed','producer_fixed','living_transition_fixed') for r in policy[k].values()]+[policy[k] for k in ('preserved_failed_full_execution','preserved_focused_execution')]
    required={NEW+n for n in own}|set(fixed)|{r['path'] for r in deps}|{r['raw_archive']['path'] for r in pairs}|{r['archive_path'] for r in new['raw_source_archives']}|{meta(NEW+'RUNTIME_COLLECTOR_PREEXECUTION.json')['raw_collector_source']['path']}
    if len(own)!=23 or len(fixed)!=13 or required!=set(rows):raise ValueError('Required277 closure differs')
    if any(rows[r['path']]!=r for r in deps) or any(rows[p]['sha256']!=h for p,h in fixed.items()):raise ValueError('Fixed dependency mismatch')
    reg=meta(NEW+'EXACT_ROLE_REGISTRY_PHYSICAL_PATH_POLICY.json')
    prior=meta('backup_logs/belief-v4-approved-release-r1_SPEC.json')
    if [r for r in prior['files'] if r['path']=='research/ROLE_REGISTRY.json']!=[reg['historical_parent653_canonical']]:raise ValueError('Prior canonical mismatches')
    admission_meta=meta(ADMISSION);transition=next(r for r in admission_meta['later_living_or_catalog_evolution'] if r['current']['path']=='research/role_registry.json')
    if transition['current']!=reg['logical_current'] or transition['prior']!=reg['historical_core507_logical']:raise ValueError('Proof histories conflated')
    bound(reg['canonical_current']);bound(reg['logical_current']);canonical=literal(reg['canonical_current']['path']);logical=literal(reg['logical_current']['path'])
    if (canonical.stat().st_dev,canonical.stat().st_ino)!=(logical.stat().st_dev,logical.stat().st_ino):raise ValueError('Local exact spellings not one physical file')
    case_names=sorted(p.name for p in canonical.parent.iterdir() if p.name in ('ROLE_REGISTRY.json','role_registry.json'))
    if case_names!=['ROLE_REGISTRY.json']:raise ValueError('Unexpected physical spellings')
    mapping=meta(MAPPING);maps=mapping['raw_mappings']
    if len(maps)!=13 or len({r['raw']['path'] for r in maps})!=13:raise ValueError('Completed13 contract membership differs')
    for row in maps:
        for role in ('raw','origin','gzip','manifest'):bound(row[role])
    result={'status':'PASS_OWN_FROZEN_STDLIB_STATIC_METADATA_BYTE_READBACK_ONLY','author_execution':'/root/formal_oracle','actual_runner_execution':'/root/formal_oracle','old04_bindings_unchanged':len(oldrows),'new05_bindings':len(rows),'supplemental_bindings':len(new['supplemental_custody_inputs']),'runtime_current_raw_pairs':len(pairs),'operative_sources':len(sources),'own_source_raw_pairs':len(new['raw_source_archives']),'changed_python_sources':sorted(changes),'unchanged_python_sources':9,'old74_test_source_exact':True,'prepared_test_methods_source_only':len(methods),'bootstrap_membership_exact':True,'completed_mapping_contract_bound':len(maps),'completed_mapping_role_bindings_rehashed':4*len(maps),'registry_single_physical_name':case_names,'original_lower_proof_labels_preserved':True,'observations':dict(sorted(OBS.items())),'reviewed_import_compile_test_inventory_phase_pack_git_network':0,'data_approval':False}
    with (HERE/'STATIC_METADATA_READBACK.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},sort_keys=True))
if __name__=='__main__':main()
