"""Own frozen stdlib metadata/byte reader; reviewed programs remain inert.

Literal binding and hashing routines adapted from own immutable attempt03 reader
7b9989f7ac02fc6156f59bd29fb994d0ce11deb73c5214ccfc5da464d72fa242.
No target import, AST evaluation, compile, phase, tree discovery or packing.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[1]
OLD2 = 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT02/'
OLD3 = 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT03/'
NEW = 'backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT04/'
PACK = 'backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT02/'
DIAG = 'backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT01/'
EXPECTED = {
 OLD2+'SOURCE_PREPARATION_FREEZE.json':'4d7bebfdb6e1d353504db18c6294735a1168a809fc88775bacc8c20984d02b00',
 OLD3+'SOURCE_PREPARATION_FREEZE.json':'4f4781a897941e28a827da09cad81a23e9588296d2861a093a46fd53ddda917b',
 NEW+'SOURCE_PREPARATION_FREEZE.json':'07fecaa4b0e938142d22878fa6396e16e6cf0a79798d676d0f87809a0f47d952',
 NEW+'SOURCE_READY.json':'aabaf34ff65729d71f1b3ed85ec4ebe938bfd871f2078671ec70792cdc89d8d0',
 NEW+'SOURCE_CHANGES_MINIMAL.diff':'74539370dbbaba41331cd3c0db17ff61f562af6d8f78544f393e929667d21216',
 PACK+'pack_exact_named_archival_text_v2.py':'37255f1e4ee610d83820ba5b6a8d80b94bcca5be4056e9761b29f9d60784c5d8',
 PACK+'EXACT_ARCHIVAL_TEXT_PACK_PLAN.json':'3161a87e4c0986b68fd4223eb58594ab65f4d5868cc811c5fbf14b462827e2b2',
 PACK+'SOURCE_PACK_PREPARATION_FREEZE.json':'5136c0cd5e396c2b3c8e69aed42ca8e53ecbd6a43068cd468e96208f09a4c33c',
 DIAG+'NAMED_RAW_MAPPING_SCOPE_LIMITATIONS.json':'a198702312a04450c920971e5807ed54ba824df6ab0aa8eb3d70a42edecd4b51',
 DIAG+'NAMED_RAW_WHITESPACE_DIAGNOSIS.json':'c55bd975fc1cb6133aaabe5f3d3676bd8e91324017c41ca908a0f2782848a229',
}
OBS = {}
def literal(name, absolute=False):
    if type(name) is not str or not name or any(c in name for c in ('\\','\0','\n','\r')):raise ValueError('Literal path required')
    p=Path(name)
    if p.is_absolute()!=absolute or p.as_posix()!=name or any(c in ('.','..') for c in p.parts):raise ValueError('Path alias rejected')
    p=p if absolute else ROOT/p
    if not p.is_file():raise ValueError('Declared file missing: '+name)
    q=p
    while q!=q.parent:
        if q.is_symlink():raise ValueError('Symlink rejected: '+name)
        q=q.parent
    return p
def hashed(name,absolute=False):
    p=literal(name,absolute);before=tuple(getattr(p.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    h=sha256()
    with p.open('rb') as f:
        while block:=f.read(65536):h.update(block)
    after=tuple(getattr(p.stat(),k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
    if before!=after:raise ValueError('Declared file evolved')
    row={'path':name,'size':before[2],'sha256':h.hexdigest()};OBS[name]=row
    if name in EXPECTED and row['sha256']!=EXPECTED[name]:raise ValueError('External fixed SHA differs: '+name)
    return row
def bound(row,absolute=False):
    if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',row['sha256']) is None:raise ValueError('Typed closed binding required')
    if hashed(row['path'],absolute)!=row:raise ValueError('Bound bytes differ: '+row['path'])
def meta(path):
    hashed(path);return json.loads(literal(path).read_bytes())
def frozen(path,count):
    value=meta(path);rows=value['frozen_inputs']
    if len(rows)!=count or len({r['path'] for r in rows})!=count:raise ValueError('Frozen membership count differs')
    for r in rows:bound(r)
    return value,{r['path']:r for r in rows}
def source(path):
    hashed(path);return literal(path).read_text()
def constants(text):
    # JSON parsing only of this source's closed quoted-data line, never eval/AST.
    line=next(s for s in text.splitlines() if s.startswith('EXACT_ORIGINAL_DIFF_EVIDENCE = '))
    return json.loads(line.split(' = ',1)[1].replace("'",'"'))
def warnings(raw):
    if b'\0' in raw:raise ValueError('Expected text raw')
    parts=raw.splitlines(keepends=True);out=[]
    for n,line in enumerate(parts,1):
        content=line[:-1] if line.endswith(b'\n') else line
        if content.endswith(b'\r'):content=content[:-1]
        if content.endswith((b' ',b'\t')):out.append({'line':n,'type':'trailing_whitespace','line_sha256':sha256(line).hexdigest()})
    if parts and parts[-1] in (b'\n',b'\r\n'):out.append({'line':len(parts),'type':'blank_line_at_eof','line_sha256':sha256(parts[-1]).hexdigest()})
    return out
def main():
    for path in EXPECTED:hashed(path)
    _,old2=frozen(OLD2+'SOURCE_PREPARATION_FREEZE.json',277)
    _,old3=frozen(OLD3+'SOURCE_PREPARATION_FREEZE.json',276)
    new,actual=frozen(NEW+'SOURCE_PREPARATION_FREEZE.json',276)
    for r in new['supplemental_custody_inputs']:bound(r)
    runtime=meta(NEW+'RUNTIME_SOURCE_FREEZE.json');pairs=runtime['runtime_bindings']
    if len(pairs)!=206 or len({r['current']['path'] for r in pairs})!=206:raise ValueError('Runtime206 differs')
    for r in pairs:
        bound(r['current'],True);bound(r['raw_archive'])
        if (r['current']['size'],r['current']['sha256'])!=(r['raw_archive']['size'],r['raw_archive']['sha256']):raise ValueError('Runtime/raw differs')
    sources=runtime['source_code_bindings']
    if len(sources)!=18 or sorted(r['path'] for r in sources)!=new['operative_source_paths']:raise ValueError('Operative18 differs')
    if sources!=meta(NEW+'RUNTIME_COLLECTOR_CONFIG.json')['source_code_bindings']:raise ValueError('Runtime/config sources differ')
    for r in sources:
        bound(r)
        if actual.get(r['path'])!=r:raise ValueError('Operative source missing')
    raws=new['raw_source_archives']
    if len(raws)!=12 or len({r['original_path'] for r in raws})!=12:raise ValueError('Own12 source/raw differs')
    changes=[]
    for r in raws:
        original=actual[r['original_path']];archive=actual[r['archive_path']]
        if (original['sha256'],original['size'])!=(archive['sha256'],archive['size']) or original['sha256']!=r['sha256']:raise ValueError('Own archive differs')
        name=Path(r['original_path']).name
        if hashed(OLD3+name)['sha256']!=original['sha256']:changes.append(name)
    if sorted(changes)!=['archive_mapping.py','common.py','inventory.py','tests.py']:raise ValueError('Unexpected Python delta')
    reason_old='new_immutable_runtime_raw_preserved_as_binary_gzip_for_strict_zero_delta'
    reason_new='new_immutable_archival_text_preserved_as_binary_gzip_for_strict_zero_delta'
    for name in ('common.py','inventory.py'):
        text=source(NEW+name)
        if text.count(reason_new)!=1 or text.replace(reason_new,reason_old)!=source(OLD3+name):raise ValueError('Reason-only delta violated')
    tests=source(NEW+'tests.py');a=tests.index('    def test_exact_six_original_diff_shape_and_actual_binary_roundtrips(self):');b=tests.index('    def mapping_fixture(self',a)
    if tests[:a]+tests[b:]!=source(OLD3+'tests.py'):raise ValueError('Old72 source changed')
    methods=re.findall(r'^    def (test_[A-Za-z0-9_]+)\(self\):',tests,re.M)
    if len(methods)!=74 or len(set(methods))!=74:raise ValueError('Unique74 method source count differs')
    bootstrap=source(NEW+'bootstrap.py');own=set(re.findall(r"'([^']+)'",bootstrap.split('OWN_NAMES = ',1)[1].split('\nFIXED = {',1)[0]))
    fixed=dict(re.findall(r"^\s*'([^']+)'\s*:\s*'([0-9a-f]{64})',?$",bootstrap.split('\nFIXED = {',1)[1].split('\n}',1)[0],re.M))
    policy=meta(NEW+'ADMISSION_POLICY.json');deps=[r for k in ('core_fixed','core_result_fixed','producer_fixed','living_transition_fixed') for r in policy[k].values()]+[policy[k] for k in ('preserved_failed_full_execution','preserved_focused_execution')]
    required={NEW+n for n in own}|set(fixed)|{r['path'] for r in deps}|{r['raw_archive']['path'] for r in pairs}|{r['archive_path'] for r in raws}|{meta(NEW+'RUNTIME_COLLECTOR_PREEXECUTION.json')['raw_collector_source']['path']}
    if len(own)!=22 or len(fixed)!=13 or required!=set(actual):raise ValueError('Closed bootstrap276 differs')
    if any(actual[r['path']]!=r for r in deps) or any(actual[p]['sha256']!=h for p,h in fixed.items()):raise ValueError('Supplemental binding substituted')
    six=constants(source(NEW+'archive_mapping.py'));plan=meta(PACK+'EXACT_ARCHIVAL_TEXT_PACK_PLAN.json')
    if six!=constants(source(PACK+'pack_exact_named_archival_text_v2.py')) or six!=meta(NEW+'ARCHIVE_MAPPING_CONTRACT_TEMPLATE.json')['exact_original_diff_evidence']:raise ValueError('Fixed six tuples disagree')
    scope=meta(DIAG+'NAMED_RAW_MAPPING_SCOPE_LIMITATIONS.json');originals=scope['new_original_diff_whitespace_paths']
    if len(six)!=6 or set(six)!={r['path'] for r in originals}:raise ValueError('Frozen diagnosis six differs')
    for r in originals:
        f=six[r['path']]
        if (r['bytes'],r['sha256'])!=(f['size'],f['sha256']):raise ValueError('Scope tuple differs')
        bound({'path':r['path'],'size':f['size'],'sha256':f['sha256']});bound({'path':f['archive_origin_path'],'size':f['size'],'sha256':f['sha256']})
        if warnings(literal(r['path']).read_bytes())!=r['exact_whitespace_lines']:raise ValueError('Actual six warnings differ')
    packfreeze=meta(PACK+'SOURCE_PACK_PREPARATION_FREEZE.json')
    for r in packfreeze['frozen_inputs']:bound(r)
    diagnosis=meta(DIAG+'NAMED_RAW_WHITESPACE_DIAGNOSIS.json');expected=list(diagnosis['proposed_mappings'])
    for row in list(expected):
        original=row['origin'];f=six.get(original['path'])
        if f is None:continue
        if row['raw']!={'path':f['archive_origin_path'],'size':f['size'],'sha256':f['sha256']}:raise ValueError('Frozen copy counterpart differs')
        k=sha256(original['path'].encode()).hexdigest()+'-'+original['sha256']
        expected.append({'raw':original,'origin':row['raw'],'whitespace_evidence':row['whitespace_evidence'],'gzip_path':'reproducibility/lossless_archives/rsm-runtime-whitespace-r1/'+k+'.gz','manifest_path':'manifests/lossless_archives/rsm-runtime-whitespace-r1/'+k+'.lossless.json'})
    expected.sort(key=lambda r:r['raw']['path']);rows=plan['planned_mappings']
    if rows!=expected or len(rows)!=13 or sum(r['raw']['size'] for r in rows)!=211105:raise ValueError('Exact13 plan differs')
    named=meta(plan['diagnosis_config']['path']);admitted={r['path']:r for r in named['admitted_bindings']};prior=set(named['prior_paths'])
    for r in rows:
        for role in ('raw','origin'):
            bound(r[role])
            if admitted.get(r[role]['path'])!=r[role]:raise ValueError('Plan not admitted')
        if r['raw']['path'] in prior or '/releases/' in '/'+r['raw']['path']:raise ValueError('Protected plan raw')
        if warnings(literal(r['raw']['path']).read_bytes())!=r['whitespace_evidence']:raise ValueError('Actual13 warnings differ')
    pre=meta(PACK+'ROOT_PACK_PREEXECUTION_TEMPLATE.json')
    if pre['root_authorizes_this_exact_pack'] is not None:raise ValueError('Template already authorizes')
    hashed(PACK+'SOURCE_PACK_READY.json')
    result={'status':'PASS_OWN_FROZEN_STDLIB_BYTE_AND_METADATA_READBACK_ONLY','author_execution':'/root/formal_oracle','actual_runner_execution':'/root/formal_oracle','at_utc':datetime.now(timezone.utc).isoformat(),'old02_bindings_unchanged':len(old2),'old03_bindings_unchanged':len(old3),'new04_bindings':len(actual),'supplemental_bindings':len(new['supplemental_custody_inputs']),'runtime_current_raw_pairs':len(pairs),'operative_sources':len(sources),'own_source_raw_pairs':len(raws),'changed_python_sources':sorted(changes),'unchanged_python_sources':8,'old72_test_source_exact':True,'test_methods_source_only':len(methods),'bootstrap_membership_exact':True,'six_fixed_tuples_exact':True,'pack_mappings':len(rows),'pack_raw_bytes':sum(r['raw']['size'] for r in rows),'observations':dict(sorted(OBS.items())),'reviewed_import_compile_test_phase_inventory_packing_git_network':0,'data_approval':False}
    with (HERE/'STATIC_BYTE_READBACK.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},sort_keys=True))
if __name__=='__main__':main()
