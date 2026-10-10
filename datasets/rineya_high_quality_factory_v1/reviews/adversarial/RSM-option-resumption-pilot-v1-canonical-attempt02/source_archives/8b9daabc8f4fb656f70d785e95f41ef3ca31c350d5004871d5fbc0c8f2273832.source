"""SOURCE-ONLY byte/runtime collector. Import only benign stdlib; never import
owned solvers/comparator/emitter/checkers, execute tests or read target semantics.
Run only after author source read+preexec SHA/raw freeze, externally anchored.
Existing exact own-attempt01 raw archives are reused, never edited or copied.
"""
import argparse,copy,datetime,hashlib,importlib.util,itertools,json,math,os,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
OLD=ROOT/'reviews/adversarial/RSM-option-resumption-pilot-v1-canonical-attempt01'
INPUT=ROOT/'reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01'
PREP=ROOT/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'
CORRECTED=ROOT/'reviews/adversarial/RSM-option-resumption-public-interface-correction-20261010/attempt01'
ACTOR='/root/adversarial_auditor'
OWN_HELPERS={'frame_solver_r1':PREP/'frame_solver_r1.py','recursive_simulator_r1':PREP/'recursive_simulator_r1.py',
    'control_probes_r2':CORRECTED/'control_probes_r2.py','run_input_review_r1':INPUT/'run_input_review_r1.py'}
CAP=25000000

def digest(raw):return hashlib.sha256(raw).hexdigest()
def relative(path):
    p=Path(path).resolve();return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def descriptor(path):
    p=Path(path).resolve();raw=p.read_bytes()
    return {'path':relative(p),'size_bytes':len(raw),'sha256':digest(raw)}
def write(name,value):
    raw=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if len(raw)>CAP:raise ValueError('artifact cap')
    with (OUT/name).open('xb') as f:f.write(raw)
    return descriptor(OUT/name)
def identity(module):
    spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
    source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
    return {'origin':origin,'source_path':str(Path(source).resolve()) if source else None,
        'cache_path':str(Path(cache).resolve()) if cache else None,
        'namespace_paths':[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []}

def main():
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description='Metadata/raw-byte preparation only')
    parser.add_argument('--expected-author-source-freeze-sha256',required=True)
    args=parser.parse_args()
    authorpath=OUT/'AUTHOR_SOURCE_PREEXEC.json';authorraw=authorpath.read_bytes()
    if digest(authorraw)!=args.expected_author_source_freeze_sha256:raise ValueError('external author preexec SHA')
    author=json.loads(authorraw)
    # Current prepared sources/config/fixture protocol must exactly match their
    # preexec raw byte archives before collection. This collector is bound too.
    for row in author['files']:
        raw=(ROOT/row['path']).read_bytes();stored=(ROOT/row['archive']).read_bytes()
        if len(raw)!=row['size_bytes'] or raw!=stored or digest(raw)!=row['sha256']:raise ValueError('author preexec exactpair')
    oldraw=(OLD/'SOURCE_FREEZE.json').read_bytes()
    if digest(oldraw)!='643a989130915f5c1689795fabcc63ed87c1394df5968cc8ffefa98f85ed69e1':raise ValueError('own immutable attempt01 sourcefreeze drift')
    old=json.loads(oldraw);bindings={};reused=0;new_bytes=0
    for row in old['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
        a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if len(a)!=row['size_bytes'] or len(b)!=row['size_bytes'] or a!=b or digest(a)!=row['sha256']:raise ValueError('reused old current/raw pair drift '+row['path'])
        if str(p) in bindings:raise ValueError('duplicate old resolved path')
        bindings[str(p)]={**row,'archive_reuse':'own immutable attempt01 exact archive; pointer only'};reused+=1
    def add(path,category):
        nonlocal reused,new_bytes
        p=Path(path).resolve();raw=p.read_bytes();h=digest(raw)
        if len(raw)>CAP:raise ValueError('new raw archive cap')
        if str(p) in bindings:
            item=bindings[str(p)]
            if item['size_bytes']!=len(raw) or item['sha256']!=h:raise ValueError('existing binding drift')
            return item
        # Prefer already exact author-preexec bytes, then own old bytes by hash.
        existing=next((ROOT/r['archive'] for r in author['files'] if r['sha256']==h and r['size_bytes']==len(raw)),None)
        if existing is None:
            existing=next((ROOT/r['archive'] for r in old['operative_bindings'] if r['sha256']==h and r['size_bytes']==len(raw)),None)
        if existing is not None:
            if existing.read_bytes()!=raw:raise ValueError('reused archive raw equality')
            archive=existing;reused+=1
        else:
            archive=OUT/'source_archives'/(h+'.source')
            if not archive.exists():
                with archive.open('xb') as f:f.write(raw)
                new_bytes+=len(raw)
            if archive.read_bytes()!=raw:raise ValueError('new archive roundtrip')
        item={**descriptor(p),'category':category,'archive':relative(archive)};bindings[str(p)]=item;return item
    # Old custody/freeze and current new source preparation are operative. Their
    # contents assert no quality outcomes. No previous actor artifacts are read.
    for p in [OLD/'SOURCE_FREEZE.json',OLD/'PREPARATION_FINAL_CUSTODY.json',authorpath]:add(p,'own immutable source preparation metadata')
    for row in author['files']:add(ROOT/row['path'],'author-read unexecuted source/config/protocol or governing attachment')
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and 'source_archives' not in p.parts and p.name not in {'SOURCE_FREEZE.json','PREPARATION_FINAL_CUSTODY.json'}:add(p,'new owned unexecuted preparation/source/fixture protocol')
    # Actual interpreter binary, every loaded concrete source and existing cache.
    # No review/helper imports take place; private producer bodies remain bytes.
    python=Path(sys.executable).resolve();add(python,'actual resolved Python executable')
    runtime={};module_report=[]
    for name,module in sorted(sys.modules.items()):
        ident=identity(module)
        if name!='__main__':runtime[name]=ident
        concrete=[]
        for kind,key in [('source_or_extension','source_path'),('cache','cache_path')]:
            value=ident[key]
            if not value:continue
            p=Path(value)
            if p.is_file():concrete.append({'kind':kind,**add(p,'actual loaded runtime '+name+' '+kind)})
            elif kind=='source_or_extension':raise ValueError('loaded concrete source absent')
        classification='namespace' if ident['namespace_paths'] else 'builtin' if ident['origin']=='built-in' else 'frozen' if ident['origin']=='frozen' else 'file-backed' if concrete else 'no-concrete-file'
        module_report.append({'module':name,**ident,'classification':classification,'concrete_bindings':concrete})
    # Declare own semantic helper source/cache identity without importing it.
    # Before-helper closure checks all these bytes; after imports checks names,
    # exact spec origins/source/cache paths and newly loaded runtime modules.
    for name,path in OWN_HELPERS.items():
        p=path.resolve();cache=Path(importlib.util.cache_from_source(str(p))).resolve()
        add(p,'declared owned helper source NOT IMPORTED')
        if cache.is_file():add(cache,'declared owned helper existing cache NOT IMPORTED')
        runtime[name]={'origin':str(p),'source_path':str(p),'cache_path':str(cache),'namespace_paths':[]}
    operative=sorted(bindings.values(),key=lambda row:row['path'])
    # Last verification includes every current/size/SHA/raw pair after collection.
    for row in operative:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p
        current=p.read_bytes();stored=(ROOT/row['archive']).read_bytes()
        if len(current)!=row['size_bytes'] or len(stored)!=row['size_bytes'] or current!=stored or digest(current)!=row['sha256'] or digest(stored)!=row['sha256']:raise ValueError('postcollection current/raw pair drift')
    if str(Path(sys.executable).resolve())!=str(python):raise ValueError('postcollection Python path drift')
    # Exact comparator semantics prefix retained from the own original source.
    current=(OUT/'factual_comparison_r2.py').read_text();original=(OLD/'canonical_review_r1.py').read_text()
    prefix=original.split('def main():',1)[0]
    if not current.startswith(prefix):raise ValueError('original semantic comparators changed')
    sourcefreeze=write('SOURCE_FREEZE.json',{'actual_author_execution':ACTOR,'stage':'CANONICAL_ATTEMPT02_SOURCE_PREPARATION_ONLY',
        'author_source_preexec':descriptor(authorpath),'external_author_source_preexec_sha256':args.expected_author_source_freeze_sha256,
        'operative_bindings':operative,'resolved_python':str(python),'python_version':sys.version,
        'runtime_module_identities':runtime,'actual_collector_runtime_report':module_report,
        'allowed_entrypoint_paths':[str((OUT/name).resolve()) for name in ['factual_comparison_r2.py','receipt_emitter_r2.py','freeze_preparation_r2.py']],
        'declared_owned_helpers_not_imported':sorted(OWN_HELPERS),'actual_semantic_comparator_prefix_sha256':digest(prefix.encode()),
        'reused_exact_archive_pointer_count':reused,'new_raw_archive_bytes_written_by_collector':new_bytes,
        'comparison_helper_imports':0,'new_tests_or_calibrations':0,'gold_comparison_executions':0,'quality_votes':0,'designated_receipts':0,
        'metadata_byte_collection_actual_execution':True,'source_only_runtime_collection_imports':'benign stdlib only, after author source/hash freeze',
        'private_opaque_custody_only':old['opaque_private_source_paths'],'no_original_c35_metadata_payloads_copied':True,
        'runtime_boundary':'exact reported module identities plus all concrete source/existingcache bytes; builtin/frozen limited by resolved interpreter binary; namespace directory contents and arbitrary dynamic in-memory code are not certified',
        'source_assessment_scope':'prepared stages have not been imported, compiled, tested or run; closure validity/semantic outputs remain for separately authorized execution',
        'disclosure':'same underlying model/common public design/root-guided separation/own original frozen solvers and input-first negative evidence reuse; authorized gold layout exposure after own input FIRST, not an unseen blind novelty claim'})
    files=[descriptor(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    custody=write('PREPARATION_FINAL_CUSTODY.json',{'actual_author_execution':ACTOR,'stage':'CANONICAL_ATTEMPT02_SOURCE_PREPARED_FINAL',
        'source_freeze':sourcefreeze,'config':descriptor(OUT/'RUN_CONFIG.json'),'factual_comparator':descriptor(OUT/'factual_comparison_r2.py'),
        'receipt_emitter':descriptor(OUT/'receipt_emitter_r2.py'),'files':files,'operative_pairs':len(operative),'runtime_identity_count':len(runtime),
        'new_raw_archive_bytes_written_by_collector':new_bytes,'comparison_helper_imports_tests_or_semantic_execution':False,
        'no_actual_quality_map_or_data_votes':True,'no_designated_receipt_published':True,'attempt01_unchanged_unexecuted':True,
        'all_new_files_below25MB':all(p['size_bytes']<=CAP for p in files),'stop_writes':True})
    print(json.dumps({'source_freeze':sourcefreeze,'custody':custody,'operative_pairs':len(operative),'runtime_identity_count':len(runtime),
        'new_archive_bytes':new_bytes,'metadata_collection_only':True,'new_semantic_imports_tests_comparison':False,'stop_writes':True}))

if __name__=='__main__':main()
