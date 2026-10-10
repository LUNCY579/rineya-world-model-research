"""Benign stdlib-only quality-authoring preexec current/raw collector.
This source is externally hashed and raw-archived before its first import/run.
It imports no semantic helper, computes no target and assigns no quality vote.
It reuses the already frozen public/input/runtime/opaque provenance archive pairs.
"""
import argparse,datetime,hashlib,json,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    expected=sys.argv[1];raw=Path(__file__).read_bytes();archive=OUT/'source_archives/freeze_quality_author_r2.preexec.raw'
    if sha(raw)!=expected or archive.read_bytes()!=raw:raise ValueError('external author-freezer source/raw barrier')
    sfpath=OUT/'SOURCE_FREEZE.json'
    if sha(sfpath.read_bytes())!='0f8bf67ebd632c89a85efbdc151bbe5db08eacb25ca8358d9cdc7e30a413b25d':raise ValueError('original sourcefreeze')
    sf=json.loads(sfpath.read_text());rows=list(sf['operative_bindings']);bound={}
    for row in rows:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve();a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if a!=b or len(a)!=row['size_bytes'] or sha(a)!=row['sha256']:raise ValueError('original exact raw pair')
        bound[str(p)]=row
    if str(Path(sys.executable).resolve())!=sf['resolved_python'] or {k:getattr(sys.flags,k) for k in sf['runtime_flags']}!=sf['runtime_flags']:raise ValueError('runtime startup')
    identities=[]
    for name,module in sorted(sys.modules.items()):
        spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None);source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
        sp=str(Path(source).resolve()) if source else None;cp=str(Path(cache).resolve()) if cache else None
        ns=[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []
        value={'origin':origin,'source_path':sp,'cache_path':cp,'namespace_paths':ns}
        if name=='__main__':
            if sp!=str(Path(__file__).resolve()) or origin is not None:raise ValueError('main identity')
        elif sf['runtime_module_identities'].get(name)!=value:raise ValueError('new runtime identity '+name)
        for p in [sp,cp]:
            if p and Path(p).is_file() and name!='__main__':
                if p not in bound:raise ValueError('unbound runtime '+name)
                a=Path(p).read_bytes();row=bound[p]
                if sha(a)!=row['sha256'] or len(a)!=row['size_bytes']:raise ValueError('runtime bytes')
        identities.append({'module':name,**value})
    # Only new owned top-level files are copied. Original raw archive pointers,
    # including opaque provenance payloads, are reused; no foreign aliases.
    for path in sorted(OUT.iterdir()):
        if path.name in {"QUALITY_PREEXEC_R2_TOOL_STDOUT.txt","QUALITY_PREEXEC_R2_TOOL_STDERR.txt","QUALITY_PREEXEC_R2_TOOL_EXIT.txt"}:continue
        if not path.is_file() or str(path.resolve()) in bound:continue
        raw=path.read_bytes();digest=sha(raw);target=OUT/'source_archives'/(digest+'.source')
        if len(raw)>25000000:raise ValueError('artifact cap')
        if target.exists():
            if target.read_bytes()!=raw:raise ValueError('archive collision')
        else:
            with target.open('xb') as handle:handle.write(raw)
        row={'path':str(path.relative_to(ROOT)),'size_bytes':len(raw),'sha256':digest,'archive':str(target.relative_to(ROOT))}
        rows.append(row);bound[str(path.resolve())]=row
    result={'actual_author_execution':'/root/adversarial_auditor','stage':'QUALITY_AUTHOR_SOURCE_AND_ACTUAL_INSPECTION_PREEXEC',
        'collector_source_sha256':expected,'external_source_freeze_sha256':'0f8bf67ebd632c89a85efbdc151bbe5db08eacb25ca8358d9cdc7e30a413b25d',
        'files':rows,'resolved_python':sf['resolved_python'],'runtime_flags':sf['runtime_flags'],'actual_loaded_stdlib_identities':identities,
        'helper_imports':0,'semantic_target_execution':False,'quality_votes_generated':False,'original_raw_archive_pointers_reused':True}
    target=OUT/'QUALITY_AUTHOR_PREEXEC_R2.json'
    with target.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    for row in rows:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if a!=b or sha(a)!=row['sha256'] or len(a)!=row['size_bytes']:raise ValueError('aftercollection drift')
    print(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':sha(target.read_bytes()),'size_bytes':target.stat().st_size,'raw_pairs':len(rows),'actual_loaded_modules':len(identities)}))
if __name__=='__main__':main()
