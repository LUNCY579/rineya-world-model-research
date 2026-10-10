"""Owned source block copied into both entry points; never imported as a helper.
Source-only preparation may inspect this text but executes no stage functions.
"""
import hashlib,json,sys
from pathlib import Path

def anchored_closure(root,out,args,phase):
    """External SHA anchors precede all owned semantic-helper imports.
    Every operative current/raw pair, actual interpreter, loaded concrete source
    and existing cache is byte/size bound. Names and reported origins are exact.
    Frozen/builtin code has a binary-bound boundary, not source execution proof.
    """
    def rawhash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if rawhash(out/'RUN_CONFIG.json')!=args.expected_config_sha256:raise ValueError('external config SHA mismatch')
    if rawhash(out/'SOURCE_FREEZE.json')!=args.expected_source_freeze_sha256:raise ValueError('external sourcefreeze SHA mismatch')
    if rawhash(args.grant)!=args.expected_grant_sha256:raise ValueError('external grant SHA mismatch')
    grant=json.loads(Path(args.grant).read_text())
    if grant.get('authorized_by')!='/root' or grant.get('phase')!=phase:raise ValueError('explicit stage execution grant')
    if grant.get('config_sha256')!=args.expected_config_sha256 or grant.get('source_freeze_sha256')!=args.expected_source_freeze_sha256:raise ValueError('grant anchors')
    freeze=json.loads((out/'SOURCE_FREEZE.json').read_text())
    actual_python=str(Path(sys.executable).resolve())
    if actual_python!=freeze['resolved_python']:raise ValueError('resolvedPython path mismatch')
    bound={};pairs=[]
    for row in freeze['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else root/p
        p=p.resolve();archive=root/row['archive'];current=p.read_bytes();stored=archive.read_bytes()
        if len(current)!=row['size_bytes'] or len(stored)!=row['size_bytes'] or current!=stored:raise ValueError('current/raw exactpair '+row['path'])
        if hashlib.sha256(current).hexdigest()!=row['sha256'] or hashlib.sha256(stored).hexdigest()!=row['sha256']:raise ValueError('current/raw SHA '+row['path'])
        if str(p) in bound:raise ValueError('duplicate operative resolved path')
        bound[str(p)]={'sha256':row['sha256'],'size_bytes':row['size_bytes']}
        pairs.append({'path':row['path'],'sha256':row['sha256'],'size_bytes':row['size_bytes'],'archive':row['archive']})
    if actual_python not in bound:raise ValueError('resolvedPython bytes unbound')
    expected=freeze['runtime_module_identities'];modules=[]
    for name,module in sorted(sys.modules.items()):
        spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
        source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
        source_path=str(Path(source).resolve()) if source else None
        cache_path=str(Path(cache).resolve()) if cache else None
        namespace=[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []
        identity={'origin':origin,'source_path':source_path,'cache_path':cache_path,'namespace_paths':namespace}
        if name=='__main__':
            if source_path not in freeze['allowed_entrypoint_paths'] or origin is not None:raise ValueError('unbound main entrypoint origin')
        elif name not in expected or expected[name]!=identity:raise ValueError('new/changed runtime module identity '+name)
        concrete=[]
        for kind,value in [('source_or_extension',source_path),('cache',cache_path)]:
            if not value:continue
            path=Path(value)
            if not path.is_file():
                if kind=='source_or_extension':raise ValueError('loaded file missing '+name)
                continue
            if value not in bound:raise ValueError('new/unbound concrete runtime '+name+' '+kind+' '+value)
            raw=path.read_bytes();actual={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
            if actual!=bound[value]:raise ValueError('loaded runtime SHA/size mismatch '+name+' '+kind)
            concrete.append({'kind':kind,'path':value,**actual})
        classification='namespace' if namespace else 'builtin' if origin=='built-in' else 'frozen' if origin=='frozen' else 'file-backed' if concrete else 'no-concrete-file'
        modules.append({'module':name,'classification':classification,**identity,'concrete_bindings':concrete,
            'boundary':'directory locations only; contents not certified' if namespace else 'bound interpreter handles builtin/frozen origin; reported concrete fields independently bound'})
    return {'external_config_sha256':args.expected_config_sha256,'external_source_freeze_sha256':args.expected_source_freeze_sha256,
        'external_grant_sha256':args.expected_grant_sha256,'resolved_python':actual_python,'all_exact_raw_pairs_verified':pairs,
        'loaded_runtime_modules':modules,'unbound_concrete_file_modules':[],'module_origin_identity_checks':True}

def anchor_arguments(parser):
    parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--expected-source-freeze-sha256',required=True)
    parser.add_argument('--grant',required=True)
    parser.add_argument('--expected-grant-sha256',required=True)
