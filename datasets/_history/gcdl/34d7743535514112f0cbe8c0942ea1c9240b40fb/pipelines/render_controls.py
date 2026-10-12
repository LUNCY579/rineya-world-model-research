"""Label-free immutable recovery controls derived from fully validated exports.

Prepared capsules are untrusted staging. Qualification rederives every new
capsule from raw manifests, inherits only an authenticated prior aggregate, and
checks actual GitHub snapshot/tree/blob bytes before publishing local authority.
The ordinary read path never supplies custody for deletion. Two local aggregate
slots avoid quadratic history; historical byte versions remain at immutable
remote snapshots. The 8 MiB aggregate and 32 MiB total controls are safe ceilings,
not guarantees for unlimited specification upgrades; hitting either retains raw
payloads until a separately reviewed maintenance change.
"""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import re
import time

from . import remote_archive as archive

POINTER = 'manifests/RENDER_CONTROLS.json'
PREFIX = 'manifests/render_controls/'
BUNDLE_SLOTS = (PREFIX+'bundle-slot-0.json', PREFIX+'bundle-slot-1.json')
MAX_CAPSULE_BYTES = 16 * 1024
MAX_BUNDLE_BYTES = archive.MAX_INDEX_BYTES
MAX_NEW_RECEIPTS = 4
MAX_RECEIPTS = 4096
MAX_CONTROL_STORAGE_BYTES = 32 * 1024 * 1024
MAX_PROGRAMME_BYTES = 512 * 1024 * 1024
METADATA_RESERVE_BYTES = 1024 * 1024
CACHE_BYTES = 2 * MAX_BUNDLE_BYTES
CAPSULE_VERSION = 'gcdl-render-controls-capsule-1'
BUNDLE_VERSION = 'gcdl-render-controls-bundle-1'
AUTHORITY_VERSION = 'gcdl-render-controls-authority-1'
FIELDS = ('manifest_path', 'dataset_id', 'view', 'split', 'source_records_examined',
          'variant_budget', 'export_version', 'training_launched', 'rendering_version',
          'rendered_rows', 'eligible_source_records')
OPTIONAL_FIELDS = ('rendering_versions', 'rendering_policy_set_version')
DESCRIPTOR_FIELDS = ('commit', 'git_blob_sha1', 'sha256', 'bytes', 'catalog_id',
                     'catalog_commit', 'archive_ref', 'catalog_path',
                     'catalog_sha256', 'catalog_git_blob_sha1', 'receipt_id')


def _raw(value):
    return (archive._canonical(value) + '\n').encode()


def _fail(message):
    raise archive.ArchiveError('render controls: ' + message)


def _check(root, deadline=None):
    archive.check_read_deadline(root)
    if deadline is not None and time.monotonic() >= deadline:
        raise archive.ReadDeadlineExceeded('render control publication deadline exceeded')
    archive._check_stop(root)


def _receipt_path(relative):
    relative = archive._relative(relative)
    path = PurePosixPath(relative)
    if path.parent != PurePosixPath('manifests') or not path.name.startswith('render-batch-') or not path.name.endswith('.json'):
        _fail('unmanaged receipt path')
    return relative


def _manifest_path(relative):
    relative = archive._relative(relative)
    path = PurePosixPath(relative)
    if len(path.parts) != 4 or path.parts[:2] != ('data', 'rendered') or path.name != 'manifest.json':
        _fail('unmanaged rendered manifest path')
    return relative


def _descriptor(index, relative):
    entry = index['files'].get(relative)
    if entry is None:
        _fail('projection requires verified archived entry: ' + relative)
    descriptor = {**entry, **index['catalogs'][entry['catalog_id']]}
    return {key: descriptor[key] for key in DESCRIPTOR_FIELDS}


def _read_local(root, relative, limit):
    return archive._local(root, relative, limit)[0]


def _capsule_path(identifier):
    if not archive._HEX64.fullmatch(str(identifier)):
        _fail('invalid capsule ID')
    return PREFIX + 'capsule-' + identifier + '.json'


def _bundle_path(identifier, slot=0):
    if not archive._HEX64.fullmatch(str(identifier)) or type(slot) is not int or slot not in (0,1):
        _fail('invalid bundle ID or slot')
    return BUNDLE_SLOTS[slot]


def _validate_descriptor(value):
    if not isinstance(value,dict) or set(value)!=set(DESCRIPTOR_FIELDS):_fail('invalid original custody descriptor')
    archive._metadata({k:value[k] for k in archive._META})
    if (not archive._HEX64.fullmatch(str(value['catalog_id'])) or not archive._HEX64.fullmatch(str(value['receipt_id']))
            or not archive._HEX64.fullmatch(str(value['catalog_sha256']))
            or not archive._HEX40.fullmatch(str(value['catalog_git_blob_sha1']))
            or value['catalog_path']!=archive.catalog_path(value['catalog_id'])
            or value['archive_ref']!=archive.archive_ref(value['catalog_commit'])):
        _fail('invalid qualified original custody descriptor')


def _capsule(value):
    keys = {'version', 'receipt_path', 'receipt_sha256', 'batch_sha256',
            'spec_sha256', 'code_sha256', 'source_code_sha256', 'protected_index_sha256',
            'datasets', 'capsule_id'}
    if (not isinstance(value, dict) or set(value) != keys or value['version'] != CAPSULE_VERSION
            or value['capsule_id'] != archive._digest({k:v for k,v in value.items() if k != 'capsule_id'})
            or not isinstance(value['datasets'], list) or not 1 <= len(value['datasets']) <= 5):
        _fail('invalid capsule shape/identity')
    _receipt_path(value['receipt_path'])
    for key in ('receipt_sha256','batch_sha256','spec_sha256','code_sha256','source_code_sha256','protected_index_sha256'):
        if not archive._HEX64.fullmatch(str(value[key])):
            _fail('invalid capsule digest')
    for dataset in value['datasets']:
        if not isinstance(dataset, dict) or set(dataset) != {'fields','code_sha256','source','manifest_custody','shards'}:
            _fail('invalid dataset projection')
        fields = dataset['fields']
        if (not isinstance(fields, dict) or not set(FIELDS) <= set(fields)
                or set(fields) - set(FIELDS) - set(OPTIONAL_FIELDS)
                or fields['split'] not in ('train','dev','ood') or fields['training_launched'] is not False
                or any(type(fields[k]) is not int or fields[k] < 0 for k in ('source_records_examined','rendered_rows','eligible_source_records','variant_budget'))):
            _fail('invalid projected policy/count fields')
        _manifest_path(fields['manifest_path'])
        if (not re.fullmatch(r'gcdl_rendered_[0-9a-f]{32}',str(fields['dataset_id']))
                or fields['view'] not in ('pointwise','listwise','pairwise','computation','distillation')
                or any(type(fields[k]) is not str or not 0<len(fields[k])<=128 for k in ('export_version','rendering_version'))
                or fields['variant_budget'] not in (1,2,3)):
            _fail('invalid projected dataset identity/policy')
        source=dataset['source']
        if not isinstance(source,dict) or set(source) != {'path','sha256','source_batch_manifest','source_batch_manifest_sha256','source_code_sha256','protected_index_sha256'}:
            _fail('invalid source projection')
        for key in ('sha256','source_batch_manifest_sha256','source_code_sha256','protected_index_sha256'):
            if not archive._HEX64.fullmatch(str(source[key])):_fail('invalid source digest')
        if (not source['path'].startswith('data/accepted/') or len(PurePosixPath(archive._relative(source['path'])).parts)!=3
                or not source['path'].endswith('.jsonl')
                or PurePosixPath(archive._relative(source['source_batch_manifest'])).parent!=PurePosixPath('manifests')
                or not PurePosixPath(source['source_batch_manifest']).name.startswith('batch-')
                or 'sealed' in source['source_batch_manifest']):_fail('unmanaged nonsealed source projection')
        if not archive._HEX64.fullmatch(str(dataset['code_sha256'])):_fail('invalid projected code digest')
        _validate_descriptor(dataset['manifest_custody'])
        if not isinstance(dataset['shards'],list) or len(dataset['shards']) > 12000:
            _fail('invalid shard inventory')
        for shard in dataset['shards']:
            if (not isinstance(shard,dict) or set(shard) != {'path','sha256','bytes','rows','custody'}
                    or type(shard['rows']) is not int or shard['rows'] < 0
                    or type(shard['bytes']) is not int or not 0 <= shard['bytes'] < archive.MAX_READ_BYTES
                    or not isinstance(shard['custody'],dict) or set(shard['custody']) != set(DESCRIPTOR_FIELDS)):
                _fail('invalid shard projection')
            _validate_descriptor(shard['custody'])
            if not archive._HEX64.fullmatch(str(shard['sha256'])):_fail('invalid projected shard digest')
            path=PurePosixPath(archive._relative(shard['path']))
            if path.parent != PurePosixPath(fields['manifest_path']).parent or not path.name.endswith('.jsonl'):
                _fail('unmanaged shard projection')
    if len(_raw(value)) > MAX_CAPSULE_BYTES:
        _fail('capsule exceeds metadata bound')
    return value


def _bundle(value):
    if (not isinstance(value,dict) or set(value) != {'version','repository','programme','predecessor','capsules','bundle_id'}
            or value['version'] != BUNDLE_VERSION or value['repository'] != archive.REPOSITORY
            or value['programme'] != archive.PROGRAMME or not isinstance(value['capsules'],dict)
            or not 1 <= len(value['capsules']) <= MAX_RECEIPTS
            or value['bundle_id'] != archive._digest({k:v for k,v in value.items() if k != 'bundle_id'})):
        _fail('invalid aggregate control bundle')
    if value['predecessor'] is not None:
        _authority(value['predecessor'])
    for relative,capsule in value['capsules'].items():
        if _capsule(capsule)['receipt_path'] != relative:
            _fail('bundle receipt key mismatch')
    if len(_raw(value)) > MAX_BUNDLE_BYTES:
        _fail('bundle exceeds metadata bound')
    return value


def _authority(value):
    keys={'version','repository','programme','bundle_path','commit','archive_ref','sha256','git_blob_sha1','bytes','authority_id'}
    if (not isinstance(value,dict) or set(value) != keys or value['version'] != AUTHORITY_VERSION
            or value['repository'] != archive.REPOSITORY or value['programme'] != archive.PROGRAMME
            or value['archive_ref'] != archive.archive_ref(value['commit'])
            or not archive._HEX64.fullmatch(str(value['sha256'])) or not archive._HEX40.fullmatch(str(value['git_blob_sha1']))
            or type(value['bytes']) is not int or not 0 < value['bytes'] <= MAX_BUNDLE_BYTES
            or value['authority_id'] != archive._digest({k:v for k,v in value.items() if k != 'authority_id'})):
        _fail('invalid control authority')
    if value['bundle_path'] not in BUNDLE_SLOTS:
        _fail('unmanaged authority bundle path')
    return value


def _pointer(root):
    path=archive._path(root,POINTER)
    if not path.exists():return None
    raw=_read_local(root,POINTER,MAX_BUNDLE_BYTES)
    pointer=_authority(archive._json(raw))
    immutable=PREFIX+'authority-'+pointer['authority_id']+'.json'
    if _read_local(root,immutable,MAX_BUNDLE_BYTES) != raw:
        _fail('authority lacks immutable local binding')
    return pointer


def _provider(root, transport=None):
    provider=transport if transport is not None else archive.read_transport(root)
    if transport is not None and hasattr(provider,'root') and archive._root(provider.root)!=root:
        _fail('transport root mismatch')
    if time.monotonic()>=provider.deadline:
        raise archive.ReadDeadlineExceeded('render control transport deadline exceeded')
    session=archive._READ_SESSION.get()
    if transport is not None and session is not None:
        if session.root!=root:_fail('read session root mismatch')
        if session.deadline is not None:provider.deadline=min(provider.deadline,session.deadline)
        if session._transport is not provider:
            session.catalogs.clear();session.catalog_bytes=0
            session.render_control_proofs={}
            session._transport=provider
    return provider


def _qualified(root, pointer, transport=None, *, fresh=False):
    """Fresh local bytes, bounded remote proof once per session/provider/pointer."""
    archive.check_read_deadline(root)
    pointer=_authority(pointer)
    raw=_read_local(root,pointer['bundle_path'],MAX_BUNDLE_BYTES)
    archive._matches(raw,pointer)
    bundle=_bundle(archive._json(raw))
    if pointer['bundle_path'] not in BUNDLE_SLOTS:
        _fail('bundle slot mismatch')
    provider=_provider(root,transport)
    session=archive._READ_SESSION.get()
    key=(provider,archive._digest(pointer))
    cache=None
    if not fresh and session is not None and provider is session._transport:
        cache=getattr(session,'render_control_proofs',None)
        if cache is None:
            cache={};session.render_control_proofs=cache
        if key in cache:
            if cache[key] != raw:_fail('changed qualified control bytes')
            return bundle
    # Original payload snapshots remain live custody even when raw manifest
    # headers are not fetched. Confirm each relevant catalog before/after the
    # control proof once per session, with no whole-index dependency.
    references={pointer['archive_ref']:pointer['commit']}
    for capsule in bundle['capsules'].values():
        for dataset in capsule['datasets']:
            for descriptor in [dataset['manifest_custody']]+[s['custody'] for s in dataset['shards']]:
                reference=descriptor['archive_ref'];commit=descriptor['catalog_commit']
                if reference!=archive.archive_ref(commit):_fail('unqualified payload catalog snapshot')
                if reference in references and references[reference]!=commit:_fail('conflicting catalog snapshot')
                references[reference]=commit
    for reference,commit in sorted(references.items()):
        if provider.fetch_ref(reference)!=commit:_fail('control or payload snapshot moved or missing')
    remote=archive._tree_blob(provider,pointer['commit'],pointer['bundle_path'],pointer)
    if remote != raw:_fail('remote control bundle differs from local bytes')
    for reference,commit in sorted(references.items()):
        if provider.fetch_ref(reference)!=commit:_fail('control or payload snapshot changed during proof')
    if cache is not None:
        while cache and sum(len(v) for v in cache.values())+len(raw)>CACHE_BYTES:
            cache.pop(next(iter(cache)))
        cache[key]=raw
    return bundle


def derive(root, relative, *, deadline=None):
    """Pure projection derivation, ALWAYS force the historical full raw validator."""
    from . import rendered
    root=archive._root(root);relative=_receipt_path(relative);_check(root,deadline)
    receipt,_=rendered.validate_receipt(root,relative,use_controls=False)
    batch=rendered._batch(root,receipt['source_manifest'])
    index=archive.load_index(root)
    datasets=[]
    for item in receipt['datasets']:
        _check(root,deadline)
        manifest=archive._json(archive.read_bytes(root,item['manifest_path']))
        source=manifest['sources'][0]
        datasets.append({'fields':{k:manifest[k] for k in FIELDS+OPTIONAL_FIELDS if k in manifest},
                         'code_sha256':archive._digest(manifest['code_hashes']),
                         'source':{**{k:source[k] for k in ('path','sha256','source_batch_manifest','source_batch_manifest_sha256')},
                                   'source_code_sha256':archive._digest(source['source_code_hashes']),
                                   'protected_index_sha256':archive._digest(source['protected_index'])},
                         'manifest_custody':_descriptor(index,item['manifest_path']),
                         'shards':[{**{k:shard[k] for k in ('path','sha256','bytes','rows')},
                                    'custody':_descriptor(index,shard['path'])} for shard in manifest['shards']]})
    body={'version':CAPSULE_VERSION,'receipt_path':relative,
          'receipt_sha256':hashlib.sha256(archive.read_bytes(root,relative)).hexdigest(),
          'batch_sha256':archive._digest(batch),'spec_sha256':receipt['spec_sha256'],
          'code_sha256':archive._digest(receipt['code_hashes']),
          'source_code_sha256':archive._digest(batch['code_hashes']),
          'protected_index_sha256':archive._digest(batch['protected_index']),'datasets':datasets}
    return _capsule({**body,'capsule_id':archive._digest(body)})



def _write(root, relative, raw, *, immutable=True):
    """Control growth stays inside 32 MiB and the original 512 MiB programme."""
    path=archive._path(root,relative)
    existing=path.stat().st_size if path.exists() else 0
    addition=max(0,len(raw)-existing)
    controls=0;total=0
    for item in root.rglob('*'):
        if item.is_symlink():
            _fail('programme symlink refuses control publication')
        if item.is_file():
            size=item.stat().st_size;total+=size
            if item.is_relative_to(root/PREFIX) or item == root/POINTER:controls+=size
    if controls+addition>MAX_CONTROL_STORAGE_BYTES or total+addition>MAX_PROGRAMME_BYTES:
        _fail('cumulative control/programme storage bound exceeded; raw retained')
    if shutil.disk_usage(root).free < addition+METADATA_RESERVE_BYTES:
        _fail('insufficient free space for bounded metadata reserve; raw retained')
    archive._write(root,relative,raw,immutable=immutable)


def pending(root, *, transport=None):
    """Missing fully archived committed receipt groups; no rendered gold read."""
    root=archive._root(root)
    pointer=_pointer(root)
    covered={} if pointer is None else _qualified(root,pointer,transport)['capsules']
    index=archive.load_index(root);result=[]
    for relative in archive.iter_paths(root,'manifests/render-batch-*.json'):
        _receipt_path(relative)
        if relative in covered:continue
        receipt=archive._json(archive.read_bytes(root,relative))
        if receipt.get('split') not in ('train','dev','ood'):
            _fail('sealed or unknown receipt cannot enter controls')
        items=receipt.get('datasets',[])
        if not items:continue
        complete=True
        for item in items:
            path=_manifest_path(item['manifest_path'])
            if path not in index['files']:
                complete=False;break
            for shard in (root/PurePosixPath(path).parent).glob('*.jsonl'):
                shard_relative=str(shard.relative_to(root))
                archive._path(root,shard_relative)
                if shard_relative not in index['files']:
                    complete=False;break
            # A committed dataset may have arbitrary bounded shards. The raw
            # validator checks complete shard custody during bounded derivation;
            # selecting a manifest alone never authorizes eviction.
        if complete:result.append(relative)
    return sorted(result)


def eligible_receipts(root, *, transport=None):
    return pending(root,transport=transport)


def prepare(root, receipt_paths=None, *, max_receipts=MAX_NEW_RECEIPTS, deadline=None, transport=None, lock_fd=None):
    """Create at most four UNTRUSTED immutable metadata capsules; no network write."""
    root=archive._root(root)
    if type(max_receipts) is not int or not 1 <= max_receipts <= MAX_NEW_RECEIPTS:_fail('invalid preparation bound')
    with archive._lock(root,lock_fd),archive.read_session(root,deadline=deadline):
        if transport is not None:_provider(root,transport)
        pointer=_pointer(root)
        previous={} if pointer is None else _qualified(root,pointer)['capsules']
        paths=sorted(set(receipt_paths if receipt_paths is not None else pending(root)))
        results=[]
        for relative in paths:
            _receipt_path(relative)
            if relative in previous:continue
            if len(results)>=max_receipts:break
            capsule=derive(root,relative,deadline=deadline)
            relative_capsule=_capsule_path(capsule['capsule_id'])
            _write(root,relative_capsule,_raw(capsule))
            results.append(relative_capsule)
        return results


def plan_bundle(root, capsule_paths, *, transport=None, lock_fd=None):
    """Aggregate qualified predecessors and <=4 untrusted capsules for backup."""
    root=archive._root(root)
    if not isinstance(capsule_paths,list) or not 1<=len(capsule_paths)<=MAX_NEW_RECEIPTS:_fail('invalid new capsule bound')
    for relative in capsule_paths:
        name=PurePosixPath(archive._relative(relative)).name
        identifier=name.removeprefix('capsule-').removesuffix('.json')
        if relative!=_capsule_path(identifier):_fail('unmanaged capsule staging path')
    with archive._lock(root,lock_fd):
        pointer=_pointer(root)
        old={} if pointer is None else _qualified(root,pointer,transport)['capsules']
        capsules=dict(old)
        for relative in capsule_paths:
            name=PurePosixPath(archive._relative(relative)).name
            identifier=name.removeprefix('capsule-').removesuffix('.json')
            if relative!=_capsule_path(identifier):_fail('unmanaged capsule staging path')
            raw=_read_local(root,relative,MAX_CAPSULE_BYTES);capsule=_capsule(archive._json(raw))
            if relative != _capsule_path(capsule['capsule_id']):_fail('unmanaged capsule staging path')
            path=capsule['receipt_path']
            if path in capsules and capsules[path] != capsule:_fail('changed qualified predecessor')
            capsules[path]=capsule
        body={'version':BUNDLE_VERSION,'repository':archive.REPOSITORY,'programme':archive.PROGRAMME,
              'predecessor':pointer,'capsules':capsules}
        bundle=_bundle({**body,'bundle_id':archive._digest(body)})
        slot=1 if pointer is not None and pointer['bundle_path']==BUNDLE_SLOTS[0] else 0
        relative=_bundle_path(bundle['bundle_id'],slot)
        _write(root,relative,_raw(bundle),immutable=False)
        return {'bundle_path':relative,'bundle_id':bundle['bundle_id'],'new_receipts':len(set(capsules)-set(old)),
                'qualified_receipts':len(capsules),'trusted':False}


def activate(root, bundle_path, control_commit, *, transport=None, deadline=None, lock_fd=None, publication_guard=None):
    """Revalidate new raw derivations, prove stored snapshot, then install authority.

    A restarted publisher rederives the bounded new set. No self-hashed staged
    capsule or database status can substitute for that derivation.
    """
    root=archive._root(root)
    if publication_guard is not None and not callable(publication_guard):_fail('invalid publication guard')
    def guard():
        _check(root,deadline)
        if publication_guard is not None and publication_guard() is False:_fail('publication guard refused')
    if bundle_path not in BUNDLE_SLOTS:_fail('unmanaged bundle publication slot')
    with archive._lock(root,lock_fd),archive.read_session(root,deadline=deadline):
        _check(root,deadline)
        if transport is not None:_provider(root,transport)
        raw=_read_local(root,bundle_path,MAX_BUNDLE_BYTES);bundle=_bundle(archive._json(raw))
        if bundle_path not in BUNDLE_SLOTS:_fail('unmanaged bundle publication slot')
        pointer=_pointer(root)
        if pointer is not None and pointer['bundle_path']==bundle_path:
            if pointer['sha256']!=hashlib.sha256(raw).hexdigest() or pointer['commit']!=control_commit:
                _fail('active slot cannot be replaced by staging')
            _qualified(root,pointer,transport,fresh=True)
            guard()
            return pointer
        if bundle['predecessor'] != pointer:_fail('stale or missing qualified predecessor')
        old={} if pointer is None else _qualified(root,pointer,transport,fresh=True)['capsules']
        if any(bundle['capsules'].get(k)!=v for k,v in old.items()):_fail('qualified predecessor changed or removed')
        new=sorted(set(bundle['capsules'])-set(old))
        if not 1<=len(new)<=MAX_NEW_RECEIPTS:_fail('activation exceeds bounded new derivations')
        for relative in new:
            if derive(root,relative,deadline=deadline) != bundle['capsules'][relative]:
                _fail('prepared projection differs from full raw derivation')
        body={'version':AUTHORITY_VERSION,'repository':archive.REPOSITORY,'programme':archive.PROGRAMME,
              'bundle_path':bundle_path,'commit':control_commit,'archive_ref':archive.archive_ref(control_commit),
              'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':archive.git_blob_sha1(raw),'bytes':len(raw)}
        authority=_authority({**body,'authority_id':archive._digest(body)})
        _qualified(root,authority,transport,fresh=True);guard()
        # Do not replace a concurrent or changed predecessor.
        if _pointer(root)!=pointer:_fail('publication predecessor changed')
        _write(root,PREFIX+'authority-'+authority['authority_id']+'.json',_raw(authority))
        guard()
        if _pointer(root)!=pointer:_fail('publication predecessor changed before pointer switch')
        _write(root,POINTER,_raw(authority),immutable=False)
        return authority


def try_projection(root, relative, receipt, batch):
    """Return equivalent validator inputs, or None for an unattested receipt."""
    root=archive._root(root);relative=_receipt_path(relative)
    pointer=_pointer(root)
    if pointer is None:return None
    bundle=_qualified(root,pointer)
    capsule=bundle['capsules'].get(relative)
    if capsule is None:return None
    if (capsule['receipt_sha256']!=hashlib.sha256(archive.read_bytes(root,relative)).hexdigest()
            or capsule['batch_sha256']!=archive._digest(batch)
            or capsule['spec_sha256']!=receipt['spec_sha256']
            or capsule['code_sha256']!=archive._digest(receipt['code_hashes'])
            or capsule['source_code_sha256']!=archive._digest(batch['code_hashes'])
            or capsule['protected_index_sha256']!=archive._digest(batch['protected_index'])
            or len(capsule['datasets'])!=len(receipt['datasets'])):
        _fail('current receipt/source/specification binding changed')
    index=archive.load_index(root);manifests=[]
    for item,dataset in zip(receipt['datasets'],capsule['datasets']):
        fields=dataset['fields'];source=dataset['source']
        if (fields['manifest_path']!=item['manifest_path'] or fields['eligible_source_records']!=item['eligible_source_records']
                or dataset['manifest_custody']!=_descriptor(index,item['manifest_path'])
                or dataset['manifest_custody']['sha256']!=item['manifest_sha256']
                or dataset['code_sha256']!=archive._digest(receipt['code_hashes'])
                or source['source_code_sha256']!=archive._digest(batch['code_hashes'])
                or source['protected_index_sha256']!=archive._digest(batch['protected_index'])):
            _fail('dataset derivation/custody binding changed')
        shards=[]
        for shard in dataset['shards']:
            if shard['custody']!=_descriptor(index,shard['path']):_fail('shard custody binding changed')
            shards.append({k:shard[k] for k in ('path','sha256','bytes','rows')})
        manifests.append({**fields,'code_hashes':receipt['code_hashes'],
                          'sources':[{**{k:source[k] for k in ('path','sha256','source_batch_manifest','source_batch_manifest_sha256')},
                                      'source_code_hashes':batch['code_hashes'],'protected_index':batch['protected_index']}],
                          'shards':shards})
    return manifests
