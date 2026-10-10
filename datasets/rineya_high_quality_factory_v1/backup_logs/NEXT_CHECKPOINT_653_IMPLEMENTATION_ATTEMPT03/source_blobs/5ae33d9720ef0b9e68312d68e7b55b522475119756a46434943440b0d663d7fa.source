"""Exact new runtime raw text evidence preserved as binary gzip under strict-zero Git delta."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import next653_common as c

SCHEMA='next653-new-runtime-raw-binary-mapping-v2'
REASON='new_immutable_runtime_raw_preserved_as_binary_gzip_for_strict_zero_delta'
ARCHIVE_COMPONENTS={'raw_archives','raw_source_archives','original_byte_archives','preexecution_raw_archives',
                    'raw_preexecution_archives_attempt03','source_blobs','runtime_raw_archives','execution_raw_archives',
                    'raw-archives','runtime-archives','byte_archives'}

def whitespace_evidence(path):
    """Opaque line digests only. Streaming memory does not scale with long lines."""
    before=c.stat_key(path);line=1;h=hashlib.sha256();tail=b'';length=0;rows=[];last_blank=None
    with path.open('rb') as stream:
        while block:=stream.read(c.CHUNK):
            if b'\0' in block:raise ValueError('Mapped new whitespace raw must be text; binary raws use ordinary size representations')
            parts=block.split(b'\n')
            for i,part in enumerate(parts):
                h.update(part);length+=len(part);tail=(tail+part)[-2:]
                if i==len(parts)-1:continue
                h.update(b'\n');end=tail[:-1] if tail.endswith(b'\r') else tail
                if end.endswith((b' ',b'\t')):rows.append({'line':line,'type':'trailing_whitespace','line_sha256':h.hexdigest()})
                last_blank={'line':line,'type':'blank_line_at_eof','line_sha256':h.hexdigest()} if length==0 or (length==1 and tail==b'\r') else None
                line+=1;h=hashlib.sha256();tail=b'';length=0
    if length:
        if tail.endswith((b' ',b'\t')):rows.append({'line':line,'type':'trailing_whitespace','line_sha256':h.hexdigest()})
        last_blank=None
    if last_blank is not None:rows.append(last_blank)
    if before!=c.stat_key(path):raise ValueError('Raw changed during opaque whitespace diagnosis')
    return rows

def mapping_shape(row,previous,admitted):
    if type(row) is not dict or set(row)!={'raw','gzip','manifest','origin','whitespace_evidence'}:
        raise ValueError('Closed exact new raw/gzip/manifest/origin mapping required')
    for role in ('raw','gzip','manifest','origin'):
        if type(row[role]) is not dict or set(row[role])!={'path','size','sha256'}:raise ValueError('Closed mapping byte binding required')
        c.entry(row[role])
    raw,packed,manifest,origin=(row[k] for k in ('raw','gzip','manifest','origin'))
    if (raw['size']>c.CAP or raw['path'] in previous or raw['path']==origin['path']
            or not set(Path(raw['path']).parts)&ARCHIVE_COMPONENTS
            or '/releases/' in '/'+raw['path'] or raw['path'] in c.METADATA_GIT_BLOBS
            or origin['path'] not in admitted or c.projection(admitted[origin['path']])!=origin
            or (raw['size'],raw['sha256'])!=(origin['size'],origin['sha256'])):
        raise ValueError('Only NEW exact raw runtime archive copies of admitted original bytes may map')
    if (packed['size']>c.CAP or manifest['size']>c.CAP
            or not packed['path'].startswith('reproducibility/lossless_archives/') or not packed['path'].endswith('.gz')
            or not manifest['path'].startswith('manifests/lossless_archives/') or not manifest['path'].endswith('.lossless.json')
            or len({raw['path'],packed['path'],manifest['path'],origin['path']})!=4):
        raise ValueError('Exact capped standard binary package outside immutable releases required')
    warnings=row['whitespace_evidence']
    if type(warnings) is not list or not warnings:raise ValueError('Actual nonempty immutable whitespace evidence required')
    seen=set()
    for finding in warnings:
        if (type(finding) is not dict or set(finding)!={'line','type','line_sha256'}
                or type(finding['line']) is not int or finding['line']<=0
                or finding['type'] not in ('trailing_whitespace','blank_line_at_eof')):
            raise ValueError('Typed exact opaque whitespace evidence differs')
        c.sha(finding['line_sha256']);key=(finding['line'],finding['type'])
        if key in seen:raise ValueError('Duplicate whitespace finding')
        seen.add(key)
    return row

def validate(cfg,admission,spec=None):
    binding=cfg['inputs']['archive_mapping_contract'];contract=c.bound_json(binding)
    if (type(contract) is not dict or set(contract)!={'schema','checkpoint_id','expected_parent','raw_mappings','grants_data_approval'}
            or contract['schema']!=SCHEMA or contract['checkpoint_id']!=cfg['checkpoint_id']
            or contract['expected_parent']!=c.PARENT or contract['grants_data_approval'] is not False
            or type(contract['raw_mappings']) is not list):
        raise ValueError('Exact checkpoint-bound raw binary mapping contract required')
    prior=c.fixed_json('prior');oldfiles,oldraw=c.all_entries(prior);previous=set(oldfiles)|set(oldraw)
    admitted={r['path']:r for r in admission['required_factory_original_and_raw_closure']}
    result={}
    if [r.get('raw',{}).get('path') for r in contract['raw_mappings']]!=sorted(r.get('raw',{}).get('path') for r in contract['raw_mappings']):
        raise ValueError('Exact raw mapping rows must have stable literal path order')
    for row in contract['raw_mappings']:
        mapping_shape(row,previous,admitted);raw,packed,manifest=(row[k] for k in ('raw','gzip','manifest'))
        if raw['path'] in result:raise ValueError('Duplicate logical raw archival mapping')
        for role in ('raw','gzip','manifest','origin'):c.validate_binding(row[role])
        if whitespace_evidence(c.safe(c.FACTORY,raw['path']))!=row['whitespace_evidence']:
            raise ValueError('Exact actual opaque raw whitespace lines differ')
        with c.safe(c.FACTORY,packed['path']).open('rb') as stream:header=stream.read(8)
        if header!=b'\x1f\x8b\x08\x00'+b'\0'*4:raise ValueError('Deterministic gzip without filename/mtime required')
        value=json.loads(c.safe(c.FACTORY,manifest['path']).read_bytes())
        expected={'raw_path':raw['path'],'raw_bytes':raw['size'],'raw_sha256':raw['sha256'],
                  'gzip_path':packed['path'],'gzip_bytes':packed['size'],'gzip_sha256':packed['sha256']}
        if type(value) is not dict or any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in expected.items()):
            raise ValueError('Exact standard raw/gzip manifest relation differs')
        c.compare_gzip(c.safe(c.FACTORY,raw['path']),c.safe(c.FACTORY,packed['path']))
        result[raw['path']]=row
    if spec is not None:
        if spec.get('new_runtime_raw_binary_mapping_contract')!=binding or not c.typed_equal(spec.get('new_runtime_raw_binary_mappings'),contract['raw_mappings']):
            raise ValueError('SPEC new raw binary mapping contract differs')
        files,raws=c.all_entries(spec)
        if {p for p,r in raws.items() if r['size']<=c.CAP}!=set(result):raise ValueError('Unbound under-cap source omission rejected')
        for path,row in result.items():
            raw=raws.get(path,{})
            if c.projection(raw)!=row['raw'] or raw.get('reason')!=REASON:raise ValueError('Exact logical raw bytes omitted or relabeled')
            expected_rep={'gzip':row['gzip'],'manifest':row['manifest'],'exact_raw_roundtrip_verified':True,'uncompressed_secret_scan':'PASS'}
            if raw.get('lossless_representations')!=[expected_rep]:raise ValueError('New immutable raw requires its one exact representation')
            for role in ('gzip','manifest'):
                if c.projection(files.get(row[role]['path'],{}))!=row[role]:raise ValueError('Actual tracked binary package absent')
    return result
