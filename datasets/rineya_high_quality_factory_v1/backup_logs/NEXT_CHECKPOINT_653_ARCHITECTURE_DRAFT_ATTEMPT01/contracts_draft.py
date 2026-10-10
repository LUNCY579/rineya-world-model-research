"""UNEXECUTED architecture draft. No main, I/O, imports of owned code or authorization.
Later implementation must freeze these source bytes/tests/config before first import.
"""
import re
AUTHOR='/root/backup_manager'
PARENT='653ba70f1f7035a5aed137102c14a1ff29d1cf99'
PRIOR_SPEC_SHA='6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'
PREFIX='datasets/rineya_high_quality_factory_v1'
CAP=25*1024*1024;CHUNK=65536;SPOOL=8*1024*1024;WORKERS=2;RESERVE=512*1024*1024
DRAFT_NOT_SOURCE_APPROVED_OR_EXECUTABLE=True

def require(ok,message):
    if not ok:raise ValueError(message)

def literal(name):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name and not any(c in name for c in '\x00\r\n'),'Invalid literal path')
    require(all(v not in ('','.','..','.git') for v in name.split('/')),'Path alias forbidden')
    return name

def integer(value):return type(value) is int and value>=0

def artifact(row,git=False):
    required={'path','size','sha256'}|({'git_blob'} if git else set())
    require(type(row) is dict and set(row)==required,'Exact artifact row keys required')
    literal(row['path']);require(integer(row['size']) and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Invalid size/SHA256')
    if git:require(type(row['git_blob']) is str and re.fullmatch('[0-9a-f]{40}',row['git_blob']),'Actual streaming GitSHA1 required')
    return row

def indexed_rows(rows,git=False):
    require(type(rows) is list and rows,'Nonempty exact paths required');out={}
    for row in rows:
        artifact(row,git);require(row['path'] not in out,'Duplicate path');out[row['path']]=row
    return out

def derive_scope(spec,previous):
    """Future SPEC must contain actual per-file GitSHA1 from source inventory streams."""
    files=indexed_rows(spec['files'],True);prior=indexed_rows(previous['files'])
    require(len(prior)==3158 and set(prior)<=set(files),'All653 parent3158 tracked paths required')
    require(all(v['size']<=CAP for v in files.values()),'Capped file exceeds25MiB')
    raws=spec['raw_size_exclusions'];require(type(raws) is list,'Raw mapping list required')
    seen=set();pairs={};refs=[];raw_bytes=0;local_bytes=0
    for raw in raws:
        artifact({k:raw[k] for k in ('path','size','sha256')});require(raw['size']>CAP and raw['path'] not in files and raw['path'] not in seen,'Invalid raw cap scope');seen.add(raw['path']);raw_bytes+=raw['size']
        require(type(raw['lossless_representations']) is list and raw['lossless_representations'],'Every raw needs actual gzip/manifest bytes')
        perraw=set()
        for rep in raw['lossless_representations']:
            require(type(rep) is dict and set(rep)=={'gzip','manifest'},'Exact representation schema required')
            for key in ('gzip','manifest'):
                artifact(rep[key]);value=files.get(rep[key]['path']);require(value is not None and all(value[k]==rep[key][k] for k in ('path','size','sha256')),'Representation not an actual capped inventory payload')
            identity=(rep['gzip']['path'],rep['manifest']['path']);require(identity not in perraw,'Duplicate same perraw representation');perraw.add(identity)
            key=(raw['path'],rep['gzip']['path']);require(key not in pairs or pairs[key]==rep['gzip'],'Pair crosswire');pairs[key]=rep['gzip']
            refs.append((raw['path'],rep['gzip']['path'],rep['manifest']['path']));local_bytes+=raw['size']
    # Historical/current relocations are separately exact hash-bound; path retention alone is insufficient.
    priorraw={e['path'] for e in previous['raw_size_exclusions']};require(priorraw<=seen,'Previous31 raw paths cannot silently disappear')
    return {'files':len(files),'capped_bytes':sum(e['size'] for e in files.values()),'source_git_groups':len({e['git_blob'] for e in files.values()}),
            'raw_paths':len(seen),'unique_raw_bytes':raw_bytes,'local_decode_iterations':len(refs),'local_decoded_bytes':local_bytes,
            'raw_gzip_pairs':len(pairs),'remote_decoded_bytes':sum(next(e['size'] for e in raws if e['path']==key[0]) for key in pairs),
            'global_manifest_paths':len({v[2] for v in refs}),'perraw_manifest_references':len(refs),
            'unique_gzip_paths':len({v['path'] for v in pairs.values()})}

def complete_freeze(freeze,mandatory,current):
    """Caller must validate supplied freeze SHA before ANY dependency import."""
    require(freeze.get('frozen_before_first_import_test_execution') is True,'Preimport source freeze missing')
    rows=indexed_rows(freeze['frozen_inputs']);require(set(mandatory)<=set(rows),'Mandatory source/test/config dependency omitted')
    for name,wanted in mandatory.items():require(rows[name]==wanted,'Mandatory typed artifact substituted')
    for name,row in rows.items():require(current(name)==row,'Frozen declared bytes changed')
    return True

def strict_delta_diff(observation):
    require(type(observation) is dict and observation=={'returncode':0,'stdout':b'','stderr':b''},'New parent653 delta must be strict exit0; historical six warnings grant no exception')
    return {'status':'STRICT_PASS','actual_exit_code':0,'new_whitespace_exceptions':0}

def assert_not_executable():raise RuntimeError('ARCHITECTURE_DRAFT_ONLY_FUTURE_SOURCE_APPROVAL_FREEZE_AND_TESTS_REQUIRED')
