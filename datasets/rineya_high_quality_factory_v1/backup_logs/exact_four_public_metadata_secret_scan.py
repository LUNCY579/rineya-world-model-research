#!/usr/bin/env python3
"""NEW exact four-file public-metadata false-positive scanner; every regex unchanged.
No source rewrite, credential output, Git mutation, or generic lexical exception.
Only fully size/SHA verified exact factory paths may admit one exact index/span/digest.
"""
from __future__ import annotations
import base64,hashlib,json,os,re
from pathlib import Path
FACTORY=Path(__file__).resolve().parent.parent
CONFIG=FACTORY/'backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json'
CONFIG_SHA='2e18c2c156aaefcee75ad6ad92fe51c66abf864150924d5ac1cd26dbeed5aa05'
ORIGINAL_SHA='91bb1b946001e425954414c1dcf644c73fe6557471fcc8aaf175d5e87aa8e9ae'
CHUNK=65536
ALLOWED_OBSERVATIONS={}
EXPECTED_PATTERN_SPECS=[(b'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})', 0), (b'sk-(?:proj-)?[A-Za-z0-9_-]{30,}', 0), (b'AKIA[A-Z0-9]{16}', 0), (b'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', 0), (b'(?im)^\\s*(?:api_key|access_token|password|secret)\\s*[=:]\\s*[\\"\']?[A-Za-z0-9_+/=-]{16,}', 10)]
EXPECTED_FILES=[{'matches': [{'end': 599, 'match_sha256': '1b53660ed742f53d3f392360f2d04874ba5ff41741093539f9dd4a86296e4de1', 'pattern_index': 1, 'preceding_literal_ta': True, 'start': 565}], 'path': 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/FINAL_RECEIPT.json', 'sha256': '1513eb28199054bfc023e462f5fde2d10f3df24160cca79bc489e97c124b9d04', 'size': 13530}, {'matches': [{'end': 502, 'match_sha256': '1b53660ed742f53d3f392360f2d04874ba5ff41741093539f9dd4a86296e4de1', 'pattern_index': 1, 'preceding_literal_ta': True, 'start': 468}], 'path': 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-execution.json', 'sha256': '2ebb35681afa918e8e969f7db1f16b74d1f8bcf8190094d05da2ba634b537341', 'size': 901}, {'matches': [{'end': 1248, 'match_sha256': '1b53660ed742f53d3f392360f2d04874ba5ff41741093539f9dd4a86296e4de1', 'pattern_index': 1, 'preceding_literal_ta': True, 'start': 1214}], 'path': 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-source-blobs/c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e.source', 'sha256': 'c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e', 'size': 2075}, {'matches': [{'end': 1248, 'match_sha256': '1b53660ed742f53d3f392360f2d04874ba5ff41741093539f9dd4a86296e4de1', 'pattern_index': 1, 'preceding_literal_ta': True, 'start': 1214}], 'path': 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/run_task_solver_attempt01.py', 'sha256': 'c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e', 'size': 2075}]


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        while b:=s.read(CHUNK):h.update(b)
    return h.hexdigest()

def validate_config():
    if CONFIG.is_symlink() or checksum(CONFIG)!=CONFIG_SHA:raise ValueError('Exact four-file exception config changed')
    c=json.loads(CONFIG.read_bytes())
    if c['files']!=EXPECTED_FILES or c['existing_regexes_unchanged'] is not True:raise ValueError('Canonical fixed exception config differs')
    return c

def normalized_name(path):
    literal=os.fspath(path);path=Path(path)
    if literal!=str(path):raise ValueError("Literal path alias rejected")
    if not path.is_absolute() or str(path)!=str(FACTORY/path.relative_to(FACTORY)):
        raise ValueError('Scanner needs literal normalized factory path')
    parts=path.relative_to(FACTORY).parts
    if any(p in ('.','..','.git') for p in parts) or any(FACTORY.joinpath(*parts[:i]).is_symlink() for i in range(1,len(parts)+1)):
        raise ValueError('Path alias/symlink rejected')
    if not path.resolve().is_relative_to(FACTORY.resolve()):raise ValueError('Scanner path outside factory')
    return path.relative_to(FACTORY).as_posix()

def bound_row(name):
    validate_config()
    return next((r for r in EXPECTED_FILES if r['path']==name),None)

def scan_stream(handle,size,patterns,row=None,chunk=CHUNK):
    if type(size) is not int or size<0 or type(chunk) is not int or not 1<=chunk<=CHUNK:raise ValueError('Invalid bounded stream arguments')
    if (type(patterns) is not list or len(patterns)!=len(EXPECTED_PATTERN_SPECS)
            or any(not isinstance(p,re.Pattern) or (p.pattern,p.flags)!=wanted for p,wanted in zip(patterns,EXPECTED_PATTERN_SPECS))):
        raise ValueError("Every original secret regex and flags must remain unchanged")
    def exact(a,b):
        if type(a) is not type(b):return False
        if type(b) is dict:return set(a)==set(b) and all(exact(a[k],b[k]) for k in b)
        if type(b) is list:return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a==b
    if row is not None and not any(exact(row,r) for r in EXPECTED_FILES):raise ValueError("Untrusted exception row")
    if row is not None and size!=row["size"]:raise ValueError("Exact metadata size differs before scan")
    h=hashlib.sha256();blob=hashlib.sha1(f'blob {size}\0'.encode());count=0;tail=b'';seen={}
    expected=None if row is None else row['matches'][0]
    def inspect(buffer,offset,final):
        for index,pattern in enumerate(patterns):
            for match in pattern.finditer(buffer):
                start,end=offset+match.start(),offset+match.end()
                # Greedy matching at a chunk end is provisional. Unknown matches reject
                # immediately; the only expected34-byte match may wait for termination.
                if row is None or index!=expected['pattern_index'] or start!=expected['start']:
                    raise ValueError('Potential secret; content withheld')
                if match.end()==len(buffer) and not final:
                    if end>expected['end']:raise ValueError('Exception token extends beyond exact bound span')
                    continue
                digest=hashlib.sha256(match.group()).hexdigest()
                preceding=match.start()>=2 and buffer[match.start()-2:match.start()]==b'ta'
                actual={'pattern_index':index,'start':start,'end':end,'match_sha256':digest,'preceding_literal_ta':preceding}
                if actual!=expected:raise ValueError('Secret match differs from exact public-metadata index/span/digest/context')
                key=(index,start,end,digest)
                seen[key]=actual # identical overlap rediscovery never becomes a second match
    while True:
        part=handle.read(chunk)
        if not part:break
        if count+len(part)>size:raise ValueError('Payload exceeds frozen size')
        h.update(part);blob.update(part)
        buffer=tail+part;offset=count-len(tail);count+=len(part)
        inspect(buffer,offset,False);tail=buffer[-4096:]
    inspect(tail,count-len(tail),True)
    if count!=size:raise ValueError('Payload shorter than frozen size')
    result={'size':count,'sha256':h.hexdigest(),'git_blob':blob.hexdigest()}
    observations=sorted(seen.values(),key=lambda x:(x['pattern_index'],x['start'],x['end']))
    if row is not None:
        if result['size']!=row['size'] or result['sha256']!=row['sha256'] or observations!=row['matches']:
            raise ValueError('Exact metadata full size/SHA/match-count differs')
    elif observations:raise ValueError('Unbound secret observation')
    return result,observations

def proof(path,patterns,chunk=CHUNK):
    name=normalized_name(path);row=bound_row(name);before=path.stat()
    with path.open('rb') as s:result,observations=scan_stream(s,before.st_size,patterns,row,chunk)
    after=path.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('Source changed during exact metadata scan')
    if row is not None:
        ALLOWED_OBSERVATIONS[name]={'path':name,'size':result['size'],'sha256':result['sha256'],'matches':observations,'status':'EXACT_PUBLIC_METADATA_FALSE_POSITIVE_BOUND','full_size_and_sha_verified_before_acceptance':True}
    return result

def report():
    return {'status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','four_exception_files':list(ALLOWED_OBSERVATIONS.values()),
            'allowed_exact_match_detections':len(ALLOWED_OBSERVATIONS),'unbound_secret_detections':0,'existing_regexes_unchanged':True,'configuration_sha256':CONFIG_SHA}
