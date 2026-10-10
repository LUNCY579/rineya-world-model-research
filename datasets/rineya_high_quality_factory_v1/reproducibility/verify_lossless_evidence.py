"""Verify oversized evidence's actual lossless bytes, including remote gzip blobs.

Raw originals stay local. A gzip payload and manifest preserve their exact bytes
within the repository cap. Remote mode consumes a byte-verified checkpoint receipt.
Actual author and runner are recorded as /root; no independent reviewer claim.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backup_logs.checkpoint_backup import SECRET_PATTERNS, REPOSITORY
from factory.common import write_json


def local_path(value):
    path=Path(value)
    if not path.is_absolute():path=ROOT/path
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Evidence path outside factory or symlink')
    return path


def verify(manifest,receipt=None):
    metadata=json.loads(manifest.read_text())
    raw=local_path(metadata['raw_path']);packed=local_path(metadata['gzip_path'])
    raw_bytes=raw.read_bytes()
    assert len(raw_bytes)==metadata['raw_bytes']
    assert hashlib.sha256(raw_bytes).hexdigest()==metadata['raw_sha256']
    assert not any(pattern.search(raw_bytes) for pattern in SECRET_PATTERNS),'Potential secret in uncompressed evidence; content withheld'
    payload=packed.read_bytes()
    assert len(payload)==metadata['gzip_bytes']
    assert hashlib.sha256(payload).hexdigest()==metadata['gzip_sha256']
    relative=packed.relative_to(ROOT).as_posix()
    result={'manifest':manifest.relative_to(ROOT).as_posix(),'gzip_path':relative,
        'raw_bytes':len(raw_bytes),'raw_sha256':metadata['raw_sha256'],
        'gzip_bytes':len(payload),'gzip_sha256':metadata['gzip_sha256'],'uncompressed_secret_scan':'PASS'}
    if receipt is not None:
        assert receipt['status']=='BACKED_UP' and receipt['commit_sha']==receipt['remote_commit_sha']
        entries=[b for b in receipt['remote_verification']['blob_receipts'] if relative in b['paths']]
        assert len(entries)==1 and entries[0]['sha256']==metadata['gzip_sha256']
        blob=entries[0]['git_blob']
        process=subprocess.run(['gh','api',f'repos/{REPOSITORY}/git/blobs/{blob}'],capture_output=True,check=True)
        remote=json.loads(process.stdout)
        assert remote['encoding']=='base64'
        payload=base64.b64decode(remote['content'])
        assert len(payload)==metadata['gzip_bytes'] and hashlib.sha256(payload).hexdigest()==metadata['gzip_sha256']
        result.update(remote_commit=receipt['commit_sha'],remote_git_blob=blob,verification_source='Actual remote GitHub blob decompressed')
    else:result['verification_source']='Actual local gzip payload decompressed; remote verification pending'
    decoded=gzip.decompress(payload)
    assert decoded==raw_bytes and hashlib.sha256(decoded).hexdigest()==metadata['raw_sha256']
    result['exact_decoded_bytes_match']=True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('manifests',nargs='+',type=Path)
    args=parser.parse_args()
    receipt=json.loads(args.receipt.read_text()) if args.receipt else None
    results=[verify(path.resolve(),receipt) for path in args.manifests]
    report={'author_execution':'/root','runner_execution':'/root','artifacts':results,'pass':True,
        'scope':'Complete raw evidence preserved through exact lossless payloads; not dataset/reviewer approval'}
    write_json(args.output,report)
    print(json.dumps({'artifacts':len(results),'pass':True,'remote_decompressed':receipt is not None}))


if __name__=='__main__':main()
