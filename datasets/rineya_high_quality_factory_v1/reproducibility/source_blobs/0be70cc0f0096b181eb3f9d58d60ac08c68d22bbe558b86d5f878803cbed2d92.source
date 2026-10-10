"""Preserve exact oversized diagnostic bytes as immutable gzip plus manifest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from factory.common import write_json
from reproducibility.verify_lossless_evidence import verify


def pack(path):
    path=path.resolve()
    assert path.is_file() and path.is_relative_to(ROOT)
    raw=path.read_bytes();payload=gzip.compress(raw,mtime=0)
    packed=path.with_name(path.name+'.gz')
    manifest=path.with_name(path.stem+'.compression_manifest.json')
    if packed.exists():assert packed.read_bytes()==payload,'Frozen compressed bytes differ'
    else:packed.write_bytes(payload)
    metadata={'packer_execution':'/root','raw_path':str(path),'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'raw_bytes':len(raw),'gzip_path':str(packed),'gzip_sha256':hashlib.sha256(payload).hexdigest(),
        'gzip_bytes':len(payload),'deterministic_gzip_mtime':0,'raw_preserved':True,
        'decompressed_bytes_equal':gzip.decompress(payload)==raw}
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        assert all(previous[key]==metadata[key] for key in ['raw_sha256','raw_bytes','gzip_sha256','gzip_bytes'])
    else:write_json(manifest,metadata)
    verify(manifest)
    return {'manifest':str(manifest.relative_to(ROOT)),'raw_bytes':len(raw),'gzip_bytes':len(payload)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('files',nargs='+',type=Path)
    args=parser.parse_args();results=[pack(path) for path in args.files]
    print(json.dumps(results))


if __name__=='__main__':main()
