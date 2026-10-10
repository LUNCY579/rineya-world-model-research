"""Losslessly archive frozen oversized evidence outside immutable releases.

The raw files are never edited or deleted. Only explicitly bound, completed
artifacts may be passed in the source freeze. This does not grant data approval.
"""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CAP = 25 * 1024 * 1024
CHUNK = 65536


def sha_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def safe(name):
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    path = ROOT / rel
    assert path.resolve().is_relative_to(ROOT.resolve())
    assert not any(p.is_symlink() for p in [path, *path.parents])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    args = parser.parse_args()
    freeze_path = safe(args.freeze)
    assert sha_file(freeze_path)['sha256'] == args.expected_freeze_sha256
    frozen = json.loads(freeze_path.read_bytes())
    assert sha_file(Path(__file__)) == frozen['source']
    assert not safe(frozen['receipt']).exists()
    for item in frozen['artifacts']:
        assert sha_file(safe(item['raw_path'])) == item['raw']
        assert item['raw']['bytes'] > CAP
        assert '/releases/' not in item['gzip_path'] and '/releases/' not in item['manifest_path']
        assert not safe(item['gzip_path']).exists() and not safe(item['manifest_path']).exists()
    results = []
    for item in frozen['artifacts']:
        raw_path = safe(item['raw_path'])
        packed = safe(item['gzip_path'])
        manifest = safe(item['manifest_path'])
        packed.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        observed = hashlib.sha256()
        count = 0
        with raw_path.open('rb') as src, packed.open('xb') as target:
            with gzip.GzipFile(filename='', fileobj=target, mode='wb', compresslevel=9, mtime=0) as out:
                for chunk in iter(lambda: src.read(CHUNK), b''):
                    observed.update(chunk)
                    count += len(chunk)
                    out.write(chunk)
        assert {'bytes': count, 'sha256': observed.hexdigest()} == item['raw']
        packed_hash = sha_file(packed)
        assert packed_hash['bytes'] <= CAP, 'Compressed artifact exceeds repository cap; preserve bytes and use a fresh shard implementation'
        decoded = hashlib.sha256()
        decoded_size = 0
        with raw_path.open('rb') as src, gzip.open(packed, 'rb') as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b''):
                assert src.read(len(chunk)) == chunk, 'Lossless byte comparison failed'
                decoded.update(chunk)
                decoded_size += len(chunk)
            assert src.read(1) == b'', 'Decoded stream ended early'
        assert {'bytes': decoded_size, 'sha256': decoded.hexdigest()} == item['raw']
        assert sha_file(raw_path) == item['raw'], 'Raw source changed during packing'
        metadata = {
            'actual_author_execution': '/root', 'actual_runner_execution': '/root',
            'raw_path': item['raw_path'], 'raw_bytes': count, 'raw_sha256': observed.hexdigest(),
            'gzip_path': item['gzip_path'], 'gzip_bytes': packed_hash['bytes'], 'gzip_sha256': packed_hash['sha256'],
            'codec': 'gzip', 'parameters': {'compresslevel': 9, 'mtime': 0, 'filename': ''},
            'stream_chunk_bytes': CHUNK, 'actual_full_byte_roundtrip_equal': True,
            'raw_preserved_unchanged': True, 'immutable_release_directory_untouched': True,
            'source_sha256': frozen['source']['sha256'], 'source_freeze_sha256': args.expected_freeze_sha256,
            'data_approval': False, 'remote_backup_claim': False,
        }
        with manifest.open('x') as out:
            json.dump(metadata, out, sort_keys=True, indent=2)
            out.write('\n')
        results.append({'raw': item['raw'], 'raw_path': item['raw_path'],
                        'gzip': packed_hash, 'gzip_path': item['gzip_path'],
                        'manifest': sha_file(manifest), 'manifest_path': item['manifest_path']})
    for item in frozen['artifacts']:
        assert sha_file(safe(item['raw_path'])) == item['raw']
    assert sha_file(Path(__file__)) == frozen['source']
    receipt = {'actual_runner_execution': '/root', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': 'PASS_EXACT_LOCAL_LOSSLESS_PACKING_RAW_AND_RELEASES_UNCHANGED',
               'source': frozen['source'], 'source_freeze_sha256': args.expected_freeze_sha256,
               'artifacts': results, 'actual_packed_artifacts': len(results),
               'data_approval': False, 'remote_backup_claim': False}
    with safe(frozen['receipt']).open('x') as out:
        json.dump(receipt, out, sort_keys=True, indent=2)
        out.write('\n')
    print(json.dumps({'status': receipt['status'], 'packed_artifacts': len(results)}))


if __name__ == '__main__':
    main()
