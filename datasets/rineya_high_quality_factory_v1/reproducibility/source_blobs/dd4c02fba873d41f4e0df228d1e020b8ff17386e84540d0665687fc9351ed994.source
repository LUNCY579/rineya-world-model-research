"""Stream exact raw/remote-gzip equality for a frozen checkpoint inventory."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'LUNCY579/rineya-world-model-research'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_entry(entry, receipt):
    raw = ROOT / entry['path']
    assert raw.is_file() and not raw.is_symlink()
    representation = entry['lossless_representations'][0]
    compressed = representation['gzip']
    manifest = representation['manifest']
    assert sha(ROOT / manifest['path']) == manifest['sha256']
    assert sha(ROOT / compressed['path']) == compressed['sha256']
    candidates = [blob for blob in receipt['remote_verification']['blob_receipts']
                  if compressed['path'] in blob['paths']]
    assert len(candidates) == 1 and candidates[0]['sha256'] == compressed['sha256']
    blob = candidates[0]['git_blob']
    response = subprocess.run(['gh', 'api', f'repos/{REPOSITORY}/git/blobs/{blob}'],
                              check=True, capture_output=True)
    remote = json.loads(response.stdout)
    assert remote['encoding'] == 'base64' and remote['sha'] == blob
    payload = base64.b64decode(remote['content'])
    assert len(payload) == compressed['size']
    assert hashlib.sha256(payload).hexdigest() == compressed['sha256']
    digest = hashlib.sha256()
    length = 0
    with raw.open('rb') as source, gzip.GzipFile(fileobj=io.BytesIO(payload)) as decoded:
        while True:
            chunk = decoded.read(1024 * 1024)
            if not chunk:
                assert source.read(1) == b''
                break
            assert source.read(len(chunk)) == chunk, 'Remote decode differs from frozen raw bytes'
            digest.update(chunk)
            length += len(chunk)
    assert length == entry['size'] and digest.hexdigest() == entry['sha256']
    return {'raw_path': entry['path'], 'raw_bytes': length, 'raw_sha256': digest.hexdigest(),
            'gzip_path': compressed['path'], 'gzip_sha256': compressed['sha256'],
            'remote_git_blob': blob, 'manifest_path': manifest['path'],
            'manifest_sha256': manifest['sha256'], 'exact_remote_decoded_bytes_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Evidence output is immutable'
    started = datetime.now(timezone.utc).isoformat()
    spec = json.loads(args.spec.read_text())
    receipt = json.loads(args.receipt.read_text())
    assert receipt['status'] == 'BACKED_UP'
    assert receipt['commit_sha'] == receipt['remote_commit_sha']
    metadata = json.loads(subprocess.run(['gh', 'api', f'repos/{REPOSITORY}'],
                                        check=True, capture_output=True).stdout)
    assert metadata['private'] and metadata['permissions']['admin']
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda entry: verify_entry(entry, receipt), spec['raw_size_exclusions']))
    report = {'author_execution': '/root', 'runner_execution': '/root', 'pass': True,
              'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'commit_sha': receipt['commit_sha'], 'remote_commit_sha': receipt['remote_commit_sha'],
              'spec_sha256': sha(args.spec), 'backup_receipt_sha256': sha(args.receipt),
              'verification_source_sha256': sha(Path(__file__)), 'repository_private': True,
              'artifacts': results, 'raw_bytes_preserved': sum(item['raw_bytes'] for item in results),
              'scope': 'Actual remote gzip bytes streamed against every oversized frozen raw file; no data approval'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': True, 'artifacts': len(results), 'raw_bytes': report['raw_bytes_preserved'],
                      'commit_sha': report['commit_sha']}))


if __name__ == '__main__':
    main()
