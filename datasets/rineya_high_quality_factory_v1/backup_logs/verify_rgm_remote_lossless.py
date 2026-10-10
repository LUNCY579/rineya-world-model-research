"""Actually fetch frozen remote gzip blobs and compare decoded original bytes."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from backup_logs import recover_remote_checkpoint as bounded

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '448c36af9370460782da3c0a4ba7c5ca4d8ba71a'
SPEC_SHA = 'd669b0b151a26e51892c7b69281cf1215a924835a8a735018d2ef5adfa553ed5'
HELPER_SHA = '3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1'
OUT = ROOT / 'quality_reports/RGM_V1_FINAL_REVIEW_REMOTE_LOSSLESS.json'


def main():
    assert not OUT.exists(), 'Preserve existing actual remote proof'
    assert bounded.hash_file(Path(bounded.__file__))['sha256'] == HELPER_SHA
    spec_path = ROOT / 'backup_logs/rgm-v1-final-review_SPEC.json'
    assert bounded.hash_file(spec_path)['sha256'] == SPEC_SHA
    spec = json.loads(spec_path.read_text())
    stage_path = ROOT / 'backup_logs/rgm-v1-final-review_STAGE_RECOVERY_V3.json'
    stage = json.loads(stage_path.read_text())
    assert stage['status'] == 'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING'
    assert stage['remote_commit_sha'] == COMMIT and stage['all1801_index_blobs_match_frozen_bytes']
    metadata = bounded.api(f'repos/{bounded.REPOSITORY}')
    assert metadata['private'] and metadata['permissions']['admin']
    remote = bounded.api(f'repos/{bounded.REPOSITORY}/git/commits/{COMMIT}')
    tree = bounded.api(f'repos/{bounded.REPOSITORY}/git/trees/{remote["tree"]["sha"]}?recursive=1')
    assert not tree.get('truncated')
    blobs = {entry['path']: entry for entry in tree['tree'] if entry['type'] == 'blob'}
    inventory = {entry['path']: entry for entry in spec['files']}
    proofs = []
    for entry in spec['raw_size_exclusions']:
        representation = entry['lossless_representations'][0]
        packed = representation['gzip']
        assert inventory[packed['path']] == packed
        blob = blobs[bounded.DESTINATION + '/' + packed['path']]
        assert blob['size'] == packed['size'] and blob['mode'] == '100644'
        raw_path = bounded.source_path(ROOT, entry['path'])
        before = raw_path.stat()

        def consume(handle):
            # Bounded compressed spool; decode raw chunks directly against local
            # original bytes. No complete raw artifact resides in memory.
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as spool:
                compressed_sha = hashlib.sha256()
                git_sha = hashlib.sha1(('blob ' + str(packed['size']) + '\0').encode())
                compressed_size = 0
                for chunk in iter(lambda: handle.read(65536), b''):
                    compressed_size += len(chunk)
                    assert compressed_size <= packed['size'], 'Unexpected remote payload size'
                    compressed_sha.update(chunk)
                    git_sha.update(chunk)
                    spool.write(chunk)
                assert compressed_size == packed['size']
                assert compressed_sha.hexdigest() == packed['sha256'] and git_sha.hexdigest() == blob['sha']
                spool.seek(0)
                raw_sha = hashlib.sha256()
                raw_size = 0
                with gzip.GzipFile(fileobj=spool, mode='rb') as decoded, raw_path.open('rb') as original:
                    for chunk in iter(lambda: decoded.read(65536), b''):
                        raw_size += len(chunk)
                        assert raw_size <= entry['size'] and chunk == original.read(len(chunk)), 'Remote decode differs from original bytes'
                        raw_sha.update(chunk)
                    assert original.read(1) == b''
                assert raw_size == entry['size'] and raw_sha.hexdigest() == entry['sha256']
                return {'raw_path': entry['path'], 'raw_bytes': raw_size, 'raw_sha256': raw_sha.hexdigest(),
                        'remote_gzip_path': packed['path'], 'remote_git_blob': blob['sha'],
                        'remote_gzip_bytes': compressed_size, 'remote_gzip_sha256': compressed_sha.hexdigest(),
                        'actual_remote_fetch_this_run': True, 'decoded_byte_for_byte_equal_to_preserved_raw': True}

        proof = bounded.process_stream(['gh', 'api', '-H', 'Accept: application/vnd.github.raw+json',
                                        '-H', 'Cache-Control: no-cache',
                                        f'repos/{bounded.REPOSITORY}/git/blobs/{blob["sha"]}'], consume)
        after = raw_path.stat()
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        proofs.append(proof)
        print(json.dumps({'actual_remote_lossless_verified': len(proofs), 'required': len(spec['raw_size_exclusions'])}), flush=True)
    assert len(proofs) == 24 and sum(item['raw_bytes'] for item in proofs) == 1188634377
    result = {'actual_runner_execution': '/root', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS_ACTUAL_REMOTE_GZIP_DECODE', 'commit_sha': COMMIT, 'spec_sha256': SPEC_SHA,
              'source_sha256': bounded.hash_file(Path(__file__))['sha256'], 'dependency_sha256': HELPER_SHA,
              'actual_remote_tree_sha': remote['tree']['sha'], 'actual_remote_gzip_proofs': proofs,
              'oversized_raw_artifacts': len(proofs), 'raw_bytes_exactly_recovered': sum(item['raw_bytes'] for item in proofs),
              'global_inventory_remote_bytes_verification': 'Separate1801file1497blobreceipt must pass beforeoverallBACKED_UP',
              'scope': 'Actual fetched remote gzip bytes decoded byte-for-byte; no data approval or old receipt edits',
              'self_inventory_boundary': 'This verifier and proof were created after the frozen1801filecheckpoint;local until a subsequent checkpoint'}
    with OUT.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'artifacts': len(proofs), 'raw_bytes': result['raw_bytes_exactly_recovered'],
                      'proof_sha256': bounded.hash_file(OUT)['sha256']}))


if __name__ == '__main__':
    main()
