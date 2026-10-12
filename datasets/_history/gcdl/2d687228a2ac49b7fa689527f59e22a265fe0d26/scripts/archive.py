#!/usr/bin/env python3
"""Bounded GCDL archive planning, remote custody verification and evacuation.

The backup workflow commits/pushes the planned catalog and creates its immutable
snapshot ref. Verification reads actual GitHub tree/blob bytes. Evacuation requires
the current independent review and user policy, and does not remove Git objects.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from goal_conditioned_decision_data.pipelines import remote_archive as archive


def _control(relative, prefix):
    relative = archive._relative(relative)
    if not relative.startswith(prefix) or not relative.endswith('.json'):
        raise archive.ArchiveError('unapproved archive control path')
    return json.loads(archive.read_text(ROOT, relative))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='prepare a committed-byte catalog; no network or deletion')
    plan.add_argument('--path', action='append', help='explicit eligible programme-relative payload, repeatable')
    plan.add_argument('--commit', help='immutable committed source SHA; defaults to HEAD')
    plan.add_argument('--max-files', type=int, default=archive.MAX_BATCH_FILES)
    plan.add_argument('--max-bytes', type=int, default=archive.MAX_BATCH_BYTES)
    verify = commands.add_parser('verify', help='verify immutable snapshot/tree/raw bytes; keep all payloads')
    verify.add_argument('--catalog', required=True)
    verify.add_argument('--catalog-commit', required=True)
    evict = commands.add_parser('evict', help='reverify all remote bytes, then evacuate only verified local payloads')
    evict.add_argument('--receipt', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            catalog = archive.plan_archive(ROOT, args.path, args.commit, max_files=args.max_files, max_bytes=args.max_bytes)
            result = {'status': 'catalog_planned', 'catalog_path': archive.catalog_path(catalog['catalog_id']),
                      'catalog_id': catalog['catalog_id'], 'files': len(catalog['files']),
                      'bytes': sum(meta['bytes'] for meta in catalog['files'].values()), 'deleted': []}
        elif args.command == 'verify':
            receipt = archive.verify_archive(ROOT, _control(args.catalog, 'manifests/remote_catalogs/'), args.catalog_commit)
            result = {'status': receipt['status'], 'receipt_id': receipt['receipt_id'],
                      'receipt_path': 'manifests/remote_custody/custody-' + receipt['receipt_id'] + '.json',
                      'catalog_commit': receipt['catalog_commit'], 'archive_ref': receipt['archive_ref'], 'deleted': []}
        else:
            from goal_conditioned_decision_data.scripts import archive_cycle
            requested=_control(args.receipt,'manifests/remote_custody/')
            index=archive.load_index(ROOT)
            pending=sorted(path for path in index['files'] if archive._path(ROOT,path).exists())
            if not pending or index['files'][pending[0]]['receipt_id']!=requested.get('receipt_id'):
                raise archive.ArchiveError('requested receipt is not the next bounded recovery cohort')
            # Use the full custody-push, pause/review guard and Git-cleanup workflow.
            result=archive_cycle.run(ROOT,automatic=False)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, 'archive ' + args.command + ' refused: ' + str(error) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
