#!/usr/bin/env python3
"""Bounded rendered-data publication only; no model training is launched."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from goal_conditioned_decision_data.decision_labels.rendering import VIEWS, export_rendered_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'dev', 'ood'), required=True)
    parser.add_argument('--view', choices=VIEWS, default='listwise')
    parser.add_argument('--max-records', type=int, required=True, help='accepted source records examined, 1..12000')
    parser.add_argument('--variant-budget', type=int, required=True, help='surface variants per view, 1..3')
    parser.add_argument('--output-dir', type=Path, help='destination base beneath this programme data/rendered')
    parser.add_argument('--source-shard', type=Path, action='append', help='optional selected accepted shard; repeatable')
    args = parser.parse_args(argv)
    try:
        manifest = export_rendered_dataset(ROOT, args.split, args.view, args.max_records,
                                           args.variant_budget, args.output_dir, args.source_shard)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, 'rendered export failed: ' + str(error) + '\n')
    print(json.dumps({'dataset_id': manifest['dataset_id'], 'manifest_path': str(ROOT / manifest['manifest_path']),
                      'source_records_examined': manifest['source_records_examined'],
                      'eligible_source_records': manifest['eligible_source_records'],
                      'rendered_rows': manifest['rendered_rows'], 'training_launched': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
