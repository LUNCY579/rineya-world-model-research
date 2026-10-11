#!/usr/bin/env python3
"""Stream accepted training views; no training job is launched."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.decision_labels.views import training_views
p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','dev','ood'],required=True);p.add_argument('--view',choices=['pointwise','pairwise','listwise'],required=True);args=p.parse_args()
for path in sorted((ROOT/'data/accepted').glob(args.split+'-*.jsonl')):
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['quality_gate']['status']!='accepted':raise RuntimeError('training export encountered quarantined data')
        for item in training_views(row)[args.view]:print(json.dumps(item,sort_keys=True,separators=(',',':')))
