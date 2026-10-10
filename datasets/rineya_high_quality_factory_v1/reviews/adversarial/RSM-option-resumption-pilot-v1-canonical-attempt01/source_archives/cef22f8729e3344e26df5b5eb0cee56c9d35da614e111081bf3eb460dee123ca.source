"""Owned target-free packet layout inspection; no solver/private imports."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3]
P=R/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl'
raw=P.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='d99074c6cd18d1bd9709a09f833bfe9fea50c0c3981ece532c3951aac220f6b1'
rows=[json.loads(x) for x in raw.splitlines()]
print(json.dumps({'rows':len(rows),'fields':list(rows[0]),'first':rows[0],
                 'join_examples':[{k:r[k] for k in r if k not in {'semantic_task','rendered_input'}} for r in rows[:20]],
                 'split_counts':{s:sum(r['split']==s for r in rows) for s in {r['split'] for r in rows}},
                 'task_modes':{s:sum(r['semantic_task']['mode']==s for r in rows) for s in {r['semantic_task']['mode'] for r in rows}}},indent=2))
