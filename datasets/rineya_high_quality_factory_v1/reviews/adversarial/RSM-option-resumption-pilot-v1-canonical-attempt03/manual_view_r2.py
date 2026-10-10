"""Present owned frozen factual evidence for actual manual assessment.
No semantic helpers, target recomputation, scores or quality decisions.
"""
import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'RSM-option-resumption-pilot-v1-input-first-attempt01'
mode=sys.argv[1]
if mode=='keys':
    for name in ['FACTUAL_EVIDENCE.json','COMPARISON_EXECUTION.json','OPERATIONAL_ARGUMENT_INSPECTION.json']:
        value=json.loads((OUT/name).read_text())
        print(name, list(value) if isinstance(value,dict) else 'list '+str(len(value)))
    for name in ['INPUT_DIVERSITY_SHORTCUT_AUDIT.json','NUMERIC_SHORTCUT_EXECUTION.json']:
        print(name,list(json.loads((OLD/name).read_text())))
elif mode=='explanations':
    rows=json.loads((OUT/'MANUAL_ROW_INSPECTION.json').read_text())
    seen=set()
    for row in rows:
        key=(row['context'],row['difficulty'],row['explanation'],tuple(row['tags']))
        if key not in seen:
            seen.add(key);print(json.dumps({'context':row['context'],'difficulty':row['difficulty'],'explanation':row['explanation'],'tags':row['tags'],'example':row['sample_id']}))
elif mode.startswith('fact:'):
    key=mode[5:];print(json.dumps(json.loads((OUT/'FACTUAL_EVIDENCE.json').read_text())[key],indent=2))
elif mode.startswith('audit:'):
    key=mode[6:];print(json.dumps(json.loads((OLD/'INPUT_DIVERSITY_SHORTCUT_AUDIT.json').read_text())[key],indent=2))
elif mode.startswith('numeric:'):
    key=mode[8:];print(json.dumps(json.loads((OLD/'NUMERIC_SHORTCUT_EXECUTION.json').read_text())[key],indent=2))
else:raise ValueError('presentation mode')
