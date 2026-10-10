"""Read already frozen own inspection material as compact manual review text.
No semantic helper, target computation, scores or machine quality decisions.
"""
import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
mode=sys.argv[1]
if mode=='rows':
    for row in json.loads((OUT/'MANUAL_ROW_INSPECTION.json').read_text()):
        print(row['sample_id'],row['record_hash'][:12],row['split'],'L'+str(row['difficulty']),row['mode'],
            'h='+str(row['history_events']),'b='+str(row['budget']),row['status'],'events='+str(row['new_events']),
            'effects='+str(row['new_effects']),str(row['final_state']),'bad='+str(len(row['failed_factual_checks'])))
else:
    groups=json.loads((OUT/'MANUAL_FAMILY_PROGRAMS.json').read_text())
    start,end=map(int,mode.split(':'))
    for group in groups[start:end]:
        print('FAMILY',group['family_index'],group['family'],group['source_stratum'],group['initial_state'],group['entry'])
        for name,option in group['options'].items():print('OPTION',name,json.dumps(option,separators=(',',':')))
        for context in group['context_inputs']:
            cp=context['checkpoint'];frames=cp['frames'] if cp else None
            print('CONTEXT',context['context'],context['entry'],'mode='+context['mode'],'budget='+str(context['budget']),
                'h='+str(context['history_events']),'cpstate='+str(cp['state'] if cp else None),'frames='+str(frames),
                'rootinit='+str(context['root_initiation']),'rootsuccess='+str(context['root_success']),context['status'],context['final_state'])
        print('OPERATIONS',group['operational_facts'])
