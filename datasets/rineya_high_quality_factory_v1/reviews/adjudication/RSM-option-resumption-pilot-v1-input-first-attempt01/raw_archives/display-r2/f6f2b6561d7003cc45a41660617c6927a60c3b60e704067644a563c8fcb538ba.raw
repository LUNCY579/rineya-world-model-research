"""Display own permitted input/prediction ledgers for actual agent inspection."""
import json,pathlib,hashlib
D=pathlib.Path(__file__).resolve().parent
B=D.parents[2]
f=json.loads((D/'display-preexecution-r2.json').read_bytes())
for v in f['bindings']:
 raw=(B/v['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256'] and raw==(B/v['archive']).read_bytes()
def expr(e):
 if 'const' in e:return str(e['const'])
 if 'reg' in e:return e['reg']
 return '$'+e['param']
def pred(p):return ' & '.join(expr(a['left'])+' '+a['cmp']+' '+expr(a['right']) for a in p['all']) or 'TRUE'
def body(items,indent=' '):
 for i in items:
  op=i['op']
  if op in ('add','scale'):print(indent+op+' '+i['register']+' '+expr(i['value']))
  elif op=='swap':print(indent+'swap')
  elif op=='call':print(indent+'call '+i['option']+'('+','.join(k+'='+expr(v) for k,v in i['args'].items())+')')
  else:
   print(indent+'if '+pred(i['test']));body(i['then'],indent+' T ');body(i['else'],indent+' E ')
for family in json.loads((D/'family-input-ledger-r2.json').read_bytes()):
 f=family['programs']['F'];t=f['task'];o=f['expected_output'];R=family['programs']['R'];T=family['programs']['T'];controls=family['controls']
 print('\nFAMILY',family['semantic_family_id'],family['split'],'initial',t['initial_state'],'entry',t['entry'])
 for opt,v in t['options'].items():
  print('OPTION',opt,v['params'],'INIT',pred(v['initiation']),'SUCCESS',pred(v['success']));body(v['body'])
 print('F expected',o['status'],o['checkpoint']['state'],'events',len(o['new_events']),'effects',o['new_effects'])
 print('R checkpoint',R['task']['checkpoint'])
 print('T expected',T['expected_output']['status'],T['expected_output']['checkpoint']['state'],'events',len(T['expected_output']['new_events']))
 print('CHANGED FIELDS',controls['one_field_diffs'])
 print('CHECKS',controls['checks'])
 print('ACTIVE FRAMES',controls['R_frame_facts']['active_frames'])
 print('REPLAY',controls['replay_witnesses'])
