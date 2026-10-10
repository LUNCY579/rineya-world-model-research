#!/usr/bin/env python3
"""Own public-only preparation failures; no future candidate/private source access."""
import hashlib,json,pathlib,runpy
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXEC='/root/adversarial_auditor'
BINDINGS=[
 {'source':'audits/rgm_planning_gap_v2_adversarial_preparation.py','source_sha256':'9b3951ccd4ae76caec4e60e64d77f3a8588909d08cb9f179d002027235787dce','preexec':'audits/execution/planning-gap-v2-adversarial-preparation.preexec.json','preexec_sha256':'01e79c772af0a8841f2ad813ec206e3d5918ce01453b35fca54254686273d2e5','failure':'audits/execution/planning-gap-v2-adversarial-preparation.failure-9b3951ccd4ae76ca.json','failure_sha256':'2a56baac0f3f1850e1829e46a1f8b527f6ea2e4c1b15f5662bd3e28931764c93','case':'four_move_base'},
 {'source':'audits/rgm_planning_gap_v2_adversarial_preparation_r2.py','source_sha256':'24f57fdc25029aea066db6f4c7ba3c340e73249461dc62898a8c5fe5be1cfe21','preexec':'audits/execution/planning-gap-v2-adversarial-preparation-r2.preexec.json','preexec_sha256':'837da6705522da166bd69f1394237d5e126669cb22b2e528e3d04c5024709351','failure':'audits/execution/planning-gap-v2-adversarial-preparation.failure-24f57fdc25029aea.json','failure_sha256':'69c02da26cccd784f75929413ae104d6d0182c645c2fd1e6069619db0cfed6dd','case':'both_initial_clearance_false_source_priority'},
 {'source':'audits/rgm_planning_gap_v2_adversarial_preparation_r3.py','source_sha256':'7d4a400965e1e3e95a0555433c2776cbebea5112f6160f7323be488f15ce4a6f','preexec':'audits/execution/planning-gap-v2-adversarial-preparation-r3.preexec.json','preexec_sha256':'241180f740dfc21d26293334dc6d754012457b2fa81cbee4de1031820f383a43','failure':'audits/execution/planning-gap-v2-adversarial-preparation.failure-7d4a400965e1e3e9.json','failure_sha256':'85cac7206b2cac169cf2d0a879df48b3cf7b21ef125e1a827a758fd65df2db93','case':'TABLE_cannot_supply_destination_obstruction'}]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def freeze(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 b=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
 if path.exists():assert path.read_bytes()==b
 else:path.write_bytes(b)
 return sha(path)
def main():
 for row in BINDINGS:
  for kind in ('source','preexec','failure'):assert sha(ROOT/row[kind])==row[kind+'_sha256']
 source=pathlib.Path(__file__);s=sha(source)
 snap=ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_preparation_failure_custody-{s}.py'
 if snap.exists():assert snap.read_bytes()==source.read_bytes()
 else:snap.write_bytes(source.read_bytes())
 pre=ROOT/'audits/execution/planning-gap-v2-adversarial-preparation-failure-custody.preexec.json'
 ph=freeze(pre,{'actual_execution_id':EXEC,'source_sha256':s,'bindings':BINDINGS,'stage':'OWN_RETAINED_FAILURE_REPLAY_SOURCE_FROZEN_BEFORE_EXECUTION','scope':'Own previous expectations/sources/failures only; no future V2 dataset/private source or other reviewer access.'})
 evidence=[]
 for i,b in enumerate(BINDINGS):
  ns=runpy.run_path(str(ROOT/b['source']),run_name='own_failure_replay')
  frozen=json.loads((ROOT/b['preexec']).read_text())
  if i<2:
   case=next(x for x in frozen['manual_exact_targets'] if x['name']==b['case'])
   actual,info=ns['solve'](case['task']);expected=case['hand_derived_exact_target']
   alternate=ns['alternate_blocks_minimum'](case['task'])
   assert canon(actual)!=canon(expected) and canon(actual['shortest_plan'])==canon(alternate)
   assert len(actual['shortest_plan'])==len(expected['shortest_plan'])
   assert tuple((m['block'],m['to']) for m in actual['shortest_plan'])<tuple((m['block'],m['to']) for m in expected['shortest_plan'])
   evidence.append({'original_binding':b,'original_hand_case':case,'actual_exact_replay':actual,'alternate_relation_minimum':alternate,'failure_reproduced':True,'meaning':'Own expected lexical minimum missed an equally short lexically earlier parking sequence; no future candidate issue.'})
  else:
   case=next(x for x in frozen['clearance_expectation_cases'] if x['name']==b['case'])
   valid=ns['validate_clearance_v2'](case['task'],case['evidence']);action=case['task']['plan'][case['evidence']['action_index']]
   assert valid is True and case['expected_valid'] is False and action['to']!='TABLE'
   evidence.append({'original_binding':b,'original_hand_case':case,'actual_valid':valid,'selected_actual_action':action,'failure_reproduced':True,'meaning':'Own intended TABLE negative accidentally selected a true non-TABLE destination-clearance witness; corrected toy changes selected action, not validator or public meaning.'})
 for row in BINDINGS:
  for kind in ('source','preexec','failure'):assert sha(ROOT/row[kind])==row[kind+'_sha256']
 path=ROOT/'audits/execution/planning-gap-v2-adversarial-preparation-failure-custody.json'
 h=freeze(path,{'actual_execution_id':EXEC,'completed':True,'source_sha256':s,'preexec_sha256':ph,'actual_reproduced_failure_count':len(evidence),'evidence':evidence,'prior_artifacts_unchanged':True,'dataset_approval':False,'review_mode':'Own fixture-error replay; common-model/public-guided preparation history retained.'})
 print(canon({'source_sha256':s,'preexec_sha256':ph,'execution_sha256':h,'reproduced_failures':len(evidence)}))
if __name__=='__main__':main()
