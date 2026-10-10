#!/opt/anaconda3/bin/python3
"""Own alternate heap/edge-relation minimum on every distinct block world/goal.
Uses only already frozen own prompt-derived predictions and supplied input view.
No canonical targets or producer/checker source used by this supplement.
"""
import datetime,hashlib,json,pathlib,runpy,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert'
sha=lambda b:hashlib.sha256(b).hexdigest()
prep=OUT/'prepare_rgm_planning_gap_v1.py';assert sha(prep.read_bytes())=='82ae6d000d5289bf439599579eb6437de5a09468388ead4fb1cef95f9dd3a987'
own=pathlib.Path(__file__).read_bytes();archive=OUT/'planning-gap-v1.expert_block_alternatives.preexecution_source.py';assert archive.read_bytes()==own
P=runpy.run_path(str(prep));canon,eq=P['canon'],P['eq']
view=ROOT/'reviews/input_only/planning-gap-v1.jsonl';vb=view.read_bytes();assert sha(vb)=='2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654'
pf=OUT/'planning-gap-v1.expert_input_predictions.jsonl';pb=pf.read_bytes();assert sha(pb)=='5fe3101f6ca6b6dcda5518b92fe659ffe254bf7862992609bccf9e7fc018f5b3'
pred={p['sample_id']:p for p in map(json.loads,pb.splitlines())};seen=set();records=[];start=time.perf_counter()
for r in map(json.loads,vb.splitlines()):
 t=r['semantic_task']
 if t['kind']!='blocks':continue
 key=canon([t['on'],t['goal_on']])
 if key in seen:continue
 seen.add(key);alt=P['block_minimum_alternative'](t);expected=pred[r['sample_id']]['expected_target']['shortest_plan']
 records.append({'sample_id':r['sample_id'],'record_hash':r['original_record_hash'],'world_goal_sha256':sha(key.encode()),'full_sequence_lex_minimum_match':eq(alt,expected),'alternate_minimum':alt,'frozen_minimum_sha256':sha(canon(expected).encode())})
assert all(r['full_sequence_lex_minimum_match'] for r in records)
report={'actual_author':'/root/expert_reviewer','stage':'OWN_ADDITIONAL_BLOCK_HEAP_RELATION_MINIMUM_VERIFICATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'elapsed_seconds':time.perf_counter()-start,'source_sha256':sha(own),'preexecution_archive_sha256':sha(archive.read_bytes()),'frozen_preparation_source_sha256':sha(prep.read_bytes()),'view_sha256':sha(vb),'frozen_prediction_sha256':sha(pb),'distinct_block_world_goals':len(records),'all_full_sequence_minima_match':True,'records':records,'scope':'Independent heap ranked by full action sequence with separately authored relation-set transitions, versus own frozen lex-BFS predictions for every distinct current block world/goal. Targets already frozen before this actual check. Does not measure human/model difficulty; no canonical gold/private source read in this supplementary execution.'}
op=OUT/'planning-gap-v1.expert_block_alternatives.execution.json';assert not op.exists();op.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(canon({k:v for k,v in report.items() if k!='records'}))
