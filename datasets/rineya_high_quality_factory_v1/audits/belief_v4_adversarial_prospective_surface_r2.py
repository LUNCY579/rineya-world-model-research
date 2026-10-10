#!/usr/bin/env python3
"""Owned supplementary raw typed input/aggregate-shape attacks after r1 fixture freeze.
R1 was deliberately a focused semantic-record resolver, not a complete prompt validator.
No production candidate/code or other verdict is inspected. Only handmade fixture convention.
"""
import collections,copy,hashlib,importlib.util,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
BINDINGS={'audits/belief_v4_adversarial_prospective_r1.py':'8b350f6ca9831d507e373e60884d02cbd21d498a993a5aa660bd4c2ca9f3cc6e','reviews/adversarial/belief-v4-prospective-fixtures-r1.json':'27b5296bbe48be4f94a962a2ec5918811d25e285988e2672a8c3cd76bc96bcda','reviews/adversarial/belief-v4-prospective-preparation-r1.json':'4ec8d89b685decd9c4cb7202853089128fe13bd7a02bd1234b6d0a7d7cc3cc3e','audits/execution/belief-v4-adversarial-prospective-r1.json':'c190f44a3d9e0837042161e761063c07885ca4a82dac6d27128103ae1f147317'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def freeze(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
def main():
 for rel,h in BINDINGS.items():assert sha(ROOT/rel)==h
 source=pathlib.Path(__file__);sh=sha(source);freeze(ROOT/f'audits/history/belief_v4_adversarial_prospective-surface-{sh}.py',source.read_bytes())
 pre=ROOT/'audits/execution/belief-v4-adversarial-prospective-surface-r2.preexec.json';freeze(pre,jb({'actual_execution_id':EXEC,'source_sha256':sh,'bindings':BINDINGS,'stage':'SOURCE_FROZEN_BEFORE_NEW_RAW_INPUT_AND_AGGREGATE_TESTS','scope':'Only new own handmade fixture program/recipe/receipt, no actual candidate/private implementation/other verdict.','disclosure':'Prior belief canonical exposure/common model/public guidance/own source reuse; no data approval.'}))
 print(json.dumps({'stage':'surface_source_frozen','source_sha256':sh,'preexec_sha256':sha(pre)},sort_keys=True),flush=True)
 spec=importlib.util.spec_from_file_location('own_prospective_r1',ROOT/'audits/belief_v4_adversarial_prospective_r1.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);start=time.perf_counter();recipe=json.loads((ROOT/'reviews/adversarial/belief-v4-prospective-fixtures-r1.json').read_text());rows,coords,manifest,claims,candhash=A.make_fixture(recipe)
 def raw_decode(r):
  def pairs(ps):
   d={}
   for k,v in ps:A.require(k not in d,'duplicate raw JSON key '+k);d[k]=v
   return d
  payload=json.loads(r['rendered_input'],object_pairs_hook=pairs);A.require(A.canon(payload)==A.canon(r['semantic_task']),'raw typed prompt payload differs from semantic record')
  # This convention is from our own frozen hand renderer, not a new production contract.
  order=['kind','worlds','observations','queries'];actual='forward' if list(payload)==order else 'reverse' if list(payload)==list(reversed(order)) else None
  A.require(actual==r['renderer'],'hand-fixture actual field style disagrees with annotation')
  return A.decode(r)
 positive=[]
 for r in rows.values():raw_decode(r);positive.append(r['sample_id'])
 tests=[];r1_scope_counterexamples=[]
 original=rows[coords[('A','identity',0,'forward')]]
 # Actual independent payload attacks; the r1 focused resolver is not claimed to cover them.
 for name,mutate in [('raw_observation_null',lambda p:p['observations'][0].update(equals=None)),('raw_world_boolean_to_integer',lambda p:p['worlds'][0]['facts'].update({original['fact_role_assignments']['variation_query']:1})),('raw_query_fact_changed',lambda p:p['queries'][0].update(fact='u'))]:
  r=copy.deepcopy(original);payload=json.loads(r['rendered_input']);mutate(payload);r['rendered_input']=json.dumps(payload,separators=(',',':'));r['record_hash']=A.record_hash(r)
  A.decode(r);r1_scope_counterexamples.append({'name':name,'r1_focused_decode_accepts':True,'raw_prompt':r['rendered_input'],'unchanged_semantic_task':r['semantic_task'],'scope':'Expected outside r1 focused resolver; cannot certify prompt binding without this independent surface check.'})
  try:raw_decode(r)
  except (ValueError,TypeError) as e:tests.append({'name':name,'rejected':True,'error':str(e)})
  else:raise ValueError('Raw input mutation accepted '+name)
 r=copy.deepcopy(original);r['renderer']='reverse';r['record_hash']=A.record_hash(r)
 try:raw_decode(r)
 except ValueError as e:tests.append({'name':'coordinated_metadata_only_style_change_requires_actual_style_check','rejected':True,'error':str(e)})
 else:raise ValueError('False actual renderer accepted')
 for name,raw in [('duplicate_outer_key',original['rendered_input'][:-1]+',"kind":"belief"}'),('duplicate_nested_fact',original['rendered_input'].replace('"u":0','"u":0,"u":0',1))]:
  r=copy.deepcopy(original);r['rendered_input']=raw;r['record_hash']=A.record_hash(r)
  try:raw_decode(r)
  except ValueError as e:tests.append({'name':name,'rejected':True,'error':str(e)})
  else:raise ValueError('Duplicate raw key accepted '+name)
 # Published aggregate identifiers can omit inline pair IDs. No unknown manifest keys are required.
 aggregate_claims=[]
 for name in sorted(A.AGGREGATE):
  c={'name':name,'scope':'corpus_variant','source_family_key':original['source_family_key'],'evidence_manifest':A.MANIFEST_PATH,'control_id':'handmade-aggregate-group'}
  A.require(set(c)<=A.CLAIM_FIELDS and len(set(c)&{'name','property'})==1 and 'pair_sample_ids' not in c,'valid original claim shape without inlinepair for aggregate');aggregate_claims.append(c)
 targets={};details={}
 for sid,r in rows.items():targets[sid],details[sid]=A.solve(raw_decode(r))
 cross,evidence=A.crossing_checks(rows,coords,targets,details)
 # Group truth, not a manifest annotation, supplies three aggregate relationships in this fixture.
 def reconstruction(rs):
  for r in rs.values():raw_decode(r)
  ts={};ds={}
  for sid,r in rs.items():ts[sid],ds[sid]=A.solve(r['semantic_task'])
  return A.crossing_checks(rs,coords,ts,ds)
 missing=dict(rows);missing.pop(coords[('G','identity',14,'forward')])
 try:reconstruction(missing)
 except (ValueError,KeyError) as e:tests.append({'name':'aggregate_pointer_cannot_credit_missing_schedule_member','rejected':True,'error':str(e)})
 else:raise ValueError('Missing aggregate member accepted')
 scalar=dict(rows);sid=coords[('A','transposed',0,'forward')];r=copy.deepcopy(scalar[sid]);r['semantic_task']['worlds'][0]['facts']['y']='linen';r['rendered_input']=json.dumps(r['semantic_task'],separators=(',',':'));r['record_hash']=A.record_hash(r);scalar[sid]=r
 try:reconstruction(scalar)
 except ValueError as e:tests.append({'name':'aggregate_scalar_annotation_cannot_substitute_for_actual_involution','rejected':True,'error':str(e)})
 else:raise ValueError('False involution accepted')
 coupled={sid:r for sid,r in rows.items() if r['fact_name_shift']==(r['id_shift']+r['position_shift'])%3}
 try:reconstruction(coupled)
 except (ValueError,KeyError) as e:tests.append({'name':'legacy9_plane_cannot_credit_full27_conditional_crossing','rejected':True,'error':str(e)})
 else:raise ValueError('Incomplete 9 schedule plane accepted')
 # A probe scope never earns required corpus crossing or inherited supplied pair coverage.
 c=copy.deepcopy(aggregate_claims[0]);c['scope']='audit_probe';c['probe_id']='unexecuted';A.require(c['scope']!='corpus_variant','audit probe cannot count as corpus');tests.append({'name':'unexecuted_probe_no_corpus_aggregate_credit','rejected_for_corpus_credit':True,'probe_execution_not_fabricated':True})
 c=copy.deepcopy(claims[original['sample_id']][0]);c.pop('pair_sample_ids')
 try:A.resolve(original,c,rows,manifest,candhash)
 except ValueError as e:tests.append({'name':'inherited_cannot_use_new_aggregate_inline_omission_permission','rejected':True,'error':str(e)})
 else:raise ValueError('Inherited aggregate-only claim accepted')
 # Strong owner schedule attack uses another legitimate decoded owner rather than an invalid role map.
 owner=rows[coords[('A','identity',1,'forward')]];c=copy.deepcopy(claims[original['sample_id']][0])
 try:A.resolve(owner,c,rows,manifest,candhash)
 except ValueError as e:tests.append({'name':'valid_decoded_owner_with_foreign_schedule_pair','rejected':True,'error':str(e)})
 else:raise ValueError('Wrong real owner schedule accepted')
 for rel,h in BINDINGS.items():assert sha(ROOT/rel)==h
 result={'actual_execution_id':EXEC,'completed':True,'stage':'PROSPECTIVE_PUBLIC_HANDMADE_SURFACE_AND_AGGREGATE_SUPPLEMENT','source_sha256':sh,'preexec_sha256':sha(pre),'bindings_before_after':BINDINGS,'actual_seconds':time.perf_counter()-start,'actual864_raw_typed_prompt_and_fixture_style_checks':len(positive),'actual_negative_tests':tests,'r1_scope_counterexamples':r1_scope_counterexamples,'aggregate_claim_shapes_without_inline_pairs':aggregate_claims,'actual_complete_group_crossings':cross,'public_scope_decisions':['Mandatory owner coordinates and actual observation length are accepted prospective requirements under the explicit new addendum.','ID+name+orderedpair exactlyone resolver; sameIDacrossnames allowed; no globaluniqueID or pair_record_hashes rule.','ThreeV3 aggregate identifier claim shapes may omit inline pairs; actual full group reconstruction is still required.','Both inherited identifiers must have inlinepairs for supplied corpus_variant claims; a new-control aggregate permission or executed probe cannot replace them.','The hand fixture raw style convention is an authored test convention, not a newly imposed exact production renderer layout.','This supplement adds independent raw input consistency checks outside the original focused r1 record-metadata resolver; it does not claim a production defect.'],'remaining_future_attacks':['Once authorized real inputs arrive, typed raw payload/semantic_task equality and actual presentation styles must be checked independently for allrows before prediction reuse.','Later current candidate comparison must test full canonical required fields/hash, exact source/split lineage, proper V3/V4 rowwise semantic preservation, every tag/locator and all actual claims.','Do not certify a diagnostic annotation or arbitrary fitted manifest representation as evidence; derive supports/prefixes/query partitions yourself.','Actual table/character/promptlength/observation-role/scalar patterns remain unmeasured for V4; future fullworld/cross-release and conditional nuisance analysis is required.'],'data_approval':False,'no_actual_V3_V4_candidate_or_private_source_or_other_verdict_read':True,'disclosure':A.DISCLOSURE}
 out=ROOT/'reviews/adversarial/belief-v4-prospective-surface-r2.json';freeze(out,jb(result));ex=ROOT/'audits/execution/belief-v4-adversarial-prospective-surface-r2.json';freeze(ex,jb({'actual_execution_id':EXEC,'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'report_sha256':sha(out),'bindings_before_after':BINDINGS,'actual_raw_checks':864,'actual_new_negative_tests':len(tests),'r1_focused_scope_counterexamples':len(r1_scope_counterexamples),'actual_aggregate_reconstruction':cross,'data_approval':False}))
 print(json.dumps({'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'report_sha256':sha(out),'execution_sha256':sha(ex),'raw_checks':864,'negative_checks':len(tests),'r1_scope_counterexamples':len(r1_scope_counterexamples)},sort_keys=True),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  sh=sha(pathlib.Path(__file__));p=ROOT/f'audits/execution/belief-v4-adversarial-prospective-surface-r2.failure-{sh[:16]}.json';freeze(p,jb({'actual_execution_id':EXEC,'completed':False,'source_sha256':sh,'error_type':type(e).__name__,'error':str(e)}));print(json.dumps({'failure_path':str(p.relative_to(ROOT)),'failure_sha256':sha(p),'error':str(e)},sort_keys=True),flush=True);raise
