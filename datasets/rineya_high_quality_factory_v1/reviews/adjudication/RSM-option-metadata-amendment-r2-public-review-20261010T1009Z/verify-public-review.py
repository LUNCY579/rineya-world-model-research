"""Narrow public custody/count verification only; no primitive or cohort checker."""
from pathlib import Path
import datetime,hashlib,json
D=Path(__file__).resolve().parent;B=D.parents[2]
def bind(p):
 b=(B/p).read_bytes();return {'path':p,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def main():
 pre=json.loads((D/'preexecution.json').read_text())
 for b in pre['bindings']:assert bind(b['path'])==b
 rf=json.loads((B/'research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R2_ROOT_FREEZE.json').read_text())
 archives=[]
 for b in rf['bindings']:
  current=bind(b['path']);archive=bind(b['archive']);assert current['sha256']==archive['sha256']==b['sha256'] and current['bytes']==archive['bytes']==b['bytes'];archives.append({'public_path':b['path'],'archive':archive})
 cfg=json.loads((B/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json').read_text());indices=[i for v in cfg['source_family_indices_by_stratum'].values() for i in v]
 assert len(indices)==len(set(indices))==12 and len(cfg['required_contexts'])==9 and len(cfg['renderers'])==2
 assert sum(i%10<8 for i in indices)==8 and sum(i%10>=8 for i in indices)==4
 result={'status':'COMPLETED_NARROW_PUBLIC_R2_DESIGN_REVIEW','actual_author_execution_id':'/root/research_architect','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conclusion':'PUBLIC_INTERFACE_CONSISTENT_NO_NEW_BLOCKING_DESIGN_ISSUE','circular_canonical_member_hash_issue_resolved':True,'operative_public_source_count':6,'root_live_bindings_and_archives_verified':archives,'prospective_counts':{'families':12,'typed_control_tasks':108,'views':216,'named_pairs':300,'owner_claims':1944,'nonmember_claims_first_eight':1344,'train_families':8,'exposed_dev_families':4},'remaining_gates':['independent checker/interpreter/primitive tests','actual generic-validator vocabulary integration','actual12-program three-view novelty/control audit','fresh input-only expert/red predictions and later independent adjudication','immutable release and remote byte verification'],'actual_production_counts_checked':False,'private_code_or_candidate_gold_read':False,'primitive_or_corpus_checker_executed':False,'data_or_code_approval':False,'bindings_before':pre['bindings'],'bindings_after':[bind(b['path']) for b in pre['bindings']],'preexecution':bind(str((D/'preexecution.json').relative_to(B))),'disclosures':['original public design author and possible later adjudicator overlap','root-guided prospective repair review','same-model/common-guidance correlation','finite-program/exposed-dev coverage without efficacy']}
 assert result['bindings_before']==result['bindings_after']
 with (D/'execution.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'execution':bind(str((D/'execution.json').relative_to(B))),'root_bindings_verified':len(archives),'new_preexecution_bindings':len(pre['bindings']),'data_approval':False,'conclusion':result['conclusion']}))
if __name__=='__main__':main()
