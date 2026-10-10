"""Isolated actual public pair-ID correction regressions; freeze before target imports."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=ROOT/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'
PREEXEC=HERE/'PREEXEC.json'
RESULT=HERE/'EXECUTION.json'


def binding(path):
    raw=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    frozen=json.loads(PREEXEC.read_bytes())
    for row in frozen['bindings']+frozen['immutable_original_bindings']:
        if binding(ROOT/row['path'])!=row:raise ValueError('Byte binding differs before correction import')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    sys.path[:0]=[str(HERE),str(OLD)]
    import control_probes_r1 as original
    import control_probes_r2 as corrected
    fixtures=json.loads((HERE/'hand_fixtures.json').read_bytes());checks=[]
    def check(name,fn,expect_rejection=False):
        try:
            result=fn();passed=not expect_rejection;error=None
            digest=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        except Exception as exception:
            passed=expect_rejection;digest=None;error={'type':type(exception).__name__,'message':str(exception)}
        checks.append({'name':name,'passed':passed,'expected_rejection':expect_rejection,'error':error,'returned_value_sha256':digest})
    def equal(a,b):
        if type(a) is not type(b) or a!=b:raise AssertionError('Expected exact typed identity')
        return a
    base=fixtures['claim']
    for index,pair in enumerate(fixtures['positive_pairs']):
        claim=copy.deepcopy(base);claim['pair_sample_ids']=pair
        check('corrected_pair_acceptance_'+str(index),lambda claim=claim:equal(corrected.claim_key(claim),(claim['control_id'],claim['property'],tuple(claim['pair_sample_ids']))))
        check('original_behavior_'+str(index),lambda claim=claim:original.claim_key(claim),expect_rejection=any(value=='' for value in pair))
    for index,pair in enumerate(fixtures['negative_pairs']):
        claim=copy.deepcopy(base);claim['pair_sample_ids']=pair
        check('pair_type_or_distinctness_reject_'+str(index),lambda claim=claim:corrected.claim_key(claim),True)
    for name,pair in [('tuple',('','opaque')),('scalar','opaque'),('null',None),('object',{})]:
        claim=copy.deepcopy(base);claim['pair_sample_ids']=pair
        check('nonarray_pair_reject_'+name,lambda claim=claim:corrected.claim_key(claim),True)
    for index,control_id in enumerate(fixtures['positive_control_ids']):
        claim=copy.deepcopy(base);claim['control_id']=control_id
        check('literal_control_id_acceptance_'+str(index),lambda claim=claim:equal(corrected.claim_key(claim)[0],claim['control_id']))
    for index,control_id in enumerate(fixtures['negative_control_ids']):
        claim=copy.deepcopy(base);claim['control_id']=control_id
        check('control_id_nonempty_string_reject_'+str(index),lambda claim=claim:corrected.claim_key(claim),True)
    for index,value in enumerate((None,[],['0'*64,'0'*64])):
        claim=copy.deepcopy(base);claim['pair_record_hashes']=value
        check('forbidden_canonical_hash_field_'+str(index),lambda claim=claim:corrected.claim_key(claim),True)
    members=fixtures['members'];owner=fixtures['owner'];entries=fixtures['entries'];candidate=fixtures['candidate_sha256']
    check('actual_join_empty_first_ID_nonmember_owner',lambda:corrected.resolve_claim(owner,base,entries,members,candidate))
    reverse_claim=copy.deepcopy(base);reverse_claim['pair_sample_ids'].reverse()
    check('reversed_claim_order_reject',lambda:corrected.resolve_claim(owner,reverse_claim,entries,members,candidate),True)
    reversed_entries=copy.deepcopy(entries);reversed_entries[0]['pair_sample_ids'].reverse();reversed_entries[0]['controlled_contexts'].reverse();reversed_entries[0]['pair_record_hashes'].reverse()
    check('matching_reversed_lookup_still_wrong_semantic_order',lambda:corrected.resolve_claim(owner,reverse_claim,reversed_entries,members,candidate),True)
    check('duplicate_lookup_empty_ID_reject',lambda:corrected.resolve_claim(owner,base,entries+[copy.deepcopy(entries[0])],members,candidate),True)
    whitespace_claim=copy.deepcopy(base);whitespace_claim['control_id']=' \t'
    whitespace_entries=copy.deepcopy(entries);whitespace_entries[0]['control_id']=' \t'
    check('whitespace_control_id_exact_join_acceptance',lambda:corrected.resolve_claim(owner,whitespace_claim,whitespace_entries,members,candidate))
    trimmed=copy.deepcopy(whitespace_claim);trimmed['control_id']=' '
    check('control_id_not_trimmed_lookup_reject',lambda:corrected.resolve_claim(owner,trimmed,whitespace_entries,members,candidate),True)
    nohash=copy.deepcopy(entries);nohash[0].pop('pair_record_hashes')
    check('optional_external_hashes_absent_valid',lambda:corrected.resolve_claim(owner,base,nohash,members,candidate))
    badhash=copy.deepcopy(entries);badhash[0]['pair_record_hashes'].reverse()
    check('reversed_external_hashes_reject',lambda:corrected.resolve_claim(owner,base,badhash,members,candidate),True)
    field_owner=fixtures['field_members'][''];field_claim=fixtures['field_claim'];field_entries=fixtures['field_entries']
    check('empty_ID_field_pair_forward_owner_acceptance',lambda:corrected.resolve_claim(field_owner,field_claim,field_entries,fixtures['field_members'],candidate))
    check('empty_ID_field_pair_reverse_owner_acceptance',lambda:corrected.resolve_claim(fixtures['field_members']['reverse'],field_claim,field_entries,fixtures['field_members'],candidate))
    for name in ('name','both'):
        claim=copy.deepcopy(base)
        if name=='name':claim['name']=claim.pop('property')
        else:claim['name']=claim['property']
        check('identifier_form_'+name,lambda claim=claim:corrected.resolve_claim(owner,claim,entries,members,candidate))
    shared=copy.deepcopy(entries);second=copy.deepcopy(entries[0]);second['name']='resume_restart_replay';second['pair_sample_ids']=['opaque-R','opaque-T'];second['controlled_contexts']=['R','T'];second['pair_record_hashes']=[members[x]['record_hash'] for x in second['pair_sample_ids']];shared.append(second)
    second_claim=copy.deepcopy(base);second_claim['property']='resume_restart_replay';second_claim['pair_sample_ids']=['opaque-R','opaque-T']
    check('shared_control_ID_across_distinct_names_acceptance',lambda:corrected.resolve_claim(owner,second_claim,shared,members,candidate))
    for row in frozen['bindings']+frozen['immutable_original_bindings']:
        if binding(ROOT/row['path'])!=row:raise ValueError('Original or new frozen bytes changed during regression')
    report={'schema':'rsm-public-interface-scoped-correction-execution-v1','actor':'/root/adversarial_auditor','started_at_utc':started,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preexecution':binding(PREEXEC),'checks':checks,'check_count':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'source_and_public_bytes_before_after_identical':True,'all_original_preparation_artifacts_unchanged':True,'scope':'Literal claim parser and supplied toy coordinate joins only. No production data, target solver rerun, full canonical schema or whole-corpus compliance claim.','source_change':'Remove unpublished nonempty gate on the two distinct pair strings; control_id remains nonempty and literal whitespace is not stripped.','model_disclosure':'Same model, root-guided public-interface defect; reuses own frozen original code/handmade metadata. No other reviewer/private checker/results read.','data_approval':False}
    with RESULT.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({'check_count':report['check_count'],'passed':report['passed'],'failed':report['failed'],'execution':binding(RESULT)}))


if __name__=='__main__':main()
