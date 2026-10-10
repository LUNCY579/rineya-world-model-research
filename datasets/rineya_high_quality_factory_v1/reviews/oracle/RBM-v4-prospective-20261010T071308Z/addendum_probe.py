"""Additive NEW nonproduction hand probe for newly published owner addendum.
Preserves the earlier minimum-V4 probe, including its duplicate-entry example.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import unittest

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
PUBLIC={
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md':'925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c',
 'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json':'2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
spec=importlib.util.spec_from_file_location('frozen_own_minimum_hand',HERE/'hand_probe.py')
hand=importlib.util.module_from_spec(spec);spec.loader.exec_module(hand)
COUNTS={'positive_inherited_joins':0,'expected_rejections':0}
NEGATIVES=[]

def join_v4(rows,sid,claim,manifest,candidate_sha):
    hand.need(type(manifest) is dict and type(manifest.get('supplied_same_table_v2_pairs')) is list,'published collection required by NEW addendum:9')
    entries=manifest['supplied_same_table_v2_pairs'];name=hand.claim_name(claim)
    matches=[entry for entry in entries if type(entry) is dict and entry.get('control_id')==claim.get('control_id')
        and entry.get('name')==name and hand.equal(entry.get('pair_sample_ids'),claim.get('pair_sample_ids'))]
    hand.need(len(matches)==1,'exactly one ID + name + ordered pair entry required by NEW addendum:9')
    return hand.join(rows,sid,claim,entries,candidate_sha,matched_owner=True)

class AddendumHandTests(unittest.TestCase):
    def reject(self,label,rows,sid,claim,manifest,h):
        try:join_v4(rows,sid,claim,manifest,h)
        except ValueError as exc:NEGATIVES.append({'label':label,'error':str(exc)});COUNTS['expected_rejections']+=1
        else:self.fail('mutation accepted '+label)

    def test_complete_local_names_members_nonmembers_both_lengths_variants_styles(self):
        rows,entries,h=hand.fixture();manifest={'supplied_same_table_v2_pairs':entries,'extra_top_level_diagnostic':'permitted'}
        for sid in rows:
            for name in hand.NAMES:
                result=join_v4(rows,sid,hand.make_claim(rows,sid,name),manifest,h)
                self.assertEqual(result['matching_named_entries'],1);COUNTS['positive_inherited_joins']+=1

    def test_shared_id_two_names_and_property_alias_not_global_unique_rule(self):
        rows,entries,h=hand.fixture();sid=next(iter(rows));manifest={'supplied_same_table_v2_pairs':entries}
        self.assertEqual(len({entry['control_id'] for entry in entries}),1)
        for name in hand.NAMES:
            c=hand.make_claim(rows,sid,name);c['property']=c.pop('name');join_v4(rows,sid,c,manifest,h)
            COUNTS['positive_inherited_joins']+=1

    def test_identical_duplicate_lookup_key_now_explicitly_rejected(self):
        rows,entries,h=hand.fixture();sid=next(iter(rows));c=hand.make_claim(rows,sid)
        manifest={'supplied_same_table_v2_pairs':entries+[deepcopy(entries[0])]}
        self.reject('duplicate exact lookup key',rows,sid,c,manifest,h)

    def test_wrong_nonmember_owner_coords_and_length_explicit_new_rejections(self):
        rows,entries,h=hand.fixture();a=next(iter(rows));c=hand.make_claim(rows,a);manifest={'supplied_same_table_v2_pairs':entries}
        d=next(s for s,r in rows.items() if r['controlled_context']=='D')
        self.reject('wrong nonmember owner length',rows,d,c,manifest,h)
        for key,value in (('auxiliary_scalar_variant','transposed'),('schedule_index',1),('renderer','reverse'),('source_family_key','foreign')):
            bad=deepcopy(rows);bad[a][key]=value;self.reject('owner '+key,bad,a,c,manifest,h)

    def test_missing_collection_name_pair_and_candidate_hash(self):
        rows,entries,h=hand.fixture();sid=next(iter(rows));c=hand.make_claim(rows,sid)
        self.reject('missing required collection',rows,sid,c,{'different_collection':entries},h)
        for mode in ('name','pair','hash','length','scope'):
            bad=deepcopy(entries)
            if mode=='name':bad[0]['name']='wrong'
            elif mode=='pair':bad[0]['pair_sample_ids'].reverse()
            elif mode=='hash':bad[0]['exact_candidate_sha256']='b'*64
            elif mode=='length':bad[0]['observation_length']=True
            else:bad[0]['scope']='audit_probe'
            self.reject('entry '+mode,rows,sid,c,{'supplied_same_table_v2_pairs':bad},h)

def run():
    bindings={}
    for p,h in PUBLIC.items():
        hand.need(sha((BASE/p).read_bytes())==h,'frozen NEW public hash mismatch');bindings[p]=h
    hand_path=HERE/'hand_probe.py';hand.need(sha(hand_path.read_bytes())=='a2a5ebb08478d0275ee6e14f6df2f099a0770f72f7e63e6b99770d34736ff577','original hand source changed')
    bindings[str(hand_path.relative_to(BASE))]=sha(hand_path.read_bytes())
    own=Path(__file__);own_hash=sha(own.read_bytes());hand.need((BASE/'reproducibility/source_blobs'/(own_hash+'.source')).read_bytes()==own.read_bytes(),'archive new source before execution')
    bindings[str(own.relative_to(BASE))]=own_hash
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AddendumHandTests))
    report={'actual_execution_id':'/root/formal_oracle','role':'Public prospective additive hand preparation; no official checker/data approval',
        'at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':bindings,'tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'counts':COUNTS,'expected_rejected_cases':NEGATIVES,
        'output':stream.getvalue(),'scope':'New owner/resolver addendum makes previously recommended owner matching and collection/unique lookup hard requirements. Earlier minimum-V4 proposal/probe/duplicate-positive execution are preserved historical interpretations, not asserted compliant with this NEW addendum. No producer/candidate/gold/factory/reviewer implementation read. No full3456 closure or canonical record-hash validation; source intentionally not a production checker.'}
    out=HERE/'addendum_execution.json';hand.need(not out.exists(),'new output immutable');out.write_text(json.dumps(report,indent=2)+'\n')
    print(hand.enc({'path':str(out.relative_to(BASE)),'sha256':sha(out.read_bytes()),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'counts':COUNTS}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':run()
