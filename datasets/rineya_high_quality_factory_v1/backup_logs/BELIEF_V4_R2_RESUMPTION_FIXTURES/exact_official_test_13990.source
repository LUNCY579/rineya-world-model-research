"""Official V4 preproduction handwritten controls, source frozen before execution.

The author's own earlier handwritten V3 source tables are unchanged dependencies.
No canonical production inputs, producer source, stored gold or private verdicts
are read. All input values below are own finite public-contract fixtures.
"""
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

BASE = next(parent for parent in Path(__file__).resolve().parents if (parent/'oracles/belief_controls_v3_independent.py').is_file())
sys.path.insert(0, str(BASE))
SOURCE = Path(__file__).with_name('belief_controls_v4_independent.py')
if not SOURCE.is_file(): SOURCE = BASE/'oracles/belief_controls_v4_independent.py'
spec = importlib.util.spec_from_file_location('oracles.own_official_v4_checker', SOURCE)
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
spec = importlib.util.spec_from_file_location('own_frozen_v3_handfixtures', BASE/'tests/oracle_belief_controls_v3_test.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
v1, v2, v3 = v4.v1, v4.v2, v4.v3
MANIFEST_PATH = 'own-handmade-v4/controls.json'
MEASURED = {'fixture_rows': 0, 'exhaustive_typed_pairs': 0, 'local_rejections': 0, 'full_runs': 0, 'local_resolution_proofs': 0, 'replay_seeds': []}


def digest(value): return hashlib.sha256(v1._json(value).encode('utf8')).hexdigest()


def ids(index, context, variant, shifts, renderer):
    return 'v3fixture-%s-%s-%s-%s%s%s-%s' % (index, context, variant, *shifts, renderer)


def row(index=12, context='A', variant='identity', shifts=(0,0,0), renderer='forward'):
    item = old.fixture(index, context, variant, shifts, renderer)
    item['generator_version'] = 'own-public-v4-handmade'
    item.update(track_id='RBM-001', oracle_version='own-public-v4-handmade', difficulty=3, generation_seed=20261010, source_provenance={'origin':'original_synthetic','sealed_evaluation_used':False}, target_format='structured', review_status='PENDING', review_receipt_ids=[])
    item['verified_target'] = v1.derive(item['semantic_task'])['target']
    claims = [claim for claim in item['family_control_claims'] if claim['name'] not in v4.INHERITED]
    contexts = 'FH' if len(item['semantic_task']['observations']) == 3 else 'EG'
    pair = [ids(index,c,variant,shifts,renderer) for c in contexts]
    for name in sorted(v4.INHERITED):
        claims.append({'name':name,'scope':'corpus_variant','source_family_key':item['source_family_key'],'pair_sample_ids':pair,'control_id':'same physical evidence '+digest(pair),'evidence_manifest':MANIFEST_PATH})
    item['family_control_claims'] = claims
    item['record_hash'] = v4.record_hash(item)
    return item


def candidate(rows): return ''.join(v1._json(item)+'\n' for item in rows).encode('utf8')


def manifest(rows, raw=None):
    raw = candidate(rows) if raw is None else raw
    entries = {}; by = {item['sample_id']:item for item in rows}
    for item in rows:
        for claim in item['family_control_claims']:
            if claim.get('name',claim.get('property')) not in v4.INHERITED or claim['scope'] != 'corpus_variant': continue
            name = claim.get('name',claim.get('property')); pair = claim['pair_sample_ids']
            key = (claim['control_id'],name,tuple(pair)); left = by[pair[0]]
            entries[key] = {'name':name,'scope':'corpus_variant','source_family_key':left['source_family_key'],'auxiliary_scalar_variant':left['auxiliary_scalar_variant'],'observation_length':len(left['semantic_task']['observations']),'control_id':claim['control_id'],'pair_sample_ids':pair,'exact_candidate_sha256':hashlib.sha256(raw).hexdigest(), 'own_optional_diagnostics':{'independent_input_pointer':pair}}
    return {'supplied_same_table_v2_pairs':list(entries.values()), 'extra_top_level_diagnostics':{'public_annotation_shape_is_open':True}}


def bundle(rows):
    raw = candidate(rows); parsed = manifest(rows, raw)
    return raw, {MANIFEST_PATH: v1._json(parsed).encode('utf8')}, parsed


def local(index=12, context='A', variant='identity', shifts=(0,0,0), renderer='forward'):
    group = [row(index,c,variant,shifts,renderer) for c in v2.CONTEXTS]
    owner = next(item for item in group if item['controlled_context'] == context)
    raw, _, parsed = bundle(group)
    claim = next(c for c in owner['family_control_claims'] if c.get('name') in v4.INHERITED)
    return owner, claim, {item['sample_id']:item for item in group}, parsed, hashlib.sha256(raw).hexdigest()


def reject(owner, claim, by, parsed, sha):
    # Rebind handmade record custody, isolating the intended semantic/encoding
    # corruption. Actual stale-hash rejection is separately tested below.
    for item in [owner, *by.values()]: item['record_hash'] = v4.record_hash(item)
    with unittest.TestCase().assertRaises(v4.BeliefControlError):
        v4.resolve_claim(owner,claim,by,parsed,sha)
    MEASURED['local_rejections'] += 1


class OwnerNamedResolution(unittest.TestCase):
    def test_all_four_families_contexts_variants_schedules_styles_and_shared_ids(self):
        for index in (12,13,18,19):
            for context in v2.CONTEXTS:
                for variant in v3.VARIANTS:
                    for style in ('forward','reverse'):
                        owner,_,by,parsed,sha = local(index,context,variant,(1,2,1),style)
                        claims = [c for c in owner['family_control_claims'] if c.get('name') in v4.INHERITED]
                        self.assertEqual(claims[0]['control_id'],claims[1]['control_id'])
                        for claim in claims:
                            proof = v4.resolve_claim(owner,claim,by,parsed,sha)
                            self.assertTrue(proof['satisfied']); self.assertEqual(proof['owner_is_pair_member'],context in 'EFGH')
                            self.assertEqual(proof['owner_coordinate']['length'],3 if context in 'DFH' else 2)
        MEASURED['local_resolution_proofs'] += 256

    def test_property_instead_of_name_and_nonmember_owner(self):
        owner,claim,by,parsed,sha = local(context='C'); claim['property'] = claim.pop('name')
        owner['record_hash'] = v4.record_hash(owner)
        proof = v4.resolve_claim(owner,claim,by,parsed,sha)
        self.assertFalse(proof['owner_is_pair_member']); self.assertEqual(proof['derived_comparison_prefix_evidence']['left']['surviving_worlds'],[])

    def test_owner_coordinate_and_actual_length_fail(self):
        for key,value in [('source_family_key','foreign'),('auxiliary_scalar_variant','transposed'),('schedule_index',1),('renderer','reverse')]:
            owner,claim,by,parsed,sha = local(); owner[key] = value; reject(owner,claim,by,parsed,sha)
        owner,claim,by,parsed,sha = local(context='D'); pair = [item['sample_id'] for item in by.values() if item['controlled_context'] in 'EG']; claim['pair_sample_ids'] = pair
        reject(owner,claim,by,parsed,sha)

    def test_reversed_foreign_unknown_and_equal_pair_ids_fail(self):
        for mode in ('reverse','foreign','unknown','same'):
            owner,claim,by,parsed,sha = local()
            if mode == 'reverse': claim['pair_sample_ids'].reverse()
            elif mode == 'unknown': claim['pair_sample_ids'][0] = 'not-an-ID'
            elif mode == 'same': claim['pair_sample_ids'][1] = claim['pair_sample_ids'][0]
            else:
                _,foreign,foreign_by,_,_ = local(index=13); by.update(foreign_by); claim['pair_sample_ids'] = foreign['pair_sample_ids']
            reject(owner,claim,by,parsed,sha)

    def test_duplicate_lookup_key_even_identical_rejected(self):
        owner,claim,by,parsed,sha = local(); entry = next(e for e in parsed['supplied_same_table_v2_pairs'] if e['name'] == claim['name'] and e['control_id'] == claim['control_id']); parsed['supplied_same_table_v2_pairs'].append(deepcopy(entry))
        reject(owner,claim,by,parsed,sha)

    def test_missing_name_id_hash_and_typed_manifest_coordinates_fail(self):
        for key,value in [('name',None),('control_id',None),('exact_candidate_sha256',None),('observation_length',True),('observation_length','2'),('source_family_key','foreign'),('auxiliary_scalar_variant',True),('scope','audit_probe'),('exact_candidate_sha256','0'*64)]:
            owner,claim,by,parsed,sha = local(); entry = next(e for e in parsed['supplied_same_table_v2_pairs'] if e['name'] == claim['name'] and e['control_id'] == claim['control_id'])
            if value is None: entry.pop(key)
            else: entry[key] = value
            reject(owner,claim,by,parsed,sha)

    def test_unpublished_annotations_and_pair_hashes_not_required(self):
        owner,claim,by,parsed,sha = local(); entry = next(e for e in parsed['supplied_same_table_v2_pairs'] if e['name'] == claim['name'] and e['control_id'] == claim['control_id']); entry.pop('own_optional_diagnostics'); entry['arbitrary_extra_annotation'] = {'private_syntax_is_not_a_gate':None}
        self.assertTrue(v4.resolve_claim(owner,claim,by,parsed,sha)['satisfied'])

    def test_closed_claim_shape_unknown_name_scope_and_cross_scope(self):
        for change in (lambda c:c.update(extra='not-public'),lambda c:c.update(property=c['name']),lambda c:c.update(name='unknown-property'),lambda c:c.update(scope='outside'),lambda c:c.update(probe_id='corpus-cannot-be-probe'),lambda c:c.pop('control_id'),lambda c:c.pop('evidence_manifest')):
            owner,claim,by,parsed,sha = local(); change(claim); reject(owner,claim,by,parsed,sha)

    def test_bool_schedule_shifts_and_assignment_indices_fail(self):
        for change in (lambda r:r.update(schedule_index=False),lambda r:r.update(id_shift=False),lambda r:r['query_role_assignments'][0].update(position=False)):
            item = row(); change(item); item['record_hash'] = v4.record_hash(item); self.assertTrue(v4.check(item))

    def test_world_query_prefix_difference_and_compound_edit_fail(self):
        changes = [lambda r:r['semantic_task']['worlds'][0]['facts'].update(feature=999),lambda r:r['semantic_task']['queries'].reverse(),lambda r:r['semantic_task']['observations'][0].update(equals='absent'),lambda r:r['semantic_task']['observations'][1].update(fact='feature')]
        for change in changes:
            owner,claim,by,parsed,sha = local(); right = by[claim['pair_sample_ids'][1]]; change(right); old.old.old.render(right); right['record_hash'] = v4.record_hash(right); reject(owner,claim,by,parsed,sha)


class TypedFiniteProof(unittest.TestCase):
    def test_exhaustive_typed_equalities_pair_differences(self):
        values = (True,1,False,0,'1','0',None,'unknown')
        for old_value,new_value in itertools.permutations(values,2):
            worlds = [{'world_id':'a','facts':{'x':True,'y':new_value,'q':None}}, {'world_id':'b','facts':{'x':False,'y':old_value,'q':1}}]
            left = {'kind':'belief','worlds':worlds,'observations':[{'fact':'x','equals':True},{'fact':'y','equals':old_value}],'queries':[{'query_id':'typed','fact':'y'}]}
            right = deepcopy(left); right['observations'][1]['equals'] = new_value
            proof = v2.matched_pair(left,right)
            self.assertEqual(proof['left']['surviving_worlds'],[]); self.assertEqual(proof['right']['surviving_worlds'],['a'])
            self.assertEqual(v1._json(proof['right']['queries'][0]['counts']),v1._json([{'value':new_value,'count':1}]))
            self.assertEqual(proof['typed_old_equals'],[type(old_value).__name__,old_value]); MEASURED['exhaustive_typed_pairs'] += 1
        self.assertEqual(MEASURED['exhaustive_typed_pairs'],56)

    def test_tombstone_like_null_not_empty_and_repeat_first_exact(self):
        left,right = old.old.tiny(3); proof = v2.matched_pair(left,right)
        self.assertTrue(proof['right']['queries'][0]['entailed']); self.assertIsNone(proof['right']['queries'][0]['guaranteed']); self.assertFalse(proof['left']['queries'][0]['entailed'])
        right['observations'][2]['equals'] = 1
        with self.assertRaises(v4.BeliefControlError): v2.matched_pair(left,right)


class ByteCustody(unittest.TestCase):
    def setUp(self): self.rows = [row(context=c) for c in v2.CONTEXTS]; self.raw,self.sources,self.parsed = bundle(self.rows)

    def test_canonical_hash_typed_and_actual_candidate_bytes(self):
        a={'value':True};b={'value':1};self.assertNotEqual(v4.record_hash(a),v4.record_hash(b)); self.assertEqual(v4._candidate(self.raw)[2],hashlib.sha256(self.raw).hexdigest())
        _,by,sha=v4._candidate(self.raw);self.assertEqual(len(v4._bind_rows(self.rows,by,sha)),8)

    def test_acceptance_review_fields_and_preserved_candidate_hash(self):
        accepted=deepcopy(self.rows)
        for item in accepted:
            item.update(review_status='ACCEPTED',reviewed_candidate_hash=item['record_hash'],review_receipt_ids=['own-hand-execution']);item['record_hash']=v4.record_hash(item)
        _,by,sha=v4._candidate(self.raw);self.assertEqual(len(v4._bind_rows(accepted,by,sha)),8)
        accepted[0]['generator_version']='changed-beyond-review';accepted[0]['record_hash']=v4.record_hash(accepted[0])
        with self.assertRaises(v4.BeliefControlError):v4._bind_rows(accepted,by,sha)

    def test_candidate_hash_cannot_be_accepted_file_hash(self):
        actual_sha=hashlib.sha256(self.raw).hexdigest(); changed=self.raw+b'\n'
        self.assertNotEqual(actual_sha,hashlib.sha256(changed).hexdigest())
        owner,claim,by,parsed,_=local();reject(owner,claim,by,parsed,hashlib.sha256(changed).hexdigest())

    def test_mutation_without_hash_rebind_and_incomplete_custody_fail(self):
        bad=deepcopy(self.rows);bad[0]['semantic_task']['worlds'][0]['facts']['feature']=True
        with self.assertRaises(v4.BeliefControlError):v4._candidate(candidate(bad))
        _,by,sha=v4._candidate(self.raw)
        with self.assertRaises(v4.BeliefControlError):v4._bind_rows(self.rows[:-1],by,sha)
        bad=deepcopy(self.rows);bad[0]['sample_id']=bad[1]['sample_id'];bad[0]['record_hash']=v4.record_hash(bad[0])
        with self.assertRaises(v4.BeliefControlError):v4._candidate(candidate(bad))

    def test_duplicate_json_keys_utf8_nonfinite_and_no_pointer_acceptance(self):
        for raw in (b'{"x":1,"x":true}',b'{"x":NaN}',b'\xff'):
            with self.assertRaises((v4.BeliefControlError,UnicodeError)):v4._strict_json(raw,'hand')
        with self.assertRaises(v4.BeliefControlError):v4._candidate(None)
        with self.assertRaises(v4.BeliefControlError):v4._manifest(MANIFEST_PATH,MANIFEST_PATH)

    def test_open_manifest_and_repository_relative_path_only(self):
        parsed,_=v4._manifest(v1._json({'open_aggregate_metadata':True}).encode(), 'aggregate.json');self.assertEqual(parsed,{'open_aggregate_metadata':True})
        for path in ('/outside.json','../outside.json','C:\\outside.json','.'): 
            with self.assertRaises(v4.BeliefControlError):v4._path(path)


class CompletePublicCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[row(index,context,variant,shifts,style) for index in (12,13,18,19) for context in v2.CONTEXTS for variant in v3.VARIANTS for shifts in itertools.product(range(3),repeat=3) for style in ('forward','reverse')]
        cls.raw,cls.sources,cls.parsed=bundle(cls.rows)
        cls.report=v4.check_batch(cls.rows,cls.sources,cls.raw,previous_rows=deepcopy(cls.rows));MEASURED.update(fixture_rows=len(cls.rows),full_runs=1)

    def test_complete_profiles_all27_schedules_counts_and_zero_errors(self):
        self.assertFalse(self.report['errors'],self.report['errors'][:6])
        expected={'records':3456,'source_families':4,'semantic_base_contexts':32,'scalar_variant_base_cases':64,'conditional_crossing_histograms':128,'scalar_pairs':1728,'field_order_pairs':1728,'opposite_status_same_observation_pairs':864,'v4_resolved_inherited_claims':6912,'v4_distinct_named_pair_entries':1728,'v4_bound_records':3456,'v4_nonmember_owner_claims':3456,'v4_manifest_files':1}
        for key,value in expected.items():self.assertEqual(self.report['counts'][key],value)
        self.assertEqual(self.report['v4_claim_length_counts'],{'2':4320,'3':2592});self.assertEqual(self.report['v4_preservation']['status'],'PROVED');self.assertEqual(self.report['v4_preservation']['added_source_families'],0)

    def test_actual_all_entry_hashes_and_manifest_bytes(self):
        self.assertTrue(all(p['exact_candidate_sha256']==hashlib.sha256(self.raw).hexdigest() and p['derived_comparison_prefix_evidence']['satisfied'] for p in self.report['v4_resolver_proofs']))
        self.assertEqual(self.report['v4_manifest_byte_bindings'][0]['sha256'],hashlib.sha256(self.sources[MANIFEST_PATH]).hexdigest());self.assertEqual(len(self.parsed['supplied_same_table_v2_pairs']),1728)

    def test_missing_schedule_and_changed_histogram_fail_closed(self):
        reduced=self.rows[:-1];raw,sources,_=bundle(reduced);report=v4.check_batch(reduced,sources,raw);MEASURED['full_runs']+=1
        self.assertTrue(report['errors']);self.assertTrue(any('27' in error or 'missing' in error or '864' in error for error in report['errors']))

    def test_profiles_tags_locators_and_schedule_corruptions(self):
        for change in (lambda r:r.update(structural_family_profile='typed_path'),lambda r:r.update(schedule_index=3),lambda r:r['skill_tags'].append('post_empty_absorption'),lambda r:r.update(property_witnesses={'missing':{'query_ids':[],'observation_indices':[],'prefix_indices':[]}})):
            item=row();change(item);item['record_hash']=v4.record_hash(item);self.assertTrue(v4.check(item))

    def test_audit_probe_cannot_substitute_supplied_inherited_pair(self):
        item=row(); name=sorted(v4.INHERITED)[0]
        for claim in item['family_control_claims']:
            if claim.get('name') in v4.INHERITED:
                claim.pop('pair_sample_ids');claim['scope']='audit_probe';claim['probe_id']='typed-probe-identity-'+claim['name']
        item['record_hash']=v4.record_hash(item);self.assertEqual(v4.check(item),[])
        # Its original inherited corpus entries remain in the other complete rows.
        # A conflicting reused probe ID for two different properties is rejected
        # by the unchanged execution ledger, rather than trusted as a pointer.
        chosen=deepcopy(self.rows)
        for r in chosen:
            r['family_control_claims']=[c for c in r['family_control_claims'] if c.get('name') not in v4.INHERITED]
            for p in sorted(v4.INHERITED):r['family_control_claims'].append({'name':p,'scope':'audit_probe','probe_id':'one-conflicting-typed-ID'})
            r['record_hash']=v4.record_hash(r)
        raw=candidate(chosen);report=v4.check_batch(chosen,{},raw);MEASURED['full_runs']+=1
        self.assertTrue(any('conflict' in error for error in report['errors']));self.assertTrue(any('missing required declared v2 supplied' in error for error in report['errors']))

    def test_stored_target_cannot_replace_semantics_and_typed_preservation(self):
        previous=deepcopy(self.rows[:8]);current=deepcopy(previous)
        for r in current:r['verified_target']={'producer_annotation':'ignored by answer derivation'}
        self.assertEqual(v1.derive(current[0]['semantic_task'])['target'],previous[0]['verified_target']);self.assertEqual(v4.preservation(current,previous)['status'],'FAILED')
        current=deepcopy(previous);current[0]['semantic_task']['worlds'][0]['facts']['feature']=True
        self.assertEqual(v4.preservation(current,previous)['status'],'FAILED');self.assertEqual(v4.preservation(self.rows,None)['status'],'NOT_RUN')

    def test_full_diagnostics_replay_tie_is_deterministic_across_hashseeds(self):
        script='import importlib.util,json; s=importlib.util.spec_from_file_location("own",'+repr(str(BASE/'tests/oracle_belief_controls_v3_test.py'))+'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.equal_score_diagnostics(),ensure_ascii=False,sort_keys=True,separators=(",",":")))'
        outputs=[]
        for seed in ('17','9001'):
            env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED=seed,PYTHONPATH=str(BASE))
            process=subprocess.run([sys.executable,'-c',script],cwd=BASE,env=env,capture_output=True,check=True,timeout=30);outputs.append(process.stdout)
            MEASURED['replay_seeds'].append({'seed':seed,'sha256':hashlib.sha256(process.stdout).hexdigest(),'bytes':len(process.stdout)})
        self.assertEqual(outputs[0],outputs[1])


if __name__ == '__main__':
    program=unittest.main(verbosity=2,exit=False)
    print('ACTUAL_V4_HANDMADE_COUNTS '+json.dumps(MEASURED,sort_keys=True))
    sys.exit(not program.result.wasSuccessful())
