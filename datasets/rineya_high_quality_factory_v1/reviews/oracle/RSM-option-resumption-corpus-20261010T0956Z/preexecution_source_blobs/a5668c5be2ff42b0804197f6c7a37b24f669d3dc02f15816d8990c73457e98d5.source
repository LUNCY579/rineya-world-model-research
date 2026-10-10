"""Handwritten public R2 corpus regressions; no production artifacts."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).parent
def local(name):
    spec=importlib.util.spec_from_file_location('rsm_corpus_'+name,HERE/(name+'.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
C=local('corpus_checker_attempt01'); H=local('hand_fixture_recipe_attempt01')
RESULTS={}

class CorpusPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=H.build(C)
        cls.original=C.check_batch(cls.b['rows'],cls.b['manifest_sources'],cls.b['source_bytes'],cls.b['candidate_bytes'],cls.b['legacy_input_bytes'])
        # Actual fixture bytes/results are post-import execution evidence, while
        # source/recipe/public/config/legacy inputs were frozen pre-import.
        (HERE/'actual-hand-candidate-attempt01.jsonl').write_bytes(cls.b['candidate_bytes'])
        (HERE/'actual-hand-manifest-attempt01.json').write_bytes(cls.b['manifest_sources'][cls.b['manifest_path']])
        (HERE/'actual-positive-report-attempt01.json').write_text(json.dumps(cls.original,sort_keys=True,indent=2)+'\n')
        RESULTS['positive']=cls.original
        RESULTS['fixture_bytes']={'candidate_sha256':sha256(cls.b['candidate_bytes']).hexdigest(),'candidate_bytes':len(cls.b['candidate_bytes']),
          'manifest_sha256':sha256(cls.b['manifest_sources'][cls.b['manifest_path']]).hexdigest(),'manifest_bytes':len(cls.b['manifest_sources'][cls.b['manifest_path']])}

    def bundle(self): return deepcopy(self.b)
    def run_batch(self,b,refresh=True):
        if refresh: H.recustody(C,b)
        return C.check_batch(b['rows'],b['manifest_sources'],b['source_bytes'],b['candidate_bytes'],b['legacy_input_bytes'])
    def reject(self,b,needle,refresh=True):
        r=self.run_batch(b,refresh); self.assertTrue(any(needle in x for x in r['errors']),r['errors']); return r
    def tasks(self,family=0):
        rows=self.b['rows'][family*18:(family+1)*18]
        return {r['controlled_context']:deepcopy(r['semantic_task']) for r in rows if r['renderer']=='forward'}

    def test_full216_all300_relations_and_actual_legacy_novelty(self):
        self.assertEqual(self.original['errors'],[])
        expected={'records':216,'source_families':12,'semantic_tasks':108,'owner_claims':1944,'nonmember_first8_claims':1344,'resolved_named_pairs':300,'manifest_named_entries':300,'proved_named_pairs':300,'train_records':144,'dev_challenge_records':72}
        for k,v in expected.items(): self.assertEqual(self.original['counts'][k],v)
        for v in self.original['novelty']['graph_views'].values():
            self.assertEqual(v['proposed_distinct_signatures'],12); self.assertEqual(v['legacy_distinct_signatures'],8); self.assertEqual(v['exact_new_legacy_collisions'],[])
        self.assertEqual(len(self.original['families']),12)

    def test_standalone_rows_shared_id_names_nonmember_optional_forms(self):
        for row in self.b['rows']: self.assertEqual(C.check(row),[])
        row=self.b['rows'][8]; first=row['family_control_claims'][0]
        self.assertNotIn(row['sample_id'],first['pair_sample_ids'])
        self.assertEqual(row['family_control_claims'][0]['control_id'],row['family_control_claims'][1]['control_id'])
        self.assertTrue(first['control_id'].startswith(' '))

    def test_empty_witnesses_and_external_hash_omission_allowed(self):
        b=self.bundle()
        for e in b['manifest']['control_pairs']: e.pop('pair_record_hashes')
        self.assertEqual(self.run_batch(b)['errors'],[])

    def test_literal_whitespace_control_id_allowed(self):
        b=self.bundle(); old=b['rows'][0]['family_control_claims'][0]['control_id']
        for row in b['rows']:
            for c in row['family_control_claims']:
                if c['control_id']==old: c['control_id']=' '
        for e in b['manifest']['control_pairs']:
            if e['control_id']==old: e['control_id']=' '
        self.assertEqual(self.run_batch(b)['errors'],[])

    def test_claim_canonical_member_hash_forbidden_cycle(self):
        b=self.bundle(); b['rows'][0]['family_control_claims'][-1]['pair_record_hashes']=['0'*64,'1'*64]
        self.reject(b,'unsupported canonical claim field')

    def test_external_hash_wrong_missing_member_reordered_typed(self):
        for value in (['0'*64,'1'*64],['0'*64],['A'*64,'B'*64],[True,'1'*64]):
            b=self.bundle(); H.recustody(C,b); b['manifest']['control_pairs'][0]['pair_record_hashes']=value
            b['manifest_sources']={b['manifest_path']:C.canonical(b['manifest']).encode()}
            self.reject(b,'external actual ordered full member hashes',False)
        b=self.bundle(); H.recustody(C,b); e=b['manifest']['control_pairs'][0]; e['pair_record_hashes'].reverse()
        b['manifest_sources']={b['manifest_path']:C.canonical(b['manifest']).encode()}
        self.reject(b,'external actual ordered full member hashes',False)

    def test_control_id_types_names_scopes_closed_claims(self):
        mutations=[('control_id',''),('control_id',1),('control_id',True),('scope','audit_probe'),('property','unknown'),('undeclared',{})]
        for key,value in mutations:
            row=deepcopy(self.b['rows'][0]); row['family_control_claims'][0][key]=value; row['record_hash']=C.record_hash(row)
            self.assertTrue(C.check(row),(key,value))
        row=deepcopy(self.b['rows'][0]); row['family_claims']=row['family_control_claims']; row['record_hash']=C.record_hash(row)
        self.assertTrue(any('superseded' in e for e in C.check(row)))

    def test_duplicate_resolver_key_and_missing_named_entry(self):
        b=self.bundle(); b['manifest']['control_pairs'].append(deepcopy(b['manifest']['control_pairs'][0]))
        self.reject(b,'duplicate resolver key')
        b=self.bundle(); b['manifest']['control_pairs'].pop(0); self.reject(b,'missing exact named resolver entry')

    def test_wrong_owner_nonmember_coordinates_and_foreign_pair(self):
        b=self.bundle(); b['rows'][8]['family_control_claims'][0]['source_family_key']='elsewhere'
        self.reject(b,'claim owner source_family_key')
        b=self.bundle(); b['rows'][0]['family_control_claims'][0]['pair_sample_ids'][0]=b['rows'][18]['sample_id']
        self.reject(b,'claim foreign source/index/split')

    def test_reversed_pairs_mixed_renderer_and_missing_actual_ids(self):
        for newids in (list(reversed(self.b['rows'][0]['family_control_claims'][0]['pair_sample_ids'])),
                       [self.b['rows'][1]['sample_id'],self.b['rows'][2]['sample_id']],['not-supplied','other-probe']):
            b=self.bundle(); b['rows'][0]['family_control_claims'][0]['pair_sample_ids']=newids
            r=self.run_batch(b); self.assertTrue(any('ordered control' in e or 'missing actual member' in e for e in r['errors']),r['errors'])

    def test_entry_coordinates_boolean_index_and_candidate_hash(self):
        for field,value,needle in [('source_family_index',True,'entry source coordinates'),('controlled_contexts',['R','F'],'entry actual ordered coordinates'),('candidate_sha256','0'*64,'entry complete candidate SHA')]:
            b=self.bundle(); H.recustody(C,b); b['manifest']['control_pairs'][0][field]=value
            b['manifest_sources']={b['manifest_path']:C.canonical(b['manifest']).encode()}; self.reject(b,needle,False)

    def test_six_sources_legacy_and_all_raw_inputs_required(self):
        for field in ('manifest_sources','source_bytes','candidate_bytes','legacy_input_bytes'):
            b=self.bundle(); b[field]=None
            r=self.run_batch(b,False); self.assertTrue(any('UNKNOWN' in e for e in r['errors']),(field,r['errors']))
        b=self.bundle(); path=list(C.PUBLIC)[-1]; del b['manifest']['source_bindings'][path]
        self.reject(b,'six public source bindings')
        b=self.bundle(); b['source_bytes'][list(C.PUBLIC)[0]]+=b'\n'; self.reject(b,'operative source')
        b=self.bundle(); b['legacy_input_bytes']+=b'\n'; self.reject(b,'target-free frozen legacy')

    def test_paths_absolute_escape_symlink_and_duplicate_json_keys(self):
        for value in ('/tmp/manifest','../escape','a/../b','a//b','a\\b'):
            row=deepcopy(self.b['rows'][0]); row['family_control_claims'][0]['evidence_manifest']=value; row['record_hash']=C.record_hash(row)
            self.assertTrue(C.check(row),value)
        link=HERE/'actual-test-symlink-attempt01'
        try:
            link.symlink_to(HERE/'nonexistent-destination')
            rel=str(link.relative_to(C.ROOT))+'/manifest.json'
            with self.assertRaisesRegex(ValueError,'symlink'): C.safe_path(rel)
        finally: link.unlink(missing_ok=True)
        with self.assertRaisesRegex(ValueError,'duplicate JSON'): C.strict_json(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError): C.strict_json(b'{"x":NaN}')

    def test_accepted_projection_exact_original_pending_bytes(self):
        b=self.bundle(); originals={r['sample_id']:r['record_hash'] for r in b['rows']}
        for row in b['rows']:
            row['review_status']='ACCEPTED'; row['review_receipt_ids']=['hand-reviewed-receipt']; row['reviewed_candidate_hash']=originals[row['sample_id']]; row['record_hash']=C.record_hash(row)
        self.assertEqual(self.run_batch(b,False)['errors'],[])
        accepted_raw=b''.join((C.canonical(r)+'\n').encode() for r in b['rows']); b['candidate_bytes']=accepted_raw
        self.reject(b,'PENDING full record custody',False)

    def test_accepted_immutable_task_and_lineage_changes_rejected(self):
        b=self.bundle(); row=b['rows'][0]; row['review_status']='ACCEPTED'; row['reviewed_candidate_hash']=row['record_hash']; row['skill_tags'].append('changed'); row['record_hash']=C.record_hash(row)
        self.reject(b,'accepted typed immutable projection',False)
        b=self.bundle(); row=b['rows'][0]; row['review_status']='ACCEPTED'; row['reviewed_candidate_hash']='0'*64; row['record_hash']=C.record_hash(row)
        self.reject(b,'accepted original record lineage',False)

    def test_raw_candidate_whitespace_exact_byte_binding(self):
        b=self.bundle(); b['candidate_bytes']+=b'\n'
        self.reject(b,'whole PENDING candidate SHA',False)

    def test_missing_cohort_style_parent_split_and_indices(self):
        b=self.bundle(); b['rows'].pop(); self.reject(b,'all nine contexts/two renderers')
        b=self.bundle(); b['rows'][1]['semantic_parent_id']='wrong-parent'; self.reject(b,'field-order typed/list/parent')
        b=self.bundle(); b['rows'][0]['split']='dev_challenge'; self.reject(b,'split')
        row=deepcopy(self.b['rows'][0]); row['source_family_index']=True; row['record_hash']=C.record_hash(row); self.assertTrue(C.check(row))

    def test_renderer_claim_no_proof_and_list_order_not_reversible(self):
        b=self.bundle(); row=b['rows'][1]; row['rendered_input']=b['rows'][0]['rendered_input']; self.reject(b,'field-order typed/list/parent/presentation')
        b=self.bundle(); row=b['rows'][1]; row['semantic_task']['options']['root']['body'].reverse(); H.rerender(C,row)
        self.reject(b,'field-order typed/list/parent/presentation')

    def test_student_contract_and_exact_typed_input(self):
        row=deepcopy(self.b['rows'][0]); row['rendered_input']='shortened\nTask JSON:\n'+C.canonical(row['semantic_task']); row['record_hash']=C.record_hash(row)
        self.assertTrue(any('complete unchanged contract' in e for e in C.check(row)))
        row=deepcopy(self.b['rows'][0]); t=deepcopy(row['semantic_task']); t['version']=True
        row['rendered_input']=row['rendered_input'].rsplit('\nTask JSON:\n',1)[0]+'\nTask JSON:\n'+C.canonical(t); row['record_hash']=C.record_hash(row)
        self.assertTrue(any('rendered task typed mismatch' in e for e in C.check(row)))

    def test_difficulty_and_witness_boolean_integer_distinctions(self):
        for key,value in [('difficulty',True),('structural_difficulty',3.0),('property_witnesses',{'dependency_metrics':{'total_F_events':True}}),('property_witnesses',{'saved_binding_locator':['frames',True,'bindings','a']}),('property_witnesses',{'replay_locator':{'option':'root','body_path':[],'index':False}})]:
            row=deepcopy(self.b['rows'][0]); row[key]=value; row['record_hash']=C.record_hash(row); self.assertTrue(C.check(row),(key,value))
        b=self.bundle(); b['rows'][0]['property_witnesses']={'dependency_metrics':{'total_F_events':1}}; self.reject(b,'false dependency metric')

    def test_actual_optional_locators_metrics_and_active_copy(self):
        b=self.bundle(); facts=self.original['families'][0]
        for row in b['rows'][:18]: row['property_witnesses']={'dependency_metrics':deepcopy(facts['metrics']),'saved_binding_locator':facts['saved_binding_locators'][0],'checkpoint_tamper_locator':facts['checkpoint_tamper_locator'],'replay_locator':facts['replay_locators'][0],'diagnostics':{'private_annotation':'no required format'}}
        self.assertEqual(self.run_batch(b)['errors'],[])
        b['rows'][0]['property_witnesses']['saved_binding_locator']=['frames',1,'bindings','b']
        # The child option binding is not the particular branch copy read next.
        if ['frames',1,'bindings','b'] not in facts['saved_binding_locators']: self.reject(b,'inactive/unrelated saved frame locator')

    def test_invalid_task_history_checkpoint_error_precedence(self):
        task=self.tasks()['R']; task['version']=True; task['history'][0]['instance_id']=True; task['checkpoint']['state']['x']=True
        self.assertEqual(C.solve(task)['error'],'invalid_task')
        task=self.tasks()['R']; task['history'][0]['instance_id']=False; task['checkpoint']['state']['x']=True
        self.assertEqual(C.solve(task)['error'],'invalid_history')
        task=self.tasks()['R']; task['checkpoint']['state']['x']=True
        self.assertEqual(C.solve(task)['error'],'invalid_checkpoint')

    def test_each_relation_rejects_false_input_witness(self):
        mutations=[('R','mode','restart'),('T','mode','resume'),('P','event_budget',0),('U','checkpoint',None),('A','entry',None),('I','entry',None),('S','options',None),('X','checkpoint',None)]
        for context,key,value in mutations:
            tasks=self.tasks()
            if context in ('A','I'): value=deepcopy(tasks['F']['entry'])
            if context=='S': value=deepcopy(tasks['F']['options'])
            if context=='X': value=deepcopy(tasks['R']['checkpoint'])
            tasks[context][key]=value
            with self.assertRaises(ValueError,msg=context): C.family_analysis(tasks)

    def test_non_idempotent_replay_truth_not_operation_name(self):
        state={'x':2,'y':2}; bind={'a':3}
        for i in ({'op':'add','register':'x','value':{'const':0}}, {'op':'scale','register':'x','value':{'const':0}}, {'op':'scale','register':'x','value':{'const':1}}, {'op':'swap'}):
            one=C.apply_effect(i,state,bind); two=C.apply_effect(i,one,bind); self.assertTrue(C.eq(one,two))
        i={'op':'add','register':'x','value':{'reg':'x'}}; one=C.apply_effect(i,state,bind); two=C.apply_effect(i,one,bind)
        self.assertEqual((one['x'],two['x']),(4,8)); self.assertFalse(C.eq(one,two))
        base=self.tasks()['F']; base['options']['root']['initiation']['all'].append({'left':{'reg':'x'},'cmp':'lt','right':{'const':6}})
        tasks=H.make_contexts(C,base)
        with self.assertRaisesRegex(ValueError,'restart root initiation'): C.family_analysis(tasks)

    def test_unused_argument_enter_only_not_positive_control(self):
        base=self.tasks()['F']; base['options']['root']['params'].append('unused'); base['entry']['args']['unused']={'const':0}
        tasks=H.make_contexts(C,base); tasks['A']=deepcopy(tasks['F']); tasks['A']['entry']['args']['unused']={'const':1}
        self.assertEqual(C.solve(tasks['A'])['status'],'completed')
        with self.assertRaisesRegex(ValueError,'later behavior difference'): C.family_analysis(tasks)

    def test_inactive_counter_and_unused_frame_tamper_rejected(self):
        tasks=self.tasks(); tasks['X']=deepcopy(tasks['R']); tasks['X']['checkpoint']['next_instance_id']+=1
        with self.assertRaisesRegex(ValueError,'tamper affects pending'): C.family_analysis(tasks)
        tasks=self.tasks(); tasks['X']=deepcopy(tasks['R']); tasks['X']['checkpoint']['frames'][1]['bindings']['b']+=1
        with self.assertRaisesRegex(ValueError,'tamper affects pending'): C.family_analysis(tasks)

    def test_P_pending_return_suffices_and_pause_without_effect_fails(self):
        tasks=self.tasks(); r=C.solve(tasks['R']); tasks['P']['event_budget']=len(r['new_events'])-1
        p=C.solve(tasks['P']); self.assertEqual(p['next_event']['event'],'return'); self.assertGreaterEqual(p['new_effects'],1)
        tasks['U']['history']=tasks['R']['history']+p['new_events']; tasks['U']['checkpoint']=p['checkpoint']
        self.assertIsInstance(C.family_analysis(tasks),dict)
        tasks=self.tasks(3); tasks['P']['event_budget']=1
        self.assertEqual(C.solve(tasks['P'])['new_effects'],0)
        with self.assertRaisesRegex(ValueError,'P commits effect'): C.family_analysis(tasks)

    def test_guard_decisiveness_success_no_rollback_and_compound_change(self):
        tasks=self.tasks(); tasks['I']['entry']['args']['a']={'const':3}
        with self.assertRaisesRegex(ValueError,'I decisive'): C.family_analysis(tasks)
        tasks=self.tasks(); tasks['S']['options']['root']['success']['all'][1]['right']={'const':100000}
        with self.assertRaisesRegex(ValueError,'one root success literal'): C.family_analysis(tasks)
        original=C.family_analysis(self.tasks()); self.assertEqual(original['targets']['S']['checkpoint']['state'],original['targets']['F']['checkpoint']['state'])

    def test_semantic_row_order_and_object_order_invariance(self):
        b=self.bundle(); b['rows'].reverse(); self.assertEqual(self.run_batch(b)['errors'],[])
        f=self.tasks()['F']; reverse=lambda x: {k:reverse(v) for k,v in reversed(list(x.items()))} if type(x) is dict else [reverse(v) for v in x] if type(x) is list else x
        self.assertTrue(C.eq(C.solve(f),C.solve(reverse(f))))

    def test_graph_constants_names_do_not_establish_novelty(self):
        tasks=self.tasks(); f=tasks['F']; altered=deepcopy(f); altered['initial_state']['x']=5; altered['options']['root']['body'][0]['value']={'const':99}
        # Scalar/parameter collapse deliberately coarsens the shared views.
        self.assertEqual(C.GRAPH.shared_primitive_signature(f),C.GRAPH.shared_primitive_signature(altered))
        altered=deepcopy(f); altered['initial_state']['x']=5
        self.assertEqual(C.GRAPH.full_option_signature(f),C.GRAPH.full_option_signature(altered))

    def test_external_binding_extras_truthful_only_and_unknown_source(self):
        b=self.bundle(); b['manifest']['source_bindings']['hand/extra.txt']='0'*64; self.reject(b,'asserted raw source binding')
        b=self.bundle(); b['source_bytes']['hand/extra.txt']=b'hand-extra'; b['manifest']['source_bindings']['hand/extra.txt']=sha256(b'hand-extra').hexdigest()
        self.assertEqual(self.run_batch(b)['errors'],[])

    def test_manifest_open_extra_entry_and_no_global_ID_uniqueness(self):
        self.assertLess(len({e['control_id'] for e in self.b['manifest']['control_pairs']}),300)
        b=self.bundle(); e=deepcopy(b['manifest']['control_pairs'][0]); e['control_id']='another truthful optional evidence ID'; b['manifest']['control_pairs'].append(e)
        report=self.run_batch(b); self.assertEqual(report['errors'],[]); self.assertEqual(report['counts']['additional_manifest_entries'],1)

    def test_full_record_hash_includes_targets_but_targets_never_prove_controls(self):
        b=self.bundle(); b['rows'][0]['verified_target']={'wrong_private_value':True}; old=b['rows'][0]['record_hash']
        self.assertNotEqual(old,C.record_hash(b['rows'][0])); self.assertEqual(self.run_batch(b)['errors'],[])
        # Semantic gold comparison belongs to separate solver/pipeline gates.

if __name__=='__main__': unittest.main()
