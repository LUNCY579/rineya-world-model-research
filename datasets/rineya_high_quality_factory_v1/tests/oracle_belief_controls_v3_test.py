"""Independent public-only v3 handmade source tables and corruption properties."""
from copy import deepcopy
import importlib.util
import itertools
import json
import hashlib
import os
from pathlib import Path
import sys
import subprocess
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracles import belief_controls_v3_independent as v3
from oracles import belief_controls_independent as v1
spec=importlib.util.spec_from_file_location('own_immutable_v2_fixture_helpers',Path(__file__).with_name('oracle_belief_controls_v2_test.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def world_table(index):
    cells={12:[(0,0,3),(0,1,3),(1,0,3),(1,1,3),(1,2,3),(2,1,3),(3,0,3)],13:[(0,0,3),(0,1,3),(1,0,3),(1,2,3),(2,0,3),(3,1,3)],18:[(0,0,4),(0,1,4),(1,0,1),(1,1,2),(1,2,1),(2,1,1)],19:[(0,0,3),(0,1,3),(1,0,1),(2,1,2)]}[index]
    e=3 if index==13 else 2;worlds=[]
    for x,y,n in cells:
        xvalue='stone' if x==e else True if index==19 and x==0 else x
        for member in range(n):
            var=([None,None,True,1] if index==18 else [None,False,0] if index in (13,19) else [None,True,1])[member] if (x,y)==(0,0) else 'fixed'
            worlds.append({'world_id':'world-'+str(len(worlds)),'facts':{'observation_control':xvalue,'observation_auxiliary':['linen','opal','moss'][y],'invariant_query':None if (x,y)==(0,0) else 'unknown','variation_query':var,'unoffered_feature':member}})
    return worlds


def fixture(index=12,context='A',variant='identity',shifts=(0,0,0),renderer='forward'):
    item=old.fixture(index,context,shifts,renderer);roles=item['fact_role_assignments'];u,v,w=shifts
    item['sample_id']=f'v3fixture-{index}-{context}-{variant}-{u}{v}{w}-{renderer}';item['generator_version']='rbm-belief-gap-3.0.0';item['auxiliary_scalar_variant']=variant
    control={'fact':roles['observation_auxiliary'],'baseline_value':'linen','supported_value':'opal'};item['auxiliary_scalar_control']=control
    task=item['semantic_task'];task['worlds']=[{'world_id':world['world_id'],'facts':{roles[role]:value for role,value in world['facts'].items()}} for world in world_table(index)]
    xvalue='stone' if context in 'EFGH' else True if index==19 else 0
    obs=[{'fact':roles['observation_control'],'equals':xvalue},{'fact':roles['observation_auxiliary'],'equals':'opal' if context in 'BGH' else 'linen'}]
    if context=='C':obs.reverse()
    if context in 'DFH':obs.append(deepcopy(obs[0]))
    task['observations']=obs
    if variant=='transposed':item['semantic_task']=v3.transpose(task,control)
    claims=[]
    for c in item['family_control_claims']:
        if c['name'] in v3.v2.NEW_FAMILY:continue
        claims.append(c)
    for lc,rc in [('E','G'),('F','H')]:
        pair=[f'v3fixture-{index}-{c}-{variant}-{u}{v}{w}-{renderer}' for c in (lc,rc)]
        for name in sorted(v3.v2.NEW_FAMILY):claims.append({'name':name,'scope':'corpus_variant','pair_sample_ids':pair,'control_id':'shared:'+':'.join(pair),'evidence_manifest':f'own-public-fixture/{index}/{variant}/{lc}{rc}.json'})
    scalarpair=[f'v3fixture-{index}-{context}-{v}-{u}{v0}{w}-{renderer}' for v in v3.VARIANTS for v0 in [shifts[1]]]
    # No sample-ID parsing is used by the checker; this helper names own rows.
    claims.append({'name':'auxiliary_scalar_transposition_invariance','scope':'corpus_variant','pair_sample_ids':scalarpair,'evidence_manifest':f'own-public-fixture/{index}/{context}-scalar.json'})
    for lc,rc in [('E','G'),('F','H')]:
        for lv,rv in [('identity','transposed'),('transposed','identity')]:
            claims.append({'name':'same_observations_auxiliary_table_counterfactual','scope':'corpus_variant','pair_sample_ids':[f'v3fixture-{index}-{c}-{sv}-{u}{v}{w}-{renderer}' for c,sv in ((lc,lv),(rc,rv))],'evidence_manifest':f'own-public-fixture/{index}/{lc}{rc}-{lv}.json'})
    claims.append({'name':'context_auxiliary_scalar_variant_crossing','scope':'corpus_variant','control_id':'whole-family-crossing','evidence_manifest':f'own-public-fixture/{index}/conditional-all.json'})
    item['family_control_claims']=claims;old.old.render(item);return item


def batch():return [fixture(index,context,variant,shifts,style) for index in (12,13,18,19) for context in v3.v2.CONTEXTS for variant in v3.VARIANTS for shifts in itertools.product(range(3),repeat=3) for style in ('forward','reverse')]


def tiny(context='E'):
    worlds=[{'world_id':'a','facts':{'x':'stone','y':'opal','q':'linen'}},{'world_id':'b','facts':{'x':'stone','y':'opal','q':True}},{'world_id':'c','facts':{'x':'stone','y':'opal','q':1}},{'world_id':'d','facts':{'x':'other','y':'linen','q':None}}]
    task={'kind':'belief','worlds':worlds,'observations':[{'fact':'x','equals':'stone'},{'fact':'y','equals':'opal' if context in 'GH' else 'linen'}],'queries':[{'query_id':'arbitrary-q','fact':'q'}]}
    if context in 'FH':task['observations'].append(deepcopy(task['observations'][0]))
    return task,{'fact':'y','baseline_value':'linen','supported_value':'opal'}


REPLAY_EVIDENCE=[]
def equal_score_diagnostics():
    left={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':'alpha','y':'other'}},{'world_id':'b','facts':{'x':'other','y':'beta'}}],'observations':[{'fact':'x','equals':'alpha'},{'fact':'y','equals':'beta'}],'queries':[{'query_id':'q','fact':'x'}]}
    right=deepcopy(left);right['worlds'][0]['facts']['y']='beta'
    rows=[{'semantic_task':task,'split':'train'} for task in (left,right)]
    members=[{'row_index':i,'sample_id':'independent-tie-'+str(i),'family':'tie-family','roles':{'x-role':'x','y-role':'y'},'analysis':v1.analyze(row['semantic_task'])} for i,row in enumerate(rows)]
    return v3.deterministic_diagnostics(v3.v2._diagnostics(members,rows))


class LocalTransposition(unittest.TestCase):
    def test_reused_python_observation_and_fact_dicts_are_occurrencewise(self):
        observation={'fact':'y','equals':'linen'};facts={'y':'linen','q':None}
        task={'kind':'belief','worlds':[{'world_id':'a','facts':facts},{'world_id':'b','facts':facts},{'world_id':'c','facts':{'y':'opal','q':1}}],'observations':[observation,observation],'queries':[{'query_id':'q','fact':'q'}]};ctl={'fact':'y','baseline_value':'linen','supported_value':'opal'}
        transformed=v3.transpose(task,ctl);proof=v3.scalar_pair(task,transformed,ctl)
        self.assertEqual([o['equals'] for o in transformed['observations']],['opal','opal']);self.assertEqual([w['facts']['y'] for w in transformed['worlds']],['opal','opal','linen']);self.assertEqual(proof['left']['surviving_worlds'],['a','b']);self.assertEqual(observation['equals'],'linen');self.assertEqual(facts['y'],'linen')
    def test_hand_derived_identity_target_and_involution(self):
        task,ctl=tiny('G');altered=v3.transpose(task,ctl);proof=v3.scalar_pair(task,altered,ctl)
        self.assertEqual(proof['left']['surviving_worlds'],['a','b','c']);self.assertEqual(proof['left']['belief_sizes'],[3,3])
        self.assertEqual(v1._json(proof['left']['queries'][0]['counts']),v1._json([{'value':'linen','count':1},{'value':1,'count':1},{'value':True,'count':1}]))
        self.assertEqual(altered['worlds'][0]['facts']['q'],'linen');self.assertEqual(v1._json(v3.transpose(altered,ctl)),v1._json(task))
    def test_both_lengths_both_same_observation_directions(self):
        for leftc,rightc in [('E','G'),('F','H')]:
            e,ctl=tiny(leftc);g,_=tiny(rightc)
            for left,right in [(e,v3.transpose(g,ctl)),(v3.transpose(e,ctl),g)]:
                proof=v3.same_observations_pair(left,right,ctl);self.assertEqual(proof['left']['surviving_worlds'],[]);self.assertEqual(proof['right']['surviving_worlds'],['a','b','c'])
    def test_selective_column_swap_and_global_literal_swap_fail(self):
        task,ctl=tiny('G');right=v3.transpose(task,ctl);right['worlds'][0]['facts']['y']='opal'
        with self.assertRaises(v3.BeliefControlError):v3.scalar_pair(task,right,ctl)
        right=v3.transpose(task,ctl);right['worlds'][0]['facts']['q']='opal'
        with self.assertRaises(v3.BeliefControlError):v3.scalar_pair(task,right,ctl)
    def test_swapping_observations_only_or_queried_auxiliary_fails(self):
        task,ctl=tiny();right=deepcopy(task);right['observations'][1]['equals']='opal'
        with self.assertRaises(v3.BeliefControlError):v3.scalar_pair(task,right,ctl)
        task['queries'][0]['fact']='y'
        with self.assertRaises(v3.BeliefControlError):v3.transpose(task,ctl)
    def test_nonstring_control_and_absent_string_fail(self):
        task,ctl=tiny();ctl['supported_value']=True
        with self.assertRaises(v3.BeliefControlError):v3.transpose(task,ctl)
        task,ctl=tiny();ctl['supported_value']='absent';right=v3.transpose(task,ctl)
        with self.assertRaises(v3.BeliefControlError):v3.scalar_pair(task,right,ctl)
    def test_same_observations_require_real_table_and_prefix(self):
        e,ctl=tiny('E');g,_=tiny('G')
        with self.assertRaises(v3.BeliefControlError):v3.same_observations_pair(e,g,ctl)
        right=v3.transpose(g,ctl);right['observations'][0]['equals']='other'
        with self.assertRaises(v3.BeliefControlError):v3.same_observations_pair(e,right,ctl)
    def test_exhaustive_local_tables_histories_and_typed_query_bins(self):
        histories=0
        for ys in itertools.product(('linen','opal','moss'),repeat=3):
            if not {'linen','opal'}<=set(ys):continue
            worlds=[{'world_id':str(i),'facts':{'y':y,'x':'linen' if i%2 else 'opal','q':value}} for i,(y,value) in enumerate(zip(ys,(None,True,1)))]
            observations=[{'fact':'y','equals':'linen'},{'fact':'y','equals':'opal'},{'fact':'x','equals':'linen'}]
            for n in range(3):
                for history in itertools.product(observations,repeat=n):
                    task={'kind':'belief','worlds':worlds,'observations':list(history),'queries':[{'query_id':'q','fact':'q'}]};ctl={'fact':'y','baseline_value':'linen','supported_value':'opal'};proof=v3.scalar_pair(task,v3.transpose(task,ctl),ctl)
                    expected=[w['world_id'] for w in worlds if all(type(w['facts'][o['fact']]) is type(o['equals']) and w['facts'][o['fact']]==o['equals'] for o in history)]
                    self.assertEqual(proof['left']['surviving_worlds'],sorted(expected));self.assertEqual(proof['left'],proof['right']);histories+=1
        self.assertEqual(histories,156)
    def test_rename_and_world_order_equivariance(self):
        task,ctl=tiny('G');before=v3.scalar_pair(task,v3.transpose(task,ctl),ctl)
        for world in task['worlds']:world['world_id']='renamed-'+world['world_id'];world['facts']={'renamed-'+f:v for f,v in world['facts'].items()}
        task['worlds'].reverse()
        for obs in task['observations']:obs['fact']='renamed-'+obs['fact']
        task['queries'][0].update(query_id='fresh-id',fact='renamed-q');ctl['fact']='renamed-y'
        after=v3.scalar_pair(task,v3.transpose(task,ctl),ctl);self.assertEqual(after['left']['belief_sizes'],before['left']['belief_sizes']);self.assertEqual(after['left']['surviving_worlds'],['renamed-a','renamed-b','renamed-c'])


class RowMetadata(unittest.TestCase):
    def test_all_context_variant_style_metadata(self):
        for index in (12,13,18,19):
            for context in v3.v2.CONTEXTS:
                for variant in v3.VARIANTS:self.assertEqual(v3.check(fixture(index,context,variant,(1,2,1),'reverse')),[])
    def test_closed_metadata_and_unpublished_alias_fail(self):
        changes=[lambda r:r.pop('auxiliary_scalar_variant'),lambda r:r.update(auxiliary_scalar_variant='swapped'),lambda r:r.update(scalar_variant='identity'),lambda r:r['auxiliary_scalar_control'].update(extra=0),lambda r:r['auxiliary_scalar_control'].update(baseline_value=True),lambda r:r['auxiliary_scalar_control'].update(supported_value='stone'),lambda r:r['auxiliary_scalar_control'].update(fact=r['semantic_task']['queries'][0]['fact'])]
        for change in changes:
            item=fixture();change(item);self.assertTrue(v3.check(item))
    def test_new_controls_are_not_row_tags_and_shared_key_scope(self):
        item=fixture();item['skill_tags'].append(next(iter(v3.NEW_FAMILY)));self.assertTrue(v3.check(item))
        item=fixture();a=item['family_control_claims'][-2];b=item['family_control_claims'][-1];a['control_id']=b['control_id']='same-physical-key';self.assertEqual(v3.check(item),[])
        b['scope']='audit_probe';b['probe_id']='must-execute';self.assertTrue(v3.check(item))
    def test_unknown_claim_and_prompt_annotation_fail(self):
        item=fixture();item['family_control_claims'][-1]['name']='unpublished';self.assertTrue(v3.check(item))
        item=fixture();item['rendered_input']='auxiliary_scalar_variant=identity\n'+item['rendered_input'];self.assertTrue(v3.check(item))


class DeterministicDiagnostics(unittest.TestCase):
    def test_equal_score_feature_and_polarity_tie_is_explicit(self):
        report=equal_score_diagnostics()
        for detail in report.values():
            best=detail['baselines']['strongest_typed_scalar_anywhere'];self.assertEqual(best['feature'],'["str","alpha"]');self.assertFalse(best['present_predicts_empty']);self.assertEqual(best['correct'],1)
    def test_two_hashseed_processes_exact_complete_diagnostic_json(self):
        base=Path(__file__).resolve().parents[1];command='import json; from oracle_belief_controls_v3_test import equal_score_diagnostics; print(json.dumps(equal_score_diagnostics(),ensure_ascii=False,sort_keys=True,separators=(",",":")))';outputs=[]
        for seed in ('1','987'):
            task_env=dict(os.environ);task_env.update(PYTHONHASHSEED=seed,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(base)+os.pathsep+str(base/'tests'))
            process=subprocess.run([sys.executable,'-c',command],cwd=base,env=task_env,capture_output=True,check=True,timeout=30);outputs.append(process.stdout);REPLAY_EVIDENCE.append({'actual_pythonhashseed':seed,'diagnostic_sha256':hashlib.sha256(process.stdout).hexdigest(),'diagnostic_bytes':len(process.stdout)})
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(json.loads(outputs[0]),equal_score_diagnostics())


class FullConditionalSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inputs=batch();cls.report=v3.check_batch(cls.inputs)
    def test_exact_full_counts_and_zero_errors(self):
        self.assertFalse(self.report['errors'],self.report['errors'][:4]);expected={'records':3456,'source_families':4,'semantic_base_contexts':32,'scalar_variant_base_cases':64,'base_context_query_observations':192,'rotation_level_query_observations':5184,'rendered_query_observations':10368,'conditional_crossing_histograms':128,'scalar_pairs':1728,'field_order_pairs':1728,'opposite_status_same_observation_pairs':864,'opposite_status_paired_records':1728,'distinct_decoded_scalar_tables':8,'distinct_visible_tables':24}
        for key,value in expected.items():self.assertEqual(self.report['counts'][key],value)
    def test_actual_rule_error_counts_and_lower_bound(self):
        measurements=self.report['extra_shortcut_diagnostics']['frozen_rule_measurements']
        for name in ('v1_original_scalar_marker_empty','prospective_marker_and_auxiliary_token_empty'):
            self.assertEqual(measurements['overall/'+name]['errors'],864)
            for split in ('train','dev_challenge'):self.assertEqual(measurements['split:'+split+'/'+name]['errors'],432)
        self.assertEqual(self.report['same_observation_rule_lower_bound']['minimum_errors_using_only_identical_matched_features'],864)
    def test_scalar_outputs_exact_and_status_pair_coverage_disjoint(self):
        for proof in self.report['corpus_controls']['auxiliary_scalar_transposition_invariance']:
            self.assertEqual(v1._json(proof['left']),v1._json(proof['right']));self.assertEqual(proof['left_prefix_supports'],proof['right_prefix_supports'])
        proofs=self.report['corpus_controls']['same_observations_auxiliary_table_counterfactual'];self.assertEqual(len({s for p in proofs for s in p['sample_ids']}),1728)
        self.assertTrue(all(c['satisfied'] for c in self.report['claim_checks']))
    def test_incomplete_variant_and_conditional_rotation_fail(self):
        inputs=[r for r in self.inputs if r['sample_id']!='v3fixture-12-H-transposed-012-forward'];report=v3.check_batch(inputs)
        self.assertTrue(any('27 schedules' in e for e in report['errors']));self.assertTrue(any('missing scalar pair H' in e for e in report['errors']));self.assertTrue(any('required864' in e for e in report['errors']))
    def test_bad_metadata_and_empty_batch_fail_closed(self):
        self.assertTrue(v3.check_batch([])['errors']);self.assertTrue(v3.check_batch(None)['errors']);self.assertTrue(v3.check_batch([{}])['errors'])


if __name__=='__main__':unittest.main()
