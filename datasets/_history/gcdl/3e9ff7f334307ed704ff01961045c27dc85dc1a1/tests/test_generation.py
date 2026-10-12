"""Meaningful invariance, split and training-view regression checks."""
import unittest
import json
from copy import deepcopy
from goal_conditioned_decision_data.generators.build import (
    generate_group, record_id, semantic_fingerprint, protected_template_signature)
from goal_conditioned_decision_data.generators.improvement import run_position_experiment
from goal_conditioned_decision_data.decision_labels.views import training_views


class GenerationTests(unittest.TestCase):
    def test_seed_reproduction_and_family_coverage(self):
        one = generate_group(77)
        self.assertEqual(one, generate_group(77))
        self.assertEqual({r['family'] for r in one}, set('ABCDEFGH'))
        self.assertTrue(all(r['id'] == record_id(r) for r in one))

    def test_semantic_fingerprint_matches_persisted_json(self):
        # Reversed/prefix candidate plans can share action dictionaries in
        # Python; JSON records cannot. Normalization must be alias-invariant.
        for record in generate_group(790):
            persisted = json.loads(json.dumps(record))
            self.assertEqual(semantic_fingerprint(record), semantic_fingerprint(persisted))

    def test_template_protection_uses_actual_sizes(self):
        signatures = {}
        for split in ('train', 'dev', 'sealed', 'ood'):
            signatures[split] = {protected_template_signature(r['input']) for r in generate_group(78, split=split)}
        for left in signatures:
            for right in signatures:
                if left != right:
                    self.assertFalse(signatures[left] & signatures[right])

    def test_surface_variants_preserve_task_semantics(self):
        records = generate_group(79)
        base = records[0]
        for record in records:
            perturbation = record['metadata']['perturbation']
            if perturbation.get('label_relation') == 'invariant':
                self.assertEqual(semantic_fingerprint(base), semantic_fingerprint(record))
                self.assertEqual(set(base['targets']['decision']['best_candidate_ids']),
                                 set(record['targets']['decision']['best_candidate_ids']))

    def test_training_views_have_no_oracle_input_fields(self):
        record = generate_group(80)[0]
        views = training_views(record)
        self.assertEqual(len(views['pointwise']), len(record['input']['candidates']))
        self.assertEqual(len(views['pairwise']), 6)
        for family in views.values():
            for view in family:
                self.assertFalse({'targets', 'oracle', 'verification', 'metadata'} & set(view['input']))
                self.assertIn('target', view)
        self.assertEqual(views['listwise'][0]['input'], record['input'])

    def test_opaque_id_rename_does_not_change_task(self):
        record = generate_group(800)[0]
        for candidate in record['input']['candidates']:
            self.assertRegex(candidate['id'], r'^c_[0-9a-f]{24}$')
        renamed = deepcopy(record)
        for index, candidate in enumerate(renamed['input']['candidates']):
            candidate['id'] = 'opaque_' + str(index)
        self.assertEqual(semantic_fingerprint(record), semantic_fingerprint(renamed))

    def test_quarantined_export_rejected_and_all_infeasible_abstains(self):
        records = generate_group(801)
        quarantined = deepcopy(records[0])
        quarantined['quality_gate'] = {'status': 'quarantined'}
        with self.assertRaises(ValueError):
            training_views(quarantined)
        infeasible = next(r for r in records if 'all_candidates_infeasible' in r['metadata']['tags'])
        labels = infeasible['targets']['decision']
        self.assertEqual(labels['best_candidate_ids'], [])
        self.assertEqual(labels['uncertainty']['abstention'], 'no_feasible_candidate')

    def test_missing_information_uses_exact_probabilities(self):
        records = generate_group(81)
        uncertain = next(r for r in records if 'missing_information' in r['metadata']['tags'])
        credit = next(c for c in uncertain['targets']['decision']['candidates'] if uncertain['metadata']['candidate_roles'][c['id']] == 'settle')
        self.assertEqual(credit['success_probability'], {'numerator': 1, 'denominator': 2})
        weighted = next(r for r in records if 'uniform_three_world_probability' in r['metadata']['tags'])
        credit = next(c for c in weighted['targets']['decision']['candidates'] if weighted['metadata']['candidate_roles'][c['id']] == 'settle')
        self.assertEqual(credit['success_probability'], {'numerator': 1, 'denominator': 3})

    def test_search_uses_matched_universe_and_bounded_proof(self):
        records = generate_group(82)
        for record in records:
            if 'search' in record['targets']:
                search = record['targets']['search']
                self.assertEqual(search['matched_candidate_ids'], [c['id'] for c in record['input']['candidates']])
                self.assertGreaterEqual(search['additional_transition_node_evaluations'], 0)
        omitted = next(r for r in records if 'proposer_omitted_solution' in r['metadata']['tags'])
        proof = omitted['targets']['bounded_proof']
        self.assertTrue(proof['solution_exists_in_bound'])
        self.assertEqual(proof['enumerated_plan_count'], 31)
        self.assertFalse(proof['global_unbounded_optimality_claim'])
        self.assertEqual(omitted['targets']['decision']['uncertainty']['label'], 'impossible_in_candidate_set')

    def test_controlled_experiment_preserves_labels_and_reduces_bias(self):
        report = run_position_experiment(groups=8)
        self.assertEqual(report['split'], 'dev')
        self.assertEqual(report['semantic_changes'], 0)
        self.assertEqual(report['oracle_label_changes'], 0)
        self.assertEqual(report['before']['maximum_position_fraction'], 1)
        self.assertEqual(report['after']['maximum_position_fraction'], 0.25)
        self.assertTrue(report['independent_review_required'])


if __name__ == '__main__':
    unittest.main()
