"""Archived profile compatibility and fail-closed identity/config checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors.integrity import validate_schema, semantic_fingerprint, validate_v2_group
from goal_conditioned_decision_data.generators.build import generate_group, record_id, semantic_fingerprint as v1_fingerprint
from goal_conditioned_decision_data.generators import profiles
from goal_conditioned_decision_data.pipelines.common import canonical, digest

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gcdl-profile-test-')
        self.root = Path(self.temp.name) / 'programme'
        directory = self.root / 'configs'
        directory.mkdir(parents=True)
        for name in ('generation_profiles.json', 'curriculum-v1.json', 'curriculum-v2.json', 'curriculum.json'):
            shutil.copyfile(ROOT / 'configs' / name, directory / name)

    def tearDown(self):
        self.temp.cleanup()

    def registry(self):
        return json.loads((self.root / 'configs/generation_profiles.json').read_text())

    def provenance(self, name='v1'):
        p = profiles.resolve_profile(name, root=self.root)
        return {key: p[key] for key in profiles.IDENTITY_FIELDS}

    def test_exact_v1_archive_and_default_identity(self):
        self.assertEqual((ROOT / 'configs/curriculum.json').read_bytes(),
                         (ROOT / 'configs/curriculum-v1.json').read_bytes())
        p = profiles.resolve_profile(root=self.root)
        self.assertEqual(p['id'], 'v1')
        self.assertEqual(p['config_hash'], 'd98651723bc9469105627dc55d78f0217eea67b98428020a5b1552452b4ca2f2')
        self.assertEqual(p['config_file_sha256'], 'c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780')
        self.assertEqual(p['config'], json.loads((ROOT / 'configs/curriculum.json').read_text()))
        self.assertEqual(p['schema_version'], 'gcdl-1.0')
        self.assertEqual(p['environment_version'], 'environments-1.0')

    def test_v2_exact_identity_and_no_mutable_resolution_cache(self):
        p = profiles.resolve_profile('v2', root=self.root)
        self.assertEqual(p['generator_version'], 'generator-2.0')
        self.assertEqual(p['config_version'], 'curriculum-2.0')
        self.assertEqual(p['config_hash'], digest(p['config']))
        self.assertEqual(profiles.resolve_record_profile(self.provenance('v2'), root=self.root), p)
        p['config']['split_templates']['dev']['blocks'] = 10000
        self.assertEqual(profiles.resolve_profile('v2', root=self.root)['config']['split_templates']['dev']['blocks'], 4)

    def test_each_record_identity_component_is_exact_without_fallback(self):
        original = self.provenance()
        for key in profiles.IDENTITY_FIELDS:
            for value in ('unknown-version', None, 1, ''):
                p = deepcopy(original); p[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    profiles.resolve_record_profile(p, root=self.root)
        raw_file_hash = profiles.resolve_profile(root=self.root)['config_file_sha256']
        p = deepcopy(original); p['config_hash'] = raw_file_hash
        with self.assertRaises(ValueError): profiles.resolve_record_profile(p, root=self.root)
        for name in ('v3', '../v1', None, True):
            with self.subTest(profile=name), self.assertRaises(ValueError): profiles.resolve_profile(name, root=self.root)

    def test_archive_byte_tamper_is_detected_after_a_successful_resolution(self):
        for name in ('v1', 'v2'):
            path = self.root / 'configs' / ('curriculum-' + name + '.json')
            before = path.read_bytes()
            profiles.resolve_profile(name, root=self.root)
            path.write_bytes(before + b' ')
            with self.subTest(profile=name), self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                profiles.resolve_profile(name, root=self.root)
            path.write_bytes(before)

    def test_registry_tamper_and_missing_archive_fail_closed(self):
        path = self.root / 'configs/generation_profiles.json'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'registry file hash mismatch'):
            profiles.resolve_profile(root=self.root)
        shutil.copyfile(ROOT / 'configs/generation_profiles.json', path)
        (self.root / 'configs/curriculum-v1.json').unlink()
        with self.assertRaises(ValueError): profiles.resolve_profile(root=self.root)

    def test_unselected_archive_damage_does_not_reinterpret_v1(self):
        path = self.root / 'configs/curriculum-v2.json'
        path.write_text('{"version":"tampered"}')
        self.assertEqual(profiles.resolve_profile('v1', root=self.root)['config_version'], 'curriculum-1.0')
        with self.assertRaises(ValueError): profiles.resolve_profile('v2', root=self.root)

    def test_registry_path_version_and_schema_allowlist_survives_repinned_fixture(self):
        # Pin changes require reviewed code. This fixture bypasses only that
        # outer pin to exercise independent structural/path allowlisting.
        for key, value in (('config_file', '../../outside.json'), ('generator_version', 'generator-99'),
                           ('environment_version', 'environments-99'), ('schema_version', 'gcdl-99')):
            data = self.registry(); data['profiles']['v1'][key] = value
            path = self.root / 'configs/generation_profiles.json'
            raw = (canonical(data) + '\n').encode(); path.write_bytes(raw)
            with patch.object(profiles, 'REGISTRY_SHA256', hashlib.sha256(raw).hexdigest()):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity or path'):
                    profiles.resolve_profile(root=self.root)
            shutil.copyfile(ROOT / 'configs/generation_profiles.json', path)

    def test_archive_canonical_identity_checked_even_with_repaired_raw_hash(self):
        path = self.root / 'configs/curriculum-v1.json'
        cfg = json.loads(path.read_text()); cfg['version'] = 'curriculum-99'
        raw = (canonical(cfg) + '\n').encode(); path.write_bytes(raw)
        data = self.registry(); data['profiles']['v1']['config_file_sha256'] = hashlib.sha256(raw).hexdigest()
        registry_raw = (canonical(data) + '\n').encode()
        (self.root / 'configs/generation_profiles.json').write_bytes(registry_raw)
        with patch.object(profiles, 'REGISTRY_SHA256', hashlib.sha256(registry_raw).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'canonical hash/version'):
                profiles.resolve_profile(root=self.root)

    def test_archive_and_configuration_directory_symlinks_are_rejected(self):
        path = self.root / 'configs/curriculum-v1.json'
        outside = Path(self.temp.name) / 'outside.json'; outside.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'): profiles.resolve_profile(root=self.root)
        path.unlink(); path.write_bytes(outside.read_bytes())
        directory = self.root / 'configs'; moved = self.root / 'archived-configs'; directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directory cannot be a symlink'):
            profiles.resolve_profile(root=self.root)

    def test_v1_schema_survives_unrelated_active_default_and_preserves_ids(self):
        rows = generate_group(710271, split='dev')
        before = [canonical(row) for row in rows]
        # A stale/new operator default must not choose the record validator.
        (self.root / 'configs/curriculum.json').write_text('{"version":"unrelated-active-default"}')
        with patch.object(profiles, 'ROOT', self.root):
            for row in rows:
                validate_schema(row)
                self.assertEqual(row['id'], record_id(row))
        self.assertEqual(before, [canonical(row) for row in rows])

    def test_v1_wrong_template_and_unknown_profile_are_rejected(self):
        row = generate_group(710272, split='dev')[0]
        for key, value in (('template_id', 'fabricated'), ('generator_version', 'generator-99'),
                           ('config_hash', '0' * 64), ('environment_version', 'environments-99')):
            changed = deepcopy(row); changed['provenance'][key] = value; changed['id'] = record_id(changed)
            with self.subTest(key=key), self.assertRaises(ValueError): validate_schema(changed)

    def test_v2_bijections_restore_exact_v1_unknown_reference_fingerprint(self):
        from goal_conditioned_decision_data.generators.version2 import generate_vocabulary_pair, restore_source_names
        seed = 710273
        rows, diagnostic = generate_vocabulary_pair(seed, config=profiles.resolve_profile('v2')['config'])
        self.assertEqual(diagnostic['status'], 'VALID')
        old = generate_group(seed, split='dev')[0]
        for row in rows:
            validate_schema(row)
            restored = restore_source_names(row)
            self.assertTrue(any(action.get('entry') == 'absent'
                                for candidate in restored['candidates'] for action in candidate['plan']))
            self.assertEqual(semantic_fingerprint(row), v1_fingerprint(old))
            self.assertEqual(row['id'], record_id(row))
        self.assertEqual(semantic_fingerprint(rows[0]), semantic_fingerprint(rows[1]))
        # Old raw normalization deliberately preserves literal unknown refs;
        # only the V2 restoration branch can equate these renamed surfaces.
        self.assertNotEqual(v1_fingerprint(rows[0]), v1_fingerprint(rows[1]))

    def test_v2_forged_unknown_reference_mapping_rejected_even_with_new_id(self):
        from goal_conditioned_decision_data.generators.version2 import generate_vocabulary_pair
        rows, _ = generate_vocabulary_pair(710274)
        changed = deepcopy(rows[1])
        del changed['metadata']['vocabulary']['maps']['entries']['absent']
        changed['id'] = record_id(changed)
        with self.assertRaises(ValueError): validate_schema(changed)
        with self.assertRaises(ValueError): semantic_fingerprint(changed)

    def test_normalization_size_gate_precedes_original_factorial_normalizer(self):
        row = generate_group(710276, split='dev')[0]
        for task in ({'environment':'blocksworld','initial_state':{'on':{str(i):'table' for i in range(7)}}},
                     {'environment':'ledger','initial_state':{'accounts':{},'entries':{}}},
                     {'environment':'register','initial_state':{'registers':{}}}):
            changed = deepcopy(row); changed['input'] = task
            with patch('goal_conditioned_decision_data.generators.build.semantic_fingerprint') as normalizer:
                with self.subTest(environment=task['environment']), self.assertRaisesRegex(ValueError, 'world-size'):
                    semantic_fingerprint(changed)
                normalizer.assert_not_called()

    def goal_rows(self):
        from goal_conditioned_decision_data.generators.version2 import generate_goal_triplet
        rows, diagnostic = generate_goal_triplet(702, split='dev')
        self.assertEqual(diagnostic['status'], 'VALID')
        return rows

    def test_v2_complete_controlled_groups_and_full_curriculum_requirement(self):
        from goal_conditioned_decision_data.generators.version2 import generate_vocabulary_pair, generate_group as v2_group
        validate_v2_group(self.goal_rows())
        pair, _ = generate_vocabulary_pair(710277)
        validate_v2_group(pair)
        with self.assertRaisesRegex(ValueError, 'full-curriculum'):
            validate_v2_group(self.goal_rows(), require_full_curriculum=True)
        validate_v2_group(v2_group(702, split='dev'), require_full_curriculum=True)
        with self.assertRaisesRegex(ValueError, 'mix generation profiles'):
            validate_v2_group(self.goal_rows()+[generate_group(710278, split='dev')[0]])

    def test_v2_c_metadata_checks_are_profile_specific_and_goal_independent(self):
        row = self.goal_rows()[0]
        for field, value in (('goal_index',True),('goal_index',-1),('goal_index',3),('goal_bundle_id','0'*64)):
            changed=deepcopy(row);changed['metadata'][field]=value;changed['id']=record_id(changed)
            with self.subTest(field=field,value=value), self.assertRaisesRegex(ValueError, 'v2 C'):
                validate_schema(changed)
        changed=deepcopy(row);changed['family']='A';changed['id']=record_id(changed)
        with self.assertRaisesRegex(ValueError, 'family C'): validate_schema(changed)
        changed=deepcopy(row);changed['input']['goal']=deepcopy(changed['input']['initial_state']);changed['id']=record_id(changed)
        with self.assertRaisesRegex(ValueError, 'differ from initial'): validate_schema(changed)
        changed=deepcopy(row);changed['input']['goal']['on'].pop(next(iter(changed['input']['goal']['on'])));changed['id']=record_id(changed)
        with self.assertRaisesRegex(ValueError, 'full-forest'): validate_schema(changed)
        changed=deepcopy(row);changed['input']['candidates'].pop();changed['id']=record_id(changed)
        with self.assertRaisesRegex(ValueError, 'six supplied'): validate_schema(changed)
        old=generate_group(710279,split='dev')[0];old['metadata'].update(goal_index=True,goal_bundle_id='legacy-metadata');old['id']=record_id(old)
        validate_schema(old)  # These V2-specific declarations do not reinterpret V1.
        from goal_conditioned_decision_data.generators.version2 import bundle_identity
        altered=deepcopy(row['input']);altered['goal']=deepcopy(row['input']['initial_state'])
        self.assertEqual(bundle_identity(altered), row['metadata']['goal_bundle_id'])

    def test_v2_group_rejects_incomplete_indices_and_changed_non_goal_input(self):
        rows=self.goal_rows()
        with self.assertRaisesRegex(ValueError, 'exactly indices'): validate_v2_group(rows[:2])
        changed=deepcopy(rows);changed[1]['metadata']['goal_index']=0;changed[1]['id']=record_id(changed[1])
        with self.assertRaisesRegex(ValueError, 'exactly indices'): validate_v2_group(changed)
        changed=deepcopy(rows);changed[1]['input']['instruction']+=' Added presentation.'
        from goal_conditioned_decision_data.generators.version2 import restore_source_names
        # Instruction does not affect namespace inversion; authenticate the
        # changed source text so only the cross-record input gate can reject it.
        old_hash=changed[1]['metadata']['vocabulary']['source_namespace_task_sha256']
        changed[1]['input']['instruction']=rows[1]['input']['instruction']
        restored=restore_source_names(changed[1]);restored['instruction']+=' Added presentation.'
        changed[1]['input']['instruction']+=' Added presentation.'
        changed[1]['metadata']['vocabulary']['source_namespace_task_sha256']=digest(restored)
        changed[1]['id']=record_id(changed[1])
        self.assertNotEqual(old_hash,changed[1]['metadata']['vocabulary']['source_namespace_task_sha256'])
        with self.assertRaisesRegex(ValueError, 'change only'): validate_v2_group(changed)

    def test_v2_group_rejects_bad_best_sets_and_missing_vocabulary_partner(self):
        rows=self.goal_rows()
        changed=deepcopy(rows);changed[1]['targets']['decision']['best_candidate_ids']=changed[0]['targets']['decision']['best_candidate_ids'];changed[1]['id']=record_id(changed[1])
        with self.assertRaisesRegex(ValueError, 'disjoint nonempty'): validate_v2_group(changed)
        changed=deepcopy(rows);changed[1]['targets']['decision']['best_candidate_ids']=[];changed[1]['id']=record_id(changed[1])
        with self.assertRaisesRegex(ValueError, 'disjoint nonempty'): validate_v2_group(changed)
        changed=deepcopy(rows)
        for candidate in changed[0]['targets']['decision']['candidates']:
            candidate['success_probability']={'numerator':0,'denominator':1}
        changed[0]['id']=record_id(changed[0])
        with self.assertRaisesRegex(ValueError, 'feasible successful'): validate_v2_group(changed)
        from goal_conditioned_decision_data.generators.version2 import generate_vocabulary_pair
        pair,_=generate_vocabulary_pair(710280)
        with self.assertRaisesRegex(ValueError, 'both DEV strata'): validate_v2_group(pair[:1])


if __name__ == '__main__':
    unittest.main()
