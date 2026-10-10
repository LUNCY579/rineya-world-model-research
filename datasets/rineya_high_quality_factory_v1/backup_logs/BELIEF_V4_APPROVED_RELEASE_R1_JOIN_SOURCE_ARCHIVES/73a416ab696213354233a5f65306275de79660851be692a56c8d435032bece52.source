"""Source-frozen actual metadata corruption replays; no Git, main or network calls."""
import copy,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
LOGS=Path(__file__).resolve().parent;FACTORY=LOGS.parent
class JoinSafety(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.freeze=json.loads((LOGS/'BELIEF_V4_APPROVED_RELEASE_R1_JOIN_SOURCE_FREEZE.json').read_bytes())
  p=LOGS/'join_belief_v4_approved_release_r1_custody.py'
  assert hashlib.sha256(p.read_bytes()).hexdigest()==cls.freeze['source']['sha256']
  assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==cls.freeze['tests']['sha256']
  for e in cls.freeze['frozen_inputs']:
   q=FACTORY/e['path'];assert not q.is_symlink() and q.stat().st_size==e['size'] and hashlib.sha256(q.read_bytes()).hexdigest()==e['sha256']
  sp=importlib.util.spec_from_file_location('frozen_3158_join_test',p);cls.m=importlib.util.module_from_spec(sp);sp.loader.exec_module(cls.m)
  cls.m.validate_preparation_freeze(cls.freeze);cls.m.load_dependencies()
  m=cls.m
  def read(n):return json.loads((FACTORY/n).read_bytes())
  def lines(n):return [json.loads(s) for s in (FACTORY/n).read_text().splitlines()]
  cls.spec=read(m.SPEC);cls.stage=read(m.STAGE);cls.local=read(m.LOCAL);cls.sl=lines(m.STAGE_LEDGER);cls.scope=read(m.CONFIG)
  cls.ec=read(m.EXECUTION_CONFIG);b=cls.ec['bindings'];cls.full=read(b['full']['path']);cls.gz=read(b['gzip']['path']);cls.fl=lines(b['full_ledger']['path']);cls.gl=lines(b['gzip_ledger']['path']);cls.fullsha=b['full']['sha256'];cls.tree=cls.full['remote_verification']['remote_tree_sha']
 def validate(self,full=None,gz=None,stage=None):return self.m.validate_proofs(self.spec,stage or self.stage,full or self.full,gz or self.gz,self.fullsha,self.tree)
 def test_actual_complete_dependency_freeze_passes(self):self.assertTrue(self.m.validate_preparation_freeze(self.freeze))
 def test_each_mandatory_preparation_path_omission_rejected(self):
  required=set(self.m.REQUIRED_PREPARATION_HASHES)|{self.freeze['source']['path'],self.freeze['tests']['path']}
  for n in sorted(required):
   with self.subTest(path=n):
    bad=copy.deepcopy(self.freeze);bad['frozen_inputs']=[e for e in bad['frozen_inputs'] if e['path']!=n]
    with self.assertRaises(ValueError):self.m.validate_preparation_freeze(bad)
 def test_scope_omission_substitution_and_declared_source_test_tamper_rejected(self):
  for key in ('source','tests','scope_config'):
   bad=copy.deepcopy(self.freeze);bad.pop(key)
   with self.assertRaises(ValueError):self.m.validate_preparation_freeze(bad)
   bad=copy.deepcopy(self.freeze);bad[key]['sha256']='0'*64
   with self.assertRaises(ValueError):self.m.validate_preparation_freeze(bad)
 def test_freeze_empty_duplicate_invalid_typed_hash_path_rows_rejected(self):
  for change in (lambda f:f.update(frozen_inputs=[]),lambda f:f['frozen_inputs'].append(f['frozen_inputs'][0]),lambda f:f['frozen_inputs'][0].update(size=True),lambda f:f['frozen_inputs'][0].update(sha256='F'*64),lambda f:f['frozen_inputs'][0].update(path='../outside'),lambda f:f.update(author_execution='/other'),lambda f:f.update(frozen_before_first_import_test_execution=False)):
   bad=copy.deepcopy(self.freeze);change(bad)
   with self.assertRaises(ValueError):self.m.validate_preparation_freeze(bad)
 def test_literal_alias_and_unlisted_path_rejected(self):
  for n in ('/outside','backup_logs/../outside','backup_logs//outside','backup_logs\\outside','backup_logs/./outside'):
   with self.assertRaises(ValueError):self.m.literal_path(n)
 def test_actual_stage_and_local_ledger_crossjoin_passes(self):
  r=self.m.stage_coverage(self.spec,self.stage,self.sl,self.local);self.assertEqual(len(r),3158);self.assertEqual(len({e['git_blob'] for e in r.values()}),2474)
 def test_stage_wrong_role_parent_commit_spec_or_unqualified_whitespace_rejected(self):
  for k,v in (('runner_execution','/other'),('expected_parent_sha','0'*40),('commit_sha','0'*40),('spec_sha256','0'*64),('actual_exit_code',0),('warning_count',5),('restricted_changed_path_count',481),('strict_diff_check','PASS'),('all_source_target_index_sha256_and_sha1_verified',False)):
   bad=copy.deepcopy(self.stage);bad[k]=v
   with self.subTest(key=k),self.assertRaises(ValueError):self.m.validate_stage(self.spec,bad)
 def test_stage_missing_source_duplicate_index_paths_and_changed_hash_rejected(self):
  for mutate in (lambda a:a.remove(next(e for e in a if e['phase']=='source_target_verified')),lambda a:next(e for e in a if e['phase']=='source_target_verified').update(sha256='0'*64),lambda a:next(e for e in a if e['phase']=='actual_index_blob_verified')['paths'].append(next(e for e in a if e['phase']=='actual_index_blob_verified')['paths'][0])):
   bad=copy.deepcopy(self.sl);mutate(bad)
   with self.assertRaises(ValueError):self.m.stage_coverage(self.spec,self.stage,bad,self.local)
 def test_distinct_local_preserved_status_and_remote_none_required(self):
  for k,v in (('status','PUSHED_REMOTE_BYTES_VERIFICATION_PENDING'),('remote_commit_sha',self.m.COMMIT),('actual_exit_code',0)):
   bad=copy.deepcopy(self.local);bad[k]=v
   with self.assertRaises(ValueError):self.m.stage_coverage(self.spec,self.stage,self.sl,bad)
 def test_actual_full_and_actual_gzip_evidence_passes(self):self.assertEqual(len(self.validate()),3158)
 def test_incomplete_wrong_source_roles_counts_or_evolution_rejected(self):
  for k,v in (('status','PENDING'),('files',3157),('local_git_verified_unique_blobs',2473),('current_source_matching_files',3157),('runner_execution','/other'),('source_freeze_sha256','0'*64),('verification_source_sha256','0'*64),('current_source_evolution',[{'path':'changed'}]),('current_source_evolution_at_finish',[{'path':'changed'}]),('local_raw_representation_decode_iterations',31),('local_raw_representation_decoded_bytes',1724728576)):
   bad=copy.deepcopy(self.full);bad[k]=v
   with self.subTest(key=k),self.assertRaises(ValueError):self.validate(full=bad)
 def test_complete_four_reports_each_missing_or_changed_rejected(self):
  for key in ('capped_source_secret_scan_before','capped_source_secret_scan_after','actual_local_git_metadata_exception_scan','actual_remote_metadata_exception_scan'):
   bad=copy.deepcopy(self.full);bad.pop(key)
   with self.subTest(key=key),self.assertRaises(ValueError):self.validate(full=bad)
   bad=copy.deepcopy(self.full);bad[key]['files']=[]
   with self.subTest(key=key),self.assertRaises(ValueError):self.validate(full=bad)
 def test_stream_context_or_git_id_substitution_rejected(self):
  for key in ('actual_local_git_metadata_exception_scan','actual_remote_metadata_exception_scan'):
   for field,v in (('stream_context','wrong'),('files',[])):
    bad=copy.deepcopy(self.full);bad[key][field]=v
    with self.assertRaises(ValueError):self.validate(full=bad)
 def test_actual_private_admin_and_before_after_six_guards_required(self):
  for key in ('checkpoint_custody_guards_before','checkpoint_custody_guards_after'):
   bad=copy.deepcopy(self.full);bad[key]['original_six_git_byte_guards']=False
   with self.assertRaises(ValueError):self.validate(full=bad)
  bad=copy.deepcopy(self.full);bad['private_repository_evidence']['viewerPermission']='WRITE'
  with self.assertRaises(ValueError):self.validate(full=bad)
 def test_stage_local_ledger_evidence_binding_required(self):
  for i in range(3):
   bad=copy.deepcopy(self.full);bad['prior_evidence'][i]['sha256']='0'*64
   with self.assertRaises(ValueError):self.validate(full=bad)
 def test_full_remote_no_cache_raw_tree_provenance_path_hash_rejected(self):
  for field,v in (('http_accept','cached'),('cache_policy','inherited'),('remote_tree_sha','0'*40)):
   bad=copy.deepcopy(self.full);bad['remote_verification'][field]=v
   with self.assertRaises(ValueError):self.validate(full=bad)
  for mutate in (lambda b:b[0].update(verification_source='cache'),lambda b:b[0].update(sha256='0'*64),lambda b:b[0].update(paths=[]),lambda b:b[0]['paths'].append('outside'),lambda b:b[0]['paths'].append(b[0]['paths'][0])):
   bad=copy.deepcopy(self.full);mutate(bad['remote_verification']['blob_receipts'])
   with self.assertRaises(ValueError):self.validate(full=bad)
 def test_gzip_exact31_not32_counts_and_all32_manifest_refs_rejected(self):
  for k,v in (('actual_remote_gzip_fetches',32),('oversized_raw_artifacts',30),('unique_manifest_path_count',32),('frozen_representation_reference_count',31),('unique_raw_gzip_pair_count',32),('raw_bytes_exactly_recovered',1766951239),('local_fullproof_decode_iterations',31)):
   bad=copy.deepcopy(self.gz);bad[k]=v
   with self.subTest(key=k),self.assertRaises(ValueError):self.validate(gz=bad)
  bad=copy.deepcopy(self.gz);bad['full_frozen_lossless_manifest_reference_coverage'].pop()
  with self.assertRaises(ValueError):self.validate(gz=bad)
 def test_gzip_pending_wrong_source_frozen_full_role_and_bound_memory_rejected(self):
  for k,v in (('status','PENDING'),('source_sha256','0'*64),('dependency_sha256','0'*64),('source_freeze_sha256','0'*64),('verified_inventory_receipt_sha256','0'*64),('stage_receipt_sha256','0'*64),('actual_runner_execution','/root'),('grants_data_approval',True),('stream_chunk_bytes',131072),('bounded_spool_memory_bytes',16777216)):
   bad=copy.deepcopy(self.gz);bad[k]=v
   with self.assertRaises(ValueError):self.validate(gz=bad)
 def test_gzip_actual_fetch_compare_wrong_sha_gitid_and_duplicate_pairs_rejected(self):
  for k,v in (('actual_remote_fetch_this_run',False),('decoded_byte_for_byte_equal_to_preserved_raw',False),('remote_git_blob','0'*40),('remote_gzip_sha256','0'*64),('raw_sha256','0'*64)):
   bad=copy.deepcopy(self.gz);bad['actual_remote_gzip_proofs'][0][k]=v
   with self.assertRaises(ValueError):self.validate(gz=bad)
  bad=copy.deepcopy(self.gz);bad['actual_remote_gzip_proofs'][1]=bad['actual_remote_gzip_proofs'][0]
  with self.assertRaises(ValueError):self.validate(gz=bad)
 def test_actual_stage_full_and_current_gitsha256_crossjoin_pass_and_tamper_reject(self):
  a=self.m.stage_coverage(self.spec,self.stage,self.sl,self.local);b=self.validate();self.m.crossjoin(a,b)
  bad=copy.deepcopy(a);bad[next(iter(bad))]['git_blob']='0'*40
  with self.assertRaises(ValueError):self.m.crossjoin(b,bad)
 def test_actual_full_and_gzip_ledgers_pass(self):self.m.validate_ledgers(self.full,self.gz,self.fl,self.gl)
 def test_ledger_missing_terminal_crosswired_body_and_local_missing_rejected(self):
  with self.assertRaises(ValueError):self.m.validate_ledgers(self.full,self.gz,self.fl[:-1],self.gl)
  with self.assertRaises(ValueError):self.m.validate_ledgers(self.full,self.gz,self.fl,self.gl[:-1])
  bad=copy.deepcopy(self.fl);next(e for e in bad if e['phase']=='actual_remote_blob_verified')['sha256']='0'*64
  with self.assertRaises(ValueError):self.m.validate_ledgers(self.full,self.gz,bad,self.gl)
  bad=[e for e in self.fl if e['phase']!='local_git_blob_verified']
  with self.assertRaises(ValueError):self.m.validate_ledgers(self.full,self.gz,bad,self.gl)
  bad=copy.deepcopy(self.gl);next(e for e in bad if e['phase']=='ACTUAL_REMOTE_GZIP_BYTE_PROOF')['remote_git_blob']='0'*40
  with self.assertRaises(ValueError):self.m.validate_ledgers(self.full,self.gz,self.fl,bad)
 def test_frozen_scope26_atcut29_later_completed_local_passes(self):self.assertTrue(self.m.validate_scope(self.scope,self.spec))
 def test_scope_counts_types_historical_deferral_or_checkpoint_insertion_rejected(self):
  for k,v in (('postcut_completed_adjudication_files',28),('files',True),('atcut_observed_deferred_paths',0),('grants_data_approval',True)):
   bad=copy.deepcopy(self.scope);bad[k]=v
   with self.assertRaises(ValueError):self.m.validate_scope(bad,self.spec)
  bad=copy.deepcopy(self.scope);bad['historical_observed_pending_atcut'].pop()
  with self.assertRaises(ValueError):self.m.validate_scope(bad,self.spec)
  bad=copy.deepcopy(self.scope);bad['completed_postcut_adjudication_artifacts'][0]['path']=self.spec['files'][0]['path']
  with self.assertRaises(ValueError):self.m.validate_scope(bad,self.spec)
 def test_actual_execution_config_four_exact_bindings_passes(self):self.assertEqual(len(self.m.validate_execution_config(self.ec)),4)
 def test_execution_config_missing_substituted_invalid_size_hash_role_rejected(self):
  for mutate in (lambda d:d['bindings'].pop('gzip_ledger'),lambda d:d['bindings']['full'].update(size=True),lambda d:d['bindings']['full'].update(sha256='no'),lambda d:d.update(actual_completed_proofs_bound=False),lambda d:d.update(actual_runner_execution='/root'),lambda d:d['bindings']['gzip'].update(path=d['bindings']['full']['path'])):
   bad=copy.deepcopy(self.ec);mutate(bad)
   with self.assertRaises(ValueError):self.m.validate_execution_config(bad)
if __name__=='__main__':unittest.main(verbosity=2)
