"""Prepared isolated corruption tests. No Git/network/main or producer semantic reads."""
from __future__ import annotations
import copy, gzip, hashlib, io, json, tempfile, unittest
from pathlib import Path
import next653_common as c
import next653_inventory as inventory
import next653_stage as stage
import next653_recovery as recovery
import next653_lossless as lossless
import next653_join as join
import next653_admission as admission
import next653_archive_mapping as archive_mapping
import next653_runtime_guard as runtime_guard

TEST_CONFIG=None

def fixture(path='fixtures/plain.txt',payload=b'ordinary archival fixture\n'):
    return {'path':path,**c.hash_stream(io.BytesIO(payload),len(payload))}

def simple_spec():
    a=fixture('fixtures/a.txt');b=fixture('fixtures/b.txt')
    return {'schema':'next-checkpoint-653-exact-inventory-v1','files':[a,b],'raw_size_exclusions':[]}

def lossless_spec():
    r=fixture('fixtures/oversized.json');r['size']=c.CAP+1
    g=fixture('fixtures/raw.gz',b'compressed fixture');m=fixture('fixtures/raw.lossless.json',b'manifest fixture')
    n=fixture('fixtures/other.lossless.json',b'other manifest')
    r['lossless_representations']=[{'gzip':c.projection(g),'manifest':c.projection(m)}, {'gzip':c.projection(g),'manifest':c.projection(n)}]
    return {'schema':'next-checkpoint-653-exact-inventory-v1','files':[g,m,n],'raw_size_exclusions':[r]}

class PathAndByteTests(unittest.TestCase):
    def test_path_aliases_traversal_generated_and_sensitive_rejected(self):
        for p in ('../outside','./alias','a//b','/absolute','a/../b','a\\b','a\x00b','a\nb','a\rb','.git/config','.env','a.key','a.pem','__pycache__/a.py','a.pyc'):
            with self.subTest(path=p),self.assertRaises(ValueError):c.name(p)
    def test_symlink_is_not_admitted(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);(root/'target').write_bytes(b'ordinary');(root/'alias').symlink_to(root/'target')
            with self.assertRaises(ValueError):c.safe(root,'alias')
    def test_typed_hash_and_size_guards(self):
        for x in (True,False,-1,'1',None):
            with self.subTest(value=x),self.assertRaises(ValueError):c.number(x)
        for x in ('a'*63,'A'*64,'g'*64,True,None):
            with self.subTest(value=x),self.assertRaises(ValueError):c.sha(x)
    def test_stream_short_overlong_and_sha1(self):
        b=b'ordinary fixture';r=c.hash_stream(io.BytesIO(b),len(b))
        self.assertEqual(r['git_blob'],hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest())
        for n in (len(b)-1,len(b)+1):
            with self.subTest(size=n),self.assertRaises(ValueError):c.hash_stream(io.BytesIO(b),n)
    def test_changed_source_tuple_rejected(self):
        r=fixture();bad=dict(r,sha256='0'*64)
        with self.assertRaises(ValueError):c.equal(bad,r)
        with self.assertRaises(ValueError):c.equal(dict(r,git_blob='0'*40),r)
    def test_copy_real_bytes_and_target_tamper_repair(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.HERE) as b:
            factory,worktree=Path(a),Path(b);payload=b'ordinary copy fixture\n';row=fixture('probe.txt',payload)
            (factory/'probe.txt').write_bytes(payload);session=c.SecretSession()
            self.assertEqual(stage.copy_file(row,session,factory,worktree),'COPIED_AND_REVERIFIED')
            target=worktree/c.PREFIX/'probe.txt';self.assertEqual(target.read_bytes(),payload)
            self.assertEqual(stage.copy_file(row,session,factory,worktree),'UNCHANGED_TARGET_REVERIFIED')
            target.write_bytes(b'tampered');stage.copy_file(row,session,factory,worktree)
            self.assertEqual(target.read_bytes(),payload)
    def test_changed_source_cannot_overwrite_frozen_target(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.HERE) as b:
            factory,worktree=Path(a),Path(b);row=fixture('probe.txt');(factory/'probe.txt').write_bytes(b'changed')
            target=worktree/c.PREFIX/'probe.txt';target.parent.mkdir(parents=True);target.write_bytes(b'preserved target')
            with self.assertRaises(ValueError):stage.copy_file(row,c.SecretSession(),factory,worktree)
            self.assertEqual(target.read_bytes(),b'preserved target')

class ExactMetadataTests(unittest.TestCase):
    def test_all_four_actual_original_paths_and_overlap_chunks(self):
        h=c.library('scanner');patterns=[c.re.compile(p,f) for p,f in h.EXPECTED_PATTERN_SPECS]
        # Opaque bytes are read from the four original authorized paths only.
        # No new archive, fixture copy, match text or solver content is emitted.
        for row in h.EXPECTED_FILES:
            p=c.safe(c.FACTORY,row['path']);opaque=p.read_bytes()
            for size in (1,2,31,4096,65536):
                with self.subTest(path=row['path'],chunk=size):
                    proof,matches=h.scan_stream(io.BytesIO(opaque),row['size'],patterns,row,size)
                    self.assertEqual(proof['sha256'],row['sha256']);self.assertEqual(matches,row['matches'])
    def test_missing_fourth_observation_rejected(self):
        session=c.SecretSession()
        for row in list(session.expected.values())[:3]:session.file(c.safe(c.FACTORY,row['path']),row['path'],'source_before')
        with self.assertRaises(ValueError):session.complete('source_before')
    def test_context_wrong_copy_alias_and_changed_actual_bytes_rejected(self):
        session=c.SecretSession();row=next(iter(session.expected.values()));opaque=c.safe(c.FACTORY,row['path']).read_bytes()
        with self.assertRaises(ValueError):session.stream(io.BytesIO(opaque),row['size'],['fixtures/copied-metadata.json'],'actual_remote')
        with self.assertRaises(ValueError):session.stream(io.BytesIO(opaque),row['size'],[row['path'],'fixtures/copied-metadata.json'],'actual_remote')
        with self.assertRaises(ValueError):session.stream(io.BytesIO(opaque),row['size'],[row['path']],'inherited_cache')
        altered=bytearray(opaque);altered[-1]^=1
        with self.assertRaises(ValueError):session.stream(io.BytesIO(altered),row['size'],[row['path']],'actual_remote')
    def test_metadata_extra_token_and_object_tamper_rejected(self):
        session=c.SecretSession();row=next(iter(session.expected.values()));opaque=c.safe(c.FACTORY,row['path']).read_bytes()
        extra=opaque+b'\n'+b'sk-'+b'A'*31
        with self.assertRaises(ValueError):session.stream(io.BytesIO(extra),len(extra),[row['path']],'actual_remote')
        with self.assertRaises(ValueError):session.stream(io.BytesIO(opaque),len(opaque),[row['path']],'actual_remote','0'*40)
    def test_standalone_secrets_across_chunk_boundary_rejected(self):
        h=c.library('scanner');patterns=[c.re.compile(p,f) for p,f in h.EXPECTED_PATTERN_SPECS]
        cases=[b'sk-'+b'A'*31,b'ghp_'+b'B'*21,b'AKIA'+b'C'*16,b'-----BEGIN '+b'PRIVATE KEY-----',b'password='+b'D'*18]
        for payload in cases:
            body=b'ordinary prefix\n'+payload+b'\n'
            for chunk in (1,2,13,65536):
                with self.subTest(case=hashlib.sha256(payload).hexdigest(),chunk=chunk),self.assertRaises(ValueError):h.scan_stream(io.BytesIO(body),len(body),patterns,None,chunk)
    def test_normal_prose_and_inline_assignment_negative_control(self):
        h=c.library('scanner');patterns=[c.re.compile(p,f) for p,f in h.EXPECTED_PATTERN_SPECS]
        body=b'ordinary prose containing an inline password='+b'E'*18+b'\n'
        h.scan_stream(io.BytesIO(body),len(body),patterns,None,65536)
    def test_pattern_index_flags_or_span_cannot_be_widened(self):
        h=c.library('scanner');patterns=[c.re.compile(p,f) for p,f in h.EXPECTED_PATTERN_SPECS];row=copy.deepcopy(h.EXPECTED_FILES[0])
        opaque=c.safe(c.FACTORY,row['path']).read_bytes()
        for key,value in (('start',row['matches'][0]['start']+1),('pattern_index',0),('match_sha256','0'*64)):
            changed=copy.deepcopy(row);changed['matches'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):h.scan_stream(io.BytesIO(opaque),len(opaque),patterns,changed)
        with self.assertRaises(ValueError):h.scan_stream(io.BytesIO(opaque),len(opaque),patterns[:-1],row)
    def test_local_and_remote_contexts_independent(self):
        s=c.SecretSession()
        for row in s.expected.values():
            p=c.safe(c.FACTORY,row['path']);s.file(p,row['path'],'source_before')
        s.complete('source_before')
        with self.assertRaises(ValueError):s.complete('actual_remote')

class ScopeAndProvenanceTests(unittest.TestCase):
    def test_groups_dynamic_dedup_and_paths_complete(self):
        files,_=c.all_entries(simple_spec());group=c.groups(files)
        self.assertEqual(len(group),1);self.assertEqual(next(iter(group.values()))['paths'],['fixtures/a.txt','fixtures/b.txt'])
    def test_same_git_id_wrong_size_or_sha_rejected(self):
        spec=simple_spec();spec['files'][1]['sha256']='0'*64
        with self.assertRaises(ValueError):c.groups(c.all_entries(spec)[0])
    def test_duplicate_path_and_capped_raw_overlap_rejected(self):
        spec=simple_spec();spec['files'].append(dict(spec['files'][0]))
        with self.assertRaises(ValueError):c.all_entries(spec)
        spec=simple_spec();spec['raw_size_exclusions']=[dict(spec['files'][0],size=c.CAP+1)]
        with self.assertRaises(ValueError):c.all_entries(spec)
    def test_previous_capped_and_raw_paths_cannot_disappear_or_defer(self):
        prior=lossless_spec();files,raw=c.all_entries(prior);allrows={**files,**raw}
        c.preserve_prior(prior,allrows)
        for p in allrows:
            changed=dict(allrows);changed.pop(p)
            with self.subTest(path=p),self.assertRaises(ValueError):c.preserve_prior(prior,changed)
            with self.subTest(deferred=p),self.assertRaises(ValueError):c.preserve_prior(prior,allrows,[p])
    def test_historical_release_byte_mutation_rejected(self):
        prior=simple_spec();prior['files'][0]['path']='data/RBM-001/releases/v0002/manifest.json';files,raw=c.all_entries(prior);changed=copy.deepcopy(files)
        changed[next(iter(changed))]['sha256']='0'*64
        with self.assertRaises(ValueError):c.preserve_prior(prior,changed)
    def test_legitimate_current_core_change_requires_exact_new_binding(self):
        prior=simple_spec();prior['files'][0]['path']='factory/pipeline.py';files,_=c.all_entries(prior);current=copy.deepcopy(files);current['factory/pipeline.py']['sha256']='0'*64
        allowed=[{'prior':c.projection(files['factory/pipeline.py']),'current':c.projection(current['factory/pipeline.py']),'preserved_parent':c.PARENT}]
        inventory.validate_evolution(prior,current,allowed)
        for bad in ([],[dict(allowed[0],preserved_parent='0'*40)]):
            with self.assertRaises(ValueError):inventory.validate_evolution(prior,current,bad)
    def test_nonexistent_pending_and_prior_path_deferral_rejected(self):
        row={'path':'fixtures/new.py','size':1,'sha256':'0'*64,'actor_execution':'/root/actor','status_at_cut':'RUNNING'}
        with self.assertRaises(ValueError):inventory.pending_cut({},[row],set())
        with self.assertRaises(ValueError):inventory.pending_cut({row['path']:None},[row],{row['path']})
    def test_actual_object_coverage_rejects_cached_missing_duplicate_crosswire(self):
        files,_=c.all_entries(simple_spec());expected=c.groups(files);row=dict(next(iter(expected.values())),actual_remote_fetch_this_run=True)
        c.validate_actual_groups([row],expected,'actual_remote_fetch_this_run')
        cases=[[],[row,row],[dict(row,actual_remote_fetch_this_run=False)],[dict(row,paths=['fixtures/a.txt'])],[dict(row,size=True)],[dict(row,sha256='0'*64)]]
        for rows in cases:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):c.validate_actual_groups(rows,expected,'actual_remote_fetch_this_run')
    def test_strict_delta_does_not_reuse_old_six_warning_exception(self):
        c.assert_strict_diff({'returncode':0,'stdout':b'','stderr':b''})
        for row in ({'returncode':2,'stdout':b'old six warnings','stderr':b''},{'returncode':0,'stdout':b'one warning','stderr':b''},{'returncode':0,'stdout':b'','stderr':b'warning'},{'returncode':False,'stdout':b'','stderr':b''},{'returncode':0,'stdout':'','stderr':b''}):
            with self.assertRaises(ValueError):c.assert_strict_diff(row)
    def test_whole_tree_outside_prefix_and_index_mode_preservation(self):
        m=c.library('manager');files,_=c.all_entries(simple_spec());parent={'tools/tool.py':('100755','1'*40),c.PREFIX+'/fixtures/a.txt':('100644','2'*40)}
        tree=m.expected_whole_tree(parent,files)
        self.assertEqual(tree['tools/tool.py'],parent['tools/tool.py']);self.assertEqual(tree[c.PREFIX+'/fixtures/a.txt'],('100644',files['fixtures/a.txt']['git_blob']))
    def test_private_git_baseline_missing_conflict_or_hash_changed_rejected(self):
        m=c.library('manager');rows=c.fixed_json('custody')['current_actual_custody']['six_protected_git_baselines'];m.normalize_baselines(c.ORIGINAL,{'protected_git_byte_baselines':rows})
        for mutation in ('missing','hash','exists','type','path','conflict'):
            changed=copy.deepcopy(rows)
            if mutation=='missing':changed.pop()
            elif mutation=='hash':changed[0]['sha256']='0'*64
            elif mutation=='exists':changed[0]['expected_exists']=False
            elif mutation=='type':changed[0]['expected_exists']=1
            elif mutation=='path':changed[0]['path']+='/alias'
            evidence={'protected_git_byte_baselines':changed}
            if mutation=='conflict':evidence['original']={'protected_git_byte_baselines_match':changed[:-1]}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.normalize_baselines(c.ORIGINAL,evidence)
    def test_remote_tree_truncation_duplicate_and_other_object_rejected(self):
        value={'truncated':False,'tree':[{'path':'tools/a','mode':'100644','type':'blob','sha':'1'*40}]};self.assertEqual(len(recovery.remote_tree(value)),1)
        for bad in (dict(value,truncated=True),dict(value,tree=value['tree']*2),{'truncated':False,'tree':[dict(value['tree'][0],type='commit')]}):
            with self.assertRaises(ValueError):recovery.remote_tree(bad)
    def test_commit_parent_or_tree_crosswire_rejected(self):
        value={'sha':'1'*40,'tree':{'sha':'2'*40},'parents':[{'sha':c.PARENT}]};recovery.validate_commit_metadata(value,'1'*40,'2'*40)
        for bad in (dict(value,parents=[]),dict(value,parents=[{'sha':'0'*40}]),dict(value,tree={'sha':'3'*40})):
            with self.assertRaises(ValueError):recovery.validate_commit_metadata(bad,'1'*40,'2'*40)

class LosslessTests(unittest.TestCase):
    def test_dynamic_pairs_repeated_manifest_refs_account_separately(self):
        spec=lossless_spec();vector=c.scope(spec);pairs,refs=lossless.representation_plan(spec)
        self.assertEqual(len(pairs),1);self.assertEqual(len(refs),2);self.assertEqual(vector['local_representation_iterations'],2)
        self.assertEqual(vector['global_manifest_paths'],2);self.assertEqual(vector['remote_decoded_bytes'],c.CAP+1)
        self.assertEqual(vector['local_decoded_bytes'],2*(c.CAP+1))
    def test_missing_gzip_manifest_and_package_inside_release_rejected(self):
        for mutate in ('gzip','manifest','release','duplicate'):
            spec=lossless_spec()
            if mutate=='gzip':spec['files'].pop(0)
            elif mutate=='manifest':spec['files'].pop(1)
            elif mutate=='release':
                spec['files'][0]['path']='data/X/releases/v1/raw.gz'
                for rep in spec['raw_size_exclusions'][0]['lossless_representations']:rep['gzip']['path']=spec['files'][0]['path']
            else:spec['raw_size_exclusions'][0]['lossless_representations'].append(copy.deepcopy(spec['raw_size_exclusions'][0]['lossless_representations'][0]))
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):c.scope(spec)
    def test_conflicting_alias_and_unsupported_archive_schema_rejected(self):
        good={'raw_path':'fixtures/raw.json','raw_bytes':4,'raw_sha256':'1'*64,'gzip_path':'fixtures/raw.gz','gzip_bytes':5,'gzip_sha256':'2'*64}
        inventory.normalize_record(good)
        for bad in (dict(good,raw_length=5),{'raw_path':'fixtures/raw.json','archive_path':'fixtures/raw.gz'}):
            with self.assertRaises(ValueError):inventory.normalize_record(bad)
    def test_local_full_byte_roundtrip_and_crc_failure(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);raw=root/'raw';packed=root/'raw.gz';raw.write_bytes(b'ordinary\n'*9000);packed.write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
            self.assertEqual(c.compare_gzip(raw,packed),raw.stat().st_size)
            raw.write_bytes(b'changed')
            with self.assertRaises(ValueError):c.compare_gzip(raw,packed)
            raw.write_bytes(b'ordinary\n'*9000);b=bytearray(packed.read_bytes());b[-8]^=1;packed.write_bytes(b)
            with self.assertRaises((ValueError,OSError,EOFError)):c.compare_gzip(raw,packed)
    def test_bounded_remote_consumer_actual_byte_and_blob_tamper(self):
        g=c.library('gzip')
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);rawpath=root/'raw';payload=b'ordinary payload\n'*8000;rawpath.write_bytes(payload)
            compressed=gzip.compress(payload,mtime=0);raw=fixture('fixtures/raw',payload);packed=fixture('fixtures/raw.gz',compressed)
            result=g.consume_remote(io.BytesIO(compressed),rawpath,raw,packed,{'sha':packed['git_blob']},root)
            self.assertTrue(result['actual_remote_fetch_this_run']);self.assertEqual(result['raw_bytes'],len(payload))
            for data,blob in ((compressed[:-1],packed['git_blob']),(compressed,'0'*40),(compressed+b'extra',packed['git_blob'])):
                with self.subTest(size=len(data),blob=blob),self.assertRaises((ValueError,OSError,EOFError)):g.consume_remote(io.BytesIO(data),rawpath,raw,packed,{'sha':blob},root)
            rawpath.write_bytes(b'changed payload\n')
            with self.assertRaises(ValueError):g.consume_remote(io.BytesIO(compressed),rawpath,raw,packed,{'sha':packed['git_blob']},root)
    def test_gzip_ledger_crosswire_missing_duplicate_and_false_fetch_rejected(self):
        spec=lossless_spec();pairs,_=lossless.representation_plan(spec);pair=next(iter(pairs.values()));raw,packed=pair['raw'],pair['gzip']
        row={'raw_path':raw['path'],'raw_bytes':raw['size'],'raw_sha256':raw['sha256'],'remote_gzip_path':packed['path'],
             'remote_git_blob':packed['git_blob'],'remote_gzip_bytes':packed['size'],'remote_gzip_sha256':packed['sha256'],
             'actual_remote_fetch_this_run':True,'decoded_byte_for_byte_equal_to_preserved_raw':True}
        lossless.validate_decode_rows([row],pairs)
        for rows in ([],[row,row],[dict(row,actual_remote_fetch_this_run=False)],[dict(row,raw_sha256='0'*64)],[dict(row,remote_gzip_bytes=True)]):
            with self.assertRaises(ValueError):lossless.validate_decode_rows(rows,pairs)

class PreparationClosureTests(unittest.TestCase):
    def test_actual_phase_roles_cannot_be_relabelled(self):
        c.validate_runner('stage','/root');c.validate_runner('lossless','/root/backup_manager')
        for phase,runner in (('stage','/root/backup_manager'),('recovery','/root/backup_manager'),('join','/root')):
            with self.subTest(phase=phase),self.assertRaises(ValueError):c.validate_runner(phase,runner)
    def test_every_mandatory_runtime_dependency_omission_rejected(self):
        for phase,keys in c.PHASE_KEYS.items():
            rows={key:{'path':'fixtures/'+key} for key in keys}
            c.validate_runtime_membership(phase,rows)
            for key in keys:
                missing=dict(rows);missing.pop(key)
                with self.subTest(phase=phase,key=key),self.assertRaises(ValueError):c.validate_runtime_membership(phase,missing)
    def test_runtime_duplicate_path_and_untyped_rows_rejected(self):
        rows={key:{'path':'same'} for key in c.PHASE_KEYS['inventory']}
        with self.assertRaises(ValueError):c.validate_runtime_membership('inventory',rows)
        rows={key:None for key in c.PHASE_KEYS['inventory']}
        with self.assertRaises(ValueError):c.validate_runtime_membership('inventory',rows)
    def test_source_preparation_freeze_omissions_and_substitution_rejected(self):
        # The bootstrap itself is already fully source-gated before this test imports it.
        import next653_bootstrap as boot
        path=c.HERE/'SOURCE_PREPARATION_FREEZE.json';base=json.loads(path.read_bytes())
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            p=Path(d)/'probe-freeze.json'
            for index,row in enumerate(base['frozen_inputs']):
                altered=copy.deepcopy(base);altered['frozen_inputs'].pop(index);p.write_text(json.dumps(altered))
                with self.subTest(omitted=row['path']),self.assertRaises(ValueError):boot.validate_freeze(p,boot.digest(p))
            for mutate in ('empty','source','test','config'):
                altered=copy.deepcopy(base)
                if mutate=='empty':altered['frozen_inputs']=[]
                else:
                    suffix={'source':'common.py','test':'tests.py','config':'POLICY.json'}[mutate]
                    row=next(r for r in altered['frozen_inputs'] if r['path'].endswith('/'+suffix));row['sha256']='0'*64
                p.write_text(json.dumps(altered))
                with self.subTest(mutation=mutate),self.assertRaises(ValueError):boot.validate_freeze(p,boot.digest(p))
    def test_exact_historical279_rule_rejects_changes_missing_archives_and_extra_stale(self):
        cfg=c.fixed_json('history');prior=c.fixed_json('prior');files,raw=c.all_entries(prior);entries={**files,**raw};rep=cfg['old_representation'];old=cfg['old_raw']
        record={'raw_path':old['path'],'raw_bytes':old['size'],'raw_sha256':old['sha256'],'gzip_path':rep['gzip']['path'],'gzip_bytes':rep['gzip']['size'],'gzip_sha256':rep['gzip']['sha256']}
        calls=[]
        inventory.historical_rule(rep['manifest']['path'],record,entries[old['path']],entries,lambda a,b:calls.append((str(a),str(b))))
        self.assertEqual(len(calls),2)
        for mutate in ('missing_archive','archive_sha','old_gzip_sha','current_raw_sha','current_manifest','other_stale_manifest'):
            changed=copy.deepcopy(entries);r=copy.deepcopy(record);manifest=rep['manifest']['path']
            archive=cfg['archive_pack_records'][0]['raw_path']
            if mutate=='missing_archive':changed.pop(archive)
            elif mutate=='archive_sha':changed[archive]['sha256']='0'*64
            elif mutate=='old_gzip_sha':r['gzip_sha256']='0'*64
            elif mutate=='current_raw_sha':changed[old['path']]['sha256']='0'*64
            elif mutate=='current_manifest':changed.pop(cfg['current_pack_record']['manifest_path'])
            else:manifest='fixtures/extra-stale.lossless.json'
            with self.subTest(mutation=mutate),self.assertRaises((ValueError,KeyError)):inventory.historical_rule(manifest,r,changed[old['path']],changed,lambda a,b:None)
    def test_metadata_report_typed_span_and_missing_row_rejected(self):
        s=c.SecretSession()
        for row in s.expected.values():s.file(c.safe(c.FACTORY,row['path']),row['path'],'source_before')
        report=s.complete('source_before');c.validate_metadata_report(report,'source_before')
        for mutation in ('span_bool','missing','duplicate','context','config'):
            changed=copy.deepcopy(report)
            if mutation=='span_bool':changed['four_exception_files'][0]['matches'][0]['pattern_index']=True
            elif mutation=='missing':changed['four_exception_files'].pop()
            elif mutation=='duplicate':changed['four_exception_files'][1]=changed['four_exception_files'][0]
            elif mutation=='context':changed['context']='actual_remote'
            else:changed['configuration_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.validate_metadata_report(changed,'source_before')
    def test_every_mandatory_phase_key_declared(self):
        self.assertEqual(c.PHASE_KEYS['inventory'],{'preparation_tests','admission_contract','archive_mapping_contract','runtime_freeze','catalog'})
        for phase in ('stage','recovery','lossless','join'):
            self.assertIn('admission_contract',c.PHASE_KEYS[phase])
            self.assertIn('archive_mapping_contract',c.PHASE_KEYS[phase])
        self.assertIn('stage_execution_config',c.PHASE_KEYS['recovery'])
        self.assertIn('recovery_execution_config',c.PHASE_KEYS['lossless'])
        self.assertIn('lossless_execution_config',c.PHASE_KEYS['join'])
    def test_binding_extra_key_alias_and_bool_size_rejected(self):
        for row in ({'path':'missing','size':True,'sha256':'0'*64},{'path':'../outside','size':0,'sha256':'0'*64},{'path':'missing','size':0,'sha256':'0'*64,'extra':True}):
            with self.assertRaises(ValueError):c.validate_binding(row)
    def test_dynamic_scope_no_count_only_fixed_replacements(self):
        a=simple_spec();one=c.scope(a);a['files'].append(fixture('fixtures/c.txt',b'different ordinary bytes'))
        two=c.scope(a);self.assertEqual(two['files'],one['files']+1);self.assertEqual(two['unique_git_blob_groups'],one['unique_git_blob_groups']+1)

def core_fixture():
    p=admission.policy();modules=p['actual_module_inventory'];runtime=p['core_fixed']['runtime_freeze']
    output='\n'.join(f'test_case ({m}.Suite.test_case) ... ok' for m in modules)+'\n\nRan 40 tests in 1.0s\n\nOK\n'
    out=c.projection(fixture('fixtures/FULL_OUTPUT.txt',output.encode()));exe=c.projection(fixture('fixtures/EXECUTION.json'))
    execution={'status':'PASS','mode':'full','actual_runner_execution_id':'/root','author_execution_id':'/root/formal_oracle',
               'actual_test_modules':modules,'actual_full_suite_module_inventory':modules,'tests_run':40,
               'errors':0,'failures':0,'skipped':0,'changed_runtime_bindings':[],
               'runtime_freeze_sha256':runtime['sha256'],'verified_preimport_bindings':507,
               'full_output_bytes':out['size'],'full_output_sha256':out['sha256']}
    completion={'actual_runner_execution_id':'/root','actual_process_exit_code':0,'full40_suite_ran':True,
                'status':'FULL40_ACTUAL40_PASS_SOURCE_507_PAIRS_UNCHANGED','actual_test_modules':modules,
                'errors':0,'failures':0,'skipped':0,'data_approval':False,'all_failed_attempts_preserved':True,
                'tests_run':40,'pass':40,'verified_current_and_raw_runtime_pairs':507,
                'runtime_freeze':{'path':runtime['path'],'bytes':runtime['size'],'sha256':runtime['sha256']},
                'outputs':[{'path':r['path'],'bytes':r['size'],'sha256':r['sha256']} for r in (exe,out)],
                'per_module_test_counts':{m:1 for m in modules}}
    completion['actual_executed_labels']=admission.derive_output(output,modules,40)[1]
    return execution,completion,output,modules,runtime,507,exe,out

def phase_fixture(phase='recovery'):
    inputs={key:c.projection(fixture('fixtures/'+key+'.json',key.encode())) for key in c.PHASE_KEYS[phase]}
    freeze=c.projection(fixture('fixtures/SOURCE_PREPARATION_FREEZE.json'))
    fields={'inventory':{'pending_observed':[],'byte_evolution':[],'raw_relocations':[],'stable_root_hold':True},
            'stage':{'allow_ignored_exact_paths':[],'message':'isolated fixture'},
            'recovery':{'commit_sha':'1'*40,'workers':2,'allow_evolving_source':False},
            'lossless':{'commit_sha':'1'*40},'join':{'commit_sha':'1'*40},'test':{}}[phase]
    value={'schema':'next-checkpoint-653-execution-v2','phase':phase,'checkpoint_id':'isolated-fixture',
           'runner_execution':'/root' if phase in ('stage','recovery') else '/root/backup_manager','expected_parent':c.PARENT,
           'authorization':{'authorizer_execution':'/root','phase_granted':phase,'source_only_preparation_is_not_grant':True},
           'source_freeze':freeze,'inputs':inputs,'fields':fields,
           'outputs':{key:'backup_logs/isolated-'+phase+'-'+key+'.json' for key in c.OUTPUT_KEYS[phase]}}
    return value,freeze,inputs

class ActualAdmissionRejectionTests(unittest.TestCase):
    def test_real_complete_full_core_producer_contract_positive(self):
        # Requires the actual completed full run supplied after ROOT source review/grant.
        result=c.validate_admission(TEST_CONFIG)
        self.assertEqual(result['core']['actual_module_count'],40)
        self.assertGreater(result['core']['actual_tests_run'],0)
        self.assertEqual(result['producer']['actual_artifacts'],240)
        self.assertEqual(result['producer']['actual_runtime_pairs'],103)
        self.assertEqual(result['producer']['actual_outputs'],8)
        self.assertTrue(result['required_factory_original_and_raw_closure'])
    def test_preserved_failed466_and_focused23_cannot_open_full_gate(self):
        args=list(core_fixture())
        bindings=[admission.policy()[key] for key in ('preserved_failed_full_execution','preserved_focused_execution')]
        for row in bindings:
            args[0]=c.bound_json(row)
            with self.subTest(path=row['path']),self.assertRaises(ValueError):admission.validate_core_result(*args)
    def test_full_core_errors_skips_fail_zero_bool_and_wrong_runner_rejected(self):
        base=list(core_fixture());admission.validate_core_result(*base)
        changes=[('status','FAIL_TESTS'),('errors',1),('skipped',1),('failures',1),('tests_run',0),
                 ('tests_run',True),('failures',False),('verified_preimport_bindings',True),
                 ('actual_runner_execution_id','/root/formal_oracle'),('runtime_freeze_sha256','0'*64),
                 ('full_output_sha256','0'*64),('full_output_bytes',False),('changed_runtime_bindings',['changed'])]
        for key,value in changes:
            args=copy.deepcopy(base);args[0][key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):admission.validate_core_result(*args)
    def test_nested_success_cannot_replace_actual_full_result(self):
        args=list(core_fixture());args[0]={'exit_code':0,'failures':0,'bindings_unchanged':True,
                                         'nested':args[0],'status':'FAIL_TESTS','errors':1,'tests_run':0}
        with self.assertRaises(ValueError):admission.validate_core_result(*args)
    def test_full_core_module_sets_and_root_count_output_crosswires_rejected(self):
        base=list(core_fixture())
        for mutation in ('missing','duplicate','wrong','root_count','root_exit_bool','root_focused','output_hash','freeze','counts_bool'):
            args=copy.deepcopy(base)
            if mutation=='missing':args[0]['actual_test_modules']=args[0]['actual_test_modules'][:-1]
            elif mutation=='duplicate':args[0]['actual_test_modules'][-1]=args[0]['actual_test_modules'][0]
            elif mutation=='wrong':args[0]['actual_full_suite_module_inventory']=['tests.unrelated']*40
            elif mutation=='root_count':args[1]['tests_run']=39
            elif mutation=='root_exit_bool':args[1]['actual_process_exit_code']=False
            elif mutation=='root_focused':args[1]['full40_suite_ran']=False
            elif mutation=='output_hash':args[1]['outputs'][1]['sha256']='0'*64
            elif mutation=='freeze':args[1]['runtime_freeze']['sha256']='0'*64
            else:args[1]['per_module_test_counts'][args[3][0]]=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):admission.validate_core_result(*args)
    def test_output_label_ambiguity_missing_duplicate_summary_and_error_fail(self):
        args=core_fixture();text,modules=args[2],args[3];first=text.splitlines()[0]
        for body in (text.replace(' ... ok',' ... ERROR',1),text.replace(first+'\n','',1),first+'\n'+text,
                     text.replace('Ran 40','Ran 39'),text.replace('tests.oracle_belief_controls_test','tests.unrelated',1),
                     text.replace('OK\n','FAILED\n'),text+'Ran 40 tests in 1.0s\nOK\n'):
            with self.assertRaises(ValueError):admission.validate_output(body,modules,40)
    def test_missing_changed_original_code_helper_or_raw_archive_rejected(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);original=root/'original.py';archive=root/'raw.source';payload=b'ordinary source fixture\n'
            original.write_bytes(payload);archive.write_bytes(payload)
            path=original.relative_to(c.FACTORY).as_posix();arc=archive.relative_to(c.FACTORY).as_posix()
            row={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
            admission.runtime_pair(path,row,arc,{}, {},set())
            for which in ('original','archive'):
                chosen=original if which=='original' else archive;chosen.write_bytes(b'changed')
                with self.assertRaises(ValueError):admission.runtime_pair(path,row,arc,{}, {},set())
                chosen.write_bytes(payload)
            archive.unlink()
            with self.assertRaises(ValueError):admission.runtime_pair(path,row,arc,{}, {},set())
    def test_only_exact_living_catalog_evolution_preserves_old_raw_bytes(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);p=root/'living.md';arc=root/'old.source';old=b'old living';new=b'new living'
            p.write_bytes(new);arc.write_bytes(old);path=p.relative_to(c.FACTORY).as_posix()
            a=c.projection(fixture(path,old));b=c.projection(fixture(path,new));z=c.projection(fixture(arc.relative_to(c.FACTORY).as_posix(),old))
            rule={'prior':a,'current':b,'archived_prior':z};mapping=admission.evolution_rows([rule],{path});used=set()
            admission.runtime_pair(path,{'bytes':len(old),'sha256':a['sha256']},z['path'],{},mapping,used)
            self.assertEqual(used,{path})
            for row in (dict(rule,archived_prior=b),dict(rule,current=a),dict(rule,prior=dict(a,sha256='0'*64))):
                with self.assertRaises(ValueError):admission.evolution_rows([row],{path})
            with self.assertRaises(ValueError):admission.evolution_rows([rule],{'some-other-living.md'})
            arc.write_bytes(b'changed')
            with self.assertRaises(ValueError):admission.runtime_pair(path,{'bytes':len(old),'sha256':a['sha256']},z['path'],{},mapping,set())
    def producer_documents(self):
        p=admission.policy();v={k:c.bound_json(r) for k,r in p['producer_fixed'].items()}
        return [v['final'],v['prepared_freeze'],v['runtime_freeze'],v['execution'],v['completion_readback'],p]
    def test_actual_fixed_producer_complete_shape_positive(self):
        a,o,runtime,operative=admission.validate_producer_shape(*self.producer_documents())
        self.assertEqual((len(a),len(o),len(runtime)),(240,8,103));self.assertTrue(operative)
    def test_arbitrary_empty_stale_stopped_producer_and_wrong_root_fail(self):
        base=self.producer_documents()
        for mutation in ('placeholder','empty_operative','missing_operative_config','unstopped','wrong_final','stale_runtime','bool_count','wrong_actor','missing_artifact','missing_output'):
            args=copy.deepcopy(base)
            if mutation=='placeholder':args[0]={'placeholder':True}
            elif mutation=='empty_operative':args[1]['mandatory_operative_source_paths']=[]
            elif mutation=='missing_operative_config':args[1]['mandatory_operative_source_paths']=args[1]['mandatory_operative_source_paths'][1:]
            elif mutation=='unstopped':args[0]['writes_stop_after_this_receipt']=False
            elif mutation=='wrong_final':args[4]['producer_final_sha256']='0'*64
            elif mutation=='stale_runtime':args[2]['prepared_freeze_sha256']='0'*64
            elif mutation=='bool_count':args[4]['verified_artifact_bindings']=True
            elif mutation=='wrong_actor':args[0]['actual_author_execution_id']='/root/unrelated'
            elif mutation=='missing_artifact':args[0]['all_owned_preparation_execution_output_inventory_excluding_this_receipt'].pop()
            else:args[0]['eight_core_artifacts'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):admission.validate_producer_shape(*args)
    def test_producer_config_reference_raw_omission_duplicate_and_crosswire_fail(self):
        base=self.producer_documents()
        for mutation in ('omit_config','omit_reference','omit_raw','duplicate','changed_source','wrong_output'):
            args=copy.deepcopy(base);rows=args[2]['bindings']
            if mutation=='omit_config':rows[:]=[r for r in rows if not r['path'].endswith('PRODUCTION_CONFIG_ATTEMPT01.json')]
            elif mutation=='omit_reference':rows[:]=[r for r in rows if not r['path'].endswith('reference_attempt01.py')]
            elif mutation=='omit_raw':rows[0].pop('runtime_archive')
            elif mutation=='duplicate':rows[-1]=rows[0]
            elif mutation=='changed_source':rows[0]['sha256']='0'*64
            else:args[3]['completed_artifacts'][0]['sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):admission.validate_producer_shape(*args)
    def test_parent_and_output_symlink_copy_rejected(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.HERE) as b:
            factory,worktree=Path(a),Path(b);(factory/'nested').mkdir();(factory/'nested/probe').write_bytes(b'ordinary')
            row=fixture('nested/probe',b'ordinary');target=worktree/c.PREFIX/'nested/probe';target.parent.mkdir(parents=True)
            target.symlink_to(factory/'nested/probe')
            with self.assertRaises(ValueError):stage.copy_file(row,c.SecretSession(),factory,worktree)
            target.unlink();target.parent.rmdir();target.parent.symlink_to(factory/'nested')
            with self.assertRaises(ValueError):stage.copy_file(row,c.SecretSession(),factory,worktree)

class CompletedPhaseContractTests(unittest.TestCase):
    def test_completed_test_empty_fields_and_receipt_provenance(self):
        value,freeze,inputs=phase_fixture('test')
        c.validate_completed_config_value(value,'test','isolated-fixture',freeze,inputs)
        changed=copy.deepcopy(value);changed['fields']={'unexpected_override':True}
        with self.assertRaisesRegex(ValueError,'Completed test phase accepts no fields'):
            c.validate_completed_config_value(changed,'test','isolated-fixture',freeze,inputs)
        # Bound isolated receipt inputs keep actual roles, freeze, runtime and output provenance intact.
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);prefix=root.relative_to(c.FACTORY).as_posix();freeze=TEST_CONFIG['_source_freeze']
            def save(filename,obj):
                path=root/filename;path.write_text(json.dumps(obj));return c.file_binding(path)
            actual_inputs={key:TEST_CONFIG['inputs'][key] for key in c.PHASE_KEYS['test']}
            positive=copy.deepcopy(value);positive['source_freeze']=freeze;positive['inputs']=actual_inputs
            positive['outputs']={'receipt':prefix+'/positive-receipt.json','failure':prefix+'/positive-failure.json'}
            positive_binding=save('positive-config.json',positive)
            text='isolated positive completed-test receipt provenance fixture'
            def receipt(config_binding):
                return {'status':'PASS','tests_run':1,'failures':0,'errors':0,'skipped':0,
                        'source_freeze':freeze,'actual_runner_execution':'/root/backup_manager',
                        'no_git_or_network_execution':True,'output':text,
                        'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'execution_config':config_binding}
            positive_receipt=save('positive-receipt.json',receipt(positive_binding))
            context={'checkpoint_id':'isolated-fixture','_source_freeze':freeze,
                     'inputs':dict(actual_inputs,preparation_tests=positive_receipt)}
            c.validate_test_receipt(context)
            negative=copy.deepcopy(positive);negative['fields']={'unexpected_override':True}
            negative['outputs']={'receipt':prefix+'/negative-receipt.json','failure':prefix+'/negative-failure.json'}
            negative_binding=save('negative-config.json',negative)
            negative_receipt=save('negative-receipt.json',receipt(negative_binding))
            negative_context=dict(context,inputs=dict(actual_inputs,preparation_tests=negative_receipt))
            with self.assertRaisesRegex(ValueError,'Completed test phase accepts no fields'):
                c.validate_test_receipt(negative_context)
    def test_completed_proof_config_freeze_output_ledger_and_dependency_substitutions(self):
        # Temp fixture receipts are unit inputs, never production success artifacts.
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);prefix=root.relative_to(c.FACTORY).as_posix();freeze=TEST_CONFIG['_source_freeze'];spec=simple_spec()
            def save(filename,value):
                path=root/filename;path.write_text(json.dumps(value));return c.file_binding(path)
            common={key:save(key+'.json',{'isolated_unit_fixture':key}) for key in c.PHASE_KEYS['join']}
            common['admission_contract']=TEST_CONFIG['inputs']['admission_contract']
            common['archive_mapping_contract']=TEST_CONFIG['inputs']['archive_mapping_contract']
            common['runtime_freeze']=TEST_CONFIG['inputs']['runtime_freeze']
            testcfg={'schema':'next-checkpoint-653-execution-v2','phase':'test','checkpoint_id':'isolated-fixture',
                     'runner_execution':'/root/backup_manager','expected_parent':c.PARENT,
                     'authorization':{'authorizer_execution':'/root','phase_granted':'test','source_only_preparation_is_not_grant':True},
                     'source_freeze':freeze,'inputs':{key:common[key] for key in c.PHASE_KEYS['test']},'fields':{},
                     'outputs':{'receipt':prefix+'/tests.json','failure':prefix+'/test-failure.json'}}
            testbinding=save('test-config.json',testcfg);text='isolated successful validator fixture'
            common['preparation_tests']=save('tests.json',{'status':'PASS','tests_run':1,'failures':0,'errors':0,'skipped':0,
                'source_freeze':freeze,'actual_runner_execution':'/root/backup_manager','no_git_or_network_execution':True,
                'output':text,'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'execution_config':testbinding})
            context={'checkpoint_id':'isolated-fixture','inputs':common,'_source_freeze':freeze}
            for phase,label,proofkey,ledgerkey in [('stage','stage_execution_config','stage','stage_ledger'),
                                                   ('recovery','recovery_execution_config','full_proof','full_ledger'),
                                                   ('lossless','lossless_execution_config','gzip_proof','gzip_ledger')]:
                value,_,_=phase_fixture(phase);value['source_freeze']=freeze
                value['inputs']={key:common[key] for key in c.PHASE_KEYS[phase]}
                value['outputs']['receipt']=common[proofkey]['path'];value['outputs']['ledger']=common[ledgerkey]['path']
                if phase=='stage':value['outputs']['local_commit']=common['local_commit']['path']
                configbinding=save(phase+'-config.json',value);common[label]=configbinding
                proof={'source_freeze':freeze,'execution_config':configbinding,'actual_runner_execution':value['runner_execution'],
                       'commit_sha':'1'*40,'spec':common['spec'],'exact_scope':c.scope(spec),
                       'stage':common['stage'],'local_commit':common['local_commit'],'stage_ledger':common['stage_ledger'],
                       'full_proof':common['full_proof'],'full_ledger':common['full_ledger'],'ledger':common[ledgerkey]}
                c.validate_completed_proof(context,phase,proof,common[proofkey],spec)
                for mutation in ('spec_as_freeze','inventory_as_config','other_output','other_ledger','other_runner','other_spec','config_phase','config_checkpoint','config_ledger','config_dependency'):
                    changed=copy.deepcopy(proof);altered=copy.deepcopy(value)
                    if mutation=='spec_as_freeze':changed['source_freeze']=common['spec']
                    elif mutation=='inventory_as_config':changed['execution_config']=common['inventory_execution_config']
                    elif mutation=='other_output':
                        altered['outputs']['receipt']=prefix+'/wrong-proof.json'
                    elif mutation=='other_ledger':changed['ledger']=common['inventory_report'];changed['stage_ledger']=common['inventory_report']
                    elif mutation=='other_runner':changed['actual_runner_execution']='/root/wrong'
                    elif mutation=='other_spec':changed['spec']=common['inventory_report']
                    elif mutation=='config_phase':altered['phase']='inventory'
                    elif mutation=='config_checkpoint':altered['checkpoint_id']='unrelated'
                    elif mutation=='config_ledger':altered['outputs']['ledger']=prefix+'/wrong-ledger.json'
                    else:altered['inputs']['admission_contract']=common['inventory_report']
                    if mutation.startswith('config_') or mutation=='other_output':
                        row=save(phase+'-'+mutation+'.json',altered);old=common[label];common[label]=row;changed['execution_config']=row
                    else:old=common[label]
                    try:
                        with self.subTest(phase=phase,mutation=mutation),self.assertRaises(ValueError):c.validate_completed_proof(context,phase,changed,common[proofkey],spec)
                    finally:common[label]=old
                if phase=='stage':
                    c.validate_completed_proof(context,phase,proof,common['local_commit'],spec,local_commit=True)
                    changed=dict(proof,execution_config=common['inventory_execution_config'])
                    with self.assertRaises(ValueError):c.validate_completed_proof(context,phase,changed,common['local_commit'],spec,local_commit=True)
    def test_each_completed_phase_actual_schema_runner_freeze_dependency_and_outputs(self):
        for phase in ('inventory','stage','recovery','lossless','join'):
            value,freeze,inputs=phase_fixture(phase)
            c.validate_completed_config_value(value,phase,'isolated-fixture',freeze,inputs)
            for mutation in ('schema','phase','runner','checkpoint','parent','freeze','missing_input','input_hash','output_alias','grant_bool'):
                v=copy.deepcopy(value)
                if mutation=='schema':v['schema']='next-checkpoint-653-exact-inventory-v1'
                elif mutation=='phase':v['phase']='test'
                elif mutation=='runner':v['runner_execution']='/root/wrong'
                elif mutation=='checkpoint':v['checkpoint_id']='different-checkpoint'
                elif mutation=='parent':v['expected_parent']='0'*40
                elif mutation=='freeze':v['source_freeze']=inputs.get('spec',c.projection(fixture('fixtures/SPEC.json')))
                elif mutation=='missing_input':v['inputs'].pop('admission_contract')
                elif mutation=='input_hash':v['inputs']['admission_contract']['sha256']='0'*64
                elif mutation=='output_alias':v['outputs']['receipt']=v['outputs']['failure']
                else:v['authorization']['source_only_preparation_is_not_grant']=1
                with self.subTest(phase=phase,mutation=mutation),self.assertRaises(ValueError):c.validate_completed_config_value(v,phase,'isolated-fixture',freeze,inputs)
    def test_inventory_test_stage_full_gzip_configs_cannot_be_relabelled(self):
        for actual in ('test','inventory','stage','recovery','lossless'):
            value,freeze,_=phase_fixture(actual)
            for expected in ('recovery','lossless'):
                if expected==actual:continue
                with self.subTest(actual=actual,expected=expected),self.assertRaises(ValueError):c.validate_completed_config_value(value,expected,'isolated-fixture',freeze,value['inputs'])
    def test_actual_fresh_transport_group_engine_and_blob_tamper(self):
        payloads=[b'ordinary group a\n',b'ordinary group b\n'];rows=[fixture('fixtures/'+str(i),p) for i,p in enumerate(payloads)]
        files={r['path']:r for r in rows};groups=c.groups(files);store={r['git_blob']:p for r,p in zip(rows,payloads)};calls=[]
        def fetch(blob,consume):calls.append(blob);return consume(io.BytesIO(store[blob]))
        actual=[recovery.fetch_group(row,c.SecretSession(),fetch) for row in groups.values()]
        c.validate_actual_groups(actual,groups,'actual_remote_fetch_this_run');self.assertEqual(sorted(calls),sorted(groups))
        blob=next(iter(store));store[blob]=b'changed'
        with self.assertRaises(ValueError):recovery.fetch_group(groups[blob],c.SecretSession(),fetch)
    def test_isolated_completed_full_gzip_ledger_chain_and_new_call_accounting(self):
        # Actual bounded transport/consumer functions on isolated bytes; no Git/network claim.
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            root=Path(d);payload=b'ordinary raw fixture\n'*1400000;rawpath=root/'raw';rawpath.write_bytes(payload)
            compressed=gzip.compress(payload,mtime=0);raw=fixture('fixtures/raw',payload)
            packed=fixture('fixtures/raw.gz',compressed);manifest=fixture('fixtures/raw.lossless.json',b'ordinary manifest\n')
            raw['lossless_representations']=[{'gzip':c.projection(packed),'manifest':c.projection(manifest)}]
            spec={'schema':'next-checkpoint-653-exact-inventory-v1','files':[packed,manifest],'raw_size_exclusions':[raw]}
            store={packed['git_blob']:compressed,manifest['git_blob']:b'ordinary manifest\n'};calls=[]
            def fetch(blob,consume):calls.append(blob);return consume(io.BytesIO(store[blob]))
            groups=c.groups(c.all_entries(spec)[0]);fullrows=[recovery.fetch_group(row,c.SecretSession(),fetch) for row in groups.values()]
            pairs,_=lossless.representation_plan(spec);decodes=lossless.fetch_pairs(pairs,fetch=fetch,resolver=lambda r:rawpath,spool_root=root)
            self.assertEqual(len(calls),len(groups)+len(pairs));self.assertEqual(calls.count(packed['git_blob']),2)
            full={'actual_remote_blob_groups':fullrows};gz={'actual_remote_decodes':decodes}
            remote=[{'event':'actual_remote_blob_verified',**r} for r in fullrows]
            local=[{'event':'local_git_blob_verified',**r,'actual_local_git_stream_this_run':True} for r in groups.values()]
            decoded=[{'event':'actual_remote_gzip_decoded',**r} for r in decodes]
            join.validate_ledger_chain(spec,full,gz,remote,local,decoded)
            for mutation in ('missing_full','extra_full','cached_full','wrong_full','missing_gzip','duplicate_gzip','cached_gzip','wrong_gzip','missing_local'):
                a,b,z=copy.deepcopy((remote,local,decoded))
                if mutation=='missing_full':a.pop()
                elif mutation=='extra_full':a.append(a[0])
                elif mutation=='cached_full':a[0]['actual_remote_fetch_this_run']=False
                elif mutation=='wrong_full':a[0]['sha256']='0'*64
                elif mutation=='missing_gzip':z.pop()
                elif mutation=='duplicate_gzip':z.append(z[0])
                elif mutation=='cached_gzip':z[0]['actual_remote_fetch_this_run']=False
                elif mutation=='wrong_gzip':z[0]['raw_sha256']='0'*64
                else:b.pop()
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):join.validate_ledger_chain(spec,full,gz,a,b,z)

class NewRawBinaryMappingTests(unittest.TestCase):
    def test_exact_six_original_diff_shape_and_actual_binary_roundtrips(self):
        self.assertEqual(len(archive_mapping.EXACT_ORIGINAL_DIFF_EVIDENCE),6)
        for path,fixed in archive_mapping.EXACT_ORIGINAL_DIFF_EVIDENCE.items():
            raw={'path':path,'size':fixed['size'],'sha256':fixed['sha256']}
            origin={'path':fixed['archive_origin_path'],'size':fixed['size'],'sha256':fixed['sha256']}
            self.assertEqual(c.projection(c.file_binding(c.FACTORY/path)),raw)
            self.assertEqual(c.projection(c.file_binding(c.FACTORY/origin['path'])),origin)
            with self.subTest(path=path),tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.FACTORY/'reproducibility/lossless_archives') as b,tempfile.TemporaryDirectory(dir=c.FACTORY/'manifests/lossless_archives') as d:
                root=Path(a);packed=Path(b)/'exact-diff.gz';packed.write_bytes(gzip.compress((c.FACTORY/path).read_bytes(),mtime=0))
                gz=c.file_binding(packed);manifest=Path(d)/'exact-diff.lossless.json'
                manifest.write_text(json.dumps({'raw_path':path,'raw_bytes':raw['size'],'raw_sha256':raw['sha256'],
                                                'gzip_path':gz['path'],'gzip_bytes':gz['size'],'gzip_sha256':gz['sha256']})+'\n')
                m=c.file_binding(manifest);warnings=archive_mapping.whitespace_evidence(c.FACTORY/path)
                row={'raw':raw,'origin':origin,'gzip':gz,'manifest':m,'whitespace_evidence':warnings}
                archive_mapping.mapping_shape(row,set(),{origin['path']:origin,path:raw})
                contract={'schema':archive_mapping.SCHEMA,'checkpoint_id':'isolated-fixture','expected_parent':c.PARENT,
                          'raw_mappings':[row],'grants_data_approval':False}
                contractpath=root/'mapping.json';contractpath.write_text(json.dumps(contract)+'\n')
                config={'checkpoint_id':'isolated-fixture','inputs':{'archive_mapping_contract':c.file_binding(contractpath)}}
                rep={'gzip':gz,'manifest':m,'exact_raw_roundtrip_verified':True,'uncompressed_secret_scan':'PASS'}
                spec={'schema':'next-checkpoint-653-exact-inventory-v2','files':[{'path':p.relative_to(c.FACTORY).as_posix(),**c.hash_file(p)} for p in (packed,manifest)],
                      'raw_size_exclusions':[dict(raw,reason=archive_mapping.REASON,lossless_representations=[rep])],
                      'new_runtime_raw_binary_mapping_contract':config['inputs']['archive_mapping_contract'],'new_runtime_raw_binary_mappings':[row]}
                actual=archive_mapping.validate(config,{'required_factory_original_and_raw_closure':[raw,origin]},spec)
                self.assertEqual(set(actual),{path});self.assertEqual(c.scope(spec)['unique_raw_gzip_pairs'],1)
                self.assertEqual(c.projection(c.file_binding(c.FACTORY/path)),raw)
                self.assertEqual(c.projection(c.file_binding(c.FACTORY/origin['path'])),origin)
    def test_exact_six_original_diff_corruptions_and_unconditional_protections(self):
        for path,fixed in archive_mapping.EXACT_ORIGINAL_DIFF_EVIDENCE.items():
            raw={'path':path,'size':fixed['size'],'sha256':fixed['sha256']}
            origin={'path':fixed['archive_origin_path'],'size':fixed['size'],'sha256':fixed['sha256']}
            row={'raw':raw,'origin':origin,'gzip':c.projection(fixture('reproducibility/lossless_archives/exact.gz')),
                 'manifest':c.projection(fixture('manifests/lossless_archives/exact.lossless.json')),
                 'whitespace_evidence':[{'line':1,'type':'trailing_whitespace','line_sha256':'1'*64}]}
            archive_mapping.mapping_shape(row,set(),{origin['path']:origin})
            for mutation in ('path','raw_hash','raw_size','origin_path','origin_hash','origin_size','unknown_diff','prior','release','opaque'):
                changed=copy.deepcopy(row);previous=set()
                if mutation=='path':changed['raw']['path']=path+'.alias'
                elif mutation=='raw_hash':changed['raw']['sha256']='0'*64;changed['origin']['sha256']='0'*64
                elif mutation=='raw_size':changed['raw']['size']+=1;changed['origin']['size']+=1
                elif mutation=='origin_path':changed['origin']['path']='quality_reports/other/raw_source_archives/'+fixed['sha256']+'.source'
                elif mutation=='origin_hash':changed['origin']['sha256']='0'*64
                elif mutation=='origin_size':changed['origin']['size']+=1
                elif mutation=='unknown_diff':changed['raw']['path']='quality_reports/ordinary/unknown.patch'
                elif mutation=='prior':previous={path}
                elif mutation=='release':changed['raw']['path']='data/RSM-001/releases/v9999/original.patch'
                else:changed['raw']['path']=next(iter(c.METADATA_GIT_BLOBS))
                admitted={changed['origin']['path']:changed['origin']}
                with self.subTest(path=path,mutation=mutation),self.assertRaises(ValueError):
                    archive_mapping.mapping_shape(changed,previous,admitted)
    def mapping_fixture(self,root,gzroot,manifestroot):
        payload=b'ordinary frozen source line \nordinary next line\n'
        original=root/'original.py';raw=root/'raw_source_archives'/'original.source'
        raw.parent.mkdir();original.write_bytes(payload);raw.write_bytes(payload)
        packed=gzroot/'fixture.gz';packed.write_bytes(gzip.compress(payload,mtime=0))
        binding=lambda p:c.file_binding(p)
        r,g,o=binding(raw),binding(packed),binding(original)
        manifest=manifestroot/'fixture.lossless.json'
        manifest.write_text(json.dumps({'raw_path':r['path'],'raw_bytes':r['size'],'raw_sha256':r['sha256'],
                                      'gzip_path':g['path'],'gzip_bytes':g['size'],'gzip_sha256':g['sha256']})+'\n')
        m=binding(manifest)
        row={'raw':r,'gzip':g,'manifest':m,'origin':o,'whitespace_evidence':archive_mapping.whitespace_evidence(raw)}
        contract={'schema':archive_mapping.SCHEMA,'checkpoint_id':'isolated-fixture','expected_parent':c.PARENT,
                  'raw_mappings':[row],'grants_data_approval':False}
        configpath=root/'mapping.json';configpath.write_text(json.dumps(contract)+'\n')
        cfg={'checkpoint_id':'isolated-fixture','inputs':{'archive_mapping_contract':binding(configpath)}}
        admission_snapshot={'required_factory_original_and_raw_closure':[o,r]}
        rep={'gzip':g,'manifest':m,'exact_raw_roundtrip_verified':True,'uncompressed_secret_scan':'PASS'}
        spec={'schema':'next-checkpoint-653-exact-inventory-v2','files':[{'path':p.relative_to(c.FACTORY).as_posix(),**c.hash_file(p)} for p in (packed,manifest)],
              'raw_size_exclusions':[{**r,'reason':archive_mapping.REASON,'lossless_representations':[rep]}],
              'new_runtime_raw_binary_mapping_contract':cfg['inputs']['archive_mapping_contract'],
              'new_runtime_raw_binary_mappings':[row]}
        return row,cfg,admission_snapshot,spec,(original,raw,packed,manifest)
    def test_exact_new_raw_binary_roundtrip_fullstream_manifest_and_scope(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.FACTORY/'reproducibility/lossless_archives') as b,tempfile.TemporaryDirectory(dir=c.FACTORY/'manifests/lossless_archives') as d:
            row,cfg,snapshot,spec,paths=self.mapping_fixture(Path(a),Path(b),Path(d))
            actual=archive_mapping.validate(cfg,snapshot,spec)
            self.assertEqual(set(actual),{row['raw']['path']});self.assertEqual(c.scope(spec)['unique_raw_gzip_pairs'],1)
            self.assertEqual(c.scope(spec)['files'],2);self.assertEqual(c.scope(spec)['remote_decoded_bytes'],row['raw']['size'])
            # Original raw remains local and exact; tracked object is compressed actual bytes.
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
            scanned=c.SecretSession().file(paths[1],row['raw']['path'],'source_before')
            self.assertEqual(scanned['sha256'],row['raw']['sha256'])
            pairs,_=lossless.representation_plan(spec);calls=[];store={spec['files'][0]['git_blob']:paths[2].read_bytes()}
            def fetch(blob,consumer):calls.append(blob);return consumer(io.BytesIO(store[blob]))
            decodes=lossless.fetch_pairs(pairs,fetch=fetch,resolver=lambda r:paths[1],spool_root=Path(a))
            self.assertEqual(len(calls),1);self.assertEqual(decodes[0]['raw_sha256'],row['raw']['sha256'])
    def test_old_tracked_or_raw_path_cannot_be_mapped_out(self):
        raw=c.projection(fixture('quality_reports/example/raw_source_archives/raw.source',b'ordinary \n'))
        origin=dict(raw,path='tests/actual.py');gz=c.projection(fixture('reproducibility/lossless_archives/raw.gz'))
        manifest=c.projection(fixture('manifests/lossless_archives/raw.lossless.json'))
        row={'raw':raw,'origin':origin,'gzip':gz,'manifest':manifest,
             'whitespace_evidence':[{'line':1,'type':'trailing_whitespace','line_sha256':'1'*64}]}
        archive_mapping.mapping_shape(row,set(),{origin['path']:origin})
        with self.assertRaises(ValueError):archive_mapping.mapping_shape(row,{raw['path']},{origin['path']:origin})
    def test_under_cap_raw_requires_exact_schema_reason_and_mapping_membership(self):
        row=fixture('fixtures/raw.source',b'ordinary \n')
        for spec in ({'files':[],'raw_size_exclusions':[row]},
                     {'schema':'next-checkpoint-653-exact-inventory-v2','files':[],'raw_size_exclusions':[dict(row,reason='ordinary_omission')]},
                     {'schema':'next-checkpoint-653-exact-inventory-v2','files':[row],'raw_size_exclusions':[dict(row,reason=archive_mapping.REASON)]}):
            with self.assertRaises(ValueError):c.all_entries(spec)
    def test_mutable_raw_origin_gzip_manifest_and_opaque_lines_rejected(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.FACTORY/'reproducibility/lossless_archives') as b,tempfile.TemporaryDirectory(dir=c.FACTORY/'manifests/lossless_archives') as d:
            row,cfg,snapshot,spec,paths=self.mapping_fixture(Path(a),Path(b),Path(d))
            archive_mapping.validate(cfg,snapshot,spec)
            for path in paths:
                original=path.read_bytes();path.write_bytes(original+b'changed')
                with self.subTest(path=path.name),self.assertRaises((ValueError,OSError,EOFError)):archive_mapping.validate(cfg,snapshot,spec)
                path.write_bytes(original)
            for mutation in ('origin','path','hash','size_bool','wrong_line','duplicate_line','no_line','known_metadata_alias','release','unadmitted'):
                r=copy.deepcopy(row);admitted={row['origin']['path']:row['origin']}
                if mutation=='origin':r['origin']['sha256']='0'*64
                elif mutation=='path':r['raw']['path']='ordinary/source.py'
                elif mutation=='hash':r['raw']['sha256']='0'*64
                elif mutation=='size_bool':r['raw']['size']=True
                elif mutation=='wrong_line':r['whitespace_evidence'][0]['line']=True
                elif mutation=='duplicate_line':r['whitespace_evidence'].append(r['whitespace_evidence'][0])
                elif mutation=='no_line':r['whitespace_evidence']=[]
                elif mutation=='known_metadata_alias':r['raw']['path']=next(iter(c.METADATA_GIT_BLOBS))
                elif mutation=='release':r['raw']['path']='data/RSM-001/releases/v9999/raw_source_archives/raw.source'
                else:admitted={}
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):archive_mapping.mapping_shape(r,set(),admitted)
    def test_missing_wrong_representation_or_fake_mapping_cannot_admit(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as a,tempfile.TemporaryDirectory(dir=c.FACTORY/'reproducibility/lossless_archives') as b,tempfile.TemporaryDirectory(dir=c.FACTORY/'manifests/lossless_archives') as d:
            row,cfg,snapshot,spec,_=self.mapping_fixture(Path(a),Path(b),Path(d))
            for mutation in ('missing_gzip','missing_manifest','extra_raw','changed_reason','wrong_package','unbound_contract'):
                s=copy.deepcopy(spec)
                if mutation=='missing_gzip':s['files'].pop(0)
                elif mutation=='missing_manifest':s['files'].pop(1)
                elif mutation=='extra_raw':s['raw_size_exclusions'].append(dict(s['raw_size_exclusions'][0],path='other/raw.source'))
                elif mutation=='changed_reason':s['raw_size_exclusions'][0]['reason']='skip'
                elif mutation=='wrong_package':s['raw_size_exclusions'][0]['lossless_representations'][0]['gzip']['sha256']='0'*64
                else:s['new_runtime_raw_binary_mapping_contract']['sha256']='0'*64
                with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):archive_mapping.validate(cfg,snapshot,s)
    def test_whitespace_raw_evidence_never_authorizes_new_text_diff_warning(self):
        warning=b'datasets/rineya_high_quality_factory_v1/new/raw.source:212: trailing whitespace.\n+ordinary line \n'
        with self.assertRaises(ValueError):c.assert_strict_diff({'returncode':2,'stdout':warning,'stderr':b''})
        self.assertEqual(c.assert_strict_diff({'returncode':0,'stdout':b'','stderr':b''})['warning_count'],0)
    def test_actual_ten_living_transition_crosswires_and_catalog_evolution_rejected(self):
        template=json.loads((c.HERE/'ADMISSION_CONTRACT_TEMPLATE.json').read_bytes())
        for key in ('template_only_not_executable','remove_template_key_after_separate_root_phase_grant'):template.pop(key)
        rows=admission.evolution_rows(template['later_living_or_catalog_evolution'],set(admission.policy()['allowed_later_living_or_catalog_paths']))
        admission.validate_living_transition(template,rows,{})
        for mutation in ('missing','old','current','archive','root'):
            t=copy.deepcopy(template);r=copy.deepcopy(rows)
            if mutation=='missing':r.pop(next(iter(r)))
            elif mutation=='root':t['living_transition_root_completion']['sha256']='0'*64
            else:r[next(iter(r))][{'old':'prior','current':'current','archive':'archived_prior'}[mutation]]['sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):admission.validate_living_transition(t,r,{})

class OperativeRuntimeGuardTests(unittest.TestCase):
    def runtime(self):
        value=c.bound_json(TEST_CONFIG['inputs']['runtime_freeze'])
        return value,value['source_code_bindings']
    def test_actual_collected_runtime_current_raw_and_complete_source_closure(self):
        value,sources=self.runtime();runtime_guard.validate_value(value,sources)
        self.assertTrue(value['runtime_bindings']);self.assertEqual(value['helper_imports_tests_inventory_git_network'],0)
        self.assertEqual(runtime_guard.validate(TEST_CONFIG,'post_phase')['status'],'PASS_EXACT_OPERATIVE_RUNTIME_CURRENT_AND_RAW')
    def test_runtime_wrong_python_cache_source_membership_count_and_role_rejected(self):
        base,sources=self.runtime()
        for mutation in ('python','version','cache_tag','preload','sources','role','bool_count','raw_size','raw_hash','extra_key','missing_source'):
            value=copy.deepcopy(base)
            if mutation=='python':value['requested_python']='/wrong/python'
            elif mutation=='version':value['python_version']='wrong'
            elif mutation=='cache_tag':value['python_cache_tag']='wrong'
            elif mutation=='preload':value['preload_modules'].pop()
            elif mutation=='sources':value['source_code_bindings'].pop()
            elif mutation=='role':value['actual_collector_execution']='/root/unrelated'
            elif mutation=='bool_count':value['helper_imports_tests_inventory_git_network']=False
            elif mutation=='raw_size':value['runtime_bindings'][0]['raw_archive']['size']=True
            elif mutation=='raw_hash':value['runtime_bindings'][0]['raw_archive']['sha256']='0'*64
            elif mutation=='extra_key':value['runtime_bindings'][0]['unbound_alias']=True
            else:value['source_code_bindings']=[]
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,KeyError)):runtime_guard.validate_value(value,sources)
    def test_runtime_stream_current_and_raw_tamper_rejected_without_production_change(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as d:
            p=Path(d)/'runtime.source';p.write_bytes(b'ordinary runtime fixture')
            row=c.file_binding(p);runtime_guard.check_binding(row)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):runtime_guard.check_binding(row)
            with self.assertRaises(ValueError):runtime_guard.check_binding(dict(row,path='../outside'))
    def test_runtime_all_phases_require_exact_collection_and_no_spec_alias(self):
        for phase,keys in c.PHASE_KEYS.items():
            self.assertIn('runtime_freeze',keys)
            rows={key:{'path':key} for key in keys};rows.pop('runtime_freeze')
            with self.subTest(phase=phase),self.assertRaises(ValueError):c.validate_runtime_membership(phase,rows)
        wrong=copy.deepcopy(TEST_CONFIG);wrong['inputs']['runtime_freeze']=wrong['_source_freeze']
        with self.assertRaises((ValueError,KeyError)):runtime_guard.validate(wrong,'preimport')

class ExactRegistryPhysicalPathTests(unittest.TestCase):
    def test_real_canonical_closure_preserves_exact_logical_and_historical_bindings(self):
        value=admission.registry_policy();logical=value['logical_current']
        witness=admission.exact_registry_alias(logical)
        closure={};admission.add(closure,logical)
        self.assertEqual(set(closure),{admission.REGISTRY_CANONICAL})
        self.assertEqual(closure[admission.REGISTRY_CANONICAL],value['canonical_current'])
        self.assertNotIn(admission.REGISTRY_LOGICAL,closure)
        actual=admission.validate(TEST_CONFIG)
        rows={r['path']:r for r in actual['required_factory_original_and_raw_closure']}
        self.assertEqual(rows[admission.REGISTRY_CANONICAL],value['canonical_current'])
        self.assertNotIn(admission.REGISTRY_LOGICAL,rows)
        self.assertEqual(actual['exact_physical_path_aliases'],[witness])
        transition=next(r for r in actual['later_living_or_catalog_evolution'] if r['current']['path']==admission.REGISTRY_LOGICAL)
        self.assertEqual(transition['current'],logical)
        self.assertEqual(transition['prior'],value['historical_core507_logical'])
        self.assertEqual(witness['historical_parent653_canonical'],value['historical_parent653_canonical'])
        observation=copy.deepcopy(witness['actual_physical_observation'])
        # Canonical-only case-sensitive restoration is a distinct truthful state.
        observation['logical_lookup_present']=False
        observation['logical_binding']=None;observation['logical_identity']=None
        admission.validate_registry_observation(logical,observation,value)

    def test_registry_exact_tuple_policy_identity_and_two_physical_name_corruptions_rejected(self):
        value=admission.registry_policy();logical=value['logical_current']
        for key,replacement in [('path','research/Role_Registry.json'),('path','fixtures/role_registry.json'),
                                ('path',admission.REGISTRY_CANONICAL),('size',19280),('size',True),('sha256','0'*64)]:
            changed=dict(logical);changed[key]=replacement
            with self.subTest(binding=key,value=replacement),self.assertRaises(ValueError):admission.exact_registry_alias(changed)
        for role in ('logical_current','canonical_current','historical_parent653_canonical','historical_core507_logical'):
            for field,replacement in [('path','research/Role_Registry.json'),('size',True),('sha256','0'*64)]:
                changed=copy.deepcopy(value);changed[role][field]=replacement
                with self.subTest(policy_role=role,field=field),self.assertRaises(ValueError):admission.validate_registry_policy(changed)
        original=admission.exact_registry_alias(logical)['actual_physical_observation']
        present=copy.deepcopy(original);present['logical_lookup_present']=True
        present['logical_binding']=logical;present['logical_identity']=copy.deepcopy(present['canonical_identity'])
        admission.validate_registry_observation(logical,present,value)
        variants=[]
        wrong=copy.deepcopy(present);wrong['logical_identity']['inode']+=1;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['logical_identity']['device']+=1;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['canonical_identity']['inode']=True;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['observed_registry_directory_entries']=['ROLE_REGISTRY.json','role_registry.json'];variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['observed_registry_directory_entries']=['Role_Registry.json'];variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['canonical_binding']['sha256']='0'*64;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['logical_binding']['size']=19280;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['logical_lookup_present']=False;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['logical_lookup_present']=1;variants.append(wrong)
        wrong=copy.deepcopy(present);wrong['unexpected_alias']='fixtures/copied.json';variants.append(wrong)
        for index,changed in enumerate(variants):
            with self.subTest(observation=index),self.assertRaises(ValueError):admission.validate_registry_observation(logical,changed,value)


class ExactFailedStageRecoveryPreparationTests(unittest.TestCase):
    def test_exact46_literal_mapping_shapes_and_mutations(self):
        self.assertEqual(len(archive_mapping.EXACT_FAILED_STAGE_EVIDENCE),46)
        for path,fixed in archive_mapping.EXACT_FAILED_STAGE_EVIDENCE.items():
            row={'raw':fixed['raw'],'origin':fixed['origin'],'gzip':c.projection(fixture('reproducibility/lossless_archives/exact.gz')),
                 'manifest':c.projection(fixture('manifests/lossless_archives/exact.lossless.json')),
                 'whitespace_evidence':[{'line':1,'type':'trailing_whitespace','line_sha256':'1'*64}]}
            archive_mapping.mapping_shape(row,set(),{})
            for mutation in ('raw_hash','raw_size','origin_hash','origin_size','origin_path','unknown_path','prior','release','opaque'):
                r=copy.deepcopy(row);previous=set()
                if mutation=='raw_hash':r['raw']['sha256']='0'*64
                elif mutation=='raw_size':r['raw']['size']+=1
                elif mutation=='origin_hash':r['origin']['sha256']='0'*64
                elif mutation=='origin_size':r['origin']['size']+=1
                elif mutation=='origin_path':r['origin']['path']+='-unapproved'
                elif mutation=='unknown_path':r['raw']['path']+='-unapproved'
                elif mutation=='prior':previous={path}
                elif mutation=='release':r['raw']['path']='data/RSM-001/releases/v9999/raw_archives/evidence.source'
                else:r['raw']['path']=next(iter(c.METADATA_GIT_BLOBS))
                with self.subTest(path=path,mutation=mutation),self.assertRaises((ValueError,KeyError)):
                    archive_mapping.mapping_shape(r,previous,{r['origin']['path']:r['origin']})
    def test_closed_existing13_plus46_mapping_extension_rejects_omissions_and_substitutions(self):
        policy=archive_mapping.exact_warning_policy();base=c.bound_json(policy['base_mapping_contract'])
        rows=copy.deepcopy(base['raw_mappings'])
        for fixed in archive_mapping.EXACT_FAILED_STAGE_EVIDENCE.values():
            rows.append({'raw':fixed['raw'],'origin':fixed['origin']})
        value={'raw_mappings':rows}
        archive_mapping.validate_closed_warning_extension(value,policy)
        for mutation in ('missing','extra','old_row','raw','origin','duplicate'):
            v=copy.deepcopy(value)
            if mutation=='missing':v['raw_mappings'].pop()
            elif mutation=='extra':v['raw_mappings'].append({'raw':dict(rows[-1]['raw'],path='unknown/evidence.source'),'origin':rows[-1]['origin']})
            elif mutation=='old_row':v['raw_mappings'][0]['gzip']['sha256']='0'*64
            elif mutation=='raw':v['raw_mappings'][-1]['raw']=dict(rows[-1]['raw'],size=True)
            elif mutation=='origin':v['raw_mappings'][-1]['origin']=dict(rows[-1]['origin'],sha256='0'*64)
            else:v['raw_mappings'].append(copy.deepcopy(rows[-1]))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):archive_mapping.validate_closed_warning_extension(v,policy)
    def test_fresh_worktree_and_failed_native_index_policy_guards(self):
        self.assertEqual(c.WORKTREE.name,'.rineya-dataset-git-worktree-r2');self.assertNotEqual(c.WORKTREE,c.FAILED_WORKTREE)
        policy=json.loads((c.HERE/'FAILED_WORKTREE_PRESERVATION_POLICY.json').read_bytes())
        # The prepared real failed index is checked without importing or executing old stage code.
        result=c.validate_failed_worktree_preservation_value(policy,verify_targets=False)
        self.assertEqual(result['status'],'PASS_FAILED_NATIVE_INDEX_AND_COPIED_TARGETS_PRESERVED')
        self.assertFalse(result['all_failed_targets_byte_verified'])
        for field,value in [('failed_worktree',str(c.WORKTREE)),('fresh_worktree',str(c.FAILED_WORKTREE)),('parent_at_failure','0'*40),('failed_target_files',True),('grants_data_approval',True)]:
            p=copy.deepcopy(policy);p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_failed_worktree_preservation_value(p,verify_targets=False)
        p=copy.deepcopy(policy);p['native_failed_index']['sha256']='0'*64
        with self.assertRaises(ValueError):c.validate_failed_worktree_preservation_value(p,verify_targets=False)
        with self.assertRaises(ValueError):c.validate_failed_worktree_preservation_value(policy,native_index_path=c.WORKTREE/'wrong-index',verify_targets=False)

def execute(cfg):
    global TEST_CONFIG
    TEST_CONFIG=cfg
    if cfg['fields']!={}:raise ValueError('Test phase accepts no production-scope overrides')
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output=stream.getvalue()
    # Test output includes method names/errors only; no opaque metadata match text.
    receipt={'status':'PASS' if result.wasSuccessful() else 'FAILED_TESTS_PRESERVED','author_execution':'/root/backup_manager',
             'actual_runner_execution':cfg['runner_execution'],'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'output':output,'output_sha256':hashlib.sha256(output.encode()).hexdigest(),'source_freeze':cfg['_source_freeze'],
             'execution_config':cfg['_config'],'no_git_or_network_execution':True,'grants_data_approval':False}
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    if not result.wasSuccessful():raise ValueError('Prepared meaningful corruption suite failed; exact output preserved')
    return {'status':'PASS','receipt':row,'tests_run':result.testsRun,'failures':0,'errors':0}
