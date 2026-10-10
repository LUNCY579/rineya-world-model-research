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
    def test_zero_errors_does_not_admit_actual_failed_integration(self):
        values={'exit_code':1,'failures':4,'errors':0,'bindings_unchanged':True}
        predicates=[{'field':[key],'equals':value} for key,value in values.items()]
        with self.assertRaises(ValueError):inventory.integration_gate(values,predicates)
        good={'exit_code':0,'failures':0,'bindings_unchanged':True}
        predicates=[{'field':[key],'equals':value} for key,value in good.items()]
        inventory.integration_gate(good,predicates)
        for bad in (dict(good,exit_code=1),dict(good,failures=4),dict(good,bindings_unchanged=False),dict(good,exit_code=False)):
            with self.assertRaises(ValueError):inventory.integration_gate(bad,predicates)
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
        self.assertEqual(c.PHASE_KEYS['join'],{'preparation_tests','spec','inventory_report','inventory_execution_config','stage','local_commit','stage_ledger','full_proof','full_ledger','gzip_proof','gzip_ledger'})
        self.assertEqual(c.PHASE_KEYS['inventory'],{'preparation_tests','integration_execution','producer_freeze','catalog'})
    def test_binding_extra_key_alias_and_bool_size_rejected(self):
        for row in ({'path':'missing','size':True,'sha256':'0'*64},{'path':'../outside','size':0,'sha256':'0'*64},{'path':'missing','size':0,'sha256':'0'*64,'extra':True}):
            with self.assertRaises(ValueError):c.validate_binding(row)
    def test_dynamic_scope_no_count_only_fixed_replacements(self):
        a=simple_spec();one=c.scope(a);a['files'].append(fixture('fixtures/c.txt',b'different ordinary bytes'))
        two=c.scope(a);self.assertEqual(two['files'],one['files']+1);self.assertEqual(two['unique_git_blob_groups'],one['unique_git_blob_groups']+1)

def execute(cfg):
    if cfg['fields']!={}:raise ValueError('Test phase accepts no production-scope overrides')
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output=stream.getvalue()
    # Test output includes method names/errors only; no opaque metadata match text.
    receipt={'status':'PASS' if result.wasSuccessful() else 'FAILED_TESTS_PRESERVED','author_execution':'/root/backup_manager',
             'actual_runner_execution':cfg['runner_execution'],'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'output':output,'output_sha256':hashlib.sha256(output.encode()).hexdigest(),'source_freeze':cfg['_source_freeze'],
             'execution_config':cfg['_config'],'no_git_or_network_execution':True,'grants_data_approval':False}
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    if not result.wasSuccessful():raise ValueError('Prepared meaningful corruption suite failed; exact output preserved')
    return {'status':'PASS','receipt':row,'tests_run':result.testsRun,'failures':0,'errors':0}
