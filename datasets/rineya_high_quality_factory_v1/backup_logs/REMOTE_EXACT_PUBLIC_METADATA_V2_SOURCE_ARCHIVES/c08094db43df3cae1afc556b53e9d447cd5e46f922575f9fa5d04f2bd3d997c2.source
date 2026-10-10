#!/usr/bin/env python3
"""Actual isolated scanner/transport/schema regressions; no Git/network/main execution.
Opaque metadata bytes are read only from original authorized paths into streams;
no metadata content is copied to a new path or disclosed in output.
"""
import concurrent.futures,copy,hashlib,importlib.util,io,json,re,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
L=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('new_exact_remote',L/'recover_remote_checkpoint_exact_public_metadata_v2.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
h=r.metadata_helper()

def oid(data):return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()

class RemoteExactSafety(unittest.TestCase):
    def setUp(self):
        h.ALLOWED_OBSERVATIONS.clear()
        for context in r._STREAM_REPORTS:r._STREAM_REPORTS[context].clear()
    def payload(self,row):return (r.FACTORY/row['path']).read_bytes()
    def runrow(self,row,context='local_git',data=None,names=None,chunk=65536,blob=None):
        data=self.payload(row) if data is None else data
        with patch.object(r,'CHUNK',chunk):
            return r.hash_stream(io.BytesIO(data),row['size'],oid(data) if blob is None else blob,True,
                [row['path']] if names is None else names,context)
    def complete(self,context):
        for row in h.EXPECTED_FILES:self.runrow(row,context)
        return r.complete_metadata_stream_report(context)
    def test_all_four_actual_opaque_metadata_streams_full_size_sha_and_git(self):
        for context in ('local_git','actual_remote'):
            report=self.complete(context);r.validate_metadata_report(report,True)
            self.assertEqual(report['allowed_exact_match_detections'],4)
    def test_overlap_context_drop_chunk1_chunk2_and_short_reads(self):
        row=h.EXPECTED_FILES[0];data=self.payload(row)
        for chunk in (1,2,34,64,4096,65536):
            with self.subTest(chunk=chunk):self.assertEqual(self.runrow(row,chunk=chunk)['sha256'],row['sha256'])
        class Short(io.BytesIO):
            def __init__(self,data,cuts):super().__init__(data);self.cuts=list(cuts)
            def read(self,n=-1):
                stop=self.cuts.pop(0) if self.cuts else self.tell()+n
                return super().read(min(n,max(1,stop-self.tell())))
        for cuts in ([4660,4661],[4661,4662]):
            self.assertEqual(r.hash_stream(Short(data,cuts),len(data),oid(data),True,[row['path']],'actual_remote')['sha256'],row['sha256'])
    def test_64k_stream_reads_bounded(self):
        row=h.EXPECTED_FILES[0]
        class Bounded(io.BytesIO):
            def read(self,n=-1):
                if n<1 or n>65536:raise AssertionError('Unbounded stream request')
                return super().read(n)
        data=self.payload(row);self.assertEqual(r.hash_stream(Bounded(data),len(data),oid(data),True,[row['path']],'local_git')['size'],len(data))
    def test_changed_metadata_full_hash_rejected_after_exact_span(self):
        row=h.EXPECTED_FILES[0];data=self.payload(row)
        with self.assertRaises(ValueError):self.runrow(row,data=data[:-1]+bytes([data[-1]^1]))
        self.assertEqual(r._STREAM_REPORTS['local_git'],{})
    def test_truncated_extra_and_wrong_size_rejected(self):
        row=h.EXPECTED_FILES[0];data=self.payload(row)
        for value in (data[:-1],data+b'X'):
            with self.subTest(size=len(value)),self.assertRaises(ValueError):self.runrow(row,data=value)
        for size in (True,-1,row['size']+1):
            with self.assertRaises(ValueError):r.hash_stream(io.BytesIO(data),size,oid(data),True,[row['path']],'local_git')
    def test_git_sha1_tampering_cannot_record_an_exception(self):
        with self.assertRaises(ValueError):self.runrow(h.EXPECTED_FILES[0],blob='0'*40)
        self.assertEqual(r._STREAM_REPORTS['local_git'],{})
    def test_greedy_extension_span_shift_and_duplicate_match_rejected(self):
        row=h.EXPECTED_FILES[0];data=self.payload(row);m=row['matches'][0]
        variants=[data[:m['end']]+b'Z'+data[m['end']+1:], b' '+data[:-1],data+b'\n'+data[m['start']:m['end']]]
        for value in variants:
            with self.assertRaises(ValueError):self.runrow(row,data=value,chunk=1)
    def test_same_content_original_two_paths_group_allowed(self):
        rows=h.EXPECTED_FILES[2:];data=self.payload(rows[0]);names=[v['path'] for v in rows]
        self.assertEqual(self.runrow(rows[0],names=names)['sha256'],rows[0]['sha256'])
        self.assertEqual(set(r._STREAM_REPORTS['local_git']),set(names))
    def test_copied_alias_case_normalization_and_symlink_context_rejected(self):
        row=h.EXPECTED_FILES[0]
        for name in (row['path']+'.copy',row['path'].upper(),'./'+row['path'],'../'+row['path'], '/'+row['path'],row['path'].replace('/','//',1)):
            with self.subTest(alias=name),self.assertRaises(ValueError):self.runrow(row,names=[name])
    def test_actual_symlink_to_original_metadata_rejected_without_copy(self):
        with tempfile.TemporaryDirectory(prefix='owned-alias-rejection-',dir=L) as t:
            alias=Path(t)/'metadata-alias.json';alias.symlink_to(r.FACTORY/h.EXPECTED_FILES[0]['path'])
            with self.assertRaises(ValueError):h.proof(alias,r.SECRET_PATTERNS)
    def test_mixed_known_unknown_and_repeated_group_context_rejected(self):
        row=h.EXPECTED_FILES[0]
        for names in ([row['path'],'backup_logs/copy.json'],[row['path'],row['path']],[],None):
            with self.assertRaises(ValueError):r.exact_secret_stream(io.BytesIO(self.payload(row)),row['size'],oid(self.payload(row)),names,'local_git')
    def test_unknown_and_invalid_provenance_rejected_for_known_metadata(self):
        with self.assertRaises(ValueError):self.runrow(h.EXPECTED_FILES[0],context='cached_remote')
    def test_every_ordinary_secret_pattern_remains_rejected_cross_chunk(self):
        values=[b'gh'+b'p_'+b'A'*30,b'sk'+b'-'+b'A'*34,b'AK'+b'IA'+b'A'*16,
            b'-----BEGIN '+b'PRIVATE KEY-----',b'\npassword='+b'A'*20]
        for payload in values:
            for chunk in (1,64,65536):
                with self.subTest(pattern_digest=hashlib.sha256(payload).hexdigest(),chunk=chunk),self.assertRaises(ValueError):
                    with patch.object(r,'CHUNK',chunk):r.hash_stream(io.BytesIO(payload),len(payload),oid(payload),True,['backup_logs/ordinary.txt'],'actual_remote')
    def test_safe_inline_prose_remains_allowed_and_unchanged_regex_flags_required(self):
        data=b'ordinary prose password=short\n'
        self.assertEqual(r.hash_stream(io.BytesIO(data),len(data),oid(data),True,['backup_logs/ordinary.txt'],'local_git')['sha256'],hashlib.sha256(data).hexdigest())
        patterns=list(r.SECRET_PATTERNS);patterns[1]=re.compile(patterns[1].pattern,re.I)
        with patch.object(r,'SECRET_PATTERNS',patterns),self.assertRaises(ValueError):self.runrow(h.EXPECTED_FILES[0])
    def test_missing_one_or_duplicate_or_extra_report_is_not_complete(self):
        for row in h.EXPECTED_FILES[:3]:self.runrow(row)
        with self.assertRaises(ValueError):r.complete_metadata_stream_report('local_git')
        self.runrow(h.EXPECTED_FILES[3]);valid=r.complete_metadata_stream_report('local_git')
        for rows in (valid['four_exception_files'][:3],valid['four_exception_files']+[valid['four_exception_files'][0]],valid['four_exception_files'][:3]+[valid['four_exception_files'][0]]):
            with self.assertRaises(ValueError):r.validate_metadata_report({**valid,'four_exception_files':rows},True)
    def test_report_hash_span_type_and_truth_tampering_rejected(self):
        valid=self.complete('actual_remote')
        for key,value in [('sha256','0'*64),('size',True),('matches',[]),('full_size_sha256_gitsha1_verified_before_exception_acceptance',False),('git_blob','0')]:
            q=copy.deepcopy(valid);q['four_exception_files'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_metadata_report(q,True)
        q=copy.deepcopy(valid);q['four_exception_files'][0]['matches'][0]['pattern_index']=True
        with self.assertRaises(ValueError):r.validate_metadata_report(q,True)
        for key,value in [('status','PASS'),('allowed_exact_match_detections',True),('unbound_secret_detections',1),('configuration_sha256','0'*64)]:
            with self.assertRaises(ValueError):r.validate_metadata_report({**valid,key:value},True)
    def test_source_checks_four_observations_and_changed_ordinary_source(self):
        for row in h.EXPECTED_FILES:r.check_source(r.FACTORY,{k:row[k] for k in ('path','size','sha256')},False)
        r.validate_metadata_report(h.report(require_complete=True))
        with tempfile.TemporaryDirectory(prefix='owned-source-race-',dir=L) as t:
            path=Path(t)/'plain.txt';path.write_bytes(b'ordinary source')
            e={'path':path.relative_to(r.FACTORY).as_posix(),**r.hash_file(path)}
            self.assertIsNone(r.check_source(r.FACTORY,e,False));path.write_bytes(b'changed source')
            with self.assertRaises(ValueError):r.check_source(r.FACTORY,e,False)
    def test_threads_report_only_completed_verified_streams_without_cross_race(self):
        groups=[[h.EXPECTED_FILES[0]['path']],[h.EXPECTED_FILES[1]['path']],[v['path'] for v in h.EXPECTED_FILES[2:]]]
        rows=[h.EXPECTED_FILES[0],h.EXPECTED_FILES[1],h.EXPECTED_FILES[2]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda x:self.runrow(x[0],'actual_remote',names=x[1]),zip(rows,groups)))
        r.validate_metadata_report(r.complete_metadata_stream_report('actual_remote'),True)
    def test_actual_scope_and_all_known_spec_entries_mandatory(self):
        spec=json.loads((L/'belief-v4-approved-release-r1_SPEC.json').read_bytes());r.validate_exact_scope(spec)
        for mutation in ('missing','hash','raw','rep'):
            q=copy.deepcopy(spec)
            if mutation=='missing':q['files'].pop()
            elif mutation=='hash':next(e for e in q['files'] if e['path']==h.EXPECTED_FILES[0]['path'])['sha256']='0'*64
            elif mutation=='raw':q['raw_size_exclusions'].pop()
            else:q['raw_size_exclusions'][0]['lossless_representations'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.validate_exact_scope(q)
    def test_actual_frozen_stage_truth_and_extra_warning_rejected(self):
        m=r.import_bound(L/'run_provisional_archival_checkpoint_v4_attempt02.py',r.STAGE_SOURCE_SHA,'stage_guard_test')
        stage=json.loads((L/'belief-v4-approved-release-r1_STAGE.json').read_bytes());r.validate_stage_observation(stage,m)
        for key,value in [('actual_exit_code',0),('warning_count',5),('restricted_changed_path_count',481),('whole_index_and_outside_parent_guard','FAIL'),('expected_parent_sha','0'*40)]:
            with self.assertRaises(ValueError):r.validate_stage_observation({**stage,key:value},m)
    def test_source_freeze_cli_rejects_wrong_runner_parent_config_or_evolution_before_execution(self):
        a=SimpleNamespace(workers=2,allow_evolving_source=False,runner='/root',commit=r.EXPECTED_COMMIT,expected_spec_sha256=r.EXPECTED_SPEC_SHA,
            metadata_config=r.FACTORY/'backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json',expected_metadata_config_sha256=r.METADATA_CONFIG_SHA,
            source_freeze=L/'never-existing-freeze.json',expected_freeze_sha256='0'*64,prior_evidence=[])
        for key,value in [('workers',1),('allow_evolving_source',True),('runner','other'),('commit','0'*40),('expected_spec_sha256','0'*64),('metadata_config',L/'copied-config.json')]:
            q=copy.copy(a);setattr(q,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):r.check_new_source_freeze(q)

if __name__=='__main__':unittest.main(verbosity=2)
