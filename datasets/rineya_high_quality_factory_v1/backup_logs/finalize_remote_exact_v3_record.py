from pathlib import Path
import hashlib
L=Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs')
p=L/'recover_remote_checkpoint_exact_public_metadata_v3.py';s=p.read_text();old=hashlib.sha256(p.read_bytes()).hexdigest()
s=s.replace("    return {'status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','context':context,", "    report={'status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','context':context,")
s=s.replace("'four_exception_files':[actual[n] for n in sorted(actual)]}\n\ndef typed_equal", "'four_exception_files':[actual[n] for n in sorted(actual)]}\n    return validate_metadata_report(report,True,context)\n\ndef typed_equal")
p.write_text(s);sha=hashlib.sha256(p.read_bytes()).hexdigest();p=L/'verify_archival_remote_lossless_exact_public_metadata_v3.py';s=p.read_text().replace(old,sha);p.write_text(s)
s=(L/'test_remote_checkpoint_exact_public_metadata_v2.py').read_text().replace('_v2.py','_v3.py')
s=s.replace("r.validate_metadata_report(report,True)","r.validate_metadata_report(report,True,context)")
s=s.replace("r.validate_metadata_report({**valid,'four_exception_files':rows},True)","r.validate_metadata_report({**valid,'four_exception_files':rows},True,'local_git')")
s=s.replace("r.validate_metadata_report(q,True)","r.validate_metadata_report(q,True,'actual_remote')")
s=s.replace("r.validate_metadata_report({**valid,key:value},True)","r.validate_metadata_report({**valid,key:value},True,'actual_remote')")
s=s.replace("r.validate_metadata_report(r.complete_metadata_stream_report('actual_remote'),True)","r.validate_metadata_report(r.complete_metadata_stream_report('actual_remote'),True,'actual_remote')")
pos=s.index("\nif __name__")
s=s[:pos]+'''
    def test_missing_expected_git_id_and_wrong_context_rejected_before_admission(self):
        row=h.EXPECTED_FILES[0];data=self.payload(row)
        for blob in (None,True,3,[],{},'', 'Z'*40):
            with self.subTest(type=type(blob).__name__),self.assertRaises(ValueError):
                r.exact_secret_stream(io.BytesIO(data),row['size'],blob,[row['path']],'local_git')
        self.assertEqual(r._STREAM_REPORTS['local_git'],{})
        report=self.complete('local_git')
        for value in (None,'actual_remote','cached'):
            with self.assertRaises(ValueError):r.validate_metadata_report(report,True,value)
        for key,value in [('context',None),('context','actual_remote')]:
            with self.assertRaises(ValueError):r.validate_metadata_report({**report,key:value},True,'local_git')
    def test_valid_looking_wrong_git_hash_cannot_be_reported_complete(self):
        report=self.complete('actual_remote');q=copy.deepcopy(report);q['four_exception_files'][0]['git_blob']='0'*40
        with self.assertRaises(ValueError):r.validate_metadata_report(q,True,'actual_remote')
        r._STREAM_REPORTS['actual_remote'][h.EXPECTED_FILES[0]['path']]['git_blob']='0'*40
        with self.assertRaises(ValueError):r.complete_metadata_stream_report('actual_remote')
    def test_full_dependency_freeze_rejects_empty_missing_duplicate_type_and_hash(self):
        freeze=json.loads((L/'REMOTE_EXACT_PUBLIC_METADATA_V3_RECOVERY_SOURCE_FREEZE.json').read_bytes());rows=freeze['frozen_inputs']
        self.assertTrue(r.validate_frozen_inputs(rows))
        for wrong in ([],rows[:-1],rows+[rows[0]],{}):
            with self.assertRaises(ValueError):r.validate_frozen_inputs(wrong)
        for name in r.MANDATORY_FROZEN_INPUTS:
            with self.subTest(missing=name),self.assertRaises(ValueError):r.validate_frozen_inputs([x for x in rows if x['path']!=name])
        q=copy.deepcopy(rows);q[0]['size']=True
        with self.assertRaises(ValueError):r.validate_frozen_inputs(q)
        q=copy.deepcopy(rows);q[0]['sha256']='0'*64
        with self.assertRaises(ValueError):r.validate_frozen_inputs(q)
'''+s[pos:]
# Removing an optional archive row can be legitimate; remove one mandatory row for missing-list test.
s=s.replace("for wrong in ([],rows[:-1],rows+[rows[0]],{}):", "for wrong in ([],[x for x in rows if x['path']!='backup_logs/recover_remote_checkpoint.py'],rows+[rows[0]],{}):")
(L/'test_remote_checkpoint_exact_public_metadata_v3.py').open('x').write(s)
s=(L/'test_archival_remote_lossless_exact_public_metadata_v2.py').read_text().replace('_v2.py','_v3.py')
s=s.replace('files=[self.packed,self.manifest]',"files=[self.packed,self.manifest]+[{k:e[k] for k in ('path','size','sha256')} for e in b.metadata_helper().EXPECTED_FILES]")
s=s.replace('        verified={"status"', '''        meta_groups={}
        for row in b.metadata_helper().EXPECTED_FILES:
            gid=b.METADATA_GIT_BLOBS[row['path']]
            if gid not in meta_groups:meta_groups[gid]={'git_blob':gid,'size':row['size'],'sha256':row['sha256'],'paths':[], 'verification_source':'actual_remote_raw_GitHub_blob'}
            meta_groups[gid]['paths'].append(row['path'])
        receipts+=list(meta_groups.values())
        verified={"status"''')
s=s.replace('"inventory":files,"files":2,','"inventory":files,"files":len(files),')
s=s.replace('"verified_files":2,"verified_unique_blobs":2', '"verified_files":len(files),"verified_unique_blobs":len(receipts)')
s=s.replace("({'git_blob':'a'*40} if stream else {})","({'git_blob':b.METADATA_GIT_BLOBS[row['path']]} if stream else {})")
s=s.replace("'existing_regexes_unchanged':True,'four_exception_files':rows}","'existing_regexes_unchanged':True,'four_exception_files':rows,**({'context':'local_git' if key=='actual_local_git_metadata_exception_scan' else 'actual_remote'} if stream else {})}")
s=s.replace('        return spec,stage,verified', "        verified['current_source_evolution']=[];verified['current_source_evolution_at_finish']=[]\n        return spec,stage,verified")
s=s.replace("freeze={'author_execution':r.AUTHOR,'frozen_before_first_import_test_execution':True,\n            'runner_source':{'path':Path(r.__file__).relative_to(r.FACTORY).as_posix(),**b.hash_file(Path(r.__file__))},'frozen_inputs':[]}", "freeze=json.loads((LOGS/'REMOTE_EXACT_PUBLIC_METADATA_V3_GZIP_SOURCE_FREEZE.json').read_bytes())")
s=s.replace("q['frozen_inputs']=[{'path':'backup_logs/recover_remote_checkpoint_exact_public_metadata_v3.py','size':0,'sha256':'0'*64}]", "q['frozen_inputs']=[]")
pos=s.index('\n\nif __name__')
s=s[:pos]+'''
    def test_wrong_git_or_local_context_substitution_in_remote_report_rejected(self):
        spec,stage,v=self.bindings()
        q=copy.deepcopy(v);q['actual_remote_metadata_exception_scan']['four_exception_files'][0]['git_blob']='0'*40
        with self.assertRaises(ValueError):r.validate_bindings(spec,stage,q,COMMIT,SPEC_SHA,b)
        q=copy.deepcopy(v);q['actual_remote_metadata_exception_scan']=copy.deepcopy(q['actual_local_git_metadata_exception_scan'])
        with self.assertRaises(ValueError):r.validate_bindings(spec,stage,q,COMMIT,SPEC_SHA,b)
    def test_metadata_git_report_cross_join_to_actual_blob_groups_required(self):
        spec,stage,v=self.bindings();q=copy.deepcopy(v)
        q['remote_verification']['blob_receipts'][2]['git_blob']='0'*40
        with self.assertRaises(ValueError):r.validate_bindings(spec,stage,q,COMMIT,SPEC_SHA,b)
    def test_source_evolution_after_or_before_is_rejected(self):
        spec,stage,v=self.bindings()
        for key in ('current_source_evolution','current_source_evolution_at_finish'):
            with self.assertRaises(ValueError):r.validate_bindings(spec,stage,{**v,key:[{'path':'changed.py'}]},COMMIT,SPEC_SHA,b)
'''+s[pos:]
(L/'test_archival_remote_lossless_exact_public_metadata_v3.py').open('x').write(s)
print('newv3 source/test drafts authored; no imports/tests yet')
