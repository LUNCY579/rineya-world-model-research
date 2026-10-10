#!/usr/bin/env python3
"""Final read-only publication/source integrity; only the report is saved.

All producer/oracle/reviewer payloads are hashed, never imported or parsed.
Archival helpers execute only against synthetic temporary fixtures.
"""
import ast
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
from collections import Counter,defaultdict
from adversarial_audit import ROOT,EXECUTION,canon,digest,require
from catalog_adversarial_audit import independent_selection,catalog_function


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def archival_shadow_checks():
    source=ROOT/'factory/pipeline.py';tree=ast.parse(source.read_text());helper_names={'preserve_source_bytes','preserve_prior_audit'}
    helper_ast=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helper_names],type_ignores=[])
    common=ast.parse((ROOT/'factory/common.py').read_text());atomic_ast=ast.Module(body=[n for n in common.body if isinstance(n,ast.FunctionDef) and n.name=='atomic'],type_ignores=[])
    checks=[]
    with tempfile.TemporaryDirectory(prefix='publication-red-shadow-') as d:
        fixture=pathlib.Path(d);ns={'ROOT':fixture,'Path':pathlib.Path,'os':os,'hashlib':hashlib}
        exec(compile(atomic_ast,str(ROOT/'factory/common.py'),'exec'),ns);exec(compile(helper_ast,str(source),'exec'),ns)
        source_file=fixture/'factory/example.py';source_file.parent.mkdir(parents=True)
        original='# é\r\nvalue = 1\r\n'.encode();source_file.write_bytes(original)
        mapping=ns['preserve_source_bytes']([source_file]);checksum=hashlib.sha256(original).hexdigest();blob=fixture/'reproducibility/source_blobs'/(checksum+'.source')
        require(mapping=={'factory/example.py':checksum} and blob.read_bytes()==original,'source helper failed exact UTF8/CRLF preservation')
        ns['preserve_source_bytes']([source_file]);require(blob.read_bytes()==original,'source helper idempotence changed blob')
        checks.append({'case':'source_bytes_UTF8_CRLF_and_idempotent_blob','pass':True})
        blob.write_bytes(b'corrupt fixture')
        try:ns['preserve_source_bytes']([source_file]);raise RuntimeError('corrupt source blob accepted')
        except AssertionError:checks.append({'case':'source_blob_mismatch_rejected','pass':True})
        audit=fixture/'audits/example.json';audit.parent.mkdir(parents=True);previous=b'{"old":"exact"}\r\n';audit.write_bytes(previous)
        ns['preserve_prior_audit'](audit,b'{"new":true}\n')
        snapshot=audit.parent/'history'/f'{audit.stem}-{hashlib.sha256(previous).hexdigest()}{audit.suffix}'
        require(snapshot.read_bytes()==previous and audit.read_bytes()==previous,'earlier machine bytes not preserved exactly')
        ns['preserve_prior_audit'](audit,previous)
        require(len(list((audit.parent/'history').iterdir()))==1,'identical rerun creates unnecessary history')
        checks.append({'case':'prior_machine_audit_exact_history_and_identical_rerun','pass':True})
        snapshot.write_bytes(b'corrupt history fixture')
        try:ns['preserve_prior_audit'](audit,b'changed');raise RuntimeError('corrupt audit snapshot accepted')
        except AssertionError:checks.append({'case':'prior_history_mismatch_rejected','pass':True})
    # Run the actual archiver in a shadow tree with no real Git/catalog calls.
    archive=ROOT/'reproducibility/archive_sources.py';archive_ns={'__name__':'shadow_archive','__file__':str(archive)}
    exec(compile(archive.read_text(),str(archive),'exec'),archive_ns)
    with tempfile.TemporaryDirectory(prefix='archive-red-shadow-') as d:
        fixture=pathlib.Path(d).resolve();archive_ns['FACTORY']=fixture
        payload=b'prior actual source bytes\r\n';checksum=hashlib.sha256(payload).hexdigest();missing='f'*64
        blob=fixture/'reproducibility/source_blobs'/(checksum+'.source');blob.parent.mkdir(parents=True);blob.write_bytes(payload)
        manifest=fixture/'data/T/releases/v1/MANIFEST.json';manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps({'track':'T','release':'v1','reproducibility_code_sha256':{'factory/old.py':checksum,'factory/missing.py':missing}}))
        archive_ns['git']=lambda args,optional=False: b'research/rineya-high-quality-dataset-v1\n' if args[:2]==['branch','--show-current'] else b'0'*40+b'\n' if args[0]=='rev-parse' else b''
        fake_parent=types.ModuleType('factory');fake_parent.__path__=[];fake_catalog=types.ModuleType('factory.catalog')
        fake_catalog.catalog=lambda:([],{'releases':[{'path':'data/T/releases/v1','active_records':1}]})
        old_modules={name:sys.modules.get(name) for name in ('factory','factory.catalog')};old_argv=list(sys.argv);old_path=list(sys.path)
        sys.modules['factory']=fake_parent;sys.modules['factory.catalog']=fake_catalog;sys.argv=['archive_sources.py','--runner',EXECUTION]
        try:
            archive_ns['main']();index=fixture/'reproducibility/SOURCE_ARCHIVE_INDEX.json';report=json.loads(index.read_text())
            require(report['preserved_requirements']==1 and report['missing_requirements']==report['missing_active_requirements']==1,'archive active/missing counts wrong')
            preserved_entry=next(e for e in report['entries'] if e['status']=='PRESERVED_EXACT_BYTES')
            require(preserved_entry['origin']['kind']=='existing_content_addressed_archive','existing blob not recognized independently of current files')
            before=index.read_bytes();archive_ns['main']()
            history=fixture/'reproducibility/archive_history'/(hashlib.sha256(before).hexdigest()+'.json')
            require(history.read_bytes()==before and blob.read_bytes()==payload,'source index rerun did not preserve exact prior index/payload')
            checks.append({'case':'existing_source_blob_reuse_missing_active_count_and_index_history','pass':True})
            blob.write_bytes(b'corrupt fixture')
            try:archive_ns['main']();raise RuntimeError('corrupt existing archive admitted')
            except RuntimeError as e:
                require('hash mismatch' in str(e),'unexpected archive rejection reason')
                checks.append({'case':'archiver_existing_blob_corruption_rejected','pass':True})
        finally:
            sys.argv=old_argv;sys.path[:]=old_path
            for name,old in old_modules.items():
                if old is None:sys.modules.pop(name,None)
                else:sys.modules[name]=old
    return checks


def historical_git_match():
    worktree=ROOT.parent/'.rineya-dataset-git-worktree';prefix='datasets/rineya_high_quality_factory_v1/'
    def git(args):return subprocess.check_output(['git',*args],cwd=worktree)
    branch=git(['branch','--show-current']).decode().strip();require(branch=='research/rineya-high-quality-dataset-v1','unexpected history branch')
    head=git(['rev-parse','HEAD']).decode().strip();checked=[]
    tree=git(['ls-tree','-r','HEAD','--',prefix+'data']).decode().splitlines()
    for line in tree:
        descriptor,name=line.split('\t',1);relative=name[len(prefix):]
        if '/releases/' not in relative:continue
        path=ROOT/relative;require(path.is_file(),'historical release file missing locally')
        expected=descriptor.split()[2];actual=git(['hash-object','--no-filters',str(path)]).decode().strip()
        require(actual==expected,'historical release bytes changed versus dedicated history: '+relative)
        checked.append({'path':relative,'git_blob':expected,'local_sha256':sha(path)})
    return {'branch':branch,'head':head,'historical_release_files_compared':len(checked),'files':checked,'method':'Git blob identity against existing dedicated HEAD; no fetch/push/object creation.'}


def main():
    failures=[];report={'reviewer_execution_id':EXECUTION,'reviewer_role':'Adversarial publication integrity reviewer','semantic_receipt_rewrites':False,'release_writes':False,'other_reviewer_verdict_contents_inspected':False,'scope':'Read-only canonical equivalence, checksum/source coverage and active catalog; archiver code tested only in temporary synthetic trees.'}
    try:
        releases=[];requirements=defaultdict(list);input_hashes={}
        for mp in sorted(ROOT.glob('data/*/releases/*/MANIFEST.json')):
            directory=mp.parent;m=json.loads(mp.read_text());checks={}
            for line in (directory/'CHECKSUMS.sha256').read_text().splitlines():
                checksum,name=line.split('  ',1);require(name==pathlib.Path(name).name,'unsafe checksum file path');require(name not in checks,'duplicate checksum entry');checks[name]=checksum
            actual={p.name for p in directory.iterdir() if p.is_file() and p.name!='CHECKSUMS.sha256'}
            require(set(checks)==actual,'immutable checksum coverage mismatch: '+str(directory))
            for name,checksum in checks.items():require(sha(directory/name)==checksum,'immutable checksum mismatch: '+str(directory/name))
            require(set(m['files'])==actual-{'MANIFEST.json'},'manifest file coverage mismatch: '+str(directory))
            for name,meta in m['files'].items():require(meta['bytes']==(directory/name).stat().st_size and meta['sha256']==sha(directory/name),'manifest bytes/digest mismatch')
            for relative,checksum in m['reproducibility_code_sha256'].items():requirements[(relative,checksum)].append(str(mp.relative_to(ROOT)))
            releases.append({'track':m['track'],'release':m['release'],'manifest_sha256':sha(mp),'checksum_inventory_sha256':sha(directory/'CHECKSUMS.sha256'),'checksum_files':len(checks),'canonical_sha256':sha(directory/'canonical.jsonl')})
            input_hashes[str(mp.relative_to(ROOT))]=sha(mp)
        require(len(releases)==16,'unexpected final immutable release inventory')
        report['immutable_release_checks']={'releases':len(releases),'all_checksum_and_manifest_file_coverage_pass':True,'details':releases}
        a=ROOT/'data/RCM-001/releases/v0002/canonical.jsonl';b=ROOT/'data/RCM-001/releases/v0003/canonical.jsonl'
        require(a.read_bytes()==b.read_bytes(),'RCM v2/v3 canonical bytes differ');new_rows=load(b);require(len(new_rows)==1200,'RCM supersession row count')
        own={r['sample_id']:r for r in load(ROOT/'reviews/adversarial/expansion-v1.jsonl') if r['sample_id'].startswith('RCM-001-')}
        candidates={r['sample_id']:r for r in load(ROOT/'data/RCM-001/candidate_renderings/expansion-v1.jsonl')}
        for r in new_rows:
            receipt=own[r['sample_id']];candidate=candidates[r['sample_id']]
            require(receipt['decision']=='ACCEPT' and receipt['record_hash']==r['reviewed_candidate_hash']==candidate['record_hash'],'RCM current row not bound to own original exact ACCEPT')
            require(receipt['receipt_id'] in r['review_receipt_ids'],'own original receipt missing from release')
            projection=dict(r);projection.pop('reviewed_candidate_hash')
            for field in ('record_hash','review_status','review_receipt_ids'):projection[field]=candidate[field]
            require(canon(projection)==canon(candidate),'RCM candidate content changed after review')
        same_receipts=(a.parent/'review_receipts.jsonl').read_bytes()==(b.parent/'review_receipts.jsonl').read_bytes()
        require(same_receipts,'RCM v2/v3 archived three-role receipt payloads differ')
        report['RCM_supersession']={'records':1200,'canonical_byte_equivalent':True,'canonical_sha256':sha(a),'archived_three_role_receipt_bytes_identical':True,'review_receipt_payloads_sha256':sha(a.parent/'review_receipts.jsonl'),'own_original_exact_ACCEPT_bindings':1200,'reviewer_payload_contents_parsed':False}
        selected,expected,families,catalog_inputs=independent_selection(ROOT);actual_rows,actual_catalog=catalog_function(ROOT)();saved_path=ROOT/'manifests/ACTIVE_RELEASE_CATALOG.json';saved=json.loads(saved_path.read_text())
        require(canon(selected)==canon(actual_rows) and canon(expected)==canon(saved['releases'])==canon(actual_catalog['releases']),'active catalog differs from independent selection')
        require(saved['active_records']==len(selected)==2572 and saved['active_source_families']==len(families)==636,'active catalog counts differ')
        rcm_old=next(e for e in expected if (e['track'],e['release'])==('RCM-001','v0002'));rcm_new=next(e for e in expected if (e['track'],e['release'])==('RCM-001','v0003'))
        require(rcm_old['status']=='SUPERSEDED' and rcm_old['active_records']==0 and rcm_new['active_records']==1200,'RCM current supersession selection incorrect')
        report['active_catalog']={'sha256':sha(saved_path),'records':2572,'source_families':636,'independent_selection_matches':True,'RCM_v0002_status':'SUPERSEDED','RCM_v0003_active_records':1200}
        active_manifests={e['path']+'/MANIFEST.json' for e in expected if e['active_records']}
        index_path=ROOT/'reproducibility/SOURCE_ARCHIVE_INDEX.json';index=json.loads(index_path.read_text());indexed={(e['original_path'],e['sha256']):e for e in index['entries']}
        require(set(indexed)==set(requirements),'source index does not cover exact current path/hash requirements')
        missing=[];preserved=0;active_missing=[]
        for pair,refs in requirements.items():
            relative,checksum=pair;entry=indexed[pair];active=bool(set(refs)&active_manifests);require(set(entry['release_manifests'])==set(refs),'source index release references differ')
            require(entry['referenced_by_active_release']==active,'source active status differs')
            blob=ROOT/'reproducibility/source_blobs'/(checksum+'.source')
            if blob.is_file():
                require(sha(blob)==checksum and entry['status']=='PRESERVED_EXACT_BYTES','source blob/index mismatch');preserved+=1
            else:
                require(entry['status']=='MISSING_EXACT_ORIGINAL_BYTES','missing bytes hidden by index')
                item={'original_path':relative,'sha256':checksum,'release_manifests':refs,'active':active};missing.append(item)
                if active:active_missing.append(item)
        require(index['requirements']==len(requirements)==55 and index['preserved_requirements']==preserved==52 and index['missing_requirements']==len(missing)==3,'source archive summary mismatch')
        require(index['missing_active_requirements']==len(active_missing)==0,'active manifest lacks exact source payload')
        require(canon(index['active_catalog_snapshot']['releases'])==canon(expected),'source archive active snapshot stale')
        blob_paths=sorted((ROOT/'reproducibility/source_blobs').glob('*.source'))
        for p in blob_paths:require(p.stem==sha(p),'unreferenced preserved source blob corrupt')
        history_paths=sorted((ROOT/'reproducibility/archive_history').glob('*.json'))
        for p in history_paths:require(p.stem==sha(p),'source index history snapshot mismatch')
        audit_history=[]
        for folder in (ROOT/'reviews/oracle/history',ROOT/'audits/leakage/history'):
            for p in sorted(folder.glob('*')):
                if not p.is_file():continue
                suffix=p.stem.rsplit('-',1)[-1];require(len(suffix)==64 and suffix==sha(p),'machine audit history filename digest mismatch')
                audit_history.append({'path':str(p.relative_to(ROOT)),'sha256':suffix})
        report['source_archive']={'index_sha256':sha(index_path),'path_hash_requirements':55,'preserved_exact_requirements':52,'source_blob_files_validated':len(blob_paths),'missing_active_requirements':0,'historical_missing_requirements':missing,'prior_index_snapshots_validated':len(history_paths),'prior_machine_audit_snapshots_validated':len(audit_history),'prior_machine_audit_snapshots':audit_history,'no_reconstructed_bytes_used_as_preserved_source':True}
        report['archival_shadow_checks']=archival_shadow_checks();report['dedicated_history_immutable_comparison']=historical_git_match()
        report['input_manifest_hashes']=input_hashes;report['current_code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'factory/pipeline.py',ROOT/'factory/common.py',ROOT/'factory/catalog.py',ROOT/'factory/status.py',ROOT/'reproducibility/archive_sources.py',pathlib.Path(__file__))}
        report['decision']='PASS_WITH_DISCLOSED_HISTORICAL_SOURCE_GAPS';report['material_blockers']=[];report['completed']=True
        report['limitations']=['The three original broad source snapshots for historical RCMv0002 remain missing; current archived sources are not represented as recovered historical bytes.','Current RCMv0003 archives release-time source requirements and preserves the identical originally reviewed canonical data/receipts.','Hashing producer/oracle/reviewer payloads verifies custody only; none are imported, executed, or used to approve semantic targets.','Dedicated existing Git history is a local immutable baseline. No new remote backup verification, network, fetch/push, training or publication is performed.']
    except Exception as e:
        report.update(decision='REVISE',completed=False,material_blockers=[{'error_type':type(e).__name__,'evidence':str(e)}])
    out=ROOT/'audits/execution/FINAL_PUBLICATION_INTEGRITY.json'
    content=(json.dumps(report,indent=2)+'\n').encode()
    if out.exists() and out.read_bytes()!=content:
        previous=out.read_bytes();history=out.parent/'history'/(out.stem+'-'+hashlib.sha256(previous).hexdigest()+'.json')
        history.parent.mkdir(exist_ok=True)
        if history.exists():require(history.read_bytes()==previous,'own prior integrity report snapshot mismatch')
        else:history.write_bytes(previous)
    out.write_bytes(content)
    print(canon({k:report.get(k) for k in ('decision','completed','material_blockers','active_catalog','source_archive','archival_shadow_checks')}))
    require(report['completed'],'final publication integrity blocker; inspect saved report')


if __name__=='__main__':main()
