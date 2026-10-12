"""Deterministic raw derivation, incremental qualification, and cold recovery."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.pipelines import render_controls as controls
from goal_conditioned_decision_data.pipelines import rendered
from goal_conditioned_decision_data.pipelines.common import canonical, digest

class Remote:
    def __init__(self,root,seconds=180):
        self.deadline=time.monotonic()+seconds;self.raw={};self.refs={};self.calls=[]
    def fetch_ref(self,ref):self.calls.append(('ref',ref));return self.refs[ref]
    def fetch_tree(self,commit):
        self.calls.append(('tree',commit))
        return {archive.PROGRAMME+'/'+p:{'type':'blob','mode':'100644','sha':archive.git_blob_sha1(r),'size':len(r)} for (c,p),r in self.raw.items() if c==commit}
    def fetch_blob(self,oid,max_bytes):
        self.calls.append(('blob',oid))
        values=[r for r in self.raw.values() if archive.git_blob_sha1(r)==oid]
        if not values:raise archive.ArchiveError('missing remote blob')
        if len(values[0])>max_bytes:raise archive.ArchiveError('unbounded blob')
        return values[0]
    def store(self,root,path,commit):
        self.raw[commit,path]=(root/path).read_bytes();self.refs[archive.archive_ref(commit)]=commit

class RenderControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=(Path(self.temp.name)/archive.PROGRAMME).resolve();self.root.mkdir()
        self.index={'files':{},'catalogs':{},'receipts':{}};self.cid='a'*64;self.commit='b'*40
        self.index['catalogs'][self.cid]={'catalog_commit':self.commit,'archive_ref':archive.archive_ref(self.commit),'catalog_path':archive.catalog_path(self.cid),'catalog_sha256':'c'*64,'catalog_git_blob_sha1':'d'*40}
        p=patch.object(archive,'load_index',side_effect=lambda root:self.index);p.start();self.addCleanup(p.stop)
        self.remote=Remote(self.root);self.remote.refs[archive.archive_ref(self.commit)]=self.commit
        p=patch.object(archive,'GitHubTransport',return_value=self.remote);p.start();self.addCleanup(p.stop)
        self.receipt=self.fixture(1)
    def write(self,path,value,archived=False):
        raw=(canonical(value)+'\n').encode() if not isinstance(value,bytes) else value
        full=self.root/path;full.parent.mkdir(parents=True,exist_ok=True);full.write_bytes(raw)
        if archived:self.index['files'][path]={'commit':self.commit,'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':archive.git_blob_sha1(raw),'bytes':len(raw),'catalog_id':self.cid,'receipt_id':'e'*64}
        return hashlib.sha256(raw).hexdigest()
    def fixture(self,seed,zero=False):
        code={'code.py':'1'*64};protected=[['PRIVATE ID','PRIVATE FINGERPRINT','PRIVATE PARENT','PRIVATE TEMPLATE','dev']]
        source=f'data/accepted/dev-{seed}.jsonl';sha=self.write(source,b'opaque source',True)
        batch={'split':'dev','seed':seed,'shard':source,'bytes':13,'sha256':sha,'accepted':1,'audit':{'verdict':'PASS'},'code_hashes':code,'protected_index':protected,'profile':'historical-profile'}
        batchpath=f'manifests/batch-dev-{seed}.json';batchsha=self.write(batchpath,batch)
        dataset='gcdl_rendered_'+digest({'seed':seed})[:32];mp=f'data/rendered/{dataset}/manifest.json';sp=f'data/rendered/{dataset}/part-00000.jsonl';shards=[]
        if not zero:
            raw=b'{"input_text":"PRIVATE PROMPT","targets":"PRIVATE TARGET"}\n';sha2=self.write(sp,raw,True);shards=[{'path':sp,'sha256':sha2,'bytes':len(raw),'rows':1}]
        view='computation' if zero else 'listwise'
        manifest={'manifest_path':mp,'dataset_id':dataset,'view':view,'split':'dev','source_records_examined':1,'variant_budget':1,'export_version':'export-fixture','rendering_version':'render-fixture','training_launched':False,'code_hashes':code,'rendered_rows':0 if zero else 1,'eligible_source_records':0 if zero else 1,'sources':[{'path':source,'sha256':sha,'source_batch_manifest':batchpath,'source_batch_manifest_sha256':batchsha,'source_code_hashes':code,'protected_index':protected}],'shards':shards,'selected_records':['PRIVATE RECORD']}
        mh=self.write(mp,manifest,True)
        spec={'views':[view],'variant_budget':1,'export_version':'export-fixture','rendering_version':'render-fixture','code_sha256':digest(code)}
        receipt={'version':'gcdl-rendered-batch-1','seed':seed,'split':'dev','source_manifest':batchpath,'source_sha256':batchsha,'source_shard_sha256':sha,'specification':spec,'spec_sha256':digest(spec),'code_hashes':code,'training_launched':False,'datasets':[{'dataset_id':dataset,'view':view,'manifest_path':mp,'manifest_sha256':mh,'rendered_rows':manifest['rendered_rows'],'eligible_source_records':manifest['eligible_source_records']}]}
        relative=f'manifests/render-batch-dev-{seed}-{digest(spec)[:16]}.json';self.write(relative,receipt);return relative
    def plan(self,paths):return controls.plan_bundle(self.root,controls.prepare(self.root,paths))
    def qualify(self,paths,commit='f'*40):
        plan=self.plan(paths);self.remote.store(self.root,plan['bundle_path'],commit)
        return controls.activate(self.root,plan['bundle_path'],commit,transport=self.remote)
    def test_untrusted_stage_falls_back_and_contains_no_raw_records(self):
        paths=controls.prepare(self.root,[self.receipt]);raw=(self.root/paths[0]).read_bytes()
        for secret in (b'PRIVATE PROMPT',b'PRIVATE TARGET',b'PRIVATE RECORD',b'PRIVATE ID',b'PRIVATE PARENT'):self.assertNotIn(secret,raw)
        with patch.object(archive,'read_text',wraps=archive.read_text) as reads:
            self.assertEqual(rendered.validate_receipt(self.root,self.receipt)[1],{'listwise':1})
            self.assertTrue(any(c.args[1].startswith('data/rendered/') for c in reads.call_args_list))
        self.assertFalse((self.root/controls.POINTER).exists())
    def test_cold_recovery_fetches_only_one_control_blob_each_session(self):
        pointer=self.qualify([self.receipt]);self.remote.calls.clear()
        mp=json.loads((self.root/self.receipt).read_text())['datasets'][0]['manifest_path'];(self.root/mp).unlink()
        for p in self.index['files']:
            if p.endswith('.jsonl'):(self.root/p).unlink()
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        with archive.read_session(self.root):
            rendered.recover(db,self.root);stats=rendered.inventory(self.root);rendered.recover(db,self.root)
        self.assertEqual(stats['rendered_rows_by_split_view'],{'dev/listwise':1})
        self.assertEqual(db.execute('SELECT status FROM rendered_jobs').fetchone()[0],'done')
        self.assertEqual(sum(c[0]=='ref' for c in self.remote.calls),4)
        self.assertEqual([c[1] for c in self.remote.calls if c[0]=='blob'],[pointer['git_blob_sha1']])
    def test_self_rehashed_stage_is_rederived_and_rejected(self):
        plan=self.plan([self.receipt]);bundle=json.loads((self.root/plan['bundle_path']).read_text());capsule=bundle['capsules'][self.receipt]
        capsule['datasets'][0]['fields']['rendered_rows']=999;capsule['capsule_id']=digest({k:v for k,v in capsule.items() if k!='capsule_id'});bundle['bundle_id']=digest({k:v for k,v in bundle.items() if k!='bundle_id'})
        forged=controls._bundle_path(bundle['bundle_id']);self.write(forged,bundle);self.remote.store(self.root,forged,'f'*40)
        with self.assertRaisesRegex(archive.ArchiveError,'full raw derivation'):controls.activate(self.root,forged,'f'*40,transport=self.remote)
        self.assertFalse((self.root/controls.POINTER).exists())
    def test_incremental_qualification_rederives_only_new_and_zero_rows_survive(self):
        self.qualify([self.receipt]);other=self.fixture(2,True);mp=json.loads((self.root/self.receipt).read_text())['datasets'][0]['manifest_path'];(self.root/mp).unlink()
        with patch.object(controls,'derive',wraps=controls.derive) as derives:self.qualify([other],'9'*40)
        self.assertEqual([c.args[1] for c in derives.call_args_list],[other,other])
        with archive.read_session(self.root):
            self.assertEqual(rendered.validate_receipt(self.root,other)[1],{'computation':0});self.assertEqual(rendered.validate_receipt(self.root,self.receipt)[1],{'listwise':1})
        self.assertEqual(controls.pending(self.root),[])
    def test_unrelated_append_allowed_relevant_custody_change_rejected(self):
        self.qualify([self.receipt]);self.write('reports/history/unrelated.json',b'opaque evidence',True)
        self.assertEqual(rendered.validate_receipt(self.root,self.receipt)[1],{'listwise':1})
        mp=json.loads((self.root/self.receipt).read_text())['datasets'][0]['manifest_path'];self.index['files'][mp]['commit']='7'*40
        with self.assertRaisesRegex(archive.ArchiveError,'custody binding'):rendered.validate_receipt(self.root,self.receipt)
    def test_original_snapshot_ref_moved_refuses_even_with_good_control_snapshot(self):
        self.qualify([self.receipt]);self.remote.refs[archive.archive_ref(self.commit)]='7'*40
        with self.assertRaisesRegex(archive.ArchiveError,'snapshot moved'):rendered.validate_receipt(self.root,self.receipt)
    def test_changed_source_profile_refuses(self):
        self.qualify([self.receipt]);bp='manifests/batch-dev-1.json';batch=json.loads((self.root/bp).read_text());batch['profile']='changed';self.write(bp,batch)
        with self.assertRaisesRegex(ValueError,'canonical source'):rendered.validate_receipt(self.root,self.receipt)
    def test_missing_remote_blob_never_installs_authority(self):
        plan=self.plan([self.receipt]);self.remote.store(self.root,plan['bundle_path'],'f'*40);self.remote.raw.clear()
        with self.assertRaises(archive.ArchiveError):controls.activate(self.root,plan['bundle_path'],'f'*40,transport=self.remote)
        self.assertFalse((self.root/controls.POINTER).exists())
    def test_deadline_stop_and_storage_bound_preserve_raw(self):
        with self.assertRaises(archive.ReadDeadlineExceeded):controls.prepare(self.root,[self.receipt],deadline=time.monotonic()-1)
        self.write('state/STOP',b'stop')
        with self.assertRaisesRegex(archive.ArchiveError,'STOP'):controls.prepare(self.root,[self.receipt])
        (self.root/'state/STOP').unlink()
        with patch.object(controls,'MAX_CONTROL_STORAGE_BYTES',1):
            with self.assertRaisesRegex(archive.ArchiveError,'storage bound'):controls.prepare(self.root,[self.receipt])
        self.assertFalse((self.root/controls.POINTER).exists())
    def test_derivation_bypasses_qualified_projection(self):
        self.qualify([self.receipt])
        with patch.object(controls,'try_projection',side_effect=AssertionError('circular derivation')):controls.derive(self.root,self.receipt)
    def test_repeated_publication_uses_two_slots_and_prior_authority_survives_staging(self):
        authority=self.qualify([self.receipt]);prior=(self.root/authority['bundle_path']).read_bytes()
        other=self.fixture(2);plan=self.plan([other])
        self.assertNotEqual(plan['bundle_path'],authority['bundle_path'])
        self.assertEqual((self.root/authority['bundle_path']).read_bytes(),prior)
        self.assertEqual(rendered.validate_receipt(self.root,self.receipt)[1],{'listwise':1})
        for seed in range(2,7):
            path=other if seed==2 else self.fixture(seed)
            self.qualify([path],str(seed)*40)
        slots=list((self.root/controls.PREFIX).glob('bundle-slot-*.json'))
        self.assertEqual(len(slots),2)
        with archive.read_session(self.root):
            self.assertEqual(rendered.inventory(self.root)['completed_source_group_exports'],6)

    def test_free_space_reserve_retains_raw_and_no_authority(self):
        disk=type('Disk',(),{'free':controls.METADATA_RESERVE_BYTES-1})()
        with patch.object(controls.shutil,'disk_usage',return_value=disk):
            with self.assertRaisesRegex(archive.ArchiveError,'metadata reserve'):
                controls.prepare(self.root,[self.receipt])
        self.assertFalse((self.root/controls.POINTER).exists())
        self.assertTrue((self.root/'data/accepted/dev-1.jsonl').exists())

    def test_foreign_or_sealed_staging_paths_refused_before_any_body_read(self):
        sealed='data/sealed/sealed-99.jsonl';self.write(sealed,b'OPAQUE SEALED GOLD')
        with patch.object(controls,'_read_local',side_effect=AssertionError('must not read body')):
            with self.assertRaises(archive.ArchiveError):controls.plan_bundle(self.root,[sealed])
            with self.assertRaises(archive.ArchiveError):controls.activate(self.root,sealed,'f'*40)
        self.assertEqual((self.root/sealed).read_bytes(),b'OPAQUE SEALED GOLD')

    def test_prepare_bound_and_partial_warm_group_skip(self):
        with self.assertRaisesRegex(archive.ArchiveError,'preparation bound'):controls.prepare(self.root,[self.receipt],max_receipts=5)
        sp=next(p for p in self.index['files'] if p.startswith('data/rendered/') and p.endswith('.jsonl'));del self.index['files'][sp]
        self.assertEqual(controls.pending(self.root),[])

if __name__=='__main__':unittest.main()
