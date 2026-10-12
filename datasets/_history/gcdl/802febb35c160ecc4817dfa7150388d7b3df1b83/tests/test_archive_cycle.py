"""Real Git and anchored-file integration for the bounded archival coordinator."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.scripts import archive_cycle as cycle

class GitRemote:
    def __init__(self,git):self.git=git;self.calls=[]
    def fetch_ref(self,reference):
        self.calls.append(('ref',reference))
        return self.git('rev-parse',reference).strip()
    def fetch_tree(self,commit):
        result={}
        for row in self.git('ls-tree','-r','-l',commit).splitlines():
            prefix,path=row.split('\t');mode,kind,oid,size=prefix.split()
            result[path]={'mode':mode,'type':kind,'sha':oid,'size':int(size)}
        return result
    def fetch_blob(self,oid,max_bytes):
        raw=subprocess.run(['git','cat-file','blob',oid],cwd=self.repo,check=True,capture_output=True).stdout
        if len(raw)>max_bytes:raise AssertionError('fixture read exceeded caller bound')
        return raw

class ArchiveCycleTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.repo=Path(temporary.name).resolve();self.root=self.repo/archive.PROGRAMME
        for directory in ('state','configs','manifests','data/accepted'):
            (self.root/directory).mkdir(parents=True,exist_ok=True)
        self.payload='data/accepted/train-99.jsonl'
        (self.root/self.payload).write_bytes(b'immutable fixture payload\n')
        (self.root/'.gitignore').write_text('state/*\n')
        self.config={'archive_enabled':True,'archive_authorization':archive.AUTHORIZATION,
                     'archive_trigger_free_bytes':10*1024**3,'min_free_bytes':10*1024**3}
        (self.root/'configs/production.json').write_text(json.dumps(self.config))
        (self.root/'state/checkpoint.json').write_text('{"seed":99}')
        self.pause=b'{"reason":"resource gate: fewer than 10 GiB free"}'
        (self.root/'state/PAUSED.json').write_bytes(self.pause)
        self.git('init','-q');self.git('config','user.name','fixture')
        self.git('config','user.email','fixture@example.invalid')
        self.git('switch','-c',cycle.BRANCH)
        self.git('remote','add','origin',cycle.compact_git.REMOTE)
        self.git('add',archive.PROGRAMME);self.git('commit','-qm','initial fixture')
        self.remote=GitRemote(self.git);self.remote.repo=self.repo
        self.order=[];self.patches=[
            patch.object(cycle,'current_independent_review',return_value=True),
            patch.object(cycle.shutil,'disk_usage',return_value=type('Disk',(),{'free':9*1024**3})()),
            patch.object(cycle,'_backup',side_effect=self.backup),
            patch.object(cycle,'_publish_snapshot',side_effect=self.publish)]
        for item in self.patches:item.start();self.addCleanup(item.stop)
        archive._CACHE.clear();archive._CACHE_SIZE=0;self.addCleanup(archive._CACHE.clear)
    def git(self,*arguments):
        return subprocess.run(['git',*arguments],cwd=self.repo,check=True,capture_output=True,text=True).stdout
    def backup(self,root,deadline):
        self.order.append('backup')
        self.git('add','--',archive.PROGRAMME)
        result=subprocess.run(['git','diff','--cached','--quiet'],cwd=self.repo)
        if result.returncode:self.git('commit','-qm','fixture backup')
        head=self.git('rev-parse','HEAD').strip()
        (root/'state/backup.json').write_text(json.dumps({'status':'confirmed','commit':head,'remote_commit':head}))
        return head
    def publish(self,root,commit,deadline,transport):
        self.order.append('publish')
        self.git('branch',archive.archive_ref(commit).removeprefix('refs/heads/'),commit)
        return archive.archive_ref(commit)
    def run_cycle(self,**kwargs):
        return cycle.run(self.root,self.config,automatic=True,transport=self.remote,compact=False,**kwargs)
    def test_default_lock_full_custody_before_unlink_and_remote_tree_survives(self):
        original=(self.root/self.payload).read_bytes()
        result=self.run_cycle()
        self.assertEqual(result['deleted_files'],2) # payload and consistent runtime snapshot
        self.assertFalse((self.root/self.payload).exists())
        self.assertEqual(self.order,['backup','backup','publish','backup','backup'])
        self.assertEqual((self.root/'state/PAUSED.json').read_bytes(),self.pause)
        self.assertEqual(self.git('status','--porcelain').strip(),'')
        with self.assertRaises(subprocess.CalledProcessError):self.git('rev-parse','HEAD:'+archive.PROGRAMME+'/'+self.payload)
        self.assertEqual(archive.read_bytes(self.root,self.payload,transport=self.remote),original)
        catalog_commit=result['archive_ref'].rsplit('/',1)[1]
        self.assertIn(archive.PROGRAMME+'/'+self.payload,self.remote.fetch_tree(catalog_commit))
        self.assertTrue((self.root/'state/checkpoint.json').exists())
        self.assertFalse(list(self.root.rglob('.archive-holding*')))
    def test_repeat_with_no_new_payload_does_not_create_another_snapshot(self):
        self.run_cycle();self.order.clear()
        result=self.run_cycle()
        self.assertEqual(result['status'],'no_local_payloads')
        self.assertEqual(self.order,[])
    def test_development_duplicate_verified_and_evicted_with_manifest_pointer(self):
        raw=b'fixture development archive'
        (self.root/'state/development_archive.tar.gz').write_bytes(raw)
        (self.root/'manifests/DEVELOPMENT_ARCHIVE.json').write_text(json.dumps({
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'archive':'state/development_archive.tar.gz'}))
        result=self.run_cycle()
        self.assertFalse((self.root/'state/development_archive.tar.gz').exists())
        self.assertFalse((self.root/'data/backups/development_archive.tar.gz').exists())
        record=json.loads((self.root/'manifests/DEVELOPMENT_ARCHIVE.json').read_text())
        self.assertTrue(record['local_payload_evicted'])
        self.assertEqual(archive.read_bytes(self.root,record['archive'],transport=self.remote),raw)
        self.assertGreater(result['deleted_bytes'],len(raw))
    def test_initial_backup_failure_preserves_payload_and_original_pause(self):
        with patch.object(cycle,'_backup',side_effect=RuntimeError('fixture push failure')):
            with self.assertRaisesRegex(RuntimeError,'push failure'):self.run_cycle()
        self.assertTrue((self.root/self.payload).exists())
        self.assertEqual((self.root/'state/PAUSED.json').read_bytes(),self.pause)
        self.assertEqual(archive.load_index(self.root)['files'],{})
    def test_custody_backup_failure_does_not_unlink_verified_payload(self):
        calls=0;original=self.backup
        def failing(*args):
            nonlocal calls
            calls+=1
            if calls==3:raise RuntimeError('fixture custody push failure')
            return original(*args)
        with patch.object(cycle,'_backup',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'custody push failure'):self.run_cycle()
        self.assertTrue((self.root/self.payload).exists())
        self.assertIn(self.payload,archive.load_index(self.root)['files'])
        # A later invocation completes the already verified cohort.
        self.run_cycle()
        self.assertFalse((self.root/self.payload).exists())
    def test_source_review_change_after_push_refuses_deletion(self):
        with patch.object(cycle,'current_independent_review',side_effect=[True,False]):
            with self.assertRaisesRegex(RuntimeError,'review changed'):self.run_cycle()
        self.assertTrue((self.root/self.payload).exists())
    def test_stop_or_quality_pause_performs_no_backup(self):
        (self.root/'state/STOP').touch();self.assertIsNone(self.run_cycle())
        (self.root/'state/STOP').unlink()
        (self.root/'state/PAUSED.json').write_text('{"reason":"oracle disagreement"}')
        self.assertIsNone(self.run_cycle());self.assertEqual(self.order,[])
    def test_operator_pause_change_survives_failed_maintenance(self):
        (self.root/'state/PAUSED.json').unlink()
        def changed(*args):
            (self.root/'state/PAUSED.json').write_text('{"reason":"operator pause"}')
            raise RuntimeError('changed')
        with patch.object(cycle,'_backup',side_effect=changed):
            with self.assertRaisesRegex(RuntimeError,'changed'):self.run_cycle()
        self.assertEqual(json.loads((self.root/'state/PAUSED.json').read_text())['reason'],'operator pause')
        self.assertTrue((self.root/self.payload).exists())

if __name__=='__main__':unittest.main()
