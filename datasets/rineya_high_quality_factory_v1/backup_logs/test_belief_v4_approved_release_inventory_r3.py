#!/usr/bin/env python3
"""Actual isolated corruption tests for NEW approved-release inventory derivative."""
import copy,gzip,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
sys.dont_write_bytecode=True
LOGS=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('release_inventory_r1',LOGS/'prepare_belief_v4_approved_release_inventory_r3.py')
b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
class BoundInventorySafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='approved-release-inventory-tests-',dir=LOGS);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def actual(self):
        cfg=json.loads(b.CONFIG.read_bytes());prior=json.loads((LOGS/'belief-v4-validated-code-candidate-r2_SPEC.json').read_bytes())
        pack=json.loads((LOGS/'BELIEF_V4_APPROVED_RELEASE_LOSSLESS_RECEIPT_ATTEMPT01.json').read_bytes())
        return cfg,prior,pack
    def fixture_entries(self):
        cfg,prior,pack=self.actual();entries={}
        rep=cfg['old_representation'];entries[rep['manifest']['path']]=copy.deepcopy(rep['manifest']);entries[rep['gzip']['path']]=copy.deepcopy(rep['gzip'])
        for p in [cfg['current_pack_record'],*cfg['archive_pack_records']]:
            for role in ('raw','gzip','manifest'):
                e=b.record_entry(p,role);entries[e['path']]=e
        md=json.loads((b.FACTORY/rep['manifest']['path']).read_bytes())
        return cfg,entries,b.normalized_record(md),b.record_entry(cfg['current_pack_record'],'raw')
    def check(self,cfg,entries,rec,raw):
        fake=Mock(return_value=True)
        out=b.verify_historical_mismatch(cfg['old_representation']['manifest']['path'],rec,raw,entries,cfg,compare=fake)
        return out,fake
    def test_exact_prior_config_and_both_bound_historical_decode_calls(self):
        cfg,prior,pack=self.actual();b.validate_historical_config(cfg,prior,pack)
        cfg,entries,rec,raw=self.fixture_entries();out,fake=self.check(cfg,entries,rec,raw)
        self.assertEqual(fake.call_count,2)
        self.assertEqual(len(out['archived_old_gzip_decode_proofs']),2)
        self.assertTrue(out['old_manifest_is_historical_not_current_raw_mapping'])
        self.assertEqual(out['current_mutable_raw']['sha256'],b.runner().CURRENT_AUDIT_SHA)
    def test_config_hash_path_size_type_extra_rule_or_archive_binding_changes_rejected(self):
        cfg,prior,pack=self.actual()
        for mutate in (lambda c:c.update(exception_count=2),lambda c:c.update(exception_count=True),
                       lambda c:c['old_raw'].update(sha256='0'*64),lambda c:c['old_representation']['gzip'].update(path='other.gz'),
                       lambda c:c['current_pack_record']['raw'].update(sha256='0'*64),lambda c:c['archive_pack_records'].pop(),
                       lambda c:c.update(grants_data_approval=True),lambda c:c.update(extra_stale_exception=True)):
            altered=copy.deepcopy(cfg);mutate(altered)
            with self.assertRaises(ValueError):b.validate_historical_config(altered,prior,pack)
    def test_missing_or_changed_archived279_raw_rejected(self):
        cfg,entries,rec,raw=self.fixture_entries()
        for arch in cfg['archive_pack_records']:
            name=arch['raw_path'];missing=copy.deepcopy(entries);missing.pop(name)
            with self.assertRaises(ValueError):self.check(cfg,missing,rec,raw)
            changed=copy.deepcopy(entries);changed[name]['sha256']='0'*64
            with self.assertRaises(ValueError):self.check(cfg,changed,rec,raw)
    def test_wrong_old_manifest_gzip_hash_size_or_path_rejected(self):
        cfg,entries,rec,raw=self.fixture_entries()
        for role in ('manifest','gzip'):
            for key,value in (('sha256','0'*64),('size',0)):
                altered=copy.deepcopy(entries);altered[cfg['old_representation'][role]['path']][key]=value
                with self.assertRaises(ValueError):self.check(cfg,altered,rec,raw)
        for key,value in (('raw_sha256','0'*64),('raw_bytes',123141332),('gzip_sha256','0'*64),('gzip_bytes',1),('gzip_path',cfg['current_pack_record']['gzip_path'])):
            changed=copy.deepcopy(rec);changed[key]=value
            with self.assertRaises(ValueError):self.check(cfg,entries,changed,raw)
    def test_current149_raw_and_new_standard_manifest_or_gzip_change_rejected(self):
        cfg,entries,rec,raw=self.fixture_entries()
        altered=copy.deepcopy(raw);altered['sha256']=cfg['old_raw']['sha256']
        with self.assertRaises(ValueError):self.check(cfg,entries,rec,altered)
        for role in ('manifest','gzip'):
            name=cfg['current_pack_record'][role+'_path'];changed=copy.deepcopy(entries);changed[name]['sha256']='0'*64
            with self.assertRaises(ValueError):self.check(cfg,changed,rec,raw)
    def test_any_other_stale_manifest_or_raw_path_rejected(self):
        cfg,entries,rec,raw=self.fixture_entries()
        with self.assertRaises(ValueError):b.verify_historical_mismatch('unbound.lossless.json',rec,raw,entries,cfg)
        raw=copy.deepcopy(raw);raw['path']='audits/diversity/unrelated.json'
        with self.assertRaises(ValueError):self.check(cfg,entries,rec,raw)
    def test_actual_streaming_gzip_bytes_then_corruption_rejected(self):
        raw=self.root/'raw';gz=self.root/'raw.gz';payload=b'ordinary bounded test evidence\n'*12000
        raw.write_bytes(payload);gz.write_bytes(gzip.compress(payload,mtime=0));self.assertTrue(b.compare_gzip(raw,gz))
        raw.write_bytes(payload[:-1]+b'X')
        with self.assertRaises(ValueError):b.compare_gzip(raw,gz)
        raw.write_bytes(payload);gz.write_bytes(gzip.compress(payload+b'extra',mtime=0))
        with self.assertRaises(ValueError):b.compare_gzip(raw,gz)
    def test_actual_all2685_required_and_prior_deferral_rejected(self):
        _,prior,_=self.actual();names={e['path'] for e in prior['files']}
        self.assertEqual(len(b.validate_prior_scope(prior,names,set())),2685)
        one=next(iter(names))
        with self.assertRaises(ValueError):b.validate_prior_scope(prior,names-{one},set())
        with self.assertRaises(ValueError):b.validate_prior_scope(prior,names,{one})
    def test_observed_pending_only_completed_prefix_included_absent_not_invented(self):
        prior={'files':[]};paths=[]
        for prefix in b.PENDING_PREFIXES:
            p=self.root/prefix/'pending.py';p.parent.mkdir(parents=True);p.write_bytes(b'own pending fixture\n');paths.append(p)
        status={p:'RUNNING' for p in b.PENDING_PREFIXES}
        with patch.object(b,'FACTORY',self.root):
            self.assertEqual(len(b.observe_deferred(paths,status,prior)),len(b.PENDING_PREFIXES))
            self.assertEqual(b.observe_deferred([],status,prior),[])
            status={p:'FINAL_COMPLETE' for p in b.PENDING_PREFIXES}
            self.assertEqual(b.observe_deferred(paths,status,prior),[])
            status={p:'RUNNING' for p in b.PENDING_PREFIXES}
            with self.assertRaises(ValueError):b.observe_deferred(paths,status,{'files':[{'path':paths[0].relative_to(self.root).as_posix()}]})
    def test_alias_normalization_does_not_expand_archive_star_schema(self):
        rec={'raw_path':'raw','raw_sha256':'a'*64,'raw_length':12,'compressed_path':'raw.gz','compressed_sha256':'b'*64,'compressed_length':13}
        out=b.normalized_record(rec);self.assertEqual(out['raw_bytes'],12);self.assertEqual(out['gzip_path'],'raw.gz')
        with self.assertRaises(ValueError):b.normalized_record({'archive_path':'raw.gz'})
    def test_snapshot_stream_secret_scan_hash_and_change_are_real(self):
        p=self.root/'safe.txt';data=b'actual ordinary bytes\n';p.write_bytes(data)
        with patch.object(b,'FACTORY',self.root):
            entry,blob=b.snapshot(p);self.assertEqual(entry['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(blob,hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest())
            p.write_bytes(b'gh'+b'p_'+b'Q'*35)
            with self.assertRaises(ValueError):b.snapshot(p)
    def test_release_union_includes_oversized_and_rejects_overlap(self):
        r=b.runner();one={'path':'data/RBM-001/releases/v0002/canonical.jsonl','size':r.MAX_BYTES+1,'sha256':'a'*64}
        self.assertEqual(len(r.release_inventory({'files':[],'raw_size_exclusions':[one]})),1)
        with self.assertRaises(ValueError):r.release_inventory({'files':[one],'raw_size_exclusions':[one]})
if __name__=='__main__':unittest.main(verbosity=2)
