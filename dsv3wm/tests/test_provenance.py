import json,tempfile,unittest
from pathlib import Path
from dsv3wm.native import sha256,EXPECTED_CORE_SHA
from dsv3wm.config import Config
from dsv3wm.provenance import verify_bound_inputs,validate_training_lineage


class ProvenanceTests(unittest.TestCase):
    def test_selected_input_must_be_present_in_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);allowed=p/'allowed';allowed.write_text('known');other=p/'other';other.write_text('unknown')
            freeze=p/'freeze.json';freeze.write_text(json.dumps({'files':{str(allowed):sha256(allowed)}}))
            verify_bound_inputs(freeze,[allowed])
            with self.assertRaises(ValueError):verify_bound_inputs(freeze,[other])
    def test_selected_input_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);data=p/'data';data.write_text('known');freeze=p/'freeze.json'
            freeze.write_text(json.dumps({'files':{str(data):sha256(data)}}));data.write_text('changed')
            with self.assertRaises(ValueError):verify_bound_inputs(freeze,[data])
    def fixture(self,p):
        dataset=p/'dataset';dataset.mkdir();(dataset/'train_register.jsonl').write_text('{}\n')
        run=p/'run';(run/'adapter').mkdir(parents=True)
        config=p/'config.json';config.write_text(json.dumps(Config(arm='A0').dict()))
        model=run/'adapter/adapter_model.safetensors';model.write_bytes(b'UNIT-TEST-NOT-MODEL-WEIGHTS')
        adapter=run/'adapter/adapter_config.json';adapter.write_text(json.dumps({'r':4,'lora_alpha':8,'lora_dropout':0.0,'target_modules':['q_proj','v_proj']}))
        metadata={'status':'pilot-trained','config':Config(arm='A0').dict(),'command':['train','--config',str(config)],
                  'config_sha256':sha256(config),'data_sha256':sha256(dataset/'train_register.jsonl'),
                  'adapter_sha256':sha256(model),'core_sha256':EXPECTED_CORE_SHA}
        (run/'run.json').write_text(json.dumps(metadata));return run,dataset,config,adapter
    def test_training_lineage_matches_selected_content(self):
        with tempfile.TemporaryDirectory() as folder:
            run,data,config,adapter=self.fixture(Path(folder))
            self.assertEqual(validate_training_lineage(run,data,'A0'),(config,adapter))
            (data/'train_register.jsonl').write_text('different corpus')
            with self.assertRaises(ValueError):validate_training_lineage(run,data,'A0')
    def test_adapter_scaling_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run,data,config,adapter=self.fixture(Path(folder))
            changed=json.loads(adapter.read_text());changed['lora_alpha']=100;adapter.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):validate_training_lineage(run,data,'A0')


if __name__=='__main__':unittest.main()
