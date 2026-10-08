import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('resource_validator',ROOT/'tools/validate_resources.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def example():return json.loads((ROOT/'examples/trajectory.json').read_text())


def test_bundled_schema_example():module.validate_record(example())


def test_temporal_crossfield_rejection():
    r=example();r['actual_outcomes'][0]['end']=-1
    with pytest.raises(ValueError):module.validate_record(r)


def test_feature_mask_crossfield_rejection():
    r=example();r['actual_outcomes'][0]['feature_valid']=[True]
    with pytest.raises(ValueError):module.validate_record(r)


def test_duplicated_action_id_rejection():
    r=example();r['actions'][1]['action_id']=r['actions'][0]['action_id']
    with pytest.raises(ValueError):module.validate_record(r)


def test_future_dependency_rejection():
    r=example();r['actions'][0]['parents']=['a2']
    with pytest.raises(ValueError):module.validate_record(r)
