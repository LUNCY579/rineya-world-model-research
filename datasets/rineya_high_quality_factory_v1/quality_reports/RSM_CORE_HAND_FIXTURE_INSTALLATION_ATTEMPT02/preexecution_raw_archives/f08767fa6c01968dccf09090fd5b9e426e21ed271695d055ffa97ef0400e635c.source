import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from factory.common import write_json,write_jsonl
from factory.catalog import catalog


class ActiveCatalog(unittest.TestCase):
    def test_erratum_then_supersession_does_not_duplicate_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old={'track_id':'RVM-001','generation_seed':20261009,'semantic_parent_id':'RVM-001-old-p0000','record_hash':'old','sample_id':'old','split':'train'}
            new={**old,'semantic_parent_id':'RVM-001-new-p0000','record_hash':'new','sample_id':'new'}
            first=root/'data/RVM-001/releases/v0001';write_json(first/'MANIFEST.json',{'track':'RVM-001','release':'v0001'});write_jsonl(first/'canonical.jsonl',[old])
            write_json(root/'quality_reports/errata/old.json',{'excluded_records':[{'record_hash':'old'}]})
            with patch('factory.catalog.ROOT',root):
                rows,report=catalog();self.assertEqual(rows,[]);self.assertEqual(report['active_records'],0)
                second=root/'data/RVM-001/releases/v0002';write_json(second/'MANIFEST.json',{'track':'RVM-001','release':'v0002','supersedes':['v0001']});write_jsonl(second/'canonical.jsonl',[new])
                rows,report=catalog();self.assertEqual(rows,[new]);self.assertEqual(report['active_source_families'],1)
                self.assertEqual(report['releases'][0]['status'],'SUPERSEDED')


if __name__=='__main__':unittest.main()
