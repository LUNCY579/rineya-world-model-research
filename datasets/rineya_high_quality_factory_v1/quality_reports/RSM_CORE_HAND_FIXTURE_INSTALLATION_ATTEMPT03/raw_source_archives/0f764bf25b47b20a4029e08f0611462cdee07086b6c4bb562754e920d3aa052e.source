import copy
import unittest
from factory.pipeline import APPROVAL_ACTORS, approval_evidence


class ReviewGate(unittest.TestCase):
    def setUp(self):
        self.row={"sample_id":"sample","record_hash":"a"}
        self.e={role:{"sample":[{"record_hash":"a","decision":"ACCEPT","reviewer_execution_id":actor,"critical_defects":[],"issues":[]}]} for role,actor in APPROVAL_ACTORS.items()}

    def test_complete_independent_evidence(self):self.assertEqual(approval_evidence(self.row,self.e)[1],[])

    def test_missing_or_stale_review_fails_closed(self):
        self.e["expert"]["sample"][0]["record_hash"]="stale"
        self.assertIn("expert",approval_evidence(self.row,self.e)[1])

    def test_later_rejection_overrides_earlier_acceptance(self):
        reject=copy.deepcopy(self.e["expert"]["sample"][0]);reject["decision"]="REVISE"
        self.e["expert"]["sample"].append(reject)
        self.assertIn("expert",approval_evidence(self.row,self.e)[1])

    def test_self_approval_fails_closed(self):
        self.e["expert"]["sample"][0]["reviewer_execution_id"]="/root"
        self.assertIn("nonindependent_approval",approval_evidence(self.row,self.e)[1])

    def test_same_execution_multiple_roles_fails(self):
        self.e["expert"]["sample"][0]["reviewer_execution_id"]="/root/adversarial"
        self.assertIn("nonindependent_approval",approval_evidence(self.row,self.e)[1])

    def test_high_issue_cannot_be_compensated_by_accept(self):
        self.e["expert"]["sample"][0]["issues"]=[{"severity":"high"}]
        self.assertIn("unresolved_serious_issue",approval_evidence(self.row,self.e)[1])


if __name__=="__main__":unittest.main()
