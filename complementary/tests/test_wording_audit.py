import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcx.rcl import templates as T  # noqa: E402
from rcx.rcl import wording_audit as WA  # noqa: E402


class TestWordingAudit(unittest.TestCase):
    def test_words_drop_slot(self):
        self.assertEqual(WA.words("Only {L} events count."), ["only", "events", "count"])

    def test_reference_copies_are_flagged(self):
        for pool in (T.SPOS, T.SNEG):
            for text in pool.values():
                self.assertTrue(WA.ref_hits(text), text)
        lightly_edited = T.SPOS["h0"].replace("carry out", "perform")
        self.assertIn("SPOS:h0", WA.ref_hits(lightly_edited))

    def test_unrelated_wording_passes(self):
        text = "Treat {L} as the sole statuses that move money; every remaining event is inert."
        self.assertEqual(WA.ref_hits(text), [])
        self.assertEqual(WA.vocab_hits(text), [])

    def test_status_words_are_caught(self):
        self.assertEqual(WA.vocab_hits("Closed or {L} events take effect."), ["closed"])
        self.assertIn("posted", WA.status_words())
        self.assertIn("pending", WA.status_words())

    def test_stem_hits_catch_inflections_only(self):
        self.assertEqual(WA.stem_hits("Posting is the default for {L}."), ["posting~posted", "posting~postedx"])
        self.assertEqual(WA.stem_hits("Rejection applies to {L}."), ["rejection~rejected"])
        self.assertEqual(WA.stem_hits("Apply every event except {L}; the operative rest stays."), [])

    def test_clusters_collapse_near_duplicates(self):
        items = [("w001", "Treat {L} as the sole statuses that move money; every remaining event is inert."),
                 ("w002", "Treat {L} as the sole statuses that move money; every other event is inert."),
                 ("w003", "Ignore anything tagged {L}.")]
        self.assertEqual(WA.clusters(items), [["w001", "w002"], ["w003"]])


if __name__ == "__main__":
    unittest.main()
