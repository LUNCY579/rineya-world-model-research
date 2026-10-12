Superseded T2b-ledger data (kit revision 2, built 2026-10-10 20:22). SPENT: the revision-2 dev screen
(runs/t2bl/dev, two RCWM turns, gate GATE.json eef7e401..., DECISIONS R29: STOP (PARTIAL), heldout_posted_twins)
trained T2b-ledger seeds 0-2 on t2bl_train.jsonl (seed 20261161, sha256 89d6a01e...) and evaluated them on
t2bl_dev.jsonl (seed 20261162, sha256 f710b17e...). Under the gate rule's pre-stated revision path this dev bank and
every adapter trained on this training set are spent: never an input of kit revision 3 or of PREREG-5, never re-read
to choose anything. Their outputs (runs/t2bl/dev) stay where the run wrote them.
Replaced at kit revision 3 (2026-10-12): training data only (both top-level key orders; more posted rows under the
stacked training surface) and a fresh dev bank under a new seed; docs/t2b/ledger/DESIGN.md section 10.
Moved here unread, never deleted. Since kit revision 3 the freshness union of the kit build includes this directory
(every data/t2bl_superseded_rev*/ file), so revision 3's training set and dev bank share no payload with these files.
docs/t2b/ledger/superseded_rev2/ holds the matching kit records, the rendered PREREG-5 draft and a copy of the
revision-2 kit code (code/).
