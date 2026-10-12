Superseded T2b-ledger kit records (kit revision 2; the dev gate rule revision 2 frozen 2026-10-10 20:52, D-48).
SPENT by the revision-2 dev screen (runs/t2bl/dev; DECISIONS R29: STOP (PARTIAL), heldout_posted_twins).
Moved here at kit revision 3 (2026-10-12), never deleted:
  GATE_RULE.json (sha 06ec07dd..., the revision-2 gate freeze: it pinned the revision-2 data, code, hook draft and
    PREREG-5 draft a06dc9a8...), GATE_RULE.md, AUDIT.json, GATE_DRYRUN.json, SCORER_SELFTEST.json, GEN_PLAN.json,
    TEST_*.json, P5_* records, DESIGN.md (revision 2), PREREG_5_DRAFT.template.md, PREREG_5_DRAFT.md (the rendered
    revision-2 draft, sha a06dc9a8..., moved from the experiment root), both hook drafts (queue_hook_t2bl.sh.draft
    sha e389df78... is byte-identical to the installed scripts/queue_hooks/88_t2bl.sh of revision 2).
  code/: a copy of scripts/t2bl_*.py as they were when revision 2 was superseded (the code the revision-2 run
    recorded; kept for the record; not importable from here).
The gate rule text and its thresholds are unchanged in revision 3 (t2bl_gate.RULE_TEXT is byte-identical).
