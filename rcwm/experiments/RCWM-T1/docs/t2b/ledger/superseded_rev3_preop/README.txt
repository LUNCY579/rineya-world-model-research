Superseded T2b-ledger kit revision 3, FIRST BUILD (2026-10-12 11:31-11:54; generator t2bl-data-v3). NOT SPENT: never
frozen, never trained on, never evaluated, no GPU, no model output. Replaced on 2026-10-12 (16:33-) by the operator
decisions on kit revision 3 (docs/t2b/ledger/DESIGN.md section 10b), before any revision-3 gate freeze. Moved / copied
here, never deleted:
  t2bl_train.jsonl (seed 20261165, sha256 43646b0d05082a91a4c88d87f8982835e8a17ef6e3ef889b0704cbbe5c1f65f3) and
    t2bl_train.SOURCE.json (8585569e...): moved from data/. Its stacked-surface rows were 300 / 656 = 0.457 posted and
    its reversed-top rows 360 / 788 = 0.457 posted (bounds then [0.40, 0.60]): a residual surface-to-status cue, which
    DESIGN.md, AUDIT.json and the code comments of this build wrongly described as 'no surface-to-status cue'.
  t2bl_dev.jsonl (seed 20261166, sha256 731bbc6a041799ec43b832ae9540cee8db2324fd48ab2dbacc420f37831c5ba8) and
    t2bl_dev.SOURCE.json (d8ae1c49...): moved from data/ for the create-only rebuild. The rebuilt data/t2bl_dev.jsonl is
    byte-identical (same seed, builder and union); its new SOURCE differs only in gen_version, the build-code hashes,
    built_local and the 'superseded_unused' note.
  AUDIT.json, DESIGN.md, GATE_DRYRUN.json, GATE_FREEZE_READY.json (sha d6c2f4b2...), GATE_RULE.md, GEN_PLAN.json,
    P5_*.json / .md, PREREG_5_DRAFT.template.md, PREREG_5_DRAFT.md (the rendered first-build draft, sha 10e09a25...),
    SCORER_SELFTEST.json, TEST_*.json, both hook drafts (queue_hook_t2bl.sh.draft sha b73a05f8...): copies of the
    first build's records, rewritten in place by the operator-decision rebuild.
  code/: a copy of scripts/t2bl_*.py as they were for the first build (kept for the record; not importable from here).
Why here and not under data/: every file under data/ except this kit's own top-level data/t2bl_* files is in the
freshness union, so a copy of this dev bank there would collide with the current (identical) dev bank, and the
regenerated training set (same seed, same streams) would be forced off every slot it shares with this build. These
files were never trained on, so they are not spent and need not be excluded from anything; docs/t2b/ledger/ is outside
every union (t2bl_make_data.DOCS_SKIP), PREREG-5's included.
