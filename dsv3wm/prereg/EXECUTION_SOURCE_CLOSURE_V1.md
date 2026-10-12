# Shared execution dependency pins

Before the next native outputs, extend the unchanged H3 teacher, Ledger B0 and
Block G1 freezes into new files. Include their original freeze bytes and the
first-party modules imported by the shared CPU helper: both package initializers,
`blocks.py`, `flops.py` and `planbench_extract.py`. The latter's bank readers are
not invoked; importing its pure definitions is required by the existing helper.
No historical bank contents are opened.

`freeze_execution_dependencies.py` validates every original pin, preserves all
original metadata/member digests and adds the helper, this registration and the
dependency files. Data, model, numerical precision, generation budgets, scoring
and scientific gate thresholds are unchanged. Execute with the new extended
freeze so all these members are checked before model loading. Existing freezes,
failed attempts and published results remain unchanged. A later training freeze
must retain the actual qualified baseline's execution-freeze reference.

The separate R/P retention V2 event-surface correction retains these dependency
pins in its own new freezer. Its data change is not part of this amendment.
