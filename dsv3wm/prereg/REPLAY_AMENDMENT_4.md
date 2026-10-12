# Replay input binding — after original pilot outcomes

Final review found that v3 checked frozen source identities but did not enforce
membership of every CLI-selected input. All executed original inputs match
their recorded hashes and intended frozen corpus/weights; no numerical result
is changed. Original v1/v2/v3 files and recorded outcomes remain untouched.

Prospective v4 replay additionally requires selected config, development data,
adapter weights, adapter config and passed probe to belong to the freeze.
Replica creation checks training-record hashes against the selected data/config/
weights, expanded typed config, parent identity and expected adapter capacity/
scaling. It binds reused exact-semantics source files as well. This correction
prevents stale/mismatched artifacts being advertised as a frozen replica.

Direct student tasks additionally require canonical parent IDs so duplicate
worlds cannot be hidden behind arbitrary labels. Source/hashing and duplicate
regression tests were added. These are implementation corrections after the
pilot, not positive efficacy observations or a retroactive confirmation study.
