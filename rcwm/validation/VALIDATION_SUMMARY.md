# Validation summary

Final CPU unit tests: **60 passed** (see pytest logs/XML). Resource check: **PASS** for 3 schemas,
4 JSONL trajectory records, 7 YAML configurations and 43 source entries. Synthetic demo: executed
250 updates on a 40,325-parameter CPU model. Native Rineya/GPU work: **not executed**.

Model limitation: the demo underpredicts a safe prerequisite-chain success. It is NOT a qualified
world model, even though average MSE decreases. Read KNOWN_LIMITATIONS and the report's validation
chapter for the exact observed values. This distinction is intentional and scientifically important.

The final 31-page PDF was rendered and every page inspected in contact sheets. Text geometry was
checked for out-of-page bounds and replacement glyphs. The source register, table of contents and
main mathematics are present. Intermediate build files and font files are not included.

Archive file integrity is recorded in MANIFEST.sha256 and can be rechecked after extraction using
tools/verify_manifest.py. Passing integrity or mechanical tests is not a certification of scientific,
security or production quality.
