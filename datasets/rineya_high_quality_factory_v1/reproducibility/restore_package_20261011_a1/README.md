# Recover the Rineya dataset factory snapshot

The immutable data snapshot contains 12,889 files and 5,083,117,507 original
bytes. Its complete private GitHub archive and every current original file
passed separate full-byte checks. All 473 later-born recovery and verification
files were saved and verified in metadata commit
46357052fd40b265dd65916069563b4ed0cdc4ed.

The portable restorer passed separate source review and 25 actual synthetic
tests on Python 3.12.6. An actual full standalone restore remains unrun.
Historical SOURCE_ONLY/UNEXECUTED annotations in the frozen code describe its
preparation; actual tests and reviews are recorded separately.

Download restore.py and SNAPSHOT_DESCRIPTOR.json from this directory at the
GitHub commit you choose, and check their exact SHA256/size bindings:

* restore.py: 42,937 bytes; SHA256
  a0d96e69d5836f16cf8e738935aa6214f40238a8febbcdccc559094f9fe2bdad
* SNAPSHOT_DESCRIPTOR.json: 8,857,032 bytes; SHA256
  238c0b9d992e4721f0776d57cc14f7d4974675c15681687decc68b658aed3034

Use Python 3.10 or later, GitHub CLI gh authenticated with read access to the
private repository LUNCY579/rineya-world-model-research, and a Unix filesystem
that preserves permissions. Keep downloaded tools outside the destination.
Destination and report must both be fresh absolute paths, with existing
canonical directory parents:

~~~sh
python3 -B /restore-tools/restore.py \
  --descriptor /restore-tools/SNAPSHOT_DESCRIPTOR.json \
  --descriptor-sha256 238c0b9d992e4721f0776d57cc14f7d4974675c15681687decc68b658aed3034 \
  --final-commit cc3b90a95b34d77bcbade4fe10e9cd3931703bb5 \
  --destination /restore-work/High-Quality_Dataset \
  --report /restore-work/RESTORE_REPORT.json
~~~

The final data commit is exactly cc3b90a95b34d77bcbade4fe10e9cd3931703bb5;
tree 78a1f1e6f2aed5f33ab7add9404f23649d1932ee. Use that full data commit
for restoration. Later metadata and restore-package commits contain recovery
evidence and are not inputs to the fixed eighteen-part data chain.

Measured minimum initial free space is 5,748,890,056 bytes: original bytes,
all 128,901,637 compressed part bytes, and a 512 MiB reserve. Fresh capacity is
checked before every write. Individual parts are bounded; the complete
repository archive is streamed.

The program verifies one authenticated immutable GitHub archive's complete
11,025-file Git tree, full Git blob bytes, modes, outer gzip CRC/EOF and native
gh exit. It restores unchanged base files and all 18 compact parts, reconstructs
210 indexed raw files, and checks all 211 representations, 169 origins and two
immutable historical bindings. The final pass rereads the exact 12,889-file
namespace, SHA256, Git blob, size and original permissions: 11,706 files at 0644,
1,068 at 0444 and 115 at 0600. Replacement device/inode/timestamps may differ.

Success requires native exit0 and a PASS restoration report. Failure preserves
the partial destination, unmatched bounded spools and a separate report where
possible. Existing destinations/reports are never overwritten. Investigate
failures and choose new fresh paths for another attempt.

RESTORE_LOCATOR.json identifies data/metadata commits, all 18 prefixes, original
proof bindings and actual test status. ROOT_METADATA_REMOTE_COMPLETION.json
is the actual completion born after the metadata checkpoint.
METADATA_EXECUTION_EVIDENCE.json.gz preserves later-born metadata sources,
grants, native logs and controls. Those files are outside the captured data
namespace and their own earlier metadata input cut. This package's own future
commit and remote verification are recorded by a later ROOT completion.

This backup preserves dataset and reproducibility bytes. It does not itself
approve scientific results, new CODE/DATA votes, publication, training or
SOURCE06 stage success. The earlier SOURCE06 capacity failure is preserved.
