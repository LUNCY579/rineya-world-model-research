# Generator improvements

RCM-FIX-001 removes inflated difficulty labels and records the actual exposed vocabulary/magnitude shifts. Initial pilot-v1 remains unchanged; pilot-v2 is fresh pending review.

RCM-FIX-002 crosses JSON key order with answer-only and trace format independently. Both orders exist within each target format. The 30 semantic parents remain 30; eight renderings per parent do not add independent concepts.

Repository backup formatting was corrected after git diff --check rejected extra blank EOF lines. The failed attempt and exact recovery commit are retained in backup_logs. Foundation commit 517a51a60a00e6b7a625edeeb8e6bc126393856c is remotely byte-verified for its 36-file inventory.
