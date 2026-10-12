# SOURCE11 conservative-v2 independent transaction assessment

Status: 27 focused tests PASS in 0.167 seconds. Final approval still requires the frozen complete 503-test regressions, commitments/evidence, and the amended operational comparison.

The original critical finding and withheld approval remain in /tmp/gcdl-source11-independent-fault-review.md. No original candidate promotion or blanket failure-equivalence claim is authorized here.

V2 preserves the durable outcome for a preexisting corrupt done receipt: capture previous done pointers before discovery; discovered validation still occurs once at its original upsert point; a validator BaseException for a previous done pointer rolls back recovery-prefix SQL, then reraises. The old setup commit preserves earlier caller work. Our paired original/candidate test confirms worker.event can no longer accidentally commit pending A as done after a late failure for existing B; both now retain the original durable queue. New discovered errors still leave the original authenticated prefix uncommitted and caller-committable; upsert exceptions are outside the validation rollback handler and retain old behavior. Interrupts, deadline exceptions and prior unrelated work are explicitly covered.

This is intentionally stricter for a late concurrent change to a previously done receipt: rollback can remove authenticated prefix progress that the old second pass might have committed. The immutable preregistration amendment explicitly records this conservative deviation. It is not blanket bit-for-bit fault equivalence.

No false custody, unvalidated done row, raw body access, sealed read, official H/v2 execution, remote API or production mutation occurs in these fixtures. They use the actual receipt validator, real SQLite and worker.event over hand-specified attested label-free metadata. Existing authoritative scientific oracle regressions remain in the original 476-test suite.
