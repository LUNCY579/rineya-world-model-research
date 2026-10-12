# Recovery resource wrapper cancellation clarification

Review before the first wrapped native run found that a monitor exception or
KeyboardInterrupt could exit the wrapper without stopping its child. The wrapper
now marks failure/interruption, terminates only its own recorded process identities
(including creation times to avoid PID reuse), waits for exit, and records cleanup.
Its native command, model, optimizer, decoder and metric semantics are unchanged.

Keep the original wrapper bytes in the SHA-named source snapshot. B0 readout V1
and teacher V1 freezes were created but never executed; create new V2 freezes
binding this correction before their native runs. The original training V2 freeze
is unchanged and valid. A0 recovery training started directly through the native
CLI, without this wrapper; its own Windows peak RSS record is available on exit.
No GPU memory measurement is implied by a host-RAM wrapper.
