#!/usr/bin/env bash
# Deferred (D-17): the non-binding pilot regeneration now runs at the END of master_chain.sh
# (via scripts/regen_chain_body.sh) so that RCWM-T2 and the mechanism jobs go first.
echo "=== $(date -Iseconds) regen deferred to end of master chain" >> "$(dirname "$0")/../runs/regen_log.txt"
