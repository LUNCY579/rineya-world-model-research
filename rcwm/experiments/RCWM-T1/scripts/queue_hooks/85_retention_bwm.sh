# Queue hook (D-46): 5c retention confirmation for BWM's two release adapters (BWM PX-04), run in an RCWM turn with the
# frozen suite (docs/retention/RET_FREEZE.json). Sourced by master_chain.sh in name order; idempotent.
# - The confirmation draw is minted ONCE, at run time (fresh OS-entropy seed, sealed); its id is recorded in
#   docs/retention/BWM_RELEASE_DRAW.txt and never re-minted, so no draw can be chosen after seeing outputs.
# - BWM's adapter files must match the sha256 pins BWM sent (2026-10-10); any mismatch skips the run.
# - ret_run.py exit codes: 0 complete; 3 deterministic stop (never retried); 5 deferred by the turn cap: sleep 60 s
#   (so BWM's yield loop sees the release; agreed with BWM) and call once more.
RET_BWM_MARK=docs/retention/BWM_RELEASE_DRAW.txt
RET_BWF=C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_Fdelta_s0/adapter
RET_BWPA=C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_Pa_v2_s1/adapter
ret_bwm_pins_ok() {
  [ "$(sha256sum "$RET_BWF/adapter_model.safetensors" | cut -c1-64)" = "2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae" ] &&
  [ "$(sha256sum "$RET_BWF/adapter_config.json" | cut -c1-64)" = "3038c77961f33461e1bef736959765434c0be473d991b0ef92c4813d269b3b8c" ] &&
  [ "$(sha256sum "$RET_BWPA/adapter_model.safetensors" | cut -c1-64)" = "ee1c5aec5e4a55e30f6df0e30a3cd68242c928cd158f9e5cf06184f409108110" ] &&
  [ "$(sha256sum "$RET_BWPA/adapter_config.json" | cut -c1-64)" = "f6a2c4efb381678ac44bebef1165d84d587a81e72c8a8ea1c27062f2bc7494bf" ]
}
if [ -f docs/retention/RET_FREEZE.json ] && [ -f docs/retention/DEV_SELECTION.json ]; then
  if ! ret_bwm_pins_ok; then
    echo "=== $(date -Iseconds) 85_retention_bwm: BWM adapter pins differ; run skipped" >> "$LOG"
  else
    if [ ! -f "$RET_BWM_MARK" ]; then                       # CPU, once: mint the sealed confirmation draw
      ret_before=$(ls docs/retention/draws/confirm_*.jsonl 2>/dev/null | sort)
      step scripts/ret_gen.py mint --purpose confirm
      ret_after=$(ls docs/retention/draws/confirm_*.jsonl 2>/dev/null | sort)
      ret_new=$(comm -13 <(echo "$ret_before") <(echo "$ret_after"))
      [ "$(echo "$ret_new" | grep -c . )" -eq 1 ] && echo "$ret_new" > "$RET_BWM_MARK"
    fi
    if [ -f "$RET_BWM_MARK" ]; then
      ret_draw=$(cat "$RET_BWM_MARK"); ret_id=$(basename "$ret_draw" .jsonl)
      if [ ! -f "runs/retention/$ret_id/DONE.json" ]; then
        step scripts/ret_run.py --mode confirm --draw "$ret_draw" --extra-adapter "bwf=$RET_BWF" --extra-adapter "bwpa=$RET_BWPA"
        ret_rc=$?
        if [ "$ret_rc" -eq 5 ]; then
          sleep 60
          step scripts/ret_run.py --mode confirm --draw "$ret_draw" --extra-adapter "bwf=$RET_BWF" --extra-adapter "bwpa=$RET_BWPA"
        fi
      fi
      if [ -f "runs/retention/$ret_id/DONE.json" ] && [ ! -f "runs/retention/$ret_id/analysis/RET_RESULTS.json" ]; then
        step scripts/ret_analysis.py analyze --run "runs/retention/$ret_id"
      fi
    fi
  fi
fi
