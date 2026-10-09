# Shared helpers for queue scripts. Source from native_v1.
# train_job <label> <outdir> <args...>: run under lease unless <outdir>/adapter exists; on failure preserve and retry once.
train_job() {
  local label=$1 out=$2; shift 2
  for attempt in 1 2; do
    [ -f "$out/adapter/adapter_model.safetensors" ] && return 0
    python scripts/gpu_run.py "$label" -- "$@" --out "$out" > "$out.stdout" 2>&1
    [ -f "$out/adapter/adapter_model.safetensors" ] && return 0
    mkdir -p runs/_failed; local t=$(date +%H%M%S)
    [ -d "$out" ] && mv "$out" "runs/_failed/$(basename $out)_fail$t"
    mv "$out.stdout" "runs/_failed/$(basename $out)_fail$t.stdout" 2>/dev/null
    echo "job $label failed (attempt $attempt)"; sleep 60
  done
  return 1
}
