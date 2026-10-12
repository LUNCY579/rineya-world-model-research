#!/usr/bin/env bash
# RCX-E01R after training (frozen PREREG_E01R, CX D-41/D-42; PREREG section 10 order): smoke the 6 adapters (one lane
# block), open-sealed (the only P3 read; refuses on any smoke failure), then the SEALED evals one lane block per cell
# (N-CFSD s0-s2, E-R2 s0-s2, E-R3 s0-s2, then OFF), then the single analysis if all 9 confirmatory cells completed.
# Every model step needs >= 12.5 GB free RAM (in practice the BWM/RCWM night windows, D-51/D-84). When the SEALED evals
# are done the quiet-window release file is written (BWM D-84).
cd "$(dirname "$0")/../../.." || exit 1
LOG=runs/RCX-E01R/cells/post_train_chain.log
C=runs/RCX-E01R/cells
WINPID=$(cat /proc/$$/winpid)
trap 'python -B -m rcx.lane release --pid '"$WINPID"' >> '"$LOG"' 2>&1' EXIT
TAGS6="E-R2_s0 E-R3_s0 E-R2_s1 E-R3_s1 E-R2_s2 E-R3_s2"
acquire() {
  python -B -m rcx.lane acquire --job "$1" --pid "$WINPID" --expected-min "$2" --need-ram 12.5 >> $LOG 2>&1 \
    || { echo "$(date +%T) lane acquire failed" >> $LOG; exit 5; }
  export RCX_LANE_TOKEN=$(python -B -c "from rcx import lane; print(lane.read_lock()['token'])")
}
release() { python -B -m rcx.lane release --pid "$WINPID" >> $LOG 2>&1; }
echo "$(date +%T) post-train chain start; winpid ${WINPID}" >> $LOG
until grep -q "TRAIN CHAIN DONE" $C/train_chain.log 2>/dev/null; do sleep 120; done
for t in $TAGS6; do
  [ -f "$C/$t/TRAIN_META.json" ] || { echo "$(date +%T) $t has no TRAIN_META.json; stopping (section 10)" >> $LOG; exit 3; }
done
# 1. smoke (50 TRAIN prompts each; never DEV or SEALED)
acquire "RCX E01R smoke x6" 40
for t in $TAGS6; do
  arm=${t%_s*}; seed=${t##*_s}
  [ -f "$C/SMOKE_${t}.json" ] && { echo "$(date +%T) skip smoke $t (done)" >> $LOG; continue; }
  echo "$(date +%T) START smoke $t" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/native_e01r.py smoke --arm "$arm" --seed "$seed" \
    > "$C/smoke_${t}.stdout.txt" 2>&1
  echo "$(date +%T) END smoke $t rc=$?" >> $LOG
done
release
# 2. open-sealed (refuses unless every smoke passed; reads P3 last)
if [ ! -f "$C/SEALED_OPEN.json" ]; then
  echo "$(date +%T) START open-sealed" >> $LOG
  python -B experiments/RCX-E01R/native_e01r.py open-sealed \
    --expect-commit-sha d4635e4292cd0e2ca2026f899d3a1cea75591c3168cb3fb67df766db323a92f2 > "$C/open_sealed.stdout.txt" 2>&1
  rc=$?
  echo "$(date +%T) END open-sealed rc=$rc" >> $LOG
  [ $rc -eq 0 ] || { echo "$(date +%T) open-sealed refused; stopping (see open_sealed.stdout.txt)" >> $LOG; exit 4; }
fi
# 3. SEALED evals, one lane block per cell
for spec in "N-CFSD 0" "N-CFSD 1" "N-CFSD 2" "E-R2 0" "E-R2 1" "E-R2 2" "E-R3 0" "E-R3 1" "E-R3 2" "OFF 0"; do
  set -- $spec
  arm=$1; seed=$2; tag="${arm}_s${seed}"; [ "$arm" = "OFF" ] && tag=OFF
  [ -f "$C/eval_sealed_${tag}.json" ] && { echo "$(date +%T) skip eval $tag (done)" >> $LOG; continue; }
  acquire "RCX E01R sealed eval ${tag}" 90
  echo "$(date +%T) START eval $tag" >> $LOG
  RCX_THREADS=4 python -B experiments/RCX-E01R/native_e01r.py eval --battery sealed --arm "$arm" --seed "$seed" \
    > "$C/eval_sealed_${tag}.stdout.txt" 2>&1
  echo "$(date +%T) END eval $tag rc=$?" >> $LOG
  release
done
echo "{\"party\": \"rcx\", \"reason\": \"E01R SEALED evals done\", \"at\": \"$(date '+%F %T')\"}" \
  > "$TEMP/rineya-quiet-release.rcx.json"
echo "$(date +%T) SEALED EVALS DONE; quiet-window release written" >> $LOG
# 4. the single analysis, only if every confirmatory cell completed (otherwise the lead declares missing cells, section 10)
n=0; for t in N-CFSD_s0 N-CFSD_s1 N-CFSD_s2 $TAGS6; do [ -f "$C/eval_sealed_${t}.json" ] && n=$((n+1)); done
if [ $n -eq 9 ]; then
  echo "$(date +%T) START analysis" >> $LOG
  PYTHONIOENCODING=utf-8 python -B experiments/RCX-E01R/analyze_e01r.py > "$C/analysis.stdout.txt" 2>&1
  echo "$(date +%T) END analysis rc=$?" >> $LOG
else
  echo "$(date +%T) only $n/9 confirmatory evals complete; analysis NOT run (lead decides --missing per section 10)" >> $LOG
fi
echo "$(date +%T) POST-TRAIN CHAIN DONE" >> $LOG
