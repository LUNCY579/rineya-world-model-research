#!/usr/bin/env bash
# RCWM night guard v2 (D-51, I-37, review note 2026-10-13): honour RCX's operator-approved GPU-idle windows.
# From NOSTART_HM (default 18:20; 4.5 h turn cap before 22:50) until 07:00, no NEW RCWM GPU turn may start:
# - a running RCWM turn (an rcwm-* lease) is never interrupted;
# - an RCWM CPU-only step (no lease, no live waiter) is never interrupted;
# - a NEW live RCWM waiter (waiting.rcwm.json with a newer 'since' and a live pid) -> the chain is stopped.
# After 07:00 the chain is relaunched if the guard stopped it. Ends at END_EPOCH. Log: runs/night_guard_log.txt.
# Bash builtins for time and file reads; a process query runs only when a new waiter marker appears (I-27, I-36).
END_EPOCH=${END_EPOCH:-0}
LOCK="$TEMP/rineya-gpu-0.lock"; WAIT="$TEMP/rineya-gpu-0.waiting.rcwm.json"
LOG="runs/night_guard_log.txt"
BASE='C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1'
log() { printf '%(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*" >> "$LOG"; }
pid_alive() { powershell -NoProfile -Command "if (Get-Process -Id $1 -ErrorAction SilentlyContinue) { 'alive' } else { 'dead' }" 2>/dev/null; }
stop_chain() {
  powershell -NoProfile -Command "\$c = Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*scripts/master_chain.sh*' }; foreach (\$p in \$c) { taskkill /F /T /PID \$p.ProcessId | Out-Null }; 'stopped ' + @(\$c).Count" >> "$LOG" 2>&1
}
start_chain() {
  powershell -NoProfile -Command "\$c = Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*scripts/master_chain.sh*' }; if (@(\$c).Count -gt 0) { 'chain alive, not relaunching' } else { \$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c \"$BASE\scripts\launch_master.cmd\"'; CurrentDirectory='$BASE'; ProcessStartupInformation=\$su}; 'relaunched rc=' + \$r.ReturnValue }" >> "$LOG" 2>&1
}
log "night guard v2 start (end epoch $END_EPOCH, no-start from ${NOSTART_HM:-1820})"
stopped=0; handled_since=""; in_window=0
while :; do
  printf -v now '%(%s)T' -1; printf -v hm '%(%H%M)T' -1; hm=$((10#$hm))
  if [ "$END_EPOCH" -gt 0 ] && [ "$now" -ge "$END_EPOCH" ]; then
    [ "$stopped" -eq 1 ] && { log "end reached; relaunching"; start_chain; }
    log "night guard end"; exit 0
  fi
  if [ "$hm" -ge "${NOSTART_HM:-1820}" ] || [ "$hm" -lt 700 ]; then
    [ "$in_window" -eq 0 ] && { log "no-start period begins"; in_window=1; handled_since=""; }
    rcwm_held=0
    if [ -f "$LOCK" ]; then rec=$(<"$LOCK"); [[ "$rec" == *'"label": "rcwm-'* ]] && rcwm_held=1; fi
    if [ "$rcwm_held" -eq 0 ] && [ -f "$WAIT" ]; then
      w=$(<"$WAIT"); since=${w#*\"since\": }; since=${since%%[,\}]*}; wpid=${w#*\"pid\": }; wpid=${wpid%%[,\}]*}
      if [ -n "$since" ] && [ "$since" != "$handled_since" ]; then
        handled_since="$since"
        if [ "$(pid_alive "$wpid")" = "alive" ]; then
          log "no-start period: a live RCWM waiter (pid $wpid) would start a GPU turn; stopping the RCWM chain"
          stop_chain; stopped=1
        fi
      fi
    fi
    sleep 15
  else
    if [ "$in_window" -eq 1 ]; then log "no-start period over"; in_window=0; fi
    if [ "$stopped" -eq 1 ]; then log "relaunching the RCWM chain"; start_chain; stopped=0; fi
    sleep 60
  fi
done
