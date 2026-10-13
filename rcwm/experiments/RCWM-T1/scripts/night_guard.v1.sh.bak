#!/usr/bin/env bash
# RCWM night guard (D-51): honour RCX's operator-approved GPU-idle windows (22:50-07:00) until END_EPOCH.
# - In the window: never let RCWM START a new GPU turn. If no rcwm-* lease is held, stop the RCWM master chain
#   (it is only waiting); if an rcwm-* lease is held, let that turn finish, then stop the chain right after release.
# - At/after 07:00: relaunch the chain once if the guard stopped it (or if no chain is alive).
# Long-lived; bash builtins for time and the lock read; process queries filter on Name == bash.exe AND the exact
# script path (never a substring the querying command itself contains; DECISIONS I-36). Log: runs/night_guard_log.txt
END_EPOCH=${END_EPOCH:-0}
LOCK="$TEMP/rineya-gpu-0.lock"
LOG="runs/night_guard_log.txt"
BASE='C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1'
log() { printf '%(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*" >> "$LOG"; }
stop_chain() {
  powershell -NoProfile -Command "\$c = Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*scripts/master_chain.sh*' }; foreach (\$p in \$c) { taskkill /F /T /PID \$p.ProcessId | Out-Null }; 'stopped ' + @(\$c).Count" >> "$LOG" 2>&1
}
start_chain() {
  powershell -NoProfile -Command "\$c = Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*scripts/master_chain.sh*' }; if (@(\$c).Count -gt 0) { 'chain alive, not relaunching' } else { \$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c \"$BASE\scripts\launch_master.cmd\"'; CurrentDirectory='$BASE'; ProcessStartupInformation=\$su}; 'relaunched rc=' + \$r.ReturnValue }" >> "$LOG" 2>&1
}
log "night guard start (end epoch $END_EPOCH)"
stopped=0; last_rcwm=0
while :; do
  printf -v now '%(%s)T' -1; printf -v hm '%(%H%M)T' -1; hm=$((10#$hm))
  [ "$END_EPOCH" -gt 0 ] && [ "$now" -ge "$END_EPOCH" ] && { [ "$stopped" -eq 1 ] && { log "end reached; relaunching"; start_chain; }; log "night guard end"; exit 0; }
  rcwm_held=0
  if [ -f "$LOCK" ]; then rec=$(<"$LOCK"); [[ "$rec" == *'"label": "rcwm-'* ]] && rcwm_held=1; fi
  # NOSTART_HM (default 1820): from then until 22:50 no NEW RCWM turn may start (a turn is capped at 4.5 h, so any
  # running turn ends by about 22:50; DECISIONS I-37). A running RCWM turn is never interrupted.
  if [ "$hm" -ge "${NOSTART_HM:-1820}" ] || [ "$hm" -lt 700 ]; then
    if [ "$rcwm_held" -eq 1 ]; then
      [ "$last_rcwm" -eq 0 ] && log "window: an RCWM turn is running; letting it finish"
      last_rcwm=1
    elif [ "$stopped" -eq 0 ] || [ "$last_rcwm" -eq 1 ]; then
      log "window: no RCWM lease held; stopping the RCWM chain"; stop_chain; stopped=1; last_rcwm=0
    fi
    sleep 15
  else
    if [ "$stopped" -eq 1 ]; then log "window over; relaunching the RCWM chain"; start_chain; stopped=0; fi
    last_rcwm=0
    sleep 60
  fi
done
