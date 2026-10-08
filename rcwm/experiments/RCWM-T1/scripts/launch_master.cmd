@echo off
rem Relaunch-safe entry point for the RCWM-T1 GPU work (start hidden via WMI; see DECISIONS I-08).
rem Duty cycle is temperature-adaptive in runtime._duty_ratio (DECISIONS I-15); set RCWM_DUTY_RATIO to force a fixed ratio.
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
"C:\Program Files\Git\bin\bash.exe" scripts/master_chain.sh
