@echo off
rem Detached launcher for the RCWM-T1 pilot queue (started hidden via WMI so it outlives the agent
rem session and no console window appears on the desktop). Yields to the peer's P-actions rerun first
rem (deadline 2026-10-06 11:45 CDT = epoch 1791305100).
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
echo launcher start %date% %time% > runs\bank_pilot\launcher_out.txt
"C:\Program Files\Git\bin\bash.exe" scripts/wait_then_queue.sh P-actions 1791305100 >> runs\bank_pilot\launcher_out.txt 2>&1
echo launcher end %date% %time% errorlevel=%errorlevel% >> runs\bank_pilot\launcher_out.txt
