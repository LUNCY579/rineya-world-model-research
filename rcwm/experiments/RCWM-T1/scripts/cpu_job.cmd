@echo off
rem Detached CPU analysis job. Usage: set LOGN=<logname>&& cpu_job.cmd <python script> <args...>
rem %* keeps comma-separated values intact (positional %1..%9 would split them).
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
set PYTHONIOENCODING=utf-8
"C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -u %* > runs\bank_pilot\analysis\%LOGN%.log 2>&1
echo exit %errorlevel% >> runs\bank_pilot\analysis\%LOGN%.log
