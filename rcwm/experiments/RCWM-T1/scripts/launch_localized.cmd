@echo off
rem D-23 localised-delta interference test. Waits (no torch, no lease) until T2 s0's eval.json exists, then
rem runs under the shared GPU lease. Analysis afterwards: scripts\localized_analysis.py (CPU).
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
set PYTHONIOENCODING=utf-8
"C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -u scripts\run_localized.py --device cuda --out-name localized --wait-for runs\t2\t2_s0\eval.json --wait-max-h 14 >> runs\bank_pilot\localized_log.txt 2>&1
echo exit %errorlevel% >> runs\bank_pilot\localized_log.txt
