@echo off
rem Generation-only rerun for the sparse arm (capture already complete; see runs/bank_pilot/sparse/CAPTURED.json).
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
"C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -u scripts\run_arm.py --bank data/bank_pilot.json --arm sparse --gen > runs\bank_pilot\sparse_gen_log.txt 2>&1
echo exit %errorlevel% >> runs\bank_pilot\sparse_gen_log.txt
