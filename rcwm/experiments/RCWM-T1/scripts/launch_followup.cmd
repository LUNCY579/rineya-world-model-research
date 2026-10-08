@echo off
cd /d "C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1"
"C:\Program Files\Git\bin\bash.exe" scripts/followup_retro.sh >> runs\bank_pilot\followup_launcher.txt 2>&1
echo launcher end %date% %time% >> runs\bank_pilot\followup_launcher.txt
