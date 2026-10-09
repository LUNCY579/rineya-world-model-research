#!/bin/bash
# Wait until the old CPU orchestrator exits (after its current job), then start the new one with dependency support.
cd /c/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1
while powershell -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*orchestrate.py --cpu*' }) { exit 0 } else { exit 1 }"; do sleep 60; done
rm -f runs/.stop_cpu
python scripts/orchestrate.py --cpu
