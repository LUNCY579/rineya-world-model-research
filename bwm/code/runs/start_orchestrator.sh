#!/bin/bash
# Launch the orchestrator outside the Claude session's process tree: WMI -> bash.exe -> python.
# (Direct WMI python + new-process-group children died with STATUS_CONTROL_C_EXIT; WMI->bash survives.)
D='C:\Users\clclc\BusinessProject\WorldModel\Rineya_BWM_001_Research_Bundle\native_v1'
powershell -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='\"C:\Program Files\Git\bin\bash.exe\" -c \"cd /c/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1 && python scripts/orchestrate.py\"'; CurrentDirectory='$D'}; 'orchestrator bash pid ' + \$r.ProcessId"
