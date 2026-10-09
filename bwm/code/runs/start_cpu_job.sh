#!/bin/bash
# One-off CPU job outside the session process tree (WMI -> bash), 8 threads, no CUDA. Usage: start_cpu_job.sh <log> <args...>
LOG=$1; shift
CMD="cd /c/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1 && BWM_DEVICE=cpu BWM_DUTY_RATIO=0 BWM_YIELD_S=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python $* > $LOG 2>&1"
powershell -Command "\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='\"C:\Program Files\Git\bin\bash.exe\" -c \"$CMD\"'}; 'cpu job bash pid ' + \$r.ProcessId"
