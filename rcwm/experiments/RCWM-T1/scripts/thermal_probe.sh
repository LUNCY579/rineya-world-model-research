#!/usr/bin/env bash
cd "$(dirname "$0")/.."
C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe -u scripts/thermal_probe.py > runs/bank_pilot/thermal_probe.txt 2>&1
