#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/python3 /root/.hermes/profiles/third/scripts/fault_scan_npu_entry.py inference
