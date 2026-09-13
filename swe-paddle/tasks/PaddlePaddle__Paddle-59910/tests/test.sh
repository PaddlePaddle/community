#!/usr/bin/env bash
set -euo pipefail
python -m pytest test/swe_paddle/test_pr59910_normal_pir.py -q
