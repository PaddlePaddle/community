#!/usr/bin/env bash
set -euo pipefail

python -m pytest test/swe_paddle/test_pr55890_vpp_overlap_schedule.py -q
